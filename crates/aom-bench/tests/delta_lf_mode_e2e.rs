//! `--delta-lf-mode=1` (`AV1E_SET_DELTALF_MODE`, family C5) end-to-end
//! byte-match gate: the port's own per-SB delta-lf decision vs real aomenc
//! `--delta-lf-mode=1 --deltaq-mode=2`. Delta-lf rides on a firing delta-q mode
//! (here the just-landed wavelet mode 2); each SB's `delta_lf_from_base` is
//! derived from its `delta_qindex` (`setup_delta_q`, encodeframe.c:380-383) and
//! coded alongside the delta-qindex. The frame filter_level is delta-lf-
//! independent (picklpf.c never reads delta_lf), so the port's LF derivation is
//! unchanged; only the header flag + the per-SB delta-lf symbols are new.
//!
//! Scope: bd8 4:2:0, dims a multiple of the 64px SB (the mode-2 scope).

use aom_bench::{EncodeCell, ToggleKnobs};

const AV1E_SET_DELTAQ_MODE: i32 = 107;
const AV1E_SET_DELTALF_MODE: i32 = 108;

fn dlf_knobs() -> ToggleKnobs {
    ToggleKnobs {
        deltaq_mode2: true,
        delta_lf_mode: true,
        ..Default::default()
    }
}

fn cell(w: usize, h: usize, cq: i32) -> EncodeCell {
    EncodeCell::real_content(
        &format!("dlf_{w}x{h}_cq{cq}"),
        "av1-1-b8-01-size-196x196",
        Some((w, h, 0, 0)),
        cq,
        0,
    )
}

fn ctrls() -> Vec<(i32, i32)> {
    vec![(AV1E_SET_DELTAQ_MODE, 2), (AV1E_SET_DELTALF_MODE, 1)]
}

fn run_cell(cell: &EncodeCell) -> Result<usize, String> {
    let c_stream = cell.c_encode_ctrls(&ctrls());
    let real = EncodeCell::frame_obu_payload(&c_stream);
    let ours = cell.port_encode_with(&c_stream, &dlf_knobs());
    if ours == real {
        return Ok(real.len());
    }
    let first = ours
        .iter()
        .zip(real.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(real.len()));
    Err(format!(
        "first diff at frame-OBU byte {first}; port {} B vs real {} B",
        ours.len(),
        real.len()
    ))
}

/// The hard byte-match gate: the port's `--delta-lf-mode=1 --deltaq-mode=2`
/// encode is byte-identical to real aomenc across the web quality range +
/// non-square shapes. Any divergence is a regression.
#[test]
fn delta_lf_mode_e2e() {
    let mut cells: Vec<EncodeCell> =
        [12, 20, 32, 48, 63].into_iter().map(|cq| cell(192, 192, cq)).collect();
    cells.push(cell(192, 128, 32));
    cells.push(cell(128, 192, 32));

    let mut matched = 0usize;
    let mut report = String::new();
    for cell in &cells {
        match run_cell(cell) {
            Ok(len) => {
                matched += 1;
                report.push_str(&format!("  MATCH    {} ({len} B)\n", cell.label));
            }
            Err(why) => report.push_str(&format!("  MISMATCH {}: {why}\n", cell.label)),
        }
    }
    eprintln!(
        "--delta-lf-mode=1 (+ --deltaq-mode=2) e2e byte-match: {matched}/{}\n{report}",
        cells.len()
    );
    assert_eq!(
        matched,
        cells.len(),
        "not all --delta-lf-mode=1 cells byte-match real aomenc:\n{report}"
    );
}

/// Anti-vacuous witness: the delta-lf decision must do real work. The
/// `--delta-lf-mode=1` reference must differ from the mode-2-only stream (the
/// delta-lf symbols add bytes), and the port WITHOUT the delta-lf arm must
/// DIVERGE from the `--delta-lf-mode=1` reference.
#[test]
fn delta_lf_mode_knob_bites() {
    let cell = cell(192, 192, 12);
    let c_stream = cell.c_encode_ctrls(&ctrls());
    let real = EncodeCell::frame_obu_payload(&c_stream);
    // The delta-lf stream must differ from mode-2-only (proves delta-lf fired).
    let mode2_only = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, 2)]));
    assert_ne!(
        real, mode2_only,
        "delta-lf must add symbols vs mode-2-only for the witness to be meaningful"
    );
    // Port with the delta-lf arm OFF (mode-2 only) diverges from the delta-lf reference...
    let without = cell.port_encode_with(
        &c_stream,
        &ToggleKnobs { deltaq_mode2: true, ..Default::default() },
    );
    assert_ne!(
        without, real,
        "port without the delta-lf arm must NOT match the --delta-lf-mode=1 stream"
    );
    // ...and ON it matches.
    let with = cell.port_encode_with(&c_stream, &dlf_knobs());
    assert_eq!(
        with, real,
        "port with the delta-lf arm must match the --delta-lf-mode=1 stream"
    );
}
