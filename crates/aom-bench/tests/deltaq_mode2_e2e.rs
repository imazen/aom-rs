//! `--deltaq-mode=2` (`DELTA_Q_PERCEPTUAL`, wavelet AC energy — family C5)
//! end-to-end byte-match gate: the port's own per-SB wavelet-energy delta-q
//! encode vs real aomenc `--deltaq-mode=2` on REAL image content. The per-SB
//! qindex is derived by the port (`av1_block_wavelet_energy_level` +
//! `av1_compute_q_from_energy_level_deltaq_mode`, the 5/3 dwt Haar-AC energy
//! → the rate-ratio segment qindex); only the frame-header fields are
//! bootstrapped, and the delta_q ones are cross-checked, never copied (see
//! `port_encode_impl`'s mode-2 arm).
//!
//! Scope: bd8 4:2:0, dims a multiple of the 64px SB (mode-2 fires; every SB
//! reads a full in-frame 64x64). The dwt kernel itself is pinned bit-exact vs
//! the exported C in `deltaq_perceptual_wavelet_diff`. Highbd + partial-SB are
//! follow-ups (the same scope note as mode 3).

use aom_bench::{EncodeCell, ToggleKnobs};

const AV1E_SET_DELTAQ_MODE: i32 = 107;

fn mode2_knobs() -> ToggleKnobs {
    ToggleKnobs {
        deltaq_mode2: true,
        ..Default::default()
    }
}

/// Real-content cell at `(w, h)` cropped from the 196x196 conformance vector.
fn cell(w: usize, h: usize, cq: i32) -> EncodeCell {
    EncodeCell::real_content(
        &format!("deltaq2_{w}x{h}_cq{cq}"),
        "av1-1-b8-01-size-196x196",
        Some((w, h, 0, 0)),
        cq,
        0,
    )
}

/// Per-cell match check. `port_encode_with` returns the assembled frame OBU
/// PAYLOAD (compare directly to the reference payload). `Ok(len)` on
/// byte-match, else the first differing byte.
fn run_cell(cell: &EncodeCell) -> Result<usize, String> {
    let c_stream = cell.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, 2)]);
    let real = EncodeCell::frame_obu_payload(&c_stream);
    let ours = cell.port_encode_with(&c_stream, &mode2_knobs());
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

/// The hard byte-match gate: every cell's port `--deltaq-mode=2` encode is
/// byte-identical to real aomenc across the web quality range + non-square
/// shapes. Any divergence is a regression.
#[test]
fn deltaq_mode2_perceptual_wavelet_e2e() {
    let mut cells: Vec<EncodeCell> =
        [12, 20, 32, 48, 63].into_iter().map(|cq| cell(192, 192, cq)).collect();
    // Non-square shapes exercise the running-base delta chain across a
    // different SB raster.
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
        "--deltaq-mode=2 (PERCEPTUAL wavelet) e2e byte-match: {matched}/{}\n{report}",
        cells.len()
    );
    assert_eq!(
        matched,
        cells.len(),
        "not all --deltaq-mode=2 cells byte-match real aomenc:\n{report}"
    );
}

/// Anti-vacuous witness: the wavelet delta-q machinery must do real work. On a
/// cell where mode-2 modulates the qindex (the reference differs from a plain
/// encode), the port WITHOUT the mode-2 arm (`deltaq_mode2 = false`) must
/// DIVERGE from the `--deltaq-mode=2` reference — so the pass above is not the
/// trivial "delta never fired" case.
#[test]
fn deltaq_mode2_knob_bites() {
    let cell = cell(192, 192, 12);
    let c_stream = cell.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, 2)]);
    let real = EncodeCell::frame_obu_payload(&c_stream);
    // The deltaq-mode=2 reference must genuinely differ from a plain encode
    // (proves the delta fired on this content).
    let plain = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&[]));
    assert_ne!(
        real, plain,
        "mode-2 must modulate on this cell for the witness to be meaningful"
    );
    // Port with the mode-2 arm OFF diverges from the mode-2 reference...
    let without = cell.port_encode_with(&c_stream, &ToggleKnobs::default());
    assert_ne!(
        without, real,
        "port without the wavelet delta-q arm must NOT match the --deltaq-mode=2 stream"
    );
    // ...and ON it matches (the gate above, re-pinned here as the witness pair).
    let with = cell.port_encode_with(&c_stream, &mode2_knobs());
    assert_eq!(
        with, real,
        "port with the wavelet delta-q arm must match the --deltaq-mode=2 stream"
    );
}
