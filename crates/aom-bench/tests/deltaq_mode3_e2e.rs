//! `--deltaq-mode=3` (`DELTA_Q_PERCEPTUAL_AI`, family C5) end-to-end byte-match
//! gate: the port's own wiener-variance per-SB delta-q encode vs real aomenc
//! `--deltaq-mode=3` on REAL image content. The per-SB qindex map is derived by
//! the port (`av1_set_mb_wiener_variance` + `av1_get_sbq_perceptual_ai`); only
//! the frame-header fields are bootstrapped (and the delta_q ones are
//! cross-checked, never copied — see `port_encode_impl`'s mode-3 arm).
//!
//! Scope: bd8 4:2:0, dims a multiple of 64 and 8px (mode-3 fires; a 128x128
//! frame is too uniform to modulate below cq63). The 196x196 partial-SB frame
//! and highbd are follow-ups (the wiener preprocessing's scope note).

use aom_bench::{EncodeCell, ToggleKnobs};

const AV1E_SET_DELTAQ_MODE: i32 = 107;

fn mode3_knobs() -> ToggleKnobs {
    ToggleKnobs {
        deltaq_mode3: true,
        ..Default::default()
    }
}

/// Real-content cell at `(w, h)` cropped from the 196x196 conformance vector.
fn cell(w: usize, h: usize, cq: i32) -> EncodeCell {
    EncodeCell::real_content(
        &format!("deltaq3_{w}x{h}_cq{cq}"),
        "av1-1-b8-01-size-196x196",
        Some((w, h, 0, 0)),
        cq,
        0,
    )
}

/// bd10 real-content cell cropped from the 640x360 b10 conformance vector
/// (4:2:0, genuine 10-bit image statistics that modulate the wiener map).
fn cell_b10(w: usize, h: usize, cq: i32) -> EncodeCell {
    EncodeCell::real_content(
        &format!("deltaq3_b10_{w}x{h}_cq{cq}"),
        "av1-1-b10-00-quantizer-00",
        Some((w, h, 0, 0)),
        cq,
        0,
    )
}

/// bd12 cell built by promoting the bd10 real content into the 12-bit range
/// (`pix << 2`) — there is no b12 conformance vector, but the highbd FP-quantize
/// arm (`av1_highbd_quantize_fp`) and the bd12 quant tables are the only bd>8
/// difference, so shifted-but-real spatial structure exercises them faithfully.
fn cell_b12(w: usize, h: usize, cq: i32) -> EncodeCell {
    let mut c = cell_b10(w, h, cq);
    c.label = format!("deltaq3_b12_{w}x{h}_cq{cq}");
    c.bd = 12;
    for p in [&mut c.y, &mut c.u, &mut c.v] {
        for v in p.iter_mut() {
            *v <<= 2;
        }
    }
    c
}

/// Per-cell match check. `port_encode_with` returns the assembled frame OBU
/// PAYLOAD (compare directly to the reference payload — the assert_byte_exact
/// convention). Returns `Ok(len)` on byte-match, else the first differing byte.
fn run_cell(cell: &EncodeCell) -> Result<usize, String> {
    let c_stream = cell.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, 3)]);
    let real = EncodeCell::frame_obu_payload(&c_stream);
    let ours = cell.port_encode_with(&c_stream, &mode3_knobs());
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

/// The hard byte-match gate: every cell's port `--deltaq-mode=3` encode is
/// byte-identical to real aomenc across the web quality range + a non-square
/// shape (192x128 = 3x2 SBs). Any divergence is a regression.
#[test]
fn deltaq_mode3_perceptual_ai_e2e() {
    let mut cells: Vec<EncodeCell> = [12, 20, 32, 48, 63]
        .into_iter()
        .map(|cq| cell(192, 192, cq))
        .collect();
    // Non-square shape (3x2 SBs) — exercises the running-base delta chain across
    // a different SB raster.
    cells.push(cell(192, 128, 32));
    cells.push(cell(128, 192, 32));
    // Highbd (bd10 + bd12) — the FP-quantize arm now dispatches
    // av1_highbd_quantize_fp for bd>8. Same 192x192 / 192x128 shapes on real
    // 10-bit content across the web quality range.
    for cq in [12, 20, 32, 48, 63] {
        cells.push(cell_b10(192, 192, cq));
    }
    cells.push(cell_b10(192, 128, 32));
    for cq in [12, 32, 63] {
        cells.push(cell_b12(192, 192, cq));
    }

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
        "--deltaq-mode=3 (PERCEPTUAL_AI) e2e byte-match: {matched}/{}\n{report}",
        cells.len()
    );
    assert_eq!(
        matched,
        cells.len(),
        "not all --deltaq-mode=3 cells byte-match real aomenc:\n{report}"
    );
}

/// Anti-vacuous witness: the wiener delta-q machinery must do real work. On a
/// cell where mode-3 modulates the qindex (the reference differs from a plain
/// encode), the port WITHOUT the mode-3 arm (`deltaq_mode3 = false`) must
/// DIVERGE from the `--deltaq-mode=3` reference — so the pass above is not the
/// trivial "delta never fired" case.
#[test]
fn deltaq_mode3_knob_bites() {
    let cell = cell(192, 192, 12);
    let c_stream = cell.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, 3)]);
    let real = EncodeCell::frame_obu_payload(&c_stream);
    // The deltaq-mode=3 reference must genuinely differ from a plain encode
    // (proves the delta fired on this content).
    let plain = EncodeCell::frame_obu_payload(&cell.c_encode_ctrls(&[]));
    assert_ne!(
        real, plain,
        "mode-3 must modulate on this cell for the witness to be meaningful"
    );
    // Port with the mode-3 arm OFF diverges from the mode-3 reference...
    let without = cell.port_encode_with(&c_stream, &ToggleKnobs::default());
    assert_ne!(
        without, real,
        "port without the wiener delta-q arm must NOT match the --deltaq-mode=3 stream"
    );
    // ...and ON it matches (the gate above, re-pinned here as the witness pair).
    let with = cell.port_encode_with(&c_stream, &mode3_knobs());
    assert_eq!(
        with, real,
        "port with the wiener delta-q arm must match the --deltaq-mode=3 stream"
    );
}
