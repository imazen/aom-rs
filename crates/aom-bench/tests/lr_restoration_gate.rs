//! Loop-restoration ENCODER-SEARCH parity gate (`--enable-restoration=1`,
//! PARITY.md C2): the port's OWN `av1_pick_filter_restoration` +
//! RU-interleaved pack + restoration header vs real aomenc with the same
//! knob, on real conformance-vector content across the quality range.
//!
//! Per cell:
//! - real: `EncodeCell::c_encode_lr()` (`AV1E_SET_ENABLE_RESTORATION=1`);
//! - port: `EncodeCell::port_encode_lr()` — pack + LF derive/apply + the
//!   ported LR search + repack with the per-SB-root RU params + derived
//!   restoration header fields (decisions NEVER copied from the bootstrap);
//! - primary record: byte-identical (EXACT) or the rd_close verdict
//!   (`|size| <= 5%` AND `zensim_drop <= 0.5` — decode BOTH with the port
//!   decoder, score vs source with zensim);
//! - decision diagnostic: the parsed frame-header restoration decision of
//!   both streams (`parse_restoration_decision`) printed per cell.
//!
//! Anti-vacuous floors: at least one cell where the REAL encoder's decision
//! restores a plane (the reference actually exercises LR), and at least one
//! where the PORT's does. The whole gate then asserts every cell within the
//! rd_close bands (bit-identical cells auto-pass and are recorded EXACT).

use aom_bench::rd_close::{self, RdBands};
use aom_bench::{EncodeCell, parse_restoration_decision};

fn cells() -> Vec<EncodeCell> {
    vec![
        // 1-SB frame across the quality range (KB-6 real-content recipe).
        EncodeCell::real_content("lr_size64_cq12", "av1-1-b8-01-size-64x64", None, 12, 0),
        EncodeCell::real_content("lr_size64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0),
        EncodeCell::real_content("lr_size64_cq48", "av1-1-b8-01-size-64x64", None, 48, 0),
        // Multi-SB + partial-edge SBs (196 = 3 SBs + 4px overhang).
        EncodeCell::real_content("lr_size196_cq20", "av1-1-b8-01-size-196x196", None, 20, 0),
        EncodeCell::real_content("lr_size196_cq48", "av1-1-b8-01-size-196x196", None, 48, 0),
        // 352x288: multi-unit grids at the smaller sizes of the unit-size
        // descent (352 -> 3 units at 128 / 6 at 64), real photographic mix.
        EncodeCell::real_content("lr_quant00_cq32", "av1-1-b8-00-quantizer-00", None, 32, 0),
        EncodeCell::real_content("lr_quant00_cq55", "av1-1-b8-00-quantizer-00", None, 55, 0),
        // 10-bit arm (the highbd search paths).
        EncodeCell::real_content("lr_b10_quant00_cq32", "av1-1-b10-00-quantizer-00", None, 32, 0),
    ]
}

#[test]
fn lr_restoration_search_rd_close_vs_real_aomenc() {
    let bands = RdBands::default();
    let mut results = Vec::new();
    let mut real_active = 0usize;
    let mut port_active = 0usize;
    let mut exact = 0usize;
    let mut decisions_equal = 0usize;

    for cell in cells() {
        let c_tu = cell.c_encode_lr();
        assert!(!c_tu.is_empty(), "{}: real LR encode failed", cell.label);
        let port_payload = cell.port_encode_lr(&c_tu);
        let port_tu = rd_close::splice_frame_obu(&c_tu, &port_payload);

        let (real_frt, real_us) = parse_restoration_decision(&c_tu);
        let (port_frt, port_us) = parse_restoration_decision(&port_tu);
        if real_frt.iter().any(|&t| t != 0) {
            real_active += 1;
        }
        if port_frt.iter().any(|&t| t != 0) {
            port_active += 1;
        }
        let decision_eq = real_frt == port_frt && (real_frt == [0; 3] || real_us == port_us);
        if decision_eq {
            decisions_equal += 1;
        }

        let r = rd_close::compare_cell(&cell.label, &cell, &port_tu, &c_tu);
        if r.bit_identical {
            exact += 1;
        }
        eprintln!(
            "{}: real_frt={real_frt:?} us={real_us:?} | port_frt={port_frt:?} us={port_us:?} \
             | decision_{} | {}",
            cell.label,
            if decision_eq { "EQUAL" } else { "DIFFERS" },
            r.verdict(&bands),
        );
        results.push(r);
    }

    eprintln!("{}", rd_close::render_table(&results, &bands));
    eprintln!(
        "LR gate summary: {}/{} cells real-LR-active, {}/{} port-LR-active, {}/{} \
         decisions equal, {}/{} bit-identical",
        real_active,
        results.len(),
        port_active,
        results.len(),
        decisions_equal,
        results.len(),
        exact,
        results.len(),
    );

    // Anti-vacuous: the reference streams must actually exercise LR on this
    // grid, and the port's search must fire somewhere too — otherwise the
    // rd_close pass would be an empty statement about the feature.
    assert!(
        real_active >= 1,
        "no cell made the REAL encoder restore a plane — the gate grid is vacuous"
    );
    assert!(
        port_active >= 1,
        "the port's LR search never fired on a grid where the reference does"
    );

    rd_close::assert_rd_close(&results, &bands);

    // BYTE-IDENTITY assertions (the section-A gate): the first complete run
    // measured 8/8 cells bit-identical with 8/8 decision equality, so this
    // gate holds the feature at full byte-exactness — any weaker outcome is
    // a regression, not a band question. (rd_close stays above for the
    // richer failure report.)
    assert_eq!(
        decisions_equal,
        results.len(),
        "every cell's restoration decision must equal the C encoder's"
    );
    for r in &results {
        assert!(
            r.bit_identical,
            "{}: LR-on encode must be BYTE-IDENTICAL to real aomenc (measured 8/8 on landing)",
            r.label
        );
    }
}
