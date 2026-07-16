//! Runnable validation of the RD-closeness harness (`aom_bench::rd_close`) —
//! the shared gate for the stills-parity bulk-port wave.
//!
//! Run with per-cell table output:
//! `cargo test -p aom-bench --test rd_close_harness -- --nocapture`
//!
//! Three properties are pinned, none vacuously:
//! 1. On the PROVEN byte-exact envelope (landed KB-6 real-content cells) the
//!    harness reports EXACT via the bit-identical fast path, zero deltas, and
//!    `assert_rd_close` passes.
//! 2. On a GENUINE divergence (a cq63 encode masquerading as the "port" side
//!    against a cq20 reference of the same source) the harness measures a
//!    large size delta and a large zensim drop, the cell is OUT of the default
//!    bands, and `assert_rd_close` panics. This proves the bands discriminate:
//!    a real quality/size regression cannot slip through as "close".
//! 3. The stream splicer is an identity when handed the reference's own frame
//!    payload, and the YUV→RGB transform handles mono/subsampling/bit-depth.

use aom_bench::EncodeCell;
use aom_bench::rd_close::{
    RdBands, assert_rd_close, compare_cell, render_table, run_stock_cell, splice_frame_obu,
    yuv_to_rgb8,
};

/// KB-6-proven real-content cells (all landed byte-match gates: 64×64 full
/// frame + the 196×196 partial-SB multi-SB frame). The stock port encode is
/// byte-identical to real aomenc on these, so the harness MUST report EXACT.
#[test]
fn rd_close_stock_cells_exact_on_proven_envelope() {
    let cells = [
        EncodeCell::real_content("rdc_64_cq12", "av1-1-b8-01-size-64x64", None, 12, 0),
        EncodeCell::real_content("rdc_64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0),
        EncodeCell::real_content("rdc_196_cq20", "av1-1-b8-01-size-196x196", None, 20, 0),
    ];
    let results: Vec<_> = cells.iter().map(run_stock_cell).collect();
    for r in &results {
        assert!(
            r.bit_identical,
            "{}: proven byte-exact cell no longer bit-identical — that is an \
             encoder REGRESSION, not a harness problem",
            r.label
        );
        assert_eq!(r.size_port, r.size_c, "{}: sizes must match", r.label);
        assert_eq!(
            r.size_delta_pct, 0.0,
            "{}: exact cells have zero size delta",
            r.label
        );
        assert_eq!(
            r.zensim_drop, 0.0,
            "{}: exact cells have zero zensim drop",
            r.label
        );
        // The recon genuinely resembles the source (anti-vacuous: a broken
        // decode or RGB conversion would crater this far below any plausible
        // value for web-range cq on real content).
        assert!(
            r.zensim_port > 20.0,
            "{}: zensim {:.2} is implausibly low — decode/convert path broken?",
            r.label,
            r.zensim_port
        );
    }
    // The gate passes on the proven envelope.
    assert_rd_close(&results, &RdBands::default());
}

/// A cq63 encode of the same source, spliced in as the "port" stream against
/// a cq20 reference, is a REAL quality+size divergence: the harness must
/// measure it and the default bands must reject it.
#[test]
fn rd_close_flags_genuine_divergence_out_of_band() {
    let cell20 = EncodeCell::real_content("rdc_div_cq20", "av1-1-b8-01-size-64x64", None, 20, 0);
    let cell63 = EncodeCell::real_content("rdc_div_cq63", "av1-1-b8-01-size-64x64", None, 63, 0);
    aom_sys_ref::ref_init();
    let c20 = cell20.c_encode();
    let c63 = cell63.c_encode();
    assert!(!c20.is_empty() && !c63.is_empty());

    let fake_port_tu = splice_frame_obu(&c20, &EncodeCell::frame_obu_payload(&c63));
    let res = compare_cell("rdc_divergence_cq63_vs_cq20", &cell20, &fake_port_tu, &c20);
    println!("{}", render_table(std::slice::from_ref(&res), &RdBands::default()));

    assert!(!res.bit_identical, "cq63 vs cq20 cannot be bit-identical");
    assert!(
        res.size_port < res.size_c,
        "cq63 must code fewer bytes than cq20 (got {} vs {})",
        res.size_port,
        res.size_c
    );
    assert!(
        res.size_delta_pct < -5.0,
        "size delta {:.2}% must exceed the -5% band on a cq20→cq63 jump",
        res.size_delta_pct
    );
    assert!(
        res.zensim_drop > 0.5,
        "zensim drop {:.3} must exceed the 0.5 band on a cq20→cq63 jump \
         (zs_c={:.3} zs_port={:.3})",
        res.zensim_drop,
        res.zensim_c,
        res.zensim_port
    );
    assert!(!res.within(&RdBands::default()));
    assert_eq!(res.verdict(&RdBands::default()), "FAIL");

    // And the gate assert genuinely fires.
    let results = vec![res];
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_rd_close(&results, &RdBands::default());
    }))
    .is_err();
    assert!(panicked, "assert_rd_close must panic on an out-of-band cell");
}

/// Splicing a stream's OWN frame payload back in reproduces it byte-for-byte
/// (OBU walk + leb128 re-encode fidelity).
#[test]
fn rd_close_splice_is_identity_on_own_payload() {
    let cell = EncodeCell::real_content("rdc_splice_id", "av1-1-b8-01-size-64x64", None, 32, 0);
    aom_sys_ref::ref_init();
    let c = cell.c_encode();
    let respliced = splice_frame_obu(&c, &EncodeCell::frame_obu_payload(&c));
    assert_eq!(respliced, c, "splice(stream, own payload) must be an identity");
}

/// The shared YUV→RGB transform: mono replicates luma, 4:2:0 upsamples
/// nearest-neighbour, bd10 rounds to 8 bits — hand-checked values.
#[test]
fn yuv_to_rgb8_mono_subsample_and_bd10() {
    // Mono 8x8, flat Y=128: BT.601 limited → (298*(128-16)+128)>>8 = 130.
    let y = vec![128u16; 64];
    let rgb = yuv_to_rgb8(&y, &[], &[], 8, 8, true, 0, 0, 8);
    assert_eq!(rgb.len(), 64);
    assert!(rgb.iter().all(|px| *px == [130, 130, 130]));

    // 4:2:0 8x8: neutral chroma (128) leaves mono math; each 2x2 luma block
    // shares one chroma sample (nearest-neighbour indexing must not panic and
    // must hit the right sample — probe with one hot chroma cell).
    let y = vec![128u16; 64];
    let mut u = vec![128u16; 16];
    let v = vec![128u16; 16];
    u[0] = 255; // affects luma pixels (0..2, 0..2) only
    let rgb = yuv_to_rgb8(&y, &u, &v, 8, 8, false, 1, 1, 8);
    assert_ne!(rgb[0], [130, 130, 130], "hot U must tint pixel (0,0)");
    assert_eq!(rgb[0], rgb[1], "pixel (0,1) shares the chroma sample");
    assert_eq!(rgb[0], rgb[8], "pixel (1,0) shares the chroma sample");
    assert_eq!(rgb[2], [130, 130, 130], "pixel (0,2) uses the next sample");

    // bd10 mono: 512 rounds to (512+2)>>2 = 128 → same 130 grey; and the
    // rounding half-up: 514 → (514+2)>>2 = 129.
    let y10 = vec![512u16; 64];
    let rgb10 = yuv_to_rgb8(&y10, &[], &[], 8, 8, true, 0, 0, 10);
    assert!(rgb10.iter().all(|px| *px == [130, 130, 130]));
    let y10b = vec![514u16; 64];
    let rgb10b = yuv_to_rgb8(&y10b, &[], &[], 8, 8, true, 0, 0, 10);
    let expect = {
        let c = 129 - 16;
        (((298 * c + 128) >> 8) as u8).min(255)
    };
    assert!(rgb10b.iter().all(|px| *px == [expect, expect, expect]));
}
