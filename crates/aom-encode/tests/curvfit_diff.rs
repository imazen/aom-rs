//! Differential: the ported CURVFIT model-rd core
//! ([`aom_encode::interp_rd::av1_model_rd_curvfit`]) vs the REAL exported C
//! `av1_model_rd_curvfit` (av1/encoder/rd.c:1064) — the
//! `MODELRD_TYPE_INTERP_FILTER` model every inter candidate's switchable
//! interp-filter search prices with (KB-16 / INTER-CHUNK2-HANDOFF §2026-07-23).
//!
//! Bit-exact f64 comparison (the port replicates C's op order; the reference
//! build has no FMA), swept over every (bsize, sse-norm category) grid row and
//! the full clamped xqr range including both clamp edges.

use aom_sys_ref as c;

#[test]
fn curvfit_matches_real_c() {
    c::ref_init();
    let mut cases = 0usize;
    // Every block size (22) hits its rcat row; sse_norm on both sides of the
    // 16.0 dcat threshold; xqr swept across and beyond the clamp range.
    let sse_norms = [
        0.001, 0.5, 1.0, 4.0, 15.9, 16.0, 16.1, 64.0, 1024.0, 65536.0,
    ];
    for bsize in 0..22usize {
        for &sse_norm in &sse_norms {
            let mut xqr = -20.0f64;
            while xqr <= 20.0 {
                let (pr, pd) = aom_encode::interp_rd::av1_model_rd_curvfit(bsize, sse_norm, xqr);
                let (cr, cd) = c::ref_model_rd_curvfit(bsize, sse_norm, xqr);
                assert_eq!(
                    pr.to_bits(),
                    cr.to_bits(),
                    "rate_f: bsize {bsize} sse_norm {sse_norm} xqr {xqr}: port {pr} vs C {cr}"
                );
                assert_eq!(
                    pd.to_bits(),
                    cd.to_bits(),
                    "dist_f: bsize {bsize} sse_norm {sse_norm} xqr {xqr}: port {pd} vs C {cd}"
                );
                cases += 1;
                xqr += 0.03125; // 1/32 — off-grid phases exercise the cubic
            }
        }
    }
    assert!(cases > 200_000, "swept {cases} cases");
}

/// The derived model (rate, dist) fold — sse=0 short-circuit, the rounding,
/// and the skip-vs-rate RDCOST fold — sanity-anchored on physics: zero sse is
/// free, and monotone sse never DECREASES the modelled distortion.
#[test]
fn model_rd_with_curvfit_fold_sanity() {
    use aom_encode::interp_rd::model_rd_with_curvfit;
    assert_eq!(model_rd_with_curvfit(12, 0, 4096, 1024, 8, 100_000), (0, 0));
    let mut last_dist = 0i64;
    for sse in [100i64, 1_000, 10_000, 100_000, 1_000_000] {
        let (rate, dist) = model_rd_with_curvfit(12, sse, 4096, 1024, 8, 100_000);
        assert!(rate >= 0);
        assert!(dist >= last_dist, "dist must be monotone in sse");
        last_dist = dist;
    }
}
