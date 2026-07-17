//! Differential tests for the `--deltaq-mode=3` (`DELTA_Q_PERCEPTUAL_AI`,
//! family C5) port pieces vs the REAL exported libaom C functions. Every port
//! function that has a directly-callable C counterpart is pinned here so a
//! transcription slip is caught before it can perturb an e2e stream.

use aom_encode::allintra_vis;
use aom_sys_ref as c;

/// `av1_get_deltaq_offset` (rd.c:466) — the DC-quant table walk from a base
/// qindex to the offset closest to `q/sqrt(beta)`. Swept over bit depth ×
/// every qindex × a fine beta grid (well past the [0.25, 4.0] clamp the
/// callers apply, so the full table-walk in both directions is exercised).
#[test]
fn get_deltaq_offset_matches_c() {
    let mut mismatches = 0usize;
    let mut checked = 0usize;
    for &bd in &[8u8, 10, 12] {
        for qindex in 0..=255i32 {
            // Betas: dense around 1.0, plus the extremes both callers can
            // produce after their own clamps, plus a few out-of-clamp values.
            for &beta in &[
                0.1, 0.25, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.25, 1.5,
                1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0, 10.0,
            ] {
                let port = allintra_vis::av1_get_deltaq_offset(bd, qindex, beta);
                let cref = c::ref_av1_get_deltaq_offset(bd, qindex, beta);
                checked += 1;
                if port != cref {
                    mismatches += 1;
                    if mismatches <= 20 {
                        eprintln!(
                            "MISMATCH bd={bd} qindex={qindex} beta={beta}: port={port} c={cref}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "av1_get_deltaq_offset diverged from C on {mismatches}/{checked} cases"
    );
}

/// Structural smoke test for the wiener-map per-SB qindex chain
/// ([`WeberVarMap::av1_get_sbq_perceptual_ai`]) — bounds + the
/// perceptual-AI direction. A uniform map (one 64x64 SB = 64 8x8 blocks):
/// raising `norm_wiener_variance` relative to the SB's wiener var raises
/// beta, which lowers (or holds) the qindex. Byte-exactness of the chain is
/// gated e2e vs `aomenc --deltaq-mode=3`; this only catches gross regressions.
#[test]
fn sbq_perceptual_ai_bounds_and_direction() {
    use aom_encode::allintra_vis::{WeberStats, WeberVarMap, DELTA_Q_RES_PERCEPTUAL};
    let mi = 16; // one BLOCK_64X64 SB
    let blk = WeberStats {
        src_variance: 4000,
        rec_variance: 3600,
        src_pix_max: 200,
        rec_pix_max: 190,
        distortion: 800,
        satd: 5000,
        max_scale: 6.0,
    };
    let mk = |norm: i64| WeberVarMap {
        stats: vec![blk; (mi * mi) as usize],
        mi_rows: mi,
        mi_cols: mi,
        norm_wiener_variance: norm,
    };
    let base = 128;
    let mut prev = i32::MAX;
    for &norm in &[1i64, 100, 1_000, 10_000, 100_000, 1_000_000] {
        let q = mk(norm).av1_get_sbq_perceptual_ai(base, 8, DELTA_Q_RES_PERCEPTUAL, mi, mi, 0, 0);
        assert!((1..=255).contains(&q), "qindex {q} out of range for norm {norm}");
        // Monotone non-increasing in norm (higher norm => higher beta => finer q).
        assert!(
            q <= prev,
            "qindex must be non-increasing in norm: norm={norm} q={q} prev={prev}"
        );
        prev = q;
    }
    // base_qindex 0 keeps qindex >= MINQ (no forced +1).
    let q0 = mk(1000).av1_get_sbq_perceptual_ai(0, 8, DELTA_Q_RES_PERCEPTUAL, mi, mi, 0, 0);
    assert!((0..=255).contains(&q0));
}
