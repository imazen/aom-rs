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
