//! Differential harness for the RD-multiplier (lambda) derivation
//! `av1_compute_rd_mult_based_on_qindex` / `av1_compute_rd_mult`
//! (av1/encoder/rd.c) vs C. Exercises the `f64` multiplier math (no FMA in the
//! reference build) exhaustively over qindex × update_type × tuning × mode ×
//! bit_depth, plus the two-pass layer/boost adjustment.

use aom_encode::rd::{
    EncMode, FrameType, FrameUpdateType, TuneMetric, av1_compute_rd_mult,
    av1_compute_rd_mult_based_on_qindex,
};
use aom_sys_ref as c;

const UPDATES: [FrameUpdateType; 7] = [
    FrameUpdateType::Kf,
    FrameUpdateType::Lf,
    FrameUpdateType::Gf,
    FrameUpdateType::Arf,
    FrameUpdateType::Overlay,
    FrameUpdateType::IntnlOverlay,
    FrameUpdateType::IntnlArf,
];
const TUNINGS: [TuneMetric; 3] = [TuneMetric::Psnr, TuneMetric::Iq, TuneMetric::Ssimulacra2];
const MODES: [EncMode; 3] = [EncMode::Good, EncMode::Realtime, EncMode::Allintra];

#[test]
fn compute_rd_mult_based_on_qindex_matches_c() {
    for &bd in &[8u8, 10, 12] {
        for &ut in &UPDATES {
            for &tuning in &TUNINGS {
                for &mode in &MODES {
                    for qindex in 0..=255i32 {
                        let got = av1_compute_rd_mult_based_on_qindex(bd, ut, qindex, tuning, mode);
                        let want = c::ref_compute_rd_mult_based_on_qindex(
                            bd as i32,
                            ut as i32,
                            qindex,
                            tuning as i32,
                            mode as i32,
                        );
                        assert_eq!(
                            got, want,
                            "rd_mult_qidx bd={bd} update={ut:?} tuning={tuning:?} mode={mode:?} qindex={qindex}"
                        );
                    }
                }
            }
        }
    }
}

/// Every `aom_tune_metric` that is neither `AOM_TUNE_IQ` (10) nor
/// `AOM_TUNE_SSIMULACRA2` (11) takes the same RD-multiplier path as PSNR. This
/// pins that assumption (so [`TuneMetric::Psnr`] faithfully represents them all):
/// the C reference for those tuning ints must equal both the C PSNR result and
/// our `TuneMetric::Psnr` result.
#[test]
fn non_iq_ssim2_tunings_match_psnr_path() {
    for &bd in &[8u8, 10, 12] {
        for &ut in &UPDATES {
            for &mode in &MODES {
                for qindex in (0..=255i32).step_by(7) {
                    let psnr_rust =
                        av1_compute_rd_mult_based_on_qindex(bd, ut, qindex, TuneMetric::Psnr, mode);
                    for other_tuning in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13] {
                        let want = c::ref_compute_rd_mult_based_on_qindex(
                            bd as i32,
                            ut as i32,
                            qindex,
                            other_tuning,
                            mode as i32,
                        );
                        assert_eq!(
                            psnr_rust, want,
                            "tuning={other_tuning} should match PSNR path: bd={bd} update={ut:?} mode={mode:?} qindex={qindex}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn compute_rd_mult_matches_c() {
    // The layer/boost adjustment fires only when is_stat_consumption_stage &&
    // !use_fixed_qp_offsets && frame_type != KEY. Sweep all the gating flags plus
    // the full layer_depth [0,7) and boost_index [0,16) index ranges.
    for &bd in &[8u8, 10, 12] {
        for &ut in &UPDATES {
            for &frame_type in &[FrameType::Key, FrameType::NonKey] {
                for &use_fixed in &[false, true] {
                    for &is_stat in &[false, true] {
                        for layer_depth in 0..7i32 {
                            for boost_index in 0..16i32 {
                                // A representative spread of qindex values.
                                for qindex in (0..=255i32).step_by(17) {
                                    let got = av1_compute_rd_mult(
                                        qindex,
                                        bd,
                                        ut,
                                        layer_depth,
                                        boost_index,
                                        frame_type,
                                        use_fixed,
                                        is_stat,
                                        TuneMetric::Psnr,
                                        EncMode::Good,
                                    );
                                    let want = c::ref_compute_rd_mult(
                                        qindex,
                                        bd as i32,
                                        ut as i32,
                                        layer_depth,
                                        boost_index,
                                        frame_type as i32,
                                        use_fixed as i32,
                                        is_stat as i32,
                                        TuneMetric::Psnr as i32,
                                        EncMode::Good as i32,
                                    );
                                    assert_eq!(
                                        got, want,
                                        "rd_mult bd={bd} update={ut:?} ft={frame_type:?} fixed={use_fixed} stat={is_stat} ld={layer_depth} bi={boost_index} qindex={qindex}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
