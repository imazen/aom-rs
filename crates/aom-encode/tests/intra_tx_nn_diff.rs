//! Differential: `ml_predict_intra_tx_depth_prune` (tx_search.rs — the
//! speed>=6 `prune_intra_tx_depths_using_nn` 8x8 model, tx_search.c:2823) vs
//! a C oracle built from the REAL `av1_nn_predict_c` (`ref_nn_predict`, with
//! `reduce_prec = 1` exactly as the :2879 call site) on the SAME transcribed
//! weight tables, plus an independent verbatim transcription of the C
//! feature engineering (`get_mean_dev_features` + the log1pf features +
//! mean/std normalization) and the `av1_intra_tx_prune_nn_thresh_8x8`
//! decision, over randomized 8x8 residuals / source variances / qindexes.
//!
//! What this pins beyond the byte gate: the port's feature builder and NN
//! arithmetic are fed through the REAL C inference, so a transcription slip
//! in the weight tables, the f32/f64 mixing of `get_dev`, the `log1pf`
//! feature forms, the normalization, the prec-reduce, or the SPLIT/LARGEST
//! threshold compare would diverge here even when the gate grid happens not
//! to reach the affected region of feature space.

use aom_encode::intra_tx_nn_weights as w;
use aom_encode::tx_search::{TxPruneType, ml_predict_intra_tx_depth_prune};
use aom_sys_ref as c;

struct XorShift(u64);
impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range_u(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }
    fn range_i(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as i64
    }
}

/// `get_dev` (tx_search.c:1697), verbatim.
fn c_get_dev(mean: f32, x2_sum: f64, num: i32) -> f32 {
    let e_x2 = (x2_sum / f64::from(num)) as f32;
    let diff = e_x2 - mean * mean;
    if diff > 0.0 { diff.sqrt() } else { 0.0 }
}

/// `get_mean_dev_features` (tx_search.c:1709) + the two log1pf features +
/// normalization (`ml_predict_intra_tx_depth_prune`, :2856-2875), verbatim
/// for the 8x8 case (subw = subh = 4, four sub-blocks).
fn c_features(diff: &[i16; 64], source_variance: u32, qindex: i32) -> [f32; 14] {
    let (bw, bh, stride) = (8usize, 8usize, 8usize);
    let (subw, subh) = (4usize, 4usize);
    let num = (bw * bh) as i32;
    let sub_num = (subw * subh) as i32;
    let mut features = [0f32; 14];
    let mut fi = 2usize;
    let mut total_x_sum = 0i32;
    let mut total_x2_sum = 0i64;
    let mut num_sub_blks = 0i32;
    let mut mean2_sum = 0.0f64;
    let mut dev_sum = 0.0f32;
    for row in (0..bh).step_by(subh) {
        for col in (0..bw).step_by(subw) {
            let mut x_sum = 0i32;
            let mut x2_sum = 0i64;
            for r in 0..subh {
                for cc in 0..subw {
                    let d = i32::from(diff[(row + r) * stride + col + cc]);
                    x_sum += d;
                    x2_sum += i64::from(d) * i64::from(d);
                }
            }
            total_x_sum += x_sum;
            total_x2_sum += x2_sum;
            let mean = x_sum as f32 / sub_num as f32;
            let dev = c_get_dev(mean, x2_sum as f64, sub_num);
            features[fi] = mean;
            features[fi + 1] = dev;
            fi += 2;
            mean2_sum += f64::from(mean * mean);
            dev_sum += dev;
            num_sub_blks += 1;
        }
    }
    let lvl0_mean = total_x_sum as f32 / num as f32;
    features[0] = lvl0_mean;
    features[1] = c_get_dev(lvl0_mean, total_x2_sum as f64, num);
    features[fi] = c_get_dev(lvl0_mean, mean2_sum, num_sub_blks);
    features[fi + 1] = dev_sum / num_sub_blks as f32;
    fi += 2;
    features[fi] = (source_variance as f32).ln_1p();
    fi += 1;
    let dc_q = i32::from(aom_quant::av1_dc_quant_qtx(qindex, 0, 8));
    features[fi] = ((dc_q * dc_q) as f32 / 256.0f32).ln_1p();
    fi += 1;
    assert_eq!(fi, 14);
    for i in 0..14 {
        features[i] = (features[i] - w::MEAN[i]) / w::STD[i];
    }
    features
}

/// 4000 randomized decisions: the port's full chain must equal the
/// REAL-`av1_nn_predict_c` oracle + threshold decision on every one, and all
/// three verdicts must be exercised.
#[test]
fn intra_tx_depth_nn_matches_real_c_inference() {
    c::ref_init();
    let mut rng = XorShift(0x5eed_1e57_0a0b_0c0d);
    let weights_flat: Vec<f32> = w::W0.iter().chain(w::W1.iter()).copied().collect();
    let bias_flat: Vec<f32> = w::B0.iter().chain(w::B1.iter()).copied().collect();
    let mut counts = [0usize; 3]; // None / Split / Largest
    for iter in 0..4000 {
        // Residual regimes: flat-ish (skip-shaped), textured, mixed — the
        // bd8 residual domain is [-255, 255].
        let mut diff = [0i16; 64];
        let regime = iter % 3;
        let amp = match regime {
            0 => rng.range_i(1, 8),
            1 => rng.range_i(32, 255),
            _ => rng.range_i(4, 64),
        };
        let bias = rng.range_i(-amp, amp);
        for d in diff.iter_mut() {
            *d = (bias + rng.range_i(-amp, amp)) as i16;
        }
        let source_variance = rng.range_u(0, 20000) as u32;
        let qindex = rng.range_u(1, 255) as i32;

        // Port: the full feature->NN->threshold chain.
        let got = ml_predict_intra_tx_depth_prune(&diff, 8, source_variance, qindex, 8);

        // Oracle: verbatim features -> REAL av1_nn_predict_c (reduce_prec=1)
        // -> the av1_intra_tx_prune_nn_thresh_8x8 decision.
        let features = c_features(&diff, source_variance, qindex);
        let score =
            c::ref_nn_predict(&features, 14, 1, &[16], &weights_flat, &bias_flat, true)[0];
        let want = if score <= w::PRUNE_THRESH[0] {
            TxPruneType::Split
        } else if score > w::PRUNE_THRESH[1] {
            TxPruneType::Largest
        } else {
            TxPruneType::None
        };
        assert_eq!(
            got, want,
            "iter {iter}: port {got:?} vs REAL-C oracle {want:?} (score {score}, sv \
             {source_variance}, q {qindex})"
        );
        counts[match got {
            TxPruneType::None => 0,
            TxPruneType::Split => 1,
            TxPruneType::Largest => 2,
        }] += 1;
    }
    eprintln!(
        "intra_tx_depth_nn_matches_real_c_inference: 4000/4000 decisions identical \
         (None {} / Split {} / Largest {})",
        counts[0], counts[1], counts[2]
    );
    assert!(
        counts.iter().all(|&n| n > 0),
        "the random sweep must exercise all three verdicts: {counts:?}"
    );
}
