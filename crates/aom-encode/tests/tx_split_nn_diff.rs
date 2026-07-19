//! Differential for `var_tx::ml_predict_tx_split` (tx_search.c:1755) — the inter
//! var-tx split-prediction NN gate. The port's manual NN eval (node-major w0,
//! ReLU hidden, linear 1-output) + `av1_nn_output_prec_reduce` + `clamp((int)
//! (score*10000), ±80000)` is checked byte-for-byte against the REAL C
//! `av1_nn_predict_c` (`ref_nn_predict`) fed the port's `get_mean_dev_features`
//! output + the transcribed weights, across every tx size that has a nnconfig,
//! over random residuals spanning the clamp boundaries.
//!
//! Weight transcription is guarded by (a) the exact-named-array + length asserts
//! in `xtask/transcribe_tx_split_nn.py`, and (b) the hardcoded spot-check below.

use aom_encode::tx_search::get_mean_dev_features;
use aom_encode::tx_split_nn_weights::TX_SPLIT_NN;
use aom_encode::var_tx::ml_predict_tx_split;
use aom_sys_ref as c;

const TXS_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TXS_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

#[test]
fn ml_predict_tx_split_matches_c_nn_eval() {
    c::ref_init();

    // (a) TX_4X4 has no NN -> -1.
    assert!(TX_SPLIT_NN[0].is_none());
    assert_eq!(ml_predict_tx_split(&[0i16; 64], 8, 0, 0, 0), -1);

    // (b) Weight-transcription spot-check (catches a wrong-array grab): the first
    // av1_tx_split_nn_weights_4x8_layer0 values (TX_4X8 = tx_size 5).
    let nn48 = TX_SPLIT_NN[5].as_ref().unwrap();
    assert_eq!(nn48.num_inputs, 8);
    assert_eq!(nn48.num_hidden, 16);
    assert_eq!(
        &nn48.w0[0..4],
        &[0.068650f32, -0.732073f32, -0.040361f32, 0.322550f32],
        "4x8 layer0 weight transcription"
    );

    let mut rng = Rng(0x7c3a_2026_0713_0009);
    let mut clamp_lo = 0usize;
    let mut clamp_hi = 0usize;
    let mut in_range = 0usize;

    for tx_size in 0..19usize {
        let Some(nn) = TX_SPLIT_NN[tx_size].as_ref() else {
            continue;
        };
        let (bw, bh) = (TXS_W[tx_size], TXS_H[tx_size]);
        // Flat layout for ref_nn_predict: weights = layer0 ++ layer1, bias =
        // layer0_bias ++ layer1_bias.
        let weights_flat: Vec<f32> = nn.w0.iter().chain(nn.w1.iter()).copied().collect();
        let bias_flat: Vec<f32> = nn.b0.iter().copied().chain(std::iter::once(nn.b1)).collect();

        for iter in 0..48 {
            // Amplitude spans small (in-range scores) to large (clamp) residuals.
            let amp = [2, 8, 30, 120, 400, 1500][iter % 6];
            let stride = bw + 8;
            let residual: Vec<i16> = (0..stride * (bh + 4))
                .map(|_| rng.range(-amp, amp + 1) as i16)
                .collect();

            let port = ml_predict_tx_split(&residual, stride, 0, 0, tx_size);

            // C eval: same features (port's get_mean_dev_features) through the
            // REAL av1_nn_predict, then the *10000 + clamp the port applies.
            let mut features = [0f32; 16];
            let ni = get_mean_dev_features(&residual, stride, bw, bh, &mut features);
            assert_eq!(ni, nn.num_inputs, "feature count tx_size={tx_size}");
            let raw = c::ref_nn_predict(
                &features[..ni],
                nn.num_inputs,
                1,
                &[nn.num_hidden as i32],
                &weights_flat,
                &bias_flat,
                true,
            )[0];
            let score_c = ((raw * 10000.0f32) as i32).clamp(-80000, 80000);

            assert_eq!(port, score_c, "tx_size={tx_size} iter={iter} amp={amp}");

            match port {
                -80000 => clamp_lo += 1,
                80000 => clamp_hi += 1,
                _ => in_range += 1,
            }
        }
    }

    // Non-vacuity: both clamp boundaries + the in-range regime are exercised.
    assert!(in_range > 50, "in-range scores: {in_range}");
    assert!(clamp_lo + clamp_hi > 5, "clamp hits: lo={clamp_lo} hi={clamp_hi}");
}
