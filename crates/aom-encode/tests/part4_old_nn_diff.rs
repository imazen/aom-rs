//! Differential: `predict_4partition_prune`'s **level>=3 OLD-model branch**
//! (`av1_ml_prune_4_partition` with `ml_model_index == 0`,
//! partition_strategy.c:1472-1497 — the KB-7 root) vs a C oracle built from
//! the REAL `av1_nn_predict_c` (`ref_nn_predict`) on the SAME transcribed
//! `OLD_*` weight tables plus a verbatim transcription of the C's
//! feature-engineering + int-score decision, over randomized raw inputs
//! (rd values / variances / part_ctx) for all three 4-way bsizes.
//!
//! What this pins beyond the byte gates: the OLD-branch scoring is fed
//! through the REAL C inference (bit-identical `av1_nn_predict_c` per
//! `cnn_partition_nn_diff`), so a transcription slip in the OLD weight
//! tables, the skipped-normalize, the `(int)(100 * score)` cast, the
//! max-minus-{500,500,200} threshold, or the label-bit decode would diverge
//! here even if the current gate grids happened not to reach it.

use aom_encode::part4_nn_weights as w;
use aom_encode::part4_prune::predict_4partition_prune;
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
}

fn get_unsigned_bits(n: u32) -> u32 {
    if n == 0 { 0 } else { 32 - n.leading_zeros() }
}

/// The C feature engineering (partition_strategy.c:1378-1456), verbatim —
/// UNnormalized (the `if (ml_model_index)` normalize is skipped at level 3).
#[allow(clippy::too_many_arguments)]
fn c_features(
    part_ctx: i32,
    best_rd: i64,
    horz_rd: [i64; 2],
    vert_rd: [i64; 2],
    split_rd: [i64; 4],
    pb_source_variance: u32,
    horz4_var: [u32; 4],
    vert4_var: [u32; 4],
) -> [f32; 18] {
    let mut features = [0f32; 18];
    let mut fi = 0usize;
    features[fi] = part_ctx as f32;
    fi += 1;
    features[fi] = get_unsigned_bits(pb_source_variance) as f32;
    fi += 1;
    let rdcost = best_rd.min(i64::from(i32::MAX)) as i32;
    let mut sub = [0i32; 8];
    let mut ri = 0usize;
    for &v in horz_rd.iter().chain(vert_rd.iter()).chain(split_rd.iter()) {
        if v > 0 && v < 1_000_000_000 {
            sub[ri.min(7)] = v as i32;
        }
        ri += 1;
        if ri == 8 {
            break;
        }
    }
    for &s in &sub {
        let mut rd_ratio = 1.0f32;
        if s > 0 && s < rdcost {
            rd_ratio = s as f32 / rdcost as f32;
        }
        features[fi] = rd_ratio;
        fi += 1;
    }
    let denom = (pb_source_variance + 1) as f32;
    for &v in horz4_var.iter().chain(vert4_var.iter()) {
        features[fi] = (((v + 1) as f32) / denom).clamp(0.1, 10.0);
        fi += 1;
    }
    assert_eq!(fi, 18);
    features
}

/// The C decision (partition_strategy.c:1472-1497) on REAL-C scores.
fn c_decision(scores: &[f32], thresh_sub: i32) -> (bool, bool) {
    let mut int_score = [0i32; 4];
    let mut max_score = -1000i32;
    for (i, &s) in scores.iter().enumerate() {
        int_score[i] = (100.0 * s) as i32;
        max_score = max_score.max(int_score[i]);
    }
    let thresh = max_score - thresh_sub;
    let (mut h4, mut v4) = (false, false);
    for (i, &is) in int_score.iter().enumerate() {
        if is >= thresh {
            if i & 1 == 1 {
                h4 = true;
            }
            if (i >> 1) & 1 == 1 {
                v4 = true;
            }
        }
    }
    (h4, v4)
}

#[test]
fn part4_old_nn_decision_matches_c() {
    c::ref_init();
    let mut rng = XorShift(0x0dd4_a11c_0de7_0001 ^ 0x9e37_79b9_7f4a_7c15);
    let cases: [(usize, usize, i32, &[f32], &[f32]); 3] = [
        (6, w::HIDDEN_16, 500, &w::OLD_W0_16[..], &w::OLD_B0_16[..]),
        (9, w::HIDDEN_32, 500, &w::OLD_W0_32[..], &w::OLD_B0_32[..]),
        (12, w::HIDDEN_64, 200, &w::OLD_W0_64[..], &w::OLD_B0_64[..]),
    ];
    let mut n = 0usize;
    let mut pruned_both = 0usize;
    for iter in 0..4000 {
        let (bsize, hidden, thresh_sub, w0, b0) = cases[iter % 3];
        let (w1, b1): (&[f32], &[f32]) = match bsize {
            6 => (&w::OLD_W1_16[..], &w::OLD_B1_16[..]),
            9 => (&w::OLD_W1_32[..], &w::OLD_B1_32[..]),
            _ => (&w::OLD_W1_64[..], &w::OLD_B1_64[..]),
        };
        // Random raw inputs in realistic magnitudes (rd up to ~999M — the
        // <1e9 validity bound — variances up to ~2^20, part_ctx 0..30).
        let part_ctx = rng.range_u(0, 30) as i32;
        let best_rd = rng.range_u(1_000, 999_000_000) as i64;
        let rd_or_zero = |rng: &mut XorShift| -> i64 {
            [0, rng.range_u(1, 1_200_000_000) as i64][rng.range_u(0, 1) as usize]
        };
        let horz_rd = [rd_or_zero(&mut rng), rd_or_zero(&mut rng)];
        let vert_rd = [rd_or_zero(&mut rng), rd_or_zero(&mut rng)];
        let split_rd = [
            rd_or_zero(&mut rng),
            rd_or_zero(&mut rng),
            rd_or_zero(&mut rng),
            rd_or_zero(&mut rng),
        ];
        let pbvar = rng.range_u(0, 1 << 20) as u32;
        let mut h4v = [0u32; 4];
        let mut v4v = [0u32; 4];
        for i in 0..4 {
            h4v[i] = rng.range_u(0, 1 << 20) as u32;
            v4v[i] = rng.range_u(0, 1 << 20) as u32;
        }

        // Port: the full old-model branch at level 3.
        let got = predict_4partition_prune(
            bsize,
            part_ctx,
            best_rd,
            [horz_rd, vert_rd],
            split_rd,
            pbvar,
            h4v,
            v4v,
            (iter / 3) % 3, // res_idx — must be inert on the old branch
            3,
            true,
            true,
        );

        // C oracle: verbatim features -> REAL av1_nn_predict_c -> C decision.
        let features = c_features(
            part_ctx, best_rd, horz_rd, vert_rd, split_rd, pbvar, h4v, v4v,
        );
        // reduce_prec = TRUE — both av1_ml_prune_4_partition call sites pass
        // `av1_nn_predict(features, nn_config, 1, score)`
        // (partition_strategy.c:1475/1502).
        let scores = c::ref_nn_predict(
            &features,
            18,
            4,
            &[hidden as i32],
            &[w0, w1].concat(),
            &[b0, b1].concat(),
            true,
        );
        let want = c_decision(&scores, thresh_sub);
        assert_eq!(
            got, want,
            "bsize={bsize} iter={iter}: port {got:?} != C {want:?} \
             (scores={scores:?} part_ctx={part_ctx} best_rd={best_rd} pbvar={pbvar})"
        );
        if !want.0 && !want.1 {
            pruned_both += 1;
        }
        n += 1;
    }
    // Sanity: the decision space is actually exercised in both directions.
    assert!(
        pruned_both > 0 && pruned_both < n,
        "degenerate coverage: {pruned_both}/{n}"
    );
    eprintln!(
        "part4_old_nn_decision_matches_c: {n} cases decision-identical ({pruned_both} pruned-both)"
    );
}
