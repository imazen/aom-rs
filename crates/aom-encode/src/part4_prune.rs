//! `av1_ml_prune_4_partition` (partition_strategy.c:1326-1523) -- the NN
//! that prunes `PARTITION_HORZ_4`/`VERT_4` candidates before the RD search
//! evaluates them. LIVE at speed 0 KEY (`part_sf.ml_prune_partition = 1`
//! unconditionally at the top of BOTH `set_allintra_speed_features_
//! framesize_independent` and `set_good_speed_features_framesize_
//! independent`, speed_features.c -- not gated by any `if (speed >= N)`).
//!
//! BOTH `ml_model_index` branches are modelled (`ml_model_index =
//! (ml_4_partition_search_level_index < 3)`, partition_strategy.c:1359;
//! the level is threaded in via `predict_4partition_prune`'s `level_index`
//! param = `min(speed, 3)`):
//!   - `== 1` ("hd_" weight set, `NEW_LABEL_SIZE=3`, mean/std-normalized
//!     features, softmax + search/not-search threshold tables) — levels
//!     0..2 (speed 0..2);
//!   - `== 0` (the "old" `av1_4_partition_nn_*` weight set, `LABEL_SIZE=4`,
//!     **UNnormalized** features — the `if (ml_model_index)` normalize is
//!     skipped — `int_score[i] = (int)(100 * score[i])`, `thresh =
//!     max_score - {500,500,200}` by bsize, then `av1_zero_array` + set
//!     from the label bits: bit0 → HORZ4, bit1 → VERT4) — level 3
//!     (speed >= 3). KB-7 root cause: this branch was unported (the port
//!     left the 4-way flags untouched at level >= 3), so at cpu-used=3/4
//!     the port evaluated HORZ_4/VERT_4 where C prunes them, flipping the
//!     cq12 4:2:0 SB-root NONE-vs-SPLIT partition near-ties.
//!
//! `ext_ml_model_decision_after_part_ab` (the external-partition-model
//! hook) requires `!frame_is_intra_only(cm)`, which is always false for our
//! all-KEY envelope, so it always returns `false` and the real NN below
//! always runs -- not modelled here (dead in this envelope).

use crate::part4_nn_weights as w;

/// `get_unsigned_bits` (common.h): `n > 0 ? get_msb(n) + 1 : 0` ==
/// `32 - n.leading_zeros()` for `n > 0`.
fn get_unsigned_bits(n: u32) -> u32 {
    if n == 0 { 0 } else { 32 - n.leading_zeros() }
}

/// `av1_nn_predict_c` (ml.c) specialized to this NN's fixed shape: exactly
/// 1 ReLU hidden layer (16/32/64 all have `num_hidden_layers == 1`, BOTH
/// model variants), then a linear (no-activation) output layer of `N`
/// nodes (`NEW_LABEL_SIZE` = 3 for the "hd_" model, `LABEL_SIZE` = 4 for
/// the level>=3 "old" model), then `av1_nn_output_prec_reduce` — BOTH
/// `av1_ml_prune_4_partition` call sites pass `reduce_prec = 1`
/// (partition_strategy.c:1475/1502), so the reduce is unconditional here.
fn nn_predict_1layer<const N: usize>(
    input: &[f32; w::FEATURE_SIZE],
    w0: &[f32],
    b0: &[f32],
    hidden: usize,
    w1: &[f32],
    b1: &[f32; N],
) -> [f32; N] {
    debug_assert_eq!(w0.len(), w::FEATURE_SIZE * hidden);
    debug_assert_eq!(b0.len(), hidden);
    debug_assert_eq!(w1.len(), hidden * N);
    // HIDDEN_32 (32) is the largest hidden layer among the 3 bsizes.
    let mut hbuf = [0f32; 32];
    for (node, hbuf_node) in hbuf.iter_mut().enumerate().take(hidden) {
        let mut val = b0[node];
        for i in 0..w::FEATURE_SIZE {
            val += w0[node * w::FEATURE_SIZE + i] * input[i];
        }
        *hbuf_node = val.max(0.0); // ReLU
    }
    let mut out = [0f32; N];
    for (node, out_node) in out.iter_mut().enumerate() {
        let mut val = b1[node];
        for (i, &hv) in hbuf.iter().enumerate().take(hidden) {
            val += w1[node * hidden + i] * hv;
        }
        *out_node = val;
    }
    // `av1_nn_output_prec_reduce` (ml.c:19): `(int)(output[i] * prec + 0.5)`
    // then `* inv_prec`. The C `+ 0.5` is a DOUBLE literal, so the f32
    // product promotes to f64 before the add and the `(int)` truncation;
    // `inv_prec = (float)(1.0 / 512)` is the exact power-of-two 1/512.
    const PREC: f32 = (1 << 9) as f32;
    const INV_PREC: f32 = 1.0 / PREC;
    for v in out.iter_mut() {
        *v = ((f64::from(*v * PREC) + 0.5) as i32) as f32 * INV_PREC;
    }
    out
}

/// `av1_nn_softmax` (ml.c): numerically-stable softmax (max-subtract, clamp
/// the shifted input to >= -10.0 before `expf` "to prevent FE_UNDERFLOW
/// errors" per the C comment).
fn softmax3(input: [f32; w::NEW_LABEL_SIZE]) -> [f32; w::NEW_LABEL_SIZE] {
    let max_input = input[0].max(input[1]).max(input[2]);
    let mut out = [0f32; w::NEW_LABEL_SIZE];
    let mut sum = 0f32;
    for i in 0..w::NEW_LABEL_SIZE {
        let normalized = (input[i] - max_input).max(-10.0);
        out[i] = normalized.exp();
        sum += out[i];
    }
    for x in out.iter_mut() {
        *x /= sum;
    }
    out
}

/// Per-bsize weight-table bundle (`convert_bsize_to_idx` restricted to the
/// 3 reachable 4-way square bsizes: 16x16/32x32/64x64).
struct Tables {
    /// `av1_partition4_search_thresh`/`not_search_thresh`'s `bsize_idx`
    /// (`0=128x128,1=64x64,2=32x32,3=16x16,4=8x8`).
    bsize_idx: usize,
    mean: &'static [f32; w::FEATURE_SIZE],
    std: &'static [f32; w::FEATURE_SIZE],
    hidden: usize,
    w0: &'static [f32],
    b0: &'static [f32],
    w1: &'static [f32],
    b1: &'static [f32; w::NEW_LABEL_SIZE],
    /// The `ml_model_index == 0` ("old", `LABEL_SIZE=4`) weight set used at
    /// level >= 3 (speed >= 3). Same hidden size as the hd_ set per the
    /// `av1_4_partition_nnconfig_*` declarations.
    old_w0: &'static [f32],
    old_b0: &'static [f32],
    old_w1: &'static [f32],
    old_b1: &'static [f32; w::LABEL_SIZE],
    /// `thresh = max_score - {500, 500, 200}` for 16x16/32x32/64x64
    /// (partition_strategy.c:1487-1492).
    old_thresh_sub: i32,
}

fn tables_for(bsize: usize) -> Option<Tables> {
    match bsize {
        6 => Some(Tables {
            bsize_idx: 3,
            mean: &w::MEAN_16,
            std: &w::STD_16,
            hidden: w::HIDDEN_16,
            w0: &w::W0_16,
            b0: &w::B0_16,
            w1: &w::W1_16,
            b1: &w::B1_16,
            old_w0: &w::OLD_W0_16,
            old_b0: &w::OLD_B0_16,
            old_w1: &w::OLD_W1_16,
            old_b1: &w::OLD_B1_16,
            old_thresh_sub: 500,
        }),
        9 => Some(Tables {
            bsize_idx: 2,
            mean: &w::MEAN_32,
            std: &w::STD_32,
            hidden: w::HIDDEN_32,
            w0: &w::W0_32,
            b0: &w::B0_32,
            w1: &w::W1_32,
            b1: &w::B1_32,
            old_w0: &w::OLD_W0_32,
            old_b0: &w::OLD_B0_32,
            old_w1: &w::OLD_W1_32,
            old_b1: &w::OLD_B1_32,
            old_thresh_sub: 500,
        }),
        12 => Some(Tables {
            bsize_idx: 1,
            mean: &w::MEAN_64,
            std: &w::STD_64,
            hidden: w::HIDDEN_64,
            w0: &w::W0_64,
            b0: &w::B0_64,
            w1: &w::W1_64,
            b1: &w::B1_64,
            old_w0: &w::OLD_W0_64,
            old_b0: &w::OLD_B0_64,
            old_w1: &w::OLD_W1_64,
            old_b1: &w::OLD_B1_64,
            old_thresh_sub: 200,
        }),
        _ => None,
    }
}

/// `av1_ml_prune_4_partition`: updates `(horz4_allowed, vert4_allowed)` in
/// place (matching the C's `int *partition_horz4_allowed`/`vert4_allowed`
/// out-params, which the function only ever conditionally OVERWRITES --
/// never reads its own prior value -- so `horz4_in`/`vert4_in` are only
/// used as the return value when a branch returns early without touching
/// them, exactly mirroring the C's early `return;` leaving the caller's
/// prior `part4_search_allowed[..]` untouched).
///
/// `rect_part_rd`/`split_rd` are `[HORZ,VERT][0,1]` / `[4]` in the SAME
/// shape [`crate::partition_pick::rd_pick_partition_real`] already
/// threads. `res_idx`: 0 = lowres (<480p), 1 = midres (>=480p, <720p),
/// 2 = hdres (>=720p) -- `AOMMIN(cm->width, cm->height)` bucketed exactly
/// as `is_480p_or_larger + is_720p_or_larger`.
#[allow(clippy::too_many_arguments)]
pub fn predict_4partition_prune(
    bsize: usize,
    part_ctx: i32,
    best_rd: i64,
    rect_part_rd: [[i64; 2]; 2],
    split_rd: [i64; 4],
    pb_source_variance: u32,
    horz4_source_var: [u32; 4],
    vert4_source_var: [u32; 4],
    res_idx: usize,
    level_index: i32,
    horz4_in: bool,
    vert4_in: bool,
) -> (bool, bool) {
    // `if (best_rd >= 1000000000) return;` -- leaves *_allowed untouched.
    if best_rd >= 1_000_000_000 {
        return (horz4_in, vert4_in);
    }
    let Some(t) = tables_for(bsize) else {
        // `convert_bsize_to_idx` returns -1 -> `if (bsize_idx < 0) return;`.
        return (horz4_in, vert4_in);
    };

    // Feature engineering (partition_strategy.c:1378-1456).
    let mut features = [0f32; w::FEATURE_SIZE];
    let mut fi = 0usize;
    features[fi] = part_ctx as f32;
    fi += 1;
    features[fi] = get_unsigned_bits(pb_source_variance) as f32;
    fi += 1;

    let rdcost = best_rd.min(i64::from(i32::MAX)) as i32;
    let mut sub_block_rdcost = [0i32; 8];
    let mut ri = 0usize;
    for &v in &rect_part_rd[0] {
        // HORZ
        if v > 0 && v < 1_000_000_000 {
            sub_block_rdcost[ri] = v as i32;
        }
        ri += 1;
    }
    for &v in &rect_part_rd[1] {
        // VERT
        if v > 0 && v < 1_000_000_000 {
            sub_block_rdcost[ri] = v as i32;
        }
        ri += 1;
    }
    for &v in &split_rd {
        if v > 0 && v < 1_000_000_000 {
            sub_block_rdcost[ri] = v as i32;
        }
        ri += 1;
    }
    for &sb_rd in &sub_block_rdcost {
        let mut rd_ratio = 1.0f32;
        if sb_rd > 0 && sb_rd < rdcost {
            rd_ratio = sb_rd as f32 / rdcost as f32;
        }
        features[fi] = rd_ratio;
        fi += 1;
    }

    let denom = (pb_source_variance + 1) as f32;
    let (low_b, high_b) = (0.1f32, 10.0f32);
    for &v in &horz4_source_var {
        let var_ratio = ((v + 1) as f32 / denom).clamp(low_b, high_b);
        features[fi] = var_ratio;
        fi += 1;
    }
    for &v in &vert4_source_var {
        let var_ratio = ((v + 1) as f32 / denom).clamp(low_b, high_b);
        features[fi] = var_ratio;
        fi += 1;
    }
    debug_assert_eq!(fi, w::FEATURE_SIZE);

    // `ml_model_index = (ml_4_partition_search_level_index < 3)`
    // (partition_strategy.c:1359): the level is 0 at speed 0, min(speed, 3)
    // above (speed_features.c:210/237/271), so level 3 == speed >= 3.
    let ml_model_index = level_index < 3;

    // `if (ml_model_index)` — the feature normalize is SKIPPED for the old
    // (level >= 3) model (partition_strategy.c:1459-1463).
    if ml_model_index {
        #[allow(clippy::needless_range_loop)] // 3 parallel arrays, indices mirror the C loop
        for i in 0..w::FEATURE_SIZE {
            features[i] = (features[i] - t.mean[i]) / t.std[i];
        }
    }

    if !ml_model_index {
        // `ml_model_index == 0` (partition_strategy.c:1472-1497): the old
        // LABEL_SIZE=4 model on UNnormalized features; `int_score[i] =
        // (int)(100 * score[i])`; `thresh = max_score - {500,500,200}`;
        // `av1_zero_array(part4_allowed)` then set from the label bits
        // (bit0 → HORZ4, bit1 → VERT4) of every label reaching thresh —
        // an OVERWRITE of the incoming flags, exactly like the C.
        let score =
            nn_predict_1layer::<{ w::LABEL_SIZE }>(&features, t.old_w0, t.old_b0, t.hidden, t.old_w1, t.old_b1);
        let mut int_score = [0i32; w::LABEL_SIZE];
        let mut max_score = -1000i32;
        for (s, is) in score.iter().zip(int_score.iter_mut()) {
            *is = (100.0 * s) as i32;
            max_score = max_score.max(*is);
        }
        let thresh = max_score - t.old_thresh_sub;
        let mut horz4 = false;
        let mut vert4 = false;
        for (i, &is) in int_score.iter().enumerate() {
            if is >= thresh {
                // C: `(i >> 0) & 1` -> HORZ4, `(i >> 1) & 1` -> VERT4.
                if i & 1 == 1 {
                    horz4 = true;
                }
                if (i >> 1) & 1 == 1 {
                    vert4 = true;
                }
            }
        }
        return (horz4, vert4);
    }

    let score = nn_predict_1layer::<{ w::NEW_LABEL_SIZE }>(&features, t.w0, t.b0, t.hidden, t.w1, t.b1);
    let probs = softmax3(score);

    let thresh_idx = (level_index as usize * 3 + res_idx) * 5 + t.bsize_idx;
    let search_thresh = w::SEARCH_THRESH[thresh_idx];
    let not_search_thresh = w::NOT_SEARCH_THRESH[thresh_idx];

    let mut horz4 = horz4_in;
    let mut vert4 = vert4_in;
    // `for (i = 1; i < NEW_LABELS; ++i)`: i==1 -> HORZ4, i==2 -> VERT4.
    if probs[1] >= search_thresh {
        horz4 = true;
    }
    if probs[1] < not_search_thresh {
        horz4 = false;
    }
    if probs[2] >= search_thresh {
        vert4 = true;
    }
    if probs[2] < not_search_thresh {
        vert4 = false;
    }
    (horz4, vert4)
}
