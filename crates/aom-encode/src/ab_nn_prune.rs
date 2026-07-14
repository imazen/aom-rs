//! `ml_prune_ab_partition` (partition_strategy.c:1223-1320) -- the NN that
//! decides which of `HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B` to actually search,
//! called from `av1_prune_ab_partitions` when `ml_prune_partition` is set
//! (LIVE at speed 0 KEY, same sf as the 4-way NN's own gate -- already
//! established in `part4_prune.rs`'s module docs) AND both rect types
//! (`partition_rect_allowed[HORZ]`/`[VERT]`) were allowed.
//!
//! Structurally simpler than the 4-way NN (`part4_prune.rs`): no mean/std
//! feature normalization (verified absent from `partition_model_weights.h`
//! -- see `xtask/transcribe_ab_nn.py`'s docstring), a single hidden layer of
//! uniform size 64 across all 4 reachable bsizes (16/32/64/128 -- BLOCK_8X8
//! has no weight table and returns early, matching `allow_ab_partition_
//! search`'s own `bsize > BLOCK_8X8` gate making that case unreachable in
//! practice), and a 16-way bitmask output decode (no softmax) instead of the
//! 4-way NN's `NEW_LABEL_SIZE=3` softmax + threshold-table lookup.
//!
//! `ext_ml_model_decision_after_rect` (the external-partition-model hook)
//! requires `!frame_is_intra_only(cm)`, always false in this all-KEY
//! envelope -- not modelled (dead), matching every other `ext_*` hook this
//! port has already established dead for the same reason.

use crate::ab_nn_weights as w;

/// `av1_nn_predict_c` (ml.c) specialized to this NN's fixed shape: 1 ReLU
/// hidden layer (uniform 64 nodes across all 4 bsizes), then a linear
/// (no-activation) output layer of 16 nodes.
fn nn_predict_1layer(
    input: &[f32; w::FEATURE_SIZE],
    w0: &[f32; w::FEATURE_SIZE * w::HIDDEN],
    b0: &[f32; w::HIDDEN],
    w1: &[f32; w::HIDDEN * w::LABEL_SIZE],
    b1: &[f32; w::LABEL_SIZE],
) -> [f32; w::LABEL_SIZE] {
    let mut hbuf = [0f32; w::HIDDEN];
    for (node, hbuf_node) in hbuf.iter_mut().enumerate() {
        let mut val = b0[node];
        for i in 0..w::FEATURE_SIZE {
            val += w0[node * w::FEATURE_SIZE + i] * input[i];
        }
        *hbuf_node = val.max(0.0); // ReLU
    }
    let mut out = [0f32; w::LABEL_SIZE];
    for (node, out_node) in out.iter_mut().enumerate() {
        let mut val = b1[node];
        for (i, &hv) in hbuf.iter().enumerate() {
            val += w1[node * w::HIDDEN + i] * hv;
        }
        *out_node = val;
    }
    out
}

struct Tables {
    w0: &'static [f32; w::FEATURE_SIZE * w::HIDDEN],
    b0: &'static [f32; w::HIDDEN],
    w1: &'static [f32; w::HIDDEN * w::LABEL_SIZE],
    b1: &'static [f32; w::LABEL_SIZE],
}

/// `bsize` -> weight bundle. `None` for `BLOCK_8X8` and anything smaller
/// (`nn_config = NULL` in the C's own switch) -- matches `allow_ab_partition_
/// search`'s `bsize > BLOCK_8X8` gate making this case unreachable in
/// practice; kept as a real (not `unreachable!()`) `None` arm for fidelity
/// with the C's own no-op early return.
fn tables_for(bsize: usize) -> Option<Tables> {
    match bsize {
        6 => Some(Tables {
            // BLOCK_16X16
            w0: &w::W0_16,
            b0: &w::B0_16,
            w1: &w::W1_16,
            b1: &w::B1_16,
        }),
        9 => Some(Tables {
            // BLOCK_32X32
            w0: &w::W0_32,
            b0: &w::B0_32,
            w1: &w::W1_32,
            b1: &w::B1_32,
        }),
        12 => Some(Tables {
            // BLOCK_64X64
            w0: &w::W0_64,
            b0: &w::B0_64,
            w1: &w::W1_64,
            b1: &w::B1_64,
        }),
        15 => Some(Tables {
            // BLOCK_128X128
            w0: &w::W0_128,
            b0: &w::B0_128,
            w1: &w::W1_128,
            b1: &w::B1_128,
        }),
        _ => None,
    }
}

/// `ml_prune_ab_partition`: returns the (possibly narrowed-from-all-true, or
/// left UNCHANGED on an early-return) `[horz_a, horz_b, vert_a, vert_b]`
/// allowed flags. `part_ctx` = `pc_tree->partitioning` (this port's
/// `pc_tree_partitioning`, already threaded since the 4-way chunk);
/// `var_ctx` = `get_unsigned_bits(x->source_variance)` -- the CALLER passes
/// the raw `x->source_variance` (gotcha #1, module docs on
/// `leaf_pick_sb_modes`), NOT `pb_source_variance`, matching the C's own
/// `ml_prune_ab_partition(cpi, pc_tree->partitioning,
/// get_unsigned_bits(x->source_variance), ...)` call site
/// (partition_strategy.c:2002-2004) despite its own comment flagging this as
/// imprecise. `rect_part_rd`/`split_rd` are `[HORZ,VERT][0,1]` / `[4]`, the
/// SAME shape [`crate::partition_pick::rd_pick_partition_real`] already
/// threads for the 4-way NN.
#[allow(clippy::too_many_arguments)]
pub fn predict_ab_partition_prune(
    bsize: usize,
    part_ctx: i32,
    x_source_variance: u32,
    best_rd: i64,
    rect_part_rd: [[i64; 2]; 2],
    split_rd: [i64; 4],
    allowed_in: [bool; 4],
) -> [bool; 4] {
    // `if (bsize < BLOCK_8X8 || best_rd >= 1000000000) return;` -- leaves
    // ab_partitions_allowed untouched.
    const BLK_1D: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    if BLK_1D[bsize] < BLK_1D[3] || best_rd >= 1_000_000_000 {
        return allowed_in;
    }
    let Some(t) = tables_for(bsize) else {
        // nn_config == NULL (BLOCK_8X8 or unreachable smaller sizes).
        return allowed_in;
    };

    // Feature engineering (partition_strategy.c:1244-1277).
    let mut features = [0f32; w::FEATURE_SIZE];
    let mut fi = 0usize;
    features[fi] = part_ctx as f32;
    fi += 1;
    features[fi] = get_unsigned_bits(x_source_variance) as f32;
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
    debug_assert_eq!(fi, w::FEATURE_SIZE);

    // No mean/std normalization for this NN (module docs).
    let score = nn_predict_1layer(&features, t.w0, t.b0, t.w1, t.b1);
    let mut int_score = [0i32; w::LABEL_SIZE];
    let mut max_score = -1000i32;
    for (i, &s) in score.iter().enumerate() {
        int_score[i] = (100.0 * s) as i32;
        max_score = max_score.max(int_score[i]);
    }

    let mut thresh = max_score;
    match bsize {
        6 => thresh -= 150, // BLOCK_16X16
        9 => thresh -= 100, // BLOCK_32X32
        _ => {}
    }

    // av1_zero_array(ab_partitions_allowed, NUM_AB_PARTS) -- an authoritative
    // OVERWRITE past this point, not a further AND-narrowing (module docs).
    let mut out = [false; 4];
    for (i, &s) in int_score.iter().enumerate() {
        if s >= thresh {
            if i & 1 != 0 {
                out[0] = true; // HORZ_A
            }
            if (i >> 1) & 1 != 0 {
                out[1] = true; // HORZ_B
            }
            if (i >> 2) & 1 != 0 {
                out[2] = true; // VERT_A
            }
            if (i >> 3) & 1 != 0 {
                out[3] = true; // VERT_B
            }
        }
    }
    out
}

/// `get_unsigned_bits` (common.h): `n > 0 ? get_msb(n) + 1 : 0` ==
/// `32 - n.leading_zeros()` for `n > 0`.
fn get_unsigned_bits(n: u32) -> u32 {
    if n == 0 { 0 } else { 32 - n.leading_zeros() }
}
