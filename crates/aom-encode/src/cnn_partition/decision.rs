//! Port of the top-level `intra_mode_cnn_partition` decision
//! (`av1/encoder/partition_strategy.c`) — everything except the
//! partition-search side-effects: run the CNN on the 64×64 window, normalise
//! `log_q`, assemble the per-bsize DNN features, run the branch DNN, and turn
//! `logits[0]` vs the res-tier thresholds into the four prune flags.
//!
//! The CNN + DNN sub-engines are each already proven bit-exact against C
//! ([`super::cnn`], [`super::nn`]); this module adds the `log_q` term, the
//! feature assembly (`branch_*` spatial slicing via the `quad_to_linear` maps),
//! the threshold selection, and the decision — all diffed against
//! `av1/encoder/partition_strategy.c` via `aom_sys_ref::ref_intra_cnn_partition_decision`.

use super::{cnn, nn, weights as w};
use aom_dsp::quant::av1_dc_quant_qtx;

/// The four prune effects `intra_mode_cnn_partition` applies to the partition
/// search state. `none_disallowed` = `partition_none_allowed = 0` (only when
/// `logits[0] > split_thresh` AND `level != 1`); `do_square_split` +
/// `rect_disabled` = `av1_disable_rect_partitions` (when `logits[0] >
/// split_thresh`); `square_split_disabled` = `av1_disable_square_split_partition`
/// (when `logits[0] < no_split_thresh`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CnnPruneDecision {
    pub none_disallowed: bool,
    pub do_square_split: bool,
    pub rect_disabled: bool,
    pub square_split_disabled: bool,
}

impl CnnPruneDecision {
    /// True when the CNN constrains the search at all (any flag set).
    pub fn prunes(&self) -> bool {
        self.none_disallowed
            || self.do_square_split
            || self.rect_disabled
            || self.square_split_disabled
    }
}

/// `log_q` feature (partition_strategy.c:193-198): `dc_q =
/// av1_dc_quant_QTX(qindex,0,bd) >> (bd-8)`, then
/// `log_q = (log1pf(dc_q*dc_q / 256) - mean) / std`.
fn compute_log_q(qindex: i32, bd: i32) -> f32 {
    let dc_q = i32::from(av1_dc_quant_qtx(qindex, 0, bd as u8)) >> (bd - 8);
    // (float)(dc_q*dc_q) / 256.0f, then log1pf.
    let log_q = (f64_to_f32_div(dc_q * dc_q)).ln_1p();
    (log_q - w::MEAN[0]) / w::STD[0]
}

/// `(float)(dc_q*dc_q) / 256.0f` — the integer product widened to f32 then
/// divided by 256 in f32 (matching the C literal `256.0f`).
#[inline]
fn f64_to_f32_div(sq: i32) -> f32 {
    (sq as f32) / 256.0
}

/// Branch DNN weight bundle for `bsize_idx` (dnn_configs[bsize_idx]: 1→branch_0
/// .. 4→branch_3). Returns `(w0, b0, w1, b1, wlogits, blogits)`.
#[allow(clippy::type_complexity)]
fn branch_dnn(
    bsize_idx: i32,
) -> (
    &'static [f32],
    &'static [f32],
    &'static [f32],
    &'static [f32],
    &'static [f32],
    &'static [f32],
) {
    match bsize_idx {
        1 => (
            &w::BRANCH_0_DNN_LAYER_0_KERNEL,
            &w::BRANCH_0_DNN_LAYER_0_BIAS,
            &w::BRANCH_0_DNN_LAYER_1_KERNEL,
            &w::BRANCH_0_DNN_LAYER_1_BIAS,
            &w::BRANCH_0_LOGITS_KERNEL,
            &w::BRANCH_0_LOGITS_BIAS,
        ),
        2 => (
            &w::BRANCH_1_DNN_LAYER_0_KERNEL,
            &w::BRANCH_1_DNN_LAYER_0_BIAS,
            &w::BRANCH_1_DNN_LAYER_1_KERNEL,
            &w::BRANCH_1_DNN_LAYER_1_BIAS,
            &w::BRANCH_1_LOGITS_KERNEL,
            &w::BRANCH_1_LOGITS_BIAS,
        ),
        3 => (
            &w::BRANCH_2_DNN_LAYER_0_KERNEL,
            &w::BRANCH_2_DNN_LAYER_0_BIAS,
            &w::BRANCH_2_DNN_LAYER_1_KERNEL,
            &w::BRANCH_2_DNN_LAYER_1_BIAS,
            &w::BRANCH_2_LOGITS_KERNEL,
            &w::BRANCH_2_LOGITS_BIAS,
        ),
        4 => (
            &w::BRANCH_3_DNN_LAYER_0_KERNEL,
            &w::BRANCH_3_DNN_LAYER_0_BIAS,
            &w::BRANCH_3_DNN_LAYER_1_KERNEL,
            &w::BRANCH_3_DNN_LAYER_1_BIAS,
            &w::BRANCH_3_LOGITS_KERNEL,
            &w::BRANCH_3_LOGITS_BIAS,
        ),
        _ => unreachable!("intra-CNN bsize_idx in 1..=4"),
    }
}

/// Assemble the DNN input features from the CNN multi-out buffer for `bsize_idx`
/// / `quad_tree_idx`, appending `log_q` last. Returns the feature count (37 for
/// 64×64, 25 for 32×32, 25 for 16×16, 41 for 8×8). Verbatim transcription of the
/// per-bsize blocks in `intra_mode_cnn_partition` (branch spatial strides
/// 2×2 / 4×4 / 8×8, quad_to_linear spatial maps).
fn assemble_features(
    cnn_buffer: &[f32; cnn::CNN_OUT_BUF_SIZE],
    bsize_idx: i32,
    quad_tree_idx: i32,
    log_q: f32,
    out: &mut [f32; 100],
) -> usize {
    // Branch bases in the multi-out buffer (see cnn::branch_region).
    let branch_0 = &cnn_buffer[0..];
    let branch_1 = &cnn_buffer[20..];
    let branch_2 = &cnn_buffer[36..];
    let branch_3 = &cnn_buffer[356..];
    let mut f = 0usize;
    match bsize_idx {
        1 => {
            // BLOCK_64X64
            for ch in 0..20 {
                out[f] = branch_0[ch];
                f += 1;
            }
            let spa = 2 * 2;
            for lin in 0..spa {
                for ch in 0..4 {
                    out[f] = branch_1[lin + ch * spa];
                    f += 1;
                }
            }
        }
        2 => {
            // BLOCK_32X32
            for idx in 0..20 {
                out[f] = branch_0[idx];
                f += 1;
            }
            let cur_lin = w::QUAD_TO_LINEAR_1[(quad_tree_idx - 1) as usize] as usize;
            let spa = 2 * 2;
            for ch in 0..4 {
                out[f] = branch_1[cur_lin + ch * spa];
                f += 1;
            }
        }
        3 => {
            // BLOCK_16X16
            let prev_quad = (quad_tree_idx - 1) / 4;
            let prev_lin = w::QUAD_TO_LINEAR_1[(prev_quad - 1) as usize] as usize;
            let prev_spa = 2 * 2;
            for ch in 0..4 {
                out[f] = branch_1[prev_lin + ch * prev_spa];
                f += 1;
            }
            let cur_lin = w::QUAD_TO_LINEAR_2[(quad_tree_idx - 5) as usize] as usize;
            let spa = 4 * 4;
            for ch in 0..20 {
                out[f] = branch_2[cur_lin + ch * spa];
                f += 1;
            }
        }
        4 => {
            // BLOCK_8X8
            let prev_quad = (quad_tree_idx - 1) / 4;
            let prev_lin = w::QUAD_TO_LINEAR_2[(prev_quad - 5) as usize] as usize;
            let prev_spa = 4 * 4;
            for ch in 0..20 {
                out[f] = branch_2[prev_lin + ch * prev_spa];
                f += 1;
            }
            let cur_lin = w::QUAD_TO_LINEAR_3[(quad_tree_idx - 21) as usize] as usize;
            let spa = 8 * 8;
            for ch in 0..20 {
                out[f] = branch_3[cur_lin + ch * spa];
                f += 1;
            }
        }
        _ => unreachable!("intra-CNN bsize_idx in 1..=4"),
    }
    out[f] = log_q;
    f += 1;
    f
}

/// Run the intra-CNN partition-prune decision for one block. `win` is the
/// parent 64×64's 65×65 luma window (replicated top/left border); `bsize_idx`
/// is `convert_bsize_to_idx` (1=64×64 .. 4=8×8); `quad_tree_idx` is the block's
/// position in the quad-tree; `level` is `intra_cnn_based_part_prune_level`
/// (1 or 2). Returns `(logits, decision)`.
pub fn predict_decision(
    win: &[u8],
    qindex: i32,
    bd: i32,
    frame_w: i32,
    frame_h: i32,
    bsize_idx: i32,
    quad_tree_idx: i32,
    level: i32,
) -> ([f32; 4], CnnPruneDecision) {
    let cnn_buffer = cnn::cnn_predict(win);
    let log_q = compute_log_q(qindex, bd);

    let mut features = [0.0f32; 100];
    let nf = assemble_features(&cnn_buffer, bsize_idx, quad_tree_idx, log_q, &mut features);

    let (w0, b0, w1, b1, wl, bl) = branch_dnn(bsize_idx);
    let mut logits = [0.0f32; 4];
    // num_outputs = BRANCH_*_NUM_LOGITS = 1; reduce_prec = 1 (as C calls it).
    nn::nn_predict(
        &features[..nf],
        &[16, 24],
        &[w0, w1, wl],
        &[b0, b1, bl],
        1,
        true,
        &mut logits,
    );

    // Res-tier thresholds (partition_strategy.c:311-329).
    let mind = frame_w.min(frame_h);
    let bi = bsize_idx as usize;
    let (split_thresh, no_split_thresh) = if mind >= 720 {
        (w::SPLIT_THRESH_HDRES[bi], w::NO_SPLIT_THRESH_HDRES[bi])
    } else if mind >= 480 {
        (w::SPLIT_THRESH_MIDRES[bi], w::NO_SPLIT_THRESH_MIDRES[bi])
    } else {
        (w::SPLIT_THRESH_LOWRES[bi], w::NO_SPLIT_THRESH_LOWRES[bi])
    };

    let mut d = CnnPruneDecision::default();
    if logits[0] > split_thresh {
        if level != 1 {
            d.none_disallowed = true;
        }
        d.do_square_split = true;
        d.rect_disabled = true;
    }
    if logits[0] < no_split_thresh {
        d.square_split_disabled = true;
    }
    (logits, d)
}
