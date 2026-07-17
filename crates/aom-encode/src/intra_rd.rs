//! Block-level intra-mode RD evaluation — the first slice of the speed-0
//! KEY-frame mode-search decision layer: for one coding block, evaluate a
//! candidate intra mode end-to-end (predict -> subtract -> forward transform +
//! quantize + trellis -> rate + transform-domain distortion -> RDCOST) and
//! pick the minimum-RD candidate from a caller-supplied list.
//!
//! Every step composes an individually C-validated piece:
//! [`aom_intra::predict_intra_high`], [`aom_dist::highbd_subtract_block`],
//! [`crate::xform_quant_optimize`], [`aom_txb::cost_coeffs_txb`],
//! [`aom_txb::get_tx_type_cost`], [`crate::mode_costs::intra_mode_info_cost_y`],
//! [`crate::dist_block_tx_domain`], [`crate::rd::rdcost`] — and the
//! *composition* is differentially validated against the identical chain of C
//! reference steps (`intra_rd_pick_diff.rs`).
//!
//! SCOPE — this is a composition primitive, deliberately narrower than
//! libaom's `av1_rd_pick_intra_sby_mode`:
//! - single-transform-block coding blocks only (`bsize` dims == `tx_size`
//!   dims; no tx-size search / tx partition),
//! - the candidate list and its order are the caller's (none of the C search's
//!   ordering, hog/variance pruning, early termination, or adaptive
//!   angle-delta refinement),
//! - one caller-fixed `tx_type` per evaluation (no tx-type search),
//! - transform-domain distortion only (no reconstruction-domain switch, no
//!   skip-vs-coded RD alternative),
//! - plane 0 (luma), KEY-frame Y mode rate (`y_mode_costs` via the above/left
//!   `intra_mode_context` pair), `palette_size[0] == 0`.

use crate::mode_costs::{IntraModeCosts, intra_mode_info_cost_y};
use crate::{
    BlockContext, OptimizeInputs, QuantKind, QuantParams, dist_block_tx_domain, rd,
    xform_quant_optimize,
};
use aom_dist::highbd_subtract_block;
use aom_entropy::partition::get_y_mode_ctx;
use aom_intra::predict_intra_high;
use aom_txb::{CoeffCostTables, TxTypeCosts, cost_coeffs_txb, get_tx_type_cost};

/// `ANGLE_STEP` (enums.h): degrees per signaled angle-delta step.
pub const ANGLE_STEP: i32 = 3;

/// Per-block prediction environment: the reconstructed neighbourhood the
/// predictor reads, the source pixels, geometry, and edge availability
/// (`intra_avail` outputs). `bsize` must have the same dimensions as
/// `tx_size` (single-txb scope).
pub struct IntraRdEnv<'a> {
    pub recon: &'a [u16],
    /// Index of the block's top-left pixel in `recon`.
    pub ref_off: usize,
    pub ref_stride: usize,
    pub src: &'a [u16],
    /// Index of the block's top-left pixel in `src`.
    pub src_off: usize,
    pub src_stride: usize,
    pub tx_size: usize,
    /// Block size (BLOCK_SIZE discriminant), dims equal to `tx_size`.
    pub bsize: usize,
    pub n_top_px: usize,
    pub n_topright_px: i32,
    pub n_left_px: usize,
    pub n_bottomleft_px: i32,
    pub disable_edge_filter: bool,
    pub filter_type: i32,
    pub bd: u8,
}

/// Rate inputs: the derived cost tables plus the frame/neighbour state that
/// selects the mode-signaling rate terms.
pub struct IntraRdRates<'a> {
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
    pub mode_costs: &'a IntraModeCosts,
    pub rdmult: i32,
    /// Above / left neighbour Y modes (`None` = unavailable -> `DC_PRED`),
    /// selecting the KEY-frame `y_mode_costs` context pair.
    pub above_mode: Option<i32>,
    pub left_mode: Option<i32>,
    pub try_palette: bool,
    pub palette_bsize_ctx: usize,
    pub palette_mode_ctx: usize,
    pub enable_filter_intra: bool,
    pub allow_intrabc: bool,
    pub reduced_tx_set: bool,
    pub lossless: bool,
}

/// One candidate: an intra mode with its angle delta (UNscaled, in
/// `[-MAX_ANGLE_DELTA, MAX_ANGLE_DELTA]`; scaled by [`ANGLE_STEP`] for
/// prediction) or a filter-intra variant (`mode` must be `DC_PRED`).
#[derive(Clone, Copy, Debug)]
pub struct IntraCandidate {
    pub mode: usize,
    pub angle_delta: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
}

/// One candidate's RD evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntraModeRd {
    /// Total rate: coefficient bits + tx_type signaling + Y mode-info signaling.
    pub rate: i32,
    /// Transform-domain distortion (`dist_block_tx_domain`).
    pub dist: i64,
    /// `RDCOST(rdmult, rate, dist)`.
    pub rd: i64,
    /// Post-trellis end-of-block.
    pub eob: u16,
}

/// Evaluate one intra candidate for one single-txb coding block: predict from
/// the reconstructed edges, subtract, transform + quantize + trellis
/// (`xform_quant_optimize`), then combine
/// `rate = cost_coeffs_txb + get_tx_type_cost + intra_mode_info_cost_y` and
/// `dist = dist_block_tx_domain` into one `RDCOST` — the RD shape of the C
/// mode loop (`this_rd = RDCOST(rdmult, this_rate, this_distortion)`).
#[allow(clippy::too_many_arguments)]
pub fn intra_mode_rd_eval(
    env: &IntraRdEnv,
    rates: &IntraRdRates,
    cand: &IntraCandidate,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    bctx: &BlockContext,
    opt: &OptimizeInputs,
) -> IntraModeRd {
    let w = crate::TX_W[env.tx_size];
    let h = crate::TX_H[env.tx_size];
    assert_eq!(
        crate::BLK_W[env.bsize],
        w,
        "bsize/tx_size width mismatch (single-txb scope)"
    );
    assert_eq!(
        crate::BLK_H[env.bsize],
        h,
        "bsize/tx_size height mismatch (single-txb scope)"
    );

    // Predict into a tight w-stride buffer (av1_predict_intra_block).
    let mut pred = vec![0u16; w * h];
    predict_intra_high(
        env.recon,
        env.ref_off,
        env.ref_stride,
        &mut pred,
        w,
        cand.mode,
        cand.angle_delta * ANGLE_STEP,
        cand.use_filter_intra,
        cand.filter_intra_mode,
        env.disable_edge_filter,
        env.filter_type,
        env.tx_size,
        env.n_top_px,
        env.n_topright_px,
        env.n_left_px,
        env.n_bottomleft_px,
        env.bd as i32,
    );

    // Residual = src - pred (aom_highbd_subtract_block).
    let mut residual = vec![0i16; w * h];
    highbd_subtract_block(
        h,
        w,
        &mut residual,
        w,
        &env.src[env.src_off..],
        env.src_stride,
        &pred,
        w,
    );

    // Forward transform + quantize + trellis (the speed-0 coefficient path).
    let r = xform_quant_optimize(&residual, env.tx_size, tx_type, kind, qp, bctx, opt);

    // Rate: post-trellis coefficient bits (av1_cost_coeffs_txb) + tx_type
    // signaling + Y mode-info signaling. The real av1_cost_coeffs_txb includes
    // get_tx_type_cost inside its eob>0 body but its eob==0 branch returns the
    // txb_skip cost ALONE (an all-zero txb signals no tx_type) — so the
    // tx_type term is gated on eob != 0.
    let coeff_rate = cost_coeffs_txb(
        &r.qcoeff,
        r.eob as usize,
        env.tx_size,
        tx_type,
        r.txb_skip_ctx,
        r.dc_sign_ctx,
        rates.coeff_costs,
    );
    let tx_type_rate = if r.eob != 0 {
        get_tx_type_cost(
            rates.tx_type_costs,
            0,
            env.tx_size,
            tx_type,
            false,
            rates.reduced_tx_set,
            rates.lossless,
            cand.use_filter_intra,
            cand.filter_intra_mode,
            cand.mode,
        )
    } else {
        0
    };
    let (above_ctx, left_ctx) = get_y_mode_ctx(rates.above_mode, rates.left_mode);
    let mode_cost = rates.mode_costs.y_mode_costs[above_ctx][left_ctx][cand.mode];
    let mode_rate = intra_mode_info_cost_y(
        rates.mode_costs,
        mode_cost,
        cand.mode,
        env.bsize,
        cand.angle_delta,
        cand.use_filter_intra,
        cand.filter_intra_mode,
        false, // use_intrabc: an intrabc block would not run the intra mode loop
        rates.try_palette,
        rates.palette_bsize_ctx,
        rates.palette_mode_ctx,
        rates.enable_filter_intra,
        rates.allow_intrabc,
        false,
        0,
    );
    let rate = coeff_rate + tx_type_rate + mode_rate;

    // Transform-domain distortion, then one RDCOST over the summed rate.
    let (dist, _sse) = dist_block_tx_domain(&r.coeff, &r.dqcoeff, env.tx_size, env.bd);
    let rd = rd::rdcost(rates.rdmult, rate, dist);

    IntraModeRd {
        rate,
        dist,
        rd,
        eob: r.eob,
    }
}

/// Evaluate every candidate and return `(argmin_index, per-candidate evals)`.
/// Ties keep the earliest candidate (strict `<` update, as the C loop's
/// `this_rd < best_rd`). The candidate order is the caller's — this does NOT
/// reproduce libaom's search order or pruning.
#[allow(clippy::too_many_arguments)]
pub fn pick_intra_mode_rd(
    env: &IntraRdEnv,
    rates: &IntraRdRates,
    candidates: &[IntraCandidate],
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    bctx: &BlockContext,
    opt: &OptimizeInputs,
) -> (usize, Vec<IntraModeRd>) {
    assert!(!candidates.is_empty());
    let evals: Vec<IntraModeRd> = candidates
        .iter()
        .map(|cand| intra_mode_rd_eval(env, rates, cand, tx_type, kind, qp, bctx, opt))
        .collect();
    let mut best = 0usize;
    let mut best_rd = i64::MAX;
    for (i, e) in evals.iter().enumerate() {
        if e.rd < best_rd {
            best_rd = e.rd;
            best = i;
        }
    }
    (best, evals)
}

// ---------------------------------------------------------------------------
// Candidate enumeration + speed-0 gating of av1_rd_pick_intra_sby_mode
// (av1/encoder/intra_mode_search.c) — the loop-head fidelity layer.
// ---------------------------------------------------------------------------

/// `INTRA_MODE_END` / `INTRA_MODES` (enums.h): 13 luma intra modes.
pub const INTRA_MODES: usize = 13;
/// `MAX_ANGLE_DELTA` (enums.h).
pub const MAX_ANGLE_DELTA: i32 = 3;
/// `LUMA_MODE_COUNT` (enums.h): `13 + 8 directional modes * 6 nonzero deltas`.
pub const LUMA_MODE_COUNT: usize = 61;

/// `intra_rd_search_mode_order` (intra_mode_search.c): the evaluation order of
/// the 13 modes at delta 0 — DC, H, V, SMOOTH, PAETH, SMOOTH_V, SMOOTH_H,
/// D135, D203, D157, D67, D113, D45 (as `PREDICTION_MODE` values).
pub const INTRA_RD_SEARCH_MODE_ORDER: [usize; INTRA_MODES] =
    [0, 2, 1, 9, 12, 10, 11, 4, 7, 6, 8, 5, 3];

/// `luma_delta_angles_order` (intra_mode_search.c): even deltas first, used
/// when `prune_luma_odd_delta_angles_in_intra` reorders the delta sweep.
pub const LUMA_DELTA_ANGLES_ORDER: [i32; 6] = [-2, 2, -3, -1, 1, 3];

/// `set_y_mode_and_delta_angle` (intra_mode_search.c, exported): map a loop
/// index `0..LUMA_MODE_COUNT` to the `(mode, angle_delta)` it evaluates.
/// Indices `< INTRA_MODES` walk [`INTRA_RD_SEARCH_MODE_ORDER`] at delta 0;
/// the rest sweep V..D67 (mode values 1..=8) x six nonzero deltas
/// (-3..-1, 1..3, or [`LUMA_DELTA_ANGLES_ORDER`] when `reorder_delta_angle_eval`).
pub fn set_y_mode_and_delta_angle(mode_idx: usize, reorder_delta_angle_eval: bool) -> (usize, i32) {
    assert!(mode_idx < LUMA_MODE_COUNT);
    if mode_idx < INTRA_MODES {
        (INTRA_RD_SEARCH_MODE_ORDER[mode_idx], 0)
    } else {
        let mode = (mode_idx - INTRA_MODES) / (MAX_ANGLE_DELTA as usize * 2) + 1; // + V_PRED
        let delta_angle_eval_idx = (mode_idx - INTRA_MODES) % (MAX_ANGLE_DELTA as usize * 2);
        let delta = if reorder_delta_angle_eval {
            LUMA_DELTA_ANGLES_ORDER[delta_angle_eval_idx]
        } else if delta_angle_eval_idx < 3 {
            delta_angle_eval_idx as i32 - 3
        } else {
            delta_angle_eval_idx as i32 - 2
        };
        (mode, delta)
    }
}

/// `av1_is_diagonal_mode` (reconintra.h): D45..D67 (mode values 3..=8).
#[inline]
pub fn is_diagonal_mode(mode: usize) -> bool {
    (3..=8).contains(&mode)
}

/// `max_txsize_lookup[BLOCK_SIZES_ALL]` (common_data.h): largest SQUARE
/// tx size for a block size (TX_4X4=0 .. TX_64X64=4) — the
/// `intra_y_mode_mask` index.
pub const MAX_TXSIZE_LOOKUP: [usize; 22] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0, 1, 1, 2, 2,
];

/// The gating inputs of the `av1_rd_pick_intra_sby_mode` candidate loop —
/// `IntraModeCfg` tool flags (all default `true` on the aomenc CLI), the
/// `intra_sf` members the loop reads, and the per-search pruning state.
///
/// Speed-0 all-intra values (speed_features.c; defaults at
/// `init_intra_mode_sf` + `set_allintra_speed_features_framesize_independent`
/// speed-0 section), each named:
/// - `disable_smooth_intra = 0` (default; never set at speed 0)
/// - `prune_filter_intra_level = 0` (default)
/// - `intra_y_mode_mask[..] = INTRA_ALL` for every tx size (default)
/// - `prune_luma_odd_delta_angles_in_intra = 0` (default)
/// - `intra_pruning_with_hog = 1` (set for allintra speed 0, thresh -1.2):
///   `directional_mode_skip_mask` is that prune's OUTPUT, an input here
///   (all-false when HOG keeps everything).
/// - `use_mb_mode_cache`: modelled OFF (MB mode cache is populated only by
///   superblock-level re-search paths, not the plain speed-0 walk).
#[derive(Clone, Debug)]
pub struct IntraSbyGates {
    /// `intra_mode_cfg.enable_diagonal_intra` (CLI default on).
    pub enable_diagonal_intra: bool,
    /// `intra_mode_cfg.enable_directional_intra`.
    pub enable_directional_intra: bool,
    /// `intra_mode_cfg.enable_smooth_intra`.
    pub enable_smooth_intra: bool,
    /// `intra_mode_cfg.enable_paeth_intra`.
    pub enable_paeth_intra: bool,
    /// `intra_mode_cfg.enable_angle_delta`.
    pub enable_angle_delta: bool,
    /// `intra_sf.disable_smooth_intra`.
    pub disable_smooth_intra: bool,
    /// `intra_sf.prune_filter_intra_level`.
    pub prune_filter_intra_level: i32,
    /// `intra_sf.intra_y_mode_mask[max_txsize_lookup[bsize]]` (bit per mode).
    pub intra_y_mode_mask: [u16; 5],
    /// `directional_mode_skip_mask` — the HOG prune output (index = mode).
    pub directional_mode_skip_mask: [bool; INTRA_MODES],
    /// `intra_sf.prune_luma_odd_delta_angles_in_intra`.
    pub prune_luma_odd_delta_angles_in_intra: bool,
}

impl IntraSbyGates {
    /// The speed-0 all-intra configuration (see struct docs), with the HOG
    /// skip mask supplied by the caller.
    pub fn speed0(directional_mode_skip_mask: [bool; INTRA_MODES]) -> Self {
        IntraSbyGates {
            enable_diagonal_intra: true,
            enable_directional_intra: true,
            enable_smooth_intra: true,
            enable_paeth_intra: true,
            enable_angle_delta: true,
            disable_smooth_intra: false,
            prune_filter_intra_level: 0,
            intra_y_mode_mask: [0x1fff; 5], // INTRA_ALL: all 13 mode bits
            directional_mode_skip_mask,
            prune_luma_odd_delta_angles_in_intra: false,
        }
    }

    /// The static skip chain of the candidate loop (intra_mode_search.c
    /// 1555-1594), for a candidate `(mode, delta)` on `bsize`: `true` =
    /// evaluate, `false` = `continue`. Excludes the model-RD prune and the
    /// odd-delta RD prune (dynamic; see
    /// [`prune_luma_odd_delta_angles_using_rd_cost`]).
    pub fn visits(&self, mode: usize, luma_delta_angle: i32, bsize: usize) -> bool {
        use aom_entropy::partition::{is_directional_mode, use_angle_delta};
        let is_directional = is_directional_mode(mode as i32);
        if is_diagonal_mode(mode) && !self.enable_diagonal_intra {
            return false;
        }
        if is_directional && !self.enable_directional_intra {
            return false;
        }
        // SMOOTH_V_PRED = 10, SMOOTH_H_PRED = 11, SMOOTH_PRED = 9, PAETH = 12.
        if (!self.enable_smooth_intra || self.disable_smooth_intra) && (mode == 11 || mode == 10) {
            return false;
        }
        if !self.enable_smooth_intra && mode == 9 {
            return false;
        }
        if self.disable_smooth_intra && self.prune_filter_intra_level == 0 && mode == 9 {
            return false;
        }
        if !self.enable_paeth_intra && mode == 12 {
            return false;
        }
        if is_directional && self.directional_mode_skip_mask[mode] {
            return false;
        }
        if is_directional
            && !(use_angle_delta(bsize) && self.enable_angle_delta)
            && luma_delta_angle != 0
        {
            return false;
        }
        (self.intra_y_mode_mask[MAX_TXSIZE_LOOKUP[bsize]] & (1 << mode)) != 0
    }

    /// The full visit sequence for one block: every `(mode, angle_delta)` the
    /// C loop evaluates (before its dynamic model-RD / odd-delta-RD prunes),
    /// in exact order.
    pub fn visit_sequence(&self, bsize: usize) -> Vec<(usize, i32)> {
        (0..LUMA_MODE_COUNT)
            .map(|idx| set_y_mode_and_delta_angle(idx, self.prune_luma_odd_delta_angles_in_intra))
            .filter(|&(mode, delta)| self.visits(mode, delta, bsize))
            .collect()
    }
}

/// `SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY` (intra_mode_search.c): per-mode RD
/// bookkeeping over deltas -4..=4 (delta `d` at index `d + MAX_ANGLE_DELTA + 1`;
/// indices 0 and 8 stay `INT64_MAX`).
pub const SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY: usize = 9;

/// `prune_luma_odd_delta_angles_using_rd_cost` (intra_mode_search.c): prune an
/// odd delta angle when both even-delta neighbours' recorded RD costs exceed
/// `best_rd + best_rd/8`. `intra_modes_rd_cost` is this mode's delta-indexed
/// RD array (see [`SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY`]). At speed 0 the
/// controlling sf is 0, so this never prunes.
pub fn prune_luma_odd_delta_angles_using_rd_cost(
    mode: usize,
    luma_delta_angle: i32,
    intra_modes_rd_cost: &[i64; SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY],
    best_rd: i64,
    prune_luma_odd_delta_angles_in_intra: bool,
) -> bool {
    if !prune_luma_odd_delta_angles_in_intra
        || !aom_entropy::partition::is_directional_mode(mode as i32)
        || (luma_delta_angle.abs() & 1) == 0
        || best_rd == i64::MAX
    {
        return false;
    }
    let rd_thresh = best_rd + (best_rd >> 3);
    intra_modes_rd_cost[(luma_delta_angle + MAX_ANGLE_DELTA) as usize] > rd_thresh
        && intra_modes_rd_cost[(luma_delta_angle + MAX_ANGLE_DELTA + 2) as usize] > rd_thresh
}

// ---------------------------------------------------------------------------
// Model-RD pruning (intra_mode_search.c) — the Hadamard-SATD gate that runs
// before each candidate's full tx search in the av1_rd_pick_intra_sby_mode
// loop. The model cost itself is [`crate::tx_search::intra_model_rd_y`].
// ---------------------------------------------------------------------------

/// `TOP_INTRA_MODEL_COUNT` (speed_features.h): the `top_intra_model_rd[]`
/// array length in the mode loop.
pub const TOP_INTRA_MODEL_COUNT: usize = 4;

/// `get_model_rd_index_for_pruning` (intra_mode_search.c): which
/// `top_intra_model_rd` slot `prune_intra_y_mode` compares against.
/// Neighbour modes: `None` = unavailable (the C's guarded
/// `xd->left_mbmi->mode` reads).
///
/// Speed-0 all-intra values (speed_features.c): `top_intra_model_count_allowed
/// = TOP_INTRA_MODEL_COUNT` (=4) and `adapt_top_model_rd_count_using_neighbors
/// = 0 (both `init_intra_mode_sf` defaults; the allintra path lowers them only
/// at speed >= 1 / speed >= 6) — so at speed 0 this is always `4 - 1 = 3`.
pub fn get_model_rd_index_for_pruning(
    cur_mode: usize,
    qindex: i32,
    top_intra_model_count_allowed: i32,
    adapt_top_model_rd_count_using_neighbors: bool,
    left_mode: Option<usize>,
    above_mode: Option<usize>,
) -> i32 {
    if !adapt_top_model_rd_count_using_neighbors {
        return top_intra_model_count_allowed - 1;
    }
    let mut model_rd_index_for_pruning = top_intra_model_count_allowed - 1;
    let is_left_mode_neq_cur_mode = left_mode.is_some_and(|m| m != cur_mode);
    let is_above_mode_neq_cur_mode = above_mode.is_some_and(|m| m != cur_mode);
    // qidx 0..=127: reduce when EITHER available neighbour mode differs;
    // qidx 128..=255: reduce only when BOTH differ.
    let reduce = if qindex <= 127 {
        is_left_mode_neq_cur_mode || is_above_mode_neq_cur_mode
    } else {
        is_left_mode_neq_cur_mode && is_above_mode_neq_cur_mode
    };
    if reduce {
        model_rd_index_for_pruning = (model_rd_index_for_pruning - 1).max(0);
    }
    model_rd_index_for_pruning
}

/// `prune_intra_y_mode` (intra_mode_search.c): sorted-insert `this_model_rd`
/// into the top-N model RDs, then prune when it exceeds
/// `1.00 * top[model_rd_index_for_pruning]` or `1.50 * best_model_rd` (both
/// DOUBLE comparisons — the i64 operands convert to f64 exactly for every
/// reachable SATD magnitude, and the reference build carries no FMA, so plain
/// Rust f64 mul + compare replicates the C bit-for-bit); otherwise lower
/// `best_model_rd`. Mutates both accumulators exactly as the C does.
pub fn prune_intra_y_mode(
    this_model_rd: i64,
    best_model_rd: &mut i64,
    top_intra_model_rd: &mut [i64],
    max_model_cnt_allowed: usize,
    model_rd_index_for_pruning: usize,
) -> bool {
    const THRESH_BEST: f64 = 1.50;
    const THRESH_TOP: f64 = 1.00;
    for i in 0..max_model_cnt_allowed {
        if this_model_rd < top_intra_model_rd[i] {
            for j in (i + 1..max_model_cnt_allowed).rev() {
                top_intra_model_rd[j] = top_intra_model_rd[j - 1];
            }
            top_intra_model_rd[i] = this_model_rd;
            break;
        }
    }
    if top_intra_model_rd[model_rd_index_for_pruning] != i64::MAX
        && (this_model_rd as f64)
            > THRESH_TOP * (top_intra_model_rd[model_rd_index_for_pruning] as f64)
    {
        return true;
    }
    if this_model_rd != i64::MAX && (this_model_rd as f64) > THRESH_BEST * (*best_model_rd as f64) {
        return true;
    }
    if this_model_rd < *best_model_rd {
        *best_model_rd = this_model_rd;
    }
    false
}

// ---------------------------------------------------------------------------
// intra_rd_variance_factor (intra_mode_search.c) — the ALLINTRA visual-quality
// RD scale applied to each candidate's this_rd in the mode loop.
// ---------------------------------------------------------------------------

/// `MI_SIZE` / `MI_SIZE_LOG2` (enums.h).
pub const MI_SIZE: usize = 4;

/// `mi_size_wide` / `mi_size_high` `[BLOCK_SIZES_ALL]` (common_data.h).
const MI_W_ALL: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H_ALL: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// `Block4x4VarInfo` (block.h): the per-4x4 source-variance cache — one entry
/// per mi position in the superblock, initialized `var = -1` /
/// `log_var = -1.0` per SB (`init_src_var_info_of_4x4_sub_blocks`, which runs
/// exactly when the variance factor is active: ALLINTRA +
/// `INTRA_RD_VAR_THRESH(speed) > 0`). The cache persists across every
/// candidate and coding block of the SB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block4x4VarInfo {
    pub var: i32,
    pub log_var: f64,
}

impl Block4x4VarInfo {
    /// One initialized (invalid) entry.
    pub fn init() -> Self {
        Block4x4VarInfo {
            var: -1,
            log_var: -1.0,
        }
    }
    /// A fresh per-superblock cache for `sb_size`.
    pub fn sb_cache(sb_size: usize) -> Vec<Block4x4VarInfo> {
        vec![Block4x4VarInfo::init(); MI_W_ALL[sb_size] * MI_H_ALL[sb_size]]
    }
}

/// `av1_calc_normalized_variance` for one 4x4 sub-block: the
/// `fn_ptr[BLOCK_4X4].vf` variance against an all-zero reference (= 16x the
/// raw per-pixel variance, "normalized" by the /16.0 in the log1p below).
/// fn_ptr resolution by stream depth: `aom_variance4x4` over the u8 planes
/// for 8-bit streams; `aom_highbd_<bd>_variance4x4` over the u16 planes for
/// bd > 8 (both individually C-validated in aom-dist).
pub(crate) fn calc_normalized_variance_4x4(buf: &[u16], off: usize, stride: usize, bd: u8) -> i32 {
    if bd > 8 {
        const ZEROS16: [u16; 4] = [0; 4];
        aom_dist::highbd_variance(&buf[off..], stride, &ZEROS16, 0, 4, 4, bd).0 as i32
    } else {
        // The production 8-bit encoder reads u8 planes; the strided window
        // holds the same 16 values, so a tight copy is kernel-identical.
        let mut w8 = [0u8; 16];
        for r in 0..4 {
            for c in 0..4 {
                debug_assert!(buf[off + r * stride + c] <= 255);
                w8[r * 4 + c] = buf[off + r * stride + c] as u8;
            }
        }
        const ZEROS8: [u8; 4] = [0; 4];
        aom_dist::variance(&w8, 4, &ZEROS8, 0, 4, 4).0 as i32
    }
}

/// The pixel-plane / geometry inputs of [`intra_rd_variance_factor`].
/// `mb_to_right_edge` / `mb_to_bottom_edge` are the MACROBLOCKD 1/8-pel edge
/// fields (negative = the block overhangs the frame; the overhang is clipped
/// out of the variance walk).
pub struct VarFactorInputs<'a> {
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    pub recon: &'a [u16],
    pub ref_off: usize,
    pub ref_stride: usize,
    pub bsize: usize,
    pub sb_size: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub mb_to_right_edge: i32,
    pub mb_to_bottom_edge: i32,
    pub bd: u8,
}

/// `intra_rd_variance_factor` (+ `compute_avg_log_variance`),
/// intra_mode_search.c: the ALLINTRA-mode visual-quality RD scale in
/// `[1.0, 3.0]` from how the block's reconstructed variance tracks its source
/// variance. `INTRA_RD_VAR_THRESH(speed) = 1.0 - 0.25 * speed` — active
/// (positive) only for speeds 0..=3; at speed 0 the threshold is 1.0.
///
/// Per 4x4 sub-block: `log1p(var/16.0)` of source (cached in `cache`, the
/// per-SB [`Block4x4VarInfo`] array) and of the CURRENT recon plane content —
/// which, in the mode loop, is whatever `av1_pick_uniform_tx_size_type_yrd`
/// left there: the reconstruction of the LAST tx size the depth sweep
/// evaluated (the C never re-runs the winner; loop-order-sensitive state).
/// Averages accumulate in the C's exact row-major order; all arithmetic is
/// f64 with no FMA (matching the reference build), and `f64::ln_1p` resolves
/// to the same libm `log1p` the oracle calls.
pub fn intra_rd_variance_factor(
    speed: i32,
    p: &VarFactorInputs,
    cache: &mut [Block4x4VarInfo],
) -> f64 {
    let threshold = 1.0 - (0.25 * f64::from(speed)); // INTRA_RD_VAR_THRESH
    if threshold <= 0.0 {
        return 1.0;
    }

    let mut variance_rd_factor = 1.0f64;
    let mut avg_log_src_variance = 0.0f64;
    let mut avg_log_recon_variance = 0.0f64;

    // compute_avg_log_variance.
    let mi_row_in_sb = (p.mi_row as usize) & (MI_H_ALL[p.sb_size] - 1);
    let mi_col_in_sb = (p.mi_col as usize) & (MI_W_ALL[p.sb_size] - 1);
    let right_overflow = if p.mb_to_right_edge < 0 {
        ((-p.mb_to_right_edge) >> 3) as usize
    } else {
        0
    };
    let bottom_overflow = if p.mb_to_bottom_edge < 0 {
        ((-p.mb_to_bottom_edge) >> 3) as usize
    } else {
        0
    };
    let bw = MI_SIZE * MI_W_ALL[p.bsize] - right_overflow;
    let bh = MI_SIZE * MI_H_ALL[p.bsize] - bottom_overflow;

    let mut i = 0usize;
    while i < bh {
        let r = mi_row_in_sb + (i >> 2); // MI_SIZE_LOG2
        let mut j = 0usize;
        while j < bw {
            let c = mi_col_in_sb + (j >> 2);
            let mi_offset = r * MI_W_ALL[p.sb_size] + c;
            let info = &mut cache[mi_offset];
            let log_src_var;
            if info.var < 0 {
                let src_var = calc_normalized_variance_4x4(
                    p.src,
                    p.src_off + i * p.src_stride + j,
                    p.src_stride,
                    p.bd,
                );
                info.var = src_var;
                log_src_var = (f64::from(src_var) / 16.0).ln_1p();
                info.log_var = log_src_var;
            } else if info.log_var < 0.0 {
                log_src_var = (f64::from(info.var) / 16.0).ln_1p();
                info.log_var = log_src_var;
            } else {
                log_src_var = info.log_var;
            }
            avg_log_src_variance += log_src_var;

            let recon_var = calc_normalized_variance_4x4(
                p.recon,
                p.ref_off + i * p.ref_stride + j,
                p.ref_stride,
                p.bd,
            );
            avg_log_recon_variance += (f64::from(recon_var) / 16.0).ln_1p();
            j += MI_SIZE;
        }
        i += MI_SIZE;
    }

    let blocks = ((bw * bh) / 16) as f64;
    avg_log_src_variance /= blocks;
    avg_log_recon_variance /= blocks;

    // intra_rd_variance_factor tail.
    avg_log_src_variance += 0.000001;
    avg_log_recon_variance += 0.000001;

    if avg_log_src_variance >= avg_log_recon_variance {
        let var_diff = avg_log_src_variance - avg_log_recon_variance;
        if var_diff > 0.5 && avg_log_recon_variance < threshold {
            variance_rd_factor = 1.0 + ((var_diff * 2.0) / avg_log_src_variance);
        }
    } else {
        let var_diff = avg_log_recon_variance - avg_log_src_variance;
        if var_diff > 0.5 && avg_log_src_variance < threshold {
            variance_rd_factor = 1.0 + (var_diff / (2.0 * avg_log_src_variance));
        }
    }

    // AOMMIN(3.0, v).
    if 3.0 < variance_rd_factor {
        3.0
    } else {
        variance_rd_factor
    }
}

/// The mode loop's ALLINTRA application:
/// `this_rd = (int64_t)(this_rd * factor)` — i64 -> f64 conversion, one f64
/// multiply, truncation toward zero (every reachable rd is far inside the
/// exact/in-range regime).
#[inline]
pub fn apply_variance_factor(rd: i64, factor: f64) -> i64 {
    (rd as f64 * factor) as i64
}

// ---------------------------------------------------------------------------
// av1_rd_pick_intra_sby_mode (intra_mode_search.c:1468) — the luma intra
// mode-loop assembly: candidate enumeration + gates + model-RD prune + full
// tx search + rate assembly + ALLINTRA variance scale + best tracking.
// ---------------------------------------------------------------------------

use crate::mode_costs::{block_signals_txsize, tx_size_cost};
use crate::tx_search::{
    TxTypeSearchPolicy, TxbWinner, TxfmYrdEnv, intra_model_rd_y,
    pick_uniform_tx_size_type_yrd_intra,
};

/// The configuration/state inputs of [`rd_pick_intra_sby_mode_y`] beyond the
/// per-block [`TxfmYrdEnv`]. Speed-0 all-intra values documented per field.
pub struct IntraSbySearchCfg<'a> {
    /// The candidate-loop gate set ([`IntraSbyGates::speed0`] at speed 0,
    /// with the HOG `directional_mode_skip_mask` output).
    pub gates: &'a IntraSbyGates,
    /// sf `intra_sf.top_intra_model_count_allowed`
    /// (= [`TOP_INTRA_MODEL_COUNT`] = 4 at speed 0).
    pub top_intra_model_count_allowed: i32,
    /// sf `intra_sf.adapt_top_model_rd_count_using_neighbors` (0 at speed 0).
    pub adapt_top_model_rd_count_using_neighbors: bool,
    /// Above / left neighbour Y modes (`None` = unavailable): select the
    /// KEY-frame `y_mode_costs[ctx][ctx]` row (`bmode_costs`) AND feed the
    /// adaptive model-prune index.
    pub above_mode: Option<i32>,
    pub left_mode: Option<i32>,
    /// `x->qindex` (the adaptive model-prune index input).
    pub qindex: i32,
    /// The mode-info signaling cost tables + `intra_mode_info_cost_y` gates.
    pub mode_costs: &'a IntraModeCosts,
    pub try_palette: bool,
    pub palette_bsize_ctx: usize,
    pub palette_mode_ctx: usize,
    /// `oxcf.intra_mode_cfg.enable_filter_intra` (CLI default on) — the
    /// filter-intra flag bit is costed on eligible bsizes.
    pub enable_filter_intra: bool,
    /// `avail_intrabc` for the mode-info cost (KEY frame with intrabc allowed).
    pub allow_intrabc: bool,
    /// The tx-search policy ([`TxTypeSearchPolicy::speed0_allintra`]).
    pub pol: &'a TxTypeSearchPolicy,
    /// `x->source_variance` (the depth-sweep regression prune input).
    pub source_variance: u32,
    /// `oxcf.txfm_cfg.enable_tx64` / `enable_rect_tx` (CLI defaults on).
    pub enable_tx64: bool,
    pub enable_rect_tx: bool,
    /// `cpi->oxcf.mode == ALLINTRA`: apply [`intra_rd_variance_factor`] to
    /// each candidate's rd (the `aomenc` all-intra target sets this).
    pub allintra: bool,
    /// `cpi->oxcf.speed` (the variance-factor threshold input).
    pub speed: i32,
    /// MACROBLOCKD `mb_to_right_edge` / `mb_to_bottom_edge` (1/8-pel; the
    /// variance factor's frame-edge clip inputs — non-negative for interior
    /// blocks, this port's txb-walk scope).
    pub mb_to_right_edge: i32,
    pub mb_to_bottom_edge: i32,
    /// The winner-mode two-pass bundle (KB-8): `None` = single-pass with
    /// `pol` + `USE_FULL_RD` (speed 0..=3, byte-identical to the pre-two-pass
    /// search); `Some` = MODE_EVAL first pass -> top-N winner stats ->
    /// WINNER_MODE_EVAL re-eval (speed>=4 all-intra).
    pub winner_mode: Option<&'a WinnerModeCfg<'a>>,
    /// The palette SEARCH configuration (`av1_rd_pick_palette_intra_sby` runs
    /// iff `oxcf.tool_cfg.enable_palette && av1_allow_palette(..)`,
    /// intra_mode_search.c:1485): `None` models `--enable-palette=0` (the
    /// standard-shim envelope every pre-existing gate runs under — the
    /// no-palette FLAG cost still applies via `try_palette` above). `Some`
    /// carries the size/colour cost tables + neighbour palette state + the
    /// palette speed-feature levels.
    pub palette: Option<PaletteModeCfg<'a>>,
}

/// The palette-search slice of [`IntraSbySearchCfg`] (present iff
/// `oxcf.tool_cfg.enable_palette`).
pub struct PaletteModeCfg<'a> {
    /// The palette size/colour-index cost tables
    /// ([`crate::mode_costs::fill_palette_costs`]).
    pub costs: &'a crate::mode_costs::PaletteCosts,
    /// Above/left neighbour palette state (`xd->above_mbmi`/`left_mbmi`
    /// palette projection) for `av1_get_palette_cache`.
    pub above: Option<aom_entropy::partition::PaletteNbrKf>,
    pub left: Option<aom_entropy::partition::PaletteNbrKf>,
    /// sf `intra_sf.prune_palette_search_level` (allintra: 0/1/2 at speeds
    /// 0/1+/3+).
    pub prune_palette_search_level: i32,
    /// sf `intra_sf.prune_luma_palette_size_search_level` (allintra: 1 at
    /// speed 0, 2 at speed>=1).
    pub prune_luma_palette_size_search_level: i32,
}

/// The winner-mode two-pass configuration (KB-8, speed>=4 all-intra): the
/// per-stage policy/method bundle `set_mode_eval_params` (rdopt_utils.h:546)
/// derives. `None` on [`IntraSbySearchCfg::winner_mode`] = single-pass
/// (speed 0..=3 — byte-identical to the pre-two-pass search).
pub struct WinnerModeCfg<'a> {
    /// The MODE_EVAL first-pass tx policy (`use_default_intra_tx_type=1`,
    /// MODE_EVAL coeff-opt/tx-domain columns) — used for the whole candidate
    /// loop AND the post-loop filter-intra search (C keeps MODE_EVAL params
    /// through both, intra_mode_search.c:1515..1685).
    pub mode_eval_pol: &'a TxTypeSearchPolicy,
    /// The WINNER_MODE_EVAL second-pass tx policy (full tx-type set,
    /// UINT_MAX coeff-opt columns ⇒ trellis always).
    pub winner_pol: &'a TxTypeSearchPolicy,
    /// First-pass tx-size method: `tx_size_search_methods[tx_size_search_level]
    /// [MODE_EVAL]` = `USE_LARGESTALL` at speed 4 (level 0, enable gate on).
    pub mode_eval_tx_size_method: usize,
    /// Second-pass method: `[WINNER_MODE_EVAL]` = `USE_FULL_RD` at speed 4.
    pub winner_tx_size_method: usize,
    /// `winner_mode_count_allowed[multi_winner_mode_type]` (rdopt_utils.h:236):
    /// 3 for MULTI_WINNER_MODE_DEFAULT (speed 4), 2 for FAST (speed 5), **1
    /// for MULTI_WINNER_MODE_OFF (speed >= 6)** — count 1 ⇔ OFF, selecting
    /// C's `else` re-eval arm (one `intra_block_yrd` on `best_mbmi`,
    /// intra_mode_search.c:1727-1737) instead of the winner-stats loop, with
    /// `store_winner_mode_stats` a hard no-op (rdopt_utils.h:688).
    pub max_winner_count: usize,
    /// sf `winner_mode_sf.prune_winner_mode_eval_level` — 0 through speed 5,
    /// 1 at allintra speed >= 6. Feeds [`bypass_winner_mode_processing`]
    /// inside the re-eval's `is_winner_mode_processing_enabled` gate.
    pub prune_winner_mode_eval_level: i32,
}

/// `bypass_winner_mode_processing` (rdopt_utils.h:403), the allintra-KEY
/// subset: level 1 skips the winner-mode re-eval for low-source-variance
/// blocks — `src_var_thresh = 64 - 48 * qindex / (MAXQ + 1)` (integer
/// division), bypass when `source_variance < src_var_thresh`. Levels 2/3
/// (txfm-skip based) and >= 4 are speed-7+ / realtime arms — modelled as
/// unreachable (no allintra speed <= 6 sets them). The mbmi-dependent inputs
/// (`use_txfm_skip`/`actual_txfm_skip`/`best_mode`) only reach the level-2+
/// arms and `have_newmv_in_inter_mode` (always false for intra), so they are
/// not parameters here.
pub fn bypass_winner_mode_processing(
    source_variance: u32,
    qindex: i32,
    prune_winner_mode_eval_level: i32,
) -> bool {
    match prune_winner_mode_eval_level {
        0 => false,
        1 => {
            let src_var_thresh = 64 - 48 * qindex / 256; // MAXQ + 1 = 256
            source_variance < src_var_thresh as u32
        }
        other => unreachable!("prune_winner_mode_eval_level {other} outside the allintra<=6 scope"),
    }
}

/// One stored winner (`WinnerModeStats`, block.h): the mbmi fields the
/// second pass restores before re-evaluating. `rd` is the FIRST-pass
/// (ALLINTRA-variance-factored) rd used for ranking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WinnerModeEntry {
    pub mode: usize,
    pub angle_delta: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    pub rd: i64,
    /// The palette-Y candidate state (size/colors/map) when this stored
    /// winner is a palette mode (store_winner_mode_stats' color_map copy,
    /// rdopt_utils.h:718-723); the second pass restores it before re-eval.
    pub palette_y: Option<crate::palette_search::PaletteYInfo>,
}

/// `store_winner_mode_stats` (rdopt_utils.h:679), intra-frame scope (the
/// inter rate bookkeeping and palette color-map copy are out of scope). The
/// `multi_winner_mode_type == OFF` early return (rdopt_utils.h:688) is the
/// `max_count == 1` arm — `winner_mode_count_allowed[OFF] = 1` and OFF is
/// the only type with count 1, so the C type check and the count check are
/// equivalent; at OFF (allintra speed >= 6) nothing is ever stored and the
/// re-eval runs on `best_mbmi` instead. Insertion sort by ascending `rd`
/// with C's exact semantics: the insert position is the FIRST index whose
/// stored rd is STRICTLY greater (ties keep earlier entries ahead); a full
/// list whose entries are all <= this_rd rejects the candidate; insertion
/// shifts the tail and drops the overflow.
pub fn store_winner_mode_stats(
    list: &mut Vec<WinnerModeEntry>,
    max_count: usize,
    entry: WinnerModeEntry,
) {
    if max_count == 1 {
        // MULTI_WINNER_MODE_OFF: "Mode stat is not required when multiwinner
        // mode processing is disabled" (rdopt_utils.h:688).
        return;
    }
    if entry.rd == i64::MAX {
        return;
    }
    debug_assert!(list.len() <= max_count);
    let mut mode_idx = 0usize;
    if !list.is_empty() {
        while mode_idx < list.len() {
            if list[mode_idx].rd > entry.rd {
                break;
            }
            mode_idx += 1;
        }
        if mode_idx == max_count {
            // No stored mode has higher rd cost than this_rd.
            return;
        }
    }
    list.insert(mode_idx, entry);
    list.truncate(max_count);
}

/// The winning candidate of one [`rd_pick_intra_sby_mode_y`] search — the
/// C loop's `best_mbmi` + output-pointer bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntraSbyBest {
    pub mode: usize,
    pub angle_delta: i32,
    /// `best_mbmi.tx_size` (the uniform winner size).
    pub tx_size: usize,
    /// The winner's per-txb (tx_type, eob, entropy ctx) — the
    /// `ctx->tx_type_map` snapshot (`av1_copy_array` on best update).
    pub winners: Vec<TxbWinner>,
    /// `*rate`: tokenonly + tx-size-adjusted + mode-info signaling.
    pub rate: i32,
    /// `*rate_tokenonly`: the tx-search rate MINUS the tx-size cost on
    /// signaling blocks (intra_mode_search.c:1624-1630 — tx_size is counted
    /// in the full rate, not the tokenonly rate, for intra blocks).
    pub rate_tokenonly: i32,
    /// `*distortion`.
    pub dist: i64,
    /// `*skippable` (always false for intra: block_rd_txfm hard-zeroes
    /// `skip_txfm` per txb, tx_search.c:3129).
    pub skippable: bool,
    /// The winning (post-variance-factor) rd — the C return value.
    pub best_rd: i64,
    /// `best_mbmi.filter_intra_mode_info`: set when the post-loop
    /// filter-intra search wins (mode is DC_PRED then).
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    /// `best_mbmi.palette_mode_info` (Y half) + the winning colour-index map
    /// (`best_palette_color_map`): set when the post-loop palette search
    /// (`av1_rd_pick_palette_intra_sby`) wins. `mode` is DC_PRED then.
    pub palette_y: Option<crate::palette_search::PaletteYInfo>,
}

/// The search outcome: the winner (`None` = no candidate beat `best_rd_in`,
/// the C's `!beat_best_rd => return INT64_MAX`) plus the per-(mode, delta)
/// rd bookkeeping table (loop-local in C; exposed for differential
/// visibility and the odd-delta prune's future higher-speed use).
pub struct IntraSbyOutcome {
    pub best: Option<IntraSbyBest>,
    pub intra_modes_rd_cost: [[i64; SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY]; INTRA_MODES],
}

/// `av1_rd_pick_intra_sby_mode` (intra_mode_search.c:1468) — the 61-candidate
/// luma mode loop at speed-0 all-intra scope. Per visit index:
/// [`set_y_mode_and_delta_angle`] -> the static gate chain
/// ([`IntraSbyGates::visits`]; `use_mb_mode_cache` modelled off) ->
/// [`prune_luma_odd_delta_angles_using_rd_cost`] (sf off at speed 0) ->
/// [`intra_model_rd_y`] at `min(TX_32X32, max_txsize_lookup[bsize])` (its
/// prediction writes into `recon` are C-faithful shared state) ->
/// [`prune_intra_y_mode`] -> [`pick_uniform_tx_size_type_yrd_intra`] with the
/// RUNNING `best_rd` as its early-exit threshold -> the tx-size-cost
/// subtraction into `rate_tokenonly` (signaling blocks; the C comment: intra
/// blocks always code tx_size in the full rate, not the tokenonly rate) ->
/// `this_rate = stats.rate + intra_mode_info_cost_y` -> `RDCOST` ->
/// [ALLINTRA: `this_rd = (int64_t)(this_rd * intra_rd_variance_factor(..))`,
/// reading the recon state the depth sweep left behind] ->
/// `intra_modes_rd_cost` bookkeeping -> strict `this_rd < best_rd` best
/// tracking (ties keep the FIRST winner).
///
/// `store_winner_mode_stats` is a hard no-op at speed 0
/// (`multi_winner_mode_type == MULTI_WINNER_MODE_OFF` returns immediately,
/// rdopt_utils.h:688). The post-loop palette search (`try_palette`) and
/// filter-intra search (`rd_pick_filter_intra_sby`) are NOT composed here yet
/// — both are live at speed 0 and remain missing pieces of the full C
/// function; the winner-mode tail IS a verified no-op at speed 0 for intra
/// (`is_winner_mode_processing_enabled` = 0: `fast_intra_tx_type_search`,
/// `enable_winner_mode_for_coeff_opt` and `enable_winner_mode_for_tx_size_srch`
/// are all 0 — rdopt_utils.h:444-476).
///
/// `env`'s candidate fields (`mode`, `angle_delta`) are overwritten per visit;
/// `use_filter_intra` is forced off (the C clears
/// `mbmi->filter_intra_mode_info.use_filter_intra` before the loop).
/// `var_cache` is the per-superblock [`Block4x4VarInfo`] array (shared across
/// candidates and across the SB's blocks).
pub fn rd_pick_intra_sby_mode_y(
    env: &mut TxfmYrdEnv,
    recon: &mut [u16],
    cfg: &IntraSbySearchCfg,
    var_cache: &mut [Block4x4VarInfo],
    best_rd_in: i64,
) -> IntraSbyOutcome {
    let bsize = env.bsize;
    env.use_filter_intra = false;
    env.filter_intra_mode = 0;

    let (above_ctx, left_ctx) = get_y_mode_ctx(cfg.above_mode, cfg.left_mode);
    let bmode_costs = &cfg.mode_costs.y_mode_costs[above_ctx][left_ctx];

    // Per-pass eval params (set_mode_eval_params, rdopt_utils.h:546): the
    // candidate loop + filter-intra search run under MODE_EVAL when the
    // winner-mode two-pass is configured, else the caller's (DEFAULT_EVAL)
    // policy. `tx_mode_search_type = select_tx_mode(cm, method)` couples the
    // tx-size-cost gating to the stage's method: USE_LARGESTALL ⇒
    // TX_MODE_LARGEST ⇒ NOT select (tx_size_cost = 0 during the first pass).
    let select_default = env.tx_mode_is_select;
    let (pass_pol, pass_method) = match cfg.winner_mode {
        Some(wm) => (wm.mode_eval_pol, wm.mode_eval_tx_size_method),
        // `--enable-tx-size-search=0` forces tx_size_search_level 3 →
        // USE_LARGESTALL even on the single-pass (speed 0..=3) arm
        // (speed_features.c:2726; tx_size_search_methods row 3).
        None if !cfg.pol.enable_tx_size_search => {
            (cfg.pol, crate::tx_search::USE_LARGESTALL)
        }
        None => (cfg.pol, crate::tx_search::USE_FULL_RD),
    };
    env.tx_mode_is_select = select_default && pass_method != crate::tx_search::USE_LARGESTALL;
    // zero_winner_mode_stats + x->winner_mode_count = 0 (:1520-1521).
    let mut winner_stats: Vec<WinnerModeEntry> = Vec::new();

    let mut best_rd = best_rd_in;
    let mut best: Option<IntraSbyBest> = None;
    let mut best_model_rd = i64::MAX;
    let mut top_intra_model_rd = [i64::MAX; TOP_INTRA_MODEL_COUNT];
    let mut intra_modes_rd_cost = [[i64::MAX; SIZE_OF_ANGLE_DELTA_RD_COST_ARRAY]; INTRA_MODES];

    // The model tx size: AOMMIN(TX_32X32, max_txsize_lookup[bsize]).
    let model_tx_size = MAX_TXSIZE_LOOKUP[bsize].min(3);

    for mode_idx in 0..LUMA_MODE_COUNT {
        let (mode, luma_delta_angle) =
            set_y_mode_and_delta_angle(mode_idx, cfg.gates.prune_luma_odd_delta_angles_in_intra);
        // The C mutates mbmi BEFORE the gate chain (set_y_mode_and_delta_angle
        // writes mode/angle, then the gates `continue`) — so the post-loop
        // stale mbmi carries index 60's (D67, +3) values into the
        // filter-intra tail. Mirror by setting env first.
        env.mode = mode;
        env.angle_delta = luma_delta_angle;
        if !cfg.gates.visits(mode, luma_delta_angle, bsize) {
            continue;
        }
        if prune_luma_odd_delta_angles_using_rd_cost(
            mode,
            luma_delta_angle,
            &intra_modes_rd_cost[mode],
            best_rd,
            cfg.gates.prune_luma_odd_delta_angles_in_intra,
        ) {
            continue;
        }

        // Model estimate + prune (the prediction walk mutates recon).
        let this_model_rd = intra_model_rd_y(env, recon, model_tx_size);
        let model_rd_index_for_pruning = get_model_rd_index_for_pruning(
            mode,
            cfg.qindex,
            cfg.top_intra_model_count_allowed,
            cfg.adapt_top_model_rd_count_using_neighbors,
            cfg.left_mode.map(|m| m as usize),
            cfg.above_mode.map(|m| m as usize),
        );
        if prune_intra_y_mode(
            this_model_rd,
            &mut best_model_rd,
            &mut top_intra_model_rd,
            cfg.top_intra_model_count_allowed as usize,
            model_rd_index_for_pruning as usize,
        ) {
            continue;
        }

        // The real prediction + tx search (the model prediction above was
        // only an estimate; av1_pick_uniform_tx_size_type_yrd redoes it
        // through the full txfm pipeline).
        let Some(choice) = pick_uniform_tx_size_type_yrd_intra(
            env,
            recon,
            best_rd,
            pass_pol,
            cfg.source_variance,
            cfg.enable_tx64,
            cfg.enable_rect_tx,
            pass_method,
            None,
        ) else {
            continue; // this_rate_tokenonly == INT_MAX
        };

        let mut this_rate_tokenonly = choice.stats.rate;
        if !env.lossless && block_signals_txsize(bsize) {
            // The tx-search rate includes the tx_size cost; for intra blocks
            // tx_size is accounted in the full rate, not the tokenonly rate.
            this_rate_tokenonly -= tx_size_cost(
                env.tx_size_costs,
                env.tx_mode_is_select,
                bsize,
                choice.best_tx_size,
                env.tx_size_ctx,
            );
        }
        let this_rate = choice.stats.rate
            + intra_mode_info_cost_y(
                cfg.mode_costs,
                bmode_costs[mode],
                mode,
                bsize,
                luma_delta_angle,
                false,
                0,
                false, // use_intrabc
                cfg.try_palette,
                cfg.palette_bsize_ctx,
                cfg.palette_mode_ctx,
                cfg.enable_filter_intra,
                cfg.allow_intrabc,
                false,
                0,
            );
        let this_distortion = choice.stats.dist;
        let mut this_rd = rd::rdcost(env.rdmult, this_rate, this_distortion);

        // Visual quality adjustment based on recon vs source variance.
        if cfg.allintra && this_rd != i64::MAX {
            let factor = intra_rd_variance_factor(
                cfg.speed,
                &VarFactorInputs {
                    src: env.src,
                    src_off: env.src_off,
                    src_stride: env.src_stride,
                    recon,
                    ref_off: env.ref_off,
                    ref_stride: env.ref_stride,
                    bsize,
                    sb_size: env.sb_size,
                    mi_row: env.mi_row,
                    mi_col: env.mi_col,
                    mb_to_right_edge: cfg.mb_to_right_edge,
                    mb_to_bottom_edge: cfg.mb_to_bottom_edge,
                    bd: env.bd,
                },
                var_cache,
            );
            this_rd = apply_variance_factor(this_rd, factor);
        }

        intra_modes_rd_cost[mode][(luma_delta_angle + MAX_ANGLE_DELTA + 1) as usize] = this_rd;

        // store_winner_mode_stats (intra_mode_search.c:1646): hard no-op when
        // the two-pass is off (MULTI_WINNER_MODE_OFF, speed 0..=3).
        if let Some(wm) = cfg.winner_mode {
            store_winner_mode_stats(
                &mut winner_stats,
                wm.max_winner_count,
                WinnerModeEntry {
                    mode,
                    angle_delta: luma_delta_angle,
                    use_filter_intra: false,
                    filter_intra_mode: 0,
                    rd: this_rd,
                    palette_y: None,
                },
            );
        }

        if this_rd < best_rd {
            best_rd = this_rd;
            best = Some(IntraSbyBest {
                mode,
                angle_delta: luma_delta_angle,
                tx_size: choice.best_tx_size,
                winners: choice.winners,
                rate: this_rate,
                rate_tokenonly: this_rate_tokenonly,
                dist: this_distortion,
                skippable: choice.stats.skip_txfm,
                best_rd: this_rd,
                use_filter_intra: false,
                filter_intra_mode: 0,
                palette_y: None,
            });
        }
    }

    // Palette search (av1_rd_pick_palette_intra_sby, intra_mode_search.c:1662)
    // — try_palette = enable_palette && av1_allow_palette(..): `cfg.palette`
    // models the enable_palette half (None under --enable-palette=0, the
    // standard-shim envelope), `cfg.try_palette` the allow half.
    if cfg.try_palette {
        if let Some(pal_cfg) = &cfg.palette {
            let args = crate::palette_search::PaletteSearchArgs {
                dc_mode_cost: bmode_costs[0],
                cfg,
                pass_pol,
                pass_method,
                palette_costs: pal_cfg.costs,
                palette_above: pal_cfg.above.as_ref(),
                palette_left: pal_cfg.left.as_ref(),
                prune_palette_search_level: pal_cfg.prune_palette_search_level,
                prune_luma_palette_size_search_level: pal_cfg.prune_luma_palette_size_search_level,
                discount_color_cost: false,
                source_variance: cfg.source_variance,
            };
            crate::palette_search::rd_pick_palette_intra_sby(
                env,
                recon,
                &args,
                &mut best_rd,
                &mut best,
                &mut winner_stats,
            );
        }
    }

    // Filter-intra search: LIVE at speed 0 (intra_mode_search.c:1672 —
    // beat_best_rd && av1_filter_intra_allowed_bsize; the seq-level flag and
    // the mode-info cost gate share the enable_filter_intra tool flag). Runs
    // under the SAME (first-pass) eval params; its candidates also feed the
    // winner-stats list (intra_mode_search.c:293).
    if best.is_some() && filter_intra_allowed_bsize(cfg.enable_filter_intra, bsize) {
        let best_mode_so_far = best.as_ref().map_or(0, |b| b.mode);
        let mode_cost_dc = bmode_costs[0];
        if let Some(fi) = rd_pick_filter_intra_sby_y(
            env,
            recon,
            cfg,
            var_cache,
            mode_cost_dc,
            best_mode_so_far,
            &mut best_rd,
            &mut best_model_rd,
            pass_pol,
            pass_method,
            &mut winner_stats,
        ) {
            best = Some(fi);
        }
    }

    // Winner-mode processing (intra_mode_search.c:1689-1737): re-evaluate the
    // stored top-N first-pass winners under WINNER_MODE_EVAL params (full
    // tx-type set + full-RD tx-size sweep + always-on trellis). Gated on
    // `beat_best_rd` (C returns INT64_MAX before this when nothing beat the
    // caller's best_rd — `best.is_some()` here). `is_winner_mode_processing_
    // enabled` is 1 on this path at speed 4/5 (fast_intra_tx_type_search=2,
    // use_intra_{dct_only,default_tx_only}=0, bypass level 0); at speed >= 6
    // the level-1 [`bypass_winner_mode_processing`] source-variance gate can
    // turn the whole re-eval off (its inputs are block-level, so the C
    // per-winner check hoists losslessly out of the loop). At
    // MULTI_WINNER_MODE_OFF (`max_winner_count == 1`, speed >= 6) the stats
    // list is empty (store is a no-op) and C's `else` arm re-evaluates
    // `best_mbmi` once (:1727-1737) — including a filter-intra winner.
    // NOTE the C asymmetry: the re-eval rd (intra_block_yrd :1216) gets NO
    // ALLINTRA variance factor, yet compares against the factored first-pass
    // best_rd.
    if let Some(wm) = cfg.winner_mode {
        if best.is_some() {
            // WINNER pass: USE_FULL_RD ⇒ TX_MODE_SELECT (non-lossless).
            env.tx_mode_is_select =
                select_default && wm.winner_tx_size_method != crate::tx_search::USE_LARGESTALL;
            let reeval: Vec<WinnerModeEntry> = if bypass_winner_mode_processing(
                cfg.source_variance,
                cfg.qindex,
                wm.prune_winner_mode_eval_level,
            ) {
                Vec::new()
            } else if wm.max_winner_count > 1 {
                std::mem::take(&mut winner_stats)
            } else {
                let b = best.as_ref().expect("gated on best.is_some()");
                vec![WinnerModeEntry {
                    mode: b.mode,
                    angle_delta: b.angle_delta,
                    use_filter_intra: b.use_filter_intra,
                    filter_intra_mode: b.filter_intra_mode,
                    rd: b.best_rd,
                    palette_y: b.palette_y.clone(),
                }]
            };
            for entry in reeval.iter() {
                // *mbmi = x->winner_mode_stats[mode_idx].mbmi (:1697).
                env.mode = entry.mode;
                env.angle_delta = entry.angle_delta;
                env.use_filter_intra = entry.use_filter_intra;
                env.filter_intra_mode = entry.filter_intra_mode;
                // intra_block_yrd (:1188): ref_best_rd = use_rd_based_breakout
                // _for_intra_tx_search ? *best_rd : INT64_MAX (:1201; the sf is
                // ON at speed>=3, carried on the winner pol).
                let ref_rd = if wm.winner_pol.use_rd_based_breakout_for_intra_tx_search {
                    best_rd
                } else {
                    i64::MAX
                };
                // Palette entry: restore the colour map + palette prediction
                // (the winner-stats color_index_map copy, :1701-1707).
                let pal_yrd = entry
                    .palette_y
                    .as_ref()
                    .map(|p| crate::tx_search::PaletteYrd {
                        colors: &p.colors,
                        size: p.size,
                        map: &p.color_map,
                        map_stride: crate::tx_search::BLK_W_B[bsize],
                    });
                let Some(choice) = pick_uniform_tx_size_type_yrd_intra(
                    env,
                    recon,
                    ref_rd,
                    wm.winner_pol,
                    cfg.source_variance,
                    cfg.enable_tx64,
                    cfg.enable_rect_tx,
                    wm.winner_tx_size_method,
                    pal_yrd.as_ref(),
                ) else {
                    continue; // rd_stats.rate == INT_MAX (:1204)
                };
                let mut this_rate_tokenonly = choice.stats.rate;
                if !env.lossless && block_signals_txsize(bsize) {
                    this_rate_tokenonly -= tx_size_cost(
                        env.tx_size_costs,
                        env.tx_mode_is_select,
                        bsize,
                        choice.best_tx_size,
                        env.tx_size_ctx,
                    );
                }
                // Palette entry: the palette size/colour/map extra rate
                // (recomputed here as intra_mode_info_cost_y does from mbmi).
                let palette_extra = entry.palette_y.as_ref().map_or(0, |p| {
                    cfg.palette.as_ref().map_or(0, |pal_cfg| {
                        crate::palette_search::palette_y_extra_rate(
                            env,
                            p,
                            pal_cfg,
                            cfg.palette_bsize_ctx,
                        )
                    })
                });
                let this_rate = choice.stats.rate
                    + if entry.use_filter_intra {
                        intra_mode_info_cost_y(
                            cfg.mode_costs,
                            bmode_costs[0],
                            0, // DC_PRED
                            bsize,
                            0,
                            true,
                            entry.filter_intra_mode,
                            false,
                            cfg.try_palette,
                            cfg.palette_bsize_ctx,
                            cfg.palette_mode_ctx,
                            cfg.enable_filter_intra,
                            cfg.allow_intrabc,
                            false,
                            0,
                        )
                    } else {
                        intra_mode_info_cost_y(
                            cfg.mode_costs,
                            bmode_costs[entry.mode],
                            entry.mode,
                            bsize,
                            entry.angle_delta,
                            false,
                            0,
                            false,
                            cfg.try_palette,
                            cfg.palette_bsize_ctx,
                            cfg.palette_mode_ctx,
                            cfg.enable_filter_intra,
                            cfg.allow_intrabc,
                            entry.palette_y.is_some(),
                            palette_extra,
                        )
                    };
                let this_rd = rd::rdcost(env.rdmult, this_rate, choice.stats.dist);
                if this_rd < best_rd {
                    best_rd = this_rd;
                    best = Some(IntraSbyBest {
                        mode: entry.mode,
                        angle_delta: entry.angle_delta,
                        tx_size: choice.best_tx_size,
                        winners: choice.winners,
                        rate: this_rate,
                        rate_tokenonly: this_rate_tokenonly,
                        dist: choice.stats.dist,
                        skippable: choice.stats.skip_txfm,
                        best_rd: this_rd,
                        use_filter_intra: entry.use_filter_intra,
                        filter_intra_mode: entry.filter_intra_mode,
                        palette_y: entry.palette_y.clone(),
                    });
                }
            }
        }
        // Leave env's tx-mode select gating as the caller provided it (the C
        // resets to DEFAULT_EVAL right after this returns, rdopt.c:3659).
        env.tx_mode_is_select = select_default;
    }

    IntraSbyOutcome {
        best,
        intra_modes_rd_cost,
    }
}

// ---------------------------------------------------------------------------
// rd_pick_filter_intra_sby (intra_mode_search.c:231) — the filter-intra
// search that follows the Y mode loop (LIVE at speed 0:
// prune_filter_intra_level = 0 and filter-intra allowed up to 32x32).
// ---------------------------------------------------------------------------

/// `FILTER_INTRA_MODES` (enums.h).
pub const FILTER_INTRA_MODES: usize = 5;

/// `av1_derived_filter_intra_mode_used_flag[INTRA_MODES]`
/// (intra_mode_search.c): the `prune_filter_intra_level == 1` per-mode gate —
/// FILTER_DC always + the filter mode matching the best-so-far Y mode.
pub const AV1_DERIVED_FILTER_INTRA_MODE_USED_FLAG: [u8; INTRA_MODES] = [
    0x01, 0x03, 0x05, 0x01, 0x01, 0x01, 0x09, 0x01, 0x01, 0x01, 0x01, 0x01, 0x11,
];

/// `av1_filter_intra_allowed_bsize` (reconintra.h): the sequence-level
/// enable flag AND both block dims <= 32.
pub fn filter_intra_allowed_bsize(enable_filter_intra_seq: bool, bsize: usize) -> bool {
    const BLK_W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const BLK_H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    enable_filter_intra_seq && BLK_W[bsize] <= 32 && BLK_H[bsize] <= 32
}

/// `model_intra_yrd_and_prune` (intra_mode_search_utils.h): the filter-intra
/// loop's model gate — [`crate::tx_search::intra_model_rd_y`] at
/// `min(TX_32X32, max_txsize_lookup[bsize])` (the prediction walk mutates
/// `recon`, same as the Y loop's model calls), pruned by INTEGER arithmetic
/// against the SHARED `best_model_rd` accumulator (threaded from the Y mode
/// loop): prune when `this > best + (best >> 2)`, else lower `best`.
pub fn model_intra_yrd_and_prune(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    best_model_rd: &mut i64,
) -> bool {
    let tx_size = MAX_TXSIZE_LOOKUP[env.bsize].min(3);
    let this_model_rd = intra_model_rd_y(env, recon, tx_size);
    if *best_model_rd != i64::MAX && this_model_rd > *best_model_rd + (*best_model_rd >> 2) {
        return true;
    } else if this_model_rd < *best_model_rd {
        *best_model_rd = this_model_rd;
    }
    false
}

/// `rd_pick_filter_intra_sby` (intra_mode_search.c:231): loop the 5
/// filter-intra variants of DC_PRED — per mode `model_intra_yrd_and_prune`
/// (shared `best_model_rd`) -> `av1_pick_uniform_tx_size_type_yrd` with the
/// RUNNING `best_rd` -> `this_rate = stats.rate + intra_mode_info_cost_y`
/// (NOTE: `*rate_tokenonly` takes the RAW tx-search rate — no tx-size-cost
/// subtraction here, unlike the Y loop) -> RDCOST -> [ALLINTRA variance
/// factor] -> strict `<` best tracking. `mbmi->angle_delta` is deliberately
/// NOT reset (the C leaves the Y loop's last value; DC/filter-intra
/// prediction ignores it) — `env.angle_delta` is left as-is likewise.
///
/// Speed-0 gates mirrored: `prune_filter_intra_level == 2` -> no search;
/// `== 1` -> only modes in
/// [`AV1_DERIVED_FILTER_INTRA_MODE_USED_FLAG`]`[best_mode_so_far]`;
/// `use_mb_mode_cache` modelled off. Returns the winner (updating `best_rd`)
/// or `None` (the C's `filter_intra_selected_flag == 0`).
#[allow(clippy::too_many_arguments)]
pub fn rd_pick_filter_intra_sby_y(
    env: &mut TxfmYrdEnv,
    recon: &mut [u16],
    cfg: &IntraSbySearchCfg,
    var_cache: &mut [Block4x4VarInfo],
    mode_cost_dc: i32,
    best_mode_so_far: usize,
    best_rd: &mut i64,
    best_model_rd: &mut i64,
    pass_pol: &TxTypeSearchPolicy,
    pass_method: usize,
    winner_stats: &mut Vec<WinnerModeEntry>,
) -> Option<IntraSbyBest> {
    if cfg.gates.prune_filter_intra_level == 2 {
        return None;
    }
    let bsize = env.bsize;
    env.mode = 0; // DC_PRED
    env.use_filter_intra = true;

    let mut selected: Option<IntraSbyBest> = None;
    for fi_mode in 0..FILTER_INTRA_MODES {
        if cfg.gates.prune_filter_intra_level == 1
            && (AV1_DERIVED_FILTER_INTRA_MODE_USED_FLAG[best_mode_so_far] & (1 << fi_mode)) == 0
        {
            continue;
        }
        env.filter_intra_mode = fi_mode;

        if model_intra_yrd_and_prune(env, recon, best_model_rd) {
            continue;
        }
        let Some(choice) = pick_uniform_tx_size_type_yrd_intra(
            env,
            recon,
            *best_rd,
            pass_pol,
            cfg.source_variance,
            cfg.enable_tx64,
            cfg.enable_rect_tx,
            pass_method,
            None,
        ) else {
            continue; // rate == INT_MAX
        };

        let (above_ctx, left_ctx) = get_y_mode_ctx(cfg.above_mode, cfg.left_mode);
        debug_assert_eq!(
            mode_cost_dc, cfg.mode_costs.y_mode_costs[above_ctx][left_ctx][0],
            "mode_cost is bmode_costs[DC_PRED]",
        );
        let this_rate = choice.stats.rate
            + intra_mode_info_cost_y(
                cfg.mode_costs,
                mode_cost_dc,
                0, // DC_PRED
                bsize,
                0, // angle rate: DC is non-directional (delta not costed)
                true,
                fi_mode,
                false,
                cfg.try_palette,
                cfg.palette_bsize_ctx,
                cfg.palette_mode_ctx,
                cfg.enable_filter_intra,
                cfg.allow_intrabc,
                false,
                0,
            );
        let mut this_rd = rd::rdcost(env.rdmult, this_rate, choice.stats.dist);
        if cfg.allintra && this_rd != i64::MAX {
            let factor = intra_rd_variance_factor(
                cfg.speed,
                &VarFactorInputs {
                    src: env.src,
                    src_off: env.src_off,
                    src_stride: env.src_stride,
                    recon,
                    ref_off: env.ref_off,
                    ref_stride: env.ref_stride,
                    bsize,
                    sb_size: env.sb_size,
                    mi_row: env.mi_row,
                    mi_col: env.mi_col,
                    mb_to_right_edge: cfg.mb_to_right_edge,
                    mb_to_bottom_edge: cfg.mb_to_bottom_edge,
                    bd: env.bd,
                },
                var_cache,
            );
            this_rd = apply_variance_factor(this_rd, factor);
        }
        // store_winner_mode_stats (intra_mode_search.c:293): the filter-intra
        // candidates feed the same winner list (no-op when two-pass is off —
        // the caller passes max 0... gated on winner_mode instead).
        if let Some(wm) = cfg.winner_mode {
            store_winner_mode_stats(
                winner_stats,
                wm.max_winner_count,
                WinnerModeEntry {
                    mode: 0, // DC_PRED (mbmi->mode)
                    angle_delta: env.angle_delta,
                    use_filter_intra: true,
                    filter_intra_mode: fi_mode,
                    rd: this_rd,
                    palette_y: None,
                },
            );
        }
        if this_rd < *best_rd {
            *best_rd = this_rd;
            selected = Some(IntraSbyBest {
                mode: 0,
                angle_delta: env.angle_delta,
                tx_size: choice.best_tx_size,
                winners: choice.winners,
                rate: this_rate,
                rate_tokenonly: choice.stats.rate,
                dist: choice.stats.dist,
                skippable: choice.stats.skip_txfm,
                best_rd: this_rd,
                use_filter_intra: true,
                filter_intra_mode: fi_mode,
                palette_y: None,
            });
        }
    }
    selected
}

#[cfg(test)]
mod winner_stats_tests {
    use super::*;

    fn e(mode: usize, rd: i64) -> WinnerModeEntry {
        WinnerModeEntry {
            mode,
            angle_delta: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            rd,
            palette_y: None,
        }
    }
    fn rds(list: &[WinnerModeEntry]) -> Vec<(usize, i64)> {
        list.iter().map(|w| (w.mode, w.rd)).collect()
    }

    /// KB-8 chunk 2d-ii: `store_winner_mode_stats` reproduces the C insertion
    /// semantics (rdopt_utils.h:679) — ascending-rd insertion at the first
    /// STRICTLY-greater slot, INT64_MAX rejection, full-list rejection when no
    /// stored rd exceeds the candidate, tail-drop on overflow, and tie-stable
    /// ordering (equal rds keep the earlier entry ahead).
    #[test]
    fn store_winner_mode_stats_matches_c_semantics() {
        let max = 3usize; // winner_mode_count_allowed[MULTI_WINNER_MODE_DEFAULT]
        let mut l = Vec::new();
        // INT64_MAX rejected outright.
        store_winner_mode_stats(&mut l, max, e(9, i64::MAX));
        assert!(l.is_empty());
        // Fill ascending out of order: 50, 10, 30 -> sorted [10,30,50].
        store_winner_mode_stats(&mut l, max, e(1, 50));
        store_winner_mode_stats(&mut l, max, e(2, 10));
        store_winner_mode_stats(&mut l, max, e(3, 30));
        assert_eq!(rds(&l), [(2, 10), (3, 30), (1, 50)]);
        // Worse than the full list's tail: rejected (C: mode_idx == max).
        store_winner_mode_stats(&mut l, max, e(4, 60));
        assert_eq!(rds(&l), [(2, 10), (3, 30), (1, 50)]);
        // Equal to the tail: ALSO rejected (strictly-greater compare — the
        // scan runs past the equal slot to mode_idx == max).
        store_winner_mode_stats(&mut l, max, e(5, 50));
        assert_eq!(rds(&l), [(2, 10), (3, 30), (1, 50)]);
        // Mid insert evicts the tail.
        store_winner_mode_stats(&mut l, max, e(6, 20));
        assert_eq!(rds(&l), [(2, 10), (6, 20), (3, 30)]);
        // Tie with a NON-tail entry: goes AFTER it (first strictly-greater
        // slot), evicting the tail.
        store_winner_mode_stats(&mut l, max, e(7, 20));
        assert_eq!(rds(&l), [(2, 10), (6, 20), (7, 20)]);
        // Best-so-far inserts at the head.
        store_winner_mode_stats(&mut l, max, e(8, 5));
        assert_eq!(rds(&l), [(8, 5), (2, 10), (6, 20)]);
        // max_count = 1 ⇔ MULTI_WINNER_MODE_OFF (winner_mode_count_allowed[OFF]
        // = 1, and OFF is the only type with count 1): C's store returns
        // immediately — "Mode stat is not required when multiwinner mode
        // processing is disabled" (rdopt_utils.h:688) — so NOTHING is ever
        // stored; the speed>=6 re-eval runs on `best_mbmi` instead
        // (intra_mode_search.c:1727-1737).
        let mut one = Vec::new();
        store_winner_mode_stats(&mut one, 1, e(1, 100));
        store_winner_mode_stats(&mut one, 1, e(3, 99));
        assert!(one.is_empty());
    }
}
