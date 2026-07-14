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

use crate::mode_costs::{intra_mode_info_cost_y, IntraModeCosts};
use crate::{
    dist_block_tx_domain, rd, xform_quant_optimize, BlockContext, OptimizeInputs, QuantKind,
    QuantParams,
};
use aom_dist::highbd_subtract_block;
use aom_entropy::partition::get_y_mode_ctx;
use aom_intra::predict_intra_high;
use aom_txb::{cost_coeffs_txb, get_tx_type_cost, CoeffCostTables, TxTypeCosts};

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
    assert_eq!(crate::BLK_W[env.bsize], w, "bsize/tx_size width mismatch (single-txb scope)");
    assert_eq!(crate::BLK_H[env.bsize], h, "bsize/tx_size height mismatch (single-txb scope)");

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
    );
    let rate = coeff_rate + tx_type_rate + mode_rate;

    // Transform-domain distortion, then one RDCOST over the summed rate.
    let (dist, _sse) = dist_block_tx_domain(&r.coeff, &r.dqcoeff, env.tx_size, env.bd);
    let rd = rd::rdcost(rates.rdmult, rate, dist);

    IntraModeRd { rate, dist, rd, eob: r.eob }
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
