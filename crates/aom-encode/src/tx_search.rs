//! Transform search primitives (libaom `av1/encoder/tx_search.c`) — the
//! per-txb pieces of `search_tx_type` for the speed-0 all-intra path:
//! - [`get_tx_mask_intra`]: the allowed tx-type set for a luma intra txb
//!   (`get_tx_mask`, intra arm);
//! - [`av1_pixel_diff_dist`] (+ [`get_txb_visible_dimensions`]): the residual
//!   SSE / mean-squared-error the search's trellis/dist policies key off.
//!
//! Speed-0 all-intra sf resolution for `get_tx_mask` (each named, values from
//! `av1/encoder/speed_features.c`):
//! - `tx_type_search.use_reduced_intra_txset = 1`
//!   (`set_allintra_speed_features_framesize_independent`, speed-0 block)
//! - `tx_type_search.prune_tx_type_using_stats = 0` (default; allintra sets
//!   it only at higher speeds) — stats prune arm never runs
//! - `tx_type_search.prune_tx_type_est_rd = 0` (default) — `prune_txk_type*`
//!   never runs, so `txk_map` stays identity
//! - `prune_2d_txfm_mode = TX_TYPE_PRUNE_1` (default) but `prune_tx_2D` is
//!   gated `is_inter` — never runs for intra
//! - `txfm_params.use_default_intra_tx_type = 0` and
//!   `use_derived_intra_tx_type_set = 0` (MODE_EVAL with
//!   `fast_intra_tx_type_search = 0`, the speed-0 default)
//! - `x->rd_model = FULL_TXFM_RD` (set by `choose_tx_size_type_from_rd`)
//!
//! CLI-default tool flags (`aomenc` defaults): `enable_flip_idtx = 1`,
//! `use_intra_dct_only = 0`.

use aom_txb::ext_tx_set_type;

/// `TX_TYPES` (enums.h).
pub const TX_TYPES: usize = 16;

/// `av1_ext_tx_used_flag[EXT_TX_SET_TYPES]` (blockd.h): bit `t` set = tx type
/// `t` usable in that ext-tx set type.
pub const AV1_EXT_TX_USED_FLAG: [u16; 6] = [0x0001, 0x0201, 0x020F, 0x0E0F, 0x0FFF, 0xFFFF];

/// `av1_reduced_intra_tx_used_flag[INTRA_MODES]` (blockd.h): the reduced
/// intra tx set (sf `use_reduced_intra_txset >= 1`), per intra direction.
pub const AV1_REDUCED_INTRA_TX_USED_FLAG: [u16; 13] = [
    0x080F, 0x040F, 0x080F, 0x020F, 0x080F, 0x040F, 0x080F, 0x080F, 0x040F, 0x080F, 0x040F,
    0x080F, 0x0C0E,
];

/// `av1_derived_intra_tx_used_flag[INTRA_MODES]` (blockd.h): the
/// residual-statistics-derived set (sf `use_reduced_intra_txset == 2`).
pub const AV1_DERIVED_INTRA_TX_USED_FLAG: [u16; 13] = [
    0x0209, 0x0403, 0x0805, 0x020F, 0x0009, 0x0009, 0x0009, 0x0805, 0x0403, 0x0205, 0x0403,
    0x0805, 0x0209,
];

/// `fimode_to_intradir[FILTER_INTRA_MODES]` (blockd.h): the intra direction a
/// filter-intra mode maps to for tx-set/tx-type decisions.
pub const FIMODE_TO_INTRADIR: [usize; 5] = [0, 1, 2, 6, 0];

/// `DCT_ADST_TX_MASK` (txfm_common.h): DCT/ADST-only (kills FLIPADST + IDTX
/// combinations when `enable_flip_idtx` is off).
pub const DCT_ADST_TX_MASK: u16 = 0x000F;

/// `txsize_sqr_up_map[TX_SIZES_ALL]` (common_data.h): TX_SIZE -> square
/// TX_SIZE class rounding UP (0..4 = 4x4..64x64).
pub const TXSIZE_SQR_UP_MAP: [usize; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];

/// `EXT_TX_SET_DTT4_IDTX_1DDCT` (enums.h `TxSetType`, value 3 — after
/// DCTONLY=0, DCT_IDTX=1, DTT4_IDTX=2): the intra set the reduced-txset sf
/// replaces with a per-direction table.
pub const EXT_TX_SET_DTT4_IDTX_1DDCT: usize = 3;

/// The `TxfmSearchParams` / tool-config gates `get_tx_mask` reads on the
/// intra path. [`TxMaskParams::speed0_allintra`] bakes the speed-0 values
/// (see module docs for the per-sf provenance).
#[derive(Clone, Copy, Debug)]
pub struct TxMaskParams {
    /// sf `tx_type_search.use_reduced_intra_txset` (0/1/2).
    pub use_reduced_intra_txset: u8,
    /// `txfm_params.use_derived_intra_tx_type_set`.
    pub use_derived_intra_tx_type_set: bool,
    /// `oxcf.txfm_cfg.enable_flip_idtx` (CLI default on).
    pub enable_flip_idtx: bool,
    /// `oxcf.txfm_cfg.use_intra_dct_only` (CLI default off).
    pub use_intra_dct_only: bool,
}

impl TxMaskParams {
    /// Speed-0 all-intra defaults.
    pub fn speed0_allintra() -> Self {
        TxMaskParams {
            use_reduced_intra_txset: 1,
            use_derived_intra_tx_type_set: false,
            enable_flip_idtx: true,
            use_intra_dct_only: false,
        }
    }
}

/// `get_tx_mask` (tx_search.c, static) — the LUMA INTRA arm: the bitmask of
/// tx types `search_tx_type` iterates for one txb, plus `txk_allowed`
/// (`Some(t)` when exactly one specific type is allowed, `None` = the mask is
/// multi-type). The candidate order is the identity `txk_map` (the est-rd
/// reorder never runs at speed 0 — see module docs).
///
/// Out of scope (labelled): the inter arms (`default_inter_tx_type_prob_thresh`
/// frame-probability forcing, `prune_tx_2D`, stats prune), the est-rd prune,
/// `use_default_intra_tx_type` (`get_default_tx_type`; sf OFF at speed 0), the
/// `rd_model == LOW_TXFM_RD` DCT-only override (the pick loop runs
/// `FULL_TXFM_RD`), and the UV path (tx type inherited from Y).
pub fn get_tx_mask_intra(
    tx_size: usize,
    mode: usize,
    use_filter_intra: bool,
    filter_intra_mode: usize,
    lossless: bool,
    reduced_tx_set_used: bool,
    p: &TxMaskParams,
) -> (u16, Option<usize>) {
    let mut txk_allowed = TX_TYPES; // "all"
    let tx_set_type = ext_tx_set_type(tx_size, false, reduced_tx_set_used);

    let intra_dir = if use_filter_intra { FIMODE_TO_INTRADIR[filter_intra_mode] } else { mode };
    let mut ext_tx_used_flag =
        if p.use_reduced_intra_txset != 0 && tx_set_type == EXT_TX_SET_DTT4_IDTX_1DDCT {
            AV1_REDUCED_INTRA_TX_USED_FLAG[intra_dir]
        } else {
            AV1_EXT_TX_USED_FLAG[tx_set_type]
        };
    if p.use_reduced_intra_txset == 2 {
        ext_tx_used_flag &= AV1_DERIVED_INTRA_TX_USED_FLAG[intra_dir];
    }

    if lossless || TXSIZE_SQR_UP_MAP[tx_size] > 3 || ext_tx_used_flag == 0x0001 || p.use_intra_dct_only
    {
        txk_allowed = 0; // DCT_DCT
    }
    if !p.enable_flip_idtx {
        ext_tx_used_flag &= DCT_ADST_TX_MASK;
    }

    let mut allowed_tx_mask: u16;
    if txk_allowed < TX_TYPES {
        allowed_tx_mask = (1 << txk_allowed) & ext_tx_used_flag;
    } else if p.use_derived_intra_tx_type_set {
        allowed_tx_mask = AV1_DERIVED_INTRA_TX_USED_FLAG[intra_dir] & ext_tx_used_flag;
    } else {
        allowed_tx_mask = ext_tx_used_flag;
        // Stats prune / est-rd prune / prune_tx_2D: all structurally off for
        // the speed-0 intra path (see module docs).
    }

    if allowed_tx_mask == 0 {
        txk_allowed = 0; // DCT_DCT (plane 0)
        allowed_tx_mask = 1 << txk_allowed;
    }

    let single = if txk_allowed < TX_TYPES { Some(txk_allowed) } else { None };
    debug_assert!(single.is_none_or(|t| allowed_tx_mask == 1 << t));
    (allowed_tx_mask, single)
}

/// The visible-dimension slice of `get_txb_dimensions` (rdopt_utils.h): a
/// txb's pixels clipped to the frame boundary. `mb_to_right_edge` /
/// `mb_to_bottom_edge` are the MACROBLOCKD edge fields (1/8-pel units,
/// negative when the block overhangs), `subsampling` the plane's.
#[allow(clippy::too_many_arguments)] // mirrors the C signature
pub fn get_txb_visible_dimensions(
    plane_bsize_w: usize,
    plane_bsize_h: usize,
    tx_w: usize,
    tx_h: usize,
    blk_row: usize,
    blk_col: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    subsampling_x: u32,
    subsampling_y: u32,
) -> (usize, usize) {
    let visible_height = if mb_to_bottom_edge >= 0 {
        tx_h
    } else {
        let block_rows = (mb_to_bottom_edge >> (3 + subsampling_y)) + plane_bsize_h as i32;
        (block_rows - ((blk_row as i32) << 2)).clamp(0, tx_h as i32) as usize
    };
    let visible_width = if mb_to_right_edge >= 0 {
        tx_w
    } else {
        let block_cols = (mb_to_right_edge >> (3 + subsampling_x)) + plane_bsize_w as i32;
        (block_cols - ((blk_col as i32) << 2)).clamp(0, tx_w as i32) as usize
    };
    (visible_width, visible_height)
}

/// `av1_pixel_diff_dist` (tx_search.c): the residual (src - pred) SSE over the
/// txb's VISIBLE pixels, plus `block_mse_q8 = 256 * sse / visible_pels`
/// (`u32::MAX` when the visible area is empty). `diff` is the plane's
/// `src_diff` buffer (stride = plane block width); `blk_row`/`blk_col` in
/// 4-pel MI units.
pub fn av1_pixel_diff_dist(
    diff: &[i16],
    diff_stride: usize,
    blk_row: usize,
    blk_col: usize,
    visible_cols: usize,
    visible_rows: usize,
) -> (u64, u32) {
    let off = (blk_row * diff_stride + blk_col) << 2; // MI_SIZE_LOG2
    let sse = aom_dist::sum_squares_2d_i16(&diff[off..], diff_stride, visible_cols, visible_rows);
    let mse_q8 = if visible_cols > 0 && visible_rows > 0 {
        ((256 * sse) / (visible_cols as u64 * visible_rows as u64)) as u32
    } else {
        u32::MAX
    };
    (sse, mse_q8)
}

// ---------------------------------------------------------------------------
// search_tx_type (tx_search.c) — the per-txb tx-type RD search, luma intra,
// speed-0 policy (see module docs), interior txbs (visible == full).
// ---------------------------------------------------------------------------

use crate::rd::rdcost;
use crate::{
    dist_block_tx_domain, xform_quant, xform_quant_optimize, BlockContext, OptimizeInputs,
    QuantKind, QuantParams, XformQuantOptResult,
};
use aom_txb::{cost_coeffs_txb, get_tx_type_cost, CoeffCostTables, TxTypeCosts};

/// `tx_size_2d[TX_SIZES_ALL]` (av1/common/common_data.h): pel count per tx.
pub const TX_SIZE_2D_TBL: [i64; 19] =
    [16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024];

const TXS_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TXS_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

/// `ROUND_POWER_OF_TWO` for i64.
#[inline]
fn round_power_of_two_i64(value: i64, n: i32) -> i64 {
    (value + ((1i64 << n) >> 1)) >> n
}

/// The trellis RD multiplier `av1_optimize_txb` derives from the block
/// `x->rdmult` (encodetxb.h `plane_rd_mult` tables; luma-intra entry is 17 in
/// BOTH the default and `use_chroma_trellis_rd_mult` tables, so the speed-0
/// allintra sf `use_chroma_trellis_rd_mult = 1` is a no-op for luma):
/// `ROUND_POWER_OF_TWO(rdmult * (8 - sharpness) * (17 << (2*(bd-8))), 5)`
/// (`rshift = 5` — PSNR tuning; IQ/SSIMULACRA2 use 7, out of scope).
#[inline]
pub fn trellis_rdmult_intra_y(rdmult: i32, sharpness: i32, bd: u8) -> i64 {
    round_power_of_two_i64(
        (rdmult as i64) * ((8 - sharpness) as i64) * ((17i64) << (2 * (bd as i32 - 8))),
        5,
    )
}

/// The speed-0 policy knobs of `search_tx_type` (each documented with its
/// speed-0 value in the module docs / commit message).
#[derive(Clone, Copy, Debug)]
pub struct TxTypeSearchPolicy {
    /// `!is_trellis_used(optimize_coefficients, DRY_RUN_NORMAL)` — speed-0
    /// allintra: `FULL_TRELLIS_OPT` (CLI `disable_trellis_quant = 0`, not
    /// lossless) => `false`.
    pub skip_trellis: bool,
    /// `txfm_params->coeff_opt_thresholds[0]` (block-MSE/qstep^2 gate for the
    /// trellis) — speed 0: `coeff_opt_thresholds[perform_coeff_opt=1]
    /// [DEFAULT_EVAL][0] = 3200` (enable_winner_mode_for_coeff_opt = 0).
    pub coeff_opt_dist_threshold: u32,
    /// `coeff_opt_thresholds[1]` (SATD gate) — speed 0: `UINT_MAX`, which
    /// short-circuits `skip_trellis_opt_based_on_satd` before any SATD work
    /// (the SATD body is unported; reaching it panics).
    pub coeff_opt_satd_threshold: u32,
    /// `txfm_params->use_transform_domain_distortion` — speed 0:
    /// `tx_domain_dist_types[tx_domain_dist_level=0][DEFAULT_EVAL] = 0`
    /// (pixel-domain during the loop, with the 64-pt/high-energy hybrid).
    pub use_transform_domain_distortion: u8,
    /// `txfm_params->tx_domain_dist_threshold` — speed 0:
    /// `tx_domain_dist_thresholds[0][DEFAULT_EVAL] = UINT_MAX`.
    pub tx_domain_dist_threshold: u32,
    /// sf `tx_sf.adaptive_txb_search_level` — speed-0 allintra: 1.
    pub adaptive_txb_search_level: i32,
    /// sf `tx_sf.tx_type_search.skip_tx_search` — speed 0: 0.
    pub skip_tx_search: bool,
    /// `oxcf.algo_cfg.sharpness` (CLI default 0).
    pub sharpness: i32,
}

impl TxTypeSearchPolicy {
    /// Speed-0 all-intra defaults (provenance per field above).
    pub fn speed0_allintra() -> Self {
        TxTypeSearchPolicy {
            skip_trellis: false,
            coeff_opt_dist_threshold: 3200,
            coeff_opt_satd_threshold: u32::MAX,
            use_transform_domain_distortion: 0,
            tx_domain_dist_threshold: u32::MAX,
            adaptive_txb_search_level: 1,
            skip_tx_search: false,
            sharpness: 0,
        }
    }
}

/// Everything `search_tx_type` reads for one interior luma intra txb.
pub struct TxTypeSearchInputs<'a> {
    /// Residual (src - pred), full `TX_W x TX_H`, stride = TX_W.
    pub residual: &'a [i16],
    /// Source pixels of the txb (u16 universal repr), stride `src_stride`.
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    /// The intra prediction, full `TX_W x TX_H` contiguous (stride = TX_W).
    pub pred: &'a [u16],
    pub tx_size: usize,
    /// Intra mode + filter-intra state (tx-set + tx-type-rate selection).
    pub mode: usize,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub bd: u8,
    /// The per-qindex quantizer rows for this plane (both FP and B are
    /// reachable: FP with trellis, B when the trellis is skipped —
    /// `USE_B_QUANT_NO_TRELLIS = 1`).
    pub rows: &'a aom_quant::PlaneQuantRows<'a>,
    /// Neighbour entropy contexts (`get_txb_ctx` inputs).
    pub bctx: &'a BlockContext<'a>,
    /// The block RD multiplier `x->rdmult`.
    pub rdmult: i32,
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
}

/// One evaluated tx type's outcome (the winner's is returned).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxTypeSearchResult {
    pub best_tx_type: usize,
    pub best_eob: u16,
    pub best_txb_ctx: u8,
    /// Winner rate (coeff bits + non-skip/skip + tx_type), dist, sse, rd.
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub rd: i64,
    pub skip_txfm: bool,
    /// The winner's quantized/dequantized coefficients (the C keeps the best
    /// dqcoeff via buffer swap; callers reconstruct from these).
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    /// Coverage/introspection: which tx types were evaluated (bit per type).
    pub evaluated_mask: u16,
}

/// `search_tx_type` (tx_search.c, static) for one INTERIOR luma intra txb at
/// the speed-0 policy: iterate the allowed tx types (identity `txk_map`), per
/// type forward-transform + quantize (+ trellis when enabled) + rate + the
/// pixel-domain/high-energy-hybrid distortion, track the strict-min RD, with
/// the `adaptive_txb_search_level` and `skip_tx_search` early breaks.
///
/// Scope (labelled): interior txbs (visible == full txb — frame-edge-clipped
/// distortion via `pixel_dist_visible_only` unported), `predict_dc_level = 0`
/// (no DC-only prediction), the SATD trellis-skip short-circuit only
/// (threshold `UINT_MAX` at speed 0; the SATD body panics if reached), no
/// `recon_intra` (callers reconstruct from the returned winner coefficients),
/// flat quant (no qmatrix), plane 0.
pub fn search_tx_type_intra(
    inp: &TxTypeSearchInputs,
    pol: &TxTypeSearchPolicy,
    ref_best_rd: i64,
) -> Option<TxTypeSearchResult> {
    let tx_size = inp.tx_size;
    let (w, h) = (TXS_W[tx_size], TXS_H[tx_size]);
    let hbd = inp.bd > 8;

    // qstep from the AC dequant lane (dequant_QTX[1] >> dequant_shift).
    let dequant_shift = if hbd { inp.bd as i32 - 5 } else { 3 };
    let qstep = (i32::from(inp.rows.dequant[1]) >> dequant_shift) as u32;

    // Residual SSE + MSE (interior => visible == full).
    let (mut block_sse_u, mut block_mse_q8) =
        av1_pixel_diff_dist(inp.residual, w, 0, 0, w, h);
    let mut block_sse = block_sse_u as i64;
    if hbd {
        let s = 2 * (inp.bd as i32 - 8);
        block_sse = (block_sse + ((1i64 << s) >> 1)) >> s;
        block_mse_q8 = (((block_mse_q8 as u64) + ((1u64 << s) >> 1)) >> s) as u32;
        block_sse_u = block_sse as u64;
    }
    let _ = block_sse_u;
    block_sse *= 16;

    // Allowed tx-type set (identity txk_map at speed 0).
    let (allowed_tx_mask, txk_allowed) = get_tx_mask_intra(
        tx_size,
        inp.mode,
        inp.use_filter_intra,
        inp.filter_intra_mode,
        inp.lossless,
        inp.reduced_tx_set_used,
        &TxMaskParams::speed0_allintra(),
    );

    // Trellis gating: block-MSE / qstep^2 threshold.
    let mut skip_trellis = pol.skip_trellis;
    let perform_block_coeff_opt = (block_mse_q8 as u64)
        <= (pol.coeff_opt_dist_threshold as u64) * (qstep as u64) * (qstep as u64);
    skip_trellis |= !perform_block_coeff_opt;

    // Distortion-domain policy.
    let mut use_transform_domain_distortion = pol.use_transform_domain_distortion > 0
        && block_mse_q8 >= pol.tx_domain_dist_threshold
        && TXSIZE_SQR_UP_MAP[tx_size] != 4;
    let mut calc_pixel_domain_distortion_final =
        pol.use_transform_domain_distortion == 1 && use_transform_domain_distortion;
    if calc_pixel_domain_distortion_final
        && (txk_allowed.is_some() || allowed_tx_mask == 0x0001)
    {
        calc_pixel_domain_distortion_final = false;
        use_transform_domain_distortion = false;
    }

    // av1_setup_quant: FP with trellis, B without (USE_B_QUANT_NO_TRELLIS=1).
    let kind = if skip_trellis { QuantKind::B } else { QuantKind::Fp };
    let qp = QuantParams::from_plane_rows(inp.rows, kind, inp.bd);
    let trellis_rdmult = trellis_rdmult_intra_y(inp.rdmult, pol.sharpness, inp.bd);
    let opt = OptimizeInputs {
        cost: inp.coeff_costs,
        rdmult: trellis_rdmult,
        sharpness: pol.sharpness,
    };

    let mut best: Option<TxTypeSearchResult> = None;
    let mut best_rd = i64::MAX;
    let mut evaluated_mask = 0u16;

    for tx_type in 0..TX_TYPES {
        if allowed_tx_mask & (1 << tx_type) == 0 {
            continue;
        }
        evaluated_mask |= 1 << tx_type;

        // SATD-based trellis skip: short-circuited at speed 0
        // (skip_trellis || threshold == UINT_MAX). The SATD body is unported.
        let skip_trellis_this = if skip_trellis || pol.coeff_opt_satd_threshold == u32::MAX {
            skip_trellis
        } else {
            unimplemented!("SATD trellis-skip body (coeff_opt_satd_threshold < UINT_MAX)")
        };

        // Forward transform + quantize (+ trellis + rate).
        let (res, rate_cost): (XformQuantOptResult, i32) = if !skip_trellis_this {
            let r = xform_quant_optimize(
                inp.residual,
                tx_size,
                tx_type,
                kind,
                &qp,
                inp.bctx,
                &opt,
            );
            // av1_optimize_txb rate += tx_type cost when eob > 0.
            let ttc = if r.eob > 0 {
                get_tx_type_cost(
                    inp.tx_type_costs,
                    0,
                    tx_size,
                    tx_type,
                    false,
                    inp.reduced_tx_set_used,
                    inp.lossless,
                    inp.use_filter_intra,
                    inp.filter_intra_mode,
                    inp.mode,
                )
            } else {
                0
            };
            let rate = r.rate + ttc;
            (r, rate)
        } else {
            // No-trellis arm: B quant, entropy ctx computed by av1_quant,
            // rate via av1_cost_coeffs_txb (+ tx_type inside its eob>0 body).
            let xq = xform_quant(inp.residual, tx_size, tx_type, kind, &qp, false);
            let (txb_skip_ctx, dc_sign_ctx) = aom_txb::get_txb_ctx(
                inp.bctx.plane_bsize,
                tx_size,
                inp.bctx.plane,
                inp.bctx.above,
                inp.bctx.left,
            );
            let rate = cost_coeffs_txb(
                &xq.qcoeff,
                xq.eob as usize,
                tx_size,
                tx_type,
                txb_skip_ctx as usize,
                dc_sign_ctx as usize,
                inp.coeff_costs,
            ) + if xq.eob > 0 {
                get_tx_type_cost(
                    inp.tx_type_costs,
                    0,
                    tx_size,
                    tx_type,
                    false,
                    inp.reduced_tx_set_used,
                    inp.lossless,
                    inp.use_filter_intra,
                    inp.filter_intra_mode,
                    inp.mode,
                )
            } else {
                0
            };
            let r = XformQuantOptResult {
                coeff: xq.coeff,
                qcoeff: xq.qcoeff,
                dqcoeff: xq.dqcoeff,
                eob: xq.eob,
                txb_entropy_ctx: xq.txb_entropy_ctx,
                rate,
                txb_skip_ctx: txb_skip_ctx as usize,
                dc_sign_ctx: dc_sign_ctx as usize,
            };
            (r, rate)
        };

        // Early rate-only termination.
        if rdcost(inp.rdmult, rate_cost, 0) > best_rd {
            continue;
        }

        // Distortion.
        let (dist, sse): (i64, i64) = if res.eob == 0 {
            (block_sse, block_sse)
        } else if use_transform_domain_distortion {
            dist_block_tx_domain(&res.coeff, &res.dqcoeff, tx_size, inp.bd)
        } else {
            // Pixel-domain with the 64-pt / high-energy tx-domain hybrid.
            let high_energy_thresh = 128i64 * 128 * TX_SIZE_2D_TBL[tx_size];
            let is_high_energy = block_sse >= high_energy_thresh;
            let is_tx64 = tx_size == 4; // TX_64X64
            let mut d = i64::MAX;
            let mut s_tx = i64::MAX;
            let mut sse_diff = i64::MAX;
            if is_tx64 || is_high_energy {
                let (dt, st) = dist_block_tx_domain(&res.coeff, &res.dqcoeff, tx_size, inp.bd);
                d = dt;
                s_tx = st;
                sse_diff = block_sse - st;
            }
            if !is_tx64 || !is_high_energy || sse_diff * 2 < s_tx {
                let tx_domain_dist = d;
                d = dist_block_px_domain_interior(
                    &res.dqcoeff,
                    tx_size,
                    tx_type,
                    inp.pred,
                    inp.src,
                    inp.src_off,
                    inp.src_stride,
                    inp.bd,
                );
                if is_high_energy && d < tx_domain_dist {
                    d = tx_domain_dist;
                }
            } else {
                d += sse_diff;
            }
            (d, block_sse)
        };

        let rd = rdcost(inp.rdmult, rate_cost, dist);
        if rd < best_rd {
            best_rd = rd;
            best = Some(TxTypeSearchResult {
                best_tx_type: tx_type,
                best_eob: res.eob,
                best_txb_ctx: res.txb_entropy_ctx,
                rate: rate_cost,
                dist,
                sse,
                rd,
                skip_txfm: false, // set from best_eob below
                qcoeff: res.qcoeff,
                dqcoeff: res.dqcoeff,
                evaluated_mask: 0,
            });
        }

        // Early termination: current best much worse than the reference.
        if pol.adaptive_txb_search_level > 0
            && (best_rd - (best_rd >> pol.adaptive_txb_search_level)) > ref_best_rd
        {
            break;
        }
        // All-zero quantization break (speed >= 1; off at speed 0).
        if pol.skip_tx_search && best.as_ref().is_some_and(|b| b.best_eob == 0) {
            break;
        }
    }

    best.map(|mut b| {
        b.skip_txfm = b.best_eob == 0;
        b.evaluated_mask = evaluated_mask;
        debug_assert!(
            !calc_pixel_domain_distortion_final,
            "calc_pixel_domain_distortion_final is structurally off at speed 0",
        );
        b
    })
}

/// `dist_block_px_domain` (tx_search.c) for an INTERIOR txb: reconstruct
/// `pred + inv_txfm(dqcoeff)` and return `16 *` the variance-kernel SSE
/// (u32; bd-normalized like `aom_highbd_{10,12}_variance`) vs the source.
#[allow(clippy::too_many_arguments)]
pub fn dist_block_px_domain_interior(
    dqcoeff: &[i32],
    tx_size: usize,
    tx_type: usize,
    pred: &[u16],
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    bd: u8,
) -> i64 {
    let (w, h) = (TXS_W[tx_size], TXS_H[tx_size]);
    let mut recon = pred[..w * h].to_vec();
    aom_transform::inv_txfm2d::av1_inv_txfm2d_add(
        dqcoeff,
        &mut recon,
        w,
        tx_type,
        tx_size,
        i32::from(bd),
    );
    let (_var, sse) =
        aom_dist::highbd_variance(&src[src_off..], src_stride, &recon, w, w, h, bd);
    16 * i64::from(sse)
}

// ---------------------------------------------------------------------------
// av1_txfm_rd_in_plane + uniform_txfm_yrd (tx_search.c) — the per-tx-size
// evaluator: foreach-txb walk (predict-from-recon -> subtract ->
// search_tx_type -> recon feedback -> entropy-ctx threading) + the intra
// skip/no-skip + tx-size-rate RD assembly.
// ---------------------------------------------------------------------------

use crate::mode_costs::{block_signals_txsize, tx_size_cost, TxSizeCosts};
use aom_dist::highbd_subtract_block;
use aom_entropy::partition::intra_avail;
use aom_intra::predict_intra_high;

/// `RD_STATS` as this walk uses it (rate `i32::MAX` = invalid).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RdStats {
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub skip_txfm: bool,
}

impl RdStats {
    /// `av1_init_rd_stats` (zero, skip = 1).
    pub fn zero() -> Self {
        RdStats { rate: 0, dist: 0, sse: 0, skip_txfm: true }
    }
    /// `av1_merge_rd_stats` (rate saturates at `INT_MAX`).
    pub fn merge(&mut self, o: &RdStats) {
        if self.rate == i32::MAX || o.rate == i32::MAX {
            self.rate = i32::MAX;
            return;
        }
        self.rate = (i64::from(self.rate) + i64::from(o.rate)).min(i64::from(i32::MAX)) as i32;
        self.dist += o.dist;
        self.sse += o.sse;
        self.skip_txfm &= o.skip_txfm;
    }
}

const MI_SIZE_WIDE_B: [usize; 22] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
const MI_SIZE_HIGH_B: [usize; 22] =
    [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];
const BLK_W_B: [usize; 22] =
    [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
const BLK_H_B: [usize; 22] =
    [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];

/// The frame/block environment of one luma coding block's tx search — the
/// MACROBLOCK(D) state `block_rd_txfm` reads, expressed as plain data. The
/// mode fields are the CURRENT `mbmi` candidate under evaluation.
pub struct TxfmYrdEnv<'a> {
    // intra_avail frame geometry (see aom_entropy::partition::intra_avail).
    pub sb_size: usize,
    pub bsize: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub up_available: bool,
    pub left_available: bool,
    pub tile_col_end: i32,
    pub tile_row_end: i32,
    pub partition: usize,
    pub mi_cols: i32,
    pub mi_rows: i32,
    // Pixel planes: `recon[ref_off]` is the block's top-left in the
    // reconstruction plane (prediction reads + reconstruction writes);
    // `src[src_off]` in the source plane.
    pub ref_off: usize,
    pub ref_stride: usize,
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    // Prediction config.
    pub disable_edge_filter: bool,
    pub filter_type: i32,
    // Candidate mode.
    pub mode: usize,
    pub angle_delta: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub bd: u8,
    // Quantizer + RD.
    pub rows: &'a aom_quant::PlaneQuantRows<'a>,
    pub rdmult: i32,
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
    // Header rates: `skip_txfm_cost[skip_ctx][0/1]` (ctx = the
    // av1_get_skip_txfm_context facade, caller-supplied) and the tx-size cost
    // table + context (get_tx_size_context facade, caller-supplied).
    pub skip_costs: &'a [[i32; 2]; 3],
    pub skip_ctx: usize,
    pub tx_size_costs: &'a TxSizeCosts,
    pub tx_size_ctx: usize,
    /// `tx_mode_search_type == TX_MODE_SELECT` (speed-0 all-intra: true —
    /// USE_FULL_RD; see rdopt_utils.h select_tx_mode).
    pub tx_mode_is_select: bool,
    /// The block's above/left entropy contexts (`av1_get_entropy_contexts`
    /// copies these into the walk's working arrays).
    pub above_ctx: &'a [i8],
    pub left_ctx: &'a [i8],
}

/// One txb's winner within a walk (the `tx_type_map` / eob state the depth
/// loop snapshots for the winning size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxbWinner {
    pub tx_type: usize,
    pub eob: u16,
    pub txb_ctx: u8,
}

/// `av1_txfm_rd_in_plane` (tx_search.c) for a luma intra block at `tx_size`
/// (uniform): the `av1_foreach_transformed_block_in_plane` raster walk; per
/// txb predict-from-recon (`intra_avail` + `predict_intra_high` INTO the
/// recon plane, matching `av1_predict_intra_block_facade`'s in-place dst
/// write) -> subtract -> [`search_tx_type_intra`] -> reconstruct the winner
/// into `recon` (the `recon_intra` feedback the next txb predicts from) ->
/// entropy-context stamp. Interior blocks (`max_blocks_*` unclipped).
///
/// Returns `None` when the search exits early (`exit_early` — for intra ANY
/// early exit invalidates, tx_search.c:3786) or `current_rd > ref_best_rd` on
/// entry; `Some(stats)` otherwise. `recon` and the working contexts are
/// modified as the C does (the caller snapshots/restores between tx sizes).
#[allow(clippy::too_many_arguments)]
pub fn txfm_rd_in_plane_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    tx_size: usize,
    ref_best_rd: i64,
    current_rd_in: i64,
    pol: &TxTypeSearchPolicy,
) -> Option<(RdStats, Vec<TxbWinner>)> {
    if current_rd_in > ref_best_rd {
        return None;
    }
    let bsize = env.bsize;
    let (bw, bh) = (BLK_W_B[bsize], BLK_H_B[bsize]);
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let (txw_unit, txh_unit) = (txw >> 2, txh >> 2);
    let max_blocks_wide = MI_SIZE_WIDE_B[bsize];
    let max_blocks_high = MI_SIZE_HIGH_B[bsize];

    // av1_get_entropy_contexts: working copies of the neighbour contexts.
    let mut t_above: Vec<i8> = env.above_ctx[..max_blocks_wide].to_vec();
    let mut t_left: Vec<i8> = env.left_ctx[..max_blocks_high].to_vec();

    let mut stats = RdStats::zero();
    let mut winners: Vec<TxbWinner> = Vec::new();
    let mut current_rd = current_rd_in;
    let mut exit_early = false;

    let mut blk_row = 0usize;
    while blk_row < max_blocks_high {
        let mut blk_col = 0usize;
        while blk_col < max_blocks_wide {
            if exit_early {
                // C: the next block_rd_txfm call marks incomplete_exit; for
                // intra exit_early alone already invalidates.
                return None;
            }

            // av1_predict_intra_block_facade: predict INTO the recon plane.
            let (n_top, n_topright, n_left, n_bottomleft) = intra_avail(
                env.sb_size,
                bsize,
                env.mi_row,
                env.mi_col,
                env.up_available,
                env.left_available,
                env.tile_col_end,
                env.tile_row_end,
                env.partition,
                tx_size,
                0,
                0,
                blk_row as i32,
                blk_col as i32,
                bw as i32,
                bh as i32,
                env.mi_cols,
                env.mi_rows,
                env.mode,
                env.angle_delta * 3, // ANGLE_STEP
                env.use_filter_intra,
            );
            let txb_off = env.ref_off + (blk_row * env.ref_stride + blk_col) * 4;
            let mut pred = vec![0u16; txw * txh];
            predict_intra_high(
                recon,
                txb_off,
                env.ref_stride,
                &mut pred,
                txw,
                env.mode,
                env.angle_delta * 3,
                env.use_filter_intra,
                env.filter_intra_mode,
                env.disable_edge_filter,
                env.filter_type,
                tx_size,
                n_top as usize,
                n_topright,
                n_left as usize,
                n_bottomleft,
                env.bd as i32,
            );
            // The C facade writes the prediction into dst (the recon plane).
            for r in 0..txh {
                recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }

            // av1_subtract_txb.
            let src_txb_off = env.src_off + (blk_row * env.src_stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            highbd_subtract_block(
                txh,
                txw,
                &mut residual,
                txw,
                &env.src[src_txb_off..],
                env.src_stride,
                &pred,
                txw,
            );

            // search_tx_type over the neighbour ctx at this txb position.
            let bctx = crate::BlockContext {
                plane_bsize: bsize,
                plane: 0,
                above: &t_above[blk_col..],
                left: &t_left[blk_row..],
            };
            let inp = TxTypeSearchInputs {
                residual: &residual,
                src: env.src,
                src_off: src_txb_off,
                src_stride: env.src_stride,
                pred: &pred,
                tx_size,
                mode: env.mode,
                use_filter_intra: env.use_filter_intra,
                filter_intra_mode: env.filter_intra_mode,
                lossless: env.lossless,
                reduced_tx_set_used: env.reduced_tx_set_used,
                bd: env.bd,
                rows: env.rows,
                bctx: &bctx,
                rdmult: env.rdmult,
                coeff_costs: env.coeff_costs,
                tx_type_costs: env.tx_type_costs,
            };
            let win = search_tx_type_intra(&inp, pol, ref_best_rd - current_rd)
                .expect("search_tx_type always yields a winner");

            // recon_intra: reconstruct the winner on top of the prediction so
            // the next txb predicts from decoded pixels.
            if win.best_eob > 0 {
                let mut tight = pred.clone();
                aom_transform::inv_txfm2d::av1_inv_txfm2d_add(
                    &win.dqcoeff,
                    &mut tight,
                    txw,
                    win.best_tx_type,
                    tx_size,
                    i32::from(env.bd),
                );
                for r in 0..txh {
                    recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }

            winners.push(TxbWinner {
                tx_type: win.best_tx_type,
                eob: win.best_eob,
                txb_ctx: win.best_txb_ctx,
            });

            // av1_set_txb_context (interior: stamp the winner's entropy ctx).
            for a in t_above[blk_col..blk_col + txw_unit].iter_mut() {
                *a = win.best_txb_ctx as i8;
            }
            for l in t_left[blk_row..blk_row + txh_unit].iter_mut() {
                *l = win.best_txb_ctx as i8;
            }

            // Intra rd accumulation: signalled non-skip.
            let this = RdStats {
                rate: win.rate,
                dist: win.dist,
                sse: win.sse,
                skip_txfm: false,
            };
            stats.merge(&this);
            let rd = rdcost(env.rdmult, win.rate, win.dist);
            current_rd += rd;
            if current_rd > ref_best_rd {
                exit_early = true;
            }

            blk_col += txw_unit;
        }
        blk_row += txh_unit;
    }

    if exit_early {
        // Set on the LAST txb: intra still invalidates (invalid_rd =
        // args.exit_early for !is_inter).
        return None;
    }
    Some((stats, winners))
}

/// `uniform_txfm_yrd` (tx_search.c, intra arm): evaluate one uniform tx size
/// for the block — tx-size rate (under TX_MODE_SELECT on signaling blocks),
/// skip/no-skip header handling (intra: always signalled non-skip;
/// `skip_txfm_rd = INT64_MAX`), the [`txfm_rd_in_plane_intra`] walk, and the
/// final `RDCOST(rate + no_skip_rate + tx_size_rate, dist)` with
/// `rate += tx_size_rate`. Returns `(rd, Some(stats))` or
/// `(i64::MAX, None)` when invalid.
pub fn uniform_txfm_yrd_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    tx_size: usize,
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
) -> (i64, Option<(RdStats, Vec<TxbWinner>)>) {
    let tx_select = env.tx_mode_is_select && block_signals_txsize(env.bsize);
    let tx_size_rate = if tx_select {
        tx_size_cost(env.tx_size_costs, true, env.bsize, tx_size, env.tx_size_ctx)
    } else {
        0
    };
    let no_skip_txfm_rate = env.skip_costs[env.skip_ctx][0];
    // Intra: skip_txfm_rd = INT64_MAX; current_rd = no_this_rd.
    let no_this_rd = rdcost(env.rdmult, no_skip_txfm_rate + tx_size_rate, 0);

    let Some((mut stats, winners)) =
        txfm_rd_in_plane_intra(env, recon, tx_size, ref_best_rd, no_this_rd, pol)
    else {
        return (i64::MAX, None);
    };
    if stats.rate == i32::MAX {
        return (i64::MAX, None);
    }

    // Intra blocks are always signalled as non-skip.
    let rd = rdcost(env.rdmult, stats.rate + no_skip_txfm_rate + tx_size_rate, stats.dist);
    stats.rate += tx_size_rate;
    (rd, Some((stats, winners)))
}

// ---------------------------------------------------------------------------
// choose_tx_size_type_from_rd + av1_pick_uniform_tx_size_type_yrd
// (tx_search.c) — the uniform tx-size depth search, luma intra.
// ---------------------------------------------------------------------------

/// `max_txsize_rect_lookup[BLOCK_SIZES_ALL]` (common_data.h).
pub const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] =
    [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];
/// `sub_tx_size_map[TX_SIZES_ALL]` (common_data.h): rect sizes halve the LONG
/// side.
pub const SUB_TX_SIZE_MAP: [usize; 19] =
    [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
/// `MAX_TX_DEPTH` (blockd.h).
pub const MAX_TX_DEPTH: i32 = 2;

/// `get_search_init_depth` (tx_search.c) for intra at speed-0 all-intra:
/// `intra_tx_size_search_init_depth_sqr = 1` (speed-0 allintra block),
/// `intra_tx_size_search_init_depth_rect = 0` (default),
/// `tx_size_search_lgr_block = 0` (default). USE_FULL_RD (never LARGESTALL
/// here — that arm returns MAX_VARTX_DEPTH upstream).
pub fn get_search_init_depth_intra_speed0(mi_width: usize, mi_height: usize) -> i32 {
    if mi_height != mi_width {
        0 // intra_tx_size_search_init_depth_rect
    } else {
        1 // intra_tx_size_search_init_depth_sqr
    }
}

/// The outcome of the depth search: the chosen uniform tx size, its per-txb
/// winners (the `tx_type_map` snapshot), and the block RD stats.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxSizeChoice {
    pub best_tx_size: usize,
    pub best_rd: i64,
    pub stats: RdStats,
    pub winners: Vec<TxbWinner>,
}

/// `choose_tx_size_type_from_rd` (tx_search.c, static) — the uniform-tx-size
/// depth sweep for a luma intra block under `TX_MODE_SELECT` at the speed-0
/// policy: start at `max_txsize_rect_lookup[bsize]` skipped down by
/// `init_depth` steps, evaluate each depth with [`uniform_txfm_yrd_intra`],
/// keep the strict-min, stop at TX_4X4 or on the low-contrast
/// (`source_variance < 256`) regression prune.
///
/// Speed-0 resolution (each named): `init_depth` per
/// [`get_search_init_depth_intra_speed0`]; `enable_tx64`/`enable_rect_tx`
/// CLI-default ON (the disable arms are `continue`s, kept via params);
/// `use_rd_based_breakout_for_intra_tx_search = false` (rd_thresh is always
/// `ref_best_rd`); `prune_intra_tx_depths_using_nn = false` (no NN pruning);
/// `x->rd_model = FULL_TXFM_RD`.
///
/// `source_variance` is `x->source_variance` (the source block's per-pixel
/// variance, computed upstream by the encoder — caller-supplied).
/// Returns `None` when no depth produced a valid RD (rate stays `INT_MAX`).
#[allow(clippy::too_many_arguments)]
pub fn choose_tx_size_type_from_rd_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
    source_variance: u32,
    enable_tx64: bool,
    enable_rect_tx: bool,
) -> Option<TxSizeChoice> {
    let bsize = env.bsize;
    let max_rect_tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
    // tx_select is TX_MODE_SELECT here (the LARGESTALL/ONLY_4X4 arms are the
    // caller's — see av1_pick_uniform_tx_size_type_yrd).
    let mut start_tx = max_rect_tx_size;
    let init_depth = get_search_init_depth_intra_speed0(
        MI_SIZE_WIDE_B[bsize],
        MI_SIZE_HIGH_B[bsize],
    );
    if init_depth == MAX_TX_DEPTH && !enable_tx64 && TXSIZE_SQR_UP_MAP[start_tx] == 4 {
        start_tx = SUB_TX_SIZE_MAP[start_tx];
    }

    let mut best: Option<TxSizeChoice> = None;
    let mut best_rd = i64::MAX;
    let mut rd = [i64::MAX; MAX_TX_DEPTH as usize + 1];
    let mut tx_size = start_tx;
    let mut depth = init_depth;
    while depth <= MAX_TX_DEPTH {
        if (!enable_tx64 && TXSIZE_SQR_UP_MAP[tx_size] == 4)
            || (!enable_rect_tx && TXS_W[tx_size] != TXS_H[tx_size])
        {
            depth += 1;
            tx_size = SUB_TX_SIZE_MAP[tx_size];
            continue;
        }
        // use_rd_based_breakout_for_intra_tx_search = false => ref_best_rd.
        let (this_rd, res) = uniform_txfm_yrd_intra(env, recon, tx_size, ref_best_rd, pol);
        rd[depth as usize] = this_rd;
        if this_rd < best_rd {
            let (stats, winners) = res.expect("valid rd implies stats");
            best_rd = this_rd;
            best = Some(TxSizeChoice { best_tx_size: tx_size, best_rd, stats, winners });
        }
        if tx_size == 0 {
            break; // TX_4X4
        }
        // Low-contrast regression prune across the two searched depths.
        if depth > init_depth && depth != MAX_TX_DEPTH && source_variance < 256 {
            let prev = rd[depth as usize - 1];
            if prev != i64::MAX && rd[depth as usize] > prev {
                break;
            }
        }
        depth += 1;
        tx_size = SUB_TX_SIZE_MAP[tx_size];
    }
    best
}

/// `av1_pick_uniform_tx_size_type_yrd` (tx_search.c) — the luma intra slice:
/// residue hashing and skip-prediction are inter-only; lossless picks the
/// smallest (4x4) transform (`choose_smallest_tx_size` = one
/// [`uniform_txfm_yrd_intra`] at TX_4X4); `USE_FULL_RD` runs the depth sweep.
/// (`USE_LARGESTALL` / winner-mode arms are out of scope at speed-0
/// MODE_EVAL.) Returns the chosen size + stats, or `None` (rate `INT_MAX`).
#[allow(clippy::too_many_arguments)]
pub fn pick_uniform_tx_size_type_yrd_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
    source_variance: u32,
    enable_tx64: bool,
    enable_rect_tx: bool,
) -> Option<TxSizeChoice> {
    // select_tx_mode (rdopt_utils.h) couples the two at frame level:
    // coded_lossless => ONLY_4X4, i.e. never TX_MODE_SELECT.
    debug_assert!(
        !(env.lossless && env.tx_mode_is_select),
        "lossless implies ONLY_4X4 (tx_mode_is_select must be false)",
    );
    if env.lossless {
        // choose_smallest_tx_size: evaluate TX_4X4 only.
        let (rd, res) = uniform_txfm_yrd_intra(env, recon, 0, ref_best_rd, pol);
        return res.map(|(stats, winners)| TxSizeChoice {
            best_tx_size: 0,
            best_rd: rd,
            stats,
            winners,
        });
    }
    choose_tx_size_type_from_rd_intra(
        env,
        recon,
        ref_best_rd,
        pol,
        source_variance,
        enable_tx64,
        enable_rect_tx,
    )
}
