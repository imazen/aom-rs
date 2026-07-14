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
