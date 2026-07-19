//! Inter / IntraBC variable-transform coeff arm — `av1_pick_recursive_tx_size
//! _type_yrd` (tx_search.c:3553) and its recursion (`select_tx_size_and_type`
//! / `select_tx_block` / `try_tx_block_no_split` / `try_tx_block_split`), plus
//! the inter per-txb leaf (`search_tx_type`, inter arm).
//!
//! Scope: the KB-15 intrabc witness config — ALLINTRA speed-0, bd8, screen KEY,
//! qidx 192 (cq48). IntraBC blocks are `is_inter_block == true`, so
//! `av1_txfm_search` routes them through this recursive var-tx path (not the
//! uniform intra path). The inter residual is FIXED up front (`recon_intra` is
//! `!is_inter`-gated, tx_search.c:930) — every leaf's tx-type RD is independent,
//! reading a static per-block `src_diff`.
//!
//! Leaf reuse: the forward-tx / quant / trellis / coeff-rate / distortion
//! primitives are shared with the intra leaf (`xform_quant`,
//! `xform_quant_optimize`, `cost_coeffs_txb`, `dist_block_*`); the inter leaf
//! differs ONLY in the tx-mask (the inter ext-tx set), the `is_inter = true`
//! tx-type cost, and the trellis rd-mult (16 vs intra's 17,
//! `plane_rd_mult[is_inter=1][luma]`, encodetxb.h:270).

use crate::rd::rdcost;
use crate::tx_search::{
    AV1_EXT_TX_USED_FLAG, DCT_ADST_TX_MASK, TX_SIZE_2D_TBL, av1_pixel_diff_dist, dist_block_px_domain,
};
use crate::{
    BlockContext, OptimizeInputs, QuantKind, QuantParams, XformQuantOptResult, dist_block_tx_domain_qm,
    dist_qmatrix, xform_quant, xform_quant_optimize,
};
use aom_dsp::txb::{
    CoeffCostTables, TxTypeCosts, cost_coeffs_txb, ext_tx_set_type, get_tx_type_cost, get_txb_ctx,
    scan,
};

/// `ROUND_POWER_OF_TWO` for i64 (local copy of the private `tx_search` helper).
#[inline]
fn round_power_of_two_i64(value: i64, n: i32) -> i64 {
    (value + ((1i64 << n) >> 1)) >> n
}

/// `TX_TYPES` (common/enums.h).
const TX_TYPES: usize = 16;

// Transform-size dimension tables (pixels) — `tx_size_wide` / `tx_size_high`,
// indexed by TX_SIZE. Mirrors `TXS_W` / `TXS_H` in `tx_search.rs`.
const TXS_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TXS_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
/// `txsize_sqr_up_map[TX_SIZE]` — the square TX_SIZE whose side == max(w,h).
const TXSIZE_SQR_UP_MAP: [usize; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];

/// `plane_rd_mult[is_inter=1][PLANE_TYPE_Y] = 16` (encodetxb.h:270 — the inter
/// row; `plane_rd_mult_chroma[1][0]` is also 16, so the speed-0 allintra
/// `use_chroma_trellis_rd_mult` sf is a no-op for inter luma just as for intra).
/// The intra analogue is 17 (`trellis_rdmult_intra_y`, tx_search.rs). `rshift`
/// is 5 for the PSNR-family tunes (7 for tune=IQ/SSIMULACRA2). sharpness 0 at
/// speed 0.
#[inline]
pub fn trellis_rdmult_inter_y(rdmult: i32, sharpness: i32, bd: u8, iq_tuning: bool) -> i64 {
    round_power_of_two_i64(
        (rdmult as i64) * ((8 - sharpness) as i64) * ((16i64) << (2 * (bd as i32 - 8))),
        if iq_tuning { 7 } else { 5 },
    )
}

/// `plane_rd_mult_chroma[is_inter=1][PLANE_TYPE_UV] = 10` (encodetxb.h:266-269)
/// selected by the allintra speed-0 `use_chroma_trellis_rd_mult = 1`
/// (speed_features.c:370); without the sf it would be `plane_rd_mult[1][1] = 20`.
/// Same `rshift` / `(8 - sharpness)` / `<< 2*(bd-8)` shape as the luma form
/// ([`trellis_rdmult_inter_y`]) — txb_rdopt.c:381-393.
#[inline]
pub fn trellis_rdmult_inter_uv(rdmult: i32, sharpness: i32, bd: u8, iq_tuning: bool) -> i64 {
    round_power_of_two_i64(
        (rdmult as i64) * ((8 - sharpness) as i64) * ((10i64) << (2 * (bd as i32 - 8))),
        if iq_tuning { 7 } else { 5 },
    )
}

/// `get_tx_mask` (tx_search.c:1776) — the CHROMA (`plane != 0`) arm. C pins
/// `txk_allowed = uv_tx_type` (the co-located luma type, tx_search.c:1841-1847)
/// so the mask collapses to a single bit (:1872-1874) — chroma NEVER searches
/// tx types (the full-set branch asserts `plane == 0`, :1882). The DCT-only
/// override (:1862-1868) and the flip/idtx strip (:1870-1871) still apply, and
/// the empty-mask fallback restores the **uv type** (not DCT_DCT) for `plane != 0`.
///
/// Returns `(allowed_tx_mask, txk_allowed)`.
pub fn get_tx_mask_inter_uv(
    tx_size: usize,
    uv_tx_type: usize,
    lossless: bool,
    reduced_tx_set_used: bool,
    enable_flip_idtx: bool,
    use_inter_dct_only: bool,
) -> (u16, usize) {
    let tx_set_type = ext_tx_set_type(tx_size, true, reduced_tx_set_used);
    let mut ext_tx_used_flag = AV1_EXT_TX_USED_FLAG[tx_set_type];
    let mut txk_allowed = uv_tx_type;

    if lossless || TXSIZE_SQR_UP_MAP[tx_size] > 3 || ext_tx_used_flag == 0x0001 || use_inter_dct_only
    {
        txk_allowed = 0; // DCT_DCT
    }
    if !enable_flip_idtx {
        ext_tx_used_flag &= DCT_ADST_TX_MASK;
    }

    let mut allowed_tx_mask = (1u16 << txk_allowed) & ext_tx_used_flag;
    if allowed_tx_mask == 0 {
        // `txk_allowed = (plane ? uv_tx_type : DCT_DCT)` — chroma restores the
        // inherited type here, unlike the luma DCT_DCT fallback.
        txk_allowed = uv_tx_type;
        allowed_tx_mask = 1 << txk_allowed;
    }
    (allowed_tx_mask, txk_allowed)
}

/// `get_tx_mask` (tx_search.c:1776) — the INTER (is_inter) arm, at the speed-0
/// DEFAULT_EVAL config (intrabc): `default_inter_tx_type_prob_thresh == INT_MAX`
/// (no forced tx type), `prune_tx_type_using_stats == 0`, `rd_model ==
/// FULL_TXFM_RD`, `use_reduced_intra_txset` is intra-set-only (inert). So the
/// mask is the full inter ext-tx set `av1_ext_tx_used_flag[tx_set_type]`, DCT-only
/// at lossless / sqr_up>TX_32X32 / `use_inter_dct_only`, flip/idtx stripped when
/// `!enable_flip_idtx`. The `prune_tx_2D` NN reorder/prune (fires for
/// `num_allowed > 5`) is applied by the CALLER after this — see [`search_tx_type_inter`].
///
/// Returns `(allowed_tx_mask, txk_allowed)` where `txk_allowed = Some(t)` pins
/// the single allowed type.
pub fn get_tx_mask_inter(
    tx_size: usize,
    lossless: bool,
    reduced_tx_set_used: bool,
    enable_flip_idtx: bool,
    use_inter_dct_only: bool,
) -> (u16, Option<usize>) {
    let mut txk_allowed = TX_TYPES; // "all"
    let tx_set_type = ext_tx_set_type(tx_size, true, reduced_tx_set_used);
    let mut ext_tx_used_flag = AV1_EXT_TX_USED_FLAG[tx_set_type];

    if lossless || TXSIZE_SQR_UP_MAP[tx_size] > 3 || ext_tx_used_flag == 0x0001 || use_inter_dct_only
    {
        txk_allowed = 0; // DCT_DCT
    }
    if !enable_flip_idtx {
        ext_tx_used_flag &= DCT_ADST_TX_MASK;
    }

    let mut allowed_tx_mask: u16;
    if txk_allowed < TX_TYPES {
        allowed_tx_mask = (1 << txk_allowed) & ext_tx_used_flag;
    } else {
        // The multi-type inter arm (tx_search.c:1881, plane 0). At speed 0
        // `prune_tx_type_using_stats == 0` (inert), `prune_tx_type_est_rd`
        // is speed>=4 (inert) — so `allowed_tx_mask` is the full set here and
        // the only active prune is `prune_tx_2D` (applied by the caller).
        allowed_tx_mask = ext_tx_used_flag;
    }

    if allowed_tx_mask == 0 {
        txk_allowed = 0; // DCT_DCT (plane 0)
        allowed_tx_mask = 1 << txk_allowed;
    }

    let single = if txk_allowed < TX_TYPES {
        Some(txk_allowed)
    } else {
        None
    };
    debug_assert!(single.is_none_or(|t| allowed_tx_mask == 1 << t));
    (allowed_tx_mask, single)
}

/// Per-txb inputs for [`search_tx_type_inter`] — one leaf of the var-tx
/// quadtree. The residual/pred are CONTIGUOUS `TXS_W x TXS_H` (stride `TXS_W`);
/// the caller extracts them from the whole-block `src_diff` / prediction at the
/// txb offset. `src`/`src_off`/`src_stride` reference the source plane (for the
/// pixel-domain distortion reconstruct).
pub struct InterLeafInputs<'a> {
    pub residual: &'a [i16],
    pub pred: &'a [u16],
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    pub tx_size: usize,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub enable_flip_idtx: bool,
    pub use_inter_dct_only: bool,
    pub bd: u8,
    pub rows: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    /// Neighbour entropy contexts (`get_txb_ctx` inputs).
    pub bctx: &'a BlockContext<'a>,
    pub rdmult: i32,
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
    /// Frame-edge visible txb extent (`get_txb_visible_dimensions`).
    pub visible_cols: usize,
    pub visible_rows: usize,
    /// CHROMA (`bctx.plane > 0`) only: the derived `uv_tx_type` that
    /// `get_tx_mask`'s `if (plane)` arm pins (tx_search.c:1841-1847 — chroma
    /// inherits the co-located LUMA tx type via `av1_get_tx_type`,
    /// blockd.h:1296-1301). `None` = the luma multi-type arm
    /// ([`get_tx_mask_inter`]). C asserts `plane == 0` on the full-set branch
    /// (tx_search.c:1882), so chroma ALWAYS carries `Some`.
    pub forced_uv_tx_type: Option<usize>,
    /// Frame QM level (`qmatrix_level_y`), `None` = QM off.
    pub qm_level: Option<usize>,
    /// `prune_2d_txfm_mode >= TX_TYPE_PRUNE_1` — enable the `prune_tx_2D` NN prune
    /// (witness config: true). When true and the multi-type inter arm has >5
    /// candidates, the NN prunes + reorders the tx-type search order.
    pub prune_2d: bool,
}

/// The winner of one inter leaf's tx-type search (`search_tx_type`, inter arm).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterLeafResult {
    pub best_tx_type: usize,
    pub best_eob: u16,
    pub best_txb_ctx: u8,
    /// `get_txb_ctx`'s pair for the WINNING type, from the pre-write neighbour
    /// contexts (the same pair the rate used). The coded-bytes writer re-derives
    /// its own pair from the PERSISTENT arrays at the tokenize read point
    /// (KB-6's write-ctx root) — these are the SEARCH-side values.
    pub txb_skip_ctx: usize,
    pub dc_sign_ctx: usize,
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub rd: i64,
    pub skip_txfm: bool,
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    pub evaluated_mask: u16,
}

/// `search_tx_type` (tx_search.c:2079) — the INTER (is_inter) arm, for one
/// var-tx leaf at the speed-0 DEFAULT_EVAL config. Mirrors the intra leaf
/// (`search_tx_type_intra`) arithmetic exactly — block_sse/mse, the trellis MSE
/// gate, the pixel-domain / high-energy-tx-domain hybrid distortion, the
/// strict-min RD with the `adaptive_txb_search_level` early break — differing
/// ONLY in: the mask ([`get_tx_mask_inter`] + `prune_tx_2D`), `is_inter = true`
/// tx-type cost, and the trellis rd-mult ([`trellis_rdmult_inter_y`]). No
/// `recon_intra` feedback (inter residual is fixed); `predict_dc_level = 0`,
/// palette / filter-intra absent, `use_transform_domain_distortion = 0` and
/// `skip_tx_search = 0` at this config.
///
/// `adaptive_txb_search_level` (=1 at the witness config) drives the in-loop
/// early break (`best_rd - (best_rd >> level) > ref_best_rd`).
pub fn search_tx_type_inter(
    inp: &InterLeafInputs,
    sharpness: i32,
    iq_tuning: bool,
    coeff_opt_dist_threshold: u32,
    adaptive_txb_search_level: i32,
    ref_best_rd: i64,
) -> Option<InterLeafResult> {
    let tx_size = inp.tx_size;
    let w = TXS_W[tx_size];
    let hbd = inp.bd > 8;

    let dequant_shift = if hbd { inp.bd as i32 - 5 } else { 3 };
    let qstep = (i32::from(inp.rows.dequant[1]) >> dequant_shift) as u32;

    // block_sse / block_mse over the visible txb (== full for interior txbs).
    let (mut block_sse_u, mut block_mse_q8) =
        av1_pixel_diff_dist(inp.residual, w, 0, 0, inp.visible_cols, inp.visible_rows);
    let mut block_sse = block_sse_u as i64;
    if hbd {
        let s = 2 * (inp.bd as i32 - 8);
        block_sse = (block_sse + ((1i64 << s) >> 1)) >> s;
        block_mse_q8 = (((block_mse_q8 as u64) + ((1u64 << s) >> 1)) >> s) as u32;
        block_sse_u = block_sse as u64;
    }
    let _ = block_sse_u;
    block_sse *= 16;

    // The allowed tx-type set (get_tx_mask inter arm; chroma pins one type).
    let (mut allowed_tx_mask, txk_allowed) = match inp.forced_uv_tx_type {
        Some(uv) => {
            let (m, t) = get_tx_mask_inter_uv(
                tx_size,
                uv,
                inp.lossless,
                inp.reduced_tx_set_used,
                inp.enable_flip_idtx,
                inp.use_inter_dct_only,
            );
            (m, Some(t))
        }
        None => get_tx_mask_inter(
            tx_size,
            inp.lossless,
            inp.reduced_tx_set_used,
            inp.enable_flip_idtx,
            inp.use_inter_dct_only,
        ),
    };
    // Search order: the natural 0..16 unless prune_tx_2D reorders it.
    let mut txk_map: [usize; TX_TYPES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    // `prune_tx_2D` (tx_search.c:1934): the multi-type inter arm, when the set has
    // more than `allowed_tx_count` (5 at PRUNE_1) candidates — the NN prunes the
    // mask + reorders `txk_map` (with TX_TYPE_INVALID padding after the kept types).
    if inp.prune_2d && txk_allowed.is_none() && allowed_tx_mask.count_ones() > 5 {
        let set = ext_tx_set_type(tx_size, true, inp.reduced_tx_set_used);
        if let Some(r) =
            crate::prune_tx_2d::prune_tx_2d(inp.residual, w, tx_size, set, 1, allowed_tx_mask)
        {
            allowed_tx_mask = r.allowed_tx_mask;
            txk_map = r.txk_map;
        }
    }

    // Trellis gating: block-MSE / qstep^2 threshold (perform_block_coeff_opt).
    let perform_block_coeff_opt =
        (block_mse_q8 as u64) <= (coeff_opt_dist_threshold as u64) * (qstep as u64) * (qstep as u64);
    let skip_trellis = !perform_block_coeff_opt;

    // av1_setup_quant: FP with trellis, B without (USE_B_QUANT_NO_TRELLIS=1).
    let kind = if skip_trellis {
        QuantKind::B
    } else {
        QuantKind::Fp
    };
    let mut qp = QuantParams::from_plane_rows(inp.rows, kind, inp.bd, inp.lossless);
    if let Some(level) = inp.qm_level {
        qp = qp.with_qm(level, 0);
    }
    // `plane_rd_mult_chroma[is_inter][plane_type]` under the allintra speed-0
    // `use_chroma_trellis_rd_mult = 1` (speed_features.c:370): inter luma 16
    // (== `plane_rd_mult[1][0]`, so the sf is a no-op there) but inter chroma
    // **10** (vs 20 without the sf) — encodetxb.h:266-273, txb_rdopt.c:387-393.
    let trellis_rdmult = if inp.bctx.plane > 0 {
        trellis_rdmult_inter_uv(inp.rdmult, sharpness, inp.bd, iq_tuning)
    } else {
        trellis_rdmult_inter_y(inp.rdmult, sharpness, inp.bd, iq_tuning)
    };
    let opt = OptimizeInputs {
        cost: inp.coeff_costs,
        rdmult: trellis_rdmult,
        sharpness,
    };

    let mut best: Option<InterLeafResult> = None;
    let mut best_rd = i64::MAX;
    let mut evaluated_mask = 0u16;

    for idx in 0..TX_TYPES {
        let tx_type = txk_map[idx];
        // prune_tx_2D pads the reordered txk_map with TX_TYPE_INVALID (255).
        if tx_type >= TX_TYPES {
            continue;
        }
        if allowed_tx_mask & (1 << tx_type) == 0 {
            continue;
        }
        evaluated_mask |= 1 << tx_type;

        // Forward transform + quantize (+ trellis + rate). At speed 0
        // `coeff_opt_satd_threshold == UINT_MAX` so the SATD trellis-skip is a
        // no-op (skip_trellis is decided by the MSE gate above).
        let (res, rate_cost): (XformQuantOptResult, i32) = if !skip_trellis {
            let r = xform_quant_optimize(inp.residual, tx_size, tx_type, kind, &qp, inp.bctx, &opt);
            let ttc = if r.eob > 0 {
                get_tx_type_cost(
                    inp.tx_type_costs,
                    inp.bctx.plane,
                    tx_size,
                    tx_type,
                    true,
                    inp.reduced_tx_set_used,
                    inp.lossless,
                    false,
                    0,
                    0,
                )
            } else {
                0
            };
            let rate = r.rate + ttc;
            (r, rate)
        } else {
            let xq = xform_quant(inp.residual, tx_size, tx_type, kind, &qp, false);
            let (txb_skip_ctx, dc_sign_ctx) = get_txb_ctx(
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
                    inp.bctx.plane,
                    tx_size,
                    tx_type,
                    true,
                    inp.reduced_tx_set_used,
                    inp.lossless,
                    false,
                    0,
                    0,
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

        // Distortion: at this config `use_transform_domain_distortion = 0`, so
        // the pixel-domain / high-energy-tx-domain hybrid (the eob==0 shortcut,
        // and for tx64 / high-energy the tx-domain fallback).
        let dqm = dist_qmatrix(&qp, tx_size, tx_type);
        let dscan = scan(tx_size, tx_type);
        let (dist, sse): (i64, i64) = if res.eob == 0 {
            (block_sse, block_sse)
        } else {
            let high_energy_thresh = 128i64 * 128 * TX_SIZE_2D_TBL[tx_size];
            let is_high_energy = block_sse >= high_energy_thresh;
            let is_tx64 = tx_size == 4; // TX_64X64
            let mut d = i64::MAX;
            let mut s_tx = i64::MAX;
            let mut sse_diff = i64::MAX;
            if is_tx64 || is_high_energy {
                let (dt, st) =
                    dist_block_tx_domain_qm(&res.coeff, &res.dqcoeff, tx_size, inp.bd, dqm, dscan, false);
                d = dt;
                s_tx = st;
                sse_diff = block_sse - st;
            }
            if !is_tx64 || !is_high_energy || sse_diff * 2 < s_tx {
                let tx_domain_dist = d;
                d = dist_block_px_domain(
                    &res.dqcoeff,
                    tx_size,
                    tx_type,
                    inp.pred,
                    inp.src,
                    inp.src_off,
                    inp.src_stride,
                    inp.bd,
                    inp.visible_cols,
                    inp.visible_rows,
                    res.eob as usize,
                    inp.lossless,
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
            best = Some(InterLeafResult {
                best_tx_type: tx_type,
                best_eob: res.eob,
                best_txb_ctx: res.txb_entropy_ctx,
                txb_skip_ctx: res.txb_skip_ctx,
                dc_sign_ctx: res.dc_sign_ctx,
                rate: rate_cost,
                dist,
                sse,
                rd,
                skip_txfm: false,
                qcoeff: res.qcoeff,
                dqcoeff: res.dqcoeff,
                evaluated_mask: 0,
            });
        }

        // adaptive_txb_search_level early break (tx_search.c:2353-2357).
        if adaptive_txb_search_level > 0
            && (best_rd - (best_rd >> adaptive_txb_search_level)) > ref_best_rd
        {
            break;
        }
        // skip_tx_search == 0 at this config (no all-zero-quant break).
    }

    best.map(|mut b| {
        b.skip_txfm = b.best_eob == 0;
        b.evaluated_mask = evaluated_mask;
        b
    })
}

// ===========================================================================
// Recursive var-tx size+type search — select_tx_block quadtree
// (tx_search.c:2601 / :2406 / :2454 / :3433 / :3553).
// ===========================================================================

use crate::tx_search::get_mean_dev_features;
use crate::tx_split_nn_weights::TX_SPLIT_NN;
use aom_dsp::entropy::partition::{txfm_partition_context, txfm_partition_update};
use aom_dsp::txb::CoeffCostSet;

/// `ml_predict_tx_split` (tx_search.c:1755) — the split-prediction NN. Returns
/// `clamp((int)(score * 10000), -80000, 80000)`, or `-1` when no NN exists for
/// this tx size (TX_4X4). `diff` is the WHOLE-block src_diff; the NN reads the
/// txb's sub-block at `(4*blk_row, 4*blk_col)` (stride `diff_stride`). Faithful
/// to `av1_nn_predict` (node-major w0, ReLU hidden, linear 1-output) +
/// `av1_nn_output_prec_reduce` (reduce_prec=1: `((int)(x*512 + 0.5))/512`, the
/// `+0.5` a f64 literal). Features come from [`get_mean_dev_features`] (no
/// normalization — unlike the intra tx-depth NN).
pub fn ml_predict_tx_split(
    diff: &[i16],
    diff_stride: usize,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
) -> i32 {
    let Some(nn) = TX_SPLIT_NN[tx_size].as_ref() else {
        return -1;
    };
    let (bw, bh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let off = 4 * blk_row * diff_stride + 4 * blk_col;
    let mut features = [0f32; 16];
    let n = get_mean_dev_features(&diff[off..], diff_stride, bw, bh, &mut features);
    debug_assert_eq!(n, nn.num_inputs, "feature count vs nn.num_inputs");

    // Hidden layer (ReLU): buf[node] = relu(bias[node] + Σ_i w0[node*ni + i]*feat[i]).
    let mut hidden = [0f32; 64];
    for node in 0..nn.num_hidden {
        let mut val = nn.b0[node];
        for i in 0..nn.num_inputs {
            val += nn.w0[node * nn.num_inputs + i] * features[i];
        }
        hidden[node] = if val > 0.0 { val } else { 0.0 };
    }
    // Output layer (1 output, linear): out = b1 + Σ_i w1[i]*hidden[i].
    let mut out = nn.b1;
    for i in 0..nn.num_hidden {
        out += nn.w1[i] * hidden[i];
    }
    // av1_nn_output_prec_reduce (reduce_prec=1): `((int)(out*512 + 0.5)) * (1/512)`.
    // C multiplies in f32 (`float * int`) then promotes to f64 for the `+ 0.5`
    // (a double literal); `1/512` is exact in f32 so `* inv_prec == / 512`.
    let scaled = out * 512.0f32;
    let out = ((f64::from(scaled) + 0.5) as i32) as f32 / 512.0f32;
    // `(int)(score * 10000)` — f32 multiply (10000 is an int in C), trunc-to-zero.
    let int_score = (out * 10000.0f32) as i32;
    int_score.clamp(-80000, 80000)
}

/// `sub_tx_size_map[TX_SIZES_ALL]` (common_data.h): one var-tx split step.
pub const SUB_TX_SIZE_MAP: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
/// `tx_size_wide_unit` / `tx_size_high_unit` (4x4 units).
pub const TX_SIZE_WIDE_UNIT: [usize; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
pub const TX_SIZE_HIGH_UNIT: [usize; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];
/// `max_txsize_rect_lookup[BLOCK_SIZES_ALL]` (common_data.h) — the block's
/// largest rectangular tx size (the var-tx quadtree root).
const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] =
    [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];
/// `MAX_VARTX_DEPTH` (enums.h:56).
const MAX_VARTX_DEPTH: i32 = 2;

/// `av1_get_txb_size_index` (blockd.h) — the `mbmi->inter_tx_size[]` index for
/// a txb at (blk_row, blk_col). Copy of the private `aom_dsp::entropy` helper.
pub fn get_txb_size_index(bsize: usize, blk_row: usize, blk_col: usize) -> usize {
    const TW_W_LOG2: [usize; 22] = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0, 1, 1, 2, 2, 3];
    const TW_H_LOG2: [usize; 22] = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 0, 2, 1, 3, 2];
    const STRIDE_LOG2: [usize; 22] = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1];
    ((blk_row >> TW_H_LOG2[bsize]) << STRIDE_LOG2[bsize]) + (blk_col >> TW_W_LOG2[bsize])
}

/// `RD_STATS` (rd.h) — the subset the var-tx recursion accumulates.
#[derive(Clone, Copy, Debug)]
struct RdStats {
    /// `INT_MAX` (== i32::MAX) marks the stats invalid (`av1_invalid_rd_stats`).
    rate: i32,
    dist: i64,
    sse: i64,
    skip_txfm: bool,
    zero_rate: i32,
}
impl RdStats {
    /// `av1_init_rd_stats`.
    fn init() -> Self {
        Self { rate: 0, dist: 0, sse: 0, skip_txfm: true, zero_rate: 0 }
    }
    /// `av1_invalid_rd_stats`.
    fn invalid() -> Self {
        Self { rate: i32::MAX, dist: i64::MAX, sse: i64::MAX, skip_txfm: false, zero_rate: 0 }
    }
    fn is_invalid(&self) -> bool {
        self.rate == i32::MAX
    }
    /// `av1_merge_rd_stats`.
    fn merge(&mut self, src: &RdStats) {
        if self.rate == i32::MAX || src.rate == i32::MAX {
            *self = RdStats::invalid();
            return;
        }
        self.rate = ((self.rate as i64) + (src.rate as i64)).min(i32::MAX as i64) as i32;
        if self.zero_rate == 0 {
            self.zero_rate = src.zero_rate;
        }
        self.dist += src.dist;
        if self.sse < i64::MAX && src.sse < i64::MAX {
            self.sse += src.sse;
        }
        self.skip_txfm &= src.skip_txfm;
    }
}

/// One coded leaf of the winning var-tx tree (for the pack coeff write + recon).
#[derive(Clone, Debug)]
pub struct VarTxLeaf {
    pub blk_row: usize,
    pub blk_col: usize,
    pub tx_size: usize,
    pub tx_type: usize,
    pub eob: u16,
    pub txb_ctx: u8,
    pub skip_txfm: bool,
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
}

/// Static context for the var-tx recursion over ONE inter/intrabc luma block.
/// The residual/pred are the WHOLE-block buffers (stride in pixels); the leaves
/// slice per txb. `above_ctx`/`left_ctx` are the block's initial ENTROPY_CONTEXT
/// neighbours; `tx_above`/`tx_left` the initial TXFM_CONTEXT neighbours.
pub struct VarTxEnv<'a> {
    pub bsize: usize,
    /// Block extent in 4x4 tx units, clipped to the frame edge (`max_block_*`).
    pub max_blocks_wide: usize,
    pub max_blocks_high: usize,
    pub residual: &'a [i16],
    pub residual_stride: usize,
    pub pred: &'a [u16],
    pub pred_stride: usize,
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub enable_flip_idtx: bool,
    pub use_inter_dct_only: bool,
    pub bd: u8,
    pub rows: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    pub rdmult: i32,
    pub coeff_costs: &'a CoeffCostSet,
    pub tx_type_costs: &'a TxTypeCosts,
    pub qm_level: Option<usize>,
    /// `mode_costs.txfm_partition_cost[ctx][0/1]` (21 contexts).
    pub txfm_partition_cost: &'a [[i32; 2]; 21],
    /// `mode_costs.skip_txfm_cost[skip_ctx][0/1]`.
    pub skip_txfm_cost: [i32; 2],
    /// Initial ENTROPY_CONTEXT neighbours (`av1_get_entropy_contexts`), length
    /// `max_blocks_{wide,high}`.
    pub above_ctx: &'a [i8],
    pub left_ctx: &'a [i8],
    /// Initial TXFM_CONTEXT neighbours (`xd->above/left_txfm_context`), same length.
    pub tx_above: &'a [u8],
    pub tx_left: &'a [u8],
    pub sharpness: i32,
    pub iq_tuning: bool,
    pub coeff_opt_dist_threshold: u32,
    pub adaptive_txb_search_level: i32,
    pub txb_split_cap: bool,
    /// `sf.tx_sf.tx_type_search.ml_tx_split_thresh` — the `ml_predict_tx_split`
    /// NN gate (bd8 only; witness value 8500). `< 0` disables the NN (the
    /// prunes-off recursion differential passes `-1`).
    pub ml_tx_split_thresh: i32,
    /// `prune_2d_txfm_mode >= TX_TYPE_PRUNE_1` — enable the leaf `prune_tx_2D` NN
    /// (witness: true). The prunes-off recursion differential passes `false`.
    pub prune_2d: bool,
    /// The var-tx quadtree init depth (`get_search_init_depth`; 0 at speed-0 sub-720p).
    pub init_depth: i32,
}

/// Result of the whole-block var-tx search.
pub struct VarTxResult {
    pub valid: bool,
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub skip_txfm: bool,
    /// `mbmi->inter_tx_size[16]` — the chosen tx-size at each txb index (built
    /// from the winning leaves; the input to `write_tx_size_vartx`).
    pub inter_tx_size: [usize; 16],
    /// The `mbmi->tx_size` (the root/largest chosen tx size).
    pub tx_size: usize,
    pub leaves: Vec<VarTxLeaf>,
}

/// `RDCOST` on an `RdStats` (RM = rdmult).
#[inline]
fn rd_of(rdmult: i32, rate: i32, dist: i64) -> i64 {
    if rate == i32::MAX || dist == i64::MAX {
        return i64::MAX;
    }
    rdcost(rdmult, rate, dist)
}

/// Extract a contiguous `TXS_W x TXS_H` sub-block from a strided plane at the
/// txb's pixel offset `(4*blk_row, 4*blk_col)`.
fn extract_i16(src: &[i16], stride: usize, blk_row: usize, blk_col: usize, txw: usize, txh: usize) -> Vec<i16> {
    let mut out = vec![0i16; txw * txh];
    let base = 4 * blk_row * stride + 4 * blk_col;
    for r in 0..txh {
        out[r * txw..r * txw + txw].copy_from_slice(&src[base + r * stride..base + r * stride + txw]);
    }
    out
}
fn extract_u16(src: &[u16], stride: usize, blk_row: usize, blk_col: usize, txw: usize, txh: usize) -> Vec<u16> {
    let mut out = vec![0u16; txw * txh];
    let base = 4 * blk_row * stride + 4 * blk_col;
    for r in 0..txh {
        out[r * txw..r * txw + txw].copy_from_slice(&src[base + r * stride..base + r * stride + txw]);
    }
    out
}

/// The no-split candidate produced by [`try_tx_block_no_split`].
struct NoSplit {
    rd: i64,
    txb_ctx: u8,
    tx_type: usize,
    eob: u16,
    skip_txfm: bool,
    qcoeff: Vec<i32>,
    dqcoeff: Vec<i32>,
}

/// `try_tx_block_no_split` (tx_search.c:2406): evaluate `tx_size` as a single
/// (unsplit) transform block. Writes the leaf RD into `rd_stats`, returns the
/// no-split candidate. `ta`/`tl` are the block-level ENTROPY_CONTEXT arrays;
/// the leaf reads them at `[blk_col..]` / `[blk_row..]`.
#[allow(clippy::too_many_arguments)]
fn try_tx_block_no_split(
    env: &VarTxEnv,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    depth: i32,
    ta: &[i8],
    tl: &[i8],
    txfm_partition_ctx: usize,
    ref_best_rd: i64,
) -> (RdStats, NoSplit) {
    let txw = TXS_W[tx_size];
    let txh = TXS_H[tx_size];
    let txwu = TX_SIZE_WIDE_UNIT[tx_size];
    let txhu = TX_SIZE_HIGH_UNIT[tx_size];

    // Frame/block-visible txb extent (interior => full txb).
    let visible_cols = (env.max_blocks_wide.saturating_sub(blk_col)).min(txwu) * 4;
    let visible_rows = (env.max_blocks_high.saturating_sub(blk_row)).min(txhu) * 4;
    let visible_cols = visible_cols.min(txw);
    let visible_rows = visible_rows.min(txh);

    let residual = extract_i16(env.residual, env.residual_stride, blk_row, blk_col, txw, txh);
    let pred = extract_u16(env.pred, env.pred_stride, blk_row, blk_col, txw, txh);
    let src_off = env.src_off + (4 * blk_row) * env.src_stride + 4 * blk_col;

    let bctx = BlockContext {
        above: &ta[blk_col..],
        left: &tl[blk_row..],
        plane: 0,
        plane_bsize: env.bsize,
    };
    // zero_blk_rate = txb_skip_cost[txb_skip_ctx][1] (get_txb_ctx at this node).
    let (txb_skip_ctx, _dc_sign_ctx) =
        get_txb_ctx(env.bsize, tx_size, 0, &ta[blk_col..], &tl[blk_row..]);
    let tables = env.coeff_costs.tables(tx_size);
    let zero_blk_rate = tables.txb_skip[txb_skip_ctx as usize * 2 + 1];

    let leaf_inputs = InterLeafInputs {
        // Luma var-tx leaf: the multi-type inter arm (chroma pins a type instead).
        forced_uv_tx_type: None,
        residual: &residual,
        pred: &pred,
        src: env.src,
        src_off,
        src_stride: env.src_stride,
        tx_size,
        lossless: env.lossless,
        reduced_tx_set_used: env.reduced_tx_set_used,
        enable_flip_idtx: env.enable_flip_idtx,
        use_inter_dct_only: env.use_inter_dct_only,
        bd: env.bd,
        rows: env.rows,
        bctx: &bctx,
        rdmult: env.rdmult,
        coeff_costs: &tables,
        tx_type_costs: env.tx_type_costs,
        visible_cols,
        visible_rows,
        qm_level: env.qm_level,
        prune_2d: env.prune_2d,
    };

    let mut rd_stats = RdStats::init();
    rd_stats.zero_rate = zero_blk_rate;

    let leaf = search_tx_type_inter(
        &leaf_inputs,
        env.sharpness,
        env.iq_tuning,
        env.coeff_opt_dist_threshold,
        env.adaptive_txb_search_level,
        ref_best_rd,
    );
    let Some(leaf) = leaf else {
        // No candidate found under ref_best_rd — invalid (rate INT_MAX).
        return (
            RdStats::invalid(),
            NoSplit {
                rd: i64::MAX,
                txb_ctx: 0,
                tx_type: 0,
                eob: 0,
                skip_txfm: false,
                qcoeff: vec![0; txw * txh],
                dqcoeff: vec![0; txw * txh],
            },
        );
    };

    // Merge the leaf stats into rd_stats (rate/dist/sse/skip_txfm; zero_rate kept).
    let leaf_stats = RdStats {
        rate: leaf.rate,
        dist: leaf.dist,
        sse: leaf.sse,
        skip_txfm: leaf.skip_txfm,
        zero_rate: 0,
    };
    rd_stats.merge(&leaf_stats);

    // pick_skip_txfm (tx_search.c:2429): !lossless && (leaf skip || RDCOST(rate,
    // dist) >= RDCOST(zero_blk_rate, sse)).
    let mut tx_type = leaf.best_tx_type;
    let mut eob = leaf.best_eob;
    let mut qcoeff = leaf.qcoeff;
    let mut dqcoeff = leaf.dqcoeff;
    let pick_skip_txfm = !env.lossless
        && (rd_stats.skip_txfm
            || rd_of(env.rdmult, rd_stats.rate, rd_stats.dist)
                >= rd_of(env.rdmult, zero_blk_rate, rd_stats.sse));
    if pick_skip_txfm {
        rd_stats.rate = zero_blk_rate;
        rd_stats.dist = rd_stats.sse;
        eob = 0;
        tx_type = 0; // DCT_DCT (update_txk_array)
        qcoeff = vec![0; txw * txh];
        dqcoeff = vec![0; txw * txh];
    }
    rd_stats.skip_txfm = pick_skip_txfm;

    // Split-flag (partition) cost for the "no split" symbol.
    if tx_size > 0 && depth < MAX_VARTX_DEPTH {
        rd_stats.rate = ((rd_stats.rate as i64)
            + env.txfm_partition_cost[txfm_partition_ctx][0] as i64)
            .min(i32::MAX as i64) as i32;
    }

    let txb_ctx = if pick_skip_txfm { 0 } else { leaf.best_txb_ctx };
    let rd = rd_of(env.rdmult, rd_stats.rate, rd_stats.dist);
    (
        rd_stats,
        NoSplit {
            rd,
            txb_ctx,
            tx_type,
            eob,
            skip_txfm: pick_skip_txfm,
            qcoeff,
            dqcoeff,
        },
    )
}

/// `try_tx_block_split` (tx_search.c:2454): split `tx_size` into `sub_tx_size_map`
/// children and recurse. Returns `(split_rd_stats, valid, leaves, split_rdcost)`.
/// Mutates the context arrays as the children commit (backtracked by the caller
/// if no-split wins).
#[allow(clippy::too_many_arguments)]
fn try_tx_block_split(
    env: &VarTxEnv,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    depth: i32,
    ta: &mut [i8],
    tl: &mut [i8],
    tx_above: &mut [u8],
    tx_left: &mut [u8],
    txfm_partition_ctx: usize,
    no_split_rd: i64,
    ref_best_rd: i64,
) -> (RdStats, bool, Vec<VarTxLeaf>, i64) {
    let sub_txs = SUB_TX_SIZE_MAP[tx_size];
    let sub_w = TX_SIZE_WIDE_UNIT[sub_txs];
    let sub_h = TX_SIZE_HIGH_UNIT[sub_txs];
    let txbw = TX_SIZE_WIDE_UNIT[tx_size];
    let txbh = TX_SIZE_HIGH_UNIT[tx_size];
    let nblks = ((txbh / sub_h) * (txbw / sub_w)) as i64;

    let mut split = RdStats::init();
    split.rate = env.txfm_partition_cost[txfm_partition_ctx][1];
    let mut split_rdcost = rd_of(env.rdmult, split.rate, split.dist);
    let mut leaves: Vec<VarTxLeaf> = Vec::new();

    let mut r = 0usize;
    while r < txbh {
        let offsetr = blk_row + r;
        if offsetr >= env.max_blocks_high {
            break;
        }
        let mut c = 0usize;
        while c < txbw {
            let offsetc = blk_col + c;
            if offsetc >= env.max_blocks_wide {
                c += sub_w;
                continue;
            }
            let child_prev = if nblks > 0 && no_split_rd != i64::MAX {
                no_split_rd / nblks
            } else {
                i64::MAX
            };
            let child_ref = if ref_best_rd == i64::MAX {
                i64::MAX
            } else {
                ref_best_rd - split_rdcost
            };
            let (child_stats, child_valid, child_leaves) = select_tx_block(
                env, offsetr, offsetc, sub_txs, depth + 1, ta, tl, tx_above, tx_left, child_prev,
                child_ref,
            );
            if !child_valid {
                return (RdStats::invalid(), false, leaves, i64::MAX);
            }
            split.merge(&child_stats);
            split_rdcost = rd_of(env.rdmult, split.rate, split.dist);
            if split_rdcost > ref_best_rd {
                return (RdStats::invalid(), false, leaves, i64::MAX);
            }
            leaves.extend(child_leaves);
            c += sub_w;
        }
        r += sub_h;
    }
    (split, true, leaves, split_rdcost)
}

/// `select_tx_block` (tx_search.c:2601): pick the best transform partition
/// (no-split vs recursive split) + type for a sub-block. Returns
/// `(rd_stats, is_cost_valid, leaves)`. `ta`/`tl` (ENTROPY_CONTEXT) and
/// `tx_above`/`tx_left` (TXFM_CONTEXT) are mutated to the WINNER's state.
#[allow(clippy::too_many_arguments)]
fn select_tx_block(
    env: &VarTxEnv,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    depth: i32,
    ta: &mut [i8],
    tl: &mut [i8],
    tx_above: &mut [u8],
    tx_left: &mut [u8],
    prev_level_rd: i64,
    ref_best_rd: i64,
) -> (RdStats, bool, Vec<VarTxLeaf>) {
    let rd_stats_init = RdStats::init();
    if ref_best_rd < 0 {
        return (rd_stats_init, false, Vec::new());
    }
    let ctx = txfm_partition_context(tx_above[blk_col], tx_left[blk_row], env.bsize, tx_size);

    // try_no_split: (enable_tx64 || sqr_up != TX_64X64) && (enable_rect_tx || w==h).
    // At the witness config both txfm_cfg flags are on => always true.
    let try_no_split = (TXSIZE_SQR_UP_MAP[tx_size] != 4 || true)
        && (TXS_W[tx_size] == TXS_H[tx_size] || true);
    let mut try_split = tx_size > 0 && depth < MAX_VARTX_DEPTH;

    // prune_tx_size_level == 0 (inert); rt skip_tx_no_split (inert).

    let mut no_split_rd = i64::MAX;
    let mut no_split_info: Option<NoSplit> = None;
    let mut no_split_stats = RdStats::invalid();
    if try_no_split {
        let (rd_stats, ns) =
            try_tx_block_no_split(env, blk_row, blk_col, tx_size, depth, ta, tl, ctx, ref_best_rd);
        no_split_rd = ns.rd;
        // prune_inter_tx_split_rd_eval_lvl == 0 (+ intrabc hard-skip): no push.
        let level = env.adaptive_txb_search_level;
        if level > 0 {
            if no_split_rd != i64::MAX
                && (no_split_rd - (no_split_rd >> (1 + level))) > ref_best_rd
            {
                return (rd_stats_init, false, Vec::new());
            }
            if no_split_rd != i64::MAX
                && (no_split_rd - (no_split_rd >> (2 + level))) > prev_level_rd
            {
                try_split = false;
            }
        }
        if env.txb_split_cap && ns.eob == 0 {
            try_split = false;
        }
        no_split_stats = rd_stats;
        no_split_info = Some(ns);
    }

    // ml_predict_tx_split (tx_search.c:2673-2680) — bd8-only ML split prune.
    // C gate: `bd == 8 && try_split && !(ref_best_rd == MAX && no_split.rd == MAX)`;
    // if `split_score < -threshold` disable split. `ml_tx_split_thresh < 0`
    // disables the NN (the prunes-off differential).
    if env.bd == 8
        && try_split
        && env.ml_tx_split_thresh >= 0
        && !(ref_best_rd == i64::MAX && no_split_rd == i64::MAX)
    {
        let score = ml_predict_tx_split(env.residual, env.residual_stride, blk_row, blk_col, tx_size);
        if score < -env.ml_tx_split_thresh {
            try_split = false;
        }
    }

    let mut split_rdcost = i64::MAX;
    let mut split_stats = RdStats::invalid();
    let mut split_leaves: Vec<VarTxLeaf> = Vec::new();
    if try_split {
        let (s, valid, leaves, srd) = try_tx_block_split(
            env,
            blk_row,
            blk_col,
            tx_size,
            depth,
            ta,
            tl,
            tx_above,
            tx_left,
            ctx,
            no_split_rd,
            no_split_rd.min(ref_best_rd),
        );
        split_stats = s;
        split_leaves = leaves;
        split_rdcost = if valid { srd } else { i64::MAX };
    }

    if no_split_rd < split_rdcost {
        let ns = no_split_info.expect("try_no_split ran");
        let txwu = TX_SIZE_WIDE_UNIT[tx_size];
        let txhu = TX_SIZE_HIGH_UNIT[tx_size];
        // av1_set_txb_context: stamp the leaf cul over the txb footprint.
        for a in ta[blk_col..blk_col + txwu].iter_mut() {
            *a = ns.txb_ctx as i8;
        }
        for l in tl[blk_row..blk_row + txhu].iter_mut() {
            *l = ns.txb_ctx as i8;
        }
        txfm_partition_update(&mut tx_above[blk_col..], &mut tx_left[blk_row..], tx_size, tx_size);
        let leaf = VarTxLeaf {
            blk_row,
            blk_col,
            tx_size,
            tx_type: ns.tx_type,
            eob: ns.eob,
            txb_ctx: ns.txb_ctx,
            skip_txfm: ns.skip_txfm,
            qcoeff: ns.qcoeff,
            dqcoeff: ns.dqcoeff,
        };
        (no_split_stats, true, vec![leaf])
    } else {
        // Split wins (contexts already committed by the recursion).
        if split_stats.is_invalid() {
            return (split_stats, false, split_leaves);
        }
        (split_stats, true, split_leaves)
    }
}

/// `select_tx_size_and_type` (tx_search.c:3433): the max-tx-size raster over the
/// block, each unit driven through [`select_tx_block`]. Returns
/// `(final_rd, VarTxResult)`; `final_rd == i64::MAX` when invalid.
fn select_tx_size_and_type(env: &VarTxEnv, ref_best_rd: i64) -> (i64, VarTxResult) {
    let invalid_res = VarTxResult {
        valid: false,
        rate: i32::MAX,
        dist: i64::MAX,
        sse: i64::MAX,
        skip_txfm: false,
        inter_tx_size: [0; 16],
        tx_size: 0,
        leaves: Vec::new(),
    };
    if ref_best_rd == 0 {
        return (i64::MAX, invalid_res);
    }
    let max_tx_size = MAX_TXSIZE_RECT_LOOKUP[env.bsize];
    let bh = TX_SIZE_HIGH_UNIT[max_tx_size];
    let bw = TX_SIZE_WIDE_UNIT[max_tx_size];

    // `av1_get_entropy_contexts` fills the FULL plane-block extent; only the
    // txb LOOP is frame-edge clipped (`max_block_wide/high`). Sizing these by
    // the clipped extent under-runs `get_txb_ctx`'s `a[..w_unit]` read on an
    // edge block whose max tx is wider than the visible remainder. Identical
    // to the clipped form for interior blocks (full == clipped), so the
    // recursion differential is unaffected.
    let full_w = crate::tx_search::MI_SIZE_WIDE_B[env.bsize];
    let full_h = crate::tx_search::MI_SIZE_HIGH_B[env.bsize];
    let mut ta: Vec<i8> = env.above_ctx[..full_w].to_vec();
    let mut tl: Vec<i8> = env.left_ctx[..full_h].to_vec();
    let mut tx_above: Vec<u8> = env.tx_above[..full_w].to_vec();
    let mut tx_left: Vec<u8> = env.tx_left[..full_h].to_vec();

    let no_skip_txfm_cost = env.skip_txfm_cost[0];
    let skip_txfm_cost = env.skip_txfm_cost[1];
    let mut skip_txfm_rd = rdcost(env.rdmult, skip_txfm_cost, 0);
    let mut no_skip_txfm_rd = rdcost(env.rdmult, no_skip_txfm_cost, 0);

    let mut rd_stats = RdStats::init();
    let mut leaves: Vec<VarTxLeaf> = Vec::new();

    let mut idy = 0usize;
    while idy < env.max_blocks_high {
        let mut idx = 0usize;
        while idx < env.max_blocks_wide {
            let best_rd_sofar = if ref_best_rd == i64::MAX {
                i64::MAX
            } else {
                ref_best_rd - skip_txfm_rd.min(no_skip_txfm_rd)
            };
            let (pn_stats, valid, pn_leaves) = select_tx_block(
                env,
                idy,
                idx,
                max_tx_size,
                env.init_depth,
                &mut ta,
                &mut tl,
                &mut tx_above,
                &mut tx_left,
                i64::MAX,
                best_rd_sofar,
            );
            if !valid || pn_stats.rate == i32::MAX {
                return (i64::MAX, invalid_res);
            }
            rd_stats.merge(&pn_stats);
            skip_txfm_rd = rdcost(env.rdmult, skip_txfm_cost, rd_stats.sse);
            no_skip_txfm_rd = rdcost(
                env.rdmult,
                ((rd_stats.rate as i64) + no_skip_txfm_cost as i64).min(i32::MAX as i64) as i32,
                rd_stats.dist,
            );
            leaves.extend(pn_leaves);
            idx += bw;
        }
        idy += bh;
    }

    if rd_stats.rate == i32::MAX {
        return (i64::MAX, invalid_res);
    }
    rd_stats.skip_txfm = skip_txfm_rd <= no_skip_txfm_rd;
    // refine_fast_tx_search_results: inert (fast_tx_search == false).

    let final_rd = if rd_stats.skip_txfm {
        rdcost(env.rdmult, skip_txfm_cost, rd_stats.sse)
    } else {
        let mut fr = rdcost(
            env.rdmult,
            ((rd_stats.rate as i64) + no_skip_txfm_cost as i64).min(i32::MAX as i64) as i32,
            rd_stats.dist,
        );
        if !env.lossless {
            fr = fr.min(rdcost(env.rdmult, skip_txfm_cost, rd_stats.sse));
        }
        fr
    };

    // Build inter_tx_size[16] (the write_tx_size_vartx input) + root tx_size
    // from the winning leaves. Each leaf stamps its tx-unit footprint.
    let mut inter_tx_size = [max_tx_size; 16];
    for leaf in &leaves {
        let txwu = TX_SIZE_WIDE_UNIT[leaf.tx_size];
        let txhu = TX_SIZE_HIGH_UNIT[leaf.tx_size];
        for dy in 0..txhu {
            for dx in 0..txwu {
                let index = get_txb_size_index(env.bsize, leaf.blk_row + dy, leaf.blk_col + dx);
                if index < 16 {
                    inter_tx_size[index] = leaf.tx_size;
                }
            }
        }
    }
    // mbmi->tx_size = tx_size_from_tx_mode(bsize, TX_MODE_SELECT) — the first
    // leaf's (top-left) size after the recursion, i.e. inter_tx_size[0].
    let tx_size = inter_tx_size[0];

    (
        final_rd,
        VarTxResult {
            valid: true,
            rate: rd_stats.rate,
            dist: rd_stats.dist,
            sse: rd_stats.sse,
            skip_txfm: rd_stats.skip_txfm,
            inter_tx_size,
            tx_size,
            leaves,
        },
    )
}

/// `av1_pick_recursive_tx_size_type_yrd` (tx_search.c:3553) — the COEFF arm
/// (the `predict_skip_txfm` skip arm + `model_based_prune` early-return are
/// handled by the caller/gated). Runs the var-tx quadtree search over the
/// block's fixed residual. `ref_best_rd` is the tx-search rd threshold
/// (`ref_best_rd - mode_rd`, av1_txfm_search:3816).
pub fn pick_recursive_tx_size_type_yrd(env: &VarTxEnv, ref_best_rd: i64) -> VarTxResult {
    let (rd, res) = select_tx_size_and_type(env, ref_best_rd);
    if rd == i64::MAX {
        return VarTxResult {
            valid: false,
            rate: i32::MAX,
            dist: i64::MAX,
            sse: i64::MAX,
            skip_txfm: false,
            inter_tx_size: [0; 16],
            tx_size: 0,
            leaves: Vec::new(),
        };
    }
    res
}

// ===========================================================================
// Inter/intrabc CHROMA — `av1_txfm_uvrd` (tx_search.c:3696) + the
// `av1_txfm_rd_in_plane` (:3751) / `block_rd_txfm` (:3065) walk, inter arm.
//
// Chroma for an inter block is UNIFORM tx (one `av1_get_max_uv_txsize`, no
// var-tx quadtree — `encode_block_inter`'s `plane ? av1_get_max_uv_txsize(..)`
// short-circuit at encodemb.c:495-505) and searches NO tx types: the type is
// INHERITED from the co-located luma txb (`av1_get_tx_type`, blockd.h:1296-1301)
// and pinned by `get_tx_mask`'s `if (plane)` arm.
// ===========================================================================

/// One chroma txb's coded state (the pack + re-encode input).
#[derive(Clone, Debug)]
pub struct InterUvLeaf {
    /// Chroma 4x4-unit position within the chroma plane block.
    pub blk_row: usize,
    pub blk_col: usize,
    pub tx_type: usize,
    pub eob: u16,
    pub txb_ctx: u8,
    pub txb_skip_ctx: usize,
    pub dc_sign_ctx: usize,
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
}

/// Inputs for [`txfm_uvrd_inter`]. `pred`/`src` are the two chroma planes
/// (index 0 = U, 1 = V); `pred` is the intrabc DV copy that
/// `av1_enc_build_inter_predictor` (rdopt.c:3601) already wrote.
pub struct InterUvEnv<'a> {
    /// `get_plane_block_size(bsize, ss_x, ss_y)` — the CHROMA plane bsize
    /// (`get_txb_ctx` / `av1_get_entropy_contexts` operate on this).
    pub plane_bsize: usize,
    /// `av1_get_max_uv_txsize(bsize, ss_x, ss_y)` — uniform across the block.
    pub uv_tx_size: usize,
    /// Chroma 4x4 units, frame-visible-clipped (`max_block_wide/high`).
    pub max_blocks_wide: usize,
    pub max_blocks_high: usize,
    pub ss_x: usize,
    pub ss_y: usize,
    /// Chroma prediction + source planes, U then V. `pred` is contiguous with
    /// stride `pred_stride`; `src` is the frame plane read at `src_off`.
    pub pred: [&'a [u16]; 2],
    pub pred_stride: usize,
    pub src: [&'a [u16]; 2],
    pub src_off: usize,
    pub src_stride: usize,
    /// The block-local LUMA `xd->tx_type_map` (stride `tx_type_map_stride`,
    /// luma 4x4 units) the var-tx search produced — chroma reads it at the
    /// subsampling-scaled-back position.
    pub tx_type_map: &'a [u8],
    pub tx_type_map_stride: usize,
    pub rows: [&'a aom_dsp::quant::PlaneQuantRows<'a>; 2],
    /// The chroma (PLANE_TYPE_UV) coefficient cost set.
    pub coeff_costs: &'a CoeffCostSet,
    pub tx_type_costs: &'a TxTypeCosts,
    /// Chroma neighbour entropy contexts (U, V), `av1_get_entropy_contexts`.
    pub above_ctx: [&'a [i8]; 2],
    pub left_ctx: [&'a [i8]; 2],
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub enable_flip_idtx: bool,
    pub use_inter_dct_only: bool,
    pub bd: u8,
    pub rdmult: i32,
    pub sharpness: i32,
    pub iq_tuning: bool,
    pub coeff_opt_dist_threshold: u32,
    pub adaptive_txb_search_level: i32,
    /// Per-plane QM levels (U, V).
    pub qm_level: [Option<usize>; 2],
}

/// The chroma RD outcome (`rd_stats_uv`) plus the per-plane coded txbs.
pub struct InterUvResult {
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    /// `rd_stats_uv->skip_txfm` — EOB-based: 1 iff EVERY U and V txb has
    /// `eob == 0` (`this_rd_stats.skip_txfm &= !eobs[block]`, tx_search.c:3126,
    /// AND-reduced by `av1_merge_rd_stats`). NOT an SSE threshold.
    pub skip_txfm: bool,
    pub u: Vec<InterUvLeaf>,
    pub v: Vec<InterUvLeaf>,
}

/// `av1_get_tx_type(xd, PLANE_TYPE_UV, blk_row, blk_col, tx_size, reduced)`
/// (blockd.h:1283-1315) for an INTER block: DCT_DCT at lossless / `sqr_up >
/// TX_32X32`, else the co-located LUMA map entry at the subsampling-scaled-back
/// position, falling back to DCT_DCT when that type is not in the chroma tx set.
pub fn uv_tx_type_inter(
    tx_size: usize,
    lossless: bool,
    reduced_tx_set_used: bool,
    tx_type_map: &[u8],
    tx_type_map_stride: usize,
    blk_row: usize,
    blk_col: usize,
    ss_x: usize,
    ss_y: usize,
) -> usize {
    if lossless || TXSIZE_SQR_UP_MAP[tx_size] > 3 {
        return 0; // DCT_DCT
    }
    // "scale back to y plane's coordinate" (blockd.h:1299-1300).
    let lr = blk_row << ss_y;
    let lc = blk_col << ss_x;
    let idx = lr * tx_type_map_stride + lc;
    let tx_type = tx_type_map.get(idx).copied().unwrap_or(0) as usize;
    // `if (!av1_ext_tx_used[tx_set_type][tx_type]) tx_type = DCT_DCT`, with the
    // set taken at `is_inter_block(mbmi)` = true.
    let d = aom_dsp::txb::ext_tx_derive(tx_size, true, reduced_tx_set_used, tx_type, false, 0, 0);
    if d.used == 0 { 0 } else { tx_type }
}

/// `av1_txfm_uvrd` (tx_search.c:3696), INTER arm. Returns `None` for C's
/// `is_cost_valid = 0` (invalid rd stats).
///
/// `perform_best_rd_based_gating_for_chroma` is 0 at speed-0 ALLINTRA
/// (`init_inter_sf`, speed_features.c:2391; only raised at GOOD speed >= 3,
/// :1311) so `chroma_ref_best_rd` stays `ref_best_rd` — and for intrabc
/// `ref_best_rd` is INT64_MAX anyway (rdopt.c:3611), making every early exit
/// here inert.
pub fn txfm_uvrd_inter(env: &InterUvEnv, ref_best_rd: i64) -> Option<InterUvResult> {
    if ref_best_rd < 0 {
        return None;
    }
    let tx_size = env.uv_tx_size;
    let txwu = TX_SIZE_WIDE_UNIT[tx_size];
    let txhu = TX_SIZE_HIGH_UNIT[tx_size];
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);

    let mut rate: i32 = 0;
    let mut dist: i64 = 0;
    let mut sse: i64 = 0;
    let mut skip_txfm = true;
    let mut out: [Vec<InterUvLeaf>; 2] = [Vec::new(), Vec::new()];

    for plane_ix in 0..2usize {
        let plane = plane_ix + 1;
        // av1_get_entropy_contexts: working copies, stamped per txb.
        let mut ta: Vec<i8> = env.above_ctx[plane_ix].to_vec();
        let mut tl: Vec<i8> = env.left_ctx[plane_ix].to_vec();
        // `current_rd` accumulates per txb; `best_rd - current_rd` is the leaf
        // budget (block_rd_txfm, tx_search.c:3108).
        let mut current_rd: i64 = 0;

        let mut blk_row = 0usize;
        while blk_row < env.max_blocks_high {
            let mut blk_col = 0usize;
            while blk_col < env.max_blocks_wide {
                // Visible extent of this txb (partial at the frame edge).
                let visible_cols = (env.max_blocks_wide - blk_col).min(txwu) * 4;
                let visible_rows = (env.max_blocks_high - blk_row).min(txhu) * 4;
                let visible_cols = visible_cols.min(txw);
                let visible_rows = visible_rows.min(txh);

                // av1_subtract_plane already ran on the whole plane; extract
                // this txb's residual from (pred, src).
                let poff = (4 * blk_row) * env.pred_stride + 4 * blk_col;
                let soff = env.src_off + (4 * blk_row) * env.src_stride + 4 * blk_col;
                let mut residual = vec![0i16; txw * txh];
                let mut pred_txb = vec![0u16; txw * txh];
                for r in 0..txh {
                    for c in 0..txw {
                        let p = env.pred[plane_ix][poff + r * env.pred_stride + c];
                        let s = env.src[plane_ix][soff + r * env.src_stride + c];
                        pred_txb[r * txw + c] = p;
                        residual[r * txw + c] = (i32::from(s) - i32::from(p)) as i16;
                    }
                }

                let uv_tt = uv_tx_type_inter(
                    tx_size,
                    env.lossless,
                    env.reduced_tx_set_used,
                    env.tx_type_map,
                    env.tx_type_map_stride,
                    blk_row,
                    blk_col,
                    env.ss_x,
                    env.ss_y,
                );

                let bctx = BlockContext {
                    above: &ta[blk_col..],
                    left: &tl[blk_row..],
                    plane,
                    plane_bsize: env.plane_bsize,
                };
                let tables = env.coeff_costs.tables(tx_size);
                let inp = InterLeafInputs {
                    forced_uv_tx_type: Some(uv_tt),
                    residual: &residual,
                    pred: &pred_txb,
                    src: env.src[plane_ix],
                    src_off: soff,
                    src_stride: env.src_stride,
                    tx_size,
                    lossless: env.lossless,
                    reduced_tx_set_used: env.reduced_tx_set_used,
                    enable_flip_idtx: env.enable_flip_idtx,
                    use_inter_dct_only: env.use_inter_dct_only,
                    bd: env.bd,
                    rows: env.rows[plane_ix],
                    bctx: &bctx,
                    rdmult: env.rdmult,
                    coeff_costs: &tables,
                    tx_type_costs: env.tx_type_costs,
                    visible_cols,
                    visible_rows,
                    qm_level: env.qm_level[plane_ix],
                    prune_2d: false,
                };
                let leaf_budget = if ref_best_rd == i64::MAX {
                    i64::MAX
                } else {
                    ref_best_rd - current_rd
                };
                let r = search_tx_type_inter(
                    &inp,
                    env.sharpness,
                    env.iq_tuning,
                    env.coeff_opt_dist_threshold,
                    env.adaptive_txb_search_level,
                    leaf_budget,
                )?;

                // av1_set_txb_context (full txb footprint).
                let ent = r.best_txb_ctx as i8;
                let (a_end, l_end) = ((blk_col + txwu).min(ta.len()), (blk_row + txhu).min(tl.len()));
                for a in ta[blk_col..a_end].iter_mut() {
                    *a = ent;
                }
                for l in tl[blk_row..l_end].iter_mut() {
                    *l = ent;
                }

                // Inter arm (tx_search.c:3120-3127).
                let no_skip_rd = rdcost(env.rdmult, r.rate, r.dist);
                let skip_rd = rdcost(env.rdmult, 0, r.sse);
                current_rd += no_skip_rd.min(skip_rd);
                skip_txfm &= r.best_eob == 0;

                rate = rate.saturating_add(r.rate);
                dist += r.dist;
                sse += r.sse;

                out[plane_ix].push(InterUvLeaf {
                    blk_row,
                    blk_col,
                    tx_type: r.best_tx_type,
                    eob: r.best_eob,
                    txb_ctx: r.best_txb_ctx,
                    txb_skip_ctx: r.txb_skip_ctx,
                    dc_sign_ctx: r.dc_sign_ctx,
                    qcoeff: r.qcoeff,
                    dqcoeff: r.dqcoeff,
                });

                blk_col += txwu;
            }
            blk_row += txhu;
        }

        // Per-plane early exit (tx_search.c:3735-3741) — inert at INT64_MAX.
        if ref_best_rd != i64::MAX {
            let this_rd = rdcost(env.rdmult, rate, dist);
            let skip_rd = rdcost(env.rdmult, 0, sse);
            if this_rd.min(skip_rd) > ref_best_rd {
                return None;
            }
        }
    }

    let [u, v] = out;
    Some(InterUvResult {
        rate,
        dist,
        sse,
        skip_txfm,
        u,
        v,
    })
}
