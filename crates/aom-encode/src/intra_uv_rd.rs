//! Chroma intra RD evaluation (libaom `av1/encoder/tx_search.c` +
//! `intra_mode_search.c`, speed-0 all-intra):
//! - [`txfm_rd_in_plane_uv`]: `av1_txfm_rd_in_plane` for one chroma plane of
//!   an intra block — the `av1_foreach_transformed_block_in_plane` walk at
//!   the (single) UV tx size, per txb predict-into-recon (incl. the CfL
//!   DC+AC path with the encoder's DC-prediction cache) -> subtract ->
//!   `search_tx_type` -> winner reconstruction -> entropy-context stamp;
//! - [`txfm_uvrd`]: `av1_txfm_uvrd` (intra arm) — both chroma planes summed
//!   with the `AOMMIN(this_rd, skip_txfm_rd) > ref_best_rd` early-out
//!   (`perform_best_rd_based_gating_for_chroma` is inter-only, so intra
//!   always gates against the unrefined `ref_best_rd`);
//! - the chroma geometry helpers (`av1_get_tx_size` plane arm /
//!   `av1_get_max_uv_txsize` / `is_chroma_reference` / the sub-8x8
//!   `setup_pred_plane` mi rounding).
//!
//! Frame-interior blocks (`max_blocks_*` unclipped), matching the luma walk.

use crate::rd::rdcost;
use crate::tx_search::{
    search_tx_type_intra, RdStats, TxTypeSearchInputs, TxTypeSearchPolicy, TxbWinner,
    MAX_TXSIZE_RECT_LOOKUP,
};
use aom_entropy::partition::{get_plane_block_size, get_uv_mode, intra_avail};
use aom_intra::cfl::{cfl_predict_block, CflCtx, CFL_BUF_LINE};
use aom_intra::predict_intra_high;
use aom_txb::{CoeffCostTables, TxTypeCosts};

const TXS_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TXS_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
const MI_W: [usize; 22] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
const MI_H: [usize; 22] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];

/// `UV_CFL_PRED` (enums.h).
pub const UV_CFL_PRED: usize = 13;

/// `av1_get_adjusted_tx_size` (blockd.h): 64-point sizes clamp to their
/// 32-point counterparts (chroma never uses 64-pt transforms).
pub fn av1_get_adjusted_tx_size(tx_size: usize) -> usize {
    match tx_size {
        4 | 12 | 11 => 3, // TX_64X64 / TX_64X32 / TX_32X64 -> TX_32X32
        18 => 10,         // TX_64X16 -> TX_32X16
        17 => 9,          // TX_16X64 -> TX_16X32
        t => t,
    }
}

/// `av1_get_max_uv_txsize` (blockd.h): the (uniform) chroma tx size —
/// `max_txsize_rect_lookup` of the subsampled plane block, 64-clamped.
pub fn av1_get_max_uv_txsize(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
    debug_assert!(plane_bsize < 22);
    av1_get_adjusted_tx_size(MAX_TXSIZE_RECT_LOOKUP[plane_bsize])
}

/// `av1_get_tx_size` (blockd.h) for a chroma plane: TX_4X4 when the segment
/// is lossless, else [`av1_get_max_uv_txsize`].
pub fn av1_get_tx_size_uv(bsize: usize, lossless: bool, ss_x: usize, ss_y: usize) -> usize {
    if lossless {
        return 0; // TX_4X4
    }
    av1_get_max_uv_txsize(bsize, ss_x, ss_y)
}

/// `is_chroma_reference` (av1_common_int.h:1456): whether this block carries
/// the chroma for its (possibly shared sub-8x8) chroma block.
pub fn is_chroma_reference(
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
) -> bool {
    let bw = MI_W[bsize] as i32;
    let bh = MI_H[bsize] as i32;
    ((mi_row & 1) != 0 || (bh & 1) == 0 || ss_y == 0)
        && ((mi_col & 1) != 0 || (bw & 1) == 0 || ss_x == 0)
}

/// The sub-8x8 mi rounding of `setup_pred_plane` (av1_common_int.h): a
/// chroma plane's dst/src pointers anchor at the EVEN mi position when the
/// block is 1 mi wide/high on a subsampled axis (the shared chroma block's
/// origin). Returns the plane pixel offset of `(mi_row, mi_col)` in a plane
/// of `stride` (top-left of the frame at `base`).
pub fn chroma_plane_offset(
    base: usize,
    stride: usize,
    mut mi_row: i32,
    mut mi_col: i32,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
) -> usize {
    if ss_y != 0 && (mi_row & 1) != 0 && MI_H[bsize] == 1 {
        mi_row -= 1;
    }
    if ss_x != 0 && (mi_col & 1) != 0 && MI_W[bsize] == 1 {
        mi_col -= 1;
    }
    let x = ((4 * mi_col) >> ss_x) as usize;
    let y = ((4 * mi_row) >> ss_y) as usize;
    base + y * stride + x
}

/// The plane dimensions `set_plane_n4` (encodeframe_utils / decodeframe)
/// installs in `pd->width/height`: subsampled block dims clamped to >= 4.
pub fn plane_px_dims(bsize: usize, ss_x: usize, ss_y: usize) -> (i32, i32) {
    let w = ((MI_W[bsize] * 4) >> ss_x).max(4) as i32;
    let h = ((MI_H[bsize] * 4) >> ss_y).max(4) as i32;
    (w, h)
}

/// The encoder's CfL DC-prediction cache (`xd->cfl.use_dc_pred_cache` +
/// `dc_pred_is_cached` + `dc_pred_cache`, blockd.h / cfl.c): during
/// `cfl_rd_pick_alpha` the DC prediction is computed once per plane, its
/// FIRST ROW stored (`cfl_store_dc_pred` copies `width` pixels), and every
/// later prediction row-replicates it (`cfl_load_dc_pred`) — exact because
/// DC_PRED yields one value for the whole block (the production RTCD
/// cfl_predict SIMD kernels rely on the same block-constant invariant by
/// broadcasting `*dst`).
pub struct CflDcCache {
    /// `use_dc_pred_cache` — true only inside `cfl_rd_pick_alpha`.
    pub use_cache: bool,
    /// `dc_pred_is_cached[CFL_PRED_U/V]`.
    pub cached: [bool; 2],
    /// `dc_pred_cache[CFL_PRED_U/V]` — the stored first row.
    pub row: [[u16; CFL_BUF_LINE]; 2],
}

impl CflDcCache {
    /// `clear_cfl_dc_pred_cache_flags`: cache off, nothing cached.
    pub fn cleared() -> Self {
        CflDcCache { use_cache: false, cached: [false; 2], row: [[0; CFL_BUF_LINE]; 2] }
    }
}

/// The per-candidate CfL prediction state `av1_predict_intra_block_facade`
/// reads for a `UV_CFL_PRED` block: the loaded luma context + the coded
/// alpha (`mbmi->cfl_alpha_idx` / `cfl_alpha_signs`) + the DC cache.
pub struct CflPredict<'a> {
    pub ctx: &'a mut CflCtx,
    pub cache: &'a mut CflDcCache,
    pub alpha_idx: i32,
    pub joint_sign: i32,
}

/// The frame/block environment of a chroma intra RD evaluation — the
/// MACROBLOCK(D) state the UV `block_rd_txfm` walk reads, expressed as plain
/// data (one struct shared by both planes; the per-candidate mode fields are
/// arguments). Frame-interior blocks.
pub struct UvRdEnv<'a> {
    // intra_avail geometry (LUMA bsize + actual mi position; chroma
    // availability flags are the `xd->chroma_up/left_available` values).
    pub sb_size: usize,
    pub bsize: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub chroma_up_available: bool,
    pub chroma_left_available: bool,
    pub tile_col_end: i32,
    pub tile_row_end: i32,
    pub partition: usize,
    pub mi_cols: i32,
    pub mi_rows: i32,
    pub ss_x: usize,
    pub ss_y: usize,
    // Pixel planes, u = index 0 / v = index 1: `recon` is passed &mut to the
    // walk; offsets anchor the block's top-left (sub-8x8 mi rounding already
    // applied — see [`chroma_plane_offset`]).
    pub ref_off: [usize; 2],
    pub ref_stride: usize,
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    pub src_off: [usize; 2],
    pub src_stride: usize,
    // Prediction config.
    pub disable_edge_filter: bool,
    pub filter_type: i32,
    // LUMA winner context (chroma tx-set/tx-type-rate selection).
    pub luma_mode: usize,
    pub luma_use_fi: bool,
    pub luma_fi_mode: usize,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub bd: u8,
    // Quantizer + RD (per-plane rows; shared UV coefficient cost tables —
    // one (uv_txs_ctx, PLANE_TYPE_UV) set covers both planes at the single
    // UV tx size).
    pub rows_u: &'a aom_quant::PlaneQuantRows<'a>,
    pub rows_v: &'a aom_quant::PlaneQuantRows<'a>,
    pub rdmult: i32,
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
    // Per-plane neighbour entropy contexts (plane 4x4 units).
    pub above_ctx: [&'a [i8]; 2],
    pub left_ctx: [&'a [i8]; 2],
}

impl UvRdEnv<'_> {
    fn src(&self, plane: usize) -> &[u16] {
        if plane == 1 { self.src_u } else { self.src_v }
    }
    fn rows(&self, plane: usize) -> &aom_quant::PlaneQuantRows<'_> {
        if plane == 1 { self.rows_u } else { self.rows_v }
    }
}

/// One txb's prediction (`av1_predict_intra_block_facade` for a chroma
/// plane): the CfL arm (DC prediction — cached or fresh — plus the
/// alpha-scaled AC) or the plain intra prediction, written INTO the recon
/// plane (the facade's in-place dst write; load-bearing for the next txb).
#[allow(clippy::too_many_arguments)]
fn predict_uv_txb(
    env: &UvRdEnv,
    recon: &mut [u16],
    plane: usize,
    uv_mode: usize,
    angle_delta_uv: i32,
    cfl: Option<&mut CflPredict>,
    tx_size: usize,
    blk_row: usize,
    blk_col: usize,
    txb_off: usize,
) {
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let mode = get_uv_mode(uv_mode) as usize;
    let (wpx, hpx) = plane_px_dims(env.bsize, env.ss_x, env.ss_y);

    if let Some(cfl) = cfl {
        debug_assert_eq!(uv_mode, UV_CFL_PRED);
        debug_assert_eq!((blk_row, blk_col), (0, 0), "CfL block == tx block");
        let pred_plane = plane - 1;
        if !(cfl.cache.use_cache && cfl.cache.cached[pred_plane]) {
            // Fresh DC prediction into the recon plane (mode == DC_PRED).
            let (n_top, n_topright, n_left, n_bottomleft) = intra_avail(
                env.sb_size,
                env.bsize,
                env.mi_row,
                env.mi_col,
                env.chroma_up_available,
                env.chroma_left_available,
                env.tile_col_end,
                env.tile_row_end,
                env.partition,
                tx_size,
                env.ss_x as i32,
                env.ss_y as i32,
                blk_row as i32,
                blk_col as i32,
                wpx,
                hpx,
                env.mi_cols,
                env.mi_rows,
                mode,
                0,
                false,
            );
            let mut pred = vec![0u16; txw * txh];
            predict_intra_high(
                recon,
                txb_off,
                env.ref_stride,
                &mut pred,
                txw,
                mode,
                0,
                false,
                0,
                env.disable_edge_filter,
                env.filter_type,
                tx_size,
                n_top as usize,
                n_topright,
                n_left as usize,
                n_bottomleft,
                env.bd as i32,
            );
            for r in 0..txh {
                recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }
            if cfl.cache.use_cache {
                // cfl_store_dc_pred: the first `width` pixels of the dc pred.
                cfl.cache.row[pred_plane][..txw]
                    .copy_from_slice(&recon[txb_off..txb_off + txw]);
                cfl.cache.cached[pred_plane] = true;
            }
        } else {
            // cfl_load_dc_pred: row-replicate the cached first row.
            for r in 0..txh {
                recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                    .copy_from_slice(&cfl.cache.row[pred_plane][..txw]);
            }
        }
        cfl_predict_block(
            cfl.ctx,
            recon,
            txb_off,
            env.ref_stride,
            tx_size,
            plane,
            cfl.alpha_idx,
            cfl.joint_sign,
            i32::from(env.bd),
        );
    } else {
        let (n_top, n_topright, n_left, n_bottomleft) = intra_avail(
            env.sb_size,
            env.bsize,
            env.mi_row,
            env.mi_col,
            env.chroma_up_available,
            env.chroma_left_available,
            env.tile_col_end,
            env.tile_row_end,
            env.partition,
            tx_size,
            env.ss_x as i32,
            env.ss_y as i32,
            blk_row as i32,
            blk_col as i32,
            wpx,
            hpx,
            env.mi_cols,
            env.mi_rows,
            mode,
            angle_delta_uv * 3, // ANGLE_STEP
            false,
        );
        let mut pred = vec![0u16; txw * txh];
        predict_intra_high(
            recon,
            txb_off,
            env.ref_stride,
            &mut pred,
            txw,
            mode,
            angle_delta_uv * 3,
            false,
            0,
            env.disable_edge_filter,
            env.filter_type,
            tx_size,
            n_top as usize,
            n_topright,
            n_left as usize,
            n_bottomleft,
            env.bd as i32,
        );
        for r in 0..txh {
            recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                .copy_from_slice(&pred[r * txw..r * txw + txw]);
        }
    }
}

/// `av1_txfm_rd_in_plane` (tx_search.c:3751) for one CHROMA plane of an
/// intra block at the (uniform) UV `tx_size`: the foreach-txb raster walk
/// over the subsampled plane block; per txb [`predict_uv_txb`] -> subtract
/// -> [`search_tx_type_intra`] (plane arm: pinned UV tx type, chroma trellis
/// rd mult, no tx-type bits) -> winner reconstruction into `recon` ->
/// entropy-context stamp. Intra rd accumulation signals non-skip per txb;
/// ANY early exit invalidates (`invalid_rd = args.exit_early` for intra).
///
/// `cfl` must be `Some` exactly when `uv_mode == UV_CFL_PRED`. Returns
/// `None` on early exit / `current_rd_in > ref_best_rd`.
#[allow(clippy::too_many_arguments)]
pub fn txfm_rd_in_plane_uv(
    env: &UvRdEnv,
    recon: &mut [u16],
    plane: usize,
    uv_mode: usize,
    angle_delta_uv: i32,
    mut cfl: Option<&mut CflPredict>,
    tx_size: usize,
    ref_best_rd: i64,
    current_rd_in: i64,
    pol: &TxTypeSearchPolicy,
) -> Option<(RdStats, Vec<TxbWinner>)> {
    if current_rd_in > ref_best_rd {
        return None;
    }
    debug_assert_eq!(cfl.is_some(), uv_mode == UV_CFL_PRED);
    let plane_bsize = get_plane_block_size(env.bsize, env.ss_x, env.ss_y);
    debug_assert!(plane_bsize < 22, "invalid chroma plane block");
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let (txw_unit, txh_unit) = (txw >> 2, txh >> 2);
    let max_blocks_wide = MI_W[plane_bsize];
    let max_blocks_high = MI_H[plane_bsize];
    let pi = plane - 1;

    // av1_get_entropy_contexts: working copies of the neighbour contexts.
    let mut t_above: Vec<i8> = env.above_ctx[pi][..max_blocks_wide].to_vec();
    let mut t_left: Vec<i8> = env.left_ctx[pi][..max_blocks_high].to_vec();

    let mut stats = RdStats::zero();
    let mut winners: Vec<TxbWinner> = Vec::new();
    let mut current_rd = current_rd_in;
    let mut exit_early = false;

    let mut blk_row = 0usize;
    while blk_row < max_blocks_high {
        let mut blk_col = 0usize;
        while blk_col < max_blocks_wide {
            if exit_early {
                return None; // intra: exit_early alone invalidates
            }
            let txb_off = env.ref_off[pi] + (blk_row * env.ref_stride + blk_col) * 4;
            predict_uv_txb(
                env,
                recon,
                plane,
                uv_mode,
                angle_delta_uv,
                cfl.as_deref_mut(),
                tx_size,
                blk_row,
                blk_col,
                txb_off,
            );
            // Snapshot the prediction (tight) for the search + recon base.
            let mut pred = vec![0u16; txw * txh];
            for r in 0..txh {
                pred[r * txw..r * txw + txw].copy_from_slice(
                    &recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw],
                );
            }

            // av1_subtract_txb.
            let src = env.src(plane);
            let src_txb_off = env.src_off[pi] + (blk_row * env.src_stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            aom_dist::highbd_subtract_block(
                txh,
                txw,
                &mut residual,
                txw,
                &src[src_txb_off..],
                env.src_stride,
                &pred,
                txw,
            );

            let bctx = crate::BlockContext {
                plane_bsize,
                plane,
                above: &t_above[blk_col..],
                left: &t_left[blk_row..],
            };
            let inp = TxTypeSearchInputs {
                residual: &residual,
                src,
                src_off: src_txb_off,
                src_stride: env.src_stride,
                pred: &pred,
                tx_size,
                plane,
                uv_mode,
                mode: env.luma_mode,
                use_filter_intra: env.luma_use_fi,
                filter_intra_mode: env.luma_fi_mode,
                lossless: env.lossless,
                reduced_tx_set_used: env.reduced_tx_set_used,
                bd: env.bd,
                rows: env.rows(plane),
                bctx: &bctx,
                rdmult: env.rdmult,
                coeff_costs: env.coeff_costs,
                tx_type_costs: env.tx_type_costs,
            };
            let win = search_tx_type_intra(&inp, pol, ref_best_rd - current_rd)
                .expect("search_tx_type always yields a winner");

            // recon_intra: reconstruct the winner over the prediction.
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

            // av1_set_txb_context (interior).
            for a in t_above[blk_col..blk_col + txw_unit].iter_mut() {
                *a = win.best_txb_ctx as i8;
            }
            for l in t_left[blk_row..blk_row + txh_unit].iter_mut() {
                *l = win.best_txb_ctx as i8;
            }

            // Intra rd accumulation: signalled non-skip.
            let this =
                RdStats { rate: win.rate, dist: win.dist, sse: win.sse, skip_txfm: false };
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
        return None;
    }
    Some((stats, winners))
}

/// `av1_txfm_uvrd` (tx_search.c:3696), intra arm: evaluate both chroma
/// planes of a non-CfL UV candidate at the (uniform) UV tx size
/// (`av1_get_tx_size(AOM_PLANE_U)`), merging their RD stats with the
/// per-plane `AOMMIN(this_rd, skip_txfm_rd) > ref_best_rd` invalidation.
/// (`ref_best_rd < 0 -> invalid`; `is_chroma_ref` is the caller's gate.)
/// Returns `(stats, winners_u, winners_v)` or `None` (invalid — the C's
/// `av1_invalid_rd_stats` + return 0).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn txfm_uvrd(
    env: &UvRdEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    uv_mode: usize,
    angle_delta_uv: i32,
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
) -> Option<(RdStats, Vec<TxbWinner>, Vec<TxbWinner>)> {
    debug_assert_ne!(uv_mode, UV_CFL_PRED, "CfL evaluates through cfl_rd_pick_alpha");
    if ref_best_rd < 0 {
        return None;
    }
    let uv_tx_size = av1_get_tx_size_uv(env.bsize, env.lossless, env.ss_x, env.ss_y);

    let mut stats = RdStats::zero();
    let mut winners_u = Vec::new();
    let mut winners_v = Vec::new();
    for plane in 1..=2usize {
        // Intra: chroma_ref_best_rd stays ref_best_rd (inter-only gating sf).
        let recon: &mut [u16] = if plane == 1 { recon_u } else { recon_v };
        let (this_stats, winners) = txfm_rd_in_plane_uv(
            env,
            recon,
            plane,
            uv_mode,
            angle_delta_uv,
            None,
            uv_tx_size,
            ref_best_rd,
            0,
            pol,
        )?;
        if this_stats.rate == i32::MAX {
            return None; // "if (this_rd_stats.rate == INT_MAX)" break
        }
        stats.merge(&this_stats);
        if plane == 1 {
            winners_u = winners;
        } else {
            winners_v = winners;
        }
        let this_rd = rdcost(env.rdmult, stats.rate, stats.dist);
        let skip_txfm_rd = rdcost(env.rdmult, 0, stats.sse);
        if this_rd.min(skip_txfm_rd) > ref_best_rd {
            return None;
        }
    }
    Some((stats, winners_u, winners_v))
}
