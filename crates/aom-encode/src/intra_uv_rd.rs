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
//!
//! # Integration point: `av1_rd_pick_intra_mode_sb` (rdopt.c:3655-3690)
//!
//! The whole-block KEY-frame intra search composes this module as:
//! 1. `av1_rd_pick_intra_sby_mode` (the landed luma pick) -> `intra_yrd`;
//! 2. `set_mode_eval_params(DEFAULT_EVAL)`;
//! 3. when `intra_yrd < best_rd && num_planes > 1`:
//!    a. if `xd->is_chroma_ref && store_cfl_required_rdo(cm, x)` (=
//!    `is_cfl_allowed(xd)` for a chroma-ref block): restore the LUMA
//!    winner's `tx_type_map` from the pick ctx (`av1_copy_array`) — the
//!    luma re-encode below replays the winner's tx types;
//!    b. `max_uv_tx_size = av1_get_tx_size(AOM_PLANE_U)` =
//!    [`av1_get_tx_size_uv`];
//!    c. `av1_rd_pick_intra_sbuv_mode(..., bsize, max_uv_tx_size)` =
//!    [`rd_pick_intra_sbuv_mode`] — whose own preamble (the CALLER contract
//!    here) is `init_sbuv_mode`, the `!is_chroma_ref` early return
//!    (`rate = rate_tokenonly = dist = 0, skippable = 1`, INT64_MAX), and
//!    `xd->cfl.store_y = store_cfl_required_rdo` gating one
//!    `av1_encode_intra_block_plane(AOM_PLANE_Y, DRY_RUN_NORMAL,
//!    optimize_seg_arr[segment_id])` — the luma-winner re-encode that fills
//!    the luma recon AND `cfl.recon_buf_q3` (per-txb `cfl_store_tx` from
//!    `encode_block_intra`), i.e. the loaded [`CflCtx`] this module takes
//!    as input;
//! 4. `rd_cost->rate = rate_y + rate_uv + skip_txfm_cost[skip_ctx][0]`,
//!    `dist = dist_y + dist_uv`, `RDCOST` — intra blocks always signal
//!    non-skip.
//!
//! MISSING for that composition (next chunks): `av1_encode_intra_block_plane`
//! (compose the landed predict/xform/quant/trellis/recon pieces over the
//! winner tx sizes/types + per-txb `cfl_store_tx`), then the partition RDO
//! above it. `rd_pick_intrabc_mode_sb` is a no-op without `allow_intrabc`.

use crate::rd::rdcost;
use crate::tx_search::{
    MAX_TXSIZE_RECT_LOOKUP, RdStats, TxTypeSearchInputs, TxTypeSearchPolicy, TxbWinner,
    get_txb_visible_dimensions, max_block_units, search_tx_type_intra,
};
use aom_entropy::partition::{get_plane_block_size, get_uv_mode, intra_avail};
use aom_intra::cfl::{CFL_BUF_LINE, CflCtx, cfl_predict_block};
use aom_intra::predict_intra_high;
use aom_txb::{CoeffCostTables, TxTypeCosts};

const TXS_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TXS_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
const MI_W: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

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

/// `scale_chroma_bsize` (reconintra.c): the block size the chroma availability
/// walk (`has_top_right`/`has_bottom_left`) sees. Sub-8x8 dimensions on a
/// subsampled axis are promoted to the shared-chroma group's size. Mirrors the
/// bit-exact decoder's `aom_decode::scale_chroma_bsize`.
fn uv_scale_chroma_bsize(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    const BLOCK_4X4: usize = 0;
    const BLOCK_4X8: usize = 1;
    const BLOCK_8X4: usize = 2;
    const BLOCK_8X8: usize = 3;
    const BLOCK_8X16: usize = 4;
    const BLOCK_16X8: usize = 5;
    const BLOCK_4X16: usize = 16;
    const BLOCK_16X4: usize = 17;
    match bsize {
        BLOCK_4X4 => match (ss_x, ss_y) {
            (1, 1) => BLOCK_8X8,
            (1, 0) => BLOCK_8X4,
            (0, 1) => BLOCK_4X8,
            _ => bsize,
        },
        BLOCK_4X8 => match (ss_x, ss_y) {
            (1, _) => BLOCK_8X8,
            _ => bsize,
        },
        BLOCK_8X4 => match (ss_x, ss_y) {
            (_, 1) => BLOCK_8X8,
            _ => bsize,
        },
        BLOCK_4X16 => match (ss_x, ss_y) {
            (1, _) => BLOCK_8X16,
            _ => bsize,
        },
        BLOCK_16X4 => match (ss_x, ss_y) {
            (_, 1) => BLOCK_16X8,
            _ => bsize,
        },
        _ => bsize,
    }
}

/// The `(bsize, mi_row, mi_col)` triple `av1_predict_intra_block` feeds to the
/// chroma availability walk (`has_top_right`/`has_bottom_left`): the
/// `scale_chroma_bsize`-promoted block size and the chroma-reference
/// (sub-8x8-rounded, `chroma_plane_offset`-style) mi anchor. The luma leaf's raw
/// `(bsize, mi_row, mi_col)` gives the walk the wrong sub-8x8 extent. The
/// edge-pixel counts (`n_top`/`n_left`) are invariant under this promotion — the
/// +1 mi-dimension and the -1 mi-position cancel in `mb_to_*_edge` — so ONLY the
/// directional top-right / bottom-left availability changes, matching the
/// bit-exact decoder (`aom-decode/src/lib.rs`: `bsize_uv` + `adj_row`/`adj_col`).
fn uv_avail_geom(
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    ss_x: usize,
    ss_y: usize,
) -> (usize, i32, i32) {
    let adj_row = if ss_y != 0 && (mi_row & 1) != 0 && MI_H[bsize] == 1 {
        mi_row - 1
    } else {
        mi_row
    };
    let adj_col = if ss_x != 0 && (mi_col & 1) != 0 && MI_W[bsize] == 1 {
        mi_col - 1
    } else {
        mi_col
    };
    (uv_scale_chroma_bsize(bsize, ss_x, ss_y), adj_row, adj_col)
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
        CflDcCache {
            use_cache: false,
            cached: [false; 2],
            row: [[0; CFL_BUF_LINE]; 2],
        }
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
    /// `mbmi->palette_mode_info.palette_size[0] > 0` — the LUMA winner has an
    /// active Y palette. Feeds `intra_mode_info_cost_uv`'s
    /// `palette_uv_mode_cost[y_palette_active][..]` context (the UV
    /// no-palette flag's cdf row differs when Y uses palette). Overwritten
    /// with the luma winner before the uv search, like `luma_mode`.
    pub luma_palette_active: bool,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub bd: u8,
    // Quantizer + RD (per-plane rows; shared UV coefficient cost tables —
    // one (uv_txs_ctx, PLANE_TYPE_UV) set covers both planes at the single
    // UV tx size).
    pub rows_u: &'a aom_quant::PlaneQuantRows<'a>,
    pub rows_v: &'a aom_quant::PlaneQuantRows<'a>,
    pub rdmult: i32,
    /// The REAL per-(txs_ctx, eob_multi_size) chroma cost table, PRE-SELECTED
    /// by the caller for this block's (single, uniform) UV `tx_size`
    /// (`av1_get_tx_size_uv` — chroma has no tx-size depth search, so unlike
    /// luma's [`crate::tx_search::TxfmYrdEnv`] one lookup at construction
    /// covers this env's whole lifetime: `CoeffCostSet::tables(uv_tx_size)`).
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
    // Per-plane neighbour entropy contexts (plane 4x4 units).
    pub above_ctx: [&'a [i8]; 2],
    pub left_ctx: [&'a [i8]; 2],
    /// Frame QM levels (`qmatrix_level_{y,u,v}`), `None` = QM off. The UV RD
    /// walk and the chroma re-encode read `[plane]` (1 = U, 2 = V).
    pub qm_levels: Option<[usize; 3]>,
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
/// (Shared with the winner re-encode walk,
/// [`crate::encode_intra::encode_intra_block_plane_uv`].)
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_uv_txb(
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
    // Chroma availability geometry for has_top_right/has_bottom_left: the
    // scale_chroma_bsize-promoted block size + chroma-reference mi anchor
    // (`av1_predict_intra_block`, reconintra.c:1783). The luma leaf's raw
    // (bsize, mi_row, mi_col) mis-sizes the directional top-right/bottom-left
    // walk for a sub-8x8 chroma-reference block; matches the bit-exact decoder.
    let (avail_bsize, avail_row, avail_col) =
        uv_avail_geom(env.bsize, env.mi_row, env.mi_col, env.ss_x, env.ss_y);

    if let Some(cfl) = cfl {
        debug_assert_eq!(uv_mode, UV_CFL_PRED);
        debug_assert_eq!((blk_row, blk_col), (0, 0), "CfL block == tx block");
        let pred_plane = plane - 1;
        if !(cfl.cache.use_cache && cfl.cache.cached[pred_plane]) {
            // Fresh DC prediction into the recon plane (mode == DC_PRED).
            let (n_top, n_topright, n_left, n_bottomleft) = intra_avail(
                env.sb_size,
                avail_bsize,
                avail_row,
                avail_col,
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
                cfl.cache.row[pred_plane][..txw].copy_from_slice(&recon[txb_off..txb_off + txw]);
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
            avail_bsize,
            avail_row,
            avail_col,
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
    cfl: Option<&mut CflPredict>,
    tx_size: usize,
    ref_best_rd: i64,
    current_rd_in: i64,
    pol: &TxTypeSearchPolicy,
) -> Option<(RdStats, Vec<TxbWinner>)> {
    txfm_rd_in_plane_uv_p(
        env,
        recon,
        plane,
        uv_mode,
        angle_delta_uv,
        cfl,
        tx_size,
        ref_best_rd,
        current_rd_in,
        pol,
        None,
    )
}

/// [`txfm_rd_in_plane_uv`] + an optional palette-UV candidate.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn txfm_rd_in_plane_uv_p(
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
    palette: Option<&PaletteUvPred>,
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

    // Frame-edge clip (max_block_wide/high with chroma subsampling): a partial
    // edge block's off-frame chroma tx sub-blocks are not searched, and per-txb
    // distortion is measured only over the visible area (get_txb_visible_dimensions).
    let (blocks_wide_visible, blocks_high_visible, mb_to_right, mb_to_bottom) = max_block_units(
        env.mi_cols,
        env.mi_rows,
        env.mi_col,
        env.mi_row,
        MI_W[env.bsize] as i32,
        MI_H[env.bsize] as i32,
        MI_W[plane_bsize] * 4,
        MI_H[plane_bsize] * 4,
        env.ss_x,
        env.ss_y,
    );

    // av1_get_entropy_contexts: working copies of the neighbour contexts.
    let mut t_above: Vec<i8> = env.above_ctx[pi][..max_blocks_wide].to_vec();
    let mut t_left: Vec<i8> = env.left_ctx[pi][..max_blocks_high].to_vec();
    // predict_dc_only_block's zero_blk_rate ctx (tx_search.c:2055-2063): the
    // BLOCK-ORIGIN skip ctx from the PERSISTENT (pre-walk) entropy arrays,
    // shared by every txb of this chroma block — same quirk as the luma walk
    // (see `TxTypeSearchInputs::predict_skip_zero_blk_rate`). Computed before
    // the walk stamps t_above/t_left.
    let predict_skip_zero_blk_rate = if pol.predict_dc_level >= 1 {
        let (origin_skip_ctx, _) =
            aom_txb::get_txb_ctx(plane_bsize, tx_size, plane, &t_above, &t_left);
        env.coeff_costs.txb_skip[origin_skip_ctx as usize * 2 + 1]
    } else {
        0
    };

    let mut stats = RdStats::zero();
    let mut winners: Vec<TxbWinner> = Vec::new();
    let mut current_rd = current_rd_in;
    let mut exit_early = false;

    // mu-64 chunk walk (see `txfm_rd_in_plane_intra`): chroma unit =
    // get_plane_block_size(BLOCK_64X64, ss) (encodemb.c:560-561); above 64x64
    // (luma) the chroma RD must reconstruct in chunk order. One chunk
    // (byte-identical) for a chroma plane block <= 64.
    let uv_unit_bsize = get_plane_block_size(12, env.ss_x, env.ss_y); // BLOCK_64X64
    let mu_w = MI_W[uv_unit_bsize].min(blocks_wide_visible);
    let mu_h = MI_H[uv_unit_bsize].min(blocks_high_visible);
    let mut chunk_r = 0usize;
    while chunk_r < blocks_high_visible {
        let unit_h = (chunk_r + mu_h).min(blocks_high_visible);
        let mut chunk_c = 0usize;
        while chunk_c < blocks_wide_visible {
            let unit_w = (chunk_c + mu_w).min(blocks_wide_visible);
            let mut blk_row = chunk_r;
            while blk_row < unit_h {
                let mut blk_col = chunk_c;
                while blk_col < unit_w {
            if exit_early {
                return None; // intra: exit_early alone invalidates
            }
            let txb_off = env.ref_off[pi] + (blk_row * env.ref_stride + blk_col) * 4;
            if let Some(pal) = palette {
                // av1_predict_intra_block's use_palette arm (plane 1/2): the
                // colour-index map fill — no spatial prediction.
                let colors = if plane == 1 {
                    pal.colors_u
                } else {
                    pal.colors_v
                };
                let (x, y) = (blk_col * 4, blk_row * 4);
                for r in 0..txh {
                    for c in 0..txw {
                        recon[txb_off + r * env.ref_stride + c] =
                            colors[pal.map[(r + y) * pal.map_stride + c + x] as usize];
                    }
                }
            } else {
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
            }
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
            // Chroma frame-edge visible extent of THIS txb (subsampled).
            let (vis_c, vis_r) = get_txb_visible_dimensions(
                MI_W[plane_bsize] * 4,
                MI_H[plane_bsize] * 4,
                txw,
                txh,
                blk_row,
                blk_col,
                mb_to_right,
                mb_to_bottom,
                env.ss_x as u32,
                env.ss_y as u32,
            );
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
                visible_cols: vis_c,
                visible_rows: vis_r,
                qm_level: env.qm_levels.map(|l| l[plane]),
                predict_skip_zero_blk_rate,
            };
            // Same unguarded C subtraction as `txfm_rd_in_plane_intra`'s luma
            // walk (tx_search.c `block_rd_txfm` is plane-generic) -- replicate
            // C's raw int64_t wraparound with `wrapping_sub`, not
            // `saturating_sub`. See the detailed comment at
            // `tx_search::txfm_rd_in_plane_intra`'s analogous call site.
            let win = search_tx_type_intra(&inp, pol, ref_best_rd.wrapping_sub(current_rd))
                .expect("search_tx_type always yields a winner");

            // recon_intra: reconstruct the winner over the prediction.
            if win.best_eob > 0 {
                let mut tight = pred.clone();
                aom_transform::inv_txfm2d::av1_inverse_transform_add(
                    &win.dqcoeff,
                    &mut tight,
                    txw,
                    win.best_tx_type,
                    tx_size,
                    i32::from(env.bd),
                    win.best_eob as usize,
                    env.lossless,
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
            chunk_c += mu_w;
        }
        chunk_r += mu_h;
    }

    if exit_early {
        return None;
    }
    Some((stats, winners))
}

/// A palette-UV candidate for the chroma RD walk
/// (`av1_predict_intra_block`'s `use_palette` arm on planes 1/2): per-txb
/// prediction is the colour-map fill from the plane's colour section. The
/// map covers the full `plane_block_width x plane_block_height` extent.
#[derive(Clone, Copy, Debug)]
pub struct PaletteUvPred<'a> {
    pub colors_u: &'a [u16; 8],
    pub colors_v: &'a [u16; 8],
    pub size: usize,
    pub map: &'a [u8],
    /// Map row stride (= chroma plane block width).
    pub map_stride: usize,
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
    txfm_uvrd_p(
        env,
        recon_u,
        recon_v,
        uv_mode,
        angle_delta_uv,
        ref_best_rd,
        pol,
        None,
    )
}

/// [`txfm_uvrd`] + an optional palette-UV candidate (the
/// `av1_rd_pick_palette_intra_sbuv` tx path — prediction becomes the
/// colour-map fill on both planes).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn txfm_uvrd_p(
    env: &UvRdEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    uv_mode: usize,
    angle_delta_uv: i32,
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
    palette: Option<&PaletteUvPred>,
) -> Option<(RdStats, Vec<TxbWinner>, Vec<TxbWinner>)> {
    debug_assert_ne!(
        uv_mode, UV_CFL_PRED,
        "CfL evaluates through cfl_rd_pick_alpha"
    );
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
        let (this_stats, winners) = txfm_rd_in_plane_uv_p(
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
            palette,
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

// ---------------------------------------------------------------------------
// CfL alpha search (intra_mode_search.c 586-848): cfl_compute_rd (fast SATD
// model / full per-plane RD) -> cfl_pick_plane_parameter (hill climb) ->
// cfl_pick_plane_rd (full RD around the estimate) -> cfl_rd_pick_alpha (the
// joint U x V sign/alpha combination scan).
// ---------------------------------------------------------------------------

/// `CFL_MAGS_SIZE` (enums.h): `(2 << CFL_ALPHABET_SIZE_LOG2) + 1` = 33 signed
/// alpha magnitudes (-16..=+16 around [`CFL_INDEX_ZERO`]).
pub const CFL_MAGS_SIZE: usize = 33;
/// `CFL_INDEX_ZERO` (enums.h): `CFL_ALPHABET_SIZE` = 16.
pub const CFL_INDEX_ZERO: i32 = 16;
const CFL_SIGN_ZERO: i32 = 0;
const CFL_SIGN_NEG: i32 = 1;
const CFL_SIGN_POS: i32 = 2;
const CFL_SIGNS: i32 = 3;

/// `cfl_idx_to_sign_and_alpha` (intra_mode_search.c:589): linear index
/// (0..33) -> (sign, coded alpha magnitude).
pub fn cfl_idx_to_sign_and_alpha(cfl_idx: i32) -> (i32, i32) {
    let cfl_linear_idx = cfl_idx - CFL_INDEX_ZERO;
    if cfl_linear_idx == 0 {
        (CFL_SIGN_ZERO, 0)
    } else {
        let sign = if cfl_linear_idx > 0 {
            CFL_SIGN_POS
        } else {
            CFL_SIGN_NEG
        };
        (sign, cfl_linear_idx.abs() - 1)
    }
}

/// `PLANE_SIGN_TO_JOINT_SIGN(plane, a, b)` (intra_mode_search.c:586):
/// `pred_plane` is `CFL_PRED_U`(0) / `CFL_PRED_V`(1).
pub fn plane_sign_to_joint_sign(pred_plane: usize, a: i32, b: i32) -> i32 {
    if pred_plane == 0 {
        a * CFL_SIGNS + b - 1
    } else {
        b * CFL_SIGNS + a - 1
    }
}

/// `RD_STATS` as the CfL joint scan uses it (rd.h): the full merge /
/// invalidate / rd-update semantics, including the `rdcost` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CflRdStats {
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub skip_txfm: bool,
    pub zero_rate: i32,
    pub rdcost: i64,
}

impl CflRdStats {
    /// `av1_invalid_rd_stats`.
    pub fn invalid() -> Self {
        CflRdStats {
            rate: i32::MAX,
            dist: i64::MAX,
            sse: i64::MAX,
            skip_txfm: false,
            zero_rate: 0,
            rdcost: i64::MAX,
        }
    }
    /// `av1_merge_rd_stats` (rd.h:156): rate saturates (invalid propagates as
    /// full invalidation), dist adds, sse adds under `INT64_MAX` guards,
    /// skip ANDs.
    pub fn merge(&mut self, o: &CflRdStats) {
        if self.rate == i32::MAX || o.rate == i32::MAX {
            *self = CflRdStats::invalid();
            return;
        }
        self.rate = (i64::from(self.rate) + i64::from(o.rate)).min(i64::from(i32::MAX)) as i32;
        if self.zero_rate == 0 {
            self.zero_rate = o.zero_rate;
        }
        self.dist += o.dist;
        if self.sse < i64::MAX && o.sse < i64::MAX {
            self.sse += o.sse;
        }
        self.skip_txfm &= o.skip_txfm;
    }
    /// `av1_rd_cost_update` (rd.h:201).
    pub fn rd_cost_update(&mut self, rdmult: i32) {
        if self.rate < i32::MAX && self.dist < i64::MAX && self.rdcost < i64::MAX {
            self.rdcost = rdcost(rdmult, self.rate, self.dist);
        } else {
            *self = CflRdStats::invalid();
        }
    }
}

/// `intra_model_rd` (intra_mode_search_utils.h:622) for a CHROMA plane with
/// `use_hadamard == 0` — the CfL fast-mode model: per model-txb predict INTO
/// the recon plane (the CfL DC+AC facade path) -> subtract ->
/// `av1_quick_txfm` (a real DCT_DCT forward transform) -> `aom_satd`,
/// accumulated i64.
#[allow(clippy::too_many_arguments)]
pub fn intra_model_rd_uv(
    env: &UvRdEnv,
    recon: &mut [u16],
    plane: usize,
    cfl: &mut CflPredict,
    tx_size: usize,
) -> i64 {
    let plane_bsize = get_plane_block_size(env.bsize, env.ss_x, env.ss_y);
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let (txw_unit, txh_unit) = (txw >> 2, txh >> 2);
    let pi = plane - 1;
    let src = env.src(plane);
    let n = txw * txh;
    // Frame-edge clip (max_block_wide/high, chroma-subsampled).
    let (blocks_wide_visible, blocks_high_visible, _mbr, _mbb) = max_block_units(
        env.mi_cols,
        env.mi_rows,
        env.mi_col,
        env.mi_row,
        MI_W[env.bsize] as i32,
        MI_H[env.bsize] as i32,
        MI_W[plane_bsize] * 4,
        MI_H[plane_bsize] * 4,
        env.ss_x,
        env.ss_y,
    );

    let mut satd_cost: i64 = 0;
    let mut blk_row = 0usize;
    while blk_row < blocks_high_visible {
        let mut blk_col = 0usize;
        while blk_col < blocks_wide_visible {
            let txb_off = env.ref_off[pi] + (blk_row * env.ref_stride + blk_col) * 4;
            predict_uv_txb(
                env,
                recon,
                plane,
                UV_CFL_PRED,
                0,
                Some(cfl),
                tx_size,
                blk_row,
                blk_col,
                txb_off,
            );
            let mut pred = vec![0u16; n];
            for r in 0..txh {
                pred[r * txw..r * txw + txw].copy_from_slice(
                    &recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw],
                );
            }
            let src_txb_off = env.src_off[pi] + (blk_row * env.src_stride + blk_col) * 4;
            let mut residual = vec![0i16; n];
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
            // av1_quick_txfm(use_hadamard=0): DCT_DCT forward transform.
            let mut coeff = vec![0i32; n];
            aom_transform::txfm2d::av1_fwd_txfm2d(&residual, &mut coeff, txw, 0, tx_size);
            satd_cost += i64::from(aom_dist::hadamard::satd(&coeff[..n]));
            blk_col += txw_unit;
        }
        blk_row += txh_unit;
    }
    satd_cost
}

/// `cfl_compute_rd` (intra_mode_search.c:601): evaluate one CfL alpha index
/// on one plane — fast mode = the SATD model ([`intra_model_rd_uv`]); full
/// mode = `av1_txfm_rd_in_plane` (budget-free) + `av1_rd_cost_update`.
/// The evaluated plane's `(sign, alpha)` derive from `cfl_idx`; the other
/// plane's sign is the dummy `CFL_SIGN_NEG`; both alpha nibbles are set to
/// the evaluated alpha (`(alpha << 4) + alpha`).
/// Returns `(cfl_cost, Option<full-RD stats>)`.
#[allow(clippy::too_many_arguments)]
pub fn cfl_compute_rd(
    env: &UvRdEnv,
    recon: &mut [u16],
    plane: usize,
    ctx: &mut CflCtx,
    cache: &mut CflDcCache,
    tx_size: usize,
    cfl_idx: i32,
    fast_mode: bool,
    pol: &TxTypeSearchPolicy,
) -> (i64, Option<CflRdStats>) {
    let pred_plane = plane - 1;
    let (cfl_sign, cfl_alpha) = cfl_idx_to_sign_and_alpha(cfl_idx);
    let dummy_sign = CFL_SIGN_NEG;
    let joint_sign = plane_sign_to_joint_sign(pred_plane, cfl_sign, dummy_sign);
    let alpha_idx = (cfl_alpha << 4) + cfl_alpha; // CFL_ALPHABET_SIZE_LOG2
    let mut cflp = CflPredict {
        ctx,
        cache,
        alpha_idx,
        joint_sign,
    };

    if fast_mode {
        let cost = intra_model_rd_uv(env, recon, plane, &mut cflp, tx_size);
        (cost, None)
    } else {
        let Some((stats, _winners)) = txfm_rd_in_plane_uv(
            env,
            recon,
            plane,
            UV_CFL_PRED,
            0,
            Some(&mut cflp),
            tx_size,
            i64::MAX, // cfl_compute_rd passes INT64_MAX — no early exit
            0,
            pol,
        ) else {
            unreachable!("budget-free UV walk is always valid");
        };
        let mut s = CflRdStats {
            rate: stats.rate,
            dist: stats.dist,
            sse: stats.sse,
            skip_txfm: stats.skip_txfm,
            zero_rate: 0,
            rdcost: 0,
        };
        s.rd_cost_update(env.rdmult);
        (s.rdcost, Some(s))
    }
}

/// `cfl_pick_plane_parameter` (intra_mode_search.c:640): the fast-SATD hill
/// climb around `CFL_INDEX_ZERO` (each direction walks while strictly
/// improving). `cfl_search_range == CFL_MAGS_SIZE` (exhaustive full-RD mode)
/// short-circuits to `CFL_INDEX_ZERO`.
#[allow(clippy::too_many_arguments)]
pub fn cfl_pick_plane_parameter(
    env: &UvRdEnv,
    recon: &mut [u16],
    plane: usize,
    ctx: &mut CflCtx,
    cache: &mut CflDcCache,
    tx_size: usize,
    cfl_search_range: usize,
    pol: &TxTypeSearchPolicy,
) -> i32 {
    debug_assert!((1..=CFL_MAGS_SIZE).contains(&cfl_search_range));
    if cfl_search_range == CFL_MAGS_SIZE {
        return CFL_INDEX_ZERO;
    }
    let mut est_best_cfl_idx = CFL_INDEX_ZERO;
    let start_cfl_idx = CFL_INDEX_ZERO;
    let (mut best_cfl_cost, _) = cfl_compute_rd(
        env,
        recon,
        plane,
        ctx,
        cache,
        tx_size,
        start_cfl_idx,
        true,
        pol,
    );
    for dir in [1i32, -1] {
        for i in 1..CFL_MAGS_SIZE as i32 {
            let cfl_idx = start_cfl_idx + dir * i;
            if !(0..CFL_MAGS_SIZE as i32).contains(&cfl_idx) {
                break;
            }
            let (cfl_cost, _) =
                cfl_compute_rd(env, recon, plane, ctx, cache, tx_size, cfl_idx, true, pol);
            if cfl_cost < best_cfl_cost {
                best_cfl_cost = cfl_cost;
                est_best_cfl_idx = cfl_idx;
            } else {
                break;
            }
        }
    }
    est_best_cfl_idx
}

/// `cfl_pick_plane_rd` (intra_mode_search.c:683): full-RD evaluation of the
/// estimated best alpha and its `cfl_search_range - 1` neighbours in each
/// direction (all other entries stay invalid).
#[allow(clippy::too_many_arguments)]
pub fn cfl_pick_plane_rd(
    env: &UvRdEnv,
    recon: &mut [u16],
    plane: usize,
    ctx: &mut CflCtx,
    cache: &mut CflDcCache,
    tx_size: usize,
    cfl_search_range: usize,
    est_best_cfl_idx: i32,
    pol: &TxTypeSearchPolicy,
) -> [CflRdStats; CFL_MAGS_SIZE] {
    debug_assert!((1..=CFL_MAGS_SIZE).contains(&cfl_search_range));
    let mut arr = [CflRdStats::invalid(); CFL_MAGS_SIZE];
    let start_cfl_idx = est_best_cfl_idx;
    let (_, s) = cfl_compute_rd(
        env,
        recon,
        plane,
        ctx,
        cache,
        tx_size,
        start_cfl_idx,
        false,
        pol,
    );
    arr[start_cfl_idx as usize] = s.expect("full mode returns stats");
    if cfl_search_range == 1 {
        return arr;
    }
    for dir in [1i32, -1] {
        for i in 1..cfl_search_range as i32 {
            let cfl_idx = start_cfl_idx + dir * i;
            if !(0..CFL_MAGS_SIZE as i32).contains(&cfl_idx) {
                break;
            }
            let (_, s) =
                cfl_compute_rd(env, recon, plane, ctx, cache, tx_size, cfl_idx, false, pol);
            arr[cfl_idx as usize] = s.expect("full mode returns stats");
        }
    }
    arr
}

/// The winning CfL parameters of [`cfl_rd_pick_alpha`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CflAlphaResult {
    pub stats: CflRdStats,
    /// `mbmi->cfl_alpha_idx` — `(alpha_u << 4) + alpha_v`.
    pub alpha_idx: u8,
    /// `mbmi->cfl_alpha_signs` — the joint sign.
    pub joint_sign: i8,
}

/// `cfl_rd_pick_alpha` (intra_mode_search.c:745): the CfL mode evaluation —
/// per-plane fast hill climbs (DC-prediction cache enabled), the
/// `cfl_search_range == 1` invalid/overhead early-outs, per-plane full-RD
/// arrays, then the joint U x V scan (skipping invalid entries and the
/// ZERO/ZERO sign combination) with the CfL signaling rate folded in, strict
/// `<` on `rdcost`. `None` = the C's `return 0` (invalid parameters).
///
/// `uv_mode_cost` is `intra_uv_mode_cost[cfl_allowed][mbmi->mode][UV_CFL_PRED]`
/// (the `cfl_search_range == 1` overhead gate reads it).
#[allow(clippy::too_many_arguments)]
pub fn cfl_rd_pick_alpha(
    env: &UvRdEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    ctx: &mut CflCtx,
    tx_size: usize,
    ref_best_rd: i64,
    cfl_search_range: usize,
    cfl_costs: &crate::mode_costs::CflCosts,
    uv_mode_cost: i32,
    pol: &TxTypeSearchPolicy,
) -> Option<CflAlphaResult> {
    debug_assert!((1..=CFL_MAGS_SIZE).contains(&cfl_search_range));
    let mut cache = CflDcCache::cleared();
    // "enable the caching of dc pred data" — xd->cfl.use_dc_pred_cache = true.
    cache.use_cache = true;

    let est_best_cfl_idx_u = cfl_pick_plane_parameter(
        env,
        recon_u,
        1,
        ctx,
        &mut cache,
        tx_size,
        cfl_search_range,
        pol,
    );
    let est_best_cfl_idx_v = cfl_pick_plane_parameter(
        env,
        recon_v,
        2,
        ctx,
        &mut cache,
        tx_size,
        cfl_search_range,
        pol,
    );

    if cfl_search_range == 1 {
        // For cfl_search_range=1: CfL index 0 on both planes = invalid mode.
        if est_best_cfl_idx_u == CFL_INDEX_ZERO && est_best_cfl_idx_v == CFL_INDEX_ZERO {
            return None; // clear_cfl_dc_pred_cache_flags + return 0
        }
        let (cfl_sign_u, cfl_alpha_u) = cfl_idx_to_sign_and_alpha(est_best_cfl_idx_u);
        let (cfl_sign_v, cfl_alpha_v) = cfl_idx_to_sign_and_alpha(est_best_cfl_idx_v);
        let joint_sign = cfl_sign_u * CFL_SIGNS + cfl_sign_v - 1;
        let rate_overhead = cfl_costs.0[joint_sign as usize][0][cfl_alpha_u as usize]
            + cfl_costs.0[joint_sign as usize][1][cfl_alpha_v as usize]
            + uv_mode_cost;
        if rdcost(env.rdmult, rate_overhead, 0) > ref_best_rd {
            return None;
        }
    }

    let cfl_rd_arr_u = cfl_pick_plane_rd(
        env,
        recon_u,
        1,
        ctx,
        &mut cache,
        tx_size,
        cfl_search_range,
        est_best_cfl_idx_u,
        pol,
    );
    let cfl_rd_arr_v = cfl_pick_plane_rd(
        env,
        recon_v,
        2,
        ctx,
        &mut cache,
        tx_size,
        cfl_search_range,
        est_best_cfl_idx_v,
        pol,
    );
    // clear_cfl_dc_pred_cache_flags(&xd->cfl): the cache scope ends here (the
    // joint scan below re-evaluates nothing).

    let mut best: Option<CflAlphaResult> = None;
    let mut best_rdcost = i64::MAX; // av1_invalid_rd_stats(best_rd_stats)
    for (ui, u_entry) in cfl_rd_arr_u.iter().enumerate() {
        if u_entry.rate == i32::MAX {
            continue;
        }
        let (cfl_sign_u, cfl_alpha_u) = cfl_idx_to_sign_and_alpha(ui as i32);
        for (vi, v_entry) in cfl_rd_arr_v.iter().enumerate() {
            if v_entry.rate == i32::MAX {
                continue;
            }
            let (cfl_sign_v, cfl_alpha_v) = cfl_idx_to_sign_and_alpha(vi as i32);
            if cfl_sign_u == CFL_SIGN_ZERO && cfl_sign_v == CFL_SIGN_ZERO {
                continue; // not a valid CfL parameter combination
            }
            let joint_sign = cfl_sign_u * CFL_SIGNS + cfl_sign_v - 1;
            let mut rd_stats = *u_entry;
            rd_stats.merge(v_entry);
            if rd_stats.rate != i32::MAX {
                rd_stats.rate += cfl_costs.0[joint_sign as usize][0][cfl_alpha_u as usize];
                rd_stats.rate += cfl_costs.0[joint_sign as usize][1][cfl_alpha_v as usize];
            }
            rd_stats.rd_cost_update(env.rdmult);
            if rd_stats.rdcost < best_rdcost {
                best_rdcost = rd_stats.rdcost;
                best = Some(CflAlphaResult {
                    stats: rd_stats,
                    alpha_idx: ((cfl_alpha_u << 4) + cfl_alpha_v) as u8,
                    joint_sign: joint_sign as i8,
                });
            }
        }
    }
    match best {
        Some(b) if b.stats.rdcost < ref_best_rd => Some(b),
        // rdcost >= ref_best_rd: invalid stats + invalid parameters.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The UV mode loop (intra_mode_search.c 864-1029): rd_pick_intra_angle_sbuv
// (the chroma angle-delta sweep) + av1_rd_pick_intra_sbuv_mode (the
// 14-candidate search over uv_rd_search_mode_order).
// ---------------------------------------------------------------------------

use crate::mode_costs::{CflCosts, IntraModeCosts, intra_mode_info_cost_uv};
use aom_entropy::partition::{is_directional_mode, use_angle_delta};

/// `uv_rd_search_mode_order[UV_INTRA_MODES]` (intra_mode_search.c:44).
pub const UV_RD_SEARCH_MODE_ORDER: [usize; 14] = [0, 13, 2, 1, 9, 12, 10, 11, 4, 7, 6, 8, 5, 3];

/// `av1_derived_chroma_intra_mode_used_flag[INTRA_MODES]`
/// (intra_mode_search.c:87) — the `prune_chroma_modes_using_luma_winner`
/// mask (sf OFF at speed 0; kept for the gated arm).
pub const AV1_DERIVED_CHROMA_INTRA_MODE_USED_FLAG: [u16; 13] = [
    0x2201, 0x2203, 0x2205, 0x2209, 0x2211, 0x2221, 0x2241, 0x2281, 0x2301, 0x2201, 0x2601, 0x2a01,
    0x3201,
];

/// `av1_is_diagonal_mode` (reconintra.h): D45..D67.
#[inline]
pub fn is_diagonal_mode(mode: i32) -> bool {
    (3..=8).contains(&mode)
}

/// The mode-loop gates `av1_rd_pick_intra_sbuv_mode` reads, with speed-0
/// all-intra defaults (each named with provenance):
/// - `intra_uv_mode_mask` = `UV_INTRA_ALL` (init_intra_sf default; the
///   caller resolves `[txsize_sqr_up_map[max_tx_size]]`),
/// - `prune_chroma_modes_using_luma_winner = 0` (default; speed >= 4),
/// - `chroma_intra_pruning_with_hog = 0` (default; speed >= 3) — the chroma
///   HOG mask is a caller input (`None` disables, matching sf 0),
/// - `prune_smooth_intra_mode_for_chroma = 0` (default; speed >= 6),
/// - `cfl_search_range = 3` (init_intra_sf default; speed >= 6 sets 1),
/// - tool config `enable_*` all ON (aomenc defaults), `try_palette` OFF
///   (`allow_screen_content_tools = 0`).
#[derive(Clone, Debug)]
pub struct UvLoopPolicy {
    pub enable_diagonal_intra: bool,
    pub enable_directional_intra: bool,
    pub enable_smooth_intra: bool,
    pub enable_paeth_intra: bool,
    pub enable_cfl_intra: bool,
    pub enable_angle_delta: bool,
    /// `intra_uv_mode_mask[txsize_sqr_up_map[max_tx_size]]`, resolved.
    pub intra_uv_mode_mask: u16,
    pub prune_chroma_modes_using_luma_winner: bool,
    /// `intra_search_state.directional_mode_skip_mask` when the chroma HOG
    /// prune ran (`chroma_intra_pruning_with_hog > 0`); `None` = sf off.
    pub chroma_hog_skip_mask: Option<[bool; 13]>,
    pub prune_smooth_for_chroma: bool,
    pub cfl_search_range: usize,
    pub try_palette: bool,
}

impl UvLoopPolicy {
    /// Speed-0 all-intra defaults (provenance per field above).
    pub fn speed0_allintra() -> Self {
        UvLoopPolicy {
            enable_diagonal_intra: true,
            enable_directional_intra: true,
            enable_smooth_intra: true,
            enable_paeth_intra: true,
            enable_cfl_intra: true,
            enable_angle_delta: true,
            intra_uv_mode_mask: 0x3FFF, // UV_INTRA_ALL
            prune_chroma_modes_using_luma_winner: false,
            chroma_hog_skip_mask: None,
            prune_smooth_for_chroma: false,
            cfl_search_range: 3,
            try_palette: false,
        }
    }
}

/// One evaluated-candidate record (introspection for the differential: the
/// exact per-mode gating + rd sequence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UvModeVisit {
    pub uv_mode: usize,
    /// `None` = the candidate was gated/invalid before producing an rd.
    pub this_rd: Option<i64>,
}

/// The winner of [`rd_pick_intra_sbuv_mode`] — the `best_mbmi` chroma fields
/// + the returned rate/distortion outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UvModeResult {
    pub uv_mode: usize,
    pub angle_delta_uv: i32,
    pub cfl_alpha_idx: u8,
    pub cfl_alpha_signs: i8,
    pub rate: i32,
    pub rate_tokenonly: i32,
    pub dist: i64,
    pub skippable: bool,
    pub best_rd: i64,
    /// `best_mbmi.palette_mode_info` (UV half) + the winning chroma
    /// colour-index map — set when the post-loop UV palette search
    /// (`av1_rd_pick_palette_intra_sbuv`) wins (`uv_mode` is UV_DC_PRED then).
    pub palette_uv: Option<crate::palette_search::PaletteUvInfo>,
}

/// `pick_intra_angle_routine_sbuv` (intra_mode_search.c:496).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pick_intra_angle_routine_sbuv(
    env: &UvRdEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    uv_mode: usize,
    angle_delta_uv: i32,
    rate_overhead: i32,
    best_rd_in: i64,
    costs: &IntraModeCosts,
    pol: &TxTypeSearchPolicy,
    lp: &UvLoopPolicy,
    best_angle_delta: &mut i32,
    best_rd: &mut i64,
    best_stats: &mut Option<(i32, i64, bool)>,
) -> i64 {
    let Some((tokenonly, _wu, _wv)) = txfm_uvrd(
        env,
        recon_u,
        recon_v,
        uv_mode,
        angle_delta_uv,
        best_rd_in,
        pol,
    ) else {
        return i64::MAX;
    };
    let this_rate = tokenonly.rate
        + intra_mode_info_cost_uv(
            costs,
            rate_overhead,
            uv_mode,
            env.bsize,
            angle_delta_uv,
            lp.try_palette,
            env.luma_palette_active,
            false,
            0,
        );
    let this_rd = rdcost(env.rdmult, this_rate, tokenonly.dist);
    if this_rd < *best_rd {
        *best_rd = this_rd;
        *best_angle_delta = angle_delta_uv;
        *best_stats = Some((tokenonly.rate, tokenonly.dist, tokenonly.skip_txfm));
    }
    this_rd
}

/// `rd_pick_intra_angle_sbuv` (intra_mode_search.c:531): the chroma
/// angle-delta search — even deltas first (angle 0 evaluated once and
/// short-circuiting the whole search when invalid), then odd deltas gated by
/// the two even neighbours vs `best_rd + (best_rd >> 5)`. Threads (and
/// improves) the caller's `best_rd`. Returns
/// `Some((best_angle, rate_tokenonly, dist, skip, best_rd))` when a new best
/// was found (`rd_stats->rate != INT_MAX`), `None` otherwise.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn rd_pick_intra_angle_sbuv(
    env: &UvRdEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    uv_mode: usize,
    rate_overhead: i32,
    mut best_rd: i64,
    costs: &IntraModeCosts,
    pol: &TxTypeSearchPolicy,
    lp: &UvLoopPolicy,
) -> Option<(i32, i32, i64, bool, i64)> {
    let mut best_angle_delta = 0i32;
    let mut best_stats: Option<(i32, i64, bool)> = None;
    let mut rd_cost = [i64::MAX; 2 * (3 + 2)];

    let mut angle_delta = 0i32;
    while angle_delta <= 3 {
        for i in 0..2i32 {
            let best_rd_in = if best_rd == i64::MAX {
                i64::MAX
            } else {
                best_rd + (best_rd >> if angle_delta == 0 { 3 } else { 5 })
            };
            let this_rd = pick_intra_angle_routine_sbuv(
                env,
                recon_u,
                recon_v,
                uv_mode,
                (1 - 2 * i) * angle_delta,
                rate_overhead,
                best_rd_in,
                costs,
                pol,
                lp,
                &mut best_angle_delta,
                &mut best_rd,
                &mut best_stats,
            );
            rd_cost[(2 * angle_delta + i) as usize] = this_rd;
            if angle_delta == 0 {
                if this_rd == i64::MAX {
                    return None;
                }
                rd_cost[1] = this_rd;
                break;
            }
        }
        angle_delta += 2;
    }

    debug_assert!(best_rd != i64::MAX);
    let mut angle_delta = 1i32;
    while angle_delta <= 3 {
        for i in 0..2i32 {
            let rd_thresh = best_rd + (best_rd >> 5);
            let skip_search = rd_cost[(2 * (angle_delta + 1) + i) as usize] > rd_thresh
                && rd_cost[(2 * (angle_delta - 1) + i) as usize] > rd_thresh;
            if !skip_search {
                pick_intra_angle_routine_sbuv(
                    env,
                    recon_u,
                    recon_v,
                    uv_mode,
                    (1 - 2 * i) * angle_delta,
                    rate_overhead,
                    best_rd,
                    costs,
                    pol,
                    lp,
                    &mut best_angle_delta,
                    &mut best_rd,
                    &mut best_stats,
                );
            }
        }
        angle_delta += 2;
    }

    // mbmi->angle_delta[UV] = best_angle_delta; return rate != INT_MAX.
    best_stats.map(|(rate, dist, skip)| (best_angle_delta, rate, dist, skip, best_rd))
}

/// `av1_rd_pick_intra_sbuv_mode` (intra_mode_search.c:864) for a
/// chroma-reference intra block (the `!xd->is_chroma_ref` early return and
/// the `store_cfl_required_rdo` luma re-encode are the CALLER's: `cfl_ctx`
/// arrives pre-loaded with the winner-luma reconstruction — the
/// `av1_encode_intra_block_plane(AOM_PLANE_Y, DRY_RUN_NORMAL)` product).
/// Non-palette modes only (`try_palette` gates the out-of-scope UV palette
/// search; OFF without screen-content tools).
///
/// `mode_costs` supplies `intra_uv_mode_cost[cfl_allowed][luma_mode][uv]` +
/// the angle/palette-flag bits; `cfl_allowed` is `is_cfl_allowed(xd)`.
/// Returns the winner + the per-candidate visit log (gating/rd sequence).
#[allow(clippy::too_many_arguments)]
pub fn rd_pick_intra_sbuv_mode(
    env: &UvRdEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl_ctx: &mut CflCtx,
    max_tx_size: usize,
    cfl_allowed: bool,
    uv_mode_costs: &[[i32; 14]; 13],
    costs: &IntraModeCosts,
    cfl_costs: &CflCosts,
    pol: &TxTypeSearchPolicy,
    lp: &UvLoopPolicy,
    palette: Option<&crate::palette_search::UvPaletteArgs>,
) -> (UvModeResult, Vec<UvModeVisit>) {
    // init_sbuv_mode: uv_mode = UV_DC_PRED, palette_size[1] = 0.
    let mut best = UvModeResult {
        uv_mode: 0,
        angle_delta_uv: 0,
        cfl_alpha_idx: 0,
        cfl_alpha_signs: 0,
        rate: 0,
        rate_tokenonly: 0,
        dist: 0,
        skippable: false,
        best_rd: i64::MAX,
        palette_uv: None,
    };
    let mut visits: Vec<UvModeVisit> = Vec::new();
    let sqr_up = crate::tx_search::TXSIZE_SQR_UP_MAP[max_tx_size];
    let _ = sqr_up; // the caller resolved intra_uv_mode_mask by this class

    for &uv_mode in UV_RD_SEARCH_MODE_ORDER.iter() {
        let mut visit = UvModeVisit {
            uv_mode,
            this_rd: None,
        };
        // Mode-signaling-rate early skip (strict >).
        let mode_rate = uv_mode_costs[env.luma_mode][uv_mode];
        if rdcost(env.rdmult, mode_rate, 0) > best.best_rd {
            visits.push(visit);
            continue;
        }
        let intra_mode = get_uv_mode(uv_mode);
        let is_diagonal = is_diagonal_mode(intra_mode);
        let is_directional = is_directional_mode(intra_mode);
        if is_diagonal && !lp.enable_diagonal_intra {
            visits.push(visit);
            continue;
        }
        if is_directional && !lp.enable_directional_intra {
            visits.push(visit);
            continue;
        }
        if lp.intra_uv_mode_mask & (1 << uv_mode) == 0 {
            visits.push(visit);
            continue;
        }
        if !lp.enable_smooth_intra && (9..=11).contains(&uv_mode) {
            visits.push(visit);
            continue;
        }
        if !lp.enable_paeth_intra && uv_mode == 12 {
            visits.push(visit);
            continue;
        }
        debug_assert!(env.luma_mode < 13);
        if lp.prune_chroma_modes_using_luma_winner
            && AV1_DERIVED_CHROMA_INTRA_MODE_USED_FLAG[env.luma_mode] & (1 << uv_mode) == 0
        {
            visits.push(visit);
            continue;
        }

        // mbmi->uv_mode = uv_mode; angle_delta[UV] = 0.
        let mut angle_delta_uv = 0i32;
        let tokenonly: (i32, i64, bool);
        let mut cfl_fields: (u8, i8) = (0, 0);

        if uv_mode == UV_CFL_PRED {
            if !cfl_allowed || !lp.enable_cfl_intra {
                visits.push(visit);
                continue;
            }
            debug_assert!(!is_directional);
            let uv_tx_size = av1_get_tx_size_uv(env.bsize, env.lossless, env.ss_x, env.ss_y);
            let Some(r) = cfl_rd_pick_alpha(
                env,
                recon_u,
                recon_v,
                cfl_ctx,
                uv_tx_size,
                best.best_rd,
                lp.cfl_search_range,
                cfl_costs,
                uv_mode_costs[env.luma_mode][UV_CFL_PRED],
                pol,
            ) else {
                visits.push(visit);
                continue;
            };
            tokenonly = (r.stats.rate, r.stats.dist, r.stats.skip_txfm);
            cfl_fields = (r.alpha_idx, r.joint_sign);
        } else if is_directional && use_angle_delta(env.bsize) && lp.enable_angle_delta {
            // Chroma HOG prune (sf 0 at speed 0 => mask None).
            if let Some(mask) = &lp.chroma_hog_skip_mask
                && mask[uv_mode]
            {
                visits.push(visit);
                continue;
            }
            let rate_overhead = uv_mode_costs[env.luma_mode][uv_mode];
            let Some((best_angle, rate, dist, skip, new_best)) = rd_pick_intra_angle_sbuv(
                env,
                recon_u,
                recon_v,
                uv_mode,
                rate_overhead,
                best.best_rd,
                costs,
                pol,
                lp,
            ) else {
                visits.push(visit);
                continue;
            };
            let _ = new_best; // best_rd re-derives from this_rd below
            angle_delta_uv = best_angle;
            tokenonly = (rate, dist, skip);
        } else {
            if uv_mode == 9 && lp.prune_smooth_for_chroma {
                // should_prune_chroma_smooth_pred_based_on_source_variance:
                // sf OFF at speed 0; the variance body is the gated caller's.
                visits.push(visit);
                continue;
            }
            let Some((stats, _wu, _wv)) =
                txfm_uvrd(env, recon_u, recon_v, uv_mode, 0, best.best_rd, pol)
            else {
                visits.push(visit);
                continue;
            };
            tokenonly = (stats.rate, stats.dist, stats.skip_txfm);
        }

        let mode_cost = uv_mode_costs[env.luma_mode][uv_mode];
        let this_rate = tokenonly.0
            + intra_mode_info_cost_uv(
                costs,
                mode_cost,
                uv_mode,
                env.bsize,
                angle_delta_uv,
                lp.try_palette,
                env.luma_palette_active,
                false,
                0,
            );
        let this_rd = rdcost(env.rdmult, this_rate, tokenonly.1);
        visit.this_rd = Some(this_rd);
        visits.push(visit);

        if this_rd < best.best_rd {
            best = UvModeResult {
                uv_mode,
                angle_delta_uv,
                cfl_alpha_idx: cfl_fields.0,
                cfl_alpha_signs: cfl_fields.1,
                rate: this_rate,
                rate_tokenonly: tokenonly.0,
                dist: tokenonly.1,
                skippable: tokenonly.2,
                best_rd: this_rd,
                palette_uv: None,
            };
        }
    }

    // try_palette = enable_palette && av1_allow_palette(..)
    // (intra_mode_search.c:1012): the UV palette search.
    if let Some(pal_args) = palette {
        if lp.try_palette {
            crate::palette_search::rd_pick_palette_intra_sbuv(
                env, recon_u, recon_v, pal_args, costs, &mut best,
            );
        }
    }
    // *mbmi = best_mbmi; assert a mode was chosen.
    assert!(best.best_rd < i64::MAX, "sbuv search must choose a mode");
    (best, visits)
}
