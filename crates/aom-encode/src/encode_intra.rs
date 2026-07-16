//! `av1_encode_intra_block_plane` (av1/encoder/encodemb.c:801) — the winner
//! re-encode pass.
//!
//! After the RD searches pick a block's winner modes / tx layout, the encoder
//! re-encodes the block for real: a per-txb `av1_foreach_transformed_block_in_plane`
//! walk running `encode_block_intra` + `av1_set_txb_context`
//! (`encode_block_intra_and_set_context`, encodemb.c:788). Per txb:
//!
//! 1. `av1_predict_intra_block_facade` — predict INTO the recon plane.
//! 2. skip arm (`xd->mi[0]->skip_txfm`): `eob = 0`, `txb_entropy_ctx = 0`, no
//!    transform (encodemb.c:722-724). In the KEY-frame intra RD path this arm
//!    is dead: `pick_sb_modes` zeroes `mbmi->skip_txfm` (partition_search.c:910)
//!    and nothing in the intra RD path sets it (the `mbmi->skip_txfm = 1`
//!    writers live in `av1_txfm_search`, inter-only — tx_search.c:3878).
//! 3. else: `av1_subtract_txb` -> `tx_type = av1_get_tx_type(Y)` (the
//!    tx_type_map read, [`get_tx_type_y`]) -> `av1_xform_quant` with
//!    `quant_idx = use_trellis ? AV1_XFORM_QUANT_FP : AV1_XFORM_QUANT_B`
//!    (`USE_B_QUANT_NO_TRELLIS == 1`, encodemb.c:737-741) -> when trellis:
//!    `get_txb_ctx` + `av1_optimize_b` (rate discarded — `dummy_rate_cost`).
//! 4. `if (*eob) av1_inverse_transform_block` — reconstruct into the recon
//!    plane (encodemb.c:759-763).
//! 5. `if (*eob == 0 && plane == 0) update_txk_array(.., DCT_DCT)` — the
//!    tx_type_map reset (encodemb.c:770-779), [`update_txk_array`].
//! 6. `if (plane == AOM_PLANE_Y && xd->cfl.store_y) cfl_store_tx(..)` — load
//!    the CfL context from the just-reconstructed luma (encodemb.c:781-785).
//! 7. `av1_set_txb_context` — stamp `txb_entropy_ctx` over the txb's
//!    above/left units (encodemb.h:141-147; full-footprint memset, NOT the
//!    frame-clipped `av1_set_entropy_contexts`).
//!
//! ## Final-encode trellis gating (verified against the default encoder config)
//!
//! `enable_optimize_b` at every intra call site is
//! `cpi->optimize_seg_arr[segment_id]` (intra_mode_search.c:899,
//! partition_search.c:422, tx_search.c:2101), which encodeframe.c:2266-2273
//! sets to `NO_TRELLIS_OPT` for lossless segments and to
//! `sf.rd_sf.optimize_coefficients` otherwise. With the default
//! `--disable-trellis-quant=0`, `init_rd_sf` (speed_features.c:2488-2493)
//! yields `FULL_TRELLIS_OPT` for non-lossless. `is_trellis_used`
//! (encodemb.h:153-159) then returns true regardless of `dry_run` (only
//! `FINAL_PASS_TRELLIS_OPT` checks `dry_run != OUTPUT_ENABLED`), so the
//! speed-0 final encode is ALWAYS `AV1_XFORM_QUANT_FP` + `av1_optimize_b`.
//! `av1_optimize_b` itself (encodemb.c:87-103) short-circuits to
//! `av1_cost_skip_txb` when `eob == 0 || !optimize_seg_arr[seg] ||
//! lossless[seg]`; the two non-eob outs are unreachable whenever
//! `use_trellis` is true (lossless forces `NO_TRELLIS_OPT` upstream), which
//! is exactly [`crate::xform_quant_optimize`]'s model. `av1_dropout_qcoeff`
//! has ZERO call sites in libaom v3.14.1 (definition only) — dropout is NOT
//! part of any encode path.
//!
//! ## Scope
//!
//! All 3 plane arms: [`encode_intra_block_plane_y`] (luma) and
//! [`encode_intra_block_plane_uv`] (each chroma plane; the `plane &&
//! !xd->is_chroma_ref` early return is the CALLER's gate — encodemb.c:806).
//! The chroma arm differs from luma per the C body exactly by: prediction
//! from `get_uv_mode(mbmi->uv_mode)` incl. the signalled-CfL path
//! (`av1_predict_intra_block_facade`'s `uv_mode == UV_CFL_PRED` arm predicts
//! DC then applies the WINNER's `cfl_alpha_idx`/`cfl_alpha_signs` AC; the
//! DC-prediction cache is INACTIVE outside `cfl_rd_pick_alpha`, so the
//! fresh-DC path runs every txb); `av1_get_tx_type`'s PLANE_TYPE_UV arm
//! ([`crate::tx_search::uv_intra_tx_type`]); the chroma trellis rd
//! multiplier (`plane_rd_mult[0][PLANE_TYPE_UV] = 13`); NO
//! `update_txk_array` reset (`plane == 0` gate, encodemb.c:770) and NO
//! `cfl_store_tx` (`plane == AOM_PLANE_Y` gate, encodemb.c:782).
//!
//! MISSING: frame-edge clipped walks (`max_block_wide/high` —
//! interior blocks only, same scope as the luma/chroma RD walks); block sizes
//! above 64x64 (the `mu_blocks` outer walk of
//! `av1_foreach_transformed_block_in_plane` degenerates to a plain raster for
//! `bsize <= 64x64`, encodemb.c:560-582 — sb128 out of the current envelope).
//!
//! The `tx_type_map` here is the RDO-time BLOCK-LOCAL buffer
//! (`xd->tx_type_map = txfm_info->tx_type_map_`, stride
//! `mi_size_wide[bsize]` — partition_search.c:895-896). Only txb-origin cells
//! are ever read on the KEY-frame path (`av1_get_tx_type` Y reads the origin;
//! intra UV never reads the luma map); non-origin cells are dead state.

use crate::intra_uv_rd::{CflDcCache, CflPredict, UV_CFL_PRED, UvRdEnv, predict_uv_txb};
use crate::tx_search::{
    BLK_H_B, BLK_W_B, MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TXS_H, TXS_W, TXSIZE_SQR_UP_MAP,
    max_block_units, trellis_rdmult_intra, trellis_rdmult_intra_y, uv_intra_tx_type,
};
use crate::{
    BlockContext, OptimizeInputs, QuantKind, QuantParams, xform_quant, xform_quant_optimize,
};
use aom_dist::highbd_subtract_block;
use aom_entropy::partition::{get_plane_block_size, intra_avail};
use aom_intra::cfl::{CflCtx, cfl_store_tx};
use aom_intra::predict_intra_high;
use aom_transform::inv_txfm2d::av1_inverse_transform_add;
use aom_txb::{CoeffCostTables, get_txb_ctx};

/// `TRELLIS_OPT_TYPE` (encodemb.h:43-48). C-valued discriminants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrellisOptType {
    /// `NO_TRELLIS_OPT` — no trellis optimization.
    NoTrellisOpt = 0,
    /// `FULL_TRELLIS_OPT` — trellis in all stages (speed-0 default,
    /// `--disable-trellis-quant=0` non-lossless).
    FullTrellisOpt = 1,
    /// `FINAL_PASS_TRELLIS_OPT` — trellis only in the final encode pass.
    FinalPassTrellisOpt = 2,
    /// `NO_ESTIMATE_YRD_TRELLIS_OPT` — trellis except in `estimate_yrd_for_sb`.
    NoEstimateYrdTrellisOpt = 3,
}

/// `is_trellis_used` (encodemb.h:153-159). `dry_run_output_enabled` is
/// `dry_run == OUTPUT_ENABLED` (tokenize.h: `OUTPUT_ENABLED = 0`,
/// `DRY_RUN_NORMAL = 1`; the sbuv-preamble re-encode passes `DRY_RUN_NORMAL`).
pub fn is_trellis_used(optimize_b: TrellisOptType, dry_run_output_enabled: bool) -> bool {
    if optimize_b == TrellisOptType::NoTrellisOpt {
        return false;
    }
    if optimize_b == TrellisOptType::FinalPassTrellisOpt && !dry_run_output_enabled {
        return false;
    }
    true
}

/// `av1_get_tx_type` (blockd.h:1283) — the `PLANE_TYPE_Y` arm: lossless or
/// `txsize_sqr_up_map[tx_size] > TX_32X32` returns `DCT_DCT`; otherwise the
/// block-local tx_type_map cell at `(blk_row, blk_col)`. The Y arm has NO
/// demote-to-DCT check (unlike UV) — the map invariantly holds in-set types
/// (C asserts `av1_ext_tx_used[set][type]`, live in the shim build).
pub fn get_tx_type_y(
    lossless: bool,
    tx_size: usize,
    tx_type_map: &[u8],
    map_stride: usize,
    blk_row: usize,
    blk_col: usize,
) -> usize {
    const TX_32X32: usize = 3;
    if lossless || TXSIZE_SQR_UP_MAP[tx_size] > TX_32X32 {
        return 0; // DCT_DCT
    }
    tx_type_map[blk_row * map_stride + blk_col] as usize
}

/// `update_txk_array` (blockd.h:1260-1281): stamp `tx_type` at the txb origin
/// cell, plus — for 64-wide/-high tx sizes — every 16x16 unit inside the txb
/// (the chroma-max-32x32 constraint workaround the C comments describe).
pub fn update_txk_array(
    tx_type_map: &mut [u8],
    map_stride: usize,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    tx_type: usize,
) {
    tx_type_map[blk_row * map_stride + blk_col] = tx_type as u8;
    let txw = TXS_W[tx_size] >> 2;
    let txh = TXS_H[tx_size] >> 2;
    // tx_size_wide_unit[TX_64X64] == 16; tx_size_wide_unit[TX_16X16] == 4.
    if txw == 16 || txh == 16 {
        let tx_unit = 4usize;
        let mut idy = 0;
        while idy < txh {
            let mut idx = 0;
            while idx < txw {
                tx_type_map[(blk_row + idy) * map_stride + blk_col + idx] = tx_type as u8;
                idx += tx_unit;
            }
            idy += tx_unit;
        }
    }
}

/// The MACROBLOCK(D) state `av1_encode_intra_block_plane` reads for the LUMA
/// plane, as plain data (the [`crate::tx_search::TxfmYrdEnv`] convention).
pub struct EncodeIntraYEnv<'a> {
    // intra_avail frame geometry (aom_entropy::partition::intra_avail).
    pub sb_size: usize,
    /// `mbmi->bsize` (luma block size; also the plane_bsize for plane 0).
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
    // Pixel planes: `recon[ref_off]` = block top-left in the reconstruction
    // plane (prediction reads + reconstruction writes); `src[src_off]` in the
    // source plane.
    pub ref_off: usize,
    pub ref_stride: usize,
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    // Prediction config.
    pub disable_edge_filter: bool,
    pub filter_type: i32,
    // Winner mode info (the mbmi fields the walk reads).
    pub mode: usize,
    /// Unscaled angle delta (x3 `ANGLE_STEP` applied internally).
    pub angle_delta: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    /// `av1_get_tx_size(AOM_PLANE_Y, xd)` = `mbmi->tx_size` (the uniform
    /// winner size; lossless forces TX_4X4 upstream).
    pub tx_size: usize,
    /// `xd->mi[0]->skip_txfm` (0 throughout the KEY-frame intra RD path).
    pub skip_txfm: bool,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub bd: u8,
    // Quantizer + trellis.
    pub rows: &'a aom_quant::PlaneQuantRows<'a>,
    /// `x->rdmult` (the trellis rdmult derives via [`trellis_rdmult_intra_y`]).
    pub rdmult: i32,
    /// `cpi->oxcf.algo_cfg.sharpness` (0 default).
    pub sharpness: i32,
    /// Coefficient cost tables at the block's (txs_ctx, PLANE_TYPE_Y) — the
    /// trellis' rate inputs (the walk discards the returned rate, C's
    /// `dummy_rate_cost`).
    pub coeff_costs: &'a CoeffCostTables<'a>,
    /// `cpi->optimize_seg_arr[mbmi->segment_id]`.
    pub enable_optimize_b: TrellisOptType,
    /// `dry_run == OUTPUT_ENABLED` (the sbuv preamble passes DRY_RUN_NORMAL
    /// => false).
    pub dry_run_output_enabled: bool,
    /// The block's above/left entropy contexts (read only when
    /// `enable_optimize_b != NO_TRELLIS_OPT`, encodemb.c:817-819).
    pub above_ctx: &'a [i8],
    pub left_ctx: &'a [i8],
}

/// One re-encoded txb's outputs (the `p->qcoeff/dqcoeff/eobs/txb_entropy_ctx`
/// slots plus the tx_type actually used).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxbEncode {
    /// The tx type used by the transform (skip arm: DCT_DCT).
    pub tx_type: usize,
    pub eob: u16,
    pub txb_entropy_ctx: u8,
    /// Quantized / dequantized coefficients (empty on the skip arm — the C
    /// leaves the shared scratch buffers untouched there).
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    /// `get_txb_ctx`'s `(txb_skip_ctx, dc_sign_ctx)` derived for this txb from
    /// the *pre*-write neighbour contexts (the same pair the trellis used to
    /// select its rate tables, exposed here for the pack-stage coefficient
    /// writer — `av1_write_coeffs_txb`/[`write_coeffs_txb_full`](aom_txb::write_coeffs_txb_full)
    /// need the identical values the RD search already computed). `0` on the
    /// `skip_txfm` arm (dead in the KEY intra envelope).
    pub txb_skip_ctx: usize,
    pub dc_sign_ctx: usize,
}

/// The walk's outputs: per-txb results in raster order plus the final local
/// entropy-context arrays (C-local `ta`/`tl`, discarded by the C caller —
/// exposed for differential visibility; the tile-level contexts are NOT
/// written by this pass).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeIntraPlaneOutcome {
    pub txbs: Vec<TxbEncode>,
    pub ta: Vec<i8>,
    pub tl: Vec<i8>,
}

/// `av1_encode_intra_block_plane(cpi, x, bsize, AOM_PLANE_Y, dry_run,
/// enable_optimize_b)` (encodemb.c:801-823) — see the module docs for the
/// per-txb sequence and gating. `recon` is predicted into and reconstructed
/// in place; `tx_type_map` (stride `mi_size_wide[bsize]`) is read per txb and
/// reset to DCT_DCT at `eob == 0`; `cfl` = `Some` models `xd->cfl.store_y`
/// (the sbuv preamble sets it via `store_cfl_required_rdo`,
/// intra_mode_search.c:890) and receives every txb's reconstructed luma.
pub fn encode_intra_block_plane_y(
    env: &EncodeIntraYEnv,
    recon: &mut [u16],
    tx_type_map: &mut [u8],
    mut cfl: Option<&mut CflCtx>,
) -> EncodeIntraPlaneOutcome {
    let bsize = env.bsize;
    let (bw, bh) = (BLK_W_B[bsize], BLK_H_B[bsize]);
    debug_assert!(
        bw <= 64 && bh <= 64,
        "mu-64 outer walk degenerates to raster only for bsize <= 64x64"
    );
    let tx_size = env.tx_size;
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let (txw_unit, txh_unit) = (txw >> 2, txh >> 2);
    let max_blocks_wide = MI_SIZE_WIDE_B[bsize];
    let max_blocks_high = MI_SIZE_HIGH_B[bsize];
    let map_stride = max_blocks_wide;
    // C clips the tx-block LOOP to the visible frame area (`max_block_wide`/
    // `max_block_high`, av1_common_int.h:1568): a partial edge block's off-frame
    // tx sub-blocks are never predicted/coded (the predictor asserts
    // `n_top_px >= 0`). Luma plane, so no subsampling. The map stride and
    // neighbour-context arrays stay the FULL block width.
    let blocks_wide_visible = max_blocks_wide.min((env.mi_cols - env.mi_col).max(0) as usize);
    let blocks_high_visible = max_blocks_high.min((env.mi_rows - env.mi_row).max(0) as usize);

    // ta/tl are the running per-txb entropy context (init from the block's real
    // above/left neighbour context, then stamped per-txb below). They feed BOTH
    // the trellis (`av1_optimize_b`, when enabled) AND the `get_txb_ctx` that
    // derives the WRITTEN `txb_skip_ctx`/`dc_sign_ctx` (the non-optimize arm
    // below). C keeps these separate: the trellis fills its own `ta/tl` only when
    // `enable_optimize_b` (encodemb.c:817-819), but the write path ALWAYS derives
    // its ctx from the real `pd->above/left_entropy_context` (encodetxb.c:596-598,
    // and the search coeff cost via `av1_cost_coeffs_txb` does the same) — never
    // gated on the trellis. Because the port shares one array for both uses, it
    // must ALWAYS seed from the real neighbour context; gating this on the trellis
    // (as an earlier version did) zeroed the cross-block context for coded-lossless
    // (trellis-off), so a block with a coded left neighbour (e.g. the right half of
    // a VERT split) got txb_skip_ctx=1/dc_sign_ctx=0 instead of the real 3/1 and
    // the coded bytes desynced the decoder. Within-block propagation was already
    // fine; only this first-txb cross-block seed was wrong.
    let mut ta = vec![0i8; max_blocks_wide];
    let mut tl = vec![0i8; max_blocks_high];
    ta.copy_from_slice(&env.above_ctx[..max_blocks_wide]);
    tl.copy_from_slice(&env.left_ctx[..max_blocks_high]);
    let use_trellis = is_trellis_used(env.enable_optimize_b, env.dry_run_output_enabled);

    let mut txbs: Vec<TxbEncode> = Vec::new();
    let mut blk_row = 0usize;
    while blk_row < blocks_high_visible {
        let mut blk_col = 0usize;
        while blk_col < blocks_wide_visible {
            // --- encode_block_intra ---
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
            for r in 0..txh {
                recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }

            let mut tx_type = 0usize; // DCT_DCT
            let (qcoeff, dqcoeff, eob, ent_ctx, txb_skip_ctx, dc_sign_ctx);
            if env.skip_txfm {
                // *eob = 0; p->txb_entropy_ctx[block] = 0 (encodemb.c:722-724).
                qcoeff = Vec::new();
                dqcoeff = Vec::new();
                eob = 0u16;
                ent_ctx = 0u8;
                // Dead arm in the KEY intra envelope (skip_txfm asserted 0 by
                // the caller); no txb_skip symbol is ever coded for it.
                txb_skip_ctx = 0;
                dc_sign_ctx = 0;
            } else {
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

                tx_type = get_tx_type_y(
                    env.lossless,
                    tx_size,
                    tx_type_map,
                    map_stride,
                    blk_row,
                    blk_col,
                );

                // quant_idx: use_trellis ? FP : (USE_B_QUANT_NO_TRELLIS ? B : FP).
                let kind = if use_trellis {
                    QuantKind::Fp
                } else {
                    QuantKind::B
                };
                let qp = QuantParams::from_plane_rows(env.rows, kind, env.bd, env.lossless);
                if use_trellis {
                    let bctx = BlockContext {
                        above: &ta[blk_col..],
                        left: &tl[blk_row..],
                        plane: 0,
                        plane_bsize: bsize,
                    };
                    let opt = OptimizeInputs {
                        cost: env.coeff_costs,
                        rdmult: trellis_rdmult_intra_y(env.rdmult, env.sharpness, env.bd),
                        sharpness: env.sharpness,
                    };
                    // av1_xform_quant(FP, use_optimize_b) + get_txb_ctx +
                    // av1_optimize_b; the rate is C's dummy_rate_cost.
                    let r =
                        xform_quant_optimize(&residual, tx_size, tx_type, kind, &qp, &bctx, &opt);
                    qcoeff = r.qcoeff;
                    dqcoeff = r.dqcoeff;
                    eob = r.eob;
                    ent_ctx = r.txb_entropy_ctx;
                    txb_skip_ctx = r.txb_skip_ctx;
                    dc_sign_ctx = r.dc_sign_ctx;
                } else {
                    let r = xform_quant(&residual, tx_size, tx_type, kind, &qp, false);
                    qcoeff = r.qcoeff;
                    dqcoeff = r.dqcoeff;
                    eob = r.eob;
                    ent_ctx = r.txb_entropy_ctx;
                    // get_txb_ctx: xform_quant (non-optimize_b) doesn't derive
                    // this internally (only the trellis needs it for its rate
                    // estimate) — the pack-stage writer needs it regardless.
                    let (sc, dc) = get_txb_ctx(bsize, tx_size, 0, &ta[blk_col..], &tl[blk_row..]);
                    txb_skip_ctx = sc as usize;
                    dc_sign_ctx = dc as usize;
                }
            }

            // if (*eob) av1_inverse_transform_block into the recon plane.
            if eob > 0 {
                let mut tight = pred.clone();
                av1_inverse_transform_add(
                    &dqcoeff,
                    &mut tight,
                    txw,
                    tx_type,
                    tx_size,
                    i32::from(env.bd),
                    eob as usize,
                    env.lossless,
                );
                for r in 0..txh {
                    recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }

            // if (*eob == 0 && plane == 0) update_txk_array(.., DCT_DCT).
            if eob == 0 {
                update_txk_array(tx_type_map, map_stride, blk_row, blk_col, tx_size, 0);
            }

            // if (plane == AOM_PLANE_Y && xd->cfl.store_y) cfl_store_tx(..).
            if let Some(ctx) = cfl.as_deref_mut() {
                cfl_store_tx(
                    ctx,
                    recon,
                    env.ref_off,
                    env.ref_stride,
                    blk_row as i32,
                    blk_col as i32,
                    tx_size,
                    bsize,
                    env.mi_row,
                    env.mi_col,
                );
            }

            // --- av1_set_txb_context (full-footprint memset) ---
            for a in ta[blk_col..blk_col + txw_unit].iter_mut() {
                *a = ent_ctx as i8;
            }
            for l in tl[blk_row..blk_row + txh_unit].iter_mut() {
                *l = ent_ctx as i8;
            }

            txbs.push(TxbEncode {
                tx_type,
                eob,
                txb_entropy_ctx: ent_ctx,
                qcoeff,
                dqcoeff,
                txb_skip_ctx,
                dc_sign_ctx,
            });
            blk_col += txw_unit;
        }
        blk_row += txh_unit;
    }

    EncodeIntraPlaneOutcome { txbs, ta, tl }
}

/// The winner CHROMA mode-info fields the UV re-encode reads from `mbmi`
/// (the RD pick's outputs).
#[derive(Clone, Copy, Debug)]
pub struct UvWinner {
    /// `mbmi->uv_mode` (UV_PREDICTION_MODE; `UV_CFL_PRED == 13`).
    pub uv_mode: usize,
    /// `mbmi->angle_delta[PLANE_TYPE_UV]`, unscaled.
    pub angle_delta_uv: i32,
    /// `mbmi->cfl_alpha_idx` (read only when `uv_mode == UV_CFL_PRED`).
    pub cfl_alpha_idx: i32,
    /// `mbmi->cfl_alpha_signs` (the joint sign).
    pub cfl_alpha_signs: i32,
}

/// The per-call knobs of the UV re-encode arm (the `encode_b_args` slice the
/// chroma walk reads; the luma arm carries these inside [`EncodeIntraYEnv`]).
#[derive(Clone, Copy, Debug)]
pub struct UvEncodeParams {
    /// `av1_get_tx_size(plane, xd)` for the chroma plane
    /// ([`crate::intra_uv_rd::av1_get_tx_size_uv`] — both planes share it).
    pub tx_size: usize,
    /// `xd->mi[0]->skip_txfm` (0 throughout the KEY-frame intra path).
    pub skip_txfm: bool,
    /// `cpi->oxcf.algo_cfg.sharpness` (0 default).
    pub sharpness: i32,
    /// `cpi->optimize_seg_arr[mbmi->segment_id]`.
    pub enable_optimize_b: TrellisOptType,
    /// `dry_run == OUTPUT_ENABLED`.
    pub dry_run_output_enabled: bool,
    /// sf `tx_sf.use_chroma_trellis_rd_mult` — ALLINTRA/RT 1 (chroma
    /// trellis multiplier 13), usage GOOD 0 (multiplier 20). See
    /// [`trellis_rdmult_intra`].
    pub use_chroma_trellis_rd_mult: bool,
}

/// `av1_encode_intra_block_plane(cpi, x, bsize, plane /* 1|2 */, dry_run,
/// enable_optimize_b)` (encodemb.c:801-823) — the CHROMA arm. See the module
/// docs for the delta vs the luma arm. The `plane && !xd->is_chroma_ref`
/// early return (encodemb.c:806) is the caller's gate; `env` carries the
/// block geometry/pixels exactly as for the UV RD walk ([`UvRdEnv`] — the
/// sub-8x8 mi rounding is baked into `ref_off`/`src_off`). `recon` is the
/// `plane` recon; `cfl` is the loaded CfL context (read only on the CfL
/// path — the luma re-encode/store must already have run, which the C
/// guarantees by plane order inside `encode_superblock`).
pub fn encode_intra_block_plane_uv(
    env: &UvRdEnv,
    winner: &UvWinner,
    prm: &UvEncodeParams,
    plane: usize,
    recon: &mut [u16],
    cfl: &mut CflCtx,
) -> EncodeIntraPlaneOutcome {
    debug_assert!(plane == 1 || plane == 2);
    let plane_bsize = get_plane_block_size(env.bsize, env.ss_x, env.ss_y);
    debug_assert!(plane_bsize < 22, "invalid chroma plane block");
    let tx_size = prm.tx_size;
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let (txw_unit, txh_unit) = (txw >> 2, txh >> 2);
    // Plane 4x4 unit dims (mi_size_wide/high of the SUBSAMPLED plane bsize).
    let max_blocks_wide = MI_SIZE_WIDE_B[plane_bsize];
    let max_blocks_high = MI_SIZE_HIGH_B[plane_bsize];
    // Frame-edge clip (max_block_wide/high, chroma-subsampled): a partial edge
    // block codes only its in-frame chroma tx sub-blocks (the entropy stamp in
    // encode_sb clips identically so the txb list lines up). Context arrays stay
    // full plane-block size.
    let (blocks_wide_visible, blocks_high_visible, _mbr, _mbb) = max_block_units(
        env.mi_cols,
        env.mi_rows,
        env.mi_col,
        env.mi_row,
        MI_SIZE_WIDE_B[env.bsize] as i32,
        MI_SIZE_HIGH_B[env.bsize] as i32,
        MI_SIZE_WIDE_B[plane_bsize] * 4,
        MI_SIZE_HIGH_B[plane_bsize] * 4,
        env.ss_x,
        env.ss_y,
    );
    let pi = plane - 1;

    // Always seed ta/tl from the real above/left neighbour context (see the luma
    // arm's comment): they feed both the trellis and the WRITTEN txb_skip_ctx/
    // dc_sign_ctx, and the write path derives its ctx from the real context
    // regardless of the trellis (encodetxb.c:596-598). Gating on the trellis
    // zeroed the cross-block seed for coded-lossless (trellis-off) chroma.
    let mut ta = vec![0i8; max_blocks_wide];
    let mut tl = vec![0i8; max_blocks_high];
    ta.copy_from_slice(&env.above_ctx[pi][..max_blocks_wide]);
    tl.copy_from_slice(&env.left_ctx[pi][..max_blocks_high]);
    let use_trellis = is_trellis_used(prm.enable_optimize_b, prm.dry_run_output_enabled);

    // The facade's CfL state: outside cfl_rd_pick_alpha the DC-prediction
    // cache is off (`use_dc_pred_cache == 0`, nothing cached), so every txb
    // runs the fresh-DC + alpha-AC path.
    let mut dc_cache = CflDcCache::cleared();

    let mut txbs: Vec<TxbEncode> = Vec::new();
    let mut blk_row = 0usize;
    while blk_row < blocks_high_visible {
        let mut blk_col = 0usize;
        while blk_col < blocks_wide_visible {
            // --- encode_block_intra ---
            // av1_predict_intra_block_facade: predict INTO the recon plane
            // (CfL arm applies the WINNER's signalled alphas).
            let txb_off = env.ref_off[pi] + (blk_row * env.ref_stride + blk_col) * 4;
            let mut cfl_predict;
            let cfl_arg = if winner.uv_mode == UV_CFL_PRED {
                cfl_predict = CflPredict {
                    ctx: cfl,
                    cache: &mut dc_cache,
                    alpha_idx: winner.cfl_alpha_idx,
                    joint_sign: winner.cfl_alpha_signs,
                };
                Some(&mut cfl_predict)
            } else {
                None
            };
            predict_uv_txb(
                env,
                recon,
                plane,
                winner.uv_mode,
                winner.angle_delta_uv,
                cfl_arg,
                tx_size,
                blk_row,
                blk_col,
                txb_off,
            );

            let mut tx_type = 0usize; // DCT_DCT
            let (qcoeff, dqcoeff, eob, ent_ctx, txb_skip_ctx, dc_sign_ctx);
            if prm.skip_txfm {
                // *eob = 0; p->txb_entropy_ctx[block] = 0 (encodemb.c:722-724).
                qcoeff = Vec::new();
                dqcoeff = Vec::new();
                eob = 0u16;
                ent_ctx = 0u8;
                // Dead arm in the KEY intra envelope (skip_txfm asserted 0).
                txb_skip_ctx = 0;
                dc_sign_ctx = 0;
            } else {
                // av1_subtract_txb: prediction snapshot (tight) as base.
                let mut pred = vec![0u16; txw * txh];
                for r in 0..txh {
                    pred[r * txw..r * txw + txw].copy_from_slice(
                        &recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw],
                    );
                }
                let src = if plane == 1 { env.src_u } else { env.src_v };
                let src_txb_off = env.src_off[pi] + (blk_row * env.src_stride + blk_col) * 4;
                let mut residual = vec![0i16; txw * txh];
                highbd_subtract_block(
                    txh,
                    txw,
                    &mut residual,
                    txw,
                    &src[src_txb_off..],
                    env.src_stride,
                    &pred,
                    txw,
                );

                // av1_get_tx_type PLANE_TYPE_UV intra arm.
                tx_type = uv_intra_tx_type(
                    winner.uv_mode,
                    env.lossless,
                    tx_size,
                    env.reduced_tx_set_used,
                );

                // quant_idx: use_trellis ? FP : (USE_B_QUANT_NO_TRELLIS ? B : FP).
                let kind = if use_trellis {
                    QuantKind::Fp
                } else {
                    QuantKind::B
                };
                let rows = if plane == 1 { env.rows_u } else { env.rows_v };
                let qp = QuantParams::from_plane_rows(rows, kind, env.bd, env.lossless);
                if use_trellis {
                    let bctx = BlockContext {
                        above: &ta[blk_col..],
                        left: &tl[blk_row..],
                        plane,
                        plane_bsize,
                    };
                    let opt = OptimizeInputs {
                        cost: env.coeff_costs,
                        rdmult: trellis_rdmult_intra(
                            env.rdmult,
                            prm.sharpness,
                            env.bd,
                            plane,
                            prm.use_chroma_trellis_rd_mult,
                        ),
                        sharpness: prm.sharpness,
                    };
                    let r =
                        xform_quant_optimize(&residual, tx_size, tx_type, kind, &qp, &bctx, &opt);
                    qcoeff = r.qcoeff;
                    dqcoeff = r.dqcoeff;
                    eob = r.eob;
                    ent_ctx = r.txb_entropy_ctx;
                    txb_skip_ctx = r.txb_skip_ctx;
                    dc_sign_ctx = r.dc_sign_ctx;
                } else {
                    let r = xform_quant(&residual, tx_size, tx_type, kind, &qp, false);
                    qcoeff = r.qcoeff;
                    dqcoeff = r.dqcoeff;
                    eob = r.eob;
                    ent_ctx = r.txb_entropy_ctx;
                    let (sc, dc) =
                        get_txb_ctx(plane_bsize, tx_size, plane, &ta[blk_col..], &tl[blk_row..]);
                    txb_skip_ctx = sc as usize;
                    dc_sign_ctx = dc as usize;
                }
            }

            // if (*eob) av1_inverse_transform_block into the recon plane.
            if eob > 0 {
                let mut tight = vec![0u16; txw * txh];
                for r in 0..txh {
                    tight[r * txw..r * txw + txw].copy_from_slice(
                        &recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw],
                    );
                }
                av1_inverse_transform_add(
                    &dqcoeff,
                    &mut tight,
                    txw,
                    tx_type,
                    tx_size,
                    i32::from(env.bd),
                    eob as usize,
                    env.lossless,
                );
                for r in 0..txh {
                    recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }

            // plane != 0: NO update_txk_array reset, NO cfl_store_tx.

            // --- av1_set_txb_context (full-footprint memset) ---
            for a in ta[blk_col..blk_col + txw_unit].iter_mut() {
                *a = ent_ctx as i8;
            }
            for l in tl[blk_row..blk_row + txh_unit].iter_mut() {
                *l = ent_ctx as i8;
            }

            txbs.push(TxbEncode {
                tx_type,
                eob,
                txb_entropy_ctx: ent_ctx,
                qcoeff,
                dqcoeff,
                txb_skip_ctx,
                dc_sign_ctx,
            });
            blk_col += txw_unit;
        }
        blk_row += txh_unit;
    }

    EncodeIntraPlaneOutcome { txbs, ta, tl }
}
