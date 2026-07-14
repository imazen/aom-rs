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
//! Luma (plane 0) only — 1 of 3 planes. MISSING: the chroma arms (UV
//! prediction incl. the signalled-CfL-alpha path; needed by the final
//! `encode_superblock`, not by `av1_rd_pick_intra_sbuv_mode`'s preamble which
//! only re-encodes luma); frame-edge clipped walks (`max_block_wide/high` —
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

use crate::tx_search::{
    BLK_H_B, BLK_W_B, MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TXS_H, TXS_W, TXSIZE_SQR_UP_MAP,
    trellis_rdmult_intra_y,
};
use crate::{
    BlockContext, OptimizeInputs, QuantKind, QuantParams, xform_quant, xform_quant_optimize,
};
use aom_dist::highbd_subtract_block;
use aom_entropy::partition::intra_avail;
use aom_intra::cfl::{CflCtx, cfl_store_tx};
use aom_intra::predict_intra_high;
use aom_transform::inv_txfm2d::av1_inv_txfm2d_add;
use aom_txb::CoeffCostTables;

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

    // ENTROPY_CONTEXT ta/tl = {0}; av1_get_entropy_contexts only when
    // enable_optimize_b (the enum truth test, encodemb.c:817-819).
    let mut ta = vec![0i8; max_blocks_wide];
    let mut tl = vec![0i8; max_blocks_high];
    if env.enable_optimize_b != TrellisOptType::NoTrellisOpt {
        ta.copy_from_slice(&env.above_ctx[..max_blocks_wide]);
        tl.copy_from_slice(&env.left_ctx[..max_blocks_high]);
    }
    let use_trellis = is_trellis_used(env.enable_optimize_b, env.dry_run_output_enabled);

    let mut txbs: Vec<TxbEncode> = Vec::new();
    let mut blk_row = 0usize;
    while blk_row < max_blocks_high {
        let mut blk_col = 0usize;
        while blk_col < max_blocks_wide {
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
            let (qcoeff, dqcoeff, eob, ent_ctx);
            if env.skip_txfm {
                // *eob = 0; p->txb_entropy_ctx[block] = 0 (encodemb.c:722-724).
                qcoeff = Vec::new();
                dqcoeff = Vec::new();
                eob = 0u16;
                ent_ctx = 0u8;
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
                let qp = QuantParams::from_plane_rows(env.rows, kind, env.bd);
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
                } else {
                    let r = xform_quant(&residual, tx_size, tx_type, kind, &qp, false);
                    qcoeff = r.qcoeff;
                    dqcoeff = r.dqcoeff;
                    eob = r.eob;
                    ent_ctx = r.txb_entropy_ctx;
                }
            }

            // if (*eob) av1_inverse_transform_block into the recon plane.
            if eob > 0 {
                let mut tight = pred.clone();
                av1_inv_txfm2d_add(
                    &dqcoeff,
                    &mut tight,
                    txw,
                    tx_type,
                    tx_size,
                    i32::from(env.bd),
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
            });
            blk_col += txw_unit;
        }
        blk_row += txh_unit;
    }

    EncodeIntraPlaneOutcome { txbs, ta, tl }
}
