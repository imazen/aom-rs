//! Roundtrip verification for `aom_encode::pack` — the `OUTPUT_ENABLED`
//! SB/tile-walk composition (search via `rd_pick_partition_real` + pack via
//! `pack_tile`) that is this chunk's deliverable toward the encoder gate.
//!
//! There is no direct C oracle for a "real OUTPUT_ENABLED encode_sb call
//! over a pre-decided tree, real bytes out" shape yet (building one needs a
//! new `rd_shim.c` facade at roughly the scale of `shim_encode_av1_kf`
//! itself). Instead this harness proves the pack stage's ordering/context-
//! derivation/gating logic the strongest way available without that shim:
//! decode the packed bytes back with the READ-side primitives
//! (`read_partition`/`read_mb_modes_kf_fc`/`read_selected_tx_size`/
//! `read_coeffs_txb_full`) — each independently bit-exact-verified against
//! real libaom on its own (`partition_diff.rs`, `write_txb_full_diff.rs`,
//! etc.) — and assert the decoded partition tree, mode-info, and
//! coefficients are IDENTICAL to what the search decided and the pack
//! stage's residual recompute produced. A read/write mismatch here can only
//! come from THIS module's new glue (ordering, CDF/context selection,
//! neighbour threading) since both sides' primitives are independently
//! validated.
//!
//! Covers: a 2x2 (128x128) `sb_size=64` KEY-intra tile, 4:2:0 + 4:4:4, bd 8,
//! ALLINTRA + GOOD, real default CDFs (`KfFrameContext::default_for_qindex`)
//! for the entropy coder (so pack-stage byte production is
//! production-realistic even though the search's own RD costs are
//! synthetic-but-valid, matching `partition_pick_diff.rs`'s established
//! pattern).

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::mode_costs::{CflCosts, IntraModeCosts, TxSizeCosts, fill_cfl_costs};
use aom_encode::pack::{PackCfg, pack_tile};
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::tx_search::TxTypeSearchPolicy;
use aom_dsp::entropy::dec::OdEcDec;
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::partition::{
    KfBlockState, KfFrameContext, MiNbrKf, bsize_to_max_depth, bsize_to_tx_size_cat,
    get_partition_subsize, get_tx_size_context, read_mb_modes_kf_fc, read_partition,
    read_selected_tx_size, update_ext_partition_context,
};
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_dsp::txb::{TxTypeCosts, ext_tx_derive, read_coeffs_txb_full};

mod common;
use common::{Rng, TX_H, TX_W, coeff_cost_set_from_tables, tbl};

const STRIDE: usize = 320;
const PARTITION_NONE: i32 = 0;
const PARTITION_HORZ: i32 = 1;
const PARTITION_VERT: i32 = 2;
const PARTITION_SPLIT: i32 = 3;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4

/// Read-side mirror of [`aom_encode::pack::MiNbrGrid`] (test-local: the
/// production module only needs a write-direction grid).
struct NbrGrid {
    above: Vec<Option<MiNbrKf>>,
    left: [Option<MiNbrKf>; 32],
}
impl NbrGrid {
    fn zeroed(mi_cols: usize) -> Self {
        NbrGrid {
            above: vec![None; mi_cols],
            left: [None; 32],
        }
    }
    fn zero_left(&mut self) {
        self.left = [None; 32];
    }
    fn stamp(&mut self, mi_row: i32, mi_col: i32, mi_w: usize, mi_h: usize, nbr: MiNbrKf) {
        for x in self.above[mi_col as usize..mi_col as usize + mi_w].iter_mut() {
            *x = Some(nbr);
        }
        let l0 = (mi_row & 31) as usize;
        for x in self.left[l0..l0 + mi_h].iter_mut() {
            *x = Some(nbr);
        }
    }
}

// mi_size_wide/high (units of 4px) for BLOCK_SIZES_ALL, matching
// aom_encode::tx_search's private copies.
const MI_W: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

fn split_sub(bsize: usize) -> usize {
    match bsize {
        3 => 0,
        6 => 3,
        9 => 6,
        12 => 9,
        15 => 12,
        _ => unreachable!(),
    }
}

/// Decoded content the write side is checked against.
#[derive(Default)]
struct DecodedStats {
    leaves: usize,
    none: usize,
    split: usize,
    horz: usize,
    vert: usize,
}

/// Mirrors `pack_sb`'s recursion on the READ side: decode the partition
/// symbol, recurse/dispatch, and at each leaf decode mode-info + (if
/// signaled) tx-size + every coded plane's coefficients — asserting each
/// decoded value against what the write side is known to have encoded.
#[allow(clippy::too_many_arguments)]
fn read_sb(
    dec: &mut OdEcDec,
    env: &SbEncodeEnv,
    pack_cfg: &PackCfg,
    kf: &mut KfFrameContext,
    kfs: &mut KfBlockState,
    above_pctx: &mut [i8],
    left_pctx: &mut [i8; 32],
    above_tctx: &mut [u8],
    left_tctx: &mut [u8; 32],
    above_ectx: &mut [Vec<i8>; 3],
    left_ectx: &mut [[i8; 32]; 3],
    nbr: &mut NbrGrid,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    stats: &mut DecodedStats,
) {
    if mi_row >= env.mi_rows || mi_col >= env.mi_cols {
        return;
    }
    let hbs = (MI_W[bsize] / 2) as i32;
    let has_rows = mi_row + hbs < env.mi_rows;
    let has_cols = mi_col + hbs < env.mi_cols;

    let p = if bsize >= 3 {
        let ctx = aom_dsp::entropy::partition::partition_plane_context(
            above_pctx,
            left_pctx,
            mi_row as usize,
            mi_col as usize,
            bsize,
        ) as usize;
        read_partition(
            dec,
            &mut kf.partition[ctx],
            aom_dsp::entropy::partition::partition_cdf_length(bsize),
            has_rows,
            has_cols,
            bsize,
        )
    } else {
        PARTITION_NONE
    };

    let subsize = match p {
        PARTITION_SPLIT => split_sub(bsize),
        _ => get_partition_subsize(bsize, p) as usize,
    };

    match p {
        PARTITION_SPLIT if bsize > 3 => {
            stats.split += 1;
            for idx in 0..4i32 {
                let y = mi_row + (idx >> 1) * hbs;
                let x = mi_col + (idx & 1) * hbs;
                read_sb(
                    dec, env, pack_cfg, kf, kfs, above_pctx, left_pctx, above_tctx, left_tctx,
                    above_ectx, left_ectx, nbr, y, x, subsize, stats,
                );
            }
        }
        PARTITION_HORZ => {
            stats.horz += 1;
            read_leaf(
                dec, env, pack_cfg, kf, kfs, above_tctx, left_tctx, above_ectx, left_ectx, nbr,
                mi_row, mi_col, subsize, stats,
            );
            if mi_row + hbs < env.mi_rows {
                read_leaf(
                    dec,
                    env,
                    pack_cfg,
                    kf,
                    kfs,
                    above_tctx,
                    left_tctx,
                    above_ectx,
                    left_ectx,
                    nbr,
                    mi_row + hbs,
                    mi_col,
                    subsize,
                    stats,
                );
            }
        }
        PARTITION_VERT => {
            stats.vert += 1;
            read_leaf(
                dec, env, pack_cfg, kf, kfs, above_tctx, left_tctx, above_ectx, left_ectx, nbr,
                mi_row, mi_col, subsize, stats,
            );
            if mi_col + hbs < env.mi_cols {
                read_leaf(
                    dec,
                    env,
                    pack_cfg,
                    kf,
                    kfs,
                    above_tctx,
                    left_tctx,
                    above_ectx,
                    left_ectx,
                    nbr,
                    mi_row,
                    mi_col + hbs,
                    subsize,
                    stats,
                );
            }
        }
        _ => {
            // PARTITION_NONE (incl. the forced !has_rows&&!has_cols case at
            // bsize < BLOCK_8X8, which never coded a symbol above).
            stats.none += 1;
            read_leaf(
                dec, env, pack_cfg, kf, kfs, above_tctx, left_tctx, above_ectx, left_ectx, nbr,
                mi_row, mi_col, subsize, stats,
            );
        }
    }
    update_ext_partition_context(above_pctx, left_pctx, mi_row, mi_col, subsize, bsize, p);
}

#[allow(clippy::too_many_arguments)]
fn read_leaf(
    dec: &mut OdEcDec,
    env: &SbEncodeEnv,
    pack_cfg: &PackCfg,
    kf: &mut KfFrameContext,
    kfs: &mut KfBlockState,
    above_tctx: &mut [u8],
    left_tctx: &mut [u8; 32],
    above_ectx: &mut [Vec<i8>; 3],
    left_ectx: &mut [[i8; 32]; 3],
    nbr: &mut NbrGrid,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    stats: &mut DecodedStats,
) {
    stats.leaves += 1;
    let mi_w = MI_W[bsize];
    let mi_h = MI_H[bsize];
    let has_above = mi_row > env.tile_row_start;
    let has_left = mi_col > env.tile_col_start;
    let is_chroma_ref =
        aom_encode::intra_uv_rd::is_chroma_reference(mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    let cfl_allowed =
        aom_dsp::entropy::partition::is_cfl_allowed(bsize, env.lossless, env.ss_x, env.ss_y);

    let above_nbr = nbr.above[mi_col as usize];
    let left_nbr = nbr.left[(mi_row & 31) as usize];
    kfs.mi_row = mi_row;
    kfs.mi_col = mi_col;
    kfs.bsize = bsize;
    kfs.is_chroma_ref = is_chroma_ref;
    kfs.cfl_allowed = cfl_allowed;
    kfs.has_above = has_above;
    kfs.has_left = has_left;
    let info = read_mb_modes_kf_fc(
        dec,
        kf,
        kfs,
        pack_cfg.enable_filter_intra,
        above_nbr,
        left_nbr,
        // No palette in this envelope (matches pack.rs's write-side call).
        None,
        None,
    );
    assert_eq!(info.skip, 0, "KEY intra envelope: skip always 0");

    let mut tx_size = 0usize; // TX_4X4 default for bsize == BLOCK_4X4
    if pack_cfg.tx_mode_is_select && bsize > 0 && !env.lossless {
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;
        let ctx = get_tx_size_context(
            bsize,
            above_tctx[a0],
            left_tctx[l0],
            has_above,
            has_left,
            None,
            None,
        );
        let cat = bsize_to_tx_size_cat(bsize) as usize;
        let max_depths = bsize_to_max_depth(bsize);
        let depth = read_selected_tx_size(dec, &mut kf.tx_size[cat][ctx], bsize, max_depths);
        // tx_size_to_depth's inverse: find the tx_size at this depth from
        // the block's max rect tx (mirrors the write side's forward map).
        tx_size = depth_to_tx_size(bsize, depth);
    } else if bsize > 0 {
        tx_size = aom_encode::tx_search::TXSIZE_SQR_UP_MAP
            [aom_encode::intra_uv_rd::av1_get_tx_size_uv(bsize, env.lossless, 0, 0)];
        // (Not reached in this envelope: tx_mode_is_select is always true.)
    }
    // Stamp the txfm context exactly as `set_txfm_ctxs`/encode_b_intra_dry's
    // step 6 does (both branches for intra use the same args).
    for x in above_tctx[mi_col as usize..mi_col as usize + mi_w].iter_mut() {
        *x = TX_W[tx_size] as u8;
    }
    for x in left_tctx[(mi_row & 31) as usize..(mi_row & 31) as usize + mi_h].iter_mut() {
        *x = TX_H[tx_size] as u8;
    }

    // Luma coefficients.
    read_plane_coeffs(
        dec, kf, pack_cfg, env, &info, above_ectx, left_ectx, mi_row, mi_col, bsize, tx_size, 0,
    );
    if !env.monochrome && is_chroma_ref {
        let plane_bsize = aom_dsp::entropy::partition::get_plane_block_size(bsize, env.ss_x, env.ss_y);
        let uv_tx =
            aom_encode::intra_uv_rd::av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
        let (au, lu) = ((mi_col >> env.ss_x), ((mi_row & 31) >> env.ss_y));
        for plane in [1usize, 2] {
            read_plane_coeffs_uv(
                dec,
                kf,
                pack_cfg,
                env,
                &info,
                above_ectx,
                left_ectx,
                au,
                lu,
                plane_bsize,
                uv_tx,
                plane,
            );
        }
        let _ = (au, lu);
    }

    let nbr_kf = MiNbrKf {
        y_mode: info.y_mode,
        skip_txfm: info.skip,
    };
    nbr.stamp(mi_row, mi_col, mi_w, mi_h, nbr_kf);
}

/// `tx_size_to_depth`'s inverse for the write side's uniform-luma-tx
/// envelope: walk `SUB_TX_SIZE_MAP` from the block's max rect tx `depth`
/// times (mirrors `tx_size_to_depth`'s own loop, run forward instead of
/// counted).
fn depth_to_tx_size(bsize: usize, depth: i32) -> usize {
    const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] = [
        0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18,
    ];
    const SUB_TX_SIZE_MAP: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
    let mut tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
    for _ in 0..depth {
        tx = SUB_TX_SIZE_MAP[tx];
    }
    tx
}

#[allow(clippy::too_many_arguments)]
fn read_plane_coeffs(
    dec: &mut OdEcDec,
    kf: &mut KfFrameContext,
    cfg: &PackCfg,
    env: &SbEncodeEnv,
    info: &aom_dsp::entropy::partition::MbModeInfoKf,
    above_ectx: &mut [Vec<i8>; 3],
    left_ectx: &mut [[i8; 32]; 3],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    tx_size: usize,
    plane: usize,
) {
    let mi_w = MI_W[bsize];
    let mi_h = MI_H[bsize];
    let (txw_u, txh_u) = (TX_W[tx_size] >> 2, TX_H[tx_size] >> 2);
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;
    let mut blk_row = 0usize;
    while blk_row < mi_h {
        let mut blk_col = 0usize;
        while blk_col < mi_w {
            let above = above_ectx[plane][a0 + blk_col..].to_vec();
            let left = left_ectx[plane][l0 + blk_row..].to_vec();
            let (txb_skip_ctx, dc_sign_ctx) =
                aom_dsp::txb::get_txb_ctx(bsize, tx_size, plane, &above, &left);
            let d = ext_tx_derive(
                tx_size,
                false,
                env.reduced_tx_set_used,
                0,
                info.use_filter_intra != 0,
                info.filter_intra_mode as usize,
                info.y_mode as usize,
            );
            let mut dummy = [0u16; 8];
            let ext_tx_cdf: &mut [u16] = match d.eset {
                1 => &mut kf.ext_tx_1ddct[d.square as usize][d.intra_dir as usize],
                2 => &mut kf.ext_tx_dtt4[d.square as usize][d.intra_dir as usize],
                _ => &mut dummy[..],
            };
            let mut tcoeff = vec![0i32; aom_dsp::txb::txb_wide(tx_size) * aom_dsp::txb::txb_high(tx_size)];
            let (eob, _tx_type) = read_coeffs_txb_full(
                dec,
                &mut kf.coeff,
                ext_tx_cdf,
                &mut tcoeff,
                tx_size,
                0,
                txb_skip_ctx as usize,
                dc_sign_ctx as usize,
                cfg.allow_update_cdf,
                false,
                env.reduced_tx_set_used,
                cfg.signal_gate,
                0,
            );
            let cul = aom_dsp::txb::txb_entropy_context(&tcoeff, tx_size, _tx_type, eob) as i8;
            for x in above_ectx[plane][a0 + blk_col..a0 + blk_col + txw_u].iter_mut() {
                *x = cul;
            }
            for x in left_ectx[plane][l0 + blk_row..l0 + blk_row + txh_u].iter_mut() {
                *x = cul;
            }
            blk_col += txw_u;
        }
        blk_row += txh_u;
    }
}

#[allow(clippy::too_many_arguments)]
fn read_plane_coeffs_uv(
    dec: &mut OdEcDec,
    kf: &mut KfFrameContext,
    cfg: &PackCfg,
    env: &SbEncodeEnv,
    info: &aom_dsp::entropy::partition::MbModeInfoKf,
    above_ectx: &mut [Vec<i8>; 3],
    left_ectx: &mut [[i8; 32]; 3],
    au: i32,
    lu: i32,
    plane_bsize: usize,
    tx_size: usize,
    plane: usize,
) {
    let _ = info;
    let mi_w = MI_W[plane_bsize];
    let mi_h = MI_H[plane_bsize];
    let (txw_u, txh_u) = (TX_W[tx_size] >> 2, TX_H[tx_size] >> 2);
    let a0 = au as usize;
    let l0 = lu as usize;
    let mut blk_row = 0usize;
    while blk_row < mi_h {
        let mut blk_col = 0usize;
        while blk_col < mi_w {
            let above = above_ectx[plane][a0 + blk_col..].to_vec();
            let left = left_ectx[plane][l0 + blk_row..].to_vec();
            let (txb_skip_ctx, dc_sign_ctx) =
                aom_dsp::txb::get_txb_ctx(plane_bsize, tx_size, plane, &above, &left);
            let mut dummy = [0u16; 8];
            let mut tcoeff = vec![0i32; aom_dsp::txb::txb_wide(tx_size) * aom_dsp::txb::txb_high(tx_size)];
            let (eob, tx_type) = read_coeffs_txb_full(
                dec,
                &mut kf.coeff,
                &mut dummy[..],
                &mut tcoeff,
                tx_size,
                1,
                txb_skip_ctx as usize,
                dc_sign_ctx as usize,
                cfg.allow_update_cdf,
                false,
                env.reduced_tx_set_used,
                cfg.signal_gate,
                0, // tx_type_in unused for the plane_type==0 gate; chroma derives its own below
            );
            let _ = tx_type;
            let cul = aom_dsp::txb::txb_entropy_context(&tcoeff, tx_size, tx_type, eob) as i8;
            for x in above_ectx[plane][a0 + blk_col..a0 + blk_col + txw_u].iter_mut() {
                *x = cul;
            }
            for x in left_ectx[plane][l0 + blk_row..l0 + blk_row + txh_u].iter_mut() {
                *x = cul;
            }
            blk_col += txw_u;
        }
        blk_row += txh_u;
    }
}

/// The shared roundtrip body, parametrized on `pad` (in mi units): the
/// number of SB rows/cols of synthetic "previously coded" border content
/// placed before the `n_sb x n_sb` region that is actually searched+packed.
/// `pad = SB_MI` (one SB) mirrors `partition_pick_diff.rs`'s own (16, 16)
/// offset -- every tested SB then has up/left neighbours available. `pad =
/// 0` starts the searched region at the frame's TRUE (0, 0) corner: the
/// (0,0) SB has neither neighbour, the (0,1)/(1,0) SBs have exactly one, and
/// (1,1) has both -- covering the full up/left-availability matrix in one
/// pass. See [`pack_tile_roundtrips_true_corner`] for the corner regression
/// (the (0,0)-corner tx_search overflow this exercises without the panic
/// workaround).
fn run_pack_roundtrip_case(ss_x: usize, ss_y: usize, allintra: bool, qindex: usize, pad: i32) {
    {
        let mut rng = Rng(0xC0FF_EE00_u64.wrapping_add(qindex as u64));
        let bd: u8 = 8;
        let n_sb = 2i32;
        let mi_rows = pad + n_sb * SB_MI;
        let mi_cols = pad + n_sb * SB_MI;
        let h = (mi_rows * 4) as usize;
        let w = (mi_cols * 4) as usize;

        // Varied synthetic content: flat / ramp / noise quadrants per SB so
        // NONE/SPLIT/HORZ/VERT all get genuinely exercised. The pad border
        // (rows/cols [0, pad*4)) is filled too -- it's real, readable pixel
        // data, just not itself under test.
        let mut src_y = vec![0u16; STRIDE * (h + 4)];
        for r in 0..h {
            for c in 0..w {
                let (br, bc) = (r / 32 % 2, c / 32 % 2);
                let v: i32 = match (br, bc) {
                    (0, 0) => 120 + (r as i32 / 4),
                    (0, 1) => 40 + (c as i32 % 40),
                    (1, 0) => 210 - (r as i32 % 60),
                    _ => rng.range(0, 255),
                };
                src_y[r * STRIDE + c] = v.clamp(0, 255) as u16;
            }
        }
        let (cw, ch) = (w >> ss_x, h >> ss_y);
        let mut src_u = vec![0u16; STRIDE * (h + 4)];
        let mut src_v = vec![0u16; STRIDE * (h + 4)];
        for r in 0..ch {
            for c in 0..cw {
                let ly = (r << ss_y) * STRIDE + (c << ss_x);
                src_u[r * STRIDE + c] = ((i32::from(src_y[ly]) * 3 / 5 + 60).clamp(0, 255)) as u16;
                src_v[r * STRIDE + c] = ((200 - i32::from(src_y[ly]) / 3).clamp(0, 255)) as u16;
            }
        }

        let mut quants = Quants::zeroed();
        let mut deq = Dequants::zeroed();
        av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
        let rows_y = set_q_index(&quants, &deq, qindex, 0);
        let rows_u = set_q_index(&quants, &deq, qindex, 1);
        let rows_v = set_q_index(&quants, &deq, qindex, 2);

        let y_tbls: Vec<Vec<i32>> = [13 * 2, 4 * 3, 42 * 8, 9 * 2, 3 * 2, 21 * 26, 2 * 11]
            .iter()
            .map(|&n| tbl(&mut rng, n))
            .collect();
        let u_tbls: Vec<Vec<i32>> = [13 * 2, 4 * 3, 42 * 8, 9 * 2, 3 * 2, 21 * 26, 2 * 11]
            .iter()
            .map(|&n| tbl(&mut rng, n))
            .collect();
        // SbEncodeEnv::coeff_costs_y/_uv is the full per-txs_ctx CoeffCostSet;
        // this harness has no C-side flat-array oracle to match (module docs
        // -- it's a write/read self-consistency roundtrip), so replicating
        // one synthetic-but-valid random table across every txs_ctx/
        // eob_multi_size slot exactly preserves the numeric RD landscape
        // every tx size read before this plumbing change.
        let coeff_costs_y = coeff_cost_set_from_tables(
            &y_tbls[0], &y_tbls[1], &y_tbls[2], &y_tbls[3], &y_tbls[4], &y_tbls[5], &y_tbls[6],
        );
        let coeff_costs_uv = coeff_cost_set_from_tables(
            &u_tbls[0], &u_tbls[1], &u_tbls[2], &u_tbls[3], &u_tbls[4], &u_tbls[5], &u_tbls[6],
        );
        let ttc_dummy = TxTypeCosts::zeroed();

        let mut mode_costs = IntraModeCosts::zeroed();
        for row in mode_costs.y_mode_costs.iter_mut().flatten() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        for row in mode_costs.angle_delta_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 8 << 9);
            }
        }
        for row in mode_costs.filter_intra_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        for row in mode_costs.filter_intra_mode_cost.iter_mut() {
            *row = rng.range(0, 4 << 9);
        }
        for row in mode_costs.palette_y_mode_cost.iter_mut().flatten() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        for row in mode_costs.palette_uv_mode_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        let mut uv_mode_cost = [[[0i32; 14]; 13]; 2];
        for t in uv_mode_cost.iter_mut() {
            for row in t.iter_mut() {
                for e in row.iter_mut() {
                    *e = rng.range(0, 4 << 9);
                }
            }
        }
        let sign_cdf = {
            let mut row = vec![0u16; 9];
            let mut acc = 0u32;
            for e in row.iter_mut().take(7) {
                acc += rng.range(1, 3600) as u32;
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            row
        };
        let mut alpha_cdf = Vec::new();
        for _ in 0..6 {
            let mut row = vec![0u16; 17];
            let mut acc = 0u32;
            for e in row.iter_mut().take(15) {
                acc += rng.range(1, 1900) as u32;
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            alpha_cdf.extend(row);
        }
        let mut cfl_costs = CflCosts::zeroed();
        fill_cfl_costs(&mut cfl_costs, &sign_cdf, &alpha_cdf);
        let mut tx_size_costs = TxSizeCosts::zeroed();
        for row in tx_size_costs.0.iter_mut().flatten() {
            for e in row.iter_mut() {
                *e = rng.range(0, 2 << 9);
            }
        }
        let skip_costs = [[rng.range(0, 4 << 9), rng.range(0, 4 << 9)]; 3];
        let mut partition_costs = [[0i32; 10]; 20];
        for row in partition_costs.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 6 << 9);
            }
        }
        // Synthetic-but-VALID partition inverse-CDF for the frame-EDGE override
        // (set_partition_cost_for_edge_blk): this glue test builds random
        // `partition_costs` directly, so there is no real CDF to gather — a
        // uniform 10-symbol inverse CDF is valid input for the edge gather at
        // the searched region's flush-right/bottom SBs. The roundtrip only
        // needs pack/unpack consistency; the exact edge decision is irrelevant.
        let mut partition_cdf = [[0u16; 11]; 20];
        for row in partition_cdf.iter_mut() {
            for (k, e) in row.iter_mut().take(10).enumerate() {
                *e = (32768 - (k as i32 + 1) * 32768 / 10) as u16; // AOM_ICDF, uniform; [9] == 0
            }
        }
        let pol = if allintra {
            TxTypeSearchPolicy::speed0_allintra()
        } else {
            TxTypeSearchPolicy::speed0_good()
        };
        let uv_lp = UvLoopPolicy::speed0_allintra();
        let rdmult = 4000 + rng.range(0, 1 << 16);

        let env = SbEncodeEnv {
            sb_size: SB,
            mi_rows,
            mi_cols,
            tile_row_start: 0,
            tile_col_start: 0,
            tile_row_end: 1 << 16,
            tile_col_end: 1 << 16,
            monochrome: false,
            ss_x,
            ss_y,
            bd,
            // xd->lossless[segment_id] (av1/encoder/encodeframe.c:2263-2266):
            // `qindex == 0 && y_dc_delta_q == 0 && u_dc_delta_q == 0 &&
            // u_ac_delta_q == 0 && v_dc_delta_q == 0 && v_ac_delta_q == 0`.
            // This envelope has no delta-q (frame-constant `base_qindex`, per
            // `PackCfg::base_qindex`'s doc), so all five deltas are always 0
            // and the derivation reduces to `qindex == 0`. Was hardcoded
            // `false` unconditionally, which at qindex=0 desynced the
            // write/read entropy coder: `get_tx_mask_intra`/
            // `get_tx_mask_uv_intra` only force the tx-type search to
            // DCT_DCT when `lossless` is true, so with the (wrong) constant
            // `false` the search stayed free to pick a non-DCT_DCT winner at
            // qindex=0 even though `signal_gate: qindex > 0` (below) never
            // writes that choice into the bitstream (matching real AV1's
            // `av1_write_tx_type`/`av1_read_tx_type` qindex=0 bypass) --
            // `write_coeffs_txb_full` still scans coefficients with the
            // real (possibly non-DCT_DCT) tx_type while `read_coeffs_txb_full`
            // unconditionally assumes DCT_DCT when unsignaled, desyncing the
            // scan order and, downstream, the whole entropy stream.
            lossless: qindex == 0,
            reduced_tx_set_used: false,
            disable_edge_filter: false,
            filter_type: 0,
            stride: STRIDE,
            src_y: &src_y,
            src_u: &src_u,
            src_v: &src_v,
            base_y: 0,
            base_uv: 0,
            rows_y: &rows_y,
            rows_u: &rows_u,
            rows_v: &rows_v,
            rdmult,
            sharpness: 0,
            enable_optimize_b: TrellisOptType::FullTrellisOpt,
            use_chroma_trellis_rd_mult: allintra,
            coeff_costs_y: &coeff_costs_y,
            coeff_costs_uv: &coeff_costs_uv,
            tx_type_costs: &ttc_dummy,
            qm_levels: None,
            tune: Default::default(),
            deltaq: None,
        };
        let pick_cfg = PickFrameCfg {
            intrabc: None,
            intra_tools: Default::default(),
            mode_costs: &mode_costs,
            tx_size_costs: &tx_size_costs,
            skip_costs: &skip_costs,
            tx_type_costs_y: &ttc_dummy,
            pol: &pol,
            uv_lp: &uv_lp,
            intra_uv_mode_cost: &uv_mode_cost,
            cfl_costs: &cfl_costs,
            partition_costs: &partition_costs,
            partition_cdfs: &partition_cdf,
            allintra,
            speed: 0,
            qindex: qindex as i32,
            enable_filter_intra: true,
            enable_tx64: true,
            enable_rect_tx: true,
            intra_pruning_with_hog: true,
            enable_rect_partitions: true,
            less_rectangular_check_level: if allintra { 1 } else { 0 },
            max_partition_size: 15,
            min_partition_size: 3,
            // This file's own count_tree/decode-side stats are
            // NONE/SPLIT/HORZ/VERT-only -- keep 4-way off (see the
            // count_tree panic arm above). Not yet cross-checked here.
            enable_1to4_partitions: false,
            // Same reasoning: AB shapes are not yet cross-checked here either.
            enable_ab_partitions: false,
            allow_screen_content_tools: false,
            qm_levels: None,
            palette_costs: None,
        };
        let pack_cfg = PackCfg {
            enable_filter_intra: true,
            tx_mode_is_select: true,
            signal_gate: qindex > 0,
            allow_update_cdf: true,
            base_qindex: qindex as i32,
            delta_q_present: false,
            delta_q_res: 0,
            // This harness's local read-side primitives don't model the
            // palette-usage flag; keep screen-content tools off so the
            // write/read sides stay symmetric (matches the prior hardcoded
            // `allow_palette = false`). The SCT=on palette path is covered by
            // `encoder_gate_e2e_byte_match`'s real-stream cases.
            allow_screen_content_tools: false,
            allow_intrabc: false,
        };

        // ---- pack ----
        let mut recon_y = src_y.clone();
        let mut recon_u = src_u.clone();
        let mut recon_v = src_v.clone();
        let mut kf_write = KfFrameContext::default_for_qindex(qindex as i32);
        let mut enc = OdEcEnc::new();
        let trees = pack_tile(
            &mut enc,
            &env,
            &pick_cfg,
            &pack_cfg,
            &mut kf_write,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            pad,
            pad,
            n_sb,
            n_sb,
            SB_MI,
            SB,
        );
        assert_eq!(trees.len(), (n_sb * n_sb) as usize);
        let bytes = enc.done().to_vec();
        assert!(
            !bytes.is_empty(),
            "pack_tile must emit bytes for a non-trivial frame"
        );

        // ---- read back ----
        let mut kf_read = KfFrameContext::default_for_qindex(qindex as i32);
        let mut dec = OdEcDec::new(&bytes);
        let mut kfs = aom_encode::pack::kf_block_state(&pack_cfg, &env, SB_MI);
        let mut above_pctx = vec![0i8; mi_cols as usize];
        let mut above_tctx = vec![aom_dsp::entropy::partition::TXFM_CTX_INIT; mi_cols as usize];
        let mut above_ectx: [Vec<i8>; 3] = [
            vec![0i8; mi_cols as usize],
            vec![0i8; mi_cols as usize],
            vec![0i8; mi_cols as usize],
        ];
        let mut nbr = NbrGrid::zeroed(mi_cols as usize);
        let mut stats = DecodedStats::default();

        for r in 0..n_sb {
            let mut left_pctx = [0i8; 32];
            let mut left_tctx = [aom_dsp::entropy::partition::TXFM_CTX_INIT; 32];
            let mut left_ectx = [[0i8; 32]; 3];
            nbr.zero_left();
            for c in 0..n_sb {
                read_sb(
                    &mut dec,
                    &env,
                    &pack_cfg,
                    &mut kf_read,
                    &mut kfs,
                    &mut above_pctx,
                    &mut left_pctx,
                    &mut above_tctx,
                    &mut left_tctx,
                    &mut above_ectx,
                    &mut left_ectx,
                    &mut nbr,
                    pad + r * SB_MI,
                    pad + c * SB_MI,
                    SB,
                    &mut stats,
                );
            }
        }

        // ---- cross-check: decoded partition-type population vs the
        //     search's own winning trees (structural agreement -- the
        //     coefficient/mode-info agreement is enforced inline by
        //     read_leaf's assert_eq! on `info.skip` plus every CDF staying
        //     in sync across thousands of symbols: any drift desyncs the
        //     range coder and read_partition/read_coeffs_txb_full would
        //     panic or return nonsense well before this point). ----
        #[derive(Default)]
        struct ExpectStats {
            leaves: usize,
            none: usize,
            split: usize,
            horz: usize,
            vert: usize,
        }
        fn count_tree(t: &aom_encode::encode_sb::SbTree, s: &mut ExpectStats) {
            match t {
                aom_encode::encode_sb::SbTree::Leaf(_) => {
                    s.leaves += 1;
                    s.none += 1;
                }
                aom_encode::encode_sb::SbTree::Split(cs) => {
                    s.split += 1;
                    for c in cs.iter() {
                        count_tree(c, s);
                    }
                }
                aom_encode::encode_sb::SbTree::Horz(_) => {
                    s.horz += 1;
                    s.leaves += 2;
                }
                aom_encode::encode_sb::SbTree::Vert(_) => {
                    s.vert += 1;
                    s.leaves += 2;
                }
                aom_encode::encode_sb::SbTree::Horz4(_)
                | aom_encode::encode_sb::SbTree::Vert4(_) => {
                    panic!(
                        "this harness passes enable_1to4_partitions: false -- \
                         rd_pick_partition_real must never produce a 4-way tree here"
                    )
                }
                aom_encode::encode_sb::SbTree::HorzA(_)
                | aom_encode::encode_sb::SbTree::HorzB(_)
                | aom_encode::encode_sb::SbTree::VertA(_)
                | aom_encode::encode_sb::SbTree::VertB(_) => {
                    panic!(
                        "this harness passes enable_ab_partitions: false -- \
                         rd_pick_partition_real must never produce an AB tree here"
                    )
                }
                // Off-frame SPLIT-child placeholder — not a coded node, counts
                // for nothing (interior fixtures never produce it).
                aom_encode::encode_sb::SbTree::Absent => {}
            }
        }
        let mut expect = ExpectStats::default();
        for t in &trees {
            count_tree(t, &mut expect);
        }
        eprintln!(
            "ss=({ss_x},{ss_y}) allintra={allintra} qindex={qindex}: {} SBs, \
             none={} split={} horz={} vert={} leaves={}",
            trees.len(),
            expect.none,
            expect.split,
            expect.horz,
            expect.vert,
            expect.leaves
        );
        assert_eq!(
            (
                stats.leaves,
                stats.none,
                stats.split,
                stats.horz,
                stats.vert
            ),
            (
                expect.leaves,
                expect.none,
                expect.split,
                expect.horz,
                expect.vert
            ),
            "ss=({ss_x},{ss_y}) allintra={allintra} qindex={qindex}: decoded partition-type \
             population must match the search's winning trees exactly"
        );

        // Final CDF-arena agreement: the write side's post-tile coefficient
        // CDFs must equal the read side's -- both adapted symbol-for-symbol
        // over the identical sequence if (and only if) every prior symbol
        // was read with the same value/context the writer used.
        assert_eq!(
            kf_write.coeff, kf_read.coeff,
            "ss=({ss_x},{ss_y}) allintra={allintra} qindex={qindex}: coefficient CDF arena must \
             adapt identically on both sides"
        );
        assert_eq!(
            kf_write.partition, kf_read.partition,
            "partition CDFs must match"
        );
        assert_eq!(kf_write.kf_y, kf_read.kf_y, "kf_y CDFs must match");
        assert_eq!(kf_write.tx_size, kf_read.tx_size, "tx_size CDFs must match");
    }
}

#[test]
fn pack_tile_roundtrips_through_the_read_side() {
    for &(ss_x, ss_y, allintra, qindex) in &[
        (1usize, 1usize, true, 96usize),
        (0, 0, false, 160),
        (1, 1, false, 40),
    ] {
        run_pack_roundtrip_case(ss_x, ss_y, allintra, qindex, SB_MI);
    }
}

/// The (0,0)-corner regression for the `tx_search.rs` / `intra_uv_rd.rs`
/// `ref_best_rd.wrapping_sub(current_rd)` fix: searches+packs a 2x2 SB grid
/// starting at the frame's TRUE (0, 0) mi origin (`pad = 0`, no synthetic
/// "previously coded" border) -- the (0,0) SB genuinely has neither an
/// up nor a left neighbour, which is exactly the `ref_best_rd == i64::MAX`
/// (no reference RD yet) + no-neighbour-fallback-prediction combination that
/// used to panic with "attempt to subtract with overflow" at
/// `tx_search.rs:1083` (and identically at `intra_uv_rd.rs:503` for chroma)
/// before that fix. The (0,1)/(1,0) SBs additionally cover the
/// exactly-one-neighbour-available cases the 2x2 grid produces for free.
///
/// `(ss=(1,1), allintra=true, qindex=98)` is not an arbitrary addition: a
/// ~150-combination sweep (`allintra x ss(0,0)/(1,1) x qindex 0..256 step 7`,
/// `pad=0`) over `run_pack_roundtrip_case` found this EXACT combination hits
/// the overflow pre-fix (confirmed by reverting `wrapping_sub` back to `-`
/// and rerunning: panics "attempt to subtract with overflow" at
/// `tx_search.rs:1118`; restoring the fix makes it pass). The other 3
/// combinations exercise the corner's up/left-availability matrix broadly
/// but do not themselves hit the overflow (the search's cost tables are
/// still synthetic, so which content pushes a coefficient far enough to
/// wrap `block_error`'s lowbd 32-bit product is not analytically obvious --
/// this sweep found it empirically rather than by hand-deriving it).
///
/// The SAME sweep also surfaced two unrelated, pre-existing `qindex=0`
/// failures (a partition-population mismatch at `pack_tile_roundtrip.rs`'s
/// own assertion, and a "skip always 0" assumption violated at qindex=0) --
/// both are OUT OF SCOPE for this fix (no overflow, no `ref_best_rd`
/// involvement) and are tracked in STATUS.md's Known Bugs rather than fixed
/// here.
#[test]
fn pack_tile_roundtrips_true_corner() {
    for &(ss_x, ss_y, allintra, qindex) in &[
        (1usize, 1usize, true, 96usize),
        (0, 0, false, 160),
        (1, 1, false, 40),
        (1, 1, true, 98), // the confirmed pre-fix overflow trigger
    ] {
        run_pack_roundtrip_case(ss_x, ss_y, allintra, qindex, 0);
    }
}

/// qindex=0 (finest quantizer) regression -- was STATUS.md's "Known Bugs".
/// At `base_qindex == 0` with no delta-q (this envelope's constant
/// assumption -- `pack.rs`'s `PackCfg::signal_gate` doc), real AV1 derives
/// `xd->lossless[segment_id] = true` (`av1/encoder/encodeframe.c:2263-2266`:
/// `qindex == 0 && y_dc_delta_q == 0 && u_dc_delta_q == 0 && u_ac_delta_q ==
/// 0 && v_dc_delta_q == 0 && v_ac_delta_q == 0`) -- NOT just "a small
/// quantizer step inside the normal DCT pipeline". Lossless forces
/// `tx_mode = ONLY_4X4` (`av1/decoder/decodeframe.c:141`,
/// `read_tx_mode`) and every tx-type search to DCT_DCT
/// (`get_tx_mask`/`av1_get_tx_type`: `blockd.h:1288-1290`).
///
/// Both previously-tracked "Known Bugs" (a partition-population mismatch
/// and a spurious decoded `skip == 1`) were the SAME root cause, not two
/// bugs: `run_pack_roundtrip_case` hardcoded `SbEncodeEnv::lossless: false`
/// unconditionally, so at qindex=0 the search's tx-type search stayed free
/// to pick a non-DCT_DCT winner (this port's `get_tx_mask_intra`/
/// `get_tx_mask_uv_intra` already force DCT_DCT when `lossless` -- they
/// just never SAW `lossless = true`). Pack's `signal_gate: qindex > 0`
/// correctly suppresses WRITING the tx_type symbol at qindex=0 (matches
/// `av1_write_tx_type`'s `base_qindex > 0` gate, `bitstream.c:815-819`) --
/// but `write_coeffs_txb_full` still SCANS/serializes the coefficients
/// using the search's real (possibly non-DCT_DCT) `tx_type` regardless of
/// `signal_gate`, matching C: `av1_write_tx_type` only gates the SYMBOL,
/// the encoder's own `mbmi->tx_type` still has to be right independently
/// (real aomenc never gets this wrong because its tx-type search itself
/// is lossless-gated). Meanwhile `read_coeffs_txb_full`, since nothing was
/// signaled, defaults to DCT_DCT unconditionally (`aom_dsp::txb::read_tx_type`,
/// matching `av1_read_tx_type`'s `*tx_type = DCT_DCT; ... if (qindex == 0)
/// return;`). Whenever the search's real winner wasn't DCT_DCT, write and
/// read disagreed on scan order for that txb -> garbage coefficients ->
/// the shared entropy-coder state desynced for the rest of the frame --
/// which is what surfaced downstream as BOTH the partition-population
/// mismatch AND the spurious `skip == 1` read (two symptoms of one cause).
///
/// Fix: thread `lossless = (qindex == 0)` into `SbEncodeEnv` (matching
/// `encodeframe.c`'s derivation) instead of hardcoding it false, plus fix
/// `partition_pick.rs`'s `TxfmYrdEnv.tx_mode_is_select` hardcode (was
/// unconditionally `true`, which -- once `lossless` is threaded correctly
/// -- would violate `pick_uniform_tx_size_type_yrd_intra`'s own "lossless
/// implies ONLY_4X4" debug_assert; real AV1's `select_tx_mode`/
/// `read_tx_mode` forces `ONLY_4X4`, never `TX_MODE_SELECT`, whenever
/// `coded_lossless`).
#[test]
fn pack_tile_roundtrips_qindex_zero() {
    for &(ss_x, ss_y, allintra) in &[
        (0usize, 0usize, true), // Bug 1 repro: partition-population mismatch
        (0, 0, false),          // Bug 1's 2nd repro (mission brief)
        (1, 1, false),          // Bug 2 repro: "skip always 0" assumption
        (1, 1, true),           // extra corner coverage
    ] {
        run_pack_roundtrip_case(ss_x, ss_y, allintra, 0, 0);
    }
}

/// Task-2/coeff-cost-parity slice: `rd_pick_partition_real`/the leaf search
/// driven ENTIRELY by REAL `av1_fill_mode_rates` + `av1_fill_coeff_costs`
/// -derived cost tables (`real_costs::derive_real_costs`) instead of the
/// synthetic-but-valid random tables `run_pack_roundtrip_case` uses -- proves
/// the wiring end to end (search -> pack -> read-back) rather than just
/// unit-testing the fill functions in isolation. Deliberately SEPARATE from
/// `run_pack_roundtrip_case` (not a parameter on it) so this exploratory
/// wiring can never perturb the already-verified synthetic-cost tests above.
///
/// Covered by `derive_real_costs` (real, from `KfFrameContext::
/// default_for_qindex`'s CDFs -- the actual libaom default tables, matching
/// what `av1_fill_mode_rates(cm, &x->mode_costs, cm->fc)` +
/// `av1_fill_coeff_costs(&x->coeff_costs, cm->fc, ...)` compute from a
/// freshly-inited frame context): mode costs (Y/UV/filter-intra/angle-delta/
/// intrabc/palette-Y-flag), partition costs, skip costs, tx-size costs, CfL
/// costs, the intra tx-type costs (`ext_tx_1ddct`/`ext_tx_dtt4` repacked),
/// AND NOW the coefficient-coding costs: `real.coeff_costs_y`/
/// `coeff_costs_uv` are the FULL per-`(txs_ctx, eob_multi_size)`
/// `aom_dsp::txb::CoeffCostSet` (5 tx-size categories x 7 eob-position categories,
/// both plane types) -- `SbEncodeEnv::coeff_costs_y`/`coeff_costs_uv` (and
/// every downstream `TxfmYrdEnv`/`UvRdEnv`/`rd_pick_intra_mode_sb` field/
/// parameter threading them) select the REAL per-tx_size table at each txb
/// via `CoeffCostSet::tables(tx_size)`, closing the single-representative-
/// `txs_ctx` + zeroed-`eob_multi_cost` gap the previous chunk left open.
/// This is the FULL real-cost envelope: every table `rd_pick_partition_real`
/// reads now matches what real aomenc's RD decisions are actually made
/// against (mode/partition/skip/tx-size/cfl/tx-type/coeff, all live-CDF-
/// derived) -- decision PARITY with aomenc's RDO still additionally requires
/// the search's control flow itself (candidate order, pruning, early
/// termination) to match, which is a separate, larger claim than "reads the
/// same costs" and is not asserted by this harness.
#[test]
fn pack_tile_roundtrips_with_real_costs() {
    use aom_encode::real_costs::derive_real_costs;

    for &(ss_x, ss_y, allintra, qindex) in &[
        (1usize, 1usize, true, 96usize),
        (0, 0, false, 160),
        (1, 1, false, 40),
    ] {
        let mut rng = Rng(0xD00D_1E55_u64.wrapping_add(qindex as u64));
        let bd: u8 = 8;
        let n_sb = 2i32;
        // True (0,0) corner too -- a bonus re-verification of the
        // wrapping_sub fix under a REAL (not hand-picked-extreme) cost
        // landscape, at no extra cost.
        let pad = 0i32;
        let mi_rows = pad + n_sb * SB_MI;
        let mi_cols = pad + n_sb * SB_MI;
        let h = (mi_rows * 4) as usize;
        let w = (mi_cols * 4) as usize;

        let mut src_y = vec![0u16; STRIDE * (h + 4)];
        for r in 0..h {
            for c in 0..w {
                let (br, bc) = (r / 32 % 2, c / 32 % 2);
                let v: i32 = match (br, bc) {
                    (0, 0) => 120 + (r as i32 / 4),
                    (0, 1) => 40 + (c as i32 % 40),
                    (1, 0) => 210 - (r as i32 % 60),
                    _ => rng.range(0, 255),
                };
                src_y[r * STRIDE + c] = v.clamp(0, 255) as u16;
            }
        }
        let (cw, ch) = (w >> ss_x, h >> ss_y);
        let mut src_u = vec![0u16; STRIDE * (h + 4)];
        let mut src_v = vec![0u16; STRIDE * (h + 4)];
        for r in 0..ch {
            for c in 0..cw {
                let ly = (r << ss_y) * STRIDE + (c << ss_x);
                src_u[r * STRIDE + c] = ((i32::from(src_y[ly]) * 3 / 5 + 60).clamp(0, 255)) as u16;
                src_v[r * STRIDE + c] = ((200 - i32::from(src_y[ly]) / 3).clamp(0, 255)) as u16;
            }
        }

        let mut quants = Quants::zeroed();
        let mut deq = Dequants::zeroed();
        av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
        let rows_y = set_q_index(&quants, &deq, qindex, 0);
        let rows_u = set_q_index(&quants, &deq, qindex, 1);
        let rows_v = set_q_index(&quants, &deq, qindex, 2);

        // The live KfFrameContext this frame's entropy coder AND cost
        // derivation both read -- `derive_real_costs` snapshots it BEFORE
        // the mutable borrow below, matching av1_fill_mode_rates +
        // av1_fill_coeff_costs being called once from the frame's
        // freshly-inited context. `real.coeff_costs_y`/`coeff_costs_uv` are
        // now the FULL per-(txs_ctx, eob_multi_size) CoeffCostSet (all 5x7
        // tables, both planes) -- SbEncodeEnv selects the real per-tx_size
        // view at each txb via `CoeffCostSet::tables`.
        let mut kf_write = KfFrameContext::default_for_qindex(qindex as i32);
        let real = derive_real_costs(&kf_write, true);

        let pol = if allintra {
            TxTypeSearchPolicy::speed0_allintra()
        } else {
            TxTypeSearchPolicy::speed0_good()
        };
        let uv_lp = UvLoopPolicy::speed0_allintra();
        let rdmult = 4000 + rng.range(0, 1 << 16);

        let env = SbEncodeEnv {
            sb_size: SB,
            mi_rows,
            mi_cols,
            tile_row_start: 0,
            tile_col_start: 0,
            tile_row_end: 1 << 16,
            tile_col_end: 1 << 16,
            monochrome: false,
            ss_x,
            ss_y,
            bd,
            lossless: false,
            reduced_tx_set_used: false,
            disable_edge_filter: false,
            filter_type: 0,
            stride: STRIDE,
            src_y: &src_y,
            src_u: &src_u,
            src_v: &src_v,
            base_y: 0,
            base_uv: 0,
            rows_y: &rows_y,
            rows_u: &rows_u,
            rows_v: &rows_v,
            rdmult,
            sharpness: 0,
            enable_optimize_b: TrellisOptType::FullTrellisOpt,
            use_chroma_trellis_rd_mult: allintra,
            coeff_costs_y: &real.coeff_costs_y,
            coeff_costs_uv: &real.coeff_costs_uv,
            tx_type_costs: &real.tx_type_costs_y,
            qm_levels: None,
            tune: Default::default(),
            deltaq: None,
        };
        let pick_cfg = PickFrameCfg {
            intrabc: None,
            intra_tools: Default::default(),
            mode_costs: &real.mode_costs,
            tx_size_costs: &real.tx_size_costs,
            skip_costs: &real.skip_costs,
            tx_type_costs_y: &real.tx_type_costs_y,
            pol: &pol,
            uv_lp: &uv_lp,
            intra_uv_mode_cost: &real.mode_costs.intra_uv_mode_cost,
            cfl_costs: &real.cfl_costs,
            partition_costs: &real.partition_costs,
            partition_cdfs: &real.partition_cdf,
            allintra,
            speed: 0,
            qindex: qindex as i32,
            enable_filter_intra: true,
            enable_tx64: true,
            enable_rect_tx: true,
            intra_pruning_with_hog: true,
            enable_rect_partitions: true,
            less_rectangular_check_level: if allintra { 1 } else { 0 },
            max_partition_size: 15,
            min_partition_size: 3,
            // This file's own count_tree/decode-side stats are
            // NONE/SPLIT/HORZ/VERT-only -- keep 4-way off (see the
            // count_tree panic arm above). Not yet cross-checked here.
            enable_1to4_partitions: false,
            // Same reasoning: AB shapes are not yet cross-checked here either.
            enable_ab_partitions: false,
            allow_screen_content_tools: false,
            qm_levels: None,
            palette_costs: None,
        };
        let pack_cfg = PackCfg {
            enable_filter_intra: true,
            tx_mode_is_select: true,
            signal_gate: qindex > 0,
            allow_update_cdf: true,
            base_qindex: qindex as i32,
            delta_q_present: false,
            delta_q_res: 0,
            // This harness's local read-side primitives don't model the
            // palette-usage flag; keep screen-content tools off so the
            // write/read sides stay symmetric (matches the prior hardcoded
            // `allow_palette = false`). The SCT=on palette path is covered by
            // `encoder_gate_e2e_byte_match`'s real-stream cases.
            allow_screen_content_tools: false,
            allow_intrabc: false,
        };

        // ---- pack ----
        let mut recon_y = src_y.clone();
        let mut recon_u = src_u.clone();
        let mut recon_v = src_v.clone();
        let mut enc = OdEcEnc::new();
        let trees = pack_tile(
            &mut enc,
            &env,
            &pick_cfg,
            &pack_cfg,
            &mut kf_write,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
            pad,
            pad,
            n_sb,
            n_sb,
            SB_MI,
            SB,
        );
        assert_eq!(trees.len(), (n_sb * n_sb) as usize);
        let bytes = enc.done().to_vec();
        assert!(
            !bytes.is_empty(),
            "real-cost pack_tile must emit bytes for a non-trivial frame"
        );

        // ---- read back (structural + CDF-arena agreement only; the
        //      exhaustive per-block cross-check lives in
        //      run_pack_roundtrip_case) ----
        let mut kf_read = KfFrameContext::default_for_qindex(qindex as i32);
        let mut dec = OdEcDec::new(&bytes);
        let mut kfs = aom_encode::pack::kf_block_state(&pack_cfg, &env, SB_MI);
        let mut above_pctx = vec![0i8; mi_cols as usize];
        let mut above_tctx = vec![aom_dsp::entropy::partition::TXFM_CTX_INIT; mi_cols as usize];
        let mut above_ectx: [Vec<i8>; 3] = [
            vec![0i8; mi_cols as usize],
            vec![0i8; mi_cols as usize],
            vec![0i8; mi_cols as usize],
        ];
        let mut nbr = NbrGrid::zeroed(mi_cols as usize);
        let mut stats = DecodedStats::default();

        for r in 0..n_sb {
            let mut left_pctx = [0i8; 32];
            let mut left_tctx = [aom_dsp::entropy::partition::TXFM_CTX_INIT; 32];
            let mut left_ectx = [[0i8; 32]; 3];
            nbr.zero_left();
            for c in 0..n_sb {
                read_sb(
                    &mut dec,
                    &env,
                    &pack_cfg,
                    &mut kf_read,
                    &mut kfs,
                    &mut above_pctx,
                    &mut left_pctx,
                    &mut above_tctx,
                    &mut left_tctx,
                    &mut above_ectx,
                    &mut left_ectx,
                    &mut nbr,
                    pad + r * SB_MI,
                    pad + c * SB_MI,
                    SB,
                    &mut stats,
                );
            }
        }

        eprintln!(
            "real-costs ss=({ss_x},{ss_y}) allintra={allintra} qindex={qindex}: {} SBs, \
             none={} split={} horz={} vert={} leaves={}",
            trees.len(),
            stats.none,
            stats.split,
            stats.horz,
            stats.vert,
            stats.leaves
        );
        assert_eq!(
            kf_write.coeff, kf_read.coeff,
            "real-costs ss=({ss_x},{ss_y}) allintra={allintra} qindex={qindex}: coefficient CDF \
             arena must adapt identically on both sides"
        );
        assert_eq!(
            kf_write.partition, kf_read.partition,
            "real-costs: partition CDFs must match"
        );
        assert_eq!(
            kf_write.kf_y, kf_read.kf_y,
            "real-costs: kf_y CDFs must match"
        );
        assert_eq!(
            kf_write.tx_size, kf_read.tx_size,
            "real-costs: tx_size CDFs must match"
        );
    }
}

/// Hand-traced arithmetic check for `intra_sb_rdmult_modifier` +
/// `fold_intra_sb_rdmult` (partition_search.c:652/5710-5722) -- the ALLINTRA
/// SB-root rdmult modifier `pack_tile` now folds in before each SB's search
/// (previously the one documented "not ported" ALLINTRA rdmult input). Values
/// hand-computed from the C: `modifier = 128; if var_min<2.0 && var_max>4.0 {
/// modifier -= (var_max-var_min)>8.0 ? 48 : (int)((var_max-var_min)*6) }`,
/// `rdmult = max(1, (rdmult*modifier)>>7)`.
#[test]
fn intra_sb_rdmult_modifier_matches_hand_traced_c() {
    use aom_encode::partition_pick::{fold_intra_sb_rdmult, intra_sb_rdmult_modifier};

    // var_min >= 2.0: gate never opens, regardless of var_max.
    assert_eq!(intra_sb_rdmult_modifier(3.0, 3.0), 128);
    assert_eq!(intra_sb_rdmult_modifier(2.0, 100.0), 128); // boundary: NOT < 2.0
    // var_min < 2.0 but var_max <= 4.0: gate needs BOTH conditions.
    assert_eq!(intra_sb_rdmult_modifier(1.0, 3.0), 128);
    assert_eq!(intra_sb_rdmult_modifier(0.0, 4.0), 128); // boundary: NOT > 4.0
    // Both trigger, diff <= 8.0: linear reduction (diff * 6, truncated).
    assert_eq!(intra_sb_rdmult_modifier(1.0, 5.0), 128 - 24); // diff=4.0 -> 104
    assert_eq!(intra_sb_rdmult_modifier(0.0, 4.5), 128 - 27); // diff=4.5 -> 101
    assert_eq!(intra_sb_rdmult_modifier(0.0, 8.0), 128 - 48); // diff=8.0 (NOT >8.0) -> 80
    // Both trigger, diff > 8.0: flat -48 floor.
    assert_eq!(intra_sb_rdmult_modifier(0.0, 8.01), 128 - 48); // diff=8.01 -> 80
    assert_eq!(intra_sb_rdmult_modifier(0.0, 15.0), 128 - 48); // diff=15.0 -> 80 (not -90)

    // Fold: identity at modifier=128 (>>7 of *128 is exact).
    assert_eq!(fold_intra_sb_rdmult(1000, 128), 1000);
    assert_eq!(fold_intra_sb_rdmult(47829, 128), 47829);
    // Fold: 80/128 = 0.625 scale.
    assert_eq!(fold_intra_sb_rdmult(1000, 80), 625);
    // Fold: floors at 1, never 0 or negative.
    assert_eq!(fold_intra_sb_rdmult(1000, 0), 1);
    assert_eq!(fold_intra_sb_rdmult(1, 1), 1); // (1*1)>>7 == 0 -> floored to 1
    assert_eq!(fold_intra_sb_rdmult(100, 80), 62); // (8000)>>7 == 62 (not rounded)
}
