//! Differential harness for `encode_sb`/`encode_b` at DRY_RUN_NORMAL — the
//! winner-tree re-encode walk (partition_search.c:1581/1419 +
//! encode_superblock:395 + av1_update_intra_mb_txb_context) — vs an
//! independent transcription of the same walk over REAL C pieces:
//!
//! - leaf plane re-encodes: `common::c_encode_intra_block_plane_y` /
//!   `c_encode_intra_block_plane_uv` (REAL-piece chains, independently
//!   validated by encode_intra_plane{,_uv}_diff);
//! - `store_y` via the REAL `store_cfl_required` (`ref_store_cfl_required`);
//! - `is_chroma_ref` via the REAL `is_chroma_reference`
//!   (`ref_is_chroma_reference`);
//! - the tile-level entropy-context stamps via the REAL `av1_get_tx_type`
//!   (Y map arm after eob-0 resets / UV arm), the REAL
//!   `av1_get_txb_entropy_context` (`ref_txb_entropy_context`), and the REAL
//!   `av1_set_entropy_contexts` (`ref_set_entropy_contexts`);
//! - the txfm-context stamp via the REAL `set_txfm_ctxs`
//!   (`ref_set_txfm_ctxs`);
//! - the partition-context stamp via the REAL `update_ext_partition_context`
//!   (`ref_update_ext_partition_context`).
//!
//! Oracle style: `encode_sb` and `encode_b` are static in libaom and only
//! reachable through `av1_rd_pick_partition`/`av1_encode` (full AV1_COMP
//! state), so the walk CONTROL FLOW is transcribed and every leaf/stamp
//! primitive is the REAL C — the established composition-oracle pattern.
//!
//! Asserts, after every whole-SB walk: all 3 recon planes, the FULL tile
//! context state (per-plane above/left entropy + partition + txfm arrays),
//! per-leaf outputs (txb tuples, store_y, chroma-ref-ness, walk order), the
//! winner tx_type_maps after the eob-0 resets, and the threaded CfL state.
//! Sweeps: NONE-at-SB / full-SPLIT / random mixed trees to 4x4 depth (420
//! sub-8x8 shared-chroma + !chroma_ref leaves with always-store CfL),
//! 420+444, bd 8/12, q sweep, CfL winners with signalled alphas, and BOTH
//! usage arms of the chroma trellis table (ALLINTRA 13 / GOOD 20).

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::{LeafWinner, SbEncodeEnv, SbTree, TileCtxState, encode_sb_dry};
use aom_encode::intra_uv_rd::chroma_plane_offset;
use aom_encode::tx_search::AV1_EXT_TX_USED_FLAG;
use aom_dsp::intra::cfl::{CFL_BUF_SQUARE, CflCtx};
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_dsp::txb::{TxTypeCosts, ext_tx_set_type};

mod common;
use common::*;

const STRIDE: usize = 256;
const MI_W_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_H_B: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
const BLK_W_L: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H_L: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

fn in_set_types(tx_size: usize, reduced: bool) -> Vec<usize> {
    let set = ext_tx_set_type(tx_size, false, reduced);
    let mask = AV1_EXT_TX_USED_FLAG[set];
    (0..16).filter(|t| mask & (1 << t) != 0).collect()
}

/// Valid uniform luma tx sizes that tile each bsize (winner sizes — the
/// C's max_txsize_rect_lookup depth chain, MAX_TX_DEPTH 2).
fn tx_choices(bsize: usize) -> &'static [usize] {
    match bsize {
        0 => &[0],         // 4x4
        3 => &[1, 0],      // 8x8
        6 => &[2, 1, 0],   // 16x16
        9 => &[3, 2, 1],   // 32x32
        12 => &[4, 3, 2],  // 64x64
        1 => &[5, 0],      // 4x8: TX_4X8 -> TX_4X4
        2 => &[6, 0],      // 8x4: TX_8X4 -> TX_4X4
        4 => &[7, 1, 0],   // 8x16: TX_8X16 -> TX_8X8 -> TX_4X4
        5 => &[8, 1, 0],   // 16x8: TX_16X8 -> TX_8X8 -> TX_4X4
        7 => &[9, 2, 1],   // 16x32: TX_16X32 -> TX_16X16 -> TX_8X8
        8 => &[10, 2, 1],  // 32x16: TX_32X16 -> TX_16X16 -> TX_8X8
        10 => &[11, 3, 2], // 32x64: TX_32X64 -> TX_32X32 -> TX_16X16
        11 => &[12, 3, 2], // 64x32: TX_64X32 -> TX_32X32 -> TX_16X16
        _ => unreachable!("64x64-partitionable bsizes only"),
    }
}

fn split_subsize(bsize: usize) -> usize {
    match bsize {
        3 => 0,
        6 => 3,
        9 => 6,
        12 => 9,
        _ => unreachable!(),
    }
}

/// `get_partition_subsize(bsize, HORZ/VERT)` for the square parents.
fn rect_subsize(bsize: usize, horz: bool) -> usize {
    match (bsize, horz) {
        (3, true) => 2,    // 8x8 -> 8x4
        (3, false) => 1,   // 8x8 -> 4x8
        (6, true) => 5,    // 16x16 -> 16x8
        (6, false) => 4,   // 16x16 -> 8x16
        (9, true) => 8,    // 32x32 -> 32x16
        (9, false) => 7,   // 32x32 -> 16x32
        (12, true) => 11,  // 64x64 -> 64x32
        (12, false) => 10, // 64x64 -> 32x64
        _ => unreachable!(),
    }
}

/// Random winner for a leaf of `bsize` (in-set map origins for the picked
/// (tx, reduced); CfL only on CfL-legal luma dims).
fn gen_winner(rng: &mut Rng, bsize: usize, reduced: bool) -> LeafWinner {
    let (bw, bh) = (BLK_W_L[bsize], BLK_H_L[bsize]);
    let choices = tx_choices(bsize);
    let tx_size = choices[(rng.next() as usize) % choices.len()];
    let mode = (rng.next() % 13) as usize;
    let angle_delta_y = if (1..=8).contains(&mode) {
        rng.range(-3, 4)
    } else {
        0
    };
    let use_fi = mode == 0 && bw <= 32 && bh <= 32 && rng.next().is_multiple_of(4);
    let filter_intra_mode = if use_fi { (rng.next() % 5) as usize } else { 0 };
    let cfl_allowed = bw <= 32 && bh <= 32;
    let uv_mode = if cfl_allowed && rng.next().is_multiple_of(3) {
        13 // UV_CFL_PRED
    } else {
        (rng.next() % 13) as usize
    };
    let angle_delta_uv = if (1..=8).contains(&uv_mode) {
        rng.range(-3, 4)
    } else {
        0
    };
    let (mbw, mbh) = (MI_W_B[bsize], MI_H_B[bsize]);
    let (txwu, txhu) = (TX_W[tx_size] >> 2, TX_H[tx_size] >> 2);
    let allowed = in_set_types(tx_size, reduced);
    let mut map: Vec<u8> = (0..mbw * mbh).map(|_| (rng.next() % 16) as u8).collect();
    for r in (0..mbh).step_by(txhu) {
        for cx in (0..mbw).step_by(txwu) {
            map[r * mbw + cx] = allowed[(rng.next() as usize) % allowed.len()] as u8;
        }
    }
    LeafWinner {
        bsize,
        mode,
        angle_delta_y,
        use_filter_intra: use_fi,
        filter_intra_mode,
        tx_size,
        uv_mode,
        angle_delta_uv,
        cfl_alpha_idx: rng.range(0, 256),
        cfl_alpha_signs: rng.range(1, 8),
        // This synthetic fixture drives encode_b_intra_dry with no smooth
        // neighbours set up (env.filter_type is 0), so 0 reproduces the pre-field
        // behaviour exactly — the re-encode now reads these per-block winner
        // fields instead of env.filter_type (luma KB-6, chroma #26).
        luma_edge_filter_type: 0,
        uv_edge_filter_type: 0,
        tx_type_map: map,
        skip_txfm: false,
        use_intrabc: false,
        inter_tx_size: [0; 16],
        dv_row: 0,
        dv_col: 0,
        dv_ref_row: 0,
        dv_ref_col: 0,
        // KEY-frame intra fixture: the inter arm (INTER-ENCODE chunk 2) is
        // never taken here, so these stay at their non-inter values.
        is_inter: false,
        ref_frame0: 0,
        ref_frame1: -1,
        inter_mode: 0,
        mv_row: 0,
        mv_col: 0,
        inter_mode_context: 0,
        // Synthetic fixture: this file drives encode_b_intra_dry/pack_leaf
        // directly from a hand-rolled winner, never through
        // leaf_pick_sb_modes's real AB-reuse-relevant path -- a placeholder
        // is fine, nothing in this test's scope reads it.
        raw_rdstats: aom_encode::partition::PartRdStats::invalid(),
        palette_y: None,
        palette_uv: None,
    }
}

/// Random NONE/SPLIT/HORZ/VERT tree; `force_deep` sends the first
/// splittable branch to 4x4 (the sub-8x8 chroma-ref/CfL shapes); `rect`
/// lets 8x8+ nodes come up HORZ/VERT (winner pairs at the rect subsize).
fn gen_tree(rng: &mut Rng, bsize: usize, reduced: bool, force_deep: bool, rect: bool) -> SbTree {
    let can_split = bsize > 0;
    if rect && bsize >= 3 && !force_deep && rng.next().is_multiple_of(3) {
        let horz = rng.next().is_multiple_of(2);
        let sub = rect_subsize(bsize, horz);
        let pair = Box::new([gen_winner(rng, sub, reduced), gen_winner(rng, sub, reduced)]);
        return if horz {
            SbTree::Horz(pair)
        } else {
            SbTree::Vert(pair)
        };
    }
    let do_split = can_split && (force_deep || rng.next().is_multiple_of(3));
    if do_split {
        let sub = split_subsize(bsize);
        let kids: Vec<SbTree> = (0..4)
            .map(|i| gen_tree(rng, sub, reduced, force_deep && i == 0, rect))
            .collect();
        SbTree::Split(Box::new(<[SbTree; 4]>::try_from(kids).ok().unwrap()))
    } else {
        SbTree::Leaf(gen_winner(rng, bsize, reduced))
    }
}

#[test]
fn encode_sb_dry_run_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0xe4c0_de5b_0000_0001);
    // 64x64 SB at mi (8,8) — interior; frame large.
    let (mi_row0, mi_col0) = (8i32, 8i32);
    let sb_bsize = 12usize;

    let mut deep_420 = 0usize;
    let mut cfl_leaves = 0usize;
    let mut store_leaves = 0usize;
    let mut nonref_leaves = 0usize;
    let mut leaves_total = 0usize;
    let mut rect_leaves = 0usize;
    let mut good_arm = 0usize;

    for case in 0..14 {
        let (ss_x, ss_y) = if case % 2 == 0 {
            (1usize, 1usize)
        } else {
            (0usize, 0usize)
        };
        let bd: u8 = if case % 3 == 2 { 12 } else { 8 };
        let qindex: usize = [16, 64, 128, 200][case % 4];
        let reduced = case % 4 == 3;
        let use_chroma_tbl = case % 2 == 0; // ALLINTRA / GOOD arms
        if !use_chroma_tbl {
            good_arm += 1;
        }
        // Tree shapes: 0 = NONE at SB; 1 = full one-level SPLIT; else random
        // (force one branch to 4x4 depth every third case).
        let force_deep = case % 3 == 2;
        let mut tree = match case {
            0 => SbTree::Leaf(gen_winner(&mut rng, sb_bsize, reduced)),
            1 => {
                let kids: Vec<SbTree> = (0..4)
                    .map(|_| SbTree::Leaf(gen_winner(&mut rng, 9, reduced)))
                    .collect();
                SbTree::Split(Box::new(<[SbTree; 4]>::try_from(kids).ok().unwrap()))
            }
            // HORZ at the SB root: two 64x32 leaves (the widest rect
            // encode_b's; CfL-illegal dims).
            12 => SbTree::Horz(Box::new([
                gen_winner(&mut rng, 11, reduced),
                gen_winner(&mut rng, 11, reduced),
            ])),
            // VERT at the SB root: two 32x64 leaves.
            13 => SbTree::Vert(Box::new([
                gen_winner(&mut rng, 10, reduced),
                gen_winner(&mut rng, 10, reduced),
            ])),
            _ => {
                // Ensure the root splits so mixed shapes occur; from case 6
                // on, 8x8+ nodes may come up HORZ/VERT (rect walk arms).
                let kids: Vec<SbTree> = (0..4)
                    .map(|i| gen_tree(&mut rng, 9, reduced, force_deep && i == 0, case >= 6))
                    .collect();
                SbTree::Split(Box::new(<[SbTree; 4]>::try_from(kids).ok().unwrap()))
            }
        };
        let mut tree_c = tree.clone();

        // Planes: random recon state + correlated source.
        let maxv = (1i64 << bd) - 1;
        let recon_y0: Vec<u16> = (0..STRIDE * 128)
            .map(|_| (rng.next() % (1u64 << bd)) as u16)
            .collect();
        let recon_u0: Vec<u16> = (0..STRIDE * 128)
            .map(|_| (rng.next() % (1u64 << bd)) as u16)
            .collect();
        let recon_v0: Vec<u16> = (0..STRIDE * 128)
            .map(|_| (rng.next() % (1u64 << bd)) as u16)
            .collect();
        let mut src_y = recon_y0.clone();
        let mut src_u = recon_u0.clone();
        let mut src_v = recon_v0.clone();
        let amp = [12i32, 48, 96, 6][case % 4];
        let base_y = 0usize;
        let base_uv = 0usize;
        let y_org = base_y + (mi_row0 as usize * 4) * STRIDE + mi_col0 as usize * 4;
        for r in 0..64usize {
            for cx in 0..64usize {
                let i = y_org + r * STRIDE + cx;
                src_y[i] = (i64::from(src_y[i]) + i64::from(rng.range(-amp, amp + 1)))
                    .clamp(0, maxv) as u16;
            }
        }
        let uv_org = chroma_plane_offset(base_uv, STRIDE, mi_row0, mi_col0, sb_bsize, ss_x, ss_y);
        let (cw, ch) = (64 >> ss_x, 64 >> ss_y);
        for r in 0..ch {
            for cx in 0..cw {
                let i = uv_org + r * STRIDE + cx;
                src_u[i] = (i64::from(src_u[i]) + i64::from(rng.range(-amp, amp + 1)))
                    .clamp(0, maxv) as u16;
                src_v[i] = (i64::from(src_v[i]) + i64::from(rng.range(-amp, amp + 1)))
                    .clamp(0, maxv) as u16;
            }
        }

        // Quantizers.
        let mut quants = Quants::zeroed();
        let mut deq = Dequants::zeroed();
        av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
        let rows_y = set_q_index(&quants, &deq, qindex, 0);
        let rows_u = set_q_index(&quants, &deq, qindex, 1);
        let rows_v = set_q_index(&quants, &deq, qindex, 2);
        let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
        let plane_rows_y = &rows_c[0..56];
        let (rows_u_c, rows_v_c) = (&rows_c[56..112], &rows_c[112..168]);
        let dequant_y = [rows_y.dequant[0], rows_y.dequant[1]];
        let dequant_u = [rows_u_c[48], rows_u_c[49]];
        let dequant_v = [rows_v_c[48], rows_v_c[49]];

        // Coefficient cost tables (Y + shared UV).
        let y_txb_skip = tbl(&mut rng, 13 * 2);
        let y_base_eob = tbl(&mut rng, 4 * 3);
        let y_base = tbl(&mut rng, 42 * 8);
        let y_eob_extra = tbl(&mut rng, 9 * 2);
        let y_dc_sign = tbl(&mut rng, 3 * 2);
        let y_lps = tbl(&mut rng, 21 * 26);
        let y_eob = tbl(&mut rng, 2 * 11);
        let u_txb_skip = tbl(&mut rng, 13 * 2);
        let u_base_eob = tbl(&mut rng, 4 * 3);
        let u_base = tbl(&mut rng, 42 * 8);
        let u_eob_extra = tbl(&mut rng, 9 * 2);
        let u_dc_sign = tbl(&mut rng, 3 * 2);
        let u_lps = tbl(&mut rng, 21 * 26);
        let u_eob = tbl(&mut rng, 2 * 11);
        // SbEncodeEnv::coeff_costs_y/_uv is now the full per-txs_ctx
        // CoeffCostSet; the C oracle still takes the 7 flat arrays directly
        // (coeff_tbls_y/coeff_tbls_uv below), so replicating them across
        // every txs_ctx/eob_multi_size slot reproduces the exact same values
        // this harness compared before (see coeff_cost_set_from_tables' doc
        // comment).
        let coeff_costs_y = coeff_cost_set_from_tables(
            &y_txb_skip,
            &y_base_eob,
            &y_base,
            &y_eob_extra,
            &y_dc_sign,
            &y_lps,
            &y_eob,
        );
        let coeff_costs_uv = coeff_cost_set_from_tables(
            &u_txb_skip,
            &u_base_eob,
            &u_base,
            &u_eob_extra,
            &u_dc_sign,
            &u_lps,
            &u_eob,
        );
        let ttc = TxTypeCosts::zeroed();
        let ttc_intra = vec![0i32; 3 * 4 * 13 * 16];
        let ttc_inter = vec![0i32; 4 * 4 * 16];
        let rdmult = rng.range(1, 1 << 22);
        let sharpness = 0;

        // Random pre-walk tile contexts (identical both sides) — a mid-tile
        // state, not just zeros.
        let mut state = TileCtxState::zeroed(64);
        for p in 0..3 {
            for x in state.above_ectx[p].iter_mut() {
                *x = (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8;
            }
            for x in state.left_ectx[p].iter_mut() {
                *x = (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8;
            }
        }
        for x in state.above_pctx.iter_mut() {
            *x = [0i8, 16, 24, 28, 30, 31][(rng.next() as usize) % 6];
        }
        for x in state.left_pctx.iter_mut() {
            *x = [0i8, 16, 24, 28, 30, 31][(rng.next() as usize) % 6];
        }
        for x in state.above_tctx.iter_mut() {
            *x = [4u8, 8, 16, 32, 64][(rng.next() as usize) % 5];
        }
        for x in state.left_tctx.iter_mut() {
            *x = [4u8, 8, 16, 32, 64][(rng.next() as usize) % 5];
        }

        let env = SbEncodeEnv {
            sb_size: 12,
            mi_rows: 512,
            mi_cols: 512,
            tile_row_start: 0,
            tile_col_start: 0,
            tile_row_end: 1 << 16,
            tile_col_end: 1 << 16,
            monochrome: false,
            ss_x,
            ss_y,
            bd,
            lossless: false,
            reduced_tx_set_used: reduced,
            disable_edge_filter: false,
            filter_type: 0,
            stride: STRIDE,
            src_y: &src_y,
            src_u: &src_u,
            src_v: &src_v,
            base_y,
            base_uv,
            rows_y: &rows_y,
            rows_u: &rows_u,
            rows_v: &rows_v,
            rdmult,
            sharpness,
            enable_optimize_b: TrellisOptType::FullTrellisOpt,
            use_chroma_trellis_rd_mult: use_chroma_tbl,
            coeff_costs_y: &coeff_costs_y,
            coeff_costs_uv: &coeff_costs_uv,
            tx_type_costs: &ttc,
            qm_levels: None,
            tune: Default::default(),
            deltaq: None,
        };

        // Pre-walk snapshot (both sides start identical).
        let state0 = state.clone();

        // ---- Rust walk ----
        let mut ry = recon_y0.clone();
        let mut ru = recon_u0.clone();
        let mut rv = recon_v0.clone();
        let mut cfl_rust = CflCtx::new(ss_x as i32, ss_y as i32);
        let mut leaves = Vec::new();
        // DRY_RUN semantics (output_enabled=false): the C oracle below aliases
        // each winner's tx_type_map (xd->tx_type_map = ctx map at dry_run,
        // encodeframe_utils.c:217) so its eob-0 resets persist — the port walk
        // must persist them identically for the post-walk map compare.
        encode_sb_dry(
            &env,
            &mut state,
            &mut ry,
            &mut ru,
            &mut rv,
            &mut cfl_rust,
            &mut tree,
            mi_row0,
            mi_col0,
            sb_bsize,
            &mut leaves,
            false,
        );

        // ---- C walk (REAL pieces), seeded from the same snapshot ----
        let mut above_p = [0i8; 64];
        above_p.copy_from_slice(&state0.above_pctx);
        let mut oracle = COracle {
            ss: (ss_x, ss_y),
            monochrome: false,
            bd,
            reduced,
            sharpness,
            use_trellis: true,
            load_ctx: true,
            use_chroma_tbl,
            mi_rows: 512,
            mi_cols: 512,
            base_y,
            stride: STRIDE,
            base_uv,
            rdmult,
            src_y: &src_y,
            src_u: &src_u,
            src_v: &src_v,
            plane_rows_y,
            rows_u_c,
            rows_v_c,
            dequant_y,
            dequant_u,
            dequant_v,
            coeff_tbls_y: (
                &y_txb_skip,
                &y_base_eob,
                &y_base,
                &y_eob_extra,
                &y_dc_sign,
                &y_lps,
                &y_eob,
            ),
            coeff_tbls_uv: (
                &u_txb_skip,
                &u_base_eob,
                &u_base,
                &u_eob_extra,
                &u_dc_sign,
                &u_lps,
                &u_eob,
            ),
            ttc: (&ttc_intra, &ttc_inter),
            above_e: [
                state0.above_ectx[0].clone(),
                state0.above_ectx[1].clone(),
                state0.above_ectx[2].clone(),
            ],
            left_e: state0.left_ectx,
            above_p,
            left_p: state0.left_pctx,
            above_t: state0.above_tctx.clone(),
            left_t: state0.left_tctx,
        };
        let mut cy = recon_y0.clone();
        let mut cu = recon_u0.clone();
        let mut cv = recon_v0.clone();
        let mut cfl_c = c::RefCflState::default();
        let mut c_leaves: Vec<CLeafOut> = Vec::new();
        oracle.encode_sb(
            &mut cy,
            &mut cu,
            &mut cv,
            &mut cfl_c,
            &mut tree_c,
            mi_row0,
            mi_col0,
            sb_bsize,
            &mut c_leaves,
            false, // DRY_RUN semantics on both sides (see the port call above)
        );

        // ---- compare ----
        let tag = format!(
            "case {case} ss {ss_x}{ss_y} bd {bd} q {qindex} reduced {reduced} \
             chroma_tbl {use_chroma_tbl} force_deep {force_deep}"
        );
        assert_eq!(leaves.len(), c_leaves.len(), "leaf count: {tag}");
        let mut any_store = false;
        for (k, (r, cc)) in leaves.iter().zip(c_leaves.iter()).enumerate() {
            let ltag = format!(
                "leaf {k} @({},{}) bs {}: {tag}",
                r.mi_row, r.mi_col, r.bsize
            );
            assert_eq!(
                (r.mi_row, r.mi_col, r.bsize),
                (cc.0, cc.1, cc.2),
                "order: {ltag}"
            );
            assert_eq!(r.is_chroma_ref, cc.3, "chroma_ref: {ltag}");
            assert_eq!(r.store_y, cc.4, "store_y: {ltag}");
            assert_eq!(r.y.txbs.len(), cc.5.len(), "y txb count: {ltag}");
            for (j, (rt, ct)) in r.y.txbs.iter().zip(cc.5.iter()).enumerate() {
                assert_eq!(rt.tx_type, ct.0, "y txb {j} tx_type: {ltag}");
                assert_eq!(rt.eob, ct.1, "y txb {j} eob: {ltag}");
                assert_eq!(rt.txb_entropy_ctx, ct.2, "y txb {j} ctx: {ltag}");
                assert_eq!(rt.qcoeff, ct.3, "y txb {j} qcoeff: {ltag}");
                assert_eq!(rt.dqcoeff, ct.4, "y txb {j} dqcoeff: {ltag}");
            }
            assert_eq!(r.u.is_some(), cc.6.is_some(), "u presence: {ltag}");
            for (plane, (ro, co)) in [(&r.u, &cc.6), (&r.v, &cc.7)].into_iter().enumerate() {
                if let (Some(ro), Some(co)) = (ro.as_ref(), co.as_ref()) {
                    assert_eq!(ro.txbs.len(), co.len(), "uv{plane} txb count: {ltag}");
                    for (j, (rt, ct)) in ro.txbs.iter().zip(co.iter()).enumerate() {
                        assert_eq!(rt.tx_type, ct.0, "uv{plane} txb {j} tx_type: {ltag}");
                        assert_eq!(rt.eob, ct.1, "uv{plane} txb {j} eob: {ltag}");
                        assert_eq!(rt.txb_entropy_ctx, ct.2, "uv{plane} txb {j} ctx: {ltag}");
                        assert_eq!(rt.qcoeff, ct.3, "uv{plane} txb {j} qcoeff: {ltag}");
                    }
                }
            }
            leaves_total += 1;
            {
                const W: [usize; 22] = [
                    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16,
                    64,
                ];
                const H: [usize; 22] = [
                    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64,
                    16,
                ];
                if W[r.bsize] != H[r.bsize] {
                    rect_leaves += 1;
                }
            }
            if r.store_y {
                store_leaves += 1;
                any_store = true;
            }
            if !r.is_chroma_ref {
                nonref_leaves += 1;
                if (ss_x, ss_y) == (1, 1) {
                    deep_420 += 1;
                }
            }
            if r.is_chroma_ref && r.u.is_some() {
                // CfL winner leaves.
                if let SbTree::Leaf(_) = tree {
                    // (winner uv_mode is inside the tree; count via store_y
                    // on chroma-ref leaves instead)
                }
                if r.store_y {
                    cfl_leaves += 1;
                }
            }
        }
        // Recon planes byte-identical.
        assert_eq!(ry, cy, "recon Y: {tag}");
        assert_eq!(ru, cu, "recon U: {tag}");
        assert_eq!(rv, cv, "recon V: {tag}");
        // Winner maps after the eob-0 resets: walk both trees in lockstep.
        fn cmp_maps(a: &SbTree, b: &SbTree, tag: &str) {
            match (a, b) {
                (SbTree::Leaf(x), SbTree::Leaf(y)) => {
                    assert_eq!(x.tx_type_map, y.tx_type_map, "winner map: {tag}");
                }
                (SbTree::Split(xs), SbTree::Split(ys)) => {
                    for (x, y) in xs.iter().zip(ys.iter()) {
                        cmp_maps(x, y, tag);
                    }
                }
                (SbTree::Horz(xs), SbTree::Horz(ys)) | (SbTree::Vert(xs), SbTree::Vert(ys)) => {
                    for (x, y) in xs.iter().zip(ys.iter()) {
                        assert_eq!(x.tx_type_map, y.tx_type_map, "rect winner map: {tag}");
                    }
                }
                _ => panic!("tree shape divergence: {tag}"),
            }
        }
        cmp_maps(&tree, &tree_c, &tag);
        // Tile context state.
        assert_eq!(
            state.above_ectx[0], oracle.above_e[0],
            "above ectx Y: {tag}"
        );
        assert_eq!(
            state.above_ectx[1], oracle.above_e[1],
            "above ectx U: {tag}"
        );
        assert_eq!(
            state.above_ectx[2], oracle.above_e[2],
            "above ectx V: {tag}"
        );
        assert_eq!(state.left_ectx, oracle.left_e, "left ectx: {tag}");
        assert_eq!(
            &state.above_pctx[..],
            &oracle.above_p[..],
            "above pctx: {tag}"
        );
        assert_eq!(state.left_pctx, oracle.left_p, "left pctx: {tag}");
        assert_eq!(state.above_tctx, oracle.above_t, "above tctx: {tag}");
        assert_eq!(state.left_tctx, oracle.left_t, "left tctx: {tag}");
        // CfL state (whenever any leaf stored).
        if any_store {
            assert_eq!(
                &cfl_rust.recon_buf_q3[..],
                &cfl_c.recon_q3[..CFL_BUF_SQUARE],
                "CfL recon_buf_q3: {tag}"
            );
            assert_eq!(cfl_rust.buf_width, cfl_c.buf_w, "CfL buf_width: {tag}");
            assert_eq!(cfl_rust.buf_height, cfl_c.buf_h, "CfL buf_height: {tag}");
        }
    }

    // Coverage floors.
    assert!(leaves_total >= 60, "leaves exercised: {leaves_total}");
    assert!(
        store_leaves >= 15,
        "store_y leaves exercised: {store_leaves}"
    );
    assert!(
        cfl_leaves >= 4,
        "chroma-ref CfL-storing leaves: {cfl_leaves}"
    );
    assert!(
        nonref_leaves >= 6,
        "!chroma_ref leaves exercised: {nonref_leaves}"
    );
    assert!(deep_420 >= 6, "420 sub-8x8 leaves exercised: {deep_420}");
    assert!(good_arm >= 6, "GOOD trellis-table cases: {good_arm}");
    assert!(
        rect_leaves >= 8,
        "HORZ/VERT rect leaves exercised: {rect_leaves}"
    );
}
