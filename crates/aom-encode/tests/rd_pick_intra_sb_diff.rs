//! Whole-block differential for `rd_pick_intra_mode_sb`
//! (av1_rd_pick_intra_mode_sb, rdopt.c:3636-3698): the luma 61-candidate
//! search (`c_mode_loop` over REAL pieces) -> the winner tx_type_map
//! construction (REAL `update_txk_array` stamps) -> the sbuv preamble's
//! store_y gate + THE LUMA WINNER RE-ENCODE (`c_encode_intra_block_plane_y`
//! over REAL pieces, loading the REAL `cfl_store_tx` context) -> the
//! 14-candidate UV loop (`c_rd_pick_intra_sbuv_mode`) -> the non-skip
//! assembly `rate_y + rate_uv + skip_cost[ctx][0]` / `dist_y + dist_uv` /
//! RDCOST.
//!
//! Asserts the FULL winner tuple both sides: luma (mode, angle, tx_size,
//! per-txb winners, rate, rate_tokenonly, dist, rd, filter-intra), the 13x9
//! luma rd table, store_y, the re-encode per-txb outputs (tx_type, eob,
//! qcoeff, dqcoeff, entropy ctx), the post-re-encode tx_type_map (the
//! ctx->tx_type_map copy incl. eob-0 DCT resets), chroma (uv_mode, angle,
//! CfL idx+signs, rate, rate_tokenonly, dist, skip, rd) + the per-candidate
//! visit log, the final recon planes (Y after search+re-encode, U/V after
//! the uv loop), the threaded CfL state, and the assembled rate/dist/rdcost.
//! Monochrome, !is_chroma_ref (sub-8x8 at even mi), CfL-forbidden (>32x32),
//! and no-beat (tight best_rd) arms included.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::intra_rd::{Block4x4VarInfo, IntraSbyGates, TOP_INTRA_MODEL_COUNT};
use aom_encode::intra_uv_rd::{chroma_plane_offset, is_chroma_reference, UvLoopPolicy, UvRdEnv};
use aom_encode::mode_costs::{fill_cfl_costs, fill_tx_size_costs, CflCosts, TxSizeCosts};
use aom_encode::rd_pick::{
    rd_pick_intra_mode_sb, RdPickUvArgs, RdPickUvOutcome, ReencodeParams,
};
use aom_encode::tx_search::{TxTypeSearchPolicy, TxfmYrdEnv};
use aom_encode::intra_rd::IntraSbySearchCfg;
use aom_intra::cfl::{CflCtx, CFL_BUF_SQUARE};
use aom_quant::{av1_build_quantizer, set_q_index, Dequants, Quants};
use aom_sys_ref as c;
use aom_txb::{fill_tx_type_costs, CoeffCostTables, TxTypeCosts};

mod common;
use common::*;

const STRIDE: usize = 256;
const BLK_W_L: [usize; 22] =
    [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
const BLK_H_L: [usize; 22] =
    [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];

fn cdf_row(rng: &mut Rng, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut row = vec![0u16; padded];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, (32000 / nsymbs as i32).max(2)) as u32;
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

#[test]
fn rd_pick_intra_mode_sb_matches_c_composition() {
    c::ref_init();
    let mut rng = Rng(0x2d91_c405_e6b3_7a18);
    // (bsize, ss_x, ss_y, mi_row, mi_col, mono)
    #[allow(clippy::type_complexity)]
    let cases: [(usize, usize, usize, i32, i32, bool); 8] = [
        (3, 1, 1, 8, 8, false),  // 8x8 @420: store_y + uv + CfL
        (6, 1, 1, 8, 8, false),  // 16x16 @420
        (9, 0, 0, 8, 8, false),  // 32x32 @444 (still CfL-legal)
        (0, 1, 1, 9, 9, false),  // 4x4 @420 sub-8x8 CHROMA-REF (odd mi)
        (0, 1, 1, 8, 8, false),  // 4x4 @420 NOT chroma-ref: uv early return,
                                 // NO store, NO re-encode
        (4, 1, 1, 8, 8, false),  // 8x16 rect @420
        (6, 1, 0, 8, 8, false),  // 16x16 @422
        (3, 0, 0, 8, 8, true),   // 8x8 MONOCHROME: uv block never runs
    ];

    let mut searched_uv = 0usize;
    let mut not_chroma_ref = 0usize;
    let mut mono_cases = 0usize;
    let mut nobeat = 0usize;
    let mut stores = 0usize;
    let mut cfl_uv_winners = 0usize;
    let mut reencode_eob0 = 0usize;
    let mut reencode_eobpos = 0usize;

    for (ci, &(bsize, ss_x, ss_y, mi_row, mi_col, mono)) in cases.iter().enumerate() {
        let (bw, bh) = (BLK_W_L[bsize], BLK_H_L[bsize]);
        let chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y);
        let cfl_allowed =
            aom_entropy::partition::is_cfl_allowed(bsize, false, ss_x, ss_y);
        for iter in 0..6 {
            let bd: u8 = [8, 10, 12][iter % 3];
            let maxv = (1i64 << bd) - 1;
            let qindex = [16, 64, 128, 200, 255][(ci + iter) % 5] as usize;
            let reduced = iter % 4 == 3;
            let allintra = iter % 3 != 1;
            let source_variance = if iter % 3 == 0 {
                rng.range(0, 256) as u32
            } else {
                256 + rng.range(0, 4096) as u32
            };
            // no-beat arm: tight budget on the last iteration of case 1.
            let best_rd_in = if ci == 1 && iter == 5 { 1 << 10 } else { i64::MAX };
            // flat arm at max q: drives winner eob-0 -> map DCT resets in
            // the re-encode.
            let flat = ci == 0 && iter == 4;
            let above_mode =
                if rng.next().is_multiple_of(4) { None } else { Some((rng.next() % 13) as i32) };
            let left_mode =
                if rng.next().is_multiple_of(4) { None } else { Some((rng.next() % 13) as i32) };
            let qindex = if flat { 255 } else { qindex };

            // ---- luma planes ----
            let ref_off = 32 * STRIDE + 32;
            let src_off = ref_off;
            let recon_y0: Vec<u16> = if flat {
                vec![(1u16 << (bd - 1)) + 5; STRIDE * 96]
            } else {
                (0..STRIDE * 96).map(|_| (rng.next() % (1u64 << bd)) as u16).collect()
            };
            let amp: i32 = if flat { 0 } else { [160, 24, 96, 64][iter % 4] };
            let mut src_y = recon_y0.clone();
            if !flat {
                for r in 0..bh {
                    for cx in 0..bw {
                        let idx = src_off + r * STRIDE + cx;
                        let v = i64::from(recon_y0[idx]) + i64::from(rng.range(-amp, amp + 1));
                        src_y[idx] = v.clamp(0, maxv) as u16;
                    }
                }
            }

            // ---- chroma planes (skipped for mono) ----
            let ref_off_uv = chroma_plane_offset(0, STRIDE, mi_row, mi_col, bsize, ss_x, ss_y);
            let mut recon_u0: Vec<u16> =
                (0..STRIDE * 128).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let mut recon_v0: Vec<u16> =
                (0..STRIDE * 128).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let mut src_u = recon_u0.clone();
            let mut src_v = recon_v0.clone();
            if !mono {
                let plane_bsize =
                    aom_entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
                let (pw, ph) = (BLK_W_L[plane_bsize], BLK_H_L[plane_bsize]);
                let base_u = rng.range(64, maxv as i32 - 63);
                let base_v = rng.range(64, maxv as i32 - 63);
                let campl = (maxv as i32) / 4;
                let r0 = ref_off_uv - STRIDE - 1;
                for r in 0..=ph {
                    for cx in 0..=pw {
                        let (rr, cc) = (r as i32 - 1, cx as i32 - 1);
                        let (u, v) = match iter % 3 {
                            0 => {
                                let s = ((cc.rem_euclid(6)) / 3) * campl;
                                (base_u + s, base_v - s)
                            }
                            1 => (
                                base_u + rr * (campl / 8).max(2) + rng.range(-3, 4),
                                base_v + rr * (campl / 8).max(2) + rng.range(-3, 4),
                            ),
                            _ => {
                                let t = rr + 2 * cc;
                                let s = ((t.rem_euclid(10)) / 5) * campl;
                                (base_u + s, base_v - s)
                            }
                        };
                        let idx = r0 + r * STRIDE + cx;
                        src_u[idx] = u.clamp(0, maxv as i32) as u16;
                        src_v[idx] = v.clamp(0, maxv as i32) as u16;
                    }
                }
                // Neighbour seeding (top row + left col mirror the source).
                let top = ref_off_uv - STRIDE - 1;
                for cx in 0..=(pw + pw.min(32)) {
                    recon_u0[top + cx] = src_u[top + cx.min(pw)];
                    recon_v0[top + cx] = src_v[top + cx.min(pw)];
                }
                for r in 1..=(ph + ph.min(32)) {
                    let dst_i = ref_off_uv - 1 + (r - 1) * STRIDE;
                    let src_i = ref_off_uv - 1 + (r.min(ph) - 1) * STRIDE;
                    recon_u0[dst_i] = src_u[src_i];
                    recon_v0[dst_i] = src_v[src_i];
                }
            }

            // ---- quantizer rows ----
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows_y = set_q_index(&quants, &deq, qindex, 0);
            let rows_u = set_q_index(&quants, &deq, qindex, 1);
            let rows_v = set_q_index(&quants, &deq, qindex, 2);
            let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
            let plane_rows_y_c = &rows_c[0..56];
            let (rows_u_c, rows_v_c) = (&rows_c[56..112], &rows_c[112..168]);
            let dequant_y = [rows_y.dequant[0], rows_y.dequant[1]];
            let dequant_u = [rows_u_c[48], rows_u_c[49]];
            let dequant_v = [rows_v_c[48], rows_v_c[49]];

            // ---- coefficient cost tables (Y + shared UV) ----
            let y_txb_skip = tbl(&mut rng, 13 * 2);
            let y_base_eob = tbl(&mut rng, 4 * 3);
            let y_base = tbl(&mut rng, 42 * 8);
            let y_eob_extra = tbl(&mut rng, 9 * 2);
            let y_dc_sign = tbl(&mut rng, 3 * 2);
            let y_lps = tbl(&mut rng, 21 * 26);
            let y_eob = tbl(&mut rng, 2 * 11);
            let coeff_costs_y = CoeffCostTables {
                txb_skip: &y_txb_skip,
                base_eob: &y_base_eob,
                base: &y_base,
                eob_extra: &y_eob_extra,
                dc_sign: &y_dc_sign,
                lps: &y_lps,
                eob: &y_eob,
            };
            let u_txb_skip = tbl(&mut rng, 13 * 2);
            let u_base_eob = tbl(&mut rng, 4 * 3);
            let u_base = tbl(&mut rng, 42 * 8);
            let u_eob_extra = tbl(&mut rng, 9 * 2);
            let u_dc_sign = tbl(&mut rng, 3 * 2);
            let u_lps = tbl(&mut rng, 21 * 26);
            let u_eob = tbl(&mut rng, 2 * 11);
            let coeff_costs_uv = CoeffCostTables {
                txb_skip: &u_txb_skip,
                base_eob: &u_base_eob,
                base: &u_base,
                eob_extra: &u_eob_extra,
                dc_sign: &u_dc_sign,
                lps: &u_lps,
                eob: &u_eob,
            };

            // ---- tx-type costs (Y search; chroma has no tx-type bits) ----
            const NUM_EXT_TX_SET: [usize; 6] = [1, 2, 5, 7, 12, 16];
            const IDX_TO_TYPE: [[usize; 4]; 2] = [[0, 3, 2, 0], [0, 5, 4, 1]];
            let mut ttc_intra_cdf = Vec::new();
            for s in 0..3 {
                let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[0][s]].max(2);
                ttc_intra_cdf.extend_from_slice(&gen_cdfs(&mut rng, 4 * 13, ns, 17));
            }
            let mut ttc_inter_cdf = Vec::new();
            for s in 0..4 {
                let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[1][s]].max(2);
                ttc_inter_cdf.extend_from_slice(&gen_cdfs(&mut rng, 4, ns, 17));
            }
            let (c_ttc_intra, c_ttc_inter) =
                c::ref_fill_tx_type_costs(&ttc_intra_cdf, &ttc_inter_cdf);
            let mut tx_type_costs = TxTypeCosts::zeroed();
            fill_tx_type_costs(&mut tx_type_costs, &ttc_intra_cdf, &ttc_inter_cdf);
            let uv_tx_type_costs = TxTypeCosts::zeroed();
            let uv_ttc_intra = vec![0i32; 3 * 4 * 13 * 16];
            let uv_ttc_inter = vec![0i32; 4 * 4 * 16];

            // ---- tx-size / skip costs / contexts ----
            let mut ts_cdf = Vec::new();
            for cat in 0..4 {
                let ns = if cat == 0 { 2 } else { 3 };
                for _ in 0..3 {
                    ts_cdf.extend_from_slice(&cdf_row4(&mut rng, ns));
                }
            }
            let mut tx_size_costs = TxSizeCosts::zeroed();
            fill_tx_size_costs(&mut tx_size_costs, &ts_cdf);
            let ts_flat: Vec<i32> = tx_size_costs.0.iter().flatten().flatten().copied().collect();
            let skip_costs =
                [[rng.cost(), rng.cost()], [rng.cost(), rng.cost()], [rng.cost(), rng.cost()]];
            let skip_ctx = (rng.next() % 3) as usize;
            let tx_size_ctx = (rng.next() % 3) as usize;
            let above_ctx_y: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_ctx_y: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let above_u: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_u: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let above_v: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_v: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let rdmult = rng.range(1, 1 << 22);

            // ---- mode-info + CfL cost tables (dual fill) ----
            let cdf_set = gen_all_cdfs(&mut rng);
            let (mode_costs, c_mode_costs) = fill_both(&cdf_set, true);
            let angle_flat: Vec<i32> =
                mode_costs.angle_delta_cost.iter().flatten().copied().collect();
            let pal_flat: Vec<i32> =
                mode_costs.palette_uv_mode_cost.iter().flatten().copied().collect();
            let sign_cdf = cdf_row(&mut rng, 8, 9);
            let mut alpha_cdf = Vec::new();
            for _ in 0..6 {
                alpha_cdf.extend(cdf_row(&mut rng, 16, 17));
            }
            let mut cfl_costs = CflCosts::zeroed();
            fill_cfl_costs(&mut cfl_costs, &sign_cdf, &alpha_cdf);
            let cfl_costs_c = c::ref_fill_cfl_costs(&sign_cdf, &alpha_cdf);
            // uv_mode_costs [2][13][14] from the dual-filled table.
            let uv_costs_2 = &mode_costs.intra_uv_mode_cost;
            let mut c_uv_rows = [[[0i32; 14]; 13]; 2];
            for (cf, plane_rows) in c_uv_rows.iter_mut().enumerate() {
                for (m, row) in plane_rows.iter_mut().enumerate() {
                    for (u, e) in row.iter_mut().enumerate() {
                        *e = c_mode_costs.uv[(cf * 13 + m) * 14 + u];
                    }
                }
            }

            // Sweep BOTH usage arms: ALLINTRA (primary; chroma trellis
            // mult 13) and GOOD (secondary; 20) — the only speed-0 tx-layer
            // sf delta between the two usages.
            let pol = if iter % 2 == 0 {
                TxTypeSearchPolicy::speed0_allintra()
            } else {
                TxTypeSearchPolicy::speed0_good()
            };
            let gates = IntraSbyGates::speed0([false; 13]);

            // ---- Rust side ----
            let mut y_env = TxfmYrdEnv {
                sb_size: 12,
                bsize,
                mi_row,
                mi_col,
                up_available: true,
                left_available: true,
                tile_col_end: 1 << 16,
                tile_row_end: 1 << 16,
                partition: 0,
                mi_cols: 512,
                mi_rows: 512,
                ref_off,
                ref_stride: STRIDE,
                src: &src_y,
                src_off,
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                mode: 0,
                angle_delta: 0,
                use_filter_intra: false,
                filter_intra_mode: 0,
                lossless: false,
                reduced_tx_set_used: reduced,
                bd,
                rows: &rows_y,
                rdmult,
                coeff_costs: &coeff_costs_y,
                tx_type_costs: &tx_type_costs,
                skip_costs: &skip_costs,
                skip_ctx,
                tx_size_costs: &tx_size_costs,
                tx_size_ctx,
                tx_mode_is_select: true,
                above_ctx: &above_ctx_y,
                left_ctx: &left_ctx_y,
            };
            let sby_cfg = IntraSbySearchCfg {
                gates: &gates,
                top_intra_model_count_allowed: TOP_INTRA_MODEL_COUNT as i32,
                adapt_top_model_rd_count_using_neighbors: false,
                above_mode,
                left_mode,
                qindex: qindex as i32,
                mode_costs: &mode_costs,
                try_palette: false,
                palette_bsize_ctx: 0,
                palette_mode_ctx: 0,
                enable_filter_intra: true,
                allow_intrabc: false,
                pol: &pol,
                source_variance,
                enable_tx64: true,
                enable_rect_tx: true,
                allintra,
                speed: 0,
                mb_to_right_edge: 1 << 12,
                mb_to_bottom_edge: 1 << 12,
            };
            let mut uv_env = UvRdEnv {
                sb_size: 12,
                bsize,
                mi_row,
                mi_col,
                chroma_up_available: true,
                chroma_left_available: true,
                tile_col_end: 1 << 16,
                tile_row_end: 1 << 16,
                partition: 0,
                mi_cols: 512,
                mi_rows: 512,
                ss_x,
                ss_y,
                ref_off: [ref_off_uv, ref_off_uv],
                ref_stride: STRIDE,
                src_u: &src_u,
                src_v: &src_v,
                src_off: [ref_off_uv, ref_off_uv],
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                luma_mode: 0,
                luma_use_fi: false,
                luma_fi_mode: 0,
                lossless: false,
                reduced_tx_set_used: reduced,
                bd,
                rows_u: &rows_u,
                rows_v: &rows_v,
                rdmult,
                coeff_costs: &coeff_costs_uv,
                tx_type_costs: &uv_tx_type_costs,
                above_ctx: [&above_u, &above_v],
                left_ctx: [&left_u, &left_v],
            };
            let lp = UvLoopPolicy::speed0_allintra();
            let mut recon_y = recon_y0.clone();
            let mut recon_u = recon_u0.clone();
            let mut recon_v = recon_v0.clone();
            let mut cfl_rust = CflCtx::new(ss_x as i32, ss_y as i32);
            let mut var_cache = Block4x4VarInfo::sb_cache(12);
            let re = ReencodeParams {
                sharpness: 0,
                enable_optimize_b: TrellisOptType::FullTrellisOpt,
            };
            let got = {
                let uv_args = if mono {
                    None
                } else {
                    Some(RdPickUvArgs {
                        env: &mut uv_env,
                        recon_u: &mut recon_u,
                        recon_v: &mut recon_v,
                        cfl: &mut cfl_rust,
                        is_chroma_ref: chroma_ref,
                        cfl_allowed,
                        intra_uv_mode_cost: uv_costs_2,
                        costs: &mode_costs,
                        cfl_costs: &cfl_costs,
                        lp: &lp,
                    })
                };
                rd_pick_intra_mode_sb(
                    &mut y_env,
                    &mut recon_y,
                    &sby_cfg,
                    &mut var_cache,
                    best_rd_in,
                    &coeff_costs_y,
                    re,
                    uv_args,
                )
            };

            // ---- C side ----
            let mut recon_y_c = recon_y0.clone();
            let mut recon_u_c = recon_u0.clone();
            let mut recon_v_c = recon_v0.clone();
            let mut cfl_c = c::RefCflState::default();
            let n_mi = 16 * 16;
            let mut cvar = vec![-1i32; n_mi];
            let mut clog = vec![-1.0f64; n_mi];
            let skip_mask = [false; 13];
            let want_y = c_mode_loop(
                bsize,
                (mi_row, mi_col, ref_off, src_off, STRIDE),
                12,
                &mut recon_y_c,
                &src_y,
                reduced,
                bd,
                plane_rows_y_c,
                dequant_y,
                &above_ctx_y,
                &left_ctx_y,
                rdmult,
                best_rd_in,
                (&y_txb_skip, &y_base_eob, &y_base, &y_eob_extra, &y_dc_sign, &y_lps, &y_eob),
                (&c_ttc_intra, &c_ttc_inter),
                &skip_costs,
                skip_ctx,
                &ts_flat,
                tx_size_ctx,
                source_variance,
                &skip_mask,
                (above_mode, left_mode),
                qindex as i32,
                &c_mode_costs,
                allintra,
                &mut cvar,
                &mut clog,
            );

            let m = format!(
                "ci={ci} bsize={bsize} ss=({ss_x},{ss_y}) mi=({mi_row},{mi_col}) mono={mono} \
                 iter={iter} bd={bd} q={qindex} red={reduced} allintra={allintra} flat={flat} \
                 best_in={best_rd_in} chroma_ref={chroma_ref} cfl_allowed={cfl_allowed}"
            );

            // rd table always comparable.
            for mode in 0..13 {
                for d in 0..9usize {
                    assert_eq!(
                        got.intra_modes_rd_cost[mode][d], want_y.rd_table[mode][d],
                        "rd_table[{mode}][{d}] {m}",
                    );
                }
            }

            let (Some(g), Some(cy)) = (&got.best, &want_y.best) else {
                assert_eq!(got.best.is_some(), want_y.best.is_some(), "beat parity {m}");
                nobeat += 1;
                // No winner: the C function sets rate = INT_MAX and returns;
                // recon state is still the search's (compared below).
                assert_eq!(recon_y, recon_y_c, "no-beat final Y recon {m}");
                continue;
            };

            // ---- luma winner ----
            assert_eq!(g.y.mode, cy.0, "y mode {m}");
            assert_eq!(g.y.angle_delta, cy.1, "y angle {m}");
            assert_eq!(g.y.tx_size, cy.2, "y tx_size {m}");
            let wins: Vec<(usize, u16, u8)> =
                g.y.winners.iter().map(|w| (w.tx_type, w.eob, w.txb_ctx)).collect();
            assert_eq!(wins, cy.3, "y winners {m}");
            assert_eq!(g.y.rate, cy.4, "y rate {m}");
            assert_eq!(g.y.rate_tokenonly, cy.5, "y rate_tokenonly {m}");
            assert_eq!(g.y.dist, cy.6, "y dist {m}");
            assert_eq!(g.y.best_rd, cy.7, "y best_rd {m}");
            assert_eq!(g.y.use_filter_intra, cy.8, "y fi {m}");
            assert_eq!(g.y.filter_intra_mode, cy.9, "y fi mode {m}");

            // ---- the C-side composition tail ----
            // winner map: REAL update_txk_array stamps over a DCT-canonical
            // base (dead non-origin cells; see rd_pick.rs docs).
            const MI_W_B: [usize; 22] =
                [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
            const MI_H_B: [usize; 22] =
                [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];
            let (mbw, mbh) = (MI_W_B[bsize], MI_H_B[bsize]);
            let mut map_c = vec![0u8; mbw * mbh];
            {
                let (txwu, txhu) = (TX_W[cy.2] >> 2, TX_H[cy.2] >> 2);
                let mut k = 0usize;
                for blk_row in (0..mbh).step_by(txhu) {
                    for blk_col in (0..mbw).step_by(txwu) {
                        c::ref_update_txk_array(
                            &mut map_c, mbw, blk_row, blk_col, cy.2, cy.3[k].0,
                        );
                        k += 1;
                    }
                }
            }

            let store_y_c = !mono && chroma_ref && cfl_allowed;
            assert_eq!(g.store_y, store_y_c, "store_y {m}");
            let mut txbs_c = Vec::new();
            if store_y_c {
                let ca = CEncPlaneArgs {
                    partition: 0,
                    bsize,
                    tx_size: cy.2,
                    geometry: (mi_row, mi_col, ref_off, src_off, STRIDE),
                    sb_size: 12,
                    src: &src_y,
                    mode: cy.0,
                    angle_delta: cy.1,
                    use_fi: cy.8,
                    fi_mode: cy.9,
                    skip_txfm: false,
                    use_trellis: true, // FULL_TRELLIS at any dry_run
                    load_ctx: true,
                    sharpness: 0,
                    reduced,
                    bd,
                    plane_rows_c: plane_rows_y_c,
                    dequant: dequant_y,
                    above_ctx: &above_ctx_y,
                    left_ctx: &left_ctx_y,
                    rdmult,
                    coeff_tbls: (
                        &y_txb_skip, &y_base_eob, &y_base, &y_eob_extra, &y_dc_sign, &y_lps,
                        &y_eob,
                    ),
                    store: true,
                    ss: (ss_x as i32, ss_y as i32),
                };
                let (t, _ta, _tl) =
                    c_encode_intra_block_plane_y(&ca, &mut recon_y_c, &mut map_c, &mut cfl_c);
                txbs_c = t;
                stores += 1;
            }

            // Re-encode outputs.
            match (&g.reencode, store_y_c) {
                (Some(renc), true) => {
                    assert_eq!(renc.txbs.len(), txbs_c.len(), "re-encode txb count {m}");
                    for (k, (r, cc)) in renc.txbs.iter().zip(txbs_c.iter()).enumerate() {
                        assert_eq!(r.tx_type, cc.0, "re-encode txb {k} tx_type {m}");
                        assert_eq!(r.eob, cc.1, "re-encode txb {k} eob {m}");
                        assert_eq!(r.txb_entropy_ctx, cc.2, "re-encode txb {k} ctx {m}");
                        assert_eq!(r.qcoeff, cc.3, "re-encode txb {k} qcoeff {m}");
                        assert_eq!(r.dqcoeff, cc.4, "re-encode txb {k} dqcoeff {m}");
                        if r.eob == 0 {
                            reencode_eob0 += 1;
                        } else {
                            reencode_eobpos += 1;
                        }
                    }
                }
                (None, false) => {}
                (r, s) => panic!("re-encode presence mismatch {m}: rust={} c={s}", r.is_some()),
            }
            assert_eq!(g.tx_type_map, map_c, "post-re-encode tx_type_map {m}");
            assert_eq!(recon_y, recon_y_c, "final Y recon {m}");

            // ---- chroma ----
            let mut rate_uv_c = 0i32;
            let mut dist_uv_c = 0i64;
            if mono {
                assert_eq!(g.uv, RdPickUvOutcome::Monochrome, "uv mono {m}");
                mono_cases += 1;
            } else if !chroma_ref {
                assert_eq!(g.uv, RdPickUvOutcome::NotChromaRef, "uv !chroma_ref {m}");
                not_chroma_ref += 1;
            } else {
                let cuv = CUvEnv {
                    partition: 0,
                    bsize,
                    mi_row,
                    mi_col,
                    ss_x,
                    ss_y,
                    ref_off: [ref_off_uv, ref_off_uv],
                    src_off: [ref_off_uv, ref_off_uv],
                    stride: STRIDE,
                    src_u: &src_u,
                    src_v: &src_v,
                    luma_mode: cy.0,
                    luma_use_fi: cy.8,
                    luma_fi_mode: cy.9,
                    lossless: false,
                    reduced,
                    bd,
                    rows_u_c,
                    rows_v_c,
                    dequant_u,
                    dequant_v,
                    above_ctx: [&above_u, &above_v],
                    left_ctx: [&left_u, &left_v],
                    rdmult,
                    coeff_tbls: (
                        &u_txb_skip, &u_base_eob, &u_base, &u_eob_extra, &u_dc_sign, &u_lps,
                        &u_eob,
                    ),
                    ttc_tables: (&uv_ttc_intra, &uv_ttc_inter),
                    use_chroma_trellis_rd_mult: pol.use_chroma_trellis_rd_mult,
                };
                let (cw, cvisits) = c_rd_pick_intra_sbuv_mode(
                    &cuv,
                    &mut recon_u_c,
                    &mut recon_v_c,
                    &mut cfl_c,
                    cfl_allowed,
                    3, // cfl_search_range at speed 0
                    &c_uv_rows[cfl_allowed as usize],
                    &angle_flat,
                    &pal_flat,
                    &cfl_costs_c,
                    false,
                );
                let RdPickUvOutcome::Searched(w, visits) = &g.uv else {
                    panic!("uv searched expected {m}");
                };
                assert_eq!(w.uv_mode, cw.0, "uv mode {m}");
                assert_eq!(w.angle_delta_uv, cw.1, "uv angle {m}");
                assert_eq!(w.cfl_alpha_idx, cw.2, "cfl idx {m}");
                assert_eq!(w.cfl_alpha_signs, cw.3, "cfl signs {m}");
                assert_eq!(w.rate, cw.4, "uv rate {m}");
                assert_eq!(w.rate_tokenonly, cw.5, "uv rate_tokenonly {m}");
                assert_eq!(w.dist, cw.6, "uv dist {m}");
                assert_eq!(w.skippable, cw.7, "uv skip {m}");
                assert_eq!(w.best_rd, cw.8, "uv best_rd {m}");
                let vr: Vec<(usize, Option<i64>)> =
                    visits.iter().map(|v| (v.uv_mode, v.this_rd)).collect();
                assert_eq!(vr, cvisits, "uv visit log {m}");
                assert_eq!(recon_u, recon_u_c, "final U recon {m}");
                assert_eq!(recon_v, recon_v_c, "final V recon {m}");
                rate_uv_c = cw.4;
                dist_uv_c = cw.6;
                searched_uv += 1;
                if cw.0 == 13 {
                    cfl_uv_winners += 1;
                }
            }
            // CfL state (threaded store -> uv loop) equality.
            if !mono {
                assert_eq!(
                    &cfl_rust.recon_buf_q3[..],
                    &cfl_c.recon_q3[..CFL_BUF_SQUARE],
                    "CfL recon_buf_q3 {m}"
                );
                assert_eq!(
                    &cfl_rust.ac_buf_q3[..],
                    &cfl_c.ac_q3[..CFL_BUF_SQUARE],
                    "CfL ac_buf_q3 {m}"
                );
                assert_eq!(cfl_rust.buf_width, cfl_c.buf_w, "CfL buf_w {m}");
                assert_eq!(cfl_rust.buf_height, cfl_c.buf_h, "CfL buf_h {m}");
                assert_eq!(
                    cfl_rust.are_parameters_computed, cfl_c.params_computed,
                    "CfL params_computed {m}"
                );
            }

            // ---- assembly ----
            let rate_c = cy.4 + rate_uv_c + skip_costs[skip_ctx][0];
            let dist_c = cy.6 + dist_uv_c;
            let rdcost_c = c::ref_rdcost(rdmult, rate_c, dist_c);
            assert_eq!(g.rate, rate_c, "assembled rate {m}");
            assert_eq!(g.dist, dist_c, "assembled dist {m}");
            assert_eq!(g.rdcost, rdcost_c, "assembled rdcost {m}");
        }
    }

    assert!(searched_uv >= 25, "uv-searched cases: {searched_uv}");
    assert!(not_chroma_ref >= 5, "!chroma_ref cases: {not_chroma_ref}");
    assert!(mono_cases >= 5, "mono cases: {mono_cases}");
    assert!(nobeat >= 1, "no-beat cases: {nobeat}");
    assert!(stores >= 25, "store_y re-encodes: {stores}");
    assert!(reencode_eobpos >= 25, "re-encode eob>0 txbs: {reencode_eobpos}");
    assert!(reencode_eob0 >= 1, "re-encode eob-0 (map resets): {reencode_eob0}");
    assert!(cfl_uv_winners >= 1, "CfL uv winners: {cfl_uv_winners}");
}
