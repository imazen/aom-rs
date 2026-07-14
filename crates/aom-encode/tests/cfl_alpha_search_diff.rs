//! Differential harness for the CfL alpha search (`cfl_rd_pick_alpha` +
//! `cfl_pick_plane_parameter` + `cfl_pick_plane_rd` + `cfl_compute_rd` +
//! the chroma `intra_model_rd` fast model, intra_mode_search.c 586-848) vs
//! the C transcription over REAL pieces (real DCT + ref_satd fast model, the
//! REAL `av1_cfl_predict_block` + dc-pred cache, the plane-aware full-RD
//! walk). Asserts the winning `(cfl_alpha_idx, cfl_alpha_signs)`, the full
//! winner RD_STATS (rate/dist/sse/skip/rdcost), validity parity, the final
//! recon planes, and the CfL AC-state lockstep — across CfL-legal shapes,
//! 420/422/444, bd 8/10/12, q sweep, `cfl_search_range` 1/2/3 and tight
//! `ref_best_rd` budgets.

use aom_encode::intra_uv_rd::{cfl_rd_pick_alpha, UvRdEnv};
use aom_encode::mode_costs::{fill_cfl_costs, CflCosts};
use aom_encode::tx_search::TxTypeSearchPolicy;
use aom_intra::cfl::{cfl_store_tx, CflCtx};
use aom_quant::{av1_build_quantizer, set_q_index, Dequants, Quants};
use aom_sys_ref as c;
use aom_txb::{CoeffCostTables, TxTypeCosts};

mod common;
use common::*;

const STRIDE: usize = 256;

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
fn cfl_rd_pick_alpha_matches_c() {
    c::ref_init();
    let mut rng = Rng(0xcf1a_5eec_0a1c_0003);
    // CfL-legal (<= 32x32) chroma-ref shapes.
    let cases: [(usize, usize, usize, i32, i32); 8] = [
        (3, 1, 1, 8, 8),  // 8x8 @420
        (6, 1, 1, 8, 8),  // 16x16 @420
        (9, 1, 1, 8, 8),  // 32x32 @420
        (0, 1, 1, 9, 9),  // 4x4 @420 sub-8x8
        (5, 1, 1, 8, 8),  // 16x8 @420
        (6, 1, 0, 8, 8),  // 16x16 @422
        (6, 0, 0, 8, 8),  // 16x16 @444
        (3, 0, 0, 8, 8),  // 8x8 @444
    ];
    const BLK_W_L: [usize; 22] =
        [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
    const BLK_H_L: [usize; 22] =
        [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
    const TX_OF_DIMS: fn(usize, usize) -> usize = |w, h| match (w, h) {
        (4, 4) => 0,
        (8, 8) => 1,
        (16, 16) => 2,
        (32, 32) => 3,
        (4, 8) => 5,
        (8, 4) => 6,
        (8, 16) => 7,
        (16, 8) => 8,
        (16, 32) => 9,
        (32, 16) => 10,
        _ => unreachable!(),
    };
    let pol = TxTypeSearchPolicy::speed0_allintra();
    let mut valid_hits = 0usize;
    let mut invalid_hits = 0usize;
    let mut nonzero_alpha_winners = 0usize;
    let mut sign_seen = [false; 8];

    for (ci, &(bsize, ss_x, ss_y, mi_row, mi_col)) in cases.iter().enumerate() {
        for iter in 0..10 {
            let bd: u8 = [8, 10, 12][iter % 3];
            let maxv = (1i64 << bd) - 1;
            let qindex = [16, 64, 128, 200, 255][iter % 5] as usize;
            let plane_bsize =
                aom_entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
            let (pw, ph) = (BLK_W_L[plane_bsize], BLK_H_L[plane_bsize]);
            let ref_off = aom_encode::intra_uv_rd::chroma_plane_offset(
                0, STRIDE, mi_row, mi_col, bsize, ss_x, ss_y,
            );

            // Luma recon -> CfL context on both sides. Correlated chroma
            // sources (luma-scaled + noise) so nonzero alphas actually win.
            let (bw, bh) = (BLK_W_L[bsize], BLK_H_L[bsize]);
            let luma: Vec<u16> =
                (0..STRIDE * 96).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let luma_off = 16 * STRIDE + 16;
            let luma_tx = TX_OF_DIMS(bw, bh);
            let mut cfl_ctx = CflCtx::new(ss_x as i32, ss_y as i32);
            let mut st_c = c::RefCflState::default();
            cfl_store_tx(
                &mut cfl_ctx, &luma, luma_off, STRIDE, 0, 0, luma_tx, bsize, mi_row, mi_col,
            );
            c::ref_cfl_store_tx(
                &mut st_c, &luma, luma_off, STRIDE, 0, 0, luma_tx, bsize, mi_row, mi_col,
                ss_x as i32, ss_y as i32, bd,
            );

            let recon_u0: Vec<u16> =
                (0..STRIDE * 128).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let recon_v0: Vec<u16> =
                (0..STRIDE * 128).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let mut src_u = recon_u0.clone();
            let mut src_v = recon_v0.clone();
            // Chroma = alpha/8 * subsampled-luma + base + noise: puts real
            // luma correlation into the source so the hill climb moves.
            let alpha_true = [3i32, -5, 8, 0, -2][iter % 5];
            let base_u = rng.range(0, maxv as i32 + 1);
            let base_v = rng.range(0, maxv as i32 + 1);
            for r in 0..ph {
                for cx in 0..pw {
                    let l = i32::from(cfl_ctx.recon_buf_q3[r.min(31) * 32 + cx.min(31)]) >> 3;
                    let nz = rng.range(-6, 7);
                    let u = base_u + (alpha_true * l) / 8 + nz;
                    let v = base_v - (alpha_true * l) / 8 + nz;
                    src_u[ref_off + r * STRIDE + cx] = u.clamp(0, maxv as i32) as u16;
                    src_v[ref_off + r * STRIDE + cx] = v.clamp(0, maxv as i32) as u16;
                }
            }

            // Tables.
            let t_txb_skip = tbl(&mut rng, 13 * 2);
            let t_base_eob = tbl(&mut rng, 4 * 3);
            let t_base = tbl(&mut rng, 42 * 8);
            let t_eob_extra = tbl(&mut rng, 9 * 2);
            let t_dc_sign = tbl(&mut rng, 3 * 2);
            let t_lps = tbl(&mut rng, 21 * 26);
            let t_eob = tbl(&mut rng, 2 * 11);
            let coeff_costs = CoeffCostTables {
                txb_skip: &t_txb_skip,
                base_eob: &t_base_eob,
                base: &t_base,
                eob_extra: &t_eob_extra,
                dc_sign: &t_dc_sign,
                lps: &t_lps,
                eob: &t_eob,
            };
            let tx_type_costs = TxTypeCosts::zeroed(); // chroma codes no tx-type bits
            let ttc_intra = vec![0i32; 3 * 4 * 13 * 16];
            let ttc_inter = vec![0i32; 4 * 4 * 16];

            // CfL signaling costs (both sides from the same CDFs).
            let sign_cdf = cdf_row(&mut rng, 8, 9);
            let mut alpha_cdf = Vec::new();
            for _ in 0..6 {
                alpha_cdf.extend(cdf_row(&mut rng, 16, 17));
            }
            let mut cfl_costs = CflCosts::zeroed();
            fill_cfl_costs(&mut cfl_costs, &sign_cdf, &alpha_cdf);
            let cfl_costs_c = c::ref_fill_cfl_costs(&sign_cdf, &alpha_cdf);
            let uv_mode_cost = rng.range(0, 20 << 9);

            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows_u = set_q_index(&quants, &deq, qindex, 1);
            let rows_v = set_q_index(&quants, &deq, qindex, 2);
            let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
            let (rows_u_c, rows_v_c) = (&rows_c[56..112], &rows_c[112..168]);
            let dequant_u = [rows_u_c[48], rows_u_c[49]];
            let dequant_v = [rows_v_c[48], rows_v_c[49]];

            let above_u: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_u: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let above_v: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_v: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let rdmult = rng.range(1, 1 << 22);
            let luma_mode = (rng.next() % 13) as usize;
            let reduced = iter % 4 == 3;
            let cfl_search_range = match iter {
                8 => 1,
                9 => 2,
                _ => 3, // speed-0 default
            };
            // Tight budget sometimes -> the final >= ref_best_rd invalidation.
            let ref_best_rd = if iter == 7 { 1 << 12 } else { i64::MAX };

            let env = UvRdEnv {
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
                ref_off: [ref_off, ref_off],
                ref_stride: STRIDE,
                src_u: &src_u,
                src_v: &src_v,
                src_off: [ref_off, ref_off],
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                luma_mode,
                luma_use_fi: false,
                luma_fi_mode: 0,
                lossless: false,
                reduced_tx_set_used: reduced,
                bd,
                rows_u: &rows_u,
                rows_v: &rows_v,
                rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                above_ctx: [&above_u, &above_v],
                left_ctx: [&left_u, &left_v],
            };
            let uv_tx = aom_encode::intra_uv_rd::av1_get_tx_size_uv(bsize, false, ss_x, ss_y);

            let mut recon_u = recon_u0.clone();
            let mut recon_v = recon_v0.clone();
            let mut ctx_r = cfl_ctx.clone();
            let rust = cfl_rd_pick_alpha(
                &env,
                &mut recon_u,
                &mut recon_v,
                &mut ctx_r,
                uv_tx,
                ref_best_rd,
                cfl_search_range,
                &cfl_costs,
                uv_mode_cost,
                &pol,
            );

            let cenv = CUvEnv {
                bsize,
                mi_row,
                mi_col,
                ss_x,
                ss_y,
                ref_off: [ref_off, ref_off],
                src_off: [ref_off, ref_off],
                stride: STRIDE,
                src_u: &src_u,
                src_v: &src_v,
                luma_mode,
                luma_use_fi: false,
                luma_fi_mode: 0,
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
                    &t_txb_skip,
                    &t_base_eob,
                    &t_base,
                    &t_eob_extra,
                    &t_dc_sign,
                    &t_lps,
                    &t_eob,
                ),
                ttc_tables: (&ttc_intra, &ttc_inter),
            };
            let mut recon_u_c = recon_u0.clone();
            let mut recon_v_c = recon_v0.clone();
            let cres = c_cfl_rd_pick_alpha(
                &cenv,
                &mut recon_u_c,
                &mut recon_v_c,
                &mut st_c,
                uv_tx,
                ref_best_rd,
                cfl_search_range,
                &cfl_costs_c,
                uv_mode_cost,
            );

            let ctx = format!(
                "case={ci} iter={iter} bsize={bsize} ss=({ss_x},{ss_y}) bd={bd} q={qindex} range={cfl_search_range}",
            );
            match (rust, cres) {
                (None, None) => invalid_hits += 1,
                (Some(r), Some((c_idx, c_js, c_stats))) => {
                    valid_hits += 1;
                    assert_eq!(r.alpha_idx, c_idx, "{ctx} alpha_idx");
                    assert_eq!(r.joint_sign, c_js, "{ctx} joint_sign");
                    assert_eq!(
                        (r.stats.rate, r.stats.dist, r.stats.sse, r.stats.skip_txfm, r.stats.rdcost),
                        (c_stats.rate, c_stats.dist, c_stats.sse, c_stats.skip, c_stats.rdcost),
                        "{ctx} winner stats",
                    );
                    sign_seen[r.joint_sign as usize] = true;
                    if r.alpha_idx != 0 {
                        nonzero_alpha_winners += 1;
                    }
                }
                (r, c_) => {
                    panic!("{ctx} validity split: rust={:?} c={:?}", r.is_some(), c_.is_some())
                }
            }
            // Recon planes + CfL AC state in lockstep regardless of outcome.
            assert_eq!(recon_u, recon_u_c, "{ctx} recon U");
            assert_eq!(recon_v, recon_v_c, "{ctx} recon V");
            assert_eq!(&ctx_r.ac_buf_q3[..], &st_c.ac_q3[..], "{ctx} ac state");
        }
    }
    assert!(valid_hits > 40, "valid CfL picks under-exercised: {valid_hits}");
    assert!(invalid_hits > 3, "invalid arms under-exercised: {invalid_hits}");
    assert!(
        nonzero_alpha_winners > 25,
        "nonzero-alpha winners under-exercised: {nonzero_alpha_winners}",
    );
    assert!(
        sign_seen.iter().filter(|&&s| s).count() >= 4,
        "joint-sign coverage too narrow: {sign_seen:?}",
    );
}
