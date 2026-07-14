//! Differential harness for `av1_rd_pick_intra_sbuv_mode` — the 14-candidate
//! chroma mode loop (uv_rd_search_mode_order: DC, CfL, H, V, SMOOTH, PAETH,
//! SMOOTH_V/H, D135/203/157/67/113/45) with the mode-signaling-rate early
//! skip, the CfL alpha search, the chroma angle-delta sweep, and the strict-<
//! first-wins best tracking — vs the C transcription over REAL pieces
//! (c_txfm_uvrd / c_cfl_rd_pick_alpha / the REAL-gate
//! ref_intra_mode_info_cost_uv). Asserts the winner tuple
//! (uv_mode, angle, cfl idx+signs, rate, rate_tokenonly, dist, skip, rd),
//! the FULL per-candidate visit log (gating + this_rd sequence — pins the
//! tie-break and every skip), the final recon planes, and the CfL AC state,
//! across 420/422/444, bd 8/10/12, q sweep, CfL-allowed and CfL-forbidden
//! (>32x32) shapes, and sub-8x8 chroma-ref blocks.

use aom_encode::intra_uv_rd::{
    chroma_plane_offset, is_chroma_reference, rd_pick_intra_sbuv_mode, UvLoopPolicy, UvRdEnv,
};
use aom_encode::mode_costs::{fill_cfl_costs, CflCosts, IntraModeCosts};
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
fn rd_pick_intra_sbuv_mode_matches_c() {
    c::ref_init();
    let mut rng = Rng(0x5b0f_00d5_0a1c_0004);
    // (bsize, ss_x, ss_y, mi_row, mi_col): CfL-legal, CfL-forbidden (>32),
    // sub-8x8 chroma-ref, all three subsamplings.
    let cases: [(usize, usize, usize, i32, i32); 10] = [
        (3, 1, 1, 8, 8),   // 8x8 @420 (CfL ok)
        (6, 1, 1, 8, 8),   // 16x16 @420
        (9, 1, 1, 8, 8),   // 32x32 @420
        (12, 1, 1, 8, 8),  // 64x64 @420 (CfL FORBIDDEN: >32)
        (0, 1, 1, 9, 9),   // 4x4 @420 sub-8x8
        (5, 1, 1, 8, 8),   // 16x8 @420
        (6, 1, 0, 8, 8),   // 16x16 @422
        (3, 1, 0, 8, 8),   // 8x8 @422
        (6, 0, 0, 8, 8),   // 16x16 @444
        (12, 0, 0, 8, 8),  // 64x64 @444 (CfL forbidden; 4-txb UV walk)
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
        (64, 64) => 4,
        (4, 8) => 5,
        (8, 4) => 6,
        (8, 16) => 7,
        (16, 8) => 8,
        (16, 32) => 9,
        (32, 16) => 10,
        _ => unreachable!(),
    };
    let mut win_counts = [0usize; 14];
    let mut angle_winners = 0usize;
    let mut cfl_winners = 0usize;
    let mut gated_visits = 0usize;

    for (ci, &(bsize, ss_x, ss_y, mi_row, mi_col)) in cases.iter().enumerate() {
        assert!(is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y));
        for iter in 0..8 {
            // Sweep BOTH usage arms (ALLINTRA chroma trellis mult 13 /
            // GOOD 20).
            let pol = if iter % 2 == 0 {
                TxTypeSearchPolicy::speed0_allintra()
            } else {
                TxTypeSearchPolicy::speed0_good()
            };
            let bd: u8 = [8, 10, 12][iter % 3];
            let maxv = (1i64 << bd) - 1;
            let qindex = [16, 64, 128, 200, 255][(ci + iter) % 5] as usize;
            let plane_bsize =
                aom_entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
            let (pw, ph) = (BLK_W_L[plane_bsize], BLK_H_L[plane_bsize]);
            let ref_off =
                chroma_plane_offset(0, STRIDE, mi_row, mi_col, bsize, ss_x, ss_y);

            // CfL context from a random luma recon (both sides). For
            // CfL-forbidden shapes the ctx is inert (mode gated) but must
            // still be threadable.
            let (bw, bh) = (BLK_W_L[bsize], BLK_H_L[bsize]);
            let luma: Vec<u16> =
                (0..STRIDE * 96).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let luma_off = 16 * STRIDE + 16;
            let mut cfl_ctx = CflCtx::new(ss_x as i32, ss_y as i32);
            let mut st_c = c::RefCflState::default();
            let cfl_allowed = bw <= 32 && bh <= 32; // is_cfl_allowed, !lossless
            if cfl_allowed {
                let luma_tx = TX_OF_DIMS(bw, bh);
                cfl_store_tx(
                    &mut cfl_ctx, &luma, luma_off, STRIDE, 0, 0, luma_tx, bsize, mi_row,
                    mi_col,
                );
                c::ref_cfl_store_tx(
                    &mut st_c, &luma, luma_off, STRIDE, 0, 0, luma_tx, bsize, mi_row, mi_col,
                    ss_x as i32, ss_y as i32, bd,
                );
            }

            // Chroma planes: content per iter — luma-correlated (CfL bait),
            // smooth gradients (DC/SMOOTH bait), directional stripes.
            let mut recon_u0: Vec<u16> =
                (0..STRIDE * 128).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let mut recon_v0: Vec<u16> =
                (0..STRIDE * 128).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let mut src_u = recon_u0.clone();
            let mut src_v = recon_v0.clone();
            let flavor = iter % 4;
            let base_u = rng.range(64, maxv as i32 - 63);
            let base_v = rng.range(64, maxv as i32 - 63);
            let amp = (maxv as i32) / 4;
            // Strong oriented content ALSO seeds the neighbour rows/cols the
            // prediction reads (1 row above / 1 col left of the block), so
            // directional predictors genuinely extrapolate it.
            let r0 = ref_off - STRIDE - 1;
            for r in 0..=ph {
                for cx in 0..=pw {
                    let (rr, cc) = (r as i32 - 1, cx as i32 - 1);
                    let (u, v) = match flavor {
                        0 => {
                            // Luma-correlated (CfL bait).
                            let l = i32::from(
                                cfl_ctx.recon_buf_q3
                                    [(rr.max(0) as usize).min(31) * 32
                                        + (cc.max(0) as usize).min(31)],
                            ) >> 3;
                            (base_u + (5 * l) / 8, base_v - (3 * l) / 8)
                        }
                        1 => {
                            // Vertical stripes (V bait).
                            let s = ((cc.rem_euclid(6)) / 3) * amp;
                            (base_u + s, base_v - s)
                        }
                        2 => {
                            // Off-axis oriented stripes (slope alternates 1:2
                            // / 2:1 per case) — lands between the base
                            // directions, so nonzero angle deltas win.
                            let t = if ci % 2 == 0 { rr + 2 * cc } else { 2 * rr + cc };
                            let s = ((t.rem_euclid(10)) / 5) * amp;
                            (base_u + s, base_v - s)
                        }
                        _ => {
                            // Horizontal ramp (H/SMOOTH_H bait) + noise.
                            (
                                base_u + rr * (amp / 8).max(2) + rng.range(-3, 4),
                                base_v + rr * (amp / 8).max(2) + rng.range(-3, 4),
                            )
                        }
                    };
                    let idx = r0 + r * STRIDE + cx;
                    src_u[idx] = u.clamp(0, maxv as i32) as u16;
                    src_v[idx] = v.clamp(0, maxv as i32) as u16;
                }
            }


            // Prediction neighbours mirror the oriented source so directional
            // modes genuinely extrapolate it: the corner + top row (extended
            // right for top-right reads, clamped to the last source column) +
            // left column (extended down for bottom-left reads).
            let top = ref_off - STRIDE - 1;
            for cx in 0..=(pw + pw.min(32)) {
                recon_u0[top + cx] = src_u[top + cx.min(pw)];
                recon_v0[top + cx] = src_v[top + cx.min(pw)];
            }
            for r in 1..=(ph + ph.min(32)) {
                let dst_i = ref_off - 1 + (r - 1) * STRIDE;
                let src_i = ref_off - 1 + (r.min(ph) - 1) * STRIDE;
                recon_u0[dst_i] = src_u[src_i];
                recon_v0[dst_i] = src_v[src_i];
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
            let tx_type_costs = TxTypeCosts::zeroed();
            let ttc_intra = vec![0i32; 3 * 4 * 13 * 16];
            let ttc_inter = vec![0i32; 4 * 4 * 16];

            // Mode costs: random uv table [13][14] + angle + palette-uv +
            // CfL costs (same CDF-derived values both sides).
            let mut uv_mode_costs = [[0i32; 14]; 13];
            for row in uv_mode_costs.iter_mut() {
                for e in row.iter_mut() {
                    *e = rng.range(0, 4 << 9);
                }
            }
            let mut costs = IntraModeCosts::zeroed();
            for row in costs.angle_delta_cost.iter_mut() {
                for e in row.iter_mut() {
                    *e = rng.range(0, 8 << 9);
                }
            }
            for row in costs.palette_uv_mode_cost.iter_mut() {
                for e in row.iter_mut() {
                    *e = rng.range(0, 4 << 9);
                }
            }
            let angle_flat: Vec<i32> =
                costs.angle_delta_cost.iter().flatten().copied().collect();
            let pal_flat: Vec<i32> =
                costs.palette_uv_mode_cost.iter().flatten().copied().collect();
            let sign_cdf = cdf_row(&mut rng, 8, 9);
            let mut alpha_cdf = Vec::new();
            for _ in 0..6 {
                alpha_cdf.extend(cdf_row(&mut rng, 16, 17));
            }
            let mut cfl_costs = CflCosts::zeroed();
            fill_cfl_costs(&mut cfl_costs, &sign_cdf, &alpha_cdf);
            let cfl_costs_c = c::ref_fill_cfl_costs(&sign_cdf, &alpha_cdf);

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
            let max_tx_size =
                aom_encode::intra_uv_rd::av1_get_tx_size_uv(bsize, false, ss_x, ss_y);

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
            let lp = UvLoopPolicy::speed0_allintra();

            let mut recon_u = recon_u0.clone();
            let mut recon_v = recon_v0.clone();
            let mut ctx_r = cfl_ctx.clone();
            let (win, visits) = rd_pick_intra_sbuv_mode(
                &env,
                &mut recon_u,
                &mut recon_v,
                &mut ctx_r,
                max_tx_size,
                cfl_allowed,
                &uv_mode_costs,
                &costs,
                &cfl_costs,
                &pol,
                &lp,
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
                use_chroma_trellis_rd_mult: pol.use_chroma_trellis_rd_mult,
            };
            let mut recon_u_c = recon_u0.clone();
            let mut recon_v_c = recon_v0.clone();
            let (cwin, cvisits) = c_rd_pick_intra_sbuv_mode(
                &cenv,
                &mut recon_u_c,
                &mut recon_v_c,
                &mut st_c,
                cfl_allowed,
                3,
                &uv_mode_costs,
                &angle_flat,
                &pal_flat,
                &cfl_costs_c,
                false,
            );

            let ctx = format!(
                "case={ci} iter={iter} bsize={bsize} ss=({ss_x},{ss_y}) bd={bd} q={qindex} y={luma_mode} flavor={flavor}",
            );
            assert_eq!(
                (
                    win.uv_mode,
                    win.angle_delta_uv,
                    win.cfl_alpha_idx,
                    win.cfl_alpha_signs,
                    win.rate,
                    win.rate_tokenonly,
                    win.dist,
                    win.skippable,
                    win.best_rd,
                ),
                cwin,
                "{ctx} winner",
            );
            let visits_t: Vec<(usize, Option<i64>)> =
                visits.iter().map(|v| (v.uv_mode, v.this_rd)).collect();
            assert_eq!(visits_t, cvisits, "{ctx} visit log");
            assert_eq!(recon_u, recon_u_c, "{ctx} recon U");
            assert_eq!(recon_v, recon_v_c, "{ctx} recon V");
            assert_eq!(&ctx_r.ac_buf_q3[..], &st_c.ac_q3[..], "{ctx} ac state");

            win_counts[win.uv_mode] += 1;
            if win.angle_delta_uv != 0 {
                angle_winners += 1;
            }
            if win.uv_mode == 13 {
                cfl_winners += 1;
            }
            gated_visits += visits_t.iter().filter(|v| v.1.is_none()).count();
        }
    }
    // Coverage: multiple distinct winners incl. CfL and nonzero angles;
    // gating arms (mode-rate skip / CfL-forbidden / invalid) exercised.
    let distinct_winners = win_counts.iter().filter(|&&n| n > 0).count();
    assert!(distinct_winners >= 4, "winner diversity too low: {win_counts:?}");
    assert!(cfl_winners > 5, "CfL winners under-exercised: {cfl_winners}");
    assert!(angle_winners > 3, "angle winners under-exercised: {angle_winners}");
    assert!(gated_visits > 40, "gated candidates under-exercised: {gated_visits}");
}
