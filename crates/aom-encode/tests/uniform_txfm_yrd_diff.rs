//! Differential harness for `uniform_txfm_yrd_intra` / `txfm_rd_in_plane_intra`
//! (tx_search.c `uniform_txfm_yrd` + `av1_txfm_rd_in_plane` + `block_rd_txfm`,
//! luma intra, speed-0 policy, interior blocks) vs the same walk over REAL C
//! pieces: per txb ref_intra_avail + ref_hbd_predict_intra (into the C-side
//! recon plane) -> ref_highbd_subtract_block -> the search_tx_type C chain
//! (ref_get_tx_mask_intra / ref_pixel_diff_dist / ref_quant_plane_rows /
//! ref_optimize_txb / ref_cost_coeffs_txb + ref_get_tx_type_cost /
//! ref_dist_block_tx_domain / ref_inv_txfm2d_add + ref_hbd_variance /
//! ref_rdcost) -> winner reconstruction into the C recon plane -> entropy-ctx
//! stamp -> current_rd accumulation + exit_early, then the intra skip/tx-size
//! rate assembly. Both sides start from IDENTICAL recon planes; the final
//! planes are compared (verifying the recon feedback the next txb predicts
//! from), plus (rd, rate, dist, sse, skip).

use aom_encode::mode_costs::{fill_tx_size_costs, tx_size_cost, TxSizeCosts};
use aom_encode::tx_search::{
    trellis_rdmult_intra_y, uniform_txfm_yrd_intra, TxTypeSearchPolicy, TxfmYrdEnv,
    TX_SIZE_2D_TBL,
};
use aom_quant::{av1_build_quantizer, set_q_index, Dequants, Quants};
use aom_sys_ref as c;
use aom_txb::{fill_tx_type_costs, scan, txb_high, txb_wide, CoeffCostTables, TxTypeCosts};

const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
const BLK_W: [usize; 22] =
    [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
const BLK_H: [usize; 22] =
    [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
const VAR_IDX: [usize; 19] = [0, 4, 9, 14, 18, 1, 3, 5, 8, 10, 13, 15, 17, 2, 7, 6, 12, 11, 16];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    fn cost(&mut self) -> i32 {
        self.range(0, 20 << 9)
    }
}

fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

fn cdf_row4(rng: &mut Rng, nsymbs: usize) -> [u16; 4] {
    let mut row = [0u16; 4];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, 32000 / nsymbs as i32) as u32;
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

fn gen_cdfs(rng: &mut Rng, count: usize, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(count * padded);
    for _ in 0..count {
        let mut row = vec![0u16; padded];
        let mut acc: u32 = 0;
        for e in row.iter_mut().take(nsymbs - 1) {
            acc += rng.range(1, (32000 / nsymbs as i32).max(2)) as u32;
            *e = (32768u32.saturating_sub(acc)).max(1) as u16;
        }
        row[nsymbs - 1] = 0;
        v.extend_from_slice(&row);
    }
    v
}

/// C-side search_tx_type for one txb (the chain of REAL pieces; loop control
/// transcribed from tx_search.c 2199-2363). Returns the winner
/// (tx_type, eob, rate, dist, sse, entropy_ctx, dqcoeff, rd).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn c_search_tx_type(
    residual: &[i16],
    pred: &[u16],
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    tx_size: usize,
    mode: usize,
    use_fi: bool,
    fi_mode: usize,
    lossless: bool,
    reduced: bool,
    bd: u8,
    plane_rows_c: &[i16],
    dequant: [i16; 2],
    t_above: &[i8],
    t_left: &[i8],
    bsize: usize,
    rdmult: i32,
    ref_best_rd: i64,
    coeff_tbls: (&[i32], &[i32], &[i32], &[i32], &[i32], &[i32], &[i32]),
    ttc_tables: (&[i32], &[i32]),
) -> (usize, u16, i32, i64, i64, u8, Vec<i32>, i64) {
    let (w, _h) = (TX_W[tx_size], TX_H[tx_size]);
    let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
    let (txb_skip, base_eob, base, eob_extra, dc_sign, lps, eob_tbl) = coeff_tbls;
    let (mask_c, _txk) = c::ref_get_tx_mask_intra(
        tx_size as i32,
        mode as i32,
        use_fi,
        fi_mode as i32,
        lossless,
        reduced,
        1,
        false,
        true,
        false,
    );
    // The residual buffer is one txb, stride = txw: use the tx-dims bsize
    // twin (interior; plane_bsize only affects stride/visible clipping).
    let tx_bsize_twin = tx_to_bsize(tx_size);
    let (bsse_raw, mut mse_c) = c::ref_pixel_diff_dist(
        residual, tx_bsize_twin as i32, tx_bsize_twin as i32, 0, 0, 0, 0, 0, 0,
    );
    let mut bsse_c = bsse_raw;
    if bd > 8 {
        let s = 2 * (bd as i32 - 8);
        bsse_c = (bsse_c + ((1i64 << s) >> 1)) >> s;
        mse_c = (((mse_c as u64) + ((1u64 << s) >> 1)) >> s) as u32;
    }
    bsse_c *= 16;
    let dequant_shift = if bd > 8 { bd as i32 - 5 } else { 3 };
    let qstep_c = (i32::from(dequant[1]) >> dequant_shift) as u64;
    let skip_trellis_c = !((mse_c as u64) <= 3200u64 * qstep_c * qstep_c);
    let kind_c = if skip_trellis_c { 1 } else { 0 };
    let trellis_rdmult = trellis_rdmult_intra_y(rdmult, 0, bd);
    let (txb_skip_ctx_c, dc_sign_ctx_c) = c::ref_get_txb_ctx(bsize, tx_size, 0, t_above, t_left);

    let mut best_rd_c = i64::MAX;
    let mut best: Option<(usize, u16, i32, i64, i64, u8, Vec<i32>)> = None;
    for tx_type in 0..16usize {
        if mask_c & (1 << tx_type) == 0 {
            continue;
        }
        let coeff = c::ref_fwd_txfm2d(tx_size, residual, w, tx_type);
        let tcoeff = coeff[..n_coeffs].to_vec();
        let mut qc = vec![0i32; n_coeffs];
        let mut dqc = vec![0i32; n_coeffs];
        let eob = c::ref_quant_plane_rows(
            kind_c,
            bd > 8,
            &tcoeff,
            plane_rows_c,
            scan(tx_size, tx_type),
            aom_txb::iscan(tx_size, tx_type),
            aom_encode::tx_scale(tx_size),
            &mut qc,
            &mut dqc,
        ) as usize;
        let ttc = |eob: usize| -> i32 {
            if eob > 0 {
                c::ref_get_tx_type_cost(
                    ttc_tables.0,
                    ttc_tables.1,
                    0,
                    tx_size as i32,
                    tx_type as i32,
                    false,
                    reduced,
                    lossless,
                    use_fi,
                    fi_mode as i32,
                    mode as i32,
                )
            } else {
                0
            }
        };
        let (eob, rate_c, ctx_c) = if !skip_trellis_c {
            if eob == 0 {
                (0usize, txb_skip[txb_skip_ctx_c as usize * 2 + 1], 0u8)
            } else {
                let (ne, r) = c::ref_optimize_txb(
                    tx_size,
                    tx_type,
                    &mut qc,
                    &mut dqc,
                    &tcoeff,
                    eob,
                    &dequant,
                    trellis_rdmult,
                    dc_sign_ctx_c as usize,
                    txb_skip_ctx_c as usize,
                    0,
                    scan(tx_size, tx_type),
                    txb_skip,
                    base_eob,
                    base,
                    eob_extra,
                    dc_sign,
                    lps,
                    eob_tbl,
                );
                let ctx = c::ref_txb_entropy_context(&qc, tx_size, tx_type, ne);
                (ne, r + ttc(ne), ctx)
            }
        } else {
            let r = c::ref_cost_coeffs_txb(
                &qc,
                eob,
                tx_size,
                tx_type,
                txb_skip_ctx_c as usize,
                dc_sign_ctx_c as usize,
                txb_skip,
                base_eob,
                base,
                eob_extra,
                dc_sign,
                lps,
                eob_tbl,
            ) + ttc(eob);
            let ctx = c::ref_txb_entropy_context(&qc, tx_size, tx_type, eob);
            (eob, r, ctx)
        };
        if c::ref_rdcost(rdmult, rate_c, 0) > best_rd_c {
            continue;
        }
        let (dist_c, sse_c) = if eob == 0 {
            (bsse_c, bsse_c)
        } else {
            let high_energy = bsse_c >= 128 * 128 * TX_SIZE_2D_TBL[tx_size];
            let is_tx64 = tx_size == 4;
            let mut d = i64::MAX;
            let mut s_tx = i64::MAX;
            let mut sse_diff = i64::MAX;
            if is_tx64 || high_energy {
                let (dt, st) = c::ref_dist_block_tx_domain(&tcoeff, &dqc, tx_size, bd);
                d = dt;
                s_tx = st;
                sse_diff = bsse_c - st;
            }
            if !is_tx64 || !high_energy || sse_diff * 2 < s_tx {
                let tx_dom = d;
                let mut recon = pred.to_vec();
                c::ref_inv_txfm2d_add(tx_size, &dqc, &mut recon, w, tx_type, bd as i32);
                let (_v, vf_sse) = c::ref_hbd_variance(
                    VAR_IDX[tx_size],
                    bd,
                    &src[src_off..],
                    src_stride,
                    &recon,
                    w,
                );
                d = 16 * i64::from(vf_sse);
                if high_energy && d < tx_dom {
                    d = tx_dom;
                }
            } else {
                d += sse_diff;
            }
            (d, bsse_c)
        };
        let rd = c::ref_rdcost(rdmult, rate_c, dist_c);
        if rd < best_rd_c {
            best_rd_c = rd;
            best = Some((tx_type, eob as u16, rate_c, dist_c, sse_c, ctx_c, dqc.clone()));
        }
        if (best_rd_c - (best_rd_c >> 1)) > ref_best_rd {
            break;
        }
    }
    let b = best.expect("C search always yields a winner");
    (b.0, b.1, b.2, b.3, b.4, b.5, b.6, best_rd_c)
}

/// BLOCK_SIZE with the same dims as a TX_SIZE.
fn tx_to_bsize(tx_size: usize) -> usize {
    const T: [usize; 19] = [0, 3, 6, 9, 12, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21];
    T[tx_size]
}

#[test]
fn uniform_txfm_yrd_intra_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0x0a11_ab0a_2d5e_a4c4);
    const STRIDE: usize = 256;
    // (bsize, tx_size): multi-txb walks + single-txb controls.
    let pairs: [(usize, usize); 7] =
        [(3, 0), (6, 1), (6, 2), (5, 1), (4, 0), (9, 2), (5, 8)];
    let mut multi_txb_valid = 0usize;
    let mut invalid_cases = 0usize;
    let mut recon_changed = 0usize;

    for (pi, &(bsize, tx_size)) in pairs.iter().enumerate() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let n_txbs = (bw / TX_W[tx_size]) * (bh / TX_H[tx_size]);
        for iter in 0..10 {
            let bd: u8 = if iter % 3 == 2 { 12 } else { 8 };
            let amp = match iter % 4 {
                0 => if bd > 8 { 4095 } else { 255 },
                1 => 24,
                2 => 6,
                _ => 96,
            };
            let qindex = [16, 64, 128, 200, 255][iter % 5] as usize;
            let mode = (rng.next() % 13) as usize;
            let angle_delta = if (1..=8).contains(&mode) { rng.range(-3, 4) } else { 0 };
            let reduced = iter % 4 == 3;

            // Frame geometry: interior block at (mi 8, mi 8) of a large frame.
            let (mi_row, mi_col) = (8, 8);
            let ref_off = 32 * STRIDE + 32;
            let src_off = 32 * STRIDE + 32;

            // Planes.
            let recon0: Vec<u16> =
                (0..STRIDE * 96).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let mut src = recon0.clone();
            for r in 0..bh {
                for cx in 0..bw {
                    let idx = src_off + r * STRIDE + cx;
                    let v = i64::from(recon0[idx]) + i64::from(rng.range(-amp, amp + 1));
                    src[idx] = v.clamp(0, (1 << bd) - 1) as u16;
                }
            }

            // Quantizer rows.
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows = set_q_index(&quants, &deq, qindex, 0);
            let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
            let plane_rows_c = &rows_c[0..7 * 8];

            // Cost tables.
            let txb_skip = tbl(&mut rng, 13 * 2);
            let base_eob = tbl(&mut rng, 4 * 3);
            let base = tbl(&mut rng, 42 * 8);
            let eob_extra = tbl(&mut rng, 9 * 2);
            let dc_sign = tbl(&mut rng, 3 * 2);
            let lps = tbl(&mut rng, 21 * 26);
            let eob_tbl = tbl(&mut rng, 2 * 11);
            let coeff_costs = CoeffCostTables {
                txb_skip: &txb_skip,
                base_eob: &base_eob,
                base: &base,
                eob_extra: &eob_extra,
                dc_sign: &dc_sign,
                lps: &lps,
                eob: &eob_tbl,
            };
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

            // tx-size costs + skip costs + contexts.
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

            let above_ctx: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_ctx: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let rdmult = rng.range(1, 1 << 22);
            let ref_best_rd = if iter == 9 { 1 << 8 } else { i64::MAX };
            let pol = TxTypeSearchPolicy::speed0_allintra();

            // ---- Rust side ----
            let env = TxfmYrdEnv {
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
                src: &src,
                src_off,
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                mode,
                angle_delta,
                use_filter_intra: false,
                filter_intra_mode: 0,
                lossless: false,
                reduced_tx_set_used: reduced,
                bd,
                rows: &rows,
                rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                skip_costs: &skip_costs,
                skip_ctx,
                tx_size_costs: &tx_size_costs,
                tx_size_ctx,
                tx_mode_is_select: true,
                above_ctx: &above_ctx,
                left_ctx: &left_ctx,
            };
            let mut recon_rust = recon0.clone();
            let (rd_rust, stats_rust) =
                uniform_txfm_yrd_intra(&env, &mut recon_rust, tx_size, ref_best_rd, &pol);

            // ---- C-side walk ----
            let tx_size_rate =
                c::ref_tx_size_cost(&ts_flat, true, bsize as i32, tx_size as i32, tx_size_ctx as i32);
            assert_eq!(
                tx_size_rate,
                tx_size_cost(&tx_size_costs, true, bsize, tx_size, tx_size_ctx),
                "tx_size_rate cross-check",
            );
            let no_skip_rate = skip_costs[skip_ctx][0];
            let no_this_rd = c::ref_rdcost(rdmult, no_skip_rate + tx_size_rate, 0);

            let mut recon_c = recon0.clone();
            let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
            let (txwu, txhu) = (txw >> 2, txh >> 2);
            let mut t_above = above_ctx[..bw >> 2].to_vec();
            let mut t_left = left_ctx[..bh >> 2].to_vec();
            let mut rate_sum: i64 = 0;
            let mut dist_sum: i64 = 0;
            let mut sse_sum: i64 = 0;
            let mut winners_c: Vec<(usize, u16, u8)> = Vec::new();
            let mut current_rd = no_this_rd;
            let mut invalid = current_rd > ref_best_rd;
            'walk: for blk_row in (0..bh >> 2).step_by(txhu) {
                for blk_col in (0..bw >> 2).step_by(txwu) {
                    if invalid {
                        break 'walk;
                    }
                    let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                        12,
                        bsize,
                        mi_row,
                        mi_col,
                        true,
                        true,
                        1 << 16,
                        1 << 16,
                        0,
                        tx_size,
                        0,
                        0,
                        blk_row as i32,
                        blk_col as i32,
                        bw as i32,
                        bh as i32,
                        512,
                        512,
                        mode,
                        angle_delta * 3,
                        false,
                    );
                    let txb_off = ref_off + (blk_row * STRIDE + blk_col) * 4;
                    let pred = c::ref_hbd_predict_intra(
                        &recon_c,
                        txb_off,
                        STRIDE,
                        mode,
                        angle_delta * 3,
                        false,
                        0,
                        false,
                        0,
                        tx_size,
                        txw,
                        txh,
                        n_top,
                        n_tr,
                        n_left,
                        n_bl,
                        bd as i32,
                    );
                    for r in 0..txh {
                        recon_c[txb_off + r * STRIDE..txb_off + r * STRIDE + txw]
                            .copy_from_slice(&pred[r * txw..r * txw + txw]);
                    }
                    let src_txb_off = src_off + (blk_row * STRIDE + blk_col) * 4;
                    let mut residual = vec![0i16; txw * txh];
                    c::ref_highbd_subtract_block(
                        txh,
                        txw,
                        &mut residual,
                        txw,
                        &src[src_txb_off..],
                        STRIDE,
                        &pred,
                        txw,
                    );
                    let (wtype, weob, wrate, wdist, wsse, wctx, wdqc, _wrd) = c_search_tx_type(
                        &residual,
                        &pred,
                        &src,
                        src_txb_off,
                        STRIDE,
                        tx_size,
                        mode,
                        false,
                        0,
                        false,
                        reduced,
                        bd,
                        plane_rows_c,
                        [rows.dequant[0], rows.dequant[1]],
                        &t_above[blk_col..],
                        &t_left[blk_row..],
                        bsize,
                        rdmult,
                        ref_best_rd - current_rd,
                        (&txb_skip, &base_eob, &base, &eob_extra, &dc_sign, &lps, &eob_tbl),
                        (&c_ttc_intra, &c_ttc_inter),
                    );
                    if weob > 0 {
                        let mut tight = pred.clone();
                        c::ref_inv_txfm2d_add(tx_size, &wdqc, &mut tight, txw, wtype, bd as i32);
                        for r in 0..txh {
                            recon_c[txb_off + r * STRIDE..txb_off + r * STRIDE + txw]
                                .copy_from_slice(&tight[r * txw..r * txw + txw]);
                        }
                    }
                    for a in t_above[blk_col..blk_col + txwu].iter_mut() {
                        *a = wctx as i8;
                    }
                    for l in t_left[blk_row..blk_row + txhu].iter_mut() {
                        *l = wctx as i8;
                    }
                    winners_c.push((wtype, weob, wctx));
                    rate_sum += i64::from(wrate);
                    dist_sum += wdist;
                    sse_sum += wsse;
                    current_rd += c::ref_rdcost(rdmult, wrate, wdist);
                    if current_rd > ref_best_rd {
                        invalid = true;
                    }
                }
            }

            let m = format!(
                "pair={pi} bsize={bsize} tx={tx_size} n_txbs={n_txbs} iter={iter} bd={bd} \
                 amp={amp} q={qindex} mode={mode}/{angle_delta} red={reduced}",
            );
            if invalid {
                assert_eq!(rd_rust, i64::MAX, "invalid rd {m}");
                assert!(stats_rust.is_none(), "invalid stats {m}");
                invalid_cases += 1;
                continue;
            }
            let rate_total = rate_sum.min(i64::from(i32::MAX)) as i32;
            let rd_c =
                c::ref_rdcost(rdmult, rate_total + no_skip_rate + tx_size_rate, dist_sum);
            let (s, wins) = stats_rust.unwrap_or_else(|| panic!("stats missing {m}"));
            assert_eq!(rd_rust, rd_c, "rd {m}");
            assert_eq!(s.rate, rate_total + tx_size_rate, "rate {m}");
            assert_eq!(s.dist, dist_sum, "dist {m}");
            assert_eq!(s.sse, sse_sum, "sse {m}");
            assert!(!s.skip_txfm, "intra signalled non-skip {m}");
            assert_eq!(recon_rust, recon_c, "recon plane {m}");
            assert_eq!(wins.len(), winners_c.len(), "winner count {m}");
            for (i, (wr, wc)) in wins.iter().zip(winners_c.iter()).enumerate() {
                assert_eq!(
                    (wr.tx_type, wr.eob, wr.txb_ctx),
                    (wc.0, wc.1, wc.2),
                    "winner {i} {m}",
                );
            }
            if n_txbs > 1 {
                multi_txb_valid += 1;
            }
            if recon_rust != recon0 {
                recon_changed += 1;
            }
        }
    }
    assert!(multi_txb_valid > 25, "multi-txb walks: {multi_txb_valid}");
    assert!(invalid_cases > 3, "invalid (early-exit) cases: {invalid_cases}");
    assert!(recon_changed > 40, "recon feedback unexercised: {recon_changed}");
}

/// C-side `uniform_txfm_yrd` for one size: the full walk + intra assembly.
/// Returns `(rd, Some((rate, dist, sse, winners)))` or `(MAX, None)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn c_uniform_txfm_yrd(
    bsize: usize,
    tx_size: usize,
    geometry: (i32, i32, usize, usize, usize),  // mi_row, mi_col, ref_off, src_off, stride
    recon_c: &mut [u16],
    src: &[u16],
    mode: usize,
    angle_delta: i32,
    lossless: bool,
    reduced: bool,
    bd: u8,
    plane_rows_c: &[i16],
    dequant: [i16; 2],
    above_ctx: &[i8],
    left_ctx: &[i8],
    rdmult: i32,
    ref_best_rd: i64,
    coeff_tbls: (&[i32], &[i32], &[i32], &[i32], &[i32], &[i32], &[i32]),
    ttc_tables: (&[i32], &[i32]),
    skip_costs: &[[i32; 2]; 3],
    skip_ctx: usize,
    ts_flat: &[i32],
    tx_size_ctx: usize,
) -> (i64, Option<(i32, i64, i64, Vec<(usize, u16, u8)>)>) {
    let (mi_row, mi_col, ref_off, src_off, stride) = geometry;
    let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
    // tx_mode_is_select = !lossless (select_tx_mode: lossless => ONLY_4X4).
    let tx_size_rate =
        c::ref_tx_size_cost(ts_flat, !lossless, bsize as i32, tx_size as i32, tx_size_ctx as i32);
    let no_skip_rate = skip_costs[skip_ctx][0];
    let no_this_rd = c::ref_rdcost(rdmult, no_skip_rate + tx_size_rate, 0);
    if no_this_rd > ref_best_rd {
        return (i64::MAX, None);
    }
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let (txwu, txhu) = (txw >> 2, txh >> 2);
    let mut t_above = above_ctx[..bw >> 2].to_vec();
    let mut t_left = left_ctx[..bh >> 2].to_vec();
    let mut rate_sum: i64 = 0;
    let mut dist_sum: i64 = 0;
    let mut sse_sum: i64 = 0;
    let mut winners: Vec<(usize, u16, u8)> = Vec::new();
    let mut current_rd = no_this_rd;
    let mut invalid = false;
    'walk: for blk_row in (0..bh >> 2).step_by(txhu) {
        for blk_col in (0..bw >> 2).step_by(txwu) {
            if invalid {
                break 'walk;
            }
            let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                12, bsize, mi_row, mi_col, true, true, 1 << 16, 1 << 16, 0, tx_size, 0, 0,
                blk_row as i32, blk_col as i32, bw as i32, bh as i32, 512, 512, mode,
                angle_delta * 3, false,
            );
            let txb_off = ref_off + (blk_row * stride + blk_col) * 4;
            let pred = c::ref_hbd_predict_intra(
                recon_c, txb_off, stride, mode, angle_delta * 3, false, 0, false, 0, tx_size,
                txw, txh, n_top, n_tr, n_left, n_bl, bd as i32,
            );
            for r in 0..txh {
                recon_c[txb_off + r * stride..txb_off + r * stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }
            let src_txb_off = src_off + (blk_row * stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            c::ref_highbd_subtract_block(
                txh, txw, &mut residual, txw, &src[src_txb_off..], stride, &pred, txw,
            );
            let (wtype, weob, wrate, wdist, wsse, wctx, wdqc, _wrd) = c_search_tx_type(
                &residual, &pred, src, src_txb_off, stride, tx_size, mode, false, 0, lossless,
                reduced, bd, plane_rows_c, dequant, &t_above[blk_col..], &t_left[blk_row..],
                bsize, rdmult, ref_best_rd - current_rd, coeff_tbls, ttc_tables,
            );
            if weob > 0 {
                let mut tight = pred.clone();
                c::ref_inv_txfm2d_add(tx_size, &wdqc, &mut tight, txw, wtype, bd as i32);
                for r in 0..txh {
                    recon_c[txb_off + r * stride..txb_off + r * stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }
            for a in t_above[blk_col..blk_col + txwu].iter_mut() {
                *a = wctx as i8;
            }
            for l in t_left[blk_row..blk_row + txhu].iter_mut() {
                *l = wctx as i8;
            }
            winners.push((wtype, weob, wctx));
            rate_sum += i64::from(wrate);
            dist_sum += wdist;
            sse_sum += wsse;
            current_rd += c::ref_rdcost(rdmult, wrate, wdist);
            if current_rd > ref_best_rd {
                invalid = true;
            }
        }
    }
    if invalid {
        return (i64::MAX, None);
    }
    let rate_total = rate_sum.min(i64::from(i32::MAX)) as i32;
    let rd = c::ref_rdcost(rdmult, rate_total + no_skip_rate + tx_size_rate, dist_sum);
    (rd, Some((rate_total + tx_size_rate, dist_sum, sse_sum, winners)))
}

#[test]
fn pick_uniform_tx_size_type_yrd_matches_c_depth_loop() {
    use aom_encode::tx_search::{
        get_search_init_depth_intra_speed0, pick_uniform_tx_size_type_yrd_intra,
        MAX_TXSIZE_RECT_LOOKUP, SUB_TX_SIZE_MAP,
    };
    c::ref_init();
    let mut rng = Rng(0xdee_bd00_2026_0714 ^ 0x0f00_0000_0000_0000);
    const STRIDE: usize = 256;
    const MI_W: [usize; 22] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
    const MI_H: [usize; 22] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];
    let bsizes = [3usize, 6, 5, 9]; // 8x8, 16x16, 16x8, 32x32
    let mut deeper_won = 0usize;
    let mut top_won = 0usize;
    let mut prune_fired = 0usize;
    let mut lossless_cases = 0usize;

    for (bi, &bsize) in bsizes.iter().enumerate() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        for iter in 0..12 {
            let bd: u8 = if iter % 3 == 2 { 12 } else { 8 };
            let amp = match iter % 4 {
                0 => if bd > 8 { 2048 } else { 160 },
                1 => 24,
                2 => 6,
                _ => 64,
            };
            let qindex = [16, 64, 128, 200][iter % 4] as usize;
            let mode = (rng.next() % 13) as usize;
            let angle_delta = if (1..=8).contains(&mode) { rng.range(-3, 4) } else { 0 };
            let reduced = iter % 4 == 3;
            let lossless = iter == 11;
            let source_variance = if iter % 3 == 0 { rng.range(0, 256) as u32 } else { 256 + rng.range(0, 4096) as u32 };
            let (mi_row, mi_col) = (8, 8);
            let ref_off = 32 * STRIDE + 32;
            let src_off = ref_off;

            let recon0: Vec<u16> =
                (0..STRIDE * 96).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
            let mut src = recon0.clone();
            for r in 0..bh {
                for cx in 0..bw {
                    let idx = src_off + r * STRIDE + cx;
                    let v = i64::from(recon0[idx]) + i64::from(rng.range(-amp, amp + 1));
                    src[idx] = v.clamp(0, (1 << bd) - 1) as u16;
                }
            }
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows = set_q_index(&quants, &deq, qindex, 0);
            let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
            let plane_rows_c = &rows_c[0..7 * 8];

            let txb_skip = tbl(&mut rng, 13 * 2);
            let base_eob = tbl(&mut rng, 4 * 3);
            let base = tbl(&mut rng, 42 * 8);
            let eob_extra = tbl(&mut rng, 9 * 2);
            let dc_sign = tbl(&mut rng, 3 * 2);
            let lps = tbl(&mut rng, 21 * 26);
            let eob_tbl = tbl(&mut rng, 2 * 11);
            let coeff_costs = CoeffCostTables {
                txb_skip: &txb_skip,
                base_eob: &base_eob,
                base: &base,
                eob_extra: &eob_extra,
                dc_sign: &dc_sign,
                lps: &lps,
                eob: &eob_tbl,
            };
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
            let above_ctx: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_ctx: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let rdmult = rng.range(1, 1 << 22);
            let ref_best_rd = i64::MAX;
            let pol = TxTypeSearchPolicy::speed0_allintra();

            let env = TxfmYrdEnv {
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
                src: &src,
                src_off,
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                mode,
                angle_delta,
                use_filter_intra: false,
                filter_intra_mode: 0,
                lossless,
                reduced_tx_set_used: reduced,
                bd,
                rows: &rows,
                rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                skip_costs: &skip_costs,
                skip_ctx,
                tx_size_costs: &tx_size_costs,
                tx_size_ctx,
                tx_mode_is_select: !lossless,
                above_ctx: &above_ctx,
                left_ctx: &left_ctx,
            };
            let mut recon_rust = recon0.clone();
            let got = pick_uniform_tx_size_type_yrd_intra(
                &env,
                &mut recon_rust,
                ref_best_rd,
                &pol,
                source_variance,
                true,
                true,
            );

            // ---- C-side depth loop (choose_tx_size_type_from_rd transcribed) ----
            let mut recon_c = recon0.clone();
            let geometry = (mi_row, mi_col, ref_off, src_off, STRIDE);
            let coeff_tbls =
                (&txb_skip[..], &base_eob[..], &base[..], &eob_extra[..], &dc_sign[..], &lps[..], &eob_tbl[..]);
            let ttc_tables = (&c_ttc_intra[..], &c_ttc_inter[..]);
            let dequant = [rows.dequant[0], rows.dequant[1]];
            #[allow(clippy::type_complexity)] // (tx, rd, rate, dist, sse, winners)
            let mut best_c: Option<(usize, i64, i32, i64, i64, Vec<(usize, u16, u8)>)> = None;
            if lossless {
                let (rd, res) = c_uniform_txfm_yrd(
                    bsize, 0, geometry, &mut recon_c, &src, mode, angle_delta, lossless,
                    reduced, bd, plane_rows_c, dequant, &above_ctx, &left_ctx, rdmult,
                    ref_best_rd, coeff_tbls, ttc_tables, &skip_costs, skip_ctx, &ts_flat,
                    tx_size_ctx,
                );
                if let Some((rate, dist, sse, w)) = res {
                    best_c = Some((0, rd, rate, dist, sse, w));
                }
                lossless_cases += 1;
            } else {
                let init_depth = get_search_init_depth_intra_speed0(MI_W[bsize], MI_H[bsize]);
                let start_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
                let mut rd_arr = [i64::MAX; 3];
                let mut best_rd_c = i64::MAX;
                let mut tx = start_tx;
                let mut depth = init_depth;
                while depth <= 2 {
                    let (rd, res) = c_uniform_txfm_yrd(
                        bsize, tx, geometry, &mut recon_c, &src, mode, angle_delta, false,
                        reduced, bd, plane_rows_c, dequant, &above_ctx, &left_ctx, rdmult,
                        ref_best_rd, coeff_tbls, ttc_tables, &skip_costs, skip_ctx, &ts_flat,
                        tx_size_ctx,
                    );
                    rd_arr[depth as usize] = rd;
                    if rd < best_rd_c {
                        best_rd_c = rd;
                        if let Some((rate, dist, sse, w)) = res {
                            best_c = Some((tx, rd, rate, dist, sse, w));
                        }
                    }
                    if tx == 0 {
                        break;
                    }
                    if depth > init_depth && depth != 2 && source_variance < 256 {
                        let prev = rd_arr[depth as usize - 1];
                        if prev != i64::MAX && rd_arr[depth as usize] > prev {
                            prune_fired += 1;
                            break;
                        }
                    }
                    depth += 1;
                    tx = SUB_TX_SIZE_MAP[tx];
                }
            }

            let m = format!(
                "bi={bi} bsize={bsize} iter={iter} bd={bd} amp={amp} q={qindex} \
                 mode={mode}/{angle_delta} var={source_variance} lossless={lossless}",
            );
            match (got, best_c) {
                (Some(g), Some(cb)) => {
                    assert_eq!(g.best_tx_size, cb.0, "tx_size {m}");
                    assert_eq!(g.best_rd, cb.1, "rd {m}");
                    assert_eq!(g.stats.rate, cb.2, "rate {m}");
                    assert_eq!(g.stats.dist, cb.3, "dist {m}");
                    assert_eq!(g.stats.sse, cb.4, "sse {m}");
                    let wins: Vec<(usize, u16, u8)> =
                        g.winners.iter().map(|w| (w.tx_type, w.eob, w.txb_ctx)).collect();
                    assert_eq!(wins, cb.5, "winners {m}");
                    assert_eq!(recon_rust, recon_c, "recon plane {m}");
                    if g.best_tx_size == MAX_TXSIZE_RECT_LOOKUP[bsize] {
                        top_won += 1;
                    } else {
                        deeper_won += 1;
                    }
                }
                (None, None) => {}
                (g, cb) => panic!("presence mismatch {m}: rust={} c={}", g.is_some(), cb.is_some()),
            }
        }
    }
    assert!(top_won > 10, "top-size winners: {top_won}");
    assert!(deeper_won > 5, "deeper-size winners: {deeper_won}");
    assert!(prune_fired > 2, "low-contrast prune never fired: {prune_fired}");
    assert!(lossless_cases >= 4, "lossless arm: {lossless_cases}");
}
