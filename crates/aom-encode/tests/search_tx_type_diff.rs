//! Differential harness for `search_tx_type_intra` (tx_search.c
//! `search_tx_type`, luma intra, speed-0 policy, interior txb) vs the same
//! loop over REAL C pieces: ref_get_tx_mask_intra + ref_pixel_diff_dist +
//! ref_set_q_index rows -> ref_fwd_txfm2d -> ref_quant_plane_rows (REAL
//! facades pick FP/B rows) -> ref_get_txb_ctx -> ref_optimize_txb /
//! ref_cost_coeffs_txb + ref_get_tx_type_cost -> ref_dist_block_tx_domain /
//! (ref_inv_txfm2d_add + ref_hbd_variance) -> ref_rdcost, with the strict-min
//! tracking and the adaptive_txb_search early break transcribed from the C
//! loop (tx_search.c 2199-2363).
//!
//! The trellis rdmult derivation (txb_rdopt.c:390, luma-intra mult 17,
//! rshift 5) is computed once and fed to BOTH sides.

use aom_encode::BlockContext;
use aom_encode::tx_search::{
    TX_SIZE_2D_TBL, TxTypeSearchInputs, TxTypeSearchPolicy, search_tx_type_intra,
    trellis_rdmult_intra_y,
};
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_txb::{CoeffCostTables, TxTypeCosts, fill_tx_type_costs, scan, txb_high, txb_wide};

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
const TX_TO_BSIZE: [usize; 19] = [
    0, 3, 6, 9, 12, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21,
];
/// sadvar_shim variance size index per TX_SIZE (w x h -> case index).
const VAR_IDX: [usize; 19] = [
    0, 4, 9, 14, 18, 1, 3, 5, 8, 10, 13, 15, 17, 2, 7, 6, 12, 11, 16,
];

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

fn gen_cdf_row(rng: &mut Rng, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut row = vec![0u16; padded];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, (32000 / nsymbs as i32).max(2)) as u32;
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

fn gen_cdfs(rng: &mut Rng, count: usize, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(count * padded);
    for _ in 0..count {
        v.extend_from_slice(&gen_cdf_row(rng, nsymbs, padded));
    }
    v
}

#[test]
fn search_tx_type_intra_matches_c_chain() {
    c::ref_init();
    let mut rng = Rng(0x5ea4_c47e_5700_0001);
    const STRIDE: usize = 160;

    // Coverage counters (non-vacuity across the policy space).
    let mut trellis_blocks = 0usize;
    let mut b_quant_blocks = 0usize;
    let mut eob0_winners = 0usize;
    let mut coded_winners = 0usize;
    let mut high_energy_hits = 0usize;
    let mut adaptive_breaks = 0usize;
    let mut multi_type_blocks = 0usize;
    let mut nondct_winners = 0usize;

    for tx_size in 0..19usize {
        let (w, h) = (TX_W[tx_size], TX_H[tx_size]);
        let bsize = TX_TO_BSIZE[tx_size];
        let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
        for iter in 0..14 {
            let bd: u8 = if iter % 3 == 2 { 12 } else { 8 };
            let rmax: i32 = if bd > 8 { 4096 } else { 256 };
            // Amplitude class: full (high-energy / B-quant), small (trellis),
            // near-zero (eob 0 at high qindex).
            let amp = match iter % 4 {
                0 => rmax - 1,
                1 => 24,
                2 => 6,
                _ => 96,
            };
            // qindex band: spread over the range; deltas zero (all-intra
            // default) plus occasional nonzero.
            let qindex = [0, 32, 96, 160, 208, 255][iter % 6] as usize;
            let ydc = if iter % 5 == 4 { rng.range(-16, 17) } else { 0 };
            let use_fi = tx_size <= 2 && iter % 7 == 3; // fi only <= 32x32 blocks
            let fi_mode = if use_fi { (rng.next() % 5) as usize } else { 0 };
            // Filter-intra implies DC_PRED (mbmi->mode semantics).
            let mode = if use_fi {
                0
            } else {
                (rng.next() % 13) as usize
            };
            let reduced = iter % 5 == 2;
            let lossless = false;

            // Pixels: pred + src -> residual = src - pred (consistent px dist).
            let src_off = 3 * STRIDE + 5;
            let src: Vec<u16> = (0..STRIDE * (h + 8))
                .map(|_| (rng.next() % (1u64 << bd)) as u16)
                .collect();
            let pred: Vec<u16> = (0..w * h)
                .map(|i| {
                    let s = i64::from(src[src_off + (i / w) * STRIDE + (i % w)]);
                    let p = s - i64::from(rng.range(-amp, amp + 1));
                    p.clamp(0, (1 << bd) - 1) as u16
                })
                .collect();
            let residual: Vec<i16> = (0..w * h)
                .map(|i| {
                    (i64::from(src[src_off + (i / w) * STRIDE + (i % w)]) - i64::from(pred[i]))
                        as i16
                })
                .collect();

            // Quantizer rows from (bd, deltas, qindex).
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, ydc, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows = set_q_index(&quants, &deq, qindex, 0);
            let rows_c = c::ref_set_q_index(bd as i32, ydc, 0, 0, 0, 0, 0, qindex as i32);
            let plane_rows_c = &rows_c[0..7 * 8];

            // Cost tables (random, shared by both sides).
            let txb_skip = tbl(&mut rng, 13 * 2);
            let base_eob = tbl(&mut rng, 4 * 3);
            let base = tbl(&mut rng, 42 * 8);
            let eob_extra = tbl(&mut rng, 9 * 2);
            let dc_sign = tbl(&mut rng, 3 * 2);
            let lps = tbl(&mut rng, 21 * 26);
            let eob_c_tbl = tbl(&mut rng, 2 * 11);
            let coeff_costs = CoeffCostTables {
                txb_skip: &txb_skip,
                base_eob: &base_eob,
                base: &base,
                eob_extra: &eob_extra,
                dc_sign: &dc_sign,
                lps: &lps,
                eob: &eob_c_tbl,
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

            // Neighbour entropy contexts + rd.
            let above: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let left: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let bctx = BlockContext {
                plane_bsize: bsize,
                plane: 0,
                above: &above,
                left: &left,
            };
            let rdmult = rng.range(1, 1 << 22);
            // Exercise the KB-8 winner-mode MODE_EVAL first-pass tx-type
            // restriction (use_default_intra_tx_type) across ~1/4 of iterations,
            // threaded into BOTH the port policy and the C ref mask so the full
            // search_tx_type chain must still match under the restricted set.
            let use_default = iter % 4 == 0;
            let pol = TxTypeSearchPolicy {
                use_default_intra_tx_type: use_default,
                ..TxTypeSearchPolicy::speed0_allintra()
            };
            // Mostly unconstrained; occasionally tight to exercise the
            // adaptive_txb_search break.
            let ref_best_rd = if iter % 9 == 8 { 1 << 10 } else { i64::MAX };

            // ---- Rust side ----
            let inp = TxTypeSearchInputs {
                residual: &residual,
                src: &src,
                src_off,
                src_stride: STRIDE,
                pred: &pred,
                tx_size,
                plane: 0,
                uv_mode: 0,
                mode,
                use_filter_intra: use_fi,
                filter_intra_mode: fi_mode,
                lossless,
                reduced_tx_set_used: reduced,
                bd,
                rows: &rows,
                bctx: &bctx,
                rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                // Interior differential: visible == full tx dims.
                visible_cols: w,
                visible_rows: h,
                qm_level: None,
                predict_skip_zero_blk_rate: 0,
            };
            let got = search_tx_type_intra(&inp, &pol, ref_best_rd);

            // ---- C chain (loop control transcribed from tx_search.c) ----
            let (mask_c, txk_c) = c::ref_get_tx_mask_intra(
                tx_size as i32,
                mode as i32,
                use_fi,
                fi_mode as i32,
                lossless,
                reduced,
                1,           // use_reduced_intra_txset (speed-0 allintra)
                false,       // use_derived_intra_tx_type_set
                true,        // enable_flip_idtx
                false,       // use_intra_dct_only
                use_default, // use_default_intra_tx_type (KB-8 MODE_EVAL pass)
                false,       // use_screen_content_tools
                0,           // prune_tx_type_using_stats (speed-0 sub-480p: off)
            );
            let _ = txk_c;
            let (bsse_raw, mut mse_c) =
                c::ref_pixel_diff_dist(&residual, bsize as i32, bsize as i32, 0, 0, 0, 0, 0, 0);
            let mut bsse_c = bsse_raw;
            if bd > 8 {
                let s = 2 * (bd as i32 - 8);
                bsse_c = (bsse_c + ((1i64 << s) >> 1)) >> s;
                mse_c = (((mse_c as u64) + ((1u64 << s) >> 1)) >> s) as u32;
            }
            bsse_c *= 16;
            let dequant_shift = if bd > 8 { bd as i32 - 5 } else { 3 };
            let qstep_c = (i32::from(plane_rows_c[6 * 8 + 1]) >> dequant_shift) as u64;
            let skip_trellis_c = !((mse_c as u64) <= 3200u64 * qstep_c * qstep_c);
            let kind_c = if skip_trellis_c { 1 } else { 0 }; // B : FP
            let trellis_rdmult = trellis_rdmult_intra_y(rdmult, 0, bd, false);
            let (txb_skip_ctx_c, dc_sign_ctx_c) =
                c::ref_get_txb_ctx(bsize, tx_size, 0, &above, &left);

            let mut best_rd_c = i64::MAX;
            #[allow(clippy::type_complexity)] // (type, eob, rate, dist, sse, ctx, qc, dqc)
            let mut best_c: Option<(
                usize,
                u16,
                i32,
                i64,
                i64,
                u8,
                Vec<i32>,
                Vec<i32>,
            )> = None;
            for tx_type in 0..16usize {
                if mask_c & (1 << tx_type) == 0 {
                    continue;
                }
                let coeff = c::ref_fwd_txfm2d(tx_size, &residual, w, tx_type);
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
                            &c_ttc_intra,
                            &c_ttc_inter,
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
                let (eob, rate_c, entropy_ctx_c) = if !skip_trellis_c {
                    if eob == 0 {
                        (0usize, txb_skip[txb_skip_ctx_c as usize * 2 + 1], 0u8)
                    } else {
                        let (new_eob, r) = c::ref_optimize_txb(
                            tx_size,
                            tx_type,
                            &mut qc,
                            &mut dqc,
                            &tcoeff,
                            eob,
                            &[rows.dequant[0], rows.dequant[1]],
                            trellis_rdmult,
                            dc_sign_ctx_c as usize,
                            txb_skip_ctx_c as usize,
                            0,
                            scan(tx_size, tx_type),
                            &txb_skip,
                            &base_eob,
                            &base,
                            &eob_extra,
                            &dc_sign,
                            &lps,
                            &eob_c_tbl,
                        );
                        let ctx = c::ref_txb_entropy_context(&qc, tx_size, tx_type, new_eob);
                        (new_eob, r + ttc(new_eob), ctx)
                    }
                } else {
                    let r = c::ref_cost_coeffs_txb(
                        &qc,
                        eob,
                        tx_size,
                        tx_type,
                        txb_skip_ctx_c as usize,
                        dc_sign_ctx_c as usize,
                        &txb_skip,
                        &base_eob,
                        &base,
                        &eob_extra,
                        &dc_sign,
                        &lps,
                        &eob_c_tbl,
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
                    // Pixel-domain with the 64-pt/high-energy hybrid
                    // (use_transform_domain_distortion = 0 at speed 0).
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
                        let mut recon = pred.clone();
                        c::ref_inv_txfm2d_add(tx_size, &dqc, &mut recon, w, tx_type, bd as i32);
                        let (_v, vf_sse) = c::ref_hbd_variance(
                            VAR_IDX[tx_size],
                            bd,
                            &src[src_off..],
                            STRIDE,
                            &recon,
                            w,
                        );
                        d = 16 * i64::from(vf_sse);
                        if high_energy && d < tx_dom {
                            d = tx_dom;
                        }
                        high_energy_hits += high_energy as usize;
                    } else {
                        d += sse_diff;
                    }
                    (d, bsse_c)
                };

                let rd = c::ref_rdcost(rdmult, rate_c, dist_c);
                if rd < best_rd_c {
                    best_rd_c = rd;
                    best_c = Some((
                        tx_type,
                        eob as u16,
                        rate_c,
                        dist_c,
                        sse_c,
                        entropy_ctx_c,
                        qc.clone(),
                        dqc.clone(),
                    ));
                }
                if (best_rd_c - (best_rd_c >> 1)) > ref_best_rd {
                    adaptive_breaks += 1;
                    break;
                }
            }

            // ---- Compare ----
            let m = format!(
                "ts={tx_size} iter={iter} bd={bd} amp={amp} q={qindex} mode={mode} \
                 fi={use_fi}/{fi_mode} red={reduced} trellis={}",
                !skip_trellis_c,
            );
            match (got, best_c) {
                (Some(g), Some(cb)) => {
                    assert_eq!(g.best_tx_type, cb.0, "tx_type {m}");
                    assert_eq!(g.best_eob, cb.1, "eob {m}");
                    assert_eq!(g.rate, cb.2, "rate {m}");
                    assert_eq!(g.dist, cb.3, "dist {m}");
                    assert_eq!(g.sse, cb.4, "sse {m}");
                    assert_eq!(g.best_txb_ctx, cb.5, "txb_ctx {m}");
                    assert_eq!(g.rd, best_rd_c, "rd {m}");
                    assert_eq!(g.skip_txfm, g.best_eob == 0, "skip {m}");
                    assert_eq!(g.qcoeff, cb.6, "qcoeff {m}");
                    assert_eq!(g.dqcoeff, cb.7, "dqcoeff {m}");
                    if skip_trellis_c {
                        b_quant_blocks += 1;
                    } else {
                        trellis_blocks += 1;
                    }
                    if g.best_eob == 0 {
                        eob0_winners += 1;
                    } else {
                        coded_winners += 1;
                    }
                    if g.evaluated_mask.count_ones() > 1 {
                        multi_type_blocks += 1;
                    }
                    if g.best_tx_type != 0 {
                        nondct_winners += 1;
                    }
                }
                (None, None) => {}
                (g, cb) => panic!("presence mismatch {m}: rust={g:?} c_some={}", cb.is_some()),
            }
        }
    }

    // Non-vacuity: every policy arm exercised.
    assert!(trellis_blocks > 40, "trellis arm: {trellis_blocks}");
    assert!(b_quant_blocks > 40, "B-quant arm: {b_quant_blocks}");
    assert!(eob0_winners > 5, "eob0 winners: {eob0_winners}");
    assert!(coded_winners > 100, "coded winners: {coded_winners}");
    assert!(
        high_energy_hits > 20,
        "high-energy hybrid: {high_energy_hits}"
    );
    assert!(adaptive_breaks > 3, "adaptive breaks: {adaptive_breaks}");
    assert!(
        multi_type_blocks > 60,
        "multi-type blocks: {multi_type_blocks}"
    );
    assert!(nondct_winners > 10, "non-DCT winners: {nondct_winners}");
}
