//! Differential harness for the block-level intra-mode RD driver
//! (`intra_mode_rd_eval` / `pick_intra_mode_rd`) vs the identical chain of C
//! reference steps: ref_hbd_predict_intra -> ref_highbd_subtract_block ->
//! ref_fwd_txfm2d -> ref_quantize_{b,fp} -> ref_get_txb_ctx ->
//! ref_optimize_txb -> ref_cost_coeffs_txb + ref_get_tx_type_cost +
//! ref_intra_mode_info_cost_y -> ref_dist_block_tx_domain -> ref_rdcost, with
//! the same argmin rule.
//!
//! This validates the COMPOSITION (wiring + argmin) of the individually
//! C-validated pieces. It does NOT claim to reproduce libaom's
//! av1_rd_pick_intra_sby_mode search (candidate order/pruning, tx-size/type
//! search, skip-RD, recon-domain distortion are out of scope — see the
//! intra_rd module docs). The (above_ctx, left_ctx) pair feeding both sides'
//! y_mode_costs lookup comes from the separately C-validated get_y_mode_ctx.

use aom_encode::intra_rd::{
    IntraCandidate, IntraModeRd, IntraRdEnv, IntraRdRates, pick_intra_mode_rd,
};
use aom_encode::mode_costs::{IntraModeCosts, fill_intra_mode_costs, filter_intra_allowed_bsize};
use aom_encode::{BlockContext, OptimizeInputs, QuantKind, QuantParams};
use aom_entropy::partition::get_y_mode_ctx;
use aom_sys_ref as c;
use aom_txb::{CoeffCostTables, TxTypeCosts, fill_tx_type_costs, scan, txb_high, txb_wide};

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
/// BLOCK_SIZE with the same dims as each TX_SIZE (single-txb blocks).
const TX_TO_BSIZE: [usize; 19] = [
    0, 3, 6, 9, 12, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21,
];
const TX_SIZE_2D: [i32; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];

fn log_scale(tx_size: usize) -> i32 {
    let p = TX_SIZE_2D[tx_size];
    (p > 256) as i32 + (p > 1024) as i32
}

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

/// Valid `nsymbs`-symbol inverse-CDF row padded to `padded` entries.
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

/// The candidate list a caller would sweep for a KEY block: all 13 modes
/// (directional ones with every angle delta) + the filter-intra variants when
/// the block is eligible.
fn build_candidates(bsize: usize, enable_fi: bool) -> Vec<IntraCandidate> {
    let mut cands = Vec::new();
    for mode in 0..13usize {
        let deltas: &[i32] = if (1..=8).contains(&mode) {
            &[-3, -2, -1, 0, 1, 2, 3]
        } else {
            &[0]
        };
        for &d in deltas {
            cands.push(IntraCandidate {
                mode,
                angle_delta: d,
                use_filter_intra: false,
                filter_intra_mode: 0,
            });
        }
    }
    if filter_intra_allowed_bsize(enable_fi, bsize) {
        for fi_mode in 0..5usize {
            cands.push(IntraCandidate {
                mode: 0,
                angle_delta: 0,
                use_filter_intra: true,
                filter_intra_mode: fi_mode,
            });
        }
    }
    cands
}

#[test]
fn pick_intra_mode_rd_matches_c_chain() {
    let mut rng = Rng(0x1417_a4d0_5e1e_c700);
    const STRIDE: usize = 160;
    const ROW0: usize = 8;
    const COL0: usize = 8;
    let mut argmin_spread = 0u64; // blocks where not every candidate tied

    for tx_size in 0..19usize {
        let (w, h) = (TX_W[tx_size], TX_H[tx_size]);
        let bsize = TX_TO_BSIZE[tx_size];
        let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
        let ls = log_scale(tx_size);
        for &bd in &[8u8, 12] {
            // Pixel planes: recon supplies prediction edges, src the block.
            let recon: Vec<u16> = (0..STRIDE * (ROW0 + h + 64))
                .map(|_| (rng.next() % (1u64 << bd)) as u16)
                .collect();
            let src: Vec<u16> = (0..STRIDE * (ROW0 + h))
                .map(|_| (rng.next() % (1u64 << bd)) as u16)
                .collect();
            let ref_off = ROW0 * STRIDE + COL0;
            let src_off = (ROW0 - 1) * STRIDE + COL0 + 1; // distinct block origin

            // Quant params (reciprocal quant/dequant like the real tables).
            let dq = [rng.range(16, 800), rng.range(16, 800)];
            let dequant = [dq[0] as i16, dq[1] as i16];
            let quant = [
                (65536 / dq[0]).clamp(1, 32767) as i16,
                (65536 / dq[1]).clamp(1, 32767) as i16,
            ];
            let zbin = [rng.range(1, 100) as i16, rng.range(1, 100) as i16];
            let round = [rng.range(1, 400) as i16, rng.range(1, 400) as i16];
            let quant_shift = [rng.range(8000, 32767) as i16, rng.range(8000, 32767) as i16];
            let kind = if bd == 8 { QuantKind::B } else { QuantKind::Fp };

            // Coefficient cost tables.
            let txb_skip = tbl(&mut rng, 13 * 2);
            let base_eob = tbl(&mut rng, 4 * 3);
            let base = tbl(&mut rng, 42 * 8);
            let eob_extra = tbl(&mut rng, 9 * 2);
            let dc_sign = tbl(&mut rng, 3 * 2);
            let lps = tbl(&mut rng, 21 * 26);
            let eob_c = tbl(&mut rng, 2 * 11);
            let coeff_costs = CoeffCostTables {
                txb_skip: &txb_skip,
                base_eob: &base_eob,
                base: &base,
                eob_extra: &eob_extra,
                dc_sign: &dc_sign,
                lps: &lps,
                eob: &eob_c,
            };

            // tx-type cost tables from random per-set CDFs (both sides).
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

            // Intra mode-cost tables from random CDFs (both sides).
            let enable_fi = (tx_size + bd as usize).is_multiple_of(2);
            let kf_y = gen_cdfs(&mut rng, 25, 13, 14);
            let y_mode = gen_cdfs(&mut rng, 4, 13, 14);
            let mut uv = gen_cdfs(&mut rng, 13, 13, 15);
            uv.extend_from_slice(&gen_cdfs(&mut rng, 13, 14, 15));
            let fi_mode_cdf = gen_cdfs(&mut rng, 1, 5, 6);
            let fi_cdfs = gen_cdfs(&mut rng, 22, 2, 3);
            let pal_y = gen_cdfs(&mut rng, 21, 2, 3);
            let angle = gen_cdfs(&mut rng, 8, 7, 8);
            let intrabc_cdf = gen_cdfs(&mut rng, 1, 2, 3);
            let c_mode_costs = c::ref_fill_intra_mode_costs(
                &kf_y,
                &y_mode,
                &uv,
                &fi_mode_cdf,
                &fi_cdfs,
                &pal_y,
                &angle,
                &intrabc_cdf,
                enable_fi,
            );
            let mut mode_costs = IntraModeCosts::zeroed();
            fill_intra_mode_costs(
                &mut mode_costs,
                &kf_y,
                &y_mode,
                &uv,
                &fi_mode_cdf,
                &fi_cdfs,
                &pal_y,
                &angle,
                &intrabc_cdf,
                enable_fi,
            );

            // Neighbour entropy contexts + trellis inputs.
            let above: Vec<i8> = (0..16)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let left: Vec<i8> = (0..16)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let rdmult = rng.range(1, 1 << 20);
            let sharpness = rng.range(0, 8);

            // Mode-rate environment.
            let above_mode = if rng.range(0, 4) == 0 {
                None
            } else {
                Some(rng.range(0, 13))
            };
            let left_mode = if rng.range(0, 4) == 0 {
                None
            } else {
                Some(rng.range(0, 13))
            };
            let (actx, lctx) = get_y_mode_ctx(above_mode, left_mode);
            let try_palette = rng.range(0, 2) == 1;
            let pal_bctx = rng.range(0, 7) as usize;
            let pal_mctx = rng.range(0, 3) as usize;
            let allow_intrabc = rng.range(0, 2) == 1;
            let lossless = false;
            let reduced = false;
            let tx_type = 0usize; // DCT_DCT (always valid; no tx-type search here)

            // Edge availability (full / no-extension alternating).
            let combo: (usize, i32, usize, i32) = if tx_size % 2 == 0 {
                (w, h as i32, h, w as i32)
            } else {
                (w, -1, h, -1)
            };

            let qp = QuantParams {
                zbin: &zbin,
                round: &round,
                quant: &quant,
                quant_shift: &quant_shift,
                dequant: &dequant,
                qm: None,
                iqm: None,
                bd,
                lossless: false,
                qm_ctx: None,
            };
            let bctx = BlockContext {
                above: &above,
                left: &left,
                plane: 0,
                plane_bsize: bsize,
            };
            let opt = OptimizeInputs {
                cost: &coeff_costs,
                rdmult: rdmult as i64,
                sharpness,
            };
            let env = IntraRdEnv {
                recon: &recon,
                ref_off,
                ref_stride: STRIDE,
                src: &src,
                src_off,
                src_stride: STRIDE,
                tx_size,
                bsize,
                n_top_px: combo.0,
                n_topright_px: combo.1,
                n_left_px: combo.2,
                n_bottomleft_px: combo.3,
                disable_edge_filter: false,
                filter_type: 1,
                bd,
            };
            let rates = IntraRdRates {
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                mode_costs: &mode_costs,
                rdmult,
                above_mode,
                left_mode,
                try_palette,
                palette_bsize_ctx: pal_bctx,
                palette_mode_ctx: pal_mctx,
                enable_filter_intra: enable_fi,
                allow_intrabc,
                reduced_tx_set: reduced,
                lossless,
            };

            let cands = build_candidates(bsize, enable_fi);
            let (got_best, got_evals) =
                pick_intra_mode_rd(&env, &rates, &cands, tx_type, kind, &qp, &bctx, &opt);

            // ---- C reference chain, same loop -------------------------------
            let sc = scan(tx_size, tx_type);
            let mut want_evals: Vec<IntraModeRd> = Vec::with_capacity(cands.len());
            for cand in &cands {
                let pred = c::ref_hbd_predict_intra(
                    &recon,
                    ref_off,
                    STRIDE,
                    cand.mode,
                    cand.angle_delta * 3,
                    cand.use_filter_intra,
                    cand.filter_intra_mode,
                    false,
                    1,
                    tx_size,
                    w,
                    h,
                    combo.0 as i32,
                    combo.1,
                    combo.2 as i32,
                    combo.3,
                    bd as i32,
                );
                let mut residual = vec![0i16; w * h];
                c::ref_highbd_subtract_block(
                    h,
                    w,
                    &mut residual,
                    w,
                    &src[src_off..],
                    STRIDE,
                    &pred,
                    w,
                );
                let coeff = c::ref_fwd_txfm2d(tx_size, &residual, w, tx_type);
                let tcoeff = &coeff[..n_coeffs];
                // bd > 8 must use the highbd (64-bit) quantizer refs, matching
                // xform_quant's bd dispatch (the lowbd ones i16-saturate).
                let (mut qc, mut dqc, eob0) = match (kind, bd > 8) {
                    (QuantKind::B, false) => c::ref_quantize_b(
                        ls,
                        tcoeff,
                        &zbin,
                        &round,
                        &quant,
                        &quant_shift,
                        &dequant,
                        sc,
                    ),
                    (QuantKind::B, true) => c::ref_highbd_quantize_b(
                        ls,
                        tcoeff,
                        &zbin,
                        &round,
                        &quant,
                        &quant_shift,
                        &dequant,
                        sc,
                    ),
                    (QuantKind::Fp, false) => {
                        c::ref_quantize_fp(ls, tcoeff, &round, &quant, &dequant, sc)
                    }
                    (QuantKind::Fp, true) => {
                        c::ref_highbd_quantize_fp(ls, tcoeff, &round, &quant, &dequant, sc)
                    }
                    (QuantKind::Dc, _) => unreachable!(),
                };
                let (skip_ctx, sign_ctx) = c::ref_get_txb_ctx(bsize, tx_size, 0, &above, &left);
                let (eob_w, _trellis_rate) = if eob0 == 0 {
                    (0usize, 0i32)
                } else {
                    c::ref_optimize_txb(
                        tx_size,
                        tx_type,
                        &mut qc,
                        &mut dqc,
                        tcoeff,
                        eob0 as usize,
                        &dequant,
                        rdmult as i64,
                        sign_ctx as usize,
                        skip_ctx as usize,
                        sharpness,
                        sc,
                        &txb_skip,
                        &base_eob,
                        &base,
                        &eob_extra,
                        &dc_sign,
                        &lps,
                        &eob_c,
                    )
                };
                // av1_cost_coeffs_txb: the eob==0 branch returns the txb_skip
                // cost alone (txb_rdopt.c:691 — no tx_type term; the coeff-only
                // shim's contract is eob >= 1); the eob>0 body adds
                // get_tx_type_cost inside.
                let (coeff_rate, tx_type_rate) = if eob_w == 0 {
                    (txb_skip[skip_ctx as usize * 2 + 1], 0)
                } else {
                    (
                        c::ref_cost_coeffs_txb(
                            &qc,
                            eob_w,
                            tx_size,
                            tx_type,
                            skip_ctx as usize,
                            sign_ctx as usize,
                            &txb_skip,
                            &base_eob,
                            &base,
                            &eob_extra,
                            &dc_sign,
                            &lps,
                            &eob_c,
                        ),
                        c::ref_get_tx_type_cost(
                            &c_ttc_intra,
                            &c_ttc_inter,
                            0,
                            tx_size as i32,
                            tx_type as i32,
                            false,
                            reduced,
                            lossless,
                            cand.use_filter_intra,
                            cand.filter_intra_mode as i32,
                            cand.mode as i32,
                        ),
                    )
                };
                let mode_cost = c_mode_costs.y_mode[(actx * 5 + lctx) * 13 + cand.mode];
                let mode_rate = c::ref_intra_mode_info_cost_y(
                    &c_mode_costs,
                    mode_cost,
                    cand.mode as i32,
                    bsize as i32,
                    cand.angle_delta,
                    cand.use_filter_intra,
                    cand.filter_intra_mode as i32,
                    false,
                    try_palette,
                    pal_bctx as i32,
                    pal_mctx as i32,
                    enable_fi,
                    allow_intrabc
                );
                let rate = coeff_rate + tx_type_rate + mode_rate;
                let (dist, _sse) = c::ref_dist_block_tx_domain(tcoeff, &dqc, tx_size, bd);
                let rd = c::ref_rdcost(rdmult, rate, dist);
                want_evals.push(IntraModeRd {
                    rate,
                    dist,
                    rd,
                    eob: eob_w as u16,
                });
            }
            let mut want_best = 0usize;
            let mut best_rd = i64::MAX;
            for (i, e) in want_evals.iter().enumerate() {
                if e.rd < best_rd {
                    best_rd = e.rd;
                    want_best = i;
                }
            }

            assert_eq!(
                got_evals, want_evals,
                "per-candidate evals ts={tx_size} bd={bd} kind={kind:?}"
            );
            assert_eq!(got_best, want_best, "argmin ts={tx_size} bd={bd}");
            if want_evals.iter().any(|e| e.rd != want_evals[0].rd) {
                argmin_spread += 1;
            }
        }
    }
    // Guard: the argmin must have been a real decision (distinct RDs), not a
    // degenerate all-equal pick, on nearly every block.
    assert!(
        argmin_spread >= 30,
        "only {argmin_spread} blocks had distinct candidate RDs"
    );
}
