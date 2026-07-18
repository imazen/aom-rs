//! Differential for `encode_coding_block_plane` — the transform-block loop that
//! lifts "one txb" to "one coding block". It iterates a plane's txbs in raster
//! order, threading the above/left ENTROPY_CONTEXT arrays (get_txb_ctx in, then a
//! footprint fill with the txb's entropy byte = av1_set_entropy_contexts interior
//! case). The oracle re-runs the same loop with the C references for the context
//! threading (ref_get_txb_ctx / ref_txb_entropy_context) and the quant+optimize,
//! writing each txb with the (separately C-validated) write_coeffs_txb_full. My
//! loop's bytes AND final contexts must match — a divergent txb order, footprint,
//! or context slice would surface immediately.

use aom_encode::{
    BlockContexts, OptimizeInputs, QuantKind, QuantParams, TxTypeContext, encode_coding_block_plane,
};
use aom_entropy::enc::OdEcEnc;
use aom_sys_ref as c;
use aom_txb::{CDF_ARENA_LEN, CoeffCostTables, ext_tx_derive, scan, write_coeffs_txb_full};

const BLK_W: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
const TXU_W: [usize; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
const TXU_H: [usize; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];
const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
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
    fn urange(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
    fn cost(&mut self) -> i32 {
        self.range(0, 20 << 9)
    }
    fn qm(&mut self) -> u8 {
        self.range(1, 256) as u8
    }
}

const REGIONS: [(usize, usize, usize); 13] = [
    (0, 5 * 13, 2),
    (195, 4, 5),
    (219, 4, 6),
    (247, 4, 7),
    (279, 4, 8),
    (315, 4, 9),
    (355, 4, 10),
    (399, 4, 11),
    (447, 5 * 2 * 9, 2),
    (717, 5 * 2 * 4, 3),
    (877, 5 * 2 * 42, 4),
    (2977, 5 * 2 * 21, 4),
    (4027, 2 * 3, 2),
];

fn gen_arena(rng: &mut Rng) -> Vec<u16> {
    let mut a = vec![0u16; CDF_ARENA_LEN];
    for &(off, count, n) in &REGIONS {
        for slot in 0..count {
            let base = off + slot * (n + 1);
            let mut acc: u32 = 0;
            for e in a[base..base + n - 1].iter_mut() {
                acc += rng.urange(1, (32000 / n as u32).max(2));
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            a[base + n - 1] = 0;
            a[base + n] = 0;
        }
    }
    a
}

fn gen_cdf(rng: &mut Rng, n: usize) -> Vec<u16> {
    let mut cdf = vec![0u16; 17.max(n + 1)];
    let mut acc = 0u32;
    for e in cdf.iter_mut().take(n.saturating_sub(1)) {
        acc += rng.urange(1, (32000 / n as u32).max(2));
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    cdf
}

// (plane_bsize, tx_size): uniform tx that tiles the block evenly.
const COMBOS: [(usize, usize); 7] = [
    (6, 1), // 16x16 / 8x8   -> 2x2
    (6, 0), // 16x16 / 4x4   -> 4x4
    (9, 2), // 32x32 / 16x16 -> 2x2
    (9, 1), // 32x32 / 8x8   -> 4x4
    (3, 0), // 8x8   / 4x4   -> 2x2
    (4, 1), // 16x8  / 8x8   -> 2x1
    (6, 6), // 16x16 / 8x4   -> 2x4
];

#[test]
fn encode_coding_block_plane_identical() {
    let mut rng = Rng(0x0c0d_1b10_c0de_0009);
    let (mut total, mut multi_txb) = (0usize, 0usize);
    for &(plane_bsize, tx_size) in &COMBOS {
        let uw = BLK_W[plane_bsize] >> 2;
        let uh = BLK_H[plane_bsize] >> 2;
        let (txw, txh) = (TXU_W[tx_size], TXU_H[tx_size]);
        let n_txb = (uw / txw) * (uh / txh);
        let full = TX_W[tx_size] * TX_H[tx_size];
        let n_coeffs = aom_txb::txb_wide(tx_size) * aom_txb::txb_high(tx_size);
        let ls = log_scale(tx_size);
        let tx_type = 0usize; // DCT_DCT: valid + in-set (tx_type write exercised)
        let sc = scan(tx_size, tx_type);
        for &plane in &[0usize, 1] {
            for iter in 0..24 {
                // One residual per txb (raster).
                let bd: u8 = if iter % 3 == 1 { 12 } else { 8 };
                let rmax = if bd > 8 { 4096 } else { 256 };
                let residuals: Vec<Vec<i16>> = (0..n_txb)
                    .map(|_| {
                        (0..full)
                            .map(|_| rng.range(-(rmax - 1), rmax) as i16)
                            .collect()
                    })
                    .collect();
                let res_refs: Vec<&[i16]> = residuals.iter().map(|v| v.as_slice()).collect();

                // Block-wide quant/cost/tx_type context (reciprocal tables).
                let dq = [rng.range(16, 800), rng.range(16, 800)];
                let dequant = [dq[0] as i16, dq[1] as i16];
                let quant = [
                    (65536 / dq[0]).clamp(1, 32767) as i16,
                    (65536 / dq[1]).clamp(1, 32767) as i16,
                ];
                let zbin = [rng.range(1, 100) as i16, rng.range(1, 100) as i16];
                let round = [rng.range(1, 400) as i16, rng.range(1, 400) as i16];
                let quant_shift = [rng.range(8000, 32767) as i16, rng.range(8000, 32767) as i16];
                let qm_v: Vec<u8> = (0..n_coeffs).map(|_| rng.qm()).collect();
                let iqm_v: Vec<u8> = qm_v
                    .iter()
                    .map(|&w| (1024 / w as u32).clamp(1, 255) as u8)
                    .collect();
                let use_qm = iter % 3 == 0;
                let (qm, iqm) = if use_qm {
                    (Some(&qm_v[..]), Some(&iqm_v[..]))
                } else {
                    (None, None)
                };
                let kind = match iter % 3 {
                    0 => QuantKind::Fp,
                    1 => QuantKind::B,
                    _ => QuantKind::Dc,
                };

                let cost_v =
                    |rng: &mut Rng, n: usize| -> Vec<i32> { (0..n).map(|_| rng.cost()).collect() };
                let txb_skip = cost_v(&mut rng, 13 * 2);
                let base_eob = cost_v(&mut rng, 4 * 3);
                let base = cost_v(&mut rng, 42 * 8);
                let eob_extra = cost_v(&mut rng, 9 * 2);
                let dc_sign = cost_v(&mut rng, 3 * 2);
                let lps = cost_v(&mut rng, 21 * 26);
                let eob_c = cost_v(&mut rng, 2 * 11);
                let rdmult = rng.range(1, 1 << 20) as i64;
                let sharpness = rng.range(0, 8);
                let upd = iter % 2 == 0;
                let ttx = TxTypeContext {
                    is_inter: iter % 2 == 0,
                    reduced: iter % 5 == 0,
                    use_filter_intra: false,
                    fi_mode: 0,
                    mode: [0usize, 1, 2, 6, 12][iter % 5],
                    signal_gate: true,
                };
                // The whole block shares one tx_type/context, so one ext-tx set num.
                let d = ext_tx_derive(
                    tx_size,
                    ttx.is_inter,
                    ttx.reduced,
                    tx_type,
                    false,
                    0,
                    ttx.mode,
                );

                let arena0 = gen_arena(&mut rng);
                let ext0 = gen_cdf(&mut rng, d.num.max(2) as usize);
                let cost = CoeffCostTables {
                    txb_skip: &txb_skip,
                    base_eob: &base_eob,
                    base: &base,
                    eob_extra: &eob_extra,
                    dc_sign: &dc_sign,
                    lps: &lps,
                    eob: &eob_c,
                };
                let qp = QuantParams {
                    zbin: &zbin,
                    round: &round,
                    quant: &quant,
                    quant_shift: &quant_shift,
                    dequant: &dequant,
                    qm,
                    iqm,
                    bd,
                    lossless: false,
                    qm_ctx: None,
                    adaptive: false,
                };
                let opt = OptimizeInputs {
                    cost: &cost,
                    rdmult,
                    sharpness,
                };

                // Rust: the whole plane in one call.
                let mut arena_r = arena0.clone();
                let mut ext_r = ext0.clone();
                let mut enc_r = OdEcEnc::new();
                let bc_r = encode_coding_block_plane(
                    &res_refs,
                    plane_bsize,
                    tx_size,
                    tx_type,
                    kind,
                    &qp,
                    &opt,
                    &ttx,
                    plane,
                    upd,
                    &mut enc_r,
                    &mut arena_r,
                    &mut ext_r,
                );
                let bytes_r = enc_r.done().to_vec();

                // Oracle: same loop, C context threading + C quant/optimize, Rust write.
                let plane_type = (plane > 0) as usize;
                let mut arena_c = arena0.clone();
                let mut ext_c = ext0.clone();
                let mut enc_c = OdEcEnc::new();
                let mut above = vec![0i8; uw];
                let mut left = vec![0i8; uh];
                let mut idx = 0;
                let mut blk_row = 0;
                while blk_row < uh {
                    let mut blk_col = 0;
                    while blk_col < uw {
                        let (skip_ctx, dc_sign_ctx) = c::ref_get_txb_ctx(
                            plane_bsize,
                            tx_size,
                            plane,
                            &above[blk_col..],
                            &left[blk_row..],
                        );
                        let coeff_c =
                            c::ref_fwd_txfm2d(tx_size, &residuals[idx], TX_W[tx_size], tx_type);
                        let src = &coeff_c[..n_coeffs];
                        let iscan = vec![0i16; n_coeffs];
                        let hbd = bd > 8;
                        let (mut qc, mut dqc, eob0) = match (kind, use_qm, hbd) {
                            (QuantKind::Fp, true, false) => c::ref_quantize_fp_qm(
                                ls, src, &round, &quant, &dequant, &qm_v, &iqm_v, sc, &iscan,
                            ),
                            (QuantKind::Fp, true, true) => c::ref_highbd_quantize_fp_qm(
                                ls, src, &round, &quant, &dequant, &qm_v, &iqm_v, sc, &iscan,
                            ),
                            (QuantKind::Fp, false, false) => {
                                c::ref_quantize_fp(ls, src, &round, &quant, &dequant, sc)
                            }
                            (QuantKind::Fp, false, true) => {
                                c::ref_highbd_quantize_fp(ls, src, &round, &quant, &dequant, sc)
                            }
                            (QuantKind::B, true, false) => c::ref_quantize_b_qm(
                                ls,
                                src,
                                &zbin,
                                &round,
                                &quant,
                                &quant_shift,
                                &dequant,
                                &qm_v,
                                &iqm_v,
                                sc,
                            ),
                            (QuantKind::B, true, true) => c::ref_highbd_quantize_b_qm(
                                ls,
                                src,
                                &zbin,
                                &round,
                                &quant,
                                &quant_shift,
                                &dequant,
                                &qm_v,
                                &iqm_v,
                                sc,
                            ),
                            (QuantKind::B, false, false) => c::ref_quantize_b(
                                ls,
                                src,
                                &zbin,
                                &round,
                                &quant,
                                &quant_shift,
                                &dequant,
                                sc,
                            ),
                            (QuantKind::B, false, true) => c::ref_highbd_quantize_b(
                                ls,
                                src,
                                &zbin,
                                &round,
                                &quant,
                                &quant_shift,
                                &dequant,
                                sc,
                            ),
                            (QuantKind::Dc, _, false) => {
                                c::ref_quantize_dc(ls, src, &round, quant[0], dequant[0], qm, iqm)
                            }
                            (QuantKind::Dc, _, true) => c::ref_highbd_quantize_dc(
                                ls, src, &round, quant[0], dequant[0], qm, iqm,
                            ),
                        };
                        let eob_w = if eob0 == 0 {
                            0
                        } else if use_qm {
                            c::ref_optimize_txb_qm(
                                tx_size,
                                tx_type,
                                &mut qc,
                                &mut dqc,
                                src,
                                eob0 as usize,
                                &dequant,
                                rdmult,
                                dc_sign_ctx as usize,
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
                                &iqm_v,
                                &qm_v,
                            )
                            .0
                        } else {
                            c::ref_optimize_txb(
                                tx_size,
                                tx_type,
                                &mut qc,
                                &mut dqc,
                                src,
                                eob0 as usize,
                                &dequant,
                                rdmult,
                                dc_sign_ctx as usize,
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
                            .0
                        };
                        write_coeffs_txb_full(
                            &mut enc_c,
                            &mut arena_c,
                            &mut ext_c,
                            &qc,
                            eob_w,
                            tx_size,
                            tx_type,
                            plane_type,
                            skip_ctx as usize,
                            dc_sign_ctx as usize,
                            upd,
                            ttx.is_inter,
                            ttx.reduced,
                            ttx.use_filter_intra,
                            ttx.fi_mode,
                            ttx.mode,
                            ttx.signal_gate,
                        );
                        let cul = c::ref_txb_entropy_context(&qc, tx_size, tx_type, eob_w) as i8;
                        above[blk_col..blk_col + txw].fill(cul);
                        left[blk_row..blk_row + txh].fill(cul);
                        idx += 1;
                        blk_col += txw;
                    }
                    blk_row += txh;
                }
                let bytes_c = enc_c.done().to_vec();

                let m = format!(
                    "pb={plane_bsize} ts={tx_size} pl={plane} kind={kind:?} bd={bd} n_txb={n_txb}"
                );
                assert_eq!(bytes_r, bytes_c, "bytes {m}");
                assert_eq!(bc_r, BlockContexts { above, left }, "final contexts {m}");
                if upd {
                    assert_eq!(arena_r, arena_c, "coeff cdf {m}");
                    assert_eq!(ext_r, ext_c, "ext_tx cdf {m}");
                }
                total += 1;
                if n_txb > 1 {
                    multi_txb += 1;
                }
            }
        }
    }
    assert!(
        multi_txb * 2 >= total,
        "too few multi-txb blocks: {multi_txb}/{total}"
    );
}
