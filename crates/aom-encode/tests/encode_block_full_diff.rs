//! Capstone+ differential: a residual block -> COMPLETE txb bitstream (txb_skip ->
//! luma tx_type -> coefficients). `encode_block_coeffs_full` runs the full speed-0
//! path then `write_coeffs_txb_full`, and must produce byte-identical bytes AND
//! identical CDF adaptation (coeff arena + ext-tx slot) to the C reference steps
//! chained (fwd -> quant -> get_txb_ctx -> optimize -> ref_write_coeffs_txb_full).

use aom_encode::{
    BlockContext, OptimizeInputs, QuantKind, QuantParams, TxTypeContext, encode_block_coeffs_full,
};
use aom_dsp::entropy::enc::OdEcEnc;
use aom_sys_ref as c;
use aom_dsp::transform::txfm2d::fwd_txfm_valid;
use aom_dsp::txb::{CDF_ARENA_LEN, CoeffCostTables, ext_tx_derive, scan, txb_high, txb_wide};

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
const TX_SIZE_2D: [i32; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];
const PLANE_BSIZES: [usize; 8] = [0, 3, 6, 9, 12, 4, 7, 10];
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

fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

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

#[test]
fn encode_block_coeffs_full_identical() {
    let mut rng = Rng(0x0b17_c0de_f011_0009);
    const TX_TYPES: [usize; 7] = [0, 1, 2, 3, 9, 10, 11];
    let (mut total, mut nonzero_eob, mut tx_written) = (0usize, 0usize, 0usize);
    for tx_size in 0..19usize {
        let full = TX_W[tx_size] * TX_H[tx_size];
        let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
        let ls = log_scale(tx_size);
        for &tx_type in &TX_TYPES {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for iter in 0..40 {
                let bd: u8 = if iter % 3 == 1 { 12 } else { 8 };
                let rmax = if bd > 8 { 4096 } else { 256 };
                let residual: Vec<i16> = (0..full)
                    .map(|_| rng.range(-(rmax - 1), rmax) as i16)
                    .collect();
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

                let txb_skip = tbl(&mut rng, 13 * 2);
                let base_eob = tbl(&mut rng, 4 * 3);
                let base = tbl(&mut rng, 42 * 8);
                let eob_extra = tbl(&mut rng, 9 * 2);
                let dc_sign = tbl(&mut rng, 3 * 2);
                let lps = tbl(&mut rng, 21 * 26);
                let eob_c = tbl(&mut rng, 2 * 11);
                let mk = |rng: &mut Rng| -> Vec<i8> {
                    (0..16)
                        .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                        .collect()
                };
                let above = mk(&mut rng);
                let left = mk(&mut rng);
                let plane = rng.range(0, 3) as usize;
                let plane_type = (plane > 0) as usize;
                let plane_bsize = PLANE_BSIZES[rng.range(0, 8) as usize];
                let rdmult = rng.range(1, 1 << 20) as i64;
                let sharpness = rng.range(0, 8);
                let upd = iter % 2 == 0;

                // tx_type signaling context.
                let is_inter = iter % 2 == 0;
                let reduced = iter % 3 == 0;
                let use_fi = iter % 4 == 0;
                let fi_mode = iter % 5;
                let mode = [0usize, 1, 2, 6, 12][iter % 5];
                let signal_gate = iter % 3 != 2;
                let d = ext_tx_derive(tx_size, is_inter, reduced, tx_type, use_fi, fi_mode, mode);
                // Skip out-of-set luma combos that would index av1_ext_tx_ind OOB.
                if plane_type == 0 && signal_gate && d.num > 1 && d.used == 0 {
                    continue;
                }

                let use_qm = iter % 3 == 0;
                let kind = match iter % 5 {
                    0 | 1 => QuantKind::Fp,
                    2 | 3 => QuantKind::B,
                    _ => QuantKind::Dc,
                };
                let (qm, iqm) = if use_qm {
                    (Some(&qm_v[..]), Some(&iqm_v[..]))
                } else {
                    (None, None)
                };

                let arena0 = gen_arena(&mut rng);
                let ext0 = gen_cdf(&mut rng, d.num.max(2) as usize);
                let mut arena_r = arena0.clone();
                let mut arena_c = arena0.clone();
                let mut ext_r = ext0.clone();
                let mut ext_c = ext0.clone();

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
                let bctx = BlockContext {
                    above: &above,
                    left: &left,
                    plane,
                    plane_bsize,
                };
                let opt = OptimizeInputs {
                    cost: &cost,
                    rdmult,
                    sharpness,
                };
                let ttx = TxTypeContext {
                    is_inter,
                    reduced,
                    use_filter_intra: use_fi,
                    fi_mode,
                    mode,
                    signal_gate,
                };

                let mut enc = OdEcEnc::new();
                let r = encode_block_coeffs_full(
                    &residual,
                    tx_size,
                    tx_type,
                    kind,
                    &qp,
                    &bctx,
                    &opt,
                    &ttx,
                    upd,
                    &mut enc,
                    &mut arena_r,
                    &mut ext_r,
                );
                let got = enc.done().to_vec();

                // Oracle chain.
                let coeff_c = c::ref_fwd_txfm2d(tx_size, &residual, TX_W[tx_size], tx_type);
                let src = &coeff_c[..n_coeffs];
                let sc = scan(tx_size, tx_type);
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
                    (QuantKind::Dc, _, true) => {
                        c::ref_highbd_quantize_dc(ls, src, &round, quant[0], dequant[0], qm, iqm)
                    }
                };
                let (skip_ctx, dc_sign_ctx) =
                    c::ref_get_txb_ctx(plane_bsize, tx_size, plane, &above, &left);
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
                let want = c::ref_write_coeffs_txb_full(
                    &qc,
                    eob_w,
                    tx_size,
                    tx_type,
                    plane_type,
                    skip_ctx as usize,
                    dc_sign_ctx as usize,
                    upd,
                    &mut arena_c,
                    &mut ext_c,
                    is_inter,
                    reduced,
                    use_fi,
                    fi_mode,
                    mode,
                    signal_gate,
                );

                let m = format!(
                    "ts={tx_size} tt={tx_type} kind={kind:?} bd={bd} pl={plane_type} gate={signal_gate} upd={upd}"
                );
                assert_eq!(r.eob as usize, eob_w, "eob {m}");
                assert_eq!(r.qcoeff, qc, "qcoeff {m}");
                assert_eq!(got, want, "bytes {m}");
                if upd {
                    assert_eq!(arena_r, arena_c, "coeff cdf {m}");
                    assert_eq!(ext_r, ext_c, "ext_tx cdf {m}");
                }
                total += 1;
                nonzero_eob += (eob_w > 0) as usize;
                if plane_type == 0 && signal_gate && d.num > 1 && eob_w > 0 {
                    tx_written += 1;
                }
            }
        }
    }
    assert!(
        nonzero_eob * 4 >= total,
        "too few nonzero-eob blocks: {nonzero_eob}/{total}"
    );
    assert!(
        tx_written > 0,
        "tx_type symbol never exercised in the full encode path"
    );
}
