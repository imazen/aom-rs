//! Capstone differential: a residual block -> entropy-coded coefficient bytes.
//! `encode_block_coeffs` runs the full speed-0 path (av1_fwd_txfm2d -> quantize ->
//! get_txb_ctx -> av1_optimize_b -> av1_write_coeffs_txb) and must produce byte-
//! identical bitstream output AND identical CDF adaptation to the same six C
//! reference steps chained. This is "residual -> real encoder output" end to end.
//! `av1_write_tx_type` (plane-0 tx_type) is out of scope on both sides.

use aom_encode::{BlockContext, OptimizeInputs, QuantKind, QuantParams, encode_block_coeffs};
use aom_entropy::enc::OdEcEnc;
use aom_sys_ref as c;
use aom_transform::txfm2d::fwd_txfm_valid;
use aom_txb::{CDF_ARENA_LEN, CoeffCostTables, scan, txb_high, txb_wide};

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

// CDF arena region layout (offset, slot_count, nsymbs), mirroring write.rs.
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

#[test]
fn encode_block_coeffs_end_to_end_identical() {
    let mut rng = Rng(0xb17e_c0de_5721_0009);
    const TX_TYPES: [usize; 7] = [0, 1, 2, 3, 9, 10, 11];
    let (mut total, mut nonzero_eob, mut byte_producing) = (0usize, 0usize, 0usize);
    for tx_size in 0..19usize {
        let full = TX_W[tx_size] * TX_H[tx_size];
        let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
        let ls = log_scale(tx_size);
        for &tx_type in &TX_TYPES {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for iter in 0..48 {
                // bd 8 (residual +-255) vs highbd 12 (residual +-4095, 64-bit
                // quantizers). Fwd transform / trellis / entropy ctx / writer are
                // all bd-independent, so highbd differs only in the quantizer.
                let bd: u8 = if iter % 3 == 1 { 12 } else { 8 };
                let rmax = if bd > 8 { 4096 } else { 256 };
                let residual: Vec<i16> = (0..full)
                    .map(|_| rng.range(-(rmax - 1), rmax) as i16)
                    .collect();
                // Reciprocal quant/dequant + reciprocal qm/iqm (real-encoder-like),
                // keeping the QM trellis' squared get_coeff_dist inside i64.
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
                let mut arena_c = arena0.clone();
                let mut arena_r = arena0.clone();

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

                // Rust: full residual -> bytes.
                let mut enc = OdEcEnc::new();
                let r = encode_block_coeffs(
                    &residual,
                    tx_size,
                    tx_type,
                    kind,
                    &qp,
                    &bctx,
                    &opt,
                    upd,
                    &mut enc,
                    &mut arena_r,
                );
                let got = enc.done().to_vec();

                // Oracle: fwd -> quant -> get_txb_ctx -> optimize -> write_coeffs.
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
                let want = c::ref_write_coeffs_txb(
                    &qc,
                    eob_w,
                    tx_size,
                    tx_type,
                    plane_type,
                    skip_ctx as usize,
                    dc_sign_ctx as usize,
                    upd,
                    &mut arena_c,
                );

                let m = format!(
                    "ts={tx_size} tt={tx_type} kind={kind:?} qm={use_qm} bd={bd} upd={upd} eob0={eob0}"
                );
                assert_eq!(r.eob as usize, eob_w, "opt eob {m}");
                assert_eq!(r.qcoeff, qc, "qcoeff {m}");
                assert_eq!(got, want, "bytes {m}");
                if upd {
                    assert_eq!(arena_r, arena_c, "cdf adaptation {m}");
                }

                total += 1;
                nonzero_eob += (eob_w > 0) as usize;
                byte_producing += (!got.is_empty()) as usize;
            }
        }
    }
    // The bitstream path must actually be exercised (nonzero eobs + real bytes).
    assert!(
        nonzero_eob * 4 >= total,
        "too few nonzero-eob blocks: {nonzero_eob}/{total}"
    );
    assert!(
        byte_producing * 2 >= total,
        "too few byte-producing blocks: {byte_producing}/{total}"
    );
}
