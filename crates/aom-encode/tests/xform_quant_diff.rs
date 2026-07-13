//! End-to-end differential for `av1_xform_quant` (bd 8 + highbd 12) vs C libaom: a random
//! residual block, run through the Rust composition (av1_fwd_txfm2d -> quantize
//! -> txb_entropy_context), must produce byte-identical coeff / qcoeff / dqcoeff
//! / eob / txb_entropy_ctx to the same three C reference steps chained. This
//! locks the block-encode wiring (coeff layout, scan selection, log_scale,
//! quantizer dispatch, entropy-ctx deferral) on top of the already bit-exact
//! sub-modules.

use aom_encode::{xform_quant, QuantKind, QuantParams};
use aom_sys_ref as c;
use aom_transform::txfm2d::fwd_txfm_valid;
use aom_txb::{txb_high, txb_wide};

const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
const TX_SIZE_2D: [i32; 19] =
    [16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024];

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
    fn qm(&mut self) -> u8 {
        self.range(1, 256) as u8
    }
}

#[test]
fn xform_quant_end_to_end_identical() {
    let mut rng = Rng(0x0a0e_c0de_7fb1_9e37);
    const TX_TYPES: [usize; 7] = [0, 1, 2, 3, 9, 10, 11];
    // Coverage guards: the test must actually exercise nonzero-eob blocks (else
    // it would trivially pass on all-zero output) and both entropy-ctx paths.
    let mut nonzero_eob = 0usize;
    let mut nonzero_ctx = 0usize;
    let mut total = 0usize;
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
                // quantizers). The forward transform is bd-independent.
                let bd: u8 = if iter % 3 == 2 { 12 } else { 8 };
                let rmax = if bd > 8 { 4096 } else { 256 };
                let residual: Vec<i16> = (0..full).map(|_| rng.range(-(rmax - 1), rmax) as i16).collect();
                // Quant tables. Small dequant + large quant => plenty of nonzero
                // coefficients survive so the eob / entropy-ctx paths are exercised.
                let zbin = [rng.range(1, 400) as i16, rng.range(1, 400) as i16];
                let round = [rng.range(1, 800) as i16, rng.range(1, 800) as i16];
                let quant = [rng.range(8000, 32767) as i16, rng.range(8000, 32767) as i16];
                let quant_shift = [rng.range(8000, 32767) as i16, rng.range(8000, 32767) as i16];
                let dequant = [rng.range(4, 500) as i16, rng.range(4, 500) as i16];
                let qm_v: Vec<u8> = (0..n_coeffs).map(|_| rng.qm()).collect();
                let iqm_v: Vec<u8> = (0..n_coeffs).map(|_| rng.qm()).collect();

                let use_qm = iter % 2 == 0;
                let kind = if iter % 4 < 2 { QuantKind::Fp } else { QuantKind::B };
                let use_optimize_b = iter % 3 == 0;
                let (qm, iqm) = if use_qm { (Some(&qm_v[..]), Some(&iqm_v[..])) } else { (None, None) };

                let qp = QuantParams {
                    zbin: &zbin,
                    round: &round,
                    quant: &quant,
                    quant_shift: &quant_shift,
                    dequant: &dequant,
                    qm,
                    iqm,
                    bd,
                };
                let got = xform_quant(&residual, tx_size, tx_type, kind, &qp, use_optimize_b);

                // Oracle: the same three C steps chained (highbd quantizer at bd>8).
                let coeff_c = c::ref_fwd_txfm2d(tx_size, &residual, TX_W[tx_size], tx_type);
                let src = &coeff_c[..n_coeffs];
                let iscan: Vec<i16> = vec![0; n_coeffs];
                let hbd = bd > 8;
                let (qc, dqc, eob) = match (kind, use_qm, hbd) {
                    (QuantKind::Fp, true, false) => c::ref_quantize_fp_qm(
                        ls, src, &round, &quant, &dequant, &qm_v, &iqm_v, sc(tx_size, tx_type), &iscan,
                    ),
                    (QuantKind::Fp, true, true) => c::ref_highbd_quantize_fp_qm(
                        ls, src, &round, &quant, &dequant, &qm_v, &iqm_v, sc(tx_size, tx_type), &iscan,
                    ),
                    (QuantKind::Fp, false, false) => c::ref_quantize_fp(ls, src, &round, &quant, &dequant, sc(tx_size, tx_type)),
                    (QuantKind::Fp, false, true) => c::ref_highbd_quantize_fp(ls, src, &round, &quant, &dequant, sc(tx_size, tx_type)),
                    (QuantKind::B, true, false) => c::ref_quantize_b_qm(
                        ls, src, &zbin, &round, &quant, &quant_shift, &dequant, &qm_v, &iqm_v, sc(tx_size, tx_type),
                    ),
                    (QuantKind::B, true, true) => c::ref_highbd_quantize_b_qm(
                        ls, src, &zbin, &round, &quant, &quant_shift, &dequant, &qm_v, &iqm_v, sc(tx_size, tx_type),
                    ),
                    (QuantKind::B, false, false) => c::ref_quantize_b(
                        ls, src, &zbin, &round, &quant, &quant_shift, &dequant, sc(tx_size, tx_type),
                    ),
                    (QuantKind::B, false, true) => c::ref_highbd_quantize_b(
                        ls, src, &zbin, &round, &quant, &quant_shift, &dequant, sc(tx_size, tx_type),
                    ),
                };
                let ctx_c = if use_optimize_b { 0 } else { c::ref_txb_entropy_context(&qc, tx_size, tx_type, eob as usize) };

                let m = format!("ts={tx_size} tt={tx_type} kind={kind:?} qm={use_qm} bd={bd} optb={use_optimize_b}");
                assert_eq!(&got.coeff[..n_coeffs], src, "coeff {m}");
                assert_eq!(got.qcoeff, qc, "qcoeff {m}");
                assert_eq!(got.dqcoeff, dqc, "dqcoeff {m}");
                assert_eq!(got.eob, eob, "eob {m}");
                assert_eq!(got.txb_entropy_ctx, ctx_c, "txb_entropy_ctx {m}");

                total += 1;
                nonzero_eob += (got.eob > 0) as usize;
                nonzero_ctx += (got.txb_entropy_ctx != 0) as usize;
            }
        }
    }
    // A large fraction of blocks should quantize to a nonzero eob, and the
    // non-deferred path should produce nonzero contexts — otherwise the equality
    // assertions above are vacuous.
    assert!(nonzero_eob * 4 >= total, "too few nonzero-eob blocks: {nonzero_eob}/{total}");
    assert!(nonzero_ctx > 0, "no nonzero txb_entropy_ctx observed ({total} blocks)");
}

/// Scan order slice for the oracle quantizer calls.
fn sc(tx_size: usize, tx_type: usize) -> &'static [i16] {
    aom_txb::scan(tx_size, tx_type)
}
