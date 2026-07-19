//! Integration harness for `reconstruct_txb` — the residual half of per-block
//! decode reconstruction (dequant → inverse transform → add-onto-prediction).
//! For realistic coefficients produced by the forward+quantize path, the
//! reconstructed pixels must match, byte for byte, C libaom v3.14.1's
//! `ref_dequant_txb` + `ref_inv_txfm2d_add` composed identically — proving the
//! wiring (stride, tx_type/tx_size threading, layout) is correct. Swept over all
//! valid transform sizes × types × bitdepths {8,10,12}.

use aom_encode::{QuantKind, QuantParams, reconstruct_txb, xform_quant};
use aom_sys_ref as c;
use aom_dsp::transform::inv_txfm2d::inv_txfm_valid;

/// Full (un-repacked) transform dims — the residual/prediction buffer size.
const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
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
    fn pixel(&mut self, bd: i32) -> u16 {
        (self.next() % (1u64 << bd)) as u16
    }
}

/// libaom `invert_quant` (av1/encoder/av1_quantize.c): the (quant, shift)
/// fixed-point pair that inverts dequant step `d`, so the quantizer produces
/// realistic qcoeff (and, importantly, realistic sparsity/eob structure).
fn invert_quant(d: i32) -> (i16, i16) {
    let l = 31 - (d as u32).leading_zeros() as i32; // get_msb
    let m = 1 + (1i64 << (16 + l)) / d as i64;
    ((m - (1 << 16)) as i16, (1i32 << (16 - l)) as i16)
}

fn check(rng: &mut Rng, tx_size: usize, tx_type: usize, bd: i32) {
    let (w, h) = (TX_W[tx_size], TX_H[tx_size]);
    let full = w * h;
    // Realistic prediction + source; residual = source − prediction.
    let pred: Vec<u16> = (0..full).map(|_| rng.pixel(bd)).collect();
    let residual: Vec<i16> = (0..full)
        .map(|i| rng.pixel(bd) as i16 - pred[i] as i16)
        .collect();

    // Realistic quantizer params derived from a chosen dequant step.
    let dq = [rng.range(4, 800) as i16, rng.range(4, 800) as i16];
    let (q0, s0) = invert_quant(dq[0] as i32);
    let (q1, s1) = invert_quant(dq[1] as i32);
    let quant = [q0, q1];
    let quant_shift = [s0, s1];
    let round = [dq[0] / 8 + 1, dq[1] / 8 + 1];
    let zbin = [dq[0] / 2 + 1, dq[1] / 2 + 1];
    let qp = QuantParams {
        zbin: &zbin,
        round: &round,
        quant: &quant,
        quant_shift: &quant_shift,
        dequant: &dq,
        qm: None,
        iqm: None,
        bd: bd as u8,
        lossless: false,
        qm_ctx: None,
        adaptive: false,
    };

    let r = xform_quant(&residual, tx_size, tx_type, QuantKind::B, &qp, false);

    // My reconstruction: prediction + dequant + inverse transform.
    let mut got = pred.clone();
    reconstruct_txb(&mut got, w, tx_size, tx_type, &r.qcoeff, dq, None, bd);

    // C oracle: the same two bit-exact kernels composed identically.
    let dq_c = c::ref_dequant_txb(&r.qcoeff, tx_size, dq, None, bd);
    let mut want = pred.clone();
    c::ref_inv_txfm2d_add(tx_size, &dq_c, &mut want, w, tx_type, bd);

    assert_eq!(
        got, want,
        "reconstruct_txb divergence ts={tx_size} ({w}x{h}) tt={tx_type} bd={bd} eob={}",
        r.eob
    );
}

#[test]
fn reconstruct_txb_matches_c_composition() {
    let mut rng = Rng(0x_5eed_1234_c0de_babe);
    let mut nonzero_seen = false;
    for tx_size in 0..19usize {
        for tx_type in 0..16usize {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for &bd in &[8i32, 10, 12] {
                for _ in 0..60 {
                    check(&mut rng, tx_size, tx_type, bd);
                }
            }
        }
    }
    // Sanity: at least one reconstruction actually changed pixels (eob>0 path).
    let (w, h) = (TX_W[2], TX_H[2]);
    let pred = vec![100u16; w * h];
    let residual: Vec<i16> = (0..w * h).map(|i| (i as i16 % 17) - 8).collect();
    let dq = [40i16, 40];
    let (q0, s0) = invert_quant(40);
    let quant = [q0, q0];
    let quant_shift = [s0, s0];
    let round = [6i16, 6];
    let zbin = [10i16, 10];
    let qp = QuantParams {
        zbin: &zbin,
        round: &round,
        quant: &quant,
        quant_shift: &quant_shift,
        dequant: &dq,
        qm: None,
        iqm: None,
        bd: 8,
        lossless: false,
        qm_ctx: None,
        adaptive: false,
    };
    let r = xform_quant(&residual, 2, 0, QuantKind::B, &qp, false);
    let mut got = pred.clone();
    reconstruct_txb(&mut got, w, 2, 0, &r.qcoeff, dq, None, 8);
    nonzero_seen |= got != pred;
    assert!(
        nonzero_seen,
        "reconstruction never altered the prediction — test is vacuous"
    );
}
