//! Differential harness for the quant-matrix ("QM") quantizer paths vs C libaom:
//! `aom_quantize_b_helper_c` (lowbd) and `aom_highbd_quantize_b_helper_c` (highbd)
//! with non-NULL `qm`/`iqm`. Both sides get identical, per-position weight tables
//! spanning the full `qm_val_t` range (1..=255), so the 32-bit `coeff*wt` /
//! `dequant*iwt` products and the wrapping `tmp32*dequant` step are exercised.

use aom_quant::{aom_highbd_quantize_b_qm, aom_quantize_b_qm};
use aom_sys_ref as c;

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
    fn i16r(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
    }
    /// qm_val_t is uint8_t; real matrices centre near 32 (1<<AOM_QM_BITS) but the
    /// differential only needs identical tables — sweep the whole valid range.
    fn qm(&mut self) -> u8 {
        1 + (self.next() % 255) as u8
    }
}

fn perm(rng: &mut Rng, n: usize) -> Vec<i16> {
    let mut v: Vec<i16> = (0..n as i16).collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

#[test]
fn quantize_b_qm_differential() {
    let mut rng = Rng(0x_5a1e_c0de_9e37_79b9);
    for &n in &[16usize, 64, 256, 1024] {
        for log_scale in 0..=2 {
            for _ in 0..2000 {
                let scan = perm(&mut rng, n);
                // Lowbd transform magnitudes (8-bit depth): ±(1<<18).
                let coeff: Vec<i32> = (0..n).map(|_| (rng.next() % (1 << 19)) as i32 - (1 << 18)).collect();
                let qm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let iqm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let zbin = [rng.i16r(1, 1500), rng.i16r(1, 1500)];
                let round = [rng.i16r(1, 2000), rng.i16r(1, 2000)];
                let quant = [rng.i16r(1, 32767), rng.i16r(1, 32767)];
                let quant_shift = [rng.i16r(1, 32767), rng.i16r(1, 32767)];
                let dequant = [rng.i16r(1, 8000), rng.i16r(1, 8000)];
                let mut q = vec![0i32; n];
                let mut dq = vec![0i32; n];
                let eob = aom_quantize_b_qm(
                    &zbin, &round, &quant, &quant_shift, &dequant, log_scale, &qm, &iqm, &scan,
                    &coeff, &mut q, &mut dq,
                );
                let (qw, dqw, ew) = c::ref_quantize_b_qm(
                    log_scale, &coeff, &zbin, &round, &quant, &quant_shift, &dequant, &qm, &iqm,
                    &scan,
                );
                assert_eq!(eob, ew, "b-qm eob n={n} ls={log_scale}");
                assert_eq!(q, qw, "b-qm qcoeff n={n} ls={log_scale}");
                assert_eq!(dq, dqw, "b-qm dqcoeff n={n} ls={log_scale}");
            }
        }
    }
}

#[test]
fn highbd_quantize_b_qm_differential() {
    let mut rng = Rng(0x_5a1e_c0de_c057_0b11);
    for &n in &[16usize, 64, 256, 1024] {
        for log_scale in 0..=2 {
            for _ in 0..2000 {
                let scan = perm(&mut rng, n);
                // Highbd (10/12-bit) transform magnitudes: ±(1<<22). coeff*wt stays
                // < 2^31 for wt <= 255.
                let coeff: Vec<i32> = (0..n).map(|_| (rng.next() % (1 << 23)) as i32 - (1 << 22)).collect();
                let qm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let iqm: Vec<u8> = (0..n).map(|_| rng.qm()).collect();
                let zbin = [rng.i16r(1, 1500), rng.i16r(1, 1500)];
                let round = [rng.i16r(1, 2000), rng.i16r(1, 2000)];
                let quant = [rng.i16r(1, 32767), rng.i16r(1, 32767)];
                let quant_shift = [rng.i16r(1, 32767), rng.i16r(1, 32767)];
                let dequant = [rng.i16r(1, 8000), rng.i16r(1, 8000)];
                let mut q = vec![0i32; n];
                let mut dq = vec![0i32; n];
                let eob = aom_highbd_quantize_b_qm(
                    &zbin, &round, &quant, &quant_shift, &dequant, log_scale, &qm, &iqm, &scan,
                    &coeff, &mut q, &mut dq,
                );
                let (qw, dqw, ew) = c::ref_highbd_quantize_b_qm(
                    log_scale, &coeff, &zbin, &round, &quant, &quant_shift, &dequant, &qm, &iqm,
                    &scan,
                );
                assert_eq!(eob, ew, "hbd b-qm eob n={n} ls={log_scale}");
                assert_eq!(q, qw, "hbd b-qm qcoeff n={n} ls={log_scale}");
                assert_eq!(dq, dqw, "hbd b-qm dqcoeff n={n} ls={log_scale}");
            }
        }
    }
}
