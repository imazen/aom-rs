//! Differential harness for the highbd (10/12-bit) quantizers vs C libaom:
//! av1_highbd_quantize_fp + aom_highbd_quantize_b (no quant matrix), log_scale
//! 0/1/2. Wider coefficient range than lowbd (12-bit transform magnitudes).

use aom_dsp::quant::{aom_highbd_quantize_b_no_qmatrix, av1_highbd_quantize_fp_no_qmatrix};
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
    fn coeff(&mut self) -> i32 {
        // 12-bit-depth transform magnitudes can exceed the 8-bit range.
        (self.next() % (1 << 23)) as i32 - (1 << 22)
    }
    fn i16r(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
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
fn highbd_quantize_fp_differential() {
    let mut rng = Rng(0x_86bd_9e37_79b9_7c15);
    for &n in &[16usize, 64, 256, 1024] {
        for log_scale in 0..=2 {
            for _ in 0..2000 {
                let scan = perm(&mut rng, n);
                let coeff: Vec<i32> = (0..n).map(|_| rng.coeff()).collect();
                let round = [rng.i16r(1, 2000), rng.i16r(1, 2000)];
                let quant = [rng.i16r(1, 32767), rng.i16r(1, 32767)];
                let dequant = [rng.i16r(1, 8000), rng.i16r(1, 8000)];
                let mut q = vec![0i32; n];
                let mut dq = vec![0i32; n];
                let eob = av1_highbd_quantize_fp_no_qmatrix(&quant, &dequant, &round, log_scale, &scan, &coeff, &mut q, &mut dq);
                let (qw, dqw, ew) = c::ref_highbd_quantize_fp(log_scale, &coeff, &round, &quant, &dequant, &scan);
                assert_eq!(eob, ew, "hbd fp eob n={n} ls={log_scale}");
                assert_eq!(q, qw, "hbd fp qcoeff n={n} ls={log_scale}");
                assert_eq!(dq, dqw, "hbd fp dqcoeff n={n} ls={log_scale}");
            }
        }
    }
}

#[test]
fn highbd_quantize_b_differential() {
    let mut rng = Rng(0x_86bd_c057_0000_b111);
    for &n in &[16usize, 64, 256, 1024] {
        for log_scale in 0..=2 {
            for _ in 0..2000 {
                let scan = perm(&mut rng, n);
                let coeff: Vec<i32> = (0..n).map(|_| rng.coeff()).collect();
                let zbin = [rng.i16r(1, 1500), rng.i16r(1, 1500)];
                let round = [rng.i16r(1, 2000), rng.i16r(1, 2000)];
                let quant = [rng.i16r(1, 32767), rng.i16r(1, 32767)];
                let quant_shift = [rng.i16r(1, 32767), rng.i16r(1, 32767)];
                let dequant = [rng.i16r(1, 8000), rng.i16r(1, 8000)];
                let mut q = vec![0i32; n];
                let mut dq = vec![0i32; n];
                let eob = aom_highbd_quantize_b_no_qmatrix(&zbin, &round, &quant, &quant_shift, &dequant, log_scale, &scan, &coeff, &mut q, &mut dq);
                let (qw, dqw, ew) = c::ref_highbd_quantize_b(log_scale, &coeff, &zbin, &round, &quant, &quant_shift, &dequant, &scan);
                assert_eq!(eob, ew, "hbd b eob n={n} ls={log_scale}");
                assert_eq!(q, qw, "hbd b qcoeff n={n} ls={log_scale}");
                assert_eq!(dq, dqw, "hbd b dqcoeff n={n} ls={log_scale}");
            }
        }
    }
}
