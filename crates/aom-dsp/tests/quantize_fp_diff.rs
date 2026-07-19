//! Differential harness for the `av1_quantize_fp` family vs C libaom v3.14.1.
//! For log_scale in {0,1,2} and a range of coefficient counts, feed identical
//! coeffs + quant params + scan to both; assert qcoeff, dqcoeff, and eob are
//! byte-identical.

use aom_sys_ref as c;
use aom_dsp::quant::av1_quantize_fp_no_qmatrix;

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
        // transform-output magnitude; clamp inside the quantizer bounds C.
        (self.next() % (1 << 19)) as i32 - (1 << 18)
    }
    fn pos_i16(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
    }
}

/// Random permutation of 0..n (a valid scan order).
fn perm(rng: &mut Rng, n: usize) -> Vec<i16> {
    let mut v: Vec<i16> = (0..n as i16).collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn check(rng: &mut Rng, log_scale: i32, n: usize, scan: &[i16]) {
    let coeff: Vec<i32> = (0..n).map(|_| rng.coeff()).collect();
    let round = [rng.pos_i16(1, 2000), rng.pos_i16(1, 2000)];
    let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
    let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];

    let mut q_got = vec![0i32; n];
    let mut dq_got = vec![0i32; n];
    let eob_got =
        av1_quantize_fp_no_qmatrix(&quant, &dequant, &round, log_scale, scan, &coeff, &mut q_got, &mut dq_got);

    let (q_want, dq_want, eob_want) = c::ref_quantize_fp(log_scale, &coeff, &round, &quant, &dequant, scan);

    assert_eq!(eob_got, eob_want, "eob mismatch log_scale={log_scale} n={n}");
    assert_eq!(q_got, q_want, "qcoeff mismatch log_scale={log_scale} n={n}\ncoeff={coeff:?}");
    assert_eq!(dq_got, dq_want, "dqcoeff mismatch log_scale={log_scale} n={n}");
}

#[test]
fn quantize_fp_differential_fuzz() {
    let mut rng = Rng(0x_9e37_79b9_7f4a_7c15);
    // coeff counts covering the tx areas that use each log_scale.
    let sizes = [16usize, 64, 256, 1024];
    for log_scale in 0..=2i32 {
        for &n in &sizes {
            for _ in 0..20_000 {
                let scan = perm(&mut rng, n);
                check(&mut rng, log_scale, n, &scan);
            }
        }
    }
}

#[test]
fn quantize_fp_edge_cases() {
    let mut rng = Rng(11);
    for log_scale in 0..=2i32 {
        let n = 64;
        let scan: Vec<i16> = (0..n as i16).collect();
        // all-zero coeffs -> eob 0, all outputs zero
        let coeff = vec![0i32; n];
        let round = [10i16, 10];
        let quant = [10000i16, 10000];
        let dequant = [100i16, 100];
        let mut q = vec![0i32; n];
        let mut dq = vec![0i32; n];
        let eob = av1_quantize_fp_no_qmatrix(&quant, &dequant, &round, log_scale, &scan, &coeff, &mut q, &mut dq);
        let (qw, dqw, eobw) = c::ref_quantize_fp(log_scale, &coeff, &round, &quant, &dequant, &scan);
        assert_eq!((eob, &q, &dq), (eobw, &qw, &dqw));
        // large saturated coeffs
        let coeff2: Vec<i32> = (0..n).map(|i| if i % 2 == 0 { 1 << 18 } else { -(1 << 18) }).collect();
        let mut q2 = vec![0i32; n];
        let mut dq2 = vec![0i32; n];
        let eob2 = av1_quantize_fp_no_qmatrix(&quant, &dequant, &round, log_scale, &scan, &coeff2, &mut q2, &mut dq2);
        let (qw2, dqw2, eobw2) = c::ref_quantize_fp(log_scale, &coeff2, &round, &quant, &dequant, &scan);
        assert_eq!((eob2, &q2, &dq2), (eobw2, &qw2, &dqw2));
        let _ = &mut rng;
    }
}
