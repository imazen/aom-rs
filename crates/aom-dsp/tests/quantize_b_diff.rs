//! Differential harness for `aom_quantize_b` (no quant matrix) vs C libaom
//! v3.14.1, across log_scale in {0,1,2}. Checks qcoeff, dqcoeff, eob.

use aom_sys_ref as c;
use aom_dsp::quant::aom_quantize_b_no_qmatrix;

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
        (self.next() % (1 << 19)) as i32 - (1 << 18)
    }
    fn pos_i16(&mut self, lo: i32, hi: i32) -> i16 {
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

fn check(rng: &mut Rng, log_scale: i32, n: usize, scan: &[i16]) {
    let coeff: Vec<i32> = (0..n).map(|_| rng.coeff()).collect();
    let zbin = [rng.pos_i16(1, 1000), rng.pos_i16(1, 1000)];
    let round = [rng.pos_i16(1, 2000), rng.pos_i16(1, 2000)];
    let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
    let quant_shift = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
    let dequant = [rng.pos_i16(1, 8000), rng.pos_i16(1, 8000)];

    let mut q_got = vec![0i32; n];
    let mut dq_got = vec![0i32; n];
    let eob_got = aom_quantize_b_no_qmatrix(
        &zbin, &round, &quant, &quant_shift, &dequant, log_scale, scan, &coeff, &mut q_got, &mut dq_got,
    );

    let (q_want, dq_want, eob_want) =
        c::ref_quantize_b(log_scale, &coeff, &zbin, &round, &quant, &quant_shift, &dequant, scan);

    assert_eq!(eob_got, eob_want, "eob mismatch log_scale={log_scale} n={n}");
    assert_eq!(q_got, q_want, "qcoeff mismatch log_scale={log_scale} n={n}\ncoeff={coeff:?}");
    assert_eq!(dq_got, dq_want, "dqcoeff mismatch log_scale={log_scale} n={n}");
}

#[test]
fn quantize_b_differential_fuzz() {
    let mut rng = Rng(0x_d1ce_f00d_a5a5_1234);
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
