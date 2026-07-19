//! Differential harness for aom_vector_var vs C libaom: variance of ref-src over
//! a 4<<bwl vector. Covers bwl 0..5 (widths 4..128) with full-swing i16 residual
//! inputs to exercise the unsigned mean_abs^2 arithmetic (which can reach ~2^32).

use aom_dsp::dist::vector_var;
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
    // 10-bit-ish residual magnitudes (the C notes diff in [-510, 510]); use the
    // full documented range so mean can reach the 16-bit / unsigned-square regime.
    fn resid(&mut self) -> i16 {
        (self.next() % 1021) as i16 - 510
    }
}

#[test]
fn vector_var_differential() {
    let mut rng = Rng(0x_ec70_9e37_79b9_7c15);
    for bwl in 0..=5i32 {
        let width = 4usize << bwl;
        for _ in 0..20000 {
            let reff: Vec<i16> = (0..width).map(|_| rng.resid()).collect();
            let src: Vec<i16> = (0..width).map(|_| rng.resid()).collect();
            assert_eq!(vector_var(&reff, &src, bwl), c::ref_vector_var(&reff, &src, bwl), "bwl={bwl}");
        }
    }
}
