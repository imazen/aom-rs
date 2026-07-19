//! Differential harness for the residual-energy kernels (aom_sum_squares_i16 +
//! aom_sum_squares_2d_i16) vs C libaom: ss = sum(v*v) over i16 values, 1-D and
//! 2-D-strided. i16 magnitudes cover the residual range (up to 12-bit).

use aom_dsp::dist::{sum_squares_2d_i16, sum_squares_i16};
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
    fn v16(&mut self) -> i16 {
        (self.next() % 65536) as i16 // full i16 (wraps); covers residual magnitudes
    }
    fn range(&mut self, hi: u32) -> u32 {
        (self.next() % hi as u64) as u32
    }
}

#[test]
fn sum_squares_i16_differential() {
    let mut rng = Rng(0x0055_c0de_9e37_79b9);
    for &n in &[1usize, 16, 64, 256, 1024, 4096] {
        for _ in 0..2000 {
            let src: Vec<i16> = (0..n).map(|_| rng.v16()).collect();
            assert_eq!(sum_squares_i16(&src), c::ref_sum_squares_i16(&src), "1d n={n}");
        }
    }
}

#[test]
fn sum_squares_2d_i16_differential() {
    let mut rng = Rng(0x0055_c057_0000_b111);
    const DIMS: [(usize, usize); 8] =
        [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (4, 16), (16, 4), (8, 32)];
    for &(w, h) in &DIMS {
        for _ in 0..2000 {
            let stride = w + rng.range(5) as usize; // strided + contiguous
            let src: Vec<i16> = (0..h * stride).map(|_| rng.v16()).collect();
            assert_eq!(
                sum_squares_2d_i16(&src, stride, w, h),
                c::ref_sum_squares_2d_i16(&src, stride, w, h),
                "2d w={w} h={h} stride={stride}"
            );
        }
    }
}
