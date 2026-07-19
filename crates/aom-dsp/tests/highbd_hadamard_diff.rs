//! Differential harness for the highbd Hadamard family (8x8/16x16/32x32) vs C
//! libaom: distinct from lowbd (i16 first pass, i32 second pass, no column swap).
//! 13-bit residual inputs (highbd dynamic range).

use aom_dsp::dist::hadamard::{highbd_hadamard_16x16, highbd_hadamard_32x32, highbd_hadamard_8x8, satd};
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
    // 13-bit residual (highbd): [-4095, 4095].
    fn diff(&mut self) -> i16 {
        (self.next() % 8191) as i16 - 4095
    }
}

#[test]
fn highbd_hadamard_satd_byte_identical() {
    let mut rng = Rng(0x4b0d_0bad_c0de_0011);
    for &n in &[8usize, 16, 32] {
        let stride = n + 4;
        for _ in 0..40_000 {
            let src: Vec<i16> = (0..stride * n).map(|_| rng.diff()).collect();
            let got: Vec<i32> = match n {
                8 => highbd_hadamard_8x8(&src, stride).to_vec(),
                16 => highbd_hadamard_16x16(&src, stride).to_vec(),
                32 => highbd_hadamard_32x32(&src, stride).to_vec(),
                _ => unreachable!(),
            };
            let want = c::ref_highbd_hadamard(n, &src, stride);
            assert_eq!(got, want, "highbd hadamard {n}x{n}");
            assert_eq!(satd(&got), c::ref_satd(&want), "highbd satd {n}x{n}");
        }
    }
}
