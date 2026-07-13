//! Differential harness for Hadamard transform + SATD vs C libaom v3.14.1.

use aom_dist::hadamard::{hadamard_16x16, hadamard_4x4, hadamard_8x8, satd};
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
    // 9-bit signed residual [-255, 255]
    fn diff(&mut self) -> i16 {
        (self.next() % 511) as i16 - 255
    }
}

#[test]
fn hadamard_satd_byte_identical() {
    let mut rng = Rng(0x_4ada_0badc0de_11);
    for &n in &[4usize, 8, 16] {
        let stride = n + 4;
        for _ in 0..50_000 {
            let src: Vec<i16> = (0..stride * n).map(|_| rng.diff()).collect();
            let got: Vec<i32> = match n {
                4 => hadamard_4x4(&src, stride).to_vec(),
                8 => hadamard_8x8(&src, stride).to_vec(),
                16 => hadamard_16x16(&src, stride).to_vec(),
                _ => unreachable!(),
            };
            let want = c::ref_hadamard(n, &src, stride);
            assert_eq!(got, want, "hadamard {n}x{n}");
            assert_eq!(satd(&got), c::ref_satd(&want), "satd {n}x{n}");
        }
    }
}
