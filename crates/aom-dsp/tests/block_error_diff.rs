//! Differential harness for the transform-domain distortion (av1_block_error) vs
//! C libaom: error = sum((coeff-dqcoeff)^2), ssz = sum(coeff^2). Lowbd (32-bit
//! products) + highbd (64-bit products, rounded-shift by 2*(bd-8)).

use aom_dsp::dist::{block_error, highbd_block_error};
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
    fn coeff(&mut self, bits: u32) -> i32 {
        (self.next() % (1 << (bits + 1))) as i32 - (1 << bits)
    }
}

#[test]
fn block_error_differential() {
    let mut rng = Rng(0x_b10c_e770_9e37_79b9);
    // tx areas.
    for &n in &[16usize, 64, 256, 1024] {
        for _ in 0..5000 {
            // Lowbd (bd=8) transform coeffs: |c|^2 must stay < 2^31 (32-bit
            // products, matching C's `int` arithmetic) => ~14-bit magnitudes.
            let coeff: Vec<i32> = (0..n).map(|_| rng.coeff(14)).collect();
            let dqcoeff: Vec<i32> = (0..n).map(|_| rng.coeff(14)).collect();
            let got = block_error(&coeff, &dqcoeff);
            let want = c::ref_block_error(&coeff, &dqcoeff);
            assert_eq!(got, want, "block_error n={n}");
        }
    }
}

#[test]
fn highbd_block_error_differential() {
    let mut rng = Rng(0x_b10c_e770_c057_0b11);
    for &n in &[16usize, 64, 256, 1024] {
        for &bd in &[8u8, 10, 12] {
            for _ in 0..2000 {
                // Highbd: 64-bit products, up to ~18-bit magnitudes.
                let coeff: Vec<i32> = (0..n).map(|_| rng.coeff(18)).collect();
                let dqcoeff: Vec<i32> = (0..n).map(|_| rng.coeff(18)).collect();
                let got = highbd_block_error(&coeff, &dqcoeff, bd);
                let want = c::ref_highbd_block_error(&coeff, &dqcoeff, bd);
                assert_eq!(got, want, "highbd_block_error n={n} bd={bd}");
            }
        }
    }
}
