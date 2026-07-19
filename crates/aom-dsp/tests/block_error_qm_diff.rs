//! Differential harness for the QM-weighted transform-domain distortion
//! (av1_block_error_qm). Two checks:
//!  1. Flat matrix (all weights 32 = 1<<AOM_QM_BITS) must equal the REAL C
//!     av1_highbd_block_error_c — this cross-validates the accumulation +
//!     bit-depth-shift structure against a genuine oracle (weight cancels).
//!  2. Random matrices must match the transcribed static-inline oracle — this
//!     covers the per-weight multiply (the only transcription-only arithmetic).

use aom_dsp::dist::block_error_qm;
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
    fn range(&mut self, hi: u32) -> u32 {
        (self.next() % hi as u64) as u32
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

// Per-bd coefficient magnitude bits (per the C overflow note: bd8=18, bd10=20,
// bd12=22 bits including sign).
fn bits_for(bd: u8) -> u32 {
    (17 + (bd as u32 - 8)) .min(21)
}

#[test]
fn block_error_qm_flat_matches_real_block_error() {
    let mut rng = Rng(0x_b10c_9111_9e37_79b9);
    for &n in &[16usize, 64, 256, 1024] {
        for &bd in &[8u8, 10, 12] {
            let b = bits_for(bd);
            for _ in 0..2000 {
                let coeff: Vec<i32> = (0..n).map(|_| rng.coeff(b)).collect();
                let dqcoeff: Vec<i32> = (0..n).map(|_| rng.coeff(b)).collect();
                let flat = vec![32u8; n]; // 1 << AOM_QM_BITS
                let scan = perm(&mut rng, n);
                let got = block_error_qm(&coeff, &dqcoeff, &flat, &scan, bd);
                // Weight cancels: reduces to the real highbd block error.
                let want = c::ref_highbd_block_error(&coeff, &dqcoeff, bd);
                assert_eq!(got, want, "flat block_error_qm vs real n={n} bd={bd}");
            }
        }
    }
}

#[test]
fn block_error_qm_weighted_matches_c() {
    let mut rng = Rng(0x_b10c_9111_c057_0b11);
    for &n in &[16usize, 64, 256, 1024] {
        for &bd in &[8u8, 10, 12] {
            let b = bits_for(bd);
            for _ in 0..2000 {
                let coeff: Vec<i32> = (0..n).map(|_| rng.coeff(b)).collect();
                let dqcoeff: Vec<i32> = (0..n).map(|_| rng.coeff(b)).collect();
                let qmatrix: Vec<u8> = (0..n).map(|_| (1 + rng.range(255)) as u8).collect();
                let scan = perm(&mut rng, n);
                let got = block_error_qm(&coeff, &dqcoeff, &qmatrix, &scan, bd);
                let want = c::ref_block_error_qm(&coeff, &dqcoeff, &qmatrix, &scan, bd);
                assert_eq!(got, want, "weighted block_error_qm n={n} bd={bd}");
            }
        }
    }
}
