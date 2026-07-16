//! Differential harness for the forward 4x4 reversible Walsh–Hadamard transform
//! `av1_fwht4x4` — the coded-lossless (`xd->lossless`) forward transform (KB-5 /
//! task #33). Two independent proofs:
//!
//!  1. **C-oracle exactness** — the port's [`av1_fwht4x4`] must equal libaom
//!     v3.14.1's real exported `av1_fwht4x4_c` (via [`c::ref_fwht4x4`]) element
//!     for element, across random residuals, several input strides, and the full
//!     bd8 / bd12 / int16 magnitude ranges. This is the gold-standard evidence
//!     (real exported C fn), not a transcription cross-check.
//!  2. **Forward→inverse round-trip** — `av1_highbd_iwht4x4_add(av1_fwht4x4(r))`
//!     applied onto a mid-gray prediction must reconstruct `pred + r` exactly for
//!     in-range pixels (the `*UNIT_QUANT_FACTOR` forward scale and the inverse's
//!     `>> UNIT_QUANT_SHIFT` cancel, and the butterflies are reversible). Drives
//!     BOTH inverse branches: the DC-only (`eob <= 1`) special case and the full
//!     16-point transform (`eob > 1`).

use aom_sys_ref as c;
use aom_transform::inv_txfm2d::{av1_fwht4x4, av1_highbd_iwht4x4_add};

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
    /// Uniform in `[lo, hi]` (inclusive).
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i32
    }
}

/// Build a 4-row buffer of `4*stride` i16s with the 4x4 residual laid out at
/// `[r*stride + col]` (row-major, row stride `stride`). `av1_fwht4x4_c` reads
/// `input[i + k*stride]` for `i,k in 0..4`, so `4*stride` bytes are enough for
/// any `stride >= 4`.
fn make_input(rng: &mut Rng, stride: usize, lo: i32, hi: i32) -> Vec<i16> {
    let mut v = vec![0i16; 4 * stride];
    for r in 0..4 {
        for col in 0..4 {
            v[r * stride + col] = rng.range(lo, hi) as i16;
        }
    }
    v
}

/// (1) Exactness vs the real C `av1_fwht4x4_c` across strides and magnitudes.
#[test]
fn fwht4x4_matches_c_oracle() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    // (lo, hi) ranges: bd8 residual, bd12 residual, and the full int16 domain
    // (the C input type) to stress the i64 intermediates.
    let ranges = [(-255i32, 255i32), (-4095, 4095), (-32768, 32767)];
    // stride 4 is the real encoder call (packed residual); the others prove the
    // input-stride handling is faithful.
    let strides = [4usize, 5, 8, 13];
    let mut checked = 0u64;
    for &(lo, hi) in &ranges {
        for &stride in &strides {
            for _ in 0..2000 {
                let input = make_input(&mut rng, stride, lo, hi);
                let mut got = vec![0i32; 16];
                av1_fwht4x4(&input, &mut got, stride);
                let want = c::ref_fwht4x4(&input, stride);
                assert_eq!(
                    got, want,
                    "av1_fwht4x4 != av1_fwht4x4_c (stride={stride} range=[{lo},{hi}])"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 3 * 4 * 2000);
}

/// (2a) Forward→inverse round-trip through the FULL inverse (`eob > 1`): a
/// general residual reconstructs `pred + r` exactly on in-range pixels.
#[test]
fn fwht4x4_roundtrip_full_inverse() {
    for &bd in &[8i32, 10, 12] {
        let mut rng = Rng(0xdead_beef_0000_0000 ^ (bd as u64));
        let pred = 1i32 << (bd - 1); // mid-gray
        let half = 1i32 << (bd - 1);
        for _ in 0..4000 {
            // r in [-half, half-1] so pred + r in [0, (1<<bd)-1] (no clip loss).
            let mut r = [0i16; 16];
            for e in r.iter_mut() {
                *e = rng.range(-half, half - 1) as i16;
            }
            // Forward (packed stride 4).
            let mut coeff = vec![0i32; 16];
            av1_fwht4x4(&r, &mut coeff, 4);
            // Inverse onto pred (recon stride 4). eob > 1 => full 16-point path.
            let mut recon = vec![pred as u16; 16];
            av1_highbd_iwht4x4_add(&coeff, &mut recon, 4, 16, bd);
            for i in 0..16 {
                let want = (pred + i32::from(r[i])) as u16;
                assert_eq!(recon[i], want, "roundtrip full-inv mismatch bd{bd} i={i}");
            }
        }
    }
}

/// (2b) Forward→inverse round-trip through the DC-only inverse (`eob <= 1`): a
/// constant residual produces a DC-only coefficient block, and the DC-only
/// inverse reconstructs the constant exactly.
#[test]
fn fwht4x4_roundtrip_dc_only_inverse() {
    for &bd in &[8i32, 10, 12] {
        let pred = 1i32 << (bd - 1);
        let half = 1i32 << (bd - 1);
        for c_val in -half..half {
            let r = [c_val as i16; 16];
            let mut coeff = vec![0i32; 16];
            av1_fwht4x4(&r, &mut coeff, 4);
            // A constant residual yields only coeff[0] nonzero.
            for (i, &cf) in coeff.iter().enumerate().skip(1) {
                assert_eq!(cf, 0, "constant residual must be DC-only (coeff[{i}] != 0)");
            }
            // eob <= 1 => DC-only inverse path.
            let mut recon = vec![pred as u16; 16];
            av1_highbd_iwht4x4_add(&coeff, &mut recon, 4, 1, bd);
            let want = (pred + c_val) as u16;
            for (i, &px) in recon.iter().enumerate() {
                assert_eq!(px, want, "roundtrip dc-only mismatch bd{bd} c={c_val} i={i}");
            }
        }
    }
}
