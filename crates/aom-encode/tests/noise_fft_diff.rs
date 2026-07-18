//! C7 grain-estimator — DIFFERENTIAL gate for the Wiener-denoise FFT front end.
//!
//! Asserts BIT-IDENTICAL `f32` output between the Rust port
//! ([`aom_encode::noise_fft`]) and the REAL exported transforms:
//!   * `fft2d`  vs `aom_fftNxN_float`  (RTCD-dispatched -> SSE2/AVX2 on this host)
//!   * `ifft2d` vs `aom_ifftNxN_float`
//!   * `NoiseTx` forward/add_energy/filter/inverse vs `aom_noise_tx_*`
//!
//! This is the FEASIBILITY PROOF for byte-exact `--denoise-noise-level`: the
//! transforms use only distinct add/sub/mul roundings (the reference build's
//! `fft.c.o` has ZERO fma instructions), and the SIMD variants are non-fused,
//! column-parallel — so the strict-`f32` scalar port matches the dispatched
//! oracle to the last bit. Every supported block size (2/4/8/16/32) and a range
//! of input scales (noise-like, pixel-like, normalized) are exercised.

use aom_encode::noise_fft::{self, NoiseTx};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform `f32` in `[lo, hi]`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        lo + u * (hi - lo)
    }
}

/// Exact `f32` bit-identity: equal bits, OR both are numerically equal zeros
/// (`+0.0`/`-0.0` compare equal and are interchangeable for these transforms).
fn bit_eq(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0)
}

fn assert_slices(tag: &str, port: &[f32], cref: &[f32]) {
    assert_eq!(port.len(), cref.len(), "{tag}: length mismatch");
    for (i, (&p, &r)) in port.iter().zip(cref).enumerate() {
        assert!(
            bit_eq(p, r),
            "{tag}: mismatch at [{i}]: port={p:?} ({:#010x}) vs C={r:?} ({:#010x})",
            p.to_bits(),
            r.to_bits()
        );
    }
}

const SIZES: [usize; 5] = [2, 4, 8, 16, 32];

#[test]
fn fft2d_matches_c() {
    c::ref_init();
    let mut rng = Rng::new(0xF7_0001);
    let scales: [(f32, f32); 4] = [(-2.0, 2.0), (0.0, 255.0), (0.0, 1.0), (-1000.0, 1000.0)];
    let mut nonzero = 0usize;
    for &bs in &SIZES {
        for &(lo, hi) in &scales {
            for _ in 0..8 {
                let n = bs * bs;
                let input: Vec<f32> = (0..n).map(|_| rng.range(lo, hi)).collect();
                let mut temp = vec![0.0f32; 2 * n];
                let mut out = vec![0.0f32; 2 * n];
                assert!(noise_fft::fft2d(bs, &input, &mut temp, &mut out));
                let cout = c::ref_noise_fft2d(bs, &input).expect("C fft2d");
                assert_slices(&format!("fft2d bs{bs} [{lo},{hi}]"), &out, &cout);
                nonzero += out.iter().filter(|&&v| v != 0.0).count();
            }
        }
    }
    assert!(nonzero > 1000, "fft2d output suspiciously trivial ({nonzero} nonzero)");
    println!("fft2d_diff: all sizes {SIZES:?} x 4 scales x 8 trials bit-identical to C");
}

#[test]
fn ifft2d_matches_c() {
    c::ref_init();
    let mut rng = Rng::new(0x1FF7_0002);
    let mut nonzero = 0usize;
    for &bs in &SIZES {
        for &(lo, hi) in &[(-2.0f32, 2.0f32), (-50.0, 50.0), (0.0, 4.0)] {
            for _ in 0..8 {
                let n = bs * bs;
                // Random packed 2*n*n spectrum (exercises every ifft read path).
                let input: Vec<f32> = (0..2 * n).map(|_| rng.range(lo, hi)).collect();
                let mut temp = vec![0.0f32; 2 * n];
                let mut out = vec![0.0f32; n];
                assert!(noise_fft::ifft2d(bs, &input, &mut temp, &mut out));
                let cout = c::ref_noise_ifft2d(bs, &input).expect("C ifft2d");
                assert_slices(&format!("ifft2d bs{bs} [{lo},{hi}]"), &out, &cout);
                nonzero += out.iter().filter(|&&v| v != 0.0).count();
            }
        }
    }
    assert!(nonzero > 1000, "ifft2d output suspiciously trivial ({nonzero} nonzero)");
    println!("ifft2d_diff: all sizes {SIZES:?} x 3 scales x 8 trials bit-identical to C");
}

#[test]
fn noise_tx_pipeline_matches_c() {
    c::ref_init();
    let mut rng = Rng::new(0x2FF7_0003);
    let mut filt_else = 0usize; // count of psd values large enough to force the else-arm
    for &bs in &SIZES {
        for _ in 0..12 {
            let n = bs * bs;
            // Pixel-like block + a non-negative PSD spanning both filter arms.
            let data: Vec<f32> = (0..n).map(|_| rng.range(0.0, 255.0)).collect();
            let psd: Vec<f32> = (0..n)
                .map(|_| {
                    let v = rng.range(0.0, 1.0);
                    // Mix small (keep-arm) and large (attenuate-else-arm) PSDs.
                    if v < 0.5 { v * 10.0 } else { v * 5.0e5 }
                })
                .collect();
            filt_else += psd.iter().filter(|&&v| v > 1.0e5).count();

            // Rust: mirror the C pipeline order exactly.
            let mut tx = NoiseTx::new(bs).expect("NoiseTx");
            tx.forward(&data);
            let mut energy = vec![0.0f32; n];
            tx.add_energy(&mut energy);
            tx.filter(&psd);
            let mut denoised = vec![0.0f32; n];
            tx.inverse(&mut denoised);

            let (c_denoised, c_energy) = c::ref_noise_tx_pipeline(bs, &data, &psd).expect("C pipeline");
            assert_slices(&format!("noise_tx energy bs{bs}"), &energy, &c_energy);
            assert_slices(&format!("noise_tx denoised bs{bs}"), &denoised, &c_denoised);
        }
    }
    // Anti-vacuity: both filter arms were actually taken across the corpus.
    assert!(filt_else > 20, "filter else-arm under-exercised ({filt_else})");
    println!("noise_tx_pipeline_diff: all sizes {SIZES:?} x 12 trials bit-identical to C (fwd+add_energy+filter+inverse)");
}
