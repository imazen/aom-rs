//! SIMD-vs-scalar permutation-equality integration test for the 2-D transform
//! drivers (`av1_inv_txfm2d_add` / `av1_fwd_txfm2d`).
//!
//! The per-kernel unit differential (`simd::tests` in the lib) pins each 1-D
//! lane kernel == scalar; THIS test pins the 2-D PASS PLUMBING that lives in
//! the drivers, not the kernels — the lr/ud flips, the `clamp_buf` clamps, the
//! `round_shift_array` shifts, the NewSqrt2/NewInvSqrt2 rect scalings, the
//! transpose loads/stores (incl. the W=4 per-lane scatter/gather tail) and the
//! final `highbd_clip_pixel_add` — SIMD == scalar, over the FULL driver clamp
//! domain the vs-C harness cannot reach.
//!
//! Why a separate SIMD-vs-scalar test (not just the vs-C harnesses): the vs-C
//! differentials (`inv_txfm2d_diff`, `txfm2d_diff`) cap inverse coefficients at
//! ±2^16 because C's `half_btf` sums two i32 products in an i32 and OVERFLOWS
//! (undefined behaviour) at the true bd12 clamp bound ±2^19. This test compares
//! the port's SIMD path to the port's SCALAR path — neither has that UB — so it
//! drives inverse coefficients across ±2^20 (they clamp to ±2^19, the |p0 + p1|
//! maximiser that the exact-i64 `hb` recipe exists to survive) and forward
//! residuals across the full i16 range. This is the zero-tolerance guarantee:
//! on crafted-but-decodable streams that push dequantised coefficients to the
//! clamp bounds, the vector path must reproduce the scalar path bit-for-bit.
//!
//! Method: `for_each_token_permutation` runs the whole (tx_size × tx_type × bd
//! × input) matrix once per token permutation, feeding IDENTICAL inputs (the
//! RNG is re-seeded with the same constant at the top of every permutation, so
//! dispatch is the only thing that varies). Every permutation's complete output
//! set must byte-match the first permutation's. The harness always includes an
//! all-off (scalar) permutation and — on AVX2 CI — a v3 (SIMD) permutation, so
//! the equality chain transitively pins SIMD == scalar. Non-vacuous: asserts a
//! SIMD permutation and a scalar permutation both ran.

use aom_dsp::transform::inv_txfm2d::{av1_inv_txfm2d_add, inv_input_len, inv_txfm_valid};
use aom_dsp::transform::txfm2d::{av1_fwd_txfm2d, fwd_txfm_valid};
use archmage::prelude::*;
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use archmage::X64V3Token;

const W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

const SEED: u64 = 0x_5119_d1ff_2d00_0001;

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
    /// Inverse dequantised coefficient pushed to ±2^20 — well past the vs-C
    /// harness's ±2^16, so a large fraction land on the driver's ±2^19 clamp
    /// (the bd12 row `clamp_buf`), which is exactly the `half_btf` i64-sum
    /// stress the vector recipe must survive. Both paths clamp identically, so
    /// there is no UB (unlike C).
    fn coeff(&mut self) -> i32 {
        (self.next() % (1 << 21)) as i32 - (1 << 20)
    }
    /// Full-range i16 forward residual (far beyond the [-255, 255] encoder
    /// range) — stresses the forward col shl-clamp + butterfly plumbing.
    fn residual(&mut self) -> i16 {
        self.next() as i16
    }
    fn pixel(&mut self, bd: i32) -> u16 {
        (self.next() % (1u64 << bd)) as u16
    }
}

/// Exact clamp-bound spike patterns for the inverse coefficient buffer — the
/// `|p0 + p1|` maximisers plus a DC-only spike. `k` selects the pattern.
fn inv_spike(k: usize, i: usize, len: usize) -> i32 {
    const B: i32 = 1 << 19; // the bd12 row clamp bound
    match k {
        0 => B,                                             // all +bound
        1 => -B,                                            // all -bound
        2 => {
            if i % 2 == 0 {
                B
            } else {
                -B
            }
        } // alternating
        3 => {
            if i % 2 == 0 {
                -B
            } else {
                B
            }
        } // alternating (other phase)
        4 => {
            if i == 0 {
                1 << 20
            } else {
                0
            }
        } // DC spike beyond the clamp
        5 => {
            if i + 1 == len {
                -(1 << 20)
            } else {
                0
            }
        } // last-coeff spike
        _ => B - 1,                                         // ±(2^19 - 1), the exact hi bound
    }
}

/// Full-range i16 spike patterns for the forward residual buffer.
fn fwd_spike(k: usize, i: usize) -> i16 {
    match k {
        0 => i16::MAX,
        1 => i16::MIN,
        2 => {
            if i % 2 == 0 {
                i16::MAX
            } else {
                i16::MIN
            }
        }
        _ => {
            if i == 0 {
                i16::MAX
            } else {
                0
            }
        }
    }
}

const RAND_REPS: usize = 5;
const INV_SPIKES: usize = 7;
const FWD_SPIKES: usize = 4;

/// Run the entire (tx_size × tx_type × bd × input) matrix through the public
/// 2-D entries under the CURRENT token permutation, collecting every output
/// (tagged with a human-readable label). Inputs are deterministic (fixed seed),
/// so only the SIMD/scalar dispatch differs between calls.
fn all_outputs() -> Vec<(String, Vec<i64>)> {
    let mut rng = Rng(SEED);
    let mut out: Vec<(String, Vec<i64>)> = Vec::new();

    for tx_size in 0..19usize {
        let (w, h) = (W[tx_size], H[tx_size]);

        // ---------- inverse: coeffs ±2^20, tight + strided dest ----------
        for tx_type in 0..16usize {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            let ilen = inv_input_len(tx_size);
            for bd in [8i32, 10, 12] {
                // strides: tight (w) and an odd strided dest (w + 3) to drive
                // the per-lane scatter tail of the transpose store.
                for &stride in &[w, w + 3] {
                    let mut push_inv = |label: String, input: &[i32], rng: &mut Rng| {
                        let dest: Vec<u16> = (0..h * stride).map(|_| rng.pixel(bd)).collect();
                        let mut got = dest.clone();
                        av1_inv_txfm2d_add(input, &mut got, stride, tx_type, tx_size, bd);
                        out.push((label, got.iter().map(|&x| x as i64).collect()));
                    };
                    for rep in 0..RAND_REPS {
                        let input: Vec<i32> = (0..ilen).map(|_| rng.coeff()).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} rand{rep}"),
                            &input,
                            &mut rng,
                        );
                    }
                    for k in 0..INV_SPIKES {
                        let input: Vec<i32> =
                            (0..ilen).map(|i| inv_spike(k, i, ilen)).collect();
                        push_inv(
                            format!("inv sz{tx_size} ty{tx_type} bd{bd} st{stride} spike{k}"),
                            &input,
                            &mut rng,
                        );
                    }
                }
            }
        }

        // ---------- forward: full-range i16 residuals ----------
        for tx_type in 0..16usize {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            let mut push_fwd = |label: String, input: &[i16]| {
                let mut got = vec![0i32; w * h];
                av1_fwd_txfm2d(input, &mut got, w, tx_type, tx_size);
                out.push((label, got.iter().map(|&x| x as i64).collect()));
            };
            for rep in 0..RAND_REPS {
                let input: Vec<i16> = (0..w * h).map(|_| rng.residual()).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} rand{rep}"), &input);
            }
            for k in 0..FWD_SPIKES {
                let input: Vec<i16> = (0..w * h).map(|i| fwd_spike(k, i)).collect();
                push_fwd(format!("fwd sz{tx_size} ty{tx_type} spike{k}"), &input);
            }
        }
    }
    out
}

#[test]
fn txfm2d_simd_equals_scalar_at_every_permutation() {
    // Fire the AOM_FORCE_SCALAR pin (if set) BEFORE the permutation harness —
    // the harness then owns token state, so both a SIMD and a scalar
    // permutation run in either dispatch mode.
    let _ = aom_dsp::dispatch::scalar_forced();

    let mut reference: Option<Vec<(String, Vec<i64>)>> = None;
    let mut simd_perms = 0usize;
    let mut scalar_perms = 0usize;

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
        if X64V3Token::summon().is_some() {
            simd_perms += 1;
        } else {
            scalar_perms += 1;
        }
        let cur = all_outputs();
        match reference.as_ref() {
            None => reference = Some(cur),
            Some(r) => {
                assert_eq!(
                    cur.len(),
                    r.len(),
                    "permutation [{tier}] produced a different number of cells"
                );
                for ((cl, cd), (rl, rd)) in cur.iter().zip(r.iter()) {
                    assert_eq!(
                        cd, rd,
                        "permutation [{tier}] diverged from the reference at cell '{cl}' \
                         (reference cell '{rl}')"
                    );
                }
            }
        }
    });

    eprintln!(
        "txfm2d SIMD==scalar parity: {report}; simd_perms={simd_perms} scalar_perms={scalar_perms}"
    );
    // Non-vacuity: a SIMD permutation must have run (AVX2 present on x86 CI),
    // and both a SIMD and a scalar permutation must have been compared so the
    // equality chain actually pins the vector path against the scalar path.
    assert!(simd_perms >= 1, "the SIMD (v3) permutation must run at least once (AVX2 CI)");
    assert!(scalar_perms >= 1, "the all-off (scalar) permutation must run at least once");
    assert!(report.permutations_run >= 2, "need >=2 permutations to compare SIMD vs scalar");
}
