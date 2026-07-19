//! SIMD dispatch for the hot quantizers (Gate 3) — bit-identical to the
//! scalar port at every tier, by construction AND by differential test
//! (`tests/quantize_fp_simd_diff.rs`).
//!
//! Structure (the aom-rs SIMD pattern):
//! * ONE generic kernel written against magetypes vector types, expanded by
//!   `#[magetypes(v3, neon, wasm128, -scalar)]` into per-tier `#[arcane]`
//!   variants (`_v3`/`_neon`/`_wasm128`).
//! * The `_scalar` variant is HAND-WRITTEN to call the transcribed scalar
//!   port verbatim ([`crate::quant::av1_quantize_fp_no_qmatrix`]) — so the
//!   `AOM_FORCE_SCALAR` pin (and non-SIMD platforms) run the exact
//!   C-differentially-validated transcription, not a re-formulation.
//! * The public entry applies the env pin, then `incant!` dispatches to the
//!   best available tier.
//!
//! # Bit-exactness argument (checked by the differential over FULL i32/i16
//! domains — adversarial inputs included, not just production tables)
//!
//! The scalar port computes per coefficient (i64 intermediates):
//! ```text
//! abs        = (coeff ^ sign) - sign                  (wrapping; i32::MIN stays i32::MIN)
//! gate       = (abs << (1+ls)) >= dequant[ac]         (i64 shift, no overflow)
//! abs_r      = clamp(abs + RP2(round[ac], ls), i16)   (i64 add)
//! tmp32      = (abs_r * quant[ac]) >> (16-ls)         (i64 mul/shift, |abs_r*quant| < 2^30)
//! qcoeff     = (tmp32 ^ sign) - sign                  (wrapping)
//! dqcoeff    = ((tmp32 *wrap dequant[ac]) >> ls ^ sign) - sign
//! eob        = 1 + max scan-index with tmp32 != 0
//! ```
//! The vector kernel reformulates only two steps, both exactly:
//! * the gate becomes `abs >= ceil(dequant / 2^(1+ls))` — the standard
//!   integer identity `(a << s) >= d  ⟺  a >= ceil(d / 2^s)` (exact for every
//!   `a` when `|d| < 2^15 << s` bounds hold, which `i16` dequant guarantees;
//!   the `ceil` is `(d + (1<<s) - 1) >> s`, exact for negative `d` too). The
//!   only negative `abs` is `coeff == i32::MIN`, and `i32::MIN < -32768 <=
//!   ceil(d/2^s)` so the gate is false in both formulations.
//! * the i64 `abs + rounding` add becomes `min(abs, 1<<17) + rounding` in i32:
//!   for `abs < 2^17` the i32 math is the i64 math; for `abs >= 2^17` both
//!   sums are `>= 2^17 - 2^15 > 32767`, so both clamp to 32767.
//! Everything else uses lane ops with the scalar port's exact semantics
//! (magetypes integer Mul/Sub are wrapping on every backend, matching the
//! port's `wrapping_mul`/`wrapping_sub`; `>>` is arithmetic).
//! `eob = 1 + max(iscan[rc])` over nonzero positions equals the scan-order
//! maximum because `iscan` is the inverse permutation of `scan`.

use archmage::prelude::*;

/// `av1_quantize_fp_no_qmatrix` with runtime SIMD dispatch. Bit-identical to
/// [`crate::quant::av1_quantize_fp_no_qmatrix`] (the differentially-validated scalar
/// port) at every dispatch tier; under `AOM_FORCE_SCALAR` it IS that function.
///
/// Same contract as the scalar port plus `iscan` (the inverse scan, from the
/// same `av1_scan_orders` row as `scan`; the SIMD tiers derive the EOB from it
/// while walking raster order). `coeff.len()` must be a multiple of 8 (every
/// AV1 transform block area is).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_no_qmatrix_dispatch(
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        quantize_fp_impl(quant, dequant, round, log_scale, scan, iscan, coeff, qcoeff, dqcoeff),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed port, verbatim (`iscan` unused there — the
/// scalar walk derives the EOB in scan order).
#[allow(clippy::too_many_arguments)]
fn quantize_fp_impl_scalar(
    _t: archmage::ScalarToken,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    _iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    crate::quant::av1_quantize_fp_no_qmatrix(
        quant, dequant, round, log_scale, scan, coeff, qcoeff, dqcoeff,
    )
}

// 256-bit kernel: the x8 generic types' backends are v3 (AVX2) / neon /
// wasm128 (x16/512-bit widths are the v4 tier's domain — a hand-slotted
// `_v4` i32x16 variant can join later if profiling justifies it).
// `-scalar` drops the macro's auto-appended scalar variant — the hand-written
// `_scalar` above (the transcribed port, verbatim) takes that slot instead.
#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn quantize_fp_impl(
    token: Token,
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    _scan: &[i16],
    iscan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    assert!(n % 8 == 0, "AV1 tx block areas are multiples of 8");
    assert!(iscan.len() >= n && qcoeff.len() >= n && dqcoeff.len() >= n);

    // Per-class (dc = index 0 / ac = index 1) scalar-side constants.
    let rounding = [
        crate::quant::round_power_of_two(round[0] as i32, log_scale),
        crate::quant::round_power_of_two(round[1] as i32, log_scale),
    ];
    // Gate threshold: (abs << (1+ls)) >= dequant  ⟺  abs >= ceil(dequant / 2^(1+ls)).
    let gs = 1 + log_scale;
    let thr = |d: i16| ((d as i32) + (1 << gs) - 1) >> gs;
    let thr_c = [thr(dequant[0]), thr(dequant[1])];

    // Lane-parameter vectors: chunk 0 has the DC coefficient in lane 0
    // (raster position 0); every other lane/chunk is AC.
    let mk = |dc: i32, ac: i32, first: bool| {
        if first {
            i32x8::from_array(token, [dc, ac, ac, ac, ac, ac, ac, ac])
        } else {
            i32x8::splat(token, ac)
        }
    };

    let (q_chunks, _) = i32x8::partition_slice_mut(token, &mut qcoeff[..n]);
    let (dq_chunks, _) = i32x8::partition_slice_mut(token, &mut dqcoeff[..n]);
    let zero = i32x8::zero(token);
    let abs_cap = i32x8::splat(token, 1 << 17);
    let clamp_lo = i32x8::splat(token, i16::MIN as i32);
    let clamp_hi = i32x8::splat(token, i16::MAX as i32);
    let mut eob_v = zero;

    for ci in 0..n / 8 {
        let first = ci == 0;
        let thr_v = mk(thr_c[0], thr_c[1], first);
        let rnd_v = mk(rounding[0], rounding[1], first);
        let qnt_v = mk(quant[0] as i32, quant[1] as i32, first);
        let dqv_v = mk(dequant[0] as i32, dequant[1] as i32, first);

        let c = i32x8::from_slice(token, &coeff[ci * 8..ci * 8 + 8]);
        // sign = c >> 31 (all-ones for negative); abs = (c ^ sign) - sign (wrapping).
        let sign = c.shr_arithmetic_const::<31>();
        let abs = (c ^ sign) - sign;
        let gate = abs.simd_ge(thr_v);

        // abs_r = clamp(min(abs, 2^17) + rounding, i16::MIN, i16::MAX)
        let abs_r = (abs.min(abs_cap) + rnd_v).clamp(clamp_lo, clamp_hi);
        // tmp32 = (abs_r * quant) >> (16 - ls), gated to 0 outside the gate.
        let prod = abs_r * qnt_v; // |abs_r| <= 32768, |quant| <= 32767: exact in i32
        let tmp = match log_scale {
            0 => prod.shr_arithmetic_const::<16>(),
            1 => prod.shr_arithmetic_const::<15>(),
            2 => prod.shr_arithmetic_const::<14>(),
            _ => unreachable!("log_scale is 0/1/2"),
        };
        let tmp = i32x8::blend(gate, tmp, zero);

        // qcoeff = (tmp ^ sign) - sign; dqcoeff = ((tmp *wrap dq) >> ls ^ sign) - sign.
        let qc = (tmp ^ sign) - sign;
        let absdq = match log_scale {
            0 => (tmp * dqv_v).shr_arithmetic_const::<0>(),
            1 => (tmp * dqv_v).shr_arithmetic_const::<1>(),
            2 => (tmp * dqv_v).shr_arithmetic_const::<2>(),
            _ => unreachable!(),
        };
        let dq = (absdq ^ sign) - sign;
        qc.store(&mut q_chunks[ci]);
        dq.store(&mut dq_chunks[ci]);

        // eob candidate: iscan[rc] + 1 where tmp != 0.
        let base = ci * 8;
        let isc = i32x8::from_array(
            token,
            core::array::from_fn(|k| iscan[base + k] as i32 + 1),
        );
        let nz = tmp.simd_ne(zero);
        eob_v = eob_v.max(i32x8::blend(nz, isc, zero));
    }

    let mx = eob_v.to_array().into_iter().max().unwrap_or(0);
    mx as u16
}
