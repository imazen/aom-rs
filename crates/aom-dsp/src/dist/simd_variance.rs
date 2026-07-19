//! SIMD kernel for `highbd_variance64` (Gate 3) — bit-identical to the scalar
//! port on the pixel domain, at every dispatch tier
//! (`tests/hbd_variance_simd_diff.rs`).
//!
//! Same aom-rs SIMD pattern as `crate::quant::simd` / `crate::cdef::simd`: ONE
//! magetypes generic kernel (`#[magetypes(v3, neon, wasm128, -scalar)]`),
//! hand-written `_scalar` variant = the transcribed port verbatim,
//! `incant!` dispatch in the caller, `aom_dispatch::scalar_forced()` pin.
//!
//! # Bit-exactness (pixel domain: `a`, `b` < `1 << bd`, `bd <= 12`)
//!
//! Per pixel the scalar port computes `diff = a - b` (|diff| < 4096),
//! `lsum += diff` per row (|row sum| <= 128 * 4095 — no i32 wrap, so lane
//! sums + a horizontal reduce give the identical total), and
//! `tsse += (diff*diff as u32) as u64`. The lane square `mullo` wraps like
//! the scalar's i32 multiply (identical low-32 bits); per-lane u32 row
//! accumulation is exact (row of 128 => 16 squares/lane, each < 2^24 =>
//! per-lane sum < 2^28), and the wrapping u32 `reduce_add` cannot wrap
//! either (row total < 128 * 2^24 = 2^31). Each row's reductions land in
//! the u64/i64 totals exactly as the scalar's do. Block widths are powers
//! of two, so `w >= 8` implies `w % 8 == 0` (asserted).

use archmage::prelude::*;

/// Scalar tier = the transcribed port, verbatim.
pub(crate) fn highbd_variance64_impl_scalar(
    _t: archmage::ScalarToken,
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (u64, i64) {
    crate::dist::highbd_variance64_scalar(a, a_stride, b, b_stride, w, h)
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
pub(crate) fn highbd_variance64_impl(
    token: Token,
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (u64, i64) {
    assert!(w >= 8 && w % 8 == 0, "block widths are powers of two");
    let widen = |s: &[u16]| -> i32x8 {
        let arr: [i32; 8] = core::array::from_fn(|k| s[k] as i32);
        i32x8::from_array(token, arr)
    };
    let mut tsum: i64 = 0;
    let mut tsse: u64 = 0;
    for y in 0..h {
        let ra = y * a_stride;
        let rb = y * b_stride;
        let mut sum_v = i32x8::zero(token);
        // Squares accumulate in i32 lanes: two's-complement adds are
        // bit-identical to the u32 adds the scalar performs, and the final
        // `as u32` reinterpretation recovers the exact row total (< 2^31,
        // per the module-doc bound, so no reduce wrap either).
        let mut sse_v = i32x8::zero(token);
        for c in (0..w).step_by(8) {
            let d = widen(&a[ra + c..ra + c + 8]) - widen(&b[rb + c..rb + c + 8]);
            sum_v = sum_v + d;
            // (diff*diff) as u32 — the lane mullo wraps exactly like the
            // scalar's i32 multiply.
            sse_v = sse_v + d * d;
        }
        tsum += i64::from(sum_v.reduce_add());
        tsse += u64::from(sse_v.reduce_add() as u32);
    }
    (tsse, tsum)
}
