//! SIMD column kernel for `txb_init_levels` (Gate 3) — bit-identical to the
//! scalar port on the FULL i32 domain, at every dispatch tier.
//!
//! Same aom-rs SIMD pattern as `aom_quant::simd` / `aom_cdef::simd`: the
//! magetypes kernel handles heights 8/16/32 (whole 8-lane column chunks);
//! the `_scalar` incant variant and the height-4 route call the transcribed
//! port verbatim ([`crate::txb_init_levels_scalar`]).
//!
//! # Bit-exactness (full domain)
//!
//! Per coefficient the scalar port computes `unsigned_abs().min(127) as u8`.
//! Lanes compute `a = (x ^ (x>>31)) - (x>>31)` (wrapping — `i32::MIN` stays
//! `i32::MIN`), then `blend(a < 0, 127, min(a, 127))`: the only negative `a`
//! is the `i32::MIN` lane, whose `unsigned_abs() = 2^31` also clamps to 127
//! in the scalar port. Everything else is `0 <= a <= i32::MAX`, where the
//! signed lane `min` equals the scalar's unsigned min. The final `as u8`
//! narrowing writes values already in `0..=127`.

use archmage::prelude::*;

use crate::{TX_PAD_BOTTOM, TX_PAD_END, TX_PAD_HOR};

/// Scalar tier = the transcribed port, verbatim.
pub(crate) fn txb_init_levels_impl_scalar(
    _t: archmage::ScalarToken,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    crate::txb_init_levels_scalar(coeff, width, height, levels)
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
pub(crate) fn txb_init_levels_impl(
    token: Token,
    coeff: &[i32],
    width: usize,
    height: usize,
    levels: &mut [u8],
) {
    let stride = height + TX_PAD_HOR;
    let tail = stride * width;
    levels[tail..tail + TX_PAD_BOTTOM * stride + TX_PAD_END].fill(0);

    let zero = i32x8::zero(token);
    let cap = i32x8::splat(token, i8::MAX as i32);
    // |x|.min(127) with the i32::MIN lane mapping to 127 exactly like the
    // scalar port's unsigned_abs().min(127) — see module docs.
    let abs127 = |x: i32x8| {
        let m = x.shr_arithmetic_const::<31>();
        let a = (x ^ m) - m; // |x| (wrapping; i32::MIN stays negative)
        i32x8::blend(a.simd_lt(zero), cap, a.min(cap))
    };

    if height == 4 {
        // One 8-lane vector = TWO 4-coeff columns; each column's 4 levels +
        // 4 pad zeros are 8 output bytes, so a pair writes bytes 0..4 and
        // 8..12 of a 16-byte window. Widths are powers of two >= 4.
        debug_assert!(width % 2 == 0);
        for p in 0..width / 2 {
            let arr = abs127(i32x8::from_slice(token, &coeff[p * 8..p * 8 + 8])).to_array();
            let out = &mut levels[p * 2 * stride..p * 2 * stride + 2 * stride];
            for k in 0..4 {
                out[k] = arr[k] as u8;
                out[stride + k] = arr[4 + k] as u8;
            }
            out[4..8].fill(0);
            out[stride + 4..stride + 8].fill(0);
        }
        return;
    }

    assert!(height % 8 == 0);
    for i in 0..width {
        let col = &coeff[i * height..(i + 1) * height];
        let out = &mut levels[i * stride..i * stride + stride];
        for c in 0..height / 8 {
            let arr = abs127(i32x8::from_slice(token, &col[c * 8..c * 8 + 8])).to_array();
            for (k, v) in arr.into_iter().enumerate() {
                out[c * 8 + k] = v as u8;
            }
        }
        out[height..height + TX_PAD_HOR].fill(0);
    }
}
