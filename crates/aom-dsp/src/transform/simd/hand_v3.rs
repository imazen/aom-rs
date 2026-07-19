//! Hand-written lane kernels — the 1-D transforms whose scalar structure is
//! not the regular ping-pong butterfly stream (`fdct.rs` / `special.rs`):
//! `fdct4`, `iadst4` (all-i64 math), `fadst4` (sinpi, i32 wrapping), and the
//! eight identity transforms. Per-lane bit-identical to the scalar ports on
//! the full i32 domain (module docs in `super` + the `tests` differential).

use archmage::X64V3Token;
use archmage::prelude::*;
use magetypes::simd::i32x8;

use crate::transform::cospi::{NEW_INV_SQRT2, NEW_SQRT2, NEW_SQRT2_BITS, cospi_arr, sinpi_arr};

use super::{hb, mul_rshiftv, rshiftv};

/// Lane twin of [`crate::transform::av1_fdct4`] (`fdct.rs`) — wrapping stage-1 adds,
/// four `half_btf`s, output permutation. Statement-for-statement.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_fdct4_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    cos_bit: i32,
    _stage_range: &[i8],
) {
    let cospi = cospi_arr(cos_bit);

    // stage 1 (wrapping adds; `-a + b` == `b - a` in two's complement)
    let bf1_0 = input[0] + input[3];
    let bf1_1 = input[1] + input[2];
    let bf1_2 = input[1] - input[2];
    let bf1_3 = input[0] - input[3];

    // stage 2
    let step0 = hb(t, cospi[32], bf1_0, cospi[32], bf1_1, cos_bit);
    let step1 = hb(t, -cospi[32], bf1_1, cospi[32], bf1_0, cos_bit);
    let step2 = hb(t, cospi[48], bf1_2, cospi[16], bf1_3, cos_bit);
    let step3 = hb(t, cospi[48], bf1_3, -cospi[16], bf1_2, cos_bit);

    // stage 3 (permutation)
    output[0] = step0;
    output[1] = step2;
    output[2] = step1;
    output[3] = step3;
}

/// A lane vector held as two `i64x4` halves (lanes 0..4 / 4..8) — the
/// representation for the all-i64 `iadst4` math.
#[derive(Clone, Copy)]
struct V64 {
    lo: core::arch::x86_64::__m256i,
    hi: core::arch::x86_64::__m256i,
}

/// Sign-extend the 8 i32 lanes to two i64x4 halves.
#[rite(v3)]
fn widen64(v: i32x8) -> V64 {
    use core::arch::x86_64::*;
    V64 {
        lo: _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v.raw())),
        hi: _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(v.raw())),
    }
}

#[rite(v3)]
fn add64(a: V64, b: V64) -> V64 {
    use core::arch::x86_64::*;
    V64 { lo: _mm256_add_epi64(a.lo, b.lo), hi: _mm256_add_epi64(a.hi, b.hi) }
}

#[rite(v3)]
fn sub64(a: V64, b: V64) -> V64 {
    use core::arch::x86_64::*;
    V64 { lo: _mm256_sub_epi64(a.lo, b.lo), hi: _mm256_sub_epi64(a.hi, b.hi) }
}

/// `c * v` per i64 lane for a NON-NEGATIVE constant `c < 2^31` — exact
/// mod 2^64 (== the scalar i64 product wherever it fits, which the iadst4
/// bounds guarantee: |v| < 2^34, c = sinpi < 2^14 → |c*v| < 2^48).
/// Decompose v = v_lo_u + v_hi_u·2^32 (unsigned dwords): `c*v mod 2^64 =
/// c*v_lo_u + ((c*v_hi_u) << 32)` with wrapping adds/shifts.
#[rite(v3)]
fn mulc64(v: V64, c: i32) -> V64 {
    use core::arch::x86_64::*;
    debug_assert!(c >= 0);
    let cv = _mm256_set1_epi64x(c as i64); // low dword of each i64 lane = c
    let part = |x: __m256i| -> __m256i {
        let lo_prod = _mm256_mul_epu32(x, cv); // c * v_lo_u (exact, < 2^63)
        let hi_prod = _mm256_mul_epu32(_mm256_srli_epi64::<32>(x), cv); // c * v_hi_u
        _mm256_add_epi64(lo_prod, _mm256_slli_epi64::<32>(hi_prod))
    };
    V64 { lo: part(v.lo), hi: part(v.hi) }
}

/// `round_shift(v, bit)` from i64 lanes to i32 lanes — add rounding, LOGICAL
/// shift, take the low dword (exact for `1 <= bit <= 32`, same identity as
/// [`super::hb`]).
#[rite]
fn rshift64(t: X64V3Token, v: V64, bit: i32) -> i32x8 {
    use core::arch::x86_64::*;
    let rnd = _mm256_set1_epi64x(1i64 << (bit - 1));
    let cnt = _mm_cvtsi32_si128(bit);
    let lo = _mm256_srl_epi64(_mm256_add_epi64(v.lo, rnd), cnt);
    let hi = _mm256_srl_epi64(_mm256_add_epi64(v.hi, rnd), cnt);
    i32x8::from_m256i(t, super::low32_of_i64(lo, hi))
}

/// Lane twin of [`crate::transform::av1_iadst4`] (`special.rs`) — the all-i64 sinpi
/// kernel. The scalar's all-zero-input early-out is an optimization, not a
/// semantic branch: on zero input every product/sum is 0 and
/// `round_shift(0, bit) == 0`, so computing through is bit-identical (the
/// differential mixes zero and nonzero columns to pin this).
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_iadst4_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    cos_bit: i32,
    _stage_range: &[i8],
) {
    let bit = cos_bit;
    let sinpi = sinpi_arr(bit);
    let x0 = widen64(input[0]);
    let x1 = widen64(input[1]);
    let x2 = widen64(input[2]);
    let x3 = widen64(input[3]);

    // stage 1
    let s0 = mulc64(x0, sinpi[1]);
    let s1 = mulc64(x0, sinpi[2]);
    let s2 = mulc64(x1, sinpi[3]);
    let s3 = mulc64(x2, sinpi[4]);
    let s4 = mulc64(x2, sinpi[1]);
    let s5 = mulc64(x3, sinpi[2]);
    let s6 = mulc64(x3, sinpi[4]);
    // stage 2
    let s7 = add64(sub64(x0, x2), x3);
    // stage 3 (the C reuse: s3 <- old s2, s2 <- sinpi[3]*s7)
    let s0 = add64(s0, s3);
    let s1 = sub64(s1, s4);
    let s3 = s2;
    let s2 = mulc64(s7, sinpi[3]);
    // stage 4
    let s0 = add64(s0, s5);
    let s1 = sub64(s1, s6);
    // stage 5
    let x0 = add64(s0, s3);
    let x1 = add64(s1, s3);
    let x2 = s2;
    let x3 = add64(s0, s1);
    // stage 6
    let x3 = sub64(x3, s3);

    output[0] = rshift64(t, x0, bit);
    output[1] = rshift64(t, x1, bit);
    output[2] = rshift64(t, x2, bit);
    output[3] = rshift64(t, x3, bit);
}

/// Lane twin of [`crate::transform::av1_fadst4`] (`special.rs`) — i32 wrapping sinpi
/// products/sums (lane mul/add/sub wrap identically), i64 `round_shift` at
/// the end. Same compute-through argument for the zero early-out as iadst4.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_fadst4_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    cos_bit: i32,
    _stage_range: &[i8],
) {
    let bit = cos_bit;
    let sinpi = sinpi_arr(bit);
    let sp = |k: usize| i32x8::splat(t, sinpi[k]);
    let (x0, x1, x2, x3) = (input[0], input[1], input[2], input[3]);

    // stage 1 — wrapping i32 products (lane mul == wrapping_mul)
    let s0 = sp(1) * x0;
    let s1 = sp(4) * x0;
    let s2 = sp(2) * x1;
    let s3 = sp(1) * x1;
    let s4 = sp(3) * x2;
    let s5 = sp(4) * x3;
    let s6 = sp(2) * x3;
    let s7 = x0 + x1;
    // stage 2
    let s7 = s7 - x3;
    // stage 3
    let x0 = s0 + s2;
    let x1 = sp(3) * s7;
    let x2 = s1 - s3;
    let x3 = s4;
    // stage 4
    let x0 = x0 + s5;
    let x2 = x2 + s6;
    // stage 5
    let s0 = x0 + x3;
    let s1 = x1;
    let s2 = x2 - x3;
    let s3 = x2 - x0;
    // stage 6
    let s3 = s3 + x3;

    output[0] = rshiftv(t, s0, bit);
    output[1] = rshiftv(t, s1, bit);
    output[2] = rshiftv(t, s2, bit);
    output[3] = rshiftv(t, s3, bit);
}

// ---- identity transforms ---------------------------------------------------
// iidentity8/32 and fidentity8/32 are wrapping doublings/quadruplings (the
// scalar `(x as i64 * 2) as i32` / `wrapping_mul(2)` — identical mod 2^32);
// the 4/16 variants are `round_shift(x * NewSqrt2-multiples, 12)` — the
// full-i64-product [`mul_rshiftv`] recipe.

/// `av1_iidentity4_c`: `round_shift(NewSqrt2 * x, NewSqrt2Bits)`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity4_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..4 {
        output[i] = mul_rshiftv(t, input[i], NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_iidentity8_c`: `(x as i64 * 2) as i32` == wrapping `x << 1`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity8_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..8 {
        output[i] = input[i].shl_const::<1>();
    }
}

/// `av1_iidentity16_c`: `round_shift(NewSqrt2 * 2 * x, NewSqrt2Bits)`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity16_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..16 {
        output[i] = mul_rshiftv(t, input[i], 2 * NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_iidentity32_c`: `(x as i64 * 4) as i32` == wrapping `x << 2`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_iidentity32_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..32 {
        output[i] = input[i].shl_const::<2>();
    }
}

/// `av1_fidentity4_c`: `round_shift(x * NewSqrt2, NewSqrt2Bits)`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity4_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..4 {
        output[i] = mul_rshiftv(t, input[i], NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_fidentity8_c`: `x.wrapping_mul(2)`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity8_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..8 {
        output[i] = input[i].shl_const::<1>();
    }
}

/// `av1_fidentity16_c`: `round_shift(x * 2 * NewSqrt2, NewSqrt2Bits)`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity16_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..16 {
        output[i] = mul_rshiftv(t, input[i], 2 * NEW_SQRT2, NEW_SQRT2_BITS);
    }
}

/// `av1_fidentity32_c`: `x.wrapping_mul(4)`.
#[rite]
#[allow(unused_variables)]
pub(crate) fn av1_fidentity32_v3(
    t: X64V3Token,
    input: &[i32x8],
    output: &mut [i32x8],
    _cos_bit: i32,
    _stage_range: &[i8],
) {
    for i in 0..32 {
        output[i] = input[i].shl_const::<2>();
    }
}

/// `NEW_INV_SQRT2` re-export site check: the inverse row pass scales by it.
#[allow(dead_code)]
const _ASSERT_INV_SQRT2: i32 = NEW_INV_SQRT2;
