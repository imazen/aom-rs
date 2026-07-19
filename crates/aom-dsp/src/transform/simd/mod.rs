//! SIMD (Gate 3) for the transform stack — lane-batched 1-D kernels + the
//! 2-D drivers' vector passes, bit-identical to the scalar port per lane.
//! x86-64 only (the module is cfg'd out elsewhere; NEON falls to scalar).
//!
//! # Shape (from the STATUS.md transform-SIMD design)
//!
//! Vectorize ACROSS independent 1-D transforms: the inverse 2-D driver's
//! COLUMN pass processes 8 adjacent columns as `i32x8` lanes — `buf[r*col_n +
//! c..c+8]` is a contiguous 8-lane load per row, NO transposes. The 1-D
//! kernel bodies are mechanical lane rewrites of the generated scalar
//! butterflies (`inv1d_v3_gen.rs`, emitted by `transpile_txfm1d.py --lanes`),
//! and the pass wrapper reproduces the driver's clamp / round-shift /
//! flip / clip-add stages lane-wise. Dispatch is per `func_col`: blocks
//! whose column kernel is in the ported set take the vector path, everything
//! else keeps the scalar per-column loop (byte-untouched, and the
//! `AOM_FORCE_SCALAR` pin routes everything there).
//!
//! # Bit-exactness argument (FULL i32 domain — stronger than the driver
//! clamp bounds; pinned by `tests` below at every token permutation)
//!
//! Every lane op reproduces the scalar op's exact semantics for ANY i32
//! input, so SIMD == scalar unconditionally (no domain reasoning needed):
//! * wrapping add/sub: magetypes `+`/`-` are wrapping on every backend,
//!   and `-a + b == b - a` in two's complement (the transpiler emits the
//!   latter).
//! * `clamp_value(v, bit)`: `bit <= 0` and `bit >= 32` are identities in
//!   the scalar port (the i64 bounds cover all of i32 at `bit == 32`); for
//!   `1..=31` the bounds are i32-representable → lane min/max. [`clampv`]
//! * `half_btf` — THE exactness trap: the scalar port wraps each PRODUCT in
//!   i32 (`w.wrapping_mul(in)`, matching C's int multiply) but sums the two
//!   products + rounding in **i64**. At driver clamp bounds a product
//!   reaches 2^32 and the sum needs 33 bits, so an i32-lane sum (libaom's
//!   own SSE4/AVX2 shape) diverges on crafted-but-decodable streams. [`hb`]
//!   reproduces the i64 sum exactly: `vpmulld` (wrapped products, ==
//!   scalar), widen each 128-half via `vpmovsxdq` to 2×i64x4, `vpaddq` sums
//!   (|p0|,|p1| <= 2^31, rnd <= 2^31 → no i64 overflow), then the
//!   arithmetic-shift + truncate pair via LOGICAL `vpsrlq` + low-dword
//!   gather — exact because `((v >>_arith b) as i32) == low32(v >>_logical
//!   b)` for any v when `1 <= b <= 32` (the differing sign-fill bits all
//!   land at positions >= 32 and are truncated away; cos_bit is 10..=13).
//!   AVX2 has no `vpsraq`; the logical+truncate trick dodges it.
//! * `round_shift(v as i64, bit)` (the positive-bit `round_shift_array`
//!   arm): the same widen → add rounding → logical shift → truncate recipe.
//!   [`rshiftv`]
//! * `highbd_clip_pixel_add`: the i32 lane add wraps like the scalar
//!   `wrapping_add`; clamp to `[0, (1<<bd)-1]` is lane min/max; the `as u16`
//!   narrowing is exact after the clamp.
//! * `lr_flip` lane reversal and `ud_flip` row reversal are pure index
//!   permutations ([`revv`] / loop order), identical to the scalar loops.
//!
//! magetypes has NO integer widening ops, so [`hb`]/[`rshiftv`] use the raw
//! `__m256i` escape (`i32x8::raw()`/`from_m256i`) with VALUE intrinsics —
//! safe inside `#[rite]`/`#[arcane]` `#[target_feature]` regions, keeping
//! `#![forbid(unsafe_code)]`.

mod hand_v3;
mod inv1d_v3_gen;
mod txfm1d_v3_gen;

use archmage::SimdToken;
use archmage::X64V3Token;
use archmage::prelude::*;
use magetypes::simd::i32x8;

use crate::transform::cospi::{NEW_INV_SQRT2, NEW_SQRT2, NEW_SQRT2_BITS};
use hand_v3::{
    av1_fadst4_v3, av1_fdct4_v3, av1_fidentity4_v3, av1_fidentity8_v3, av1_fidentity16_v3,
    av1_fidentity32_v3, av1_iadst4_v3, av1_iidentity4_v3, av1_iidentity8_v3, av1_iidentity16_v3,
    av1_iidentity32_v3,
};
use inv1d_v3_gen::{
    av1_iadst8_v3, av1_iadst16_v3, av1_idct4_v3, av1_idct8_v3, av1_idct16_v3, av1_idct32_v3,
    av1_idct64_v3,
};
use txfm1d_v3_gen::{
    av1_fadst8_v3, av1_fadst16_v3, av1_fdct8_v3, av1_fdct16_v3, av1_fdct32_v3, av1_fdct64_v3,
};

/// `half_btf` on 8 lanes — the exact-i64 recipe (see the module docs).
/// Bit-identical to [`crate::transform::fdct::half_btf`] per lane for ANY i32 lanes and
/// any `cos_bit` in `1..=32` (the transforms use 10..=13).
#[rite]
pub(crate) fn hb(t: X64V3Token, w0: i32, in0: i32x8, w1: i32, in1: i32x8, cos_bit: i32) -> i32x8 {
    use core::arch::x86_64::*;
    // Wrapped i32 products, exactly like the scalar port's wrapping_mul.
    let p0 = _mm256_mullo_epi32(_mm256_set1_epi32(w0), in0.raw());
    let p1 = _mm256_mullo_epi32(_mm256_set1_epi32(w1), in1.raw());
    // Widen to i64 and sum with the rounding constant — no i64 overflow:
    // |p0|,|p1| <= 2^31 and rnd <= 2^31, so |sum| <= 2^32 + 2^31 < 2^63.
    let rnd = _mm256_set1_epi64x(1i64 << (cos_bit - 1));
    let lo = _mm256_add_epi64(
        _mm256_add_epi64(
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p0)),
            _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p1)),
        ),
        rnd,
    );
    let hi = _mm256_add_epi64(
        _mm256_add_epi64(
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(p0)),
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(p1)),
        ),
        rnd,
    );
    // (sum >>_arith bit) as i32 == low32(sum >>_logical bit) for bit <= 32.
    let cnt = _mm_cvtsi32_si128(cos_bit);
    i32x8::from_m256i(t, low32_of_i64(_mm256_srl_epi64(lo, cnt), _mm256_srl_epi64(hi, cnt)))
}

/// Gather the low dword of each i64 lane of (`lo`, `hi`) into one `__m256i`.
#[rite(v3)]
fn low32_of_i64(
    lo: core::arch::x86_64::__m256i,
    hi: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let a = _mm256_permutevar8x32_epi32(lo, idx);
    let b = _mm256_permutevar8x32_epi32(hi, idx);
    _mm256_blend_epi32::<0b1111_0000>(a, b)
}

/// `clamp_value(v, bit)` on lanes — identical to the scalar port for any i32
/// lanes and any `bit`: `<= 0` and `>= 32` are identities (the scalar i64
/// bounds cover all of i32 there), else lane min/max on the i32 bounds.
#[rite]
pub(crate) fn clampv(t: X64V3Token, v: i32x8, bit: i8) -> i32x8 {
    if bit <= 0 || bit >= 32 {
        return v;
    }
    let hi = ((1i64 << (bit - 1)) - 1) as i32;
    let lo = (-(1i64 << (bit - 1))) as i32;
    v.clamp(i32x8::splat(t, lo), i32x8::splat(t, hi))
}

/// `wrapping_neg` on lanes (`0 - v` wraps identically).
#[rite]
#[allow(dead_code)] // used by the iadst/fdct lane kernels (next chunks)
pub(crate) fn negv(t: X64V3Token, v: i32x8) -> i32x8 {
    i32x8::zero(t) - v
}

/// `round_shift(v as i64, bit)` on lanes for `bit` in `1..=32` — widen, add
/// rounding, logical shift, truncate (the same identity as [`hb`]).
#[rite]
fn rshiftv(t: X64V3Token, v: i32x8, bit: i32) -> i32x8 {
    use core::arch::x86_64::*;
    debug_assert!((1..=32).contains(&bit));
    let rnd = _mm256_set1_epi64x(1i64 << (bit - 1));
    let lo = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(v.raw())), rnd);
    let hi = _mm256_add_epi64(
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(v.raw())),
        rnd,
    );
    let cnt = _mm_cvtsi32_si128(bit);
    i32x8::from_m256i(t, low32_of_i64(_mm256_srl_epi64(lo, cnt), _mm256_srl_epi64(hi, cnt)))
}

/// Reverse the 8 lanes (for `lr_flip` column groups).
#[rite]
fn revv(t: X64V3Token, v: i32x8) -> i32x8 {
    use core::arch::x86_64::*;
    let idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    i32x8::from_m256i(t, _mm256_permutevar8x32_epi32(v.raw(), idx))
}

/// `round_shift(v as i64 * mul, bit)` on lanes — the full-i64-product recipe
/// (`vpmuldq` even/odd, exact 32×32→64 signed products == `v as i64 * mul`),
/// add rounding (no overflow: |mul| <= 2^14 at the call sites → |prod| <=
/// 2^45), LOGICAL shift + take the low dword of each i64 lane (exact for
/// `1 <= bit <= 32`). Bit-identical to the scalar
/// `round_shift(v as i64 * mul as i64, bit)` for ANY i32 `v`.
#[rite]
pub(crate) fn mul_rshiftv(t: X64V3Token, v: i32x8, mul: i32, bit: i32) -> i32x8 {
    use core::arch::x86_64::*;
    debug_assert!((1..=32).contains(&bit) && mul.unsigned_abs() < (1 << 15));
    let m = _mm256_set1_epi32(mul);
    // vpmuldq reads the SIGNED low dword of each 64-bit lane.
    let even = _mm256_mul_epi32(v.raw(), m); // source lanes 0,2,4,6
    let odd = _mm256_mul_epi32(_mm256_srli_epi64::<32>(v.raw()), m); // lanes 1,3,5,7
    let rnd = _mm256_set1_epi64x(1i64 << (bit - 1));
    let cnt = _mm_cvtsi32_si128(bit);
    let re = _mm256_srl_epi64(_mm256_add_epi64(even, rnd), cnt);
    let ro = _mm256_srl_epi64(_mm256_add_epi64(odd, rnd), cnt);
    // Valid low dwords of `re` sit at dword positions 0,2,4,6 (source lanes
    // 0,2,4,6); shift `ro`'s up to 1,3,5,7 and blend.
    let out = _mm256_blend_epi32::<0b1010_1010>(re, _mm256_slli_epi64::<32>(ro));
    i32x8::from_m256i(t, out)
}

/// The NEGATIVE-bit `round_shift_array` arm on lanes: `clamp_i64(v << k)`
/// truncated to i32 — widen to i64 halves, shift left (exact: k <= 4 →
/// |v<<k| < 2^36), clamp to the i32 range with cmpgt/blendv min/max (AVX2
/// has no vpmin/maxq), take low dwords. Bit-identical to the scalar arm
/// (`((1i64 << k) * v).clamp(i32::MIN, i32::MAX) as i32`) for ANY i32 v.
/// Used by the FORWARD col pass (fwd shift[0] == 2); the inverse shifts are
/// all positive-bit.
#[rite]
fn shl_clamp64v(t: X64V3Token, v: i32x8, k: i32) -> i32x8 {
    use core::arch::x86_64::*;
    debug_assert!((1..=4).contains(&k));
    let cnt = _mm_cvtsi32_si128(k);
    let min_v = _mm256_set1_epi64x(i32::MIN as i64);
    let max_v = _mm256_set1_epi64x(i32::MAX as i64);
    let part = |x: __m128i| -> __m256i {
        let w = _mm256_sll_epi64(_mm256_cvtepi32_epi64(x), cnt);
        // min(w, max): if w > max take max; then max(_, min): if min > w take min.
        let w = _mm256_blendv_epi8(w, max_v, _mm256_cmpgt_epi64(w, max_v));
        _mm256_blendv_epi8(w, min_v, _mm256_cmpgt_epi64(min_v, w))
    };
    let lo = part(_mm256_castsi256_si128(v.raw()));
    let hi = part(_mm256_extracti128_si256::<1>(v.raw()));
    i32x8::from_m256i(t, low32_of_i64(lo, hi))
}

/// Sign-extend 8 i16s (the forward transform's residual input) to i32 lanes.
/// The fixed-size array round-trip lets LLVM emit `vpmovsxwd`.
#[rite]
fn widen16(t: X64V3Token, s: &[i16]) -> i32x8 {
    let a: [i16; 8] = s[..8].try_into().unwrap();
    i32x8::from_array(t, core::array::from_fn(|j| a[j] as i32))
}

/// 8x8 i32 in-register transpose (unpack32 → unpack64 → permute2x128, the
/// standard 24-op AVX2 pattern) — a pure lane permutation, so exactness is
/// structural. Used by the row passes (strided side of the tile).
#[rite]
fn transpose8(t: X64V3Token, v: &[i32x8]) -> [i32x8; 8] {
    use core::arch::x86_64::*;
    let a0 = _mm256_unpacklo_epi32(v[0].raw(), v[1].raw());
    let a1 = _mm256_unpackhi_epi32(v[0].raw(), v[1].raw());
    let a2 = _mm256_unpacklo_epi32(v[2].raw(), v[3].raw());
    let a3 = _mm256_unpackhi_epi32(v[2].raw(), v[3].raw());
    let a4 = _mm256_unpacklo_epi32(v[4].raw(), v[5].raw());
    let a5 = _mm256_unpackhi_epi32(v[4].raw(), v[5].raw());
    let a6 = _mm256_unpacklo_epi32(v[6].raw(), v[7].raw());
    let a7 = _mm256_unpackhi_epi32(v[6].raw(), v[7].raw());
    let b0 = _mm256_unpacklo_epi64(a0, a2);
    let b1 = _mm256_unpackhi_epi64(a0, a2);
    let b2 = _mm256_unpacklo_epi64(a1, a3);
    let b3 = _mm256_unpackhi_epi64(a1, a3);
    let b4 = _mm256_unpacklo_epi64(a4, a6);
    let b5 = _mm256_unpackhi_epi64(a4, a6);
    let b6 = _mm256_unpacklo_epi64(a5, a7);
    let b7 = _mm256_unpackhi_epi64(a5, a7);
    [
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b0, b4)),
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b1, b5)),
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b2, b6)),
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x20>(b3, b7)),
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b0, b4)),
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b1, b5)),
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b2, b6)),
        i32x8::from_m256i(t, _mm256_permute2x128_si256::<0x31>(b3, b7)),
    ]
}

/// 1-D kernel selector — TXFM_TYPE ids 0..=11 (DCT4..64, ADST4/8/16,
/// IDTX4/8/16/32), one enum per direction. ALL 12 are ported in each
/// direction; the `Option` maps stay for unknown-id safety (→ scalar loop).
#[derive(Clone, Copy)]
enum Inv1d {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
    Adst4,
    Adst8,
    Adst16,
    Idtx4,
    Idtx8,
    Idtx16,
    Idtx32,
}

fn inv_kernel(txfm_type: i32) -> Option<Inv1d> {
    match txfm_type {
        0 => Some(Inv1d::Dct4),
        1 => Some(Inv1d::Dct8),
        2 => Some(Inv1d::Dct16),
        3 => Some(Inv1d::Dct32),
        4 => Some(Inv1d::Dct64),
        5 => Some(Inv1d::Adst4),
        6 => Some(Inv1d::Adst8),
        7 => Some(Inv1d::Adst16),
        8 => Some(Inv1d::Idtx4),
        9 => Some(Inv1d::Idtx8),
        10 => Some(Inv1d::Idtx16),
        11 => Some(Inv1d::Idtx32),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum Fwd1d {
    Dct4,
    Dct8,
    Dct16,
    Dct32,
    Dct64,
    Adst4,
    Adst8,
    Adst16,
    Idtx4,
    Idtx8,
    Idtx16,
    Idtx32,
}

fn fwd_kernel(txfm_type: i32) -> Option<Fwd1d> {
    match txfm_type {
        0 => Some(Fwd1d::Dct4),
        1 => Some(Fwd1d::Dct8),
        2 => Some(Fwd1d::Dct16),
        3 => Some(Fwd1d::Dct32),
        4 => Some(Fwd1d::Dct64),
        5 => Some(Fwd1d::Adst4),
        6 => Some(Fwd1d::Adst8),
        7 => Some(Fwd1d::Adst16),
        8 => Some(Fwd1d::Idtx4),
        9 => Some(Fwd1d::Idtx8),
        10 => Some(Fwd1d::Idtx16),
        11 => Some(Fwd1d::Idtx32),
        _ => None,
    }
}

/// The kernel's point count (== how many input/output vectors it consumes).
fn inv_kernel_n(k: Inv1d) -> usize {
    match k {
        Inv1d::Dct4 | Inv1d::Adst4 | Inv1d::Idtx4 => 4,
        Inv1d::Dct8 | Inv1d::Adst8 | Inv1d::Idtx8 => 8,
        Inv1d::Dct16 | Inv1d::Adst16 | Inv1d::Idtx16 => 16,
        Inv1d::Dct32 | Inv1d::Idtx32 => 32,
        Inv1d::Dct64 => 64,
    }
}

/// The forward kernel's point count (== how many input/output vectors it
/// consumes) — the symmetric twin of [`inv_kernel_n`].
fn fwd_kernel_n(k: Fwd1d) -> usize {
    match k {
        Fwd1d::Dct4 | Fwd1d::Adst4 | Fwd1d::Idtx4 => 4,
        Fwd1d::Dct8 | Fwd1d::Adst8 | Fwd1d::Idtx8 => 8,
        Fwd1d::Dct16 | Fwd1d::Adst16 | Fwd1d::Idtx16 => 16,
        Fwd1d::Dct32 | Fwd1d::Idtx32 => 32,
        Fwd1d::Dct64 => 64,
    }
}

/// Direct-dispatch the selected inverse 1-D lane kernel (rite→rite calls
/// inline into the caller's feature region; kernels are `#[target_feature]`
/// fns and cannot be stored as plain fn pointers).
#[rite]
fn run_inv1d(
    t: X64V3Token,
    k: Inv1d,
    input: &[i32x8],
    out: &mut [i32x8],
    cos_bit: i32,
    stage_range: &[i8],
) {
    match k {
        Inv1d::Dct4 => av1_idct4_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Dct8 => av1_idct8_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Dct16 => av1_idct16_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Dct32 => av1_idct32_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Dct64 => av1_idct64_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Adst4 => av1_iadst4_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Adst8 => av1_iadst8_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Adst16 => av1_iadst16_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Idtx4 => av1_iidentity4_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Idtx8 => av1_iidentity8_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Idtx16 => av1_iidentity16_v3(t, input, out, cos_bit, stage_range),
        Inv1d::Idtx32 => av1_iidentity32_v3(t, input, out, cos_bit, stage_range),
    }
}

#[rite]
fn run_fwd1d(
    t: X64V3Token,
    k: Fwd1d,
    input: &[i32x8],
    out: &mut [i32x8],
    cos_bit: i32,
    stage_range: &[i8],
) {
    match k {
        Fwd1d::Dct4 => av1_fdct4_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Dct8 => av1_fdct8_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Dct16 => av1_fdct16_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Dct32 => av1_fdct32_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Dct64 => av1_fdct64_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Adst4 => av1_fadst4_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Adst8 => av1_fadst8_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Adst16 => av1_fadst16_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Idtx4 => av1_fidentity4_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Idtx8 => av1_fidentity8_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Idtx16 => av1_fidentity16_v3(t, input, out, cos_bit, stage_range),
        Fwd1d::Idtx32 => av1_fidentity32_v3(t, input, out, cos_bit, stage_range),
    }
}

/// Vector column pass of `av1_inv_txfm2d_add` — 8 columns per group.
/// Returns `false` (the caller runs the scalar loop) when the column kernel
/// isn't ported, the width has no full 8-column groups, or SIMD is
/// unavailable / pinned off. On `true` the pass is complete, bit-identical
/// to the scalar loop (module-docs argument + the `tests` differential).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_col_pass(
    txfm_type_col: i32,
    buf: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) -> bool {
    if col_n % 8 != 0 {
        return false;
    }
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    let Some(t) = X64V3Token::summon() else {
        return false;
    };
    let Some(kernel) = inv_kernel(txfm_type_col) else {
        return false;
    };
    debug_assert_eq!(inv_kernel_n(kernel), row_n);
    inv_col_pass(
        t,
        kernel,
        buf,
        output,
        stride,
        col_n,
        row_n,
        shift1_bit,
        col_clamp,
        stage_range,
        ud_flip,
        lr_flip,
        bd,
    );
    true
}

/// Vector ROW pass of `av1_inv_txfm2d_add` — 8 rows per lane batch.
/// Contiguous loads (`mod_input[c*row_n + r..r+8]` — the input is stored
/// column-major), the optional NewInvSqrt2 rect scaling + row clamp, the
/// row kernel, `round_shift_array(-shift[0])`, then the strided store into
/// row-major `buf` via 8x8 transposes (per-lane scatter for the W=4 tail).
/// Returns `false` → caller runs the scalar loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_inv_row_pass(
    txfm_type_row: i32,
    mod_input: &[i32],
    buf: &mut [i32],
    col_n: usize,
    row_n: usize,
    rect1: bool,
    shift0_bit: i32,
    row_clamp: i8,
    stage_range: &[i8; 12],
) -> bool {
    if row_n % 8 != 0 {
        return false;
    }
    let _ = aom_dispatch::scalar_forced();
    let Some(t) = X64V3Token::summon() else {
        return false;
    };
    let Some(kernel) = inv_kernel(txfm_type_row) else {
        return false;
    };
    debug_assert_eq!(inv_kernel_n(kernel), col_n);
    inv_row_pass(
        t, kernel, mod_input, buf, col_n, row_n, rect1, shift0_bit, row_clamp, stage_range,
    );
    true
}

/// The lane-batched inverse row pass body (8 rows per iteration).
#[arcane]
#[allow(clippy::too_many_arguments)]
fn inv_row_pass(
    t: X64V3Token,
    kernel: Inv1d,
    mod_input: &[i32],
    buf: &mut [i32],
    col_n: usize,
    row_n: usize,
    rect1: bool,
    shift0_bit: i32,
    row_clamp: i8,
    stage_range: &[i8; 12],
) {
    debug_assert!(col_n <= 64 && row_n % 8 == 0);
    let mut tin = [i32x8::zero(t); 64];
    let mut tout = [i32x8::zero(t); 64];
    let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;
    for rg in (0..row_n).step_by(8) {
        for (c, ti) in tin[..col_n].iter_mut().enumerate() {
            let mut v = i32x8::from_slice(t, &mod_input[c * row_n + rg..c * row_n + rg + 8]);
            if rect1 {
                // round_shift(x * NewInvSqrt2, NewSqrt2Bits) — the rect scaling.
                v = mul_rshiftv(t, v, NEW_INV_SQRT2, NEW_SQRT2_BITS);
            }
            *ti = clampv(t, v, row_clamp); // the driver's clamp_buf(bd+8)
        }
        run_inv1d(t, kernel, &tin[..col_n], &mut tout[..col_n], cos_bit, stage_range);
        if shift0_bit > 0 {
            // round_shift_array(buf_row, -shift[0]); shift[0] in {0,-1,-2}.
            for to in tout[..col_n].iter_mut() {
                *to = rshiftv(t, *to, shift0_bit);
            }
        }
        // Store: buf[(rg+k)*col_n + c] = tout[c].lane(k) — transpose 8x8
        // tiles for the col_n%8==0 groups, per-lane scatter for the W=4 tail.
        let full = col_n & !7;
        for cg in (0..full).step_by(8) {
            let tr = transpose8(t, &tout[cg..cg + 8]);
            for (k, trk) in tr.iter().enumerate() {
                let base = (rg + k) * col_n + cg;
                trk.store((&mut buf[base..base + 8]).try_into().unwrap());
            }
        }
        for c in full..col_n {
            let a = tout[c].to_array();
            for (k, &av) in a.iter().enumerate() {
                buf[(rg + k) * col_n + c] = av;
            }
        }
    }
}

/// Vector COLUMN pass of `fwd_txfm2d_core` — 8 columns per lane batch.
/// Contiguous i16 loads (`input[src_r*stride + c..c+8]`), the negative-bit
/// `round_shift_array` input stage (`v << 2` i64-clamped), the col kernel,
/// `round_shift_array(-shift[1])`, then contiguous stores into row-major
/// `buf` (lane-reversed at the mirrored position under `lr_flip`).
/// Returns `false` → caller runs the scalar loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_col_pass(
    txfm_type_col: i32,
    input: &[i16],
    buf: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
    shift1_bit: i32,
    cos_bit_col: i32,
    ud_flip: bool,
    lr_flip: bool,
) -> bool {
    if col_n % 8 != 0 {
        return false;
    }
    let _ = aom_dispatch::scalar_forced();
    let Some(t) = X64V3Token::summon() else {
        return false;
    };
    let Some(kernel) = fwd_kernel(txfm_type_col) else {
        return false;
    };
    debug_assert_eq!(fwd_kernel_n(kernel), row_n); // col kernel spans the H points
    fwd_col_pass(
        t, kernel, input, buf, stride, col_n, row_n, shift0, shift1_bit, cos_bit_col, ud_flip,
        lr_flip,
    );
    true
}

/// The lane-batched forward column pass body (8 columns per iteration).
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fwd_col_pass(
    t: X64V3Token,
    kernel: Fwd1d,
    input: &[i16],
    buf: &mut [i32],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift0: i32,
    shift1_bit: i32,
    cos_bit_col: i32,
    ud_flip: bool,
    lr_flip: bool,
) {
    debug_assert!(row_n <= 64 && col_n % 8 == 0);
    let mut tin = [i32x8::zero(t); 64];
    let mut tout = [i32x8::zero(t); 64];
    let sr = [0i8; 12]; // fwd kernels ignore stage_range
    for cg in (0..col_n).step_by(8) {
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let src_r = if ud_flip { row_n - r - 1 } else { r };
            let mut v = widen16(t, &input[src_r * stride + cg..src_r * stride + cg + 8]);
            if shift0 > 0 {
                // round_shift_array(temp_in, -shift[0]) with shift[0]=2 →
                // the NEGATIVE-bit arm: (v << 2) clamped to i32 in i64.
                v = shl_clamp64v(t, v, shift0);
            }
            *ti = v;
        }
        run_fwd1d(t, kernel, &tin[..row_n], &mut tout[..row_n], cos_bit_col, &sr);
        for (r, to) in tout[..row_n].iter_mut().enumerate() {
            let v = if shift1_bit > 0 { rshiftv(t, *to, shift1_bit) } else { *to };
            // Scalar: buf[r*col_n + dst_c] = temp_out[r], dst_c lr-flipped.
            if lr_flip {
                let base = r * col_n + (col_n - cg - 8);
                revv(t, v).store((&mut buf[base..base + 8]).try_into().unwrap());
            } else {
                let base = r * col_n + cg;
                v.store((&mut buf[base..base + 8]).try_into().unwrap());
            }
        }
    }
}

/// Vector ROW pass of `fwd_txfm2d_core` — 8 rows per lane batch. Strided
/// loads from row-major `buf` via 8x8 transposes (per-lane gather for the
/// W=4 tail), the row kernel, `round_shift_array(-shift[2])`, the optional
/// NewSqrt2 rect scaling (AFTER the shift, matching the scalar order), then
/// contiguous stores (`output[c*row_n + r..r+8]` — output is column-major).
/// Returns `false` → caller runs the scalar loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fwd_row_pass(
    txfm_type_row: i32,
    buf: &[i32],
    output: &mut [i32],
    col_n: usize,
    row_n: usize,
    shift2_bit: i32,
    cos_bit_row: i32,
    rect1: bool,
) -> bool {
    if row_n % 8 != 0 {
        return false;
    }
    let _ = aom_dispatch::scalar_forced();
    let Some(t) = X64V3Token::summon() else {
        return false;
    };
    let Some(kernel) = fwd_kernel(txfm_type_row) else {
        return false;
    };
    debug_assert_eq!(fwd_kernel_n(kernel), col_n); // row kernel spans the W points
    fwd_row_pass(t, kernel, buf, output, col_n, row_n, shift2_bit, cos_bit_row, rect1);
    true
}

/// The lane-batched forward row pass body (8 rows per iteration).
#[arcane]
#[allow(clippy::too_many_arguments)]
fn fwd_row_pass(
    t: X64V3Token,
    kernel: Fwd1d,
    buf: &[i32],
    output: &mut [i32],
    col_n: usize,
    row_n: usize,
    shift2_bit: i32,
    cos_bit_row: i32,
    rect1: bool,
) {
    debug_assert!(col_n <= 64 && row_n % 8 == 0);
    let mut tin = [i32x8::zero(t); 64];
    let mut tout = [i32x8::zero(t); 64];
    let sr = [0i8; 12];
    for rg in (0..row_n).step_by(8) {
        // Load: tin[c].lane(k) = buf[(rg+k)*col_n + c] — transpose 8x8 tiles
        // (contiguous row loads), per-lane gather for the W=4 tail.
        let full = col_n & !7;
        for cg in (0..full).step_by(8) {
            let mut rows = [i32x8::zero(t); 8];
            for (k, rk) in rows.iter_mut().enumerate() {
                let base = (rg + k) * col_n + cg;
                *rk = i32x8::from_slice(t, &buf[base..base + 8]);
            }
            let tr = transpose8(t, &rows);
            tin[cg..cg + 8].copy_from_slice(&tr);
        }
        for c in full..col_n {
            tin[c] = i32x8::from_array(t, core::array::from_fn(|k| buf[(rg + k) * col_n + c]));
        }
        run_fwd1d(t, kernel, &tin[..col_n], &mut tout[..col_n], cos_bit_row, &sr);
        for (c, to) in tout[..col_n].iter_mut().enumerate() {
            let mut v = *to;
            if shift2_bit > 0 {
                v = rshiftv(t, v, shift2_bit); // round_shift_array(-shift[2])
            }
            if rect1 {
                // round_shift(v * NewSqrt2, NewSqrt2Bits) — AFTER the shift.
                v = mul_rshiftv(t, v, NEW_SQRT2, NEW_SQRT2_BITS);
            }
            // Scalar: output[c*row_n + r] = row_buffer[c] — contiguous per c.
            let base = c * row_n + rg;
            v.store((&mut output[base..base + 8]).try_into().unwrap());
        }
    }
}

/// The lane-batched column pass body — the scalar per-column loop of
/// `av1_inv_txfm2d_add`, 8 columns per iteration (module docs carry the
/// per-stage exactness argument).
#[arcane]
#[allow(clippy::too_many_arguments)]
fn inv_col_pass(
    t: X64V3Token,
    kernel: Inv1d,
    buf: &[i32],
    output: &mut [u16],
    stride: usize,
    col_n: usize,
    row_n: usize,
    shift1_bit: i32,
    col_clamp: i8,
    stage_range: &[i8; 12],
    ud_flip: bool,
    lr_flip: bool,
    bd: i32,
) {
    debug_assert!(row_n <= 64 && col_n % 8 == 0);
    let mut tin = [i32x8::zero(t); 64];
    let mut tout = [i32x8::zero(t); 64];
    let zero = i32x8::zero(t);
    let pix_hi = i32x8::splat(t, (1i32 << bd) - 1);
    for c in (0..col_n).step_by(8) {
        // Gather 8 columns: under lr_flip, scalar output column `c+j` reads
        // buf column `col_n-1-(c+j)` — i.e. the ascending 8-column load at
        // `col_n-c-8`, lanes reversed.
        for (r, ti) in tin[..row_n].iter_mut().enumerate() {
            let v = if lr_flip {
                let base = r * col_n + (col_n - c - 8);
                revv(t, i32x8::from_slice(t, &buf[base..base + 8]))
            } else {
                let base = r * col_n + c;
                i32x8::from_slice(t, &buf[base..base + 8])
            };
            *ti = clampv(t, v, col_clamp); // the driver's clamp_buf
        }
        let cos_bit = crate::transform::inv_txfm2d::INV_COS_BIT;
        run_inv1d(t, kernel, &tin[..row_n], &mut tout[..row_n], cos_bit, stage_range);
        // round_shift_array(to, -shift[1]) — shift[1] is always negative for
        // the inverse sizes, so this is the positive-bit arm.
        for to in tout[..row_n].iter_mut() {
            *to = rshiftv(t, *to, shift1_bit);
        }
        // Reconstruction: output row r takes tout[row_n-1-r] under ud_flip.
        for r in 0..row_n {
            let src = tout[if ud_flip { row_n - r - 1 } else { r }];
            let idx = r * stride + c;
            let d: [u16; 8] = output[idx..idx + 8].try_into().unwrap();
            let dv = i32x8::from_array(t, core::array::from_fn(|j| d[j] as i32));
            // (dest + trans) wraps i32 like the scalar wrapping_add, then
            // clamps to the pixel range — `as u16` is exact after the clamp.
            let s = (dv + src).clamp(zero, pix_hi).to_array();
            for (j, &sv) in s.iter().enumerate() {
                output[idx + j] = sv as u16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! SIMD-vs-scalar differential for the lane kernels (Gate-3 parity rule:
    //! integer SIMD MUST be bit-identical to the scalar port) — per the
    //! STATUS.md differential plan: inputs sweep the driver clamp bounds
    //! ±2^(bd+7) for bd 8/10/12 (dense random + the exact boundary values +
    //! sign patterns engineered to maximize |p0 + p1| in half_btf), PLUS
    //! full-range i32 (the lane ops are exact on the whole domain, so the
    //! test asserts the whole domain), × cos_bit 10..=13 × the stage_range
    //! values the drivers pass (16/18/20 per `opt_range`, + the 1-D
    //! harness's 17). Every case runs at every token permutation; a counter
    //! proves the v3 arm actually ran (non-vacuous even under
    //! AOM_FORCE_SCALAR — the permutation harness owns token state).

    use super::*;
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    type ScalarKernel = fn(&[i32], &mut [i32], i32, &[i8]);

    /// One direction-erased kernel id for the test table.
    #[derive(Clone, Copy)]
    enum K {
        I(Inv1d),
        F(Fwd1d),
    }

    struct Case {
        name: &'static str,
        size: usize,
        scalar: ScalarKernel,
        v3: K,
    }

    fn cases() -> Vec<Case> {
        use K::{F, I};
        vec![
            Case { name: "idct4", size: 4, scalar: crate::transform::av1_idct4, v3: I(Inv1d::Dct4) },
            Case { name: "idct8", size: 8, scalar: crate::transform::av1_idct8, v3: I(Inv1d::Dct8) },
            Case { name: "idct16", size: 16, scalar: crate::transform::av1_idct16, v3: I(Inv1d::Dct16) },
            Case { name: "idct32", size: 32, scalar: crate::transform::av1_idct32, v3: I(Inv1d::Dct32) },
            Case { name: "idct64", size: 64, scalar: crate::transform::av1_idct64, v3: I(Inv1d::Dct64) },
            Case { name: "iadst4", size: 4, scalar: crate::transform::av1_iadst4, v3: I(Inv1d::Adst4) },
            Case { name: "iadst8", size: 8, scalar: crate::transform::av1_iadst8, v3: I(Inv1d::Adst8) },
            Case { name: "iadst16", size: 16, scalar: crate::transform::av1_iadst16, v3: I(Inv1d::Adst16) },
            Case { name: "iidentity4", size: 4, scalar: crate::transform::av1_iidentity4, v3: I(Inv1d::Idtx4) },
            Case { name: "iidentity8", size: 8, scalar: crate::transform::av1_iidentity8, v3: I(Inv1d::Idtx8) },
            Case {
                name: "iidentity16",
                size: 16,
                scalar: crate::transform::av1_iidentity16,
                v3: I(Inv1d::Idtx16),
            },
            Case {
                name: "iidentity32",
                size: 32,
                scalar: crate::transform::av1_iidentity32,
                v3: I(Inv1d::Idtx32),
            },
            Case { name: "fdct4", size: 4, scalar: crate::transform::av1_fdct4, v3: F(Fwd1d::Dct4) },
            Case { name: "fdct8", size: 8, scalar: crate::transform::av1_fdct8, v3: F(Fwd1d::Dct8) },
            Case { name: "fdct16", size: 16, scalar: crate::transform::av1_fdct16, v3: F(Fwd1d::Dct16) },
            Case { name: "fdct32", size: 32, scalar: crate::transform::av1_fdct32, v3: F(Fwd1d::Dct32) },
            Case { name: "fdct64", size: 64, scalar: crate::transform::av1_fdct64, v3: F(Fwd1d::Dct64) },
            Case { name: "fadst4", size: 4, scalar: crate::transform::av1_fadst4, v3: F(Fwd1d::Adst4) },
            Case { name: "fadst8", size: 8, scalar: crate::transform::av1_fadst8, v3: F(Fwd1d::Adst8) },
            Case { name: "fadst16", size: 16, scalar: crate::transform::av1_fadst16, v3: F(Fwd1d::Adst16) },
            Case { name: "fidentity4", size: 4, scalar: crate::transform::av1_fidentity4, v3: F(Fwd1d::Idtx4) },
            Case { name: "fidentity8", size: 8, scalar: crate::transform::av1_fidentity8, v3: F(Fwd1d::Idtx8) },
            Case {
                name: "fidentity16",
                size: 16,
                scalar: crate::transform::av1_fidentity16,
                v3: F(Fwd1d::Idtx16),
            },
            Case {
                name: "fidentity32",
                size: 32,
                scalar: crate::transform::av1_fidentity32,
                v3: F(Fwd1d::Idtx32),
            },
        ]
    }

    /// Run one lane batch through the selected v3 kernel (the test-side
    /// arcane entry — kernels are `#[target_feature]` fns and cannot be
    /// stored as plain fn pointers).
    #[arcane]
    fn run_v3(
        t: X64V3Token,
        k: K,
        input: &[i32x8],
        out: &mut [i32x8],
        cos_bit: i32,
        stage_range: &[i8],
    ) {
        match k {
            K::I(k) => run_inv1d(t, k, input, out, cos_bit, stage_range),
            K::F(k) => run_fwd1d(t, k, input, out, cos_bit, stage_range),
        }
    }

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
        /// Uniform in [-(1<<bits), 1<<bits] (the driver clamp domains).
        fn bounded(&mut self, bits: u32) -> i32 {
            let range = (1i64 << (bits + 1)) + 1;
            ((self.next() as i64).rem_euclid(range) - (1i64 << bits)) as i32
        }
    }

    /// Run one 8-column batch through the v3 kernel and the scalar kernel
    /// per column; assert every lane matches.
    fn assert_batch(
        t: X64V3Token,
        case: &Case,
        cols: &[[i32; 8]], // cols[r][lane] — row-major lane batch
        cos_bit: i32,
        stage_range: &[i8],
        label: &str,
    ) {
        let n = case.size;
        let mut vin = vec![i32x8::zero(t); n];
        for (r, c) in cols.iter().enumerate() {
            vin[r] = i32x8::from_array(t, *c);
        }
        let mut vout = vec![i32x8::zero(t); n];
        run_v3(t, case.v3, &vin, &mut vout, cos_bit, stage_range);

        let mut sin = vec![0i32; n];
        let mut sout = vec![0i32; n];
        for lane in 0..8 {
            for r in 0..n {
                sin[r] = cols[r][lane];
            }
            (case.scalar)(&sin, &mut sout, cos_bit, stage_range);
            for r in 0..n {
                assert_eq!(
                    vout[r].to_array()[lane],
                    sout[r],
                    "{}: {label} lane={lane} row={r} cos_bit={cos_bit} sr={} input={sin:?}",
                    case.name,
                    stage_range[0],
                );
            }
        }
    }

    #[test]
    fn inv1d_v3_bit_identical_to_scalar_at_every_tier() {
        // Fire the AOM_FORCE_SCALAR pin (if set) BEFORE the permutation
        // harness — the harness then owns token state, so the v3 arm runs
        // in its enabled permutations in BOTH dispatch modes.
        let _ = aom_dispatch::scalar_forced();
        let mut v3_ran = 0usize;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |tier| {
            let Some(t) = X64V3Token::summon() else {
                return; // scalar-only permutation: nothing to compare
            };
            v3_ran += 1;
            let mut rng = Rng(0x_7ab5_11fe_c0de_0001);
            // Driver stage_range values (opt_range 16/18/20) + the 1-D
            // harness's 17; the drivers pass INV_COS_BIT=12, sweep 10..=13.
            for &sr in &[16i8, 17, 18, 20] {
                let stage_range = [sr; 12];
                for cos_bit in 10..=13 {
                    for case in cases() {
                        let n = case.size;
                        for &bits in &[15u32, 17, 19] {
                            // (a) driver-clamp-domain dense random: the col
                            // pass clamps input to max(bd+6,16) bits, the
                            // row pass to bd+8 — sweep ±2^15/2^17/2^19.
                            for rep in 0..24 {
                                let cols: Vec<[i32; 8]> = (0..n)
                                    .map(|_| core::array::from_fn(|_| rng.bounded(bits)))
                                    .collect();
                                assert_batch(
                                    t,
                                    &case,
                                    &cols,
                                    cos_bit,
                                    &stage_range,
                                    &format!("[{tier}] rand b{bits} rep{rep}"),
                                );
                            }
                            // (b) exact clamp-bound sign patterns — the
                            // half_btf |p0 + p1| maximizers: all +B, all -B,
                            // alternating ±B (both phases), random-ish ±B.
                            let b = 1i32 << bits;
                            let pats: [&dyn Fn(usize, usize) -> i32; 5] = [
                                &|_, _| b,
                                &|_, _| -b,
                                &|r, l| if (r + l) % 2 == 0 { b } else { -b },
                                &|r, l| if (r + l) % 2 == 0 { -b } else { b },
                                &|r, l| if (r * 7 + l * 3) % 5 < 2 { b } else { -b },
                            ];
                            for (pi, pat) in pats.iter().enumerate() {
                                let cols: Vec<[i32; 8]> =
                                    (0..n).map(|r| core::array::from_fn(|l| pat(r, l))).collect();
                                assert_batch(
                                    t,
                                    &case,
                                    &cols,
                                    cos_bit,
                                    &stage_range,
                                    &format!("[{tier}] bound b{bits} pat{pi}"),
                                );
                            }
                        }
                        // (c) FULL-i32 random (the lane ops are exact on the
                        // whole domain — assert it there) + extreme lanes
                        // mixed with all-zero columns.
                        for rep in 0..24 {
                            let cols: Vec<[i32; 8]> = (0..n)
                                .map(|_| core::array::from_fn(|_| rng.next() as i32))
                                .collect();
                            assert_batch(
                                t,
                                &case,
                                &cols,
                                cos_bit,
                                &stage_range,
                                &format!("[{tier}] full-i32 rep{rep}"),
                            );
                        }
                        let mut cols = vec![[0i32; 8]; n];
                        cols[0] = [
                            i32::MIN,
                            i32::MAX,
                            0,
                            -1,
                            1 << 19,
                            -(1 << 19),
                            i32::MIN + 1,
                            i32::MAX - 1,
                        ];
                        cols[n - 1] = [i32::MAX, i32::MIN, 1, 0, -(1 << 19), 1 << 19, -2, 2];
                        assert_batch(
                            t,
                            &case,
                            &cols,
                            cos_bit,
                            &stage_range,
                            &format!("[{tier}] extremes+zero-cols"),
                        );
                    }
                }
            }
        });
        eprintln!("inv1d v3 parity: {report}, v3 permutations run: {v3_ran}");
        assert!(v3_ran >= 1, "the v3 arm must run at least once (AVX2 CI)");
        assert!(report.permutations_run >= 2);
    }
}
