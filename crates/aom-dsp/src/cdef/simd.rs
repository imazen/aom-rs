//! SIMD row kernel for `cdef_filter_block_16` (Gate 3) — bit-identical to the
//! scalar core on the structural CDEF domain, at every dispatch tier
//! (`tests/cdef_filter_simd_diff.rs`).
//!
//! Same aom-rs SIMD pattern as `crate::quant::simd`: ONE magetypes generic
//! kernel (`#[magetypes(v3, neon, wasm128, -scalar)]`), a hand-written
//! `_scalar` variant that IS the transcribed scalar core, `incant!` dispatch,
//! `aom_dispatch::scalar_forced()` pin at the entry.
//!
//! Layout: one `i16x8` vector per block row (width-8 blocks — the luma 8x8
//! path, the bulk of CDEF cost). Width-4 blocks take the scalar core
//! unconditionally for now. All neighbour loads are contiguous 8-lane row
//! loads at the (per-call constant) `cdef_dir` offsets.
//!
//! # Bit-exactness argument (structural domain)
//!
//! The frame walk feeds `in_buf` values that are either real pixels
//! (`<= (1<<bd)-1 <= 4095`) or the border fill `CDEF_VERY_LARGE (16384)`,
//! and strengths from the header (`pri <= 15 << coeff_shift`,
//! `sec <= 4 << coeff_shift`, `coeff_shift = bd-8 <= 4`, damping 3..6 + cs).
//! Within that domain every intermediate fits the scalar core's OWN
//! narrowings, lane for lane:
//! * `diff = p - x` ∈ [-4095, 16384] — exact in i16.
//! * `constrain`: `a = |diff| <= 16384`; `threshold - (a >> shift)` ∈
//!   [-16384, 3840]; the clamp to `[0, a]` and the sign re-apply are lane
//!   min/max/xor-sub — exact. The scalar's `threshold == 0` early-return
//!   equals the formula's value (the clamp floor is 0), and the vector
//!   kernel skips the class entirely in that case (contribution 0) while
//!   still running the min/max updates, exactly like the scalar core.
//! * `tap * constrain` is truncated `as i16` by the scalar core; the lane
//!   `mullo` IS that truncation. The `sum` accumulator wraps i16 in both.
//! * `|sum| <= 12*pri_thr + 12*sec_thr <= 3648` in-domain, so the final
//!   `(8 + sum - (sum<0)) >> 4` fits i16 exactly; `x + adj` wraps i16 in
//!   BOTH implementations (scalar casts through i16 deliberately).
//! * min/max tracking compares nonneg values <= 16384 — exact in i16; the
//!   `p != CDEF_VERY_LARGE` max-exclusion becomes a blend to 0 (never wins
//!   a max whose floor is `x >= 0`); min is unconditional in both.
//!
//! The differential sweeps this domain densely (all bd, all strength/damping
//! combos, VERY_LARGE border mixes) plus the boundary values, at every token
//! permutation.

use archmage::prelude::*;

use crate::cdef::{CDEF_BSTRIDE, CDEF_VERY_LARGE, PRI_TAPS, SEC_TAPS, cdef_dir, constrain, get_msb};

/// Dispatch entry used by [`crate::cdef::cdef_filter_block_16`] for width-8 blocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_16_w8(
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cdef_filter_16_w8_impl(
            dst,
            dst_off,
            dstride,
            in_buf,
            in_off,
            pri_strength,
            sec_strength,
            dir,
            pri_damping,
            sec_damping,
            coeff_shift,
            block_height,
            enable_primary,
            enable_secondary
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed core, verbatim (via the width-8 store shape).
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w8_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    crate::cdef::cdef_filter_block_core(
        in_buf,
        in_off,
        pri_strength,
        sec_strength,
        dir,
        pri_damping,
        sec_damping,
        coeff_shift,
        8,
        block_height,
        enable_primary,
        enable_secondary,
        |i, j, y| dst[dst_off + i * dstride + j] = y as u16,
    );
}

/// `a >> shift` for a per-call runtime shift in 0..=15 (lane-const shifts
/// only exist as const generics; the match is perfectly predicted since the
/// shift is fixed per block).
macro_rules! shr_by {
    ($v:expr, $sh:expr) => {
        match $sh {
            0 => $v,
            1 => $v.shr_arithmetic_const::<1>(),
            2 => $v.shr_arithmetic_const::<2>(),
            3 => $v.shr_arithmetic_const::<3>(),
            4 => $v.shr_arithmetic_const::<4>(),
            5 => $v.shr_arithmetic_const::<5>(),
            6 => $v.shr_arithmetic_const::<6>(),
            7 => $v.shr_arithmetic_const::<7>(),
            8 => $v.shr_arithmetic_const::<8>(),
            9 => $v.shr_arithmetic_const::<9>(),
            10 => $v.shr_arithmetic_const::<10>(),
            11 => $v.shr_arithmetic_const::<11>(),
            12 => $v.shr_arithmetic_const::<12>(),
            13 => $v.shr_arithmetic_const::<13>(),
            14 => $v.shr_arithmetic_const::<14>(),
            _ => $v.shr_arithmetic_const::<15>(),
        }
    };
}

/// Width-4 dispatch entry (two rows per 8-lane vector; `block_height` must be
/// even — the caller routes odd heights to the scalar core).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_16_w4(
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        cdef_filter_16_w4_impl(
            dst,
            dst_off,
            dstride,
            in_buf,
            in_off,
            pri_strength,
            sec_strength,
            dir,
            pri_damping,
            sec_damping,
            coeff_shift,
            block_height,
            enable_primary,
            enable_secondary
        ),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the transcribed core, verbatim (width-4 store shape).
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w4_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    crate::cdef::cdef_filter_block_core(
        in_buf,
        in_off,
        pri_strength,
        sec_strength,
        dir,
        pri_damping,
        sec_damping,
        coeff_shift,
        4,
        block_height,
        enable_primary,
        enable_secondary,
        |i, j, y| dst[dst_off + i * dstride + j] = y as u16,
    );
}

#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w4_impl(
    token: Token,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    assert!(block_height % 2 == 0, "caller routes odd heights to scalar");
    let clipping_required = enable_primary && enable_secondary;
    let s = CDEF_BSTRIDE as i32;
    let pri_taps = &PRI_TAPS[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = &SEC_TAPS;
    let pri_shift = if pri_strength != 0 {
        (pri_damping - get_msb(pri_strength as u32)).max(0)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        (sec_damping - get_msb(sec_strength as u32)).max(0)
    } else {
        0
    };

    let zero = i16x8::zero(token);
    let eight = i16x8::splat(token, 8);
    let vl = i16x8::splat(token, CDEF_VERY_LARGE as i16);
    let pri_t = i16x8::splat(token, pri_strength as i16);
    let sec_t = i16x8::splat(token, sec_strength as i16);

    let constrain_v = |d: i16x8, thr: i16x8, shift: i32| -> i16x8 {
        let m = d.shr_arithmetic_const::<15>();
        let a = (d ^ m) - m;
        let c = (thr - shr_by!(a, shift)).clamp(zero, a);
        (c ^ m) - m
    };

    // Two-row gather: lanes [row_i .. 4px, row_i+1 .. 4px].
    let load2 = |idx: i32| -> i16x8 {
        let a = idx as usize;
        let b = (idx + s) as usize;
        let mut arr = [0u16; 8];
        arr[..4].copy_from_slice(&in_buf[a..a + 4]);
        arr[4..].copy_from_slice(&in_buf[b..b + 4]);
        u16x8::from_array(token, arr).bitcast_i16x8()
    };

    let mut i = 0i32;
    while (i as usize) < block_height {
        let base = in_off as i32 + i * s;
        let x = load2(base);
        let mut sum = zero;
        let mut maxv = x;
        let mut minv = x;
        for k in 0..2usize {
            if enable_primary {
                let off = cdef_dir(dir, k);
                let p0 = load2(base + off);
                let p1 = load2(base - off);
                if pri_strength != 0 {
                    let tap = i16x8::splat(token, pri_taps[k] as i16);
                    sum = sum + tap * constrain_v(p0 - x, pri_t, pri_shift);
                    sum = sum + tap * constrain_v(p1 - x, pri_t, pri_shift);
                }
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(p0.simd_eq(vl), zero, p0));
                    maxv = maxv.max(i16x8::blend(p1.simd_eq(vl), zero, p1));
                    minv = minv.min(p0);
                    minv = minv.min(p1);
                }
            }
            if enable_secondary {
                let o0 = cdef_dir(dir + 2, k);
                let o1 = cdef_dir(dir - 2, k);
                let s0 = load2(base + o0);
                let s1 = load2(base - o0);
                let s2 = load2(base + o1);
                let s3 = load2(base - o1);
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(s0.simd_eq(vl), zero, s0));
                    maxv = maxv.max(i16x8::blend(s1.simd_eq(vl), zero, s1));
                    maxv = maxv.max(i16x8::blend(s2.simd_eq(vl), zero, s2));
                    maxv = maxv.max(i16x8::blend(s3.simd_eq(vl), zero, s3));
                    minv = minv.min(s0).min(s1).min(s2).min(s3);
                }
                if sec_strength != 0 {
                    let tap = i16x8::splat(token, sec_taps[k] as i16);
                    sum = sum + tap * constrain_v(s0 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s1 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s2 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s3 - x, sec_t, sec_shift);
                }
            }
        }
        let m = sum.shr_arithmetic_const::<15>();
        let adj = (sum + m + eight).shr_arithmetic_const::<4>();
        let mut y = x + adj;
        if clipping_required {
            y = y.max(minv).min(maxv);
        }
        let out = y.bitcast_u16x8().to_array();
        let r0 = dst_off + i as usize * dstride;
        let r1 = dst_off + (i as usize + 1) * dstride;
        dst[r0..r0 + 4].copy_from_slice(&out[..4]);
        dst[r1..r1 + 4].copy_from_slice(&out[4..]);
        i += 2;
    }
}

#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn cdef_filter_16_w8_impl(
    token: Token,
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    in_buf: &[u16],
    in_off: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: i32,
    pri_damping: i32,
    sec_damping: i32,
    coeff_shift: i32,
    block_height: usize,
    enable_primary: bool,
    enable_secondary: bool,
) {
    let clipping_required = enable_primary && enable_secondary;
    let s = CDEF_BSTRIDE as i32;
    let pri_taps = &PRI_TAPS[((pri_strength >> coeff_shift) & 1) as usize];
    let sec_taps = &SEC_TAPS;
    // Per-call constants of constrain(): shift = max(0, damping - msb(thr)).
    // Guarded: msb(0) is undefined — a zero threshold contributes 0 and the
    // vector kernel skips that class's constrain entirely (see module docs).
    let pri_shift = if pri_strength != 0 {
        (pri_damping - get_msb(pri_strength as u32)).max(0)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        (sec_damping - get_msb(sec_strength as u32)).max(0)
    } else {
        0
    };

    let zero = i16x8::zero(token);
    let eight = i16x8::splat(token, 8);
    let vl = i16x8::splat(token, CDEF_VERY_LARGE as i16);
    let pri_t = i16x8::splat(token, pri_strength as i16);
    let sec_t = i16x8::splat(token, sec_strength as i16);

    // constrain() on 8 lanes: sign*(clamp(thr - (|d|>>shift), 0, |d|)).
    // (thr == 0 is handled by the caller skipping the class.)
    let constrain_v = |d: i16x8, thr: i16x8, shift: i32| -> i16x8 {
        let m = d.shr_arithmetic_const::<15>(); // -1 where negative
        let a = (d ^ m) - m; // |d| (wrapping, like the scalar core's i32 path in-domain)
        let c = (thr - shr_by!(a, shift)).clamp(zero, a);
        (c ^ m) - m
    };

    let load = |idx: i32| -> i16x8 {
        u16x8::from_slice(token, &in_buf[idx as usize..idx as usize + 8]).bitcast_i16x8()
    };

    for i in 0..block_height as i32 {
        let base = in_off as i32 + i * s;
        let x = load(base);
        let mut sum = zero;
        let mut maxv = x;
        let mut minv = x;
        for k in 0..2usize {
            if enable_primary {
                let off = cdef_dir(dir, k);
                let p0 = load(base + off);
                let p1 = load(base - off);
                if pri_strength != 0 {
                    let tap = i16x8::splat(token, pri_taps[k] as i16);
                    sum = sum + tap * constrain_v(p0 - x, pri_t, pri_shift);
                    sum = sum + tap * constrain_v(p1 - x, pri_t, pri_shift);
                }
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(p0.simd_eq(vl), zero, p0));
                    maxv = maxv.max(i16x8::blend(p1.simd_eq(vl), zero, p1));
                    minv = minv.min(p0);
                    minv = minv.min(p1);
                }
            }
            if enable_secondary {
                let o0 = cdef_dir(dir + 2, k);
                let o1 = cdef_dir(dir - 2, k);
                let s0 = load(base + o0);
                let s1 = load(base - o0);
                let s2 = load(base + o1);
                let s3 = load(base - o1);
                if clipping_required {
                    maxv = maxv.max(i16x8::blend(s0.simd_eq(vl), zero, s0));
                    maxv = maxv.max(i16x8::blend(s1.simd_eq(vl), zero, s1));
                    maxv = maxv.max(i16x8::blend(s2.simd_eq(vl), zero, s2));
                    maxv = maxv.max(i16x8::blend(s3.simd_eq(vl), zero, s3));
                    minv = minv.min(s0).min(s1).min(s2).min(s3);
                }
                if sec_strength != 0 {
                    let tap = i16x8::splat(token, sec_taps[k] as i16);
                    sum = sum + tap * constrain_v(s0 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s1 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s2 - x, sec_t, sec_shift);
                    sum = sum + tap * constrain_v(s3 - x, sec_t, sec_shift);
                }
            }
        }
        // y = x + ((8 + sum - (sum<0)) >> 4), computed exactly as the scalar
        // core does through i16 (in-domain |sum| <= 3648 — see module docs).
        let m = sum.shr_arithmetic_const::<15>(); // -1 where sum < 0
        let adj = (sum + m + eight).shr_arithmetic_const::<4>();
        let mut y = x + adj;
        if clipping_required {
            y = y.max(minv).min(maxv);
        }
        let row = dst_off + i as usize * dstride;
        y.bitcast_u16x8()
            .store((&mut dst[row..row + 8]).try_into().unwrap());
    }
}

