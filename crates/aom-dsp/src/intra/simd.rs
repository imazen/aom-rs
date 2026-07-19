//! SIMD row kernels for the non-directional highbd intra predictors (Gate 3)
//! — bit-identical to the scalar core at every dispatch tier
//! (`tests/intra_simd_diff.rs`).
//!
//! Same aom-rs SIMD pattern as `crate::cdef` / `crate::loopfilter` / `aom_txb`: ONE
//! magetypes generic kernel (`#[magetypes(define(i32x8), v3, neon, wasm128,
//! -scalar)]`), a hand-written `_scalar` variant that IS the transcribed
//! scalar core, `incant!` dispatch, `aom_dispatch::scalar_forced()` pin at the
//! entry.
//!
//! # Scope
//!
//! The arithmetic predictors — SMOOTH / SMOOTH_V / SMOOTH_H (per-pixel weighted
//! blends) and PAETH (per-pixel base-distance select). These are `predict_highbd`'s
//! compute-heavy modes. The pure-movement modes (DC family fill, V copy, H
//! per-row fill) are memset/memcpy slice ops in [`crate::intra::predict_highbd`] — the
//! optimal, byte-trivially-safe form for a fill/copy (glibc AVX2 memset/memcpy),
//! with no dispatch needed because a store cannot perturb bytes.
//!
//! # Layout
//!
//! Each kernel vectorizes over the block's COLUMNS: one `i32x8` per 8 columns,
//! looping rows. magetypes has no widening load, so the column-varying `u16`
//! samples (the above row) and `u8` smooth weights are pre-widened once per
//! block into a stack `[i32; 64]` (block width ≤ 64) and read with
//! `i32x8::from_slice`. Widths are all multiples of 4 (`TX_W`/`TX_H`); the
//! `full = bw & !7` columns run the vector body, the trailing `bw - full`
//! (only 4, and only when `bw == 4`) run the scalar tail — which IS the scalar
//! core, so it is bit-exact by construction. A dedicated 2-rows-per-vector
//! path for `bw == 4` blocks is a documented follow-up (small blocks carry
//! little of the per-block pixel volume).
//!
//! # Bit-exactness (full highbd domain, bd 8/10/12)
//!
//! The scalar core does all of this predictor arithmetic in `i32` on `u16`
//! samples (`≤ (1<<bd)-1 ≤ 4095`); this kernel runs the SAME `i32` math in
//! lanes, term for term, in the SAME association order, so it reproduces the
//! scalar result lane-for-lane:
//!
//! * SMOOTH: `p = wh*above + (256-wh)*below + ww*left + (256-ww)*right` with
//!   `wh, ww ∈ [0,255]`, samples `∈ [0,4095]` → each product `≤ 255*4095 ≈ 1.04M`,
//!   the 4-term sum `≤ ~4.19M`, all inside `i32`. Every term is non-negative
//!   (`256 - w ≥ 1`), so `p ≥ 0` and the rounding shift `(p + 256) >> 9`
//!   (`shr_arithmetic_const::<9>`) equals the scalar `divide_round(p, 9)` (an
//!   arithmetic shift of a non-negative value == the scalar `>>`). SMOOTH_V /
//!   SMOOTH_H are the 2-term analogues with `>> 8`.
//! * PAETH: `base = top + left - top_left ∈ [-4095, 8190]`; `abs_diff(a,b) =
//!   |a-b|` is `(a-b).abs()` with `|a-b| ≤ 8190` (never the `i32::MIN` corner);
//!   the `<=` chain (`p_left<=p_top && p_left<=p_top_left` → left, else
//!   `p_top<=p_top_left` → top, else top_left) becomes two `simd_le` masks and a
//!   nested `blend(mask, if_true, if_false)` — the exact scalar selection.
//! * The final value is always `∈ [0,4095]`, so the `as u16` narrowing store is
//!   exact (no truncation).
//!
//! `bd == 8` runs this same path (samples are `u16` at every bit depth); the
//! differential proves the dispatch entry against the scalar core at every token
//! tier over bd 8/10/12.

// The predictors carry the reference edges as separate slices (above / left /
// weights) plus the block geometry — inherently many arguments; and the scalar
// cores are the verbatim indexed transcriptions (kept index-for-index faithful
// to the C loops), so range-loop indexing is intentional.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use archmage::prelude::*;

use crate::intra::weights::SMOOTH_WEIGHT_LOG2_SCALE;

// The rounding shift constants (`9` for SMOOTH, `8` for SMOOTH_V/H) and the
// smooth scale (`256`) are baked as literals in the vector kernels; lock the
// table's scale so a future weight-table edit can't silently desync them.
const _: () = assert!(SMOOTH_WEIGHT_LOG2_SCALE == 8);

const SCALE: i32 = 1 << SMOOTH_WEIGHT_LOG2_SCALE; // 256

// ===========================================================================
// SMOOTH
// ===========================================================================

/// Dispatch entry for the SMOOTH predictor (highbd `u16`). `above_row` is the
/// `bw` above samples (`above.at(0..bw)`), `left` the `bh` left samples,
/// `sw_w`/`sw_h` the `SMOOTH_WEIGHTS` slices for the block's width/height.
pub(crate) fn smooth(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    sw_w: &[u8],
    sw_h: &[u8],
) {
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        smooth_impl(dst, stride, bw, bh, above_row, left, sw_w, sw_h),
        [v3, neon, wasm128, scalar]
    )
}

/// SMOOTH scalar core — verbatim transcription (the differential reference AND
/// the scalar dispatch tier).
pub(crate) fn smooth_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    sw_w: &[u8],
    sw_h: &[u8],
) {
    let below = left[bh - 1] as i32;
    let right = above_row[bw - 1] as i32;
    let log2 = 1 + SMOOTH_WEIGHT_LOG2_SCALE;
    for r in 0..bh {
        for c in 0..bw {
            let wh = sw_h[r] as i32;
            let ww = sw_w[c] as i32;
            let p = wh * above_row[c] as i32
                + (SCALE - wh) * below
                + ww * left[r] as i32
                + (SCALE - ww) * right;
            dst[r * stride + c] = crate::intra::divide_round(p, log2) as u16;
        }
    }
}

fn smooth_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    sw_w: &[u8],
    sw_h: &[u8],
) {
    smooth_scalar(dst, stride, bw, bh, above_row, left, sw_w, sw_h);
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
fn smooth_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    sw_w: &[u8],
    sw_h: &[u8],
) {
    let below = left[bh - 1] as i32;
    let right = above_row[bw - 1] as i32;
    let full = bw & !7;

    // Pre-widen the column-varying inputs once (magetypes has no widening load).
    let mut above_i32 = [0i32; 64];
    let mut sww_i32 = [0i32; 64];
    for c in 0..bw {
        above_i32[c] = above_row[c] as i32;
        sww_i32[c] = sw_w[c] as i32;
    }

    let scale_v = i32x8::splat(token, SCALE);
    let right_v = i32x8::splat(token, right);
    let round = i32x8::splat(token, 1 << SMOOTH_WEIGHT_LOG2_SCALE); // 1<<(9-1)=256

    for r in 0..bh {
        let wh = sw_h[r] as i32;
        let left_r = left[r] as i32;
        let wh_v = i32x8::splat(token, wh);
        let below_term = i32x8::splat(token, (SCALE - wh) * below);
        let left_r_v = i32x8::splat(token, left_r);
        let row = r * stride;
        let mut c = 0;
        while c < full {
            let above_v = i32x8::from_slice(token, &above_i32[c..c + 8]);
            let ww_v = i32x8::from_slice(token, &sww_i32[c..c + 8]);
            // p = wh*above + (256-wh)*below + ww*left_r + (256-ww)*right
            let p = wh_v * above_v + below_term + ww_v * left_r_v + (scale_v - ww_v) * right_v;
            let out = (p + round).shr_arithmetic_const::<9>();
            let a = out.to_array();
            for (dv, &av) in dst[row + c..row + c + 8].iter_mut().zip(a.iter()) {
                *dv = av as u16;
            }
            c += 8;
        }
        while c < bw {
            let ww = sw_w[c] as i32;
            let p = wh * above_row[c] as i32
                + (SCALE - wh) * below
                + ww * left_r
                + (SCALE - ww) * right;
            dst[row + c] = crate::intra::divide_round(p, 1 + SMOOTH_WEIGHT_LOG2_SCALE) as u16;
            c += 1;
        }
    }
}

// ===========================================================================
// SMOOTH_V
// ===========================================================================

/// Dispatch entry for SMOOTH_V. `below = left[bh-1]`, `sw_h` the height weights.
pub(crate) fn smooth_v(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    below: i32,
    sw_h: &[u8],
) {
    let _ = aom_dispatch::scalar_forced();
    incant!(
        smooth_v_impl(dst, stride, bw, bh, above_row, below, sw_h),
        [v3, neon, wasm128, scalar]
    )
}

/// SMOOTH_V scalar core — verbatim transcription.
pub(crate) fn smooth_v_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    below: i32,
    sw_h: &[u8],
) {
    let log2 = SMOOTH_WEIGHT_LOG2_SCALE;
    for r in 0..bh {
        let w = sw_h[r] as i32;
        for c in 0..bw {
            let p = w * above_row[c] as i32 + (SCALE - w) * below;
            dst[r * stride + c] = crate::intra::divide_round(p, log2) as u16;
        }
    }
}

fn smooth_v_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    below: i32,
    sw_h: &[u8],
) {
    smooth_v_scalar(dst, stride, bw, bh, above_row, below, sw_h);
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
fn smooth_v_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    below: i32,
    sw_h: &[u8],
) {
    let full = bw & !7;
    let mut above_i32 = [0i32; 64];
    for c in 0..bw {
        above_i32[c] = above_row[c] as i32;
    }
    let round = i32x8::splat(token, 1 << (SMOOTH_WEIGHT_LOG2_SCALE - 1)); // 1<<(8-1)=128
    for r in 0..bh {
        let w = sw_h[r] as i32;
        let w_v = i32x8::splat(token, w);
        let below_term = i32x8::splat(token, (SCALE - w) * below);
        let row = r * stride;
        let mut c = 0;
        while c < full {
            let above_v = i32x8::from_slice(token, &above_i32[c..c + 8]);
            let p = w_v * above_v + below_term;
            let out = (p + round).shr_arithmetic_const::<8>();
            let a = out.to_array();
            for (dv, &av) in dst[row + c..row + c + 8].iter_mut().zip(a.iter()) {
                *dv = av as u16;
            }
            c += 8;
        }
        while c < bw {
            let p = w * above_row[c] as i32 + (SCALE - w) * below;
            dst[row + c] = crate::intra::divide_round(p, SMOOTH_WEIGHT_LOG2_SCALE) as u16;
            c += 1;
        }
    }
}

// ===========================================================================
// SMOOTH_H
// ===========================================================================

/// Dispatch entry for SMOOTH_H. `right = above.at(bw-1)`, `sw_w` the width weights.
pub(crate) fn smooth_h(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &[u16],
    right: i32,
    sw_w: &[u8],
) {
    let _ = aom_dispatch::scalar_forced();
    incant!(
        smooth_h_impl(dst, stride, bw, bh, left, right, sw_w),
        [v3, neon, wasm128, scalar]
    )
}

/// SMOOTH_H scalar core — verbatim transcription.
pub(crate) fn smooth_h_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &[u16],
    right: i32,
    sw_w: &[u8],
) {
    let log2 = SMOOTH_WEIGHT_LOG2_SCALE;
    for r in 0..bh {
        for c in 0..bw {
            let w = sw_w[c] as i32;
            let p = w * left[r] as i32 + (SCALE - w) * right;
            dst[r * stride + c] = crate::intra::divide_round(p, log2) as u16;
        }
    }
}

fn smooth_h_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &[u16],
    right: i32,
    sw_w: &[u8],
) {
    smooth_h_scalar(dst, stride, bw, bh, left, right, sw_w);
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
fn smooth_h_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &[u16],
    right: i32,
    sw_w: &[u8],
) {
    let full = bw & !7;
    let mut sww_i32 = [0i32; 64];
    for c in 0..bw {
        sww_i32[c] = sw_w[c] as i32;
    }
    let scale_v = i32x8::splat(token, SCALE);
    let right_v = i32x8::splat(token, right);
    let round = i32x8::splat(token, 1 << (SMOOTH_WEIGHT_LOG2_SCALE - 1)); // 128
    for r in 0..bh {
        let left_r = left[r] as i32;
        let left_r_v = i32x8::splat(token, left_r);
        let row = r * stride;
        let mut c = 0;
        while c < full {
            let w_v = i32x8::from_slice(token, &sww_i32[c..c + 8]);
            let p = w_v * left_r_v + (scale_v - w_v) * right_v;
            let out = (p + round).shr_arithmetic_const::<8>();
            let a = out.to_array();
            for (dv, &av) in dst[row + c..row + c + 8].iter_mut().zip(a.iter()) {
                *dv = av as u16;
            }
            c += 8;
        }
        while c < bw {
            let w = sw_w[c] as i32;
            let p = w * left_r + (SCALE - w) * right;
            dst[row + c] = crate::intra::divide_round(p, SMOOTH_WEIGHT_LOG2_SCALE) as u16;
            c += 1;
        }
    }
}

// ===========================================================================
// PAETH
// ===========================================================================

/// Dispatch entry for PAETH. `top_left = above.top_left()`.
pub(crate) fn paeth(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    top_left: i32,
) {
    let _ = aom_dispatch::scalar_forced();
    incant!(
        paeth_impl(dst, stride, bw, bh, above_row, left, top_left),
        [v3, neon, wasm128, scalar]
    )
}

/// PAETH scalar core — verbatim transcription.
pub(crate) fn paeth_scalar(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    top_left: i32,
) {
    for r in 0..bh {
        for c in 0..bw {
            dst[r * stride + c] =
                crate::intra::paeth_single_i32(left[r] as i32, above_row[c] as i32, top_left) as u16;
        }
    }
}

fn paeth_impl_scalar(
    _t: archmage::ScalarToken,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    top_left: i32,
) {
    paeth_scalar(dst, stride, bw, bh, above_row, left, top_left);
}

#[magetypes(define(i32x8), v3, neon, wasm128, -scalar)]
fn paeth_impl(
    token: Token,
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above_row: &[u16],
    left: &[u16],
    top_left: i32,
) {
    let full = bw & !7;
    let mut above_i32 = [0i32; 64];
    for c in 0..bw {
        above_i32[c] = above_row[c] as i32;
    }
    let tl_v = i32x8::splat(token, top_left);
    for r in 0..bh {
        let left_r = left[r] as i32;
        let left_v = i32x8::splat(token, left_r);
        let row = r * stride;
        let mut c = 0;
        while c < full {
            let top = i32x8::from_slice(token, &above_i32[c..c + 8]);
            // base = top + left - top_left; distances to each; nearest wins.
            let base = top + left_v - tl_v;
            let p_left = (base - left_v).abs();
            let p_top = (base - top).abs();
            let p_tl = (base - tl_v).abs();
            // scalar: p_left<=p_top && p_left<=p_tl -> left;
            //         else p_top<=p_tl -> top; else top_left.
            let use_left = p_left.simd_le(p_top) & p_left.simd_le(p_tl);
            let use_top = p_top.simd_le(p_tl);
            let sel = i32x8::blend(use_left, left_v, i32x8::blend(use_top, top, tl_v));
            let a = sel.to_array();
            for (dv, &av) in dst[row + c..row + c + 8].iter_mut().zip(a.iter()) {
                *dv = av as u16;
            }
            c += 8;
        }
        while c < bw {
            dst[row + c] = crate::intra::paeth_single_i32(left_r, above_row[c] as i32, top_left) as u16;
            c += 1;
        }
    }
}
