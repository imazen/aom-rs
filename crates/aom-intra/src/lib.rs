//! aom-intra — bit-exact AV1 intra predictors (port of libaom v3.14.1
//! `aom_dsp/intrapred.c`). Non-directional lowbd family: DC / DC_top / DC_left
//! / DC_128 / V / H / Paeth / Smooth / Smooth_V / Smooth_H, generic over block
//! size. `above` must have `above[-1]` (top-left) valid (index via `AboveRef`).
//!
//! Validated byte-for-byte against C for every (mode × block size).

pub mod dir;
mod weights;
use weights::{SMOOTH_WEIGHTS, SMOOTH_WEIGHT_LOG2_SCALE};

/// Prediction mode indices (must match the shim's `mode` ordering).
pub const DC: usize = 0;
pub const DC_TOP: usize = 1;
pub const DC_LEFT: usize = 2;
pub const DC_128: usize = 3;
pub const V: usize = 4;
pub const H: usize = 5;
pub const PAETH: usize = 6;
pub const SMOOTH: usize = 7;
pub const SMOOTH_V: usize = 8;
pub const SMOOTH_H: usize = 9;

#[inline]
fn divide_round(value: i32, bits: i32) -> i32 {
    (value + (1 << (bits - 1))) >> bits
}

#[inline]
fn abs_diff(a: i32, b: i32) -> i32 {
    if a > b {
        a - b
    } else {
        b - a
    }
}

#[inline]
fn paeth_single(left: i32, top: i32, top_left: i32) -> u8 {
    let base = top + left - top_left;
    let p_left = abs_diff(base, left);
    let p_top = abs_diff(base, top);
    let p_top_left = abs_diff(base, top_left);
    if p_left <= p_top && p_left <= p_top_left {
        left as u8
    } else if p_top <= p_top_left {
        top as u8
    } else {
        top_left as u8
    }
}

/// A view over the `above` row that also exposes the top-left sample at index -1
/// (like the C `above[-1]`). `data[0]` is the top-left; `above(i)` reads `[i]`.
pub struct AboveRef<'a>(pub &'a [u8]);
impl AboveRef<'_> {
    #[inline]
    fn at(&self, i: usize) -> i32 {
        self.0[i + 1] as i32
    }
    #[inline]
    fn top_left(&self) -> i32 {
        self.0[0] as i32
    }
}

/// Run intra predictor `mode` into `dst` (row-major, `stride` per row).
/// `above` includes the top-left sample at slot 0; `left` is `bh` samples.
pub fn predict(
    mode: usize,
    dst: &mut [u8],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &AboveRef,
    left: &[u8],
) {
    match mode {
        DC => {
            let count = (bw + bh) as i32;
            let mut sum = 0i32;
            for i in 0..bw {
                sum += above.at(i);
            }
            for &l in left.iter().take(bh) {
                sum += l as i32;
            }
            let dc = ((sum + (count >> 1)) / count) as u8;
            fill(dst, stride, bw, bh, dc);
        }
        DC_TOP => {
            let mut sum = 0i32;
            for i in 0..bw {
                sum += above.at(i);
            }
            let dc = ((sum + (bw as i32 >> 1)) / bw as i32) as u8;
            fill(dst, stride, bw, bh, dc);
        }
        DC_LEFT => {
            let mut sum = 0i32;
            for &l in left.iter().take(bh) {
                sum += l as i32;
            }
            let dc = ((sum + (bh as i32 >> 1)) / bh as i32) as u8;
            fill(dst, stride, bw, bh, dc);
        }
        DC_128 => fill(dst, stride, bw, bh, 128),
        V => {
            for r in 0..bh {
                for c in 0..bw {
                    dst[r * stride + c] = above.at(c) as u8;
                }
            }
        }
        H => {
            for r in 0..bh {
                let v = left[r];
                for c in 0..bw {
                    dst[r * stride + c] = v;
                }
            }
        }
        PAETH => {
            let tl = above.top_left();
            for r in 0..bh {
                for c in 0..bw {
                    dst[r * stride + c] = paeth_single(left[r] as i32, above.at(c), tl);
                }
            }
        }
        SMOOTH => {
            let below = left[bh - 1] as i32;
            let right = above.at(bw - 1);
            let sw_w = &SMOOTH_WEIGHTS[bw - 4..];
            let sw_h = &SMOOTH_WEIGHTS[bh - 4..];
            let log2 = 1 + SMOOTH_WEIGHT_LOG2_SCALE;
            let scale = 1i32 << SMOOTH_WEIGHT_LOG2_SCALE;
            for r in 0..bh {
                for c in 0..bw {
                    let wh = sw_h[r] as i32;
                    let ww = sw_w[c] as i32;
                    let p = wh * above.at(c)
                        + (scale - wh) * below
                        + ww * left[r] as i32
                        + (scale - ww) * right;
                    dst[r * stride + c] = divide_round(p, log2) as u8;
                }
            }
        }
        SMOOTH_V => {
            let below = left[bh - 1] as i32;
            let sw = &SMOOTH_WEIGHTS[bh - 4..];
            let log2 = SMOOTH_WEIGHT_LOG2_SCALE;
            let scale = 1i32 << SMOOTH_WEIGHT_LOG2_SCALE;
            for r in 0..bh {
                let w = sw[r] as i32;
                for c in 0..bw {
                    let p = w * above.at(c) + (scale - w) * below;
                    dst[r * stride + c] = divide_round(p, log2) as u8;
                }
            }
        }
        SMOOTH_H => {
            let right = above.at(bw - 1);
            let sw = &SMOOTH_WEIGHTS[bw - 4..];
            let log2 = SMOOTH_WEIGHT_LOG2_SCALE;
            let scale = 1i32 << SMOOTH_WEIGHT_LOG2_SCALE;
            for r in 0..bh {
                for c in 0..bw {
                    let w = sw[c] as i32;
                    let p = w * left[r] as i32 + (scale - w) * right;
                    dst[r * stride + c] = divide_round(p, log2) as u8;
                }
            }
        }
        _ => unreachable!(),
    }
}

#[inline]
fn fill(dst: &mut [u8], stride: usize, bw: usize, bh: usize, v: u8) {
    for r in 0..bh {
        for c in 0..bw {
            dst[r * stride + c] = v;
        }
    }
}

/// Highbd (`u16`) view over the `above` row exposing `above[-1]` at slot 0.
pub struct AboveRef16<'a>(pub &'a [u16]);
impl AboveRef16<'_> {
    #[inline]
    fn at(&self, i: usize) -> i32 {
        self.0[i + 1] as i32
    }
    #[inline]
    fn top_left(&self) -> i32 {
        self.0[0] as i32
    }
}

/// Highbd intra prediction (10/12-bit). Same math as [`predict`] on `u16`; only
/// `DC_128` depends on `bd` (`128 << (bd-8)`).
#[allow(clippy::too_many_arguments)]
pub fn predict_highbd(
    mode: usize, dst: &mut [u16], stride: usize, bw: usize, bh: usize,
    above: &AboveRef16, left: &[u16], bd: i32,
) {
    let fill16 = |dst: &mut [u16], v: u16| {
        for r in 0..bh {
            for c in 0..bw {
                dst[r * stride + c] = v;
            }
        }
    };
    match mode {
        DC => {
            let count = (bw + bh) as i32;
            let mut sum = 0i32;
            for i in 0..bw { sum += above.at(i); }
            for &l in left.iter().take(bh) { sum += l as i32; }
            fill16(dst, ((sum + (count >> 1)) / count) as u16);
        }
        DC_TOP => {
            let mut sum = 0i32;
            for i in 0..bw { sum += above.at(i); }
            fill16(dst, ((sum + (bw as i32 >> 1)) / bw as i32) as u16);
        }
        DC_LEFT => {
            let mut sum = 0i32;
            for &l in left.iter().take(bh) { sum += l as i32; }
            fill16(dst, ((sum + (bh as i32 >> 1)) / bh as i32) as u16);
        }
        DC_128 => fill16(dst, (128u32 << (bd - 8)) as u16),
        V => {
            for r in 0..bh { for c in 0..bw { dst[r * stride + c] = above.at(c) as u16; } }
        }
        H => {
            for r in 0..bh { let v = left[r]; for c in 0..bw { dst[r * stride + c] = v; } }
        }
        PAETH => {
            let tl = above.top_left();
            for r in 0..bh {
                for c in 0..bw {
                    dst[r * stride + c] = paeth_single_i32(left[r] as i32, above.at(c), tl) as u16;
                }
            }
        }
        SMOOTH => {
            let below = left[bh - 1] as i32;
            let right = above.at(bw - 1);
            let sw_w = &SMOOTH_WEIGHTS[bw - 4..];
            let sw_h = &SMOOTH_WEIGHTS[bh - 4..];
            let log2 = 1 + SMOOTH_WEIGHT_LOG2_SCALE;
            let scale = 1i32 << SMOOTH_WEIGHT_LOG2_SCALE;
            for r in 0..bh {
                for c in 0..bw {
                    let wh = sw_h[r] as i32;
                    let ww = sw_w[c] as i32;
                    let p = wh * above.at(c) + (scale - wh) * below + ww * left[r] as i32 + (scale - ww) * right;
                    dst[r * stride + c] = divide_round(p, log2) as u16;
                }
            }
        }
        SMOOTH_V => {
            let below = left[bh - 1] as i32;
            let sw = &SMOOTH_WEIGHTS[bh - 4..];
            let log2 = SMOOTH_WEIGHT_LOG2_SCALE;
            let scale = 1i32 << SMOOTH_WEIGHT_LOG2_SCALE;
            for r in 0..bh {
                let w = sw[r] as i32;
                for c in 0..bw {
                    dst[r * stride + c] = divide_round(w * above.at(c) + (scale - w) * below, log2) as u16;
                }
            }
        }
        SMOOTH_H => {
            let right = above.at(bw - 1);
            let sw = &SMOOTH_WEIGHTS[bw - 4..];
            let log2 = SMOOTH_WEIGHT_LOG2_SCALE;
            let scale = 1i32 << SMOOTH_WEIGHT_LOG2_SCALE;
            for r in 0..bh {
                for c in 0..bw {
                    let w = sw[c] as i32;
                    dst[r * stride + c] = divide_round(w * left[r] as i32 + (scale - w) * right, log2) as u16;
                }
            }
        }
        _ => unreachable!(),
    }
}

#[inline]
fn paeth_single_i32(left: i32, top: i32, top_left: i32) -> i32 {
    let base = top + left - top_left;
    let p_left = abs_diff(base, left);
    let p_top = abs_diff(base, top);
    let p_top_left = abs_diff(base, top_left);
    if p_left <= p_top && p_left <= p_top_left { left }
    else if p_top <= p_top_left { top }
    else { top_left }
}
