//! aom-intra — bit-exact AV1 intra predictors (port of libaom v3.14.1
//! `aom_dsp/intrapred.c`). Non-directional lowbd family: DC / DC_top / DC_left
//! / DC_128 / V / H / Paeth / Smooth / Smooth_V / Smooth_H, generic over block
//! size. `above` must have `above[-1]` (top-left) valid (index via `AboveRef`).
//!
//! Validated byte-for-byte against C for every (mode × block size).


#![forbid(unsafe_code)]
pub mod dir;
pub mod edge;
mod weights;
use archmage::autoversion;
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

/// Full transform dims per `TX_SIZE` (`tx_size_wide` / `tx_size_high`).
const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

/// Assemble the non-directional intra reference edges — the constant fills,
/// contiguous copy, and edge replication of libaom's
/// `highbd_build_non_directional_intra_predictors` (reconintra.c). `#[autoversion]`
/// compiles one `#[target_feature]`-gated variant per SIMD tier (AVX-512 / AVX2 /
/// NEON / WASM / scalar) plus a runtime dispatcher, so the `base±1` fills and the
/// contiguous above copy lower to vector splats / stores; byte-identical to the
/// scalar path. The strided left-column gather stays scalar (per-arch gather buys
/// little at these edge lengths).
///
/// `recon[ref_off]` is the block's top-left pixel; the above row is
/// `recon[ref_off - ref_stride ..]`, the left column
/// `recon[ref_off - 1 + i*ref_stride]`. `above_row` / `left_col` are the `[-1..]`
/// windows: index 0 is the top-left corner, index `1+i` the i-th edge sample
/// (len `1 + txwpx` / `1 + txhpx`). `av1_mode` is the AV1 `PREDICTION_MODE`
/// (DC=0, SMOOTH=9, SMOOTH_V=10, SMOOTH_H=11, PAETH=12). The neighbour
/// availability counts `n_top_px ≤ txwpx` / `n_left_px ≤ txhpx` are the caller's
/// job (the decode driver's availability logic). All five non-directional modes
/// need above and left; only PAETH also reads the corner.
/// Geometry + neighbour availability for [`assemble_nd_edges`], bundled to keep
/// the vectorized assembly within a sane argument count.
struct NdEdge {
    ref_off: usize,
    ref_stride: usize,
    av1_mode: usize,
    txwpx: usize,
    txhpx: usize,
    n_top_px: usize,
    n_left_px: usize,
    base: i32,
}

#[autoversion]
fn assemble_nd_edges(recon: &[u16], g: &NdEdge, above_row: &mut [u16], left_col: &mut [u16]) {
    let NdEdge { ref_off, ref_stride, av1_mode, txwpx, txhpx, n_top_px, n_left_px, base } = *g;

    // Left column: default base+1, then the real samples with the last one
    // replicated, or the above-corner fallback when only the top is available.
    let lo = (base + 1) as u16;
    for e in left_col[..1 + txhpx].iter_mut() {
        *e = lo;
    }
    if n_left_px > 0 {
        let loff = ref_off - 1;
        for i in 0..n_left_px {
            left_col[1 + i] = recon[loff + i * ref_stride]; // strided gather (scalar)
        }
        let last = left_col[n_left_px]; // == C left_col[n_left_px - 1]
        for e in left_col[1 + n_left_px..1 + txhpx].iter_mut() {
            *e = last;
        }
    } else if n_top_px > 0 {
        let a0 = recon[ref_off - ref_stride];
        for e in left_col[1..1 + txhpx].iter_mut() {
            *e = a0;
        }
    }

    // Above row: default base-1, then the real samples with the last replicated,
    // or the left-corner fallback when only the left is available.
    let ao = (base - 1) as u16;
    for e in above_row[..1 + txwpx].iter_mut() {
        *e = ao;
    }
    if n_top_px > 0 {
        let aoff = ref_off - ref_stride;
        above_row[1..1 + n_top_px].copy_from_slice(&recon[aoff..aoff + n_top_px]);
        let last = above_row[n_top_px];
        for e in above_row[1 + n_top_px..1 + txwpx].iter_mut() {
            *e = last;
        }
    } else if n_left_px > 0 {
        let l0 = recon[ref_off - 1];
        for e in above_row[1..1 + txwpx].iter_mut() {
            *e = l0;
        }
    }

    // Top-left corner (only PAETH reads it).
    if av1_mode == 12 {
        let corner = if n_top_px > 0 && n_left_px > 0 {
            recon[ref_off - ref_stride - 1]
        } else if n_top_px > 0 {
            recon[ref_off - ref_stride]
        } else if n_left_px > 0 {
            recon[ref_off - 1]
        } else {
            base as u16
        };
        above_row[0] = corner;
        left_col[0] = corner;
    }
}

/// Build the intra prediction for a non-directional mode (DC / SMOOTH / SMOOTH_V
/// / SMOOTH_H / PAETH) into `dst` — the highbd path of libaom's
/// `av1_predict_intra_block` non-directional branch
/// (`highbd_build_non_directional_intra_predictors`, reconintra.c). Assembles the
/// reference edges from the reconstructed neighbours (via the archmage-vectorized
/// [`assemble_nd_edges`]) then runs the predictor.
///
/// `recon[ref_off]` is the block top-left in the reconstruction plane (row stride
/// `ref_stride`); `dst` is the output block (row stride `dst_stride`). `av1_mode`
/// is the AV1 `PREDICTION_MODE`. `n_top_px`/`n_left_px` are the available
/// neighbour counts (`≤ txwpx`/`txhpx`), computed by the decode driver.
#[allow(clippy::too_many_arguments)]
pub fn build_non_directional_intra_high(
    recon: &[u16],
    ref_off: usize,
    ref_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    av1_mode: usize,
    tx_size: usize,
    n_top_px: usize,
    n_left_px: usize,
    bd: i32,
) {
    let txwpx = TX_W[tx_size];
    let txhpx = TX_H[tx_size];
    let base = 128i32 << (bd - 8);

    // [-1..] reference windows: index 0 is the top-left corner.
    let mut above_buf = [0u16; 1 + 64];
    let mut left_buf = [0u16; 1 + 64];
    let g = NdEdge { ref_off, ref_stride, av1_mode, txwpx, txhpx, n_top_px, n_left_px, base };
    assemble_nd_edges(recon, &g, &mut above_buf[..1 + txwpx], &mut left_buf[..1 + txhpx]);

    // Map AV1 mode → predictor index; DC picks the availability variant.
    let pmode = match av1_mode {
        0 => match (n_left_px > 0, n_top_px > 0) {
            (true, true) => DC,
            (false, true) => DC_TOP,
            (true, false) => DC_LEFT,
            (false, false) => DC_128,
        },
        9 => SMOOTH,
        10 => SMOOTH_V,
        11 => SMOOTH_H,
        12 => PAETH,
        _ => unreachable!("build_non_directional_intra_high: non-directional modes only"),
    };

    let above = AboveRef16(&above_buf[..1 + txwpx]);
    predict_highbd(pmode, dst, dst_stride, txwpx, txhpx, &above, &left_buf[1..1 + txhpx], bd);
}
