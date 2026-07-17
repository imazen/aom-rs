//! aom-dist — bit-exact AV1 distortion metrics (port of libaom v3.14.1
//! `aom_dsp/sad.c`, `variance.c`). SAD, variance, and bilinear sub-pixel
//! variance — the workhorses of encoder motion search / RDO (speed-0 path).

#![forbid(unsafe_code)]

pub mod simd;
mod simd_variance;
pub mod hadamard;

const FILTER_BITS: i32 = 7;

/// `bilinear_filters_2t` from `aom_dsp/aom_filter.h` (8 subpel positions).
#[rustfmt::skip]
pub static BILINEAR_FILTERS_2T: [[u8; 2]; 8] = [
    [128, 0], [112, 16], [96, 32], [80, 48],
    [64, 64], [48, 80], [32, 96], [16, 112],
];

/// `aom_sad<W>x<H>_c`: sum of absolute differences.
pub fn sad(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut s: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            s += (a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32).unsigned_abs();
        }
    }
    s
}

/// `aom_sse_c`: sum of squared errors over a generic w×h region (RD distortion).
pub fn sse(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> i64 {
    let mut sse: i64 = 0;
    for y in 0..h {
        for x in 0..w {
            let diff = (a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32).abs();
            sse += (diff * diff) as i64;
        }
    }
    sse
}

/// `aom_highbd_sse_c`: SSE over 16-bit samples.
pub fn highbd_sse(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> i64 {
    let mut sse: i64 = 0;
    for y in 0..h {
        for x in 0..w {
            let diff = (a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32).abs();
            sse += (diff * diff) as i64;
        }
    }
    sse
}

/// `aom_sad<W>x<H>_avg_c`: SAD of `src` against the rounded average of `ref` and
/// a contiguous `second_pred` (compound-prediction motion search). Matches
/// `aom_comp_avg_pred` (comp = ROUND_POWER_OF_TWO(ref+second_pred, 1)) followed
/// by `sad`.
pub fn sad_avg(
    src: &[u8], src_stride: usize, ref_: &[u8], ref_stride: usize, second_pred: &[u8],
    w: usize, h: usize,
) -> u32 {
    let mut s: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let comp = (ref_[y * ref_stride + x] as u32 + second_pred[y * w + x] as u32 + 1) >> 1;
            s += (src[y * src_stride + x] as i32 - comp as i32).unsigned_abs();
        }
    }
    s
}

/// `aom_obmc_sad<W>x<H>_c`: overlapped block motion-comp SAD. `wsrc` and `mask`
/// are contiguous i32 buffers (stride = width); `sad += (|wsrc - pre*mask| + 2048)
/// >> 12` (ROUND_POWER_OF_TWO by 12).
pub fn obmc_sad(pre: &[u8], pre_stride: usize, wsrc: &[i32], mask: &[i32], w: usize, h: usize) -> u32 {
    let mut sad: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let d = (wsrc[y * w + x] - pre[y * pre_stride + x] as i32 * mask[y * w + x]).unsigned_abs();
            sad += (d + 2048) >> 12;
        }
    }
    sad
}

/// `aom_masked_sad<W>x<H>_c`: SAD of `src` against an A64-mask blend of `ref`
/// and a contiguous `second_pred` (wedge / difference-weighted compound RD).
/// `AOM_BLEND_A64(m,a,b) = (m*a + (64-m)*b + 32) >> 6`. When `invert_mask`, the
/// roles of `ref` (strided) and `second_pred` (stride = width) as blend operands
/// a/b are swapped.
#[allow(clippy::too_many_arguments)]
pub fn masked_sad(
    src: &[u8], src_stride: usize, ref_: &[u8], ref_stride: usize, second_pred: &[u8],
    msk: &[u8], msk_stride: usize, invert_mask: bool, w: usize, h: usize,
) -> u32 {
    let mut sad: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let rp = ref_[y * ref_stride + x] as i32;
            let sp = second_pred[y * w + x] as i32;
            let (a, b) = if invert_mask { (sp, rp) } else { (rp, sp) };
            let m = msk[y * msk_stride + x] as i32;
            let pred = (m * a + (64 - m) * b + 32) >> 6;
            sad += (pred - src[y * src_stride + x] as i32).unsigned_abs();
        }
    }
    sad
}

/// `aom_highbd_obmc_sad<W>x<H>_c`: highbd OBMC SAD (samples 16-bit; wsrc/mask i32).
pub fn highbd_obmc_sad(pre: &[u16], pre_stride: usize, wsrc: &[i32], mask: &[i32], w: usize, h: usize) -> u32 {
    let mut sad: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let d = (wsrc[y * w + x] - pre[y * pre_stride + x] as i32 * mask[y * w + x]).unsigned_abs();
            sad += (d + 2048) >> 12;
        }
    }
    sad
}

/// `aom_highbd_masked_sad<W>x<H>_c`: highbd wedge / diff-weighted compound SAD.
/// Mask stays 8-bit (0..=64); samples are 16-bit.
#[allow(clippy::too_many_arguments)]
pub fn highbd_masked_sad(
    src: &[u16], src_stride: usize, ref_: &[u16], ref_stride: usize, second_pred: &[u16],
    msk: &[u8], msk_stride: usize, invert_mask: bool, w: usize, h: usize,
) -> u32 {
    let mut sad: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let rp = ref_[y * ref_stride + x] as i32;
            let sp = second_pred[y * w + x] as i32;
            let (a, b) = if invert_mask { (sp, rp) } else { (rp, sp) };
            let m = msk[y * msk_stride + x] as i32;
            let pred = (m * a + (64 - m) * b + 32) >> 6;
            sad += (pred - src[y * src_stride + x] as i32).unsigned_abs();
        }
    }
    sad
}

/// `aom_highbd_sad<W>x<H>_c`: SAD over 16-bit (10/12-bit) samples.
pub fn highbd_sad(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut s: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            s += (a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32).unsigned_abs();
        }
    }
    s
}

/// `aom_highbd_sad<W>x<H>_avg_c`: highbd compound-prediction SAD.
pub fn highbd_sad_avg(
    src: &[u16], src_stride: usize, ref_: &[u16], ref_stride: usize, second_pred: &[u16],
    w: usize, h: usize,
) -> u32 {
    let mut s: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let comp = (ref_[y * ref_stride + x] as u32 + second_pred[y * w + x] as u32 + 1) >> 1;
            s += (src[y * src_stride + x] as i32 - comp as i32).unsigned_abs();
        }
    }
    s
}

/// libaom `highbd_variance64`: returns (sse: u64, sum: i64). Note `tsse`
/// accumulates the 32-bit-truncated square (`(uint32_t)(diff*diff)`) into u64.
///
/// SIMD-dispatched (Gate 3): widths >= 8 take the magetypes kernel in
/// `simd_variance.rs` (bit-identical on the pixel domain — see its module
/// docs); width 4 and the `AOM_FORCE_SCALAR` pin run the scalar twin.
fn highbd_variance64(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> (u64, i64) {
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    if w < 8 {
        return highbd_variance64_scalar(a, a_stride, b, b_stride, w, h);
    }
    archmage::incant!(
        simd_variance::highbd_variance64_impl(a, a_stride, b, b_stride, w, h),
        [v3, neon, wasm128, scalar]
    )
}

/// The scalar transcription (the reference twin — never SIMD-routed). Public
/// for the SIMD differential (`hbd_variance_simd_diff.rs`).
pub fn highbd_variance64_scalar(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> (u64, i64) {
    let mut tsum: i64 = 0;
    let mut tsse: u64 = 0;
    for y in 0..h {
        let mut lsum: i32 = 0;
        for x in 0..w {
            let diff = a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32;
            lsum += diff;
            tsse += ((diff * diff) as u32) as u64;
        }
        tsum += lsum as i64;
    }
    (tsse, tsum)
}

/// `aom_highbd_<bd>_variance<W>x<H>_c`: returns (variance, sse). `bd` ∈ {8,10,12}.
pub fn highbd_variance(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize, bd: u8) -> (u32, u32) {
    let (sse_long, sum_long) = highbd_variance64(a, a_stride, b, b_stride, w, h);
    // bd-dependent normalisation (ROUND_POWER_OF_TWO), matching libaom
    // highbd_8/10/12_variance.
    let (sse, sum): (u32, i32) = match bd {
        8 => (sse_long as u32, sum_long as i32),
        10 => (
            ((sse_long + (1 << 3)) >> 4) as u32,
            ((sum_long + (1 << 1)) >> 2) as i32,
        ),
        _ => (
            ((sse_long + (1 << 7)) >> 8) as u32,
            ((sum_long + (1 << 3)) >> 4) as i32,
        ),
    };
    // variance.c HIGHBD_VAR: the 8-bit variant computes
    // `*sse - (uint32_t)(((int64_t)sum * sum) / (W * H))` (WRAPS when the
    // rounded terms drive it negative), while the 10/12-bit variants compute
    // an i64 var and CLAMP `(var >= 0) ? var : 0` — the bd normalisation can
    // round sse below sum^2/n for near-flat differences.
    let var = if bd == 8 {
        sse.wrapping_sub(((i64::from(sum) * i64::from(sum)) / (w * h) as i64) as u32)
    } else {
        let v = i64::from(sse) - (i64::from(sum) * i64::from(sum)) / (w * h) as i64;
        if v >= 0 { v as u32 } else { 0 }
    };
    (var, sse)
}

/// `aom_highbd_<bd>_sub_pixel_variance<W>x<H>_c`: highbd bilinear (2-tap)
/// interpolate `a` at (xoffset, yoffset) into a 16-bit intermediate, then highbd
/// variance against `b`. Returns (variance, sse).
#[allow(clippy::too_many_arguments)]
pub fn highbd_sub_pixel_variance(
    a: &[u16], a_stride: usize, xoffset: usize, yoffset: usize,
    b: &[u16], b_stride: usize, w: usize, h: usize, bd: u8,
) -> (u32, u32) {
    // First pass (horizontal), pixel_step = 1, u16 -> u16.
    let fx = BILINEAR_FILTERS_2T[xoffset];
    let mut fdata3 = vec![0u16; (h + 1) * w];
    for i in 0..(h + 1) {
        for j in 0..w {
            let a0 = a[i * a_stride + j] as i32;
            let a1 = a[i * a_stride + j + 1] as i32;
            fdata3[i * w + j] = rpo2(a0 * fx[0] as i32 + a1 * fx[1] as i32, FILTER_BITS);
        }
    }
    // Second pass (vertical), pixel_step = w, u16 -> u16 (highbd keeps 16-bit).
    let fy = BILINEAR_FILTERS_2T[yoffset];
    let mut temp2 = vec![0u16; h * w];
    for i in 0..h {
        for j in 0..w {
            let v0 = fdata3[i * w + j] as i32;
            let v1 = fdata3[(i + 1) * w + j] as i32;
            temp2[i * w + j] = rpo2(v0 * fy[0] as i32 + v1 * fy[1] as i32, FILTER_BITS);
        }
    }
    highbd_variance(&temp2, w, b, b_stride, w, h, bd)
}

/// libaom `variance()`: returns (sse, sum).
fn variance_raw(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> (u32, i32) {
    let mut tsum: i32 = 0;
    let mut tsse: u32 = 0;
    for y in 0..h {
        for x in 0..w {
            let diff = a[y * a_stride + x] as i32 - b[y * b_stride + x] as i32;
            tsum += diff;
            tsse = tsse.wrapping_add((diff * diff) as u32);
        }
    }
    (tsse, tsum)
}

/// `aom_variance<W>x<H>_c`: returns (variance, sse).
pub fn variance(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> (u32, u32) {
    let (sse, sum) = variance_raw(a, a_stride, b, b_stride, w, h);
    let var = sse.wrapping_sub(((sum as i64 * sum as i64) / (w * h) as i64) as u32);
    (var, sse)
}

#[inline]
fn rpo2(v: i32, n: i32) -> u16 {
    ((v + ((1 << n) >> 1)) >> n) as u16
}

/// `aom_sub_pixel_variance<W>x<H>_c`: bilinear interpolate `a` at (xoff,yoff)
/// then variance against `b`. Returns (variance, sse).
#[allow(clippy::too_many_arguments)]
pub fn sub_pixel_variance(
    a: &[u8], a_stride: usize, xoffset: usize, yoffset: usize,
    b: &[u8], b_stride: usize, w: usize, h: usize,
) -> (u32, u32) {
    // First pass (horizontal): output (h+1) x w into u16 fdata3.
    let fx = BILINEAR_FILTERS_2T[xoffset];
    let mut fdata3 = vec![0u16; (h + 1) * w];
    for i in 0..(h + 1) {
        for j in 0..w {
            let a0 = a[i * a_stride + j] as i32;
            let a1 = a[i * a_stride + j + 1] as i32; // pixel_step = 1
            fdata3[i * w + j] = rpo2(a0 * fx[0] as i32 + a1 * fx[1] as i32, FILTER_BITS);
        }
    }
    // Second pass (vertical): output h x w into u8 temp2. pixel_step = w.
    let fy = BILINEAR_FILTERS_2T[yoffset];
    let mut temp2 = vec![0u8; h * w];
    for i in 0..h {
        for j in 0..w {
            let v0 = fdata3[i * w + j] as i32;
            let v1 = fdata3[(i + 1) * w + j] as i32; // pixel_step = w
            temp2[i * w + j] = rpo2(v0 * fy[0] as i32 + v1 * fy[1] as i32, FILTER_BITS) as u8;
        }
    }
    variance(&temp2, w, b, b_stride, w, h)
}

/// `av1_block_error_c` (`av1/encoder/rdopt.c`): transform-domain distortion.
/// Returns `(error, ssz)` where `error = sum((coeff-dqcoeff)^2)` and
/// `ssz = sum(coeff^2)`. Lowbd (8-bit): the per-element products are 32-bit
/// (matching C's `int` arithmetic — wraps like C on overflow) and accumulate
/// into 64-bit. Used by the encoder's RD search.
pub fn block_error(coeff: &[i32], dqcoeff: &[i32]) -> (i64, i64) {
    let n = coeff.len();
    let mut error = 0i64;
    let mut sqcoeff = 0i64;
    for i in 0..n {
        let diff = coeff[i].wrapping_sub(dqcoeff[i]);
        error += diff.wrapping_mul(diff) as i64;
        sqcoeff += coeff[i].wrapping_mul(coeff[i]) as i64;
    }
    (error, sqcoeff)
}

/// `av1_highbd_block_error_c` (`av1/encoder/rdopt.c`): highbd transform-domain
/// distortion. Like [`block_error`] but the products are 64-bit (no wrap) and
/// both sums are rounded-shifted by `2*(bd-8)`. Returns `(error, ssz)`.
pub fn highbd_block_error(coeff: &[i32], dqcoeff: &[i32], bd: u8) -> (i64, i64) {
    let n = coeff.len();
    let mut error = 0i64;
    let mut sqcoeff = 0i64;
    for i in 0..n {
        let diff = coeff[i] as i64 - dqcoeff[i] as i64;
        error += diff * diff;
        sqcoeff += coeff[i] as i64 * coeff[i] as i64;
    }
    let shift = 2 * (bd as i32 - 8);
    let rounding = (1i64 << shift) >> 1;
    error = (error + rounding) >> shift;
    sqcoeff = (sqcoeff + rounding) >> shift;
    (error, sqcoeff)
}

/// `aom_subtract_block_c` (`aom_dsp/subtract.c`): the residual generator —
/// `diff[r][c] = src[r][c] - pred[r][c]`, row by row. Natively strided (the
/// `diff`/`src`/`pred` row strides are independent). This is the input the
/// encoder feeds to the forward transform.
#[allow(clippy::too_many_arguments)]
pub fn subtract_block(
    rows: usize, cols: usize, diff: &mut [i16], diff_stride: usize,
    src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize,
) {
    for r in 0..rows {
        let (d, s, p) = (r * diff_stride, r * src_stride, r * pred_stride);
        for c in 0..cols {
            diff[d + c] = src[s + c] as i16 - pred[p + c] as i16;
        }
    }
}

/// `aom_highbd_subtract_block_c`: highbd (10/12-bit) residual generator. Same as
/// [`subtract_block`] but 16-bit `src`/`pred`; the difference is truncated to
/// `i16` exactly as the C stores `int` into `int16_t`.
#[allow(clippy::too_many_arguments)]
pub fn highbd_subtract_block(
    rows: usize, cols: usize, diff: &mut [i16], diff_stride: usize,
    src: &[u16], src_stride: usize, pred: &[u16], pred_stride: usize,
) {
    for r in 0..rows {
        let (d, s, p) = (r * diff_stride, r * src_stride, r * pred_stride);
        for c in 0..cols {
            diff[d + c] = (src[s + c] as i32 - pred[p + c] as i32) as i16;
        }
    }
}

/// `av1_block_error_qm` (`av1/encoder/tx_search.c`): the quant-matrix-weighted
/// transform-domain distortion used by the QM RD path. Per coefficient the diff
/// and coeff are scaled by `qmatrix[scan[i]]`, squared, and rounded `>> 2*AOM_QM_BITS`
/// (=10); both sums then get the `>> 2*(bd-8)` bit-depth normalization. Returns
/// `(error, ssz)`. With a flat matrix (all weights `1<<AOM_QM_BITS` = 32) this
/// reduces to [`highbd_block_error`].
pub fn block_error_qm(coeff: &[i32], dqcoeff: &[i32], qmatrix: &[u8], scan: &[i16], bd: u8) -> (i64, i64) {
    let shift = 2 * (bd as i32 - 8);
    let rounding = (1i64 << shift) >> 1;
    let mut error = 0i64;
    let mut sqcoeff = 0i64;
    for i in 0..coeff.len() {
        let weight = qmatrix[scan[i] as usize] as i64;
        let dd = (coeff[i] as i64 - dqcoeff[i] as i64) * weight;
        let cc = coeff[i] as i64 * weight;
        // 2*AOM_QM_BITS = 10; rounding half = 1 << 9 = 512.
        error += (dd * dd + (1 << 9)) >> 10;
        sqcoeff += (cc * cc + (1 << 9)) >> 10;
    }
    error = (error + rounding) >> shift;
    sqcoeff = (sqcoeff + rounding) >> shift;
    (error, sqcoeff)
}

/// `aom_sum_squares_i16_c` (`aom_dsp/sum_squares.c`): sum of squared i16 values
/// (residual energy). The per-element `v*v` is 32-bit (matching C's `int`) and
/// accumulates into u64.
pub fn sum_squares_i16(src: &[i16]) -> u64 {
    let mut ss = 0u64;
    for &v in src {
        let v = v as i32;
        ss += (v * v) as u64;
    }
    ss
}

/// `aom_sum_squares_2d_i16_c`: the 2-D strided residual energy over a
/// `width x height` block with row stride `src_stride`.
pub fn sum_squares_2d_i16(src: &[i16], src_stride: usize, width: usize, height: usize) -> u64 {
    let mut ss = 0u64;
    for r in 0..height {
        let base = r * src_stride;
        for c in 0..width {
            let v = src[base + c] as i32;
            ss += (v * v) as u64;
        }
    }
    ss
}

/// `aom_vector_var_c` (`aom_dsp/avg.c`): variance of `ref - src` over a
/// `4<<bwl`-wide vector, used by the motion/RD search. `mean_abs^2` is computed
/// in **unsigned** 32-bit (it can reach ~2^32 for bwl=5), and the final subtract
/// is unsigned then reinterpreted — replicated exactly for bit-identity.
pub fn vector_var(reff: &[i16], src: &[i16], bwl: i32) -> i32 {
    let width = 4usize << bwl;
    let mut sse: i32 = 0;
    let mut mean: i32 = 0;
    for i in 0..width {
        let diff = reff[i] as i32 - src[i] as i32;
        mean += diff;
        sse += diff * diff;
    }
    let mean_abs = mean.unsigned_abs();
    (sse as u32).wrapping_sub(mean_abs.wrapping_mul(mean_abs) >> (bwl as u32 + 2)) as i32
}
