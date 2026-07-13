//! aom-dist — bit-exact AV1 distortion metrics (port of libaom v3.14.1
//! `aom_dsp/sad.c`, `variance.c`). SAD, variance, and bilinear sub-pixel
//! variance — the workhorses of encoder motion search / RDO (speed-0 path).

#![forbid(unsafe_code)]

pub mod simd;
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
fn highbd_variance64(a: &[u16], a_stride: usize, b: &[u16], b_stride: usize, w: usize, h: usize) -> (u64, i64) {
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
    let var = sse.wrapping_sub(((sum as i64 * sum as i64) / (w * h) as i64) as u32);
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
