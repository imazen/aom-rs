//! aom-dist — bit-exact AV1 distortion metrics (port of libaom v3.14.1
//! `aom_dsp/sad.c`, `variance.c`). SAD, variance, and bilinear sub-pixel
//! variance — the workhorses of encoder motion search / RDO (speed-0 path).

pub mod avx2;
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
