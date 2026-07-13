//! aom-convolve — bit-exact AV1 inter-prediction convolution (port of libaom
//! v3.14.1 `av1/common/convolve.c`), lowbd single-reference. Encoder critical
//! path (motion compensation). Starts with x/y separable EIGHTTAP_REGULAR.

const FILTER_BITS: i32 = 7;
const ROUND0_BITS: i32 = 3;

/// `av1_sub_pel_filters_8` (EIGHTTAP_REGULAR), 16 subpel positions × 8 taps,
/// from `av1/common/filter.h`.
#[rustfmt::skip]
pub static SUB_PEL_FILTERS_8: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],      [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],  [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0], [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],  [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0],  [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],  [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0], [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],  [0, 0, -2, 8, 126, -6, 2, 0],
];

#[inline]
fn rpo2(v: i32, n: i32) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

#[inline]
fn clip_pixel(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// `av1_convolve_x_sr_c` (EIGHTTAP_REGULAR). `src_off` is the interior origin;
/// `src` must have >=3 valid samples before and >=4 after in the x direction.
#[allow(clippy::too_many_arguments)]
pub fn convolve_x_sr(
    src: &[u8], src_off: usize, src_stride: usize, dst: &mut [u8], dst_stride: usize,
    w: usize, h: usize, subpel_x: usize,
) {
    let fo = 8 / 2 - 1; // 3
    let bits = FILTER_BITS - ROUND0_BITS;
    let filt = &SUB_PEL_FILTERS_8[subpel_x & 15];
    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize + (y * src_stride) as isize + x as isize - fo;
            let mut res = 0i32;
            for k in 0..8 {
                res += filt[k] as i32 * src[(base + k as isize) as usize] as i32;
            }
            res = rpo2(res, ROUND0_BITS);
            dst[y * dst_stride + x] = clip_pixel(rpo2(res, bits));
        }
    }
}

/// `av1_convolve_y_sr_c` (EIGHTTAP_REGULAR). `src` must have >=3 rows before and
/// >=4 after the interior origin in the y direction.
#[allow(clippy::too_many_arguments)]
pub fn convolve_y_sr(
    src: &[u8], src_off: usize, src_stride: usize, dst: &mut [u8], dst_stride: usize,
    w: usize, h: usize, subpel_y: usize,
) {
    let fo = 8 / 2 - 1; // 3
    let filt = &SUB_PEL_FILTERS_8[subpel_y & 15];
    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize + ((y as isize - fo) * src_stride as isize) + x as isize;
            let mut res = 0i32;
            for k in 0..8 {
                res += filt[k] as i32 * src[(base + (k as isize) * src_stride as isize) as usize] as i32;
            }
            dst[y * dst_stride + x] = clip_pixel(rpo2(res, FILTER_BITS));
        }
    }
}
