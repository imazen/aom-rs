//! aom-convolve — bit-exact AV1 inter-prediction convolution (port of libaom
//! v3.14.1 `av1/common/convolve.c`), lowbd single-reference. Encoder critical
//! path (motion compensation). Starts with x/y separable EIGHTTAP_REGULAR.


#![forbid(unsafe_code)]
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

/// `av1_sub_pel_filters_8smooth` (EIGHTTAP_SMOOTH).
#[rustfmt::skip]
pub static SUB_PEL_FILTERS_8SMOOTH: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],    [0, 2, 28, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],   [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],   [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],  [0, -2, 16, 54, 48, 12, 0, 0],
    [0, -2, 14, 52, 52, 14, -2, 0],[0, 0, 12, 48, 54, 16, -2, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],  [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],   [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],   [0, 0, 2, 34, 62, 28, 2, 0],
];

/// `av1_sub_pel_filters_8sharp` (EIGHTTAP_SHARP / MULTITAP_SHARP).
#[rustfmt::skip]
pub static SUB_PEL_FILTERS_8SHARP: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],          [-2, 2, -6, 126, 8, -2, 2, 0],
    [-2, 6, -12, 124, 16, -6, 4, -2],    [-2, 8, -18, 120, 26, -10, 6, -2],
    [-4, 10, -22, 116, 38, -14, 6, -2],  [-4, 10, -22, 108, 48, -18, 8, -2],
    [-4, 10, -24, 100, 60, -20, 8, -2],  [-4, 10, -24, 90, 70, -22, 10, -2],
    [-4, 12, -24, 80, 80, -24, 12, -4],  [-2, 10, -22, 70, 90, -24, 10, -4],
    [-2, 8, -20, 60, 100, -24, 10, -4],  [-2, 8, -18, 48, 108, -22, 10, -4],
    [-2, 6, -14, 38, 116, -22, 10, -4],  [-2, 6, -10, 26, 120, -18, 8, -2],
    [-2, 4, -6, 16, 124, -12, 6, -2],    [0, 2, -2, 8, 126, -6, 2, -2],
];

/// Select the subpel kernel row for filter `ftype` (0=regular,1=smooth,2=sharp).
#[inline]
fn kernel(ftype: usize, subpel: usize) -> &'static [i16; 8] {
    let table: &[[i16; 8]; 16] = match ftype {
        0 => &SUB_PEL_FILTERS_8,
        1 => &SUB_PEL_FILTERS_8SMOOTH,
        2 => &SUB_PEL_FILTERS_8SHARP,
        _ => panic!("bad filter type"),
    };
    &table[subpel & 15]
}

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
    w: usize, h: usize, subpel_x: usize, ftype: usize,
) {
    let fo = 8 / 2 - 1; // 3
    let bits = FILTER_BITS - ROUND0_BITS;
    let filt = kernel(ftype, subpel_x);
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

/// `av1_convolve_2d_sr_c` (EIGHTTAP_REGULAR, lowbd, SR: round_0=3, round_1=11,
/// bits=0). `src` needs a border of >=3 (top/left) and >=4 (bottom/right).
#[allow(clippy::too_many_arguments)]
pub fn convolve_2d_sr(
    src: &[u8], src_off: usize, src_stride: usize, dst: &mut [u8], dst_stride: usize,
    w: usize, h: usize, subpel_x: usize, subpel_y: usize, ftype: usize,
) {
    const BD: i32 = 8;
    const ROUND_1: i32 = 2 * FILTER_BITS - ROUND0_BITS; // 11
    let taps = 8usize;
    let fo = taps / 2 - 1; // 3
    let im_h = h + taps - 1;
    let im_stride = w;
    let xf = kernel(ftype, subpel_x);
    let yf = kernel(ftype, subpel_y);

    // Horizontal pass into int16 intermediate.
    let mut im = vec![0i16; im_h * im_stride];
    let src_horiz = src_off as isize - fo as isize * src_stride as isize;
    for y in 0..im_h {
        for x in 0..w {
            let base = src_horiz + (y * src_stride) as isize + x as isize - fo as isize;
            let mut sum = 1i32 << (BD + FILTER_BITS - 1);
            for k in 0..taps {
                sum += xf[k] as i32 * src[(base + k as isize) as usize] as i32;
            }
            im[y * im_stride + x] = rpo2(sum, ROUND0_BITS) as i16;
        }
    }

    // Vertical pass.
    let offset_bits = BD + 2 * FILTER_BITS - ROUND0_BITS; // 19
    let round_offset = (1i32 << (offset_bits - ROUND_1)) + (1i32 << (offset_bits - ROUND_1 - 1));
    let bits = FILTER_BITS * 2 - ROUND0_BITS - ROUND_1; // 0
    for y in 0..h {
        for x in 0..w {
            let mut sum = 1i32 << offset_bits;
            for k in 0..taps {
                // src_vert[(y - fo + k)] with src_vert = im + fo rows -> im[(y+k)]
                sum += yf[k] as i32 * im[(y + k) * im_stride + x] as i32;
            }
            let res = (rpo2(sum, ROUND_1) - round_offset) as i16;
            dst[y * dst_stride + x] = clip_pixel(rpo2(res as i32, bits));
        }
    }
}

/// `av1_convolve_y_sr_c` (EIGHTTAP_REGULAR). `src` must have >=3 rows before and
/// >=4 after the interior origin in the y direction.
#[allow(clippy::too_many_arguments)]
pub fn convolve_y_sr(
    src: &[u8], src_off: usize, src_stride: usize, dst: &mut [u8], dst_stride: usize,
    w: usize, h: usize, subpel_y: usize, ftype: usize,
) {
    let fo = 8 / 2 - 1; // 3
    let filt = kernel(ftype, subpel_y);
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
