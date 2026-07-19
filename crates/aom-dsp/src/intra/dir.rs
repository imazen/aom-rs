//! Directional intra predictors z1/z2/z3, bit-exact port of libaom v3.14.1
//! `av1/common/reconintra.c` (`av1_dr_prediction_z{1,2,3}_c`). Plus the
//! `dr_intra_derivative` angle table and `av1_get_dx/dy`.
//!
//! z2 reads `above`/`left` at negative offsets, so callers must provide edges
//! via [`EdgeRef`] with front padding.

/// `dr_intra_derivative[90]` from `reconintra.h`.
#[rustfmt::skip]
pub static DR_INTRA_DERIVATIVE: [i16; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0,
    178, 0, 0, 151, 0, 0, 132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80,
    0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0, 45, 0, 0, 0, 40, 0, 0, 35,
    0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0, 0,
    7, 0, 0, 3, 0, 0,
];

pub fn get_dx(angle: i32) -> i32 {
    if angle > 0 && angle < 90 {
        DR_INTRA_DERIVATIVE[angle as usize] as i32
    } else if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32
    } else {
        1
    }
}

pub fn get_dy(angle: i32) -> i32 {
    if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32
    } else if angle > 180 && angle < 270 {
        DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32
    } else {
        1
    }
}

/// An edge (above/left) with `pad` valid samples before index 0.
pub struct EdgeRef<'a> {
    data: &'a [u8],
    pad: usize,
}
impl<'a> EdgeRef<'a> {
    pub fn new(data: &'a [u8], pad: usize) -> Self {
        EdgeRef { data, pad }
    }
    #[inline]
    fn at(&self, i: i32) -> i32 {
        self.data[(self.pad as i32 + i) as usize] as i32
    }
}

#[inline]
fn rpo2_5(v: i32) -> u8 {
    ((v + 16) >> 5) as u8
}

#[inline]
fn rpo2_5_16(v: i32) -> u16 {
    ((v + 16) >> 5) as u16
}

/// An edge (above/left) of 10/12-bit samples with `pad` valid samples before
/// index 0 (the highbd analogue of [`EdgeRef`]).
pub struct EdgeRef16<'a> {
    data: &'a [u16],
    pad: usize,
}
impl<'a> EdgeRef16<'a> {
    pub fn new(data: &'a [u16], pad: usize) -> Self {
        EdgeRef16 { data, pad }
    }
    #[inline]
    fn at(&self, i: i32) -> i32 {
        self.data[(self.pad as i32 + i) as usize] as i32
    }
}

/// `av1_dr_prediction_z1_c` (dy == 1, dx > 0).
pub fn z1(dst: &mut [u8], stride: usize, bw: usize, bh: usize, above: &EdgeRef, up: i32, dx: i32) {
    let max_base_x = (((bw + bh) as i32) - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut x = dx;
    for r in 0..bh {
        let base = x >> frac_bits;
        let shift = ((x << up) & 0x3F) >> 1;
        if base >= max_base_x {
            let fillv = above.at(max_base_x) as u8;
            for rr in r..bh {
                for c in 0..bw {
                    dst[rr * stride + c] = fillv;
                }
            }
            return;
        }
        let mut base = base;
        for c in 0..bw {
            dst[r * stride + c] = if base < max_base_x {
                rpo2_5(above.at(base) * (32 - shift) + above.at(base + 1) * shift)
            } else {
                above.at(max_base_x) as u8
            };
            base += base_inc;
        }
        x += dx;
    }
}

/// `av1_dr_prediction_z2_c` (dx > 0, dy > 0).
#[allow(clippy::too_many_arguments)]
pub fn z2(
    dst: &mut [u8],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef,
    left: &EdgeRef,
    up_above: i32,
    up_left: i32,
    dx: i32,
    dy: i32,
) {
    let min_base_x = -(1 << up_above);
    let frac_bits_x = 6 - up_above;
    let frac_bits_y = 6 - up_left;
    for r in 0..bh {
        for c in 0..bw {
            let y = (r + 1) as i32;
            let x = ((c as i32) << 6) - y * dx;
            let base_x = x >> frac_bits_x;
            let val = if base_x >= min_base_x {
                let shift = ((x * (1 << up_above)) & 0x3F) >> 1;
                rpo2_5(above.at(base_x) * (32 - shift) + above.at(base_x + 1) * shift)
            } else {
                let x2 = (c + 1) as i32;
                let y2 = ((r as i32) << 6) - x2 * dy;
                let base_y = y2 >> frac_bits_y;
                let shift = ((y2 * (1 << up_left)) & 0x3F) >> 1;
                rpo2_5(left.at(base_y) * (32 - shift) + left.at(base_y + 1) * shift)
            };
            dst[r * stride + c] = val;
        }
    }
}

/// `av1_dr_prediction_z3_c` (dx == 1, dy > 0).
pub fn z3(dst: &mut [u8], stride: usize, bw: usize, bh: usize, left: &EdgeRef, up: i32, dy: i32) {
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut y = dy;
    for c in 0..bw {
        let mut base = y >> frac_bits;
        let shift = ((y << up) & 0x3F) >> 1;
        for r in 0..bh {
            if base < max_base_y {
                dst[r * stride + c] =
                    rpo2_5(left.at(base) * (32 - shift) + left.at(base + 1) * shift);
                base += base_inc;
            } else {
                let fillv = left.at(max_base_y) as u8;
                for rr in r..bh {
                    dst[rr * stride + c] = fillv;
                }
                break;
            }
        }
        y += dy;
    }
}

/// `av1_highbd_dr_prediction_z1_c` (dy == 1, dx > 0). Highbd analogue of [`z1`];
/// `bd` is unused (the two-tap interpolation of in-range samples stays in range).
pub fn z1_high(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef16,
    up: i32,
    dx: i32,
) {
    let max_base_x = (((bw + bh) as i32) - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut x = dx;
    for r in 0..bh {
        let base = x >> frac_bits;
        let shift = ((x << up) & 0x3F) >> 1;
        if base >= max_base_x {
            let fillv = above.at(max_base_x) as u16;
            for rr in r..bh {
                for c in 0..bw {
                    dst[rr * stride + c] = fillv;
                }
            }
            return;
        }
        let mut base = base;
        for c in 0..bw {
            dst[r * stride + c] = if base < max_base_x {
                rpo2_5_16(above.at(base) * (32 - shift) + above.at(base + 1) * shift)
            } else {
                above.at(max_base_x) as u16
            };
            base += base_inc;
        }
        x += dx;
    }
}

/// `av1_highbd_dr_prediction_z2_c` (dx > 0, dy > 0). Highbd analogue of [`z2`].
#[allow(clippy::too_many_arguments)]
pub fn z2_high(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    above: &EdgeRef16,
    left: &EdgeRef16,
    up_above: i32,
    up_left: i32,
    dx: i32,
    dy: i32,
) {
    let min_base_x = -(1 << up_above);
    let frac_bits_x = 6 - up_above;
    let frac_bits_y = 6 - up_left;
    for r in 0..bh {
        for c in 0..bw {
            let y = (r + 1) as i32;
            let x = ((c as i32) << 6) - y * dx;
            let base_x = x >> frac_bits_x;
            let val = if base_x >= min_base_x {
                let shift = ((x * (1 << up_above)) & 0x3F) >> 1;
                rpo2_5_16(above.at(base_x) * (32 - shift) + above.at(base_x + 1) * shift)
            } else {
                let x2 = (c + 1) as i32;
                let y2 = ((r as i32) << 6) - x2 * dy;
                let base_y = y2 >> frac_bits_y;
                let shift = ((y2 * (1 << up_left)) & 0x3F) >> 1;
                rpo2_5_16(left.at(base_y) * (32 - shift) + left.at(base_y + 1) * shift)
            };
            dst[r * stride + c] = val;
        }
    }
}

/// `av1_highbd_dr_prediction_z3_c` (dx == 1, dy > 0). Highbd analogue of [`z3`].
pub fn z3_high(
    dst: &mut [u16],
    stride: usize,
    bw: usize,
    bh: usize,
    left: &EdgeRef16,
    up: i32,
    dy: i32,
) {
    let max_base_y = ((bw + bh) as i32 - 1) << up;
    let frac_bits = 6 - up;
    let base_inc = 1 << up;
    let mut y = dy;
    for c in 0..bw {
        let mut base = y >> frac_bits;
        let shift = ((y << up) & 0x3F) >> 1;
        for r in 0..bh {
            if base < max_base_y {
                dst[r * stride + c] =
                    rpo2_5_16(left.at(base) * (32 - shift) + left.at(base + 1) * shift);
                base += base_inc;
            } else {
                let fillv = left.at(max_base_y) as u16;
                for rr in r..bh {
                    dst[rr * stride + c] = fillv;
                }
                break;
            }
        }
        y += dy;
    }
}
