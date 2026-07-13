//! Hand-ported forward transforms whose structure differs from the regular
//! ping-pong butterflies: `av1_fadst4` (sinpi, scalar temporaries) and the
//! identity transforms. Bit-exact ports of libaom v3.14.1
//! `av1/encoder/av1_fwd_txfm1d.c`. All harness-verified.

use crate::cospi::{sinpi_arr, NEW_SQRT2, NEW_SQRT2_BITS};
use crate::fdct::round_shift;

// ============================ INVERSE ==================================

/// Bit-exact port of `av1_iadst4` (`av1_inv_txfm1d.c`). Uses int64 throughout;
/// `range_check_value64` is a no-op in the production config.
pub fn av1_iadst4(input: &[i32], output: &mut [i32], cos_bit: i32, _stage_range: &[i8]) {
    let bit = cos_bit;
    let sinpi = sinpi_arr(bit);
    let x0 = input[0] as i64;
    let x2 = input[2] as i64;
    let x3 = input[3] as i64;

    if (input[0] | input[1] | input[2] | input[3]) == 0 {
        output[0] = 0;
        output[1] = 0;
        output[2] = 0;
        output[3] = 0;
        return;
    }

    let sp = |k: usize| sinpi[k] as i64;

    // stage 1
    let s0 = sp(1).wrapping_mul(x0);
    let s1 = sp(2).wrapping_mul(x0);
    let s2 = sp(3).wrapping_mul(input[1] as i64);
    let s3 = sp(4).wrapping_mul(x2);
    let s4 = sp(1).wrapping_mul(x2);
    let s5 = sp(2).wrapping_mul(x3);
    let s6 = sp(4).wrapping_mul(x3);
    // stage 2
    let s7 = (x0.wrapping_sub(x2)).wrapping_add(x3);
    // stage 3 (note the C reuse: s3 <- old s2, s2 <- sinpi[3]*s7)
    let s0 = s0.wrapping_add(s3);
    let s1 = s1.wrapping_sub(s4);
    let s3 = s2;
    let s2 = sp(3).wrapping_mul(s7);
    // stage 4
    let s0 = s0.wrapping_add(s5);
    let s1 = s1.wrapping_sub(s6);
    // stage 5
    let x0 = s0.wrapping_add(s3);
    let x1 = s1.wrapping_add(s3);
    let x2 = s2;
    let x3 = s0.wrapping_add(s1);
    // stage 6
    let x3 = x3.wrapping_sub(s3);

    output[0] = round_shift(x0, bit);
    output[1] = round_shift(x1, bit);
    output[2] = round_shift(x2, bit);
    output[3] = round_shift(x3, bit);
}

/// `av1_iidentity4_c`
pub fn av1_iidentity4(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..4 {
        output[i] = round_shift(NEW_SQRT2 as i64 * input[i] as i64, NEW_SQRT2_BITS);
    }
}

/// `av1_iidentity8_c`
pub fn av1_iidentity8(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..8 {
        output[i] = (input[i] as i64 * 2) as i32;
    }
}

/// `av1_iidentity16_c`
pub fn av1_iidentity16(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..16 {
        output[i] = round_shift(NEW_SQRT2 as i64 * 2 * input[i] as i64, NEW_SQRT2_BITS);
    }
}

/// `av1_iidentity32_c`
pub fn av1_iidentity32(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..32 {
        output[i] = (input[i] as i64 * 4) as i32;
    }
}

/// Bit-exact port of `av1_fadst4`.
pub fn av1_fadst4(input: &[i32], output: &mut [i32], cos_bit: i32, _stage_range: &[i8]) {
    let bit = cos_bit;
    let sinpi = sinpi_arr(bit);
    let (x0, x1, x2, x3) = (input[0], input[1], input[2], input[3]);

    if (x0 | x1 | x2 | x3) == 0 {
        output[0] = 0;
        output[1] = 0;
        output[2] = 0;
        output[3] = 0;
        return;
    }

    // stage 1 — all products are wrapping 32-bit (range_check is a no-op).
    let s0 = sinpi[1].wrapping_mul(x0);
    let s1 = sinpi[4].wrapping_mul(x0);
    let s2 = sinpi[2].wrapping_mul(x1);
    let s3 = sinpi[1].wrapping_mul(x1);
    let s4 = sinpi[3].wrapping_mul(x2);
    let s5 = sinpi[4].wrapping_mul(x3);
    let s6 = sinpi[2].wrapping_mul(x3);
    let s7 = x0.wrapping_add(x1);

    // stage 2
    let s7 = s7.wrapping_sub(x3);

    // stage 3
    let x0 = s0.wrapping_add(s2);
    let x1 = sinpi[3].wrapping_mul(s7);
    let x2 = s1.wrapping_sub(s3);
    let x3 = s4;

    // stage 4
    let x0 = x0.wrapping_add(s5);
    let x2 = x2.wrapping_add(s6);

    // stage 5
    let s0 = x0.wrapping_add(x3);
    let s1 = x1;
    let s2 = x2.wrapping_sub(x3);
    let s3 = x2.wrapping_sub(x0);

    // stage 6
    let s3 = s3.wrapping_add(x3);

    // 1-D transform scaling factor is sqrt(2).
    output[0] = round_shift(s0 as i64, bit);
    output[1] = round_shift(s1 as i64, bit);
    output[2] = round_shift(s2 as i64, bit);
    output[3] = round_shift(s3 as i64, bit);
}

/// `av1_fidentity4_c`
pub fn av1_fidentity4(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..4 {
        output[i] = round_shift(input[i] as i64 * NEW_SQRT2 as i64, NEW_SQRT2_BITS);
    }
}

/// `av1_fidentity8_c`
pub fn av1_fidentity8(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..8 {
        output[i] = input[i].wrapping_mul(2);
    }
}

/// `av1_fidentity16_c`
pub fn av1_fidentity16(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..16 {
        output[i] = round_shift(input[i] as i64 * 2 * NEW_SQRT2 as i64, NEW_SQRT2_BITS);
    }
}

/// `av1_fidentity32_c`
pub fn av1_fidentity32(input: &[i32], output: &mut [i32], _cos_bit: i32, _stage_range: &[i8]) {
    for i in 0..32 {
        output[i] = input[i].wrapping_mul(4);
    }
}
