//! Forward DCT 1-D butterflies, bit-exact ports of libaom v3.14.1
//! `av1/encoder/av1_fwd_txfm1d.c`.
//!
//! Fidelity notes (must not "clean up"):
//! * `half_btf` computes each product in **wrapping 32-bit** then widens to i64,
//!   exactly as C `(int64_t)(w0 * in0)`. The libaom source comment explicitly
//!   documents that wrapping 32-bit multiply yields the correct final result.
//! * In the default production config `CONFIG_COEFFICIENT_RANGE_CHECKING` and
//!   `DO_RANGE_CHECK_CLAMP` are both OFF, so `av1_range_check_buf` and
//!   `range_check_value` are no-ops. We omit them (documented divergence: none —
//!   behaviour is identical for the oracle build config).

use crate::cospi::cospi_arr;

/// libaom `half_btf(w0, in0, w1, in1, bit)` — bit-exact.
#[inline]
pub fn half_btf(w0: i32, in0: i32, w1: i32, in1: i32, bit: i32) -> i32 {
    // C: (int64_t)(w0 * in0) + (int64_t)(w1 * in1)
    // The inner multiply is 32-bit and may wrap; that is intentional.
    let result_64 = (w0.wrapping_mul(in0) as i64) + (w1.wrapping_mul(in1) as i64);
    let intermediate = result_64 + (1i64 << (bit - 1));
    (intermediate >> bit) as i32
}

/// libaom `round_shift(value, bit)` — bit-exact. (Not used by fdct4; provided
/// for the rest of the transform family.)
#[inline]
pub fn round_shift(value: i64, bit: i32) -> i32 {
    debug_assert!(bit >= 1);
    ((value + (1i64 << (bit - 1))) >> bit) as i32
}

/// Bit-exact port of `av1_fdct4`.
///
/// `output` must have length >= 4. `stage_range` is accepted for signature
/// parity with libaom but is only consulted by the (disabled) range checker.
pub fn av1_fdct4(input: &[i32], output: &mut [i32], cos_bit: i32, _stage_range: &[i8]) {
    let cospi = cospi_arr(cos_bit);

    // stage 1
    let mut bf1 = [0i32; 4];
    bf1[0] = input[0].wrapping_add(input[3]);
    bf1[1] = input[1].wrapping_add(input[2]);
    bf1[2] = input[2].wrapping_neg().wrapping_add(input[1]);
    bf1[3] = input[3].wrapping_neg().wrapping_add(input[0]);

    // stage 2
    let mut step = [0i32; 4];
    step[0] = half_btf(cospi[32], bf1[0], cospi[32], bf1[1], cos_bit);
    step[1] = half_btf(-cospi[32], bf1[1], cospi[32], bf1[0], cos_bit);
    step[2] = half_btf(cospi[48], bf1[2], cospi[16], bf1[3], cos_bit);
    step[3] = half_btf(cospi[48], bf1[3], -cospi[16], bf1[2], cos_bit);

    // stage 3 (permutation)
    output[0] = step[0];
    output[1] = step[2];
    output[2] = step[1];
    output[3] = step[3];
}
