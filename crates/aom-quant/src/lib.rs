//! aom-quant — bit-exact AV1 quantization kernels (port of libaom v3.14.1).
//!
//! Validated byte-for-byte against the C reference by differential harnesses in
//! `tests/`. Starts with the `av1_quantize_fp` family (the encoder fast-path
//! quantizer, no quant-matrix), which is the stage directly downstream of the
//! forward transform.

/// `ROUND_POWER_OF_TWO(value, n)` from `aom_ports/mem.h` — bit-exact.
/// Note `(1<<n)>>1` yields 0 at n=0, so this is well-defined for `log_scale==0`.
#[inline]
fn round_power_of_two(value: i32, n: i32) -> i32 {
    (value + ((1 << n) >> 1)) >> n
}

/// `AOMSIGN(x)`: -1 if negative, else 0.
#[inline]
fn aomsign(x: i32) -> i32 {
    if x < 0 {
        -1
    } else {
        0
    }
}

/// Bit-exact port of `av1_quantize_fp_no_qmatrix` (`av1/encoder/av1_quantize.c`).
/// This is the body of `av1_quantize_fp_c` / `_32x32_c` / `_64x64_c` for the
/// no-quant-matrix case (`log_scale` = 0 / 1 / 2 respectively).
///
/// Writes `qcoeff` (quantized) and `dqcoeff` (dequantized) and returns the EOB.
/// `quant`, `dequant`, `round` are the `[dc, ac]` parameter pairs; `scan` is the
/// coefficient scan order (length `coeff.len()`).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_no_qmatrix(
    quant: &[i16; 2],
    dequant: &[i16; 2],
    round: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let rounding = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    let mut eob: u16 = 0;
    for i in 0..n {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize; // dc uses index 0, ac uses index 1
        let thresh = dequant[ac] as i64;
        let coeff_v = coeff[rc];
        let coeff_sign = aomsign(coeff_v);
        // int arithmetic then widen, as in C.
        let mut abs_coeff = (coeff_v ^ coeff_sign).wrapping_sub(coeff_sign) as i64;
        let mut tmp32: i32 = 0;
        if (abs_coeff << (1 + log_scale)) >= thresh {
            abs_coeff = (abs_coeff + rounding[ac] as i64).clamp(i16::MIN as i64, i16::MAX as i64);
            tmp32 = ((abs_coeff * quant[ac] as i64) >> (16 - log_scale)) as i32;
            if tmp32 != 0 {
                qcoeff[rc] = (tmp32 ^ coeff_sign).wrapping_sub(coeff_sign);
                let abs_dqcoeff = tmp32.wrapping_mul(dequant[ac] as i32) >> log_scale;
                dqcoeff[rc] = (abs_dqcoeff ^ coeff_sign).wrapping_sub(coeff_sign);
            }
        }
        if tmp32 != 0 {
            eob = (i + 1) as u16;
        }
    }
    eob
}

/// `av1_quantize_fp` (log_scale 0). Signature mirrors the C entry (unused
/// `zbin`/`quant_shift`/`iscan` args omitted).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp(
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    scan: &[i16],
) -> u16 {
    av1_quantize_fp_no_qmatrix(quant, dequant, round, 0, scan, coeff, qcoeff, dqcoeff)
}

/// `av1_quantize_fp_32x32` (log_scale 1).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_32x32(
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    scan: &[i16],
) -> u16 {
    av1_quantize_fp_no_qmatrix(quant, dequant, round, 1, scan, coeff, qcoeff, dqcoeff)
}

/// `av1_quantize_fp_64x64` (log_scale 2).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_64x64(
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    scan: &[i16],
) -> u16 {
    av1_quantize_fp_no_qmatrix(quant, dequant, round, 2, scan, coeff, qcoeff, dqcoeff)
}
