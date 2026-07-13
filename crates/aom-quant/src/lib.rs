//! aom-quant — bit-exact AV1 quantization kernels (port of libaom v3.14.1).
//!
//! Validated byte-for-byte against the C reference by differential harnesses in
//! `tests/`. Starts with the `av1_quantize_fp` family (the encoder fast-path
//! quantizer, no quant-matrix), which is the stage directly downstream of the
//! forward transform.


#![forbid(unsafe_code)]
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

const AOM_QM_BITS: i32 = 5;

/// Bit-exact port of `aom_quantize_b_helper_c` (`aom_dsp/quantize.c`) for the
/// no-quant-matrix case (`wt = iwt = 1<<AOM_QM_BITS`). The "b" quantizer with a
/// dead-zone (`zbin`) pre-scan and two-step `quant`/`quant_shift`.
#[allow(clippy::too_many_arguments)]
pub fn aom_quantize_b_no_qmatrix(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    let wt: i32 = 1 << AOM_QM_BITS; // 32; no quant matrix
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);

    // Pre-scan pass (from the end): trim trailing dead-zone coefficients.
    let mut non_zero_count = n as i32;
    for i in (0..n).rev() {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let c = coeff[rc].wrapping_mul(wt);
        if c < zbins[ac].wrapping_mul(1 << AOM_QM_BITS) && c > nzbins[ac].wrapping_mul(1 << AOM_QM_BITS) {
            non_zero_count -= 1;
        } else {
            break;
        }
    }

    let mut eob: i32 = -1;
    for i in 0..non_zero_count as usize {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let coeff_sign = aomsign(coeff_v);
        let abs_coeff = (coeff_v ^ coeff_sign).wrapping_sub(coeff_sign);
        if abs_coeff.wrapping_mul(wt) >= (zbins[ac] << AOM_QM_BITS) {
            let clamped = (abs_coeff.wrapping_add(round_power_of_two(round[ac] as i32, log_scale)))
                .clamp(i16::MIN as i32, i16::MAX as i32);
            let mut tmp = clamped as i64;
            tmp *= wt as i64;
            let tmp32 = ((((tmp * quant[ac] as i64) >> 16) + tmp) * quant_shift[ac] as i64
                >> (16 - log_scale + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (tmp32 ^ coeff_sign).wrapping_sub(coeff_sign);
            // iwt = 32 -> dequant = (dequant[ac]*32 + 16) >> 5 == dequant[ac]
            let dequant_v = (dequant[ac] as i32 * wt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
            let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ coeff_sign).wrapping_sub(coeff_sign);
            if tmp32 != 0 {
                eob = i as i32;
            }
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `aom_quantize_b_helper_c` (`aom_dsp/quantize.c`) *with* a
/// quant matrix: `qm[rc]` weights the coefficient (`wt`), `iqm[rc]` weights the
/// dequant (`iwt`). This is the general form of [`aom_quantize_b_no_qmatrix`]
/// (`wt = iwt = 1<<AOM_QM_BITS`). `qm`/`iqm` are indexed by raster position `rc`
/// (same length as `coeff`). Integer widths mirror the C exactly (`coeff*wt`,
/// `abs_coeff*wt`, `dequant` are 32-bit; the round/quant chain is 64-bit).
// Scan/eob indexing (reverse-break pre-scan + `eob = i` forward pass) mirrors the
// C `for` loops 1:1, so the index is load-bearing — keep the explicit `scan[i]`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn aom_quantize_b_qm(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);

    // Pre-scan pass (from the end): trim trailing dead-zone coefficients.
    let mut non_zero_count = n as i32;
    for i in (0..n).rev() {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let wt = qm[rc] as i32;
        let c = coeff[rc].wrapping_mul(wt);
        if c < zbins[ac].wrapping_mul(1 << AOM_QM_BITS) && c > nzbins[ac].wrapping_mul(1 << AOM_QM_BITS) {
            non_zero_count -= 1;
        } else {
            break;
        }
    }

    let mut eob: i32 = -1;
    for i in 0..non_zero_count as usize {
        let rc = scan[i] as usize;
        let ac = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let coeff_sign = aomsign(coeff_v);
        let abs_coeff = (coeff_v ^ coeff_sign).wrapping_sub(coeff_sign);
        let wt = qm[rc] as i32;
        if abs_coeff.wrapping_mul(wt) >= (zbins[ac] << AOM_QM_BITS) {
            let clamped = (abs_coeff.wrapping_add(round_power_of_two(round[ac] as i32, log_scale)))
                .clamp(i16::MIN as i32, i16::MAX as i32);
            let mut tmp = clamped as i64;
            tmp *= wt as i64;
            let tmp32 = (((((tmp * quant[ac] as i64) >> 16) + tmp) * quant_shift[ac] as i64)
                >> (16 - log_scale + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (tmp32 ^ coeff_sign).wrapping_sub(coeff_sign);
            let iwt = iqm[rc] as i32;
            let dequant_v = (dequant[ac] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
            let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ coeff_sign).wrapping_sub(coeff_sign);
            if tmp32 != 0 {
                eob = i as i32;
            }
        }
    }
    (eob + 1) as u16
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

/// Bit-exact port of `highbd_quantize_fp_helper_c` (`av1/encoder/av1_quantize.c`)
/// for the no-quant-matrix path. Highbd (10/12-bit) FP quantizer: like the lowbd
/// path but with 64-bit arithmetic throughout (no int16 clamp on the rounded
/// coefficient). Returns eob.
#[allow(clippy::too_many_arguments)]
pub fn av1_highbd_quantize_fp_no_qmatrix(
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
    let shift = 16 - log_scale;
    let lsr = [
        round_power_of_two(round[0] as i32, log_scale),
        round_power_of_two(round[1] as i32, log_scale),
    ];
    let mut eob: i32 = -1;
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let sign = aomsign(coeff_v);
        let abs_coeff = (coeff_v ^ sign).wrapping_sub(sign);
        if ((abs_coeff as i64) << (1 + log_scale)) >= dequant[rc01] as i64 {
            let tmp = abs_coeff as i64 + lsr[rc01] as i64;
            let abs_qcoeff = ((tmp * quant[rc01] as i64) >> shift) as i32;
            qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
            let abs_dqcoeff = abs_qcoeff.wrapping_mul(dequant[rc01] as i32) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
            if abs_qcoeff != 0 {
                eob = i as i32;
            }
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `aom_highbd_quantize_b_helper_c` (`aom_dsp/quantize.c`) for
/// the no-quant-matrix case (`wt = iwt = 1<<AOM_QM_BITS`). Highbd "b" quantizer:
/// dead-zone (`zbin`) pre-scan + two-step `quant`/`quant_shift`, 64-bit.
#[allow(clippy::too_many_arguments)]
pub fn aom_highbd_quantize_b_no_qmatrix(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];
    let wt = 1i64 << AOM_QM_BITS;

    // Pre-scan pass (wt = 32): keep coeffs outside the ZBIN dead-zone.
    let mut idx_arr = Vec::with_capacity(n);
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let coeff_w = coeff[rc] as i64 * wt;
        if coeff_w >= zbins[(rc != 0) as usize] as i64 * wt
            || coeff_w <= nzbins[(rc != 0) as usize] as i64 * wt
        {
            idx_arr.push(i);
        }
    }

    let mut eob: i32 = -1;
    for &ii in &idx_arr {
        let rc = scan[ii] as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let sign = aomsign(coeff_v);
        let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        let tmp1 = abs_coeff + round_power_of_two(round[rc01] as i32, log_scale) as i64;
        let tmpw = tmp1 * wt;
        let tmp2 = ((tmpw * quant[rc01] as i64) >> 16) + tmpw;
        let abs_qcoeff = ((tmp2 * quant_shift[rc01] as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
        qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
        // iwt = 32: dequant = (dequant*32 + 16) >> 5 == dequant.
        let dq = ((dequant[rc01] as i32 * (1 << AOM_QM_BITS)) + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let abs_dqcoeff = abs_qcoeff.wrapping_mul(dq) >> log_scale;
        dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
        if abs_qcoeff != 0 {
            eob = ii as i32;
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `aom_highbd_quantize_b_helper_c` (`aom_dsp/quantize.c`)
/// *with* a quant matrix: general form of [`aom_highbd_quantize_b_no_qmatrix`]
/// with per-position `wt = qm[rc]` / `iwt = iqm[rc]`. Integer widths mirror the
/// C exactly (`coeff*wt` and `dequant` are 32-bit; the quant chain is 64-bit).
#[allow(clippy::too_many_arguments)]
pub fn aom_highbd_quantize_b_qm(
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let zbins = [
        round_power_of_two(zbin[0] as i32, log_scale),
        round_power_of_two(zbin[1] as i32, log_scale),
    ];
    let nzbins = [-zbins[0], -zbins[1]];

    // Pre-scan pass (forward): keep coeffs outside the ZBIN dead-zone. `coeff*wt`
    // is 32-bit in C.
    let mut idx_arr = Vec::with_capacity(n);
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let ac = (rc != 0) as usize;
        let wt = qm[rc] as i32;
        let coeff_w = coeff[rc].wrapping_mul(wt);
        if coeff_w >= zbins[ac].wrapping_mul(1 << AOM_QM_BITS)
            || coeff_w <= nzbins[ac].wrapping_mul(1 << AOM_QM_BITS)
        {
            idx_arr.push(i);
        }
    }

    let mut eob: i32 = -1;
    for &ii in &idx_arr {
        let rc = scan[ii] as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let sign = aomsign(coeff_v);
        let wt = qm[rc] as i64;
        let iwt = iqm[rc] as i32;
        let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        let tmp1 = abs_coeff + round_power_of_two(round[rc01] as i32, log_scale) as i64;
        let tmpw = tmp1 * wt;
        let tmp2 = ((tmpw * quant[rc01] as i64) >> 16) + tmpw;
        let abs_qcoeff = ((tmp2 * quant_shift[rc01] as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
        qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
        let dq = (dequant[rc01] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let abs_dqcoeff = abs_qcoeff.wrapping_mul(dq) >> log_scale;
        dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
        if abs_qcoeff != 0 {
            eob = ii as i32;
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of the QM branch of `quantize_fp_helper_c`
/// (`av1/encoder/av1_quantize.c`) — the lowbd VarDCT-FP quantizer with a quant
/// matrix. `wt = qm[rc]` / `iwt = iqm[rc]` per raster position. The rounded
/// coefficient is clamped to the i16 range (unlike the highbd FP variant).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_fp_qm(
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let rounding = [
        round_power_of_two(round[0] as i32, log_scale) as i64,
        round_power_of_two(round[1] as i32, log_scale) as i64,
    ];
    let mut eob: i32 = -1;
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let wt = qm[rc] as i64;
        let iwt = iqm[rc] as i32;
        let dequant_v = (dequant[rc01] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let sign = aomsign(coeff_v);
        let mut abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        let mut tmp32 = 0i32;
        if abs_coeff * wt >= ((dequant[rc01] as i64) << (AOM_QM_BITS - (1 + log_scale))) {
            abs_coeff += rounding[rc01];
            abs_coeff = abs_coeff.clamp(i16::MIN as i64, i16::MAX as i64);
            tmp32 = ((abs_coeff * wt * quant[rc01] as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (tmp32 ^ sign).wrapping_sub(sign);
            let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
        }
        if tmp32 != 0 {
            eob = i as i32;
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of the QM branch of `highbd_quantize_fp_helper_c`
/// (`av1/encoder/av1_quantize.c`) — the highbd (10/12-bit) VarDCT-FP quantizer
/// with a quant matrix. No i16 clamp on the rounded coefficient; the quant chain
/// is 64-bit throughout.
#[allow(clippy::too_many_arguments)]
pub fn av1_highbd_quantize_fp_qm(
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    log_scale: i32,
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let shift = 16 - log_scale;
    let mut eob: i32 = -1;
    for (i, &sc) in scan[..n].iter().enumerate() {
        let rc = sc as usize;
        let rc01 = (rc != 0) as usize;
        let coeff_v = coeff[rc];
        let wt = qm[rc] as i64;
        let iwt = iqm[rc] as i32;
        let dequant_v = (dequant[rc01] as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
        let sign = aomsign(coeff_v);
        let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
        if abs_coeff * wt >= ((dequant[rc01] as i64) << (AOM_QM_BITS - (1 + log_scale))) {
            let tmp = abs_coeff + round_power_of_two(round[rc01] as i32, log_scale) as i64;
            let abs_qcoeff = ((tmp * quant[rc01] as i64 * wt) >> (shift + AOM_QM_BITS)) as i32;
            qcoeff[rc] = (abs_qcoeff ^ sign).wrapping_sub(sign);
            let abs_dqcoeff = abs_qcoeff.wrapping_mul(dequant_v) >> log_scale;
            dqcoeff[rc] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
            if abs_qcoeff != 0 {
                eob = i as i32;
            }
        }
    }
    (eob + 1) as u16
}

/// Bit-exact port of `quantize_dc` (`av1/encoder/av1_quantize.c`) — the DC-only
/// quantizer (`AV1_XFORM_QUANT_DC`): quantizes coefficient 0 only, zeroing the
/// rest. `quant`/`dequant` are the DC scalars; `round[0]` is the DC round. `qm`/
/// `iqm` (when `Some`) weight position 0. Lowbd: the rounded coeff is clamped to
/// the i16 range. Returns eob (0 or 1).
#[allow(clippy::too_many_arguments)]
pub fn av1_quantize_dc(
    round: &[i16; 2],
    quant: i16,
    dequant: i16,
    log_scale: i32,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let wt = qm.map_or(1 << AOM_QM_BITS, |m| m[0] as i32);
    let iwt = iqm.map_or(1 << AOM_QM_BITS, |m| m[0] as i32);
    let coeff_v = coeff[0];
    let sign = aomsign(coeff_v);
    let abs_coeff = (coeff_v ^ sign).wrapping_sub(sign);
    let clamped = (abs_coeff.wrapping_add(round_power_of_two(round[0] as i32, log_scale)))
        .clamp(i16::MIN as i32, i16::MAX as i32);
    let tmp32 = ((clamped as i64 * wt as i64 * quant as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
    qcoeff[0] = (tmp32 ^ sign).wrapping_sub(sign);
    let dequant_v = (dequant as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
    let abs_dqcoeff = tmp32.wrapping_mul(dequant_v) >> log_scale;
    dqcoeff[0] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
    (tmp32 != 0) as u16
}

/// Bit-exact port of `highbd_quantize_dc` (`av1/encoder/av1_quantize.c`) — the
/// highbd (10/12-bit) DC-only quantizer. Like [`av1_quantize_dc`] but 64-bit
/// with no i16 clamp on the rounded coefficient. Returns eob (0 or 1).
#[allow(clippy::too_many_arguments)]
pub fn av1_highbd_quantize_dc(
    round: &[i16; 2],
    quant: i16,
    dequant: i16,
    log_scale: i32,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
    coeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    let n = coeff.len();
    qcoeff[..n].fill(0);
    dqcoeff[..n].fill(0);
    let wt = qm.map_or(1 << AOM_QM_BITS, |m| m[0] as i64);
    let iwt = iqm.map_or(1 << AOM_QM_BITS, |m| m[0] as i32);
    let coeff_v = coeff[0];
    let sign = aomsign(coeff_v);
    let abs_coeff = ((coeff_v ^ sign).wrapping_sub(sign)) as i64;
    let tmp = abs_coeff + round_power_of_two(round[0] as i32, log_scale) as i64;
    let tmpw = tmp * wt;
    let abs_qcoeff = ((tmpw * quant as i64) >> (16 - log_scale + AOM_QM_BITS)) as i32;
    qcoeff[0] = (abs_qcoeff ^ sign).wrapping_sub(sign);
    let dequant_v = (dequant as i32 * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
    let abs_dqcoeff = abs_qcoeff.wrapping_mul(dequant_v) >> log_scale;
    dqcoeff[0] = (abs_dqcoeff ^ sign).wrapping_sub(sign);
    (abs_qcoeff != 0) as u16
}
