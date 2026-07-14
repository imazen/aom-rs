//! `av1_build_quantizer` (av1/encoder/av1_quantize.c): derive the complete
//! encoder quantizer tables — `quant`/`quant_shift` (the fixed-point inverse),
//! `quant_fp`/`round_fp` (the fast-path quantizer), `zbin`, `round`, and the
//! `dequant` step — for every qindex, per plane (Y/U/V) and coefficient class
//! (dc = lane 0, ac = lanes 1..8, repeated to SIMD width), from the per-plane
//! delta-q parameters, bit depth, and sharpness. Bit-exact port of libaom
//! v3.14.1, validated against the real exported C function.

use crate::quant_common::{av1_ac_quant_qtx, av1_dc_quant_qtx};
use crate::round_power_of_two;

/// `QINDEX_RANGE` (av1/common/quant_common.h): number of base qindex values.
pub const QINDEX_RANGE: usize = 256;

/// `QUANTS` (av1/encoder/av1_quantize.h): forward-quantizer tables. Each row
/// holds 8 lanes (SIMD width): lane 0 = dc, lane 1 = ac, lanes 2..8 = ac
/// repeated. All fields use the TX coefficient shift/scale (`_QTX`).
pub struct Quants {
    pub y_quant: [[i16; 8]; QINDEX_RANGE],
    pub y_quant_shift: [[i16; 8]; QINDEX_RANGE],
    pub y_zbin: [[i16; 8]; QINDEX_RANGE],
    pub y_round: [[i16; 8]; QINDEX_RANGE],
    pub y_quant_fp: [[i16; 8]; QINDEX_RANGE],
    pub u_quant_fp: [[i16; 8]; QINDEX_RANGE],
    pub v_quant_fp: [[i16; 8]; QINDEX_RANGE],
    pub y_round_fp: [[i16; 8]; QINDEX_RANGE],
    pub u_round_fp: [[i16; 8]; QINDEX_RANGE],
    pub v_round_fp: [[i16; 8]; QINDEX_RANGE],
    pub u_quant: [[i16; 8]; QINDEX_RANGE],
    pub v_quant: [[i16; 8]; QINDEX_RANGE],
    pub u_quant_shift: [[i16; 8]; QINDEX_RANGE],
    pub v_quant_shift: [[i16; 8]; QINDEX_RANGE],
    pub u_zbin: [[i16; 8]; QINDEX_RANGE],
    pub v_zbin: [[i16; 8]; QINDEX_RANGE],
    pub u_round: [[i16; 8]; QINDEX_RANGE],
    pub v_round: [[i16; 8]; QINDEX_RANGE],
}

/// `Dequants` (av1/encoder/av1_quantize.h): dequantization step per qindex,
/// same 8-lane dc/ac layout, TX scale.
pub struct Dequants {
    pub y_dequant_qtx: [[i16; 8]; QINDEX_RANGE],
    pub u_dequant_qtx: [[i16; 8]; QINDEX_RANGE],
    pub v_dequant_qtx: [[i16; 8]; QINDEX_RANGE],
}

const ZERO_TABLE: [[i16; 8]; QINDEX_RANGE] = [[0; 8]; QINDEX_RANGE];

impl Quants {
    /// All-zero tables, boxed (the struct is ~72 KiB; keep it off deep stacks).
    /// Filled by [`av1_build_quantizer`].
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            y_quant: ZERO_TABLE,
            y_quant_shift: ZERO_TABLE,
            y_zbin: ZERO_TABLE,
            y_round: ZERO_TABLE,
            y_quant_fp: ZERO_TABLE,
            u_quant_fp: ZERO_TABLE,
            v_quant_fp: ZERO_TABLE,
            y_round_fp: ZERO_TABLE,
            u_round_fp: ZERO_TABLE,
            v_round_fp: ZERO_TABLE,
            u_quant: ZERO_TABLE,
            v_quant: ZERO_TABLE,
            u_quant_shift: ZERO_TABLE,
            v_quant_shift: ZERO_TABLE,
            u_zbin: ZERO_TABLE,
            v_zbin: ZERO_TABLE,
            u_round: ZERO_TABLE,
            v_round: ZERO_TABLE,
        })
    }
}

impl Dequants {
    /// All-zero tables, boxed. Filled by [`av1_build_quantizer`].
    pub fn zeroed() -> Box<Self> {
        Box::new(Self {
            y_dequant_qtx: ZERO_TABLE,
            u_dequant_qtx: ZERO_TABLE,
            v_dequant_qtx: ZERO_TABLE,
        })
    }
}

/// `invert_quant` (av1_quantize.c): fixed-point inverse of the dequant step
/// `d`, split into a 16-bit multiplier (stored biased by `-2^16`) and a
/// power-of-two shift, such that `aom_quantize_b` computes
/// `((x * quant) >> 16 + x) * quant_shift >> 16` ~= `x / d`.
/// `d` is a dequant table value (>= 4, < 2^15), so `l <= 14` and the
/// `1 << (16 + l)` below stays in `i32` range.
fn invert_quant(quant: &mut i16, shift: &mut i16, d: i32) {
    let t = d as u32;
    let l = 31 - t.leading_zeros() as i32; // get_msb
    let m = 1 + (1i32 << (16 + l)) / d;
    *quant = (m - (1 << 16)) as i16;
    *shift = 1 << (16 - l);
}

/// `get_qzbin_factor` (av1_quantize.c): zero-bin scale factor in Q7 — 64 for
/// lossless (q == 0), else 84 below a per-bit-depth dc-quant threshold, 80 at
/// or above it.
fn get_qzbin_factor(q: i32, bit_depth: u8) -> i32 {
    let quant = i32::from(av1_dc_quant_qtx(q, 0, bit_depth));
    let threshold = match bit_depth {
        8 => 148,
        10 => 592,
        12 => 2368,
        _ => panic!("bit_depth must be 8, 10 or 12"),
    };
    if q == 0 {
        64
    } else if quant < threshold {
        84
    } else {
        80
    }
}

/// Bit-exact port of `av1_build_quantizer` (av1/encoder/av1_quantize.c):
/// fill `quants` + `deq` for all `QINDEX_RANGE` base qindex values from the
/// per-plane dc/ac delta-q values, `bit_depth` (8/10/12), and `sharpness`
/// (0..=7; non-zero lowers the rounding factors away from lossless).
#[allow(clippy::too_many_arguments)]
pub fn av1_build_quantizer(
    bit_depth: u8,
    y_dc_delta_q: i32,
    u_dc_delta_q: i32,
    u_ac_delta_q: i32,
    v_dc_delta_q: i32,
    v_ac_delta_q: i32,
    quants: &mut Quants,
    deq: &mut Dequants,
    sharpness: i32,
) {
    let sharpness_adjustment = 16 * (7 - sharpness) / 7;

    for q in 0..QINDEX_RANGE {
        let qi = q as i32;
        let qzbin_factor = get_qzbin_factor(qi, bit_depth);
        let mut qrounding_factor = if q == 0 { 64 } else { 48 };

        for i in 0..2 {
            let mut qrounding_factor_fp = 64;

            if sharpness != 0 && q != 0 {
                qrounding_factor = 64 - sharpness_adjustment;
                qrounding_factor_fp = 64 - sharpness_adjustment;
            }

            // y quantizer with TX scale
            let quant_qtx = i32::from(if i == 0 {
                av1_dc_quant_qtx(qi, y_dc_delta_q, bit_depth)
            } else {
                av1_ac_quant_qtx(qi, 0, bit_depth)
            });
            invert_quant(
                &mut quants.y_quant[q][i],
                &mut quants.y_quant_shift[q][i],
                quant_qtx,
            );
            quants.y_quant_fp[q][i] = ((1 << 16) / quant_qtx) as i16;
            quants.y_round_fp[q][i] = ((qrounding_factor_fp * quant_qtx) >> 7) as i16;
            quants.y_zbin[q][i] = round_power_of_two(qzbin_factor * quant_qtx, 7) as i16;
            quants.y_round[q][i] = ((qrounding_factor * quant_qtx) >> 7) as i16;
            deq.y_dequant_qtx[q][i] = quant_qtx as i16;

            // u quantizer with TX scale
            let quant_qtx = i32::from(if i == 0 {
                av1_dc_quant_qtx(qi, u_dc_delta_q, bit_depth)
            } else {
                av1_ac_quant_qtx(qi, u_ac_delta_q, bit_depth)
            });
            invert_quant(
                &mut quants.u_quant[q][i],
                &mut quants.u_quant_shift[q][i],
                quant_qtx,
            );
            quants.u_quant_fp[q][i] = ((1 << 16) / quant_qtx) as i16;
            quants.u_round_fp[q][i] = ((qrounding_factor_fp * quant_qtx) >> 7) as i16;
            quants.u_zbin[q][i] = round_power_of_two(qzbin_factor * quant_qtx, 7) as i16;
            quants.u_round[q][i] = ((qrounding_factor * quant_qtx) >> 7) as i16;
            deq.u_dequant_qtx[q][i] = quant_qtx as i16;

            // v quantizer with TX scale
            let quant_qtx = i32::from(if i == 0 {
                av1_dc_quant_qtx(qi, v_dc_delta_q, bit_depth)
            } else {
                av1_ac_quant_qtx(qi, v_ac_delta_q, bit_depth)
            });
            invert_quant(
                &mut quants.v_quant[q][i],
                &mut quants.v_quant_shift[q][i],
                quant_qtx,
            );
            quants.v_quant_fp[q][i] = ((1 << 16) / quant_qtx) as i16;
            quants.v_round_fp[q][i] = ((qrounding_factor_fp * quant_qtx) >> 7) as i16;
            quants.v_zbin[q][i] = round_power_of_two(qzbin_factor * quant_qtx, 7) as i16;
            quants.v_round[q][i] = ((qrounding_factor * quant_qtx) >> 7) as i16;
            deq.v_dequant_qtx[q][i] = quant_qtx as i16;
        }

        for i in 2..8 {
            // 8: SIMD width
            quants.y_quant[q][i] = quants.y_quant[q][1];
            quants.y_quant_fp[q][i] = quants.y_quant_fp[q][1];
            quants.y_round_fp[q][i] = quants.y_round_fp[q][1];
            quants.y_quant_shift[q][i] = quants.y_quant_shift[q][1];
            quants.y_zbin[q][i] = quants.y_zbin[q][1];
            quants.y_round[q][i] = quants.y_round[q][1];
            deq.y_dequant_qtx[q][i] = deq.y_dequant_qtx[q][1];

            quants.u_quant[q][i] = quants.u_quant[q][1];
            quants.u_quant_fp[q][i] = quants.u_quant_fp[q][1];
            quants.u_round_fp[q][i] = quants.u_round_fp[q][1];
            quants.u_quant_shift[q][i] = quants.u_quant_shift[q][1];
            quants.u_zbin[q][i] = quants.u_zbin[q][1];
            quants.u_round[q][i] = quants.u_round[q][1];
            deq.u_dequant_qtx[q][i] = deq.u_dequant_qtx[q][1];

            quants.v_quant[q][i] = quants.v_quant[q][1];
            quants.v_quant_fp[q][i] = quants.v_quant_fp[q][1];
            quants.v_round_fp[q][i] = quants.v_round_fp[q][1];
            quants.v_quant_shift[q][i] = quants.v_quant_shift[q][1];
            quants.v_zbin[q][i] = quants.v_zbin[q][1];
            quants.v_round[q][i] = quants.v_round[q][1];
            deq.v_dequant_qtx[q][i] = deq.v_dequant_qtx[q][1];
        }
    }
}
