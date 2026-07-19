//! `av1_build_quantizer` (av1/encoder/av1_quantize.c): derive the complete
//! encoder quantizer tables — `quant`/`quant_shift` (the fixed-point inverse),
//! `quant_fp`/`round_fp` (the fast-path quantizer), `zbin`, `round`, and the
//! `dequant` step — for every qindex, per plane (Y/U/V) and coefficient class
//! (dc = lane 0, ac = lanes 1..8, repeated to SIMD width), from the per-plane
//! delta-q parameters, bit depth, and sharpness. Bit-exact port of libaom
//! v3.14.1, validated against the real exported C function.

use crate::quant::quant_common::{av1_ac_quant_qtx, av1_dc_quant_qtx};
use crate::quant::round_power_of_two;

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

/// One plane's quantizer rows for a given qindex — the seven `*_QTX` pointers
/// `set_q_index` (av1/encoder/av1_quantize.c) installs into that plane's
/// `MACROBLOCK_PLANE`. Each row is the 8-lane `[dc, ac, ac, ...]` layout.
#[derive(Clone, Copy, Debug)]
pub struct PlaneQuantRows<'a> {
    /// `quant_QTX` — the B-quantizer multiplier row.
    pub quant: &'a [i16; 8],
    /// `quant_fp_QTX` — the FP-quantizer multiplier row.
    pub quant_fp: &'a [i16; 8],
    /// `round_fp_QTX` — the FP rounding row.
    pub round_fp: &'a [i16; 8],
    /// `quant_shift_QTX` — the B-quantizer shift row.
    pub quant_shift: &'a [i16; 8],
    /// `zbin_QTX` — the B-quantizer zero-bin row.
    pub zbin: &'a [i16; 8],
    /// `round_QTX` — the B rounding row.
    pub round: &'a [i16; 8],
    /// `dequant_QTX` — the dequantization step row.
    pub dequant: &'a [i16; 8],
}

/// `set_q_index` (av1/encoder/av1_quantize.c, static): select the per-`qindex`
/// quantizer rows for `plane` (0 = Y, 1 = U, 2 = V) out of the
/// [`av1_build_quantizer`]-filled tables — exactly the rows the C function
/// assigns to `x->plane[plane]`. Bit-exact vs C (differential-tested).
pub fn set_q_index<'a>(
    quants: &'a Quants,
    dequants: &'a Dequants,
    qindex: usize,
    plane: usize,
) -> PlaneQuantRows<'a> {
    assert!(qindex < QINDEX_RANGE);
    match plane {
        0 => PlaneQuantRows {
            quant: &quants.y_quant[qindex],
            quant_fp: &quants.y_quant_fp[qindex],
            round_fp: &quants.y_round_fp[qindex],
            quant_shift: &quants.y_quant_shift[qindex],
            zbin: &quants.y_zbin[qindex],
            round: &quants.y_round[qindex],
            dequant: &dequants.y_dequant_qtx[qindex],
        },
        1 => PlaneQuantRows {
            quant: &quants.u_quant[qindex],
            quant_fp: &quants.u_quant_fp[qindex],
            round_fp: &quants.u_round_fp[qindex],
            quant_shift: &quants.u_quant_shift[qindex],
            zbin: &quants.u_zbin[qindex],
            round: &quants.u_round[qindex],
            dequant: &dequants.u_dequant_qtx[qindex],
        },
        2 => PlaneQuantRows {
            quant: &quants.v_quant[qindex],
            quant_fp: &quants.v_quant_fp[qindex],
            round_fp: &quants.v_round_fp[qindex],
            quant_shift: &quants.v_quant_shift[qindex],
            zbin: &quants.v_zbin[qindex],
            round: &quants.v_round[qindex],
            dequant: &dequants.v_dequant_qtx[qindex],
        },
        _ => panic!("plane must be 0..3"),
    }
}

// ---------------------------------------------------------------------------
// av1_set_quantizer — the frame quantizer settings (chroma delta-q + QM levels)
// ---------------------------------------------------------------------------

/// The encoder tuning arms `av1_set_quantizer` branches on
/// (`aom_tune_metric`, aom/aomcx.h — only the IQ/SSIMULACRA2 distinction
/// matters here; every other tune takes the `Psnr` arm).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantTuning {
    /// Every tune other than IQ / SSIMULACRA2.
    Psnr,
    /// `AOM_TUNE_IQ` (10).
    Iq,
    /// `AOM_TUNE_SSIMULACRA2` (11).
    Ssimulacra2,
}

/// `av1_set_quantizer`'s outputs (av1/encoder/av1_quantize.c:878): the frame
/// `CommonQuantParams` fields the header + quantizer build consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantizerSettings {
    /// `quant_params->base_qindex = AOMMAX(delta_q_present_flag, q)`.
    pub base_qindex: i32,
    /// `y_dc_delta_q` (always 0 on this path).
    pub y_dc_delta_q: i32,
    pub u_dc_delta_q: i32,
    pub u_ac_delta_q: i32,
    pub v_dc_delta_q: i32,
    pub v_ac_delta_q: i32,
    /// `qmatrix_level_{y,u,v}` — ALWAYS computed (C does so regardless of
    /// `enable_qm`; they only reach the bitstream when `using_qmatrix`).
    pub qmatrix_level_y: i32,
    pub qmatrix_level_u: i32,
    pub qmatrix_level_v: i32,
}

/// `av1_set_quantizer` (av1/encoder/av1_quantize.c:878-1032) minus the
/// HDR-deltaq arm (`enable_hdr_deltaq` — 10-bit BT.2020-only, out of the
/// stills envelope): the frame base qindex, the chroma delta-q derivation,
/// and the QM-level formula selection.
///
/// Chroma delta-q (`--enable-chroma-deltaq`, disabled in lossless `q == 0`):
/// - tune=IQ/SSIMULACRA2 (empirically derived ramps, av1_quantize.c:891-956):
///   - 4:2:0 (ss 1,1): `dc = ac = -clamp(base/2 - 14, 0, offset)` with
///     `offset = 20` for SSIMULACRA2 else `16` (chroma quality boost).
///   - 4:2:2 (ss 1,0): `dc = 0, ac = clamp(base/2, 0, 6)`.
///   - 4:4:4 (ss 0,0): `dc = 0, ac = clamp(base/2, 0, 24)`.
/// - other tunes: the constant `2/2/2/2` (av1_quantize.c:961-965).
///
/// QM levels (av1_quantize.c:987-1031):
/// - tune=IQ/SSIMULACRA2: luma = `aom_get_qmlevel_luma_ssimulacra2` for
///   SSIMULACRA2, `aom_get_qmlevel_allintra` for IQ; chroma =
///   `aom_get_qmlevel_444_chroma` at 4:4:4, else `aom_get_qmlevel_allintra`.
/// - else allintra: `aom_get_qmlevel_allintra` both.
/// - else: `aom_get_qmlevel` both.
/// Chroma levels evaluate at `base_qindex + {u,v}_ac_delta_q`; `v` mirrors
/// `u` unless `separate_uv_delta_q`.
///
/// `mono` frames take `ss_x = ss_y = 1` in C (the 4:2:0 arm computes values;
/// the header writer never codes them for a single-plane stream).
#[allow(clippy::too_many_arguments)]
pub fn av1_set_quantizer(
    min_qmlevel: i32,
    max_qmlevel: i32,
    q: i32,
    enable_chroma_deltaq: bool,
    is_allintra: bool,
    tuning: QuantTuning,
    ss_x: i32,
    ss_y: i32,
    separate_uv_delta_q: bool,
    delta_q_present: bool,
) -> QuantizerSettings {
    use crate::quant::quant_common::{
        aom_get_qmlevel, aom_get_qmlevel_444_chroma, aom_get_qmlevel_allintra,
        aom_get_qmlevel_luma_ssimulacra2,
    };
    let base_qindex = (delta_q_present as i32).max(q);
    let iq_family = matches!(tuning, QuantTuning::Iq | QuantTuning::Ssimulacra2);

    let (u_dc, u_ac, v_dc, v_ac) = if enable_chroma_deltaq && q != 0 {
        if iq_family {
            let (dc, ac) = if ss_x == 1 && ss_y == 1 {
                // 4:2:0 (and mono): constant chroma boost, ramped down at
                // very low qindexes.
                let offset = if tuning == QuantTuning::Ssimulacra2 {
                    20
                } else {
                    16
                };
                let dc = -((base_qindex / 2) - 14).clamp(0, offset);
                (dc, dc)
            } else if ss_x == 1 && ss_y == 0 {
                // 4:2:2: constant chroma AC increase with low-q ramp-down.
                (0, (base_qindex / 2).clamp(0, 6))
            } else if ss_x == 0 && ss_y == 0 {
                // 4:4:4: constant chroma AC increase with low-q ramp-down.
                (0, (base_qindex / 2).clamp(0, 24))
            } else {
                // 4:4:0 — unreachable through the codec API (no such fmt).
                (0, 0)
            };
            (dc, ac, dc, ac)
        } else {
            // TODO(aomedia:2717) arm in C: constant 2 everywhere.
            (2, 2, 2, 2)
        }
    } else {
        (0, 0, 0, 0)
    };

    // Select the luma/chroma QM formulas (function-pointer selection in C).
    let luma = |qi: i32| -> i32 {
        match tuning {
            QuantTuning::Ssimulacra2 => aom_get_qmlevel_luma_ssimulacra2(qi, min_qmlevel, max_qmlevel),
            QuantTuning::Iq => aom_get_qmlevel_allintra(qi, min_qmlevel, max_qmlevel),
            QuantTuning::Psnr if is_allintra => aom_get_qmlevel_allintra(qi, min_qmlevel, max_qmlevel),
            QuantTuning::Psnr => aom_get_qmlevel(qi, min_qmlevel, max_qmlevel),
        }
    };
    let chroma = |qi: i32| -> i32 {
        if iq_family {
            if ss_x == 0 && ss_y == 0 {
                aom_get_qmlevel_444_chroma(qi, min_qmlevel, max_qmlevel)
            } else {
                aom_get_qmlevel_allintra(qi, min_qmlevel, max_qmlevel)
            }
        } else if is_allintra {
            aom_get_qmlevel_allintra(qi, min_qmlevel, max_qmlevel)
        } else {
            aom_get_qmlevel(qi, min_qmlevel, max_qmlevel)
        }
    };

    let qmatrix_level_y = luma(base_qindex);
    let qmatrix_level_u = chroma(base_qindex + u_ac);
    let qmatrix_level_v = if separate_uv_delta_q {
        chroma(base_qindex + v_ac)
    } else {
        qmatrix_level_u
    };

    QuantizerSettings {
        base_qindex,
        y_dc_delta_q: 0,
        u_dc_delta_q: u_dc,
        u_ac_delta_q: u_ac,
        v_dc_delta_q: v_dc,
        v_ac_delta_q: v_ac,
        qmatrix_level_y,
        qmatrix_level_u,
        qmatrix_level_v,
    }
}
