//! Rate-distortion model primitives (libaom `av1/encoder/rd.c` + `rd.h`).
//! Fixed-point, bit-exact:
//! - the Laplacian `(rate, dist)` model the fast RD search uses
//!   ([`model_rd_from_var_lapndz`]);
//! - the exact RD cost macros ([`rdcost`], [`rdcost_neg_r`]);
//! - the qindex → RD-multiplier (lambda) derivation
//!   ([`av1_compute_rd_mult_based_on_qindex`], [`av1_compute_rd_mult`]).

use aom_dsp::quant::av1_dc_quant_qtx;

// Generated from av1/encoder/rd.c model_rd_norm tables (104 entries each).
const RATE_TAB_Q10: [i32; 104] = [
    65536, 6086, 5574, 5275, 5063, 4899, 4764, 4651, 4553, 4389, 4255, 4142, 4044, 3958, 3881,
    3811, 3748, 3635, 3538, 3453, 3376, 3307, 3244, 3186, 3133, 3037, 2952, 2877, 2809, 2747, 2690,
    2638, 2589, 2501, 2423, 2353, 2290, 2232, 2179, 2130, 2084, 2001, 1928, 1862, 1802, 1748, 1698,
    1651, 1608, 1530, 1460, 1398, 1342, 1290, 1243, 1199, 1159, 1086, 1021, 963, 911, 864, 821,
    781, 745, 680, 623, 574, 530, 490, 455, 424, 395, 345, 304, 269, 239, 213, 190, 171, 154, 126,
    104, 87, 73, 61, 52, 44, 38, 28, 21, 16, 12, 10, 8, 6, 5, 3, 2, 1, 1, 1, 0, 0,
];
const DIST_TAB_Q10: [i32; 104] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 21, 24, 26,
    29, 31, 34, 36, 39, 44, 49, 54, 59, 64, 69, 73, 78, 88, 97, 106, 115, 124, 133, 142, 151, 167,
    184, 200, 215, 231, 245, 260, 274, 301, 327, 351, 375, 397, 418, 439, 458, 495, 528, 559, 587,
    613, 637, 659, 680, 717, 749, 777, 801, 823, 842, 859, 874, 899, 919, 936, 949, 960, 969, 977,
    983, 994, 1001, 1006, 1010, 1013, 1015, 1017, 1018, 1020, 1022, 1022, 1023, 1023, 1023, 1024,
];
const XSQ_IQ_Q10: [i32; 104] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144, 160, 176, 192,
    208, 224, 256, 288, 320, 352, 384, 416, 448, 480, 544, 608, 672, 736, 800, 864, 928, 992, 1120,
    1248, 1376, 1504, 1632, 1760, 1888, 2016, 2272, 2528, 2784, 3040, 3296, 3552, 3808, 4064, 4576,
    5088, 5600, 6112, 6624, 7136, 7648, 8160, 9184, 10208, 11232, 12256, 13280, 14304, 15328,
    16352, 18400, 20448, 22496, 24544, 26592, 28640, 30688, 32736, 36832, 40928, 45024, 49120,
    53216, 57312, 61408, 65504, 73696, 81888, 90080, 98272, 106464, 114656, 122848, 131040, 147424,
    163808, 180192, 196576, 212960, 229344, 245728,
];

const AV1_PROB_COST_SHIFT: u32 = 9;

/// `model_rd_norm`: 4-MSB-sampled interpolation of the Laplacian rate/dist tables.
/// Returns `(r_q10, d_q10)`.
fn model_rd_norm(xsq_q10: i32) -> (i32, i32) {
    let tmp = (xsq_q10 >> 2) + 8;
    let k = (tmp as u32).ilog2() as i32 - 3; // get_msb(tmp) - 3
    let xq = ((k << 3) + ((tmp >> k) & 0x7)) as usize;
    let a_q10 = ((xsq_q10 - XSQ_IQ_Q10[xq]) << 10) >> (2 + k);
    let b_q10 = (1 << 10) - a_q10;
    let r_q10 = (RATE_TAB_Q10[xq] * b_q10 + RATE_TAB_Q10[xq + 1] * a_q10) >> 10;
    let d_q10 = (DIST_TAB_Q10[xq] * b_q10 + DIST_TAB_Q10[xq + 1] * a_q10) >> 10;
    (r_q10, d_q10)
}

/// `av1_model_rd_from_var_lapndz`: model the `(rate, dist)` of a Laplacian source
/// of variance `var`, block area `1<<n_log2` pixels, uniformly quantized with
/// step `qstep`. Fixed-point, bit-exact vs C.
pub fn model_rd_from_var_lapndz(var: i64, n_log2: u32, qstep: u32) -> (i32, i64) {
    if var == 0 {
        return (0, 0);
    }
    const MAX_XSQ_Q10: u64 = 245727;
    let num = ((qstep as u64 * qstep as u64) << (n_log2 + 10)) + (var as u64 >> 1);
    let xsq_q10 = (num / var as u64).min(MAX_XSQ_Q10) as i32;
    let (r_q10, d_q10) = model_rd_norm(xsq_q10);
    // ROUND_POWER_OF_TWO(r_q10 << n_log2, 10 - AV1_PROB_COST_SHIFT = 1).
    let shift = 10 - AV1_PROB_COST_SHIFT;
    let rate = ((((r_q10 as i64) << n_log2) + (1 << (shift - 1))) >> shift) as i32;
    let dist = (var * d_q10 as i64 + 512) >> 10;
    (rate, dist)
}

// ---------------------------------------------------------------------------
// RD cost macros (av1/encoder/rd.h)
// ---------------------------------------------------------------------------

/// `RDDIV_BITS` (av1/encoder/rd.h): distortion is weighted by `1 << RDDIV_BITS`.
const RDDIV_BITS: u32 = 7;

/// `ROUND_POWER_OF_TWO(value, n)` (`aom_ports/mem.h`) on a 64-bit value.
/// `(1 << n) >> 1` is 0 at `n == 0`, so this is well-defined there.
#[inline]
fn round_power_of_two_i64(value: i64, n: u32) -> i64 {
    (value + ((1i64 << n) >> 1)) >> n
}

/// `RDCOST(RM, R, D)` (av1/encoder/rd.h) — the exact rate-distortion cost.
///
/// `rm` is the RD multiplier (lambda), `rate` the rate in AV1's
/// `AV1_PROB_COST`-scaled units (`1 << 9` per bit), and `dist` the distortion.
/// Bit-exact integer form:
/// `ROUND_POWER_OF_TWO((i64)rate * rm, 9) + (dist << RDDIV_BITS)`.
#[inline]
pub fn rdcost(rm: i32, rate: i32, dist: i64) -> i64 {
    round_power_of_two_i64((rate as i64) * (rm as i64), AV1_PROB_COST_SHIFT)
        + (dist * (1 << RDDIV_BITS))
}

/// `RDCOST_NEG_R(RM, R, D)` (av1/encoder/rd.h) — the RD cost when the rate term
/// is subtracted (used where a candidate *saves* rate).
#[inline]
pub fn rdcost_neg_r(rm: i32, rate: i32, dist: i64) -> i64 {
    (dist * (1 << RDDIV_BITS))
        - round_power_of_two_i64((rate as i64) * (rm as i64), AV1_PROB_COST_SHIFT)
}

// ---------------------------------------------------------------------------
// RD multiplier / lambda from qindex (av1/encoder/rd.c)
// ---------------------------------------------------------------------------

/// `FRAME_UPDATE_TYPE` (av1/encoder/ratectrl.h) — a frame's role in the GOP.
/// The RD multiplier only distinguishes `Kf` vs `Gf`/`Arf` vs everything else,
/// but the full enum is modelled for fidelity. Discriminants match C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameUpdateType {
    /// `KF_UPDATE` — key frame.
    Kf = 0,
    /// `LF_UPDATE` — leaf (normal inter) frame.
    Lf = 1,
    /// `GF_UPDATE` — golden frame.
    Gf = 2,
    /// `ARF_UPDATE` — alt-ref frame.
    Arf = 3,
    /// `OVERLAY_UPDATE`.
    Overlay = 4,
    /// `INTNL_OVERLAY_UPDATE`.
    IntnlOverlay = 5,
    /// `INTNL_ARF_UPDATE`.
    IntnlArf = 6,
}

/// The `aom_tune_metric` values (av1/aomcx.h) that change the RD multiplier.
/// Every other tuning takes the same path as [`TuneMetric::Psnr`] (the default).
/// Discriminants match the C enum values used by `av1_compute_rd_mult_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuneMetric {
    /// `AOM_TUNE_PSNR` (and any tuning that is neither `Iq` nor `Ssimulacra2`).
    Psnr = 0,
    /// `AOM_TUNE_IQ`.
    Iq = 10,
    /// `AOM_TUNE_SSIMULACRA2`.
    Ssimulacra2 = 11,
}

/// `MODE` (av1/encoder/enc_enums.h) — the encode mode. Only `Realtime` changes
/// the RD-multiplier tuning weight. Discriminants match C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncMode {
    /// `GOOD` — good-quality (non-realtime) mode.
    Good = 0,
    /// `REALTIME`.
    Realtime = 1,
    /// `ALLINTRA`.
    Allintra = 2,
}

/// `FRAME_TYPE` (av1/common/enums.h) as far as [`av1_compute_rd_mult`] reads it:
/// it only branches on `!= KEY_FRAME`. Discriminants match C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    /// `KEY_FRAME`.
    Key = 0,
    /// Any non-key frame (`INTER_FRAME` / `INTRA_ONLY_FRAME` / `S_FRAME`).
    NonKey = 1,
}

/// `rd_boost_factor[16]` (av1/encoder/rd.c).
const RD_BOOST_FACTOR: [i64; 16] = [64, 32, 32, 32, 24, 16, 12, 12, 8, 8, 4, 4, 2, 2, 1, 0];
/// `rd_layer_depth_factor[7]` (av1/encoder/rd.c).
const RD_LAYER_DEPTH_FACTOR: [i64; 7] = [160, 160, 160, 160, 192, 208, 224];

/// `def_kf_rd_multiplier` (av1/encoder/rd.c). `q` is the DC quantizer step.
#[inline]
fn def_kf_rd_multiplier(q: i32) -> f64 {
    3.3 + 0.0015 * q as f64
}
/// `def_arf_rd_multiplier` (av1/encoder/rd.c).
#[inline]
fn def_arf_rd_multiplier(q: i32) -> f64 {
    3.25 + 0.0015 * q as f64
}
/// `def_inter_rd_multiplier` (av1/encoder/rd.c).
#[inline]
fn def_inter_rd_multiplier(q: i32) -> f64 {
    3.2 + 0.0015 * q as f64
}

/// `av1_compute_rd_mult_based_on_qindex` (av1/encoder/rd.c) — the base RD
/// multiplier (lambda) for a given qindex, bit-exact vs C.
///
/// `bit_depth` is 8/10/12. Uses IEEE-754 `f64` in the same op order as C; the
/// reference build has no FMA (no `-mfma`), so the separate multiply/add match
/// Rust's strict-FP `*`/`+`. Returns a value in `[1, i32::MAX]`.
pub fn av1_compute_rd_mult_based_on_qindex(
    bit_depth: u8,
    update_type: FrameUpdateType,
    qindex: i32,
    tuning: TuneMetric,
    mode: EncMode,
) -> i32 {
    let q = av1_dc_quant_qtx(qindex, 0, bit_depth) as i32;
    // C: `int64_t rdmult = q * q;` (fits in `int` for every valid qindex).
    let mut rdmult: i64 = (q as i64) * (q as i64);
    let def_rd_q_mult = match update_type {
        FrameUpdateType::Kf => def_kf_rd_multiplier(q),
        FrameUpdateType::Gf | FrameUpdateType::Arf => def_arf_rd_multiplier(q),
        _ => def_inter_rd_multiplier(q),
    };
    rdmult = (rdmult as f64 * def_rd_q_mult) as i64;

    if matches!(tuning, TuneMetric::Iq | TuneMetric::Ssimulacra2) {
        let weight: i32 = if mode == EncMode::Realtime {
            32
        } else {
            (((255 - qindex) * 3) / 4).clamp(0, 72) + 128
        };
        rdmult = (rdmult as f64 * weight as f64 / 128.0) as i64;
    }

    match bit_depth {
        8 => {}
        10 => rdmult = round_power_of_two_i64(rdmult, 4),
        12 => rdmult = round_power_of_two_i64(rdmult, 8),
        _ => return -1,
    }
    if rdmult > 0 {
        rdmult.min(i32::MAX as i64) as i32
    } else {
        1
    }
}

/// `av1_compute_rd_mult` (av1/encoder/rd.c) — [`av1_compute_rd_mult_based_on_qindex`]
/// plus the two-pass layer-depth / ARF-boost adjustment, bit-exact vs C.
///
/// The adjustment only fires when `is_stat_consumption_stage`,
/// `!use_fixed_qp_offsets`, and `frame_type != Key`. `layer_depth` indexes
/// `rd_layer_depth_factor[0..7]`; `boost_index` indexes `rd_boost_factor[0..16]`.
#[allow(clippy::too_many_arguments)]
pub fn av1_compute_rd_mult(
    qindex: i32,
    bit_depth: u8,
    update_type: FrameUpdateType,
    layer_depth: i32,
    boost_index: i32,
    frame_type: FrameType,
    use_fixed_qp_offsets: bool,
    is_stat_consumption_stage: bool,
    tuning: TuneMetric,
    mode: EncMode,
) -> i32 {
    let mut rdmult =
        av1_compute_rd_mult_based_on_qindex(bit_depth, update_type, qindex, tuning, mode) as i64;
    if is_stat_consumption_stage && !use_fixed_qp_offsets && frame_type != FrameType::Key {
        rdmult = (rdmult * RD_LAYER_DEPTH_FACTOR[layer_depth as usize]) >> 7;
        rdmult += (rdmult * RD_BOOST_FACTOR[boost_index as usize]) >> 7;
    }
    if rdmult > 0 {
        rdmult.min(i32::MAX as i64) as i32
    } else {
        1
    }
}

// ---------------------------------------------------------------------------
// Per-bit search multipliers + plane-quantizer setup
// (av1/encoder/rd.{c,h} + av1/encoder/av1_quantize.c av1_init_plane_quantizers)
// ---------------------------------------------------------------------------

/// `RD_EPB_SHIFT` (av1/encoder/rd.h).
const RD_EPB_SHIFT: i32 = 6;

/// `av1_set_error_per_bit` (rd.h): the mv-cost → l2-error multiplier,
/// `max(rdmult >> RD_EPB_SHIFT, 1)`.
#[inline]
pub fn av1_set_error_per_bit(rdmult: i32) -> i32 {
    (rdmult >> RD_EPB_SHIFT).max(1)
}

/// `av1_convert_qindex_to_q` (av1/encoder/ratectrl.c): the AC quantizer step
/// scaled down to the classic Q range (`/4`, `/16`, `/64` per bit depth).
#[inline]
pub fn av1_convert_qindex_to_q(qindex: i32, bit_depth: u8) -> f64 {
    match bit_depth {
        8 => f64::from(aom_dsp::quant::av1_ac_quant_qtx(qindex, 0, 8)) / 4.0,
        10 => f64::from(aom_dsp::quant::av1_ac_quant_qtx(qindex, 0, 10)) / 16.0,
        12 => f64::from(aom_dsp::quant::av1_ac_quant_qtx(qindex, 0, 12)) / 64.0,
        _ => -1.0,
    }
}

/// `av1_set_sad_per_bit` (rd.c): the SAD-per-bit multiplier for motion search.
/// The C keeps per-bit-depth luts (`init_me_luts_bd`); each entry is
/// `(int)(0.0418 * q + 2.4107)` of [`av1_convert_qindex_to_q`], computed here
/// directly (bit-exact — pure f64 multiply/add, no FMA in the reference build).
#[inline]
pub fn av1_set_sad_per_bit(qindex: i32, bit_depth: u8) -> i32 {
    let q = av1_convert_qindex_to_q(qindex, bit_depth);
    (0.0418 * q + 2.4107) as i32
}

/// Output of [`init_plane_quantizers`]: the per-superblock quantizer state
/// `av1_init_plane_quantizers` derives (the scalar fields of `MACROBLOCK`;
/// the per-plane rows come from [`aom_dsp::quant::set_q_index`] at the same
/// `qindex`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneQuantSetup {
    /// `x->qindex` — the effective qindex (base + delta-q, segment-adjusted).
    pub qindex: i32,
    /// The block RD multiplier from `av1_compute_rd_mult(qindex + y_dc_delta_q, ...)`.
    pub rdmult: i32,
    /// `x->errorperbit`.
    pub errorperbit: i32,
    /// `x->sadperbit`.
    pub sadperbit: i32,
    /// `x->seg_skip_block` — the segment's `SEG_LVL_SKIP` feature.
    pub seg_skip_block: bool,
}

/// The qindex-derivation slice of `av1_init_plane_quantizers`
/// (av1/encoder/av1_quantize.c): `clamp(base_qindex [+ delta_qindex], 0, 255)`
/// then [`aom_dsp::quant::av1_get_qindex`] for the segment, then the RD multiplier,
/// error-per-bit and sad-per-bit from that qindex.
///
/// Scope (labelled): `sb_qp_sweep` is modelled OFF (`qindex_rd == qindex`, the
/// default outside the debug sweep), and `set_qmatrix` (the global `gqmatrix`
/// table install, QM-only) is not modelled — the QM path takes caller-supplied
/// matrices via [`crate::QuantParams`].
#[allow(clippy::too_many_arguments)]
pub fn init_plane_quantizers(
    seg: &aom_dsp::quant::Segmentation,
    segment_id: usize,
    base_qindex: i32,
    delta_qindex: i32,
    delta_q_present: bool,
    y_dc_delta_q: i32,
    bit_depth: u8,
    update_type: FrameUpdateType,
    layer_depth: i32,
    boost_index: i32,
    frame_type: FrameType,
    use_fixed_qp_offsets: bool,
    is_stat_consumption_stage: bool,
    tuning: TuneMetric,
    mode: EncMode,
) -> PlaneQuantSetup {
    let current_qindex = if delta_q_present {
        base_qindex + delta_qindex
    } else {
        base_qindex
    }
    .clamp(0, 255);
    let qindex = aom_dsp::quant::av1_get_qindex(seg, segment_id, current_qindex);
    // sb_qp_sweep off => qindex_rd == qindex.
    let qindex_rdmult = qindex + y_dc_delta_q;
    let rdmult = av1_compute_rd_mult(
        qindex_rdmult,
        bit_depth,
        update_type,
        layer_depth,
        boost_index,
        frame_type,
        use_fixed_qp_offsets,
        is_stat_consumption_stage,
        tuning,
        mode,
    );
    PlaneQuantSetup {
        qindex,
        rdmult,
        errorperbit: av1_set_error_per_bit(rdmult),
        sadperbit: av1_set_sad_per_bit(qindex, bit_depth),
        seg_skip_block: seg.feature_active(segment_id, aom_dsp::quant::SEG_LVL_SKIP),
    }
}
