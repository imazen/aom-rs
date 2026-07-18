//! Variance Boost delta-q (`--deltaq-mode=6`, `DELTA_Q_VARIANCE_BOOST` — the
//! tune=IQ/SSIMULACRA2 default): the per-superblock qindex derivation from
//! source variance. Ports (libaom v3.14.1):
//!
//! - `av1_get_variance_boost_block_variance` (av1/encoder/aq_variance.c:184):
//!   the 64 8x8-subblock variances-vs-zero of a 64x64 SB, sorted, sampled at
//!   octile 5 with 1:2:1 neighbour weighting.
//! - `av1_get_sbq_variance_boost` (av1/encoder/allintra_vis.c:1072): the
//!   still-picture boost curve mapping (variance, base qindex, strength) to
//!   the SB qindex.
//! - `aom_get_variance_boost_delta_q_res` (av1/encoder/encodeframe.c:1920):
//!   the base-qindex-dependent `delta_q_res` (1/2/4/8).
//! - `av1_adjust_q_from_delta_q_res` (av1/encoder/rd.c:494): deadzone-rounded
//!   quantization of the per-SB qindex onto the `delta_q_res` grid against
//!   the running `current_base_qindex`.
//! - `av1_convert_qindex_to_q` / `av1_convert_q_to_qindex`
//!   (av1/encoder/ratectrl.c:199/:211).
//!
//! Floating point note: `av1_get_sbq_variance_boost` uses `f64` `log2` /
//! `round` in C's exact operation order. `log2` resolves to the platform
//! libm in both builds here (the same glibc), so the byte gates hold locally;
//! `round` (half away from zero) == Rust `f64::round`.

use aom_quant::{av1_ac_quant_qtx, av1_dc_quant_qtx};

/// `MAXQ` / `MINQ` (av1/common/quant_common.h).
const MAXQ: i32 = 255;
const MINQ: i32 = 0;

/// `av1_get_deltaq_offset` (rd.c:466): the qindex delta whose DC quantizer
/// step is closest to `q(base) / sqrt(beta)`. `beta > 1` lowers the qindex
/// (finer quant), `beta < 1` raises it. Walks the DC-quant table (
/// [`av1_dc_quant_qtx`], exhaustively bit-exact vs C) one qindex at a time
/// from `qindex` until the stepped-down/up quant crosses `newq`. Shared by
/// both the Perceptual-AI arm ([`av1_get_sbq_perceptual_ai`]) and the
/// rate-guided arm.
pub fn av1_get_deltaq_offset(bit_depth: u8, qindex: i32, beta: f64) -> i32 {
    debug_assert!(beta > 0.0);
    let mut q = i32::from(av1_dc_quant_qtx(qindex, 0, bit_depth));
    // `(int)rint(q / sqrt(beta))`: rint = round to nearest, ties to even, in
    // the default rounding mode; the double is integer-valued so the int cast
    // is exact.
    let newq = (f64::from(q) / beta.sqrt()).round_ties_even() as i32;
    let orig_qindex = qindex;
    let mut qindex = qindex;
    if newq == q {
        return 0;
    }
    if newq < q {
        while qindex > 0 {
            qindex -= 1;
            q = i32::from(av1_dc_quant_qtx(qindex, 0, bit_depth));
            if newq >= q {
                break;
            }
        }
    } else {
        while qindex < MAXQ {
            qindex += 1;
            q = i32::from(av1_dc_quant_qtx(qindex, 0, bit_depth));
            if newq <= q {
                break;
            }
        }
    }
    qindex - orig_qindex
}

/// `VAR_BOOST_MAX_DELTAQ_RANGE` (allintra_vis.c:39).
const VAR_BOOST_MAX_DELTAQ_RANGE: i32 = 80;
/// `VAR_BOOST_MAX_BOOST` (allintra_vis.c:41).
const VAR_BOOST_MAX_BOOST: f64 = 8.0;

/// One 8x8 variance against an all-zero reference — `fn_ptr[BLOCK_8X8].vf`
/// with `ref = av1_all_zeros` exactly as `av1_get_variance_boost_block_
/// variance` calls it: `aom_variance8x8` at bd8, `aom_highbd_{8,10,12}_
/// variance8x8` at high bit depth (aom_dsp/variance.c — the bd-dependent
/// `ROUND_POWER_OF_TWO` normalization of sse/sum before the `sse -
/// sum*sum/64` variance).
fn variance8x8_vs_zero(src: &[u16], off: usize, stride: usize, bd: u8) -> u32 {
    let mut sse: u64 = 0;
    let mut sum: i64 = 0;
    for r in 0..8 {
        let row = &src[off + r * stride..off + r * stride + 8];
        for &px in row {
            let d = i64::from(px);
            sum += d;
            sse += (d * d) as u64;
        }
    }
    // highbd_{8,10,12}_variance narrow sse/sum per bit depth
    // (variance.c:298-325); bd8's `variance()` accumulates in u32/int but the
    // 8x8 all-positive sums cannot exceed them (max sse 64*255^2 < 2^22).
    // ROUND_POWER_OF_TWO on the vs-zero sums (both non-negative here).
    let rpot = |v: u64, n: u32| -> u64 { (v + ((1u64 << n) >> 1)) >> n };
    let (sse32, sum32): (u32, i32) = match bd {
        8 => (sse as u32, sum as i32),
        10 => (rpot(sse, 4) as u32, rpot(sum as u64, 2) as i32),
        12 => (rpot(sse, 8) as u32, rpot(sum as u64, 4) as i32),
        _ => unreachable!("bd must be 8/10/12"),
    };
    // VAR/HIGHBD_VAR: `*sse - (uint32_t)(((int64_t)sum * sum) / (W * H))`.
    sse32.wrapping_sub(((i64::from(sum32) * i64::from(sum32)) / 64) as u32)
}

/// `av1_get_variance_boost_block_variance` (aq_variance.c:184): the 64
/// 8x8-subblock variances (each `vf(...) / 64`, truncating) of the 64x64 SB
/// at `off`, sorted ascending, sampled at octile 5 (indices 31/39/47) with
/// 1:2:1 weighting and +2 rounding. `src` must cover the full 64x64 extent
/// (frame-edge SBs read the replicate-extended border, exactly as C's
/// `av1_setup_src_planes` sources do).
pub fn variance_boost_block_variance(src: &[u16], off: usize, stride: usize, bd: u8) -> u32 {
    const SUBBLOCKS_IN_SB_DIM: usize = 8;
    const SUBBLOCKS_IN_SB: usize = 64;
    const SUBBLOCKS_IN_OCTILE: usize = SUBBLOCKS_IN_SB / 8;
    const OCTILE: usize = 5;
    let mut variances = [0u32; SUBBLOCKS_IN_SB];
    for sb_i in 0..SUBBLOCKS_IN_SB_DIM {
        for sb_j in 0..SUBBLOCKS_IN_SB_DIM {
            variances[sb_i * SUBBLOCKS_IN_SB_DIM + sb_j] =
                variance8x8_vs_zero(src, off + (sb_i * 8) * stride + sb_j * 8, stride, bd) / 64;
        }
    }
    variances.sort_unstable(); // qsort by value — ties interchangeable
    let middle_index = OCTILE * SUBBLOCKS_IN_OCTILE - 1;
    let lower_index = (SUBBLOCKS_IN_OCTILE - 1).max(middle_index - SUBBLOCKS_IN_OCTILE);
    let upper_index = (SUBBLOCKS_IN_SB - 1).min(middle_index + SUBBLOCKS_IN_OCTILE);
    (variances[lower_index] + variances[middle_index] * 2 + variances[upper_index] + 2) / 4
}

/// `av1_convert_qindex_to_q` (ratectrl.c:199).
pub fn av1_convert_qindex_to_q(qindex: i32, bit_depth: u8) -> f64 {
    match bit_depth {
        8 => f64::from(av1_ac_quant_qtx(qindex, 0, 8)) / 4.0,
        10 => f64::from(av1_ac_quant_qtx(qindex, 0, 10)) / 16.0,
        12 => f64::from(av1_ac_quant_qtx(qindex, 0, 12)) / 64.0,
        _ => unreachable!("bd must be 8/10/12"),
    }
}

/// `av1_convert_q_to_qindex` (ratectrl.c:211): first qindex whose q matches
/// or exceeds `q`.
pub fn av1_convert_q_to_qindex(q: f64, bit_depth: u8) -> i32 {
    let mut qindex = MINQ;
    while qindex < MAXQ && av1_convert_qindex_to_q(qindex, bit_depth) < q {
        qindex += 1;
    }
    qindex
}

/// `av1_get_sbq_variance_boost` (allintra_vis.c:1072) with the SB variance
/// already computed ([`variance_boost_block_variance`]): the Variance Boost
/// still-picture curve. `deltaq_strength` is the `--deltaq-strength` percent
/// (default 100). Returns the SB qindex (>= MINQ + 1 — always lossy).
pub fn av1_get_sbq_variance_boost(
    base_qindex: i32,
    bit_depth: u8,
    deltaq_strength: u32,
    mut variance: u32,
) -> i32 {
    // strength = clamp((deltaq_strength / 100) * 3, 0, 6)
    let strength = ((f64::from(deltaq_strength) / 100.0) * 3.0).clamp(0.0, 6.0);
    if variance == 0 {
        variance = 1;
    }
    // qstep_ratio = clamp(0.15 * strength * (-log2(variance) + 10) + 1, 1, 8)
    let qstep_ratio = (0.15 * strength * (-f64::from(variance).log2() + 10.0) + 1.0)
        .clamp(1.0, VAR_BOOST_MAX_BOOST);
    let base_q = av1_convert_qindex_to_q(base_qindex, bit_depth);
    let target_q = base_q / qstep_ratio;
    let target_qindex = av1_convert_q_to_qindex(target_q, bit_depth);
    // boost = round((base_qindex + 544) * (base_qindex - target_qindex) / 1279)
    let boost = ((f64::from(base_qindex) + 544.0) * f64::from(base_qindex - target_qindex)
        / 1279.0)
        .round() as i32;
    let boost = boost.min(VAR_BOOST_MAX_DELTAQ_RANGE);
    (base_qindex - boost).max(MINQ + 1)
}

/// `aom_get_variance_boost_delta_q_res` (encodeframe.c:1920): finer delta-q
/// grids at low base qindex, coarser at high (signaling-overhead balance).
pub fn variance_boost_delta_q_res(qindex: i32) -> i32 {
    if qindex >= 160 {
        8
    } else if qindex >= 120 {
        4
    } else if qindex >= 80 {
        2
    } else {
        1
    }
}

/// `av1_adjust_q_from_delta_q_res` (rd.c:494): quantize `curr_qindex` onto
/// the `delta_q_res` grid relative to the running `prev_qindex`, with a
/// `res/4` deadzone, clamped to `[res, 256 - res]` first and `>= MINQ + 1`
/// after.
pub fn av1_adjust_q_from_delta_q_res(delta_q_res: i32, prev_qindex: i32, curr_qindex: i32) -> i32 {
    let curr = curr_qindex.clamp(delta_q_res, 256 - delta_q_res);
    let sign = if curr - prev_qindex >= 0 { 1 } else { -1 };
    let deadzone = delta_q_res / 4;
    let qmask = !(delta_q_res - 1);
    let abs_dq = ((curr - prev_qindex).abs() + deadzone) & qmask;
    (prev_qindex + sign * abs_dq).max(MINQ + 1)
}

/// The per-SB qindex of `setup_delta_q` (encodeframe.c:341-370) under
/// `DELTA_Q_VARIANCE_BOOST`: boost from the SB's source variance, then
/// deadzone-quantize against the RUNNING `current_base_qindex` (updated by
/// the caller per C's `av1_update_state` gate: SB-root
/// `bsize != sb_size || !skip`).
#[allow(clippy::too_many_arguments)]
pub fn setup_delta_q_variance_boost(
    src: &[u16],
    sb_off: usize,
    stride: usize,
    bd: u8,
    base_qindex: i32,
    deltaq_strength: u32,
    delta_q_res: i32,
    current_base_qindex: i32,
) -> i32 {
    let variance = variance_boost_block_variance(src, sb_off, stride, bd);
    let boosted = av1_get_sbq_variance_boost(base_qindex, bd, deltaq_strength, variance);
    av1_adjust_q_from_delta_q_res(delta_q_res, current_base_qindex, boosted)
}

/// The per-SB qindex of `setup_delta_q` (encodeframe.c:341-370) under
/// `DELTA_Q_PERCEPTUAL_AI` (mode 3): the wiener-variance-map qindex
/// ([`WeberVarMap::av1_get_sbq_perceptual_ai`], keyed on the FRAME
/// `base_qindex`), then the SAME deadzone-quantize against the RUNNING
/// `current_base_qindex` mode 6 uses. `sb_mi` is `mi_size_wide[sb_size]`.
#[allow(clippy::too_many_arguments)]
pub fn setup_delta_q_perceptual_ai(
    map: &WeberVarMap,
    base_qindex: i32,
    bit_depth: u8,
    delta_q_res: i32,
    sb_mi: i32,
    mi_row: i32,
    mi_col: i32,
    current_base_qindex: i32,
) -> i32 {
    let current =
        map.av1_get_sbq_perceptual_ai(base_qindex, bit_depth, delta_q_res, sb_mi, sb_mi, mi_row, mi_col);
    av1_adjust_q_from_delta_q_res(delta_q_res, current_base_qindex, current)
}

// ===========================================================================
// `--deltaq-mode=2` (DELTA_Q_PERCEPTUAL, wavelet AC energy — the arm selected
// by `DELTA_Q_PERCEPTUAL_MODULATION == 1`, encodeframe.h:25). Ports (libaom
// v3.14.1):
//   - dwt.c: the 5/3 dyadic wavelet (`av1_fdwt8x8_uint8_input`, a pure-C RTCD
//     entry — `#define av1_fdwt8x8_uint8_input av1_fdwt8x8_uint8_input_c`, no
//     SIMD) + `haar_ac_sad` + `av1_haar_ac_sad_mxn_uint8_input`.
//   - aq_variance.c: `haar_ac_energy` (:124) / `log_block_wavelet_energy`
//     (:138) / `av1_block_wavelet_energy_level` (:143) /
//     `av1_compute_q_from_energy_level_deltaq_mode` (:153).
//   - ratectrl.c: `av1_rc_bits_per_mb` (:271, the KEY-frame / AOM_Q allintra
//     arm: neither CBR branch fires) / `find_qindex_by_rate` (:1420) /
//     `av1_compute_qdelta_by_rate` (:2676).
//   - `DEFAULT_DELTA_Q_RES_PERCEPTUAL` (= [`DELTA_Q_RES_PERCEPTUAL`] = 4) is
//     the mode-2 delta-q grid (encodeframe.c:2287-2288), same as mode 3.
//
// Single-frame (lone still): `energy_midpoint = DEFAULT_E_MIDPOINT` (10.0) —
// `is_stat_consumption_stage_twopass` is false, so no two-pass frame-average.
// **Scope (this landing): bd8** (the dwt reads the u8 source directly);
// frame dims a multiple of the 64px SB (every SB reads a full in-frame 64x64,
// like the mode-3 initial scope). All `log1p`/`round` resolve to the same
// glibc libm as the C build (the Variance-Boost `log2` envelope note applies).
// ===========================================================================

/// `tran_low_t` (av1/common/enums.h) — the wavelet coefficient type.
type TranLow = i32;

/// `analysis_53_row` (dwt.c:20): one row 5/3 lifting pass. `x` is the copied
/// input row (C memcpy's it before the call, so reads never see the in-place
/// output writes); `out[0..hw]` receives the lowpass, `out[hw..]` the highpass
/// (C's `lowpass = &c[row]`, `highpass = &c[row] + hw`, non-overlapping).
fn analysis_53_row(length: usize, x: &[TranLow], out: &mut [TranLow], hw: usize) {
    let half = length >> 1;
    let mut xi = 0usize;
    for k in 0..half - 1 {
        out[k] = x[xi] * 2;
        let r = x[xi];
        xi += 1;
        out[hw + k] = x[xi] - ((r + x[xi + 1] + 1) >> 1);
        xi += 1;
    }
    let last = half - 1;
    out[last] = x[xi] * 2;
    let r_tail = x[xi];
    xi += 1;
    out[hw + last] = x[xi] - r_tail;
    // Update pass: `*a += (r + *b + 1) >> 1`, `r` seeded from `*highpass`.
    let mut r = out[hw];
    for k in 0..half {
        out[k] += (r + out[hw + k] + 1) >> 1;
        r = out[hw + k];
    }
}

/// `analysis_53_col` (dwt.c:46): the column 5/3 lifting pass (different lowpass
/// scaling + highpass rounding than the row pass). `x` is the copied input
/// column; `out[0..hh]` lowpass, `out[hh..]` highpass.
fn analysis_53_col(length: usize, x: &[TranLow], out: &mut [TranLow], hh: usize) {
    let half = length >> 1;
    let mut xi = 0usize;
    for k in 0..half - 1 {
        out[k] = x[xi];
        let r = x[xi];
        xi += 1;
        out[hh + k] = ((x[xi] * 2) - (r + x[xi + 1]) + 2) >> 2;
        xi += 1;
    }
    let last = half - 1;
    out[last] = x[xi];
    let r_tail = x[xi];
    xi += 1;
    out[hh + last] = (x[xi] - r_tail + 1) >> 1;
    let mut r = out[hh];
    for k in 0..half {
        out[k] += (r + out[hh + k] + 1) >> 1;
        r = out[hh + k];
    }
}

/// `av1_fdwt8x8_uint8_input_c` (dwt.c:112) = `dyadic_analyze_53_uint8_input(4,
/// 8, 8, input, stride, output, pitch_c=8, dwt_scale_bits=2, hbd)`. Loads the
/// 8x8 source `<< 2`, then runs 4 levels of the separable 5/3 analysis (the
/// last level bails at `nh < 2`). `output` is the 8x8 coefficient block, row
/// stride 8. bd8 reads the u8 source; the hbd arm reads the u16 (unused in
/// this landing but written to match the C dispatch on `hbd`).
fn av1_fdwt8x8_uint8_input(src: &[u16], off: usize, stride: usize, bd: u8, output: &mut [TranLow; 64]) {
    const SCALE: i32 = 2;
    let hbd = bd > 8;
    for i in 0..8 {
        for j in 0..8 {
            let px = src[off + i * stride + j];
            let v = if hbd { i32::from(px) } else { i32::from(px as u8) };
            output[i * 8 + j] = v << SCALE;
        }
    }
    let mut hh = 8usize;
    let mut hw = 8usize;
    let mut line = [0i32; 8];
    let mut col_out = [0i32; 8];
    for _lv in 0..4 {
        let nh = hh;
        hh = (hh + 1) >> 1;
        let nw = hw;
        hw = (hw + 1) >> 1;
        if nh < 2 || nw < 2 {
            return;
        }
        for i in 0..nh {
            line[..nw].copy_from_slice(&output[i * 8..i * 8 + nw]);
            analysis_53_row(nw, &line[0..nw], &mut output[i * 8..i * 8 + nw], hw);
        }
        for j in 0..nw {
            for i in 0..nh {
                line[i] = output[i * 8 + j];
            }
            analysis_53_col(nh, &line[0..nh], &mut col_out[0..nh], hh);
            for i in 0..nh {
                output[i * 8 + j] = col_out[i];
            }
        }
    }
}

/// `haar_ac_sad` (dwt.c:117): the sum of `|coeff|` over the three AC quadrants
/// of the 8x8 wavelet block (every position except the top-left 4x4 LL band).
fn haar_ac_sad_8x8(output: &[TranLow; 64]) -> i32 {
    let mut acsad = 0i32;
    for r in 0..8usize {
        for c in 0..8usize {
            if r >= 4 || c >= 4 {
                acsad += output[r * 8 + c].abs();
            }
        }
    }
    acsad
}

/// `av1_haar_ac_sad_mxn_uint8_input` (dwt.c:135): the total AC wavelet energy
/// of a `num_8x8_rows`×`num_8x8_cols` grid of 8x8 blocks starting at `off`.
fn haar_ac_sad_mxn(
    src: &[u16],
    off: usize,
    stride: usize,
    bd: u8,
    num_8x8_rows: usize,
    num_8x8_cols: usize,
) -> i64 {
    let mut energy = 0i64;
    let mut out = [0i32; 64];
    for r8 in 0..num_8x8_rows {
        for c8 in 0..num_8x8_cols {
            let blk_off = off + c8 * 8 + r8 * 8 * stride;
            av1_fdwt8x8_uint8_input(src, blk_off, stride, bd, &mut out);
            energy += i64::from(haar_ac_sad_8x8(&out));
        }
    }
    energy
}

/// Differential entry point for [`haar_ac_sad_mxn`] (the dwt + AC-SAD chain),
/// matching the exported `av1_haar_ac_sad_mxn_uint8_input` signature. `src`
/// holds u8 samples (bd8) in a u16 buffer with the given `stride`.
#[doc(hidden)]
pub fn haar_ac_sad_mxn_for_test(
    src: &[u16],
    off: usize,
    stride: usize,
    bd: u8,
    num_8x8_rows: usize,
    num_8x8_cols: usize,
) -> i64 {
    haar_ac_sad_mxn(src, off, stride, bd, num_8x8_rows, num_8x8_cols)
}

/// `haar_ac_energy` (aq_variance.c:124): the SB wavelet AC energy normalized by
/// pixel count. `sb_w`/`sb_h` are the SB pixel dims (`block_size_wide/high[bs]`)
/// and `num_pels_log2` is `num_pels_log2_lookup[bs]` (= `log2(sb_w*sb_h)`).
/// **Exact cast order:** C is `(unsigned int)((uint64_t)var * 256) >> npl` —
/// the truncating u32 cast binds tighter than `>>`, so the product is narrowed
/// to 32 bits *before* the shift.
fn haar_ac_energy(src: &[u16], off: usize, stride: usize, bd: u8, sb_w: usize, sb_h: usize, num_pels_log2: u32) -> u32 {
    let var = haar_ac_sad_mxn(src, off, stride, bd, sb_h / 8, sb_w / 8);
    (((var as u64).wrapping_mul(256)) as u32) >> num_pels_log2
}

/// `DEFAULT_E_MIDPOINT` (aq_variance.c:122) — the single-frame wavelet-energy
/// midpoint (two-pass uses the frame average instead).
const DEFAULT_E_MIDPOINT: f64 = 10.0;
/// `ENERGY_MIN` / `ENERGY_MAX` (aq_variance.c:33-34).
const ENERGY_MIN: i32 = -4;
const ENERGY_MAX: i32 = 1;
/// `segment_id[ENERGY_SPAN]` (aq_variance.c:39), indexed by `energy - ENERGY_MIN`.
const SEGMENT_ID: [usize; 6] = [0, 1, 1, 2, 3, 4];
/// `deltaq_rate_ratio[MAX_SEGMENTS]` (aq_variance.c:31).
const DELTAQ_RATE_RATIO: [f64; 8] = [2.5, 2.0, 1.5, 1.0, 0.75, 1.0, 1.0, 1.0];

/// `av1_block_wavelet_energy_level` (aq_variance.c:143): the clamped rounded
/// log-wavelet-energy of the SB relative to the (single-frame) midpoint.
fn block_wavelet_energy_level(src: &[u16], off: usize, stride: usize, bd: u8, sb_w: usize, sb_h: usize, num_pels_log2: u32) -> i32 {
    let haar_sad = haar_ac_energy(src, off, stride, bd, sb_w, sb_h, num_pels_log2);
    // log_block_wavelet_energy = log1p((double)haar_sad).
    let energy = f64::from(haar_sad).ln_1p() - DEFAULT_E_MIDPOINT;
    // clamp((int)round(energy), ENERGY_MIN, ENERGY_MAX). C round() = ties away
    // from zero == Rust f64::round.
    (energy.round() as i32).clamp(ENERGY_MIN, ENERGY_MAX)
}

/// `av1_rc_bits_per_mb` (ratectrl.c:271) for the **KEY-frame / AOM_Q allintra**
/// path with `correction_factor = 1.0`: neither the CBR-inter nor the
/// CBR-keyframe branch fires (`rc_cfg.mode != AOM_CBR`), so it reduces to
/// `(int)(enumerator / q)`. `enumerator = get_bpmb_enumerator(KEY, screen)`
/// (ratectrl.c:255): 2_000_000 (non-screen) / 1_000_000 (screen).
fn rc_bits_per_mb_key(qindex: i32, bit_depth: u8, is_screen_content: bool) -> i32 {
    let q = av1_convert_qindex_to_q(qindex, bit_depth);
    let enumerator: i32 = if is_screen_content { 1_000_000 } else { 2_000_000 };
    (f64::from(enumerator) * 1.0 / q) as i32
}

/// `find_qindex_by_rate` (ratectrl.c:1420): the smallest qindex in
/// `[best, worst]` whose modeled bits-per-mb is `<= desired`. Binary search
/// (`> desired` raises the floor).
fn find_qindex_by_rate(desired_bits_per_mb: i32, bit_depth: u8, is_screen_content: bool, best: i32, worst: i32) -> i32 {
    let mut low = best;
    let mut high = worst;
    while low < high {
        let mid = (low + high) >> 1;
        let mid_bits = rc_bits_per_mb_key(mid, bit_depth, is_screen_content);
        if mid_bits > desired_bits_per_mb {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

/// `av1_compute_qdelta_by_rate` (ratectrl.c:2676): the qindex delta to hit
/// `rate_target_ratio` × the base bits-per-mb, searching `[best, worst]`.
/// `best`/`worst` are `rc->best_quality`/`rc->worst_quality`
/// (`av1_quantizer_to_qindex(rc_min/max_quantizer)` — 0/255 for the allintra
/// default `--min-q=0 --max-q=63`).
fn compute_qdelta_by_rate(qindex: i32, bit_depth: u8, is_screen_content: bool, rate_target_ratio: f64, best: i32, worst: i32) -> i32 {
    let base_bits_per_mb = rc_bits_per_mb_key(qindex, bit_depth, is_screen_content);
    let target_bits_per_mb = (rate_target_ratio * f64::from(base_bits_per_mb)) as i32;
    let target_index = find_qindex_by_rate(target_bits_per_mb, bit_depth, is_screen_content, best, worst);
    target_index - qindex
}

/// `av1_compute_q_from_energy_level_deltaq_mode` (aq_variance.c:153) under
/// `DELTA_Q_PERCEPTUAL_MODULATION == 1`: map the clamped wavelet energy level
/// to a rate segment, take the rate-ratio qindex delta, apply the lossless
/// guard, and add to the base. `best`/`worst` = `rc->best/worst_quality`.
#[allow(clippy::too_many_arguments)]
fn compute_q_from_energy_level(base_qindex: i32, bit_depth: u8, is_screen_content: bool, block_var_level: i32, best: i32, worst: i32) -> i32 {
    debug_assert!((ENERGY_MIN..=ENERGY_MAX).contains(&block_var_level));
    let rate_level = SEGMENT_ID[(block_var_level - ENERGY_MIN) as usize];
    let mut qindex_delta = compute_qdelta_by_rate(
        base_qindex,
        bit_depth,
        is_screen_content,
        DELTAQ_RATE_RATIO[rate_level],
        best,
        worst,
    );
    // Disallow a segment qindex 0 when the base is not 0 (lossless guard).
    if base_qindex != 0 && (base_qindex + qindex_delta) == 0 {
        qindex_delta = -base_qindex + 1;
    }
    base_qindex + qindex_delta
}

/// `rc->best/worst_quality` for the allintra default `--min-q=0 --max-q=63`
/// (`av1_quantizer_to_qindex(0)=0`, `(63)=255`).
pub const PERCEPTUAL_BEST_QUALITY: i32 = MINQ;
pub const PERCEPTUAL_WORST_QUALITY: i32 = MAXQ;

/// The per-SB qindex of `setup_delta_q` (encodeframe.c:330-342) under
/// `DELTA_Q_PERCEPTUAL` (mode 2, wavelet arm): the SB wavelet energy level
/// ([`block_wavelet_energy_level`]) → the rate-ratio qindex
/// ([`compute_q_from_energy_level`], keyed on the FRAME `base_qindex`), then
/// the SAME deadzone-quantize against the RUNNING `current_base_qindex` modes
/// 3/6 use. `sb_w`/`sb_h` are the SB pixel dims (`block_size_wide/high`),
/// `num_pels_log2` is `num_pels_log2_lookup[sb_size]`.
#[allow(clippy::too_many_arguments)]
pub fn setup_delta_q_perceptual(
    src: &[u16],
    sb_off: usize,
    stride: usize,
    bd: u8,
    base_qindex: i32,
    is_screen_content: bool,
    sb_w: usize,
    sb_h: usize,
    num_pels_log2: u32,
    delta_q_res: i32,
    current_base_qindex: i32,
) -> i32 {
    let level = block_wavelet_energy_level(src, sb_off, stride, bd, sb_w, sb_h, num_pels_log2);
    let current = compute_q_from_energy_level(
        base_qindex,
        bd,
        is_screen_content,
        level,
        PERCEPTUAL_BEST_QUALITY,
        PERCEPTUAL_WORST_QUALITY,
    );
    av1_adjust_q_from_delta_q_res(delta_q_res, current_base_qindex, current)
}

// ===========================================================================
// `--deltaq-mode=3` (DELTA_Q_PERCEPTUAL_AI, family C5): the wiener-variance
// per-superblock qindex map. Ports (libaom v3.14.1, allintra_vis.c):
//   - `WeberStats` (encoder.h:2363): the per-8x8 source/recon statistics.
//   - `get_satd` / `get_sse` / `get_max_scale` / `get_window_wiener_var` /
//     `get_var_perceptual_ai` (:93-246): the map-window reductions.
//   - `av1_get_sbq_perceptual_ai` (:743): the per-SB qindex from the wiener
//     variance vs the frame `norm_wiener_variance`, via `av1_get_deltaq_offset`.
// The heavy preprocessing that BUILDS the map + `norm_wiener_variance`
// (`av1_set_mb_wiener_variance`) lands separately. All f64 `sqrt`/`log`/`exp`
// resolve to the same glibc libm as the C build (same envelope note as the
// Variance-Boost `log2`), so the byte gates hold locally.
// ===========================================================================

/// `DEFAULT_DELTA_Q_RES_PERCEPTUAL` (enums.h:499) — the CONSTANT delta-q grid
/// resolution for `DELTA_Q_PERCEPTUAL` / `DELTA_Q_PERCEPTUAL_AI`
/// (encodeframe.c:2289-2290), unlike Variance Boost's qindex-dependent res.
pub const DELTA_Q_RES_PERCEPTUAL: i32 = 4;

/// `mi_size_wide[BLOCK_8X8]` — the `weber_bsize` mi step the per-8x8 wiener
/// map is indexed on (`cpi->weber_bsize = BLOCK_8X8`, allintra_vis.c:66).
const WEBER_MI_STEP: i32 = 2;

/// `WeberStats` (encoder.h:2363): the per-8x8 source/recon statistics
/// `av1_set_mb_wiener_variance` fills for the perceptual-AI delta-q map.
/// `mb_wiener_variance` (the struct's first field) is written but never read
/// by any map reduction, so it is omitted here.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WeberStats {
    pub src_variance: i64,
    pub rec_variance: i64,
    pub src_pix_max: i16,
    pub rec_pix_max: i16,
    pub distortion: i64,
    pub satd: i64,
    pub max_scale: f64,
}

/// The frame-level wiener-variance map + normalizer that the per-SB
/// perceptual-AI qindex reads (`cpi->mb_weber_stats` + `norm_wiener_variance`).
/// `stats` is laid out exactly as C's `aom_calloc(mi_rows*mi_cols)` and indexed
/// `(mi_row/2)*mi_cols + mi_col/2` (`frame_info.mi_cols == mi_params.mi_cols`,
/// encoder.c:1102). Bounds come from the same `mi_cols`/`mi_rows`.
pub struct WeberVarMap {
    pub stats: Vec<WeberStats>,
    pub mi_rows: i32,
    pub mi_cols: i32,
    pub norm_wiener_variance: i64,
}

impl WeberVarMap {
    #[inline]
    fn at(&self, row: i32, col: i32) -> &WeberStats {
        &self.stats[((row / WEBER_MI_STEP) * self.mi_cols + col / WEBER_MI_STEP) as usize]
    }

    /// `get_satd` (allintra_vis.c:93): mean `.satd` over the in-frame 8x8
    /// blocks of the `mi_wide`×`mi_high` window, `>= 1`. The `(int)` casts on
    /// the divide + return are replicated (the accumulation is i64).
    fn get_satd(&self, mi_wide: i32, mi_high: i32, mi_row: i32, mi_col: i32) -> i64 {
        let mut satd: i64 = 0;
        let mut mb_count: i32 = 0;
        let mut row = mi_row;
        while row < mi_row + mi_high {
            let mut col = mi_col;
            while col < mi_col + mi_wide {
                if !(row >= self.mi_rows || col >= self.mi_cols) {
                    satd += self.at(row, col).satd;
                    mb_count += 1;
                }
                col += WEBER_MI_STEP;
            }
            row += WEBER_MI_STEP;
        }
        if mb_count != 0 {
            satd = i64::from((satd / i64::from(mb_count)) as i32);
        }
        satd.max(1)
    }

    /// `get_sse` (allintra_vis.c:121): mean `.distortion` over the window,
    /// `>= 1` (same `(int)`-cast structure as [`Self::get_satd`]).
    fn get_sse(&self, mi_wide: i32, mi_high: i32, mi_row: i32, mi_col: i32) -> i64 {
        let mut distortion: i64 = 0;
        let mut mb_count: i32 = 0;
        let mut row = mi_row;
        while row < mi_row + mi_high {
            let mut col = mi_col;
            while col < mi_col + mi_wide {
                if !(row >= self.mi_rows || col >= self.mi_cols) {
                    distortion += self.at(row, col).distortion;
                    mb_count += 1;
                }
                col += WEBER_MI_STEP;
            }
            row += WEBER_MI_STEP;
        }
        if mb_count != 0 {
            distortion = i64::from((distortion / i64::from(mb_count)) as i32);
        }
        distortion.max(1)
    }

    /// `get_max_scale` (allintra_vis.c:150): the min `.max_scale >= 1.0` over
    /// the window, seeded at `10.0` (blocks with `max_scale < 1.0` skipped).
    fn get_max_scale(&self, mi_wide: i32, mi_high: i32, mi_row: i32, mi_col: i32) -> f64 {
        let mut min_max_scale = 10.0f64;
        let mut row = mi_row;
        while row < mi_row + mi_high {
            let mut col = mi_col;
            while col < mi_col + mi_wide {
                if !(row >= self.mi_rows || col >= self.mi_cols) {
                    let ms = self.at(row, col).max_scale;
                    if ms >= 1.0 && ms < min_max_scale {
                        min_max_scale = ms;
                    }
                }
                col += WEBER_MI_STEP;
            }
            row += WEBER_MI_STEP;
        }
        min_max_scale
    }

    /// `get_window_wiener_var` (allintra_vis.c:173): the wiener-variance
    /// estimate over one window — a distortion/contrast ratio with a `0.1`
    /// regularizer, `/ mb_count`, `>= 1`. All accumulators start at `1.0`.
    fn get_window_wiener_var(&self, mi_wide: i32, mi_high: i32, mi_row: i32, mi_col: i32) -> i32 {
        let mut mb_count: i32 = 0;
        let mut base_num = 1.0f64;
        let mut base_den = 1.0f64;
        let mut base_reg = 1.0f64;
        let mut row = mi_row;
        while row < mi_row + mi_high {
            let mut col = mi_col;
            while col < mi_col + mi_wide {
                if !(row >= self.mi_rows || col >= self.mi_cols) {
                    let w = self.at(row, col);
                    base_num += (w.distortion as f64)
                        * (w.src_variance as f64).sqrt()
                        * f64::from(w.rec_pix_max);
                    base_den += (f64::from(w.rec_pix_max) * (w.src_variance as f64).sqrt()
                        - f64::from(w.src_pix_max) * (w.rec_variance as f64).sqrt())
                    .abs();
                    base_reg += (w.distortion as f64).sqrt() * f64::from(w.src_pix_max).sqrt() * 0.1;
                    mb_count += 1;
                }
                col += WEBER_MI_STEP;
            }
            row += WEBER_MI_STEP;
        }
        let sb_wiener_var = (((base_num + base_reg) / (base_den + base_reg)) / mb_count as f64) as i32;
        sb_wiener_var.max(1)
    }

    /// `get_var_perceptual_ai` (allintra_vis.c:216): the window wiener var of
    /// the SB, min'd with the four half-SB-shifted neighbour windows that stay
    /// in frame — a spatial smoothing that damps isolated peaks.
    fn get_var_perceptual_ai(&self, mi_wide: i32, mi_high: i32, mi_row: i32, mi_col: i32) -> i32 {
        let mut sb = self.get_window_wiener_var(mi_wide, mi_high, mi_row, mi_col);
        if mi_row >= mi_high / 2 {
            sb = sb.min(self.get_window_wiener_var(mi_wide, mi_high, mi_row - mi_high / 2, mi_col));
        }
        if mi_row <= self.mi_rows - mi_high - (mi_high / 2) {
            sb = sb.min(self.get_window_wiener_var(mi_wide, mi_high, mi_row + mi_high / 2, mi_col));
        }
        if mi_col >= mi_wide / 2 {
            sb = sb.min(self.get_window_wiener_var(mi_wide, mi_high, mi_row, mi_col - mi_wide / 2));
        }
        if mi_col <= self.mi_cols - mi_wide - (mi_wide / 2) {
            sb = sb.min(self.get_window_wiener_var(mi_wide, mi_high, mi_row, mi_col + mi_wide / 2));
        }
        sb
    }

    /// `av1_get_sbq_perceptual_ai` (allintra_vis.c:743, the default
    /// non-rate-guide arm): the per-SB qindex. `beta = norm / sb_wiener_var`,
    /// floored by `1/min_max_scale`, clamped to `[0.25, 4]`, mapped to a
    /// qindex offset ([`av1_get_deltaq_offset`]), clamped to
    /// `±(delta_q_res*20 - 1)`, then to `[MINQ(+1), MAXQ]`. `bit_depth` is
    /// the raw 8/10/12; `mi_wide`/`mi_high` are the SB's mi extent.
    #[allow(clippy::too_many_arguments)]
    pub fn av1_get_sbq_perceptual_ai(
        &self,
        base_qindex: i32,
        bit_depth: u8,
        delta_q_res: i32,
        mi_wide: i32,
        mi_high: i32,
        mi_row: i32,
        mi_col: i32,
    ) -> i32 {
        let sb_wiener_var = self.get_var_perceptual_ai(mi_wide, mi_high, mi_row, mi_col);
        let mut beta = self.norm_wiener_variance as f64 / f64::from(sb_wiener_var);
        let min_max_scale = self.get_max_scale(mi_wide, mi_high, mi_row, mi_col).max(1.0);
        beta = 1.0 / (1.0 / beta).min(min_max_scale);
        // Cap so the delta q stays near the base q.
        beta = beta.min(4.0);
        beta = beta.max(0.25);
        let mut offset = av1_get_deltaq_offset(bit_depth, base_qindex, beta);
        offset = offset.min(delta_q_res * 20 - 1);
        offset = offset.max(-delta_q_res * 20 + 1);
        let mut qindex = base_qindex + offset;
        qindex = qindex.min(MAXQ);
        qindex = qindex.max(MINQ);
        if base_qindex > MINQ {
            qindex = qindex.max(MINQ + 1);
        }
        qindex
    }
}

impl WeberVarMap {
    /// `estimate_wiener_var_norm` (allintra_vis.c:490): the first estimate of
    /// `norm_wiener_variance` — a satd/sqrt(sse)-weighted geometric mean of the
    /// per-SB wiener variance (`exp(sum(w*ln(var)) / sum(w))`), `>= 1`.
    fn estimate_norm(&self, sb_mi: i32) -> i64 {
        let mut sb_wiener_log = 0.0f64;
        let mut sb_count = 0.0f64;
        let mut row = 0i32;
        while row < self.mi_rows {
            let mut col = 0i32;
            while col < self.mi_cols {
                let var = self.get_var_perceptual_ai(sb_mi, sb_mi, row, col);
                let satd = self.get_satd(sb_mi, sb_mi, row, col);
                let sse = self.get_sse(sb_mi, sb_mi, row, col);
                let scaled_satd = satd as f64 / (sse as f64).sqrt();
                sb_wiener_log += scaled_satd * f64::from(var).ln();
                sb_count += scaled_satd;
                col += sb_mi;
            }
            row += sb_mi;
        }
        let mut norm = 1i64;
        if sb_count > 0.0 {
            norm = (sb_wiener_log / sb_count).exp() as i64;
        }
        norm.max(1)
    }

    /// One refinement iteration of `norm_wiener_variance` (allintra_vis.c:649-679,
    /// run twice): re-weights each SB by `norm/beta` with `beta` clamped to
    /// `[0.25, 4]` and SBs whose `beta < 1/min_max_scale` skipped, then re-takes
    /// the weighted geometric mean.
    fn refine_norm(&self, sb_mi: i32, norm: i64) -> i64 {
        let mut sb_wiener_log = 0.0f64;
        let mut sb_count = 0.0f64;
        let mut row = 0i32;
        while row < self.mi_rows {
            let mut col = 0i32;
            while col < self.mi_cols {
                let var = self.get_var_perceptual_ai(sb_mi, sb_mi, row, col);
                let mut beta = norm as f64 / f64::from(var);
                let min_max_scale = self.get_max_scale(sb_mi, sb_mi, row, col).max(1.0);
                beta = beta.min(4.0);
                beta = beta.max(0.25);
                if beta < 1.0 / min_max_scale {
                    col += sb_mi;
                    continue;
                }
                let var = (norm as f64 / beta) as i32;
                let satd = self.get_satd(sb_mi, sb_mi, row, col);
                let sse = self.get_sse(sb_mi, sb_mi, row, col);
                let scaled_satd = satd as f64 / (sse as f64).sqrt();
                sb_wiener_log += scaled_satd * f64::from(var).ln();
                sb_count += scaled_satd;
                col += sb_mi;
            }
            row += sb_mi;
        }
        let mut out = norm;
        if sb_count > 0.0 {
            out = (sb_wiener_log / sb_count).exp() as i64;
        }
        out.max(1)
    }

    /// `norm_wiener_variance` (allintra_vis.c:644-680): the initial estimate then
    /// the two refinement iterations. `sb_mi` is `mi_size_wide[sb_size]`.
    fn compute_norm_wiener_variance(&self, sb_mi: i32) -> i64 {
        let mut norm = self.estimate_norm(sb_mi);
        for _ in 0..2 {
            norm = self.refine_norm(sb_mi, norm);
        }
        norm
    }
}

/// `av1_set_mb_wiener_variance` (allintra_vis.c:592) — build the per-8x8
/// [`WeberVarMap`] + `norm_wiener_variance` for `--deltaq-mode=3`. For each 8x8
/// source block: the intra-mode SATD search over all 13 intra modes at
/// `angle_delta = 0` (`av1_calc_mb_wiener_var_row`, :343-360) with the SOURCE
/// pixels as the predictor neighbours (:345-347 uses src, not recon, so
/// single/multi-thread match), then FP-quantize the DCT of the best mode's
/// residual (`AV1_XFORM_QUANT_FP`), reconstruct, and record the Weber stats.
/// Finally derive `norm_wiener_variance`.
///
/// **Scope (this landing): bd8, single tile, frame dims a multiple of 8px.**
/// The highbd FP-quantize arm and the partial-edge source-border extension
/// (the KB-6 partial-SB analogue) are follow-ups. `base_qindex` is the frame
/// qindex (`rc_cfg.cq_level`, :612); `sb_size` is the seq SB BLOCK enum and
/// `sb_mi` its mi extent (the norm grid step). `disable_edge_filter` is
/// `!seq_params->enable_intra_edge_filter`; the per-block edge `filter_type`
/// is `0` (the preprocessing nulls the above/left mbmi, :335-339).
/// One 8x8 wiener block's predict→subtract→forward-DCT (the body shared by the
/// mode-search loop and the best-mode requant in [`av1_set_mb_wiener_variance`]).
/// Predicts `mode` at `angle_delta = 0` from the SOURCE neighbours into `pred`,
/// writes `residual = src - pred`, and the DCT_DCT forward transform into
/// `coeff` (`av1_quick_txfm(use_hadamard=0)`).
#[allow(clippy::too_many_arguments)]
fn wiener_block_residual_dct(
    src_y: &[u16],
    src_off: usize,
    stride: usize,
    sb_size: usize,
    row: i32,
    col: i32,
    mi_rows: i32,
    mi_cols: i32,
    mode: usize,
    disable_edge_filter: bool,
    bd: u8,
    pred: &mut [u16],
    residual: &mut [i16],
    coeff: &mut [i32],
) {
    const BLOCK_8X8: usize = 3;
    const TX_8X8: usize = 1;
    const DCT_DCT: usize = 0;
    const PARTITION_NONE: usize = 0;
    const BS: usize = 8;
    let (n_top, n_topright, n_left, n_bottomleft) = aom_entropy::partition::intra_avail(
        sb_size, BLOCK_8X8, row, col, row > 0, col > 0, mi_cols, mi_rows, PARTITION_NONE, TX_8X8, 0,
        0, 0, 0, BS as i32, BS as i32, mi_cols, mi_rows, mode, 0, false,
    );
    aom_intra::predict_intra_high(
        src_y,
        src_off,
        stride,
        pred,
        BS,
        mode,
        0,
        false,
        0,
        disable_edge_filter,
        0,
        TX_8X8,
        n_top as usize,
        n_topright,
        n_left as usize,
        n_bottomleft,
        i32::from(bd),
    );
    aom_dist::highbd_subtract_block(BS, BS, residual, BS, &src_y[src_off..], stride, pred, BS);
    aom_transform::txfm2d::av1_fwd_txfm2d(residual, coeff, BS, DCT_DCT, TX_8X8);
}

#[allow(clippy::too_many_arguments)]
pub fn av1_set_mb_wiener_variance(
    src_y: &[u16],
    base_y: usize,
    stride: usize,
    mi_rows: i32,
    mi_cols: i32,
    base_qindex: i32,
    bd: u8,
    quants: &aom_quant::Quants,
    deq: &aom_quant::Dequants,
    sb_size: usize,
    sb_mi: i32,
    disable_edge_filter: bool,
) -> WeberVarMap {
    const TX_8X8: usize = 1;
    const DCT_DCT: usize = 0;
    const INTRA_MODE_END: usize = 13; // NEARESTMV (INTRA_MODE_START = DC_PRED = 0)
    const BS: usize = 8; // tx_size_wide[TX_8X8]
    const N: usize = BS * BS;

    let qi = base_qindex as usize;
    let round = [quants.y_round_fp[qi][0], quants.y_round_fp[qi][1]];
    let quant = [quants.y_quant_fp[qi][0], quants.y_quant_fp[qi][1]];
    let dequant = [deq.y_dequant_qtx[qi][0], deq.y_dequant_qtx[qi][1]];
    let scan = aom_txb::scan(TX_8X8, DCT_DCT);

    let mut stats = vec![WeberStats::default(); (mi_rows * mi_cols) as usize];

    let mut pred = [0u16; N];
    let mut residual = [0i16; N];
    let mut coeff = [0i32; N];
    let mut qcoeff = [0i32; N];
    let mut dqcoeff = [0i32; N];

    let mut row = 0i32;
    while row < mi_rows {
        let mut col = 0i32;
        while col < mi_cols {
            let src_off = base_y + (row as usize * 4) * stride + col as usize * 4;

            // --- intra-mode SATD search (av1_calc_mb_wiener_var_row :343-360) ---
            let mut best_mode = 0usize; // DC_PRED
            let mut best_intra_cost = i32::MAX;
            for mode in 0..INTRA_MODE_END {
                wiener_block_residual_dct(
                    src_y, src_off, stride, sb_size, row, col, mi_rows, mi_cols, mode,
                    disable_edge_filter, bd, &mut pred, &mut residual, &mut coeff,
                );
                let intra_cost = aom_dist::hadamard::satd(&coeff);
                if intra_cost < best_intra_cost {
                    best_intra_cost = intra_cost;
                    best_mode = mode;
                }
            }

            // --- best mode: predict, DCT, FP-quantize, reconstruct (:362-396) ---
            wiener_block_residual_dct(
                src_y, src_off, stride, sb_size, row, col, mi_rows, mi_cols, best_mode,
                disable_edge_filter, bd, &mut pred, &mut residual, &mut coeff,
            );
            qcoeff.fill(0);
            dqcoeff.fill(0);
            let eob = aom_quant::av1_quantize_fp(
                &coeff, &round, &quant, &dequant, &mut qcoeff, &mut dqcoeff, scan,
            );
            // pred += inv(dqcoeff): pred now holds the reconstruction.
            aom_transform::inv_txfm2d::av1_inverse_transform_add(
                &dqcoeff, &mut pred, BS, DCT_DCT, TX_8X8, i32::from(bd), eob as usize, false,
            );

            // --- Weber statistics (:397-460) ---
            let mut w = WeberStats {
                src_pix_max: 1,
                rec_pix_max: 1,
                ..WeberStats::default()
            };
            let (mut src_mean, mut rec_mean, mut dist_mean) = (0i64, 0i64, 0i64);
            for pr in 0..BS {
                for pc in 0..BS {
                    let src_pix = i32::from(src_y[src_off + pr * stride + pc]);
                    let rec_pix = i32::from(pred[pr * BS + pc]);
                    src_mean += i64::from(src_pix);
                    rec_mean += i64::from(rec_pix);
                    dist_mean += i64::from(src_pix - rec_pix);
                    w.src_variance += i64::from(src_pix) * i64::from(src_pix);
                    w.rec_variance += i64::from(rec_pix) * i64::from(rec_pix);
                    w.src_pix_max = w.src_pix_max.max(src_pix as i16);
                    w.rec_pix_max = w.rec_pix_max.max(rec_pix as i16);
                    let d = src_pix - rec_pix;
                    w.distortion += i64::from(d * d);
                }
            }
            let pix_num = N as i64;
            w.src_variance -= (src_mean * src_mean) / pix_num;
            w.rec_variance -= (rec_mean * rec_mean) / pix_num;
            w.distortion -= (dist_mean * dist_mean) / pix_num;
            w.satd = i64::from(best_intra_cost);
            let mut max_scale = 0i32;
            for &qc in &qcoeff[1..N] {
                max_scale = max_scale.max(qc.abs());
            }
            w.max_scale = f64::from(max_scale);

            stats[((row / WEBER_MI_STEP) * mi_cols + col / WEBER_MI_STEP) as usize] = w;
            col += WEBER_MI_STEP;
        }
        row += WEBER_MI_STEP;
    }

    let mut map = WeberVarMap {
        stats,
        mi_rows,
        mi_cols,
        norm_wiener_variance: 0,
    };
    map.norm_wiener_variance = map.compute_norm_wiener_variance(sb_mi);
    map
}
