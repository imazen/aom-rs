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

use aom_quant::av1_ac_quant_qtx;

/// `MAXQ` / `MINQ` (av1/common/quant_common.h).
const MAXQ: i32 = 255;
const MINQ: i32 = 0;

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
