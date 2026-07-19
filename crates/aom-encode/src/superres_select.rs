//! Encoder-side superres denominator SELECTION —
//! `av1/encoder/superres_scale.c`.
//!
//! The FIXED mode takes the denominator straight from the user knob (handled by
//! the resize/downscale path in [`crate::resize`]). The QTHRESH / AUTO / RANDOM
//! modes *derive* the denominator per frame; this module ports that derivation
//! for the single-frame ALLINTRA / KEY envelope (the same envelope the rest of
//! the encoder track byte-matches). The chosen denominator then feeds the
//! identical downscale + coded-width encode + `write_superres_scale` pipeline
//! the FIXED gate already proves byte-exact.
//!
//! What is ported here (KEY frame, `rc_end_usage = AOM_Q`, one-pass, no resize):
//! - `analyze_hor_freq` — the 16×4 H_DCT horizontal-frequency energy analysis
//!   over the source luma (bit-exact vs the exported `av1_fwd_txfm2d_16x4`).
//! - `get_superres_denom_from_qindex_energy` — the energy/qindex threshold walk.
//! - `get_superres_denom_for_qindex` (KEY, `sr_kf = 1`, KF_UPDATE) — wraps the
//!   two above with the `energy_by_q2` threshold. The AUTO-only recode bump
//!   (`av1_superres_in_recode_allowed`) never fires for QTHRESH/RANDOM.
//! - `calculate_next_superres_scale` — the QTHRESH / RANDOM / (SOLO) AUTO arms.
//!
//! Validated: `analyze_hor_freq` + `get_superres_denom_from_qindex_energy` vs
//! the faithful C-facade shim (`aom_sys_ref::ref_superres_*`, calling the real
//! exported leaf math), AND end-to-end vs real `aomenc --superres-mode=...`
//! (the denom the real encoder actually chose is embedded in the stream — the
//! top-tier evidence).

use aom_dsp::transform::txfm2d::av1_fwd_txfm2d;

/// `SCALE_NUMERATOR` — superres denominators are relative to 8 (denom 8 == no
/// superres; the coded width equals the upscaled width).
pub const SCALE_NUMERATOR: u8 = 8;

/// `H_DCT` tx_type / `TX_16X4` tx_size indices for `av1_fwd_txfm2d`.
const H_DCT: usize = 11;
const TX_16X4: usize = 14;

// superres_scale.c thresholds.
const SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME_SOLO: f64 = 0.012;
const SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME: f64 = 0.008;
const SUPERRES_ENERGY_BY_AC_THRESH: f64 = 0.2;

/// `lcg_next` (av1/encoder/random.h): the 32-bit LCG step.
#[inline]
fn lcg_next(state: &mut u32) -> u32 {
    *state = (u64::from(*state) * 1_103_515_245u64 + 12345) as u32;
    *state
}

/// `lcg_rand16` (av1/encoder/random.h): `(lcg_next(state) / 65536) % 32768`.
#[inline]
fn lcg_rand16(state: &mut u32) -> u32 {
    (lcg_next(state) / 65536) % 32768
}

/// `ROUND_POWER_OF_TWO(value, n)` for `u64` (`aom_dsp_common.h`).
#[inline]
fn round_power_of_two_u64(value: u64, n: u32) -> u64 {
    (value + (1u64 << (n - 1))) >> n
}

/// `av1_convert_qindex_to_q(qindex, AOM_BITS_8)` (ratectrl.c:199):
/// `av1_ac_quant_QTX(qindex, 0, AOM_BITS_8) / 4.0`. The energy threshold always
/// uses the 8-bit conversion regardless of the source bit depth (superres_scale.c
/// hardcodes `AOM_BITS_8`).
#[inline]
fn av1_convert_qindex_to_q_bits8(qindex: i32) -> f64 {
    f64::from(aom_dsp::quant::av1_ac_quant_qtx(qindex, 0, 8)) / 4.0
}

/// `analyze_hor_freq` (superres_scale.c): the 16×4 horizontal DCT
/// frequency-energy analysis over the source luma plane.
///
/// `src[i * stride + j]` is the luma sample at row `i`, col `j` (values
/// `0..(2^bd - 1)`; for a tight plane `stride == width`). Returns the 16-entry
/// cumulative energy vector (`energy[0]` is unused, matching C which only
/// writes `energy[1..=15]`).
///
/// The transform result is bit-identical to `av1_fwd_txfm2d_16x4` for any
/// `bd` (the forward transform's only `bd` dependence is a no-op range check in
/// the production build); the sole `bd`-dependent step is the accumulation shift
/// `2 + 2*(bd - 8)`. C reads the 8-bit and high-bit-depth planes through
/// different pointer paths but transforms the identical 16×4 pixel window, so
/// building a tight window here is bit-equivalent for both.
#[must_use]
pub fn analyze_hor_freq(
    src: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bd: u8,
) -> [f64; 16] {
    let mut freq_energy = [0u64; 16];
    // bd8 -> 2, bd10 -> 6, bd12 -> 10.
    let shift = (2 + 2 * (i32::from(bd) - 8)) as u32;
    let w = width as i32;
    let h = height as i32;
    let mut coeff = [0i32; 64];
    let mut blk = [0i16; 64];
    let mut n: u64 = 0;

    let mut i = 0i32;
    while i < h - 4 {
        let mut j = 0i32;
        while j < w - 16 {
            for ii in 0..4usize {
                let row = (i as usize + ii) * stride + j as usize;
                for jj in 0..16usize {
                    blk[ii * 16 + jj] = src[row + jj] as i16;
                }
            }
            av1_fwd_txfm2d(&blk, &mut coeff, 16, H_DCT, TX_16X4);
            for k in 1..16usize {
                let c0 = i64::from(coeff[k]);
                let c1 = i64::from(coeff[k + 16]);
                let c2 = i64::from(coeff[k + 32]);
                let c3 = i64::from(coeff[k + 48]);
                let this_energy = (c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3) as u64;
                freq_energy[k] += round_power_of_two_u64(this_energy, shift);
            }
            n += 1;
            j += 16;
        }
        i += 4;
    }

    let mut energy = [0f64; 16];
    if n != 0 {
        for k in 1..16usize {
            energy[k] = freq_energy[k] as f64 / n as f64;
        }
        // Convert to cumulative energy (C: `for (k=14; k>0; --k) energy[k] += energy[k+1]`).
        let mut k = 14i32;
        while k > 0 {
            energy[k as usize] += energy[(k + 1) as usize];
            k -= 1;
        }
    } else {
        for k in 1..16usize {
            energy[k] = 1e+20;
        }
    }
    energy
}

/// `get_superres_denom_from_qindex_energy` (superres_scale.c).
#[must_use]
pub fn get_superres_denom_from_qindex_energy(
    qindex: i32,
    energy: &[f64; 16],
    threshq: f64,
    threshp: f64,
) -> u8 {
    let q = av1_convert_qindex_to_q_bits8(qindex);
    let tq = threshq * q * q;
    let tp = threshp * energy[1];
    // AOMMIN(tq, tp) — plain `(a < b) ? a : b`.
    let thresh = if tq < tp { tq } else { tp };
    let sn = i32::from(SCALE_NUMERATOR);
    let mut k = sn * 2; // 16
    while k > sn {
        if energy[(k - 1) as usize] > thresh {
            break;
        }
        k -= 1;
    }
    (3 * sn - k) as u8
}

/// `get_superres_denom_for_qindex` (superres_scale.c) restricted to a KEY frame
/// (`sr_kf = 1`, `update_type == KF_UPDATE`). `frames_to_key_le_1` selects the
/// SOLO vs non-SOLO `energy_by_q2` threshold (single-frame KEY ⇒ SOLO). The
/// recode bump is AUTO-only and never applies here.
#[must_use]
pub fn get_superres_denom_for_qindex_key(
    src: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bd: u8,
    qindex: i32,
    frames_to_key_le_1: bool,
) -> u8 {
    let energy = analyze_hor_freq(src, width, height, stride, bd);
    let energy_by_q2_thresh = if frames_to_key_le_1 {
        SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME_SOLO
    } else {
        SUPERRES_ENERGY_BY_Q2_THRESH_KEYFRAME
    };
    get_superres_denom_from_qindex_energy(
        qindex,
        &energy,
        energy_by_q2_thresh,
        SUPERRES_ENERGY_BY_AC_THRESH,
    )
}

/// The QTHRESH arm of `calculate_next_superres_scale` for a KEY frame in the
/// single-frame AOM_Q envelope.
///
/// - `q` is the qindex the rate controller picked (for AOM_Q + single KEY frame
///   this is `rc::base_qindex_from_cq(cq)` — see [`crate::rc`]).
/// - `kf_qthresh_qindex` is `av1_quantizer_to_qindex(--superres-kf-qthresh)`
///   (the 1..63 knob converted to a 0..255 qindex, av1_cx_iface.c:1590).
/// - `allow_screen_content_tools` short-circuits to no-superres (C `break`).
///
/// Returns the denominator (8 == no superres).
#[must_use]
pub fn superres_denom_qthresh_key(
    src: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bd: u8,
    q: i32,
    kf_qthresh_qindex: i32,
    allow_screen_content_tools: bool,
    frames_to_key_le_1: bool,
) -> u8 {
    if allow_screen_content_tools {
        return SCALE_NUMERATOR;
    }
    // AOM_Q mode: `av1_set_target_rate` is not called (only VBR/CQ).
    if q <= kf_qthresh_qindex {
        return SCALE_NUMERATOR;
    }
    get_superres_denom_for_qindex_key(src, width, height, stride, bd, q, frames_to_key_le_1)
}

/// `SUPERRES_AUTO_SEARCH_TYPE` (speed_features.h): the AUTO-mode search strategy.
/// For ALLINTRA it is unconditionally `Dual`
/// (`set_allintra_speed_features_framesize_independent`, speed_features.c:384).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SuperresAutoSearchType {
    /// Tries all possible superres ratios (uses the user fixed denom).
    All,
    /// Tries no-superres AND the q-based ratio (the ALLINTRA default).
    Dual,
    /// Only applies the q-based ratio.
    Solo,
}

/// The AUTO arm of `calculate_next_superres_scale` for a KEY frame in the
/// single-frame AOM_Q envelope.
///
/// For the recode loop to run, `av1_superres_in_recode_allowed` requires
/// `search_type != Solo && frames_to_key > 1`; a single KEY still has
/// `frames_to_key <= 1`, so recode never fires and the AUTO-derived denom is
/// used directly (the recode-based multi-pass search is a follow-up that is
/// structurally unreachable for a single-frame KEY still). `frames_to_key_le_1`
/// also gates the SOLO recode bump (`AOMMAX(denom, 9)`) inside
/// `get_superres_denom_for_qindex`, which likewise never applies here.
///
/// - `Dual`/`Solo`: derive the denom from the energy analysis (`Solo` uses a
///   qthresh of 128, `Dual` uses 0).
/// - `All`: use the user fixed KEY denom (`kf_scale_denominator`).
///
/// Returns the denominator (8 == no superres).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn superres_denom_auto_key(
    src: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bd: u8,
    q: i32,
    allow_screen_content_tools: bool,
    frames_to_key_le_1: bool,
    search_type: SuperresAutoSearchType,
    kf_scale_denominator: u8,
) -> u8 {
    if allow_screen_content_tools {
        return SCALE_NUMERATOR;
    }
    let qthresh = if search_type == SuperresAutoSearchType::Solo {
        128
    } else {
        0
    };
    if q <= qthresh {
        return SCALE_NUMERATOR;
    }
    if search_type == SuperresAutoSearchType::All {
        kf_scale_denominator
    } else {
        get_superres_denom_for_qindex_key(src, width, height, stride, bd, q, frames_to_key_le_1)
    }
}

/// The initial superres-RANDOM seed (`static unsigned int seed = 34567` in
/// `calculate_next_superres_scale`). In libaom this is a FUNCTION-STATIC that
/// persists across every RANDOM frame in the process — the Nth RANDOM frame
/// encoded draws the Nth value of the sequence.
pub const SUPERRES_RANDOM_SEED_INIT: u32 = 34567;

/// One draw of the RANDOM arm of `calculate_next_superres_scale`:
/// `lcg_rand16(&seed) % 9 + 8`, advancing `seed`. Thread `seed` (starting at
/// [`SUPERRES_RANDOM_SEED_INIT`]) across successive RANDOM frames to reproduce
/// libaom's process-global static-seed sequence (34567 → 11, 14, 15, 9, …).
#[must_use]
pub fn superres_denom_random(seed: &mut u32) -> u8 {
    (lcg_rand16(seed) % 9 + 8) as u8
}

/// The denominator libaom's RANDOM mode picks for the FIRST RANDOM frame in a
/// fresh process (the single-frame drop-in case): `seed = 34567` → 11.
#[must_use]
pub fn superres_denom_random_first_frame() -> u8 {
    let mut seed = SUPERRES_RANDOM_SEED_INIT;
    superres_denom_random(&mut seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_first_frame_is_deterministic() {
        // seed 34567 -> first lcg_rand16 draw -> denom 11 (see the RANDOM arm).
        assert_eq!(superres_denom_random_first_frame(), 11);
    }

    #[test]
    fn random_seed_sequence_matches_c_static() {
        // The process-global static seed (34567) advances once per RANDOM frame;
        // the first four draws are 11, 14, 15, 9 (verified vs real aomenc across
        // consecutive encodes in one process).
        let mut seed = SUPERRES_RANDOM_SEED_INIT;
        let seq: Vec<u8> = (0..4).map(|_| superres_denom_random(&mut seed)).collect();
        assert_eq!(seq, vec![11, 14, 15, 9]);
    }

    #[test]
    fn convert_qindex_matches_ac_quant_over_4() {
        for q in [0, 32, 96, 128, 200, 255] {
            let want = f64::from(aom_dsp::quant::av1_ac_quant_qtx(q, 0, 8)) / 4.0;
            assert_eq!(av1_convert_qindex_to_q_bits8(q), want);
        }
    }

    #[test]
    fn tiny_frame_yields_no_superres() {
        // width <= 16 or height <= 4 -> n == 0 -> energy = 1e20 -> denom 8.
        let src = vec![100u16; 12 * 12];
        let e = analyze_hor_freq(&src, 12, 12, 12, 8);
        assert_eq!(e[1], 1e20);
        assert_eq!(
            get_superres_denom_from_qindex_energy(128, &e, 0.012, 0.2),
            8
        );
    }
}
