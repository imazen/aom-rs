//! Rate-control-level qindex derivation.
//!
//! The single seam this module owns is the constant-quality (`AOM_Q`) mapping
//! from the user-facing 0-63 `--cq-level` value to the internal 0-255
//! `base_qindex` that libaom writes into the frame header. In the port's
//! single-frame ALLINTRA / GOOD KEY-frame envelope this replaces reading
//! `base_qindex` straight off the real parsed header — the port now derives it
//! itself, byte-identically to `aomenc` (see
//! `tests/qindex_from_cq_diff.rs`).
//!
//! This is deliberately isolated from the quantizer *kernels*
//! (`aom_dsp::quant::{av1_build_quantizer, set_q_index, av1_quantize_*}`): those turn
//! a qindex into dequant tables, whereas this turns a cq level into a qindex.
//! No kernel state is read or modified here.

/// libaom `quantizer_to_qindex` (`av1/encoder/av1_quantize.c:1033`): the fixed
/// table that converts the 0-63 Q-range value passed in from outside to the
/// 0-255 qindex range used internally. Note the non-uniform top end
/// (`... 240, 244, 249, 255`) — the last two steps are +5/+6, not +4.
const QUANTIZER_TO_QINDEX: [i32; 64] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92,
    96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168,
    172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244,
    249, 255,
];

/// libaom `av1_quantizer_to_qindex` (`av1/encoder/av1_quantize.c:1041`):
/// `quantizer_to_qindex[quantizer]`. `quantizer` must be in `0..=63` (libaom
/// range-checks `--cq-level` to `[0, 63]` at `av1/av1_cx_iface.c:802`).
///
/// # Panics
/// Panics if `quantizer` is outside `0..=63`.
#[must_use]
pub fn quantizer_to_qindex(quantizer: i32) -> i32 {
    QUANTIZER_TO_QINDEX[usize::try_from(quantizer).expect("quantizer must be in 0..=63")]
}

/// The `base_qindex` libaom assigns to a single KEY frame encoded under
/// constant-quality (`AOM_Q`, `rc_end_usage = AOM_Q`) with `--cq-level = cq`,
/// for both ALLINTRA (`usage = 2`) and GOOD (`usage = 0`) usage.
///
/// libaom converts the cq level to a qindex once, in the config layer:
/// `rc_cfg->cq_level = av1_quantizer_to_qindex(extra_cfg->cq_level)`
/// (`av1/av1_cx_iface.c:1256`). For a single KEY frame under `AOM_Q` the rate
/// controller takes `av1_rc_pick_q_and_bounds` →
/// `rc_pick_q_and_bounds_q_mode` → `get_intra_q_and_bounds`, whose
/// `frames_to_key <= 1 && mode == AOM_Q` branch (`ratectrl.c:1832-1837`) sets
/// `active_best_quality = active_worst_quality = cq_level` directly — no
/// kf_boost, no active-worst adjustment, no gf/arf offset. `av1_set_quantizer`
/// then stores `base_qindex = AOMMAX(delta_q_present_flag, q)`
/// (`av1_quantize.c:884`), which with delta-q off is just `q`. The final
/// `[best_quality, worst_quality]` clamp is `[0, 255]` for the default
/// `rc_min/max_quantizer` and so is inert. This holds for both ALLINTRA and
/// GOOD usage (the all-intra perceptual-deltaq path that would otherwise set
/// `base_qindex` at `allintra_vis.c:612` is gated off when `deltaq_mode == 0`).
/// Verified byte-identical against the real encoder across `cq 0..=63` ×
/// `usage {GOOD, ALLINTRA}` × `bd {8, 10, 12}` in `qindex_from_cq_diff.rs`.
///
/// # Panics
/// Panics if `cq` is outside `0..=63`.
#[must_use]
pub fn base_qindex_from_cq(cq: i32) -> i32 {
    quantizer_to_qindex(cq)
}

/// The `base_qindex` libaom assigns to a single KEY frame under `AOM_Q` with
/// `--cq-level = cq` AND the qindex clamp bounds `--min-q = min_q` /
/// `--max-q = max_q` (all three are 0..=63 quantizer levels). Extends
/// [`base_qindex_from_cq`] with the `[best_quality, worst_quality]` clamp that
/// `rc_pick_q_and_bounds_q_mode` applies (`ratectrl.c:2158-2161`,
/// `q = *bottom_index = clamp(active_best_quality, rc->best_quality,
/// rc->worst_quality)`): `rc->best_quality = av1_quantizer_to_qindex(min_q)` and
/// `rc->worst_quality = av1_quantizer_to_qindex(max_q)` (from
/// `rc_cfg->best/worst_allowed_q`, `encoder.c:1003-1004`; the `--min-q`/`--max-q`
/// CLI values are mapped through the same `quantizer_to_qindex` table in
/// `av1_cx_iface`). For the lone-KEY `AOM_Q` branch `active_best = active_worst
/// = cq_level` (`ratectrl.c:1832`), then `if (cq_level > 0) active_best =
/// AOMMAX(1, active_best)` (`ratectrl.c:2156`), and `av1_set_quantizer` stores
/// `base_qindex = q` (delta-q off). The default `min_q=0`/`max_q=63` map to
/// `[0, 255]`, making the clamp inert — so `base_qindex_from_cq_clamped(cq, 0,
/// 63) == base_qindex_from_cq(cq)`. Verified byte-identical against the real
/// encoder across a `(cq, min_q, max_q)` sweep in `min_max_q_diff.rs`.
///
/// # Panics
/// Panics if any of `cq`, `min_q`, `max_q` is outside `0..=63`.
#[must_use]
pub fn base_qindex_from_cq_clamped(cq: i32, min_q: i32, max_q: i32) -> i32 {
    let cq_level = quantizer_to_qindex(cq);
    let best_quality = quantizer_to_qindex(min_q);
    let worst_quality = quantizer_to_qindex(max_q);
    let active_best = if cq_level > 0 {
        cq_level.max(1)
    } else {
        cq_level
    };
    // libaom's `clamp` macro (`value < low ? low : value > high ? high : value`)
    // — NOT Rust's `clamp`, which panics when `low > high` (an out-of-order
    // min_q/max_q config; aomenc validates min_q <= max_q, but mirror C's macro
    // rather than panic).
    if active_best < best_quality {
        best_quality
    } else if active_best > worst_quality {
        worst_quality
    } else {
        active_best
    }
}

/// The `base_qindex` libaom assigns to a **low-delay P (inter leaf) frame** —
/// frame 1+ of a `--lag-in-frames=0 --end-usage=q` clip (the INTER-ENCODE
/// roadmap §3 simplest inter config) — with `--cq-level = cq`.
///
/// Under `AOM_Q` + `has_no_stats_stage` (`lag_in_frames == 0`), the rate
/// controller takes `av1_rc_pick_q_and_bounds` → `rc_pick_q_and_bounds` →
/// `rc_pick_q_and_bounds_q_mode` (`ratectrl.c:2133`) — NOT
/// `rc_pick_q_and_bounds_no_stats_cq`, which is dead code
/// (`#if USE_UNRESTRICTED_Q_IN_CQ_MODE`, and that macro is `0`,
/// `ratectrl.c:42`). Frame 1 of a `--limit=2` clip is a trailing `LF_UPDATE`
/// leaf of the trivial lag=0 GF group (`update_type[1] == LF_UPDATE`, via
/// `define_gf_group_pass0` with `max_layer_depth_allowed == 0`), so it is
/// neither KEY nor GF/ARF. `get_active_best_quality` (`ratectrl.c:2057`) hits
/// its `is_leaf_frame && rc_mode == AOM_Q` arm and returns `cq_level` verbatim
/// (`ratectrl.c:2092-2093`) — no kf_boost, no gf/arf q-offset. Then
/// `q = clamp(active_best_quality, best_quality, worst_quality)`
/// (`ratectrl.c:2160`) and `av1_set_quantizer` stores `base_qindex = q`
/// (`av1_quantize.c:884`, delta-q off). So the low-delay leaf P-frame
/// `base_qindex` is just `quantizer_to_qindex(cq)` — the SAME value as
/// [`base_qindex_from_cq`], but reached through the inter (non-KEY)
/// `get_active_best_quality` leaf path rather than the lone-KEY branch.
///
/// The KEY frame of the SAME multi-frame clip is DIFFERENT: it takes the
/// `get_intra_q_and_bounds` kf_boost (`frames_to_key > 1`), giving a LOWER
/// qindex (e.g. `cq 48 → KEY 80` vs `P 192`). So this is not a trivial identity
/// with the KEY qindex — it is specifically the leaf-P value.
///
/// Verified byte-identical against real `aomenc`'s coded frame-1 `base_qindex`
/// across the decodable cq sweep in
/// `aom-bench/tests/inter_rc_qindex_diff.rs`.
///
/// # Panics
/// Panics if `cq` is outside `0..=63`.
#[must_use]
pub fn base_qindex_lowdelay_p_from_cq(cq: i32) -> i32 {
    quantizer_to_qindex(cq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowdelay_p_qindex_is_the_cq_lookup() {
        // The low-delay leaf-P CQ path resolves to the same qindex as the direct
        // cq lookup (no kf/gf boost on the trailing LF_UPDATE leaf). The
        // byte-vs-aomenc proof lives in inter_rc_qindex_diff.rs.
        for cq in 0..=63 {
            assert_eq!(base_qindex_lowdelay_p_from_cq(cq), quantizer_to_qindex(cq));
        }
    }

    #[test]
    fn table_endpoints_and_length() {
        // The verified endpoints (av1_quantize.c:1033) and the two irregular
        // top steps that KB-1/KB-2 pin (quantizer 62 -> 249, 63 -> 255).
        assert_eq!(QUANTIZER_TO_QINDEX.len(), 64);
        assert_eq!(quantizer_to_qindex(0), 0);
        assert_eq!(quantizer_to_qindex(1), 4);
        assert_eq!(quantizer_to_qindex(61), 244);
        assert_eq!(quantizer_to_qindex(62), 249);
        assert_eq!(quantizer_to_qindex(63), 255);
        // Uniform +4 across the whole regular span [0, 61].
        for q in 0..=61 {
            assert_eq!(quantizer_to_qindex(q), q * 4);
        }
    }

    #[test]
    fn base_qindex_is_the_cq_lookup() {
        for cq in 0..=63 {
            assert_eq!(base_qindex_from_cq(cq), quantizer_to_qindex(cq));
        }
    }
}
