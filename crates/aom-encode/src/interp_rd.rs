//! INTER-ENCODE: the switchable interp-filter RATE model for inter leaves —
//! the curvefit model-rd + the reduced §3 zero-MV filter pick.
//!
//! ## Why an ENCODE-side filter model exists at all
//!
//! During encoding the frame's `interp_filter` is SWITCHABLE — the coded
//! non-switchable header value (e.g. EIGHTTAP_REGULAR at cq60) is a POST-encode
//! selection — so every inter candidate pays the switchable-filter signalling
//! rate `rs = av1_get_switchable_rate(..)` in its RD (`rd.c:1234`,
//! `SWITCHABLE_INTERP_RATE_FACTOR * switchable_interp_costs[ctx][filter]`),
//! even though the §3 frame never writes a per-block filter symbol. The filter
//! itself comes from `av1_interpolation_filter_search` (interp_search.c:674),
//! whose accept step gives MULTITAP_SHARP a `mul = 90` (10%) RD discount when
//! `sf->interp_sf.use_more_sharp_interp` is set (speed_features.c:1139 — the
//! GOOD-mode base at EVERY speed, non-boosted frames). Omitting this model
//! under-prices inter leaves by `rs` and flips partition near-ties (the
//! measured 64x128 cropped-SB128 VERT/SPLIT divergence, KB-16).
//!
//! ## The reduced zero-MV search
//!
//! At a full-pel MV every filter family's phase-0 kernel is the identity, so
//! all predictions are IDENTICAL and `calc_interp_skip_pred_flag` marks both
//! directions skippable (`skip_pred == default_interp_skip_flags`) — C's
//! `interpolation_filter_rd` then REUSES the REGULAR evaluation's model
//! rate/dist for SMOOTH and SHARP, and the whole search collapses to a pure
//! rate compare over the shared curvefit model stats:
//!
//! 1. `*rd = RDCOST(rdmult, rs_REG + model_rate, model_dist)` (the REGULAR
//!    baseline from `av1_interpolation_filter_search`'s preamble);
//! 2. for SMOOTH then SHARP (`find_best_non_dual_interp_filter`'s non-dual
//!    loop; `skip_sharp_interp_filter_search = 0` at speed 0 so both run):
//!    reject early when `RDCOST(rdmult, rs_f, 0) * mul / 100 > *rd`, else
//!    accept when `RDCOST(rdmult, rs_f + model_rate, model_dist) * mul / 100
//!    < *rd` (mul = 90 iff SHARP && use_more_sharp_interp, else 100).
//!
//! The model stats are `model_rd_for_sb_with_curvfit` (model_rd.h:214):
//! per-plane sse over the padded, frame-clipped plane block →
//! `model_rd_with_curvfit` (`MODELRD_TYPE_INTERP_FILTER = MODELRD_CURVFIT`,
//! model_rd.h:31) → summed. The curvefit core `av1_model_rd_curvfit`
//! (rd.c:1064) is a REAL EXPORTED C function — differential-locked in
//! `crates/aom-encode/tests/curvfit_diff.rs`.
//!
//! ## Scope
//!
//! Zero-MV (full-pel identity) candidates only — the ONLY regime the inter RD
//! arm codes today. A subpel-MV rung must evaluate real per-filter predictions
//! (`interp_model_rd_eval` per filter) instead of reusing one model.

use crate::curvfit_tables::{BSIZE_CURVFIT_MODEL_CAT, INTERP_DGRID_CURV, INTERP_RGRID_CURV};
use aom_dsp::entropy::default_cdfs::DEFAULT_SWITCHABLE_INTERP;
use aom_dsp::txb::cost_tokens_from_cdf;

/// `SWITCHABLE_INTERP_RATE_FACTOR` (av1/encoder/rd.h:58).
pub const SWITCHABLE_INTERP_RATE_FACTOR: i32 = 1;

/// `InterpFilter` values (av1/common/filter.h).
pub const EIGHTTAP_REGULAR: usize = 0;
pub const EIGHTTAP_SMOOTH: usize = 1;
pub const MULTITAP_SHARP: usize = 2;

/// `mode_costs->switchable_interp_costs[SWITCHABLE_FILTER_CONTEXTS][SWITCHABLE_FILTERS]`
/// — `av1_fill_mode_rates`' `av1_cost_tokens_from_cdf` over the (adapting)
/// `switchable_interp_cdf`. The §3 frame writes no filter symbols, so the CDF
/// never adapts and the DEFAULT-derived table is the per-SB refresh fixpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwitchableInterpCosts(pub [[i32; 3]; 16]);

impl SwitchableInterpCosts {
    pub fn from_default_cdfs() -> Self {
        let mut out = [[0i32; 3]; 16];
        for (row, cdf) in out.iter_mut().zip(DEFAULT_SWITCHABLE_INTERP.iter()) {
            cost_tokens_from_cdf(row, cdf, None);
        }
        SwitchableInterpCosts(out)
    }
}

/// `interp_cubic` (rd.c:945) — Catmull-Rom cubic over 4 consecutive grid
/// points, exact C op order.
fn interp_cubic(p: &[f64], x: f64) -> f64 {
    p[1] + 0.5
        * x
        * (p[2] - p[0]
            + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3]
                + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))
}

/// `sse_norm_curvfit_model_cat_lookup` (rd.c:968).
fn sse_norm_cat(sse_norm: f64) -> usize {
    usize::from(sse_norm > 16.0)
}

/// `av1_model_rd_curvfit` (rd.c:1064) — bit-exact vs the exported C fn
/// (`curvfit_diff.rs`). Returns `(rate_f, dist_by_sse_norm_f)`.
pub fn av1_model_rd_curvfit(bsize: usize, sse_norm: f64, xqr: f64) -> (f64, f64) {
    const X_START: f64 = -15.5;
    const X_STEP: f64 = 0.5;
    const X_END: f64 = 16.5;
    const EPSILON: f64 = 1e-6;
    let rcat = BSIZE_CURVFIT_MODEL_CAT[bsize];
    let dcat = sse_norm_cat(sse_norm);

    let xqr = xqr.max(X_START + X_STEP + EPSILON);
    let xqr = xqr.min(X_END - X_STEP - EPSILON);
    let x = (xqr - X_START) / X_STEP;
    let xi = x.floor() as i32;
    let xo = x - f64::from(xi);
    debug_assert!(xi > 0);
    let xi = xi as usize;

    let rate_f = interp_cubic(&INTERP_RGRID_CURV[rcat][xi - 1..], xo);
    let dist_f = interp_cubic(&INTERP_DGRID_CURV[dcat][xi - 1..], xo);
    (rate_f, dist_f)
}

/// `model_rd_with_curvfit` (model_rd.h:117): one plane's model (rate, dist)
/// from its RAW sse. `qstep_raw` is the plane's AC dequant (`dequant_QTX[1]`);
/// the `>> (bd8 ? 3 : bd-5)` fold is the caller-visible `dequant_shift`.
pub fn model_rd_with_curvfit(
    plane_bsize: usize,
    sse: i64,
    num_samples: i32,
    qstep_raw: i32,
    bd: u8,
    rdmult: i32,
) -> (i32, i64) {
    let dequant_shift = if bd > 8 { i32::from(bd) - 5 } else { 3 };
    let qstep = (qstep_raw >> dequant_shift).max(1);
    if sse == 0 {
        return (0, 0);
    }
    let sse_norm = sse as f64 / f64::from(num_samples);
    let qstepsqr = f64::from(qstep) * f64::from(qstep);
    let xqr = (sse_norm / qstepsqr).log2();
    let (rate_f, dist_by_sse_norm_f) = av1_model_rd_curvfit(plane_bsize, sse_norm, xqr);

    let dist_f = dist_by_sse_norm_f * sse_norm;
    let mut rate_i = ((rate_f * f64::from(num_samples)).max(0.0) + 0.5) as i32;
    let mut dist_i = ((dist_f * f64::from(num_samples)).max(0.0) + 0.5) as i64;

    // Check if skip is better.
    if rate_i == 0 {
        dist_i = sse << 4;
    } else if crate::rd::rdcost(rdmult, rate_i, dist_i) >= crate::rd::rdcost(rdmult, 0, sse << 4) {
        rate_i = 0;
        dist_i = sse << 4;
    }
    (rate_i, dist_i)
}

/// The reduced §3 zero-MV filter pick (module docs): given the SHARED model
/// stats (identical predictions across filters), reproduce
/// `av1_interpolation_filter_search` + `find_best_non_dual_interp_filter`'s
/// accept chain and return `(winner_filter, rs)`.
///
/// `ctx` = `av1_get_pred_context_switchable_interp(xd, 0)` (non-dual reads
/// direction 0 only).
pub fn pick_interp_filter_zero_mv(
    costs: &SwitchableInterpCosts,
    ctx: usize,
    use_more_sharp_interp: bool,
    rdmult: i32,
    model_rate: i32,
    model_dist: i64,
) -> (usize, i32) {
    let rs_of = |f: usize| SWITCHABLE_INTERP_RATE_FACTOR * costs.0[ctx][f];
    let mut best_filter = EIGHTTAP_REGULAR;
    let mut best_rs = rs_of(EIGHTTAP_REGULAR);
    // The REGULAR baseline (av1_interpolation_filter_search preamble).
    let mut best_rd = crate::rd::rdcost(rdmult, best_rs + model_rate, model_dist);

    for f in [EIGHTTAP_SMOOTH, MULTITAP_SHARP] {
        let mul: i64 = if f == MULTITAP_SHARP && use_more_sharp_interp {
            90
        } else {
            100
        };
        let tmp_rs = rs_of(f);
        // interpolation_filter_rd's head early-out.
        let min_rd = crate::rd::rdcost(rdmult, tmp_rs, 0);
        if min_rd * mul / 100 > best_rd {
            continue;
        }
        // skip_pred == default flags ⇒ the model stats are REUSED verbatim.
        let tmp_rd = crate::rd::rdcost(rdmult, tmp_rs + model_rate, model_dist);
        if tmp_rd * mul / 100 < best_rd {
            best_rd = tmp_rd;
            best_rs = tmp_rs;
            best_filter = f;
        }
    }
    (best_filter, best_rs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contexts whose neighbours used REGULAR (ctx 0) or gave no information
    /// (ctx 3 — the no-neighbour context every §3 SB-first block lands on)
    /// must make REGULAR the cheapest filter; a SHARP-neighbour context (ctx
    /// 2) must favor SHARP. The entropy-side sanity that the ctx rows encode
    /// neighbour filters — and that "REGULAR wins unless the SHARP discount
    /// fires" holds exactly where the measured §3 blocks read (ctx 0/3).
    #[test]
    fn default_costs_follow_neighbour_contexts() {
        let c = SwitchableInterpCosts::from_default_cdfs();
        for ctx in [0usize, 3] {
            assert!(c.0[ctx][0] <= c.0[ctx][1], "ctx {ctx} smooth");
            assert!(c.0[ctx][0] <= c.0[ctx][2], "ctx {ctx} sharp");
        }
        assert!(
            c.0[2][2] < c.0[2][0],
            "a SHARP-neighbour context must make SHARP cheaper than REGULAR"
        );
        // The two measured §3 anchor values (instrumented C): ctx 3 REGULAR
        // 109, ctx 3 SHARP 3931.
        assert_eq!(c.0[3][0], 109);
        assert_eq!(c.0[3][2], 3931);
    }

    /// With the sharp discount OFF, identical model stats keep REGULAR; with
    /// it ON and a dist-heavy model, SHARP takes over exactly when
    /// `0.9 * rd_sharp < rd_regular` — the measured 64x128 mechanism.
    #[test]
    fn sharp_discount_flips_dist_heavy_blocks() {
        let c = SwitchableInterpCosts::from_default_cdfs();
        let ctx = 3usize; // no-neighbour context (measured on the §3 frames)
        let rdmult = 2_784_108; // C's x->rdmult at qindex 240 (measured)
        let (rate, dist) = (2000i32, 2_000_000i64);
        let (f_off, rs_off) = pick_interp_filter_zero_mv(&c, ctx, false, rdmult, rate, dist);
        assert_eq!(f_off, EIGHTTAP_REGULAR);
        assert_eq!(rs_off, SWITCHABLE_INTERP_RATE_FACTOR * c.0[ctx][0]);
        let (f_on, rs_on) = pick_interp_filter_zero_mv(&c, ctx, true, rdmult, rate, dist);
        assert_eq!(
            f_on, MULTITAP_SHARP,
            "the mul=90 discount must flip a dist-heavy block onto SHARP"
        );
        assert_eq!(rs_on, SWITCHABLE_INTERP_RATE_FACTOR * c.0[ctx][2]);
        // And a rate-dominated (tiny-dist) block stays REGULAR even with the
        // discount — the 64x64 behaviour.
        let (f_small, _) = pick_interp_filter_zero_mv(&c, ctx, true, rdmult, 10, 100);
        assert_eq!(f_small, EIGHTTAP_REGULAR);
    }
}
