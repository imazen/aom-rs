//! INTER-ENCODE chunk 2: the inter mode / reference / MV-mode COST tables and
//! the per-frame inter CDF set they derive from.
//!
//! The §3 low-delay P codes `primary_ref_frame = PRIMARY_REF_NONE`
//! ([`crate::inter_frame::PRIMARY_REF_NONE`]), so its entropy context is
//! `av1_setup_past_independence` + `av1_default_coef_probs` — i.e. the
//! **DEFAULT** inter CDFs, with no reference-frame context to inherit. That
//! makes the inter cost tables derivable standalone from
//! `aom_dsp::entropy::default_cdfs`, with no `FrameContext` threading.
//!
//! C sources (all verified against `reference/libaom` v3.14.1):
//! - the cost fills: `av1_fill_mode_rates` (`av1/encoder/rd.c:220-285`) —
//!   `av1_cost_tokens_from_cdf` over `fc->{intra_inter,single_ref,newmv,zeromv,
//!   refmv,drl}_cdf`, gated on `!frame_is_intra_only(cm)`;
//! - the inter-mode rate: `cost_mv_ref` (`av1/encoder/rdopt.c:958`);
//! - the reference-frame rate: `estimate_ref_frame_costs`
//!   (`av1/encoder/rdopt.c:995`), single-reference arm.
//!
//! The context constants come from `av1/common/enums.h:482-516`.

use aom_dsp::entropy::default_cdfs::{
    DEFAULT_DRL, DEFAULT_INTRA_INTER, DEFAULT_NEWMV, DEFAULT_REFMV, DEFAULT_SINGLE_REF,
    DEFAULT_ZEROMV,
};
use aom_dsp::txb::cost_tokens_from_cdf;

/// `NEWMV_MODE_CONTEXTS` (enums.h:482).
pub const NEWMV_MODE_CONTEXTS: usize = 6;
/// `GLOBALMV_MODE_CONTEXTS` (enums.h:483).
pub const GLOBALMV_MODE_CONTEXTS: usize = 2;
/// `REFMV_MODE_CONTEXTS` (enums.h:484).
pub const REFMV_MODE_CONTEXTS: usize = 6;
/// `DRL_MODE_CONTEXTS` (enums.h:485).
pub const DRL_MODE_CONTEXTS: usize = 3;
/// `INTRA_INTER_CONTEXTS` (enums.h:514).
pub const INTRA_INTER_CONTEXTS: usize = 4;
/// `REF_CONTEXTS` (enums.h:516).
pub const REF_CONTEXTS: usize = 3;
/// `SINGLE_REFS - 1` — the six single-reference bit-tree slots p1..p6.
pub const SINGLE_REF_BITS: usize = 6;

/// `GLOBALMV_OFFSET` (enums.h:487).
pub const GLOBALMV_OFFSET: i32 = 3;
/// `REFMV_OFFSET` (enums.h:488).
pub const REFMV_OFFSET: i32 = 4;
/// `NEWMV_CTX_MASK` (enums.h:490).
pub const NEWMV_CTX_MASK: i32 = (1 << GLOBALMV_OFFSET) - 1;
/// `GLOBALMV_CTX_MASK` (enums.h:491).
pub const GLOBALMV_CTX_MASK: i32 = (1 << (REFMV_OFFSET - GLOBALMV_OFFSET)) - 1;
/// `REFMV_CTX_MASK` (enums.h:492).
pub const REFMV_CTX_MASK: i32 = (1 << (8 - REFMV_OFFSET)) - 1;

// `PREDICTION_MODE` inter values (enums.h:337-349).
/// `NEARESTMV`.
pub const NEARESTMV: i32 = 13;
/// `NEARMV`.
pub const NEARMV: i32 = 14;
/// `GLOBALMV`.
pub const GLOBALMV: i32 = 15;
/// `NEWMV`.
pub const NEWMV: i32 = 16;

// `MV_REFERENCE_FRAME` values (enums.h:601-610).
/// `INTRA_FRAME`.
pub const INTRA_FRAME: i32 = 0;
/// `LAST_FRAME`.
pub const LAST_FRAME: i32 = 1;
/// `NONE_FRAME`.
pub const NONE_FRAME: i32 = -1;

/// The per-frame INTER CDF set a P frame codes with. For
/// `primary_ref_frame == PRIMARY_REF_NONE` these start at the spec defaults
/// (`av1_setup_past_independence`); they then ADAPT as symbols are written
/// (`disable_cdf_update == 0`), so the pack threads this alongside the
/// `KfFrameContext` and the writers mutate it in place.
///
/// Kept in `aom-encode` (rather than extending `aom_dsp`'s `KfFrameContext`)
/// so the inter track stays additive over the shared entropy crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterFrameCdfs {
    /// `fc->intra_inter_cdf[INTRA_INTER_CONTEXTS]`.
    pub intra_inter: [[u16; 3]; INTRA_INTER_CONTEXTS],
    /// `fc->single_ref_cdf[REF_CONTEXTS][SINGLE_REFS-1]`.
    pub single_ref: [[[u16; 3]; SINGLE_REF_BITS]; REF_CONTEXTS],
    /// `fc->newmv_cdf[NEWMV_MODE_CONTEXTS]`.
    pub newmv: [[u16; 3]; NEWMV_MODE_CONTEXTS],
    /// `fc->zeromv_cdf[GLOBALMV_MODE_CONTEXTS]`.
    pub zeromv: [[u16; 3]; GLOBALMV_MODE_CONTEXTS],
    /// `fc->refmv_cdf[REFMV_MODE_CONTEXTS]`.
    pub refmv: [[u16; 3]; REFMV_MODE_CONTEXTS],
    /// `fc->drl_cdf[DRL_MODE_CONTEXTS]`.
    pub drl: [[u16; 3]; DRL_MODE_CONTEXTS],
}

impl Default for InterFrameCdfs {
    fn default() -> Self {
        Self::defaults()
    }
}

impl InterFrameCdfs {
    /// The spec default inter CDFs — what a `PRIMARY_REF_NONE` frame starts
    /// from (`av1_setup_past_independence` → `av1_default_...`).
    pub fn defaults() -> Self {
        InterFrameCdfs {
            intra_inter: DEFAULT_INTRA_INTER,
            single_ref: DEFAULT_SINGLE_REF,
            newmv: DEFAULT_NEWMV,
            zeromv: DEFAULT_ZEROMV,
            refmv: DEFAULT_REFMV,
            drl: DEFAULT_DRL,
        }
    }

    /// Gather the 16-slot CDF blob `aom_dsp::entropy::partition::write_ref_frames`
    /// consumes, with each single-reference slot pre-selected by its own
    /// prediction context (the caller's `av1_get_pred_context_single_ref_pN`).
    /// Compound slots (0..=9) are left at the p1 row: the §3 single-reference
    /// envelope never writes them (`reference_mode_is_select == false`,
    /// `is_compound == false`), so they are unreachable, and leaving them
    /// unwritten keeps the blob's compound half inert rather than plausible.
    ///
    /// Slot map (partition.rs): `10..=15` = single_ref p1,p2,p3,p4,p5,p6, whose
    /// `fc->single_ref_cdf[ctx][j]` second index `j` is `0..=5` in that order.
    pub fn single_ref_blob(&self, ctx: &SingleRefCtx) -> [[u16; 3]; 16] {
        let mut blob = [[0u16; 3]; 16];
        blob[10] = self.single_ref[ctx.p1 as usize][0];
        blob[11] = self.single_ref[ctx.p2 as usize][1];
        blob[12] = self.single_ref[ctx.p3 as usize][2];
        blob[13] = self.single_ref[ctx.p4 as usize][3];
        blob[14] = self.single_ref[ctx.p5 as usize][4];
        blob[15] = self.single_ref[ctx.p6 as usize][5];
        blob
    }

    /// Copy the (possibly adapted) single-reference rows back out of a blob
    /// that `write_ref_frames` mutated, so the frame context carries the
    /// adaptation forward to the next block. Only the rows on the taken path
    /// actually changed; copying all six is harmless because each slot maps to
    /// exactly one `[ctx][j]` cell.
    pub fn absorb_single_ref_blob(&mut self, ctx: &SingleRefCtx, blob: &[[u16; 3]; 16]) {
        self.single_ref[ctx.p1 as usize][0] = blob[10];
        self.single_ref[ctx.p2 as usize][1] = blob[11];
        self.single_ref[ctx.p3 as usize][2] = blob[12];
        self.single_ref[ctx.p4 as usize][3] = blob[13];
        self.single_ref[ctx.p5 as usize][4] = blob[14];
        self.single_ref[ctx.p6 as usize][5] = blob[15];
    }
}

/// The six single-reference prediction contexts
/// (`av1_get_pred_context_single_ref_p1..p6`, `av1/common/pred_common.c`),
/// gathered once per block from the neighbour reference counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SingleRefCtx {
    pub p1: i32,
    pub p2: i32,
    pub p3: i32,
    pub p4: i32,
    pub p5: i32,
    pub p6: i32,
}

impl SingleRefCtx {
    /// Derive all six contexts from `av1_collect_neighbors_ref_counts`' result
    /// (`aom_dsp::entropy::partition::collect_neighbors_ref_counts`) — the
    /// per-block gather C does at the top of `write_ref_frames` /
    /// `estimate_ref_frame_costs`.
    ///
    /// Each `av1_get_pred_context_single_ref_pN` is the already-validated
    /// count-grouping helper in the entropy crate; this is only the naming map:
    /// p1 = forward vs backward, p2 = BWDREF/ALTREF2 vs ALTREF, p3 = LAST/LAST2
    /// vs LAST3/GOLDEN, p4 = LAST vs LAST2, p5 = LAST3 vs GOLDEN,
    /// p6 = BWDREF vs ALTREF2.
    pub fn from_neighbor_ref_counts(rc: &[u8; 8]) -> Self {
        use aom_dsp::entropy::partition as p;
        SingleRefCtx {
            p1: p::single_ref_p1_context(rc),
            p2: p::pred_ctx_brfarf2_or_arf(rc),
            p3: p::pred_ctx_ll2_or_l3gld(rc),
            p4: p::pred_ctx_last_or_last2(rc),
            p5: p::pred_ctx_last3_or_gld(rc),
            p6: p::pred_ctx_brf_or_arf2(rc),
        }
    }
}

/// `ModeCosts`' inter half (`av1/encoder/block.h`), filled by
/// `av1_fill_mode_rates` (`rd.c:220-285`) from [`InterFrameCdfs`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterModeCosts {
    /// `mode_costs->intra_inter_cost[ctx][is_inter]`.
    pub intra_inter_cost: [[i32; 2]; INTRA_INTER_CONTEXTS],
    /// `mode_costs->single_ref_cost[ctx][j][bit]`.
    pub single_ref_cost: [[[i32; 2]; SINGLE_REF_BITS]; REF_CONTEXTS],
    /// `mode_costs->newmv_mode_cost[ctx][bit]`.
    pub newmv_mode_cost: [[i32; 2]; NEWMV_MODE_CONTEXTS],
    /// `mode_costs->zeromv_mode_cost[ctx][bit]`.
    pub zeromv_mode_cost: [[i32; 2]; GLOBALMV_MODE_CONTEXTS],
    /// `mode_costs->refmv_mode_cost[ctx][bit]`.
    pub refmv_mode_cost: [[i32; 2]; REFMV_MODE_CONTEXTS],
    /// `mode_costs->drl_mode_cost0[ctx][bit]`.
    pub drl_mode_cost0: [[i32; 2]; DRL_MODE_CONTEXTS],
}

/// `av1_fill_mode_rates`' inter arm (`rd.c:220-285`): every inter cost row is
/// `av1_cost_tokens_from_cdf(row, cdf, NULL)`.
pub fn derive_inter_mode_costs(cdfs: &InterFrameCdfs) -> InterModeCosts {
    let mut out = InterModeCosts {
        intra_inter_cost: [[0; 2]; INTRA_INTER_CONTEXTS],
        single_ref_cost: [[[0; 2]; SINGLE_REF_BITS]; REF_CONTEXTS],
        newmv_mode_cost: [[0; 2]; NEWMV_MODE_CONTEXTS],
        zeromv_mode_cost: [[0; 2]; GLOBALMV_MODE_CONTEXTS],
        refmv_mode_cost: [[0; 2]; REFMV_MODE_CONTEXTS],
        drl_mode_cost0: [[0; 2]; DRL_MODE_CONTEXTS],
    };
    for i in 0..INTRA_INTER_CONTEXTS {
        cost_tokens_from_cdf(&mut out.intra_inter_cost[i], &cdfs.intra_inter[i], None);
    }
    for i in 0..REF_CONTEXTS {
        for j in 0..SINGLE_REF_BITS {
            cost_tokens_from_cdf(&mut out.single_ref_cost[i][j], &cdfs.single_ref[i][j], None);
        }
    }
    for i in 0..NEWMV_MODE_CONTEXTS {
        cost_tokens_from_cdf(&mut out.newmv_mode_cost[i], &cdfs.newmv[i], None);
    }
    for i in 0..GLOBALMV_MODE_CONTEXTS {
        cost_tokens_from_cdf(&mut out.zeromv_mode_cost[i], &cdfs.zeromv[i], None);
    }
    for i in 0..REFMV_MODE_CONTEXTS {
        cost_tokens_from_cdf(&mut out.refmv_mode_cost[i], &cdfs.refmv[i], None);
    }
    for i in 0..DRL_MODE_CONTEXTS {
        cost_tokens_from_cdf(&mut out.drl_mode_cost0[i], &cdfs.drl[i], None);
    }
    out
}

/// `cost_mv_ref` (`av1/encoder/rdopt.c:958`), single-reference arm: the rate of
/// the inter-mode symbol cascade `write_inter_mode` codes
/// (newmv → zeromv → refmv), each on its own slice of `mode_context`.
///
/// Compound modes are out of the §3 envelope and hit the `debug_assert`.
pub fn cost_mv_ref(costs: &InterModeCosts, mode: i32, mode_context: i32) -> i32 {
    debug_assert!(
        (NEARESTMV..=NEWMV).contains(&mode),
        "cost_mv_ref: single-reference inter modes only (got {mode})"
    );
    let newmv_ctx = (mode_context & NEWMV_CTX_MASK) as usize;
    if mode == NEWMV {
        return costs.newmv_mode_cost[newmv_ctx][0];
    }
    let mut cost = costs.newmv_mode_cost[newmv_ctx][1];
    let globalmv_ctx = ((mode_context >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK) as usize;
    if mode == GLOBALMV {
        return cost + costs.zeromv_mode_cost[globalmv_ctx][0];
    }
    cost += costs.zeromv_mode_cost[globalmv_ctx][1];
    let refmv_ctx = ((mode_context >> REFMV_OFFSET) & REFMV_CTX_MASK) as usize;
    cost += costs.refmv_mode_cost[refmv_ctx][i32::from(mode != NEARESTMV) as usize];
    cost
}

/// `estimate_ref_frame_costs`' single-reference arm (`rdopt.c:995`) reduced to
/// `ref_costs_single[LAST_FRAME]`: the `is_inter` symbol plus the three
/// single-reference bit-tree symbols LAST takes (p1 level-0 forward, p3
/// level-1 last/last2 group, p4 level-2 last).
///
/// The §3 two-frame low-delay clip resolves every reference slot to frame 0, so
/// LAST is the only reference the encoder can pick — the other six
/// `ref_costs_single` entries are unreachable and deliberately not computed.
pub fn ref_cost_single_last(
    costs: &InterModeCosts,
    intra_inter_ctx: i32,
    ctx: &SingleRefCtx,
) -> i32 {
    costs.intra_inter_cost[intra_inter_ctx as usize][1]
        + costs.single_ref_cost[ctx.p1 as usize][0][0]
        + costs.single_ref_cost[ctx.p3 as usize][2][0]
        + costs.single_ref_cost[ctx.p4 as usize][3][0]
}

/// `ref_costs_single[INTRA_FRAME]` (`rdopt.c:1009`): the cost of signalling an
/// INTRA block inside an inter frame — the `is_inter == 0` symbol. The inter RD
/// arm competes against the intra winner, and in an inter frame the intra
/// winner must also pay this.
pub fn ref_cost_intra_in_inter(costs: &InterModeCosts, intra_inter_ctx: i32) -> i32 {
    costs.intra_inter_cost[intra_inter_ctx as usize][0]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The port's default inter CDF tables must be the AOM_ICDF-inverted form
    /// of libaom's `default_{newmv,zeromv,refmv,drl}_cdf`
    /// (`av1/common/entropymode.c:438-451`). The C literals are transcribed
    /// here so a mistranscription in `default_cdfs.rs` fails loudly rather than
    /// silently shifting every inter mode cost.
    ///
    /// `AOM_CDF2(a)` expands to `{ AOM_ICDF(a), AOM_ICDF(32768), 0 }` and
    /// `AOM_ICDF(x) == 32768 - x`, so each row is `[32768 - a, 0, 0]`.
    #[test]
    fn default_inter_cdfs_match_libaom() {
        const C_NEWMV: [u16; 6] = [24035, 16630, 15339, 8386, 12222, 4676];
        const C_ZEROMV: [u16; 2] = [2175, 1054];
        const C_REFMV: [u16; 6] = [23974, 24188, 17848, 28622, 24312, 19923];
        const C_DRL: [u16; 3] = [13104, 24560, 18945];
        let cdfs = InterFrameCdfs::defaults();
        for (i, &a) in C_NEWMV.iter().enumerate() {
            assert_eq!(cdfs.newmv[i], [32768 - a, 0, 0], "newmv row {i}");
        }
        for (i, &a) in C_ZEROMV.iter().enumerate() {
            assert_eq!(cdfs.zeromv[i], [32768 - a, 0, 0], "zeromv row {i}");
        }
        for (i, &a) in C_REFMV.iter().enumerate() {
            assert_eq!(cdfs.refmv[i], [32768 - a, 0, 0], "refmv row {i}");
        }
        for (i, &a) in C_DRL.iter().enumerate() {
            assert_eq!(cdfs.drl[i], [32768 - a, 0, 0], "drl row {i}");
        }
    }

    /// Cost polarity follows the INVERSE-CDF storage convention: a row
    /// `[32768 - a, 0, 0]` gives symbol 0 the probability `a`, so the cheaper
    /// symbol is the one C's `AOM_CDF2` literal made likely.
    ///
    /// Both `default_zeromv_cdf` literals are tiny (2175, 1054 of 32768) ⇒
    /// GLOBALMV (symbol 0) is the IMPROBABLE branch and must cost strictly
    /// more than "not GLOBALMV". This is the entropy-side cross-check on the
    /// measured ground truth that `aomenc` codes the zero-MV P as NEARESTMV
    /// rather than GLOBALMV.
    #[test]
    fn default_inter_costs_are_well_formed() {
        let cdfs = InterFrameCdfs::defaults();
        let c = derive_inter_mode_costs(&cdfs);
        for i in 0..INTRA_INTER_CONTEXTS {
            assert!(
                c.intra_inter_cost[i][0] > 0 && c.intra_inter_cost[i][1] > 0,
                "intra_inter ctx {i} cost must be positive"
            );
        }
        for i in 0..GLOBALMV_MODE_CONTEXTS {
            assert!(
                c.zeromv_mode_cost[i][0] > c.zeromv_mode_cost[i][1],
                "zeromv ctx {i}: GLOBALMV is the improbable branch, so it must \
                 cost more than not-GLOBALMV (got {:?})",
                c.zeromv_mode_cost[i]
            );
        }
        // The same convention on newmv: every default row's literal is < 16384
        // for ctx 3..5 (8386/12222/4676) ⇒ symbol 0 (== NEWMV) is improbable
        // there and must cost more than symbol 1.
        for i in [3usize, 5] {
            assert!(
                c.newmv_mode_cost[i][0] > c.newmv_mode_cost[i][1],
                "newmv ctx {i}: symbol 0 is improbable (got {:?})",
                c.newmv_mode_cost[i]
            );
        }
    }

    /// `cost_mv_ref` must decompose exactly as C's cascade — NEARESTMV pays
    /// newmv[1] + zeromv[1] + refmv[0], and the three context slices must be
    /// extracted with the enums.h masks.
    #[test]
    fn cost_mv_ref_matches_c_cascade() {
        let c = derive_inter_mode_costs(&InterFrameCdfs::defaults());
        // A mode_context exercising all three slices distinctly.
        let mode_ctx: i32 = 0b0101_1010;
        let newmv_ctx = (mode_ctx & NEWMV_CTX_MASK) as usize;
        let gmv_ctx = ((mode_ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK) as usize;
        let refmv_ctx = ((mode_ctx >> REFMV_OFFSET) & REFMV_CTX_MASK) as usize;

        assert_eq!(
            cost_mv_ref(&c, NEWMV, mode_ctx),
            c.newmv_mode_cost[newmv_ctx][0]
        );
        assert_eq!(
            cost_mv_ref(&c, GLOBALMV, mode_ctx),
            c.newmv_mode_cost[newmv_ctx][1] + c.zeromv_mode_cost[gmv_ctx][0]
        );
        assert_eq!(
            cost_mv_ref(&c, NEARESTMV, mode_ctx),
            c.newmv_mode_cost[newmv_ctx][1]
                + c.zeromv_mode_cost[gmv_ctx][1]
                + c.refmv_mode_cost[refmv_ctx][0]
        );
        assert_eq!(
            cost_mv_ref(&c, NEARMV, mode_ctx),
            c.newmv_mode_cost[newmv_ctx][1]
                + c.zeromv_mode_cost[gmv_ctx][1]
                + c.refmv_mode_cost[refmv_ctx][1]
        );
        // Anti-vacuity: NEARESTMV and NEARMV must differ (they take opposite
        // refmv bits), so the cascade is not collapsing to a constant.
        assert_ne!(
            cost_mv_ref(&c, NEARESTMV, mode_ctx),
            cost_mv_ref(&c, NEARMV, mode_ctx)
        );
    }

    /// The single-reference blob must round-trip through
    /// `single_ref_blob` / `absorb_single_ref_blob` at the slots
    /// `write_ref_frames` uses, and place each `[ctx][j]` in its documented slot.
    #[test]
    fn single_ref_blob_slots_round_trip() {
        let mut cdfs = InterFrameCdfs::defaults();
        let ctx = SingleRefCtx {
            p1: 0,
            p2: 1,
            p3: 2,
            p4: 1,
            p5: 0,
            p6: 2,
        };
        let blob = cdfs.single_ref_blob(&ctx);
        assert_eq!(blob[10], cdfs.single_ref[0][0], "slot 10 == p1 row j=0");
        assert_eq!(blob[12], cdfs.single_ref[2][2], "slot 12 == p3 row j=2");
        assert_eq!(blob[13], cdfs.single_ref[1][3], "slot 13 == p4 row j=3");
        // Compound slots stay inert.
        assert_eq!(blob[0], [0u16; 3]);

        // Mutate as an adapting writer would, then absorb.
        let mut blob2 = blob;
        blob2[10] = [111, 0, 1];
        blob2[13] = [222, 0, 1];
        cdfs.absorb_single_ref_blob(&ctx, &blob2);
        assert_eq!(cdfs.single_ref[0][0], [111, 0, 1]);
        assert_eq!(cdfs.single_ref[1][3], [222, 0, 1]);
    }
}
