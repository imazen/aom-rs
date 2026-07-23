//! INTER-ENCODE chunk 2, sub-step 2f: the single-reference inter mode RD arm.
//!
//! C: `av1_rd_pick_inter_mode_sb` (`av1/encoder/rdopt.c` ~6180) →
//! `handle_inter_mode` (`:3063`), reduced to the INTER-ENCODE-ROADMAP §3
//! envelope AND to the SKIP arm.
//!
//! # SCOPE — what this does and does NOT do
//!
//! Implemented (rung 1 of the roadmap ladder — the zero-MV translational P):
//! - the three **search-free** single-reference modes `NEARESTMV`, `NEARMV`,
//!   `GLOBALMV`, whose MVs come from [`aom_dsp::entropy::dv_ref::find_inter_mv_refs`]
//!   rather than a motion search;
//! - the **SKIP** arm only (`skip_txfm = 1`, zero residual coded), gated on
//!   `predict_skip_txfm` — C's own fast-path predicate;
//! - `SIMPLE_TRANSLATION` motion mode, `LAST_FRAME` single reference, no
//!   compound, no interintra, no DRL beyond index 0.
//!
//! NOT implemented here (each a named later rung / chunk):
//! - **`NEWMV`** — the motion search (`inter_me::single_motion_search`) is
//!   ported and real-C-locked, but wiring it needs the MV coder in the pack and
//!   the DRL loop; rung 2.
//! - **the COEFF arm** — a nonzero-residual inter block needs
//!   `av1_txfm_search`'s tx path. When the block does not predict-skip this
//!   function returns `None` (declines the candidate) rather than emitting a
//!   block it cannot code exactly. A declined candidate means intra wins the
//!   leaf, which makes the byte gate FAIL LOUDLY rather than silently produce a
//!   wrong bitstream.
//! - compound / OBMC / warp / interintra / global-motion estimation — all
//!   CLI-disabled in the §3 config and structurally absent from the search.
//!
//! # Distortion convention
//!
//! `set_skip_txfm` (`tx_search.c:245-281`): the skip arm's distortion is
//! `ROUND_POWER_OF_TWO(sse, 2*(bd-8)) << 4` — the same scaling the ported
//! intrabc skip arm uses (`intrabc_search.rs`), so the inter RD is directly
//! comparable against the assembled intra RD at the step-6 competition site.

use crate::inter_costs::{
    cost_mv_ref, ref_cost_single_last, InterModeCosts, SingleRefCtx, GLOBALMV, LAST_FRAME,
    NEARESTMV, NEARMV,
};
use crate::inter_frame::RefFrame;

/// `SIMPLE_TRANSLATION` (enums.h:398) — the only motion mode the §3 config
/// allows (`switchable_motion_mode = 0`, obmc/warp CLI-disabled).
pub const SIMPLE_TRANSLATION: i32 = 0;

/// Everything the leaf inter search needs, mirroring
/// [`crate::intrabc_search::IntrabcLeafArgs`]'s shape. The recon planes are NOT
/// here: the inter predictor reads the REFERENCE frame, not the current recon.
pub struct InterLeafArgs<'a> {
    // --- geometry ---
    pub bsize: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub mi_rows: i32,
    pub mi_cols: i32,
    pub monochrome: bool,
    pub is_chroma_ref: bool,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: u8,

    // --- source pixels (the residual base) ---
    pub stride: usize,
    pub src_y: &'a [u16],
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    /// Block origin offsets into the strided source planes.
    pub off_y: usize,
    pub off_uv: usize,

    // --- the reference frame (frame 0's filtered, border-extended recon) ---
    pub ref_frame: &'a RefFrame,

    // --- ref-MV candidates (`find_inter_mv_refs`, already byte-exact vs C) ---
    /// `mode_context` for the inter-mode CDF/cost slices.
    pub mode_context: i32,
    /// `NEARESTMV`'s motion vector (1/8-pel).
    pub nearest_mv: (i32, i32),
    /// `NEARMV`'s motion vector (1/8-pel).
    pub near_mv: (i32, i32),
    /// `GLOBALMV`'s motion vector — identity global motion ⇒ `(0, 0)` in the §3
    /// config (`--enable-global-motion=0`).
    pub global_mv: (i32, i32),
    /// `ref_mv_count` from the scan — a DRL index > 0 is only codeable for
    /// NEARMV/NEWMV, which rung 1 does not use, so this only gates NEARMV's
    /// availability.
    pub ref_mv_count: i32,

    // --- RD inputs ---
    pub rdmult: i32,
    pub qindex: i32,
    pub reduced_tx_set_used: bool,
    pub costs: &'a InterModeCosts,
    pub skip_costs: &'a [[i32; 2]; 3],
    /// `av1_get_skip_txfm_context(xd)`.
    pub skip_ctx: usize,
    /// `av1_get_intra_inter_context(xd)`.
    pub intra_inter_ctx: i32,
    /// The six `av1_get_pred_context_single_ref_pN` contexts.
    pub single_ref_ctx: SingleRefCtx,

    // --- the switchable interp-filter RATE model (crate::interp_rd) ---
    /// `mode_costs->switchable_interp_costs`.
    pub interp_costs: &'a crate::interp_rd::SwitchableInterpCosts,
    /// `av1_get_pred_context_switchable_interp(xd, 0)` (non-dual: dir 0).
    pub interp_ctx: usize,
    /// `sf->interp_sf.use_more_sharp_interp` (GOOD base: `!boosted`).
    pub use_more_sharp_interp: bool,
    /// The AC dequant `dequant_QTX[1]` (raw, pre `dequant_shift`) — identical
    /// across planes in the §3 zero-delta-q header; the curvefit qstep input.
    pub dequant_ac: i32,
}

/// The winning inter candidate — C's `best_mbmi` reduced to the fields the §3
/// skip-only envelope actually codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterBest {
    /// `mbmi->mode` (`NEARESTMV` / `NEARMV` / `GLOBALMV`).
    pub mode: i32,
    /// `mbmi->interp_filters` (non-dual: one filter both directions) — the
    /// SEARCH-time winner of the switchable filter pick. Never coded in the §3
    /// bitstream (the frame filter is non-switchable post-hoc) but stamped on
    /// the neighbour grid: it feeds later blocks'
    /// `av1_get_pred_context_switchable_interp`.
    pub interp_filter: u8,
    /// `mbmi->ref_frame[0]` — always `LAST_FRAME` here.
    pub ref_frame0: i8,
    /// `mbmi->mv[0]`, 1/8-pel.
    pub mv_row: i32,
    pub mv_col: i32,
    /// The `mode_context` the rate was priced against (frozen for the pack).
    pub mode_context: i32,
    /// `mbmi->skip_txfm` — always true in this scope.
    pub skip_txfm: bool,
    /// `rd_stats->rate` = ref cost + mode cost + `skip_txfm_cost[ctx][1]`.
    pub rate: i32,
    /// `rd_stats->dist` = the scaled prediction SSE.
    pub dist: i64,
    /// `RDCOST(rdmult, rate, dist)`.
    pub rdcost: i64,
}

/// `RDCOST` (`av1/encoder/rd.h`): `((rdmult * rate) >> 9) + (dist << 4)` in the
/// port's shared form.
fn rdcost(rdmult: i32, rate: i32, dist: i64) -> i64 {
    crate::rd::rdcost(rdmult, rate, dist)
}

/// Copy the co-located reference block into `dst` — the degenerate zero-MV
/// inter predictor. A full-pel MV of (0,0) makes
/// `av1_enc_build_inter_predictor` a plain block copy: no subpel filter taps
/// and no interior edge extension, so this is exact rather than an
/// approximation of the MC path.
///
/// Reads past the reference CROP are coordinate-clamped (edge replication) —
/// exactly what C's border-extended reference buffer contains there
/// (`aom_extend_frame_borders` replicates edge pixels), so a padded sub-8x8
/// chroma tail or an edge block reads the same values C's MC reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_colocated(
    refp: &[u16],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    dst: &mut [u16],
    // `dst`'s row stride — the FULL block width, which differs from `w`
    // whenever the block is partly outside the frame.
    dst_stride: usize,
) {
    debug_assert!(ref_w > 0 && ref_h > 0, "empty reference plane");
    for r in 0..h {
        let sy = (y + r).min(ref_h - 1);
        let s = sy * ref_stride;
        let d = r * dst_stride;
        for c in 0..w {
            dst[d + c] = refp[s + (x + c).min(ref_w - 1)];
        }
    }
}

/// Sum of squared error between a strided source region and a tight prediction,
/// clipped to the frame-VISIBLE part of the block (C accumulates distortion
/// only over visible pixels; the invisible tail of an edge block carries no
/// source).
fn sse_visible(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    pred: &[u16],
    w: usize,
    vis_w: usize,
    vis_h: usize,
) -> i64 {
    let mut acc: i64 = 0;
    for r in 0..vis_h {
        for c in 0..vis_w {
            let d = i32::from(src[src_off + r * src_stride + c]) - i32::from(pred[r * w + c]);
            acc += i64::from(d) * i64::from(d);
        }
    }
    acc
}

/// Residual (src − pred) over the FULL block, invisible tail zero-filled —
/// `av1_subtract_plane`'s shape, which `predict_skip_txfm` consumes.
fn residual_full(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    pred: &[u16],
    w: usize,
    h: usize,
    vis_w: usize,
    vis_h: usize,
) -> Vec<i16> {
    let mut out = vec![0i16; w * h];
    for r in 0..vis_h {
        for c in 0..vis_w {
            out[r * w + c] = (i32::from(src[src_off + r * src_stride + c])
                - i32::from(pred[r * w + c])) as i16;
        }
    }
    out
}

/// `set_skip_txfm`'s distortion scaling (`tx_search.c:245-281`):
/// `ROUND_POWER_OF_TWO(sse, 2*(bd-8)) << 4`.
fn scale_dist(sse: i64, bd: u8) -> i64 {
    let scaled = if bd > 8 {
        let sh = 2 * (u32::from(bd) - 8);
        (sse + (1 << (sh - 1))) >> sh
    } else {
        sse
    };
    scaled << 4
}

/// `av1_rd_pick_inter_mode_sb` reduced to the §3 single-reference, SKIP-only,
/// search-free-mode envelope (see the module docs for exactly what is in and
/// out of scope).
///
/// Returns the best inter candidate, or `None` when no candidate is codeable in
/// this scope — either the block does not predict-skip (the COEFF arm is a
/// later rung) or the reference read would leave the plane. `None` means the
/// caller keeps its intra winner.
pub fn rd_pick_inter_mode_sb(a: &InterLeafArgs, best_rd_in: i64) -> Option<InterBest> {
    let bw = crate::tx_search::BLK_W_B[a.bsize];
    let bh = crate::tx_search::BLK_H_B[a.bsize];
    let mi_w = crate::tx_search::MI_SIZE_WIDE_B[a.bsize];
    let mi_h = crate::tx_search::MI_SIZE_HIGH_B[a.bsize];

    // Frame-visible extent of this block (edge blocks are partly outside).
    let vis_mi_w = mi_w.min((a.mi_cols - a.mi_col).max(0) as usize);
    let vis_mi_h = mi_h.min((a.mi_rows - a.mi_row).max(0) as usize);
    let vis_w = (vis_mi_w * 4).min(bw);
    let vis_h = (vis_mi_h * 4).min(bh);
    if vis_w == 0 || vis_h == 0 {
        return None;
    }

    // The reference cost is mode-independent (single reference, LAST only).
    let ref_rate = ref_cost_single_last(a.costs, a.intra_inter_ctx, &a.single_ref_ctx);

    // Candidate modes that need NO motion search, in C's evaluation order
    // (`av1_default_mode_order`, rdopt.c:111: NEARESTMV → [NEWMV] → NEARMV →
    // GLOBALMV) so a strict-`<` best keeps the same winner C's loop keeps on
    // an exact RD tie. NEARMV is only available when the scan produced a
    // second stack entry (C's `ref_mv_count` gate in `handle_inter_mode`'s
    // DRL setup).
    let mut cands: Vec<(i32, (i32, i32))> = vec![(NEARESTMV, a.nearest_mv)];
    if a.ref_mv_count > 1 {
        cands.push((NEARMV, a.near_mv));
    }
    cands.push((GLOBALMV, a.global_mv));

    let px_x = a.mi_col as usize * 4;
    let px_y = a.mi_row as usize * 4;

    // --- the ZERO-MV predictor + per-plane sse, computed ONCE (every rung-1
    //     candidate shares the identical co-located-copy prediction) ---
    //
    // LUMA: only the VISIBLE extent is differenced — C's `set_skip_txfm` sse
    // and the residual clip to `max_block_wide/high` at frame edges. `pred_y`
    // stays zero in the invisible tail, matching `av1_subtract_plane`'s
    // zero-filled tail.
    let mut pred_y = vec![0u16; bw * bh];
    copy_colocated(
        &a.ref_frame.y,
        a.ref_frame.stride,
        a.ref_frame.width,
        a.ref_frame.height,
        px_x,
        px_y,
        vis_w,
        vis_h,
        &mut pred_y,
        bw,
    );
    let luma_sse = sse_visible(a.src_y, a.off_y, a.stride, &pred_y, bw, vis_w, vis_h);

    // CHROMA (when this leaf is a chroma ref): C (`set_skip_txfm`,
    // tx_search.c:245-281 — and identically `model_rd_for_sb_with_curvfit`'s
    // `get_txb_dimensions`) works on the PADDED plane block
    // (`get_plane_block_size` — a sub-8x8 luma leaf's chroma-ref covers the
    // full 4x4-minimum chroma block at the COVERING position, not a
    // `bw >> ss_x` strip; the KB-15 skip-arm chroma-extent lesson), clipped to
    // the frame-visible part. `a.off_uv` is the covering-position source
    // offset (`chroma_plane_offset`); the ref read mirrors it.
    let mut chroma_sse: i64 = 0;
    // Per-plane (sse, plane_bsize, num_samples) for the curvefit model.
    let mut chroma_model: Vec<(i64, usize, i32)> = Vec::new();
    if !a.monochrome && a.is_chroma_ref {
        let plane_bsize =
            aom_dsp::entropy::partition::get_plane_block_size(a.bsize, a.ss_x, a.ss_y);
        let cw_full = crate::tx_search::BLK_W_B[plane_bsize];
        let ch_full = crate::tx_search::BLK_H_B[plane_bsize];
        // Frame-visible chroma extent in 4px units → px.
        let (cvis_wu, cvis_hu, _, _) = crate::tx_search::max_block_units(
            a.mi_cols,
            a.mi_rows,
            a.mi_col,
            a.mi_row,
            mi_w as i32,
            mi_h as i32,
            cw_full,
            ch_full,
            a.ss_x,
            a.ss_y,
        );
        let cvis_w = (cvis_wu * 4).min(cw_full);
        let cvis_h = (cvis_hu * 4).min(ch_full);
        // Covering position (chroma-reference mi base: mi - (mi & ss)).
        let cpx_x = (((a.mi_col - (a.mi_col & a.ss_x as i32)) as usize) * 4) >> a.ss_x;
        let cpx_y = (((a.mi_row - (a.mi_row & a.ss_y as i32)) as usize) * 4) >> a.ss_y;
        for (plane_ref, plane_src) in [(&a.ref_frame.u, a.src_u), (&a.ref_frame.v, a.src_v)] {
            let mut cpred = vec![0u16; cw_full * ch_full];
            copy_colocated(
                plane_ref,
                a.ref_frame.stride_uv,
                a.ref_frame.width_uv,
                a.ref_frame.height_uv,
                cpx_x,
                cpx_y,
                cvis_w,
                cvis_h,
                &mut cpred,
                cw_full,
            );
            let sse = sse_visible(plane_src, a.off_uv, a.stride, &cpred, cw_full, cvis_w, cvis_h);
            chroma_sse += sse;
            chroma_model.push((sse, plane_bsize, (cvis_w * cvis_h) as i32));
        }
    }

    // --- the switchable interp-filter pick (crate::interp_rd module docs):
    //     identical predictions across filters ⇒ the model stats are shared
    //     and the C search collapses to the biased rate compare. The winner's
    //     `rs` joins every candidate's rate (C: `rate2_nocoeff`), exactly as
    //     C's SWITCHABLE-during-encode frame filter demands. ---
    let (model_rate, model_dist) = {
        let mut rate_sum: i64 = 0;
        let mut dist_sum: i64 = 0;
        let (r, d) = crate::interp_rd::model_rd_with_curvfit(
            a.bsize,
            luma_sse,
            (vis_w * vis_h) as i32,
            a.dequant_ac,
            a.bd,
            a.rdmult,
        );
        rate_sum += i64::from(r);
        dist_sum += d;
        for &(sse, plane_bsize, ns) in &chroma_model {
            let (r, d) = crate::interp_rd::model_rd_with_curvfit(
                plane_bsize,
                sse,
                ns,
                a.dequant_ac,
                a.bd,
                a.rdmult,
            );
            rate_sum += i64::from(r);
            dist_sum += d;
        }
        (rate_sum.min(i64::from(i32::MAX)) as i32, dist_sum)
    };
    let (interp_filter, rs) = crate::interp_rd::pick_interp_filter_zero_mv(
        a.interp_costs,
        a.interp_ctx,
        a.use_more_sharp_interp,
        a.rdmult,
        model_rate,
        model_dist,
    );

    // --- SKIP gate: C's own `predict_skip_txfm` fast path ---
    let residual = residual_full(a.src_y, a.off_y, a.stride, &pred_y, bw, bh, vis_w, vis_h);
    let skips = crate::intrabc_search::predict_skip_txfm(
        &residual,
        bw,
        bh,
        a.bsize,
        luma_sse,
        a.qindex,
        i32::from(a.bd),
        a.reduced_tx_set_used,
    );

    let mut best: Option<InterBest> = None;
    for (mode, mv) in cands {
        // rung 1 codes only the zero-MV predictor exactly (a plain co-located
        // copy). A nonzero candidate MV needs the MC path (sub-step 2e) and is
        // rung 2 — decline rather than mispredict.
        if mv != (0, 0) {
            continue;
        }
        if !skips {
            // The COEFF arm is a later rung — decline instead of coding a block
            // whose transform decisions we cannot yet reproduce exactly.
            continue;
        }

        let dist = scale_dist(luma_sse + chroma_sse, a.bd);
        let rate = ref_rate
            + cost_mv_ref(a.costs, mode, a.mode_context)
            + rs
            + a.skip_costs[a.skip_ctx][1];
        let rd = rdcost(a.rdmult, rate, dist);
        if rd >= best_rd_in {
            continue;
        }
        let better = match &best {
            None => true,
            Some(b) => rd < b.rdcost,
        };
        if better {
            best = Some(InterBest {
                mode,
                interp_filter: interp_filter as u8,
                ref_frame0: LAST_FRAME as i8,
                mv_row: mv.0,
                mv_col: mv.1,
                mode_context: a.mode_context,
                skip_txfm: true,
                rate,
                dist,
                rdcost: rd,
            });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_ref(w: usize, h: usize, val: u16) -> RefFrame {
        RefFrame {
            y: vec![val; w * h],
            u: vec![val; (w / 2) * (h / 2)],
            v: vec![val; (w / 2) * (h / 2)],
            stride: w,
            stride_uv: w / 2,
            width: w,
            height: h,
            width_uv: w / 2,
            height_uv: h / 2,
            order_hint: 0,
        }
    }

    /// A block whose source EQUALS the reference must predict-skip with zero
    /// distortion, and the winner must be the cheapest-rate mode. With the
    /// default CDFs that is NEARESTMV (GLOBALMV pays the improbable zeromv
    /// branch) — the same mode `aomenc` codes for the zero-MV P, measured with
    /// the instrumented libaom decoder.
    #[test]
    fn identical_source_picks_nearestmv_skip_at_zero_dist() {
        let costs =
            crate::inter_costs::derive_inter_mode_costs(&crate::inter_costs::InterFrameCdfs::defaults());
        let skip_costs = [[100i32, 50]; 3];
        let rf = flat_ref(64, 64, 128);
        let src = vec![128u16; 64 * 64];
        let src_uv = vec![128u16; 32 * 32];
        let a = InterLeafArgs {
            bsize: 12, // BLOCK_64X64
            mi_row: 0,
            mi_col: 0,
            mi_rows: 16,
            mi_cols: 16,
            monochrome: true,
            is_chroma_ref: false,
            ss_x: 1,
            ss_y: 1,
            bd: 8,
            stride: 64,
            src_y: &src,
            src_u: &src_uv,
            src_v: &src_uv,
            off_y: 0,
            off_uv: 0,
            ref_frame: &rf,
            mode_context: 0,
            nearest_mv: (0, 0),
            near_mv: (0, 0),
            global_mv: (0, 0),
            ref_mv_count: 1,
            rdmult: 100,
            qindex: 240,
            reduced_tx_set_used: false,
            costs: &costs,
            skip_costs: &skip_costs,
            skip_ctx: 0,
            intra_inter_ctx: 0,
            single_ref_ctx: SingleRefCtx::default(),
            interp_costs: &crate::interp_rd::SwitchableInterpCosts::from_default_cdfs(),
            interp_ctx: 3,
            use_more_sharp_interp: false,
            dequant_ac: 0,
        };
        let best = rd_pick_inter_mode_sb(&a, i64::MAX).expect("a perfect match must be codeable");
        assert_eq!(best.mode, NEARESTMV, "NEARESTMV is the cheapest zero-MV mode");
        assert_eq!(best.dist, 0, "identical source and reference ⇒ zero SSE");
        assert!(best.skip_txfm);
        assert_eq!(best.ref_frame0, LAST_FRAME as i8);
        // Anti-vacuity: the rate must contain all four components — ref +
        // mode + the switchable interp-filter rs (REGULAR at ctx 3 here:
        // zero model stats and no sharp bias keep the baseline filter) +
        // skip.
        let interp = crate::interp_rd::SwitchableInterpCosts::from_default_cdfs();
        let expect_rate = ref_cost_single_last(&costs, 0, &SingleRefCtx::default())
            + cost_mv_ref(&costs, NEARESTMV, 0)
            + crate::interp_rd::SWITCHABLE_INTERP_RATE_FACTOR * interp.0[3][0]
            + skip_costs[0][1];
        assert_eq!(best.rate, expect_rate);
    }

    /// A block whose source is FAR from the reference must NOT predict-skip, so
    /// the arm declines (the COEFF arm is a later rung). This is the guard that
    /// keeps a non-skip block from being emitted as skip.
    #[test]
    fn high_residual_block_declines() {
        let costs =
            crate::inter_costs::derive_inter_mode_costs(&crate::inter_costs::InterFrameCdfs::defaults());
        let skip_costs = [[100i32, 50]; 3];
        let rf = flat_ref(64, 64, 0);
        // A high-contrast source against a flat-zero reference.
        let mut src = vec![0u16; 64 * 64];
        for (i, p) in src.iter_mut().enumerate() {
            *p = if (i / 64 + i % 64) % 2 == 0 { 0 } else { 255 };
        }
        let src_uv = vec![128u16; 32 * 32];
        let a = InterLeafArgs {
            bsize: 12,
            mi_row: 0,
            mi_col: 0,
            mi_rows: 16,
            mi_cols: 16,
            monochrome: true,
            is_chroma_ref: false,
            ss_x: 1,
            ss_y: 1,
            bd: 8,
            stride: 64,
            src_y: &src,
            src_u: &src_uv,
            src_v: &src_uv,
            off_y: 0,
            off_uv: 0,
            ref_frame: &rf,
            mode_context: 0,
            nearest_mv: (0, 0),
            near_mv: (0, 0),
            global_mv: (0, 0),
            ref_mv_count: 1,
            rdmult: 100,
            // A LOW qindex: fine quantization ⇒ the residual will not be
            // predicted away.
            qindex: 20,
            reduced_tx_set_used: false,
            costs: &costs,
            skip_costs: &skip_costs,
            skip_ctx: 0,
            intra_inter_ctx: 0,
            single_ref_ctx: SingleRefCtx::default(),
            interp_costs: &crate::interp_rd::SwitchableInterpCosts::from_default_cdfs(),
            interp_ctx: 3,
            use_more_sharp_interp: false,
            dequant_ac: 0,
        };
        assert!(
            rd_pick_inter_mode_sb(&a, i64::MAX).is_none(),
            "a high-residual block must decline (COEFF arm not in scope), not code as skip"
        );
    }

    /// A tight `best_rd_in` budget must reject the candidate — the arm never
    /// returns a winner worse than the caller's incumbent (C's `ref_best_rd`).
    #[test]
    fn respects_ref_best_rd_budget() {
        let costs =
            crate::inter_costs::derive_inter_mode_costs(&crate::inter_costs::InterFrameCdfs::defaults());
        let skip_costs = [[100i32, 50]; 3];
        let rf = flat_ref(64, 64, 128);
        let src = vec![128u16; 64 * 64];
        let src_uv = vec![128u16; 32 * 32];
        let mk = |budget: i64| {
            let a = InterLeafArgs {
                bsize: 12,
                mi_row: 0,
                mi_col: 0,
                mi_rows: 16,
                mi_cols: 16,
                monochrome: true,
                is_chroma_ref: false,
                ss_x: 1,
                ss_y: 1,
                bd: 8,
                stride: 64,
                src_y: &src,
                src_u: &src_uv,
                src_v: &src_uv,
                off_y: 0,
                off_uv: 0,
                ref_frame: &rf,
                mode_context: 0,
                nearest_mv: (0, 0),
                near_mv: (0, 0),
                global_mv: (0, 0),
                ref_mv_count: 1,
                rdmult: 100,
                qindex: 240,
                reduced_tx_set_used: false,
                costs: &costs,
                skip_costs: &skip_costs,
                skip_ctx: 0,
                intra_inter_ctx: 0,
                single_ref_ctx: SingleRefCtx::default(),
            interp_costs: &crate::interp_rd::SwitchableInterpCosts::from_default_cdfs(),
            interp_ctx: 3,
            use_more_sharp_interp: false,
            dequant_ac: 0,
            };
            rd_pick_inter_mode_sb(&a, budget)
        };
        assert!(mk(i64::MAX).is_some(), "unbounded budget accepts");
        assert!(mk(0).is_none(), "a zero budget must reject every candidate");
    }
}
