//! `av1_rd_pick_partition` (av1/encoder/partition_search.c:5653) — the
//! partition RDO layer: the speed-0 GOOD KEY-frame SURVEY (source-cited) and
//! the first landed slice, the NONE-vs-SPLIT recursion skeleton.
//!
//! # Survey: the speed-0 GOOD KEY-frame partition search shape
//!
//! Stage order at each square node (partition_search.c:5653-6046):
//!
//! 1. `init_partition_search_state_params` (:3311): blk geometry
//!    (`mi_step = mi_size_wide/2`, `has_rows/cols` vs frame mi dims,
//!    `bsize_at_least_8x8`), `pl_ctx_idx = partition_plane_context` (>=8x8;
//!    ported in aom-entropy), `partition_cost =
//!    mode_costs->partition_cost[pl_ctx_idx]`, flags
//!    `do_square_split = bsize_at_least_8x8`, `do_rectangular_split =
//!    enable_rect_partitions && bsize_at_least_8x8`,
//!    `partition_none_allowed = has_rows && has_cols`,
//!    `partition_rect_allowed[H/V]` (+ chroma-INVALID subsize guards).
//! 2. Edge-cost override (`!av1_blk_has_rows_and_cols` ->
//!    `set_partition_cost_for_edge_blk`, the read_partition-mirroring
//!    gather); `use_square_partition_only_threshold` clamp (framesize-
//!    dependent; rect-only effect).
//! 3. `av1_set_offsets`; the `mode == ALLINTRA` SB-level
//!    `intra_sb_rdmult_modifier` from `log_sub_block_var` (:5710-5724) —
//!    OFF for the usage-GOOD envelope; `setup_block_rdmult` (:596): at
//!    GOOD/KEY/NO_AQ/no-delta-q/default-tuning it is exactly
//!    `x->rdmult = cpi->rd.RDMULT` (frame-constant) with a floor at 1 —
//!    CONSTANT across the whole recursion; `av1_rd_cost_update(rdmult,
//!    &best_rdc)` re-derives the incoming budget's rdcost.
//! 4. `av1_save_context` (entropy ctx + mi grid snapshot; pixels are NOT
//!    saved — winner pixels propagate via the dry-run encode below).
//! 5. `av1_prune_partitions_before_search` (partition_strategy.c:1648): at
//!    GOOD speed-0 KEY this is a NO-OP for the none/split slice —
//!    `rect_partition_eval_thresh` (rect-only),
//!    `prune_rectangular_split_based_on_qidx = 0`,
//!    `prune_sub_8x8_partition_level = 0`,
//!    `intra_cnn_based_part_prune_level = 0` at speed 0 (set only at
//!    `speed >= 1`, speed_features.c:1177-1178), simple-motion arms
//!    inter-only. `av1_prune_partitions_by_max_min_bsize`: clamps against
//!    `x->sb_enc.{min,max}_partition_size` (defaults 4x4/128 for KEY —
//!    simple-motion auto-min-max is inter-only).
//! 6. `BEGIN_PARTITION_SEARCH` retry label; `must_find_valid_partition`
//!    resets limits when the first pass found nothing (SB-level only).
//! 7. ALLINTRA variance force-split / rect-prune arm (:5791-5827) — gated
//!    `mode == ALLINTRA`: OFF for the usage-GOOD envelope.
//! 8. NONE stage `none_partition_search` (:4399): gates
//!    (`terminate || !partition_none_allowed`);
//!    `set_none_partition_params` (:4215) — `pt_cost =
//!    partition_cost[NONE]` (>=8x8; `INT_MAX -> 0` guard for
//!    edge-overridden costs), `best_remain = av1_rd_stats_subtraction(
//!    rdmult, best_rdc, {rate: pt_cost})`; `pick_sb_modes` (the leaf: KEY
//!    -> `av1_rd_pick_intra_mode_sb`, ported as
//!    [`crate::rd_pick::rd_pick_intra_mode_sb`]);
//!    `av1_rd_cost_update`; `none_rd`/`part_none_rd` bookkeeping; on valid
//!    rate: `rate += pt_cost` (>=8x8) + rdcost recompute, strict-<
//!    best update (`partitioning = NONE` when >=8x8),
//!    `prune_partitions_after_none` (:4247 — ENTIRELY inter-gated:
//!    the skippable-breakout arm and `simple_motion_search_early_term_none`
//!    both require `!frame_is_intra_only`; NO-OP for KEY),
//!    `prune_rect_part_using_none_pred_mode` (sf 0 at GOOD speed 0);
//!    `av1_restore_context`.
//! 9. SPLIT stage `split_partition_search` (:4512): gates
//!    (`terminate || !do_square_split`); `sum_rdc = {rate:
//!    partition_cost[SPLIT], rdcost: RDCOST(rate, 0)}`; 4 children
//!    `while sum_rdc.rdcost < best_rdc.rdcost`: child origin
//!    `(mi_row + (idx>>1)*mi_step, mi_col + (idx&1)*mi_step)`,
//!    out-of-frame children skipped (`continue`), `best_remain =
//!    av1_rd_stats_subtraction(rdmult, best_rdc, sum_rdc)` (rate can go
//!    NEGATIVE -> `RDCOST_NEG_R` arm of `av1_calculate_rd_cost`),
//!    recursive `av1_rd_pick_partition` (false -> invalid sum + break),
//!    accumulate rate/dist + `av1_rd_cost_update`; after the loop
//!    `reached_last_index = (idx == 4)`; if reached && `sum < best`:
//!    `sum_rdc.rdcost = RDCOST(rate, dist)` recompute,
//!    `split_partition_penalty_level = 0` at GOOD speed-0 KEY (only the
//!    low-complexity-decode paths set it, and `is_key_frame -> 0` there,
//!    speed_features.c:659,682) => penalty factor 1.0; strict-< best
//!    update (`partitioning = SPLIT`; the STORED best rdcost is the
//!    un-penalized recompute). `less_rectangular_check_level = 0` at GOOD
//!    speed 0 (allintra sets 1; rect-only effect anyway).
//! 10. `early_term_after_none_split = 0` at GOOD speed 0;
//!     `skip_non_sq_part_based_on_none = 0`; `prune_partitions_after_split`
//!     ml arm inter-only. Then the RECT stage (LIVE at speed 0:
//!     `do_rectangular_split = 1`, `ml_prune_partition = 1` prunes via NN),
//!     AB stage (`prune_ext_partition_types_search_level = 1`,
//!     `ext_partition_eval_thresh`), and 4-way stage
//!     (`prune_part4_search = 2`) — ALL OUT OF THIS SLICE (see Scope).
//! 11. must-find retry (SB only), `*rd_cost = best_rdc`, and THE WINNER
//!     DRY-RUN ENCODE (:5998-6026): at SB level `encode_sb(OUTPUT_ENABLED)`;
//!     below SB, `should_do_dry_run_encode_for_current_block(sb_size,
//!     max_partition_size, pc_tree->index, bsize)` (:5556 — `bsize >
//!     max_partition -> false`; `index != 3 -> true`; the 4th child only
//!     when `bsize == max_partition != sub_sb_size`) gates
//!     `encode_sb(DRY_RUN_NORMAL)` of the winner subtree. **This dry-run
//!     encode is how siblings/parents see the winner's reconstruction and
//!     entropy contexts** (save/restore_context snapshots contexts, never
//!     pixels) — it runs `av1_encode_intra_block_plane` for ALL planes,
//!     making the chroma re-encode arm the critical path for the next
//!     layer. `x->rdmult` restore.
//!
//! # Scope of the landed slice
//!
//! [`rd_pick_partition_none_split`] = stages 1 (none/split fields) + 8 + 9 +
//! the found/best threading, for INTERIOR square blocks with leaf evaluation
//! supplied by the caller — the recursion/cost/threshold CONTROL FLOW.
//! **MISSING (fractions): 2 of 10 partition types are searched (NONE,
//! SPLIT); RECT/AB/4-way stages (live at speed 0) are not ported; the
//! winner dry-run encode_sb (sibling pixel/context propagation) is not
//! ported — it needs the all-planes re-encode + tokenize-context layer;
//! edge-block cost override + must-find retry + save/restore context
//! threading are not ported.** The differential constrains the C side to
//! the same 2-type search, validating control flow, not the full partition
//! decision.

use crate::rd::{rdcost, rdcost_neg_r};

/// `RD_STATS` rate/dist/rdcost slice with the C invalid conventions
/// (`av1_init_rd_stats` zeroes; `av1_invalid_rd_stats` sets
/// rate `INT_MAX` / dist `INT64_MAX` / rdcost `INT64_MAX`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartRdStats {
    pub rate: i32,
    pub dist: i64,
    pub rdcost: i64,
}

impl PartRdStats {
    /// `av1_init_rd_stats` (rate/dist/rdcost slice).
    pub fn init() -> Self {
        PartRdStats { rate: 0, dist: 0, rdcost: 0 }
    }
    /// `av1_invalid_rd_stats`.
    pub fn invalid() -> Self {
        PartRdStats { rate: i32::MAX, dist: i64::MAX, rdcost: i64::MAX }
    }
    pub fn is_invalid(&self) -> bool {
        self.rate == i32::MAX
    }
}

/// `av1_calculate_rd_cost` (rd.h:193-199): negative rates (legal after
/// `av1_rd_stats_subtraction`) take the `RDCOST_NEG_R` arm.
pub fn calculate_rd_cost(mult: i32, rate: i32, dist: i64) -> i64 {
    if rate >= 0 {
        rdcost(mult, rate, dist)
    } else {
        rdcost_neg_r(mult, -rate, dist)
    }
}

/// `av1_rd_cost_update` (rd.h:201-208).
pub fn rd_cost_update(mult: i32, rd: &mut PartRdStats) {
    if rd.rate < i32::MAX && rd.dist < i64::MAX && rd.rdcost < i64::MAX {
        rd.rdcost = calculate_rd_cost(mult, rd.rate, rd.dist);
    } else {
        *rd = PartRdStats::invalid();
    }
}

/// `av1_rd_stats_subtraction` (rd.h:210-223).
pub fn rd_stats_subtraction(mult: i32, left: &PartRdStats, right: &PartRdStats) -> PartRdStats {
    if left.rate == i32::MAX
        || right.rate == i32::MAX
        || left.dist == i64::MAX
        || right.dist == i64::MAX
        || left.rdcost == i64::MAX
        || right.rdcost == i64::MAX
    {
        PartRdStats::invalid()
    } else {
        let rate = left.rate - right.rate;
        let dist = left.dist - right.dist;
        PartRdStats { rate, dist, rdcost: calculate_rd_cost(mult, rate, dist) }
    }
}

/// `mi_size_wide[BLOCK_SIZES_ALL]` for the square sizes used here.
const MI_SIZE_WIDE_SQ: [usize; 22] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];

/// `get_partition_subsize(bsize, PARTITION_SPLIT)` for square bsizes
/// (BLOCK_8X8=3 -> 4X4=0, 16X16=6 -> 8X8=3, 32X32=9 -> 6, 64X64=12 -> 9,
/// 128X128=15 -> 12).
pub fn split_subsize(bsize: usize) -> usize {
    match bsize {
        3 => 0,
        6 => 3,
        9 => 6,
        12 => 9,
        15 => 12,
        _ => panic!("split_subsize: non-splittable square bsize {bsize}"),
    }
}

/// The per-node inputs of the NONE-vs-SPLIT slice: geometry + the
/// `partition_cost[pl_ctx_idx]` row for THIS node (the caller resolves
/// `partition_plane_context` — aom-entropy owns the ported facade) + the
/// stage-gate flags `init_partition_search_state_params` derives (with the
/// pre-search prunes already applied by the caller; all no-ops at GOOD
/// speed-0 KEY interior — module docs #5).
pub struct PartNodeParams<'a> {
    pub bsize: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    /// Frame mi dims (`has_rows/cols` + out-of-frame child skips).
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// `partition_cost[pl_ctx_idx]` — `[NONE, HORZ, VERT, SPLIT, ..]`; this
    /// slice reads indices 0 and 3.
    pub partition_cost: &'a [i32],
    /// `partition_none_allowed` (init: `has_rows && has_cols`).
    pub partition_none_allowed: bool,
    /// `do_square_split` (init: `bsize_at_least_8x8`, then max/min-bsize
    /// prunes).
    pub do_square_split: bool,
}

/// One leaf (PARTITION_NONE) evaluation — `pick_sb_modes` as a callback:
/// `(mi_row, mi_col, bsize, best_remain) -> PartRdStats` (the
/// `rd_cost` out-state incl. the `rate == INT_MAX -> rdcost = INT64_MAX`
/// normalization, partition_search.c:970). The whole-block intra evaluator
/// is [`crate::rd_pick::rd_pick_intra_mode_sb`]; the callback boundary keeps
/// this slice pure control flow (and lets the differential pin the recursion
/// with deterministic leaves).
pub type LeafEval<'a> = dyn FnMut(i32, i32, usize, &PartRdStats) -> PartRdStats + 'a;

/// A node of the returned partition tree (`pc_tree` slice).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartTree {
    /// PARTITION_NONE winner (the leaf stats INCLUDING `pt_cost`).
    None(PartRdStats),
    /// PARTITION_SPLIT winner: the summed stats + 4 children in raster
    /// order.
    Split(PartRdStats, Vec<PartTree>),
    /// No partitioning beat the incoming budget (`found_best_partition`
    /// false — at sub-SB levels the C returns false and the parent
    /// invalidates its sum).
    NotFound,
}

/// The per-node partition-cost row provider: `bsize -> partition_cost` row +
/// gate flags (the caller owns partition_plane_context threading; for the
/// constrained interior sweep the ctx is positionally fixed per node).
pub type NodeParamsFn<'a> = dyn FnMut(i32, i32, usize) -> (Vec<i32>, bool, bool) + 'a;

/// `av1_rd_pick_partition` — the NONE + SPLIT slice (module docs #8/#9):
/// evaluate PARTITION_NONE via `leaf`, then PARTITION_SPLIT via recursion,
/// with the exact budget threading (`best_remain` subtractions, running
/// `sum_rdc` early-outs, strict-< best updates, the >=8x8 `pt_cost`
/// handling, the un-penalized stored split rdcost). Returns
/// `(tree, best_rdc, found)` — `found` mirrors the C bool return; when
/// false, `best_rdc` is the (unbeaten) incoming budget and `tree` is
/// [`PartTree::NotFound`].
///
/// Interior-or-edge geometry is honored for child skips (`mi_row/col`
/// bounds) and `partition_none_allowed`; the EDGE partition-cost override
/// (module docs #2) is the caller's (not ported).
#[allow(clippy::too_many_arguments)]
pub fn rd_pick_partition_none_split(
    node_params: &mut NodeParamsFn,
    leaf: &mut LeafEval,
    rdmult: i32,
    mi_row: i32,
    mi_col: i32,
    mi_rows: i32,
    mi_cols: i32,
    bsize: usize,
    mut best_rdc: PartRdStats,
) -> (PartTree, PartRdStats, bool) {
    // best_rdc.rdcost < 0 -> invalid (av1_rd_pick_partition:5675-5678).
    if best_rdc.rdcost < 0 {
        return (PartTree::NotFound, PartRdStats::invalid(), false);
    }
    let (partition_cost, partition_none_allowed, do_square_split) =
        node_params(mi_row, mi_col, bsize);
    let bsize_at_least_8x8 = bsize >= 3; // BLOCK_8X8
    let mi_step = (MI_SIZE_WIDE_SQ[bsize] / 2) as i32;

    let mut found = false;
    let mut best_tree = PartTree::NotFound;

    // ---- PARTITION_NONE stage (none_partition_search) ----
    if partition_none_allowed {
        let mut pt_cost = 0i32;
        if bsize_at_least_8x8 {
            pt_cost = if partition_cost[0] < i32::MAX { partition_cost[0] } else { 0 };
        }
        // best_remain = best_rdc - {rate: pt_cost} (set_none_partition_params).
        let mut partition_rdcost = PartRdStats::init();
        partition_rdcost.rate = pt_cost;
        rd_cost_update(rdmult, &mut partition_rdcost);
        let best_remain = rd_stats_subtraction(rdmult, &best_rdc, &partition_rdcost);

        let mut this_rdc = leaf(mi_row, mi_col, bsize, &best_remain);
        rd_cost_update(rdmult, &mut this_rdc);

        if this_rdc.rate != i32::MAX {
            if bsize_at_least_8x8 {
                this_rdc.rate += pt_cost;
                this_rdc.rdcost = rdcost(rdmult, this_rdc.rate, this_rdc.dist);
            }
            if this_rdc.rdcost < best_rdc.rdcost {
                best_rdc = this_rdc;
                found = true;
                // pc_tree->partitioning = PARTITION_NONE (>=8x8; 4x4 has no
                // partitioning field to set — the leaf IS the block).
                best_tree = PartTree::None(this_rdc);
            }
        }
        // prune_partitions_after_none: NO-OP for KEY (module docs #8).
        // av1_restore_context: context threading is the caller's layer.
    }

    // ---- PARTITION_SPLIT stage (split_partition_search) ----
    if do_square_split {
        let subsize = split_subsize(bsize);
        let mut sum_rdc = PartRdStats::init();
        sum_rdc.rate = partition_cost[3]; // PARTITION_SPLIT
        sum_rdc.rdcost = rdcost(rdmult, sum_rdc.rate, 0);

        let mut children: Vec<PartTree> = Vec::new();
        let mut idx = 0usize;
        while idx < 4 && sum_rdc.rdcost < best_rdc.rdcost {
            let x_idx = ((idx & 1) as i32) * mi_step;
            let y_idx = ((idx >> 1) as i32) * mi_step;
            // Out-of-frame children are skipped (`continue` — they still
            // count toward reached_last_index; split_partition_search:4561).
            if mi_row + y_idx >= mi_rows || mi_col + x_idx >= mi_cols {
                children.push(PartTree::NotFound);
                idx += 1;
                continue;
            }
            let best_remain = rd_stats_subtraction(rdmult, &best_rdc, &sum_rdc);
            let (child_tree, child_rdc, child_found) = rd_pick_partition_none_split(
                node_params,
                leaf,
                rdmult,
                mi_row + y_idx,
                mi_col + x_idx,
                mi_rows,
                mi_cols,
                subsize,
                best_remain,
            );
            if !child_found {
                sum_rdc = PartRdStats::invalid();
                children.push(child_tree);
                break;
            }
            sum_rdc.rate += child_rdc.rate;
            sum_rdc.dist += child_rdc.dist;
            rd_cost_update(rdmult, &mut sum_rdc);
            children.push(child_tree);
            idx += 1;
        }
        let reached_last_index = idx == 4;

        if reached_last_index && sum_rdc.rdcost < best_rdc.rdcost {
            // split_partition_penalty_level = 0 at GOOD speed-0 KEY =>
            // penalty factor 1.0; the stored rdcost is the recompute.
            sum_rdc.rdcost = rdcost(rdmult, sum_rdc.rate, sum_rdc.dist);
            if sum_rdc.rdcost < best_rdc.rdcost {
                best_rdc = sum_rdc;
                found = true;
                best_tree = PartTree::Split(sum_rdc, children);
            }
        }
    }

    if found {
        (best_tree, best_rdc, true)
    } else {
        // The C returns found=false; rd_cost keeps the incoming budget's
        // stats semantics (the caller only consumes it when found).
        (PartTree::NotFound, best_rdc, false)
    }
}
