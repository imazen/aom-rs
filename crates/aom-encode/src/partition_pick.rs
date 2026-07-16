//! `av1_rd_pick_partition` with REAL leaves — the NONE-vs-SPLIT recursion
//! (partition_search.c:5653) wired to [`crate::rd_pick::rd_pick_intra_mode_sb`]
//! leaf evaluation and the winner-subtree DRY_RUN [`crate::encode_sb`]
//! propagation, over live tile contexts, recon planes, and the mi-grid mode
//! state. This makes the partition skeleton REAL on its two ported types:
//! sibling blocks see each other's winner pixels (recon), entropy/partition/
//! txfm contexts (the dry-run stamps), and mi-grid neighbour modes.
//!
//! # The per-node C sequence modelled (partition_search.c:5653-6046)
//!
//! 1. `init_partition_search_state_params` — geometry + `partition_cost =
//!    mode_costs->partition_cost[pl_ctx_idx]` (`partition_plane_context`
//!    over the LIVE partition-context arrays) + stage gates.
//! 2. `av1_set_offsets` + `setup_block_rdmult` (rdmult CONSTANT across the
//!    recursion: frame RDMULT at GOOD/KEY/NO_AQ; at ALLINTRA the per-SB
//!    `intra_sb_rdmult_modifier` fold is applied ONCE at the SB root and
//!    stays constant below it — the caller passes the folded value) +
//!    `av1_rd_cost_update(rdmult, &best_rdc)`.
//! 3. `av1_save_context` (:5754) — per-plane above/left ENTROPY contexts +
//!    partition contexts + txfm contexts over the node extent. Pixels are
//!    NOT saved: winner pixels propagate via the dry-run encode alone.
//! 4. NONE stage (`none_partition_search`:4399): `pick_sb_modes` (the leaf
//!    — [`leaf_pick_sb_modes`]) with `best_remain = best_rdc - pt_cost`;
//!    strict-< best update; `av1_restore_context` at the stage tail (:4492).
//! 5. SPLIT stage (`split_partition_search`:4512): 4 recursive children
//!    (`pc_index` = child idx) with running `sum_rdc` budgets; the stage
//!    tail restores context when `bsize <= max_partition_size || bsize ==
//!    sb_size` (:4645-4647; always true in this envelope).
//! 6. The winner encode (:5998-6026): at the SB root the C emits
//!    OUTPUT_ENABLED (adds pack-stage CDF/token state; contexts/pixels are
//!    stamped identically) — modelled as the same DRY_RUN walk; below SB,
//!    `should_do_dry_run_encode_for_current_block` (:5556) gates the
//!    DRY_RUN winner-subtree walk: `bsize > max_partition -> false`;
//!    `pc_index != 3 -> true`; a 4th child re-encodes only when `bsize ==
//!    max_partition != sub_sb_size` (its data is otherwise re-created by
//!    the nearest `index != 3` ancestor's own dry-run before any reader).
//!
//! # mi-grid neighbour semantics (verified)
//!
//! The encoder's mi alloc granularity is `mi_alloc_bsize =
//! default_min_partition_size = BLOCK_4X4` (`enc_set_mb_mi`,
//! encoder_utils.h:93-99) — every 4x4 cell owns a struct; `set_mi_offsets`
//! repoints the ORIGIN grid cell at pick time and `av1_update_state`'s grid
//! fill (encode_b, ANY dry_run) points the block's cells at the winner. The
//! leaf search reads neighbour WINNER modes through `xd->above/left_mbmi`
//! (the cells at `(mi_row-1, mi_col)` / `(mi_row, mi_col-1)`), which the
//! dry-run discipline keeps coherent for every reader (any never-re-encoded
//! `index == 3` subtree has no reader before its ancestor's walk). The
//! [`ModeGrid`] models the mode byte per cell — the only mi-grid field the
//! KEY intra leaf reads (neighbour `skip_txfm` is 0 for every KEY intra
//! block so the skip context is constantly 0; `is_inter` is always false so
//! `get_tx_size_context` takes the txfm-context bytes unmodified).
//!
//! # Leaf inputs derived live (pick_sb_modes, :850-975)
//!
//! - `x->source_variance = av1_get_perpixel_variance_facade(...)`
//!   ([`perpixel_variance_y`]: plane-0 variance vs the flat
//!   `AV1_[HIGH_]VAR_OFFS` buffer `>> num_pels_log2`, encodeframe.c:190).
//! - the HOG `directional_mode_skip_mask` (`intra_pruning_with_hog = 1` at
//!   speed 0 BOTH usages; threshold `thresh[0] = -1.2`,
//!   intra_mode_search.c:1505).
//! - above/left neighbour Y modes from the [`ModeGrid`].
//! - `skip_ctx = 0` (KEY intra invariant, asserted); `tx_size_ctx =
//!   get_tx_size_context` over the live txfm-context arrays.
//! - the entropy-context slices at the block position (what
//!   `av1_get_entropy_contexts` copies).
//!
//! # The RECT stage (partition_search.c:3520-3648; wired :5875)
//!
//! **THE KEY-FRAME RECT STAGE IS NN-FREE.** Every NN prune around HORZ/VERT
//! is `!frame_is_intra_only(cm)`-gated and therefore DEAD in the one-KEY-
//! frame envelope, for BOTH usages (each gate re-verified in v3.14.1):
//! - `av1_ml_prune_rect_partition` (the 9-feature rect NN,
//!   partition_strategy.c:1124): gate at partition_search.c:4336 requires
//!   `!frame_is_intra_only` (also `!ml_early_term_after_part_split_level`,
//!   which is 1 sub-720p — either kills it);
//! - `av1_ml_early_term_after_split` (partition_strategy.c:1017): gate at
//!   partition_search.c:4323 requires `!frame_is_intra_only`;
//! - `simple_motion_search_prune_rect` (sf = 1 both usages):
//!   partition_strategy.c:1822 requires `!frame_is_intra_only`;
//! - `prune_rect_part_using_none_pred_mode` +
//!   `prune_rect_part_using_4x4_var_deviation`: sfs set at ALLINTRA
//!   speed >= 6 only (speed_features.c:539-540) — 0/false at speed 0;
//! - `prune_partitions_after_none` (:4247) and
//!   `prune_partitions_after_split` (:4309) are ENTIRELY
//!   `!frame_is_intra_only`-gated — both no-ops at KEY;
//! - `early_term_after_none_split` (:5851) = 0 at speed 0 both usages
//!   (ALLINTRA speed >= 4, GOOD speed >= 3);
//!   `skip_non_sq_part_based_on_none` (:5859) = 0 at speed 0;
//! - `av1_prune_partitions_before_search` (partition_strategy.c:1648)
//!   reduces at speed-0 KEY to the `bsize > rect_partition_eval_thresh`
//!   check with the DEFAULT `BLOCK_128X128` (speed_features.c:2313) — dead
//!   for the <= 64 envelope;
//! - `use_square_partition_only_threshold` (:5700 `bsize > thresh` rect
//!   kill): BLOCK_64X64 sub-480p / BLOCK_128X128 at 480p+ (ALLINTRA
//!   speed 0, speed_features.c:176-182) — `bsize > 64` never holds in the
//!   sb <= 64 envelope;
//! - `reuse_prev_rd_results_for_part_ab = 1` both usages, but
//!   `ctx->rd_mode_is_ready` (pick_sb_modes:854 early-return) is only ever
//!   set on AB-stage contexts — dead until the AB chunk.
//!
//! Flag init (`init_partition_search_state_params`:3380-3399):
//! `do_rectangular_split = enable_rect_partitions && bsize_at_least_8x8`;
//! `partition_rect_allowed[HORZ] = do_rect && has_cols &&
//! get_plane_block_size(HORZ subsize) != BLOCK_INVALID` (the 4:2:2
//! tall-block guard; VERT mirrored with `has_rows`); `prune_rect_part`
//! zeroed; `none_rd = 0`, `split_rd/rect_part_rd` zeroed.
//! `av1_prune_partitions_by_max_min_bsize` (partition_strategy.c:1837)
//! extends over rect: gt-max -> `av1_set_square_split_only` (none off,
//! square on, rect off); le-min -> `av1_disable_rect_partitions` + the
//! square-only clamp.
//!
//! **The per-node ALLINTRA variance arm (:5791-5827) runs BEFORE the NONE
//! stage** for `oxcf.mode == ALLINTRA` at bsize >= 16x16 (speed 0: the
//! `prune_rect_part_using_4x4_var_deviation` sibling arm is sf-dead):
//! `log_sub_block_var` (:5572 — min/max `log1p(var_4x4/16)` over the
//! block's 4x4 source sub-blocks, NO cache) and, when `var_min < 0.272 &&
//! var_max - var_min > 3.0`, forces `partition_none_allowed = 0;
//! terminate_partition_search = 0; do_square_split = 1` ([`log_sub_block_var`]).
//!
//! The one LIVE usage-differing rect knob: `less_rectangular_check_level`
//! (ALLINTRA 1 / GOOD 0 at speed 0; level 2 at ALLINTRA speed >= 3) — the
//! SPLIT stage's ELSE arm (:4630-4640): when NOT (reached_last_index &&
//! sum < best) and (`level == 2 || idx <= 2`), `do_rectangular_split &=
//! !(none_rd > 0 && none_rd < sum_rdc.rdcost)`. `none_rd` is stored at
//! :4458-4459 immediately after the NONE leaf's `av1_rd_cost_update` —
//! **BEFORE pt_cost is added** (pt_cost lands at :4470; the
//! WITH-pt_cost value goes to `part_none_rd`, consumed only by intra-dead
//! prunes). `split_rd[idx]` (:4566) is the child's none_rd out-value —
//! consumed only by the intra-dead NN prunes; threaded for shape.
//!
//! `rectangular_partition_search` (:3520), per type i in {HORZ, VERT}:
//! 1. `is_rect_part_allowed` (:3506): `!terminate &&
//!    partition_rect_allowed[i] && !prune_rect_part[i] &&
//!    (do_rectangular_split || active_edge)`; `av1_active_h/v_edge`
//!    (encodeframe_utils.c:767/797) at the one-pass shape: active iff the
//!    node's mi range straddles 0 or mi_rows/mi_cols.
//! 2. `sum_rdc = {rate: partition_cost[type], rdcost: RDCOST(rate, 0)}`.
//! 3. sub-block 0 at (mi_row, mi_col): `rd_pick_rect_partition` (:3471) =
//!    `best_remain = best - sum` -> `pick_sb_modes(partition_type)` (the
//!    leaf with `mbmi->partition = HORZ/VERT` — feeds the
//!    has_top_right/has_bottom_left tables, which branch only on
//!    VERT_A/VERT_B) -> `av1_rd_cost_update` -> accumulate (rate INT_MAX ->
//!    sum rdcost MAX); records `rect_part_rd[i][0]` (an AB-stage input).
//! 4. If `sum < best && is_not_edge` (has_rows for HORZ / has_cols VERT):
//!    `is_rect_ctx_is_ready[i]` bookkeeping (:3605-3612: palette-free —
//!    always in this envelope — and `uv_mode != UV_CFL_PRED`; the AB
//!    stage's reuse input), then `av1_update_state +
//!    encode_superblock(DRY_RUN_NORMAL)` of sub 0 — the MID-STAGE
//!    propagation (sub 1 reads sub 0's winner pixels, entropy/txfm
//!    contexts AND mi-grid modes; encode_superblock DIRECTLY: no
//!    partition-ctx stamp, no rdmult save — [`crate::encode_sb::encode_b_intra_dry`]
//!    composes exactly those pieces since ctx->mic already carries the
//!    pick's partition). Then sub-block 1 at the edge position
//!    (`mi_row_edge`/`mi_col_edge` = origin + mi_step, :3323).
//! 5. Best update: `sum < best` -> rdcost recompute -> strict-< ->
//!    `partitioning = HORZ/VERT`; the ELSE records
//!    `rect_part_win_info->rect_part_win[i] = false` (NULL outside the
//!    split recursion — an AB-stage input, next chunk).
//! 6. `av1_restore_context` at EACH type's loop tail (:3644) — HORZ's
//!    sub-0 encode debris (contexts) is restored before VERT evaluates;
//!    pixels/mi-grid are NOT restored (C behavior: rect sub-0 pixel debris
//!    persists until the winner encode overwrites the node's extent).
//!
//! # Scope
//!
//! NONE + SPLIT + HORZ + VERT (4 of 10 partition types), KEY intra,
//! interior SBs, sb_size <= 64, no segmentation, min_partition >= 8x8
//! (every rect leaf >= 16x8 is a chroma reference; 8x8 nodes have rect
//! clamped off by le-min). MISSING: AB (HORZ_A/B, VERT_A/B) + 4-way
//! (HORZ_4/VERT_4) stages — `rect_part_rd`/`is_rect_ctx_is_ready`/
//! `rect_part_win` are tracked but their consumers are next chunk; rect at
//! min_partition 4x4 (8x4/4x8 leaves: sub-8x8 shared-chroma pairing + the
//! C's uv_mode read on !chroma_ref sub-0 in `is_rect_ctx_is_ready`); the
//! edge-block partition-cost override + edge rect (sub-0-only HORZ/VERT);
//! the SB-level must-find retry; the SB-root ALLINTRA
//! `intra_sb_rdmult_modifier` fold (:5710 — the folded rdmult is an input
//! here); and the OUTPUT pack-stage state at the SB root.

use crate::encode_sb::{
    LeafEncodeOut, LeafWinner, SbEncodeEnv, SbTree, TileCtxState, encode_sb_dry,
};
use crate::hog::prune_intra_mode_with_hog_y;
use crate::intra_rd::{Block4x4VarInfo, IntraSbyGates, IntraSbySearchCfg};
use crate::intra_uv_rd::{
    UV_CFL_PRED, UvLoopPolicy, UvRdEnv, av1_get_tx_size_uv, chroma_plane_offset,
    is_chroma_reference,
};
use crate::mode_costs::TxSizeCosts;
use crate::mode_costs::{CflCosts, IntraModeCosts};
use crate::partition::{PartRdStats, rd_cost_update, rd_stats_subtraction, split_subsize};
use crate::rd_pick::{RdPickUvArgs, RdPickUvOutcome, ReencodeParams, rd_pick_intra_mode_sb};
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TxTypeSearchPolicy, TxfmYrdEnv};
use aom_dist::highbd_variance;
use aom_entropy::partition::{
    allow_palette, get_partition_subsize, get_plane_block_size, get_tx_size_context,
    palette_bsize_ctx, palette_mode_ctx, partition_plane_context,
};
use aom_intra::cfl::CflCtx;
use aom_txb::TxTypeCosts;

/// `num_pels_log2_lookup[BLOCK_SIZES_ALL]` (common_data.h).
const NUM_PELS_LOG2: [u32; 22] = [
    4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10,
];

/// `av1_get_perpixel_variance(_facade)` for plane 0 (encodeframe.c:190):
/// block variance against the flat `AV1_[HIGH_]VAR_OFFS` buffer (128 <<
/// (bd-8)), `ROUND_POWER_OF_TWO`-normalized by the pel count. Composes the
/// bit-exact [`aom_dist::highbd_variance`] (the `aom_highbd_<bd>_variance`
/// family; the bd-8 variant is numerically the lowbd kernel `aomenc` uses
/// for 8-bit sources).
pub fn perpixel_variance_y(src: &[u16], off: usize, stride: usize, bsize: usize, bd: u8) -> u32 {
    const BLK_W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const BLK_H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    let (w, h) = (BLK_W[bsize], BLK_H[bsize]);
    let offs = vec![128u16 << (bd - 8); w];
    let (var, _sse) = highbd_variance(&src[off..], stride, &offs, 0, w, h, bd);
    let bits = NUM_PELS_LOG2[bsize];
    (var + (1 << (bits - 1))) >> bits
}

/// `log_sub_block_var` (partition_search.c:5572): the min/max
/// `log1p(var/16.0)` over the block's 4x4 SOURCE sub-blocks (frame-edge
/// overhang clipped out via the `mb_to_*_edge` fields; NO per-SB cache —
/// direct `av1_calc_normalized_variance` calls, unlike the leaf
/// variance-factor arm). Feeds the per-node ALLINTRA variance arm
/// (:5791-5827) and the SB-root rdmult modifier (:5710, not ported).
/// f64 arithmetic matches the reference build (`f64::ln_1p` = libm
/// `log1p`; the AOMMIN/MAX fold over `(double)var`).
pub fn log_sub_block_var(
    src: &[u16],
    off: usize,
    stride: usize,
    bsize: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    bd: u8,
) -> (f64, f64) {
    let right_overflow = if mb_to_right_edge < 0 {
        ((-mb_to_right_edge) >> 3) as usize
    } else {
        0
    };
    let bottom_overflow = if mb_to_bottom_edge < 0 {
        ((-mb_to_bottom_edge) >> 3) as usize
    } else {
        0
    };
    let bw = 4 * MI_SIZE_WIDE_B[bsize] - right_overflow;
    let bh = 4 * MI_SIZE_HIGH_B[bsize] - bottom_overflow;
    let mut min_var_4x4 = f64::from(i32::MAX);
    let mut max_var_4x4 = 0.0f64;
    let mut i = 0usize;
    while i < bh {
        let mut j = 0usize;
        while j < bw {
            let var = crate::intra_rd::calc_normalized_variance_4x4(
                src,
                off + i * stride + j,
                stride,
                bd,
            );
            min_var_4x4 = min_var_4x4.min(f64::from(var));
            max_var_4x4 = max_var_4x4.max(f64::from(var));
            j += 4;
        }
        i += 4;
    }
    ((min_var_4x4 / 16.0).ln_1p(), (max_var_4x4 / 16.0).ln_1p())
}

/// `x->intra_sb_rdmult_modifier` (partition_search.c:5710-5722): the ALLINTRA
/// SB-root rdmult scale, derived once per superblock from
/// [`log_sub_block_var`]'s `(var_min, var_max)` taken over the WHOLE SB
/// (`bsize == cm->seq_params->sb_size`, not a sub-node). `128` (identity
/// under the `>>7` fold in [`fold_intra_sb_rdmult`]) unless the SB spans both
/// very-flat (`var_min < 2.0`) and very-detailed (`var_max > 4.0`) 4x4
/// sub-blocks, in which case the multiplier is reduced (floor `128-48=80`,
/// i.e. `>>7` ~= 0.625x) — a flatter effective rdmult for SBs whose content
/// mixes smooth and busy regions, so RD decisions don't over-favor the busy
/// region's higher bit cost.
pub fn intra_sb_rdmult_modifier(var_min: f64, var_max: f64) -> i32 {
    let mut modifier = 128i32;
    if var_min < 2.0 && var_max > 4.0 {
        if (var_max - var_min) > 8.0 {
            modifier -= 48;
        } else {
            modifier -= ((var_max - var_min) * 6.0) as i32;
        }
    }
    modifier
}

/// `setup_block_rdmult`'s ALLINTRA tail (partition_search.c:652-655): fold
/// [`intra_sb_rdmult_modifier`] into `rdmult` (`(rdmult * modifier) >> 7` in
/// 64-bit to avoid the 32-bit product overflowing before the shift, matching
/// the C's explicit `(int64_t)` cast), floored at 1 (`rdmult > 0 ? rdmult :
/// 1` — the modifier can drive it to 0 or negative for extreme `var`
/// spreads). The caller applies this ONCE per SB and holds it constant for
/// every node/leaf below the SB root (the C sets `x->intra_sb_rdmult_modifier`
/// once at the root and every deeper `setup_block_rdmult` call re-reads the
/// SAME stale field value).
pub fn fold_intra_sb_rdmult(rdmult: i32, modifier: i32) -> i32 {
    let folded = ((i64::from(rdmult) * i64::from(modifier)) >> 7) as i32;
    if folded > 0 { folded } else { 1 }
}

/// The per-mi-cell winner Y mode state (`av1_update_state`'s mi-grid fill;
/// module docs). `stride` = frame `mi_cols`.
pub struct ModeGrid {
    pub modes: Vec<u8>,
    /// Per-mi winner `uv_mode` (UV_PREDICTION_MODE), stamped alongside `modes`.
    /// Read for the per-block CHROMA intra-edge-filter type
    /// (`get_intra_edge_filter_type(xd, plane=1)`, reconintra.c:974) — the
    /// chroma analogue of the luma `modes` read. Dead (0) on non-chroma-ref
    /// leaves, but the chroma-neighbour lookup addresses the chroma-reference
    /// mi cell (base_mi + offsets), which always carries the real value.
    pub uv_modes: Vec<u8>,
    pub stride: usize,
}

impl ModeGrid {
    /// All-DC grid (harness seeds real neighbour history where relevant).
    pub fn dc(mi_rows: usize, mi_cols: usize) -> Self {
        ModeGrid {
            modes: vec![0; mi_rows * mi_cols],
            uv_modes: vec![0; mi_rows * mi_cols],
            stride: mi_cols,
        }
    }
    fn stamp(
        &mut self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        mode: u8,
        uv_mode: u8,
        mi_rows: i32,
        mi_cols: i32,
    ) {
        let rows = (MI_SIZE_HIGH_B[bsize] as i32).min(mi_rows - mi_row) as usize;
        let cols = (MI_SIZE_WIDE_B[bsize] as i32).min(mi_cols - mi_col) as usize;
        for r in 0..rows {
            let base = (mi_row as usize + r) * self.stride + mi_col as usize;
            self.modes[base..base + cols].fill(mode);
            self.uv_modes[base..base + cols].fill(uv_mode);
        }
    }
    fn at(&self, mi_row: i32, mi_col: i32) -> u8 {
        self.modes[mi_row as usize * self.stride + mi_col as usize]
    }
    fn at_uv(&self, mi_row: i32, mi_col: i32) -> u8 {
        self.uv_modes[mi_row as usize * self.stride + mi_col as usize]
    }
}

/// The frame-level leaf-search configuration (`pick_sb_modes` +
/// `av1_rd_pick_intra_mode_sb` inputs shared across leaves).
pub struct PickFrameCfg<'a> {
    pub mode_costs: &'a IntraModeCosts,
    /// KEY-frame Y mode costs `y_mode_costs[above_ctx][left_ctx][mode]`
    /// selected per leaf via `intra_mode_context[neighbour mode]`.
    pub tx_size_costs: &'a TxSizeCosts,
    pub skip_costs: &'a [[i32; 2]; 3],
    pub tx_type_costs_y: &'a TxTypeCosts,
    pub pol: &'a TxTypeSearchPolicy,
    pub uv_lp: &'a UvLoopPolicy,
    pub intra_uv_mode_cost: &'a [[[i32; 14]; 13]; 2],
    pub cfl_costs: &'a CflCosts,
    /// `mode_costs->partition_cost[pl_ctx][..]` rows.
    pub partition_costs: &'a [[i32; 10]; 20],
    /// `oxcf.mode == ALLINTRA` (the leaf variance-factor arm; the caller
    /// also pre-folds the SB rdmult modifier into `SbEncodeEnv::rdmult`).
    pub allintra: bool,
    pub speed: i32,
    pub qindex: i32,
    pub enable_filter_intra: bool,
    pub enable_tx64: bool,
    pub enable_rect_tx: bool,
    /// sf `intra_sf.intra_pruning_with_hog` (1 at speed 0, both usages;
    /// KEY-frame threshold row -1.2).
    pub intra_pruning_with_hog: bool,
    /// `oxcf.part_cfg.enable_rect_partitions` (aomenc default 1; the
    /// `do_rectangular_split`/`partition_rect_allowed` init input).
    pub enable_rect_partitions: bool,
    /// sf `part_sf.less_rectangular_check_level` (ALLINTRA 1 / GOOD 0 at
    /// speed 0) — the SPLIT-stage ELSE arm's rect kill (module docs).
    pub less_rectangular_check_level: i32,
    /// `x->sb_enc.max_partition_size` (default `BLOCK_128X128` at KEY —
    /// the dry-run gate + split-restore inputs).
    pub max_partition_size: usize,
    /// `x->sb_enc.min_partition_size` (default `BLOCK_4X4` at KEY;
    /// `aomenc --min-partition-size` raises it — the
    /// `av1_prune_partitions_by_max_min_bsize` clamp,
    /// partition_strategy.c:1837).
    pub min_partition_size: usize,
    /// `oxcf.part_cfg.enable_1to4_partitions` (aomenc default true —
    /// `cfg->disable_1to4_partition_type == 0`; verified against
    /// `av1_cx_iface.c:1124`). Gates the WHOLE `PARTITION_HORZ_4`/`VERT_4`
    /// stage (`prune_4_way_partition_search`'s `partition4_allowed &=
    /// enable_1to4_partitions`, partition_search.c:4165-4166) — set `false`
    /// to reproduce this port's pre-4-way behavior exactly (existing
    /// callers that don't yet cross-check 4-way trees use this).
    pub enable_1to4_partitions: bool,
    /// `oxcf.part_cfg.enable_ab_partitions` (aomenc default true —
    /// `disable_ab_partition_type == 0`; same pattern as
    /// `enable_1to4_partitions`). Gates the WHOLE `PARTITION_HORZ_A/HORZ_B/
    /// VERT_A/VERT_B` stage — `false` at every pre-existing call site,
    /// preserving their established (NONE/SPLIT/HORZ/VERT[/4-way])
    /// behavior/assertions exactly; `true` only at the AB-relevant test
    /// files, matching the established convention from the 4-way port
    /// (module docs on the "4-way partitions ported" milestone in
    /// STATUS.md).
    pub enable_ab_partitions: bool,
    /// `cm->features.allow_screen_content_tools`. Gates the palette-Y
    /// mode flag cost in every DC_PRED leaf's `intra_mode_info_cost_y`
    /// (`av1_allow_palette(allow_screen_content_tools, bsize)`): even when
    /// this port never *picks* palette (the palette search is out of scope),
    /// a screen-content frame still SIGNALS the no-palette flag for each
    /// DC_PRED block at `bsize >= BLOCK_8X8`, and that flag's rate is part of
    /// the leaf RD comparison. Omitting it under-costs DC_PRED and can flip a
    /// near-tie against the directional modes the real encoder picks.
    pub allow_screen_content_tools: bool,
}

/// One leaf evaluation's differential-visibility record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafVisit {
    pub mi_row: i32,
    pub mi_col: i32,
    pub bsize: usize,
    /// The `best_remain` budget the leaf received (rdcost).
    pub budget: i64,
    /// The leaf's returned (rate, dist, rdcost) after the INT_MAX
    /// normalization.
    pub rate: i32,
    pub dist: i64,
    pub rdcost: i64,
}

/// `pick_sb_modes` (partition_search.c:850) for a KEY intra leaf: derive the
/// per-leaf inputs live (module docs) and run the whole-block
/// [`rd_pick_intra_mode_sb`]. `partition` is the `mbmi->partition = partition`
/// install (:887) — the has_top_right/has_bottom_left availability input.
/// Returns the normalized rd stats + the winner as an [`LeafWinner`] (None
/// when `rate == INT_MAX`) + `x->source_variance` (`get_perpixel_variance_
/// facade`'s result for THIS leaf — gotcha #1 in STATUS.md's AB-partition
/// plan: this is returned UNCONDITIONALLY, win or lose, matching the C
/// setting the MACROBLOCK-level mutable field before the mode search can
/// fail; `ml_prune_ab_partition`'s NN feature reads whatever the LAST such
/// call left it at, not the node's own correctly-scoped `pb_source_variance`).
#[allow(clippy::too_many_arguments)]
fn leaf_pick_sb_modes(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &TileCtxState,
    grid: &ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    partition: usize,
    best_remain: &PartRdStats,
) -> (PartRdStats, Option<LeafWinner>, u32) {
    // av1_rd_cost_update(x->rdmult, &best_rd) on entry (pick_sb_modes:927).
    let mut best_rd = *best_remain;
    rd_cost_update(env.rdmult, &mut best_rd);

    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_h = MI_SIZE_HIGH_B[bsize];
    let up_available = mi_row > env.tile_row_start;
    let left_available = mi_col > env.tile_col_start;
    let is_chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;

    // x->source_variance (pick_sb_modes:919).
    let source_variance = perpixel_variance_y(env.src_y, ref_off_y, env.stride, bsize, env.bd);

    // The HOG directional prune mask (av1_rd_pick_intra_sby_mode preamble).
    let mut skip_mask = [false; 13];
    if cfg.intra_pruning_with_hog {
        // Interior blocks: mb_to_*_edge large positive.
        let mb_right = (env.mi_cols - mi_w as i32 - mi_col) * 4 * 8;
        let mb_bottom = (env.mi_rows - mi_h as i32 - mi_row) * 4 * 8;
        prune_intra_mode_with_hog_y(
            env.src_y,
            ref_off_y,
            env.stride,
            bsize,
            mb_right,
            mb_bottom,
            -1.2,
            &mut skip_mask,
        );
    }
    let mut gates = IntraSbyGates::speed0(skip_mask);
    // Speed-2 all-intra intra-mode deltas (set_allintra_speed_features_framesize
    // _independent, speed_features.c:429/431): prune SMOOTH_H_PRED / SMOOTH_V_PRED
    // from the luma mode search (disable_smooth_intra), and restrict the
    // filter-intra search to the FILTER modes derived from the best-so-far Y mode
    // (prune_filter_intra_level=1). Both are inert at speed<2 (IntraSbyGates::speed0
    // leaves them off), so speed 0/1 byte-match gates are unaffected. Only applied
    // for allintra: GOOD has its own disable_smooth_intra schedule and, in this
    // port's envelope, always runs at speed 0.
    if cfg.allintra && cfg.speed >= 2 {
        gates.disable_smooth_intra = true;
        // C: 1 at speed>=2, 2 at speed>=4 (unmodeled). 1 covers speed 2/3.
        gates.prune_filter_intra_level = 1;
    }

    // Neighbour winner modes (module docs: the mi-grid reads).
    let above_mode = if up_available {
        Some(i32::from(grid.at(mi_row - 1, mi_col)))
    } else {
        None
    };
    let left_mode = if left_available {
        Some(i32::from(grid.at(mi_row, mi_col - 1)))
    } else {
        None
    };

    // Luma intra edge filter type (reconintra.c get_intra_edge_filter_type):
    // 1 iff the above OR left neighbour is a SMOOTH mode (SMOOTH_PRED=9,
    // SMOOTH_V_PRED=10, SMOOTH_H_PRED=11). C re-derives this per block from
    // the live mode-info grid; the frozen SB-level `env.filter_type` misses
    // sub-block neighbours committed during the partition search (e.g. a
    // SMOOTH VERT_4 strip-0 feeding strip-1's angled prediction).
    let is_smooth_luma = |m: Option<i32>| m.is_some_and(|md| (9..=11).contains(&md));
    let luma_edge_filter_type = i32::from(is_smooth_luma(above_mode) || is_smooth_luma(left_mode));

    // skip ctx: every KEY intra neighbour has skip_txfm == 0 => ctx 0.
    let skip_ctx = 0usize;
    let tx_size_ctx = get_tx_size_context(
        bsize,
        tile.above_tctx[a0],
        tile.left_tctx[l0],
        up_available,
        left_available,
        None,
        None,
    );

    let above_y: Vec<i8> = tile.above_ectx[0][a0..a0 + mi_w].to_vec();
    let left_y: Vec<i8> = tile.left_ectx[0][l0..l0 + mi_h].to_vec();
    let mut y_env = TxfmYrdEnv {
        sb_size: env.sb_size,
        bsize,
        mi_row,
        mi_col,
        up_available,
        left_available,
        tile_col_end: env.tile_col_end,
        tile_row_end: env.tile_row_end,
        partition,
        mi_cols: env.mi_cols,
        mi_rows: env.mi_rows,
        ref_off: ref_off_y,
        ref_stride: env.stride,
        src: env.src_y,
        src_off: ref_off_y,
        src_stride: env.stride,
        disable_edge_filter: env.disable_edge_filter,
        filter_type: luma_edge_filter_type,
        mode: 0,
        angle_delta: 0,
        use_filter_intra: false,
        filter_intra_mode: 0,
        lossless: env.lossless,
        reduced_tx_set_used: env.reduced_tx_set_used,
        bd: env.bd,
        rows: env.rows_y,
        rdmult: env.rdmult,
        coeff_costs: env.coeff_costs_y,
        tx_type_costs: cfg.tx_type_costs_y,
        skip_costs: cfg.skip_costs,
        skip_ctx,
        tx_size_costs: cfg.tx_size_costs,
        tx_size_ctx,
        // select_tx_mode (rdopt_utils.h): coded_lossless => cm->features.tx_mode
        // = ONLY_4X4, never TX_MODE_SELECT (`av1/decoder/decodeframe.c:141`'s
        // `read_tx_mode` returns ONLY_4X4 unconditionally when coded_lossless).
        // This port doesn't model TX_MODE_LARGEST (the other ONLY_4X4
        // alternative), so within this envelope TX_MODE_SELECT holds exactly
        // when NOT lossless. Was hardcoded `true` regardless of `env.lossless`
        // -- harmless while every caller also hardcoded `lossless: false`, but
        // a real bug once qindex=0 correctly threads `lossless: true`: it
        // would violate `pick_uniform_tx_size_type_yrd_intra`'s own "lossless
        // implies ONLY_4X4" debug_assert, and feeds a nonzero tx-size-signal
        // rate cost (`tx_search.rs`'s `tx_select` gate) into the RD search for
        // a symbol the pack stage never actually writes at lossless
        // (`pack.rs`'s `cfg.tx_mode_is_select && !env.lossless` gate).
        tx_mode_is_select: !env.lossless,
        above_ctx: &above_y,
        left_ctx: &left_y,
    };
    let sby_cfg = IntraSbySearchCfg {
        gates: &gates,
        // `intra_sf.top_intra_model_count_allowed` (speed_features.c:2443 default
        // TOP_INTRA_MODEL_COUNT=4; :404 -> 3 at allintra speed>=1; :533 -> 2 at
        // speed>=4, unmodeled). Drives the `top_intra_model_rd[]` slot
        // `prune_intra_y_mode` compares against (index = count-1). At speed 0/1
        // the byte-match gates were inert to 4-vs-3 (the mode set kept fewer than
        // 4 competitive models); the speed-2 `disable_smooth_intra` prune shrinks
        // the mode set enough that the 4-vs-3 slot difference tips the winner, so
        // this must be the correct 3 at speed>=1 to byte-match.
        top_intra_model_count_allowed: if cfg.allintra && cfg.speed >= 1 { 3 } else { 4 },
        adapt_top_model_rd_count_using_neighbors: false,
        above_mode,
        left_mode,
        qindex: cfg.qindex,
        mode_costs: cfg.mode_costs,
        // av1_allow_palette(cm->features.allow_screen_content_tools, bsize):
        // adds the palette-Y no-palette flag cost to DC_PRED leaves in
        // screen-content frames (the palette SEARCH stays out of scope --
        // try_palette only feeds intra_mode_info_cost_y's flag-cost term, it
        // does not enable a palette candidate). palette_mode_ctx counts
        // neighbours that use a Y palette; this port never picks palette, so
        // every neighbour's palette_size[0] is 0 and the context is always 0
        // (computed via the real helper with the known-zero neighbour sizes so
        // the invariant is explicit, not silently assumed).
        try_palette: allow_palette(cfg.allow_screen_content_tools, bsize),
        palette_bsize_ctx: palette_bsize_ctx(bsize) as usize,
        palette_mode_ctx: palette_mode_ctx(up_available, 0, left_available, 0) as usize,
        enable_filter_intra: cfg.enable_filter_intra,
        allow_intrabc: false,
        pol: cfg.pol,
        source_variance,
        enable_tx64: cfg.enable_tx64,
        enable_rect_tx: cfg.enable_rect_tx,
        allintra: cfg.allintra,
        speed: cfg.speed,
        mb_to_right_edge: (env.mi_cols - mi_w as i32 - mi_col) * 4 * 8,
        mb_to_bottom_edge: (env.mi_rows - mi_h as i32 - mi_row) * 4 * 8,
    };
    let mut var_cache = Block4x4VarInfo::sb_cache(env.sb_size);

    // Chroma args (num_planes > 1).
    let ref_off_uv = chroma_plane_offset(
        env.base_uv,
        env.stride,
        mi_row,
        mi_col,
        bsize,
        env.ss_x,
        env.ss_y,
    );
    let mut chroma_up_available = up_available;
    let mut chroma_left_available = left_available;
    if env.ss_x != 0 && mi_w < 2 {
        chroma_left_available = (mi_col - 1) > env.tile_col_start;
    }
    if env.ss_y != 0 && mi_h < 2 {
        chroma_up_available = (mi_row - 1) > env.tile_row_start;
    }
    let plane_bsize = get_plane_block_size(bsize, env.ss_x, env.ss_y);
    let (pmw, pmh) = (MI_SIZE_WIDE_B[plane_bsize], MI_SIZE_HIGH_B[plane_bsize]);
    let au = (mi_col >> env.ss_x) as usize;
    let lu = ((mi_row & 31) >> env.ss_y) as usize;
    let above_u: Vec<i8> = tile.above_ectx[1][au..au + pmw].to_vec();
    let left_u: Vec<i8> = tile.left_ectx[1][lu..lu + pmh].to_vec();
    let above_v: Vec<i8> = tile.above_ectx[2][au..au + pmw].to_vec();
    let left_v: Vec<i8> = tile.left_ectx[2][lu..lu + pmh].to_vec();
    const BLK_W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const BLK_H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    let cfl_allowed = !env.lossless && BLK_W[bsize] <= 32 && BLK_H[bsize] <= 32;
    // Chroma has no tx-size depth search (av1_get_tx_size_uv is a pure
    // function of bsize/lossless/subsampling) -- pre-select the ONE real
    // per-txs_ctx table this leaf's whole UV search+encode lifetime uses,
    // matching `av1_get_tx_size(AOM_PLANE_U, xd)` re-derived later inside
    // `rd_pick_intra_mode_sb` (same inputs, same value).
    let uv_tx_size = av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
    let uv_coeff_tables = env.coeff_costs_uv.tables(uv_tx_size);

    // Per-block CHROMA intra-edge-filter type (`get_intra_edge_filter_type(xd,
    // plane=1)`, reconintra.c:974): 1 iff the chroma above OR left neighbour's
    // `uv_mode` is a SMOOTH mode (UV_SMOOTH_PRED=9 / UV_SMOOTH_V=10 /
    // UV_SMOOTH_H=11). This is the chroma analogue of the luma
    // `luma_edge_filter_type` recompute above (KB-2); the frozen SB-level
    // `env.filter_type` (always 0) was the same pre-KB-2 bug on the chroma
    // plane. The chroma neighbour mbmi is the chroma-reference mi of the
    // above/left chroma unit (av1_common_int.h:1400-1416): from the block's
    // top-left-most covered luma mi `base = (mi_row - (mi_row & ss_y),
    // mi_col - (mi_col & ss_x))`, `above = base + (-1, +ss_x)` and
    // `left = base + (+ss_y, -1)`. The `chroma_*_available` flags mirror C's
    // NULL-neighbour guard; the in-frame check is a panic-safety net (a
    // neighbour outside the frame reads as DC_PRED=0 = not smooth).
    let is_smooth_uv = |uvm: u8| (9..=11).contains(&uvm);
    let base_row = mi_row - (mi_row & env.ss_y as i32);
    let base_col = mi_col - (mi_col & env.ss_x as i32);
    let uv_mode_at = |r: i32, c: i32| -> u8 {
        if r >= 0 && c >= 0 && r < env.mi_rows && c < env.mi_cols {
            grid.at_uv(r, c)
        } else {
            0
        }
    };
    let chroma_edge_filter_type = i32::from(
        (chroma_up_available
            && is_smooth_uv(uv_mode_at(base_row - 1, base_col + env.ss_x as i32)))
            || (chroma_left_available
                && is_smooth_uv(uv_mode_at(base_row + env.ss_y as i32, base_col - 1))),
    );

    let mut uv_env = UvRdEnv {
        sb_size: env.sb_size,
        bsize,
        mi_row,
        mi_col,
        chroma_up_available,
        chroma_left_available,
        tile_col_end: env.tile_col_end,
        tile_row_end: env.tile_row_end,
        partition,
        mi_cols: env.mi_cols,
        mi_rows: env.mi_rows,
        ss_x: env.ss_x,
        ss_y: env.ss_y,
        ref_off: [ref_off_uv, ref_off_uv],
        ref_stride: env.stride,
        src_u: env.src_u,
        src_v: env.src_v,
        src_off: [ref_off_uv, ref_off_uv],
        src_stride: env.stride,
        disable_edge_filter: env.disable_edge_filter,
        filter_type: chroma_edge_filter_type,
        luma_mode: 0,
        luma_use_fi: false,
        luma_fi_mode: 0,
        lossless: env.lossless,
        reduced_tx_set_used: env.reduced_tx_set_used,
        bd: env.bd,
        rows_u: env.rows_u,
        rows_v: env.rows_v,
        rdmult: env.rdmult,
        coeff_costs: &uv_coeff_tables,
        tx_type_costs: env.tx_type_costs,
        above_ctx: [&above_u, &above_v],
        left_ctx: [&left_u, &left_v],
    };

    let re = ReencodeParams {
        sharpness: env.sharpness,
        enable_optimize_b: env.enable_optimize_b,
    };
    let outcome = {
        let uv_args = if env.monochrome {
            None
        } else {
            Some(RdPickUvArgs {
                env: &mut uv_env,
                recon_u,
                recon_v,
                cfl,
                is_chroma_ref,
                cfl_allowed,
                intra_uv_mode_cost: cfg.intra_uv_mode_cost,
                costs: cfg.mode_costs,
                cfl_costs: cfg.cfl_costs,
                lp: cfg.uv_lp,
            })
        };
        rd_pick_intra_mode_sb(
            &mut y_env,
            recon_y,
            &sby_cfg,
            &mut var_cache,
            best_rd.rdcost,
            env.coeff_costs_y,
            re,
            uv_args,
        )
    };

    match outcome.best {
        None => {
            // rd_cost->rate == INT_MAX -> rdcost = INT64_MAX
            // (pick_sb_modes:969-970). x->source_variance was STILL set
            // above (unconditionally, before the mode search could fail) --
            // returned regardless of the loss.
            (PartRdStats::invalid(), None, source_variance)
        }
        Some(best) => {
            let stats = PartRdStats {
                rate: best.rate,
                dist: best.dist,
                rdcost: best.rdcost,
            };
            let (uv_mode, angle_delta_uv, cfl_alpha_idx, cfl_alpha_signs) = match &best.uv {
                RdPickUvOutcome::Searched(w, _) => (
                    w.uv_mode,
                    w.angle_delta_uv,
                    i32::from(w.cfl_alpha_idx),
                    i32::from(w.cfl_alpha_signs),
                ),
                // !chroma_ref / monochrome: the uv mbmi fields are dead
                // state (nothing reads them — store_cfl_required only reads
                // uv_mode on chroma-ref blocks; packing is chroma-ref-gated).
                _ => (0, 0, 0, 0),
            };
            let winner = LeafWinner {
                bsize,
                mode: best.y.mode,
                angle_delta_y: best.y.angle_delta,
                use_filter_intra: best.y.use_filter_intra,
                filter_intra_mode: best.y.filter_intra_mode,
                tx_size: best.y.tx_size,
                luma_edge_filter_type,
                uv_mode,
                angle_delta_uv,
                cfl_alpha_idx,
                cfl_alpha_signs,
                uv_edge_filter_type: chroma_edge_filter_type,
                tx_type_map: best.tx_type_map,
                skip_txfm: false,
                raw_rdstats: stats,
            };
            (stats, Some(winner), source_variance)
        }
    }
}

/// `should_do_dry_run_encode_for_current_block` (partition_search.c:5556).
pub fn should_do_dry_run_encode(
    sb_size: usize,
    max_partition_size: usize,
    pc_index: usize,
    bsize: usize,
) -> bool {
    if bsize > max_partition_size {
        return false;
    }
    if pc_index != 3 {
        return true;
    }
    let sub_sb_size = split_subsize(sb_size);
    bsize == max_partition_size && sub_sb_size != max_partition_size
}

/// The saved node context (`av1_save_context`, encodeframe_utils.c:579):
/// per-plane above/left entropy + partition + txfm context slices over the
/// node extent.
struct SavedCtx {
    above_e: [Vec<i8>; 3],
    left_e: [Vec<i8>; 3],
    above_p: Vec<i8>,
    left_p: Vec<i8>,
    above_t: Vec<u8>,
    left_t: Vec<u8>,
}

fn save_context(
    tile: &TileCtxState,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
) -> SavedCtx {
    let w = MI_SIZE_WIDE_B[bsize];
    let h = MI_SIZE_HIGH_B[bsize];
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;
    SavedCtx {
        above_e: [
            tile.above_ectx[0][a0..a0 + w].to_vec(),
            tile.above_ectx[1][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].to_vec(),
            tile.above_ectx[2][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].to_vec(),
        ],
        left_e: [
            tile.left_ectx[0][l0..l0 + h].to_vec(),
            tile.left_ectx[1][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].to_vec(),
            tile.left_ectx[2][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].to_vec(),
        ],
        above_p: tile.above_pctx[a0..a0 + w].to_vec(),
        left_p: tile.left_pctx[l0..l0 + h].to_vec(),
        above_t: tile.above_tctx[a0..a0 + w].to_vec(),
        left_t: tile.left_tctx[l0..l0 + h].to_vec(),
    }
}

fn restore_context(
    tile: &mut TileCtxState,
    saved: &SavedCtx,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
) {
    let w = MI_SIZE_WIDE_B[bsize];
    let h = MI_SIZE_HIGH_B[bsize];
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;
    tile.above_ectx[0][a0..a0 + w].copy_from_slice(&saved.above_e[0]);
    tile.above_ectx[1][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].copy_from_slice(&saved.above_e[1]);
    tile.above_ectx[2][a0 >> ss_x..(a0 >> ss_x) + (w >> ss_x)].copy_from_slice(&saved.above_e[2]);
    tile.left_ectx[0][l0..l0 + h].copy_from_slice(&saved.left_e[0]);
    tile.left_ectx[1][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].copy_from_slice(&saved.left_e[1]);
    tile.left_ectx[2][l0 >> ss_y..(l0 >> ss_y) + (h >> ss_y)].copy_from_slice(&saved.left_e[2]);
    tile.above_pctx[a0..a0 + w].copy_from_slice(&saved.above_p);
    tile.left_pctx[l0..l0 + h].copy_from_slice(&saved.left_p);
    tile.above_tctx[a0..a0 + w].copy_from_slice(&saved.above_t);
    tile.left_tctx[l0..l0 + h].copy_from_slice(&saved.left_t);
}

/// `rd_pick_rect_partition` (partition_search.c:3471): one rect sub-block
/// pick — the `best - sum` budget subtraction, the leaf at
/// `partition_type`, and the `sum_rdc` accumulation (rate `INT_MAX` -> sum
/// rdcost `INT64_MAX`). Returns `(this_rdc.rdcost — the `rect_part_rd`
/// record, the winner, x->source_variance as of this call — gotcha #1,
/// module docs on [`leaf_pick_sb_modes`])`.
#[allow(clippy::too_many_arguments)]
fn rd_pick_rect_partition(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &TileCtxState,
    grid: &ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    mi_row: i32,
    mi_col: i32,
    subsize: usize,
    partition_type: usize,
    best_rdc: &PartRdStats,
    sum_rdc: &mut PartRdStats,
    visits: &mut Vec<LeafVisit>,
) -> (i64, Option<LeafWinner>, u32) {
    let best_remain = rd_stats_subtraction(env.rdmult, best_rdc, sum_rdc);
    let (this_rdc, winner, source_variance) = leaf_pick_sb_modes(
        env,
        cfg,
        tile,
        grid,
        recon_y,
        recon_u,
        recon_v,
        cfl,
        mi_row,
        mi_col,
        subsize,
        partition_type,
        &best_remain,
    );
    visits.push(LeafVisit {
        mi_row,
        mi_col,
        bsize: subsize,
        budget: best_remain.rdcost,
        rate: this_rdc.rate,
        dist: this_rdc.dist,
        rdcost: this_rdc.rdcost,
    });
    // (av1_rd_cost_update(x->rdmult, &this_rdc) at :3487 — a no-op on the
    // leaf's already-consistent rdcost, as at the NONE stage.)
    if this_rdc.rate == i32::MAX {
        sum_rdc.rdcost = i64::MAX;
    } else {
        sum_rdc.rate += this_rdc.rate;
        sum_rdc.dist += this_rdc.dist;
        rd_cost_update(env.rdmult, sum_rdc);
    }
    (this_rdc.rdcost, winner, source_variance)
}

/// `rd_pick_4partition` (partition_search.c:3919): the HORZ_4/VERT_4
/// sub-block loop — 4 equal strips, sequential RD budget accumulation
/// reusing [`rd_pick_rect_partition`] as the per-leaf primitive (its
/// budget-subtraction + accumulate-or-invalidate shape is exactly
/// `rd_try_subblock`, already partition-type/position-generic), with
/// dry-run propagation after each non-last subblock (`is_last = i==3`) —
/// mirrors the rect stage's own sub-0-then-sub-1 propagation, generalized
/// to 4. Returns `(sum_rdc, Some(winners))` only on a genuine win
/// (`sum_rdc.rdcost < best_rdc.rdcost` after the FINAL `av1_rd_cost_update`,
/// :3962-3963) — callers don't need a redundant outer check.
///
/// Interior-envelope simplification (matching the existing Horz/Vert
/// interior-only scope): callers must only invoke this when all 4
/// quarter-strips are guaranteed to fit in-frame (`mi_row/col +
/// 3*quarter_step` within bounds) — the C's own per-i frame-bound trim
/// (`:3947-3948`, which can code fewer than 4 leaves at a frame edge) is
/// NOT modelled; an edge 4-way candidate is simply not attempted (next
/// lift, matching "the edge-block partition-cost override + edge rect"
/// already listed as out of scope for HORZ/VERT).
#[allow(clippy::too_many_arguments)]
fn rd_pick_4partition(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &mut TileCtxState,
    grid: &mut ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    mi_row: i32,
    mi_col: i32,
    subsize: usize,
    partition_type: usize, // 8 = PARTITION_HORZ_4, 9 = PARTITION_VERT_4
    is_horz4: bool,
    quarter_step: i32,
    partition_cost: &[i32; 10],
    best_rdc: &PartRdStats,
    visits: &mut Vec<LeafVisit>,
    last_source_variance: &mut u32,
) -> (PartRdStats, Option<Box<[LeafWinner; 4]>>) {
    // set_4_part_ctx_and_rdcost (:3898-3916).
    let mut sum_rdc = PartRdStats::init();
    sum_rdc.rate = partition_cost[partition_type];
    sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, 0);

    let mut w: [Option<LeafWinner>; 4] = [None, None, None, None];
    #[allow(clippy::needless_range_loop)]
    // i drives position calc + w[i] store/reread, not a simple iterate
    for i in 0..4usize {
        let (r, c) = if is_horz4 {
            (mi_row + (i as i32) * quarter_step, mi_col)
        } else {
            (mi_row, mi_col + (i as i32) * quarter_step)
        };
        debug_assert!(
            if is_horz4 {
                r < env.mi_rows
            } else {
                c < env.mi_cols
            },
            "caller must only invoke rd_pick_4partition when all 4 strips fit (module docs)"
        );
        let (_rd_i, winner, source_variance) = rd_pick_rect_partition(
            env,
            cfg,
            tile,
            grid,
            recon_y,
            recon_u,
            recon_v,
            cfl,
            r,
            c,
            subsize,
            partition_type,
            best_rdc,
            &mut sum_rdc,
            visits,
        );
        // x->source_variance mutates unconditionally on every subblock
        // attempt, win or lose (gotcha #1, module docs on leaf_pick_sb_modes).
        *last_source_variance = source_variance;
        w[i] = winner;
        // rd_try_subblock's own early-bail (:3161-3164), checked by the
        // caller loop here exactly as rd_pick_rect_partition's own caller
        // checks it between sub-blocks.
        if sum_rdc.rdcost >= best_rdc.rdcost {
            return (PartRdStats::invalid(), None);
        }
        if i < 3 {
            // is_last = (i == SUB_PARTITIONS_PART4 - 1) — propagate winner
            // pixels/contexts/mi-grid for the NEXT strip's leaf search.
            let wi = w[i].as_mut().expect("valid sum implies a winner");
            let _ = crate::encode_sb::encode_b_intra_dry(
                env,
                tile,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                wi,
                r,
                c,
                partition_type,
            );
            grid.stamp(r, c, subsize, wi.mode as u8, wi.uv_mode as u8, env.mi_rows, env.mi_cols);
        }
    }
    // Calculate the total cost and update the best partition (:3962-3967).
    rd_cost_update(env.rdmult, &mut sum_rdc);
    if sum_rdc.rdcost >= best_rdc.rdcost {
        return (PartRdStats::invalid(), None);
    }
    let arr: [LeafWinner; 4] =
        w.map(|x| x.expect("interior 4-way envelope: all 4 subblocks present on a win"));
    (sum_rdc, Some(Box::new(arr)))
}

/// `allow_ab_partition_search` (partition_search.c:3992-4020): simplifies at
/// speed 0 KEY to `do_rectangular_split && bsize > BLOCK_8X8 && has_rows &&
/// has_cols` — `ab_bsize_thresh` stays its `BLOCK_8X8` default
/// (`ext_part_eval_based_on_cur_best` is 0 at speed 0 both usages, already
/// established via the 4-way port's own module docs on
/// `partition4_allowed_base`), and `prune_ext_part_state`
/// (`prune_ext_part_none_skippable`) requires `skip_non_sq_part_based_on_
/// none >= 1`, the SAME sf already established dead at speed 0 in the
/// rect-stage module docs — so `!prune_ext_part_state` is always true and
/// omitted.
fn allow_ab_partition_search(
    bsize: usize,
    do_rectangular_split: bool,
    has_rows: bool,
    has_cols: bool,
    best_rdcost: i64,
) -> bool {
    // Do not prune if there is no valid partition (:4002).
    if best_rdcost == i64::MAX {
        return true;
    }
    const BLK_1D: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    do_rectangular_split && BLK_1D[bsize] > BLK_1D[3] && has_rows && has_cols
}

/// `av1_prune_ab_partitions` (partition_strategy.c:1901-2029): the AB gating
/// pipeline — base gate, RD-ratio structural pruning
/// (`prune_ext_partition_types_search_level == 1`, LIVE at speed 0 both
/// usages — STATUS.md's AB plan point 3, unlike 4-way's own consumption of
/// the SAME sf at `== 2` which is dead), then `ml_prune_ab_partition` (LIVE,
/// same `ml_prune_partition` sf 4-way already established live, gated on
/// BOTH rect types being allowed — matches the C's own extra gate).
/// `evaluate_ab_partition_based_on_split` (:2009-2028) is DEAD
/// (`prune_ext_part_using_split_info >= 2`, established 0 at speed 0 in the
/// 4-way chunk) — omitted. `x_source_variance` is the STALE
/// `x->source_variance` (gotcha #1) fed ONLY to the NN, never the
/// structural pruning (which correctly uses `pb_source_variance`).
#[allow(clippy::too_many_arguments)]
fn prune_ab_partitions(
    ext_partition_allowed: bool,
    enable_ab_partitions: bool,
    partition_rect_allowed: [bool; 2],
    pc_tree_partitioning: i32,
    pb_source_variance: u32,
    x_source_variance: u32,
    best_rdcost: i64,
    rect_part_rd: [[i64; 2]; 2],
    split_rd_in: [i64; 4],
    bsize: usize,
) -> [bool; 4] {
    let mut horzab = ext_partition_allowed && enable_ab_partitions && partition_rect_allowed[0];
    let mut vertab = ext_partition_allowed && enable_ab_partitions && partition_rect_allowed[1];

    // level == 1 (LIVE at speed 0 both usages, :1924-1934).
    horzab &= pc_tree_partitioning == 1 // PARTITION_HORZ
        || (pc_tree_partitioning == 0 && pb_source_variance < 32) // NONE
        || pc_tree_partitioning == 3; // SPLIT
    vertab &= pc_tree_partitioning == 2 // PARTITION_VERT
        || (pc_tree_partitioning == 0 && pb_source_variance < 32)
        || pc_tree_partitioning == 3;

    // horz_rd[0]=(horz_rd[0]<INT64_MAX?horz_rd[0]:0) etc (:1941-1948) --
    // MUTATES the underlying rect_part_rd/split_rd storage in the C (the
    // locals there are pointers into part_state's own fields), so the LATER
    // ml_prune_ab_partition call sees these SAME clamped values too, not the
    // raw ones -- replicated here by passing the clamped locals to BOTH the
    // structural pruning below AND predict_ab_partition_prune.
    let clamp = |v: i64| if v < i64::MAX { v } else { 0 };
    let horz_rd = [clamp(rect_part_rd[0][0]), clamp(rect_part_rd[0][1])];
    let vert_rd = [clamp(rect_part_rd[1][0]), clamp(rect_part_rd[1][1])];
    let split_rd = [
        clamp(split_rd_in[0]),
        clamp(split_rd_in[1]),
        clamp(split_rd_in[2]),
        clamp(split_rd_in[3]),
    ];

    let mut allowed = [false; 4]; // [HORZ_A, HORZ_B, VERT_A, VERT_B]
    allowed[0] = horzab;
    allowed[1] = horzab;
    let horz_a_rd = horz_rd[1] + split_rd[0] + split_rd[1];
    let horz_b_rd = horz_rd[0] + split_rd[2] + split_rd[3];
    // case 1 (level == 1): /16*14 (:1961-1962).
    allowed[0] &= horz_a_rd / 16 * 14 < best_rdcost;
    allowed[1] &= horz_b_rd / 16 * 14 < best_rdcost;

    allowed[2] = vertab;
    allowed[3] = vertab;
    let vert_a_rd = vert_rd[1] + split_rd[0] + split_rd[2];
    let vert_b_rd = vert_rd[0] + split_rd[1] + split_rd[3];
    allowed[2] &= vert_a_rd / 16 * 14 < best_rdcost;
    allowed[3] &= vert_b_rd / 16 * 14 < best_rdcost;

    // ml_prune_ab_partition (:1995-2004): only when BOTH rect types allowed.
    if enable_ab_partitions
        && ext_partition_allowed
        && partition_rect_allowed[0]
        && partition_rect_allowed[1]
    {
        allowed = crate::ab_nn_prune::predict_ab_partition_prune(
            bsize,
            pc_tree_partitioning,
            x_source_variance,
            best_rdcost,
            [horz_rd, vert_rd],
            split_rd,
            allowed,
        );
    }
    allowed
}

/// `rd_pick_ab_part` + `rd_test_partition3` (partition_search.c:3650-3692,
/// 3177-3221) fused: evaluate ONE AB partition type's 3 sub-blocks, reusing
/// [`rd_pick_rect_partition`] as the per-subblock primitive (STATUS.md's AB
/// plan) — early-bail after EVERY sub-block (not just the last), dry-run
/// propagation after sub-blocks 0 and 1 (not 2, the last). `ab_type`:
/// 0=HORZ_A, 1=HORZ_B, 2=VERT_A, 3=VERT_B (`AB_PART_TYPE` order;
/// `PARTITION_TYPE` value = `ab_type + 4`). `reuse[i]`: `Some(winner)` when
/// `reuse_prev_rd_results_for_part_ab` applies to sub-block `i` (only i=0/1
/// are ever reused, matching the C's own `is_ctx_ready[..][0/1]` shape;
/// sub-block 2 is NEVER reused) — copies `winner.raw_rdstats` verbatim into
/// the accumulation instead of running a fresh search (`pick_sb_modes`'s
/// `rd_mode_is_ready` early-return, partition_search.c:854-861: no budget
/// re-check, the caller's own accumulate-then-compare-to-best_rdc is what
/// decides if it still fits) and does NOT touch `x->source_variance` (the
/// C's own early return skips the assignment that lives later in
/// `pick_sb_modes` — gotcha #1's mechanism doesn't fire for a reused
/// sub-block).
#[allow(clippy::too_many_arguments)]
fn rd_pick_ab_part(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &mut TileCtxState,
    grid: &mut ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    ab_type: usize,
    partition_cost: &[i32; 10],
    best_rdc: &PartRdStats,
    visits: &mut Vec<LeafVisit>,
    last_source_variance: &mut u32,
    reuse: [Option<&LeafWinner>; 2],
) -> (PartRdStats, Option<Box<[LeafWinner; 3]>>) {
    let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
    let bsize2 = split_subsize(bsize);
    let partition_type = 4 + ab_type; // PARTITION_HORZ_A..VERT_B
    let subsize = get_partition_subsize(bsize, partition_type as i32) as usize;

    // ab_subsize/ab_mi_pos (partition_search.c:3805-3831).
    let (positions, sizes): ([(i32, i32); 3], [usize; 3]) = match ab_type {
        0 => (
            // HORZ_A: top-left quarter, top-right quarter, bottom half.
            [
                (mi_row, mi_col),
                (mi_row, mi_col + hbs),
                (mi_row + hbs, mi_col),
            ],
            [bsize2, bsize2, subsize],
        ),
        1 => (
            // HORZ_B: top half, bottom-left quarter, bottom-right quarter.
            [
                (mi_row, mi_col),
                (mi_row + hbs, mi_col),
                (mi_row + hbs, mi_col + hbs),
            ],
            [subsize, bsize2, bsize2],
        ),
        2 => (
            // VERT_A: top-left quarter, bottom-left quarter, right half.
            [
                (mi_row, mi_col),
                (mi_row + hbs, mi_col),
                (mi_row, mi_col + hbs),
            ],
            [bsize2, bsize2, subsize],
        ),
        3 => (
            // VERT_B: left half, top-right quarter, bottom-right quarter.
            [
                (mi_row, mi_col),
                (mi_row, mi_col + hbs),
                (mi_row + hbs, mi_col + hbs),
            ],
            [subsize, bsize2, bsize2],
        ),
        _ => unreachable!("ab_type is 0..4"),
    };

    let mut sum_rdc = PartRdStats::init();
    sum_rdc.rate = partition_cost[partition_type];
    sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, 0);

    let mut w: [Option<LeafWinner>; 3] = [None, None, None];
    for i in 0..3usize {
        let (r, c) = positions[i];
        let sz = sizes[i];
        let winner = if i < 2 && reuse[i].is_some() {
            let reused = reuse[i].expect("checked Some above");
            let this_rdc = reused.raw_rdstats;
            if this_rdc.rate == i32::MAX {
                sum_rdc.rdcost = i64::MAX;
            } else {
                sum_rdc.rate += this_rdc.rate;
                sum_rdc.dist += this_rdc.dist;
                rd_cost_update(env.rdmult, &mut sum_rdc);
            }
            visits.push(LeafVisit {
                mi_row: r,
                mi_col: c,
                bsize: sz,
                budget: 0, // reused: no budget was computed (module docs)
                rate: this_rdc.rate,
                dist: this_rdc.dist,
                rdcost: this_rdc.rdcost,
            });
            Some(reused.clone())
        } else {
            let (_rd_i, winner, source_variance) = rd_pick_rect_partition(
                env,
                cfg,
                tile,
                grid,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                r,
                c,
                sz,
                partition_type,
                best_rdc,
                &mut sum_rdc,
                visits,
            );
            *last_source_variance = source_variance;
            winner
        };
        w[i] = winner;
        // rd_try_subblock's own early-bail (:3161-3164), checked after
        // EVERY sub-block (not just the last).
        if sum_rdc.rdcost >= best_rdc.rdcost {
            return (PartRdStats::invalid(), None);
        }
        if i < 2 {
            // Dry-run propagation after sub-blocks 0 and 1 (not 2, the
            // last) -- even for a REUSED sub-block: the C's own
            // rd_try_subblock calls av1_update_state+encode_superblock
            // unconditionally when `!is_last`, regardless of whether
            // pick_sb_modes took the reuse early-return (module docs).
            let wi = w[i].as_mut().expect("valid sum implies a winner");
            let _ = crate::encode_sb::encode_b_intra_dry(
                env,
                tile,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                wi,
                r,
                c,
                partition_type,
            );
            grid.stamp(r, c, sz, wi.mode as u8, wi.uv_mode as u8, env.mi_rows, env.mi_cols);
        }
    }
    // Calculate the total cost and update the best partition (:3211-3218;
    // matches rd_pick_4partition's own single-check precedent for this
    // "double check" the C's rd_test_partition3 itself performs — both
    // checks compare the SAME av1_rd_cost_update-derived value in this
    // non-negative-accumulated-rate context, so they're not independent).
    rd_cost_update(env.rdmult, &mut sum_rdc);
    if sum_rdc.rdcost >= best_rdc.rdcost {
        return (PartRdStats::invalid(), None);
    }
    let arr: [LeafWinner; 3] = w.map(|x| x.expect("valid sum implies a winner"));
    (sum_rdc, Some(Box::new(arr)))
}

/// `av1_rd_pick_partition` with REAL leaves, NONE + SPLIT + HORZ + VERT —
/// see the module docs for the exact C sequence. `none_rd_out` mirrors the
/// C's `none_rd` out-pointer (the parent's `split_rd[idx]` slot; consumed
/// only by intra-dead NN prunes — threaded for shape). Returns `(winner
/// tree, best stats, found)`; when found and the dry-run gate passes, the
/// winner subtree has been re-encoded (recons + contexts + mode grid
/// stamped) exactly as the C leaves it for siblings.
#[allow(clippy::too_many_arguments)]
/// Extract the 65×65 luma window (stride 65) the intra CNN partition prune runs
/// on: the containing 64×64's `frame(-1,-1)` origin with the frame border
/// edge-replicated (`av1_copy_and_extend_frame`). In this envelope `sb_size ==
/// BLOCK_64X64`, so the containing 64×64 is the SB; its mi origin is `(mi_row,
/// mi_col)` rounded down to the 16-mi (64px) grid. Reads the bd8 source (u16,
/// 0..=255) as u8, clamping to the frame crop (`mi_{rows,cols}*4`, exact for the
/// multiple-of-64 e2e frames — non-multiple crops would need the true width).
fn extract_intra_cnn_window(env: &SbEncodeEnv, mi_row: i32, mi_col: i32) -> Vec<u8> {
    const SB64_MIB: i32 = 16; // BLOCK_64X64 in mi units
    let sb_py = (mi_row / SB64_MIB) * SB64_MIB * 4;
    let sb_px = (mi_col / SB64_MIB) * SB64_MIB * 4;
    let crop_h = env.mi_rows * 4;
    let crop_w = env.mi_cols * 4;
    let mut win = vec![0u8; 65 * 65];
    for i in 0..65i32 {
        let r = (sb_py + i - 1).clamp(0, crop_h - 1) as usize;
        for j in 0..65i32 {
            let c = (sb_px + j - 1).clamp(0, crop_w - 1) as usize;
            win[(i * 65 + j) as usize] = env.src_y[env.base_y + r * env.stride + c] as u8;
        }
    }
    win
}

/// `use_square_partition_only_threshold` for the ALLINTRA path
/// (`set_allintra_speed_feature_framesize_dependent`, speed_features.c:175-316).
/// Returns the BLOCK_SIZE enum value; the caller disables rect for inner blocks
/// with `bsize > threshold`. Framesize tiers: `is_480p_or_larger` = min(w,h) >=
/// 480, `is_720p_or_larger` = min(w,h) >= 720 — bucketed exactly as C.
fn use_square_partition_only_threshold_allintra(speed: i32, w: i32, h: i32) -> usize {
    let min_dim = w.min(h);
    let is_480p_or_larger = min_dim >= 480;
    let is_720p_or_larger = min_dim >= 720;
    // Base, all speeds (:175-182): 128X128 (>=480p) / 64X64 (sub-480p).
    let mut t: usize = if is_480p_or_larger { 15 } else { 12 };
    if speed >= 1 {
        // :211-217: 128X128 (720p+) / 64X64 (480p+) / 32X32 (sub-480p).
        t = if is_720p_or_larger {
            15
        } else if is_480p_or_larger {
            12
        } else {
            9
        };
    }
    if speed >= 2 {
        // :238-242: 64X64 (720p+) / 32X32 (else).
        t = if is_720p_or_larger { 12 } else { 9 };
    }
    if speed >= 6 {
        t = 6; // 16X16 (:315).
    }
    t
}

pub fn rd_pick_partition_real(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &mut TileCtxState,
    grid: &mut ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mut best_rdc: PartRdStats,
    pc_index: usize,
    // `x->part_search_info.quad_tree_idx` — the block's position in the SB's
    // quad-tree (0 at the 64×64 SB root; a SPLIT into child `idx` recurses with
    // `4*quad_tree_idx + idx + 1`, partition_search.c:4574). Feeds the intra
    // CNN partition prune's per-sub-block feature selection.
    quad_tree_idx: i32,
    mut none_rd_out: Option<&mut i64>,
    visits: &mut Vec<LeafVisit>,
    // `x->source_variance` (gotcha #1, module docs on `leaf_pick_sb_modes`):
    // a MACROBLOCK-level (i.e. truly frame-global, not node-scoped) mutable
    // field every leaf search overwrites unconditionally, win or lose. An
    // in/out threading param (mirrors the established `none_rd_out`
    // out-param pattern in this same function): read on entry (whatever the
    // caller's own last leaf search left it at), updated after EVERY leaf
    // search this call makes (NONE, every SPLIT child's own recursion,
    // RECT's sub-0/sub-1, AB's sub-blocks, 4-way's sub-blocks) in
    // chronological C-execution order, so by the time this call returns it
    // holds exactly what the real `x->source_variance` would.
    last_source_variance: &mut u32,
) -> (Option<SbTree>, PartRdStats, bool) {
    // if (none_rd) *none_rd = 0 (:5682).
    if let Some(out) = none_rd_out.as_deref_mut() {
        *out = 0;
    }
    if best_rdc.rdcost < 0 {
        return (None, PartRdStats::invalid(), false);
    }
    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_step = (mi_w / 2) as i32;
    let bsize_at_least_8x8 = bsize >= 3;
    let has_rows = mi_row + mi_step < env.mi_rows;
    let has_cols = mi_col + mi_step < env.mi_cols;
    let mut partition_none_allowed = has_rows && has_cols;
    let mut do_square_split = bsize_at_least_8x8;
    // Rect flag init (:3382-3399) incl. the get_plane_block_size chroma
    // guard (4:2:2 kills VERT-of-8x8 4x8 subsizes; 4:4:0 the HORZ mirror).
    let mut do_rectangular_split = cfg.enable_rect_partitions && bsize_at_least_8x8;
    let mut partition_rect_allowed = [false; 2];
    if do_rectangular_split {
        let horz_subsize = get_partition_subsize(bsize, 1) as usize;
        let vert_subsize = get_partition_subsize(bsize, 2) as usize;
        partition_rect_allowed[0] =
            has_cols && get_plane_block_size(horz_subsize, env.ss_x, env.ss_y) != 255;
        partition_rect_allowed[1] =
            has_rows && get_plane_block_size(vert_subsize, env.ss_x, env.ss_y) != 255;
    }

    // "Disable rectangular partitions for inner blocks when the current block
    // is forced to only use square partitions" (partition_search.c:5749, run
    // after init + the edge-block handling, BEFORE av1_prune_partitions_before
    // _search / the CNN). `use_square_partition_only_threshold` is a
    // framesize-DEPENDENT ALLINTRA speed feature: sub-480p it is BLOCK_64X64 at
    // speed 0 (so `bsize > 64X64` never holds in a <=64X64 SB — a no-op, which
    // is why speed-0 never needed this) but drops to BLOCK_32X32 at speed >= 1,
    // so a 64X64 inner block (has_rows && has_cols) gets its rect partitions
    // killed. `&= !has_rows` (HORZ) / `&= !has_cols` (VERT) leaves the fitting
    // rect on true frame-edge blocks, matching C.
    if cfg.allintra {
        let sq_only_thresh = use_square_partition_only_threshold_allintra(
            cfg.speed,
            env.mi_cols * 4,
            env.mi_rows * 4,
        );
        if bsize > sq_only_thresh {
            partition_rect_allowed[0] &= !has_rows;
            partition_rect_allowed[1] &= !has_cols;
        }
    }
    // prune_rect_part (:3385) / terminate_partition_search (:3380): no live
    // writer in the speed-0 KEY envelope (module docs).
    let prune_rect_part = [false; 2];
    let terminate_partition_search = false;

    // ---- intra CNN partition prune (av1_prune_partitions_before_search ->
    //      intra_mode_cnn_partition, partition_strategy.c:1779-1791). Runs
    //      BEFORE av1_prune_partitions_by_max_min_bsize (the C order at
    //      partition_search.c:5761 then :5765). Gated on the speed-1 sf level
    //      (0 at speed 0 => this whole block is a no-op, speed-0 frozen),
    //      frame-intra (KEY, always here), sb_size >= BLOCK_64X64, bsize in
    //      8×8..=64×64, and the whole block inside the frame. The CNN's only
    //      effect is the four search-space flags. ----
    //
    // `part_sf.intra_cnn_based_part_prune_level` = `SpeedFeatures::set_allintra`
    // (0 at speed 0; `allow_screen_content_tools ? 0 : 2` at speed >= 1, allintra
    // path). Derived from the existing cfg fields so speed-0 (and GOOD) stay
    // frozen — the level is 0 there, making this whole block a no-op.
    let intra_cnn_based_part_prune_level =
        if cfg.allintra && cfg.speed >= 1 && !cfg.allow_screen_content_tools {
            2
        } else {
            0
        };
    if intra_cnn_based_part_prune_level != 0
        && env.sb_size >= 12 // BLOCK_64X64
        && bsize <= 12 // BLOCK_64X64
        && bsize_at_least_8x8
        && mi_row + MI_SIZE_HIGH_B[bsize] as i32 <= env.mi_rows
        && mi_col + MI_SIZE_WIDE_B[bsize] as i32 <= env.mi_cols
    {
        // convert_bsize_to_idx: 64X64->1, 32X32->2, 16X16->3, 8X8->4.
        let bsize_idx = match bsize {
            12 => 1,
            9 => 2,
            6 => 3,
            3 => 4,
            _ => 0,
        };
        if bsize_idx != 0 {
            let win = extract_intra_cnn_window(env, mi_row, mi_col);
            let (_logits, dec) = crate::cnn_partition::decision::predict_decision(
                &win,
                cfg.qindex,
                i32::from(env.bd),
                env.mi_cols * 4,
                env.mi_rows * 4,
                bsize_idx,
                quad_tree_idx,
                intra_cnn_based_part_prune_level,
            );
            // logits[0] > split_thresh: disallow NONE (level != 1) +
            // do_square_split + av1_disable_rect_partitions.
            if dec.none_disallowed {
                partition_none_allowed = false;
            }
            if dec.do_square_split {
                do_square_split = true;
            }
            if dec.rect_disabled {
                do_rectangular_split = false;
                partition_rect_allowed = [false, false];
            }
            // logits[0] < no_split_thresh: av1_disable_square_split_partition.
            if dec.square_split_disabled {
                do_square_split = false;
            }
        }
    }

    // av1_prune_partitions_by_max_min_bsize (partition_strategy.c:1837).
    const BLK_1D: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    if BLK_1D[bsize] > BLK_1D[cfg.max_partition_size] {
        // av1_set_square_split_only (encodeframe_utils.h:266).
        partition_none_allowed = false;
        do_square_split = true;
        do_rectangular_split = false;
        partition_rect_allowed = [false, false];
    } else if BLK_1D[bsize] <= BLK_1D[cfg.min_partition_size] {
        // av1_disable_rect_partitions (encodeframe_utils.h:253) + the
        // le-min square clamp.
        do_rectangular_split = false;
        partition_rect_allowed = [false, false];
        if has_rows && has_cols {
            do_square_split = false;
        }
        partition_none_allowed = !do_square_split;
    }

    // partition_cost[pl_ctx_idx] over the LIVE partition contexts.
    let pl_ctx = if bsize_at_least_8x8 {
        partition_plane_context(
            &tile.above_pctx,
            &tile.left_pctx,
            mi_row as usize,
            mi_col as usize,
            bsize,
        ) as usize
    } else {
        0
    };
    let partition_cost = &cfg.partition_costs[pl_ctx];

    // av1_rd_cost_update(x->rdmult, &best_rdc) (:5744).
    rd_cost_update(env.rdmult, &mut best_rdc);

    // av1_save_context (:5754).
    let saved = save_context(tile, mi_row, mi_col, bsize, env.ss_x, env.ss_y);

    // The per-node ALLINTRA variance arm (:5791-5827; module docs): at
    // speed 0 only the >= BLOCK_16X16 force-split branch is live (the
    // rect-prune sibling needs the speed >= 6
    // prune_rect_part_using_4x4_var_deviation sf).
    if cfg.allintra && bsize >= 6 {
        let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
        let mb_right = (env.mi_cols - mi_w as i32 - mi_col) * 4 * 8;
        let mb_bottom = (env.mi_rows - MI_SIZE_HIGH_B[bsize] as i32 - mi_row) * 4 * 8;
        let (var_min, var_max) = log_sub_block_var(
            env.src_y, ref_off_y, env.stride, bsize, mb_right, mb_bottom, env.bd,
        );
        if var_min < 0.272 && (var_max - var_min) > 3.0 {
            partition_none_allowed = false;
            // terminate_partition_search = 0 (:5817): already false — no
            // live setter in the envelope.
            do_square_split = true;
        }
    }

    let mut found = false;
    let mut best_tree: Option<SbTree> = None;
    // `pc_tree->partitioning` (context_tree.c:150 inits PARTITION_NONE at
    // alloc; each stage overwrites on its own win, :4478/:4628/:3013/etc).
    // Feeds the 4-way ML prune's `part_ctx` feature; the final (4-way-
    // stage) write is unread THIS chunk (AB, which would read it next, is
    // not yet ported) -- kept live rather than deleted since it's exactly
    // what the AB chunk needs.
    let mut pc_tree_partitioning: i32 = 0; // PARTITION_NONE
    // part_search_state->none_rd (:3366; the :4458 store is PRE-pt_cost).
    let mut none_rd: i64 = 0;

    // ---- PARTITION_NONE stage ----
    if partition_none_allowed {
        let mut pt_cost = 0i32;
        if bsize_at_least_8x8 {
            pt_cost = if partition_cost[0] < i32::MAX {
                partition_cost[0]
            } else {
                0
            };
        }
        let mut partition_rdcost = PartRdStats::init();
        partition_rdcost.rate = pt_cost;
        rd_cost_update(env.rdmult, &mut partition_rdcost);
        let best_remain = rd_stats_subtraction(env.rdmult, &best_rdc, &partition_rdcost);

        let (mut this_rdc, winner, source_variance) = leaf_pick_sb_modes(
            env,
            cfg,
            tile,
            grid,
            recon_y,
            recon_u,
            recon_v,
            cfl,
            mi_row,
            mi_col,
            bsize,
            0,
            &best_remain,
        );
        *last_source_variance = source_variance;
        visits.push(LeafVisit {
            mi_row,
            mi_col,
            bsize,
            budget: best_remain.rdcost,
            rate: this_rdc.rate,
            dist: this_rdc.dist,
            rdcost: this_rdc.rdcost,
        });
        // *none_rd / part_search_state->none_rd = this_rdc.rdcost
        // (:4458-4459) — BEFORE the pt_cost fold below.
        none_rd = this_rdc.rdcost;
        if let Some(out) = none_rd_out {
            *out = this_rdc.rdcost;
        }
        // (pick_sb_modes normalized INT_MAX already; av1_rd_cost_update at
        // the stage is folded into the leaf's returned rdcost.)
        if this_rdc.rate != i32::MAX {
            if bsize_at_least_8x8 {
                this_rdc.rate += pt_cost;
                this_rdc.rdcost = crate::rd::rdcost(env.rdmult, this_rdc.rate, this_rdc.dist);
            }
            if this_rdc.rdcost < best_rdc.rdcost {
                best_rdc = this_rdc;
                found = true;
                pc_tree_partitioning = 0; // PARTITION_NONE
                best_tree = Some(SbTree::Leaf(winner.expect("valid rate has a winner")));
            }
        }
        // av1_restore_context at the NONE-stage tail (:4492).
        restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    }

    // split_rd[4] (:3367/:4566): the children's none_rd out-values, zeroed
    // once at init_partition_search_state_params like the C's
    // `av1_zero(part_search_state->split_rd)` (hoisted out of the
    // do_square_split block so it survives -- unmutated, all-zero -- to the
    // 4-way ML prune below when SPLIT doesn't run, exactly matching
    // split_partition_search's own early return leaving it untouched).
    let mut split_rd = [0i64; 4];
    // is_split_ctx_is_ready[0]/[1] + the children's own winner (AB-stage
    // inputs, partition_search.c:4598-4608): survive past this `if
    // do_square_split` block regardless of whether SPLIT ultimately becomes
    // the OVERALL winner (the C sets this bookkeeping unconditionally inside
    // the per-child loop, not gated on SPLIT's own eventual best_rdc
    // comparison).
    let mut is_split_ctx_is_ready = [false; 2];
    let mut split_child_leaf_for_reuse: [Option<LeafWinner>; 2] = [None, None];
    // ---- PARTITION_SPLIT stage ----
    if do_square_split {
        let subsize = split_subsize(bsize);
        let mut sum_rdc = PartRdStats::init();
        sum_rdc.rate = partition_cost[3];
        sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, 0);

        let mut children: Vec<Option<SbTree>> = Vec::new();
        let mut idx = 0usize;
        while idx < 4 && sum_rdc.rdcost < best_rdc.rdcost {
            let y = mi_row + ((idx as i32) >> 1) * mi_step;
            let x = mi_col + ((idx as i32) & 1) * mi_step;
            if y >= env.mi_rows || x >= env.mi_cols {
                children.push(None);
                idx += 1;
                continue;
            }
            let best_remain = rd_stats_subtraction(env.rdmult, &best_rdc, &sum_rdc);
            let (child_tree, child_rdc, child_found) = rd_pick_partition_real(
                env,
                cfg,
                tile,
                grid,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                y,
                x,
                subsize,
                best_remain,
                idx,
                // child quad_tree_idx = 4*parent + idx + 1 (partition_search.c:4574).
                4 * quad_tree_idx + idx as i32 + 1,
                Some(&mut split_rd[idx]),
                visits,
                last_source_variance,
            );
            if !child_found {
                sum_rdc = PartRdStats::invalid();
                children.push(child_tree);
                break;
            }
            sum_rdc.rate += child_rdc.rate;
            sum_rdc.dist += child_rdc.dist;
            rd_cost_update(env.rdmult, &mut sum_rdc);
            // Set split ctx as ready for use (:4598-4608).
            //
            // C: `bsize <= BLOCK_8X8 || pc_tree->split[idx]->partitioning ==
            // PARTITION_NONE` -- the `bsize<=BLOCK_8X8` disjunct only
            // matters when THIS SPLIT's own parent bsize is 8x8 (children
            // 4x4, which structurally can ONLY ever be NONE -- no partition
            // type exists below BLOCK_4X4), so it's equivalent to (not an
            // approximation of) `SbTree::Leaf` alone: whenever the
            // disjunct's left side would fire, the right side (`==
            // PARTITION_NONE`, i.e. `SbTree::Leaf`) is ALREADY
            // unconditionally true for that same child. AB itself never
            // reads this state at a bsize<=8x8 parent anyway
            // (allow_ab_partition_search requires bsize > BLOCK_8X8).
            if idx <= 1
                && let Some(SbTree::Leaf(w)) = child_tree.as_ref()
                && w.uv_mode != UV_CFL_PRED
            {
                is_split_ctx_is_ready[idx] = true;
                split_child_leaf_for_reuse[idx] = Some(w.clone());
            }
            children.push(child_tree);
            idx += 1;
        }
        let reached_last_index = idx == 4;

        if reached_last_index && sum_rdc.rdcost < best_rdc.rdcost {
            // split_partition_penalty_level = 0 => factor 1.0.
            sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, sum_rdc.dist);
            if sum_rdc.rdcost < best_rdc.rdcost {
                best_rdc = sum_rdc;
                found = true;
                pc_tree_partitioning = 3; // PARTITION_SPLIT
                let kids: Vec<SbTree> = children
                    .into_iter()
                    .map(|t| t.expect("found split has 4 found children"))
                    .collect();
                best_tree = Some(SbTree::Split(Box::new(
                    <[SbTree; 4]>::try_from(kids).ok().unwrap(),
                )));
            }
        } else if cfg.less_rectangular_check_level > 0 {
            // The less_rectangular_check arm (:4630-4640; ALLINTRA level 1
            // at speed 0): when SPLIT did not complete-and-beat and
            // (level == 2 || the loop exited at idx <= 2), kill rect if the
            // NONE leaf (PRE-pt_cost none_rd) was valid and beat the
            // split-stage sum.
            if cfg.less_rectangular_check_level == 2 || idx <= 2 {
                let partition_none_valid = none_rd > 0;
                let partition_none_better = none_rd < sum_rdc.rdcost;
                if partition_none_valid && partition_none_better {
                    do_rectangular_split = false;
                }
            }
        }
        // The SPLIT-stage restore (:4645-4647): gated `bsize <=
        // max_partition_size || bsize == sb_size` — always true here.
        debug_assert!(bsize <= cfg.max_partition_size || bsize == env.sb_size);
        restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    }

    // ---- rectangular partition stage (rectangular_partition_search,
    // :3520; wired :5875) ----
    // Between SPLIT and rect the C runs early_term_after_none_split (sf 0),
    // skip_non_sq_part_based_on_none (sf 0) and prune_partitions_after_split
    // (entirely !frame_is_intra_only-gated) — verified no-ops (module docs).
    let mut rect_part_rd = [[0i64; 2]; 2]; // :3368 — an AB-stage input.
    let mut is_rect_ctx_is_ready = [false; 2]; // :3373 — an AB-stage input.
    // The rect stage's own sub-0 winner, captured whenever
    // is_rect_ctx_is_ready[i] gets set -- HORZ_B/VERT_B's own reuse input
    // (`mode_srch_ctx[HORZ_B][0] = &pc_tree->horizontal[0]` unconditionally
    // in the C's `set_mode_search_ctx`, partition_search.c:3698-3699; ACTUAL
    // use still gated by is_rect_ctx_is_ready[i] at the AB stage, matching
    // the C's own `is_ctx_ready[ab_part_type][0]` guard).
    let mut rect_sub0_for_reuse: [Option<LeafWinner>; 2] = [None, None];
    for i in 0..2usize {
        // is_rect_part_allowed (:3506) with av1_active_h/v_edge at the
        // one-pass shape (encodeframe_utils.c:787/817): active iff the
        // node's mi range straddles 0 or the frame mi end.
        let (mi_pos, dim_end) = if i == 0 {
            (mi_row, env.mi_rows)
        } else {
            (mi_col, env.mi_cols)
        };
        let active_edge = (0 >= mi_pos && 0 < mi_pos + mi_step)
            || (dim_end >= mi_pos && dim_end < mi_pos + mi_step);
        if terminate_partition_search
            || !partition_rect_allowed[i]
            || prune_rect_part[i]
            || !(do_rectangular_split || active_edge)
        {
            continue;
        }
        let partition_type = 1 + i; // PARTITION_HORZ / PARTITION_VERT
        let subsize = get_partition_subsize(bsize, partition_type as i32) as usize;
        let mut sum_rdc = PartRdStats::init();
        sum_rdc.rate = partition_cost[partition_type];
        sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, 0);

        // Sub-block 0 at the origin (:3596).
        let (rd0, mut w0, sv0) = rd_pick_rect_partition(
            env,
            cfg,
            tile,
            grid,
            recon_y,
            recon_u,
            recon_v,
            cfl,
            mi_row,
            mi_col,
            subsize,
            partition_type,
            &best_rdc,
            &mut sum_rdc,
            visits,
        );
        *last_source_variance = sv0;
        rect_part_rd[i][0] = rd0;

        // is_not_edge_block[i] (:3550): has_rows for HORZ / has_cols VERT.
        let is_not_edge_block = if i == 0 { has_rows } else { has_cols };
        let mut w1: Option<LeafWinner> = None;
        if sum_rdc.rdcost < best_rdc.rdcost && is_not_edge_block {
            let w0 = w0.as_mut().expect("valid rect sum implies a sub-0 winner");
            // is_rect_ctx_is_ready (:3605-3612): palette-free (envelope:
            // try_palette off) and uv_mode != UV_CFL_PRED.
            if w0.uv_mode != UV_CFL_PRED {
                is_rect_ctx_is_ready[i] = true;
                rect_sub0_for_reuse[i] = Some(w0.clone());
            }
            // av1_update_state + encode_superblock(DRY_RUN_NORMAL)
            // (:3613-3616) — the MID-STAGE propagation: sub 1 reads sub 0's
            // winner pixels, entropy/txfm contexts, and mi-grid modes.
            // encode_b_intra_dry composes exactly update_state +
            // encode_superblock for a KEY intra leaf (no partition-ctx
            // stamp, no rdmult save; ctx->mic already carries the pick's
            // partition — module docs #4).
            let _ = crate::encode_sb::encode_b_intra_dry(
                env,
                tile,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                w0,
                mi_row,
                mi_col,
                partition_type,
            );
            grid.stamp(
                mi_row,
                mi_col,
                subsize,
                w0.mode as u8,
                w0.uv_mode as u8,
                env.mi_rows,
                env.mi_cols,
            );
            // Sub-block 1 at the edge position (mi_row_edge/mi_col_edge =
            // origin + mi_step, :3323-3324).
            let (r1, c1) = if i == 0 {
                (mi_row + mi_step, mi_col)
            } else {
                (mi_row, mi_col + mi_step)
            };
            let (rd1, got, sv1) = rd_pick_rect_partition(
                env,
                cfg,
                tile,
                grid,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                r1,
                c1,
                subsize,
                partition_type,
                &best_rdc,
                &mut sum_rdc,
                visits,
            );
            *last_source_variance = sv1;
            rect_part_rd[i][1] = rd1;
            w1 = got;
        }
        // Best update (:3626-3632).
        if sum_rdc.rdcost < best_rdc.rdcost {
            sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, sum_rdc.dist);
            if sum_rdc.rdcost < best_rdc.rdcost {
                best_rdc = sum_rdc;
                found = true;
                pc_tree_partitioning = 1 + i as i32; // PARTITION_HORZ / PARTITION_VERT
                let pair = Box::new([
                    w0.take().expect("rect winner sub 0"),
                    w1.take().expect("interior rect winner sub 1"),
                ]);
                best_tree = Some(if i == 0 {
                    SbTree::Horz(pair)
                } else {
                    SbTree::Vert(pair)
                });
            }
        }
        // else: rect_part_win_info->rect_part_win[i] = false (:3634-3636) —
        // an AB-stage input (non-NULL only under a SPLIT parent's
        // recursion); next chunk.
        // av1_restore_context at EACH type's loop tail (:3644) — HORZ's
        // sub-0 encode debris restored before VERT evaluates.
        restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    }
    // is_rect_ctx_is_ready + rect_sub0_for_reuse: consumed by the AB stage
    // below (HORZ_B/VERT_B's own reuse input). rect_part_rd is ALSO consumed
    // by the 4-way ML prune further down.

    // ---- AB partition stage (ab_partitions_search, partition_search.c:
    // 3762-3885; wired :5895-5906) ----
    //
    // Real control-flow position: BETWEEN rect and 4-way (rectangular_
    // partition_search -> pb_source_variance -> allow_ab_partition_search ->
    // ab_partitions_search -> prune_4_way_partition_search -> rd_pick_
    // 4partition, partition_search.c:5875-5946) -- placed here, before the
    // existing 4-way stage below, to match.
    if cfg.enable_ab_partitions && !terminate_partition_search {
        // pb_source_variance (:5882-5885): a pure fn of block origin+bsize
        // (av1_get_perpixel_variance_facade) -- identical to the NONE
        // stage's own value and to the 4-way stage's own local computation
        // further down; recomputed here rather than hoisted+shared, to
        // avoid touching the already-shipped 4-way stage's own code
        // (harmless duplication, matches this crate's established per-site
        // convention for small helper computations -- e.g. pack.rs's own
        // PARTITION_* const duplication, module docs).
        let ab_node_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
        let pb_source_variance =
            perpixel_variance_y(env.src_y, ab_node_off_y, env.stride, bsize, env.bd);

        // prune_ext_part_state (prune_ext_part_none_skippable, :3979-3989):
        // requires sf skip_non_sq_part_based_on_none >= 1, established 0 at
        // speed 0 both usages (SAME sf the rect-stage module docs already
        // verified dead) -- always false, omitted from the gate below.
        let ext_partition_allowed = allow_ab_partition_search(
            bsize,
            do_rectangular_split,
            has_rows,
            has_cols,
            best_rdc.rdcost,
        );

        let ab_partitions_allowed = prune_ab_partitions(
            ext_partition_allowed,
            cfg.enable_ab_partitions,
            partition_rect_allowed,
            pc_tree_partitioning,
            pb_source_variance,
            *last_source_variance,
            best_rdc.rdcost,
            rect_part_rd,
            split_rd,
            bsize,
        );

        #[allow(clippy::needless_range_loop)]
        // ab_type selects HORZ_A/HORZ_B/VERT_A/VERT_B throughout (partition
        // type, reuse-source array, SbTree variant), not a simple iterate
        // over ab_partitions_allowed alone.
        for ab_type in 0..4usize {
            if !ab_partitions_allowed[ab_type] {
                continue;
            }
            // is_ctx_ready / set_mode_search_ctx (partition_search.c:
            // 3785-3802, 3695-3709) -- reuse_prev_rd_results_for_part_ab is
            // LIVE unconditionally at speed 0 both usages (STATUS.md plan
            // point 5), so no runtime sf check is needed here (matches the
            // established "hardcode known-always-true sf" convention this
            // file already uses elsewhere, e.g. ml_prune_partition).
            let reuse: [Option<&LeafWinner>; 2] = match ab_type {
                0 => [
                    // HORZ_A
                    is_split_ctx_is_ready[0]
                        .then(|| split_child_leaf_for_reuse[0].as_ref())
                        .flatten(),
                    // C NESTS sub-block 1's reuse inside sub-block 0's
                    // (partition_search.c:3858-3868: `if (is_ctx_ready[..][0]) {
                    // ...; if (is_ctx_ready[..][1]) { ... } }`) — sub-block 1 is
                    // reused ONLY when sub-block 0 is ALSO ready. A split[0] that
                    // sub-split (partitioning != NONE) leaves is_split_ctx_is_ready[0]
                    // false, so C re-searches BOTH top quarters even if split[1]
                    // was a NONE leaf; reusing [1] independently would feed the
                    // split-context winner into the HORZ_A top-right quarter and
                    // skip its fresh (different-neighbour) search.
                    (is_split_ctx_is_ready[0] && is_split_ctx_is_ready[1])
                        .then(|| split_child_leaf_for_reuse[1].as_ref())
                        .flatten(),
                ],
                1 => [
                    // HORZ_B
                    is_rect_ctx_is_ready[0]
                        .then(|| rect_sub0_for_reuse[0].as_ref())
                        .flatten(),
                    None,
                ],
                2 => [
                    // VERT_A
                    is_split_ctx_is_ready[0]
                        .then(|| split_child_leaf_for_reuse[0].as_ref())
                        .flatten(),
                    None,
                ],
                3 => [
                    // VERT_B
                    is_rect_ctx_is_ready[1]
                        .then(|| rect_sub0_for_reuse[1].as_ref())
                        .flatten(),
                    None,
                ],
                _ => unreachable!("ab_type is 0..4"),
            };
            let (sum_rdc, winners) = rd_pick_ab_part(
                env,
                cfg,
                tile,
                grid,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                mi_row,
                mi_col,
                bsize,
                ab_type,
                partition_cost,
                &best_rdc,
                visits,
                last_source_variance,
                reuse,
            );
            if let Some(w) = winners {
                best_rdc = sum_rdc;
                found = true;
                pc_tree_partitioning = (4 + ab_type) as i32;
                best_tree = Some(match ab_type {
                    0 => SbTree::HorzA(w),
                    1 => SbTree::HorzB(w),
                    2 => SbTree::VertA(w),
                    3 => SbTree::VertB(w),
                    _ => unreachable!("ab_type is 0..4"),
                });
            }
            // av1_restore_context at rd_pick_ab_part's OWN tail
            // (partition_search.c:3691), unconditionally per type attempted
            // -- matches the rect stage's own per-type restore.
            restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
        }
    }

    // ---- 4-way partition stage (rd_pick_4partition / prune_4_way_
    // partition_search, partition_search.c:3919/4120; wired :5911-5936) ----
    //
    // prune_ext_part_state (prune_ext_part_none_skippable, :3979-3989):
    // requires sf `skip_non_sq_part_based_on_none >= 1`, which is 0 at
    // speed 0 both usages (the SAME sf field the rect-stage module docs
    // already verified dead there) — so the C's `&& !prune_ext_part_state`
    // factor is always true here and omitted.
    //
    // ext_partition_eval_thresh stays at its av1_reset_part_sf default
    // (BLOCK_8X8) at speed 0 both usages — every override in
    // speed_features.c is gated `if (speed >= 5)` (verified against the
    // checked-in v3.14.1 source). ext_part_eval_based_on_cur_best is 0 at
    // speed 0 both usages (allintra never sets it; good's only setter is
    // `if (speed >= 5)`) so it never raises the threshold to BLOCK_128X128.
    let partition4_allowed_base = cfg.enable_1to4_partitions
        && do_rectangular_split
        && BLK_1D[bsize] > BLK_1D[3] // > BLOCK_8X8
        && has_rows
        && has_cols;
    // prune_part4_search == 2 at speed 0 both usages (verified): disables
    // 4-way when the block's pixel width is below
    // `min_partition_size_1d << 2`.
    let width_ok = BLK_1D[bsize] >= (BLK_1D[cfg.min_partition_size] << 2);
    let mut part4_allowed = [false, false]; // [HORZ4, VERT4]
    // Interior-envelope simplification (module docs on rd_pick_4partition):
    // this port only attempts a 4-way type when ALL 4 quarter-strips are
    // guaranteed in-frame, not just the half-block `has_rows`/`has_cols`
    // extent the C itself checks -- the C's own per-i frame-bound trim
    // (coding fewer than 4 strips at a frame edge) is out of scope, same
    // boundary as the existing interior-only HORZ/VERT scope.
    let quarter_step_mi = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
    let all_4_rows_fit = mi_row + 3 * quarter_step_mi < env.mi_rows;
    let all_4_cols_fit = mi_col + 3 * quarter_step_mi < env.mi_cols;
    if partition4_allowed_base && width_ok {
        let horz4_subsize = get_partition_subsize(bsize, 8) as usize; // PARTITION_HORZ_4
        let vert4_subsize = get_partition_subsize(bsize, 9) as usize; // PARTITION_VERT_4
        part4_allowed[0] = partition_rect_allowed[0]
            && all_4_rows_fit
            && get_plane_block_size(horz4_subsize, env.ss_x, env.ss_y) != 255;
        part4_allowed[1] = partition_rect_allowed[1]
            && all_4_cols_fit
            && get_plane_block_size(vert4_subsize, env.ss_x, env.ss_y) != 255;

        // av1_ml_prune_4_partition (LIVE at speed 0 -- module docs on
        // part4_prune.rs). Only runs when BOTH rect types were allowed
        // (partition_search.c:4191-4193) -- matches the C's own extra gate.
        if partition_rect_allowed[0] && partition_rect_allowed[1] {
            const BLK_W: [usize; 22] = [
                4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
            ];
            const BLK_H: [usize; 22] = [
                4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
            ];
            let node_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
            // pb_source_variance = x->source_variance: identical to the
            // NONE-stage leaf's own value regardless of whether NONE ran
            // (av1_get_perpixel_variance_facade is a pure fn of block
            // origin+bsize -- module docs on av1_rd_pick_partition's
            // pb_source_variance init/fallback, partition_search.c:4457/
            // 5882-5885).
            let pb_source_variance =
                perpixel_variance_y(env.src_y, node_off_y, env.stride, bsize, env.bd);
            let mut horz4_var = [0u32; 4];
            let bh4 = BLK_H[horz4_subsize];
            for (i, v) in horz4_var.iter_mut().enumerate() {
                *v = perpixel_variance_y(
                    env.src_y,
                    node_off_y + i * bh4 * env.stride,
                    env.stride,
                    horz4_subsize,
                    env.bd,
                );
            }
            let mut vert4_var = [0u32; 4];
            let bw4 = BLK_W[vert4_subsize];
            for (i, v) in vert4_var.iter_mut().enumerate() {
                *v = perpixel_variance_y(
                    env.src_y,
                    node_off_y + i * bw4,
                    env.stride,
                    vert4_subsize,
                    env.bd,
                );
            }
            // res_idx = is_480p_or_larger + is_720p_or_larger, from
            // AOMMIN(cm->width, cm->height). Derived from mi_cols/mi_rows*4
            // (the coded frame size) rather than a separate exact-pixel
            // field -- exact for every case this port currently tests
            // (SB-aligned frames); a frame whose true width/height sits
            // strictly between its mi-rounded size and the 480/720
            // boundary would misclassify (documented gap, not silently
            // assumed away).
            let frame_w_px = env.mi_cols * 4;
            let frame_h_px = env.mi_rows * 4;
            let min_dim = frame_w_px.min(frame_h_px);
            let res_idx = usize::from(min_dim >= 480) + usize::from(min_dim >= 720);
            // ml_4_partition_search_level_index (part_sf): 0 at speed 0, then
            // min(speed,3) — 1 at speed>=1 (speed_features.c:210), 2 at speed>=2
            // (:237), 3 at speed>=3 (:271) — mirrors SpeedFeatures::set_allintra.
            // The port's threshold-table path is correct for levels 0/1/2 (and the
            // C table is byte-identical for levels 1 and 2, so this is a byte no-op
            // at speed 2); predict_4partition_prune guards level_index >= 3 (the
            // speed>=3 alternate-NN branch is unmodeled = #10) and returns the
            // inputs unchanged there, so the clamp is safe.
            let ml_4_level_index = cfg.speed.clamp(0, 3);
            let (h4, v4) = crate::part4_prune::predict_4partition_prune(
                bsize,
                pc_tree_partitioning,
                best_rdc.rdcost,
                rect_part_rd,
                split_rd,
                pb_source_variance,
                horz4_var,
                vert4_var,
                res_idx,
                ml_4_level_index,
                part4_allowed[0],
                part4_allowed[1],
            );
            part4_allowed[0] = h4;
            part4_allowed[1] = v4;
        }
    }

    if !terminate_partition_search {
        let quarter_step = quarter_step_mi;
        #[allow(clippy::needless_range_loop)]
        // i selects HORZ4(0)/VERT4(1) throughout, not a simple iterate
        for i in 0..2usize {
            // PARTITION_VERT_4 also requires has_cols at the call site
            // (:5936) -- already implied by partition4_allowed_base above
            // (redundant in C too; kept for fidelity, module docs).
            if !part4_allowed[i] || (i == 1 && !has_cols) {
                continue;
            }
            let partition_type = 8 + i; // PARTITION_HORZ_4 / PARTITION_VERT_4
            let subsize = get_partition_subsize(bsize, partition_type as i32) as usize;
            let (sum_rdc, winners) = rd_pick_4partition(
                env,
                cfg,
                tile,
                grid,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                mi_row,
                mi_col,
                subsize,
                partition_type,
                i == 0,
                quarter_step,
                partition_cost,
                &best_rdc,
                visits,
                last_source_variance,
            );
            if let Some(w) = winners {
                best_rdc = sum_rdc;
                found = true;
                // Unread this chunk (AB, next, is what would read it) --
                // kept live since it's exactly the AB chunk's `part_ctx`.
                #[allow(unused_assignments)]
                {
                    pc_tree_partitioning = partition_type as i32;
                }
                best_tree = Some(if i == 0 {
                    SbTree::Horz4(w)
                } else {
                    SbTree::Vert4(w)
                });
            }
            restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
        }
    }

    // ---- the winner encode (:5998-6026) ----
    if found {
        let tree = best_tree.as_mut().expect("found implies a tree");
        let do_encode = if bsize == env.sb_size {
            // The C emits OUTPUT_ENABLED at the SB root (pack-stage adds;
            // contexts/pixels identical) — modelled as the same DRY_RUN walk.
            true
        } else {
            should_do_dry_run_encode(env.sb_size, cfg.max_partition_size, pc_index, bsize)
        };
        if do_encode {
            let mut leaves: Vec<LeafEncodeOut> = Vec::new();
            encode_sb_dry(
                env,
                tile,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                tree,
                mi_row,
                mi_col,
                bsize,
                &mut leaves,
            );
            // av1_update_state's mi-grid fill: leaf footprints are disjoint,
            // so stamping from the walk's leaf list is order-equivalent.
            stamp_grid_from_tree(grid, tree, mi_row, mi_col, bsize, env.mi_rows, env.mi_cols);
        }
    }

    if found {
        (best_tree, best_rdc, true)
    } else {
        (None, best_rdc, false)
    }
}

fn stamp_grid_from_tree(
    grid: &mut ModeGrid,
    tree: &SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    match tree {
        SbTree::Leaf(w) => {
            grid.stamp(mi_row, mi_col, bsize, w.mode as u8, w.uv_mode as u8, mi_rows, mi_cols);
        }
        SbTree::Split(kids) => {
            let sub = split_subsize(bsize);
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            for (idx, child) in kids.iter().enumerate() {
                stamp_grid_from_tree(
                    grid,
                    child,
                    mi_row + ((idx as i32) >> 1) * hbs,
                    mi_col + ((idx as i32) & 1) * hbs,
                    sub,
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Horz(subs) => {
            let sub = get_partition_subsize(bsize, 1) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(mi_row, mi_col, sub, subs[0].mode as u8, subs[0].uv_mode as u8, mi_rows, mi_cols);
            if mi_row + hbs < mi_rows {
                grid.stamp(
                    mi_row + hbs,
                    mi_col,
                    sub,
                    subs[1].mode as u8,
                    subs[1].uv_mode as u8,
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Vert(subs) => {
            let sub = get_partition_subsize(bsize, 2) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(mi_row, mi_col, sub, subs[0].mode as u8, subs[0].uv_mode as u8, mi_rows, mi_cols);
            if mi_col + hbs < mi_cols {
                grid.stamp(
                    mi_row,
                    mi_col + hbs,
                    sub,
                    subs[1].mode as u8,
                    subs[1].uv_mode as u8,
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Horz4(subs) => {
            // PARTITION_HORZ_4: 4 strips at mi_row + i*quarter_step, i>0
            // gated by the frame bound (module docs; matches encode_sb.rs's
            // encode_sb_dry / pack.rs's pack_sb).
            let sub = get_partition_subsize(bsize, 8) as usize;
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, w) in subs.iter().enumerate() {
                let this_mi_row = mi_row + (i as i32) * quarter_step;
                if i > 0 && this_mi_row >= mi_rows {
                    break;
                }
                grid.stamp(this_mi_row, mi_col, sub, w.mode as u8, w.uv_mode as u8, mi_rows, mi_cols);
            }
        }
        SbTree::Vert4(subs) => {
            // PARTITION_VERT_4: 4 strips at mi_col + i*quarter_step, i>0
            // gated by the frame bound.
            let sub = get_partition_subsize(bsize, 9) as usize;
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, w) in subs.iter().enumerate() {
                let this_mi_col = mi_col + (i as i32) * quarter_step;
                if i > 0 && this_mi_col >= mi_cols {
                    break;
                }
                grid.stamp(mi_row, this_mi_col, sub, w.mode as u8, w.uv_mode as u8, mi_rows, mi_cols);
            }
        }
        SbTree::HorzA(subs) => {
            // PARTITION_HORZ_A: interior-only, no frame-bound gating on any
            // of the 3 sub-blocks (module docs on encode_sb.rs's
            // SbTree::HorzA).
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 4) as usize; // PARTITION_HORZ_A
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(mi_row, mi_col, bsize2, subs[0].mode as u8, subs[0].uv_mode as u8, mi_rows, mi_cols);
            grid.stamp(
                mi_row,
                mi_col + hbs,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col,
                sub,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
        }
        SbTree::HorzB(subs) => {
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 5) as usize; // PARTITION_HORZ_B
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(mi_row, mi_col, sub, subs[0].mode as u8, subs[0].uv_mode as u8, mi_rows, mi_cols);
            grid.stamp(
                mi_row + hbs,
                mi_col,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertA(subs) => {
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 6) as usize; // PARTITION_VERT_A
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(mi_row, mi_col, bsize2, subs[0].mode as u8, subs[0].uv_mode as u8, mi_rows, mi_cols);
            grid.stamp(
                mi_row + hbs,
                mi_col,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row,
                mi_col + hbs,
                sub,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertB(subs) => {
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 7) as usize; // PARTITION_VERT_B
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(mi_row, mi_col, sub, subs[0].mode as u8, subs[0].uv_mode as u8, mi_rows, mi_cols);
            grid.stamp(
                mi_row,
                mi_col + hbs,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                mi_rows,
                mi_cols,
            );
        }
    }
}
