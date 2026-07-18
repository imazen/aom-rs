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
use crate::hog::{prune_intra_mode_with_hog_uv, prune_intra_mode_with_hog_y};
use crate::intra_rd::{Block4x4VarInfo, IntraSbyGates, IntraSbySearchCfg, WinnerModeCfg};
use crate::intra_uv_rd::{
    UV_CFL_PRED, UvLoopPolicy, UvRdEnv, av1_get_tx_size_uv, chroma_plane_offset,
    is_chroma_reference,
};
use crate::mode_costs::TxSizeCosts;
use crate::mode_costs::{CflCosts, IntraModeCosts};
use crate::partition::{PartRdStats, rd_cost_update, rd_stats_subtraction, split_subsize};
use crate::rd_pick::{RdPickUvArgs, RdPickUvOutcome, ReencodeParams, rd_pick_intra_mode_sb};
use crate::speed_features::{MODE_EVAL, SpeedFeatures, WINNER_MODE_EVAL};
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TxTypeSearchPolicy, TxfmYrdEnv};
use aom_dist::highbd_variance;
use aom_entropy::partition::{
    allow_palette, get_partition_subsize, get_plane_block_size, get_tx_size_context,
    is_cfl_allowed, palette_bsize_ctx, palette_mode_ctx, partition_gather_horz_alike,
    partition_gather_vert_alike, partition_plane_context,
};
use aom_intra::cfl::CflCtx;
use aom_txb::{TxTypeCosts, cost_symbol, cost_tokens_from_cdf};

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
    /// Per-mi winner `bsize` (BLOCK_SIZE), stamped alongside `modes`. Read by
    /// the speed>=6 neighbour-bsize prunes: `is_neighbor_blk_larger_than_cur_
    /// blk` (partition_search.c:4352 — the `prune_rect_part_using_none_pred_
    /// mode` DC/SMOOTH arm) and the `prune_sub_8x8_partition_level` gate
    /// (partition_strategy.c:1760-1773) — both `xd->left_mbmi->bsize` /
    /// `xd->above_mbmi->bsize` reads off the live mi grid.
    pub bsizes: Vec<u8>,
    /// Per-mi winner palette state (`mbmi->palette_mode_info` projection:
    /// `[size_y, size_uv]` + the 3×8 colour array), stamped alongside
    /// `modes`. Read by `av1_get_palette_cache` (above/left neighbour colour
    /// merge) and `av1_get_palette_mode_ctx` (neighbour palette-active
    /// flags) in the leaf search. Empty (never indexed) unless the frame
    /// enables the palette search — non-palette frames pay nothing.
    pub pal_sizes: Vec<[u8; 2]>,
    pub pal_colors: Vec<[u16; 24]>,
    /// Per-mi winner DV projection (`mbmi->mv[0]`/`use_intrabc`/`skip_txfm`),
    /// stamped alongside `modes`. Read by the intrabc leaf search's dv-ref
    /// derivation (`find_dv_ref_mvs` neighbour source) and the skip-txfm
    /// context. Empty (never indexed) unless the frame enables the intrabc
    /// search — non-intrabc frames pay nothing (exactly like `pal_sizes`).
    pub dvs: Vec<crate::intrabc_search::DvCell>,
    pub stride: usize,
}

impl ModeGrid {
    /// All-DC grid (harness seeds real neighbour history where relevant).
    pub fn dc(mi_rows: usize, mi_cols: usize) -> Self {
        ModeGrid {
            modes: vec![0; mi_rows * mi_cols],
            uv_modes: vec![0; mi_rows * mi_cols],
            bsizes: vec![0; mi_rows * mi_cols],
            pal_sizes: Vec::new(),
            pal_colors: Vec::new(),
            dvs: Vec::new(),
            stride: mi_cols,
        }
    }
    /// [`Self::dc`] + allocated palette / intrabc-DV neighbour state (screen-
    /// content frames only). `intrabc` allocates the DV grid the leaf search
    /// reads; `palette` allocates the palette-cache/ctx grid.
    pub fn dc_screen(mi_rows: usize, mi_cols: usize, palette: bool, intrabc: bool) -> Self {
        let n = mi_rows * mi_cols;
        ModeGrid {
            modes: vec![0; n],
            uv_modes: vec![0; n],
            bsizes: vec![0; n],
            pal_sizes: if palette { vec![[0; 2]; n] } else { Vec::new() },
            pal_colors: if palette { vec![[0; 24]; n] } else { Vec::new() },
            dvs: if intrabc {
                vec![crate::intrabc_search::DvCell::default(); n]
            } else {
                Vec::new()
            },
            stride: mi_cols,
        }
    }
    /// [`Self::dc`] + allocated palette-neighbour state (palette-search
    /// frames only).
    pub fn dc_with_palette(mi_rows: usize, mi_cols: usize) -> Self {
        Self::dc_screen(mi_rows, mi_cols, true, false)
    }
    #[allow(clippy::too_many_arguments)]
    fn stamp(
        &mut self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        mode: u8,
        uv_mode: u8,
        pal: Option<&crate::palette_search::PaletteYInfo>,
        pal_uv: Option<&crate::palette_search::PaletteUvInfo>,
        dv: crate::intrabc_search::DvCell,
        mi_rows: i32,
        mi_cols: i32,
    ) {
        let rows = (MI_SIZE_HIGH_B[bsize] as i32).min(mi_rows - mi_row) as usize;
        let cols = (MI_SIZE_WIDE_B[bsize] as i32).min(mi_cols - mi_col) as usize;
        // The block's palette projection (Y + UV halves), stamped on EVERY
        // covered mi cell like C's shared mbmi — a no-op (empty vecs) on
        // non-palette frames.
        let mut psz = [0u8; 2];
        let mut pcol = [0u16; 24];
        if let Some(p) = pal {
            psz[0] = p.size as u8;
            pcol[..p.size].copy_from_slice(&p.colors[..p.size]);
        }
        if let Some(p) = pal_uv {
            psz[1] = p.size as u8;
            pcol[8..8 + p.size].copy_from_slice(&p.colors_u[..p.size]);
            pcol[16..16 + p.size].copy_from_slice(&p.colors_v[..p.size]);
        }
        for r in 0..rows {
            let base = (mi_row as usize + r) * self.stride + mi_col as usize;
            self.modes[base..base + cols].fill(mode);
            self.uv_modes[base..base + cols].fill(uv_mode);
            self.bsizes[base..base + cols].fill(bsize as u8);
            if !self.pal_sizes.is_empty() {
                self.pal_sizes[base..base + cols].fill(psz);
                self.pal_colors[base..base + cols].fill(pcol);
            }
            if !self.dvs.is_empty() {
                self.dvs[base..base + cols].fill(dv);
            }
        }
    }
    /// The above/left neighbour DV projection for `find_dv_ref_mvs` — the
    /// `DvNbr` at `(mi_row, mi_col)`. Returns the intra default (`use_intrabc =
    /// false`) when the grid carries no DV state or the cell is out of frame.
    fn dv_at(&self, mi_row: i32, mi_col: i32) -> crate::intrabc_search::DvCell {
        if self.dvs.is_empty() || mi_row < 0 || mi_col < 0 {
            return crate::intrabc_search::DvCell::default();
        }
        self.dvs[mi_row as usize * self.stride + mi_col as usize]
    }
    /// The above/left neighbour palette projection for `av1_get_palette_cache`
    /// / `av1_get_palette_mode_ctx` — `None` when the grid carries no palette
    /// state or the neighbour cell is out of frame.
    fn palette_nbr_at(
        &self,
        mi_row: i32,
        mi_col: i32,
    ) -> Option<aom_entropy::partition::PaletteNbrKf> {
        if self.pal_sizes.is_empty() || mi_row < 0 || mi_col < 0 {
            return None;
        }
        let idx = mi_row as usize * self.stride + mi_col as usize;
        let sz = self.pal_sizes[idx];
        Some(aom_entropy::partition::PaletteNbrKf {
            size: [i32::from(sz[0]), i32::from(sz[1])],
            colors: self.pal_colors[idx],
        })
    }
    fn at(&self, mi_row: i32, mi_col: i32) -> u8 {
        self.modes[mi_row as usize * self.stride + mi_col as usize]
    }
    fn at_uv(&self, mi_row: i32, mi_col: i32) -> u8 {
        self.uv_modes[mi_row as usize * self.stride + mi_col as usize]
    }
    fn bsize_at(&self, mi_row: i32, mi_col: i32) -> usize {
        self.bsizes[mi_row as usize * self.stride + mi_col as usize] as usize
    }
}

/// `is_neighbor_blk_larger_than_cur_blk` (partition_search.c:4352): whether
/// an AVAILABLE left/above neighbour block's pixel AREA exceeds the current
/// block's. The neighbour bsizes are the mi-grid stamps at `(mi_row,
/// mi_col-1)` / `(mi_row-1, mi_col)` (`xd->left_mbmi` / `xd->above_mbmi` —
/// the same live-grid reads as the mode-context / edge-filter neighbours).
fn is_neighbor_blk_larger_than_cur_blk(
    grid: &ModeGrid,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    up_available: bool,
    left_available: bool,
) -> bool {
    const BLK_W: [usize; 22] = [
        4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
    ];
    const BLK_H: [usize; 22] = [
        4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
    ];
    let cur_blk_area = BLK_W[bsize] * BLK_H[bsize];
    if left_available {
        let left = grid.bsize_at(mi_row, mi_col - 1);
        if BLK_W[left] * BLK_H[left] > cur_blk_area {
            return true;
        }
    }
    if up_available {
        let above = grid.bsize_at(mi_row - 1, mi_col);
        if BLK_W[above] * BLK_H[above] > cur_blk_area {
            return true;
        }
    }
    false
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
    /// Raw `fc->partition_cdf[pl_ctx]` rows (`EXT_PARTITION_TYPES + 1` wide).
    /// Read only at frame-EDGE blocks, where `set_partition_cost_for_edge_blk`
    /// (partition_search.c:3411) gathers the CDF to a 2-way split-vs-not
    /// distribution and re-derives the partition cost (the precomputed 10-way
    /// `partition_costs` can't be un-summed). Interior blocks never read this.
    pub partition_cdfs: &'a [[u16; 11]; 20],
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
    /// Frame QM levels (`qmatrix_level_{y,u,v}` from `av1_set_quantizer`),
    /// `None` = QM off (`--enable-qm` default). Threaded into every leaf's
    /// luma + chroma RD search so QM shapes the mode/tx/partition winners
    /// exactly as C (av1_setup_qmatrix runs inside the search's xform_quant).
    pub qm_levels: Option<[usize; 3]>,
    /// The palette size/colour-index cost tables — `Some` models
    /// `oxcf.tool_cfg.enable_palette` (the C default is ON; every
    /// pre-existing byte gate runs the standard shim's `--enable-palette=0`
    /// and passes `None`, keeping those envelopes byte-identical by
    /// construction). The palette SEARCH itself additionally requires
    /// `av1_allow_palette(allow_screen_content_tools, bsize)` per leaf.
    pub palette_costs: Option<&'a crate::mode_costs::PaletteCosts>,
    /// The intrabc (intra-block-copy) leaf-search frame state — `Some` models
    /// `oxcf.kf_cfg.enable_intrabc` on a frame whose header codes
    /// `allow_intrabc` (screen content). Carries the source-frame hash table +
    /// DV signalling costs + var-tx split costs the `rd_pick_intrabc_mode_sb`
    /// arm needs. `None` = every non-screen envelope, byte-stable by
    /// construction (the step-6 arm never runs).
    pub intrabc: Option<IntrabcFrameCfg<'a>>,
    /// CLI intra-tool toggles (`oxcf.intra_mode_cfg` → the LUMA
    /// candidate-loop `enable_*` gates; [`IntraToolCfg`]). `Default` = the
    /// aomenc defaults (all enabled) = the pre-toggle behavior exactly.
    pub intra_tools: IntraToolCfg,
}

/// Frame-constant inputs for the intrabc leaf search (the parts of
/// [`crate::intrabc_search::IntrabcLeafArgs`] that don't vary per leaf). Built
/// once per frame in `pack_tile` when intrabc is enabled.
#[derive(Clone, Copy)]
pub struct IntrabcFrameCfg<'a> {
    /// The source-frame hash table (`build_intrabc_hash_table` from the SOURCE
    /// luma, encodeframe.c:2199).
    pub hash: &'a crate::intrabc_search::IntrabcHashTable,
    /// `x->dv_costs` (`av1_fill_dv_costs` from the frame's ndvc CDFs).
    pub dv_costs: &'a crate::intrabc_search::DvCosts,
    /// `txfm_partition_cost` (`fill_txfm_partition_costs`, rd.c:108).
    pub txfm_partition_costs: [[i32; 2]; 21],
    /// `x->errorperbit = AOMMAX(rdmult >> 6, 1)` (av1_set_error_per_bit).
    pub error_per_bit: i32,
    /// `x->sadperbit` (`av1_set_sad_per_bit`).
    pub sad_per_bit: i32,
    /// `cpi->mv_search_params.mv_step_param` (`av1_init_search_range(max(w,h))`).
    pub mv_step_param: usize,
}

/// `oxcf.intra_mode_cfg` LUMA candidate-loop tool toggles (av1_cx_iface.c
/// `ctrl_set_enable_*`, defaults all ON) — threaded into every leaf's
/// [`IntraSbyGates`] (the chroma loop's copies live on
/// [`UvLoopPolicy`], supplied by the caller via `PickFrameCfg::uv_lp`).
/// Distinct from the sf-driven gate fields (`disable_smooth_intra`,
/// `intra_y_mode_mask`, …): C keeps both and the visit chain
/// (`IntraSbyGates::visits`, intra_mode_search.c:1555-1594) reads them
/// independently. `Default` = the aomenc defaults (all enabled).
///
/// `enable_filter_intra` / `enable_intra_edge_filter` are NOT here — they
/// are SEQUENCE-header bits threaded separately (`PickFrameCfg::
/// enable_filter_intra`, `SbEncodeEnv::disable_edge_filter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntraToolCfg {
    /// `--enable-diagonal-intra` (D45..D203; av1_is_diagonal_mode).
    pub enable_diagonal_intra: bool,
    /// `--enable-directional-intra` (V/H/D45..D67 + angle deltas).
    pub enable_directional_intra: bool,
    /// `--enable-smooth-intra` (SMOOTH/SMOOTH_V/SMOOTH_H).
    pub enable_smooth_intra: bool,
    /// `--enable-paeth-intra` (PAETH_PRED).
    pub enable_paeth_intra: bool,
    /// `--enable-angle-delta` (nonzero deltas on directional modes).
    pub enable_angle_delta: bool,
}

impl Default for IntraToolCfg {
    fn default() -> Self {
        IntraToolCfg {
            enable_diagonal_intra: true,
            enable_directional_intra: true,
            enable_smooth_intra: true,
            enable_paeth_intra: true,
            enable_angle_delta: true,
        }
    }
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
        // `intra_sf.intra_pruning_with_hog` level (allintra: base 1, speed>=2 ->
        // 2, speed>=3 -> 3, speed>=6 -> 4; SpeedFeatures::set_allintra,
        // speed_features.c:360/430/455/531). The C threshold table
        // `{-1.2,-1.2,-0.6,0.4}` (intra_mode_search.c:1505, indexed by
        // `level-1`) maps levels 1,2 -> -1.2, level 3 -> -0.6, level 4 -> 0.4.
        // GOOD (non-allintra) runs at speed 0 in this envelope -> level 1.
        // (Derived inline from cfg.speed, mirroring `disable_smooth_intra` /
        // `top_intra_model_count_allowed` below.)
        let luma_hog_level = if cfg.allintra {
            if cfg.speed >= 6 {
                4
            } else if cfg.speed >= 3 {
                3
            } else if cfg.speed >= 2 {
                2
            } else {
                1
            }
        } else {
            1
        };
        let th = [-1.2f32, -1.2, -0.6, 0.4][luma_hog_level - 1];
        prune_intra_mode_with_hog_y(
            env.src_y,
            ref_off_y,
            env.stride,
            bsize,
            mb_right,
            mb_bottom,
            th,
            &mut skip_mask,
        );
    }
    let mut gates = IntraSbyGates::speed0(skip_mask);
    // CLI intra-tool toggles (oxcf.intra_mode_cfg → the candidate-loop
    // enable_* gates; av1_cx_iface.c defaults all ON). Independent of the
    // sf-driven fields set below — C keeps both and the visit chain reads
    // them separately (IntraSbyGates::visits, intra_mode_search.c:1555-1594).
    gates.enable_diagonal_intra = cfg.intra_tools.enable_diagonal_intra;
    gates.enable_directional_intra = cfg.intra_tools.enable_directional_intra;
    gates.enable_smooth_intra = cfg.intra_tools.enable_smooth_intra;
    gates.enable_paeth_intra = cfg.intra_tools.enable_paeth_intra;
    gates.enable_angle_delta = cfg.intra_tools.enable_angle_delta;
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
        // C: 1 at speed>=2 (:431), 2 at speed>=6 (:529 — level 2 disables the
        // filter-intra search entirely, rd_pick_filter_intra_sby_y's first
        // gate). Speeds 4/5 stay at 1 (no assignment between :431 and :529).
        gates.prune_filter_intra_level = if cfg.speed >= 6 { 2 } else { 1 };
    }
    // `intra_sf.prune_luma_odd_delta_angles_in_intra` (speed_features.c:535)
    // — allintra speed>=6: evens-first delta sweep (`set_y_mode_and_delta_
    // angle` reorder) + the even-neighbour full-RD odd-delta prune
    // (`prune_luma_odd_delta_angles_using_rd_cost`, both already modelled in
    // intra_rd.rs and driven off this gate flag).
    if cfg.allintra && cfg.speed >= 6 {
        gates.prune_luma_odd_delta_angles_in_intra = true;
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
        qindex: cfg.qindex,
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
        // `select_tx_mode` (rdopt_utils.h): coded_lossless → ONLY_4X4;
        // USE_LARGESTALL (--enable-tx-size-search=0 → level 3 at every
        // stage) → TX_MODE_LARGEST; else TX_MODE_SELECT.
        tx_mode_is_select: !env.lossless && cfg.pol.enable_tx_size_search,
        above_ctx: &above_y,
        left_ctx: &left_y,
        qm_levels: cfg.qm_levels,
    };
    // KB-8 (chunk 2d-iv): the speed>=4 all-intra winner-mode two-pass bundle
    // (av1_rd_pick_intra_sby_mode's MODE_EVAL loop → top-3 store_winner_mode_
    // stats → WINNER_MODE_EVAL re-eval). The per-stage policies/methods are
    // derived from the speed features (set_mode_eval_params, rdopt_utils.h:546),
    // threading the caller pol's CLI-driven skip_trellis/sharpness. None below
    // speed 4 (multi_winner_mode_type == MULTI_WINNER_MODE_OFF → single-pass).
    let wm_parts = (cfg.allintra && cfg.speed >= 4).then(|| {
        let mut sf =
            SpeedFeatures::set_allintra(cfg.speed, cfg.allow_screen_content_tools, env.bd > 8);
        // `--enable-tx-size-search=0`: winner_mode_sf.tx_size_search_level = 3
        // AFTER the speed derivation (speed_features.c:2726) → every stage
        // method resolves USE_LARGESTALL.
        if !cfg.pol.enable_tx_size_search {
            sf.tx_size_search_level = 3;
        }
        // multi_winner_mode_type: DEFAULT(2)/FAST(1) at speed 4/5 → the
        // winner-stats loop; OFF(0) at speed >= 6 → count 1 = the
        // single-best re-eval arm with no stats stored (intra_rd.rs).
        let mut mode_eval_pol =
            sf.tx_type_search_policy_for_stage(MODE_EVAL, cfg.pol.skip_trellis, cfg.pol.sharpness);
        let mut winner_pol = sf.tx_type_search_policy_for_stage(
            WINNER_MODE_EVAL,
            cfg.pol.skip_trellis,
            cfg.pol.sharpness,
        );
        // CLI tx-type toggles are stage-INDEPENDENT oxcf reads in C
        // (get_tx_mask) — carry them from the caller's policy onto the
        // derived stage policies; MODE_EVAL additionally ORs the CLI
        // `--use-intra-default-tx-only` knob (rdopt_utils.h:579-581 — the
        // caller's policy carries it; WINNER forces 0, :612).
        for p in [&mut mode_eval_pol, &mut winner_pol] {
            p.enable_flip_idtx = cfg.pol.enable_flip_idtx;
            p.use_intra_dct_only = cfg.pol.use_intra_dct_only;
        }
        mode_eval_pol.use_default_intra_tx_type |= cfg.pol.use_default_intra_tx_type;
        // The stage-derived policies also inherit the frame's tune knobs from
        // cfg.pol (set_mode_eval_params re-derives use_qm_dist_metric from
        // oxcf per stage, rdopt_utils.h:554 — carrying the caller's flags is
        // the same resolution). with_tune_knobs preserves the toggles above.
        let tune = crate::TuneKnobs {
            use_qm_dist_metric: cfg.pol.use_qm_dist_metric,
            iq_tuning: cfg.pol.iq_tuning,
        };
        (
            mode_eval_pol.with_tune_knobs(tune),
            winner_pol.with_tune_knobs(tune),
            sf.tx_size_search_method_for_stage(MODE_EVAL),
            sf.tx_size_search_method_for_stage(WINNER_MODE_EVAL),
            sf.winner_mode_count_allowed(),
            sf.prune_winner_mode_eval_level,
        )
    });
    let wm_cfg = wm_parts
        .as_ref()
        .map(|(me, win, me_m, win_m, count, prune_lvl)| WinnerModeCfg {
            mode_eval_pol: me,
            winner_pol: win,
            mode_eval_tx_size_method: *me_m,
            winner_tx_size_method: *win_m,
            max_winner_count: *count,
            prune_winner_mode_eval_level: *prune_lvl,
        });
    // Above/left neighbour palette projections (xd->above_mbmi/left_mbmi):
    // read from the committed mode grid like the Y/UV neighbour modes above.
    let palette_above = if up_available {
        grid.palette_nbr_at(mi_row - 1, mi_col)
    } else {
        None
    };
    let palette_left = if left_available {
        grid.palette_nbr_at(mi_row, mi_col - 1)
    } else {
        None
    };
    // The palette-search cfg (enable_palette && the sf levels). The sf
    // levels: allintra per SpeedFeatures::set_allintra (speed 0: search
    // level 0 + size-search level 1); GOOD runs speed 0 in this envelope
    // where both are the init_intra_sf defaults (0, 0).
    let palette_cfg = cfg.palette_costs.map(|costs| {
        let (prune_search, prune_size_search) = if cfg.allintra {
            let sf =
                SpeedFeatures::set_allintra(cfg.speed, cfg.allow_screen_content_tools, env.bd > 8);
            (
                sf.prune_palette_search_level,
                sf.prune_luma_palette_size_search_level,
            )
        } else {
            (0, 0)
        };
        crate::intra_rd::PaletteModeCfg {
            costs,
            above: palette_above,
            left: palette_left,
            prune_palette_search_level: prune_search,
            prune_luma_palette_size_search_level: prune_size_search,
        }
    });
    let sby_cfg = IntraSbySearchCfg {
        gates: &gates,
        // `intra_sf.top_intra_model_count_allowed` (speed_features.c:2443 default
        // TOP_INTRA_MODEL_COUNT=4; :404 -> 3 at allintra speed>=1; -> 2 at
        // speed>=6, :533). Drives the `top_intra_model_rd[]` slot
        // `prune_intra_y_mode` compares against (index = count-1, adjusted by
        // `get_model_rd_index_for_pruning` when the speed>=6
        // adapt_top_model_rd_count_using_neighbors sf is on). At speed 0/1 the
        // byte-match gates were inert to 4-vs-3 (the mode set kept fewer than 4
        // competitive models); the speed-2 `disable_smooth_intra` prune shrinks
        // the mode set enough that the 4-vs-3 slot difference tips the winner,
        // so this must be the correct 3 at speed>=1 to byte-match.
        top_intra_model_count_allowed: if cfg.allintra && cfg.speed >= 6 {
            2
        } else if cfg.allintra && cfg.speed >= 1 {
            3
        } else {
            4
        },
        // speed_features.c:534 — allintra speed>=6 (see get_model_rd_index_for
        // _pruning; reads the SAME above/left neighbour modes threaded below).
        adapt_top_model_rd_count_using_neighbors: cfg.allintra && cfg.speed >= 6,
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
        // av1_get_palette_mode_ctx: neighbours that use a Y palette. On
        // non-palette frames (palette_costs None / no screen content) the
        // grid carries no palette state and both flags are structurally 0 —
        // the pre-existing always-0 invariant, now explicit.
        palette_mode_ctx: palette_mode_ctx(
            up_available,
            palette_above
                .as_ref()
                .map_or(0, |p| i32::from(p.size[0] > 0)),
            left_available,
            palette_left
                .as_ref()
                .map_or(0, |p| i32::from(p.size[0] > 0)),
        ) as usize,
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
        winner_mode: wm_cfg.as_ref(),
        palette: palette_cfg,
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
    // `is_cfl_allowed(xd)` (blockd.h): non-lossless => w/h <= 32; LOSSLESS =>
    // CfL is still allowed when the partition size equals the transform size,
    // i.e. `get_plane_block_size(bsize, ssx, ssy) == BLOCK_4X4` (a 420 8x8 or
    // sub-8x8 chroma-ref leaf). The previous `!env.lossless && w<=32 && h<=32`
    // banned CfL outright at coded-lossless, so every 8x8-and-below chroma-ref
    // leaf lost the CfL candidate real aomenc picks (~16k rate on the KB-5
    // 420 cq0 cell's 8x8 leaves) and the 16x16 NONE-vs-SPLIT near-tie flipped.
    let cfl_allowed = is_cfl_allowed(bsize, env.lossless, env.ss_x, env.ss_y);
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
        (chroma_up_available && is_smooth_uv(uv_mode_at(base_row - 1, base_col + env.ss_x as i32)))
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
        luma_palette_active: false,
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
        qm_levels: cfg.qm_levels,
    };

    let re = ReencodeParams {
        sharpness: env.sharpness,
        enable_optimize_b: env.enable_optimize_b,
        tune: env.tune,
    };

    // `intra_sf.prune_chroma_modes_using_luma_winner` (speed_features.c:480) —
    // default 0; allintra speed>=4 -> 1. Prunes chroma modes not in
    // `av1_derived_chroma_intra_mode_used_flag[luma_winner_mode]`
    // (intra_mode_search.c:939-941). The consumer already lives in the uv loop
    // (intra_uv_rd.rs:1497); this turns it on. Only the LUMA winner mode is read,
    // so it is fully determined by the (already-decided) luma search.
    let prune_chroma_luma_winner = cfg.allintra && cfg.speed >= 4;
    // Per-block CHROMA directional-mode HOG prune (`intra_sf.chroma_intra_pruning
    // _with_hog`, speed_features.c:454). Off (level 0) at speed 0/1/2; allintra
    // speed>=3 -> level 2 (SpeedFeatures::set_allintra) — but the UNCONDITIONAL
    // TAIL of set_allintra_speed_features_framesize_independent (:608-616)
    // force-disables it whenever `prune_chroma_modes_using_luma_winner` is on
    // (allintra speed>=4), so it is live at exactly speed 3 in the modeled
    // range (KB-7 second root: keeping it on at speed 4 HOG-pruned a
    // directional uv mode C evaluates and picks). C computes the skip mask
    // lazily on the first directional uv candidate (intra_mode_search.c:959-972)
    // with the intra-frame threshold `thresh[1][level-1]` (= -1.2 at level 2);
    // precomputing it here is byte-equivalent (the mask is deterministic and is
    // only ever read in that same `is_directional && use_angle_delta` branch,
    // intra_uv_rd.rs:1457-1464). Only computed on chroma-ref blocks of a
    // non-monochrome frame — where the uv mode search actually evaluates modes.
    // (Level derived inline from cfg.speed, mirroring the luma HOG level above;
    // kept in lockstep with SpeedFeatures::set_allintra's tail.)
    let chroma_hog_level = if cfg.allintra && cfg.speed >= 3 && !prune_chroma_luma_winner {
        2
    } else {
        0
    };
    // `should_prune_chroma_smooth_pred_based_on_source_variance`
    // (intra_mode_search.c:850-862, gated by the speed>=6 sf
    // `prune_smooth_intra_mode_for_chroma`, :528): prune UV_SMOOTH_PRED when
    // the per-pixel SOURCE variance of BOTH chroma planes is < 20
    // (`av1_get_perpixel_variance_facade` per plane over the PLANE bsize;
    // short-circuit — V only measured when U passes, matching C's loop).
    // Deterministic per block, so precomputed here and carried on the
    // UvLoopPolicy (the loop consumer at intra_uv_rd.rs reads the flag on
    // the non-directional arm exactly where C calls the helper).
    let prune_smooth_for_chroma = cfg.allintra
        && cfg.speed >= 6
        && !env.monochrome
        && is_chroma_ref
        && perpixel_variance_y(env.src_u, ref_off_uv, env.stride, plane_bsize, env.bd) < 20
        && perpixel_variance_y(env.src_v, ref_off_uv, env.stride, plane_bsize, env.bd) < 20;
    // `intra_sf.cfl_search_range`: 3 (init default) through speed 5; 1 at
    // speed>=6 (:532) — est-only CfL refinement with the range-1
    // invalid/overhead early-outs (`cfl_rd_pick_alpha`, intra_uv_rd.rs).
    let cfl_search_range = if cfg.allintra && cfg.speed >= 6 { 1 } else { 3 };
    let chroma_hog_lp;
    let uv_lp: &UvLoopPolicy = if (chroma_hog_level > 0
        || prune_chroma_luma_winner
        || prune_smooth_for_chroma
        || cfl_search_range != 3)
        && !env.monochrome
        && is_chroma_ref
    {
        // The HOG mask only when the chroma HOG level is live (speed 3 —
        // the speed>=4 tail zeroes it, see chroma_hog_level above); the
        // luma-winner prune flag independently (speed>=4). Both feed the
        // same UvLoopPolicy the uv loop consumes.
        let mask = (chroma_hog_level > 0).then(|| {
            let mut mask = [false; 13];
            // C's collect_hog_data uses the LUMA `mb_to_*_edge` (1/8 luma-pel), then
            // `>> ss` inside prune_intra_mode_with_hog_uv.
            let mb_right = (env.mi_cols - mi_w as i32 - mi_col) * 4 * 8;
            let mb_bottom = (env.mi_rows - mi_h as i32 - mi_row) * 4 * 8;
            let th = [-1.2f32, -1.2, -0.6, 0.4][chroma_hog_level - 1];
            prune_intra_mode_with_hog_uv(
                env.src_u, ref_off_uv, env.stride, bsize, env.ss_x, env.ss_y, mb_right, mb_bottom,
                th, &mut mask,
            );
            mask
        });
        chroma_hog_lp = UvLoopPolicy {
            chroma_hog_skip_mask: mask,
            prune_chroma_modes_using_luma_winner: prune_chroma_luma_winner,
            prune_smooth_for_chroma,
            cfl_search_range,
            // intra_mode_info_cost_uv's try_palette (intra_mode_search_utils.h)
            // = av1_allow_palette(allow_screen_content_tools, mbmi->bsize) —
            // PER LEAF, and (like the Y flag) NOT gated on enable_palette: a
            // screen-content frame costs the UV no-palette flag on every
            // chroma-ref UV_DC candidate regardless of --enable-palette.
            try_palette: allow_palette(cfg.allow_screen_content_tools, bsize),
            ..cfg.uv_lp.clone()
        };
        &chroma_hog_lp
    } else if allow_palette(cfg.allow_screen_content_tools, bsize) != cfg.uv_lp.try_palette {
        // Same per-leaf try_palette recompute on the no-HOG path (the
        // frame-level cfg.uv_lp can't carry a bsize-dependent value).
        chroma_hog_lp = UvLoopPolicy {
            try_palette: allow_palette(cfg.allow_screen_content_tools, bsize),
            ..cfg.uv_lp.clone()
        };
        &chroma_hog_lp
    } else {
        cfg.uv_lp
    };

    // Intrabc leaf-search args (Some only on a screen-content frame that codes
    // `allow_intrabc`). The dv_grid closure reads the search ModeGrid's DV
    // state at offsets relative to (mi_row, mi_col) — exactly what
    // `find_dv_ref_mvs` requests.
    let dv_grid_closure = |rr: i32, rc: i32| grid.dv_at(mi_row + rr, mi_col + rc).to_nbr();
    let ibc_args = cfg.intrabc.as_ref().map(|ibc| {
        // av1_get_skip_txfm_context(xd) = above->skip_txfm + left->skip_txfm.
        let skip_ctx = (if up_available {
            usize::from(grid.dv_at(mi_row - 1, mi_col).skip_txfm)
        } else {
            0
        }) + (if left_available {
            usize::from(grid.dv_at(mi_row, mi_col - 1).skip_txfm)
        } else {
            0
        });
        crate::intrabc_search::IntrabcLeafArgs {
            sb_size: env.sb_size,
            bsize,
            mi_row,
            mi_col,
            mi_rows: env.mi_rows,
            mi_cols: env.mi_cols,
            tile: aom_entropy::dv_ref::DvTileBounds {
                mi_row_start: env.tile_row_start,
                mi_row_end: env.tile_row_end,
                mi_col_start: env.tile_col_start,
                mi_col_end: env.tile_col_end,
            },
            mib_size_log2: MI_SIZE_WIDE_B[env.sb_size].trailing_zeros() as i32,
            up_available,
            left_available,
            is_chroma_ref,
            monochrome: env.monochrome,
            ss_x: env.ss_x,
            ss_y: env.ss_y,
            bd: env.bd,
            partition,
            stride: env.stride,
            src_y: env.src_y,
            src_u: env.src_u,
            src_v: env.src_v,
            off_y: ref_off_y,
            off_uv: ref_off_uv,
            hash: ibc.hash,
            dv_costs: ibc.dv_costs,
            dv_grid: &dv_grid_closure,
            rdmult: env.rdmult,
            qindex: cfg.qindex,
            reduced_tx_set_used: env.reduced_tx_set_used,
            error_per_bit: ibc.error_per_bit,
            sad_per_bit: ibc.sad_per_bit,
            mv_step_param: ibc.mv_step_param,
            intrabc_cost: &cfg.mode_costs.intrabc_cost,
            skip_costs: cfg.skip_costs,
            skip_ctx,
            txfm_partition_costs: &ibc.txfm_partition_costs,
            rows_y: env.rows_y,
            rows_u: env.rows_u,
            rows_v: env.rows_v,
            coeff_costs_y: env.coeff_costs_y,
            coeff_costs_uv: env.coeff_costs_uv,
            tx_type_costs: env.tx_type_costs,
            sharpness: env.sharpness,
            enable_optimize_b: env.enable_optimize_b,
            qm_levels: env.qm_levels,
            above_ctx: [
                &tile.above_ectx[0][..],
                &tile.above_ectx[1][..],
                &tile.above_ectx[2][..],
            ],
            left_ctx: [
                &tile.left_ectx[0][..],
                &tile.left_ectx[1][..],
                &tile.left_ectx[2][..],
            ],
        }
    });

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
                lp: uv_lp,
                // The UV palette-search slice (same enable gate + neighbour
                // state as the luma slice; dc_mode_cost / y_palette_active
                // are overwritten from the luma winner in rd_pick).
                palette: cfg.palette_costs.map(|costs| {
                    crate::palette_search::UvPaletteArgs {
                        dc_mode_cost: 0,
                        costs,
                        above: palette_above,
                        left: palette_left,
                        bsize_ctx: palette_bsize_ctx(bsize) as usize,
                        y_palette_active: false,
                        early_term: {
                            // sf intra_sf.early_term_chroma_palette_size_search
                            // (allintra base :364 = 1 at EVERY allintra speed;
                            // GOOD speed-0 = init default 0).
                            cfg.allintra
                        },
                        pol: cfg.pol,
                    }
                }),
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
            ibc_args.as_ref(),
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
            let (uv_mode, angle_delta_uv, cfl_alpha_idx, cfl_alpha_signs, palette_uv) =
                match &best.uv {
                    RdPickUvOutcome::Searched(w, _) => (
                        w.uv_mode,
                        w.angle_delta_uv,
                        i32::from(w.cfl_alpha_idx),
                        i32::from(w.cfl_alpha_signs),
                        w.palette_uv.clone(),
                    ),
                    // !chroma_ref / monochrome: the uv mbmi fields are dead
                    // state (nothing reads them — store_cfl_required only reads
                    // uv_mode on chroma-ref blocks; packing is chroma-ref-gated).
                    _ => (0, 0, 0, 0, None),
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
                palette_y: best.y.palette_y.clone(),
                palette_uv,
                skip_txfm: best.use_intrabc && best.skip_txfm,
                use_intrabc: best.use_intrabc,
                dv_row: best.dv_row,
                dv_col: best.dv_col,
                dv_ref_row: best.dv_ref_row,
                dv_ref_col: best.dv_ref_col,
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
                false,
            );
            grid.stamp(
                r,
                c,
                subsize,
                wi.mode as u8,
                wi.uv_mode as u8,
                wi.palette_y.as_ref(),
                wi.palette_uv.as_ref(),
                wi.dv_cell(),
                env.mi_rows,
                env.mi_cols,
            );
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

/// `part_sf.ext_partition_eval_thresh` resolved for the **allintra KEY**
/// path: the bsize threshold BOTH extended-partition stages compare against
/// (`allow_ab_partition_search` partition_search.c:4005 `bsize >
/// ab_bsize_thresh`, and `prune_4_way_partition_search` :4136 `bsize >
/// part4_bsize_thresh` — the same sf field read at both sites;
/// `ext_part_eval_based_on_cur_best`, the only per-site adjustment, is set
/// exclusively on the GOOD path at :1013 and stays its 0 default here).
///
/// Value resolution (returns the BLOCK_SIZE enum value):
/// - default `BLOCK_8X8` (init_part_sf, speed_features.c:2312) for every
///   speed <= 4: the framesize-independent setter first assigns the field in
///   its `speed >= 5` block (:510), and the qindex-dependent `speed >= 2`
///   overrides (:2939-2965, `aggr = AOMMIN(4, speed-2)`) are all dead on a
///   KEY frame below speed 5 — the aggr<=1 arms are gated `!boosted` (KEY is
///   boosted: frame_is_boosted → frame_is_kf_gf_arf) and the aggr==2 arm is
///   gated `!frame_is_intra_only`.
/// - speed 5: the framesize-independent :510-511 sets `screen ? BLOCK_8X8 :
///   BLOCK_16X16`; then the qindex-dependent aggr==3 arm (:2947-2962) sets
///   `BLOCK_128X128` UNCONDITIONALLY for `!is_480p_or_larger` frames (no
///   boosted/intra gate — LIVE on KEY). Its two >=480p sub-arms stay dead on
///   KEY (`!frame_is_intra_only` gates both), keeping the :510 value there.
/// - speed >= 6: the qindex-dependent aggr==4 `else` arm (:2963-2964) sets
///   `BLOCK_128X128` unconditionally for every frame size.
///
/// `bsize > BLOCK_128X128` never holds, so a 128 threshold disables AB and
/// 4-way partitions outright. Frame w/h derive from mi dims like the
/// established `use_square_partition_only_threshold_allintra` caller and the
/// 4-way `res_idx` (exact for SB-aligned frames; same documented gap).
fn ext_partition_eval_thresh_allintra_key(
    allintra: bool,
    speed: i32,
    w: i32,
    h: i32,
    allow_screen_content_tools: bool,
) -> usize {
    if !allintra || speed < 5 {
        return 3; // BLOCK_8X8 (default; GOOD stays speed-0-frozen out of scope)
    }
    let is_480p_or_larger = w.min(h) >= 480;
    if speed >= 6 || !is_480p_or_larger {
        15 // BLOCK_128X128 — AB + 4-way disabled
    } else if allow_screen_content_tools {
        3 // BLOCK_8X8 (:510)
    } else {
        6 // BLOCK_16X16 (:511)
    }
}

/// `allow_ab_partition_search` (partition_search.c:3992-4020): simplifies on
/// the intra KEY path to `do_rectangular_split && bsize > ab_bsize_thresh &&
/// has_rows && has_cols` — `ab_bsize_thresh` is
/// [`ext_partition_eval_thresh_allintra_key`] (`ext_part_eval_based_on_cur_
/// best` is 0 outside GOOD — see that helper's docs), and `prune_ext_part_
/// state` (`prune_ext_part_none_skippable`) requires `skip_non_sq_part_based_
/// on_none >= 1`, the SAME sf already established dead at speed 0 in the
/// rect-stage module docs — so `!prune_ext_part_state` is always true and
/// omitted. The bsize compare is the raw BLOCK_SIZE enum order, exactly as C
/// (`bsize > ab_bsize_thresh`); partition nodes are square so this equals the
/// former 1-D-width compare for every reachable input.
fn allow_ab_partition_search(
    bsize: usize,
    ab_bsize_thresh: usize,
    do_rectangular_split: bool,
    has_rows: bool,
    has_cols: bool,
    best_rdcost: i64,
) -> bool {
    // Do not prune if there is no valid partition (:4002).
    if best_rdcost == i64::MAX {
        return true;
    }
    do_rectangular_split && bsize > ab_bsize_thresh && has_rows && has_cols
}

/// `evaluate_ab_partition_based_on_split` (partition_strategy.c:1870): keep an
/// AB candidate only when at least `num_win_thresh` of {this block's own
/// HORZ/VERT rect win, split child `idx1` picked NONE, split child `idx2`
/// picked NONE} hold. `num_win_thresh = AOMMIN(3 * (2*(MAXQ-qindex)/MAXQ), 3)`
/// — the integer division makes it **3 for qindex <= 127 and 0 for
/// qindex >= 128** (the "conservative pruning for high quantizers" comment:
/// the prune is inert at high q). `rect_win = None` models C's NULL
/// `rect_part_win_info` (the SB-root call): the fallback is
/// `pc_tree->partitioning == rect_part`. `child_nonone[i]` is true only when
/// that split child completed with a non-NONE partitioning (C counts a NULL
/// child, an aborted child — alloc-init partitioning stays NONE — and a
/// completed NONE child all as wins).
fn evaluate_ab_partition_based_on_split(
    rect_win: Option<bool>,
    pc_tree_partitioning: i32,
    rect_part: i32,
    qindex: i32,
    child_nonone: [bool; 2],
) -> bool {
    const MAXQ: i32 = 255;
    let num_win_thresh = (3 * (2 * (MAXQ - qindex) / MAXQ)).min(3);
    let sub_part_win = match rect_win {
        None => pc_tree_partitioning == rect_part,
        Some(w) => w,
    };
    let mut num_win = i32::from(sub_part_win);
    num_win += i32::from(!child_nonone[0]);
    num_win += i32::from(!child_nonone[1]);
    num_win >= num_win_thresh
}

/// `av1_prune_ab_partitions` (partition_strategy.c:1901-2029): the AB gating
/// pipeline — base gate, RD-ratio structural pruning
/// (`prune_ext_partition_types_search_level == 1`, LIVE at speed 0 both
/// usages — STATUS.md's AB plan point 3, unlike 4-way's own consumption of
/// the SAME sf at `== 2` which is dead), then `ml_prune_ab_partition` (LIVE,
/// same `ml_prune_partition` sf 4-way already established live, gated on
/// BOTH rect types being allowed — matches the C's own extra gate).
/// [`evaluate_ab_partition_based_on_split`] (:2009-2028,
/// `prune_ext_part_using_split_info >= 2` = allintra speed>=4) is applied by
/// the CALLER (it needs the node's rect-win flags + split-children state).
/// `x_source_variance` is the STALE `x->source_variance` (gotcha #1) fed ONLY
/// to the NN, never the structural pruning (which correctly uses
/// `pb_source_variance`).
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
                false,
            );
            grid.stamp(
                r,
                c,
                sz,
                wi.mode as u8,
                wi.uv_mode as u8,
                wi.palette_y.as_ref(),
                wi.palette_uv.as_ref(),
                wi.dv_cell(),
                env.mi_rows,
                env.mi_cols,
            );
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

/// `set_partition_cost_for_edge_blk` (partition_search.c:3411): the partition
/// cost row for a frame-EDGE block. `read_partition` (decodeframe.c) codes a
/// gathered 2-way symbol at an edge — HORZ/SPLIT at the bottom, VERT/SPLIT at
/// the right — or a forced SPLIT at the bottom-right corner, so the RD search
/// must charge that gathered cost instead of the full 10-way `partition_cost`.
/// `cdf_row` is `fc->partition_cdf[pl_ctx]` (`EXT_PARTITION_TYPES + 1` wide);
/// all non-coded partition types stay at `av1_cost_symbol(0)` (max cost). The
/// caller applies this only when `bsize_at_least_8x8 && !(has_rows && has_cols)`
/// (the C `pl_ctx_idx >= 0` assert + the `!av1_blk_has_rows_and_cols` gate at
/// partition_search.c:5695).
pub(crate) fn set_partition_cost_for_edge_blk(
    cdf_row: &[u16],
    bsize: usize,
    has_rows: bool,
    has_cols: bool,
) -> [i32; 10] {
    let mut ec = [cost_symbol(0); 10]; // max_cost = av1_cost_symbol(0)
    if has_cols {
        // At the bottom, the two possibilities are HORZ and SPLIT.
        let bot = partition_gather_vert_alike(cdf_row, bsize);
        cost_tokens_from_cdf(&mut ec, &bot, Some(&[1, 3])); // PARTITION_HORZ, PARTITION_SPLIT
    } else if has_rows {
        // At the right, the two possibilities are VERT and SPLIT.
        let rhs = partition_gather_horz_alike(cdf_row, bsize);
        cost_tokens_from_cdf(&mut ec, &rhs, Some(&[2, 3])); // PARTITION_VERT, PARTITION_SPLIT
    } else {
        // At the bottom right, we always split.
        ec[3] = 0; // PARTITION_SPLIT
    }
    ec
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
    // `rect_part_win_info` (av1_rd_pick_partition's 13th param): the OUT flags
    // this block's rect stage clears when HORZ/VERT loses (:3634-3636). C
    // passes `&split_part_rect_win[idx]` in the SPLIT recursion (:4586) and
    // NULL everywhere else (the SB-root calls, encodeframe.c:826+). Consumed
    // by the parent's split-info prunes AND this block's own AB evaluate
    // (`evaluate_ab_partition_based_on_split` reads the CURRENT node's flags).
    mut rect_part_win_out: Option<&mut [bool; 2]>,
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
    // writer in the speed 0..=5 KEY envelope (module docs); at allintra
    // speed>=6 the post-NONE `prune_rect_part_using_none_pred_mode`
    // (partition_search.c:4488) sets the flags off the NONE winner's mode.
    let mut prune_rect_part = [false; 2];
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
    // (0 at speed 0; `allow_screen_content_tools ? 0 : 2` at speed >= 1,
    // speed_features.c:387-388; `allow_screen_content_tools ? 1 : 2` at
    // speed >= 5, :512-513 — only the screen arm moves). Derived from the
    // existing cfg fields so speed-0 (and GOOD) stay frozen — the level is 0
    // there, making this whole block a no-op.
    // ---- av1_prune_partitions_before_search, the speed>=6 arms
    //      (partition_strategy.c:1736-1773 — run BEFORE the CNN prune :1779
    //      and before av1_prune_partitions_by_max_min_bsize, matching C's
    //      intra-function order). The :1735 `rect_partition_eval_thresh` arm
    //      stays a no-op here (no live setter on boosted KEY frames — module
    //      docs). Both arms below are `allow_screen_content_tools ? 0 : ...`
    //      sfs (speed_features.c:537-542), so screen frames skip them. ----
    if cfg.allintra && cfg.speed >= 6 && !cfg.allow_screen_content_tools {
        // `prune_rectangular_split_based_on_qidx == 2` (:1742-1757): disable
        // rect partitions for bsize < max_prune_bsize, where max_prune_bsize
        // steps down from BLOCK_32X32 by one square size per qindex third
        // (qidx 0-85: prune below 32X32; 86-170: below 16X16; 171-255: below
        // 8X8). sqr_bsize_step = BLOCK_32X32 - BLOCK_16X16 = 3.
        let max_bsize = (9 - (cfg.qindex * 3 / 256) * 3).max(0); // QINDEX_RANGE = 256
        let max_prune_bsize = max_bsize.min(9) as usize;
        if bsize < max_prune_bsize {
            // av1_disable_rect_partitions (encodeframe_utils.h:253).
            do_rectangular_split = false;
            partition_rect_allowed = [false, false];
        }
        // `prune_sub_8x8_partition_level == 1` (:1760-1773): at BLOCK_8X8,
        // disable all splits when BOTH neighbours are available and either
        // neighbour block is larger than 8x8 (bsize ENUM compare in C —
        // `left_mbmi->bsize > BLOCK_8X8`).
        if bsize == 3 {
            let up_avail = mi_row > env.tile_row_start;
            let left_avail = mi_col > env.tile_col_start;
            let prune_sub_8x8 = left_avail
                && up_avail
                && (grid.bsize_at(mi_row, mi_col - 1) > 3 || grid.bsize_at(mi_row - 1, mi_col) > 3);
            if prune_sub_8x8 {
                // av1_disable_all_splits (encodeframe_utils.h:261): square +
                // rect off; partition_none_allowed untouched.
                do_square_split = false;
                do_rectangular_split = false;
                partition_rect_allowed = [false, false];
            }
        }
    }

    let intra_cnn_based_part_prune_level = if cfg.allintra && cfg.speed >= 1 {
        if cfg.allow_screen_content_tools {
            i32::from(cfg.speed >= 5)
        } else {
            2
        }
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
    // set_partition_cost_for_edge_blk (partition_search.c:3411, called at
    // :5695 `if (!av1_blk_has_rows_and_cols(...))`): at a frame edge
    // read_partition does NOT code the full 10-way partition symbol — it codes
    // a gathered 2-way distribution (HORZ/SPLIT at the bottom, VERT/SPLIT at
    // the right) or a forced SPLIT at the bottom-right corner. The RD search
    // must charge the SAME cost or it mis-prices the partition decision at the
    // edge SBs (KB-6 partial-SB). Interior blocks (has_rows && has_cols) keep
    // the full 10-way cost. Only meaningful for bsize_at_least_8x8 (the C
    // assert; sub-8x8 blocks signal no partition symbol and pl_ctx is 0).
    let edge_partition_cost: [i32; 10];
    let partition_cost: &[i32; 10] = if bsize_at_least_8x8 && !(has_rows && has_cols) {
        edge_partition_cost =
            set_partition_cost_for_edge_blk(&cfg.partition_cdfs[pl_ctx], bsize, has_rows, has_cols);
        &edge_partition_cost
    } else {
        &cfg.partition_costs[pl_ctx]
    };

    // av1_rd_cost_update(x->rdmult, &best_rdc) (:5744).
    rd_cost_update(env.rdmult, &mut best_rdc);

    // av1_save_context (:5754).
    let saved = save_context(tile, mi_row, mi_col, bsize, env.ss_x, env.ss_y);

    // The per-node ALLINTRA variance arm (:5791-5827; module docs): two
    // branches over the SAME log_sub_block_var stats. Arm 1 (>= BLOCK_16X16
    // force-split, speed-independent) is live at every speed; arm 2 (rect
    // prune when the 4x4 variance deviation is LOW) needs the speed>=6
    // `prune_rect_part_using_4x4_var_deviation` sf (:539) — which ALSO
    // widens the stats computation to sub-16x16 blocks (`bsize_at_least_
    // 16x16 || prune_rect...`, :5797). `!x->must_find_valid_partition`
    // (:5795) is always true in this interior envelope (the flag only rises
    // on a no-valid-partition retry, which structurally can't happen here —
    // same established simplification as the AB stage's handling).
    let prune_rect_part_using_4x4_var_deviation = cfg.allintra && cfg.speed >= 6;
    if cfg.allintra && (bsize >= 6 || prune_rect_part_using_4x4_var_deviation) {
        let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
        let mb_right = (env.mi_cols - mi_w as i32 - mi_col) * 4 * 8;
        let mb_bottom = (env.mi_rows - MI_SIZE_HIGH_B[bsize] as i32 - mi_row) * 4 * 8;
        let (var_min, var_max) = log_sub_block_var(
            env.src_y, ref_off_y, env.stride, bsize, mb_right, mb_bottom, env.bd,
        );
        if bsize >= 6 && var_min < 0.272 && (var_max - var_min) > 3.0 {
            partition_none_allowed = false;
            // terminate_partition_search = 0 (:5817): already false — no
            // live setter in the envelope.
            do_square_split = true;
        } else if prune_rect_part_using_4x4_var_deviation && (var_max - var_min) < 3.0 {
            // "Prune rectangular partitions if the variance deviation of 4x4
            // sub-blocks within the block is less than a threshold" (:5819).
            // NOTE: only do_rectangular_split — partition_rect_allowed stays.
            do_rectangular_split = false;
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
            let none_winner_mode = winner.as_ref().map(|w| w.mode);
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
            // `prune_rect_part_using_none_pred_mode` (partition_search.c:
            // 4488-4489, allintra speed>=6 sf :540): reads the NONE winner's
            // luma mode (`pc_tree->none->mic.mode`) — NOT gated on the NONE
            // candidate actually beating best_rdc, only on a valid rate.
            if cfg.allintra && cfg.speed >= 6 {
                let mode = none_winner_mode.expect("valid rate has a winner");
                if mode == 0 || mode == 9 {
                    // DC_PRED / SMOOTH_PRED: low variation — prune both rects
                    // when a left/above neighbour block is LARGER (pixel
                    // area) than this block (:4375-4382).
                    let up_avail = mi_row > env.tile_row_start;
                    let left_avail = mi_col > env.tile_col_start;
                    if is_neighbor_blk_larger_than_cur_blk(
                        grid, mi_row, mi_col, bsize, up_avail, left_avail,
                    ) {
                        prune_rect_part = [true, true];
                    }
                } else if mode == 8 || mode == 1 || mode == 5 {
                    // D67 / V / D113: near-vertical pattern — prune HORZ.
                    prune_rect_part[0] = true;
                } else if mode == 6 || mode == 2 || mode == 7 {
                    // D157 / H / D203: near-horizontal pattern — prune VERT.
                    prune_rect_part[1] = true;
                }
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
    // `split_part_rect_win[4]` (encodeframe_utils.h:181, init TRUE at :3358-59):
    // per-split-child HORZ/VERT win flags, cleared by the child's own rect
    // stage on a loss. Consumed by `prune_4_partition_using_split_info`
    // (level>=1, allintra speed>=3) — like split_rd, survives the SPLIT block
    // whether or not SPLIT wins.
    let mut split_part_rect_win = [[true; 2]; 4];
    // `pc_tree->split[idx]->partitioning != PARTITION_NONE` for the AB
    // evaluate (partition_strategy.c:1883-1893): false covers ALL of C's
    // +1-win cases — child never allocated (loop broke early / off-frame),
    // search aborted (C's alloc-init partitioning stays PARTITION_NONE), or
    // completed with NONE.
    let mut split_child_nonone = [false; 4];
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
                Some(&mut split_part_rect_win[idx]),
                visits,
                last_source_variance,
            );
            split_child_nonone[idx] =
                matches!(&child_tree, Some(t) if !matches!(t, SbTree::Leaf(_)));
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
                // Off-frame quadrants at a partial edge SB are `None` here
                // (pushed at the `y >= mi_rows || x >= mi_cols` trim above,
                // mirroring the C's `pc_tree->split[idx]` for an out-of-bounds
                // origin that `encode_sb`/`write_modes_sb` never code). They
                // become `SbTree::Absent` placeholders — every tree walker
                // guards the same frame bound at entry, so an `Absent` slot is
                // never inspected. (An interior SB has all 4 children `Some`.)
                let kids: Vec<SbTree> = children
                    .into_iter()
                    .map(|t| t.unwrap_or(SbTree::Absent))
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
                false,
            );
            grid.stamp(
                mi_row,
                mi_col,
                subsize,
                w0.mode as u8,
                w0.uv_mode as u8,
                w0.palette_y.as_ref(),
                w0.palette_uv.as_ref(),
                w0.dv_cell(),
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
        if sum_rdc.rdcost >= best_rdc.rdcost {
            // Update HORZ / VERT win flag (:3634-3636): the rect type was
            // evaluated and did NOT beat the running best.
            if let Some(win) = rect_part_win_out.as_deref_mut() {
                win[i] = false;
            }
        }
        if sum_rdc.rdcost < best_rdc.rdcost {
            sum_rdc.rdcost = crate::rd::rdcost(env.rdmult, sum_rdc.rate, sum_rdc.dist);
            if sum_rdc.rdcost < best_rdc.rdcost {
                best_rdc = sum_rdc;
                found = true;
                pc_tree_partitioning = 1 + i as i32; // PARTITION_HORZ / PARTITION_VERT
                // At a partial edge SB the second sub-block is off-frame and
                // was never searched (`is_not_edge_block` false, sub-1 skipped
                // above — partition_search.c:3604); it becomes a never-coded
                // placeholder. Every walker guards the sub-1 frame bound before
                // touching it. Interior blocks always have a real sub-1 winner.
                let sub1 = match w1.take() {
                    Some(w) => w,
                    None => {
                        debug_assert!(!is_not_edge_block, "rect sub-1 absent only at a frame edge");
                        crate::encode_sb::LeafWinner::off_frame_placeholder(subsize)
                    }
                };
                let pair = Box::new([w0.take().expect("rect winner sub 0"), sub1]);
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

    // `part_sf.ext_partition_eval_thresh` for BOTH extended-partition stages
    // below (the AB gate :4005 and the 4-way gate :4136 read the same sf
    // field). BLOCK_8X8 (default) through speed 4; speed 5 disables AB+4-way
    // on sub-480p frames (BLOCK_128X128) — see the helper's docs.
    let ext_partition_eval_thresh = ext_partition_eval_thresh_allintra_key(
        cfg.allintra,
        cfg.speed,
        env.mi_cols * 4,
        env.mi_rows * 4,
        cfg.allow_screen_content_tools,
    );

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
            ext_partition_eval_thresh,
            do_rectangular_split,
            has_rows,
            has_cols,
            best_rdc.rdcost,
        );

        let mut ab_partitions_allowed = prune_ab_partitions(
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
        // Split-info AB prune (partition_strategy.c:2009-2028):
        // `prune_ext_part_using_split_info >= 2` = allintra speed>=4
        // (level 1 at speed>=3 per :446, 2 at speed>=4 per :476 — the level-1
        // consumers are the 4-way prune below, not this). Reads this node's
        // OWN rect-win flags (C's `rect_part_win_info` param — the SB-root
        // gets NULL → the pc_tree_partitioning fallback) + the split
        // children's NONE-ness. Sub-block index pairs per type: HORZ_A (0,1),
        // HORZ_B (2,3), VERT_A (0,2), VERT_B (1,3).
        if cfg.allintra && cfg.speed >= 4 {
            let rect_win = rect_part_win_out.as_deref();
            const AB_ARGS: [(usize, i32, usize, usize); 4] = [
                (0, 1, 0, 1), // HORZ_A: rect HORZ, split 0,1
                (0, 1, 2, 3), // HORZ_B: rect HORZ, split 2,3
                (1, 2, 0, 2), // VERT_A: rect VERT, split 0,2
                (1, 2, 1, 3), // VERT_B: rect VERT, split 1,3
            ];
            for (ab_type, &(dir, rect_part, i1, i2)) in AB_ARGS.iter().enumerate() {
                if ab_partitions_allowed[ab_type] {
                    ab_partitions_allowed[ab_type] &= evaluate_ab_partition_based_on_split(
                        rect_win.map(|w| w[dir]),
                        pc_tree_partitioning,
                        rect_part,
                        cfg.qindex,
                        [split_child_nonone[i1], split_child_nonone[i2]],
                    );
                }
            }
        }

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
    // ext_partition_eval_thresh: BLOCK_8X8 (the av1_reset_part_sf default)
    // through speed 4; the allintra speed>=5 setter + the qindex-dependent
    // aggr>=3 arms move it — resolved by ext_partition_eval_thresh_allintra
    // _key (see its docs; C reads the SAME sf field here, :4136, as in
    // allow_ab_partition_search, :4005). ext_part_eval_based_on_cur_best is
    // 0 outside GOOD (allintra never sets it; good's only setter is
    // `if (speed >= 5)`) so it never raises the threshold to BLOCK_128X128.
    // The bsize compare is the raw BLOCK_SIZE enum order, exactly as C
    // (`bsize > part4_bsize_thresh`); partition nodes are square so this
    // equals the former 1-D-width compare for every reachable input. C's
    // rdcost==INT64_MAX don't-prune early-out (:4131) is unreachable on this
    // interior envelope (NONE always yields a valid rd) — same established
    // simplification as the AB stage's must_find_valid_partition handling.
    let partition4_allowed_base = cfg.enable_1to4_partitions
        && do_rectangular_split
        && bsize > ext_partition_eval_thresh // > BLOCK_8X8 through speed 4
        // No 4-way at BLOCK_128X128: 128x32 / 32x128 are not in the block-size
        // enum (partition_search.c:4166 `partition4_allowed &= bsize !=
        // BLOCK_128X128`; the valid-types helper repeats it, :5181). Inert at
        // sb64 (bsize never 128) but load-bearing at sb128 — without it a 128
        // root would probe an invalid HORZ_4/VERT_4 subsize.
        && bsize != 15 // BLOCK_128X128
        && has_rows
        && has_cols;
    // prune_part4_search (partition_search.c:4152): disables 4-way when the
    // block's pixel width is below `min_partition_size_1d <<
    // prune_part4_search`. Allintra base 2 (speed_features.c:355, both
    // usages, verified); speed>=6 -> 3 (:543 — inert there: the
    // BLOCK_128X128 ext threshold above already kills 4-way outright, but
    // carried for source faithfulness).
    let prune_part4_search = if cfg.allintra && cfg.speed >= 6 { 3 } else { 2 };
    let width_ok = BLK_1D[bsize] >= (BLK_1D[cfg.min_partition_size] << prune_part4_search);
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
            // Levels 0/1/2 take the hd_-model threshold-table path; level 3
            // (speed >= 3) takes the OLD-model int-score path (`ml_model_index =
            // (level < 3)`, partition_strategy.c:1359), which OVERWRITES both
            // flags from zero per the label bits — C semantics; it can resurrect
            // a flag the pre-ML per-type gates cleared (KB-7 root: this branch
            // used to be an unported no-op, so the port searched HORZ_4/VERT_4
            // where C prunes them, flipping the speed-3/4 cq12 4:2:0 SB-root
            // partition near-ties).
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
            // Re-AND the interior-envelope frame-fit guard (all 4 strips
            // in-frame — see the module docs on rd_pick_4partition's edge
            // scope): inert for interior blocks; keeps a level-3 OVERWRITE
            // from resurrecting a 4-way at a frame-edge node this port's
            // 4-way walk does not model. (C itself codes fewer strips at an
            // edge; that whole shape is out of the current envelope.)
            part4_allowed[0] = h4 && all_4_rows_fit;
            part4_allowed[1] = v4 && all_4_cols_fit;
        }

        // prune_4_partition_using_split_info (partition_search.c:4023):
        // `prune_ext_part_using_split_info != 0` = allintra speed>=3 (:446).
        // Prune HORZ4/VERT4 when fewer than `AOMMIN(3*(MAXQ-qindex)/MAXQ + 1,
        // 3)` of the 4 split children kept their HORZ/VERT rect-win flag
        // (init TRUE; the child's rect stage clears it on a loss — children
        // whose rect types were never evaluated keep the win). Runs AFTER the
        // ML prune (C order); its per-type gate skips already-disallowed
        // types. Empirically a byte no-op on the speed-3 gate grid (verified
        // while unported); live from this landing — the speed-3 gate
        // re-verifies per run.
        if cfg.allintra && cfg.speed >= 3 {
            const MAXQ: i32 = 255;
            let num_win_thresh = (3 * (MAXQ - cfg.qindex) / MAXQ + 1).min(3);
            #[allow(clippy::needless_range_loop)] // i = HORZ/VERT axis of both arrays
            for i in 0..2usize {
                if !part4_allowed[i] {
                    continue;
                }
                let num_child_rect_win: i32 = (0..4)
                    .map(|idx| i32::from(split_part_rect_win[idx][i]))
                    .sum();
                if num_child_rect_win < num_win_thresh {
                    part4_allowed[i] = false;
                }
            }
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
        // C runs OUTPUT_ENABLED at the SB root (:6010) and DRY_RUN_NORMAL on
        // non-SB winner subtrees (:6023, gated by
        // should_do_dry_run_encode_for_current_block). The pack-stage adds
        // (CDF counts, tokens) live in pack.rs; the contexts/pixels of this
        // walk are identical — but the tx_type_map semantics are NOT: the
        // DRY walk aliases the winner (ctx) maps so eob-0 -> DCT_DCT resets
        // persist, while OUTPUT_ENABLED copies to the frame map and leaves
        // the winner maps untouched (encodeframe_utils.c:217-231). The pack
        // re-walks this same tree, so the SB-root walk here must NOT leak
        // its resets into the maps the pack re-quantizes from — see
        // encode_b_intra_dry's doc (the KB-4 coded-eob divergence).
        let (do_encode, output_enabled) = if bsize == env.sb_size {
            (true, true)
        } else {
            (
                should_do_dry_run_encode(env.sb_size, cfg.max_partition_size, pc_index, bsize),
                false,
            )
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
                output_enabled,
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
            grid.stamp(
                mi_row,
                mi_col,
                bsize,
                w.mode as u8,
                w.uv_mode as u8,
                w.palette_y.as_ref(),
                w.palette_uv.as_ref(),
                w.dv_cell(),
                mi_rows,
                mi_cols,
            );
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
            grid.stamp(
                mi_row,
                mi_col,
                sub,
                subs[0].mode as u8,
                subs[0].uv_mode as u8,
                subs[0].palette_y.as_ref(),
                subs[0].palette_uv.as_ref(),
                subs[0].dv_cell(),
                mi_rows,
                mi_cols,
            );
            if mi_row + hbs < mi_rows {
                grid.stamp(
                    mi_row + hbs,
                    mi_col,
                    sub,
                    subs[1].mode as u8,
                    subs[1].uv_mode as u8,
                    subs[1].palette_y.as_ref(),
                    subs[1].palette_uv.as_ref(),
                    subs[1].dv_cell(),
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Vert(subs) => {
            let sub = get_partition_subsize(bsize, 2) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(
                mi_row,
                mi_col,
                sub,
                subs[0].mode as u8,
                subs[0].uv_mode as u8,
                subs[0].palette_y.as_ref(),
                subs[0].palette_uv.as_ref(),
                subs[0].dv_cell(),
                mi_rows,
                mi_cols,
            );
            if mi_col + hbs < mi_cols {
                grid.stamp(
                    mi_row,
                    mi_col + hbs,
                    sub,
                    subs[1].mode as u8,
                    subs[1].uv_mode as u8,
                    subs[1].palette_y.as_ref(),
                    subs[1].palette_uv.as_ref(),
                    subs[1].dv_cell(),
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
                grid.stamp(
                    this_mi_row,
                    mi_col,
                    sub,
                    w.mode as u8,
                    w.uv_mode as u8,
                    w.palette_y.as_ref(),
                    w.palette_uv.as_ref(),
                    w.dv_cell(),
                    mi_rows,
                    mi_cols,
                );
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
                grid.stamp(
                    mi_row,
                    this_mi_col,
                    sub,
                    w.mode as u8,
                    w.uv_mode as u8,
                    w.palette_y.as_ref(),
                    w.palette_uv.as_ref(),
                    w.dv_cell(),
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::HorzA(subs) => {
            // PARTITION_HORZ_A: interior-only, no frame-bound gating on any
            // of the 3 sub-blocks (module docs on encode_sb.rs's
            // SbTree::HorzA).
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 4) as usize; // PARTITION_HORZ_A
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(
                mi_row,
                mi_col,
                bsize2,
                subs[0].mode as u8,
                subs[0].uv_mode as u8,
                subs[0].palette_y.as_ref(),
                subs[0].palette_uv.as_ref(),
                subs[0].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row,
                mi_col + hbs,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                subs[1].palette_y.as_ref(),
                subs[1].palette_uv.as_ref(),
                subs[1].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col,
                sub,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                subs[2].palette_y.as_ref(),
                subs[2].palette_uv.as_ref(),
                subs[2].dv_cell(),
                mi_rows,
                mi_cols,
            );
        }
        SbTree::HorzB(subs) => {
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 5) as usize; // PARTITION_HORZ_B
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(
                mi_row,
                mi_col,
                sub,
                subs[0].mode as u8,
                subs[0].uv_mode as u8,
                subs[0].palette_y.as_ref(),
                subs[0].palette_uv.as_ref(),
                subs[0].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                subs[1].palette_y.as_ref(),
                subs[1].palette_uv.as_ref(),
                subs[1].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                subs[2].palette_y.as_ref(),
                subs[2].palette_uv.as_ref(),
                subs[2].dv_cell(),
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertA(subs) => {
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 6) as usize; // PARTITION_VERT_A
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(
                mi_row,
                mi_col,
                bsize2,
                subs[0].mode as u8,
                subs[0].uv_mode as u8,
                subs[0].palette_y.as_ref(),
                subs[0].palette_uv.as_ref(),
                subs[0].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                subs[1].palette_y.as_ref(),
                subs[1].palette_uv.as_ref(),
                subs[1].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row,
                mi_col + hbs,
                sub,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                subs[2].palette_y.as_ref(),
                subs[2].palette_uv.as_ref(),
                subs[2].dv_cell(),
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertB(subs) => {
            let bsize2 = split_subsize(bsize);
            let sub = get_partition_subsize(bsize, 7) as usize; // PARTITION_VERT_B
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            grid.stamp(
                mi_row,
                mi_col,
                sub,
                subs[0].mode as u8,
                subs[0].uv_mode as u8,
                subs[0].palette_y.as_ref(),
                subs[0].palette_uv.as_ref(),
                subs[0].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row,
                mi_col + hbs,
                bsize2,
                subs[1].mode as u8,
                subs[1].uv_mode as u8,
                subs[1].palette_y.as_ref(),
                subs[1].palette_uv.as_ref(),
                subs[1].dv_cell(),
                mi_rows,
                mi_cols,
            );
            grid.stamp(
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                subs[2].mode as u8,
                subs[2].uv_mode as u8,
                subs[2].palette_y.as_ref(),
                subs[2].palette_uv.as_ref(),
                subs[2].dv_cell(),
                mi_rows,
                mi_cols,
            );
        }
        // Off-frame placeholder — unreachable past the entry frame-bound guard.
        SbTree::Absent => {}
    }
}

/// `av1_rd_use_partition` (partition_search.c:1764) — the VAR_BASED_PARTITION
/// walk (allintra speed >= 7, KEY): apply the pre-calculated partition tree
/// (the [`crate::var_part::choose_var_based_partitioning_key`] `bsize` stamps
/// in `vbp_stamps`, read back per node via
/// [`crate::var_part::get_partition_from_stamps`] = C's `get_partition`)
/// running [`leaf_pick_sb_modes`] — the SAME full-RD leaf mode search the
/// partition search uses (`use_nonrd_pick_mode` stays 0 until speed 8) — at
/// each tree leaf, with C's exact context-propagation shape:
///
/// - **HORZ/VERT** (:1869-1938): pick sub 0, then `av1_update_state` +
///   `encode_superblock(DRY_RUN_NORMAL)` (= [`encode_b_intra_dry`], the rect
///   stage's own mid-stage propagation) before picking sub 1. Sub 1 runs only
///   when in-frame (`mi_row/col + hbs < mi_rows/cols`) — a frame-edge rect
///   codes sub 0 alone. Both leaves take `invalid_rdc` budgets (INT64_MAX —
///   the fixed tree never early-outs a leaf).
/// - **SPLIT** (:1940-1974): recurse per quadrant with `do_recon = i != 3`
///   (the C call's `i != SUB_PARTITIONS_SPLIT - 1`) — children 0..2 re-encode
///   their winner subtree (context propagation for the next sibling), child 3
///   skips it (the parent's own walk covers it). Off-frame quadrants are
///   skipped ([`SbTree::Absent`]).
/// - **Per node** (:1815/:2051 + :2064-2086): `av1_save_context` on entry,
///   `av1_restore_context` after the switch, then `if (do_recon) encode_sb`
///   — OUTPUT_ENABLED at the SB root, DRY_RUN_NORMAL below
///   ([`encode_sb_dry`] with the same `output_enabled` tx_type_map
///   semantics as the pick path's winner walk) + the mi-grid stamp
///   (`stamp_grid_from_tree`).
///
/// Structurally DEAD at allintra speed 7 (verified against source, KB-11):
/// - The PARTITION_NONE re-evaluation (:1827-1852) and the split-of-NONEs
///   re-evaluation (:1986-2040) — both gated on
///   `adjust_var_based_rd_partitioning` (`is_adjust_var_based_part_enabled`
///   :1714 needs 1/2; the `> 2` chosen-split gate) which is **0 outside
///   REALTIME** (init :2288; the =2 setter :2002 + the qindex-dep :2896 are
///   both REALTIME-only). So `none_rdc`/`chosen_rdc` stay invalid, the
///   `last_part < chosen` compare always keeps the tree's partition, and
///   `use_partition_none` never fires. NOT ported (documented).
/// - `setup_block_rdmult` (:1824): resets `x->rdmult = cpi->rd.RDMULT` then
///   applies the ALLINTRA `intra_sb_rdmult_modifier` fold — which the VAR
///   path leaves at its per-SB reset value 128 (encodeframe.c:1303; only
///   av1_rd_pick_partition's root recomputes it, partition_search.c:5715) —
///   identity. `env.rdmult` must be the UNMODIFIED frame RDMULT (the caller
///   `pack_tile` skips the SB fold on this path).
/// - `bsize == BLOCK_16X16 && cpi->vaq_refresh` mb_energy (:1817), AQ /
///   delta-q / SSIM rdmult arms — all off in this envelope.
///
/// The partition rate (`partition_cost[pl][partition]`, :2043-2047 — the
/// plain 10-way table; rd_use_partition does NOT use rd_pick's
/// `set_partition_cost_for_edge_blk` edge gather) is folded faithfully into
/// the returned stats, but on this path the RD totals are DECISION-INERT:
/// nothing compares them (the tree is fixed; the none/chosen arms are dead).
///
/// Returns `(tree, last_part_rdc)`.
#[allow(clippy::too_many_arguments)]
pub fn rd_use_partition_real(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &mut TileCtxState,
    grid: &mut ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    vbp_stamps: &[u8],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    do_recon: bool,
    visits: &mut Vec<LeafVisit>,
    last_source_variance: &mut u32,
) -> (SbTree, PartRdStats) {
    debug_assert!(
        mi_row < env.mi_rows && mi_col < env.mi_cols,
        "callers skip off-frame quadrants (partition_search.c:1799/:1952)"
    );
    // In rt mode, currently the min partition size is BLOCK_8X8 (:1803) —
    // the KEY variance tree never stamps below it.
    debug_assert!(bsize >= 3, "bsize >= default_min_partition_size (:1803)");
    let bs = MI_SIZE_WIDE_B[bsize] as i32;
    let hbs = bs / 2;
    let invalid = PartRdStats::invalid();

    // get_partition (av1_common_int.h:1775) over the stamp grid.
    let partition = crate::var_part::get_partition_from_stamps(
        vbp_stamps,
        env.mi_rows,
        env.mi_cols,
        mi_row,
        mi_col,
        bsize,
    );
    let subsize = get_partition_subsize(bsize, partition) as usize;

    // partition_plane_context (:1778-1780) + the plain 10-way cost row.
    let pl_ctx = partition_plane_context(
        &tile.above_pctx,
        &tile.left_pctx,
        mi_row as usize,
        mi_col as usize,
        bsize,
    ) as usize;
    let partition_cost = &cfg.partition_costs[pl_ctx];

    // av1_save_context (:1815).
    let saved = save_context(tile, mi_row, mi_col, bsize, env.ss_x, env.ss_y);

    // C: av1_invalid_rd_stats(&last_part_rdc) at entry (:1805) — every
    // reachable arm below assigns before reading (the compiler enforces it,
    // so the invalid-init is elided rather than dead-stored).
    let mut last_part_rdc;
    let tree: SbTree = match partition {
        // PARTITION_NONE (:1861-1864).
        0 => {
            let (this_rdc, winner, sv) = leaf_pick_sb_modes(
                env, cfg, tile, grid, recon_y, recon_u, recon_v, cfl, mi_row, mi_col, bsize, 0,
                &invalid,
            );
            *last_source_variance = sv;
            visits.push(LeafVisit {
                mi_row,
                mi_col,
                bsize,
                budget: invalid.rdcost,
                rate: this_rdc.rate,
                dist: this_rdc.dist,
                rdcost: this_rdc.rdcost,
            });
            last_part_rdc = this_rdc;
            SbTree::Leaf(winner.expect("unbounded-budget leaf pick always finds a winner"))
        }
        // PARTITION_HORZ (:1869) / PARTITION_VERT (:1903) — same shape on
        // the other axis.
        1 | 2 => {
            let is_horz = partition == 1;
            let (this_rdc, w0, sv) = leaf_pick_sb_modes(
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
                partition as usize,
                &invalid,
            );
            *last_source_variance = sv;
            visits.push(LeafVisit {
                mi_row,
                mi_col,
                bsize: subsize,
                budget: invalid.rdcost,
                rate: this_rdc.rate,
                dist: this_rdc.dist,
                rdcost: this_rdc.rdcost,
            });
            last_part_rdc = this_rdc;
            let mut w0 = w0.expect("unbounded-budget leaf pick always finds a winner");
            let sub1_in_frame = if is_horz {
                mi_row + hbs < env.mi_rows
            } else {
                mi_col + hbs < env.mi_cols
            };
            // (:1885-1908 / :1919-1938) — bsize >= BLOCK_8X8 always holds
            // here (tree min); rate != INT_MAX always holds (unbounded leaf).
            if last_part_rdc.rate != i32::MAX && sub1_in_frame {
                // av1_update_state + encode_superblock(DRY_RUN_NORMAL)
                // (:1890-1892) — the mid-stage propagation, exactly the rect
                // stage's own shape.
                let _ = crate::encode_sb::encode_b_intra_dry(
                    env,
                    tile,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    &mut w0,
                    mi_row,
                    mi_col,
                    partition as usize,
                    false,
                );
                grid.stamp(
                    mi_row,
                    mi_col,
                    subsize,
                    w0.mode as u8,
                    w0.uv_mode as u8,
                    w0.palette_y.as_ref(),
                    w0.palette_uv.as_ref(),
                    w0.dv_cell(),
                    env.mi_rows,
                    env.mi_cols,
                );
                let (r1, c1) = if is_horz {
                    (mi_row + hbs, mi_col)
                } else {
                    (mi_row, mi_col + hbs)
                };
                let (tmp_rdc, w1, sv1) = leaf_pick_sb_modes(
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
                    partition as usize,
                    &invalid,
                );
                *last_source_variance = sv1;
                visits.push(LeafVisit {
                    mi_row: r1,
                    mi_col: c1,
                    bsize: subsize,
                    budget: invalid.rdcost,
                    rate: tmp_rdc.rate,
                    dist: tmp_rdc.dist,
                    rdcost: tmp_rdc.rdcost,
                });
                // (:1899-1902): INT_MAX invalidates; else accumulate.
                if tmp_rdc.rate == i32::MAX || tmp_rdc.dist == i64::MAX {
                    last_part_rdc = PartRdStats::invalid();
                } else {
                    last_part_rdc.rate += tmp_rdc.rate;
                    last_part_rdc.dist += tmp_rdc.dist;
                    last_part_rdc.rdcost += tmp_rdc.rdcost;
                }
                let w1 = w1.expect("unbounded-budget leaf pick always finds a winner");
                if is_horz {
                    SbTree::Horz(Box::new([w0, w1]))
                } else {
                    SbTree::Vert(Box::new([w0, w1]))
                }
            } else {
                // Frame-edge rect: sub 0 alone. The SbTree Horz/Vert
                // variants carry both winners (interior envelope, module
                // docs on encode_sb.rs) — an edge single-strip rect is
                // representable only once that envelope lifts. The KEY
                // variance tree produces edge rects solely on
                // non-multiple-of-64 frames (var_part.rs's
                // `edge_vert_single_strip_stamp`), outside the current
                // speed-7 gate grid.
                unimplemented!(
                    "frame-edge single-strip {} at ({mi_row},{mi_col}) bsize {bsize}: \
                     out of the interior-envelope SbTree rect representation",
                    if is_horz { "HORZ" } else { "VERT" }
                )
            }
        }
        // PARTITION_SPLIT (:1940-1974).
        3 => {
            last_part_rdc = PartRdStats::init();
            let mut kids: [SbTree; 4] = [
                SbTree::Absent,
                SbTree::Absent,
                SbTree::Absent,
                SbTree::Absent,
            ];
            #[allow(clippy::needless_range_loop)] // i drives y/x AND the kids slot
            for i in 0..4usize {
                let y = mi_row + ((i as i32) >> 1) * hbs;
                let x = mi_col + ((i as i32) & 1) * hbs;
                if y >= env.mi_rows || x >= env.mi_cols {
                    continue;
                }
                let (child_tree, tmp_rdc) = rd_use_partition_real(
                    env,
                    cfg,
                    tile,
                    grid,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    vbp_stamps,
                    y,
                    x,
                    subsize,
                    /*do_recon=*/ i != 3,
                    visits,
                    last_source_variance,
                );
                kids[i] = child_tree;
                if tmp_rdc.rate == i32::MAX || tmp_rdc.dist == i64::MAX {
                    last_part_rdc = PartRdStats::invalid();
                    break;
                }
                last_part_rdc.rate += tmp_rdc.rate;
                last_part_rdc.dist += tmp_rdc.dist;
            }
            SbTree::Split(Box::new(kids))
        }
        other => {
            // VERT_A/VERT_B/HORZ_A/HORZ_B/HORZ_4/VERT_4: "Cannot handle
            // extended partition types" (:1976-1982) — the variance tree
            // never produces them.
            unreachable!("av1_rd_use_partition: extended partition {other} from the vbp tree")
        }
    };

    // (:2043-2047): fold the partition cost, recompute the rdcost.
    if last_part_rdc.rate < i32::MAX {
        last_part_rdc.rate += partition_cost[partition as usize];
        last_part_rdc.rdcost =
            crate::rd::rdcost(env.rdmult, last_part_rdc.rate, last_part_rdc.dist);
    }

    // (:2049-2062, the last_part/none winner selection): none_rdc and
    // chosen_rdc are both invalid on this path (module docs), so the tree's
    // partition always stands — nothing to do.

    // av1_restore_context (:2064).
    restore_context(tile, &saved, mi_row, mi_col, bsize, env.ss_x, env.ss_y);

    // (:2072-2086): the winner re-encode. OUTPUT_ENABLED at the SB root
    // (with set_cb_offsets, folded into encode_sb_dry's own walk),
    // DRY_RUN_NORMAL below.
    if do_recon {
        let mut tree = tree;
        let output_enabled = bsize == env.sb_size;
        let mut leaves: Vec<LeafEncodeOut> = Vec::new();
        encode_sb_dry(
            env,
            tile,
            recon_y,
            recon_u,
            recon_v,
            cfl,
            &mut tree,
            mi_row,
            mi_col,
            bsize,
            &mut leaves,
            output_enabled,
        );
        stamp_grid_from_tree(grid, &tree, mi_row, mi_col, bsize, env.mi_rows, env.mi_cols);
        return (tree, last_part_rdc);
    }

    (tree, last_part_rdc)
}

#[cfg(test)]
mod ext_partition_eval_thresh_tests {
    use super::ext_partition_eval_thresh_allintra_key;

    /// `ext_partition_eval_thresh` resolution for the allintra KEY path,
    /// asserted against the source arms (speed_features.c:510-511 + the
    /// qindex-dependent :2939-2965 `aggr = AOMMIN(4, speed-2)` cascade with
    /// the boosted/!intra arms dead on KEY — see the helper's docs).
    #[test]
    fn thresh_matches_source_arms() {
        // Speeds 0..=4: the init default BLOCK_8X8 at every size/screen combo
        // (the qindex-dependent aggr<=2 arms are all !boosted / !intra gated).
        for speed in 0..=4 {
            for &(w, h) in &[(64, 64), (128, 128), (640, 480), (1280, 720)] {
                for &sc in &[false, true] {
                    assert_eq!(
                        ext_partition_eval_thresh_allintra_key(true, speed, w, h, sc),
                        3,
                        "speed {speed} {w}x{h} sc={sc}"
                    );
                }
            }
        }
        // Speed 5, sub-480p: BLOCK_128X128 unconditionally (:2952) — AB +
        // 4-way disabled (bsize > BLOCK_128X128 never holds).
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 5, 64, 64, false),
            15
        );
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 5, 128, 128, true),
            15
        );
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 5, 640, 360, false),
            15
        );
        // Speed 5, >=480p: the framesize-independent :510-511 value survives
        // (screen ? BLOCK_8X8 : BLOCK_16X16).
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 5, 640, 480, false),
            6
        );
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 5, 1280, 720, false),
            6
        );
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 5, 640, 480, true),
            3
        );
        // Speed >= 6: BLOCK_128X128 for every size (:2963 else arm).
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 6, 1280, 720, false),
            15
        );
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(true, 9, 64, 64, true),
            15
        );
        // Non-allintra (GOOD speed-0 envelope): default.
        assert_eq!(
            ext_partition_eval_thresh_allintra_key(false, 5, 64, 64, false),
            3
        );
    }
}

#[cfg(test)]
mod edge_partition_cost_tests {
    use super::*;

    /// Witness for `set_partition_cost_for_edge_blk` (the CHUNK-3 frame-edge
    /// partition-cost override). The gather + `cost_tokens_from_cdf` +
    /// `cost_symbol` primitives are each already proven byte-exact vs C
    /// (`aom_entropy::partition` write tests, `aom_txb::prob_cost` diff tests);
    /// this locks the COMPOSITION the C source dictates — the per-edge gather
    /// choice, the `[HORZ,SPLIT]`/`[VERT,SPLIT]` inverse maps, the
    /// `av1_cost_symbol(0)` fill, and the forced-SPLIT corner — which is where a
    /// porting mistake would live.
    #[test]
    fn edge_cost_matches_gathered_composition() {
        // An asymmetric but valid 10-symbol partition inverse CDF (BLOCK_64X64,
        // so every partition type participates — the 128x128 VERT_4/HORZ_4 skip
        // does not apply). Probabilities sum to CDF_PROB_TOP (32768).
        let probs = [
            8000i32, 2000, 6000, 5000, 3000, 1000, 2000, 1768, 2000, 2000,
        ];
        let mut cdf = [0u16; 11];
        let mut cum = 0i32;
        for (k, &p) in probs.iter().enumerate() {
            cum += p;
            cdf[k] = (32768 - cum) as u16; // AOM_ICDF
        }
        assert_eq!(
            cdf[9], 0,
            "a valid inverse CDF terminates at 0 on its last symbol"
        );
        let bsize = 12; // BLOCK_64X64
        let max = cost_symbol(0);

        // Bottom edge (has_cols, !has_rows): HORZ + SPLIT via vert_alike gather.
        let bot = set_partition_cost_for_edge_blk(&cdf, bsize, false, true);
        let mut expect_bot = [max; 10];
        let g = partition_gather_vert_alike(&cdf, bsize);
        cost_tokens_from_cdf(&mut expect_bot, &g, Some(&[1, 3]));
        assert_eq!(
            bot, expect_bot,
            "bottom edge = vert_alike over [HORZ, SPLIT]"
        );
        for (i, &c) in bot.iter().enumerate() {
            if i != 1 && i != 3 {
                assert_eq!(c, max, "bottom edge: uncoded type {i} stays max_cost");
            }
        }

        // Right edge (has_rows, !has_cols): VERT + SPLIT via horz_alike gather.
        let rhs = set_partition_cost_for_edge_blk(&cdf, bsize, true, false);
        let mut expect_rhs = [max; 10];
        let g2 = partition_gather_horz_alike(&cdf, bsize);
        cost_tokens_from_cdf(&mut expect_rhs, &g2, Some(&[2, 3]));
        assert_eq!(
            rhs, expect_rhs,
            "right edge = horz_alike over [VERT, SPLIT]"
        );
        for (i, &c) in rhs.iter().enumerate() {
            if i != 2 && i != 3 {
                assert_eq!(c, max, "right edge: uncoded type {i} stays max_cost");
            }
        }

        // For an asymmetric CDF the two edges must differ — proves the per-edge
        // gather/inv-map selection is not accidentally swapped.
        assert_ne!(
            bot[3], rhs[3],
            "bottom vs right SPLIT cost must differ for an asymmetric CDF"
        );

        // Bottom-right corner (!has_rows, !has_cols): forced SPLIT, rest max.
        let corner = set_partition_cost_for_edge_blk(&cdf, bsize, false, false);
        assert_eq!(corner[3], 0, "corner forces PARTITION_SPLIT (cost 0)");
        for (i, &c) in corner.iter().enumerate() {
            if i != 3 {
                assert_eq!(c, max, "corner: non-SPLIT type {i} stays max_cost");
            }
        }
    }
}

// ===========================================================================
// KB-12 — speed >= 8: av1_nonrd_use_partition (partition_search.c:2960).
// HANDOFF: written under kill-order, NEVER COMPILED — see HANDOFF-SPEED89.md.
// ===========================================================================

/// `av1_nonrd_use_partition` (partition_search.c:2960) for the allintra KEY
/// path: the SINGLE-PASS walk over the VBP-stamped tree — per leaf
/// `pick_sb_modes_nonrd` (here: the hybrid dispatch below) then
/// `encode_b_nonrd` IMMEDIATELY (here: [`crate::encode_sb::encode_b_intra_dry`]
/// — recon + context stamps; bits come from the unchanged `pack_sb` re-walk).
/// NO save/restore, NO mid-strip dry re-encode, NO root winner walk
/// (contrast [`rd_use_partition_real`]).
///
/// Dead-on-KEY arms NOT modelled (verified against source):
/// - `try_split_partition` — `nonrd_check_partition_split` stays 0 (:3001).
/// - `try_merge` — `!frame_is_intra_only` gate (:3089).
/// - `direct_partition_merging` — `!frame_is_intra_only` (:3106).
/// - `x->reuse_inter_pred` (:2998) — inter only.
/// - `set_mode_eval_params(DEFAULT_EVAL)` (:2996) — the full-RD arm's leaf
///   machinery already runs DEFAULT_EVAL-shaped (leaf_pick_sb_modes).
///
/// `speed`: drives the hybrid gate (2 at speed 8, 0 at speed 9 —
/// `hybrid_intra_pickmode`) and the speed-9 estimate-loop prunes.
#[allow(clippy::too_many_arguments)]
pub fn nonrd_use_partition_real(
    env: &SbEncodeEnv,
    cfg: &PickFrameCfg,
    tile: &mut TileCtxState,
    grid: &mut ModeGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    vbp_stamps: &[u8],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    visits: &mut Vec<LeafVisit>,
    last_source_variance: &mut u32,
) -> SbTree {
    debug_assert!(mi_row < env.mi_rows && mi_col < env.mi_cols);
    debug_assert!(bsize >= 3, "only square blocks 8x8..128x128 (:2971)");
    let bs = MI_SIZE_WIDE_B[bsize] as i32;
    let hbs = bs / 2;

    let partition = crate::var_part::get_partition_from_stamps(
        vbp_stamps,
        env.mi_rows,
        env.mi_cols,
        mi_row,
        mi_col,
        bsize,
    );
    let subsize = get_partition_subsize(bsize, partition) as usize;

    match partition {
        // PARTITION_NONE (:3017-3030).
        0 => {
            let w = nonrd_leaf_pick_and_encode(
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
                visits,
                last_source_variance,
            );
            SbTree::Leaf(w)
        }
        // PARTITION_HORZ (:3055) / PARTITION_VERT (:3031) — pick+encode strip
        // 0, then strip 1 gated `mi_+hbs in frame && bsize > BLOCK_8X8`.
        1 | 2 => {
            let is_horz = partition == 1;
            let w0 = nonrd_leaf_pick_and_encode(
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
                partition as usize,
                visits,
                last_source_variance,
            );
            let sub1_in_frame = if is_horz {
                mi_row + hbs < env.mi_rows
            } else {
                mi_col + hbs < env.mi_cols
            };
            // C gate: `&& bsize > BLOCK_8X8` (:3046/:3070) — port bsize 3.
            if sub1_in_frame && bsize > 3 {
                let (r1, c1) = if is_horz {
                    (mi_row + hbs, mi_col)
                } else {
                    (mi_row, mi_col + hbs)
                };
                let w1 = nonrd_leaf_pick_and_encode(
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
                    partition as usize,
                    visits,
                    last_source_variance,
                );
                if is_horz {
                    SbTree::Horz(Box::new([w0, w1]))
                } else {
                    SbTree::Vert(Box::new([w0, w1]))
                }
            } else {
                // Same interior-envelope limitation as rd_use_partition_real:
                // the SbTree rect variants carry both winners.
                unimplemented!(
                    "frame-edge single-strip nonrd rect at ({mi_row},{mi_col}) bsize {bsize}"
                )
            }
        }
        // PARTITION_SPLIT (:3078-3117): plain recursion (try_merge/direct
        // merging are KEY-dead, module docs).
        3 => {
            let mut kids: [SbTree; 4] = [
                SbTree::Absent,
                SbTree::Absent,
                SbTree::Absent,
                SbTree::Absent,
            ];
            for (i, kid) in kids.iter_mut().enumerate() {
                let y = mi_row + ((i as i32) >> 1) * hbs;
                let x = mi_col + ((i as i32) & 1) * hbs;
                if y >= env.mi_rows || x >= env.mi_cols {
                    continue;
                }
                *kid = nonrd_use_partition_real(
                    env,
                    cfg,
                    tile,
                    grid,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    vbp_stamps,
                    y,
                    x,
                    subsize,
                    visits,
                    last_source_variance,
                );
            }
            SbTree::Split(Box::new(kids))
        }
        other => unreachable!("av1_nonrd_use_partition: extended partition {other} (:3119-3125)"),
    }
}

/// One nonrd leaf: `pick_sb_modes_nonrd` (partition_search.c:2254 —
/// `hybrid_intra_mode_search` on KEY, :2325) + `encode_b_nonrd` (:2089 —
/// port: `encode_b_intra_dry` + grid stamp; bits via `pack_sb`).
#[allow(clippy::too_many_arguments)]
fn nonrd_leaf_pick_and_encode(
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
    partition: usize,
    visits: &mut Vec<LeafVisit>,
    last_source_variance: &mut u32,
) -> LeafWinner {
    // x->source_variance: pick_sb_modes_nonrd:2306-2311 recomputes per leaf
    // (bsize < sb_size, or the SB-level value is the identical
    // perpixel-variance — module docs in nonrd_pickmode.rs).
    let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
    let source_variance = perpixel_variance_y(env.src_y, ref_off_y, env.stride, bsize, env.bd);
    *last_source_variance = source_variance;

    // hybrid_intra_pickmode: 2 at speed 8, 0 at speed >= 9
    // (speed_features.c:578 / :598).
    let hybrid = if cfg.speed >= 9 { 0 } else { 2 };

    if crate::nonrd_pickmode::hybrid_use_rdopt(hybrid, bsize, source_variance) {
        // Full-RD arm: av1_rd_pick_intra_mode_sb with INT64_MAX budget
        // (partition_search.c:769) — the EXISTING leaf machinery.
        let invalid = PartRdStats::invalid();
        let (this_rdc, winner, _sv) = leaf_pick_sb_modes(
            env, cfg, tile, grid, recon_y, recon_u, recon_v, cfl, mi_row, mi_col, bsize, partition,
            &invalid,
        );
        visits.push(LeafVisit {
            mi_row,
            mi_col,
            bsize,
            budget: invalid.rdcost,
            rate: this_rdc.rate,
            dist: this_rdc.dist,
            rdcost: this_rdc.rdcost,
        });
        let mut w = winner.expect("unbounded-budget leaf pick always finds a winner");
        // output_enabled = true (OUTPUT_ENABLED): the C nonrd walk encodes
        // every leaf dry_run=0 (encode_b_nonrd, partition_search.c:2100). Per
        // KB-4 that gives the tx_type_map COPY semantics — eob-0 -> DCT_DCT
        // resets go to a transient frame map, leaving the search winner's
        // `w.tx_type_map` intact for the byte-producing `pack_sb` re-walk
        // (pack.rs, also OUTPUT_ENABLED). `false` (alias) would leak resets
        // into the winner map and re-introduce the KB-4 bug on the full-RD
        // arm (non-DCT winner quantizing to eob 0). Matches the speed-7
        // rd_use_partition_real SB-root walk (output_enabled = bsize==sb_size).
        let _ = crate::encode_sb::encode_b_intra_dry(
            env, tile, recon_y, recon_u, recon_v, cfl, &mut w, mi_row, mi_col, partition, true,
        );
        grid.stamp(
            mi_row,
            mi_col,
            bsize,
            w.mode as u8,
            w.uv_mode as u8,
            w.palette_y.as_ref(),
            w.palette_uv.as_ref(),
            w.dv_cell(),
            env.mi_rows,
            env.mi_cols,
        );
        return w;
    }

    // Estimate arm: av1_nonrd_pick_intra_mode (nonrd_pickmode.c:1582).
    let up_available = mi_row > env.tile_row_start;
    let left_available = mi_col > env.tile_col_start;
    let above_mode = if up_available {
        grid.at(mi_row - 1, mi_col) as usize
    } else {
        0
    };
    let left_mode = if left_available {
        grid.at(mi_row, mi_col - 1) as usize
    } else {
        0
    };
    // KF y-mode ctx pair (intra_mode_context[A]/[L]) — same table the
    // full-RD leaf uses.
    const IMC: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
    let bmode_costs = &cfg.mode_costs.y_mode_costs[IMC[above_mode.min(12)]][IMC[left_mode.min(12)]];
    // skip ctx 0 (KEY intra invariant — leaf_pick_sb_modes' own).
    let skip_cost = &cfg.skip_costs[0];
    // Luma edge filter type (smooth above/left) — leaf_pick_sb_modes pattern.
    let is_smooth = |m: usize| (9..=11).contains(&m);
    let luma_edge_filter_type = i32::from(
        (up_available && is_smooth(above_mode)) || (left_available && is_smooth(left_mode)),
    );

    let lctx = crate::nonrd_pickmode::NonrdIntraLeafCtx {
        bmode_costs,
        skip_cost,
        above_mode,
        left_mode,
        up_available,
        left_available,
        source_variance,
        partition,
        prune_h_pred_using_best_mode_so_far: cfg.speed >= 9,
        enable_intra_mode_pruning_using_neighbors: cfg.speed >= 9,
        prune_intra_mode_using_best_sad_so_far: cfg.speed >= 9,
        allow_screen_content_tools: cfg.allow_screen_content_tools,
        luma_edge_filter_type,
    };
    let pick = crate::nonrd_pickmode::nonrd_pick_intra_mode(
        env, &lctx, recon_y, mi_row, mi_col, bsize, env.rdmult,
    );
    visits.push(LeafVisit {
        mi_row,
        mi_col,
        bsize,
        budget: i64::MAX,
        rate: pick.rd.rate,
        dist: pick.rd.dist,
        rdcost: pick.rd.rdcost,
    });

    // ctx->mic snapshot → LeafWinner (store_coding_context_nonrd +
    // init_mbmi_nonrd fields): uv = DC (the chroma answer), angle 0,
    // filter_intra off, palette zero, tx_type_map all DCT_DCT, skip_txfm
    // false (encode_b_nonrd forces mi->skip_txfm = 0 for intra, :2120).
    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_h = MI_SIZE_HIGH_B[bsize];
    // Chroma edge filter type for the encode (leaf_pick_sb_modes' chroma
    // pattern) — DC-only uv makes it decision-inert, recomputed for fidelity.
    let base_row = mi_row - (mi_row & env.ss_y as i32);
    let base_col = mi_col - (mi_col & env.ss_x as i32);
    let mut chroma_up_available = up_available;
    let mut chroma_left_available = left_available;
    if env.ss_x != 0 && mi_w < 2 {
        chroma_left_available = (mi_col - 1) > env.tile_col_start;
    }
    if env.ss_y != 0 && mi_h < 2 {
        chroma_up_available = (mi_row - 1) > env.tile_row_start;
    }
    let uv_at = |r: i32, c: i32| -> u8 {
        if r >= 0 && c >= 0 && r < env.mi_rows && c < env.mi_cols {
            grid.at_uv(r, c)
        } else {
            0
        }
    };
    let is_smooth_uv = |m: u8| (9..=11).contains(&m);
    let uv_edge_filter_type = i32::from(
        (chroma_up_available && is_smooth_uv(uv_at(base_row - 1, base_col + env.ss_x as i32)))
            || (chroma_left_available
                && is_smooth_uv(uv_at(base_row + env.ss_y as i32, base_col - 1))),
    );

    let mut w = LeafWinner {
        bsize,
        mode: pick.mode,
        angle_delta_y: 0,
        use_filter_intra: false,
        filter_intra_mode: 0,
        tx_size: pick.tx_size,
        luma_edge_filter_type,
        uv_mode: 0, // UV_DC_PRED — nonrd_pickmode.c:1735
        angle_delta_uv: 0,
        cfl_alpha_idx: 0,
        cfl_alpha_signs: 0,
        uv_edge_filter_type,
        tx_type_map: vec![0; mi_w * mi_h], // DCT_DCT
        skip_txfm: false,
        use_intrabc: false,
        dv_row: 0,
        dv_col: 0,
        dv_ref_row: 0,
        dv_ref_col: 0,
        raw_rdstats: pick.rd,
        // Palette is guarded dead on the nonrd estimate arm (init_mbmi_nonrd
        // zeroes palette sizes; the palette search arm needs
        // allow_screen_content_tools=1, dead on the canon grid).
        palette_y: None,
        palette_uv: None,
    };
    // output_enabled = true (OUTPUT_ENABLED) — see the full-RD arm above.
    // On the estimate arm `w.tx_type_map` is all-DCT so copy/alias are
    // identical here, but true keeps the faithful C semantics.
    let _ = crate::encode_sb::encode_b_intra_dry(
        env, tile, recon_y, recon_u, recon_v, cfl, &mut w, mi_row, mi_col, partition, true,
    );
    grid.stamp(
        mi_row,
        mi_col,
        bsize,
        w.mode as u8,
        w.uv_mode as u8,
        w.palette_y.as_ref(),
        w.palette_uv.as_ref(),
        w.dv_cell(),
        env.mi_rows,
        env.mi_cols,
    );
    w
}
