//! Minimal `SPEED_FEATURES` config for the **all-intra KEY-frame** path
//! (libaom `av1/encoder/speed_features.c`
//! `set_allintra_speed_features_framesize_independent` +
//! `set_allintra_speed_feature_framesize_dependent`), mapping a
//! `cpu_used`/speed level to the exact sf-derived values this port's search +
//! pack pipeline consumes.
//!
//! # Why this exists
//!
//! Before this module the pipeline hardcoded speed-0 values at every call site
//! (`TxTypeSearchPolicy::speed0_allintra()`, `PickFrameCfg { speed: 0,
//! intra_pruning_with_hog: true, less_rectangular_check_level: 1, .. }`, etc.).
//! Gate 2 (`aomenc --cpu-used` 0..=9 bit-identical) needs those values to vary
//! by speed level. [`SpeedFeatures::set_allintra`] centralizes the mapping.
//!
//! # Speed-0 is a frozen no-op
//!
//! `set_allintra(0, ..)` reproduces **exactly** the values the pipeline
//! hardcoded before this module existed — locked field-by-field by
//! `speed0_allintra_matches_hardcoded` in the unit tests below. Introducing
//! this module therefore cannot change any speed-0 byte-match gate.
//!
//! # Coverage of the field set (honest fraction)
//!
//! This models the sf fields that (a) the port already consumes at speed 0 and
//! (b) differ speed-0 → speed-N on an intra-still KEY frame with CDEF +
//! loop-restoration search disabled (the current e2e harness envelope). The
//! speed cascade is transcribed faithfully for **speeds 0..=5** (the intra-still
//! deltas at each level; inert inter/motion/CDEF/loop-restoration fields are
//! documented per-block below, not carried as struct fields). Speed 4 carries
//! the full KB-8 winner-mode two-pass parameter set (the `prune_chroma_modes_
//! using_luma_winner` chroma prune :480, the `LPF_PICK_FROM_FULL_IMAGE_NON_
//! DUAL` loop-filter search :496 — wired via `lf_search::pick_filter_level`'s
//! `non_dual` flag, not an sf field — plus the winner-mode two-pass subsystem
//! :488-505). Speed 5 re-parameterizes it (`multi_winner_mode_type` DEFAULT ->
//! FAST :524) and disables the AB/4-way extended partitions on sub-480p frames
//! via `ext_partition_eval_thresh` (:510 + the qindex-dependent :2947-2963
//! aggr=3 arm; NOT a struct field here — framesize+qindex dependent, resolved
//! by `partition_pick::ext_partition_eval_thresh_allintra_key`). Speed >= 6
//! (`prune_filter_intra_level = 2` at :529, `top_intra_model_count_allowed = 2`
//! at :533, `multi_winner_mode_type = OFF` at :561, LPF_PICK_FROM_Q at :559,
//! etc.) is unmodeled. The `lpf_sf` CDEF/restoration fields are carried for
//! provenance but do not affect bytes (CDEF + restoration off).
//!
//! Source line citations are against libaom v3.14.1 (git 03087864).

use crate::tx_search::TxTypeSearchPolicy;

/// `MODE_EVAL_TYPES` index used everywhere in this port: winner-mode two-pass
/// evaluation (`enable_winner_mode_for_*`) first activates at speed >= 4 for
/// all-intra (speed_features.c:502-505), so speed 0..=3 always read the
/// `DEFAULT_EVAL` column (rd.h:95, `get_rd_opt_coeff_thresh` `!enable_winner`
/// branch, rd.h:317-321).
pub const DEFAULT_EVAL: usize = 0;
/// The `MODE_EVAL` column (rdopt_utils.h `MODE_EVAL_TYPE`): the first pass of
/// the winner-mode two-pass, evaluating ALL candidate modes with the cheaper
/// per-stage thresholds. Consumed by the two-pass wiring (KB-8 chunk 2d-iv,
/// partition_pick.rs) via [`SpeedFeatures::tx_type_search_policy_for_stage`].
pub const MODE_EVAL: usize = 1;
/// The `WINNER_MODE_EVAL` column: the second pass, re-evaluating the stored
/// top-N winners with the most accurate per-stage thresholds.
pub const WINNER_MODE_EVAL: usize = 2;

/// `tx_domain_dist_thresholds[4][MODE_EVAL_TYPES]` (speed_features.c:54-59) —
/// verbatim. Indexed by `rd_sf.tx_domain_dist_thres_level`.
const TX_DOMAIN_DIST_THRESHOLDS: [[u32; 3]; 4] = [
    [u32::MAX, u32::MAX, u32::MAX],
    [22026, 22026, 22026],
    [1377, 1377, 1377],
    [0, 0, 0],
];

/// `tx_domain_dist_types[TX_DOMAIN_DIST_LEVELS=4][MODE_EVAL_TYPES]`
/// (speed_features.c:71-74) — verbatim. Indexed by `rd_sf.tx_domain_dist_level`.
const TX_DOMAIN_DIST_TYPES: [[u32; 3]; 4] =
    [[0, 2, 0], [1, 2, 0], [2, 2, 0], [2, 2, 2]];

/// `coeff_opt_thresholds[9][MODE_EVAL_TYPES][2]` (speed_features.c:88-98) —
/// verbatim. Indexed by `rd_sf.perform_coeff_opt`; inner `[2]` is `[dist, satd]`.
const COEFF_OPT_THRESHOLDS: [[[u32; 2]; 3]; 9] = [
    [[u32::MAX, u32::MAX], [u32::MAX, u32::MAX], [u32::MAX, u32::MAX]],
    [[3200, u32::MAX], [250, u32::MAX], [u32::MAX, u32::MAX]],
    [[1728, u32::MAX], [142, u32::MAX], [u32::MAX, u32::MAX]],
    [[864, u32::MAX], [142, u32::MAX], [u32::MAX, u32::MAX]],
    [[432, u32::MAX], [86, u32::MAX], [u32::MAX, u32::MAX]],
    [[864, 97], [142, 16], [u32::MAX, u32::MAX]],
    [[432, 97], [86, 16], [u32::MAX, u32::MAX]],
    [[216, 25], [86, 10], [u32::MAX, u32::MAX]],
    [[216, 25], [0, 10], [u32::MAX, u32::MAX]],
];

// `TX_TYPE_PRUNE_MODE` (speed_features.h:197-205).
/// `TX_TYPE_PRUNE_1` — the tx-type ML-prune aggressiveness at speed 0.
pub const TX_TYPE_PRUNE_1: i32 = 1;
/// `TX_TYPE_PRUNE_2` — speed-1 tx-type ML-prune aggressiveness.
pub const TX_TYPE_PRUNE_2: i32 = 2;

// `CDEF_PICK_METHOD` (speed_features.h:164-169).
/// `CDEF_FULL_SEARCH` — full CDEF strength search (speed 0).
pub const CDEF_FULL_SEARCH: i32 = 0;
/// `CDEF_FAST_SEARCH_LVL1` — reduced CDEF strength search (speed 1).
pub const CDEF_FAST_SEARCH_LVL1: i32 = 1;

/// `TOP_INTRA_MODEL_COUNT` (enums.h:391) — the default number of top intra
/// luma modes carried from model-RD to full RD.
pub const TOP_INTRA_MODEL_COUNT: i32 = 4;

/// The intra-still-relevant subset of libaom's `SPEED_FEATURES`, resolved for
/// one speed level on the **all-intra KEY** path. Field names mirror the C
/// `sf->group.field` (the group prefix is dropped; see the doc on each field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeedFeatures {
    // ---- part_sf ---------------------------------------------------------
    /// `part_sf.less_rectangular_check_level` — allintra base 1
    /// (speed_features.c:352), speed>=3 -> 2. Drives the SPLIT-stage rect kill.
    pub less_rectangular_check_level: i32,
    /// `part_sf.intra_cnn_based_part_prune_level` — default 0
    /// (init_part_sf:2311); speed>=1 -> `allow_screen_content_tools ? 0 : 2`
    /// (speed_features.c:387-388); speed>=5 -> `allow_screen_content_tools ?
    /// 1 : 2` (:512-513, only the screen arm moves). CNN split-vs-nonsplit
    /// partition prune on intra SBs.
    pub intra_cnn_based_part_prune_level: i32,
    /// `part_sf.reuse_best_prediction_for_part_ab` — default 0
    /// (partition_search.c:89 / speed_features.c:2324); speed>=1 -> 1
    /// (speed_features.c:397). Seeds the AB extended-partition mode cache.
    pub reuse_best_prediction_for_part_ab: i32,
    /// `part_sf.ml_4_partition_search_level_index` (framesize-DEPENDENT) —
    /// default 0 (init_part_sf:2305); speed>=1 -> 1 (speed_features.c:210).
    /// Indexes the HORZ4/VERT4 ML-prune thresholds; not frame-gated.
    pub ml_4_partition_search_level_index: i32,

    // ---- intra_sf --------------------------------------------------------
    /// `intra_sf.intra_pruning_with_hog` — allintra base 1
    /// (speed_features.c:360); speed>=2 -> 2, speed>=3 -> 3. Luma HOG
    /// directional-mode prune aggressiveness. NOTE for bytes: the C threshold
    /// table `thresh[4] = {-1.2, -1.2, -0.6, 0.4}` (intra_mode_search.c:1505,
    /// indexed by `level-1`) makes level 1 and level 2 IDENTICAL (-1.2), so the
    /// 1->2 bump at speed 2 does not move the HOG prune output; the threshold
    /// only changes at speed>=3 (level 3 -> -0.6).
    pub intra_pruning_with_hog: i32,
    /// `intra_sf.chroma_intra_pruning_with_hog` — default 0 (init_intra_sf, off);
    /// allintra speed>=3 -> 2 (speed_features.c:454), speed>=5 -> 3 (:515) —
    /// BUT the :608-615 final override zeroes it whenever
    /// `prune_chroma_modes_using_luma_winner` is on (speed>=4), so the net
    /// value is 2 ONLY at speed 3 and 0 everywhere else. Turns ON the CHROMA
    /// directional-mode HOG prune (intra_mode_search.c:959-972): for an intra
    /// frame the threshold table is `thresh[1] = {-1.2, -1.2, -0.6, 0.4}`
    /// (indexed by `level-1`), so level 2 -> -1.2. Prunes UV_V_PRED..UV_D67_PRED
    /// from the chroma-mode search when the U-plane HOG score <= threshold.
    pub chroma_intra_pruning_with_hog: i32,
    /// `intra_sf.prune_chroma_modes_using_luma_winner` — default 0
    /// (init_intra_sf); allintra speed>=4 -> 1 (speed_features.c:480). Prunes any
    /// chroma `uv_mode` not flagged in
    /// `av1_derived_chroma_intra_mode_used_flag[luma_winner_mode]`
    /// (intra_mode_search.c:939-941). Consumed in the uv mode loop
    /// (intra_uv_rd.rs:1497). Wired per-block from `cfg.speed` in
    /// `partition_pick.rs` (mirroring the inline `chroma_intra_pruning_with_hog`
    /// level derivation there); this field documents + unit-asserts the value.
    pub prune_chroma_modes_using_luma_winner: bool,
    /// `intra_sf.disable_smooth_intra` — default 0 (init_intra_sf:2438); allintra
    /// speed>=2 -> 1 (speed_features.c:429). Prunes SMOOTH_H_PRED / SMOOTH_V_PRED
    /// from the luma intra-mode search (intra_mode_search.c:1564-1567); SMOOTH_PRED
    /// survives while `prune_filter_intra_level != 0` (the :1574 interaction).
    pub disable_smooth_intra: bool,
    /// `intra_sf.prune_filter_intra_level` — default 0 (init_intra_sf:2440);
    /// allintra speed>=2 -> 1 (speed_features.c:431), speed>=4 -> 2. At level 1
    /// the filter-intra search (rd_pick_filter_intra_sby, intra_mode_search.c:264)
    /// only evaluates the FILTER modes derived from the best-so-far Y mode.
    pub prune_filter_intra_level: i32,
    /// `intra_sf.prune_palette_search_level` — default 0 (init_intra_sf:2431);
    /// speed>=1 -> 1 (speed_features.c:402).
    pub prune_palette_search_level: i32,
    /// `intra_sf.prune_luma_palette_size_search_level` — allintra base 1
    /// (speed_features.c:362); speed>=1 -> 2 (speed_features.c:403).
    pub prune_luma_palette_size_search_level: i32,
    /// `intra_sf.top_intra_model_count_allowed` — default
    /// [`TOP_INTRA_MODEL_COUNT`] (=4, init_intra_sf:2443); speed>=1 -> 3
    /// (speed_features.c:404), speed>=4 -> 2. Top luma modes taken to full RD.
    pub top_intra_model_count_allowed: i32,

    // ---- tx_sf -----------------------------------------------------------
    /// `tx_sf.adaptive_txb_search_level` — allintra base 1
    /// (speed_features.c:366); speed>=1 -> 2 (speed_features.c:406). txb-RD
    /// early-exit aggressiveness.
    pub adaptive_txb_search_level: i32,
    /// `tx_sf.intra_tx_size_search_init_depth_rect` — default 0
    /// (init_tx_sf:2453); speed>=1 -> 1 (speed_features.c:409).
    pub intra_tx_size_search_init_depth_rect: i32,
    /// `tx_sf.intra_tx_size_search_init_depth_sqr` — allintra base 1
    /// (speed_features.c:367); unchanged through speed 1.
    pub intra_tx_size_search_init_depth_sqr: i32,
    /// `tx_sf.model_based_prune_tx_search_level` — allintra base 1
    /// (speed_features.c:368); speed>=1 -> 0 (speed_features.c:410). NOTE the
    /// 1->0 reversal: speed 1 *disables* model-based tx-search pruning.
    pub model_based_prune_tx_search_level: i32,
    /// `tx_sf.use_chroma_trellis_rd_mult` — allintra base 1
    /// (speed_features.c:370). Chroma trellis rd-mult table select.
    pub use_chroma_trellis_rd_mult: bool,
    /// `tx_sf.use_rd_based_breakout_for_intra_tx_search` — default 0
    /// (init_tx_sf:2472); allintra speed>=3 -> 1 (speed_features.c:460).
    /// Tightens the intra tx-size depth loop's early-exit threshold to
    /// `AOMMIN(ref_best_rd, best_rd)` (tx_search.c:3030) and switches the
    /// winner-mode re-eval's ref_best_rd from INT64_MAX to the running best
    /// (intra_mode_search.c:1201). DELIBERATELY left false here at speed 3
    /// (empirically a byte no-op on the speed-3 gate grid with the current
    /// single-pass search); the KB-8 chunk-2d-iv speed-4 flip sets it and
    /// re-verifies the speed-3 gate.
    pub use_rd_based_breakout_for_intra_tx_search: bool,

    // ---- tx_sf.tx_type_search --------------------------------------------
    /// `tx_sf.tx_type_search.ml_tx_split_thresh` — default 8500
    /// (init_tx_sf:2458); speed>=1 -> 4000 (speed_features.c:411).
    pub tx_ml_tx_split_thresh: i32,
    /// `tx_sf.tx_type_search.prune_2d_txfm_mode` — default
    /// [`TX_TYPE_PRUNE_1`] (init_tx_sf:2457); speed>=1 -> [`TX_TYPE_PRUNE_2`]
    /// (speed_features.c:412).
    pub prune_2d_txfm_mode: i32,
    /// `tx_sf.tx_type_search.skip_tx_search` — default 0 (init_tx_sf:2463);
    /// speed>=1 -> 1 (speed_features.c:413). The all-zero-quant early break.
    pub skip_tx_search: bool,
    /// `tx_sf.tx_type_search.use_reduced_intra_txset` — allintra base 1
    /// (speed_features.c:369). Unchanged through speed 1.
    pub use_reduced_intra_txset: bool,
    /// `tx_sf.tx_type_search.fast_intra_tx_type_search` — default 0
    /// (init_tx_sf:2461); allintra speed>=4 -> 2 (speed_features.c:489). At 2,
    /// `set_mode_eval_params(MODE_EVAL)` sets `use_default_intra_tx_type=1`
    /// (the first pass evaluates only the intra mode's default tx type);
    /// `is_winner_mode_processing_enabled` also returns 1 whenever this is nonzero
    /// (unless `use_intra_dct_only`/`use_intra_default_tx_only`), enabling the
    /// WINNER_MODE_EVAL re-eval. Not yet SET in `set_allintra` (KB-8 chunk 2d).
    pub fast_intra_tx_type_search: i32,
    /// `tx_sf.tx_type_search.winner_mode_tx_type_pruning` — default 0
    /// (init_tx_sf:2466); allintra speed>=4 -> 2 (speed_features.c:488). Selects
    /// the per-stage `prune_2d_txfm_mode` row (`set_tx_type_prune`, rdopt_utils.h:
    /// 498): row `winner_mode_tx_type_pruning-1`, col `is_winner_mode`. Not yet
    /// SET in `set_allintra` (KB-8 chunk 2d).
    pub winner_mode_tx_type_pruning: i32,
    /// `tx_sf.tx_type_search.prune_tx_type_est_rd` — default 0
    /// (init_tx_sf:2465); allintra speed>=4 -> 1 (speed_features.c:491). Gates
    /// the est-rd tx-type prune + txk_map reorder in `get_tx_mask`'s multi-type
    /// arm (LIVE on intra in the WINNER pass — no inter gate). Not yet SET in
    /// `set_allintra` (KB-8 chunk 2d-iv).
    pub prune_tx_type_est_rd: bool,

    // ---- winner_mode_sf --------------------------------------------------
    /// `winner_mode_sf.enable_winner_mode_for_coeff_opt` — default 0
    /// (init:2511); allintra speed>=4 -> 1 (speed_features.c:502). When 1, the
    /// coeff-opt threshold is stage-selected: MODE_EVAL uses the MODE_EVAL
    /// column, WINNER_MODE_EVAL the WINNER column (rd.h:317-339). Not yet SET
    /// in `set_allintra` (KB-8 chunk 2d).
    pub enable_winner_mode_for_coeff_opt: bool,
    /// `winner_mode_sf.enable_winner_mode_for_use_tx_domain_dist` — default 0
    /// (init:2513); allintra speed>=4 -> 1 (speed_features.c:503). Stage-selects
    /// the tx-domain distortion type/threshold columns (rdopt_utils.h:516-544).
    /// Not yet SET in `set_allintra` (KB-8 chunk 2d).
    pub enable_winner_mode_for_use_tx_domain_dist: bool,
    /// `winner_mode_sf.enable_winner_mode_for_tx_size_srch` — default 0
    /// (init:2512); allintra speed>=4 -> 1 (speed_features.c:505). Stage-selects
    /// the tx-size search method (rdopt_utils.h:478-493): MODE_EVAL uses the
    /// MODE_EVAL column of `tx_size_search_methods[tx_size_search_level]`,
    /// WINNER the WINNER column. Not yet SET in `set_allintra` (KB-8 chunk 2d).
    pub enable_winner_mode_for_tx_size_srch: bool,
    /// `winner_mode_sf.multi_winner_mode_type` — default 0 = MULTI_WINNER_MODE_OFF
    /// (init:2514); allintra speed>=4 -> MULTI_WINNER_MODE_DEFAULT (**=2**,
    /// speed_features.h:230, speed_features.c:504), speed>=5 ->
    /// MULTI_WINNER_MODE_FAST (**=1**, speed_features.h:226). Indexes
    /// `winner_mode_count_allowed[]` = `{1, 2, 3}` for OFF/FAST/DEFAULT
    /// (rdopt_utils.h:236): the number of top modes stored by
    /// `store_winner_mode_stats` and re-evaluated. Not yet SET in
    /// `set_allintra` (KB-8 chunk 2d-iv).
    pub multi_winner_mode_type: i32,
    /// `winner_mode_sf.tx_size_search_level` — default 0 (init:2510). Indexes
    /// the row of `tx_size_search_methods[4][MODE_EVAL_TYPES]`. Stays 0 on the
    /// all-intra path through speed 8 (the allintra cascade never bumps it).
    pub tx_size_search_level: i32,

    // ---- rd_sf -----------------------------------------------------------
    /// `rd_sf.perform_coeff_opt` — allintra base 1 (speed_features.c:383);
    /// speed>=1 -> 2 (speed_features.c:415). Indexes [`COEFF_OPT_THRESHOLDS`].
    pub perform_coeff_opt: i32,
    /// `rd_sf.tx_domain_dist_level` — default 0 (init_rd_sf:2501); speed>=1 ->
    /// 1 (speed_features.c:416). Indexes [`TX_DOMAIN_DIST_TYPES`].
    pub tx_domain_dist_level: i32,
    /// `rd_sf.tx_domain_dist_thres_level` — default 0 (init_rd_sf:2502);
    /// speed>=1 -> 1 (speed_features.c:417). Indexes
    /// [`TX_DOMAIN_DIST_THRESHOLDS`].
    pub tx_domain_dist_thres_level: i32,

    // ---- lpf_sf (CDEF / loop-restoration search) -------------------------
    // Carried for provenance; the current e2e harness encodes the reference
    // with CDEF + restoration OFF, so these do not yet affect bytes.
    /// `lpf_sf.cdef_pick_method` — default [`CDEF_FULL_SEARCH`]
    /// (init_lpf_sf:2533); speed>=1 -> [`CDEF_FAST_SEARCH_LVL1`]
    /// (speed_features.c:419).
    pub cdef_pick_method: i32,
    /// `lpf_sf.dual_sgr_penalty_level` — default 0; speed>=1 -> 1
    /// (speed_features.c:420).
    pub dual_sgr_penalty_level: i32,
    /// `lpf_sf.enable_sgr_ep_pruning` — default 0; speed>=1 -> 1
    /// (speed_features.c:421).
    pub enable_sgr_ep_pruning: i32,
}

impl SpeedFeatures {
    /// Resolve the all-intra KEY-frame speed features for `speed`, transcribed
    /// from `set_allintra_speed_features_framesize_independent` +
    /// `set_allintra_speed_feature_framesize_dependent`.
    ///
    /// `allow_screen_content_tools` = `cm->features.allow_screen_content_tools`
    /// and `use_hbd` = `oxcf.use_highbitdepth` are the two inputs the cascade
    /// branches on. **Only speed 0 and 1 are fully modeled** (see the module
    /// docs); a `speed >= 2` argument applies the speed-1 deltas for the
    /// modeled fields but omits the additional speed-2+ deltas.
    pub fn set_allintra(speed: i32, allow_screen_content_tools: bool, _use_hbd: bool) -> Self {
        // ---- base (speed-0) values = allintra base block overrides layered
        //      over the init_*_sf defaults. ----
        let mut sf = SpeedFeatures {
            // part_sf
            less_rectangular_check_level: 1, // allintra base (speed_features.c:352)
            intra_cnn_based_part_prune_level: 0, // init_part_sf:2311
            reuse_best_prediction_for_part_ab: 0, // init_part_sf:2324
            ml_4_partition_search_level_index: 0, // init_part_sf:2305
            // intra_sf
            intra_pruning_with_hog: 1, // allintra base (speed_features.c:360)
            chroma_intra_pruning_with_hog: 0, // init_intra_sf default (off)
            prune_chroma_modes_using_luma_winner: false, // init_intra_sf default (off)
            disable_smooth_intra: false, // init_intra_sf:2438
            prune_filter_intra_level: 0, // init_intra_sf:2440
            prune_palette_search_level: 0, // init_intra_sf:2431
            prune_luma_palette_size_search_level: 1, // allintra base (:362)
            top_intra_model_count_allowed: TOP_INTRA_MODEL_COUNT, // init_intra_sf:2443
            // tx_sf
            adaptive_txb_search_level: 1, // allintra base (:366)
            intra_tx_size_search_init_depth_rect: 0, // init_tx_sf:2453
            intra_tx_size_search_init_depth_sqr: 1, // allintra base (:367)
            model_based_prune_tx_search_level: 1, // allintra base (:368)
            use_chroma_trellis_rd_mult: true, // allintra base (:370)
            use_rd_based_breakout_for_intra_tx_search: false, // init_tx_sf:2472 (see field doc — speed>=3 flip deferred to KB-8 2d-iv)
            // tx_sf.tx_type_search
            tx_ml_tx_split_thresh: 8500, // init_tx_sf:2458
            prune_2d_txfm_mode: TX_TYPE_PRUNE_1, // init_tx_sf:2457
            skip_tx_search: false, // init_tx_sf:2463
            use_reduced_intra_txset: true, // allintra base (:369)
            fast_intra_tx_type_search: 0, // init_tx_sf:2461
            winner_mode_tx_type_pruning: 0, // init_tx_sf:2466
            prune_tx_type_est_rd: false, // init_tx_sf:2465
            // winner_mode_sf (all off until speed>=4 — KB-8 chunk 2d wires these)
            enable_winner_mode_for_coeff_opt: false, // init:2511
            enable_winner_mode_for_use_tx_domain_dist: false, // init:2513
            enable_winner_mode_for_tx_size_srch: false, // init:2512
            multi_winner_mode_type: 0, // init:2514 (MULTI_WINNER_MODE_OFF)
            tx_size_search_level: 0, // init:2510
            // rd_sf
            perform_coeff_opt: 1, // allintra base (:383)
            tx_domain_dist_level: 0, // init_rd_sf:2501
            tx_domain_dist_thres_level: 0, // init_rd_sf:2502
            // lpf_sf
            cdef_pick_method: CDEF_FULL_SEARCH, // init_lpf_sf:2533
            dual_sgr_penalty_level: 0,
            enable_sgr_ep_pruning: 0,
        };

        // ---- if (speed >= 1) { ... } (speed_features.c:386-422 independent,
        //      :209-234 dependent) — the intra-still-relevant deltas. ----
        if speed >= 1 {
            // part_sf (independent + dependent)
            sf.intra_cnn_based_part_prune_level = if allow_screen_content_tools { 0 } else { 2 };
            sf.reuse_best_prediction_for_part_ab = 1;
            sf.ml_4_partition_search_level_index = 1; // dependent setter (:210)
            // intra_sf
            sf.prune_palette_search_level = 1;
            sf.prune_luma_palette_size_search_level = 2;
            sf.top_intra_model_count_allowed = 3;
            // tx_sf
            sf.adaptive_txb_search_level = 2;
            sf.intra_tx_size_search_init_depth_rect = 1;
            sf.model_based_prune_tx_search_level = 0;
            // tx_sf.tx_type_search
            sf.tx_ml_tx_split_thresh = 4000;
            sf.prune_2d_txfm_mode = TX_TYPE_PRUNE_2;
            sf.skip_tx_search = true;
            // rd_sf
            sf.perform_coeff_opt = 2;
            sf.tx_domain_dist_level = 1;
            sf.tx_domain_dist_thres_level = 1;
            // lpf_sf
            sf.cdef_pick_method = CDEF_FAST_SEARCH_LVL1;
            sf.dual_sgr_penalty_level = 1;
            sf.enable_sgr_ep_pruning = 1;
        }

        // ---- if (speed >= 2) { ... }
        //   framesize-DEPENDENT block (speed_features.c:236-267): the intra-still
        //   relevant delta is `ml_4_partition_search_level_index = 2` (:237). The
        //   other :236 fields are inert on the all-intra KEY path within this
        //   port's envelope: `use_square_partition_only_threshold = BLOCK_32X32`
        //   for <480p is UNCHANGED from speed 1 (KB-3); `partition_search_breakout_
        //   {dist,rate}_thr` is INTER-only (partition_search.c:4260 gates on
        //   `!frame_is_intra_only`); `prune_tx_type_using_stats` needs >=480p and
        //   `prune_tx_size_level` needs use_hbd (both false here).
        //   framesize-INDEPENDENT block (speed_features.c:424-437): the intra-still
        //   relevant deltas are `disable_smooth_intra=1` (:429), `intra_pruning_
        //   with_hog=2` (:430), `prune_filter_intra_level=1` (:431), and
        //   `perform_coeff_opt=3` (:433). `auto_mv_step_size`/`simple_motion_search_
        //   prune_agg` are inter/motion (inert on all-intra KEY); the two lpf_sf
        //   fields (`prune_wiener_based_on_src_var`, `prune_sgr_based_on_wiener`)
        //   are loop-restoration search (OFF in the allintra default envelope).
        //   qindex-DEPENDENT block (speed_features.c:2939): `ext_partition_eval_
        //   thresh = BLOCK_128X128` is gated on `!boosted`, and a KEY frame is
        //   always boosted (frame_is_kf_gf_arf via frame_is_intra_only,
        //   encoder.h:4055) => inert here.
        if speed >= 2 {
            // part_sf (framesize-dependent, :237)
            sf.ml_4_partition_search_level_index = 2;
            // intra_sf (:429-431)
            sf.disable_smooth_intra = true;
            sf.intra_pruning_with_hog = 2;
            sf.prune_filter_intra_level = 1;
            // rd_sf (:433)
            sf.perform_coeff_opt = 3;
        }

        // ---- if (speed >= 3) { ... }
        //   framesize-DEPENDENT block (speed_features.c:269-290): the intra-still
        //   relevant delta is `ml_4_partition_search_level_index = 3` (:271) — but
        //   at level 3 C switches to a DIFFERENT NN model with no threshold table
        //   (partition_strategy.c:1359), which the port's `part4_prune` treats by
        //   leaving the HORZ_4/VERT_4 allowed flags untouched (part4_prune.rs:238);
        //   the 4-way ML prune only bites on the HORZ_4/VERT_4 search at <=32x32
        //   blocks and is a byte no-op on this grid (empirically verified). Inert
        //   here: `ml_early_term_after_part_split_level = 0` (:270, both consumers
        //   `!frame_is_intra_only` — partition_search.c:4322/4335), `max_intra_bsize
        //   = BLOCK_32X32` (:285, only `init_mode_skip_mask`'s INTER ref-frame mask,
        //   rdopt.c:4217), `partition_search_breakout_{dist,rate}_thr` (:286-287,
        //   INTER), `prune_tx_size_level = 3` (:289, gated on `use_hbd`, false here).
        //   framesize-INDEPENDENT block (speed_features.c:439-469): the intra-still
        //   relevant deltas are `less_rectangular_check_level = 2` (:444),
        //   `chroma_intra_pruning_with_hog = 2` (:454) and `intra_pruning_with_hog
        //   = 3` (:455). Inert here: `high_precision_mv_usage`/`search_method`/
        //   `full_pixel_search_level`/`simple_motion_search_prune_agg` (motion/INTER),
        //   `recode_loop`/`screen_detection_mode2_fast_detection` (high-level; the
        //   harness bootstraps the parsed header and the fixed-q allintra path does
        //   not recode), the four `lpf_sf` wiener/sgr fields (loop-restoration search
        //   OFF), `prune_palette_search_level = 2` (:456, `av1_allow_palette` needs
        //   `allow_screen_content_tools` — palette search never runs on non-screen
        //   cells), `prune_ext_part_using_split_info = 1` (:446, 4-way split-info
        //   prune — HORZ_4/VERT_4 only, byte no-op on this grid), `adaptive_txb
        //   _search_level = 2` (:458, already 2 since speed 1), `use_skip_flag
        //   _prediction = 2` (:459, vestigial — indexes `predict_skip_levels` into
        //   `winner_mode_params->skip_txfm_level`, a table the port's non-winner-mode
        //   intra tx path does not consume) and `use_rd_based_breakout_for_intra_tx
        //   _search` (:460, intra tx-size-search early-exit — byte no-op on this grid).
        if speed >= 3 {
            // part_sf (framesize-independent, :444)
            sf.less_rectangular_check_level = 2;
            // part_sf (framesize-dependent, :271) — see the level-3 note above.
            sf.ml_4_partition_search_level_index = 3;
            // intra_sf (:454-455)
            sf.chroma_intra_pruning_with_hog = 2;
            sf.intra_pruning_with_hog = 3;
            // intra_sf (:456) — palette-search prune (inert: palette off on
            // non-screen cells); carried for source faithfulness.
            sf.prune_palette_search_level = 2;
            // tx_sf (:460) — the intra tx-size depth-loop early-exit tightening
            // (AOMMIN(ref_best_rd, best_rd)) + the winner-mode re-eval's
            // running-best ref (intra_mode_search.c:1201). Empirically a byte
            // no-op on the speed-3 gate grid (verified while this was false
            // through the KB-8 chunks); now set to the C value — the speed-3
            // gate re-verifies per landing.
            sf.use_rd_based_breakout_for_intra_tx_search = true;
        }

        // ---- if (speed >= 4) { ... } (speed_features.c:471-506 independent,
        //      :292-302 dependent). PARTIAL PORT — the intra-still-relevant deltas
        //      split into (A) wired now, (B) LIVE-but-unported (KB-8), (C) inert.
        //
        //   (A) WIRED on the bd8 4:2:0 allintra KEY path:
        //     - `prune_chroma_modes_using_luma_winner = 1` (:480): chroma-mode
        //       prune keyed on the luma winner (below; consumer intra_uv_rd.rs:1497,
        //       wired per-block via cfg.speed in partition_pick.rs).
        //     - `lpf_pick = LPF_PICK_FROM_FULL_IMAGE_NON_DUAL` (:496): the Y
        //       loop-filter-level search drops the two single-direction refine
        //       passes (picklpf.c:376). NOT an sf struct field here — wired via the
        //       `non_dual` flag on `lf_search::pick_filter_level` (the harness
        //       passes `speed >= 4`). Byte-affecting on nonzero-LF cells.
        //
        //   (B) PORTED by the KB-8 chunk series (SATD trellis-skip 16d4d85, stage
        //       policies 7bd30fb, USE_LARGESTALL 42bdffc, default-tx-type 96eeb71 +
        //       threading, two-pass skeleton 0ee9f97, est-rd prune 264bba4, and the
        //       flip in THIS commit). Original inventory (now live):
        //     - The WINNER-MODE two-pass subsystem for the LUMA intra search
        //       (`multi_winner_mode_type = MULTI_WINNER_MODE_DEFAULT` :504,
        //       `enable_winner_mode_for_coeff_opt` :502, `_for_use_tx_domain_dist`
        //       :503, `_for_tx_size_srch` :505): av1_rd_pick_intra_sby_mode runs the
        //       mode loop with MODE_EVAL params (intra_mode_search.c:1515) then
        //       re-evaluates the top-`winner_mode_count_allowed[..]=3` winners with
        //       WINNER_MODE_EVAL params (:1689-1737). The port's luma search is
        //       single-pass (intra_rd.rs:888). Governs the mono cells that diverge.
        //     - `perform_coeff_opt = 5` (:493): its DEFAULT_EVAL column is
        //       `[864, 97]` — the satd threshold 97 (< UINT_MAX) requires the SATD
        //       trellis-skip body, which is unimplemented (tx_search.rs:664). Feeds
        //       both the winner-mode luma passes AND the DEFAULT_EVAL chroma search.
        //     - `tx_domain_dist_thres_level = 3` (:494) — chroma/winner tx-domain
        //       dist threshold; part of the same eval-param set.
        //     - tx-type: `fast_intra_tx_type_search = 2` (:489, MODE_EVAL uses the
        //       default tx type only), `prune_2d_txfm_mode = TX_TYPE_PRUNE_3` (:490),
        //       `prune_tx_type_est_rd = 1` (:491), `winner_mode_tx_type_pruning = 2`
        //       (:488) — the tx-type search changes, coupled to the two-pass.
        //     - `top_intra_model_count_allowed` stays **3** at speed 4 — the `= 2`
        //       drop is the speed>=5 block (:533), NOT speed 4 (an earlier KB-8
        //       note mis-attributed it; verified against the source).
        //     - `prune_ext_part_using_split_info = 2` (:476): turns on the AB
        //       `evaluate_ab_partition_based_on_split` prune (partition_strategy.c:
        //       2009-2028) that the port omits as dead at <=speed3 (partition_pick
        //       .rs:1203).
        //
        //   (C) INERT on this path (byte no-op, verified):
        //     - `early_term_after_none_split = 1` (:477): only fires when NONE and
        //       SPLIT rd are BOTH INT64_MAX and `bsize != sb_size`
        //       (partition_search.c:5851) — NONE always yields a valid rd on
        //       textured content, so it never triggers here.
        //     - `ml_predict_breakout_level = 3` (:478): at bd8 the field is already
        //       3 from speed 0/1 (`use_hbd ? .. : 3`, :357/396), so speed 4 is a
        //       no-op at this bit depth.
        //     - Motion/MV (`subpel_search_method`, `simple_motion_search_prune_agg
        //       = LVL4`, `simple_motion_search_reduce_search_steps`,
        //       `simple_motion_subpel_force_stop`, `reduce_search_range`,
        //       `hash_max_8x8_intrabc_blocks`) — inter/motion/intrabc, none run on
        //       the all-intra KEY path.
        //     - TPL (`prune_starting_mv`, `subpel_force_stop`, `search_method`) — no
        //       TPL stage for a single all-intra KEY frame.
        //     - `cdef_pick_method = CDEF_FAST_SEARCH_LVL3` (:497) — CDEF off in the
        //       allintra envelope.
        //     - framesize-DEPENDENT (:292-302): `partition_search_breakout_dist_thr`
        //       (INTER), `prune_tx_type_using_stats = 2` (needs is_480p_or_larger —
        //       false on the {64,128}^2 grid).
        if speed >= 4 {
            // intra_sf (:480) — LIVE, consumer wired (see (A) above).
            sf.prune_chroma_modes_using_luma_winner = true;
            // ---- KB-8 winner-mode two-pass deltas (chunk 2d-iv flip) ----
            // tx_sf.tx_type_search
            sf.winner_mode_tx_type_pruning = 2; // :488
            sf.fast_intra_tx_type_search = 2; // :489 (MODE_EVAL default-tx-type only)
            sf.prune_2d_txfm_mode = 3; // TX_TYPE_PRUNE_3 (:490); stage-overridden per set_tx_type_prune
            sf.prune_tx_type_est_rd = true; // :491 (est-rd prune LIVE in the WINNER pass)
            // rd_sf
            sf.perform_coeff_opt = 5; // :493 (finite SATD threshold ⇒ trellis SATD-skip live)
            sf.tx_domain_dist_thres_level = 3; // :494 (thresholds {0,0,0})
            // winner_mode_sf (:502-505)
            sf.enable_winner_mode_for_coeff_opt = true;
            sf.enable_winner_mode_for_use_tx_domain_dist = true;
            sf.multi_winner_mode_type = 2; // MULTI_WINNER_MODE_DEFAULT (:504) → top-3 re-eval
            sf.enable_winner_mode_for_tx_size_srch = true; // :505 (MODE_EVAL → USE_LARGESTALL)
        }

        // ---- if (speed >= 5) { ... } (speed_features.c:508-525 independent;
        //      the framesize-DEPENDENT setter has NO speed-5 block — :302 jumps
        //      from `speed >= 4` to `speed >= 6`). Intra-still-relevant deltas:
        //
        //   LIVE on the allintra KEY path:
        //     - `multi_winner_mode_type = MULTI_WINNER_MODE_FAST` (:524): the
        //       luma winner-mode two-pass stores/re-evaluates the top-2 winners
        //       instead of speed 4's top-3 (`winner_mode_count_allowed`,
        //       rdopt_utils.h:236).
        //     - `ext_partition_eval_thresh = screen ? BLOCK_8X8 : BLOCK_16X16`
        //       (:510-511) + the qindex-dependent aggr=3 overrides (:2947-2963,
        //       `aggr = AOMMIN(4, speed-2)`; the `!is_480p_or_larger` arm sets
        //       BLOCK_128X128 UNCONDITIONALLY — no boosted/intra gate — so
        //       sub-480p KEY frames disable AB + 4-way entirely). Framesize +
        //       qindex dependent ⇒ NOT a struct field here; resolved by
        //       `partition_pick::ext_partition_eval_thresh_allintra_key` (the
        //       boosted/!intra arms are dead on KEY: frame_is_boosted holds and
        //       frame_is_intra_only kills the `!frame_is_intra_only` gates).
        //
        //   Set-then-overridden (transcribed for source faithfulness):
        //     - `chroma_intra_pruning_with_hog = 3` (:515) — zeroed right back
        //       by the :608-615 final override below, because
        //       `prune_chroma_modes_using_luma_winner` is 1 from speed >= 4.
        //       Net: chroma HOG is OFF at speed >= 4 (including 5).
        //
        //   Screen-content-only (non-screen value unchanged):
        //     - `intra_cnn_based_part_prune_level = screen ? 1 : 2` (:512-513):
        //       for non-screen frames 2 == the speed>=1 value (no change); for
        //       screen frames the CNN prune turns ON at level 1 (was 0).
        //
        //   INERT on this path (verified against source):
        //     - `simple_motion_search_prune_agg = SIMPLE_AGG_LVL5` (:509) —
        //       every simple-motion consumer is `!frame_is_intra_only` (the
        //       KB-3 elimination list).
        //     - `use_coarse_filter_level_search = 0` (:517) — the init default
        //       is already 0 (init_lpf_sf:2532; only the GOOD path :1356 sets
        //       it nonzero), so this is a literal no-op for allintra.
        //     - `disable_wiener_filter` / `disable_sgr_filter = true`
        //       (:519-520) — loop-restoration search is OFF in the allintra
        //       envelope (CLAUDE.md primary-envelope note).
        //     - `prune_mesh_search = PRUNE_MESH_SEARCH_LVL_2` (:522) — mesh
        //       search is motion/intrabc-only (intrabc mesh runs only on
        //       screen-content intra frames; no intrabc in this envelope).
        //     - qindex-dependent `speed == 5` `winner_mode_tx_type_pruning = 3`
        //       (:3059-3068) — gated `!(frame_is_intra_only || screen)`, never
        //       on an allintra KEY frame (stays 2).
        //     - qindex-dependent `speed >= 5` `prune_sub_8x8_partition_level =
        //       0` (:3070-3077) — the field is only raised at speed >= 6
        //       (:541), so at speed 5 it is already 0.
        //     - qindex-dependent `rect_partition_eval_thresh` (:2980-2991,
        //       aggr 0 -> 1 at speed 5) — gated `!boosted`; KEY is boosted.
        if speed >= 5 {
            sf.intra_cnn_based_part_prune_level = if allow_screen_content_tools { 1 } else { 2 };
            sf.chroma_intra_pruning_with_hog = 3; // :515 (zeroed below)
            sf.multi_winner_mode_type = 1; // MULTI_WINNER_MODE_FAST (:524) → top-2 re-eval
        }

        // The unconditional tail of set_allintra_speed_features_framesize_
        // independent (speed_features.c:608-616, AFTER every speed block):
        // "As the speed feature prune_chroma_modes_using_luma_winner already
        // constrains the number of chroma directional mode evaluations to a
        // maximum of 1, the HOG computation ... does not seem to help" —
        // the chroma HOG prune is force-DISABLED whenever the luma-winner
        // chroma prune is on (allintra speed >= 4; this also deadens the
        // speed>=5/6 `chroma_intra_pruning_with_hog = 3/4` settings in
        // allintra). KB-7 second root: the port kept the chroma HOG live at
        // speed 4, HOG-pruning a directional uv mode (e.g. UV_V_PRED) that C
        // evaluates and picks — flipping the speed-4 cq12/cq32 4:2:0 cells.
        if sf.prune_chroma_modes_using_luma_winner {
            sf.chroma_intra_pruning_with_hog = 0;
        }

        sf
    }

    /// Build the [`TxTypeSearchPolicy`] this speed level implies. `skip_trellis`
    /// (`!is_trellis_used(..)`, from CLI `disable_trellis_quant` / lossless) and
    /// `sharpness` (`oxcf.algo_cfg.sharpness`) are CLI-driven, not speed-driven,
    /// so they are threaded in by the caller. The threshold fields are resolved
    /// from the sf levels through the `DEFAULT_EVAL` column of the C tables
    /// (`av1_set_speed_features_qindex_dependent` copies these into
    /// `winner_mode_params`, speed_features.c:2794-2809; `get_rd_opt_coeff_thresh`
    /// selects `DEFAULT_EVAL` while `enable_winner_mode_for_coeff_opt == 0`,
    /// which holds for speed 0..=3).
    pub fn tx_type_search_policy(&self, skip_trellis: bool, sharpness: i32) -> TxTypeSearchPolicy {
        // The single-pass path is the DEFAULT_EVAL stage (its column resolution
        // is column 0 regardless of the winner-mode enables — see
        // `resolve_eval_col`), so speed 0..=3 (and the current, pre-two-pass
        // speed-4 luma/chroma search) are byte-identical to the prior hard-coded
        // derivation.
        self.tx_type_search_policy_for_stage(DEFAULT_EVAL, skip_trellis, sharpness)
    }

    /// Resolve the `MODE_EVAL_TYPES` column a threshold table is read at for the
    /// given eval `stage`, honouring the winner-mode enable gate. Mirrors the
    /// `!enable ? DEFAULT_EVAL : (is_winner ? WINNER_MODE_EVAL : MODE_EVAL)`
    /// selection shared by `get_rd_opt_coeff_thresh` (rd.h:317-339) and
    /// `set_tx_domain_dist_params` (rdopt_utils.h:516-544). At `DEFAULT_EVAL`
    /// (the single-pass caller) the column is always 0, so the winner-mode
    /// machinery is a strict no-op there.
    fn resolve_eval_col(stage: usize, enable: bool) -> usize {
        if stage == DEFAULT_EVAL || !enable {
            DEFAULT_EVAL
        } else {
            stage
        }
    }

    /// Build the [`TxTypeSearchPolicy`] for one winner-mode evaluation `stage`
    /// (`DEFAULT_EVAL` / `MODE_EVAL` / `WINNER_MODE_EVAL`), the data half of
    /// `set_mode_eval_params` (rdopt_utils.h:546). Only the coeff-opt and
    /// tx-domain-distortion columns are stage-selected here (via
    /// [`Self::resolve_eval_col`] and the `enable_winner_mode_for_coeff_opt` /
    /// `enable_winner_mode_for_use_tx_domain_dist` gates); the tx-size search
    /// method, tx-type set (`use_default_intra_tx_type`) and tx-type prune are
    /// carried by their own structs and threaded separately (KB-8 chunks 2b/2c).
    /// The `use_qm_dist_metric` branch (forces tx-domain dist on) is out of scope
    /// — QM is OFF on the allintra envelope (CLAUDE.md primary-envelope note).
    pub fn tx_type_search_policy_for_stage(
        &self,
        stage: usize,
        skip_trellis: bool,
        sharpness: i32,
    ) -> TxTypeSearchPolicy {
        let coeff_col = Self::resolve_eval_col(stage, self.enable_winner_mode_for_coeff_opt);
        let txd_col = Self::resolve_eval_col(stage, self.enable_winner_mode_for_use_tx_domain_dist);
        let coeff_row = &COEFF_OPT_THRESHOLDS[self.perform_coeff_opt as usize][coeff_col];
        // set_mode_eval_params(MODE_EVAL): use_default_intra_tx_type =
        // (fast_intra_tx_type_search == 2 || use_intra_default_tx_only). Only the
        // MODE_EVAL stage sets it; DEFAULT_EVAL/WINNER_MODE_EVAL force it to 0
        // (rdopt_utils.h:576/636). `use_intra_default_tx_only` is a CLI flag (off
        // on the allintra envelope).
        let use_default_intra_tx_type = stage == MODE_EVAL && self.fast_intra_tx_type_search == 2;
        TxTypeSearchPolicy {
            skip_trellis,
            coeff_opt_dist_threshold: coeff_row[0],
            coeff_opt_satd_threshold: coeff_row[1],
            use_transform_domain_distortion: TX_DOMAIN_DIST_TYPES[self.tx_domain_dist_level as usize]
                [txd_col] as u8,
            tx_domain_dist_threshold: TX_DOMAIN_DIST_THRESHOLDS
                [self.tx_domain_dist_thres_level as usize][txd_col],
            adaptive_txb_search_level: self.adaptive_txb_search_level,
            skip_tx_search: self.skip_tx_search,
            sharpness,
            use_chroma_trellis_rd_mult: self.use_chroma_trellis_rd_mult,
            intra_tx_size_init_depth_rect: self.intra_tx_size_search_init_depth_rect,
            intra_tx_size_init_depth_sqr: self.intra_tx_size_search_init_depth_sqr,
            use_default_intra_tx_type,
            // Non-screen textured envelope; screen-content would thread the real
            // cpi->use_screen_content_tools here.
            use_screen_content_tools: false,
            use_rd_based_breakout_for_intra_tx_search: self
                .use_rd_based_breakout_for_intra_tx_search,
            prune_tx_type_est_rd: self.prune_tx_type_est_rd,
            prune_2d_txfm_mode: {
                // set_tx_type_prune (rdopt_utils.h:498): the raw sf value,
                // overridden per stage when winner_mode_tx_type_pruning != 0.
                // DEFAULT_EVAL always passes winner_mode_tx_type_pruning = 0
                // (set_mode_eval_params:560), keeping the raw sf value.
                let wm_prune = if stage == DEFAULT_EVAL {
                    0
                } else {
                    self.winner_mode_tx_type_pruning
                };
                if wm_prune != 0 {
                    // prune_mode[4][2] (rdopt_utils.h:507): rows by
                    // winner_mode_tx_type_pruning-1, cols [MODE_EVAL, WINNER].
                    const PRUNE_MODE: [[i32; 2]; 4] = [[3, 0], [4, 0], [5, 2], [5, 3]];
                    PRUNE_MODE[(wm_prune - 1) as usize][usize::from(stage == WINNER_MODE_EVAL)]
                } else {
                    self.prune_2d_txfm_mode
                }
            },
        }
    }

    /// `set_tx_size_search_method` (rdopt_utils.h:478): the tx-size search
    /// method for one eval stage — `tx_size_search_methods[tx_size_search_
    /// level]` (speed_features.c:106, copied into `winner_mode_params` at
    /// :2822), column stage-selected under `enable_winner_mode_for_tx_size_
    /// srch` (same gate shape as [`Self::resolve_eval_col`]). Values are the
    /// `TX_SIZE_SEARCH_METHOD` enum re-exported from
    /// [`crate::tx_search`] (`USE_FULL_RD`=0 / `USE_FAST_RD`=1 /
    /// `USE_LARGESTALL`=2).
    pub fn tx_size_search_method_for_stage(&self, stage: usize) -> usize {
        // tx_size_search_methods[4][MODE_EVAL_TYPES] (speed_features.c:106).
        const TX_SIZE_SEARCH_METHODS: [[usize; 3]; 4] =
            [[0, 2, 0], [1, 2, 0], [2, 2, 0], [2, 2, 2]];
        let col = Self::resolve_eval_col(stage, self.enable_winner_mode_for_tx_size_srch);
        TX_SIZE_SEARCH_METHODS[self.tx_size_search_level as usize][col]
    }

    /// `winner_mode_count_allowed[multi_winner_mode_type]` (rdopt_utils.h:236):
    /// the top-N list size `store_winner_mode_stats` keeps — `{1, 2, 3}` for
    /// OFF / FAST / DEFAULT.
    pub fn winner_mode_count_allowed(&self) -> usize {
        [1usize, 2, 3][self.multi_winner_mode_type as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scaffold's central no-op guarantee: `set_allintra(0, ..)` reproduces
    /// exactly the values the pipeline hardcoded before this module existed, so
    /// no speed-0 byte-match gate can move. Compared field-by-field because
    /// `TxTypeSearchPolicy` derives no `PartialEq`.
    #[test]
    fn speed0_allintra_matches_hardcoded() {
        let sf = SpeedFeatures::set_allintra(0, false, false);

        // The tx-type search policy must equal the frozen `speed0_allintra()`.
        let got = sf.tx_type_search_policy(false, 0);
        let want = TxTypeSearchPolicy::speed0_allintra();
        assert_eq!(got.skip_trellis, want.skip_trellis);
        assert_eq!(got.coeff_opt_dist_threshold, want.coeff_opt_dist_threshold);
        assert_eq!(got.coeff_opt_satd_threshold, want.coeff_opt_satd_threshold);
        assert_eq!(
            got.use_transform_domain_distortion,
            want.use_transform_domain_distortion
        );
        assert_eq!(got.tx_domain_dist_threshold, want.tx_domain_dist_threshold);
        assert_eq!(got.adaptive_txb_search_level, want.adaptive_txb_search_level);
        assert_eq!(got.skip_tx_search, want.skip_tx_search);
        assert_eq!(got.sharpness, want.sharpness);
        assert_eq!(
            got.use_chroma_trellis_rd_mult,
            want.use_chroma_trellis_rd_mult
        );

        // The scalar sf fields the pipeline hardcoded at speed-0 allintra.
        assert_eq!(sf.less_rectangular_check_level, 1);
        assert_eq!(sf.intra_pruning_with_hog, 1);
        assert!(sf.use_chroma_trellis_rd_mult);
        assert_eq!(sf.perform_coeff_opt, 1);
        assert_eq!(sf.adaptive_txb_search_level, 1);
        assert_eq!(sf.intra_cnn_based_part_prune_level, 0);
        assert_eq!(sf.reuse_best_prediction_for_part_ab, 0);
        assert_eq!(sf.top_intra_model_count_allowed, TOP_INTRA_MODEL_COUNT);
        assert_eq!(sf.model_based_prune_tx_search_level, 1);
        assert_eq!(sf.prune_2d_txfm_mode, TX_TYPE_PRUNE_1);
        assert!(!sf.skip_tx_search);
        assert_eq!(sf.tx_domain_dist_level, 0);
        assert_eq!(sf.tx_domain_dist_thres_level, 0);
        assert_eq!(sf.intra_tx_size_search_init_depth_rect, 0);
        assert_eq!(sf.tx_ml_tx_split_thresh, 8500);
        assert_eq!(sf.cdef_pick_method, CDEF_FULL_SEARCH);
        // The speed-2 intra deltas stay at their speed-0 defaults at speed 0.
        assert!(!sf.disable_smooth_intra);
        assert_eq!(sf.prune_filter_intra_level, 0);
        assert_eq!(sf.intra_pruning_with_hog, 1);
        assert_eq!(sf.ml_4_partition_search_level_index, 0);
    }

    /// The speed-2 all-intra deltas, asserted against the source values
    /// (`set_allintra_*` speed>=2 blocks). At speed 2 the speed>=1 deltas remain
    /// in force and these additional fields flip.
    #[test]
    fn speed2_allintra_deltas_match_source() {
        let sf = SpeedFeatures::set_allintra(2, false, false);
        // NEW at speed 2 (framesize-independent :429-433 + dependent :237).
        assert!(sf.disable_smooth_intra); // :429
        assert_eq!(sf.intra_pruning_with_hog, 2); // :430
        assert_eq!(sf.prune_filter_intra_level, 1); // :431
        assert_eq!(sf.perform_coeff_opt, 3); // :433
        assert_eq!(sf.ml_4_partition_search_level_index, 2); // :237
        // Carried from speed 1 (unchanged at speed 2).
        assert_eq!(sf.adaptive_txb_search_level, 2);
        assert_eq!(sf.top_intra_model_count_allowed, 3);
        assert_eq!(sf.intra_tx_size_search_init_depth_rect, 1);
        assert!(sf.skip_tx_search);
        assert_eq!(sf.less_rectangular_check_level, 1); // still 1 (bumps to 2 at speed>=3)
        // Derived tx policy at speed 2: perform_coeff_opt=3 -> dist threshold 864.
        let pol = sf.tx_type_search_policy(false, 0);
        assert_eq!(pol.coeff_opt_dist_threshold, 864); // coeff_opt_thresholds[3][0][0]
        assert_eq!(pol.coeff_opt_satd_threshold, u32::MAX);
    }

    /// The speed-3 all-intra deltas, asserted against the source values
    /// (`set_allintra_*` speed>=3 blocks). At speed 3 the speed>=1/2 deltas remain
    /// in force and these additional fields flip.
    #[test]
    fn speed3_allintra_deltas_match_source() {
        let sf = SpeedFeatures::set_allintra(3, false, false);
        // NEW at speed 3 (framesize-independent :444/454/455/456 + dependent :271).
        assert_eq!(sf.less_rectangular_check_level, 2); // :444 (base 1 -> 2)
        assert_eq!(sf.chroma_intra_pruning_with_hog, 2); // :454 (0 -> 2, chroma HOG on)
        assert_eq!(sf.intra_pruning_with_hog, 3); // :455 (2 -> 3, luma HOG thresh -0.6)
        assert_eq!(sf.prune_palette_search_level, 2); // :456 (1 -> 2)
        assert_eq!(sf.ml_4_partition_search_level_index, 3); // :271 (2 -> 3)
        // Carried from speed 1/2 (unchanged at speed 3).
        assert!(sf.disable_smooth_intra); // speed>=2 :429
        assert_eq!(sf.prune_filter_intra_level, 1); // speed>=2 :431
        assert_eq!(sf.perform_coeff_opt, 3); // speed>=2 :433
        assert_eq!(sf.adaptive_txb_search_level, 2); // speed>=1 :406 (== speed>=3 :458)
        assert_eq!(sf.top_intra_model_count_allowed, 3); // speed>=1 :404
        assert!(sf.skip_tx_search); // speed>=1 :413
        // tx_sf (:460) — set at speed>=3 (C value; empirically byte-no-op on the
        // speed-3 gate grid, re-verified per landing).
        assert!(sf.use_rd_based_breakout_for_intra_tx_search);
        // Derived tx policy at speed 3: perform_coeff_opt=3 -> dist threshold 864
        // (winner-mode DEFAULT_EVAL column holds through speed 3).
        let pol = sf.tx_type_search_policy(false, 0);
        assert_eq!(pol.coeff_opt_dist_threshold, 864); // coeff_opt_thresholds[3][0][0]
        assert_eq!(pol.coeff_opt_satd_threshold, u32::MAX);
    }

    /// The speed-4 all-intra deltas THIS PARTIAL PORT models (see the module doc
    /// + the `if speed >= 4` block for the full LIVE/inert/unported breakdown).
    /// Only `prune_chroma_modes_using_luma_winner` is carried as an sf struct
    /// field (the NON_DUAL loop-filter delta lives in `lf_search`, and the
    /// winner-mode two-pass / coeff-opt-5 / tx-type deltas are unported KB-8).
    #[test]
    fn speed4_allintra_deltas_match_source() {
        let sf = SpeedFeatures::set_allintra(4, false, false);
        // NEW-and-WIRED at speed 4 (:480).
        assert!(sf.prune_chroma_modes_using_luma_winner);
        // The :454 speed>=3 value 2 is OVERRIDDEN to 0 by the unconditional
        // tail of set_allintra_speed_features_framesize_independent
        // (:608-616) whenever prune_chroma_modes_using_luma_winner is on
        // (allintra speed>=4) — KB-7 second root; empirically confirmed
        // (the instrumented C encoder computes ZERO chroma HOG masks at
        // cpu-used=4, and the speed-4 byte gate is 64/64 only with 0 here).
        assert_eq!(sf.chroma_intra_pruning_with_hog, 0);
        assert_eq!(sf.intra_pruning_with_hog, 3); // :455
        assert_eq!(sf.less_rectangular_check_level, 2); // :444
        assert!(sf.disable_smooth_intra); // speed>=2 :429
        assert_eq!(sf.prune_filter_intra_level, 1); // speed>=2 :431 (bumps to 2 only at speed>=5)
        // The KB-8 winner-mode two-pass deltas (chunk 2d-iv flip) — the REAL
        // speed-4 values per speed_features.c:471-506.
        assert_eq!(sf.perform_coeff_opt, 5); // :493
        assert_eq!(sf.tx_domain_dist_thres_level, 3); // :494
        assert_eq!(sf.winner_mode_tx_type_pruning, 2); // :488
        assert_eq!(sf.fast_intra_tx_type_search, 2); // :489
        assert_eq!(sf.prune_2d_txfm_mode, 3); // TX_TYPE_PRUNE_3 :490
        assert!(sf.prune_tx_type_est_rd); // :491
        assert!(sf.enable_winner_mode_for_coeff_opt); // :502
        assert!(sf.enable_winner_mode_for_use_tx_domain_dist); // :503
        assert_eq!(sf.multi_winner_mode_type, 2); // MULTI_WINNER_MODE_DEFAULT :504
        assert!(sf.enable_winner_mode_for_tx_size_srch); // :505
        assert_eq!(sf.winner_mode_count_allowed(), 3);
        assert!(sf.use_rd_based_breakout_for_intra_tx_search); // :460 (speed>=3)
        // top_intra_model_count_allowed stays 3 (the =2 drop is speed>=6, :533).
        assert_eq!(sf.top_intra_model_count_allowed, 3);
        // DEFAULT_EVAL (chroma + single-pass) policy: coeff row [5][0] = {864, 97}
        // — the finite SATD threshold turns the trellis SATD-skip ON; tx-domain
        // threshold 0 (type stays 1 = hybrid, tx_domain_dist_level unchanged).
        let pol = sf.tx_type_search_policy(false, 0);
        assert_eq!(pol.coeff_opt_dist_threshold, 864);
        assert_eq!(pol.coeff_opt_satd_threshold, 97);
        assert_eq!(pol.use_transform_domain_distortion, 1);
        assert_eq!(pol.tx_domain_dist_threshold, 0);
        // MODE_EVAL / WINNER policies (the two-pass stages).
        let me = sf.tx_type_search_policy_for_stage(MODE_EVAL, false, 0);
        assert_eq!((me.coeff_opt_dist_threshold, me.coeff_opt_satd_threshold), (142, 16));
        assert!(me.use_default_intra_tx_type);
        assert_eq!(me.prune_2d_txfm_mode, 4); // prune_mode[1][0] = TX_TYPE_PRUNE_4
        let win = sf.tx_type_search_policy_for_stage(WINNER_MODE_EVAL, false, 0);
        assert_eq!(
            (win.coeff_opt_dist_threshold, win.coeff_opt_satd_threshold),
            (u32::MAX, u32::MAX)
        );
        assert!(!win.use_default_intra_tx_type);
        assert_eq!(win.prune_2d_txfm_mode, 0); // prune_mode[1][1] = TX_TYPE_PRUNE_0
        assert!(win.prune_tx_type_est_rd);
        // Tx-size methods: MODE_EVAL LARGESTALL, WINNER FULL_RD.
        assert_eq!(sf.tx_size_search_method_for_stage(MODE_EVAL), 2);
        assert_eq!(sf.tx_size_search_method_for_stage(WINNER_MODE_EVAL), 0);
    }

    /// The speed-5 all-intra deltas, asserted against the source values
    /// (`set_allintra_*` speed>=5 block :508-525 + the :608-615 final
    /// override; the framesize-dependent setter has no speed-5 block). The
    /// framesize+qindex-dependent `ext_partition_eval_thresh` lives in
    /// `partition_pick::ext_partition_eval_thresh_allintra_key` (tested
    /// there), not on this struct.
    #[test]
    fn speed5_allintra_deltas_match_source() {
        let sf = SpeedFeatures::set_allintra(5, false, false);
        // NEW at speed 5: the winner-mode two-pass narrows to top-2 (:524).
        assert_eq!(sf.multi_winner_mode_type, 1); // MULTI_WINNER_MODE_FAST
        assert_eq!(sf.winner_mode_count_allowed(), 2);
        // chroma HOG: set to 3 (:515) then zeroed by the :608-615 override
        // (prune_chroma_modes_using_luma_winner on since speed 4).
        assert_eq!(sf.chroma_intra_pruning_with_hog, 0);
        assert!(sf.prune_chroma_modes_using_luma_winner);
        // Non-screen CNN prune level: 2 at speed>=1, UNCHANGED at speed 5
        // (:512-513 only moves the screen-content arm 0 -> 1).
        assert_eq!(sf.intra_cnn_based_part_prune_level, 2);
        let screen = SpeedFeatures::set_allintra(5, true, false);
        assert_eq!(screen.intra_cnn_based_part_prune_level, 1);
        // Everything else carries the speed-4 values (verified: no other
        // intra-still field appears in the speed>=5 block).
        assert_eq!(sf.top_intra_model_count_allowed, 3); // the =2 drop is speed>=6 (:533)
        assert_eq!(sf.prune_filter_intra_level, 1); // the =2 bump is speed>=6 (:529)
        assert_eq!(sf.intra_pruning_with_hog, 3); // the =4 bump is speed>=6 (:531)
        assert_eq!(sf.less_rectangular_check_level, 2);
        assert!(sf.disable_smooth_intra);
        assert_eq!(sf.perform_coeff_opt, 5);
        assert_eq!(sf.tx_domain_dist_thres_level, 3);
        assert_eq!(sf.winner_mode_tx_type_pruning, 2); // qindex-dep =3 arm is !intra-only
        assert_eq!(sf.fast_intra_tx_type_search, 2);
        assert_eq!(sf.prune_2d_txfm_mode, 3);
        assert!(sf.prune_tx_type_est_rd);
        assert!(sf.enable_winner_mode_for_coeff_opt);
        assert!(sf.enable_winner_mode_for_use_tx_domain_dist);
        assert!(sf.enable_winner_mode_for_tx_size_srch);
        assert!(sf.use_rd_based_breakout_for_intra_tx_search);
        assert_eq!(sf.ml_4_partition_search_level_index, 3);
        // Stage policies are identical to speed 4 (same table indices); only
        // the winner count differs. (Field-wise: TxTypeSearchPolicy derives no
        // PartialEq.)
        let sf4 = SpeedFeatures::set_allintra(4, false, false);
        for stage in [DEFAULT_EVAL, MODE_EVAL, WINNER_MODE_EVAL] {
            let p5 = sf.tx_type_search_policy_for_stage(stage, false, 0);
            let p4 = sf4.tx_type_search_policy_for_stage(stage, false, 0);
            assert_eq!(p5.coeff_opt_dist_threshold, p4.coeff_opt_dist_threshold);
            assert_eq!(p5.coeff_opt_satd_threshold, p4.coeff_opt_satd_threshold);
            assert_eq!(
                p5.use_transform_domain_distortion,
                p4.use_transform_domain_distortion
            );
            assert_eq!(p5.tx_domain_dist_threshold, p4.tx_domain_dist_threshold);
            assert_eq!(p5.prune_2d_txfm_mode, p4.prune_2d_txfm_mode);
            assert_eq!(p5.use_default_intra_tx_type, p4.use_default_intra_tx_type);
            assert_eq!(sf.tx_size_search_method_for_stage(stage), sf4.tx_size_search_method_for_stage(stage));
        }
        assert_eq!(sf4.winner_mode_count_allowed(), 3);
    }

    /// KB-8 chunk 2a: the stage-aware [`SpeedFeatures::tx_type_search_policy_for_stage`]
    /// derivation reproduces the per-`MODE_EVAL_TYPE` columns of the C threshold
    /// tables (`get_rd_opt_coeff_thresh` rd.h:313 + `set_tx_domain_dist_params`
    /// rdopt_utils.h:516), driven off the winner-mode enable gates. Validated on
    /// the REAL speed-4 parameter set (`perform_coeff_opt=5`,
    /// `tx_domain_dist_level=1`, `tx_domain_dist_thres_level=3`, both enables ON)
    /// applied to a hand-built sf — the set_allintra(4) flip is deferred to
    /// chunk 2d, so this validates the machinery independently.
    #[test]
    fn winner_mode_stage_policies_match_c_tables() {
        let mut sf = SpeedFeatures::set_allintra(4, false, false);
        sf.perform_coeff_opt = 5; // speed_features.c:493
        sf.tx_domain_dist_level = 1; // carried from speed 1 (:416); types row {1,2,0}
        sf.tx_domain_dist_thres_level = 3; // :494; thresholds row {0,0,0}
        sf.enable_winner_mode_for_coeff_opt = true; // :502
        sf.enable_winner_mode_for_use_tx_domain_dist = true; // :503
        sf.winner_mode_tx_type_pruning = 2; // :488
        sf.prune_2d_txfm_mode = 3; // TX_TYPE_PRUNE_3 (:490)
        sf.prune_tx_type_est_rd = true; // :491

        // coeff_opt_thresholds[5] = { {864,97}, {142,16}, {MAX,MAX} } [dist,satd].
        let def = sf.tx_type_search_policy_for_stage(DEFAULT_EVAL, false, 0);
        assert_eq!((def.coeff_opt_dist_threshold, def.coeff_opt_satd_threshold), (864, 97));
        let me = sf.tx_type_search_policy_for_stage(MODE_EVAL, false, 0);
        assert_eq!((me.coeff_opt_dist_threshold, me.coeff_opt_satd_threshold), (142, 16));
        let win = sf.tx_type_search_policy_for_stage(WINNER_MODE_EVAL, false, 0);
        assert_eq!(
            (win.coeff_opt_dist_threshold, win.coeff_opt_satd_threshold),
            (u32::MAX, u32::MAX),
            "winner pass: SATD threshold UINT_MAX ⇒ trellis always run"
        );

        // tx_domain_dist: types[level=1] = {1,2,0}, thresholds[thres_level=3] = {0,0,0}.
        assert_eq!((def.use_transform_domain_distortion, def.tx_domain_dist_threshold), (1, 0));
        assert_eq!((me.use_transform_domain_distortion, me.tx_domain_dist_threshold), (2, 0));
        assert_eq!((win.use_transform_domain_distortion, win.tx_domain_dist_threshold), (0, 0));

        // Tx-type prune resolution (set_tx_type_prune, winner_mode_tx_type_
        // pruning=2 -> prune_mode row 1 = {PRUNE_4, PRUNE_0}); DEFAULT_EVAL
        // keeps the raw sf PRUNE_3. est_rd carried un-staged.
        assert_eq!(def.prune_2d_txfm_mode, 3);
        assert_eq!(me.prune_2d_txfm_mode, 4);
        assert_eq!(win.prune_2d_txfm_mode, 0);
        assert!(def.prune_tx_type_est_rd && me.prune_tx_type_est_rd && win.prune_tx_type_est_rd);
        // use_default_intra_tx_type: MODE_EVAL only (fast_intra_tx_type_search=2).
        sf.fast_intra_tx_type_search = 2; // :489
        let me2 = sf.tx_type_search_policy_for_stage(MODE_EVAL, false, 0);
        assert!(me2.use_default_intra_tx_type);
        assert!(!sf.tx_type_search_policy_for_stage(DEFAULT_EVAL, false, 0).use_default_intra_tx_type);
        assert!(
            !sf.tx_type_search_policy_for_stage(WINNER_MODE_EVAL, false, 0)
                .use_default_intra_tx_type
        );

        // Tx-size method per stage (tx_size_search_methods[level=0] =
        // {FULL_RD, LARGESTALL, FULL_RD}, gated by enable_for_tx_size_srch).
        sf.enable_winner_mode_for_tx_size_srch = true; // :505
        assert_eq!(sf.tx_size_search_method_for_stage(DEFAULT_EVAL), 0); // USE_FULL_RD
        assert_eq!(sf.tx_size_search_method_for_stage(MODE_EVAL), 2); // USE_LARGESTALL
        assert_eq!(sf.tx_size_search_method_for_stage(WINNER_MODE_EVAL), 0); // USE_FULL_RD
        sf.enable_winner_mode_for_tx_size_srch = false;
        assert_eq!(sf.tx_size_search_method_for_stage(MODE_EVAL), 0); // gate off -> DEFAULT col

        // winner_mode_count_allowed = {1,2,3} (OFF/FAST/DEFAULT).
        sf.multi_winner_mode_type = 2; // MULTI_WINNER_MODE_DEFAULT (:504)
        assert_eq!(sf.winner_mode_count_allowed(), 3);
        sf.multi_winner_mode_type = 1; // MULTI_WINNER_MODE_FAST (speed>=5)
        assert_eq!(sf.winner_mode_count_allowed(), 2);
        sf.multi_winner_mode_type = 0;
        assert_eq!(sf.winner_mode_count_allowed(), 1);

        // The legacy single-pass entry point IS the DEFAULT_EVAL stage.
        let legacy = sf.tx_type_search_policy(false, 0);
        assert_eq!(legacy.coeff_opt_dist_threshold, def.coeff_opt_dist_threshold);
        assert_eq!(legacy.coeff_opt_satd_threshold, def.coeff_opt_satd_threshold);
        assert_eq!(legacy.use_transform_domain_distortion, def.use_transform_domain_distortion);
        assert_eq!(legacy.tx_domain_dist_threshold, def.tx_domain_dist_threshold);
    }

    /// The no-op guarantee the two-pass rests on: with the winner-mode enables
    /// OFF (every speed 0..=3, and the current pre-chunk-2d speed 4), ALL three
    /// eval stages resolve to the DEFAULT_EVAL column — so threading a stage
    /// through the search cannot change any byte until the enables are flipped.
    #[test]
    fn stage_policies_collapse_when_winner_mode_disabled() {
        // Even with a finite (speed>=4) coeff row, disabled enables collapse
        // every stage to DEFAULT_EVAL.
        let mut sf = SpeedFeatures::set_allintra(3, false, false);
        sf.perform_coeff_opt = 5;
        sf.tx_domain_dist_level = 1;
        sf.tx_domain_dist_thres_level = 3;
        // enables remain false (speed 3 defaults).
        for stage in [DEFAULT_EVAL, MODE_EVAL, WINNER_MODE_EVAL] {
            let p = sf.tx_type_search_policy_for_stage(stage, false, 0);
            assert_eq!(p.coeff_opt_dist_threshold, 864, "stage {stage} coeff dist");
            assert_eq!(p.coeff_opt_satd_threshold, 97, "stage {stage} coeff satd");
            assert_eq!(p.use_transform_domain_distortion, 1, "stage {stage} txd type");
            assert_eq!(p.tx_domain_dist_threshold, 0, "stage {stage} txd thresh");
        }
    }

    /// The speed-1 all-intra deltas, asserted against the source values (items
    /// 1-17 of the enumeration in STATUS.md). Non-screen-content branch.
    #[test]
    fn speed1_allintra_deltas_match_source() {
        let sf = SpeedFeatures::set_allintra(1, false, false);

        // part_sf
        assert_eq!(sf.intra_cnn_based_part_prune_level, 2);
        assert_eq!(sf.reuse_best_prediction_for_part_ab, 1);
        assert_eq!(sf.ml_4_partition_search_level_index, 1);
        assert_eq!(sf.less_rectangular_check_level, 1); // unchanged at speed 1
        // intra_sf
        assert_eq!(sf.prune_palette_search_level, 1);
        assert_eq!(sf.prune_luma_palette_size_search_level, 2);
        assert_eq!(sf.top_intra_model_count_allowed, 3);
        assert_eq!(sf.intra_pruning_with_hog, 1); // unchanged at speed 1
        // tx_sf
        assert_eq!(sf.adaptive_txb_search_level, 2);
        assert_eq!(sf.intra_tx_size_search_init_depth_rect, 1);
        assert_eq!(sf.model_based_prune_tx_search_level, 0);
        assert_eq!(sf.tx_ml_tx_split_thresh, 4000);
        assert_eq!(sf.prune_2d_txfm_mode, TX_TYPE_PRUNE_2);
        assert!(sf.skip_tx_search);
        // rd_sf
        assert_eq!(sf.perform_coeff_opt, 2);
        assert_eq!(sf.tx_domain_dist_level, 1);
        assert_eq!(sf.tx_domain_dist_thres_level, 1);
        // lpf_sf
        assert_eq!(sf.cdef_pick_method, CDEF_FAST_SEARCH_LVL1);

        // The derived tx policy at speed 1 (DEFAULT_EVAL column, winner-mode
        // off through speed 3).
        let pol = sf.tx_type_search_policy(false, 0);
        assert_eq!(pol.coeff_opt_dist_threshold, 1728); // coeff_opt_thresholds[2][0][0]
        assert_eq!(pol.coeff_opt_satd_threshold, u32::MAX);
        assert_eq!(pol.use_transform_domain_distortion, 1); // tx_domain_dist_types[1][0]
        assert_eq!(pol.tx_domain_dist_threshold, 22026); // tx_domain_dist_thresholds[1][0]
        assert_eq!(pol.adaptive_txb_search_level, 2);
        assert!(pol.skip_tx_search);
    }

    /// Screen-content flips only the CNN partition-prune level to 0 at speed 1.
    #[test]
    fn speed1_screen_content_disables_cnn_prune() {
        let sf = SpeedFeatures::set_allintra(1, true, false);
        assert_eq!(sf.intra_cnn_based_part_prune_level, 0);
        // everything else stays at the non-screen speed-1 values.
        assert_eq!(sf.reuse_best_prediction_for_part_ab, 1);
        assert_eq!(sf.perform_coeff_opt, 2);
    }
}
