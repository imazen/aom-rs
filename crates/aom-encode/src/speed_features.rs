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
//! (b) differ speed-0 → speed-1 on an intra-still KEY frame with CDEF +
//! loop-restoration search disabled (the current e2e harness envelope). The
//! speed cascade is transcribed faithfully for **speed 0 and speed 1 only**;
//! speeds >= 2 introduce further fields (`disable_smooth_intra` at 2,
//! `chroma_intra_pruning_with_hog` at 3, the winner-mode two-pass subsystem at
//! 4, ...) that are **not yet modeled** — do not treat a `set_allintra(n>=2)`
//! result as complete. The `lpf_sf` fields are carried for provenance but do
//! not yet affect bytes (the harness encodes the reference with CDEF +
//! restoration off).
//!
//! Source line citations are against libaom v3.14.1 (git 03087864).

use crate::tx_search::TxTypeSearchPolicy;

/// `MODE_EVAL_TYPES` index used everywhere in this port: winner-mode two-pass
/// evaluation (`enable_winner_mode_for_*`) first activates at speed >= 4 for
/// all-intra (speed_features.c:502-505), so speed 0..=3 always read the
/// `DEFAULT_EVAL` column (rd.h:95, `get_rd_opt_coeff_thresh` `!enable_winner`
/// branch, rd.h:317-321).
const DEFAULT_EVAL: usize = 0;

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
    /// (speed_features.c:387-388). CNN split-vs-nonsplit partition prune on
    /// intra SBs.
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
            // tx_sf.tx_type_search
            tx_ml_tx_split_thresh: 8500, // init_tx_sf:2458
            prune_2d_txfm_mode: TX_TYPE_PRUNE_1, // init_tx_sf:2457
            skip_tx_search: false, // init_tx_sf:2463
            use_reduced_intra_txset: true, // allintra base (:369)
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
        let coeff_row = &COEFF_OPT_THRESHOLDS[self.perform_coeff_opt as usize][DEFAULT_EVAL];
        TxTypeSearchPolicy {
            skip_trellis,
            coeff_opt_dist_threshold: coeff_row[0],
            coeff_opt_satd_threshold: coeff_row[1],
            use_transform_domain_distortion: TX_DOMAIN_DIST_TYPES[self.tx_domain_dist_level as usize]
                [DEFAULT_EVAL] as u8,
            tx_domain_dist_threshold: TX_DOMAIN_DIST_THRESHOLDS
                [self.tx_domain_dist_thres_level as usize][DEFAULT_EVAL],
            adaptive_txb_search_level: self.adaptive_txb_search_level,
            skip_tx_search: self.skip_tx_search,
            sharpness,
            use_chroma_trellis_rd_mult: self.use_chroma_trellis_rd_mult,
            intra_tx_size_init_depth_rect: self.intra_tx_size_search_init_depth_rect,
            intra_tx_size_init_depth_sqr: self.intra_tx_size_search_init_depth_sqr,
        }
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
