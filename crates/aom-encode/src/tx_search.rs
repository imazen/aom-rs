//! Transform search primitives (libaom `av1/encoder/tx_search.c`) — the
//! per-txb pieces of `search_tx_type` for the speed-0 all-intra path:
//! - [`get_tx_mask_intra`]: the allowed tx-type set for a luma intra txb
//!   (`get_tx_mask`, intra arm);
//! - [`av1_pixel_diff_dist`] (+ [`get_txb_visible_dimensions`]): the residual
//!   SSE / mean-squared-error the search's trellis/dist policies key off.
//!
//! Speed-0 all-intra sf resolution for `get_tx_mask` (each named, values from
//! `av1/encoder/speed_features.c`):
//! - `tx_type_search.use_reduced_intra_txset = 1`
//!   (`set_allintra_speed_features_framesize_independent`, speed-0 block)
//! - `tx_type_search.prune_tx_type_using_stats = 0` (default; allintra sets
//!   it only at higher speeds) — stats prune arm never runs
//! - `tx_type_search.prune_tx_type_est_rd = 0` (default) — `prune_txk_type*`
//!   never runs, so `txk_map` stays identity
//! - `prune_2d_txfm_mode = TX_TYPE_PRUNE_1` (default) but `prune_tx_2D` is
//!   gated `is_inter` — never runs for intra
//! - `txfm_params.use_default_intra_tx_type = 0` and
//!   `use_derived_intra_tx_type_set = 0` (MODE_EVAL with
//!   `fast_intra_tx_type_search = 0`, the speed-0 default)
//! - `x->rd_model = FULL_TXFM_RD` (set by `choose_tx_size_type_from_rd`)
//!
//! CLI-default tool flags (`aomenc` defaults): `enable_flip_idtx = 1`,
//! `use_intra_dct_only = 0`.

use aom_txb::ext_tx_set_type;

/// `TX_TYPES` (enums.h).
pub const TX_TYPES: usize = 16;

/// `av1_ext_tx_used_flag[EXT_TX_SET_TYPES]` (blockd.h): bit `t` set = tx type
/// `t` usable in that ext-tx set type.
pub const AV1_EXT_TX_USED_FLAG: [u16; 6] = [0x0001, 0x0201, 0x020F, 0x0E0F, 0x0FFF, 0xFFFF];

/// `av1_reduced_intra_tx_used_flag[INTRA_MODES]` (blockd.h): the reduced
/// intra tx set (sf `use_reduced_intra_txset >= 1`), per intra direction.
pub const AV1_REDUCED_INTRA_TX_USED_FLAG: [u16; 13] = [
    0x080F, 0x040F, 0x080F, 0x020F, 0x080F, 0x040F, 0x080F, 0x080F, 0x040F, 0x080F, 0x040F, 0x080F,
    0x0C0E,
];

/// `av1_derived_intra_tx_used_flag[INTRA_MODES]` (blockd.h): the
/// residual-statistics-derived set (sf `use_reduced_intra_txset == 2`).
pub const AV1_DERIVED_INTRA_TX_USED_FLAG: [u16; 13] = [
    0x0209, 0x0403, 0x0805, 0x020F, 0x0009, 0x0009, 0x0009, 0x0805, 0x0403, 0x0205, 0x0403, 0x0805,
    0x0209,
];

/// `fimode_to_intradir[FILTER_INTRA_MODES]` (blockd.h): the intra direction a
/// filter-intra mode maps to for tx-set/tx-type decisions.
pub const FIMODE_TO_INTRADIR: [usize; 5] = [0, 1, 2, 6, 0];

/// `DCT_ADST_TX_MASK` (txfm_common.h): DCT/ADST-only (kills FLIPADST + IDTX
/// combinations when `enable_flip_idtx` is off).
pub const DCT_ADST_TX_MASK: u16 = 0x000F;

/// `txsize_sqr_up_map[TX_SIZES_ALL]` (common_data.h): TX_SIZE -> square
/// TX_SIZE class rounding UP (0..4 = 4x4..64x64).
pub const TXSIZE_SQR_UP_MAP: [usize; 19] =
    [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];

/// `EXT_TX_SET_DTT4_IDTX_1DDCT` (enums.h `TxSetType`, value 3 — after
/// DCTONLY=0, DCT_IDTX=1, DTT4_IDTX=2): the intra set the reduced-txset sf
/// replaces with a per-direction table.
pub const EXT_TX_SET_DTT4_IDTX_1DDCT: usize = 3;

/// The `TxfmSearchParams` / tool-config gates `get_tx_mask` reads on the
/// intra path. [`TxMaskParams::speed0_allintra`] bakes the speed-0 values
/// (see module docs for the per-sf provenance).
#[derive(Clone, Copy, Debug)]
pub struct TxMaskParams {
    /// sf `tx_type_search.use_reduced_intra_txset` (0/1/2).
    pub use_reduced_intra_txset: u8,
    /// `txfm_params.use_derived_intra_tx_type_set`.
    pub use_derived_intra_tx_type_set: bool,
    /// `txfm_params.use_default_intra_tx_type` — default 0 (speed 0..=3).
    /// `set_mode_eval_params(MODE_EVAL)` sets this to 1 when
    /// `fast_intra_tx_type_search == 2` (allintra speed>=4) or
    /// `use_intra_default_tx_only`. When 1, `get_tx_mask` (tx_search.c:1807)
    /// overrides `txk_allowed` to [`get_default_tx_type_y`] BEFORE the ext-tx
    /// masking — the LUMA search then evaluates only the mode's default tx type
    /// (or DCT_DCT for tx>=32x32 / screen / lossless). Set true only in the
    /// winner-mode MODE_EVAL first pass (KB-8 chunk 2d).
    pub use_default_intra_tx_type: bool,
    /// `oxcf.txfm_cfg.enable_flip_idtx` (CLI default on).
    pub enable_flip_idtx: bool,
    /// `oxcf.txfm_cfg.use_intra_dct_only` (CLI default off).
    pub use_intra_dct_only: bool,
    /// sf `tx_sf.tx_type_search.prune_tx_type_using_stats` (0/1/2, default 0).
    /// ALLINTRA sets 1 at speed>=2 / 2 at speed>=4, but ONLY when
    /// `is_480p_or_larger` (speed_features.c:262/300). Reached only in the LUMA
    /// multi-type arm ([`get_tx_mask_intra`]); prunes tx types whose KF frame
    /// probability falls below the threshold (see the impl). 0 = off (all
    /// sub-480p gates + speed<2), byte-identical to the pre-stats path.
    pub prune_tx_type_using_stats: u8,
    /// `cpi->use_screen_content_tools` — an input to [`get_default_tx_type_y`]
    /// (screen content forces DCT_DCT). Only read when
    /// `use_default_intra_tx_type` is set; false on the non-screen textured
    /// envelope.
    pub use_screen_content_tools: bool,
}

impl TxMaskParams {
    /// Speed-0 all-intra defaults.
    pub fn speed0_allintra() -> Self {
        TxMaskParams {
            use_reduced_intra_txset: 1,
            use_derived_intra_tx_type_set: false,
            use_default_intra_tx_type: false,
            enable_flip_idtx: true,
            use_intra_dct_only: false,
            use_screen_content_tools: false,
            prune_tx_type_using_stats: 0,
        }
    }
}

/// `default_tx_type_probs[KF_UPDATE]` (encoder_utils.c:44, the `FRAME_UPDATE_TYPE
/// == KF_UPDATE` slab): the per-`TX_SIZE_ALL` (0..18) frame-probability row a lone
/// KEY still uses (`copy_frame_prob_info` seeds `frame_probs.tx_type_probs` from
/// these defaults when `prune_tx_type_using_stats` is on, encoder_utils.h:820).
/// The `1024, 0…` rows are the tx sizes forced to DCT_DCT before the multi-type
/// arm (sqr_up > TX_32X32), so the prune never reads them.
pub const DEFAULT_TX_TYPE_PROBS_KF: [[i32; 16]; 19] = [
    [221, 189, 214, 292, 0, 0, 0, 0, 0, 2, 38, 68, 0, 0, 0, 0],
    [262, 203, 216, 239, 0, 0, 0, 0, 0, 1, 37, 66, 0, 0, 0, 0],
    [315, 231, 239, 226, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [222, 188, 214, 287, 0, 0, 0, 0, 0, 2, 50, 61, 0, 0, 0, 0],
    [256, 182, 205, 282, 0, 0, 0, 0, 0, 2, 21, 76, 0, 0, 0, 0],
    [281, 214, 217, 222, 0, 0, 0, 0, 0, 1, 48, 41, 0, 0, 0, 0],
    [263, 194, 225, 225, 0, 0, 0, 0, 0, 2, 15, 100, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [170, 192, 242, 293, 0, 0, 0, 0, 0, 1, 68, 58, 0, 0, 0, 0],
    [199, 210, 213, 291, 0, 0, 0, 0, 0, 1, 14, 96, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

/// `thresh_arr[2][7]` (tx_search.c:1888) — indexed `[prune_tx_type_using_stats -
/// 1][update_type]`. For a lone KEY still `update_type = KF_UPDATE = 0`, so the
/// effective threshold is 10 for BOTH levels (the columns differ only for the
/// inter update types).
pub const TX_TYPE_STATS_THRESH: [[i32; 7]; 2] =
    [[10, 15, 15, 10, 15, 15, 15], [10, 17, 17, 10, 17, 17, 17]];

/// `get_default_tx_type(PLANE_TYPE_Y, xd, tx_size, use_screen_content_tools)`
/// (blockd.h:1175): the single luma tx type used when
/// `use_default_intra_tx_type` is set. Returns `DCT_DCT` (== `DEFAULT_INTER_TX
/// _TYPE`, blockd.h:42) for lossless, `tx_size >= TX_32X32` (enum value >= 3, so
/// 32x32/64x64 AND every rectangular size), or screen content; otherwise the
/// intra mode's default tx type `intra_mode_to_tx_type` = [`INTRA_MODE_TO_TX_
/// TYPE`]`[mode]`. (For a filter-intra block the caller passes `mode = DC_PRED`,
/// matching C's `mbmi->mode`.)
pub fn get_default_tx_type_y(
    mode: usize,
    tx_size: usize,
    lossless: bool,
    use_screen_content_tools: bool,
) -> usize {
    // TX_32X32 == 3 in the TX_SIZE enum; `tx_size >= TX_32X32` per blockd.h:1181.
    if lossless || tx_size >= 3 || use_screen_content_tools {
        0 // DCT_DCT
    } else {
        INTRA_MODE_TO_TX_TYPE[mode]
    }
}

/// `get_tx_mask` (tx_search.c, static) — the LUMA INTRA arm: the bitmask of
/// tx types `search_tx_type` iterates for one txb, plus `txk_allowed`
/// (`Some(t)` when exactly one specific type is allowed, `None` = the mask is
/// multi-type). The candidate order is the identity `txk_map` (the est-rd
/// reorder never runs at speed 0 — see module docs).
///
/// Out of scope (labelled): the inter arms (`default_inter_tx_type_prob_thresh`
/// frame-probability forcing, `prune_tx_2D`, stats prune), the est-rd prune,
/// `use_default_intra_tx_type` (`get_default_tx_type`; sf OFF at speed 0), the
/// `rd_model == LOW_TXFM_RD` DCT-only override (the pick loop runs
/// `FULL_TXFM_RD`), and the UV path (tx type inherited from Y).
pub fn get_tx_mask_intra(
    tx_size: usize,
    mode: usize,
    use_filter_intra: bool,
    filter_intra_mode: usize,
    lossless: bool,
    reduced_tx_set_used: bool,
    p: &TxMaskParams,
) -> (u16, Option<usize>) {
    let mut txk_allowed = TX_TYPES; // "all"
    // use_default_intra_tx_type override (tx_search.c:1807): pin the LUMA search
    // to the mode's single default tx type BEFORE the ext-tx masking below. C
    // uses mbmi->mode here (NOT the filter-intra-mapped intra_dir), so pass the
    // raw `mode` — for a filter-intra block the caller passes mode = DC_PRED.
    // The later lossless / sqr_up>32 / dct_only force-to-DCT check still applies
    // on top (matching the C ordering).
    if p.use_default_intra_tx_type {
        txk_allowed = get_default_tx_type_y(mode, tx_size, lossless, p.use_screen_content_tools);
    }
    let tx_set_type = ext_tx_set_type(tx_size, false, reduced_tx_set_used);

    let intra_dir = if use_filter_intra {
        FIMODE_TO_INTRADIR[filter_intra_mode]
    } else {
        mode
    };
    let mut ext_tx_used_flag =
        if p.use_reduced_intra_txset != 0 && tx_set_type == EXT_TX_SET_DTT4_IDTX_1DDCT {
            AV1_REDUCED_INTRA_TX_USED_FLAG[intra_dir]
        } else {
            AV1_EXT_TX_USED_FLAG[tx_set_type]
        };
    if p.use_reduced_intra_txset == 2 {
        ext_tx_used_flag &= AV1_DERIVED_INTRA_TX_USED_FLAG[intra_dir];
    }

    if lossless
        || TXSIZE_SQR_UP_MAP[tx_size] > 3
        || ext_tx_used_flag == 0x0001
        || p.use_intra_dct_only
    {
        txk_allowed = 0; // DCT_DCT
    }
    if !p.enable_flip_idtx {
        ext_tx_used_flag &= DCT_ADST_TX_MASK;
    }

    let mut allowed_tx_mask: u16;
    if txk_allowed < TX_TYPES {
        allowed_tx_mask = (1 << txk_allowed) & ext_tx_used_flag;
    } else if p.use_derived_intra_tx_type_set {
        allowed_tx_mask = AV1_DERIVED_INTRA_TX_USED_FLAG[intra_dir] & ext_tx_used_flag;
    } else {
        // `assert(plane == 0)` — the LUMA multi-type arm (tx_search.c:1881).
        allowed_tx_mask = ext_tx_used_flag;
        // prune_tx_type_using_stats (tx_search.c:1887): drop tx types whose KF
        // frame probability is below the threshold, but never drop the max-prob
        // allowed type. For a lone KEY still `update_type = KF_UPDATE = 0`, so
        // `tx_type_probs = default_tx_type_probs[KF_UPDATE][tx_size]` (seeded by
        // copy_frame_prob_info) and `thresh = thresh_arr[level-1][0]`.
        if p.prune_tx_type_using_stats != 0 {
            let probs = &DEFAULT_TX_TYPE_PROBS_KF[tx_size];
            let thresh = TX_TYPE_STATS_THRESH[(p.prune_tx_type_using_stats - 1) as usize][0];
            let mut prune: u16 = 0;
            let mut max_prob: i32 = -1;
            let mut max_idx: usize = 0;
            for i in 0..TX_TYPES {
                if probs[i] > max_prob && (allowed_tx_mask & (1 << i)) != 0 {
                    max_prob = probs[i];
                    max_idx = i;
                }
                if probs[i] < thresh {
                    prune |= 1 << i;
                }
            }
            if (prune >> max_idx) & 1 != 0 {
                prune &= !(1 << max_idx);
            }
            allowed_tx_mask &= !prune;
        }
        // est-rd prune / prune_tx_2D: handled by the caller's est-rd stage
        // (prune_txk_type_intra), not here.
    }

    if allowed_tx_mask == 0 {
        txk_allowed = 0; // DCT_DCT (plane 0)
        allowed_tx_mask = 1 << txk_allowed;
    }

    let single = if txk_allowed < TX_TYPES {
        Some(txk_allowed)
    } else {
        None
    };
    debug_assert!(single.is_none_or(|t| allowed_tx_mask == 1 << t));
    (allowed_tx_mask, single)
}

/// `intra_mode_to_tx_type` (blockd.h): the per-intra-direction default tx
/// type (DCT_DCT=0 / ADST_DCT=1 / DCT_ADST=2 / ADST_ADST=3), indexed by
/// `PREDICTION_MODE`.
pub const INTRA_MODE_TO_TX_TYPE: [usize; 13] = [
    0, // DC_PRED      -> DCT_DCT
    1, // V_PRED       -> ADST_DCT
    2, // H_PRED       -> DCT_ADST
    0, // D45_PRED     -> DCT_DCT
    3, // D135_PRED    -> ADST_ADST
    1, // D113_PRED    -> ADST_DCT
    2, // D157_PRED    -> DCT_ADST
    2, // D203_PRED    -> DCT_ADST
    1, // D67_PRED     -> ADST_DCT
    3, // SMOOTH_PRED  -> ADST_ADST
    1, // SMOOTH_V_PRED-> ADST_DCT
    2, // SMOOTH_H_PRED-> DCT_ADST
    3, // PAETH_PRED   -> ADST_ADST
];

/// `av1_get_tx_type` (blockd.h) for `PLANE_TYPE_UV` on an INTRA block: chroma
/// derives its tx type from the block's UV prediction mode
/// (`intra_mode_to_tx_type(mbmi, PLANE_TYPE_UV)` reads
/// `get_uv_mode(mbmi->uv_mode)` — UV_CFL_PRED maps to DC_PRED), demoted to
/// DCT_DCT when the ext-tx set of `tx_size` does not admit it
/// (`!av1_ext_tx_used[tx_set_type][tx_type]`), and forced to DCT_DCT outright
/// for lossless segments or `txsize_sqr_up_map[tx_size] > TX_32X32`.
/// (The inter arm — Y-plane `tx_type_map` sharing at the chroma-scaled
/// position — is out of scope for the intra RD search.)
pub fn uv_intra_tx_type(
    uv_mode: usize,
    lossless: bool,
    tx_size: usize,
    reduced_tx_set_used: bool,
) -> usize {
    if lossless || TXSIZE_SQR_UP_MAP[tx_size] > 3 {
        return 0; // DCT_DCT
    }
    let mode = aom_entropy::partition::get_uv_mode(uv_mode) as usize;
    let tx_type = INTRA_MODE_TO_TX_TYPE[mode];
    let tx_set_type = ext_tx_set_type(tx_size, false, reduced_tx_set_used);
    if AV1_EXT_TX_USED_FLAG[tx_set_type] & (1 << tx_type) == 0 {
        0
    } else {
        tx_type
    }
}

/// `get_tx_mask` (tx_search.c, static) — the CHROMA INTRA arm (`plane != 0`):
/// `txk_allowed` is pinned to [`uv_intra_tx_type`] ("tx_type of PLANE_TYPE_UV
/// should be the same as PLANE_TYPE_Y"), masked against the ext-tx-used flag
/// (which, under sf `use_reduced_intra_txset` on `EXT_TX_SET_DTT4_IDTX_1DDCT`
/// sizes, is the per-direction reduced table selected by the **LUMA** intra
/// direction `mbmi->mode` — or `fimode_to_intradir` when the luma winner used
/// filter-intra). KEY GOTCHA (tx_search.c:1942-46): when the masked set comes
/// out empty, the mask RESETS to `1 << uv_tx_type` — the UV tx type is used
/// even when outside the reduced set (the luma fallback is DCT_DCT instead).
/// Returns `(mask, txk_allowed)`; for chroma the mask is always exactly one
/// bit.
#[allow(clippy::too_many_arguments)]
pub fn get_tx_mask_uv_intra(
    tx_size: usize,
    uv_mode: usize,
    luma_mode: usize,
    luma_use_filter_intra: bool,
    luma_filter_intra_mode: usize,
    lossless: bool,
    reduced_tx_set_used: bool,
    p: &TxMaskParams,
) -> (u16, usize) {
    let uv_tx_type = uv_intra_tx_type(uv_mode, lossless, tx_size, reduced_tx_set_used);
    let mut txk_allowed = uv_tx_type;
    let tx_set_type = ext_tx_set_type(tx_size, false, reduced_tx_set_used);

    let intra_dir = if luma_use_filter_intra {
        FIMODE_TO_INTRADIR[luma_filter_intra_mode]
    } else {
        luma_mode
    };
    let mut ext_tx_used_flag =
        if p.use_reduced_intra_txset != 0 && tx_set_type == EXT_TX_SET_DTT4_IDTX_1DDCT {
            AV1_REDUCED_INTRA_TX_USED_FLAG[intra_dir]
        } else {
            AV1_EXT_TX_USED_FLAG[tx_set_type]
        };
    if p.use_reduced_intra_txset == 2 {
        ext_tx_used_flag &= AV1_DERIVED_INTRA_TX_USED_FLAG[intra_dir];
    }

    if lossless
        || TXSIZE_SQR_UP_MAP[tx_size] > 3
        || ext_tx_used_flag == 0x0001
        || p.use_intra_dct_only
    {
        txk_allowed = 0; // DCT_DCT
    }
    if !p.enable_flip_idtx {
        ext_tx_used_flag &= DCT_ADST_TX_MASK;
    }

    // txk_allowed < TX_TYPES always holds on the chroma arm.
    let mut allowed_tx_mask = (1u16 << txk_allowed) & ext_tx_used_flag;
    if allowed_tx_mask == 0 {
        // "txk_allowed = (plane ? uv_tx_type : DCT_DCT)" — the chroma reset.
        txk_allowed = uv_tx_type;
        allowed_tx_mask = 1 << txk_allowed;
    }
    debug_assert_eq!(allowed_tx_mask, 1 << txk_allowed);
    (allowed_tx_mask, txk_allowed)
}

/// The visible-dimension slice of `get_txb_dimensions` (rdopt_utils.h): a
/// txb's pixels clipped to the frame boundary. `mb_to_right_edge` /
/// `mb_to_bottom_edge` are the MACROBLOCKD edge fields (1/8-pel units,
/// negative when the block overhangs), `subsampling` the plane's.
#[allow(clippy::too_many_arguments)] // mirrors the C signature
pub fn get_txb_visible_dimensions(
    plane_bsize_w: usize,
    plane_bsize_h: usize,
    tx_w: usize,
    tx_h: usize,
    blk_row: usize,
    blk_col: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    subsampling_x: u32,
    subsampling_y: u32,
) -> (usize, usize) {
    let visible_height = if mb_to_bottom_edge >= 0 {
        tx_h
    } else {
        let block_rows = (mb_to_bottom_edge >> (3 + subsampling_y)) + plane_bsize_h as i32;
        (block_rows - ((blk_row as i32) << 2)).clamp(0, tx_h as i32) as usize
    };
    let visible_width = if mb_to_right_edge >= 0 {
        tx_w
    } else {
        let block_cols = (mb_to_right_edge >> (3 + subsampling_x)) + plane_bsize_w as i32;
        (block_cols - ((blk_col as i32) << 2)).clamp(0, tx_w as i32) as usize
    };
    (visible_width, visible_height)
}

/// `max_block_wide` / `max_block_high` (av1_common_int.h:1567/1581): the
/// frame-edge-clipped tx-block extent of a plane block, in that plane's 4-pel MI
/// units, PLUS the block's `mb_to_right_edge`/`mb_to_bottom_edge` (1/8-pel, luma
/// level — reused for `get_txb_visible_dimensions`). `bw_mi`/`bh_mi` are the
/// block's LUMA mi dims; `plane_bsize_w`/`plane_bsize_h` the plane block PIXEL
/// dims (= `block_size_wide/high[plane_bsize]`); `ss_x`/`ss_y` the plane
/// subsampling. For an interior block this returns the full plane block; for a
/// partial-SB edge block it clips to the visible frame area. Luma (ss = 0) gives
/// the same result as the interior loops' inline `min(max_blocks, mi-mi)` clip.
#[allow(clippy::too_many_arguments)]
pub fn max_block_units(
    mi_cols: i32,
    mi_rows: i32,
    mi_col: i32,
    mi_row: i32,
    bw_mi: i32,
    bh_mi: i32,
    plane_bsize_w: usize,
    plane_bsize_h: usize,
    ss_x: usize,
    ss_y: usize,
) -> (usize, usize, i32, i32) {
    let mb_to_right = (mi_cols - bw_mi - mi_col) * 32; // MI_SIZE(4) * 8
    let mb_to_bottom = (mi_rows - bh_mi - mi_row) * 32;
    let mut mbw = plane_bsize_w as i32;
    let mut mbh = plane_bsize_h as i32;
    if mb_to_right < 0 {
        mbw += mb_to_right >> (3 + ss_x);
    }
    if mb_to_bottom < 0 {
        mbh += mb_to_bottom >> (3 + ss_y);
    }
    (
        (mbw >> 2).max(0) as usize,
        (mbh >> 2).max(0) as usize,
        mb_to_right,
        mb_to_bottom,
    )
}

/// `av1_pixel_diff_dist` (tx_search.c): the residual (src - pred) SSE over the
/// txb's VISIBLE pixels, plus `block_mse_q8 = 256 * sse / visible_pels`
/// (`u32::MAX` when the visible area is empty). `diff` is the plane's
/// `src_diff` buffer (stride = plane block width); `blk_row`/`blk_col` in
/// 4-pel MI units.
pub fn av1_pixel_diff_dist(
    diff: &[i16],
    diff_stride: usize,
    blk_row: usize,
    blk_col: usize,
    visible_cols: usize,
    visible_rows: usize,
) -> (u64, u32) {
    let off = (blk_row * diff_stride + blk_col) << 2; // MI_SIZE_LOG2
    let sse = aom_dist::sum_squares_2d_i16(&diff[off..], diff_stride, visible_cols, visible_rows);
    let mse_q8 = if visible_cols > 0 && visible_rows > 0 {
        ((256 * sse) / (visible_cols as u64 * visible_rows as u64)) as u32
    } else {
        u32::MAX
    };
    (sse, mse_q8)
}

/// `dc_coeff_scale[TX_SIZES_ALL]` (encodemb.h:168): 12-bit fixed-point
/// DCT_DCT forward-transform normalization per tx size — sqrt(2) for 1:2/2:1,
/// 2 for 1:4/4:1, an extra 2 for <=8x8; 64-point sizes unsupported (0).
const DC_COEFF_SCALE: [u16; 19] = [
    1024, 2048, 4096, 4096, 0, 1448, 1448, 2896, 2896, 2896, 2896, 0, 0, 2048, 2048, 4096, 4096, 0,
    0,
];

/// `pixel_diff_stats` (tx_search.c:151): the residual block's SSE, mean and
/// variance over the VISIBLE 4x4s of the txb — the `predict_dc_only_block`
/// inputs. Returns `(sse, block_mse_q8, per_px_mean, block_var)` with C's
/// exact DOUBLE `norm_factor` arithmetic (the mse here can differ by one
/// q8 unit from [`av1_pixel_diff_dist`]'s integer division — when the
/// predict-dc path is live the DOUBLE form feeds the trellis/tx-domain
/// gates, so it must be this form, not the integer one). `per_px_mean` is
/// the transform-domain (<<7) signed mean; `block_var = sse - norm*sum²`
/// (double-truncated).
fn pixel_diff_stats(
    diff: &[i16],
    diff_stride: usize,
    visible_cols: usize,
    visible_rows: usize,
) -> (u64, u32, i64, u64) {
    // aom_sum_sse_2d_i16: sum + sum-of-squares over the visible region.
    let mut sse = 0u64;
    let mut sum = 0i32;
    for r in 0..visible_rows {
        let base = r * diff_stride;
        for &d in &diff[base..base + visible_cols] {
            sum += i32::from(d);
            sse += (i64::from(d) * i64::from(d)) as u64;
        }
    }
    if visible_cols > 0 && visible_rows > 0 {
        let norm_factor = 1.0f64 / (visible_cols * visible_rows) as f64;
        let sign_sum: i64 = if sum > 0 { 1 } else { -1 };
        // Conversion to transform domain (C truncating double->int64 cast).
        let per_px_mean = sign_sum * (((norm_factor * f64::from(sum.abs())) as i64) << 7);
        let block_mse_q8 = (norm_factor * (256 * sse) as f64) as u32;
        // (uint64_t)(norm_factor * sum * sum) — left-to-right double eval.
        let block_var = sse - (norm_factor * f64::from(sum) * f64::from(sum)) as u64;
        (sse, block_mse_q8, per_px_mean, block_var)
    } else {
        (sse, u32::MAX, i64::MAX, u64::MAX)
    }
}

// ---------------------------------------------------------------------------
// search_tx_type (tx_search.c) — the per-txb tx-type RD search, luma intra,
// speed-0 policy (see module docs), interior txbs (visible == full).
// ---------------------------------------------------------------------------

use crate::rd::rdcost;
use crate::{
    BlockContext, OptimizeInputs, QuantKind, QuantParams, XformQuantOptResult, xform_quant,
    xform_quant_optimize,
};
use aom_txb::{
    CoeffCostSet, CoeffCostTables, TxTypeCosts, cost_coeffs_txb, cost_coeffs_txb_laplacian,
    get_tx_type_cost,
};

/// `tx_size_2d[TX_SIZES_ALL]` (av1/common/common_data.h): pel count per tx.
pub const TX_SIZE_2D_TBL: [i64; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];

pub(crate) const TXS_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
pub(crate) const TXS_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];

/// `sqrt_tx_pixels_2d[TX_SIZES_ALL]` (tx_search.c:78) — ~sqrt(pel count) per tx.
const SQRT_TX_PIXELS_2D: [i64; 19] = [
    4, 8, 16, 32, 32, 6, 6, 12, 12, 23, 23, 32, 32, 8, 8, 16, 16, 23, 23,
];

/// `av1_get_tx_scale` (idct.c:24): `(pels > 256) + (pels > 1024)` where
/// `pels = tx_size_2d[tx_size]` (0 for ≤256-pel, 1 for 512/1024, 2 for >1024).
#[inline]
fn av1_get_tx_scale(tx_size: usize) -> i32 {
    let pels = TX_SIZE_2D_TBL[tx_size];
    i32::from(pels > 256) + i32::from(pels > 1024)
}

/// `av1_get_max_eob` (blockd.h:1601): the coefficient count `aom_satd` sums.
/// 64-point transforms preserve only the top-left area (capped at 1024 / 512).
#[inline]
fn av1_get_max_eob(tx_size: usize) -> usize {
    match tx_size {
        4 | 11 | 12 => 1024, // TX_64X64 / TX_32X64 / TX_64X32
        17 | 18 => 512,      // TX_16X64 / TX_64X16
        _ => TX_SIZE_2D_TBL[tx_size] as usize,
    }
}

/// `skip_trellis_opt_based_on_satd` (tx_search.c:1980), specialized to the
/// all-intra path where `dc_only_blk` is ALWAYS 0: allintra `dc_blk_pred_level`
/// stays 0 until speed≥6 (speed_features.c — the `=1` at :563 is the speed≥6
/// block), so `predict_dc_level = predict_dc_levels[0] = {0,0,0}` at every
/// MODE_EVAL_TYPE ⇒ `predict_dc_block = false` ⇒ `dc_only_blk = 0`. Hence the
/// SATD is always taken over all `av1_get_max_eob` coefficients (never the
/// `abs(coeff[0])` DC-only branch), and `av1_xform` (not `av1_xform_dc_only`)
/// runs. Forward-transforms `residual` for `tx_type`, sums |coeff|
/// (== `aom_satd_c`), scales by `RIGHT_SIGNED_SHIFT(satd, MAX_TX_SCALE −
/// av1_get_tx_scale)` then `>> (bd−8)`, and returns whether that EXCEEDS
/// `satd_threshold * qstep * sqrt_tx_pixels_2d[tx_size]` (⇒ SKIP trellis for
/// this block). The `skip_trellis || threshold == UINT_MAX` early return
/// (tx_search.c:1986) is applied by the caller before this runs.
fn skip_trellis_opt_based_on_satd(
    residual: &[i16],
    tx_size: usize,
    tx_type: usize,
    bd: u8,
    qstep: u32,
    satd_threshold: u32,
) -> bool {
    // av1_xform: forward transform into a full TX_W*TX_H buffer (64-point sizes
    // repack their valid area into the first av1_get_max_eob entries in-place;
    // TXS_W/TXS_H == the crate's TX_W/TX_H the quant path uses).
    let full = TXS_W[tx_size] * TXS_H[tx_size];
    let mut coeff = vec![0i32; full];
    aom_transform::txfm2d::av1_fwd_txfm2d(residual, &mut coeff, TXS_W[tx_size], tx_type, tx_size);
    let n_coeffs = av1_get_max_eob(tx_size);
    // aom_satd: Σ|coeff| over the first n_coeffs (aom_dist::hadamard::satd is
    // proven byte-identical to aom_satd_c — aom-dist/tests/hadamard_diff.rs).
    let mut satd = i64::from(aom_dist::hadamard::satd(&coeff[..n_coeffs]));
    // RIGHT_SIGNED_SHIFT(satd, MAX_TX_SCALE − tx_scale): MAX_TX_SCALE = 1, so the
    // shift is 1/0/−1 (a negative shift is a LEFT shift). Widened to i64 so the
    // 64-point left-shift can't overflow C's `int` domain on this path.
    let shift = 1i32 - av1_get_tx_scale(tx_size);
    satd = if shift < 0 {
        satd << (-shift)
    } else {
        satd >> shift
    };
    satd >>= i64::from(bd) - 8; // bd == 8 ⇒ no-op
    (satd as u64)
        > u64::from(satd_threshold) * u64::from(qstep) * (SQRT_TX_PIXELS_2D[tx_size] as u64)
}

/// `ROUND_POWER_OF_TWO` for i64.
#[inline]
fn round_power_of_two_i64(value: i64, n: i32) -> i64 {
    (value + ((1i64 << n) >> 1)) >> n
}

/// `prune_factors[5]` (tx_search.c:1071) — est-rd prune aggressiveness per
/// `prune_2d_txfm_mode` (TX_TYPE_PRUNE_0..4), scale 1000. (PRUNE_5 never
/// coexists with `prune_tx_type_est_rd` on a multi-type intra mask: at
/// speed>=6 it is the MODE_EVAL value, whose mask is single-type.)
const PRUNE_FACTORS: [i64; 5] = [200, 200, 120, 80, 40];

/// `sort_rd` (tx_search.c:1076): C's exact stable insertion sort of the
/// (est-rd, tx-type) pairs, ascending by rd.
fn sort_rd(rds: &mut [i64], txk: &mut [usize], len: usize) {
    for i in 1..len {
        for j in 0..i {
            if rds[j] > rds[i] {
                let temprd = rds[i];
                let tempi = txk[i];
                let mut k = i;
                while k > j {
                    rds[k] = rds[k - 1];
                    txk[k] = txk[k - 1];
                    k -= 1;
                }
                rds[j] = temprd;
                txk[j] = tempi;
                break;
            }
        }
    }
}

/// `prune_txk_type` (tx_search.c:1320) — the est-rd tx-type prune the
/// WINNER_MODE_EVAL pass runs at speed>=4 (`prune_tx_type_est_rd`, gated in
/// `get_tx_mask`'s multi-type arm on `num_allowed > 2`; no inter gate). For
/// each allowed tx type: forward transform + **B-quant** (av1_setup_quant with
/// AV1_XFORM_QUANT_B, quant_b_adapt=0 — regardless of the trellis setting),
/// FAST rate via [`cost_coeffs_txb_laplacian`] (+ tx-type cost — C folds it
/// inside `av1_cost_coeffs_txb_laplacian`), tx-domain distortion
/// (`dist_block_tx_domain`), `RDCOST` (0 fixed up to 1). Sorts `txk_map`
/// ascending by est-rd ([`sort_rd`] — non-allowed types fill the tail in
/// REVERSE tx-type order, matching C's `txk_map[last--]` walk) and returns the
/// PRUNE mask: everything except the best, minus the candidates whose relative
/// est-rd excess `1000*(rd-rd0)/rd0` stays under `prune_factor` (the loop
/// BREAKS at the first exceeder — later candidates stay pruned even if closer).
#[allow(clippy::too_many_arguments)]
fn prune_txk_type_intra(
    inp: &TxTypeSearchInputs,
    tx_size: usize,
    allowed_tx_mask: u16,
    prune_factor: i64,
    txk_map: &mut [usize; TX_TYPES],
    use_qm_dist_metric: bool,
) -> u16 {
    // C's prune_txk_type also runs av1_setup_qmatrix per tx type
    // (tx_search.c:1349): QM shapes the B-quant estimate AND (under the
    // QM-PSNR metric) the tx-domain distortion, exactly like the main loop.
    let mut qp_b = QuantParams::from_plane_rows(inp.rows, QuantKind::B, inp.bd, inp.lossless);
    if let Some(level) = inp.qm_level {
        qp_b = qp_b.with_qm(level, inp.plane);
    }
    let (txb_skip_ctx, dc_sign_ctx) = aom_txb::get_txb_ctx(
        inp.bctx.plane_bsize,
        tx_size,
        inp.bctx.plane,
        inp.bctx.above,
        inp.bctx.left,
    );
    let _ = dc_sign_ctx; // the laplacian estimate has no dc-sign term
    let mut rds = [0i64; TX_TYPES];
    let mut num_cand = 0usize;
    let mut last = TX_TYPES - 1;

    for tx_type in 0..TX_TYPES {
        if allowed_tx_mask & (1 << tx_type) == 0 {
            txk_map[last] = tx_type;
            last = last.wrapping_sub(1);
            continue;
        }
        // do txfm and quantization (av1_xform_quant, AV1_XFORM_QUANT_B).
        let xq = xform_quant(inp.residual, tx_size, tx_type, QuantKind::B, &qp_b, false);
        // estimate rate cost (av1_cost_coeffs_txb_laplacian, adjust_eob=0);
        // C includes get_tx_type_cost inside — the port's split adds it here.
        let mut rate_cost = cost_coeffs_txb_laplacian(
            &xq.qcoeff,
            xq.eob as usize,
            tx_size,
            tx_type,
            txb_skip_ctx as usize,
            inp.coeff_costs,
        );
        rate_cost += get_tx_type_cost(
            inp.tx_type_costs,
            inp.plane,
            tx_size,
            tx_type,
            false,
            inp.reduced_tx_set_used,
            inp.lossless,
            inp.use_filter_intra,
            inp.filter_intra_mode,
            inp.mode,
        );
        // tx-domain dist — the QM-PSNR metric weights it by the per-transform
        // forward matrix (same gate as the main loop, tx_search.c:1361).
        let (dist, _sse) = crate::dist_block_tx_domain_qm(
            &xq.coeff,
            &xq.dqcoeff,
            tx_size,
            inp.bd,
            crate::dist_qmatrix(&qp_b, tx_size, tx_type),
            aom_txb::scan(tx_size, tx_type),
            use_qm_dist_metric,
        );

        txk_map[num_cand] = tx_type;
        rds[num_cand] = rdcost(inp.rdmult, rate_cost, dist);
        if rds[num_cand] == 0 {
            rds[num_cand] = 1;
        }
        num_cand += 1;
    }

    if num_cand == 0 {
        return 0xFFFF;
    }

    sort_rd(&mut rds, txk_map, num_cand);
    let mut prune: u16 = !(1u16 << txk_map[0]);

    // 0 < prune_factor <= 1000 controls aggressiveness.
    for idx in 1..num_cand {
        let factor = 1000 * (rds[idx] - rds[0]) / rds[0];
        if factor < prune_factor {
            prune &= !(1u16 << txk_map[idx]);
        } else {
            break;
        }
    }
    prune
}

/// The trellis RD multiplier `av1_optimize_txb` derives from the block
/// `x->rdmult` (encodetxb.h `plane_rd_mult` tables; luma-intra entry is 17 in
/// BOTH the default and `use_chroma_trellis_rd_mult` tables, so the speed-0
/// allintra sf `use_chroma_trellis_rd_mult = 1` is a no-op for luma):
/// `ROUND_POWER_OF_TWO(rdmult * (8 - sharpness) * (17 << (2*(bd-8))), rshift)`
/// where `rshift = 5` for the PSNR-family tunes and **7** for
/// `tune=IQ`/`tune=SSIMULACRA2` (`iq_tuning`; txb_rdopt.c:378-386 — the
/// larger shift biases the trellis towards keeping more coefficients).
#[inline]
pub fn trellis_rdmult_intra_y(rdmult: i32, sharpness: i32, bd: u8, iq_tuning: bool) -> i64 {
    round_power_of_two_i64(
        (rdmult as i64) * ((8 - sharpness) as i64) * ((17i64) << (2 * (bd as i32 - 8))),
        if iq_tuning { 7 } else { 5 },
    )
}

/// The intra trellis RD multiplier for any plane. `av1_optimize_txb`
/// (txb_rdopt.c:387-395) selects the table by the sf
/// `tx_sf.use_chroma_trellis_rd_mult`:
/// - sf == 1: `plane_rd_mult_chroma[is_inter=0] = {17, 13}` (encodetxb.h:266)
///   — set by the ALLINTRA (speed_features.c:370) and RT (:2035)
///   framesize-independent setters;
/// - sf == 0: the default `plane_rd_mult[is_inter=0] = {17, 20}`
///   (encodetxb.h:270) — **usage GOOD never sets the sf** (default 0,
///   speed_features.c:2474; absent from `set_good_speed_features_framesize_
///   independent`), so the GOOD KEY-frame envelope uses **20** for chroma.
///
/// Luma is 17 in both tables — the flag only matters for chroma. `iq_tuning`
/// selects the `tune=IQ`/`tune=SSIMULACRA2` `rshift = 7` arm (vs the default
/// 5; txb_rdopt.c:378-386).
#[inline]
pub fn trellis_rdmult_intra(
    rdmult: i32,
    sharpness: i32,
    bd: u8,
    plane: usize,
    use_chroma_trellis_rd_mult: bool,
    iq_tuning: bool,
) -> i64 {
    let mult: i64 = if plane == 0 {
        17
    } else if use_chroma_trellis_rd_mult {
        13
    } else {
        20
    };
    round_power_of_two_i64(
        (rdmult as i64) * ((8 - sharpness) as i64) * (mult << (2 * (bd as i32 - 8))),
        if iq_tuning { 7 } else { 5 },
    )
}

/// The speed-0 policy knobs of `search_tx_type` (each documented with its
/// speed-0 value in the module docs / commit message).
#[derive(Clone, Copy, Debug)]
pub struct TxTypeSearchPolicy {
    /// `!is_trellis_used(optimize_coefficients, DRY_RUN_NORMAL)` — speed-0
    /// allintra: `FULL_TRELLIS_OPT` (CLI `disable_trellis_quant = 0`, not
    /// lossless) => `false`.
    pub skip_trellis: bool,
    /// `txfm_params->coeff_opt_thresholds[0]` (block-MSE/qstep^2 gate for the
    /// trellis) — speed 0: `coeff_opt_thresholds[perform_coeff_opt=1]
    /// [DEFAULT_EVAL][0] = 3200` (enable_winner_mode_for_coeff_opt = 0).
    pub coeff_opt_dist_threshold: u32,
    /// `coeff_opt_thresholds[1]` (SATD gate) — speed 0: `UINT_MAX`, which
    /// short-circuits `skip_trellis_opt_based_on_satd` before any SATD work
    /// (the SATD body is unported; reaching it panics).
    pub coeff_opt_satd_threshold: u32,
    /// `txfm_params->use_transform_domain_distortion` — speed 0:
    /// `tx_domain_dist_types[tx_domain_dist_level=0][DEFAULT_EVAL] = 0`
    /// (pixel-domain during the loop, with the 64-pt/high-energy hybrid).
    pub use_transform_domain_distortion: u8,
    /// `txfm_params->tx_domain_dist_threshold` — speed 0:
    /// `tx_domain_dist_thresholds[0][DEFAULT_EVAL] = UINT_MAX`.
    pub tx_domain_dist_threshold: u32,
    /// sf `tx_sf.adaptive_txb_search_level` — speed-0 allintra: 1.
    pub adaptive_txb_search_level: i32,
    /// sf `tx_sf.tx_type_search.skip_tx_search` — speed 0: 0.
    pub skip_tx_search: bool,
    /// `oxcf.algo_cfg.sharpness` (CLI default 0).
    pub sharpness: i32,
    /// sf `tx_sf.use_chroma_trellis_rd_mult` — the chroma trellis-table
    /// select (see [`trellis_rdmult_intra`]): ALLINTRA/RT set 1, **usage
    /// GOOD leaves the default 0** (chroma multiplier 20, not 13). Luma
    /// unaffected.
    pub use_chroma_trellis_rd_mult: bool,
    /// sf `tx_sf.intra_tx_size_search_init_depth_rect` — the tx-size depth
    /// search FLOOR for a rectangular (w != h) intra block: 0 at speed 0
    /// (searches down to the smallest sub-tx), 1 at speed >= 1 (stops one
    /// shrink-step shallower, skipping that smallest sub-tx). Mirrors
    /// `SpeedFeatures::intra_tx_size_search_init_depth_rect` (speed_features.c:409).
    pub intra_tx_size_init_depth_rect: i32,
    /// sf `tx_sf.intra_tx_size_search_init_depth_sqr` — the square (w == h)
    /// block floor; 1 for all speeds (speed_features.c:367).
    pub intra_tx_size_init_depth_sqr: i32,
    /// `txfm_params.use_default_intra_tx_type` — the MODE_EVAL first-pass
    /// tx-type restriction (see [`TxMaskParams::use_default_intra_tx_type`]).
    /// Threaded into [`search_tx_type_intra`]'s tx-mask; false for DEFAULT_EVAL
    /// / WINNER_MODE_EVAL, so speed 0..=3 (single-pass DEFAULT_EVAL) is
    /// byte-identical. Set true only by the MODE_EVAL stage at speed>=4.
    pub use_default_intra_tx_type: bool,
    /// `cpi->use_screen_content_tools` — the other [`get_default_tx_type_y`]
    /// input, threaded alongside `use_default_intra_tx_type`. False on the
    /// non-screen textured envelope.
    pub use_screen_content_tools: bool,
    /// sf `tx_sf.use_rd_based_breakout_for_intra_tx_search` — default 0
    /// (init_tx_sf:2472), allintra speed>=3 -> 1 (speed_features.c:460). Two
    /// consumers: (a) the tx-size depth loop's per-depth early-exit threshold
    /// becomes `AOMMIN(ref_best_rd, best_rd)` (tx_search.c:3030); (b) the
    /// winner-mode re-eval `intra_block_yrd` passes the running `*best_rd`
    /// instead of INT64_MAX (intra_mode_search.c:1201). Left false by the
    /// port's speed 0..=3 derivation (empirically a byte no-op on the speed-3
    /// gate grid, but kept false until the KB-8 speed-4 flip re-verifies it).
    pub use_rd_based_breakout_for_intra_tx_search: bool,
    /// sf `tx_sf.tx_type_search.prune_tx_type_est_rd` — default 0
    /// (init_tx_sf:2465); allintra speed>=4 -> 1 (speed_features.c:491). Gates
    /// the [`prune_txk_type_intra`] est-rd prune in `get_tx_mask`'s multi-type
    /// arm (`num_allowed > 2`, tx_search.c:1911 — NO inter gate, so it runs on
    /// intra in the WINNER_MODE_EVAL pass). Also REORDERS the tx-type search
    /// order (`txk_map` sorted by est-rd). False for speed 0..=3.
    pub prune_tx_type_est_rd: bool,
    /// sf `tx_sf.tx_type_search.prune_tx_type_using_stats` (0/1/2) — the
    /// KF-frame-probability luma tx-type prune (see
    /// [`TxMaskParams::prune_tx_type_using_stats`]). Framesize+speed dependent:
    /// ALLINTRA sets 1 at speed>=2 / 2 at speed>=4, is_480p_or_larger only. 0
    /// on every sub-480p gate and speed<2 (byte-identical to the pre-stats path).
    pub prune_tx_type_using_stats: u8,
    /// `txfm_params->prune_2d_txfm_mode` — the STAGE-resolved tx-type prune
    /// level (`set_tx_type_prune`, rdopt_utils.h:498): the raw sf value when
    /// `winner_mode_tx_type_pruning == 0`, else
    /// `prune_mode[winner_mode_tx_type_pruning-1][is_winner_mode]`. Its only
    /// INTRA consumer is [`PRUNE_FACTORS`] indexing inside the est-rd prune
    /// (the 2D-NN prune `prune_tx_2D` is inter-only, tx_search.c:1935). Inert
    /// while `prune_tx_type_est_rd` is false.
    pub prune_2d_txfm_mode: i32,
    /// `txfm_params->predict_dc_level` — the STAGE-resolved skip/DC-block
    /// prediction level (`set_mode_eval_params` copies
    /// `predict_dc_levels[dc_blk_pred_level][stage]`, rdopt_utils.h:564/588/
    /// 616; table at speed_features.c:137-139). 0 through speed 5 on every
    /// stage (`dc_blk_pred_level = 0` -> row {0,0,0}); at allintra speed>=6
    /// (`dc_blk_pred_level = 1` -> row {1,1,0}) the DEFAULT_EVAL + MODE_EVAL
    /// tx-type searches run `predict_dc_only_block` (tx_search.c:2011): a
    /// residual-mean/variance skip-block predictor that short-circuits the
    /// whole per-txb search with an eob-0 skip result. Level 1 never sets
    /// `dc_only_blk` (that needs level > 1).
    pub predict_dc_level: u32,
    /// sf `tx_sf.prune_intra_tx_depths_using_nn` — default 0; allintra
    /// speed>=6 -> 1 (speed_features.c:553). Enables the 8x8-block NN
    /// tx-depth prune in `choose_tx_size_type_from_rd`'s largest-depth walk
    /// (stage-independent — copied from the raw sf like
    /// `prune_tx_type_est_rd`); gates the [`NnDepthPruneCtx`] threading.
    pub prune_intra_tx_depths_using_nn: bool,
    /// `oxcf.txfm_cfg.enable_flip_idtx` (`--enable-flip-idtx`, CLI default
    /// on) — stage-INDEPENDENT (C reads oxcf directly in `get_tx_mask`,
    /// tx_search.c). When off, masks the FLIPADST/IDTX tx-type family out of
    /// every ext-tx set ([`DCT_ADST_TX_MASK`]).
    pub enable_flip_idtx: bool,
    /// `oxcf.txfm_cfg.use_intra_dct_only` (`--use-intra-dct-only`, CLI
    /// default off) — stage-independent like `enable_flip_idtx`. Forces
    /// `txk_allowed = DCT_DCT` for every luma intra txb.
    pub use_intra_dct_only: bool,
    /// `oxcf.txfm_cfg.enable_tx_size_search` (`--enable-tx-size-search`, CLI
    /// default on). OFF forces `winner_mode_sf.tx_size_search_level = 3`
    /// (speed_features.c:2726) → USE_LARGESTALL at every eval stage → the
    /// frame codes `tx_mode = TX_MODE_LARGEST` (`select_tx_mode`). Carried on
    /// the policy because the policy is the port's oxcf.txfm_cfg carrier:
    /// consumers are the single-pass method pick (intra_rd.rs, USE_FULL_RD →
    /// USE_LARGESTALL), the winner-mode sf derivation (partition_pick.rs,
    /// level 3), and the leaf `tx_mode_is_select` init (partition_pick.rs).
    pub enable_tx_size_search: bool,
    /// `txfm_params->use_qm_dist_metric` (`set_mode_eval_params`,
    /// rdopt_utils.h:554: `dist_metric == AOM_DIST_METRIC_QM_PSNR`). Gates the
    /// QM-weighted transform-domain distortion in the search
    /// ([`crate::dist_block_tx_domain_qm`]) and upgrades the trellis to the
    /// QM-PSNR arm ([`crate::QuantParams::with_qm_dist_metric`]). Callers set
    /// it via [`Self::with_tune_knobs`] so the tx-domain-distortion force
    /// (rdopt_utils.h:516-522) is applied consistently. Default false (PSNR).
    pub use_qm_dist_metric: bool,
    /// `tune_cfg.tuning ∈ {AOM_TUNE_IQ, AOM_TUNE_SSIMULACRA2}` — selects the
    /// trellis `rshift = 7` (vs 5) arm in [`trellis_rdmult_intra`]
    /// (txb_rdopt.c:378-386). Default false (PSNR-family tunes).
    pub iq_tuning: bool,
}

impl TxTypeSearchPolicy {
    /// Speed-0 all-intra defaults (provenance per field above).
    pub fn speed0_allintra() -> Self {
        TxTypeSearchPolicy {
            skip_trellis: false,
            coeff_opt_dist_threshold: 3200,
            coeff_opt_satd_threshold: u32::MAX,
            use_transform_domain_distortion: 0,
            tx_domain_dist_threshold: u32::MAX,
            adaptive_txb_search_level: 1,
            skip_tx_search: false,
            sharpness: 0,
            use_chroma_trellis_rd_mult: true,
            intra_tx_size_init_depth_rect: 0,
            intra_tx_size_init_depth_sqr: 1,
            use_default_intra_tx_type: false,
            use_screen_content_tools: false,
            use_rd_based_breakout_for_intra_tx_search: false,
            prune_tx_type_est_rd: false,
            prune_tx_type_using_stats: 0,
            prune_2d_txfm_mode: 1, // TX_TYPE_PRUNE_1 (init_tx_sf:2457); inert while est_rd off
            predict_dc_level: 0,   // predict_dc_levels[0][*] (dc_blk_pred_level 0 through speed 5)
            prune_intra_tx_depths_using_nn: false, // default off (speed>=6 only)
            enable_flip_idtx: true, // --enable-flip-idtx CLI default
            use_intra_dct_only: false, // --use-intra-dct-only CLI default
            enable_tx_size_search: true, // --enable-tx-size-search CLI default
            use_qm_dist_metric: false, // AOM_DIST_METRIC_PSNR (default)
            iq_tuning: false,      // AOM_TUNE_PSNR-family (default)
        }
    }

    /// Apply the `oxcf.tune_cfg` knobs ([`crate::TuneKnobs`]) to a derived
    /// policy. Under the QM-PSNR distortion metric C FORCES transform-domain
    /// distortion on unconditionally — `set_tx_domain_dist_params`'s early
    /// return (`use_transform_domain_distortion = 1`,
    /// `tx_domain_dist_threshold = 0`, rdopt_utils.h:516-522) — because the
    /// QM-weighted PSNR is computed in transform space. A default
    /// (`TuneKnobs::default()`) input returns the policy unchanged.
    pub fn with_tune_knobs(mut self, tune: crate::TuneKnobs) -> Self {
        self.use_qm_dist_metric = tune.use_qm_dist_metric;
        self.iq_tuning = tune.iq_tuning;
        if tune.use_qm_dist_metric {
            self.use_transform_domain_distortion = 1;
            self.tx_domain_dist_threshold = 0;
        }
        self
    }

    /// Speed-0 usage-GOOD defaults — the KEY-frame encoder-gate envelope
    /// (`aomenc --cpu-used=0` with one forced KEY frame runs usage GOOD,
    /// not ALLINTRA). Verified vs `set_good_speed_features_framesize_
    /// independent` speed-0 base (speed_features.c:1091-1163): every field
    /// equals the allintra value EXCEPT `use_chroma_trellis_rd_mult`
    /// (never set on the GOOD path — default 0, speed_features.c:2474).
    pub fn speed0_good() -> Self {
        TxTypeSearchPolicy {
            use_chroma_trellis_rd_mult: false,
            ..Self::speed0_allintra()
        }
    }
}

/// Everything `search_tx_type` reads for one interior intra txb (luma or
/// chroma — `plane` selects the tx-mask arm, the trellis rd multiplier, and
/// the tx-type-rate gate).
pub struct TxTypeSearchInputs<'a> {
    /// Residual (src - pred), full `TX_W x TX_H`, stride = TX_W.
    pub residual: &'a [i16],
    /// Source pixels of the txb (u16 universal repr), stride `src_stride`.
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    /// The intra prediction, full `TX_W x TX_H` contiguous (stride = TX_W).
    pub pred: &'a [u16],
    pub tx_size: usize,
    /// Plane (0 = luma; 1/2 = chroma). Chroma pins the tx type to
    /// [`uv_intra_tx_type`], uses the chroma trellis rd multiplier
    /// (`plane_rd_mult_chroma[0][1] = 13` under the speed-0 allintra sf
    /// `use_chroma_trellis_rd_mult = 1`), and codes no tx-type bits.
    pub plane: usize,
    /// The block's UV prediction mode (read when `plane > 0` for the UV tx
    /// type; `UV_CFL_PRED = 13` maps to DC).
    pub uv_mode: usize,
    /// LUMA intra mode + filter-intra state (tx-set + tx-type-rate selection
    /// — the chroma reduced-txset flag is also selected by the LUMA
    /// direction).
    pub mode: usize,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub bd: u8,
    /// The per-qindex quantizer rows for this plane (both FP and B are
    /// reachable: FP with trellis, B when the trellis is skipped —
    /// `USE_B_QUANT_NO_TRELLIS = 1`).
    pub rows: &'a aom_quant::PlaneQuantRows<'a>,
    /// Neighbour entropy contexts (`get_txb_ctx` inputs).
    pub bctx: &'a BlockContext<'a>,
    /// The block RD multiplier `x->rdmult`.
    pub rdmult: i32,
    pub coeff_costs: &'a CoeffCostTables<'a>,
    pub tx_type_costs: &'a TxTypeCosts,
    /// Frame-edge visible txb extent (`get_txb_visible_dimensions`,
    /// rdopt_utils.h): the distortion (`av1_pixel_diff_dist` residual SSE +
    /// `dist_block_px_domain` recon SSE) is measured over ONLY these
    /// top-left visible pixels; the off-frame overhang is excluded, matching
    /// C's `pixel_dist` / `pixel_diff_dist`. For interior txbs these equal
    /// the full `TX_W x TX_H` and the computation is byte-identical to the
    /// prior interior-only path.
    pub visible_cols: usize,
    pub visible_rows: usize,
    /// Frame QM level for THIS plane (`qmatrix_level_{y,u,v}`), `None` = QM
    /// off. With `Some`, every `xform_quant`/`xform_quant_optimize` in the
    /// search resolves the per-(tx_size, tx_type) matrices internally
    /// (`av1_setup_qmatrix`) — so QM shapes the tx-type/tx-size RD exactly as
    /// C, not just the final coefficients.
    pub qm_level: Option<usize>,
    /// The `zero_blk_rate` of `predict_dc_only_block`'s skip fast path
    /// (tx_search.c:2055-2063): `txb_skip_cost[origin_skip_ctx][1]` where
    /// `origin_skip_ctx` is `get_txb_ctx` at the **block origin** computed
    /// from the PERSISTENT (block-entry) entropy arrays — C re-derives
    /// `ctxa`/`ctxl` via `av1_get_entropy_contexts` from
    /// `pd->above_entropy_context` (never mutated during the RD walk) and
    /// passes the UN-offset pointers, so EVERY txb of the block shares the
    /// origin txb's skip ctx here, not its own walking ctx. Callers compute
    /// it once per (block, tx_size) from the pre-walk contexts; only read
    /// when `pol.predict_dc_level >= 1` (allintra speed >= 6), so 0 is fine
    /// below that.
    pub predict_skip_zero_blk_rate: i32,
}

/// One evaluated tx type's outcome (the winner's is returned).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxTypeSearchResult {
    pub best_tx_type: usize,
    pub best_eob: u16,
    pub best_txb_ctx: u8,
    /// Winner rate (coeff bits + non-skip/skip + tx_type), dist, sse, rd.
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub rd: i64,
    pub skip_txfm: bool,
    /// The winner's quantized/dequantized coefficients (the C keeps the best
    /// dqcoeff via buffer swap; callers reconstruct from these).
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    /// Coverage/introspection: which tx types were evaluated (bit per type).
    pub evaluated_mask: u16,
}

/// `search_tx_type` (tx_search.c, static) for one INTERIOR luma intra txb at
/// the speed-0 policy: iterate the allowed tx types (identity `txk_map`), per
/// type forward-transform + quantize (+ trellis when enabled) + rate + the
/// pixel-domain/high-energy-hybrid distortion, track the strict-min RD, with
/// the `adaptive_txb_search_level` and `skip_tx_search` early breaks.
///
/// Scope (labelled): interior txbs (visible == full txb — frame-edge-clipped
/// distortion via `pixel_dist_visible_only` unported), `predict_dc_level = 0`
/// (no DC-only prediction), the SATD trellis-skip short-circuit only
/// (threshold `UINT_MAX` at speed 0; the SATD body panics if reached), no
/// `recon_intra` (callers reconstruct from the returned winner coefficients),
/// flat quant (no qmatrix), plane 0.
pub fn search_tx_type_intra(
    inp: &TxTypeSearchInputs,
    pol: &TxTypeSearchPolicy,
    ref_best_rd: i64,
) -> Option<TxTypeSearchResult> {
    let tx_size = inp.tx_size;
    // Only the txb WIDTH is needed here now (residual stride + block_sse math);
    // the distortion measures `inp.visible_{cols,rows}`, not the full tx height.
    let w = TXS_W[tx_size];
    let hbd = inp.bd > 8;

    // qstep from the AC dequant lane (dequant_QTX[1] >> dequant_shift).
    let dequant_shift = if hbd { inp.bd as i32 - 5 } else { 3 };
    let qstep = (i32::from(inp.rows.dequant[1]) >> dequant_shift) as u32;

    // `predict_dc_block` (tx_search.c:2118-2119): the skip/DC-block predictor
    // runs when the stage's `predict_dc_level >= 1` (allintra speed>=6
    // DEFAULT_EVAL + MODE_EVAL — `dc_blk_pred_level = 1`) on non-64-point txs.
    // Level 1 predicts SKIP blocks only (both planes); the `dc_only_blk` arm
    // needs level > 1 (speed-7+ machinery, unmodeled — the debug_assert
    // below pins the envelope).
    let predict_dc_block =
        pol.predict_dc_level >= 1 && TXS_W[tx_size] != 64 && TXS_H[tx_size] != 64;
    // Residual SSE + MSE over the VISIBLE txb area (== full for interior txbs;
    // clipped to the frame edge for partial-SB txbs, matching C
    // `pixel_diff_dist`). With the predictor live, BOTH come from
    // `pixel_diff_stats` instead (C :2025 — its DOUBLE-arithmetic mse then
    // feeds the trellis/tx-domain gates below).
    let (mut block_sse_u, mut block_mse_q8);
    if predict_dc_block {
        debug_assert!(
            pol.predict_dc_level <= 1,
            "dc_only (level>1) is speed-7+ machinery"
        );
        let (sse, mse, per_px_mean, raw_var) =
            pixel_diff_stats(inp.residual, w, inp.visible_cols, inp.visible_rows);
        debug_assert_ne!(
            mse,
            u32::MAX,
            "predict path needs a visible txb (C :2028 assert)"
        );
        block_sse_u = sse;
        block_mse_q8 = mse;
        // predict_dc_only_block (tx_search.c:2011-2076), the level-1 skip arm:
        // a residual with variance below `1.8*qstep^2` AND a transform-domain
        // mean below the DC qstep quantizes to an all-zero (eob 0) DCT_DCT
        // block — short-circuit the whole tx-type search with the skip result.
        let var_threshold = (1.8f64 * f64::from(qstep) * f64::from(qstep)) as u64;
        let block_var = if hbd {
            let s = 2 * (i32::from(inp.bd) - 8);
            (raw_var + ((1u64 << s) >> 1)) >> s
        } else {
            raw_var
        };
        if block_var < var_threshold {
            // NOTE the dc qstep uses a FIXED >>3 (C :2024), not the
            // bd-dependent dequant_shift the AC lane above uses.
            let dc_qstep = i32::from(inp.rows.dequant[0]) >> 3;
            if per_px_mean.abs() * i64::from(DC_COEFF_SCALE[tx_size]) < (i64::from(dc_qstep) << 12)
            {
                // The skip fast path (:2039-2070): eob 0, DCT_DCT,
                // entropy ctx 0, dist = sse<<4 (sse bd-rounded FIRST),
                // rate = the block-origin zero_blk_rate (see the
                // `predict_skip_zero_blk_rate` field docs).
                let mut sse_s = sse as i64;
                if hbd {
                    sse_s = round_power_of_two_i64(sse_s, 2 * (i32::from(inp.bd) - 8));
                }
                let dist = sse_s << 4;
                let rate = inp.predict_skip_zero_blk_rate;
                let full = TXS_W[tx_size] * TXS_H[tx_size];
                return Some(TxTypeSearchResult {
                    best_tx_type: 0, // DCT_DCT (:2115)
                    best_eob: 0,
                    best_txb_ctx: 0, // txb_entropy_ctx[block] = 0 (:2069)
                    rate,
                    dist,
                    sse: dist,
                    rd: rdcost(inp.rdmult, rate, dist),
                    skip_txfm: true,
                    qcoeff: vec![0; full],
                    dqcoeff: vec![0; full],
                    evaluated_mask: 0,
                });
            }
            // predict_dc_level > 1 would set dc_only_blk here — unreachable
            // at level 1 (the assert above).
        }
    } else {
        (block_sse_u, block_mse_q8) =
            av1_pixel_diff_dist(inp.residual, w, 0, 0, inp.visible_cols, inp.visible_rows);
    }
    let mut block_sse = block_sse_u as i64;
    if hbd {
        let s = 2 * (inp.bd as i32 - 8);
        block_sse = (block_sse + ((1i64 << s) >> 1)) >> s;
        block_mse_q8 = (((block_mse_q8 as u64) + ((1u64 << s) >> 1)) >> s) as u32;
        block_sse_u = block_sse as u64;
    }
    let _ = block_sse_u;
    block_sse *= 16;

    // Allowed tx-type set (identity txk_map at speed 0); the chroma arm pins
    // the UV tx type.
    // The tx-mask params: speed-0 allintra values for the fixed fields, with the
    // per-stage `use_default_intra_tx_type` / `use_screen_content_tools` threaded
    // from the policy (both false for DEFAULT_EVAL, so speed 0..=3 is unchanged;
    // the MODE_EVAL first pass at speed>=4 sets use_default). The chroma arm
    // ignores use_default (its tx type is derived from luma), so passing it is
    // harmless there.
    let tx_mask_params = TxMaskParams {
        use_default_intra_tx_type: pol.use_default_intra_tx_type,
        use_screen_content_tools: pol.use_screen_content_tools,
        // CLI tx-type toggles (oxcf.txfm_cfg → get_tx_mask reads them
        // directly, stage-independent): defaults on/off in every policy
        // constructor, so pre-toggle behavior is byte-identical.
        enable_flip_idtx: pol.enable_flip_idtx,
        use_intra_dct_only: pol.use_intra_dct_only,
        // sf tx_type_search.prune_tx_type_using_stats (framesize+speed dependent;
        // 0 outside the >=480p speed>=2 allintra envelope, so byte-inert there).
        prune_tx_type_using_stats: pol.prune_tx_type_using_stats,
        ..TxMaskParams::speed0_allintra()
    };
    let (allowed_tx_mask, txk_allowed) = if inp.plane == 0 {
        get_tx_mask_intra(
            tx_size,
            inp.mode,
            inp.use_filter_intra,
            inp.filter_intra_mode,
            inp.lossless,
            inp.reduced_tx_set_used,
            &tx_mask_params,
        )
    } else {
        let (m, t) = get_tx_mask_uv_intra(
            tx_size,
            inp.uv_mode,
            inp.mode,
            inp.use_filter_intra,
            inp.filter_intra_mode,
            inp.lossless,
            inp.reduced_tx_set_used,
            &tx_mask_params,
        );
        (m, Some(t))
    };
    let mut allowed_tx_mask = allowed_tx_mask;

    // Est-rd tx-type prune + search-order reorder (get_tx_mask's multi-type
    // arm, tx_search.c:1911): `num_allowed > 2 && prune_tx_type_est_rd` — no
    // inter gate, so this RUNS on intra (the speed>=4 WINNER_MODE_EVAL pass;
    // the MODE_EVAL pass's default-tx-type mask is single-type and never
    // reaches it). txk_map stays the natural 0..16 order when the prune is
    // off (the speed 0..=3 loop order — byte-identical).
    let mut txk_map: [usize; TX_TYPES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    if txk_allowed.is_none()
        && pol.prune_tx_type_est_rd
        && allowed_tx_mask.count_ones() > 2
        && inp.plane == 0
    {
        // Intra ext-tx sets cap at 7 types, so the `num_allowed <= 7`
        // prune_txk_type arm is always taken (prune_txk_type_separ is
        // inter-only territory, EXT_TX_SET_ALL16).
        debug_assert!(allowed_tx_mask.count_ones() <= 7);
        let pf = PRUNE_FACTORS[pol.prune_2d_txfm_mode as usize];
        let prune = prune_txk_type_intra(
            inp,
            tx_size,
            allowed_tx_mask,
            pf,
            &mut txk_map,
            pol.use_qm_dist_metric,
        );
        allowed_tx_mask &= !prune;
        // "Need to have at least one transform type allowed" (tx_search.c:1944):
        // prune_txk_type always keeps txk_map[0], so the mask stays non-empty.
        debug_assert_ne!(allowed_tx_mask, 0);
    }

    // Trellis gating: block-MSE / qstep^2 threshold.
    let mut skip_trellis = pol.skip_trellis;
    let perform_block_coeff_opt = (block_mse_q8 as u64)
        <= (pol.coeff_opt_dist_threshold as u64) * (qstep as u64) * (qstep as u64);
    skip_trellis |= !perform_block_coeff_opt;

    // Distortion-domain policy.
    let mut use_transform_domain_distortion = pol.use_transform_domain_distortion > 0
        && block_mse_q8 >= pol.tx_domain_dist_threshold
        && TXSIZE_SQR_UP_MAP[tx_size] != 4;
    let mut calc_pixel_domain_distortion_final =
        pol.use_transform_domain_distortion == 1 && use_transform_domain_distortion;
    if calc_pixel_domain_distortion_final && (txk_allowed.is_some() || allowed_tx_mask == 0x0001) {
        calc_pixel_domain_distortion_final = false;
        use_transform_domain_distortion = false;
    }

    // av1_setup_quant: FP with trellis, B without (USE_B_QUANT_NO_TRELLIS=1).
    let kind = if skip_trellis {
        QuantKind::B
    } else {
        QuantKind::Fp
    };
    let mut qp = QuantParams::from_plane_rows(inp.rows, kind, inp.bd, inp.lossless);
    if let Some(level) = inp.qm_level {
        qp = qp.with_qm(level, inp.plane);
        if pol.use_qm_dist_metric {
            // AOM_DIST_METRIC_QM_PSNR: the trellis weights its distortion by
            // the forward matrix (av1_optimize_txb, txb_rdopt.c:346-351).
            qp = qp.with_qm_dist_metric();
        }
    }
    let trellis_rdmult = trellis_rdmult_intra(
        inp.rdmult,
        pol.sharpness,
        inp.bd,
        inp.plane,
        pol.use_chroma_trellis_rd_mult,
        pol.iq_tuning,
    );
    let opt = OptimizeInputs {
        cost: inp.coeff_costs,
        rdmult: trellis_rdmult,
        sharpness: pol.sharpness,
    };

    let mut best: Option<TxTypeSearchResult> = None;
    let mut best_rd = i64::MAX;
    let mut evaluated_mask = 0u16;

    // Iterate in txk_map order (tx_search.c:2199: `tx_type = txk_map[idx]`) —
    // the natural 0..16 order unless the est-rd prune reordered it. The order
    // is byte-load-bearing: ties keep the FIRST winner, and the adaptive-txb /
    // skip-tx-search early breaks are cumulative-order-dependent.
    for idx in 0..TX_TYPES {
        let tx_type = txk_map[idx];
        if allowed_tx_mask & (1 << tx_type) == 0 {
            continue;
        }
        evaluated_mask |= 1 << tx_type;

        // SATD-based trellis skip (tx_search.c:2217, skip_trellis_opt_based_on
        // _satd): short-circuited when skip_trellis || threshold == UINT_MAX
        // (:1986 — the speed 0..=3 DEFAULT_EVAL path, coeff_opt_satd_threshold is
        // always UINT_MAX there, so this is a no-op below speed 4). At speed>=4
        // perform_coeff_opt=5 gives a finite satd threshold; the body forward-
        // transforms + Σ|coeff| and may skip trellis for large residuals.
        let skip_trellis_this = if skip_trellis || pol.coeff_opt_satd_threshold == u32::MAX {
            skip_trellis
        } else {
            skip_trellis_opt_based_on_satd(
                inp.residual,
                tx_size,
                tx_type,
                inp.bd,
                qstep,
                pol.coeff_opt_satd_threshold,
            )
        };

        // Forward transform + quantize (+ trellis + rate).
        let (res, rate_cost): (XformQuantOptResult, i32) = if !skip_trellis_this {
            let r = xform_quant_optimize(inp.residual, tx_size, tx_type, kind, &qp, inp.bctx, &opt);
            // av1_optimize_txb rate += tx_type cost when eob > 0.
            let ttc = if r.eob > 0 {
                get_tx_type_cost(
                    inp.tx_type_costs,
                    inp.plane,
                    tx_size,
                    tx_type,
                    false,
                    inp.reduced_tx_set_used,
                    inp.lossless,
                    inp.use_filter_intra,
                    inp.filter_intra_mode,
                    inp.mode,
                )
            } else {
                0
            };
            let rate = r.rate + ttc;
            (r, rate)
        } else {
            // No-trellis arm: B quant, entropy ctx computed by av1_quant,
            // rate via av1_cost_coeffs_txb (+ tx_type inside its eob>0 body).
            let xq = xform_quant(inp.residual, tx_size, tx_type, kind, &qp, false);
            let (txb_skip_ctx, dc_sign_ctx) = aom_txb::get_txb_ctx(
                inp.bctx.plane_bsize,
                tx_size,
                inp.bctx.plane,
                inp.bctx.above,
                inp.bctx.left,
            );
            let rate = cost_coeffs_txb(
                &xq.qcoeff,
                xq.eob as usize,
                tx_size,
                tx_type,
                txb_skip_ctx as usize,
                dc_sign_ctx as usize,
                inp.coeff_costs,
            ) + if xq.eob > 0 {
                get_tx_type_cost(
                    inp.tx_type_costs,
                    inp.plane,
                    tx_size,
                    tx_type,
                    false,
                    inp.reduced_tx_set_used,
                    inp.lossless,
                    inp.use_filter_intra,
                    inp.filter_intra_mode,
                    inp.mode,
                )
            } else {
                0
            };
            let r = XformQuantOptResult {
                coeff: xq.coeff,
                qcoeff: xq.qcoeff,
                dqcoeff: xq.dqcoeff,
                eob: xq.eob,
                txb_entropy_ctx: xq.txb_entropy_ctx,
                rate,
                txb_skip_ctx: txb_skip_ctx as usize,
                dc_sign_ctx: dc_sign_ctx as usize,
            };
            (r, rate)
        };

        // Early rate-only termination.
        if rdcost(inp.rdmult, rate_cost, 0) > best_rd {
            continue;
        }

        // Distortion. The transform-domain arms carry the per-(tx_size,
        // tx_type) forward matrix C threads via `quant_param.qmatrix`
        // (av1_setup_qmatrix inside the loop, tx_search.c:2204-2249) — only
        // weighted under the QM-PSNR metric (`dist_block_tx_domain`'s
        // `qmatrix == NULL || !use_qm_dist_metric` gate, :1150/:1159).
        let dqm = crate::dist_qmatrix(&qp, tx_size, tx_type);
        let dscan = aom_txb::scan(tx_size, tx_type);
        let (dist, sse): (i64, i64) = if res.eob == 0 {
            (block_sse, block_sse)
        } else if use_transform_domain_distortion {
            crate::dist_block_tx_domain_qm(
                &res.coeff,
                &res.dqcoeff,
                tx_size,
                inp.bd,
                dqm,
                dscan,
                pol.use_qm_dist_metric,
            )
        } else {
            // Pixel-domain with the 64-pt / high-energy tx-domain hybrid.
            let high_energy_thresh = 128i64 * 128 * TX_SIZE_2D_TBL[tx_size];
            let is_high_energy = block_sse >= high_energy_thresh;
            let is_tx64 = tx_size == 4; // TX_64X64
            let mut d = i64::MAX;
            let mut s_tx = i64::MAX;
            let mut sse_diff = i64::MAX;
            if is_tx64 || is_high_energy {
                let (dt, st) = crate::dist_block_tx_domain_qm(
                    &res.coeff,
                    &res.dqcoeff,
                    tx_size,
                    inp.bd,
                    dqm,
                    dscan,
                    pol.use_qm_dist_metric,
                );
                d = dt;
                s_tx = st;
                sse_diff = block_sse - st;
            }
            if !is_tx64 || !is_high_energy || sse_diff * 2 < s_tx {
                let tx_domain_dist = d;
                d = dist_block_px_domain(
                    &res.dqcoeff,
                    tx_size,
                    tx_type,
                    inp.pred,
                    inp.src,
                    inp.src_off,
                    inp.src_stride,
                    inp.bd,
                    inp.visible_cols,
                    inp.visible_rows,
                    res.eob as usize,
                    inp.lossless,
                );
                if is_high_energy && d < tx_domain_dist {
                    d = tx_domain_dist;
                }
            } else {
                d += sse_diff;
            }
            (d, block_sse)
        };

        let rd = rdcost(inp.rdmult, rate_cost, dist);
        if rd < best_rd {
            best_rd = rd;
            best = Some(TxTypeSearchResult {
                best_tx_type: tx_type,
                best_eob: res.eob,
                best_txb_ctx: res.txb_entropy_ctx,
                rate: rate_cost,
                dist,
                sse,
                rd,
                skip_txfm: false, // set from best_eob below
                qcoeff: res.qcoeff,
                dqcoeff: res.dqcoeff,
                evaluated_mask: 0,
            });
        }

        // Early termination: current best much worse than the reference.
        if pol.adaptive_txb_search_level > 0
            && (best_rd - (best_rd >> pol.adaptive_txb_search_level)) > ref_best_rd
        {
            break;
        }
        // All-zero quantization break (speed >= 1; off at speed 0).
        if pol.skip_tx_search && best.as_ref().is_some_and(|b| b.best_eob == 0) {
            break;
        }
    }

    best.map(|mut b| {
        b.skip_txfm = b.best_eob == 0;
        b.evaluated_mask = evaluated_mask;
        // `calc_pixel_domain_distortion_final` (tx_search.c:2378-2381): when
        // transform-domain distortion ranked the candidates (speed >= 1,
        // high-MSE multi-tx-type non-64 blocks), recompute the WINNER's
        // distortion in the pixel domain and reset `sse` to `block_sse`. The
        // tx_type selection is unchanged (already decided by transform-domain
        // RD); only the returned dist/sse/rd that feed the mode/partition RDO
        // are corrected. `dist_block_px_domain_interior` is the same pure
        // pixel-domain reconstruct+SSE the C `dist_block_px_domain` computes
        // (already used as the pixel-domain component of the speed-0 hybrid).
        if calc_pixel_domain_distortion_final && b.best_eob != 0 {
            b.dist = dist_block_px_domain(
                &b.dqcoeff,
                tx_size,
                b.best_tx_type,
                inp.pred,
                inp.src,
                inp.src_off,
                inp.src_stride,
                inp.bd,
                inp.visible_cols,
                inp.visible_rows,
                b.best_eob as usize,
                inp.lossless,
            );
            b.sse = block_sse;
            b.rd = rdcost(inp.rdmult, b.rate, b.dist);
        }
        b
    })
}

/// `dist_block_px_domain` (tx_search.c): reconstruct `pred + inv_txfm(dqcoeff)`
/// over the FULL txb, then return `16 *` the SSE vs the source measured over the
/// `visible_cols x visible_rows` top-left sub-rect only (C `pixel_dist` ->
/// `pixel_dist_visible_only`). For an interior txb `visible == (TX_W, TX_H)` and
/// this is exactly the `aom_highbd_<bd>_variance` sse the interior path returned;
/// for a partial-SB edge txb it matches `aom_highbd_sse_odd_size` +
/// `ROUND_POWER_OF_TWO(sse, (bd-8)*2)` (both go through `highbd_variance64`, so
/// the per-term u32 truncation + bd normalisation are identical either way).
#[allow(clippy::too_many_arguments)]
pub fn dist_block_px_domain(
    dqcoeff: &[i32],
    tx_size: usize,
    tx_type: usize,
    pred: &[u16],
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    bd: u8,
    visible_cols: usize,
    visible_rows: usize,
    eob: usize,
    lossless: bool,
) -> i64 {
    let (w, h) = (TXS_W[tx_size], TXS_H[tx_size]);
    let mut recon = pred[..w * h].to_vec();
    aom_transform::inv_txfm2d::av1_inverse_transform_add(
        dqcoeff,
        &mut recon,
        w,
        tx_type,
        tx_size,
        i32::from(bd),
        eob,
        lossless,
    );
    let (_var, sse) = aom_dist::highbd_variance(
        &src[src_off..],
        src_stride,
        &recon,
        w,
        visible_cols,
        visible_rows,
        bd,
    );
    16 * i64::from(sse)
}

// ---------------------------------------------------------------------------
// av1_txfm_rd_in_plane + uniform_txfm_yrd (tx_search.c) — the per-tx-size
// evaluator: foreach-txb walk (predict-from-recon -> subtract ->
// search_tx_type -> recon feedback -> entropy-ctx threading) + the intra
// skip/no-skip + tx-size-rate RD assembly.
// ---------------------------------------------------------------------------

use crate::mode_costs::{TxSizeCosts, block_signals_txsize, tx_size_cost};
use aom_dist::highbd_subtract_block;
use aom_entropy::partition::intra_avail;
use aom_intra::predict_intra_high;

/// `RD_STATS` as this walk uses it (rate `i32::MAX` = invalid).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RdStats {
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub skip_txfm: bool,
}

impl RdStats {
    /// `av1_init_rd_stats` (zero, skip = 1).
    pub fn zero() -> Self {
        RdStats {
            rate: 0,
            dist: 0,
            sse: 0,
            skip_txfm: true,
        }
    }
    /// `av1_merge_rd_stats` (rate saturates at `INT_MAX`).
    pub fn merge(&mut self, o: &RdStats) {
        if self.rate == i32::MAX || o.rate == i32::MAX {
            self.rate = i32::MAX;
            return;
        }
        self.rate = (i64::from(self.rate) + i64::from(o.rate)).min(i64::from(i32::MAX)) as i32;
        self.dist += o.dist;
        self.sse += o.sse;
        self.skip_txfm &= o.skip_txfm;
    }
}

pub(crate) const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
pub(crate) const MI_SIZE_HIGH_B: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
pub(crate) const BLK_W_B: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
pub(crate) const BLK_H_B: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

/// The frame/block environment of one luma coding block's tx search — the
/// MACROBLOCK(D) state `block_rd_txfm` reads, expressed as plain data. The
/// mode fields are the CURRENT `mbmi` candidate under evaluation.
pub struct TxfmYrdEnv<'a> {
    // intra_avail frame geometry (see aom_entropy::partition::intra_avail).
    pub sb_size: usize,
    pub bsize: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub up_available: bool,
    pub left_available: bool,
    pub tile_col_end: i32,
    pub tile_row_end: i32,
    pub partition: usize,
    pub mi_cols: i32,
    pub mi_rows: i32,
    // Pixel planes: `recon[ref_off]` is the block's top-left in the
    // reconstruction plane (prediction reads + reconstruction writes);
    // `src[src_off]` in the source plane.
    pub ref_off: usize,
    pub ref_stride: usize,
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    // Prediction config.
    pub disable_edge_filter: bool,
    pub filter_type: i32,
    // Candidate mode.
    pub mode: usize,
    pub angle_delta: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub bd: u8,
    // Quantizer + RD.
    pub rows: &'a aom_quant::PlaneQuantRows<'a>,
    /// `x->qindex` (the frame base qindex on this fixed-q KEY envelope) —
    /// read by the speed>=6 intra-tx-depth NN's `log_dc_q_square` feature
    /// (`av1_dc_quant_QTX(x->qindex, 0, bd)`, tx_search.c:2867 — the RAW
    /// frame qindex, NOT the y-dc-delta-adjusted `dequant_QTX[0]`).
    pub qindex: i32,
    pub rdmult: i32,
    /// The full REAL per-(txs_ctx, eob_multi_size) luma cost-table set;
    /// [`txfm_rd_in_plane_intra`] selects the table for the CANDIDATE
    /// `tx_size` it is invoked with (this env is shared unchanged across a
    /// block's whole tx-size depth search — see `CoeffCostSet`'s docs for why
    /// a single representative table would bias the RD comparison across
    /// depths).
    pub coeff_costs: &'a CoeffCostSet,
    pub tx_type_costs: &'a TxTypeCosts,
    // Header rates: `skip_txfm_cost[skip_ctx][0/1]` (ctx = the
    // av1_get_skip_txfm_context facade, caller-supplied) and the tx-size cost
    // table + context (get_tx_size_context facade, caller-supplied).
    pub skip_costs: &'a [[i32; 2]; 3],
    pub skip_ctx: usize,
    pub tx_size_costs: &'a TxSizeCosts,
    pub tx_size_ctx: usize,
    /// `tx_mode_search_type == TX_MODE_SELECT` (speed-0 all-intra: true —
    /// USE_FULL_RD; see rdopt_utils.h select_tx_mode).
    pub tx_mode_is_select: bool,
    /// The block's above/left entropy contexts (`av1_get_entropy_contexts`
    /// copies these into the walk's working arrays).
    pub above_ctx: &'a [i8],
    pub left_ctx: &'a [i8],
    /// Frame QM levels (`qmatrix_level_{y,u,v}`), `None` = QM off. The luma
    /// walk reads `[0]`; the CfL winner re-encode inherits it.
    pub qm_levels: Option<[usize; 3]>,
}

/// One txb's winner within a walk (the `tx_type_map` / eob state the depth
/// loop snapshots for the winning size).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxbWinner {
    pub tx_type: usize,
    pub eob: u16,
    pub txb_ctx: u8,
}

/// A palette-Y candidate for the yrd walk (`av1_predict_intra_block`'s
/// `use_palette` arm): per-txb prediction is `palette[map[(r+y)*stride+c+x]]`
/// — the colour-index map fill — instead of spatial intra prediction. The
/// map covers the FULL `block_width x block_height` pixel extent of the
/// block (extended from the visible crop), stride = `map_stride`
/// (= block width in pixels).
#[derive(Clone, Copy, Debug)]
pub struct PaletteYrd<'a> {
    /// `palette_colors[0..size]` (the Y section).
    pub colors: &'a [u16; 8],
    /// `palette_size[0]`.
    pub size: usize,
    /// The colour-index map (`block_width * block_height`).
    pub map: &'a [u8],
    /// Map row stride in pixels (= block width).
    pub map_stride: usize,
}

/// `av1_txfm_rd_in_plane` (tx_search.c) for a luma intra block at `tx_size`
/// (uniform): the `av1_foreach_transformed_block_in_plane` raster walk; per
/// txb predict-from-recon (`intra_avail` + `predict_intra_high` INTO the
/// recon plane, matching `av1_predict_intra_block_facade`'s in-place dst
/// write) -> subtract -> [`search_tx_type_intra`] -> reconstruct the winner
/// into `recon` (the `recon_intra` feedback the next txb predicts from) ->
/// entropy-context stamp. Interior blocks (`max_blocks_*` unclipped).
///
/// Returns `None` when the search exits early (`exit_early` — for intra ANY
/// early exit invalidates, tx_search.c:3786) or `current_rd > ref_best_rd` on
/// entry; `Some(stats)` otherwise. `recon` and the working contexts are
/// modified as the C does (the caller snapshots/restores between tx sizes).
#[allow(clippy::too_many_arguments)]
pub fn txfm_rd_in_plane_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    tx_size: usize,
    ref_best_rd: i64,
    current_rd_in: i64,
    pol: &TxTypeSearchPolicy,
    // `enable_nn_prune_intra_tx_depths` (tx_search.c:3022): Some only from
    // the depth loop's largest-tx iteration when the speed>=6 NN sf is on.
    mut nn_prune: Option<NnDepthPruneCtx<'_>>,
    palette: Option<&PaletteYrd>,
) -> Option<(RdStats, Vec<TxbWinner>)> {
    if current_rd_in > ref_best_rd {
        return None;
    }
    let bsize = env.bsize;
    let (bw, bh) = (BLK_W_B[bsize], BLK_H_B[bsize]);
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let (txw_unit, txh_unit) = (txw >> 2, txh >> 2);
    let max_blocks_wide = MI_SIZE_WIDE_B[bsize];
    let max_blocks_high = MI_SIZE_HIGH_B[bsize];
    // C clips the tx-block LOOP to the visible frame area (`max_block_wide`/
    // `max_block_high`, av1_common_int.h:1568): a partial edge block's off-frame
    // tx sub-blocks are never searched/predicted (the predictor asserts
    // `n_top_px >= 0`). Luma plane, no subsampling. Context arrays stay full.
    let blocks_wide_visible = max_blocks_wide.min((env.mi_cols - env.mi_col).max(0) as usize);
    let blocks_high_visible = max_blocks_high.min((env.mi_rows - env.mi_row).max(0) as usize);

    // av1_get_entropy_contexts: working copies of the neighbour contexts.
    let mut t_above: Vec<i8> = env.above_ctx[..max_blocks_wide].to_vec();
    let mut t_left: Vec<i8> = env.left_ctx[..max_blocks_high].to_vec();
    // predict_dc_only_block's zero_blk_rate ctx (tx_search.c:2055-2063): the
    // BLOCK-ORIGIN skip ctx from the PERSISTENT (pre-walk) entropy arrays —
    // shared by every txb of this block (see the
    // `TxTypeSearchInputs::predict_skip_zero_blk_rate` docs). Computed here
    // BEFORE the walk stamps t_above/t_left.
    let predict_skip_zero_blk_rate = if pol.predict_dc_level >= 1 {
        let (origin_skip_ctx, _) = aom_txb::get_txb_ctx(bsize, tx_size, 0, &t_above, &t_left);
        env.coeff_costs.tables(tx_size).txb_skip[origin_skip_ctx as usize * 2 + 1]
    } else {
        0
    };

    let mut stats = RdStats::zero();
    let mut winners: Vec<TxbWinner> = Vec::new();
    let mut current_rd = current_rd_in;
    let mut exit_early = false;

    // `av1_foreach_transformed_block_in_plane` mu-64 chunk walk (encodemb.c:
    // 560-582): a coding block > 64x64 is searched/reconstructed in 64x64-unit
    // chunk order — above 64x64 the per-txb prediction pixels AND the
    // has_top_right/has_bottom_left availability are chunk-order-dependent, so
    // a raster walk would compute a different tx RD than C and flip the winner.
    // Luma mu = 16; one chunk (byte-identical raster) for bsize <= 64x64.
    let mu_w = MI_SIZE_WIDE_B[12].min(blocks_wide_visible); // BLOCK_64X64
    let mu_h = MI_SIZE_HIGH_B[12].min(blocks_high_visible);
    let mut chunk_r = 0usize;
    while chunk_r < blocks_high_visible {
        let unit_h = (chunk_r + mu_h).min(blocks_high_visible);
        let mut chunk_c = 0usize;
        while chunk_c < blocks_wide_visible {
            let unit_w = (chunk_c + mu_w).min(blocks_wide_visible);
            let mut blk_row = chunk_r;
            while blk_row < unit_h {
                let mut blk_col = chunk_c;
                while blk_col < unit_w {
            if exit_early {
                // C: the next block_rd_txfm call marks incomplete_exit; for
                // intra exit_early alone already invalidates.
                return None;
            }

            // av1_predict_intra_block_facade: predict INTO the recon plane.
            let txb_off = env.ref_off + (blk_row * env.ref_stride + blk_col) * 4;
            let mut pred = vec![0u16; txw * txh];
            if let Some(pal) = palette {
                // av1_predict_intra_block's use_palette arm: the colour-index
                // map fill at this txb's pixel offset (x = blk_col*4,
                // y = blk_row*4) — no spatial prediction, no neighbour reads.
                let (x, y) = (blk_col * 4, blk_row * 4);
                for r in 0..txh {
                    for c in 0..txw {
                        pred[r * txw + c] =
                            pal.colors[pal.map[(r + y) * pal.map_stride + c + x] as usize];
                    }
                }
            } else {
                let (n_top, n_topright, n_left, n_bottomleft) = intra_avail(
                    env.sb_size,
                    bsize,
                    env.mi_row,
                    env.mi_col,
                    env.up_available,
                    env.left_available,
                    env.tile_col_end,
                    env.tile_row_end,
                    env.partition,
                    tx_size,
                    0,
                    0,
                    blk_row as i32,
                    blk_col as i32,
                    bw as i32,
                    bh as i32,
                    env.mi_cols,
                    env.mi_rows,
                    env.mode,
                    env.angle_delta * 3, // ANGLE_STEP
                    env.use_filter_intra,
                );
                predict_intra_high(
                    recon,
                    txb_off,
                    env.ref_stride,
                    &mut pred,
                    txw,
                    env.mode,
                    env.angle_delta * 3,
                    env.use_filter_intra,
                    env.filter_intra_mode,
                    env.disable_edge_filter,
                    env.filter_type,
                    tx_size,
                    n_top as usize,
                    n_topright,
                    n_left as usize,
                    n_bottomleft,
                    env.bd as i32,
                );
            }
            // The C facade writes the prediction into dst (the recon plane).
            for r in 0..txh {
                recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }

            // av1_subtract_txb.
            let src_txb_off = env.src_off + (blk_row * env.src_stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            highbd_subtract_block(
                txh,
                txw,
                &mut residual,
                txw,
                &env.src[src_txb_off..],
                env.src_stride,
                &pred,
                txw,
            );

            // ml_predict_intra_tx_depth_prune (block_rd_txfm, tx_search.c:
            // 3089-3097): with the NN enabled for this (largest-depth) walk,
            // evaluate it on the freshly-subtracted residual. The model's own
            // gates (:2836-2842): !lossless, the tx covers the whole block
            // (txsize_to_bsize[tx_size] == bsize — with the TX_8X8-only model
            // support this reduces to an 8x8 block's single txb), bd 8.
            // TX_PRUNE_LARGEST aborts this depth's evaluation
            // (av1_invalid_rd_stats + exit_early); TX_PRUNE_SPLIT is consumed
            // by the depth loop before the next (smaller) size.
            if let Some(nn) = nn_prune.as_mut() {
                if !env.lossless && tx_size == 1 && bsize == 3 && env.bd == 8 {
                    *nn.outcome = ml_predict_intra_tx_depth_prune(
                        &residual,
                        txw,
                        nn.source_variance,
                        env.qindex,
                        env.bd,
                    );
                    if *nn.outcome == TxPruneType::Largest {
                        return None;
                    }
                }
            }

            // search_tx_type over the neighbour ctx at this txb position.
            let bctx = crate::BlockContext {
                plane_bsize: bsize,
                plane: 0,
                above: &t_above[blk_col..],
                left: &t_left[blk_row..],
            };
            // The real per-txs_ctx/eob_multi_size table for THIS CANDIDATE
            // tx_size (env.coeff_costs is shared across every depth the
            // caller's tx-size search tries; the lookup must happen here,
            // per candidate, not once at env construction).
            let coeff_tables = env.coeff_costs.tables(tx_size);
            // Frame-edge visible extent of THIS txb (get_txb_visible_dimensions):
            // the block's mb_to_*_edge (set_mi_row_col, av1_common_int.h) clipped
            // per-txb. Luma plane (ss = 0); interior => full (txw, txh).
            let bw_mi = MI_SIZE_WIDE_B[bsize] as i32;
            let bh_mi = MI_SIZE_HIGH_B[bsize] as i32;
            let mb_to_right_edge = (env.mi_cols - bw_mi - env.mi_col) * 32; // MI_SIZE*8
            let mb_to_bottom_edge = (env.mi_rows - bh_mi - env.mi_row) * 32;
            let (vis_cols, vis_rows) = get_txb_visible_dimensions(
                bw_mi as usize * 4,
                bh_mi as usize * 4,
                txw,
                txh,
                blk_row,
                blk_col,
                mb_to_right_edge,
                mb_to_bottom_edge,
                0,
                0,
            );
            let inp = TxTypeSearchInputs {
                residual: &residual,
                src: env.src,
                src_off: src_txb_off,
                src_stride: env.src_stride,
                pred: &pred,
                tx_size,
                plane: 0,
                uv_mode: 0,
                mode: env.mode,
                use_filter_intra: env.use_filter_intra,
                filter_intra_mode: env.filter_intra_mode,
                lossless: env.lossless,
                reduced_tx_set_used: env.reduced_tx_set_used,
                bd: env.bd,
                rows: env.rows,
                bctx: &bctx,
                rdmult: env.rdmult,
                coeff_costs: &coeff_tables,
                tx_type_costs: env.tx_type_costs,
                visible_cols: vis_cols,
                visible_rows: vis_rows,
                qm_level: env.qm_levels.map(|l| l[0]),
                predict_skip_zero_blk_rate,
            };
            // `block_rd_txfm` (tx_search.c:3104) computes
            // `args->best_rd - args->current_rd` as a RAW int64_t subtraction
            // with NO `ref_best_rd == INT64_MAX` special case (unlike e.g.
            // `av1_txfm_search`'s `rd_thresh` derivation, tx_search.c:3816-3817,
            // which explicitly guards it). At the true (0,0) frame corner
            // `ref_best_rd` is genuinely `INT64_MAX` (no reference RD yet) and
            // `current_rd` can go deeply negative -- a real, already-verified
            // C behaviour: `block_error`'s lowbd path wraps its per-coefficient
            // product at 32 bits (matching C's `int`, see aom-dist::block_error),
            // and a no-neighbour corner prediction (DC fallback, no top/left)
            // can produce a spatially-uniform residual whose DC coefficient is
            // far larger than the locally-predicted interior case ever sees,
            // pushing that wrap into `dist`/`current_rd`. C's compiled binary
            // (no UBSan/ftrapv in the production build) performs this same
            // subtraction as a plain two's-complement wraparound, and the
            // wrapped value is NOT dead: `adaptive_txb_search_level` is 1 at
            // speed-0 allintra (`TxTypeSearchPolicy::speed0_allintra`), so
            // `search_tx_type_intra`'s early-termination compare
            // (`(best_rd - (best_rd >> level)) > ref_best_rd`) reads it. So we
            // replicate C's exact wraparound with `wrapping_sub` -- NOT a
            // `saturating_sub`, which would compute a different (non-C) value.
            // Empirically confirmed reachable (not just theoretical): the pre-fix
            // `-` panicked "attempt to subtract with overflow" at this exact site
            // for allintra/4:2:0/qindex=98 at the true (0,0) corner -- see
            // `pack_tile_roundtrips_true_corner`'s `allintra=true` case.
            let win = search_tx_type_intra(&inp, pol, ref_best_rd.wrapping_sub(current_rd))
                .expect("search_tx_type always yields a winner");

            // recon_intra: reconstruct the winner on top of the prediction so
            // the next txb predicts from decoded pixels. C's `recon_intra`
            // (tx_search.c:930-932) guards this on `best_eob &&
            // (blk_row + tx_size_high_unit < mi_size_high[plane_bsize] ||
            //  blk_col + tx_size_wide_unit < mi_size_wide[plane_bsize])` --
            // i.e. it reconstructs a txb ONLY when it is NOT the last
            // (bottom-right-most) txb of the block. Nothing chains from the
            // last txb, so C deliberately leaves the raw PREDICTION there
            // rather than the reconstruction. This is not cosmetic: the
            // ALLINTRA `intra_rd_variance_factor` reads `xd->plane[0].dst.buf`
            // right after this tx-size search, and for a DC-predicted (flat)
            // block that leftover determines whether the block's recon
            // variance reads as ~0 (flat prediction -> factor up to 3.0) or as
            // the high-variance reconstruction (factor 1.0). Reconstructing the
            // last txb here made the factor 1.0 where C's is 3.0, letting the
            // search accept small screen-content blocks (e.g. an 8x4 DC leaf of
            // a HORZ split) that C's inflated rd rejects -- the (2,12,BLOCK_8X8)
            // real=NONE / ours=HORZ AB-probe divergence. Interior-only walk:
            // `max_blocks_{high,wide}` are the unclipped `mi_size_{high,wide}
            // [plane_bsize]` this function already uses for the loop bounds.
            let not_last_txb =
                blk_row + txh_unit < max_blocks_high || blk_col + txw_unit < max_blocks_wide;
            if win.best_eob > 0 && not_last_txb {
                let mut tight = pred.clone();
                aom_transform::inv_txfm2d::av1_inverse_transform_add(
                    &win.dqcoeff,
                    &mut tight,
                    txw,
                    win.best_tx_type,
                    tx_size,
                    i32::from(env.bd),
                    win.best_eob as usize,
                    env.lossless,
                );
                for r in 0..txh {
                    recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }

            winners.push(TxbWinner {
                tx_type: win.best_tx_type,
                eob: win.best_eob,
                txb_ctx: win.best_txb_ctx,
            });

            // av1_set_txb_context (interior: stamp the winner's entropy ctx).
            for a in t_above[blk_col..blk_col + txw_unit].iter_mut() {
                *a = win.best_txb_ctx as i8;
            }
            for l in t_left[blk_row..blk_row + txh_unit].iter_mut() {
                *l = win.best_txb_ctx as i8;
            }

            // Intra rd accumulation: signalled non-skip.
            let this = RdStats {
                rate: win.rate,
                dist: win.dist,
                sse: win.sse,
                skip_txfm: false,
            };
            stats.merge(&this);
            let rd = rdcost(env.rdmult, win.rate, win.dist);
            current_rd += rd;
            if current_rd > ref_best_rd {
                exit_early = true;
            }

            blk_col += txw_unit;
                }
                blk_row += txh_unit;
            }
            chunk_c += mu_w;
        }
        chunk_r += mu_h;
    }

    if exit_early {
        // Set on the LAST txb: intra still invalidates (invalid_rd =
        // args.exit_early for !is_inter).
        return None;
    }
    Some((stats, winners))
}

/// `uniform_txfm_yrd` (tx_search.c, intra arm): evaluate one uniform tx size
/// for the block — tx-size rate (under TX_MODE_SELECT on signaling blocks),
/// skip/no-skip header handling (intra: always signalled non-skip;
/// `skip_txfm_rd = INT64_MAX`), the [`txfm_rd_in_plane_intra`] walk, and the
/// final `RDCOST(rate + no_skip_rate + tx_size_rate, dist)` with
/// `rate += tx_size_rate`. Returns `(rd, Some(stats))` or
/// `(i64::MAX, None)` when invalid.
pub fn uniform_txfm_yrd_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    tx_size: usize,
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
    nn_prune: Option<NnDepthPruneCtx<'_>>,
    // Palette-Y candidate (av1_rd_pick_palette_intra_sby's tx search —
    // prediction becomes the colour-map fill).
    palette: Option<&PaletteYrd>,
) -> (i64, Option<(RdStats, Vec<TxbWinner>)>) {
    let tx_select = env.tx_mode_is_select && block_signals_txsize(env.bsize);
    let tx_size_rate = if tx_select {
        tx_size_cost(env.tx_size_costs, true, env.bsize, tx_size, env.tx_size_ctx)
    } else {
        0
    };
    let no_skip_txfm_rate = env.skip_costs[env.skip_ctx][0];
    // Intra: skip_txfm_rd = INT64_MAX; current_rd = no_this_rd.
    let no_this_rd = rdcost(env.rdmult, no_skip_txfm_rate + tx_size_rate, 0);

    let Some((mut stats, winners)) = txfm_rd_in_plane_intra(
        env,
        recon,
        tx_size,
        ref_best_rd,
        no_this_rd,
        pol,
        nn_prune,
        palette,
    ) else {
        return (i64::MAX, None);
    };
    if stats.rate == i32::MAX {
        return (i64::MAX, None);
    }

    // Intra blocks are always signalled as non-skip.
    let rd = rdcost(
        env.rdmult,
        stats.rate + no_skip_txfm_rate + tx_size_rate,
        stats.dist,
    );
    stats.rate += tx_size_rate;
    (rd, Some((stats, winners)))
}

// ---------------------------------------------------------------------------
// choose_tx_size_type_from_rd + av1_pick_uniform_tx_size_type_yrd
// (tx_search.c) — the uniform tx-size depth search, luma intra.
// ---------------------------------------------------------------------------

/// `max_txsize_rect_lookup[BLOCK_SIZES_ALL]` (common_data.h).
pub const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] = [
    0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18,
];

/// `TX_PRUNE_TYPE` (block.h:504): the intra-tx-depth NN's verdict, carried
/// across the depth loop like C's `txfm_params->nn_prune_depths_for_intra_tx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxPruneType {
    /// `TX_PRUNE_NONE`.
    None,
    /// `TX_PRUNE_SPLIT`: skip the split (smaller) tx depths.
    Split,
    /// `TX_PRUNE_LARGEST`: abort the largest-depth evaluation.
    Largest,
}

/// The `enable_nn_prune_intra_tx_depths` bundle threaded into the txb walk
/// (C sets the flag on `txfm_params` and the walk reads `x->source_variance`
/// off the shared MACROBLOCK; the port threads both explicitly).
pub struct NnDepthPruneCtx<'a> {
    /// Where the verdict lands (`txfm_params->nn_prune_depths_for_intra_tx`).
    pub outcome: &'a mut TxPruneType,
    /// `x->source_variance` — the `log1pf` feature input.
    pub source_variance: u32,
}

/// `get_dev` (tx_search.c:1697): `sqrtf(max(0, x2_sum/num - mean^2))` with
/// C's exact double-division-then-float arithmetic.
fn get_dev(mean: f32, x2_sum: f64, num: i32) -> f32 {
    let e_x2 = (x2_sum / f64::from(num)) as f32;
    let diff = e_x2 - mean * mean;
    if diff > 0.0 { diff.sqrt() } else { 0.0 }
}

/// `get_mean_dev_features` (tx_search.c:1709): block + sub-block residual
/// mean/deviation features. Layout for the 8x8 intra-tx-split model
/// (bw = bh = 8 -> subw = subh = 4, four sub-blocks): `[0]` block mean,
/// `[1]` block dev, `[2..10]` per-sub (mean, dev) in raster order, `[10]`
/// deviation of sub-means, `[11]` mean of sub-devs. Returns the next
/// feature index (12).
fn get_mean_dev_features(
    data: &[i16],
    stride: usize,
    bw: usize,
    bh: usize,
    features: &mut [f32],
) -> usize {
    let subh = if bh >= bw { bh >> 1 } else { bh };
    let subw = if bw >= bh { bw >> 1 } else { bw };
    let num = (bw * bh) as i32;
    let sub_num = (subw * subh) as i32;
    let mut feature_idx = 2usize;
    let mut total_x_sum = 0i32;
    let mut total_x2_sum = 0i64;
    let mut num_sub_blks = 0i32;
    let mut mean2_sum = 0.0f64;
    let mut dev_sum = 0.0f32;
    let mut row = 0usize;
    while row < bh {
        let mut col = 0usize;
        while col < bw {
            // aom_get_blk_sse_sum over the (subw x subh) sub-block.
            let mut x_sum = 0i32;
            let mut x2_sum = 0i64;
            for r in 0..subh {
                let base = (row + r) * stride + col;
                for &d in &data[base..base + subw] {
                    x_sum += i32::from(d);
                    x2_sum += i64::from(d) * i64::from(d);
                }
            }
            total_x_sum += x_sum;
            total_x2_sum += x2_sum;
            let mean = x_sum as f32 / sub_num as f32;
            let dev = get_dev(mean, x2_sum as f64, sub_num);
            features[feature_idx] = mean;
            features[feature_idx + 1] = dev;
            feature_idx += 2;
            mean2_sum += f64::from(mean * mean);
            dev_sum += dev;
            num_sub_blks += 1;
            col += subw;
        }
        row += subh;
    }
    let lvl0_mean = total_x_sum as f32 / num as f32;
    features[0] = lvl0_mean;
    features[1] = get_dev(lvl0_mean, total_x2_sum as f64, num);
    // Deviation of means; mean of deviations.
    features[feature_idx] = get_dev(lvl0_mean, mean2_sum, num_sub_blks);
    features[feature_idx + 1] = dev_sum / num_sub_blks as f32;
    feature_idx + 2
}

/// `ml_predict_intra_tx_depth_prune` (tx_search.c:2823): the 8x8-block NN
/// verdict over the largest-tx residual. The model gates (":2836-2842")
/// — lossless / tx not covering the block / bd != 8 / tx != TX_8X8 — are
/// checked by the CALLER (the walk), which passes the txb residual (`diff`
/// at the block origin, stride = txb width = 8). Features: 12 mean/dev
/// values + `log1pf(source_variance)` + `log1pf(dc_q^2/256)` with dc_q =
/// `av1_dc_quant_QTX(qindex, 0, bd) >> (bd - 8)`; normalized by the
/// transcribed mean/std tables; one ReLU hidden layer (16) + linear output
/// + `av1_nn_output_prec_reduce` (reduce_prec = 1 at the call site :2879);
/// verdict per `av1_intra_tx_prune_nn_thresh_8x8`.
pub fn ml_predict_intra_tx_depth_prune(
    diff: &[i16],
    diff_stride: usize,
    source_variance: u32,
    qindex: i32,
    bd: u8,
) -> TxPruneType {
    use crate::intra_tx_nn_weights as w;
    let mut features = [0f32; w::NUM_FEATURES];
    let mut fi = get_mean_dev_features(diff, diff_stride, 8, 8, &mut features);
    features[fi] = (source_variance as f32).ln_1p();
    fi += 1;
    let dc_q = i32::from(aom_quant::av1_dc_quant_qtx(qindex, 0, bd)) >> (bd - 8);
    // log1pf((float)(dc_q * dc_q) / 256.0f).
    let log_dc_q_square = ((dc_q * dc_q) as f32 / 256.0f32).ln_1p();
    features[fi] = log_dc_q_square;
    fi += 1;
    debug_assert_eq!(fi, w::NUM_FEATURES);
    for i in 0..w::NUM_FEATURES {
        features[i] = (features[i] - w::MEAN[i]) / w::STD[i];
    }

    // av1_nn_predict_c (ml.c): one ReLU hidden layer, linear 1-wide output,
    // then av1_nn_output_prec_reduce (the call site passes reduce_prec = 1).
    let mut hidden = [0f32; w::NUM_HIDDEN];
    for (node, h) in hidden.iter_mut().enumerate() {
        let mut val = w::B0[node];
        for i in 0..w::NUM_FEATURES {
            val += w::W0[node * w::NUM_FEATURES + i] * features[i];
        }
        *h = val.max(0.0); // ReLU
    }
    let mut score = w::B1[0];
    for (i, &hv) in hidden.iter().enumerate() {
        score += w::W1[i] * hv;
    }
    // av1_nn_output_prec_reduce (ml.c:19): C's `+ 0.5` is a DOUBLE literal.
    const PREC: f32 = (1 << 9) as f32;
    const INV_PREC: f32 = 1.0 / PREC;
    score = ((f64::from(score * PREC) + 0.5) as i32) as f32 * INV_PREC;

    if score <= w::PRUNE_THRESH[0] {
        TxPruneType::Split
    } else if score > w::PRUNE_THRESH[1] {
        TxPruneType::Largest
    } else {
        TxPruneType::None
    }
}

/// `TX_SIZE_SEARCH_METHOD` (enc_enums.h:262): the tx-size search strategy for a
/// uniform intra block. `USE_FULL_RD` runs the depth sweep; `USE_LARGESTALL`
/// evaluates only the single largest uniform tx (the winner-mode MODE_EVAL first
/// pass, `tx_size_search_methods[..][MODE_EVAL]`). `USE_FAST_RD` (the RT path) is
/// not reached on the all-intra winner-mode grid.
pub const USE_FULL_RD: usize = 0;
pub const USE_FAST_RD: usize = 1;
pub const USE_LARGESTALL: usize = 2;

/// `tx_size_from_tx_mode(bsize, TX_MODE_LARGEST)` + the `choose_largest_tx_size`
/// enable-adjustment (tx_search.c:2715). Returns the single tx size the
/// `USE_LARGESTALL` arm evaluates: `max_txsize_rect_lookup[bsize]`, then demoted
/// when `enable_tx64`/`enable_rect_tx` are off. On the all-intra envelope BOTH
/// are ON (CLI defaults), so this is exactly `max_txsize_rect_lookup[bsize]`; the
/// demotion tables are carried verbatim so the arm stays correct if a caller
/// disables tx64/rect.
fn choose_largest_tx_size_intra(bsize: usize, enable_tx64: bool, enable_rect_tx: bool) -> usize {
    // tx_size_from_tx_mode(bs, TX_MODE_LARGEST) == max_txsize_rect_lookup[bs]
    // (BLOCK_4X4 -> TX_4X4 is already encoded in the table).
    let tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
    match (enable_tx64, enable_rect_tx) {
        (true, true) => tx,
        // tx_size_max_32[TX_SIZES_ALL] — cap the 64-wide/high sizes at 32.
        (false, true) => {
            const TX_SIZE_MAX_32: [usize; 19] = [
                0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 3, 3, 13, 14, 15, 16, 9, 10,
            ];
            TX_SIZE_MAX_32[tx]
        }
        // tx_size_max_square[TX_SIZES_ALL] — collapse rect to the inscribed square.
        (true, false) => {
            const TX_SIZE_MAX_SQUARE: [usize; 19] =
                [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
            TX_SIZE_MAX_SQUARE[tx]
        }
        // tx_size_max_32_square[TX_SIZES_ALL] — square AND 32-capped.
        (false, false) => {
            const TX_SIZE_MAX_32_SQUARE: [usize; 19] =
                [0, 1, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
            TX_SIZE_MAX_32_SQUARE[tx]
        }
    }
}
/// `sub_tx_size_map[TX_SIZES_ALL]` (common_data.h): rect sizes halve the LONG
/// side.
pub const SUB_TX_SIZE_MAP: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
/// `MAX_TX_DEPTH` (blockd.h).
pub const MAX_TX_DEPTH: i32 = 2;

/// `get_search_init_depth` (tx_search.c) for an intra block: the per-shape
/// tx-size depth-search FLOOR. `tx_size_search_lgr_block = 0` (default) => the
/// (rect, sqr) init-depth arm; USE_FULL_RD (never LARGESTALL here — that arm
/// returns MAX_VARTX_DEPTH upstream). Rectangular (w != h) blocks use
/// `init_depth_rect`, square (w == h) blocks `init_depth_sqr`; both are speed
/// features carried on [`TxTypeSearchPolicy`] (rect: 0 at speed 0, 1 at
/// speed >= 1; sqr: 1 for all speeds — speed_features.c:367/409).
pub fn get_search_init_depth_intra(
    mi_width: usize,
    mi_height: usize,
    init_depth_rect: i32,
    init_depth_sqr: i32,
) -> i32 {
    if mi_height != mi_width {
        init_depth_rect
    } else {
        init_depth_sqr
    }
}

/// The outcome of the depth search: the chosen uniform tx size, its per-txb
/// winners (the `tx_type_map` snapshot), and the block RD stats.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxSizeChoice {
    pub best_tx_size: usize,
    pub best_rd: i64,
    pub stats: RdStats,
    pub winners: Vec<TxbWinner>,
}

/// `choose_tx_size_type_from_rd` (tx_search.c, static) — the uniform-tx-size
/// depth sweep for a luma intra block under `TX_MODE_SELECT` at the speed-0
/// policy: start at `max_txsize_rect_lookup[bsize]` skipped down by
/// `init_depth` steps, evaluate each depth with [`uniform_txfm_yrd_intra`],
/// keep the strict-min, stop at TX_4X4 or on the low-contrast
/// (`source_variance < 256`) regression prune.
///
/// Per-speed resolution (each named): `init_depth` per
/// [`get_search_init_depth_intra`] from the pol's rect/sqr init-depths (rect 0
/// at speed 0, 1 at speed >= 1; sqr 1 always); `enable_tx64`/`enable_rect_tx`
/// CLI-default ON (the disable arms are `continue`s, kept via params);
/// `use_rd_based_breakout_for_intra_tx_search = false` (rd_thresh is always
/// `ref_best_rd`); `prune_intra_tx_depths_using_nn = false` (no NN pruning);
/// `x->rd_model = FULL_TXFM_RD`.
///
/// `source_variance` is `x->source_variance` (the source block's per-pixel
/// variance, computed upstream by the encoder — caller-supplied).
/// Returns `None` when no depth produced a valid RD (rate stays `INT_MAX`).
#[allow(clippy::too_many_arguments)]
pub fn choose_tx_size_type_from_rd_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
    source_variance: u32,
    enable_tx64: bool,
    enable_rect_tx: bool,
    palette: Option<&PaletteYrd>,
) -> Option<TxSizeChoice> {
    let bsize = env.bsize;
    let max_rect_tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
    // tx_select is TX_MODE_SELECT here (the LARGESTALL/ONLY_4X4 arms are the
    // caller's — see av1_pick_uniform_tx_size_type_yrd).
    let mut start_tx = max_rect_tx_size;
    let init_depth = get_search_init_depth_intra(
        MI_SIZE_WIDE_B[bsize],
        MI_SIZE_HIGH_B[bsize],
        pol.intra_tx_size_init_depth_rect,
        pol.intra_tx_size_init_depth_sqr,
    );
    if init_depth == MAX_TX_DEPTH && !enable_tx64 && TXSIZE_SQR_UP_MAP[start_tx] == 4 {
        start_tx = SUB_TX_SIZE_MAP[start_tx];
    }

    let mut best: Option<TxSizeChoice> = None;
    let mut best_rd = i64::MAX;
    let mut rd = [i64::MAX; MAX_TX_DEPTH as usize + 1];
    // `txfm_params->nn_prune_depths_for_intra_tx` (reset TX_PRUNE_NONE per
    // block, tx_search.c:3059): the largest-depth walk's NN verdict.
    let mut nn_outcome = TxPruneType::None;
    let mut tx_size = start_tx;
    let mut depth = init_depth;
    while depth <= MAX_TX_DEPTH {
        if (!enable_tx64 && TXSIZE_SQR_UP_MAP[tx_size] == 4)
            || (!enable_rect_tx && TXS_W[tx_size] != TXS_H[tx_size])
        {
            depth += 1;
            tx_size = SUB_TX_SIZE_MAP[tx_size];
            continue;
        }
        // `if (nn_prune_depths == TX_PRUNE_SPLIT) break;` (tx_search.c:3016)
        // — the largest depth's verdict skips the smaller sizes.
        if nn_outcome == TxPruneType::Split {
            break;
        }
        // `enable_nn_prune_intra_tx_depths = sf && tx_size == start_tx`
        // (:3022): the NN only ever evaluates on the largest-depth walk.
        let nn_ctx =
            (pol.prune_intra_tx_depths_using_nn && tx_size == start_tx).then(|| NnDepthPruneCtx {
                outcome: &mut nn_outcome,
                source_variance,
            });
        // rd_thresh (tx_search.c:3030): with use_rd_based_breakout_for_intra_
        // tx_search ON, tighten the per-depth early-exit to the running
        // across-depth best; else the caller's ref_best_rd.
        let rd_thresh = if pol.use_rd_based_breakout_for_intra_tx_search {
            ref_best_rd.min(best_rd)
        } else {
            ref_best_rd
        };
        let (this_rd, res) =
            uniform_txfm_yrd_intra(env, recon, tx_size, rd_thresh, pol, nn_ctx, palette);
        rd[depth as usize] = this_rd;
        if this_rd < best_rd {
            let (stats, winners) = res.expect("valid rd implies stats");
            best_rd = this_rd;
            best = Some(TxSizeChoice {
                best_tx_size: tx_size,
                best_rd,
                stats,
                winners,
            });
        }
        if tx_size == 0 {
            break; // TX_4X4
        }
        // Low-contrast regression prune across the two searched depths.
        if depth > init_depth && depth != MAX_TX_DEPTH && source_variance < 256 {
            let prev = rd[depth as usize - 1];
            if prev != i64::MAX && rd[depth as usize] > prev {
                break;
            }
        }
        depth += 1;
        tx_size = SUB_TX_SIZE_MAP[tx_size];
    }
    best
}

/// `av1_pick_uniform_tx_size_type_yrd` (tx_search.c:3628) — the luma intra
/// slice: residue hashing and skip-prediction are inter-only. Dispatch mirrors
/// the C (:3683): lossless picks the smallest (4x4) transform
/// (`choose_smallest_tx_size` = one [`uniform_txfm_yrd_intra`] at TX_4X4);
/// `USE_LARGESTALL` evaluates the single largest uniform tx
/// ([`choose_largest_tx_size_intra`] — the winner-mode MODE_EVAL first pass);
/// otherwise (`USE_FULL_RD`) the depth sweep runs. Returns the chosen size +
/// stats, or `None` (rate `INT_MAX`).
#[allow(clippy::too_many_arguments)]
pub fn pick_uniform_tx_size_type_yrd_intra(
    env: &TxfmYrdEnv,
    recon: &mut [u16],
    ref_best_rd: i64,
    pol: &TxTypeSearchPolicy,
    source_variance: u32,
    enable_tx64: bool,
    enable_rect_tx: bool,
    tx_size_search_method: usize,
    palette: Option<&PaletteYrd>,
) -> Option<TxSizeChoice> {
    // select_tx_mode (rdopt_utils.h) couples the two at frame level:
    // coded_lossless => ONLY_4X4, i.e. never TX_MODE_SELECT.
    debug_assert!(
        !(env.lossless && env.tx_mode_is_select),
        "lossless implies ONLY_4X4 (tx_mode_is_select must be false)",
    );
    if env.lossless {
        // choose_smallest_tx_size: evaluate TX_4X4 only.
        let (rd, res) = uniform_txfm_yrd_intra(env, recon, 0, ref_best_rd, pol, None, palette);
        return res.map(|(stats, winners)| TxSizeChoice {
            best_tx_size: 0,
            best_rd: rd,
            stats,
            winners,
        });
    }
    if tx_size_search_method == USE_LARGESTALL {
        // choose_largest_tx_size: one uniform_txfm_yrd at the single largest tx
        // size (no depth sweep, no size-RD comparison, no low-contrast prune).
        let tx_size = choose_largest_tx_size_intra(env.bsize, enable_tx64, enable_rect_tx);
        let (rd, res) =
            uniform_txfm_yrd_intra(env, recon, tx_size, ref_best_rd, pol, None, palette);
        return res.map(|(stats, winners)| TxSizeChoice {
            best_tx_size: tx_size,
            best_rd: rd,
            stats,
            winners,
        });
    }
    choose_tx_size_type_from_rd_intra(
        env,
        recon,
        ref_best_rd,
        pol,
        source_variance,
        enable_tx64,
        enable_rect_tx,
        palette,
    )
}

// ---------------------------------------------------------------------------
// intra_model_rd (intra_mode_search_utils.h) — the Hadamard-SATD model cost
// that feeds prune_intra_y_mode in the av1_rd_pick_intra_sby_mode loop.
// ---------------------------------------------------------------------------

/// `wht_fwd_txfm` / `highbd_wht_fwd_txfm` (hybrid_fwd_txfm.c) + `aom_satd`:
/// the Walsh-Hadamard transform of one model txb's residual, then the sum of
/// absolute coefficients. Buffer-depth dispatch mirrors `av1_quick_txfm` with
/// `use_hadamard = 1`: 8-bit buffers use the lowbd `aom_hadamard_NxN` kernels
/// for every size; highbd buffers (`bd > 8`) use lowbd `aom_hadamard_4x4` at
/// TX_4X4 (its output fits 15 bits) and `aom_highbd_hadamard_NxN` above.
fn wht_satd(residual: &[i16], stride: usize, tx_size: usize, bd: u8) -> i32 {
    use aom_dist::hadamard::{
        hadamard_4x4, hadamard_8x8, hadamard_16x16, hadamard_32x32, highbd_hadamard_8x8,
        highbd_hadamard_16x16, highbd_hadamard_32x32, satd,
    };
    match (bd > 8, tx_size) {
        (_, 0) => satd(&hadamard_4x4(residual, stride)),
        (false, 1) => satd(&hadamard_8x8(residual, stride)),
        (false, 2) => satd(&hadamard_16x16(residual, stride)),
        (false, 3) => satd(&hadamard_32x32(residual, stride)),
        (true, 1) => satd(&highbd_hadamard_8x8(residual, stride)),
        (true, 2) => satd(&highbd_hadamard_16x16(residual, stride)),
        (true, 3) => satd(&highbd_hadamard_32x32(residual, stride)),
        _ => unreachable!("model tx size is TX_4X4..TX_32X32 (square)"),
    }
}

/// `intra_model_rd` (intra_mode_search_utils.h) for luma (`plane == 0`) with
/// `use_hadamard == 1` — the mode-loop call site (intra_mode_search.c:1602).
/// Per model-txb raster walk: `av1_predict_intra_block_facade` (the prediction
/// is written INTO the recon plane, so later txbs predict from earlier txbs'
/// *predictions* — no reconstruction happens here) -> `av1_subtract_block` ->
/// `av1_quick_txfm(use_hadamard=1)` -> `aom_satd`, accumulated into i64.
///
/// `tx_size` is the caller's `AOMMIN(TX_32X32, max_txsize_lookup[bsize])`
/// (always square). The recon-plane prediction writes are load-bearing
/// caller-visible state: the C facade writes into `pd->dst` in place, and the
/// bytes it leaves behind are what the *next* candidate's walks read wherever
/// edge availability reaches not-yet-overwritten pixels. Interior blocks
/// (`max_block_wide/high` unclipped), matching [`txfm_rd_in_plane_intra`].
///
/// The `env` quantizer/cost fields are unused here — only geometry, the
/// candidate mode fields, and `bd` are read.
pub fn intra_model_rd_y(env: &TxfmYrdEnv, recon: &mut [u16], tx_size: usize) -> i64 {
    assert!(tx_size <= 3, "model tx size is square TX_4X4..TX_32X32");
    let bsize = env.bsize;
    let (bw, bh) = (BLK_W_B[bsize], BLK_H_B[bsize]);
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    // stepr/stepc = tx_size_high/wide_unit; max_blocks_* in 4x4 units.
    let (txw_unit, txh_unit) = (txw >> 2, txh >> 2);
    let max_blocks_wide = MI_SIZE_WIDE_B[bsize];
    let max_blocks_high = MI_SIZE_HIGH_B[bsize];
    // C clips the tx-block LOOP to the visible frame area (`max_block_wide`/
    // `max_block_high`, av1_common_int.h:1568): a partial edge block's off-frame
    // tx sub-blocks are never searched/predicted (the predictor asserts
    // `n_top_px >= 0`). Luma plane, no subsampling.
    let blocks_wide_visible = max_blocks_wide.min((env.mi_cols - env.mi_col).max(0) as usize);
    let blocks_high_visible = max_blocks_high.min((env.mi_rows - env.mi_row).max(0) as usize);

    let mut satd_cost: i64 = 0;
    // mu-64 chunk walk (see `txfm_rd_in_plane_intra`): the model predicts INTO
    // the recon plane and later txbs predict from earlier predictions, so > 64
    // blocks must walk in chunk order. Luma mu = 16; one chunk for bsize <= 64.
    let mu_w = MI_SIZE_WIDE_B[12].min(blocks_wide_visible); // BLOCK_64X64
    let mu_h = MI_SIZE_HIGH_B[12].min(blocks_high_visible);
    let mut chunk_r = 0usize;
    while chunk_r < blocks_high_visible {
        let unit_h = (chunk_r + mu_h).min(blocks_high_visible);
        let mut chunk_c = 0usize;
        while chunk_c < blocks_wide_visible {
            let unit_w = (chunk_c + mu_w).min(blocks_wide_visible);
            let mut blk_row = chunk_r;
            while blk_row < unit_h {
                let mut blk_col = chunk_c;
                while blk_col < unit_w {
            // av1_predict_intra_block_facade: predict INTO the recon plane.
            let (n_top, n_topright, n_left, n_bottomleft) = intra_avail(
                env.sb_size,
                bsize,
                env.mi_row,
                env.mi_col,
                env.up_available,
                env.left_available,
                env.tile_col_end,
                env.tile_row_end,
                env.partition,
                tx_size,
                0,
                0,
                blk_row as i32,
                blk_col as i32,
                bw as i32,
                bh as i32,
                env.mi_cols,
                env.mi_rows,
                env.mode,
                env.angle_delta * 3, // ANGLE_STEP
                env.use_filter_intra,
            );
            let txb_off = env.ref_off + (blk_row * env.ref_stride + blk_col) * 4;
            let mut pred = vec![0u16; txw * txh];
            predict_intra_high(
                recon,
                txb_off,
                env.ref_stride,
                &mut pred,
                txw,
                env.mode,
                env.angle_delta * 3,
                env.use_filter_intra,
                env.filter_intra_mode,
                env.disable_edge_filter,
                env.filter_type,
                tx_size,
                n_top as usize,
                n_topright,
                n_left as usize,
                n_bottomleft,
                env.bd as i32,
            );
            for r in 0..txh {
                recon[txb_off + r * env.ref_stride..txb_off + r * env.ref_stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }

            // av1_subtract_block into a tight txw-stride buffer (the C stores
            // at block_size_wide[plane_bsize] stride and reads it back with
            // the same stride — values per (r, c) identical).
            let src_txb_off = env.src_off + (blk_row * env.src_stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            highbd_subtract_block(
                txh,
                txw,
                &mut residual,
                txw,
                &env.src[src_txb_off..],
                env.src_stride,
                &pred,
                txw,
            );

            satd_cost += i64::from(wht_satd(&residual, txw, tx_size, env.bd));
            blk_col += txw_unit;
                }
                blk_row += txh_unit;
            }
            chunk_c += mu_w;
        }
        chunk_r += mu_h;
    }
    satd_cost
}

#[cfg(test)]
mod satd_skip_tests {
    use super::*;
    use aom_sys_ref as c;

    /// `av1_get_tx_scale` (idct.c:24) + `av1_get_max_eob` (blockd.h:1601) match
    /// the C definitions for every one of the 19 `TX_SIZES_ALL`.
    #[test]
    fn tx_scale_and_max_eob_match_c_defs() {
        for ts in 0..19usize {
            let pels = TX_SIZE_2D_TBL[ts];
            assert_eq!(
                av1_get_tx_scale(ts),
                i32::from(pels > 256) + i32::from(pels > 1024),
                "av1_get_tx_scale ts={ts}"
            );
            let want_eob = match ts {
                4 | 11 | 12 => 1024,
                17 | 18 => 512,
                _ => TX_SIZE_2D_TBL[ts] as usize,
            };
            assert_eq!(av1_get_max_eob(ts), want_eob, "av1_get_max_eob ts={ts}");
        }
        // sqrt_tx_pixels_2d spot-checks (tx_search.c:78).
        assert_eq!(SQRT_TX_PIXELS_2D[0], 4); // TX_4X4
        assert_eq!(SQRT_TX_PIXELS_2D[4], 32); // TX_64X64
        assert_eq!(SQRT_TX_PIXELS_2D[9], 23); // TX_16X32 (sqrt(512)≈23)
    }

    /// The full `skip_trellis_opt_based_on_satd` decision matches a REAL-C
    /// reference across tx sizes, residual magnitudes, and thresholds straddling
    /// the boundary. The satd anchor is `c::ref_satd` (= the exported
    /// `aom_satd_c`); the test also asserts the port's `aom_dist::hadamard::satd`
    /// equals it on the same coeffs, so the decision is validated end-to-end
    /// against real C (satd leaf) + the transcribed tx_search.c:1993-2000 scale
    /// / compare arithmetic.
    #[test]
    fn satd_skip_decision_matches_c() {
        let bd = 8u8;
        // Deterministic xorshift residual generator.
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        // Representative tx sizes across all three scales + rect + 64-point.
        for &ts in &[0usize, 1, 2, 3, 4, 7, 9, 11, 12, 17, 18] {
            let full = TXS_W[ts] * TXS_H[ts];
            for mag in [1i16, 7, 40, 300] {
                let mut residual = vec![0i16; full];
                for r in residual.iter_mut() {
                    *r = (next() % (2 * mag as u32 + 1)) as i16 - mag;
                }
                // Real-C reference satd over the forward transform (DCT_DCT).
                let mut coeff = vec![0i32; full];
                aom_transform::txfm2d::av1_fwd_txfm2d(&residual, &mut coeff, TXS_W[ts], 0, ts);
                let n = av1_get_max_eob(ts);
                assert_eq!(
                    aom_dist::hadamard::satd(&coeff[..n]),
                    c::ref_satd(&coeff[..n]),
                    "port satd != aom_satd_c (ts={ts} mag={mag})"
                );
                let satd_c = i64::from(c::ref_satd(&coeff[..n]));
                let shift = 1i32 - av1_get_tx_scale(ts);
                let scaled = if shift < 0 {
                    satd_c << (-shift)
                } else {
                    satd_c >> shift
                };
                let scaled = (scaled >> (i64::from(bd) - 8)) as u64;
                let sqrt_px = SQRT_TX_PIXELS_2D[ts] as u64;
                for qstep in [4u32, 20, 100] {
                    let denom = (qstep as u64 * sqrt_px).max(1);
                    let base = scaled / denom;
                    for dt in [-1i64, 0, 1, 2] {
                        let thr = (base as i64 + dt).max(0) as u32;
                        let want = scaled > u64::from(thr) * u64::from(qstep) * sqrt_px;
                        let got = skip_trellis_opt_based_on_satd(&residual, ts, 0, bd, qstep, thr);
                        assert_eq!(got, want, "ts={ts} mag={mag} qstep={qstep} thr={thr}");
                    }
                }
            }
        }
    }

    /// Below speed 4 the satd threshold is `UINT_MAX`, so the caller's early
    /// return path holds `skip_trellis` unchanged — the body never runs. Assert
    /// the boundary contract the call site relies on (tx_search.c:1986): with a
    /// finite threshold the body CAN return true (skip) for a large residual,
    /// and a tiny residual under a large threshold never skips.
    #[test]
    fn satd_skip_boundary_contract() {
        let full = TXS_W[2] * TXS_H[2]; // TX_16X16
        // Large residual, threshold 0 => satd>0 => skip.
        let big = vec![200i16; full];
        assert!(skip_trellis_opt_based_on_satd(&big, 2, 0, 8, 20, 0));
        // Small residual, huge threshold => never skip.
        let mut small = vec![0i16; full];
        small[0] = 1;
        assert!(!skip_trellis_opt_based_on_satd(
            &small,
            2,
            0,
            8,
            20,
            u32::MAX - 1
        ));
    }
}

#[cfg(test)]
mod est_rd_prune_tests {
    use super::*;

    /// KB-8 chunk 2d-iii: [`sort_rd`] reproduces C's insertion sort
    /// (tx_search.c:1076) — ascending by rd, ties keep the earlier candidate
    /// FIRST (the `rds[j] > rds[i]` strict compare skips equal slots).
    #[test]
    fn sort_rd_matches_c_semantics() {
        let mut rds = [50i64, 10, 30, 10, 20];
        let mut txk = [0usize, 1, 2, 3, 4];
        sort_rd(&mut rds, &mut txk, 5);
        assert_eq!(rds, [10, 10, 20, 30, 50]);
        // Ties (types 1 and 3, both rd 10): insertion order preserved.
        assert_eq!(txk, [1, 3, 4, 2, 0]);
        // Already sorted stays put.
        let mut r2 = [1i64, 2, 3];
        let mut t2 = [7usize, 8, 9];
        sort_rd(&mut r2, &mut t2, 3);
        assert_eq!((r2, t2), ([1, 2, 3], [7usize, 8, 9]));
        // len <= 1: no-op.
        let mut r1 = [5i64];
        let mut t1 = [2usize];
        sort_rd(&mut r1, &mut t1, 1);
        assert_eq!((r1[0], t1[0]), (5, 2));
    }

    /// The `PRUNE_FACTORS` table matches tx_search.c:1071, and the prune-factor
    /// survival rule (`1000*(rd-rd0)/rd0 < pf`, break at first exceeder) is
    /// what [`prune_txk_type_intra`]'s tail implements — pinned here on the
    /// arithmetic directly (the full function is exercised end-to-end by the
    /// speed-4 gate once the two-pass flips).
    #[test]
    fn prune_factor_table_and_rule() {
        assert_eq!(PRUNE_FACTORS, [200, 200, 120, 80, 40]);
        // WINNER pass at speed 4: prune_2d = TX_TYPE_PRUNE_0 -> pf = 200:
        // survivors are candidates within +20.0% of the best est-rd, up to the
        // first exceeder.
        let rds = [1000i64, 1150, 1199, 1201, 1190];
        let pf = PRUNE_FACTORS[0];
        let mut kept = vec![0usize];
        for idx in 1..rds.len() {
            let factor = 1000 * (rds[idx] - rds[0]) / rds[0];
            if factor < pf {
                kept.push(idx);
            } else {
                break; // 1201 exceeds; 1190 after it stays pruned
            }
        }
        assert_eq!(kept, [0, 1, 2]); // 1150 (+15.0%), 1199 (+19.9%) survive
    }
}

#[cfg(test)]
mod largest_tx_tests {
    use super::*;

    /// KB-8 chunk 2b: `choose_largest_tx_size_intra` (the `USE_LARGESTALL` arm)
    /// reproduces C's `choose_largest_tx_size` (tx_search.c:2715) — the single
    /// largest uniform tx size the winner-mode MODE_EVAL first pass evaluates.
    #[test]
    fn largest_tx_size_matches_c() {
        // enable_tx64 && enable_rect_tx (the all-intra CLI-default envelope):
        // exactly max_txsize_rect_lookup[bsize] for every block size.
        for (bsize, &rect) in MAX_TXSIZE_RECT_LOOKUP.iter().enumerate() {
            assert_eq!(
                choose_largest_tx_size_intra(bsize, true, true),
                rect,
                "bsize {bsize} (tx64+rect on) must be max_txsize_rect_lookup"
            );
        }
        // BLOCK_64X64 (idx 12, rect = TX_64X64 = 4): the four enable combos hit
        // the four distinct demotion tables (verified vs C tx_size_max_* tables).
        assert_eq!(choose_largest_tx_size_intra(12, true, true), 4); // TX_64X64
        assert_eq!(choose_largest_tx_size_intra(12, false, true), 3); // -> TX_32X32
        assert_eq!(choose_largest_tx_size_intra(12, true, false), 4); // square already
        assert_eq!(choose_largest_tx_size_intra(12, false, false), 3); // TX_32X32
        // BLOCK_16X64 (idx 20, rect = TX_16X64 = 17): rect demotions differ.
        assert_eq!(choose_largest_tx_size_intra(20, true, true), 17); // TX_16X64
        assert_eq!(choose_largest_tx_size_intra(20, false, true), 9); // -> TX_16X32
        assert_eq!(choose_largest_tx_size_intra(20, true, false), 2); // -> TX_16X16
        assert_eq!(choose_largest_tx_size_intra(20, false, false), 2); // TX_16X16
        // BLOCK_32X32 (idx 9): already square+<=32, invariant across combos.
        for combo in [(true, true), (false, true), (true, false), (false, false)] {
            assert_eq!(choose_largest_tx_size_intra(9, combo.0, combo.1), 3);
        }
    }

    /// KB-8 chunk 2c: `get_default_tx_type_y` (blockd.h:1175) boundary contract.
    /// The full override is exercised against real C by tx_mask_diff.rs's
    /// `tx_mask_intra_matches_c` (use_default x screen sweep); this pins the
    /// helper directly.
    #[test]
    fn default_tx_type_boundary() {
        // Small square sizes (TX_4X4/8X8/16X16 = 0/1/2) return the mode's default.
        for (mode, want) in INTRA_MODE_TO_TX_TYPE.iter().enumerate() {
            assert_eq!(get_default_tx_type_y(mode, 2, false, false), *want); // TX_16X16
        }
        // tx_size >= TX_32X32 (3) -> DCT_DCT for every mode.
        assert_eq!(get_default_tx_type_y(4, 3, false, false), 0); // TX_32X32, D135->ADST_ADST demoted
        assert_eq!(get_default_tx_type_y(4, 4, false, false), 0); // TX_64X64
        assert_eq!(get_default_tx_type_y(4, 5, false, false), 0); // TX_4X8 (rect, enum>=3)
        // lossless or screen content -> DCT_DCT even at a small size.
        assert_eq!(get_default_tx_type_y(4, 0, true, false), 0); // lossless
        assert_eq!(get_default_tx_type_y(4, 0, false, true), 0); // screen content
        // Non-degenerate: V_PRED (mode 1) at TX_8X8 is ADST_DCT (1), NOT DCT.
        assert_eq!(get_default_tx_type_y(1, 1, false, false), 1);
    }
}
