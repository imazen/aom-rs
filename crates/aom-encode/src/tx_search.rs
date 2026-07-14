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
    0x080F, 0x040F, 0x080F, 0x020F, 0x080F, 0x040F, 0x080F, 0x080F, 0x040F, 0x080F, 0x040F,
    0x080F, 0x0C0E,
];

/// `av1_derived_intra_tx_used_flag[INTRA_MODES]` (blockd.h): the
/// residual-statistics-derived set (sf `use_reduced_intra_txset == 2`).
pub const AV1_DERIVED_INTRA_TX_USED_FLAG: [u16; 13] = [
    0x0209, 0x0403, 0x0805, 0x020F, 0x0009, 0x0009, 0x0009, 0x0805, 0x0403, 0x0205, 0x0403,
    0x0805, 0x0209,
];

/// `fimode_to_intradir[FILTER_INTRA_MODES]` (blockd.h): the intra direction a
/// filter-intra mode maps to for tx-set/tx-type decisions.
pub const FIMODE_TO_INTRADIR: [usize; 5] = [0, 1, 2, 6, 0];

/// `DCT_ADST_TX_MASK` (txfm_common.h): DCT/ADST-only (kills FLIPADST + IDTX
/// combinations when `enable_flip_idtx` is off).
pub const DCT_ADST_TX_MASK: u16 = 0x000F;

/// `txsize_sqr_up_map[TX_SIZES_ALL]` (common_data.h): TX_SIZE -> square
/// TX_SIZE class rounding UP (0..4 = 4x4..64x64).
pub const TXSIZE_SQR_UP_MAP: [usize; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];

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
    /// `oxcf.txfm_cfg.enable_flip_idtx` (CLI default on).
    pub enable_flip_idtx: bool,
    /// `oxcf.txfm_cfg.use_intra_dct_only` (CLI default off).
    pub use_intra_dct_only: bool,
}

impl TxMaskParams {
    /// Speed-0 all-intra defaults.
    pub fn speed0_allintra() -> Self {
        TxMaskParams {
            use_reduced_intra_txset: 1,
            use_derived_intra_tx_type_set: false,
            enable_flip_idtx: true,
            use_intra_dct_only: false,
        }
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
    let tx_set_type = ext_tx_set_type(tx_size, false, reduced_tx_set_used);

    let intra_dir = if use_filter_intra { FIMODE_TO_INTRADIR[filter_intra_mode] } else { mode };
    let mut ext_tx_used_flag =
        if p.use_reduced_intra_txset != 0 && tx_set_type == EXT_TX_SET_DTT4_IDTX_1DDCT {
            AV1_REDUCED_INTRA_TX_USED_FLAG[intra_dir]
        } else {
            AV1_EXT_TX_USED_FLAG[tx_set_type]
        };
    if p.use_reduced_intra_txset == 2 {
        ext_tx_used_flag &= AV1_DERIVED_INTRA_TX_USED_FLAG[intra_dir];
    }

    if lossless || TXSIZE_SQR_UP_MAP[tx_size] > 3 || ext_tx_used_flag == 0x0001 || p.use_intra_dct_only
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
        allowed_tx_mask = ext_tx_used_flag;
        // Stats prune / est-rd prune / prune_tx_2D: all structurally off for
        // the speed-0 intra path (see module docs).
    }

    if allowed_tx_mask == 0 {
        txk_allowed = 0; // DCT_DCT (plane 0)
        allowed_tx_mask = 1 << txk_allowed;
    }

    let single = if txk_allowed < TX_TYPES { Some(txk_allowed) } else { None };
    debug_assert!(single.is_none_or(|t| allowed_tx_mask == 1 << t));
    (allowed_tx_mask, single)
}

/// The visible-dimension slice of `get_txb_dimensions` (rdopt_utils.h): a
/// txb's pixels clipped to the frame boundary. `mb_to_right_edge` /
/// `mb_to_bottom_edge` are the MACROBLOCKD edge fields (1/8-pel units,
/// negative when the block overhangs), `subsampling` the plane's.
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
