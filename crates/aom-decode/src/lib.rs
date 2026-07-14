//! KEY-frame tile reconstruction driver — the first aom-rs layer that turns
//! entropy-coded tile bytes into decoded pixels.
//!
//! This crate composes the already-bit-exact building blocks into libaom's
//! decode interleave (`av1/decoder/decodeframe.c`):
//!
//! - partition walk: `decode_partition` — [`aom_entropy::partition::read_partition`]
//!   per node with the threaded above/left partition context, dispatching leaf
//!   blocks in the exact `DEC_BLOCK` order (all 10 partition types);
//! - per leaf (`parse_decode_block`): mode-info decode
//!   ([`aom_entropy::partition::read_mb_modes_kf`]) followed by, per plane-0
//!   transform block in raster order (`decode_token_recon_block`, intra path):
//!   coefficient decode ([`aom_txb::read_coeffs_txb_full`] with
//!   [`aom_txb::get_txb_ctx`] neighbour contexts) **then** intra prediction
//!   ([`aom_entropy::partition::intra_avail`] +
//!   [`aom_intra::predict_intra_high`] into the reconstruction plane) **then**
//!   dequant + inverse transform + add ([`aom_encode::reconstruct_txb`]) — the
//!   read → predict → reconstruct per-txb interleave `decode_token_recon_block`
//!   uses (prediction of a block reads reconstructed pixels of previously
//!   decoded blocks, so the interleave is load-bearing);
//! - tile loop: `decode_tile_kf` — the SB row/col walk with the C's context
//!   lifetimes (above contexts zeroed once per tile, left contexts zeroed per
//!   SB row, `av1_reset_entropy_context` on skip blocks,
//!   `av1_set_entropy_contexts` frame-edge clipping).
//!
//! # Scope (honest limits of this cut)
//!
//! - **KEY frame, intra only.** No inter path, no motion compensation.
//! - **All three planes reconstructed.** `monochrome = true` decodes luma only
//!   (a complete real configuration); otherwise the U/V planes are fully
//!   reconstructed at 4:4:4, 4:2:2, or 4:2:0 ([`KfTileConfig::subsampling_x`]/
//!   [`KfTileConfig::subsampling_y`]): per-block `is_chroma_reference` (sub-8x8
//!   blocks share one chroma block, coded on the group's bottom-right member,
//!   covering the merged area from the parity-adjusted plane origin), the
//!   whole-block chroma transform (`av1_get_max_uv_txsize`), chroma
//!   coefficients on the per-plane entropy-context arrays, UV intra prediction
//!   (`get_uv_mode`, chroma availability/`scale_chroma_bsize` geometry, the
//!   chroma-neighbour edge-filter type), and chroma-from-luma: reconstructed
//!   luma is stored per txb ([`aom_intra::cfl::cfl_store_tx`],
//!   `store_cfl_required` — non-reference members always store), and
//!   `UV_CFL_PRED` blocks add the alpha-scaled zero-mean AC on the DC
//!   prediction ([`aom_intra::cfl::cfl_predict_block`]). Chroma transform
//!   types are not coded (mode-implied, ext-tx-set demoted).
//! - **Frame tx mode ([`KfTileConfig::tx_mode`])**: `TX_MODE_LARGEST` (per-block
//!   `tx_size = max_txsize_rect_lookup[bsize]`, no tx-size bits) **and**
//!   `TX_MODE_SELECT` — the per-block tx-size depth symbol on
//!   `tx_size_cdf[bsize_to_tx_size_cat][get_tx_size_context]` (`read_tx_size` →
//!   `read_selected_tx_size`, `decodeframe.c`; intra blocks code it even when
//!   skipped), with the `above_txfm_context`/`left_txfm_context` byte arrays
//!   (init 64 per tile / per SB row, stamped full-footprint by `set_txfm_ctxs`
//!   after every block, `parse_decode_block` order). Under SELECT a block's tx
//!   grid is real: the within-block multi-txb interleave (later txbs predict
//!   from earlier txbs' reconstruction *inside* the block) is exercised
//!   non-degenerately. `ONLY_4X4` is out of scope: in C it only arises with
//!   `coded_lossless` (off in this cut), where `read_tx_size`'s lossless
//!   branch — not modelled here — is what produces TX_4X4 everywhere.
//! - **Full FRAME_CONTEXT context selection** for every symbol this path codes:
//!   the driver keeps a per-mi mode-info grid ([`MiNbrKf`]: y mode +
//!   skip_txfm, stamped over each block's frame-cropped footprint like the C
//!   mi grid) and hands the `xd->above_mbmi` / `xd->left_mbmi` neighbours to
//!   [`aom_entropy::partition::read_mb_modes_kf_fc`], which picks each
//!   symbol's CDF instance from the [`KfFrameContext`] arrays exactly as
//!   `read_intra_frame_mode_info` does (kf_y by neighbour intra-mode
//!   contexts, skip by neighbour skip flags, angle deltas by coded mode on
//!   ONE shared array, uv by `[cfl_allowed][y_mode]`, filter-intra by bsize
//!   with the real mode-dependent gate). The same grid feeds `get_filt_type`
//!   (smooth-neighbour edge-filter selection). In the coefficient loop the
//!   luma tx-type CDF is selected per txb as
//!   `intra_ext_tx_cdf[eset][square_tx_size][intra_dir]` (`av1_read_tx_type`),
//!   with `intra_dir = fimode_to_intradir[..]` for filter-intra blocks.
//! - **Delta-q ([`KfTileConfig::delta_q_present`])**: the per-superblock
//!   delta-qindex is decoded at each SB's upper-left coded block
//!   (`read_delta_qindex` inside the mode-info read, with the normative
//!   `clamp(base + reduced * delta_q_res, 1, MAXQ)` carry update of
//!   `read_delta_q_params`, decodemv.c), and every coded block's dequant is
//!   then recomputed from the running `current_base_qindex` exactly as
//!   `parse_decode_block` does (decodeframe.c): per plane,
//!   `[av1_dc_quant_QTX(cur_q, dc_delta, bd), av1_ac_quant_QTX(cur_q,
//!   ac_delta, bd)]` with the frame's per-plane dc/ac deltas
//!   (`y_dc_delta_q`; `u_dc/u_ac`; `v_dc/v_ac` — Y's AC never takes a delta)
//!   via [`plane_dequants`]. Without delta-q the same formula runs once at
//!   the frame level (`setup_segmentation_dequant`, segment 0). The tx-type
//!   signalling gate stays frame-level (`av1_read_tx_type` reads
//!   `xd->qindex[segment_id]`, not the delta-modified carry). Delta-LF
//!   ([`KfTileConfig::delta_lf_present`], single or multi) is decoded and its
//!   clamped carries are threaded through every block's mode info; the
//!   TILE decoder itself does not filter (like C), but `frame.rs`'s
//!   deblock stage reads `delta_lf_from_base`/`delta_lf[]` from the block
//!   records when it filters the frame.
//! - **Segmentation ([`KfTileConfig::seg`])**: per-block segment ids are
//!   decoded exactly as `read_intra_segment_id` (pre- or post-skip by
//!   `segid_preskip`; a post-skip skipped block takes the spatial
//!   prediction), with the spatial-pred context over a current-frame
//!   segment-id map (`av1_get_spatial_seg_pred`, stamped per block like
//!   `set_segment_id`); `SEG_LVL_ALT_Q` shifts each coded block's dequant
//!   through `av1_get_qindex` (composing with the delta-q carry exactly as
//!   `parse_decode_block` recomputes `seg_dequant_QTX`), and the tx-type
//!   signalling gate reads the per-segment frame-level qindex. `SEG_LVL_SKIP`
//!   forces the skip flag. Lossless SEGMENTS (`xd->lossless[i]`, forced
//!   TX_4X4 + WHT) are out of scope and asserted away.
//! - **Superblock size ([`KfTileConfig::sb_size_128`])**: BOTH 64x64 and
//!   128x128 (`seq_params->use_128x128_superblock`). The flag sets the
//!   per-tile `mib_size` (16 or 32 mi/side) and the partition-tree root
//!   ([`BLOCK_64X64`] / [`BLOCK_128X128`]); the tile SB row/col walk steps
//!   by it, the CDEF per-64x64-unit strength index goes 4-way within a 128
//!   SB (`read_cdef`'s `cdef_transmitted[4]`), and the loop-restoration
//!   corners-in-sb reader sees the 32-mi SB extent. (RU size and CDEF's own
//!   64x64 filter-block granularity are independent of this flag.)
//! - **PALETTE** (`KfTileConfig::allow_screen_content_tools`): per-block
//!   `av1_allow_palette`-gated flags/size/colours ([`read_mb_modes_kf_fc`]) +
//!   the colour-index map tokens (a separate step after mode-info, mirroring
//!   `av1_visit_palette` — [`get_block_dimensions`] +
//!   [`decode_color_map_tokens`]) + reconstruction (a palette tx block's
//!   pixels come from the map indexing the palette, bypassing ordinary intra
//!   prediction only — the residual add is unaffected). A palette-neighbour
//!   grid ([`PaletteNbrKf`], mirrors [`MiNbrKf`]) feeds the colour cache;
//!   only allocated on screen-content frames.
//! - **Off / fixed in this cut**: intra block copy,
//!   quantization matrices (flat dequant), CDF update always on
//!   (`disable_cdf_update` unsupported — the mode-symbol readers adapt
//!   unconditionally), and no in-tile loop filters (this driver returns the
//!   PRE-FILTER reconstruction, like C's tile decode; `frame.rs` applies
//!   deblocking frame-wide afterwards — CDEF/restoration stay unapplied;
//!   CDEF *strengths* are entropy-decoded, and delta-LF levels are carried
//!   as documented above).
//! - Frame dimensions are whole mode-info (4px) units; non-multiple-of-SB sizes
//!   are supported (partition edge gathers + `max_block_wide/high` txb clipping
//!   + `av1_set_entropy_contexts` edge zeroing).
//!
//! # Validation
//!
//! The write side of every symbol here is byte-identical to C libaom, so the
//! full-tile encode→decode roundtrip in `tests/tile_roundtrip.rs` (a mirror
//! mini-encoder driving the same walk with the write-side counterparts and its
//! own prediction→residual→quantize→reconstruct feedback loop) pins this driver
//! to the C decoder: byte-identical reconstruction planes, lockstep CDF arenas,
//! and per-leaf mode-info equality.

#![forbid(unsafe_code)]

pub mod frame;

use aom_encode::reconstruct_txb;
use aom_entropy::cdf::read_symbol;
use aom_entropy::dec::OdEcDec;
use aom_entropy::dv_ref::{DvGrid, DvNbr, DvTileBounds, assign_and_validate_dv, find_dv_ref_mvs};
use aom_entropy::partition::{
    KfBlockState, KfFrameContext, MbModeInfoKf, MiNbrKf, PaletteNbrKf, TXFM_CTX_INIT, TxMode,
    allow_palette as av1_allow_palette, bsize_to_max_depth, bsize_to_tx_size_cat,
    decode_color_map_tokens, depth_to_tx_size, get_block_dimensions, get_partition_subsize,
    get_plane_block_size, get_tx_size_context, get_uv_mode, intra_avail, is_cfl_allowed,
    partition_cdf_length, partition_plane_context, read_mb_modes_kf_fc, read_partition,
    read_selected_tx_size, set_txfm_ctxs, spatial_seg_pred, tx_size_from_tx_mode,
    txfm_partition_context, txfm_partition_update, update_ext_partition_context,
};
use aom_intra::cfl::{CflCtx, cfl_predict_block, cfl_store_tx};
use aom_intra::predict_intra_high;
use aom_quant::{
    MAX_SEGMENTS, SEG_LVL_MAX, SEG_LVL_SKIP, Segmentation, av1_ac_quant_qtx, av1_dc_quant_qtx,
    av1_get_qindex,
};
use aom_txb::{
    ext_tx_derive, get_txb_ctx, read_coeffs_txb_full, txb_entropy_context, txb_high, txb_wide,
};

// ---- spec constants (av1/common/common_data.h) --------------------------------

/// `mi_size_wide[BLOCK_SIZES_ALL]`: block width in 4x4 mode-info units.
pub const MI_SIZE_WIDE: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
/// `mi_size_high[BLOCK_SIZES_ALL]`: block height in 4x4 mode-info units.
pub const MI_SIZE_HIGH: [i32; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
/// `block_size_wide[BLOCK_SIZES_ALL]`: block width in pixels.
pub const BLOCK_SIZE_WIDE: [i32; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
/// `block_size_high[BLOCK_SIZES_ALL]`: block height in pixels.
pub const BLOCK_SIZE_HIGH: [i32; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
/// `max_txsize_rect_lookup[BLOCK_SIZES_ALL]`: the largest (rectangular) transform
/// for each block size — the per-block `tx_size` under `TX_MODE_LARGEST`
/// (`tx_size_from_tx_mode`, no tx-size bits coded).
pub const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] = [
    0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18,
];
/// `tx_size_wide[TX_SIZES_ALL]` / `tx_size_high[TX_SIZES_ALL]`: transform pixels.
pub const TX_SIZE_WIDE: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
pub const TX_SIZE_HIGH: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
/// `tx_size_wide_unit` / `tx_size_high_unit`: transform dims in 4x4 mi units.
pub const TX_SIZE_WIDE_UNIT: [usize; 19] =
    [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
pub const TX_SIZE_HIGH_UNIT: [usize; 19] =
    [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];

pub const BLOCK_8X8: usize = 3;
pub const BLOCK_64X64: usize = 12;
pub const BLOCK_128X128: usize = 15;

pub const PARTITION_NONE: usize = 0;
pub const PARTITION_HORZ: usize = 1;
pub const PARTITION_VERT: usize = 2;
pub const PARTITION_SPLIT: usize = 3;
pub const PARTITION_HORZ_A: usize = 4;
pub const PARTITION_HORZ_B: usize = 5;
pub const PARTITION_VERT_A: usize = 6;
pub const PARTITION_VERT_B: usize = 7;
pub const PARTITION_HORZ_4: usize = 8;
pub const PARTITION_VERT_4: usize = 9;

pub const DC_PRED: i32 = 0;
const SMOOTH_PRED: i32 = 9;
const SMOOTH_H_PRED: i32 = 11;
/// `UV_CFL_PRED` (enums.h): the chroma-from-luma UV mode.
pub const UV_CFL_PRED: i32 = 13;
/// `ANGLE_STEP`: coded angle deltas scale by 3 degrees.
pub const ANGLE_STEP: i32 = 3;

/// The 64x64-superblock mi count (16 mi/side) — `mi_size_high[BLOCK_64X64]`.
/// [`KfTileConfig::sb_size_128`] selects the LIVE per-tile value (16 or 32);
/// this constant is the 64x64 default/fallback used where the config isn't
/// in scope.
const SB_MI: i32 = 16;

// ---- chroma spec helpers (av1_common_int.h / blockd.h / reconintra.c) ------------

/// `is_chroma_reference` (av1_common_int.h): does this block carry the chroma
/// information for its (possibly shared) chroma area? With subsampling, blocks
/// with an odd mi count on that axis are chroma-referenced only at odd mi
/// positions (the bottom/right member of the group codes the merged chroma).
pub fn is_chroma_reference(
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
) -> bool {
    let bw = MI_SIZE_WIDE[bsize];
    let bh = MI_SIZE_HIGH[bsize];
    ((mi_row & 0x01) != 0 || (bh & 0x01) == 0 || ss_y == 0)
        && ((mi_col & 0x01) != 0 || (bw & 0x01) == 0 || ss_x == 0)
}

/// `av1_get_adjusted_tx_size` (blockd.h): 64-wide/high transforms clamp to 32
/// on that axis for chroma (and other adjusted-size users).
pub fn adjusted_tx_size(tx_size: usize) -> usize {
    match tx_size {
        4 | 11 | 12 => 3, // TX_64X64 / TX_32X64 / TX_64X32 -> TX_32X32
        18 => 10,         // TX_64X16 -> TX_32X16
        17 => 9,          // TX_16X64 -> TX_16X32
        t => t,
    }
}

/// `av1_get_max_uv_txsize` (blockd.h): the chroma transform size — the largest
/// rectangular transform of the chroma plane block size, 64-clamped. This is
/// the whole-block chroma `tx_size` (`av1_get_tx_size(plane > 0)`, lossless off).
pub fn max_uv_txsize(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
    debug_assert_ne!(plane_bsize, 255, "invalid chroma block size");
    adjusted_tx_size(MAX_TXSIZE_RECT_LOOKUP[plane_bsize])
}

/// `scale_chroma_bsize` (reconintra.c): the block size the chroma availability
/// logic (`has_top_right`/`has_bottom_left`) sees — sub-8x8 dimensions on a
/// subsampled axis are promoted to the shared-chroma group's size.
pub fn scale_chroma_bsize(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    const BLOCK_4X4: usize = 0;
    const BLOCK_4X8: usize = 1;
    const BLOCK_8X4: usize = 2;
    const BLOCK_8X16: usize = 4;
    const BLOCK_16X8: usize = 5;
    const BLOCK_4X16: usize = 16;
    const BLOCK_16X4: usize = 17;
    match bsize {
        BLOCK_4X4 => match (ss_x, ss_y) {
            (1, 1) => BLOCK_8X8,
            (1, 0) => BLOCK_8X4,
            (0, 1) => BLOCK_4X8,
            _ => bsize,
        },
        BLOCK_4X8 => match (ss_x, ss_y) {
            (1, _) => BLOCK_8X8,
            _ => bsize,
        },
        BLOCK_8X4 => match (ss_x, ss_y) {
            (_, 1) => BLOCK_8X8,
            _ => bsize,
        },
        BLOCK_4X16 => match (ss_x, ss_y) {
            (1, _) => BLOCK_8X16,
            _ => bsize,
        },
        BLOCK_16X4 => match (ss_x, ss_y) {
            (_, 1) => BLOCK_16X8,
            _ => bsize,
        },
        _ => bsize,
    }
}

/// `_intra_mode_to_tx_type[INTRA_MODES]` (blockd.h): the transform type an intra
/// prediction mode implies for chroma (UV transform types are not coded).
const INTRA_MODE_TO_TX_TYPE: [usize; 13] = [0, 1, 2, 0, 3, 1, 2, 2, 1, 3, 1, 2, 3];

/// `av1_get_tx_type` (blockd.h), intra UV: the mode-implied type, demoted to
/// DCT_DCT when the block's ext-tx set does not carry it.
pub fn uv_tx_type(uv_mode: i32, uv_tx_size: usize, reduced_tx_set: bool) -> usize {
    let t = INTRA_MODE_TO_TX_TYPE[get_uv_mode(uv_mode as usize) as usize];
    if ext_tx_derive(uv_tx_size, false, reduced_tx_set, t, false, 0, 0).used == 1 {
        t
    } else {
        0
    }
}

/// `ROUND_POWER_OF_TWO`.
#[inline]
fn round_power_of_two(value: i32, n: usize) -> i32 {
    (value + ((1 << n) >> 1)) >> n
}

/// `max_block_wide` / `max_block_high` (blockd.h) for an arbitrary plane: the
/// plane block's in-frame extent in the plane's 4px units — the plane block
/// size reduced by the (negative, luma-eighth-pel) frame-edge distance scaled
/// by the plane subsampling.
pub fn max_block_units_ss(plane_px: i32, mb_to_edge: i32, ss: usize) -> usize {
    let px = if mb_to_edge < 0 {
        plane_px + (mb_to_edge >> (3 + ss))
    } else {
        plane_px
    };
    (px >> 2) as usize
}

// ---- configuration -------------------------------------------------------------

/// Frame/tile-level configuration for the KEY-frame luma decode driver. The tile
/// is the whole frame (tile origin 0,0; tile ends = frame ends). See the crate
/// docs for the gates that are fixed off in this cut.
#[derive(Clone, Debug)]
pub struct KfTileConfig {
    /// Frame height/width in 4x4 mode-info units (whole-mi frame sizes only).
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// Bit depth (8/10/12); pixels are u16 at every depth.
    pub bd: i32,
    /// `seq_params->monochrome`: when true no UV symbols exist and the luma
    /// reconstruction is the complete frame. When false the U/V planes are
    /// fully reconstructed (mode-info symbols, chroma coefficients, intra +
    /// CfL prediction) at the configured subsampling.
    pub monochrome: bool,
    /// `seq_params->subsampling_x` / `subsampling_y` (only meaningful when
    /// `!monochrome`): (0,0) = 4:4:4, (1,0) = 4:2:2, (1,1) = 4:2:0. A
    /// subsampled axis requires an even mi count (the C `set_mb_mi` aligns
    /// frame mi dimensions to 8 pixels, so shared-chroma groups never straddle
    /// the frame edge).
    pub subsampling_x: usize,
    pub subsampling_y: usize,
    /// `cdef_info.cdef_bits` (0..=3): per-64x64 CDEF strength literal width.
    pub cdef_bits: u32,
    /// `!seq_params->enable_intra_edge_filter`.
    pub disable_edge_filter: bool,
    /// `seq_params->enable_filter_intra` (the bsize/mode gates are per block).
    pub enable_filter_intra: bool,
    /// `features.tx_mode`: `TX_MODE_LARGEST` (no tx-size bits) or
    /// `TX_MODE_SELECT` (per-block tx-size depth symbols). `ONLY_4X4` requires
    /// `coded_lossless`, which is out of scope.
    pub tx_mode: TxMode,
    /// `features.reduced_tx_set_used`.
    pub reduced_tx_set: bool,
    /// `quant_params.base_qindex` (the frame base; a block's effective
    /// qindex is `av1_get_qindex(seg, segment_id, base_or_carry)`). Also the
    /// tx-type signalling gate: `av1_read_tx_type` codes tx types only when
    /// `xd->qindex[segment_id]` — the per-segment FRAME-level qindex, not
    /// the delta-q-modified carry — is non-zero. `base_qindex == 0` with all
    /// plane deltas zero would be `coded_lossless` in C (a different decode
    /// path, out of scope): give a zero-base config a non-zero plane delta.
    pub base_qindex: i32,
    /// `quant_params.y_dc_delta_q` / `u_dc_delta_q` / `u_ac_delta_q` /
    /// `v_dc_delta_q` / `v_ac_delta_q` (read_quantization; each in
    /// `[-63, 63]`). Y's AC has no delta — `base_qindex` is the Y AC point.
    pub y_dc_delta_q: i32,
    pub u_dc_delta_q: i32,
    pub u_ac_delta_q: i32,
    pub v_dc_delta_q: i32,
    pub v_ac_delta_q: i32,
    /// `delta_q_info.delta_q_present_flag` (requires `base_qindex > 0`, as
    /// the C frame header only codes it then) + `delta_q_res` (1/2/4/8).
    pub delta_q_present: bool,
    pub delta_q_res: i32,
    /// `delta_q_info.delta_lf_present_flag` / `delta_lf_multi` /
    /// `delta_lf_res` (1/2/4/8). Only codable when `delta_q_present` (the C
    /// frame header nests it there). Decoded + carried per block; no
    /// reconstruction effect (loop filters are not applied in this cut).
    pub delta_lf_present: bool,
    pub delta_lf_multi: bool,
    pub delta_lf_res: i32,
    /// Loop-restoration frame geometry (`cm->rst_info` slice): per-plane
    /// frame restoration type + unit size + the frame crop dims that size
    /// the unit grid. Default (all `RESTORE_NONE`) codes no RU params —
    /// the pre-restoration tile syntax is unchanged.
    pub lr: aom_entropy::lr::LrFrameConfig,
    /// `cm->seg` — the frame's segmentation state (`read_segmentation`, the
    /// KEY-frame form: when enabled, `update_map`/`update_data` are forced on
    /// and `temporal_update` off). When enabled, per-block segment ids are
    /// coded in the tile (the spatial-pred symbol / spatial prediction on
    /// skip), `SEG_LVL_ALT_Q` shifts each block's dequant qindex
    /// (`av1_get_qindex`), and `SEG_LVL_SKIP` forces the block skip flag.
    /// No segment may be LOSSLESS (effective qindex 0 with all plane deltas
    /// zero — `xd->lossless[i]`): that switches the C per-block transform
    /// path (forced TX_4X4 + WHT), which is out of scope (asserted).
    pub seg: Segmentation,
    /// `seq_params->sb_size == BLOCK_128X128` (`use_128x128_superblock` from
    /// the sequence header): `false` = 64x64 superblocks (`mib_size` = 16 mi
    /// per side, partition tree roots at [`BLOCK_64X64`]); `true` = 128x128
    /// (`mib_size` = 32, roots at [`BLOCK_128X128`]). Drives the tile SB
    /// walk step, the partition-tree root bsize, the per-SB left-context
    /// reset stride, [`aom_entropy::partition::read_cdef`]'s
    /// `cdef_transmitted[4]` unit indexing (already generic on `sb_size`),
    /// and the loop-restoration corners-in-sb SB extent
    /// ([`aom_entropy::lr::lr_corners_in_sb`]'s `mi_size_wide`/`mi_size_high`
    /// args). CDEF's own 64x64 filter-block granularity and restoration's
    /// RU (unit) size are independent of this flag (spec-fixed / a separate
    /// config axis, respectively).
    pub sb_size_128: bool,
    /// `features.allow_screen_content_tools` (frame header): gates PALETTE mode
    /// per-block (`av1_allow_palette`, [`allow_palette`] — also needs the
    /// block's own bsize <= 64x64). Intra block copy (`allow_intrabc`) is the
    /// OTHER screen-content tool this flag enables in C.
    pub allow_screen_content_tools: bool,
    /// `p.allow_intrabc` (frame header): intra block copy. Monochrome intrabc
    /// KEY frames are in the envelope; colour intrabc is still rejected upstream
    /// in `frame.rs` pending chroma reconstruction. Drives
    /// [`KfBlockState::allow_intrabc`].
    pub allow_intrabc: bool,
}

/// `av1_calculate_segdata` (av1/common/seg_common.c) — the derived
/// segmentation facts: `segid_preskip` (any active feature `>=
/// SEG_LVL_REF_FRAME` — the id must be read before the skip flag) and
/// `last_active_segid` (the highest segment with any active feature, the
/// segment-id alphabet bound).
pub fn calculate_segdata(seg: &Segmentation) -> (bool, i32) {
    /// `SEG_LVL_REF_FRAME` (seg_common.h).
    const SEG_LVL_REF_FRAME: usize = 5;
    let mut segid_preskip = false;
    let mut last_active_segid = 0i32;
    for i in 0..MAX_SEGMENTS {
        for j in 0..SEG_LVL_MAX {
            if seg.feature_mask[i] & (1 << j) != 0 {
                segid_preskip |= j >= SEG_LVL_REF_FRAME;
                last_active_segid = i as i32;
            }
        }
    }
    (segid_preskip, last_active_segid)
}

/// The per-plane `[dc, ac]` dequant steps for an effective qindex — the shared
/// formula of `setup_segmentation_dequant` (frame level, decodeframe.c) and
/// the per-block delta-q recompute in `parse_decode_block`: per plane the
/// frame's dc/ac deltas fold in via `av1_{dc,ac}_quant_QTX` (which clamp
/// `qindex + delta` to `[0, MAXQ]`); Y's AC delta is always 0. `qindex` is the
/// block's EFFECTIVE index — `av1_get_qindex(seg, segment_id, base_or_carry)`,
/// i.e. one row of the C's per-segment `seg_dequant_QTX` table (the identity
/// row without `SEG_LVL_ALT_Q`).
pub fn plane_dequants(cfg: &KfTileConfig, qindex: i32) -> [[i16; 2]; 3] {
    let bd = cfg.bd as u8;
    [
        [
            av1_dc_quant_qtx(qindex, cfg.y_dc_delta_q, bd),
            av1_ac_quant_qtx(qindex, 0, bd),
        ],
        [
            av1_dc_quant_qtx(qindex, cfg.u_dc_delta_q, bd),
            av1_ac_quant_qtx(qindex, cfg.u_ac_delta_q, bd),
        ],
        [
            av1_dc_quant_qtx(qindex, cfg.v_dc_delta_q, bd),
            av1_ac_quant_qtx(qindex, cfg.v_ac_delta_q, bd),
        ],
    ]
}

// The tile's CDF state is libaom's FRAME_CONTEXT itself —
// [`aom_entropy::partition::KfFrameContext`] (the KEY-frame-intra slice at C
// dims). The driver selects every symbol's instance from its per-context
// arrays; the coefficient arena (`KfFrameContext::coeff`) must be sized
// `aom_txb::CDF_ARENA_LEN`.

// ---- decode result --------------------------------------------------------------

/// One decoded leaf block: its position/size, the partition type that created it,
/// the decoded mode info, and the per-txb `(eob, tx_type)` in raster order
/// (plane 0; skip blocks record `(0, 0)` per txb). `txbs_uv` holds the chroma
/// txbs (plane 1's in raster order, then plane 2's) for chroma-reference
/// blocks — empty for monochrome / non-chroma-reference blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedBlockKf {
    pub mi_row: i32,
    pub mi_col: i32,
    pub bsize: usize,
    pub partition: usize,
    pub info: MbModeInfoKf,
    pub tx_size: usize,
    pub txbs: Vec<(usize, usize)>,
    pub txbs_uv: Vec<(usize, usize)>,
}

/// A decoded KEY-frame tile: the reconstruction planes (superblock-aligned;
/// the frame crop is `width x height` luma pixels / `width_uv x height_uv`
/// chroma pixels at the top-left), the pre-order partition sequence (every
/// visited node, including uncoded forced partitions), and the per-leaf decode
/// records. The chroma planes are empty for monochrome.
#[derive(Clone, Debug)]
pub struct KfTileDecode {
    pub recon: Vec<u16>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
    pub recon_u: Vec<u16>,
    pub recon_v: Vec<u16>,
    pub stride_uv: usize,
    pub width_uv: usize,
    pub height_uv: usize,
    pub tree: Vec<i8>,
    pub blocks: Vec<DecodedBlockKf>,
    /// Per-plane restoration-unit parameters in unit-grid raster order
    /// (`rst_info[plane].unit_info`), decoded interleaved with the SB walk
    /// (`loop_restoration_read_sb_coeffs`); empty when the plane's frame
    /// restoration type is `RESTORE_NONE`.
    pub lr_units: [Vec<aom_entropy::lr::LrUnitInfo>; 3],
}

/// One tile's mi-space extent within the frame (`TileInfo::mi_row_start` /
/// `mi_row_end` / `mi_col_start` / `mi_col_end`, `av1_tile_set_row` /
/// `av1_tile_set_col`, tile_common.c: `row_start_sb[row] << mib_size_log2`,
/// clamped to the frame's `mi_rows`/`mi_cols`). Drives tile-relative
/// `up_available`/`left_available` (`mi_row/col > mi_row/col_start` — NOT
/// `> 0`: a block at a tile's own top/left edge has no available neighbour
/// even when the tile itself sits interior to the frame) and
/// [`aom_entropy::partition::intra_avail`]'s `tile_col_end`/`tile_row_end`
/// (has_top_right/has_bottom_left must not reach across a tile boundary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileBoundsKf {
    pub mi_row_start: i32,
    pub mi_row_end: i32,
    pub mi_col_start: i32,
    pub mi_col_end: i32,
}

impl TileBoundsKf {
    /// The single-tile envelope: one tile spanning the whole frame — what
    /// [`decode_tile_kf`] always uses.
    fn whole_frame(cfg: &KfTileConfig) -> Self {
        TileBoundsKf {
            mi_row_start: 0,
            mi_row_end: cfg.mi_rows,
            mi_col_start: 0,
            mi_col_end: cfg.mi_cols,
        }
    }
}

/// One tile's entropy-coded byte payload + mi-space bounds, in raster
/// `(tile_row, tile_col)` order — the input to [`decode_frame_tiles_kf`].
#[derive(Clone, Copy, Debug)]
pub struct TileBytesKf<'a> {
    pub bytes: &'a [u8],
    pub bounds: TileBoundsKf,
}

// ---- shared driver helpers (also used by the roundtrip mirror encoder) ----------

/// `max_block_wide` / `max_block_high` (av1/common/blockd.h), luma: the block's
/// in-frame extent in 4x4 units — full size, reduced by the (negative)
/// eighth-pel distance past the frame edge.
pub fn max_block_units(full_px: i32, mb_to_edge: i32) -> usize {
    let px = if mb_to_edge < 0 {
        full_px + (mb_to_edge >> 3)
    } else {
        full_px
    };
    (px >> 2) as usize
}

/// `av1_read_tx_type`'s intra CDF selection:
/// `intra_ext_tx_cdf[eset][square_tx_size][intra_dir]` out of the frame
/// context's per-eset arrays ([`KfFrameContext::ext_tx_1ddct`] /
/// [`KfFrameContext::ext_tx_dtt4`], passed as disjoint borrows so the caller
/// keeps the coefficient arena). The intra direction is
/// `fimode_to_intradir[filter_intra_mode]` for a filter-intra block, else the
/// Y mode. DCT-only sets (eset 0) never code a symbol — any slot satisfies the
/// unused argument.
pub fn intra_ext_tx_cdf<'a>(
    ext_tx_1ddct: &'a mut [[[u16; 8]; 13]; 4],
    ext_tx_dtt4: &'a mut [[[u16; 6]; 13]; 4],
    tx_size: usize,
    reduced_tx_set: bool,
    use_filter_intra: bool,
    filter_intra_mode: usize,
    y_mode: usize,
) -> &'a mut [u16] {
    let d = ext_tx_derive(
        tx_size,
        false,
        reduced_tx_set,
        0,
        use_filter_intra,
        filter_intra_mode,
        y_mode,
    );
    match d.eset {
        1 => &mut ext_tx_1ddct[d.square as usize][d.intra_dir as usize],
        2 => &mut ext_tx_dtt4[d.square as usize][d.intra_dir as usize],
        _ => &mut ext_tx_dtt4[0][0], // DCT-only set: never coded
    }
}

/// `av1_read_tx_type`'s INTER CDF selection for an intra-block-copy block:
/// `inter_ext_tx_cdf[eset][square_tx_size]` out of the frame context's padded
/// `[EXT_TX_SETS_INTER][EXT_TX_SIZES][CDF_SIZE(TX_TYPES)]` table. Intrabc is
/// `is_inter_block`, so `ext_tx_derive(is_inter = true)` picks the inter set; the
/// intra-direction / filter-intra arguments are unused on the inter path.
/// eset 0 (DCT-only) codes nothing — any slot satisfies the unused argument.
pub fn inter_ext_tx_cdf(
    inter_ext_tx: &mut [[[u16; 17]; 4]; 4],
    tx_size: usize,
    reduced_tx_set: bool,
) -> &mut [u16] {
    let d = ext_tx_derive(tx_size, true, reduced_tx_set, 0, false, 0, 0);
    &mut inter_ext_tx[d.eset as usize][d.square as usize]
}

// ---- the driver -----------------------------------------------------------------

/// A [`DvGrid`] view over the decode driver's per-mi block-vector grid
/// (`TileKf::mi_dv`), addressed by offset from the current block's
/// `(mi_row, mi_col)`. `av1_find_mv_refs`'s own tile-bounds `is_inside` gate
/// filters candidates, so out-of-frame offsets return a non-contributing
/// default cell.
struct MiDvGrid<'a> {
    mi_dv: &'a [DvNbr],
    cols: i32,
    rows: i32,
    mi_row: i32,
    mi_col: i32,
}

impl DvGrid for MiDvGrid<'_> {
    fn get(&self, row_offset: i32, col_offset: i32) -> DvNbr {
        let r = self.mi_row + row_offset;
        let c = self.mi_col + col_offset;
        if r >= 0 && r < self.rows && c >= 0 && c < self.cols {
            self.mi_dv[(r * self.cols + c) as usize]
        } else {
            DvNbr::default()
        }
    }
}

/// `TXFM_PARTITION_CONTEXTS` = `(TX_SIZES - TX_8X8) * 6 - 3` = 21 (`enums.h`):
/// the number of var-tx split-flag CDF contexts.
const TXFM_PARTITION_CONTEXTS: usize = 21;

/// `default_txfm_partition_cdf` (`av1/common/entropymode.c`): the fixed default
/// var-tx split-flag CDF — one `AOM_CDF2(x)` per context, stored as
/// `[32768 - x, 0, 0]` (matching every other 2-symbol default, e.g.
/// `DEFAULT_INTRABC`). Copied verbatim into the frame context at setup (NOT
/// qindex-selected), then adapted per read.
///
/// FORK NOTE: this CDF logically belongs inside
/// [`aom_entropy::partition::KfFrameContext`] alongside every other frame CDF,
/// but that struct is outside this fork's edit scope, so the decoder hosts it on
/// [`TileKf`] as per-tile state (reset in [`TileKf::start_tile`] like the
/// per-tile `KfFrameContext`). For a KEY frame this is byte-identical — the
/// context always loads from this fixed default and there is no cross-frame CDF
/// carry (`primary_ref_frame == PRIMARY_REF_NONE`). Relocate into
/// `KfFrameContext` when integrating.
const DEFAULT_TXFM_PARTITION_CDF: [[u16; 3]; TXFM_PARTITION_CONTEXTS] = [
    [4187, 0, 0],
    [8922, 0, 0],
    [11921, 0, 0],
    [8453, 0, 0],
    [14572, 0, 0],
    [20635, 0, 0],
    [13977, 0, 0],
    [21881, 0, 0],
    [21763, 0, 0],
    [5589, 0, 0],
    [12764, 0, 0],
    [21487, 0, 0],
    [6219, 0, 0],
    [13460, 0, 0],
    [18544, 0, 0],
    [4753, 0, 0],
    [11222, 0, 0],
    [18368, 0, 0],
    [4603, 0, 0],
    [10367, 0, 0],
    [16680, 0, 0],
];

/// `sub_tx_size_map[TX_SIZES_ALL]` (`common_data.h`): one var-tx quadtree split step.
const SUB_TX_SIZE_MAP: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
/// `MAX_VARTX_DEPTH` (`enums.h`).
const VARTX_MAX_DEPTH: i32 = 2;
/// `TX_4X4` transform-size index — the var-tx recursion's terminal size.
const TX_4X4_IDX: usize = 0;

/// `max_block_wide` (`av1_common_int.h`) for luma (plane 0, subsampling 0): the
/// block's transform-unit width, clipped to the frame's right edge via the
/// 1/8-pel `mb_to_right_edge`.
fn max_block_wide_luma(bsize: usize, mb_to_right_edge: i32) -> i32 {
    let mut w = BLOCK_SIZE_WIDE[bsize];
    if mb_to_right_edge < 0 {
        w += mb_to_right_edge >> 3; // 3 + subsampling_x(=0)
    }
    w >> 2 // MI_SIZE_LOG2
}

/// `max_block_high` (`av1_common_int.h`) for luma (plane 0, subsampling 0).
fn max_block_high_luma(bsize: usize, mb_to_bottom_edge: i32) -> i32 {
    let mut h = BLOCK_SIZE_HIGH[bsize];
    if mb_to_bottom_edge < 0 {
        h += mb_to_bottom_edge >> 3; // 3 + subsampling_y(=0)
    }
    h >> 2 // MI_SIZE_LOG2
}

/// Record a var-tx leaf tx size for the uniformity guard: the first leaf sets
/// the reference size; any later leaf of a different size flags the partition
/// non-uniform (see [`read_tx_size_vartx`]).
fn vartx_record_leaf(first_leaf: &mut i32, non_uniform: &mut bool, leaf: usize) {
    if *first_leaf < 0 {
        *first_leaf = leaf as i32;
    } else if *first_leaf != leaf as i32 {
        *non_uniform = true;
    }
}

/// `read_tx_size_vartx` (`av1/decoder/decodeframe.c`): the recursive
/// variable-transform-size split READER for an inter / intrabc block, the exact
/// inverse of the encoder's [`aom_entropy::partition::write_tx_size_vartx`]. At
/// each quadtree node it reads a split flag from `txfm_partition_cdf[ctx]` (2
/// symbols), stamps the neighbour txfm-context arrays via
/// [`txfm_partition_update`], and recurses into the `sub_tx_size_map` children
/// down to `MAX_VARTX_DEPTH`. `tx_size_out` receives C's `mbmi->tx_size` — set at
/// exactly the leaves where C sets it (an out-of-bounds child leaves the running
/// value untouched, matching C's early `return`).
///
/// The residual coefficient loop consumes only this scalar `tx_size_out`, so a
/// block whose var-tx partition is NON-uniform (distinct tx sizes per txb — only
/// possible for blocks taller/wider than 2 tx units) would still need the full
/// `inter_tx_size[]` walk in the coeff reader. The KEY-frame intrabc streams this
/// path serves only produce uniform partitions (BLOCK_4X8 → one split flag →
/// either TX_4X8 or TX_4X4, both uniform), for which the scalar is exact.
#[allow(clippy::too_many_arguments)]
fn read_tx_size_vartx(
    dec: &mut OdEcDec,
    txfm_partition_cdf: &mut [[u16; 3]; TXFM_PARTITION_CONTEXTS],
    above_ctx: &mut [u8],
    left_ctx: &mut [u8],
    bsize: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    tx_size: usize,
    depth: i32,
    blk_row: i32,
    blk_col: i32,
    tx_size_out: &mut usize,
    // Uniformity guard: `first_leaf` is the first leaf tx size seen (-1 = none
    // yet); `non_uniform` is set if any later leaf differs. The residual coeff
    // loop consumes only the scalar `tx_size_out`, which is exact ONLY when the
    // whole block tiles uniformly — so the caller rejects a non-uniform read
    // (rather than silently mis-tiling the coefficients and desyncing).
    first_leaf: &mut i32,
    non_uniform: &mut bool,
) {
    let max_blocks_high = max_block_high_luma(bsize, mb_to_bottom_edge);
    let max_blocks_wide = max_block_wide_luma(bsize, mb_to_right_edge);
    if blk_row >= max_blocks_high || blk_col >= max_blocks_wide {
        return;
    }
    let (bc, br) = (blk_col as usize, blk_row as usize);
    if depth == VARTX_MAX_DEPTH {
        *tx_size_out = tx_size;
        vartx_record_leaf(first_leaf, non_uniform, tx_size);
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
        return;
    }

    let ctx = txfm_partition_context(above_ctx[bc], left_ctx[br], bsize, tx_size);
    let is_split = read_symbol(dec, &mut txfm_partition_cdf[ctx], 2);
    if is_split != 0 {
        let sub_txs = SUB_TX_SIZE_MAP[tx_size];
        if sub_txs == TX_4X4_IDX {
            *tx_size_out = sub_txs;
            vartx_record_leaf(first_leaf, non_uniform, sub_txs);
            txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], sub_txs, tx_size);
            return;
        }
        let bsw = TX_SIZE_WIDE_UNIT[sub_txs] as i32;
        let bsh = TX_SIZE_HIGH_UNIT[sub_txs] as i32;
        let mut row = 0;
        while row < TX_SIZE_HIGH_UNIT[tx_size] as i32 {
            let mut col = 0;
            while col < TX_SIZE_WIDE_UNIT[tx_size] as i32 {
                read_tx_size_vartx(
                    dec,
                    txfm_partition_cdf,
                    above_ctx,
                    left_ctx,
                    bsize,
                    mb_to_right_edge,
                    mb_to_bottom_edge,
                    sub_txs,
                    depth + 1,
                    blk_row + row,
                    blk_col + col,
                    tx_size_out,
                    first_leaf,
                    non_uniform,
                );
                col += bsw;
            }
            row += bsh;
        }
    } else {
        *tx_size_out = tx_size;
        vartx_record_leaf(first_leaf, non_uniform, tx_size);
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
    }
}

/// Intra-block-copy chroma prediction: copy a `w x h` block from `src_plane`
/// at the DV-derived integer position `src_off`, applying the 2-tap intrabc
/// bilinear filter when the (subsampled) chroma DV lands at half-pel. libaom's
/// `av1_intrabc_bilinear_filter` is `{128,0}` at full-pel and `{64,64}` at
/// half-pel; across all four subpel cases the `av1_highbd_convolve_*_sr_intrabc`
/// kernels reduce to these closed forms, bit-identical at every bit depth (the
/// FILTER_BITS=7 rounding collapses to a simple average). `subpel_x`/`subpel_y`
/// are each 0 (full) or 8 (half); the half-pel axes read one extra column/row,
/// which DV validity guarantees is already reconstructed.
#[allow(clippy::too_many_arguments)]
fn intrabc_chroma_predict(
    src_plane: &[u16],
    src_off: usize,
    stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    subpel_x: i32,
    subpel_y: i32,
    bd: i32,
) {
    let max = (1i32 << bd) - 1;
    let clip = |v: i32| v.clamp(0, max) as u16;
    match (subpel_x != 0, subpel_y != 0) {
        (false, false) => {
            for r in 0..h {
                let s = src_off + r * stride;
                let d = r * dst_stride;
                dst[d..d + w].copy_from_slice(&src_plane[s..s + w]);
            }
        }
        (true, false) => {
            // horizontal half-pel: (a + b + 1) >> 1
            for r in 0..h {
                let s = src_off + r * stride;
                let d = r * dst_stride;
                for c in 0..w {
                    let a = src_plane[s + c] as i32;
                    let b = src_plane[s + c + 1] as i32;
                    dst[d + c] = clip((a + b + 1) >> 1);
                }
            }
        }
        (false, true) => {
            // vertical half-pel: (a + b + 1) >> 1
            for r in 0..h {
                let s = src_off + r * stride;
                let d = r * dst_stride;
                for c in 0..w {
                    let a = src_plane[s + c] as i32;
                    let b = src_plane[s + c + stride] as i32;
                    dst[d + c] = clip((a + b + 1) >> 1);
                }
            }
        }
        (true, true) => {
            // both half-pel: (a00 + a01 + a10 + a11 + 2) >> 2
            for r in 0..h {
                let s = src_off + r * stride;
                let d = r * dst_stride;
                for c in 0..w {
                    let a00 = src_plane[s + c] as i32;
                    let a01 = src_plane[s + c + 1] as i32;
                    let a10 = src_plane[s + c + stride] as i32;
                    let a11 = src_plane[s + c + stride + 1] as i32;
                    dst[d + c] = clip((a00 + a01 + a10 + a11 + 2) >> 2);
                }
            }
        }
    }
}

struct TileKf<'c> {
    cfg: &'c KfTileConfig,
    /// Luma reconstruction plane, SB-aligned, `stride` = aligned width in px.
    recon: Vec<u16>,
    stride: usize,
    /// Chroma reconstruction planes (`stride_uv = stride >> ss_x`); empty when
    /// monochrome.
    recon_u: Vec<u16>,
    recon_v: Vec<u16>,
    stride_uv: usize,
    /// Per-plane coefficient entropy contexts (`xd->above_entropy_context[plane]`
    /// / `xd->left_entropy_context[plane]`): above spans the aligned tile width
    /// (one i8 per mi col, zeroed once per tile; chroma planes index it at
    /// `adjusted_mi_col >> ss_x`); left is the one-SB-tall rolling column,
    /// zeroed at each SB row, indexed by `(mi_row & 31) >> ss_y`.
    above_e: [Vec<i8>; 3],
    left_e: [[i8; 32]; 3],
    /// Partition contexts with the same lifetimes/indexing.
    above_p: Vec<i8>,
    left_p: [i8; 32],
    /// Txfm contexts (`above_txfm_context` / `left_txfm_context_buffer`): one
    /// byte per mi holding the neighbouring transform's pixel width (above) /
    /// height (left), reset to [`TXFM_CTX_INIT`] (=64) per tile / per SB row
    /// and stamped by `set_txfm_ctxs` after every block's tx size resolves.
    /// Feeds `get_tx_size_context` under `TX_MODE_SELECT`.
    above_t: Vec<u8>,
    left_t: [u8; 32],
    /// Var-tx split-flag CDF (`txfm_partition_cdf[TXFM_PARTITION_CONTEXTS]`),
    /// read by intrabc (is_inter) blocks that signal their transform size via
    /// [`read_tx_size_vartx`]. Hosted here rather than in the shared
    /// [`KfFrameContext`] (see [`DEFAULT_TXFM_PARTITION_CDF`]); reset to the
    /// fixed default per tile in [`TileKf::start_tile`], adapts within a tile.
    txfm_partition: [[u16; 3]; TXFM_PARTITION_CONTEXTS],
    /// Per-mi mode-info grid (frame-cropped stamps, like the C mi grid): the
    /// [`MiNbrKf`] projection (`y_mode` + `skip_txfm`) every context selection
    /// reads through `xd->above_mbmi` / `xd->left_mbmi`, and which also feeds
    /// `get_filt_type` (edge-filter type 1 when a neighbour's y mode is
    /// SMOOTH/SMOOTH_V/SMOOTH_H).
    mi: Vec<MiNbrKf>,
    /// Per-mi block-vector grid: the intrabc projection of `xd->mi[]` that
    /// `av1_find_mv_refs(INTRA_FRAME)` scans (`use_intrabc`, own `bsize`, and the
    /// block's DV `mv[0]`), kept parallel to `mi` rather than fattening the
    /// encode-shared [`MiNbrKf`]. Frame-cropped stamps (like `mi`); every cell
    /// is a non-intrabc DC default until an intrabc block stamps its DV. Also
    /// carries the neighbour `is_inter_block`/`bsize` `get_tx_size_context` reads.
    mi_dv: Vec<DvNbr>,
    /// Per-mi luma transform-type grid (`cm->tx_type_map`, mi granularity): the
    /// luma tx-type each 4x4 unit resolved to, stamped over every luma txb's mi
    /// footprint. Read only by colour intrabc chroma reconstruction, where the
    /// chroma tx-type is the CO-LOCATED luma tx-type (`av1_get_tx_type`,
    /// `is_inter_block` chroma branch) rather than a UV-mode-derived one — the
    /// co-location can reach a sibling block in a shared sub-8x8 chroma group,
    /// so this must be tile-wide, not per-block. Empty (never indexed) unless
    /// the frame is colour AND allows intrabc, so ordinary frames pay nothing.
    luma_tt: Vec<u8>,
    /// Per-mi UV-mode grid (same frame-cropped stamps): what the chroma
    /// `get_filt_type` reads through `xd->chroma_above_mbmi` /
    /// `xd->chroma_left_mbmi` (`is_smooth(mbmi, plane > 0)` checks the
    /// neighbour's uv_mode). Non-chroma-reference blocks stamp `UV_DC_PRED`
    /// (the C `read_intra_frame_mode_info` else-branch), but the chroma
    /// neighbour pointers only ever land on chroma-reference cells.
    mi_uv: Vec<i8>,
    /// Per-mi palette-neighbour grid ([`PaletteNbrKf`], same frame-cropped stamping as
    /// `mi` above): what `xd->above_mbmi->palette_mode_info` /
    /// `xd->left_mbmi->palette_mode_info` project for `get_palette_cache`'s
    /// neighbour-colour merge. Only allocated (`mi_rows x mi_cols`) when
    /// `cfg.allow_screen_content_tools` — empty (never indexed, since `st.allow_palette`
    /// is then always false) otherwise, so non-screen-content frames pay nothing.
    mi_palette: Vec<PaletteNbrKf>,
    /// The CURRENT frame's segment-id map (`cm->cur_frame->seg_map`,
    /// mi_rows x mi_cols): each block stamps its resolved segment id over its
    /// frame-cropped footprint (`set_segment_id`, decodemv.c), and the next
    /// blocks' spatial predictions (`av1_get_spatial_seg_pred`) read their
    /// up-left/up/left cells — always already-decoded positions. Zeroed like
    /// the C's freshly-allocated map; untouched when segmentation is off.
    seg_map: Vec<u8>,
    /// The CfL luma store (one per tile, like `xd->cfl`): sub-8x8 members of a
    /// shared-chroma group accumulate into it across blocks.
    cfl: CflCtx,
    /// The live per-plane `[dc, ac]` dequant rows (`pd->seg_dequant_QTX[0]`,
    /// segment 0): frame-constant from `base_qindex` without delta-q,
    /// recomputed per coded block from the running `current_base_qindex`
    /// carry when `delta_q_present` (`parse_decode_block`, decodeframe.c).
    dequants: [[i16; 2]; 3],
    st: KfBlockState,
    tree: Vec<i8>,
    blocks: Vec<DecodedBlockKf>,
    /// Per-plane restoration-unit params (unit-grid raster order); sized
    /// `horz_units * vert_units` for restored planes, empty otherwise.
    lr_units: [Vec<aom_entropy::lr::LrUnitInfo>; 3],
    /// `xd->wiener_info` / `xd->sgrproj_info` — the per-plane RU-params
    /// prediction references (`av1_reset_loop_restoration` at tile start).
    lr_refs: aom_entropy::lr::LrRefState,
    /// This tile's mi-space extent (`xd->tile`) — see [`TileBoundsKf`]. Set
    /// by `start_tile` at the beginning of each tile's decode.
    tile: TileBoundsKf,
}

impl<'c> TileKf<'c> {
    fn new(cfg: &'c KfTileConfig, recon_init: u16) -> Self {
        assert!(cfg.mi_rows > 0 && cfg.mi_cols > 0, "empty frame");
        assert!(matches!(cfg.bd, 8 | 10 | 12), "bd must be 8/10/12");
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        assert!(
            ss_x < 2 && ss_y < 2 && (ss_x, ss_y) != (0, 1),
            "(0,1) is 4:4:0 — not an AV1 config"
        );
        assert!(
            !cfg.delta_q_present || cfg.base_qindex > 0,
            "delta_q_present requires base_qindex > 0 (the C frame header only codes it then)"
        );
        assert!(
            !cfg.delta_lf_present || cfg.delta_q_present,
            "delta_lf_present is nested inside delta_q_present in the C frame header"
        );
        if !cfg.monochrome {
            // C set_mb_mi aligns frame mi dims to 8 px, so a subsampled axis is
            // always even — shared-chroma groups never straddle the frame edge.
            assert!(
                (ss_x == 0 || (cfg.mi_cols as usize).is_multiple_of(2))
                    && (ss_y == 0 || (cfg.mi_rows as usize).is_multiple_of(2)),
                "subsampled axes require even mi dimensions"
            );
        }
        // Derived segmentation facts (av1_calculate_segdata) + the per-segment
        // SEG_LVL_SKIP mask the skip read resolves against.
        let (segid_preskip, last_active_segid) = calculate_segdata(&cfg.seg);
        let seg_skip_feature: [bool; MAX_SEGMENTS] =
            std::array::from_fn(|i| cfg.seg.feature_mask[i] & (1 << SEG_LVL_SKIP) != 0);
        if cfg.seg.enabled {
            // xd->lossless[i] (decodeframe.c:5166): a lossless SEGMENT flips
            // the per-block transform path (read_tx_size -> TX_4X4 + WHT),
            // out of scope. Only ids 0..=last_active_segid are decodable.
            for i in 0..=last_active_segid as usize {
                assert!(
                    av1_get_qindex(&cfg.seg, i, cfg.base_qindex) != 0
                        || cfg.y_dc_delta_q != 0
                        || cfg.u_dc_delta_q != 0
                        || cfg.u_ac_delta_q != 0
                        || cfg.v_dc_delta_q != 0
                        || cfg.v_ac_delta_q != 0,
                    "segment {i} is lossless (effective qindex 0) — out of scope"
                );
            }
        }
        // `seq_params->sb_size`: mib_size = mi_size_high[sb_size] (16 or 32
        // mi/side), sb_size_block = the partition-tree root BLOCK_SIZE.
        let mib_size: i32 = if cfg.sb_size_128 { 2 * SB_MI } else { SB_MI };
        let sb_size_block: usize = if cfg.sb_size_128 {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        let aligned_mi_cols =
            (cfg.mi_cols as usize).div_ceil(mib_size as usize) * mib_size as usize;
        let aligned_mi_rows =
            (cfg.mi_rows as usize).div_ceil(mib_size as usize) * mib_size as usize;
        let stride = aligned_mi_cols * 4;
        let stride_uv = if cfg.monochrome { 0 } else { stride >> ss_x };
        let uv_len = if cfg.monochrome {
            0
        } else {
            stride_uv * ((aligned_mi_rows * 4) >> ss_y)
        };
        let st = KfBlockState {
            segid_preskip,
            seg_enabled: cfg.seg.enabled,
            // KEY frames force update_map = 1 when segmentation is enabled
            // (setup_segmentation's PRIMARY_REF_NONE arm).
            update_map: cfg.seg.enabled,
            seg_pred: 0,
            seg_cdf_num: 0,
            last_active_segid,
            seg_skip_feature,
            mi_row: 0,
            mi_col: 0,
            mib_size,
            sb_size: sb_size_block,
            bsize: sb_size_block,
            coded_lossless: false,
            allow_intrabc: cfg.allow_intrabc,
            cdef_bits: cfg.cdef_bits,
            dq_present: cfg.delta_q_present,
            dlf_present: cfg.delta_lf_present,
            dlf_multi: cfg.delta_lf_multi,
            num_planes: if cfg.monochrome { 1 } else { 3 },
            dq_res: cfg.delta_q_res,
            dlf_res: cfg.delta_lf_res,
            monochrome: cfg.monochrome,
            is_chroma_ref: !cfg.monochrome, // 4:4:4 when chroma is modelled
            cfl_allowed: false,
            allow_palette: false,
            bit_depth: cfg.bd,
            // The mode-dependent real gate is applied via the follow-up
            // read_filter_intra_mode_info call; the in-driver read never fires.
            filter_allowed: false,
            mb_to_top_edge: 0,
            has_above: false,
            has_left: false,
            cdef_transmitted: [false; 4],
            // xd->current_base_qindex = quant_params.base_qindex per tile
            // (decode_tile setup, decodeframe.c); the delta-lf carries start
            // at zero (av1_reset_loop_filter_delta).
            current_base_qindex: cfg.base_qindex,
            xd_delta_lf: [0; 4],
            xd_delta_lf_from_base: 0,
        };
        let mut t = TileKf {
            cfg,
            recon: vec![recon_init; stride * aligned_mi_rows * 4],
            stride,
            recon_u: vec![recon_init; uv_len],
            recon_v: vec![recon_init; uv_len],
            stride_uv,
            above_e: [
                vec![0; aligned_mi_cols],
                vec![0; aligned_mi_cols],
                vec![0; aligned_mi_cols],
            ],
            left_e: [[0; 32]; 3],
            above_p: vec![0; aligned_mi_cols],
            left_p: [0; 32],
            above_t: vec![TXFM_CTX_INIT; aligned_mi_cols],
            left_t: [TXFM_CTX_INIT; 32],
            txfm_partition: DEFAULT_TXFM_PARTITION_CDF,
            mi: vec![
                MiNbrKf {
                    y_mode: 0,
                    skip_txfm: 0
                };
                (cfg.mi_rows * cfg.mi_cols) as usize
            ],
            mi_dv: vec![DvNbr::default(); (cfg.mi_rows * cfg.mi_cols) as usize],
            luma_tt: if cfg.allow_intrabc && !cfg.monochrome {
                vec![0u8; (cfg.mi_rows * cfg.mi_cols) as usize]
            } else {
                Vec::new()
            },
            mi_uv: vec![0; (cfg.mi_rows * cfg.mi_cols) as usize],
            mi_palette: if cfg.allow_screen_content_tools {
                vec![PaletteNbrKf::default(); (cfg.mi_rows * cfg.mi_cols) as usize]
            } else {
                Vec::new()
            },
            seg_map: vec![0; (cfg.mi_rows * cfg.mi_cols) as usize],
            cfl: CflCtx::new(ss_x as i32, ss_y as i32),
            // setup_segmentation_dequant: the frame-level dequant rows from
            // base_qindex (the live values until a per-block recompute).
            dequants: plane_dequants(cfg, cfg.base_qindex),
            st,
            tree: Vec::new(),
            blocks: Vec::new(),
            lr_units: std::array::from_fn(|p| {
                if cfg.lr.frame_restoration_type[p] == aom_entropy::lr::RESTORE_NONE {
                    Vec::new()
                } else {
                    let (hu, vu) = cfg.lr.plane_units(p, ss_x, ss_y);
                    vec![aom_entropy::lr::LrUnitInfo::default(); (hu * vu) as usize]
                }
            }),
            lr_refs: aom_entropy::lr::LrRefState::default(),
            tile: TileBoundsKf::whole_frame(cfg),
        };
        // Run the same per-tile reset `start_tile` applies to any later tile
        // — for the first (or only) tile this re-touches already-fresh state
        // (the arrays above are already zeroed/defaulted), but keeps `new`
        // and `start_tile` as the ONE place that logic lives.
        let whole_frame = TileBoundsKf::whole_frame(cfg);
        t.start_tile(whole_frame);
        t
    }

    /// Reset the per-TILE transient state at the start of a new tile's
    /// decode (`decode_tile`'s prologue + `decode_tiles`' per-tile setup,
    /// decodeframe.c): `av1_zero_above_context` — the tile's OWN column-range
    /// slice of `above_e`/`above_p`/`above_t` (chroma planes additionally
    /// ss_x-scaled), NOT the whole frame-wide array: tiles in the same tile
    /// row conceptually share one underlying per-tile-row buffer in C
    /// (`above_contexts->entropy[plane][tile_row]`, absolute-mi_col-indexed);
    /// reusing ONE array across every tile (rows and columns alike) and
    /// zeroing only the incoming tile's own slice is bit-exact for a
    /// single-threaded sequential tile walk (no tile ever reads outside its
    /// own column range, and every tile's own slice is freshly zeroed right
    /// before that tile's first read of it) — `av1_reset_loop_filter_delta`
    /// (delta-lf carries), `av1_reset_loop_restoration` (wiener/sgrproj
    /// prediction refs), `cfl_init` (the CfL store — `av1_init_macroblockd`,
    /// called per tile in `decode_tiles`, resets it via `cfl_init`), and
    /// `xd->current_base_qindex = quant_params.base_qindex` (the delta-q
    /// carry restarts at the frame base every tile, `decode_tiles`) + the
    /// frame-level dequant recompute that follows from it. Left context
    /// (`av1_zero_left_context`) is NOT reset here — it is a per-SB-ROW
    /// reset (`decode_one_tile`'s row loop), independent of tile boundaries
    /// (it already fires for the first SB row of every tile, same as any
    /// other row).
    fn start_tile(&mut self, tile: TileBoundsKf) {
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let mib_size = self.st.mib_size as usize;
        let width = (tile.mi_col_end - tile.mi_col_start) as usize;
        let aligned_width = width.div_ceil(mib_size) * mib_size;
        let a0 = tile.mi_col_start as usize;
        self.above_e[0][a0..a0 + aligned_width].fill(0);
        if !cfg.monochrome {
            let uv_a0 = a0 >> ss_x;
            let uv_len = aligned_width >> ss_x;
            self.above_e[1][uv_a0..uv_a0 + uv_len].fill(0);
            self.above_e[2][uv_a0..uv_a0 + uv_len].fill(0);
        }
        self.above_p[a0..a0 + aligned_width].fill(0);
        self.above_t[a0..a0 + aligned_width].fill(TXFM_CTX_INIT);
        // Per-tile fresh var-tx CDF, mirroring the per-tile `KfFrameContext`
        // reload (`tile_data->tctx = *cm->fc`); adapts within the tile only.
        self.txfm_partition = DEFAULT_TXFM_PARTITION_CDF;

        self.cfl = CflCtx::new(ss_x as i32, ss_y as i32);
        self.lr_refs = aom_entropy::lr::LrRefState::default();
        self.st.xd_delta_lf = [0; 4];
        self.st.xd_delta_lf_from_base = 0;
        self.st.current_base_qindex = cfg.base_qindex;
        self.dequants = plane_dequants(cfg, cfg.base_qindex);
        self.tile = tile;
    }

    /// Decode one tile: the `decode_tile` SB row/col loop within
    /// `self.tile`'s bounds (`start_tile` must have been called first — sets
    /// `self.tile` plus the per-tile context/CfL/LR-ref/delta-lf/qindex
    /// resets). Left contexts are zeroed once per SB row (fires for every
    /// row of every tile, matching `av1_zero_left_context`'s per-row call
    /// inside `decode_tile`), each superblock decoded through the recursive
    /// partition walk. The SB-bound checks inside the recursion itself
    /// (`decode_partition`'s early return, `has_rows`/`has_cols`) stay
    /// FRAME-relative (`self.cfg.mi_rows`/`mi_cols`), matching the C
    /// (`cm->mi_params.mi_rows`/`mi_cols`, decodeframe.c) exactly: every
    /// non-final tile boundary is SB-grid-aligned by construction (only the
    /// frame's true bottom/right edge ever clips a partial superblock), so a
    /// tile-rooted SB's recursion never needs to distinguish "my tile's edge"
    /// from "the frame's edge" — they coincide whenever it matters.
    fn decode_one_tile(&mut self, dec: &mut OdEcDec, cdfs: &mut KfFrameContext) {
        let sb_size = self.st.sb_size;
        let mib_size = self.st.mib_size;
        let mut mi_row = self.tile.mi_row_start;
        while mi_row < self.tile.mi_row_end {
            self.left_e = [[0; 32]; 3]; // av1_zero_left_context per SB row (all planes)
            self.left_p = [0; 32];
            self.left_t = [TXFM_CTX_INIT; 32]; // ..incl the left txfm-context bytes
            let mut mi_col = self.tile.mi_col_start;
            while mi_col < self.tile.mi_col_end {
                self.decode_partition(dec, cdfs, mi_row, mi_col, sb_size);
                mi_col += mib_size;
            }
            mi_row += mib_size;
        }
    }

    /// Assemble the [`KfTileDecode`] result from the (possibly multi-tile)
    /// accumulated frame state.
    fn into_decode(self) -> KfTileDecode {
        let cfg = self.cfg;
        KfTileDecode {
            recon: self.recon,
            stride: self.stride,
            width: cfg.mi_cols as usize * 4,
            height: cfg.mi_rows as usize * 4,
            recon_u: self.recon_u,
            recon_v: self.recon_v,
            stride_uv: self.stride_uv,
            width_uv: if cfg.monochrome {
                0
            } else {
                (cfg.mi_cols as usize * 4) >> cfg.subsampling_x
            },
            height_uv: if cfg.monochrome {
                0
            } else {
                (cfg.mi_rows as usize * 4) >> cfg.subsampling_y
            },
            tree: self.tree,
            blocks: self.blocks,
            lr_units: self.lr_units,
        }
    }

    /// The `xd->above_mbmi` / `xd->left_mbmi` neighbours of the block at
    /// `(mi_row, mi_col)`: the mi-grid entries directly above / left of the
    /// block origin, `None` when off the tile (`up_available`/`left_available`
    /// — `set_mi_row_col` gates `above_mbmi`/`left_mbmi` by exactly these,
    /// TILE-relative, not frame-relative: a block at a tile's own top/left
    /// edge has no available neighbour even when interior to the frame,
    /// though the underlying `mi` grid is frame-persistent and may still
    /// hold a previous tile's stamped data there).
    fn neighbours(&self, mi_row: i32, mi_col: i32) -> (Option<MiNbrKf>, Option<MiNbrKf>) {
        let cols = self.cfg.mi_cols;
        let above = (mi_row > self.tile.mi_row_start)
            .then(|| self.mi[((mi_row - 1) * cols + mi_col) as usize]);
        let left = (mi_col > self.tile.mi_col_start)
            .then(|| self.mi[(mi_row * cols + mi_col - 1) as usize]);
        (above, left)
    }

    /// Stamp the block's mode info over its frame-cropped mi footprint (the
    /// mi-grid stamp `set_offsets` clips with `x_mis`/`y_mis`).
    fn stamp_mi(&mut self, mi_row: i32, mi_col: i32, bsize: usize, cell: MiNbrKf) {
        let x_mis = MI_SIZE_WIDE[bsize].min(self.cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(self.cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * self.cfg.mi_cols + mi_col) as usize;
            self.mi[base..base + x_mis as usize].fill(cell);
        }
    }

    /// [`Self::stamp_mi`]'s block-vector twin: stamp the intrabc projection
    /// ([`DvNbr`]) over the block's frame-cropped mi footprint so later blocks'
    /// `av1_find_mv_refs` scans and `get_tx_size_context` inter checks see it.
    fn stamp_dv(&mut self, mi_row: i32, mi_col: i32, bsize: usize, cell: DvNbr) {
        let x_mis = MI_SIZE_WIDE[bsize].min(self.cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(self.cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * self.cfg.mi_cols + mi_col) as usize;
            self.mi_dv[base..base + x_mis as usize].fill(cell);
        }
    }

    /// [`Self::neighbours`]'s palette-info twin — `None` when `mi_palette` is empty
    /// (non-screen-content frames, where it's never allocated).
    fn palette_neighbours(
        &self,
        mi_row: i32,
        mi_col: i32,
    ) -> (Option<PaletteNbrKf>, Option<PaletteNbrKf>) {
        if self.mi_palette.is_empty() {
            return (None, None);
        }
        let cols = self.cfg.mi_cols;
        let above = (mi_row > self.tile.mi_row_start)
            .then(|| self.mi_palette[((mi_row - 1) * cols + mi_col) as usize]);
        let left = (mi_col > self.tile.mi_col_start)
            .then(|| self.mi_palette[(mi_row * cols + mi_col - 1) as usize]);
        (above, left)
    }

    /// [`Self::stamp_mi`]'s palette-info twin — a no-op when `mi_palette` is empty.
    fn stamp_palette(&mut self, mi_row: i32, mi_col: i32, bsize: usize, cell: PaletteNbrKf) {
        if self.mi_palette.is_empty() {
            return;
        }
        let x_mis = MI_SIZE_WIDE[bsize].min(self.cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(self.cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * self.cfg.mi_cols + mi_col) as usize;
            self.mi_palette[base..base + x_mis as usize].fill(cell);
        }
    }

    /// `av1_set_entropy_contexts` (av1/common/blockd.c): fill the txb's
    /// above/left context footprint (of `plane`, from the plane's context base
    /// indices) with its cul level, zeroing the beyond-frame part when a
    /// non-zero fill crosses the frame edge. `blocks_wide`/`blocks_high` are
    /// the plane block's in-frame extent (`max_block_wide/high(xd,
    /// plane_bsize, plane)`); the edge distances are the block's luma
    /// eighth-pel values.
    #[allow(clippy::too_many_arguments)]
    fn set_entropy_ctx(
        &mut self,
        plane: usize,
        cul: i8,
        a_base: usize,
        l_base: usize,
        blk_row: usize,
        blk_col: usize,
        txw: usize,
        txh: usize,
        blocks_wide: usize,
        blocks_high: usize,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
    ) {
        let a0 = a_base + blk_col;
        if cul != 0 && mb_to_right_edge < 0 {
            let n = txw.min(blocks_wide.saturating_sub(blk_col));
            self.above_e[plane][a0..a0 + n].fill(cul);
            self.above_e[plane][a0 + n..a0 + txw].fill(0);
        } else {
            self.above_e[plane][a0..a0 + txw].fill(cul);
        }
        let l0 = l_base + blk_row;
        if cul != 0 && mb_to_bottom_edge < 0 {
            let n = txh.min(blocks_high.saturating_sub(blk_row));
            self.left_e[plane][l0..l0 + n].fill(cul);
            self.left_e[plane][l0 + n..l0 + txh].fill(0);
        } else {
            self.left_e[plane][l0..l0 + txh].fill(cul);
        }
    }

    /// One leaf block: `parse_decode_block` (mode info + tx sizing + skip
    /// entropy-reset) followed by the intra `decode_token_recon_block` txb loop.
    fn decode_block(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
    ) {
        let cfg = self.cfg;
        // set_mi_row_col (av1_common_int.h): TILE-relative, not frame-relative
        // — `xd->up_available = (mi_row > tile->mi_row_start)`. A block at a
        // tile's own top/left edge has no available neighbour even when the
        // tile itself sits interior to the frame (tiles code independently).
        let up_available = mi_row > self.tile.mi_row_start;
        let left_available = mi_col > self.tile.mi_col_start;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        // xd->is_chroma_ref (set_mi_row_col): does this block carry the (merged)
        // chroma information? Gates the UV mode-info symbols and the chroma
        // plane loop. The C decode_mbmi_block also rejects >=8x8 blocks whose
        // chroma plane size is invalid ("Invalid block size", e.g. tall shapes
        // in 4:2:2) — asserted here (the roundtrip never produces them).
        let chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y);
        if !cfg.monochrome && bsize >= BLOCK_8X8 && (ss_x != 0 || ss_y != 0) {
            assert_ne!(
                get_plane_block_size(bsize, ss_x, ss_y),
                255,
                "invalid chroma block size (non-conformant partition for this subsampling)"
            );
        }
        let cfl_allowed = !cfg.monochrome && is_cfl_allowed(bsize, false, ss_x, ss_y);
        let (above, left) = self.neighbours(mi_row, mi_col);
        let (above_palette, left_palette) = self.palette_neighbours(mi_row, mi_col);

        // --- decode_mbmi_block: the KEY-frame mode info, every symbol's CDF
        // selected from the frame context by neighbour/block state ---
        self.st.mi_row = mi_row;
        self.st.mi_col = mi_col;
        self.st.bsize = bsize;
        self.st.is_chroma_ref = chroma_ref;
        self.st.cfl_allowed = cfl_allowed;
        self.st.mb_to_top_edge = -(mi_row * 32);
        self.st.has_above = up_available;
        self.st.has_left = left_available;
        // av1_allow_palette (blockd.h): allow_screen_content_tools AND the block's
        // OWN bsize <= MAX_PALETTE_BLOCK_WIDTH/HEIGHT (64x64) AND >= BLOCK_8X8.
        self.st.allow_palette = av1_allow_palette(cfg.allow_screen_content_tools, bsize);
        // read_intra_segment_id's spatial prediction (av1_get_spatial_seg_pred,
        // step 1): the up-left/up/left cells of the CURRENT frame's segment-id
        // map — always positions already decoded and stamped by this walk.
        if cfg.seg.enabled {
            let cols = cfg.mi_cols;
            let cell = |r: i32, c: i32| self.seg_map[(r * cols + c) as usize];
            let (pred, cdf_num) = spatial_seg_pred(
                (up_available && left_available).then(|| cell(mi_row - 1, mi_col - 1)),
                up_available.then(|| cell(mi_row - 1, mi_col)),
                left_available.then(|| cell(mi_row, mi_col - 1)),
            );
            self.st.seg_pred = pred;
            self.st.seg_cdf_num = cdf_num;
        }
        let mut info = read_mb_modes_kf_fc(
            dec,
            cdfs,
            &mut self.st,
            cfg.enable_filter_intra,
            above,
            left,
            above_palette,
            left_palette,
        );
        // Intra block copy: read_mb_modes_kf_fc has already read use_intrabc and,
        // when set, the RAW block-vector difference into info.dv_row/dv_col.
        // Resolve the final DV here — the predictor scan (av1_find_mv_refs at
        // INTRA_FRAME + av1_find_best_ref_mvs, then av1_find_ref_dv / assign_dv)
        // reads the decoded-so-far mi grid, which the driver owns (not the
        // entropy layer). The C bitstream order is preserved: the use_intrabc
        // symbol and the mv-diff were read in read_intrabc_info; the predictor
        // derivation here reads no bits.
        if info.use_intrabc != 0 {
            let dv_tile = DvTileBounds {
                mi_row_start: self.tile.mi_row_start,
                mi_row_end: self.tile.mi_row_end,
                mi_col_start: self.tile.mi_col_start,
                mi_col_end: self.tile.mi_col_end,
            };
            let mib_size = self.st.mib_size;
            let grid = MiDvGrid {
                mi_dv: &self.mi_dv,
                cols: cfg.mi_cols,
                rows: cfg.mi_rows,
                mi_row,
                mi_col,
            };
            let (nearest_r, nearest_c, near_r, near_c) = find_dv_ref_mvs(
                mi_row,
                mi_col,
                bsize,
                partition,
                up_available,
                left_available,
                dv_tile,
                cfg.mi_rows,
                cfg.mi_cols,
                mib_size,
                grid,
            );
            let (dv_row, dv_col) = assign_and_validate_dv(
                (nearest_r, nearest_c),
                (near_r, near_c),
                info.dv_row,
                info.dv_col,
                self.tile.mi_row_start,
                mib_size,
                mi_row,
                mi_col,
                bsize,
                dv_tile,
                mib_size.trailing_zeros() as i32,
                chroma_ref,
                self.st.num_planes,
                ss_x as i32,
                ss_y as i32,
            )
            .expect("intrabc DV failed validity (non-conformant stream)");
            info.dv_row = dv_row;
            info.dv_col = dv_col;
        }
        // Stamp this block's palette facts over its footprint — matches stamp_mi's
        // placement, right after the mode-info read (subsequent blocks' above/left
        // palette-cache lookups must see it).
        self.stamp_palette(
            mi_row,
            mi_col,
            bsize,
            PaletteNbrKf {
                size: info.palette_size,
                colors: info.palette_colors,
            },
        );
        // set_segment_id (read_intra_segment_id, decodemv.c): stamp the
        // block's resolved id over its frame-cropped mi footprint. The C
        // stamps between the segment read and the rest of the mode info;
        // nothing in between reads the map, so stamping here is equivalent.
        if cfg.seg.enabled {
            let x_mis = MI_SIZE_WIDE[bsize].min(cfg.mi_cols - mi_col);
            let y_mis = MI_SIZE_HIGH[bsize].min(cfg.mi_rows - mi_row);
            for r in 0..y_mis {
                let base = ((mi_row + r) * cfg.mi_cols + mi_col) as usize;
                self.seg_map[base..base + x_mis as usize].fill(info.segment_id as u8);
            }
        }

        // av1_visit_palette(..., av1_decode_palette_tokens) (decodeframe.c): the
        // colour-index MAP tokens — a SEPARATE step from the mode-info flags/size/
        // colours just read above (needs av1_get_block_dimensions' block-geometry
        // inputs, which the mode-info driver doesn't carry). Y decodes iff
        // palette_size[0]>0; chroma (ONE shared map, indexed by BOTH U and V during
        // reconstruction) iff palette_size[1]>0 — gated on `plane==0 || is_chroma_ref`
        // like av1_visit_palette itself (a non-chroma-reference block can never reach
        // here with palette_size[1]>0, since read_mb_modes_kf_fc's uv_dc_pred gate
        // already requires is_chroma_ref).
        let pal_mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let pal_mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let mut color_map_y: Vec<u8> = Vec::new();
        let mut color_map_uv: Vec<u8> = Vec::new();
        let mut uv_map_wpx = 0usize;
        if info.palette_size[0] > 0 {
            let (wpx, hpx, rows, cols) = get_block_dimensions(
                bsize,
                0,
                ss_x,
                ss_y,
                pal_mb_to_right_edge,
                pal_mb_to_bottom_edge,
            );
            // palette_{y,uv}_color_index_cdf[n - PALETTE_MIN_SIZE] (PALETTE_MIN_SIZE=2).
            let n = info.palette_size[0];
            color_map_y = decode_color_map_tokens(
                dec,
                n,
                wpx,
                hpx,
                rows,
                cols,
                &mut cdfs.palette_y_color_index[(n - 2) as usize],
            );
        }
        if info.palette_size[1] > 0 {
            let (wpx, hpx, rows, cols) = get_block_dimensions(
                bsize,
                1,
                ss_x,
                ss_y,
                pal_mb_to_right_edge,
                pal_mb_to_bottom_edge,
            );
            let n = info.palette_size[1];
            color_map_uv = decode_color_map_tokens(
                dec,
                n,
                wpx,
                hpx,
                rows,
                cols,
                &mut cdfs.palette_uv_color_index[(n - 2) as usize],
            );
            uv_map_wpx = wpx;
        }

        // --- parse_decode_block: the block's transform size (read_tx_size) +
        // txfm-context stamp, in the C statement order (after the mode info /
        // palette tokens, before the skip entropy reset) ---
        let bw = MI_SIZE_WIDE[bsize] as usize;
        let bh = MI_SIZE_HIGH[bsize] as usize;
        // read_tx_size (decodeframe.c), intra: the xd->lossless[segment_id]
        // TX_4X4 preemption is off in this scope (coded_lossless rejected;
        // lossless SEGMENTS asserted away in TileKf::new); a signalling block
        // (bsize > BLOCK_4X4) under TX_MODE_SELECT codes its tx-size depth —
        // intra blocks code it even when skip_txfm is set (`!is_inter ||
        // allow_select_inter` is true for intra) — else the tx_mode fallback.
        let a_off = mi_col as usize;
        let l_off = (mi_row & 31) as usize;
        let tx_size = if info.use_intrabc != 0 {
            // Intrabc is `is_inter_block`, so `inter_block_tx` is set and the tx
            // size follows the INTER path (decodeframe.c:1179-1198):
            //  - TX_MODE_SELECT && block_signals_txsize && !skip  → the var-tx
            //    quadtree (`read_tx_size_vartx`, `txfm_partition_cdf` — NOT the
            //    intra `tx_size_cdf`). It stamps the txfm-context arrays itself
            //    via `txfm_partition_update`, so no `set_txfm_ctxs` here.
            //  - otherwise → `read_tx_size(is_inter=true, allow_select=!skip)`:
            //    with `is_inter` the `read_selected` arm needs
            //    `allow_select && select && signalling`, which is exactly the
            //    var-tx case above, so this always resolves to the fallback
            //    (`tx_size_from_tx_mode` when signalling, else max rect) — no
            //    tx-size symbol — then `set_txfm_ctxs(skip && is_inter = skip)`.
            if cfg.tx_mode == TxMode::Select && bsize > 0 && info.skip == 0 {
                let max_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
                let bw_u = TX_SIZE_WIDE_UNIT[max_tx] as i32;
                let bh_u = TX_SIZE_HIGH_UNIT[max_tx] as i32;
                let width_u = MI_SIZE_WIDE[bsize];
                let height_u = MI_SIZE_HIGH[bsize];
                let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
                let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
                let mut vartx_tx = max_tx;
                let mut first_leaf = -1i32;
                let mut non_uniform = false;
                let mut idy = 0;
                while idy < height_u {
                    let mut idx = 0;
                    while idx < width_u {
                        read_tx_size_vartx(
                            dec,
                            &mut self.txfm_partition,
                            &mut self.above_t[a_off..],
                            &mut self.left_t[l_off..],
                            bsize,
                            mb_to_right_edge,
                            mb_to_bottom_edge,
                            max_tx,
                            0,
                            idy,
                            idx,
                            &mut vartx_tx,
                            &mut first_leaf,
                            &mut non_uniform,
                        );
                        idx += bw_u;
                    }
                    idy += bh_u;
                }
                // The coeff loop tiles the whole block with this one scalar
                // tx_size, exact only for a UNIFORM var-tx partition. A
                // non-uniform intrabc partition (possible only for blocks >2 tx
                // units in a dim) would need the per-txb inter_tx_size[] walk;
                // reject rather than silently mis-tile the coefficients. The
                // KEY-frame intrabc screen-content streams verified here only
                // produce uniform partitions, so this never fires on them.
                assert!(
                    !non_uniform,
                    "non-uniform intrabc var-tx partition not supported \
                     (mi=({mi_row},{mi_col}) bsize={bsize}); would desync the coeff walk"
                );
                vartx_tx
            } else {
                let ts = if bsize > 0 {
                    tx_size_from_tx_mode(bsize, cfg.tx_mode)
                } else {
                    MAX_TXSIZE_RECT_LOOKUP[bsize]
                };
                set_txfm_ctxs(
                    &mut self.above_t[a_off..],
                    &mut self.left_t[l_off..],
                    ts,
                    bw,
                    bh,
                    info.skip != 0,
                );
                ts
            }
        } else if bsize > 0 {
            // Intra, block_signals_txsize: read_tx_size(is_inter=false,
            // allow_select=!skip). `!is_inter` makes `(!is_inter || …)` true, so a
            // signalling block under TX_MODE_SELECT codes its tx-size depth
            // (intra codes it even when skip_txfm is set) — else the tx_mode
            // fallback. set_txfm_ctxs' skip arg is `skip && is_inter` = 0.
            let tx_size = if cfg.tx_mode == TxMode::Select {
                let cat = bsize_to_tx_size_cat(bsize) as usize;
                // get_tx_size_context reads is_inter_block(above/left_mbmi); on a
                // KEY frame that is true only for an intrabc neighbour, whose
                // block_size_wide/high then drives the context (else None).
                let above_inter_bsize = up_available
                    .then(|| self.mi_dv[((mi_row - 1) * cfg.mi_cols + mi_col) as usize])
                    .filter(|d| d.use_intrabc)
                    .map(|d| d.bsize);
                let left_inter_bsize = left_available
                    .then(|| self.mi_dv[(mi_row * cfg.mi_cols + mi_col - 1) as usize])
                    .filter(|d| d.use_intrabc)
                    .map(|d| d.bsize);
                let ctx = get_tx_size_context(
                    bsize,
                    self.above_t[a_off],
                    self.left_t[l_off],
                    up_available,
                    left_available,
                    above_inter_bsize,
                    left_inter_bsize,
                );
                let depth = read_selected_tx_size(
                    dec,
                    &mut cdfs.tx_size[cat][ctx],
                    bsize,
                    bsize_to_max_depth(bsize),
                );
                depth_to_tx_size(depth, bsize)
            } else {
                tx_size_from_tx_mode(bsize, cfg.tx_mode)
            };
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                tx_size,
                bw,
                bh,
                false,
            );
            tx_size
        } else {
            // Intra, non-signalling (BLOCK_4X4): max rect (TX_4X4), no symbol.
            let tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                tx_size,
                bw,
                bh,
                false,
            );
            tx_size
        };

        // parse_decode_block (decodeframe.c): with delta-q present, every
        // block's dequant is recomputed from the running current_base_qindex
        // (already advanced by this block's SB-level delta read inside the
        // mode-info decode — mbmi->current_qindex == the carry); the C
        // refills all MAX_SEGMENTS seg_dequant_QTX rows via
        // av1_get_qindex(seg, i, carry) and the txb read consumes row
        // [mbmi->segment_id] — computed here directly for the block's
        // segment. Without delta-q the frame-level rows come from
        // setup_segmentation_dequant (xd->qindex[i] = av1_get_qindex(seg, i,
        // base_qindex)) — the same formula on the never-moved carry. The
        // per-plane dc/ac deltas fold in through av1_{dc,ac}_quant_QTX.
        if cfg.delta_q_present || cfg.seg.enabled {
            debug_assert!(
                !cfg.delta_q_present || self.st.current_base_qindex == info.current_qindex
            );
            let eff_qindex = av1_get_qindex(
                &cfg.seg,
                info.segment_id as usize,
                self.st.current_base_qindex,
            );
            self.dequants = plane_dequants(cfg, eff_qindex);
        }

        // The chroma-side geometry (used by the skip reset, the chroma txb
        // loop, and CfL): the plane origin is the shared-chroma group's — a
        // sub-8x8 dimension at an odd mi position on a subsampled axis shifts
        // back one mi (setup_pred_plane / set_entropy_context adjustment).
        let adj_row = if ss_y != 0 && (mi_row & 1) != 0 && MI_SIZE_HIGH[bsize] == 1 {
            mi_row - 1
        } else {
            mi_row
        };
        let adj_col = if ss_x != 0 && (mi_col & 1) != 0 && MI_SIZE_WIDE[bsize] == 1 {
            mi_col - 1
        } else {
            mi_col
        };
        let uv_a_base = (adj_col >> ss_x) as usize;
        let uv_l_base = ((adj_row & 31) >> ss_y) as usize;

        // --- parse_decode_block tail: skip blocks reset their entropy context
        // (av1_reset_entropy_context: plane 0 always; chroma planes when this
        // block is the chroma reference, over the chroma plane-bsize footprint
        // from the adjusted context bases) ---
        if info.skip != 0 {
            let a0 = mi_col as usize;
            self.above_e[0][a0..a0 + bw].fill(0);
            let l0 = (mi_row & 31) as usize;
            self.left_e[0][l0..l0 + bh].fill(0);
            if !cfg.monochrome && chroma_ref {
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let (uw, uh) = (
                    MI_SIZE_WIDE[plane_bsize] as usize,
                    MI_SIZE_HIGH[plane_bsize] as usize,
                );
                for plane in 1..=2 {
                    self.above_e[plane][uv_a_base..uv_a_base + uw].fill(0);
                    self.left_e[plane][uv_l_base..uv_l_base + uh].fill(0);
                }
            }
        }

        // --- decode_token_recon_block (intra): per-txb read -> predict -> recon ---
        let (txw, txh) = (TX_SIZE_WIDE_UNIT[tx_size], TX_SIZE_HIGH_UNIT[tx_size]);
        let (txwpx, txhpx) = (TX_SIZE_WIDE[tx_size], TX_SIZE_HIGH[tx_size]);
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let max_blocks_wide = max_block_units(BLOCK_SIZE_WIDE[bsize], mb_to_right_edge);
        let max_blocks_high = max_block_units(BLOCK_SIZE_HIGH[bsize], mb_to_bottom_edge);
        // get_filt_type (reconintra.c), luma: 1 when the above or left neighbour
        // block (the same xd->above_mbmi/left_mbmi the mode contexts read) has a
        // smooth y mode.
        let is_smooth = |m: Option<MiNbrKf>| {
            m.is_some_and(|n| (SMOOTH_PRED..=SMOOTH_H_PRED).contains(&n.y_mode))
        };
        let filt_type = (is_smooth(above) || is_smooth(left)) as i32;
        // av1_read_tx_type gate: !skip_txfm && !seg-SKIP && qindex > 0 —
        // xd->qindex[segment_id] = av1_get_qindex over the FRAME base qindex
        // (decodeframe.c:5165, NOT the delta-q carry). skip == 0 already
        // implies the segment's SEG_LVL_SKIP feature is inactive (read_skip
        // returns a forced 1 when it is active).
        let signal_gate = info.skip == 0
            && av1_get_qindex(&cfg.seg, info.segment_id as usize, cfg.base_qindex) > 0;
        let area = txb_wide(tx_size) * txb_high(tx_size);
        let mut tcoeff = vec![0i32; area];
        let mut scratch = vec![0u16; txwpx * txhpx];
        let mut txbs = Vec::new();

        let mut blk_row = 0usize;
        while blk_row < max_blocks_high {
            let mut blk_col = 0usize;
            while blk_col < max_blocks_wide {
                // (1) coefficients — read_coeffs_tx_intra_block (skipped blocks
                // code nothing; their contexts stay at the reset zeros).
                let (eob, tx_type) = if info.skip == 0 {
                    let a0 = mi_col as usize + blk_col;
                    let l0 = (mi_row & 31) as usize + blk_row;
                    let (tsc, dsc) = get_txb_ctx(
                        bsize,
                        tx_size,
                        0,
                        &self.above_e[0][a0..],
                        &self.left_e[0][l0..],
                    );
                    // Intrabc is is_inter_block, so av1_read_tx_type selects the
                    // tx-type CDF from inter_ext_tx_cdf (and maps the symbol with
                    // the inter set type); a normal intra block uses the intra
                    // ext-tx sets keyed on (square tx size, intra direction).
                    let ext = if info.use_intrabc != 0 {
                        inter_ext_tx_cdf(&mut cdfs.inter_ext_tx, tx_size, cfg.reduced_tx_set)
                    } else {
                        intra_ext_tx_cdf(
                            &mut cdfs.ext_tx_1ddct,
                            &mut cdfs.ext_tx_dtt4,
                            tx_size,
                            cfg.reduced_tx_set,
                            info.use_filter_intra != 0,
                            info.filter_intra_mode as usize,
                            info.y_mode as usize,
                        )
                    };
                    let (eob, tt) = read_coeffs_txb_full(
                        dec,
                        &mut cdfs.coeff,
                        ext,
                        &mut tcoeff,
                        tx_size,
                        0,
                        tsc as usize,
                        dsc as usize,
                        true,
                        info.use_intrabc != 0,
                        cfg.reduced_tx_set,
                        signal_gate,
                        0,
                    );
                    let cul = txb_entropy_context(&tcoeff, tx_size, tt, eob) as i8;
                    self.set_entropy_ctx(
                        0,
                        cul,
                        mi_col as usize,
                        (mi_row & 31) as usize,
                        blk_row,
                        blk_col,
                        txw,
                        txh,
                        max_blocks_wide,
                        max_blocks_high,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                    );
                    (eob, tt)
                } else {
                    (0, 0)
                };

                // Record the luma tx-type in `cm->tx_type_map` (mi granularity)
                // exactly as C's `update_txk_array`: the txb's TOP-LEFT mi cell
                // only; a 64-level transform (a 64px side => 16 mi units)
                // additionally stamps every 16x16 (4-mi) unit of its footprint.
                // Cells left unwritten stay DCT_DCT (the zeroed map). Colour
                // intrabc chroma reads the co-located luma tx-type from here;
                // empty (skipped) on monochrome / non-intrabc frames. Skip blocks
                // stamp DCT_DCT (0), matching C.
                if !self.luma_tt.is_empty() {
                    let cols = cfg.mi_cols as usize;
                    let r0 = mi_row as usize + blk_row;
                    let c0 = mi_col as usize + blk_col;
                    self.luma_tt[r0 * cols + c0] = tx_type as u8;
                    if txw == 16 || txh == 16 {
                        let rmax = txh.min(max_blocks_high - blk_row);
                        let cmax = txw.min(max_blocks_wide - blk_col);
                        let mut idy = 0;
                        while idy < rmax {
                            let mut idx = 0;
                            while idx < cmax {
                                self.luma_tt[(r0 + idy) * cols + c0 + idx] = tx_type as u8;
                                idx += 4;
                            }
                            idy += 4;
                        }
                    }
                }

                // (2) intra prediction into the reconstruction plane.
                let (n_top, n_tr, n_left, n_bl) = intra_avail(
                    self.st.sb_size,
                    bsize,
                    mi_row,
                    mi_col,
                    up_available,
                    left_available,
                    self.tile.mi_col_end,
                    self.tile.mi_row_end,
                    partition,
                    tx_size,
                    0,
                    0,
                    blk_row as i32,
                    blk_col as i32,
                    BLOCK_SIZE_WIDE[bsize],
                    BLOCK_SIZE_HIGH[bsize],
                    cfg.mi_cols,
                    cfg.mi_rows,
                    info.y_mode as usize,
                    info.angle_delta_y * ANGLE_STEP,
                    info.use_filter_intra != 0,
                );
                let off = ((mi_row * 4) as usize + blk_row * 4) * self.stride
                    + (mi_col * 4) as usize
                    + blk_col * 4;
                if info.use_intrabc != 0 {
                    // Intra block copy, luma: an integer block copy from the
                    // DV-referenced region of the SAME reconstruction plane. The
                    // DV is read at MV_SUBPEL_NONE (full-pel) and validated by
                    // av1_is_dv_valid to reference only already-decoded pixels, so
                    // the source (off shifted by dv/8) is always reconstructed and
                    // never overlaps this block's pending tx units. Luma needs no
                    // interpolation (av1_dc_128... the intrabc convolve collapses
                    // to a copy at integer positions).
                    let src = (off as i32
                        + (info.dv_row >> 3) * self.stride as i32
                        + (info.dv_col >> 3)) as usize;
                    for r in 0..txhpx {
                        let s = src + r * self.stride;
                        scratch[r * txwpx..(r + 1) * txwpx]
                            .copy_from_slice(&self.recon[s..s + txwpx]);
                    }
                } else if info.palette_size[0] > 0 {
                    // av1_predict_intra_block's palette branch (reconintra.c): pixels
                    // come directly from the colour-index map + palette LUT — no
                    // directional/DC prediction math (the surrounding residual
                    // add below is unaffected: palette replaces PREDICTION only).
                    // The map covers the whole coding block (BLOCK_SIZE_WIDE[bsize]
                    // stride); this tx block reads its (blk_col*4, blk_row*4)
                    // pixel sub-rectangle.
                    let map_w = BLOCK_SIZE_WIDE[bsize] as usize;
                    let (x0, y0) = (blk_col * 4, blk_row * 4);
                    for r in 0..txhpx {
                        for c in 0..txwpx {
                            let idx = color_map_y[(y0 + r) * map_w + x0 + c] as usize;
                            scratch[r * txwpx + c] = info.palette_colors[idx];
                        }
                    }
                } else {
                    predict_intra_high(
                        &self.recon,
                        off,
                        self.stride,
                        &mut scratch,
                        txwpx,
                        info.y_mode as usize,
                        info.angle_delta_y * ANGLE_STEP,
                        info.use_filter_intra != 0,
                        info.filter_intra_mode as usize,
                        cfg.disable_edge_filter,
                        filt_type,
                        tx_size,
                        usize::try_from(n_top).expect("n_top_px must be non-negative"),
                        n_tr,
                        usize::try_from(n_left).expect("n_left_px must be non-negative"),
                        n_bl,
                        cfg.bd,
                    );
                }
                for r in 0..txhpx {
                    let d = off + r * self.stride;
                    self.recon[d..d + txwpx].copy_from_slice(&scratch[r * txwpx..(r + 1) * txwpx]);
                }

                // (3) dequant + inverse transform + add (only when residual
                // exists) — the block-effective luma dequant row.
                if info.skip == 0 && eob > 0 {
                    reconstruct_txb(
                        &mut self.recon[off..],
                        self.stride,
                        tx_size,
                        tx_type,
                        &tcoeff,
                        self.dequants[0],
                        None,
                        cfg.bd,
                    );
                }
                // (4) CfL luma store (predict_and_reconstruct_intra_block tail,
                // store_cfl_required): non-chroma-reference blocks always store
                // (a later group member may pick CfL); the chroma-reference
                // block stores only when it actually uses CfL. Runs for skip
                // blocks too (their reconstruction is the prediction).
                if !cfg.monochrome && (!chroma_ref || info.uv_mode == UV_CFL_PRED) {
                    let block_off = (mi_row * 4) as usize * self.stride + (mi_col * 4) as usize;
                    cfl_store_tx(
                        &mut self.cfl,
                        &self.recon,
                        block_off,
                        self.stride,
                        blk_row as i32,
                        blk_col as i32,
                        tx_size,
                        bsize,
                        mi_row,
                        mi_col,
                    );
                }
                txbs.push((eob, tx_type));
                blk_col += txw;
            }
            blk_row += txh;
        }

        // --- decode_token_recon_block, planes 1..2: the chroma txb loop of the
        // (single, <=64x64) 64x64 chunk — runs after ALL of plane 0, so the
        // block's own luma is already in the CfL store. Only the
        // chroma-reference block of a shared group decodes chroma, covering
        // the merged area from the adjusted plane origin. ---
        let mut txbs_uv = Vec::new();
        if !cfg.monochrome && chroma_ref {
            let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
            assert_ne!(plane_bsize, 255, "invalid chroma block size");
            let uv_tx = max_uv_txsize(bsize, ss_x, ss_y);
            let (uv_txw, uv_txh) = (TX_SIZE_WIDE_UNIT[uv_tx], TX_SIZE_HIGH_UNIT[uv_tx]);
            let (uv_txwpx, uv_txhpx) = (TX_SIZE_WIDE[uv_tx], TX_SIZE_HIGH[uv_tx]);
            // unit_width/height: the luma extent, ceil-scaled to chroma units.
            let unit_width = round_power_of_two(max_blocks_wide as i32, ss_x) as usize;
            let unit_height = round_power_of_two(max_blocks_high as i32, ss_y) as usize;
            // av1_set_entropy_contexts' frame-edge clip uses the CHROMA plane
            // block's in-frame extent.
            let blocks_wide_uv =
                max_block_units_ss(BLOCK_SIZE_WIDE[plane_bsize], mb_to_right_edge, ss_x);
            let blocks_high_uv =
                max_block_units_ss(BLOCK_SIZE_HIGH[plane_bsize], mb_to_bottom_edge, ss_y);
            // Prediction geometry: pd->width/height (chroma px, min 4), the
            // scaled block size the has_top_right/bottom_left walk sees, and
            // the chroma availability (equal to group-origin availability).
            let wpx = ((MI_SIZE_WIDE[bsize] * 4) >> ss_x).max(4);
            let hpx = ((MI_SIZE_HIGH[bsize] * 4) >> ss_y).max(4);
            let bsize_uv = scale_chroma_bsize(bsize, ss_x, ss_y);
            // set_mi_row_col's chroma_up_available/chroma_left_available:
            // equal to the luma up_available/left_available EXCEPT the
            // sub-8x8-odd-position group case, where it's `(mi_row/col - 1) >
            // tile->mi_row/col_start` — exactly `adj_row/col >
            // tile.mi_row/col_start` in both cases (adj_row/adj_col already
            // encode the "-1 when sub-8x8 odd position" shift above).
            let up_uv = adj_row > self.tile.mi_row_start;
            let left_uv = adj_col > self.tile.mi_col_start;
            // get_filt_type(xd, plane > 0): smoothness of the chroma
            // above/left neighbours — the bottom-right-most mi of the
            // neighbouring chroma region (set_mi_row_col's chroma_above_mbmi /
            // chroma_left_mbmi), read from the uv-mode grid.
            let cols = cfg.mi_cols;
            let base_row = mi_row - (mi_row & ss_y as i32);
            let base_col = mi_col - (mi_col & ss_x as i32);
            let uv_smooth = |m: i8| (9..=11).contains(&m);
            let ab_sm = up_uv
                && uv_smooth(self.mi_uv[((base_row - 1) * cols + base_col + ss_x as i32) as usize]);
            let le_sm = left_uv
                && uv_smooth(self.mi_uv[((base_row + ss_y as i32) * cols + base_col - 1) as usize]);
            let filt_type_uv = (ab_sm || le_sm) as i32;
            // The block origin in the chroma planes.
            let uv_org = ((adj_row * 4) >> ss_y) as usize * self.stride_uv
                + ((adj_col * 4) >> ss_x) as usize;
            // Chroma transform types are not coded: the UV intra mode implies
            // the type, demoted to DCT_DCT outside the block's ext-tx set
            // (av1_get_tx_type, PLANE_TYPE_UV intra).
            let tt_uv = uv_tx_type(info.uv_mode, uv_tx, cfg.reduced_tx_set);
            let mode_uv = get_uv_mode(info.uv_mode as usize) as usize;
            let uv_area = txb_wide(uv_tx) * txb_high(uv_tx);
            let mut tcoeff_uv = vec![0i32; uv_area];
            let mut scratch_uv = vec![0u16; uv_txwpx * uv_txhpx];
            let mut no_ext: [u16; 0] = [];

            for plane in 1..=2usize {
                let mut blk_row = 0usize;
                while blk_row < unit_height {
                    let mut blk_col = 0usize;
                    while blk_col < unit_width {
                        // Chroma tx-type: for intrabc (is_inter) it is the
                        // CO-LOCATED luma tx-type (av1_get_tx_type inter branch),
                        // read from the luma tx_type_map at (mi_row+(blk_row<<ss_y),
                        // mi_col+(blk_col<<ss_x)) — the block's OWN mi origin, not
                        // the shared-group base — then re-validated against the
                        // inter ext-tx set for uv_tx and demoted to DCT_DCT if
                        // unused. Ordinary intra chroma uses the block-level
                        // uv_tx_type computed above.
                        let tt_uv_eff = if info.use_intrabc != 0 {
                            let lr = mi_row as usize + (blk_row << ss_y);
                            let lc = mi_col as usize + (blk_col << ss_x);
                            let luma_tt = self.luma_tt[lr * cfg.mi_cols as usize + lc] as usize;
                            if ext_tx_derive(uv_tx, true, cfg.reduced_tx_set, luma_tt, false, 0, 0)
                                .used
                                == 1
                            {
                                luma_tt
                            } else {
                                0
                            }
                        } else {
                            tt_uv
                        };
                        // (1) chroma coefficients (read_coeffs_tx_intra_block).
                        let eob = if info.skip == 0 {
                            let (tsc, dsc) = get_txb_ctx(
                                plane_bsize,
                                uv_tx,
                                plane,
                                &self.above_e[plane][uv_a_base + blk_col..],
                                &self.left_e[plane][uv_l_base + blk_row..],
                            );
                            let (eob, _tt) = read_coeffs_txb_full(
                                dec,
                                &mut cdfs.coeff,
                                &mut no_ext, // plane_type 1: no tx_type symbol
                                &mut tcoeff_uv,
                                uv_tx,
                                1,
                                tsc as usize,
                                dsc as usize,
                                true,
                                false,
                                cfg.reduced_tx_set,
                                false,
                                tt_uv_eff,
                            );
                            let cul = txb_entropy_context(&tcoeff_uv, uv_tx, tt_uv_eff, eob) as i8;
                            self.set_entropy_ctx(
                                plane,
                                cul,
                                uv_a_base,
                                uv_l_base,
                                blk_row,
                                blk_col,
                                uv_txw,
                                uv_txh,
                                blocks_wide_uv,
                                blocks_high_uv,
                                mb_to_right_edge,
                                mb_to_bottom_edge,
                            );
                            eob
                        } else {
                            0
                        };

                        // (2) chroma intra prediction (av1_predict_intra_block_facade):
                        // ordinary intra with mode = get_uv_mode(uv_mode) — DC for
                        // CfL — then the CfL AC contribution on top.
                        let (n_top, n_tr, n_left, n_bl) = intra_avail(
                            self.st.sb_size,
                            bsize_uv,
                            adj_row,
                            adj_col,
                            up_uv,
                            left_uv,
                            self.tile.mi_col_end,
                            self.tile.mi_row_end,
                            partition,
                            uv_tx,
                            ss_x as i32,
                            ss_y as i32,
                            blk_row as i32,
                            blk_col as i32,
                            wpx,
                            hpx,
                            cfg.mi_cols,
                            cfg.mi_rows,
                            mode_uv,
                            info.angle_delta_uv * ANGLE_STEP,
                            false,
                        );
                        let off_uv = uv_org + (blk_row * 4) * self.stride_uv + blk_col * 4;
                        if info.use_intrabc != 0 {
                            // Intra block copy, chroma: reuse the luma DV, scaled
                            // by subsampling. mv_q4 = dv << (1-ss) is in 1/16
                            // chroma-pel; the integer chroma-pixel offset is
                            // mv_q4>>4 and the 2-tap intrabc bilinear fires when
                            // mv_q4&15 == 8 (only when the integer-luma-pel DV is
                            // odd on a subsampled axis — 4:4:4 is always a copy).
                            // Source is this block's own chroma recon plane, which
                            // DV validity keeps already-decoded.
                            let mvq4_row = info.dv_row << (1 - ss_y as i32);
                            let mvq4_col = info.dv_col << (1 - ss_x as i32);
                            let src = (off_uv as isize
                                + (mvq4_row >> 4) as isize * self.stride_uv as isize
                                + (mvq4_col >> 4) as isize)
                                as usize;
                            let plane_recon = if plane == 1 {
                                &self.recon_u
                            } else {
                                &self.recon_v
                            };
                            intrabc_chroma_predict(
                                plane_recon,
                                src,
                                self.stride_uv,
                                &mut scratch_uv,
                                uv_txwpx,
                                uv_txwpx,
                                uv_txhpx,
                                mvq4_col & 15,
                                mvq4_row & 15,
                                cfg.bd,
                            );
                        } else if info.palette_size[1] > 0 {
                            // av1_predict_intra_block's palette branch, chroma: ONE
                            // shared colour-index map for U and V (uv_map_wpx-strided,
                            // from av1_get_block_dimensions(bsize, plane=1, ...)),
                            // looked up against palette_colors[plane * PALETTE_MAX_SIZE]
                            // (plane 1 = U, plane 2 = V — the palette_colors offset
                            // matches this loop's own `plane` var directly).
                            let (x0, y0) = (blk_col * 4, blk_row * 4);
                            let pal_base = plane * 8;
                            for r in 0..uv_txhpx {
                                for c in 0..uv_txwpx {
                                    let idx = color_map_uv[(y0 + r) * uv_map_wpx + x0 + c] as usize;
                                    scratch_uv[r * uv_txwpx + c] =
                                        info.palette_colors[pal_base + idx];
                                }
                            }
                        } else {
                            let plane_recon = if plane == 1 {
                                &self.recon_u
                            } else {
                                &self.recon_v
                            };
                            predict_intra_high(
                                plane_recon,
                                off_uv,
                                self.stride_uv,
                                &mut scratch_uv,
                                uv_txwpx,
                                mode_uv,
                                info.angle_delta_uv * ANGLE_STEP,
                                false,
                                0,
                                cfg.disable_edge_filter,
                                filt_type_uv,
                                uv_tx,
                                usize::try_from(n_top).expect("n_top_px must be non-negative"),
                                n_tr,
                                usize::try_from(n_left).expect("n_left_px must be non-negative"),
                                n_bl,
                                cfg.bd,
                            );
                        }
                        if info.uv_mode == UV_CFL_PRED && info.use_intrabc == 0 {
                            cfl_predict_block(
                                &mut self.cfl,
                                &mut scratch_uv,
                                0,
                                uv_txwpx,
                                uv_tx,
                                plane,
                                info.cfl_alpha_idx,
                                info.cfl_joint_sign,
                                cfg.bd,
                            );
                        }
                        {
                            let plane_recon = if plane == 1 {
                                &mut self.recon_u
                            } else {
                                &mut self.recon_v
                            };
                            for r in 0..uv_txhpx {
                                let d = off_uv + r * self.stride_uv;
                                plane_recon[d..d + uv_txwpx]
                                    .copy_from_slice(&scratch_uv[r * uv_txwpx..(r + 1) * uv_txwpx]);
                            }
                            // (3) dequant + inverse transform + add — the
                            // block-effective dequant row of this plane.
                            if info.skip == 0 && eob > 0 {
                                reconstruct_txb(
                                    &mut plane_recon[off_uv..],
                                    self.stride_uv,
                                    uv_tx,
                                    tt_uv_eff,
                                    &tcoeff_uv,
                                    self.dequants[plane],
                                    None,
                                    cfg.bd,
                                );
                            }
                        }
                        txbs_uv.push(if eob > 0 { (eob, tt_uv_eff) } else { (0, 0) });
                        blk_col += uv_txw;
                    }
                    blk_row += uv_txh;
                }
            }
        }

        self.stamp_mi(
            mi_row,
            mi_col,
            bsize,
            MiNbrKf {
                y_mode: info.y_mode,
                skip_txfm: info.skip,
            },
        );
        // Block-vector grid stamp (intrabc projection of xd->mi): on a KEY frame
        // ref_frame[0] is always INTRA_FRAME and ref_frame[1] NONE_FRAME; only
        // use_intrabc + the block's own DV (mv[0]) and bsize are consulted by the
        // next block's av1_find_mv_refs / get_tx_size_context. `mode` is read only
        // by is_global_mv_block (never a match on KEY frames), so it stays 0.
        self.stamp_dv(
            mi_row,
            mi_col,
            bsize,
            DvNbr {
                bsize,
                ref_frame0: 0,
                ref_frame1: -1,
                use_intrabc: info.use_intrabc != 0,
                mode: 0,
                mv0_row: info.dv_row,
                mv0_col: info.dv_col,
                mv1_row: 0,
                mv1_col: 0,
            },
        );
        // The uv-mode grid stamp: non-chroma-reference blocks carry UV_DC_PRED
        // (read_intra_frame_mode_info's else-branch), which the tail returns.
        {
            let x_mis = MI_SIZE_WIDE[bsize].min(cfg.mi_cols - mi_col);
            let y_mis = MI_SIZE_HIGH[bsize].min(cfg.mi_rows - mi_row);
            for r in 0..y_mis {
                let base = ((mi_row + r) * cfg.mi_cols + mi_col) as usize;
                self.mi_uv[base..base + x_mis as usize].fill(info.uv_mode as i8);
            }
        }
        self.blocks.push(DecodedBlockKf {
            mi_row,
            mi_col,
            bsize,
            partition,
            info,
            tx_size,
            txbs,
            txbs_uv,
        });
    }

    /// The per-SB restoration-unit parameter reads
    /// (`loop_restoration_read_sb_coeffs` over the
    /// `av1_loop_restoration_corners_in_sb` rectangle, decodeframe.c:1325):
    /// plane-major, then unit-grid raster order within the SB's rectangle.
    fn read_lr_units(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mi_row: i32,
        mi_col: i32,
    ) {
        use aom_entropy::lr;
        let num_planes = if self.cfg.monochrome { 1 } else { 3 };
        let (ss_x, ss_y) = (self.cfg.subsampling_x, self.cfg.subsampling_y);
        for plane in 0..num_planes {
            let frt = self.cfg.lr.frame_restoration_type[plane];
            if frt == lr::RESTORE_NONE {
                continue;
            }
            let Some((rcol0, rcol1, rrow0, rrow1)) = lr::lr_corners_in_sb(
                &self.cfg.lr,
                plane,
                ss_x,
                ss_y,
                mi_row,
                mi_col,
                self.st.mib_size,
                self.st.mib_size,
            ) else {
                continue;
            };
            let (hu, _) = self.cfg.lr.plane_units(plane, ss_x, ss_y);
            for rrow in rrow0..rrow1 {
                for rcol in rcol0..rcol1 {
                    let unit_idx = (rrow * hu + rcol) as usize;
                    self.lr_units[plane][unit_idx] = lr::read_lr_unit(
                        dec,
                        frt,
                        plane,
                        &mut self.lr_refs,
                        &mut cdfs.switchable_restore,
                        &mut cdfs.wiener_restore,
                        &mut cdfs.sgrproj_restore,
                    );
                }
            }
        }
    }

    /// `decode_partition` (decodeframe.c): the recursive partition walk. Reads
    /// the partition symbol per in-frame node (forced NONE below 8x8; the 2-way
    /// edge gathers and forced SPLIT are inside `read_partition`), dispatches the
    /// leaf blocks in the exact `DEC_BLOCK` order, and stamps the neighbour
    /// partition context.
    fn decode_partition(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
    ) {
        if mi_row >= self.cfg.mi_rows || mi_col >= self.cfg.mi_cols {
            return;
        }
        // decode_partition's parse arm (decodeframe.c:1325-1343): at the
        // superblock root, the parameters of every restoration unit whose
        // top-left corner falls in this SB are coded here, per plane, before
        // the SB's first partition symbol (av1_loop_restoration_corners_in_sb
        // returns 0 for any bsize != cm->seq_params->sb_size — the ACTUAL
        // configured SB root, BLOCK_64X64 or BLOCK_128X128 — so only the
        // root fires).
        if bsize == self.st.sb_size {
            self.read_lr_units(dec, cdfs, mi_row, mi_col);
        }
        let hbs = MI_SIZE_WIDE[bsize] / 2;
        let quarter_step = MI_SIZE_WIDE[bsize] / 4;
        let has_rows = (mi_row + hbs) < self.cfg.mi_rows;
        let has_cols = (mi_col + hbs) < self.cfg.mi_cols;
        let p = if bsize < BLOCK_8X8 {
            PARTITION_NONE
        } else {
            let ctx = partition_plane_context(
                &self.above_p,
                &self.left_p,
                mi_row as usize,
                mi_col as usize,
                bsize,
            ) as usize;
            read_partition(
                dec,
                &mut cdfs.partition[ctx],
                partition_cdf_length(bsize),
                has_rows,
                has_cols,
                bsize,
            ) as usize
        };
        self.tree.push(p as i8);
        let subsize = get_partition_subsize(bsize, p as i32) as usize;
        assert_ne!(subsize, 255, "invalid partition {p} for bsize {bsize}");
        let bsize2 = get_partition_subsize(bsize, PARTITION_SPLIT as i32) as usize;

        match p {
            PARTITION_NONE => self.decode_block(dec, cdfs, mi_row, mi_col, subsize, p),
            PARTITION_HORZ => {
                self.decode_block(dec, cdfs, mi_row, mi_col, subsize, p);
                if has_rows {
                    self.decode_block(dec, cdfs, mi_row + hbs, mi_col, subsize, p);
                }
            }
            PARTITION_VERT => {
                self.decode_block(dec, cdfs, mi_row, mi_col, subsize, p);
                if has_cols {
                    self.decode_block(dec, cdfs, mi_row, mi_col + hbs, subsize, p);
                }
            }
            PARTITION_SPLIT => {
                self.decode_partition(dec, cdfs, mi_row, mi_col, subsize);
                self.decode_partition(dec, cdfs, mi_row, mi_col + hbs, subsize);
                self.decode_partition(dec, cdfs, mi_row + hbs, mi_col, subsize);
                self.decode_partition(dec, cdfs, mi_row + hbs, mi_col + hbs, subsize);
            }
            PARTITION_HORZ_A => {
                self.decode_block(dec, cdfs, mi_row, mi_col, bsize2, p);
                self.decode_block(dec, cdfs, mi_row, mi_col + hbs, bsize2, p);
                self.decode_block(dec, cdfs, mi_row + hbs, mi_col, subsize, p);
            }
            PARTITION_HORZ_B => {
                self.decode_block(dec, cdfs, mi_row, mi_col, subsize, p);
                self.decode_block(dec, cdfs, mi_row + hbs, mi_col, bsize2, p);
                self.decode_block(dec, cdfs, mi_row + hbs, mi_col + hbs, bsize2, p);
            }
            PARTITION_VERT_A => {
                self.decode_block(dec, cdfs, mi_row, mi_col, bsize2, p);
                self.decode_block(dec, cdfs, mi_row + hbs, mi_col, bsize2, p);
                self.decode_block(dec, cdfs, mi_row, mi_col + hbs, subsize, p);
            }
            PARTITION_VERT_B => {
                self.decode_block(dec, cdfs, mi_row, mi_col, subsize, p);
                self.decode_block(dec, cdfs, mi_row, mi_col + hbs, bsize2, p);
                self.decode_block(dec, cdfs, mi_row + hbs, mi_col + hbs, bsize2, p);
            }
            PARTITION_HORZ_4 => {
                for i in 0..4 {
                    let this_mi_row = mi_row + i * quarter_step;
                    if i > 0 && this_mi_row >= self.cfg.mi_rows {
                        break;
                    }
                    self.decode_block(dec, cdfs, this_mi_row, mi_col, subsize, p);
                }
            }
            PARTITION_VERT_4 => {
                for i in 0..4 {
                    let this_mi_col = mi_col + i * quarter_step;
                    if i > 0 && this_mi_col >= self.cfg.mi_cols {
                        break;
                    }
                    self.decode_block(dec, cdfs, mi_row, this_mi_col, subsize, p);
                }
            }
            _ => unreachable!("invalid partition type {p}"),
        }
        update_ext_partition_context(
            &mut self.above_p,
            &mut self.left_p,
            mi_row,
            mi_col,
            subsize,
            bsize,
            p as i32,
        );
    }
}

/// Decode one KEY-frame luma tile spanning the WHOLE FRAME (the single-tile
/// envelope, `TileBoundsKf::whole_frame`): the `decode_tile` SB row/col loop
/// — above contexts zeroed once, left contexts zeroed per SB row, each
/// superblock decoded through the recursive partition walk with the per-leaf
/// mode-info → coefficient → predict → reconstruct interleave.
///
/// `recon_init` fills the reconstruction plane before decoding; a conformant
/// walk never *reads* an unwritten pixel (the availability logic only exposes
/// previously reconstructed samples), so the roundtrip test gives encoder and
/// decoder different fills to turn any availability bug into a hard mismatch.
///
/// For `TileInfoHeader::{cols,rows}` > 1x1, use [`decode_frame_tiles_kf`].
pub fn decode_tile_kf(
    dec: &mut OdEcDec,
    cfg: &KfTileConfig,
    cdfs: &mut KfFrameContext,
    recon_init: u16,
) -> KfTileDecode {
    let mut t = TileKf::new(cfg, recon_init);
    t.decode_one_tile(dec, cdfs);
    t.into_decode()
}

/// Decode a KEY frame with `tiles.len()` tiles (`tiles.len() == 1` is exactly
/// [`decode_tile_kf`]'s envelope, byte-for-byte — both build a fresh
/// [`TileKf`], `start_tile` it once, and run one [`TileKf::decode_one_tile`]).
///
/// Each tile gets: `av1_tile_init`'s bounds ([`TileBytesKf::bounds`]), a
/// FRESH [`KfFrameContext`] (`tile_data->tctx = *cm->fc`, decodeframe.c's
/// per-tile setup — CDF adaptation does NOT carry across tiles; every tile
/// starts from the same default/base-qindex context and independently
/// adapts), and a fresh [`OdEcDec`] over its own byte slice. Tiles are
/// decoded in the given (raster, `tile_row`-major) order into ONE shared
/// frame-aligned reconstruction: the recon planes and the mi/seg_map grids
/// are frame-persistent across all tiles (each tile writes only its own
/// `[mi_row_start, mi_row_end) x [mi_col_start, mi_col_end)` region; a block
/// at a tile's own top/left edge never reads a neighbouring tile's data even
/// though the underlying grids are shared — see [`TileBoundsKf`]).
///
/// NOT modelled: the `context_update_tile_id` tile's post-decode adapted
/// CDFs becoming the frame's saved context (`REFRESH_FRAME_CONTEXT_BACKWARD`,
/// decodeframe.c:5489, `*cm->fc = pbi->tile_data[context_update_tile_id].tctx`).
/// That save only affects OTHER frames that reference this one as
/// `primary_ref_frame` (inter prediction's CDF carry-in) — it has zero effect
/// on this frame's own pixels, and this driver decodes exactly one KEY frame
/// per call with no forward reference chain, so the saved context is never
/// read back. Multiple tile GROUPS (a frame's tiles split across more than
/// one `OBU_TILE_GROUP`) are also out of scope — see `frame.rs`.
pub fn decode_frame_tiles_kf(
    tiles: &[TileBytesKf],
    cfg: &KfTileConfig,
    recon_init: u16,
) -> KfTileDecode {
    assert!(!tiles.is_empty(), "at least one tile");
    let mut t = TileKf::new(cfg, recon_init);
    for tb in tiles {
        let mut dec = OdEcDec::new(tb.bytes);
        let mut cdfs = KfFrameContext::default_for_qindex(cfg.base_qindex);
        t.start_tile(tb.bounds);
        t.decode_one_tile(&mut dec, &mut cdfs);
    }
    t.into_decode()
}
