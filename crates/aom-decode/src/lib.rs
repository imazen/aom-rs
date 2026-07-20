//! KEY-frame tile reconstruction driver — the first aom-rs layer that turns
//! entropy-coded tile bytes into decoded pixels.
//!
//! This crate composes the already-bit-exact building blocks into libaom's
//! decode interleave (`av1/decoder/decodeframe.c`):
//!
//! - partition walk: `decode_partition` — [`aom_dsp::entropy::partition::read_partition`]
//!   per node with the threaded above/left partition context, dispatching leaf
//!   blocks in the exact `DEC_BLOCK` order (all 10 partition types);
//! - per leaf (`parse_decode_block`): mode-info decode
//!   ([`aom_dsp::entropy::partition::read_mb_modes_kf`]) followed by, per plane-0
//!   transform block in raster order (`decode_token_recon_block`, intra path):
//!   coefficient decode ([`aom_dsp::txb::read_coeffs_txb_full`] with
//!   [`aom_dsp::txb::get_txb_ctx`] neighbour contexts) **then** intra prediction
//!   ([`aom_dsp::entropy::partition::intra_avail`] +
//!   [`aom_dsp::intra::predict_intra_high`] into the reconstruction plane) **then**
//!   dequant + inverse transform + add ([`aom_dsp::recon::reconstruct_txb`]) — the
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
//!   luma is stored per txb ([`aom_dsp::intra::cfl::cfl_store_tx`],
//!   `store_cfl_required` — non-reference members always store), and
//!   `UV_CFL_PRED` blocks add the alpha-scaled zero-mean AC on the DC
//!   prediction ([`aom_dsp::intra::cfl::cfl_predict_block`]). Chroma transform
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
//!   [`aom_dsp::entropy::partition::read_mb_modes_kf_fc`], which picks each
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
//! - **`disable_cdf_update` supported**: `cfg.disable_cdf_update` sets
//!   `dec.allow_update_cdf = !disable_cdf_update`, so when the flag is set the
//!   symbol reader ([`aom_dsp::entropy::read_symbol`]) leaves every CDF at its
//!   loaded/initial value for the whole tile (no post-decode adaptation); the
//!   flag-off path adapts unconditionally and stays byte-identical.
//! - **Off / fixed in this cut**: intra block copy,
//!   quantization matrices (flat dequant), and no in-tile loop filters (this driver returns the
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

/// Byte-exact AV1 film-grain synthesis (post-reconstruction output stage).
pub mod film_grain;
pub mod superres;

// `pub` (doc-hidden) so the encoder's forward-QM path can REUSE the inverse-QM
// selector + `iwt_matrix_ref` bases instead of committing a duplicate ~459KB
// table. Interim internal-crate coupling: at release both QM tables (fwd
// `wt_matrix_ref` in aom-quant + inv `iwt_matrix_ref` here) consolidate into one
// shared crate. Only `iqmatrix` is exposed; the rest of the module stays private.
#[doc(hidden)]
pub mod qm;
mod qm_tables;

use aom_dsp::entropy::cdf::read_symbol;
use aom_dsp::recon::{ReconScratch, reconstruct_txb_into};

/// Lossless residual reconstruction: dequantize the 4x4 coefficient block (flat,
/// qindex-0 dequant) and add the inverse 4x4 Walsh–Hadamard transform onto the
/// prediction already in `dst` — the `xd->lossless` arm of
/// `av1_inverse_transform_block` (forced `TX_4X4` + WHT). Mirrors
/// [`reconstruct_txb`] but swaps `av1_inv_txfm2d_add` for the WHT; the caller
/// gates on `eob > 0` (skip blocks reconstruct to the prediction alone).
fn reconstruct_txb_wht(
    dst: &mut [u16],
    stride: usize,
    qcoeff: &[i32],
    dequant: [i16; 2],
    eob: usize,
    bd: i32,
) {
    // TX_4X4: area 16, tx_scale 0, no quant matrix. dequant_txb reproduces the C
    // decoder's dqcoeff exactly (same 20/24-bit masks + bitdepth clamp).
    let mut dqcoeff = [0i32; 16];
    aom_dsp::txb::dequant_txb(qcoeff, &mut dqcoeff, TX_4X4_IDX, dequant, None, bd);
    aom_dsp::transform::inv_txfm2d::av1_highbd_iwht4x4_add(&dqcoeff, dst, stride, eob, bd);
}
use aom_dsp::entropy::dec::OdEcDec;
use aom_dsp::entropy::dv_ref::{
    DvGrid, DvNbr, DvTileBounds, assign_and_validate_dv, find_dv_ref_mvs, find_inter_mv_refs,
    find_samples, select_samples,
};
use aom_dsp::entropy::partition::{
    KfBlockState, KfFrameContext, MbModeInfoKf, MiNbrKf, PaletteNbrKf, TXFM_CTX_INIT, TxMode,
    allow_palette as av1_allow_palette, bsize_to_max_depth, bsize_to_tx_size_cat,
    decode_color_map_tokens, depth_to_tx_size, get_block_dimensions, get_partition_subsize,
    get_plane_block_size, get_tx_size_context, get_uv_mode, intra_avail, is_cfl_allowed,
    partition_cdf_length, partition_plane_context, read_mb_modes_kf_fc, read_partition,
    read_selected_tx_size, set_txfm_ctxs, spatial_seg_pred, tx_size_from_tx_mode,
    txfm_partition_context, txfm_partition_update, update_ext_partition_context,
};
use aom_dsp::intra::cfl::{CflCtx, cfl_predict_block, cfl_store_tx};
use aom_dsp::intra::predict_intra_high;
use aom_dsp::quant::{
    MAX_SEGMENTS, SEG_LVL_MAX, SEG_LVL_SKIP, Segmentation, av1_ac_quant_qtx, av1_dc_quant_qtx,
    av1_get_qindex,
};
use aom_dsp::txb::{
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
    /// `seq_params->color_config.matrix_coefficients` (CICP). Only needed for
    /// film-grain synthesis: `AOM_CICP_MC_IDENTITY` (0) selects the luma legal
    /// range for chroma under `clip_to_restricted_range`.
    pub matrix_coefficients: i32,
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
    pub lr: aom_dsp::entropy::lr::LrFrameConfig,
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
    /// reset stride, [`aom_dsp::entropy::partition::read_cdef`]'s
    /// `cdef_transmitted[4]` unit indexing (already generic on `sb_size`),
    /// and the loop-restoration corners-in-sb SB extent
    /// ([`aom_dsp::entropy::lr::lr_corners_in_sb`]'s `mi_size_wide`/`mi_size_high`
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
    /// `quant_params.using_qmatrix` (`read_quantization`): the frame codes
    /// per-position inverse-QM weights into each 2-D-transform block's dequant
    /// (`av1_get_iqmatrix` / [`crate::qm`]). When false every block takes the
    /// flat dequant (the pre-QM behaviour), so a non-QM frame is unaffected.
    pub using_qmatrix: bool,
    /// `quant_params.qmatrix_level_y` / `_u` / `_v` (`QM_LEVEL_BITS` = 4 each,
    /// `0..=15`): the per-plane QM level selecting the `iwt_matrix_ref` set.
    /// Only meaningful when `using_qmatrix`; level 15 is the flat matrix.
    pub qm_y: usize,
    pub qm_u: usize,
    pub qm_v: usize,
    /// `features.disable_cdf_update` (frame header uncompressed bit; the encoder
    /// sets it under `cdf_update_mode == 0`, and it is forced on by
    /// `error_resilient_mode` for reference frames). When true the tile symbol
    /// reader does NOT adapt CDFs: every `read_symbol` leaves its CDF at the
    /// loaded/initial value for the whole tile. Threaded to the reader as
    /// `dec.allow_update_cdf = !disable_cdf_update`
    /// (`av1/decoder/decodeframe.c`: `allow_update_cdf = !large_scale &&
    /// !disable_cdf_update`). The single-tile KEY frame is never `large_scale`,
    /// so the reader's flag is exactly `!disable_cdf_update`.
    pub disable_cdf_update: bool,
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

/// The per-plane QM levels for a segment — `setup_segmentation_dequant`'s
/// `qmlevel_{y,u,v}` (`decodeframe.c`). `av1_use_qmatrix` is
/// `using_qmatrix && !xd->lossless[segment_id]`; when it holds each plane uses
/// its frame-level `qmatrix_level_*`, otherwise the flat top level
/// (`NUM_QM_LEVELS - 1`). `xd->lossless[i]` is the segment's FRAME-level
/// effective qindex (`av1_get_qindex` on `base_qindex`, not the delta-q carry)
/// being 0 with all plane dc/ac deltas zero — the same condition
/// `is_coded_lossless` sums over. Feeds [`crate::qm::iqmatrix`] per block.
fn frame_qm_levels(cfg: &KfTileConfig, segment_id: usize) -> [usize; 3] {
    let flat = qm::NUM_QM_LEVELS - 1;
    if !cfg.using_qmatrix {
        return [flat; 3];
    }
    let seg_qindex = av1_get_qindex(&cfg.seg, segment_id, cfg.base_qindex);
    let seg_lossless = seg_qindex == 0
        && cfg.y_dc_delta_q == 0
        && cfg.u_dc_delta_q == 0
        && cfg.u_ac_delta_q == 0
        && cfg.v_dc_delta_q == 0
        && cfg.v_ac_delta_q == 0;
    if seg_lossless {
        [flat; 3]
    } else {
        [cfg.qm_y, cfg.qm_u, cfg.qm_v]
    }
}

// The tile's CDF state is libaom's FRAME_CONTEXT itself —
// [`aom_dsp::entropy::partition::KfFrameContext`] (the KEY-frame-intra slice at C
// dims). The driver selects every symbol's instance from its per-context
// arrays; the coefficient arena (`KfFrameContext::coeff`) must be sized
// `aom_dsp::txb::CDF_ARENA_LEN`.

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
    /// Loop-filter mode-info the INTER path carries out-of-band (the intra
    /// `info` fields don't describe an inter block): `Some((ref_frame[0], mode))`
    /// for an inter block, `None` for intra/intrabc. C's `av1_loop_filter_frame`
    /// reads `ref_frame[0]`, `is_inter_block`, and `mode_lf_lut[mode]` per block
    /// to pick the per-block filter level (`ref_deltas`/`mode_deltas`) and the
    /// `skip_txfm && is_inter` boundary-skip decision; for an inter block those
    /// come from here, for intra/intrabc from `info` (ref0 = INTRA_FRAME,
    /// is_inter = `use_intrabc`, mode = `y_mode`).
    pub inter_lf: Option<(i32, i32)>,
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
    pub lr_units: [Vec<aom_dsp::entropy::lr::LrUnitInfo>; 3],
    /// `Some(reason)` when the tile walk hit a corrupt-frame / out-of-envelope
    /// condition on untrusted input (see [`TileKf::corrupt`]). The
    /// `Result`-returning tile-payload entry points in `frame.rs` translate
    /// this into an `Err` and discard the partial reconstruction; `None` on
    /// every conformant in-envelope decode.
    pub corrupt: Option<String>,
}

/// A stored reference frame for inter prediction: the FILTERED reconstruction
/// (post deblock/CDEF/superres/loop-restoration, PRE film-grain) at the coded
/// (post-superres upscaled) resolution, superblock-aligned strides matching the
/// producing `KfTileDecode`. `build_inter_predictor` (aom-inter) reads these
/// planes and edge-replicates OOB reads, so no explicit border is needed. The
/// walking-skeleton target (`av1-1-b8-01-size-64x64` frame 1) references only
/// frame 0's `RefFrame`.
#[derive(Clone, Debug)]
pub struct RefFrame {
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
    pub stride: usize,
    pub stride_uv: usize,
    /// Plane VISIBLE (crop) dimensions for MC edge replication — the reference's
    /// `y_crop_width/height` / `uv_crop_width/height`, i.e. the values C's
    /// `av1_setup_pre_planes` loads into `pre_buf->width/height` (from
    /// `src->crop_widths/crop_heights`) and `build_mc_border` clamps against.
    /// These are the frame's coded (post-superres-upscale) VISIBLE dims, NOT the
    /// SB/mi-aligned recon extent: on a partial-edge frame (dims not a multiple of
    /// 8px) the mi-aligned recon is TALLER/WIDER than the crop, and a bottom/right
    /// edge block's interp taps must edge-replicate at the CROP boundary — reading
    /// the invisible mi-aligned recon rows past it diverges from C.
    pub width: usize,
    pub height: usize,
    pub width_uv: usize,
    pub height_uv: usize,
    pub order_hint: i32,
}

impl RefFrame {
    /// Capture the filtered reconstruction from a post-filtered `KfTileDecode`
    /// (call after [`crate::frame::run_post_filters`], before crop/film-grain).
    /// `crop_*` are the frame's VISIBLE dimensions (`y_crop_width/height` and
    /// `uv_crop_width/height`), which drive the MC border clamp — see the field
    /// docs on [`RefFrame`]. The recon buffers + strides stay SB/mi-aligned.
    pub fn from_filtered(
        t: &KfTileDecode,
        order_hint: i32,
        crop_w: usize,
        crop_h: usize,
        crop_w_uv: usize,
        crop_h_uv: usize,
    ) -> Self {
        RefFrame {
            y: t.recon.clone(),
            u: t.recon_u.clone(),
            v: t.recon_v.clone(),
            stride: t.stride,
            stride_uv: t.stride_uv,
            width: crop_w,
            height: crop_h,
            width_uv: crop_w_uv,
            height_uv: crop_h_uv,
            order_hint,
        }
    }
}

/// The inter-frame-level state the inter mode-info driver + motion compensation
/// need beyond the shared [`KfTileConfig`]: the single reference frame and the
/// header flags that gate the per-block reads. The walking-skeleton envelope
/// (single LAST reference, `SINGLE_REFERENCE` mode, `primary_ref = NONE`,
/// `tx_mode = LARGEST`, no segmentation / skip-mode / delta-q).
#[derive(Clone, Copy)]
pub struct InterFrameCfg<'r> {
    /// The single reference (`LAST_FRAME`): frame 0's filtered recon.
    pub last: &'r RefFrame,
    pub allow_high_precision_mv: bool,
    pub cur_frame_force_integer_mv: bool,
    /// Frame-level `interp_filter` (`4 == SWITCHABLE`).
    pub interp_filter: i32,
    pub switchable_motion_mode: bool,
    pub allow_ref_frame_mvs: bool,
    pub reference_mode_select: bool,
    pub enable_dual_filter: bool,
    pub allow_warped_motion: bool,
    pub skip_mode_allowed: bool,
    /// `seq_header.enable_interintra_compound`: gates the inter-intra flag read
    /// on an interintra-allowed inter block.
    pub enable_interintra_compound: bool,
    /// Per-reference global-motion `wmtype` (`global_motion[ref].wmtype`;
    /// `0 = IDENTITY`). The inter MV scan (`find_inter_mv_refs`) currently assumes
    /// IDENTITY global motion (the base MV for a NEWMV/GLOBALMV block with an empty
    /// ref-mv list); a non-identity ref is guarded (a later chunk — every target
    /// through 16x18 is identity-GM).
    pub gm_wmtype: [u8; 7],
    pub order_hint: i32,
}

/// The inter mode-info CDFs, loaded from the `default_cdfs` tables (the
/// `primary_ref = NONE` default-context load) and threaded across a tile's inter
/// blocks so they ADAPT (`update_cdf`) exactly like [`KfFrameContext`] does for
/// intra: hosted on [`TileKf::inter_cdfs`], reset to defaults per tile in
/// [`TileKf::start_tile`], read+adapted in place by every inter symbol read.
/// `Copy` so [`TileKf::decode_block_inter`] can take a local snapshot, adapt it
/// through the reads (the `single_ref` sub-tree is assembled into a scratch by
/// [`InterCdfs::ref_frame_cdfs`] and its adaptations copied back), then persist.
#[derive(Clone, Copy)]
struct InterCdfs {
    intra_inter: [[u16; 3]; 4],
    single_ref: [[[u16; 3]; 6]; 3],
    newmv: [[u16; 3]; 6],
    zeromv: [[u16; 3]; 2],
    refmv: [[u16; 3]; 6],
    drl: [[u16; 3]; 3],
    switchable_interp: [[u16; 4]; 16],
    nmv_joints: [u16; 5],
    nmv_comps: [[u16; 69]; 2],
    /// `obmc_cdf[BLOCK_SIZES_ALL]` (CDF_SIZE(2)): the 2-symbol OBMC-vs-SIMPLE
    /// flag read when the motion-mode ceiling is `OBMC_CAUSAL`.
    obmc: [[u16; 3]; 22],
    /// `motion_mode_cdf[BLOCK_SIZES_ALL]` (CDF_SIZE(MOTION_MODES=3)): the
    /// 3-symbol SIMPLE/OBMC/WARP mode read when the ceiling is `WARPED_CAUSAL`.
    motion_mode: [[u16; 4]; 22],
    /// `interintra_cdf[BLOCK_SIZE_GROUPS]` (CDF_SIZE(2)): the inter-intra flag.
    interintra: [[u16; 3]; 4],
    /// `interintra_mode_cdf[BLOCK_SIZE_GROUPS]` (CDF_SIZE(INTERINTRA_MODES=4)):
    /// the inter-intra mode, read when the flag is set (decodemv.c:1391).
    interintra_mode: [[u16; 5]; 4],
    /// `wedge_interintra_cdf[BLOCK_SIZES_ALL]` (CDF_SIZE(2)): wedge-vs-smooth,
    /// read when `av1_is_wedge_used(bsize)` (decodemv.c:1398).
    wedge_interintra: [[u16; 3]; 22],
    /// `wedge_idx_cdf[BLOCK_SIZES_ALL]` (CDF_SIZE(MAX_WEDGE_TYPES=16)): the
    /// wedge shape (decodemv.c:1402). Shared with inter-inter COMPOUND_WEDGE.
    wedge_idx: [[u16; 17]; 22],
    /// `y_mode_cdf[BLOCK_SIZE_GROUPS]` (CDF_SIZE(INTRA_MODES=13)): the Y mode of
    /// an INTRA block inside this inter frame (`read_intra_block_mode_info`,
    /// decodemv.c:1077) — the size-group table, NOT the KEY frame's
    /// neighbour-context `kf_y_cdf`. Lives here rather than on [`KfFrameContext`]
    /// because the KEY-frame path never codes it.
    y_mode: [[u16; 14]; 4],
}

impl InterCdfs {
    fn defaults() -> Self {
        use aom_dsp::entropy::default_cdfs as d;
        InterCdfs {
            intra_inter: d::DEFAULT_INTRA_INTER,
            single_ref: d::DEFAULT_SINGLE_REF,
            newmv: d::DEFAULT_NEWMV,
            zeromv: d::DEFAULT_ZEROMV,
            refmv: d::DEFAULT_REFMV,
            drl: d::DEFAULT_DRL,
            switchable_interp: d::DEFAULT_SWITCHABLE_INTERP,
            nmv_joints: d::DEFAULT_NMV_JOINTS,
            nmv_comps: d::DEFAULT_NMV_COMPS,
            obmc: d::DEFAULT_OBMC,
            motion_mode: d::DEFAULT_MOTION_MODE,
            interintra: d::DEFAULT_INTERINTRA,
            interintra_mode: d::DEFAULT_INTERINTRA_MODE,
            wedge_interintra: d::DEFAULT_WEDGE_INTERINTRA,
            wedge_idx: d::DEFAULT_WEDGE_IDX,
            y_mode: d::DEFAULT_Y_MODE,
        }
    }

    /// Assemble the 16-entry ref-frame CDF array `read_ref_frames` indexes: the
    /// single-reference sub-tree slots `[10..16]` selected at their pred
    /// contexts from the neighbour ref counts. Compound slots `[0..10]` are
    /// unused under `SINGLE_REFERENCE` (never read).
    fn ref_frame_cdfs(&self, rc: &[u8; 8]) -> [[u16; 3]; 16] {
        use aom_dsp::entropy::partition as p;
        let mut cdfs = [[0u16; 3]; 16];
        cdfs[10] = self.single_ref[p::single_ref_p1_context(rc) as usize][0];
        cdfs[11] = self.single_ref[p::pred_ctx_brfarf2_or_arf(rc) as usize][1];
        cdfs[12] = self.single_ref[p::pred_ctx_ll2_or_l3gld(rc) as usize][2];
        cdfs[13] = self.single_ref[p::pred_ctx_last_or_last2(rc) as usize][3];
        cdfs[14] = self.single_ref[p::pred_ctx_last3_or_gld(rc) as usize][4];
        cdfs[15] = self.single_ref[p::pred_ctx_brf_or_arf2(rc) as usize][5];
        cdfs
    }
}

/// One tile's mi-space extent within the frame (`TileInfo::mi_row_start` /
/// `mi_row_end` / `mi_col_start` / `mi_col_end`, `av1_tile_set_row` /
/// `av1_tile_set_col`, tile_common.c: `row_start_sb[row] << mib_size_log2`,
/// clamped to the frame's `mi_rows`/`mi_cols`). Drives tile-relative
/// `up_available`/`left_available` (`mi_row/col > mi_row/col_start` — NOT
/// `> 0`: a block at a tile's own top/left edge has no available neighbour
/// even when the tile itself sits interior to the frame) and
/// [`aom_dsp::entropy::partition::intra_avail`]'s `tile_col_end`/`tile_row_end`
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
/// [`aom_dsp::entropy::partition::KfFrameContext`] alongside every other frame CDF,
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
/// At a var-tx leaf: (a) track uniformity via `first_leaf`/`non_uniform`, and
/// (b) stamp the leaf's tx size over its mi footprint in `leaf_grid` (per-4x4,
/// block-relative, `grid_stride` = block width in mi units, `grid_h` = height).
/// The reconstruction walk ([`collect_vartx_leaves`]) reads this grid at a
/// node's top-left cell to decide split-vs-leaf, exactly as C's
/// `decode_reconstruct_tx` reads `mbmi->inter_tx_size[]`. Footprint clamped to
/// the grid extent (edge blocks).
#[allow(clippy::too_many_arguments)]
fn vartx_leaf(
    first_leaf: &mut i32,
    non_uniform: &mut bool,
    leaf_grid: &mut [u8],
    grid_stride: usize,
    grid_h: usize,
    blk_row: i32,
    blk_col: i32,
    leaf: usize,
) {
    if *first_leaf < 0 {
        *first_leaf = leaf as i32;
    } else if *first_leaf != leaf as i32 {
        *non_uniform = true;
    }
    let (br, bc) = (blk_row as usize, blk_col as usize);
    let lh = TX_SIZE_HIGH_UNIT[leaf].min(grid_h - br);
    let lw = TX_SIZE_WIDE_UNIT[leaf].min(grid_stride - bc);
    for r in 0..lh {
        for c in 0..lw {
            leaf_grid[(br + r) * grid_stride + bc + c] = leaf as u8;
        }
    }
}

/// `read_tx_size_vartx` (`av1/decoder/decodeframe.c`): the recursive
/// variable-transform-size split READER for an inter / intrabc block, the exact
/// inverse of the encoder's [`aom_dsp::entropy::partition::write_tx_size_vartx`]. At
/// each quadtree node it reads a split flag from `txfm_partition_cdf[ctx]` (2
/// symbols), stamps the neighbour txfm-context arrays via
/// [`txfm_partition_update`], and recurses into the `sub_tx_size_map` children
/// down to `MAX_VARTX_DEPTH`. `tx_size_out` receives C's `mbmi->tx_size` — set at
/// exactly the leaves where C sets it (an out-of-bounds child leaves the running
/// value untouched, matching C's early `return`).
///
/// This is the SIZE-READ phase (C's `read_block_tx_size`): it fills `leaf_grid`
/// with the per-4x4 leaf tx sizes (C's `mbmi->inter_tx_size[]`) so the later
/// reconstruction phase ([`collect_vartx_leaves`]) can walk the same quadtree.
/// `non_uniform` records whether the leaves differ; the caller uses it to pick
/// the uniform fast loop vs the general leaf-walk.
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
    first_leaf: &mut i32,
    non_uniform: &mut bool,
    leaf_grid: &mut [u8],
    grid_stride: usize,
    grid_h: usize,
) {
    let max_blocks_high = max_block_high_luma(bsize, mb_to_bottom_edge);
    let max_blocks_wide = max_block_wide_luma(bsize, mb_to_right_edge);
    if blk_row >= max_blocks_high || blk_col >= max_blocks_wide {
        return;
    }
    let (bc, br) = (blk_col as usize, blk_row as usize);
    if depth == VARTX_MAX_DEPTH {
        *tx_size_out = tx_size;
        vartx_leaf(
            first_leaf,
            non_uniform,
            leaf_grid,
            grid_stride,
            grid_h,
            blk_row,
            blk_col,
            tx_size,
        );
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
        return;
    }

    let ctx = txfm_partition_context(above_ctx[bc], left_ctx[br], bsize, tx_size);
    let is_split = read_symbol(dec, &mut txfm_partition_cdf[ctx], 2);
    if is_split != 0 {
        let sub_txs = SUB_TX_SIZE_MAP[tx_size];
        if sub_txs == TX_4X4_IDX {
            *tx_size_out = sub_txs;
            vartx_leaf(
                first_leaf,
                non_uniform,
                leaf_grid,
                grid_stride,
                grid_h,
                blk_row,
                blk_col,
                sub_txs,
            );
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
                    leaf_grid,
                    grid_stride,
                    grid_h,
                );
                col += bsw;
            }
            row += bsh;
        }
    } else {
        *tx_size_out = tx_size;
        vartx_leaf(
            first_leaf,
            non_uniform,
            leaf_grid,
            grid_stride,
            grid_h,
            blk_row,
            blk_col,
            tx_size,
        );
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
    }
}

/// Reconstruction-phase walk (C's `decode_reconstruct_tx`, luma): re-derive the
/// var-tx quadtree from the per-4x4 `leaf_grid` produced by the size-read phase
/// and emit the leaf `(blk_row, blk_col, tx_size)` triples in DFS quadrant order
/// (TL, TR, BL, BR) — the order the coefficient stream expects. A node is a leaf
/// iff its size equals the stored leaf size at its top-left cell; no bitstream is
/// read here. `max_blocks_*` clamp edge blocks.
#[allow(clippy::too_many_arguments)]
fn collect_vartx_leaves(
    leaf_grid: &[u8],
    grid_stride: usize,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    max_blocks_high: usize,
    max_blocks_wide: usize,
    out: &mut Vec<(usize, usize, usize)>,
) {
    if blk_row >= max_blocks_high || blk_col >= max_blocks_wide {
        return;
    }
    let plane_tx = leaf_grid[blk_row * grid_stride + blk_col] as usize;
    if tx_size == plane_tx {
        out.push((blk_row, blk_col, tx_size));
        return;
    }
    let sub_txs = SUB_TX_SIZE_MAP[tx_size];
    let bsw = TX_SIZE_WIDE_UNIT[sub_txs];
    let bsh = TX_SIZE_HIGH_UNIT[sub_txs];
    let row_end = TX_SIZE_HIGH_UNIT[tx_size].min(max_blocks_high - blk_row);
    let col_end = TX_SIZE_WIDE_UNIT[tx_size].min(max_blocks_wide - blk_col);
    let mut row = 0;
    while row < row_end {
        let mut col = 0;
        while col < col_end {
            collect_vartx_leaves(
                leaf_grid,
                grid_stride,
                blk_row + row,
                blk_col + col,
                sub_txs,
                max_blocks_high,
                max_blocks_wide,
                out,
            );
            col += bsw;
        }
        row += bsh;
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
    /// Per-mi coded interpolation-filter grid `(y_filter, x_filter)` — the
    /// neighbour projection of `MB_MODE_INFO::interp_filters` that
    /// `av1_get_pred_context_switchable_interp` (pred_common.c) reads through
    /// `xd->mi[-1]` / `xd->mi[-mi_stride]` when selecting the switchable-interp
    /// CDF context. Kept parallel to `mi_dv` (which supplies the neighbour's
    /// `ref_frame[0/1]` — `get_ref_filter_type` only returns the stored filter
    /// when the neighbour references the current block's ref). Frame-cropped
    /// stamps like `mi_dv`; every cell is `(0, 0)` = `EIGHTTAP_REGULAR` (C's mi
    /// grid init) until an inter block stamps its resolved filters. Only inter
    /// frames read it (KEY frames never reach the switchable-interp path), so
    /// ordinary KEY decodes never index it.
    mi_interp: Vec<(u8, u8)>,
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
    /// The current block's per-plane QM level (`av1_use_qmatrix` resolved:
    /// `qmatrix_level_{y,u,v}` when the frame uses QM and this block's segment
    /// is not lossless, else the flat `NUM_QM_LEVELS - 1`). Recomputed per
    /// coded block alongside `dequants`; consumed by [`crate::qm::iqmatrix`] at
    /// each `reconstruct_txb`. `[15; 3]` (flat) when the frame doesn't use QM,
    /// so a non-QM frame is byte-identical to the pre-QM path.
    block_qm_level: [usize; 3],
    st: KfBlockState,
    tree: Vec<i8>,
    blocks: Vec<DecodedBlockKf>,
    /// Per-plane restoration-unit params (unit-grid raster order); sized
    /// `horz_units * vert_units` for restored planes, empty otherwise.
    lr_units: [Vec<aom_dsp::entropy::lr::LrUnitInfo>; 3],
    /// `xd->wiener_info` / `xd->sgrproj_info` — the per-plane RU-params
    /// prediction references (`av1_reset_loop_restoration` at tile start).
    lr_refs: aom_dsp::entropy::lr::LrRefState,
    /// This tile's mi-space extent (`xd->tile`) — see [`TileBoundsKf`]. Set
    /// by `start_tile` at the beginning of each tile's decode.
    tile: TileBoundsKf,
    /// When set, this frame is an INTER frame: `decode_block` takes the inter
    /// mode-info + motion-compensation path ([`TileKf::decode_block_inter`])
    /// instead of the KEY intra path. `None` for a KEY frame (the default).
    inter: Option<InterFrameCfg<'c>>,
    /// The tile's inter mode-info CDFs ([`InterCdfs`]), threaded across inter
    /// blocks so they adapt like the intra [`KfFrameContext`]. Reset to defaults
    /// per tile in [`start_tile`]; unused (but harmless) on KEY frames.
    inter_cdfs: InterCdfs,
    /// Reusable dequant + inverse-transform scratch for
    /// [`aom_dsp::recon::reconstruct_txb_into`]. Gate 3: the decoder
    /// reconstructs one transform block at a time, and allocating these two
    /// buffers per block was the largest allocator caller in the decode
    /// profile. Contents are fully overwritten on every use, so reuse is
    /// byte-for-byte identical to a fresh allocation.
    recon_scratch: ReconScratch,
    /// Set when the tile walk hits a corrupt-frame or out-of-envelope
    /// condition on UNTRUSTED input (a malformed bitstream that libaom would
    /// reject with `AOM_CODEC_CORRUPT_FRAME`, or a valid-but-unsupported
    /// feature the current envelope does not decode). Instead of panicking —
    /// a denial-of-service on the untrusted AVIF decode path — the walk sets
    /// this reason string and unwinds via the re-entrant `corrupt` guards; the
    /// `Result`-returning tile-payload entry points surface it as `Err`. `None`
    /// on every conformant in-envelope stream, so the guard is a
    /// never-taken branch and the decode stays byte-identical.
    corrupt: Option<String>,
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
        // coded_lossless (C's is_coded_lossless, decodeframe.c): every segment
        // lossless (effective qindex 0 with all plane dc/ac deltas zero —
        // `xd->lossless[i]`), or `base_qindex == 0` with segmentation off. It
        // forces the per-block transform to TX_4X4 + the 4x4 WHT and narrows
        // is_cfl_allowed to BLOCK_4X4. A MIXED frame (some-but-not-all segments
        // lossless) is rejected in parse_frame_header, so within a tile the
        // lossless status is uniform and this single flag drives every block.
        let plane_deltas_zero = cfg.y_dc_delta_q == 0
            && cfg.u_dc_delta_q == 0
            && cfg.u_ac_delta_q == 0
            && cfg.v_dc_delta_q == 0
            && cfg.v_ac_delta_q == 0;
        let coded_lossless = plane_deltas_zero
            && if cfg.seg.enabled {
                (0..MAX_SEGMENTS).all(|i| av1_get_qindex(&cfg.seg, i, cfg.base_qindex) == 0)
            } else {
                cfg.base_qindex == 0
            };
        if cfg.seg.enabled {
            debug_assert!(
                coded_lossless
                    || !(0..=last_active_segid as usize).any(|i| {
                        plane_deltas_zero && av1_get_qindex(&cfg.seg, i, cfg.base_qindex) == 0
                    }),
                "mixed lossless segments reached TileKf::new (rejected upstream)"
            );
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
            coded_lossless,
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
            mi_interp: vec![(0u8, 0u8); (cfg.mi_rows * cfg.mi_cols) as usize],
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
            // Frame-level QM levels (segment 0; recomputed per block below).
            // Flat when QM is off, so the flat dequant matches the pre-QM path.
            block_qm_level: frame_qm_levels(cfg, 0),
            st,
            tree: Vec::new(),
            blocks: Vec::new(),
            lr_units: std::array::from_fn(|p| {
                if cfg.lr.frame_restoration_type[p] == aom_dsp::entropy::lr::RESTORE_NONE {
                    Vec::new()
                } else {
                    let (hu, vu) = cfg.lr.plane_units(p, ss_x, ss_y);
                    vec![aom_dsp::entropy::lr::LrUnitInfo::default(); (hu * vu) as usize]
                }
            }),
            lr_refs: aom_dsp::entropy::lr::LrRefState::default(),
            tile: TileBoundsKf::whole_frame(cfg),
            inter: None,
            inter_cdfs: InterCdfs::defaults(),
            recon_scratch: ReconScratch::default(),
            corrupt: None,
        };
        // Run the same per-tile reset `start_tile` applies to any later tile
        // — for the first (or only) tile this re-touches already-fresh state
        // (the arrays above are already zeroed/defaulted), but keeps `new`
        // and `start_tile` as the ONE place that logic lives.
        let whole_frame = TileBoundsKf::whole_frame(cfg);
        t.start_tile(whole_frame);
        t
    }

    /// Flag the current frame as corrupt / out-of-envelope and unwind the walk.
    /// The first reason wins (later re-entrant guards short-circuit before
    /// reaching a second site). Cold + out-of-line: never taken on a
    /// conformant in-envelope stream, so the decode of valid input is
    /// unaffected. See [`TileKf::corrupt`].
    #[cold]
    #[inline(never)]
    fn mark_corrupt(&mut self, reason: impl Into<String>) {
        if self.corrupt.is_none() {
            self.corrupt = Some(reason.into());
        }
    }

    /// True once [`mark_corrupt`] has fired — the re-entrant walk guards test
    /// this to stop decoding a poisoned frame.
    #[inline]
    fn is_corrupt(&self) -> bool {
        self.corrupt.is_some()
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
        // Same per-tile reload for the inter mode-info CDFs (primary_ref = NONE →
        // default context); they then adapt across the tile's inter blocks.
        self.inter_cdfs = InterCdfs::defaults();

        self.cfl = CflCtx::new(ss_x as i32, ss_y as i32);
        self.lr_refs = aom_dsp::entropy::lr::LrRefState::default();
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
                if self.is_corrupt() {
                    return;
                }
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
            corrupt: self.corrupt,
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

    /// [`Self::stamp_dv`]'s interp-filter twin — stamp the block's resolved
    /// `(y_filter, x_filter)` over its frame-cropped mi footprint so later
    /// switchable inter blocks' `av1_get_pred_context_switchable_interp`
    /// neighbour reads see it (`cm->mi[..]->interp_filters`).
    fn stamp_interp(&mut self, mi_row: i32, mi_col: i32, bsize: usize, cell: (u8, u8)) {
        let x_mis = MI_SIZE_WIDE[bsize].min(self.cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(self.cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * self.cfg.mi_cols + mi_col) as usize;
            self.mi_interp[base..base + x_mis as usize].fill(cell);
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

    /// `foreach_overlappable_nb_above` (obmc.h): walk the mi row above the block
    /// across its width, stepping by each above-neighbour's mi width (a width-4
    /// block is pair-adjusted to its chroma-carrying second half), collecting
    /// inter (`is_neighbor_overlappable`) neighbours up to `nb_max`. Returns
    /// `(rel_mi_col, op_mi_size, neighbour)` per hit.
    fn overlappable_above(
        &self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        nb_max: i32,
    ) -> Vec<(i32, i32, DvNbr, (u8, u8))> {
        let cols = self.cfg.mi_cols;
        let mut out = Vec::new();
        if mi_row <= self.tile.mi_row_start {
            return out; // !up_available
        }
        // mi_size_wide[BLOCK_64X64] = 16.
        const MI64: i32 = 16;
        let width = MI_SIZE_WIDE[bsize];
        let end_col = (mi_col + width).min(cols);
        let mut amc = mi_col;
        while amc < end_col && (out.len() as i32) < nb_max {
            let d0 = self.mi_dv[((mi_row - 1) * cols + amc) as usize];
            let mut mi_step = MI_SIZE_WIDE[d0.bsize].min(MI64);
            // The neighbour mbmi (and its coded interp filter, for the OBMC strip
            // MC — C uses `above_mbmi->interp_filters`) come from the SAME grid cell.
            let (nb, nb_if) = if mi_step == 1 {
                amc &= !1;
                mi_step = 2;
                let idx = ((mi_row - 1) * cols + amc + 1) as usize;
                (self.mi_dv[idx], self.mi_interp[idx])
            } else {
                (d0, self.mi_interp[((mi_row - 1) * cols + amc) as usize])
            };
            if nb.use_intrabc || nb.ref_frame0 > 0 {
                out.push((amc - mi_col, width.min(mi_step), nb, nb_if));
            }
            amc += mi_step;
        }
        out
    }

    /// `foreach_overlappable_nb_left` (obmc.h): the left-column twin of
    /// [`Self::overlappable_above`]. Returns `(rel_mi_row, op_mi_size, neighbour)`.
    fn overlappable_left(
        &self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        nb_max: i32,
    ) -> Vec<(i32, i32, DvNbr, (u8, u8))> {
        let cols = self.cfg.mi_cols;
        let mut out = Vec::new();
        if mi_col <= self.tile.mi_col_start {
            return out; // !left_available
        }
        const MI64: i32 = 16;
        let height = MI_SIZE_HIGH[bsize];
        let end_row = (mi_row + height).min(self.cfg.mi_rows);
        let mut amr = mi_row;
        while amr < end_row && (out.len() as i32) < nb_max {
            let d0 = self.mi_dv[(amr * cols + mi_col - 1) as usize];
            let mut mi_step = MI_SIZE_HIGH[d0.bsize].min(MI64);
            let (nb, nb_if) = if mi_step == 1 {
                amr &= !1;
                mi_step = 2;
                let idx = ((amr + 1) * cols + mi_col - 1) as usize;
                (self.mi_dv[idx], self.mi_interp[idx])
            } else {
                (d0, self.mi_interp[(amr * cols + mi_col - 1) as usize])
            };
            if nb.use_intrabc || nb.ref_frame0 > 0 {
                out.push((amr - mi_row, height.min(mi_step), nb, nb_if));
            }
            amr += mi_step;
        }
        out
    }

    /// `av1_count_overlappable_neighbors` (reconinter.c:801): whether the block
    /// has any overlappable (inter) above/left neighbour — the OBMC/warp
    /// motion-mode gate. Only `!= 0` is consulted by `motion_mode_allowed`.
    fn has_overlappable_neighbors(&self, mi_row: i32, mi_col: i32, bsize: usize) -> bool {
        if !is_motion_variation_allowed_bsize(bsize) {
            return false;
        }
        if !self
            .overlappable_above(mi_row, mi_col, bsize, i32::MAX)
            .is_empty()
        {
            return true;
        }
        !self
            .overlappable_left(mi_row, mi_col, bsize, i32::MAX)
            .is_empty()
    }

    /// `av1_findSamples` (mvref_common.c:1118), count-only: the number of warp
    /// projection samples = above/left/top-left neighbours whose single reference
    /// matches `ref_frame` (`ref[0] == ref_frame && ref[1] == NONE`), capped at
    /// `LEAST_SQUARES_SAMPLES_MAX = 8`. Feeds the `num_proj_ref >= 1` arm of the
    /// motion-mode ceiling. The TOP-RIGHT scan (mvref_common.c:1223, gated by
    /// `has_top_right`) is not ported: it only ADDS samples, so the ceiling's
    /// `>= 1` test is unaffected whenever the above/left/top-left scan already
    /// yields a sample — true for every OBMC block in the chunk-4 target (whose
    /// OBMC block is at the frame's left+right edges, so its 4 samples all come
    /// from the above scan). The exact count + `record_samples` for warp land
    /// with chunk 5 (WARPED_CAUSAL).
    fn num_proj_ref(&self, mi_row: i32, mi_col: i32, bsize: usize, ref_frame: i32) -> i32 {
        const MAX: i32 = 8; // LEAST_SQUARES_SAMPLES_MAX
        let cols = self.cfg.mi_cols;
        let rows = self.cfg.mi_rows;
        let up = mi_row > self.tile.mi_row_start;
        let left = mi_col > self.tile.mi_col_start;
        let width = MI_SIZE_WIDE[bsize];
        let height = MI_SIZE_HIGH[bsize];
        let matches = |d: &DvNbr| d.ref_frame0 == ref_frame && d.ref_frame1 == -1;
        let mut np = 0i32;
        let mut do_tl = true;

        if up {
            let d0 = self.mi_dv[((mi_row - 1) * cols + mi_col) as usize];
            let sbw = MI_SIZE_WIDE[d0.bsize];
            if width <= sbw {
                let col_offset = (-mi_col) % sbw;
                if col_offset < 0 {
                    do_tl = false;
                }
                if matches(&d0) {
                    np += 1;
                    if np >= MAX {
                        return MAX;
                    }
                }
            } else {
                let end = width.min(cols - mi_col);
                let mut i = 0;
                while i < end {
                    let d = self.mi_dv[((mi_row - 1) * cols + mi_col + i) as usize];
                    if matches(&d) {
                        np += 1;
                        if np >= MAX {
                            return MAX;
                        }
                    }
                    i += MI_SIZE_WIDE[d.bsize];
                }
            }
        }

        if left {
            let d0 = self.mi_dv[(mi_row * cols + mi_col - 1) as usize];
            let sbh = MI_SIZE_HIGH[d0.bsize];
            if height <= sbh {
                let row_offset = (-mi_row) % sbh;
                if row_offset < 0 {
                    do_tl = false;
                }
                if matches(&d0) {
                    np += 1;
                    if np >= MAX {
                        return MAX;
                    }
                }
            } else {
                let end = height.min(rows - mi_row);
                let mut i = 0;
                while i < end {
                    let d = self.mi_dv[((mi_row + i) * cols + mi_col - 1) as usize];
                    if matches(&d) {
                        np += 1;
                        if np >= MAX {
                            return MAX;
                        }
                    }
                    i += MI_SIZE_HIGH[d.bsize];
                }
            }
        }

        if do_tl && left && up {
            let d = self.mi_dv[((mi_row - 1) * cols + mi_col - 1) as usize];
            if matches(&d) {
                np += 1;
                if np >= MAX {
                    return MAX;
                }
            }
        }
        np
    }

    /// `motion_mode_allowed` (blockd.h:1477): the motion-mode ceiling given the
    /// block's mode/refs and the frame's `allow_warped_motion`. Returns
    /// `0 = SIMPLE_TRANSLATION`, `1 = OBMC_CAUSAL`, `2 = WARPED_CAUSAL`. Global
    /// motion is identity here (`gm_type = IDENTITY`, not `> TRANSLATION`), so
    /// `is_global_mv_block` is unconditionally false and its early-out is skipped.
    fn motion_mode_ceiling(
        &self,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        mode: i32,
        ref0: i32,
        ref1: i32,
        inter: &InterFrameCfg,
    ) -> i32 {
        // NEARESTMV=13 .. NEWMV=16 are the single-ref inter modes.
        const NEARESTMV: i32 = 13;
        if !self.has_overlappable_neighbors(mi_row, mi_col, bsize) {
            return 0;
        }
        let is_inter_mode = mode >= NEARESTMV;
        // ref1 != INTRA_FRAME (0) and !has_second_ref (ref1 <= INTRA_FRAME).
        if is_motion_variation_allowed_bsize(bsize) && is_inter_mode && ref1 < 0 {
            let npr = self.num_proj_ref(mi_row, mi_col, bsize, ref0);
            if npr >= 1 && inter.allow_warped_motion && !inter.cur_frame_force_integer_mv {
                // && !av1_is_scaled: single unscaled LAST ref -> not scaled.
                return 2; // WARPED_CAUSAL
            }
            return 1; // OBMC_CAUSAL
        }
        0
    }

    /// `dec_build_prediction_by_above_preds` + `av1_build_obmc_inter_prediction`
    /// (ABOVE half), luma: for each overlappable above-neighbour, MC-predict a
    /// narrow `op_mi_size*4`-wide × `overlap`-tall strip from the reference using
    /// the NEIGHBOUR's mv/ref/filter, then feather-blend it into the block's own
    /// predictor (already in `recon`) with the vertical OBMC mask. Chroma OBMC is
    /// skipped here (`av1_skip_u4x4_pred_in_obmc` returns 1 for a `<= 8x8`
    /// chroma-plane block in the ABOVE direction — true for this target's
    /// BLOCK_16X8 -> BLOCK_8X4 chroma); a chroma-carrying above-OBMC block is a
    /// later target and is asserted-guarded.
    fn obmc_above_blend(&mut self, mi_row: i32, mi_col: i32, bsize: usize, inter: &InterFrameCfg) {
        let (ss_x, ss_y) = (self.cfg.subsampling_x, self.cfg.subsampling_y);
        // Chroma above-OBMC fires unless `av1_skip_u4x4_pred_in_obmc(dir=0)` skips
        // it (a `<= 8x8` chroma plane block: BLOCK_8X8/8X16/16X8 at 4:2:0). The
        // chroma strips + blend are added per-plane after each neighbour's luma.
        let do_chroma = !self.cfg.monochrome && !skip_u4x4_pred_in_obmc(bsize, ss_x, ss_y, 0);
        let nb_max = MAX_NEIGHBOR_OBMC[mi_size_wide_log2(bsize)];
        let neighbours = self.overlappable_above(mi_row, mi_col, bsize, nb_max);
        if neighbours.is_empty() {
            return;
        }
        let bsize_high = BLOCK_SIZE_HIGH[bsize]; // px
        let overlap = bsize_high.min(64) >> 1; // luma overlap rows
        let width = MI_SIZE_WIDE[bsize];
        // dec_build_prediction_by_above_preds edge adjustments (decodeframe.c:736).
        let this_height = MI_SIZE_HIGH[bsize] * 4;
        let pred_height = (this_height / 2).min(32);
        let block_mb_to_right = (self.cfg.mi_cols - width - mi_col) * 32;
        let block_mb_to_bottom = (self.cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let nb_mb_to_bottom = block_mb_to_bottom + (this_height - pred_height) * 8;
        let nb_mb_to_top = -(mi_row * 32);
        let last = inter.last;
        for (rel_mi_col, op_mi_size, nb, nb_if) in neighbours {
            // The OBMC strip is MC'd with the NEIGHBOUR's coded interp filter
            // (av1_setup_build_prediction_by_above_pred keeps `above_mbmi->
            // interp_filters`). (y_filter, x_filter) = (nb_if.0, nb_if.1).
            let (nb_filter_y, nb_filter_x) = (nb_if.0 as usize, nb_if.1 as usize);
            let above_mi_col = mi_col + rel_mi_col;
            // Luma prediction dims (dec_build_prediction_by_above_pred, ss=0):
            // bw = op_mi_size*4; bh = clamp(bsize_high>>1, 4, 32).
            let bw = (op_mi_size * 4) as usize;
            let bh = (bsize_high >> 1).clamp(4, 32) as usize;
            // OBMC-neighbour mb edges (av1_setup_build_prediction_by_above_pred).
            let nb_mb_to_left = -(above_mi_col * 32);
            let nb_mb_to_right = block_mb_to_right + (width - rel_mi_col - op_mi_size) * 32;
            let (nmv_r, nmv_c) = clamp_mv_umv_border_px(
                nb.mv0_row,
                nb.mv0_col,
                bw as i32,
                bh as i32,
                nb_mb_to_left,
                nb_mb_to_right,
                nb_mb_to_top,
                nb_mb_to_bottom,
                0,
                0,
            );
            let mut scratch = vec![0u16; bw * bh];
            aom_dsp::inter::build_inter_predictor(
                &last.y,
                last.stride,
                last.width,
                last.height,
                &mut scratch,
                0,
                bw,
                (above_mi_col * 4) as usize, // ref-read pix col
                (mi_row * 4) as usize,       // ref-read pix row (current block row)
                bw,
                bh,
                nmv_r,
                nmv_c,
                0,
                0,
                nb_filter_x,
                nb_filter_y,
            );
            // Blend the top `overlap` rows of the block's own predictor with the
            // neighbour strip (build_obmc_inter_pred_above -> aom_blend_a64_vmask).
            let plane_col = (rel_mi_col * 4) as usize;
            let dst_off = (mi_row * 4) as usize * self.stride + (mi_col * 4) as usize + plane_col;
            let mask = aom_dsp::inter::get_obmc_mask(overlap as usize);
            aom_dsp::inter::blend_a64_vmask(
                &mut self.recon,
                dst_off,
                self.stride,
                &scratch,
                0,
                bw,
                mask,
                bw,
                overlap as usize,
            );
            // Chroma above-OBMC (build_obmc_inter_pred_above, U then V). Strip MC
            // dims (decodeframe.c:726): bw_c = (op_mi_size*4)>>ss_x, bh_c =
            // clamp(bsize_high>>(ss_y+1), 4, 64>>(ss_y+1)); blended over the top
            // `overlap>>ss_y` rows with the subsampled mask. Position is the same
            // chroma column as the strip's neighbour column.
            if do_chroma {
                let overlap_c = (overlap as usize) >> ss_y; // rows blended
                let bw_c = ((op_mi_size * 4) >> ss_x) as usize;
                let bh_c = ((bsize_high >> (ss_y + 1)).clamp(4, 64 >> (ss_y + 1))) as usize;
                let (cr, cc) = clamp_mv_umv_border_px(
                    nb.mv0_row,
                    nb.mv0_col,
                    bw_c as i32,
                    bh_c as i32,
                    nb_mb_to_left,
                    nb_mb_to_right,
                    nb_mb_to_top,
                    nb_mb_to_bottom,
                    ss_x,
                    ss_y,
                );
                let uv_col = ((above_mi_col * 4) >> ss_x) as usize;
                let uv_row = ((mi_row * 4) >> ss_y) as usize;
                let dst_off_c = uv_row * self.stride_uv + uv_col;
                let mask_c = aom_dsp::inter::get_obmc_mask(overlap_c);
                for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                    let mut scratch_c = vec![0u16; bw_c * bh_c];
                    aom_dsp::inter::build_inter_predictor(
                        src,
                        last.stride_uv,
                        last.width_uv,
                        last.height_uv,
                        &mut scratch_c,
                        0,
                        bw_c,
                        uv_col,
                        uv_row,
                        bw_c,
                        bh_c,
                        cr,
                        cc,
                        ss_x,
                        ss_y,
                        nb_filter_x,
                        nb_filter_y,
                    );
                    aom_dsp::inter::blend_a64_vmask(
                        dst, dst_off_c, self.stride_uv, &scratch_c, 0, bw_c, mask_c, bw_c,
                        overlap_c,
                    );
                }
            }
        }
    }

    /// `dec_build_prediction_by_left_preds` + `av1_build_obmc_inter_prediction`
    /// (LEFT half), luma: the left-column twin of [`Self::obmc_above_blend`],
    /// feather-blending with the horizontal OBMC mask. INERT for the chunk-4
    /// target (the OBMC block is at the frame's left edge -> no left neighbour),
    /// but faithful for a future left-OBMC target.
    fn obmc_left_blend(&mut self, mi_row: i32, mi_col: i32, bsize: usize, inter: &InterFrameCfg) {
        let (ss_x, ss_y) = (self.cfg.subsampling_x, self.cfg.subsampling_y);
        let nb_max = MAX_NEIGHBOR_OBMC[mi_size_high_log2(bsize)];
        let neighbours = self.overlappable_left(mi_row, mi_col, bsize, nb_max);
        if neighbours.is_empty() {
            return;
        }
        // Chroma LEFT-OBMC always fires for a non-mono block (av1_skip_u4x4_pred_in
        // _obmc skips only the ABOVE direction, dir==0). Added per-plane below.
        let do_chroma = !self.cfg.monochrome;
        let bsize_wide = BLOCK_SIZE_WIDE[bsize];
        let overlap = bsize_wide.min(64) >> 1; // luma overlap cols
        let height = MI_SIZE_HIGH[bsize];
        let this_width = MI_SIZE_WIDE[bsize] * 4;
        let pred_width = (this_width / 2).min(32);
        let block_mb_to_bottom = (self.cfg.mi_rows - height - mi_row) * 32;
        let block_mb_to_right = (self.cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let nb_mb_to_right = block_mb_to_right + (this_width - pred_width) * 8;
        let nb_mb_to_left = -(mi_col * 32);
        let last = inter.last;
        for (rel_mi_row, op_mi_size, nb, nb_if) in neighbours {
            // Strip MC uses the NEIGHBOUR's coded interp filter.
            let (nb_filter_y, nb_filter_x) = (nb_if.0 as usize, nb_if.1 as usize);
            let left_mi_row = mi_row + rel_mi_row;
            let bw = (bsize_wide >> 1).clamp(4, 32) as usize;
            let bh = (op_mi_size * 4) as usize;
            let nb_mb_to_top = -(left_mi_row * 32);
            let nb_mb_to_bottom = block_mb_to_bottom + (height - rel_mi_row - op_mi_size) * 32;
            let (nmv_r, nmv_c) = clamp_mv_umv_border_px(
                nb.mv0_row,
                nb.mv0_col,
                bw as i32,
                bh as i32,
                nb_mb_to_left,
                nb_mb_to_right,
                nb_mb_to_top,
                nb_mb_to_bottom,
                0,
                0,
            );
            let mut scratch = vec![0u16; bw * bh];
            aom_dsp::inter::build_inter_predictor(
                &last.y,
                last.stride,
                last.width,
                last.height,
                &mut scratch,
                0,
                bw,
                (mi_col * 4) as usize,
                (left_mi_row * 4) as usize,
                bw,
                bh,
                nmv_r,
                nmv_c,
                0,
                0,
                nb_filter_x,
                nb_filter_y,
            );
            let plane_row = (rel_mi_row * 4) as usize;
            let dst_off = ((mi_row * 4) as usize + plane_row) * self.stride + (mi_col * 4) as usize;
            let mask = aom_dsp::inter::get_obmc_mask(overlap as usize);
            aom_dsp::inter::blend_a64_hmask(
                &mut self.recon,
                dst_off,
                self.stride,
                &scratch,
                0,
                bw,
                mask,
                overlap as usize,
                bh,
            );
            // Chroma left-OBMC (build_obmc_inter_pred_left, U then V). Strip MC dims
            // (decodeframe.c:781): bw_c = clamp(bsize_wide>>(ss_x+1), 4, 64>>(ss_x+1)),
            // bh_c = (op_mi_size*4)>>ss_y; blended over the left `overlap>>ss_x` cols
            // (a SUBSET of the >=4-wide strip for small chroma) with the subsampled
            // mask. Never skipped (dir==1).
            if do_chroma {
                let overlap_c = (overlap as usize) >> ss_x; // cols blended
                let bw_c = ((bsize_wide >> (ss_x + 1)).clamp(4, 64 >> (ss_x + 1))) as usize;
                let bh_c = ((op_mi_size * 4) >> ss_y) as usize;
                let (cr, cc) = clamp_mv_umv_border_px(
                    nb.mv0_row,
                    nb.mv0_col,
                    bw_c as i32,
                    bh_c as i32,
                    nb_mb_to_left,
                    nb_mb_to_right,
                    nb_mb_to_top,
                    nb_mb_to_bottom,
                    ss_x,
                    ss_y,
                );
                let uv_col = ((mi_col * 4) >> ss_x) as usize;
                let uv_row = ((left_mi_row * 4) >> ss_y) as usize;
                let dst_off_c = uv_row * self.stride_uv + uv_col;
                let mask_c = aom_dsp::inter::get_obmc_mask(overlap_c);
                for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                    let mut scratch_c = vec![0u16; bw_c * bh_c];
                    aom_dsp::inter::build_inter_predictor(
                        src,
                        last.stride_uv,
                        last.width_uv,
                        last.height_uv,
                        &mut scratch_c,
                        0,
                        bw_c,
                        uv_col,
                        uv_row,
                        bw_c,
                        bh_c,
                        cr,
                        cc,
                        ss_x,
                        ss_y,
                        nb_filter_x,
                        nb_filter_y,
                    );
                    aom_dsp::inter::blend_a64_hmask(
                        dst, dst_off_c, self.stride_uv, &scratch_c, 0, bw_c, mask_c, overlap_c,
                        bh_c,
                    );
                }
            }
        }
    }

    /// One leaf block: `parse_decode_block` (mode info + tx sizing + skip
    /// entropy-reset) followed by the intra `decode_token_recon_block` txb loop.
    /// The INTER-frame single-block mode-info + motion-compensation path,
    /// mirroring `read_inter_frame_mode_info` + `read_inter_block_mode_info`
    /// (decodemv.c) then `dec_build_inter_predictors` (decodeframe.c). The
    /// walking-skeleton envelope was single LAST reference, `SINGLE_REFERENCE`,
    /// `SIMPLE_TRANSLATION`, `skip = 1`; chunk 4 adds `OBMC_CAUSAL` (motion_mode
    /// read + above/left neighbour feather-blend) and inter var-tx
    /// (`TX_MODE_SELECT`). The pre-mode reads that are no-ops in this envelope
    /// (segment_id, skip_mode, cdef-for-skip, delta-q) are asserted off.
    #[allow(clippy::too_many_arguments)]
    fn decode_block_inter(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
        inter: &InterFrameCfg,
    ) {
        use aom_dsp::entropy::partition as ep;
        // PREDICTION_MODE inter values (enums.h).
        const NEARESTMV: i32 = 13;
        const NEARMV: i32 = 14;
        const GLOBALMV: i32 = 15;
        const NEWMV: i32 = 16;
        const SWITCHABLE: i32 = 4;

        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let cols = cfg.mi_cols;
        let up_available = mi_row > self.tile.mi_row_start;
        let left_available = mi_col > self.tile.mi_col_start;

        // Envelope invariants (STEP-0 census): these pre-mode reads are inert.
        // Out-of-envelope inter features are rejected as a clean error (not a
        // panic) so a malformed / unsupported inter frame from untrusted input
        // returns `Err` instead of aborting the decode.
        if cfg.seg.enabled {
            self.mark_corrupt("inter: segmentation not supported in this decode envelope");
            return;
        }
        if inter.skip_mode_allowed {
            self.mark_corrupt("inter: skip_mode not supported in this decode envelope");
            return;
        }
        if cfg.delta_q_present {
            self.mark_corrupt("inter: delta-q not supported in this decode envelope");
            return;
        }
        // tx_mode is TX_MODE_SELECT for the OBMC target (av1-1-b8-01-size-16x18):
        // inter blocks code a var-tx quadtree (read_tx_size_vartx). The earlier
        // walking-skeleton / ratchet targets were TX_MODE_LARGEST; both are
        // handled below (the var-tx read collapses to the single largest tx when
        // the frame codes LARGEST).
        if !matches!(cfg.tx_mode, TxMode::Largest | TxMode::Select) {
            self.mark_corrupt(format!("inter: unsupported tx_mode {:?}", cfg.tx_mode));
            return;
        }

        // Neighbour projections for the mode-info contexts.
        let (above_mi, left_mi) = self.neighbours(mi_row, mi_col);
        let above_dv = up_available.then(|| self.mi_dv[((mi_row - 1) * cols + mi_col) as usize]);
        let left_dv = left_available.then(|| self.mi_dv[(mi_row * cols + mi_col - 1) as usize]);
        // The neighbours' coded interp filters (parallel to `mi_dv`), for
        // `av1_get_pred_context_switchable_interp`. Gated by the same edge
        // availability as `above_dv`/`left_dv`, so they zip cleanly below.
        let above_if =
            up_available.then(|| self.mi_interp[((mi_row - 1) * cols + mi_col) as usize]);
        let left_if = left_available.then(|| self.mi_interp[(mi_row * cols + mi_col - 1) as usize]);
        let dv_inter = |d: DvNbr| d.use_intrabc || d.ref_frame0 > 0;

        // Snapshot the tile's persistent inter CDFs; every read below adapts this
        // local copy (via `read_symbol`/`update_cdf` when `dec.allow_update_cdf`),
        // and it is persisted back to `self.inter_cdfs` at the end of the block so
        // the next inter block sees the adapted state (matching C's in-place
        // `tile_data->tctx` adaptation).
        let mut icdfs = self.inter_cdfs;

        // --- read_inter_frame_mode_info pre-mode reads ---
        // segment_id (seg off -> 0); skip_mode (allowed off -> 0): no reads.
        // read_skip_txfm.
        let skip_ctx = ep::skip_txfm_context(
            above_mi.map_or(0, |m| m.skip_txfm),
            left_mi.map_or(0, |m| m.skip_txfm),
        ) as usize;
        let skip = ep::read_skip(dec, &mut cdfs.skip[skip_ctx], false);
        // read_cdef (decodemv.c, ordered after read_skip, before read_delta_q /
        // read_is_inter): the FIRST non-skip block in each 64x64 CDEF unit reads
        // that unit's `cdef_bits`-wide strength literal; skip blocks and
        // already-read units read nothing. This is a real entropy read on any
        // CDEF-enabled frame (`cdef_bits > 0`) — omitting it desyncs the
        // arithmetic decoder for every following symbol. (The earlier envelope
        // targets — 16x18, 64x66 — had `cdef_bits == 0` or a skip mi(0,0), so the
        // gap was inert; `av1-1-b8-01-size-16x66` frame 1 has `cdef_bits == 1`
        // with a non-skip NEWMV mi(0,0), so the missing read shifted its `read_mv`
        // and mv desynced to (0,-15) vs C's (-1,-7).)
        let coded_lossless = self.st.coded_lossless;
        let allow_intrabc = self.st.allow_intrabc;
        let mib_size = self.st.mib_size;
        let sb_size = self.st.sb_size;
        let cdef_bits = self.st.cdef_bits;
        let cdef_strength = ep::read_cdef(
            dec,
            coded_lossless,
            allow_intrabc,
            mi_row,
            mi_col,
            mib_size,
            sb_size,
            skip,
            &mut self.st.cdef_transmitted,
            cdef_bits,
        );
        // read_delta_q_params: delta_q_present off -> no read.

        // read_is_inter_block.
        let ii_ctx = ep::get_intra_inter_context(
            up_available,
            above_dv.is_some_and(dv_inter),
            left_available,
            left_dv.is_some_and(dv_inter),
        ) as usize;
        let is_inter = ep::read_is_inter(dec, &mut icdfs.intra_inter[ii_ctx], false, false);

        // --- is_inter == 0: an INTRA-coded block inside this inter frame
        // (`read_inter_frame_mode_info`'s else-arm, decodemv.c:1550). Everything
        // from here on is the EXISTING byte-exact KEY-frame intra decode: the
        // mode-info tail (shared with `read_mb_modes_kf_fc`, differing only in
        // the Y-mode CDF) then `decode_intra_block_body` (shared verbatim — C
        // installs one frame-type-independent intra recon visitor pair,
        // decodeframe.c:2756/:2761). The block reads `ref_frame = [INTRA_FRAME,
        // NONE_FRAME]`, so `is_inter_block` is false for it and every downstream
        // gate (tx-size's `inter_block_tx`, `av1_read_tx_type`'s `inter_block`,
        // `decode_token_recon_block`'s branch) takes its intra arm unchanged.
        if is_inter == 0 {
            // Persist the inter CDFs adapted so far — `read_is_inter` mutated
            // `intra_inter[ii_ctx]`, and the intra path below never touches
            // `icdfs` again except for the Y-mode row (written back inline).
            let chroma_ref = !cfg.monochrome && is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y);
            let cfl_allowed =
                !cfg.monochrome && is_cfl_allowed(bsize, self.st.coded_lossless, ss_x, ss_y);
            let (above_palette, left_palette) = self.palette_neighbours(mi_row, mi_col);
            self.st.mi_row = mi_row;
            self.st.mi_col = mi_col;
            self.st.bsize = bsize;
            self.st.is_chroma_ref = chroma_ref;
            self.st.cfl_allowed = cfl_allowed;
            self.st.mb_to_top_edge = -(mi_row * 32);
            self.st.has_above = up_available;
            self.st.has_left = left_available;
            self.st.allow_palette = av1_allow_palette(cfg.allow_screen_content_tools, bsize);
            // The prefix `read_inter_frame_mode_info` already consumed
            // (segmentation / delta-q / skip-mode are asserted off above, so
            // segment_id is 0 and the delta-lf carry is the frame default).
            let mut info = MbModeInfoKf {
                segment_id: 0,
                skip,
                cdef_strength,
                current_qindex: cfg.base_qindex,
                delta_lf: [0; 4],
                delta_lf_from_base: 0,
                use_intrabc: 0,
                dv_row: 0,
                dv_col: 0,
                y_mode: 0,
                angle_delta_y: 0,
                uv_mode: 0,
                cfl_alpha_idx: 0,
                cfl_joint_sign: 0,
                angle_delta_uv: 0,
                palette_size: [0, 0],
                palette_colors: [0; 24],
                use_filter_intra: 0,
                filter_intra_mode: 0,
            };
            // `y_mode_cdf[size_group_lookup[bsize]]` (decodemv.c:1077). Copied
            // out and written back so the adaptation persists across blocks,
            // exactly as C adapts `ec_ctx->y_mode_cdf` in place.
            let grp = ep::y_mode_size_group(bsize);
            let mut y_cdf = icdfs.y_mode[grp];
            ep::read_intra_block_mode_info_fc(
                dec,
                cdfs,
                &mut self.st,
                &mut y_cdf,
                cfg.enable_filter_intra,
                above_mi.is_some(),
                left_mi.is_some(),
                above_palette,
                left_palette,
                &mut info,
            );
            icdfs.y_mode[grp] = y_cdf;
            self.inter_cdfs = icdfs;
            self.decode_intra_block_body(
                dec,
                cdfs,
                mi_row,
                mi_col,
                bsize,
                partition,
                info,
                chroma_ref,
                up_available,
                left_available,
                above_mi,
                left_mi,
            );
            return;
        }

        // --- read_inter_block_mode_info (single reference) ---
        let rc = ep::collect_neighbors_ref_counts(
            up_available,
            above_dv.is_some_and(|d| d.use_intrabc),
            above_dv.map_or(0, |d| d.ref_frame0),
            above_dv.map_or(-1, |d| d.ref_frame1),
            left_available,
            left_dv.is_some_and(|d| d.use_intrabc),
            left_dv.map_or(0, |d| d.ref_frame0),
            left_dv.map_or(-1, |d| d.ref_frame1),
        );
        let mut ref_cdfs = icdfs.ref_frame_cdfs(&rc);
        let (is_compound, _crt, ref0, ref1) = ep::read_ref_frames(
            dec,
            &mut ref_cdfs,
            false,
            false,
            inter.reference_mode_select,
            false,
        );
        // `ref_frame_cdfs` assembled `ref_cdfs` from disjoint `single_ref` rows
        // (each single-ref sub-tree at its own pred context); copy the adapted
        // rows back so the adaptation persists across blocks. Only the rows
        // `read_ref_frames` actually read changed; copying all six is a no-op for
        // the rest. (Compound slots 0..10 are never read under SINGLE_REFERENCE.)
        icdfs.single_ref[ep::single_ref_p1_context(&rc) as usize][0] = ref_cdfs[10];
        icdfs.single_ref[ep::pred_ctx_brfarf2_or_arf(&rc) as usize][1] = ref_cdfs[11];
        icdfs.single_ref[ep::pred_ctx_ll2_or_l3gld(&rc) as usize][2] = ref_cdfs[12];
        icdfs.single_ref[ep::pred_ctx_last_or_last2(&rc) as usize][3] = ref_cdfs[13];
        icdfs.single_ref[ep::pred_ctx_last3_or_gld(&rc) as usize][4] = ref_cdfs[14];
        icdfs.single_ref[ep::pred_ctx_brf_or_arf2(&rc) as usize][5] = ref_cdfs[15];
        if is_compound || ref0 != 1 || ref1 != -1 {
            self.mark_corrupt(
                "inter: only a single LAST reference is decoded in this envelope \
                 (compound / non-LAST references unsupported)",
            );
            return;
        }
        // find_inter_mv_refs below hardcodes IDENTITY global motion (base MV (0,0),
        // gm_type 0). A frame whose reference carries non-identity global motion
        // (`global_motion[ref].wmtype > IDENTITY`) needs the real global-MV base +
        // is_global_mv_block gating — a later chunk. Guarded so such a frame pins
        // cleanly here rather than reading a wrong NEWMV base and desyncing (every
        // target through 16x18 is identity-GM; e.g. 16x66 uses global motion).
        if inter.gm_wmtype[ref0 as usize] != 0 {
            self.mark_corrupt(format!(
                "inter: non-identity global motion not supported (ref {ref0} wmtype {})",
                inter.gm_wmtype[ref0 as usize]
            ));
            return;
        }

        // find_inter_mv_refs (identity GM, empty temporal field per the census).
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
        let imv = find_inter_mv_refs(
            ref0,
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
            inter.allow_ref_frame_mvs,
            (0, 0),
            0,
            [0i8; 8],
            inter.allow_high_precision_mv,
            inter.cur_frame_force_integer_mv,
            grid,
        );

        // read_inter_mode (single-ref: mode_context passes through verbatim).
        let mode = ep::read_inter_mode(
            dec,
            &mut icdfs.newmv,
            &mut icdfs.zeromv,
            &mut icdfs.refmv,
            imv.mode_context,
        );
        // read_drl_idx: weights as u16 (values are well under 2^16, see dv_ref).
        let weights_u16: [u16; 8] = std::array::from_fn(|i| imv.weight[i] as u16);
        let ref_mv_idx = ep::read_drl_idx(
            dec,
            &mut icdfs.drl,
            mode,
            imv.ref_mv_count as i32,
            &weights_u16,
        );

        // assign_mv: resolve the predictor per mode, then read the MV.
        let precision = if inter.cur_frame_force_integer_mv {
            -1
        } else if inter.allow_high_precision_mv {
            1
        } else {
            0
        };
        let (mv_row, mv_col) = match mode {
            NEWMV => {
                // ref_mv[0] = nearest, or stack[ref_mv_idx] when the list has >1.
                let base = if imv.ref_mv_count > 1 {
                    imv.stack[ref_mv_idx as usize]
                } else {
                    imv.nearest
                };
                let [c0, c1] = &mut icdfs.nmv_comps;
                let (dr, dc) = ep::read_mv(dec, &mut icdfs.nmv_joints, c0, c1, precision);
                (base.0 + dr, base.1 + dc)
            }
            NEARESTMV => imv.nearest,
            NEARMV => {
                if ref_mv_idx > 0 {
                    imv.stack[(1 + ref_mv_idx) as usize]
                } else {
                    imv.near
                }
            }
            GLOBALMV => (0, 0), // identity global motion (census: all IDENTITY)
            _ => {
                self.mark_corrupt(format!("inter: unsupported single-ref mode {mode}"));
                return;
            }
        };

        // Inter-intra (decodemv.c:1383-1407 — AFTER assign_mv, BEFORE findSamples /
        // read_motion_mode): the flag, then (when set) the mode and the optional
        // wedge flag + shape index. Coded for an interintra-allowed single-ref
        // inter block when `enable_interintra_compound`. `av1_is_wedge_used` is
        // true for EVERY interintra-allowed bsize (BLOCK_8X8..BLOCK_32X32 all have
        // 16 wedge types), so the wedge flag is always read when the mode is —
        // carried as a gate anyway, matching C.
        let mut ref1 = ref1;
        let (interintra, ii_mode, ii_use_wedge, ii_wedge_idx) = if inter
            .enable_interintra_compound
            && is_interintra_allowed(bsize, mode, ref0, ref1)
        {
            let grp = SIZE_GROUP_LOOKUP[bsize];
            // Four DISJOINT fields of `icdfs`, so the borrows do not conflict.
            let r = ep::read_interintra_info(
                dec,
                true,
                &mut icdfs.interintra[grp],
                &mut icdfs.interintra_mode[grp],
                aom_dsp::inter::interintra::is_wedge_used(bsize),
                &mut icdfs.wedge_interintra[bsize],
                &mut icdfs.wedge_idx[bsize],
            );
            if r.0 != 0 {
                // `mbmi->ref_frame[1] = INTRA_FRAME` (decodemv.c:1393). This is
                // load-bearing beyond bookkeeping: `av1_findSamples` counts a warp
                // sample only for a neighbour with `ref_frame[1] == NONE_FRAME`
                // (-1), so an interintra neighbour must NOT look single-ref there.
                // (`collect_neighbors_ref_counts` and the MV scan both gate on
                // `> INTRA_FRAME`, so they are unaffected either way.)
                ref1 = 0;
            }
            r
        } else {
            (0, 0, 0, 0)
        };
        let interintra = interintra != 0;

        // read_motion_mode (decodemv.c:1422 — BEFORE read_mb_interp_filter). A
        // symbol is read only when the frame allows switchable motion modes AND
        // (via motion_mode_allowed) the block is motion-variation-allowed
        // (min(bw,bh) >= 8) with >= 1 overlappable inter neighbour. The ceiling
        // selects the 2-symbol obmc_cdf (OBMC ceiling) or the 3-symbol
        // motion_mode_cdf (WARP ceiling); the resolved mode may still be SIMPLE.
        // The earlier targets read nothing (64x64 skeleton: no neighbours; 16x16
        // ratchet: BLOCK_16X4 fails the size gate). WARPED_CAUSAL is chunk 5.
        //
        // An INTER-INTRA block reads NO motion-mode symbol: C gates the whole read
        // on `mbmi->ref_frame[1] != INTRA_FRAME` (decodemv.c:1421) after seeding
        // `motion_mode = SIMPLE_TRANSLATION` (:1414), so inter-intra and
        // OBMC/WARPED_CAUSAL are mutually exclusive by construction.
        let motion_mode = if interintra {
            0 // SIMPLE_TRANSLATION — no symbol read
        } else if inter.switchable_motion_mode {
            let ceiling = self.motion_mode_ceiling(mi_row, mi_col, bsize, mode, ref0, ref1, inter);
            ep::read_motion_mode(
                dec,
                &mut icdfs.obmc[bsize],
                &mut icdfs.motion_mode[bsize],
                ceiling,
            )
        } else {
            0 // SIMPLE_TRANSLATION
        };
        // WARPED_CAUSAL (chunk 5): gather the warp samples (av1_findSamples),
        // select (av1_selectSamples when num_proj_ref > 1), derive the local
        // AFFINE model (av1_find_projection). C: decodemv.c:1484-1503. An invalid
        // model marks `wm.invalid` -> MC falls back to translational (allow_warp ->
        // global(identity) -> TRANSLATION_PRED). Ordering vs read_mb_interp_filter is
        // moot: find_projection reads NO entropy symbols. `warp_luma` (a usable local
        // model) drives the affine MC below; luma always passes the per-plane >= 8
        // gate (motion-variation requires min dim >= 8), chroma is re-gated at its MC.
        let warp_params: Option<aom_dsp::inter::warp::WarpedMotionParams> = if motion_mode == 2 {
            let warp_grid = MiDvGrid {
                mi_dv: &self.mi_dv,
                cols: cfg.mi_cols,
                rows: cfg.mi_rows,
                mi_row,
                mi_col,
            };
            let mut ws = find_samples(
                &warp_grid,
                &dv_tile,
                mib_size,
                cfg.mi_rows,
                cfg.mi_cols,
                mi_row,
                mi_col,
                MI_SIZE_WIDE[bsize],
                MI_SIZE_HIGH[bsize],
                partition,
                ref0,
                up_available,
                left_available,
            );
            if ws.np > 1 {
                select_samples(
                    &mut ws,
                    mv_row,
                    mv_col,
                    BLOCK_SIZE_WIDE[bsize],
                    BLOCK_SIZE_HIGH[bsize],
                );
            }
            let mut wm = aom_dsp::inter::warp::WarpedMotionParams {
                wmtype: aom_dsp::inter::warp::AFFINE,
                ..Default::default()
            };
            let invalid = aom_dsp::inter::warp::find_projection(
                ws.np,
                &ws.pts,
                &ws.pts_inref,
                BLOCK_SIZE_WIDE[bsize],
                BLOCK_SIZE_HIGH[bsize],
                mv_row,
                mv_col,
                &mut wm,
                mi_row,
                mi_col,
            ) != 0;
            wm.invalid = invalid as u8;
            Some(wm)
        } else {
            None
        };
        let warp_luma = warp_params.filter(|w| w.invalid == 0);

        // read_mb_interp_filter (decodemv.c:1481 — AFTER motion_mode). av1_is_interp_needed
        // (reconinter.h:420) gates on motion_mode != WARPED_CAUSAL: a WARPED_CAUSAL block
        // reads NO interp symbol (set_default_interp_filters). WARPED_CAUSAL is guarded off
        // above, so interp_needed holds here; the switchable per-direction neighbour-context
        // read (chunk 6) follows.
        let interp_needed = motion_mode != 2; // 2 = WARPED_CAUSAL
        let is_switchable = inter.interp_filter == SWITCHABLE;
        // av1_extract_interp_filter: dir 0 = y_filter (vertical), dir 1 = x_filter.
        let (filter_y, filter_x) = if !interp_needed {
            // set_default_interp_filters -> av1_broadcast_interp_filter(
            // av1_unswitchable_filter(frame_filter)): EIGHTTAP_REGULAR (0) for a
            // SWITCHABLE frame, else the frame's fixed filter. No symbol is read.
            let f = if is_switchable {
                0
            } else {
                inter.interp_filter as usize
            };
            (f, f)
        } else if is_switchable {
            // A switchable frame reads a per-direction filter on the CDF context
            // selected from the neighbour filters
            // (av1_get_pred_context_switchable_interp). The neighbour projections
            // are Some((ref_frame0, ref_frame1, y_filter, x_filter)) when the edge
            // is available (C's xd->mi[-1] / xd->mi[-mi_stride]); get_ref_filter_type
            // returns the neighbour's coded filter only when it references the
            // current block's ref, else SWITCHABLE_FILTERS. With no neighbours the
            // context is (dir==0 ? 3 : 11), matching the single-block skeleton.
            let above_flt = above_dv
                .zip(above_if)
                .map(|(d, (yf, xf))| (d.ref_frame0, d.ref_frame1, yf as usize, xf as usize));
            let left_flt = left_dv
                .zip(left_if)
                .map(|(d, (yf, xf))| (d.ref_frame0, d.ref_frame1, yf as usize, xf as usize));
            let cur_is_compound = ref1 > 0; // ref_frame[1] > INTRA_FRAME
            let ctx0 = ep::get_pred_context_switchable_interp(
                0,
                ref0,
                cur_is_compound,
                above_flt,
                left_flt,
            ) as usize;
            let ctx1 = ep::get_pred_context_switchable_interp(
                1,
                ref0,
                cur_is_compound,
                above_flt,
                left_flt,
            ) as usize;
            // ctx0 (dir 0) and ctx1 (dir 1) always differ by the 8-wide dir offset
            // -> distinct CDF rows, no aliasing. Non-dual reads only ctx0.
            let mut c0 = icdfs.switchable_interp[ctx0];
            let mut c1 = icdfs.switchable_interp[ctx1];
            let (f0, f1) = ep::read_mb_interp_filter(
                dec,
                &mut c0,
                &mut c1,
                true,
                true,
                inter.enable_dual_filter,
            );
            icdfs.switchable_interp[ctx0] = c0;
            icdfs.switchable_interp[ctx1] = c1;
            (f0 as usize, f1 as usize)
        } else {
            // av1_broadcast_interp_filter(frame_filter): the fixed frame filter.
            (inter.interp_filter as usize, inter.interp_filter as usize)
        };

        // tx_size (decodeframe.c:1179-1198, inter path): TX_MODE_SELECT + a
        // signalling block (bsize > BLOCK_4X4) + !skip -> the inter var-tx
        // quadtree (read_tx_size_vartx over txfm_partition_cdf; it stamps the
        // txfm-context arrays itself). Else the tx_mode fallback (TX_MODE_LARGEST
        // for the 64x64 / 16x16 / 64x66 targets — a single per-block tx, no
        // symbol). Every block in the 16x18 OBMC target is Select + skip=0 and
        // resolves to a UNIFORM leaf tiling (the strips stay TX_4X16; the OBMC
        // BLOCK_16X8 splits to 2x TX_8X8) — a non-uniform partition would need the
        // reconstruction-phase leaf walk (collect_vartx_leaves) and is guarded.
        let a_off = mi_col as usize;
        let l_off = (mi_row & 31) as usize;
        let bw4 = MI_SIZE_WIDE[bsize] as usize;
        let bh4 = MI_SIZE_HIGH[bsize] as usize;
        let mut vartx_leaf_grid: Vec<u8> = Vec::new();
        let mut vartx_non_uniform = false;
        let tx_size = if cfg.tx_mode == TxMode::Select && bsize > 0 && skip == 0 {
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
            vartx_leaf_grid = vec![0u8; bw4 * bh4];
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
                        &mut vartx_leaf_grid,
                        bw4,
                        bh4,
                    );
                    idx += bw_u;
                }
                idy += bh_u;
            }
            vartx_non_uniform = non_uniform;
            vartx_tx
        } else {
            // read_tx_size fallback (decodeframe.c:1179-1198): a SKIP inter block
            // (allow_select = !skip = false) or a non-SELECT frame reads NO tx-size
            // symbol and derives `tx_size_from_tx_mode`, then MUST stamp the
            // txfm-context arrays via `set_txfm_ctxs(tx, bw, bh, skip && is_inter)`.
            // The var-tx quadtree above stamps itself; this else-arm did not, so a
            // skip block left `above_t`/`left_t` at the init 64. A later var-tx
            // block whose above/left neighbour is that skip block then read a wrong
            // `txfm_partition` context whenever the true stamp (block width/height
            // px) straddles the tx dim — the q63 F1 desync (mi(16,0) skip
            // BLOCK_16X64 → above_t 16 in C vs 64 in the port; mi(32,0)'s TX_32X64
            // read saw above=1 in C, 0 in the port).
            let ts = tx_size_from_tx_mode(bsize, cfg.tx_mode);
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                ts,
                bw4,
                bh4,
                skip != 0,
            );
            ts
        };
        // The leaf grid drives the non-uniform reconstruction walk
        // (collect_vartx_leaves); every block in this target is uniform (guarded),
        // so the uniform residual loop below tiles with `tx_size` = the read size.
        let _ = &vartx_leaf_grid;
        if vartx_non_uniform {
            self.mark_corrupt(format!(
                "inter: non-uniform var-tx not supported (block {bsize} @ mi({mi_row},{mi_col}))"
            ));
            return;
        }

        // --- parse_decode_block tail (decodeframe.c:1219): a SKIP block resets
        // its entropy-context footprint to zero (`av1_reset_entropy_context`,
        // blockd.c:58) — plane 0 always, chroma planes when this block is the
        // chroma reference, each over its own plane_bsize footprint. This is NOT
        // intra-specific: C runs it for every skipped block before
        // `decode_token_recon_block`, and a skip block reads (and therefore
        // stamps) no coefficients, so without the reset the footprint keeps the
        // stale culs of whatever block last occupied those context cells.
        //
        // The port had this only on the intra path. It stayed invisible while the
        // probe pinned before any block could read across a skipped inter
        // neighbour's stale cells; the first real victim is mi(44,18) (the first
        // intra-in-inter block), whose LEFT neighbour mi(44,16) is a skipped inter
        // block: its stale non-zero culs flip mi(44,18)'s `txb_skip_ctx`, so all
        // three of its txbs still decode all-zero (the same symbol VALUES C reads)
        // but off a different `txb_skip_cdf` row — the arithmetic decoder drifts
        // and the next block's `skip_txfm` reads 0 where C reads 1.
        if skip != 0 {
            let a0 = mi_col as usize;
            let l0 = (mi_row & 31) as usize;
            self.above_e[0][a0..a0 + bw4].fill(0);
            self.left_e[0][l0..l0 + bh4].fill(0);
            let reset_chroma =
                !cfg.monochrome && is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y);
            if reset_chroma {
                // Same chroma-reference origin shift the coefficient loop uses
                // (setup_pred_plane's odd-position adjustment).
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
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let uw = MI_SIZE_WIDE[plane_bsize] as usize;
                let uh = MI_SIZE_HIGH[plane_bsize] as usize;
                let uv_a_base = (adj_col >> ss_x) as usize;
                let uv_l_base = ((adj_row & 31) >> ss_y) as usize;
                for plane in 1..=2 {
                    self.above_e[plane][uv_a_base..uv_a_base + uw].fill(0);
                    self.left_e[plane][uv_l_base..uv_l_base + uh].fill(0);
                }
            }
        }

        // --- motion compensation (predict phase; NO entropy reads) ---
        let (cmv_row, cmv_col) = clamp_mv_to_umv_border(mv_row, mv_col, mi_row, mi_col, bsize, cfg);
        let bw_px = (MI_SIZE_WIDE[bsize] * 4) as usize;
        let bh_px = (MI_SIZE_HIGH[bsize] * 4) as usize;
        let last = inter.last;
        let blk_x = (mi_col * 4) as usize;
        let blk_y = (mi_row * 4) as usize;
        let dst_off = blk_y * self.stride + blk_x;
        // Luma MC: WARPED_CAUSAL affine warp (av1_warp_plane) with a valid local
        // model; else translational. OBMC uses translational here (its overlap
        // blend runs after, below). Luma block is always >= 8 -> passes
        // av1_init_warp_params' per-plane size gate.
        if let Some(wm) = warp_luma {
            aom_dsp::inter::warp::warp_affine(
                &wm.wmmat,
                &last.y,
                last.width,
                last.height,
                last.stride,
                &mut self.recon,
                dst_off,
                self.stride,
                blk_x as i32,
                blk_y as i32,
                bw_px,
                bh_px,
                0,
                0,
                wm.alpha,
                wm.beta,
                wm.gamma,
                wm.delta,
            );
        } else {
            aom_dsp::inter::build_inter_predictor(
                &last.y,
                last.stride,
                last.width,
                last.height,
                &mut self.recon,
                dst_off,
                self.stride,
                blk_x,
                blk_y,
                bw_px,
                bh_px,
                cmv_row,
                cmv_col,
                0,
                0,
                filter_x,
                filter_y,
            );
        }
        // Chroma prediction only at the chroma-reference block (sub-8x8 members
        // share one chroma block, coded on the group's bottom/right member). The
        // shared-group origin is `adj` (setup_pred_plane's odd-position shift).
        let chroma_ref = !cfg.monochrome && is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y);
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
        if chroma_ref {
            let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
            let uv_org_x = ((adj_col * 4) >> ss_x) as usize;
            let uv_org_y = ((adj_row * 4) >> ss_y) as usize;
            // is_sub8x8_inter (chroma): a luma side == 4 on a subsampled axis means
            // the chroma covers > 1 luma block; predict per covered luma block using
            // its OWN MV (build_inter_predictors_sub8x8). Every covered block is
            // inter here (all-inter frame). Otherwise a single whole-block chroma MC.
            let is_sub4_x = BLOCK_SIZE_WIDE[bsize] == 4 && ss_x != 0;
            let is_sub4_y = BLOCK_SIZE_HIGH[bsize] == 4 && ss_y != 0;
            if is_sub4_x || is_sub4_y {
                let b4_w = (BLOCK_SIZE_WIDE[bsize] >> ss_x) as usize;
                let b4_h = (BLOCK_SIZE_HIGH[bsize] >> ss_y) as usize;
                let b8_w = BLOCK_SIZE_WIDE[plane_bsize] as usize;
                let b8_h = BLOCK_SIZE_HIGH[plane_bsize] as usize;
                let row_start: i32 = if is_sub4_y { -1 } else { 0 };
                let col_start: i32 = if is_sub4_x { -1 } else { 0 };
                let cols = cfg.mi_cols;
                let mut y = 0usize;
                let mut row = row_start;
                while y < b8_h {
                    let mut x = 0usize;
                    let mut col = col_start;
                    while x < b8_w {
                        // The covered sub-block's own MV: xd->mi[row*stride+col].
                        // The current block (row==0 && col==0) is not yet stamped
                        // into `mi_dv` (stamp is at the end of this fn), so use the
                        // just-decoded MV; neighbours come from the grid.
                        let (smv_r, smv_c) = if row == 0 && col == 0 {
                            (mv_row, mv_col)
                        } else {
                            let d = self.mi_dv[((mi_row + row) * cols + (mi_col + col)) as usize];
                            (d.mv0_row, d.mv0_col)
                        };
                        // Per-plane clamp for this covered sub-block (C's sub8x8
                        // predictor runs dec_calc_subpel_params per b4 with the
                        // block's shared luma edges + the b4 dims).
                        let (smv_r, smv_c) = clamp_mv_to_umv_border_plane(
                            smv_r,
                            smv_c,
                            mi_row,
                            mi_col,
                            bsize,
                            b4_w as i32,
                            b4_h as i32,
                            ss_x,
                            ss_y,
                            &cfg,
                        );
                        let bxu = uv_org_x + x;
                        let byu = uv_org_y + y;
                        let doff = byu * self.stride_uv + bxu;
                        for (dst, src) in
                            [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)]
                        {
                            aom_dsp::inter::build_inter_predictor(
                                src,
                                last.stride_uv,
                                last.width_uv,
                                last.height_uv,
                                dst,
                                doff,
                                self.stride_uv,
                                bxu,
                                byu,
                                b4_w,
                                b4_h,
                                smv_r,
                                smv_c,
                                ss_x,
                                ss_y,
                                filter_x,
                                filter_y,
                            );
                        }
                        x += b4_w;
                        col += 1;
                    }
                    y += b4_h;
                    row += 1;
                }
            } else {
                let bw_uv = bw_px >> ss_x;
                let bh_uv = bh_px >> ss_y;
                // Per-plane MV clamp (clamp_mv_to_umv_border_sb with the CHROMA
                // subsampling): C re-clamps the RAW block MV against chroma-scaled
                // edges + the chroma prediction dims (decodeframe.c:625). Reusing
                // the luma clamp (`cmv_*`) is wrong once the clamp fires — the
                // chroma bounds differ (edges * (1<<(1-ss)), spel from `bw_uv`).
                let (cmv_row_uv, cmv_col_uv) = clamp_mv_to_umv_border_plane(
                    mv_row,
                    mv_col,
                    mi_row,
                    mi_col,
                    bsize,
                    bw_uv as i32,
                    bh_uv as i32,
                    ss_x,
                    ss_y,
                    &cfg,
                );
                let doff = uv_org_y * self.stride_uv + uv_org_x;
                // Chroma warp only when the (subsampled) plane block is >= 8 in both
                // dims (av1_init_warp_params); otherwise translational.
                let warp_chroma = warp_luma.filter(|_| bw_uv >= 8 && bh_uv >= 8);
                if let Some(wm) = warp_chroma {
                    for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                        aom_dsp::inter::warp::warp_affine(
                            &wm.wmmat,
                            src,
                            last.width_uv,
                            last.height_uv,
                            last.stride_uv,
                            dst,
                            doff,
                            self.stride_uv,
                            uv_org_x as i32,
                            uv_org_y as i32,
                            bw_uv,
                            bh_uv,
                            ss_x,
                            ss_y,
                            wm.alpha,
                            wm.beta,
                            wm.gamma,
                            wm.delta,
                        );
                    }
                } else {
                    // Translational chroma MC uses the PER-PLANE clamped MV
                    // (cmv_*_uv), not the luma-clamped cmv_* — see the
                    // clamp_mv_to_umv_border_plane call above.
                    for (dst, src) in [(&mut self.recon_u, &last.u), (&mut self.recon_v, &last.v)] {
                        aom_dsp::inter::build_inter_predictor(
                            src,
                            last.stride_uv,
                            last.width_uv,
                            last.height_uv,
                            dst,
                            doff,
                            self.stride_uv,
                            uv_org_x,
                            uv_org_y,
                            bw_uv,
                            bh_uv,
                            cmv_row_uv,
                            cmv_col_uv,
                            ss_x,
                            ss_y,
                            filter_x,
                            filter_y,
                        );
                    }
                }
            }
        }

        // --- INTER-INTRA (av1_build_interintra_predictor, reconinter.c:1162): for
        // each plane, build an intra predictor over the WHOLE plane block and blend
        // it onto that plane's freshly-built inter predictor.
        //
        // C runs this inside `dec_build_inter_predictors`' plane loop, immediately
        // after each plane's inter prediction lands in `dst` (decodeframe.c:694-702).
        // Doing all planes here instead is equivalent: the luma blend writes only
        // inside the luma block and chroma MC reads the REFERENCE frame, never the
        // current recon, so no plane's inputs depend on another plane's blend.
        //
        // Two conventions must not be crossed (they differ between the arms, see
        // `combine_interintra`): the SMOOTH mask is prebuilt at PLANE resolution and
        // blended 1:1, while the WEDGE mask is at LUMA resolution and box-averaged
        // down by `subw`/`subh` inside the blend.
        if interintra {
            let ii_intra_mode = aom_dsp::inter::interintra::INTERINTRA_TO_INTRA_MODE[ii_mode as usize];
            // `get_intra_edge_filter_type` (reconintra.c:974). Inert for every
            // inter-intra block — the four modes are DC/V/H/SMOOTH and the angle
            // delta is forced to 0 (decodemv.c:1395), so V/H predict at exactly
            // 90/180 where the edge filter is skipped — but derived faithfully.
            let is_smooth = |m: Option<MiNbrKf>| {
                m.is_some_and(|n| (SMOOTH_PRED..=SMOOTH_H_PRED).contains(&n.y_mode))
            };
            let filt_type = (is_smooth(above_mi) || is_smooth(left_mi)) as i32;

            // Luma.
            let ii_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
            let (n_top, n_tr, n_left, n_bl) = ep::intra_avail(
                self.st.sb_size,
                bsize,
                mi_row,
                mi_col,
                up_available,
                left_available,
                self.tile.mi_col_end,
                self.tile.mi_row_end,
                partition,
                ii_tx,
                0,
                0,
                0,
                0,
                BLOCK_SIZE_WIDE[bsize],
                BLOCK_SIZE_HIGH[bsize],
                cfg.mi_cols,
                cfg.mi_rows,
                ii_intra_mode,
                0,
                false,
            );
            let mut intra_pred = vec![0u16; bw_px * bh_px];
            predict_intra_high(
                &self.recon,
                dst_off,
                self.stride,
                &mut intra_pred,
                bw_px,
                ii_intra_mode,
                0,
                false,
                0,
                cfg.disable_edge_filter,
                filt_type,
                ii_tx,
                usize::try_from(n_top).expect("n_top_px must be non-negative"),
                n_tr,
                usize::try_from(n_left).expect("n_left_px must be non-negative"),
                n_bl,
                cfg.bd,
            );
            // C blends in place (`comppred == interpred`). Every output pixel is a
            // function of the SAME position in both sources, so lifting the inter
            // predictor into a scratch first is numerically identical and lets the
            // destination be borrowed mutably.
            let mut inter_pred = vec![0u16; bw_px * bh_px];
            for r in 0..bh_px {
                let s = dst_off + r * self.stride;
                inter_pred[r * bw_px..(r + 1) * bw_px].copy_from_slice(&self.recon[s..s + bw_px]);
            }
            aom_dsp::inter::interintra::combine_interintra(
                ii_mode as usize,
                ii_use_wedge != 0,
                ii_wedge_idx as usize,
                bsize,
                bsize,
                &mut self.recon[dst_off..],
                self.stride,
                &inter_pred,
                bw_px,
                &intra_pred,
                bw_px,
            );

            // Chroma. `is_chroma_ref` is always true for an inter-intra bsize
            // (>= BLOCK_8X8), so C's `plane && !is_chroma_ref` break never fires.
            if chroma_ref {
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let uv_tx = MAX_TXSIZE_RECT_LOOKUP[plane_bsize];
                let uw = BLOCK_SIZE_WIDE[plane_bsize] as usize;
                let uh = BLOCK_SIZE_HIGH[plane_bsize] as usize;
                let wpx = ((MI_SIZE_WIDE[bsize] * 4) >> ss_x).max(4);
                let hpx = ((MI_SIZE_HIGH[bsize] * 4) >> ss_y).max(4);
                let bsize_uv = scale_chroma_bsize(bsize, ss_x, ss_y);
                let up_uv = adj_row > self.tile.mi_row_start;
                let left_uv = adj_col > self.tile.mi_col_start;
                let uv_org_x = ((adj_col * 4) >> ss_x) as usize;
                let uv_org_y = ((adj_row * 4) >> ss_y) as usize;
                let uv_off = uv_org_y * self.stride_uv + uv_org_x;
                let (n_top, n_tr, n_left, n_bl) = ep::intra_avail(
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
                    0,
                    0,
                    wpx,
                    hpx,
                    cfg.mi_cols,
                    cfg.mi_rows,
                    ii_intra_mode,
                    0,
                    false,
                );
                let mut intra_uv = vec![0u16; uw * uh];
                let mut inter_uv = vec![0u16; uw * uh];
                for plane in 1..=2 {
                    let src_plane: &[u16] = if plane == 1 {
                        &self.recon_u
                    } else {
                        &self.recon_v
                    };
                    predict_intra_high(
                        src_plane,
                        uv_off,
                        self.stride_uv,
                        &mut intra_uv,
                        uw,
                        ii_intra_mode,
                        0,
                        false,
                        0,
                        cfg.disable_edge_filter,
                        filt_type,
                        uv_tx,
                        usize::try_from(n_top).expect("n_top_px must be non-negative"),
                        n_tr,
                        usize::try_from(n_left).expect("n_left_px must be non-negative"),
                        n_bl,
                        cfg.bd,
                    );
                    for r in 0..uh {
                        let s = uv_off + r * self.stride_uv;
                        inter_uv[r * uw..(r + 1) * uw]
                            .copy_from_slice(&src_plane[s..s + uw]);
                    }
                    let dst_plane: &mut [u16] = if plane == 1 {
                        &mut self.recon_u
                    } else {
                        &mut self.recon_v
                    };
                    aom_dsp::inter::interintra::combine_interintra(
                        ii_mode as usize,
                        ii_use_wedge != 0,
                        ii_wedge_idx as usize,
                        bsize,
                        plane_bsize,
                        &mut dst_plane[uv_off..],
                        self.stride_uv,
                        &inter_uv,
                        uw,
                        &intra_uv,
                        uw,
                    );
                }
            }
        }

        // --- OBMC (predict_inter_block, decodeframe.c:878): after the block's own
        // predictor is built (luma + chroma above) and BEFORE the residual add,
        // an OBMC_CAUSAL block feather-blends its predictor with predictors from
        // its overlappable above/left neighbours' motion. Non-switchable frame ->
        // every neighbour shares the frame filter (filter_x == filter_y); a
        // switchable frame would feed each neighbour's own stored filter (chunk 6).
        if motion_mode == 1 {
            self.obmc_above_blend(mi_row, mi_col, bsize, inter);
            self.obmc_left_blend(mi_row, mi_col, bsize, inter);
        }

        // --- reconstruction: read residual coefficients + ADD onto the MC
        // prediction (decode_token_recon_block inter path). Skip blocks read no
        // coeffs. tx_size is the (uniform) var-tx / LARGEST per-block luma tx
        // tiling the block, then (at the chroma reference) one U and one V tx, in
        // that plane order — all within the single <=64px 64x64 chunk the target's
        // blocks occupy.
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let mut luma_tt0 = 0usize; // co-located luma tx-type for inter chroma
        if skip == 0 {
            // av1_read_tx_type gate: !skip && qindex(seg,base) > 0. Segmentation
            // is off in this envelope (asserted above) so segment_id == 0.
            let signal_gate = av1_get_qindex(&cfg.seg, 0, cfg.base_qindex) > 0;
            let max_blocks_wide = max_block_units(BLOCK_SIZE_WIDE[bsize], mb_to_right_edge);
            let max_blocks_high = max_block_units(BLOCK_SIZE_HIGH[bsize], mb_to_bottom_edge);
            // Luma plane.
            let txw = TX_SIZE_WIDE_UNIT[tx_size];
            let txh = TX_SIZE_HIGH_UNIT[tx_size];
            let larea = txb_wide(tx_size) * txb_high(tx_size);
            let mut tcoeff = vec![0i32; larea];
            let mut blk_row = 0usize;
            while blk_row < max_blocks_high {
                let mut blk_col = 0usize;
                while blk_col < max_blocks_wide {
                    let a0 = mi_col as usize + blk_col;
                    let l0 = (mi_row & 31) as usize + blk_row;
                    let (tsc, dsc) = get_txb_ctx(
                        bsize,
                        tx_size,
                        0,
                        &self.above_e[0][a0..],
                        &self.left_e[0][l0..],
                    );
                    // is_inter block -> av1_read_tx_type selects the inter ext-tx CDF.
                    let ext = inter_ext_tx_cdf(&mut cdfs.inter_ext_tx, tx_size, cfg.reduced_tx_set);
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
                        true,
                        cfg.reduced_tx_set,
                        signal_gate,
                        0,
                    );
                    if blk_row == 0 && blk_col == 0 {
                        luma_tt0 = tt;
                    }
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
                    if eob > 0 {
                        let off = ((mi_row * 4) as usize + blk_row * 4) * self.stride
                            + (mi_col * 4) as usize
                            + blk_col * 4;
                        let iqm = qm::iqmatrix(self.block_qm_level[0], 0, tx_size, tt);
                        reconstruct_txb_into(
                            &mut self.recon[off..],
                            self.stride,
                            tx_size,
                            tt,
                            &tcoeff,
                            self.dequants[0],
                            iqm,
                            cfg.bd,
                            &mut self.recon_scratch,
                        );
                    }
                    blk_col += txw;
                }
                blk_row += txh;
            }
            // Chroma planes (only at the chroma reference block).
            if chroma_ref {
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let uv_tx = max_uv_txsize(bsize, ss_x, ss_y);
                let uv_txw = TX_SIZE_WIDE_UNIT[uv_tx];
                let uv_txh = TX_SIZE_HIGH_UNIT[uv_tx];
                let uv_area = txb_wide(uv_tx) * txb_high(uv_tx);
                let uv_a_base = (adj_col >> ss_x) as usize;
                let uv_l_base = ((adj_row & 31) >> ss_y) as usize;
                let uv_org_x = ((adj_col * 4) >> ss_x) as usize;
                let uv_org_y = ((adj_row * 4) >> ss_y) as usize;
                let blocks_wide_uv =
                    max_block_units_ss(BLOCK_SIZE_WIDE[plane_bsize], mb_to_right_edge, ss_x);
                let blocks_high_uv =
                    max_block_units_ss(BLOCK_SIZE_HIGH[plane_bsize], mb_to_bottom_edge, ss_y);
                // Inter chroma tx-type = the CO-LOCATED luma tx-type
                // (av1_get_tx_type is_inter branch), validated against the uv inter
                // ext-tx set and demoted to DCT_DCT when unused. For this uniform
                // single-tx block the co-location is the block's own luma tx-type.
                let tt_uv = if ext_tx_derive(uv_tx, true, cfg.reduced_tx_set, luma_tt0, false, 0, 0)
                    .used
                    == 1
                {
                    luma_tt0
                } else {
                    0
                };
                let mut tcoeff_uv = vec![0i32; uv_area];
                let mut no_ext: [u16; 0] = [];
                for plane in 1..=2usize {
                    let mut blk_row = 0usize;
                    while blk_row < blocks_high_uv {
                        let mut blk_col = 0usize;
                        while blk_col < blocks_wide_uv {
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
                                &mut no_ext,
                                &mut tcoeff_uv,
                                uv_tx,
                                1,
                                tsc as usize,
                                dsc as usize,
                                true,
                                false,
                                cfg.reduced_tx_set,
                                false,
                                tt_uv,
                            );
                            let cul = txb_entropy_context(&tcoeff_uv, uv_tx, tt_uv, eob) as i8;
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
                            if eob > 0 {
                                let off = (uv_org_y + blk_row * 4) * self.stride_uv
                                    + uv_org_x
                                    + blk_col * 4;
                                let iqm =
                                    qm::iqmatrix(self.block_qm_level[plane], plane, uv_tx, tt_uv);
                                let dst = if plane == 1 {
                                    &mut self.recon_u
                                } else {
                                    &mut self.recon_v
                                };
                                reconstruct_txb_into(
                                    &mut dst[off..],
                                    self.stride_uv,
                                    uv_tx,
                                    tt_uv,
                                    &tcoeff_uv,
                                    self.dequants[plane],
                                    iqm,
                                    cfg.bd,
                                    &mut self.recon_scratch,
                                );
                            }
                            blk_col += uv_txw;
                        }
                        blk_row += uv_txh;
                    }
                }
            }
        }
        // Persist the adapted inter mode-info CDFs for the next inter block.
        self.inter_cdfs = icdfs;

        // Stamp the neighbour grids (inert for a single block; needed for the
        // ratchet's spatial scan + skip/interp contexts).
        self.stamp_mi(
            mi_row,
            mi_col,
            bsize,
            MiNbrKf {
                y_mode: 0,
                skip_txfm: skip,
            },
        );
        self.stamp_dv(
            mi_row,
            mi_col,
            bsize,
            DvNbr {
                bsize,
                ref_frame0: ref0,
                ref_frame1: ref1,
                use_intrabc: false,
                mode,
                mv0_row: mv_row,
                mv0_col: mv_col,
                mv1_row: 0,
                mv1_col: 0,
            },
        );
        // Stamp the block's coded interp filters so later switchable inter blocks'
        // av1_get_pred_context_switchable_interp neighbour reads see them.
        self.stamp_interp(mi_row, mi_col, bsize, (filter_y as u8, filter_x as u8));
        // Minimal per-block record for the post-filter / output structures. The
        // inter block carries no intra fields; `skip`/`current_qindex` are what
        // the deblock reads (the mode-info's inter-ness lives in the DV grid).
        let info = MbModeInfoKf {
            segment_id: 0,
            skip,
            // The per-64x64-unit CDEF strength `read_cdef` returned for THIS block
            // (-1 = skip / unit already coded). The frame CDEF walk (apply_cdef)
            // stamps it on every covered 64x64 unit, exactly as for intra blocks.
            cdef_strength,
            current_qindex: cfg.base_qindex,
            delta_lf: [0; 4],
            delta_lf_from_base: 0,
            use_intrabc: 0,
            dv_row: 0,
            dv_col: 0,
            y_mode: 0,
            angle_delta_y: 0,
            uv_mode: 0,
            cfl_alpha_idx: 0,
            cfl_joint_sign: 0,
            angle_delta_uv: 0,
            palette_size: [0, 0],
            palette_colors: [0; 24],
            use_filter_intra: 0,
            filter_intra_mode: 0,
        };
        self.tree.push(partition as i8);
        self.blocks.push(DecodedBlockKf {
            mi_row,
            mi_col,
            bsize,
            partition,
            info,
            tx_size,
            txbs: Vec::new(),
            txbs_uv: Vec::new(),
            // Inter block: carry ref_frame[0] + mode for the loop-filter grid
            // (build_lf_inputs derives is_inter/ref/mode_lf from this).
            inter_lf: Some((ref0, mode)),
        });
    }

    fn decode_block(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
    ) {
        if self.is_corrupt() {
            return;
        }
        // INTER frame: take the motion-compensation mode-info path. `inter` is
        // `Copy`, so copying it out releases the borrow on `self`.
        if let Some(inter) = self.inter {
            self.decode_block_inter(dec, cdfs, mi_row, mi_col, bsize, partition, &inter);
            return;
        }
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
        // is_cfl_allowed narrows to BLOCK_4X4 when the block is lossless (uniform
        // across the frame in this envelope — mixed is rejected upstream).
        let cfl_allowed =
            !cfg.monochrome && is_cfl_allowed(bsize, self.st.coded_lossless, ss_x, ss_y);
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
        self.decode_intra_block_body(
            dec,
            cdfs,
            mi_row,
            mi_col,
            bsize,
            partition,
            info,
            chroma_ref,
            up_available,
            left_available,
            above,
            left,
        );
    }

    /// Everything an intra block does AFTER its mode info is read: the palette
    /// colour-map tokens, the tx-size read + txfm-context stamp, the skip
    /// entropy reset, the `decode_token_recon_block` per-txb read -> predict ->
    /// reconstruct walk, and the neighbour-grid stamps.
    ///
    /// Shared VERBATIM by both intra paths, because C shares it too: the
    /// decoder installs `read_coeffs_tx_intra_block` /
    /// `predict_and_reconstruct_intra_block` once (decodeframe.c:2756/:2761) with
    /// NO frame-type dependence, and `decode_token_recon_block` branches only on
    /// `!is_inter_block(mbmi)` (:920). An intra block in an INTER frame has
    /// `ref_frame[0] == INTRA_FRAME` and `use_intrabc == 0`, so it takes these
    /// exact arms:
    ///
    ///  * tx size: `inter_block_tx` is 0 (decodeframe.c:1179), so the var-tx
    ///    quadtree is skipped and `read_tx_size(.., is_inter = 0, ..)` runs — and
    ///    `(!is_inter || allow_select_inter)` at :1150 is unconditionally true, so
    ///    it is `read_selected_tx_size` exactly as on a KEY frame. `set_txfm_ctxs`
    ///    gets `skip_txfm && is_inter_block(mbmi)` = 0 (:1197).
    ///  * tx type: `av1_read_tx_type`'s `inter_block` is 0, so it reads
    ///    `intra_ext_tx_cdf[eset][square_tx_size][intra_dir]` (decodemv.c:665),
    ///    with `intra_dir` = `fimode_to_intradir[filter_intra_mode]` when
    ///    filter-intra is on, else the Y mode (:660-664).
    ///
    /// So the intra-in-inter block needs no special-casing here at all — which is
    /// the point of sharing this body rather than transcribing a second copy.
    #[allow(clippy::too_many_arguments)]
    fn decode_intra_block_body(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfFrameContext,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
        info: MbModeInfoKf,
        chroma_ref: bool,
        up_available: bool,
        left_available: bool,
        above: Option<MiNbrKf>,
        left: Option<MiNbrKf>,
    ) {
        let cfg = self.cfg;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
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
        // Var-tx leaf grid (C's `mbmi->inter_tx_size[]`), block-relative per-4x4,
        // filled by the size-read phase for an intrabc var-tx block. When the
        // partition is non-uniform the reconstruction phase walks it
        // (`collect_vartx_leaves`) instead of tiling with a single tx size.
        let bw4 = MI_SIZE_WIDE[bsize] as usize;
        let bh4 = MI_SIZE_HIGH[bsize] as usize;
        let mut vartx_leaf_grid: Vec<u8> = Vec::new();
        let mut vartx_non_uniform = false;
        let tx_size = if self.st.coded_lossless {
            // read_tx_size (decodeframe.c): xd->lossless[segment_id] preempts to
            // TX_4X4 before any tx-size symbol / block_signals_txsize test. The
            // else-arm of read_block_tx_size then stamps set_txfm_ctxs(TX_4X4, ...,
            // skip && is_inter_block); is_inter_block on a KEY frame is
            // use_intrabc. The var-tx quadtree is gated on !lossless in C, so it
            // never runs here.
            set_txfm_ctxs(
                &mut self.above_t[a_off..],
                &mut self.left_t[l_off..],
                TX_4X4_IDX,
                bw,
                bh,
                info.skip != 0 && info.use_intrabc != 0,
            );
            TX_4X4_IDX
        } else if info.use_intrabc != 0 {
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
                vartx_leaf_grid = vec![0u8; bw4 * bh4];
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
                            &mut vartx_leaf_grid,
                            bw4,
                            bh4,
                        );
                        idx += bw_u;
                    }
                    idy += bh_u;
                }
                // For a UNIFORM partition the coeff loop below tiles the whole
                // block with this single scalar tx size. A NON-uniform partition
                // (distinct leaf sizes) instead drives the reconstruction phase
                // through `collect_vartx_leaves` over `vartx_leaf_grid`; flag it.
                vartx_non_uniform = non_uniform;
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
                // get_tx_size_context (blockd.h) overrides the txfm-context-array
                // term with the NEIGHBOUR'S BLOCK size whenever that neighbour is
                // `is_inter_block` — which is `use_intrabc || ref_frame[0] >
                // INTRA_FRAME` (blockd.h:372). On a KEY frame only intrabc
                // qualifies (every stamp carries ref_frame0 == INTRA_FRAME), so
                // this is behaviour-identical there; inside an INTER frame an
                // intra block's neighbours are usually genuine inter blocks, and
                // missing them picks the wrong `tx_size_cdf` row — the same
                // symbol on different probabilities, which drifts the arithmetic
                // decoder and desyncs a few reads later.
                let is_inter_nbr = |d: &DvNbr| d.use_intrabc || d.ref_frame0 > 0;
                let above_inter_bsize = up_available
                    .then(|| self.mi_dv[((mi_row - 1) * cfg.mi_cols + mi_col) as usize])
                    .filter(is_inter_nbr)
                    .map(|d| d.bsize);
                let left_inter_bsize = left_available
                    .then(|| self.mi_dv[(mi_row * cfg.mi_cols + mi_col - 1) as usize])
                    .filter(is_inter_nbr)
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
        // QM level per plane for this block: only the segment's lossless status
        // varies it (qmatrix_level_* is frame-constant), so recompute only when
        // the frame uses QM. Non-QM frames keep the flat init and the pre-QM
        // (flat-dequant) path byte-for-byte.
        if cfg.using_qmatrix {
            self.block_qm_level = frame_qm_levels(cfg, info.segment_id as usize);
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

        // Non-uniform intrabc var-tx: the reconstruction phase
        // (`decode_reconstruct_tx`) walks the per-leaf partition rather than
        // tiling the block with one scalar tx size. Each leaf reads its own
        // coeffs + tx_type (inter ext-tx at the leaf size) in DFS order, then the
        // integer block copy + inverse transform, all at the leaf's size. The
        // uniform fast loop below is skipped in this case.
        let do_uniform = !(info.use_intrabc != 0 && vartx_non_uniform);
        if !do_uniform {
            let max_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
            let bw_mt = TX_SIZE_WIDE_UNIT[max_tx];
            let bh_mt = TX_SIZE_HIGH_UNIT[max_tx];
            let mut leaves: Vec<(usize, usize, usize)> = Vec::new();
            let mut r = 0;
            while r < max_blocks_high {
                let mut c = 0;
                while c < max_blocks_wide {
                    collect_vartx_leaves(
                        &vartx_leaf_grid,
                        bw4,
                        r,
                        c,
                        max_tx,
                        max_blocks_high,
                        max_blocks_wide,
                        &mut leaves,
                    );
                    c += bw_mt;
                }
                r += bh_mt;
            }
            let mut nu_tcoeff = vec![0i32; txb_wide(max_tx) * txb_high(max_tx)];
            let mut nu_scratch = vec![0u16; TX_SIZE_WIDE[max_tx] * TX_SIZE_HIGH[max_tx]];
            for &(blk_row, blk_col, cur_tx) in &leaves {
                let (ltxw, ltxh) = (TX_SIZE_WIDE_UNIT[cur_tx], TX_SIZE_HIGH_UNIT[cur_tx]);
                let (ltxwpx, ltxhpx) = (TX_SIZE_WIDE[cur_tx], TX_SIZE_HIGH[cur_tx]);
                let larea = txb_wide(cur_tx) * txb_high(cur_tx);
                let (eob, tx_type) = if info.skip == 0 {
                    let a0 = mi_col as usize + blk_col;
                    let l0 = (mi_row & 31) as usize + blk_row;
                    let (tsc, dsc) = get_txb_ctx(
                        bsize,
                        cur_tx,
                        0,
                        &self.above_e[0][a0..],
                        &self.left_e[0][l0..],
                    );
                    let ext = inter_ext_tx_cdf(&mut cdfs.inter_ext_tx, cur_tx, cfg.reduced_tx_set);
                    let (eob, tt) = read_coeffs_txb_full(
                        dec,
                        &mut cdfs.coeff,
                        ext,
                        &mut nu_tcoeff[..larea],
                        cur_tx,
                        0,
                        tsc as usize,
                        dsc as usize,
                        true,
                        true,
                        cfg.reduced_tx_set,
                        signal_gate,
                        0,
                    );
                    let cul = txb_entropy_context(&nu_tcoeff[..larea], cur_tx, tt, eob) as i8;
                    self.set_entropy_ctx(
                        0,
                        cul,
                        mi_col as usize,
                        (mi_row & 31) as usize,
                        blk_row,
                        blk_col,
                        ltxw,
                        ltxh,
                        max_blocks_wide,
                        max_blocks_high,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                    );
                    (eob, tt)
                } else {
                    (0, 0)
                };
                // Luma tx_type_map stamp (top-left + 64-level), per leaf, so the
                // chroma co-location reads the right leaf's tx-type.
                if !self.luma_tt.is_empty() {
                    let cols = cfg.mi_cols as usize;
                    let r0 = mi_row as usize + blk_row;
                    let c0 = mi_col as usize + blk_col;
                    self.luma_tt[r0 * cols + c0] = tx_type as u8;
                    if ltxw == 16 || ltxh == 16 {
                        let rmax = ltxh.min(max_blocks_high - blk_row);
                        let cmax = ltxw.min(max_blocks_wide - blk_col);
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
                let off = ((mi_row * 4) as usize + blk_row * 4) * self.stride
                    + (mi_col * 4) as usize
                    + blk_col * 4;
                let src = (off as i32
                    + (info.dv_row >> 3) * self.stride as i32
                    + (info.dv_col >> 3)) as usize;
                for r in 0..ltxhpx {
                    let s = src + r * self.stride;
                    nu_scratch[r * ltxwpx..(r + 1) * ltxwpx]
                        .copy_from_slice(&self.recon[s..s + ltxwpx]);
                }
                for r in 0..ltxhpx {
                    let d = off + r * self.stride;
                    self.recon[d..d + ltxwpx]
                        .copy_from_slice(&nu_scratch[r * ltxwpx..(r + 1) * ltxwpx]);
                }
                if info.skip == 0 && eob > 0 {
                    let iqm = qm::iqmatrix(self.block_qm_level[0], 0, cur_tx, tx_type);
                    reconstruct_txb_into(
                        &mut self.recon[off..],
                        self.stride,
                        cur_tx,
                        tx_type,
                        &nu_tcoeff[..larea],
                        self.dequants[0],
                        iqm,
                        cfg.bd,
                        &mut self.recon_scratch,
                    );
                }
                txbs.push((eob, tx_type));
            }
        }

        // decode_token_recon_block (decodeframe.c:929-962): iterate the block in
        // 64x64 chunks (max_unit_bsize = BLOCK_64X64) and, within each chunk, do
        // plane 0's txbs then plane 1/2's txbs. For blocks larger than 64x64 this
        // interleaves luma/chroma per 64-unit (a 128-wide block decodes L,U,V of
        // its first 64x64, THEN L,U,V of the next), which the arithmetic decoder
        // requires; for <=64x64 blocks there is exactly one chunk, so the order
        // is identical to the previous plane-major-over-the-whole-block walk.
        let mut txbs_uv = Vec::new();
        let mu_w = max_blocks_wide.min(MI_SIZE_WIDE[BLOCK_64X64] as usize);
        let mu_h = max_blocks_high.min(MI_SIZE_HIGH[BLOCK_64X64] as usize);
        let mut chunk_row = 0usize;
        while do_uniform && chunk_row < max_blocks_high {
            let mut chunk_col = 0usize;
            while chunk_col < max_blocks_wide {
                let luma_row_end = (chunk_row + mu_h).min(max_blocks_high);
                let luma_col_end = (chunk_col + mu_w).min(max_blocks_wide);
                let mut blk_row = chunk_row;
                while blk_row < luma_row_end {
                    let mut blk_col = chunk_col;
                    while blk_col < luma_col_end {
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
                                inter_ext_tx_cdf(
                                    &mut cdfs.inter_ext_tx,
                                    tx_size,
                                    cfg.reduced_tx_set,
                                )
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
                                + (info.dv_col >> 3))
                                as usize;
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
                            self.recon[d..d + txwpx]
                                .copy_from_slice(&scratch[r * txwpx..(r + 1) * txwpx]);
                        }

                        // (3) dequant + inverse transform + add (only when residual
                        // exists) — the block-effective luma dequant row.
                        if info.skip == 0 && eob > 0 {
                            if self.st.coded_lossless {
                                // lossless: TX_4X4 + WHT with the qindex-0 dequant.
                                reconstruct_txb_wht(
                                    &mut self.recon[off..],
                                    self.stride,
                                    &tcoeff,
                                    self.dequants[0],
                                    eob,
                                    cfg.bd,
                                );
                            } else {
                                let iqm = qm::iqmatrix(self.block_qm_level[0], 0, tx_size, tx_type);
                                reconstruct_txb_into(
                                    &mut self.recon[off..],
                                    self.stride,
                                    tx_size,
                                    tx_type,
                                    &tcoeff,
                                    self.dequants[0],
                                    iqm,
                                    cfg.bd,
                                    &mut self.recon_scratch,
                                );
                            }
                        }
                        // (4) CfL luma store (predict_and_reconstruct_intra_block tail,
                        // store_cfl_required): non-chroma-reference blocks always store
                        // (a later group member may pick CfL); the chroma-reference
                        // block stores only when it actually uses CfL. Runs for skip
                        // blocks too (their reconstruction is the prediction).
                        if !cfg.monochrome && (!chroma_ref || info.uv_mode == UV_CFL_PRED) {
                            let block_off =
                                (mi_row * 4) as usize * self.stride + (mi_col * 4) as usize;
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
                if !cfg.monochrome && chroma_ref {
                    let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                    assert_ne!(plane_bsize, 255, "invalid chroma block size");
                    // av1_get_tx_size(plane > 0): lossless forces TX_4X4 (the chroma txb
                    // loop tiles + reads coeffs at 4x4), else the max rect uv tx size.
                    let uv_tx = if self.st.coded_lossless {
                        TX_4X4_IDX
                    } else {
                        max_uv_txsize(bsize, ss_x, ss_y)
                    };
                    let (uv_txw, uv_txh) = (TX_SIZE_WIDE_UNIT[uv_tx], TX_SIZE_HIGH_UNIT[uv_tx]);
                    let (uv_txwpx, uv_txhpx) = (TX_SIZE_WIDE[uv_tx], TX_SIZE_HIGH[uv_tx]);
                    // unit_width/height: THIS 64x64 chunk's luma extent (clamped to the
                    // block), ceil-scaled to chroma units (decodeframe.c:944-947).
                    let unit_width =
                        round_power_of_two((chunk_col + mu_w).min(max_blocks_wide) as i32, ss_x)
                            as usize;
                    let unit_height =
                        round_power_of_two((chunk_row + mu_h).min(max_blocks_high) as i32, ss_y)
                            as usize;
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
                        && uv_smooth(
                            self.mi_uv[((base_row - 1) * cols + base_col + ss_x as i32) as usize],
                        );
                    let le_sm = left_uv
                        && uv_smooth(
                            self.mi_uv[((base_row + ss_y as i32) * cols + base_col - 1) as usize],
                        );
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
                        let mut blk_row = chunk_row >> ss_y;
                        while blk_row < unit_height {
                            let mut blk_col = chunk_col >> ss_x;
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
                                    let luma_tt =
                                        self.luma_tt[lr * cfg.mi_cols as usize + lc] as usize;
                                    if ext_tx_derive(
                                        uv_tx,
                                        true,
                                        cfg.reduced_tx_set,
                                        luma_tt,
                                        false,
                                        0,
                                        0,
                                    )
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
                                    let cul = txb_entropy_context(&tcoeff_uv, uv_tx, tt_uv_eff, eob)
                                        as i8;
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
                                            let idx = color_map_uv[(y0 + r) * uv_map_wpx + x0 + c]
                                                as usize;
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
                                        usize::try_from(n_top)
                                            .expect("n_top_px must be non-negative"),
                                        n_tr,
                                        usize::try_from(n_left)
                                            .expect("n_left_px must be non-negative"),
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
                                        plane_recon[d..d + uv_txwpx].copy_from_slice(
                                            &scratch_uv[r * uv_txwpx..(r + 1) * uv_txwpx],
                                        );
                                    }
                                    // (3) dequant + inverse transform + add — the
                                    // block-effective dequant row of this plane.
                                    if info.skip == 0 && eob > 0 {
                                        if self.st.coded_lossless {
                                            // lossless: TX_4X4 + WHT, this plane's qindex-0 dequant.
                                            reconstruct_txb_wht(
                                                &mut plane_recon[off_uv..],
                                                self.stride_uv,
                                                &tcoeff_uv,
                                                self.dequants[plane],
                                                eob,
                                                cfg.bd,
                                            );
                                        } else {
                                            let iqm = qm::iqmatrix(
                                                self.block_qm_level[plane],
                                                plane,
                                                uv_tx,
                                                tt_uv_eff,
                                            );
                                            reconstruct_txb_into(
                                                &mut plane_recon[off_uv..],
                                                self.stride_uv,
                                                uv_tx,
                                                tt_uv_eff,
                                                &tcoeff_uv,
                                                self.dequants[plane],
                                                iqm,
                                                cfg.bd,
                                                &mut self.recon_scratch,
                                            );
                                        }
                                    }
                                }
                                txbs_uv.push(if eob > 0 { (eob, tt_uv_eff) } else { (0, 0) });
                                blk_col += uv_txw;
                            }
                            blk_row += uv_txh;
                        }
                    }
                }
                chunk_col += mu_w;
            }
            chunk_row += mu_h;
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
            // Intra/intrabc block: the loop-filter grid derives ref/is_inter/
            // mode from `info` (ref0 = INTRA_FRAME, is_inter = use_intrabc).
            inter_lf: None,
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
        use aom_dsp::entropy::lr;
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
        // Poison guard: once a corrupt / out-of-envelope condition is flagged,
        // every subsequent recursion no-ops so the walk unwinds cleanly.
        if self.is_corrupt() {
            return;
        }
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
        // A partition value whose subsize is BLOCK_INVALID is a corrupt frame
        // (a malformed bitstream decoded an out-of-range partition for this
        // bsize). libaom rejects it; the port must not index tables with it.
        if subsize == 255 {
            self.mark_corrupt(format!("corrupt frame: invalid partition {p} for bsize {bsize}"));
            return;
        }
        // Conformance (decode_partition, decodeframe.c:1359-1371): a partition
        // whose sub-block subsamples to an invalid chroma block size is a
        // corrupt frame — libaom aborts it with AOM_CODEC_CORRUPT_FRAME BEFORE
        // the block is created. This is what makes the PORTRAIT luma sizes
        // (which map to BLOCK_INVALID at ss=(1,0)) structurally impossible in a
        // conformant 4:2:2 stream, so the chroma loop filter / reconstruction
        // never index max_txsize_rect_lookup[BLOCK_INVALID]. Scoped to 4:2:2
        // (the only mode where a valid partition subsize can subsample to
        // BLOCK_INVALID; equivalent to C's check for every conformant stream).
        // Inert for conformant streams — exercised, and proven non-firing, by
        // the 14-stream 4:2:2 gate in real_bitstream.rs.
        if self.cfg.subsampling_x == 1
            && self.cfg.subsampling_y == 0
            && get_plane_block_size(subsize, self.cfg.subsampling_x, self.cfg.subsampling_y) == 255
        {
            self.mark_corrupt(format!(
                "4:2:2 corrupt frame: block size index {subsize} invalid with subsampling (1,0)"
            ));
            return;
        }
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
        // `av1/decoder/decodeframe.c`: `r->allow_update_cdf = allow_update_cdf`
        // where `allow_update_cdf = (!large_scale) && !disable_cdf_update`. The
        // single-tile KEY frame is never `large_scale`, so the reader adapts
        // CDFs iff `!disable_cdf_update`. Gates BOTH the mode-info symbol reads
        // (partition/intra/mv/... via `read_symbol`) AND the coefficient reads
        // (`aom_dsp::txb::read_coeffs_txb_full`, whose `rsym` delegates to
        // `read_symbol`), since `read_symbol` is the sole `update_cdf` site.
        dec.allow_update_cdf = !cfg.disable_cdf_update;
        let mut cdfs = KfFrameContext::default_for_qindex(cfg.base_qindex);
        t.start_tile(tb.bounds);
        t.decode_one_tile(&mut dec, &mut cdfs);
    }
    t.into_decode()
}

/// Decode the tiles of an INTER frame (single reference, the walking-skeleton
/// envelope — see [`TileKf::decode_block_inter`]). Mirrors
/// [`decode_frame_tiles_kf`] but drives the inter mode-info + motion-
/// compensation path via `t.inter`. `primary_ref = NONE`, so the shared
/// [`KfFrameContext`] (partition/skip CDFs) loads defaults exactly like a KEY
/// frame; the inter-specific CDFs are built inline per block from the default
/// tables.
pub fn decode_frame_tiles_inter(
    tiles: &[TileBytesKf],
    cfg: &KfTileConfig,
    inter: &InterFrameCfg,
    recon_init: u16,
) -> KfTileDecode {
    assert!(!tiles.is_empty(), "at least one tile");
    let mut t = TileKf::new(cfg, recon_init);
    t.inter = Some(*inter);
    for tb in tiles {
        let mut dec = OdEcDec::new(tb.bytes);
        dec.allow_update_cdf = !cfg.disable_cdf_update;
        let mut cdfs = KfFrameContext::default_for_qindex(cfg.base_qindex);
        t.start_tile(tb.bounds);
        t.decode_one_tile(&mut dec, &mut cdfs);
    }
    t.into_decode()
}

/// `clamp_mv_to_umv_border_sb` (reconinter.h:343), the general **per-plane** form.
/// C applies this once per plane in `dec_calc_subpel_params` (decodeframe.c:611/625)
/// with the plane's `subsampling_x`/`subsampling_y` and the plane prediction dims
/// `bw`/`bh` (`= inter_pred_params->block_{width,height}` = `xd->plane[p].{width,
/// height}`, plane pixels): it scales the MV **and** the (luma-domain 1/8-pel)
/// block edges by `1 << (1 - ss)` before clamping, using a `spel` margin computed
/// from the plane dims. The returned `MV mv_q4` is in the plane's q4 (1/16-plane-
/// pel) grid.
///
/// This port keeps the MV in 1/8-pel-luma units and lets
/// [`aom_dsp::inter::build_inter_predictor`] re-apply the `1 << (1 - ss)` plane scaling,
/// so this returns `mv_q4 / (1 << (1 - ss))` (division is exact: for `ss = 0` the
/// q4 value is even — the edge/spel terms are multiples of 16 — and for `ss = 1`
/// the scale is 1). `mb_to_*` are the CURRENT block's luma edges (mi_row/mi_col/
/// bsize), shared by every plane; `bw`/`bh` are this plane's prediction dims.
#[allow(clippy::too_many_arguments)]
fn clamp_mv_to_umv_border_plane(
    mv_row: i32,
    mv_col: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    bw: i32,
    bh: i32,
    ss_x: usize,
    ss_y: usize,
    cfg: &KfTileConfig,
) -> (i32, i32) {
    const AOM_INTERP_EXTEND: i32 = 4;
    const SUBPEL_BITS: i32 = 4;
    const SUBPEL_SHIFTS: i32 = 16;
    let sx = 1i32 << (1 - ss_x as i32);
    let sy = 1i32 << (1 - ss_y as i32);
    let mb_to_left = -(mi_col * 4 * 8);
    let mb_to_right = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 4 * 8;
    let mb_to_top = -(mi_row * 4 * 8);
    let mb_to_bottom = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 4 * 8;
    let spel_left = (AOM_INTERP_EXTEND + bw) << SUBPEL_BITS;
    let spel_right = spel_left - SUBPEL_SHIFTS;
    let spel_top = (AOM_INTERP_EXTEND + bh) << SUBPEL_BITS;
    let spel_bottom = spel_top - SUBPEL_SHIFTS;
    let col_min = mb_to_left * sx - spel_left;
    let col_max = mb_to_right * sx + spel_right;
    let row_min = mb_to_top * sy - spel_top;
    let row_max = mb_to_bottom * sy + spel_bottom;
    let cq4 = (mv_col * sx).clamp(col_min, col_max);
    let rq4 = (mv_row * sy).clamp(row_min, row_max);
    (rq4 / sy, cq4 / sx)
}

/// `clamp_mv_to_umv_border_sb` (reconinter.h:343) in the LUMA (`ss = 0`) domain —
/// a thin wrapper over [`clamp_mv_to_umv_border_plane`] with the luma plane dims
/// (`block_size_{wide,high}[bsize]`). Returns a 1/8-pel MV. The q4 limits are
/// always even, so the luma clamp is exact in 1/8-pel and
/// [`aom_dsp::inter::build_inter_predictor`] rescales it per plane.
fn clamp_mv_to_umv_border(
    mv_row: i32,
    mv_col: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    cfg: &KfTileConfig,
) -> (i32, i32) {
    let bw = MI_SIZE_WIDE[bsize] * 4;
    let bh = MI_SIZE_HIGH[bsize] * 4;
    clamp_mv_to_umv_border_plane(mv_row, mv_col, mi_row, mi_col, bsize, bw, bh, 0, 0, cfg)
}

// ===================================================================
// OBMC (chunk 4) — motion_mode ceiling + overlapped-block MC helpers.
// ===================================================================

/// `is_motion_variation_allowed_bsize` (blockd.h:1460): OBMC/warp are allowed
/// only for blocks whose smaller side is `>= 8` px.
fn is_motion_variation_allowed_bsize(bsize: usize) -> bool {
    BLOCK_SIZE_WIDE[bsize].min(BLOCK_SIZE_HIGH[bsize]) >= 8
}

/// `is_interintra_allowed` (blockd.h:1430): the block can carry inter-intra
/// (bsize BLOCK_8X8..=BLOCK_32X32, a single-ref inter mode, single reference).
/// The flag is coded (and hence must be read for entropy sync) whenever this
/// holds AND `enable_interintra_compound`. BLOCK_8X8=3, BLOCK_32X32=9;
/// SINGLE_INTER_MODE_START=NEARESTMV=13, SINGLE_INTER_MODE_END=NEAREST_NEARESTMV=17.
fn is_interintra_allowed(bsize: usize, mode: i32, ref0: i32, ref1: i32) -> bool {
    (3..=9).contains(&bsize) && (13..17).contains(&mode) && ref0 > 0 && ref1 <= 0
}

/// `size_group_lookup[bsize]` (common_data.h): selects `interintra_cdf[group]`.
const SIZE_GROUP_LOOKUP: [usize; 22] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2,
];

/// `max_neighbor_obmc[log2(mi_width)]` (blockd.h:1471): the OBMC neighbour cap.
const MAX_NEIGHBOR_OBMC: [i32; 6] = [0, 1, 2, 3, 4, 4];

/// `mi_size_wide_log2[bsize]`: log2 of the block's mi width (a power of two).
fn mi_size_wide_log2(bsize: usize) -> usize {
    MI_SIZE_WIDE[bsize].trailing_zeros() as usize
}

/// `mi_size_high_log2[bsize]`: log2 of the block's mi height (a power of two).
fn mi_size_high_log2(bsize: usize) -> usize {
    MI_SIZE_HIGH[bsize].trailing_zeros() as usize
}

/// `av1_skip_u4x4_pred_in_obmc` (reconinter.c:818): for a plane whose subsampled
/// block size is `<= 8x8` in one dimension, OBMC blends only from the LEFT (skip
/// ABOVE). `dir` is 0 = above, 1 = left. (`DISABLE_CHROMA_U8X8_OBMC == 0`, so the
/// one-sided form.)
fn skip_u4x4_pred_in_obmc(bsize: usize, ss_x: usize, ss_y: usize, dir: i32) -> bool {
    // BLOCK_4X4 = 0, BLOCK_4X8 = 1, BLOCK_8X4 = 2 (enums.h).
    let bsize_plane = get_plane_block_size(bsize, ss_x, ss_y);
    matches!(bsize_plane, 0 | 1 | 2) && dir == 0
}

/// `clamp_mv_to_umv_border_sb` (reconinter.h:343) with the caller's explicit
/// `mb_to_*_edge` (1/8-pel LUMA) and the PREDICTION block dims `bw`/`bh` (px, in
/// the target PLANE's resolution) — the OBMC-neighbour form (the edges are the
/// OBMC-adjusted values, the dims are the narrow overlap strip, not the coding
/// block). `ss_x`/`ss_y` select the plane: the luma edges are scaled by
/// `1 << (1 - ss)` and the `spel_*` bounds use the subsampled `bw`/`bh`, exactly
/// as C's `clamp_mv_to_umv_border_sb`. Returns the (possibly clamped) MV in
/// 1/8-pel LUMA units ([`aom_dsp::inter::build_inter_predictor`] rescales per plane).
#[allow(clippy::too_many_arguments)]
fn clamp_mv_umv_border_px(
    mv_row: i32,
    mv_col: i32,
    bw: i32,
    bh: i32,
    mb_to_left: i32,
    mb_to_right: i32,
    mb_to_top: i32,
    mb_to_bottom: i32,
    ss_x: usize,
    ss_y: usize,
) -> (i32, i32) {
    const AOM_INTERP_EXTEND: i32 = 4;
    const SUBPEL_BITS: i32 = 4;
    const SUBPEL_SHIFTS: i32 = 16;
    let sx = 1i32 << (1 - ss_x as i32);
    let sy = 1i32 << (1 - ss_y as i32);
    let spel_left = (AOM_INTERP_EXTEND + bw) << SUBPEL_BITS;
    let spel_right = spel_left - SUBPEL_SHIFTS;
    let spel_top = (AOM_INTERP_EXTEND + bh) << SUBPEL_BITS;
    let spel_bottom = spel_top - SUBPEL_SHIFTS;
    let col_min = mb_to_left * sx - spel_left;
    let col_max = mb_to_right * sx + spel_right;
    let row_min = mb_to_top * sy - spel_top;
    let row_max = mb_to_bottom * sy + spel_bottom;
    let cq4 = (mv_col * sx).clamp(col_min, col_max);
    let rq4 = (mv_row * sy).clamp(row_min, row_max);
    (rq4 / sy, cq4 / sx)
}
