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
//!   clamped carries are threaded through every block's mode info, but has
//!   no reconstruction effect here — loop filters are not applied (the same
//!   holds in C until the filter stage reads `delta_lf_from_base`/
//!   `delta_lf[]` from the mi grid).
//! - **Off / fixed in this cut**: segmentation, palette, intra block copy,
//!   quantization matrices (flat dequant), superblock size 64x64 (no
//!   128x128), CDF update always on (`disable_cdf_update` unsupported — the
//!   mode-symbol readers adapt unconditionally), and no loop filters
//!   (deblock/CDEF/restoration are not applied to the reconstruction; CDEF
//!   *strengths* are entropy-decoded, and delta-LF levels are carried as
//!   documented above).
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

use aom_encode::reconstruct_txb;
use aom_entropy::dec::OdEcDec;
use aom_entropy::partition::{
    KfBlockState, KfFrameContext, MbModeInfoKf, MiNbrKf, TXFM_CTX_INIT, TxMode, bsize_to_max_depth,
    bsize_to_tx_size_cat, depth_to_tx_size, get_partition_subsize, get_plane_block_size,
    get_tx_size_context, get_uv_mode, intra_avail, is_cfl_allowed, partition_cdf_length,
    partition_plane_context, read_mb_modes_kf_fc, read_partition, read_selected_tx_size,
    set_txfm_ctxs, tx_size_from_tx_mode, update_ext_partition_context,
};
use aom_intra::cfl::{CflCtx, cfl_predict_block, cfl_store_tx};
use aom_intra::predict_intra_high;
use aom_quant::{av1_ac_quant_qtx, av1_dc_quant_qtx};
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

/// Superblock size fixed at 64x64 in this cut: 16 mi per SB side.
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
    /// `quant_params.base_qindex` (the frame base; segmentation is off so
    /// there is one frame qindex). Also the tx-type signalling gate:
    /// `av1_read_tx_type` codes tx types only when `xd->qindex[segment_id]`
    /// — the FRAME-level qindex, not the delta-q-modified carry — is
    /// non-zero. `base_qindex == 0` with all plane deltas zero would be
    /// `coded_lossless` in C (a different decode path, out of scope): give a
    /// zero-base config a non-zero plane delta.
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
}

/// The per-plane `[dc, ac]` dequant steps for an effective qindex — the shared
/// formula of `setup_segmentation_dequant` (frame level, decodeframe.c) and
/// the per-block delta-q recompute in `parse_decode_block`: per plane the
/// frame's dc/ac deltas fold in via `av1_{dc,ac}_quant_QTX` (which clamp
/// `qindex + delta` to `[0, MAXQ]`); Y's AC delta is always 0. Segmentation is
/// off in this scope, so this is the `segment_id = 0` row of
/// `seg_dequant_QTX` (`av1_get_qindex` is the identity without `SEG_LVL_ALT_Q`).
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

// ---- the driver -----------------------------------------------------------------

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
    /// Per-mi mode-info grid (frame-cropped stamps, like the C mi grid): the
    /// [`MiNbrKf`] projection (`y_mode` + `skip_txfm`) every context selection
    /// reads through `xd->above_mbmi` / `xd->left_mbmi`, and which also feeds
    /// `get_filt_type` (edge-filter type 1 when a neighbour's y mode is
    /// SMOOTH/SMOOTH_V/SMOOTH_H).
    mi: Vec<MiNbrKf>,
    /// Per-mi UV-mode grid (same frame-cropped stamps): what the chroma
    /// `get_filt_type` reads through `xd->chroma_above_mbmi` /
    /// `xd->chroma_left_mbmi` (`is_smooth(mbmi, plane > 0)` checks the
    /// neighbour's uv_mode). Non-chroma-reference blocks stamp `UV_DC_PRED`
    /// (the C `read_intra_frame_mode_info` else-branch), but the chroma
    /// neighbour pointers only ever land on chroma-reference cells.
    mi_uv: Vec<i8>,
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
        let aligned_mi_cols = (cfg.mi_cols as usize).div_ceil(SB_MI as usize) * SB_MI as usize;
        let aligned_mi_rows = (cfg.mi_rows as usize).div_ceil(SB_MI as usize) * SB_MI as usize;
        let stride = aligned_mi_cols * 4;
        let stride_uv = if cfg.monochrome { 0 } else { stride >> ss_x };
        let uv_len = if cfg.monochrome {
            0
        } else {
            stride_uv * ((aligned_mi_rows * 4) >> ss_y)
        };
        let st = KfBlockState {
            segid_preskip: false,
            seg_enabled: false,
            update_map: false,
            seg_pred: 0,
            last_active_segid: 0,
            seg_skip_active: false,
            mi_row: 0,
            mi_col: 0,
            mib_size: SB_MI,
            sb_size: BLOCK_64X64,
            bsize: BLOCK_64X64,
            coded_lossless: false,
            allow_intrabc: false,
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
        TileKf {
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
            mi: vec![
                MiNbrKf {
                    y_mode: 0,
                    skip_txfm: 0
                };
                (cfg.mi_rows * cfg.mi_cols) as usize
            ],
            mi_uv: vec![0; (cfg.mi_rows * cfg.mi_cols) as usize],
            cfl: CflCtx::new(ss_x as i32, ss_y as i32),
            // setup_segmentation_dequant: the frame-level dequant rows from
            // base_qindex (the live values until a per-block recompute).
            dequants: plane_dequants(cfg, cfg.base_qindex),
            st,
            tree: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// The `xd->above_mbmi` / `xd->left_mbmi` neighbours of the block at
    /// `(mi_row, mi_col)`: the mi-grid entries directly above / left of the
    /// block origin, `None` when off the tile (`up_available`/`left_available`).
    fn neighbours(&self, mi_row: i32, mi_col: i32) -> (Option<MiNbrKf>, Option<MiNbrKf>) {
        let cols = self.cfg.mi_cols;
        let above = (mi_row > 0).then(|| self.mi[((mi_row - 1) * cols + mi_col) as usize]);
        let left = (mi_col > 0).then(|| self.mi[(mi_row * cols + mi_col - 1) as usize]);
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
        let up_available = mi_row > 0;
        let left_available = mi_col > 0;
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
        let info = read_mb_modes_kf_fc(
            dec,
            cdfs,
            &mut self.st,
            cfg.enable_filter_intra,
            above,
            left,
        );

        // --- parse_decode_block: the block's transform size (read_tx_size) +
        // txfm-context stamp, in the C statement order (after the mode info /
        // palette tokens, before the skip entropy reset) ---
        let bw = MI_SIZE_WIDE[bsize] as usize;
        let bh = MI_SIZE_HIGH[bsize] as usize;
        // read_tx_size (decodeframe.c), intra: coded_lossless (TX_4X4
        // preemption) is off in this scope; a signalling block (bsize >
        // BLOCK_4X4) under TX_MODE_SELECT codes its tx-size depth — intra
        // blocks code it even when skip_txfm is set (`!is_inter ||
        // allow_select_inter` is true for intra) — else the tx_mode fallback.
        let tx_size = if bsize > 0 {
            // block_signals_txsize
            if cfg.tx_mode == TxMode::Select {
                let cat = bsize_to_tx_size_cat(bsize) as usize;
                let ctx = get_tx_size_context(
                    bsize,
                    self.above_t[mi_col as usize],
                    self.left_t[(mi_row & 31) as usize],
                    up_available,
                    left_available,
                    None, // KEY frame, intrabc off: neighbours are never inter
                    None,
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
            }
        } else {
            MAX_TXSIZE_RECT_LOOKUP[bsize]
        };
        // set_txfm_ctxs(tx_size, xd->width, xd->height, skip && is_inter, xd):
        // full (not frame-clipped) footprint; the skip arg is always 0 for
        // intra blocks, so a skipped intra block still stamps its tx dims.
        set_txfm_ctxs(
            &mut self.above_t[mi_col as usize..],
            &mut self.left_t[(mi_row & 31) as usize..],
            tx_size,
            bw,
            bh,
            false,
        );

        // parse_decode_block (decodeframe.c): with delta-q present, every
        // block's dequant is recomputed from the running current_base_qindex
        // (already advanced by this block's SB-level delta read inside the
        // mode-info decode — mbmi->current_qindex == the carry). C refills
        // all MAX_SEGMENTS rows; segmentation is off here, so only the
        // segment-0 row is live. The per-plane dc/ac deltas fold in through
        // av1_{dc,ac}_quant_QTX.
        if cfg.delta_q_present {
            debug_assert_eq!(self.st.current_base_qindex, info.current_qindex);
            self.dequants = plane_dequants(cfg, self.st.current_base_qindex);
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
        // av1_read_tx_type gate: !skip_txfm && !seg-skip && qindex > 0 —
        // xd->qindex[segment_id], the FRAME-level qindex (not the delta-q
        // carry), so the gate is frame-constant with segmentation off.
        let signal_gate = cfg.base_qindex > 0 && info.skip == 0;
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
                    let ext = intra_ext_tx_cdf(
                        &mut cdfs.ext_tx_1ddct,
                        &mut cdfs.ext_tx_dtt4,
                        tx_size,
                        cfg.reduced_tx_set,
                        info.use_filter_intra != 0,
                        info.filter_intra_mode as usize,
                        info.y_mode as usize,
                    );
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
                        false,
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

                // (2) intra prediction into the reconstruction plane.
                let (n_top, n_tr, n_left, n_bl) = intra_avail(
                    BLOCK_64X64,
                    bsize,
                    mi_row,
                    mi_col,
                    up_available,
                    left_available,
                    cfg.mi_cols,
                    cfg.mi_rows,
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
            let up_uv = adj_row > 0;
            let left_uv = adj_col > 0;
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
                            eob
                        } else {
                            0
                        };

                        // (2) chroma intra prediction (av1_predict_intra_block_facade):
                        // ordinary intra with mode = get_uv_mode(uv_mode) — DC for
                        // CfL — then the CfL AC contribution on top.
                        let (n_top, n_tr, n_left, n_bl) = intra_avail(
                            BLOCK_64X64,
                            bsize_uv,
                            adj_row,
                            adj_col,
                            up_uv,
                            left_uv,
                            cfg.mi_cols,
                            cfg.mi_rows,
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
                        {
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
                        if info.uv_mode == UV_CFL_PRED {
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
                                    tt_uv,
                                    &tcoeff_uv,
                                    self.dequants[plane],
                                    None,
                                    cfg.bd,
                                );
                            }
                        }
                        txbs_uv.push(if eob > 0 { (eob, tt_uv) } else { (0, 0) });
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

/// Decode one KEY-frame luma tile (the whole frame): the `decode_tile` SB
/// row/col loop — above contexts zeroed once, left contexts zeroed per SB row,
/// each superblock decoded through the recursive partition walk with the
/// per-leaf mode-info → coefficient → predict → reconstruct interleave.
///
/// `recon_init` fills the reconstruction plane before decoding; a conformant
/// walk never *reads* an unwritten pixel (the availability logic only exposes
/// previously reconstructed samples), so the roundtrip test gives encoder and
/// decoder different fills to turn any availability bug into a hard mismatch.
pub fn decode_tile_kf(
    dec: &mut OdEcDec,
    cfg: &KfTileConfig,
    cdfs: &mut KfFrameContext,
    recon_init: u16,
) -> KfTileDecode {
    let mut t = TileKf::new(cfg, recon_init);
    let mut mi_row = 0;
    while mi_row < cfg.mi_rows {
        t.left_e = [[0; 32]; 3]; // av1_zero_left_context per SB row (all planes)
        t.left_p = [0; 32];
        t.left_t = [TXFM_CTX_INIT; 32]; // ..incl the left txfm-context bytes
        let mut mi_col = 0;
        while mi_col < cfg.mi_cols {
            t.decode_partition(dec, cdfs, mi_row, mi_col, BLOCK_64X64);
            mi_col += SB_MI;
        }
        mi_row += SB_MI;
    }
    KfTileDecode {
        recon: t.recon,
        stride: t.stride,
        width: cfg.mi_cols as usize * 4,
        height: cfg.mi_rows as usize * 4,
        recon_u: t.recon_u,
        recon_v: t.recon_v,
        stride_uv: t.stride_uv,
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
        tree: t.tree,
        blocks: t.blocks,
    }
}
