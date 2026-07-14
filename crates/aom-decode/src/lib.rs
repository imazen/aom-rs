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
//! - **Plane 0 (luma) reconstruction only.** With `monochrome = true` this is
//!   the complete frame reconstruction for a real AV1 configuration; with
//!   `monochrome = false` (4:4:4) the chroma *mode-info symbols* are decoded but
//!   chroma planes are not reconstructed (CfL prediction is not ported).
//! - **`TX_MODE_LARGEST`**: per-block `tx_size = max_txsize_rect_lookup[bsize]`,
//!   which codes no tx-size bits — matching the landed KEY-frame mode path
//!   ([`MbModeInfoKf`] carries no tx size; `TX_MODE_SELECT` needs the
//!   `get_tx_size_context` neighbour facade, which is not ported). Consequence:
//!   blocks ≤ 64x64 have exactly one luma txb, so the *within-block* multi-txb
//!   interleave is structurally present but degenerate; the *across-block*
//!   reconstruction feedback and entropy-context threading are fully exercised.
//! - **Shared, context-pre-selected CDFs** (the landed [`KfCdfs`] simplification):
//!   each mode-info symbol adapts one shared CDF instance rather than the full
//!   neighbour-selected `FRAME_CONTEXT` array. The only per-block CDF *selection*
//!   done here is forced by alphabet consistency: the UV mode CDF has 14 symbols
//!   for CfL-allowed blocks and 13 otherwise, so [`KfTileCdfs`] keeps the two
//!   instances the real `uv_mode_cdf[cfl_allowed][..]` split implies, and the
//!   ext-tx CDF is kept per set type (5- and 7-symbol alphabets). Full
//!   FRAME_CONTEXT context selection is the next layer.
//! - **Off / fixed in this cut**: segmentation, palette, intra block copy,
//!   delta-q / delta-lf (so the dequant step is frame-constant; per-block
//!   `av1_dc/ac_quant_QTX` recompute is not wired), quantization matrices
//!   (flat dequant), superblock size 64x64 (no 128x128), CDF update always on
//!   (`disable_cdf_update` unsupported — the mode-symbol readers adapt
//!   unconditionally), and no loop filters (deblock/CDEF/restoration are not
//!   applied to the reconstruction; CDEF *strengths* are entropy-decoded).
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
    KfBlockState, KfCdfs, MbModeInfoKf, get_partition_subsize, intra_avail, is_cfl_allowed,
    partition_cdf_length, partition_plane_context, read_filter_intra_mode_info, read_mb_modes_kf,
    read_partition, update_ext_partition_context,
};
use aom_intra::predict_intra_high;
use aom_txb::{
    ext_tx_set_type, get_txb_ctx, read_coeffs_txb_full, txb_entropy_context, txb_high, txb_wide,
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
/// `ANGLE_STEP`: coded angle deltas scale by 3 degrees.
pub const ANGLE_STEP: i32 = 3;

/// Superblock size fixed at 64x64 in this cut: 16 mi per SB side.
const SB_MI: i32 = 16;

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
    /// reconstruction is the complete frame. When false the driver models
    /// 4:4:4 (`is_chroma_ref` always true) and decodes UV mode-info symbols
    /// without reconstructing chroma.
    pub monochrome: bool,
    /// `cdef_info.cdef_bits` (0..=3): per-64x64 CDEF strength literal width.
    pub cdef_bits: u32,
    /// `!seq_params->enable_intra_edge_filter`.
    pub disable_edge_filter: bool,
    /// `seq_params->enable_filter_intra` (the bsize/mode gates are per block).
    pub enable_filter_intra: bool,
    /// `features.reduced_tx_set_used`.
    pub reduced_tx_set: bool,
    /// The `qindex > 0` term of the tx-type signalling gate
    /// (`av1_read_tx_type`: tx types are only coded when the frame qindex is
    /// non-zero; segmentation is off so there is one frame qindex).
    pub base_qindex_gt0: bool,
    /// Frame-constant `[dc, ac]` dequant steps (`seg_dequant_QTX[0]`, flat/no QM).
    pub dequant: [i16; 2],
}

/// Every CDF the KEY-frame luma tile decode touches. All are *shared* adapting
/// instances (see crate docs); the encoder mirror must start from identical
/// contents for the roundtrip.
#[derive(Clone, Debug)]
pub struct KfTileCdfs {
    /// The per-block mode-info CDFs. `kf.uv_mode` is a scratch slot: the driver
    /// swaps [`Self::uv_mode_cfl`] / [`Self::uv_mode_nocfl`] in per block (the
    /// two UV alphabets — 14 vs 13 symbols — must be separate instances, exactly
    /// as the real `uv_mode_cdf[cfl_allowed][..]` split implies).
    pub kf: KfCdfs,
    pub uv_mode_cfl: [u16; 15],
    pub uv_mode_nocfl: [u16; 15],
    /// The coefficient-CDF arena (`aom_txb::CDF_ARENA_LEN` u16).
    pub coeff: Vec<u16>,
    /// Intra ext-tx CDFs per set type: `EXT_TX_SET_DTT4_IDTX` (5 symbols;
    /// 16x16-class or `reduced_tx_set`) and `EXT_TX_SET_DTT4_IDTX_1DDCT`
    /// (7 symbols; smaller-than-16x16 class). 32/64-class intra blocks are
    /// DCT-only and code nothing.
    pub ext_tx_dtt4_idtx: [u16; 6],
    pub ext_tx_dtt4_idtx_1ddct: [u16; 8],
    /// Partition CDF arena: 20 contexts, each an ns-symbol CDF sized by its
    /// context's block-size level (4/8/10 symbols).
    pub partition: [[u16; 11]; 20],
}

// ---- decode result --------------------------------------------------------------

/// One decoded leaf block: its position/size, the partition type that created it,
/// the decoded mode info, and the per-txb `(eob, tx_type)` in raster order
/// (plane 0; skip blocks record `(0, 0)` per txb).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedBlockKf {
    pub mi_row: i32,
    pub mi_col: i32,
    pub bsize: usize,
    pub partition: usize,
    pub info: MbModeInfoKf,
    pub tx_size: usize,
    pub txbs: Vec<(usize, usize)>,
}

/// A decoded KEY-frame luma tile: the reconstruction plane (superblock-aligned;
/// the frame crop is `width x height` pixels at the top-left), the pre-order
/// partition sequence (every visited node, including uncoded forced partitions),
/// and the per-leaf decode records.
#[derive(Clone, Debug)]
pub struct KfTileDecode {
    pub recon: Vec<u16>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
    pub tree: Vec<i8>,
    pub blocks: Vec<DecodedBlockKf>,
}

// ---- shared driver helpers (also used by the roundtrip mirror encoder) ----------

/// `av1_filter_intra_allowed` (av1/common/blockd.h) for a KEY intra block with
/// palette off: the sequence flag, the ≤32x32 bsize gate, and `mode == DC_PRED`.
/// The mode term is why the flag is coded *after* the intra mode — the driver
/// reads/writes it as a follow-up to the `read/write_mb_modes_kf` call (whose
/// flat `filter_allowed` input cannot depend on the mode decoded inside it).
pub fn filter_intra_allowed(enable_filter_intra: bool, bsize: usize, y_mode: i32) -> bool {
    enable_filter_intra
        && y_mode == DC_PRED
        && BLOCK_SIZE_WIDE[bsize] <= 32
        && BLOCK_SIZE_HIGH[bsize] <= 32
}

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

/// Select the intra ext-tx CDF for a tx size out of the per-set-type instances.
/// Set types 0 (DCT-only) never code a symbol; any buffer satisfies the unused
/// parameter.
pub fn intra_ext_tx_cdf<'a>(
    dtt4_idtx: &'a mut [u16; 6],
    dtt4_idtx_1ddct: &'a mut [u16; 8],
    tx_size: usize,
    reduced_tx_set: bool,
) -> &'a mut [u16] {
    match ext_tx_set_type(tx_size, false, reduced_tx_set) {
        3 => dtt4_idtx_1ddct,
        _ => dtt4_idtx, // set 2 (DTT4_IDTX) or the never-coded DCT-only sets
    }
}

// ---- the driver -----------------------------------------------------------------

struct TileKf<'c> {
    cfg: &'c KfTileConfig,
    /// Luma reconstruction plane, SB-aligned, `stride` = aligned width in px.
    recon: Vec<u16>,
    stride: usize,
    /// Plane-0 coefficient entropy contexts: above spans the aligned tile width
    /// (one i8 per mi col, zeroed once per tile); left is the one-SB-tall rolling
    /// column, zeroed at each SB row, indexed by `mi_row & 31`.
    above_e: Vec<i8>,
    left_e: [i8; 32],
    /// Partition contexts with the same lifetimes/indexing.
    above_p: Vec<i8>,
    left_p: [i8; 32],
    /// Per-mi "luma mode is smooth" grid (frame-cropped stamps) — feeds
    /// `get_filt_type` (the intra edge-filter type is 1 when the above or left
    /// neighbour block's y mode is SMOOTH/SMOOTH_V/SMOOTH_H).
    smooth: Vec<u8>,
    st: KfBlockState,
    tree: Vec<i8>,
    blocks: Vec<DecodedBlockKf>,
}

impl<'c> TileKf<'c> {
    fn new(cfg: &'c KfTileConfig, recon_init: u16) -> Self {
        assert!(cfg.mi_rows > 0 && cfg.mi_cols > 0, "empty frame");
        assert!(matches!(cfg.bd, 8 | 10 | 12), "bd must be 8/10/12");
        let aligned_mi_cols = (cfg.mi_cols as usize).div_ceil(SB_MI as usize) * SB_MI as usize;
        let aligned_mi_rows = (cfg.mi_rows as usize).div_ceil(SB_MI as usize) * SB_MI as usize;
        let stride = aligned_mi_cols * 4;
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
            dq_present: false,
            dlf_present: false,
            dlf_multi: false,
            num_planes: if cfg.monochrome { 1 } else { 3 },
            dq_res: 1,
            dlf_res: 1,
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
            current_base_qindex: 0,
            xd_delta_lf: [0; 4],
            xd_delta_lf_from_base: 0,
        };
        TileKf {
            cfg,
            recon: vec![recon_init; stride * aligned_mi_rows * 4],
            stride,
            above_e: vec![0; aligned_mi_cols],
            left_e: [0; 32],
            above_p: vec![0; aligned_mi_cols],
            left_p: [0; 32],
            smooth: vec![0; (cfg.mi_rows * cfg.mi_cols) as usize],
            st,
            tree: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// `get_filt_type` (reconintra.c), luma: 1 when the above or left neighbour
    /// *block* (the mi at the block origin's up/left) has a smooth y mode.
    fn filt_type(&self, mi_row: i32, mi_col: i32, up: bool, left: bool) -> i32 {
        let cols = self.cfg.mi_cols;
        let ab = up && self.smooth[((mi_row - 1) * cols + mi_col) as usize] != 0;
        let le = left && self.smooth[(mi_row * cols + mi_col - 1) as usize] != 0;
        (ab || le) as i32
    }

    /// Stamp the block's "smooth y mode" bit over its frame-cropped mi footprint
    /// (the mi-grid stamp `set_offsets` clips with `x_mis`/`y_mis`).
    fn stamp_smooth(&mut self, mi_row: i32, mi_col: i32, bsize: usize, y_mode: i32) {
        let sm = ((SMOOTH_PRED..=SMOOTH_H_PRED).contains(&y_mode)) as u8;
        let x_mis = MI_SIZE_WIDE[bsize].min(self.cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(self.cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * self.cfg.mi_cols + mi_col) as usize;
            self.smooth[base..base + x_mis as usize].fill(sm);
        }
    }

    /// `av1_set_entropy_contexts` (av1/common/blockd.c), plane 0: fill the txb's
    /// above/left context footprint with its cul level, zeroing the beyond-frame
    /// part when a non-zero fill crosses the frame edge.
    #[allow(clippy::too_many_arguments)]
    fn set_entropy_ctx(
        &mut self,
        cul: i8,
        mi_row: i32,
        mi_col: i32,
        blk_row: usize,
        blk_col: usize,
        txw: usize,
        txh: usize,
        blocks_wide: usize,
        blocks_high: usize,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
    ) {
        let a0 = mi_col as usize + blk_col;
        if cul != 0 && mb_to_right_edge < 0 {
            let n = txw.min(blocks_wide - blk_col);
            self.above_e[a0..a0 + n].fill(cul);
            self.above_e[a0 + n..a0 + txw].fill(0);
        } else {
            self.above_e[a0..a0 + txw].fill(cul);
        }
        let l0 = (mi_row & 31) as usize + blk_row;
        if cul != 0 && mb_to_bottom_edge < 0 {
            let n = txh.min(blocks_high - blk_row);
            self.left_e[l0..l0 + n].fill(cul);
            self.left_e[l0 + n..l0 + txh].fill(0);
        } else {
            self.left_e[l0..l0 + txh].fill(cul);
        }
    }

    /// One leaf block: `parse_decode_block` (mode info + tx sizing + skip
    /// entropy-reset) followed by the intra `decode_token_recon_block` txb loop.
    fn decode_block(
        &mut self,
        dec: &mut OdEcDec,
        cdfs: &mut KfTileCdfs,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
    ) {
        let cfg = self.cfg;
        let up_available = mi_row > 0;
        let left_available = mi_col > 0;
        let cfl_allowed = !cfg.monochrome && is_cfl_allowed(bsize, false, 0, 0);

        // --- decode_mbmi_block: the KEY-frame mode info ---
        self.st.mi_row = mi_row;
        self.st.mi_col = mi_col;
        self.st.bsize = bsize;
        self.st.cfl_allowed = cfl_allowed;
        self.st.mb_to_top_edge = -(mi_row * 32);
        self.st.has_above = up_available;
        self.st.has_left = left_available;
        // Alphabet-consistent UV CDF selection (14-symbol CfL vs 13-symbol).
        let saved_uv = cdfs.kf.uv_mode;
        cdfs.kf.uv_mode = if cfl_allowed {
            cdfs.uv_mode_cfl
        } else {
            cdfs.uv_mode_nocfl
        };
        let mut info = read_mb_modes_kf(dec, &mut cdfs.kf, &mut self.st);
        if cfl_allowed {
            cdfs.uv_mode_cfl = cdfs.kf.uv_mode;
        } else {
            cdfs.uv_mode_nocfl = cdfs.kf.uv_mode;
        }
        cdfs.kf.uv_mode = saved_uv;
        // Filter-intra follow-up with the C-exact mode-dependent gate (the last
        // mode-info symbol; see `filter_intra_allowed`).
        let fi_allowed = filter_intra_allowed(cfg.enable_filter_intra, bsize, info.y_mode);
        let (use_fi, fi_mode) =
            read_filter_intra_mode_info(dec, &mut cdfs.kf.fi_use, &mut cdfs.kf.fi_mode, fi_allowed);
        info.use_filter_intra = use_fi;
        info.filter_intra_mode = fi_mode;

        // --- parse_decode_block tail: skip blocks reset their entropy context ---
        let bw = MI_SIZE_WIDE[bsize] as usize;
        let bh = MI_SIZE_HIGH[bsize] as usize;
        if info.skip != 0 {
            let a0 = mi_col as usize;
            self.above_e[a0..a0 + bw].fill(0);
            let l0 = (mi_row & 31) as usize;
            self.left_e[l0..l0 + bh].fill(0);
        }

        // --- decode_token_recon_block (intra): per-txb read -> predict -> recon ---
        let tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize]; // TX_MODE_LARGEST, no bits
        let (txw, txh) = (TX_SIZE_WIDE_UNIT[tx_size], TX_SIZE_HIGH_UNIT[tx_size]);
        let (txwpx, txhpx) = (TX_SIZE_WIDE[tx_size], TX_SIZE_HIGH[tx_size]);
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let max_blocks_wide = max_block_units(BLOCK_SIZE_WIDE[bsize], mb_to_right_edge);
        let max_blocks_high = max_block_units(BLOCK_SIZE_HIGH[bsize], mb_to_bottom_edge);
        let filt_type = self.filt_type(mi_row, mi_col, up_available, left_available);
        // av1_read_tx_type gate: !skip_txfm && !seg-skip && qindex > 0.
        let signal_gate = cfg.base_qindex_gt0 && info.skip == 0;
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
                    let (tsc, dsc) =
                        get_txb_ctx(bsize, tx_size, 0, &self.above_e[a0..], &self.left_e[l0..]);
                    let ext = intra_ext_tx_cdf(
                        &mut cdfs.ext_tx_dtt4_idtx,
                        &mut cdfs.ext_tx_dtt4_idtx_1ddct,
                        tx_size,
                        cfg.reduced_tx_set,
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
                        cul,
                        mi_row,
                        mi_col,
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

                // (3) dequant + inverse transform + add (only when residual exists).
                if info.skip == 0 && eob > 0 {
                    reconstruct_txb(
                        &mut self.recon[off..],
                        self.stride,
                        tx_size,
                        tx_type,
                        &tcoeff,
                        cfg.dequant,
                        None,
                        cfg.bd,
                    );
                }
                txbs.push((eob, tx_type));
                blk_col += txw;
            }
            blk_row += txh;
        }

        self.stamp_smooth(mi_row, mi_col, bsize, info.y_mode);
        self.blocks.push(DecodedBlockKf {
            mi_row,
            mi_col,
            bsize,
            partition,
            info,
            tx_size,
            txbs,
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
        cdfs: &mut KfTileCdfs,
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
    cdfs: &mut KfTileCdfs,
    recon_init: u16,
) -> KfTileDecode {
    let mut t = TileKf::new(cfg, recon_init);
    let mut mi_row = 0;
    while mi_row < cfg.mi_rows {
        t.left_e = [0; 32]; // av1_zero_left_context per SB row
        t.left_p = [0; 32];
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
        tree: t.tree,
        blocks: t.blocks,
    }
}
