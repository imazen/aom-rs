//! The pack stage: `OUTPUT_ENABLED` — walk a *picked* partition tree
//! ([`crate::partition_pick::rd_pick_partition_real`]'s winner) a second
//! time, this time driving a real entropy coder, to emit partition symbols,
//! per-leaf KEY-frame mode-info, and per-leaf coefficient bytes. This is the
//! SB/tile-walk composition + coefficient-write half of the encoder gate's
//! "compose the per-SB partition RDO across a whole frame" chunk.
//!
//! Mirrors libaom's actual architecture: `av1_rd_pick_partition` (the
//! DRY_RUN search, decides `pc_tree`) is followed by a **separate**
//! `encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, OUTPUT_ENABLED, bsize,
//! pc_tree, NULL)` call (`av1/encoder/partition_search.c:2073/5133/5458/
//! 5545/6433`) that is the one true real encode. This module's [`pack_sb`]
//! IS that second call; [`pack_tile`] is the per-SB-row/col loop
//! (`av1/encoder/encodeframe.c`'s `encode_sb_row`) that calls search then
//! pack for every superblock in raster order.
//!
//! # Why re-running `encode_b_intra_dry` is correct, not a shortcut
//!
//! [`crate::encode_intra::encode_intra_block_plane_y`]/`_uv`'s only
//! `dry_run_output_enabled`-gated behavior is [`crate::encode_intra::is_trellis_used`]'s
//! `FinalPassTrellisOpt` check (`encodemb.h`). For the default speed-0
//! envelope (`NoEstimateYrdTrellisOpt`) and the `NO_TRELLIS`/lossless arms
//! `is_trellis_used` returns the same value **regardless** of the flag, so
//! the pack matched even when the flag was hardcoded. `FINAL_PASS_TRELLIS_OPT`
//! (`--disable-trellis-quant=2`) is the exception: the final encode MUST
//! trellis where the search did not, so [`crate::encode_sb::encode_b_intra_dry`]
//! now takes an `output_enabled` argument and this pack passes `true`
//! (C's `OUTPUT_ENABLED`). Re-running it over the SAME winning leaf, from the
//! SAME starting context state, reproduces byte-identical
//! qcoeff/eob/tx_type/dqcoeff to what a true `OUTPUT_ENABLED` call would —
//! this module reuses that validated code path instead of a parallel copy,
//! and only adds the two things `OUTPUT_ENABLED` actually changes: symbol
//! emission (partition / mode-info / tx-size / coefficients) and CDF
//! adaptation. Both search and pack walks visit the SAME winning leaves in
//! the SAME order starting from the SAME zeroed initial tile state (search's
//! own `av1_save_context`/`av1_restore_context` rollback ensures only the
//! winning subtree's contribution survives in its context arrays by the time
//! [`crate::partition_pick::rd_pick_partition_real`] returns) — so the two
//! independently-progressing [`TileCtxState`] instances ([`pack_tile`] keeps
//! one for search, one for pack) stay in lockstep across the whole tile
//! without needing to snapshot/restore between them.
//!
//! # Scope
//!
//! Matches [`crate::partition_pick::rd_pick_partition_real`]'s envelope:
//! `NONE`/`SPLIT`/`HORZ`/`VERT` (4 of 10 partition types), KEY intra,
//! interior SBs, `sb_size <= 64`, no segmentation, no delta-q/delta-lf, no
//! palette, `allow_intrabc = false`, uniform (`TX_MODE_SELECT`) luma tx
//! size. `cdef_bits = 0` models `enable_cdef = 0` (the CDEF-strength literal
//! is zero-width, so `write_literal` is a no-op regardless of the value
//! passed — matches the frame-header `cdef_bits` derivation when the search
//! never finds more than one strength, which is what "off" collapses to).
//! MISSING (mechanical extensions once needed): AB/4-way partition shapes,
//! segmentation-driven per-block qindex, delta-q/delta-lf signaling,
//! palette, intrabc, SB128, multi-tile.

use crate::encode_intra::TxbEncode;
use crate::encode_sb::{LeafWinner, SbEncodeEnv, SbTree, TileCtxState, encode_b_intra_dry};
use crate::intra_uv_rd::{av1_get_tx_size_uv, is_chroma_reference};
use crate::partition::{PartRdStats, split_subsize};
use crate::partition_pick::{ModeGrid, PickFrameCfg, rd_pick_partition_real};
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TXS_H, TXS_W};

/// `ROUND_POWER_OF_TWO(value, n)` for the chroma-subsampling scale (n = ss ∈
/// {0, 1}): `n == 0` is the identity (4:4:4 / the luma plane), `n == 1` is
/// `(value + 1) >> 1` (the 2:1 chroma bound). Matches the C macro on the only
/// values the mu-64 chroma bounds pass it.
#[inline]
fn round_pow2(value: usize, n: usize) -> usize {
    if n == 0 {
        value
    } else {
        (value + (1 << (n - 1))) >> n
    }
}
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::lr::{
    LrFrameConfig, LrRefState, LrUnitInfo, RESTORE_NONE as LR_RESTORE_NONE, lr_corners_in_sb,
    write_lr_unit,
};
use aom_dsp::entropy::partition::{
    KfBlockState, KfFrameContext, MbModeInfoKf, MiNbrKf, PaletteNbrKf, allow_palette,
    bsize_to_max_depth, bsize_to_tx_size_cat, encode_color_map_tokens, get_partition_subsize,
    get_tx_size_context, is_cfl_allowed, partition_cdf_length, partition_plane_context,
    tx_size_to_depth, update_ext_partition_context, write_mb_modes_kf_fc, write_partition,
    write_selected_tx_size,
};
use aom_dsp::intra::cfl::CflCtx;
use aom_dsp::txb::{ext_tx_derive, write_coeffs_txb_full};

/// `PARTITION_NONE`/`HORZ`/`VERT`/`SPLIT` C values (matches
/// [`crate::encode_sb`]'s private copies — duplicated here since they're not
/// exported).
const PARTITION_NONE: i32 = 0;
const PARTITION_HORZ: i32 = 1;
const PARTITION_VERT: i32 = 2;
const PARTITION_SPLIT: i32 = 3;
const PARTITION_HORZ_A: i32 = 4;
const PARTITION_HORZ_B: i32 = 5;
const PARTITION_VERT_A: i32 = 6;
const PARTITION_VERT_B: i32 = 7;
const PARTITION_HORZ_4: i32 = 8;
const PARTITION_VERT_4: i32 = 9;

/// `DEFAULT_DELTA_LF_RES` (av1/common/enums.h:505) — the `--delta-lf-mode=1`
/// per-SB delta-lf grid resolution.
const DELTA_LF_RES: i32 = 2;
/// `MAX_LOOP_FILTER` (av1/common/av1_loopfilter.h:27) — the delta-lf clamp.
const MAX_LOOP_FILTER: i32 = 63;

/// Frame-level pack-stage constants beyond what [`SbEncodeEnv`] already
/// carries for the residual recompute.
#[derive(Clone, Copy, Debug)]
pub struct PackCfg {
    /// `seq_params->enable_filter_intra` — threaded to
    /// [`write_mb_modes_kf_fc`] and MUST equal the value
    /// [`PickFrameCfg::enable_filter_intra`] used during the search (same
    /// seq-header flag).
    pub enable_filter_intra: bool,
    /// `cm->features.tx_mode == TX_MODE_SELECT` (speed-0 default: true —
    /// `tx_search.rs`'s `tx_mode_is_select` doc: "speed-0 all-intra: true").
    pub tx_mode_is_select: bool,
    /// The frame's `av1_write_tx_type` gate (`bitstream.c:815-819`):
    /// `((!seg.enabled && base_qindex > 0) || (seg.enabled &&
    /// qindex[segment_id] > 0)) && !skip_txfm && !seg_skip`. No
    /// segmentation and `skip_txfm` always 0 in this envelope, so this
    /// reduces to `base_qindex > 0` — a frame constant.
    pub signal_gate: bool,
    /// `!cm->features.disable_cdf_update` — whether symbol writes adapt
    /// their CDFs.
    pub allow_update_cdf: bool,
    /// The frame's `current_base_qindex` init (`quant_params->base_qindex`;
    /// with delta-q off every block's `current_qindex` is this constant).
    pub base_qindex: i32,
    /// `delta_q_info.delta_q_present_flag` — per-SB delta-q signaling
    /// (`--deltaq-mode=6`; requires [`SbEncodeEnv::deltaq`] to derive the
    /// per-SB qindexes). False = the proven envelope, byte-identical.
    pub delta_q_present: bool,
    /// `delta_q_info.delta_q_res` (1/2/4/8) — read only when
    /// `delta_q_present`.
    pub delta_q_res: i32,
    /// `cm->features.allow_screen_content_tools` — gates PALETTE mode per
    /// block (`av1_allow_palette`, also needs the block's own bsize in
    /// `[BLOCK_8X8, 64x64]`). When true, every eligible DC-predicted block
    /// codes a palette-usage flag (this envelope never uses palette, so the
    /// flag is always the `no-palette` symbol) — the decoder unconditionally
    /// reads it, so omitting it desyncs the whole tile-group. MUST equal the
    /// value the real frame header carries.
    pub allow_screen_content_tools: bool,
    /// The frame header's `allow_intrabc` bit — when true, every KEY block
    /// codes a `use_intrabc` flag (and its DV when set), exactly as the decoder
    /// reads it. MUST equal the value the real frame header carries.
    pub allow_intrabc: bool,
}

/// Per-64x64-unit CDEF signalling inputs for a CDEF-enabled pack walk —
/// the mi-grid facts C's `write_cdef` reads (`cm->cdef_info.cdef_bits` +
/// the `mbmi->cdef_strength` the search stamped at each unit's top-left
/// mi, `bitstream.c:880-919`). Absent (`None` on [`MiNbrGrid::cdef`]) for
/// the CDEF-off envelope, where the strength literal is zero-width.
#[derive(Clone, Debug)]
pub struct CdefPackState {
    /// `cdef_info.cdef_bits` — the strength-literal width (0..=3).
    pub cdef_bits: u32,
    /// Per-64x64-unit strength index, `nvfb x nhfb` raster
    /// ([`crate::pickcdef::CdefSearchResult::unit_strength`]). Units the
    /// search never stamped hold 0 and are never written (all their blocks
    /// are skip).
    pub unit_strength: Vec<i32>,
    /// Units per row (`nhfb`).
    pub nhfb: i32,
}

/// Per-MI-position neighbour tracking for [`write_mb_modes_kf_fc`]'s
/// `above`/`left: Option<MiNbrKf>` — the same shape/reset discipline as
/// [`TileCtxState`]'s other above/left arrays (above indexed by absolute
/// `mi_col`, zeroed once per tile; left indexed by `mi_row & 31`, zeroed at
/// each SB row) — plus the frame-constant CDEF signalling inputs (also
/// per-tile pack-walk state; `None` = CDEF off).
pub struct MiNbrGrid {
    above: Vec<Option<MiNbrKf>>,
    left: [Option<MiNbrKf>; 32],
    /// CDEF signalling inputs for [`pack_leaf`]'s `write_cdef` path.
    pub cdef: Option<CdefPackState>,
    /// The palette twin (`xd->above_mbmi->palette_mode_info` projection),
    /// stamped alongside `above`/`left` per PACKED leaf — the write-side
    /// colour-cache (`av1_get_palette_cache` inside `write_palette_mode_info`)
    /// and UV-flag context inputs. `None` = neighbour absent OR palette-free
    /// (both read as size 0 / no colours).
    pal_above: Vec<Option<PaletteNbrKf>>,
    pal_left: [Option<PaletteNbrKf>; 32],
}

impl MiNbrGrid {
    /// All-absent neighbours (tile start: `av1_zero_above_context`'s
    /// mode-info analogue — no MI has been coded yet). CDEF off.
    pub fn zeroed(mi_cols: usize) -> Self {
        MiNbrGrid {
            above: vec![None; mi_cols],
            left: [None; 32],
            cdef: None,
            pal_above: vec![None; mi_cols],
            pal_left: [None; 32],
        }
    }
    /// `av1_zero_left_context`'s mode-info analogue: called at each SB row
    /// start.
    pub fn zero_left(&mut self) {
        self.left = [None; 32];
        self.pal_left = [None; 32];
    }
    #[allow(clippy::too_many_arguments)]
    fn stamp(
        &mut self,
        mi_row: i32,
        mi_col: i32,
        mi_w: usize,
        mi_h: usize,
        mi_cols: i32,
        mi_rows: i32,
        nbr: MiNbrKf,
        pal: Option<PaletteNbrKf>,
    ) {
        let a0 = mi_col as usize;
        // C clips the mode-info write to x_mis/y_mis (av1_update_state,
        // encodeframe_utils.c:353): a partial edge block writes only its
        // in-frame mi cells, leaving off-frame neighbour slots untouched.
        let x_mis = mi_w.min((mi_cols - mi_col).max(0) as usize);
        let y_mis = mi_h.min((mi_rows - mi_row).max(0) as usize);
        for x in self.above[a0..a0 + x_mis].iter_mut() {
            *x = Some(nbr);
        }
        for x in self.pal_above[a0..a0 + x_mis].iter_mut() {
            *x = pal;
        }
        let l0 = (mi_row & 31) as usize;
        for x in self.left[l0..l0 + y_mis].iter_mut() {
            *x = Some(nbr);
        }
        for x in self.pal_left[l0..l0 + y_mis].iter_mut() {
            *x = pal;
        }
    }
}

/// A frame-constant [`KfBlockState`] (segmentation/palette/intrabc/delta-q
/// all off, matching [`rd_pick_partition_real`]'s stated envelope) — the
/// per-leaf fields (`mi_row`/`mi_col`/`bsize`/`is_chroma_ref`/`cfl_allowed`/
/// `has_above`/`has_left`) are overwritten by [`pack_leaf`] every call; the
/// mutable carries (`cdef_transmitted`/`current_base_qindex`/`xd_delta_lf*`)
/// self-reset at each SB's upper-left position (`write_cdef`'s own
/// `mi_row/col & sb_mask == 0` check) so one instance can be reused for the
/// whole tile.
pub fn kf_block_state(cfg: &PackCfg, env: &SbEncodeEnv, mib_size: i32) -> KfBlockState {
    KfBlockState {
        segid_preskip: false,
        seg_enabled: false,
        update_map: false,
        seg_pred: 0,
        seg_cdf_num: 0,
        last_active_segid: 0,
        seg_skip_feature: [false; 8],
        mi_row: 0,
        mi_col: 0,
        mib_size,
        sb_size: env.sb_size,
        bsize: env.sb_size,
        coded_lossless: env.lossless,
        allow_intrabc: cfg.allow_intrabc,
        cdef_bits: 0,
        dq_present: cfg.delta_q_present,
        // `--delta-lf-mode=1`: delta-lf rides on the delta-q framework
        // (`env.deltaq` Some => a firing delta-q mode). `delta_lf_res =
        // DEFAULT_DELTA_LF_RES` (2), `delta_lf_multi = DEFAULT_DELTA_LF_MULTI`
        // (0/single). encodeframe.c:2321-2324.
        dlf_present: env.deltaq.map_or(false, |d| d.delta_lf_present),
        dlf_multi: false,
        num_planes: if env.monochrome { 1 } else { 3 },
        dq_res: cfg.delta_q_res,
        dlf_res: if env.deltaq.map_or(false, |d| d.delta_lf_present) {
            DELTA_LF_RES
        } else {
            0
        },
        monochrome: env.monochrome,
        is_chroma_ref: true,
        cfl_allowed: false,
        allow_palette: false,
        bit_depth: i32::from(env.bd),
        filter_allowed: false,
        mb_to_top_edge: 0,
        has_above: false,
        has_left: false,
        cdef_transmitted: [false; 4],
        current_base_qindex: cfg.base_qindex,
        xd_delta_lf: [0; 4],
        xd_delta_lf_from_base: 0,
    }
}

/// Pack (`OUTPUT_ENABLED`) one `PARTITION_NONE`/rect leaf: mode-info, the
/// luma tx-size symbol (if signaled), then every coded plane's coefficient
/// bytes — libaom's exact `write_modes_b` order (`write_mbmi_b` -> [palette,
/// excluded] -> tx_size -> `write_tokens_b`, `bitstream.c:1516-1567`).
#[allow(clippy::too_many_arguments)]
pub fn pack_leaf(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    cfg: &PackCfg,
    kf: &mut KfFrameContext,
    kfs: &mut KfBlockState,
    tile: &mut TileCtxState,
    nbr: &mut MiNbrGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    winner: &mut LeafWinner,
    mi_row: i32,
    mi_col: i32,
    partition: usize,
    sb_current_qindex: i32,
) {
    let bsize = winner.bsize;
    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_h = MI_SIZE_HIGH_B[bsize];
    let is_chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    let cfl_allowed = is_cfl_allowed(bsize, env.lossless, env.ss_x, env.ss_y);
    let has_above = mi_row > env.tile_row_start;
    let has_left = mi_col > env.tile_col_start;

    // ---- 1. write_mbmi_b: mode-info (write_mb_modes_kf_fc). ----
    let above_nbr = nbr.above[mi_col as usize];
    let left_nbr = nbr.left[(mi_row & 31) as usize];
    // write_cdef reads the strength stamped at the CDEF unit's FIRST mi
    // (bitstream.c:909-915: `mi_grid_base[mi_row & ~(cdef_size-1),
    // mi_col & ~..]->cdef_strength`) — the per-unit grid lookup below IS
    // that read (a 64x64 unit = 16 mi; `>> 4` == `& !15` then unit index).
    // 0 when CDEF is off: the literal is zero-width there, never coded.
    let cdef_strength = nbr.cdef.as_ref().map_or(0, |c| {
        c.unit_strength[((mi_row >> 4) * c.nhfb + (mi_col >> 4)) as usize]
    });
    // The winner's palette projection (Y + UV halves) — feeds the mode-info
    // syntax AND the neighbour stamp.
    let mut palette_size = [0i32; 2];
    let mut palette_colors = [0u16; 24];
    if let Some(p) = &winner.palette_y {
        palette_size[0] = p.size as i32;
        palette_colors[..p.size].copy_from_slice(&p.colors[..p.size]);
    }
    if let Some(p) = &winner.palette_uv {
        palette_size[1] = p.size as i32;
        palette_colors[8..8 + p.size].copy_from_slice(&p.colors_u[..p.size]);
        palette_colors[16..16 + p.size].copy_from_slice(&p.colors_v[..p.size]);
    }
    // `--delta-lf-mode=1` per-SB delta-lf (setup_delta_q, encodeframe.c:380-383):
    // `delta_lf_from_base = ((delta_qindex/4 + res/2) & ~(res-1))` clamped to
    // [-MAX_LOOP_FILTER, MAX_LOOP_FILTER], where `delta_qindex = current_qindex
    // - FRAME base_qindex`. Every leaf in an SB derives the same value (shared
    // SB qindex); only the SB-root leaf actually codes it (super_block_upper_left
    // gate in write_delta_q_params_sb). `delta_lf[lf_id]` mirrors it (multi off).
    let delta_lf_val = if env.deltaq.map_or(false, |d| d.delta_lf_present) {
        let delta_qindex = sb_current_qindex - cfg.base_qindex;
        let lfmask = !(DELTA_LF_RES - 1);
        let dlf = (delta_qindex / 4 + DELTA_LF_RES / 2) & lfmask;
        dlf.clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER)
    } else {
        0
    };
    let info = MbModeInfoKf {
        segment_id: 0,
        skip: i32::from(winner.skip_txfm),
        cdef_strength,
        current_qindex: sb_current_qindex,
        delta_lf: [delta_lf_val; 4],
        delta_lf_from_base: delta_lf_val,
        use_intrabc: i32::from(winner.use_intrabc),
        // write_intrabc_info signals the diff dv - dv_ref (encode_mv,
        // MV_SUBPEL_NONE); dead (0) for a non-intrabc leaf.
        dv_row: if winner.use_intrabc {
            winner.dv_row - winner.dv_ref_row
        } else {
            0
        },
        dv_col: if winner.use_intrabc {
            winner.dv_col - winner.dv_ref_col
        } else {
            0
        },
        y_mode: winner.mode as i32,
        angle_delta_y: winner.angle_delta_y,
        uv_mode: winner.uv_mode as i32,
        cfl_alpha_idx: winner.cfl_alpha_idx,
        cfl_joint_sign: winner.cfl_alpha_signs,
        angle_delta_uv: winner.angle_delta_uv,
        palette_size,
        palette_colors,
        use_filter_intra: i32::from(winner.use_filter_intra),
        filter_intra_mode: winner.filter_intra_mode as i32,
    };
    kfs.mi_row = mi_row;
    kfs.mi_col = mi_col;
    kfs.bsize = bsize;
    kfs.is_chroma_ref = is_chroma_ref;
    kfs.cfl_allowed = cfl_allowed;
    kfs.has_above = has_above;
    kfs.has_left = has_left;
    // `av1_allow_palette(cm->features.allow_screen_content_tools, bsize)`
    // (blockd.h) — the SAME per-block gate the decoder applies
    // (`aom-decode`'s `decode_block`: `st.allow_palette =
    // av1_allow_palette(...)`). When screen-content tools are on, every
    // DC-predicted block in `[BLOCK_8X8, 64x64]` codes a palette-usage flag;
    // this envelope never uses palette, so `write_mb_modes_kf_fc` emits the
    // `no-palette` symbol, but the flag MUST still be written or the decoder
    // (which reads it unconditionally) desyncs from here to the tile end.
    kfs.allow_palette = allow_palette(cfg.allow_screen_content_tools, bsize);
    // xd->mb_to_top_edge: the palette colour-cache's SB-row gate
    // (av1_get_palette_cache reads `row = -mb_to_top_edge >> 3`).
    kfs.mb_to_top_edge = -((mi_row * 4) * 8);
    // The above/left neighbour palette projections (xd->above_mbmi/left_mbmi),
    // tracked leaf-by-leaf like the y_mode/skip neighbours.
    let above_pal = if has_above {
        nbr.pal_above[mi_col as usize]
    } else {
        None
    };
    let left_pal = if has_left {
        nbr.pal_left[(mi_row & 31) as usize]
    } else {
        None
    };
    write_mb_modes_kf_fc(
        enc,
        &info,
        kf,
        kfs,
        cfg.enable_filter_intra,
        above_nbr,
        left_nbr,
        above_pal,
        left_pal,
    );

    // ---- 1b. palette colour-map tokens (write_modes_b's palette loop,
    //      bitstream.c:1520-1533: after the mode info, before tx_size) —
    //      the wavefront token walk over the VISIBLE rows x cols of the
    //      (block-width-stride) winning map. ----
    if let Some(p) = &winner.palette_y {
        let block_width = 4 * mi_w;
        let bwv = mi_w.min((env.mi_cols - mi_col).max(0) as usize);
        let bhv = mi_h.min((env.mi_rows - mi_row).max(0) as usize);
        encode_color_map_tokens(
            enc,
            p.size as i32,
            block_width,
            &p.color_map,
            bhv * 4,
            bwv * 4,
            &mut kf.palette_y_color_index[p.size - 2],
        );
    }
    if let Some(p) = &winner.palette_uv {
        // Chroma plane dims (av1_get_block_dimensions(bsize, 1)), incl. the
        // sub-8x8 correction.
        let (pw, _ph, rows, cols) = crate::palette_search::chroma_block_dims(
            bsize,
            mi_row,
            mi_col,
            env.mi_rows,
            env.mi_cols,
            env.ss_x,
            env.ss_y,
        );
        encode_color_map_tokens(
            enc,
            p.size as i32,
            pw,
            &p.color_map,
            rows,
            cols,
            &mut kf.palette_uv_color_index[p.size - 2],
        );
    }

    // ---- 2. tx_size symbol (write_selected_tx_size), gated exactly as
    //     write_modes_b's TX_MODE_SELECT branch (bitstream.c:1538-1548); for
    //     intra `is_inter_tx` is always false so that branch collapses to
    //     `tx_mode_is_select && block_signals_txsize(bsize) && !lossless`.
    //     Reads the PRE-block above/left txfm context (before this leaf's
    //     own stamp) -- the stamp itself happens inside encode_b_intra_dry
    //     below (its step 6), unconditionally, matching set_txfm_ctxs being
    //     called with the same args on both branch sides for intra. ----
    // Intrabc leaf: the block is is_inter, so the tx size follows the INTER
    // path. In this port's skip-only intrabc scope the block always chose the
    // skip arm ⇒ no var-tx quadtree symbol and no coeffs (the decoder infers
    // the fallback tx size from tx_mode, frame.rs:1933) — so the whole tx-size
    // symbol + coeff write below is skipped (guarded by `!winner.use_intrabc`
    // here and `!winner.skip_txfm` at the coeff write).
    // Intrabc COEFF arm: `is_inter_tx` is true and `skip_txfm` false, so C
    // takes the var-tx branch (bitstream.c:1542-1552) — the
    // `write_tx_size_vartx` raster over max_tx_size units, which reads AND
    // updates the txfm-partition contexts itself (no `set_txfm_ctxs` here, and
    // the encode walk deliberately skipped its stamp on the OUTPUT path).
    if cfg.tx_mode_is_select && bsize > 0 && !env.lossless && winner.use_intrabc && !winner.skip_txfm
    {
        let max_tx = crate::tx_search::MAX_TXSIZE_RECT_LOOKUP[bsize];
        let txbh = crate::var_tx::TX_SIZE_HIGH_UNIT[max_tx];
        let txbw = crate::var_tx::TX_SIZE_WIDE_UNIT[max_tx];
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;
        let (_, _, mb_to_right, mb_to_bottom) = crate::tx_search::max_block_units(
            env.mi_cols,
            env.mi_rows,
            mi_col,
            mi_row,
            mi_w as i32,
            mi_h as i32,
            crate::tx_search::BLK_W_B[bsize],
            crate::tx_search::BLK_H_B[bsize],
            0,
            0,
        );
        let mut idy = 0usize;
        while idy < mi_h {
            let mut idx = 0usize;
            while idx < mi_w {
                aom_dsp::entropy::partition::write_tx_size_vartx(
                    enc,
                    &mut kf.txfm_partition,
                    bsize,
                    &winner.inter_tx_size,
                    mb_to_right,
                    mb_to_bottom,
                    &mut tile.above_tctx[a0..a0 + mi_w],
                    &mut tile.left_tctx[l0..l0 + mi_h],
                    max_tx,
                    0,
                    idy as i32,
                    idx as i32,
                );
                idx += txbw;
            }
            idy += txbh;
        }
    }

    if cfg.tx_mode_is_select && bsize > 0 && !env.lossless && !winner.use_intrabc {
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;
        let ctx = get_tx_size_context(
            bsize,
            tile.above_tctx[a0],
            tile.left_tctx[l0],
            has_above,
            has_left,
            None,
            None,
        );
        let cat = bsize_to_tx_size_cat(bsize) as usize;
        let depth = tx_size_to_depth(winner.tx_size, bsize);
        let max_depths = bsize_to_max_depth(bsize);
        write_selected_tx_size(enc, &mut kf.tx_size[cat][ctx], bsize, depth, max_depths);
    }

    // ---- 3. residual/coefficient recompute (reuses the validated dry-run
    //     leaf encode -- see module docs for why this reproduces the true
    //     OUTPUT_ENABLED result in this envelope). ----
    // OUTPUT_ENABLED (C bitstream write == the same walk as the SB-root
    // winner encode): the winner tx_type_map must arrive here exactly as the
    // SEARCH left it — both this walk and the search's SB-root walk model
    // C's single OUTPUT_ENABLED pass, whose eob-0 resets go to the frame map,
    // never back into ctx (see encode_b_intra_dry's doc).
    let out = encode_b_intra_dry(
        env, tile, recon_y, recon_u, recon_v, cfl, winner, mi_row, mi_col, partition, true,
    );

    // ---- 4. write_tokens_b: coefficient bytes, gated on !skip_txfm (always
    //     true in the KEY intra envelope, asserted by encode_b_intra_dry). ----
    if !winner.skip_txfm && winner.use_intrabc {
        // INTER (intrabc) coefficients: `write_tokens_b`'s inter arm
        // (bitstream.c:1444-1471) — 64x64-chunk outer in LUMA mi units, planes
        // INTERLEAVED per chunk, each plane's sub-walk being
        // `write_inter_txb_coeff` (:1395) over `get_vartx_max_txsize` units.
        // Luma descends the `inter_tx_size` quadtree (pack_txb_tokens); chroma
        // is UNIFORM. The re-encode produced each plane's txbs in exactly this
        // order, so per-plane cursors consume them in lockstep.
        let uv_tx = av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
        let (max_bw, max_bh, mb_to_right, mb_to_bottom) = crate::tx_search::max_block_units(
            env.mi_cols,
            env.mi_rows,
            mi_col,
            mi_row,
            mi_w as i32,
            mi_h as i32,
            TXS_W[0] * mi_w, // block_size_wide[bsize] = 4 * mi_w
            TXS_H[0] * mi_h,
            0,
            0,
        );
        let mu_w = MI_SIZE_WIDE_B[12].min(mi_w);
        let mu_h = MI_SIZE_HIGH_B[12].min(mi_h);
        let max_tx = crate::tx_search::MAX_TXSIZE_RECT_LOOKUP[bsize];
        let (bkw, bkh) = (
            crate::var_tx::TX_SIZE_WIDE_UNIT[max_tx],
            crate::var_tx::TX_SIZE_HIGH_UNIT[max_tx],
        );
        let (utxw_u, utxh_u) = (TXS_W[uv_tx] >> 2, TXS_H[uv_tx] >> 2);
        let (mut yc, mut uc, mut vc) = (0usize, 0usize, 0usize);
        let has_uv = out.u.is_some();
        let plane_bsize_pack =
            aom_dsp::entropy::partition::get_plane_block_size(bsize, env.ss_x, env.ss_y);
        let (cvis_w, cvis_h, _, _) = crate::tx_search::max_block_units(
            env.mi_cols,
            env.mi_rows,
            mi_col,
            mi_row,
            mi_w as i32,
            mi_h as i32,
            crate::tx_search::BLK_W_B[plane_bsize_pack],
            crate::tx_search::BLK_H_B[plane_bsize_pack],
            env.ss_x,
            env.ss_y,
        );
        let mut row = 0usize;
        while row < mi_h {
            let mut col = 0usize;
            while col < mi_w {
                // plane 0 (Y): the var-tx quadtree over this chunk's units.
                let uh = (row + mu_h).min(mi_h);
                let uw = (col + mu_w).min(mi_w);
                let mut br = row;
                while br < uh {
                    let mut bc = col;
                    while bc < uw {
                        pack_vartx_txb(
                            enc, kf, cfg, env, winner, &out.y.txbs, &mut yc, br, bc, max_tx,
                            max_bw, max_bh,
                        );
                        bc += bkw;
                    }
                    br += bkh;
                }
                // planes 1/2: uniform uv tx over the subsampled chunk.
                if has_uv {
                    let cuh = ((row + mu_h) >> env.ss_y).min(cvis_h);
                    let cuw = ((col + mu_w) >> env.ss_x).min(cvis_w);
                    for (plane, cursor) in [(1usize, &mut uc), (2usize, &mut vc)] {
                        let txbs = if plane == 1 {
                            &out.u.as_ref().unwrap().txbs
                        } else {
                            &out.v.as_ref().unwrap().txbs
                        };
                        let mut cbr = row >> env.ss_y;
                        while cbr < cuh {
                            let mut cbc = col >> env.ss_x;
                            while cbc < cuw {
                                if *cursor < txbs.len() {
                                    write_one_txb_inter(
                                        enc,
                                        kf,
                                        cfg,
                                        env,
                                        &txbs[*cursor],
                                        uv_tx,
                                        plane,
                                    );
                                    *cursor += 1;
                                }
                                cbc += utxw_u;
                            }
                            cbr += utxh_u;
                        }
                    }
                }
                col += mu_w;
            }
            row += mu_h;
        }
    }

    // The INTRA coefficient write (av1_write_intra_coeffs_mb). Excludes the
    // intrabc COEFF arm handled above; step 5 below runs for BOTH.
    if !winner.skip_txfm && !winner.use_intrabc {
        let uv_tx = av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
        // av1_write_intra_coeffs_mb (encodetxb.c:431-472): the 64x64-chunk
        // outer walk, planes INTERLEAVED per chunk (L, then U, then V) — the
        // decoder's read order for coding blocks > 64x64 (KB-1 encoder
        // cross-check). The recon walk produced each plane's txbs in this same
        // chunk-major order, so per-plane cursors consume them in lockstep.
        // Luma mu = mi_size_wide[BLOCK_64X64] = 16; degenerates to
        // plane-sequential (a single chunk) for bsize <= 64x64 — byte-identical
        // to the former three plane loops.
        let max_bw = mi_w.min((env.mi_cols - mi_col).max(0) as usize);
        let max_bh = mi_h.min((env.mi_rows - mi_row).max(0) as usize);
        let mu_w = MI_SIZE_WIDE_B[12].min(max_bw); // BLOCK_64X64
        let mu_h = MI_SIZE_HIGH_B[12].min(max_bh);
        let (ytxw_u, ytxh_u) = (TXS_W[winner.tx_size] >> 2, TXS_H[winner.tx_size] >> 2);
        let (utxw_u, utxh_u) = (TXS_W[uv_tx] >> 2, TXS_H[uv_tx] >> 2);
        let (mut yc, mut uc, mut vc) = (0usize, 0usize, 0usize);
        let mut row = 0usize;
        while row < max_bh {
            let mut col = 0usize;
            while col < max_bw {
                // plane 0 (Y): raster over this chunk's luma tx blocks.
                let uh = (row + mu_h).min(max_bh);
                let uw = (col + mu_w).min(max_bw);
                let mut br = row;
                while br < uh {
                    let mut bc = col;
                    while bc < uw {
                        write_one_txb(enc, kf, cfg, env, winner, &out.y.txbs[yc], winner.tx_size, 0);
                        yc += 1;
                        bc += ytxw_u;
                    }
                    br += ytxh_u;
                }
                // planes 1/2 (U, then V): ss-scaled chunk bounds (C breaks the
                // plane loop when !is_chroma_ref — mono / sub-8x8 luma-only).
                if is_chroma_ref {
                    let cuh = round_pow2((row + mu_h).min(max_bh), env.ss_y);
                    let cuw = round_pow2((col + mu_w).min(max_bw), env.ss_x);
                    if let Some(u) = &out.u {
                        let mut br = row >> env.ss_y;
                        while br < cuh {
                            let mut bc = col >> env.ss_x;
                            while bc < cuw {
                                write_one_txb(enc, kf, cfg, env, winner, &u.txbs[uc], uv_tx, 1);
                                uc += 1;
                                bc += utxw_u;
                            }
                            br += utxh_u;
                        }
                    }
                    if let Some(v) = &out.v {
                        let mut br = row >> env.ss_y;
                        while br < cuh {
                            let mut bc = col >> env.ss_x;
                            while bc < cuw {
                                write_one_txb(enc, kf, cfg, env, winner, &v.txbs[vc], uv_tx, 2);
                                vc += 1;
                                bc += utxw_u;
                            }
                            br += utxh_u;
                        }
                    }
                }
                col += mu_w;
            }
            row += mu_h;
        }
        debug_assert_eq!(yc, out.y.txbs.len(), "mu-64 pack consumed all Y txbs");
        debug_assert!(
            out.u.as_ref().is_none_or(|u| uc == u.txbs.len()),
            "mu-64 pack consumed all U txbs"
        );
        debug_assert!(
            out.v.as_ref().is_none_or(|v| vc == v.txbs.len()),
            "mu-64 pack consumed all V txbs"
        );
    }

    // ---- 5. neighbour-grid stamp for the next block's Y-mode/skip ctx. ----
    let nbr_kf = MiNbrKf {
        y_mode: winner.mode as i32,
        skip_txfm: i32::from(winner.skip_txfm),
    };
    let nbr_pal =
        (winner.palette_y.is_some() || winner.palette_uv.is_some()).then_some(PaletteNbrKf {
            size: palette_size,
            colors: palette_colors,
        });
    nbr.stamp(
        mi_row,
        mi_col,
        mi_w,
        mi_h,
        env.mi_cols,
        env.mi_rows,
        nbr_kf,
        nbr_pal,
    );
}

/// Pack-stage single-txb coefficient write: emit one txb's bytes via
/// `write_coeffs_txb_full`, reusing the `txb_skip_ctx`/`dc_sign_ctx` the
/// residual recompute already derived (the SAME pair the trellis used to
/// select its rate tables). Called from [`pack_leaf`]'s mu-64 interleaved walk
/// (`av1_write_intra_coeffs_mb`, encodetxb.c:431-472).
#[allow(clippy::too_many_arguments)]
fn write_one_txb(
    enc: &mut OdEcEnc,
    kf: &mut KfFrameContext,
    cfg: &PackCfg,
    env: &SbEncodeEnv,
    winner: &LeafWinner,
    txb: &TxbEncode,
    tx_size: usize,
    plane: usize,
) {
    let plane_type = usize::from(plane > 0);
    let mut dummy = [0u16; 8];
    let ext_tx_cdf: &mut [u16] = if plane_type == 0 {
        let d = ext_tx_derive(
            tx_size,
            false, // is_inter
            env.reduced_tx_set_used,
            txb.tx_type,
            winner.use_filter_intra,
            winner.filter_intra_mode,
            winner.mode,
        );
        match d.eset {
            1 => &mut kf.ext_tx_1ddct[d.square as usize][d.intra_dir as usize],
            2 => &mut kf.ext_tx_dtt4[d.square as usize][d.intra_dir as usize],
            _ => &mut dummy[..],
        }
    } else {
        &mut dummy[..]
    };
    write_coeffs_txb_full(
        enc,
        &mut kf.coeff,
        ext_tx_cdf,
        &txb.qcoeff,
        txb.eob as usize,
        tx_size,
        txb.tx_type,
        plane_type,
        txb.txb_skip_ctx,
        txb.dc_sign_ctx,
        cfg.allow_update_cdf,
        false, // is_inter
        env.reduced_tx_set_used,
        winner.use_filter_intra,
        winner.filter_intra_mode,
        winner.mode,
        cfg.signal_gate,
    );
}

/// Pack (`OUTPUT_ENABLED`) walk over a picked [`SbTree`]: write the
/// partition symbol at each node (`write_modes_sb`'s exact recursion,
/// `bitstream.c`, extended to the `HORZ`/`VERT` rect shapes this port's
/// search also produces — the upstream `write_modes_b`/`write_modes_sb` in
/// `aom-entropy` only handle `NONE`/`SPLIT`), then dispatch [`pack_leaf`] at
/// each `PARTITION_NONE`/rect sub-block. Mirrors
/// [`crate::encode_sb::encode_sb_dry`] shape-for-shape (frame-bound gating
/// for `HORZ`'s/`VERT`'s second sub-block included).
#[allow(clippy::too_many_arguments)]
pub fn pack_sb(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    cfg: &PackCfg,
    kf: &mut KfFrameContext,
    kfs: &mut KfBlockState,
    tile: &mut TileCtxState,
    nbr: &mut MiNbrGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    tree: &mut SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    sb_current_qindex: i32,
) {
    if mi_row >= env.mi_rows || mi_col >= env.mi_cols {
        return;
    }
    let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
    let mi_step = hbs;
    let has_rows = mi_row + mi_step < env.mi_rows;
    let has_cols = mi_col + mi_step < env.mi_cols;

    let (p, subsize) = match tree {
        SbTree::Leaf(_) => (PARTITION_NONE, bsize),
        SbTree::Split(_) => (PARTITION_SPLIT, split_subsize(bsize)),
        SbTree::Horz(_) => (
            PARTITION_HORZ,
            get_partition_subsize(bsize, PARTITION_HORZ) as usize,
        ),
        SbTree::Vert(_) => (
            PARTITION_VERT,
            get_partition_subsize(bsize, PARTITION_VERT) as usize,
        ),
        SbTree::Horz4(_) => (
            PARTITION_HORZ_4,
            get_partition_subsize(bsize, PARTITION_HORZ_4) as usize,
        ),
        SbTree::Vert4(_) => (
            PARTITION_VERT_4,
            get_partition_subsize(bsize, PARTITION_VERT_4) as usize,
        ),
        SbTree::HorzA(_) => (
            PARTITION_HORZ_A,
            get_partition_subsize(bsize, PARTITION_HORZ_A) as usize,
        ),
        SbTree::HorzB(_) => (
            PARTITION_HORZ_B,
            get_partition_subsize(bsize, PARTITION_HORZ_B) as usize,
        ),
        SbTree::VertA(_) => (
            PARTITION_VERT_A,
            get_partition_subsize(bsize, PARTITION_VERT_A) as usize,
        ),
        SbTree::VertB(_) => (
            PARTITION_VERT_B,
            get_partition_subsize(bsize, PARTITION_VERT_B) as usize,
        ),
        // Off-frame placeholder: unreachable here (the entry guard returned
        // for its off-frame origin), but the match must be exhaustive.
        SbTree::Absent => return,
    };

    if bsize >= 3 {
        // BLOCK_8X8: write_partition's own internal size gate makes this
        // redundant for smaller sizes, but partition_plane_context's
        // neighbour read is only meaningful (and only computed by the
        // search) from 8x8 up -- matches rd_pick_partition_real's own
        // `bsize_at_least_8x8` gate.
        let ctx = partition_plane_context(
            &tile.above_pctx,
            &tile.left_pctx,
            mi_row as usize,
            mi_col as usize,
            bsize,
        ) as usize;
        write_partition(
            enc,
            &mut kf.partition[ctx],
            partition_cdf_length(bsize),
            p,
            has_rows,
            has_cols,
            bsize,
        );
    }

    match tree {
        SbTree::Leaf(w) => {
            pack_leaf(
                enc,
                env,
                cfg,
                kf,
                kfs,
                tile,
                nbr,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                w,
                mi_row,
                mi_col,
                PARTITION_NONE as usize,
                sb_current_qindex,
            );
        }
        SbTree::Split(children) => {
            for (idx, child) in children.iter_mut().enumerate() {
                let y = mi_row + ((idx as i32) >> 1) * hbs;
                let x = mi_col + ((idx as i32) & 1) * hbs;
                pack_sb(
                    enc, env, cfg, kf, kfs, tile, nbr, recon_y, recon_u, recon_v, cfl, child, y, x,
                    subsize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::Horz(subs) => {
            let [s0, s1] = &mut **subs;
            pack_leaf(
                enc,
                env,
                cfg,
                kf,
                kfs,
                tile,
                nbr,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_HORZ as usize,
                sb_current_qindex,
            );
            if mi_row + hbs < env.mi_rows {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s1,
                    mi_row + hbs,
                    mi_col,
                    PARTITION_HORZ as usize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::Vert(subs) => {
            let [s0, s1] = &mut **subs;
            pack_leaf(
                enc,
                env,
                cfg,
                kf,
                kfs,
                tile,
                nbr,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_VERT as usize,
                sb_current_qindex,
            );
            if mi_col + hbs < env.mi_cols {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s1,
                    mi_row,
                    mi_col + hbs,
                    PARTITION_VERT as usize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::Horz4(subs) => {
            // encode_sb PARTITION_HORZ_4 (:1690-1697): 4 strips at
            // mi_row + i*quarter_step, i>0 gated by the frame bound.
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, s) in subs.iter_mut().enumerate() {
                let this_mi_row = mi_row + (i as i32) * quarter_step;
                if i > 0 && this_mi_row >= env.mi_rows {
                    break;
                }
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    this_mi_row,
                    mi_col,
                    PARTITION_HORZ_4 as usize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::Vert4(subs) => {
            // encode_sb PARTITION_VERT_4 (:1699-1705): 4 strips at
            // mi_col + i*quarter_step, i>0 gated by the frame bound.
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, s) in subs.iter_mut().enumerate() {
                let this_mi_col = mi_col + (i as i32) * quarter_step;
                if i > 0 && this_mi_col >= env.mi_cols {
                    break;
                }
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    mi_row,
                    this_mi_col,
                    PARTITION_VERT_4 as usize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::HorzA(subs) => {
            // encode_sb PARTITION_HORZ_A (:1652-1660): interior-only, no
            // frame-bound gating on any of the 3 sub-blocks (module docs on
            // encode_sb.rs's SbTree::HorzA).
            let [s0, s1, s2] = &mut **subs;
            for (s, r, c) in [
                (s0, mi_row, mi_col),
                (s1, mi_row, mi_col + hbs),
                (s2, mi_row + hbs, mi_col),
            ] {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    r,
                    c,
                    PARTITION_HORZ_A as usize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::HorzB(subs) => {
            // encode_sb PARTITION_HORZ_B (:1661-1667).
            let [s0, s1, s2] = &mut **subs;
            for (s, r, c) in [
                (s0, mi_row, mi_col),
                (s1, mi_row + hbs, mi_col),
                (s2, mi_row + hbs, mi_col + hbs),
            ] {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    r,
                    c,
                    PARTITION_HORZ_B as usize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::VertA(subs) => {
            // encode_sb PARTITION_VERT_A (:1668-1676): column-axis mirror of
            // HORZ_A.
            let [s0, s1, s2] = &mut **subs;
            for (s, r, c) in [
                (s0, mi_row, mi_col),
                (s1, mi_row + hbs, mi_col),
                (s2, mi_row, mi_col + hbs),
            ] {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    r,
                    c,
                    PARTITION_VERT_A as usize,
                    sb_current_qindex,
                );
            }
        }
        SbTree::VertB(subs) => {
            // encode_sb PARTITION_VERT_B (:1677-1684): column-axis mirror of
            // HORZ_B.
            let [s0, s1, s2] = &mut **subs;
            for (s, r, c) in [
                (s0, mi_row, mi_col),
                (s1, mi_row, mi_col + hbs),
                (s2, mi_row + hbs, mi_col + hbs),
            ] {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    r,
                    c,
                    PARTITION_VERT_B as usize,
                    sb_current_qindex,
                );
            }
        }
        // Off-frame placeholder — unreachable past the entry frame-bound guard.
        SbTree::Absent => {}
    }

    update_ext_partition_context(
        &mut tile.above_pctx,
        &mut tile.left_pctx,
        mi_row,
        mi_col,
        subsize,
        bsize,
        p,
    );
}

/// Pack a whole tile: search ([`rd_pick_partition_real`]) then pack
/// ([`pack_sb`]) each SB in raster order, threading two independently-
/// progressing [`TileCtxState`]s (search's own save/restore keeps it
/// winners-only by the time it returns, so pack's separate forward-only walk
/// stays in lockstep — see the module docs) and the running
/// [`MiNbrGrid`]/[`KfFrameContext`] — libaom's `write_modes`/`encode_sb_row`
/// tile loop: above context zeroed once per tile, left zeroed at each SB row
/// start. `(mi_row0, mi_col0)` is the tile's first SB position (frame-
/// absolute mi units; `0, 0` for a frame-first single tile). Returns the
/// winning trees (one per SB, row-major) for differential visibility; the
/// entropy-coded bytes accumulate in `enc`.
#[allow(clippy::too_many_arguments)]
pub fn pack_tile(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    pick_cfg: &PickFrameCfg,
    pack_cfg: &PackCfg,
    kf: &mut KfFrameContext,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    mi_row0: i32,
    mi_col0: i32,
    n_sb_rows: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
) -> Vec<SbTree> {
    pack_tile_lr(
        enc, env, pick_cfg, pack_cfg, kf, recon_y, recon_u, recon_v, mi_row0, mi_col0, n_sb_rows,
        n_sb_cols, sb_mi, sb_size, None,
    )
}

/// The loop-restoration pack inputs: the frame-level decision
/// (`av1_pick_filter_restoration`'s outcome) whose per-RU parameters are
/// written INTERLEAVED in the tile data at each superblock root, BEFORE the
/// SB's first partition symbol (`loop_restoration_write_sb_coeffs`,
/// av1/encoder/bitstream.c — the exact mirror of the decoder's
/// `loop_restoration_read_sb_coeffs` placement already proven byte-exact on
/// the decode side).
pub struct LrPackParams<'a> {
    /// Frame geometry + per-plane `frame_restoration_type` / unit sizes.
    pub cfg: LrFrameConfig,
    /// Per-plane unit params in unit-grid raster order (empty for
    /// `RESTORE_NONE` planes).
    pub units: [&'a [LrUnitInfo]; 3],
    pub num_planes: usize,
}

/// [`pack_tile`] plus the interleaved loop-restoration unit writes.
/// `lr = None` is byte-identical to [`pack_tile`].
#[allow(clippy::too_many_arguments)]
pub fn pack_tile_lr(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    pick_cfg: &PickFrameCfg,
    pack_cfg: &PackCfg,
    kf: &mut KfFrameContext,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    mi_row0: i32,
    mi_col0: i32,
    n_sb_rows: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
    lr: Option<&LrPackParams<'_>>,
) -> Vec<SbTree> {
    // C write_modes (bitstream.c): `w->allow_update_cdf = !large_scale_tile
    // && !disable_cdf_update` — the tile writer's symbol adaptation gate
    // (aom_write_symbol adapts iff set). large_scale_tile is out of this
    // envelope (0).
    enc.allow_update_cdf = pack_cfg.allow_update_cdf;

    let mi_cols = env.mi_cols as usize;
    let mut search_tile = TileCtxState::zeroed(mi_cols);
    let mut pack_tile_ctx = TileCtxState::zeroed(mi_cols);
    let mut grid = ModeGrid::dc_screen(
        env.mi_rows as usize,
        mi_cols,
        pick_cfg.palette_costs.is_some(),
        pick_cfg.intrabc.is_some(),
    );
    let mut nbr = MiNbrGrid::zeroed(mi_cols);
    let mut kfs = kf_block_state(pack_cfg, env, sb_mi);
    let mut trees = Vec::new();
    // `av1_reset_loop_restoration` (write_modes, bitstream.c): per-tile LR
    // delta-coding references.
    let mut lr_refs = LrRefState::default();

    // `part_sf.partition_search_type == VAR_BASED_PARTITION` — allintra
    // speed >= 7 exactly (speed_features.c:571 is its only allintra setter;
    // `SpeedFeatures::partition_search_type` documents the field, derived
    // inline here per the established `cfg.allintra && cfg.speed >= N`
    // pattern). Flips the per-SB encoder from the RD partition search to
    // `av1_choose_var_based_partitioning` + `av1_rd_use_partition`
    // (encode_rd_sb, encodeframe.c:876-895).
    let use_var_based_partition = pick_cfg.allintra && pick_cfg.speed >= 7;
    let mut vbp_stamps = if use_var_based_partition {
        vec![0u8; env.mi_rows as usize * mi_cols]
    } else {
        Vec::new()
    };
    let vbp_frame = if use_var_based_partition {
        // `cm->width * cm->height` feeds set_vbp_thresholds' <720p arm; the
        // true crop dims aren't threaded to pack_tile, so use the mi-aligned
        // dims and LOUDLY reject the (up-to-3px-per-axis) window where the
        // crop could straddle the 1280x720 boundary and flip the thresholds.
        const RESOLUTION_720P: i64 = 1280 * 720;
        let mi_px = i64::from(env.mi_cols * 4) * i64::from(env.mi_rows * 4);
        let min_crop_px = i64::from(env.mi_cols * 4 - 3) * i64::from(env.mi_rows * 4 - 3);
        assert_eq!(
            mi_px < RESOLUTION_720P,
            min_crop_px < RESOLUTION_720P,
            "VBP threshold resolution arm is crop-ambiguous at {}x{} mi-aligned: \
             thread the true crop dims",
            env.mi_cols * 4,
            env.mi_rows * 4
        );
        Some(crate::var_part::VbpFrame {
            mi_rows: env.mi_rows,
            mi_cols: env.mi_cols,
            // Single-tile envelope: tile ends == frame ends (the env's
            // tile_row_end/tile_col_end are unclamped sentinels).
            tile_mi_row_end: env.mi_rows.min(env.tile_row_end),
            tile_mi_col_end: env.mi_cols.min(env.tile_col_end),
            num_pixels: mi_px,
            sb_size,
            qindex: pick_cfg.qindex,
            bit_depth: env.bd,
            ss_x: if env.monochrome { 1 } else { env.ss_x },
            ss_y: if env.monochrome { 1 } else { env.ss_y },
        })
    } else {
        None
    };
    // Variance Boost delta-q: the SEARCH-side running `xd->current_base_qindex`
    // (reset to base at the tile start, encodeframe.c:1235; advanced per SB by
    // `av1_update_state` — unconditionally on this KEY-intra envelope, where
    // the SB-root `skip_txfm` is structurally 0 so the `bsize != sb_size ||
    // !skip` gate always passes). The WRITE side keeps its own identical
    // tracker inside [`KfBlockState`] (`write_delta_q_params`' semantics).
    let mut search_base_qindex = env
        .deltaq
        .map(|d| d.base_qindex)
        .unwrap_or(pack_cfg.base_qindex);

    for r in 0..n_sb_rows {
        search_tile.left_ectx = [[0; 32]; 3];
        search_tile.left_pctx = [0; 32];
        search_tile.left_tctx = [aom_dsp::entropy::partition::TXFM_CTX_INIT; 32];
        pack_tile_ctx.left_ectx = [[0; 32]; 3];
        pack_tile_ctx.left_pctx = [0; 32];
        pack_tile_ctx.left_tctx = [aom_dsp::entropy::partition::TXFM_CTX_INIT; 32];
        nbr.zero_left();
        for c in 0..n_sb_cols {
            let mi_row = mi_row0 + r * sb_mi;
            let mi_col = mi_col0 + c * sb_mi;

            // `setup_delta_q` (encodeframe.c:341, DELTA_Q_VARIANCE_BOOST):
            // derive this SB's qindex from source variance against the
            // running base, re-select the quantizer rows
            // (`av1_init_plane_quantizers` -> `set_q_index`) and recompute
            // the SB base rdmult from the ADJUSTED qindex (the allintra
            // variance modifier below folds on top, exactly C's
            // init_plane_quantizers -> setup_block_rdmult order).
            let (sb_current_qindex, dq_rows) = if let Some(dq) = &env.deltaq {
                // `setup_delta_q` (encodeframe.c:341): mode 3 (PERCEPTUAL_AI)
                // reads the SB qindex from the precomputed wiener-variance map;
                // mode 6 (VARIANCE_BOOST) derives it from the SB source
                // variance. Both then deadzone-quantize against the running
                // base via `av1_adjust_q_from_delta_q_res`.
                let adjusted = if let Some(map) = dq.perceptual_ai {
                    crate::allintra_vis::setup_delta_q_perceptual_ai(
                        map,
                        dq.base_qindex,
                        env.bd,
                        dq.delta_q_res,
                        dq.sb_mi,
                        mi_row,
                        mi_col,
                        search_base_qindex,
                    )
                } else if let Some(is_screen) = dq.perceptual_wavelet {
                    // `setup_delta_q` (encodeframe.c:330, DELTA_Q_PERCEPTUAL):
                    // the SB source wavelet AC energy → the rate-ratio qindex,
                    // deadzone-quantized against the running base. SB is square
                    // (sb_mi×sb_mi); num_pels_log2 = log2(sb_px²).
                    let sb_off = env.base_y
                        + (mi_row as usize * 4) * env.stride
                        + mi_col as usize * 4;
                    let sb_px = dq.sb_mi as usize * 4;
                    let num_pels_log2 = (sb_px * sb_px).trailing_zeros();
                    crate::allintra_vis::setup_delta_q_perceptual(
                        env.src_y,
                        sb_off,
                        env.stride,
                        env.bd,
                        dq.base_qindex,
                        is_screen,
                        sb_px,
                        sb_px,
                        num_pels_log2,
                        dq.delta_q_res,
                        search_base_qindex,
                    )
                } else {
                    let sb_off = env.base_y
                        + (mi_row as usize * 4) * env.stride
                        + mi_col as usize * 4;
                    crate::allintra_vis::setup_delta_q_variance_boost(
                        env.src_y,
                        sb_off,
                        env.stride,
                        env.bd,
                        dq.base_qindex,
                        dq.deltaq_strength,
                        dq.delta_q_res,
                        search_base_qindex,
                    )
                };
                // av1_update_state: advance the running base (see the init
                // comment for the always-true gate on this envelope).
                search_base_qindex = adjusted;
                let rows = (
                    aom_dsp::quant::set_q_index(dq.quants, dq.deq, adjusted as usize, 0),
                    aom_dsp::quant::set_q_index(dq.quants, dq.deq, adjusted as usize, 1),
                    aom_dsp::quant::set_q_index(dq.quants, dq.deq, adjusted as usize, 2),
                );
                (adjusted, Some(rows))
            } else {
                (pack_cfg.base_qindex, None)
            };
            let sb_base_rdmult = if env.deltaq.is_some() {
                // av1_compute_rd_mult at the SB's adjusted qindex
                // (qindex_rdmult = qindex + y_dc_delta_q, y_dc_delta_q == 0).
                crate::rd::av1_compute_rd_mult_based_on_qindex(
                    env.bd,
                    crate::rd::FrameUpdateType::Kf,
                    sb_current_qindex,
                    if env.tune.iq_tuning {
                        crate::rd::TuneMetric::Iq
                    } else {
                        crate::rd::TuneMetric::Psnr
                    },
                    if pick_cfg.allintra {
                        crate::rd::EncMode::Allintra
                    } else {
                        crate::rd::EncMode::Good
                    },
                )
            } else {
                env.rdmult
            };

            // ALLINTRA SB-root rdmult modifier (setup_block_rdmult,
            // partition_search.c:652/5710-5722): computed ONCE per SB from
            // the whole-SB source variance, then held constant for every
            // node/leaf below it (both the search and the pack walk use the
            // SAME folded env for this SB). NOT on the VAR_BASED_PARTITION
            // path: only av1_rd_pick_partition's SB root recomputes
            // `x->intra_sb_rdmult_modifier` (:5715) — `encode_rd_sb`'s VBP
            // arm leaves it at the per-SB reset 128 (encodeframe.c:1303), so
            // setup_block_rdmult's ALLINTRA fold is IDENTITY at speed >= 7.
            let sb_rdmult = if pick_cfg.allintra && !use_var_based_partition {
                let mi_w = MI_SIZE_WIDE_B[sb_size] as i32;
                let mi_h = MI_SIZE_HIGH_B[sb_size] as i32;
                let ref_off_y =
                    env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
                let mb_to_right_edge = (env.mi_cols - mi_w - mi_col) * 4 * 8;
                let mb_to_bottom_edge = (env.mi_rows - mi_h - mi_row) * 4 * 8;
                let (var_min, var_max) = crate::partition_pick::log_sub_block_var(
                    env.src_y,
                    ref_off_y,
                    env.stride,
                    sb_size,
                    mb_to_right_edge,
                    mb_to_bottom_edge,
                    env.bd,
                );
                let modifier = crate::partition_pick::intra_sb_rdmult_modifier(var_min, var_max);
                crate::partition_pick::fold_intra_sb_rdmult(sb_base_rdmult, modifier)
            } else {
                sb_base_rdmult
            };
            // Coefficient AND mode cost update, `INTERNAL_COST_UPD_SB` (speed 0's
            // default; `av1_set_cost_upd_freq` -> `av1_fill_coeff_costs(&x->coeff_costs,
            // xd->tile_ctx, ...)` AND `av1_fill_mode_rates(cm, &x->mode_costs,
            // xd->tile_ctx)`, encodeframe_utils.c:1643/1658). Real libaom re-derives BOTH
            // the LV_MAP_COEFF_COST / eob tables AND every `av1_fill_mode_rates` table
            // (y_mode / tx_size / angle_delta / partition / skip / cfl / intra tx-type)
            // from the CURRENT (adapting) tile entropy context at the start of every
            // superblock; the search + encode of this SB then use those costs, so as the
            // CDFs adapt over the frame the RD rate tracks them. `kf` adapted through
            // every prior SB's `pack_sb` (the search doesn't touch it), so a full
            // `derive_real_costs(kf, ..)` here reproduces both updates in one shot. Static
            // frame-init costs diverge on steep content, which codes enough symbols to
            // move the CDFs and flip near-tie mode decisions (e.g. DC vs a directional
            // mode on a steep diagonal ramp); SB 0 / single-SB frames read the frame-init
            // defaults unchanged, since nothing adapted yet.
            // KB-12: allintra speed >= 9 flips coeff/mode_cost_upd_level to
            // INTERNAL_COST_UPD_OFF for <4k (set_allintra_speed_feature_
            // framesize_dependent, speed_features.c:166-177 -- runs AFTER the
            // framesize-independent SBROW setting :593-594 and wins) -> the
            // per-SB cost refresh STOPS: every SB reads the FRAME-INIT tables
            // (pick_cfg/env), byte-visible on multi-SB (128 sq) frames.
            // HANDOFF(speed 9): 4k+ frames keep INTERNAL_COST_UPD_SBROW --
            // out of the canon envelope, unmodelled.
            let cost_upd_off = pick_cfg.allintra && pick_cfg.speed >= 9;
            let sb_real = (!cost_upd_off)
                .then(|| crate::real_costs::derive_real_costs(kf, pick_cfg.enable_filter_intra));
            // Variance-Boost delta-q per-SB quantizer rows (None = identity when
            // delta-q is off — env.rows_* — so speeds 0-9 stay byte-unchanged).
            let sb_env = if let Some(sb_real) = &sb_real {
                SbEncodeEnv {
                    rdmult: sb_rdmult,
                    coeff_costs_y: &sb_real.coeff_costs_y,
                    coeff_costs_uv: &sb_real.coeff_costs_uv,
                    tx_type_costs: &sb_real.tx_type_costs_y,
                    rows_y: dq_rows.as_ref().map(|r| &r.0).unwrap_or(env.rows_y),
                    rows_u: dq_rows.as_ref().map(|r| &r.1).unwrap_or(env.rows_u),
                    rows_v: dq_rows.as_ref().map(|r| &r.2).unwrap_or(env.rows_v),
                    ..*env
                }
            } else {
                SbEncodeEnv {
                    rdmult: sb_rdmult,
                    rows_y: dq_rows.as_ref().map(|r| &r.0).unwrap_or(env.rows_y),
                    rows_u: dq_rows.as_ref().map(|r| &r.1).unwrap_or(env.rows_u),
                    rows_v: dq_rows.as_ref().map(|r| &r.2).unwrap_or(env.rows_v),
                    ..*env
                }
            };
            // KB-12: at speed >= 9 (allintra, <4k) `cost_upd_off` is true and
            // `sb_real` is None -> INTERNAL_COST_UPD_OFF: the SB reads the
            // FRAME-INIT cost tables (`pick_cfg`). Speeds 0-8 keep `sb_real`
            // Some -> the per-SB `derive_real_costs` refresh, byte-identical to
            // the pre-Option behaviour.
            let sb_pick_cfg = match &sb_real {
                Some(sb_real) => PickFrameCfg {
                    mode_costs: &sb_real.mode_costs,
                    tx_size_costs: &sb_real.tx_size_costs,
                    skip_costs: &sb_real.skip_costs,
                    tx_type_costs_y: &sb_real.tx_type_costs_y,
                    intra_uv_mode_cost: &sb_real.mode_costs.intra_uv_mode_cost,
                    cfl_costs: &sb_real.cfl_costs,
                    partition_costs: &sb_real.partition_costs,
                    // partition_cdfs stays the FRAME-INIT table (the `..*pick_cfg`
                    // spread): C's `set_partition_cost_for_edge_blk`
                    // (partition_search.c:3415) gathers from `cm->fc->partition_cdf`
                    // — the frame-level context, which does NOT adapt during the
                    // frame — NOT from the per-SB-updated tile context that feeds
                    // `partition_costs` above (a shipped-libaom mixed-source quirk,
                    // measured: C's edge gather rows == default_partition_cdf at
                    // every bottom-edge node of the 196x196 encode while its
                    // interior costs track the adapting tile state).
                    ..*pick_cfg
                },
                None => PickFrameCfg { ..*pick_cfg },
            };

            let mut cfl_search = CflCtx::new(env.ss_x as i32, env.ss_y as i32);
            let mut visits = Vec::new();
            // x->source_variance: 0 at the top of a fresh SB in this
            // single-SB-frame-scoped harness (no prior-SB carry-over
            // modelled — see rd_pick_partition_real's own module docs on
            // this in/out param). By the time any AB stage actually reads
            // it (bsize >= 16x16), an earlier leaf search within THIS SAME
            // node's own NONE/SPLIT/RECT stages has always already
            // overwritten it in every case this port's envelope reaches.
            let mut last_source_variance = 0u32;
            let mut tree = if let Some(vf) = &vbp_frame {
                // encode_rd_sb's VAR_BASED_PARTITION arm (encodeframe.c:
                // 876-895): av1_choose_var_based_partitioning fixes the tree.
                // Speed 7 replays it with the full-RD `av1_rd_use_partition`
                // (do_recon=1 at the SB root -> OUTPUT_ENABLED winner walk);
                // speed >= 8 runs the KB-12 single-pass `av1_nonrd_use_
                // partition` nonrd walk. The 16x16 min/max-sub-var split prune
                // (`vbp_prune_16x16_split_using_min_max_sub_blk_var`,
                // speed_features.c:600) is a speed-9 sf — inert <720p but
                // threaded faithfully.
                crate::var_part::choose_var_based_partitioning_key(
                    &mut vbp_stamps,
                    vf,
                    env.src_y,
                    env.base_y,
                    env.stride,
                    mi_row,
                    mi_col,
                    /*vbp_prune_16x16_split_using_min_max_sub_blk_var=*/
                    pick_cfg.speed >= 9,
                );
                if pick_cfg.speed >= 8 {
                    crate::partition_pick::nonrd_use_partition_real(
                        &sb_env,
                        &sb_pick_cfg,
                        &mut search_tile,
                        &mut grid,
                        recon_y,
                        recon_u,
                        recon_v,
                        &mut cfl_search,
                        &vbp_stamps,
                        mi_row,
                        mi_col,
                        sb_size,
                        &mut visits,
                        &mut last_source_variance,
                    )
                } else {
                    let (tree, _stats) = crate::partition_pick::rd_use_partition_real(
                        &sb_env,
                        &sb_pick_cfg,
                        &mut search_tile,
                        &mut grid,
                        recon_y,
                        recon_u,
                        recon_v,
                        &mut cfl_search,
                        &vbp_stamps,
                        mi_row,
                        mi_col,
                        sb_size,
                        /*do_recon=*/ true,
                        &mut visits,
                        &mut last_source_variance,
                    );
                    tree
                }
            } else {
                let (tree, _stats, found) = rd_pick_partition_real(
                    &sb_env,
                    &sb_pick_cfg,
                    &mut search_tile,
                    &mut grid,
                    recon_y,
                    recon_u,
                    recon_v,
                    &mut cfl_search,
                    mi_row,
                    mi_col,
                    sb_size,
                    PartRdStats::invalid(),
                    0,
                    0, // quad_tree_idx: 0 at the SB (64×64) root
                    None,
                    None, // rect_part_win_info: NULL at the SB root (encodeframe.c:826)
                    &mut visits,
                    &mut last_source_variance,
                );
                assert!(
                    found,
                    "partition search must find a valid tree at ({mi_row}, {mi_col})"
                );
                tree.expect("found implies a winning tree")
            };

            // C `write_modes_sb` (bitstream.c:1625-1645): at the superblock
            // root — BEFORE the partition symbol — write every restoration
            // unit whose corner falls inside this SB, per plane, in
            // (plane, rrow, rcol) order. `lr_corners_in_sb` reproduces
            // `av1_loop_restoration_corners_in_sb` (the C gates bsize ==
            // sb_size internally; this loop runs only at SB roots). The LR
            // CDFs adapt inside the SAME tile context (`kf`) as every other
            // symbol.
            if let Some(lr) = lr {
                for plane in 0..lr.num_planes {
                    if lr.cfg.frame_restoration_type[plane] == LR_RESTORE_NONE {
                        continue;
                    }
                    if let Some((rc0, rc1, rr0, rr1)) = lr_corners_in_sb(
                        &lr.cfg, plane, env.ss_x, env.ss_y, mi_row, mi_col, sb_mi, sb_mi,
                    ) {
                        let (hu, _) = lr.cfg.plane_units(plane, env.ss_x, env.ss_y);
                        for rr in rr0..rr1 {
                            for rc in rc0..rc1 {
                                let runit_idx = (rc + rr * hu) as usize;
                                write_lr_unit(
                                    enc,
                                    &lr.units[plane][runit_idx],
                                    lr.cfg.frame_restoration_type[plane],
                                    plane,
                                    &mut lr_refs,
                                    &mut kf.switchable_restore,
                                    &mut kf.wiener_restore,
                                    &mut kf.sgrproj_restore,
                                    pack_cfg.allow_update_cdf,
                                );
                            }
                        }
                    }
                }
            }

            let mut cfl_pack = CflCtx::new(env.ss_x as i32, env.ss_y as i32);
            pack_sb(
                enc,
                &sb_env,
                pack_cfg,
                kf,
                &mut kfs,
                &mut pack_tile_ctx,
                &mut nbr,
                recon_y,
                recon_u,
                recon_v,
                &mut cfl_pack,
                &mut tree,
                mi_row,
                mi_col,
                sb_size,
                sb_current_qindex,
            );
            trees.push(tree);
        }
    }
    trees
}

/// Phase-2 pack walk for the TWO-PASS (post-filter-search) frame encode:
/// re-walk ALREADY-PICKED [`SbTree`]s — from a phase-1 [`pack_tile`] whose
/// bits were discarded — writing the real bitstream, optionally with the
/// per-unit CDEF strength literals interleaved.
///
/// This models libaom's actual architecture for filters whose parameters
/// are searched AFTER the frame encode but signalled INSIDE the tile data
/// (CDEF): C's encode pass (`encode_sb(.., OUTPUT_ENABLED, ..)`,
/// encodeframe.c) adapts one tile-context copy (seeded from `cm->fc`) and
/// produces the reconstruction; deblock + `av1_cdef_search` run on that
/// recon; `av1_pack_bitstream` then seeds a SECOND fresh tile context from
/// the same `cm->fc` and re-writes every symbol, now emitting `write_cdef`
/// literals. The port's phase 1 is a `pack_tile` call into a throwaway
/// entropy coder (its `kf` adaptation and recon ARE the encode pass); this
/// function is the pack pass. CDEF strength literals are raw
/// (CDF-free) writes, so this pass's `kf` adapts through the IDENTICAL
/// symbol stream phase 1 saw — the per-SB `derive_real_costs(kf)` states
/// match SB-for-SB, and the leaf re-encode reproduces phase 1's
/// coefficients byte-for-byte (the same both-walks-identical argument the
/// fused [`pack_tile`] already rests on, split across two calls).
///
/// `recon_*` MUST arrive in the same initial state phase 1 started from
/// (NOT phase 1's final or deblocked output): the walk re-encodes every
/// leaf, and intra prediction reads recon neighbours as of the walk
/// position. `kf` MUST be a fresh `default_for_qindex` context.
///
/// `cdef`: `Some` = write `cdef.cdef_bits`-wide strength literals per
/// 64x64 unit (the searched [`crate::pickcdef::CdefSearchResult`] mapped
/// into a [`CdefPackState`]); `None` = CDEF-off (byte-identical to the
/// fused [`pack_tile`] walk over the same trees).
#[allow(clippy::too_many_arguments)]
pub fn pack_tile_from_trees(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    pick_cfg: &PickFrameCfg,
    pack_cfg: &PackCfg,
    kf: &mut KfFrameContext,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    trees: &mut [SbTree],
    mi_row0: i32,
    mi_col0: i32,
    n_sb_rows: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
    cdef: Option<CdefPackState>,
) {
    assert_eq!(
        trees.len(),
        (n_sb_rows * n_sb_cols) as usize,
        "one picked tree per SB"
    );
    let mi_cols = env.mi_cols as usize;
    let mut pack_tile_ctx = TileCtxState::zeroed(mi_cols);
    let mut nbr = MiNbrGrid::zeroed(mi_cols);
    let mut kfs = kf_block_state(pack_cfg, env, sb_mi);
    if let Some(c) = cdef {
        kfs.cdef_bits = c.cdef_bits;
        nbr.cdef = Some(c);
    }

    // Variance-Boost delta-q running base (mirrors pack_tile's tracker): reset
    // to base at the tile start, advanced per SB by setup_delta_q_variance_boost
    // below. With delta-q OFF this stays the constant base_qindex and every
    // derivation below reduces to pack_tile's `deltaq: None` arm — byte-identical
    // to the pre-tune CDEF repack path.
    let mut search_base_qindex = env
        .deltaq
        .map(|d| d.base_qindex)
        .unwrap_or(pack_cfg.base_qindex);

    for r in 0..n_sb_rows {
        pack_tile_ctx.left_ectx = [[0; 32]; 3];
        pack_tile_ctx.left_pctx = [0; 32];
        pack_tile_ctx.left_tctx = [aom_dsp::entropy::partition::TXFM_CTX_INIT; 32];
        nbr.zero_left();
        for c in 0..n_sb_cols {
            let mi_row = mi_row0 + r * sb_mi;
            let mi_col = mi_col0 + c * sb_mi;

            // Per-SB delta-q (Variance Boost) derivation — mirrors pack_tile's
            // (same source-only inputs → identical per-SB qindex sequence as
            // phase 1). With delta-q OFF this is (base_qindex, None) and
            // `sb_base_rdmult == env.rdmult`, i.e. byte-identical to the prior
            // CDEF-repack behaviour.
            let (sb_current_qindex, dq_rows) = if let Some(dq) = &env.deltaq {
                let sb_off =
                    env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
                let adjusted = crate::allintra_vis::setup_delta_q_variance_boost(
                    env.src_y,
                    sb_off,
                    env.stride,
                    env.bd,
                    dq.base_qindex,
                    dq.deltaq_strength,
                    dq.delta_q_res,
                    search_base_qindex,
                );
                search_base_qindex = adjusted;
                let rows = (
                    aom_dsp::quant::set_q_index(dq.quants, dq.deq, adjusted as usize, 0),
                    aom_dsp::quant::set_q_index(dq.quants, dq.deq, adjusted as usize, 1),
                    aom_dsp::quant::set_q_index(dq.quants, dq.deq, adjusted as usize, 2),
                );
                (adjusted, Some(rows))
            } else {
                (pack_cfg.base_qindex, None)
            };
            let sb_base_rdmult = if env.deltaq.is_some() {
                crate::rd::av1_compute_rd_mult_based_on_qindex(
                    env.bd,
                    crate::rd::FrameUpdateType::Kf,
                    sb_current_qindex,
                    if env.tune.iq_tuning {
                        crate::rd::TuneMetric::Iq
                    } else {
                        crate::rd::TuneMetric::Psnr
                    },
                    if pick_cfg.allintra {
                        crate::rd::EncMode::Allintra
                    } else {
                        crate::rd::EncMode::Good
                    },
                )
            } else {
                env.rdmult
            };

            // Identical per-SB folds to pack_tile's (same inputs → same
            // values as phase 1 derived at this SB): the ALLINTRA rdmult
            // modifier and the INTERNAL_COST_UPD_SB cost refresh. The CDEF
            // repack runs speed-0 only (VAR_BASED_PARTITION is speed>=7), so
            // the ALLINTRA fold is unconditional here — folding onto the
            // delta-q-adjusted `sb_base_rdmult` (== env.rdmult when off).
            let sb_rdmult = if pick_cfg.allintra {
                let mi_w = MI_SIZE_WIDE_B[sb_size] as i32;
                let mi_h = MI_SIZE_HIGH_B[sb_size] as i32;
                let ref_off_y =
                    env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
                let mb_to_right_edge = (env.mi_cols - mi_w - mi_col) * 4 * 8;
                let mb_to_bottom_edge = (env.mi_rows - mi_h - mi_row) * 4 * 8;
                let (var_min, var_max) = crate::partition_pick::log_sub_block_var(
                    env.src_y,
                    ref_off_y,
                    env.stride,
                    sb_size,
                    mb_to_right_edge,
                    mb_to_bottom_edge,
                    env.bd,
                );
                let modifier = crate::partition_pick::intra_sb_rdmult_modifier(var_min, var_max);
                crate::partition_pick::fold_intra_sb_rdmult(sb_base_rdmult, modifier)
            } else {
                sb_base_rdmult
            };
            // KB-12: allintra speed >= 9 flips coeff/mode_cost_upd_level to
            // INTERNAL_COST_UPD_OFF for <4k (set_allintra_speed_feature_
            // framesize_dependent, speed_features.c:166-177 -- runs AFTER the
            // framesize-independent SBROW setting :593-594 and wins) -> the
            // per-SB cost refresh STOPS: every SB reads the FRAME-INIT tables
            // (pick_cfg/env), byte-visible on multi-SB (128 sq) frames.
            // HANDOFF(speed 9): 4k+ frames keep INTERNAL_COST_UPD_SBROW --
            // out of the canon envelope, unmodelled.
            let cost_upd_off = pick_cfg.allintra && pick_cfg.speed >= 9;
            let sb_real = (!cost_upd_off)
                .then(|| crate::real_costs::derive_real_costs(kf, pick_cfg.enable_filter_intra));
            // Variance-Boost delta-q per-SB quantizer rows (None = identity when
            // delta-q is off — env.rows_* — so speeds 0-9 stay byte-unchanged).
            let sb_env = if let Some(sb_real) = &sb_real {
                SbEncodeEnv {
                    rdmult: sb_rdmult,
                    coeff_costs_y: &sb_real.coeff_costs_y,
                    coeff_costs_uv: &sb_real.coeff_costs_uv,
                    tx_type_costs: &sb_real.tx_type_costs_y,
                    rows_y: dq_rows.as_ref().map(|r| &r.0).unwrap_or(env.rows_y),
                    rows_u: dq_rows.as_ref().map(|r| &r.1).unwrap_or(env.rows_u),
                    rows_v: dq_rows.as_ref().map(|r| &r.2).unwrap_or(env.rows_v),
                    ..*env
                }
            } else {
                SbEncodeEnv {
                    rdmult: sb_rdmult,
                    rows_y: dq_rows.as_ref().map(|r| &r.0).unwrap_or(env.rows_y),
                    rows_u: dq_rows.as_ref().map(|r| &r.1).unwrap_or(env.rows_u),
                    rows_v: dq_rows.as_ref().map(|r| &r.2).unwrap_or(env.rows_v),
                    ..*env
                }
            };

            let mut cfl_pack = CflCtx::new(env.ss_x as i32, env.ss_y as i32);
            let tree = &mut trees[(r * n_sb_cols + c) as usize];
            pack_sb(
                enc,
                &sb_env,
                pack_cfg,
                kf,
                &mut kfs,
                &mut pack_tile_ctx,
                &mut nbr,
                recon_y,
                recon_u,
                recon_v,
                &mut cfl_pack,
                tree,
                mi_row,
                mi_col,
                sb_size,
                sb_current_qindex,
            );
        }
    }
}

/// `pack_txb_tokens` (bitstream.c) LUMA descent: recurse the `inter_tx_size`
/// quadtree and write each leaf's coefficients, consuming `txbs` in the same
/// depth-first order [`crate::encode_sb::encode_b_intra_dry`]'s intrabc COEFF
/// arm produced them.
#[allow(clippy::too_many_arguments)]
fn pack_vartx_txb(
    enc: &mut OdEcEnc,
    kf: &mut KfFrameContext,
    cfg: &PackCfg,
    env: &SbEncodeEnv,
    winner: &LeafWinner,
    txbs: &[TxbEncode],
    cursor: &mut usize,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    max_blocks_wide: usize,
    max_blocks_high: usize,
) {
    if blk_row >= max_blocks_high || blk_col >= max_blocks_wide {
        return;
    }
    let idx = crate::var_tx::get_txb_size_index(winner.bsize, blk_row, blk_col);
    let plane_tx_size = winner.inter_tx_size[idx];
    if tx_size == plane_tx_size {
        if *cursor < txbs.len() {
            write_one_txb_inter(enc, kf, cfg, env, &txbs[*cursor], tx_size, 0);
            *cursor += 1;
        }
        return;
    }
    let sub_txs = crate::var_tx::SUB_TX_SIZE_MAP[tx_size];
    let bsw = crate::var_tx::TX_SIZE_WIDE_UNIT[sub_txs];
    let bsh = crate::var_tx::TX_SIZE_HIGH_UNIT[sub_txs];
    let row_end = crate::var_tx::TX_SIZE_HIGH_UNIT[tx_size].min(max_blocks_high - blk_row);
    let col_end = crate::var_tx::TX_SIZE_WIDE_UNIT[tx_size].min(max_blocks_wide - blk_col);
    let mut row = 0usize;
    while row < row_end {
        let mut col = 0usize;
        while col < col_end {
            pack_vartx_txb(
                enc,
                kf,
                cfg,
                env,
                winner,
                txbs,
                cursor,
                blk_row + row,
                blk_col + col,
                sub_txs,
                max_blocks_wide,
                max_blocks_high,
            );
            col += bsw;
        }
        row += bsh;
    }
}

/// [`write_one_txb`]'s INTER twin: the tx-type symbol rides the
/// `inter_ext_tx_cdf[eset][square_tx_size]` region (bitstream.c:832-836) and
/// `write_coeffs_txb_full` gets `is_inter = true` (which also selects the inter
/// ext-tx SET). Chroma writes no tx-type symbol at all.
#[allow(clippy::too_many_arguments)]
fn write_one_txb_inter(
    enc: &mut OdEcEnc,
    kf: &mut KfFrameContext,
    cfg: &PackCfg,
    env: &SbEncodeEnv,
    txb: &TxbEncode,
    tx_size: usize,
    plane: usize,
) {
    let plane_type = usize::from(plane > 0);
    let mut dummy = [0u16; 8];
    let ext_tx_cdf: &mut [u16] = if plane_type == 0 {
        let d = aom_dsp::txb::ext_tx_derive(
            tx_size,
            true, // is_inter
            env.reduced_tx_set_used,
            txb.tx_type,
            false,
            0,
            0,
        );
        if d.eset > 0 {
            &mut kf.inter_ext_tx[d.eset as usize][d.square as usize]
        } else {
            &mut dummy[..]
        }
    } else {
        &mut dummy[..]
    };
    write_coeffs_txb_full(
        enc,
        &mut kf.coeff,
        ext_tx_cdf,
        &txb.qcoeff,
        txb.eob as usize,
        tx_size,
        txb.tx_type,
        plane_type,
        txb.txb_skip_ctx,
        txb.dc_sign_ctx,
        cfg.allow_update_cdf,
        true, // is_inter
        env.reduced_tx_set_used,
        false,
        0,
        0,
        cfg.signal_gate,
    );
}
