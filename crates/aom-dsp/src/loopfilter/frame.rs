//! Deblocking frame-application layer — port of libaom v3.14.1
//! `av1/common/av1_loopfilter.c` (the `lpf_opt_level == 0` / non-`_opt` path
//! the single-threaded decoder takes: `loop_filter_rows` →
//! `av1_thread_loop_filter_rows` → `av1_filter_block_plane_vert/horz` →
//! `set_lpf_parameters` → the `aom_lpf_*` kernels in this crate) plus the
//! frame-init tables (`av1_loop_filter_init` + `av1_loop_filter_frame_init`).
//!
//! Semantics found in C (cited against `reference/libaom`):
//! - Ordering (`thread_common.c:467-491 loop_filter_rows`): per 32-mi
//!   (`MAX_MIB_SIZE`) superblock-row strip → per plane → per direction
//!   (ALL vertical edges of the strip first, then ALL horizontal) → per
//!   32-mi superblock column. This strip-interleaved order equals the spec's
//!   whole-frame vert-then-horz order because a filter's footprint never
//!   leaves the two transform units adjacent to its edge (tap extent < the
//!   min tx dim that selected the filter length), and tx units never cross
//!   the 128-px strip boundary.
//! - Per-edge derivation (`av1_loopfilter.c:224-328 set_lpf_parameters`):
//!   an edge is considered only at transform-unit boundaries
//!   (`coord & (tx dim - 1) == 0`, av1_loopfilter.c:260-265), never at the
//!   plane's left/top border (`coord != 0`, :273), and is filtered iff
//!   `(curr_level || pv_lvl) && (!pv_skip_txfm || !curr_skipped || pu_edge)`
//!   (:298-299) — where the skip terms are `skip_txfm && is_inter_block`
//!   (:271,:287-288; always false on an all-intra KEY frame) and `pu_edge`
//!   is a prediction-block boundary of the CURRENT block (:289-295).
//! - Filter length (:300-311): from `min(tx dim log2)` of the two sides —
//!   luma `{4,8,14,14,14}[dim]` (`tx_dim_to_filter_length`, :220), chroma
//!   `dim == 0 ? 4 : 6`.
//! - Level (:269-315): `get_filter_level` (:68-108) per side; the edge uses
//!   the current block's level, falling back to the previous block's when
//!   the current is 0 (:315). Level 0 on both sides ⇒ unfiltered.
//! - The walk advances by the CURRENT position's transform width/height
//!   (`av1_filter_block_plane_vert:1347-1349`), one 4-px line at a time in
//!   the other axis, so a 16-px-tall vertical edge is filtered as four
//!   4-px kernel calls with independently re-derived parameters.
//! - Chroma positions map to the bottom/right mi of the co-located luma
//!   block: `mi_row = ss_y | ((y << ss_y) >> MI_SIZE_LOG2)` (:246-247).
//! - `filter_vert`/`filter_horz` (:906-1110, :1508-1712, `USE_SINGLE` arm)
//!   dispatch on filter_length {4,6,8,14} to the `aom_lpf_*` /
//!   `aom_highbd_lpf_*` kernels with `lfthr[level]`'s mblim/lim/hev_thr.
//!
//! Pixels are `u16` at every bit depth here; bd == 8 runs the highbd kernels
//! with `bd = 8`, which is arithmetically identical to the C lowbd path
//! (`aom_dsp/loopfilter.c:104-134` vs `:602-635`: at shift = bd-8 = 0 the
//! clamps, thresholds and rounding coincide) — and the differential harness
//! proves it, since the C oracle runs the REAL lowbd path for bd-8 frames.
//!
//! Flattening contract: C walks a grid of `MB_MODE_INFO *` where one struct
//! serves a whole coding block. [`LfMi`] duplicates the fields per mi cell;
//! [`LfMi::tx_size`] must hold the PLANE-0 `get_transform_size` result for
//! that cell (`av1_loopfilter.c:196-218` pre-lossless): the block's
//! `tx_size` for intra or skipped-inter cells, the cell's `inter_tx_size`
//! entry for non-skip inter cells. All-intra KEY decoding stamps the block
//! `tx_size` everywhere, which is exact.

use crate::loopfilter::highbd;

pub const MAX_LOOP_FILTER: i32 = 63;
const MI_SIZE_LOG2: u32 = 2;
const MI_SIZE: usize = 4;
/// `MAX_MIB_SIZE` — the strip/SB-column step of the walk, independent of the
/// sequence superblock size (64 or 128).
const MAX_MIB_SIZE: usize = 32;
pub const MAX_SEGMENTS: usize = 8;
const REF_FRAMES: usize = 8;
const MAX_MODE_LF_DELTAS: usize = 2;
/// `FRAME_LF_COUNT`: delta-lf ids (Y vert, Y horz, U, V).
pub const FRAME_LF_COUNT: usize = 4;
const TX_4X4: usize = 0;
const BLOCK_INVALID: u8 = 255;

// ---- conversion tables (av1/common/common_data.h @ v3.14.1) --------------------

/// `tx_size_wide[TX_SIZES_ALL]` (common_data.h:234).
const TX_SIZE_WIDE: [u32; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
/// `tx_size_high[TX_SIZES_ALL]` (common_data.h:241).
const TX_SIZE_HIGH: [u32; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
/// `tx_size_wide_unit[TX_SIZES_ALL]` (common_data.h:246).
const TX_SIZE_WIDE_UNIT: [usize; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
/// `tx_size_high_unit[TX_SIZES_ALL]` (common_data.h:251).
const TX_SIZE_HIGH_UNIT: [usize; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];
/// `tx_size_wide_unit_log2[TX_SIZES_ALL]` (common_data.h:261).
const TX_SIZE_WIDE_UNIT_LOG2: [u32; 19] = [0, 1, 2, 3, 4, 0, 1, 1, 2, 2, 3, 3, 4, 0, 2, 1, 3, 2, 4];
/// `tx_size_high_unit_log2[TX_SIZES_ALL]` (common_data.h:271).
const TX_SIZE_HIGH_UNIT_LOG2: [u32; 19] = [0, 1, 2, 3, 4, 1, 0, 2, 1, 3, 2, 4, 3, 2, 0, 3, 1, 4, 2];
/// `block_size_wide[BLOCK_SIZES_ALL]` (common_data.h:46).
const BLOCK_SIZE_WIDE: [u32; 22] = [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
/// `block_size_high[BLOCK_SIZES_ALL]` (common_data.h:52).
const BLOCK_SIZE_HIGH: [u32; 22] = [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
/// `max_txsize_rect_lookup[BLOCK_SIZES_ALL]` (common_data.h:126).
const MAX_TXSIZE_RECT_LOOKUP: [u8; 22] = [
    0,  // 4x4    -> TX_4X4
    5, 6, 1, // 4x8 8x4 8x8
    7, 8, 2, // 8x16 16x8 16x16
    9, 10, 3, // 16x32 32x16 32x32
    11, 12, 4, // 32x64 64x32 64x64
    4, 4, 4, // 64x128 128x64 128x128 -> TX_64X64
    13, 14, // 4x16 16x4
    15, 16, // 8x32 32x8
    17, 18, // 16x64 64x16
];
/// `av1_ss_size_lookup[BLOCK_SIZES_ALL][ss_x][ss_y]` (common_data.c:17).
const SS_SIZE_LOOKUP: [[[u8; 2]; 2]; 22] = [
    [[0, 0], [0, 0]],                            // 4x4
    [[1, 0], [BLOCK_INVALID, 0]],                // 4x8
    [[2, BLOCK_INVALID], [0, 0]],                // 8x4
    [[3, 2], [1, 0]],                            // 8x8
    [[4, 3], [BLOCK_INVALID, 1]],                // 8x16
    [[5, BLOCK_INVALID], [3, 2]],                // 16x8
    [[6, 5], [4, 3]],                            // 16x16
    [[7, 6], [BLOCK_INVALID, 4]],                // 16x32
    [[8, BLOCK_INVALID], [6, 5]],                // 32x16
    [[9, 8], [7, 6]],                            // 32x32
    [[10, 9], [BLOCK_INVALID, 7]],               // 32x64
    [[11, BLOCK_INVALID], [9, 8]],               // 64x32
    [[12, 11], [10, 9]],                         // 64x64
    [[13, 12], [BLOCK_INVALID, 10]],             // 64x128
    [[14, BLOCK_INVALID], [12, 11]],             // 128x64
    [[15, 14], [13, 12]],                        // 128x128
    [[16, 1], [BLOCK_INVALID, 1]],               // 4x16
    [[17, BLOCK_INVALID], [2, 2]],               // 16x4
    [[18, 4], [BLOCK_INVALID, 16]],              // 8x32
    [[19, BLOCK_INVALID], [5, 17]],              // 32x8
    [[20, 7], [BLOCK_INVALID, 18]],              // 16x64
    [[21, BLOCK_INVALID], [8, 19]],              // 64x16
];
/// `mode_lf_lut[MB_MODE_COUNT]` (av1_loopfilter.c:41): maps a prediction mode
/// to the mode-delta index — 0 for all intra + GLOBALMV + GLOBAL_GLOBALMV,
/// 1 for the other inter modes.
pub const MODE_LF_LUT: [u8; 25] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 13 intra modes
    1, 1, 0, 1, // NEARESTMV NEARMV GLOBALMV NEWMV
    1, 1, 1, 1, 1, 1, 0, 1, // compound (GLOBAL_GLOBALMV == 0)
];
/// `tx_dim_to_filter_length[TX_SIZES]` (av1_loopfilter.c:220).
const TX_DIM_TO_FILTER_LENGTH: [u8; 5] = [4, 8, 14, 14, 14];
/// `seg_lvl_lf_lut[MAX_MB_PLANE][2]` (av1_loopfilter.c:31), re-based to the
/// 0..4 LF-feature index (C SEG_LVL id − 1): Y_V, Y_H, U, V.
const SEG_LVL_LF_LUT: [[usize; 2]; 3] = [[0, 1], [2, 2], [3, 3]];
/// `delta_lf_id_lut[MAX_MB_PLANE][2]` (av1_loopfilter.c:37).
const DELTA_LF_ID_LUT: [[usize; 2]; 3] = [[0, 1], [2, 2], [3, 3]];

/// `av1_get_adjusted_tx_size` (blockd.h:1366): 64-dim transforms clamp to 32.
fn adjusted_tx_size(tx: usize) -> usize {
    match tx {
        4 | 12 | 11 => 3, // TX_64X64 / TX_64X32 / TX_32X64 -> TX_32X32
        18 => 10,         // TX_64X16 -> TX_32X16
        17 => 9,          // TX_16X64 -> TX_16X32
        _ => tx,
    }
}

/// `get_plane_block_size` (blockd.h): `av1_ss_size_lookup[bsize][ss_x][ss_y]`.
fn get_plane_block_size(bsize: usize, ss_x: usize, ss_y: usize) -> u8 {
    SS_SIZE_LOOKUP[bsize][ss_x][ss_y]
}

/// `av1_get_max_uv_txsize` (blockd.h:1377).
fn max_uv_txsize(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
    debug_assert_ne!(plane_bsize, BLOCK_INVALID, "invalid chroma block size");
    adjusted_tx_size(MAX_TXSIZE_RECT_LOOKUP[plane_bsize as usize] as usize)
}

// ---- inputs ---------------------------------------------------------------------

/// Per-mi-cell mode-info facts the filter reads (the flattened `MB_MODE_INFO`
/// projection — see the module doc for the [`LfMi::tx_size`] contract).
#[derive(Clone, Copy, Debug, Default)]
pub struct LfMi {
    /// `mbmi->bsize`.
    pub bsize: u8,
    /// Plane-0 `get_transform_size` result at this cell (pre-lossless).
    pub tx_size: u8,
    /// `mbmi->segment_id`.
    pub segment_id: u8,
    /// `mbmi->ref_frame[0]` (0 = INTRA_FRAME).
    pub ref0: u8,
    /// `mode_lf_lut[mbmi->mode]` (see [`MODE_LF_LUT`]).
    pub mode_lf: u8,
    /// `is_inter_block(mbmi)` — intrabc or `ref_frame[0] > INTRA_FRAME`.
    pub is_inter: bool,
    /// `mbmi->skip_txfm`.
    pub skip_txfm: bool,
    /// `mbmi->delta_lf_from_base` (single-delta mode).
    pub delta_lf_from_base: i8,
    /// `mbmi->delta_lf[FRAME_LF_COUNT]` (multi-delta mode).
    pub delta_lf: [i8; FRAME_LF_COUNT],
}

/// The frame's mi grid of [`LfMi`] cells: `mi[row * stride + col]`, populated
/// for all `mi_rows x mi_cols` (a fully-coded frame — the C `mbmi == NULL`
/// uncoded-tile arm is not modeled).
pub struct LfMiGrid<'a> {
    pub mi: &'a [LfMi],
    pub stride: usize,
    pub mi_rows: i32,
    pub mi_cols: i32,
}

/// Segmentation LF features (`seg_common.h` projected to the 4 LF features):
/// index 0..4 = SEG_LVL_ALT_LF_Y_V, _Y_H, _U, _V (C ids 1..=4).
#[derive(Clone, Copy, Debug, Default)]
pub struct LfSeg {
    pub enabled: bool,
    pub active: [[bool; 4]; MAX_SEGMENTS],
    pub data: [[i32; 4]; MAX_SEGMENTS],
}

impl LfSeg {
    /// `segfeature_active` for an LF feature (0..4).
    fn is_active(&self, segment_id: usize, lf_feature: usize) -> bool {
        self.enabled && self.active[segment_id][lf_feature]
    }
}

/// The frame's loop-filter parameters (`struct loopfilter` +
/// `delta_q_info` flags + `xd->lossless`).
#[derive(Clone, Copy, Debug)]
pub struct LfParams {
    /// `lf.filter_level[2]` (luma vert / horz).
    pub filter_level: [i32; 2],
    pub filter_level_u: i32,
    pub filter_level_v: i32,
    /// `lf.sharpness_level` (0..=7).
    pub sharpness: i32,
    pub mode_ref_delta_enabled: bool,
    /// `lf.ref_deltas[REF_FRAMES]` ([0] = INTRA_FRAME).
    pub ref_deltas: [i8; REF_FRAMES],
    /// `lf.mode_deltas[MAX_MODE_LF_DELTAS]`.
    pub mode_deltas: [i8; MAX_MODE_LF_DELTAS],
    /// `delta_q_info.delta_lf_present_flag` / `delta_lf_multi`.
    pub delta_lf_present: bool,
    pub delta_lf_multi: bool,
    /// `xd->lossless[MAX_SEGMENTS]`.
    pub lossless: [bool; MAX_SEGMENTS],
    pub seg: LfSeg,
}

impl Default for LfParams {
    fn default() -> Self {
        Self {
            filter_level: [0, 0],
            filter_level_u: 0,
            filter_level_v: 0,
            sharpness: 0,
            mode_ref_delta_enabled: false,
            ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            mode_deltas: [0, 0],
            delta_lf_present: false,
            delta_lf_multi: false,
            lossless: [false; MAX_SEGMENTS],
            seg: LfSeg::default(),
        }
    }
}

// ---- frame-init tables (av1_loop_filter_init + av1_loop_filter_frame_init) ------

/// `loop_filter_info_n`: per-level thresholds (`(mblim, lim, hev_thr)`, one
/// lane of the C 16-byte vectors) + the precomputed level table
/// `lvl[plane][segment][dir][ref][mode]`.
pub struct LfInfo {
    pub lfthr: [(u8, u8, u8); (MAX_LOOP_FILTER + 1) as usize],
    pub lvl: [[[[[u8; MAX_MODE_LF_DELTAS]; REF_FRAMES]; 2]; MAX_SEGMENTS]; 3],
}

fn clamp_lf(v: i32) -> i32 {
    v.clamp(0, MAX_LOOP_FILTER)
}

/// `av1_loop_filter_init` (hev_thr = lvl >> 4) + `av1_loop_filter_frame_init`
/// (av1_loopfilter.c:110-194): sharpness limits + the seg/ref/mode-resolved
/// level table. Planes whose frame level is zero keep zeroed `lvl` rows
/// (plane 0 zero-level BREAKS out of the plane loop like the C).
pub fn lf_frame_init(p: &LfParams, plane_start: usize, plane_end: usize) -> LfInfo {
    let mut lfi = LfInfo {
        lfthr: [(0, 0, 0); (MAX_LOOP_FILTER + 1) as usize],
        lvl: [[[[[0; MAX_MODE_LF_DELTAS]; REF_FRAMES]; 2]; MAX_SEGMENTS]; 3],
    };
    // update_sharpness (av1_loopfilter.c:47-66) + the hev_thr init (:120-121).
    for lvl in 0..=MAX_LOOP_FILTER {
        let mut block_inside_limit =
            lvl >> ((p.sharpness > 0) as i32 + (p.sharpness > 4) as i32);
        if p.sharpness > 0 && block_inside_limit > 9 - p.sharpness {
            block_inside_limit = 9 - p.sharpness;
        }
        if block_inside_limit < 1 {
            block_inside_limit = 1;
        }
        lfi.lfthr[lvl as usize] = (
            (2 * (lvl + 2) + block_inside_limit) as u8,
            block_inside_limit as u8,
            (lvl >> 4) as u8,
        );
    }

    let filt_lvl = [p.filter_level[0], p.filter_level_u, p.filter_level_v];
    let filt_lvl_r = [p.filter_level[1], p.filter_level_u, p.filter_level_v];
    for plane in plane_start..plane_end.min(3) {
        if plane == 0 && filt_lvl[0] == 0 && filt_lvl_r[0] == 0 {
            break;
        }
        if (plane == 1 && filt_lvl[1] == 0) || (plane == 2 && filt_lvl[2] == 0) {
            continue;
        }
        for seg_id in 0..MAX_SEGMENTS {
            #[allow(clippy::needless_range_loop)]
            for dir in 0..2 {
                let mut lvl_seg = if dir == 0 { filt_lvl[plane] } else { filt_lvl_r[plane] };
                let f = SEG_LVL_LF_LUT[plane][dir];
                if p.seg.is_active(seg_id, f) {
                    lvl_seg = clamp_lf(lvl_seg + p.seg.data[seg_id][f]);
                }
                if !p.mode_ref_delta_enabled {
                    for r in 0..REF_FRAMES {
                        for m in 0..MAX_MODE_LF_DELTAS {
                            lfi.lvl[plane][seg_id][dir][r][m] = lvl_seg as u8;
                        }
                    }
                } else {
                    let scale = 1 << (lvl_seg >> 5);
                    lfi.lvl[plane][seg_id][dir][0][0] =
                        clamp_lf(lvl_seg + p.ref_deltas[0] as i32 * scale) as u8;
                    for r in 1..REF_FRAMES {
                        for m in 0..MAX_MODE_LF_DELTAS {
                            let inter_lvl = lvl_seg
                                + p.ref_deltas[r] as i32 * scale
                                + p.mode_deltas[m] as i32 * scale;
                            lfi.lvl[plane][seg_id][dir][r][m] = clamp_lf(inter_lvl) as u8;
                        }
                    }
                }
            }
        }
    }
    lfi
}

// ---- per-edge derivation ---------------------------------------------------------

/// `get_filter_level` (av1_loopfilter.c:68-108).
fn get_filter_level(p: &LfParams, lfi: &LfInfo, dir_idx: usize, plane: usize, mi: &LfMi) -> u8 {
    let segment_id = mi.segment_id as usize;
    if p.delta_lf_present {
        let delta_lf = if p.delta_lf_multi {
            mi.delta_lf[DELTA_LF_ID_LUT[plane][dir_idx]] as i32
        } else {
            mi.delta_lf_from_base as i32
        };
        let base_level = match plane {
            0 => p.filter_level[dir_idx],
            1 => p.filter_level_u,
            _ => p.filter_level_v,
        };
        let mut lvl_seg = clamp_lf(delta_lf + base_level);
        let f = SEG_LVL_LF_LUT[plane][dir_idx];
        if p.seg.is_active(segment_id, f) {
            lvl_seg = clamp_lf(lvl_seg + p.seg.data[segment_id][f]);
        }
        if p.mode_ref_delta_enabled {
            let scale = 1 << (lvl_seg >> 5);
            lvl_seg += p.ref_deltas[mi.ref0 as usize] as i32 * scale;
            if mi.ref0 > 0 {
                lvl_seg += p.mode_deltas[mi.mode_lf as usize] as i32 * scale;
            }
            lvl_seg = clamp_lf(lvl_seg);
        }
        lvl_seg as u8
    } else {
        lfi.lvl[plane][segment_id][dir_idx][mi.ref0 as usize][mi.mode_lf as usize]
    }
}

/// `get_transform_size` (av1_loopfilter.c:196-218), flattened per the
/// [`LfMi::tx_size`] contract.
fn get_transform_size(p: &LfParams, mi: &LfMi, plane: usize, ss_x: usize, ss_y: usize) -> usize {
    if p.lossless[mi.segment_id as usize] {
        return TX_4X4;
    }
    if plane == 0 {
        mi.tx_size as usize
    } else {
        max_uv_txsize(mi.bsize as usize, ss_x, ss_y)
    }
}

const VERT_EDGE: usize = 0;

/// `set_lpf_parameters` (av1_loopfilter.c:224-328) → `(tx_size, filter_length,
/// level)`; `filter_length == 0` means no filtering at this position. `x`/`y`
/// are plane-pixel coordinates; `plane_w`/`plane_h` the plane's CROP dims.
#[allow(clippy::too_many_arguments)]
fn set_lpf_parameters(
    grid: &LfMiGrid,
    p: &LfParams,
    lfi: &LfInfo,
    edge_dir: usize,
    x: u32,
    y: u32,
    plane: usize,
    ss_x: usize,
    ss_y: usize,
    plane_w: u32,
    plane_h: u32,
) -> (usize, u8, u8) {
    if plane_w <= x || plane_h <= y {
        return (TX_4X4, 0, 0);
    }

    let mi_row = (ss_y as u32 | ((y << ss_y) >> MI_SIZE_LOG2)) as usize;
    let mi_col = (ss_x as u32 | ((x << ss_x) >> MI_SIZE_LOG2)) as usize;
    let idx = mi_row * grid.stride + mi_col;
    let mi = &grid.mi[idx];
    let ts = get_transform_size(p, mi, plane, ss_x, ss_y);

    let coord = if edge_dir == VERT_EDGE { x } else { y };
    let transform_masks = if edge_dir == VERT_EDGE {
        TX_SIZE_WIDE[ts] - 1
    } else {
        TX_SIZE_HIGH[ts] - 1
    };
    if coord & transform_masks != 0 {
        return (ts, 0, 0);
    }

    let curr_level = get_filter_level(p, lfi, edge_dir, plane, mi);
    let curr_skipped = mi.skip_txfm && mi.is_inter;
    let mut level = curr_level;
    let mut filter_length = 0u8;
    if coord != 0 {
        let prev_idx = if edge_dir == VERT_EDGE {
            idx - (1 << ss_x)
        } else {
            idx - (grid.stride << ss_y)
        };
        let mi_prev = &grid.mi[prev_idx];
        let pv_ts = get_transform_size(p, mi_prev, plane, ss_x, ss_y);
        let pv_lvl = get_filter_level(p, lfi, edge_dir, plane, mi_prev);
        let pv_skip_txfm = mi_prev.skip_txfm && mi_prev.is_inter;
        let bsize = get_plane_block_size(mi.bsize as usize, ss_x, ss_y);
        debug_assert_ne!(bsize, BLOCK_INVALID);
        let prediction_masks = if edge_dir == VERT_EDGE {
            BLOCK_SIZE_WIDE[bsize as usize] - 1
        } else {
            BLOCK_SIZE_HIGH[bsize as usize] - 1
        };
        let pu_edge = (coord & prediction_masks) == 0;
        // If both blocks are inter-skipped, only a prediction-block edge is
        // deblocked.
        if (curr_level != 0 || pv_lvl != 0) && (!pv_skip_txfm || !curr_skipped || pu_edge) {
            let dim = if edge_dir == VERT_EDGE {
                TX_SIZE_WIDE_UNIT_LOG2[ts].min(TX_SIZE_WIDE_UNIT_LOG2[pv_ts])
            } else {
                TX_SIZE_HIGH_UNIT_LOG2[ts].min(TX_SIZE_HIGH_UNIT_LOG2[pv_ts])
            } as usize;
            filter_length = if plane != 0 {
                if dim == 0 { 4 } else { 6 }
            } else {
                TX_DIM_TO_FILTER_LENGTH[dim]
            };
            // Use the not-skipped side's level when the current block is
            // skipped but the previous is not.
            level = if curr_level != 0 { curr_level } else { pv_lvl };
        }
    }
    (ts, filter_length, level)
}

// ---- the plane walks --------------------------------------------------------------

fn round_pot(v: i32, n: usize) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

/// `av1_filter_block_plane_vert` / `_horz` (av1_loopfilter.c:1305-1352,
/// :1907-1953) for one 32x32-mi superblock region of one plane.
#[allow(clippy::too_many_arguments)]
fn filter_block_plane(
    dir: usize,
    buf: &mut [u16],
    stride: usize,
    bd: i32,
    grid: &LfMiGrid,
    p: &LfParams,
    lfi: &LfInfo,
    plane: usize,
    ss_x: usize,
    ss_y: usize,
    plane_w: u32,
    plane_h: u32,
    mi_row: usize,
    mi_col: usize,
) {
    let plane_mi_rows = round_pot(grid.mi_rows, ss_y);
    let plane_mi_cols = round_pot(grid.mi_cols, ss_x);
    let y_range = (plane_mi_rows - ((mi_row >> ss_y) as i32)).min((MAX_MIB_SIZE >> ss_y) as i32);
    let x_range = (plane_mi_cols - ((mi_col >> ss_x) as i32)).min((MAX_MIB_SIZE >> ss_x) as i32);
    let (y_range, x_range) = (y_range as usize, x_range as usize);
    // Plane-pixel origin of this superblock region.
    let x0 = (mi_col * MI_SIZE) >> ss_x;
    let y0 = (mi_row * MI_SIZE) >> ss_y;
    let origin = y0 * stride + x0;

    if dir == 0 {
        for y in 0..y_range {
            let mut x = 0usize;
            while x < x_range {
                let curr_x = (x0 + x * MI_SIZE) as u32;
                let curr_y = (y0 + y * MI_SIZE) as u32;
                let (ts, len, level) = set_lpf_parameters(
                    grid, p, lfi, VERT_EDGE, curr_x, curr_y, plane, ss_x, ss_y, plane_w, plane_h,
                );
                if len > 0 {
                    let (mblim, lim, hev) = lfi.lfthr[level as usize];
                    let center = origin + y * MI_SIZE * stride + x * MI_SIZE;
                    highbd::vertical(len as u32, buf, center, stride, mblim, lim, hev, bd);
                }
                x += TX_SIZE_WIDE_UNIT[ts];
            }
        }
    } else {
        for x in 0..x_range {
            let mut y = 0usize;
            while y < y_range {
                let curr_x = (x0 + x * MI_SIZE) as u32;
                let curr_y = (y0 + y * MI_SIZE) as u32;
                let (ts, len, level) = set_lpf_parameters(
                    grid, p, lfi, 1, curr_x, curr_y, plane, ss_x, ss_y, plane_w, plane_h,
                );
                if len > 0 {
                    let (mblim, lim, hev) = lfi.lfthr[level as usize];
                    let center = origin + y * MI_SIZE * stride + x * MI_SIZE;
                    highbd::horizontal(len as u32, buf, center, stride, mblim, lim, hev, bd);
                }
                y += TX_SIZE_HIGH_UNIT[ts];
            }
        }
    }
}

// ---- whole-frame entry -------------------------------------------------------------

/// The frame's reconstruction planes (u16 samples at every bit depth; strides
/// in samples). Buffers must cover the mi-aligned area (`mi_cols*4 x
/// mi_rows*4` luma samples, subsampled for chroma) — filter taps read up to 7
/// samples past a crop-interior edge, always within the transform units
/// adjacent to it.
pub struct LfFrameBuf<'a> {
    pub y: &'a mut [u16],
    pub y_stride: usize,
    pub u: &'a mut [u16],
    pub v: &'a mut [u16],
    pub uv_stride: usize,
    /// Luma CROP dims (the coded frame size; chroma dims derive by
    /// subsampling like the C `crop_widths[1]`).
    pub crop_width: u32,
    pub crop_height: u32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: i32,
}

/// `av1_loop_filter_frame_mt(..., partial_frame = 0, lpf_opt_level = 0)` with
/// one worker — i.e. `check_planes_to_loop_filter` (thread_common.h:324-344)
/// then `av1_loop_filter_frame_init` then `loop_filter_rows`
/// (thread_common.c:467-491): per 32-mi strip → plane → dir(vert, horz) →
/// 32-mi superblock column. The caller applies the decoder's outer gate
/// (`decodeframe.c:5408`: skip entirely unless `filter_level[0] ||
/// filter_level[1]`) — but `check_planes_to_loop_filter` re-enforces it
/// here, so an unconditional call is also faithful.
pub fn loop_filter_frame(
    buf: &mut LfFrameBuf,
    grid: &LfMiGrid,
    p: &LfParams,
    plane_start: usize,
    plane_end: usize,
) {
    // check_planes_to_loop_filter / set_planes_to_loop_filter.
    let planes_to_lf = [
        (p.filter_level[0] != 0 || p.filter_level[1] != 0) && plane_start == 0 && 0 < plane_end,
        p.filter_level_u != 0 && plane_start <= 1 && 1 < plane_end,
        p.filter_level_v != 0 && plane_start <= 2 && 2 < plane_end,
    ];
    // "If the luma plane is purposely not filtered, neither are the chroma
    // planes."
    if !planes_to_lf[0] && plane_start == 0 && 0 < plane_end {
        return;
    }
    if !planes_to_lf[0] && !planes_to_lf[1] && !planes_to_lf[2] {
        return;
    }

    let lfi = lf_frame_init(p, plane_start, plane_end);

    let uv_w = (buf.crop_width + buf.ss_x as u32) >> buf.ss_x;
    let uv_h = (buf.crop_height + buf.ss_y as u32) >> buf.ss_y;

    let mut mi_row = 0usize;
    while (mi_row as i32) < grid.mi_rows {
        #[allow(clippy::needless_range_loop)]
        for plane in 0..3 {
            if !planes_to_lf[plane] {
                continue;
            }
            for dir in 0..2 {
                let mut mi_col = 0usize;
                while (mi_col as i32) < grid.mi_cols {
                    let (pb, stride, ss_x, ss_y, w, h): (&mut [u16], usize, usize, usize, u32, u32) =
                        match plane {
                            0 => (buf.y, buf.y_stride, 0, 0, buf.crop_width, buf.crop_height),
                            1 => (buf.u, buf.uv_stride, buf.ss_x, buf.ss_y, uv_w, uv_h),
                            _ => (buf.v, buf.uv_stride, buf.ss_x, buf.ss_y, uv_w, uv_h),
                        };
                    filter_block_plane(
                        dir, pb, stride, buf.bd, grid, p, &lfi, plane, ss_x, ss_y, w, h, mi_row,
                        mi_col,
                    );
                    mi_col += MAX_MIB_SIZE;
                }
            }
        }
        mi_row += MAX_MIB_SIZE;
    }
}
