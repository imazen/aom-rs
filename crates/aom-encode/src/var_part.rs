//! `av1_choose_var_based_partitioning` (av1/encoder/var_based_part.c) — the
//! **KEY-frame (allintra) arm** of the variance-based partitioner that
//! `partition_search_type == VAR_BASED_PARTITION` (allintra speed >= 7)
//! switches the superblock encoder to (`encode_rd_sb`, encodeframe.c:876-895):
//! the partition tree is FIXED up front from source-variance thresholds (no
//! RD partition search), then `av1_rd_use_partition` walks the fixed tree
//! running the normal RD mode search per block
//! ([`crate::partition_pick::rd_use_partition_real`]).
//!
//! Scope (matching the port's Gate-2 envelope): **KEY frames only** —
//! `is_key_frame` is true, so the reference/motion machinery (setup_planes,
//! chroma_check color-sensitivity, y_sad, zeromv-skip, low-temp-var flags,
//! the 8x8-avg source-vs-ref leaf fill) is structurally unreachable and NOT
//! ported. What is ported, line-faithful:
//!
//! 1. **Thresholds** (`set_vbp_thresholds` KEY arm, var_based_part.c:654-673
//!    → `set_vbp_thresholds_key_frame` :535-560): `threshold_base = 120 *
//!    av1_ac_quant_QTX(qindex, 0, bit_depth)`; `[0]=[1]=base`; `<720p:
//!    [2]=base/3, [3]=base>>1` else `[2]=[3]=base>>2`; `[4]=base<<2`.
//!    `rt_sf.force_large_partition_blocks_intra` (which is what consumes the
//!    speed-7 `var_part_split_threshold_shift = 7`) is 0 on this path — it
//!    only rises at allintra speed>=8/720p+ (speed_features.c:327) and in RT
//!    (:1647) — so the shift-steps arm is dead and the speed-7 shift is
//!    byte-INERT on KEY frames (carried in [`crate::speed_features`] for
//!    provenance). The frame-level `av1_set_variance_partition_thresholds`
//!    copy + `threshold_minmax` are likewise dead here: the per-SB
//!    `set_vbp_thresholds` call fully overwrites all five local thresholds
//!    for key frames, and `threshold_minmax` feeds only the
//!    `compute_minmax_variance` arm which is hardcoded 0 (:1114).
//! 2. **The 4x4-downsampled variance tree fill**
//!    (`fill_variance_tree_leaves` KEY arm :1156-1167 → `fill_variance_4x4avg`
//!    :390-423): per 4x4 sub-block, `sum = aom_avg_4x4(src) - 128`,
//!    `sse = sum*sum` (dst is the implicit flat-128 "prediction");
//!    out-of-frame 4x4s (top-left at/past the crop) contribute zeros.
//!    `border_offset_4x4` stays 0 (temporal filtering never runs on key
//!    frames, :1135).
//! 3. **The force-split stage-2 walk** (:1788-1894, key arms): 16x16 nodes
//!    with `variance > thresholds[3]` force PARTITION_SPLIT up the tree
//!    (16→32→64→root); 32x32 nodes with `variance > thresholds[2]` force
//!    32→64→root. The 64x64/128x128 levels have no key-frame forcing rules;
//!    a 64x64 SB (`is_small_sb`) always forces the (structural) root split.
//! 4. **The partition assignment** (`set_vt_partitioning` :149-253 + the
//!    :1896-1942 descent): key frames take the split for `bsize >
//!    BLOCK_32X32` or `variance > (threshold << 4)`; NONE when the block
//!    fits and `variance < threshold`; otherwise the VERT/HORZ pair checks
//!    (both halves' variance under threshold + a valid chroma plane size);
//!    else descend, bottoming out at four BLOCK_8X8 leaves per 16x16. The
//!    result is written as `bsize` stamps at each leaf's top-left mi cell
//!    (`set_block_size` :136-147), the exact structure C's
//!    `get_partition` (av1_common_int.h:1775) reads back —
//!    [`get_partition_from_stamps`] here.
//!
//! Differential status: `av1_choose_var_based_partitioning` is not exported
//! from the reference build, so per the evidence hierarchy the tree logic is
//! validated transcription + the end-to-end byte gates (any partition
//! difference desyncs the bitstream immediately); the one arithmetic kernel,
//! [`avg_4x4`], is differentially locked against the REAL exported
//! `aom_avg_4x4_c` (`avg_4x4_diff.rs`).

use aom_entropy::partition::{get_partition_subsize, get_plane_block_size};
use aom_quant::av1_ac_quant_qtx;

use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B};

const BLOCK_8X8: usize = 3;
const BLOCK_16X16: usize = 6;
const BLOCK_32X32: usize = 9;
const BLOCK_64X64: usize = 12;
const BLOCK_128X128: usize = 15;
const BLOCK_INVALID: usize = 255;

/// `PART_EVAL_STATUS` (var_based_part.c:38-45).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PartEval {
    /// Evaluate all partition types.
    All,
    /// Force PARTITION_SPLIT.
    OnlySplit,
    /// Force PARTITION_NONE.
    OnlyNone,
}

/// `VPartVar` (encoder/block.h): running (sse, sum, log2_count) for one
/// variance-tree node + the derived `variance`.
#[derive(Clone, Copy, Default, Debug)]
struct VPartVar {
    sum_square_error: u32,
    sum_error: i32,
    log2_count: i32,
    variance: i32,
}

/// `VPVariance`: the none/horz/vert partition variances of one node.
#[derive(Clone, Copy, Default, Debug)]
struct VPVariance {
    none: VPartVar,
    horz: [VPartVar; 2],
    vert: [VPartVar; 2],
}

/// `VP8x8` (KEY usage): four 4x4 leaf records + the 8x8 sums. C's `VP4x4`
/// children carry a full `VPVariance` each, but on the key path only their
/// `.none` totals are ever written/read (`fill_variance_4x4avg` /
/// `tree_to_node(BLOCK_8X8)`), so the leaf is stored as a bare [`VPartVar`].
#[derive(Clone, Copy, Default, Debug)]
struct Vp8x8 {
    part_variances: VPVariance,
    split: [VPartVar; 4],
}

#[derive(Clone, Copy, Default, Debug)]
struct Vp16x16 {
    part_variances: VPVariance,
    split: [Vp8x8; 4],
}

#[derive(Clone, Copy, Default, Debug)]
struct Vp32x32 {
    part_variances: VPVariance,
    split: [Vp16x16; 4],
}

#[derive(Clone, Copy, Default, Debug)]
struct Vp64x64 {
    part_variances: VPVariance,
    split: [Vp32x32; 4],
}

/// `VP128x128` — the full per-SB tree (`vt->split` = `td->vt64x64`).
#[derive(Clone, Default, Debug)]
struct Vp128x128 {
    part_variances: VPVariance,
    split: [Vp64x64; 4],
}

/// `GET_BLK_IDX_X(idx, level)` (var_based_part.h:24).
#[inline]
fn blk_idx_x(idx: usize, level: usize) -> usize {
    (idx & 1) << level
}

/// `GET_BLK_IDX_Y(idx, level)` (var_based_part.h:25).
#[inline]
fn blk_idx_y(idx: usize, level: usize) -> usize {
    (idx >> 1) << level
}

/// `aom_avg_4x4_c` (aom_dsp/avg.c:32) / `aom_highbd_avg_4x4_c` (:74) —
/// identical arithmetic on this port's u16 pixel buffers (`(sum + 8) >> 4`;
/// a 4x4 of 12-bit samples sums to <= 65520, no overflow). Differentially
/// locked vs the REAL exported `aom_avg_4x4_c` in `avg_4x4_diff.rs`.
#[inline]
pub fn avg_4x4(src: &[u16], off: usize, stride: usize) -> i32 {
    let mut sum = 0u32;
    for r in 0..4 {
        let row = &src[off + r * stride..off + r * stride + 4];
        for &p in row {
            sum += u32::from(p);
        }
    }
    ((sum + 8) >> 4) as i32
}

/// `fill_variance` (var_based_part.c:103).
#[inline]
fn fill_variance(s2: u32, s: i32, c: i32, v: &mut VPartVar) {
    v.sum_square_error = s2;
    v.sum_error = s;
    v.log2_count = c;
}

/// `get_variance` (var_based_part.c:109) — the C expression verbatim,
/// including the u32 wrap of the `(sum*sum)>>log2` subtraction and the
/// int truncation of the `256 * (...) >> log2` scale.
#[inline]
fn get_variance(v: &mut VPartVar) {
    let sum_sq = (i64::from(v.sum_error) * i64::from(v.sum_error)) >> v.log2_count;
    let diff = v.sum_square_error.wrapping_sub(sum_sq as u32);
    // C: (int)(256 * diff >> log2_count) — `256 * (uint32)diff` is a u32
    // multiply in C (both operands int-promoted; diff is unsigned so the
    // arithmetic wraps mod 2^32), then >> log2_count, then (int) cast.
    v.variance = (256u32.wrapping_mul(diff) >> v.log2_count) as i32;
}

/// `sum_2_variances` (var_based_part.c:117).
#[inline]
fn sum_2_variances(a: &VPartVar, b: &VPartVar, r: &mut VPartVar) {
    debug_assert_eq!(a.log2_count, b.log2_count);
    fill_variance(
        a.sum_square_error + b.sum_square_error,
        a.sum_error + b.sum_error,
        a.log2_count + 1,
        r,
    );
}

/// `fill_variance_tree` (var_based_part.c:124) on one node: sums the four
/// children's `none` totals into the node's horz/vert/none partitions.
fn fill_variance_node(children: &[VPartVar; 4], pv: &mut VPVariance) {
    let mut horz0 = VPartVar::default();
    let mut horz1 = VPartVar::default();
    let mut vert0 = VPartVar::default();
    let mut vert1 = VPartVar::default();
    sum_2_variances(&children[0], &children[1], &mut horz0);
    sum_2_variances(&children[2], &children[3], &mut horz1);
    sum_2_variances(&children[0], &children[2], &mut vert0);
    sum_2_variances(&children[1], &children[3], &mut vert1);
    pv.horz = [horz0, horz1];
    pv.vert = [vert0, vert1];
    let (v0, v1) = (pv.vert[0], pv.vert[1]);
    sum_2_variances(&v0, &v1, &mut pv.none);
}

#[inline]
fn child_nones(pvs: [&VPVariance; 4]) -> [VPartVar; 4] {
    [pvs[0].none, pvs[1].none, pvs[2].none, pvs[3].none]
}

/// Frame/tile geometry + quantizer inputs for one SB's variance partitioning.
#[derive(Clone, Copy, Debug)]
pub struct VbpFrame {
    /// `cm->mi_params.mi_rows` / `mi_cols`.
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// `tile->mi_row_end` / `mi_col_end` (== mi_rows/mi_cols single-tile).
    pub tile_mi_row_end: i32,
    pub tile_mi_col_end: i32,
    /// `cm->width * cm->height` (the crop pixel count, for the resolution
    /// threshold arms).
    pub num_pixels: i64,
    /// `cm->seq_params->sb_size` (BLOCK_64X64 = 12 or BLOCK_128X128 = 15).
    pub sb_size: usize,
    /// The SB qindex (`base_qindex`; no delta-q/segments in this envelope —
    /// the :1683-1690 clamp chain resolves to `base_qindex`).
    pub qindex: i32,
    pub bit_depth: u8,
    /// Chroma subsampling as C's `xd->plane[AOM_PLANE_U]` carries it:
    /// monochrome frames get (1, 1) (`av1_setup_block_planes` sets planes
    /// >= num_planes to ss (1,1)).
    pub ss_x: usize,
    pub ss_y: usize,
}

/// `set_vbp_thresholds` (var_based_part.c:654) — KEY-frame arm only
/// (`set_vbp_thresholds_key_frame` :535, `force_large_partition_blocks_intra
/// == 0` — module docs). Returns `thresholds[5]`.
pub fn set_vbp_thresholds_key(qindex: i32, bit_depth: u8, num_pixels: i64) -> [i64; 5] {
    const RESOLUTION_720P: i64 = 1280 * 720;
    let ac_q = av1_ac_quant_qtx(qindex, 0, bit_depth);
    let threshold_base: i64 = 120i64 * i64::from(ac_q);
    let mut thresholds = [0i64; 5];
    thresholds[0] = threshold_base;
    thresholds[1] = threshold_base;
    if num_pixels < RESOLUTION_720P {
        thresholds[2] = threshold_base / 3;
        thresholds[3] = threshold_base >> 1;
    } else {
        // force_large_partition_blocks_intra == 0 => shift_val stays 2.
        thresholds[2] = threshold_base >> 2;
        thresholds[3] = threshold_base >> 2;
    }
    thresholds[4] = threshold_base << 2;
    thresholds
}

/// `set_block_size` (var_based_part.c:136): stamp `bsize` at the block's
/// top-left mi cell when it is inside the frame. `stamps` is the
/// frame-sized (mi_rows x mi_cols, row-major) bsize grid this SB's
/// assignment writes and [`get_partition_from_stamps`] reads.
#[inline]
fn set_block_size(stamps: &mut [u8], f: &VbpFrame, mi_row: i32, mi_col: i32, bsize: usize) {
    if f.mi_cols > mi_col && f.mi_rows > mi_row {
        stamps[(mi_row * f.mi_cols + mi_col) as usize] = bsize as u8;
    }
}

/// `get_plane_block_size` for the guard in `set_vt_partitioning` — the same
/// subsampled-lookup the partition search uses.
fn plane_block_size(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    get_plane_block_size(bsize, ss_x, ss_y)
}

/// One level's variance node view (`variance_node` / `tree_to_node`).
struct NodeView<'a> {
    part_variances: &'a mut VPVariance,
}

/// `set_vt_partitioning` (var_based_part.c:149) — KEY-frame arms live
/// (`frame_is_intra_only` true). Returns true when a partitioning was set
/// at this node (stop descending).
#[allow(clippy::too_many_arguments)]
fn set_vt_partitioning(
    stamps: &mut [u8],
    f: &VbpFrame,
    node: NodeView,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    threshold: i64,
    bsize_min: usize,
    force_split: PartEval,
) -> bool {
    let block_width = MI_SIZE_WIDE_B[bsize] as i32;
    let block_height = block_width; // square sizes only (C asserts this)
    let mut bs_width_check = block_width;
    let mut bs_height_check = block_height;
    let mut bs_width_vert_check = block_width >> 1;
    let mut bs_height_horiz_check = block_height >> 1;
    // "On the right and bottom boundary we only need to check if half the
    // bsize fits, because boundary is extended up to 64. So do this check
    // only for sb_size = 64X64." (:162-174)
    if f.sb_size == BLOCK_64X64 {
        if f.tile_mi_col_end == f.mi_cols {
            bs_width_check = (block_width >> 1) + 1;
            bs_width_vert_check = (block_width >> 2) + 1;
        }
        if f.tile_mi_row_end == f.mi_rows {
            bs_height_check = (block_height >> 1) + 1;
            bs_height_horiz_check = (block_height >> 2) + 1;
        }
    }

    if mi_col + bs_width_check <= f.tile_mi_col_end
        && mi_row + bs_height_check <= f.tile_mi_row_end
        && force_split == PartEval::OnlyNone
    {
        set_block_size(stamps, f, mi_row, mi_col, bsize);
        return true;
    }
    if force_split == PartEval::OnlySplit {
        return false;
    }

    if bsize == bsize_min {
        // (Structurally dead in this file's call graph — every call site
        // passes bsize > bsize_min — kept transcription-faithful.)
        get_variance(&mut node.part_variances.none);
        if mi_col + bs_width_check <= f.tile_mi_col_end
            && mi_row + bs_height_check <= f.tile_mi_row_end
            && i64::from(node.part_variances.none.variance) < threshold
        {
            set_block_size(stamps, f, mi_row, mi_col, bsize);
            return true;
        }
        false
    } else if bsize > bsize_min {
        // Variance already computed to set the force_split (key frames
        // recompute here, :202).
        get_variance(&mut node.part_variances.none);
        // For key frame: take split for bsize above 32X32 or very high
        // variance (:204-208).
        if bsize > BLOCK_32X32 || i64::from(node.part_variances.none.variance) > (threshold << 4) {
            return false;
        }
        // If variance is low, take the bsize (no split).
        if mi_col + bs_width_check <= f.tile_mi_col_end
            && mi_row + bs_height_check <= f.tile_mi_row_end
            && i64::from(node.part_variances.none.variance) < threshold
        {
            set_block_size(stamps, f, mi_row, mi_col, bsize);
            return true;
        }
        // Check vertical split (:217-232).
        if mi_row + bs_height_check <= f.tile_mi_row_end
            && mi_col + bs_width_vert_check <= f.tile_mi_col_end
        {
            let subsize = get_partition_subsize(bsize, 2) as usize;
            let pbs = plane_block_size(subsize, f.ss_x, f.ss_y);
            get_variance(&mut node.part_variances.vert[0]);
            get_variance(&mut node.part_variances.vert[1]);
            if i64::from(node.part_variances.vert[0].variance) < threshold
                && i64::from(node.part_variances.vert[1].variance) < threshold
                && pbs < BLOCK_INVALID
            {
                set_block_size(stamps, f, mi_row, mi_col, subsize);
                set_block_size(stamps, f, mi_row, mi_col + block_width / 2, subsize);
                return true;
            }
        }
        // Check horizontal split (:234-249).
        if mi_col + bs_width_check <= f.tile_mi_col_end
            && mi_row + bs_height_horiz_check <= f.tile_mi_row_end
        {
            let subsize = get_partition_subsize(bsize, 1) as usize;
            let pbs = plane_block_size(subsize, f.ss_x, f.ss_y);
            get_variance(&mut node.part_variances.horz[0]);
            get_variance(&mut node.part_variances.horz[1]);
            if i64::from(node.part_variances.horz[0].variance) < threshold
                && i64::from(node.part_variances.horz[1].variance) < threshold
                && pbs < BLOCK_INVALID
            {
                set_block_size(stamps, f, mi_row, mi_col, subsize);
                set_block_size(stamps, f, mi_row + block_height / 2, mi_col, subsize);
                return true;
            }
        }
        false
    } else {
        false
    }
}

/// `av1_choose_var_based_partitioning` (var_based_part.c:1601) — KEY arm.
/// Fixes the partition for the SB at `(mi_row, mi_col)` as `bsize` stamps at
/// leaf top-left mi cells in `stamps` (a `mi_rows * mi_cols` row-major grid).
///
/// `src_y`/`base_y`/`stride` — the frame source plane
/// ([`crate::encode_sb::SbEncodeEnv`] conventions); the SB's pixels start at
/// `base_y + (mi_row*4)*stride + mi_col*4` (C's `x->plane[0].src.buf` after
/// `av1_set_offsets`).
///
/// `vbp_prune_16x16_split_using_min_max_sub_blk_var` = the rt speed feature
/// (:1806-1809; allintra stays false through speed 8 — the speed-9 flip is
/// documented for KB-12).
#[allow(clippy::too_many_arguments)]
pub fn choose_var_based_partitioning_key(
    stamps: &mut [u8],
    f: &VbpFrame,
    src_y: &[u16],
    base_y: usize,
    stride: usize,
    mi_row: i32,
    mi_col: i32,
    vbp_prune_16x16_split_using_min_max_sub_blk_var: bool,
) {
    debug_assert!(f.sb_size == BLOCK_64X64 || f.sb_size == BLOCK_128X128);
    let is_small_sb = f.sb_size == BLOCK_64X64;
    let num_64x64_blocks = if is_small_sb { 1usize } else { 4 };

    let thresholds = set_vbp_thresholds_key(f.qindex, f.bit_depth, f.num_pixels);

    // force_split[85]: 0 root, 1-4 the 64x64s, 5-20 the 32x32s, 21-84 the
    // 16x16s (:1610/:1699).
    let mut force_split = [PartEval::All; 85];

    let mut vt = Box::new(Vp128x128::default());

    // ---- fill_variance_tree_leaves (:1105), KEY arm: 4x4-downsampled
    //      leaf fill; pixels_wide/high from the SB's frame overhang
    //      (xd->mb_to_right_edge / mb_to_bottom_edge, :1125-1126). ----
    let sb_px = if is_small_sb { 64i32 } else { 128 };
    let pixels_wide = sb_px.min((f.mi_cols - mi_col) * 4);
    let pixels_high = sb_px.min((f.mi_rows - mi_row) * 4);
    let sb_off = base_y + (mi_row as usize * 4) * stride + mi_col as usize * 4;

    for blk64_idx in 0..num_64x64_blocks {
        let x64_idx = blk_idx_x(blk64_idx, 6);
        let y64_idx = blk_idx_y(blk64_idx, 6);
        let blk64_scale_idx = blk64_idx << 2;
        force_split[blk64_idx + 1] = PartEval::All;
        for lvl1_idx in 0..4usize {
            let x32_idx = x64_idx + blk_idx_x(lvl1_idx, 5);
            let y32_idx = y64_idx + blk_idx_y(lvl1_idx, 5);
            let lvl1_scale_idx = (blk64_scale_idx + lvl1_idx) << 2;
            force_split[5 + blk64_scale_idx + lvl1_idx] = PartEval::All;
            for lvl2_idx in 0..4usize {
                let x16_idx = x32_idx + blk_idx_x(lvl2_idx, 4);
                let y16_idx = y32_idx + blk_idx_y(lvl2_idx, 4);
                let split_index = 21 + lvl1_scale_idx + lvl2_idx;
                force_split[split_index] = PartEval::All;
                let vst = &mut vt.split[blk64_idx].split[lvl1_idx].split[lvl2_idx];
                // Go down to 4x4 down-sampling for variance (:1156-1167).
                for lvl3_idx in 0..4usize {
                    let x8_idx = x16_idx + blk_idx_x(lvl3_idx, 3);
                    let y8_idx = y16_idx + blk_idx_y(lvl3_idx, 3);
                    let vst2 = &mut vst.split[lvl3_idx];
                    // fill_variance_4x4avg (:390): border_offset_4x4 == 0 on
                    // key frames.
                    for idx in 0..4usize {
                        let x4_idx = x8_idx + blk_idx_x(idx, 2);
                        let y4_idx = y8_idx + blk_idx_y(idx, 2);
                        let mut sse = 0u32;
                        let mut sum = 0i32;
                        if (x4_idx as i32) < pixels_wide && (y4_idx as i32) < pixels_high {
                            let src_avg = avg_4x4(src_y, sb_off + y4_idx * stride + x4_idx, stride);
                            let dst_avg = 128;
                            sum = src_avg - dst_avg;
                            sse = (sum * sum) as u32;
                        }
                        fill_variance(sse, sum, 0, &mut vst2.split[idx]);
                    }
                }
            }
        }
    }

    // ---- the stage-2 force-split walk (:1788-1881), key arms ----
    for blk64_idx in 0..num_64x64_blocks {
        let blk64_scale_idx = blk64_idx << 2;
        for lvl1_idx in 0..4usize {
            let lvl1_scale_idx = (blk64_scale_idx + lvl1_idx) << 2;
            for lvl2_idx in 0..4usize {
                // (key frames only reach this body, :1796.)
                let vtemp = &mut vt.split[blk64_idx].split[lvl1_idx].split[lvl2_idx];
                for lvl3_idx in 0..4usize {
                    let sp = vtemp.split[lvl3_idx].split;
                    fill_variance_node(&sp, &mut vtemp.split[lvl3_idx].part_variances);
                }
                let nones = child_nones([
                    &vtemp.split[0].part_variances,
                    &vtemp.split[1].part_variances,
                    &vtemp.split[2].part_variances,
                    &vtemp.split[3].part_variances,
                ]);
                fill_variance_node(&nones, &mut vtemp.part_variances);
                // If variance of this 16x16 block is above the threshold,
                // force block to split (:1801-1813).
                get_variance(&mut vtemp.part_variances.none);
                if i64::from(vtemp.part_variances.none.variance) > thresholds[3] {
                    let split_index = 21 + lvl1_scale_idx + lvl2_idx;
                    force_split[split_index] = if vbp_prune_16x16_split_using_min_max_sub_blk_var {
                        // get_part_eval_based_on_sub_blk_var (:1530).
                        let mut max_8x8 = 0i32;
                        let mut min_8x8 = i32::MAX;
                        for sp in &mut vtemp.split {
                            get_variance(&mut sp.part_variances.none);
                            max_8x8 = max_8x8.max(sp.part_variances.none.variance);
                            min_8x8 = min_8x8.min(sp.part_variances.none.variance);
                        }
                        if i64::from(max_8x8 - min_8x8) > (thresholds[3] << 2) {
                            PartEval::OnlySplit
                        } else {
                            PartEval::OnlyNone
                        }
                    } else {
                        PartEval::OnlySplit
                    };
                    force_split[5 + blk64_scale_idx + lvl1_idx] = PartEval::OnlySplit;
                    force_split[blk64_idx + 1] = PartEval::OnlySplit;
                    force_split[0] = PartEval::OnlySplit;
                }
            }
            {
                let v32 = &mut vt.split[blk64_idx].split[lvl1_idx];
                let nones = child_nones([
                    &v32.split[0].part_variances,
                    &v32.split[1].part_variances,
                    &v32.split[2].part_variances,
                    &v32.split[3].part_variances,
                ]);
                fill_variance_node(&nones, &mut v32.part_variances);
                // 32x32 threshold check (:1825-1852; the !is_key_frame
                // second/third disjuncts are dead here).
                if force_split[5 + blk64_scale_idx + lvl1_idx] == PartEval::All {
                    get_variance(&mut v32.part_variances.none);
                    let var_32x32 = v32.part_variances.none.variance;
                    if i64::from(var_32x32) > thresholds[2] {
                        force_split[5 + blk64_scale_idx + lvl1_idx] = PartEval::OnlySplit;
                        force_split[blk64_idx + 1] = PartEval::OnlySplit;
                        force_split[0] = PartEval::OnlySplit;
                    }
                }
            }
        }
        if force_split[1 + blk64_idx] == PartEval::All {
            let v64 = &mut vt.split[blk64_idx];
            let nones = child_nones([
                &v64.split[0].part_variances,
                &v64.split[1].part_variances,
                &v64.split[2].part_variances,
                &v64.split[3].part_variances,
            ]);
            fill_variance_node(&nones, &mut v64.part_variances);
            get_variance(&mut v64.part_variances.none);
            // (the max/min 64x64 spread rule is !is_key_frame, :1873.)
        }
        if is_small_sb {
            force_split[0] = PartEval::OnlySplit;
        }
    }

    // Root 128x128 fill (:1883-1894): both root force rules are
    // !is_key_frame; the fill itself only runs when the root survived as
    // PART_EVAL_ALL (128-SB frames with no forced splits).
    if force_split[0] == PartEval::All {
        let nones = child_nones([
            &vt.split[0].part_variances,
            &vt.split[1].part_variances,
            &vt.split[2].part_variances,
            &vt.split[3].part_variances,
        ]);
        let mut pv = vt.part_variances;
        fill_variance_node(&nones, &mut pv);
        vt.part_variances = pv;
    }

    // ---- the partition assignment descent (:1896-1942) ----
    let root_set = mi_col + 32 <= f.tile_mi_col_end
        && mi_row + 32 <= f.tile_mi_row_end
        && set_vt_partitioning(
            stamps,
            f,
            NodeView {
                part_variances: &mut vt.part_variances,
            },
            BLOCK_128X128,
            mi_row,
            mi_col,
            thresholds[0],
            BLOCK_16X16,
            force_split[0],
        );
    if !root_set {
        for blk64_idx in 0..num_64x64_blocks {
            let x64_idx = blk_idx_x(blk64_idx, 4) as i32;
            let y64_idx = blk_idx_y(blk64_idx, 4) as i32;
            let blk64_scale_idx = blk64_idx << 2;
            if set_vt_partitioning(
                stamps,
                f,
                NodeView {
                    part_variances: &mut vt.split[blk64_idx].part_variances,
                },
                BLOCK_64X64,
                mi_row + y64_idx,
                mi_col + x64_idx,
                thresholds[1],
                BLOCK_16X16,
                force_split[1 + blk64_idx],
            ) {
                continue;
            }
            for lvl1_idx in 0..4usize {
                let x32_idx = blk_idx_x(lvl1_idx, 3) as i32;
                let y32_idx = blk_idx_y(lvl1_idx, 3) as i32;
                let lvl1_scale_idx = (blk64_scale_idx + lvl1_idx) << 2;
                if set_vt_partitioning(
                    stamps,
                    f,
                    NodeView {
                        part_variances: &mut vt.split[blk64_idx].split[lvl1_idx].part_variances,
                    },
                    BLOCK_32X32,
                    mi_row + y64_idx + y32_idx,
                    mi_col + x64_idx + x32_idx,
                    thresholds[2],
                    BLOCK_16X16,
                    force_split[5 + blk64_scale_idx + lvl1_idx],
                ) {
                    continue;
                }
                for lvl2_idx in 0..4usize {
                    let x16_idx = blk_idx_x(lvl2_idx, 2) as i32;
                    let y16_idx = blk_idx_y(lvl2_idx, 2) as i32;
                    let split_index = 21 + lvl1_scale_idx + lvl2_idx;
                    if set_vt_partitioning(
                        stamps,
                        f,
                        NodeView {
                            part_variances: &mut vt.split[blk64_idx].split[lvl1_idx].split
                                [lvl2_idx]
                                .part_variances,
                        },
                        BLOCK_16X16,
                        mi_row + y64_idx + y32_idx + y16_idx,
                        mi_col + x64_idx + x32_idx + x16_idx,
                        thresholds[3],
                        BLOCK_8X8,
                        force_split[split_index],
                    ) {
                        continue;
                    }
                    for lvl3_idx in 0..4usize {
                        let x8_idx = blk_idx_x(lvl3_idx, 1) as i32;
                        let y8_idx = blk_idx_y(lvl3_idx, 1) as i32;
                        set_block_size(
                            stamps,
                            f,
                            mi_row + y64_idx + y32_idx + y16_idx + y8_idx,
                            mi_col + x64_idx + x32_idx + x16_idx + x8_idx,
                            BLOCK_8X8,
                        );
                    }
                }
            }
        }
    }
}

/// `get_partition` (av1_common_int.h:1775) over the [`set_block_size`]
/// stamp grid: derive the partition type at `(mi_row, mi_col, bsize)` from
/// the stamped leaf bsizes. The variance tree only produces
/// NONE/HORZ/VERT/SPLIT shapes, but the derivation is transcribed in full
/// (the extended-partition disambiguation included) so it stays faithful at
/// frame edges.
pub fn get_partition_from_stamps(
    stamps: &[u8],
    mi_rows: i32,
    mi_cols: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> i32 {
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
    const PARTITION_INVALID: i32 = -1;

    if mi_row >= mi_rows || mi_col >= mi_cols {
        return PARTITION_INVALID;
    }
    let at = |r: i32, c: i32| -> usize { stamps[(r * mi_cols + c) as usize] as usize };
    let subsize = at(mi_row, mi_col);
    if subsize == bsize {
        return PARTITION_NONE;
    }
    let bhigh = MI_SIZE_WIDE_B[bsize] as i32; // square: high == wide
    let bwide = bhigh;
    let sshigh = MI_SIZE_HIGH_B[subsize] as i32;
    let sswide = MI_SIZE_WIDE_B[subsize] as i32;

    if bsize > BLOCK_8X8 && mi_row + bwide / 2 < mi_rows && mi_col + bhigh / 2 < mi_cols {
        // The block might be using an extended partition type.
        let mbmi_right = at(mi_row, mi_col + bwide / 2);
        let mbmi_below = at(mi_row + bhigh / 2, mi_col);
        if sswide == bwide {
            // PARTITION_HORZ_4, PARTITION_HORZ or PARTITION_HORZ_B.
            if sshigh * 4 == bhigh {
                return PARTITION_HORZ_4;
            }
            debug_assert_eq!(sshigh * 2, bhigh);
            if mbmi_below == subsize {
                return PARTITION_HORZ;
            }
            return PARTITION_HORZ_B;
        } else if sshigh == bhigh {
            // PARTITION_VERT_4, PARTITION_VERT or PARTITION_VERT_B.
            if sswide * 4 == bwide {
                return PARTITION_VERT_4;
            }
            debug_assert_eq!(sswide * 2, bwide);
            if mbmi_right == subsize {
                return PARTITION_VERT;
            }
            return PARTITION_VERT_B;
        } else {
            if sswide * 2 != bwide || sshigh * 2 != bhigh {
                return PARTITION_SPLIT;
            }
            if MI_SIZE_WIDE_B[mbmi_below] as i32 == bwide {
                return PARTITION_HORZ_A;
            }
            if MI_SIZE_HIGH_B[mbmi_right] as i32 == bhigh {
                return PARTITION_VERT_A;
            }
            return PARTITION_SPLIT;
        }
    }
    let vert_split = sswide < bwide;
    let horz_split = sshigh < bhigh;
    let split_idx = ((vert_split as usize) << 1) | horz_split as usize;
    debug_assert_ne!(split_idx, 0);
    [
        PARTITION_INVALID,
        PARTITION_HORZ,
        PARTITION_VERT,
        PARTITION_SPLIT,
    ][split_idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `get_variance` matches the plain (sse - sum^2/n) * 256 / n definition
    /// on in-range inputs (the C expression form is wrap-faithful; this
    /// pins the arithmetic on representative values).
    #[test]
    fn get_variance_matches_definition() {
        // 16 4x4 leaves of a 16x16: log2_count accumulates to 4 at the 16x16
        // none node (each leaf log2 0, three sum levels).
        let mut v = VPartVar {
            sum_square_error: 16 * 40 * 40,
            sum_error: 16 * 40,
            log2_count: 4,
            variance: 0,
        };
        get_variance(&mut v);
        // sse - sum^2 >> 4 = 25600 - (640*640)>>4 = 25600 - 25600 = 0.
        assert_eq!(v.variance, 0);

        let mut v = VPartVar {
            sum_square_error: 10_000,
            sum_error: 100,
            log2_count: 4,
            variance: 0,
        };
        get_variance(&mut v);
        // (10000 - 625) * 256 >> 4 = 9375 * 16 = 150000.
        assert_eq!(v.variance, 150_000);
    }

    /// Flat content ⇒ zero variance everywhere ⇒ the KEY tree bottoms out at
    /// 32x32 NONE stamps (64x64 must split per the `bsize > BLOCK_32X32`
    /// key rule; 32x32 var 0 < threshold).
    #[test]
    fn flat_64_sb_stamps_four_32x32() {
        let stride = 80usize;
        let src = vec![128u16; stride * 72];
        let f = VbpFrame {
            mi_rows: 16,
            mi_cols: 16,
            tile_mi_row_end: 16,
            tile_mi_col_end: 16,
            num_pixels: 64 * 64,
            sb_size: BLOCK_64X64,
            qindex: 100,
            bit_depth: 8,
            ss_x: 1,
            ss_y: 1,
        };
        let mut stamps = vec![0u8; 16 * 16];
        choose_var_based_partitioning_key(&mut stamps, &f, &src, 0, stride, 0, 0, false);
        for (r, c) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
            assert_eq!(stamps[r * 16 + c], BLOCK_32X32 as u8, "at ({r},{c})");
        }
        assert_eq!(
            get_partition_from_stamps(&stamps, 16, 16, 0, 0, BLOCK_64X64),
            3, // PARTITION_SPLIT
        );
        for (r, c) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
            assert_eq!(
                get_partition_from_stamps(&stamps, 16, 16, r, c, BLOCK_32X32),
                0, // PARTITION_NONE
                "at ({r},{c})"
            );
        }
    }

    /// A hard vertical edge inside one 32x32 (flat 16x16 quadrants): the
    /// 32x32's own variance exceeds thresholds[2] so the stage-2 walk
    /// FORCE-SPLITS it (PART_EVAL_ONLY_SPLIT fires before the rect arms can
    /// run — on KEY frames the interior rect stamps are reachable only on
    /// exact `variance == threshold` ties); the flat 16x16s under it stay
    /// NONE. Pins the force-split propagation + the SPLIT derivation.
    #[test]
    fn interior_edge_32x32_force_splits_to_16s() {
        let stride = 80usize;
        let mut src = vec![64u16; stride * 72];
        // Top-left 32x32: columns 0..16 = 64, 16..32 = 192 (each 16x16
        // flat); rest of the SB flat 128.
        for r in 0..64 {
            for c in 0..64 {
                src[r * stride + c] = if r < 32 && c < 32 {
                    if c < 16 { 64 } else { 192 }
                } else {
                    128
                };
            }
        }
        let f = VbpFrame {
            mi_rows: 16,
            mi_cols: 16,
            tile_mi_row_end: 16,
            tile_mi_col_end: 16,
            num_pixels: 64 * 64,
            sb_size: BLOCK_64X64,
            qindex: 220,
            bit_depth: 8,
            ss_x: 1,
            ss_y: 1,
        };
        let mut stamps = vec![0u8; 16 * 16];
        choose_var_based_partitioning_key(&mut stamps, &f, &src, 0, stride, 0, 0, false);
        assert_eq!(
            get_partition_from_stamps(&stamps, 16, 16, 0, 0, BLOCK_32X32),
            3, // PARTITION_SPLIT — forced by the 32x32 variance rule
        );
        for (r, c) in [(0, 0), (0, 4), (4, 0), (4, 4)] {
            assert_eq!(stamps[r * 16 + c], BLOCK_16X16 as u8, "at ({r},{c})");
            assert_eq!(
                get_partition_from_stamps(&stamps, 16, 16, r as i32, c as i32, BLOCK_16X16),
                0, // the flat 16x16s stay NONE
            );
        }
        // The flat quadrants stay NONE-32.
        assert_eq!(
            get_partition_from_stamps(&stamps, 16, 16, 8, 8, BLOCK_32X32),
            0
        );
    }

    /// Frame-edge rect stamps: on a 48x48 frame (mi_cols = 12) the (0,8)
    /// 32x32's right half is out of frame — the NONE fit check fails
    /// (`bs_width_check = (8>>1)+1 = 5`, 8+5 > 12) but the VERT half-width
    /// check passes (`(8>>2)+1 = 3`, 8+3 <= 12), so flat content stamps the
    /// visible 16x32 (the out-of-frame sibling stamp is skipped) and the
    /// stamp grid derives PARTITION_VERT via the base table (the
    /// ext-partition arm is bounds-gated off).
    #[test]
    fn edge_vert_single_strip_stamp() {
        let stride = 80usize;
        let src = vec![128u16; stride * 72];
        let f = VbpFrame {
            mi_rows: 12,
            mi_cols: 12,
            tile_mi_row_end: 12,
            tile_mi_col_end: 12,
            num_pixels: 48 * 48,
            sb_size: BLOCK_64X64,
            qindex: 100,
            bit_depth: 8,
            ss_x: 1,
            ss_y: 1,
        };
        let mut stamps = vec![0u8; 12 * 12];
        choose_var_based_partitioning_key(&mut stamps, &f, &src, 0, stride, 0, 0, false);
        // Interior 32x32 at (0,0): NONE (flat).
        assert_eq!(
            get_partition_from_stamps(&stamps, 12, 12, 0, 0, BLOCK_32X32),
            0
        );
        // Right-edge 32x32 at (0,8): single-strip VERT.
        assert_eq!(stamps[8], 7, "16x32 stamp at (0,8)"); // BLOCK_16X32 = 7
        assert_eq!(
            get_partition_from_stamps(&stamps, 12, 12, 0, 8, BLOCK_32X32),
            2, // PARTITION_VERT
        );
        // Bottom-edge 32x32 at (8,0): single-strip HORZ (fit fails on rows,
        // horz half-height check passes) -- 32x16 = BLOCK_32X16 = 8.
        assert_eq!(stamps[8 * 12], 8, "32x16 stamp at (8,0)");
        assert_eq!(
            get_partition_from_stamps(&stamps, 12, 12, 8, 0, BLOCK_32X32),
            1, // PARTITION_HORZ
        );
    }
}
