//! Uncompressed frame-header components (libaom `av1/encoder/bitstream.c`),
//! written through [`WriteBitBuffer`]. Byte-identical to C libaom. The functions
//! here are `static inline` in libaom, so their oracles are the same control flow
//! driven through the real `aom_wb` primitives (validated by `wb_diff`), plus
//! independent spec-layout anchors in the tests.

use crate::wb::WriteBitBuffer;

/// `write_delta_q`: a present-flag + 7-bit inverse-signed value (0 => just the flag).
fn write_delta_q(wb: &mut WriteBitBuffer, delta_q: i32) {
    if delta_q != 0 {
        wb.write_bit(1);
        wb.write_inv_signed_literal(delta_q, 6);
    } else {
        wb.write_bit(0);
    }
}

/// The `CommonQuantParams` fields the frame-header quantization block reads.
#[derive(Clone, Copy, Debug)]
pub struct QuantParamsHeader {
    pub base_qindex: i32,
    pub y_dc_delta_q: i32,
    pub u_dc_delta_q: i32,
    pub u_ac_delta_q: i32,
    pub v_dc_delta_q: i32,
    pub v_ac_delta_q: i32,
    pub using_qmatrix: bool,
    pub qmatrix_level_y: i32,
    pub qmatrix_level_u: i32,
    pub qmatrix_level_v: i32,
}

/// `encode_quantization`: the frame-header quantization params — base qindex
/// (`QINDEX_BITS`=8), the y/u/v dc/ac delta-qs (u/v only for `num_planes > 1`,
/// with the `diff_uv_delta` and `separate_uv_delta_q` gating), and the quant
/// matrix flag + levels (`QM_LEVEL_BITS`=4).
pub fn encode_quantization(
    wb: &mut WriteBitBuffer,
    qp: &QuantParamsHeader,
    num_planes: usize,
    separate_uv_delta_q: bool,
) {
    wb.write_literal(qp.base_qindex, 8);
    write_delta_q(wb, qp.y_dc_delta_q);
    if num_planes > 1 {
        let diff_uv_delta =
            qp.u_dc_delta_q != qp.v_dc_delta_q || qp.u_ac_delta_q != qp.v_ac_delta_q;
        if separate_uv_delta_q {
            wb.write_bit(diff_uv_delta as u32);
        }
        write_delta_q(wb, qp.u_dc_delta_q);
        write_delta_q(wb, qp.u_ac_delta_q);
        if diff_uv_delta {
            write_delta_q(wb, qp.v_dc_delta_q);
            write_delta_q(wb, qp.v_ac_delta_q);
        }
    }
    wb.write_bit(qp.using_qmatrix as u32);
    if qp.using_qmatrix {
        wb.write_literal(qp.qmatrix_level_y, 4);
        wb.write_literal(qp.qmatrix_level_u, 4);
        if separate_uv_delta_q {
            wb.write_literal(qp.qmatrix_level_v, 4);
        }
    }
}

/// The loop-filter frame-header state (`cm->lf` + the resolved primary-ref-frame
/// "last" deltas — the caller picks `av1_set_default_*_deltas` when there is no
/// primary ref buffer).
#[derive(Clone, Copy, Debug)]
pub struct LoopfilterHeader {
    pub allow_intrabc: bool,
    pub filter_level: [i32; 2],
    pub filter_level_u: i32,
    pub filter_level_v: i32,
    pub sharpness_level: i32,
    pub mode_ref_delta_enabled: bool,
    pub mode_ref_delta_update: bool,
    pub ref_deltas: [i8; 8],       // REF_FRAMES
    pub mode_deltas: [i8; 2],      // MAX_MODE_LF_DELTAS
    pub last_ref_deltas: [i8; 8],
    pub last_mode_deltas: [i8; 2],
}

/// `encode_loopfilter` (`av1/encoder/bitstream.c`): the loop-filter params —
/// y/uv filter levels, sharpness, and (when meaningful) the per-ref / per-mode
/// delta updates vs the previous frame's deltas. Writes nothing when
/// `allow_intrabc`.
pub fn encode_loopfilter(wb: &mut WriteBitBuffer, lf: &LoopfilterHeader, num_planes: usize) {
    if lf.allow_intrabc {
        return;
    }
    wb.write_literal(lf.filter_level[0], 6);
    wb.write_literal(lf.filter_level[1], 6);
    if num_planes > 1 && (lf.filter_level[0] != 0 || lf.filter_level[1] != 0) {
        wb.write_literal(lf.filter_level_u, 6);
        wb.write_literal(lf.filter_level_v, 6);
    }
    wb.write_literal(lf.sharpness_level, 3);
    wb.write_bit(lf.mode_ref_delta_enabled as u32);

    let meaningful = lf.mode_ref_delta_update
        && (lf.ref_deltas.iter().zip(&lf.last_ref_deltas).any(|(a, b)| a != b)
            || lf.mode_deltas.iter().zip(&lf.last_mode_deltas).any(|(a, b)| a != b));
    wb.write_bit(meaningful as u32);
    if !meaningful {
        return;
    }
    for (&delta, &last) in lf.ref_deltas.iter().zip(&lf.last_ref_deltas) {
        let changed = delta != last;
        wb.write_bit(changed as u32);
        if changed {
            wb.write_inv_signed_literal(delta as i32, 6);
        }
    }
    for (&delta, &last) in lf.mode_deltas.iter().zip(&lf.last_mode_deltas) {
        let changed = delta != last;
        wb.write_bit(changed as u32);
        if changed {
            wb.write_inv_signed_literal(delta as i32, 6);
        }
    }
}

/// The CDEF frame-header state (`cm->cdef_info`).
#[derive(Clone, Copy, Debug)]
pub struct CdefHeader {
    pub enable_cdef: bool,
    pub allow_intrabc: bool,
    pub cdef_damping: i32,
    pub cdef_bits: i32,
    pub nb_cdef_strengths: usize,
    pub cdef_strengths: [i32; 8],
    pub cdef_uv_strengths: [i32; 8],
}

/// `encode_cdef` (`av1/encoder/bitstream.c`): CDEF params — damping (`-3`, 2 bits),
/// `cdef_bits` (2 bits), then `nb_cdef_strengths` y (and, for `num_planes > 1`, uv)
/// strengths at `CDEF_STRENGTH_BITS`=6. Writes nothing when CDEF is disabled or intrabc.
pub fn encode_cdef(wb: &mut WriteBitBuffer, cdef: &CdefHeader, num_planes: usize) {
    if !cdef.enable_cdef || cdef.allow_intrabc {
        return;
    }
    wb.write_literal(cdef.cdef_damping - 3, 2);
    wb.write_literal(cdef.cdef_bits, 2);
    for i in 0..cdef.nb_cdef_strengths {
        wb.write_literal(cdef.cdef_strengths[i], 6);
        if num_planes > 1 {
            wb.write_literal(cdef.cdef_uv_strengths[i], 6);
        }
    }
}

// ---- segmentation ---------------------------------------------------------

const MAX_SEGMENTS: usize = 8;
const SEG_LVL_MAX: usize = 8;
/// `av1_seg_feature_data_max` table (`seg_common.c`): MAXQ, then MAX_LOOP_FILTER×4,
/// then 7 (REF_FRAME), 0 (SKIP), 0 (GLOBALMV).
const SEG_FEATURE_DATA_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];
/// `av1_is_segfeature_signed` table: the ALT_Q + 4 ALT_LF features are signed.
const SEG_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];

/// `get_unsigned_bits` (`common.h`): `num > 0 ? get_msb(num) + 1 : 0`.
fn get_unsigned_bits(num_values: u32) -> u32 {
    if num_values == 0 { 0 } else { 32 - num_values.leading_zeros() }
}

/// The segmentation frame-header state (`cm->seg` + `primary_ref_frame`).
#[derive(Clone, Debug)]
pub struct SegmentationHeader {
    pub enabled: bool,
    /// `primary_ref_frame != PRIMARY_REF_NONE` — gates the update flags.
    pub has_primary_ref: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    /// `feature_mask[seg]` — bit `j` set means feature `j` is active for segment `seg`.
    pub feature_mask: [u32; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// `encode_segmentation` (`av1/encoder/bitstream.c`): the segmentation params —
/// enabled flag, the update-map/temporal/update-data flags (only with a primary
/// ref), then, when `update_data`, per (segment × feature) an active bit and the
/// clamped feature value (inv-signed for the signed features, plain literal
/// otherwise, both at `get_unsigned_bits(data_max)`).
pub fn encode_segmentation(wb: &mut WriteBitBuffer, seg: &SegmentationHeader) {
    wb.write_bit(seg.enabled as u32);
    if !seg.enabled {
        return;
    }
    if seg.has_primary_ref {
        wb.write_bit(seg.update_map as u32);
        if seg.update_map {
            wb.write_bit(seg.temporal_update as u32);
        }
        wb.write_bit(seg.update_data as u32);
    }
    if seg.update_data {
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                let active = seg.feature_mask[i] & (1 << j) != 0;
                wb.write_bit(active as u32);
                if active {
                    let data_max = SEG_FEATURE_DATA_MAX[j];
                    let ubits = get_unsigned_bits(data_max as u32);
                    let data = seg.feature_data[i][j].clamp(-data_max, data_max);
                    if SEG_FEATURE_SIGNED[j] {
                        wb.write_inv_signed_literal(data, ubits);
                    } else {
                        wb.write_literal(data, ubits);
                    }
                }
            }
        }
    }
}

// ---- interpolation filter / frame size ------------------------------------

const SWITCHABLE: i32 = 4; // SWITCHABLE_FILTERS + 1
const LOG_SWITCHABLE_FILTERS: u32 = 2;
const SCALE_NUMERATOR: i32 = 8;
const SUPERRES_SCALE_DENOMINATOR_MIN: i32 = SCALE_NUMERATOR + 1;
const SUPERRES_SCALE_BITS: u32 = 3;

/// `write_frame_interp_filter`: a SWITCHABLE flag, and (when not switchable) the
/// filter index at `LOG_SWITCHABLE_FILTERS`=2 bits.
pub fn write_frame_interp_filter(wb: &mut WriteBitBuffer, filter: i32) {
    wb.write_bit((filter == SWITCHABLE) as u32);
    if filter != SWITCHABLE {
        wb.write_literal(filter, LOG_SWITCHABLE_FILTERS);
    }
}

/// `write_superres_scale`: nothing when superres is disabled; otherwise a scale
/// flag and (when scaling) the denominator offset at `SUPERRES_SCALE_BITS`=3.
pub fn write_superres_scale(wb: &mut WriteBitBuffer, enable_superres: bool, scale_denominator: i32) {
    if !enable_superres {
        return;
    }
    if scale_denominator == SCALE_NUMERATOR {
        wb.write_bit(0);
    } else {
        wb.write_bit(1);
        wb.write_literal(scale_denominator - SUPERRES_SCALE_DENOMINATOR_MIN, SUPERRES_SCALE_BITS);
    }
}

/// `write_render_size`: a scaling-active flag, and (when active) render width/height
/// minus one at 16 bits each.
pub fn write_render_size(wb: &mut WriteBitBuffer, scaling_active: bool, render_width: i32, render_height: i32) {
    wb.write_bit(scaling_active as u32);
    if scaling_active {
        wb.write_literal(render_width - 1, 16);
        wb.write_literal(render_height - 1, 16);
    }
}

/// The frame-size frame-header state (`write_frame_size` inputs).
#[derive(Clone, Copy, Debug)]
pub struct FrameSizeHeader {
    pub frame_size_override: bool,
    pub num_bits_width: u32,
    pub num_bits_height: u32,
    pub superres_upscaled_width: i32,
    pub superres_upscaled_height: i32,
    pub enable_superres: bool,
    pub scale_denominator: i32,
    pub scaling_active: bool,
    pub render_width: i32,
    pub render_height: i32,
}

/// `write_frame_size`: the coded width/height minus one (only when overriding the
/// sequence-header size), then the superres scale and render size.
pub fn write_frame_size(wb: &mut WriteBitBuffer, fs: &FrameSizeHeader) {
    let coded_width = fs.superres_upscaled_width - 1;
    let coded_height = fs.superres_upscaled_height - 1;
    if fs.frame_size_override {
        wb.write_literal(coded_width, fs.num_bits_width);
        wb.write_literal(coded_height, fs.num_bits_height);
    }
    write_superres_scale(wb, fs.enable_superres, fs.scale_denominator);
    write_render_size(wb, fs.scaling_active, fs.render_width, fs.render_height);
}

// ---- tile info ------------------------------------------------------------

const MAX_TILE_COLS: usize = 64;
const MAX_TILE_ROWS: usize = 64;

/// `CEIL_POWER_OF_TWO(value, n)` (`aom_ports/mem.h`): `ceil(value / 2^n)`.
fn ceil_power_of_two(value: i32, n: u32) -> i32 {
    (value + (1 << n) - 1) >> n
}

/// `wb_write_uniform` (`av1/encoder/bitstream.c`): the uncompressed-header form of
/// `write_uniform` — a value `v` in `[0, n)` coded in `l-1` or `l` bits where
/// `l = get_unsigned_bits(n)` and `m = (1 << l) - n`.
pub fn wb_write_uniform(wb: &mut WriteBitBuffer, n: i32, v: i32) {
    let l = get_unsigned_bits(n as u32);
    if l == 0 {
        return;
    }
    let m = (1i32 << l) - n;
    if v < m {
        wb.write_literal(v, l - 1);
    } else {
        wb.write_literal(m + ((v - m) >> 1), l - 1);
        wb.write_literal((v - m) & 1, 1);
    }
}

/// The tile-info frame-header state (`cm->mi_params` + `cm->tiles`).
#[derive(Clone, Debug)]
pub struct TileInfoHeader {
    pub mi_cols: i32,
    pub mi_rows: i32,
    pub mib_size_log2: u32,
    pub uniform_spacing: bool,
    pub log2_cols: i32,
    pub min_log2_cols: i32,
    pub max_log2_cols: i32,
    pub log2_rows: i32,
    pub min_log2_rows: i32,
    pub max_log2_rows: i32,
    pub cols: usize,
    pub rows: usize,
    pub col_start_sb: [i32; MAX_TILE_COLS + 1],
    pub row_start_sb: [i32; MAX_TILE_ROWS + 1],
    pub max_width_sb: i32,
    pub max_height_sb: i32,
}

/// `write_tile_info_max_tile`: uniform-spacing flag, then either the unary
/// log2-cols/rows increments (uniform) or the per-tile `wb_write_uniform` sizes
/// (explicit).
pub fn write_tile_info_max_tile(wb: &mut WriteBitBuffer, t: &TileInfoHeader) {
    let mut width_sb = ceil_power_of_two(t.mi_cols, t.mib_size_log2);
    let mut height_sb = ceil_power_of_two(t.mi_rows, t.mib_size_log2);
    wb.write_bit(t.uniform_spacing as u32);
    if t.uniform_spacing {
        for _ in 0..(t.log2_cols - t.min_log2_cols) {
            wb.write_bit(1);
        }
        if t.log2_cols < t.max_log2_cols {
            wb.write_bit(0);
        }
        for _ in 0..(t.log2_rows - t.min_log2_rows) {
            wb.write_bit(1);
        }
        if t.log2_rows < t.max_log2_rows {
            wb.write_bit(0);
        }
    } else {
        for i in 0..t.cols {
            let size_sb = t.col_start_sb[i + 1] - t.col_start_sb[i];
            wb_write_uniform(wb, width_sb.min(t.max_width_sb), size_sb - 1);
            width_sb -= size_sb;
        }
        for i in 0..t.rows {
            let size_sb = t.row_start_sb[i + 1] - t.row_start_sb[i];
            wb_write_uniform(wb, height_sb.min(t.max_height_sb), size_sb - 1);
            height_sb -= size_sb;
        }
    }
}

/// `write_tile_info`: `write_tile_info_max_tile`, then (for >1 tile) the CDF-update
/// tile id (all zero here) and the tile-size-bytes-minus-one field (=3, 2 bits).
pub fn write_tile_info(wb: &mut WriteBitBuffer, t: &TileInfoHeader) {
    write_tile_info_max_tile(wb, t);
    if t.rows * t.cols > 1 {
        wb.write_literal(0, (t.log2_cols + t.log2_rows) as u32);
        wb.write_literal(3, 2);
    }
}
