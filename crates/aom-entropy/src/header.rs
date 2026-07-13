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

// ---- loop restoration -----------------------------------------------------

const RESTORE_NONE: u8 = 0;
const RESTORE_WIENER: u8 = 1;
const RESTORE_SGRPROJ: u8 = 2;
const RESTORE_SWITCHABLE: u8 = 3;

/// The loop-restoration frame-header state (`cm->rst_info` + seq/features flags).
#[derive(Clone, Copy, Debug)]
pub struct RestorationHeader {
    pub enable_restoration: bool,
    pub allow_intrabc: bool,
    /// Per-plane `frame_restoration_type` (`RESTORE_*`).
    pub frame_restoration_type: [u8; 3],
    /// `sb_size == BLOCK_128X128`.
    pub sb_size_128: bool,
    pub restoration_unit_size: [i32; 3],
    pub subsampling_x: i32,
    pub subsampling_y: i32,
}

/// `encode_restoration_mode` (`av1/encoder/bitstream.c`): the per-plane restoration
/// type (2 bits, mapped NONE=00 WIENER=10 SGRPROJ=11 SWITCHABLE=01), then (when any
/// plane restores) the luma restoration-unit-size increments, and the chroma
/// unit-size-differs flag.
pub fn encode_restoration_mode(wb: &mut WriteBitBuffer, r: &RestorationHeader, num_planes: usize) {
    if !r.enable_restoration || r.allow_intrabc {
        return;
    }
    let mut all_none = true;
    let mut chroma_none = true;
    for p in 0..num_planes {
        let ft = r.frame_restoration_type[p];
        if ft != RESTORE_NONE {
            all_none = false;
            chroma_none &= p == 0;
        }
        match ft {
            RESTORE_NONE => {
                wb.write_bit(0);
                wb.write_bit(0);
            }
            RESTORE_WIENER => {
                wb.write_bit(1);
                wb.write_bit(0);
            }
            RESTORE_SGRPROJ => {
                wb.write_bit(1);
                wb.write_bit(1);
            }
            RESTORE_SWITCHABLE => {
                wb.write_bit(0);
                wb.write_bit(1);
            }
            _ => unreachable!(),
        }
    }
    if !all_none {
        let sb_size = if r.sb_size_128 { 128 } else { 64 };
        let rus = r.restoration_unit_size[0];
        if sb_size == 64 {
            wb.write_bit((rus > 64) as u32);
        }
        if rus > 64 {
            wb.write_bit((rus > 128) as u32);
        }
    }
    if num_planes > 1 {
        let s = r.subsampling_x.min(r.subsampling_y);
        if s != 0 && !chroma_none {
            wb.write_bit((r.restoration_unit_size[1] != r.restoration_unit_size[0]) as u32);
        }
    }
}

// ---- frame-level delta-q / delta-lf + tx mode -----------------------------

/// `get_msb` (`aom_ports/bitops.h`): the index of the most-significant set bit
/// (`floor(log2(n))`), for `n > 0`.
fn get_msb(n: u32) -> u32 {
    31 - n.leading_zeros()
}

/// The frame-level delta-q / delta-lf params (`cm->delta_q_info`).
#[derive(Clone, Copy, Debug)]
pub struct DeltaQParams {
    pub base_qindex: i32,
    pub delta_q_present: bool,
    pub delta_q_res: i32, // a power of two
    pub allow_intrabc: bool,
    pub delta_lf_present: bool,
    pub delta_lf_res: i32, // a power of two
    pub delta_lf_multi: bool,
}

/// `write_delta_q_params` (the frame-header block in `write_uncompressed_header_obu`):
/// only when `base_qindex > 0` — the delta-q present flag, its log2 resolution
/// (2 bits), and (when not intrabc) the delta-lf present flag with its log2
/// resolution + multi flag.
pub fn write_delta_q_params(wb: &mut WriteBitBuffer, d: &DeltaQParams) {
    if d.base_qindex > 0 {
        wb.write_bit(d.delta_q_present as u32);
        if d.delta_q_present {
            wb.write_literal(get_msb(d.delta_q_res as u32) as i32, 2);
            if !d.allow_intrabc {
                wb.write_bit(d.delta_lf_present as u32);
            }
            if d.delta_lf_present {
                wb.write_literal(get_msb(d.delta_lf_res as u32) as i32, 2);
                wb.write_bit(d.delta_lf_multi as u32);
            }
        }
    }
}

/// `write_tx_mode` (inline in the uncompressed header): a single bit
/// `tx_mode == TX_MODE_SELECT`, suppressed when the frame is coded-lossless
/// (then `tx_mode` is forced to `ONLY_4X4`).
pub fn write_tx_mode(wb: &mut WriteBitBuffer, coded_lossless: bool, tx_mode_select: bool) {
    if !coded_lossless {
        wb.write_bit(tx_mode_select as u32);
    }
}

// ---- film grain -----------------------------------------------------------

/// The film-grain params written into the frame header (`aom_film_grain_t`), plus
/// the seq/frame context the writer reads (monochrome, subsampling, inter frame).
#[derive(Clone, Debug)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub random_seed: i32,
    pub is_inter_frame: bool,
    pub update_parameters: bool,
    pub ref_idx: i32,
    pub num_y_points: i32,
    pub scaling_points_y: [[i32; 2]; 14],
    pub monochrome: bool,
    pub chroma_scaling_from_luma: bool,
    pub subsampling_x: i32,
    pub subsampling_y: i32,
    pub num_cb_points: i32,
    pub scaling_points_cb: [[i32; 2]; 10],
    pub num_cr_points: i32,
    pub scaling_points_cr: [[i32; 2]; 10],
    pub scaling_shift: i32,
    pub ar_coeff_lag: i32,
    pub ar_coeffs_y: [i32; 24],
    pub ar_coeffs_cb: [i32; 25],
    pub ar_coeffs_cr: [i32; 25],
    pub ar_coeff_shift: i32,
    pub grain_scale_shift: i32,
    pub cb_mult: i32,
    pub cb_luma_mult: i32,
    pub cb_offset: i32,
    pub cr_mult: i32,
    pub cr_luma_mult: i32,
    pub cr_offset: i32,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// `write_film_grain_params` (`av1/encoder/bitstream.c`): the grain apply flag and,
/// when applying, the random seed, the copy-from-ref index (when not updating), or
/// the full scaling-point / AR-coeff / multiplier parameter set. The `!update`
/// ref-search is encoder logic with no byte effect beyond the 3-bit `ref_idx`.
pub fn write_film_grain_params(wb: &mut WriteBitBuffer, p: &FilmGrainParams) {
    wb.write_bit(p.apply_grain as u32);
    if !p.apply_grain {
        return;
    }
    wb.write_literal(p.random_seed, 16);
    if p.is_inter_frame {
        wb.write_bit(p.update_parameters as u32);
    }
    if !p.update_parameters {
        wb.write_literal(p.ref_idx, 3);
        return;
    }
    wb.write_literal(p.num_y_points, 4);
    for pt in &p.scaling_points_y[..p.num_y_points as usize] {
        wb.write_literal(pt[0], 8);
        wb.write_literal(pt[1], 8);
    }
    if !p.monochrome {
        wb.write_bit(p.chroma_scaling_from_luma as u32);
    }
    let chroma_absent = p.monochrome
        || p.chroma_scaling_from_luma
        || (p.subsampling_x == 1 && p.subsampling_y == 1 && p.num_y_points == 0);
    if !chroma_absent {
        wb.write_literal(p.num_cb_points, 4);
        for pt in &p.scaling_points_cb[..p.num_cb_points as usize] {
            wb.write_literal(pt[0], 8);
            wb.write_literal(pt[1], 8);
        }
        wb.write_literal(p.num_cr_points, 4);
        for pt in &p.scaling_points_cr[..p.num_cr_points as usize] {
            wb.write_literal(pt[0], 8);
            wb.write_literal(pt[1], 8);
        }
    }
    wb.write_literal(p.scaling_shift - 8, 2);
    wb.write_literal(p.ar_coeff_lag, 2);
    let num_pos_luma = 2 * p.ar_coeff_lag * (p.ar_coeff_lag + 1);
    let num_pos_chroma = num_pos_luma + (p.num_y_points > 0) as i32;
    if p.num_y_points != 0 {
        for &c in &p.ar_coeffs_y[..num_pos_luma as usize] {
            wb.write_literal(c + 128, 8);
        }
    }
    if p.num_cb_points != 0 || p.chroma_scaling_from_luma {
        for &c in &p.ar_coeffs_cb[..num_pos_chroma as usize] {
            wb.write_literal(c + 128, 8);
        }
    }
    if p.num_cr_points != 0 || p.chroma_scaling_from_luma {
        for &c in &p.ar_coeffs_cr[..num_pos_chroma as usize] {
            wb.write_literal(c + 128, 8);
        }
    }
    wb.write_literal(p.ar_coeff_shift - 6, 2);
    wb.write_literal(p.grain_scale_shift, 2);
    if p.num_cb_points != 0 {
        wb.write_literal(p.cb_mult, 8);
        wb.write_literal(p.cb_luma_mult, 8);
        wb.write_literal(p.cb_offset, 9);
    }
    if p.num_cr_points != 0 {
        wb.write_literal(p.cr_mult, 8);
        wb.write_literal(p.cr_luma_mult, 8);
        wb.write_literal(p.cr_offset, 9);
    }
    wb.write_bit(p.overlap_flag as u32);
    wb.write_bit(p.clip_to_restricted_range as u32);
}

// ---- global motion --------------------------------------------------------

const IDENTITY: u8 = 0;
const TRANSLATION: u8 = 1;
const ROTZOOM: u8 = 2;
const AFFINE: u8 = 3;

const GM_ALPHA_MAX: i32 = 1 << 12; // 1 << GM_ABS_ALPHA_BITS
const SUBEXPFIN_K: u16 = 3;
const GM_ALPHA_PREC_DIFF: u32 = 1; // WARPEDMODEL_PREC_BITS - GM_ALPHA_PREC_BITS = 16-15
const GM_ALPHA_PREC_BITS: i32 = 15;
const GM_ABS_TRANS_BITS: u32 = 12;
const GM_ABS_TRANS_ONLY_BITS: u32 = 9; // GM_ABS_TRANS_BITS - GM_TRANS_PREC_BITS + 3
const GM_TRANS_PREC_DIFF: u32 = 10; // WARPEDMODEL_PREC_BITS - GM_TRANS_PREC_BITS = 16-6
const GM_TRANS_ONLY_PREC_DIFF: u32 = 13; // WARPEDMODEL_PREC_BITS - 3

/// A single reference frame's global-motion model (`WarpedMotionParams`).
#[derive(Clone, Copy, Debug)]
pub struct WarpedMotionParams {
    pub wmtype: u8,
    pub wmmat: [i32; 6],
}

/// `write_global_motion_params` (`av1/encoder/bitstream.c`): the transform-type flags
/// (IDENTITY/ROTZOOM/TRANSLATION), then the rot-zoom / affine / translation model
/// parameters, each subexp-coded (`write_signed_primitive_refsubexpfin`) relative to
/// the reference frame's parameter at the matching precision.
pub fn write_global_motion_params(
    wb: &mut WriteBitBuffer,
    params: &WarpedMotionParams,
    ref_params: &WarpedMotionParams,
    allow_hp: bool,
) {
    let ty = params.wmtype;
    wb.write_bit((ty != IDENTITY) as u32);
    if ty != IDENTITY {
        wb.write_bit((ty == ROTZOOM) as u32);
        if ty != ROTZOOM {
            wb.write_bit((ty == TRANSLATION) as u32);
        }
    }
    let alpha_n = (GM_ALPHA_MAX + 1) as u16;
    if ty >= ROTZOOM {
        wb.write_signed_primitive_refsubexpfin(
            alpha_n,
            SUBEXPFIN_K,
            ((ref_params.wmmat[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS)) as i16,
            ((params.wmmat[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS)) as i16,
        );
        wb.write_signed_primitive_refsubexpfin(
            alpha_n,
            SUBEXPFIN_K,
            (ref_params.wmmat[3] >> GM_ALPHA_PREC_DIFF) as i16,
            (params.wmmat[3] >> GM_ALPHA_PREC_DIFF) as i16,
        );
    }
    if ty >= AFFINE {
        wb.write_signed_primitive_refsubexpfin(
            alpha_n,
            SUBEXPFIN_K,
            (ref_params.wmmat[4] >> GM_ALPHA_PREC_DIFF) as i16,
            (params.wmmat[4] >> GM_ALPHA_PREC_DIFF) as i16,
        );
        wb.write_signed_primitive_refsubexpfin(
            alpha_n,
            SUBEXPFIN_K,
            ((ref_params.wmmat[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS)) as i16,
            ((params.wmmat[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS)) as i16,
        );
    }
    if ty >= TRANSLATION {
        let trans_bits = if ty == TRANSLATION {
            GM_ABS_TRANS_ONLY_BITS - !allow_hp as u32
        } else {
            GM_ABS_TRANS_BITS
        };
        let trans_prec_diff = if ty == TRANSLATION {
            GM_TRANS_ONLY_PREC_DIFF + !allow_hp as u32
        } else {
            GM_TRANS_PREC_DIFF
        };
        let trans_n = ((1i32 << trans_bits) + 1) as u16;
        wb.write_signed_primitive_refsubexpfin(
            trans_n,
            SUBEXPFIN_K,
            (ref_params.wmmat[0] >> trans_prec_diff) as i16,
            (params.wmmat[0] >> trans_prec_diff) as i16,
        );
        wb.write_signed_primitive_refsubexpfin(
            trans_n,
            SUBEXPFIN_K,
            (ref_params.wmmat[1] >> trans_prec_diff) as i16,
            (params.wmmat[1] >> trans_prec_diff) as i16,
        );
    }
}

/// `write_global_motion` (`av1/encoder/bitstream.c`): the per-inter-frame loop over
/// the 7 reference frames (`LAST_FRAME..=ALTREF_FRAME`), each written against the
/// previous frame's model (or the identity default when there is no previous frame).
pub fn write_global_motion(
    wb: &mut WriteBitBuffer,
    global_motion: &[WarpedMotionParams; 7],
    ref_global_motion: &[WarpedMotionParams; 7],
    allow_hp: bool,
) {
    for (gm, refgm) in global_motion.iter().zip(ref_global_motion.iter()) {
        write_global_motion_params(wb, gm, refgm, allow_hp);
    }
}

// ---- sequence header ------------------------------------------------------

/// The `SequenceHeader` fields written by `write_sequence_header` (the middle chunk
/// of the sequence-header OBU — not the profile/timing/color-config framing).
#[derive(Clone, Copy, Debug)]
pub struct SequenceHeaderParams {
    pub num_bits_width: u32,
    pub num_bits_height: u32,
    pub max_frame_width: i32,
    pub max_frame_height: i32,
    pub reduced_still_picture_hdr: bool,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length: i32,
    pub frame_id_length: i32,
    pub sb_size_128: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_dist_wtd_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub force_screen_content_tools: i32, // 0, 1, or 2 (SELECT)
    pub force_integer_mv: i32,           // 0, 1, or 2 (SELECT)
    pub order_hint_bits_minus_1: i32,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
}

/// `write_sequence_header` (`av1/encoder/bitstream.c`): frame-size bit-widths + max
/// dimensions, the frame-id lengths (unless reduced still-picture), the superblock
/// size, the intra/inter tool-enable flags, order-hint config, the
/// screen-content-tools / integer-mv SELECT coding, and the post-filter enables.
pub fn write_sequence_header(wb: &mut WriteBitBuffer, s: &SequenceHeaderParams) {
    wb.write_literal(s.num_bits_width as i32 - 1, 4);
    wb.write_literal(s.num_bits_height as i32 - 1, 4);
    wb.write_literal(s.max_frame_width - 1, s.num_bits_width);
    wb.write_literal(s.max_frame_height - 1, s.num_bits_height);

    if !s.reduced_still_picture_hdr {
        wb.write_bit(s.frame_id_numbers_present_flag as u32);
        if s.frame_id_numbers_present_flag {
            wb.write_literal(s.delta_frame_id_length - 2, 4);
            wb.write_literal(s.frame_id_length - s.delta_frame_id_length - 1, 3);
        }
    }

    // write_sb_size
    wb.write_bit(s.sb_size_128 as u32);

    wb.write_bit(s.enable_filter_intra as u32);
    wb.write_bit(s.enable_intra_edge_filter as u32);

    if !s.reduced_still_picture_hdr {
        wb.write_bit(s.enable_interintra_compound as u32);
        wb.write_bit(s.enable_masked_compound as u32);
        wb.write_bit(s.enable_warped_motion as u32);
        wb.write_bit(s.enable_dual_filter as u32);
        wb.write_bit(s.enable_order_hint as u32);
        if s.enable_order_hint {
            wb.write_bit(s.enable_dist_wtd_comp as u32);
            wb.write_bit(s.enable_ref_frame_mvs as u32);
        }
        if s.force_screen_content_tools == 2 {
            wb.write_bit(1);
        } else {
            wb.write_bit(0);
            wb.write_bit(s.force_screen_content_tools as u32);
        }
        if s.force_screen_content_tools > 0 {
            if s.force_integer_mv == 2 {
                wb.write_bit(1);
            } else {
                wb.write_bit(0);
                wb.write_bit(s.force_integer_mv as u32);
            }
        }
        if s.enable_order_hint {
            wb.write_literal(s.order_hint_bits_minus_1, 3);
        }
    }

    wb.write_bit(s.enable_superres as u32);
    wb.write_bit(s.enable_cdef as u32);
    wb.write_bit(s.enable_restoration as u32);
}

// ---- ext tile info --------------------------------------------------------

/// `write_ext_tile_info` (`av1/encoder/bitstream.c`, large-scale-tile path):
/// byte-align (padding governed by the writer's current bit position), then (for >1
/// tile) the tile-column-size-bytes and tile-size-bytes fields (both written as
/// 0 = 1 byte here). The `saved_wb` snapshot has no byte effect.
pub fn write_ext_tile_info(wb: &mut WriteBitBuffer, rows: usize, cols: usize) {
    wb.byte_align_zeros();
    if rows * cols > 1 {
        wb.write_literal(0, 2);
        wb.write_literal(0, 2);
    }
}

// ---- color config ---------------------------------------------------------

/// The `SequenceHeader` color fields written by `write_color_config`.
#[derive(Clone, Copy, Debug)]
pub struct ColorConfigParams {
    pub bit_depth: i32, // 8, 10, 12
    pub profile: i32,   // 0, 1, 2
    pub monochrome: bool,
    pub color_primaries: i32,
    pub transfer_characteristics: i32,
    pub matrix_coefficients: i32,
    pub color_range: bool,
    pub subsampling_x: i32,
    pub subsampling_y: i32,
    pub chroma_sample_position: i32,
    pub separate_uv_delta_q: bool,
}

/// `write_bitdepth` (`av1/encoder/bitstream.c`): the high-bitdepth flag, plus (for
/// PROFILE_2 above 8-bit) the 10-vs-12-bit selector.
fn write_bitdepth(wb: &mut WriteBitBuffer, bit_depth: i32, profile: i32) {
    wb.write_bit((bit_depth != 8) as u32); // AOM_BITS_8 ? 0 : 1
    if profile == 2 && bit_depth != 8 {
        wb.write_bit((bit_depth == 12) as u32); // AOM_BITS_10 ? 0 : 1
    }
}

/// `write_color_config` (`av1/encoder/bitstream.c`): bit depth, the monochrome flag
/// (except PROFILE_1), the CICP color description (present flag + primaries/transfer/
/// matrix), the color range, the profile/bit-depth-gated subsampling, the chroma
/// sample position (for 4:2:0), and the separate-uv-delta-q flag. The many spec
/// asserts have no byte effect and are omitted.
pub fn write_color_config(wb: &mut WriteBitBuffer, c: &ColorConfigParams) {
    write_bitdepth(wb, c.bit_depth, c.profile);
    let is_monochrome = c.monochrome;
    if c.profile != 1 {
        wb.write_bit(is_monochrome as u32);
    }
    if c.color_primaries == 2 && c.transfer_characteristics == 2 && c.matrix_coefficients == 2 {
        wb.write_bit(0); // no color description
    } else {
        wb.write_bit(1); // color description present
        wb.write_literal(c.color_primaries, 8);
        wb.write_literal(c.transfer_characteristics, 8);
        wb.write_literal(c.matrix_coefficients, 8);
    }
    if is_monochrome {
        wb.write_bit(c.color_range as u32);
        return;
    }
    let is_srgb = c.color_primaries == 1 && c.transfer_characteristics == 13 && c.matrix_coefficients == 0;
    if !is_srgb {
        wb.write_bit(c.color_range as u32);
        if c.profile == 2 && c.bit_depth == 12 {
            wb.write_bit(c.subsampling_x as u32);
            if c.subsampling_x != 0 {
                wb.write_bit(c.subsampling_y as u32);
            }
        }
        if c.subsampling_x == 1 && c.subsampling_y == 1 {
            wb.write_literal(c.chroma_sample_position, 2);
        }
    }
    wb.write_bit(c.separate_uv_delta_q as u32);
}

// ---- timing info / decoder model ------------------------------------------

/// The `aom_timing_info_t` fields written by `write_timing_info_header`.
#[derive(Clone, Copy, Debug)]
pub struct TimingInfoHeader {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture: u32,
}

/// `write_timing_info_header`: two 32-bit tick fields, the equal-picture-interval
/// flag, and (when equal) the ticks-per-picture minus one as a uvlc.
pub fn write_timing_info_header(wb: &mut WriteBitBuffer, t: &TimingInfoHeader) {
    wb.write_unsigned_literal(t.num_units_in_display_tick, 32);
    wb.write_unsigned_literal(t.time_scale, 32);
    wb.write_bit(t.equal_picture_interval as u32);
    if t.equal_picture_interval {
        wb.write_uvlc(t.num_ticks_per_picture - 1);
    }
}

/// The `aom_dec_model_info_t` fields written by `write_decoder_model_info`.
#[derive(Clone, Copy, Debug)]
pub struct DecoderModelInfo {
    pub encoder_decoder_buffer_delay_length: i32,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length: i32,
    pub frame_presentation_time_length: i32,
}

/// `write_decoder_model_info`: the three `-1` delay/time lengths (5 bits each) around
/// the 32-bit decoding-tick field.
pub fn write_decoder_model_info(wb: &mut WriteBitBuffer, d: &DecoderModelInfo) {
    wb.write_literal(d.encoder_decoder_buffer_delay_length - 1, 5);
    wb.write_unsigned_literal(d.num_units_in_decoding_tick, 32);
    wb.write_literal(d.buffer_removal_time_length - 1, 5);
    wb.write_literal(d.frame_presentation_time_length - 1, 5);
}

/// `write_dec_model_op_parameters`: the decoder/encoder buffer delays (at
/// `buffer_delay_length` bits) + the low-delay-mode flag, for one operating point.
pub fn write_dec_model_op_parameters(
    wb: &mut WriteBitBuffer,
    decoder_buffer_delay: u32,
    encoder_buffer_delay: u32,
    low_delay_mode_flag: bool,
    buffer_delay_length: u32,
) {
    wb.write_unsigned_literal(decoder_buffer_delay, buffer_delay_length);
    wb.write_unsigned_literal(encoder_buffer_delay, buffer_delay_length);
    wb.write_bit(low_delay_mode_flag as u32);
}

// ---- sequence-header OBU (top-level assembly) -----------------------------

const MAX_NUM_OPERATING_POINTS: usize = 32;

/// The full sequence-header OBU state (`av1_write_sequence_header_obu` inputs) —
/// the OBU-level flags + operating points, composed with the sequence-header body
/// and color config.
#[derive(Clone, Debug)]
pub struct SequenceHeaderObu {
    pub profile: i32,
    pub still_picture: bool,
    pub reduced_still_picture_hdr: bool,
    pub timing_info_present: bool,
    pub timing_info: TimingInfoHeader,
    pub decoder_model_info_present_flag: bool,
    pub decoder_model_info: DecoderModelInfo,
    pub display_model_info_present_flag: bool,
    pub operating_points_cnt_minus_1: i32,
    pub operating_point_idc: [i32; MAX_NUM_OPERATING_POINTS],
    pub seq_level_idx: [i32; MAX_NUM_OPERATING_POINTS],
    pub tier: [i32; MAX_NUM_OPERATING_POINTS],
    pub op_decoder_model_param_present: [bool; MAX_NUM_OPERATING_POINTS],
    pub op_display_model_param_present: [bool; MAX_NUM_OPERATING_POINTS],
    pub op_decoder_buffer_delay: [u32; MAX_NUM_OPERATING_POINTS],
    pub op_encoder_buffer_delay: [u32; MAX_NUM_OPERATING_POINTS],
    pub op_low_delay_mode_flag: [bool; MAX_NUM_OPERATING_POINTS],
    pub op_initial_display_delay: [i32; MAX_NUM_OPERATING_POINTS],
    pub seq_header: SequenceHeaderParams,
    pub color_config: ColorConfigParams,
    pub film_grain_params_present: bool,
}

/// `write_bitstream_level`: the sequence level index at `LEVEL_BITS`=5.
fn write_bitstream_level(wb: &mut WriteBitBuffer, seq_level_idx: i32) {
    wb.write_literal(seq_level_idx, 5);
}

/// `av1_write_sequence_header_obu` (`av1/encoder/bitstream.c`): the complete
/// sequence-header OBU payload — profile, still-picture / reduced-header flags,
/// timing + decoder-model info, the operating-points loop (idc, level, tier,
/// per-op decoder/display model params), then the sequence-header body, color
/// config, film-grain-present flag, and the byte-alignment trailing bits.
pub fn write_sequence_header_obu(wb: &mut WriteBitBuffer, s: &SequenceHeaderObu) {
    wb.write_literal(s.profile, 3); // PROFILE_BITS
    wb.write_bit(s.still_picture as u32);
    wb.write_bit(s.reduced_still_picture_hdr as u32);
    if s.reduced_still_picture_hdr {
        write_bitstream_level(wb, s.seq_level_idx[0]);
    } else {
        wb.write_bit(s.timing_info_present as u32);
        if s.timing_info_present {
            write_timing_info_header(wb, &s.timing_info);
            wb.write_bit(s.decoder_model_info_present_flag as u32);
            if s.decoder_model_info_present_flag {
                write_decoder_model_info(wb, &s.decoder_model_info);
            }
        }
        wb.write_bit(s.display_model_info_present_flag as u32);
        wb.write_literal(s.operating_points_cnt_minus_1, 5); // OP_POINTS_CNT_MINUS_1_BITS
        for i in 0..=(s.operating_points_cnt_minus_1 as usize) {
            wb.write_literal(s.operating_point_idc[i], 12); // OP_POINTS_IDC_BITS
            write_bitstream_level(wb, s.seq_level_idx[i]);
            if s.seq_level_idx[i] >= 8 {
                // SEQ_LEVEL_4_0
                wb.write_bit(s.tier[i] as u32);
            }
            if s.decoder_model_info_present_flag {
                wb.write_bit(s.op_decoder_model_param_present[i] as u32);
                if s.op_decoder_model_param_present[i] {
                    write_dec_model_op_parameters(
                        wb,
                        s.op_decoder_buffer_delay[i],
                        s.op_encoder_buffer_delay[i],
                        s.op_low_delay_mode_flag[i],
                        s.decoder_model_info.encoder_decoder_buffer_delay_length as u32,
                    );
                }
            }
            if s.display_model_info_present_flag {
                wb.write_bit(s.op_display_model_param_present[i] as u32);
                if s.op_display_model_param_present[i] {
                    wb.write_literal(s.op_initial_display_delay[i] - 1, 4);
                }
            }
        }
    }
    write_sequence_header(wb, &s.seq_header);
    write_color_config(wb, &s.color_config);
    wb.write_bit(s.film_grain_params_present as u32);
    wb.add_trailing_bits();
}

// ---- frame-header OBU: prefix ---------------------------------------------

/// The bounded prefix state of `write_uncompressed_header_obu` (through the ref
/// order hints — before the per-frame-type frame-size / ref-map signaling).
#[derive(Clone, Debug)]
pub struct FrameHeaderPrefix {
    pub reduced_still_picture_hdr: bool,
    pub show_existing_frame: bool, // encode_show_existing_frame(cm)
    pub existing_fb_idx_to_show: i32,
    pub decoder_model_info_present_flag: bool,
    pub equal_picture_interval: bool,
    pub frame_presentation_time: u32,
    pub frame_presentation_time_length: u32,
    pub frame_id_numbers_present_flag: bool,
    pub frame_id_length: u32,
    pub display_frame_id: i32,
    pub frame_type: i32, // KEY=0 INTER=1 INTRA_ONLY=2 S=3
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub force_screen_content_tools: i32,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: i32,
    pub cur_frame_force_integer_mv: bool,
    pub superres_upscaled_width: i32,
    pub superres_upscaled_height: i32,
    pub max_frame_width: i32,
    pub max_frame_height: i32,
    pub current_frame_id: i32,
    pub enable_order_hint: bool,
    pub order_hint: i32,
    pub order_hint_bits_minus_1: i32,
    pub primary_ref_frame: i32,
    pub buffer_removal_time_present: bool,
    pub operating_points_cnt_minus_1: i32,
    pub op_decoder_model_param_present: [bool; 32],
    pub operating_point_idc: [i32; 32],
    pub temporal_layer_id: i32,
    pub spatial_layer_id: i32,
    pub buffer_removal_times: [u32; 32],
    pub buffer_removal_time_length: u32,
    pub refresh_frame_flags: i32,
    pub ref_frame_map_order_hint: [i32; 8],
}

/// The prefix of `write_uncompressed_header_obu` (`av1/encoder/bitstream.c`): the
/// show-existing-frame path, frame type / show / showable / error-resilient flags,
/// disable-cdf-update, screen-content-tools + integer-mv, current-frame-id, the
/// frame-size-override flag, order hint, primary ref, the buffer-removal-times loop,
/// refresh flags, and the error-resilient ref order hints. Returns
/// `(frame_size_override_flag, returned_early)` — the caller threads the override
/// into the per-frame-type body; `returned_early` mirrors the show-existing-frame
/// early return. The many spec asserts + `aom_internal_error` paths have no byte
/// effect and are omitted (callers pass spec-valid state).
pub fn write_frame_header_prefix(wb: &mut WriteBitBuffer, p: &FrameHeaderPrefix) -> (i32, bool) {
    let sframe = p.frame_type == 3;
    let intra_only = p.frame_type == 0 || p.frame_type == 2;
    let tu_pts = |wb: &mut WriteBitBuffer| {
        wb.write_unsigned_literal(p.frame_presentation_time, p.frame_presentation_time_length);
    };
    if !p.reduced_still_picture_hdr {
        if p.show_existing_frame {
            wb.write_bit(1);
            wb.write_literal(p.existing_fb_idx_to_show, 3);
            if p.decoder_model_info_present_flag && !p.equal_picture_interval {
                tu_pts(wb);
            }
            if p.frame_id_numbers_present_flag {
                wb.write_literal(p.display_frame_id, p.frame_id_length);
            }
            return (0, true);
        }
        wb.write_bit(0);
        wb.write_literal(p.frame_type, 2);
        wb.write_bit(p.show_frame as u32);
        if p.show_frame {
            if p.decoder_model_info_present_flag && !p.equal_picture_interval {
                tu_pts(wb);
            }
        } else {
            wb.write_bit(p.showable_frame as u32);
        }
        if sframe {
            // assert(error_resilient_mode) — no bytes
        } else if !(p.frame_type == 0 && p.show_frame) {
            wb.write_bit(p.error_resilient_mode as u32);
        }
    }
    wb.write_bit(p.disable_cdf_update as u32);
    if p.force_screen_content_tools == 2 {
        wb.write_bit(p.allow_screen_content_tools as u32);
    }
    if p.allow_screen_content_tools && p.force_integer_mv == 2 {
        wb.write_bit(p.cur_frame_force_integer_mv as u32);
    }
    let mut frame_size_override_flag = 0;
    if !p.reduced_still_picture_hdr {
        if p.frame_id_numbers_present_flag {
            wb.write_literal(p.current_frame_id, p.frame_id_length);
        }
        frame_size_override_flag = if sframe {
            1
        } else {
            (p.superres_upscaled_width != p.max_frame_width
                || p.superres_upscaled_height != p.max_frame_height) as i32
        };
        if !sframe {
            wb.write_bit(frame_size_override_flag as u32);
        }
        if p.enable_order_hint {
            wb.write_literal(p.order_hint, (p.order_hint_bits_minus_1 + 1) as u32);
        }
        if !p.error_resilient_mode && !intra_only {
            wb.write_literal(p.primary_ref_frame, 3); // PRIMARY_REF_BITS
        }
    }
    if p.decoder_model_info_present_flag {
        wb.write_bit(p.buffer_removal_time_present as u32);
        if p.buffer_removal_time_present {
            for op in 0..=(p.operating_points_cnt_minus_1 as usize) {
                if p.op_decoder_model_param_present[op] {
                    let idc = p.operating_point_idc[op];
                    if idc == 0
                        || (((idc >> p.temporal_layer_id) & 0x1) != 0
                            && ((idc >> (p.spatial_layer_id + 8)) & 0x1) != 0)
                    {
                        wb.write_unsigned_literal(
                            p.buffer_removal_times[op],
                            p.buffer_removal_time_length,
                        );
                    }
                }
            }
        }
    }
    if (p.frame_type == 0 && !p.show_frame) || p.frame_type == 1 || p.frame_type == 2 {
        wb.write_literal(p.refresh_frame_flags, 8); // REF_FRAMES
    }
    if (!intra_only || p.refresh_frame_flags != 0xff)
        && p.error_resilient_mode
        && p.enable_order_hint
    {
        for &oh in &p.ref_frame_map_order_hint {
            wb.write_literal(oh, (p.order_hint_bits_minus_1 + 1) as u32);
        }
    }
    (frame_size_override_flag, false)
}

// ---- frame size with refs -------------------------------------------------

/// The `write_frame_size_with_refs` state — the current frame's dimensions plus,
/// per reference (LAST..ALTREF), whether the ref buffer is present and its crop /
/// render dimensions, and the superres + fallback `write_frame_size` inputs.
#[derive(Clone, Copy, Debug)]
pub struct FrameSizeWithRefs {
    pub superres_upscaled_width: i32,
    pub superres_upscaled_height: i32,
    pub render_width: i32,
    pub render_height: i32,
    pub ref_cfg_valid: [bool; 7],
    pub ref_y_crop_width: [i32; 7],
    pub ref_y_crop_height: [i32; 7],
    pub ref_render_width: [i32; 7],
    pub ref_render_height: [i32; 7],
    pub enable_superres: bool,
    pub scale_denominator: i32,
    /// Fallback `write_frame_size` (with `frame_size_override` forced true).
    pub frame_size: FrameSizeHeader,
}

/// `write_frame_size_with_refs` (`av1/encoder/bitstream.c`): for each reference in
/// turn, a `found` bit (the ref buffer's crop + render dims equal the current
/// frame's) — on the first match, the superres scale, then stop; if no ref matches,
/// the full frame size (override = 1). `found` is only reassigned when the ref
/// buffer is present, matching the C carry-over across absent refs.
pub fn write_frame_size_with_refs(wb: &mut WriteBitBuffer, w: &FrameSizeWithRefs) {
    let mut found = 0i32;
    for r in 0..7 {
        if w.ref_cfg_valid[r] {
            found = (w.superres_upscaled_width == w.ref_y_crop_width[r]
                && w.superres_upscaled_height == w.ref_y_crop_height[r]) as i32;
            found &= (w.render_width == w.ref_render_width[r]
                && w.render_height == w.ref_render_height[r]) as i32;
        }
        wb.write_bit(found as u32);
        if found != 0 {
            write_superres_scale(wb, w.enable_superres, w.scale_denominator);
            break;
        }
    }
    if found == 0 {
        write_frame_size(wb, &w.frame_size);
    }
}

// ---- frame-header OBU: inter-frame ref signaling --------------------------

/// The INTER/S-frame reference-signaling state (`write_uncompressed_header_obu`).
#[derive(Clone, Debug)]
pub struct InterRefSignaling {
    pub enable_order_hint: bool,
    pub frame_refs_short_signaling: bool,
    /// `get_ref_frame_map_idx(LAST..ALTREF)` (index 0 = LAST, 3 = GOLDEN).
    pub ref_map_idx: [i32; 7],
    pub set_ref_frame_config: bool,
    pub rtc_reference: [i32; 7],
    pub rtc_ref_idx: [i32; 7],
    pub number_spatial_layers: i32,
    pub frame_id_numbers_present_flag: bool,
    pub frame_id_length: u32,
    pub current_frame_id: i32,
    pub ref_frame_id: [i32; 8], // indexed by map_idx
    pub delta_frame_id_length: u32,
}

/// The INTER/S-frame reference-signaling loop from `write_uncompressed_header_obu`:
/// the (order-hint-gated) short-signaling flag with its LAST/GOLDEN map indices, then
/// per reference the 3-bit ref-map index (with the real-time set-ref-frame-config
/// special case) and, when frame ids are present, the modular delta-frame-id. The
/// `internal_error` on an invalid delta has no byte effect and is omitted.
pub fn write_inter_ref_signaling(wb: &mut WriteBitBuffer, s: &InterRefSignaling) {
    if s.enable_order_hint {
        wb.write_bit(s.frame_refs_short_signaling as u32);
    }
    if s.frame_refs_short_signaling {
        wb.write_literal(s.ref_map_idx[0], 3); // LAST (REF_FRAMES_LOG2)
        wb.write_literal(s.ref_map_idx[3], 3); // GOLDEN
    }
    let mut first_ref_map_idx = -1i32; // INVALID_IDX
    if s.set_ref_frame_config {
        for r in 0..7 {
            if s.rtc_reference[r] == 1 {
                first_ref_map_idx = s.rtc_ref_idx[r];
                break;
            }
        }
    }
    for r in 0..7 {
        if !s.frame_refs_short_signaling {
            if s.set_ref_frame_config
                && first_ref_map_idx != -1
                && s.number_spatial_layers == 1
                && !s.enable_order_hint
            {
                let map_idx = if s.rtc_reference[r] != 0 { s.ref_map_idx[r] } else { first_ref_map_idx };
                wb.write_literal(map_idx, 3);
            } else {
                wb.write_literal(s.ref_map_idx[r], 3);
            }
        }
        if s.frame_id_numbers_present_flag {
            let i = s.ref_map_idx[r] as usize;
            let m = 1i32 << s.frame_id_length;
            let delta = ((s.current_frame_id - s.ref_frame_id[i] + m) % m) - 1;
            wb.write_literal(delta, s.delta_frame_id_length);
        }
    }
}

// ---- frame-header OBU: connective flags -----------------------------------

/// The refresh-frame-context bit (`write_uncompressed_header_obu`, just before
/// `write_tile_info`): written only when back-adaptation might apply (not reduced
/// still picture and CDF update not disabled). `refresh_frame_context_disabled` is
/// `features->refresh_frame_context == REFRESH_FRAME_CONTEXT_DISABLED`.
pub fn write_refresh_frame_context(
    wb: &mut WriteBitBuffer,
    reduced_still_picture_hdr: bool,
    disable_cdf_update: bool,
    refresh_frame_context_disabled: bool,
) {
    let might_bwd_adapt = !reduced_still_picture_hdr && !disable_cdf_update;
    if might_bwd_adapt {
        wb.write_bit(refresh_frame_context_disabled as u32);
    }
}

/// The frame-header trailing flags (`write_uncompressed_header_obu`, after the TX
/// mode): the reference-mode SELECT bit (non-intra only), the skip-mode flag (when
/// allowed), the warped-motion flag (when the frame might allow it), and the
/// reduced-tx-set flag. `intra_only` = `frame_is_intra_only(cm)`.
#[allow(clippy::too_many_arguments)]
pub fn write_frame_header_trailing_flags(
    wb: &mut WriteBitBuffer,
    intra_only: bool,
    reference_mode_select: bool,
    skip_mode_allowed: bool,
    skip_mode_flag: bool,
    might_allow_warped_motion: bool,
    allow_warped_motion: bool,
    reduced_tx_set_used: bool,
) {
    if !intra_only {
        wb.write_bit(reference_mode_select as u32);
    }
    if skip_mode_allowed {
        wb.write_bit(skip_mode_flag as u32);
    }
    if might_allow_warped_motion {
        wb.write_bit(allow_warped_motion as u32);
    }
    wb.write_bit(reduced_tx_set_used as u32);
}
