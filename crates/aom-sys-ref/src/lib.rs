//! FFI bindings to the pinned reference **C libaom v3.14.1**.
//!
//! This crate exists ONLY to serve as the differential oracle. Nothing in the
//! shipping library links against it. Symbols are declared as needed, per
//! module, as we bring differential harnesses online.

pub type Txfm1dFn =
    unsafe extern "C" fn(input: *const i32, output: *mut i32, cos_bit: i8, stage_range: *const i8);

extern "C" {
    // Runtime CPU detection: populates the SIMD dispatch pointers (e.g.
    // av1_round_shift_array). Some `_c` entry points internally call these
    // dispatched functions, so this must run once before any oracle call.
    fn av1_rtcd();
    fn aom_dsp_rtcd();
    fn aom_scale_rtcd();
}

// Entropy-coder shim (shim/entropy_shim.c) — opaque od_ec_enc / od_ec_dec.
extern "C" {
    fn shim_enc_new(size: u32) -> *mut core::ffi::c_void;
    fn shim_enc_bool(e: *mut core::ffi::c_void, val: i32, f: u32);
    fn shim_enc_cdf(e: *mut core::ffi::c_void, s: i32, icdf: *const u16, nsyms: i32);
    fn shim_enc_done(e: *mut core::ffi::c_void, nbytes: *mut u32) -> *const u8;
    fn shim_enc_free(e: *mut core::ffi::c_void);
    fn shim_dec_new(buf: *const u8, sz: u32) -> *mut core::ffi::c_void;
    fn shim_dec_bool(d: *mut core::ffi::c_void, f: u32) -> i32;
    fn shim_dec_cdf(d: *mut core::ffi::c_void, icdf: *const u16, nsyms: i32) -> i32;
    fn shim_dec_free(d: *mut core::ffi::c_void);
    fn shim_update_cdf(cdf: *mut u16, val: i32, nsymbs: i32);
    fn shim_adapt_encode(
        syms: *const i32, n: i32, cdf_init: *const u16, nsymbs: i32, out: *mut u8, out_cap: u32,
    ) -> u32;
    fn shim_adapt_decode(
        buf: *const u8, sz: u32, n: i32, cdf_init: *const u16, nsymbs: i32, out_syms: *mut i32,
    );
}

// convolve_shim.c — av1_convolve_{x,y}_sr (EIGHTTAP_REGULAR).
extern "C" {
    fn shim_convolve_x_sr(src: *const u8, ss: i32, dst: *mut u8, ds: i32, w: i32, h: i32, subpel: i32, ftype: i32);
    fn shim_convolve_y_sr(src: *const u8, ss: i32, dst: *mut u8, ds: i32, w: i32, h: i32, subpel: i32, ftype: i32);
    fn shim_convolve_2d_sr(src: *const u8, ss: i32, dst: *mut u8, ds: i32, w: i32, h: i32, spx: i32, spy: i32, ftype: i32);
}

/// Reference `av1_convolve_2d_sr_c`.
#[allow(clippy::too_many_arguments)]
pub fn ref_convolve_2d_sr(src: &[u8], src_off: usize, ss: usize, w: usize, h: usize, spx: usize, spy: usize, ftype: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    unsafe { shim_convolve_2d_sr(src.as_ptr().add(src_off), ss as i32, dst.as_mut_ptr(), w as i32, w as i32, h as i32, spx as i32, spy as i32, ftype as i32) }
    dst
}

/// Reference `av1_convolve_x_sr_c`. `src` points at the interior origin.
pub fn ref_convolve_x_sr(src: &[u8], src_off: usize, ss: usize, w: usize, h: usize, subpel: usize, ftype: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    unsafe { shim_convolve_x_sr(src.as_ptr().add(src_off), ss as i32, dst.as_mut_ptr(), w as i32, w as i32, h as i32, subpel as i32, ftype as i32) }
    dst
}

/// Reference `av1_convolve_y_sr_c`.
pub fn ref_convolve_y_sr(src: &[u8], src_off: usize, ss: usize, w: usize, h: usize, subpel: usize, ftype: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    unsafe { shim_convolve_y_sr(src.as_ptr().add(src_off), ss as i32, dst.as_mut_ptr(), w as i32, w as i32, h as i32, subpel as i32, ftype as i32) }
    dst
}

// aom_dsp/avg.c — Hadamard transform + SATD.
extern "C" {
    pub fn aom_hadamard_4x4_c(src: *const i16, stride: isize, coeff: *mut i32);
    pub fn aom_hadamard_8x8_c(src: *const i16, stride: isize, coeff: *mut i32);
    pub fn aom_hadamard_16x16_c(src: *const i16, stride: isize, coeff: *mut i32);
    pub fn aom_hadamard_32x32_c(src: *const i16, stride: isize, coeff: *mut i32);
    pub fn aom_highbd_hadamard_8x8_c(src: *const i16, stride: isize, coeff: *mut i32);
    pub fn aom_highbd_hadamard_16x16_c(src: *const i16, stride: isize, coeff: *mut i32);
    pub fn aom_highbd_hadamard_32x32_c(src: *const i16, stride: isize, coeff: *mut i32);
    pub fn aom_satd_c(coeff: *const i32, length: i32) -> i32;
}

/// Reference `aom_highbd_hadamard_<n>x<n>_c` for `n` in {8,16,32}.
pub fn ref_highbd_hadamard(n: usize, src: &[i16], stride: usize) -> Vec<i32> {
    let mut coeff = vec![0i32; n * n];
    unsafe {
        match n {
            8 => aom_highbd_hadamard_8x8_c(src.as_ptr(), stride as isize, coeff.as_mut_ptr()),
            16 => aom_highbd_hadamard_16x16_c(src.as_ptr(), stride as isize, coeff.as_mut_ptr()),
            32 => aom_highbd_hadamard_32x32_c(src.as_ptr(), stride as isize, coeff.as_mut_ptr()),
            _ => unreachable!(),
        }
    }
    coeff
}

/// Reference `aom_hadamard_<n>x<n>_c` for `n` in {4,8,16,32}; returns `n*n` coeffs.
pub fn ref_hadamard(n: usize, src: &[i16], stride: usize) -> Vec<i32> {
    let mut coeff = vec![0i32; n * n];
    unsafe {
        match n {
            4 => aom_hadamard_4x4_c(src.as_ptr(), stride as isize, coeff.as_mut_ptr()),
            8 => aom_hadamard_8x8_c(src.as_ptr(), stride as isize, coeff.as_mut_ptr()),
            16 => aom_hadamard_16x16_c(src.as_ptr(), stride as isize, coeff.as_mut_ptr()),
            32 => aom_hadamard_32x32_c(src.as_ptr(), stride as isize, coeff.as_mut_ptr()),
            _ => unreachable!(),
        }
    }
    coeff
}

/// Reference `aom_satd_c`.
pub fn ref_satd(coeff: &[i32]) -> i32 {
    unsafe { aom_satd_c(coeff.as_ptr(), coeff.len() as i32) }
}

// av1/common/cdef_block.c — CDEF direction search + filter block.
extern "C" {
    pub fn cdef_find_dir_c(img: *const u16, stride: i32, var: *mut i32, coeff_shift: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_cdef_filter8(
        variant: i32, dst: *mut u8, dstride: i32, in_: *const u16, pri: i32, sec: i32, dir: i32,
        prid: i32, secd: i32, cshift: i32, bw: i32, bh: i32,
    );
}

/// Reference `cdef_filter_8_<variant>_c`. `in_buf` is u16 with stride
/// CDEF_BSTRIDE; `in_off` is the block origin. Returns the `bw*bh` result.
#[allow(clippy::too_many_arguments)]
pub fn ref_cdef_filter8(
    variant: i32, in_buf: &[u16], in_off: usize, pri: i32, sec: i32, dir: i32, prid: i32,
    secd: i32, cshift: i32, bw: usize, bh: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; bw * bh];
    unsafe {
        shim_cdef_filter8(
            variant, dst.as_mut_ptr(), bw as i32, in_buf.as_ptr().add(in_off), pri, sec, dir,
            prid, secd, cshift, bw as i32, bh as i32,
        )
    }
    dst
}

/// Reference `cdef_find_dir_c`; returns (best_dir, var).
pub fn ref_cdef_find_dir(img: &[u16], stride: usize, coeff_shift: i32) -> (i32, i32) {
    let mut var = 0i32;
    let dir = unsafe { cdef_find_dir_c(img.as_ptr(), stride as i32, &mut var, coeff_shift) };
    (dir, var)
}

// sadvar_shim.c — SAD / variance / sub-pixel variance dispatch (22 sizes).
extern "C" {
    fn shim_sad(idx: i32, s: *const u8, ss: i32, r: *const u8, rs: i32) -> u32;
    fn shim_variance(idx: i32, a: *const u8, as_: i32, b: *const u8, bs: i32, sse: *mut u32) -> u32;
    fn shim_subpel_var(idx: i32, a: *const u8, as_: i32, xo: i32, yo: i32, b: *const u8, bs: i32, sse: *mut u32) -> u32;
}

/// Reference `aom_sad<W>x<H>_c` for size index `idx`.
pub fn ref_sad(idx: usize, s: &[u8], ss: usize, r: &[u8], rs: usize) -> u32 {
    unsafe { shim_sad(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32) }
}

extern "C" {
    fn shim_sad_avg(i: i32, s: *const u8, ss: i32, r: *const u8, rs: i32, sp: *const u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_masked_sad(i: i32, s: *const u8, ss: i32, r: *const u8, rs: i32, sp: *const u8, m: *const u8, ms: i32, inv: i32) -> u32;
    fn shim_obmc_sad(i: i32, r: *const u8, rs: i32, ws: *const i32, m: *const i32) -> u32;
    fn shim_sse(a: *const u8, as_: i32, b: *const u8, bs: i32, w: i32, h: i32) -> i64;
    fn shim_hbd_sse(a: *const u16, as_: i32, b: *const u16, bs: i32, w: i32, h: i32) -> i64;
}

/// Reference `aom_sse_c` (sum of squared errors, generic w×h).
pub fn ref_sse(a: &[u8], as_: usize, b: &[u8], bs: usize, w: usize, h: usize) -> i64 {
    unsafe { shim_sse(a.as_ptr(), as_ as i32, b.as_ptr(), bs as i32, w as i32, h as i32) }
}

/// Reference `aom_highbd_sse_c`.
pub fn ref_hbd_sse(a: &[u16], as_: usize, b: &[u16], bs: usize, w: usize, h: usize) -> i64 {
    unsafe { shim_hbd_sse(a.as_ptr(), as_ as i32, b.as_ptr(), bs as i32, w as i32, h as i32) }
}

extern "C" {
    fn av1_block_error_c(coeff: *const i32, dqcoeff: *const i32, block_size: isize, ssz: *mut i64) -> i64;
    fn av1_highbd_block_error_c(coeff: *const i32, dqcoeff: *const i32, block_size: isize, ssz: *mut i64, bd: i32) -> i64;
    fn aom_subtract_block_c(rows: i32, cols: i32, diff: *mut i16, diff_stride: isize, src: *const u8, src_stride: isize, pred: *const u8, pred_stride: isize);
    fn shim_highbd_subtract_block(rows: i32, cols: i32, diff: *mut i16, diff_stride: i32, src: *const u16, src_stride: i32, pred: *const u16, pred_stride: i32);
    fn shim_block_error_qm(coeff: *const i32, dqcoeff: *const i32, block_size: isize, qmatrix: *const u8, scan: *const i16, ssz: *mut i64, bd: i32) -> i64;
    fn av1_model_rd_from_var_lapndz(var: i64, n_log2: u32, qstep: u32, rate: *mut i32, dist: *mut i64);
    fn aom_sum_squares_i16_c(src: *const i16, n: u32) -> u64;
    fn aom_sum_squares_2d_i16_c(src: *const i16, src_stride: i32, width: i32, height: i32) -> u64;
    fn aom_vector_var_c(reff: *const i16, src: *const i16, bwl: i32) -> i32;
    fn shim_wb_apply(data: *const u32, bits: *const i32, kind: *const i32, n: i32, out: *mut u8) -> u32;
    fn aom_uleb_size_in_bytes(value: u64) -> usize;
    fn aom_uleb_encode(value: u64, available: usize, coded_value: *mut u8, coded_size: *mut usize) -> i32;
    fn aom_uleb_decode(buffer: *const u8, available: usize, value: *mut u64, length: *mut usize) -> i32;
    fn shim_write_obu_header(obu_type: i32, has_nonzero_op: i32, is_layer_specific: i32, obu_extension: i32, dst: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_quantization(base_qindex: i32, y_dc: i32, u_dc: i32, u_ac: i32, v_dc: i32, v_ac: i32, using_qm: i32, qm_y: i32, qm_u: i32, qm_v: i32, num_planes: i32, separate_uv: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_loopfilter(allow_intrabc: i32, fl0: i32, fl1: i32, flu: i32, flv: i32, sharpness: i32, mode_ref_enabled: i32, mode_ref_update: i32, ref_deltas: *const i8, mode_deltas: *const i8, last_ref: *const i8, last_mode: *const i8, num_planes: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_cdef(enable_cdef: i32, allow_intrabc: i32, damping: i32, cdef_bits: i32, nb: i32, y: *const i32, uv: *const i32, num_planes: i32, out: *mut u8) -> u32;
    fn shim_encode_segmentation(enabled: i32, has_primary_ref: i32, update_map: i32, temporal_update: i32, update_data: i32, feature_mask: *const u32, feature_data: *const i32, out: *mut u8) -> u32;
    fn shim_write_frame_interp_filter(filter: i32, out: *mut u8) -> u32;
    fn shim_write_superres_scale(enable_superres: i32, denom: i32, out: *mut u8) -> u32;
    fn shim_write_render_size(scaling_active: i32, rw: i32, rh: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_frame_size(frame_size_override: i32, num_bits_width: i32, num_bits_height: i32, up_w: i32, up_h: i32, enable_superres: i32, denom: i32, scaling_active: i32, rw: i32, rh: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_tile_info(mi_cols: i32, mi_rows: i32, mib_size_log2: i32, uniform_spacing: i32, log2_cols: i32, min_log2_cols: i32, max_log2_cols: i32, log2_rows: i32, min_log2_rows: i32, max_log2_rows: i32, cols: i32, rows: i32, col_start_sb: *const i32, row_start_sb: *const i32, max_width_sb: i32, max_height_sb: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_restoration_mode(enable_restoration: i32, allow_intrabc: i32, frame_restoration_type: *const i32, sb_size_128: i32, restoration_unit_size: *const i32, ssx: i32, ssy: i32, num_planes: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_delta_q_params(base_qindex: i32, delta_q_present: i32, delta_q_res: i32, allow_intrabc: i32, delta_lf_present: i32, delta_lf_res: i32, delta_lf_multi: i32, out: *mut u8) -> u32;
    fn shim_write_tx_mode(coded_lossless: i32, tx_mode_select: i32, out: *mut u8) -> u32;
    fn shim_write_film_grain_params(s: *const i32, spy: *const i32, spcb: *const i32, spcr: *const i32, ary: *const i32, arcb: *const i32, arcr: *const i32, out: *mut u8) -> u32;
}

/// Reference `write_film_grain_params` (transcribed control flow over the real aom_wb).
/// `s` packs the 24 scalars in the order the shim reads them; the point/coeff arrays
/// are passed flat.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_film_grain_params(s: &[i32; 24], spy: &[i32; 28], spcb: &[i32; 20], spcr: &[i32; 20], ary: &[i32; 24], arcb: &[i32; 25], arcr: &[i32; 25]) -> Vec<u8> {
    let mut out = vec![0u8; 256];
    let n = unsafe { shim_write_film_grain_params(s.as_ptr(), spy.as_ptr(), spcb.as_ptr(), spcr.as_ptr(), ary.as_ptr(), arcb.as_ptr(), arcr.as_ptr(), out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_delta_q_params` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_delta_q_params(base_qindex: i32, delta_q_present: bool, delta_q_res: i32, allow_intrabc: bool, delta_lf_present: bool, delta_lf_res: i32, delta_lf_multi: bool) -> Vec<u8> {
    let mut out = vec![0u8; 8];
    let n = unsafe { shim_write_delta_q_params(base_qindex, delta_q_present as i32, delta_q_res, allow_intrabc as i32, delta_lf_present as i32, delta_lf_res, delta_lf_multi as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_tx_mode`.
pub fn ref_write_tx_mode(coded_lossless: bool, tx_mode_select: bool) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe { shim_write_tx_mode(coded_lossless as i32, tx_mode_select as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `encode_restoration_mode` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_restoration_mode(enable_restoration: bool, allow_intrabc: bool, frame_restoration_type: &[i32; 3], sb_size_128: bool, restoration_unit_size: &[i32; 3], ssx: i32, ssy: i32, num_planes: usize) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe { shim_encode_restoration_mode(enable_restoration as i32, allow_intrabc as i32, frame_restoration_type.as_ptr(), sb_size_128 as i32, restoration_unit_size.as_ptr(), ssx, ssy, num_planes as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_tile_info` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_tile_info(mi_cols: i32, mi_rows: i32, mib_size_log2: u32, uniform_spacing: bool, log2_cols: i32, min_log2_cols: i32, max_log2_cols: i32, log2_rows: i32, min_log2_rows: i32, max_log2_rows: i32, cols: usize, rows: usize, col_start_sb: &[i32; 65], row_start_sb: &[i32; 65], max_width_sb: i32, max_height_sb: i32) -> Vec<u8> {
    let mut out = vec![0u8; 128];
    let n = unsafe {
        shim_write_tile_info(mi_cols, mi_rows, mib_size_log2 as i32, uniform_spacing as i32, log2_cols, min_log2_cols, max_log2_cols, log2_rows, min_log2_rows, max_log2_rows, cols as i32, rows as i32, col_start_sb.as_ptr(), row_start_sb.as_ptr(), max_width_sb, max_height_sb, out.as_mut_ptr())
    };
    out.truncate(n as usize);
    out
}

/// Reference `encode_segmentation` (transcribed control flow over the real aom_wb + seg tables).
pub fn ref_encode_segmentation(enabled: bool, has_primary_ref: bool, update_map: bool, temporal_update: bool, update_data: bool, feature_mask: &[u32; 8], feature_data: &[[i32; 8]; 8]) -> Vec<u8> {
    let flat: Vec<i32> = feature_data.iter().flatten().copied().collect();
    let mut out = vec![0u8; 64];
    let n = unsafe { shim_encode_segmentation(enabled as i32, has_primary_ref as i32, update_map as i32, temporal_update as i32, update_data as i32, feature_mask.as_ptr(), flat.as_ptr(), out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_frame_interp_filter`.
pub fn ref_write_frame_interp_filter(filter: i32) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe { shim_write_frame_interp_filter(filter, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_superres_scale`.
pub fn ref_write_superres_scale(enable_superres: bool, denom: i32) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe { shim_write_superres_scale(enable_superres as i32, denom, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_render_size`.
pub fn ref_write_render_size(scaling_active: bool, rw: i32, rh: i32) -> Vec<u8> {
    let mut out = vec![0u8; 8];
    let n = unsafe { shim_write_render_size(scaling_active as i32, rw, rh, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_frame_size`.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_frame_size(frame_size_override: bool, num_bits_width: u32, num_bits_height: u32, up_w: i32, up_h: i32, enable_superres: bool, denom: i32, scaling_active: bool, rw: i32, rh: i32) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe { shim_write_frame_size(frame_size_override as i32, num_bits_width as i32, num_bits_height as i32, up_w, up_h, enable_superres as i32, denom, scaling_active as i32, rw, rh, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `encode_cdef` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_cdef(enable_cdef: bool, allow_intrabc: bool, damping: i32, cdef_bits: i32, nb: usize, y: &[i32; 8], uv: &[i32; 8], num_planes: usize) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe { shim_encode_cdef(enable_cdef as i32, allow_intrabc as i32, damping, cdef_bits, nb as i32, y.as_ptr(), uv.as_ptr(), num_planes as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `encode_loopfilter` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_loopfilter(allow_intrabc: bool, filter_level: [i32; 2], flu: i32, flv: i32, sharpness: i32, mode_ref_enabled: bool, mode_ref_update: bool, ref_deltas: &[i8; 8], mode_deltas: &[i8; 2], last_ref: &[i8; 8], last_mode: &[i8; 2], num_planes: usize) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe { shim_encode_loopfilter(allow_intrabc as i32, filter_level[0], filter_level[1], flu, flv, sharpness, mode_ref_enabled as i32, mode_ref_update as i32, ref_deltas.as_ptr(), mode_deltas.as_ptr(), last_ref.as_ptr(), last_mode.as_ptr(), num_planes as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `encode_quantization` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_quantization(base_qindex: i32, y_dc: i32, u_dc: i32, u_ac: i32, v_dc: i32, v_ac: i32, using_qm: bool, qm_y: i32, qm_u: i32, qm_v: i32, num_planes: usize, separate_uv: bool) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe { shim_encode_quantization(base_qindex, y_dc, u_dc, u_ac, v_dc, v_ac, using_qm as i32, qm_y, qm_u, qm_v, num_planes as i32, separate_uv as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `av1_write_obu_header` byte output (transcribed shim). Returns the header bytes.
pub fn ref_write_obu_header(obu_type: u32, has_nonzero_op: bool, is_layer_specific: bool, obu_extension: u8) -> Vec<u8> {
    let mut dst = [0u8; 2];
    let n = unsafe { shim_write_obu_header(obu_type as i32, has_nonzero_op as i32, is_layer_specific as i32, obu_extension as i32, dst.as_mut_ptr()) };
    dst[..n as usize].to_vec()
}

/// Reference `aom_uleb_size_in_bytes`.
pub fn ref_uleb_size_in_bytes(value: u64) -> usize {
    unsafe { aom_uleb_size_in_bytes(value) }
}

/// Reference `aom_uleb_encode`. Returns `Some(bytes)` on success (rc 0), `None` on failure.
pub fn ref_uleb_encode(value: u64, available: usize) -> Option<Vec<u8>> {
    let mut coded = vec![0u8; 16];
    let mut coded_size = 0usize;
    let rc = unsafe { aom_uleb_encode(value, available, coded.as_mut_ptr(), &mut coded_size) };
    if rc == 0 {
        coded.truncate(coded_size);
        Some(coded)
    } else {
        None
    }
}

/// Reference `aom_uleb_decode`. Returns `Some((value, length))` or `None`.
pub fn ref_uleb_decode(buffer: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut length = 0usize;
    let rc = unsafe { aom_uleb_decode(buffer.as_ptr(), buffer.len(), &mut value, &mut length) };
    if rc == 0 { Some((value, length)) } else { None }
}

/// Reference `aom_write_bit_buffer`: apply a sequence of literal ops (kind 0 =
/// signed literal, 1 = unsigned literal, 2 = inv-signed literal) and return the
/// produced bytes.
pub fn ref_wb_apply(data: &[u32], bits: &[i32], kind: &[i32]) -> Vec<u8> {
    let n = data.len();
    let mut out = vec![0u8; 1 << 16];
    let written = unsafe { shim_wb_apply(data.as_ptr(), bits.as_ptr(), kind.as_ptr(), n as i32, out.as_mut_ptr()) };
    out.truncate(written as usize);
    out
}

/// Reference `aom_vector_var_c`.
pub fn ref_vector_var(reff: &[i16], src: &[i16], bwl: i32) -> i32 {
    unsafe { aom_vector_var_c(reff.as_ptr(), src.as_ptr(), bwl) }
}

/// Reference `aom_sum_squares_i16_c` (sum of squared i16 values).
pub fn ref_sum_squares_i16(src: &[i16]) -> u64 {
    unsafe { aom_sum_squares_i16_c(src.as_ptr(), src.len() as u32) }
}

/// Reference `aom_sum_squares_2d_i16_c` (2-D strided residual energy).
pub fn ref_sum_squares_2d_i16(src: &[i16], src_stride: usize, width: usize, height: usize) -> u64 {
    unsafe { aom_sum_squares_2d_i16_c(src.as_ptr(), src_stride as i32, width as i32, height as i32) }
}

/// Reference `av1_model_rd_from_var_lapndz` (Laplacian RD model). Returns (rate, dist).
pub fn ref_model_rd_from_var_lapndz(var: i64, n_log2: u32, qstep: u32) -> (i32, i64) {
    let mut rate = 0i32;
    let mut dist = 0i64;
    unsafe { av1_model_rd_from_var_lapndz(var, n_log2, qstep, &mut rate, &mut dist) }
    (rate, dist)
}

/// Reference `av1_block_error_qm` (QM-weighted transform-domain distortion; the
/// static inline is transcribed in sadvar_shim.c). Returns (error, ssz).
pub fn ref_block_error_qm(coeff: &[i32], dqcoeff: &[i32], qmatrix: &[u8], scan: &[i16], bd: u8) -> (i64, i64) {
    let mut ssz = 0i64;
    let err = unsafe { shim_block_error_qm(coeff.as_ptr(), dqcoeff.as_ptr(), coeff.len() as isize, qmatrix.as_ptr(), scan.as_ptr(), &mut ssz, bd as i32) };
    (err, ssz)
}

/// Reference `aom_subtract_block_c` (residual = src - pred). Writes `diff`.
#[allow(clippy::too_many_arguments)]
pub fn ref_subtract_block(rows: usize, cols: usize, diff: &mut [i16], diff_stride: usize, src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) {
    unsafe { aom_subtract_block_c(rows as i32, cols as i32, diff.as_mut_ptr(), diff_stride as isize, src.as_ptr(), src_stride as isize, pred.as_ptr(), pred_stride as isize) }
}

/// Reference `aom_highbd_subtract_block_c` (residual = src - pred, u16).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_subtract_block(rows: usize, cols: usize, diff: &mut [i16], diff_stride: usize, src: &[u16], src_stride: usize, pred: &[u16], pred_stride: usize) {
    unsafe { shim_highbd_subtract_block(rows as i32, cols as i32, diff.as_mut_ptr(), diff_stride as i32, src.as_ptr(), src_stride as i32, pred.as_ptr(), pred_stride as i32) }
}

/// Reference `av1_block_error_c` (transform-domain distortion). Returns (error, ssz).
pub fn ref_block_error(coeff: &[i32], dqcoeff: &[i32]) -> (i64, i64) {
    let mut ssz = 0i64;
    let err = unsafe { av1_block_error_c(coeff.as_ptr(), dqcoeff.as_ptr(), coeff.len() as isize, &mut ssz) };
    (err, ssz)
}

/// Reference `av1_highbd_block_error_c` (highbd transform-domain distortion).
pub fn ref_highbd_block_error(coeff: &[i32], dqcoeff: &[i32], bd: u8) -> (i64, i64) {
    let mut ssz = 0i64;
    let err = unsafe { av1_highbd_block_error_c(coeff.as_ptr(), dqcoeff.as_ptr(), coeff.len() as isize, &mut ssz, bd as i32) };
    (err, ssz)
}

/// Reference `aom_obmc_sad<W>x<H>_c` (overlapped block motion-comp SAD).
pub fn ref_obmc_sad(idx: usize, r: &[u8], rs: usize, ws: &[i32], m: &[i32]) -> u32 {
    unsafe { shim_obmc_sad(idx as i32, r.as_ptr(), rs as i32, ws.as_ptr(), m.as_ptr()) }
}

/// Reference `aom_masked_sad<W>x<H>_c` (wedge / diff-weighted compound RD).
#[allow(clippy::too_many_arguments)]
pub fn ref_masked_sad(idx: usize, s: &[u8], ss: usize, r: &[u8], rs: usize, sp: &[u8], m: &[u8], ms: usize, inv: bool) -> u32 {
    unsafe { shim_masked_sad(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32, sp.as_ptr(), m.as_ptr(), ms as i32, inv as i32) }
}

/// Reference `aom_sad<W>x<H>_avg_c` (compound-prediction SAD) for size index `idx`.
///
/// `aom_sad*_avg_c` internally calls the RTCD-dispatched `aom_comp_avg_pred`
/// (a null fn-pointer until `aom_dsp_rtcd()` runs), so init RTCD first.
pub fn ref_sad_avg(idx: usize, s: &[u8], ss: usize, r: &[u8], rs: usize, sp: &[u8]) -> u32 {
    ref_init();
    unsafe { shim_sad_avg(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32, sp.as_ptr()) }
}

extern "C" {
    fn shim_sad_prod64(s: *const u8, ss: i32, r: *const u8, rs: i32) -> u32;
    fn shim_sad_prod128(s: *const u8, ss: i32, r: *const u8, rs: i32) -> u32;
}

/// Production-dispatch SAD (post-RTCD → C's AVX2 path) — the true perf-gate
/// baseline. `w` must be 64 or 128.
pub fn prod_sad(w: usize, s: &[u8], ss: usize, r: &[u8], rs: usize) -> u32 {
    ref_init();
    unsafe {
        match w {
            64 => shim_sad_prod64(s.as_ptr(), ss as i32, r.as_ptr(), rs as i32),
            128 => shim_sad_prod128(s.as_ptr(), ss as i32, r.as_ptr(), rs as i32),
            _ => unreachable!(),
        }
    }
}

/// Reference `aom_variance<W>x<H>_c`; returns (variance, sse).
pub fn ref_variance(idx: usize, a: &[u8], as_: usize, b: &[u8], bs: usize) -> (u32, u32) {
    let mut sse = 0u32;
    let v = unsafe { shim_variance(idx as i32, a.as_ptr(), as_ as i32, b.as_ptr(), bs as i32, &mut sse) };
    (v, sse)
}

/// Reference `aom_sub_pixel_variance<W>x<H>_c`; returns (variance, sse).
#[allow(clippy::too_many_arguments)]
pub fn ref_subpel_var(idx: usize, a: &[u8], as_: usize, xo: usize, yo: usize, b: &[u8], bs: usize) -> (u32, u32) {
    let mut sse = 0u32;
    let v = unsafe {
        shim_subpel_var(idx as i32, a.as_ptr(), as_ as i32, xo as i32, yo as i32, b.as_ptr(), bs as i32, &mut sse)
    };
    (v, sse)
}

// hbd_sadvar_shim.c — highbd SAD / variance (CONVERT_TO_BYTEPTR internally).
extern "C" {
    fn shim_hbd_sad(i: i32, s: *const u16, ss: i32, r: *const u16, rs: i32) -> u32;
    fn shim_hbd_var(i: i32, bd: i32, a: *const u16, as_: i32, b: *const u16, bs: i32, sse: *mut u32) -> u32;
    fn shim_hbd_subpel_var(i: i32, bd: i32, a: *const u16, as_: i32, xo: i32, yo: i32, b: *const u16, bs: i32, sse: *mut u32) -> u32;
    fn shim_hbd_sad_avg(i: i32, s: *const u16, ss: i32, r: *const u16, rs: i32, p: *const u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_hbd_masked_sad(i: i32, s: *const u16, ss: i32, r: *const u16, rs: i32, p: *const u16, m: *const u8, ms: i32, inv: i32) -> u32;
    fn shim_hbd_obmc_sad(i: i32, r: *const u16, rs: i32, ws: *const i32, m: *const i32) -> u32;
}

/// Reference `aom_highbd_obmc_sad<W>x<H>_c`.
pub fn ref_hbd_obmc_sad(idx: usize, r: &[u16], rs: usize, ws: &[i32], m: &[i32]) -> u32 {
    unsafe { shim_hbd_obmc_sad(idx as i32, r.as_ptr(), rs as i32, ws.as_ptr(), m.as_ptr()) }
}

/// Reference `aom_highbd_masked_sad<W>x<H>_c` (highbd wedge / compound SAD).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_masked_sad(idx: usize, s: &[u16], ss: usize, r: &[u16], rs: usize, p: &[u16], m: &[u8], ms: usize, inv: bool) -> u32 {
    unsafe { shim_hbd_masked_sad(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32, p.as_ptr(), m.as_ptr(), ms as i32, inv as i32) }
}

/// Reference `aom_highbd_sad<W>x<H>_avg_c` (highbd compound-prediction SAD).
/// Calls `ref_init()` first (invokes RTCD-dispatched `aom_highbd_comp_avg_pred`).
pub fn ref_hbd_sad_avg(idx: usize, s: &[u16], ss: usize, r: &[u16], rs: usize, p: &[u16]) -> u32 {
    ref_init();
    unsafe { shim_hbd_sad_avg(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32, p.as_ptr()) }
}

/// Reference `aom_highbd_sad<W>x<H>_c` for size index `idx`.
pub fn ref_hbd_sad(idx: usize, s: &[u16], ss: usize, r: &[u16], rs: usize) -> u32 {
    unsafe { shim_hbd_sad(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32) }
}

/// Reference `aom_highbd_<bd>_variance<W>x<H>_c`; returns (variance, sse).
pub fn ref_hbd_variance(idx: usize, bd: u8, a: &[u16], as_: usize, b: &[u16], bs: usize) -> (u32, u32) {
    let mut sse = 0u32;
    let v = unsafe {
        shim_hbd_var(idx as i32, bd as i32, a.as_ptr(), as_ as i32, b.as_ptr(), bs as i32, &mut sse)
    };
    (v, sse)
}

/// Reference `aom_highbd_<bd>_sub_pixel_variance<W>x<H>_c`; returns (variance, sse).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_subpel_var(idx: usize, bd: u8, a: &[u16], as_: usize, xo: usize, yo: usize, b: &[u16], bs: usize) -> (u32, u32) {
    let mut sse = 0u32;
    let v = unsafe {
        shim_hbd_subpel_var(idx as i32, bd as i32, a.as_ptr(), as_ as i32, xo as i32, yo as i32, b.as_ptr(), bs as i32, &mut sse)
    };
    (v, sse)
}

// hbd_lpf_shim.c — highbd deblocking edge filters.
extern "C" {
    fn shim_hbd_lpf(dir: i32, width: i32, s: *mut u16, p: i32, bl: *const u8, li: *const u8, th: *const u8, bd: i32);
}

/// Apply a reference highbd loop filter in place. `dir`: 'h'/'v'.
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_lpf(dir: u8, width: u32, buf: &mut [u16], center: usize, pitch: usize, bl: u8, li: u8, th: u8, bd: i32) {
    let (b, l, t) = ([bl], [li], [th]);
    let d = if dir == b'h' { 0 } else { 1 };
    unsafe {
        shim_hbd_lpf(d, width as i32, buf.as_mut_ptr().add(center), pitch as i32, b.as_ptr(), l.as_ptr(), t.as_ptr(), bd);
    }
}

// aom_dsp/loopfilter.c — deblocking edge filters.
pub type LpfFn = unsafe extern "C" fn(*mut u8, i32, *const u8, *const u8, *const u8);
extern "C" {
    pub fn aom_lpf_horizontal_4_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
    pub fn aom_lpf_horizontal_6_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
    pub fn aom_lpf_horizontal_8_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
    pub fn aom_lpf_horizontal_14_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
    pub fn aom_lpf_vertical_4_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
    pub fn aom_lpf_vertical_6_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
    pub fn aom_lpf_vertical_8_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
    pub fn aom_lpf_vertical_14_c(s: *mut u8, p: i32, b: *const u8, l: *const u8, t: *const u8);
}

/// Apply a reference loop filter in place. `dir`: 'h'/'v'. `center` is the
/// index of `s[0]` in `buf`.
#[allow(clippy::too_many_arguments)]
pub fn ref_lpf(dir: u8, width: u32, buf: &mut [u8], center: usize, pitch: usize, blimit: u8, limit: u8, thresh: u8) {
    let b = [blimit];
    let l = [limit];
    let t = [thresh];
    let f: LpfFn = match (dir, width) {
        (b'h', 4) => aom_lpf_horizontal_4_c,
        (b'h', 6) => aom_lpf_horizontal_6_c,
        (b'h', 8) => aom_lpf_horizontal_8_c,
        (b'h', 14) => aom_lpf_horizontal_14_c,
        (b'v', 4) => aom_lpf_vertical_4_c,
        (b'v', 6) => aom_lpf_vertical_6_c,
        (b'v', 8) => aom_lpf_vertical_8_c,
        (b'v', 14) => aom_lpf_vertical_14_c,
        _ => unreachable!(),
    };
    unsafe {
        f(buf.as_mut_ptr().add(center), pitch as i32, b.as_ptr(), l.as_ptr(), t.as_ptr());
    }
}

// av1/common/reconintra.c — directional predictors (edges passed at +pad).
extern "C" {
    pub fn av1_dr_prediction_z1_c(
        dst: *mut u8, stride: isize, bw: i32, bh: i32, above: *const u8, left: *const u8,
        upsample_above: i32, dx: i32, dy: i32,
    );
    pub fn av1_dr_prediction_z2_c(
        dst: *mut u8, stride: isize, bw: i32, bh: i32, above: *const u8, left: *const u8,
        upsample_above: i32, upsample_left: i32, dx: i32, dy: i32,
    );
    pub fn av1_dr_prediction_z3_c(
        dst: *mut u8, stride: isize, bw: i32, bh: i32, above: *const u8, left: *const u8,
        upsample_left: i32, dx: i32, dy: i32,
    );
}

/// Reference directional predictor. `above`/`left` are padded buffers; the C
/// pointer is taken at offset `pad`. Returns the `bw*bh` block (stride = bw).
#[allow(clippy::too_many_arguments)]
pub fn ref_dr_pred(
    kind: u8, bw: usize, bh: usize, above: &[u8], left: &[u8], pad: usize,
    up_above: i32, up_left: i32, dx: i32, dy: i32,
) -> Vec<u8> {
    let mut dst = vec![0u8; bw * bh];
    let ap = unsafe { above.as_ptr().add(pad) };
    let lp = unsafe { left.as_ptr().add(pad) };
    unsafe {
        match kind {
            1 => av1_dr_prediction_z1_c(dst.as_mut_ptr(), bw as isize, bw as i32, bh as i32, ap, lp, up_above, dx, dy),
            2 => av1_dr_prediction_z2_c(dst.as_mut_ptr(), bw as isize, bw as i32, bh as i32, ap, lp, up_above, up_left, dx, dy),
            3 => av1_dr_prediction_z3_c(dst.as_mut_ptr(), bw as isize, bw as i32, bh as i32, ap, lp, up_left, dx, dy),
            _ => unreachable!(),
        }
    }
    dst
}

// intra_shim.c — dispatch to aom_<mode>_predictor_<W>x<H>_c.
extern "C" {
    fn shim_intra_pred(
        mode: i32, size_idx: i32, dst: *mut u8, stride: isize, above: *const u8, left: *const u8,
    );
}

extern "C" {
    fn shim_highbd_intra_pred(
        mode: i32, size_idx: i32, dst: *mut u16, stride: isize, above: *const u16, left: *const u16, bd: i32,
    );
}

/// Reference highbd intra prediction. Returns the `bw*bh` predicted block.
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_intra_pred(
    mode: usize, size_idx: usize, bw: usize, bh: usize, above_tl: &[u16], left: &[u16], bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; bw * bh];
    unsafe {
        shim_highbd_intra_pred(
            mode as i32, size_idx as i32, dst.as_mut_ptr(), bw as isize,
            above_tl.as_ptr().add(1), left.as_ptr(), bd,
        )
    }
    dst
}

/// Reference intra prediction. `above_tl` has the top-left at index 0 followed
/// by the `bw` above samples; `left` has `bh` samples. Returns the `bw*bh`
/// predicted block (stride = bw).
pub fn ref_intra_pred(
    mode: usize,
    size_idx: usize,
    bw: usize,
    bh: usize,
    above_tl: &[u8],
    left: &[u8],
) -> Vec<u8> {
    let mut dst = vec![0u8; bw * bh];
    unsafe {
        // C `above` points past the top-left sample so above[-1] is valid.
        shim_intra_pred(
            mode as i32, size_idx as i32, dst.as_mut_ptr(), bw as isize,
            above_tl.as_ptr().add(1), left.as_ptr(),
        )
    }
    dst
}

/// Reference `update_cdf`: returns the updated cdf array (length nsymbs+1).
pub fn ref_update_cdf(cdf: &[u16], val: i32, nsymbs: usize) -> Vec<u16> {
    let mut out = cdf.to_vec();
    unsafe { shim_update_cdf(out.as_mut_ptr(), val, nsymbs as i32) }
    out
}

/// Reference adaptive symbol encode (single shared adapting context).
pub fn ref_adapt_encode(syms: &[i32], cdf_init: &[u16], nsymbs: usize) -> Vec<u8> {
    let mut out = vec![0u8; syms.len() * 4 + 64];
    let n = unsafe {
        shim_adapt_encode(
            syms.as_ptr(), syms.len() as i32, cdf_init.as_ptr(), nsymbs as i32,
            out.as_mut_ptr(), out.len() as u32,
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference adaptive symbol decode.
pub fn ref_adapt_decode(buf: &[u8], n: usize, cdf_init: &[u16], nsymbs: usize) -> Vec<i32> {
    let mut out = vec![0i32; n];
    unsafe {
        shim_adapt_decode(
            buf.as_ptr(), buf.len() as u32, n as i32, cdf_init.as_ptr(), nsymbs as i32,
            out.as_mut_ptr(),
        )
    }
    out
}

/// One entropy-coder op for the reference encoder/decoder.
#[derive(Clone)]
pub enum EcOp {
    Bool { val: i32, f: u32 },
    Cdf { s: i32, icdf: Vec<u16> },
}

/// Reference-encode a sequence of ops; return the finalized byte buffer.
pub fn ref_ec_encode(ops: &[EcOp]) -> Vec<u8> {
    unsafe {
        let e = shim_enc_new(1024);
        for op in ops {
            match op {
                EcOp::Bool { val, f } => shim_enc_bool(e, *val, *f),
                EcOp::Cdf { s, icdf } => shim_enc_cdf(e, *s, icdf.as_ptr(), icdf.len() as i32),
            }
        }
        let mut n: u32 = 0;
        let p = shim_enc_done(e, &mut n);
        let out = std::slice::from_raw_parts(p, n as usize).to_vec();
        shim_enc_free(e);
        out
    }
}

/// Reference-decode `ops` (using each op's `f`/`icdf`) from `buf`; return the
/// decoded symbol/bit for each op.
pub fn ref_ec_decode(buf: &[u8], ops: &[EcOp]) -> Vec<i32> {
    unsafe {
        let d = shim_dec_new(buf.as_ptr(), buf.len() as u32);
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            let r = match op {
                EcOp::Bool { f, .. } => shim_dec_bool(d, *f),
                EcOp::Cdf { icdf, .. } => shim_dec_cdf(d, icdf.as_ptr(), icdf.len() as i32),
            };
            out.push(r);
        }
        shim_dec_free(d);
        out
    }
}

/// Initialize the reference library's dispatch tables exactly once.
pub fn ref_init() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        av1_rtcd();
        aom_dsp_rtcd();
        aom_scale_rtcd();
    });
}

extern "C" {
    // av1/encoder/av1_fwd_txfm1d.c — forward 1D transforms.
    pub fn av1_fdct4(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fdct8(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fdct16(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fdct32(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fdct64(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fadst4(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fadst8(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fadst16(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fidentity4_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fidentity8_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fidentity16_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_fidentity32_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);

    // av1/common/av1_inv_txfm1d.c — inverse 1D transforms.
    pub fn av1_idct4(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_idct8(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_idct16(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_idct32(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_idct64(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_iadst4(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_iadst8(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_iadst16(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_iidentity4_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_iidentity8_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_iidentity16_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);
    pub fn av1_iidentity32_c(i: *const i32, o: *mut i32, c: i8, s: *const i8);
}

/// Call a reference forward 1D transform, returning `n` output coefficients.
pub fn ref_txfm1d(
    f: unsafe extern "C" fn(*const i32, *mut i32, i8, *const i8),
    input: &[i32],
    cos_bit: i8,
    stage_range: &[i8],
) -> Vec<i32> {
    let mut out = vec![0i32; input.len()];
    unsafe { f(input.as_ptr(), out.as_mut_ptr(), cos_bit, stage_range.as_ptr()) }
    out
}

/// Convenience wrapper kept for the original fdct4 harness.
pub fn ref_fdct4(input: &[i32; 4], cos_bit: i8, stage_range: &[i8; 8]) -> [i32; 4] {
    let mut out = [0i32; 4];
    unsafe { av1_fdct4(input.as_ptr(), out.as_mut_ptr(), cos_bit, stage_range.as_ptr()) }
    out
}

// av1/encoder/av1_fwd_txfm2d.c — forward 2D entry points (one per TX_SIZE).
// Signature: (const int16_t*, int32_t*, int stride, TX_TYPE tx_type, int bd).
pub type Fwd2dFn = unsafe extern "C" fn(*const i16, *mut i32, i32, i32, i32);
extern "C" {
    pub fn av1_fwd_txfm2d_4x4_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_8x8_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_16x16_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_32x32_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_64x64_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_4x8_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_8x4_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_8x16_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_16x8_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_16x32_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_32x16_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_32x64_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_64x32_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_4x16_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_16x4_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_8x32_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_32x8_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_16x64_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
    pub fn av1_fwd_txfm2d_64x16_c(i: *const i16, o: *mut i32, s: i32, t: i32, bd: i32);
}

/// Reference forward 2-D transform for `tx_size` (0..19), returning `wide*high`
/// coefficients. `bd` is fixed at 8 (does not affect output).
pub fn ref_fwd_txfm2d(tx_size: usize, input: &[i16], stride: usize, tx_type: usize) -> Vec<i32> {
    const W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
    const H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
    let f: Fwd2dFn = match tx_size {
        0 => av1_fwd_txfm2d_4x4_c,
        1 => av1_fwd_txfm2d_8x8_c,
        2 => av1_fwd_txfm2d_16x16_c,
        3 => av1_fwd_txfm2d_32x32_c,
        4 => av1_fwd_txfm2d_64x64_c,
        5 => av1_fwd_txfm2d_4x8_c,
        6 => av1_fwd_txfm2d_8x4_c,
        7 => av1_fwd_txfm2d_8x16_c,
        8 => av1_fwd_txfm2d_16x8_c,
        9 => av1_fwd_txfm2d_16x32_c,
        10 => av1_fwd_txfm2d_32x16_c,
        11 => av1_fwd_txfm2d_32x64_c,
        12 => av1_fwd_txfm2d_64x32_c,
        13 => av1_fwd_txfm2d_4x16_c,
        14 => av1_fwd_txfm2d_16x4_c,
        15 => av1_fwd_txfm2d_8x32_c,
        16 => av1_fwd_txfm2d_32x8_c,
        17 => av1_fwd_txfm2d_16x64_c,
        18 => av1_fwd_txfm2d_64x16_c,
        _ => unreachable!(),
    };
    ref_init();
    let mut out = vec![0i32; W[tx_size] * H[tx_size]];
    unsafe { f(input.as_ptr(), out.as_mut_ptr(), stride as i32, tx_type as i32, 8) }
    out
}

// av1/common/av1_inv_txfm2d.c — inverse 2D add entry points.
// Signature: (const int32_t*, uint16_t* dest, int stride, TX_TYPE, int bd).
pub type Inv2dFn = unsafe extern "C" fn(*const i32, *mut u16, i32, i32, i32);
extern "C" {
    pub fn av1_inv_txfm2d_add_4x4_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_8x8_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_16x16_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_32x32_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_64x64_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_4x8_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_8x4_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_8x16_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_16x8_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_16x32_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_32x16_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_32x64_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_64x32_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_4x16_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_16x4_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_8x32_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_32x8_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_16x64_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
    pub fn av1_inv_txfm2d_add_64x16_c(i: *const i32, o: *mut u16, s: i32, t: i32, bd: i32);
}

// av1/encoder/av1_quantize.c — fast-path quantizers (no quant matrix).
pub type QuantFpFn = unsafe extern "C" fn(
    *const i32, isize, *const i16, *const i16, *const i16, *const i16, *mut i32, *mut i32,
    *const i16, *mut u16, *const i16, *const i16,
);
extern "C" {
    pub fn av1_quantize_fp_c(
        coeff: *const i32, n: isize, zbin: *const i16, round: *const i16, quant: *const i16,
        quant_shift: *const i16, qcoeff: *mut i32, dqcoeff: *mut i32, dequant: *const i16,
        eob: *mut u16, scan: *const i16, iscan: *const i16,
    );
    pub fn av1_quantize_fp_32x32_c(
        coeff: *const i32, n: isize, zbin: *const i16, round: *const i16, quant: *const i16,
        quant_shift: *const i16, qcoeff: *mut i32, dqcoeff: *mut i32, dequant: *const i16,
        eob: *mut u16, scan: *const i16, iscan: *const i16,
    );
    pub fn av1_quantize_fp_64x64_c(
        coeff: *const i32, n: isize, zbin: *const i16, round: *const i16, quant: *const i16,
        quant_shift: *const i16, qcoeff: *mut i32, dqcoeff: *mut i32, dequant: *const i16,
        eob: *mut u16, scan: *const i16, iscan: *const i16,
    );
}

/// Reference `av1_quantize_fp` family. `log_scale` selects 0/1/2. Returns
/// (qcoeff, dqcoeff, eob).
pub fn ref_quantize_fp(
    log_scale: i32,
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    scan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let mut qcoeff = vec![0i32; n];
    let mut dqcoeff = vec![0i32; n];
    let mut eob: u16 = 0;
    // zbin/quant_shift/iscan are unused by the fp path but must be valid ptrs.
    let dummy = vec![0i16; n.max(2)];
    let f: QuantFpFn = match log_scale {
        0 => av1_quantize_fp_c,
        1 => av1_quantize_fp_32x32_c,
        2 => av1_quantize_fp_64x64_c,
        _ => unreachable!(),
    };
    unsafe {
        f(
            coeff.as_ptr(), n as isize, dummy.as_ptr(), round.as_ptr(), quant.as_ptr(),
            dummy.as_ptr(), qcoeff.as_mut_ptr(), dqcoeff.as_mut_ptr(), dequant.as_ptr(),
            &mut eob, scan.as_ptr(), dummy.as_ptr(),
        )
    }
    (qcoeff, dqcoeff, eob)
}

// aom_dsp/quantize.c — "b" quantizer helper (dead-zone + quant/quant_shift).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn aom_quantize_b_helper_c(
        coeff: *const i32, n: isize, zbin: *const i16, round: *const i16, quant: *const i16,
        quant_shift: *const i16, qcoeff: *mut i32, dqcoeff: *mut i32, dequant: *const i16,
        eob: *mut u16, scan: *const i16, iscan: *const i16, qm: *const u8, iqm: *const u8,
        log_scale: i32,
    );
}

/// Reference `aom_quantize_b_helper_c` with no quant matrix. Returns
/// (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_quantize_b(
    log_scale: i32,
    coeff: &[i32],
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    scan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let mut qcoeff = vec![0i32; n];
    let mut dqcoeff = vec![0i32; n];
    let mut eob: u16 = 0;
    let dummy = vec![0i16; n.max(2)];
    unsafe {
        aom_quantize_b_helper_c(
            coeff.as_ptr(), n as isize, zbin.as_ptr(), round.as_ptr(), quant.as_ptr(),
            quant_shift.as_ptr(), qcoeff.as_mut_ptr(), dqcoeff.as_mut_ptr(), dequant.as_ptr(),
            &mut eob, scan.as_ptr(), dummy.as_ptr(), std::ptr::null(), std::ptr::null(), log_scale,
        )
    }
    (qcoeff, dqcoeff, eob)
}

/// Reference `aom_quantize_b_helper_c` *with* a quant matrix (`qm`/`iqm` indexed
/// by raster position). Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_quantize_b_qm(
    log_scale: i32,
    coeff: &[i32],
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let mut qcoeff = vec![0i32; n];
    let mut dqcoeff = vec![0i32; n];
    let mut eob: u16 = 0;
    let dummy = vec![0i16; n.max(2)];
    unsafe {
        aom_quantize_b_helper_c(
            coeff.as_ptr(), n as isize, zbin.as_ptr(), round.as_ptr(), quant.as_ptr(),
            quant_shift.as_ptr(), qcoeff.as_mut_ptr(), dqcoeff.as_mut_ptr(), dequant.as_ptr(),
            &mut eob, scan.as_ptr(), dummy.as_ptr(), qm.as_ptr(), iqm.as_ptr(), log_scale,
        )
    }
    (qcoeff, dqcoeff, eob)
}

/// Reference `aom_highbd_quantize_b_helper_c` *with* a quant matrix. Returns
/// (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_quantize_b_qm(
    log_scale: i32,
    coeff: &[i32],
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let mut qcoeff = vec![0i32; n];
    let mut dqcoeff = vec![0i32; n];
    let mut eob: u16 = 0;
    let dummy = vec![0i16; n.max(2)];
    unsafe {
        aom_highbd_quantize_b_helper_c(
            coeff.as_ptr(), n as isize, zbin.as_ptr(), round.as_ptr(), quant.as_ptr(),
            quant_shift.as_ptr(), qcoeff.as_mut_ptr(), dqcoeff.as_mut_ptr(), dequant.as_ptr(),
            &mut eob, scan.as_ptr(), dummy.as_ptr(), qm.as_ptr(), iqm.as_ptr(), log_scale,
        )
    }
    (qcoeff, dqcoeff, eob)
}

/// Reference inverse 2-D transform+add for `tx_size` (0..19). `dest` is the
/// bd-bit pixel buffer to reconstruct onto (modified in place).
pub fn ref_inv_txfm2d_add(
    tx_size: usize,
    input: &[i32],
    dest: &mut [u16],
    stride: usize,
    tx_type: usize,
    bd: i32,
) {
    let f: Inv2dFn = match tx_size {
        0 => av1_inv_txfm2d_add_4x4_c,
        1 => av1_inv_txfm2d_add_8x8_c,
        2 => av1_inv_txfm2d_add_16x16_c,
        3 => av1_inv_txfm2d_add_32x32_c,
        4 => av1_inv_txfm2d_add_64x64_c,
        5 => av1_inv_txfm2d_add_4x8_c,
        6 => av1_inv_txfm2d_add_8x4_c,
        7 => av1_inv_txfm2d_add_8x16_c,
        8 => av1_inv_txfm2d_add_16x8_c,
        9 => av1_inv_txfm2d_add_16x32_c,
        10 => av1_inv_txfm2d_add_32x16_c,
        11 => av1_inv_txfm2d_add_32x64_c,
        12 => av1_inv_txfm2d_add_64x32_c,
        13 => av1_inv_txfm2d_add_4x16_c,
        14 => av1_inv_txfm2d_add_16x4_c,
        15 => av1_inv_txfm2d_add_8x32_c,
        16 => av1_inv_txfm2d_add_32x8_c,
        17 => av1_inv_txfm2d_add_16x64_c,
        18 => av1_inv_txfm2d_add_64x16_c,
        _ => unreachable!(),
    };
    ref_init();
    unsafe { f(input.as_ptr(), dest.as_mut_ptr(), stride as i32, tx_type as i32, bd) }
}

// txb_shim.c — transform-block coefficient-coding kernels + scan/ctx data.
extern "C" {
    fn shim_txb_init_levels(coeff: *const i32, width: i32, height: i32, levels: *mut u8);
    fn shim_get_nz_map_contexts(levels: *const u8, scan: *const i16, eob: i32, tx_size: i32, tx_class: i32, out: *mut i8);
    fn shim_eob_pos_token(eob: i32, extra: *mut i32) -> i32;
    fn shim_nz_ctx_offset(tx_size: i32) -> *const i8;
    fn shim_scan(tx_size: i32, tx_type: i32) -> *const i16;
    fn shim_iscan(tx_size: i32, tx_type: i32) -> *const i16;
    fn shim_cost_tokens_from_cdf(costs: *mut i32, cdf: *const u16, inv_map: *const i32);
    fn shim_get_txb_ctx(plane_bsize: i32, tx_size: i32, plane: i32, a: *const i8, l: *const i8, out: *mut i32);
    fn shim_txb_entropy_context(qcoeff: *const i32, tx_size: i32, tx_type: i32, eob: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_optimize_txb(tx_size: i32, tx_type: i32, qcoeff: *mut i32, dqcoeff: *mut i32, tcoeff: *const i32, eob: i32, dequant: *const i16, rdmult: i64, dc_sign_ctx: i32, txb_skip_ctx: i32, sharpness: i32, scan: *const i16, txb_skip_cost: *const i32, base_eob_cost: *const i32, base_cost: *const i32, eob_extra_cost: *const i32, dc_sign_cost: *const i32, lps_cost: *const i32, eob_cost: *const i32, iqm: *const u8, qm: *const u8, out_rate: *mut i32) -> i32;
    fn shim_get_dqv(dequant: *const i16, coeff_idx: i32, iqm: *const u8) -> i32;
    fn shim_get_coeff_dist(tcoeff: i32, dqcoeff: i32, shift: i32, qm: *const u8, coeff_idx: i32) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn shim_two_coeff_cost_simple(ci: i32, abs_qc: i32, coeff_ctx: i32, base: *const i32, lps: *const i32, bhl: i32, tx_class: i32, levels: *const u8, cost_low: *mut i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_coeff_cost_eob(ci: i32, abs_qc: i32, sign: i32, coeff_ctx: i32, dc_sign_ctx: i32, base_eob: *const i32, dc_sign: *const i32, lps: *const i32, bhl: i32, tx_class: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_coeff_cost_general(is_last: i32, ci: i32, abs_qc: i32, sign: i32, coeff_ctx: i32, dc_sign_ctx: i32, base_eob: *const i32, base: *const i32, dc_sign: *const i32, lps: *const i32, bhl: i32, tx_class: i32, levels: *const u8) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_ext_tx_derive(tx_size: i32, is_inter: i32, reduced: i32, tx_type: i32, use_fi: i32, fi_mode: i32, mode: i32, out: *mut i32);
    #[allow(clippy::too_many_arguments)]
    fn shim_fill_lv_map(txb_skip_cdf: *const u16, base_eob_cdf: *const u16, base_cdf: *const u16, eob_extra_cdf: *const u16, dc_sign_cdf: *const u16, br_cdf: *const u16, o_txb_skip: *mut i32, o_base_eob: *mut i32, o_base: *mut i32, o_eob_extra: *mut i32, o_dc_sign: *mut i32, o_lps: *mut i32);
    #[allow(clippy::too_many_arguments)]
    fn shim_cost_coeffs_txb(qcoeff: *const i32, eob: i32, tx_size: i32, tx_type: i32, txb_skip_ctx: i32, dc_sign_ctx: i32, txb_skip_cost: *const i32, base_eob_cost: *const i32, base_cost: *const i32, eob_extra_cost: *const i32, dc_sign_cost: *const i32, lps_cost: *const i32, eob_cost: *const i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_coeffs_txb(tcoeff: *const i32, eob: i32, tx_size: i32, tx_type: i32, plane_type: i32, txb_skip_ctx: i32, dc_sign_ctx: i32, allow_update_cdf: i32, cdfs: *mut u16, ext_tx_cdf: *mut u16, is_inter: i32, reduced: i32, use_fi: i32, fi_mode: i32, mode: i32, signal_gate: i32, out: *mut u8, out_cap: i32) -> i32;
}

/// Reference `av1_txb_init_levels_c` (writes into `levels`).
pub fn ref_txb_init_levels(coeff: &[i32], width: usize, height: usize, levels: &mut [u8]) {
    unsafe { shim_txb_init_levels(coeff.as_ptr(), width as i32, height as i32, levels.as_mut_ptr()) }
}

/// Reference `av1_get_nz_map_contexts_c` (writes `out[scan[i]]` for `i < eob`).
pub fn ref_get_nz_map_contexts(levels: &[u8], scan: &[i16], eob: usize, tx_size: usize, tx_class: i32, out: &mut [i8]) {
    unsafe {
        shim_get_nz_map_contexts(levels.as_ptr(), scan.as_ptr(), eob as i32, tx_size as i32, tx_class, out.as_mut_ptr())
    }
}

/// Reference `av1_get_eob_pos_token`; returns (token, extra).
pub fn ref_eob_pos_token(eob: i32) -> (i32, i32) {
    let mut extra = 0i32;
    let t = unsafe { shim_eob_pos_token(eob, &mut extra) };
    (t, extra)
}

/// Copy of the C `av1_nz_map_ctx_offset[tx_size]` table's first `len` entries.
pub fn ref_nz_ctx_offset(tx_size: usize, len: usize) -> Vec<i8> {
    unsafe { core::slice::from_raw_parts(shim_nz_ctx_offset(tx_size as i32), len).to_vec() }
}

/// Copy of the C `av1_scan_orders[tx_size][tx_type].scan` (first `len` entries).
pub fn ref_scan_order(tx_size: usize, tx_type: usize, len: usize) -> Vec<i16> {
    unsafe { core::slice::from_raw_parts(shim_scan(tx_size as i32, tx_type as i32), len).to_vec() }
}

/// Copy of the C `av1_scan_orders[tx_size][tx_type].iscan` (first `len` entries).
pub fn ref_iscan_order(tx_size: usize, tx_type: usize, len: usize) -> Vec<i16> {
    unsafe { core::slice::from_raw_parts(shim_iscan(tx_size as i32, tx_type as i32), len).to_vec() }
}

/// Reference `av1_write_coeffs_txb` (transcribed harness). Mutates `cdfs` when
/// `allow_update_cdf`; returns the produced bitstream bytes.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_coeffs_txb(tcoeff: &[i32], eob: usize, tx_size: usize, tx_type: usize, plane_type: usize, txb_skip_ctx: usize, dc_sign_ctx: usize, allow_update_cdf: bool, cdfs: &mut [u16]) -> Vec<u8> {
    let mut out = vec![0u8; 1 << 16];
    let mut dummy = [0u16; 16];
    // signal_gate = 0 => no tx_type write (reproduces the coeff-only path).
    let n = unsafe {
        shim_write_coeffs_txb(tcoeff.as_ptr(), eob as i32, tx_size as i32, tx_type as i32, plane_type as i32, txb_skip_ctx as i32, dc_sign_ctx as i32, allow_update_cdf as i32, cdfs.as_mut_ptr(), dummy.as_mut_ptr(), 0, 0, 0, 0, 0, 0, out.as_mut_ptr(), out.len() as i32)
    };
    out.truncate(n as usize);
    out
}

/// Reference full txb writer: `txb_skip` + (luma) `av1_write_tx_type` + coeffs,
/// matching aom-txb's `write_coeffs_txb_full`. `ext_tx_cdf` is the selected ext-tx
/// CDF slot; the tx_type context mirrors the encoder mbmi/frame state. Returns bytes.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_coeffs_txb_full(tcoeff: &[i32], eob: usize, tx_size: usize, tx_type: usize, plane_type: usize, txb_skip_ctx: usize, dc_sign_ctx: usize, allow_update_cdf: bool, cdfs: &mut [u16], ext_tx_cdf: &mut [u16], is_inter: bool, reduced: bool, use_fi: bool, fi_mode: usize, mode: usize, signal_gate: bool) -> Vec<u8> {
    let mut out = vec![0u8; 1 << 16];
    let n = unsafe {
        shim_write_coeffs_txb(tcoeff.as_ptr(), eob as i32, tx_size as i32, tx_type as i32, plane_type as i32, txb_skip_ctx as i32, dc_sign_ctx as i32, allow_update_cdf as i32, cdfs.as_mut_ptr(), ext_tx_cdf.as_mut_ptr(), is_inter as i32, reduced as i32, use_fi as i32, fi_mode as i32, mode as i32, signal_gate as i32, out.as_mut_ptr(), out.len() as i32)
    };
    out.truncate(n as usize);
    out
}

/// Reference `av1_cost_coeffs_txb` (warehouse_efficients_txb, tx_type cost out
/// of scope). Cost tables are flat: txb_skip_cost[13][2], base_eob_cost[4][3],
/// base_cost[42][8], eob_extra_cost[9][2], dc_sign_cost[3][2], lps_cost[21][26],
/// eob_cost[2][11].
#[allow(clippy::too_many_arguments)]
pub fn ref_cost_coeffs_txb(qcoeff: &[i32], eob: usize, tx_size: usize, tx_type: usize, txb_skip_ctx: usize, dc_sign_ctx: usize, txb_skip_cost: &[i32], base_eob_cost: &[i32], base_cost: &[i32], eob_extra_cost: &[i32], dc_sign_cost: &[i32], lps_cost: &[i32], eob_cost: &[i32]) -> i32 {
    unsafe {
        shim_cost_coeffs_txb(qcoeff.as_ptr(), eob as i32, tx_size as i32, tx_type as i32, txb_skip_ctx as i32, dc_sign_ctx as i32, txb_skip_cost.as_ptr(), base_eob_cost.as_ptr(), base_cost.as_ptr(), eob_extra_cost.as_ptr(), dc_sign_cost.as_ptr(), lps_cost.as_ptr(), eob_cost.as_ptr())
    }
}

/// Reference `av1_cost_tokens_from_cdf`: derive per-symbol costs from an
/// `nsymbs`-symbol inverse-CDF. `inv_map` (or None) permutes the output.
pub fn ref_cost_tokens_from_cdf(nsymbs: usize, cdf: &[u16], inv_map: Option<&[i32]>) -> Vec<i32> {
    let mut costs = vec![0i32; nsymbs];
    let im = inv_map.map_or(core::ptr::null(), |m| m.as_ptr());
    unsafe { shim_cost_tokens_from_cdf(costs.as_mut_ptr(), cdf.as_ptr(), im) }
    costs
}

/// Reference `av1_fill_coeff_costs` per (txs_ctx, plane). Returns the flat
/// (txb_skip[13*2], base_eob[4*3], base[42*8], eob_extra[9*2], dc_sign[3*2],
/// lps[21*26]) tables.
#[allow(clippy::type_complexity)]
pub fn ref_fill_lv_map(txb_skip_cdf: &[u16], base_eob_cdf: &[u16], base_cdf: &[u16], eob_extra_cdf: &[u16], dc_sign_cdf: &[u16], br_cdf: &[u16]) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut txb_skip = vec![0i32; 13 * 2];
    let mut base_eob = vec![0i32; 4 * 3];
    let mut base = vec![0i32; 42 * 8];
    let mut eob_extra = vec![0i32; 9 * 2];
    let mut dc_sign = vec![0i32; 3 * 2];
    let mut lps = vec![0i32; 21 * 26];
    unsafe {
        shim_fill_lv_map(txb_skip_cdf.as_ptr(), base_eob_cdf.as_ptr(), base_cdf.as_ptr(), eob_extra_cdf.as_ptr(), dc_sign_cdf.as_ptr(), br_cdf.as_ptr(), txb_skip.as_mut_ptr(), base_eob.as_mut_ptr(), base.as_mut_ptr(), eob_extra.as_mut_ptr(), dc_sign.as_mut_ptr(), lps.as_mut_ptr());
    }
    (txb_skip, base_eob, base, eob_extra, dc_sign, lps)
}

/// Reference ext-tx derivation for `av1_write_tx_type`. Returns
/// (set_type, num, eset, square_tx_size, symb, used, intra_dir).
#[allow(clippy::too_many_arguments)]
pub fn ref_ext_tx_derive(tx_size: usize, is_inter: bool, reduced: bool, tx_type: usize, use_fi: bool, fi_mode: usize, mode: usize) -> [i32; 7] {
    let mut out = [0i32; 7];
    unsafe { shim_ext_tx_derive(tx_size as i32, is_inter as i32, reduced as i32, tx_type as i32, use_fi as i32, fi_mode as i32, mode as i32, out.as_mut_ptr()) }
    out
}

/// Reference `get_two_coeff_cost_simple`; returns (cost, cost_low).
#[allow(clippy::too_many_arguments)]
pub fn ref_two_coeff_cost_simple(ci: usize, abs_qc: i32, coeff_ctx: usize, base: &[i32], lps: &[i32], bhl: u32, tx_class: i32, levels: &[u8]) -> (i32, i32) {
    let mut cost_low = 0i32;
    let cost = unsafe { shim_two_coeff_cost_simple(ci as i32, abs_qc, coeff_ctx as i32, base.as_ptr(), lps.as_ptr(), bhl as i32, tx_class, levels.as_ptr(), &mut cost_low) };
    (cost, cost_low)
}

/// Reference `get_coeff_cost_eob`.
#[allow(clippy::too_many_arguments)]
pub fn ref_coeff_cost_eob(ci: usize, abs_qc: i32, sign: usize, coeff_ctx: usize, dc_sign_ctx: usize, base_eob: &[i32], dc_sign: &[i32], lps: &[i32], bhl: u32, tx_class: i32) -> i32 {
    unsafe { shim_coeff_cost_eob(ci as i32, abs_qc, sign as i32, coeff_ctx as i32, dc_sign_ctx as i32, base_eob.as_ptr(), dc_sign.as_ptr(), lps.as_ptr(), bhl as i32, tx_class) }
}

/// Reference `get_coeff_cost_general`.
#[allow(clippy::too_many_arguments)]
pub fn ref_coeff_cost_general(is_last: bool, ci: usize, abs_qc: i32, sign: usize, coeff_ctx: usize, dc_sign_ctx: usize, base_eob: &[i32], base: &[i32], dc_sign: &[i32], lps: &[i32], bhl: u32, tx_class: i32, levels: &[u8]) -> i32 {
    unsafe { shim_coeff_cost_general(is_last as i32, ci as i32, abs_qc, sign as i32, coeff_ctx as i32, dc_sign_ctx as i32, base_eob.as_ptr(), base.as_ptr(), dc_sign.as_ptr(), lps.as_ptr(), bhl as i32, tx_class, levels.as_ptr()) }
}

/// Reference `av1_optimize_txb` (non-QM trellis). Optimizes qcoeff/dqcoeff in
/// place; returns (eob, rate_cost).
#[allow(clippy::too_many_arguments)]
pub fn ref_optimize_txb(tx_size: usize, tx_type: usize, qcoeff: &mut [i32], dqcoeff: &mut [i32], tcoeff: &[i32], eob: usize, dequant: &[i16], rdmult: i64, dc_sign_ctx: usize, txb_skip_ctx: usize, sharpness: i32, scan: &[i16], txb_skip: &[i32], base_eob: &[i32], base: &[i32], eob_extra: &[i32], dc_sign: &[i32], lps: &[i32], eob_cost: &[i32]) -> (usize, i32) {
    let mut rate = 0i32;
    let e = unsafe { shim_optimize_txb(tx_size as i32, tx_type as i32, qcoeff.as_mut_ptr(), dqcoeff.as_mut_ptr(), tcoeff.as_ptr(), eob as i32, dequant.as_ptr(), rdmult, dc_sign_ctx as i32, txb_skip_ctx as i32, sharpness, scan.as_ptr(), txb_skip.as_ptr(), base_eob.as_ptr(), base.as_ptr(), eob_extra.as_ptr(), dc_sign.as_ptr(), lps.as_ptr(), eob_cost.as_ptr(), core::ptr::null(), core::ptr::null(), &mut rate) };
    (e as usize, rate)
}

/// Reference `av1_optimize_txb` *with* a quant matrix: `iqm` folds into the
/// per-position dequant (`get_dqv`), `qm` folds into the distortion
/// (`get_coeff_dist`). Both indexed by raster position. Returns (eob, rate).
#[allow(clippy::too_many_arguments)]
pub fn ref_optimize_txb_qm(tx_size: usize, tx_type: usize, qcoeff: &mut [i32], dqcoeff: &mut [i32], tcoeff: &[i32], eob: usize, dequant: &[i16], rdmult: i64, dc_sign_ctx: usize, txb_skip_ctx: usize, sharpness: i32, scan: &[i16], txb_skip: &[i32], base_eob: &[i32], base: &[i32], eob_extra: &[i32], dc_sign: &[i32], lps: &[i32], eob_cost: &[i32], iqm: &[u8], qm: &[u8]) -> (usize, i32) {
    let mut rate = 0i32;
    let e = unsafe { shim_optimize_txb(tx_size as i32, tx_type as i32, qcoeff.as_mut_ptr(), dqcoeff.as_mut_ptr(), tcoeff.as_ptr(), eob as i32, dequant.as_ptr(), rdmult, dc_sign_ctx as i32, txb_skip_ctx as i32, sharpness, scan.as_ptr(), txb_skip.as_ptr(), base_eob.as_ptr(), base.as_ptr(), eob_extra.as_ptr(), dc_sign.as_ptr(), lps.as_ptr(), eob_cost.as_ptr(), iqm.as_ptr(), qm.as_ptr(), &mut rate) };
    (e as usize, rate)
}

/// Reference `get_dqv` (per-position dequant, folding iqmatrix).
pub fn ref_get_dqv(dequant: &[i16; 2], coeff_idx: usize, iqm: Option<&[u8]>) -> i32 {
    let iqp = iqm.map_or(core::ptr::null(), |s| s.as_ptr());
    unsafe { shim_get_dqv(dequant.as_ptr(), coeff_idx as i32, iqp) }
}

/// Reference `get_coeff_dist` (squared-error distortion, folding qmatrix).
pub fn ref_get_coeff_dist(tcoeff: i32, dqcoeff: i32, shift: i32, qm: Option<&[u8]>, coeff_idx: usize) -> i64 {
    let qp = qm.map_or(core::ptr::null(), |s| s.as_ptr());
    unsafe { shim_get_coeff_dist(tcoeff, dqcoeff, shift, qp, coeff_idx as i32) }
}

/// Reference `get_txb_ctx`; returns (txb_skip_ctx, dc_sign_ctx).
pub fn ref_get_txb_ctx(plane_bsize: usize, tx_size: usize, plane: usize, a: &[i8], l: &[i8]) -> (i32, i32) {
    let mut out = [0i32; 2];
    unsafe { shim_get_txb_ctx(plane_bsize as i32, tx_size as i32, plane as i32, a.as_ptr(), l.as_ptr(), out.as_mut_ptr()) }
    (out[0], out[1])
}

/// Reference `av1_get_txb_entropy_context`.
pub fn ref_txb_entropy_context(qcoeff: &[i32], tx_size: usize, tx_type: usize, eob: usize) -> u8 {
    unsafe { shim_txb_entropy_context(qcoeff.as_ptr(), tx_size as i32, tx_type as i32, eob as i32) as u8 }
}

// intra_edge_shim.c — intra edge filter / upsample DSP + strength decisions.
extern "C" {
    fn shim_intra_edge_strength(bs0: i32, bs1: i32, delta: i32, ty: i32) -> i32;
    fn shim_use_intra_edge_upsample(bs0: i32, bs1: i32, delta: i32, ty: i32) -> i32;
    fn shim_filter_intra_edge(p: *mut u8, sz: i32, strength: i32);
    fn shim_upsample_intra_edge(p: *mut u8, sz: i32);
}

/// Reference `intra_edge_filter_strength`.
pub fn ref_intra_edge_strength(bs0: i32, bs1: i32, delta: i32, ty: i32) -> i32 {
    unsafe { shim_intra_edge_strength(bs0, bs1, delta, ty) }
}

/// Reference `av1_use_intra_edge_upsample`.
pub fn ref_use_intra_edge_upsample(bs0: i32, bs1: i32, delta: i32, ty: i32) -> i32 {
    unsafe { shim_use_intra_edge_upsample(bs0, bs1, delta, ty) }
}

/// Reference `av1_filter_intra_edge_c`: filters `buf[off..off+sz]` in place.
pub fn ref_filter_intra_edge(buf: &mut [u8], off: usize, sz: usize, strength: i32) {
    unsafe { shim_filter_intra_edge(buf.as_mut_ptr().add(off), sz as i32, strength) }
}

/// Reference `av1_upsample_intra_edge_c`: `buf[off..]` is logical index 0; the
/// kernel reads `[-1..]` and writes `[-2 .. 2*sz-2]`.
pub fn ref_upsample_intra_edge(buf: &mut [u8], off: usize, sz: usize) {
    unsafe { shim_upsample_intra_edge(buf.as_mut_ptr().add(off), sz as i32) }
}

// Highbd intra edge DSP (exported reconintra.c symbols).
extern "C" {
    fn av1_highbd_filter_intra_edge_c(p: *mut u16, sz: i32, strength: i32);
    fn av1_highbd_upsample_intra_edge_c(p: *mut u16, sz: i32, bd: i32);
}

/// Reference `av1_highbd_filter_intra_edge_c` (filters `buf[off..off+sz]`).
pub fn ref_highbd_filter_intra_edge(buf: &mut [u16], off: usize, sz: usize, strength: i32) {
    unsafe { av1_highbd_filter_intra_edge_c(buf.as_mut_ptr().add(off), sz as i32, strength) }
}

/// Reference `av1_highbd_upsample_intra_edge_c`.
pub fn ref_highbd_upsample_intra_edge(buf: &mut [u16], off: usize, sz: usize, bd: u8) {
    unsafe { av1_highbd_upsample_intra_edge_c(buf.as_mut_ptr().add(off), sz as i32, bd as i32) }
}

// Highbd quantizers (exported av1_quantize.c / quantize.c symbols).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn av1_highbd_quantize_fp_c(coeff: *const i32, n: isize, zbin: *const i16, round: *const i16, quant: *const i16, quant_shift: *const i16, qcoeff: *mut i32, dqcoeff: *mut i32, dequant: *const i16, eob: *mut u16, scan: *const i16, iscan: *const i16, log_scale: i32);
    #[allow(clippy::too_many_arguments)]
    pub fn aom_highbd_quantize_b_helper_c(coeff: *const i32, n: isize, zbin: *const i16, round: *const i16, quant: *const i16, quant_shift: *const i16, qcoeff: *mut i32, dqcoeff: *mut i32, dequant: *const i16, eob: *mut u16, scan: *const i16, iscan: *const i16, qm: *const u8, iqm: *const u8, log_scale: i32);
}

/// Reference `av1_highbd_quantize_fp` (no qmatrix). Returns (qcoeff, dqcoeff, eob).
pub fn ref_highbd_quantize_fp(log_scale: i32, coeff: &[i32], round: &[i16; 2], quant: &[i16; 2], dequant: &[i16; 2], scan: &[i16]) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq, mut eob) = (vec![0i32; n], vec![0i32; n], 0u16);
    let dummy = vec![0i16; n.max(2)];
    unsafe { av1_highbd_quantize_fp_c(coeff.as_ptr(), n as isize, dummy.as_ptr(), round.as_ptr(), quant.as_ptr(), dummy.as_ptr(), q.as_mut_ptr(), dq.as_mut_ptr(), dequant.as_ptr(), &mut eob, scan.as_ptr(), dummy.as_ptr(), log_scale); }
    (q, dq, eob)
}

/// Reference `aom_highbd_quantize_b_helper_c` (no qmatrix). Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_quantize_b(log_scale: i32, coeff: &[i32], zbin: &[i16; 2], round: &[i16; 2], quant: &[i16; 2], quant_shift: &[i16; 2], dequant: &[i16; 2], scan: &[i16]) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq, mut eob) = (vec![0i32; n], vec![0i32; n], 0u16);
    let dummy = vec![0i16; n.max(2)];
    unsafe { aom_highbd_quantize_b_helper_c(coeff.as_ptr(), n as isize, zbin.as_ptr(), round.as_ptr(), quant.as_ptr(), quant_shift.as_ptr(), q.as_mut_ptr(), dq.as_mut_ptr(), dequant.as_ptr(), &mut eob, scan.as_ptr(), dummy.as_ptr(), core::ptr::null(), core::ptr::null(), log_scale); }
    (q, dq, eob)
}

// FP quant-matrix path: the static helpers reached via the real facades (see
// shim/quant_fp_shim.c). round/quant/dequant are the [2]-entry QTX tables.
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_quantize_fp_qm(coeff: *const i32, n: i32, round: *const i16, quant: *const i16, dequant: *const i16, scan: *const i16, iscan: *const i16, qm: *const u8, iqm: *const u8, log_scale: i32, qcoeff: *mut i32, dqcoeff: *mut i32) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn shim_highbd_quantize_fp_qm(coeff: *const i32, n: i32, round: *const i16, quant: *const i16, dequant: *const i16, scan: *const i16, iscan: *const i16, qm: *const u8, iqm: *const u8, log_scale: i32, qcoeff: *mut i32, dqcoeff: *mut i32) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn shim_quantize_dc(coeff: *const i32, n: i32, round: *const i16, quant: i16, dequant: i16, qm: *const u8, iqm: *const u8, log_scale: i32, qcoeff: *mut i32, dqcoeff: *mut i32) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn shim_highbd_quantize_dc(coeff: *const i32, n: i32, round: *const i16, quant: i16, dequant: i16, qm: *const u8, iqm: *const u8, log_scale: i32, qcoeff: *mut i32, dqcoeff: *mut i32) -> u16;
}

/// Reference `av1_quantize_dc_facade` (`quantize_dc`, DC-only). `qm`/`iqm` are
/// `None` for the flat path. Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_quantize_dc(log_scale: i32, coeff: &[i32], round: &[i16; 2], quant: i16, dequant: i16, qm: Option<&[u8]>, iqm: Option<&[u8]>) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let qp = qm.map_or(core::ptr::null(), |s| s.as_ptr());
    let iqp = iqm.map_or(core::ptr::null(), |s| s.as_ptr());
    let eob = unsafe { shim_quantize_dc(coeff.as_ptr(), n as i32, round.as_ptr(), quant, dequant, qp, iqp, log_scale, q.as_mut_ptr(), dq.as_mut_ptr()) };
    (q, dq, eob)
}

/// Reference `av1_highbd_quantize_dc_facade` (`highbd_quantize_dc`, DC-only).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_quantize_dc(log_scale: i32, coeff: &[i32], round: &[i16; 2], quant: i16, dequant: i16, qm: Option<&[u8]>, iqm: Option<&[u8]>) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let qp = qm.map_or(core::ptr::null(), |s| s.as_ptr());
    let iqp = iqm.map_or(core::ptr::null(), |s| s.as_ptr());
    let eob = unsafe { shim_highbd_quantize_dc(coeff.as_ptr(), n as i32, round.as_ptr(), quant, dequant, qp, iqp, log_scale, q.as_mut_ptr(), dq.as_mut_ptr()) };
    (q, dq, eob)
}

/// Reference lowbd `av1_quantize_fp_facade` QM path (`quantize_fp_helper_c` with
/// non-NULL qm/iqm). Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_quantize_fp_qm(log_scale: i32, coeff: &[i32], round: &[i16; 2], quant: &[i16; 2], dequant: &[i16; 2], qm: &[u8], iqm: &[u8], scan: &[i16], iscan: &[i16]) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let eob = unsafe { shim_quantize_fp_qm(coeff.as_ptr(), n as i32, round.as_ptr(), quant.as_ptr(), dequant.as_ptr(), scan.as_ptr(), iscan.as_ptr(), qm.as_ptr(), iqm.as_ptr(), log_scale, q.as_mut_ptr(), dq.as_mut_ptr()) };
    (q, dq, eob)
}

/// Reference highbd `av1_highbd_quantize_fp_facade` QM path
/// (`highbd_quantize_fp_helper_c` with non-NULL qm/iqm). Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_quantize_fp_qm(log_scale: i32, coeff: &[i32], round: &[i16; 2], quant: &[i16; 2], dequant: &[i16; 2], qm: &[u8], iqm: &[u8], scan: &[i16], iscan: &[i16]) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let eob = unsafe { shim_highbd_quantize_fp_qm(coeff.as_ptr(), n as i32, round.as_ptr(), quant.as_ptr(), dequant.as_ptr(), scan.as_ptr(), iscan.as_ptr(), qm.as_ptr(), iqm.as_ptr(), log_scale, q.as_mut_ptr(), dq.as_mut_ptr()) };
    (q, dq, eob)
}
