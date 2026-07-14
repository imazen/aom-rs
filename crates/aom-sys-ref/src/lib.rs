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
    fn shim_write_tile_group_header(start_tile: i32, end_tile: i32, tiles_log2: i32, present_flag: i32, out: *mut u8) -> u32;
    fn shim_write_tile_info(mi_cols: i32, mi_rows: i32, mib_size_log2: i32, uniform_spacing: i32, log2_cols: i32, min_log2_cols: i32, max_log2_cols: i32, log2_rows: i32, min_log2_rows: i32, max_log2_rows: i32, cols: i32, rows: i32, col_start_sb: *const i32, row_start_sb: *const i32, max_width_sb: i32, max_height_sb: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_restoration_mode(enable_restoration: i32, allow_intrabc: i32, frame_restoration_type: *const i32, sb_size_128: i32, restoration_unit_size: *const i32, ssx: i32, ssy: i32, num_planes: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_delta_q_params(base_qindex: i32, delta_q_present: i32, delta_q_res: i32, allow_intrabc: i32, delta_lf_present: i32, delta_lf_res: i32, delta_lf_multi: i32, out: *mut u8) -> u32;
    fn shim_write_tx_mode(coded_lossless: i32, tx_mode_select: i32, out: *mut u8) -> u32;
    fn shim_write_film_grain_params(s: *const i32, spy: *const i32, spcb: *const i32, spcr: *const i32, ary: *const i32, arcb: *const i32, arcr: *const i32, out: *mut u8) -> u32;
    fn shim_wb_signed_subexpfin(n: i32, k: i32, ref_: i32, v: i32, out: *mut u8) -> u32;
    fn shim_write_global_motion(wmtype: *const i32, wmmat: *const i32, refmat: *const i32, allow_hp: i32, out: *mut u8) -> u32;
    fn shim_write_sequence_header(s: *const i32, out: *mut u8) -> u32;
    fn shim_write_ext_tile_info(pre_bits: i32, rows: i32, cols: i32, out: *mut u8) -> u32;
    fn shim_write_color_config(c: *const i32, out: *mut u8) -> u32;
    fn shim_wb_uvlc(v: u32, out: *mut u8) -> u32;
    fn shim_write_timing_info(disp_tick: u32, time_scale: u32, equal_pic: i32, ticks_per_pic: u32, out: *mut u8) -> u32;
    fn shim_write_decoder_model_info(ed_delay_len: i32, dec_tick: u32, rem_time_len: i32, pres_time_len: i32, out: *mut u8) -> u32;
    fn shim_write_dec_model_op(dec_delay: u32, enc_delay: u32, low_delay: i32, delay_len: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_sequence_header_obu(top: *const i64, sh: *const i64, cc: *const i64, idc: *const i64, level: *const i64, tier: *const i64, dmpp: *const i64, dispp: *const i64, decdelay: *const i64, encdelay: *const i64, lowdelay: *const i64, initdelay: *const i64, out: *mut u8) -> u32;
    fn shim_write_frame_header_prefix(t: *const i64, op_dmpp: *const i64, op_idc: *const i64, brt: *const i64, ref_oh: *const i64, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_frame_size_with_refs(up_w: i32, up_h: i32, rw: i32, rh: i32, valid: *const i32, ycw: *const i32, ych: *const i32, rrw: *const i32, rrh: *const i32, enable_superres: i32, denom: i32, fs_num_bits_w: i32, fs_num_bits_h: i32, fs_up_w: i32, fs_up_h: i32, fs_scaling_active: i32, fs_rw: i32, fs_rh: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_ref_signaling(enable_order_hint: i32, short_sig: i32, ref_map_idx: *const i32, set_rfc: i32, rtc_reference: *const i32, rtc_ref_idx: *const i32, num_spatial_layers: i32, frame_id_present: i32, frame_id_len: i32, current_frame_id: i32, ref_frame_id: *const i32, diff_len: i32, out: *mut u8) -> u32;
    fn shim_write_refresh_frame_context(reduced: i32, disable_cdf: i32, rfc_disabled: i32, out: *mut u8) -> u32;
    fn shim_partition_cdf_length(bsize: i32) -> i32;
    fn shim_partition_gather_vert(out: *mut u16, cdf_in: *const u16, bsize: i32);
    fn shim_partition_gather_horz(out: *mut u16, cdf_in: *const u16, bsize: i32);
    fn shim_partition_plane_context(above: *const i8, left: *const i8, mi_row: i32, mi_col: i32, bsize: i32) -> i32;
    fn shim_write_partition(partition_cdf: *mut u16, cdf_len: i32, p: i32, has_rows: i32, has_cols: i32, bsize: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_skip_txfm_context(above_present: i32, above_skip: i32, left_present: i32, left_skip: i32) -> i32;
    fn shim_write_skip(skip_cdf: *mut u16, seg_skip_active: i32, skip_txfm: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_write_delta_qindex(delta_q_cdf: *mut u16, delta_qindex: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_write_delta_lflevel(delta_lf_cdf: *mut u16, delta_lflevel: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_write_cfl_alphas(cfl_sign_cdf: *mut u16, cfl_alpha_cdf: *mut u16, idx: i32, joint_sign: i32, out: *mut u8, out_sign_cdf: *mut u16, out_alpha_cdf: *mut u16) -> u32;
    fn shim_get_y_mode_ctx(above_present: i32, above_mode: i32, left_present: i32, left_mode: i32) -> i32;
    fn shim_write_intra_y_mode_kf(kf_y_cdf: *mut u16, mode: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_size_group_lookup(bsize: i32) -> i32;
    fn shim_write_intra_uv_mode(uv_mode_cdf: *mut u16, uv_mode: i32, cfl_allowed: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_write_angle_delta(cdf: *mut u16, angle_delta: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_mb_interp_filter(cdf0: *mut u16, cdf1: *mut u16, interp_needed: i32, is_switchable: i32, enable_dual: i32, f0: i32, f1: i32, out: *mut u8, out0: *mut u16, out1: *mut u16) -> u32;
    fn shim_get_intra_inter_context(has_above: i32, above_inter: i32, has_left: i32, left_inter: i32) -> i32;
    fn shim_get_skip_mode_context(ha: i32, a_sm: i32, hl: i32, l_sm: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_skip_mode(cdf: *mut u16, frame_flag: i32, seg_skip: i32, comp_allowed: i32, seg_ref_gmv: i32, skip_mode: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_reference_mode_context(ha: i32, a_r0: i32, a_r1: i32, a_ibc: i32, hl: i32, l_r0: i32, l_r1: i32, l_ibc: i32) -> i32;
    fn shim_single_ref_p1_context(ref_counts: *const u8) -> i32;
    fn shim_single_ref_p2_context(rc: *const u8) -> i32;
    fn shim_single_ref_p3_context(rc: *const u8) -> i32;
    fn shim_single_ref_p4_context(rc: *const u8) -> i32;
    fn shim_single_ref_p5_context(rc: *const u8) -> i32;
    fn shim_single_ref_p6_context(rc: *const u8) -> i32;
    fn shim_uni_comp_ref_p_context(rc: *const u8) -> i32;
    fn shim_uni_comp_ref_p1_context(rc: *const u8) -> i32;
    fn shim_uni_comp_ref_p2_context(rc: *const u8) -> i32;
    fn shim_neg_interleave(x: i32, ref_: i32, max: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_intrabc_info(intrabc_cdf: *mut u16, joints: *mut u16, comp0: *mut u16, comp1: *mut u16, use_intrabc: i32, diff_row: i32, diff_col: i32, out: *mut u8, out_ibc: *mut u16, out_joints: *mut u16, out_c0: *mut u16, out_c1: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_segment_id(cdf: *mut u16, seg_enabled: i32, update_map: i32, skip_txfm: i32, segment_id: i32, pred: i32, last_active_segid: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_txfm_partition_context(above: u8, left: u8, bsize: i32, tx_size: i32) -> i32;
    fn shim_txfm_partition_update(above_ctx: *mut u8, left_ctx: *mut u8, tx_size: i32, txb_size: i32);
    #[allow(clippy::too_many_arguments)]
    fn shim_write_tx_size_vartx(bsize: i32, top_tx_size: i32, inter_tx_size: *const u8, mb_to_right_edge: i32, mb_to_bottom_edge: i32, above_in: *const u8, left_in: *const u8, cdf: *mut u16, out: *mut u8, above_out: *mut u8, left_out: *mut u8, cdf_out: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_txfm_size(bsize: i32, max_tx: i32, inter_tx_size: *const u8, mb_to_right_edge: i32, mb_to_bottom_edge: i32, above_in: *const u8, left_in: *const u8, cdf: *mut u16, out: *mut u8, above_out: *mut u8, left_out: *mut u8, cdf_out: *mut u16) -> u32;
    fn shim_get_palette_bsize_ctx(bsize: i32) -> i32;
    fn shim_get_palette_mode_ctx(ha: i32, a_psize: i32, hl: i32, l_psize: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_palette_flags_sizes(mode_dc: i32, n_y: i32, y_mode_cdf: *mut u16, y_size_cdf: *mut u16, uv_dc: i32, n_uv: i32, uv_mode_cdf: *mut u16, uv_size_cdf: *mut u16, out: *mut u8, o_ym: *mut u16, o_ys: *mut u16, o_um: *mut u16, o_us: *mut u16) -> u32;
    fn shim_delta_encode_palette_colors(colors: *const i32, num: i32, bit_depth: i32, min_val: i32, out: *mut u8) -> u32;
    fn shim_pack_map_tokens(n: i32, num: i32, tokens: *const u8, color_ctxs: *const u8, map_cdf: *mut u16, out: *mut u8, map_cdf_out: *mut u16) -> u32;
    fn shim_write_palette_colors_v(colors_v: *const u16, n: i32, bit_depth: i32, out: *mut u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_palette_cache(plane: i32, mb_to_top_edge: i32, ha: i32, a_colors: *const u16, a_size0: i32, a_size1: i32, hl: i32, l_colors: *const u16, l_size0: i32, l_size1: i32, out_cache: *mut u16) -> i32;
    fn shim_index_color_cache(cache: *const u16, n_cache: i32, colors: *const u16, n_colors: i32, found: *mut u8, out_colors: *mut i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_palette_mode_info(mode_dc: i32, uv_dc: i32, bit_depth: i32, bsize_ctx: i32, y_mode_ctx: i32, uv_mode_ctx: i32, palette_size: *const u8, palette_colors: *const u16, mb_to_top_edge: i32, ha: i32, a_colors: *const u16, a_s0: i32, a_s1: i32, hl: i32, l_colors: *const u16, l_s0: i32, l_s1: i32, y_mode_cdf: *mut u16, y_size_cdf: *mut u16, uv_mode_cdf: *mut u16, uv_size_cdf: *mut u16, out: *mut u8, o_ym: *mut u16, o_ys: *mut u16, o_um: *mut u16, o_us: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_interintra_info(interintra: i32, ii_cdf: *mut u16, ii_mode: i32, ii_mode_cdf: *mut u16, wedge_used: i32, use_wedge: i32, wedge_ii_cdf: *mut u16, wedge_index: i32, wedge_idx_cdf: *mut u16, out: *mut u8, o_ii: *mut u16, o_iim: *mut u16, o_wii: *mut u16, o_wix: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_comp_group_idx_context(ha: i32, a_rf0: i32, a_rf1: i32, a_cgi: i32, hl: i32, l_rf0: i32, l_rf1: i32, l_cgi: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_compound_type_info(masked_used: i32, comp_group_idx: i32, cgi_cdf: *mut u16, dist_wtd: i32, compound_idx: i32, cidx_cdf: *mut u16, wedge_used: i32, comp_type: i32, ctype_cdf: *mut u16, wedge_index: i32, wedge_idx_cdf: *mut u16, wedge_sign: i32, mask_type: i32, out: *mut u8, o_cgi: *mut u16, o_cidx: *mut u16, o_ctype: *mut u16, o_wix: *mut u16) -> u32;
    fn shim_get_relative_dist(enable: i32, bits_minus_1: i32, a: i32, b: i32) -> i32;
    fn shim_get_pred_context_seg_id(ha: i32, a_sip: i32, hl: i32, l_sip: i32) -> i32;
    fn shim_is_inter_compound_mode(mode: i32) -> i32;
    fn shim_is_inter_singleref_mode(mode: i32) -> i32;
    fn shim_have_nearmv_in_inter_mode(mode: i32) -> i32;
    fn shim_mode_context_analyzer(rf0: i32, rf1: i32, mc_val: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_collect_neighbors_ref_counts(ha: i32, a_intrabc: i32, a_rf0: i32, a_rf1: i32, hl: i32, l_intrabc: i32, l_rf0: i32, l_rf1: i32, out_counts: *mut u8);
    fn shim_get_partition_subsize(bsize: i32, partition: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_update_ext_partition_context(mi_row: i32, mi_col: i32, subsize: i32, bsize: i32, partition: i32, above_in: *const i8, left_in: *const i8, above_out: *mut i8, left_out: *mut i8);
    #[allow(clippy::too_many_arguments)]
    fn shim_write_partition_node(above_in: *const i8, left_in: *const i8, mi_row: i32, mi_col: i32, bsize: i32, partition: i32, mi_rows: i32, mi_cols: i32, arena: *mut u16, out: *mut u8, above_out: *mut i8, left_out: *mut i8, arena_out: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_modes_sb(above_in: *const i8, left_in: *const i8, mi_row: i32, mi_col: i32, bsize: i32, tree: *const i8, tree_len: i32, arena: *mut u16, out: *mut u8, above_out: *mut i8, left_out: *mut i8, arena_out: *mut u16, tree_consumed: *mut i32) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_modes_tile(n_sb_rows: i32, n_sb_cols: i32, sb_mi: i32, sb_size: i32, tree: *const i8, arena: *mut u16, out: *mut u8, above_out: *mut i8, arena_out: *mut u16, tree_consumed: *mut i32) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_block_mvs(mode: i32, is_compound: i32, diff_row0: i32, diff_col0: i32, diff_row1: i32, diff_col1: i32, usehp: i32, joints: *mut u16, comp0: *mut u16, comp1: *mut u16, out: *mut u8, o_joints: *mut u16, o_c0: *mut u16, o_c1: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_mode_drl(seg_skip: i32, mode: i32, mode_ctx: i32, inter_compound_mode_cdf: *mut u16, newmv_cdf: *mut u16, zeromv_cdf: *mut u16, refmv_cdf: *mut u16, drl_cdf: *mut u16, ref_mv_idx: i32, ref_mv_count: i32, weight: *const u16, out: *mut u8, o_icm: *mut u16, o_newmv: *mut u16, o_zeromv: *mut u16, o_refmv: *mut u16, o_drl: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_mode_tail(interintra_allowed: i32, interintra: i32, ii_cdf: *mut u16, ii_mode: i32, ii_mode_cdf: *mut u16, wedge_used_ii: i32, use_wedge_ii: i32, wedge_ii_cdf: *mut u16, ii_wedge_index: i32, wedge_idx_cdf: *mut u16, motion_mode_present: i32, obmc_cdf: *mut u16, mm_cdf: *mut u16, last_motion_mode_allowed: i32, motion_mode: i32, has_second_ref: i32, masked_used: i32, comp_group_idx: i32, cgi_cdf: *mut u16, dist_wtd: i32, compound_idx: i32, cidx_cdf: *mut u16, wedge_used_ct: i32, comp_type: i32, ctype_cdf: *mut u16, ct_wedge_index: i32, wedge_sign: i32, mask_type: i32, interp_needed: i32, is_switchable: i32, enable_dual: i32, f0: i32, f1: i32, interp_cdf0: *mut u16, interp_cdf1: *mut u16, out: *mut u8, o_all: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_segment_id(update_map: i32, preskip: i32, segid_preskip: i32, skip: i32, temporal_update: i32, seg_id_predicted: i32, pred_cdf: *mut u16, seg_cdf: *mut u16, seg_enabled: i32, segment_id: i32, seg_pred: i32, last_active_segid: i32, out: *mut u8, o_predcdf: *mut u16, o_segcdf: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_prefix(update_map: i32, segid_preskip: i32, temporal_update: i32, seg_id_predicted: i32, pred_cdf: *mut u16, seg_cdf: *mut u16, seg_enabled: i32, segment_id: i32, seg_pred: i32, last_active_segid: i32, skip_mode_cdf: *mut u16, frame_skip_mode_flag: i32, sm_seg_skip: i32, sm_comp_allowed: i32, sm_seg_ref_gmv: i32, skip_mode: i32, skip_cdf: *mut u16, skip_seg_active: i32, skip_txfm: i32, coded_lossless: i32, allow_intrabc: i32, mi_row: i32, mi_col: i32, mib_size: i32, sb_size: i32, cdef_trans_in: *const i32, cdef_bits: i32, cdef_strength: i32, dq_present: i32, dlf_present: i32, dlf_multi: i32, num_planes: i32, bsize: i32, cur_qindex: i32, cur_base_qindex: i32, dq_res: i32, mbmi_dlf: *const i32, xd_dlf_in: *const i32, mbmi_dlf_base: i32, xd_dlf_base_in: i32, dlf_res: i32, dq_cdf: *mut u16, dlf_multi_cdf: *mut u16, dlf_cdf: *mut u16, intra_inter_cdf: *mut u16, seg_ref_frame_active: i32, seg_globalmv_active: i32, is_inter: i32, out: *mut u8, out_skip: *mut i32, out_skip_mode: *mut i32, o_predcdf: *mut u16, o_segcdf: *mut u16, o_smcdf: *mut u16, o_skipcdf: *mut u16, o_cdef_trans: *mut i32, o_dqcdf: *mut u16, o_dlfmcdf: *mut u16, o_dlfcdf: *mut u16, o_base: *mut i32, o_xd_dlf: *mut i32, o_xd_dlf_base: *mut i32, o_iicdf: *mut u16) -> u32;
    fn shim_use_angle_delta(bsize: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_delta_q_params_sb(dq_present: i32, dlf_present: i32, dlf_multi: i32, num_planes: i32, bsize: i32, sb_size: i32, skip: i32, sbul: i32, cur_qindex: i32, cur_base_qindex: i32, dq_res: i32, mbmi_dlf: *const i32, xd_dlf_in: *const i32, mbmi_dlf_base: i32, xd_dlf_base_in: i32, dlf_res: i32, dq_cdf: *mut u16, dlf_multi_cdf: *mut u16, dlf_cdf: *mut u16, out: *mut u8, o_dqcdf: *mut u16, o_dlfmcdf: *mut u16, o_dlfcdf: *mut u16, o_base: *mut i32, o_xd_dlf: *mut i32, o_xd_dlf_base: *mut i32) -> u32;
    fn shim_is_directional_mode(mode: i32) -> i32;
    fn shim_get_uv_mode(uv_mode: i32) -> i32;
    fn shim_allow_palette(allow_sct: i32, bsize: i32) -> i32;
    fn shim_is_cfl_allowed(bsize: i32, seg_id: i32, lossless: i32, ssx: i32, ssy: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_cdef(coded_lossless: i32, allow_intrabc: i32, mi_row: i32, mi_col: i32, mib_size: i32, sb_size: i32, skip: i32, transmitted_in: *const i32, cdef_bits: i32, cdef_strength: i32, out: *mut u8, transmitted_out: *mut i32) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_mb_modes_kf_prefix(segid_preskip: i32, seg_enabled: i32, update_map: i32, segment_id: i32, seg_pred: i32, last_active_segid: i32, seg_cdf: *mut u16, seg_skip_active: i32, skip_txfm: i32, skip_cdf: *mut u16, coded_lossless: i32, allow_intrabc: i32, mi_row: i32, mi_col: i32, mib_size: i32, sb_size: i32, cdef_trans_in: *const i32, cdef_bits: i32, cdef_strength: i32, dq_present: i32, dlf_present: i32, dlf_multi: i32, num_planes: i32, bsize: i32, cur_qindex: i32, cur_base_qindex: i32, dq_res: i32, mbmi_dlf: *const i32, xd_dlf_in: *const i32, mbmi_dlf_base: i32, xd_dlf_base_in: i32, dlf_res: i32, dq_cdf: *mut u16, dlf_multi_cdf: *mut u16, dlf_cdf: *mut u16, out: *mut u8, out_skip: *mut i32, o_segcdf: *mut u16, o_skipcdf: *mut u16, o_cdef_trans: *mut i32, o_dqcdf: *mut u16, o_dlfmcdf: *mut u16, o_dlfcdf: *mut u16, o_base: *mut i32, o_xd_dlf: *mut i32, o_xd_dlf_base: *mut i32) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_kf_tail(allow_intrabc: i32, intrabc_cdf: *mut u16, joints: *mut u16, comp0: *mut u16, comp1: *mut u16, use_intrabc: i32, diff_row: i32, diff_col: i32, mode: i32, bsize: i32, y_cdf: *mut u16, angle_delta_y: i32, y_angle_cdf: *mut u16, monochrome: i32, is_chroma_ref: i32, uv_mode: i32, cfl_allowed: i32, cfl_idx: i32, cfl_joint_sign: i32, angle_delta_uv: i32, uv_mode_cdf: *mut u16, cfl_sign_cdf: *mut u16, cfl_alpha_cdf: *mut u16, uv_angle_cdf: *mut u16, allow_palette: i32, bit_depth: i32, palette_size: *const u8, palette_colors: *const u16, mb_to_top_edge: i32, ha: i32, a_colors: *const u16, a_s0: i32, a_s1: i32, hl: i32, l_colors: *const u16, l_s0: i32, l_s1: i32, pal_y_mode_cdf: *mut u16, pal_y_size_cdf: *mut u16, pal_uv_mode_cdf: *mut u16, pal_uv_size_cdf: *mut u16, filter_allowed: i32, use_filter_intra: i32, filter_intra_mode: i32, fi_use_cdf: *mut u16, fi_mode_cdf: *mut u16, out: *mut u8, o_intrabc: *mut u16, o_joints: *mut u16, o_c0: *mut u16, o_c1: *mut u16, o_all: *mut u16) -> u32;
    fn shim_write_intra_y_and_angle(mode: i32, bsize: i32, y_cdf: *mut u16, angle_delta_y: i32, y_angle_cdf: *mut u16, out: *mut u8, o_ycdf: *mut u16, o_acdf: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_intra_uv_and_angle(monochrome: i32, is_chroma_ref: i32, uv_mode: i32, cfl_allowed: i32, bsize: i32, cfl_idx: i32, cfl_joint_sign: i32, angle_delta_uv: i32, uv_mode_cdf: *mut u16, cfl_sign_cdf: *mut u16, cfl_alpha_cdf: *mut u16, uv_angle_cdf: *mut u16, out: *mut u8, o_uvcdf: *mut u16, o_signcdf: *mut u16, o_alphacdf: *mut u16, o_uvacdf: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_intra_pred_modes(mode: i32, bsize: i32, y_cdf: *mut u16, angle_delta_y: i32, y_angle_cdf: *mut u16, monochrome: i32, is_chroma_ref: i32, uv_mode: i32, cfl_allowed: i32, cfl_idx: i32, cfl_joint_sign: i32, angle_delta_uv: i32, uv_mode_cdf: *mut u16, cfl_sign_cdf: *mut u16, cfl_alpha_cdf: *mut u16, uv_angle_cdf: *mut u16, allow_palette: i32, bit_depth: i32, palette_size: *const u8, palette_colors: *const u16, mb_to_top_edge: i32, ha: i32, a_colors: *const u16, a_s0: i32, a_s1: i32, hl: i32, l_colors: *const u16, l_s0: i32, l_s1: i32, pal_y_mode_cdf: *mut u16, pal_y_size_cdf: *mut u16, pal_uv_mode_cdf: *mut u16, pal_uv_size_cdf: *mut u16, filter_allowed: i32, use_filter_intra: i32, filter_intra_mode: i32, fi_use_cdf: *mut u16, fi_mode_cdf: *mut u16, out: *mut u8, o_all: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_comp_index_context(enable: i32, bits_minus_1: i32, cur_order_hint: i32, fwd_order_hint: i32, bck_order_hint: i32, ha: i32, a_has2: i32, a_cidx: i32, a_rf0: i32, hl: i32, l_has2: i32, l_cidx: i32, l_rf0: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_ref_frames(cdfs: *mut u16, seg_ref: i32, seg_skipgmv: i32, rmode_select: i32, comp_allowed: i32, is_compound: i32, comp_ref_type: i32, ref0: i32, ref1: i32, out: *mut u8, out_cdfs: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_comp_reference_type_context(ha: i32, a_r0: i32, a_r1: i32, a_ibc: i32, hl: i32, l_r0: i32, l_r1: i32, l_ibc: i32) -> i32;
    fn shim_write_motion_mode(obmc_cdf: *mut u16, mm_cdf: *mut u16, last_allowed: i32, mm: i32, out: *mut u8, out_obmc: *mut u16, out_mm: *mut u16) -> u32;
    fn shim_write_inter_compound_mode(cdf: *mut u16, mode: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_write_is_inter(cdf: *mut u16, seg_ref: i32, seg_gmv: i32, is_inter: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_write_filter_intra(use_cdf: *mut u16, mode_cdf: *mut u16, allowed: i32, use_fi: i32, mode: i32, out: *mut u8, out_use: *mut u16, out_mode: *mut u16) -> u32;
    fn shim_bsize_to_max_depth(bsize: i32) -> i32;
    fn shim_bsize_to_tx_size_cat(bsize: i32) -> i32;
    fn shim_write_selected_tx_size(cdf: *mut u16, bsize: i32, depth: i32, max_depths: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    fn shim_get_mv_joint(row: i32, col: i32) -> i32;
    fn shim_get_mv_class(z: i32) -> i32;
    fn shim_encode_mv_component(cdf: *mut u16, comp: i32, precision: i32, out: *mut u8, out_cdf: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_mv(joints_cdf: *mut u16, comp0: *mut u16, comp1: *mut u16, diff_row: i32, diff_col: i32, usehp: i32, out: *mut u8, out_joints: *mut u16, out_comp0: *mut u16, out_comp1: *mut u16) -> u32;
    fn shim_write_drl_idx(drl_cdf: *mut u16, mode: i32, ref_mv_idx: i32, ref_mv_count: i32, weight: *const u16, out: *mut u8, out_cdf: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_mode(newmv_cdf: *mut u16, zeromv_cdf: *mut u16, refmv_cdf: *mut u16, mode: i32, mode_ctx: i32, out: *mut u8, out_newmv: *mut u16, out_zeromv: *mut u16, out_refmv: *mut u16) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_frame_header_trailing_flags(intra_only: i32, ref_mode_select: i32, skip_allowed: i32, skip_flag: i32, might_warp: i32, allow_warp: i32, reduced_tx_set: i32, out: *mut u8) -> u32;
}

/// Reference `partition_cdf_length`.
pub fn ref_partition_cdf_length(bsize: i32) -> i32 {
    unsafe { shim_partition_cdf_length(bsize) }
}

/// Reference `av1_encode_mv` (joint + 2 components over the pristine C od_ec + real helpers).
#[allow(clippy::type_complexity)]
pub fn ref_encode_mv(joints_cdf: &[u16; 5], comp0: &[u16; 69], comp1: &[u16; 69], diff_row: i32, diff_col: i32, usehp: i32) -> (Vec<u8>, [u16; 5], [u16; 69], [u16; 69]) {
    let mut jc = *joints_cdf; let mut c0 = *comp0; let mut c1 = *comp1;
    let mut out = vec![0u8; 48];
    let mut oj = [0u16; 5]; let mut o0 = [0u16; 69]; let mut o1 = [0u16; 69];
    let n = unsafe { shim_encode_mv(jc.as_mut_ptr(), c0.as_mut_ptr(), c1.as_mut_ptr(), diff_row, diff_col, usehp, out.as_mut_ptr(), oj.as_mut_ptr(), o0.as_mut_ptr(), o1.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, oj, o0, o1)
}

/// Reference `encode_mv_component` (over the pristine C od_ec + the real av1_get_mv_class).
pub fn ref_encode_mv_component(cdf: &[u16; 69], comp: i32, precision: i32) -> (Vec<u8>, [u16; 69]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 32];
    let mut out_cdf = [0u16; 69];
    let n = unsafe { shim_encode_mv_component(c.as_mut_ptr(), comp, precision, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_mb_interp_filter`.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_mb_interp_filter(cdf0: &[u16; 4], cdf1: &[u16; 4], interp_needed: bool, is_switchable: bool, enable_dual: bool, f0: i32, f1: i32) -> (Vec<u8>, [u16; 4], [u16; 4]) {
    let mut c0 = *cdf0; let mut c1 = *cdf1; let mut out = vec![0u8; 16]; let mut o0 = [0u16; 4]; let mut o1 = [0u16; 4];
    let n = unsafe { shim_write_mb_interp_filter(c0.as_mut_ptr(), c1.as_mut_ptr(), interp_needed as i32, is_switchable as i32, enable_dual as i32, f0, f1, out.as_mut_ptr(), o0.as_mut_ptr(), o1.as_mut_ptr()) };
    out.truncate(n as usize); (out, o0, o1)
}

/// Reference `av1_get_pred_context_single_ref_p2` (brfarf2_or_arf).
pub fn ref_single_ref_p2_context(rc: &[u8; 8]) -> i32 { unsafe { shim_single_ref_p2_context(rc.as_ptr()) } }

/// Reference `av1_get_pred_context_single_ref_p3` (ll2_or_l3gld).
pub fn ref_single_ref_p3_context(rc: &[u8; 8]) -> i32 { unsafe { shim_single_ref_p3_context(rc.as_ptr()) } }

/// Reference `av1_get_pred_context_single_ref_p4` (last_or_last2).
pub fn ref_single_ref_p4_context(rc: &[u8; 8]) -> i32 { unsafe { shim_single_ref_p4_context(rc.as_ptr()) } }

/// Reference `av1_get_pred_context_single_ref_p5` (last3_or_gld).
pub fn ref_single_ref_p5_context(rc: &[u8; 8]) -> i32 { unsafe { shim_single_ref_p5_context(rc.as_ptr()) } }

/// Reference `av1_get_pred_context_single_ref_p6` (brf_or_arf2).
pub fn ref_single_ref_p6_context(rc: &[u8; 8]) -> i32 { unsafe { shim_single_ref_p6_context(rc.as_ptr()) } }

/// Reference `write_ref_frames` (cascade over the pristine C od_ec + update_cdf).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_ref_frames(cdfs: &[u16; 48], seg_ref: bool, seg_skipgmv: bool, rmode_select: bool, comp_allowed: bool, is_compound: bool, comp_ref_type: i32, ref0: i32, ref1: i32) -> (Vec<u8>, [u16; 48]) {
    let mut c = *cdfs; let mut out = vec![0u8; 32]; let mut oc = [0u16; 48];
    let n = unsafe { shim_write_ref_frames(c.as_mut_ptr(), seg_ref as i32, seg_skipgmv as i32, rmode_select as i32, comp_allowed as i32, is_compound as i32, comp_ref_type, ref0, ref1, out.as_mut_ptr(), oc.as_mut_ptr()) };
    out.truncate(n as usize); (out, oc)
}

/// Reference `write_intrabc_info` (flag + av1_encode_dv over the pristine C od_ec).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_intrabc_info(intrabc_cdf: &[u16; 3], joints: &[u16; 5], comp0: &[u16; 69], comp1: &[u16; 69], use_intrabc: i32, diff_row: i32, diff_col: i32) -> (Vec<u8>, [u16; 3], [u16; 5], [u16; 69], [u16; 69]) {
    let mut ib = *intrabc_cdf; let mut jc = *joints; let mut c0 = *comp0; let mut c1 = *comp1;
    let mut out = vec![0u8; 48]; let mut oib = [0u16; 3]; let mut oj = [0u16; 5]; let mut o0 = [0u16; 69]; let mut o1 = [0u16; 69];
    let n = unsafe { shim_write_intrabc_info(ib.as_mut_ptr(), jc.as_mut_ptr(), c0.as_mut_ptr(), c1.as_mut_ptr(), use_intrabc, diff_row, diff_col, out.as_mut_ptr(), oib.as_mut_ptr(), oj.as_mut_ptr(), o0.as_mut_ptr(), o1.as_mut_ptr()) };
    out.truncate(n as usize); (out, oib, oj, o0, o1)
}

/// Reference `av1_neg_interleave` (real exported fn).
pub fn ref_neg_interleave(x: i32, ref_: i32, max: i32) -> i32 { unsafe { shim_neg_interleave(x, ref_, max) } }

/// Reference `write_segment_id` (over the pristine C od_ec + update_cdf).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_segment_id(cdf: &[u16; 9], seg_enabled: bool, update_map: bool, skip_txfm: bool, segment_id: i32, pred: i32, last_active_segid: i32) -> (Vec<u8>, [u16; 9]) {
    let mut c = *cdf; let mut out = vec![0u8; 16]; let mut oc = [0u16; 9];
    let n = unsafe { shim_write_segment_id(c.as_mut_ptr(), seg_enabled as i32, update_map as i32, skip_txfm as i32, segment_id, pred, last_active_segid, out.as_mut_ptr(), oc.as_mut_ptr()) };
    out.truncate(n as usize); (out, oc)
}

/// Reference the 3 uni-comp-ref contexts (facades over the real exported fns).
pub fn ref_uni_comp_ref_p_context(rc: &[u8; 8]) -> i32 { unsafe { shim_uni_comp_ref_p_context(rc.as_ptr()) } }
pub fn ref_uni_comp_ref_p1_context(rc: &[u8; 8]) -> i32 { unsafe { shim_uni_comp_ref_p1_context(rc.as_ptr()) } }
pub fn ref_uni_comp_ref_p2_context(rc: &[u8; 8]) -> i32 { unsafe { shim_uni_comp_ref_p2_context(rc.as_ptr()) } }

/// Reference `av1_get_pred_context_single_ref_p1` (facade over the real exported fn).
pub fn ref_single_ref_p1_context(ref_counts: &[u8; 8]) -> i32 {
    unsafe { shim_single_ref_p1_context(ref_counts.as_ptr()) }
}

/// Reference `av1_get_comp_reference_type_context` (facade over the real exported fn).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_comp_reference_type_context(ha: bool, a_r0: i32, a_r1: i32, a_ibc: bool, hl: bool, l_r0: i32, l_r1: i32, l_ibc: bool) -> i32 {
    unsafe { shim_get_comp_reference_type_context(ha as i32, a_r0, a_r1, a_ibc as i32, hl as i32, l_r0, l_r1, l_ibc as i32) }
}

/// Reference `av1_get_reference_mode_context` (facade over the real exported fn).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_reference_mode_context(ha: bool, a_r0: i32, a_r1: i32, a_ibc: bool, hl: bool, l_r0: i32, l_r1: i32, l_ibc: bool) -> i32 {
    unsafe { shim_get_reference_mode_context(ha as i32, a_r0, a_r1, a_ibc as i32, hl as i32, l_r0, l_r1, l_ibc as i32) }
}

/// Reference `av1_get_skip_mode_context` (facade over the real fn).
pub fn ref_get_skip_mode_context(ha: bool, a_sm: i32, hl: bool, l_sm: i32) -> i32 {
    unsafe { shim_get_skip_mode_context(ha as i32, a_sm, hl as i32, l_sm) }
}

/// Reference `write_skip_mode` (over the pristine C od_ec + update_cdf).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_skip_mode(cdf: &[u16; 3], frame_flag: bool, seg_skip: bool, comp_allowed: bool, seg_ref_gmv: bool, skip_mode: i32) -> (Vec<u8>, [u16; 3]) {
    let mut c = *cdf; let mut out = vec![0u8; 16]; let mut oc = [0u16; 3];
    let n = unsafe { shim_write_skip_mode(c.as_mut_ptr(), frame_flag as i32, seg_skip as i32, comp_allowed as i32, seg_ref_gmv as i32, skip_mode, out.as_mut_ptr(), oc.as_mut_ptr()) };
    out.truncate(n as usize); (out, oc)
}

/// Reference `txfm_partition_context` (static inline, av1_common_int.h).
pub fn ref_txfm_partition_context(above: u8, left: u8, bsize: i32, tx_size: i32) -> i32 {
    unsafe { shim_txfm_partition_context(above, left, bsize, tx_size) }
}

/// Reference `txfm_partition_update` — fills above[0..bw]=txw, left[0..bh]=txh in place.
pub fn ref_txfm_partition_update(above: &mut [u8], left: &mut [u8], tx_size: i32, txb_size: i32) {
    unsafe { shim_txfm_partition_update(above.as_mut_ptr(), left.as_mut_ptr(), tx_size, txb_size) };
}

/// Reference `write_tx_size_vartx` (the recursion, over the pristine C od_ec + real
/// libaom helpers). Returns (bytes, above_ctx[32], left_ctx[32], adapted_cdf[21][3]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_tx_size_vartx(
    bsize: i32,
    top_tx_size: i32,
    inter_tx_size: &[u8; 16],
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    above_in: &[u8; 32],
    left_in: &[u8; 32],
    cdf: &[u16; 63],
) -> (Vec<u8>, [u8; 32], [u8; 32], [u16; 63]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 64];
    let mut ao = [0u8; 32];
    let mut lo = [0u8; 32];
    let mut co = [0u16; 63];
    let n = unsafe {
        shim_write_tx_size_vartx(
            bsize, top_tx_size, inter_tx_size.as_ptr(), mb_to_right_edge, mb_to_bottom_edge,
            above_in.as_ptr(), left_in.as_ptr(), c.as_mut_ptr(), out.as_mut_ptr(),
            ao.as_mut_ptr(), lo.as_mut_ptr(), co.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ao, lo, co)
}

/// Reference block-level inter var-tx-size loop (write_modes_b portion, over pristine C
/// od_ec). Returns (bytes, above_ctx[32], left_ctx[32], adapted_cdf[21][3]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_txfm_size(
    bsize: i32, max_tx: i32, inter_tx_size: &[u8; 16], mb_to_right_edge: i32, mb_to_bottom_edge: i32,
    above_in: &[u8; 32], left_in: &[u8; 32], cdf: &[u16; 63],
) -> (Vec<u8>, [u8; 32], [u8; 32], [u16; 63]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 128];
    let mut ao = [0u8; 32];
    let mut lo = [0u8; 32];
    let mut co = [0u16; 63];
    let n = unsafe {
        shim_write_inter_txfm_size(bsize, max_tx, inter_tx_size.as_ptr(), mb_to_right_edge,
            mb_to_bottom_edge, above_in.as_ptr(), left_in.as_ptr(), c.as_mut_ptr(), out.as_mut_ptr(),
            ao.as_mut_ptr(), lo.as_mut_ptr(), co.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, ao, lo, co)
}

/// Reference `av1_get_palette_bsize_ctx` (static inline, pred_common.h).
pub fn ref_get_palette_bsize_ctx(bsize: i32) -> i32 {
    unsafe { shim_get_palette_bsize_ctx(bsize) }
}

/// Reference `av1_get_palette_mode_ctx` (facade over the real static inline).
pub fn ref_get_palette_mode_ctx(ha: bool, a_psize: i32, hl: bool, l_psize: i32) -> i32 {
    unsafe { shim_get_palette_mode_ctx(ha as i32, a_psize, hl as i32, l_psize) }
}

/// Reference palette Y/UV mode+size symbol coding (write_palette_mode_info minus the
/// colour payload). Returns (bytes, y_mode_cdf[3], y_size_cdf[8], uv_mode_cdf[3],
/// uv_size_cdf[8]) all adapted.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_palette_flags_sizes(
    mode_dc: bool,
    n_y: i32,
    y_mode_cdf: &[u16; 3],
    y_size_cdf: &[u16; 8],
    uv_dc: bool,
    n_uv: i32,
    uv_mode_cdf: &[u16; 3],
    uv_size_cdf: &[u16; 8],
) -> (Vec<u8>, [u16; 3], [u16; 8], [u16; 3], [u16; 8]) {
    let (mut ym, mut ys, mut um, mut us) = (*y_mode_cdf, *y_size_cdf, *uv_mode_cdf, *uv_size_cdf);
    let mut out = vec![0u8; 32];
    let (mut oym, mut oys, mut oum, mut ous) = ([0u16; 3], [0u16; 8], [0u16; 3], [0u16; 8]);
    let n = unsafe {
        shim_write_palette_flags_sizes(
            mode_dc as i32, n_y, ym.as_mut_ptr(), ys.as_mut_ptr(), uv_dc as i32, n_uv,
            um.as_mut_ptr(), us.as_mut_ptr(), out.as_mut_ptr(), oym.as_mut_ptr(),
            oys.as_mut_ptr(), oum.as_mut_ptr(), ous.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oym, oys, oum, ous)
}

/// Reference `pack_map_tokens` (palette colour-index map, over pristine C od_ec). map_cdf
/// is the [PALETTE_COLOR_INDEX_CONTEXTS=5][9] slice for the palette size. Returns
/// (bytes, adapted map_cdf[45]).
pub fn ref_pack_map_tokens(n: i32, tokens: &[u8], color_ctxs: &[u8], map_cdf: &[u16; 45]) -> (Vec<u8>, [u16; 45]) {
    let mut mc = *map_cdf;
    let mut out = vec![0u8; 256];
    let mut mco = [0u16; 45];
    let num = tokens.len() as i32;
    let n_out = unsafe {
        shim_pack_map_tokens(n, num, tokens.as_ptr(), color_ctxs.as_ptr(), mc.as_mut_ptr(),
            out.as_mut_ptr(), mco.as_mut_ptr())
    };
    out.truncate(n_out as usize);
    (out, mco)
}

/// Reference `delta_encode_palette_colors` (over pristine C od_ec, real aom_ceil_log2).
pub fn ref_delta_encode_palette_colors(colors: &[i32], bit_depth: i32, min_val: i32) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_delta_encode_palette_colors(colors.as_ptr(), colors.len() as i32, bit_depth, min_val, out.as_mut_ptr())
    };
    out.truncate(n as usize);
    out
}

/// Reference V-channel palette colour coding (write_palette_colors_uv's V portion,
/// over pristine C od_ec, real av1_get_palette_delta_bits_v). colors_v need not be sorted.
pub fn ref_write_palette_colors_v(colors_v: &[u16], bit_depth: i32) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_write_palette_colors_v(colors_v.as_ptr(), colors_v.len() as i32, bit_depth, out.as_mut_ptr())
    };
    out.truncate(n as usize);
    out
}

/// Reference interintra sub-symbols (write_mbmi_b portion, over pristine C od_ec). CDFs
/// are pre-selected. Returns (bytes, ii_cdf[3], ii_mode_cdf[5], wedge_ii_cdf[3],
/// wedge_idx_cdf[17]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_interintra_info(
    interintra: i32, ii_cdf: &[u16; 3], ii_mode: i32, ii_mode_cdf: &[u16; 5],
    wedge_used: bool, use_wedge: i32, wedge_ii_cdf: &[u16; 3], wedge_index: i32,
    wedge_idx_cdf: &[u16; 17],
) -> (Vec<u8>, [u16; 3], [u16; 5], [u16; 3], [u16; 17]) {
    let (mut ii, mut iim, mut wii, mut wix) = (*ii_cdf, *ii_mode_cdf, *wedge_ii_cdf, *wedge_idx_cdf);
    let mut out = vec![0u8; 32];
    let (mut oii, mut oiim, mut owii, mut owix) = ([0u16; 3], [0u16; 5], [0u16; 3], [0u16; 17]);
    let n = unsafe {
        shim_write_interintra_info(
            interintra, ii.as_mut_ptr(), ii_mode, iim.as_mut_ptr(), wedge_used as i32, use_wedge,
            wii.as_mut_ptr(), wedge_index, wix.as_mut_ptr(), out.as_mut_ptr(),
            oii.as_mut_ptr(), oiim.as_mut_ptr(), owii.as_mut_ptr(), owix.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oii, oiim, owii, owix)
}

/// Reference `get_comp_group_idx_context` (facade over the real static inline).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_comp_group_idx_context(
    ha: bool, a_rf0: i32, a_rf1: i32, a_cgi: i32,
    hl: bool, l_rf0: i32, l_rf1: i32, l_cgi: i32,
) -> i32 {
    unsafe { shim_get_comp_group_idx_context(ha as i32, a_rf0, a_rf1, a_cgi, hl as i32, l_rf0, l_rf1, l_cgi) }
}

/// Reference `get_relative_dist` (static inline, mvref_common.h).
pub fn ref_get_relative_dist(enable: bool, bits_minus_1: i32, a: i32, b: i32) -> i32 {
    unsafe { shim_get_relative_dist(enable as i32, bits_minus_1, a, b) }
}

/// Reference `av1_get_pred_context_seg_id` (facade): above+left seg_id_predicted -> {0,1,2}.
pub fn ref_get_pred_context_seg_id(ha: bool, a_sip: i32, hl: bool, l_sip: i32) -> i32 {
    unsafe { shim_get_pred_context_seg_id(ha as i32, a_sip, hl as i32, l_sip) }
}

/// Reference `is_inter_compound_mode` / `is_inter_singleref_mode` /
/// `have_nearmv_in_inter_mode` / `av1_mode_context_analyzer` (facade over the real fn).
pub fn ref_is_inter_compound_mode(mode: i32) -> bool { unsafe { shim_is_inter_compound_mode(mode) != 0 } }
pub fn ref_is_inter_singleref_mode(mode: i32) -> bool { unsafe { shim_is_inter_singleref_mode(mode) != 0 } }
pub fn ref_have_nearmv_in_inter_mode(mode: i32) -> bool { unsafe { shim_have_nearmv_in_inter_mode(mode) != 0 } }
pub fn ref_mode_context_analyzer(rf0: i32, rf1: i32, mc_val: i32) -> i32 {
    unsafe { shim_mode_context_analyzer(rf0, rf1, mc_val) }
}

/// Reference `av1_collect_neighbors_ref_counts` (facade): the 8-entry ref-frame tally
/// from the above/left inter neighbours.
#[allow(clippy::too_many_arguments)]
pub fn ref_collect_neighbors_ref_counts(
    ha: bool, a_intrabc: bool, a_rf0: i32, a_rf1: i32,
    hl: bool, l_intrabc: bool, l_rf0: i32, l_rf1: i32,
) -> [u8; 8] {
    let mut counts = [0u8; 8];
    unsafe {
        shim_collect_neighbors_ref_counts(ha as i32, a_intrabc as i32, a_rf0, a_rf1, hl as i32,
            l_intrabc as i32, l_rf0, l_rf1, counts.as_mut_ptr())
    };
    counts
}

/// Reference `get_partition_subsize` (static inline, common_data.h).
pub fn ref_get_partition_subsize(bsize: i32, partition: i32) -> i32 {
    unsafe { shim_get_partition_subsize(bsize, partition) }
}

/// Reference `write_modes` tile loop (partition-only, stubbed blocks) over an
/// n_sb_rows x n_sb_cols grid of SBs. Returns (bytes, above[128], arena[220], consumed).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_modes_tile(
    n_sb_rows: i32, n_sb_cols: i32, sb_mi: i32, sb_size: i32, tree: &[i8], arena: &[u16; 220],
) -> (Vec<u8>, [i8; 128], [u16; 220], i32) {
    let mut ar = *arena;
    let mut out = vec![0u8; 512];
    let (mut ao, mut aro) = ([0i8; 128], [0u16; 220]);
    let mut consumed = 0i32;
    let n = unsafe {
        shim_write_modes_tile(n_sb_rows, n_sb_cols, sb_mi, sb_size, tree.as_ptr(), ar.as_mut_ptr(),
            out.as_mut_ptr(), ao.as_mut_ptr(), aro.as_mut_ptr(), &mut consumed)
    };
    out.truncate(n as usize);
    (out, ao, aro, consumed)
}

/// Reference `write_modes_sb` partition-tree recursion (fully-in-frame, stubbed blocks).
/// tree is the pre-order partition sequence. Returns (bytes, above[64], left[32],
/// arena[220], tree_consumed).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_modes_sb(
    above_in: &[i8; 64], left_in: &[i8; 32], mi_row: i32, mi_col: i32, bsize: i32, tree: &[i8],
    arena: &[u16; 220],
) -> (Vec<u8>, [i8; 64], [i8; 32], [u16; 220], i32) {
    let mut ar = *arena;
    let mut out = vec![0u8; 256];
    let (mut ao, mut lo, mut aro) = ([0i8; 64], [0i8; 32], [0u16; 220]);
    let mut consumed = 0i32;
    let n = unsafe {
        shim_write_modes_sb(above_in.as_ptr(), left_in.as_ptr(), mi_row, mi_col, bsize,
            tree.as_ptr(), tree.len() as i32, ar.as_mut_ptr(), out.as_mut_ptr(), ao.as_mut_ptr(),
            lo.as_mut_ptr(), aro.as_mut_ptr(), &mut consumed)
    };
    out.truncate(n as usize);
    (out, ao, lo, aro, consumed)
}

/// Reference `write_modes_sb` per-node partition step (context-select -> write_partition
/// -> context-update, over one od_ec). arena is [PARTITION_CONTEXTS=20][11] = 220 flat.
/// Returns (bytes, above[64], left[32], arena[220]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_partition_node(
    above_in: &[i8; 64], left_in: &[i8; 32], mi_row: i32, mi_col: i32, bsize: i32, partition: i32,
    mi_rows: i32, mi_cols: i32, arena: &[u16; 220],
) -> (Vec<u8>, [i8; 64], [i8; 32], [u16; 220]) {
    let mut ar = *arena;
    let mut out = vec![0u8; 16];
    let (mut ao, mut lo, mut aro) = ([0i8; 64], [0i8; 32], [0u16; 220]);
    let n = unsafe {
        shim_write_partition_node(above_in.as_ptr(), left_in.as_ptr(), mi_row, mi_col, bsize,
            partition, mi_rows, mi_cols, ar.as_mut_ptr(), out.as_mut_ptr(), ao.as_mut_ptr(),
            lo.as_mut_ptr(), aro.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, ao, lo, aro)
}

/// Reference `update_ext_partition_context` (facade). above is a 64-slot buffer, left a
/// 32-slot (MAX_MIB_SIZE) buffer. Returns the updated (above[64], left[32]).
#[allow(clippy::too_many_arguments)]
pub fn ref_update_ext_partition_context(
    mi_row: i32, mi_col: i32, subsize: i32, bsize: i32, partition: i32,
    above_in: &[i8; 64], left_in: &[i8; 32],
) -> ([i8; 64], [i8; 32]) {
    let (mut ao, mut lo) = ([0i8; 64], [0i8; 32]);
    unsafe {
        shim_update_ext_partition_context(mi_row, mi_col, subsize, bsize, partition,
            above_in.as_ptr(), left_in.as_ptr(), ao.as_mut_ptr(), lo.as_mut_ptr())
    };
    (ao, lo)
}

/// Reference inter-block MV coding (the mode-dependent av1_encode_mv calls, over pristine
/// C od_ec). Returns (bytes, joints[5], comp0[69], comp1[69]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_block_mvs(
    mode: i32, is_compound: bool, diff_row0: i32, diff_col0: i32, diff_row1: i32, diff_col1: i32,
    usehp: i32, joints: &[u16; 5], comp0: &[u16; 69], comp1: &[u16; 69],
) -> (Vec<u8>, [u16; 5], [u16; 69], [u16; 69]) {
    let (mut jo, mut c0, mut c1) = (*joints, *comp0, *comp1);
    let mut out = vec![0u8; 64];
    let (mut ojo, mut oc0, mut oc1) = ([0u16; 5], [0u16; 69], [0u16; 69]);
    let n = unsafe {
        shim_write_inter_block_mvs(mode, is_compound as i32, diff_row0, diff_col0, diff_row1,
            diff_col1, usehp, jo.as_mut_ptr(), c0.as_mut_ptr(), c1.as_mut_ptr(), out.as_mut_ptr(),
            ojo.as_mut_ptr(), oc0.as_mut_ptr(), oc1.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, ojo, oc0, oc1)
}

/// Reference inter mode + drl coding (over pristine C od_ec). inter_compound_mode_cdf is
/// pre-selected [mode_ctx] (9 entries); newmv/refmv are [6][3], zeromv [2][3], drl [3][3].
/// Returns (bytes, icm[9], newmv[18], zeromv[6], refmv[18], drl[9]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_inter_mode_drl(
    seg_skip: bool, mode: i32, mode_ctx: i32, icm_cdf: &[u16; 9], newmv_cdf: &[u16; 18],
    zeromv_cdf: &[u16; 6], refmv_cdf: &[u16; 18], drl_cdf: &[u16; 9], ref_mv_idx: i32,
    ref_mv_count: i32, weight: &[u16],
) -> (Vec<u8>, [u16; 9], [u16; 18], [u16; 6], [u16; 18], [u16; 9]) {
    let (mut icm, mut nm, mut zm, mut rm, mut drl) = (*icm_cdf, *newmv_cdf, *zeromv_cdf, *refmv_cdf, *drl_cdf);
    let mut out = vec![0u8; 32];
    let (mut oicm, mut onm, mut ozm, mut orm, mut odrl) = ([0u16; 9], [0u16; 18], [0u16; 6], [0u16; 18], [0u16; 9]);
    let n = unsafe {
        shim_write_inter_mode_drl(seg_skip as i32, mode, mode_ctx, icm.as_mut_ptr(), nm.as_mut_ptr(),
            zm.as_mut_ptr(), rm.as_mut_ptr(), drl.as_mut_ptr(), ref_mv_idx, ref_mv_count,
            weight.as_ptr(), out.as_mut_ptr(), oicm.as_mut_ptr(), onm.as_mut_ptr(), ozm.as_mut_ptr(),
            orm.as_mut_ptr(), odrl.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, oicm, onm, ozm, orm, odrl)
}

/// Inputs for the inter mode-body tail oracle (interintra + motion_mode + compound_type
/// + interp_filter). interintra/compound_type share the one wedge_idx_cdf.
pub struct InterTailRef<'a> {
    pub interintra_allowed: bool,
    pub interintra: i32,
    pub ii_cdf: &'a [u16; 3],
    pub ii_mode: i32,
    pub ii_mode_cdf: &'a [u16; 5],
    pub wedge_used_ii: bool,
    pub use_wedge_ii: i32,
    pub wedge_ii_cdf: &'a [u16; 3],
    pub ii_wedge_index: i32,
    pub wedge_idx_cdf: &'a [u16; 17],
    pub motion_mode_present: bool,
    pub obmc_cdf: &'a [u16; 3],
    pub mm_cdf: &'a [u16; 4],
    pub last_motion_mode_allowed: i32,
    pub motion_mode: i32,
    pub has_second_ref: bool,
    pub masked_used: bool,
    pub comp_group_idx: i32,
    pub cgi_cdf: &'a [u16; 3],
    pub dist_wtd: bool,
    pub compound_idx: i32,
    pub cidx_cdf: &'a [u16; 3],
    pub wedge_used_ct: bool,
    pub comp_type: i32,
    pub ctype_cdf: &'a [u16; 3],
    pub ct_wedge_index: i32,
    pub wedge_sign: i32,
    pub mask_type: i32,
    pub interp_needed: bool,
    pub is_switchable: bool,
    pub enable_dual: bool,
    pub f0: i32,
    pub f1: i32,
    pub interp_cdf0: &'a [u16; 4],
    pub interp_cdf1: &'a [u16; 4],
}

/// Reference inter mode-body tail over one od_ec. Returns (bytes, all adapted CDFs packed:
/// ii[3] ii_mode[5] wedge_ii[3] wedge_idx[17] obmc[3] mm[4] cgi[3] cidx[3] ctype[3]
/// interp0[4] interp1[4] = 52).
pub fn ref_write_inter_mode_tail(inp: &InterTailRef) -> (Vec<u8>, [u16; 52]) {
    let mut ii = *inp.ii_cdf;
    let mut iim = *inp.ii_mode_cdf;
    let mut wii = *inp.wedge_ii_cdf;
    let mut wix = *inp.wedge_idx_cdf;
    let mut obmc = *inp.obmc_cdf;
    let mut mm = *inp.mm_cdf;
    let mut cgi = *inp.cgi_cdf;
    let mut cidx = *inp.cidx_cdf;
    let mut ct = *inp.ctype_cdf;
    let mut ic0 = *inp.interp_cdf0;
    let mut ic1 = *inp.interp_cdf1;
    let mut out = vec![0u8; 64];
    let mut o_all = [0u16; 52];
    let n = unsafe {
        shim_write_inter_mode_tail(
            inp.interintra_allowed as i32, inp.interintra, ii.as_mut_ptr(), inp.ii_mode,
            iim.as_mut_ptr(), inp.wedge_used_ii as i32, inp.use_wedge_ii, wii.as_mut_ptr(),
            inp.ii_wedge_index, wix.as_mut_ptr(), inp.motion_mode_present as i32, obmc.as_mut_ptr(),
            mm.as_mut_ptr(), inp.last_motion_mode_allowed, inp.motion_mode, inp.has_second_ref as i32,
            inp.masked_used as i32, inp.comp_group_idx, cgi.as_mut_ptr(), inp.dist_wtd as i32,
            inp.compound_idx, cidx.as_mut_ptr(), inp.wedge_used_ct as i32, inp.comp_type,
            ct.as_mut_ptr(), inp.ct_wedge_index, inp.wedge_sign, inp.mask_type, inp.interp_needed as i32,
            inp.is_switchable as i32, inp.enable_dual as i32, inp.f0, inp.f1, ic0.as_mut_ptr(),
            ic1.as_mut_ptr(), out.as_mut_ptr(), o_all.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, o_all)
}

/// Inputs for the `pack_inter_mode_mvs` prefix oracle (inter_segment_id -> skip_mode ->
/// skip -> inter_segment_id -> cdef -> delta_q -> is_inter).
pub struct InterPrefixRef<'a> {
    pub update_map: bool,
    pub segid_preskip: bool,
    pub temporal_update: bool,
    pub seg_id_predicted: i32,
    pub pred_cdf: &'a [u16; 3],
    pub seg_cdf: &'a [u16; 9],
    pub seg_enabled: bool,
    pub segment_id: i32,
    pub seg_pred: i32,
    pub last_active_segid: i32,
    pub skip_mode_cdf: &'a [u16; 3],
    pub frame_skip_mode_flag: bool,
    pub sm_seg_skip: bool,
    pub sm_comp_allowed: bool,
    pub sm_seg_ref_gmv: bool,
    pub skip_mode: i32,
    pub skip_cdf: &'a [u16; 3],
    pub skip_seg_active: bool,
    pub skip_txfm: i32,
    pub coded_lossless: bool,
    pub allow_intrabc: bool,
    pub mi_row: i32,
    pub mi_col: i32,
    pub mib_size: i32,
    pub sb_size: i32,
    pub cdef_trans: &'a [i32; 4],
    pub cdef_bits: i32,
    pub cdef_strength: i32,
    pub dq_present: bool,
    pub dlf_present: bool,
    pub dlf_multi: bool,
    pub num_planes: i32,
    pub bsize: i32,
    pub cur_qindex: i32,
    pub cur_base_qindex: i32,
    pub dq_res: i32,
    pub mbmi_dlf: &'a [i32; 4],
    pub xd_dlf: &'a [i32; 4],
    pub mbmi_dlf_base: i32,
    pub xd_dlf_base: i32,
    pub dlf_res: i32,
    pub dq_cdf: &'a [u16; 5],
    pub dlf_multi_cdf: &'a [u16; 20],
    pub dlf_cdf: &'a [u16; 5],
    pub intra_inter_cdf: &'a [u16; 3],
    pub seg_ref_frame_active: bool,
    pub seg_globalmv_active: bool,
    pub is_inter: i32,
}

/// Outputs of the inter-prefix oracle.
pub struct InterPrefixOut {
    pub bytes: Vec<u8>,
    pub skip: i32,
    pub skip_mode: i32,
    pub pred_cdf: [u16; 3],
    pub seg_cdf: [u16; 9],
    pub skip_mode_cdf: [u16; 3],
    pub skip_cdf: [u16; 3],
    pub cdef_trans: [i32; 4],
    pub dq_cdf: [u16; 5],
    pub dlf_multi_cdf: [u16; 20],
    pub dlf_cdf: [u16; 5],
    pub base_qindex: i32,
    pub xd_dlf: [i32; 4],
    pub xd_dlf_base: i32,
    pub intra_inter_cdf: [u16; 3],
}

/// Reference `pack_inter_mode_mvs` prefix over one od_ec.
pub fn ref_write_inter_prefix(inp: &InterPrefixRef) -> InterPrefixOut {
    let mut pc = *inp.pred_cdf;
    let mut sc = *inp.seg_cdf;
    let mut smc = *inp.skip_mode_cdf;
    let mut skc = *inp.skip_cdf;
    let mut dqc = *inp.dq_cdf;
    let mut dlmc = *inp.dlf_multi_cdf;
    let mut dlc = *inp.dlf_cdf;
    let mut iic = *inp.intra_inter_cdf;
    let mut out = vec![0u8; 64];
    let (mut skip, mut skip_mode) = (0i32, 0i32);
    let (mut opc, mut osc, mut osmc, mut oskc, mut octr) = ([0u16; 3], [0u16; 9], [0u16; 3], [0u16; 3], [0i32; 4]);
    let (mut odqc, mut odlmc, mut odlc) = ([0u16; 5], [0u16; 20], [0u16; 5]);
    let (mut obase, mut oxd, mut oxdb, mut oiic) = (0i32, [0i32; 4], 0i32, [0u16; 3]);
    let n = unsafe {
        shim_write_inter_prefix(
            inp.update_map as i32, inp.segid_preskip as i32, inp.temporal_update as i32,
            inp.seg_id_predicted, pc.as_mut_ptr(), sc.as_mut_ptr(), inp.seg_enabled as i32,
            inp.segment_id, inp.seg_pred, inp.last_active_segid, smc.as_mut_ptr(),
            inp.frame_skip_mode_flag as i32, inp.sm_seg_skip as i32, inp.sm_comp_allowed as i32,
            inp.sm_seg_ref_gmv as i32, inp.skip_mode, skc.as_mut_ptr(), inp.skip_seg_active as i32,
            inp.skip_txfm, inp.coded_lossless as i32, inp.allow_intrabc as i32, inp.mi_row,
            inp.mi_col, inp.mib_size, inp.sb_size, inp.cdef_trans.as_ptr(), inp.cdef_bits,
            inp.cdef_strength, inp.dq_present as i32, inp.dlf_present as i32, inp.dlf_multi as i32,
            inp.num_planes, inp.bsize, inp.cur_qindex, inp.cur_base_qindex, inp.dq_res,
            inp.mbmi_dlf.as_ptr(), inp.xd_dlf.as_ptr(), inp.mbmi_dlf_base, inp.xd_dlf_base,
            inp.dlf_res, dqc.as_mut_ptr(), dlmc.as_mut_ptr(), dlc.as_mut_ptr(), iic.as_mut_ptr(),
            inp.seg_ref_frame_active as i32, inp.seg_globalmv_active as i32, inp.is_inter,
            out.as_mut_ptr(), &mut skip, &mut skip_mode, opc.as_mut_ptr(), osc.as_mut_ptr(),
            osmc.as_mut_ptr(), oskc.as_mut_ptr(), octr.as_mut_ptr(), odqc.as_mut_ptr(),
            odlmc.as_mut_ptr(), odlc.as_mut_ptr(), &mut obase, oxd.as_mut_ptr(), &mut oxdb,
            oiic.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    InterPrefixOut {
        bytes: out, skip, skip_mode, pred_cdf: opc, seg_cdf: osc, skip_mode_cdf: osmc,
        skip_cdf: oskc, cdef_trans: octr, dq_cdf: odqc, dlf_multi_cdf: odlmc, dlf_cdf: odlc,
        base_qindex: obase, xd_dlf: oxd, xd_dlf_base: oxdb, intra_inter_cdf: oiic,
    }
}

/// Reference `write_inter_segment_id` (bitstream.c:920, over pristine C od_ec). Returns
/// (bytes, pred_cdf[3], seg_cdf[9]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_segment_id(
    update_map: bool, preskip: bool, segid_preskip: bool, skip: bool, temporal_update: bool,
    seg_id_predicted: i32, pred_cdf: &[u16; 3], seg_cdf: &[u16; 9], seg_enabled: bool,
    segment_id: i32, seg_pred: i32, last_active_segid: i32,
) -> (Vec<u8>, [u16; 3], [u16; 9]) {
    let (mut pc, mut sc) = (*pred_cdf, *seg_cdf);
    let mut out = vec![0u8; 16];
    let (mut opc, mut osc) = ([0u16; 3], [0u16; 9]);
    let n = unsafe {
        shim_write_inter_segment_id(update_map as i32, preskip as i32, segid_preskip as i32,
            skip as i32, temporal_update as i32, seg_id_predicted, pc.as_mut_ptr(), sc.as_mut_ptr(),
            seg_enabled as i32, segment_id, seg_pred, last_active_segid, out.as_mut_ptr(),
            opc.as_mut_ptr(), osc.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, opc, osc)
}

/// Reference per-superblock `write_delta_q_params` (bitstream.c:960, the mode-info
/// driver — distinct from the header delta-q-config writer). Over pristine C od_ec.
/// Returns (bytes, dq_cdf[5], dlf_multi_cdf[20], dlf_cdf[5], new_base_qindex,
/// new_xd_delta_lf[4], new_xd_delta_lf_from_base).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_delta_q_params_sb(
    dq_present: bool, dlf_present: bool, dlf_multi: bool, num_planes: i32,
    bsize: i32, sb_size: i32, skip: i32, sbul: bool, cur_qindex: i32, cur_base_qindex: i32,
    dq_res: i32, mbmi_dlf: &[i32; 4], xd_dlf: &[i32; 4], mbmi_dlf_base: i32, xd_dlf_base: i32,
    dlf_res: i32, dq_cdf: &[u16; 5], dlf_multi_cdf: &[u16; 20], dlf_cdf: &[u16; 5],
) -> (Vec<u8>, [u16; 5], [u16; 20], [u16; 5], i32, [i32; 4], i32) {
    let (mut dqc, mut dlmc, mut dlc) = (*dq_cdf, *dlf_multi_cdf, *dlf_cdf);
    let mut out = vec![0u8; 64];
    let (mut odqc, mut odlmc, mut odlc) = ([0u16; 5], [0u16; 20], [0u16; 5]);
    let (mut o_base, mut o_xd_dlf, mut o_xd_dlf_base) = (0i32, [0i32; 4], 0i32);
    let n = unsafe {
        shim_write_delta_q_params_sb(
            dq_present as i32, dlf_present as i32, dlf_multi as i32, num_planes, bsize, sb_size,
            skip, sbul as i32, cur_qindex, cur_base_qindex, dq_res, mbmi_dlf.as_ptr(),
            xd_dlf.as_ptr(), mbmi_dlf_base, xd_dlf_base, dlf_res, dqc.as_mut_ptr(),
            dlmc.as_mut_ptr(), dlc.as_mut_ptr(), out.as_mut_ptr(), odqc.as_mut_ptr(),
            odlmc.as_mut_ptr(), odlc.as_mut_ptr(), &mut o_base, o_xd_dlf.as_mut_ptr(),
            &mut o_xd_dlf_base,
        )
    };
    out.truncate(n as usize);
    (out, odqc, odlmc, odlc, o_base, o_xd_dlf, o_xd_dlf_base)
}

/// Reference `write_cdef` (mode-info driver, over pristine C od_ec). Returns (bytes,
/// updated cdef_transmitted[4]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_cdef(
    coded_lossless: bool, allow_intrabc: bool, mi_row: i32, mi_col: i32, mib_size: i32,
    sb_size: i32, skip: i32, transmitted: &[i32; 4], cdef_bits: i32, cdef_strength: i32,
) -> (Vec<u8>, [i32; 4]) {
    let mut out = vec![0u8; 8];
    let mut tout = [0i32; 4];
    let n = unsafe {
        shim_write_cdef(coded_lossless as i32, allow_intrabc as i32, mi_row, mi_col, mib_size,
            sb_size, skip, transmitted.as_ptr(), cdef_bits, cdef_strength, out.as_mut_ptr(),
            tout.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, tout)
}

/// Inputs for the `write_mb_modes_kf` prefix oracle (segment_id -> skip -> segment_id
/// -> cdef -> delta_q_params).
pub struct KfPrefixRef<'a> {
    pub segid_preskip: bool,
    pub seg_enabled: bool,
    pub update_map: bool,
    pub segment_id: i32,
    pub seg_pred: i32,
    pub last_active_segid: i32,
    pub seg_cdf: &'a [u16; 9],
    pub seg_skip_active: bool,
    pub skip_txfm: i32,
    pub skip_cdf: &'a [u16; 3],
    pub coded_lossless: bool,
    pub allow_intrabc: bool,
    pub mi_row: i32,
    pub mi_col: i32,
    pub mib_size: i32,
    pub sb_size: i32,
    pub cdef_trans: &'a [i32; 4],
    pub cdef_bits: i32,
    pub cdef_strength: i32,
    pub dq_present: bool,
    pub dlf_present: bool,
    pub dlf_multi: bool,
    pub num_planes: i32,
    pub bsize: i32,
    pub cur_qindex: i32,
    pub cur_base_qindex: i32,
    pub dq_res: i32,
    pub mbmi_dlf: &'a [i32; 4],
    pub xd_dlf: &'a [i32; 4],
    pub mbmi_dlf_base: i32,
    pub xd_dlf_base: i32,
    pub dlf_res: i32,
    pub dq_cdf: &'a [u16; 5],
    pub dlf_multi_cdf: &'a [u16; 20],
    pub dlf_cdf: &'a [u16; 5],
}

/// Outputs of the `write_mb_modes_kf` prefix oracle: coded bytes, the write_skip return,
/// every adapted CDF, and the updated cdef/delta-lf state.
pub struct KfPrefixOut {
    pub bytes: Vec<u8>,
    pub skip: i32,
    pub seg_cdf: [u16; 9],
    pub skip_cdf: [u16; 3],
    pub cdef_trans: [i32; 4],
    pub dq_cdf: [u16; 5],
    pub dlf_multi_cdf: [u16; 20],
    pub dlf_cdf: [u16; 5],
    pub base_qindex: i32,
    pub xd_dlf: [i32; 4],
    pub xd_dlf_base: i32,
}

/// Reference `write_mb_modes_kf` prefix (segment_id -> skip -> segment_id -> cdef ->
/// delta_q_params) over one od_ec, threading write_skip's return into cdef/delta_q.
pub fn ref_write_mb_modes_kf_prefix(inp: &KfPrefixRef) -> KfPrefixOut {
    let mut seg = *inp.seg_cdf;
    let mut skc = *inp.skip_cdf;
    let mut dqc = *inp.dq_cdf;
    let mut dlmc = *inp.dlf_multi_cdf;
    let mut dlc = *inp.dlf_cdf;
    let mut out = vec![0u8; 64];
    let mut skip = 0i32;
    let (mut oseg, mut oskc, mut octr) = ([0u16; 9], [0u16; 3], [0i32; 4]);
    let (mut odqc, mut odlmc, mut odlc) = ([0u16; 5], [0u16; 20], [0u16; 5]);
    let (mut obase, mut oxd, mut oxdb) = (0i32, [0i32; 4], 0i32);
    let n = unsafe {
        shim_write_mb_modes_kf_prefix(
            inp.segid_preskip as i32, inp.seg_enabled as i32, inp.update_map as i32, inp.segment_id,
            inp.seg_pred, inp.last_active_segid, seg.as_mut_ptr(), inp.seg_skip_active as i32,
            inp.skip_txfm, skc.as_mut_ptr(), inp.coded_lossless as i32, inp.allow_intrabc as i32,
            inp.mi_row, inp.mi_col, inp.mib_size, inp.sb_size, inp.cdef_trans.as_ptr(),
            inp.cdef_bits, inp.cdef_strength, inp.dq_present as i32, inp.dlf_present as i32,
            inp.dlf_multi as i32, inp.num_planes, inp.bsize, inp.cur_qindex, inp.cur_base_qindex,
            inp.dq_res, inp.mbmi_dlf.as_ptr(), inp.xd_dlf.as_ptr(), inp.mbmi_dlf_base,
            inp.xd_dlf_base, inp.dlf_res, dqc.as_mut_ptr(), dlmc.as_mut_ptr(), dlc.as_mut_ptr(),
            out.as_mut_ptr(), &mut skip, oseg.as_mut_ptr(), oskc.as_mut_ptr(), octr.as_mut_ptr(),
            odqc.as_mut_ptr(), odlmc.as_mut_ptr(), odlc.as_mut_ptr(), &mut obase, oxd.as_mut_ptr(),
            &mut oxdb,
        )
    };
    out.truncate(n as usize);
    KfPrefixOut {
        bytes: out, skip, seg_cdf: oseg, skip_cdf: oskc, cdef_trans: octr, dq_cdf: odqc,
        dlf_multi_cdf: odlmc, dlf_cdf: odlc, base_qindex: obase, xd_dlf: oxd, xd_dlf_base: oxdb,
    }
}

/// Reference `av1_use_angle_delta` / `av1_is_directional_mode` / `get_uv_mode` /
/// `av1_allow_palette` / `is_cfl_allowed` — the intra-prediction-mode driver gates.
pub fn ref_use_angle_delta(bsize: i32) -> bool { unsafe { shim_use_angle_delta(bsize) != 0 } }
pub fn ref_is_directional_mode(mode: i32) -> bool { unsafe { shim_is_directional_mode(mode) != 0 } }
pub fn ref_get_uv_mode(uv_mode: i32) -> i32 { unsafe { shim_get_uv_mode(uv_mode) } }
pub fn ref_allow_palette(allow_sct: bool, bsize: i32) -> bool {
    unsafe { shim_allow_palette(allow_sct as i32, bsize) != 0 }
}
pub fn ref_is_cfl_allowed(bsize: i32, seg_id: i32, lossless: bool, ssx: i32, ssy: i32) -> bool {
    unsafe { shim_is_cfl_allowed(bsize, seg_id, lossless as i32, ssx, ssy) != 0 }
}

/// Reference write_intra_prediction_modes piece 1 (Y mode + gated Y angle delta, over
/// pristine C od_ec + real gates). Returns (bytes, y_cdf[14], y_angle_cdf[8]).
pub fn ref_write_intra_y_and_angle(
    mode: i32, bsize: i32, y_cdf: &[u16; 14], angle_delta_y: i32, y_angle_cdf: &[u16; 8],
) -> (Vec<u8>, [u16; 14], [u16; 8]) {
    let (mut yc, mut ac) = (*y_cdf, *y_angle_cdf);
    let mut out = vec![0u8; 16];
    let (mut oyc, mut oac) = ([0u16; 14], [0u16; 8]);
    let n = unsafe {
        shim_write_intra_y_and_angle(mode, bsize, yc.as_mut_ptr(), angle_delta_y, ac.as_mut_ptr(),
            out.as_mut_ptr(), oyc.as_mut_ptr(), oac.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, oyc, oac)
}

/// Reference write_intra_prediction_modes piece 2 (UV mode + cfl + gated UV angle, over
/// pristine C od_ec). Returns (bytes, uv_mode_cdf[15], cfl_sign_cdf[9],
/// cfl_alpha_cdf[102], uv_angle_cdf[8]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_intra_uv_and_angle(
    monochrome: bool, is_chroma_ref: bool, uv_mode: i32, cfl_allowed: bool, bsize: i32,
    cfl_idx: i32, cfl_joint_sign: i32, angle_delta_uv: i32,
    uv_mode_cdf: &[u16; 15], cfl_sign_cdf: &[u16; 9], cfl_alpha_cdf: &[u16; 102], uv_angle_cdf: &[u16; 8],
) -> (Vec<u8>, [u16; 15], [u16; 9], [u16; 102], [u16; 8]) {
    let (mut uc, mut sc, mut ac, mut uac) = (*uv_mode_cdf, *cfl_sign_cdf, *cfl_alpha_cdf, *uv_angle_cdf);
    let mut out = vec![0u8; 32];
    let (mut ouc, mut osc, mut oac, mut ouac) = ([0u16; 15], [0u16; 9], [0u16; 102], [0u16; 8]);
    let n = unsafe {
        shim_write_intra_uv_and_angle(
            monochrome as i32, is_chroma_ref as i32, uv_mode, cfl_allowed as i32, bsize, cfl_idx,
            cfl_joint_sign, angle_delta_uv, uc.as_mut_ptr(), sc.as_mut_ptr(), ac.as_mut_ptr(),
            uac.as_mut_ptr(), out.as_mut_ptr(), ouc.as_mut_ptr(), osc.as_mut_ptr(), oac.as_mut_ptr(),
            ouac.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ouc, osc, oac, ouac)
}

/// Inputs for the full `write_intra_prediction_modes` oracle. CDFs are the caller's
/// context-selected slices; palette neighbour arrays use the full 3*PALETTE_MAX_SIZE layout.
pub struct IntraPredModesRef<'a> {
    pub mode: i32,
    pub bsize: i32,
    pub y_cdf: &'a [u16; 14],
    pub angle_delta_y: i32,
    pub y_angle_cdf: &'a [u16; 8],
    pub monochrome: bool,
    pub is_chroma_ref: bool,
    pub uv_mode: i32,
    pub cfl_allowed: bool,
    pub cfl_idx: i32,
    pub cfl_joint_sign: i32,
    pub angle_delta_uv: i32,
    pub uv_mode_cdf: &'a [u16; 15],
    pub cfl_sign_cdf: &'a [u16; 9],
    pub cfl_alpha_cdf: &'a [u16; 102],
    pub uv_angle_cdf: &'a [u16; 8],
    pub allow_palette: bool,
    pub bit_depth: i32,
    pub palette_size: &'a [u8; 2],
    pub palette_colors: &'a [u16; 24],
    pub mb_to_top_edge: i32,
    pub ha: bool,
    pub a_colors: &'a [u16; 24],
    pub a_size: &'a [i32; 2],
    pub hl: bool,
    pub l_colors: &'a [u16; 24],
    pub l_size: &'a [i32; 2],
    pub pal_y_mode_cdf: &'a [u16; 3],
    pub pal_y_size_cdf: &'a [u16; 8],
    pub pal_uv_mode_cdf: &'a [u16; 3],
    pub pal_uv_size_cdf: &'a [u16; 8],
    pub filter_allowed: bool,
    pub use_filter_intra: i32,
    pub filter_intra_mode: i32,
    pub fi_use_cdf: &'a [u16; 3],
    pub fi_mode_cdf: &'a [u16; 6],
}

/// Reference full `write_intra_prediction_modes` over one od_ec. Returns (bytes, all
/// adapted CDFs packed: y[14] y_angle[8] uv[15] cfl_sign[9] cfl_alpha[102] uv_angle[8]
/// pal_y_mode[3] pal_y_size[8] pal_uv_mode[3] pal_uv_size[8] fi_use[3] fi_mode[6] = 187).
pub fn ref_write_intra_pred_modes(inp: &IntraPredModesRef) -> (Vec<u8>, [u16; 187]) {
    let mut yc = *inp.y_cdf;
    let mut yac = *inp.y_angle_cdf;
    let mut uc = *inp.uv_mode_cdf;
    let mut sc = *inp.cfl_sign_cdf;
    let mut ac = *inp.cfl_alpha_cdf;
    let mut uac = *inp.uv_angle_cdf;
    let mut pym = *inp.pal_y_mode_cdf;
    let mut pys = *inp.pal_y_size_cdf;
    let mut pum = *inp.pal_uv_mode_cdf;
    let mut pus = *inp.pal_uv_size_cdf;
    let mut fiu = *inp.fi_use_cdf;
    let mut fim = *inp.fi_mode_cdf;
    let mut out = vec![0u8; 128];
    let mut o_all = [0u16; 187];
    let n = unsafe {
        shim_write_intra_pred_modes(
            inp.mode, inp.bsize, yc.as_mut_ptr(), inp.angle_delta_y, yac.as_mut_ptr(),
            inp.monochrome as i32, inp.is_chroma_ref as i32, inp.uv_mode, inp.cfl_allowed as i32,
            inp.cfl_idx, inp.cfl_joint_sign, inp.angle_delta_uv, uc.as_mut_ptr(), sc.as_mut_ptr(),
            ac.as_mut_ptr(), uac.as_mut_ptr(), inp.allow_palette as i32, inp.bit_depth,
            inp.palette_size.as_ptr(), inp.palette_colors.as_ptr(), inp.mb_to_top_edge,
            inp.ha as i32, inp.a_colors.as_ptr(), inp.a_size[0], inp.a_size[1], inp.hl as i32,
            inp.l_colors.as_ptr(), inp.l_size[0], inp.l_size[1], pym.as_mut_ptr(), pys.as_mut_ptr(),
            pum.as_mut_ptr(), pus.as_mut_ptr(), inp.filter_allowed as i32, inp.use_filter_intra,
            inp.filter_intra_mode, fiu.as_mut_ptr(), fim.as_mut_ptr(), out.as_mut_ptr(),
            o_all.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, o_all)
}

/// Reference `write_mb_modes_kf` tail (intrabc + is_intrabc_block early-return + intra)
/// over one od_ec. `intra` supplies the write_intra_prediction_modes state. Returns
/// (bytes, intrabc_cdf[3], joints[5], mv_comp0[69], mv_comp1[69], intra CDFs o_all[187]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_kf_tail(
    allow_intrabc: bool, intrabc_cdf: &[u16; 3], joints: &[u16; 5], comp0: &[u16; 69],
    comp1: &[u16; 69], use_intrabc: bool, diff_row: i32, diff_col: i32, intra: &IntraPredModesRef,
) -> (Vec<u8>, [u16; 3], [u16; 5], [u16; 69], [u16; 69], [u16; 187]) {
    let (mut ib, mut jo, mut c0, mut c1) = (*intrabc_cdf, *joints, *comp0, *comp1);
    let mut yc = *intra.y_cdf;
    let mut yac = *intra.y_angle_cdf;
    let mut uc = *intra.uv_mode_cdf;
    let mut sc = *intra.cfl_sign_cdf;
    let mut ac = *intra.cfl_alpha_cdf;
    let mut uac = *intra.uv_angle_cdf;
    let mut pym = *intra.pal_y_mode_cdf;
    let mut pys = *intra.pal_y_size_cdf;
    let mut pum = *intra.pal_uv_mode_cdf;
    let mut pus = *intra.pal_uv_size_cdf;
    let mut fiu = *intra.fi_use_cdf;
    let mut fim = *intra.fi_mode_cdf;
    let mut out = vec![0u8; 128];
    let (mut oib, mut ojo, mut oc0, mut oc1) = ([0u16; 3], [0u16; 5], [0u16; 69], [0u16; 69]);
    let mut o_all = [0u16; 187];
    let n = unsafe {
        shim_kf_tail(
            allow_intrabc as i32, ib.as_mut_ptr(), jo.as_mut_ptr(), c0.as_mut_ptr(), c1.as_mut_ptr(),
            use_intrabc as i32, diff_row, diff_col, intra.mode, intra.bsize, yc.as_mut_ptr(),
            intra.angle_delta_y, yac.as_mut_ptr(), intra.monochrome as i32, intra.is_chroma_ref as i32,
            intra.uv_mode, intra.cfl_allowed as i32, intra.cfl_idx, intra.cfl_joint_sign,
            intra.angle_delta_uv, uc.as_mut_ptr(), sc.as_mut_ptr(), ac.as_mut_ptr(), uac.as_mut_ptr(),
            intra.allow_palette as i32, intra.bit_depth, intra.palette_size.as_ptr(),
            intra.palette_colors.as_ptr(), intra.mb_to_top_edge, intra.ha as i32,
            intra.a_colors.as_ptr(), intra.a_size[0], intra.a_size[1], intra.hl as i32,
            intra.l_colors.as_ptr(), intra.l_size[0], intra.l_size[1], pym.as_mut_ptr(),
            pys.as_mut_ptr(), pum.as_mut_ptr(), pus.as_mut_ptr(), intra.filter_allowed as i32,
            intra.use_filter_intra, intra.filter_intra_mode, fiu.as_mut_ptr(), fim.as_mut_ptr(),
            out.as_mut_ptr(), oib.as_mut_ptr(), ojo.as_mut_ptr(), oc0.as_mut_ptr(), oc1.as_mut_ptr(),
            o_all.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oib, ojo, oc0, oc1, o_all)
}

/// Reference `get_comp_index_context` (body transcribed; ref-buffer order hints passed
/// directly, real get_relative_dist + ctx arithmetic).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_comp_index_context(
    enable: bool, bits_minus_1: i32, cur: i32, fwd: i32, bck: i32,
    ha: bool, a_has2: bool, a_cidx: i32, a_rf0: i32,
    hl: bool, l_has2: bool, l_cidx: i32, l_rf0: i32,
) -> i32 {
    unsafe {
        shim_get_comp_index_context(enable as i32, bits_minus_1, cur, fwd, bck, ha as i32,
            a_has2 as i32, a_cidx, a_rf0, hl as i32, l_has2 as i32, l_cidx, l_rf0)
    }
}

/// Reference compound-type coding (write_mbmi_b portion, over pristine C od_ec). CDFs
/// pre-selected. Returns (bytes, cgi_cdf[3], cidx_cdf[3], ctype_cdf[3], wedge_idx[17]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_compound_type_info(
    masked_used: bool, comp_group_idx: i32, cgi_cdf: &[u16; 3],
    dist_wtd: bool, compound_idx: i32, cidx_cdf: &[u16; 3],
    wedge_used: bool, comp_type: i32, ctype_cdf: &[u16; 3],
    wedge_index: i32, wedge_idx_cdf: &[u16; 17], wedge_sign: i32, mask_type: i32,
) -> (Vec<u8>, [u16; 3], [u16; 3], [u16; 3], [u16; 17]) {
    let (mut cgi, mut cidx, mut ct, mut wix) = (*cgi_cdf, *cidx_cdf, *ctype_cdf, *wedge_idx_cdf);
    let mut out = vec![0u8; 32];
    let (mut ocgi, mut ocidx, mut oct, mut owix) = ([0u16; 3], [0u16; 3], [0u16; 3], [0u16; 17]);
    let n = unsafe {
        shim_write_compound_type_info(
            masked_used as i32, comp_group_idx, cgi.as_mut_ptr(), dist_wtd as i32, compound_idx,
            cidx.as_mut_ptr(), wedge_used as i32, comp_type, ct.as_mut_ptr(), wedge_index,
            wix.as_mut_ptr(), wedge_sign, mask_type, out.as_mut_ptr(),
            ocgi.as_mut_ptr(), ocidx.as_mut_ptr(), oct.as_mut_ptr(), owix.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ocgi, ocidx, oct, owix)
}

/// Reference `av1_get_palette_cache` (facade). Neighbour `palette_colors` are the full
/// 3*PALETTE_MAX_SIZE layout. Returns (cache[0..n], n).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_palette_cache(
    plane: i32, mb_to_top_edge: i32,
    ha: bool, a_colors: &[u16; 24], a_size0: i32, a_size1: i32,
    hl: bool, l_colors: &[u16; 24], l_size0: i32, l_size1: i32,
) -> (Vec<u16>, i32) {
    let mut cache = vec![0u16; 16];
    let n = unsafe {
        shim_get_palette_cache(plane, mb_to_top_edge, ha as i32, a_colors.as_ptr(), a_size0, a_size1,
            hl as i32, l_colors.as_ptr(), l_size0, l_size1, cache.as_mut_ptr())
    };
    cache.truncate(n as usize);
    (cache, n)
}

/// Reference `av1_index_color_cache` (exported). Returns (found[0..n_cache],
/// out_colors[0..n_out], n_out).
pub fn ref_index_color_cache(cache: &[u16], colors: &[u16]) -> (Vec<u8>, Vec<i32>, i32) {
    let mut found = vec![0u8; cache.len().max(1)];
    let mut out_colors = vec![0i32; colors.len().max(1)];
    let n = unsafe {
        shim_index_color_cache(cache.as_ptr(), cache.len() as i32, colors.as_ptr(),
            colors.len() as i32, found.as_mut_ptr(), out_colors.as_mut_ptr())
    };
    out_colors.truncate(n as usize);
    (found, out_colors, n)
}

/// Reference full `write_palette_mode_info` (flags + sizes + colours end-to-end, over
/// pristine C od_ec + the real cache/index/delta-bits fns). CDFs are pre-selected by
/// the caller. Returns (bytes, y_mode[3], y_size[8], uv_mode[3], uv_size[8]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_palette_mode_info(
    mode_dc: bool, uv_dc: bool, bit_depth: i32,
    palette_size: &[u8; 2], palette_colors: &[u16; 24],
    mb_to_top_edge: i32,
    ha: bool, a_colors: &[u16; 24], a_size: &[i32; 2],
    hl: bool, l_colors: &[u16; 24], l_size: &[i32; 2],
    y_mode_cdf: &[u16; 3], y_size_cdf: &[u16; 8],
    uv_mode_cdf: &[u16; 3], uv_size_cdf: &[u16; 8],
) -> (Vec<u8>, [u16; 3], [u16; 8], [u16; 3], [u16; 8]) {
    let (mut ym, mut ys, mut um, mut us) = (*y_mode_cdf, *y_size_cdf, *uv_mode_cdf, *uv_size_cdf);
    let mut out = vec![0u8; 128];
    let (mut oym, mut oys, mut oum, mut ous) = ([0u16; 3], [0u16; 8], [0u16; 3], [0u16; 8]);
    let n = unsafe {
        shim_write_palette_mode_info(
            mode_dc as i32, uv_dc as i32, bit_depth, 0, 0, 0,
            palette_size.as_ptr(), palette_colors.as_ptr(), mb_to_top_edge,
            ha as i32, a_colors.as_ptr(), a_size[0], a_size[1],
            hl as i32, l_colors.as_ptr(), l_size[0], l_size[1],
            ym.as_mut_ptr(), ys.as_mut_ptr(), um.as_mut_ptr(), us.as_mut_ptr(),
            out.as_mut_ptr(), oym.as_mut_ptr(), oys.as_mut_ptr(), oum.as_mut_ptr(), ous.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oym, oys, oum, ous)
}

/// Reference `av1_get_intra_inter_context` (facade over the real exported fn).
pub fn ref_get_intra_inter_context(has_above: bool, above_inter: bool, has_left: bool, left_inter: bool) -> i32 {
    unsafe { shim_get_intra_inter_context(has_above as i32, above_inter as i32, has_left as i32, left_inter as i32) }
}

/// Reference `write_motion_mode`.
pub fn ref_write_motion_mode(obmc_cdf: &[u16; 3], mm_cdf: &[u16; 4], last_allowed: i32, mm: i32) -> (Vec<u8>, [u16; 3], [u16; 4]) {
    let mut o = *obmc_cdf; let mut m = *mm_cdf; let mut out = vec![0u8; 16]; let mut oo = [0u16; 3]; let mut om = [0u16; 4];
    let n = unsafe { shim_write_motion_mode(o.as_mut_ptr(), m.as_mut_ptr(), last_allowed, mm, out.as_mut_ptr(), oo.as_mut_ptr(), om.as_mut_ptr()) };
    out.truncate(n as usize); (out, oo, om)
}

/// Reference `write_inter_compound_mode`.
pub fn ref_write_inter_compound_mode(cdf: &[u16; 9], mode: i32) -> (Vec<u8>, [u16; 9]) {
    let mut c = *cdf; let mut out = vec![0u8; 16]; let mut oc = [0u16; 9];
    let n = unsafe { shim_write_inter_compound_mode(c.as_mut_ptr(), mode, out.as_mut_ptr(), oc.as_mut_ptr()) };
    out.truncate(n as usize); (out, oc)
}

/// Reference `write_is_inter`.
pub fn ref_write_is_inter(cdf: &[u16; 3], seg_ref: bool, seg_gmv: bool, is_inter: i32) -> (Vec<u8>, [u16; 3]) {
    let mut c = *cdf; let mut out = vec![0u8; 16]; let mut oc = [0u16; 3];
    let n = unsafe { shim_write_is_inter(c.as_mut_ptr(), seg_ref as i32, seg_gmv as i32, is_inter, out.as_mut_ptr(), oc.as_mut_ptr()) };
    out.truncate(n as usize); (out, oc)
}

/// Reference `write_filter_intra_mode_info` (over the pristine C od_ec + update_cdf).
pub fn ref_write_filter_intra(use_cdf: &[u16; 3], mode_cdf: &[u16; 6], allowed: bool, use_fi: i32, mode: i32) -> (Vec<u8>, [u16; 3], [u16; 6]) {
    let mut u = *use_cdf; let mut m = *mode_cdf;
    let mut out = vec![0u8; 16];
    let mut ou = [0u16; 3]; let mut om = [0u16; 6];
    let n = unsafe { shim_write_filter_intra(u.as_mut_ptr(), m.as_mut_ptr(), allowed as i32, use_fi, mode, out.as_mut_ptr(), ou.as_mut_ptr(), om.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, ou, om)
}

/// Reference `bsize_to_max_depth` / `bsize_to_tx_size_cat`.
pub fn ref_bsize_to_max_depth(bsize: i32) -> usize { unsafe { shim_bsize_to_max_depth(bsize) as usize } }
pub fn ref_bsize_to_tx_size_cat(bsize: i32) -> i32 { unsafe { shim_bsize_to_tx_size_cat(bsize) } }

/// Reference `write_selected_tx_size` (over the pristine C od_ec + update_cdf).
pub fn ref_write_selected_tx_size(cdf: &[u16; 4], bsize: i32, depth: i32, max_depths: i32) -> (Vec<u8>, [u16; 4]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 4];
    let n = unsafe { shim_write_selected_tx_size(c.as_mut_ptr(), bsize, depth, max_depths, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_angle_delta` (over the pristine C od_ec + update_cdf).
pub fn ref_write_angle_delta(cdf: &[u16; 8], angle_delta: i32) -> (Vec<u8>, [u16; 8]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 8];
    let n = unsafe { shim_write_angle_delta(c.as_mut_ptr(), angle_delta, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `av1_get_mv_joint`.
pub fn ref_get_mv_joint(row: i32, col: i32) -> i32 {
    unsafe { shim_get_mv_joint(row, col) }
}

/// Reference `av1_get_mv_class` -> (class, offset).
pub fn ref_get_mv_class(z: i32) -> (i32, i32) {
    let v = unsafe { shim_get_mv_class(z) };
    (v >> 20, v & 0xFFFFF)
}

/// Reference `write_drl_idx` (over the pristine C od_ec + the real av1_drl_ctx).
pub fn ref_write_drl_idx(drl_cdf: &[u16; 9], mode: i32, ref_mv_idx: i32, ref_mv_count: i32, weight: &[u16; 4]) -> (Vec<u8>, [u16; 9]) {
    let mut cdf = *drl_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 9];
    let n = unsafe { shim_write_drl_idx(cdf.as_mut_ptr(), mode, ref_mv_idx, ref_mv_count, weight.as_ptr(), out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_inter_mode` (3-symbol cascade over the pristine C od_ec + update_cdf).
/// Returns coded bytes + the adapted newmv[18]/zeromv[6]/refmv[18] flat CDF arrays.
#[allow(clippy::type_complexity)]
pub fn ref_write_inter_mode(newmv: &[u16; 18], zeromv: &[u16; 6], refmv: &[u16; 18], mode: i32, mode_ctx: i32) -> (Vec<u8>, [u16; 18], [u16; 6], [u16; 18]) {
    let mut nm = *newmv; let mut zm = *zeromv; let mut rm = *refmv;
    let mut out = vec![0u8; 16];
    let mut onm = [0u16; 18]; let mut ozm = [0u16; 6]; let mut orm = [0u16; 18];
    let n = unsafe { shim_write_inter_mode(nm.as_mut_ptr(), zm.as_mut_ptr(), rm.as_mut_ptr(), mode, mode_ctx, out.as_mut_ptr(), onm.as_mut_ptr(), ozm.as_mut_ptr(), orm.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, onm, ozm, orm)
}

/// Reference `size_group_lookup[bsize]`.
pub fn ref_size_group_lookup(bsize: i32) -> usize {
    unsafe { shim_size_group_lookup(bsize) as usize }
}

/// Reference `write_intra_uv_mode` (transcribed symbol over the pristine C od_ec + update_cdf).
pub fn ref_write_intra_uv_mode(uv_mode_cdf: &[u16; 15], uv_mode: i32, cfl_allowed: bool) -> (Vec<u8>, [u16; 15]) {
    let mut cdf = *uv_mode_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 15];
    let n = unsafe { shim_write_intra_uv_mode(cdf.as_mut_ptr(), uv_mode, cfl_allowed as i32, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `get_y_mode_cdf` context (real intra_mode_context table + block-mode rule).
pub fn ref_get_y_mode_ctx(above_present: bool, above_mode: i32, left_present: bool, left_mode: i32) -> (usize, usize) {
    let v = unsafe { shim_get_y_mode_ctx(above_present as i32, above_mode, left_present as i32, left_mode) };
    ((v >> 8) as usize, (v & 0xff) as usize)
}

/// Reference `write_intra_y_mode_kf` (transcribed symbol over the pristine C od_ec + update_cdf).
pub fn ref_write_intra_y_mode_kf(kf_y_cdf: &[u16; 14], mode: i32) -> (Vec<u8>, [u16; 14]) {
    let mut cdf = *kf_y_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 14];
    let n = unsafe { shim_write_intra_y_mode_kf(cdf.as_mut_ptr(), mode, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_cfl_alphas` (transcribed over the pristine C od_ec + update_cdf).
/// Returns coded bytes, the adapted sign CDF (9), and the adapted alpha CDFs (6x17 flat).
pub fn ref_write_cfl_alphas(cfl_sign_cdf: &[u16; 9], cfl_alpha_cdf: &[u16; 102], idx: i32, joint_sign: i32) -> (Vec<u8>, [u16; 9], [u16; 102]) {
    let mut sc = *cfl_sign_cdf;
    let mut ac = *cfl_alpha_cdf;
    let mut out = vec![0u8; 16];
    let mut osc = [0u16; 9];
    let mut oac = [0u16; 102];
    let n = unsafe { shim_write_cfl_alphas(sc.as_mut_ptr(), ac.as_mut_ptr(), idx, joint_sign, out.as_mut_ptr(), osc.as_mut_ptr(), oac.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, osc, oac)
}

/// Reference `write_delta_lflevel` (transcribed over the pristine C od_ec + update_cdf).
pub fn ref_write_delta_lflevel(delta_lf_cdf: &[u16; 5], delta_lflevel: i32) -> (Vec<u8>, [u16; 5]) {
    let mut cdf = *delta_lf_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 5];
    let n = unsafe { shim_write_delta_lflevel(cdf.as_mut_ptr(), delta_lflevel, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_delta_qindex` (transcribed over the pristine C od_ec + update_cdf).
pub fn ref_write_delta_qindex(delta_q_cdf: &[u16; 5], delta_qindex: i32) -> (Vec<u8>, [u16; 5]) {
    let mut cdf = *delta_q_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 5];
    let n = unsafe { shim_write_delta_qindex(cdf.as_mut_ptr(), delta_qindex, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `av1_get_skip_txfm_context` (facade over the real static inline).
pub fn ref_skip_txfm_context(above_present: bool, above_skip: i32, left_present: bool, left_skip: i32) -> i32 {
    unsafe { shim_skip_txfm_context(above_present as i32, above_skip, left_present as i32, left_skip) }
}

/// Reference `write_skip` (transcribed symbol over the pristine C od_ec + update_cdf).
pub fn ref_write_skip(skip_cdf: &[u16; 3], seg_skip_active: bool, skip_txfm: i32) -> (Vec<u8>, [u16; 3]) {
    let mut cdf = *skip_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 3];
    let n = unsafe { shim_write_skip(cdf.as_mut_ptr(), seg_skip_active as i32, skip_txfm, out.as_mut_ptr(), out_cdf.as_mut_ptr()) };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_partition` (transcribed body over the pristine C od_ec + update_cdf).
/// Returns the coded bytes and the adapted partition CDF (`cdf_len+1` meaningful entries).
pub fn ref_write_partition(partition_cdf: &[u16; 11], cdf_len: i32, p: i32, has_rows: bool, has_cols: bool, bsize: i32) -> (Vec<u8>, [u16; 11]) {
    let mut cdf = *partition_cdf;
    let mut out = vec![0u8; 64];
    let mut out_cdf = [0u16; 11];
    let n = unsafe {
        shim_write_partition(cdf.as_mut_ptr(), cdf_len, p, has_rows as i32, has_cols as i32, bsize, out.as_mut_ptr(), out_cdf.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `partition_plane_context` (facade over the real static inline).
pub fn ref_partition_plane_context(above: &[i8], left: &[i8], mi_row: i32, mi_col: i32, bsize: i32) -> i32 {
    unsafe { shim_partition_plane_context(above.as_ptr(), left.as_ptr(), mi_row, mi_col, bsize) }
}

/// Reference `partition_gather_vert_alike`.
pub fn ref_partition_gather_vert(cdf_in: &[u16; 11], bsize: i32) -> [u16; 2] {
    let mut out = [0u16; 2];
    unsafe { shim_partition_gather_vert(out.as_mut_ptr(), cdf_in.as_ptr(), bsize) };
    out
}

/// Reference `partition_gather_horz_alike`.
pub fn ref_partition_gather_horz(cdf_in: &[u16; 11], bsize: i32) -> [u16; 2] {
    let mut out = [0u16; 2];
    unsafe { shim_partition_gather_horz(out.as_mut_ptr(), cdf_in.as_ptr(), bsize) };
    out
}

/// Reference refresh-frame-context bit.
pub fn ref_write_refresh_frame_context(reduced: bool, disable_cdf: bool, rfc_disabled: bool) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe { shim_write_refresh_frame_context(reduced as i32, disable_cdf as i32, rfc_disabled as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference frame-header trailing flags.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_frame_header_trailing_flags(intra_only: bool, ref_mode_select: bool, skip_allowed: bool, skip_flag: bool, might_warp: bool, allow_warp: bool, reduced_tx_set: bool) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe { shim_write_frame_header_trailing_flags(intra_only as i32, ref_mode_select as i32, skip_allowed as i32, skip_flag as i32, might_warp as i32, allow_warp as i32, reduced_tx_set as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference INTER/S-frame ref signaling (transcribed over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_ref_signaling(enable_order_hint: bool, short_sig: bool, ref_map_idx: &[i32; 7], set_rfc: bool, rtc_reference: &[i32; 7], rtc_ref_idx: &[i32; 7], num_spatial_layers: i32, frame_id_present: bool, frame_id_len: u32, current_frame_id: i32, ref_frame_id: &[i32; 8], diff_len: u32) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_write_inter_ref_signaling(enable_order_hint as i32, short_sig as i32, ref_map_idx.as_ptr(), set_rfc as i32, rtc_reference.as_ptr(), rtc_ref_idx.as_ptr(), num_spatial_layers, frame_id_present as i32, frame_id_len as i32, current_frame_id, ref_frame_id.as_ptr(), diff_len as i32, out.as_mut_ptr())
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_frame_size_with_refs` (transcribed over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_frame_size_with_refs(up_w: i32, up_h: i32, rw: i32, rh: i32, valid: &[i32; 7], ycw: &[i32; 7], ych: &[i32; 7], rrw: &[i32; 7], rrh: &[i32; 7], enable_superres: bool, denom: i32, fs_num_bits_w: u32, fs_num_bits_h: u32, fs_up_w: i32, fs_up_h: i32, fs_scaling_active: bool, fs_rw: i32, fs_rh: i32) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_write_frame_size_with_refs(up_w, up_h, rw, rh, valid.as_ptr(), ycw.as_ptr(), ych.as_ptr(), rrw.as_ptr(), rrh.as_ptr(), enable_superres as i32, denom, fs_num_bits_w as i32, fs_num_bits_h as i32, fs_up_w, fs_up_h, fs_scaling_active as i32, fs_rw, fs_rh, out.as_mut_ptr())
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_uncompressed_header_obu` prefix (transcribed over the real aom_wb).
pub fn ref_write_frame_header_prefix(t: &[i64; 34], op_dmpp: &[i64; 32], op_idc: &[i64; 32], brt: &[i64; 32], ref_oh: &[i64; 8]) -> Vec<u8> {
    let mut out = vec![0u8; 256];
    let n = unsafe { shim_write_frame_header_prefix(t.as_ptr(), op_dmpp.as_ptr(), op_idc.as_ptr(), brt.as_ptr(), ref_oh.as_ptr(), out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `av1_write_sequence_header_obu` — the REAL exported function, fed a
/// `SequenceHeader` populated from the packed params (direct oracle, not a
/// transcription).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_sequence_header_obu(top: &[i64; 16], sh: &[i64; 24], cc: &[i64; 11], idc: &[i64; 32], level: &[i64; 32], tier: &[i64; 32], dmpp: &[i64; 32], dispp: &[i64; 32], decdelay: &[i64; 32], encdelay: &[i64; 32], lowdelay: &[i64; 32], initdelay: &[i64; 32]) -> Vec<u8> {
    let mut out = vec![0u8; 4096];
    let n = unsafe {
        shim_write_sequence_header_obu(top.as_ptr(), sh.as_ptr(), cc.as_ptr(), idc.as_ptr(), level.as_ptr(), tier.as_ptr(), dmpp.as_ptr(), dispp.as_ptr(), decdelay.as_ptr(), encdelay.as_ptr(), lowdelay.as_ptr(), initdelay.as_ptr(), out.as_mut_ptr())
    };
    out.truncate(n as usize);
    out
}

/// Reference `aom_wb_write_uvlc`.
pub fn ref_wb_uvlc(v: u32) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe { shim_wb_uvlc(v, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_timing_info_header`.
pub fn ref_write_timing_info(disp_tick: u32, time_scale: u32, equal_pic: bool, ticks_per_pic: u32) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe { shim_write_timing_info(disp_tick, time_scale, equal_pic as i32, ticks_per_pic, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_decoder_model_info`.
pub fn ref_write_decoder_model_info(ed_delay_len: i32, dec_tick: u32, rem_time_len: i32, pres_time_len: i32) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe { shim_write_decoder_model_info(ed_delay_len, dec_tick, rem_time_len, pres_time_len, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_dec_model_op_parameters`.
pub fn ref_write_dec_model_op(dec_delay: u32, enc_delay: u32, low_delay: bool, delay_len: u32) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe { shim_write_dec_model_op(dec_delay, enc_delay, low_delay as i32, delay_len as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_color_config` (transcribed control flow over the real aom_wb).
/// `c` packs the 11 scalars in the order the shim reads them.
pub fn ref_write_color_config(c: &[i32; 11]) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe { shim_write_color_config(c.as_ptr(), out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_ext_tile_info` after `pre_bits` zero bits (transcribed over the real aom_wb).
pub fn ref_write_ext_tile_info(pre_bits: i32, rows: usize, cols: usize) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe { shim_write_ext_tile_info(pre_bits, rows as i32, cols as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_sequence_header` (transcribed control flow over the real aom_wb).
/// `s` packs the 24 scalars in the order the shim reads them.
pub fn ref_write_sequence_header(s: &[i32; 24]) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe { shim_write_sequence_header(s.as_ptr(), out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `write_global_motion` (transcribed control flow over the real aom_wb).
pub fn ref_write_global_motion(wmtype: &[i32; 7], wmmat: &[i32; 42], refmat: &[i32; 42], allow_hp: bool) -> Vec<u8> {
    let mut out = vec![0u8; 512];
    let n = unsafe { shim_write_global_motion(wmtype.as_ptr(), wmmat.as_ptr(), refmat.as_ptr(), allow_hp as i32, out.as_mut_ptr()) };
    out.truncate(n as usize);
    out
}

/// Reference `aom_wb_write_signed_primitive_refsubexpfin`.
pub fn ref_wb_signed_subexpfin(n: i32, k: i32, ref_: i32, v: i32) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let nn = unsafe { shim_wb_signed_subexpfin(n, k, ref_, v, out.as_mut_ptr()) };
    out.truncate(nn as usize);
    out
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

/// Reference `write_tile_group_header` (transcribed over the real aom_wb).
pub fn ref_write_tile_group_header(start_tile: i32, end_tile: i32, tiles_log2: i32, present_flag: bool) -> Vec<u8> {
    let mut out = vec![0u8; 8];
    let n = unsafe { shim_write_tile_group_header(start_tile, end_tile, tiles_log2, present_flag as i32, out.as_mut_ptr()) };
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
    pub fn av1_highbd_dr_prediction_z1_c(
        dst: *mut u16, stride: isize, bw: i32, bh: i32, above: *const u16, left: *const u16,
        upsample_above: i32, dx: i32, dy: i32, bd: i32,
    );
    pub fn av1_highbd_dr_prediction_z2_c(
        dst: *mut u16, stride: isize, bw: i32, bh: i32, above: *const u16, left: *const u16,
        upsample_above: i32, upsample_left: i32, dx: i32, dy: i32, bd: i32,
    );
    pub fn av1_highbd_dr_prediction_z3_c(
        dst: *mut u16, stride: isize, bw: i32, bh: i32, above: *const u16, left: *const u16,
        upsample_left: i32, dx: i32, dy: i32, bd: i32,
    );
}

/// Reference highbd intra prediction dispatch (`av1_predict_intra_block` routing,
/// minus palette / CfL): pick the predictor family, derive `p_angle` for
/// directional modes, and build. `recon[ref_off]` is the block top-left. Returns
/// the `txw*txh` block (row stride `txw`).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_predict_intra(
    recon: &[u16], ref_off: usize, ref_stride: usize, mode: usize, angle_delta: i32, use_filter_intra: bool, filter_intra_mode: usize, disable_edge_filter: bool, filt_type: i32, tx_size: usize, txw: usize, txh: usize, n_top_px: i32, n_topright_px: i32, n_left_px: i32, n_bottomleft_px: i32, bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_predict_intra(
            recon.as_ptr().add(ref_off), ref_stride as i32, mode as i32, angle_delta, use_filter_intra as i32, filter_intra_mode as i32, disable_edge_filter as i32, filt_type, tx_size as i32,
            n_top_px, n_topright_px, n_left_px, n_bottomleft_px, bd, dst.as_mut_ptr(), txw as i32,
        )
    }
    dst
}

/// Reference highbd filter-intra predictor (`highbd_filter_intra_predictor`).
/// `above` is a `[-1..]` view (index 0 the corner), `left` is `left[0..bh]`;
/// `mode` is the `FILTER_INTRA_MODE`. Returns the `bw*bh` block (row stride `bw`).
pub fn ref_hbd_filter_intra(
    tx_size: usize, bw: usize, bh: usize, above: &[u16], left: &[u16], mode: usize, bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; bw * bh];
    unsafe {
        shim_hbd_filter_intra_predict(
            dst.as_mut_ptr(), bw as isize, tx_size as i32, above.as_ptr(), left.as_ptr(), mode as i32, bd,
        )
    }
    dst
}

/// Reference highbd directional intra builder (`highbd_build_directional_and_
/// filter_intra_predictors`, directional path): edge assembly (with above-right /
/// below-left) + corner-filter + edge filter/upsample + angle dispatch.
/// `recon[ref_off]` is the block top-left (row stride `ref_stride`). `n_topright_px`
/// / `n_bottomleft_px` are `-1` when unavailable. Returns the `txw*txh` block.
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_build_dir_intra(
    recon: &[u16], ref_off: usize, ref_stride: usize, p_angle: i32, disable_edge_filter: bool, filt_type: i32, tx_size: usize, txw: usize, txh: usize, n_top_px: i32, n_topright_px: i32, n_left_px: i32, n_bottomleft_px: i32, bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_build_dir_intra(
            recon.as_ptr().add(ref_off), ref_stride as i32, p_angle, disable_edge_filter as i32, filt_type, tx_size as i32,
            n_top_px, n_topright_px, n_left_px, n_bottomleft_px, 0, 0, bd, dst.as_mut_ptr(), txw as i32,
        )
    }
    dst
}

/// Reference highbd filter-intra builder (`highbd_build_directional_and_filter_
/// intra_predictors`, `use_filter_intra` branch): assemble the reference edges
/// (all-need) then run the recursive filter-intra predictor `filter_intra_mode`.
/// `recon[ref_off]` is the block top-left. Returns the `txw*txh` block.
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_build_filter_intra(
    recon: &[u16], ref_off: usize, ref_stride: usize, filter_intra_mode: i32, tx_size: usize, txw: usize, txh: usize, n_top_px: i32, n_topright_px: i32, n_left_px: i32, n_bottomleft_px: i32, bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_build_dir_intra(
            recon.as_ptr().add(ref_off), ref_stride as i32, 90, 0, 0, tx_size as i32,
            n_top_px, n_topright_px, n_left_px, n_bottomleft_px, 1, filter_intra_mode, bd, dst.as_mut_ptr(), txw as i32,
        )
    }
    dst
}

/// Reference highbd directional predictor dispatch (`highbd_dr_predictor`):
/// route by `angle` to z1/z2/z3 or V/H. `above_data`/`left_data` are padded `u16`
/// buffers; the C edge pointer is taken at offset `pad` (`data[pad]` is the edge
/// origin, `data[pad-1]` the corner). Returns the `txw*txh` block (row stride `txw`).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_dr_predict(
    tx_size: usize, txw: usize, txh: usize, above_data: &[u16], left_data: &[u16], pad: usize, up_above: i32, up_left: i32, angle: i32, bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_dr_predict(
            dst.as_mut_ptr(), txw as isize, tx_size as i32,
            above_data.as_ptr().add(pad), left_data.as_ptr().add(pad), up_above, up_left, angle, bd,
        )
    }
    dst
}

/// Reference highbd directional predictor. `above`/`left` are padded `u16`
/// buffers; the C pointer is taken at offset `pad`. Returns the `bw*bh` block
/// (row stride `bw`).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_dr_pred(
    kind: u8, bw: usize, bh: usize, above: &[u16], left: &[u16], pad: usize,
    up_above: i32, up_left: i32, dx: i32, dy: i32, bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; bw * bh];
    let ap = unsafe { above.as_ptr().add(pad) };
    let lp = unsafe { left.as_ptr().add(pad) };
    unsafe {
        match kind {
            1 => av1_highbd_dr_prediction_z1_c(dst.as_mut_ptr(), bw as isize, bw as i32, bh as i32, ap, lp, up_above, dx, dy, bd),
            2 => av1_highbd_dr_prediction_z2_c(dst.as_mut_ptr(), bw as isize, bw as i32, bh as i32, ap, lp, up_above, up_left, dx, dy, bd),
            3 => av1_highbd_dr_prediction_z3_c(dst.as_mut_ptr(), bw as isize, bw as i32, bh as i32, ap, lp, up_left, dx, dy, bd),
            _ => unreachable!(),
        }
    }
    dst
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
    fn shim_hbd_build_nd_intra(
        r: *const u16, ref_stride: i32, av1_mode: i32, tx_size: i32, n_top_px: i32, n_left_px: i32, bd: i32, dst: *mut u16, dst_stride: i32,
    );
    fn shim_hbd_dr_predict(
        dst: *mut u16, stride: isize, tx_size: i32, above: *const u16, left: *const u16, up_above: i32, up_left: i32, angle: i32, bd: i32,
    );
    fn shim_hbd_build_dir_intra(
        r: *const u16, ref_stride: i32, p_angle: i32, disable_edge_filter: i32, filt_type: i32, tx_size: i32, n_top_px: i32, n_topright_px: i32, n_left_px: i32, n_bottomleft_px: i32, use_filter_intra: i32, filter_intra_mode: i32, bd: i32, dst: *mut u16, dst_stride: i32,
    );
    fn shim_hbd_filter_intra_predict(
        dst: *mut u16, stride: isize, tx_size: i32, above: *const u16, left: *const u16, mode: i32, bd: i32,
    );
    fn shim_hbd_predict_intra(
        r: *const u16, ref_stride: i32, mode: i32, angle_delta: i32, use_filter_intra: i32, filter_intra_mode: i32, disable_edge_filter: i32, filt_type: i32, tx_size: i32, n_top_px: i32, n_topright_px: i32, n_left_px: i32, n_bottomleft_px: i32, bd: i32, dst: *mut u16, dst_stride: i32,
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

/// Reference highbd non-directional intra builder (`av1_predict_intra_block`
/// branch, `highbd_build_non_directional_intra_predictors`): assemble the
/// reference edges from the reconstruction buffer (`recon[ref_off]` is the block
/// top-left, row stride `ref_stride`) with availability counts, then predict.
/// `av1_mode` is the AV1 `PREDICTION_MODE`. Returns the `txw*txh` predicted block
/// (row stride `txw`).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_build_nd_intra(
    recon: &[u16], ref_off: usize, ref_stride: usize, av1_mode: usize, tx_size: usize, txw: usize, txh: usize, n_top_px: usize, n_left_px: usize, bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_build_nd_intra(
            recon.as_ptr().add(ref_off), ref_stride as i32, av1_mode as i32, tx_size as i32,
            n_top_px as i32, n_left_px as i32, bd, dst.as_mut_ptr(), txw as i32,
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
    fn shim_dequant_txb(qcoeff: *const i32, dqcoeff: *mut i32, area: i32, tx_size: i32, dequant: *const i16, iqmatrix: *const u8, bd: i32);
}

/// Reference `av1_txb_init_levels_c` (writes into `levels`).
pub fn ref_txb_init_levels(coeff: &[i32], width: usize, height: usize, levels: &mut [u8]) {
    unsafe { shim_txb_init_levels(coeff.as_ptr(), width as i32, height as i32, levels.as_mut_ptr()) }
}

/// Reference decoder dequant (`av1_read_coeffs_txb` math, decodetxb.c): signed
/// `qcoeff` (raster, len `area`) → `dqcoeff` (raster). `iqmatrix` per raster
/// position, `None` for no quant matrix. Applies the `0xfffff`/`0xffffff` masks,
/// `av1_get_tx_scale` shift, and `±(1<<(7+bd))` clamp exactly as the C decoder.
pub fn ref_dequant_txb(qcoeff: &[i32], tx_size: usize, dequant: [i16; 2], iqmatrix: Option<&[u8]>, bd: i32) -> Vec<i32> {
    let area = qcoeff.len();
    let mut dq = vec![0i32; area];
    let iqp = iqmatrix.map_or(core::ptr::null(), |s| s.as_ptr());
    unsafe { shim_dequant_txb(qcoeff.as_ptr(), dq.as_mut_ptr(), area as i32, tx_size as i32, dequant.as_ptr(), iqp, bd) }
    dq
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

// av1/common/reconintra.c — intra neighbour availability (verbatim-paste shim).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_has_top_right(sb_size: i32, bsize: i32, mi_row: i32, mi_col: i32, top_available: i32, right_available: i32, partition: i32, txsz: i32, row_off: i32, col_off: i32, ss_x: i32, ss_y: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_has_bottom_left(sb_size: i32, bsize: i32, mi_row: i32, mi_col: i32, bottom_available: i32, left_available: i32, partition: i32, txsz: i32, row_off: i32, col_off: i32, ss_x: i32, ss_y: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_intra_avail(sb_size: i32, bsize: i32, mi_row: i32, mi_col: i32, up_available: i32, left_available: i32, tile_col_end: i32, tile_row_end: i32, partition: i32, tx_size: i32, ss_x: i32, ss_y: i32, row_off: i32, col_off: i32, wpx: i32, hpx: i32, mi_cols: i32, mi_rows: i32, mode: i32, angle_delta: i32, use_filter_intra: i32, out: *mut i32);
}

/// Reference `has_top_right` (reconintra.c): is the block's top-right reference
/// available and coded? Returns 0/1.
#[allow(clippy::too_many_arguments)]
pub fn ref_has_top_right(sb_size: usize, bsize: usize, mi_row: i32, mi_col: i32, top_available: bool, right_available: bool, partition: usize, txsz: usize, row_off: i32, col_off: i32, ss_x: i32, ss_y: i32) -> i32 {
    unsafe { shim_has_top_right(sb_size as i32, bsize as i32, mi_row, mi_col, top_available as i32, right_available as i32, partition as i32, txsz as i32, row_off, col_off, ss_x, ss_y) }
}

/// Reference `has_bottom_left` (reconintra.c): is the block's bottom-left
/// reference available and coded? Returns 0/1.
#[allow(clippy::too_many_arguments)]
pub fn ref_has_bottom_left(sb_size: usize, bsize: usize, mi_row: i32, mi_col: i32, bottom_available: bool, left_available: bool, partition: usize, txsz: usize, row_off: i32, col_off: i32, ss_x: i32, ss_y: i32) -> i32 {
    unsafe { shim_has_bottom_left(sb_size as i32, bsize as i32, mi_row, mi_col, bottom_available as i32, left_available as i32, partition as i32, txsz as i32, row_off, col_off, ss_x, ss_y) }
}

/// Reference intra neighbour-availability composition (the counts computed inside
/// `av1_predict_intra_block`). Returns `(n_top_px, n_topright_px, n_left_px,
/// n_bottomleft_px)`.
#[allow(clippy::too_many_arguments)]
pub fn ref_intra_avail(sb_size: usize, bsize: usize, mi_row: i32, mi_col: i32, up_available: bool, left_available: bool, tile_col_end: i32, tile_row_end: i32, partition: usize, tx_size: usize, ss_x: i32, ss_y: i32, row_off: i32, col_off: i32, wpx: i32, hpx: i32, mi_cols: i32, mi_rows: i32, mode: usize, angle_delta: i32, use_filter_intra: bool) -> (i32, i32, i32, i32) {
    let mut out = [0i32; 4];
    unsafe {
        shim_intra_avail(sb_size as i32, bsize as i32, mi_row, mi_col, up_available as i32, left_available as i32, tile_col_end, tile_row_end, partition as i32, tx_size as i32, ss_x, ss_y, row_off, col_off, wpx, hpx, mi_cols, mi_rows, mode as i32, angle_delta, use_filter_intra as i32, out.as_mut_ptr());
    }
    (out[0], out[1], out[2], out[3])
}

// av1/encoder/rd.c + rd.h (RD multiplier / RDCOST) and av1/common/quant_common.c
// (dc/ac quant lookups) — rd_shim.c. All exported symbols / real macros; no RTCD.
extern "C" {
    fn shim_compute_rd_mult_based_on_qindex(bit_depth: i32, update_type: i32, qindex: i32, tuning: i32, mode: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_compute_rd_mult(qindex: i32, bit_depth: i32, update_type: i32, layer_depth: i32, boost_index: i32, frame_type: i32, use_fixed_qp_offsets: i32, is_stat_consumption_stage: i32, tuning: i32, mode: i32) -> i32;
    fn shim_dc_quant_qtx(qindex: i32, delta: i32, bit_depth: i32) -> i32;
    fn shim_ac_quant_qtx(qindex: i32, delta: i32, bit_depth: i32) -> i32;
    fn shim_rdcost(rm: i32, rate: i32, dist: i64) -> i64;
    fn shim_rdcost_neg_r(rm: i32, rate: i32, dist: i64) -> i64;
    fn shim_dist_block_tx_domain(coeff: *const i32, dqcoeff: *const i32, tx_size: i32, bd: i32, out_dist: *mut i64, out_sse: *mut i64);
}

/// Reference `av1_compute_rd_mult_based_on_qindex` (rd.c). Enum args are passed
/// as their integer C values.
pub fn ref_compute_rd_mult_based_on_qindex(bit_depth: i32, update_type: i32, qindex: i32, tuning: i32, mode: i32) -> i32 {
    unsafe { shim_compute_rd_mult_based_on_qindex(bit_depth, update_type, qindex, tuning, mode) }
}

/// Reference `av1_compute_rd_mult` (rd.c).
#[allow(clippy::too_many_arguments)]
pub fn ref_compute_rd_mult(qindex: i32, bit_depth: i32, update_type: i32, layer_depth: i32, boost_index: i32, frame_type: i32, use_fixed_qp_offsets: i32, is_stat_consumption_stage: i32, tuning: i32, mode: i32) -> i32 {
    unsafe { shim_compute_rd_mult(qindex, bit_depth, update_type, layer_depth, boost_index, frame_type, use_fixed_qp_offsets, is_stat_consumption_stage, tuning, mode) }
}

/// Reference `av1_dc_quant_QTX` (quant_common.c).
pub fn ref_dc_quant_qtx(qindex: i32, delta: i32, bit_depth: i32) -> i32 {
    unsafe { shim_dc_quant_qtx(qindex, delta, bit_depth) }
}

/// Reference `av1_ac_quant_QTX` (quant_common.c).
pub fn ref_ac_quant_qtx(qindex: i32, delta: i32, bit_depth: i32) -> i32 {
    unsafe { shim_ac_quant_qtx(qindex, delta, bit_depth) }
}

/// Reference `RDCOST(rm, rate, dist)` macro (rd.h).
pub fn ref_rdcost(rm: i32, rate: i32, dist: i64) -> i64 {
    unsafe { shim_rdcost(rm, rate, dist) }
}

/// Reference `RDCOST_NEG_R(rm, rate, dist)` macro (rd.h).
pub fn ref_rdcost_neg_r(rm: i32, rate: i32, dist: i64) -> i64 {
    unsafe { shim_rdcost_neg_r(rm, rate, dist) }
}

/// Reference `dist_block_tx_domain` non-QM path (tx_search.c) — transform-domain
/// `(dist, sse)` for one txb (`buffer_length` derived from `tx_size` inside C).
pub fn ref_dist_block_tx_domain(coeff: &[i32], dqcoeff: &[i32], tx_size: usize, bd: u8) -> (i64, i64) {
    let (mut d, mut s) = (0i64, 0i64);
    unsafe { shim_dist_block_tx_domain(coeff.as_ptr(), dqcoeff.as_ptr(), tx_size as i32, bd as i32, &mut d, &mut s) };
    (d, s)
}

// ---- av1_build_quantizer oracle (rd_shim.c) ---------------------------------

extern "C" {
    fn shim_build_quantizer(
        bit_depth: i32,
        y_dc_delta_q: i32,
        u_dc_delta_q: i32,
        u_ac_delta_q: i32,
        v_dc_delta_q: i32,
        v_ac_delta_q: i32,
        sharpness: i32,
        out: *mut i16,
    ) -> i32;
}

/// Number of `i16` values written by [`ref_build_quantizer`]: 21 tables x
/// `QINDEX_RANGE` (256) x 8 SIMD lanes.
pub const BUILD_QUANTIZER_OUT_LEN: usize = 21 * 256 * 8;

/// Reference `av1_build_quantizer` (av1/encoder/av1_quantize.c — the REAL
/// exported function, no transcription). Fills `out` with the 21 quantizer
/// tables flattened in declaration order; see rd_shim.c for the index map.
#[allow(clippy::too_many_arguments)]
pub fn ref_build_quantizer(
    bit_depth: i32,
    y_dc_delta_q: i32,
    u_dc_delta_q: i32,
    u_ac_delta_q: i32,
    v_dc_delta_q: i32,
    v_ac_delta_q: i32,
    sharpness: i32,
    out: &mut [i16],
) {
    assert_eq!(out.len(), BUILD_QUANTIZER_OUT_LEN);
    let rc = unsafe {
        shim_build_quantizer(
            bit_depth,
            y_dc_delta_q,
            u_dc_delta_q,
            u_ac_delta_q,
            v_dc_delta_q,
            v_ac_delta_q,
            sharpness,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "shim_build_quantizer allocation failed");
}

// ---- tx-type signaling cost oracles (rd_shim.c) ------------------------------

extern "C" {
    fn shim_fill_tx_type_costs(
        intra_cdf: *const u16,
        inter_cdf: *const u16,
        out_intra: *mut i32,
        out_inter: *mut i32,
    );
    fn shim_get_tx_type_cost(
        intra_costs: *const i32,
        inter_costs: *const i32,
        plane: i32,
        tx_size: i32,
        tx_type: i32,
        is_inter: i32,
        reduced_tx_set_used: i32,
        lossless: i32,
        use_filter_intra: i32,
        filter_intra_mode: i32,
        mode: i32,
    ) -> i32;
}

/// Flat lengths for the tx-type cost tables / CDF inputs
/// (`EXT_TX_SETS_INTRA`=3, `EXT_TX_SETS_INTER`=4, `EXT_TX_SIZES`=4,
/// `INTRA_MODES`=13, `TX_TYPES`=16, CDF rows are `TX_TYPES+1` wide).
pub const TX_TYPE_COSTS_INTRA_LEN: usize = 3 * 4 * 13 * 16;
pub const TX_TYPE_COSTS_INTER_LEN: usize = 4 * 4 * 16;
pub const TX_TYPE_CDF_INTRA_LEN: usize = 3 * 4 * 13 * 17;
pub const TX_TYPE_CDF_INTER_LEN: usize = 4 * 4 * 17;

/// Reference tx-type slice of `av1_fill_mode_rates` (rd.c): fill the
/// intra/inter tx-type cost tables from flat CDF arrays (see rd_shim.c for the
/// layouts). Outputs are zero-initialized here (ungated combos stay 0).
pub fn ref_fill_tx_type_costs(intra_cdf: &[u16], inter_cdf: &[u16]) -> (Vec<i32>, Vec<i32>) {
    assert_eq!(intra_cdf.len(), TX_TYPE_CDF_INTRA_LEN);
    assert_eq!(inter_cdf.len(), TX_TYPE_CDF_INTER_LEN);
    let mut out_intra = vec![0i32; TX_TYPE_COSTS_INTRA_LEN];
    let mut out_inter = vec![0i32; TX_TYPE_COSTS_INTER_LEN];
    unsafe {
        shim_fill_tx_type_costs(
            intra_cdf.as_ptr(),
            inter_cdf.as_ptr(),
            out_intra.as_mut_ptr(),
            out_inter.as_mut_ptr(),
        );
    }
    (out_intra, out_inter)
}

/// Reference `get_tx_type_cost` (txb_rdopt.c): the plane-0 tx_type signaling
/// rate, looked up in the flat tables from [`ref_fill_tx_type_costs`].
#[allow(clippy::too_many_arguments)]
pub fn ref_get_tx_type_cost(
    intra_costs: &[i32],
    inter_costs: &[i32],
    plane: i32,
    tx_size: i32,
    tx_type: i32,
    is_inter: bool,
    reduced_tx_set_used: bool,
    lossless: bool,
    use_filter_intra: bool,
    filter_intra_mode: i32,
    mode: i32,
) -> i32 {
    assert_eq!(intra_costs.len(), TX_TYPE_COSTS_INTRA_LEN);
    assert_eq!(inter_costs.len(), TX_TYPE_COSTS_INTER_LEN);
    unsafe {
        shim_get_tx_type_cost(
            intra_costs.as_ptr(),
            inter_costs.as_ptr(),
            plane,
            tx_size,
            tx_type,
            is_inter as i32,
            reduced_tx_set_used as i32,
            lossless as i32,
            use_filter_intra as i32,
            filter_intra_mode,
            mode,
        )
    }
}
