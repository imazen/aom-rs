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
        syms: *const i32,
        n: i32,
        cdf_init: *const u16,
        nsymbs: i32,
        out: *mut u8,
        out_cap: u32,
    ) -> u32;
    fn shim_adapt_decode(
        buf: *const u8,
        sz: u32,
        n: i32,
        cdf_init: *const u16,
        nsymbs: i32,
        out_syms: *mut i32,
    );
}

// convolve_shim.c — av1_convolve_{x,y}_sr (EIGHTTAP_REGULAR).
extern "C" {
    fn shim_convolve_x_sr(
        src: *const u8,
        ss: i32,
        dst: *mut u8,
        ds: i32,
        w: i32,
        h: i32,
        subpel: i32,
        ftype: i32,
    );
    fn shim_convolve_y_sr(
        src: *const u8,
        ss: i32,
        dst: *mut u8,
        ds: i32,
        w: i32,
        h: i32,
        subpel: i32,
        ftype: i32,
    );
    fn shim_convolve_2d_sr(
        src: *const u8,
        ss: i32,
        dst: *mut u8,
        ds: i32,
        w: i32,
        h: i32,
        spx: i32,
        spy: i32,
        ftype: i32,
    );
}

/// Reference `av1_convolve_2d_sr_c`.
#[allow(clippy::too_many_arguments)]
pub fn ref_convolve_2d_sr(
    src: &[u8],
    src_off: usize,
    ss: usize,
    w: usize,
    h: usize,
    spx: usize,
    spy: usize,
    ftype: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    unsafe {
        shim_convolve_2d_sr(
            src.as_ptr().add(src_off),
            ss as i32,
            dst.as_mut_ptr(),
            w as i32,
            w as i32,
            h as i32,
            spx as i32,
            spy as i32,
            ftype as i32,
        )
    }
    dst
}

/// Reference `av1_convolve_x_sr_c`. `src` points at the interior origin.
pub fn ref_convolve_x_sr(
    src: &[u8],
    src_off: usize,
    ss: usize,
    w: usize,
    h: usize,
    subpel: usize,
    ftype: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    unsafe {
        shim_convolve_x_sr(
            src.as_ptr().add(src_off),
            ss as i32,
            dst.as_mut_ptr(),
            w as i32,
            w as i32,
            h as i32,
            subpel as i32,
            ftype as i32,
        )
    }
    dst
}

/// Reference `av1_convolve_y_sr_c`.
pub fn ref_convolve_y_sr(
    src: &[u8],
    src_off: usize,
    ss: usize,
    w: usize,
    h: usize,
    subpel: usize,
    ftype: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    unsafe {
        shim_convolve_y_sr(
            src.as_ptr().add(src_off),
            ss as i32,
            dst.as_mut_ptr(),
            w as i32,
            w as i32,
            h as i32,
            subpel as i32,
            ftype as i32,
        )
    }
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
    pub fn aom_avg_4x4_c(s: *const u8, p: i32) -> u32;
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

/// Reference `aom_avg_4x4_c` (lowbd): `src` must hold 4 rows of 4 samples at
/// `stride`; returns `(sum + 8) >> 4`. The variance-based partitioner's
/// KEY-frame 4x4 downsampling kernel (var_based_part.c
/// `fill_variance_4x4avg`).
pub fn ref_avg_4x4(src: &[u8], stride: usize) -> u32 {
    assert!(src.len() >= 3 * stride + 4);
    unsafe { aom_avg_4x4_c(src.as_ptr(), stride as i32) }
}

// av1/common/cdef_block.c — CDEF direction search + filter block.
extern "C" {
    pub fn cdef_find_dir_c(img: *const u16, stride: i32, var: *mut i32, coeff_shift: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_cdef_filter8(
        variant: i32,
        dst: *mut u8,
        dstride: i32,
        in_: *const u16,
        pri: i32,
        sec: i32,
        dir: i32,
        prid: i32,
        secd: i32,
        cshift: i32,
        bw: i32,
        bh: i32,
    );
}

/// Reference `cdef_filter_8_<variant>_c`. `in_buf` is u16 with stride
/// CDEF_BSTRIDE; `in_off` is the block origin. Returns the `bw*bh` result.
#[allow(clippy::too_many_arguments)]
pub fn ref_cdef_filter8(
    variant: i32,
    in_buf: &[u16],
    in_off: usize,
    pri: i32,
    sec: i32,
    dir: i32,
    prid: i32,
    secd: i32,
    cshift: i32,
    bw: usize,
    bh: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; bw * bh];
    unsafe {
        shim_cdef_filter8(
            variant,
            dst.as_mut_ptr(),
            bw as i32,
            in_buf.as_ptr().add(in_off),
            pri,
            sec,
            dir,
            prid,
            secd,
            cshift,
            bw as i32,
            bh as i32,
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

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_cdef_filter16(
        variant: i32,
        dst: *mut u16,
        dstride: i32,
        in_buf: *const u16,
        pri: i32,
        sec: i32,
        dir: i32,
        prid: i32,
        secd: i32,
        cshift: i32,
        bw: i32,
        bh: i32,
    );
}

/// Reference `cdef_filter_16_<variant>_c` (u16 store). Same layout contract
/// as [`ref_cdef_filter8`].
#[allow(clippy::too_many_arguments)]
pub fn ref_cdef_filter16(
    variant: i32,
    in_buf: &[u16],
    in_off: usize,
    pri: i32,
    sec: i32,
    dir: i32,
    prid: i32,
    secd: i32,
    cshift: i32,
    bw: usize,
    bh: usize,
) -> Vec<u16> {
    let mut dst = vec![0u16; bw * bh];
    unsafe {
        shim_cdef_filter16(
            variant,
            dst.as_mut_ptr(),
            bw as i32,
            in_buf.as_ptr().add(in_off),
            pri,
            sec,
            dir,
            prid,
            secd,
            cshift,
            bw as i32,
            bh as i32,
        )
    }
    dst
}

// sadvar_shim.c — SAD / variance / sub-pixel variance dispatch (22 sizes).
extern "C" {
    fn shim_sad(idx: i32, s: *const u8, ss: i32, r: *const u8, rs: i32) -> u32;
    fn shim_variance(idx: i32, a: *const u8, as_: i32, b: *const u8, bs: i32, sse: *mut u32)
        -> u32;
    fn shim_subpel_var(
        idx: i32,
        a: *const u8,
        as_: i32,
        xo: i32,
        yo: i32,
        b: *const u8,
        bs: i32,
        sse: *mut u32,
    ) -> u32;
}

/// Reference `aom_sad<W>x<H>_c` for size index `idx`.
pub fn ref_sad(idx: usize, s: &[u8], ss: usize, r: &[u8], rs: usize) -> u32 {
    unsafe { shim_sad(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32) }
}

extern "C" {
    fn shim_sad_avg(i: i32, s: *const u8, ss: i32, r: *const u8, rs: i32, sp: *const u8) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_masked_sad(
        i: i32,
        s: *const u8,
        ss: i32,
        r: *const u8,
        rs: i32,
        sp: *const u8,
        m: *const u8,
        ms: i32,
        inv: i32,
    ) -> u32;
    fn shim_obmc_sad(i: i32, r: *const u8, rs: i32, ws: *const i32, m: *const i32) -> u32;
    fn shim_sse(a: *const u8, as_: i32, b: *const u8, bs: i32, w: i32, h: i32) -> i64;
    fn shim_hbd_sse(a: *const u16, as_: i32, b: *const u16, bs: i32, w: i32, h: i32) -> i64;
}

/// Reference `aom_sse_c` (sum of squared errors, generic w×h).
pub fn ref_sse(a: &[u8], as_: usize, b: &[u8], bs: usize, w: usize, h: usize) -> i64 {
    unsafe {
        shim_sse(
            a.as_ptr(),
            as_ as i32,
            b.as_ptr(),
            bs as i32,
            w as i32,
            h as i32,
        )
    }
}

/// Reference `aom_highbd_sse_c`.
pub fn ref_hbd_sse(a: &[u16], as_: usize, b: &[u16], bs: usize, w: usize, h: usize) -> i64 {
    unsafe {
        shim_hbd_sse(
            a.as_ptr(),
            as_ as i32,
            b.as_ptr(),
            bs as i32,
            w as i32,
            h as i32,
        )
    }
}

extern "C" {
    fn av1_block_error_c(
        coeff: *const i32,
        dqcoeff: *const i32,
        block_size: isize,
        ssz: *mut i64,
    ) -> i64;
    fn av1_highbd_block_error_c(
        coeff: *const i32,
        dqcoeff: *const i32,
        block_size: isize,
        ssz: *mut i64,
        bd: i32,
    ) -> i64;
    fn aom_subtract_block_c(
        rows: i32,
        cols: i32,
        diff: *mut i16,
        diff_stride: isize,
        src: *const u8,
        src_stride: isize,
        pred: *const u8,
        pred_stride: isize,
    );
    fn shim_highbd_subtract_block(
        rows: i32,
        cols: i32,
        diff: *mut i16,
        diff_stride: i32,
        src: *const u16,
        src_stride: i32,
        pred: *const u16,
        pred_stride: i32,
    );
    fn shim_block_error_qm(
        coeff: *const i32,
        dqcoeff: *const i32,
        block_size: isize,
        qmatrix: *const u8,
        scan: *const i16,
        ssz: *mut i64,
        bd: i32,
    ) -> i64;
    fn av1_model_rd_from_var_lapndz(
        var: i64,
        n_log2: u32,
        qstep: u32,
        rate: *mut i32,
        dist: *mut i64,
    );
    fn aom_sum_squares_i16_c(src: *const i16, n: u32) -> u64;
    fn aom_sum_squares_2d_i16_c(src: *const i16, src_stride: i32, width: i32, height: i32) -> u64;
    fn aom_vector_var_c(reff: *const i16, src: *const i16, bwl: i32) -> i32;
    fn shim_wb_apply(
        data: *const u32,
        bits: *const i32,
        kind: *const i32,
        n: i32,
        out: *mut u8,
    ) -> u32;
    fn aom_uleb_size_in_bytes(value: u64) -> usize;
    fn aom_uleb_encode(
        value: u64,
        available: usize,
        coded_value: *mut u8,
        coded_size: *mut usize,
    ) -> i32;
    fn aom_uleb_decode(
        buffer: *const u8,
        available: usize,
        value: *mut u64,
        length: *mut usize,
    ) -> i32;
    fn shim_write_obu_header(
        obu_type: i32,
        has_nonzero_op: i32,
        is_layer_specific: i32,
        obu_extension: i32,
        dst: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_quantization(
        base_qindex: i32,
        y_dc: i32,
        u_dc: i32,
        u_ac: i32,
        v_dc: i32,
        v_ac: i32,
        using_qm: i32,
        qm_y: i32,
        qm_u: i32,
        qm_v: i32,
        num_planes: i32,
        separate_uv: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_loopfilter(
        allow_intrabc: i32,
        fl0: i32,
        fl1: i32,
        flu: i32,
        flv: i32,
        sharpness: i32,
        mode_ref_enabled: i32,
        mode_ref_update: i32,
        ref_deltas: *const i8,
        mode_deltas: *const i8,
        last_ref: *const i8,
        last_mode: *const i8,
        num_planes: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_cdef(
        enable_cdef: i32,
        allow_intrabc: i32,
        damping: i32,
        cdef_bits: i32,
        nb: i32,
        y: *const i32,
        uv: *const i32,
        num_planes: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_encode_segmentation(
        enabled: i32,
        has_primary_ref: i32,
        update_map: i32,
        temporal_update: i32,
        update_data: i32,
        feature_mask: *const u32,
        feature_data: *const i32,
        out: *mut u8,
    ) -> u32;
    fn shim_write_frame_interp_filter(filter: i32, out: *mut u8) -> u32;
    fn shim_write_superres_scale(enable_superres: i32, denom: i32, out: *mut u8) -> u32;
    fn shim_write_render_size(scaling_active: i32, rw: i32, rh: i32, out: *mut u8) -> u32;
    // superres_shim.c — faithful transcription of the static superres
    // denom-selection functions (calling the real exported leaf math).
    fn shim_superres_analyze_hor_freq(
        src: *const u16,
        width: i32,
        height: i32,
        stride: i32,
        bd: i32,
        energy_out: *mut f64,
    );
    fn shim_superres_denom_from_qindex_energy(
        qindex: i32,
        energy: *const f64,
        threshq: f64,
        threshp: f64,
    ) -> u8;
    #[allow(clippy::too_many_arguments)]
    fn shim_superres_denom_qthresh_key(
        src: *const u16,
        w: i32,
        h: i32,
        stride: i32,
        bd: i32,
        q: i32,
        kf_qthresh_qindex: i32,
        allow_scc: i32,
        frames_to_key_le_1: i32,
    ) -> u8;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_frame_size(
        frame_size_override: i32,
        num_bits_width: i32,
        num_bits_height: i32,
        up_w: i32,
        up_h: i32,
        enable_superres: i32,
        denom: i32,
        scaling_active: i32,
        rw: i32,
        rh: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_tile_group_header(
        start_tile: i32,
        end_tile: i32,
        tiles_log2: i32,
        present_flag: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_write_tile_info(
        mi_cols: i32,
        mi_rows: i32,
        mib_size_log2: i32,
        uniform_spacing: i32,
        log2_cols: i32,
        min_log2_cols: i32,
        max_log2_cols: i32,
        log2_rows: i32,
        min_log2_rows: i32,
        max_log2_rows: i32,
        cols: i32,
        rows: i32,
        col_start_sb: *const i32,
        row_start_sb: *const i32,
        max_width_sb: i32,
        max_height_sb: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_restoration_mode(
        enable_restoration: i32,
        allow_intrabc: i32,
        frame_restoration_type: *const i32,
        sb_size_128: i32,
        restoration_unit_size: *const i32,
        ssx: i32,
        ssy: i32,
        num_planes: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_delta_q_params(
        base_qindex: i32,
        delta_q_present: i32,
        delta_q_res: i32,
        allow_intrabc: i32,
        delta_lf_present: i32,
        delta_lf_res: i32,
        delta_lf_multi: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_write_tx_mode(coded_lossless: i32, tx_mode_select: i32, out: *mut u8) -> u32;
    fn shim_write_film_grain_params(
        s: *const i32,
        spy: *const i32,
        spcb: *const i32,
        spcr: *const i32,
        ary: *const i32,
        arcb: *const i32,
        arcr: *const i32,
        out: *mut u8,
    ) -> u32;
    fn shim_wb_signed_subexpfin(n: i32, k: i32, ref_: i32, v: i32, out: *mut u8) -> u32;
    fn shim_write_global_motion(
        wmtype: *const i32,
        wmmat: *const i32,
        refmat: *const i32,
        allow_hp: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_write_sequence_header(s: *const i32, out: *mut u8) -> u32;
    fn shim_write_ext_tile_info(pre_bits: i32, rows: i32, cols: i32, out: *mut u8) -> u32;
    fn shim_write_color_config(c: *const i32, out: *mut u8) -> u32;
    fn shim_wb_uvlc(v: u32, out: *mut u8) -> u32;
    fn shim_write_timing_info(
        disp_tick: u32,
        time_scale: u32,
        equal_pic: i32,
        ticks_per_pic: u32,
        out: *mut u8,
    ) -> u32;
    fn shim_write_decoder_model_info(
        ed_delay_len: i32,
        dec_tick: u32,
        rem_time_len: i32,
        pres_time_len: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_write_dec_model_op(
        dec_delay: u32,
        enc_delay: u32,
        low_delay: i32,
        delay_len: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_sequence_header_obu(
        top: *const i64,
        sh: *const i64,
        cc: *const i64,
        idc: *const i64,
        level: *const i64,
        tier: *const i64,
        dmpp: *const i64,
        dispp: *const i64,
        decdelay: *const i64,
        encdelay: *const i64,
        lowdelay: *const i64,
        initdelay: *const i64,
        out: *mut u8,
    ) -> u32;
    fn shim_write_frame_header_prefix(
        t: *const i64,
        op_dmpp: *const i64,
        op_idc: *const i64,
        brt: *const i64,
        ref_oh: *const i64,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_frame_size_with_refs(
        up_w: i32,
        up_h: i32,
        rw: i32,
        rh: i32,
        valid: *const i32,
        ycw: *const i32,
        ych: *const i32,
        rrw: *const i32,
        rrh: *const i32,
        enable_superres: i32,
        denom: i32,
        fs_num_bits_w: i32,
        fs_num_bits_h: i32,
        fs_up_w: i32,
        fs_up_h: i32,
        fs_scaling_active: i32,
        fs_rw: i32,
        fs_rh: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_ref_signaling(
        enable_order_hint: i32,
        short_sig: i32,
        ref_map_idx: *const i32,
        set_rfc: i32,
        rtc_reference: *const i32,
        rtc_ref_idx: *const i32,
        num_spatial_layers: i32,
        frame_id_present: i32,
        frame_id_len: i32,
        current_frame_id: i32,
        ref_frame_id: *const i32,
        diff_len: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_write_refresh_frame_context(
        reduced: i32,
        disable_cdf: i32,
        rfc_disabled: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_partition_cdf_length(bsize: i32) -> i32;
    fn shim_partition_gather_vert(out: *mut u16, cdf_in: *const u16, bsize: i32);
    fn shim_partition_gather_horz(out: *mut u16, cdf_in: *const u16, bsize: i32);
    fn shim_partition_plane_context(
        above: *const i8,
        left: *const i8,
        mi_row: i32,
        mi_col: i32,
        bsize: i32,
    ) -> i32;
    fn shim_write_partition(
        partition_cdf: *mut u16,
        cdf_len: i32,
        p: i32,
        has_rows: i32,
        has_cols: i32,
        bsize: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_skip_txfm_context(
        above_present: i32,
        above_skip: i32,
        left_present: i32,
        left_skip: i32,
    ) -> i32;
    fn shim_write_skip(
        skip_cdf: *mut u16,
        seg_skip_active: i32,
        skip_txfm: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_write_delta_qindex(
        delta_q_cdf: *mut u16,
        delta_qindex: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_write_delta_lflevel(
        delta_lf_cdf: *mut u16,
        delta_lflevel: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_write_cfl_alphas(
        cfl_sign_cdf: *mut u16,
        cfl_alpha_cdf: *mut u16,
        idx: i32,
        joint_sign: i32,
        out: *mut u8,
        out_sign_cdf: *mut u16,
        out_alpha_cdf: *mut u16,
    ) -> u32;
    fn shim_get_y_mode_ctx(
        above_present: i32,
        above_mode: i32,
        left_present: i32,
        left_mode: i32,
    ) -> i32;
    fn shim_write_intra_y_mode_kf(
        kf_y_cdf: *mut u16,
        mode: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_size_group_lookup(bsize: i32) -> i32;
    fn shim_write_intra_uv_mode(
        uv_mode_cdf: *mut u16,
        uv_mode: i32,
        cfl_allowed: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_write_angle_delta(
        cdf: *mut u16,
        angle_delta: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_mb_interp_filter(
        cdf0: *mut u16,
        cdf1: *mut u16,
        interp_needed: i32,
        is_switchable: i32,
        enable_dual: i32,
        f0: i32,
        f1: i32,
        out: *mut u8,
        out0: *mut u16,
        out1: *mut u16,
    ) -> u32;
    fn shim_get_intra_inter_context(
        has_above: i32,
        above_inter: i32,
        has_left: i32,
        left_inter: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_pred_context_switchable_interp(
        dir: i32,
        cur_r0: i32,
        cur_r1: i32,
        has_above: i32,
        a_r0: i32,
        a_r1: i32,
        a_yf: i32,
        a_xf: i32,
        has_left: i32,
        l_r0: i32,
        l_r1: i32,
        l_yf: i32,
        l_xf: i32,
    ) -> i32;
    fn shim_get_skip_mode_context(ha: i32, a_sm: i32, hl: i32, l_sm: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_skip_mode(
        cdf: *mut u16,
        frame_flag: i32,
        seg_skip: i32,
        comp_allowed: i32,
        seg_ref_gmv: i32,
        skip_mode: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_reference_mode_context(
        ha: i32,
        a_r0: i32,
        a_r1: i32,
        a_ibc: i32,
        hl: i32,
        l_r0: i32,
        l_r1: i32,
        l_ibc: i32,
    ) -> i32;
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
    fn shim_write_intrabc_info(
        intrabc_cdf: *mut u16,
        joints: *mut u16,
        comp0: *mut u16,
        comp1: *mut u16,
        use_intrabc: i32,
        diff_row: i32,
        diff_col: i32,
        out: *mut u8,
        out_ibc: *mut u16,
        out_joints: *mut u16,
        out_c0: *mut u16,
        out_c1: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_segment_id(
        cdf: *mut u16,
        seg_enabled: i32,
        update_map: i32,
        skip_txfm: i32,
        segment_id: i32,
        pred: i32,
        last_active_segid: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_txfm_partition_context(above: u8, left: u8, bsize: i32, tx_size: i32) -> i32;
    fn shim_txfm_partition_update(
        above_ctx: *mut u8,
        left_ctx: *mut u8,
        tx_size: i32,
        txb_size: i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_write_tx_size_vartx(
        bsize: i32,
        top_tx_size: i32,
        inter_tx_size: *const u8,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        above_in: *const u8,
        left_in: *const u8,
        cdf: *mut u16,
        out: *mut u8,
        above_out: *mut u8,
        left_out: *mut u8,
        cdf_out: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_txfm_size(
        bsize: i32,
        max_tx: i32,
        inter_tx_size: *const u8,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        above_in: *const u8,
        left_in: *const u8,
        cdf: *mut u16,
        out: *mut u8,
        above_out: *mut u8,
        left_out: *mut u8,
        cdf_out: *mut u16,
    ) -> u32;
    fn shim_get_palette_bsize_ctx(bsize: i32) -> i32;
    fn shim_get_palette_mode_ctx(ha: i32, a_psize: i32, hl: i32, l_psize: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_palette_flags_sizes(
        mode_dc: i32,
        n_y: i32,
        y_mode_cdf: *mut u16,
        y_size_cdf: *mut u16,
        uv_dc: i32,
        n_uv: i32,
        uv_mode_cdf: *mut u16,
        uv_size_cdf: *mut u16,
        out: *mut u8,
        o_ym: *mut u16,
        o_ys: *mut u16,
        o_um: *mut u16,
        o_us: *mut u16,
    ) -> u32;
    fn shim_delta_encode_palette_colors(
        colors: *const i32,
        num: i32,
        bit_depth: i32,
        min_val: i32,
        out: *mut u8,
    ) -> u32;
    fn shim_pack_map_tokens(
        n: i32,
        num: i32,
        tokens: *const u8,
        color_ctxs: *const u8,
        map_cdf: *mut u16,
        out: *mut u8,
        map_cdf_out: *mut u16,
    ) -> u32;
    fn shim_write_palette_colors_v(
        colors_v: *const u16,
        n: i32,
        bit_depth: i32,
        out: *mut u8,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_palette_cache(
        plane: i32,
        mb_to_top_edge: i32,
        ha: i32,
        a_colors: *const u16,
        a_size0: i32,
        a_size1: i32,
        hl: i32,
        l_colors: *const u16,
        l_size0: i32,
        l_size1: i32,
        out_cache: *mut u16,
    ) -> i32;
    fn shim_index_color_cache(
        cache: *const u16,
        n_cache: i32,
        colors: *const u16,
        n_colors: i32,
        found: *mut u8,
        out_colors: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_palette_mode_info(
        mode_dc: i32,
        uv_dc: i32,
        bit_depth: i32,
        bsize_ctx: i32,
        y_mode_ctx: i32,
        uv_mode_ctx: i32,
        palette_size: *const u8,
        palette_colors: *const u16,
        mb_to_top_edge: i32,
        ha: i32,
        a_colors: *const u16,
        a_s0: i32,
        a_s1: i32,
        hl: i32,
        l_colors: *const u16,
        l_s0: i32,
        l_s1: i32,
        y_mode_cdf: *mut u16,
        y_size_cdf: *mut u16,
        uv_mode_cdf: *mut u16,
        uv_size_cdf: *mut u16,
        out: *mut u8,
        o_ym: *mut u16,
        o_ys: *mut u16,
        o_um: *mut u16,
        o_us: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_interintra_info(
        interintra: i32,
        ii_cdf: *mut u16,
        ii_mode: i32,
        ii_mode_cdf: *mut u16,
        wedge_used: i32,
        use_wedge: i32,
        wedge_ii_cdf: *mut u16,
        wedge_index: i32,
        wedge_idx_cdf: *mut u16,
        out: *mut u8,
        o_ii: *mut u16,
        o_iim: *mut u16,
        o_wii: *mut u16,
        o_wix: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_comp_group_idx_context(
        ha: i32,
        a_rf0: i32,
        a_rf1: i32,
        a_cgi: i32,
        hl: i32,
        l_rf0: i32,
        l_rf1: i32,
        l_cgi: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_compound_type_info(
        masked_used: i32,
        comp_group_idx: i32,
        cgi_cdf: *mut u16,
        dist_wtd: i32,
        compound_idx: i32,
        cidx_cdf: *mut u16,
        wedge_used: i32,
        comp_type: i32,
        ctype_cdf: *mut u16,
        wedge_index: i32,
        wedge_idx_cdf: *mut u16,
        wedge_sign: i32,
        mask_type: i32,
        out: *mut u8,
        o_cgi: *mut u16,
        o_cidx: *mut u16,
        o_ctype: *mut u16,
        o_wix: *mut u16,
    ) -> u32;
    fn shim_get_relative_dist(enable: i32, bits_minus_1: i32, a: i32, b: i32) -> i32;
    fn shim_get_pred_context_seg_id(ha: i32, a_sip: i32, hl: i32, l_sip: i32) -> i32;
    fn shim_is_inter_compound_mode(mode: i32) -> i32;
    fn shim_is_inter_singleref_mode(mode: i32) -> i32;
    fn shim_have_nearmv_in_inter_mode(mode: i32) -> i32;
    fn shim_mode_context_analyzer(rf0: i32, rf1: i32, mc_val: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_collect_neighbors_ref_counts(
        ha: i32,
        a_intrabc: i32,
        a_rf0: i32,
        a_rf1: i32,
        hl: i32,
        l_intrabc: i32,
        l_rf0: i32,
        l_rf1: i32,
        out_counts: *mut u8,
    );
    fn shim_get_partition_subsize(bsize: i32, partition: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_update_ext_partition_context(
        mi_row: i32,
        mi_col: i32,
        subsize: i32,
        bsize: i32,
        partition: i32,
        above_in: *const i8,
        left_in: *const i8,
        above_out: *mut i8,
        left_out: *mut i8,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_write_partition_node(
        above_in: *const i8,
        left_in: *const i8,
        mi_row: i32,
        mi_col: i32,
        bsize: i32,
        partition: i32,
        mi_rows: i32,
        mi_cols: i32,
        arena: *mut u16,
        out: *mut u8,
        above_out: *mut i8,
        left_out: *mut i8,
        arena_out: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_modes_sb(
        above_in: *const i8,
        left_in: *const i8,
        mi_row: i32,
        mi_col: i32,
        bsize: i32,
        tree: *const i8,
        tree_len: i32,
        arena: *mut u16,
        out: *mut u8,
        above_out: *mut i8,
        left_out: *mut i8,
        arena_out: *mut u16,
        tree_consumed: *mut i32,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_modes_tile(
        n_sb_rows: i32,
        n_sb_cols: i32,
        sb_mi: i32,
        sb_size: i32,
        tree: *const i8,
        arena: *mut u16,
        out: *mut u8,
        above_out: *mut i8,
        arena_out: *mut u16,
        tree_consumed: *mut i32,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_block_mvs(
        mode: i32,
        is_compound: i32,
        diff_row0: i32,
        diff_col0: i32,
        diff_row1: i32,
        diff_col1: i32,
        usehp: i32,
        joints: *mut u16,
        comp0: *mut u16,
        comp1: *mut u16,
        out: *mut u8,
        o_joints: *mut u16,
        o_c0: *mut u16,
        o_c1: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_mode_drl(
        seg_skip: i32,
        mode: i32,
        mode_ctx: i32,
        inter_compound_mode_cdf: *mut u16,
        newmv_cdf: *mut u16,
        zeromv_cdf: *mut u16,
        refmv_cdf: *mut u16,
        drl_cdf: *mut u16,
        ref_mv_idx: i32,
        ref_mv_count: i32,
        weight: *const u16,
        out: *mut u8,
        o_icm: *mut u16,
        o_newmv: *mut u16,
        o_zeromv: *mut u16,
        o_refmv: *mut u16,
        o_drl: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_mode_tail(
        interintra_allowed: i32,
        interintra: i32,
        ii_cdf: *mut u16,
        ii_mode: i32,
        ii_mode_cdf: *mut u16,
        wedge_used_ii: i32,
        use_wedge_ii: i32,
        wedge_ii_cdf: *mut u16,
        ii_wedge_index: i32,
        wedge_idx_cdf: *mut u16,
        motion_mode_present: i32,
        obmc_cdf: *mut u16,
        mm_cdf: *mut u16,
        last_motion_mode_allowed: i32,
        motion_mode: i32,
        has_second_ref: i32,
        masked_used: i32,
        comp_group_idx: i32,
        cgi_cdf: *mut u16,
        dist_wtd: i32,
        compound_idx: i32,
        cidx_cdf: *mut u16,
        wedge_used_ct: i32,
        comp_type: i32,
        ctype_cdf: *mut u16,
        ct_wedge_index: i32,
        wedge_sign: i32,
        mask_type: i32,
        interp_needed: i32,
        is_switchable: i32,
        enable_dual: i32,
        f0: i32,
        f1: i32,
        interp_cdf0: *mut u16,
        interp_cdf1: *mut u16,
        out: *mut u8,
        o_all: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_segment_id(
        update_map: i32,
        preskip: i32,
        segid_preskip: i32,
        skip: i32,
        temporal_update: i32,
        seg_id_predicted: i32,
        pred_cdf: *mut u16,
        seg_cdf: *mut u16,
        seg_enabled: i32,
        segment_id: i32,
        seg_pred: i32,
        last_active_segid: i32,
        out: *mut u8,
        o_predcdf: *mut u16,
        o_segcdf: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_prefix(
        update_map: i32,
        segid_preskip: i32,
        temporal_update: i32,
        seg_id_predicted: i32,
        pred_cdf: *mut u16,
        seg_cdf: *mut u16,
        seg_enabled: i32,
        segment_id: i32,
        seg_pred: i32,
        last_active_segid: i32,
        skip_mode_cdf: *mut u16,
        frame_skip_mode_flag: i32,
        sm_seg_skip: i32,
        sm_comp_allowed: i32,
        sm_seg_ref_gmv: i32,
        skip_mode: i32,
        skip_cdf: *mut u16,
        skip_seg_active: i32,
        skip_txfm: i32,
        coded_lossless: i32,
        allow_intrabc: i32,
        mi_row: i32,
        mi_col: i32,
        mib_size: i32,
        sb_size: i32,
        cdef_trans_in: *const i32,
        cdef_bits: i32,
        cdef_strength: i32,
        dq_present: i32,
        dlf_present: i32,
        dlf_multi: i32,
        num_planes: i32,
        bsize: i32,
        cur_qindex: i32,
        cur_base_qindex: i32,
        dq_res: i32,
        mbmi_dlf: *const i32,
        xd_dlf_in: *const i32,
        mbmi_dlf_base: i32,
        xd_dlf_base_in: i32,
        dlf_res: i32,
        dq_cdf: *mut u16,
        dlf_multi_cdf: *mut u16,
        dlf_cdf: *mut u16,
        intra_inter_cdf: *mut u16,
        seg_ref_frame_active: i32,
        seg_globalmv_active: i32,
        is_inter: i32,
        out: *mut u8,
        out_skip: *mut i32,
        out_skip_mode: *mut i32,
        o_predcdf: *mut u16,
        o_segcdf: *mut u16,
        o_smcdf: *mut u16,
        o_skipcdf: *mut u16,
        o_cdef_trans: *mut i32,
        o_dqcdf: *mut u16,
        o_dlfmcdf: *mut u16,
        o_dlfcdf: *mut u16,
        o_base: *mut i32,
        o_xd_dlf: *mut i32,
        o_xd_dlf_base: *mut i32,
        o_iicdf: *mut u16,
    ) -> u32;
    fn shim_use_angle_delta(bsize: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_delta_q_params_sb(
        dq_present: i32,
        dlf_present: i32,
        dlf_multi: i32,
        num_planes: i32,
        bsize: i32,
        sb_size: i32,
        skip: i32,
        sbul: i32,
        cur_qindex: i32,
        cur_base_qindex: i32,
        dq_res: i32,
        mbmi_dlf: *const i32,
        xd_dlf_in: *const i32,
        mbmi_dlf_base: i32,
        xd_dlf_base_in: i32,
        dlf_res: i32,
        dq_cdf: *mut u16,
        dlf_multi_cdf: *mut u16,
        dlf_cdf: *mut u16,
        out: *mut u8,
        o_dqcdf: *mut u16,
        o_dlfmcdf: *mut u16,
        o_dlfcdf: *mut u16,
        o_base: *mut i32,
        o_xd_dlf: *mut i32,
        o_xd_dlf_base: *mut i32,
    ) -> u32;
    fn shim_is_directional_mode(mode: i32) -> i32;
    fn shim_get_uv_mode(uv_mode: i32) -> i32;
    fn shim_allow_palette(allow_sct: i32, bsize: i32) -> i32;
    fn shim_is_cfl_allowed(bsize: i32, seg_id: i32, lossless: i32, ssx: i32, ssy: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_cdef(
        coded_lossless: i32,
        allow_intrabc: i32,
        mi_row: i32,
        mi_col: i32,
        mib_size: i32,
        sb_size: i32,
        skip: i32,
        transmitted_in: *const i32,
        cdef_bits: i32,
        cdef_strength: i32,
        out: *mut u8,
        transmitted_out: *mut i32,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_mb_modes_kf_prefix(
        segid_preskip: i32,
        seg_enabled: i32,
        update_map: i32,
        segment_id: i32,
        seg_pred: i32,
        last_active_segid: i32,
        seg_cdf: *mut u16,
        seg_skip_active: i32,
        skip_txfm: i32,
        skip_cdf: *mut u16,
        coded_lossless: i32,
        allow_intrabc: i32,
        mi_row: i32,
        mi_col: i32,
        mib_size: i32,
        sb_size: i32,
        cdef_trans_in: *const i32,
        cdef_bits: i32,
        cdef_strength: i32,
        dq_present: i32,
        dlf_present: i32,
        dlf_multi: i32,
        num_planes: i32,
        bsize: i32,
        cur_qindex: i32,
        cur_base_qindex: i32,
        dq_res: i32,
        mbmi_dlf: *const i32,
        xd_dlf_in: *const i32,
        mbmi_dlf_base: i32,
        xd_dlf_base_in: i32,
        dlf_res: i32,
        dq_cdf: *mut u16,
        dlf_multi_cdf: *mut u16,
        dlf_cdf: *mut u16,
        out: *mut u8,
        out_skip: *mut i32,
        o_segcdf: *mut u16,
        o_skipcdf: *mut u16,
        o_cdef_trans: *mut i32,
        o_dqcdf: *mut u16,
        o_dlfmcdf: *mut u16,
        o_dlfcdf: *mut u16,
        o_base: *mut i32,
        o_xd_dlf: *mut i32,
        o_xd_dlf_base: *mut i32,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_kf_tail(
        allow_intrabc: i32,
        intrabc_cdf: *mut u16,
        joints: *mut u16,
        comp0: *mut u16,
        comp1: *mut u16,
        use_intrabc: i32,
        diff_row: i32,
        diff_col: i32,
        mode: i32,
        bsize: i32,
        y_cdf: *mut u16,
        angle_delta_y: i32,
        y_angle_cdf: *mut u16,
        monochrome: i32,
        is_chroma_ref: i32,
        uv_mode: i32,
        cfl_allowed: i32,
        cfl_idx: i32,
        cfl_joint_sign: i32,
        angle_delta_uv: i32,
        uv_mode_cdf: *mut u16,
        cfl_sign_cdf: *mut u16,
        cfl_alpha_cdf: *mut u16,
        uv_angle_cdf: *mut u16,
        allow_palette: i32,
        bit_depth: i32,
        palette_size: *const u8,
        palette_colors: *const u16,
        mb_to_top_edge: i32,
        ha: i32,
        a_colors: *const u16,
        a_s0: i32,
        a_s1: i32,
        hl: i32,
        l_colors: *const u16,
        l_s0: i32,
        l_s1: i32,
        pal_y_mode_cdf: *mut u16,
        pal_y_size_cdf: *mut u16,
        pal_uv_mode_cdf: *mut u16,
        pal_uv_size_cdf: *mut u16,
        filter_allowed: i32,
        use_filter_intra: i32,
        filter_intra_mode: i32,
        fi_use_cdf: *mut u16,
        fi_mode_cdf: *mut u16,
        out: *mut u8,
        o_intrabc: *mut u16,
        o_joints: *mut u16,
        o_c0: *mut u16,
        o_c1: *mut u16,
        o_all: *mut u16,
    ) -> u32;
    fn shim_write_intra_y_and_angle(
        mode: i32,
        bsize: i32,
        y_cdf: *mut u16,
        angle_delta_y: i32,
        y_angle_cdf: *mut u16,
        out: *mut u8,
        o_ycdf: *mut u16,
        o_acdf: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_intra_uv_and_angle(
        monochrome: i32,
        is_chroma_ref: i32,
        uv_mode: i32,
        cfl_allowed: i32,
        bsize: i32,
        cfl_idx: i32,
        cfl_joint_sign: i32,
        angle_delta_uv: i32,
        uv_mode_cdf: *mut u16,
        cfl_sign_cdf: *mut u16,
        cfl_alpha_cdf: *mut u16,
        uv_angle_cdf: *mut u16,
        out: *mut u8,
        o_uvcdf: *mut u16,
        o_signcdf: *mut u16,
        o_alphacdf: *mut u16,
        o_uvacdf: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_intra_pred_modes(
        mode: i32,
        bsize: i32,
        y_cdf: *mut u16,
        angle_delta_y: i32,
        y_angle_cdf: *mut u16,
        monochrome: i32,
        is_chroma_ref: i32,
        uv_mode: i32,
        cfl_allowed: i32,
        cfl_idx: i32,
        cfl_joint_sign: i32,
        angle_delta_uv: i32,
        uv_mode_cdf: *mut u16,
        cfl_sign_cdf: *mut u16,
        cfl_alpha_cdf: *mut u16,
        uv_angle_cdf: *mut u16,
        allow_palette: i32,
        bit_depth: i32,
        palette_size: *const u8,
        palette_colors: *const u16,
        mb_to_top_edge: i32,
        ha: i32,
        a_colors: *const u16,
        a_s0: i32,
        a_s1: i32,
        hl: i32,
        l_colors: *const u16,
        l_s0: i32,
        l_s1: i32,
        pal_y_mode_cdf: *mut u16,
        pal_y_size_cdf: *mut u16,
        pal_uv_mode_cdf: *mut u16,
        pal_uv_size_cdf: *mut u16,
        filter_allowed: i32,
        use_filter_intra: i32,
        filter_intra_mode: i32,
        fi_use_cdf: *mut u16,
        fi_mode_cdf: *mut u16,
        out: *mut u8,
        o_all: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_comp_index_context(
        enable: i32,
        bits_minus_1: i32,
        cur_order_hint: i32,
        fwd_order_hint: i32,
        bck_order_hint: i32,
        ha: i32,
        a_has2: i32,
        a_cidx: i32,
        a_rf0: i32,
        hl: i32,
        l_has2: i32,
        l_cidx: i32,
        l_rf0: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_ref_frames(
        cdfs: *mut u16,
        seg_ref: i32,
        seg_skipgmv: i32,
        rmode_select: i32,
        comp_allowed: i32,
        is_compound: i32,
        comp_ref_type: i32,
        ref0: i32,
        ref1: i32,
        out: *mut u8,
        out_cdfs: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_comp_reference_type_context(
        ha: i32,
        a_r0: i32,
        a_r1: i32,
        a_ibc: i32,
        hl: i32,
        l_r0: i32,
        l_r1: i32,
        l_ibc: i32,
    ) -> i32;
    fn shim_write_motion_mode(
        obmc_cdf: *mut u16,
        mm_cdf: *mut u16,
        last_allowed: i32,
        mm: i32,
        out: *mut u8,
        out_obmc: *mut u16,
        out_mm: *mut u16,
    ) -> u32;
    fn shim_write_inter_compound_mode(
        cdf: *mut u16,
        mode: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_write_is_inter(
        cdf: *mut u16,
        seg_ref: i32,
        seg_gmv: i32,
        is_inter: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_write_filter_intra(
        use_cdf: *mut u16,
        mode_cdf: *mut u16,
        allowed: i32,
        use_fi: i32,
        mode: i32,
        out: *mut u8,
        out_use: *mut u16,
        out_mode: *mut u16,
    ) -> u32;
    fn shim_bsize_to_max_depth(bsize: i32) -> i32;
    fn shim_bsize_to_tx_size_cat(bsize: i32) -> i32;
    fn shim_write_selected_tx_size(
        cdf: *mut u16,
        bsize: i32,
        depth: i32,
        max_depths: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    fn shim_get_mv_joint(row: i32, col: i32) -> i32;
    fn shim_get_mv_class(z: i32) -> i32;
    fn shim_encode_mv_component(
        cdf: *mut u16,
        comp: i32,
        precision: i32,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_mv(
        joints_cdf: *mut u16,
        comp0: *mut u16,
        comp1: *mut u16,
        diff_row: i32,
        diff_col: i32,
        usehp: i32,
        out: *mut u8,
        out_joints: *mut u16,
        out_comp0: *mut u16,
        out_comp1: *mut u16,
    ) -> u32;
    fn shim_write_drl_idx(
        drl_cdf: *mut u16,
        mode: i32,
        ref_mv_idx: i32,
        ref_mv_count: i32,
        weight: *const u16,
        out: *mut u8,
        out_cdf: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_inter_mode(
        newmv_cdf: *mut u16,
        zeromv_cdf: *mut u16,
        refmv_cdf: *mut u16,
        mode: i32,
        mode_ctx: i32,
        out: *mut u8,
        out_newmv: *mut u16,
        out_zeromv: *mut u16,
        out_refmv: *mut u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_frame_header_trailing_flags(
        intra_only: i32,
        ref_mode_select: i32,
        skip_allowed: i32,
        skip_flag: i32,
        might_warp: i32,
        allow_warp: i32,
        reduced_tx_set: i32,
        out: *mut u8,
    ) -> u32;
}

/// Reference `partition_cdf_length`.
pub fn ref_partition_cdf_length(bsize: i32) -> i32 {
    unsafe { shim_partition_cdf_length(bsize) }
}

/// Reference `av1_encode_mv` (joint + 2 components over the pristine C od_ec + real helpers).
#[allow(clippy::type_complexity)]
pub fn ref_encode_mv(
    joints_cdf: &[u16; 5],
    comp0: &[u16; 69],
    comp1: &[u16; 69],
    diff_row: i32,
    diff_col: i32,
    usehp: i32,
) -> (Vec<u8>, [u16; 5], [u16; 69], [u16; 69]) {
    let mut jc = *joints_cdf;
    let mut c0 = *comp0;
    let mut c1 = *comp1;
    let mut out = vec![0u8; 48];
    let mut oj = [0u16; 5];
    let mut o0 = [0u16; 69];
    let mut o1 = [0u16; 69];
    let n = unsafe {
        shim_encode_mv(
            jc.as_mut_ptr(),
            c0.as_mut_ptr(),
            c1.as_mut_ptr(),
            diff_row,
            diff_col,
            usehp,
            out.as_mut_ptr(),
            oj.as_mut_ptr(),
            o0.as_mut_ptr(),
            o1.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oj, o0, o1)
}

/// Reference `encode_mv_component` (over the pristine C od_ec + the real av1_get_mv_class).
pub fn ref_encode_mv_component(cdf: &[u16; 69], comp: i32, precision: i32) -> (Vec<u8>, [u16; 69]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 32];
    let mut out_cdf = [0u16; 69];
    let n = unsafe {
        shim_encode_mv_component(
            c.as_mut_ptr(),
            comp,
            precision,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_mb_interp_filter`.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_mb_interp_filter(
    cdf0: &[u16; 4],
    cdf1: &[u16; 4],
    interp_needed: bool,
    is_switchable: bool,
    enable_dual: bool,
    f0: i32,
    f1: i32,
) -> (Vec<u8>, [u16; 4], [u16; 4]) {
    let mut c0 = *cdf0;
    let mut c1 = *cdf1;
    let mut out = vec![0u8; 16];
    let mut o0 = [0u16; 4];
    let mut o1 = [0u16; 4];
    let n = unsafe {
        shim_write_mb_interp_filter(
            c0.as_mut_ptr(),
            c1.as_mut_ptr(),
            interp_needed as i32,
            is_switchable as i32,
            enable_dual as i32,
            f0,
            f1,
            out.as_mut_ptr(),
            o0.as_mut_ptr(),
            o1.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, o0, o1)
}

/// Reference `av1_get_pred_context_single_ref_p2` (brfarf2_or_arf).
pub fn ref_single_ref_p2_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_single_ref_p2_context(rc.as_ptr()) }
}

/// Reference `av1_get_pred_context_single_ref_p3` (ll2_or_l3gld).
pub fn ref_single_ref_p3_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_single_ref_p3_context(rc.as_ptr()) }
}

/// Reference `av1_get_pred_context_single_ref_p4` (last_or_last2).
pub fn ref_single_ref_p4_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_single_ref_p4_context(rc.as_ptr()) }
}

/// Reference `av1_get_pred_context_single_ref_p5` (last3_or_gld).
pub fn ref_single_ref_p5_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_single_ref_p5_context(rc.as_ptr()) }
}

/// Reference `av1_get_pred_context_single_ref_p6` (brf_or_arf2).
pub fn ref_single_ref_p6_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_single_ref_p6_context(rc.as_ptr()) }
}

/// Reference `write_ref_frames` (cascade over the pristine C od_ec + update_cdf).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_ref_frames(
    cdfs: &[u16; 48],
    seg_ref: bool,
    seg_skipgmv: bool,
    rmode_select: bool,
    comp_allowed: bool,
    is_compound: bool,
    comp_ref_type: i32,
    ref0: i32,
    ref1: i32,
) -> (Vec<u8>, [u16; 48]) {
    let mut c = *cdfs;
    let mut out = vec![0u8; 32];
    let mut oc = [0u16; 48];
    let n = unsafe {
        shim_write_ref_frames(
            c.as_mut_ptr(),
            seg_ref as i32,
            seg_skipgmv as i32,
            rmode_select as i32,
            comp_allowed as i32,
            is_compound as i32,
            comp_ref_type,
            ref0,
            ref1,
            out.as_mut_ptr(),
            oc.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oc)
}

/// Reference `write_intrabc_info` (flag + av1_encode_dv over the pristine C od_ec).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_intrabc_info(
    intrabc_cdf: &[u16; 3],
    joints: &[u16; 5],
    comp0: &[u16; 69],
    comp1: &[u16; 69],
    use_intrabc: i32,
    diff_row: i32,
    diff_col: i32,
) -> (Vec<u8>, [u16; 3], [u16; 5], [u16; 69], [u16; 69]) {
    let mut ib = *intrabc_cdf;
    let mut jc = *joints;
    let mut c0 = *comp0;
    let mut c1 = *comp1;
    let mut out = vec![0u8; 48];
    let mut oib = [0u16; 3];
    let mut oj = [0u16; 5];
    let mut o0 = [0u16; 69];
    let mut o1 = [0u16; 69];
    let n = unsafe {
        shim_write_intrabc_info(
            ib.as_mut_ptr(),
            jc.as_mut_ptr(),
            c0.as_mut_ptr(),
            c1.as_mut_ptr(),
            use_intrabc,
            diff_row,
            diff_col,
            out.as_mut_ptr(),
            oib.as_mut_ptr(),
            oj.as_mut_ptr(),
            o0.as_mut_ptr(),
            o1.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oib, oj, o0, o1)
}

/// Reference `av1_neg_interleave` (real exported fn).
pub fn ref_neg_interleave(x: i32, ref_: i32, max: i32) -> i32 {
    unsafe { shim_neg_interleave(x, ref_, max) }
}

/// Reference `write_segment_id` (over the pristine C od_ec + update_cdf).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_segment_id(
    cdf: &[u16; 9],
    seg_enabled: bool,
    update_map: bool,
    skip_txfm: bool,
    segment_id: i32,
    pred: i32,
    last_active_segid: i32,
) -> (Vec<u8>, [u16; 9]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut oc = [0u16; 9];
    let n = unsafe {
        shim_write_segment_id(
            c.as_mut_ptr(),
            seg_enabled as i32,
            update_map as i32,
            skip_txfm as i32,
            segment_id,
            pred,
            last_active_segid,
            out.as_mut_ptr(),
            oc.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oc)
}

/// Reference the 3 uni-comp-ref contexts (facades over the real exported fns).
pub fn ref_uni_comp_ref_p_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_uni_comp_ref_p_context(rc.as_ptr()) }
}
pub fn ref_uni_comp_ref_p1_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_uni_comp_ref_p1_context(rc.as_ptr()) }
}
pub fn ref_uni_comp_ref_p2_context(rc: &[u8; 8]) -> i32 {
    unsafe { shim_uni_comp_ref_p2_context(rc.as_ptr()) }
}

/// Reference `av1_get_pred_context_single_ref_p1` (facade over the real exported fn).
pub fn ref_single_ref_p1_context(ref_counts: &[u8; 8]) -> i32 {
    unsafe { shim_single_ref_p1_context(ref_counts.as_ptr()) }
}

/// Reference `av1_get_comp_reference_type_context` (facade over the real exported fn).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_comp_reference_type_context(
    ha: bool,
    a_r0: i32,
    a_r1: i32,
    a_ibc: bool,
    hl: bool,
    l_r0: i32,
    l_r1: i32,
    l_ibc: bool,
) -> i32 {
    unsafe {
        shim_get_comp_reference_type_context(
            ha as i32,
            a_r0,
            a_r1,
            a_ibc as i32,
            hl as i32,
            l_r0,
            l_r1,
            l_ibc as i32,
        )
    }
}

/// Reference `av1_get_reference_mode_context` (facade over the real exported fn).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_reference_mode_context(
    ha: bool,
    a_r0: i32,
    a_r1: i32,
    a_ibc: bool,
    hl: bool,
    l_r0: i32,
    l_r1: i32,
    l_ibc: bool,
) -> i32 {
    unsafe {
        shim_get_reference_mode_context(
            ha as i32,
            a_r0,
            a_r1,
            a_ibc as i32,
            hl as i32,
            l_r0,
            l_r1,
            l_ibc as i32,
        )
    }
}

/// Reference `av1_get_skip_mode_context` (facade over the real fn).
pub fn ref_get_skip_mode_context(ha: bool, a_sm: i32, hl: bool, l_sm: i32) -> i32 {
    unsafe { shim_get_skip_mode_context(ha as i32, a_sm, hl as i32, l_sm) }
}

/// Reference `write_skip_mode` (over the pristine C od_ec + update_cdf).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_skip_mode(
    cdf: &[u16; 3],
    frame_flag: bool,
    seg_skip: bool,
    comp_allowed: bool,
    seg_ref_gmv: bool,
    skip_mode: i32,
) -> (Vec<u8>, [u16; 3]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut oc = [0u16; 3];
    let n = unsafe {
        shim_write_skip_mode(
            c.as_mut_ptr(),
            frame_flag as i32,
            seg_skip as i32,
            comp_allowed as i32,
            seg_ref_gmv as i32,
            skip_mode,
            out.as_mut_ptr(),
            oc.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oc)
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
            bsize,
            top_tx_size,
            inter_tx_size.as_ptr(),
            mb_to_right_edge,
            mb_to_bottom_edge,
            above_in.as_ptr(),
            left_in.as_ptr(),
            c.as_mut_ptr(),
            out.as_mut_ptr(),
            ao.as_mut_ptr(),
            lo.as_mut_ptr(),
            co.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ao, lo, co)
}

/// Reference block-level inter var-tx-size loop (write_modes_b portion, over pristine C
/// od_ec). Returns (bytes, above_ctx[32], left_ctx[32], adapted_cdf[21][3]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_txfm_size(
    bsize: i32,
    max_tx: i32,
    inter_tx_size: &[u8; 16],
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    above_in: &[u8; 32],
    left_in: &[u8; 32],
    cdf: &[u16; 63],
) -> (Vec<u8>, [u8; 32], [u8; 32], [u16; 63]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 128];
    let mut ao = [0u8; 32];
    let mut lo = [0u8; 32];
    let mut co = [0u16; 63];
    let n = unsafe {
        shim_write_inter_txfm_size(
            bsize,
            max_tx,
            inter_tx_size.as_ptr(),
            mb_to_right_edge,
            mb_to_bottom_edge,
            above_in.as_ptr(),
            left_in.as_ptr(),
            c.as_mut_ptr(),
            out.as_mut_ptr(),
            ao.as_mut_ptr(),
            lo.as_mut_ptr(),
            co.as_mut_ptr(),
        )
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
            mode_dc as i32,
            n_y,
            ym.as_mut_ptr(),
            ys.as_mut_ptr(),
            uv_dc as i32,
            n_uv,
            um.as_mut_ptr(),
            us.as_mut_ptr(),
            out.as_mut_ptr(),
            oym.as_mut_ptr(),
            oys.as_mut_ptr(),
            oum.as_mut_ptr(),
            ous.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oym, oys, oum, ous)
}

/// Reference `pack_map_tokens` (palette colour-index map, over pristine C od_ec). map_cdf
/// is the [PALETTE_COLOR_INDEX_CONTEXTS=5][9] slice for the palette size. Returns
/// (bytes, adapted map_cdf[45]).
pub fn ref_pack_map_tokens(
    n: i32,
    tokens: &[u8],
    color_ctxs: &[u8],
    map_cdf: &[u16; 45],
) -> (Vec<u8>, [u16; 45]) {
    let mut mc = *map_cdf;
    let mut out = vec![0u8; 256];
    let mut mco = [0u16; 45];
    let num = tokens.len() as i32;
    let n_out = unsafe {
        shim_pack_map_tokens(
            n,
            num,
            tokens.as_ptr(),
            color_ctxs.as_ptr(),
            mc.as_mut_ptr(),
            out.as_mut_ptr(),
            mco.as_mut_ptr(),
        )
    };
    out.truncate(n_out as usize);
    (out, mco)
}

/// Reference `delta_encode_palette_colors` (over pristine C od_ec, real aom_ceil_log2).
pub fn ref_delta_encode_palette_colors(colors: &[i32], bit_depth: i32, min_val: i32) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_delta_encode_palette_colors(
            colors.as_ptr(),
            colors.len() as i32,
            bit_depth,
            min_val,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference V-channel palette colour coding (write_palette_colors_uv's V portion,
/// over pristine C od_ec, real av1_get_palette_delta_bits_v). colors_v need not be sorted.
pub fn ref_write_palette_colors_v(colors_v: &[u16], bit_depth: i32) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_write_palette_colors_v(
            colors_v.as_ptr(),
            colors_v.len() as i32,
            bit_depth,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference interintra sub-symbols (write_mbmi_b portion, over pristine C od_ec). CDFs
/// are pre-selected. Returns (bytes, ii_cdf[3], ii_mode_cdf[5], wedge_ii_cdf[3],
/// wedge_idx_cdf[17]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_interintra_info(
    interintra: i32,
    ii_cdf: &[u16; 3],
    ii_mode: i32,
    ii_mode_cdf: &[u16; 5],
    wedge_used: bool,
    use_wedge: i32,
    wedge_ii_cdf: &[u16; 3],
    wedge_index: i32,
    wedge_idx_cdf: &[u16; 17],
) -> (Vec<u8>, [u16; 3], [u16; 5], [u16; 3], [u16; 17]) {
    let (mut ii, mut iim, mut wii, mut wix) =
        (*ii_cdf, *ii_mode_cdf, *wedge_ii_cdf, *wedge_idx_cdf);
    let mut out = vec![0u8; 32];
    let (mut oii, mut oiim, mut owii, mut owix) = ([0u16; 3], [0u16; 5], [0u16; 3], [0u16; 17]);
    let n = unsafe {
        shim_write_interintra_info(
            interintra,
            ii.as_mut_ptr(),
            ii_mode,
            iim.as_mut_ptr(),
            wedge_used as i32,
            use_wedge,
            wii.as_mut_ptr(),
            wedge_index,
            wix.as_mut_ptr(),
            out.as_mut_ptr(),
            oii.as_mut_ptr(),
            oiim.as_mut_ptr(),
            owii.as_mut_ptr(),
            owix.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oii, oiim, owii, owix)
}

/// Reference `get_comp_group_idx_context` (facade over the real static inline).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_comp_group_idx_context(
    ha: bool,
    a_rf0: i32,
    a_rf1: i32,
    a_cgi: i32,
    hl: bool,
    l_rf0: i32,
    l_rf1: i32,
    l_cgi: i32,
) -> i32 {
    unsafe {
        shim_get_comp_group_idx_context(
            ha as i32, a_rf0, a_rf1, a_cgi, hl as i32, l_rf0, l_rf1, l_cgi,
        )
    }
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
pub fn ref_is_inter_compound_mode(mode: i32) -> bool {
    unsafe { shim_is_inter_compound_mode(mode) != 0 }
}
pub fn ref_is_inter_singleref_mode(mode: i32) -> bool {
    unsafe { shim_is_inter_singleref_mode(mode) != 0 }
}
pub fn ref_have_nearmv_in_inter_mode(mode: i32) -> bool {
    unsafe { shim_have_nearmv_in_inter_mode(mode) != 0 }
}
pub fn ref_mode_context_analyzer(rf0: i32, rf1: i32, mc_val: i32) -> i32 {
    unsafe { shim_mode_context_analyzer(rf0, rf1, mc_val) }
}

/// Reference `av1_collect_neighbors_ref_counts` (facade): the 8-entry ref-frame tally
/// from the above/left inter neighbours.
#[allow(clippy::too_many_arguments)]
pub fn ref_collect_neighbors_ref_counts(
    ha: bool,
    a_intrabc: bool,
    a_rf0: i32,
    a_rf1: i32,
    hl: bool,
    l_intrabc: bool,
    l_rf0: i32,
    l_rf1: i32,
) -> [u8; 8] {
    let mut counts = [0u8; 8];
    unsafe {
        shim_collect_neighbors_ref_counts(
            ha as i32,
            a_intrabc as i32,
            a_rf0,
            a_rf1,
            hl as i32,
            l_intrabc as i32,
            l_rf0,
            l_rf1,
            counts.as_mut_ptr(),
        )
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
    n_sb_rows: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: i32,
    tree: &[i8],
    arena: &[u16; 220],
) -> (Vec<u8>, [i8; 128], [u16; 220], i32) {
    let mut ar = *arena;
    let mut out = vec![0u8; 512];
    let (mut ao, mut aro) = ([0i8; 128], [0u16; 220]);
    let mut consumed = 0i32;
    let n = unsafe {
        shim_write_modes_tile(
            n_sb_rows,
            n_sb_cols,
            sb_mi,
            sb_size,
            tree.as_ptr(),
            ar.as_mut_ptr(),
            out.as_mut_ptr(),
            ao.as_mut_ptr(),
            aro.as_mut_ptr(),
            &mut consumed,
        )
    };
    out.truncate(n as usize);
    (out, ao, aro, consumed)
}

/// Reference `write_modes_sb` partition-tree recursion (fully-in-frame, stubbed blocks).
/// tree is the pre-order partition sequence. Returns (bytes, above[64], left[32],
/// arena[220], tree_consumed).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_modes_sb(
    above_in: &[i8; 64],
    left_in: &[i8; 32],
    mi_row: i32,
    mi_col: i32,
    bsize: i32,
    tree: &[i8],
    arena: &[u16; 220],
) -> (Vec<u8>, [i8; 64], [i8; 32], [u16; 220], i32) {
    let mut ar = *arena;
    let mut out = vec![0u8; 256];
    let (mut ao, mut lo, mut aro) = ([0i8; 64], [0i8; 32], [0u16; 220]);
    let mut consumed = 0i32;
    let n = unsafe {
        shim_write_modes_sb(
            above_in.as_ptr(),
            left_in.as_ptr(),
            mi_row,
            mi_col,
            bsize,
            tree.as_ptr(),
            tree.len() as i32,
            ar.as_mut_ptr(),
            out.as_mut_ptr(),
            ao.as_mut_ptr(),
            lo.as_mut_ptr(),
            aro.as_mut_ptr(),
            &mut consumed,
        )
    };
    out.truncate(n as usize);
    (out, ao, lo, aro, consumed)
}

/// Reference `write_modes_sb` per-node partition step (context-select -> write_partition
/// -> context-update, over one od_ec). arena is [PARTITION_CONTEXTS=20][11] = 220 flat.
/// Returns (bytes, above[64], left[32], arena[220]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_partition_node(
    above_in: &[i8; 64],
    left_in: &[i8; 32],
    mi_row: i32,
    mi_col: i32,
    bsize: i32,
    partition: i32,
    mi_rows: i32,
    mi_cols: i32,
    arena: &[u16; 220],
) -> (Vec<u8>, [i8; 64], [i8; 32], [u16; 220]) {
    let mut ar = *arena;
    let mut out = vec![0u8; 16];
    let (mut ao, mut lo, mut aro) = ([0i8; 64], [0i8; 32], [0u16; 220]);
    let n = unsafe {
        shim_write_partition_node(
            above_in.as_ptr(),
            left_in.as_ptr(),
            mi_row,
            mi_col,
            bsize,
            partition,
            mi_rows,
            mi_cols,
            ar.as_mut_ptr(),
            out.as_mut_ptr(),
            ao.as_mut_ptr(),
            lo.as_mut_ptr(),
            aro.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ao, lo, aro)
}

/// Reference `update_ext_partition_context` (facade). above is a 64-slot buffer, left a
/// 32-slot (MAX_MIB_SIZE) buffer. Returns the updated (above[64], left[32]).
#[allow(clippy::too_many_arguments)]
pub fn ref_update_ext_partition_context(
    mi_row: i32,
    mi_col: i32,
    subsize: i32,
    bsize: i32,
    partition: i32,
    above_in: &[i8; 64],
    left_in: &[i8; 32],
) -> ([i8; 64], [i8; 32]) {
    let (mut ao, mut lo) = ([0i8; 64], [0i8; 32]);
    unsafe {
        shim_update_ext_partition_context(
            mi_row,
            mi_col,
            subsize,
            bsize,
            partition,
            above_in.as_ptr(),
            left_in.as_ptr(),
            ao.as_mut_ptr(),
            lo.as_mut_ptr(),
        )
    };
    (ao, lo)
}

/// Reference inter-block MV coding (the mode-dependent av1_encode_mv calls, over pristine
/// C od_ec). Returns (bytes, joints[5], comp0[69], comp1[69]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_block_mvs(
    mode: i32,
    is_compound: bool,
    diff_row0: i32,
    diff_col0: i32,
    diff_row1: i32,
    diff_col1: i32,
    usehp: i32,
    joints: &[u16; 5],
    comp0: &[u16; 69],
    comp1: &[u16; 69],
) -> (Vec<u8>, [u16; 5], [u16; 69], [u16; 69]) {
    let (mut jo, mut c0, mut c1) = (*joints, *comp0, *comp1);
    let mut out = vec![0u8; 64];
    let (mut ojo, mut oc0, mut oc1) = ([0u16; 5], [0u16; 69], [0u16; 69]);
    let n = unsafe {
        shim_write_inter_block_mvs(
            mode,
            is_compound as i32,
            diff_row0,
            diff_col0,
            diff_row1,
            diff_col1,
            usehp,
            jo.as_mut_ptr(),
            c0.as_mut_ptr(),
            c1.as_mut_ptr(),
            out.as_mut_ptr(),
            ojo.as_mut_ptr(),
            oc0.as_mut_ptr(),
            oc1.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ojo, oc0, oc1)
}

/// Reference inter mode + drl coding (over pristine C od_ec). inter_compound_mode_cdf is
/// pre-selected [mode_ctx] (9 entries); newmv/refmv are [6][3], zeromv [2][3], drl [3][3].
/// Returns (bytes, icm[9], newmv[18], zeromv[6], refmv[18], drl[9]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_inter_mode_drl(
    seg_skip: bool,
    mode: i32,
    mode_ctx: i32,
    icm_cdf: &[u16; 9],
    newmv_cdf: &[u16; 18],
    zeromv_cdf: &[u16; 6],
    refmv_cdf: &[u16; 18],
    drl_cdf: &[u16; 9],
    ref_mv_idx: i32,
    ref_mv_count: i32,
    weight: &[u16],
) -> (Vec<u8>, [u16; 9], [u16; 18], [u16; 6], [u16; 18], [u16; 9]) {
    let (mut icm, mut nm, mut zm, mut rm, mut drl) =
        (*icm_cdf, *newmv_cdf, *zeromv_cdf, *refmv_cdf, *drl_cdf);
    let mut out = vec![0u8; 32];
    let (mut oicm, mut onm, mut ozm, mut orm, mut odrl) =
        ([0u16; 9], [0u16; 18], [0u16; 6], [0u16; 18], [0u16; 9]);
    let n = unsafe {
        shim_write_inter_mode_drl(
            seg_skip as i32,
            mode,
            mode_ctx,
            icm.as_mut_ptr(),
            nm.as_mut_ptr(),
            zm.as_mut_ptr(),
            rm.as_mut_ptr(),
            drl.as_mut_ptr(),
            ref_mv_idx,
            ref_mv_count,
            weight.as_ptr(),
            out.as_mut_ptr(),
            oicm.as_mut_ptr(),
            onm.as_mut_ptr(),
            ozm.as_mut_ptr(),
            orm.as_mut_ptr(),
            odrl.as_mut_ptr(),
        )
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
            inp.interintra_allowed as i32,
            inp.interintra,
            ii.as_mut_ptr(),
            inp.ii_mode,
            iim.as_mut_ptr(),
            inp.wedge_used_ii as i32,
            inp.use_wedge_ii,
            wii.as_mut_ptr(),
            inp.ii_wedge_index,
            wix.as_mut_ptr(),
            inp.motion_mode_present as i32,
            obmc.as_mut_ptr(),
            mm.as_mut_ptr(),
            inp.last_motion_mode_allowed,
            inp.motion_mode,
            inp.has_second_ref as i32,
            inp.masked_used as i32,
            inp.comp_group_idx,
            cgi.as_mut_ptr(),
            inp.dist_wtd as i32,
            inp.compound_idx,
            cidx.as_mut_ptr(),
            inp.wedge_used_ct as i32,
            inp.comp_type,
            ct.as_mut_ptr(),
            inp.ct_wedge_index,
            inp.wedge_sign,
            inp.mask_type,
            inp.interp_needed as i32,
            inp.is_switchable as i32,
            inp.enable_dual as i32,
            inp.f0,
            inp.f1,
            ic0.as_mut_ptr(),
            ic1.as_mut_ptr(),
            out.as_mut_ptr(),
            o_all.as_mut_ptr(),
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
    let (mut opc, mut osc, mut osmc, mut oskc, mut octr) =
        ([0u16; 3], [0u16; 9], [0u16; 3], [0u16; 3], [0i32; 4]);
    let (mut odqc, mut odlmc, mut odlc) = ([0u16; 5], [0u16; 20], [0u16; 5]);
    let (mut obase, mut oxd, mut oxdb, mut oiic) = (0i32, [0i32; 4], 0i32, [0u16; 3]);
    let n = unsafe {
        shim_write_inter_prefix(
            inp.update_map as i32,
            inp.segid_preskip as i32,
            inp.temporal_update as i32,
            inp.seg_id_predicted,
            pc.as_mut_ptr(),
            sc.as_mut_ptr(),
            inp.seg_enabled as i32,
            inp.segment_id,
            inp.seg_pred,
            inp.last_active_segid,
            smc.as_mut_ptr(),
            inp.frame_skip_mode_flag as i32,
            inp.sm_seg_skip as i32,
            inp.sm_comp_allowed as i32,
            inp.sm_seg_ref_gmv as i32,
            inp.skip_mode,
            skc.as_mut_ptr(),
            inp.skip_seg_active as i32,
            inp.skip_txfm,
            inp.coded_lossless as i32,
            inp.allow_intrabc as i32,
            inp.mi_row,
            inp.mi_col,
            inp.mib_size,
            inp.sb_size,
            inp.cdef_trans.as_ptr(),
            inp.cdef_bits,
            inp.cdef_strength,
            inp.dq_present as i32,
            inp.dlf_present as i32,
            inp.dlf_multi as i32,
            inp.num_planes,
            inp.bsize,
            inp.cur_qindex,
            inp.cur_base_qindex,
            inp.dq_res,
            inp.mbmi_dlf.as_ptr(),
            inp.xd_dlf.as_ptr(),
            inp.mbmi_dlf_base,
            inp.xd_dlf_base,
            inp.dlf_res,
            dqc.as_mut_ptr(),
            dlmc.as_mut_ptr(),
            dlc.as_mut_ptr(),
            iic.as_mut_ptr(),
            inp.seg_ref_frame_active as i32,
            inp.seg_globalmv_active as i32,
            inp.is_inter,
            out.as_mut_ptr(),
            &mut skip,
            &mut skip_mode,
            opc.as_mut_ptr(),
            osc.as_mut_ptr(),
            osmc.as_mut_ptr(),
            oskc.as_mut_ptr(),
            octr.as_mut_ptr(),
            odqc.as_mut_ptr(),
            odlmc.as_mut_ptr(),
            odlc.as_mut_ptr(),
            &mut obase,
            oxd.as_mut_ptr(),
            &mut oxdb,
            oiic.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    InterPrefixOut {
        bytes: out,
        skip,
        skip_mode,
        pred_cdf: opc,
        seg_cdf: osc,
        skip_mode_cdf: osmc,
        skip_cdf: oskc,
        cdef_trans: octr,
        dq_cdf: odqc,
        dlf_multi_cdf: odlmc,
        dlf_cdf: odlc,
        base_qindex: obase,
        xd_dlf: oxd,
        xd_dlf_base: oxdb,
        intra_inter_cdf: oiic,
    }
}

/// Reference `write_inter_segment_id` (bitstream.c:920, over pristine C od_ec). Returns
/// (bytes, pred_cdf[3], seg_cdf[9]).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_segment_id(
    update_map: bool,
    preskip: bool,
    segid_preskip: bool,
    skip: bool,
    temporal_update: bool,
    seg_id_predicted: i32,
    pred_cdf: &[u16; 3],
    seg_cdf: &[u16; 9],
    seg_enabled: bool,
    segment_id: i32,
    seg_pred: i32,
    last_active_segid: i32,
) -> (Vec<u8>, [u16; 3], [u16; 9]) {
    let (mut pc, mut sc) = (*pred_cdf, *seg_cdf);
    let mut out = vec![0u8; 16];
    let (mut opc, mut osc) = ([0u16; 3], [0u16; 9]);
    let n = unsafe {
        shim_write_inter_segment_id(
            update_map as i32,
            preskip as i32,
            segid_preskip as i32,
            skip as i32,
            temporal_update as i32,
            seg_id_predicted,
            pc.as_mut_ptr(),
            sc.as_mut_ptr(),
            seg_enabled as i32,
            segment_id,
            seg_pred,
            last_active_segid,
            out.as_mut_ptr(),
            opc.as_mut_ptr(),
            osc.as_mut_ptr(),
        )
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
    dq_present: bool,
    dlf_present: bool,
    dlf_multi: bool,
    num_planes: i32,
    bsize: i32,
    sb_size: i32,
    skip: i32,
    sbul: bool,
    cur_qindex: i32,
    cur_base_qindex: i32,
    dq_res: i32,
    mbmi_dlf: &[i32; 4],
    xd_dlf: &[i32; 4],
    mbmi_dlf_base: i32,
    xd_dlf_base: i32,
    dlf_res: i32,
    dq_cdf: &[u16; 5],
    dlf_multi_cdf: &[u16; 20],
    dlf_cdf: &[u16; 5],
) -> (Vec<u8>, [u16; 5], [u16; 20], [u16; 5], i32, [i32; 4], i32) {
    let (mut dqc, mut dlmc, mut dlc) = (*dq_cdf, *dlf_multi_cdf, *dlf_cdf);
    let mut out = vec![0u8; 64];
    let (mut odqc, mut odlmc, mut odlc) = ([0u16; 5], [0u16; 20], [0u16; 5]);
    let (mut o_base, mut o_xd_dlf, mut o_xd_dlf_base) = (0i32, [0i32; 4], 0i32);
    let n = unsafe {
        shim_write_delta_q_params_sb(
            dq_present as i32,
            dlf_present as i32,
            dlf_multi as i32,
            num_planes,
            bsize,
            sb_size,
            skip,
            sbul as i32,
            cur_qindex,
            cur_base_qindex,
            dq_res,
            mbmi_dlf.as_ptr(),
            xd_dlf.as_ptr(),
            mbmi_dlf_base,
            xd_dlf_base,
            dlf_res,
            dqc.as_mut_ptr(),
            dlmc.as_mut_ptr(),
            dlc.as_mut_ptr(),
            out.as_mut_ptr(),
            odqc.as_mut_ptr(),
            odlmc.as_mut_ptr(),
            odlc.as_mut_ptr(),
            &mut o_base,
            o_xd_dlf.as_mut_ptr(),
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
    coded_lossless: bool,
    allow_intrabc: bool,
    mi_row: i32,
    mi_col: i32,
    mib_size: i32,
    sb_size: i32,
    skip: i32,
    transmitted: &[i32; 4],
    cdef_bits: i32,
    cdef_strength: i32,
) -> (Vec<u8>, [i32; 4]) {
    let mut out = vec![0u8; 8];
    let mut tout = [0i32; 4];
    let n = unsafe {
        shim_write_cdef(
            coded_lossless as i32,
            allow_intrabc as i32,
            mi_row,
            mi_col,
            mib_size,
            sb_size,
            skip,
            transmitted.as_ptr(),
            cdef_bits,
            cdef_strength,
            out.as_mut_ptr(),
            tout.as_mut_ptr(),
        )
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
            inp.segid_preskip as i32,
            inp.seg_enabled as i32,
            inp.update_map as i32,
            inp.segment_id,
            inp.seg_pred,
            inp.last_active_segid,
            seg.as_mut_ptr(),
            inp.seg_skip_active as i32,
            inp.skip_txfm,
            skc.as_mut_ptr(),
            inp.coded_lossless as i32,
            inp.allow_intrabc as i32,
            inp.mi_row,
            inp.mi_col,
            inp.mib_size,
            inp.sb_size,
            inp.cdef_trans.as_ptr(),
            inp.cdef_bits,
            inp.cdef_strength,
            inp.dq_present as i32,
            inp.dlf_present as i32,
            inp.dlf_multi as i32,
            inp.num_planes,
            inp.bsize,
            inp.cur_qindex,
            inp.cur_base_qindex,
            inp.dq_res,
            inp.mbmi_dlf.as_ptr(),
            inp.xd_dlf.as_ptr(),
            inp.mbmi_dlf_base,
            inp.xd_dlf_base,
            inp.dlf_res,
            dqc.as_mut_ptr(),
            dlmc.as_mut_ptr(),
            dlc.as_mut_ptr(),
            out.as_mut_ptr(),
            &mut skip,
            oseg.as_mut_ptr(),
            oskc.as_mut_ptr(),
            octr.as_mut_ptr(),
            odqc.as_mut_ptr(),
            odlmc.as_mut_ptr(),
            odlc.as_mut_ptr(),
            &mut obase,
            oxd.as_mut_ptr(),
            &mut oxdb,
        )
    };
    out.truncate(n as usize);
    KfPrefixOut {
        bytes: out,
        skip,
        seg_cdf: oseg,
        skip_cdf: oskc,
        cdef_trans: octr,
        dq_cdf: odqc,
        dlf_multi_cdf: odlmc,
        dlf_cdf: odlc,
        base_qindex: obase,
        xd_dlf: oxd,
        xd_dlf_base: oxdb,
    }
}

/// Reference `av1_use_angle_delta` / `av1_is_directional_mode` / `get_uv_mode` /
/// `av1_allow_palette` / `is_cfl_allowed` — the intra-prediction-mode driver gates.
pub fn ref_use_angle_delta(bsize: i32) -> bool {
    unsafe { shim_use_angle_delta(bsize) != 0 }
}
pub fn ref_is_directional_mode(mode: i32) -> bool {
    unsafe { shim_is_directional_mode(mode) != 0 }
}
pub fn ref_get_uv_mode(uv_mode: i32) -> i32 {
    unsafe { shim_get_uv_mode(uv_mode) }
}
pub fn ref_allow_palette(allow_sct: bool, bsize: i32) -> bool {
    unsafe { shim_allow_palette(allow_sct as i32, bsize) != 0 }
}
pub fn ref_is_cfl_allowed(bsize: i32, seg_id: i32, lossless: bool, ssx: i32, ssy: i32) -> bool {
    unsafe { shim_is_cfl_allowed(bsize, seg_id, lossless as i32, ssx, ssy) != 0 }
}

/// Reference write_intra_prediction_modes piece 1 (Y mode + gated Y angle delta, over
/// pristine C od_ec + real gates). Returns (bytes, y_cdf[14], y_angle_cdf[8]).
pub fn ref_write_intra_y_and_angle(
    mode: i32,
    bsize: i32,
    y_cdf: &[u16; 14],
    angle_delta_y: i32,
    y_angle_cdf: &[u16; 8],
) -> (Vec<u8>, [u16; 14], [u16; 8]) {
    let (mut yc, mut ac) = (*y_cdf, *y_angle_cdf);
    let mut out = vec![0u8; 16];
    let (mut oyc, mut oac) = ([0u16; 14], [0u16; 8]);
    let n = unsafe {
        shim_write_intra_y_and_angle(
            mode,
            bsize,
            yc.as_mut_ptr(),
            angle_delta_y,
            ac.as_mut_ptr(),
            out.as_mut_ptr(),
            oyc.as_mut_ptr(),
            oac.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oyc, oac)
}

/// Reference write_intra_prediction_modes piece 2 (UV mode + cfl + gated UV angle, over
/// pristine C od_ec). Returns (bytes, uv_mode_cdf[15], cfl_sign_cdf[9],
/// cfl_alpha_cdf[102], uv_angle_cdf[8]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_intra_uv_and_angle(
    monochrome: bool,
    is_chroma_ref: bool,
    uv_mode: i32,
    cfl_allowed: bool,
    bsize: i32,
    cfl_idx: i32,
    cfl_joint_sign: i32,
    angle_delta_uv: i32,
    uv_mode_cdf: &[u16; 15],
    cfl_sign_cdf: &[u16; 9],
    cfl_alpha_cdf: &[u16; 102],
    uv_angle_cdf: &[u16; 8],
) -> (Vec<u8>, [u16; 15], [u16; 9], [u16; 102], [u16; 8]) {
    let (mut uc, mut sc, mut ac, mut uac) =
        (*uv_mode_cdf, *cfl_sign_cdf, *cfl_alpha_cdf, *uv_angle_cdf);
    let mut out = vec![0u8; 32];
    let (mut ouc, mut osc, mut oac, mut ouac) = ([0u16; 15], [0u16; 9], [0u16; 102], [0u16; 8]);
    let n = unsafe {
        shim_write_intra_uv_and_angle(
            monochrome as i32,
            is_chroma_ref as i32,
            uv_mode,
            cfl_allowed as i32,
            bsize,
            cfl_idx,
            cfl_joint_sign,
            angle_delta_uv,
            uc.as_mut_ptr(),
            sc.as_mut_ptr(),
            ac.as_mut_ptr(),
            uac.as_mut_ptr(),
            out.as_mut_ptr(),
            ouc.as_mut_ptr(),
            osc.as_mut_ptr(),
            oac.as_mut_ptr(),
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
            inp.mode,
            inp.bsize,
            yc.as_mut_ptr(),
            inp.angle_delta_y,
            yac.as_mut_ptr(),
            inp.monochrome as i32,
            inp.is_chroma_ref as i32,
            inp.uv_mode,
            inp.cfl_allowed as i32,
            inp.cfl_idx,
            inp.cfl_joint_sign,
            inp.angle_delta_uv,
            uc.as_mut_ptr(),
            sc.as_mut_ptr(),
            ac.as_mut_ptr(),
            uac.as_mut_ptr(),
            inp.allow_palette as i32,
            inp.bit_depth,
            inp.palette_size.as_ptr(),
            inp.palette_colors.as_ptr(),
            inp.mb_to_top_edge,
            inp.ha as i32,
            inp.a_colors.as_ptr(),
            inp.a_size[0],
            inp.a_size[1],
            inp.hl as i32,
            inp.l_colors.as_ptr(),
            inp.l_size[0],
            inp.l_size[1],
            pym.as_mut_ptr(),
            pys.as_mut_ptr(),
            pum.as_mut_ptr(),
            pus.as_mut_ptr(),
            inp.filter_allowed as i32,
            inp.use_filter_intra,
            inp.filter_intra_mode,
            fiu.as_mut_ptr(),
            fim.as_mut_ptr(),
            out.as_mut_ptr(),
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
    allow_intrabc: bool,
    intrabc_cdf: &[u16; 3],
    joints: &[u16; 5],
    comp0: &[u16; 69],
    comp1: &[u16; 69],
    use_intrabc: bool,
    diff_row: i32,
    diff_col: i32,
    intra: &IntraPredModesRef,
) -> (
    Vec<u8>,
    [u16; 3],
    [u16; 5],
    [u16; 69],
    [u16; 69],
    [u16; 187],
) {
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
            allow_intrabc as i32,
            ib.as_mut_ptr(),
            jo.as_mut_ptr(),
            c0.as_mut_ptr(),
            c1.as_mut_ptr(),
            use_intrabc as i32,
            diff_row,
            diff_col,
            intra.mode,
            intra.bsize,
            yc.as_mut_ptr(),
            intra.angle_delta_y,
            yac.as_mut_ptr(),
            intra.monochrome as i32,
            intra.is_chroma_ref as i32,
            intra.uv_mode,
            intra.cfl_allowed as i32,
            intra.cfl_idx,
            intra.cfl_joint_sign,
            intra.angle_delta_uv,
            uc.as_mut_ptr(),
            sc.as_mut_ptr(),
            ac.as_mut_ptr(),
            uac.as_mut_ptr(),
            intra.allow_palette as i32,
            intra.bit_depth,
            intra.palette_size.as_ptr(),
            intra.palette_colors.as_ptr(),
            intra.mb_to_top_edge,
            intra.ha as i32,
            intra.a_colors.as_ptr(),
            intra.a_size[0],
            intra.a_size[1],
            intra.hl as i32,
            intra.l_colors.as_ptr(),
            intra.l_size[0],
            intra.l_size[1],
            pym.as_mut_ptr(),
            pys.as_mut_ptr(),
            pum.as_mut_ptr(),
            pus.as_mut_ptr(),
            intra.filter_allowed as i32,
            intra.use_filter_intra,
            intra.filter_intra_mode,
            fiu.as_mut_ptr(),
            fim.as_mut_ptr(),
            out.as_mut_ptr(),
            oib.as_mut_ptr(),
            ojo.as_mut_ptr(),
            oc0.as_mut_ptr(),
            oc1.as_mut_ptr(),
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
    enable: bool,
    bits_minus_1: i32,
    cur: i32,
    fwd: i32,
    bck: i32,
    ha: bool,
    a_has2: bool,
    a_cidx: i32,
    a_rf0: i32,
    hl: bool,
    l_has2: bool,
    l_cidx: i32,
    l_rf0: i32,
) -> i32 {
    unsafe {
        shim_get_comp_index_context(
            enable as i32,
            bits_minus_1,
            cur,
            fwd,
            bck,
            ha as i32,
            a_has2 as i32,
            a_cidx,
            a_rf0,
            hl as i32,
            l_has2 as i32,
            l_cidx,
            l_rf0,
        )
    }
}

/// Reference compound-type coding (write_mbmi_b portion, over pristine C od_ec). CDFs
/// pre-selected. Returns (bytes, cgi_cdf[3], cidx_cdf[3], ctype_cdf[3], wedge_idx[17]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_compound_type_info(
    masked_used: bool,
    comp_group_idx: i32,
    cgi_cdf: &[u16; 3],
    dist_wtd: bool,
    compound_idx: i32,
    cidx_cdf: &[u16; 3],
    wedge_used: bool,
    comp_type: i32,
    ctype_cdf: &[u16; 3],
    wedge_index: i32,
    wedge_idx_cdf: &[u16; 17],
    wedge_sign: i32,
    mask_type: i32,
) -> (Vec<u8>, [u16; 3], [u16; 3], [u16; 3], [u16; 17]) {
    let (mut cgi, mut cidx, mut ct, mut wix) = (*cgi_cdf, *cidx_cdf, *ctype_cdf, *wedge_idx_cdf);
    let mut out = vec![0u8; 32];
    let (mut ocgi, mut ocidx, mut oct, mut owix) = ([0u16; 3], [0u16; 3], [0u16; 3], [0u16; 17]);
    let n = unsafe {
        shim_write_compound_type_info(
            masked_used as i32,
            comp_group_idx,
            cgi.as_mut_ptr(),
            dist_wtd as i32,
            compound_idx,
            cidx.as_mut_ptr(),
            wedge_used as i32,
            comp_type,
            ct.as_mut_ptr(),
            wedge_index,
            wix.as_mut_ptr(),
            wedge_sign,
            mask_type,
            out.as_mut_ptr(),
            ocgi.as_mut_ptr(),
            ocidx.as_mut_ptr(),
            oct.as_mut_ptr(),
            owix.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ocgi, ocidx, oct, owix)
}

/// Reference `av1_get_palette_cache` (facade). Neighbour `palette_colors` are the full
/// 3*PALETTE_MAX_SIZE layout. Returns (cache[0..n], n).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_palette_cache(
    plane: i32,
    mb_to_top_edge: i32,
    ha: bool,
    a_colors: &[u16; 24],
    a_size0: i32,
    a_size1: i32,
    hl: bool,
    l_colors: &[u16; 24],
    l_size0: i32,
    l_size1: i32,
) -> (Vec<u16>, i32) {
    let mut cache = vec![0u16; 16];
    let n = unsafe {
        shim_get_palette_cache(
            plane,
            mb_to_top_edge,
            ha as i32,
            a_colors.as_ptr(),
            a_size0,
            a_size1,
            hl as i32,
            l_colors.as_ptr(),
            l_size0,
            l_size1,
            cache.as_mut_ptr(),
        )
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
        shim_index_color_cache(
            cache.as_ptr(),
            cache.len() as i32,
            colors.as_ptr(),
            colors.len() as i32,
            found.as_mut_ptr(),
            out_colors.as_mut_ptr(),
        )
    };
    out_colors.truncate(n as usize);
    (found, out_colors, n)
}

/// Reference full `write_palette_mode_info` (flags + sizes + colours end-to-end, over
/// pristine C od_ec + the real cache/index/delta-bits fns). CDFs are pre-selected by
/// the caller. Returns (bytes, y_mode[3], y_size[8], uv_mode[3], uv_size[8]).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn ref_write_palette_mode_info(
    mode_dc: bool,
    uv_dc: bool,
    bit_depth: i32,
    palette_size: &[u8; 2],
    palette_colors: &[u16; 24],
    mb_to_top_edge: i32,
    ha: bool,
    a_colors: &[u16; 24],
    a_size: &[i32; 2],
    hl: bool,
    l_colors: &[u16; 24],
    l_size: &[i32; 2],
    y_mode_cdf: &[u16; 3],
    y_size_cdf: &[u16; 8],
    uv_mode_cdf: &[u16; 3],
    uv_size_cdf: &[u16; 8],
) -> (Vec<u8>, [u16; 3], [u16; 8], [u16; 3], [u16; 8]) {
    let (mut ym, mut ys, mut um, mut us) = (*y_mode_cdf, *y_size_cdf, *uv_mode_cdf, *uv_size_cdf);
    let mut out = vec![0u8; 128];
    let (mut oym, mut oys, mut oum, mut ous) = ([0u16; 3], [0u16; 8], [0u16; 3], [0u16; 8]);
    let n = unsafe {
        shim_write_palette_mode_info(
            mode_dc as i32,
            uv_dc as i32,
            bit_depth,
            0,
            0,
            0,
            palette_size.as_ptr(),
            palette_colors.as_ptr(),
            mb_to_top_edge,
            ha as i32,
            a_colors.as_ptr(),
            a_size[0],
            a_size[1],
            hl as i32,
            l_colors.as_ptr(),
            l_size[0],
            l_size[1],
            ym.as_mut_ptr(),
            ys.as_mut_ptr(),
            um.as_mut_ptr(),
            us.as_mut_ptr(),
            out.as_mut_ptr(),
            oym.as_mut_ptr(),
            oys.as_mut_ptr(),
            oum.as_mut_ptr(),
            ous.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oym, oys, oum, ous)
}

/// Reference `av1_get_intra_inter_context` (facade over the real exported fn).
pub fn ref_get_intra_inter_context(
    has_above: bool,
    above_inter: bool,
    has_left: bool,
    left_inter: bool,
) -> i32 {
    unsafe {
        shim_get_intra_inter_context(
            has_above as i32,
            above_inter as i32,
            has_left as i32,
            left_inter as i32,
        )
    }
}

/// Reference `av1_get_pred_context_switchable_interp` (facade over the real
/// exported fn). Neighbours are `Some((ref0, ref1, y_filter, x_filter))` when
/// available, `None` otherwise.
#[allow(clippy::too_many_arguments)]
pub fn ref_get_pred_context_switchable_interp(
    dir: usize,
    cur_ref0: i32,
    cur_is_compound: bool,
    above: Option<(i32, i32, usize, usize)>,
    left: Option<(i32, i32, usize, usize)>,
) -> i32 {
    let (ha, a_r0, a_r1, a_yf, a_xf) =
        above.map_or((0, 0, 0, 0, 0), |(r0, r1, yf, xf)| (1, r0, r1, yf as i32, xf as i32));
    let (hl, l_r0, l_r1, l_yf, l_xf) =
        left.map_or((0, 0, 0, 0, 0), |(r0, r1, yf, xf)| (1, r0, r1, yf as i32, xf as i32));
    unsafe {
        shim_get_pred_context_switchable_interp(
            dir as i32,
            cur_ref0,
            cur_is_compound as i32,
            ha,
            a_r0,
            a_r1,
            a_yf,
            a_xf,
            hl,
            l_r0,
            l_r1,
            l_yf,
            l_xf,
        )
    }
}

/// Reference `write_motion_mode`.
pub fn ref_write_motion_mode(
    obmc_cdf: &[u16; 3],
    mm_cdf: &[u16; 4],
    last_allowed: i32,
    mm: i32,
) -> (Vec<u8>, [u16; 3], [u16; 4]) {
    let mut o = *obmc_cdf;
    let mut m = *mm_cdf;
    let mut out = vec![0u8; 16];
    let mut oo = [0u16; 3];
    let mut om = [0u16; 4];
    let n = unsafe {
        shim_write_motion_mode(
            o.as_mut_ptr(),
            m.as_mut_ptr(),
            last_allowed,
            mm,
            out.as_mut_ptr(),
            oo.as_mut_ptr(),
            om.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oo, om)
}

/// Reference `write_inter_compound_mode`.
pub fn ref_write_inter_compound_mode(cdf: &[u16; 9], mode: i32) -> (Vec<u8>, [u16; 9]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut oc = [0u16; 9];
    let n = unsafe {
        shim_write_inter_compound_mode(c.as_mut_ptr(), mode, out.as_mut_ptr(), oc.as_mut_ptr())
    };
    out.truncate(n as usize);
    (out, oc)
}

/// Reference `write_is_inter`.
pub fn ref_write_is_inter(
    cdf: &[u16; 3],
    seg_ref: bool,
    seg_gmv: bool,
    is_inter: i32,
) -> (Vec<u8>, [u16; 3]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut oc = [0u16; 3];
    let n = unsafe {
        shim_write_is_inter(
            c.as_mut_ptr(),
            seg_ref as i32,
            seg_gmv as i32,
            is_inter,
            out.as_mut_ptr(),
            oc.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, oc)
}

/// Reference `write_filter_intra_mode_info` (over the pristine C od_ec + update_cdf).
pub fn ref_write_filter_intra(
    use_cdf: &[u16; 3],
    mode_cdf: &[u16; 6],
    allowed: bool,
    use_fi: i32,
    mode: i32,
) -> (Vec<u8>, [u16; 3], [u16; 6]) {
    let mut u = *use_cdf;
    let mut m = *mode_cdf;
    let mut out = vec![0u8; 16];
    let mut ou = [0u16; 3];
    let mut om = [0u16; 6];
    let n = unsafe {
        shim_write_filter_intra(
            u.as_mut_ptr(),
            m.as_mut_ptr(),
            allowed as i32,
            use_fi,
            mode,
            out.as_mut_ptr(),
            ou.as_mut_ptr(),
            om.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, ou, om)
}

/// Reference `bsize_to_max_depth` / `bsize_to_tx_size_cat`.
pub fn ref_bsize_to_max_depth(bsize: i32) -> usize {
    unsafe { shim_bsize_to_max_depth(bsize) as usize }
}
pub fn ref_bsize_to_tx_size_cat(bsize: i32) -> i32 {
    unsafe { shim_bsize_to_tx_size_cat(bsize) }
}

/// Reference `write_selected_tx_size` (over the pristine C od_ec + update_cdf).
pub fn ref_write_selected_tx_size(
    cdf: &[u16; 4],
    bsize: i32,
    depth: i32,
    max_depths: i32,
) -> (Vec<u8>, [u16; 4]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 4];
    let n = unsafe {
        shim_write_selected_tx_size(
            c.as_mut_ptr(),
            bsize,
            depth,
            max_depths,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_angle_delta` (over the pristine C od_ec + update_cdf).
pub fn ref_write_angle_delta(cdf: &[u16; 8], angle_delta: i32) -> (Vec<u8>, [u16; 8]) {
    let mut c = *cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 8];
    let n = unsafe {
        shim_write_angle_delta(
            c.as_mut_ptr(),
            angle_delta,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
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
pub fn ref_write_drl_idx(
    drl_cdf: &[u16; 9],
    mode: i32,
    ref_mv_idx: i32,
    ref_mv_count: i32,
    weight: &[u16; 4],
) -> (Vec<u8>, [u16; 9]) {
    let mut cdf = *drl_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 9];
    let n = unsafe {
        shim_write_drl_idx(
            cdf.as_mut_ptr(),
            mode,
            ref_mv_idx,
            ref_mv_count,
            weight.as_ptr(),
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_inter_mode` (3-symbol cascade over the pristine C od_ec + update_cdf).
/// Returns coded bytes + the adapted newmv[18]/zeromv[6]/refmv[18] flat CDF arrays.
#[allow(clippy::type_complexity)]
pub fn ref_write_inter_mode(
    newmv: &[u16; 18],
    zeromv: &[u16; 6],
    refmv: &[u16; 18],
    mode: i32,
    mode_ctx: i32,
) -> (Vec<u8>, [u16; 18], [u16; 6], [u16; 18]) {
    let mut nm = *newmv;
    let mut zm = *zeromv;
    let mut rm = *refmv;
    let mut out = vec![0u8; 16];
    let mut onm = [0u16; 18];
    let mut ozm = [0u16; 6];
    let mut orm = [0u16; 18];
    let n = unsafe {
        shim_write_inter_mode(
            nm.as_mut_ptr(),
            zm.as_mut_ptr(),
            rm.as_mut_ptr(),
            mode,
            mode_ctx,
            out.as_mut_ptr(),
            onm.as_mut_ptr(),
            ozm.as_mut_ptr(),
            orm.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, onm, ozm, orm)
}

/// Reference `size_group_lookup[bsize]`.
pub fn ref_size_group_lookup(bsize: i32) -> usize {
    unsafe { shim_size_group_lookup(bsize) as usize }
}

/// Reference `write_intra_uv_mode` (transcribed symbol over the pristine C od_ec + update_cdf).
pub fn ref_write_intra_uv_mode(
    uv_mode_cdf: &[u16; 15],
    uv_mode: i32,
    cfl_allowed: bool,
) -> (Vec<u8>, [u16; 15]) {
    let mut cdf = *uv_mode_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 15];
    let n = unsafe {
        shim_write_intra_uv_mode(
            cdf.as_mut_ptr(),
            uv_mode,
            cfl_allowed as i32,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `get_y_mode_cdf` context (real intra_mode_context table + block-mode rule).
pub fn ref_get_y_mode_ctx(
    above_present: bool,
    above_mode: i32,
    left_present: bool,
    left_mode: i32,
) -> (usize, usize) {
    let v = unsafe {
        shim_get_y_mode_ctx(
            above_present as i32,
            above_mode,
            left_present as i32,
            left_mode,
        )
    };
    ((v >> 8) as usize, (v & 0xff) as usize)
}

/// Reference `write_intra_y_mode_kf` (transcribed symbol over the pristine C od_ec + update_cdf).
pub fn ref_write_intra_y_mode_kf(kf_y_cdf: &[u16; 14], mode: i32) -> (Vec<u8>, [u16; 14]) {
    let mut cdf = *kf_y_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 14];
    let n = unsafe {
        shim_write_intra_y_mode_kf(
            cdf.as_mut_ptr(),
            mode,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_cfl_alphas` (transcribed over the pristine C od_ec + update_cdf).
/// Returns coded bytes, the adapted sign CDF (9), and the adapted alpha CDFs (6x17 flat).
pub fn ref_write_cfl_alphas(
    cfl_sign_cdf: &[u16; 9],
    cfl_alpha_cdf: &[u16; 102],
    idx: i32,
    joint_sign: i32,
) -> (Vec<u8>, [u16; 9], [u16; 102]) {
    let mut sc = *cfl_sign_cdf;
    let mut ac = *cfl_alpha_cdf;
    let mut out = vec![0u8; 16];
    let mut osc = [0u16; 9];
    let mut oac = [0u16; 102];
    let n = unsafe {
        shim_write_cfl_alphas(
            sc.as_mut_ptr(),
            ac.as_mut_ptr(),
            idx,
            joint_sign,
            out.as_mut_ptr(),
            osc.as_mut_ptr(),
            oac.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, osc, oac)
}

/// Reference `write_delta_lflevel` (transcribed over the pristine C od_ec + update_cdf).
pub fn ref_write_delta_lflevel(delta_lf_cdf: &[u16; 5], delta_lflevel: i32) -> (Vec<u8>, [u16; 5]) {
    let mut cdf = *delta_lf_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 5];
    let n = unsafe {
        shim_write_delta_lflevel(
            cdf.as_mut_ptr(),
            delta_lflevel,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_delta_qindex` (transcribed over the pristine C od_ec + update_cdf).
pub fn ref_write_delta_qindex(delta_q_cdf: &[u16; 5], delta_qindex: i32) -> (Vec<u8>, [u16; 5]) {
    let mut cdf = *delta_q_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 5];
    let n = unsafe {
        shim_write_delta_qindex(
            cdf.as_mut_ptr(),
            delta_qindex,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `av1_get_skip_txfm_context` (facade over the real static inline).
pub fn ref_skip_txfm_context(
    above_present: bool,
    above_skip: i32,
    left_present: bool,
    left_skip: i32,
) -> i32 {
    unsafe {
        shim_skip_txfm_context(
            above_present as i32,
            above_skip,
            left_present as i32,
            left_skip,
        )
    }
}

/// Reference `write_skip` (transcribed symbol over the pristine C od_ec + update_cdf).
pub fn ref_write_skip(
    skip_cdf: &[u16; 3],
    seg_skip_active: bool,
    skip_txfm: i32,
) -> (Vec<u8>, [u16; 3]) {
    let mut cdf = *skip_cdf;
    let mut out = vec![0u8; 16];
    let mut out_cdf = [0u16; 3];
    let n = unsafe {
        shim_write_skip(
            cdf.as_mut_ptr(),
            seg_skip_active as i32,
            skip_txfm,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `write_partition` (transcribed body over the pristine C od_ec + update_cdf).
/// Returns the coded bytes and the adapted partition CDF (`cdf_len+1` meaningful entries).
pub fn ref_write_partition(
    partition_cdf: &[u16; 11],
    cdf_len: i32,
    p: i32,
    has_rows: bool,
    has_cols: bool,
    bsize: i32,
) -> (Vec<u8>, [u16; 11]) {
    let mut cdf = *partition_cdf;
    let mut out = vec![0u8; 64];
    let mut out_cdf = [0u16; 11];
    let n = unsafe {
        shim_write_partition(
            cdf.as_mut_ptr(),
            cdf_len,
            p,
            has_rows as i32,
            has_cols as i32,
            bsize,
            out.as_mut_ptr(),
            out_cdf.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    (out, out_cdf)
}

/// Reference `partition_plane_context` (facade over the real static inline).
pub fn ref_partition_plane_context(
    above: &[i8],
    left: &[i8],
    mi_row: i32,
    mi_col: i32,
    bsize: i32,
) -> i32 {
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
pub fn ref_write_refresh_frame_context(
    reduced: bool,
    disable_cdf: bool,
    rfc_disabled: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe {
        shim_write_refresh_frame_context(
            reduced as i32,
            disable_cdf as i32,
            rfc_disabled as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference frame-header trailing flags.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_frame_header_trailing_flags(
    intra_only: bool,
    ref_mode_select: bool,
    skip_allowed: bool,
    skip_flag: bool,
    might_warp: bool,
    allow_warp: bool,
    reduced_tx_set: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe {
        shim_write_frame_header_trailing_flags(
            intra_only as i32,
            ref_mode_select as i32,
            skip_allowed as i32,
            skip_flag as i32,
            might_warp as i32,
            allow_warp as i32,
            reduced_tx_set as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference INTER/S-frame ref signaling (transcribed over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_inter_ref_signaling(
    enable_order_hint: bool,
    short_sig: bool,
    ref_map_idx: &[i32; 7],
    set_rfc: bool,
    rtc_reference: &[i32; 7],
    rtc_ref_idx: &[i32; 7],
    num_spatial_layers: i32,
    frame_id_present: bool,
    frame_id_len: u32,
    current_frame_id: i32,
    ref_frame_id: &[i32; 8],
    diff_len: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_write_inter_ref_signaling(
            enable_order_hint as i32,
            short_sig as i32,
            ref_map_idx.as_ptr(),
            set_rfc as i32,
            rtc_reference.as_ptr(),
            rtc_ref_idx.as_ptr(),
            num_spatial_layers,
            frame_id_present as i32,
            frame_id_len as i32,
            current_frame_id,
            ref_frame_id.as_ptr(),
            diff_len as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_frame_size_with_refs` (transcribed over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_frame_size_with_refs(
    up_w: i32,
    up_h: i32,
    rw: i32,
    rh: i32,
    valid: &[i32; 7],
    ycw: &[i32; 7],
    ych: &[i32; 7],
    rrw: &[i32; 7],
    rrh: &[i32; 7],
    enable_superres: bool,
    denom: i32,
    fs_num_bits_w: u32,
    fs_num_bits_h: u32,
    fs_up_w: i32,
    fs_up_h: i32,
    fs_scaling_active: bool,
    fs_rw: i32,
    fs_rh: i32,
) -> Vec<u8> {
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_write_frame_size_with_refs(
            up_w,
            up_h,
            rw,
            rh,
            valid.as_ptr(),
            ycw.as_ptr(),
            ych.as_ptr(),
            rrw.as_ptr(),
            rrh.as_ptr(),
            enable_superres as i32,
            denom,
            fs_num_bits_w as i32,
            fs_num_bits_h as i32,
            fs_up_w,
            fs_up_h,
            fs_scaling_active as i32,
            fs_rw,
            fs_rh,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_uncompressed_header_obu` prefix (transcribed over the real aom_wb).
pub fn ref_write_frame_header_prefix(
    t: &[i64; 34],
    op_dmpp: &[i64; 32],
    op_idc: &[i64; 32],
    brt: &[i64; 32],
    ref_oh: &[i64; 8],
) -> Vec<u8> {
    let mut out = vec![0u8; 256];
    let n = unsafe {
        shim_write_frame_header_prefix(
            t.as_ptr(),
            op_dmpp.as_ptr(),
            op_idc.as_ptr(),
            brt.as_ptr(),
            ref_oh.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `av1_write_sequence_header_obu` — the REAL exported function, fed a
/// `SequenceHeader` populated from the packed params (direct oracle, not a
/// transcription).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_sequence_header_obu(
    top: &[i64; 16],
    sh: &[i64; 24],
    cc: &[i64; 11],
    idc: &[i64; 32],
    level: &[i64; 32],
    tier: &[i64; 32],
    dmpp: &[i64; 32],
    dispp: &[i64; 32],
    decdelay: &[i64; 32],
    encdelay: &[i64; 32],
    lowdelay: &[i64; 32],
    initdelay: &[i64; 32],
) -> Vec<u8> {
    let mut out = vec![0u8; 4096];
    let n = unsafe {
        shim_write_sequence_header_obu(
            top.as_ptr(),
            sh.as_ptr(),
            cc.as_ptr(),
            idc.as_ptr(),
            level.as_ptr(),
            tier.as_ptr(),
            dmpp.as_ptr(),
            dispp.as_ptr(),
            decdelay.as_ptr(),
            encdelay.as_ptr(),
            lowdelay.as_ptr(),
            initdelay.as_ptr(),
            out.as_mut_ptr(),
        )
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
pub fn ref_write_timing_info(
    disp_tick: u32,
    time_scale: u32,
    equal_pic: bool,
    ticks_per_pic: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe {
        shim_write_timing_info(
            disp_tick,
            time_scale,
            equal_pic as i32,
            ticks_per_pic,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_decoder_model_info`.
pub fn ref_write_decoder_model_info(
    ed_delay_len: i32,
    dec_tick: u32,
    rem_time_len: i32,
    pres_time_len: i32,
) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe {
        shim_write_decoder_model_info(
            ed_delay_len,
            dec_tick,
            rem_time_len,
            pres_time_len,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_dec_model_op_parameters`.
pub fn ref_write_dec_model_op(
    dec_delay: u32,
    enc_delay: u32,
    low_delay: bool,
    delay_len: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe {
        shim_write_dec_model_op(
            dec_delay,
            enc_delay,
            low_delay as i32,
            delay_len as i32,
            out.as_mut_ptr(),
        )
    };
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
    let n =
        unsafe { shim_write_ext_tile_info(pre_bits, rows as i32, cols as i32, out.as_mut_ptr()) };
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
pub fn ref_write_global_motion(
    wmtype: &[i32; 7],
    wmmat: &[i32; 42],
    refmat: &[i32; 42],
    allow_hp: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; 512];
    let n = unsafe {
        shim_write_global_motion(
            wmtype.as_ptr(),
            wmmat.as_ptr(),
            refmat.as_ptr(),
            allow_hp as i32,
            out.as_mut_ptr(),
        )
    };
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
pub fn ref_write_film_grain_params(
    s: &[i32; 24],
    spy: &[i32; 28],
    spcb: &[i32; 20],
    spcr: &[i32; 20],
    ary: &[i32; 24],
    arcb: &[i32; 25],
    arcr: &[i32; 25],
) -> Vec<u8> {
    let mut out = vec![0u8; 256];
    let n = unsafe {
        shim_write_film_grain_params(
            s.as_ptr(),
            spy.as_ptr(),
            spcb.as_ptr(),
            spcr.as_ptr(),
            ary.as_ptr(),
            arcb.as_ptr(),
            arcr.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_delta_q_params` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_delta_q_params(
    base_qindex: i32,
    delta_q_present: bool,
    delta_q_res: i32,
    allow_intrabc: bool,
    delta_lf_present: bool,
    delta_lf_res: i32,
    delta_lf_multi: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; 8];
    let n = unsafe {
        shim_write_delta_q_params(
            base_qindex,
            delta_q_present as i32,
            delta_q_res,
            allow_intrabc as i32,
            delta_lf_present as i32,
            delta_lf_res,
            delta_lf_multi as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_tx_mode`.
pub fn ref_write_tx_mode(coded_lossless: bool, tx_mode_select: bool) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    let n = unsafe {
        shim_write_tx_mode(
            coded_lossless as i32,
            tx_mode_select as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `encode_restoration_mode` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_restoration_mode(
    enable_restoration: bool,
    allow_intrabc: bool,
    frame_restoration_type: &[i32; 3],
    sb_size_128: bool,
    restoration_unit_size: &[i32; 3],
    ssx: i32,
    ssy: i32,
    num_planes: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe {
        shim_encode_restoration_mode(
            enable_restoration as i32,
            allow_intrabc as i32,
            frame_restoration_type.as_ptr(),
            sb_size_128 as i32,
            restoration_unit_size.as_ptr(),
            ssx,
            ssy,
            num_planes as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_tile_group_header` (transcribed over the real aom_wb).
pub fn ref_write_tile_group_header(
    start_tile: i32,
    end_tile: i32,
    tiles_log2: i32,
    present_flag: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; 8];
    let n = unsafe {
        shim_write_tile_group_header(
            start_tile,
            end_tile,
            tiles_log2,
            present_flag as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `write_tile_info` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_write_tile_info(
    mi_cols: i32,
    mi_rows: i32,
    mib_size_log2: u32,
    uniform_spacing: bool,
    log2_cols: i32,
    min_log2_cols: i32,
    max_log2_cols: i32,
    log2_rows: i32,
    min_log2_rows: i32,
    max_log2_rows: i32,
    cols: usize,
    rows: usize,
    col_start_sb: &[i32; 65],
    row_start_sb: &[i32; 65],
    max_width_sb: i32,
    max_height_sb: i32,
) -> Vec<u8> {
    let mut out = vec![0u8; 128];
    let n = unsafe {
        shim_write_tile_info(
            mi_cols,
            mi_rows,
            mib_size_log2 as i32,
            uniform_spacing as i32,
            log2_cols,
            min_log2_cols,
            max_log2_cols,
            log2_rows,
            min_log2_rows,
            max_log2_rows,
            cols as i32,
            rows as i32,
            col_start_sb.as_ptr(),
            row_start_sb.as_ptr(),
            max_width_sb,
            max_height_sb,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `encode_segmentation` (transcribed control flow over the real aom_wb + seg tables).
pub fn ref_encode_segmentation(
    enabled: bool,
    has_primary_ref: bool,
    update_map: bool,
    temporal_update: bool,
    update_data: bool,
    feature_mask: &[u32; 8],
    feature_data: &[[i32; 8]; 8],
) -> Vec<u8> {
    let flat: Vec<i32> = feature_data.iter().flatten().copied().collect();
    let mut out = vec![0u8; 64];
    let n = unsafe {
        shim_encode_segmentation(
            enabled as i32,
            has_primary_ref as i32,
            update_map as i32,
            temporal_update as i32,
            update_data as i32,
            feature_mask.as_ptr(),
            flat.as_ptr(),
            out.as_mut_ptr(),
        )
    };
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

/// Reference `analyze_hor_freq` (superres_scale.c): the 16x4 H_DCT horizontal
/// frequency-energy analysis over `src` (tight/strided luma, `stride` samples
/// per row, values 0..(2^bd-1)). Returns the 16-entry cumulative energy vector
/// (index 0 unused). Calls the real exported `av1_fwd_txfm2d_16x4_c`.
pub fn ref_superres_analyze_hor_freq(
    src: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bd: u8,
) -> [f64; 16] {
    let mut energy = [0f64; 16];
    unsafe {
        shim_superres_analyze_hor_freq(
            src.as_ptr(),
            width as i32,
            height as i32,
            stride as i32,
            i32::from(bd),
            energy.as_mut_ptr(),
        );
    }
    energy
}

/// Reference `get_superres_denom_from_qindex_energy` (superres_scale.c). Calls
/// the real exported `av1_convert_qindex_to_q`.
pub fn ref_superres_denom_from_qindex_energy(
    qindex: i32,
    energy: &[f64; 16],
    threshq: f64,
    threshp: f64,
) -> u8 {
    unsafe { shim_superres_denom_from_qindex_energy(qindex, energy.as_ptr(), threshq, threshp) }
}

/// Reference KEY-frame QTHRESH-arm superres denom selection
/// (`calculate_next_superres_scale` + `get_superres_denom_for_qindex`,
/// single-KEY AOM_Q envelope). Returns the denominator (8..16; 8 = no superres).
#[allow(clippy::too_many_arguments)]
pub fn ref_superres_denom_qthresh_key(
    src: &[u16],
    w: usize,
    h: usize,
    stride: usize,
    bd: u8,
    q: i32,
    kf_qthresh_qindex: i32,
    allow_scc: bool,
    frames_to_key_le_1: bool,
) -> u8 {
    unsafe {
        shim_superres_denom_qthresh_key(
            src.as_ptr(),
            w as i32,
            h as i32,
            stride as i32,
            i32::from(bd),
            q,
            kf_qthresh_qindex,
            allow_scc as i32,
            frames_to_key_le_1 as i32,
        )
    }
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
pub fn ref_write_frame_size(
    frame_size_override: bool,
    num_bits_width: u32,
    num_bits_height: u32,
    up_w: i32,
    up_h: i32,
    enable_superres: bool,
    denom: i32,
    scaling_active: bool,
    rw: i32,
    rh: i32,
) -> Vec<u8> {
    let mut out = vec![0u8; 16];
    let n = unsafe {
        shim_write_frame_size(
            frame_size_override as i32,
            num_bits_width as i32,
            num_bits_height as i32,
            up_w,
            up_h,
            enable_superres as i32,
            denom,
            scaling_active as i32,
            rw,
            rh,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `encode_cdef` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_cdef(
    enable_cdef: bool,
    allow_intrabc: bool,
    damping: i32,
    cdef_bits: i32,
    nb: usize,
    y: &[i32; 8],
    uv: &[i32; 8],
    num_planes: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe {
        shim_encode_cdef(
            enable_cdef as i32,
            allow_intrabc as i32,
            damping,
            cdef_bits,
            nb as i32,
            y.as_ptr(),
            uv.as_ptr(),
            num_planes as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `encode_loopfilter` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_loopfilter(
    allow_intrabc: bool,
    filter_level: [i32; 2],
    flu: i32,
    flv: i32,
    sharpness: i32,
    mode_ref_enabled: bool,
    mode_ref_update: bool,
    ref_deltas: &[i8; 8],
    mode_deltas: &[i8; 2],
    last_ref: &[i8; 8],
    last_mode: &[i8; 2],
    num_planes: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe {
        shim_encode_loopfilter(
            allow_intrabc as i32,
            filter_level[0],
            filter_level[1],
            flu,
            flv,
            sharpness,
            mode_ref_enabled as i32,
            mode_ref_update as i32,
            ref_deltas.as_ptr(),
            mode_deltas.as_ptr(),
            last_ref.as_ptr(),
            last_mode.as_ptr(),
            num_planes as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `encode_quantization` (transcribed control flow over the real aom_wb).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_quantization(
    base_qindex: i32,
    y_dc: i32,
    u_dc: i32,
    u_ac: i32,
    v_dc: i32,
    v_ac: i32,
    using_qm: bool,
    qm_y: i32,
    qm_u: i32,
    qm_v: i32,
    num_planes: usize,
    separate_uv: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; 32];
    let n = unsafe {
        shim_encode_quantization(
            base_qindex,
            y_dc,
            u_dc,
            u_ac,
            v_dc,
            v_ac,
            using_qm as i32,
            qm_y,
            qm_u,
            qm_v,
            num_planes as i32,
            separate_uv as i32,
            out.as_mut_ptr(),
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `av1_write_obu_header` byte output (transcribed shim). Returns the header bytes.
pub fn ref_write_obu_header(
    obu_type: u32,
    has_nonzero_op: bool,
    is_layer_specific: bool,
    obu_extension: u8,
) -> Vec<u8> {
    let mut dst = [0u8; 2];
    let n = unsafe {
        shim_write_obu_header(
            obu_type as i32,
            has_nonzero_op as i32,
            is_layer_specific as i32,
            obu_extension as i32,
            dst.as_mut_ptr(),
        )
    };
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
    if rc == 0 {
        Some((value, length))
    } else {
        None
    }
}

/// Reference `aom_write_bit_buffer`: apply a sequence of literal ops (kind 0 =
/// signed literal, 1 = unsigned literal, 2 = inv-signed literal) and return the
/// produced bytes.
pub fn ref_wb_apply(data: &[u32], bits: &[i32], kind: &[i32]) -> Vec<u8> {
    let n = data.len();
    let mut out = vec![0u8; 1 << 16];
    let written = unsafe {
        shim_wb_apply(
            data.as_ptr(),
            bits.as_ptr(),
            kind.as_ptr(),
            n as i32,
            out.as_mut_ptr(),
        )
    };
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
    unsafe {
        aom_sum_squares_2d_i16_c(src.as_ptr(), src_stride as i32, width as i32, height as i32)
    }
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
pub fn ref_block_error_qm(
    coeff: &[i32],
    dqcoeff: &[i32],
    qmatrix: &[u8],
    scan: &[i16],
    bd: u8,
) -> (i64, i64) {
    let mut ssz = 0i64;
    let err = unsafe {
        shim_block_error_qm(
            coeff.as_ptr(),
            dqcoeff.as_ptr(),
            coeff.len() as isize,
            qmatrix.as_ptr(),
            scan.as_ptr(),
            &mut ssz,
            bd as i32,
        )
    };
    (err, ssz)
}

/// Reference `aom_subtract_block_c` (residual = src - pred). Writes `diff`.
#[allow(clippy::too_many_arguments)]
pub fn ref_subtract_block(
    rows: usize,
    cols: usize,
    diff: &mut [i16],
    diff_stride: usize,
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
) {
    unsafe {
        aom_subtract_block_c(
            rows as i32,
            cols as i32,
            diff.as_mut_ptr(),
            diff_stride as isize,
            src.as_ptr(),
            src_stride as isize,
            pred.as_ptr(),
            pred_stride as isize,
        )
    }
}

/// Reference `aom_highbd_subtract_block_c` (residual = src - pred, u16).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_subtract_block(
    rows: usize,
    cols: usize,
    diff: &mut [i16],
    diff_stride: usize,
    src: &[u16],
    src_stride: usize,
    pred: &[u16],
    pred_stride: usize,
) {
    unsafe {
        shim_highbd_subtract_block(
            rows as i32,
            cols as i32,
            diff.as_mut_ptr(),
            diff_stride as i32,
            src.as_ptr(),
            src_stride as i32,
            pred.as_ptr(),
            pred_stride as i32,
        )
    }
}

/// Reference `av1_block_error_c` (transform-domain distortion). Returns (error, ssz).
pub fn ref_block_error(coeff: &[i32], dqcoeff: &[i32]) -> (i64, i64) {
    let mut ssz = 0i64;
    let err = unsafe {
        av1_block_error_c(
            coeff.as_ptr(),
            dqcoeff.as_ptr(),
            coeff.len() as isize,
            &mut ssz,
        )
    };
    (err, ssz)
}

/// Reference `av1_highbd_block_error_c` (highbd transform-domain distortion).
pub fn ref_highbd_block_error(coeff: &[i32], dqcoeff: &[i32], bd: u8) -> (i64, i64) {
    let mut ssz = 0i64;
    let err = unsafe {
        av1_highbd_block_error_c(
            coeff.as_ptr(),
            dqcoeff.as_ptr(),
            coeff.len() as isize,
            &mut ssz,
            bd as i32,
        )
    };
    (err, ssz)
}

/// Reference `aom_obmc_sad<W>x<H>_c` (overlapped block motion-comp SAD).
pub fn ref_obmc_sad(idx: usize, r: &[u8], rs: usize, ws: &[i32], m: &[i32]) -> u32 {
    unsafe { shim_obmc_sad(idx as i32, r.as_ptr(), rs as i32, ws.as_ptr(), m.as_ptr()) }
}

/// Reference `aom_masked_sad<W>x<H>_c` (wedge / diff-weighted compound RD).
#[allow(clippy::too_many_arguments)]
pub fn ref_masked_sad(
    idx: usize,
    s: &[u8],
    ss: usize,
    r: &[u8],
    rs: usize,
    sp: &[u8],
    m: &[u8],
    ms: usize,
    inv: bool,
) -> u32 {
    unsafe {
        shim_masked_sad(
            idx as i32,
            s.as_ptr(),
            ss as i32,
            r.as_ptr(),
            rs as i32,
            sp.as_ptr(),
            m.as_ptr(),
            ms as i32,
            inv as i32,
        )
    }
}

/// Reference `aom_sad<W>x<H>_avg_c` (compound-prediction SAD) for size index `idx`.
///
/// `aom_sad*_avg_c` internally calls the RTCD-dispatched `aom_comp_avg_pred`
/// (a null fn-pointer until `aom_dsp_rtcd()` runs), so init RTCD first.
pub fn ref_sad_avg(idx: usize, s: &[u8], ss: usize, r: &[u8], rs: usize, sp: &[u8]) -> u32 {
    ref_init();
    unsafe {
        shim_sad_avg(
            idx as i32,
            s.as_ptr(),
            ss as i32,
            r.as_ptr(),
            rs as i32,
            sp.as_ptr(),
        )
    }
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
    let v = unsafe {
        shim_variance(
            idx as i32,
            a.as_ptr(),
            as_ as i32,
            b.as_ptr(),
            bs as i32,
            &mut sse,
        )
    };
    (v, sse)
}

/// Reference `aom_sub_pixel_variance<W>x<H>_c`; returns (variance, sse).
#[allow(clippy::too_many_arguments)]
pub fn ref_subpel_var(
    idx: usize,
    a: &[u8],
    as_: usize,
    xo: usize,
    yo: usize,
    b: &[u8],
    bs: usize,
) -> (u32, u32) {
    let mut sse = 0u32;
    let v = unsafe {
        shim_subpel_var(
            idx as i32,
            a.as_ptr(),
            as_ as i32,
            xo as i32,
            yo as i32,
            b.as_ptr(),
            bs as i32,
            &mut sse,
        )
    };
    (v, sse)
}

// hbd_sadvar_shim.c — highbd SAD / variance (CONVERT_TO_BYTEPTR internally).
extern "C" {
    fn shim_hbd_sad(i: i32, s: *const u16, ss: i32, r: *const u16, rs: i32) -> u32;
    fn shim_hbd_var(
        i: i32,
        bd: i32,
        a: *const u16,
        as_: i32,
        b: *const u16,
        bs: i32,
        sse: *mut u32,
    ) -> u32;
    fn shim_hbd_subpel_var(
        i: i32,
        bd: i32,
        a: *const u16,
        as_: i32,
        xo: i32,
        yo: i32,
        b: *const u16,
        bs: i32,
        sse: *mut u32,
    ) -> u32;
    fn shim_hbd_sad_avg(
        i: i32,
        s: *const u16,
        ss: i32,
        r: *const u16,
        rs: i32,
        p: *const u16,
    ) -> u32;
    #[allow(clippy::too_many_arguments)]
    fn shim_hbd_masked_sad(
        i: i32,
        s: *const u16,
        ss: i32,
        r: *const u16,
        rs: i32,
        p: *const u16,
        m: *const u8,
        ms: i32,
        inv: i32,
    ) -> u32;
    fn shim_hbd_obmc_sad(i: i32, r: *const u16, rs: i32, ws: *const i32, m: *const i32) -> u32;
}

/// Reference `aom_highbd_obmc_sad<W>x<H>_c`.
pub fn ref_hbd_obmc_sad(idx: usize, r: &[u16], rs: usize, ws: &[i32], m: &[i32]) -> u32 {
    unsafe { shim_hbd_obmc_sad(idx as i32, r.as_ptr(), rs as i32, ws.as_ptr(), m.as_ptr()) }
}

/// Reference `aom_highbd_masked_sad<W>x<H>_c` (highbd wedge / compound SAD).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_masked_sad(
    idx: usize,
    s: &[u16],
    ss: usize,
    r: &[u16],
    rs: usize,
    p: &[u16],
    m: &[u8],
    ms: usize,
    inv: bool,
) -> u32 {
    unsafe {
        shim_hbd_masked_sad(
            idx as i32,
            s.as_ptr(),
            ss as i32,
            r.as_ptr(),
            rs as i32,
            p.as_ptr(),
            m.as_ptr(),
            ms as i32,
            inv as i32,
        )
    }
}

/// Reference `aom_highbd_sad<W>x<H>_avg_c` (highbd compound-prediction SAD).
/// Calls `ref_init()` first (invokes RTCD-dispatched `aom_highbd_comp_avg_pred`).
pub fn ref_hbd_sad_avg(idx: usize, s: &[u16], ss: usize, r: &[u16], rs: usize, p: &[u16]) -> u32 {
    ref_init();
    unsafe {
        shim_hbd_sad_avg(
            idx as i32,
            s.as_ptr(),
            ss as i32,
            r.as_ptr(),
            rs as i32,
            p.as_ptr(),
        )
    }
}

/// Reference `aom_highbd_sad<W>x<H>_c` for size index `idx`.
pub fn ref_hbd_sad(idx: usize, s: &[u16], ss: usize, r: &[u16], rs: usize) -> u32 {
    unsafe { shim_hbd_sad(idx as i32, s.as_ptr(), ss as i32, r.as_ptr(), rs as i32) }
}

/// Reference `aom_highbd_<bd>_variance<W>x<H>_c`; returns (variance, sse).
pub fn ref_hbd_variance(
    idx: usize,
    bd: u8,
    a: &[u16],
    as_: usize,
    b: &[u16],
    bs: usize,
) -> (u32, u32) {
    let mut sse = 0u32;
    let v = unsafe {
        shim_hbd_var(
            idx as i32,
            bd as i32,
            a.as_ptr(),
            as_ as i32,
            b.as_ptr(),
            bs as i32,
            &mut sse,
        )
    };
    (v, sse)
}

/// Reference `aom_highbd_<bd>_sub_pixel_variance<W>x<H>_c`; returns (variance, sse).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_subpel_var(
    idx: usize,
    bd: u8,
    a: &[u16],
    as_: usize,
    xo: usize,
    yo: usize,
    b: &[u16],
    bs: usize,
) -> (u32, u32) {
    let mut sse = 0u32;
    let v = unsafe {
        shim_hbd_subpel_var(
            idx as i32,
            bd as i32,
            a.as_ptr(),
            as_ as i32,
            xo as i32,
            yo as i32,
            b.as_ptr(),
            bs as i32,
            &mut sse,
        )
    };
    (v, sse)
}

// hbd_lpf_shim.c — highbd deblocking edge filters.
extern "C" {
    fn shim_hbd_lpf(
        dir: i32,
        width: i32,
        s: *mut u16,
        p: i32,
        bl: *const u8,
        li: *const u8,
        th: *const u8,
        bd: i32,
    );
}

/// Apply a reference highbd loop filter in place. `dir`: 'h'/'v'.
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_lpf(
    dir: u8,
    width: u32,
    buf: &mut [u16],
    center: usize,
    pitch: usize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
) {
    let (b, l, t) = ([bl], [li], [th]);
    let d = if dir == b'h' { 0 } else { 1 };
    unsafe {
        shim_hbd_lpf(
            d,
            width as i32,
            buf.as_mut_ptr().add(center),
            pitch as i32,
            b.as_ptr(),
            l.as_ptr(),
            t.as_ptr(),
            bd,
        );
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
pub fn ref_lpf(
    dir: u8,
    width: u32,
    buf: &mut [u8],
    center: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
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
        f(
            buf.as_mut_ptr().add(center),
            pitch as i32,
            b.as_ptr(),
            l.as_ptr(),
            t.as_ptr(),
        );
    }
}

// av1/common/reconintra.c — directional predictors (edges passed at +pad).
extern "C" {
    pub fn av1_dr_prediction_z1_c(
        dst: *mut u8,
        stride: isize,
        bw: i32,
        bh: i32,
        above: *const u8,
        left: *const u8,
        upsample_above: i32,
        dx: i32,
        dy: i32,
    );
    pub fn av1_dr_prediction_z2_c(
        dst: *mut u8,
        stride: isize,
        bw: i32,
        bh: i32,
        above: *const u8,
        left: *const u8,
        upsample_above: i32,
        upsample_left: i32,
        dx: i32,
        dy: i32,
    );
    pub fn av1_dr_prediction_z3_c(
        dst: *mut u8,
        stride: isize,
        bw: i32,
        bh: i32,
        above: *const u8,
        left: *const u8,
        upsample_left: i32,
        dx: i32,
        dy: i32,
    );
    pub fn av1_highbd_dr_prediction_z1_c(
        dst: *mut u16,
        stride: isize,
        bw: i32,
        bh: i32,
        above: *const u16,
        left: *const u16,
        upsample_above: i32,
        dx: i32,
        dy: i32,
        bd: i32,
    );
    pub fn av1_highbd_dr_prediction_z2_c(
        dst: *mut u16,
        stride: isize,
        bw: i32,
        bh: i32,
        above: *const u16,
        left: *const u16,
        upsample_above: i32,
        upsample_left: i32,
        dx: i32,
        dy: i32,
        bd: i32,
    );
    pub fn av1_highbd_dr_prediction_z3_c(
        dst: *mut u16,
        stride: isize,
        bw: i32,
        bh: i32,
        above: *const u16,
        left: *const u16,
        upsample_left: i32,
        dx: i32,
        dy: i32,
        bd: i32,
    );
}

/// Reference highbd intra prediction dispatch (`av1_predict_intra_block` routing,
/// minus palette / CfL): pick the predictor family, derive `p_angle` for
/// directional modes, and build. `recon[ref_off]` is the block top-left. Returns
/// the `txw*txh` block (row stride `txw`).
#[allow(clippy::too_many_arguments)]
pub fn ref_hbd_predict_intra(
    recon: &[u16],
    ref_off: usize,
    ref_stride: usize,
    mode: usize,
    angle_delta: i32,
    use_filter_intra: bool,
    filter_intra_mode: usize,
    disable_edge_filter: bool,
    filt_type: i32,
    tx_size: usize,
    txw: usize,
    txh: usize,
    n_top_px: i32,
    n_topright_px: i32,
    n_left_px: i32,
    n_bottomleft_px: i32,
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_predict_intra(
            recon.as_ptr().add(ref_off),
            ref_stride as i32,
            mode as i32,
            angle_delta,
            use_filter_intra as i32,
            filter_intra_mode as i32,
            disable_edge_filter as i32,
            filt_type,
            tx_size as i32,
            n_top_px,
            n_topright_px,
            n_left_px,
            n_bottomleft_px,
            bd,
            dst.as_mut_ptr(),
            txw as i32,
        )
    }
    dst
}

/// Reference highbd filter-intra predictor (`highbd_filter_intra_predictor`).
/// `above` is a `[-1..]` view (index 0 the corner), `left` is `left[0..bh]`;
/// `mode` is the `FILTER_INTRA_MODE`. Returns the `bw*bh` block (row stride `bw`).
pub fn ref_hbd_filter_intra(
    tx_size: usize,
    bw: usize,
    bh: usize,
    above: &[u16],
    left: &[u16],
    mode: usize,
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; bw * bh];
    unsafe {
        shim_hbd_filter_intra_predict(
            dst.as_mut_ptr(),
            bw as isize,
            tx_size as i32,
            above.as_ptr(),
            left.as_ptr(),
            mode as i32,
            bd,
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
    recon: &[u16],
    ref_off: usize,
    ref_stride: usize,
    p_angle: i32,
    disable_edge_filter: bool,
    filt_type: i32,
    tx_size: usize,
    txw: usize,
    txh: usize,
    n_top_px: i32,
    n_topright_px: i32,
    n_left_px: i32,
    n_bottomleft_px: i32,
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_build_dir_intra(
            recon.as_ptr().add(ref_off),
            ref_stride as i32,
            p_angle,
            disable_edge_filter as i32,
            filt_type,
            tx_size as i32,
            n_top_px,
            n_topright_px,
            n_left_px,
            n_bottomleft_px,
            0,
            0,
            bd,
            dst.as_mut_ptr(),
            txw as i32,
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
    recon: &[u16],
    ref_off: usize,
    ref_stride: usize,
    filter_intra_mode: i32,
    tx_size: usize,
    txw: usize,
    txh: usize,
    n_top_px: i32,
    n_topright_px: i32,
    n_left_px: i32,
    n_bottomleft_px: i32,
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_build_dir_intra(
            recon.as_ptr().add(ref_off),
            ref_stride as i32,
            90,
            0,
            0,
            tx_size as i32,
            n_top_px,
            n_topright_px,
            n_left_px,
            n_bottomleft_px,
            1,
            filter_intra_mode,
            bd,
            dst.as_mut_ptr(),
            txw as i32,
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
    tx_size: usize,
    txw: usize,
    txh: usize,
    above_data: &[u16],
    left_data: &[u16],
    pad: usize,
    up_above: i32,
    up_left: i32,
    angle: i32,
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_dr_predict(
            dst.as_mut_ptr(),
            txw as isize,
            tx_size as i32,
            above_data.as_ptr().add(pad),
            left_data.as_ptr().add(pad),
            up_above,
            up_left,
            angle,
            bd,
        )
    }
    dst
}

/// Reference highbd directional predictor. `above`/`left` are padded `u16`
/// buffers; the C pointer is taken at offset `pad`. Returns the `bw*bh` block
/// (row stride `bw`).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_dr_pred(
    kind: u8,
    bw: usize,
    bh: usize,
    above: &[u16],
    left: &[u16],
    pad: usize,
    up_above: i32,
    up_left: i32,
    dx: i32,
    dy: i32,
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; bw * bh];
    let ap = unsafe { above.as_ptr().add(pad) };
    let lp = unsafe { left.as_ptr().add(pad) };
    unsafe {
        match kind {
            1 => av1_highbd_dr_prediction_z1_c(
                dst.as_mut_ptr(),
                bw as isize,
                bw as i32,
                bh as i32,
                ap,
                lp,
                up_above,
                dx,
                dy,
                bd,
            ),
            2 => av1_highbd_dr_prediction_z2_c(
                dst.as_mut_ptr(),
                bw as isize,
                bw as i32,
                bh as i32,
                ap,
                lp,
                up_above,
                up_left,
                dx,
                dy,
                bd,
            ),
            3 => av1_highbd_dr_prediction_z3_c(
                dst.as_mut_ptr(),
                bw as isize,
                bw as i32,
                bh as i32,
                ap,
                lp,
                up_left,
                dx,
                dy,
                bd,
            ),
            _ => unreachable!(),
        }
    }
    dst
}

/// Reference directional predictor. `above`/`left` are padded buffers; the C
/// pointer is taken at offset `pad`. Returns the `bw*bh` block (stride = bw).
#[allow(clippy::too_many_arguments)]
pub fn ref_dr_pred(
    kind: u8,
    bw: usize,
    bh: usize,
    above: &[u8],
    left: &[u8],
    pad: usize,
    up_above: i32,
    up_left: i32,
    dx: i32,
    dy: i32,
) -> Vec<u8> {
    let mut dst = vec![0u8; bw * bh];
    let ap = unsafe { above.as_ptr().add(pad) };
    let lp = unsafe { left.as_ptr().add(pad) };
    unsafe {
        match kind {
            1 => av1_dr_prediction_z1_c(
                dst.as_mut_ptr(),
                bw as isize,
                bw as i32,
                bh as i32,
                ap,
                lp,
                up_above,
                dx,
                dy,
            ),
            2 => av1_dr_prediction_z2_c(
                dst.as_mut_ptr(),
                bw as isize,
                bw as i32,
                bh as i32,
                ap,
                lp,
                up_above,
                up_left,
                dx,
                dy,
            ),
            3 => av1_dr_prediction_z3_c(
                dst.as_mut_ptr(),
                bw as isize,
                bw as i32,
                bh as i32,
                ap,
                lp,
                up_left,
                dx,
                dy,
            ),
            _ => unreachable!(),
        }
    }
    dst
}

// intra_shim.c — dispatch to aom_<mode>_predictor_<W>x<H>_c.
extern "C" {
    fn shim_intra_pred(
        mode: i32,
        size_idx: i32,
        dst: *mut u8,
        stride: isize,
        above: *const u8,
        left: *const u8,
    );
}

extern "C" {
    fn shim_highbd_intra_pred(
        mode: i32,
        size_idx: i32,
        dst: *mut u16,
        stride: isize,
        above: *const u16,
        left: *const u16,
        bd: i32,
    );
    fn shim_hbd_build_nd_intra(
        r: *const u16,
        ref_stride: i32,
        av1_mode: i32,
        tx_size: i32,
        n_top_px: i32,
        n_left_px: i32,
        bd: i32,
        dst: *mut u16,
        dst_stride: i32,
    );
    fn shim_hbd_dr_predict(
        dst: *mut u16,
        stride: isize,
        tx_size: i32,
        above: *const u16,
        left: *const u16,
        up_above: i32,
        up_left: i32,
        angle: i32,
        bd: i32,
    );
    fn shim_hbd_build_dir_intra(
        r: *const u16,
        ref_stride: i32,
        p_angle: i32,
        disable_edge_filter: i32,
        filt_type: i32,
        tx_size: i32,
        n_top_px: i32,
        n_topright_px: i32,
        n_left_px: i32,
        n_bottomleft_px: i32,
        use_filter_intra: i32,
        filter_intra_mode: i32,
        bd: i32,
        dst: *mut u16,
        dst_stride: i32,
    );
    fn shim_hbd_filter_intra_predict(
        dst: *mut u16,
        stride: isize,
        tx_size: i32,
        above: *const u16,
        left: *const u16,
        mode: i32,
        bd: i32,
    );
    fn shim_hbd_predict_intra(
        r: *const u16,
        ref_stride: i32,
        mode: i32,
        angle_delta: i32,
        use_filter_intra: i32,
        filter_intra_mode: i32,
        disable_edge_filter: i32,
        filt_type: i32,
        tx_size: i32,
        n_top_px: i32,
        n_topright_px: i32,
        n_left_px: i32,
        n_bottomleft_px: i32,
        bd: i32,
        dst: *mut u16,
        dst_stride: i32,
    );
}

/// Reference highbd intra prediction. Returns the `bw*bh` predicted block.
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_intra_pred(
    mode: usize,
    size_idx: usize,
    bw: usize,
    bh: usize,
    above_tl: &[u16],
    left: &[u16],
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; bw * bh];
    unsafe {
        shim_highbd_intra_pred(
            mode as i32,
            size_idx as i32,
            dst.as_mut_ptr(),
            bw as isize,
            above_tl.as_ptr().add(1),
            left.as_ptr(),
            bd,
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
    recon: &[u16],
    ref_off: usize,
    ref_stride: usize,
    av1_mode: usize,
    tx_size: usize,
    txw: usize,
    txh: usize,
    n_top_px: usize,
    n_left_px: usize,
    bd: i32,
) -> Vec<u16> {
    let mut dst = vec![0u16; txw * txh];
    unsafe {
        shim_hbd_build_nd_intra(
            recon.as_ptr().add(ref_off),
            ref_stride as i32,
            av1_mode as i32,
            tx_size as i32,
            n_top_px as i32,
            n_left_px as i32,
            bd,
            dst.as_mut_ptr(),
            txw as i32,
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
            mode as i32,
            size_idx as i32,
            dst.as_mut_ptr(),
            bw as isize,
            above_tl.as_ptr().add(1),
            left.as_ptr(),
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
            syms.as_ptr(),
            syms.len() as i32,
            cdf_init.as_ptr(),
            nsymbs as i32,
            out.as_mut_ptr(),
            out.len() as u32,
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
            buf.as_ptr(),
            buf.len() as u32,
            n as i32,
            cdf_init.as_ptr(),
            nsymbs as i32,
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
    unsafe {
        f(
            input.as_ptr(),
            out.as_mut_ptr(),
            cos_bit,
            stage_range.as_ptr(),
        )
    }
    out
}

/// Convenience wrapper kept for the original fdct4 harness.
pub fn ref_fdct4(input: &[i32; 4], cos_bit: i8, stage_range: &[i8; 8]) -> [i32; 4] {
    let mut out = [0i32; 4];
    unsafe {
        av1_fdct4(
            input.as_ptr(),
            out.as_mut_ptr(),
            cos_bit,
            stage_range.as_ptr(),
        )
    }
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
    const W: [usize; 19] = [
        4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
    ];
    const H: [usize; 19] = [
        4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
    ];
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
    unsafe {
        f(
            input.as_ptr(),
            out.as_mut_ptr(),
            stride as i32,
            tx_type as i32,
            8,
        )
    }
    out
}

// av1/encoder/hybrid_fwd_txfm.c — the 4x4 reversible Walsh–Hadamard forward
// transform for coded-lossless blocks. Shared for high and low bit depth (no
// separate highbd variant). Signature: (const int16_t*, tran_low_t* /*int32*/,
// int stride).
extern "C" {
    pub fn av1_fwht4x4_c(input: *const i16, output: *mut i32, stride: i32);
}

/// Reference forward 4x4 Walsh–Hadamard (`av1_fwht4x4_c`) — the coded-lossless
/// forward transform. `input` is the 4x4 residual with row stride `stride`
/// (must be readable at `input[i + 3*stride]` for `i in 0..4`); returns the
/// 16-entry raster coefficient block. Bit-depth-independent.
pub fn ref_fwht4x4(input: &[i16], stride: usize) -> Vec<i32> {
    ref_init();
    let mut out = vec![0i32; 16];
    unsafe {
        av1_fwht4x4_c(input.as_ptr(), out.as_mut_ptr(), stride as i32);
    }
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

extern "C" {
    // Lossless 4x4 Walsh–Hadamard inverse-add. The _c kernels use libaom's
    // packed-pointer highbd convention (CONVERT_TO_SHORTPTR shifts the dest
    // pointer <<1), unlike the native-u16* inv_txfm2d_add_*_c above, so the shim
    // wraps CONVERT_TO_BYTEPTR and dispatches on eob. Takes a native u16 dest.
    fn shim_highbd_iwht4x4_add(input: *const i32, dest: *mut u16, stride: i32, eob: i32, bd: i32);
}

// av1/encoder/av1_quantize.c — fast-path quantizers (no quant matrix).
pub type QuantFpFn = unsafe extern "C" fn(
    *const i32,
    isize,
    *const i16,
    *const i16,
    *const i16,
    *const i16,
    *mut i32,
    *mut i32,
    *const i16,
    *mut u16,
    *const i16,
    *const i16,
);
extern "C" {
    pub fn av1_quantize_fp_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
    );
    pub fn av1_quantize_fp_32x32_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
    );
    pub fn av1_quantize_fp_64x64_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
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
            coeff.as_ptr(),
            n as isize,
            dummy.as_ptr(),
            round.as_ptr(),
            quant.as_ptr(),
            dummy.as_ptr(),
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant.as_ptr(),
            &mut eob,
            scan.as_ptr(),
            dummy.as_ptr(),
        )
    }
    (qcoeff, dqcoeff, eob)
}

// aom_dsp/quantize.c — "b" quantizer helper (dead-zone + quant/quant_shift).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn aom_quantize_b_helper_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
        qm: *const u8,
        iqm: *const u8,
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
            coeff.as_ptr(),
            n as isize,
            zbin.as_ptr(),
            round.as_ptr(),
            quant.as_ptr(),
            quant_shift.as_ptr(),
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant.as_ptr(),
            &mut eob,
            scan.as_ptr(),
            dummy.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            log_scale,
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
            coeff.as_ptr(),
            n as isize,
            zbin.as_ptr(),
            round.as_ptr(),
            quant.as_ptr(),
            quant_shift.as_ptr(),
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant.as_ptr(),
            &mut eob,
            scan.as_ptr(),
            dummy.as_ptr(),
            qm.as_ptr(),
            iqm.as_ptr(),
            log_scale,
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
            coeff.as_ptr(),
            n as isize,
            zbin.as_ptr(),
            round.as_ptr(),
            quant.as_ptr(),
            quant_shift.as_ptr(),
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant.as_ptr(),
            &mut eob,
            scan.as_ptr(),
            dummy.as_ptr(),
            qm.as_ptr(),
            iqm.as_ptr(),
            log_scale,
        )
    }
    (qcoeff, dqcoeff, eob)
}

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn aom_quantize_b_adaptive_helper_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
        qm: *const u8,
        iqm: *const u8,
        log_scale: i32,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn aom_highbd_quantize_b_adaptive_helper_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
        qm: *const u8,
        iqm: *const u8,
        log_scale: i32,
    );
}

/// Reference `aom_quantize_b_adaptive_helper_c` / `aom_highbd_..._c` (the
/// `--quant-b-adapt` dead-zone quantizer). `hbd` selects the 64-bit highbd
/// helper. `qm`/`iqm` = `None` is the no-matrix case (the funnel passes NULL).
/// Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_quantize_b_adaptive(
    hbd: bool,
    log_scale: i32,
    coeff: &[i32],
    zbin: &[i16; 2],
    round: &[i16; 2],
    quant: &[i16; 2],
    quant_shift: &[i16; 2],
    dequant: &[i16; 2],
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
    scan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let mut qcoeff = vec![0i32; n];
    let mut dqcoeff = vec![0i32; n];
    let mut eob: u16 = 0;
    let dummy = vec![0i16; n.max(2)];
    let qm_ptr = qm.map_or(std::ptr::null(), |m| m.as_ptr());
    let iqm_ptr = iqm.map_or(std::ptr::null(), |m| m.as_ptr());
    let f = if hbd {
        aom_highbd_quantize_b_adaptive_helper_c
    } else {
        aom_quantize_b_adaptive_helper_c
    };
    unsafe {
        f(
            coeff.as_ptr(),
            n as isize,
            zbin.as_ptr(),
            round.as_ptr(),
            quant.as_ptr(),
            quant_shift.as_ptr(),
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            dequant.as_ptr(),
            &mut eob,
            scan.as_ptr(),
            dummy.as_ptr(),
            qm_ptr,
            iqm_ptr,
            log_scale,
        )
    }
    (qcoeff, dqcoeff, eob)
}

extern "C" {
    /// `default_tx_type_probs[FRAME_UPDATE_TYPES=7][TX_SIZES_ALL=19][TX_TYPES=16]`
    /// (encoder_utils.c:44) — the frame-probability defaults `copy_frame_prob_info`
    /// seeds `frame_probs.tx_type_probs` from for the `prune_tx_type_using_stats`
    /// sf. Exported `extern const int` (encoder_utils.h:29).
    static default_tx_type_probs: [[[i32; 16]; 19]; 7];
}

/// The `KF_UPDATE` (== 0) slab of the real exported `default_tx_type_probs` — the
/// per-`TX_SIZE_ALL` prob row a lone KEY still uses for the stats prune.
#[must_use]
pub fn ref_default_tx_type_probs_kf() -> [[i32; 16]; 19] {
    unsafe { default_tx_type_probs[0] }
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
    unsafe {
        f(
            input.as_ptr(),
            dest.as_mut_ptr(),
            stride as i32,
            tx_type as i32,
            bd,
        )
    }
}

/// `av1_highbd_iwht4x4_add` (`av1/common/idct.c`): the lossless 4x4 Walsh–Hadamard
/// inverse-add. Dispatches on `eob` (>1 -> 16-point, else DC-only) exactly like
/// the C wrapper. `input` is the dequantized 4x4 block; `dest` holds the
/// prediction on entry and the reconstruction on return (bd-bit samples, row
/// `stride`).
pub fn ref_highbd_iwht4x4_add(input: &[i32], dest: &mut [u16], stride: usize, eob: usize, bd: i32) {
    unsafe {
        shim_highbd_iwht4x4_add(
            input.as_ptr(),
            dest.as_mut_ptr(),
            stride as i32,
            eob as i32,
            bd,
        );
    }
}

// txb_shim.c — transform-block coefficient-coding kernels + scan/ctx data.
extern "C" {
    fn shim_txb_init_levels(coeff: *const i32, width: i32, height: i32, levels: *mut u8);
    fn shim_get_nz_map_contexts(
        levels: *const u8,
        scan: *const i16,
        eob: i32,
        tx_size: i32,
        tx_class: i32,
        out: *mut i8,
    );
    fn shim_eob_pos_token(eob: i32, extra: *mut i32) -> i32;
    fn shim_nz_ctx_offset(tx_size: i32) -> *const i8;
    fn shim_scan(tx_size: i32, tx_type: i32) -> *const i16;
    fn shim_iscan(tx_size: i32, tx_type: i32) -> *const i16;
    fn shim_cost_tokens_from_cdf(costs: *mut i32, cdf: *const u16, inv_map: *const i32);
    fn shim_get_txb_ctx(
        plane_bsize: i32,
        tx_size: i32,
        plane: i32,
        a: *const i8,
        l: *const i8,
        out: *mut i32,
    );
    fn shim_txb_entropy_context(qcoeff: *const i32, tx_size: i32, tx_type: i32, eob: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_optimize_txb(
        tx_size: i32,
        tx_type: i32,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        tcoeff: *const i32,
        eob: i32,
        dequant: *const i16,
        rdmult: i64,
        dc_sign_ctx: i32,
        txb_skip_ctx: i32,
        sharpness: i32,
        scan: *const i16,
        txb_skip_cost: *const i32,
        base_eob_cost: *const i32,
        base_cost: *const i32,
        eob_extra_cost: *const i32,
        dc_sign_cost: *const i32,
        lps_cost: *const i32,
        eob_cost: *const i32,
        iqm: *const u8,
        qm: *const u8,
        out_rate: *mut i32,
    ) -> i32;
    fn shim_get_dqv(dequant: *const i16, coeff_idx: i32, iqm: *const u8) -> i32;
    fn shim_get_coeff_dist(
        tcoeff: i32,
        dqcoeff: i32,
        shift: i32,
        qm: *const u8,
        coeff_idx: i32,
    ) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn shim_two_coeff_cost_simple(
        ci: i32,
        abs_qc: i32,
        coeff_ctx: i32,
        base: *const i32,
        lps: *const i32,
        bhl: i32,
        tx_class: i32,
        levels: *const u8,
        cost_low: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_coeff_cost_eob(
        ci: i32,
        abs_qc: i32,
        sign: i32,
        coeff_ctx: i32,
        dc_sign_ctx: i32,
        base_eob: *const i32,
        dc_sign: *const i32,
        lps: *const i32,
        bhl: i32,
        tx_class: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_coeff_cost_general(
        is_last: i32,
        ci: i32,
        abs_qc: i32,
        sign: i32,
        coeff_ctx: i32,
        dc_sign_ctx: i32,
        base_eob: *const i32,
        base: *const i32,
        dc_sign: *const i32,
        lps: *const i32,
        bhl: i32,
        tx_class: i32,
        levels: *const u8,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_ext_tx_derive(
        tx_size: i32,
        is_inter: i32,
        reduced: i32,
        tx_type: i32,
        use_fi: i32,
        fi_mode: i32,
        mode: i32,
        out: *mut i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_fill_lv_map(
        txb_skip_cdf: *const u16,
        base_eob_cdf: *const u16,
        base_cdf: *const u16,
        eob_extra_cdf: *const u16,
        dc_sign_cdf: *const u16,
        br_cdf: *const u16,
        o_txb_skip: *mut i32,
        o_base_eob: *mut i32,
        o_base: *mut i32,
        o_eob_extra: *mut i32,
        o_dc_sign: *mut i32,
        o_lps: *mut i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_cost_coeffs_txb(
        qcoeff: *const i32,
        eob: i32,
        tx_size: i32,
        tx_type: i32,
        txb_skip_ctx: i32,
        dc_sign_ctx: i32,
        txb_skip_cost: *const i32,
        base_eob_cost: *const i32,
        base_cost: *const i32,
        eob_extra_cost: *const i32,
        dc_sign_cost: *const i32,
        lps_cost: *const i32,
        eob_cost: *const i32,
    ) -> i32;
    fn shim_cost_coeffs_txb_laplacian(
        qcoeff: *const i32,
        eob: i32,
        tx_size: i32,
        tx_type: i32,
        txb_skip_ctx: i32,
        txb_skip_cost: *const i32,
        eob_extra_cost: *const i32,
        eob_cost: *const i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_write_coeffs_txb(
        tcoeff: *const i32,
        eob: i32,
        tx_size: i32,
        tx_type: i32,
        plane_type: i32,
        txb_skip_ctx: i32,
        dc_sign_ctx: i32,
        allow_update_cdf: i32,
        cdfs: *mut u16,
        ext_tx_cdf: *mut u16,
        is_inter: i32,
        reduced: i32,
        use_fi: i32,
        fi_mode: i32,
        mode: i32,
        signal_gate: i32,
        out: *mut u8,
        out_cap: i32,
    ) -> i32;
    fn shim_dequant_txb(
        qcoeff: *const i32,
        dqcoeff: *mut i32,
        area: i32,
        tx_size: i32,
        dequant: *const i16,
        iqmatrix: *const u8,
        bd: i32,
    );
}

/// Reference `av1_txb_init_levels_c` (writes into `levels`).
pub fn ref_txb_init_levels(coeff: &[i32], width: usize, height: usize, levels: &mut [u8]) {
    unsafe {
        shim_txb_init_levels(
            coeff.as_ptr(),
            width as i32,
            height as i32,
            levels.as_mut_ptr(),
        )
    }
}

/// Reference decoder dequant (`av1_read_coeffs_txb` math, decodetxb.c): signed
/// `qcoeff` (raster, len `area`) → `dqcoeff` (raster). `iqmatrix` per raster
/// position, `None` for no quant matrix. Applies the `0xfffff`/`0xffffff` masks,
/// `av1_get_tx_scale` shift, and `±(1<<(7+bd))` clamp exactly as the C decoder.
pub fn ref_dequant_txb(
    qcoeff: &[i32],
    tx_size: usize,
    dequant: [i16; 2],
    iqmatrix: Option<&[u8]>,
    bd: i32,
) -> Vec<i32> {
    let area = qcoeff.len();
    let mut dq = vec![0i32; area];
    let iqp = iqmatrix.map_or(core::ptr::null(), |s| s.as_ptr());
    unsafe {
        shim_dequant_txb(
            qcoeff.as_ptr(),
            dq.as_mut_ptr(),
            area as i32,
            tx_size as i32,
            dequant.as_ptr(),
            iqp,
            bd,
        )
    }
    dq
}

/// Reference `av1_get_nz_map_contexts_c` (writes `out[scan[i]]` for `i < eob`).
pub fn ref_get_nz_map_contexts(
    levels: &[u8],
    scan: &[i16],
    eob: usize,
    tx_size: usize,
    tx_class: i32,
    out: &mut [i8],
) {
    unsafe {
        shim_get_nz_map_contexts(
            levels.as_ptr(),
            scan.as_ptr(),
            eob as i32,
            tx_size as i32,
            tx_class,
            out.as_mut_ptr(),
        )
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
pub fn ref_write_coeffs_txb(
    tcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    plane_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    allow_update_cdf: bool,
    cdfs: &mut [u16],
) -> Vec<u8> {
    let mut out = vec![0u8; 1 << 16];
    let mut dummy = [0u16; 16];
    // signal_gate = 0 => no tx_type write (reproduces the coeff-only path).
    let n = unsafe {
        shim_write_coeffs_txb(
            tcoeff.as_ptr(),
            eob as i32,
            tx_size as i32,
            tx_type as i32,
            plane_type as i32,
            txb_skip_ctx as i32,
            dc_sign_ctx as i32,
            allow_update_cdf as i32,
            cdfs.as_mut_ptr(),
            dummy.as_mut_ptr(),
            0,
            0,
            0,
            0,
            0,
            0,
            out.as_mut_ptr(),
            out.len() as i32,
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference full txb writer: `txb_skip` + (luma) `av1_write_tx_type` + coeffs,
/// matching aom-txb's `write_coeffs_txb_full`. `ext_tx_cdf` is the selected ext-tx
/// CDF slot; the tx_type context mirrors the encoder mbmi/frame state. Returns bytes.
#[allow(clippy::too_many_arguments)]
pub fn ref_write_coeffs_txb_full(
    tcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    plane_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    allow_update_cdf: bool,
    cdfs: &mut [u16],
    ext_tx_cdf: &mut [u16],
    is_inter: bool,
    reduced: bool,
    use_fi: bool,
    fi_mode: usize,
    mode: usize,
    signal_gate: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; 1 << 16];
    let n = unsafe {
        shim_write_coeffs_txb(
            tcoeff.as_ptr(),
            eob as i32,
            tx_size as i32,
            tx_type as i32,
            plane_type as i32,
            txb_skip_ctx as i32,
            dc_sign_ctx as i32,
            allow_update_cdf as i32,
            cdfs.as_mut_ptr(),
            ext_tx_cdf.as_mut_ptr(),
            is_inter as i32,
            reduced as i32,
            use_fi as i32,
            fi_mode as i32,
            mode as i32,
            signal_gate as i32,
            out.as_mut_ptr(),
            out.len() as i32,
        )
    };
    out.truncate(n as usize);
    out
}

/// Reference `av1_cost_coeffs_txb` (warehouse_efficients_txb, tx_type cost out
/// of scope). Cost tables are flat: txb_skip_cost[13][2], base_eob_cost[4][3],
/// base_cost[42][8], eob_extra_cost[9][2], dc_sign_cost[3][2], lps_cost[21][26],
/// eob_cost[2][11].
#[allow(clippy::too_many_arguments)]
pub fn ref_cost_coeffs_txb(
    qcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    txb_skip_cost: &[i32],
    base_eob_cost: &[i32],
    base_cost: &[i32],
    eob_extra_cost: &[i32],
    dc_sign_cost: &[i32],
    lps_cost: &[i32],
    eob_cost: &[i32],
) -> i32 {
    unsafe {
        shim_cost_coeffs_txb(
            qcoeff.as_ptr(),
            eob as i32,
            tx_size as i32,
            tx_type as i32,
            txb_skip_ctx as i32,
            dc_sign_ctx as i32,
            txb_skip_cost.as_ptr(),
            base_eob_cost.as_ptr(),
            base_cost.as_ptr(),
            eob_extra_cost.as_ptr(),
            dc_sign_cost.as_ptr(),
            lps_cost.as_ptr(),
            eob_cost.as_ptr(),
        )
    }
}

/// Reference `av1_cost_coeffs_txb_laplacian` (adjust_eob=0, tx_type cost out of
/// scope — matching [`ref_cost_coeffs_txb`]'s split): the est-rd prune's fast
/// Laplacian rate. Uses the REAL txb_rdopt_utils.h statics
/// (costLUT/const_term/loge_par) + the pristine get_eob_cost.
#[allow(clippy::too_many_arguments)]
pub fn ref_cost_coeffs_txb_laplacian(
    qcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    txb_skip_ctx: usize,
    txb_skip_cost: &[i32],
    eob_extra_cost: &[i32],
    eob_cost: &[i32],
) -> i32 {
    unsafe {
        shim_cost_coeffs_txb_laplacian(
            qcoeff.as_ptr(),
            eob as i32,
            tx_size as i32,
            tx_type as i32,
            txb_skip_ctx as i32,
            txb_skip_cost.as_ptr(),
            eob_extra_cost.as_ptr(),
            eob_cost.as_ptr(),
        )
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
pub fn ref_fill_lv_map(
    txb_skip_cdf: &[u16],
    base_eob_cdf: &[u16],
    base_cdf: &[u16],
    eob_extra_cdf: &[u16],
    dc_sign_cdf: &[u16],
    br_cdf: &[u16],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut txb_skip = vec![0i32; 13 * 2];
    let mut base_eob = vec![0i32; 4 * 3];
    let mut base = vec![0i32; 42 * 8];
    let mut eob_extra = vec![0i32; 9 * 2];
    let mut dc_sign = vec![0i32; 3 * 2];
    let mut lps = vec![0i32; 21 * 26];
    unsafe {
        shim_fill_lv_map(
            txb_skip_cdf.as_ptr(),
            base_eob_cdf.as_ptr(),
            base_cdf.as_ptr(),
            eob_extra_cdf.as_ptr(),
            dc_sign_cdf.as_ptr(),
            br_cdf.as_ptr(),
            txb_skip.as_mut_ptr(),
            base_eob.as_mut_ptr(),
            base.as_mut_ptr(),
            eob_extra.as_mut_ptr(),
            dc_sign.as_mut_ptr(),
            lps.as_mut_ptr(),
        );
    }
    (txb_skip, base_eob, base, eob_extra, dc_sign, lps)
}

/// Reference ext-tx derivation for `av1_write_tx_type`. Returns
/// (set_type, num, eset, square_tx_size, symb, used, intra_dir).
#[allow(clippy::too_many_arguments)]
pub fn ref_ext_tx_derive(
    tx_size: usize,
    is_inter: bool,
    reduced: bool,
    tx_type: usize,
    use_fi: bool,
    fi_mode: usize,
    mode: usize,
) -> [i32; 7] {
    let mut out = [0i32; 7];
    unsafe {
        shim_ext_tx_derive(
            tx_size as i32,
            is_inter as i32,
            reduced as i32,
            tx_type as i32,
            use_fi as i32,
            fi_mode as i32,
            mode as i32,
            out.as_mut_ptr(),
        )
    }
    out
}

/// Reference `get_two_coeff_cost_simple`; returns (cost, cost_low).
#[allow(clippy::too_many_arguments)]
pub fn ref_two_coeff_cost_simple(
    ci: usize,
    abs_qc: i32,
    coeff_ctx: usize,
    base: &[i32],
    lps: &[i32],
    bhl: u32,
    tx_class: i32,
    levels: &[u8],
) -> (i32, i32) {
    let mut cost_low = 0i32;
    let cost = unsafe {
        shim_two_coeff_cost_simple(
            ci as i32,
            abs_qc,
            coeff_ctx as i32,
            base.as_ptr(),
            lps.as_ptr(),
            bhl as i32,
            tx_class,
            levels.as_ptr(),
            &mut cost_low,
        )
    };
    (cost, cost_low)
}

/// Reference `get_coeff_cost_eob`.
#[allow(clippy::too_many_arguments)]
pub fn ref_coeff_cost_eob(
    ci: usize,
    abs_qc: i32,
    sign: usize,
    coeff_ctx: usize,
    dc_sign_ctx: usize,
    base_eob: &[i32],
    dc_sign: &[i32],
    lps: &[i32],
    bhl: u32,
    tx_class: i32,
) -> i32 {
    unsafe {
        shim_coeff_cost_eob(
            ci as i32,
            abs_qc,
            sign as i32,
            coeff_ctx as i32,
            dc_sign_ctx as i32,
            base_eob.as_ptr(),
            dc_sign.as_ptr(),
            lps.as_ptr(),
            bhl as i32,
            tx_class,
        )
    }
}

/// Reference `get_coeff_cost_general`.
#[allow(clippy::too_many_arguments)]
pub fn ref_coeff_cost_general(
    is_last: bool,
    ci: usize,
    abs_qc: i32,
    sign: usize,
    coeff_ctx: usize,
    dc_sign_ctx: usize,
    base_eob: &[i32],
    base: &[i32],
    dc_sign: &[i32],
    lps: &[i32],
    bhl: u32,
    tx_class: i32,
    levels: &[u8],
) -> i32 {
    unsafe {
        shim_coeff_cost_general(
            is_last as i32,
            ci as i32,
            abs_qc,
            sign as i32,
            coeff_ctx as i32,
            dc_sign_ctx as i32,
            base_eob.as_ptr(),
            base.as_ptr(),
            dc_sign.as_ptr(),
            lps.as_ptr(),
            bhl as i32,
            tx_class,
            levels.as_ptr(),
        )
    }
}

/// Reference `av1_optimize_txb` (non-QM trellis). Optimizes qcoeff/dqcoeff in
/// place; returns (eob, rate_cost).
#[allow(clippy::too_many_arguments)]
pub fn ref_optimize_txb(
    tx_size: usize,
    tx_type: usize,
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    tcoeff: &[i32],
    eob: usize,
    dequant: &[i16],
    rdmult: i64,
    dc_sign_ctx: usize,
    txb_skip_ctx: usize,
    sharpness: i32,
    scan: &[i16],
    txb_skip: &[i32],
    base_eob: &[i32],
    base: &[i32],
    eob_extra: &[i32],
    dc_sign: &[i32],
    lps: &[i32],
    eob_cost: &[i32],
) -> (usize, i32) {
    let mut rate = 0i32;
    let e = unsafe {
        shim_optimize_txb(
            tx_size as i32,
            tx_type as i32,
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            tcoeff.as_ptr(),
            eob as i32,
            dequant.as_ptr(),
            rdmult,
            dc_sign_ctx as i32,
            txb_skip_ctx as i32,
            sharpness,
            scan.as_ptr(),
            txb_skip.as_ptr(),
            base_eob.as_ptr(),
            base.as_ptr(),
            eob_extra.as_ptr(),
            dc_sign.as_ptr(),
            lps.as_ptr(),
            eob_cost.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            &mut rate,
        )
    };
    (e as usize, rate)
}

/// Reference `av1_optimize_txb` *with* a quant matrix: `iqm` folds into the
/// per-position dequant (`get_dqv`), `qm` folds into the distortion
/// (`get_coeff_dist`). Both indexed by raster position. Returns (eob, rate).
#[allow(clippy::too_many_arguments)]
pub fn ref_optimize_txb_qm(
    tx_size: usize,
    tx_type: usize,
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    tcoeff: &[i32],
    eob: usize,
    dequant: &[i16],
    rdmult: i64,
    dc_sign_ctx: usize,
    txb_skip_ctx: usize,
    sharpness: i32,
    scan: &[i16],
    txb_skip: &[i32],
    base_eob: &[i32],
    base: &[i32],
    eob_extra: &[i32],
    dc_sign: &[i32],
    lps: &[i32],
    eob_cost: &[i32],
    iqm: &[u8],
    qm: &[u8],
) -> (usize, i32) {
    let mut rate = 0i32;
    let e = unsafe {
        shim_optimize_txb(
            tx_size as i32,
            tx_type as i32,
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
            tcoeff.as_ptr(),
            eob as i32,
            dequant.as_ptr(),
            rdmult,
            dc_sign_ctx as i32,
            txb_skip_ctx as i32,
            sharpness,
            scan.as_ptr(),
            txb_skip.as_ptr(),
            base_eob.as_ptr(),
            base.as_ptr(),
            eob_extra.as_ptr(),
            dc_sign.as_ptr(),
            lps.as_ptr(),
            eob_cost.as_ptr(),
            iqm.as_ptr(),
            qm.as_ptr(),
            &mut rate,
        )
    };
    (e as usize, rate)
}

/// Reference `get_dqv` (per-position dequant, folding iqmatrix).
pub fn ref_get_dqv(dequant: &[i16; 2], coeff_idx: usize, iqm: Option<&[u8]>) -> i32 {
    let iqp = iqm.map_or(core::ptr::null(), |s| s.as_ptr());
    unsafe { shim_get_dqv(dequant.as_ptr(), coeff_idx as i32, iqp) }
}

/// Reference `get_coeff_dist` (squared-error distortion, folding qmatrix).
pub fn ref_get_coeff_dist(
    tcoeff: i32,
    dqcoeff: i32,
    shift: i32,
    qm: Option<&[u8]>,
    coeff_idx: usize,
) -> i64 {
    let qp = qm.map_or(core::ptr::null(), |s| s.as_ptr());
    unsafe { shim_get_coeff_dist(tcoeff, dqcoeff, shift, qp, coeff_idx as i32) }
}

/// Reference `get_txb_ctx`; returns (txb_skip_ctx, dc_sign_ctx).
pub fn ref_get_txb_ctx(
    plane_bsize: usize,
    tx_size: usize,
    plane: usize,
    a: &[i8],
    l: &[i8],
) -> (i32, i32) {
    let mut out = [0i32; 2];
    unsafe {
        shim_get_txb_ctx(
            plane_bsize as i32,
            tx_size as i32,
            plane as i32,
            a.as_ptr(),
            l.as_ptr(),
            out.as_mut_ptr(),
        )
    }
    (out[0], out[1])
}

/// Reference `av1_get_txb_entropy_context`.
pub fn ref_txb_entropy_context(qcoeff: &[i32], tx_size: usize, tx_type: usize, eob: usize) -> u8 {
    unsafe {
        shim_txb_entropy_context(qcoeff.as_ptr(), tx_size as i32, tx_type as i32, eob as i32) as u8
    }
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
    pub fn av1_highbd_quantize_fp_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
        log_scale: i32,
    );
    #[allow(clippy::too_many_arguments)]
    pub fn aom_highbd_quantize_b_helper_c(
        coeff: *const i32,
        n: isize,
        zbin: *const i16,
        round: *const i16,
        quant: *const i16,
        quant_shift: *const i16,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
        dequant: *const i16,
        eob: *mut u16,
        scan: *const i16,
        iscan: *const i16,
        qm: *const u8,
        iqm: *const u8,
        log_scale: i32,
    );
}

/// Reference `av1_highbd_quantize_fp` (no qmatrix). Returns (qcoeff, dqcoeff, eob).
pub fn ref_highbd_quantize_fp(
    log_scale: i32,
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    scan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq, mut eob) = (vec![0i32; n], vec![0i32; n], 0u16);
    let dummy = vec![0i16; n.max(2)];
    unsafe {
        av1_highbd_quantize_fp_c(
            coeff.as_ptr(),
            n as isize,
            dummy.as_ptr(),
            round.as_ptr(),
            quant.as_ptr(),
            dummy.as_ptr(),
            q.as_mut_ptr(),
            dq.as_mut_ptr(),
            dequant.as_ptr(),
            &mut eob,
            scan.as_ptr(),
            dummy.as_ptr(),
            log_scale,
        );
    }
    (q, dq, eob)
}

/// Reference `aom_highbd_quantize_b_helper_c` (no qmatrix). Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_quantize_b(
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
    let (mut q, mut dq, mut eob) = (vec![0i32; n], vec![0i32; n], 0u16);
    let dummy = vec![0i16; n.max(2)];
    unsafe {
        aom_highbd_quantize_b_helper_c(
            coeff.as_ptr(),
            n as isize,
            zbin.as_ptr(),
            round.as_ptr(),
            quant.as_ptr(),
            quant_shift.as_ptr(),
            q.as_mut_ptr(),
            dq.as_mut_ptr(),
            dequant.as_ptr(),
            &mut eob,
            scan.as_ptr(),
            dummy.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            log_scale,
        );
    }
    (q, dq, eob)
}

// FP quant-matrix path: the static helpers reached via the real facades (see
// shim/quant_fp_shim.c). round/quant/dequant are the [2]-entry QTX tables.
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_quantize_fp_qm(
        coeff: *const i32,
        n: i32,
        round: *const i16,
        quant: *const i16,
        dequant: *const i16,
        scan: *const i16,
        iscan: *const i16,
        qm: *const u8,
        iqm: *const u8,
        log_scale: i32,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
    ) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn shim_highbd_quantize_fp_qm(
        coeff: *const i32,
        n: i32,
        round: *const i16,
        quant: *const i16,
        dequant: *const i16,
        scan: *const i16,
        iscan: *const i16,
        qm: *const u8,
        iqm: *const u8,
        log_scale: i32,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
    ) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn shim_quantize_dc(
        coeff: *const i32,
        n: i32,
        round: *const i16,
        quant: i16,
        dequant: i16,
        qm: *const u8,
        iqm: *const u8,
        log_scale: i32,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
    ) -> u16;
    #[allow(clippy::too_many_arguments)]
    fn shim_highbd_quantize_dc(
        coeff: *const i32,
        n: i32,
        round: *const i16,
        quant: i16,
        dequant: i16,
        qm: *const u8,
        iqm: *const u8,
        log_scale: i32,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
    ) -> u16;
}

// Forward QM selector oracle: av1_qm_init packs pointers to the static
// wt_matrix_ref into gqmatrix[q][c][t]; the shim reads them back (see
// shim/qm_shim.c).
extern "C" {
    fn shim_qm_gqmatrix(q: i32, c: i32, t: i32, out: *mut u8, out_cap: i32) -> i32;
    fn shim_qm_giqmatrix(q: i32, c: i32, t: i32, out: *mut u8, out_cap: i32) -> i32;
    fn shim_get_qmlevel(qindex: i32, first: i32, last: i32) -> i32;
    fn shim_get_qmlevel_allintra(qindex: i32, first: i32, last: i32) -> i32;
    fn shim_get_qmlevel_luma_ssimulacra2(qindex: i32, first: i32, last: i32) -> i32;
    fn shim_get_qmlevel_444_chroma(qindex: i32, first: i32, last: i32) -> i32;
}

/// Real libaom `aom_get_qmlevel` (default-tune qindex -> QM level).
pub fn ref_get_qmlevel(qindex: i32, first: i32, last: i32) -> i32 {
    unsafe { shim_get_qmlevel(qindex, first, last) }
}

/// Real libaom `aom_get_qmlevel_allintra` (all-intra qindex -> QM level).
pub fn ref_get_qmlevel_allintra(qindex: i32, first: i32, last: i32) -> i32 {
    unsafe { shim_get_qmlevel_allintra(qindex, first, last) }
}

/// Real libaom `aom_get_qmlevel_luma_ssimulacra2` (tune=SSIMULACRA2 luma
/// qindex -> QM level).
pub fn ref_get_qmlevel_luma_ssimulacra2(qindex: i32, first: i32, last: i32) -> i32 {
    unsafe { shim_get_qmlevel_luma_ssimulacra2(qindex, first, last) }
}

/// Real libaom `aom_get_qmlevel_444_chroma` (tune=IQ/SSIMULACRA2 4:4:4 chroma
/// qindex -> QM level).
pub fn ref_get_qmlevel_444_chroma(qindex: i32, first: i32, last: i32) -> i32 {
    unsafe { shim_get_qmlevel_444_chroma(qindex, first, last) }
}

/// Real libaom forward QM matrix for (qm level `q`, plane group `c` in
/// `0=luma / 1=chroma`, raw tx size `t`), exactly as `av1_qm_init` packs
/// `gqmatrix[q][c][t]` from the file-static `wt_matrix_ref`. Returns `None` for
/// the flat level (`q == NUM_QM_LEVELS - 1`, a NULL pointer). The returned
/// bytes are read through the genuine C init pointer — the priority-1 oracle for
/// [`aom_quant::qmatrix`].
pub fn ref_qm_gqmatrix(q: usize, c: usize, t: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; 1024]; // max matrix area == 32*32
    let len = unsafe {
        shim_qm_gqmatrix(
            q as i32,
            c as i32,
            t as i32,
            out.as_mut_ptr(),
            out.len() as i32,
        )
    };
    assert!(
        len != -2,
        "shim_qm_gqmatrix: out_cap overflow for (q={q}, c={c}, t={t})"
    );
    if len < 0 {
        return None;
    }
    out.truncate(len as usize);
    Some(out)
}

/// Real libaom INVERSE QM matrix `giqmatrix[q][c][t]` (from `iwt_matrix_ref`),
/// same conventions as [`ref_qm_gqmatrix`]. The priority-1 oracle for the
/// decode-side `iqmatrix` selector the encoder reuses via `aom_decode::qm`.
pub fn ref_iqm_giqmatrix(q: usize, c: usize, t: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; 1024];
    let len = unsafe {
        shim_qm_giqmatrix(
            q as i32,
            c as i32,
            t as i32,
            out.as_mut_ptr(),
            out.len() as i32,
        )
    };
    assert!(
        len != -2,
        "shim_qm_giqmatrix: out_cap overflow for (q={q}, c={c}, t={t})"
    );
    if len < 0 {
        return None;
    }
    out.truncate(len as usize);
    Some(out)
}

/// Reference `av1_quantize_dc_facade` (`quantize_dc`, DC-only). `qm`/`iqm` are
/// `None` for the flat path. Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_quantize_dc(
    log_scale: i32,
    coeff: &[i32],
    round: &[i16; 2],
    quant: i16,
    dequant: i16,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let qp = qm.map_or(core::ptr::null(), |s| s.as_ptr());
    let iqp = iqm.map_or(core::ptr::null(), |s| s.as_ptr());
    let eob = unsafe {
        shim_quantize_dc(
            coeff.as_ptr(),
            n as i32,
            round.as_ptr(),
            quant,
            dequant,
            qp,
            iqp,
            log_scale,
            q.as_mut_ptr(),
            dq.as_mut_ptr(),
        )
    };
    (q, dq, eob)
}

/// Reference `av1_highbd_quantize_dc_facade` (`highbd_quantize_dc`, DC-only).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_quantize_dc(
    log_scale: i32,
    coeff: &[i32],
    round: &[i16; 2],
    quant: i16,
    dequant: i16,
    qm: Option<&[u8]>,
    iqm: Option<&[u8]>,
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let qp = qm.map_or(core::ptr::null(), |s| s.as_ptr());
    let iqp = iqm.map_or(core::ptr::null(), |s| s.as_ptr());
    let eob = unsafe {
        shim_highbd_quantize_dc(
            coeff.as_ptr(),
            n as i32,
            round.as_ptr(),
            quant,
            dequant,
            qp,
            iqp,
            log_scale,
            q.as_mut_ptr(),
            dq.as_mut_ptr(),
        )
    };
    (q, dq, eob)
}

/// Reference lowbd `av1_quantize_fp_facade` QM path (`quantize_fp_helper_c` with
/// non-NULL qm/iqm). Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_quantize_fp_qm(
    log_scale: i32,
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    iscan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let eob = unsafe {
        shim_quantize_fp_qm(
            coeff.as_ptr(),
            n as i32,
            round.as_ptr(),
            quant.as_ptr(),
            dequant.as_ptr(),
            scan.as_ptr(),
            iscan.as_ptr(),
            qm.as_ptr(),
            iqm.as_ptr(),
            log_scale,
            q.as_mut_ptr(),
            dq.as_mut_ptr(),
        )
    };
    (q, dq, eob)
}

/// Reference highbd `av1_highbd_quantize_fp_facade` QM path
/// (`highbd_quantize_fp_helper_c` with non-NULL qm/iqm). Returns (qcoeff, dqcoeff, eob).
#[allow(clippy::too_many_arguments)]
pub fn ref_highbd_quantize_fp_qm(
    log_scale: i32,
    coeff: &[i32],
    round: &[i16; 2],
    quant: &[i16; 2],
    dequant: &[i16; 2],
    qm: &[u8],
    iqm: &[u8],
    scan: &[i16],
    iscan: &[i16],
) -> (Vec<i32>, Vec<i32>, u16) {
    let n = coeff.len();
    let (mut q, mut dq) = (vec![0i32; n], vec![0i32; n]);
    let eob = unsafe {
        shim_highbd_quantize_fp_qm(
            coeff.as_ptr(),
            n as i32,
            round.as_ptr(),
            quant.as_ptr(),
            dequant.as_ptr(),
            scan.as_ptr(),
            iscan.as_ptr(),
            qm.as_ptr(),
            iqm.as_ptr(),
            log_scale,
            q.as_mut_ptr(),
            dq.as_mut_ptr(),
        )
    };
    (q, dq, eob)
}

// av1/common/reconintra.c — intra neighbour availability (verbatim-paste shim).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_has_top_right(
        sb_size: i32,
        bsize: i32,
        mi_row: i32,
        mi_col: i32,
        top_available: i32,
        right_available: i32,
        partition: i32,
        txsz: i32,
        row_off: i32,
        col_off: i32,
        ss_x: i32,
        ss_y: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_has_bottom_left(
        sb_size: i32,
        bsize: i32,
        mi_row: i32,
        mi_col: i32,
        bottom_available: i32,
        left_available: i32,
        partition: i32,
        txsz: i32,
        row_off: i32,
        col_off: i32,
        ss_x: i32,
        ss_y: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_intra_avail(
        sb_size: i32,
        bsize: i32,
        mi_row: i32,
        mi_col: i32,
        up_available: i32,
        left_available: i32,
        tile_col_end: i32,
        tile_row_end: i32,
        partition: i32,
        tx_size: i32,
        ss_x: i32,
        ss_y: i32,
        row_off: i32,
        col_off: i32,
        wpx: i32,
        hpx: i32,
        mi_cols: i32,
        mi_rows: i32,
        mode: i32,
        angle_delta: i32,
        use_filter_intra: i32,
        out: *mut i32,
    );
}

/// Reference `has_top_right` (reconintra.c): is the block's top-right reference
/// available and coded? Returns 0/1.
#[allow(clippy::too_many_arguments)]
pub fn ref_has_top_right(
    sb_size: usize,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    top_available: bool,
    right_available: bool,
    partition: usize,
    txsz: usize,
    row_off: i32,
    col_off: i32,
    ss_x: i32,
    ss_y: i32,
) -> i32 {
    unsafe {
        shim_has_top_right(
            sb_size as i32,
            bsize as i32,
            mi_row,
            mi_col,
            top_available as i32,
            right_available as i32,
            partition as i32,
            txsz as i32,
            row_off,
            col_off,
            ss_x,
            ss_y,
        )
    }
}

/// Reference `has_bottom_left` (reconintra.c): is the block's bottom-left
/// reference available and coded? Returns 0/1.
#[allow(clippy::too_many_arguments)]
pub fn ref_has_bottom_left(
    sb_size: usize,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    bottom_available: bool,
    left_available: bool,
    partition: usize,
    txsz: usize,
    row_off: i32,
    col_off: i32,
    ss_x: i32,
    ss_y: i32,
) -> i32 {
    unsafe {
        shim_has_bottom_left(
            sb_size as i32,
            bsize as i32,
            mi_row,
            mi_col,
            bottom_available as i32,
            left_available as i32,
            partition as i32,
            txsz as i32,
            row_off,
            col_off,
            ss_x,
            ss_y,
        )
    }
}

/// Reference intra neighbour-availability composition (the counts computed inside
/// `av1_predict_intra_block`). Returns `(n_top_px, n_topright_px, n_left_px,
/// n_bottomleft_px)`.
#[allow(clippy::too_many_arguments)]
pub fn ref_intra_avail(
    sb_size: usize,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    up_available: bool,
    left_available: bool,
    tile_col_end: i32,
    tile_row_end: i32,
    partition: usize,
    tx_size: usize,
    ss_x: i32,
    ss_y: i32,
    row_off: i32,
    col_off: i32,
    wpx: i32,
    hpx: i32,
    mi_cols: i32,
    mi_rows: i32,
    mode: usize,
    angle_delta: i32,
    use_filter_intra: bool,
) -> (i32, i32, i32, i32) {
    let mut out = [0i32; 4];
    unsafe {
        shim_intra_avail(
            sb_size as i32,
            bsize as i32,
            mi_row,
            mi_col,
            up_available as i32,
            left_available as i32,
            tile_col_end,
            tile_row_end,
            partition as i32,
            tx_size as i32,
            ss_x,
            ss_y,
            row_off,
            col_off,
            wpx,
            hpx,
            mi_cols,
            mi_rows,
            mode as i32,
            angle_delta,
            use_filter_intra as i32,
            out.as_mut_ptr(),
        );
    }
    (out[0], out[1], out[2], out[3])
}

// av1/common/cfl.c — the exported per-size CfL `_c` kernels, reached through the
// exported `_c` getter tables (one getter per subsample family + subtract-average +
// hbd predict). Plain externs into libaom.a; the kernels are pure loops (no RTCD).
// TX_SIZE parameters are `uint8_t` in C (UENUM1BYTE) — declared `u8` here.
type CflSubsampleHbdFn =
    unsafe extern "C" fn(input: *const u16, input_stride: i32, output_q3: *mut u16);
type CflSubtractAverageFn = unsafe extern "C" fn(src: *const u16, dst: *mut i16);
type CflPredictHbdFn =
    unsafe extern "C" fn(src: *const i16, dst: *mut u16, dst_stride: i32, alpha_q3: i32, bd: i32);
extern "C" {
    fn cfl_get_luma_subsampling_420_hbd_c(tx_size: u8) -> Option<CflSubsampleHbdFn>;
    fn cfl_get_luma_subsampling_422_hbd_c(tx_size: u8) -> Option<CflSubsampleHbdFn>;
    fn cfl_get_luma_subsampling_444_hbd_c(tx_size: u8) -> Option<CflSubsampleHbdFn>;
    fn cfl_get_subtract_average_fn_c(tx_size: u8) -> Option<CflSubtractAverageFn>;
    fn cfl_get_predict_hbd_fn_c(tx_size: u8) -> Option<CflPredictHbdFn>;
}

/// Reference `cfl_luma_subsampling_{420,422,444}_hbd_c` for the LUMA `tx_size`
/// (dims ≤ 32×32): subsample `input` (strided luma, u16) into the Q3
/// `CFL_BUF_LINE`(32)-strided `out` buffer. `ss = (ss_x, ss_y)` selects the
/// family: (1,1)=420, (1,0)=422, (0,0)=444.
pub fn ref_cfl_subsample_hbd(
    ss: (i32, i32),
    tx_size: usize,
    input: &[u16],
    input_stride: usize,
    out: &mut [u16; 1024],
) {
    let f = unsafe {
        match ss {
            (1, 1) => cfl_get_luma_subsampling_420_hbd_c(tx_size as u8),
            (1, 0) => cfl_get_luma_subsampling_422_hbd_c(tx_size as u8),
            (0, 0) => cfl_get_luma_subsampling_444_hbd_c(tx_size as u8),
            _ => panic!("invalid subsampling {ss:?}"),
        }
    }
    .expect("no C cfl subsample kernel for this tx_size");
    unsafe { f(input.as_ptr(), input_stride as i32, out.as_mut_ptr()) }
}

/// Reference `cfl_subtract_average_WxH_c` for the CHROMA `tx_size`: zero-mean
/// the Q3 surface (`src`) into `dst`; both `CFL_BUF_LINE`-strided.
pub fn ref_cfl_subtract_average(tx_size: usize, src: &[u16; 1024], dst: &mut [i16; 1024]) {
    let f = unsafe { cfl_get_subtract_average_fn_c(tx_size as u8) }
        .expect("no C cfl subtract-average kernel for this tx_size");
    unsafe { f(src.as_ptr(), dst.as_mut_ptr()) }
}

/// Reference `cfl_predict_hbd_WxH_c` for the CHROMA `tx_size`: `dst` holds the
/// DC prediction on entry and receives `clip(dst + scaled_luma(alpha_q3, ac))`.
pub fn ref_cfl_predict_hbd(
    tx_size: usize,
    ac: &[i16; 1024],
    dst: &mut [u16],
    dst_stride: usize,
    alpha_q3: i32,
    bd: i32,
) {
    let f = unsafe { cfl_get_predict_hbd_fn_c(tx_size as u8) }
        .expect("no C cfl predict kernel for this tx_size");
    unsafe {
        f(
            ac.as_ptr(),
            dst.as_mut_ptr(),
            dst_stride as i32,
            alpha_q3,
            bd,
        )
    }
}

// dec_shim.c — decoder-track MACROBLOCKD facades over real static inlines
// (pred_common.h / av1_common_int.h / blockd.h; scale_chroma_bsize is the one
// verbatim transcription), the default KF FRAME_CONTEXT dump via the REAL
// av1_setup_past_independence, and the REAL public codec API (av1_cx/av1_dx).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_get_tx_size_context(
        bsize: i32,
        above_txfm: u8,
        left_txfm: u8,
        up_available: i32,
        left_available: i32,
        above_bsize: i32,
        above_inter: i32,
        left_bsize: i32,
        left_inter: i32,
    ) -> i32;
    fn shim_set_txfm_ctxs(
        tx_size: i32,
        n4_w: i32,
        n4_h: i32,
        skip: i32,
        above: *mut u8,
        left: *mut u8,
    );
    fn shim_is_chroma_reference(mi_row: i32, mi_col: i32, bsize: i32, ss_x: i32, ss_y: i32) -> i32;
    fn shim_get_max_uv_txsize(bsize: i32, ss_x: i32, ss_y: i32) -> i32;
    fn shim_spatial_seg_pred(
        up_available: i32,
        left_available: i32,
        ul: i32,
        u: i32,
        l: i32,
    ) -> i32;
    fn shim_intra_mode_to_tx_type(y_mode: i32, uv_mode: i32, plane_type: i32) -> i32;
    fn shim_av1_get_tx_type_uv_intra(
        y_mode: i32,
        uv_mode: i32,
        uv_tx_size: i32,
        reduced_tx_set: i32,
        lossless: i32,
    ) -> i32;
    fn shim_tx_size_from_tx_mode(bsize: i32, tx_mode: i32) -> i32;
    fn shim_depth_to_tx_size(depth: i32, bsize: i32) -> i32;
    fn shim_scale_chroma_bsize(bsize: i32, ss_x: i32, ss_y: i32) -> i32;
    fn shim_dump_default_kf_fc(base_qindex: i32, out: *mut u16) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        aq_mode: i32,
        two_pass: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
    /// min-q/max-q variant of `shim_encode_av1_kf` (append-only): base config
    /// identical to `shim_encode_av1_kf` (single pass, aq 0, cdef/restoration
    /// off, sb64, single tile, palette/intrabc off, QM off), plus the qindex
    /// clamp bounds `--min-q = min_q` / `--max-q = max_q` (both 0..63).
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_minmaxq(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        usage: i32,
        min_q: i32,
        max_q: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
    /// Generic-controls variant of `shim_encode_av1_kf` (the C8-C11
    /// toggle-sweep infrastructure, append-only): base config identical to
    /// `shim_encode_av1_kf` (single pass, aq 0, cdef/restoration off, sb64,
    /// single tile, palette/intrabc off), plus `n_ctrls` raw
    /// `(aome_enc_control_id, value)` pairs applied after the base set.
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_ctrls(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        usage: i32,
        ctrl_ids: *const i32,
        ctrl_vals: *const i32,
        n_ctrls: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
    /// Ctrl-id cross-check for [`cx_ctrl`] (see dec_shim.c): the REAL
    /// `aome_enc_control_id` value for a fixed probe index.
    fn shim_cx_ctrl_id_by_probe(probe: i32) -> i32;
    /// disable_cdf_update variant of `shim_encode_av1_kf` (decoder-track
    /// disable_cdf_update gate, append-only addition — every function above is
    /// untouched): single-pass KEY encode with `AV1E_SET_CDF_UPDATE_MODE=0`
    /// (`--cdf-update-mode=0`), which forces `disable_cdf_update=1` in the
    /// emitted uncompressed header for every frame.
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_disable_cdf(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
    /// SB128 variant of `shim_encode_av1_kf` (decoder-track SB128 work,
    /// append-only addition — `shim_encode_av1_kf` above is untouched):
    /// same params plus explicit `sb_size_128` (0 = --sb-size=64, nonzero =
    /// --sb-size=128).
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_sb128(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        aq_mode: i32,
        two_pass: i32,
        sb_size_128: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
    /// Multi-tile variant of `shim_encode_av1_kf` (decoder-track multi-tile
    /// work, append-only addition — `shim_encode_av1_kf` /
    /// `shim_encode_av1_kf_sb128` above are untouched): same params as the
    /// SB128 variant plus explicit `tile_columns_log2`/`tile_rows_log2`
    /// (`AV1E_SET_TILE_COLUMNS`/`_ROWS` — the CODED value IS the log2 tile
    /// count: 0 = 1 column/row, 1 = 2, 2 = 4, ...).
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_tiles(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        aq_mode: i32,
        two_pass: i32,
        sb_size_128: i32,
        tile_columns_log2: i32,
        tile_rows_log2: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn shim_decode_av1_kf(
        data: *const u8,
        len: usize,
        expect_w: i32,
        expect_h: i32,
        y: *mut u16,
        u: *mut u16,
        v: *mut u16,
        info_out: *mut i32,
    ) -> i32;
    fn shim_lr_units_roundtrip(
        units: *const i32,
        n: i32,
        out: *mut u8,
        out_cap: i64,
        readback: *mut i32,
        cdf_out: *mut u16,
    ) -> i64;
    /// EXPORTED `aom_count_primitive_refsubexpfin` (aom_dsp/binary_codes_writer.c).
    fn aom_count_primitive_refsubexpfin(n: u16, k: u16, r: u16, v: u16) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_lr_corners_in_sb(
        w: i32,
        h: i32,
        ss_x: i32,
        ss_y: i32,
        unit_size: *const i32,
        plane: i32,
        mi_row: i32,
        mi_col: i32,
        bsize: i32,
        out: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_wiener_convolve(
        src: *const u16,
        dst: *mut u16,
        buf_w: i32,
        buf_h: i32,
        off_x: i32,
        off_y: i32,
        w: i32,
        h: i32,
        hf: *const i16,
        vf: *const i16,
        bd: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_apply_sgr(
        src: *const u16,
        dst: *mut u16,
        buf_w: i32,
        buf_h: i32,
        off_x: i32,
        off_y: i32,
        w: i32,
        h: i32,
        ep: i32,
        xqd0: i32,
        xqd1: i32,
        bd: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_lr_filter_frame(
        y: *mut u16,
        u: *mut u16,
        v: *mut u16,
        dy: *const u16,
        du: *const u16,
        dv: *const u16,
        w: i32,
        h: i32,
        y_stride: i32,
        uv_stride: i32,
        num_planes: i32,
        ss_x: i32,
        ss_y: i32,
        bd: i32,
        optimized: i32,
        frame_rtype: *const i32,
        unit_size: *const i32,
        units0: *const i32,
        units1: *const i32,
        units2: *const i32,
    ) -> i32;
}

/// Words per unit in [`ref_lr_filter_frame`] packing:
/// `[rtype, v0, v1, v2, h0, h1, h2, ep, xqd0, xqd1]`.
pub const LRF_WORDS: usize = 10;

/// The REAL whole-frame loop restoration (real `av1_loop_restoration_save_
/// boundary_lines` both passes in the decoder's ordering + real
/// `av1_loop_restoration_filter_frame` over bordered YV12 buffers; real
/// `av1_alloc_restoration_struct/_buffers` geometry). Filters `y/u/v` in
/// place; `dy/du/dv` are the deblocked (pre-CDEF) planes (unused when
/// `optimized`). bd 8 runs the production lowbd u8 frame. Call
/// [`ref_init`] first (RTCD kernels).
#[allow(clippy::too_many_arguments)]
pub fn ref_lr_filter_frame(
    y: &mut [u16],
    u: &mut [u16],
    v: &mut [u16],
    dy: &[u16],
    du: &[u16],
    dv: &[u16],
    w: usize,
    h: usize,
    y_stride: usize,
    uv_stride: usize,
    num_planes: usize,
    ss_x: usize,
    ss_y: usize,
    bd: i32,
    optimized: bool,
    frame_rtype: [i32; 3],
    unit_size: [i32; 3],
    units: [&[i32]; 3],
) {
    for (p, us) in units.iter().enumerate() {
        assert!(us.len().is_multiple_of(LRF_WORDS) || frame_rtype[p] == 0);
    }
    let rc = unsafe {
        shim_lr_filter_frame(
            y.as_mut_ptr(),
            u.as_mut_ptr(),
            v.as_mut_ptr(),
            dy.as_ptr(),
            du.as_ptr(),
            dv.as_ptr(),
            w as i32,
            h as i32,
            y_stride as i32,
            uv_stride as i32,
            num_planes as i32,
            ss_x as i32,
            ss_y as i32,
            bd,
            optimized as i32,
            frame_rtype.as_ptr(),
            unit_size.as_ptr(),
            units[0].as_ptr(),
            units[1].as_ptr(),
            units[2].as_ptr(),
        )
    };
    assert_eq!(rc, 0, "shim_lr_filter_frame failed ({rc})");
}

/// REAL `av1_wiener_convolve_add_src_c` (bd 8, lowbd u8 path) /
/// `av1_highbd_wiener_convolve_add_src_c` (bd > 8) over a padded
/// `buf_w x buf_h` u16 buffer: filters the `w x h` block at `(off_x, off_y)`
/// in place on `dst`.
#[allow(clippy::too_many_arguments)]
pub fn ref_wiener_convolve(
    src: &[u16],
    dst: &mut [u16],
    buf_w: usize,
    buf_h: usize,
    off_x: usize,
    off_y: usize,
    w: usize,
    h: usize,
    hfilter: &[i16; 8],
    vfilter: &[i16; 8],
    bd: i32,
) {
    assert_eq!(src.len(), buf_w * buf_h);
    assert_eq!(dst.len(), buf_w * buf_h);
    let rc = unsafe {
        shim_wiener_convolve(
            src.as_ptr(),
            dst.as_mut_ptr(),
            buf_w as i32,
            buf_h as i32,
            off_x as i32,
            off_y as i32,
            w as i32,
            h as i32,
            hfilter.as_ptr(),
            vfilter.as_ptr(),
            bd,
        )
    };
    assert_eq!(rc, 0, "shim_wiener_convolve failed");
}

/// REAL `av1_apply_selfguided_restoration_c` over the same buffer convention.
#[allow(clippy::too_many_arguments)]
pub fn ref_apply_sgr(
    src: &[u16],
    dst: &mut [u16],
    buf_w: usize,
    buf_h: usize,
    off_x: usize,
    off_y: usize,
    w: usize,
    h: usize,
    ep: usize,
    xqd: [i32; 2],
    bd: i32,
) {
    assert_eq!(src.len(), buf_w * buf_h);
    assert_eq!(dst.len(), buf_w * buf_h);
    let rc = unsafe {
        shim_apply_sgr(
            src.as_ptr(),
            dst.as_mut_ptr(),
            buf_w as i32,
            buf_h as i32,
            off_x as i32,
            off_y as i32,
            w as i32,
            h as i32,
            ep as i32,
            xqd[0],
            xqd[1],
            bd,
        )
    };
    assert_eq!(rc, 0, "shim_apply_sgr failed ({rc})");
}

/// REAL `av1_alloc_restoration_struct` geometry + REAL
/// `av1_loop_restoration_corners_in_sb` for one (plane, superblock). Returns
/// `(hit, horz_units, vert_units, [rcol0, rcol1, rrow0, rrow1])`.
#[allow(clippy::too_many_arguments)]
pub fn ref_lr_corners_in_sb(
    w: i32,
    h: i32,
    ss_x: i32,
    ss_y: i32,
    unit_size: [i32; 3],
    plane: usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> (bool, i32, i32, [i32; 4]) {
    let mut out = [0i32; 6];
    let hit = unsafe {
        shim_lr_corners_in_sb(
            w,
            h,
            ss_x,
            ss_y,
            unit_size.as_ptr(),
            plane as i32,
            mi_row,
            mi_col,
            bsize as i32,
            out.as_mut_ptr(),
        )
    };
    assert!(hit >= 0, "shim_lr_corners_in_sb failed");
    (hit != 0, out[0], out[1], [out[2], out[3], out[4], out[5]])
}

/// Words per unit in [`ref_lr_units_roundtrip`] packing:
/// `[plane, frame_rtype, unit_rtype, v0, v1, v2, h0, h1, h2, ep, xqd0, xqd1]`.
pub const LRU_WORDS: usize = 12;

/// Write a sequence of restoration-unit parameter sets with the REAL C
/// arithmetic writer (transcribed encoder `loop_restoration_write_sb_coeffs`
/// control flow over the EXPORTED `aom_write_primitive_refsubexpfin` +
/// `aom_write_symbol` on the REAL default LR CDFs), then read them back with
/// the REAL C reader (transcribed decoder control flow over the EXPORTED
/// `aom_read_primitive_refsubexpfin` + `aom_read_symbol`). Returns
/// `(bitstream, readback_units, final_cdfs)` where `final_cdfs` is
/// `switchable[4] ++ wiener[3] ++ sgrproj[3]` after reader adaptation.
pub fn ref_lr_units_roundtrip(units: &[i32]) -> (Vec<u8>, Vec<i32>, [u16; 10]) {
    assert!(units.len().is_multiple_of(LRU_WORDS));
    let n = units.len() / LRU_WORDS;
    let mut out = vec![0u8; 1 << 20];
    let mut readback = vec![0i32; units.len()];
    let mut cdfs = [0u16; 10];
    let len = unsafe {
        shim_lr_units_roundtrip(
            units.as_ptr(),
            n as i32,
            out.as_mut_ptr(),
            out.len() as i64,
            readback.as_mut_ptr(),
            cdfs.as_mut_ptr(),
        )
    };
    assert!(len > 0, "shim_lr_units_roundtrip failed ({len})");
    out.truncate(len as usize);
    (out, readback, cdfs)
}

/// Reference `get_tx_size_context` (pred_common.h) over a constructed
/// MACROBLOCKD (neighbour txfm-context bytes + availability + neighbour
/// bsize/inter-ness).
#[allow(clippy::too_many_arguments)]
pub fn ref_get_tx_size_context(
    bsize: usize,
    above_txfm: u8,
    left_txfm: u8,
    up_available: bool,
    left_available: bool,
    above_bsize: usize,
    above_inter: bool,
    left_bsize: usize,
    left_inter: bool,
) -> i32 {
    unsafe {
        shim_get_tx_size_context(
            bsize as i32,
            above_txfm,
            left_txfm,
            up_available as i32,
            left_available as i32,
            above_bsize as i32,
            above_inter as i32,
            left_bsize as i32,
            left_inter as i32,
        )
    }
}

/// Reference `set_txfm_ctxs` (av1_common_int.h): stamps `above[..n4_w]` /
/// `left[..n4_h]`.
pub fn ref_set_txfm_ctxs(
    tx_size: usize,
    n4_w: usize,
    n4_h: usize,
    skip: bool,
    above: &mut [u8],
    left: &mut [u8],
) {
    assert!(above.len() >= n4_w && left.len() >= n4_h);
    unsafe {
        shim_set_txfm_ctxs(
            tx_size as i32,
            n4_w as i32,
            n4_h as i32,
            skip as i32,
            above.as_mut_ptr(),
            left.as_mut_ptr(),
        )
    }
}

/// Reference `is_chroma_reference` (av1_common_int.h).
pub fn ref_is_chroma_reference(
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    ss_x: i32,
    ss_y: i32,
) -> bool {
    unsafe { shim_is_chroma_reference(mi_row, mi_col, bsize as i32, ss_x, ss_y) != 0 }
}

/// Reference `av1_get_max_uv_txsize` (blockd.h). Only valid `(bsize, ss)`
/// combinations (the C asserts the plane bsize is real).
pub fn ref_get_max_uv_txsize(bsize: usize, ss_x: i32, ss_y: i32) -> usize {
    unsafe { shim_get_max_uv_txsize(bsize as i32, ss_x, ss_y) as usize }
}

/// Reference `av1_get_spatial_seg_pred` (pred_common.h, `skip_over4x4 = 0`):
/// the (pred, cdf_num) of a block whose up-left/up/left neighbour segment ids
/// are `ul`/`u`/`l` (each `< MAX_SEGMENTS`) with the given availability.
pub fn ref_spatial_seg_pred(
    up_available: bool,
    left_available: bool,
    ul: u8,
    u: u8,
    l: u8,
) -> (i32, usize) {
    let r = unsafe {
        shim_spatial_seg_pred(
            up_available as i32,
            left_available as i32,
            ul as i32,
            u as i32,
            l as i32,
        )
    };
    (r & 0xff, (r >> 8) as usize)
}

/// Reference `intra_mode_to_tx_type` (blockd.h).
pub fn ref_intra_mode_to_tx_type(y_mode: usize, uv_mode: usize, plane_type: usize) -> usize {
    unsafe { shim_intra_mode_to_tx_type(y_mode as i32, uv_mode as i32, plane_type as i32) as usize }
}

/// Reference `av1_get_tx_type` (blockd.h), intra UV arm.
pub fn ref_av1_get_tx_type_uv_intra(
    y_mode: usize,
    uv_mode: usize,
    uv_tx_size: usize,
    reduced_tx_set: bool,
    lossless: bool,
) -> usize {
    unsafe {
        shim_av1_get_tx_type_uv_intra(
            y_mode as i32,
            uv_mode as i32,
            uv_tx_size as i32,
            reduced_tx_set as i32,
            lossless as i32,
        ) as usize
    }
}

/// Reference `tx_size_from_tx_mode` (blockd.h).
pub fn ref_tx_size_from_tx_mode(bsize: usize, tx_mode: i32) -> usize {
    unsafe { shim_tx_size_from_tx_mode(bsize as i32, tx_mode) as usize }
}

/// Reference `depth_to_tx_size` (blockd.h).
pub fn ref_depth_to_tx_size(depth: i32, bsize: usize) -> usize {
    unsafe { shim_depth_to_tx_size(depth, bsize as i32) as usize }
}

/// Reference `scale_chroma_bsize` (reconintra.c; verbatim transcription in
/// dec_shim.c — the fn is static in a .c file).
pub fn ref_scale_chroma_bsize(bsize: usize, ss_x: i32, ss_y: i32) -> usize {
    unsafe { shim_scale_chroma_bsize(bsize as i32, ss_x, ss_y) as usize }
}

/// Flat u16 length of the default-KF-FRAME_CONTEXT dump (see dec_shim.c for
/// the field order; the coefficient arena is the trailing 4045).
pub const DUMP_KF_FC_LEN: usize = 7061;

/// Dump the REAL `av1_setup_past_independence` default KF FRAME_CONTEXT for a
/// `base_qindex` as a flat u16 buffer mirroring `KfFrameContext`'s field order.
pub fn ref_dump_default_kf_fc(base_qindex: i32) -> Vec<u16> {
    let mut out = vec![0u16; DUMP_KF_FC_LEN];
    let rc = unsafe { shim_dump_default_kf_fc(base_qindex, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "shim_dump_default_kf_fc failed ({rc})");
    out
}

/// Encode one KEY frame through the REAL `aom_codec_av1_cx` public API (the
/// path the aomenc CLI drives) with `--usage=<usage> --cpu-used=<cpu_used>
/// --end-usage=q --cq-level=<cq_level> --enable-cdef=<enable_cdef>
/// --enable-restoration=<enable_restoration> --sb-size=64 --deltaq-mode=0
/// --aq-mode=<aq_mode> --enable-palette=0 --enable-intrabc=0`. `usage` 0 =
/// GOOD, 2 = ALL_INTRA (the zenavif/avifenc still-image mode). `aq_mode` 1
/// (variance) / 2 (complexity) enable SEGMENTATION on intra frames — 8
/// `SEG_LVL_ALT_Q` segments (av1_vaq_frame_setup / av1_setup_in_frame_q_adj)
/// — but ONLY with `two_pass` (the full firstpass-stats + last-pass
/// sequence): a one-pass encode takes `encode_without_recode` and never
/// runs the aq segmentation setup.
/// Planes are u16 at every bit depth; chroma dims are `(w+ss)>>ss`.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
    aq_mode: u32,
    two_pass: bool,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            aq_mode as i32,
            two_pass as i32,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf failed ({n})");
    out.truncate(n as usize);
    out
}

/// Encode one KEY frame through the REAL `aom_codec_av1_cx` API exactly like
/// [`ref_encode_av1_kf`] (single pass, cdef/restoration off, aq 0, sb64, single
/// tile, QM off) but with the `--min-q = min_q` / `--max-q = max_q` qindex clamp
/// bounds (both 0..63 quantizer levels). Used to validate the port's
/// `base_qindex_from_cq_clamped` derivation against the real encoder's header.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_minmaxq(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    usage: u32,
    min_q: i32,
    max_q: i32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_minmaxq(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            usage as i32,
            min_q,
            max_q,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_minmaxq failed ({n})");
    out.truncate(n as usize);
    out
}

/// Encoder control IDs (`enum aome_enc_control_id`, `aom/aomcx.h`, libaom
/// v3.14.1) for [`ref_encode_av1_kf_ctrls`] — the C8-C11 toggle-sweep knob
/// set. The enum is a stable public ABI (entries are never renumbered);
/// `cx_ctrl_ids_match_reference_headers` (tests) asserts every constant
/// against the pinned reference checkout via `shim_cx_ctrl_id_by_probe`.
pub mod cx_ctrl {
    /// `--cdf-update-mode` (0 = no CDF update for any frame).
    pub const AV1E_SET_CDF_UPDATE_MODE: i32 = 44;
    /// `--enable-rect-partitions` (default 1).
    pub const AV1E_SET_ENABLE_RECT_PARTITIONS: i32 = 73;
    /// `--enable-ab-partitions` (default 1).
    pub const AV1E_SET_ENABLE_AB_PARTITIONS: i32 = 74;
    /// `--enable-1to4-partitions` (default 1).
    pub const AV1E_SET_ENABLE_1TO4_PARTITIONS: i32 = 75;
    /// `--min-partition-size` in PIXELS, one of {4,8,16,32,64,128} (default 4).
    pub const AV1E_SET_MIN_PARTITION_SIZE: i32 = 76;
    /// `--max-partition-size` in PIXELS, one of {4,8,16,32,64,128} (default 128).
    pub const AV1E_SET_MAX_PARTITION_SIZE: i32 = 77;
    /// `--enable-intra-edge-filter` (default 1; a SEQUENCE-header bit).
    pub const AV1E_SET_ENABLE_INTRA_EDGE_FILTER: i32 = 78;
    /// `--enable-tx64` (default 1).
    pub const AV1E_SET_ENABLE_TX64: i32 = 80;
    /// `--enable-flip-idtx` (default 1).
    pub const AV1E_SET_ENABLE_FLIP_IDTX: i32 = 81;
    /// `--enable-rect-tx` (default 1).
    pub const AV1E_SET_ENABLE_RECT_TX: i32 = 82;
    /// `--enable-filter-intra` (default 1; a SEQUENCE-header bit).
    pub const AV1E_SET_ENABLE_FILTER_INTRA: i32 = 98;
    /// `--enable-smooth-intra` (default 1).
    pub const AV1E_SET_ENABLE_SMOOTH_INTRA: i32 = 99;
    /// `--enable-paeth-intra` (default 1).
    pub const AV1E_SET_ENABLE_PAETH_INTRA: i32 = 100;
    /// `--enable-cfl-intra` (default 1).
    pub const AV1E_SET_ENABLE_CFL_INTRA: i32 = 101;
    /// `--enable-angle-delta` (default 1).
    pub const AV1E_SET_ENABLE_ANGLE_DELTA: i32 = 106;
    /// `--reduced-tx-type-set` (default 0; a FRAME-header bit).
    pub const AV1E_SET_REDUCED_TX_TYPE_SET: i32 = 118;
    /// `--use-intra-dct-only` (default 0).
    pub const AV1E_SET_INTRA_DCT_ONLY: i32 = 119;
    /// `--use-intra-default-tx-only` (default 0).
    pub const AV1E_SET_INTRA_DEFAULT_TX_ONLY: i32 = 121;
    /// `--enable-diagonal-intra` (default 1).
    pub const AV1E_SET_ENABLE_DIAGONAL_INTRA: i32 = 141;
    /// `--enable-directional-intra` (default 1).
    pub const AV1E_SET_ENABLE_DIRECTIONAL_INTRA: i32 = 145;
    /// `--enable-tx-size-search` (default 1; 0 forces the largest tx size
    /// per block -> frame header `tx_mode = TX_MODE_LARGEST`).
    pub const AV1E_SET_ENABLE_TX_SIZE_SEARCH: i32 = 146;
    /// `--disable-trellis-quant` (default 3 = NO_ESTIMATE_YRD_TRELLIS_OPT;
    /// 0 = FULL, 1 = NO, 2 = FINAL_PASS — init_rd_sf, speed_features.c:2479).
    pub const AV1E_SET_DISABLE_TRELLIS_QUANT: i32 = 62;
    /// `--coeff-cost-upd-freq` (default 0 = COST_UPD_SB; 1 SBROW / 2 TILE /
    /// 3 OFF). NOTE: C skips ALL cost updates when disable_cdf_update
    /// (av1_set_cost_upd_freq's early return).
    pub const AV1E_SET_COEFF_COST_UPD_FREQ: i32 = 126;
    /// `--mode-cost-upd-freq` (same value space as coeff).
    pub const AV1E_SET_MODE_COST_UPD_FREQ: i32 = 127;
    /// `--dv-cost-upd-freq` — DV costs are intrabc-only: INERT on this
    /// envelope (intrabc off); the ctrl exists for completeness.
    pub const AV1E_SET_DV_COST_UPD_FREQ: i32 = 142;

    /// `--sb-size=64|128` (`AV1E_SET_SUPERBLOCK_SIZE`, aomcx.h:664, verified
    /// value 56). The argument is an `aom_superblock_size_t`
    /// (aom_codec.h:347-350): `AOM_SUPERBLOCK_SIZE_64X64 = 0`,
    /// [`AOM_SUPERBLOCK_SIZE_128X128`]` = 1`, `AOM_SUPERBLOCK_SIZE_DYNAMIC = 2`.
    /// Applied through the generic `ref_encode_av1_kf_ctrls` path (the shim
    /// runs any `(id, val)` pair via `aom_codec_control`); not in
    /// [`PROBE_TABLE`] (the constant is directly header-verified, and the
    /// existing `shim_cx_ctrl_id_by_probe` probe order is append-only).
    pub const AV1E_SET_SUPERBLOCK_SIZE: i32 = 56;
    /// `aom_superblock_size_t` value for `--sb-size=128` (aom_codec.h:349).
    pub const AOM_SUPERBLOCK_SIZE_128X128: i32 = 1;

    /// `(probe_index, constant)` table for the header cross-check test —
    /// probe order matches `shim_cx_ctrl_id_by_probe` (dec_shim.c).
    pub const PROBE_TABLE: [(i32, i32); 25] = [
        (0, AV1E_SET_CDF_UPDATE_MODE),
        (1, AV1E_SET_ENABLE_RECT_PARTITIONS),
        (2, AV1E_SET_ENABLE_AB_PARTITIONS),
        (3, AV1E_SET_ENABLE_1TO4_PARTITIONS),
        (4, AV1E_SET_MIN_PARTITION_SIZE),
        (5, AV1E_SET_MAX_PARTITION_SIZE),
        (6, AV1E_SET_ENABLE_INTRA_EDGE_FILTER),
        (7, AV1E_SET_ENABLE_TX64),
        (8, AV1E_SET_ENABLE_FLIP_IDTX),
        (9, AV1E_SET_ENABLE_RECT_TX),
        (10, AV1E_SET_ENABLE_FILTER_INTRA),
        (11, AV1E_SET_ENABLE_SMOOTH_INTRA),
        (12, AV1E_SET_ENABLE_PAETH_INTRA),
        (13, AV1E_SET_ENABLE_CFL_INTRA),
        (14, AV1E_SET_ENABLE_ANGLE_DELTA),
        (15, AV1E_SET_REDUCED_TX_TYPE_SET),
        (16, AV1E_SET_INTRA_DCT_ONLY),
        (17, AV1E_SET_INTRA_DEFAULT_TX_ONLY),
        (18, AV1E_SET_ENABLE_DIAGONAL_INTRA),
        (19, AV1E_SET_ENABLE_DIRECTIONAL_INTRA),
        (20, AV1E_SET_ENABLE_TX_SIZE_SEARCH),
        (21, AV1E_SET_DISABLE_TRELLIS_QUANT),
        (22, AV1E_SET_COEFF_COST_UPD_FREQ),
        (23, AV1E_SET_MODE_COST_UPD_FREQ),
        (24, AV1E_SET_DV_COST_UPD_FREQ),
    ];
}

/// The REAL `aome_enc_control_id` value for a probe index (the
/// [`cx_ctrl::PROBE_TABLE`] cross-check; see `shim_cx_ctrl_id_by_probe`).
pub fn ref_cx_ctrl_id_by_probe(probe: i32) -> i32 {
    unsafe { shim_cx_ctrl_id_by_probe(probe) }
}

/// Generic-controls variant of [`ref_encode_av1_kf`] (the C8-C11
/// toggle-sweep infrastructure, append-only — `ref_encode_av1_kf` above is
/// untouched): base config identical to `ref_encode_av1_kf` with
/// `enable_cdef=false, enable_restoration=false, aq_mode=0, two_pass=false`
/// (the stock stills envelope), plus `ctrls` — raw
/// `(aome_enc_control_id, value)` pairs ([`cx_ctrl`]) applied through
/// `aom_codec_control` after the base set, in order.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_ctrls(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    usage: u32,
    ctrls: &[(i32, i32)],
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let ids: Vec<i32> = ctrls.iter().map(|&(id, _)| id).collect();
    let vals: Vec<i32> = ctrls.iter().map(|&(_, v)| v).collect();
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_ctrls(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            usage as i32,
            ids.as_ptr(),
            vals.as_ptr(),
            ids.len() as i32,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_ctrls failed ({n})");
    out.truncate(n as usize);
    out
}

/// disable_cdf_update variant of [`ref_encode_av1_kf`] (decoder-track
/// disable_cdf_update gate, append-only addition — `ref_encode_av1_kf` above is
/// untouched): a single-pass GOOD/ALLINTRA KEY encode with
/// `AV1E_SET_CDF_UPDATE_MODE=0` (`--cdf-update-mode=0`). That control forces
/// `cm->features.disable_cdf_update = 1` for every frame
/// (`av1/encoder/encoder.c` `case 0:`), so the emitted shown-KEY uncompressed
/// header carries `disable_cdf_update = 1`. Returns the real libaom bitstream.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_disable_cdf(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_disable_cdf(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_disable_cdf failed ({n})");
    out.truncate(n as usize);
    out
}

/// SB128 variant of [`ref_encode_av1_kf`] (decoder-track SB128 work,
/// append-only addition — `ref_encode_av1_kf` above is untouched): same
/// params plus `sb_size_128` (`false` = `--sb-size=64`, `true` =
/// `--sb-size=128`, `AV1E_SET_SUPERBLOCK_SIZE` /
/// `AOM_SUPERBLOCK_SIZE_128X128`).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_sb128(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
    aq_mode: u32,
    two_pass: bool,
    sb_size_128: bool,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_sb128(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            aq_mode as i32,
            two_pass as i32,
            sb_size_128 as i32,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_sb128 failed ({n})");
    out.truncate(n as usize);
    out
}

/// Multi-tile variant of [`ref_encode_av1_kf`] (decoder-track multi-tile
/// work, append-only addition — `ref_encode_av1_kf` / `ref_encode_av1_kf_sb128`
/// above are untouched): same params as the SB128 variant plus
/// `tile_columns_log2`/`tile_rows_log2` (`AV1E_SET_TILE_COLUMNS`/`_ROWS` —
/// the CODED value IS the log2 tile count: 0 = 1 column/row (single tile,
/// matching the two functions above), 1 = 2, 2 = 4, ...).
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_tiles(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
    aq_mode: u32,
    two_pass: bool,
    sb_size_128: bool,
    tile_columns_log2: i32,
    tile_rows_log2: i32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_tiles(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            aq_mode as i32,
            two_pass as i32,
            sb_size_128 as i32,
            tile_columns_log2,
            tile_rows_log2,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_tiles failed ({n})");
    out.truncate(n as usize);
    out
}

/// Decoded planes + stream info from the REAL C decoder (`aom_codec_av1_dx`).
pub struct RefDecodedFrame {
    /// Cropped planes, tight row-major u16 (empty u/v when monochrome).
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
    /// `[bit_depth, monochrome, ss_x, ss_y, width, height]`.
    pub info: [i32; 6],
}

/// Decode AV1 bytes through the REAL `aom_codec_av1_dx` public API — the gold
/// pixel oracle. Errors (non-zero shim rc) panic; `expect_w/h` pin the output
/// dims so the caller's buffers are sized before the decode.
pub fn ref_decode_av1_kf(data: &[u8], expect_w: usize, expect_h: usize) -> RefDecodedFrame {
    let mut y = vec![0u16; expect_w * expect_h];
    let mut u = vec![0u16; expect_w * expect_h];
    let mut v = vec![0u16; expect_w * expect_h];
    let mut info = [0i32; 6];
    let rc = unsafe {
        shim_decode_av1_kf(
            data.as_ptr(),
            data.len(),
            expect_w as i32,
            expect_h as i32,
            y.as_mut_ptr(),
            u.as_mut_ptr(),
            v.as_mut_ptr(),
            info.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "shim_decode_av1_kf failed ({rc})");
    let (mono, ss_x, ss_y) = (info[1] != 0, info[2] as usize, info[3] as usize);
    if mono {
        u.clear();
        v.clear();
    } else {
        let cw = (expect_w + ss_x) >> ss_x;
        let ch = (expect_h + ss_y) >> ss_y;
        u.truncate(cw * ch);
        v.truncate(cw * ch);
    }
    RefDecodedFrame { y, u, v, info }
}

// Loop-filter application oracles (dec_shim.c section 4): facades over the
// REAL exported av1_loop_filter_frame_init + av1_filter_block_plane_vert/horz
// driven in the exact single-threaded loop_filter_rows order.
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_lf_frame_init_tables(
        filter_level: *const i32,
        sharpness: i32,
        mode_ref_delta_enabled: i32,
        ref_deltas: *const i8,
        mode_deltas: *const i8,
        seg_enabled: i32,
        seg_active: *const i32,
        seg_data: *const i32,
        plane_start: i32,
        plane_end: i32,
        lfthr_out: *mut u8,
        lvl_out: *mut u8,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_lf_filter_frame(
        y: *mut u16,
        y_stride: i32,
        u: *mut u16,
        v: *mut u16,
        uv_stride: i32,
        crop_w: i32,
        crop_h: i32,
        ss_x: i32,
        ss_y: i32,
        bd: i32,
        mi_rows: i32,
        mi_cols: i32,
        grid_stride: i32,
        g_bsize: *const i32,
        g_txsize: *const i32,
        g_seg: *const i32,
        g_ref0: *const i32,
        g_mode: *const i32,
        g_skip: *const i32,
        g_intrabc: *const i32,
        g_dlf_base: *const i8,
        g_dlf: *const i8,
        filter_level: *const i32,
        sharpness: i32,
        mode_ref_delta_enabled: i32,
        ref_deltas: *const i8,
        mode_deltas: *const i8,
        delta_lf_present: i32,
        delta_lf_multi: i32,
        lossless: *const i32,
        seg_enabled: i32,
        seg_active: *const i32,
        seg_data: *const i32,
        plane_start: i32,
        plane_end: i32,
    ) -> i32;
}

/// Loop-filter frame parameters for the reference facades (mirrors
/// `struct loopfilter` + the delta-lf flags + `xd->lossless` + the 4
/// segmentation LF features [Y_V, Y_H, U, V] per segment).
#[derive(Clone, Copy)]
pub struct RefLfParams {
    /// `[y_vert, y_horz, u, v]`.
    pub filter_level: [i32; 4],
    pub sharpness: i32,
    pub mode_ref_delta_enabled: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
    pub delta_lf_present: bool,
    pub delta_lf_multi: bool,
    pub lossless: [bool; 8],
    pub seg_enabled: bool,
    pub seg_active: [[bool; 4]; 8],
    pub seg_data: [[i32; 4]; 8],
}

impl RefLfParams {
    fn seg_flat(&self) -> ([i32; 32], [i32; 32]) {
        let (mut a, mut d) = ([0i32; 32], [0i32; 32]);
        for s in 0..8 {
            for f in 0..4 {
                a[s * 4 + f] = self.seg_active[s][f] as i32;
                d[s * 4 + f] = self.seg_data[s][f];
            }
        }
        (a, d)
    }
}

/// Reference `av1_loop_filter_init` + `av1_loop_filter_frame_init` table dump:
/// `(lfthr[64] as (mblim, lim, hev_thr), lvl[plane][seg][dir][ref][mode])`.
pub fn ref_lf_frame_init_tables(
    p: &RefLfParams,
    plane_start: i32,
    plane_end: i32,
) -> (Vec<[u8; 3]>, Vec<u8>) {
    let (a, d) = p.seg_flat();
    let mut lfthr = vec![0u8; 64 * 3];
    let mut lvl = vec![0u8; 3 * 8 * 2 * 8 * 2];
    unsafe {
        shim_lf_frame_init_tables(
            p.filter_level.as_ptr(),
            p.sharpness,
            p.mode_ref_delta_enabled as i32,
            p.ref_deltas.as_ptr(),
            p.mode_deltas.as_ptr(),
            p.seg_enabled as i32,
            a.as_ptr(),
            d.as_ptr(),
            plane_start,
            plane_end,
            lfthr.as_mut_ptr(),
            lvl.as_mut_ptr(),
        );
    }
    (
        lfthr.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
        lvl,
    )
}

/// Per-mi-cell grid arrays for [`ref_lf_filter_frame`] (parallel, all length
/// `mi_rows * grid_stride`, cells beyond `mi_cols` ignored). `dlf` holds 4
/// i8 per cell.
pub struct RefLfGrid<'a> {
    pub mi_rows: i32,
    pub mi_cols: i32,
    pub grid_stride: i32,
    pub bsize: &'a [i32],
    pub txsize: &'a [i32],
    pub seg: &'a [i32],
    pub ref0: &'a [i32],
    pub mode: &'a [i32],
    pub skip: &'a [i32],
    pub intrabc: &'a [i32],
    pub dlf_base: &'a [i8],
    pub dlf: &'a [i8],
}

/// Run the REAL loop-filter application over u16 planes (bd == 8 uses the
/// real lowbd path on internal u8 copies — what the production decoder does
/// for 8-bit streams). `y` must hold `y_stride * (mi_rows * 4)` samples; the
/// chroma planes `uv_stride * ((mi_rows * 4) >> ss_y)` (empty + `uv_stride ==
/// 0` for monochrome). Filters in place.
#[allow(clippy::too_many_arguments)]
pub fn ref_lf_filter_frame(
    y: &mut [u16],
    y_stride: usize,
    u: &mut [u16],
    v: &mut [u16],
    uv_stride: usize,
    crop_w: i32,
    crop_h: i32,
    ss_x: i32,
    ss_y: i32,
    bd: i32,
    grid: &RefLfGrid,
    p: &RefLfParams,
    plane_start: i32,
    plane_end: i32,
) {
    let ncells = (grid.mi_rows * grid.grid_stride) as usize;
    assert!(
        grid.bsize.len() >= ncells
            && grid.txsize.len() >= ncells
            && grid.seg.len() >= ncells
            && grid.ref0.len() >= ncells
            && grid.mode.len() >= ncells
            && grid.skip.len() >= ncells
            && grid.intrabc.len() >= ncells
            && grid.dlf_base.len() >= ncells
            && grid.dlf.len() >= 4 * ncells
    );
    assert!(y.len() >= y_stride * (grid.mi_rows as usize * 4));
    assert!(u.len() >= uv_stride * ((grid.mi_rows as usize * 4) >> ss_y) && v.len() == u.len());
    let (a, d) = p.seg_flat();
    let lossless: [i32; 8] = core::array::from_fn(|i| p.lossless[i] as i32);
    let rc = unsafe {
        shim_lf_filter_frame(
            y.as_mut_ptr(),
            y_stride as i32,
            u.as_mut_ptr(),
            v.as_mut_ptr(),
            uv_stride as i32,
            crop_w,
            crop_h,
            ss_x,
            ss_y,
            bd,
            grid.mi_rows,
            grid.mi_cols,
            grid.grid_stride,
            grid.bsize.as_ptr(),
            grid.txsize.as_ptr(),
            grid.seg.as_ptr(),
            grid.ref0.as_ptr(),
            grid.mode.as_ptr(),
            grid.skip.as_ptr(),
            grid.intrabc.as_ptr(),
            grid.dlf_base.as_ptr(),
            grid.dlf.as_ptr(),
            p.filter_level.as_ptr(),
            p.sharpness,
            p.mode_ref_delta_enabled as i32,
            p.ref_deltas.as_ptr(),
            p.mode_deltas.as_ptr(),
            p.delta_lf_present as i32,
            p.delta_lf_multi as i32,
            lossless.as_ptr(),
            p.seg_enabled as i32,
            a.as_ptr(),
            d.as_ptr(),
            plane_start,
            plane_end,
        )
    };
    assert_eq!(rc, 0, "shim_lf_filter_frame failed ({rc})");
}

// dec_shim.c §5 — the REAL av1_cdef_frame walk over a synthetic
// AV1_COMMON + mi grid (skip_txfm per mi, cdef_strength per 64x64 unit).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_cdef_frame(
        y: *mut u16,
        y_stride: i32,
        u: *mut u16,
        v: *mut u16,
        uv_stride: i32,
        mi_rows: i32,
        mi_cols: i32,
        num_planes: i32,
        ss_x: i32,
        ss_y: i32,
        bd: i32,
        damping: i32,
        strengths: *const i32,
        uv_strengths: *const i32,
        skip: *const i32,
        unit_strength: *const i32,
    ) -> i32;
}

/// Drive the REAL `av1_cdef_frame` (single-threaded decoder path) in place
/// over u16 planes (bd 8 runs the real lowbd u8 path internally). `skip` is
/// per-mi `skip_txfm` (`mi_rows x mi_cols`); `unit_strength` the per-64x64-fb
/// decoded `cdef_strength` index (`ceil(mi_rows/16) x ceil(mi_cols/16)`,
/// -1 = none read). Plane strides must be >= `align16(mi_cols*4) >> ss`.
/// Calls `ref_init()` first (the walk dispatches `cdef_copy_rect8_*`,
/// `cdef_find_dir[_dual]` and `cdef_filter_{8,16}_*` through RTCD).
#[allow(clippy::too_many_arguments)]
pub fn ref_cdef_frame(
    y: &mut [u16],
    y_stride: usize,
    u: &mut [u16],
    v: &mut [u16],
    uv_stride: usize,
    mi_rows: i32,
    mi_cols: i32,
    num_planes: usize,
    ss_x: usize,
    ss_y: usize,
    bd: i32,
    damping: i32,
    strengths: &[i32; 8],
    uv_strengths: &[i32; 8],
    skip: &[i32],
    unit_strength: &[i32],
) {
    ref_init();
    let nvfb = (mi_rows as usize).div_ceil(16);
    let nhfb = (mi_cols as usize).div_ceil(16);
    let luma_stride = ((mi_cols as usize * 4) + 15) & !15;
    assert_eq!(skip.len(), (mi_rows * mi_cols) as usize);
    assert_eq!(unit_strength.len(), nvfb * nhfb);
    assert!(y_stride >= luma_stride && y.len() >= y_stride * (mi_rows as usize * 4));
    if num_planes > 1 {
        assert!(uv_stride >= luma_stride >> ss_x);
        assert!(u.len() >= uv_stride * ((mi_rows as usize * 4) >> ss_y) && v.len() == u.len());
    }
    let rc = unsafe {
        shim_cdef_frame(
            y.as_mut_ptr(),
            y_stride as i32,
            u.as_mut_ptr(),
            v.as_mut_ptr(),
            uv_stride as i32,
            mi_rows,
            mi_cols,
            num_planes as i32,
            ss_x as i32,
            ss_y as i32,
            bd,
            damping,
            strengths.as_ptr(),
            uv_strengths.as_ptr(),
            skip.as_ptr(),
            unit_strength.as_ptr(),
        )
    };
    assert_eq!(rc, 0, "shim_cdef_frame failed ({rc})");
}

// av1/encoder/rd.c + rd.h (RD multiplier / RDCOST) and av1/common/quant_common.c
// (dc/ac quant lookups) — rd_shim.c. All exported symbols / real macros; no RTCD.
extern "C" {
    fn shim_compute_rd_mult_based_on_qindex(
        bit_depth: i32,
        update_type: i32,
        qindex: i32,
        tuning: i32,
        mode: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_compute_rd_mult(
        qindex: i32,
        bit_depth: i32,
        update_type: i32,
        layer_depth: i32,
        boost_index: i32,
        frame_type: i32,
        use_fixed_qp_offsets: i32,
        is_stat_consumption_stage: i32,
        tuning: i32,
        mode: i32,
    ) -> i32;
    fn shim_dc_quant_qtx(qindex: i32, delta: i32, bit_depth: i32) -> i32;
    fn shim_ac_quant_qtx(qindex: i32, delta: i32, bit_depth: i32) -> i32;
    fn shim_rdcost(rm: i32, rate: i32, dist: i64) -> i64;
    fn shim_rdcost_neg_r(rm: i32, rate: i32, dist: i64) -> i64;
    #[allow(clippy::too_many_arguments)]
    fn shim_fill_coeff_costs(
        qindex: i32,
        txs_ctx: i32,
        plane: i32,
        eob_multi_size: i32,
        out_txb_skip: *mut i32,
        out_base_eob: *mut i32,
        out_base: *mut i32,
        out_eob_extra: *mut i32,
        out_dc_sign: *mut i32,
        out_lps: *mut i32,
        out_eob_cost: *mut i32,
    );
    fn shim_dist_block_tx_domain(
        coeff: *const i32,
        dqcoeff: *const i32,
        tx_size: i32,
        bd: i32,
        out_dist: *mut i64,
        out_sse: *mut i64,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_intra_cnn_partition_decision(
        win: *const u8,
        qindex: i32,
        bit_depth: i32,
        frame_w: i32,
        frame_h: i32,
        bsize_idx: i32,
        quad_tree_idx: i32,
        level: i32,
        force_cscalar: i32,
        out_logits: *mut f32,
        out_flags: *mut i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_nn_predict(
        features: *const f32,
        num_inputs: i32,
        num_outputs: i32,
        num_hidden_layers: i32,
        hidden_nodes: *const i32,
        weights_flat: *const f32,
        bias_flat: *const f32,
        reduce_prec: i32,
        output: *mut f32,
    );
    fn shim_intra_cnn_run(win: *const u8, force_cscalar: i32, out_cnn_buffer: *mut f32);
}

/// `CNN_OUT_BUF_SIZE` (partition_cnn_weights.h) — the intra-CNN multi-out buffer
/// length: branch_0[20] + branch_1[16] + branch_2[320] + branch_3[1280].
pub const INTRA_CNN_OUT_BUF_SIZE: usize = 1636;

/// Runs `av1_cnn_predict_img_multi_out` with the intra-CNN config on the 65×65
/// luma window `win` (stride 65, replicated-border origin) and returns the raw
/// multi-out buffer (`INTRA_CNN_OUT_BUF_SIZE` floats). `force_cscalar` forces the
/// inner convolve to the scalar `_c` variant (the bit-exact transcription oracle
/// for the Rust port); otherwise the dispatched (AVX2) variant runs, matching the
/// encoder. Not thread-safe when `force_cscalar` — call from one test thread.
pub fn ref_intra_cnn_run(win: &[u8], force_cscalar: bool) -> Vec<f32> {
    assert!(win.len() >= 65 * 65, "CNN window must be at least 65x65");
    let mut out = vec![0.0f32; INTRA_CNN_OUT_BUF_SIZE];
    unsafe {
        shim_intra_cnn_run(win.as_ptr(), i32::from(force_cscalar), out.as_mut_ptr());
    }
    out
}

/// Oracle for `av1/encoder/ml.c` `av1_nn_predict_c` (+ `av1_nn_output_prec_reduce`
/// when `reduce_prec`). `hidden_nodes` gives the per-hidden-layer node counts;
/// `weights_flat`/`bias_flat` are the per-layer weight/bias tables concatenated
/// in NN_CONFIG order (`weights[l][node*num_in + i]`, `bias[l][node]`; the final
/// entry is the linear output layer). Returns the `num_outputs` logits.
#[allow(clippy::too_many_arguments)]
pub fn ref_nn_predict(
    features: &[f32],
    num_inputs: usize,
    num_outputs: usize,
    hidden_nodes: &[i32],
    weights_flat: &[f32],
    bias_flat: &[f32],
    reduce_prec: bool,
) -> Vec<f32> {
    assert!(features.len() >= num_inputs);
    let mut out = vec![0.0f32; num_outputs];
    unsafe {
        shim_nn_predict(
            features.as_ptr(),
            num_inputs as i32,
            num_outputs as i32,
            hidden_nodes.len() as i32,
            hidden_nodes.as_ptr(),
            weights_flat.as_ptr(),
            bias_flat.as_ptr(),
            i32::from(reduce_prec),
            out.as_mut_ptr(),
        );
    }
    out
}

/// Oracle for `av1/encoder/partition_strategy.c` `intra_mode_cnn_partition`
/// (the speed>=1 intra CNN split-vs-nonsplit partition prune). `win` is the
/// 65x65 luma window (stride 65, row-major) = the block's `frame(-1,-1)` origin
/// with replicated top/left borders. `bsize_idx` is `convert_bsize_to_idx`
/// (1=64X64 .. 4=8X8); `quad_tree_idx` is `x->part_search_info.quad_tree_idx`;
/// `level` is `intra_cnn_based_part_prune_level` (1 or 2). Returns
/// `(logits[4], flags[4])` where flags are
/// `[none_disallowed, do_square_split, rect_disabled, square_split_disabled]`.
/// `force_cscalar` forces the inner CNN convolve to the scalar `_c` variant
/// (bit-exact transcription target); `false` uses the dispatched (AVX2) path
/// the encoder actually runs (the flag-parity target).
#[allow(clippy::too_many_arguments)]
pub fn ref_intra_cnn_partition_decision(
    win: &[u8],
    qindex: i32,
    bit_depth: i32,
    frame_w: i32,
    frame_h: i32,
    bsize_idx: i32,
    quad_tree_idx: i32,
    level: i32,
    force_cscalar: bool,
) -> ([f32; 4], [i32; 4]) {
    assert!(win.len() >= 65 * 65, "CNN window must be at least 65x65");
    let mut logits = [0.0f32; 4];
    let mut flags = [0i32; 4];
    unsafe {
        shim_intra_cnn_partition_decision(
            win.as_ptr(),
            qindex,
            bit_depth,
            frame_w,
            frame_h,
            bsize_idx,
            quad_tree_idx,
            level,
            i32::from(force_cscalar),
            logits.as_mut_ptr(),
            flags.as_mut_ptr(),
        );
    }
    (logits, flags)
}

/// Reference `av1_compute_rd_mult_based_on_qindex` (rd.c). Enum args are passed
/// as their integer C values.
pub fn ref_compute_rd_mult_based_on_qindex(
    bit_depth: i32,
    update_type: i32,
    qindex: i32,
    tuning: i32,
    mode: i32,
) -> i32 {
    unsafe { shim_compute_rd_mult_based_on_qindex(bit_depth, update_type, qindex, tuning, mode) }
}

/// Reference `av1_compute_rd_mult` (rd.c).
#[allow(clippy::too_many_arguments)]
pub fn ref_compute_rd_mult(
    qindex: i32,
    bit_depth: i32,
    update_type: i32,
    layer_depth: i32,
    boost_index: i32,
    frame_type: i32,
    use_fixed_qp_offsets: i32,
    is_stat_consumption_stage: i32,
    tuning: i32,
    mode: i32,
) -> i32 {
    unsafe {
        shim_compute_rd_mult(
            qindex,
            bit_depth,
            update_type,
            layer_depth,
            boost_index,
            frame_type,
            use_fixed_qp_offsets,
            is_stat_consumption_stage,
            tuning,
            mode,
        )
    }
}

/// Reference `av1_dc_quant_QTX` (quant_common.c).
pub fn ref_dc_quant_qtx(qindex: i32, delta: i32, bit_depth: i32) -> i32 {
    unsafe { shim_dc_quant_qtx(qindex, delta, bit_depth) }
}

/// Reference `av1_fill_coeff_costs` (rd.c) for one `(txs_ctx, plane)` coeff
/// table + one `(eob_multi_size, plane)` eob table, over the KF-default coeff
/// CDFs at `qindex`. Returns
/// `(txb_skip[26], base_eob[12], base[336], eob_extra[18], dc_sign[6], lps[546], eob_cost[22])`.
#[allow(clippy::type_complexity)]
pub fn ref_fill_coeff_costs(
    qindex: i32,
    txs_ctx: usize,
    plane: usize,
    eob_multi_size: usize,
) -> (
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
) {
    let mut txb_skip = vec![0i32; 13 * 2];
    let mut base_eob = vec![0i32; 4 * 3];
    let mut base = vec![0i32; 42 * 8];
    let mut eob_extra = vec![0i32; 9 * 2];
    let mut dc_sign = vec![0i32; 3 * 2];
    let mut lps = vec![0i32; 21 * 26];
    let mut eob_cost = vec![0i32; 2 * 11];
    unsafe {
        shim_fill_coeff_costs(
            qindex,
            txs_ctx as i32,
            plane as i32,
            eob_multi_size as i32,
            txb_skip.as_mut_ptr(),
            base_eob.as_mut_ptr(),
            base.as_mut_ptr(),
            eob_extra.as_mut_ptr(),
            dc_sign.as_mut_ptr(),
            lps.as_mut_ptr(),
            eob_cost.as_mut_ptr(),
        );
    }
    (txb_skip, base_eob, base, eob_extra, dc_sign, lps, eob_cost)
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
pub fn ref_dist_block_tx_domain(
    coeff: &[i32],
    dqcoeff: &[i32],
    tx_size: usize,
    bd: u8,
) -> (i64, i64) {
    let (mut d, mut s) = (0i64, 0i64);
    unsafe {
        shim_dist_block_tx_domain(
            coeff.as_ptr(),
            dqcoeff.as_ptr(),
            tx_size as i32,
            bd as i32,
            &mut d,
            &mut s,
        )
    };
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

// ---- intra mode-info signaling cost oracles (rd_shim.c) ----------------------

extern "C" {
    fn shim_fill_intra_mode_costs(
        kf_y_cdf: *const u16,
        y_mode_cdf: *const u16,
        uv_mode_cdf: *const u16,
        filter_intra_mode_cdf: *const u16,
        filter_intra_cdfs: *const u16,
        palette_y_mode_cdf: *const u16,
        angle_delta_cdf: *const u16,
        intrabc_cdf: *const u16,
        enable_filter_intra: i32,
        out_y_mode: *mut i32,
        out_mbmode: *mut i32,
        out_uv: *mut i32,
        out_fi_mode: *mut i32,
        out_fi: *mut i32,
        out_pal_y_mode: *mut i32,
        out_angle: *mut i32,
        out_intrabc: *mut i32,
    );
    fn shim_intra_mode_info_cost_y(
        filter_intra_cost: *const i32,
        filter_intra_mode_cost: *const i32,
        angle_delta_cost: *const i32,
        intrabc_cost: *const i32,
        palette_y_mode_cost: *const i32,
        mode_cost: i32,
        mode: i32,
        bsize: i32,
        angle_delta_y: i32,
        use_filter_intra: i32,
        filter_intra_mode: i32,
        use_intrabc: i32,
        try_palette: i32,
        palette_bsize_ctx: i32,
        palette_mode_ctx: i32,
        enable_filter_intra: i32,
        allow_intrabc: i32,
    ) -> i32;
}

/// The 8 intra mode-cost tables from [`ref_fill_intra_mode_costs`], flat, in
/// the layouts documented in rd_shim.c.
pub struct RefIntraModeCosts {
    pub y_mode: Vec<i32>,     // [5][5][13]
    pub mbmode: Vec<i32>,     // [4][13]
    pub uv: Vec<i32>,         // [2][13][14]
    pub fi_mode: Vec<i32>,    // [5]
    pub fi: Vec<i32>,         // [22][2]
    pub pal_y_mode: Vec<i32>, // [7][3][2]
    pub angle: Vec<i32>,      // [8][7]
    pub intrabc: Vec<i32>,    // [2]
}

/// Reference intra-mode slices of `av1_fill_mode_rates` (rd.c). CDF inputs are
/// flat with `nsymbs+1`-padded rows (see rd_shim.c). Outputs zero-initialized
/// (rows gated off — e.g. filter-intra-ineligible block sizes — stay 0).
#[allow(clippy::too_many_arguments)]
pub fn ref_fill_intra_mode_costs(
    kf_y_cdf: &[u16],
    y_mode_cdf: &[u16],
    uv_mode_cdf: &[u16],
    filter_intra_mode_cdf: &[u16],
    filter_intra_cdfs: &[u16],
    palette_y_mode_cdf: &[u16],
    angle_delta_cdf: &[u16],
    intrabc_cdf: &[u16],
    enable_filter_intra: bool,
) -> RefIntraModeCosts {
    assert_eq!(kf_y_cdf.len(), 5 * 5 * 14);
    assert_eq!(y_mode_cdf.len(), 4 * 14);
    assert_eq!(uv_mode_cdf.len(), 2 * 13 * 15);
    assert_eq!(filter_intra_mode_cdf.len(), 6);
    assert_eq!(filter_intra_cdfs.len(), 22 * 3);
    assert_eq!(palette_y_mode_cdf.len(), 7 * 3 * 3);
    assert_eq!(angle_delta_cdf.len(), 8 * 8);
    assert_eq!(intrabc_cdf.len(), 3);
    let mut r = RefIntraModeCosts {
        y_mode: vec![0; 5 * 5 * 13],
        mbmode: vec![0; 4 * 13],
        uv: vec![0; 2 * 13 * 14],
        fi_mode: vec![0; 5],
        fi: vec![0; 22 * 2],
        pal_y_mode: vec![0; 7 * 3 * 2],
        angle: vec![0; 8 * 7],
        intrabc: vec![0; 2],
    };
    unsafe {
        shim_fill_intra_mode_costs(
            kf_y_cdf.as_ptr(),
            y_mode_cdf.as_ptr(),
            uv_mode_cdf.as_ptr(),
            filter_intra_mode_cdf.as_ptr(),
            filter_intra_cdfs.as_ptr(),
            palette_y_mode_cdf.as_ptr(),
            angle_delta_cdf.as_ptr(),
            intrabc_cdf.as_ptr(),
            enable_filter_intra as i32,
            r.y_mode.as_mut_ptr(),
            r.mbmode.as_mut_ptr(),
            r.uv.as_mut_ptr(),
            r.fi_mode.as_mut_ptr(),
            r.fi.as_mut_ptr(),
            r.pal_y_mode.as_mut_ptr(),
            r.angle.as_mut_ptr(),
            r.intrabc.as_mut_ptr(),
        );
    }
    r
}

/// Reference `intra_mode_info_cost_y` (intra_mode_search_utils.h), for the
/// `palette_size[0] == 0` path. The C assert (exclusive mode flags) is LIVE:
/// callers must pass `use_intrabc`/`use_filter_intra` only with `DC_PRED`.
#[allow(clippy::too_many_arguments)]
pub fn ref_intra_mode_info_cost_y(
    costs: &RefIntraModeCosts,
    mode_cost: i32,
    mode: i32,
    bsize: i32,
    angle_delta_y: i32,
    use_filter_intra: bool,
    filter_intra_mode: i32,
    use_intrabc: bool,
    try_palette: bool,
    palette_bsize_ctx: i32,
    palette_mode_ctx: i32,
    enable_filter_intra: bool,
    allow_intrabc: bool,
) -> i32 {
    unsafe {
        shim_intra_mode_info_cost_y(
            costs.fi.as_ptr(),
            costs.fi_mode.as_ptr(),
            costs.angle.as_ptr(),
            costs.intrabc.as_ptr(),
            costs.pal_y_mode.as_ptr(),
            mode_cost,
            mode,
            bsize,
            angle_delta_y,
            use_filter_intra as i32,
            filter_intra_mode,
            use_intrabc as i32,
            try_palette as i32,
            palette_bsize_ctx,
            palette_mode_ctx,
            enable_filter_intra as i32,
            allow_intrabc as i32,
        )
    }
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

// ---------------------------------------------------------------------------
// set_q_index / av1_init_plane_quantizers helpers (rd_shim.c)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn shim_set_q_index(
        bit_depth: i32,
        y_dc_delta_q: i32,
        u_dc_delta_q: i32,
        u_ac_delta_q: i32,
        v_dc_delta_q: i32,
        v_ac_delta_q: i32,
        sharpness: i32,
        qindex: i32,
        out: *mut i16,
    ) -> i32;
    fn shim_get_qindex(
        enabled: i32,
        feature_mask: *const u32,
        altq_data: *const i16,
        segment_id: i32,
        base_qindex: i32,
    ) -> i32;
    fn shim_error_per_bit(rdmult: i32) -> i32;
    fn shim_sad_per_bit(qindex: i32, bit_depth: i32) -> i32;
}

/// Reference `set_q_index` (av1/encoder/av1_quantize.c, static — transcribed
/// row selection over the REAL exported `av1_build_quantizer`): the seven
/// 8-lane rows each plane's `MACROBLOCK_PLANE` receives for `qindex`.
/// Layout: `[plane 0..3][7][8]`, row order `quant / quant_fp / round_fp /
/// quant_shift / zbin / round / dequant` (the C assignment order).
#[allow(clippy::too_many_arguments)]
pub fn ref_set_q_index(
    bit_depth: i32,
    y_dc_delta_q: i32,
    u_dc_delta_q: i32,
    u_ac_delta_q: i32,
    v_dc_delta_q: i32,
    v_ac_delta_q: i32,
    sharpness: i32,
    qindex: i32,
) -> Vec<i16> {
    let mut out = vec![0i16; 3 * 7 * 8];
    let rc = unsafe {
        shim_set_q_index(
            bit_depth,
            y_dc_delta_q,
            u_dc_delta_q,
            u_ac_delta_q,
            v_dc_delta_q,
            v_ac_delta_q,
            sharpness,
            qindex,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "shim_set_q_index allocation failed");
    out
}

/// Reference `av1_get_qindex` (av1/common/quant_common.c, REAL exported fn)
/// over a marshalled `struct segmentation` (`enabled` + per-segment
/// `feature_mask` and `SEG_LVL_ALT_Q` data).
pub fn ref_get_qindex(
    enabled: bool,
    feature_mask: &[u32; 8],
    altq_data: &[i16; 8],
    segment_id: i32,
    base_qindex: i32,
) -> i32 {
    unsafe {
        shim_get_qindex(
            enabled as i32,
            feature_mask.as_ptr(),
            altq_data.as_ptr(),
            segment_id,
            base_qindex,
        )
    }
}

/// Reference `av1_set_error_per_bit` (rd.h static inline, pristine C
/// recompiled in the shim TU).
pub fn ref_error_per_bit(rdmult: i32) -> i32 {
    unsafe { shim_error_per_bit(rdmult) }
}

/// Reference sad-per-bit (rd.c `init_me_luts_bd` entry formula over the REAL
/// exported `av1_convert_qindex_to_q`; `av1_set_sad_per_bit` is a lut lookup
/// of exactly this value).
pub fn ref_sad_per_bit(qindex: i32, bit_depth: i32) -> i32 {
    unsafe { shim_sad_per_bit(qindex, bit_depth) }
}

unsafe extern "C" {
    fn shim_quant_plane_rows(
        kind: i32,
        is_hbd: i32,
        coeff: *const i32,
        n: i32,
        rows: *const i16,
        scan: *const i16,
        iscan: *const i16,
        log_scale: i32,
        qcoeff: *mut i32,
        dqcoeff: *mut i32,
    ) -> u16;
}

/// Quantize through the REAL exported quantize facades with a full
/// `MACROBLOCK_PLANE` (all seven QTX rows installed as `set_q_index` installs
/// them), so the facade — not the caller — picks the rows for `kind`
/// (0 = FP, 1 = B, 2 = DC). `rows` is one plane's `[7][8]` blob in
/// [`ref_set_q_index`] order. Flat path (no qmatrix). Call [`ref_init`] first
/// (the FP/B facades dispatch RTCD kernels).
#[allow(clippy::too_many_arguments)]
pub fn ref_quant_plane_rows(
    kind: i32,
    is_hbd: bool,
    coeff: &[i32],
    rows: &[i16],
    scan: &[i16],
    iscan: &[i16],
    log_scale: i32,
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
) -> u16 {
    assert_eq!(rows.len(), 7 * 8);
    assert_eq!(coeff.len(), qcoeff.len());
    assert_eq!(coeff.len(), dqcoeff.len());
    assert!(scan.len() >= coeff.len() && iscan.len() >= coeff.len());
    unsafe {
        shim_quant_plane_rows(
            kind,
            is_hbd as i32,
            coeff.as_ptr(),
            coeff.len() as i32,
            rows.as_ptr(),
            scan.as_ptr(),
            iscan.as_ptr(),
            log_scale,
            qcoeff.as_mut_ptr(),
            dqcoeff.as_mut_ptr(),
        )
    }
}

// ---------------------------------------------------------------------------
// av1_rd_pick_intra_sby_mode candidate-loop head (rd_shim.c)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn shim_set_y_mode_and_delta_angle(mode_idx: i32, reorder: i32, out_delta: *mut i32) -> i32;
    fn shim_intra_sby_visits(
        mode: i32,
        luma_delta_angle: i32,
        bsize: i32,
        enable_diagonal_intra: i32,
        enable_directional_intra: i32,
        enable_smooth_intra: i32,
        enable_paeth_intra: i32,
        enable_angle_delta: i32,
        disable_smooth_intra: i32,
        prune_filter_intra_level: i32,
        intra_y_mode_mask: *const u16,
        directional_mode_skip_mask: *const u8,
    ) -> i32;
    fn shim_prune_odd_delta(
        mode: i32,
        luma_delta_angle: i32,
        intra_modes_rd_cost: *const i64,
        best_rd: i64,
        prune: i32,
    ) -> i32;
}

/// The REAL exported `set_y_mode_and_delta_angle` (intra_mode_search.c):
/// returns `(mode, angle_delta_y)` for a candidate loop index.
pub fn ref_set_y_mode_and_delta_angle(mode_idx: i32, reorder: bool) -> (i32, i32) {
    let mut delta = 0i32;
    let mode = unsafe { shim_set_y_mode_and_delta_angle(mode_idx, reorder as i32, &mut delta) };
    (mode, delta)
}

/// Reference for the candidate loop's static skip chain (transcription over
/// the REAL `av1_is_diagonal_mode` / `av1_is_directional_mode` /
/// `av1_use_angle_delta` / `max_txsize_lookup`). Returns `true` = evaluated.
#[allow(clippy::too_many_arguments)]
pub fn ref_intra_sby_visits(
    mode: i32,
    luma_delta_angle: i32,
    bsize: i32,
    enable_diagonal_intra: bool,
    enable_directional_intra: bool,
    enable_smooth_intra: bool,
    enable_paeth_intra: bool,
    enable_angle_delta: bool,
    disable_smooth_intra: bool,
    prune_filter_intra_level: i32,
    intra_y_mode_mask: &[u16; 5],
    directional_mode_skip_mask: &[u8; 13],
) -> bool {
    unsafe {
        shim_intra_sby_visits(
            mode,
            luma_delta_angle,
            bsize,
            enable_diagonal_intra as i32,
            enable_directional_intra as i32,
            enable_smooth_intra as i32,
            enable_paeth_intra as i32,
            enable_angle_delta as i32,
            disable_smooth_intra as i32,
            prune_filter_intra_level,
            intra_y_mode_mask.as_ptr(),
            directional_mode_skip_mask.as_ptr(),
        ) != 0
    }
}

/// Reference `prune_luma_odd_delta_angles_using_rd_cost` (transcription; the
/// C fn is static and pure).
pub fn ref_prune_odd_delta(
    mode: i32,
    luma_delta_angle: i32,
    intra_modes_rd_cost: &[i64; 9],
    best_rd: i64,
    prune: bool,
) -> bool {
    unsafe {
        shim_prune_odd_delta(
            mode,
            luma_delta_angle,
            intra_modes_rd_cost.as_ptr(),
            best_rd,
            prune as i32,
        ) != 0
    }
}

// ---------------------------------------------------------------------------
// search_tx_type building blocks (rd_shim.c)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn shim_get_tx_mask_intra(
        tx_size: i32,
        mode: i32,
        use_filter_intra: i32,
        filter_intra_mode: i32,
        lossless: i32,
        reduced_tx_set_used: i32,
        use_reduced_intra_txset: i32,
        use_derived_intra_tx_type_set: i32,
        enable_flip_idtx: i32,
        use_intra_dct_only: i32,
        use_default_intra_tx_type: i32,
        use_screen_content_tools: i32,
        prune_tx_type_using_stats: i32,
        out_txk_allowed: *mut i32,
    ) -> i32;
    fn shim_pixel_diff_dist(
        src_diff: *const i16,
        n_diff: i32,
        plane_bsize: i32,
        tx_bsize: i32,
        blk_row: i32,
        blk_col: i32,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        subsampling_x: i32,
        subsampling_y: i32,
        out_mse_q8: *mut u32,
    ) -> i64;
}

/// Reference `get_tx_mask` LUMA-INTRA arm (transcription over the REAL
/// `av1_get_ext_tx_set_type` + REAL blockd.h used-flag tables). Returns
/// `(allowed_tx_mask, txk_allowed)` with `txk_allowed == 16` meaning "all".
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn ref_get_tx_mask_intra(
    tx_size: i32,
    mode: i32,
    use_filter_intra: bool,
    filter_intra_mode: i32,
    lossless: bool,
    reduced_tx_set_used: bool,
    use_reduced_intra_txset: i32,
    use_derived_intra_tx_type_set: bool,
    enable_flip_idtx: bool,
    use_intra_dct_only: bool,
    use_default_intra_tx_type: bool,
    use_screen_content_tools: bool,
    prune_tx_type_using_stats: i32,
) -> (u16, i32) {
    let mut txk = 0i32;
    let mask = unsafe {
        shim_get_tx_mask_intra(
            tx_size,
            mode,
            use_filter_intra as i32,
            filter_intra_mode,
            lossless as i32,
            reduced_tx_set_used as i32,
            use_reduced_intra_txset,
            use_derived_intra_tx_type_set as i32,
            enable_flip_idtx as i32,
            use_intra_dct_only as i32,
            use_default_intra_tx_type as i32,
            use_screen_content_tools as i32,
            prune_tx_type_using_stats,
            &mut txk,
        )
    };
    (mask as u16, txk)
}

/// The REAL EXPORTED `av1_pixel_diff_dist` (tx_search.c) over a marshalled
/// MACROBLOCK. `src_diff` is the plane-block residual (stride = plane block
/// width). Call [`ref_init`] first (`aom_sum_squares_2d_i16` is RTCD).
/// Returns `(sse, block_mse_q8)`.
#[allow(clippy::too_many_arguments)]
pub fn ref_pixel_diff_dist(
    src_diff: &[i16],
    plane_bsize: i32,
    tx_bsize: i32,
    blk_row: i32,
    blk_col: i32,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    subsampling_x: i32,
    subsampling_y: i32,
) -> (i64, u32) {
    let mut mse = 0u32;
    let sse = unsafe {
        shim_pixel_diff_dist(
            src_diff.as_ptr(),
            src_diff.len() as i32,
            plane_bsize,
            tx_bsize,
            blk_row,
            blk_col,
            mb_to_right_edge,
            mb_to_bottom_edge,
            subsampling_x,
            subsampling_y,
            &mut mse,
        )
    };
    assert!(sse >= 0, "shim_pixel_diff_dist allocation failed");
    (sse, mse)
}

// ---------------------------------------------------------------------------
// tx-size signaling cost (rd_shim.c)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn shim_fill_tx_size_costs(tx_size_cdf: *const u16, out: *mut i32);
    fn shim_tx_size_cost(
        costs: *const i32,
        tx_mode_is_select: i32,
        bsize: i32,
        tx_size: i32,
        tx_size_ctx: i32,
    ) -> i32;
}

/// Reference tx-size cost fill (transcription of the rd.c slice over the REAL
/// `av1_cost_tokens_from_cdf`). `tx_size_cdf` flat `[4][3][4]`; returns flat
/// `[4][3][3]`.
pub fn ref_fill_tx_size_costs(tx_size_cdf: &[u16]) -> Vec<i32> {
    assert_eq!(tx_size_cdf.len(), 4 * 3 * 4);
    let mut out = vec![0i32; 4 * 3 * 3];
    unsafe { shim_fill_tx_size_costs(tx_size_cdf.as_ptr(), out.as_mut_ptr()) };
    out
}

/// Reference `tx_size_cost` (tx_search.h; transcription over the REAL
/// `bsize_to_tx_size_cat` / `tx_size_to_depth` / `block_signals_txsize`).
pub fn ref_tx_size_cost(
    costs: &[i32],
    tx_mode_is_select: bool,
    bsize: i32,
    tx_size: i32,
    tx_size_ctx: i32,
) -> i32 {
    assert_eq!(costs.len(), 4 * 3 * 3);
    unsafe {
        shim_tx_size_cost(
            costs.as_ptr(),
            tx_mode_is_select as i32,
            bsize,
            tx_size,
            tx_size_ctx,
        )
    }
}

// intra_mode_search.c model-rd prune statics — verbatim transcriptions in
// rd_shim.c (the double-threshold math compiled as C).
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_get_model_rd_index_for_pruning(
        cur_mode: i32,
        qindex: i32,
        top_intra_model_count_allowed: i32,
        adapt_top_model_rd_count_using_neighbors: i32,
        left_available: i32,
        left_mode: i32,
        up_available: i32,
        above_mode: i32,
    ) -> i32;
    fn shim_prune_intra_y_mode(
        this_model_rd: i64,
        best_model_rd: *mut i64,
        top_intra_model_rd: *mut i64,
        max_model_cnt_allowed: i32,
        model_rd_index_for_pruning: i32,
    ) -> i32;
}

/// Reference `get_model_rd_index_for_pruning` (intra_mode_search.c,
/// transcription). Neighbour modes: `None` = unavailable.
pub fn ref_get_model_rd_index_for_pruning(
    cur_mode: usize,
    qindex: i32,
    top_intra_model_count_allowed: i32,
    adapt_top_model_rd_count_using_neighbors: bool,
    left_mode: Option<usize>,
    above_mode: Option<usize>,
) -> i32 {
    unsafe {
        shim_get_model_rd_index_for_pruning(
            cur_mode as i32,
            qindex,
            top_intra_model_count_allowed,
            adapt_top_model_rd_count_using_neighbors as i32,
            left_mode.is_some() as i32,
            left_mode.unwrap_or(0) as i32,
            above_mode.is_some() as i32,
            above_mode.unwrap_or(0) as i32,
        )
    }
}

/// Reference `prune_intra_y_mode` (intra_mode_search.c, transcription).
/// Mutates `best_model_rd` + `top_intra_model_rd` exactly as the C does;
/// returns the prune decision.
pub fn ref_prune_intra_y_mode(
    this_model_rd: i64,
    best_model_rd: &mut i64,
    top_intra_model_rd: &mut [i64],
    max_model_cnt_allowed: usize,
    model_rd_index_for_pruning: usize,
) -> bool {
    assert!(max_model_cnt_allowed <= top_intra_model_rd.len());
    assert!(model_rd_index_for_pruning < top_intra_model_rd.len());
    unsafe {
        shim_prune_intra_y_mode(
            this_model_rd,
            best_model_rd,
            top_intra_model_rd.as_mut_ptr(),
            max_model_cnt_allowed as i32,
            model_rd_index_for_pruning as i32,
        ) != 0
    }
}

// intra_rd_variance_factor (intra_mode_search.c statics) — verbatim
// transcription in rd_shim.c over the REAL 4x4 variance kernels + libm log1p.
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_intra_rd_variance_factor(
        speed: i32,
        src: *const u16,
        src_off: i32,
        src_stride: i32,
        recon: *const u16,
        ref_off: i32,
        ref_stride: i32,
        bsize: i32,
        sb_size: i32,
        mi_row: i32,
        mi_col: i32,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        bd: i32,
        cache_var: *mut i32,
        cache_log_var: *mut f64,
    ) -> f64;
}

/// Reference `intra_rd_variance_factor` (+ `compute_avg_log_variance` +
/// `av1_calc_normalized_variance`), transcription over the REAL
/// `aom_variance4x4_c` / `aom_highbd_{8,10,12}_variance4x4_c` + libm `log1p`.
/// The per-4x4 source-var cache halves (`var` init -1, `log_var` init -1.0,
/// one entry per mi position in the superblock) are mutated as the C does.
#[allow(clippy::too_many_arguments)]
pub fn ref_intra_rd_variance_factor(
    speed: i32,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    recon: &[u16],
    ref_off: usize,
    ref_stride: usize,
    bsize: usize,
    sb_size: usize,
    mi_row: i32,
    mi_col: i32,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    bd: u8,
    cache_var: &mut [i32],
    cache_log_var: &mut [f64],
) -> f64 {
    assert_eq!(cache_var.len(), cache_log_var.len());
    unsafe {
        shim_intra_rd_variance_factor(
            speed,
            src.as_ptr(),
            src_off as i32,
            src_stride as i32,
            recon.as_ptr(),
            ref_off as i32,
            ref_stride as i32,
            bsize as i32,
            sb_size as i32,
            mi_row,
            mi_col,
            mb_to_right_edge,
            mb_to_bottom_edge,
            bd as i32,
            cache_var.as_mut_ptr(),
            cache_log_var.as_mut_ptr(),
        )
    }
}

// HOG intra-mode prune (intra_mode_search_utils.h) — hog_shim.c includes the
// REAL header (its own static weights/nnconfig + the real static-inline
// generate_hog bodies). The NN oracle is av1_nn_predict_avx2 — the variant
// RTCD resolves to on the AVX2-capable reference environment (f32 accumulation
// order differs from the C/SSE3 variants).
extern "C" {
    fn shim_hog_nn_predict(hist: *const f32, reduce_prec: i32, scores: *mut f32);
    fn shim_hog_nn_predict_dispatched(hist: *const f32, reduce_prec: i32, scores: *mut f32);
    fn shim_generate_hog(
        src: *const u16,
        src_off: i32,
        stride: i32,
        rows: i32,
        cols: i32,
        bd: i32,
        hist: *mut f32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_prune_intra_mode_with_hog_y(
        src: *const u16,
        src_off: i32,
        stride: i32,
        bsize: i32,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        bd: i32,
        th: f32,
        mask: *mut u8,
    );
}

/// Reference HOG-model NN predict (`av1_nn_predict_avx2` on the real
/// `av1_intra_hog_model_nnconfig`).
pub fn ref_hog_nn_predict(hist: &[f32; 32], reduce_prec: bool) -> [f32; 8] {
    let mut scores = [0f32; 8];
    unsafe { shim_hog_nn_predict(hist.as_ptr(), reduce_prec as i32, scores.as_mut_ptr()) };
    scores
}

/// The RTCD-dispatched `av1_nn_predict` on the same config (call `ref_init`
/// first) — lets a harness prove the dispatch resolves to the AVX2 variant on
/// the running machine.
pub fn ref_hog_nn_predict_dispatched(hist: &[f32; 32], reduce_prec: bool) -> [f32; 8] {
    let mut scores = [0f32; 8];
    unsafe {
        shim_hog_nn_predict_dispatched(hist.as_ptr(), reduce_prec as i32, scores.as_mut_ptr())
    };
    scores
}

/// Reference `generate_hog` (the REAL `lowbd_generate_hog` /
/// `highbd_generate_hog` static inlines; bd selects the variant + buffer
/// depth). `rows`/`cols` are the caller's edge-clipped dims.
pub fn ref_generate_hog(
    src: &[u16],
    src_off: usize,
    stride: usize,
    rows: usize,
    cols: usize,
    bd: u8,
) -> [f32; 32] {
    let mut hist = [0f32; 32];
    unsafe {
        shim_generate_hog(
            src.as_ptr(),
            src_off as i32,
            stride as i32,
            rows as i32,
            cols as i32,
            bd as i32,
            hist.as_mut_ptr(),
        )
    };
    hist
}

/// Reference `prune_intra_mode_with_hog` (luma): transcribed
/// clip/scale/threshold wrapper over the REAL generate_hog +
/// av1_nn_predict_avx2. Returns the 13-entry directional skip mask.
#[allow(clippy::too_many_arguments)]
pub fn ref_prune_intra_mode_with_hog_y(
    src: &[u16],
    src_off: usize,
    stride: usize,
    bsize: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    bd: u8,
    th: f32,
) -> [bool; 13] {
    let mut mask = [0u8; 13];
    unsafe {
        shim_prune_intra_mode_with_hog_y(
            src.as_ptr(),
            src_off as i32,
            stride as i32,
            bsize as i32,
            mb_to_right_edge,
            mb_to_bottom_edge,
            bd as i32,
            th,
            mask.as_mut_ptr(),
        )
    };
    mask.map(|b| b != 0)
}

// ---- chroma intra RD oracles (rd_shim.c) -------------------------------------

extern "C" {
    fn shim_get_tx_type_uv_intra(
        uv_mode: i32,
        lossless: i32,
        tx_size: i32,
        reduced_tx_set_used: i32,
    ) -> i32;
    fn shim_get_tx_mask_uv_intra(
        tx_size: i32,
        uv_mode: i32,
        luma_mode: i32,
        luma_use_filter_intra: i32,
        luma_filter_intra_mode: i32,
        lossless: i32,
        reduced_tx_set_used: i32,
        use_reduced_intra_txset: i32,
        enable_flip_idtx: i32,
        use_intra_dct_only: i32,
        out_txk_allowed: *mut i32,
    ) -> i32;
    fn shim_fill_cfl_costs(cfl_sign_cdf: *const u16, cfl_alpha_cdf: *const u16, out: *mut i32);
    fn shim_fill_palette_uv_mode_costs(palette_uv_mode_cdf: *const u16, out: *mut i32);
    fn shim_intra_mode_info_cost_uv(
        angle_delta_cost: *const i32,
        palette_uv_mode_cost: *const i32,
        mode_cost: i32,
        uv_mode: i32,
        bsize: i32,
        angle_delta_uv: i32,
        try_palette: i32,
        y_palette_active: i32,
    ) -> i32;
    fn shim_cfl_store_tx(
        luma: *const u16,
        block_off: i32,
        stride: i32,
        row: i32,
        col: i32,
        tx_size: i32,
        bsize: i32,
        mi_row: i32,
        mi_col: i32,
        ss_x: i32,
        ss_y: i32,
        bd: i32,
        recon_q3: *mut u16,
        buf_w: *mut i32,
        buf_h: *mut i32,
        params_computed: *mut i32,
    );
    fn shim_cfl_predict_block(
        recon_q3: *mut u16,
        ac_q3: *mut i16,
        buf_w: *mut i32,
        buf_h: *mut i32,
        params_computed: *mut i32,
        dst: *mut u16,
        dst_off: i32,
        dst_stride: i32,
        tx_size: i32,
        plane: i32,
        cfl_alpha_idx: i32,
        cfl_alpha_signs: i32,
        bsize: i32,
        lossless: i32,
        ss_x: i32,
        ss_y: i32,
        bd: i32,
    );
}

/// The REAL `av1_get_tx_type` (blockd.h) for `PLANE_TYPE_UV` on an INTRA
/// block over a stack MACROBLOCKD stub (uv_mode + lossless marshalled).
pub fn ref_get_tx_type_uv_intra(
    uv_mode: usize,
    lossless: bool,
    tx_size: usize,
    reduced: bool,
) -> usize {
    let t = unsafe {
        shim_get_tx_type_uv_intra(
            uv_mode as i32,
            lossless as i32,
            tx_size as i32,
            reduced as i32,
        )
    };
    assert!(t >= 0, "shim_get_tx_type_uv_intra alloc failed");
    t as usize
}

/// Reference chroma-intra arm of `get_tx_mask` (tx_search.c) — transcription
/// over the REAL `av1_get_tx_type` + real header statics. Returns
/// `(mask, txk_allowed)`.
#[allow(clippy::too_many_arguments)]
pub fn ref_get_tx_mask_uv_intra(
    tx_size: usize,
    uv_mode: usize,
    luma_mode: usize,
    luma_use_filter_intra: bool,
    luma_filter_intra_mode: usize,
    lossless: bool,
    reduced: bool,
    use_reduced_intra_txset: u8,
    enable_flip_idtx: bool,
    use_intra_dct_only: bool,
) -> (u16, usize) {
    let mut txk: i32 = -1;
    let mask = unsafe {
        shim_get_tx_mask_uv_intra(
            tx_size as i32,
            uv_mode as i32,
            luma_mode as i32,
            luma_use_filter_intra as i32,
            luma_filter_intra_mode as i32,
            lossless as i32,
            reduced as i32,
            use_reduced_intra_txset as i32,
            enable_flip_idtx as i32,
            use_intra_dct_only as i32,
            &mut txk,
        )
    };
    (mask as u16, txk as usize)
}

/// Reference CfL slice of `av1_fill_mode_rates` (rd.c): flat
/// `[8][2][16]` costs from one padded sign row (9) + flat `[6][17]` alpha
/// CDFs, via the REAL `av1_cost_tokens_from_cdf` + real `CFL_*` macros.
pub fn ref_fill_cfl_costs(cfl_sign_cdf: &[u16], cfl_alpha_cdf: &[u16]) -> Vec<i32> {
    assert_eq!(cfl_sign_cdf.len(), 9);
    assert_eq!(cfl_alpha_cdf.len(), 6 * 17);
    let mut out = vec![0i32; 8 * 2 * 16];
    unsafe {
        shim_fill_cfl_costs(
            cfl_sign_cdf.as_ptr(),
            cfl_alpha_cdf.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    out
}

/// Reference palette-UV-flag cost fill (rd.c): flat `[2][2]` from `[2][3]`.
pub fn ref_fill_palette_uv_mode_costs(palette_uv_mode_cdf: &[u16]) -> Vec<i32> {
    assert_eq!(palette_uv_mode_cdf.len(), 2 * 3);
    let mut out = vec![0i32; 2 * 2];
    unsafe { shim_fill_palette_uv_mode_costs(palette_uv_mode_cdf.as_ptr(), out.as_mut_ptr()) };
    out
}

/// Reference `intra_mode_info_cost_uv` (intra_mode_search_utils.h,
/// `palette_size[1] == 0` path) over the REAL header gates. Cost tables flat
/// (`angle_delta [8][7]`, `palette_uv_mode [2][2]`).
#[allow(clippy::too_many_arguments)]
pub fn ref_intra_mode_info_cost_uv(
    angle_delta_cost: &[i32],
    palette_uv_mode_cost: &[i32],
    mode_cost: i32,
    uv_mode: usize,
    bsize: usize,
    angle_delta_uv: i32,
    try_palette: bool,
    y_palette_active: bool,
) -> i32 {
    assert_eq!(angle_delta_cost.len(), 8 * 7);
    assert_eq!(palette_uv_mode_cost.len(), 4);
    unsafe {
        shim_intra_mode_info_cost_uv(
            angle_delta_cost.as_ptr(),
            palette_uv_mode_cost.as_ptr(),
            mode_cost,
            uv_mode as i32,
            bsize as i32,
            angle_delta_uv,
            try_palette as i32,
            y_palette_active as i32,
        )
    }
}

/// CfL context state threaded through the REAL encoder-side CfL facades:
/// `(recon_buf_q3, ac_buf_q3, buf_width, buf_height, are_parameters_computed)`.
pub struct RefCflState {
    pub recon_q3: [u16; 1024],
    pub ac_q3: [i16; 1024],
    pub buf_w: i32,
    pub buf_h: i32,
    pub params_computed: bool,
}

impl Default for RefCflState {
    fn default() -> Self {
        RefCflState {
            recon_q3: [0; 1024],
            ac_q3: [0; 1024],
            buf_w: 0,
            buf_h: 0,
            params_computed: false,
        }
    }
}

/// The REAL exported `cfl_store_tx` (cfl.c): subsample one reconstructed luma
/// tx block into the CfL Q3 buffer (production RTCD subsampling kernels),
/// tracking the written surface + invalidating the AC parameters. `luma` is a
/// u16 plane at any bit depth; `(row, col)` the txb offset in luma mi units;
/// `mi_row`/`mi_col` the block position (sub-8x8 shared-chroma adjustment).
#[allow(clippy::too_many_arguments)]
pub fn ref_cfl_store_tx(
    st: &mut RefCflState,
    luma: &[u16],
    block_off: usize,
    stride: usize,
    row: i32,
    col: i32,
    tx_size: usize,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    ss_x: i32,
    ss_y: i32,
    bd: u8,
) {
    let mut pc = st.params_computed as i32;
    unsafe {
        shim_cfl_store_tx(
            luma.as_ptr(),
            block_off as i32,
            stride as i32,
            row,
            col,
            tx_size as i32,
            bsize as i32,
            mi_row,
            mi_col,
            ss_x,
            ss_y,
            bd as i32,
            st.recon_q3.as_mut_ptr(),
            &mut st.buf_w,
            &mut st.buf_h,
            &mut pc,
        )
    };
    st.params_computed = pc != 0;
}

/// The REAL exported `av1_cfl_predict_block` (cfl.c): lazily pad + subtract
/// the block average (first call after a store), then add the alpha-scaled AC
/// into `dst` (which must already hold the DC prediction). `plane` is 1 (U)
/// or 2 (V); alpha comes from the coded `(cfl_alpha_idx, cfl_alpha_signs)`.
#[allow(clippy::too_many_arguments)]
pub fn ref_cfl_predict_block(
    st: &mut RefCflState,
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    tx_size: usize,
    plane: usize,
    cfl_alpha_idx: i32,
    cfl_alpha_signs: i32,
    bsize: usize,
    lossless: bool,
    ss_x: i32,
    ss_y: i32,
    bd: u8,
) {
    let mut pc = st.params_computed as i32;
    unsafe {
        shim_cfl_predict_block(
            st.recon_q3.as_mut_ptr(),
            st.ac_q3.as_mut_ptr(),
            &mut st.buf_w,
            &mut st.buf_h,
            &mut pc,
            dst.as_mut_ptr(),
            dst_off as i32,
            dst_stride as i32,
            tx_size as i32,
            plane as i32,
            cfl_alpha_idx,
            cfl_alpha_signs,
            bsize as i32,
            lossless as i32,
            ss_x,
            ss_y,
            bd as i32,
        )
    };
    st.params_computed = pc != 0;
}

// ---- winner re-encode (av1_encode_intra_block_plane) LUMA map oracles --------

extern "C" {
    fn shim_get_tx_type_y(
        lossless: i32,
        tx_size: i32,
        reduced_tx_set_used: i32,
        tx_type_map: *const u8,
        map_stride: i32,
        blk_row: i32,
        blk_col: i32,
    ) -> i32;
    fn shim_update_txk_array(
        tx_type_map: *mut u8,
        map_stride: i32,
        blk_row: i32,
        blk_col: i32,
        tx_size: i32,
        tx_type: i32,
    ) -> i32;
}

/// The REAL `av1_get_tx_type` (blockd.h:1283) for `PLANE_TYPE_Y` on an INTRA
/// block over a stack MACROBLOCKD stub with the RDO-time block-local
/// `tx_type_map` (stride `mi_size_wide[bsize]`) marshalled in. The map's
/// origin cells must hold in-set types for `(tx_size, reduced)` — the C
/// in-set assert is LIVE in this build.
pub fn ref_get_tx_type_y(
    lossless: bool,
    tx_size: usize,
    reduced: bool,
    tx_type_map: &[u8],
    map_stride: usize,
    blk_row: usize,
    blk_col: usize,
) -> usize {
    let t = unsafe {
        shim_get_tx_type_y(
            lossless as i32,
            tx_size as i32,
            reduced as i32,
            tx_type_map.as_ptr(),
            map_stride as i32,
            blk_row as i32,
            blk_col as i32,
        )
    };
    assert!(t >= 0, "shim_get_tx_type_y alloc failed");
    t as usize
}

/// The REAL `update_txk_array` (blockd.h:1260) over the block-local
/// `tx_type_map` — the `eob == 0` DCT_DCT reset write (encodemb.c:770-779)
/// incl. the 64-side 16x16-unit fill.
pub fn ref_update_txk_array(
    tx_type_map: &mut [u8],
    map_stride: usize,
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    tx_type: usize,
) {
    let r = unsafe {
        shim_update_txk_array(
            tx_type_map.as_mut_ptr(),
            map_stride as i32,
            blk_row as i32,
            blk_col as i32,
            tx_size as i32,
            tx_type as i32,
        )
    };
    assert!(r == 0, "shim_update_txk_array alloc failed");
}

// ---- partition RDO primitives (REAL rd.h static inlines) ---------------------

extern "C" {
    fn shim_rd_cost_update(mult: i32, rate: *mut i32, dist: *mut i64, rdcost: *mut i64);
    #[allow(clippy::too_many_arguments)]
    fn shim_rd_stats_subtraction(
        mult: i32,
        l_rate: i32,
        l_dist: i64,
        l_rdcost: i64,
        r_rate: i32,
        r_dist: i64,
        r_rdcost: i64,
        o_rate: *mut i32,
        o_dist: *mut i64,
        o_rdcost: *mut i64,
    );
}

/// The REAL `av1_rd_cost_update` (rd.h:201) on a `(rate, dist, rdcost)`
/// slice.
pub fn ref_rd_cost_update(mult: i32, rate: i32, dist: i64, rdcost: i64) -> (i32, i64, i64) {
    let (mut r, mut d, mut c) = (rate, dist, rdcost);
    unsafe { shim_rd_cost_update(mult, &mut r, &mut d, &mut c) };
    (r, d, c)
}

/// The REAL `av1_rd_stats_subtraction` (rd.h:210).
#[allow(clippy::too_many_arguments)]
pub fn ref_rd_stats_subtraction(
    mult: i32,
    left: (i32, i64, i64),
    right: (i32, i64, i64),
) -> (i32, i64, i64) {
    let (mut r, mut d, mut c) = (0i32, 0i64, 0i64);
    unsafe {
        shim_rd_stats_subtraction(
            mult, left.0, left.1, left.2, right.0, right.1, right.2, &mut r, &mut d, &mut c,
        )
    };
    (r, d, c)
}

// ---- encode_sb (winner dry-run walk) context facades (rd_shim.c) ----------

unsafe extern "C" {
    fn shim_store_cfl_required(monochrome: i32, is_chroma_ref: i32, uv_mode: i32) -> i32;
    fn shim_set_entropy_contexts(
        above: *mut i8,
        left: *mut i8,
        plane: i32,
        plane_bsize: i32,
        tx_size: i32,
        has_eob: i32,
        aoff: i32,
        loff: i32,
    ) -> i32;
}

/// The REAL `store_cfl_required` (cfl.h:38) — the NON-rdo `store_y` gate of
/// `encode_superblock` (intra arm: `is_inter_block == 0` marshalled).
pub fn ref_store_cfl_required(monochrome: bool, is_chroma_ref: bool, uv_mode: usize) -> bool {
    let r =
        unsafe { shim_store_cfl_required(monochrome as i32, is_chroma_ref as i32, uv_mode as i32) };
    assert!(r >= 0, "shim_store_cfl_required alloc failed");
    r != 0
}

/// The REAL `av1_set_entropy_contexts` (blockd.c:29) for an INTERIOR block
/// (`mb_to_right/bottom_edge = 0` — the unclipped memset arms). `above`/
/// `left` are the tile-level plane entropy contexts AT THE BLOCK ORIGIN
/// (the `pd->above/left_entropy_context` pointers after `set_offsets`);
/// `aoff`/`loff` are the txb's block-relative 4x4 offsets.
#[allow(clippy::too_many_arguments)]
pub fn ref_set_entropy_contexts(
    above: &mut [i8],
    left: &mut [i8],
    plane: usize,
    plane_bsize: usize,
    tx_size: usize,
    has_eob: i32,
    aoff: usize,
    loff: usize,
) {
    let r = unsafe {
        shim_set_entropy_contexts(
            above.as_mut_ptr(),
            left.as_mut_ptr(),
            plane as i32,
            plane_bsize as i32,
            tx_size as i32,
            has_eob,
            aoff as i32,
            loff as i32,
        )
    };
    assert!(r >= 0, "shim_set_entropy_contexts alloc failed");
}

// ---- rect-partition-stage facades (rd_shim.c; encoder track) ---------------

unsafe extern "C" {
    fn shim_get_plane_block_size(bsize: i32, ss_x: i32, ss_y: i32) -> i32;
    fn shim_log_sub_block_var(
        src: *const u16,
        off: i32,
        stride: i32,
        bsize: i32,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        bd: i32,
        out_min: *mut f64,
        out_max: *mut f64,
    ) -> i32;
}

/// The REAL `get_plane_block_size` (blockd.h over `av1_ss_size_lookup`);
/// 255 = `BLOCK_INVALID` (the `partition_rect_allowed` chroma guard).
pub fn ref_get_plane_block_size(bsize: usize, ss_x: usize, ss_y: usize) -> usize {
    (unsafe { shim_get_plane_block_size(bsize as i32, ss_x as i32, ss_y as i32) } & 0xff) as usize
}

/// `log_sub_block_var` (partition_search.c:5572 — transcribed static loop
/// over the REAL `av1_calc_normalized_variance` + real variance4x4 kernels +
/// libm log1p): `(log1p(min var/16), log1p(max var/16))` across the block's
/// 4x4 source sub-blocks. Feeds the per-node ALLINTRA variance arm
/// (:5791-5827).
pub fn ref_log_sub_block_var(
    src: &[u16],
    off: usize,
    stride: usize,
    bsize: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    bd: u8,
) -> (f64, f64) {
    let (mut mn, mut mx) = (0f64, 0f64);
    let r = unsafe {
        shim_log_sub_block_var(
            src.as_ptr(),
            off as i32,
            stride as i32,
            bsize as i32,
            mb_to_right_edge,
            mb_to_bottom_edge,
            i32::from(bd),
            &mut mn,
            &mut mx,
        )
    };
    assert!(r >= 0, "shim_log_sub_block_var alloc failed");
    (mn, mx)
}

unsafe extern "C" {
    fn shim_filter_intra_allowed_bsize_x(enable_filter_intra: i32, bsize: i32) -> i32;
}

/// The REAL `av1_filter_intra_allowed_bsize` (reconintra.h) — the
/// `rd_pick_filter_intra_sby` call-site gate (intra_mode_search.c:1672):
/// seq enable AND both block dims <= 32.
pub fn ref_filter_intra_allowed_bsize(enable_filter_intra: bool, bsize: usize) -> bool {
    unsafe { shim_filter_intra_allowed_bsize_x(enable_filter_intra as i32, bsize as i32) != 0 }
}

// dec_shim.c section 4 (append-only addition): palette colour-index map oracles —
// av1_get_palette_color_index_context is EXPORTED (directly bound, no facade needed);
// shim_get_block_dimensions facades the static-inline av1_get_block_dimensions;
// shim_decode_palette_tokens facades the REAL exported av1_decode_palette_tokens
// end-to-end over a real aom_reader byte stream.
unsafe extern "C" {
    fn av1_get_palette_color_index_context(
        color_map: *const u8,
        stride: i32,
        r: i32,
        c: i32,
        palette_size: i32,
        color_order: *mut u8,
        color_idx: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_get_block_dimensions(
        bsize: i32,
        plane: i32,
        ss_x: i32,
        ss_y: i32,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        width: *mut i32,
        height: *mut i32,
        rows: *mut i32,
        cols: *mut i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_decode_palette_tokens(
        data: *const u8,
        len: usize,
        plane: i32,
        bsize: i32,
        n_colors: i32,
        ss_x: i32,
        ss_y: i32,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
        map_cdf_in: *const u16,
        map_cdf_out: *mut u16,
        color_map_out: *mut u8,
    ) -> i32;
}

/// `MAX_PALETTE_BLOCK_WIDTH` / `MAX_PALETTE_BLOCK_HEIGHT` (`av1/common/blockd.h`): the
/// colour-index map's fixed stride/height (64x64, the largest palette-allowed block).
pub const REF_MAX_PALETTE_BLOCK_WIDTH: usize = 64;
pub const REF_MAX_PALETTE_BLOCK_HEIGHT: usize = 64;
/// `PALETTE_SIZES * PALETTE_COLOR_INDEX_CONTEXTS * CDF_SIZE(PALETTE_COLORS)` = `7*5*9`:
/// one plane's full colour-index CDF array, flattened.
pub const REF_PALETTE_MAP_CDF_LEN: usize = 7 * 5 * 9;

/// The REAL exported `av1_get_palette_color_index_context` (`av1/common/entropymode.c`):
/// the colour-index map token context at `(r, c)`. Returns `(color_order, color_idx,
/// color_ctx)` — see `aom_entropy::partition::get_palette_color_index_context`, which
/// this is the oracle for.
pub fn ref_get_palette_color_index_context(
    color_map: &[u8],
    stride: usize,
    r: usize,
    c: usize,
    palette_size: i32,
) -> ([u8; 8], i32, i32) {
    let mut color_order = [0u8; 8];
    let mut color_idx = 0i32;
    let ctx = unsafe {
        av1_get_palette_color_index_context(
            color_map.as_ptr(),
            stride as i32,
            r as i32,
            c as i32,
            palette_size,
            color_order.as_mut_ptr(),
            &mut color_idx,
        )
    };
    (color_order, color_idx, ctx)
}

/// The REAL `av1_get_block_dimensions` (`av1/common/blockd.h`, static inline facade):
/// `(width, height, rows, cols)` — see `aom_entropy::partition::get_block_dimensions`.
pub fn ref_get_block_dimensions(
    bsize: usize,
    plane: usize,
    ss_x: usize,
    ss_y: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
) -> (usize, usize, usize, usize) {
    let (mut width, mut height, mut rows, mut cols) = (0i32, 0i32, 0i32, 0i32);
    unsafe {
        shim_get_block_dimensions(
            bsize as i32,
            plane as i32,
            ss_x as i32,
            ss_y as i32,
            mb_to_right_edge,
            mb_to_bottom_edge,
            &mut width,
            &mut height,
            &mut rows,
            &mut cols,
        )
    };
    (
        width as usize,
        height as usize,
        rows as usize,
        cols as usize,
    )
}

/// The REAL exported `av1_decode_palette_tokens` (`av1/decoder/detokenize.c`), driven
/// end-to-end over a real `aom_reader` byte stream via a minimal MACROBLOCKD facade —
/// see `aom_entropy::partition::decode_color_map_tokens`, which this is the oracle for.
/// `map_cdf_in` is `REF_PALETTE_MAP_CDF_LEN` u16 (the plane's full colour-index CDF
/// array); returns `(color_map [REF_MAX_PALETTE_BLOCK_WIDTH * REF_MAX_PALETTE_BLOCK_HEIGHT],
/// map_cdf_out [REF_PALETTE_MAP_CDF_LEN])`. Panics on a non-zero shim rc.
#[allow(clippy::too_many_arguments)]
pub fn ref_decode_palette_tokens(
    data: &[u8],
    plane: usize,
    bsize: usize,
    n_colors: i32,
    ss_x: usize,
    ss_y: usize,
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    map_cdf_in: &[u16],
) -> (Vec<u8>, Vec<u16>) {
    assert_eq!(map_cdf_in.len(), REF_PALETTE_MAP_CDF_LEN);
    let mut color_map = vec![0u8; REF_MAX_PALETTE_BLOCK_WIDTH * REF_MAX_PALETTE_BLOCK_HEIGHT];
    let mut map_cdf_out = vec![0u16; REF_PALETTE_MAP_CDF_LEN];
    let rc = unsafe {
        shim_decode_palette_tokens(
            data.as_ptr(),
            data.len(),
            plane as i32,
            bsize as i32,
            n_colors,
            ss_x as i32,
            ss_y as i32,
            mb_to_right_edge,
            mb_to_bottom_edge,
            map_cdf_in.as_ptr(),
            map_cdf_out.as_mut_ptr(),
            color_map.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "shim_decode_palette_tokens failed ({rc})");
    (color_map, map_cdf_out)
}

// dec_shim.c section 4 (append-only addition): shim_encode_av1_kf_screen_content —
// same real encoder path as shim_encode_av1_kf, plus explicit enable_palette/
// enable_intrabc (AV1E_SET_ENABLE_PALETTE/_INTRABC — 0/1). The three pre-existing
// encode entry points (shim_encode_av1_kf/_sb128/_tiles) are UNCHANGED, still
// hardcoding both off.
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_screen_content(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        aq_mode: i32,
        two_pass: i32,
        enable_palette: i32,
        enable_intrabc: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// Encode one KEY frame through the REAL `aom_codec_av1_cx` encoder
/// (`--sb-size=64`, single tile — see `ref_encode_av1_kf`'s doc for the shared
/// control list) with explicit `--enable-palette=<enable_palette>
/// --enable-intrabc=<enable_intrabc>`. Panics on a negative shim return.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_screen_content(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
    aq_mode: u32,
    two_pass: bool,
    enable_palette: bool,
    enable_intrabc: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4 + 4096).max(65536)];
    let n = unsafe {
        shim_encode_av1_kf_screen_content(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            aq_mode as i32,
            two_pass as i32,
            enable_palette as i32,
            enable_intrabc as i32,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_screen_content failed ({n})");
    out.truncate(n as usize);
    out
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_lossless(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cpu_used: i32,
        usage: i32,
        two_pass: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// Encode one CODED-LOSSLESS KEY frame through the REAL `aom_codec_av1_cx`
/// encoder with `--lossless=1` (base_qindex 0, coded_lossless, ONLY_4X4 + the
/// 4x4 WHT transform). `--sb-size=64`, single tile, no palette / intrabc / cdef
/// / restoration. `usage` selects `AOM_USAGE_GOOD_QUALITY` (0) or
/// `AOM_USAGE_ALL_INTRA` (2). Panics on a negative shim return.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_lossless(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cpu_used: i32,
    usage: u32,
    two_pass: bool,
) -> Vec<u8> {
    // Lossless streams are larger than lossy — size the buffer generously.
    let mut out = vec![0u8; (w * h * 8 + 65536).max(1 << 20)];
    let n = unsafe {
        shim_encode_av1_kf_lossless(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cpu_used,
            usage as i32,
            two_pass as i32,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_lossless failed ({n})");
    out.truncate(n as usize);
    out
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_qm(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        aq_mode: i32,
        two_pass: i32,
        qm_min: i32,
        qm_max: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// Quantization-matrix variant of [`ref_encode_av1_kf`] (decoder-track QM-gate
/// work, append-only addition — every shim above is untouched): same params
/// plus `qm_min`/`qm_max` (`AV1E_SET_QM_MIN`/`AV1E_SET_QM_MAX`) with
/// `AV1E_SET_ENABLE_QM=1`. `--sb-size=64`, single tile, no palette / intrabc,
/// non-lossless. Setting `qm_min == qm_max == L` forces
/// `qmatrix_level_{y,u,v} = L` for every plane; pass a non-flat `L`
/// (`< NUM_QM_LEVELS - 1 = 15`) so the stream exercises a genuine QM. Panics on
/// a negative shim return.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_qm(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
    aq_mode: u32,
    two_pass: bool,
    qm_min: i32,
    qm_max: i32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_qm(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            aq_mode as i32,
            two_pass as i32,
            qm_min,
            qm_max,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_qm failed ({n})");
    out.truncate(n as usize);
    out
}

unsafe extern "C" {
    fn shim_encode_av1_kf_superres(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        superres_denom: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// Fixed-denominator superres variant of [`ref_encode_av1_kf`] (decoder-track
/// superres-gate work, append-only addition — every shim above is untouched):
/// `AV1E_SET_SUPERRES_MODE = AOM_SUPERRES_FIXED` +
/// `AV1E_SET_SUPERRES_DENOMINATOR = superres_denom` (9..=16). `w`/`h` are the
/// FULL (upscaled/display) dims fed to the encoder; it codes the frame at the
/// reduced width `(w*8 + denom/2)/denom` and the decoder upscales back to `w`
/// (horizontal only). `--sb-size=64`, single tile, deltaq/aq off, one-pass, no
/// palette / intrabc / qm / lossless. Panics on a negative shim return.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_superres(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
    superres_denom: i32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_superres(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            superres_denom,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_superres failed ({n})");
    out.truncate(n as usize);
    out
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_superres_mode(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        enable_cdef: i32,
        enable_restoration: i32,
        usage: i32,
        superres_mode: i32,
        superres_qthresh: i32,
        superres_kf_qthresh: i32,
        superres_denom: i32,
        superres_kf_denom: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// Derived-denominator superres variant of [`ref_encode_av1_kf_superres`]
/// (append-only; the FIXED wrapper above is untouched): the encoder chooses the
/// superres denominator itself via `calculate_next_superres_scale`.
/// `superres_mode` is the `aom_superres_mode` enum (2 = RANDOM, 3 = QTHRESH,
/// 4 = AUTO); `superres_qthresh`/`superres_kf_qthresh` are the 1..=63 CLI knobs
/// (`--superres-qthresh`/`--superres-kf-qthresh`); `superres_denom`/
/// `superres_kf_denom` are used only by AUTO_ALL. Same envelope as the FIXED
/// wrapper (--end-usage=q, --sb-size=64, single tile, one-pass). Panics on a
/// negative shim return.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_superres_mode(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    enable_cdef: bool,
    enable_restoration: bool,
    usage: u32,
    superres_mode: i32,
    superres_qthresh: i32,
    superres_kf_qthresh: i32,
    superres_denom: i32,
    superres_kf_denom: i32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_superres_mode(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            enable_cdef as i32,
            enable_restoration as i32,
            usage as i32,
            superres_mode,
            superres_qthresh,
            superres_kf_qthresh,
            superres_denom,
            superres_kf_denom,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_superres_mode failed ({n})");
    out.truncate(n as usize);
    out
}

// dec_shim.c section "intrabc DV prediction facades" (append-only addition):
// shim_find_dv_ref_mvs drives the REAL EXPORTED av1_find_mv_refs +
// av1_find_best_ref_mvs (ref_frame=INTRA_FRAME) over a synthetic MI grid;
// shim_find_ref_dv / shim_is_dv_valid facade the real `static inline`
// av1_find_ref_dv / av1_is_dv_valid (mvref_common.h) — see
// aom_entropy::dv_ref, which these are the oracles for.
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_find_dv_ref_mvs(
        mi_row: i32,
        mi_col: i32,
        bsize: i32,
        own_partition: i32,
        up_available: i32,
        left_available: i32,
        tile_mi_row_start: i32,
        tile_mi_row_end: i32,
        tile_mi_col_start: i32,
        tile_mi_col_end: i32,
        frame_mi_rows: i32,
        frame_mi_cols: i32,
        mib_size: i32,
        g_bsize: *const u8,
        g_ref_frame0: *const i8,
        g_ref_frame1: *const i8,
        g_use_intrabc: *const u8,
        g_mode: *const u8,
        g_mv0_row: *const i16,
        g_mv0_col: *const i16,
        g_mv1_row: *const i16,
        g_mv1_col: *const i16,
        out_nearest_row: *mut i32,
        out_nearest_col: *mut i32,
        out_near_row: *mut i32,
        out_near_col: *mut i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_find_inter_mv_refs(
        rf0: i32,
        mi_row: i32,
        mi_col: i32,
        bsize: i32,
        own_partition: i32,
        up_available: i32,
        left_available: i32,
        tile_mi_row_start: i32,
        tile_mi_row_end: i32,
        tile_mi_col_start: i32,
        tile_mi_col_end: i32,
        frame_mi_rows: i32,
        frame_mi_cols: i32,
        mib_size: i32,
        allow_ref_frame_mvs: i32,
        sign_bias: *const i8,
        allow_high_precision_mv: i32,
        is_integer_mv: i32,
        g_bsize: *const u8,
        g_ref_frame0: *const i8,
        g_ref_frame1: *const i8,
        g_use_intrabc: *const u8,
        g_mode: *const u8,
        g_mv0_row: *const i16,
        g_mv0_col: *const i16,
        g_mv1_row: *const i16,
        g_mv1_col: *const i16,
        out_mode_context: *mut i32,
        out_ref_mv_count: *mut i32,
        out_stack_row: *mut i32,
        out_stack_col: *mut i32,
        out_weight: *mut i32,
        out_nearest_row: *mut i32,
        out_nearest_col: *mut i32,
        out_near_row: *mut i32,
        out_near_col: *mut i32,
        out_global_row: *mut i32,
        out_global_col: *mut i32,
    );
    fn shim_find_ref_dv(
        mi_row: i32,
        mib_size: i32,
        tile_mi_row_start: i32,
        out_row: *mut i32,
        out_col: *mut i32,
    );
    #[allow(clippy::too_many_arguments)]
    fn shim_is_dv_valid(
        dv_row: i32,
        dv_col: i32,
        mi_row: i32,
        mi_col: i32,
        bsize: i32,
        tile_mi_row_start: i32,
        tile_mi_row_end: i32,
        tile_mi_col_start: i32,
        tile_mi_col_end: i32,
        mib_size_log2: i32,
        is_chroma_ref: i32,
        num_planes: i32,
        ss_x: i32,
        ss_y: i32,
    ) -> i32;
}

/// Window size for [`ref_find_dv_ref_mvs`]'s synthetic MI grid — MUST match
/// `DV_GRID_DIM` in `dec_shim.c`. `mi_row`/`mi_col` and `bsize` must place the
/// current block entirely within `[0, REF_DV_GRID_DIM)` on both axes with
/// margin for the scan's reach (see `dv_ref_diff.rs`).
pub const REF_DV_GRID_DIM: usize = 128;

/// One synthetic neighbour cell for [`ref_find_dv_ref_mvs`]'s flat grid input.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefDvNbr {
    pub bsize: u8,
    pub ref_frame0: i8,
    pub ref_frame1: i8,
    pub use_intrabc: bool,
    pub mode: u8,
    pub mv0_row: i16,
    pub mv0_col: i16,
    pub mv1_row: i16,
    pub mv1_col: i16,
}

/// The REAL exported `av1_find_mv_refs(ref_frame=INTRA_FRAME)` +
/// `av1_find_best_ref_mvs`, driven over a synthetic
/// `REF_DV_GRID_DIM x REF_DV_GRID_DIM` MI grid (row-major, `grid[row *
/// REF_DV_GRID_DIM + col]`) — see `aom_entropy::dv_ref::find_dv_ref_mvs`,
/// which this is the oracle for. Returns `(nearest_row, nearest_col,
/// near_row, near_col)` in 1/8-pel units.
#[allow(clippy::too_many_arguments)]
pub fn ref_find_dv_ref_mvs(
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    own_partition: usize,
    up_available: bool,
    left_available: bool,
    tile_mi_row_start: i32,
    tile_mi_row_end: i32,
    tile_mi_col_start: i32,
    tile_mi_col_end: i32,
    frame_mi_rows: i32,
    frame_mi_cols: i32,
    mib_size: i32,
    grid: &[RefDvNbr],
) -> (i32, i32, i32, i32) {
    assert_eq!(grid.len(), REF_DV_GRID_DIM * REF_DV_GRID_DIM);
    let n = grid.len();
    let mut g_bsize = Vec::with_capacity(n);
    let mut g_ref_frame0 = Vec::with_capacity(n);
    let mut g_ref_frame1 = Vec::with_capacity(n);
    let mut g_use_intrabc = Vec::with_capacity(n);
    let mut g_mode = Vec::with_capacity(n);
    let mut g_mv0_row = Vec::with_capacity(n);
    let mut g_mv0_col = Vec::with_capacity(n);
    let mut g_mv1_row = Vec::with_capacity(n);
    let mut g_mv1_col = Vec::with_capacity(n);
    for c in grid {
        g_bsize.push(c.bsize);
        g_ref_frame0.push(c.ref_frame0);
        g_ref_frame1.push(c.ref_frame1);
        g_use_intrabc.push(c.use_intrabc as u8);
        g_mode.push(c.mode);
        g_mv0_row.push(c.mv0_row);
        g_mv0_col.push(c.mv0_col);
        g_mv1_row.push(c.mv1_row);
        g_mv1_col.push(c.mv1_col);
    }
    let (mut nr, mut nc, mut rr, mut rc) = (0i32, 0i32, 0i32, 0i32);
    unsafe {
        shim_find_dv_ref_mvs(
            mi_row,
            mi_col,
            bsize as i32,
            own_partition as i32,
            up_available as i32,
            left_available as i32,
            tile_mi_row_start,
            tile_mi_row_end,
            tile_mi_col_start,
            tile_mi_col_end,
            frame_mi_rows,
            frame_mi_cols,
            mib_size,
            g_bsize.as_ptr(),
            g_ref_frame0.as_ptr(),
            g_ref_frame1.as_ptr(),
            g_use_intrabc.as_ptr(),
            g_mode.as_ptr(),
            g_mv0_row.as_ptr(),
            g_mv0_col.as_ptr(),
            g_mv1_row.as_ptr(),
            g_mv1_col.as_ptr(),
            &mut nr,
            &mut nc,
            &mut rr,
            &mut rc,
        );
    }
    (nr, nc, rr, rc)
}

/// The C-side output surface of [`ref_find_inter_mv_refs`] — mirrors the port's
/// `aom_entropy::dv_ref::InterMvRefs` (C `uint16_t` weight / `uint8_t` count
/// widened to `i32` for the FFI boundary). Only `[0, ref_mv_count)` of
/// `stack`/`weight` is meaningful.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefInterMvRefs {
    pub mode_context: i32,
    pub ref_mv_count: i32,
    pub stack: [(i32, i32); 8],
    pub weight: [i32; 8],
    pub nearest: (i32, i32),
    pub near: (i32, i32),
    pub global_mv: (i32, i32),
}

/// The REAL exported `av1_find_mv_refs` at a SINGLE inter reference `rf0` +
/// `av1_find_best_ref_mvs`, driven over the same synthetic
/// `REF_DV_GRID_DIM x REF_DV_GRID_DIM` grid as [`ref_find_dv_ref_mvs`] — the
/// oracle for `aom_entropy::dv_ref::find_inter_mv_refs`. Global motion is
/// IDENTITY; `allow_ref_frame_mvs` runs against an all-INVALID `tpl_mvs` (the
/// empty-motion-field envelope the port models).
#[allow(clippy::too_many_arguments)]
pub fn ref_find_inter_mv_refs(
    rf0: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    own_partition: usize,
    up_available: bool,
    left_available: bool,
    tile_mi_row_start: i32,
    tile_mi_row_end: i32,
    tile_mi_col_start: i32,
    tile_mi_col_end: i32,
    frame_mi_rows: i32,
    frame_mi_cols: i32,
    mib_size: i32,
    allow_ref_frame_mvs: bool,
    sign_bias: [i8; 8],
    allow_high_precision_mv: bool,
    is_integer_mv: bool,
    grid: &[RefDvNbr],
) -> RefInterMvRefs {
    assert_eq!(grid.len(), REF_DV_GRID_DIM * REF_DV_GRID_DIM);
    let n = grid.len();
    let mut g_bsize = Vec::with_capacity(n);
    let mut g_ref_frame0 = Vec::with_capacity(n);
    let mut g_ref_frame1 = Vec::with_capacity(n);
    let mut g_use_intrabc = Vec::with_capacity(n);
    let mut g_mode = Vec::with_capacity(n);
    let mut g_mv0_row = Vec::with_capacity(n);
    let mut g_mv0_col = Vec::with_capacity(n);
    let mut g_mv1_row = Vec::with_capacity(n);
    let mut g_mv1_col = Vec::with_capacity(n);
    for c in grid {
        g_bsize.push(c.bsize);
        g_ref_frame0.push(c.ref_frame0);
        g_ref_frame1.push(c.ref_frame1);
        g_use_intrabc.push(c.use_intrabc as u8);
        g_mode.push(c.mode);
        g_mv0_row.push(c.mv0_row);
        g_mv0_col.push(c.mv0_col);
        g_mv1_row.push(c.mv1_row);
        g_mv1_col.push(c.mv1_col);
    }
    let mut mode_context = 0i32;
    let mut ref_mv_count = 0i32;
    let mut stack_row = [0i32; 8];
    let mut stack_col = [0i32; 8];
    let mut weight = [0i32; 8];
    let (mut nr, mut nc, mut rr, mut rc) = (0i32, 0i32, 0i32, 0i32);
    let (mut gr, mut gc) = (0i32, 0i32);
    unsafe {
        shim_find_inter_mv_refs(
            rf0,
            mi_row,
            mi_col,
            bsize as i32,
            own_partition as i32,
            up_available as i32,
            left_available as i32,
            tile_mi_row_start,
            tile_mi_row_end,
            tile_mi_col_start,
            tile_mi_col_end,
            frame_mi_rows,
            frame_mi_cols,
            mib_size,
            allow_ref_frame_mvs as i32,
            sign_bias.as_ptr(),
            allow_high_precision_mv as i32,
            is_integer_mv as i32,
            g_bsize.as_ptr(),
            g_ref_frame0.as_ptr(),
            g_ref_frame1.as_ptr(),
            g_use_intrabc.as_ptr(),
            g_mode.as_ptr(),
            g_mv0_row.as_ptr(),
            g_mv0_col.as_ptr(),
            g_mv1_row.as_ptr(),
            g_mv1_col.as_ptr(),
            &mut mode_context,
            &mut ref_mv_count,
            stack_row.as_mut_ptr(),
            stack_col.as_mut_ptr(),
            weight.as_mut_ptr(),
            &mut nr,
            &mut nc,
            &mut rr,
            &mut rc,
            &mut gr,
            &mut gc,
        );
    }
    let mut stack = [(0i32, 0i32); 8];
    for i in 0..8 {
        stack[i] = (stack_row[i], stack_col[i]);
    }
    RefInterMvRefs {
        mode_context,
        ref_mv_count,
        stack,
        weight,
        nearest: (nr, nc),
        near: (rr, rc),
        global_mv: (gr, gc),
    }
}

/// The REAL `static inline` `av1_find_ref_dv` (`mvref_common.h`) — see
/// `aom_entropy::dv_ref::find_ref_dv`, which this is the oracle for.
pub fn ref_find_ref_dv(mi_row: i32, mib_size: i32, tile_mi_row_start: i32) -> (i32, i32) {
    let (mut row, mut col) = (0i32, 0i32);
    unsafe { shim_find_ref_dv(mi_row, mib_size, tile_mi_row_start, &mut row, &mut col) };
    (row, col)
}

/// The REAL `static inline` `av1_is_dv_valid` (`mvref_common.h`) — see
/// `aom_entropy::dv_ref::is_dv_valid`, which this is the oracle for.
#[allow(clippy::too_many_arguments)]
pub fn ref_is_dv_valid(
    dv_row: i32,
    dv_col: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    tile_mi_row_start: i32,
    tile_mi_row_end: i32,
    tile_mi_col_start: i32,
    tile_mi_col_end: i32,
    mib_size_log2: i32,
    is_chroma_ref: bool,
    num_planes: i32,
    ss_x: i32,
    ss_y: i32,
) -> bool {
    let rc = unsafe {
        shim_is_dv_valid(
            dv_row,
            dv_col,
            mi_row,
            mi_col,
            bsize as i32,
            tile_mi_row_start,
            tile_mi_row_end,
            tile_mi_col_start,
            tile_mi_col_end,
            mib_size_log2,
            is_chroma_ref as i32,
            num_planes,
            ss_x,
            ss_y,
        )
    };
    rc != 0
}

extern "C" {
    fn shim_dump_default_inter_ext_tx(base_qindex: i32, out: *mut u16) -> i32;
}

/// Length of the `inter_ext_tx_cdf[EXT_TX_SETS_INTER][EXT_TX_SIZES][CDF_SIZE(TX_TYPES)]`
/// dump: 4 * 4 * 17 = 272 u16.
pub const DUMP_INTER_EXT_TX_LEN: usize = 4 * 4 * 17;

/// Dump the compiled default `fc->inter_ext_tx_cdf` (the full padded
/// `[4][4][17]` table) from the REAL `av1_setup_past_independence` default
/// frame context. Verifies aom-entropy's `DEFAULT_INTER_EXT_TX`.
pub fn ref_dump_default_inter_ext_tx(base_qindex: i32) -> Vec<u16> {
    let mut out = vec![0u16; DUMP_INTER_EXT_TX_LEN];
    let rc = unsafe { shim_dump_default_inter_ext_tx(base_qindex, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "shim_dump_default_inter_ext_tx failed ({rc})");
    out
}

// ---------------------------------------------------------------------------
// dec_shim.c (append-only addition): shim_add_film_grain — the REAL exported
// av1_add_film_grain (av1/decoder/grain_synthesis.c) as the film-grain
// synthesis oracle. Everything above is UNCHANGED.
// ---------------------------------------------------------------------------
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_add_film_grain(
        blob: *const i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        mc_identity: i32,
        d_w: i32,
        d_h: i32,
        y: *const u16,
        u: *const u16,
        v: *const u16,
        out_y: *mut u16,
        out_u: *mut u16,
        out_v: *mut u16,
    ) -> i32;
}

/// Number of `i32` in a packed film-grain params blob (see `shim_add_film_grain`
/// / `fill_grain_params` in dec_shim.c for the field order).
pub const FILM_GRAIN_BLOB_LEN: usize = 160;

/// Reference film-grain synthesis via the REAL exported `av1_add_film_grain`
/// (`av1/decoder/grain_synthesis.c`). `blob` is the packed `aom_film_grain_t`
/// (`FILM_GRAIN_BLOB_LEN` ints; layout mirrors dec_shim.c `fill_grain_params`).
/// Planes are `u16` row-major tight; `mono` -> empty chroma (only Y grained).
/// Returns the grained `(y, u, v)`.
#[allow(clippy::too_many_arguments)]
pub fn ref_add_film_grain(
    blob: &[i32],
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    mc_identity: bool,
    d_w: usize,
    d_h: usize,
    y: &[u16],
    u: &[u16],
    v: &[u16],
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    assert_eq!(blob.len(), FILM_GRAIN_BLOB_LEN, "grain blob length");
    let cw = if mono {
        0
    } else {
        (d_w + ss_x as usize) >> ss_x
    };
    let ch = if mono {
        0
    } else {
        (d_h + ss_y as usize) >> ss_y
    };
    let mut out_y = vec![0u16; d_w * d_h];
    let mut out_u = vec![0u16; cw * ch];
    let mut out_v = vec![0u16; cw * ch];
    let rc = unsafe {
        shim_add_film_grain(
            blob.as_ptr(),
            bd,
            mono as i32,
            ss_x,
            ss_y,
            mc_identity as i32,
            d_w as i32,
            d_h as i32,
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            out_y.as_mut_ptr(),
            out_u.as_mut_ptr(),
            out_v.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "shim_add_film_grain failed ({rc})");
    (out_y, out_u, out_v)
}

// ---------------------------------------------------------------------------
// dec_shim.c (append-only addition): shim_encode_av1_kf_film_grain — the REAL
// aom_codec_av1_cx public API with AV1E_SET_FILM_GRAIN_TEST_VECTOR set, to
// produce a KEY stream carrying film_grain_params_present=1 + per-frame grain
// params. Everything above is UNCHANGED.
// ---------------------------------------------------------------------------
extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_film_grain(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        usage: i32,
        grain_test_vector: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// Encode a single KEY frame WITH film grain via the REAL `aom_codec_av1_cx`
/// (dec_shim.c `shim_encode_av1_kf_film_grain`): sets
/// `AV1E_SET_FILM_GRAIN_TEST_VECTOR = grain_test_vector` (1..=16, an index into
/// libaom's built-in `film_grain_test_vectors[]`), so the stream carries
/// `film_grain_params_present=1` + per-frame grain params. Planes are `u16`
/// row-major tight. Returns the bitstream bytes.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_film_grain(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    usage: u32,
    grain_test_vector: i32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_film_grain(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            usage as i32,
            grain_test_vector,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_film_grain failed ({n})");
    out.truncate(n as usize);
    out
}

// ---------------------------------------------------------------------------
// dec_shim.c (append-only): the --film-grain-table (AV1E_SET_FILM_GRAIN_TABLE)
// path. `shim_write_grain_table_test_vector` serializes a built-in test vector
// to a canonical `filmgrn1` file via the REAL aom_film_grain_table_write;
// `shim_encode_av1_kf_film_grain_table` encodes a KEY frame reading grain params
// from that file. Both feed the C7 table-inject byte-exactness gate.
// ---------------------------------------------------------------------------
unsafe extern "C" {
    fn shim_write_grain_table_test_vector(idx: i32, path: *const core::ffi::c_char) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_film_grain_table(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        usage: i32,
        table_path: *const core::ffi::c_char,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// Write libaom's built-in `film_grain_test_vectors[idx-1]` (`idx` in `1..=16`)
/// to `path` as a canonical `filmgrn1` grain-table file, via the REAL
/// `aom_film_grain_table_write` (dec_shim.c `shim_write_grain_table_test_vector`).
/// The single entry spans `[0, i64::MAX)`. Panics on error.
pub fn ref_write_grain_table_test_vector(idx: i32, path: &std::path::Path) {
    let cpath = std::ffi::CString::new(path.as_os_str().to_str().expect("utf8 path"))
        .expect("path has interior NUL");
    let rc = unsafe { shim_write_grain_table_test_vector(idx, cpath.as_ptr()) };
    assert_eq!(
        rc, 0,
        "shim_write_grain_table_test_vector({idx}) failed ({rc})"
    );
}

/// Encode a single KEY frame WITH a film-grain TABLE via the REAL
/// `aom_codec_av1_cx` (dec_shim.c `shim_encode_av1_kf_film_grain_table`): sets
/// `AV1E_SET_FILM_GRAIN_TABLE = table_path` (the `--film-grain-table` path), so
/// the stream carries `film_grain_params_present=1` + the per-frame grain params
/// looked up from the file. Planes are `u16` row-major tight. Returns the bytes.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_film_grain_table(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    usage: u32,
    table_path: &std::path::Path,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let cpath = std::ffi::CString::new(table_path.as_os_str().to_str().expect("utf8 path"))
        .expect("path has interior NUL");
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_film_grain_table(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            usage as i32,
            cpath.as_ptr(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_film_grain_table failed ({n})");
    out.truncate(n as usize);
    out
}

// ---------------------------------------------------------------------------
// dec_shim.c (append-only): shim_encode_av1_kf_defaults — a genuinely FLAGLESS
// allintra KEY encode (config_default(usage) + the fixed-Q / sb64 / single-tile
// operating point, NO coding-tool controls). For usage=ALL_INTRA the tools sit
// at their true defaults — cdef OFF, loop-restoration ON, qm OFF — i.e. exactly
// what a plain `aomenc --allintra` produces. The DEFAULT-parity reference.
// ---------------------------------------------------------------------------
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_encode_av1_kf_defaults(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        usage: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// A plain `aomenc --allintra --end-usage=q --cq-level=N --cpu-used=M` encode of
/// one KEY frame via the REAL `aom_codec_av1_cx` (dec_shim.c
/// `shim_encode_av1_kf_defaults`): every coding-TOOL control is left at its
/// ALLINTRA default (cdef OFF, **loop-restoration ON**, qm OFF, palette/intrabc
/// at config defaults), so this is the true-default stream the port's default
/// path must byte-match. Only the fixed-Q operating point + the SB64 / single-
/// tile envelope are set. Panics on error.
pub fn ref_encode_av1_kf_defaults(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    usage: u32,
) -> Vec<u8> {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
    };
    assert_eq!(y.len(), w * h);
    assert!(mono || (u.len() == cw * ch && v.len() == cw * ch));
    let mut out = vec![0u8; w * h * 8 + 65536];
    let n = unsafe {
        shim_encode_av1_kf_defaults(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            usage as i32,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_defaults failed ({n})");
    out.truncate(n as usize);
    out
}

// ---------------------------------------------------------------------------
// dec_shim.c (append-only): noise-strength solver differential oracle (C7
// grain-estimator chunk 2). Drives the REAL exported aom_noise_strength_solver_*
// over (mean,std) observations; the Rust port must reproduce the solved curve /
// fitted LUT bit-for-bit.
// ---------------------------------------------------------------------------
unsafe extern "C" {
    fn shim_noise_strength_solve(
        means: *const f64,
        stds: *const f64,
        nobs: i32,
        num_bins: i32,
        bit_depth: i32,
        out_x: *mut f64,
    ) -> i32;
    fn shim_noise_strength_fit_piecewise(
        means: *const f64,
        stds: *const f64,
        nobs: i32,
        num_bins: i32,
        bit_depth: i32,
        max_points: i32,
        out_points_xy: *mut f64,
        out_num_points: *mut i32,
    ) -> i32;
}

/// Run the REAL `aom_noise_strength_solver_*` over `(means, stds)` observations
/// and return the solved per-bin strength curve (`solver.eqns.x`, `num_bins`
/// values), or `None` if the solve failed (singular).
pub fn ref_noise_strength_solve(
    means: &[f64],
    stds: &[f64],
    num_bins: usize,
    bit_depth: i32,
) -> Option<Vec<f64>> {
    assert_eq!(means.len(), stds.len());
    let mut out = vec![0.0f64; num_bins];
    let ok = unsafe {
        shim_noise_strength_solve(
            means.as_ptr(),
            stds.as_ptr(),
            means.len() as i32,
            num_bins as i32,
            bit_depth,
            out.as_mut_ptr(),
        )
    };
    (ok != 0).then_some(out)
}

/// Run the REAL solver + `aom_noise_strength_solver_fit_piecewise` and return
/// the reduced LUT points, or `None` on failure.
pub fn ref_noise_strength_fit_piecewise(
    means: &[f64],
    stds: &[f64],
    num_bins: usize,
    bit_depth: i32,
    max_points: i32,
) -> Option<Vec<[f64; 2]>> {
    assert_eq!(means.len(), stds.len());
    let mut xy = vec![0.0f64; num_bins * 2];
    let mut n = 0i32;
    let ok = unsafe {
        shim_noise_strength_fit_piecewise(
            means.as_ptr(),
            stds.as_ptr(),
            means.len() as i32,
            num_bins as i32,
            bit_depth,
            max_points,
            xy.as_mut_ptr(),
            &mut n,
        )
    };
    (ok != 0).then(|| {
        (0..n as usize)
            .map(|i| [xy[2 * i], xy[2 * i + 1]])
            .collect()
    })
}

unsafe extern "C" {
    fn shim_flat_block_finder_run(
        pixels: *const u16,
        w: i32,
        h: i32,
        block_size: i32,
        bit_depth: i32,
        use_highbd: i32,
        out_flat: *mut u8,
    ) -> i32;
}

/// Run the REAL `aom_flat_block_finder_init` + `_run` over a `w×h` plane
/// (`pixels`, row-major, stride `w`) and return `(flat_blocks_map, num_flat)`.
/// The map is `num_blocks_w * num_blocks_h` bytes. `use_highbd` selects the
/// pixel read width on the C side (bit_depth drives normalization). `None` on
/// C error.
pub fn ref_flat_block_finder_run(
    pixels: &[u16],
    w: usize,
    h: usize,
    block_size: usize,
    bit_depth: i32,
    use_highbd: bool,
) -> Option<(Vec<u8>, i32)> {
    assert_eq!(pixels.len(), w * h);
    let nbw = w.div_ceil(block_size);
    let nbh = h.div_ceil(block_size);
    let mut out = vec![0u8; nbw * nbh];
    let num_flat = unsafe {
        shim_flat_block_finder_run(
            pixels.as_ptr(),
            w as i32,
            h as i32,
            block_size as i32,
            bit_depth,
            use_highbd as i32,
            out.as_mut_ptr(),
        )
    };
    (num_flat >= 0).then_some((out, num_flat))
}

unsafe extern "C" {
    fn shim_noise_fft2d(block_size: i32, input: *const f32, out: *mut f32) -> i32;
    fn shim_noise_ifft2d(block_size: i32, input: *const f32, out: *mut f32) -> i32;
    fn shim_noise_tx_pipeline(
        block_size: i32,
        data: *const f32,
        psd: *const f32,
        out_denoised: *mut f32,
        out_energy: *mut f32,
    ) -> i32;
}

/// Run the REAL RTCD-dispatched `aom_fftNxN_float` over an `n×n` real `input`
/// and return the packed `2·n·n` `[re, im]` spectrum. `None` for an unsupported
/// block size. The dispatched impl (SSE2/AVX2, non-fused) is bit-identical to
/// the scalar path `aom_encode::noise_fft::fft2d` mirrors.
pub fn ref_noise_fft2d(block_size: usize, input: &[f32]) -> Option<Vec<f32>> {
    assert_eq!(input.len(), block_size * block_size);
    let mut out = vec![0.0f32; 2 * block_size * block_size];
    let ok = unsafe { shim_noise_fft2d(block_size as i32, input.as_ptr(), out.as_mut_ptr()) };
    (ok != 0).then_some(out)
}

/// Run the REAL RTCD-dispatched `aom_ifftNxN_float` over a packed `2·n·n`
/// `input` and return the `n×n` real output. `None` for an unsupported size.
pub fn ref_noise_ifft2d(block_size: usize, input: &[f32]) -> Option<Vec<f32>> {
    assert_eq!(input.len(), 2 * block_size * block_size);
    let mut out = vec![0.0f32; block_size * block_size];
    let ok = unsafe { shim_noise_ifft2d(block_size as i32, input.as_ptr(), out.as_mut_ptr()) };
    (ok != 0).then_some(out)
}

/// Run the REAL `aom_noise_tx_*` pipeline (forward → add_energy → filter →
/// inverse) over one `n×n` block. Returns `(denoised, energy)`, each `n×n`.
/// Mirrors `aom_encode::noise_fft::NoiseTx`. `None` for an unsupported size.
pub fn ref_noise_tx_pipeline(
    block_size: usize,
    data: &[f32],
    psd: &[f32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    let n = block_size * block_size;
    assert_eq!(data.len(), n);
    assert_eq!(psd.len(), n);
    let mut denoised = vec![0.0f32; n];
    let mut energy = vec![0.0f32; n];
    let ok = unsafe {
        shim_noise_tx_pipeline(
            block_size as i32,
            data.as_ptr(),
            psd.as_ptr(),
            denoised.as_mut_ptr(),
            energy.as_mut_ptr(),
        )
    };
    (ok != 0).then_some((denoised, energy))
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_noise_model_fit(
        shape: i32,
        lag: i32,
        bit_depth: i32,
        use_highbd: i32,
        y: *const u16,
        u: *const u16,
        v: *const u16,
        dy: *const u16,
        du: *const u16,
        dv: *const u16,
        w: i32,
        h: i32,
        sy: i32,
        su: i32,
        sv: i32,
        csx: i32,
        csy: i32,
        flat_blocks: *const u8,
        block_size: i32,
        out_status: *mut i32,
        out_n: *mut i32,
        out_ar_x: *mut f64,
        out_ar_gain: *mut f64,
        out_strength_x: *mut f64,
        random_seed: i32,
        table_path: *const core::ffi::c_char,
    ) -> i32;
}

/// Result of the REAL `aom_noise_model_update` (+ optional
/// `get_grain_parameters`) — see [`ref_noise_model_fit`].
pub struct NoiseModelFit {
    /// `aom_noise_status_t` as an int (0 = OK, 3 = DIFFERENT_NOISE_TYPE, ...).
    pub status: i32,
    /// `combined_state[c].eqns.n` per channel.
    pub n: [usize; 3],
    /// `combined_state[c].eqns.x` (the fitted AR coeffs), `3 × 32` row-major.
    pub ar_x: Vec<f64>,
    /// `combined_state[c].ar_gain` per channel.
    pub ar_gain: [f64; 3],
    /// `combined_state[c].strength_solver.eqns.x`, `3 × 20` row-major.
    pub strength_x: Vec<f64>,
    /// The serialized `filmgrn1` grain table from `get_grain_parameters`, if
    /// requested and the update succeeded.
    pub grain_table: Option<Vec<u8>>,
}

/// Run the REAL `aom_noise_model_init` + `aom_noise_model_update` over a
/// synthetic frame (`data`/`denoised`, 3 planes of `u16`, `strides` in `u16`
/// units; `!use_highbd` truncates to 8-bit on the C side) and return the fitted
/// combined-state AR coefficients, ar_gain, solved strength curves, and update
/// status. When `write_grain`, also runs `get_grain_parameters` and returns the
/// serialized grain table for byte comparison.
#[allow(clippy::too_many_arguments)]
pub fn ref_noise_model_fit(
    shape: i32,
    lag: i32,
    bit_depth: i32,
    use_highbd: bool,
    data: [&[u16]; 3],
    denoised: [&[u16]; 3],
    w: usize,
    h: usize,
    strides: [usize; 3],
    csx: i32,
    csy: i32,
    flat_blocks: &[u8],
    block_size: usize,
    random_seed: i32,
    write_grain: bool,
) -> NoiseModelFit {
    let mut status = 0i32;
    let mut n = [0i32; 3];
    let mut ar_x = vec![0.0f64; 3 * 32];
    let mut ar_gain = [0.0f64; 3];
    let mut strength_x = vec![0.0f64; 3 * 20];

    let table_path = if write_grain {
        Some(std::env::temp_dir().join(format!(
            "aomrs_nm_grain_{}_{}.tbl",
            std::process::id(),
            random_seed
        )))
    } else {
        None
    };
    let cpath = table_path
        .as_ref()
        .map(|p| std::ffi::CString::new(p.to_str().expect("utf8 path")).expect("path NUL"));
    let cpath_ptr = cpath.as_ref().map_or(core::ptr::null(), |c| c.as_ptr());
    let _ = table_path.as_ref().map(std::fs::remove_file); // clear stale

    let ok = unsafe {
        shim_noise_model_fit(
            shape,
            lag,
            bit_depth,
            use_highbd as i32,
            data[0].as_ptr(),
            data[1].as_ptr(),
            data[2].as_ptr(),
            denoised[0].as_ptr(),
            denoised[1].as_ptr(),
            denoised[2].as_ptr(),
            w as i32,
            h as i32,
            strides[0] as i32,
            strides[1] as i32,
            strides[2] as i32,
            csx,
            csy,
            flat_blocks.as_ptr(),
            block_size as i32,
            &mut status,
            n.as_mut_ptr(),
            ar_x.as_mut_ptr(),
            ar_gain.as_mut_ptr(),
            strength_x.as_mut_ptr(),
            random_seed,
            cpath_ptr,
        )
    };
    assert_eq!(ok, 1, "shim_noise_model_fit failed (status={status})");
    let grain_table = table_path.as_ref().and_then(|p| {
        let b = std::fs::read(p).ok();
        let _ = std::fs::remove_file(p);
        b
    });
    NoiseModelFit {
        status,
        n: [n[0] as usize, n[1] as usize, n[2] as usize],
        ar_x,
        ar_gain,
        strength_x,
        grain_table,
    }
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_wiener_denoise_2d(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        sy: i32,
        su: i32,
        sv: i32,
        csx: i32,
        csy: i32,
        psd_y: *const f32,
        psd_u: *const f32,
        psd_v: *const f32,
        block_size: i32,
        bit_depth: i32,
        use_highbd: i32,
        out_y: *mut u16,
        out_u: *mut u16,
        out_v: *mut u16,
    ) -> i32;
}

/// Run the REAL `aom_wiener_denoise_2d` over a synthetic frame (3 planes of
/// `u16`; `!use_highbd` truncates to 8-bit) with flat noise PSDs, and return the
/// denoised planes (`u16`). `None` if the C call reports failure.
#[allow(clippy::too_many_arguments)]
pub fn ref_wiener_denoise_2d(
    data: [&[u16]; 3],
    w: usize,
    h: usize,
    strides: [usize; 3],
    csx: i32,
    csy: i32,
    psd: [&[f32]; 3],
    block_size: usize,
    bit_depth: i32,
    use_highbd: bool,
    plane_lens: [usize; 3],
) -> Option<[Vec<u16>; 3]> {
    let mut out = [
        vec![0u16; plane_lens[0]],
        vec![0u16; plane_lens[1]],
        vec![0u16; plane_lens[2]],
    ];
    let (o0, rest) = out.split_at_mut(1);
    let (o1, o2) = rest.split_at_mut(1);
    let ok = unsafe {
        shim_wiener_denoise_2d(
            data[0].as_ptr(),
            data[1].as_ptr(),
            data[2].as_ptr(),
            w as i32,
            h as i32,
            strides[0] as i32,
            strides[1] as i32,
            strides[2] as i32,
            csx,
            csy,
            psd[0].as_ptr(),
            psd[1].as_ptr(),
            psd[2].as_ptr(),
            block_size as i32,
            bit_depth,
            use_highbd as i32,
            o0[0].as_mut_ptr(),
            o1[0].as_mut_ptr(),
            o2[0].as_mut_ptr(),
        )
    };
    (ok != 0).then_some(out)
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn shim_denoise_and_model_run(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        ss_x: i32,
        ss_y: i32,
        bit_depth: i32,
        block_size: i32,
        noise_level: f32,
        random_seed: i32,
        table_path: *const core::ffi::c_char,
        out_den_y: *mut u16,
        out_den_u: *mut u16,
        out_den_v: *mut u16,
        out_apply_grain: *mut i32,
    ) -> i32;
}

/// Result of the REAL `aom_denoise_and_model_run` (via a YV12 buffer) — the
/// serialized grain table and the tight denoised planes. See
/// [`ref_denoise_and_model_run`].
pub struct DenoiseAndModelResult {
    pub apply_grain: bool,
    pub grain_table: Option<Vec<u8>>,
    pub denoised: [Vec<u16>; 3],
}

/// Run the REAL end-to-end `aom_denoise_and_model_run` over tight `u16` planes
/// (32-aligned dims, 4:2:0/4:4:4) and return the estimated grain table + the
/// denoised planes. Mirrors [`aom_encode::denoise::estimate_film_grain`].
#[allow(clippy::too_many_arguments)]
pub fn ref_denoise_and_model_run(
    data: [&[u16]; 3],
    w: usize,
    h: usize,
    ss_x: i32,
    ss_y: i32,
    bit_depth: i32,
    block_size: usize,
    noise_level: f32,
    random_seed: i32,
) -> DenoiseAndModelResult {
    let cw = w >> ss_x;
    let ch = h >> ss_y;
    let mut den = [vec![0u16; w * h], vec![0u16; cw * ch], vec![0u16; cw * ch]];
    let mut apply_grain = 0i32;
    let path = std::env::temp_dir().join(format!(
        "aomrs_dam_{}_{}.tbl",
        std::process::id(),
        random_seed
    ));
    let _ = std::fs::remove_file(&path);
    let cpath = std::ffi::CString::new(path.to_str().expect("utf8 path")).expect("NUL");
    let (d0, rest) = den.split_at_mut(1);
    let (d1, d2) = rest.split_at_mut(1);
    let ok = unsafe {
        shim_denoise_and_model_run(
            data[0].as_ptr(),
            data[1].as_ptr(),
            data[2].as_ptr(),
            w as i32,
            h as i32,
            ss_x,
            ss_y,
            bit_depth,
            block_size as i32,
            noise_level,
            random_seed,
            cpath.as_ptr(),
            d0[0].as_mut_ptr(),
            d1[0].as_mut_ptr(),
            d2[0].as_mut_ptr(),
            &mut apply_grain,
        )
    };
    let grain_table = if ok != 0 {
        let b = std::fs::read(&path).ok();
        let _ = std::fs::remove_file(&path);
        b
    } else {
        None
    };
    DenoiseAndModelResult {
        apply_grain: apply_grain != 0,
        grain_table,
        denoised: den,
    }
}

/// The REAL exported `aom_count_primitive_refsubexpfin`
/// (aom_dsp/binary_codes_writer.c): coded bit count of
/// `aom_write_primitive_refsubexpfin(n, k, ref, v)`.
pub fn ref_count_primitive_refsubexpfin(n: u16, k: u16, r: u16, v: u16) -> i32 {
    unsafe { aom_count_primitive_refsubexpfin(n, k, r, v) }
}

// ---------------------------------------------------------------------------
// Loop-restoration ENCODER-SEARCH oracles (pickrst_shim.c): thin wrappers
// over the EXPORTED `_c` numeric core of av1/encoder/pickrst.c.
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn shim_compute_stats(
        wiener_win: i32,
        dgd: *const u8,
        src: *const u8,
        h_start: i32,
        h_end: i32,
        v_start: i32,
        v_end: i32,
        dgd_stride: i32,
        src_stride: i32,
        m: *mut i64,
        h: *mut i64,
        use_downsampled_wiener_stats: i32,
    );
    fn shim_compute_stats_highbd(
        wiener_win: i32,
        dgd: *const u16,
        src: *const u16,
        h_start: i32,
        h_end: i32,
        v_start: i32,
        v_end: i32,
        dgd_stride: i32,
        src_stride: i32,
        m: *mut i64,
        h: *mut i64,
        bit_depth: i32,
    );
    fn shim_pixel_proj_error(
        src8: *const u8,
        src16: *const u16,
        width: i32,
        height: i32,
        src_stride: i32,
        dat8: *const u8,
        dat16: *const u16,
        dat_stride: i32,
        flt0: *mut i32,
        flt0_stride: i32,
        flt1: *mut i32,
        flt1_stride: i32,
        xq: *mut i32,
        ep: i32,
        highbd: i32,
    ) -> i64;
    fn shim_calc_proj_params(
        src8: *const u8,
        src16: *const u16,
        width: i32,
        height: i32,
        src_stride: i32,
        dat8: *const u8,
        dat16: *const u16,
        dat_stride: i32,
        flt0: *mut i32,
        flt0_stride: i32,
        flt1: *mut i32,
        flt1_stride: i32,
        h_out: *mut i64,
        c_out: *mut i64,
        ep: i32,
        highbd: i32,
    );
    fn shim_selfguided_restoration(
        dgd8: *const u8,
        dgd16: *const u16,
        width: i32,
        height: i32,
        dgd_stride: i32,
        flt0: *mut i32,
        flt1: *mut i32,
        flt_stride: i32,
        ep: i32,
        bit_depth: i32,
        highbd: i32,
    ) -> i32;
}

/// REAL `av1_compute_stats_c` (lowbd Wiener autocorrelation): `dgd`/`src`
/// are u8 planes; returns `(M, H)` sized `win2` / `win2*win2`.
#[allow(clippy::too_many_arguments)]
pub fn ref_compute_stats(
    wiener_win: usize,
    dgd: &[u8],
    src: &[u8],
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    dgd_stride: i32,
    src_stride: i32,
    use_downsampled_wiener_stats: bool,
) -> (Vec<i64>, Vec<i64>) {
    let win2 = wiener_win * wiener_win;
    let mut m = vec![0i64; win2];
    let mut h = vec![0i64; win2 * win2];
    unsafe {
        shim_compute_stats(
            wiener_win as i32,
            dgd.as_ptr(),
            src.as_ptr(),
            h_start,
            h_end,
            v_start,
            v_end,
            dgd_stride,
            src_stride,
            m.as_mut_ptr(),
            h.as_mut_ptr(),
            use_downsampled_wiener_stats as i32,
        );
    }
    (m, h)
}

/// REAL `av1_compute_stats_highbd_c`: u16 planes, bd 8/10/12.
#[allow(clippy::too_many_arguments)]
pub fn ref_compute_stats_highbd(
    wiener_win: usize,
    dgd: &[u16],
    src: &[u16],
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    dgd_stride: i32,
    src_stride: i32,
    bit_depth: i32,
) -> (Vec<i64>, Vec<i64>) {
    let win2 = wiener_win * wiener_win;
    let mut m = vec![0i64; win2];
    let mut h = vec![0i64; win2 * win2];
    unsafe {
        shim_compute_stats_highbd(
            wiener_win as i32,
            dgd.as_ptr(),
            src.as_ptr(),
            h_start,
            h_end,
            v_start,
            v_end,
            dgd_stride,
            src_stride,
            m.as_mut_ptr(),
            h.as_mut_ptr(),
            bit_depth,
        );
    }
    (m, h)
}

/// REAL `av1_{lowbd,highbd}_pixel_proj_error_c`. Lowbd (`highbd=false`)
/// reads `src8`/`dat8`; highbd reads `src16`/`dat16`. `flt0`/`flt1` are the
/// SGR pass outputs; `xq` the decoded projection weights; `ep` the
/// `av1_sgr_params` index.
#[allow(clippy::too_many_arguments)]
pub fn ref_pixel_proj_error(
    src8: &[u8],
    src16: &[u16],
    width: i32,
    height: i32,
    src_stride: i32,
    dat8: &[u8],
    dat16: &[u16],
    dat_stride: i32,
    flt0: &mut [i32],
    flt0_stride: i32,
    flt1: &mut [i32],
    flt1_stride: i32,
    xq: [i32; 2],
    ep: i32,
    highbd: bool,
) -> i64 {
    let mut xq_c = xq;
    unsafe {
        shim_pixel_proj_error(
            src8.as_ptr(),
            src16.as_ptr(),
            width,
            height,
            src_stride,
            dat8.as_ptr(),
            dat16.as_ptr(),
            dat_stride,
            flt0.as_mut_ptr(),
            flt0_stride,
            flt1.as_mut_ptr(),
            flt1_stride,
            xq_c.as_mut_ptr(),
            ep,
            highbd as i32,
        )
    }
}

/// REAL `av1_calc_proj_params[_high_bd]_c`: the SGR least-squares
/// accumulators. Returns `(H[2][2] row-major, C[2])`.
#[allow(clippy::too_many_arguments)]
pub fn ref_calc_proj_params(
    src8: &[u8],
    src16: &[u16],
    width: i32,
    height: i32,
    src_stride: i32,
    dat8: &[u8],
    dat16: &[u16],
    dat_stride: i32,
    flt0: &mut [i32],
    flt0_stride: i32,
    flt1: &mut [i32],
    flt1_stride: i32,
    ep: i32,
    highbd: bool,
) -> ([i64; 4], [i64; 2]) {
    let mut h_out = [0i64; 4];
    let mut c_out = [0i64; 2];
    unsafe {
        shim_calc_proj_params(
            src8.as_ptr(),
            src16.as_ptr(),
            width,
            height,
            src_stride,
            dat8.as_ptr(),
            dat16.as_ptr(),
            dat_stride,
            flt0.as_mut_ptr(),
            flt0_stride,
            flt1.as_mut_ptr(),
            flt1_stride,
            h_out.as_mut_ptr(),
            c_out.as_mut_ptr(),
            ep,
            highbd as i32,
        );
    }
    (h_out, c_out)
}

/// REAL `av1_selfguided_restoration_c` — the `flt0`/`flt1` producer the SGR
/// search projects against. `dgd8`/`dgd16` are FULL padded buffers with the
/// `width x height` block at element offset `off` (>= 3 rows + 3 cols of
/// margin on every side — the C reads ±3 around the block).
#[allow(clippy::too_many_arguments)]
pub fn ref_selfguided_restoration(
    dgd8: &[u8],
    dgd16: &[u16],
    off: usize,
    width: i32,
    height: i32,
    dgd_stride: i32,
    flt0: &mut [i32],
    flt1: &mut [i32],
    flt_stride: i32,
    ep: i32,
    bit_depth: i32,
    highbd: bool,
) -> i32 {
    unsafe {
        shim_selfguided_restoration(
            if highbd {
                std::ptr::null()
            } else {
                dgd8.as_ptr().add(off)
            },
            if highbd {
                dgd16.as_ptr().add(off)
            } else {
                std::ptr::null()
            },
            width,
            height,
            dgd_stride,
            flt0.as_mut_ptr(),
            flt1.as_mut_ptr(),
            flt_stride,
            ep,
            bit_depth,
            highbd as i32,
        )
    }
}

// ---------------------------------------------------------------------------
// tune=IQ / tune=SSIMULACRA2 knob-explicit encode (C4 stills-quality family)
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn shim_encode_av1_kf_tune(
        y: *const u16,
        u: *const u16,
        v: *const u16,
        w: i32,
        h: i32,
        bd: i32,
        mono: i32,
        ss_x: i32,
        ss_y: i32,
        cq_level: i32,
        cpu_used: i32,
        usage: i32,
        tuning: i32,
        sharpness: i32,
        enable_adaptive_sharpness: i32,
        dist_metric: i32,
        enable_chroma_deltaq: i32,
        deltaq_mode: i32,
        deltaq_strength: i32,
        enable_qm: i32,
        qm_min: i32,
        qm_max: i32,
        enable_cdef: i32,
        out: *mut u8,
        out_cap: usize,
    ) -> i64;
}

/// `AOM_TUNE_IQ` (aom/aomcx.h:1786).
pub const AOM_TUNE_IQ: i32 = 10;
/// `AOM_TUNE_SSIMULACRA2` (aom/aomcx.h:1803).
pub const AOM_TUNE_SSIMULACRA2: i32 = 11;
/// `AOM_DIST_METRIC_PSNR` (aom/aomcx.h:1816).
pub const AOM_DIST_METRIC_PSNR: i32 = 0;
/// `AOM_DIST_METRIC_QM_PSNR` (aom/aomcx.h:1820).
pub const AOM_DIST_METRIC_QM_PSNR: i32 = 1;
/// `CDEF_ADAPTIVE` (av1/encoder/pickcdef.h:28) — the `--enable-cdef=3` arm
/// the tune=IQ/SSIMULACRA2 bundle installs.
pub const CDEF_CONTROL_ADAPTIVE: i32 = 3;

/// The explicit knob set of [`ref_encode_av1_kf_tune`]. Every field follows
/// the shim's `-1 = leave the (tuning-bundle or usage) default` convention;
/// `Default` = all `-1` (a stock encode, byte-identical to
/// `ref_encode_av1_kf` modulo the deltaq-mode default the base shim pins to 0
/// — pass `deltaq_mode: 0` to reproduce the base envelope exactly).
#[derive(Clone, Copy, Debug)]
pub struct RefTuneKnobs {
    /// `AOME_SET_TUNING` (issued FIRST, so it installs the `handle_tuning`
    /// bundle which later explicit knobs override): [`AOM_TUNE_IQ`] /
    /// [`AOM_TUNE_SSIMULACRA2`].
    pub tuning: i32,
    /// `AOME_SET_SHARPNESS` 0..=7.
    pub sharpness: i32,
    /// `AV1E_SET_ENABLE_ADAPTIVE_SHARPNESS` 0/1.
    pub enable_adaptive_sharpness: i32,
    /// `AV1E_SET_DIST_METRIC`: [`AOM_DIST_METRIC_PSNR`] /
    /// [`AOM_DIST_METRIC_QM_PSNR`].
    pub dist_metric: i32,
    /// `AV1E_SET_ENABLE_CHROMA_DELTAQ` 0/1.
    pub enable_chroma_deltaq: i32,
    /// `AV1E_SET_DELTAQ_MODE` (0 = off, 6 = DELTA_Q_VARIANCE_BOOST).
    pub deltaq_mode: i32,
    /// `AV1E_SET_DELTAQ_STRENGTH` (percent, default 100).
    pub deltaq_strength: i32,
    /// `AV1E_SET_ENABLE_QM` 0/1 (when 1, `qm_min`/`qm_max` must be set).
    pub enable_qm: i32,
    /// `AV1E_SET_QM_MIN` (only issued when `enable_qm == 1`).
    pub qm_min: i32,
    /// `AV1E_SET_QM_MAX` (only issued when `enable_qm == 1`).
    pub qm_max: i32,
    /// `AV1E_SET_ENABLE_CDEF` 0/1/2/3 ([`CDEF_CONTROL_ADAPTIVE`]).
    pub enable_cdef: i32,
}

impl Default for RefTuneKnobs {
    fn default() -> Self {
        RefTuneKnobs {
            tuning: -1,
            sharpness: -1,
            enable_adaptive_sharpness: -1,
            dist_metric: -1,
            enable_chroma_deltaq: -1,
            deltaq_mode: -1,
            deltaq_strength: -1,
            enable_qm: -1,
            qm_min: -1,
            qm_max: -1,
            enable_cdef: -1,
        }
    }
}

/// Real single-pass KEY encode with the tune=IQ / tune=SSIMULACRA2 knob set
/// (see `shim_encode_av1_kf_tune`): baseline controls match
/// [`ref_encode_av1_kf`] (`--sb-size=64`, single tile, restoration/aq off, no
/// palette/intrabc, end-usage=q), with `knobs` layered on top — the tuning
/// control first (installs the whole `handle_tuning` bundle), then each
/// explicit knob (>= 0) as an override, exactly the aomenc CLI ordering.
#[allow(clippy::too_many_arguments)]
pub fn ref_encode_av1_kf_tune(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq_level: i32,
    cpu_used: i32,
    usage: u32,
    knobs: &RefTuneKnobs,
) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 8).max(1 << 20)];
    let n = unsafe {
        shim_encode_av1_kf_tune(
            y.as_ptr(),
            u.as_ptr(),
            v.as_ptr(),
            w as i32,
            h as i32,
            bd,
            mono as i32,
            ss_x,
            ss_y,
            cq_level,
            cpu_used,
            usage as i32,
            knobs.tuning,
            knobs.sharpness,
            knobs.enable_adaptive_sharpness,
            knobs.dist_metric,
            knobs.enable_chroma_deltaq,
            knobs.deltaq_mode,
            knobs.deltaq_strength,
            knobs.enable_qm,
            knobs.qm_min,
            knobs.qm_max,
            knobs.enable_cdef,
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert!(n > 0, "shim_encode_av1_kf_tune failed ({n})");
    out.truncate(n as usize);
    out
}

// ---------------------------------------------------------------------------
// deltaq-mode=3 (DELTA_Q_PERCEPTUAL_AI, family C5) reference oracles.
// Append-only; `av1_get_deltaq_offset` is a plain libaom.a export (rd.c:466),
// a table walk over `av1_dc_quant_QTX` (no RTCD dispatch).
// ---------------------------------------------------------------------------
extern "C" {
    fn av1_get_deltaq_offset(bit_depth: i32, qindex: i32, beta: f64) -> i32;
}

/// Reference `av1_get_deltaq_offset` (rd.c:466): the exported libaom fn that
/// maps `(bit_depth, base qindex, beta)` to the qindex offset whose DC quant
/// step is closest to `q(base)/sqrt(beta)`. `bit_depth` is the raw 8/10/12
/// (`aom_bit_depth_t`).
pub fn ref_av1_get_deltaq_offset(bit_depth: u8, qindex: i32, beta: f64) -> i32 {
    unsafe { av1_get_deltaq_offset(i32::from(bit_depth), qindex, beta) }
}

// ---------------------------------------------------------------------------
// deltaq-mode=2 (DELTA_Q_PERCEPTUAL, wavelet AC energy) reference oracle.
// `av1_haar_ac_sad_mxn_uint8_input` (dwt.c:135) is a plain libaom.a export
// whose only kernel, `av1_fdwt8x8_uint8_input`, is a pure-C RTCD entry
// (`#define av1_fdwt8x8_uint8_input av1_fdwt8x8_uint8_input_c`, no SIMD), so
// this links directly without RTCD setup.
// ---------------------------------------------------------------------------
extern "C" {
    fn av1_haar_ac_sad_mxn_uint8_input(
        input: *const u8,
        stride: i32,
        hbd: i32,
        num_8x8_rows: i32,
        num_8x8_cols: i32,
    ) -> i64;
}

/// Reference `av1_haar_ac_sad_mxn_uint8_input` (dwt.c:135): the total AC
/// wavelet energy of a `num_8x8_rows`×`num_8x8_cols` grid of 8x8 blocks over an
/// 8-bit (`hbd = 0`) source with the given `stride`.
pub fn ref_av1_haar_ac_sad_mxn_uint8_input(
    input: &[u8],
    stride: i32,
    num_8x8_rows: i32,
    num_8x8_cols: i32,
) -> i64 {
    unsafe {
        av1_haar_ac_sad_mxn_uint8_input(input.as_ptr(), stride, 0, num_8x8_rows, num_8x8_cols)
    }
}

// ---- av1/common/resize.c: encoder-side source downscale (exported C) ----
extern "C" {
    fn av1_resize_plane(
        input: *const u8,
        height: i32,
        width: i32,
        in_stride: i32,
        output: *mut u8,
        height2: i32,
        width2: i32,
        out_stride: i32,
    ) -> bool;
    fn av1_calculate_scaled_superres_size(width: *mut i32, height: *mut i32, superres_denom: i32);
}

/// Oracle: exported libaom `av1_resize_plane` (non-normative encoder resize).
/// Returns the downscaled plane (`width2 * height2` bytes, tightly packed).
pub fn ref_resize_plane(
    input: &[u8],
    height: i32,
    width: i32,
    in_stride: i32,
    height2: i32,
    width2: i32,
) -> Vec<u8> {
    let mut out = vec![0u8; (width2 * height2) as usize];
    let ok = unsafe {
        av1_resize_plane(
            input.as_ptr(),
            height,
            width,
            in_stride,
            out.as_mut_ptr(),
            height2,
            width2,
            width2,
        )
    };
    assert!(ok, "av1_resize_plane failed");
    out
}

/// Oracle: exported libaom `av1_calculate_scaled_superres_size` — the coded
/// (downscaled) width for a given superres denominator (horizontal only).
pub fn ref_calculate_scaled_superres_width(width: i32, superres_denom: i32) -> i32 {
    let mut w = width;
    let mut h = width; // height is ignored by the C fn (void)
    unsafe { av1_calculate_scaled_superres_size(&mut w, &mut h, superres_denom) };
    w
}

extern "C" {
    // shim/dec_shim.c: drives the file-static highbd_resize_plane via the
    // exported av1_resize_and_extend_frame_nonnormative (highbd YV12, 1 plane).
    fn shim_highbd_resize_plane(
        input: *const u16,
        height: i32,
        width: i32,
        in_stride: i32,
        output: *mut u16,
        height2: i32,
        width2: i32,
        bd: i32,
    ) -> i64;
}

/// Oracle: libaom `highbd_resize_plane` (bd>8 encoder source downscale), reached
/// through the exported `av1_resize_and_extend_frame_nonnormative`. Returns the
/// downscaled plane (`width2 * height2` u16, tightly packed). `ref_init` first —
/// the wrapper's border extend is RTCD-dispatched.
pub fn ref_highbd_resize_plane(
    input: &[u16],
    height: i32,
    width: i32,
    in_stride: i32,
    height2: i32,
    width2: i32,
    bd: i32,
) -> Vec<u16> {
    ref_init();
    let mut out = vec![0u16; (width2 * height2) as usize];
    let rc = unsafe {
        shim_highbd_resize_plane(
            input.as_ptr(),
            height,
            width,
            in_stride,
            out.as_mut_ptr(),
            height2,
            width2,
            bd,
        )
    };
    assert_eq!(rc, 0, "shim_highbd_resize_plane failed ({rc})");
    out
}

extern "C" {
    // shim/dec_shim.c: exported optimized 8-bit source scaler
    // av1_resize_and_extend_frame_c (EIGHTTAP_SMOOTH, phase 8) over an
    // aom_extend_frame_borders_c-extended YV12.
    fn shim_resize_and_extend_frame_8bit(
        input: *const u16,
        width: i32,
        height: i32,
        in_stride: i32,
        output: *mut u16,
        width2: i32,
        height2: i32,
    ) -> i64;
}

/// Oracle: libaom `av1_resize_and_extend_frame_c` (the optimized 8-bit source
/// downscale, `EIGHTTAP_SMOOTH` / phase 8) — the superres denom-16 corner.
/// `input` is `width * height` tight u8-valued u16; returns the `width2 *
/// height2` downscaled plane. `ref_init` first (RTCD-dispatched internals).
pub fn ref_resize_and_extend_frame_8bit(
    input: &[u16],
    width: i32,
    height: i32,
    in_stride: i32,
    width2: i32,
    height2: i32,
) -> Vec<u16> {
    ref_init();
    let mut out = vec![0u16; (width2 * height2) as usize];
    let rc = unsafe {
        shim_resize_and_extend_frame_8bit(
            input.as_ptr(),
            width,
            height,
            in_stride,
            out.as_mut_ptr(),
            width2,
            height2,
        )
    };
    assert_eq!(rc, 0, "shim_resize_and_extend_frame_8bit failed ({rc})");
    out
}

// inter_shim.c — decoder single-ref translational inter predictor (crate
// aom-inter, chunk 1d): the real `inter_predictor` facade + the verbatim
// `build_mc_border` border oracle.
extern "C" {
    fn shim_inter_predictor(
        src: *const u8,
        src_stride: i32,
        dst: *mut u8,
        dst_stride: i32,
        w: i32,
        h: i32,
        subpel_x: i32,
        subpel_y: i32,
        filter_x: i32,
        filter_y: i32,
    );
    fn shim_build_mc_border(
        plane: *const u8,
        src_stride: i32,
        w: i32,
        h: i32,
        x: i32,
        y: i32,
        b_w: i32,
        b_h: i32,
        dst: *mut u8,
    );
}

/// Reference libaom `inter_predictor` (reconinter.h:255) — the unscaled lowbd
/// single-ref SR facade over the real `av1_convolve_2d_facade`. `src`/`src_off`
/// point at the block top-left interior of a bordered region; `subpel_x`/`subpel_y`
/// are in `0..=15`; `filter_x`/`filter_y` select the 8-tap family (0/1/2). Returns
/// the `w`×`h` predictor (dst stride `w`).
#[allow(clippy::too_many_arguments)]
pub fn ref_inter_predictor(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    subpel_x: usize,
    subpel_y: usize,
    filter_x: usize,
    filter_y: usize,
) -> Vec<u8> {
    ref_init();
    let mut dst = vec![0u8; w * h];
    unsafe {
        shim_inter_predictor(
            src.as_ptr().add(src_off),
            src_stride as i32,
            dst.as_mut_ptr(),
            w as i32,
            w as i32,
            h as i32,
            subpel_x as i32,
            subpel_y as i32,
            filter_x as i32,
            filter_y as i32,
        )
    }
    dst
}

/// Reference libaom `build_mc_border` (decodeframe.c:455): gather a `b_w`×`b_h`
/// block from plane `plane` (`ref_w`×`ref_h`, stride `ref_stride`) starting at
/// `(gx, gy)` (may be negative), edge-replicating any out-of-plane region. Returns
/// the tightly-packed `b_w`×`b_h` scratch.
#[allow(clippy::too_many_arguments)]
pub fn ref_build_mc_border(
    plane: &[u8],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    gx: i32,
    gy: i32,
    b_w: usize,
    b_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; b_w * b_h];
    unsafe {
        shim_build_mc_border(
            plane.as_ptr(),
            ref_stride as i32,
            ref_w as i32,
            ref_h as i32,
            gx,
            gy,
            b_w as i32,
            b_h as i32,
            dst.as_mut_ptr(),
        )
    }
    dst
}
