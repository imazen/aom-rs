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

// intra_shim.c — dispatch to aom_<mode>_predictor_<W>x<H>_c.
extern "C" {
    fn shim_intra_pred(
        mode: i32, size_idx: i32, dst: *mut u8, stride: isize, above: *const u8, left: *const u8,
    );
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
