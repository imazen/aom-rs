//! Inverse 2-D transform + reconstruction, bit-exact port of libaom v3.14.1
//! `av1/common/av1_inv_txfm2d.c` (`inv_txfm2d_add_c` + facade + entry points).
//!
//! Row-first then column, with `NewInvSqrt2` rectangular scaling, per-stage
//! `clamp_value` (stage_range = a bd-dependent constant — both branches of
//! `av1_gen_inv_stage_range` assign the same `opt_range`), input `clamp_buf`
//! stages, `av1_round_shift_array` shifts, and `highbd_clip_pixel_add`
//! reconstruction onto the destination. Depends on `bd` (8/10/12).

use crate::transform::cospi::{NEW_SQRT2_BITS, NEW_INV_SQRT2};
use crate::transform::fdct::{clamp_value, round_shift};
use crate::transform::txfm2d::{
    get_rect_tx_log_ratio, log2_idx, FLIP_CFG, HTX_TAB, TXFM_TYPE_LS, TX_SIZE_HIGH, TX_SIZE_WIDE,
    VTX_TAB,
};
use crate::transform::{
    av1_iadst16, av1_iadst4, av1_iadst8, av1_idct16, av1_idct32, av1_idct4, av1_idct64, av1_idct8,
    av1_iidentity16, av1_iidentity32, av1_iidentity4, av1_iidentity8,
};

type Txfm1d = fn(&[i32], &mut [i32], i32, &[i8]);

pub(crate) const INV_COS_BIT: i32 = 12;

// av1_inv_txfm_shift_ls[tx_size][0..2]
#[rustfmt::skip]
static INV_SHIFT: [[i8; 2]; 19] = [
    [0, -4], [-1, -4], [-2, -4], [-2, -4], [-2, -4],
    [0, -4], [0, -4], [-1, -4], [-1, -4], [-1, -4],
    [-1, -4], [-1, -4], [-1, -4], [-1, -4], [-1, -4],
    [-2, -4], [-2, -4], [-2, -4], [-2, -4],
];

fn inv_txfm_func(txfm_type: i32) -> Txfm1d {
    match txfm_type {
        0 => av1_idct4,
        1 => av1_idct8,
        2 => av1_idct16,
        3 => av1_idct32,
        4 => av1_idct64,
        5 => av1_iadst4,
        6 => av1_iadst8,
        7 => av1_iadst16,
        8 => av1_iidentity4,
        9 => av1_iidentity8,
        10 => av1_iidentity16,
        11 => av1_iidentity32,
        _ => panic!("invalid inv txfm_type {txfm_type}"),
    }
}

/// (opt_range_col, opt_range_row) from `av1_gen_inv_stage_range` — the only
/// output-affecting product of that function (both if-branches assign the same
/// value; the assert-only `real_range` path is disabled).
fn opt_range(bd: i32) -> (i8, i8) {
    match bd {
        8 => (16, 16),
        10 => (16, 18),
        12 => (18, 20),
        _ => panic!("bd must be 8/10/12"),
    }
}

struct Cfg {
    shift: [i8; 2],
    func_col: Txfm1d,
    func_row: Txfm1d,
    /// Raw TXFM_TYPE ids (0..=11) — the SIMD per-kernel dispatch keys.
    txfm_type_col: i32,
    txfm_type_row: i32,
    ud_flip: bool,
    lr_flip: bool,
    valid: bool,
}

fn get_inv_txfm_cfg(tx_type: usize, tx_size: usize) -> Cfg {
    let (ud_flip, lr_flip) = FLIP_CFG[tx_type];
    let txw_idx = log2_idx(TX_SIZE_WIDE[tx_size]);
    let txh_idx = log2_idx(TX_SIZE_HIGH[tx_size]);
    let txfm_type_col = TXFM_TYPE_LS[txh_idx][VTX_TAB[tx_type]];
    let txfm_type_row = TXFM_TYPE_LS[txw_idx][HTX_TAB[tx_type]];
    let valid = txfm_type_col != -1 && txfm_type_row != -1;
    Cfg {
        shift: INV_SHIFT[tx_size],
        func_col: if valid { inv_txfm_func(txfm_type_col) } else { av1_idct4 },
        func_row: if valid { inv_txfm_func(txfm_type_row) } else { av1_idct4 },
        txfm_type_col,
        txfm_type_row,
        ud_flip,
        lr_flip,
        valid,
    }
}

/// Is `(tx_type, tx_size)` a supported inverse-transform combination?
pub fn inv_txfm_valid(tx_type: usize, tx_size: usize) -> bool {
    get_inv_txfm_cfg(tx_type, tx_size).valid
}

fn round_shift_array(arr: &mut [i32], bit: i32) {
    if bit == 0 {
        return;
    }
    if bit > 0 {
        for v in arr.iter_mut() {
            *v = round_shift(*v as i64, bit);
        }
    } else {
        for v in arr.iter_mut() {
            let widened = (1i64 << (-bit)) * (*v as i64);
            *v = widened.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
}

#[inline]
fn clamp_buf(buf: &mut [i32], bit: i8) {
    for v in buf.iter_mut() {
        *v = clamp_value(*v, bit);
    }
}

#[inline]
fn highbd_clip_pixel_add(dest: u16, trans: i32, bd: i32) -> u16 {
    let hi = (1i32 << bd) - 1;
    ((dest as i32).wrapping_add(trans)).clamp(0, hi) as u16
}

/// Remap the (possibly 32-capped) coefficient `input` into the full
/// `col_n*row_n` buffer with zeros, matching the C entry points for the 5
/// large sizes. Returns the full modified input buffer.
fn remap_input(input: &[i32], tx_size: usize, col_n: usize, row_n: usize) -> Vec<i32> {
    let mut mod_input = vec![0i32; col_n * row_n];
    match tx_size {
        4 => {
            // 64x64: 32x32 -> 64x64
            for col in 0..32 {
                mod_input[col * 64..col * 64 + 32].copy_from_slice(&input[col * 32..col * 32 + 32]);
            }
        }
        11 => {
            // 32x64: 32x32 -> mod[col*64..+32]
            for col in 0..32 {
                mod_input[col * 64..col * 64 + 32].copy_from_slice(&input[col * 32..col * 32 + 32]);
            }
        }
        12 => {
            // 64x32: 32x32 contiguous -> first half
            mod_input[..32 * 32].copy_from_slice(&input[..32 * 32]);
        }
        17 => {
            // 16x64: 16x32 -> mod[col*64..+32]
            for col in 0..16 {
                mod_input[col * 64..col * 64 + 32].copy_from_slice(&input[col * 32..col * 32 + 32]);
            }
        }
        18 => {
            // 64x16: 32x16 contiguous -> first half
            mod_input[..16 * 32].copy_from_slice(&input[..16 * 32]);
        }
        _ => {
            mod_input.copy_from_slice(input);
        }
    }
    mod_input
}

/// Expected packed coefficient input length for a given tx_size (what the C
/// `av1_inv_txfm2d_add_*_c` entry point consumes).
pub fn inv_input_len(tx_size: usize) -> usize {
    match tx_size {
        4 | 11 | 12 => 32 * 32,
        17 | 18 => 16 * 32,
        _ => TX_SIZE_WIDE[tx_size] * TX_SIZE_HIGH[tx_size],
    }
}

/// Public inverse 2-D transform + add. `output` is a `bd`-bit pixel buffer of
/// at least `row_n*stride`; residuals are reconstructed onto it in place.
pub fn av1_inv_txfm2d_add(
    input: &[i32],
    output: &mut [u16],
    stride: usize,
    tx_type: usize,
    tx_size: usize,
    bd: i32,
) {
    let cfg = get_inv_txfm_cfg(tx_type, tx_size);
    assert!(cfg.valid, "unsupported inverse (tx_type={tx_type}, tx_size={tx_size})");
    let col_n = TX_SIZE_WIDE[tx_size];
    let row_n = TX_SIZE_HIGH[tx_size];
    let shift = cfg.shift;
    let rect_type = get_rect_tx_log_ratio(col_n as i64, row_n as i64);
    let (opt_range_col, opt_range_row) = opt_range(bd);
    let stage_range_row = [opt_range_row; 12];
    let stage_range_col = [opt_range_col; 12];

    let mod_input = remap_input(input, tx_size, col_n, row_n);

    let mut buf = vec![0i32; col_n * row_n];
    let mut temp_in = [0i32; 64];
    let mut temp_out = [0i32; 64];

    // Rows — the SIMD row pass (8-row lane batches) is bit-identical to this
    // scalar loop (crate::transform::simd docs + differentials); it declines (false) when
    // the row kernel isn't ported / row_n < 8 / SIMD unavailable or pinned off.
    #[cfg(target_arch = "x86_64")]
    let rows_done = crate::transform::simd::try_inv_row_pass(
        cfg.txfm_type_row,
        &mod_input,
        &mut buf,
        col_n,
        row_n,
        rect_type.abs() == 1,
        -(shift[0] as i32),
        (bd + 8) as i8,
        &stage_range_row,
    );
    #[cfg(not(target_arch = "x86_64"))]
    let rows_done = false;
    if !rows_done {
        for r in 0..row_n {
            let ti = &mut temp_in[0..col_n];
            if rect_type.abs() == 1 {
                for c in 0..col_n {
                    ti[c] = round_shift(
                        mod_input[c * row_n + r] as i64 * NEW_INV_SQRT2 as i64,
                        NEW_SQRT2_BITS,
                    );
                }
            } else {
                for c in 0..col_n {
                    ti[c] = mod_input[c * row_n + r];
                }
            }
            clamp_buf(ti, (bd + 8) as i8);
            (cfg.func_row)(ti, &mut buf[r * col_n..r * col_n + col_n], INV_COS_BIT, &stage_range_row);
            round_shift_array(&mut buf[r * col_n..r * col_n + col_n], -(shift[0] as i32));
        }
    }

    // Columns — same contract: the SIMD column pass (8-column lane batches)
    // is bit-identical to the scalar loop and declines when not applicable.
    let col_clamp = (bd + 6).max(16) as i8;
    #[cfg(target_arch = "x86_64")]
    let cols_done = crate::transform::simd::try_inv_col_pass(
        cfg.txfm_type_col,
        &buf,
        output,
        stride,
        col_n,
        row_n,
        -(shift[1] as i32),
        col_clamp,
        &stage_range_col,
        cfg.ud_flip,
        cfg.lr_flip,
        bd,
    );
    #[cfg(not(target_arch = "x86_64"))]
    let cols_done = false;
    if !cols_done {
        for c in 0..col_n {
            let ti = &mut temp_in[0..row_n];
            for r in 0..row_n {
                let cc = if cfg.lr_flip { col_n - c - 1 } else { c };
                ti[r] = buf[r * col_n + cc];
            }
            clamp_buf(ti, col_clamp);
            let to = &mut temp_out[0..row_n];
            (cfg.func_col)(ti, to, INV_COS_BIT, &stage_range_col);
            round_shift_array(to, -(shift[1] as i32));
            for r in 0..row_n {
                let src = if cfg.ud_flip { to[row_n - r - 1] } else { to[r] };
                let idx = r * stride + c;
                output[idx] = highbd_clip_pixel_add(output[idx], src, bd);
            }
        }
    }
}

/// `UNIT_QUANT_SHIFT` (`aom_dsp/txfm_common.h`): the extra shift the reversible
/// 4x4 Walsh–Hadamard folds into its input, cancelling the lossless dequant's
/// unit quantizer (dc/ac step 4 at qindex 0) so `level * 4 >> 2 == level`.
const UNIT_QUANT_SHIFT: i32 = 2;

/// `av1_highbd_iwht4x4_add` (`av1/common/av1_inv_txfm2d.c`): the 4x4 reversible
/// Walsh–Hadamard inverse transform used for `xd->lossless` blocks (forced
/// `TX_4X4`, tx_type always `DCT_DCT`). `input` is the DEQUANTIZED 4x4
/// coefficient block in raster order (as fed to [`av1_inv_txfm2d_add`] on the
/// non-lossless path); `output` is a `bd`-bit pixel buffer of at least
/// `4*stride`, holding the prediction on entry and the reconstruction on return.
///
/// Dispatches on `eob` exactly like the C wrapper: `eob > 1` runs the full
/// 16-point transform, otherwise the DC-only special case. Per idct.c the
/// `eob <= 1` branch is significant for lossless (it produces a different,
/// correct result than the full transform on a DC-only block — not merely an
/// optimization), so the two kernels are NOT interchangeable.
pub fn av1_highbd_iwht4x4_add(input: &[i32], output: &mut [u16], stride: usize, eob: usize, bd: i32) {
    if eob > 1 {
        av1_highbd_iwht4x4_16_add(input, output, stride, bd);
    } else {
        av1_highbd_iwht4x4_1_add(input, output, stride, bd);
    }
}

/// `av1_inverse_transform_block` (`av1/common/idct.c`): the recon inverse-transform
/// dispatch used on BOTH the encode and decode reconstruction paths. For a
/// coded-lossless block (`xd->lossless[segment_id]`, which forces `TX_4X4` +
/// `DCT_DCT`) it applies the reversible 4x4 Walsh–Hadamard inverse
/// [`av1_highbd_iwht4x4_add`] (eob-dispatched); otherwise the regular
/// [`av1_inv_txfm2d_add`]. `input` is the dequantized coefficient block; `dst`
/// holds the prediction on entry and the reconstruction on return. Mirrors the
/// forward split in `xform_quant` (DCT vs [`av1_fwht4x4`]) so the encoder's
/// reconstruction (and any pixel-domain distortion) matches the decoder bit for
/// bit at qindex 0.
#[allow(clippy::too_many_arguments)]
pub fn av1_inverse_transform_add(
    input: &[i32],
    dst: &mut [u16],
    stride: usize,
    tx_type: usize,
    tx_size: usize,
    bd: i32,
    eob: usize,
    lossless: bool,
) {
    if lossless {
        av1_highbd_iwht4x4_add(input, dst, stride, eob, bd);
    } else {
        av1_inv_txfm2d_add(input, dst, stride, tx_type, tx_size, bd);
    }
}

/// `av1_highbd_iwht4x4_16_add_c` — the full 4-point reversible Walsh–Hadamard
/// applied column-then-row. Bit-exact port; `range_check_value(_, bd+1)` is a
/// no-op in the production config (coefficient range checking off).
fn av1_highbd_iwht4x4_16_add(input: &[i32], dest: &mut [u16], stride: usize, bd: i32) {
    let mut out = [0i32; 16];
    // Column pass: iteration i reads input[i], input[i+4], input[i+8],
    // input[i+12] and writes out[i], out[i+4], out[i+8], out[i+12].
    for i in 0..4 {
        let mut a1 = input[i] >> UNIT_QUANT_SHIFT;
        let mut c1 = input[i + 4] >> UNIT_QUANT_SHIFT;
        let mut d1 = input[i + 8] >> UNIT_QUANT_SHIFT;
        let mut b1 = input[i + 12] >> UNIT_QUANT_SHIFT;
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        out[i] = a1;
        out[i + 4] = b1;
        out[i + 8] = c1;
        out[i + 12] = d1;
    }
    // Row pass: iteration i reads out[4i..4i+4] and writes dest column i,
    // rows 0..3 (dest[i], dest[i+stride], dest[i+2*stride], dest[i+3*stride]).
    for i in 0..4 {
        let mut a1 = out[4 * i];
        let mut c1 = out[4 * i + 1];
        let mut d1 = out[4 * i + 2];
        let mut b1 = out[4 * i + 3];
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        dest[i] = highbd_clip_pixel_add(dest[i], a1, bd);
        dest[i + stride] = highbd_clip_pixel_add(dest[i + stride], b1, bd);
        dest[i + 2 * stride] = highbd_clip_pixel_add(dest[i + 2 * stride], c1, bd);
        dest[i + 3 * stride] = highbd_clip_pixel_add(dest[i + 3 * stride], d1, bd);
    }
}

/// `av1_highbd_iwht4x4_1_add_c` — the DC-only special case (`eob <= 1`). Uses
/// only the DC coefficient; bit-exact port.
fn av1_highbd_iwht4x4_1_add(input: &[i32], dest: &mut [u16], stride: usize, bd: i32) {
    let a1 = input[0] >> UNIT_QUANT_SHIFT;
    let e1 = a1 >> 1;
    let tmp = [a1 - e1, e1, e1, e1];
    // Row pass: iteration i reads tmp[i] and writes dest column i, rows 0..3.
    for i in 0..4 {
        let e = tmp[i] >> 1;
        let a = tmp[i] - e;
        dest[i] = highbd_clip_pixel_add(dest[i], a, bd);
        dest[i + stride] = highbd_clip_pixel_add(dest[i + stride], e, bd);
        dest[i + 2 * stride] = highbd_clip_pixel_add(dest[i + 2 * stride], e, bd);
        dest[i + 3 * stride] = highbd_clip_pixel_add(dest[i + 3 * stride], e, bd);
    }
}

/// `av1_fwht4x4_c` (`av1/encoder/hybrid_fwd_txfm.c:24`): the 4x4 reversible,
/// orthonormal Walsh–Hadamard FORWARD transform used for `xd->lossless` blocks
/// (forced `TX_4X4`, `tx_type` always `DCT_DCT`). Shared for high and low bit
/// depth — there is no separate `av1_highbd_fwht4x4` (the highbd dispatch reaches
/// this same function).
///
/// `input` is the 4x4 residual (row-major, row stride `stride`); `output` is the
/// 16-entry raster coefficient block. Intermediates are `tran_high_t` (i64);
/// output is `tran_low_t` (i32).
///
/// This is the transpose-dual of the inverse [`av1_highbd_iwht4x4_add`]: the
/// inverse shifts its INPUT down by `UNIT_QUANT_SHIFT`, the forward scales its
/// pass-1 OUTPUT up by `UNIT_QUANT_FACTOR == 1 << UNIT_QUANT_SHIFT`, so the pair
/// is identity at the lossless unit quantizer (`level * 4 >> 2 == level`). Two
/// details are load-bearing and preserved verbatim: (a) each pass writes its four
/// results in the `(a1, c1, d1, b1)` permutation, NOT identity; (b) the
/// `* UNIT_QUANT_FACTOR` lands only on pass 1.
pub fn av1_fwht4x4(input: &[i16], output: &mut [i32], stride: usize) {
    const UNIT_QUANT_FACTOR: i64 = 1 << UNIT_QUANT_SHIFT;
    // Pass 0 (input columns -> output rows), NO shift, NO factor. Iteration i
    // reads input column i (input[i], input[i+stride], input[i+2*stride],
    // input[i+3*stride]) and writes output[4*i .. 4*i+4].
    for i in 0..4 {
        let mut a1 = i64::from(input[i]);
        let mut b1 = i64::from(input[i + stride]);
        let mut c1 = i64::from(input[i + 2 * stride]);
        let mut d1 = i64::from(input[i + 3 * stride]);

        a1 += b1;
        d1 -= c1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= c1;
        d1 += b1;
        output[4 * i] = a1 as i32;
        output[4 * i + 1] = c1 as i32;
        output[4 * i + 2] = d1 as i32;
        output[4 * i + 3] = b1 as i32;
    }
    // Pass 1 (output raster-columns -> output columns), APPLY *UNIT_QUANT_FACTOR.
    // Iteration i reads output column i (output[i], output[i+4], output[i+8],
    // output[i+12]) and writes the same column, scaled.
    for i in 0..4 {
        let mut a1 = i64::from(output[i]);
        let mut b1 = i64::from(output[i + 4]);
        let mut c1 = i64::from(output[i + 8]);
        let mut d1 = i64::from(output[i + 12]);

        a1 += b1;
        d1 -= c1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= c1;
        d1 += b1;
        output[i] = (a1 * UNIT_QUANT_FACTOR) as i32;
        output[i + 4] = (c1 * UNIT_QUANT_FACTOR) as i32;
        output[i + 8] = (d1 * UNIT_QUANT_FACTOR) as i32;
        output[i + 12] = (b1 * UNIT_QUANT_FACTOR) as i32;
    }
}
