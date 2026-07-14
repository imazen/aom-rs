//! Chroma-from-luma (CfL) prediction pipeline — port of libaom v3.14.1
//! `av1/common/cfl.{h,c}` (the `_c` reference kernels), decoder flow.
//!
//! The pipeline has three stages, all operating on the `CFL_BUF_LINE`(=32)-strided
//! Q3 buffers of [`CflCtx`]:
//!
//! 1. **Store** ([`cfl_store_tx`]): after each luma transform block of a
//!    store-required block is reconstructed (`predict_and_reconstruct_intra_block`,
//!    decodeframe.c), its pixels are subsampled into `recon_buf_q3` at the
//!    co-located chroma position — `cfl_luma_subsampling_{420,422,444}_hbd_c`
//!    (values are left-shifted so every chroma sample is luma·8, "Q3"). Sub-8x8
//!    blocks sharing one chroma block store into the same buffer at parity-adjusted
//!    offsets (`sub8x8_adjust_offset`), and the written surface is tracked in
//!    `buf_width`/`buf_height` for frame-boundary padding.
//! 2. **Average** ([`CflCtx::compute_parameters`], called lazily by
//!    [`cfl_predict_block`]): pad the stored surface out to the chroma tx block
//!    (`cfl_pad` replicates the last column/row when chroma coverage exceeds the
//!    stored luma, e.g. at frame edges), then `cfl_subtract_average` — the mean
//!    over the tx block is subtracted, leaving the zero-mean "AC" contribution
//!    in `ac_buf_q3`.
//! 3. **Predict** ([`cfl_predict_block`]): with the DC prediction already in
//!    `dst` (the caller runs the ordinary DC intra predictor first, as
//!    `av1_predict_intra_block_facade` does), each sample becomes
//!    `clip_pixel_highbd(dst + ROUND_POWER_OF_TWO_SIGNED(alpha_q3 * ac_q3, 6))` —
//!    `cfl_predict_hbd_c`, with `alpha_q3` from the coded joint sign + alpha
//!    index via [`cfl_idx_to_alpha`].
//!
//! # Bit-depth note
//!
//! Only the **hbd** (`uint16_t` pixel) kernels are ported: the decode driver keeps
//! every plane as `u16` regardless of bit depth. At `bd == 8` the C lbd kernels
//! (`cfl_luma_subsampling_*_lbd_c` / `cfl_predict_lbd_c`) compute the identical
//! arithmetic on the same values — the subsampling sums and `<<` shifts do not
//! depend on the container type at 8-bit ranges, and `clip_pixel` ==
//! `clip_pixel_highbd(·, 8)` — so the hbd port is the complete behaviour.
//! (The encoder-only DC-prediction cache — `cfl_store_dc_pred` /
//! `cfl_load_dc_pred`, used by `cfl_rd_pick_alpha` — and the inter-path
//! `cfl_store_block` are not ported.)
//!
//! # Validation
//!
//! Hand-traced vectors (`tests/cfl_vectors.rs`) pin each stage's arithmetic to
//! values computed by hand from the C source, and the full-tile encode↔decode
//! roundtrip in aom-decode exercises the pipeline end to end. A **direct
//! differential test against the C oracle is deferred** (the `cfl_subsample_*_c`
//! / `cfl_subtract_average_*_c` / `cfl_predict_hbd_*_c` symbols are exported
//! from libaom.a, but `aom-sys-ref` carries live encoder-track WIP and cannot
//! take new externs this chunk); until that shim lands, a *shared* misreading
//! of the C here would not be caught by the roundtrip (both sides use this
//! port) — only by the hand-traced vectors.

/// `CFL_BUF_LINE` (blockd.h): the Q3 buffers' row stride — the widest / tallest
/// chroma block CfL can predict (`CFL_MAX_BLOCK_SIZE == BLOCK_32X32`).
pub const CFL_BUF_LINE: usize = 32;
/// `CFL_BUF_SQUARE`: total Q3 buffer area.
pub const CFL_BUF_SQUARE: usize = CFL_BUF_LINE * CFL_BUF_LINE;

/// `CFL_CTX` (blockd.h), decoder slice: the reconstructed-luma Q3 buffer, the
/// zero-mean AC buffer, the written-surface extent, and the chroma subsampling.
#[derive(Clone)]
pub struct CflCtx {
    /// `recon_buf_q3`: subsampled reconstructed luma, Q3, `CFL_BUF_LINE`-strided.
    pub recon_buf_q3: [u16; CFL_BUF_SQUARE],
    /// `ac_buf_q3`: the zero-mean AC contribution after `subtract_average`.
    pub ac_buf_q3: [i16; CFL_BUF_SQUARE],
    /// `buf_width` / `buf_height`: the written surface, in chroma pixels.
    pub buf_width: i32,
    pub buf_height: i32,
    /// `are_parameters_computed`: `ac_buf_q3` is valid for the current store.
    pub are_parameters_computed: bool,
    /// `subsampling_x` / `subsampling_y` from the sequence header.
    pub subsampling_x: i32,
    pub subsampling_y: i32,
}

impl CflCtx {
    /// `cfl_init` (cfl.c): zeroed buffers, parameters invalid.
    pub fn new(subsampling_x: i32, subsampling_y: i32) -> Self {
        CflCtx {
            recon_buf_q3: [0; CFL_BUF_SQUARE],
            ac_buf_q3: [0; CFL_BUF_SQUARE],
            buf_width: 0,
            buf_height: 0,
            are_parameters_computed: false,
            subsampling_x,
            subsampling_y,
        }
    }

    /// `cfl_compute_parameters` (cfl.c): pad the stored surface to the chroma tx
    /// block, then subtract the block average into `ac_buf_q3`. Asserts the
    /// parameters are not already computed (C behaviour).
    fn compute_parameters(&mut self, tx_size: usize) {
        debug_assert!(!self.are_parameters_computed);
        let width = TX_W[tx_size];
        let height = TX_H[tx_size];
        cfl_pad(self, width as i32, height as i32);
        subtract_average(&self.recon_buf_q3, &mut self.ac_buf_q3, width, height);
        self.are_parameters_computed = true;
    }
}

// tx_size_wide / tx_size_high (common_data.h), TX_SIZES_ALL.
const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
// block_size_wide / block_size_high (common_data.h), BLOCK_SIZES_ALL.
const BLK_W: [i32; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H: [i32; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

/// `cfl_luma_subsampling_420_hbd_c`: each output Q3 sample is the sum of a 2x2
/// luma quad, `<< 1` (·8 total). Output rows advance one `CFL_BUF_LINE` per
/// *pair* of luma rows.
pub fn subsample_420_hbd(
    input: &[u16],
    mut in_off: usize,
    in_stride: usize,
    out: &mut [u16; CFL_BUF_SQUARE],
    mut out_off: usize,
    width: usize,
    height: usize,
) {
    let mut j = 0;
    while j < height {
        let mut i = 0;
        while i < width {
            let top = in_off + i;
            let bot = top + in_stride;
            let quad = u32::from(input[top])
                + u32::from(input[top + 1])
                + u32::from(input[bot])
                + u32::from(input[bot + 1]);
            out[out_off + (i >> 1)] = (quad << 1) as u16;
            i += 2;
        }
        in_off += in_stride << 1;
        out_off += CFL_BUF_LINE;
        j += 2;
    }
}

/// `cfl_luma_subsampling_422_hbd_c`: horizontal pair sum `<< 2`.
pub fn subsample_422_hbd(
    input: &[u16],
    mut in_off: usize,
    in_stride: usize,
    out: &mut [u16; CFL_BUF_SQUARE],
    mut out_off: usize,
    width: usize,
    height: usize,
) {
    for _ in 0..height {
        let mut i = 0;
        while i < width {
            let pair = u32::from(input[in_off + i]) + u32::from(input[in_off + i + 1]);
            out[out_off + (i >> 1)] = (pair << 2) as u16;
            i += 2;
        }
        in_off += in_stride;
        out_off += CFL_BUF_LINE;
    }
}

/// `cfl_luma_subsampling_444_hbd_c`: straight copy `<< 3`.
pub fn subsample_444_hbd(
    input: &[u16],
    mut in_off: usize,
    in_stride: usize,
    out: &mut [u16; CFL_BUF_SQUARE],
    mut out_off: usize,
    width: usize,
    height: usize,
) {
    for _ in 0..height {
        for i in 0..width {
            out[out_off + i] = input[in_off + i] << 3;
        }
        in_off += in_stride;
        out_off += CFL_BUF_LINE;
    }
}

/// `cfl_pad` (cfl.c): when the chroma tx block exceeds the stored luma surface
/// (frame-boundary overrun), replicate the last stored column rightward, then
/// the last stored row downward. Extends `buf_width`/`buf_height` to the padded
/// size.
fn cfl_pad(cfl: &mut CflCtx, width: i32, height: i32) {
    let diff_width = width - cfl.buf_width;
    let diff_height = height - cfl.buf_height;

    if diff_width > 0 {
        let min_height = height - diff_height;
        let mut off = (width - diff_width) as usize;
        for _ in 0..min_height {
            let last_pixel = cfl.recon_buf_q3[off - 1];
            for i in 0..diff_width as usize {
                cfl.recon_buf_q3[off + i] = last_pixel;
            }
            off += CFL_BUF_LINE;
        }
        cfl.buf_width = width;
    }
    if diff_height > 0 {
        let mut off = ((height - diff_height) * CFL_BUF_LINE as i32) as usize;
        for _ in 0..diff_height {
            for i in 0..width as usize {
                cfl.recon_buf_q3[off + i] = cfl.recon_buf_q3[off - CFL_BUF_LINE + i];
            }
            off += CFL_BUF_LINE;
        }
        cfl.buf_height = height;
    }
}

/// `subtract_average_c` (cfl.c): `avg = (sum + round_offset) >> num_pel_log2`
/// over the `width`x`height` Q3 region, then `dst = src - avg` per sample. The
/// size-specific C wrappers pass `round_offset = (width*height)/2` and
/// `num_pel_log2 = log2(width*height)` (`CFL_SUB_AVG_FN` table, verified for
/// all 14 CfL sizes).
pub fn subtract_average(
    src: &[u16; CFL_BUF_SQUARE],
    dst: &mut [i16; CFL_BUF_SQUARE],
    width: usize,
    height: usize,
) {
    let num_pel_log2 = (width * height).trailing_zeros();
    let round_offset = ((width * height) >> 1) as i32;
    let mut sum = round_offset;
    let mut off = 0usize;
    for _ in 0..height {
        for i in 0..width {
            sum += i32::from(src[off + i]);
        }
        off += CFL_BUF_LINE;
    }
    let avg = sum >> num_pel_log2;
    off = 0;
    for _ in 0..height {
        for i in 0..width {
            dst[off + i] = (i32::from(src[off + i]) - avg) as i16;
        }
        off += CFL_BUF_LINE;
    }
}

/// `cfl_idx_to_alpha` (cfl.c) with the `CFL_SIGN_U/V` + `CFL_IDX_U/V` macros
/// (enums.h): decode the per-plane signed `alpha_q3` from the coded joint sign
/// and 8-bit alpha index. `plane` is 1 (U) or 2 (V) — C's `CFL_PRED_TYPE` is
/// `plane - 1`. Sign 0 → alpha 0 (and the corresponding index nibble is not
/// coded); otherwise `±(abs + 1)` for abs in 0..16.
pub fn cfl_idx_to_alpha(alpha_idx: i32, joint_sign: i32, plane: usize) -> i32 {
    const CFL_SIGN_ZERO: i32 = 0;
    const CFL_SIGN_POS: i32 = 2;
    const CFL_SIGNS: i32 = 3;
    let sign_u = ((joint_sign + 1) * 11) >> 5;
    let alpha_sign = if plane == 1 {
        sign_u
    } else {
        (joint_sign + 1) - CFL_SIGNS * sign_u
    };
    if alpha_sign == CFL_SIGN_ZERO {
        return 0;
    }
    let abs_alpha_q3 = if plane == 1 {
        alpha_idx >> 4
    } else {
        alpha_idx & 15
    };
    if alpha_sign == CFL_SIGN_POS {
        abs_alpha_q3 + 1
    } else {
        -abs_alpha_q3 - 1
    }
}

/// `get_scaled_luma_q0` (cfl.h): `ROUND_POWER_OF_TWO_SIGNED(alpha_q3 * ac_q3, 6)`.
#[inline]
fn scaled_luma_q0(alpha_q3: i32, ac_q3: i16) -> i32 {
    let v = alpha_q3 * i32::from(ac_q3);
    if v < 0 {
        -((-v + 32) >> 6)
    } else {
        (v + 32) >> 6
    }
}

/// `clip_pixel_highbd` (aom_dsp_common.h).
#[inline]
fn clip_pixel_highbd(val: i32, bd: i32) -> u16 {
    let max = match bd {
        10 => 1023,
        12 => 4095,
        _ => 255,
    };
    val.clamp(0, max) as u16
}

/// `cfl_predict_hbd_c`: `dst = clip(dst + scaled_ac)` per sample — `dst` must
/// already hold the DC prediction.
#[allow(clippy::too_many_arguments)]
pub fn cfl_predict_hbd(
    ac_buf_q3: &[i16; CFL_BUF_SQUARE],
    dst: &mut [u16],
    mut dst_off: usize,
    dst_stride: usize,
    alpha_q3: i32,
    bd: i32,
    width: usize,
    height: usize,
) {
    let mut ac_off = 0usize;
    for _ in 0..height {
        for i in 0..width {
            dst[dst_off + i] = clip_pixel_highbd(
                scaled_luma_q0(alpha_q3, ac_buf_q3[ac_off + i]) + i32::from(dst[dst_off + i]),
                bd,
            );
        }
        dst_off += dst_stride;
        ac_off += CFL_BUF_LINE;
    }
}

/// `sub8x8_adjust_offset` (cfl.c): blocks with a 4-pixel dimension share their
/// chroma block with neighbours; the *bottom/right* members of the group (odd
/// mi position on a subsampled axis) store their luma at the shifted buffer
/// offset.
fn sub8x8_adjust_offset(
    cfl: &CflCtx,
    mi_row: i32,
    mi_col: i32,
    row_out: &mut i32,
    col_out: &mut i32,
) {
    if (mi_row & 0x01) != 0 && cfl.subsampling_y != 0 {
        debug_assert_eq!(*row_out, 0);
        *row_out += 1;
    }
    if (mi_col & 0x01) != 0 && cfl.subsampling_x != 0 {
        debug_assert_eq!(*col_out, 0);
        *col_out += 1;
    }
}

/// `cfl_store` (cfl.c): subsample one reconstructed luma tx block into
/// `recon_buf_q3` at the chroma-scaled `(row, col)` position (mi units), and
/// track the written surface. `input`/`in_off`/`in_stride` address the luma
/// pixels of the tx block.
fn cfl_store(
    cfl: &mut CflCtx,
    input: &[u16],
    in_off: usize,
    in_stride: usize,
    row: i32,
    col: i32,
    tx_size: usize,
) {
    let width = TX_W[tx_size] as i32;
    let height = TX_H[tx_size] as i32;
    const TX_OFF_LOG2: i32 = 2; // MI_SIZE_LOG2
    let sub_x = cfl.subsampling_x;
    let sub_y = cfl.subsampling_y;
    let store_row = row << (TX_OFF_LOG2 - sub_y);
    let store_col = col << (TX_OFF_LOG2 - sub_x);
    let store_height = height >> sub_y;
    let store_width = width >> sub_x;

    // Invalidate current parameters.
    cfl.are_parameters_computed = false;

    // Track the written surface for chroma-overrun padding.
    if col == 0 && row == 0 {
        cfl.buf_width = store_width;
        cfl.buf_height = store_height;
    } else {
        cfl.buf_width = cfl.buf_width.max(store_col + store_width);
        cfl.buf_height = cfl.buf_height.max(store_row + store_height);
    }

    debug_assert!(store_row + store_height <= CFL_BUF_LINE as i32);
    debug_assert!(store_col + store_width <= CFL_BUF_LINE as i32);

    let out_off = (store_row as usize) * CFL_BUF_LINE + store_col as usize;
    let (w, h) = (width as usize, height as usize);
    if sub_x == 1 {
        if sub_y == 1 {
            subsample_420_hbd(
                input,
                in_off,
                in_stride,
                &mut cfl.recon_buf_q3,
                out_off,
                w,
                h,
            );
        } else {
            subsample_422_hbd(
                input,
                in_off,
                in_stride,
                &mut cfl.recon_buf_q3,
                out_off,
                w,
                h,
            );
        }
    } else {
        subsample_444_hbd(
            input,
            in_off,
            in_stride,
            &mut cfl.recon_buf_q3,
            out_off,
            w,
            h,
        );
    }
}

/// `cfl_store_tx` (cfl.c) — the DECODER's per-luma-txb store
/// (`predict_and_reconstruct_intra_block`): store the just-reconstructed luma
/// tx block at `(row, col)` (the txb offset within the block, luma mi units)
/// into the CfL buffer. `block_off` addresses the *block origin* in the luma
/// plane (C's `pd->dst.buf`); the input pixels are read at
/// `block_off + (row * stride + col) * 4`. Blocks with a 4-pixel dimension
/// apply the sub-8x8 shared-chroma offset adjustment from the block's mi
/// position.
#[allow(clippy::too_many_arguments)]
pub fn cfl_store_tx(
    cfl: &mut CflCtx,
    luma: &[u16],
    block_off: usize,
    stride: usize,
    mut row: i32,
    mut col: i32,
    tx_size: usize,
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
) {
    let in_off = block_off + ((row as usize * stride + col as usize) << 2);
    if BLK_H[bsize] == 4 || BLK_W[bsize] == 4 {
        // Only dimensions of size 4 can have an odd offset.
        debug_assert!(!((col & 1) != 0 && TX_W[tx_size] != 4));
        debug_assert!(!((row & 1) != 0 && TX_H[tx_size] != 4));
        sub8x8_adjust_offset(cfl, mi_row, mi_col, &mut row, &mut col);
    }
    cfl_store(cfl, luma, in_off, stride, row, col, tx_size);
}

/// `av1_cfl_predict_block` (cfl.c), decoder path (no DC-prediction cache):
/// lazily compute the padded zero-mean AC parameters for this chroma tx block,
/// derive the plane's `alpha_q3` from the coded CfL joint sign + alpha index,
/// and add the scaled AC into `dst` (which holds the DC prediction). `plane`
/// is 1 (U) or 2 (V).
#[allow(clippy::too_many_arguments)]
pub fn cfl_predict_block(
    cfl: &mut CflCtx,
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    tx_size: usize,
    plane: usize,
    cfl_alpha_idx: i32,
    cfl_joint_sign: i32,
    bd: i32,
) {
    if !cfl.are_parameters_computed {
        cfl.compute_parameters(tx_size);
    }
    let alpha_q3 = cfl_idx_to_alpha(cfl_alpha_idx, cfl_joint_sign, plane);
    debug_assert!((TX_H[tx_size] - 1) * CFL_BUF_LINE + TX_W[tx_size] <= CFL_BUF_SQUARE);
    cfl_predict_hbd(
        &cfl.ac_buf_q3,
        dst,
        dst_off,
        dst_stride,
        alpha_q3,
        bd,
        TX_W[tx_size],
        TX_H[tx_size],
    );
}
