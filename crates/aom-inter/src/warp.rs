//! `warp` — bit-exact AV1 **local warped motion** (`WARPED_CAUSAL`) for the
//! decoder, lowbd (bd = 8). Port of libaom v3.14.1's warp core:
//!
//! ```text
//! av1_find_projection    (av1/common/warped_motion.c:906)
//!   -> find_affine_int    (:796)   least-squares 6-param affine model
//!   -> av1_get_shear_params (:243) alpha/beta/gamma/delta + shear validity
//! av1_warp_plane / av1_warp_affine_c (:518)  the per-8x8 affine warp filter
//! ```
//!
//! `av1_findSamples` / `av1_selectSamples` (the neighbour-sample gather that
//! feeds `find_projection`) is a decode-side concern (it walks the mode-info
//! grid) and lives in the decoder driver; this module owns the pure arithmetic:
//! the model derivation and the warp filter kernel, both differentially locked
//! vs the **real exported C** (`av1_warp_affine_c`, `av1_find_projection`,
//! `av1_get_shear_params`) in `tests/warp_diff.rs`.
//!
//! # Scope
//! - lowbd (bd = 8), **single-reference, non-compound** (`is_compound == 0`)
//!   warp prediction — the `WARPED_CAUSAL` decode path
//!   (`av1_make_inter_predictor` -> `WARP_PRED` -> `av1_warp_plane`), with the
//!   decoder's fixed `ConvolveParams` (`round_0 = 3`, so `reduce_bits_horiz = 3`,
//!   `reduce_bits_vert = 11`);
//! - the AFFINE model derivation used by both the decoder and encoder.
//!
//! **NOT** handled (asserted/absent, later chunks): highbd (bd 10/12,
//! `av1_highbd_warp_affine_c`), compound / distance-weighted warp
//! (`conv_params.is_compound`), and the `USE_LIMITED_PREC_MULT` model path
//! (off in the reference build).

#![allow(clippy::too_many_arguments)]

// --- constants (av1/common/mv.h, warped_motion.h, aom_dsp/aom_filter.h) ---
const WARPEDMODEL_PREC_BITS: i32 = 16;
const WARPEDPIXEL_PREC_SHIFTS: i32 = 64;
const WARPEDDIFF_PREC_BITS: i32 = 10; // WARPEDMODEL_PREC_BITS - WARPEDPIXEL_PREC_BITS
const WARP_PARAM_REDUCE_BITS: i32 = 6;
const WARPEDMODEL_TRANS_CLAMP: i32 = 128 << WARPEDMODEL_PREC_BITS; // 1<<23
const WARPEDMODEL_NONDIAGAFFINE_CLAMP: i32 = 1 << (WARPEDMODEL_PREC_BITS - 3); // 1<<13
const FILTER_BITS: i32 = 7;
const DIV_LUT_PREC_BITS: i32 = 14;
const DIV_LUT_BITS: i32 = 8;
const LS_MV_MAX: i32 = 256;
const LS_STEP: i32 = 8;
const LS_MAT_DOWN_BITS: i32 = 2;
const MI_SIZE: i32 = 4;

/// `TransformationType::AFFINE` (av1/common/mv.h) — `DEFAULT_WMTYPE`.
pub const AFFINE: u8 = 3;

/// `WarpedMotionParams` (av1/common/mv.h) reduced to the fields the decoder warp
/// path reads/writes: the 6-param model `wmmat`, the derived shear params
/// (`alpha`/`beta`/`gamma`/`delta`), `wmtype`, and `invalid`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WarpedMotionParams {
    pub wmmat: [i32; 6],
    pub alpha: i16,
    pub beta: i16,
    pub gamma: i16,
    pub delta: i16,
    pub wmtype: u8,
    pub invalid: u8,
}

// --- ROUND_POWER_OF_TWO family (aom_ports/mem.h) ---
#[inline]
fn round_power_of_two(value: i32, n: i32) -> i32 {
    (value + ((1 << n) >> 1)) >> n
}
#[inline]
fn round_power_of_two_64(value: i64, n: i32) -> i64 {
    (value + ((1i64 << n) >> 1)) >> n
}
#[inline]
fn round_power_of_two_signed(value: i32, n: i32) -> i32 {
    if value < 0 {
        -round_power_of_two(-value, n)
    } else {
        round_power_of_two(value, n)
    }
}
#[inline]
fn round_power_of_two_signed_64(value: i64, n: i32) -> i64 {
    if value < 0 {
        -round_power_of_two_64(-value, n)
    } else {
        round_power_of_two_64(value, n)
    }
}
#[inline]
fn clamp_i32(value: i32, low: i32, high: i32) -> i32 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}
#[inline]
fn clamp_i64(value: i64, low: i64, high: i64) -> i64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}
/// `get_msb(n)` (aom_ports/bitops.h) — index of the most significant set bit;
/// valid only for `n != 0`.
#[inline]
fn get_msb(n: u32) -> i32 {
    debug_assert!(n != 0);
    31 - n.leading_zeros() as i32
}

/// `resolve_divisor_64` (warped_motion.c:172) — decompose `d` into `y/2^shift`
/// with `y` at `DIV_LUT_PREC_BITS`. Returns `(y, shift)`.
fn resolve_divisor_64(d: u64) -> (i16, i16) {
    let mut shift: i32 = if (d >> 32) != 0 {
        get_msb((d >> 32) as u32) + 32
    } else {
        get_msb(d as u32)
    };
    let e: i64 = (d - (1u64 << shift)) as i64;
    let f: i64 = if shift > DIV_LUT_BITS {
        round_power_of_two_64(e, shift - DIV_LUT_BITS)
    } else {
        e << (DIV_LUT_BITS - shift)
    };
    shift += DIV_LUT_PREC_BITS;
    (DIV_LUT[f as usize] as i16, shift as i16)
}

/// `resolve_divisor_32` (warped_motion.c:189).
fn resolve_divisor_32(d: u32) -> (i16, i16) {
    let mut shift: i32 = get_msb(d);
    let e: i32 = (d - (1u32 << shift)) as i32;
    let f: i32 = if shift > DIV_LUT_BITS {
        round_power_of_two(e, shift - DIV_LUT_BITS)
    } else {
        e << (DIV_LUT_BITS - shift)
    };
    shift += DIV_LUT_PREC_BITS;
    (DIV_LUT[f as usize] as i16, shift as i16)
}

#[inline]
fn is_affine_valid(wm: &WarpedMotionParams) -> bool {
    wm.wmmat[2] > 0
}

/// `is_affine_shear_allowed` (warped_motion.c:210).
#[inline]
fn is_affine_shear_allowed(alpha: i16, beta: i16, gamma: i16, delta: i16) -> bool {
    !((4 * (alpha as i32).abs() + 7 * (beta as i32).abs() >= (1 << WARPEDMODEL_PREC_BITS))
        || (4 * (gamma as i32).abs() + 4 * (delta as i32).abs() >= (1 << WARPEDMODEL_PREC_BITS)))
}

/// `av1_get_shear_params` (warped_motion.c:243) — derive alpha/beta/gamma/delta
/// from `wmmat`; returns `false` for an invalid affine set or a shear the fast
/// warp filter can't represent. (The `check_model_consistency` debug asserts are
/// `NDEBUG`-gated in C and omitted.)
pub fn get_shear_params(wm: &mut WarpedMotionParams) -> bool {
    if !is_affine_valid(wm) {
        return false;
    }
    let mat = wm.wmmat;
    wm.alpha = clamp_i32(
        mat[2] - (1 << WARPEDMODEL_PREC_BITS),
        i16::MIN as i32,
        i16::MAX as i32,
    ) as i16;
    wm.beta = clamp_i32(mat[3], i16::MIN as i32, i16::MAX as i32) as i16;

    let (yv, shift) = resolve_divisor_32(mat[2].unsigned_abs());
    let shift = shift as i32;
    let y: i16 = (yv as i32 * if mat[2] < 0 { -1 } else { 1 }) as i16;

    let v: i64 = ((mat[4] as i64) * (1i64 << WARPEDMODEL_PREC_BITS)) * (y as i64);
    wm.gamma = clamp_i32(
        round_power_of_two_signed_64(v, shift) as i32,
        i16::MIN as i32,
        i16::MAX as i32,
    ) as i16;
    let v: i64 = ((mat[3] as i64) * (mat[4] as i64)) * (y as i64);
    wm.delta = clamp_i32(
        mat[5] - round_power_of_two_signed_64(v, shift) as i32 - (1 << WARPEDMODEL_PREC_BITS),
        i16::MIN as i32,
        i16::MAX as i32,
    ) as i16;

    wm.alpha = (round_power_of_two_signed(wm.alpha as i32, WARP_PARAM_REDUCE_BITS)
        * (1 << WARP_PARAM_REDUCE_BITS)) as i16;
    wm.beta = (round_power_of_two_signed(wm.beta as i32, WARP_PARAM_REDUCE_BITS)
        * (1 << WARP_PARAM_REDUCE_BITS)) as i16;
    wm.gamma = (round_power_of_two_signed(wm.gamma as i32, WARP_PARAM_REDUCE_BITS)
        * (1 << WARP_PARAM_REDUCE_BITS)) as i16;
    wm.delta = (round_power_of_two_signed(wm.delta as i32, WARP_PARAM_REDUCE_BITS)
        * (1 << WARP_PARAM_REDUCE_BITS)) as i16;

    is_affine_shear_allowed(wm.alpha, wm.beta, wm.gamma, wm.delta)
}

// LS_SQUARE / LS_PRODUCT{1,2} (warped_motion.c:709) — LS_STEP == 8, downshift 4.
#[inline]
fn ls_square(a: i32) -> i32 {
    (a * a * 4 + a * 4 * LS_STEP + LS_STEP * LS_STEP * 2) >> (2 + LS_MAT_DOWN_BITS)
}
#[inline]
fn ls_product1(a: i32, b: i32) -> i32 {
    (a * b * 4 + (a + b) * 2 * LS_STEP + LS_STEP * LS_STEP) >> (2 + LS_MAT_DOWN_BITS)
}
#[inline]
fn ls_product2(a: i32, b: i32) -> i32 {
    (a * b * 4 + (a + b) * 2 * LS_STEP + LS_STEP * LS_STEP * 2) >> (2 + LS_MAT_DOWN_BITS)
}

// get_mult_shift_{diag,ndiag} — the USE_LIMITED_PREC_MULT == 0 variants
// (warped_motion.c:780).
#[inline]
fn get_mult_shift_ndiag(px: i64, idet: i16, shift: i32) -> i32 {
    let v: i64 = px * (idet as i64);
    clamp_i64(
        round_power_of_two_signed_64(v, shift),
        -(WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64) + 1,
        WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64 - 1,
    ) as i32
}
#[inline]
fn get_mult_shift_diag(px: i64, idet: i16, shift: i32) -> i32 {
    let v: i64 = px * (idet as i64);
    clamp_i64(
        round_power_of_two_signed_64(v, shift),
        (1i64 << WARPEDMODEL_PREC_BITS) - WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64 + 1,
        (1i64 << WARPEDMODEL_PREC_BITS) + WARPEDMODEL_NONDIAGAFFINE_CLAMP as i64 - 1,
    ) as i32
}

/// `find_affine_int` (warped_motion.c:796). `bw`/`bh` are the block size in
/// pixels (`block_size_wide/high[bsize]`); `pts1`/`pts2` are the sample source /
/// in-reference points (1/8-pel), `np` of them. Returns 1 on a degenerate set
/// (writes nothing usable), 0 on success (fills `wm.wmmat`).
#[allow(clippy::needless_range_loop)]
fn find_affine_int(
    np: usize,
    pts1: &[i32],
    pts2: &[i32],
    bw: i32,
    bh: i32,
    mvy: i32,
    mvx: i32,
    wm: &mut WarpedMotionParams,
    mi_row: i32,
    mi_col: i32,
) -> i32 {
    let mut a = [[0i32; 2]; 2];
    let mut bx = [0i32; 2];
    let mut by = [0i32; 2];

    let rsuy = bh / 2 - 1;
    let rsux = bw / 2 - 1;
    let suy = rsuy * 8;
    let sux = rsux * 8;
    let duy = suy + mvy;
    let dux = sux + mvx;

    for i in 0..np {
        let dx = pts2[i * 2] - dux;
        let dy = pts2[i * 2 + 1] - duy;
        let sx = pts1[i * 2] - sux;
        let sy = pts1[i * 2 + 1] - suy;
        if (sx - dx).abs() < LS_MV_MAX && (sy - dy).abs() < LS_MV_MAX {
            a[0][0] += ls_square(sx);
            a[0][1] += ls_product1(sx, sy);
            a[1][1] += ls_square(sy);
            bx[0] += ls_product2(sx, dx);
            bx[1] += ls_product1(sy, dx);
            by[0] += ls_product1(sx, dy);
            by[1] += ls_product2(sy, dy);
        }
    }

    let det: i64 = (a[0][0] as i64) * (a[1][1] as i64) - (a[0][1] as i64) * (a[0][1] as i64);
    if det == 0 {
        return 1;
    }

    let (idet_mag, shift_raw) = resolve_divisor_64(det.unsigned_abs());
    let mut idet: i16 = (idet_mag as i32 * if det < 0 { -1 } else { 1 }) as i16;
    let mut shift: i32 = shift_raw as i32 - WARPEDMODEL_PREC_BITS;
    if shift < 0 {
        idet = ((idet as i32) << (-shift)) as i16;
        shift = 0;
    }

    let px0 = (a[1][1] as i64) * (bx[0] as i64) - (a[0][1] as i64) * (bx[1] as i64);
    let px1 = -(a[0][1] as i64) * (bx[0] as i64) + (a[0][0] as i64) * (bx[1] as i64);
    let py0 = (a[1][1] as i64) * (by[0] as i64) - (a[0][1] as i64) * (by[1] as i64);
    let py1 = -(a[0][1] as i64) * (by[0] as i64) + (a[0][0] as i64) * (by[1] as i64);

    wm.wmmat[2] = get_mult_shift_diag(px0, idet, shift);
    wm.wmmat[3] = get_mult_shift_ndiag(px1, idet, shift);
    wm.wmmat[4] = get_mult_shift_ndiag(py0, idet, shift);
    wm.wmmat[5] = get_mult_shift_diag(py1, idet, shift);

    let isuy = mi_row * MI_SIZE + rsuy;
    let isux = mi_col * MI_SIZE + rsux;
    let vx = mvx * (1 << (WARPEDMODEL_PREC_BITS - 3))
        - (isux * (wm.wmmat[2] - (1 << WARPEDMODEL_PREC_BITS)) + isuy * wm.wmmat[3]);
    let vy = mvy * (1 << (WARPEDMODEL_PREC_BITS - 3))
        - (isux * wm.wmmat[4] + isuy * (wm.wmmat[5] - (1 << WARPEDMODEL_PREC_BITS)));
    wm.wmmat[0] = clamp_i32(vx, -WARPEDMODEL_TRANS_CLAMP, WARPEDMODEL_TRANS_CLAMP - 1);
    wm.wmmat[1] = clamp_i32(vy, -WARPEDMODEL_TRANS_CLAMP, WARPEDMODEL_TRANS_CLAMP - 1);
    0
}

/// `av1_find_projection` (warped_motion.c:906). The caller sets `wm.wmtype =
/// AFFINE`. Returns 1 when the block cannot use a valid warp model (caller marks
/// `wm.invalid = 1`), 0 on success.
pub fn find_projection(
    np: usize,
    pts1: &[i32],
    pts2: &[i32],
    bw: i32,
    bh: i32,
    mvy: i32,
    mvx: i32,
    wm: &mut WarpedMotionParams,
    mi_row: i32,
    mi_col: i32,
) -> i32 {
    debug_assert_eq!(wm.wmtype, AFFINE);
    if find_affine_int(np, pts1, pts2, bw, bh, mvy, mvx, wm, mi_row, mi_col) != 0 {
        return 1;
    }
    if !get_shear_params(wm) {
        return 1;
    }
    0
}

/// `av1_warp_affine_c` (warped_motion.c:518) — the bd8, **non-compound** affine
/// warp filter. `ref_plane`/`width`/`height`/`stride` describe the reference
/// plane (edge-clamped internally, so no pre-bordering); `pred`/`pred_off`/
/// `p_stride` the destination (the `p_width`×`p_height` block written at
/// `pred_off`, block-relative). `p_col`/`p_row` are the block's pixel position in
/// the (sub-sampled) plane. Uses the decoder's fixed `ConvolveParams`
/// (`round_0 = 3`).
#[allow(clippy::needless_range_loop)]
pub fn warp_affine(
    mat: &[i32; 6],
    ref_plane: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    pred: &mut [u16],
    pred_off: usize,
    p_stride: usize,
    p_col: i32,
    p_row: i32,
    p_width: usize,
    p_height: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    alpha: i16,
    beta: i16,
    gamma: i16,
    delta: i16,
) {
    const BD: i32 = 8;
    const ROUND_0: i32 = 3; // conv_params.round_0 for bd8 SR (get_conv_params_no_round)
    let reduce_bits_horiz = ROUND_0;
    let reduce_bits_vert = 2 * FILTER_BITS - ROUND_0; // 11 (non-compound)
    let offset_bits_horiz = BD + FILTER_BITS - 1; // 14
    let offset_bits_vert = BD + 2 * FILTER_BITS - ROUND_0; // 19

    let ssx = subsampling_x as i32;
    let ssy = subsampling_y as i32;
    let w = width as i32;
    let h = height as i32;
    let pw = p_width as i32;
    let ph = p_height as i32;

    let alpha = alpha as i32;
    let beta = beta as i32;
    let gamma = gamma as i32;
    let delta = delta as i32;

    let mut tmp = [0i32; 15 * 8];

    let mut i = p_row;
    while i < p_row + ph {
        let mut j = p_col;
        while j < p_col + pw {
            let src_x = (j + 4) << ssx;
            let src_y = (i + 4) << ssy;
            let dst_x = mat[2] as i64 * src_x as i64 + mat[3] as i64 * src_y as i64 + mat[0] as i64;
            let dst_y = mat[4] as i64 * src_x as i64 + mat[5] as i64 * src_y as i64 + mat[1] as i64;
            let x4 = dst_x >> ssx;
            let y4 = dst_y >> ssy;

            let ix4 = (x4 >> WARPEDMODEL_PREC_BITS) as i32;
            let mut sx4 = (x4 & ((1i64 << WARPEDMODEL_PREC_BITS) - 1)) as i32;
            let iy4 = (y4 >> WARPEDMODEL_PREC_BITS) as i32;
            let mut sy4 = (y4 & ((1i64 << WARPEDMODEL_PREC_BITS) - 1)) as i32;

            sx4 += alpha * (-4) + beta * (-4);
            sy4 += gamma * (-4) + delta * (-4);

            sx4 &= !((1 << WARP_PARAM_REDUCE_BITS) - 1);
            sy4 &= !((1 << WARP_PARAM_REDUCE_BITS) - 1);

            // Horizontal filter: 15 rows of 8.
            for k in -7..8 {
                let iy = clamp_i32(iy4 + k, 0, h - 1);
                let mut sx = sx4 + beta * (k + 4);
                for l in -4..4 {
                    let ix = ix4 + l - 3;
                    let offs = (round_power_of_two(sx, WARPEDDIFF_PREC_BITS)
                        + WARPEDPIXEL_PREC_SHIFTS) as usize;
                    let coeffs = &AV1_WARPED_FILTER[offs];
                    let mut sum = 1i32 << offset_bits_horiz;
                    for m in 0..8usize {
                        let sample_x = clamp_i32(ix + m as i32, 0, w - 1);
                        sum += ref_plane[iy as usize * stride + sample_x as usize] as i32
                            * coeffs[m] as i32;
                    }
                    sum = round_power_of_two(sum, reduce_bits_horiz);
                    tmp[((k + 7) * 8 + (l + 4)) as usize] = sum;
                    sx += alpha;
                }
            }

            // Vertical filter (non-compound), cropping at the block edge.
            let kmax = 4.min(p_row + ph - i - 4);
            let lmax = 4.min(p_col + pw - j - 4);
            for k in -4..kmax {
                let mut sy = sy4 + delta * (k + 4);
                for l in -4..lmax {
                    let offs = (round_power_of_two(sy, WARPEDDIFF_PREC_BITS)
                        + WARPEDPIXEL_PREC_SHIFTS) as usize;
                    let coeffs = &AV1_WARPED_FILTER[offs];
                    let mut sum = 1i32 << offset_bits_vert;
                    for m in 0..8usize {
                        sum += tmp[((k + m as i32 + 4) * 8 + (l + 4)) as usize] * coeffs[m] as i32;
                    }
                    sum = round_power_of_two(sum, reduce_bits_vert);
                    let out_row = (i - p_row + k + 4) as usize;
                    let out_col = (j - p_col + l + 4) as usize;
                    let px = sum - (1 << (BD - 1)) - (1 << BD);
                    pred[pred_off + out_row * p_stride + out_col] = clamp_i32(px, 0, 255) as u16;
                    sy += gamma;
                }
            }
            j += 8;
        }
        i += 8;
    }
}

pub(crate) static AV1_WARPED_FILTER: [[i16; 8]; 193] = [
    [0, 0, 127, 1, 0, 0, 0, 0],
    [0, -1, 127, 2, 0, 0, 0, 0],
    [1, -3, 127, 4, -1, 0, 0, 0],
    [1, -4, 126, 6, -2, 1, 0, 0],
    [1, -5, 126, 8, -3, 1, 0, 0],
    [1, -6, 125, 11, -4, 1, 0, 0],
    [1, -7, 124, 13, -4, 1, 0, 0],
    [2, -8, 123, 15, -5, 1, 0, 0],
    [2, -9, 122, 18, -6, 1, 0, 0],
    [2, -10, 121, 20, -6, 1, 0, 0],
    [2, -11, 120, 22, -7, 2, 0, 0],
    [2, -12, 119, 25, -8, 2, 0, 0],
    [3, -13, 117, 27, -8, 2, 0, 0],
    [3, -13, 116, 29, -9, 2, 0, 0],
    [3, -14, 114, 32, -10, 3, 0, 0],
    [3, -15, 113, 35, -10, 2, 0, 0],
    [3, -15, 111, 37, -11, 3, 0, 0],
    [3, -16, 109, 40, -11, 3, 0, 0],
    [3, -16, 108, 42, -12, 3, 0, 0],
    [4, -17, 106, 45, -13, 3, 0, 0],
    [4, -17, 104, 47, -13, 3, 0, 0],
    [4, -17, 102, 50, -14, 3, 0, 0],
    [4, -17, 100, 52, -14, 3, 0, 0],
    [4, -18, 98, 55, -15, 4, 0, 0],
    [4, -18, 96, 58, -15, 3, 0, 0],
    [4, -18, 94, 60, -16, 4, 0, 0],
    [4, -18, 91, 63, -16, 4, 0, 0],
    [4, -18, 89, 65, -16, 4, 0, 0],
    [4, -18, 87, 68, -17, 4, 0, 0],
    [4, -18, 85, 70, -17, 4, 0, 0],
    [4, -18, 82, 73, -17, 4, 0, 0],
    [4, -18, 80, 75, -17, 4, 0, 0],
    [4, -18, 78, 78, -18, 4, 0, 0],
    [4, -17, 75, 80, -18, 4, 0, 0],
    [4, -17, 73, 82, -18, 4, 0, 0],
    [4, -17, 70, 85, -18, 4, 0, 0],
    [4, -17, 68, 87, -18, 4, 0, 0],
    [4, -16, 65, 89, -18, 4, 0, 0],
    [4, -16, 63, 91, -18, 4, 0, 0],
    [4, -16, 60, 94, -18, 4, 0, 0],
    [3, -15, 58, 96, -18, 4, 0, 0],
    [4, -15, 55, 98, -18, 4, 0, 0],
    [3, -14, 52, 100, -17, 4, 0, 0],
    [3, -14, 50, 102, -17, 4, 0, 0],
    [3, -13, 47, 104, -17, 4, 0, 0],
    [3, -13, 45, 106, -17, 4, 0, 0],
    [3, -12, 42, 108, -16, 3, 0, 0],
    [3, -11, 40, 109, -16, 3, 0, 0],
    [3, -11, 37, 111, -15, 3, 0, 0],
    [2, -10, 35, 113, -15, 3, 0, 0],
    [3, -10, 32, 114, -14, 3, 0, 0],
    [2, -9, 29, 116, -13, 3, 0, 0],
    [2, -8, 27, 117, -13, 3, 0, 0],
    [2, -8, 25, 119, -12, 2, 0, 0],
    [2, -7, 22, 120, -11, 2, 0, 0],
    [1, -6, 20, 121, -10, 2, 0, 0],
    [1, -6, 18, 122, -9, 2, 0, 0],
    [1, -5, 15, 123, -8, 2, 0, 0],
    [1, -4, 13, 124, -7, 1, 0, 0],
    [1, -4, 11, 125, -6, 1, 0, 0],
    [1, -3, 8, 126, -5, 1, 0, 0],
    [1, -2, 6, 126, -4, 1, 0, 0],
    [0, -1, 4, 127, -3, 1, 0, 0],
    [0, 0, 2, 127, -1, 0, 0, 0],
    [0, 0, 0, 127, 1, 0, 0, 0],
    [0, 0, -1, 127, 2, 0, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -5, 127, 6, -2, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [-1, 2, -7, 126, 11, -4, 2, -1],
    [-1, 3, -8, 125, 13, -5, 2, -1],
    [-1, 3, -10, 124, 16, -6, 3, -1],
    [-1, 4, -11, 123, 18, -7, 3, -1],
    [-1, 4, -12, 122, 20, -7, 3, -1],
    [-1, 4, -13, 121, 23, -8, 3, -1],
    [-2, 5, -14, 120, 25, -9, 4, -1],
    [-1, 5, -15, 119, 27, -10, 4, -1],
    [-1, 5, -16, 118, 30, -11, 4, -1],
    [-2, 6, -17, 116, 33, -12, 5, -1],
    [-2, 6, -17, 114, 35, -12, 5, -1],
    [-2, 6, -18, 113, 38, -13, 5, -1],
    [-2, 7, -19, 111, 41, -14, 6, -2],
    [-2, 7, -19, 110, 43, -15, 6, -2],
    [-2, 7, -20, 108, 46, -15, 6, -2],
    [-2, 7, -20, 106, 49, -16, 6, -2],
    [-2, 7, -21, 104, 51, -16, 7, -2],
    [-2, 7, -21, 102, 54, -17, 7, -2],
    [-2, 8, -21, 100, 56, -18, 7, -2],
    [-2, 8, -22, 98, 59, -18, 7, -2],
    [-2, 8, -22, 96, 62, -19, 7, -2],
    [-2, 8, -22, 94, 64, -19, 7, -2],
    [-2, 8, -22, 91, 67, -20, 8, -2],
    [-2, 8, -22, 89, 69, -20, 8, -2],
    [-2, 8, -22, 87, 72, -21, 8, -2],
    [-2, 8, -21, 84, 74, -21, 8, -2],
    [-2, 8, -22, 82, 77, -21, 8, -2],
    [-2, 8, -21, 79, 79, -21, 8, -2],
    [-2, 8, -21, 77, 82, -22, 8, -2],
    [-2, 8, -21, 74, 84, -21, 8, -2],
    [-2, 8, -21, 72, 87, -22, 8, -2],
    [-2, 8, -20, 69, 89, -22, 8, -2],
    [-2, 8, -20, 67, 91, -22, 8, -2],
    [-2, 7, -19, 64, 94, -22, 8, -2],
    [-2, 7, -19, 62, 96, -22, 8, -2],
    [-2, 7, -18, 59, 98, -22, 8, -2],
    [-2, 7, -18, 56, 100, -21, 8, -2],
    [-2, 7, -17, 54, 102, -21, 7, -2],
    [-2, 7, -16, 51, 104, -21, 7, -2],
    [-2, 6, -16, 49, 106, -20, 7, -2],
    [-2, 6, -15, 46, 108, -20, 7, -2],
    [-2, 6, -15, 43, 110, -19, 7, -2],
    [-2, 6, -14, 41, 111, -19, 7, -2],
    [-1, 5, -13, 38, 113, -18, 6, -2],
    [-1, 5, -12, 35, 114, -17, 6, -2],
    [-1, 5, -12, 33, 116, -17, 6, -2],
    [-1, 4, -11, 30, 118, -16, 5, -1],
    [-1, 4, -10, 27, 119, -15, 5, -1],
    [-1, 4, -9, 25, 120, -14, 5, -2],
    [-1, 3, -8, 23, 121, -13, 4, -1],
    [-1, 3, -7, 20, 122, -12, 4, -1],
    [-1, 3, -7, 18, 123, -11, 4, -1],
    [-1, 3, -6, 16, 124, -10, 3, -1],
    [-1, 2, -5, 13, 125, -8, 3, -1],
    [-1, 2, -4, 11, 126, -7, 2, -1],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -2, 6, 127, -5, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, 0, 2, 127, -1, 0, 0],
    [0, 0, 0, 1, 127, 0, 0, 0],
    [0, 0, 0, -1, 127, 2, 0, 0],
    [0, 0, 1, -3, 127, 4, -1, 0],
    [0, 0, 1, -4, 126, 6, -2, 1],
    [0, 0, 1, -5, 126, 8, -3, 1],
    [0, 0, 1, -6, 125, 11, -4, 1],
    [0, 0, 1, -7, 124, 13, -4, 1],
    [0, 0, 2, -8, 123, 15, -5, 1],
    [0, 0, 2, -9, 122, 18, -6, 1],
    [0, 0, 2, -10, 121, 20, -6, 1],
    [0, 0, 2, -11, 120, 22, -7, 2],
    [0, 0, 2, -12, 119, 25, -8, 2],
    [0, 0, 3, -13, 117, 27, -8, 2],
    [0, 0, 3, -13, 116, 29, -9, 2],
    [0, 0, 3, -14, 114, 32, -10, 3],
    [0, 0, 3, -15, 113, 35, -10, 2],
    [0, 0, 3, -15, 111, 37, -11, 3],
    [0, 0, 3, -16, 109, 40, -11, 3],
    [0, 0, 3, -16, 108, 42, -12, 3],
    [0, 0, 4, -17, 106, 45, -13, 3],
    [0, 0, 4, -17, 104, 47, -13, 3],
    [0, 0, 4, -17, 102, 50, -14, 3],
    [0, 0, 4, -17, 100, 52, -14, 3],
    [0, 0, 4, -18, 98, 55, -15, 4],
    [0, 0, 4, -18, 96, 58, -15, 3],
    [0, 0, 4, -18, 94, 60, -16, 4],
    [0, 0, 4, -18, 91, 63, -16, 4],
    [0, 0, 4, -18, 89, 65, -16, 4],
    [0, 0, 4, -18, 87, 68, -17, 4],
    [0, 0, 4, -18, 85, 70, -17, 4],
    [0, 0, 4, -18, 82, 73, -17, 4],
    [0, 0, 4, -18, 80, 75, -17, 4],
    [0, 0, 4, -18, 78, 78, -18, 4],
    [0, 0, 4, -17, 75, 80, -18, 4],
    [0, 0, 4, -17, 73, 82, -18, 4],
    [0, 0, 4, -17, 70, 85, -18, 4],
    [0, 0, 4, -17, 68, 87, -18, 4],
    [0, 0, 4, -16, 65, 89, -18, 4],
    [0, 0, 4, -16, 63, 91, -18, 4],
    [0, 0, 4, -16, 60, 94, -18, 4],
    [0, 0, 3, -15, 58, 96, -18, 4],
    [0, 0, 4, -15, 55, 98, -18, 4],
    [0, 0, 3, -14, 52, 100, -17, 4],
    [0, 0, 3, -14, 50, 102, -17, 4],
    [0, 0, 3, -13, 47, 104, -17, 4],
    [0, 0, 3, -13, 45, 106, -17, 4],
    [0, 0, 3, -12, 42, 108, -16, 3],
    [0, 0, 3, -11, 40, 109, -16, 3],
    [0, 0, 3, -11, 37, 111, -15, 3],
    [0, 0, 2, -10, 35, 113, -15, 3],
    [0, 0, 3, -10, 32, 114, -14, 3],
    [0, 0, 2, -9, 29, 116, -13, 3],
    [0, 0, 2, -8, 27, 117, -13, 3],
    [0, 0, 2, -8, 25, 119, -12, 2],
    [0, 0, 2, -7, 22, 120, -11, 2],
    [0, 0, 1, -6, 20, 121, -10, 2],
    [0, 0, 1, -6, 18, 122, -9, 2],
    [0, 0, 1, -5, 15, 123, -8, 2],
    [0, 0, 1, -4, 13, 124, -7, 1],
    [0, 0, 1, -4, 11, 125, -6, 1],
    [0, 0, 1, -3, 8, 126, -5, 1],
    [0, 0, 1, -2, 6, 126, -4, 1],
    [0, 0, 0, -1, 4, 127, -3, 1],
    [0, 0, 0, 0, 2, 127, -1, 0],
    [0, 0, 0, 0, 2, 127, -1, 0],
];

static DIV_LUT: [u16; 257] = [
    16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768, 15709, 15650,
    15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142, 15087, 15033, 14980, 14926,
    14873, 14821, 14769, 14717, 14665, 14614, 14564, 14513, 14463, 14413, 14364, 14315, 14266,
    14218, 14170, 14122, 14075, 14028, 13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662,
    13618, 13574, 13530, 13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107,
    13066, 13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633, 12596,
    12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228, 12193, 12157, 12122,
    12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848, 11815, 11782, 11749, 11716, 11683,
    11651, 11619, 11586, 11555, 11523, 11491, 11460, 11429, 11398, 11367, 11336, 11305, 11275,
    11245, 11215, 11185, 11155, 11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894,
    10866, 10838, 10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
    10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255, 10230, 10205,
    10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986, 9963, 9939, 9916, 9892, 9869,
    9846, 9823, 9800, 9777, 9754, 9732, 9709, 9687, 9664, 9642, 9620, 9598, 9576, 9554, 9533, 9511,
    9489, 9468, 9447, 9425, 9404, 9383, 9362, 9341, 9321, 9300, 9279, 9259, 9239, 9218, 9198, 9178,
    9158, 9138, 9118, 9098, 9079, 9059, 9039, 9020, 9001, 8981, 8962, 8943, 8924, 8905, 8886, 8867,
    8849, 8830, 8812, 8793, 8775, 8756, 8738, 8720, 8702, 8684, 8666, 8648, 8630, 8613, 8595, 8577,
    8560, 8542, 8525, 8508, 8490, 8473, 8456, 8439, 8422, 8405, 8389, 8372, 8355, 8339, 8322, 8306,
    8289, 8273, 8257, 8240, 8224, 8208, 8192,
];
