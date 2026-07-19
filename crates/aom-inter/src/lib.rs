//! aom-inter — bit-exact AV1 **single-reference translational** motion-compensated
//! prediction for the decoder, lowbd (bd = 8). Port of libaom v3.14.1's decoder
//! inter-prediction core:
//!
//! ```text
//! dec_build_inter_predictor      (av1/decoder/decodeframe.c:681)
//!   -> dec_calc_subpel_params    (decodeframe.c:565, unscaled branch :620)
//!      + extend_mc_border        (decodeframe.c:526) / build_mc_border (:455)
//!   -> av1_make_inter_predictor  (av1/common/reconinter.c:77, TRANSLATION_PRED)
//!      -> inter_predictor        (reconinter.h:255, unscaled)
//!         -> av1_convolve_2d_facade (av1/common/convolve.c:638)
//!            -> convolve_2d_facade_single (convolve.c:616): copy / x / y / 2d
//! ```
//!
//! The byte-exact separable convolution kernels live in the sibling crate
//! [`aom_convolve`] (already differentially locked vs the real C
//! `av1_convolve_{x,y,2d}_sr_c`). This crate adds the pieces around them that the
//! decoder inter path needs: the sub-pel derivation (MV + block position +
//! subsampling -> `subpel_x`/`subpel_y` + integer reference offset), the
//! out-of-frame reference border replication (`build_mc_border`), the full-pel
//! copy case, and the `convolve_2d_facade_single` dispatch.
//!
//! # Scope / what is and isn't handled (matches the chunk-1d walking skeleton)
//! Handled, bit-exact:
//! - lowbd (bd = 8) single-reference, translational (`TRANSLATION_PRED`) prediction;
//! - the **unscaled** reference path only (`is_scaled == false`, i.e. the reference
//!   frame has the same dimensions as the current frame — `x_step_q4 == y_step_q4
//!   == SUBPEL_SHIFTS`). This is the common inter case;
//! - all four facade sub-cases: full-pel copy (`subpel == 0,0`), x-only, y-only,
//!   separable 2-D — including **dual** interpolation filters (`filter_x !=
//!   filter_y`);
//! - the EIGHTTAP_REGULAR (0) / EIGHTTAP_SMOOTH (1) / MULTITAP_SHARP (2) filter
//!   families, 8-tap and — for a block side `<= 4` — the 4-tap kernels
//!   (`av1_get_interp_filter_params_with_block_size`, filter.h:248; selected per
//!   direction, x by `w` and y by `h`). The 16x16 ratchet's `BLOCK_16X4` luma and
//!   `8x2` chroma strips exercise this;
//! - out-of-frame reference reads via edge replication (`build_mc_border`).
//!
//! **NOT** handled (asserted / documented, for later chunks):
//! - highbd (bd 10/12), compound / masked / OBMC / warp prediction, and the scaled
//!   reference path (`av1_convolve_2d_scale`);
//! - IntraBC's 2-tap bilinear filter (`av1_intrabc_filter_params`).
//!
//! # Differential coverage
//! `tests/inter_pred_diff.rs` locks the facade + convolution against the **real C**
//! `inter_predictor` (via the `aom-sys-ref` `shim_inter_predictor` oracle) and
//! locks `build_mc_border` against the **real C** `build_mc_border` body (via
//! `shim_build_mc_border`). The MV/subsampling -> subpel derivation
//! ([`build_inter_predictor`]) is a faithful transcription of
//! `dec_calc_subpel_params`; it is validated end-to-end by the decoder frame-MD5
//! gate (chunk 1f), while everything downstream of it is differentially locked here.

#![forbid(unsafe_code)]

pub mod warp;

// --- constants (aom_dsp/aom_filter.h, aom_scale/yv12config.h) ---
const FILTER_BITS: i32 = 7;
const ROUND0_BITS: i32 = 3;
const SUBPEL_BITS: i32 = 4;
const SUBPEL_MASK: i32 = (1 << SUBPEL_BITS) - 1; // 15
const SUBPEL_TAPS: usize = 8;
const AOM_INTERP_EXTEND: i32 = 4;

#[inline]
fn rpo2(v: i32, n: i32) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

#[inline]
fn clip_pixel(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// `av1_sub_pel_filters_4` (filter.h): the 4-tap subpel kernel for
/// `EIGHTTAP_REGULAR` / `MULTITAP_SHARP` blocks with a side `<= 4`. Stored 8-wide
/// (`taps == SUBPEL_TAPS`) with the four non-zero coefficients centred at indices
/// 2..6 and zeros elsewhere, so the ordinary 8-tap convolution loop runs it
/// unchanged.
static SUB_PEL_FILTERS_4: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -4, 126, 8, -2, 0, 0],
    [0, 0, -8, 122, 18, -4, 0, 0],
    [0, 0, -10, 116, 28, -6, 0, 0],
    [0, 0, -12, 110, 38, -8, 0, 0],
    [0, 0, -12, 102, 48, -10, 0, 0],
    [0, 0, -14, 94, 58, -10, 0, 0],
    [0, 0, -12, 84, 66, -10, 0, 0],
    [0, 0, -12, 76, 76, -12, 0, 0],
    [0, 0, -10, 66, 84, -12, 0, 0],
    [0, 0, -10, 58, 94, -14, 0, 0],
    [0, 0, -10, 48, 102, -12, 0, 0],
    [0, 0, -8, 38, 110, -12, 0, 0],
    [0, 0, -6, 28, 116, -10, 0, 0],
    [0, 0, -4, 18, 122, -8, 0, 0],
    [0, 0, -2, 8, 126, -4, 0, 0],
];

/// `av1_sub_pel_filters_4smooth` (filter.h): the 4-tap kernel for
/// `EIGHTTAP_SMOOTH` blocks with a side `<= 4`. Same 8-wide layout as
/// [`SUB_PEL_FILTERS_4`].
static SUB_PEL_FILTERS_4SMOOTH: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, 30, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, 0, 14, 54, 48, 12, 0, 0],
    [0, 0, 12, 52, 52, 12, 0, 0],
    [0, 0, 12, 48, 54, 14, 0, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 30, 0, 0],
];

/// Select the subpel kernel row for filter `ftype` (0 = regular, 1 = smooth,
/// 2 = sharp) and direction phase `subpel`. When `use4` is set the block's side
/// in this direction is `<= 4`, so libaom's
/// `av1_get_interp_filter_params_with_block_size` selects the 4-tap table
/// (`av1_interp_4tap`: regular/sharp -> `av1_sub_pel_filters_4`, smooth ->
/// `av1_sub_pel_filters_4smooth`); otherwise the 8-tap table from
/// [`aom_convolve`]. Both are 8-wide, so callers run the same convolution loop.
#[inline]
fn kernel(ftype: usize, subpel: usize, use4: bool) -> &'static [i16; 8] {
    let table: &[[i16; 8]; 16] = if use4 {
        match ftype {
            0 | 2 => &SUB_PEL_FILTERS_4,
            1 => &SUB_PEL_FILTERS_4SMOOTH,
            _ => panic!("aom-inter: unsupported InterpFilter {ftype} (0/1/2 only)"),
        }
    } else {
        match ftype {
            0 => &aom_convolve::SUB_PEL_FILTERS_8,
            1 => &aom_convolve::SUB_PEL_FILTERS_8SMOOTH,
            2 => &aom_convolve::SUB_PEL_FILTERS_8SHARP,
            _ => panic!("aom-inter: unsupported InterpFilter {ftype} (0/1/2 only)"),
        }
    };
    &table[subpel & 15]
}

/// `av1_convolve_2d_sr_c` (lowbd, SR: round_0 = 3, round_1 = 11, bits = 0) with a
/// **separate** horizontal and vertical filter — the dual-filter/mixed-tap
/// generalization of [`aom_convolve::convolve_2d_sr`] (which takes a single
/// `ftype`, all 8-tap). Used whenever the filters differ OR a side is `<= 4` (so
/// that direction uses the 4-tap kernel). `use4_x`/`use4_y` select the 4-tap table
/// per direction (`w <= 4` / `h <= 4`). `src_off` is the interior origin; `src`
/// needs a border of >= 3 (top/left) and >= 4 (bottom/right).
#[allow(clippy::too_many_arguments)]
fn convolve_2d_sr_dual(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    subpel_x: usize,
    subpel_y: usize,
    filter_x: usize,
    filter_y: usize,
    use4_x: bool,
    use4_y: bool,
) {
    const BD: i32 = 8;
    const ROUND_1: i32 = 2 * FILTER_BITS - ROUND0_BITS; // 11
    let taps = SUBPEL_TAPS;
    let fo = taps / 2 - 1; // 3
    let im_h = h + taps - 1;
    let im_stride = w;
    let xf = kernel(filter_x, subpel_x, use4_x);
    let yf = kernel(filter_y, subpel_y, use4_y);

    // Horizontal pass into an int16 intermediate.
    let mut im = vec![0i16; im_h * im_stride];
    let src_horiz = src_off as isize - fo as isize * src_stride as isize;
    for y in 0..im_h {
        for x in 0..w {
            let base = src_horiz + (y * src_stride) as isize + x as isize - fo as isize;
            let mut sum = 1i32 << (BD + FILTER_BITS - 1);
            for k in 0..taps {
                sum += xf[k] as i32 * src[(base + k as isize) as usize] as i32;
            }
            im[y * im_stride + x] = rpo2(sum, ROUND0_BITS) as i16;
        }
    }

    // Vertical pass.
    let offset_bits = BD + 2 * FILTER_BITS - ROUND0_BITS; // 19
    let round_offset = (1i32 << (offset_bits - ROUND_1)) + (1i32 << (offset_bits - ROUND_1 - 1));
    let bits = FILTER_BITS * 2 - ROUND0_BITS - ROUND_1; // 0
    for y in 0..h {
        for x in 0..w {
            let mut sum = 1i32 << offset_bits;
            for k in 0..taps {
                sum += yf[k] as i32 * im[(y + k) * im_stride + x] as i32;
            }
            let res = (rpo2(sum, ROUND_1) - round_offset) as i16;
            dst[y * dst_stride + x] = clip_pixel(rpo2(res as i32, bits));
        }
    }
}

/// `av1_convolve_x_sr_c` (lowbd SR single horizontal pass) with an explicit
/// per-direction 4-tap selection — the 4-tap-aware analogue of
/// [`aom_convolve::convolve_x_sr`], used when `w <= 4`. Byte-identical rounding
/// (`round_0 = 3`, then `FILTER_BITS - round_0`).
#[allow(clippy::too_many_arguments)]
fn convolve_x_sr_k(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    subpel_x: usize,
    filter_x: usize,
    use4_x: bool,
) {
    let fo = SUBPEL_TAPS / 2 - 1; // 3
    let bits = FILTER_BITS - ROUND0_BITS;
    let filt = kernel(filter_x, subpel_x, use4_x);
    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize + (y * src_stride) as isize + x as isize - fo as isize;
            let mut res = 0i32;
            for k in 0..SUBPEL_TAPS {
                res += filt[k] as i32 * src[(base + k as isize) as usize] as i32;
            }
            res = rpo2(res, ROUND0_BITS);
            dst[y * dst_stride + x] = clip_pixel(rpo2(res, bits));
        }
    }
}

/// `av1_convolve_y_sr_c` (lowbd SR single vertical pass) with an explicit 4-tap
/// selection — the 4-tap-aware analogue of [`aom_convolve::convolve_y_sr`], used
/// when `h <= 4`. Byte-identical rounding (`FILTER_BITS`).
#[allow(clippy::too_many_arguments)]
fn convolve_y_sr_k(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    subpel_y: usize,
    filter_y: usize,
    use4_y: bool,
) {
    let fo = SUBPEL_TAPS / 2 - 1; // 3
    let filt = kernel(filter_y, subpel_y, use4_y);
    for y in 0..h {
        for x in 0..w {
            let base =
                src_off as isize + ((y as isize - fo as isize) * src_stride as isize) + x as isize;
            let mut res = 0i32;
            for k in 0..SUBPEL_TAPS {
                res += filt[k] as i32
                    * src[(base + (k as isize) * src_stride as isize) as usize] as i32;
            }
            dst[y * dst_stride + x] = clip_pixel(rpo2(res, FILTER_BITS));
        }
    }
}

/// The unscaled `inter_predictor` facade (reconinter.h:255 ->
/// `av1_convolve_2d_facade` -> `convolve_2d_facade_single`, convolve.c:616),
/// lowbd single-ref SR.
///
/// Dispatches on `need_x = subpel_x != 0`, `need_y = subpel_y != 0`:
/// - neither -> full-pel block copy (libaom `aom_convolve_copy`, no rounding);
/// - x only  -> [`aom_convolve::convolve_x_sr`] with `filter_x`;
/// - y only  -> [`aom_convolve::convolve_y_sr`] with `filter_y`;
/// - both    -> [`aom_convolve::convolve_2d_sr`] (`filter_x == filter_y`) or the
///   dual-filter [`convolve_2d_sr_dual`].
///
/// `src`/`src_off`/`src_stride` describe the (bordered) reference region: `src_off`
/// is the block top-left, with >= 3 samples of border before and >= 4 after in each
/// direction that is sub-pel filtered. Writes the `w`×`h` predictor to
/// `dst[y * dst_stride + x]`. `subpel_x`/`subpel_y` are in `0..=SUBPEL_MASK` (15).
#[allow(clippy::too_many_arguments)]
pub fn inter_predictor(
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    w: usize,
    h: usize,
    subpel_x: usize,
    subpel_y: usize,
    filter_x: usize,
    filter_y: usize,
) {
    let need_x = subpel_x != 0;
    let need_y = subpel_y != 0;
    // av1_get_interp_filter_params_with_block_size selects the 4-tap table for a
    // side <= 4 (per direction: x uses `w`, y uses `h`). MULTITAP_SHARP2 (3) would
    // stay 8-tap, but this crate only carries filters 0/1/2.
    let use4_x = w <= 4;
    let use4_y = h <= 4;
    if !need_x && !need_y {
        // aom_convolve_copy: plain block copy, no rounding.
        for y in 0..h {
            for x in 0..w {
                dst[y * dst_stride + x] = src[src_off + y * src_stride + x];
            }
        }
    } else if need_x && !need_y {
        if use4_x {
            convolve_x_sr_k(
                src, src_off, src_stride, dst, dst_stride, w, h, subpel_x, filter_x, use4_x,
            );
        } else {
            aom_convolve::convolve_x_sr(
                src, src_off, src_stride, dst, dst_stride, w, h, subpel_x, filter_x,
            );
        }
    } else if !need_x && need_y {
        if use4_y {
            convolve_y_sr_k(
                src, src_off, src_stride, dst, dst_stride, w, h, subpel_y, filter_y, use4_y,
            );
        } else {
            aom_convolve::convolve_y_sr(
                src, src_off, src_stride, dst, dst_stride, w, h, subpel_y, filter_y,
            );
        }
    } else if filter_x == filter_y && !use4_x && !use4_y {
        aom_convolve::convolve_2d_sr(
            src, src_off, src_stride, dst, dst_stride, w, h, subpel_x, subpel_y, filter_x,
        );
    } else {
        // Dual filters and/or a 4-tap side: the mixed-kernel 2-D pass.
        convolve_2d_sr_dual(
            src, src_off, src_stride, dst, dst_stride, w, h, subpel_x, subpel_y, filter_x,
            filter_y, use4_x, use4_y,
        );
    }
}

/// `build_mc_border` (av1/decoder/decodeframe.c:455) — gather a `b_w`×`b_h` block
/// (`dst`, tightly packed, stride `b_w`) from reference plane `reff` (frame origin,
/// `ref_w`×`ref_h`, stride `ref_stride`), replicating the frame edge for any part of
/// the requested `[gx, gx + b_w) × [gy, gy + b_h)` region that lies outside the
/// plane. `gx`/`gy` may be negative. Bit-exact port of the C body (values are lowbd,
/// so u16 samples truncate losslessly to the u8 scratch libaom's convolvers consume).
#[allow(clippy::too_many_arguments)]
pub fn build_mc_border(
    reff: &[u16],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    gx: i32,
    gy: i32,
    b_w: usize,
    b_h: usize,
    dst: &mut [u8],
) {
    let w = ref_w as i32;
    let h = ref_h as i32;
    let x = gx;
    let mut y = gy;
    // C: ref_row = src - x - y*stride = plane origin; then clamp the starting row.
    let mut row: i32 = if y >= h {
        h - 1
    } else if y > 0 {
        y
    } else {
        0
    };
    for by in 0..b_h {
        let mut left = if x < 0 { (-x) as usize } else { 0 };
        if left > b_w {
            left = b_w;
        }
        let mut right = if x + b_w as i32 > w {
            (x + b_w as i32 - w) as usize
        } else {
            0
        };
        if right > b_w {
            right = b_w;
        }
        let copy = b_w - left - right;
        let row_base = row as usize * ref_stride;
        let dst_row = by * b_w;
        if left > 0 {
            let v = reff[row_base] as u8; // ref_row[0]
            for i in 0..left {
                dst[dst_row + i] = v;
            }
        }
        if copy > 0 {
            let sstart = row_base + (x + left as i32) as usize; // ref_row + x + left
            for i in 0..copy {
                dst[dst_row + left + i] = reff[sstart + i] as u8;
            }
        }
        if right > 0 {
            let v = reff[row_base + (ref_w - 1)] as u8; // ref_row[w-1]
            for i in 0..right {
                dst[dst_row + left + copy + i] = v;
            }
        }
        y += 1;
        if y > 0 && y < h {
            row += 1;
        }
    }
}

/// Single-ref translational inter predictor for one plane, lowbd (bd = 8), unscaled.
///
/// Reproduces the decoder chain `dec_calc_subpel_params` (unscaled branch,
/// decodeframe.c:620) + `extend_mc_border`/`build_mc_border` + `inter_predictor`.
///
/// - `ref_plane`: reference frame plane samples as `u16` (bd8 values `0..=255`),
///   row-major, stride `ref_stride`; `ref_w`,`ref_h` are the plane's valid
///   dimensions (this plane's pixels) used for edge replication.
/// - `dst`: destination plane; the `w`×`h` predictor is written at
///   `dst[dst_off + y*dst_stride + x]`.
/// - `blk_x`,`blk_y`: block top-left in the plane (plane pixels).
/// - `w`,`h`: block size in plane pixels (`w > 4 && h > 4`, see crate docs).
/// - `mv_row`,`mv_col`: block MV in 1/8-pel **luma** units.
/// - `ss_x`,`ss_y`: plane subsampling (0,0 luma; 1,1 chroma-420).
/// - `filter_x`,`filter_y`: InterpFilter type (0/1/2) per direction.
///
/// # Sub-pel derivation (`dec_calc_subpel_params`, decodeframe.c:620-648)
/// The luma 1/8-pel MV is scaled to this plane's q4 (1/16-pel) grid by
/// `mv * (1 << (1 - ss))` (reconinter.h:353), split into an integer reference
/// offset and a `0..=15` sub-pel phase, exactly as C.
///
/// The frame-edge MV clamp `clamp_mv_to_umv_border_sb` (reconinter.h:343) needs the
/// block's `mb_to_{left,right,top,bottom}_edge` extents, which are a
/// higher-level (frame-walk) concern; the decoder caller applies it upstream and it
/// is exercised by the chunk-1f frame-MD5 gate. This function takes the final motion
/// vector and derives the reference sampling faithfully from it.
#[allow(clippy::too_many_arguments)]
pub fn build_inter_predictor(
    ref_plane: &[u16],
    ref_stride: usize,
    ref_w: usize,
    ref_h: usize,
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    blk_x: usize,
    blk_y: usize,
    w: usize,
    h: usize,
    mv_row: i32,
    mv_col: i32,
    ss_x: usize,
    ss_y: usize,
    filter_x: usize,
    filter_y: usize,
) {
    // `w <= 4` / `h <= 4` are handled: [`inter_predictor`] selects the 4-tap
    // kernel per direction (av1_get_interp_filter_params_with_block_size). The
    // 4-tap tables are stored 8-wide with zero outer taps, so the border margin
    // (AOM_INTERP_EXTEND) and gather below are unchanged.
    debug_assert!(ss_x <= 1 && ss_y <= 1);

    // --- dec_calc_subpel_params, unscaled branch (decodeframe.c:620) ---
    // Luma 1/8-pel MV -> this plane's q4 (1/16-pel) units.
    let mv_q4_col = mv_col * (1i32 << (1 - ss_x as i32));
    let mv_q4_row = mv_row * (1i32 << (1 - ss_y as i32));
    let subpel_x = (mv_q4_col & SUBPEL_MASK) as usize; // 0..=15
    let subpel_y = (mv_q4_row & SUBPEL_MASK) as usize;
    let pos_x = (blk_x as i32) << SUBPEL_BITS;
    let pos_y = (blk_y as i32) << SUBPEL_BITS;
    let x0 = (pos_x + mv_q4_col) >> SUBPEL_BITS; // integer ref col (may be < 0)
    let y0 = (pos_y + mv_q4_row) >> SUBPEL_BITS; // integer ref row (may be < 0)

    // --- extended reference block (extend_mc_border, decodeframe.c:505-514) ---
    // x_pad/y_pad match libaom: on the unscaled path they are set iff the sub-pel
    // phase is nonzero (`subpel_*_mv || x_step_q4 != SUBPEL_SHIFTS`, and
    // x_step_q4 == SUBPEL_SHIFTS here). A sub-pel phase implies nonzero motion, so
    // the enclosing "motion or misaligned dims" guard is always satisfied when a
    // margin is needed. When no margin is needed (`subpel == 0`) the gather is the
    // block itself; when the block/margin crosses the frame it is edge-replicated —
    // both reproduce C's output whether or not C's optimizer skipped the copy.
    let pad_x = subpel_x != 0;
    let pad_y = subpel_y != 0;
    let mx = if pad_x { AOM_INTERP_EXTEND - 1 } else { 0 }; // 3
    let my = if pad_y { AOM_INTERP_EXTEND - 1 } else { 0 };
    let gx = x0 - mx;
    let gy = y0 - my;
    // b_w = (x0 + w + AOM_INTERP_EXTEND) - (x0 - (AOM_INTERP_EXTEND - 1)) = w + 7.
    let extra = (2 * AOM_INTERP_EXTEND - 1) as usize; // 7
    let b_w = w + if pad_x { extra } else { 0 };
    let b_h = h + if pad_y { extra } else { 0 };

    // Gather the bordered region into a u8 scratch (edge-replicated for OOB), then
    // run the facade and scatter u8 -> u16 dst.
    let mut scratch = vec![0u8; b_w * b_h];
    build_mc_border(
        ref_plane,
        ref_stride,
        ref_w,
        ref_h,
        gx,
        gy,
        b_w,
        b_h,
        &mut scratch,
    );
    let interior = (my as usize) * b_w + mx as usize;

    let mut tmp = vec![0u8; w * h];
    inter_predictor(
        &scratch, interior, b_w, &mut tmp, w, w, h, subpel_x, subpel_y, filter_x, filter_y,
    );

    for y in 0..h {
        for x in 0..w {
            dst[dst_off + y * dst_stride + x] = tmp[y * w + x] as u16;
        }
    }
}

// ===================================================================
// OBMC (overlapped block motion compensation) — mask tables + blend.
// ===================================================================
//
// After a block's own inter predictor is built, an `OBMC_CAUSAL` block blends
// it with predictors generated from its overlappable above/left neighbours'
// motion (each neighbour's strip MC uses the existing [`build_inter_predictor`]
// with the NEIGHBOUR's mv/ref/filter). The blend is a raised-cosine feather:
// the block's own predictor dominates far from the neighbour edge and the
// neighbour's predictor near it. Ported from `av1/common/reconinter.c`:
// `obmc_mask_{1,2,4,8,16,32,64}` (:751-782), `av1_get_obmc_mask` (:774), and the
// `aom_blend_a64_vmask` (above) / `aom_blend_a64_hmask` (left) blends
// (`aom_dsp/blend_a64_{v,h}mask.c`) used by `build_obmc_inter_pred_{above,left}`
// (:852/:891). Differentially locked in `tests/inter_pred_diff.rs`.

// obmc_mask_N[overlap_position] (reconinter.c:751-782).
static OBMC_MASK_1: [u8; 1] = [64];
static OBMC_MASK_2: [u8; 2] = [45, 64];
static OBMC_MASK_4: [u8; 4] = [39, 50, 59, 64];
static OBMC_MASK_8: [u8; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
static OBMC_MASK_16: [u8; 16] = [
    34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64,
];
static OBMC_MASK_32: [u8; 32] = [
    33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
];
static OBMC_MASK_64: [u8; 64] = [
    33, 34, 35, 35, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 44, 44, 45, 46, 47, 47, 48, 49, 50, 51,
    51, 51, 52, 52, 53, 54, 55, 56, 56, 56, 57, 57, 58, 58, 59, 60, 60, 60, 60, 60, 61, 62, 62, 62,
    62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
];

/// `av1_get_obmc_mask` (reconinter.c:774): the raised-cosine OBMC feather mask
/// for an overlap length of `length` (a power of two in `1..=64`).
pub fn get_obmc_mask(length: usize) -> &'static [u8] {
    match length {
        1 => &OBMC_MASK_1,
        2 => &OBMC_MASK_2,
        4 => &OBMC_MASK_4,
        8 => &OBMC_MASK_8,
        16 => &OBMC_MASK_16,
        32 => &OBMC_MASK_32,
        64 => &OBMC_MASK_64,
        _ => panic!("aom-inter: invalid OBMC mask length {length} (want a power of two 1..=64)"),
    }
}

const AOM_BLEND_A64_ROUND_BITS: i32 = 6;
const AOM_BLEND_A64_MAX_ALPHA: i32 = 1 << AOM_BLEND_A64_ROUND_BITS; // 64

/// `AOM_BLEND_A64(m, v0, v1)` (aom_dsp/blend.h): `round(m*v0 + (64-m)*v1, 6)`.
#[inline]
fn blend_a64(m: i32, v0: u16, v1: u16) -> u16 {
    rpo2(
        m * v0 as i32 + (AOM_BLEND_A64_MAX_ALPHA - m) * v1 as i32,
        AOM_BLEND_A64_ROUND_BITS,
    ) as u16
}

/// `aom_blend_a64_vmask_c` (aom_dsp/blend_a64_vmask.c) — the **vertical** (per-row)
/// mask blend used by OBMC's ABOVE neighbours (`build_obmc_inter_pred_above`,
/// reconinter.c:852). In-place form: `src0 == dst` (the block's own predictor in
/// the recon buffer), matching libaom's call
/// `aom_blend_a64_vmask(dst, dst_stride, dst, dst_stride, tmp, tmp_stride, ...)`.
/// Each output row `i` uses the scalar weight `mask[i]`:
/// `dst = round(mask[i]*dst + (64-mask[i])*src1, 6)`. Planes are `u16` (bd8
/// values `0..=255`); the arithmetic is byte-identical to C's `u8` kernel.
#[allow(clippy::too_many_arguments)]
pub fn blend_a64_vmask(
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    src1: &[u16],
    src1_off: usize,
    src1_stride: usize,
    mask: &[u8],
    w: usize,
    h: usize,
) {
    for i in 0..h {
        let m = mask[i] as i32;
        for j in 0..w {
            let d = dst[dst_off + i * dst_stride + j];
            let s1 = src1[src1_off + i * src1_stride + j];
            dst[dst_off + i * dst_stride + j] = blend_a64(m, d, s1);
        }
    }
}

/// `aom_blend_a64_hmask_c` (aom_dsp/blend_a64_hmask.c) — the **horizontal**
/// (per-column) mask blend used by OBMC's LEFT neighbours
/// (`build_obmc_inter_pred_left`, reconinter.c:891). In-place form (`src0 == dst`).
/// Each output column `j` uses the scalar weight `mask[j]`.
#[allow(clippy::too_many_arguments)]
pub fn blend_a64_hmask(
    dst: &mut [u16],
    dst_off: usize,
    dst_stride: usize,
    src1: &[u16],
    src1_off: usize,
    src1_stride: usize,
    mask: &[u8],
    w: usize,
    h: usize,
) {
    for i in 0..h {
        for j in 0..w {
            let m = mask[j] as i32;
            let d = dst[dst_off + i * dst_stride + j];
            let s1 = src1[src1_off + i * src1_stride + j];
            dst[dst_off + i * dst_stride + j] = blend_a64(m, d, s1);
        }
    }
}
