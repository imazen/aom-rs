//! Loop-restoration ENCODER search — the numeric core of
//! `av1/encoder/pickrst.c` (v3.14.1): Wiener autocorrelation stats, the
//! iterative separable-symmetric Wiener solve, the filter-score gate and the
//! integer tap finalization; the SGR projection least-squares + error and
//! the `ep` search live here too. The per-unit RD walk and the frame-level
//! decision (`av1_pick_filter_restoration`) build on these in this module.
//!
//! All pixel buffers are `u16` planes (the port-wide convention); the lowbd
//! (bd 8) arithmetic matches C's `uint8_t` paths exactly because every value
//! fits in the u8 range and the accumulator widths below are C's.

use crate::sgr::SGR_PARAMS;
use aom_entropy::lr::{WIENER_HALFWIN, WIENER_WIN};

/// `WIENER_WIN2` / `WIENER_HALFWIN1` (restoration.h).
pub const WIENER_WIN2: usize = WIENER_WIN * WIENER_WIN;
const WIENER_HALFWIN1: usize = WIENER_HALFWIN + 1;
/// `WIENER_WIN_REDUCED` (restoration.h): the 5-tap luma window under
/// `lpf_sf.reduce_wiener_window_size`.
pub const WIENER_WIN_REDUCED: usize = WIENER_WIN - 2;
/// `WIENER_STATS_DOWNSAMPLE_FACTOR` (restoration.h).
pub const WIENER_STATS_DOWNSAMPLE_FACTOR: i32 = 4;
/// `WIENER_FILT_STEP` = `1 << WIENER_FILT_PREC_BITS` (restoration.h).
pub const WIENER_FILT_STEP: i64 = 1 << 7;
/// `WIENER_FILT_BITS` = `(4 + 5 + 6) * 2` (restoration.h).
const WIENER_FILT_BITS: i64 = 30;
/// `WIENER_TAP_SCALE_FACTOR` (pickrst.c): working precision of the solve.
const WIENER_TAP_SCALE_FACTOR: i64 = 1 << 16;
/// `NUM_WIENER_ITERS` (pickrst.c).
const NUM_WIENER_ITERS: i32 = 5;

/// `find_average` (pickrst.h): the u8-truncating mean of the lowbd window.
/// `u16` values in u8 range; identical arithmetic.
fn find_average(
    dgd: &[u16],
    dgd_origin: usize,
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    stride: i32,
) -> u16 {
    let mut sum: u64 = 0;
    for i in v_start..v_end {
        for j in h_start..h_end {
            sum += dgd[dgd_origin + (i * stride + j) as usize] as u64;
        }
    }
    (sum / (((v_end - v_start) * (h_end - h_start)) as u64)) as u16
}

/// `acc_stat_one_line` (pickrst.c): one source row's contribution to the
/// int32 row accumulators (`count` = the dgd row this line is centred on).
#[allow(clippy::too_many_arguments)]
fn acc_stat_one_line(
    dgd: &[u16],
    dgd_origin: usize,
    src_row: &[u16],
    dgd_stride: i32,
    h_start: i32,
    h_end: i32,
    avg: u16,
    wiener_halfwin: i32,
    wiener_win2: usize,
    m_row: &mut [i32],
    h_row: &mut [i32],
    count: i32,
) {
    let mut y = [0i16; WIENER_WIN2];
    for j in h_start..h_end {
        let x = src_row[j as usize] as i16 - avg as i16;
        let mut idx = 0usize;
        for k in -wiener_halfwin..=wiener_halfwin {
            for l in -wiener_halfwin..=wiener_halfwin {
                // Window reads may go up to ±3 outside the rect — negative
                // plane coords land in the extended border BEFORE the
                // origin (C pointer semantics).
                let off = dgd_origin as isize + ((count + l) * dgd_stride + (j + k)) as isize;
                y[idx] = dgd[off as usize] as i16 - avg as i16;
                idx += 1;
            }
        }
        debug_assert_eq!(idx, wiener_win2);
        for k in 0..wiener_win2 {
            m_row[k] += y[k] as i32 * x as i32;
            for l in k..wiener_win2 {
                // H is symmetric; fill the upper triangle here (copied down
                // outside the pixel loops).
                h_row[k * wiener_win2 + l] += y[k] as i32 * y[l] as i32;
            }
        }
    }
}

/// `av1_compute_stats_c` (pickrst.c): the lowbd Wiener autocorrelation
/// vector `M[win2]` and matrix `H[win2 * win2]` of the window around each
/// source pixel in the `[h_start, h_end) x [v_start, v_end)` rect, about the
/// dgd mean, optionally with 4x vertical downsampling
/// (`lpf_sf.use_downsampled_wiener_stats`).
#[allow(clippy::too_many_arguments)]
pub fn compute_stats(
    wiener_win: usize,
    dgd: &[u16],
    dgd_origin: usize,
    src: &[u16],
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    dgd_stride: i32,
    src_stride: i32,
    m: &mut [i64],
    h: &mut [i64],
    use_downsampled_wiener_stats: bool,
) {
    let wiener_win2 = wiener_win * wiener_win;
    let wiener_halfwin = (wiener_win >> 1) as i32;
    let avg = find_average(dgd, dgd_origin, h_start, h_end, v_start, v_end, dgd_stride);
    let mut m_row = [0i32; WIENER_WIN2];
    let mut h_row = [0i32; WIENER_WIN2 * WIENER_WIN2];
    let mut downsample_factor = if use_downsampled_wiener_stats {
        WIENER_STATS_DOWNSAMPLE_FACTOR
    } else {
        1
    };

    m[..wiener_win2].fill(0);
    h[..wiener_win2 * wiener_win2].fill(0);

    let mut i = v_start;
    while i < v_end {
        if use_downsampled_wiener_stats && (v_end - i < WIENER_STATS_DOWNSAMPLE_FACTOR) {
            downsample_factor = v_end - i;
        }
        m_row[..wiener_win2].fill(0);
        h_row[..wiener_win2 * wiener_win2].fill(0);
        acc_stat_one_line(
            dgd,
            dgd_origin,
            &src[(i * src_stride) as usize..],
            dgd_stride,
            h_start,
            h_end,
            avg,
            wiener_halfwin,
            wiener_win2,
            &mut m_row,
            &mut h_row,
            i,
        );
        for k in 0..wiener_win2 {
            // Scale by the downsampling factor (1 when not downsampling).
            m[k] += m_row[k] as i64 * downsample_factor as i64;
            for l in k..wiener_win2 {
                h[k * wiener_win2 + l] +=
                    h_row[k * wiener_win2 + l] as i64 * downsample_factor as i64;
            }
        }
        i += downsample_factor;
    }

    for k in 0..wiener_win2 {
        for l in k + 1..wiener_win2 {
            h[l * wiener_win2 + k] = h[k * wiener_win2 + l];
        }
    }
}

/// `find_average_highbd` (pickrst.h).
fn find_average_highbd(
    dgd: &[u16],
    dgd_origin: usize,
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    stride: i32,
) -> u16 {
    // Identical to the lowbd form on u16 planes.
    find_average(dgd, dgd_origin, h_start, h_end, v_start, v_end, stride)
}

/// `av1_compute_stats_highbd_c` (pickrst.c): i64 accumulation with the
/// `bit_depth_divider` normalization (1 / 4 / 16 for bd 8 / 10 / 12).
#[allow(clippy::too_many_arguments)]
pub fn compute_stats_highbd(
    wiener_win: usize,
    dgd: &[u16],
    dgd_origin: usize,
    src: &[u16],
    h_start: i32,
    h_end: i32,
    v_start: i32,
    v_end: i32,
    dgd_stride: i32,
    src_stride: i32,
    m: &mut [i64],
    h: &mut [i64],
    bit_depth: i32,
) {
    let wiener_win2 = wiener_win * wiener_win;
    let wiener_halfwin = (wiener_win >> 1) as i32;
    let avg = find_average_highbd(dgd, dgd_origin, h_start, h_end, v_start, v_end, dgd_stride);
    let bit_depth_divider: i64 = match bit_depth {
        12 => 16,
        10 => 4,
        _ => 1,
    };

    m[..wiener_win2].fill(0);
    h[..wiener_win2 * wiener_win2].fill(0);
    let mut y = [0i32; WIENER_WIN2];
    for i in v_start..v_end {
        for j in h_start..h_end {
            let x = src[(i * src_stride + j) as usize] as i32 - avg as i32;
            let mut idx = 0usize;
            for k in -wiener_halfwin..=wiener_halfwin {
                for l in -wiener_halfwin..=wiener_halfwin {
                    let off = dgd_origin as isize + ((i + l) * dgd_stride + (j + k)) as isize;
                    y[idx] = dgd[off as usize] as i32 - avg as i32;
                    idx += 1;
                }
            }
            debug_assert_eq!(idx, wiener_win2);
            for k in 0..wiener_win2 {
                m[k] += y[k] as i64 * x as i64;
                for l in k..wiener_win2 {
                    h[k * wiener_win2 + l] += y[k] as i64 * y[l] as i64;
                }
            }
        }
    }
    for k in 0..wiener_win2 {
        m[k] /= bit_depth_divider;
        h[k * wiener_win2 + k] /= bit_depth_divider;
        for l in k + 1..wiener_win2 {
            h[k * wiener_win2 + l] /= bit_depth_divider;
            h[l * wiener_win2 + k] = h[k * wiener_win2 + l];
        }
    }
}

/// `wrap_index` (pickrst.c).
#[inline]
fn wrap_index(i: usize, wiener_win: usize) -> usize {
    let wiener_halfwin1 = (wiener_win >> 1) + 1;
    if i >= wiener_halfwin1 {
        wiener_win - 1 - i
    } else {
        i
    }
}

/// `split_wiener_filter_coefficients` (pickrst.c): `w = w1 * SCALE + w2`.
fn split_wiener_filter_coefficients(wiener_win: usize, w: &[i32], w1: &mut [i32], w2: &mut [i32]) {
    for i in 0..wiener_win {
        w1[i] = w[i] / WIENER_TAP_SCALE_FACTOR as i32;
        w2[i] = w[i] - w1[i] * WIENER_TAP_SCALE_FACTOR as i32;
        debug_assert_eq!(w[i] as i64, w1[i] as i64 * WIENER_TAP_SCALE_FACTOR + w2[i] as i64);
    }
}

/// `multiply_and_scale` (pickrst.c): `x * w / SCALE` where
/// `w = w1 * SCALE + w2`, without overflowing the direct product.
#[inline]
fn multiply_and_scale(x: i64, w1: i32, w2: i32) -> i64 {
    x * w1 as i64 + x * w2 as i64 / WIENER_TAP_SCALE_FACTOR
}

/// `linsolve_wiener` (pickrst.c): Gaussian elimination with partial pivoting
/// and the b/278065963 overflow-reworked scaling; taps out in
/// `WIENER_TAP_SCALE_FACTOR` fixed point. Returns false when singular.
fn linsolve_wiener(n: usize, a: &mut [i64], stride: usize, b: &mut [i64], x: &mut [i64]) -> bool {
    for k in 0..n.saturating_sub(1) {
        // Partial pivoting: bring the row with the largest pivot to the top.
        for i in (k + 1..n).rev() {
            if a[(i - 1) * stride + k].abs() < a[i * stride + k].abs() {
                for j in 0..n {
                    a.swap(i * stride + j, (i - 1) * stride + j);
                }
                b.swap(i, i - 1);
            }
        }

        let mut max_abs_akj: i64 = 0;
        for j in 0..n {
            let abs_akj = a[k * stride + j].abs();
            if abs_akj > max_abs_akj {
                max_abs_akj = abs_akj;
            }
        }
        let scale_threshold: i64 = 1 << 22;
        let scaler_a: i64 = if max_abs_akj < scale_threshold { 1 } else { 1 << 6 };
        let scaler_c: i64 = if max_abs_akj < scale_threshold { 1 } else { 1 << 7 };
        let scaler = scaler_c * scaler_a;

        // Forward elimination (row-echelon form).
        for i in k..n - 1 {
            if a[k * stride + k] == 0 {
                return false;
            }
            let c = a[(i + 1) * stride + k] / scaler_c;
            let cd = a[k * stride + k];
            for j in 0..n {
                a[(i + 1) * stride + j] -= a[k * stride + j] / scaler_a * c / cd * scaler;
            }
            b[i + 1] -= c * b[k] / cd * scaler_c;
        }
    }
    // Back-substitution.
    for i in (0..n).rev() {
        if a[i * stride + i] == 0 {
            return false;
        }
        let mut c: i64 = 0;
        for j in i + 1..n {
            c += a[i * stride + j] * x[j] / WIENER_TAP_SCALE_FACTOR;
        }
        x[i] = WIENER_TAP_SCALE_FACTOR * (b[i] - c) / a[i * stride + i];
    }
    true
}

/// `update_a_sep_sym` / `update_b_sep_sym` (pickrst.c): fix one direction's
/// taps, re-solve the other. `dir == 0` updates `a` (vertical) from fixed
/// `b`; `dir == 1` updates `b` from fixed `a`. `m`/`h` are the win2 /
/// win2*win2 stats.
fn update_sep_sym(dir: usize, wiener_win: usize, m: &[i64], h: &[i64], a: &mut [i32], b: &mut [i32]) {
    let wiener_win2 = wiener_win * wiener_win;
    let wiener_halfwin1 = (wiener_win >> 1) + 1;
    let mut s = [0i64; WIENER_WIN];
    let mut aa = [0i64; WIENER_HALFWIN1];
    let mut bb = [0i64; WIENER_HALFWIN1 * WIENER_HALFWIN1];
    let mut f1 = [0i32; WIENER_WIN];
    let mut f2 = [0i32; WIENER_WIN];

    // Mc[i] = M + i*win (row i); Hc[i*win + j] = H + i*win*win2 + j*win.
    let mc = |i: usize, j: usize| m[i * wiener_win + j];
    let hc = |i: usize, j: usize, k: usize| h[i * wiener_win * wiener_win2 + j * wiener_win + k];

    if dir == 0 {
        // update_a_sep_sym: A[jj] += Mc[i][j] * b[i] / SCALE
        for i in 0..wiener_win {
            for j in 0..wiener_win {
                let jj = wrap_index(j, wiener_win);
                aa[jj] += mc(i, j) * b[i] as i64 / WIENER_TAP_SCALE_FACTOR;
            }
        }
        split_wiener_filter_coefficients(wiener_win, b, &mut f1, &mut f2);
        for i in 0..wiener_win {
            for j in 0..wiener_win {
                for k in 0..wiener_win {
                    let kk = wrap_index(k, wiener_win);
                    for l in 0..wiener_win {
                        let ll = wrap_index(l, wiener_win);
                        // Hc[j * win + i][k * win2 + l] * b[i] / SCALE, then
                        // * b[j] / SCALE via multiply_and_scale.
                        let x = hc(j, i, k * wiener_win2 + l) * b[i] as i64
                            / WIENER_TAP_SCALE_FACTOR;
                        bb[ll * wiener_halfwin1 + kk] += multiply_and_scale(x, f1[j], f2[j]);
                    }
                }
            }
        }
    } else {
        // update_b_sep_sym: A[ii] += Mc[i][j] * a[j] / SCALE
        for i in 0..wiener_win {
            let ii = wrap_index(i, wiener_win);
            for j in 0..wiener_win {
                aa[ii] += mc(i, j) * a[j] as i64 / WIENER_TAP_SCALE_FACTOR;
            }
        }
        split_wiener_filter_coefficients(wiener_win, a, &mut f1, &mut f2);
        for i in 0..wiener_win {
            let ii = wrap_index(i, wiener_win);
            for j in 0..wiener_win {
                let jj = wrap_index(j, wiener_win);
                for k in 0..wiener_win {
                    for l in 0..wiener_win {
                        let x = hc(i, j, k * wiener_win2 + l) * a[k] as i64
                            / WIENER_TAP_SCALE_FACTOR;
                        bb[jj * wiener_halfwin1 + ii] += multiply_and_scale(x, f1[l], f2[l]);
                    }
                }
            }
        }
    }

    // Normalization enforcement in the system of equations itself.
    for i in 0..wiener_halfwin1 - 1 {
        aa[i] -= aa[wiener_halfwin1 - 1] * 2 + bb[i * wiener_halfwin1 + wiener_halfwin1 - 1]
            - 2 * bb[(wiener_halfwin1 - 1) * wiener_halfwin1 + (wiener_halfwin1 - 1)];
    }
    for i in 0..wiener_halfwin1 - 1 {
        for j in 0..wiener_halfwin1 - 1 {
            bb[i * wiener_halfwin1 + j] -= 2
                * (bb[i * wiener_halfwin1 + (wiener_halfwin1 - 1)]
                    + bb[(wiener_halfwin1 - 1) * wiener_halfwin1 + j]
                    - 2 * bb[(wiener_halfwin1 - 1) * wiener_halfwin1 + (wiener_halfwin1 - 1)]);
        }
    }
    if linsolve_wiener(wiener_halfwin1 - 1, &mut bb, wiener_halfwin1, &mut aa, &mut s) {
        s[wiener_halfwin1 - 1] = WIENER_TAP_SCALE_FACTOR;
        for i in wiener_halfwin1..wiener_win {
            s[i] = s[wiener_win - 1 - i];
            s[wiener_halfwin1 - 1] -= 2 * s[i];
        }
        let out = if dir == 0 { a } else { b };
        for i in 0..wiener_win {
            out[i] = s[i].clamp(-(1 << (WIENER_FILT_BITS - 1)), (1 << (WIENER_FILT_BITS - 1)) - 1)
                as i32;
        }
    }
}

/// `wiener_decompose_sep_sym` (pickrst.c): 4 alternating solve iterations
/// from the identity-ish init filter; outputs the two directions' taps in
/// `WIENER_TAP_SCALE_FACTOR` fixed point.
pub fn wiener_decompose_sep_sym(
    wiener_win: usize,
    m: &[i64],
    h: &[i64],
    a: &mut [i32; WIENER_WIN],
    b: &mut [i32; WIENER_WIN],
) {
    // init_filt = WIENER_FILT_TAP{0,1,2,3}_MIDV mirror.
    const INIT_FILT: [i32; WIENER_WIN] = [3, -7, 15, 106, 15, -7, 3];
    let plane_off = (WIENER_WIN - wiener_win) >> 1;
    for i in 0..wiener_win {
        let v = (WIENER_TAP_SCALE_FACTOR / WIENER_FILT_STEP) as i32 * INIT_FILT[i + plane_off];
        a[i] = v;
        b[i] = v;
    }
    let mut iter = 1;
    while iter < NUM_WIENER_ITERS {
        update_sep_sym(0, wiener_win, m, h, a, b);
        update_sep_sym(1, wiener_win, m, h, a, b);
        iter += 1;
    }
}

/// `compute_score` (pickrst.c): `x'Hx - 2x'M` of the finalized integer
/// filter minus the identity filter's score; positive means the learned
/// filter is WORSE than identity (revert to RESTORE_NONE).
pub fn compute_score(
    wiener_win: usize,
    m: &[i64],
    h: &[i64],
    vfilt: &[i16; 8],
    hfilt: &[i16; 8],
) -> i64 {
    let mut ab = [0i32; WIENER_WIN * WIENER_WIN];
    let mut a = [0i16; WIENER_WIN];
    let mut b = [0i16; WIENER_WIN];
    let plane_off = (WIENER_WIN - wiener_win) >> 1;
    let wiener_win2 = wiener_win * wiener_win;

    a[WIENER_HALFWIN] = WIENER_FILT_STEP as i16;
    b[WIENER_HALFWIN] = WIENER_FILT_STEP as i16;
    for i in 0..WIENER_HALFWIN {
        a[i] = vfilt[i];
        a[WIENER_WIN - i - 1] = vfilt[i];
        b[i] = hfilt[i];
        b[WIENER_WIN - i - 1] = hfilt[i];
        a[WIENER_HALFWIN] -= 2 * a[i];
        b[WIENER_HALFWIN] -= 2 * b[i];
    }
    for k in 0..wiener_win {
        for l in 0..wiener_win {
            ab[k * wiener_win + l] = a[l + plane_off] as i32 * b[k + plane_off] as i32;
        }
    }
    let mut p: i64 = 0;
    let mut q: i64 = 0;
    for k in 0..wiener_win2 {
        p += ab[k] as i64 * m[k] / WIENER_FILT_STEP / WIENER_FILT_STEP;
        for l in 0..wiener_win2 {
            q += ab[k] as i64 * h[k * wiener_win2 + l] * ab[l] as i64
                / WIENER_FILT_STEP
                / WIENER_FILT_STEP
                / WIENER_FILT_STEP
                / WIENER_FILT_STEP;
        }
    }
    let score = q - 2 * p;

    let ip = m[wiener_win2 >> 1];
    let iq = h[(wiener_win2 >> 1) * wiener_win2 + (wiener_win2 >> 1)];
    let iscore = iq - 2 * ip;

    score - iscore
}

/// `finalize_sym_filter` (pickrst.c): fixed-point taps to the coded integer
/// taps with rounding, per-tap clips, symmetric mirror and the implicit
/// centre; the 5-tap window shifts its taps into slots 1/2.
pub fn finalize_sym_filter(wiener_win: usize, f: &[i32; WIENER_WIN], fi: &mut [i16; 8]) {
    const TAP_MINV: [i16; 3] = [-5, -23, -17];
    const TAP_MAXV: [i16; 3] = [10, 8, 46];
    let wiener_halfwin = wiener_win >> 1;
    *fi = [0; 8];
    for i in 0..wiener_halfwin {
        let dividend = f[i] as i64 * WIENER_FILT_STEP;
        let divisor = WIENER_TAP_SCALE_FACTOR;
        fi[i] = if dividend < 0 {
            ((dividend - divisor / 2) / divisor) as i16
        } else {
            ((dividend + divisor / 2) / divisor) as i16
        };
    }
    if wiener_win == WIENER_WIN {
        fi[0] = fi[0].clamp(TAP_MINV[0], TAP_MAXV[0]);
        fi[1] = fi[1].clamp(TAP_MINV[1], TAP_MAXV[1]);
        fi[2] = fi[2].clamp(TAP_MINV[2], TAP_MAXV[2]);
    } else {
        fi[2] = fi[1].clamp(TAP_MINV[2], TAP_MAXV[2]);
        fi[1] = fi[0].clamp(TAP_MINV[1], TAP_MAXV[1]);
        fi[0] = 0;
    }
    // Satisfy filter constraints.
    fi[WIENER_WIN - 1] = fi[0];
    fi[WIENER_WIN - 2] = fi[1];
    fi[WIENER_WIN - 3] = fi[2];
    // The central element has an implicit +WIENER_FILT_STEP.
    fi[3] = -2 * (fi[0] + fi[1] + fi[2]);
}

// ---------------------------------------------------------------------------
// SGR search numeric core.
// ---------------------------------------------------------------------------

/// `SGRPROJ_RST_BITS` / `SGRPROJ_PRJ_BITS` (restoration.h).
const SGRPROJ_RST_BITS: i32 = 4;
const SGRPROJ_PRJ_BITS: i32 = 7;

/// `av1_lowbd_pixel_proj_error_c` + `av1_highbd_pixel_proj_error_c`
/// (pickrst.c) on u16 planes: the SSE of the xq-projected SGR restoration
/// against the source. The lowbd and highbd forms round differently
/// (ROUND_POWER_OF_TWO vs add-half-then-shift with `+d - s` recomposition) —
/// both are ported exactly.
#[allow(clippy::too_many_arguments)]
pub fn pixel_proj_error(
    src: &[u16],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[u16],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    xq: [i32; 2],
    ep: usize,
    highbd: bool,
) -> i64 {
    let (rads, _) = SGR_PARAMS[ep];
    let r0 = rads[0] > 0;
    let r1 = rads[1] > 0;
    let mut err: i64 = 0;
    if !highbd {
        for i in 0..height {
            for j in 0..width {
                let d = dat[dat_off + i * dat_stride + j] as i32;
                let s = src[src_off + i * src_stride + j] as i32;
                let u = d << SGRPROJ_RST_BITS;
                let mut v = u << SGRPROJ_PRJ_BITS;
                if r0 {
                    v += xq[0] * (flt0[i * flt0_stride + j] - u);
                }
                if r1 {
                    v += xq[1] * (flt1[i * flt1_stride + j] - u);
                }
                let e = if r0 || r1 {
                    // ROUND_POWER_OF_TWO(v, 11) - src
                    ((v + (1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)))
                        >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS))
                        - s
                } else {
                    d - s
                };
                err += e as i64 * e as i64;
            }
        }
    } else {
        let half: i32 = 1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1);
        for i in 0..height {
            for j in 0..width {
                let d = dat[dat_off + i * dat_stride + j] as i32;
                let s = src[src_off + i * src_stride + j] as i32;
                if r0 || r1 {
                    let u = d << SGRPROJ_RST_BITS;
                    let mut v = half;
                    if r0 {
                        v += xq[0] * (flt0[i * flt0_stride + j] - u);
                    }
                    if r1 {
                        v += xq[1] * (flt1[i * flt1_stride + j] - u);
                    }
                    let e = (v >> (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS)) + d - s;
                    err += e as i64 * e as i64;
                } else {
                    let e = d - s;
                    err += e as i64 * e as i64;
                }
            }
        }
    }
    err
}

/// `av1_calc_proj_params_c` + `_high_bd_c` (pickrst.c): the least-squares
/// normal-equation accumulators `H` (2x2) and `C` (2), divided by the pixel
/// count. Identical arithmetic for lowbd/highbd on u16 planes (the C pair
/// differs only in pointer types).
#[allow(clippy::too_many_arguments)]
pub fn calc_proj_params(
    src: &[u16],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[u16],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    ep: usize,
) -> ([[i64; 2]; 2], [i64; 2]) {
    let (rads, _) = SGR_PARAMS[ep];
    let size = (width * height) as i64;
    let mut hh = [[0i64; 2]; 2];
    let mut cc = [0i64; 2];
    let (r0, r1) = (rads[0] > 0, rads[1] > 0);
    for i in 0..height {
        for j in 0..width {
            let u = (dat[dat_off + i * dat_stride + j] as i32) << SGRPROJ_RST_BITS;
            let s = ((src[src_off + i * src_stride + j] as i32) << SGRPROJ_RST_BITS) - u;
            if r0 && r1 {
                let f1 = flt0[i * flt0_stride + j] - u;
                let f2 = flt1[i * flt1_stride + j] - u;
                hh[0][0] += f1 as i64 * f1 as i64;
                hh[1][1] += f2 as i64 * f2 as i64;
                hh[0][1] += f1 as i64 * f2 as i64;
                cc[0] += f1 as i64 * s as i64;
                cc[1] += f2 as i64 * s as i64;
            } else if r0 {
                let f1 = flt0[i * flt0_stride + j] - u;
                hh[0][0] += f1 as i64 * f1 as i64;
                cc[0] += f1 as i64 * s as i64;
            } else if r1 {
                let f2 = flt1[i * flt1_stride + j] - u;
                hh[1][1] += f2 as i64 * f2 as i64;
                cc[1] += f2 as i64 * s as i64;
            }
        }
    }
    if r0 && r1 {
        hh[0][0] /= size;
        hh[0][1] /= size;
        hh[1][1] /= size;
        hh[1][0] = hh[0][1];
        cc[0] /= size;
        cc[1] /= size;
    } else if r0 {
        hh[0][0] /= size;
        cc[0] /= size;
    } else if r1 {
        hh[1][1] /= size;
        cc[1] /= size;
    }
    (hh, cc)
}

/// `signed_rounded_divide` (pickrst.c).
#[inline]
fn signed_rounded_divide(dividend: i64, divisor: i64) -> i64 {
    if dividend < 0 {
        (dividend - divisor / 2) / divisor
    } else {
        (dividend + divisor / 2) / divisor
    }
}

/// `get_proj_subspace` (pickrst.c): solve the 2x2 (or scalar) normal
/// equations for the projection weights `xq`, with the C overflow guards.
#[allow(clippy::too_many_arguments)]
pub fn get_proj_subspace(
    src: &[u16],
    src_off: usize,
    width: usize,
    height: usize,
    src_stride: usize,
    dat: &[u16],
    dat_off: usize,
    dat_stride: usize,
    flt0: &[i32],
    flt0_stride: usize,
    flt1: &[i32],
    flt1_stride: usize,
    ep: usize,
) -> [i32; 2] {
    let (rads, _) = SGR_PARAMS[ep];
    let mut xq = [0i32; 2];
    let (hh, cc) = calc_proj_params(
        src, src_off, width, height, src_stride, dat, dat_off, dat_stride, flt0, flt0_stride,
        flt1, flt1_stride, ep,
    );
    let h = [hh[0][0], hh[0][1], hh[1][0], hh[1][1]];
    let c = cc;
    if rads[0] == 0 {
        let det = h[3];
        if det == 0 {
            return xq;
        }
        xq[0] = 0;
        xq[1] = signed_rounded_divide(c[1] * (1 << SGRPROJ_PRJ_BITS), det) as i32;
    } else if rads[1] == 0 {
        let det = h[0];
        if det == 0 {
            return xq;
        }
        xq[0] = signed_rounded_divide(c[0] * (1 << SGRPROJ_PRJ_BITS), det) as i32;
        xq[1] = 0;
    } else {
        let det = h[0] * h[3] - h[1] * h[2];
        if det == 0 {
            return xq;
        }
        let shift: i64 = 1 << SGRPROJ_PRJ_BITS;
        let div1 = h[3] * c[0] - h[1] * c[1];
        xq[0] = if (div1 > 0 && i64::MAX / shift < div1) || (div1 < 0 && i64::MIN / shift > div1) {
            signed_rounded_divide(div1, det / shift) as i32
        } else {
            signed_rounded_divide(div1 * shift, det) as i32
        };
        let div2 = h[0] * c[1] - h[2] * c[0];
        xq[1] = if (div2 > 0 && i64::MAX / shift < div2) || (div2 < 0 && i64::MIN / shift > div2) {
            signed_rounded_divide(div2, det / shift) as i32
        } else {
            signed_rounded_divide(div2 * shift, det) as i32
        };
    }
    xq
}

/// `encode_xq` (pickrst.c): projection weights to the coded `xqd` domain
/// with the per-radius clamps.
pub fn encode_xq(xq: [i32; 2], ep: usize) -> [i32; 2] {
    use aom_entropy::lr::{SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1, SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1};
    let (rads, _) = SGR_PARAMS[ep];
    let mut xqd = [0i32; 2];
    if rads[0] == 0 {
        xqd[0] = 0;
        xqd[1] = ((1 << SGRPROJ_PRJ_BITS) - xq[1]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
    } else if rads[1] == 0 {
        xqd[0] = xq[0].clamp(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
        xqd[1] = ((1 << SGRPROJ_PRJ_BITS) - xqd[0]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
    } else {
        xqd[0] = xq[0].clamp(SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MAX0);
        xqd[1] = ((1 << SGRPROJ_PRJ_BITS) - xqd[0] - xq[1]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
    }
    xqd
}

// ---------------------------------------------------------------------------
// The per-unit RD search + frame-level decision
// (`restoration_search` / `av1_pick_filter_restoration`, pickrst.c).
// ---------------------------------------------------------------------------

use crate::frame::{
    at, extend_frame, filter_unit, save_boundary_lines, StripeBoundaries, MARGIN_H, MARGIN_V,
};
use crate::sgr::{decode_xq, selfguided_restoration};
use aom_entropy::lr::{
    count_sgrproj_bits, count_wiener_bits, lr_corners_in_sb, LrFrameConfig,
    LrUnitInfo, SgrprojInfoLr, WienerInfoLr, RESTORATION_PROC_UNIT_SIZE, RESTORATION_UNITSIZE_MAX,
    RESTORATION_UNIT_OFFSET, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
    SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1, SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1, WIENER_WIN_CHROMA,
};

/// `RESTORE_TYPES` / `RESTORE_SWITCHABLE_TYPES` (enums.h).
const RESTORE_TYPES: usize = 4;
const RESTORE_SWITCHABLE_TYPES: usize = 3;
/// `AV1_PROB_COST_SHIFT` (av1/encoder/cost.h).
const AV1_PROB_COST_SHIFT: i64 = 9;
/// `NUM_WIENER_ITERS` neighbours: search penalties (pickrst.c).
const DUAL_SGR_PENALTY_MULT: f64 = 0.01;
const WIENER_SGR_PENALTY_MULT: f64 = 0.005;
/// `RESTORATION_UNITPELS_MAX` (restoration.h): flt scratch sizing.
const RESTORATION_UNITPELS_MAX: usize =
    (RESTORATION_UNITSIZE_MAX as usize * 3 / 2 + 2 * 3 + 16)
        * (RESTORATION_UNITSIZE_MAX as usize * 3 / 2 + 2 * 3 + 8);

/// `sgproj_ep_grp1_seed` / `sgproj_ep_grp2_3` (pickrst.c): the pruned-ep
/// search ladder.
const SGRPROJ_EP_GRP1_START_IDX: i32 = 0;
const SGRPROJ_EP_GRP1_END_IDX: i32 = 9;
const SGRPROJ_EP_GRP1_SEED: [i32; 4] = [0, 3, 6, 9];
const SGRPROJ_EP_GRP2_3: [[i32; 14]; 2] = [
    [10, 10, 11, 11, 12, 12, 13, 13, 13, 13, -1, -1, -1, -1],
    [14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15],
];

/// `RDCOST_DBL_WITH_NATIVE_BD_DIST` (av1/encoder/rd.h).
#[inline]
fn rdcost_dbl_with_native_bd_dist(rdmult: i64, rate: i64, dist: i64, bd: i32) -> f64 {
    (rate as f64 * rdmult as f64) / ((1i64 << AV1_PROB_COST_SHIFT) as f64)
        + ((dist >> (2 * (bd - 8))) as f64) * ((1 << 7) as f64)
}

/// The `lpf_sf` slice `av1_pick_filter_restoration` consumes
/// (speed_features.h `LOOP_FILTER_SPEED_FEATURES`), plus the two frame
/// inputs the pruning heuristics need.
#[derive(Clone, Copy, Debug)]
pub struct LrSearchSf {
    pub disable_wiener_filter: bool,
    pub disable_sgr_filter: bool,
    pub disable_loop_restoration_luma: bool,
    pub disable_loop_restoration_chroma: bool,
    pub disable_wiener_coeff_refine_search: bool,
    /// 0 off; 1/2 = the `scale[]` ladder of the src-var prune.
    pub prune_wiener_based_on_src_var: i32,
    /// 0 off; 1 = rdcost-ratio gate; 2 = best-rtype gate.
    pub prune_sgr_based_on_wiener: i32,
    /// 0 full 16-ep; 1 = seeds+neighbours+groups; >=2 = seeds only.
    pub enable_sgr_ep_pruning: i32,
    pub reduce_wiener_window_size: bool,
    pub use_downsampled_wiener_stats: bool,
    pub dual_sgr_penalty_level: i32,
    pub switchable_lr_with_bias_level: i32,
    /// Luma-scale unit-size search bounds (`min/max_lr_unit_size`).
    pub min_lr_unit_size: i32,
    pub max_lr_unit_size: i32,
}

impl Default for LrSearchSf {
    /// Speed-0 defaults (speed_features.c framesize-independent tail +
    /// qindex-dependent size-search init).
    fn default() -> Self {
        LrSearchSf {
            disable_wiener_filter: false,
            disable_sgr_filter: false,
            disable_loop_restoration_luma: false,
            disable_loop_restoration_chroma: false,
            disable_wiener_coeff_refine_search: false,
            prune_wiener_based_on_src_var: 0,
            prune_sgr_based_on_wiener: 0,
            enable_sgr_ep_pruning: 0,
            reduce_wiener_window_size: false,
            use_downsampled_wiener_stats: false,
            dual_sgr_penalty_level: 0,
            switchable_lr_with_bias_level: 0,
            min_lr_unit_size: RESTORATION_PROC_UNIT_SIZE,
            max_lr_unit_size: RESTORATION_UNITSIZE_MAX,
        }
    }
}

/// One plane's pixels for the search: the ORIGINAL source, the deblocked
/// (pre-CDEF) recon and the current (post-CDEF) recon — `deblocked` and
/// `cur` may be the same content when CDEF did not run, matching the C
/// encoder's two `save_boundary_lines` passes.
pub struct LrPlanePixels<'a> {
    pub src: &'a [u16],
    pub deblocked: &'a [u16],
    pub cur: &'a [u16],
    pub stride: usize,
}

/// Frame-level inputs of `av1_pick_filter_restoration`.
pub struct LrSearchInput<'a> {
    pub planes: Vec<LrPlanePixels<'a>>,
    /// Luma crop dims (the RU grid domain; superres not in this envelope).
    pub crop_width: i32,
    pub crop_height: i32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bit_depth: i32,
    /// `seq_params->use_highbitdepth` routing: false = the lowbd (u8) C
    /// arithmetic (bd 8), true = the highbd paths.
    pub highbd: bool,
    /// `cpi->rd.RDMULT`.
    pub rdmult: i64,
    /// `av1_dc_quant_QTX(base_qindex, 0, bit_depth)` — only read when
    /// `prune_wiener_based_on_src_var > 0`.
    pub dc_quant_qtx: i32,
    /// Superblock geometry: `mib_size_log2` (4=sb64, 5=sb128) and the mi
    /// grid extent.
    pub mib_size_log2: i32,
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// Tile bounds in superblock units (`tiles.row_start_sb` pairs), raster
    /// iterated rows-outer. Single tile: `[(0, sb_rows)]` / `[(0, sb_cols)]`.
    pub tile_sb_rows: Vec<(i32, i32)>,
    pub tile_sb_cols: Vec<(i32, i32)>,
    /// `av1_fill_lr_rates` outputs (cost_tokens_from_cdf of the frame-init
    /// wiener/sgrproj/switchable restore CDFs).
    pub wiener_restore_cost: [i32; 2],
    pub sgrproj_restore_cost: [i32; 2],
    pub switchable_restore_cost: [i32; 3],
    pub sf: LrSearchSf,
}

/// `av1_pick_filter_restoration`'s decision.
#[derive(Clone, Debug, Default)]
pub struct LrSearchOutcome {
    /// The chosen luma restoration unit size (all planes share it, `s = 0`).
    pub unit_size: i32,
    pub frame_restoration_type: [u8; 3],
    /// Per-plane unit params in unit-grid raster order for the chosen size
    /// (empty when that plane is `RESTORE_NONE`).
    pub units: [Vec<LrUnitInfo>; 3],
}

/// `RestUnitSearchInfo` (pickrst.h) — C zero-initializes (memset), so the
/// wiener/sgrproj members here are ZEROS, not the syntax defaults.
#[derive(Clone, Copy)]
struct RestUnitSearchInfo {
    best_rtype: [u8; 3],
    wiener: WienerInfoLr,
    sgrproj: SgrprojInfoLr,
}

impl Default for RestUnitSearchInfo {
    fn default() -> Self {
        RestUnitSearchInfo {
            best_rtype: [RESTORE_NONE; 3],
            wiener: WienerInfoLr {
                vfilter: [0; 8],
                hfilter: [0; 8],
            },
            sgrproj: SgrprojInfoLr { ep: 0, xqd: [0, 0] },
        }
    }
}

/// One plane's staged buffers: the extended dgd (recon) in the padded
/// frame-walk layout, the trial dst, the stripe boundaries, and the source.
struct PlaneCtx<'a> {
    plane: usize,
    pw: i32,
    ph: i32,
    sx: usize,
    sy: usize,
    w_stride: usize,
    dgd_pad: Vec<u16>,
    dst_pad: Vec<u16>,
    bnd: StripeBoundaries,
    src: &'a [u16],
    src_stride: usize,
    flt0: Vec<i32>,
    flt1: Vec<i32>,
}

impl<'a> PlaneCtx<'a> {
    /// Stage one plane: pad + `av1_extend_frame` the current recon, build
    /// the boundary context exactly as the encoder's two
    /// `av1_loop_restoration_save_boundary_lines` passes do.
    fn new(input: &LrSearchInput<'a>, plane: usize) -> PlaneCtx<'a> {
        let p = &input.planes[plane];
        let (sx, sy) = if plane > 0 {
            (input.ss_x, input.ss_y)
        } else {
            (0, 0)
        };
        let pw = (input.crop_width + (1 << sx) - 1) >> sx;
        let ph = (input.crop_height + (1 << sy) - 1) >> sy;
        let (pwu, phu) = (pw as usize, ph as usize);

        // Boundary buffers (av1_alloc_restoration_buffers geometry — stripes
        // counted on the LUMA extent).
        let mi_h = ((input.crop_height + 7) & !7) as usize;
        let ext_h = RESTORATION_UNIT_OFFSET as usize + mi_h;
        let num_stripes = ext_h.div_ceil(64);
        let b_stride = (pwu + 2 * 4 + 31) & !31;
        let mut bnd = StripeBoundaries {
            above: vec![0; num_stripes * 2 * b_stride],
            below: vec![0; num_stripes * 2 * b_stride],
            stride: b_stride,
        };
        // Encoder ordering (cdef_restoration_frame): pass 0 (internal stripe
        // context) on the DEBLOCKED frame BEFORE CDEF; pass 1 (frame edges)
        // on the CURRENT frame after CDEF.
        save_boundary_lines(&mut bnd, p.deblocked, p.stride, pwu, phu, sy, false);
        save_boundary_lines(&mut bnd, p.cur, p.stride, pwu, phu, sy, true);

        // Padded dgd + trial dst (frame-walk layout).
        let w_stride = pwu + 2 * MARGIN_H;
        let mut dgd_pad = vec![0u16; w_stride * (phu + 2 * MARGIN_V)];
        for r in 0..phu {
            dgd_pad[at(w_stride, r as isize, 0)..at(w_stride, r as isize, pw as isize)]
                .copy_from_slice(&p.cur[r * p.stride..][..pwu]);
        }
        extend_frame(&mut dgd_pad, pwu, phu, w_stride);
        let dst_pad = vec![0u16; w_stride * (phu + 2 * MARGIN_V)];

        PlaneCtx {
            plane,
            pw,
            ph,
            sx,
            sy,
            w_stride,
            dgd_pad,
            dst_pad,
            bnd,
            src: p.src,
            src_stride: p.stride,
            flt0: vec![0i32; RESTORATION_UNITPELS_MAX],
            flt1: vec![0i32; RESTORATION_UNITPELS_MAX],
        }
    }

    /// Padded-buffer element offset of plane coord `(row, col)`.
    #[inline]
    fn pad_off(&self, row: i32, col: i32) -> usize {
        at(self.w_stride, row as isize, col as isize)
    }

    /// `sse_restoration_unit` (pickrst.c): SSE of source vs the trial dst
    /// over the unit rect.
    fn sse_dst(&self, limits: (i32, i32, i32, i32)) -> i64 {
        let (v0, v1, h0, h1) = limits;
        let mut sse: i64 = 0;
        for r in v0..v1 {
            let s = &self.src[r as usize * self.src_stride..];
            let d = &self.dst_pad[self.pad_off(r, 0)..];
            for c in h0..h1 {
                let e = s[c as usize] as i64 - d[c as usize] as i64;
                sse += e * e;
            }
        }
        sse
    }

    /// SSE of source vs the CURRENT recon (RESTORE_NONE) over the rect.
    fn sse_none(&self, limits: (i32, i32, i32, i32)) -> i64 {
        let (v0, v1, h0, h1) = limits;
        let mut sse: i64 = 0;
        for r in v0..v1 {
            let s = &self.src[r as usize * self.src_stride..];
            let d = &self.dgd_pad[self.pad_off(r, 0)..];
            for c in h0..h1 {
                let e = s[c as usize] as i64 - d[c as usize] as i64;
                sse += e * e;
            }
        }
        sse
    }

    /// `var_restoration_unit` (`aom_var_2d_u8/u16` / (w*h)): source variance
    /// over the rect.
    fn src_var(&self, limits: (i32, i32, i32, i32)) -> u64 {
        let (v0, v1, h0, h1) = limits;
        let (w, h) = ((h1 - h0) as u64, (v1 - v0) as u64);
        let mut ss: u64 = 0;
        let mut s: u64 = 0;
        for r in v0..v1 {
            let row = &self.src[r as usize * self.src_stride..];
            for c in h0..h1 {
                let v = row[c as usize] as u64;
                ss += v * v;
                s += v;
            }
        }
        (ss - s * s / (w * h)) / (w * h)
    }

    /// `try_restoration_unit` (pickrst.c): run the REAL per-unit filter
    /// (stripe boundaries, optimized_lr = 0 like the encoder) into the trial
    /// dst, return the unit SSE vs source.
    fn try_restoration_unit(
        &mut self,
        limits: (i32, i32, i32, i32),
        rui: &LrUnitInfo,
        bit_depth: i32,
    ) -> i64 {
        filter_unit(
            &mut self.dgd_pad,
            &mut self.dst_pad,
            self.w_stride,
            rui,
            &self.bnd,
            self.ph as usize,
            self.sx,
            self.sy,
            bit_depth,
            limits,
            false,
        );
        self.sse_dst(limits)
    }
}

/// `RestSearchCtxt`'s per-plane mutable search state.
struct RscState {
    sse: [i64; RESTORE_SWITCHABLE_TYPES],
    total_sse: [i64; RESTORE_TYPES],
    total_bits: [i64; RESTORE_TYPES],
    ref_wiener: WienerInfoLr,
    ref_sgrproj: SgrprojInfoLr,
    switchable_ref_wiener: WienerInfoLr,
    switchable_ref_sgrproj: SgrprojInfoLr,
    skip_sgr_eval: bool,
}

impl RscState {
    fn new() -> Self {
        RscState {
            sse: [0; RESTORE_SWITCHABLE_TYPES],
            total_sse: [0; RESTORE_TYPES],
            total_bits: [0; RESTORE_TYPES],
            ref_wiener: WienerInfoLr::default(),
            ref_sgrproj: SgrprojInfoLr::default(),
            switchable_ref_wiener: WienerInfoLr::default(),
            switchable_ref_sgrproj: SgrprojInfoLr::default(),
            skip_sgr_eval: false,
        }
    }

    /// `rsc_on_tile`.
    fn on_tile(&mut self) {
        self.ref_wiener = WienerInfoLr::default();
        self.ref_sgrproj = SgrprojInfoLr::default();
        self.switchable_ref_wiener = WienerInfoLr::default();
        self.switchable_ref_sgrproj = SgrprojInfoLr::default();
    }

    /// `reset_rsc`.
    fn reset(&mut self) {
        self.total_sse = [0; RESTORE_TYPES];
        self.total_bits = [0; RESTORE_TYPES];
    }
}

/// `search_norestore` (pickrst.c).
fn search_norestore(ctx: &PlaneCtx<'_>, limits: (i32, i32, i32, i32), rsc: &mut RscState) {
    rsc.sse[RESTORE_NONE as usize] = ctx.sse_none(limits);
    rsc.total_sse[RESTORE_NONE as usize] += rsc.sse[RESTORE_NONE as usize];
}

/// `finer_search_wiener` (pickrst.c): the ±{4,2,1} symmetric tap refinement
/// driven by real filter applications.
fn finer_search_wiener(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    limits: (i32, i32, i32, i32),
    rui: &mut LrUnitInfo,
    wiener_win: usize,
) -> i64 {
    let plane_off = (WIENER_WIN - wiener_win) >> 1;
    let mut err = ctx.try_restoration_unit(limits, rui, input.bit_depth);
    if input.sf.disable_wiener_coeff_refine_search {
        return err;
    }
    let tap_min = [-5i16, -23, -17];
    let tap_max = [10i16, 8, 46];
    const START_STEP: i16 = 4;

    // dir 0 = hfilter first (like C), then vfilter, at each step size.
    let mut s = START_STEP;
    while s >= 1 {
        for dir in 0..2 {
            for p in plane_off..WIENER_HALFWIN {
                let mut skip = false;
                loop {
                    let f = if dir == 0 {
                        &mut rui.wiener.hfilter
                    } else {
                        &mut rui.wiener.vfilter
                    };
                    if f[p] - s >= tap_min[p] {
                        f[p] -= s;
                        f[WIENER_WIN - p - 1] -= s;
                        f[WIENER_HALFWIN] += 2 * s;
                        let err2 = ctx.try_restoration_unit(limits, rui, input.bit_depth);
                        if err2 > err {
                            let f = if dir == 0 {
                                &mut rui.wiener.hfilter
                            } else {
                                &mut rui.wiener.vfilter
                            };
                            f[p] += s;
                            f[WIENER_WIN - p - 1] += s;
                            f[WIENER_HALFWIN] -= 2 * s;
                        } else {
                            err = err2;
                            skip = true;
                            // At the highest step size continue moving in the
                            // same direction.
                            if s == START_STEP {
                                continue;
                            }
                        }
                    }
                    break;
                }
                if skip {
                    break;
                }
                loop {
                    let f = if dir == 0 {
                        &mut rui.wiener.hfilter
                    } else {
                        &mut rui.wiener.vfilter
                    };
                    if f[p] + s <= tap_max[p] {
                        f[p] += s;
                        f[WIENER_WIN - p - 1] += s;
                        f[WIENER_HALFWIN] -= 2 * s;
                        let err2 = ctx.try_restoration_unit(limits, rui, input.bit_depth);
                        if err2 > err {
                            let f = if dir == 0 {
                                &mut rui.wiener.hfilter
                            } else {
                                &mut rui.wiener.vfilter
                            };
                            f[p] -= s;
                            f[WIENER_WIN - p - 1] -= s;
                            f[WIENER_HALFWIN] += 2 * s;
                        } else {
                            err = err2;
                            if s == START_STEP {
                                continue;
                            }
                        }
                    }
                    break;
                }
            }
        }
        s >>= 1;
    }
    err
}

/// `search_wiener` (pickrst.c).
#[allow(clippy::too_many_arguments)]
fn search_wiener(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    limits: (i32, i32, i32, i32),
    rsc: &mut RscState,
    rusi: &mut RestUnitSearchInfo,
) {
    let bits_none = input.wiener_restore_cost[0] as i64;

    // Skip Wiener search for low variance contents.
    if input.sf.prune_wiener_based_on_src_var > 0 {
        let scale = [0u64, 1, 2];
        let qs = (input.dc_quant_qtx >> 3) as u64;
        let thresh = (qs * qs * scale[input.sf.prune_wiener_based_on_src_var as usize]) >> 4;
        let src_var = ctx.src_var(limits);
        let prune_wiener = (src_var < thresh) || (rsc.sse[RESTORE_NONE as usize] == 0);
        if prune_wiener {
            rsc.total_bits[RESTORE_WIENER as usize] += bits_none;
            rsc.total_sse[RESTORE_WIENER as usize] += rsc.sse[RESTORE_NONE as usize];
            rusi.best_rtype[RESTORE_WIENER as usize - 1] = RESTORE_NONE;
            rsc.sse[RESTORE_WIENER as usize] = i64::MAX;
            if input.sf.prune_sgr_based_on_wiener == 2 {
                rsc.skip_sgr_eval = true;
            }
            return;
        }
    }

    let wiener_win = if ctx.plane == 0 {
        WIENER_WIN
    } else {
        WIENER_WIN_CHROMA
    };
    let reduced_wiener_win = if input.sf.reduce_wiener_window_size {
        if ctx.plane == 0 {
            WIENER_WIN_REDUCED
        } else {
            WIENER_WIN_CHROMA
        }
    } else {
        wiener_win
    };

    let mut m = [0i64; WIENER_WIN2];
    let mut h = [0i64; WIENER_WIN2 * WIENER_WIN2];
    let (v0, v1, h0, h1) = limits;
    let dgd_origin = ctx.pad_off(0, 0);
    if input.highbd {
        compute_stats_highbd(
            reduced_wiener_win,
            &ctx.dgd_pad,
            dgd_origin,
            ctx.src,
            h0,
            h1,
            v0,
            v1,
            ctx.w_stride as i32,
            ctx.src_stride as i32,
            &mut m,
            &mut h,
            input.bit_depth,
        );
    } else {
        compute_stats(
            reduced_wiener_win,
            &ctx.dgd_pad,
            dgd_origin,
            ctx.src,
            h0,
            h1,
            v0,
            v1,
            ctx.w_stride as i32,
            ctx.src_stride as i32,
            &mut m,
            &mut h,
            input.sf.use_downsampled_wiener_stats,
        );
    }

    let mut vfilter = [0i32; WIENER_WIN];
    let mut hfilter = [0i32; WIENER_WIN];
    wiener_decompose_sep_sym(reduced_wiener_win, &m, &h, &mut vfilter, &mut hfilter);

    let mut rui = LrUnitInfo {
        restoration_type: RESTORE_WIENER,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: SgrprojInfoLr { ep: 0, xqd: [0, 0] },
    };
    finalize_sym_filter(reduced_wiener_win, &vfilter, &mut rui.wiener.vfilter);
    finalize_sym_filter(reduced_wiener_win, &hfilter, &mut rui.wiener.hfilter);

    // Filter-score gate: revert to identity (NONE) when the learned filter
    // does not reduce x'Hx - 2x'M.
    if compute_score(
        reduced_wiener_win,
        &m,
        &h,
        &rui.wiener.vfilter,
        &rui.wiener.hfilter,
    ) > 0
    {
        rsc.total_bits[RESTORE_WIENER as usize] += bits_none;
        rsc.total_sse[RESTORE_WIENER as usize] += rsc.sse[RESTORE_NONE as usize];
        rusi.best_rtype[RESTORE_WIENER as usize - 1] = RESTORE_NONE;
        rsc.sse[RESTORE_WIENER as usize] = i64::MAX;
        if input.sf.prune_sgr_based_on_wiener == 2 {
            rsc.skip_sgr_eval = true;
        }
        return;
    }

    rsc.sse[RESTORE_WIENER as usize] =
        finer_search_wiener(ctx, input, limits, &mut rui, reduced_wiener_win);
    rusi.wiener = rui.wiener;

    let bits_wiener = input.wiener_restore_cost[1] as i64
        + ((count_wiener_bits(wiener_win, &rusi.wiener, &rsc.ref_wiener) as i64)
            << AV1_PROB_COST_SHIFT);

    let cost_none = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_none >> 4,
        rsc.sse[RESTORE_NONE as usize],
        input.bit_depth,
    );
    let cost_wiener = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_wiener >> 4,
        rsc.sse[RESTORE_WIENER as usize],
        input.bit_depth,
    );

    let rtype = if cost_wiener < cost_none {
        RESTORE_WIENER
    } else {
        RESTORE_NONE
    };
    rusi.best_rtype[RESTORE_WIENER as usize - 1] = rtype;

    if input.sf.prune_sgr_based_on_wiener == 1 {
        rsc.skip_sgr_eval = cost_wiener > (1.01 * cost_none);
    } else if input.sf.prune_sgr_based_on_wiener == 2 {
        rsc.skip_sgr_eval = rusi.best_rtype[RESTORE_WIENER as usize - 1] == RESTORE_NONE;
    }

    rsc.total_sse[RESTORE_WIENER as usize] += rsc.sse[rtype as usize];
    rsc.total_bits[RESTORE_WIENER as usize] += if cost_wiener < cost_none {
        bits_wiener
    } else {
        bits_none
    };
    if cost_wiener < cost_none {
        rsc.ref_wiener = rusi.wiener;
    }
}

/// `apply_sgr` (pickrst.c): the SGR passes over the unit in procunit tiles,
/// producing flt0/flt1 at `flt_stride`.
#[allow(clippy::too_many_arguments)]
fn apply_sgr_unit(
    ctx: &mut PlaneCtx<'_>,
    ep: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    pu_width: usize,
    pu_height: usize,
    flt_stride: usize,
    bit_depth: i32,
) {
    let mut i = 0usize;
    while i < height {
        let h = pu_height.min(height - i);
        let mut j = 0usize;
        while j < width {
            let w = pu_width.min(width - j);
            let flt_off = i * flt_stride + j;
            let (f0, f1) = (&mut ctx.flt0[flt_off..], &mut ctx.flt1[flt_off..]);
            selfguided_restoration(
                &ctx.dgd_pad,
                dgd_off + i * ctx.w_stride + j,
                ctx.w_stride,
                w,
                h,
                f0,
                f1,
                flt_stride,
                ep,
                bit_depth,
            );
            j += pu_width;
        }
        i += pu_height;
    }
}

/// `get_pixel_proj_error` (pickrst.c): xqd -> xq, then the exact projected
/// SSE.
#[allow(clippy::too_many_arguments)]
fn get_pixel_proj_error_xqd(
    ctx: &PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    flt_stride: usize,
    xqd: [i32; 2],
    ep: usize,
) -> i64 {
    let xq = decode_xq(&xqd, ep);
    pixel_proj_error(
        ctx.src,
        src_off,
        width,
        height,
        ctx.src_stride,
        &ctx.dgd_pad,
        dgd_off,
        ctx.w_stride,
        &ctx.flt0,
        flt_stride,
        &ctx.flt1,
        flt_stride,
        xq,
        ep,
        input.highbd,
    )
}

/// `finer_search_pixel_proj_error` (pickrst.c): the ±{2,1} xqd refinement.
#[allow(clippy::too_many_arguments)]
fn finer_search_pixel_proj_error(
    ctx: &PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    flt_stride: usize,
    start_step: i32,
    xqd: &mut [i32; 2],
    ep: usize,
) -> i64 {
    let mut err =
        get_pixel_proj_error_xqd(ctx, input, src_off, dgd_off, width, height, flt_stride, *xqd, ep);
    let (rads, _) = SGR_PARAMS[ep];
    let tap_min = [SGRPROJ_PRJ_MIN0, SGRPROJ_PRJ_MIN1];
    let tap_max = [SGRPROJ_PRJ_MAX0, SGRPROJ_PRJ_MAX1];
    let mut s = start_step;
    while s >= 1 {
        for p in 0..2 {
            if (rads[0] == 0 && p == 0) || (rads[1] == 0 && p == 1) {
                continue;
            }
            let mut skip = false;
            loop {
                if xqd[p] - s >= tap_min[p] {
                    xqd[p] -= s;
                    let err2 = get_pixel_proj_error_xqd(
                        ctx, input, src_off, dgd_off, width, height, flt_stride, *xqd, ep,
                    );
                    if err2 > err {
                        xqd[p] += s;
                    } else {
                        err = err2;
                        skip = true;
                        if s == start_step {
                            continue;
                        }
                    }
                }
                break;
            }
            if skip {
                break;
            }
            loop {
                if xqd[p] + s <= tap_max[p] {
                    xqd[p] += s;
                    let err2 = get_pixel_proj_error_xqd(
                        ctx, input, src_off, dgd_off, width, height, flt_stride, *xqd, ep,
                    );
                    if err2 > err {
                        xqd[p] -= s;
                    } else {
                        err = err2;
                        if s == start_step {
                            continue;
                        }
                    }
                }
                break;
            }
        }
        s >>= 1;
    }
    err
}

/// `compute_sgrproj_err` (pickrst.c) for one `ep`.
#[allow(clippy::too_many_arguments)]
fn compute_sgrproj_err(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    pu_width: usize,
    pu_height: usize,
    ep: usize,
    flt_stride: usize,
) -> ([i32; 2], i64) {
    apply_sgr_unit(
        ctx,
        ep,
        dgd_off,
        width,
        height,
        pu_width,
        pu_height,
        flt_stride,
        input.bit_depth,
    );
    let exq = get_proj_subspace(
        ctx.src,
        src_off,
        width,
        height,
        ctx.src_stride,
        &ctx.dgd_pad,
        dgd_off,
        ctx.w_stride,
        &ctx.flt0,
        flt_stride,
        &ctx.flt1,
        flt_stride,
        ep,
    );
    let mut exqd = encode_xq(exq, ep);
    let err = finer_search_pixel_proj_error(
        ctx, input, src_off, dgd_off, width, height, flt_stride, 2, &mut exqd, ep,
    );
    (exqd, err)
}

/// `search_selfguided_restoration` (pickrst.c): the ep ladder.
#[allow(clippy::too_many_arguments)]
fn search_selfguided_restoration(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    src_off: usize,
    dgd_off: usize,
    width: usize,
    height: usize,
    pu_width: usize,
    pu_height: usize,
) -> SgrprojInfoLr {
    let flt_stride = ((width + 7) & !7) + 8;
    let mut bestep = 0i32;
    let mut besterr: i64 = -1;
    let mut bestxqd = [0i32; 2];
    let consider = |ctx: &mut PlaneCtx<'_>,
                        ep: i32,
                        bestep: &mut i32,
                        besterr: &mut i64,
                        bestxqd: &mut [i32; 2]| {
        let (exqd, err) = compute_sgrproj_err(
            ctx, input, src_off, dgd_off, width, height, pu_width, pu_height, ep as usize,
            flt_stride,
        );
        if *besterr == -1 || err < *besterr {
            *bestep = ep;
            *besterr = err;
            *bestxqd = exqd;
        }
    };
    if input.sf.enable_sgr_ep_pruning == 0 {
        for ep in 0..16 {
            consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
        }
    } else {
        // Evaluate the four group-1 seeds.
        for &ep in &SGRPROJ_EP_GRP1_SEED {
            consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
        }
        if input.sf.enable_sgr_ep_pruning < 2 {
            // Left/right of the winner within group 1.
            let bestep_ref = bestep;
            let mut ep = bestep_ref - 1;
            while ep < bestep_ref + 2 {
                if ep >= SGRPROJ_EP_GRP1_START_IDX && ep <= SGRPROJ_EP_GRP1_END_IDX {
                    consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
                }
                ep += 2;
            }
            // The two group-2/3 rows indexed by the current winner.
            for idx in 0..2 {
                let ep = SGRPROJ_EP_GRP2_3[idx][bestep as usize];
                consider(ctx, ep, &mut bestep, &mut besterr, &mut bestxqd);
            }
        }
    }
    SgrprojInfoLr {
        ep: bestep,
        xqd: bestxqd,
    }
}

/// `search_sgrproj` (pickrst.c).
fn search_sgrproj(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    limits: (i32, i32, i32, i32),
    rsc: &mut RscState,
    rusi: &mut RestUnitSearchInfo,
) {
    let bits_none = input.sgrproj_restore_cost[0] as i64;
    if rsc.skip_sgr_eval {
        rsc.total_bits[RESTORE_SGRPROJ as usize] += bits_none;
        rsc.total_sse[RESTORE_SGRPROJ as usize] += rsc.sse[RESTORE_NONE as usize];
        rusi.best_rtype[RESTORE_SGRPROJ as usize - 1] = RESTORE_NONE;
        rsc.sse[RESTORE_SGRPROJ as usize] = i64::MAX;
        return;
    }

    let (v0, v1, h0, h1) = limits;
    let dgd_off = ctx.pad_off(v0, h0);
    let src_off = v0 as usize * ctx.src_stride + h0 as usize;
    let procunit_width = (RESTORATION_PROC_UNIT_SIZE >> ctx.sx) as usize;
    let procunit_height = (RESTORATION_PROC_UNIT_SIZE >> ctx.sy) as usize;

    rusi.sgrproj = search_selfguided_restoration(
        ctx,
        input,
        src_off,
        dgd_off,
        (h1 - h0) as usize,
        (v1 - v0) as usize,
        procunit_width,
        procunit_height,
    );

    let rui = LrUnitInfo {
        restoration_type: RESTORE_SGRPROJ,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: rusi.sgrproj,
    };
    rsc.sse[RESTORE_SGRPROJ as usize] = ctx.try_restoration_unit(limits, &rui, input.bit_depth);

    let bits_sgr = input.sgrproj_restore_cost[1] as i64
        + ((count_sgrproj_bits(&rusi.sgrproj, &rsc.ref_sgrproj) as i64) << AV1_PROB_COST_SHIFT);
    let cost_none = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_none >> 4,
        rsc.sse[RESTORE_NONE as usize],
        input.bit_depth,
    );
    let mut cost_sgr = rdcost_dbl_with_native_bd_dist(
        input.rdmult,
        bits_sgr >> 4,
        rsc.sse[RESTORE_SGRPROJ as usize],
        input.bit_depth,
    );
    if rusi.sgrproj.ep < 10 {
        cost_sgr *= 1.0 + DUAL_SGR_PENALTY_MULT * input.sf.dual_sgr_penalty_level as f64;
    }

    let rtype = if cost_sgr < cost_none {
        RESTORE_SGRPROJ
    } else {
        RESTORE_NONE
    };
    rusi.best_rtype[RESTORE_SGRPROJ as usize - 1] = rtype;

    rsc.total_sse[RESTORE_SGRPROJ as usize] += rsc.sse[if rtype == RESTORE_SGRPROJ {
        RESTORE_SGRPROJ as usize
    } else {
        RESTORE_NONE as usize
    }];
    rsc.total_bits[RESTORE_SGRPROJ as usize] += if cost_sgr < cost_none {
        bits_sgr
    } else {
        bits_none
    };
    if cost_sgr < cost_none {
        rsc.ref_sgrproj = rusi.sgrproj;
    }
}

/// `search_switchable` (pickrst.c).
fn search_switchable(
    ctx: &PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    rsc: &mut RscState,
    rusi: &mut RestUnitSearchInfo,
) {
    let wiener_win = if ctx.plane == 0 {
        WIENER_WIN
    } else {
        WIENER_WIN_CHROMA
    };

    let mut best_cost = 0.0f64;
    let mut best_bits: i64 = 0;
    let mut best_rtype = RESTORE_NONE;

    for r in 0..RESTORE_SWITCHABLE_TYPES as u8 {
        // Prune on SSE, not on the previous search's pick (see pickrst.c).
        if r > RESTORE_NONE && rsc.sse[r as usize] > rsc.sse[RESTORE_NONE as usize] {
            continue;
        }

        let sse = rsc.sse[r as usize];
        let coeff_pcost: i64 = match r {
            RESTORE_NONE => 0,
            RESTORE_WIENER => {
                count_wiener_bits(wiener_win, &rusi.wiener, &rsc.switchable_ref_wiener) as i64
            }
            _ => count_sgrproj_bits(&rusi.sgrproj, &rsc.switchable_ref_sgrproj) as i64,
        };
        let coeff_bits = coeff_pcost << AV1_PROB_COST_SHIFT;
        let bits = input.switchable_restore_cost[r as usize] as i64 + coeff_bits;
        let mut cost =
            rdcost_dbl_with_native_bd_dist(input.rdmult, bits >> 4, sse, input.bit_depth);
        if r == RESTORE_SGRPROJ && rusi.sgrproj.ep < 10 {
            cost *= 1.0 + DUAL_SGR_PENALTY_MULT * input.sf.dual_sgr_penalty_level as f64;
        }
        if r == RESTORE_WIENER || r == RESTORE_SGRPROJ {
            cost *= 1.0 + WIENER_SGR_PENALTY_MULT * input.sf.switchable_lr_with_bias_level as f64;
        }
        if r == 0 || cost < best_cost {
            best_cost = cost;
            best_bits = bits;
            best_rtype = r;
        }
    }

    rusi.best_rtype[RESTORE_SWITCHABLE as usize - 1] = best_rtype;

    rsc.total_sse[RESTORE_SWITCHABLE as usize] += rsc.sse[best_rtype as usize];
    rsc.total_bits[RESTORE_SWITCHABLE as usize] += best_bits;
    if best_rtype == RESTORE_WIENER {
        rsc.switchable_ref_wiener = rusi.wiener;
    }
    if best_rtype == RESTORE_SGRPROJ {
        rsc.switchable_ref_sgrproj = rusi.sgrproj;
    }
}

/// `av1_derive_flags_for_lr_processing` (pickrst.c).
fn derive_flags_for_lr_processing(sf: &LrSearchSf) -> [bool; RESTORE_TYPES] {
    let w = sf.disable_wiener_filter;
    let s = sf.disable_sgr_filter;
    [w && s, w, s, w || s]
}

/// `restoration_search` (pickrst.c): one plane at one unit size — the
/// SB-coding-order unit walk running each enabled search fn per unit.
#[allow(clippy::too_many_arguments)]
fn restoration_search(
    ctx: &mut PlaneCtx<'_>,
    input: &LrSearchInput<'_>,
    lr_geom: &LrFrameConfig,
    rsc: &mut RscState,
    rusi: &mut [RestUnitSearchInfo],
    disable_lr_filter: &[bool; RESTORE_TYPES],
) {
    let plane = ctx.plane;
    let ru_size = lr_geom.unit_size[plane];
    let ext_size = ru_size * 3 / 2;
    let (horz_units, vert_units) = lr_geom.plane_units(plane, input.ss_x, input.ss_y);
    let plane_num_units = (horz_units * vert_units) as usize;
    let num_rtypes = if plane_num_units > 1 {
        RESTORE_TYPES
    } else {
        RESTORE_SWITCHABLE_TYPES
    };
    let mib_size = 1i32 << input.mib_size_log2;

    rsc.reset();

    for &(sb_row_start, sb_row_end) in &input.tile_sb_rows {
        for &(sb_col_start, sb_col_end) in &input.tile_sb_cols {
            // Reset reference parameters for delta-coding at tile start.
            rsc.on_tile();

            for sb_row in sb_row_start..sb_row_end {
                let mi_row = sb_row << input.mib_size_log2;
                for sb_col in sb_col_start..sb_col_end {
                    let mi_col = sb_col << input.mib_size_log2;
                    let Some((rcol0, rcol1, rrow0, rrow1)) = lr_corners_in_sb(
                        lr_geom, plane, input.ss_x, input.ss_y, mi_row, mi_col, mib_size, mib_size,
                    ) else {
                        continue;
                    };

                    for rrow in rrow0..rrow1 {
                        let y0 = rrow * ru_size;
                        let remaining_h = ctx.ph - y0;
                        let h = if remaining_h < ext_size {
                            remaining_h
                        } else {
                            ru_size
                        };
                        let mut v_start = y0;
                        let mut v_end = y0 + h;
                        debug_assert!(v_end <= ctx.ph);
                        // Offset upwards to align with the processing stripe.
                        let voffset = RESTORATION_UNIT_OFFSET >> ctx.sy;
                        v_start = (v_start - voffset).max(0);
                        if v_end < ctx.ph {
                            v_end -= voffset;
                        }

                        for rcol in rcol0..rcol1 {
                            let x0 = rcol * ru_size;
                            let remaining_w = ctx.pw - x0;
                            let w = if remaining_w < ext_size {
                                remaining_w
                            } else {
                                ru_size
                            };
                            let limits = (v_start, v_end, x0, x0 + w);
                            let unit_idx = (rrow * horz_units + rcol) as usize;

                            rsc.skip_sgr_eval = false;
                            for r in 0..num_rtypes {
                                if disable_lr_filter[r] {
                                    continue;
                                }
                                match r {
                                    0 => search_norestore(ctx, limits, rsc),
                                    1 => search_wiener(ctx, input, limits, rsc, &mut rusi[unit_idx]),
                                    2 => search_sgrproj(ctx, input, limits, rsc, &mut rusi[unit_idx]),
                                    _ => search_switchable(ctx, input, rsc, &mut rusi[unit_idx]),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `copy_unit_info` (pickrst.c).
fn copy_unit_info(frame_rtype: u8, rusi: &RestUnitSearchInfo) -> LrUnitInfo {
    debug_assert!(frame_rtype > 0);
    let rtype = rusi.best_rtype[frame_rtype as usize - 1];
    let mut u = LrUnitInfo {
        restoration_type: rtype,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: SgrprojInfoLr { ep: 0, xqd: [0, 0] },
    };
    if rtype == RESTORE_WIENER {
        u.wiener = rusi.wiener;
    } else {
        u.sgrproj = rusi.sgrproj;
    }
    u
}

/// `av1_pick_filter_restoration` (pickrst.c): the frame-level search over
/// unit sizes and planes. Returns the chosen unit size, per-plane frame
/// restoration types and per-unit parameters.
pub fn pick_filter_restoration(input: &LrSearchInput<'_>) -> LrSearchOutcome {
    let num_planes = input.planes.len();
    let sb_wide = 1i32 << (input.mib_size_log2 + 2); // block_size_wide[sb_size]

    // The minimum allowed unit size at a syntax level is 1 superblock.
    let min_lr_unit_size = input.sf.min_lr_unit_size.max(sb_wide);
    let max_lr_unit_size = input.sf.max_lr_unit_size.max(min_lr_unit_size);

    let mut outcome = LrSearchOutcome {
        unit_size: max_lr_unit_size,
        frame_restoration_type: [RESTORE_NONE; 3],
        units: [Vec::new(), Vec::new(), Vec::new()],
    };

    // Decide which planes to search.
    let plane_start = if input.sf.disable_loop_restoration_luma {
        1usize
    } else {
        0
    };
    let plane_end = if num_planes == 1 || input.sf.disable_loop_restoration_chroma {
        0usize
    } else {
        2
    };
    if plane_start > plane_end {
        return outcome;
    }

    let disable_lr_filter = derive_flags_for_lr_processing(&input.sf);
    // Wiener+SGR both disabled: nothing to search (the C search loop would
    // skip every fn and pick NONE everywhere).
    if disable_lr_filter[RESTORE_NONE as usize] {
        return outcome;
    }

    // Stage the searched planes (av1_extend_frame + boundary saves happen
    // once, before the size loop).
    let mut ctxs: Vec<PlaneCtx<'_>> = (plane_start..=plane_end)
        .map(|p| PlaneCtx::new(input, p))
        .collect();

    let mut best_cost = f64::MAX;
    let mut best_luma_unit_size = max_lr_unit_size;
    let mut rsc = RscState::new();

    let mut luma_unit_size = max_lr_unit_size;
    while luma_unit_size >= min_lr_unit_size {
        let lr_geom = LrFrameConfig {
            frame_restoration_type: [RESTORE_WIENER; 3], // corners fn ignores this
            unit_size: [luma_unit_size; 3],
            crop_width: input.crop_width,
            crop_height: input.crop_height,
            superres_denom: 0,
        };

        let mut bits_this_size: i64 = 0;
        let mut sse_this_size: i64 = 0;
        let mut best_rtype: [u8; 3] = [RESTORE_NONE; 3];
        let mut rusi_this_size: Vec<Vec<RestUnitSearchInfo>> = Vec::new();

        for (ci, plane) in (plane_start..=plane_end).enumerate() {
            let ctx = &mut ctxs[ci];
            let (hu, vu) = lr_geom.plane_units(plane, input.ss_x, input.ss_y);
            let plane_num_units = (hu * vu) as usize;
            let mut rusi = vec![RestUnitSearchInfo::default(); plane_num_units];

            restoration_search(ctx, input, &lr_geom, &mut rsc, &mut rusi, &disable_lr_filter);

            let num_rtypes = if plane_num_units > 1 {
                RESTORE_TYPES
            } else {
                RESTORE_SWITCHABLE_TYPES
            };
            let mut best_cost_this_plane = f64::MAX;
            for r in 0..num_rtypes {
                if disable_lr_filter[r] {
                    continue;
                }
                // switchable_lr_with_bias_level restricts to SWITCHABLE.
                if input.sf.switchable_lr_with_bias_level > 0
                    && (r == RESTORE_WIENER as usize || r == RESTORE_SGRPROJ as usize)
                {
                    continue;
                }
                let cost_this_plane = rdcost_dbl_with_native_bd_dist(
                    input.rdmult,
                    rsc.total_bits[r] >> 4,
                    rsc.total_sse[r],
                    input.bit_depth,
                );
                if cost_this_plane < best_cost_this_plane {
                    best_cost_this_plane = cost_this_plane;
                    best_rtype[plane] = r as u8;
                }
            }

            bits_this_size += rsc.total_bits[best_rtype[plane] as usize];
            sse_this_size += rsc.total_sse[best_rtype[plane] as usize];
            rusi_this_size.push(rusi);
        }

        let cost_this_size = rdcost_dbl_with_native_bd_dist(
            input.rdmult,
            bits_this_size >> 4,
            sse_this_size,
            input.bit_depth,
        );

        if cost_this_size < best_cost {
            best_cost = cost_this_size;
            best_luma_unit_size = luma_unit_size;
            // Copy parameters out before the next size overwrites them.
            let mut all_none = true;
            for (ci, plane) in (plane_start..=plane_end).enumerate() {
                outcome.frame_restoration_type[plane] = best_rtype[plane];
                outcome.units[plane].clear();
                if best_rtype[plane] != RESTORE_NONE {
                    all_none = false;
                    for u in &rusi_this_size[ci] {
                        outcome.units[plane].push(copy_unit_info(best_rtype[plane], u));
                    }
                }
            }
            // Heuristic: all NONE at this size -> smaller sizes won't help.
            if all_none {
                break;
            }
        } else {
            // Heuristic: worse than the previous (larger) size -> stop.
            break;
        }

        luma_unit_size >>= 1;
    }

    outcome.unit_size = best_luma_unit_size;
    outcome
}
