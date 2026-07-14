//! Byte-exact AV1 film-grain synthesis — a faithful port of libaom v3.14.1
//! `av1/decoder/grain_synthesis.c` (`av1_add_film_grain` / `add_film_grain_run`).
//!
//! Film grain is a post-reconstruction OUTPUT stage: a seeded LFSR builds grain
//! templates from a fixed gaussian sequence, a piecewise-linear scaling function
//! maps luma -> grain scale, and the scaled grain is blended into the decoded
//! planes with subblock overlap and optional chroma-from-luma. Deterministic and
//! bit-exact vs the C reference (see tests/film_grain_diff.rs, gated on the REAL
//! exported `av1_add_film_grain`).
//!
//! The C reference has separate lowbd (uint8_t) and hbd (uint16_t) noise-add
//! paths; they are numerically identical at `bit_depth == 8` (the hbd offset /
//! clip / scale-LUT formulas all reduce to the lowbd constants when
//! `bit_depth - 8 == 0`), so this port carries ONE `bit_depth`-parameterized
//! [`add_noise_to_block`] that matches C at 8/10/12-bit. Planes are `u16` at
//! every depth (as the decoder stores them); arithmetic is `i32`, matching C
//! `int`.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use aom_entropy::header::FilmGrainParams;

include!("film_grain_gaussian.rs");

const GAUSS_BITS: i32 = 11;
const LUMA_SUBBLOCK_SIZE_Y: i32 = 32;
const LUMA_SUBBLOCK_SIZE_X: i32 = 32;
const MIN_LUMA_LEGAL_RANGE: i32 = 16;
const MAX_LUMA_LEGAL_RANGE: i32 = 235;
const MIN_CHROMA_LEGAL_RANGE: i32 = 16;
const MAX_CHROMA_LEGAL_RANGE: i32 = 240;

#[inline]
fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// Per-plane scaling LUTs (256 entries each), mirroring `aom_grain_scaling_lut_t`.
struct ScalingLut {
    y: [i32; 256],
    cb: [i32; 256],
    cr: [i32; 256],
}

/// `aom_grain_rng_t`: the 16-bit LFSR register.
struct GrainRng {
    random_register: u16,
}

impl GrainRng {
    /// `get_random_number` (grain_synthesis.c) — a number in `[0, 2^bits - 1]`.
    #[inline]
    fn get_random_number(&mut self, bits: i32) -> i32 {
        let r = self.random_register;
        let bit = ((r) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        self.random_register = (r >> 1) | (bit << 15);
        ((self.random_register >> (16 - bits)) as i32) & ((1 << bits) - 1)
    }
}

/// `init_random_generator` — seed the LFSR for a given luma line.
fn init_random_generator(rng: &mut GrainRng, luma_line: i32, seed: u16) {
    let msb = ((seed >> 8) & 255) as i32;
    let lsb = (seed & 255) as i32;
    let mut r = (msb << 8) + lsb;
    let luma_num = luma_line >> 5;
    r ^= ((luma_num * 37 + 178) & 255) << 8;
    r ^= (luma_num * 173 + 105) & 255;
    rng.random_register = r as u16;
}

/// Build `pred_pos_luma` / `pred_pos_chroma` (the AR neighbour offsets), exactly
/// as `init_arrays` does. Each entry is `[row_off, col_off, use_luma_avg]`.
fn build_pred_pos(p: &FilmGrainParams) -> (Vec<[i32; 3]>, Vec<[i32; 3]>) {
    let lag = p.ar_coeff_lag;
    let num_pos_luma = 2 * lag * (lag + 1);
    let mut num_pos_chroma = num_pos_luma;
    if p.num_y_points > 0 {
        num_pos_chroma += 1;
    }
    let mut pred_pos_luma = vec![[0i32; 3]; num_pos_luma as usize];
    let mut pred_pos_chroma = vec![[0i32; 3]; num_pos_chroma as usize];

    let mut idx = 0usize;
    for row in -lag..0 {
        for col in -lag..(lag + 1) {
            pred_pos_luma[idx] = [row, col, 0];
            pred_pos_chroma[idx] = [row, col, 0];
            idx += 1;
        }
    }
    for col in -lag..0 {
        pred_pos_luma[idx] = [0, col, 0];
        pred_pos_chroma[idx] = [0, col, 0];
        idx += 1;
    }
    if p.num_y_points > 0 {
        pred_pos_chroma[idx] = [0, 0, 1];
    }
    (pred_pos_luma, pred_pos_chroma)
}

/// `generate_luma_grain_block`.
fn generate_luma_grain_block(
    p: &FilmGrainParams,
    bit_depth: i32,
    rng: &mut GrainRng,
    pred_pos_luma: &[[i32; 3]],
    block: &mut [i32],
    block_y: i32,
    block_x: i32,
    stride: i32,
    left_pad: i32,
    top_pad: i32,
    right_pad: i32,
    bottom_pad: i32,
) {
    if p.num_y_points == 0 {
        for v in block.iter_mut() {
            *v = 0;
        }
        return;
    }
    let gauss_sec_shift = 12 - bit_depth + p.grain_scale_shift;
    let num_pos_luma = (2 * p.ar_coeff_lag * (p.ar_coeff_lag + 1)) as usize;
    let rounding_offset = 1 << (p.ar_coeff_shift - 1);
    let grain_min = -(1 << (bit_depth - 1));
    let grain_max = (1 << (bit_depth - 1)) - 1;

    for i in 0..block_y {
        for j in 0..block_x {
            let g = GAUSSIAN_SEQUENCE[rng.get_random_number(GAUSS_BITS) as usize];
            block[(i * stride + j) as usize] = (g + ((1 << gauss_sec_shift) >> 1)) >> gauss_sec_shift;
        }
    }

    for i in top_pad..(block_y - bottom_pad) {
        for j in left_pad..(block_x - right_pad) {
            let mut wsum = 0i32;
            for pos in 0..num_pos_luma {
                let r = i + pred_pos_luma[pos][0];
                let c = j + pred_pos_luma[pos][1];
                wsum += p.ar_coeffs_y[pos] * block[(r * stride + c) as usize];
            }
            let idx = (i * stride + j) as usize;
            block[idx] = clamp_i32(
                block[idx] + ((wsum + rounding_offset) >> p.ar_coeff_shift),
                grain_min,
                grain_max,
            );
        }
    }
}

/// `generate_chroma_grain_blocks`.
fn generate_chroma_grain_blocks(
    p: &FilmGrainParams,
    bit_depth: i32,
    rng: &mut GrainRng,
    pred_pos_chroma: &[[i32; 3]],
    luma_block: &[i32],
    cb_block: &mut [i32],
    cr_block: &mut [i32],
    luma_grain_stride: i32,
    chroma_block_y: i32,
    chroma_block_x: i32,
    chroma_grain_stride: i32,
    left_pad: i32,
    top_pad: i32,
    right_pad: i32,
    bottom_pad: i32,
    ss_y: i32,
    ss_x: i32,
) {
    let gauss_sec_shift = 12 - bit_depth + p.grain_scale_shift;
    let mut num_pos_chroma = 2 * p.ar_coeff_lag * (p.ar_coeff_lag + 1);
    if p.num_y_points > 0 {
        num_pos_chroma += 1;
    }
    let rounding_offset = 1 << (p.ar_coeff_shift - 1);
    let chroma_grain_block_size = (chroma_block_y * chroma_grain_stride) as usize;
    let grain_min = -(1 << (bit_depth - 1));
    let grain_max = (1 << (bit_depth - 1)) - 1;

    if p.num_cb_points != 0 || p.chroma_scaling_from_luma {
        init_random_generator(rng, 7 << 5, p.random_seed as u16);
        for i in 0..chroma_block_y {
            for j in 0..chroma_block_x {
                let g = GAUSSIAN_SEQUENCE[rng.get_random_number(GAUSS_BITS) as usize];
                cb_block[(i * chroma_grain_stride + j) as usize] =
                    (g + ((1 << gauss_sec_shift) >> 1)) >> gauss_sec_shift;
            }
        }
    } else {
        for v in &mut cb_block[..chroma_grain_block_size] {
            *v = 0;
        }
    }

    if p.num_cr_points != 0 || p.chroma_scaling_from_luma {
        init_random_generator(rng, 11 << 5, p.random_seed as u16);
        for i in 0..chroma_block_y {
            for j in 0..chroma_block_x {
                let g = GAUSSIAN_SEQUENCE[rng.get_random_number(GAUSS_BITS) as usize];
                cr_block[(i * chroma_grain_stride + j) as usize] =
                    (g + ((1 << gauss_sec_shift) >> 1)) >> gauss_sec_shift;
            }
        }
    } else {
        for v in &mut cr_block[..chroma_grain_block_size] {
            *v = 0;
        }
    }

    for i in top_pad..(chroma_block_y - bottom_pad) {
        for j in left_pad..(chroma_block_x - right_pad) {
            let mut wsum_cb = 0i32;
            let mut wsum_cr = 0i32;
            for pos in 0..num_pos_chroma as usize {
                if pred_pos_chroma[pos][2] == 0 {
                    let r = i + pred_pos_chroma[pos][0];
                    let c = j + pred_pos_chroma[pos][1];
                    let cidx = (r * chroma_grain_stride + c) as usize;
                    wsum_cb += p.ar_coeffs_cb[pos] * cb_block[cidx];
                    wsum_cr += p.ar_coeffs_cr[pos] * cr_block[cidx];
                } else if pred_pos_chroma[pos][2] == 1 {
                    let mut av_luma = 0i32;
                    let luma_coord_y = ((i - top_pad) << ss_y) + top_pad;
                    let luma_coord_x = ((j - left_pad) << ss_x) + left_pad;
                    for k in luma_coord_y..(luma_coord_y + ss_y + 1) {
                        for l in luma_coord_x..(luma_coord_x + ss_x + 1) {
                            av_luma += luma_block[(k * luma_grain_stride + l) as usize];
                        }
                    }
                    av_luma = (av_luma + ((1 << (ss_y + ss_x)) >> 1)) >> (ss_y + ss_x);
                    wsum_cb += p.ar_coeffs_cb[pos] * av_luma;
                    wsum_cr += p.ar_coeffs_cr[pos] * av_luma;
                }
                // pred_pos_chroma[pos][2] is only ever 0 or 1 (built above); the
                // C "prediction between two chroma components" error is unreachable.
            }
            if p.num_cb_points != 0 || p.chroma_scaling_from_luma {
                let idx = (i * chroma_grain_stride + j) as usize;
                cb_block[idx] = clamp_i32(
                    cb_block[idx] + ((wsum_cb + rounding_offset) >> p.ar_coeff_shift),
                    grain_min,
                    grain_max,
                );
            }
            if p.num_cr_points != 0 || p.chroma_scaling_from_luma {
                let idx = (i * chroma_grain_stride + j) as usize;
                cr_block[idx] = clamp_i32(
                    cr_block[idx] + ((wsum_cr + rounding_offset) >> p.ar_coeff_shift),
                    grain_min,
                    grain_max,
                );
            }
        }
    }
}

/// `init_scaling_function`.
fn init_scaling_function(scaling_points: &[[i32; 2]], num_points: i32, lut: &mut [i32; 256]) {
    if num_points == 0 {
        return;
    }
    for i in 0..scaling_points[0][0] {
        lut[i as usize] = scaling_points[0][1];
    }
    for point in 0..(num_points as usize - 1) {
        let delta_y = scaling_points[point + 1][1] - scaling_points[point][1];
        let delta_x = scaling_points[point + 1][0] - scaling_points[point][0];
        // C: int64_t delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
        // the multiply is computed in `int`, then widened.
        let quotient = (65536 + (delta_x >> 1)) / delta_x;
        let delta = (delta_y * quotient) as i64;
        for x in 0..delta_x {
            lut[(scaling_points[point][0] + x) as usize] =
                scaling_points[point][1] + (((x as i64 * delta + 32768) >> 16) as i32);
        }
    }
    for i in scaling_points[num_points as usize - 1][0]..256 {
        lut[i as usize] = scaling_points[num_points as usize - 1][1];
    }
}

/// `scale_LUT` — sample the LUT with 10/12-bit interpolation.
#[inline]
fn scale_lut(lut: &[i32; 256], index: i32, bit_depth: i32) -> i32 {
    let x = (index >> (bit_depth - 8)) as usize;
    if (bit_depth - 8) == 0 || x == 255 {
        lut[x]
    } else {
        lut[x]
            + (((lut[x + 1] - lut[x]) * (index & ((1 << (bit_depth - 8)) - 1))
                + (1 << (bit_depth - 9)))
                >> (bit_depth - 8))
    }
}

/// Unified `add_noise_to_block` / `add_noise_to_block_hbd` (identical at
/// `bit_depth == 8`). Planes are `u16`; `*_base` are element offsets into them.
fn add_noise_to_block(
    p: &FilmGrainParams,
    bit_depth: i32,
    scaling_lut: &ScalingLut,
    luma: &mut [u16],
    luma_base: usize,
    cb: &mut [u16],
    cb_base: usize,
    cr: &mut [u16],
    cr_base: usize,
    luma_stride: usize,
    chroma_stride: usize,
    luma_grain: &[i32],
    luma_grain_base: usize,
    cb_grain: &[i32],
    cb_grain_base: usize,
    cr_grain: &[i32],
    cr_grain_base: usize,
    luma_grain_stride: usize,
    chroma_grain_stride: usize,
    half_luma_height: i32,
    half_luma_width: i32,
    ss_y: i32,
    ss_x: i32,
    mc_identity: bool,
) {
    let mut cb_mult = p.cb_mult - 128;
    let mut cb_luma_mult = p.cb_luma_mult - 128;
    let mut cb_offset = (p.cb_offset << (bit_depth - 8)) - (1 << bit_depth);
    let mut cr_mult = p.cr_mult - 128;
    let mut cr_luma_mult = p.cr_luma_mult - 128;
    let mut cr_offset = (p.cr_offset << (bit_depth - 8)) - (1 << bit_depth);

    let rounding_offset = 1 << (p.scaling_shift - 1);

    let apply_y = p.num_y_points > 0;
    let apply_cb = p.num_cb_points > 0 || p.chroma_scaling_from_luma;
    let apply_cr = p.num_cr_points > 0 || p.chroma_scaling_from_luma;

    if p.chroma_scaling_from_luma {
        cb_mult = 0;
        cb_luma_mult = 64;
        cb_offset = 0;
        cr_mult = 0;
        cr_luma_mult = 64;
        cr_offset = 0;
    }

    let min_luma;
    let max_luma;
    let min_chroma;
    let max_chroma;
    if p.clip_to_restricted_range {
        min_luma = MIN_LUMA_LEGAL_RANGE << (bit_depth - 8);
        max_luma = MAX_LUMA_LEGAL_RANGE << (bit_depth - 8);
        if mc_identity {
            min_chroma = MIN_LUMA_LEGAL_RANGE << (bit_depth - 8);
            max_chroma = MAX_LUMA_LEGAL_RANGE << (bit_depth - 8);
        } else {
            min_chroma = MIN_CHROMA_LEGAL_RANGE << (bit_depth - 8);
            max_chroma = MAX_CHROMA_LEGAL_RANGE << (bit_depth - 8);
        }
    } else {
        min_luma = 0;
        min_chroma = 0;
        max_luma = (256 << (bit_depth - 8)) - 1;
        max_chroma = (256 << (bit_depth - 8)) - 1;
    }
    let index_max = (256 << (bit_depth - 8)) - 1;

    for i in 0..(half_luma_height << (1 - ss_y)) {
        for j in 0..(half_luma_width << (1 - ss_x)) {
            let average_luma;
            if ss_x != 0 {
                let base = luma_base + ((i << ss_y) as usize) * luma_stride + ((j << ss_x) as usize);
                average_luma = (luma[base] as i32 + luma[base + 1] as i32 + 1) >> 1;
            } else {
                average_luma =
                    luma[luma_base + ((i << ss_y) as usize) * luma_stride + j as usize] as i32;
            }

            if apply_cb {
                let cpix_idx = cb_base + (i as usize) * chroma_stride + j as usize;
                let cpix = cb[cpix_idx] as i32;
                let sidx = clamp_i32(
                    ((average_luma * cb_luma_mult + cb_mult * cpix) >> 6) + cb_offset,
                    0,
                    index_max,
                );
                let s = scale_lut(&scaling_lut.cb, sidx, bit_depth);
                let grain = cb_grain[cb_grain_base + (i as usize) * chroma_grain_stride + j as usize];
                cb[cpix_idx] = clamp_i32(
                    cpix + ((s * grain + rounding_offset) >> p.scaling_shift),
                    min_chroma,
                    max_chroma,
                ) as u16;
            }

            if apply_cr {
                let cpix_idx = cr_base + (i as usize) * chroma_stride + j as usize;
                let cpix = cr[cpix_idx] as i32;
                let sidx = clamp_i32(
                    ((average_luma * cr_luma_mult + cr_mult * cpix) >> 6) + cr_offset,
                    0,
                    index_max,
                );
                let s = scale_lut(&scaling_lut.cr, sidx, bit_depth);
                let grain = cr_grain[cr_grain_base + (i as usize) * chroma_grain_stride + j as usize];
                cr[cpix_idx] = clamp_i32(
                    cpix + ((s * grain + rounding_offset) >> p.scaling_shift),
                    min_chroma,
                    max_chroma,
                ) as u16;
            }
        }
    }

    if apply_y {
        for i in 0..(half_luma_height << 1) {
            for j in 0..(half_luma_width << 1) {
                let lidx = luma_base + (i as usize) * luma_stride + j as usize;
                let lpix = luma[lidx] as i32;
                let s = scale_lut(&scaling_lut.y, lpix, bit_depth);
                let grain =
                    luma_grain[luma_grain_base + (i as usize) * luma_grain_stride + j as usize];
                luma[lidx] = clamp_i32(
                    lpix + ((s * grain + rounding_offset) >> p.scaling_shift),
                    min_luma,
                    max_luma,
                ) as u16;
            }
        }
    }
}

/// `copy_area` — grain `int` buffer to grain `int` buffer (never aliasing).
fn copy_area(
    src: &[i32],
    src_base: usize,
    src_stride: usize,
    dst: &mut [i32],
    dst_base: usize,
    dst_stride: usize,
    width: i32,
    height: i32,
) {
    for r in 0..height as usize {
        for c in 0..width as usize {
            dst[dst_base + r * dst_stride + c] = src[src_base + r * src_stride + c];
        }
    }
}

/// `extend_even` — replicate the last odd column/row of the luma plane.
fn extend_even(dst: &mut [u16], dst_stride: usize, width: usize, height: usize) {
    if width % 2 == 0 && height % 2 == 0 {
        return;
    }
    if width & 1 == 1 {
        for i in 0..height {
            dst[i * dst_stride + width] = dst[i * dst_stride + width - 1];
        }
    }
    let w = (width + 1) & !1;
    if height & 1 == 1 {
        for c in 0..w {
            dst[height * dst_stride + c] = dst[(height - 1) * dst_stride + c];
        }
    }
}

/// `ver_boundary_overlap` — width 1 or 2 vertical stitch.
fn ver_boundary_overlap(
    left: &[i32],
    left_base: usize,
    left_stride: usize,
    right: &[i32],
    right_base: usize,
    right_stride: usize,
    dst: &mut [i32],
    dst_base: usize,
    dst_stride: usize,
    width: i32,
    height: i32,
    grain_min: i32,
    grain_max: i32,
) {
    if width == 1 {
        for r in 0..height as usize {
            let v = (left[left_base + r * left_stride] * 23
                + right[right_base + r * right_stride] * 22
                + 16)
                >> 5;
            dst[dst_base + r * dst_stride] = clamp_i32(v, grain_min, grain_max);
        }
    } else if width == 2 {
        for r in 0..height as usize {
            let l = left_base + r * left_stride;
            let ri = right_base + r * right_stride;
            let d = dst_base + r * dst_stride;
            dst[d] = clamp_i32((27 * left[l] + 17 * right[ri] + 16) >> 5, grain_min, grain_max);
            dst[d + 1] =
                clamp_i32((17 * left[l + 1] + 27 * right[ri + 1] + 16) >> 5, grain_min, grain_max);
        }
    }
}

/// `hor_boundary_overlap` — height 1 or 2 horizontal stitch.
fn hor_boundary_overlap(
    top: &[i32],
    top_base: usize,
    top_stride: usize,
    bottom: &[i32],
    bottom_base: usize,
    bottom_stride: usize,
    dst: &mut [i32],
    dst_base: usize,
    dst_stride: usize,
    width: i32,
    height: i32,
    grain_min: i32,
    grain_max: i32,
) {
    if height == 1 {
        for c in 0..width as usize {
            let v = (top[top_base + c] * 23 + bottom[bottom_base + c] * 22 + 16) >> 5;
            dst[dst_base + c] = clamp_i32(v, grain_min, grain_max);
        }
    } else if height == 2 {
        for c in 0..width as usize {
            dst[dst_base + c] = clamp_i32(
                (27 * top[top_base + c] + 17 * bottom[bottom_base + c] + 16) >> 5,
                grain_min,
                grain_max,
            );
            dst[dst_base + dst_stride + c] = clamp_i32(
                (17 * top[top_base + top_stride + c] + 27 * bottom[bottom_base + bottom_stride + c]
                    + 16)
                    >> 5,
                grain_min,
                grain_max,
            );
        }
    }
}

/// `add_film_grain_run` — generate the templates and stitch grain into the
/// already-extended planes (`luma`: `width x height`; chroma: subsampled).
fn add_film_grain_run(
    p: &FilmGrainParams,
    bit_depth: i32,
    luma: &mut [u16],
    cb: &mut [u16],
    cr: &mut [u16],
    height: i32,
    width: i32,
    luma_stride: usize,
    chroma_stride: usize,
    ss_y: i32,
    ss_x: i32,
    mc_identity: bool,
) {
    let mut scaling_lut = ScalingLut {
        y: [0; 256],
        cb: [0; 256],
        cr: [0; 256],
    };

    let mut rng = GrainRng {
        random_register: p.random_seed as u16,
    };

    let left_pad = 3i32;
    let right_pad = 3i32;
    let top_pad = 3i32;
    let bottom_pad = 0i32;
    let ar_padding = 3i32;

    let chroma_subblock_size_y = LUMA_SUBBLOCK_SIZE_Y >> ss_y;
    let chroma_subblock_size_x = LUMA_SUBBLOCK_SIZE_X >> ss_x;

    let luma_block_size_y = top_pad + 2 * ar_padding + LUMA_SUBBLOCK_SIZE_Y * 2 + bottom_pad;
    let luma_block_size_x =
        left_pad + 2 * ar_padding + LUMA_SUBBLOCK_SIZE_X * 2 + 2 * ar_padding + right_pad;

    let chroma_block_size_y =
        top_pad + (2 >> ss_y) * ar_padding + chroma_subblock_size_y * 2 + bottom_pad;
    let chroma_block_size_x = left_pad
        + (2 >> ss_x) * ar_padding
        + chroma_subblock_size_x * 2
        + (2 >> ss_x) * ar_padding
        + right_pad;

    let luma_grain_stride = luma_block_size_x;
    let chroma_grain_stride = chroma_block_size_x;

    let overlap = p.overlap_flag;

    let grain_min = -(1 << (bit_depth - 1));
    let grain_max = (1 << (bit_depth - 1)) - 1;

    let (pred_pos_luma, pred_pos_chroma) = build_pred_pos(p);

    // grain template blocks
    let mut luma_grain_block = vec![0i32; (luma_block_size_y * luma_grain_stride) as usize];
    let mut cb_grain_block = vec![0i32; (chroma_block_size_y * chroma_grain_stride) as usize];
    let mut cr_grain_block = vec![0i32; (chroma_block_size_y * chroma_grain_stride) as usize];

    // overlap line/column buffers (sizes per init_arrays)
    let mut y_line_buf = vec![0i32; luma_stride * 2];
    let mut cb_line_buf = vec![0i32; chroma_stride * (2 >> ss_y) as usize];
    let mut cr_line_buf = vec![0i32; chroma_stride * (2 >> ss_y) as usize];
    let mut y_col_buf = vec![0i32; ((LUMA_SUBBLOCK_SIZE_Y + 2) * 2) as usize];
    let col_h = (chroma_subblock_size_y + (2 >> ss_y)) * (2 >> ss_x);
    let mut cb_col_buf = vec![0i32; col_h as usize];
    let mut cr_col_buf = vec![0i32; col_h as usize];

    generate_luma_grain_block(
        p,
        bit_depth,
        &mut rng,
        &pred_pos_luma,
        &mut luma_grain_block,
        luma_block_size_y,
        luma_block_size_x,
        luma_grain_stride,
        left_pad,
        top_pad,
        right_pad,
        bottom_pad,
    );

    generate_chroma_grain_blocks(
        p,
        bit_depth,
        &mut rng,
        &pred_pos_chroma,
        &luma_grain_block,
        &mut cb_grain_block,
        &mut cr_grain_block,
        luma_grain_stride,
        chroma_block_size_y,
        chroma_block_size_x,
        chroma_grain_stride,
        left_pad,
        top_pad,
        right_pad,
        bottom_pad,
        ss_y,
        ss_x,
    );

    init_scaling_function(&p.scaling_points_y, p.num_y_points, &mut scaling_lut.y);
    if p.chroma_scaling_from_luma {
        scaling_lut.cb = scaling_lut.y;
        scaling_lut.cr = scaling_lut.y;
    } else {
        init_scaling_function(&p.scaling_points_cb, p.num_cb_points, &mut scaling_lut.cb);
        init_scaling_function(&p.scaling_points_cr, p.num_cr_points, &mut scaling_lut.cr);
    }

    let luma_stride_i = luma_stride as i32;
    let chroma_stride_i = chroma_stride as i32;
    let luma_grain_stride_u = luma_grain_stride as usize;
    let chroma_grain_stride_u = chroma_grain_stride as usize;

    let mut y = 0i32;
    while y < height / 2 {
        init_random_generator(&mut rng, y * 2, p.random_seed as u16);

        let mut x = 0i32;
        while x < width / 2 {
            let mut offset_y = rng.get_random_number(8);
            let offset_x = (offset_y >> 4) & 15;
            offset_y &= 15;

            let luma_offset_y = left_pad + 2 * ar_padding + (offset_y << 1);
            let luma_offset_x = top_pad + 2 * ar_padding + (offset_x << 1);

            let chroma_offset_y =
                top_pad + (2 >> ss_y) * ar_padding + offset_y * (2 >> ss_y);
            let chroma_offset_x =
                left_pad + (2 >> ss_x) * ar_padding + offset_x * (2 >> ss_x);

            if overlap && x != 0 {
                ver_boundary_overlap(
                    &y_col_buf.clone(),
                    0,
                    2,
                    &luma_grain_block,
                    (luma_offset_y * luma_grain_stride + luma_offset_x) as usize,
                    luma_grain_stride_u,
                    &mut y_col_buf,
                    0,
                    2,
                    2,
                    (LUMA_SUBBLOCK_SIZE_Y + 2).min(height - (y << 1)),
                    grain_min,
                    grain_max,
                );

                let cver_w = 2 >> ss_x;
                let cver_h = (chroma_subblock_size_y + (2 >> ss_y))
                    .min((height - (y << 1)) >> ss_y);
                ver_boundary_overlap(
                    &cb_col_buf.clone(),
                    0,
                    cver_w as usize,
                    &cb_grain_block,
                    (chroma_offset_y * chroma_grain_stride + chroma_offset_x) as usize,
                    chroma_grain_stride_u,
                    &mut cb_col_buf,
                    0,
                    cver_w as usize,
                    cver_w,
                    cver_h,
                    grain_min,
                    grain_max,
                );
                ver_boundary_overlap(
                    &cr_col_buf.clone(),
                    0,
                    cver_w as usize,
                    &cr_grain_block,
                    (chroma_offset_y * chroma_grain_stride + chroma_offset_x) as usize,
                    chroma_grain_stride_u,
                    &mut cr_col_buf,
                    0,
                    cver_w as usize,
                    cver_w,
                    cver_h,
                    grain_min,
                    grain_max,
                );

                let i = if y != 0 { 1i32 } else { 0i32 };
                let cb_col_off = (i * (2 - ss_y) * (2 - ss_x)) as usize;
                add_noise_to_block(
                    p,
                    bit_depth,
                    &scaling_lut,
                    luma,
                    (((y + i) << 1) * luma_stride_i + (x << 1)) as usize,
                    cb,
                    (((y + i) << (1 - ss_y)) * chroma_stride_i + (x << (1 - ss_x))) as usize,
                    cr,
                    (((y + i) << (1 - ss_y)) * chroma_stride_i + (x << (1 - ss_x))) as usize,
                    luma_stride,
                    chroma_stride,
                    &y_col_buf,
                    (i * 4) as usize,
                    &cb_col_buf,
                    cb_col_off,
                    &cr_col_buf,
                    cb_col_off,
                    2,
                    (2 - ss_x) as usize,
                    (LUMA_SUBBLOCK_SIZE_Y >> 1).min(height / 2 - y) - i,
                    1,
                    ss_y,
                    ss_x,
                    mc_identity,
                );
            }

            if overlap && y != 0 {
                if x != 0 {
                    let src = y_line_buf.clone();
                    hor_boundary_overlap(
                        &src,
                        (x << 1) as usize,
                        luma_stride,
                        &y_col_buf,
                        0,
                        2,
                        &mut y_line_buf,
                        (x << 1) as usize,
                        luma_stride,
                        2,
                        2,
                        grain_min,
                        grain_max,
                    );
                    let csrc_cb = cb_line_buf.clone();
                    hor_boundary_overlap(
                        &csrc_cb,
                        (x * (2 >> ss_x)) as usize,
                        chroma_stride,
                        &cb_col_buf,
                        0,
                        (2 >> ss_x) as usize,
                        &mut cb_line_buf,
                        (x * (2 >> ss_x)) as usize,
                        chroma_stride,
                        2 >> ss_x,
                        2 >> ss_y,
                        grain_min,
                        grain_max,
                    );
                    let csrc_cr = cr_line_buf.clone();
                    hor_boundary_overlap(
                        &csrc_cr,
                        (x * (2 >> ss_x)) as usize,
                        chroma_stride,
                        &cr_col_buf,
                        0,
                        (2 >> ss_x) as usize,
                        &mut cr_line_buf,
                        (x * (2 >> ss_x)) as usize,
                        chroma_stride,
                        2 >> ss_x,
                        2 >> ss_y,
                        grain_min,
                        grain_max,
                    );
                }

                let xb = if x != 0 { x + 1 } else { 0 };
                let xstep = if x != 0 { 1 } else { 0 };
                let ysrc = y_line_buf.clone();
                hor_boundary_overlap(
                    &ysrc,
                    (xb << 1) as usize,
                    luma_stride,
                    &luma_grain_block,
                    (luma_offset_y * luma_grain_stride + luma_offset_x + (xstep << 1)) as usize,
                    luma_grain_stride_u,
                    &mut y_line_buf,
                    (xb << 1) as usize,
                    luma_stride,
                    (LUMA_SUBBLOCK_SIZE_X - (xstep << 1)).min(width - (xb << 1)),
                    2,
                    grain_min,
                    grain_max,
                );

                let cbase = (xb << (1 - ss_x)) as usize;
                let cgoff = (chroma_offset_y * chroma_grain_stride
                    + chroma_offset_x
                    + (xstep << (1 - ss_x))) as usize;
                let cw = (chroma_subblock_size_x - (xstep << (1 - ss_x)))
                    .min((width - (xb << 1)) >> ss_x);
                let cbsrc = cb_line_buf.clone();
                hor_boundary_overlap(
                    &cbsrc,
                    cbase,
                    chroma_stride,
                    &cb_grain_block,
                    cgoff,
                    chroma_grain_stride_u,
                    &mut cb_line_buf,
                    cbase,
                    chroma_stride,
                    cw,
                    2 >> ss_y,
                    grain_min,
                    grain_max,
                );
                let crsrc = cr_line_buf.clone();
                hor_boundary_overlap(
                    &crsrc,
                    cbase,
                    chroma_stride,
                    &cr_grain_block,
                    cgoff,
                    chroma_grain_stride_u,
                    &mut cr_line_buf,
                    cbase,
                    chroma_stride,
                    cw,
                    2 >> ss_y,
                    grain_min,
                    grain_max,
                );

                add_noise_to_block(
                    p,
                    bit_depth,
                    &scaling_lut,
                    luma,
                    ((y << 1) * luma_stride_i + (x << 1)) as usize,
                    cb,
                    ((y << (1 - ss_y)) * chroma_stride_i + (x << (1 - ss_x))) as usize,
                    cr,
                    ((y << (1 - ss_y)) * chroma_stride_i + (x << (1 - ss_x))) as usize,
                    luma_stride,
                    chroma_stride,
                    &y_line_buf,
                    (x << 1) as usize,
                    &cb_line_buf,
                    (x << (1 - ss_x)) as usize,
                    &cr_line_buf,
                    (x << (1 - ss_x)) as usize,
                    luma_stride,
                    chroma_stride,
                    1,
                    (LUMA_SUBBLOCK_SIZE_X >> 1).min(width / 2 - x),
                    ss_y,
                    ss_x,
                    mc_identity,
                );
            }

            let i = if overlap && y != 0 { 1i32 } else { 0i32 };
            let j = if overlap && x != 0 { 1i32 } else { 0i32 };

            add_noise_to_block(
                p,
                bit_depth,
                &scaling_lut,
                luma,
                (((y + i) << 1) * luma_stride_i + ((x + j) << 1)) as usize,
                cb,
                (((y + i) << (1 - ss_y)) * chroma_stride_i + ((x + j) << (1 - ss_x))) as usize,
                cr,
                (((y + i) << (1 - ss_y)) * chroma_stride_i + ((x + j) << (1 - ss_x))) as usize,
                luma_stride,
                chroma_stride,
                &luma_grain_block,
                ((luma_offset_y + (i << 1)) * luma_grain_stride + luma_offset_x + (j << 1)) as usize,
                &cb_grain_block,
                ((chroma_offset_y + (i << (1 - ss_y))) * chroma_grain_stride
                    + chroma_offset_x
                    + (j << (1 - ss_x))) as usize,
                &cr_grain_block,
                ((chroma_offset_y + (i << (1 - ss_y))) * chroma_grain_stride
                    + chroma_offset_x
                    + (j << (1 - ss_x))) as usize,
                luma_grain_stride_u,
                chroma_grain_stride_u,
                (LUMA_SUBBLOCK_SIZE_Y >> 1).min(height / 2 - y) - i,
                (LUMA_SUBBLOCK_SIZE_X >> 1).min(width / 2 - x) - j,
                ss_y,
                ss_x,
                mc_identity,
            );

            if overlap {
                if x != 0 {
                    copy_area(
                        &y_col_buf.clone(),
                        (LUMA_SUBBLOCK_SIZE_Y << 1) as usize,
                        2,
                        &mut y_line_buf,
                        (x << 1) as usize,
                        luma_stride,
                        2,
                        2,
                    );
                    copy_area(
                        &cb_col_buf.clone(),
                        (chroma_subblock_size_y << (1 - ss_x)) as usize,
                        (2 >> ss_x) as usize,
                        &mut cb_line_buf,
                        (x << (1 - ss_x)) as usize,
                        chroma_stride,
                        2 >> ss_x,
                        2 >> ss_y,
                    );
                    copy_area(
                        &cr_col_buf.clone(),
                        (chroma_subblock_size_y << (1 - ss_x)) as usize,
                        (2 >> ss_x) as usize,
                        &mut cr_line_buf,
                        (x << (1 - ss_x)) as usize,
                        chroma_stride,
                        2 >> ss_x,
                        2 >> ss_y,
                    );
                }

                let xb = if x != 0 { x + 1 } else { 0 };
                copy_area(
                    &luma_grain_block,
                    ((luma_offset_y + LUMA_SUBBLOCK_SIZE_Y) * luma_grain_stride
                        + luma_offset_x
                        + (if x != 0 { 2 } else { 0 })) as usize,
                    luma_grain_stride_u,
                    &mut y_line_buf,
                    (xb << 1) as usize,
                    luma_stride,
                    (LUMA_SUBBLOCK_SIZE_X).min(width - (x << 1)) - (if x != 0 { 2 } else { 0 }),
                    2,
                );
                let cxoff = if x != 0 { 2 >> ss_x } else { 0 };
                copy_area(
                    &cb_grain_block,
                    ((chroma_offset_y + chroma_subblock_size_y) * chroma_grain_stride
                        + chroma_offset_x
                        + cxoff) as usize,
                    chroma_grain_stride_u,
                    &mut cb_line_buf,
                    (xb << (1 - ss_x)) as usize,
                    chroma_stride,
                    (chroma_subblock_size_x).min((width - (x << 1)) >> ss_x) - cxoff,
                    2 >> ss_y,
                );
                copy_area(
                    &cr_grain_block,
                    ((chroma_offset_y + chroma_subblock_size_y) * chroma_grain_stride
                        + chroma_offset_x
                        + cxoff) as usize,
                    chroma_grain_stride_u,
                    &mut cr_line_buf,
                    (xb << (1 - ss_x)) as usize,
                    chroma_stride,
                    (chroma_subblock_size_x).min((width - (x << 1)) >> ss_x) - cxoff,
                    2 >> ss_y,
                );

                copy_area(
                    &luma_grain_block,
                    (luma_offset_y * luma_grain_stride + luma_offset_x + LUMA_SUBBLOCK_SIZE_X)
                        as usize,
                    luma_grain_stride_u,
                    &mut y_col_buf,
                    0,
                    2,
                    2,
                    (LUMA_SUBBLOCK_SIZE_Y + 2).min(height - (y << 1)),
                );
                let ccol_h =
                    (chroma_subblock_size_y + (2 >> ss_y)).min((height - (y << 1)) >> ss_y);
                copy_area(
                    &cb_grain_block,
                    (chroma_offset_y * chroma_grain_stride
                        + chroma_offset_x
                        + chroma_subblock_size_x) as usize,
                    chroma_grain_stride_u,
                    &mut cb_col_buf,
                    0,
                    (2 >> ss_x) as usize,
                    2 >> ss_x,
                    ccol_h,
                );
                copy_area(
                    &cr_grain_block,
                    (chroma_offset_y * chroma_grain_stride
                        + chroma_offset_x
                        + chroma_subblock_size_x) as usize,
                    chroma_grain_stride_u,
                    &mut cr_col_buf,
                    0,
                    (2 >> ss_x) as usize,
                    2 >> ss_x,
                    ccol_h,
                );
            }

            x += LUMA_SUBBLOCK_SIZE_X >> 1;
        }
        y += LUMA_SUBBLOCK_SIZE_Y >> 1;
    }
}

/// Apply film grain to a decoded frame's planes, byte-exact to the C reference
/// `av1_add_film_grain`. Inputs are the CROPPED reconstruction planes (`u16`,
/// tight `d_w`/`cw`-strided); returns the CROPPED grained planes (`u`/`v` empty
/// when `mono`). `bit_depth` / `mc_identity` come from the color config; `ss_x`/
/// `ss_y` are the chroma subsampling (unused for `mono`).
pub fn add_film_grain(
    p: &FilmGrainParams,
    bit_depth: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    mc_identity: bool,
    d_w: usize,
    d_h: usize,
    src_y: &[u16],
    src_u: &[u16],
    src_v: &[u16],
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    // av1_add_film_grain rounds the working dims up to even.
    let width = if d_w % 2 == 1 { d_w + 1 } else { d_w };
    let height = if d_h % 2 == 1 { d_h + 1 } else { d_h };
    let luma_stride = width;

    // For a monochrome image the C treats it as I420 (ss 1,1) for the (unused)
    // chroma template sizing; the luma output is independent of that choice.
    let (css_x, css_y) = if mono { (1, 1) } else { (ss_x, ss_y) };
    let cw = (width >> css_x as usize).max(0);
    let ch = (height >> css_y as usize).max(0);
    let chroma_stride = cw;

    // Source (cropped) chroma dims = decoder's stored chroma dims.
    let src_cw = if mono {
        0
    } else {
        (d_w + ss_x as usize) >> ss_x
    };
    let src_ch = if mono {
        0
    } else {
        (d_h + ss_y as usize) >> ss_y
    };

    // Internal luma plane (even dims), cropped src copied top-left, then extended.
    let mut luma = vec![0u16; luma_stride * height];
    for r in 0..d_h {
        for c in 0..d_w {
            luma[r * luma_stride + c] = src_y[r * d_w + c];
        }
    }
    extend_even(&mut luma, luma_stride, d_w, d_h);

    // Internal chroma planes. Always allocated (the run reads/writes grain
    // templates + col/line buffers even for mono); the PLANE is written only
    // when apply_cb/apply_cr (false for mono).
    let mut cb = vec![0u16; chroma_stride * ch];
    let mut cr = vec![0u16; chroma_stride * ch];
    if !mono {
        for r in 0..src_ch {
            for c in 0..src_cw {
                cb[r * chroma_stride + c] = src_u[r * src_cw + c];
                cr[r * chroma_stride + c] = src_v[r * src_cw + c];
            }
        }
    }

    add_film_grain_run(
        p,
        bit_depth,
        &mut luma,
        &mut cb,
        &mut cr,
        height as i32,
        width as i32,
        luma_stride,
        chroma_stride,
        css_y,
        css_x,
        mc_identity,
    );

    // Crop back to the coded dims.
    let mut out_y = vec![0u16; d_w * d_h];
    for r in 0..d_h {
        for c in 0..d_w {
            out_y[r * d_w + c] = luma[r * luma_stride + c];
        }
    }
    let mut out_u = Vec::new();
    let mut out_v = Vec::new();
    if !mono {
        out_u = vec![0u16; src_cw * src_ch];
        out_v = vec![0u16; src_cw * src_ch];
        for r in 0..src_ch {
            for c in 0..src_cw {
                out_u[r * src_cw + c] = cb[r * chroma_stride + c];
                out_v[r * src_cw + c] = cr[r * chroma_stride + c];
            }
        }
    }

    (out_y, out_u, out_v)
}
