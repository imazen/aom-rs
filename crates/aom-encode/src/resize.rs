//! Encoder-side frame resize (source downscale) — `av1/common/resize.c`.
//!
//! This is the **non-normative** resize the encoder uses to downscale the
//! source before encoding. It is the encode-side counterpart to the decoder's
//! normative superres *upscale* (`aom_decode::superres`): for superres the
//! source is downscaled horizontally here, coded at the reduced `FrameWidth`,
//! and the decoder upscales it back with `av1_upscale_normative`.
//!
//! For an ALLINTRA (usage=2) KEY still with `--superres-mode`, libaom takes the
//! `DISALLOW_RECODE` path (`encode_without_recode`) and, for the highbd path or
//! any non-1/16-multiple ratio (superres denom 9..15), routes the source scale
//! through `av1_resize_and_extend_frame_nonnormative` → [`resize_plane`] per
//! plane (verified against `reference/libaom`). Superres is horizontal-only, so
//! the vertical pass is `height2 == height` → an exact `memcpy` identity.
//!
//! Every function here is a verbatim port of the corresponding `resize.c`
//! function and is validated byte-for-byte against the exported C symbol
//! `av1_resize_plane` (`tests/resize_plane_diff.rs`).

// ---- constants (resize.c / aom_scale) ----
/// `SCALE_NUMERATOR` — superres/resize denominators are relative to 8.
const SCALE_NUMERATOR: i32 = 8;
const FILTER_BITS: i32 = 7;
const SUBPEL_TAPS: usize = 8;
const RS_SUBPEL_BITS: i32 = 6;
const RS_SUBPEL_MASK: i32 = (1 << RS_SUBPEL_BITS) - 1; // 63
const RS_SCALE_SUBPEL_BITS: i32 = 14;
const RS_SCALE_EXTRA_BITS: i32 = RS_SCALE_SUBPEL_BITS - RS_SUBPEL_BITS; // 8
const RS_SCALE_EXTRA_OFF: i32 = 1 << (RS_SCALE_EXTRA_BITS - 1); // 128

/// libaom `av1_down2_symeven_half_filter` (resize.h).
const DOWN2_SYMEVEN_HALF_FILTER: [i16; 4] = [56, 12, -3, -1];
/// libaom `av1_down2_symodd_half_filter` (resize.h).
const DOWN2_SYMODD_HALF_FILTER: [i16; 4] = [64, 35, 0, -3];

#[inline]
fn clip_pixel(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

#[inline]
fn round_power_of_two(value: i32, n: i32) -> i32 {
    (value + (1 << (n - 1))) >> n
}

/// `choose_interp_filter` (resize.c:219). Selects the 0.5/0.625/0.75/0.875/1.0
/// band interpolation kernel bank from the down/up ratio.
fn choose_interp_filter(in_length: i32, out_length: i32) -> &'static [[i16; SUBPEL_TAPS]; 64] {
    let out_length16 = out_length * 16;
    if out_length16 >= in_length * 16 {
        &FILTER_1000
    } else if out_length16 >= in_length * 13 {
        &FILTER_875
    } else if out_length16 >= in_length * 11 {
        &FILTER_750
    } else if out_length16 >= in_length * 9 {
        &FILTER_625
    } else {
        &FILTER_500
    }
}

/// `interpolate_core` (resize.c:233), `interp_taps == SUBPEL_TAPS == 8`.
fn interpolate_core(
    input: &[u8],
    in_length: i32,
    output: &mut [u8],
    out_length: i32,
    interp_filters: &[[i16; SUBPEL_TAPS]; 64],
) {
    let interp_taps = SUBPEL_TAPS as i32;
    let delta: i32 = ((((in_length as u32) << RS_SCALE_SUBPEL_BITS) + (out_length as u32) / 2)
        / out_length as u32) as i32;
    let offset: i32 = if in_length > out_length {
        (((in_length - out_length) << (RS_SCALE_SUBPEL_BITS - 1)) + out_length / 2) / out_length
    } else {
        -((((out_length - in_length) << (RS_SCALE_SUBPEL_BITS - 1)) + out_length / 2) / out_length)
    };

    let sample = |int_pel: i32, sub_pel: i32, clamp_lo: bool, clamp_hi: bool| -> u8 {
        // `int_pel - interp_taps/2 + 1 + k` (k in 0..taps). Interior samples are
        // provably in [0, in_length-1] via the x1/x2 bounds below; the initial/
        // end/short parts clamp explicitly, matching resize.c.
        let filter = &interp_filters[sub_pel as usize];
        let mut sum: i32 = 0;
        for k in 0..interp_taps {
            let mut pk = int_pel - interp_taps / 2 + 1 + k;
            if clamp_lo && clamp_hi {
                pk = pk.clamp(0, in_length - 1);
            } else if clamp_lo {
                pk = pk.max(0);
            } else if clamp_hi {
                pk = pk.min(in_length - 1);
            }
            sum += filter[k as usize] as i32 * input[pk as usize] as i32;
        }
        clip_pixel(round_power_of_two(sum, FILTER_BITS))
    };

    // x1: first output x where the leftmost tap is >= 0.
    let mut x = 0i32;
    let mut y = offset + RS_SCALE_EXTRA_OFF;
    while (y >> RS_SCALE_SUBPEL_BITS) < (interp_taps / 2 - 1) {
        x += 1;
        y += delta;
    }
    let x1 = x;
    // x2: last output x where the rightmost tap is < in_length.
    x = out_length - 1;
    y = delta * x + offset + RS_SCALE_EXTRA_OFF;
    while (y >> RS_SCALE_SUBPEL_BITS) + interp_taps / 2 >= in_length {
        x -= 1;
        y -= delta;
    }
    let x2 = x;

    let mut optr = 0usize;
    if x1 > x2 {
        x = 0;
        y = offset + RS_SCALE_EXTRA_OFF;
        while x < out_length {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, true, true);
            optr += 1;
            x += 1;
            y += delta;
        }
    } else {
        // Initial part (clamp low).
        x = 0;
        y = offset + RS_SCALE_EXTRA_OFF;
        while x < x1 {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, true, false);
            optr += 1;
            x += 1;
            y += delta;
        }
        // Middle part (no clamp).
        while x <= x2 {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, false, false);
            optr += 1;
            x += 1;
            y += delta;
        }
        // End part (clamp high).
        while x < out_length {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, false, true);
            optr += 1;
            x += 1;
            y += delta;
        }
    }
}

/// `interpolate` (resize.c:315).
fn interpolate(input: &[u8], in_length: i32, output: &mut [u8], out_length: i32) {
    let interp_filters = choose_interp_filter(in_length, out_length);
    interpolate_core(input, in_length, output, out_length, interp_filters);
}

/// `down2_symeven` (resize.c:339). `start_offset` is 0 for all resize_plane
/// callers, kept for fidelity.
fn down2_symeven(input: &[u8], length: i32, output: &mut [u8], start_offset: i32) {
    let filter = &DOWN2_SYMEVEN_HALF_FILTER;
    let filter_len_half = filter.len() as i32; // 4
    let mut l1 = filter_len_half;
    let mut l2 = length - filter_len_half;
    l1 += l1 & 1;
    l2 += l2 & 1;
    let mut optr = 0usize;
    let at = |i: i32| -> i32 { input[i as usize] as i32 };
    if l1 > l2 {
        let mut i = start_offset;
        while i < length {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at((i - j).max(0)) + at((i + 1 + j).min(length - 1)))
                    * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
    } else {
        let mut i = start_offset;
        // Initial part.
        while i < l1 {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at((i - j).max(0)) + at(i + 1 + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
        // Middle part.
        while i < l2 {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at(i - j) + at(i + 1 + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
        // End part.
        while i < length {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at(i - j) + at((i + 1 + j).min(length - 1))) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
    }
}

/// `down2_symodd` (resize.c:394).
fn down2_symodd(input: &[u8], length: i32, output: &mut [u8]) {
    let filter = &DOWN2_SYMODD_HALF_FILTER;
    let filter_len_half = filter.len() as i32; // 4
    let mut l1 = filter_len_half - 1;
    let mut l2 = length - filter_len_half + 1;
    l1 += l1 & 1;
    l2 += l2 & 1;
    let mut optr = 0usize;
    let at = |i: i32| -> i32 { input[i as usize] as i32 };
    if l1 > l2 {
        let mut i = 0;
        while i < length {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                let lo = if i - j < 0 { 0 } else { i - j };
                let hi = if i + j >= length { length - 1 } else { i + j };
                sum += (at(lo) + at(hi)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
    } else {
        let mut i = 0;
        // Initial part.
        while i < l1 {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                let lo = if i - j < 0 { 0 } else { i - j };
                sum += (at(lo) + at(i + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
        // Middle part.
        while i < l2 {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                sum += (at(i - j) + at(i + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
        // End part.
        while i < length {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                let hi = if i + j >= length { length - 1 } else { i + j };
                sum += (at(i - j) + at(hi)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel(sum);
            optr += 1;
            i += 2;
        }
    }
}

/// `get_down2_length` (resize.c:449).
fn get_down2_length(mut length: i32, steps: i32) -> i32 {
    for _ in 0..steps {
        length = (length + 1) >> 1;
    }
    length
}

/// `get_down2_steps` (resize.c:454).
fn get_down2_steps(mut in_length: i32, out_length: i32) -> i32 {
    let mut steps = 0;
    loop {
        let proj = get_down2_length(in_length, 1);
        if proj >= out_length {
            steps += 1;
            in_length = proj;
            if in_length == 1 {
                break;
            }
        } else {
            break;
        }
    }
    steps
}

/// `resize_multistep` (resize.c:470). One dimension: repeated exact halvings
/// (`down2_*`) followed by a final polyphase `interpolate` if the halved length
/// still doesn't match. `length == olength` short-circuits to a copy — this is
/// the superres vertical-pass identity.
fn resize_multistep(input: &[u8], length: i32, output: &mut [u8], olength: i32) {
    if length == olength {
        output[..length as usize].copy_from_slice(&input[..length as usize]);
        return;
    }
    let steps = get_down2_steps(length, olength);
    if steps > 0 {
        let mut filteredlength = length;
        let mut cur: Vec<u8> = input[..length as usize].to_vec();
        for _ in 0..steps {
            let proj = get_down2_length(filteredlength, 1);
            let mut out = vec![0u8; proj as usize];
            if filteredlength & 1 == 1 {
                down2_symodd(&cur, filteredlength, &mut out);
            } else {
                down2_symeven(&cur, filteredlength, &mut out, 0);
            }
            cur = out;
            filteredlength = proj;
        }
        if filteredlength != olength {
            interpolate(&cur, filteredlength, output, olength);
        } else {
            output[..olength as usize].copy_from_slice(&cur[..olength as usize]);
        }
    } else {
        interpolate(input, length, output, olength);
    }
}

/// `av1_calculate_scaled_superres_size` / `calculate_scaled_size_helper`
/// (resize.c:1273/1296): the coded (downscaled) width for a superres
/// denominator. Horizontal only. Unlike the decoder's header-read
/// `coded_frame_width`, the encoder clamps the result to at least
/// `min(16, upscaled_width)` to satisfy the spec's min-16 frame-width
/// constraint (this only differs for tiny widths, e.g. w=16 denom=9 → 16 not
/// 14). `denom == SCALE_NUMERATOR` (8) means no scaling.
pub fn coded_superres_width(upscaled_width: i32, superres_denom: i32) -> i32 {
    if superres_denom == SCALE_NUMERATOR {
        return upscaled_width;
    }
    let min_dim = upscaled_width.min(16);
    let dim = (upscaled_width * SCALE_NUMERATOR + superres_denom / 2) / superres_denom;
    dim.max(min_dim)
}

/// `av1_resize_plane` (resize.c:578). Downscale one plane from
/// `width×height` (row stride `in_stride`) to `width2×height2` (row stride
/// `out_stride`): horizontal `resize_multistep` per row into a `width2×height`
/// scratch, then vertical `resize_multistep` per column into `output`.
///
/// Byte-identical to the exported C `av1_resize_plane` for all in/out sizes
/// (`tests/resize_plane_diff.rs`).
pub fn resize_plane(
    input: &[u8],
    height: i32,
    width: i32,
    in_stride: i32,
    output: &mut [u8],
    height2: i32,
    width2: i32,
    out_stride: i32,
) {
    debug_assert!(width > 0 && height > 0 && width2 > 0 && height2 > 0);
    let mut intbuf = vec![0u8; (width2 * height) as usize];
    for i in 0..height {
        let row_in = &input[(i * in_stride) as usize..];
        let row_out = &mut intbuf[(i * width2) as usize..((i + 1) * width2) as usize];
        resize_multistep(row_in, width, row_out, width2);
    }
    let mut arrbuf = vec![0u8; height as usize];
    let mut arrbuf2 = vec![0u8; height2 as usize];
    for i in 0..width2 {
        for r in 0..height {
            arrbuf[r as usize] = intbuf[(r * width2 + i) as usize];
        }
        resize_multistep(&arrbuf, height, &mut arrbuf2, height2);
        for r in 0..height2 {
            output[(r * out_stride + i) as usize] = arrbuf2[r as usize];
        }
    }
}

// ---- highbd (10/12-bit) variants (resize.c CONFIG_AV1_HIGHBITDEPTH) ----------
// Structurally identical to the 8-bit path above (same filter banks, same delta/
// offset/x1/x2, same down2 half-filters, same resize_multistep control flow);
// they differ ONLY in the pixel type (u16) and the clamp bound (`(1<<bd)-1` via
// clip_pixel_highbd instead of clip_pixel's 255). The bd==8 arm is cross-checked
// against the proven 8-bit `resize_plane` in the differential.

#[inline]
fn clip_pixel_highbd(val: i32, bd: i32) -> u16 {
    val.clamp(0, (1 << bd) - 1) as u16
}

/// `highbd_interpolate_core` (resize.c:679).
fn highbd_interpolate_core(
    input: &[u16],
    in_length: i32,
    output: &mut [u16],
    out_length: i32,
    bd: i32,
    interp_filters: &[[i16; SUBPEL_TAPS]; 64],
) {
    let interp_taps = SUBPEL_TAPS as i32;
    let delta: i32 = ((((in_length as u32) << RS_SCALE_SUBPEL_BITS) + (out_length as u32) / 2)
        / out_length as u32) as i32;
    let offset: i32 = if in_length > out_length {
        (((in_length - out_length) << (RS_SCALE_SUBPEL_BITS - 1)) + out_length / 2) / out_length
    } else {
        -((((out_length - in_length) << (RS_SCALE_SUBPEL_BITS - 1)) + out_length / 2) / out_length)
    };

    let sample = |int_pel: i32, sub_pel: i32, clamp_lo: bool, clamp_hi: bool| -> u16 {
        let filter = &interp_filters[sub_pel as usize];
        let mut sum: i32 = 0;
        for k in 0..interp_taps {
            let mut pk = int_pel - interp_taps / 2 + 1 + k;
            if clamp_lo && clamp_hi {
                pk = pk.clamp(0, in_length - 1);
            } else if clamp_lo {
                pk = pk.max(0);
            } else if clamp_hi {
                pk = pk.min(in_length - 1);
            }
            sum += filter[k as usize] as i32 * input[pk as usize] as i32;
        }
        clip_pixel_highbd(round_power_of_two(sum, FILTER_BITS), bd)
    };

    let mut x = 0i32;
    let mut y = offset + RS_SCALE_EXTRA_OFF;
    while (y >> RS_SCALE_SUBPEL_BITS) < (interp_taps / 2 - 1) {
        x += 1;
        y += delta;
    }
    let x1 = x;
    x = out_length - 1;
    y = delta * x + offset + RS_SCALE_EXTRA_OFF;
    while (y >> RS_SCALE_SUBPEL_BITS) + interp_taps / 2 >= in_length {
        x -= 1;
        y -= delta;
    }
    let x2 = x;

    let mut optr = 0usize;
    if x1 > x2 {
        x = 0;
        y = offset + RS_SCALE_EXTRA_OFF;
        while x < out_length {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, true, true);
            optr += 1;
            x += 1;
            y += delta;
        }
    } else {
        x = 0;
        y = offset + RS_SCALE_EXTRA_OFF;
        while x < x1 {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, true, false);
            optr += 1;
            x += 1;
            y += delta;
        }
        while x <= x2 {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, false, false);
            optr += 1;
            x += 1;
            y += delta;
        }
        while x < out_length {
            let int_pel = y >> RS_SCALE_SUBPEL_BITS;
            let sub_pel = (y >> RS_SCALE_EXTRA_BITS) & RS_SUBPEL_MASK;
            output[optr] = sample(int_pel, sub_pel, false, true);
            optr += 1;
            x += 1;
            y += delta;
        }
    }
}

fn highbd_interpolate(input: &[u16], in_length: i32, output: &mut [u16], out_length: i32, bd: i32) {
    let interp_filters = choose_interp_filter(in_length, out_length);
    highbd_interpolate_core(input, in_length, output, out_length, bd, interp_filters);
}

/// `highbd_down2_symeven` (resize.c:771).
fn highbd_down2_symeven(input: &[u16], length: i32, output: &mut [u16], bd: i32) {
    let filter = &DOWN2_SYMEVEN_HALF_FILTER;
    let filter_len_half = filter.len() as i32;
    let mut l1 = filter_len_half;
    let mut l2 = length - filter_len_half;
    l1 += l1 & 1;
    l2 += l2 & 1;
    let mut optr = 0usize;
    let at = |i: i32| -> i32 { input[i as usize] as i32 };
    if l1 > l2 {
        let mut i = 0;
        while i < length {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at((i - j).max(0)) + at((i + 1 + j).min(length - 1)))
                    * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
    } else {
        let mut i = 0;
        while i < l1 {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at((i - j).max(0)) + at(i + 1 + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
        while i < l2 {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at(i - j) + at(i + 1 + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
        while i < length {
            let mut sum = 1 << (FILTER_BITS - 1);
            for j in 0..filter_len_half {
                sum += (at(i - j) + at((i + 1 + j).min(length - 1))) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
    }
}

/// `highbd_down2_symodd` (resize.c:826).
fn highbd_down2_symodd(input: &[u16], length: i32, output: &mut [u16], bd: i32) {
    let filter = &DOWN2_SYMODD_HALF_FILTER;
    let filter_len_half = filter.len() as i32;
    let mut l1 = filter_len_half - 1;
    let mut l2 = length - filter_len_half + 1;
    l1 += l1 & 1;
    l2 += l2 & 1;
    let mut optr = 0usize;
    let at = |i: i32| -> i32 { input[i as usize] as i32 };
    if l1 > l2 {
        let mut i = 0;
        while i < length {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                let lo = if i - j < 0 { 0 } else { i - j };
                let hi = if i + j >= length { length - 1 } else { i + j };
                sum += (at(lo) + at(hi)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
    } else {
        let mut i = 0;
        while i < l1 {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                let lo = if i - j < 0 { 0 } else { i - j };
                sum += (at(lo) + at(i + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
        while i < l2 {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                sum += (at(i - j) + at(i + j)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
        while i < length {
            let mut sum = (1 << (FILTER_BITS - 1)) + at(i) * filter[0] as i32;
            for j in 1..filter_len_half {
                let hi = if i + j >= length { length - 1 } else { i + j };
                sum += (at(i - j) + at(hi)) * filter[j as usize] as i32;
            }
            sum >>= FILTER_BITS;
            output[optr] = clip_pixel_highbd(sum, bd);
            optr += 1;
            i += 2;
        }
    }
}

/// `highbd_resize_multistep` (resize.c:879).
fn highbd_resize_multistep(input: &[u16], length: i32, output: &mut [u16], olength: i32, bd: i32) {
    if length == olength {
        output[..length as usize].copy_from_slice(&input[..length as usize]);
        return;
    }
    let steps = get_down2_steps(length, olength);
    if steps > 0 {
        let mut filteredlength = length;
        let mut cur: Vec<u16> = input[..length as usize].to_vec();
        for _ in 0..steps {
            let proj = get_down2_length(filteredlength, 1);
            let mut out = vec![0u16; proj as usize];
            if filteredlength & 1 == 1 {
                highbd_down2_symodd(&cur, filteredlength, &mut out, bd);
            } else {
                highbd_down2_symeven(&cur, filteredlength, &mut out, bd);
            }
            cur = out;
            filteredlength = proj;
        }
        if filteredlength != olength {
            highbd_interpolate(&cur, filteredlength, output, olength, bd);
        } else {
            output[..olength as usize].copy_from_slice(&cur[..olength as usize]);
        }
    } else {
        highbd_interpolate(input, length, output, olength, bd);
    }
}

/// `highbd_resize_plane` (resize.c:935) — the bd>8 arm of the encoder source
/// downscale. Byte-identical to the C via the exported
/// `av1_resize_and_extend_frame_nonnormative` (`aom_sys_ref::ref_highbd_resize_plane`).
pub fn highbd_resize_plane(
    input: &[u16],
    height: i32,
    width: i32,
    in_stride: i32,
    output: &mut [u16],
    height2: i32,
    width2: i32,
    out_stride: i32,
    bd: i32,
) {
    debug_assert!(width > 0 && height > 0 && width2 > 0 && height2 > 0);
    let mut intbuf = vec![0u16; (width2 * height) as usize];
    for i in 0..height {
        let row_in = &input[(i * in_stride) as usize..];
        let row_out = &mut intbuf[(i * width2) as usize..((i + 1) * width2) as usize];
        highbd_resize_multistep(row_in, width, row_out, width2, bd);
    }
    let mut arrbuf = vec![0u16; height as usize];
    let mut arrbuf2 = vec![0u16; height2 as usize];
    for i in 0..width2 {
        for r in 0..height {
            arrbuf[r as usize] = intbuf[(r * width2 + i) as usize];
        }
        highbd_resize_multistep(&arrbuf, height, &mut arrbuf2, height2, bd);
        for r in 0..height2 {
            output[(r * out_stride + i) as usize] = arrbuf2[r as usize];
        }
    }
}

pub(crate) static FILTER_500: [[i16; 8]; 64] = [
    [-3, 0, 35, 64, 35, 0, -3, 0],
    [-3, 0, 34, 64, 36, 0, -3, 0],
    [-3, -1, 34, 64, 36, 1, -3, 0],
    [-3, -1, 33, 64, 37, 1, -3, 0],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-3, -1, 31, 64, 39, 1, -3, 0],
    [-3, -1, 31, 63, 39, 2, -3, 0],
    [-2, -2, 30, 63, 40, 2, -3, 0],
    [-2, -2, 29, 63, 41, 2, -3, 0],
    [-2, -2, 29, 63, 41, 3, -4, 0],
    [-2, -2, 28, 63, 42, 3, -4, 0],
    [-2, -2, 27, 63, 43, 3, -4, 0],
    [-2, -3, 27, 63, 43, 4, -4, 0],
    [-2, -3, 26, 62, 44, 5, -4, 0],
    [-2, -3, 25, 62, 45, 5, -4, 0],
    [-2, -3, 25, 62, 45, 5, -4, 0],
    [-2, -3, 24, 62, 46, 5, -4, 0],
    [-2, -3, 23, 61, 47, 6, -4, 0],
    [-2, -3, 23, 61, 47, 6, -4, 0],
    [-2, -3, 22, 61, 48, 7, -4, -1],
    [-2, -3, 21, 60, 49, 7, -4, 0],
    [-1, -4, 20, 60, 49, 8, -4, 0],
    [-1, -4, 20, 60, 50, 8, -4, -1],
    [-1, -4, 19, 59, 51, 9, -4, -1],
    [-1, -4, 19, 59, 51, 9, -4, -1],
    [-1, -4, 18, 58, 52, 10, -4, -1],
    [-1, -4, 17, 58, 52, 11, -4, -1],
    [-1, -4, 16, 58, 53, 11, -4, -1],
    [-1, -4, 16, 57, 53, 12, -4, -1],
    [-1, -4, 15, 57, 54, 12, -4, -1],
    [-1, -4, 15, 56, 54, 13, -4, -1],
    [-1, -4, 14, 56, 55, 13, -4, -1],
    [-1, -4, 14, 55, 55, 14, -4, -1],
    [-1, -4, 13, 55, 56, 14, -4, -1],
    [-1, -4, 13, 54, 56, 15, -4, -1],
    [-1, -4, 12, 54, 57, 15, -4, -1],
    [-1, -4, 12, 53, 57, 16, -4, -1],
    [-1, -4, 11, 53, 58, 16, -4, -1],
    [-1, -4, 11, 52, 58, 17, -4, -1],
    [-1, -4, 10, 52, 58, 18, -4, -1],
    [-1, -4, 9, 51, 59, 19, -4, -1],
    [-1, -4, 9, 51, 59, 19, -4, -1],
    [-1, -4, 8, 50, 60, 20, -4, -1],
    [0, -4, 8, 49, 60, 20, -4, -1],
    [0, -4, 7, 49, 60, 21, -3, -2],
    [-1, -4, 7, 48, 61, 22, -3, -2],
    [0, -4, 6, 47, 61, 23, -3, -2],
    [0, -4, 6, 47, 61, 23, -3, -2],
    [0, -4, 5, 46, 62, 24, -3, -2],
    [0, -4, 5, 45, 62, 25, -3, -2],
    [0, -4, 5, 45, 62, 25, -3, -2],
    [0, -4, 5, 44, 62, 26, -3, -2],
    [0, -4, 4, 43, 63, 27, -3, -2],
    [0, -4, 3, 43, 63, 27, -2, -2],
    [0, -4, 3, 42, 63, 28, -2, -2],
    [0, -4, 3, 41, 63, 29, -2, -2],
    [0, -3, 2, 41, 63, 29, -2, -2],
    [0, -3, 2, 40, 63, 30, -2, -2],
    [0, -3, 2, 39, 63, 31, -1, -3],
    [0, -3, 1, 39, 64, 31, -1, -3],
    [0, -3, 1, 38, 64, 32, -1, -3],
    [0, -3, 1, 37, 64, 33, -1, -3],
    [0, -3, 1, 36, 64, 34, -1, -3],
    [0, -3, 0, 36, 64, 34, 0, -3],
];

pub(crate) static FILTER_625: [[i16; 8]; 64] = [
    [-1, -8, 33, 80, 33, -8, -1, 0],
    [-1, -8, 31, 80, 34, -8, -1, 1],
    [-1, -8, 30, 80, 35, -8, -1, 1],
    [-1, -8, 29, 80, 36, -7, -2, 1],
    [-1, -8, 28, 80, 37, -7, -2, 1],
    [-1, -8, 27, 80, 38, -7, -2, 1],
    [0, -8, 26, 79, 39, -7, -2, 1],
    [0, -8, 25, 79, 40, -7, -2, 1],
    [0, -8, 24, 79, 41, -7, -2, 1],
    [0, -8, 23, 78, 42, -6, -2, 1],
    [0, -8, 22, 78, 43, -6, -2, 1],
    [0, -8, 21, 78, 44, -6, -2, 1],
    [0, -8, 20, 78, 45, -5, -3, 1],
    [0, -8, 19, 77, 47, -5, -3, 1],
    [0, -8, 18, 77, 48, -5, -3, 1],
    [0, -8, 17, 77, 49, -5, -3, 1],
    [0, -8, 16, 76, 50, -4, -3, 1],
    [0, -8, 15, 76, 51, -4, -3, 1],
    [0, -8, 15, 75, 52, -3, -4, 1],
    [0, -7, 14, 74, 53, -3, -4, 1],
    [0, -7, 13, 74, 54, -3, -4, 1],
    [0, -7, 12, 73, 55, -2, -4, 1],
    [0, -7, 11, 73, 56, -2, -4, 1],
    [0, -7, 10, 72, 57, -1, -4, 1],
    [1, -7, 10, 71, 58, -1, -5, 1],
    [0, -7, 9, 71, 59, 0, -5, 1],
    [1, -7, 8, 70, 60, 0, -5, 1],
    [1, -7, 7, 69, 61, 1, -5, 1],
    [1, -6, 6, 68, 62, 1, -5, 1],
    [0, -6, 6, 68, 62, 2, -5, 1],
    [1, -6, 5, 67, 63, 2, -5, 1],
    [1, -6, 5, 66, 64, 3, -6, 1],
    [1, -6, 4, 65, 65, 4, -6, 1],
    [1, -6, 3, 64, 66, 5, -6, 1],
    [1, -5, 2, 63, 67, 5, -6, 1],
    [1, -5, 2, 62, 68, 6, -6, 0],
    [1, -5, 1, 62, 68, 6, -6, 1],
    [1, -5, 1, 61, 69, 7, -7, 1],
    [1, -5, 0, 60, 70, 8, -7, 1],
    [1, -5, 0, 59, 71, 9, -7, 0],
    [1, -5, -1, 58, 71, 10, -7, 1],
    [1, -4, -1, 57, 72, 10, -7, 0],
    [1, -4, -2, 56, 73, 11, -7, 0],
    [1, -4, -2, 55, 73, 12, -7, 0],
    [1, -4, -3, 54, 74, 13, -7, 0],
    [1, -4, -3, 53, 74, 14, -7, 0],
    [1, -4, -3, 52, 75, 15, -8, 0],
    [1, -3, -4, 51, 76, 15, -8, 0],
    [1, -3, -4, 50, 76, 16, -8, 0],
    [1, -3, -5, 49, 77, 17, -8, 0],
    [1, -3, -5, 48, 77, 18, -8, 0],
    [1, -3, -5, 47, 77, 19, -8, 0],
    [1, -3, -5, 45, 78, 20, -8, 0],
    [1, -2, -6, 44, 78, 21, -8, 0],
    [1, -2, -6, 43, 78, 22, -8, 0],
    [1, -2, -6, 42, 78, 23, -8, 0],
    [1, -2, -7, 41, 79, 24, -8, 0],
    [1, -2, -7, 40, 79, 25, -8, 0],
    [1, -2, -7, 39, 79, 26, -8, 0],
    [1, -2, -7, 38, 80, 27, -8, -1],
    [1, -2, -7, 37, 80, 28, -8, -1],
    [1, -2, -7, 36, 80, 29, -8, -1],
    [1, -1, -8, 35, 80, 30, -8, -1],
    [1, -1, -8, 34, 80, 31, -8, -1],
];

pub(crate) static FILTER_750: [[i16; 8]; 64] = [
    [2, -11, 25, 96, 25, -11, 2, 0],
    [2, -11, 24, 96, 26, -11, 2, 0],
    [2, -11, 22, 96, 28, -11, 2, 0],
    [2, -10, 21, 96, 29, -12, 2, 0],
    [2, -10, 19, 96, 31, -12, 2, 0],
    [2, -10, 18, 95, 32, -11, 2, 0],
    [2, -10, 17, 95, 34, -12, 2, 0],
    [2, -9, 15, 95, 35, -12, 2, 0],
    [2, -9, 14, 94, 37, -12, 2, 0],
    [2, -9, 13, 94, 38, -12, 2, 0],
    [2, -8, 12, 93, 40, -12, 1, 0],
    [2, -8, 11, 93, 41, -12, 1, 0],
    [2, -8, 9, 92, 43, -12, 1, 1],
    [2, -8, 8, 92, 44, -12, 1, 1],
    [2, -7, 7, 91, 46, -12, 1, 0],
    [2, -7, 6, 90, 47, -12, 1, 1],
    [2, -7, 5, 90, 49, -12, 1, 0],
    [2, -6, 4, 89, 50, -12, 1, 0],
    [2, -6, 3, 88, 52, -12, 0, 1],
    [2, -6, 2, 87, 54, -12, 0, 1],
    [2, -5, 1, 86, 55, -12, 0, 1],
    [2, -5, 0, 85, 57, -12, 0, 1],
    [2, -5, -1, 84, 58, -11, 0, 1],
    [2, -5, -2, 83, 60, -11, 0, 1],
    [2, -4, -2, 82, 61, -11, -1, 1],
    [1, -4, -3, 81, 63, -10, -1, 1],
    [2, -4, -4, 80, 64, -10, -1, 1],
    [1, -4, -4, 79, 66, -10, -1, 1],
    [1, -3, -5, 77, 67, -9, -1, 1],
    [1, -3, -6, 76, 69, -9, -1, 1],
    [1, -3, -6, 75, 70, -8, -2, 1],
    [1, -2, -7, 74, 71, -8, -2, 1],
    [1, -2, -7, 72, 72, -7, -2, 1],
    [1, -2, -8, 71, 74, -7, -2, 1],
    [1, -2, -8, 70, 75, -6, -3, 1],
    [1, -1, -9, 69, 76, -6, -3, 1],
    [1, -1, -9, 67, 77, -5, -3, 1],
    [1, -1, -10, 66, 79, -4, -4, 1],
    [1, -1, -10, 64, 80, -4, -4, 2],
    [1, -1, -10, 63, 81, -3, -4, 1],
    [1, -1, -11, 61, 82, -2, -4, 2],
    [1, 0, -11, 60, 83, -2, -5, 2],
    [1, 0, -11, 58, 84, -1, -5, 2],
    [1, 0, -12, 57, 85, 0, -5, 2],
    [1, 0, -12, 55, 86, 1, -5, 2],
    [1, 0, -12, 54, 87, 2, -6, 2],
    [1, 0, -12, 52, 88, 3, -6, 2],
    [0, 1, -12, 50, 89, 4, -6, 2],
    [0, 1, -12, 49, 90, 5, -7, 2],
    [1, 1, -12, 47, 90, 6, -7, 2],
    [0, 1, -12, 46, 91, 7, -7, 2],
    [1, 1, -12, 44, 92, 8, -8, 2],
    [1, 1, -12, 43, 92, 9, -8, 2],
    [0, 1, -12, 41, 93, 11, -8, 2],
    [0, 1, -12, 40, 93, 12, -8, 2],
    [0, 2, -12, 38, 94, 13, -9, 2],
    [0, 2, -12, 37, 94, 14, -9, 2],
    [0, 2, -12, 35, 95, 15, -9, 2],
    [0, 2, -12, 34, 95, 17, -10, 2],
    [0, 2, -11, 32, 95, 18, -10, 2],
    [0, 2, -12, 31, 96, 19, -10, 2],
    [0, 2, -12, 29, 96, 21, -10, 2],
    [0, 2, -11, 28, 96, 22, -11, 2],
    [0, 2, -11, 26, 96, 24, -11, 2],
];

pub(crate) static FILTER_875: [[i16; 8]; 64] = [
    [3, -8, 13, 112, 13, -8, 3, 0],
    [2, -7, 12, 112, 15, -8, 3, -1],
    [3, -7, 10, 112, 17, -9, 3, -1],
    [2, -6, 8, 112, 19, -9, 3, -1],
    [2, -6, 7, 112, 21, -10, 3, -1],
    [2, -5, 6, 111, 22, -10, 3, -1],
    [2, -5, 4, 111, 24, -10, 3, -1],
    [2, -4, 3, 110, 26, -11, 3, -1],
    [2, -4, 1, 110, 28, -11, 3, -1],
    [2, -4, 0, 109, 30, -12, 4, -1],
    [1, -3, -1, 108, 32, -12, 4, -1],
    [1, -3, -2, 108, 34, -13, 4, -1],
    [1, -2, -4, 107, 36, -13, 4, -1],
    [1, -2, -5, 106, 38, -13, 4, -1],
    [1, -1, -6, 105, 40, -14, 4, -1],
    [1, -1, -7, 104, 42, -14, 4, -1],
    [1, -1, -7, 103, 44, -15, 4, -1],
    [1, 0, -8, 101, 46, -15, 4, -1],
    [1, 0, -9, 100, 48, -15, 4, -1],
    [1, 0, -10, 99, 50, -15, 4, -1],
    [1, 1, -11, 97, 53, -16, 4, -1],
    [0, 1, -11, 96, 55, -16, 4, -1],
    [0, 1, -12, 95, 57, -16, 4, -1],
    [0, 2, -13, 93, 59, -16, 4, -1],
    [0, 2, -13, 91, 61, -16, 4, -1],
    [0, 2, -14, 90, 63, -16, 4, -1],
    [0, 2, -14, 88, 65, -16, 4, -1],
    [0, 2, -15, 86, 67, -16, 4, 0],
    [0, 3, -15, 84, 69, -17, 4, 0],
    [0, 3, -16, 83, 71, -17, 4, 0],
    [0, 3, -16, 81, 73, -16, 3, 0],
    [0, 3, -16, 79, 75, -16, 3, 0],
    [0, 3, -16, 77, 77, -16, 3, 0],
    [0, 3, -16, 75, 79, -16, 3, 0],
    [0, 3, -16, 73, 81, -16, 3, 0],
    [0, 4, -17, 71, 83, -16, 3, 0],
    [0, 4, -17, 69, 84, -15, 3, 0],
    [0, 4, -16, 67, 86, -15, 2, 0],
    [-1, 4, -16, 65, 88, -14, 2, 0],
    [-1, 4, -16, 63, 90, -14, 2, 0],
    [-1, 4, -16, 61, 91, -13, 2, 0],
    [-1, 4, -16, 59, 93, -13, 2, 0],
    [-1, 4, -16, 57, 95, -12, 1, 0],
    [-1, 4, -16, 55, 96, -11, 1, 0],
    [-1, 4, -16, 53, 97, -11, 1, 1],
    [-1, 4, -15, 50, 99, -10, 0, 1],
    [-1, 4, -15, 48, 100, -9, 0, 1],
    [-1, 4, -15, 46, 101, -8, 0, 1],
    [-1, 4, -15, 44, 103, -7, -1, 1],
    [-1, 4, -14, 42, 104, -7, -1, 1],
    [-1, 4, -14, 40, 105, -6, -1, 1],
    [-1, 4, -13, 38, 106, -5, -2, 1],
    [-1, 4, -13, 36, 107, -4, -2, 1],
    [-1, 4, -13, 34, 108, -2, -3, 1],
    [-1, 4, -12, 32, 108, -1, -3, 1],
    [-1, 4, -12, 30, 109, 0, -4, 2],
    [-1, 3, -11, 28, 110, 1, -4, 2],
    [-1, 3, -11, 26, 110, 3, -4, 2],
    [-1, 3, -10, 24, 111, 4, -5, 2],
    [-1, 3, -10, 22, 111, 6, -5, 2],
    [-1, 3, -10, 21, 112, 7, -6, 2],
    [-1, 3, -9, 19, 112, 8, -6, 2],
    [-1, 3, -9, 17, 112, 10, -7, 3],
    [-1, 3, -8, 15, 112, 12, -7, 2],
];

pub(crate) static FILTER_1000: [[i16; 8]; 64] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];
