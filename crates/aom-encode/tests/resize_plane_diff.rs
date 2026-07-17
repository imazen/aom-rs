//! Differential: the ported non-normative encoder resize
//! [`aom_encode::resize::resize_plane`] vs the exported libaom C symbol
//! `av1_resize_plane` (bound in `aom_sys_ref::ref_resize_plane`). This is the
//! source downscale the encoder applies before a superres KEY encode.
//!
//! Best evidence tier: the oracle is the REAL exported C function, so a
//! misreading of the resize.c control flow shows up as a byte mismatch.

use aom_encode::resize::resize_plane;
use aom_sys_ref as c;

/// Deterministic per-pixel content (LCG); varied per (w,h,seed) so every cell
/// exercises different residual/edge statistics.
fn make_plane(width: i32, height: i32, stride: i32, seed: u64) -> Vec<u8> {
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let mut buf = vec![0u8; (stride * height) as usize];
    for y in 0..height {
        for x in 0..width {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            buf[(y * stride + x) as usize] = (state >> 33) as u8;
        }
    }
    buf
}

fn coded_w(w: i32, denom: i32) -> i32 {
    c::ref_calculate_scaled_superres_width(w, denom)
}

/// The superres source downscale is horizontal-only (`height2 == height`), so
/// the vertical `resize_multistep` short-circuits to an identity copy while the
/// horizontal pass runs `interpolate` (denom 9..15) or a `down2` halving.
#[test]
fn resize_plane_superres_horizontal_matches_c() {
    let sizes = [
        (64, 64),
        (65, 48),
        (100, 76),
        (128, 96),
        (196, 196),
        (256, 144),
        (352, 288),
        (17, 9),
        (33, 16),
        (48, 48),
        (129, 65),
    ];
    let mut checked = 0;
    for &(w, h) in &sizes {
        for denom in 9..=16 {
            let w2 = coded_w(w, denom);
            if w2 == w {
                continue; // SCALE_NUMERATOR / no scaling
            }
            let h2 = h; // horizontal only
            let in_stride = w + 13;
            let src = make_plane(
                w,
                h,
                in_stride,
                (w as u64) << 20 | (h as u64) << 8 | denom as u64,
            );
            let mut mine = vec![0u8; (w2 * h2) as usize];
            resize_plane(&src, h, w, in_stride, &mut mine, h2, w2, w2);
            let refout = c::ref_resize_plane(&src, h, w, in_stride, h2, w2);
            assert_eq!(
                mine, refout,
                "superres downscale mismatch: w={w} h={h} denom={denom} -> w2={w2}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 60, "expected many superres cells, got {checked}");
}

/// General 2-D downscales — exercises the vertical resize path, multi-step
/// halvings (`down2_symeven`/`down2_symodd`), the short-input branch, and every
/// `choose_interp_filter` band.
#[test]
fn resize_plane_general_2d_matches_c() {
    let cases = [
        (64, 64, 32, 32),   // exact halving both dims (down2_symeven)
        (100, 100, 50, 50), // even both
        (96, 72, 57, 43),   // 0.6 band interpolate both
        (128, 128, 64, 48), // halving w, 0.75 h
        (63, 47, 31, 23),   // odd dims (down2_symodd on the halving)
        (200, 150, 80, 60), // 0.4 band (< 0.5)
        (50, 50, 49, 49),   // near-1.0 band (interpolate only, no halving)
        (33, 21, 8, 5),     // multi-step: two halvings then interpolate
        (17, 17, 9, 9),     // tiny odd
        (12, 8, 3, 2),      // very short input (short branch in down2)
    ];
    for &(w, h, w2, h2) in &cases {
        let in_stride = w + 5;
        let src = make_plane(
            w,
            h,
            in_stride,
            (w as u64) << 24 | (w2 as u64) << 8 | h2 as u64,
        );
        let mut mine = vec![0u8; (w2 * h2) as usize];
        resize_plane(&src, h, w, in_stride, &mut mine, h2, w2, w2);
        let refout = c::ref_resize_plane(&src, h, w, in_stride, h2, w2);
        assert_eq!(mine, refout, "2D resize mismatch: {w}x{h} -> {w2}x{h2}");
    }
}

/// The port must honor a destination stride wider than `width2` (real YV12
/// buffers carry a border). Compare the valid region against the tight oracle.
#[test]
fn resize_plane_strided_output_matches_c() {
    for &(w, h, denom) in &[(128, 96, 11), (196, 196, 13), (256, 144, 9)] {
        let w2 = coded_w(w, denom);
        let h2 = h;
        let in_stride = w + 8;
        let out_stride = w2 + 9;
        let src = make_plane(w, h, in_stride, 0xABCD ^ denom as u64);
        let mut mine = vec![0u8; (out_stride * h2) as usize];
        resize_plane(&src, h, w, in_stride, &mut mine, h2, w2, out_stride);
        let refout = c::ref_resize_plane(&src, h, w, in_stride, h2, w2);
        for r in 0..h2 {
            let mrow = &mine[(r * out_stride) as usize..(r * out_stride + w2) as usize];
            let rrow = &refout[(r * w2) as usize..(r * w2 + w2) as usize];
            assert_eq!(mrow, rrow, "strided-out row {r} mismatch (denom={denom})");
        }
    }
}

/// Cross-lock the encoder coded-width function used for wiring against the
/// exported C `av1_calculate_scaled_superres_size` (includes the min-16 clamp).
#[test]
fn coded_superres_width_matches_c() {
    for w in [
        16, 17, 18, 20, 24, 63, 64, 96, 100, 128, 196, 255, 256, 352, 1920,
    ] {
        for denom in 8..=16 {
            let ours = aom_encode::resize::coded_superres_width(w, denom);
            let c_w = c::ref_calculate_scaled_superres_width(w, denom);
            assert_eq!(ours, c_w, "coded width mismatch: w={w} denom={denom}");
        }
    }
}

/// The decoder's header-read `coded_frame_width` (no clamp) equals the encoder's
/// clamped width in the regime real superres cells live in (the unclamped
/// result already >= min(16, w)); they diverge only for tiny widths.
#[test]
fn decode_coded_width_agrees_where_unclamped() {
    for w in [63, 64, 96, 100, 128, 196, 256, 352, 1920] {
        for denom in 9..=16 {
            let dec = aom_decode::superres::coded_frame_width(w, denom);
            let enc = aom_encode::resize::coded_superres_width(w, denom);
            assert_eq!(dec, enc, "w={w} denom={denom}: decode/encode width differ");
        }
    }
}
