//! Differential: the ported highbd (10/12-bit) encoder source downscale
//! [`aom_encode::resize::highbd_resize_plane`] vs libaom's `highbd_resize_plane`
//! (reached through the exported `av1_resize_and_extend_frame_nonnormative`,
//! bound in `aom_sys_ref::ref_highbd_resize_plane`).
//!
//! Two evidence tiers: (1) bd==8 equivalence against the PROVEN 8-bit
//! `resize_plane` (which is itself byte-exact vs exported C) locks the entire
//! u16 control flow + shared math; (2) bd 10/12 against the real C oracle locks
//! the `(1<<bd)-1` clamp bound on out-of-8-bit sums.

use aom_encode::resize::{highbd_resize_plane, resize_plane};
use aom_sys_ref as c;

fn make_plane_u16(width: i32, height: i32, stride: i32, maxval: u16, seed: u64) -> Vec<u16> {
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678);
    let mut buf = vec![0u16; (stride * height) as usize];
    let span = u32::from(maxval) + 1;
    for y in 0..height {
        for x in 0..width {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            buf[(y * stride + x) as usize] = ((state >> 33) as u32 % span) as u16;
        }
    }
    buf
}

fn coded_w(w: i32, denom: i32) -> i32 {
    c::ref_calculate_scaled_superres_width(w, denom)
}

/// bd==8: the highbd path on 0..255 values must reproduce the proven 8-bit
/// `resize_plane` exactly (same math, clamp bound 255 == (1<<8)-1).
#[test]
fn highbd_resize_plane_bd8_equals_8bit() {
    let cases = [
        (64, 64, 32, 32),
        (100, 76, 57, 76),
        (128, 96, 64, 96),
        (196, 196, 128, 196),
        (63, 47, 31, 23),
        (33, 21, 8, 5),
        (17, 9, 9, 9),
    ];
    for &(w, h, w2, h2) in &cases {
        let in_stride = w + 7;
        let src16 = make_plane_u16(w, h, in_stride, 255, (w as u64) << 16 | h as u64);
        let src8: Vec<u8> = src16.iter().map(|&p| p as u8).collect();

        let mut hi = vec![0u16; (w2 * h2) as usize];
        highbd_resize_plane(&src16, h, w, in_stride, &mut hi, h2, w2, w2, 8);

        let mut lo = vec![0u8; (w2 * h2) as usize];
        resize_plane(&src8, h, w, in_stride, &mut lo, h2, w2, w2);
        let lo16: Vec<u16> = lo.iter().map(|&p| p as u16).collect();

        assert_eq!(hi, lo16, "bd8 highbd != 8bit for {w}x{h}->{w2}x{h2}");
    }
}

/// bd 10/12 superres horizontal downscale (height2==height) vs the real C oracle.
#[test]
fn highbd_resize_plane_superres_matches_c() {
    let sizes = [
        (64, 64),
        (100, 76),
        (128, 96),
        (196, 196),
        (256, 144),
        (33, 16),
    ];
    let mut checked = 0;
    for bd in [10, 12] {
        let maxval = ((1u32 << bd) - 1) as u16;
        for &(w, h) in &sizes {
            for denom in 9..=14 {
                let w2 = coded_w(w, denom);
                if w2 == w {
                    continue;
                }
                let in_stride = w + 11;
                let src = make_plane_u16(
                    w,
                    h,
                    in_stride,
                    maxval,
                    (bd as u64) << 32 | (w as u64) << 8 | denom as u64,
                );
                let mut mine = vec![0u16; (w2 * h) as usize];
                highbd_resize_plane(&src, h, w, in_stride, &mut mine, h, w2, w2, bd);
                let refout = c::ref_highbd_resize_plane(&src, h, w, in_stride, h, w2, bd);
                assert_eq!(
                    mine, refout,
                    "bd{bd} superres w={w} h={h} denom={denom} -> w2={w2}"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 40,
        "expected many highbd superres cells, got {checked}"
    );
}

/// bd 10/12 general 2-D downscales (vertical resize active) vs the real C oracle.
#[test]
fn highbd_resize_plane_general_2d_matches_c() {
    let cases = [
        (64, 64, 32, 32),
        (100, 100, 50, 50),
        (96, 72, 57, 43),
        (63, 47, 31, 23),
        (200, 150, 80, 60),
        (17, 17, 9, 9),
    ];
    for bd in [10, 12] {
        let maxval = ((1u32 << bd) - 1) as u16;
        for &(w, h, w2, h2) in &cases {
            let in_stride = w + 5;
            let src = make_plane_u16(
                w,
                h,
                in_stride,
                maxval,
                (bd as u64) << 40 | (w2 as u64) << 8 | h2 as u64,
            );
            let mut mine = vec![0u16; (w2 * h2) as usize];
            highbd_resize_plane(&src, h, w, in_stride, &mut mine, h2, w2, w2, bd);
            let refout = c::ref_highbd_resize_plane(&src, h, w, in_stride, h2, w2, bd);
            assert_eq!(mine, refout, "bd{bd} 2D {w}x{h}->{w2}x{h2}");
        }
    }
}
