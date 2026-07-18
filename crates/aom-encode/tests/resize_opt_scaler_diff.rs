//! Differential harness for the optimized 8-bit source scaler
//! (`av1_resize_and_extend_frame`, EIGHTTAP_SMOOTH / phase 8 — the superres
//! denom-16 / exact-1/2-horizontal corner) vs the exported
//! `av1_resize_and_extend_frame_c` (driven over an `aom_extend_frame_borders_c`
//! edge-extended YV12). Covers the horizontal 2:1 downscale at luma (128->64)
//! and chroma (64->32) plane sizes across several content patterns.

use aom_encode::resize::{has_optimized_scaler, optimized_downscale_plane_8bit};
use aom_sys_ref as c;

fn gen_content(kind: u32, w: usize, h: usize) -> Vec<u8> {
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for col in 0..w {
            let v: u32 = match kind {
                0 => ((r * 37 + col * 23) as u32) & 255, // gradient
                1 => {
                    if (col & 3) < 2 { 235 } else { 12 } // vertical stripes (HF)
                }
                2 => (((r ^ col) as u32).wrapping_mul(59)) & 255, // checker-ish
                3 => 40 + (col as u32 * 175 / w as u32),          // smooth horizontal ramp
                _ => ((r.wrapping_mul(101).wrapping_add(col.wrapping_mul(197))) & 255) as u32,
            };
            y[r * w + col] = v as u8;
        }
    }
    y
}

fn check(kind: u32, w: usize, h: usize, dw: usize, dh: usize) {
    assert!(
        has_optimized_scaler(w as i32, h as i32, dw as i32, dh as i32),
        "test dims must hit the optimized scaler"
    );
    let src = gen_content(kind, w, h);
    let port = optimized_downscale_plane_8bit(&src, w, h, dw, dh);
    let src16: Vec<u16> = src.iter().map(|&p| u16::from(p)).collect();
    let refr = c::ref_resize_and_extend_frame_8bit(
        &src16, w as i32, h as i32, w as i32, dw as i32, dh as i32,
    );
    let refr_u8: Vec<u8> = refr.iter().map(|&p| p as u8).collect();
    assert_eq!(
        port.len(),
        dw * dh,
        "kind{kind} {w}x{h}->{dw}x{dh}: port length"
    );
    for i in 0..dw * dh {
        assert_eq!(
            port[i],
            refr_u8[i],
            "kind{kind} {w}x{h}->{dw}x{dh}: pixel {i} (r={},c={}) port {} != ref {}",
            i / dw,
            i % dw,
            port[i],
            refr_u8[i]
        );
    }
}

#[test]
fn optimized_scaler_2to1_horizontal_matches_c() {
    c::ref_init();
    for kind in 0..5u32 {
        // luma 128->64 (denom 16), chroma 64->32, plus a couple other even widths.
        check(kind, 128, 128, 64, 128);
        check(kind, 64, 64, 32, 64);
        check(kind, 96, 128, 48, 128);
        check(kind, 160, 96, 80, 96);
    }
}

/// Partial-block dims: dst width/height NOT a multiple of 16 — the last 16×16
/// block clips to a smaller `work_w`/`work_h`, taking C's `aom_scaled_2d_c`
/// border path. A denom-16 superres frame whose coded width isn't a multiple of
/// 16 (e.g. 100→50) hits this, so it must be bit-exact too.
#[test]
fn optimized_scaler_partial_blocks_match_c() {
    c::ref_init();
    for kind in 0..5u32 {
        check(kind, 100, 100, 50, 100); // dst 50x100: partial cols AND rows
        check(kind, 72, 72, 36, 72); // dst 36x72
        check(kind, 40, 40, 20, 40); // dst 20x40 (single partial col block)
        check(kind, 196, 100, 98, 100); // dst 98x100 (superres 196@denom16-ish)
    }
}
