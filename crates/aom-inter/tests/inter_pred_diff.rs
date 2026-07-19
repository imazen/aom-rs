//! Differential harness for the decoder single-ref translational inter predictor
//! (chunk 1d) vs the **real exported C** libaom v3.14.1.
//!
//! Two independent byte-identity locks:
//!  1. `facade_matches_c` — the `inter_predictor` facade + convolution
//!     (copy / x-only / y-only / 2-D, single- and dual-filter) vs the real C
//!     `inter_predictor` (`aom_sys_ref::ref_inter_predictor`, wrapping
//!     `av1_convolve_2d_facade`). Sub-pel phases fed directly in `0..16`.
//!  2. `build_mc_border_matches_c` — the out-of-frame reference edge replication
//!     vs the real C `build_mc_border` (`aom_sys_ref::ref_build_mc_border`).
//!
//! Composed, these lock everything downstream of the sub-pel derivation in
//! `build_inter_predictor`. The MV/subsampling -> (subpel, integer offset)
//! derivation itself (`dec_calc_subpel_params`, transcribed faithfully) is
//! validated end-to-end by the decoder frame-MD5 gate (chunk 1f); `smoke_build_
//! inter_predictor` here only checks the wiring (no panics, zero-MV == plain copy,
//! frame-edge MVs stay in range).

use aom_inter::{
    blend_a64_hmask, blend_a64_vmask, build_inter_predictor, build_mc_border, get_obmc_mask,
    inter_predictor,
};
use aom_sys_ref::{
    ref_blend_a64_hmask, ref_blend_a64_vmask, ref_build_mc_border, ref_get_obmc_mask,
    ref_inter_predictor,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// uniform in `0..n`
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
    fn byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    /// uniform in `lo..=hi`
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32) as i32
    }
}

const SIZES: [usize; 4] = [8, 16, 32, 64];

/// (w, h, blk_x, blk_y, mv_row, mv_col, ss_x, ss_y) for a `build_inter_predictor`
/// smoke case.
type EdgeCase = (usize, usize, usize, usize, i32, i32, usize, usize);

#[test]
fn facade_matches_c() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let iters = 40_000;

    // Anti-vacuity: every facade sub-case must actually be exercised.
    let (mut n_copy, mut n_x, mut n_y, mut n_2d, mut n_dual) = (0u32, 0u32, 0u32, 0u32, 0u32);

    for _ in 0..iters {
        let w = SIZES[rng.below(4) as usize];
        let h = SIZES[rng.below(4) as usize];
        let filter_x = rng.below(3) as usize; // 0/1/2
        let filter_y = rng.below(3) as usize;
        let subpel_x = rng.below(16) as usize; // 0..=15
        let subpel_y = rng.below(16) as usize;

        // Bordered source region: interior block top-left at (3,3), >=3 before /
        // >=4 after in each direction (worst case = the 2-D path).
        let stride = w + 7;
        let rows = h + 7;
        let src_off = 3 * stride + 3;
        let src: Vec<u8> = (0..stride * rows).map(|_| rng.byte()).collect();

        let mut dst_port = vec![0u8; w * h];
        inter_predictor(
            &src,
            src_off,
            stride,
            &mut dst_port,
            w,
            w,
            h,
            subpel_x,
            subpel_y,
            filter_x,
            filter_y,
        );
        let dst_c = ref_inter_predictor(
            &src, src_off, stride, w, h, subpel_x, subpel_y, filter_x, filter_y,
        );

        assert_eq!(
            dst_port, dst_c,
            "facade mismatch: w={w} h={h} spx={subpel_x} spy={subpel_y} fx={filter_x} fy={filter_y}"
        );

        match (subpel_x != 0, subpel_y != 0) {
            (false, false) => n_copy += 1,
            (true, false) => n_x += 1,
            (false, true) => n_y += 1,
            (true, true) => {
                n_2d += 1;
                if filter_x != filter_y {
                    n_dual += 1;
                }
            }
        }
    }

    assert!(n_copy > 0, "copy case never exercised");
    assert!(n_x > 0, "x-only case never exercised");
    assert!(n_y > 0, "y-only case never exercised");
    assert!(n_2d > 0, "2-D case never exercised");
    assert!(n_dual > 0, "dual-filter 2-D case never exercised");
}

/// 4-tap facade lock (chunk 2, the 16x16 ratchet): a block side `<= 4` selects
/// libaom's 4-tap kernel per direction (`av1_get_interp_filter_params_with_block_
/// size`; the shim `ref_inter_predictor` passes the real params). Exercises the
/// exact shapes the `av1-1-b8-01-size-16x16` inter frame codes — luma 16x4
/// (8-tap x + 4-tap y) and chroma 8x2 (8-tap x + 4-tap y) — plus w<=4 x-4-tap and
/// mixed cases, vs the real C `inter_predictor`.
#[test]
fn facade_4tap_matches_c() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let iters = 40_000;
    // Sizes include sub-8 sides so >=1 direction takes the 4-tap table.
    const S4: [usize; 4] = [2, 4, 8, 16];

    // Anti-vacuity witnesses for the new 4-tap paths.
    let (mut n_4x, mut n_4y, mut n_mixed, mut n_16x4, mut n_8x2) = (0u32, 0u32, 0u32, 0u32, 0u32);

    for _ in 0..iters {
        let w = S4[rng.below(4) as usize];
        let h = S4[rng.below(4) as usize];
        // Require >= 1 sub-8 side so the 4-tap kernel is actually selected.
        if w > 4 && h > 4 {
            continue;
        }
        let filter_x = rng.below(3) as usize;
        let filter_y = rng.below(3) as usize;
        let subpel_x = rng.below(16) as usize;
        let subpel_y = rng.below(16) as usize;

        let stride = w + 7;
        let rows = h + 7;
        let src_off = 3 * stride + 3;
        let src: Vec<u8> = (0..stride * rows).map(|_| rng.byte()).collect();

        let mut dst_port = vec![0u8; w * h];
        inter_predictor(
            &src,
            src_off,
            stride,
            &mut dst_port,
            w,
            w,
            h,
            subpel_x,
            subpel_y,
            filter_x,
            filter_y,
        );
        let dst_c = ref_inter_predictor(
            &src, src_off, stride, w, h, subpel_x, subpel_y, filter_x, filter_y,
        );
        assert_eq!(
            dst_port, dst_c,
            "4-tap facade mismatch: w={w} h={h} spx={subpel_x} spy={subpel_y} fx={filter_x} fy={filter_y}"
        );

        if w <= 4 && subpel_x != 0 {
            n_4x += 1;
        }
        if h <= 4 && subpel_y != 0 {
            n_4y += 1;
        }
        if (w <= 4) != (h <= 4) && subpel_x != 0 && subpel_y != 0 {
            n_mixed += 1;
        }
        if w == 16 && h == 4 && subpel_x != 0 && subpel_y != 0 {
            n_16x4 += 1;
        }
        if w == 8 && h == 2 && subpel_x != 0 && subpel_y != 0 {
            n_8x2 += 1;
        }
    }

    assert!(n_4x > 0, "4-tap x (w<=4) never exercised");
    assert!(n_4y > 0, "4-tap y (h<=4) never exercised");
    assert!(n_mixed > 0, "mixed 8-tap/4-tap 2-D never exercised");
    assert!(n_16x4 > 0, "the luma 16x4 shape never exercised");
    assert!(n_8x2 > 0, "the chroma 8x2 shape never exercised");
}

#[test]
fn build_mc_border_matches_c() {
    let mut rng = Rng(0xd1b5_4a32_d192_ed03);
    let iters = 40_000;

    // Anti-vacuity: exercise fully-inside, left/right/top/bottom OOB, and corners.
    let (mut n_inside, mut n_left, mut n_right, mut n_top, mut n_bottom) =
        (0u32, 0u32, 0u32, 0u32, 0u32);

    for _ in 0..iters {
        let ref_w = rng.range_i32(1, 40) as usize;
        let ref_h = rng.range_i32(1, 40) as usize;
        let ref_stride = ref_w + rng.below(8) as usize; // >= ref_w, arbitrary slack
        let plane_u8: Vec<u8> = (0..ref_stride * ref_h).map(|_| rng.byte()).collect();
        let plane_u16: Vec<u16> = plane_u8.iter().map(|&v| v as u16).collect();

        let b_w = rng.range_i32(1, 40) as usize;
        let b_h = rng.range_i32(1, 40) as usize;
        // gx/gy span negative (top/left OOB) through past the frame (bottom/right OOB).
        let gx = rng.range_i32(-10, ref_w as i32 + 10);
        let gy = rng.range_i32(-10, ref_h as i32 + 10);

        let mut dst_port = vec![0u8; b_w * b_h];
        build_mc_border(
            &plane_u16,
            ref_stride,
            ref_w,
            ref_h,
            gx,
            gy,
            b_w,
            b_h,
            &mut dst_port,
        );
        let dst_c = ref_build_mc_border(&plane_u8, ref_stride, ref_w, ref_h, gx, gy, b_w, b_h);

        assert_eq!(
            dst_port, dst_c,
            "build_mc_border mismatch: ref={ref_w}x{ref_h} stride={ref_stride} \
             g=({gx},{gy}) b={b_w}x{b_h}"
        );

        if gx < 0 {
            n_left += 1;
        }
        if gx + b_w as i32 > ref_w as i32 {
            n_right += 1;
        }
        if gy < 0 {
            n_top += 1;
        }
        if gy + b_h as i32 > ref_h as i32 {
            n_bottom += 1;
        }
        if gx >= 0 && gy >= 0 && gx + b_w as i32 <= ref_w as i32 && gy + b_h as i32 <= ref_h as i32
        {
            n_inside += 1;
        }
    }

    assert!(n_inside > 0, "fully-inside case never exercised");
    assert!(n_left > 0, "left-OOB case never exercised");
    assert!(n_right > 0, "right-OOB case never exercised");
    assert!(n_top > 0, "top-OOB case never exercised");
    assert!(n_bottom > 0, "bottom-OOB case never exercised");
}

/// Wiring sanity for the public `build_inter_predictor` (the sub-pel derivation is
/// C-locked end-to-end by chunk 1f, not here): a zero-MV interior block must equal a
/// plain copy of the co-located reference block, and frame-edge MVs must not panic
/// and must stay in lowbd range.
#[test]
fn smoke_build_inter_predictor() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let ref_w = 128usize;
    let ref_h = 96usize;
    let ref_stride = ref_w;
    let ref_plane: Vec<u16> = (0..ref_stride * ref_h).map(|_| rng.byte() as u16).collect();

    // (1) Zero MV, interior block, luma (ss 0,0) => exact copy of the ref block.
    let (w, h) = (32usize, 16usize);
    let (blk_x, blk_y) = (40usize, 24usize);
    let dst_stride = w;
    let mut dst = vec![0u16; w * h];
    build_inter_predictor(
        &ref_plane, ref_stride, ref_w, ref_h, &mut dst, 0, dst_stride, blk_x, blk_y, w, h, 0, 0, 0,
        0, 0, 0,
    );
    for y in 0..h {
        for x in 0..w {
            assert_eq!(
                dst[y * dst_stride + x],
                ref_plane[(blk_y + y) * ref_stride + (blk_x + x)],
                "zero-MV copy diverged at ({x},{y})"
            );
        }
    }

    // (2) Frame-edge / off-frame MVs must not panic and stay in range. Includes
    // sub-pel phases and a block whose reference origin goes negative (top-left OOB,
    // driving build_mc_border through the public path) plus a chroma (ss 1,1) block.
    let cases: &[EdgeCase] = &[
        (8, 8, 0, 0, -43, -43, 0, 0),    // off top-left, sub-pel both dirs
        (16, 16, 120, 88, 60, 60, 0, 0), // off bottom-right, full-pel
        (64, 8, 0, 40, 3, -3, 0, 0),     // straddles left edge, sub-pel
        (8, 8, 20, 20, 5, -7, 1, 1),     // chroma 420, sub-pel both dirs
        (32, 32, 96, 64, 13, 21, 0, 0),  // large interior block, sub-pel both dirs
    ];
    for &(w, h, bx, by, mvr, mvc, ssx, ssy) in cases {
        let dst_stride = w;
        let mut dst = vec![0u16; w * h];
        build_inter_predictor(
            &ref_plane, ref_stride, ref_w, ref_h, &mut dst, 0, dst_stride, bx, by, w, h, mvr, mvc,
            ssx, ssy, 0, 0,
        );
        for &v in &dst {
            assert!(v <= 255, "lowbd predictor out of range: {v}");
        }
    }
}

// ===================================================================
// OBMC (chunk 4) — mask table + A64 blend differentials vs real C.
// ===================================================================

/// `av1_get_obmc_mask` (reconinter.c:774): the port table byte-matches C for
/// every legal overlap length.
#[test]
fn obmc_mask_matches_c() {
    for &len in &[1usize, 2, 4, 8, 16, 32, 64] {
        let port = get_obmc_mask(len);
        let c = ref_get_obmc_mask(len);
        assert_eq!(port, &c[..], "av1_get_obmc_mask({len}) diverges");
    }
}

/// `aom_blend_a64_vmask_c` / `aom_blend_a64_hmask_c`: the OBMC A64 blends
/// byte-match C across random pixels + random masks AND the real OBMC feather
/// masks, over the power-of-two shapes OBMC uses (`aom_blend_a64_*mask` asserts
/// `IS_POWER_OF_TWO(w)` / `(h)`). The port operates in-place (`src0 == dst`, as
/// the OBMC caller does); the reference computes it out-of-place from the same
/// `src0`, so agreement locks the identical arithmetic.
#[test]
fn blend_a64_masks_match_c() {
    let mut rng = Rng(0x0b3c_a55e_11d0_7a41);
    // Power-of-two block shapes (bw x bh) OBMC exercises.
    let shapes: &[(usize, usize)] = &[
        (8, 4),
        (4, 8),
        (8, 8),
        (16, 4),
        (4, 16),
        (16, 8),
        (32, 16),
        (16, 32),
        (64, 32),
        (32, 64),
        (4, 4),
        (2, 2),
    ];
    for &(w, h) in shapes {
        for iter in 0..40 {
            let src0: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
            let src1: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();

            // --- vmask (per-row mask, length h) ---
            // On even iterations use the real OBMC feather mask for `h`; else a
            // random 0..=64 mask (A64 alpha range).
            let vmask: Vec<u8> = if iter % 2 == 0 {
                get_obmc_mask(h).to_vec()
            } else {
                (0..h).map(|_| rng.below(65) as u8).collect()
            };
            let mut port_v: Vec<u16> = src0.iter().map(|&b| b as u16).collect();
            let src1_v: Vec<u16> = src1.iter().map(|&b| b as u16).collect();
            blend_a64_vmask(&mut port_v, 0, w, &src1_v, 0, w, &vmask, w, h);
            let c_v = ref_blend_a64_vmask(&src0, &src1, &vmask, w, h);
            for i in 0..w * h {
                assert_eq!(
                    port_v[i], c_v[i] as u16,
                    "vmask diverges at {i} (w={w} h={h} iter={iter})"
                );
            }

            // --- hmask (per-column mask, length w) ---
            let hmask: Vec<u8> = if iter % 2 == 0 {
                get_obmc_mask(w).to_vec()
            } else {
                (0..w).map(|_| rng.below(65) as u8).collect()
            };
            let mut port_h: Vec<u16> = src0.iter().map(|&b| b as u16).collect();
            let src1_h: Vec<u16> = src1.iter().map(|&b| b as u16).collect();
            blend_a64_hmask(&mut port_h, 0, w, &src1_h, 0, w, &hmask, w, h);
            let c_h = ref_blend_a64_hmask(&src0, &src1, &hmask, w, h);
            for i in 0..w * h {
                assert_eq!(
                    port_h[i], c_h[i] as u16,
                    "hmask diverges at {i} (w={w} h={h} iter={iter})"
                );
            }
        }
    }
}
