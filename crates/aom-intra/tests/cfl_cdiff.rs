//! Direct C differentials for the CfL kernels — the exported per-size `_c`
//! kernels from cfl.c (reached through the exported `_c` getter tables) vs the
//! aom-intra ports. Upgrades the hand-traced `cfl_vectors.rs` coverage to
//! byte-identity against the real libaom kernels over every CfL-legal tx size
//! (both dims ≤ 32), all three subsample families, and bd 8/10/12.

use aom_intra::cfl::{
    cfl_predict_hbd, subsample_420_hbd, subsample_422_hbd, subsample_444_hbd, subtract_average,
    CFL_BUF_LINE,
};
use aom_sys_ref as c;

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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// (tx_size, w, h) for every tx size with both dims ≤ 32 (the CfL domain —
/// `is_cfl_allowed` caps the luma partition at 32x32, and chroma tx sizes
/// never exceed 32x32).
const CFL_TX: [(usize, usize, usize); 14] = [
    (0, 4, 4),
    (1, 8, 8),
    (2, 16, 16),
    (3, 32, 32),
    (5, 4, 8),
    (6, 8, 4),
    (7, 8, 16),
    (8, 16, 8),
    (9, 16, 32),
    (10, 32, 16),
    (13, 4, 16),
    (14, 16, 4),
    (15, 8, 32),
    (16, 32, 8),
];

#[test]
fn cfl_subsample_hbd_matches_c() {
    let mut rng = Rng(0x1EE7_0AC0_5EED_0001);
    let mut cases = 0u32;
    for &(tx, w, h) in &CFL_TX {
        for &(ss_x, ss_y) in &[(1i32, 1i32), (1, 0), (0, 0)] {
            for &bd in &[8i32, 10, 12] {
                for _ in 0..40 {
                    let stride = w + (rng.below(3) as usize) * 8;
                    let mut input = vec![0u16; stride * h + 16];
                    let maxv = (1u64 << bd) - 1;
                    for p in input.iter_mut() {
                        *p = rng.below(maxv + 1) as u16;
                    }
                    let mut out_c = [0u16; 1024];
                    let mut out_r = [0u16; 1024];
                    c::ref_cfl_subsample_hbd((ss_x, ss_y), tx, &input, stride, &mut out_c);
                    match (ss_x, ss_y) {
                        (1, 1) => subsample_420_hbd(&input, 0, stride, &mut out_r, 0, w, h),
                        (1, 0) => subsample_422_hbd(&input, 0, stride, &mut out_r, 0, w, h),
                        _ => subsample_444_hbd(&input, 0, stride, &mut out_r, 0, w, h),
                    }
                    assert_eq!(
                        out_c, out_r,
                        "subsample ss=({ss_x},{ss_y}) tx={tx} {w}x{h} bd={bd}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 14 * 3 * 3 * 40);
}

#[test]
fn cfl_subtract_average_matches_c() {
    let mut rng = Rng(0x1EE7_0AC0_5EED_0002);
    let mut cases = 0u32;
    for &(tx, w, h) in &CFL_TX {
        for _ in 0..120 {
            // A full random q3 surface: samples outside the w x h window prove
            // the read footprint (they must not affect the sum).
            let mut q3 = [0u16; 1024];
            for p in q3.iter_mut() {
                *p = rng.below(4095 * 8 + 1) as u16; // up to bd-12 pixel * 8
            }
            let mut dst_c = [0i16; 1024];
            let mut dst_r = [0i16; 1024];
            c::ref_cfl_subtract_average(tx, &q3, &mut dst_c);
            subtract_average(&q3, &mut dst_r, w, h);
            assert_eq!(dst_c, dst_r, "subtract_average tx={tx} {w}x{h}");
            cases += 1;
        }
    }
    assert_eq!(cases, 14 * 120);
}

#[test]
fn cfl_predict_hbd_matches_c() {
    let mut rng = Rng(0x1EE7_0AC0_5EED_0003);
    let mut cases = 0u32;
    for &(tx, w, h) in &CFL_TX {
        for &bd in &[8i32, 10, 12] {
            for _ in 0..40 {
                // AC contribution: zero-mean q3 values from subtract_average's
                // range (± max_pixel * 8).
                let span = ((1i64 << bd) * 8) as u64;
                let mut ac = [0i16; 1024];
                for p in ac.iter_mut() {
                    *p = (rng.below(2 * span + 1) as i64 - span as i64) as i16;
                }
                // alpha_q3 from cfl_idx_to_alpha's range: ±(1..=16), 0.
                let alpha_q3 = (rng.below(33) as i32) - 16;
                // dst starts as the DC prediction (in-range pixels).
                let stride = w + (rng.below(2) as usize) * 8;
                let maxv = (1u64 << bd) - 1;
                let mut dst_c = vec![0u16; stride * h];
                for p in dst_c.iter_mut() {
                    *p = rng.below(maxv + 1) as u16;
                }
                let mut dst_r = dst_c.clone();
                c::ref_cfl_predict_hbd(tx, &ac, &mut dst_c, stride, alpha_q3, bd);
                cfl_predict_hbd(&ac, &mut dst_r, 0, stride, alpha_q3, bd, w, h);
                assert_eq!(
                    dst_c, dst_r,
                    "predict tx={tx} {w}x{h} bd={bd} alpha={alpha_q3}"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 14 * 3 * 40);
    let _ = CFL_BUF_LINE;
}
