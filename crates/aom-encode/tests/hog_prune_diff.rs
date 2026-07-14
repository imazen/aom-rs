//! Differential: the HOG intra-mode prune vs the REAL C pieces compiled from
//! the header (hog_shim.c includes intra_mode_search_utils.h — its own static
//! weights/nnconfig and the real lowbd/highbd_generate_hog bodies):
//! - `hog_nn_predict` vs `av1_nn_predict_avx2` (f32-bit equality) AND vs the
//!   RTCD-dispatched `av1_nn_predict` — proving the dispatch resolves to the
//!   AVX2 variant on this machine (the accumulation order the port mirrors);
//! - `generate_hog` vs the real Sobel-histogram statics across depths,
//!   content classes and frame-edge-clipped dims;
//! - `prune_intra_mode_with_hog_y` end-to-end mask equality, thresholds
//!   including the speed-0 `-1.2f`.

use aom_encode::hog::{HOG_BINS, generate_hog, hog_nn_predict, prune_intra_mode_with_hog_y};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    fn f01(&mut self) -> f32 {
        (self.next() % (1 << 20)) as f32 / (1u64 << 20) as f32
    }
}

#[test]
fn hog_nn_predict_matches_avx2_and_dispatch() {
    c::ref_init();
    let mut rng = Rng(0x09a1_a55e_11ea_a7ed);
    let mut nonpos_scores = 0usize;
    let mut pos_scores = 0usize;
    for case in 0..20_000 {
        // Histogram-shaped inputs (normalized non-negative, summing ~1) plus
        // raw-float regimes (negatives, zeros, large) — the kernel math must
        // match on any input.
        let mut hist = [0f32; HOG_BINS];
        match case % 4 {
            0 => {
                let mut total = 0f32;
                for h in hist.iter_mut() {
                    *h = rng.f01();
                    total += *h;
                }
                for h in hist.iter_mut() {
                    *h /= total;
                }
            }
            1 => {
                // One-hot-ish: mass in a few bins (typical directional HOG).
                for _ in 0..3 {
                    hist[(rng.next() % 32) as usize] = rng.f01();
                }
            }
            2 => {
                for h in hist.iter_mut() {
                    *h = (rng.f01() - 0.5) * 8.0;
                }
            }
            _ => {} // all-zero
        }
        for reduce in [false, true] {
            let got = hog_nn_predict(&hist, reduce);
            let want = c::ref_hog_nn_predict(&hist, reduce);
            let disp = c::ref_hog_nn_predict_dispatched(&hist, reduce);
            for i in 0..8 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "avx2 score[{i}] {} vs {} case={case} reduce={reduce}",
                    got[i],
                    want[i],
                );
                assert_eq!(
                    want[i].to_bits(),
                    disp[i].to_bits(),
                    "RTCD dispatch is not the AVX2 variant on this machine \
                     (score[{i}] {} vs {}, case={case})",
                    want[i],
                    disp[i],
                );
                if got[i] <= 0.0 {
                    nonpos_scores += 1;
                } else {
                    pos_scores += 1;
                }
            }
        }
    }
    assert!(
        nonpos_scores > 10_000,
        "non-positive scores: {nonpos_scores}"
    );
    assert!(pos_scores > 10_000, "positive scores: {pos_scores}");
}

/// Fill a rows x cols window with one content class.
#[allow(clippy::too_many_arguments)]
fn fill_content(
    rng: &mut Rng,
    plane: &mut [u16],
    off: usize,
    stride: usize,
    cols: usize,
    rows: usize,
    class: usize,
    bd: u8,
) {
    let maxv = (1i64 << bd) - 1;
    let base = (rng.next() % (1 << bd)) as i64;
    for r in 0..rows {
        for cx in 0..cols {
            let v: i64 = match class {
                0 => base,                                              // flat (all-zero hist)
                1 => (rng.next() % (1 << bd)) as i64,                   // noise
                2 => base + 3 * cx as i64,                              // vertical edges (dy=0)
                3 => base + 3 * r as i64,                               // horizontal edges (dx=0)
                4 => base + 2 * (cx as i64 + r as i64),                 // diagonal
                _ => base + ((cx / 4 + r / 4) % 2) as i64 * (maxv / 2), // checker
            };
            plane[off + r * stride + cx] = v.clamp(0, maxv) as u16;
        }
    }
}

#[test]
fn generate_hog_matches_c() {
    c::ref_init();
    let mut rng = Rng(0x50be_1097_ad1e_0714);
    const STRIDE: usize = 160;
    let mut nonzero_hists = 0usize;
    for case in 0..900 {
        let bd: u8 = [8, 10, 12][case % 3];
        let class = case % 6;
        // rows/cols: full block dims and frame-edge-clipped (non-multiple)
        // values, incl. degenerate 2/3 (interior walk empty -> all-zero hist).
        let dims = [2usize, 3, 4, 6, 8, 12, 16, 30, 32, 64];
        let rows = dims[(rng.next() as usize) % dims.len()];
        let cols = dims[(rng.next() as usize) % dims.len()];
        let off = 8 * STRIDE + 8;
        let mut plane = vec![0u16; STRIDE * 96];
        for v in plane.iter_mut() {
            *v = (rng.next() % (1 << bd)) as u16;
        }
        fill_content(&mut rng, &mut plane, off, STRIDE, cols, rows, class, bd);

        let got = generate_hog(&plane, off, STRIDE, rows, cols);
        let want = c::ref_generate_hog(&plane, off, STRIDE, rows, cols, bd);
        for b in 0..HOG_BINS {
            assert_eq!(
                got[b].to_bits(),
                want[b].to_bits(),
                "hist[{b}] {} vs {} case={case} bd={bd} class={class} {rows}x{cols}",
                got[b],
                want[b],
            );
        }
        if got.iter().any(|&v| v != 0.0) {
            nonzero_hists += 1;
        }
    }
    assert!(nonzero_hists > 400, "nonzero histograms: {nonzero_hists}");
}

#[test]
fn prune_intra_mode_with_hog_matches_c() {
    c::ref_init();
    let mut rng = Rng(0xd09f_00d5_2026_0714);
    const STRIDE: usize = 160;
    let mut some_pruned = 0usize;
    let mut none_pruned = 0usize;
    let mut all_pruned = 0usize;
    let mut clipped_cases = 0usize;
    for case in 0..400 {
        let bd: u8 = [8, 10, 12][case % 3];
        let bsize = [0usize, 3, 4, 6, 9, 12][case % 6];
        let class = (rng.next() as usize) % 6;
        // Speed-0 threshold -1.2 plus sweeps around the score range so the
        // <= boundary and both mask polarities get exercised.
        let th = match case % 4 {
            0 => -1.2f32,
            1 => -6.0,
            2 => 6.0,
            _ => (rng.range(-40, 41) as f32) / 10.0,
        };
        const BLK_W: [usize; 22] = [
            4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
        ];
        const BLK_H: [usize; 22] = [
            4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
        ];
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        // Frame-edge overhang on some cases (1/8-pel negative edges).
        let (right_edge, bottom_edge) = if case % 5 == 4 && bw >= 8 {
            clipped_cases += 1;
            (-(8 * (bw as i32 / 2)), -(8 * (bh as i32 / 4).max(1)))
        } else {
            (1 << 12, 1 << 12)
        };
        let off = 8 * STRIDE + 8;
        let mut plane = vec![0u16; STRIDE * 96];
        for v in plane.iter_mut() {
            *v = (rng.next() % (1 << bd)) as u16;
        }
        fill_content(&mut rng, &mut plane, off, STRIDE, bw, bh, class, bd);

        let mut got = [false; 13];
        prune_intra_mode_with_hog_y(
            &plane,
            off,
            STRIDE,
            bsize,
            right_edge,
            bottom_edge,
            th,
            &mut got,
        );
        let want = c::ref_prune_intra_mode_with_hog_y(
            &plane,
            off,
            STRIDE,
            bsize,
            right_edge,
            bottom_edge,
            bd,
            th,
        );
        assert_eq!(
            got, want,
            "mask case={case} bsize={bsize} bd={bd} class={class} th={th}"
        );
        let n = got.iter().filter(|&&b| b).count();
        if n == 0 {
            none_pruned += 1;
        } else if n == 8 {
            all_pruned += 1;
        } else {
            some_pruned += 1;
        }
    }
    assert!(some_pruned > 60, "partial prunes: {some_pruned}");
    assert!(none_pruned > 20, "no-prune cases: {none_pruned}");
    assert!(all_pruned > 20, "all-pruned cases: {all_pruned}");
    assert!(clipped_cases > 30, "edge-clipped cases: {clipped_cases}");
}
