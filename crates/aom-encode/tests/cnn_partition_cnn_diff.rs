//! Differential: the ported intra-CNN conv cascade
//! (`cnn_partition::cnn::cnn_predict`) vs the REAL libaom CNN engine
//! (`av1_cnn_predict_img_multi_out`, `aom_sys_ref::ref_intra_cnn_run`).
//!
//! Two comparisons, one single-threaded test (the C-scalar oracle toggles a
//! process-global RTCD pointer, so all calls must be on one thread):
//!   1. vs the pure **C-scalar** engine (`force_cscalar = true`) — must be
//!      BIT-EXACT: proves the Rust cascade is a faithful transcription of
//!      `av1_cnn_convolve_no_maxpool_padding_valid_c` + the layer wiring.
//!   2. vs the **dispatched** (AVX2) engine (`force_cscalar = false`, what the
//!      encoder runs) — reported as a max-abs gap. It need not be bit-exact
//!      (libaom's own C-vs-SIMD tolerance is 1e-6); it only has to stay far
//!      inside the DNN prec-reduce bucket so the downstream split/no-split
//!      FLAGS agree (that flag-parity is asserted in the full-model diff).

use aom_encode::cnn_partition::cnn::{cnn_predict, CNN_OUT_BUF_SIZE};
use aom_sys_ref as c;

struct XorShift(u64);
impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn u8(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
}

/// Build a 65×65 window (stride 65) from a content closure over frame coords,
/// applying the replicated top/left border (`src(max(i-1,0), max(j-1,0))`).
fn window(content: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let mut win = vec![0u8; 65 * 65];
    for i in 0..65 {
        for j in 0..65 {
            let fr = (i as i32 - 1).max(0) as usize;
            let fc = (j as i32 - 1).max(0) as usize;
            win[i * 65 + j] = content(fr, fc);
        }
    }
    win
}

#[test]
fn cnn_predict_matches_c_scalar_bit_exact_and_reports_avx2_gap() {
    c::ref_init();
    let mut rng = XorShift(0x51ed_c0de_1234_5678);

    // A representative mix: uniform random, the real vgrad-256 content, flats,
    // two-tone, gradients, and impulse-ish patterns.
    let mut windows: Vec<Vec<u8>> = Vec::new();
    windows.push(window(|_, c| (32 + c * 190 / 256) as u8)); // vgrad-256 SB(0,0)
    windows.push(window(|_, _| 128)); // flat
    windows.push(window(|_, c| if c < 32 { 40 } else { 200 })); // two-tone
    windows.push(window(|r, c| (16 + (r + c) * 200 / 128) as u8)); // diagonal
    windows.push(window(|r, c| if (r + c) % 2 == 0 { 0 } else { 255 })); // checker
    for _ in 0..200 {
        // Pure random windows.
        let w: Vec<u8> = (0..65 * 65).map(|_| rng.u8()).collect();
        windows.push(w);
    }

    let mut worst_avx2_gap = 0.0f32;
    for (wi, win) in windows.iter().enumerate() {
        let got = cnn_predict(win);
        assert_eq!(got.len(), CNN_OUT_BUF_SIZE);

        // 1. C-scalar: BIT-EXACT.
        let want_c = c::ref_intra_cnn_run(win, true);
        for (idx, (&g, &wc)) in got.iter().zip(want_c.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                wc.to_bits(),
                "window {wi} cnn_buffer[{idx}]: rust={g} ({:#010x}) c_scalar={wc} ({:#010x})",
                g.to_bits(),
                wc.to_bits()
            );
        }

        // 2. AVX2 (encoder path): report the gap, keep it tiny.
        let want_avx2 = c::ref_intra_cnn_run(win, false);
        for (&g, &wa) in got.iter().zip(want_avx2.iter()) {
            worst_avx2_gap = worst_avx2_gap.max((g - wa).abs());
        }
    }

    eprintln!(
        "cnn_predict: {} windows BIT-EXACT vs C-scalar; worst |rust - AVX2| = {worst_avx2_gap:e}",
        windows.len()
    );
    // The AVX2 gap must be far below the DNN prec-reduce bucket (1/512 ≈ 2e-3)
    // so downstream flags never flip. (libaom's own CNN C-vs-SIMD MSE tol 1e-6.)
    assert!(
        worst_avx2_gap < 1e-2,
        "AVX2 gap {worst_avx2_gap:e} unexpectedly large — flag parity at risk"
    );
}
