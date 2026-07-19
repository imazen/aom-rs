//! Safe SIMD SAD via archmage `#[autoversion]` — no `unsafe`, no raw
//! `core::arch` intrinsics. `#[autoversion]` compiles one `#[target_feature]`-
//! gated variant per tier (AVX-512/AVX2/NEON/WASM/scalar) — unlocking LLVM's
//! auto-vectorizer to lower the sum-of-abs-diff loop to `psadbw` / `uabd` — plus
//! a runtime dispatcher. Result is byte-identical to scalar [`crate::dist::sad`].
//!
//! NOTE on dispatch: the generated `sad_simd` dispatcher pays a small
//! feature-check per call. In an encoder that is amortized by placing the
//! dispatch at the motion-search-loop entry (an `#[arcane]` boundary that calls
//! the SAD kernel per candidate); that entry point does not exist in this
//! kernel-only crate yet, so a per-block microbenchmark of the dispatcher is
//! dispatch-bound, not kernel-bound.

use archmage::autoversion;

/// Sum of absolute differences over a `w x h` block. Byte-identical to
/// [`crate::dist::sad`]; auto-vectorized, picks the best SIMD tier at runtime.
#[autoversion]
pub fn sad_simd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let arow = &a[y * a_stride..y * a_stride + w];
        let brow = &b[y * b_stride..y * b_stride + w];
        for x in 0..w {
            sum += (arow[x] as i32 - brow[x] as i32).unsigned_abs();
        }
    }
    sum
}
