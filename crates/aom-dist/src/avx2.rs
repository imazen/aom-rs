//! AVX2 SAD specialization. Contract (same as libaom's C-vs-SIMD): it must
//! produce output **identical** to the scalar [`crate::sad`], verified by a
//! lane-level differential test. Widths are multiples of 16.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// AVX2 SAD for `w % 16 == 0`. Returns identical result to scalar `sad`.
///
/// # Safety
/// Caller must ensure AVX2 is available and the buffers are large enough
/// (`(h-1)*stride + w` valid for each plane).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sad_avx2(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    debug_assert!(w % 16 == 0);
    let mut acc = _mm256_setzero_si256(); // 4x u64 partial sums (32-wide chunks)
    let mut acc16 = _mm_setzero_si128(); // 2x u64 partial sums (16-wide chunks)
    for y in 0..h {
        let arow = a.as_ptr().add(y * a_stride);
        let brow = b.as_ptr().add(y * b_stride);
        let mut x = 0;
        while x + 32 <= w {
            let va = _mm256_loadu_si256(arow.add(x) as *const __m256i);
            let vb = _mm256_loadu_si256(brow.add(x) as *const __m256i);
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
            x += 32;
        }
        while x + 16 <= w {
            let va = _mm_loadu_si128(arow.add(x) as *const __m128i);
            let vb = _mm_loadu_si128(brow.add(x) as *const __m128i);
            acc16 = _mm_add_epi64(acc16, _mm_sad_epu8(va, vb));
            x += 16;
        }
    }
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256::<1>(acc);
    let s = _mm_add_epi64(_mm_add_epi64(lo, hi), acc16);
    let l0 = _mm_extract_epi64::<0>(s) as u64;
    let l1 = _mm_extract_epi64::<1>(s) as u64;
    (l0 + l1) as u32
}

/// Dispatch: AVX2 when available and `w % 16 == 0`, else scalar. Result is
/// always identical to [`crate::sad`].
pub fn sad_simd(a: &[u8], a_stride: usize, b: &[u8], b_stride: usize, w: usize, h: usize) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if w % 16 == 0 && is_x86_feature_detected!("avx2") {
            return unsafe { sad_avx2(a, a_stride, b, b_stride, w, h) };
        }
    }
    crate::sad(a, a_stride, b, b_stride, w, h)
}
