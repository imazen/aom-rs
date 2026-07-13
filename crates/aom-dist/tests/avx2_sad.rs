//! (1) Lane-level differential: AVX2 SAD == scalar SAD == C, for every
//! multiple-of-16 width. (2) A coarse perf ratio vs C's own (AVX2-dispatched)
//! aom_sad — the performance-gate methodology in miniature.

use aom_dist::avx2::sad_simd;
use aom_dist::sad;
use aom_sys_ref as c;

const SIZES: [(usize, usize, usize); 8] = [
    // (w, h, size_idx in aom-sys-ref SIZES table)
    (16, 16, 9), (16, 32, 10), (32, 32, 14), (32, 64, 15),
    (64, 64, 18), (64, 128, 19), (128, 64, 20), (128, 128, 21),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn u8(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

#[test]
fn avx2_sad_lane_identical() {
    let mut rng = Rng(0x_a2_5ad_1234_5678);
    for &(w, h, idx) in SIZES.iter() {
        let stride = w + 8;
        for _ in 0..5000 {
            let a: Vec<u8> = (0..stride * (h + 1)).map(|_| rng.u8()).collect();
            let b: Vec<u8> = (0..stride * (h + 1)).map(|_| rng.u8()).collect();
            let scalar = sad(&a, stride, &b, stride, w, h);
            let simd = sad_simd(&a, stride, &b, stride, w, h);
            let cref = c::ref_sad(idx, &a, stride, &b, stride);
            assert_eq!(simd, scalar, "avx2 vs scalar {w}x{h}");
            assert_eq!(simd, cref, "avx2 vs C {w}x{h}");
        }
    }
}

// Coarse wall-time ratio vs C. Not a CI gate (single untuned run, no fixed HW
// pinning) — just exercises the perf-gate measurement path and prints a ratio.
#[test]
#[ignore] // run with --ignored; timing is environment-sensitive
fn avx2_sad_perf_ratio() {
    use std::time::Instant;
    let mut rng = Rng(1);
    let (w, h, idx) = (64, 64, 18);
    let stride = w;
    let a: Vec<u8> = (0..stride * h).map(|_| rng.u8()).collect();
    let b: Vec<u8> = (0..stride * h).map(|_| rng.u8()).collect();
    let iters = 2_000_000u32;

    let mut acc = 0u64;
    let t0 = Instant::now();
    for _ in 0..iters {
        acc += sad_simd(&a, stride, &b, stride, w, h) as u64;
    }
    let rust = t0.elapsed().as_secs_f64();

    // Direct AVX2 kernel (no per-call feature detection) — isolates kernel cost.
    let mut accd = 0u64;
    let t0d = Instant::now();
    for _ in 0..iters {
        accd += unsafe { aom_dist::avx2::sad_avx2(&a, stride, &b, stride, w, h) } as u64;
    }
    let rust_direct = t0d.elapsed().as_secs_f64();
    assert_eq!(acc, accd);

    // Like-for-like: C's PRODUCTION dispatch (AVX2), the real perf-gate baseline.
    let t1 = Instant::now();
    let mut acc2 = 0u64;
    for _ in 0..iters {
        acc2 += c::prod_sad(w, &a, stride, &b, stride) as u64;
    }
    let cc = t1.elapsed().as_secs_f64();

    // Also the scalar _c for reference.
    let t2 = Instant::now();
    let mut acc3 = 0u64;
    for _ in 0..iters {
        acc3 += c::ref_sad(idx, &a, stride, &b, stride) as u64;
    }
    let cscalar = t2.elapsed().as_secs_f64();

    assert_eq!(acc, acc2);
    assert_eq!(acc, acc3);
    eprintln!(
        "SAD {w}x{h}: rust-dispatch {rust:.3}s ({:.2}x) | rust-avx2-direct {rust_direct:.3}s ({:.2}x) | C-avx2(prod) {cc:.3}s (1.00x, gate<=1.20x) | C-scalar {cscalar:.3}s",
        rust / cc,
        rust_direct / cc
    );
}
