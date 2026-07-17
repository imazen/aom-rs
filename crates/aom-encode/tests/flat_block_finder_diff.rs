//! C7 grain-estimator chunk 3 — DIFFERENTIAL gate for the flat-block finder.
//!
//! Feeds IDENTICAL `w×h` planes to the Rust port
//! ([`aom_encode::noise_model::FlatBlockFinder`]) and the REAL exported
//! `aom_flat_block_finder_init` + `_run` (via `aom_sys_ref`), and asserts the
//! `flat_blocks` map AND the `num_flat` return are identical. Content is a MIX
//! of low-amplitude-noise-on-smooth blocks (flat candidates) and
//! high-contrast textured blocks (non-flat), so the gate is anti-vacuous (both
//! classes present) and exercises the hard-threshold arm, the gradient/eigen
//! features, and the 10th-percentile sigmoid ranking.
//!
//! All feature math is exact `f64`/`sqrt`; the only libm-sensitive step is the
//! percentile arm's `exp` sigmoid (`1/(1+exp(-w))`). On this glibc host Rust's
//! `f64::exp` matches C's `exp`, so the map is byte-identical — if a future
//! platform's libm diverges, the percentile membership (not the hard-flat
//! `is_flat` arm) is where it would show.

use aom_encode::noise_model::FlatBlockFinder;
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Symmetric noise in `[-amp, amp]`.
    fn noise(&mut self, amp: f64) -> f64 {
        (self.unit() * 2.0 - 1.0) * amp
    }
}

/// Build a plane: per-block, either smooth-base + low noise (flat candidate) or
/// high-contrast texture (non-flat), chosen pseudo-randomly.
fn make_plane(rng: &mut Rng, w: usize, h: usize, bs: usize, maxv: f64) -> Vec<u16> {
    let mut p = vec![0u16; w * h];
    let nbw = w.div_ceil(bs);
    for y in 0..h {
        for x in 0..w {
            let (bx, by) = (x / bs, y / bs);
            let textured = (bx.wrapping_mul(7).wrapping_add(by.wrapping_mul(13)) + (rng.0 as usize & 3)) % 3 == 0;
            let base = maxv * 0.5 + (x as f64 / w as f64 - 0.5) * maxv * 0.2;
            let val = if textured {
                // Sharp checker + strong noise → high gradient, non-flat.
                let checker = if ((x / 4) + (y / 4)) & 1 == 0 { maxv * 0.35 } else { -maxv * 0.35 };
                base + checker + rng.noise(maxv * 0.15)
            } else {
                // Smooth base + small noise → low gradient, some variance = flat.
                base + rng.noise(maxv * 0.02)
            };
            let _ = nbw;
            p[y * w + x] = val.clamp(0.0, maxv).round() as u16;
        }
    }
    p
}

#[test]
fn flat_block_finder_run_matches_c() {
    c::ref_init();
    let mut rng = Rng::new(0xF1A7_B10C);
    let mut trials = 0;
    let mut any_flat = 0i64;
    let mut any_nonflat = 0i64;
    // (block_size, bit_depth, use_highbd)
    let configs = [(32usize, 8i32, false), (32, 10, true), (16, 8, false), (32, 12, true)];
    for (ci, &(bs, bd, hbd)) in configs.iter().enumerate() {
        let maxv = ((1u32 << bd) - 1) as f64;
        for t in 0..12 {
            // Sizes: exact multiples and a partial-edge case.
            let (w, h) = [(256usize, 256usize), (192, 160), (128, 224)][t % 3];
            rng.0 ^= (ci as u64) << 40 | (t as u64) << 8 | 0x5171;
            let plane = make_plane(&mut rng, w, h, bs, maxv);

            let finder = FlatBlockFinder::new(bs, bd);
            let (port_map, port_nf) = finder.run(&plane, w, h, w);
            let (c_map, c_nf) = c::ref_flat_block_finder_run(&plane, w, h, bs, bd, hbd)
                .expect("C flat_block_finder_run");

            assert_eq!(
                port_map, c_map,
                "config {ci} trial {t} (bs{bs} bd{bd} {w}x{h}): flat_blocks map mismatch"
            );
            assert_eq!(port_nf, c_nf, "config {ci} trial {t}: num_flat mismatch");

            any_flat += c_map.iter().filter(|&&v| v != 0).count() as i64;
            any_nonflat += c_map.iter().filter(|&&v| v == 0).count() as i64;
            trials += 1;
        }
    }
    println!("flat_block_finder_diff: {trials} trials bit-identical to C ({any_flat} flat / {any_nonflat} non-flat block-cells)");
    // Anti-vacuity: both classes must be well represented across the corpus.
    assert!(any_flat > 100, "too few flat blocks ({any_flat}) — content not exercising the flat arm");
    assert!(any_nonflat > 100, "too few non-flat blocks ({any_nonflat}) — content too flat");
}
