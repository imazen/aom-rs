//! Differential gate for byte-exact AV1 film-grain synthesis.
//!
//! Oracle: the REAL exported `av1_add_film_grain` (`av1/decoder/grain_synthesis.c`)
//! via `aom_sys_ref::ref_add_film_grain` (dec_shim.c `shim_add_film_grain` builds
//! two `aom_image_t`s and calls the exported function — NOT a transcription).
//!
//! For each trial we feed IDENTICAL (grain params, reconstruction planes) to the
//! Rust port [`aom_decode::film_grain::add_film_grain`] and to C, and assert the
//! grained output planes are byte-identical. Recon planes are random (maximal AC
//! content, exercising the full scaling-LUT range) bounded to the valid pixel
//! range; grain params are random but spec-valid (strictly-increasing scaling
//! points, in-range AR coeffs / shifts). Anti-vacuous: apply_grain=1, recon has
//! AC content, and grain actually changes a large fraction of pixels.

use aom_entropy::header::FilmGrainParams;
use aom_sys_ref::{FILM_GRAIN_BLOB_LEN, ref_add_film_grain};

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
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        assert!(hi > lo);
        lo + (self.next_u64() % ((hi - lo) as u64)) as i32
    }
    fn boolean(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// Pack a `FilmGrainParams` + bit_depth into the flat i32 blob layout expected
/// by dec_shim.c `fill_grain_params` (the C oracle marshalling channel). The
/// Rust port reads `FilmGrainParams` directly; this blob feeds ONLY the oracle.
fn pack_blob(p: &FilmGrainParams, bit_depth: i32) -> Vec<i32> {
    let mut b = Vec::with_capacity(FILM_GRAIN_BLOB_LEN);
    b.push(p.num_y_points);
    for pt in &p.scaling_points_y {
        b.push(pt[0]);
        b.push(pt[1]);
    }
    b.push(p.num_cb_points);
    for pt in &p.scaling_points_cb {
        b.push(pt[0]);
        b.push(pt[1]);
    }
    b.push(p.num_cr_points);
    for pt in &p.scaling_points_cr {
        b.push(pt[0]);
        b.push(pt[1]);
    }
    b.push(p.scaling_shift);
    b.push(p.ar_coeff_lag);
    b.extend_from_slice(&p.ar_coeffs_y);
    b.extend_from_slice(&p.ar_coeffs_cb);
    b.extend_from_slice(&p.ar_coeffs_cr);
    b.push(p.ar_coeff_shift);
    b.push(p.cb_mult);
    b.push(p.cb_luma_mult);
    b.push(p.cb_offset);
    b.push(p.cr_mult);
    b.push(p.cr_luma_mult);
    b.push(p.cr_offset);
    b.push(p.overlap_flag as i32);
    b.push(p.clip_to_restricted_range as i32);
    b.push(bit_depth);
    b.push(p.chroma_scaling_from_luma as i32);
    b.push(p.grain_scale_shift);
    b.push(p.random_seed);
    assert_eq!(b.len(), FILM_GRAIN_BLOB_LEN);
    b
}

/// Generate `count` strictly-increasing scaling-point x-values in `[0, 255]`
/// with nonzero grain-magnitude y-values, returning `(points, actual_count)`.
fn rand_scaling_points<const N: usize>(rng: &mut Rng, want: i32) -> ([[i32; 2]; N], i32) {
    let mut pts = [[0i32; 2]; N];
    let mut cur = rng.range(0, 20);
    let mut n = 0i32;
    let step_hi = (240 / want).max(2);
    for i in 0..want.min(N as i32) {
        if cur > 255 {
            break;
        }
        pts[i as usize] = [cur, rng.range(16, 200)];
        n += 1;
        cur += rng.range(1, step_hi + 1);
    }
    (pts, n)
}

/// Fill the AR coefficients for a given lag (num_pos entries in `[-128, 127]`).
fn rand_ar(rng: &mut Rng, num_pos: i32, out: &mut [i32]) {
    for c in out.iter_mut().take(num_pos as usize) {
        *c = rng.range(-128, 128);
    }
}

/// A random but spec-valid Y-only grain param set (no chroma points, no cfl).
fn rand_params_y_only(rng: &mut Rng) -> FilmGrainParams {
    let mut p = FilmGrainParams {
        apply_grain: true,
        update_parameters: true,
        random_seed: rng.range(1, 65536),
        scaling_shift: rng.range(8, 12),
        ar_coeff_lag: rng.range(0, 4),
        ar_coeff_shift: rng.range(6, 10),
        grain_scale_shift: rng.range(0, 4),
        overlap_flag: rng.boolean(),
        clip_to_restricted_range: rng.boolean(),
        ..Default::default()
    };
    let want_y = rng.range(1, 15);
    let (ypts, yn) = rand_scaling_points::<14>(rng, want_y);
    p.scaling_points_y = ypts;
    p.num_y_points = yn;
    let num_pos_luma = 2 * p.ar_coeff_lag * (p.ar_coeff_lag + 1);
    rand_ar(rng, num_pos_luma, &mut p.ar_coeffs_y);
    p
}

/// Random reconstruction planes (u16, tight) for the given format, bounded to
/// `[0, (1<<bd)-1]`, with maximal AC content.
fn rand_recon(
    rng: &mut Rng,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    d_w: usize,
    d_h: usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let maxv = (1i32 << bd) - 1;
    let mut y = vec![0u16; d_w * d_h];
    for v in y.iter_mut() {
        *v = rng.range(0, maxv + 1) as u16;
    }
    if mono {
        return (y, Vec::new(), Vec::new());
    }
    let cw = (d_w + ss_x as usize) >> ss_x;
    let ch = (d_h + ss_y as usize) >> ss_y;
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for x in u.iter_mut() {
        *x = rng.range(0, maxv + 1) as u16;
    }
    for x in v.iter_mut() {
        *x = rng.range(0, maxv + 1) as u16;
    }
    (y, u, v)
}

/// Assert the recon has AC content (not a flat plane) — anti-vacuous guard.
fn has_ac(plane: &[u16]) -> bool {
    plane.windows(2).any(|w| w[0] != w[1])
}

/// (mono, ss_x, ss_y) format tuples the decoder envelope covers.
const FORMATS: &[(bool, i32, i32)] = &[
    (true, 1, 1),   // monochrome
    (false, 1, 1),  // 4:2:0
    (false, 0, 0),  // 4:4:4
    (false, 1, 0),  // 4:2:2
];

// A couple of sizes, including odd dims to exercise `extend_even` + edge clamps.
const SIZES: &[(usize, usize)] = &[(64, 64), (96, 64), (66, 34), (34, 66)];

#[test]
fn film_grain_y_only_matches_c() {
    let mut rng = Rng::new(0xF11_6A11);
    let mut total = 0u64;
    let mut changed_trials = 0u64;

    for &bd in &[8i32, 10, 12] {
        for &(mono, ss_x, ss_y) in FORMATS {
            for &(d_w, d_h) in SIZES {
                for _ in 0..12 {
                    let p = rand_params_y_only(&mut rng);
                    let mc_identity = rng.boolean();
                    let (y, u, v) = rand_recon(&mut rng, bd, mono, ss_x, ss_y, d_w, d_h);
                    assert!(has_ac(&y), "recon must have AC content");
                    let blob = pack_blob(&p, bd);

                    let (cy, cu, cv) = ref_add_film_grain(
                        &blob, bd, mono, ss_x, ss_y, mc_identity, d_w, d_h, &y, &u, &v,
                    );
                    let (ry, ru, rv) = aom_decode::film_grain::add_film_grain(
                        &p, bd, mono, ss_x, ss_y, mc_identity, d_w, d_h, &y, &u, &v,
                    );

                    assert_eq!(
                        ry, cy,
                        "Y plane mismatch bd={bd} mono={mono} ss=({ss_x},{ss_y}) \
                         size={d_w}x{d_h} seed={} clip={} overlap={}",
                        p.random_seed, p.clip_to_restricted_range, p.overlap_flag
                    );
                    assert_eq!(ru, cu, "U plane mismatch bd={bd} size={d_w}x{d_h}");
                    assert_eq!(rv, cv, "V plane mismatch bd={bd} size={d_w}x{d_h}");

                    // Anti-vacuous: Y-grain (num_y_points>0) must change pixels.
                    if ry != y {
                        changed_trials += 1;
                    }
                    total += 1;
                }
            }
        }
    }

    // Grain must actually alter output on the vast majority of trials (Y grain
    // is applied wherever the scaling LUT is nonzero, which is everywhere here).
    assert!(
        changed_trials * 10 >= total * 9,
        "film grain changed pixels in only {changed_trials}/{total} trials — \
         suspiciously vacuous"
    );
    assert!(total >= 500, "expected a broad sweep, got {total} trials");
    eprintln!("film_grain_y_only: {total} trials byte-identical, {changed_trials} altered pixels");
}
