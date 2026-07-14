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

/// A random spec-valid grain param set WITH chroma grain. `cfl` selects
/// chroma-scaling-from-luma (num_cb/cr_points stay 0, chroma AR coeffs set,
/// multipliers forced by the synthesis); otherwise independent cb/cr scaling
/// points + multipliers. `num_y_points > 0` so chroma is present in every
/// format (`chroma_absent` gates chroma off for 4:2:0 when num_y_points==0).
fn rand_params_chroma(rng: &mut Rng, cfl: bool) -> FilmGrainParams {
    let mut p = rand_params_y_only(rng); // gives num_y_points >= 1
    let num_pos_luma = 2 * p.ar_coeff_lag * (p.ar_coeff_lag + 1);
    let num_pos_chroma = num_pos_luma + 1; // num_y_points > 0 -> cfl luma-avg pos
    if cfl {
        p.chroma_scaling_from_luma = true;
        p.num_cb_points = 0;
        p.num_cr_points = 0;
        rand_ar(rng, num_pos_chroma, &mut p.ar_coeffs_cb);
        rand_ar(rng, num_pos_chroma, &mut p.ar_coeffs_cr);
    } else {
        p.chroma_scaling_from_luma = false;
        let want_cb = rng.range(1, 11);
        let (cbpts, cbn) = rand_scaling_points::<10>(rng, want_cb);
        p.scaling_points_cb = cbpts;
        p.num_cb_points = cbn;
        let want_cr = rng.range(1, 11);
        let (crpts, crn) = rand_scaling_points::<10>(rng, want_cr);
        p.scaling_points_cr = crpts;
        p.num_cr_points = crn;
        rand_ar(rng, num_pos_chroma, &mut p.ar_coeffs_cb);
        rand_ar(rng, num_pos_chroma, &mut p.ar_coeffs_cr);
        p.cb_mult = rng.range(0, 256);
        p.cb_luma_mult = rng.range(0, 256);
        p.cb_offset = rng.range(0, 512);
        p.cr_mult = rng.range(0, 256);
        p.cr_luma_mult = rng.range(0, 256);
        p.cr_offset = rng.range(0, 512);
    }
    p
}

/// Shared driver for the chroma sweeps (non-mono formats only).
fn chroma_sweep(seed: u64, cfl: bool) {
    let mut rng = Rng::new(seed);
    let formats: &[(bool, i32, i32)] = &[(false, 1, 1), (false, 0, 0), (false, 1, 0)];
    let mut total = 0u64;
    let mut chroma_changed = 0u64;

    for &bd in &[8i32, 10, 12] {
        for &(mono, ss_x, ss_y) in formats {
            for &(d_w, d_h) in SIZES {
                for _ in 0..12 {
                    let p = rand_params_chroma(&mut rng, cfl);
                    let mc_identity = rng.boolean();
                    let (y, u, v) = rand_recon(&mut rng, bd, mono, ss_x, ss_y, d_w, d_h);
                    assert!(has_ac(&u) && has_ac(&v), "recon chroma must have AC content");
                    let blob = pack_blob(&p, bd);

                    let (cy, cu, cv) = ref_add_film_grain(
                        &blob, bd, mono, ss_x, ss_y, mc_identity, d_w, d_h, &y, &u, &v,
                    );
                    let (ry, ru, rv) = aom_decode::film_grain::add_film_grain(
                        &p, bd, mono, ss_x, ss_y, mc_identity, d_w, d_h, &y, &u, &v,
                    );

                    assert_eq!(
                        ry, cy,
                        "Y mismatch cfl={cfl} bd={bd} ss=({ss_x},{ss_y}) size={d_w}x{d_h} \
                         seed={} clip={} overlap={}",
                        p.random_seed, p.clip_to_restricted_range, p.overlap_flag
                    );
                    assert_eq!(
                        ru, cu,
                        "U mismatch cfl={cfl} bd={bd} ss=({ss_x},{ss_y}) size={d_w}x{d_h} \
                         seed={} num_cb={} clip={} overlap={}",
                        p.random_seed, p.num_cb_points, p.clip_to_restricted_range, p.overlap_flag
                    );
                    assert_eq!(
                        rv, cv,
                        "V mismatch cfl={cfl} bd={bd} ss=({ss_x},{ss_y}) size={d_w}x{d_h} \
                         seed={} num_cr={}",
                        p.random_seed, p.num_cr_points
                    );

                    if ru != u || rv != v {
                        chroma_changed += 1;
                    }
                    total += 1;
                }
            }
        }
    }

    assert!(
        chroma_changed * 10 >= total * 7,
        "chroma grain changed pixels in only {chroma_changed}/{total} trials (cfl={cfl})"
    );
    assert!(total >= 400, "expected a broad chroma sweep, got {total} trials");
    eprintln!(
        "film_grain chroma cfl={cfl}: {total} trials byte-identical, {chroma_changed} chroma-altered"
    );
}

#[test]
fn film_grain_chroma_matches_c() {
    chroma_sweep(0xC420_A11, false);
}

#[test]
fn film_grain_cfl_matches_c() {
    chroma_sweep(0x0CF1_A11, true);
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

// ---------------------------------------------------------------------------
// END-TO-END gate: REAL film-grain streams decode byte-identical to the C
// decoder WITH grain applied.
//
// Encode a KEY frame with AV1E_SET_FILM_GRAIN_TEST_VECTOR (built-in grain param
// sets from libaom's grain_test_vectors.h), then compare `decode_frame_obus`
// (grain synthesized post-reconstruction) BYTE-IDENTICALLY against the REAL C
// decoder aom_codec_av1_dx (which applies grain by default in
// aom_codec_get_frame). This proves the whole chain: seq/frame film-grain
// parse -> synthesis wiring -> byte-exact pixels, AND that grain is genuinely
// applied (an ungrained reconstruction of the same stream differs).
// ---------------------------------------------------------------------------
use aom_decode::frame::{
    apply_cdef, apply_deblock, apply_restoration, decode_frame_obus, decode_frame_obus_prefilter,
};
use aom_sys_ref::{ref_decode_av1_kf, ref_encode_av1_kf_film_grain};

/// Photographic-ish content (smooth gradient + sinusoid + noise) that stays in
/// the decode envelope (not synthetic-few-colours, which could trip the
/// encoder's screen-content path). Mirrors real_bitstream.rs::gen_plane.
fn gen_photo_plane(w: usize, h: usize, bd: i32, seed: u64, chroma: bool) -> Vec<u16> {
    let mut rng = Rng::new(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 0.25 + 0.5 * (0.6 * fx + 0.4 * fy);
            let wave = 0.12 * ((fx * 9.0).sin() * (fy * 7.0).cos());
            let noise = ((rng.next_u64() >> 40) as i64 % 33 - 16) as f64 / maxv as f64;
            let mut val = base + wave + noise * if chroma { 2.0 } else { 4.0 };
            val = val.clamp(0.0, 1.0);
            p[r * w + col] = (val * maxv as f64).round() as u16;
        }
    }
    p
}

/// Reconstruct the CROPPED display planes of `bytes` WITHOUT film grain, by
/// replaying `decode_frame_obus`'s exact filter chain (deblock -> CDEF ->
/// restoration) and cropping — the pre-grain reference for the anti-vacuous
/// "grain changed pixels" check.
fn ungrained_planes(
    bytes: &[u8],
    w: usize,
    h: usize,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (mut t, cfg, header) = decode_frame_obus_prefilter(bytes).unwrap();
    if header.loopfilter.filter_level != [0, 0] {
        apply_deblock(&mut t, &cfg, &header);
    }
    let cd = &header.cdef;
    let do_cdef = cd.cdef_bits != 0 || cd.cdef_strengths[0] != 0 || cd.cdef_uv_strengths[0] != 0;
    let do_lr = cfg.lr.any_enabled();
    let optimized_lr = !do_cdef;
    let pre_cdef =
        (do_lr && !optimized_lr).then(|| (t.recon.clone(), t.recon_u.clone(), t.recon_v.clone()));
    if do_cdef {
        apply_cdef(&mut t, &cfg, &header);
    }
    if do_lr {
        apply_restoration(&mut t, &cfg, pre_cdef.as_ref(), optimized_lr);
    }
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        y[r * w..(r + 1) * w].copy_from_slice(&t.recon[r * t.stride..r * t.stride + w]);
    }
    if mono {
        return (y, Vec::new(), Vec::new());
    }
    let cw = (w + ss_x as usize) >> ss_x;
    let ch = (h + ss_y as usize) >> ss_y;
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        u[r * cw..(r + 1) * cw]
            .copy_from_slice(&t.recon_u[r * t.stride_uv..r * t.stride_uv + cw]);
        v[r * cw..(r + 1) * cw]
            .copy_from_slice(&t.recon_v[r * t.stride_uv..r * t.stride_uv + cw]);
    }
    (y, u, v)
}

#[test]
fn film_grain_streams_decode_byte_identical_to_c() {
    // Grain test vectors (grain_test_vectors.h): 1 = rich chroma, no overlap;
    // 2 = ar_coeff_lag 3, overlap; 15 = chroma_scaling_from_luma (cfl), overlap.
    let vectors = [1i32, 2, 15];
    // (bd, (ss_x, ss_y), mono): 8/10-bit x 4:2:0/4:4:4 + monochrome.
    let formats: &[(i32, (i32, i32), bool)] = &[
        (8, (1, 1), false),
        (8, (0, 0), false),
        (8, (1, 1), true),
        (10, (1, 1), false),
        (10, (0, 0), false),
    ];
    let sizes = [(64usize, 64usize), (96, 80)];

    let mut total = 0u32;
    let mut chroma_grain_seen = 0u32;
    let mut cfl_seen = 0u32;
    let mut chroma_changed = 0u32;

    for &gv in &vectors {
        for &(bd, (ss_x, ss_y), mono) in formats {
            for &(w, h) in &sizes {
                let (cw, ch) = if mono {
                    (0, 0)
                } else {
                    ((w + ss_x as usize) >> ss_x, (h + ss_y as usize) >> ss_y)
                };
                let seed = ((w as u64) << 40)
                    ^ ((h as u64) << 24)
                    ^ ((bd as u64) << 12)
                    ^ ((gv as u64) << 4)
                    ^ (mono as u64);
                let y = gen_photo_plane(w, h, bd, seed ^ 0x1111, false);
                let u = gen_photo_plane(cw, ch, bd, seed ^ 0x2222, true);
                let v = gen_photo_plane(cw, ch, bd, seed ^ 0x3333, true);

                // REAL encoder bytes carrying film grain (cpu-used=0, GOOD).
                let bytes = ref_encode_av1_kf_film_grain(
                    &y, &u, &v, w, h, bd, mono, ss_x, ss_y, 32, 0, 0, gv,
                );

                // Rust decode (grain synthesized). Hard-error => test fails.
                let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
                    panic!(
                        "decode_frame_obus rejected grain stream gv={gv} {w}x{h} \
                         bd{bd} mono={mono} ss=({ss_x},{ss_y}): {e}"
                    )
                });

                // Anti-vacuous: the stream really signals + applies grain.
                let (_t, _cfg, header) = decode_frame_obus_prefilter(&bytes).unwrap();
                assert!(
                    header.film_grain_params_present,
                    "gv={gv}: seq film_grain_params_present not set"
                );
                assert!(header.film_grain.apply_grain, "gv={gv}: apply_grain=0");
                assert!(
                    header.film_grain.num_y_points > 0,
                    "gv={gv}: num_y_points=0 (would be a vacuous grain)"
                );

                // Gold oracle: REAL C decoder on the same bytes (grain applied).
                let cref = ref_decode_av1_kf(&bytes, w, h);
                assert_eq!(cref.info[0], bd);
                assert_eq!(cref.info[1] != 0, mono);
                assert_eq!(
                    rust.y, cref.y,
                    "LUMA mismatch gv={gv} {w}x{h} bd{bd} mono={mono} ss=({ss_x},{ss_y})"
                );
                if mono {
                    assert!(rust.u.is_empty() && rust.v.is_empty());
                } else {
                    assert_eq!(
                        rust.u, cref.u,
                        "U mismatch gv={gv} {w}x{h} bd{bd} ss=({ss_x},{ss_y})"
                    );
                    assert_eq!(
                        rust.v, cref.v,
                        "V mismatch gv={gv} {w}x{h} bd{bd} ss=({ss_x},{ss_y})"
                    );
                }

                // Anti-vacuous: grain genuinely changed pixels (an ungrained
                // reconstruction of the SAME stream differs). Because rust==cref,
                // this also proves the C decoder APPLIED grain (didn't skip).
                let (uy, uu, uv) = ungrained_planes(&bytes, w, h, mono, ss_x, ss_y);
                assert_ne!(
                    rust.y, uy,
                    "grain did not change luma (C skipped grain?) gv={gv} {w}x{h} bd{bd}"
                );

                if !mono {
                    if header.film_grain.num_cb_points > 0
                        || header.film_grain.chroma_scaling_from_luma
                    {
                        chroma_grain_seen += 1;
                    }
                    if header.film_grain.chroma_scaling_from_luma {
                        cfl_seen += 1;
                    }
                    if rust.u != uu || rust.v != uv {
                        chroma_changed += 1;
                    }
                }
                total += 1;
            }
        }
    }

    assert!(total >= 8, "need >=8 real grain streams, got {total}");
    assert!(chroma_grain_seen > 0, "no chroma-grain stream decoded");
    assert!(cfl_seen > 0, "no chroma-from-luma (cfl) stream decoded");
    assert!(
        chroma_changed > 0,
        "chroma grain never changed chroma pixels (vacuous)"
    );
    eprintln!(
        "film_grain e2e: {total} REAL streams byte-identical vs C (grain applied); \
         chroma_grain={chroma_grain_seen} cfl={cfl_seen} chroma_changed={chroma_changed}"
    );
}
