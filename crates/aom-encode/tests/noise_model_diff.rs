//! C7 grain-estimator — DIFFERENTIAL gate for the AR-coefficient NOISE MODEL.
//!
//! Feeds an IDENTICAL synthetic frame (`data`/`denoised`, 3 planes) + flat-block
//! map to the Rust port ([`aom_encode::noise_model::NoiseModel`]) and the REAL
//! exported `aom_noise_model_init` + `_update` (+ `_get_grain_parameters`), and
//! asserts BIT-IDENTICAL:
//!   * the fitted combined-state AR coefficients (`eqns.x`, per channel),
//!   * the AR gain (`ar_gain`),
//!   * the solved noise-strength curves (`strength_solver.eqns.x`),
//!   * the update status, AND
//!   * the quantized `aom_film_grain_t` (via serialized `filmgrn1` bytes).
//!
//! Exercised across `ar_coeff_lag` 1/2/3, DIAMOND + SQUARE shapes, 4:4:4 +
//! 4:2:0 subsampling, and 8-bit (lowbd read) + 10-bit (highbd). The noise is
//! spatially correlated so the fitted coefficients are non-trivial (anti-vacuous).

use aom_encode::grain_table::{write_film_grain_table, GrainTableEntry};
use aom_encode::noise_model::{NoiseModel, NoiseModelParams, NoiseShape, NoiseStatus};
use aom_dsp::entropy::header::FilmGrainParams;
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Symmetric int noise in `[-amp, amp]`.
    fn noise(&mut self, amp: i32) -> i32 {
        (self.next() % (2 * amp as u64 + 1)) as i32 - amp
    }
}

/// A single plane and its denoised counterpart (`u16` storage), plus stride.
struct Plane {
    data: Vec<u16>,
    denoised: Vec<u16>,
    stride: usize,
}

/// Build a plane: `denoised` = smooth ramp, `data` = denoised + spatially
/// correlated noise (a cheap AR-ish signal so the fit is non-degenerate).
fn make_plane(rng: &mut Rng, w: usize, h: usize, maxv: i32, base: i32) -> Plane {
    let mut denoised = vec![0u16; w * h];
    let mut noise = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            // Smooth ramp base.
            let d = base + (x as i32 * 13 + y as i32 * 7) % 24 - 12;
            denoised[y * w + x] = d.clamp(0, maxv) as u16;
            // Correlated noise: blend left/up neighbours + white.
            let left = if x > 0 { noise[y * w + x - 1] } else { 0 };
            let up = if y > 0 { noise[(y - 1) * w + x] } else { 0 };
            noise[y * w + x] = (left + up) / 3 + rng.noise(9);
        }
    }
    let data: Vec<u16> = (0..w * h)
        .map(|i| (denoised[i] as i32 + noise[i]).clamp(0, maxv) as u16)
        .collect();
    Plane { data, denoised, stride: w }
}

fn bit_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0)
}

fn assert_f64(tag: &str, port: &[f64], cref: &[f64]) {
    for (i, (&p, &r)) in port.iter().zip(cref).enumerate() {
        assert!(bit_eq(p, r), "{tag}[{i}]: port={p:?} vs C={r:?}");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    shape: NoiseShape,
    lag: i32,
    bit_depth: i32,
    use_highbd: bool,
    w: usize,
    h: usize,
    csx: i32,
    csy: i32,
    block_size: usize,
    seed: u64,
) -> (NoiseStatus, usize) {
    let maxv = (1i32 << bit_depth) - 1;
    let mut rng = Rng::new(seed);
    let cw = w >> csx;
    let ch = h >> csy;
    let planes = [
        make_plane(&mut rng, w, h, maxv, maxv / 2),
        make_plane(&mut rng, cw, ch, maxv, maxv / 2 + 20),
        make_plane(&mut rng, cw, ch, maxv, maxv / 2 - 20),
    ];
    // Flat-block map: mostly flat with a few holes (both arms exercised),
    // identical for port and C.
    let nbw = w.div_ceil(block_size);
    let nbh = h.div_ceil(block_size);
    let flat: Vec<u8> = (0..nbw * nbh)
        .map(|i| if i % 7 == 3 { 0 } else { 1 })
        .collect();
    let n_flat = flat.iter().filter(|&&v| v != 0).count();

    let data = [&planes[0].data[..], &planes[1].data[..], &planes[2].data[..]];
    let denoised = [
        &planes[0].denoised[..],
        &planes[1].denoised[..],
        &planes[2].denoised[..],
    ];
    let strides = [planes[0].stride, planes[1].stride, planes[2].stride];
    let seed16 = 0x51_23i32;

    // ---- Port ----
    let params = NoiseModelParams { shape, lag, bit_depth, use_highbd };
    let mut model = NoiseModel::new(params).expect("NoiseModel::new");
    let status = model.update(data, denoised, w, h, strides, [csx, csy], &flat, block_size);
    let mut fg = FilmGrainParams { random_seed: seed16, ..Default::default() };
    let got_grain = model.get_grain_parameters(&mut fg);

    // ---- C oracle ----
    let cfit = c::ref_noise_model_fit(
        shape as i32,
        lag,
        bit_depth,
        use_highbd,
        data,
        denoised,
        w,
        h,
        strides,
        csx,
        csy,
        &flat,
        block_size,
        seed16,
        true,
    );

    assert_eq!(status as i32, cfit.status, "status mismatch");
    // Per-channel AR coeffs, gain, strength curve.
    for cc in 0..3 {
        let nc = cfit.n[cc];
        let port_x = model.combined_ar_coeffs(cc);
        assert_eq!(port_x.len(), nc, "chan {cc}: eqns.n mismatch");
        assert_f64(&format!("ar_x c{cc}"), port_x, &cfit.ar_x[cc * 32..cc * 32 + nc]);
        assert!(
            bit_eq(model.combined_ar_gain(cc), cfit.ar_gain[cc]),
            "ar_gain c{cc}: port={} vs C={}",
            model.combined_ar_gain(cc),
            cfit.ar_gain[cc]
        );
        assert_f64(
            &format!("strength_x c{cc}"),
            model.combined_strength_curve(cc),
            &cfit.strength_x[cc * 20..cc * 20 + 20],
        );
    }

    // Grain params: compare the serialized table bytes.
    if status == NoiseStatus::Ok || status == NoiseStatus::DifferentNoiseType {
        assert!(got_grain, "port get_grain_parameters returned false");
        let c_table = cfit.grain_table.expect("C wrote a grain table");
        let port_table = write_film_grain_table(&[GrainTableEntry {
            params: fg,
            start_time: 0,
            end_time: i64::MAX,
        }]);
        assert_eq!(
            port_table,
            c_table,
            "grain params serialization differs\n--- C ---\n{}\n--- port ---\n{}",
            String::from_utf8_lossy(&c_table),
            String::from_utf8_lossy(&port_table)
        );
    }
    (status, n_flat)
}

#[test]
fn noise_model_fit_matches_c() {
    c::ref_init();
    let mut cases = 0;
    let mut ok_cases = 0;
    // (shape, lag, bit_depth, use_highbd, csx, csy)
    let configs = [
        (NoiseShape::Diamond, 2, 8, false, 1, 1),
        (NoiseShape::Square, 2, 8, false, 1, 1),
        (NoiseShape::Diamond, 3, 10, true, 1, 1),
        (NoiseShape::Square, 2, 10, true, 0, 0),
        (NoiseShape::Diamond, 1, 8, false, 0, 0),
        (NoiseShape::Square, 3, 10, true, 1, 1),
    ];
    for (ci, &(shape, lag, bd, hbd, csx, csy)) in configs.iter().enumerate() {
        for (ti, &(w, h)) in [(128usize, 128usize), (96, 128)].iter().enumerate() {
            let (st, _) = run_case(
                shape,
                lag,
                bd,
                hbd,
                w,
                h,
                csx,
                csy,
                32,
                0x9100 ^ (ci as u64) << 8 ^ ti as u64,
            );
            cases += 1;
            if st == NoiseStatus::Ok {
                ok_cases += 1;
            }
        }
    }
    // Anti-vacuity: the fit path (status OK) actually ran on most cases.
    assert!(ok_cases >= cases - 2, "too many non-OK fits ({ok_cases}/{cases})");
    println!("noise_model_diff: {cases} configs (lag 1/2/3 x DIAMOND/SQUARE x 8/10-bit x 444/420) bit-identical to C ({ok_cases} OK fits)");
}
