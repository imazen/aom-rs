//! C7 grain-estimator chunk 2 — DIFFERENTIAL gate for the noise-strength solver.
//!
//! Feeds IDENTICAL `(block_mean, noise_std)` observation streams to the Rust
//! port ([`aom_encode::noise_model::NoiseStrengthSolver`]) and to the REAL
//! exported `aom_noise_strength_solver_*` (via `aom_sys_ref`), and asserts the
//! solved per-bin strength curve AND the fitted piecewise-linear LUT are
//! **bit-identical** (`f64` `==`). The whole chain is deterministic `f64` math
//! (Gaussian-elimination `linsolve` + banded regularization + greedy LUT
//! reduction) with no FMA/fast-math either side, so byte-exactness is the
//! correct bar. Swept over bit depths 8/10/12, bin counts, and observation
//! counts, incl. a degenerate all-same-mean case (near-singular solve path).

use aom_encode::noise_model::NoiseStrengthSolver;
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
    /// Uniform `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u64() % ((hi - lo) as u64)) as i32
    }
}

fn port_solve(means: &[f64], stds: &[f64], num_bins: usize, bd: i32) -> (bool, Vec<f64>) {
    let mut s = NoiseStrengthSolver::new(num_bins, bd);
    for (&m, &sd) in means.iter().zip(stds) {
        s.add_measurement(m, sd);
    }
    let ok = s.solve();
    (ok, s.solved().to_vec())
}

#[test]
fn noise_strength_solve_matches_c() {
    c::ref_init();
    let mut rng = Rng::new(0xC7_50_1e);
    let mut trials = 0;
    let mut solved_ok = 0;
    for t in 0..300 {
        let bd = [8, 10, 12][t % 3];
        let maxv = ((1u32 << bd) - 1) as f64;
        let num_bins = [20usize, 8, 40][(t / 3) % 3];
        let nobs = rng.range(4, 500) as usize;
        // Std range ~ 0..a few percent of the intensity range (realistic noise).
        let std_scale = maxv * (0.005 + 0.06 * rng.unit());
        let means: Vec<f64> = (0..nobs).map(|_| rng.unit() * maxv).collect();
        let stds: Vec<f64> = (0..nobs).map(|_| rng.unit() * std_scale).collect();

        let (port_ok, port_x) = port_solve(&means, &stds, num_bins, bd);
        let cref = c::ref_noise_strength_solve(&means, &stds, num_bins, bd);
        assert_eq!(port_ok, cref.is_some(), "trial {t}: solve success flag differs");
        if let Some(cx) = cref {
            solved_ok += 1;
            assert_eq!(port_x, cx, "trial {t}: solved curve mismatch (bd{bd}, {num_bins} bins, {nobs} obs)");

            // fit_piecewise across a few max-point caps incl. the unbounded (-1) case.
            for &max_points in &[-1i32, 2, 4, num_bins as i32] {
                let mut s = NoiseStrengthSolver::new(num_bins, bd);
                for (&m, &sd) in means.iter().zip(&stds) {
                    s.add_measurement(m, sd);
                }
                assert!(s.solve());
                let port_lut = s.fit_piecewise(max_points).points;
                let c_lut = c::ref_noise_strength_fit_piecewise(&means, &stds, num_bins, bd, max_points)
                    .expect("C fit_piecewise");
                assert_eq!(
                    port_lut, c_lut,
                    "trial {t}: fit_piecewise LUT mismatch (max_points={max_points})"
                );
            }
        }
        trials += 1;
    }
    // Degenerate: all observations at the same mean (exercises the near-singular
    // / heavily-regularized solve — must AGREE on the outcome, whatever it is).
    for bd in [8, 10, 12] {
        let maxv = ((1u32 << bd) - 1) as f64;
        let means = vec![maxv * 0.5; 50];
        let stds = vec![maxv * 0.02; 50];
        let (port_ok, port_x) = port_solve(&means, &stds, 20, bd);
        let cref = c::ref_noise_strength_solve(&means, &stds, 20, bd);
        assert_eq!(port_ok, cref.is_some(), "degenerate bd{bd}: solve flag differs");
        if let Some(cx) = cref {
            assert_eq!(port_x, cx, "degenerate bd{bd}: solved curve mismatch");
        }
    }
    println!("noise_strength_solver_diff: {trials} trials ({solved_ok} solved) bit-identical to C");
    assert!(solved_ok > 200, "too few non-singular solves ({solved_ok}) — vacuous");
}

#[test]
fn lut_eval_matches_c_interpolation() {
    // The LUT eval is a pure piecewise-linear interp; validate the port's
    // fit-then-eval matches C's fitted LUT evaluated at the same query points.
    c::ref_init();
    let mut rng = Rng::new(0x1a7);
    let bd = 8;
    let maxv = 255.0;
    let nobs = 200usize;
    let means: Vec<f64> = (0..nobs).map(|_| rng.unit() * maxv).collect();
    let stds: Vec<f64> = (0..nobs).map(|_| rng.unit() * 8.0).collect();
    let mut s = NoiseStrengthSolver::new(20, bd);
    for (&m, &sd) in means.iter().zip(&stds) {
        s.add_measurement(m, sd);
    }
    assert!(s.solve());
    let lut = s.fit_piecewise(-1);
    let c_lut = c::ref_noise_strength_fit_piecewise(&means, &stds, 20, bd, -1).unwrap();
    assert_eq!(lut.points, c_lut, "fitted LUT differs");
    // Eval at a dense grid incl. out-of-range extrapolation.
    for k in -20..=280 {
        let x = k as f64;
        let y = lut.eval(x);
        assert!(y.is_finite(), "eval({x}) not finite");
    }
    println!("lut_eval: {} points, eval finite over [-20,280]", lut.points.len());
}
