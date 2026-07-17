//! Film-grain NOISE-MODEL estimator — port of `aom_dsp/noise_model.c` (the
//! `--denoise-noise-level` grain-estimation path, C7). This is the first
//! chunk: the **noise-strength solver** (`aom_noise_strength_solver_*`) + its
//! `linsolve` core + the piecewise-linear LUT fit. It models noise standard
//! deviation as a piecewise-linear function of block intensity by accumulating
//! per-block `(mean, std)` observations into a banded normal-equation system,
//! regularizing, and solving.
//!
//! **All `f64`, matching C's exact operation order** (no FMA / fast-math either
//! side), so the port is bit-identical to the exported C functions — validated
//! by `crates/aom-encode/tests/noise_strength_solver_diff.rs` against
//! `aom_noise_strength_solver_*` / `aom_noise_strength_lut_eval` /
//! `aom_noise_strength_solver_fit_piecewise`.
//!
//! Remaining estimator chunks (see PARITY C7): flat-block finder, the AR
//! `noise_model` + `get_grain_parameters` quantize, the Wiener FFT denoise, and
//! the `denoise_and_model_run` orchestrator + encoder wiring. A byte-exact
//! `--denoise-noise-level` stream is float/FFT-determinism-gated; the realistic
//! per-kernel deliverable is this kind of differential parity.

/// `TINY_NEAR_ZERO` (`aom_dsp/mathutils.h`).
const TINY_NEAR_ZERO: f64 = 1.0E-16;

/// `fclamp` (`aom_dsp/aom_dsp_common.h`).
#[inline]
fn fclamp(value: f64, low: f64, high: f64) -> f64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

/// `linsolve` (`aom_dsp/mathutils.h`): Gaussian elimination with partial
/// pivoting. Solves `A x = b` for `x`; `a` and `b` are clobbered (scratch).
/// `stride` is the row stride of `a`. Returns `false` on a (near-)singular
/// pivot. Bit-exact op-order match to C.
fn linsolve(n: usize, a: &mut [f64], stride: usize, b: &mut [f64], x: &mut [f64]) -> bool {
    // Forward elimination.
    for k in 0..n.saturating_sub(1) {
        // Bring the largest magnitude to the diagonal position.
        let mut i = n - 1;
        while i > k {
            if a[(i - 1) * stride + k].abs() < a[i * stride + k].abs() {
                for j in 0..n {
                    let c = a[i * stride + j];
                    a[i * stride + j] = a[(i - 1) * stride + j];
                    a[(i - 1) * stride + j] = c;
                }
                let c = b[i];
                b[i] = b[i - 1];
                b[i - 1] = c;
            }
            i -= 1;
        }
        for i in k..(n - 1) {
            if a[k * stride + k].abs() < TINY_NEAR_ZERO {
                return false;
            }
            let c = a[(i + 1) * stride + k] / a[k * stride + k];
            for j in 0..n {
                a[(i + 1) * stride + j] -= c * a[k * stride + j];
            }
            b[i + 1] -= c * b[k];
        }
    }
    // Backward substitution.
    for i in (0..n).rev() {
        if a[i * stride + i].abs() < TINY_NEAR_ZERO {
            return false;
        }
        let mut c = 0.0;
        for j in (i + 1)..n {
            c += a[i * stride + j] * x[j];
        }
        x[i] = (b[i] - c) / a[i * stride + i];
    }
    true
}

/// `aom_equation_system_t` — the normal-equation system `A x = b` (dense `n×n`).
#[derive(Clone, Debug)]
struct EquationSystem {
    a: Vec<f64>, // n*n row-major
    b: Vec<f64>,
    x: Vec<f64>,
    n: usize,
}

impl EquationSystem {
    fn new(n: usize) -> Self {
        EquationSystem {
            a: vec![0.0; n * n],
            b: vec![0.0; n],
            x: vec![0.0; n],
            n,
        }
    }

    /// `equation_system_solve`: solve a COPY of `(A, b)` into `x` (leaving `A`,
    /// `b` untouched), via `linsolve`. Returns success.
    fn solve(&mut self) -> bool {
        let n = self.n;
        let mut a = self.a.clone();
        let mut b = self.b.clone();
        linsolve(n, &mut a, n, &mut b, &mut self.x)
    }
}

/// `aom_noise_strength_solver_t` — models noise std as a function of intensity,
/// over `num_bins` evenly-spaced intensity bins in `[min, max]`.
#[derive(Clone, Debug)]
pub struct NoiseStrengthSolver {
    eqns: EquationSystem,
    min_intensity: f64,
    max_intensity: f64,
    num_bins: usize,
    num_equations: i32,
    total: f64,
}

impl NoiseStrengthSolver {
    /// `aom_noise_strength_solver_init(solver, num_bins, bit_depth)`.
    pub fn new(num_bins: usize, bit_depth: i32) -> Self {
        NoiseStrengthSolver {
            eqns: EquationSystem::new(num_bins),
            min_intensity: 0.0,
            max_intensity: ((1u32 << bit_depth) - 1) as f64,
            num_bins,
            num_equations: 0,
            total: 0.0,
        }
    }

    /// `noise_strength_solver_get_bin_index`.
    fn get_bin_index(&self, value: f64) -> f64 {
        let val = fclamp(value, self.min_intensity, self.max_intensity);
        let range = self.max_intensity - self.min_intensity;
        (self.num_bins as f64 - 1.0) * (val - self.min_intensity) / range
    }

    /// `noise_strength_solver_get_value` — evaluate the current solution at `x`.
    pub fn get_value(&self, x: f64) -> f64 {
        let bin = self.get_bin_index(x);
        let bin_i0 = bin.floor() as usize;
        let bin_i1 = (self.num_bins - 1).min(bin_i0 + 1);
        let a = bin - bin_i0 as f64;
        (1.0 - a) * self.eqns.x[bin_i0] + a * self.eqns.x[bin_i1]
    }

    /// `aom_noise_strength_solver_add_measurement(solver, block_mean, noise_std)`.
    pub fn add_measurement(&mut self, block_mean: f64, noise_std: f64) {
        let bin = self.get_bin_index(block_mean);
        let bin_i0 = bin.floor() as usize;
        let bin_i1 = (self.num_bins - 1).min(bin_i0 + 1);
        let a = bin - bin_i0 as f64;
        let n = self.num_bins;
        self.eqns.a[bin_i0 * n + bin_i0] += (1.0 - a) * (1.0 - a);
        self.eqns.a[bin_i1 * n + bin_i0] += a * (1.0 - a);
        self.eqns.a[bin_i1 * n + bin_i1] += a * a;
        self.eqns.a[bin_i0 * n + bin_i1] += a * (1.0 - a);
        self.eqns.b[bin_i0] += (1.0 - a) * noise_std;
        self.eqns.b[bin_i1] += a * noise_std;
        self.total += noise_std;
        self.num_equations += 1;
    }

    /// `aom_noise_strength_solver_solve(solver)` — adds banded (tridiagonal)
    /// smoothness regularization proportional to the constraint count plus a
    /// small ridge toward the mean noise strength, then solves. Returns success.
    /// Matches C: the ridge term is folded into `eqns.b` IN PLACE (persists
    /// across calls), while `A` is regularized on a scratch copy.
    pub fn solve(&mut self) -> bool {
        let n = self.num_bins;
        let k_alpha = 2.0 * (self.num_equations as f64) / n as f64;

        // Regularize a copy of A (leave the accumulated A intact for the caller).
        let mut a = self.eqns.a.clone();
        for i in 0..n {
            let i_lo = if i == 0 { 0 } else { i - 1 };
            let i_hi = (n - 1).min(i + 1);
            a[i * n + i_lo] -= k_alpha;
            a[i * n + i] += 2.0 * k_alpha;
            a[i * n + i_hi] -= k_alpha;
        }

        // Small regularization toward the average noise strength.
        let mean = self.total / self.num_equations as f64;
        for i in 0..n {
            a[i * n + i] += 1.0 / 8192.;
            self.eqns.b[i] += mean / 8192.;
        }

        // equation_system_solve on (regularized A, updated b).
        let mut b = self.eqns.b.clone();
        linsolve(n, &mut a, n, &mut b, &mut self.eqns.x)
    }

    /// The solved per-bin strength curve (`solver.eqns.x`) — valid after
    /// [`Self::solve`].
    pub fn solved(&self) -> &[f64] {
        &self.eqns.x
    }

    /// `aom_noise_strength_solver_get_center(solver, i)`.
    pub fn get_center(&self, i: usize) -> f64 {
        let range = self.max_intensity - self.min_intensity;
        (i as f64) / (self.num_bins as f64 - 1.0) * range + self.min_intensity
    }

    /// `aom_noise_strength_solver_fit_piecewise(solver, max_output_points)` —
    /// greedily reduce the solved per-bin curve to a piecewise-linear LUT,
    /// removing interior points whose removal least increases the local
    /// approximation residual (never the endpoints), until under
    /// `max_output_points` and the average residual exceeds the bit-depth-
    /// normalized tolerance. `max_output_points < 0` → `num_bins`.
    pub fn fit_piecewise(&self, max_output_points: i32) -> NoiseStrengthLut {
        let k_tolerance = self.max_intensity * 0.00625 / 255.0;
        let mut lut = NoiseStrengthLut {
            points: (0..self.num_bins)
                .map(|i| [self.get_center(i), self.eqns.x[i]])
                .collect(),
        };
        let max_output_points = if max_output_points < 0 {
            self.num_bins as i32
        } else {
            max_output_points
        };

        let mut residual = vec![0.0f64; self.num_bins];
        self.update_piecewise_linear_residual(&lut, &mut residual, 0, self.num_bins);

        while lut.points.len() > 2 {
            let mut min_index = 1usize;
            for j in 1..(lut.points.len() - 1) {
                if residual[j] < residual[min_index] {
                    min_index = j;
                }
            }
            let dx = lut.points[min_index + 1][0] - lut.points[min_index - 1][0];
            let avg_residual = residual[min_index] / dx;
            if lut.points.len() as i32 <= max_output_points && avg_residual > k_tolerance {
                break;
            }
            // Remove point `min_index`. C `memmove`s only the POINTS array and
            // leaves the fixed-length `residual` array UN-shifted (entries past
            // `min_index` keep stale values that the next min-search reads —
            // reproduced here for bit-exactness), recomputing just the two
            // neighbours of the removed point.
            lut.points.remove(min_index);
            self.update_piecewise_linear_residual(&lut, &mut residual, min_index - 1, min_index + 1);
        }
        lut
    }

    /// `update_piecewise_linear_residual` — the area between the solver curve and
    /// the LUT segment that would bridge `[x_{i-1}, x_{i+1})` if point `i` were
    /// removed, for `i` in `[start, end)`.
    fn update_piecewise_linear_residual(
        &self,
        lut: &NoiseStrengthLut,
        residual: &mut [f64],
        start: usize,
        end: usize,
    ) {
        let dx = 255. / self.num_bins as f64;
        let hi = end.min(lut.points.len().saturating_sub(1));
        for i in start.max(1)..hi {
            let lower = 0i32.max(self.get_bin_index(lut.points[i - 1][0]).floor() as i32);
            let upper =
                (self.num_bins as i32 - 1).min(self.get_bin_index(lut.points[i + 1][0]).ceil() as i32);
            let mut r = 0.0;
            let mut j = lower;
            while j <= upper {
                let x = self.get_center(j as usize);
                if x < lut.points[i - 1][0] {
                    j += 1;
                    continue;
                }
                if x >= lut.points[i + 1][0] {
                    j += 1;
                    continue;
                }
                let y = self.eqns.x[j as usize];
                let a = (x - lut.points[i - 1][0])
                    / (lut.points[i + 1][0] - lut.points[i - 1][0]);
                let estimate_y = lut.points[i - 1][1] * (1.0 - a) + lut.points[i + 1][1] * a;
                r += (y - estimate_y).abs();
                j += 1;
            }
            residual[i] = r * dx;
        }
    }
}

/// `aom_noise_strength_lut_t` — a piecewise-linear `(x, y)` curve.
#[derive(Clone, Debug)]
pub struct NoiseStrengthLut {
    pub points: Vec<[f64; 2]>,
}

impl NoiseStrengthLut {
    /// `aom_noise_strength_lut_eval(lut, x)` — piecewise-linear interpolation
    /// with constant extrapolation outside `[x_0, x_{n-1}]`.
    pub fn eval(&self, x: f64) -> f64 {
        let p = &self.points;
        if x < p[0][0] {
            return p[0][1];
        }
        for i in 0..(p.len() - 1) {
            if x >= p[i][0] && x <= p[i + 1][0] {
                let a = (x - p[i][0]) / (p[i + 1][0] - p[i][0]);
                return p[i + 1][1] * a + p[i][1] * (1.0 - a);
            }
        }
        p[p.len() - 1][1]
    }
}
