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

    /// `equation_system_clear` — zero `A`, `b`, `x`.
    fn clear(&mut self) {
        self.a.iter_mut().for_each(|v| *v = 0.0);
        self.b.iter_mut().for_each(|v| *v = 0.0);
        self.x.iter_mut().for_each(|v| *v = 0.0);
    }

    /// `equation_system_copy` — copy `A`, `b`, `x` from `src` (same `n`).
    fn copy_from(&mut self, src: &EquationSystem) {
        self.a.copy_from_slice(&src.a);
        self.b.copy_from_slice(&src.b);
        self.x.copy_from_slice(&src.x);
    }

    /// `equation_system_add` — accumulate `src`'s `A` and `b` (NOT `x`).
    fn add(&mut self, src: &EquationSystem) {
        for (d, s) in self.a.iter_mut().zip(&src.a) {
            *d += s;
        }
        for (d, s) in self.b.iter_mut().zip(&src.b) {
            *d += s;
        }
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

    /// `noise_strength_solver_clear` — reset the accumulator.
    fn clear(&mut self) {
        self.eqns.clear();
        self.num_equations = 0;
        self.total = 0.0;
    }

    /// `noise_strength_solver_add` — accumulate `src` into `self`.
    fn add(&mut self, src: &NoiseStrengthSolver) {
        self.eqns.add(&src.eqns);
        self.num_equations += src.num_equations;
        self.total += src.total;
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

/// `kLowPolyNumParams` — the planar (yd, xd, 1) low-order model.
const K_LOW_POLY_NUM_PARAMS: usize = 3;

/// `multiply_mat` (`aom_dsp/mathutils.h`): `res = m1 (m1_rows×inner) · m2
/// (inner×m2_cols)`, row-major, plain `f64` accumulation.
fn multiply_mat(
    m1: &[f64],
    m2: &[f64],
    res: &mut [f64],
    m1_rows: usize,
    inner_dim: usize,
    m2_cols: usize,
) {
    let mut idx = 0;
    for row in 0..m1_rows {
        for col in 0..m2_cols {
            let mut sum = 0.0;
            for inner in 0..inner_dim {
                sum += m1[row * inner_dim + inner] * m2[inner * m2_cols + col];
            }
            res[idx] = sum;
            idx += 1;
        }
    }
}

/// `aom_flat_block_finder_t` — finds low-gradient ("flat") blocks a noise model
/// can safely sample. Port of `aom_flat_block_finder_*` (`aom_dsp/noise_model.c`).
/// `A` is the fixed planar basis (`n×3`); `ata_inv` is `(AᵀA)⁻¹` (3×3),
/// precomputed via the lazy-inverse solve. All `f64`.
#[derive(Clone, Debug)]
pub struct FlatBlockFinder {
    a: Vec<f64>,              // n×3 planar basis
    ata_inv: [f64; K_LOW_POLY_NUM_PARAMS * K_LOW_POLY_NUM_PARAMS],
    block_size: usize,
    normalization: f64,
}

impl FlatBlockFinder {
    /// `aom_flat_block_finder_init(finder, block_size, bit_depth, use_highbd)`.
    /// (`use_highbd` only distinguishes the pixel read width — the port reads
    /// `u16` pixels uniformly, so it needs only `bit_depth` for normalization.)
    pub fn new(block_size: usize, bit_depth: i32) -> Self {
        let n = block_size * block_size;
        let mut a = vec![0.0f64; K_LOW_POLY_NUM_PARAMS * n];
        // AtA (3×3) accumulated, then inverted.
        let mut eqns = EquationSystem::new(K_LOW_POLY_NUM_PARAMS);
        let half = (block_size / 2) as f64;
        for y in 0..block_size {
            let yd = (y as f64 - half) / half;
            for x in 0..block_size {
                let xd = (x as f64 - half) / half;
                let coords = [yd, xd, 1.0];
                let row = y * block_size + x;
                a[K_LOW_POLY_NUM_PARAMS * row] = yd;
                a[K_LOW_POLY_NUM_PARAMS * row + 1] = xd;
                a[K_LOW_POLY_NUM_PARAMS * row + 2] = 1.0;
                for i in 0..K_LOW_POLY_NUM_PARAMS {
                    for j in 0..K_LOW_POLY_NUM_PARAMS {
                        eqns.a[K_LOW_POLY_NUM_PARAMS * i + j] += coords[i] * coords[j];
                    }
                }
            }
        }
        // Lazy inverse: solve AtA · x = e_i for each identity column.
        let mut ata_inv = [0.0f64; K_LOW_POLY_NUM_PARAMS * K_LOW_POLY_NUM_PARAMS];
        for i in 0..K_LOW_POLY_NUM_PARAMS {
            for b in eqns.b.iter_mut() {
                *b = 0.0;
            }
            eqns.b[i] = 1.0;
            eqns.solve();
            for j in 0..K_LOW_POLY_NUM_PARAMS {
                ata_inv[j * K_LOW_POLY_NUM_PARAMS + i] = eqns.x[j];
            }
        }
        FlatBlockFinder {
            a,
            ata_inv,
            block_size,
            normalization: ((1u32 << bit_depth) - 1) as f64,
        }
    }

    /// `aom_flat_block_finder_extract_block`: extract a (clamped-edge) block,
    /// fit the planar model, and return `(plane, residual_block)`. Offsets are
    /// SIGNED and clamped to `[0, w-1] × [0, h-1]` (the Wiener denoise pads with
    /// `offs = -block_size`; for the flat-block-finder's `offs ≥ 0` the low
    /// clamp is inert, so behaviour there is unchanged).
    pub(crate) fn extract_block(
        &self,
        data: &[u16],
        w: usize,
        h: usize,
        stride: usize,
        offsx: i32,
        offsy: i32,
    ) -> (Vec<f64>, Vec<f64>) {
        let bs = self.block_size;
        let n = bs * bs;
        let mut block = vec![0.0f64; n];
        for yi in 0..bs {
            let y = (offsy + yi as i32).clamp(0, h as i32 - 1) as usize;
            for xi in 0..bs {
                let x = (offsx + xi as i32).clamp(0, w as i32 - 1) as usize;
                block[yi * bs + xi] = data[y * stride + x] as f64 / self.normalization;
            }
        }
        let mut ata_inv_b = [0.0f64; K_LOW_POLY_NUM_PARAMS];
        let mut plane_coords = [0.0f64; K_LOW_POLY_NUM_PARAMS];
        let mut plane = vec![0.0f64; n];
        multiply_mat(&block, &self.a, &mut ata_inv_b, 1, n, K_LOW_POLY_NUM_PARAMS);
        multiply_mat(&self.ata_inv, &ata_inv_b, &mut plane_coords, K_LOW_POLY_NUM_PARAMS, K_LOW_POLY_NUM_PARAMS, 1);
        multiply_mat(&self.a, &plane_coords, &mut plane, n, K_LOW_POLY_NUM_PARAMS, 1);
        for i in 0..n {
            block[i] -= plane[i];
        }
        (plane, block)
    }

    /// `aom_flat_block_finder_run`: score every block by gradient-covariance
    /// flatness features, mark hard-thresholded flats, and additionally mark the
    /// top-10th-percentile sigmoid scores. Returns `(flat_blocks map, num_flat)`.
    /// The flat_blocks map is `num_blocks_w * num_blocks_h`, values `0/1/255`
    /// exactly as C (`is_flat ? 255` then `|= 1` for percentile) — the count is
    /// bit-exact; the percentile arm's `exp` sigmoid is the only libm-sensitive
    /// step (`is_flat` and everything else is exact `f64`/`sqrt`).
    pub fn run(&self, data: &[u16], w: usize, h: usize, stride: usize) -> (Vec<u8>, i32) {
        let bs = self.block_size;
        let n = bs * bs;
        let k_trace = 0.15 / (32.0 * 32.0);
        let k_ratio = 1.25;
        let k_norm = 0.08 / (32.0 * 32.0);
        let k_var = 0.005 / n as f64;
        let nbw = (w + bs - 1) / bs;
        let nbh = (h + bs - 1) / bs;
        let mut num_flat = 0i32;
        let mut flat_blocks = vec![0u8; nbw * nbh];
        // (score, index) pairs.
        let mut scores: Vec<(f32, usize)> = Vec::with_capacity(nbw * nbh);
        scores.resize(nbw * nbh, (0.0, 0));

        for by in 0..nbh {
            for bx in 0..nbw {
                let (_plane, block) = self.extract_block(data, w, h, stride, (bx * bs) as i32, (by * bs) as i32);
                let (mut gxx, mut gxy, mut gyy) = (0.0f64, 0.0f64, 0.0f64);
                let (mut mean, mut var) = (0.0f64, 0.0f64);
                for yi in 1..(bs - 1) {
                    for xi in 1..(bs - 1) {
                        let gx = (block[yi * bs + xi + 1] - block[yi * bs + xi - 1]) / 2.0;
                        let gy = (block[yi * bs + xi + bs] - block[yi * bs + xi - bs]) / 2.0;
                        gxx += gx * gx;
                        gxy += gx * gy;
                        gyy += gy * gy;
                        let value = block[yi * bs + xi];
                        mean += value;
                        var += value * value;
                    }
                }
                let denom = ((bs - 2) * (bs - 2)) as f64;
                mean /= denom;
                gxx /= denom;
                gxy /= denom;
                gyy /= denom;
                var = var / denom - mean * mean;

                let trace = gxx + gyy;
                let det = gxx * gyy - gxy * gxy;
                let e1 = (trace + (trace * trace - 4.0 * det).sqrt()) / 2.0;
                let e2 = (trace - (trace * trace - 4.0 * det).sqrt()) / 2.0;
                let norm = e1;
                let ratio = e1 / e2.max(1e-6);
                let is_flat =
                    trace < k_trace && ratio < k_ratio && norm < k_norm && var > k_var;
                let weights = [-6682.0f64, -0.2056, 13087.0, -12434.0, 2.5694];
                let mut sum_weights = weights[0] * var
                    + weights[1] * ratio
                    + weights[2] * trace
                    + weights[3] * norm
                    + weights[4];
                sum_weights = fclamp(sum_weights, -25.0, 100.0);
                let score = (1.0 / (1.0 + (-sum_weights).exp())) as f32;
                let idx = by * nbw + bx;
                flat_blocks[idx] = if is_flat { 255 } else { 0 };
                scores[idx] = (if var > k_var { score } else { 0.0 }, idx);
                num_flat += is_flat as i32;
            }
        }

        // qsort by score (ascending). C's compare_scores is a strict float
        // comparison returning 0 on ties; the flat_blocks OUTPUT depends only on
        // the percentile threshold VALUE (`>=`), so tie-ordering is immaterial.
        // Use a stable sort keyed on score only, mirroring the comparator.
        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        let top_nth = nbw * nbh * 90 / 100;
        let score_threshold = scores[top_nth].0;
        for &(sc, index) in &scores {
            if sc >= score_threshold {
                num_flat += (flat_blocks[index] == 0) as i32;
                flat_blocks[index] |= 1;
            }
        }
        (flat_blocks, num_flat)
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

// ===========================================================================
// AR-coefficient NOISE MODEL — port of the `aom_noise_model_*` core of
// `aom_dsp/noise_model.c`: the auto-regressive least-squares fit over flat
// blocks + the noise-strength solver integration + the grain-parameter
// quantize (`aom_noise_model_get_grain_parameters`). All `f64`, matching C's
// exact operation order; `sqrt` is IEEE-correct (always matches) and `log2`
// routes to the same libm as C. Validated by `tests/noise_model_diff.rs`
// against the REAL exported `aom_noise_model_init/update/get_grain_parameters`.
// ===========================================================================

use aom_dsp::entropy::header::FilmGrainParams;

/// `kMaxLag` (`aom_dsp/noise_model.c`).
const K_MAX_LAG: i32 = 4;
/// `kNumBins` for the per-channel strength solver (`noise_state_init`).
const K_NUM_BINS: usize = 20;

/// `aom_noise_shape` — the AR-coefficient support shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseShape {
    /// `AOM_NOISE_SHAPE_DIAMOND`
    Diamond = 0,
    /// `AOM_NOISE_SHAPE_SQUARE`
    Square = 1,
}

/// `aom_noise_model_params_t`.
#[derive(Clone, Copy, Debug)]
pub struct NoiseModelParams {
    pub shape: NoiseShape,
    pub lag: i32,
    pub bit_depth: i32,
    pub use_highbd: bool,
}

/// `aom_noise_status_t` — result of [`NoiseModel::update`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseStatus {
    Ok = 0,
    InvalidArgument = 1,
    InsufficientFlatBlocks = 2,
    DifferentNoiseType = 3,
    InternalError = 4,
}

/// `num_coeffs(params)`.
fn num_coeffs(params: &NoiseModelParams) -> usize {
    let n = 2 * params.lag + 1;
    match params.shape {
        NoiseShape::Diamond => (params.lag * (params.lag + 1)) as usize,
        NoiseShape::Square => ((n * n) / 2) as usize,
    }
}

/// `aom_noise_state_t` — per-channel AR system + strength solver.
#[derive(Clone, Debug)]
struct NoiseState {
    eqns: EquationSystem,
    strength_solver: NoiseStrengthSolver,
    num_observations: i32,
    ar_gain: f64,
}

impl NoiseState {
    /// `noise_state_init(state, n, bit_depth)`.
    fn new(n: usize, bit_depth: i32) -> Self {
        NoiseState {
            eqns: EquationSystem::new(n),
            strength_solver: NoiseStrengthSolver::new(K_NUM_BINS, bit_depth),
            num_observations: 0,
            ar_gain: 1.0,
        }
    }
}

/// `set_chroma_coefficient_fallback_soln(eqns)` — zero the AR coeffs but keep
/// the luma-correlation term (last) from the raw normal equations.
fn set_chroma_coefficient_fallback_soln(eqns: &mut EquationSystem) {
    let tol = 1e-6;
    let n = eqns.n;
    let last = n - 1;
    eqns.x.iter_mut().for_each(|v| *v = 0.0);
    if eqns.a[last * n + last].abs() > tol {
        eqns.x[last] = eqns.b[last] / eqns.a[last * n + last];
    }
}

/// `aom_noise_model_t` — complete AR-coefficient noise model for a planar
/// video (latest-frame + aggregated combined estimate, per channel).
#[derive(Clone, Debug)]
pub struct NoiseModel {
    params: NoiseModelParams,
    combined_state: [NoiseState; 3],
    latest_state: [NoiseState; 3],
    coords: Vec<[i32; 2]>,
    n: usize,
}

impl NoiseModel {
    /// `aom_noise_model_init(model, params)`. `None` on invalid params.
    pub fn new(params: NoiseModelParams) -> Option<Self> {
        let n = num_coeffs(&params);
        let lag = params.lag;
        if params.lag < 1 || params.lag > K_MAX_LAG {
            return None;
        }
        if !(params.bit_depth == 8 || params.bit_depth == 10 || params.bit_depth == 12) {
            return None;
        }
        let bit_depth = params.bit_depth;
        let combined_state =
            [NoiseState::new(n, bit_depth), NoiseState::new(n + 1, bit_depth), NoiseState::new(n + 1, bit_depth)];
        let latest_state =
            [NoiseState::new(n, bit_depth), NoiseState::new(n + 1, bit_depth), NoiseState::new(n + 1, bit_depth)];

        // Build the coefficient sample offsets (coords), matching C's scan.
        let mut coords = Vec::with_capacity(n);
        for y in -lag..=0 {
            let max_x = if y == 0 { -1 } else { lag };
            for x in -lag..=max_x {
                match params.shape {
                    NoiseShape::Diamond => {
                        if x.abs() <= y + lag {
                            coords.push([x, y]);
                        }
                    }
                    NoiseShape::Square => coords.push([x, y]),
                }
            }
        }
        debug_assert_eq!(coords.len(), n);
        Some(NoiseModel { params, combined_state, latest_state, coords, n })
    }

    /// `aom_noise_model_update` — fit the AR model + noise strength from a raw
    /// frame and its denoised variant over the flat-block map. Planes are
    /// `u16` row-major (`use_highbd=false` stores 8-bit values in `u16`);
    /// `strides` are in `u16` units. An empty plane (`&[]`) means "absent"
    /// (mirrors C's NULL, stopping the per-channel loop).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        data: [&[u16]; 3],
        denoised: [&[u16]; 3],
        w: usize,
        h: usize,
        strides: [usize; 3],
        chroma_sub_log2: [i32; 2],
        flat_blocks: &[u8],
        block_size: usize,
    ) -> NoiseStatus {
        let num_blocks_w = w.div_ceil(block_size);
        let num_blocks_h = h.div_ceil(block_size);
        let mut y_model_different = false;

        if block_size <= 1 {
            return NoiseStatus::InvalidArgument;
        }
        if (block_size as i32) < self.params.lag * 2 + 1 {
            return NoiseStatus::InvalidArgument;
        }

        for c in 0..3 {
            self.latest_state[c].eqns.clear();
            self.latest_state[c].num_observations = 0;
            self.latest_state[c].strength_solver.clear();
        }

        let num_flat = flat_blocks[..num_blocks_h * num_blocks_w].iter().filter(|&&v| v != 0).count();
        if num_flat <= 1 {
            return NoiseStatus::InsufficientFlatBlocks;
        }

        for channel in 0..3 {
            let no_subsampling = [0, 0];
            let sub = if channel > 0 { chroma_sub_log2 } else { no_subsampling };
            let is_chroma = channel != 0;
            if data[channel].is_empty() || denoised[channel].is_empty() {
                break;
            }
            self.add_block_observations(
                channel, w, h, strides[channel], sub, data, denoised, strides[0], flat_blocks,
                block_size, num_blocks_w, num_blocks_h,
            );

            if !ar_equation_system_solve(&mut self.latest_state[channel], is_chroma) {
                if is_chroma {
                    set_chroma_coefficient_fallback_soln(&mut self.latest_state[channel].eqns);
                } else {
                    return NoiseStatus::InternalError;
                }
            }

            // `add_noise_std_observations` needs the just-solved luma AR coeffs.
            let coeffs = self.latest_state[channel].eqns.x.clone();
            self.add_noise_std_observations(
                channel, &coeffs, w, h, strides[channel], sub, data, denoised, strides[0],
                flat_blocks, block_size, num_blocks_w, num_blocks_h,
            );

            if !self.latest_state[channel].strength_solver.solve() {
                return NoiseStatus::InternalError;
            }

            if channel == 0
                && self.combined_state[channel].strength_solver.num_equations > 0
                && self.is_noise_model_different()
            {
                y_model_different = true;
            }
            if y_model_different {
                continue;
            }

            self.combined_state[channel].num_observations +=
                self.latest_state[channel].num_observations;
            let latest_eqns = self.latest_state[channel].eqns.clone();
            self.combined_state[channel].eqns.add(&latest_eqns);
            if !ar_equation_system_solve(&mut self.combined_state[channel], is_chroma) {
                if is_chroma {
                    set_chroma_coefficient_fallback_soln(&mut self.combined_state[channel].eqns);
                } else {
                    return NoiseStatus::InternalError;
                }
            }
            let latest_solver = self.latest_state[channel].strength_solver.clone();
            self.combined_state[channel].strength_solver.add(&latest_solver);
            if !self.combined_state[channel].strength_solver.solve() {
                return NoiseStatus::InternalError;
            }
        }

        if y_model_different {
            NoiseStatus::DifferentNoiseType
        } else {
            NoiseStatus::Ok
        }
    }

    /// `add_block_observations` — accumulate the AR normal equations over the
    /// flat-block residual neighbourhoods.
    #[allow(clippy::too_many_arguments)]
    fn add_block_observations(
        &mut self,
        c: usize,
        w: usize,
        h: usize,
        stride: usize,
        sub: [i32; 2],
        data: [&[u16]; 3],
        denoised: [&[u16]; 3],
        alt_stride: usize,
        flat_blocks: &[u8],
        block_size: usize,
        num_blocks_w: usize,
        num_blocks_h: usize,
    ) {
        let lag = self.params.lag;
        let num_coords = self.n;
        let normalization = ((1i64 << self.params.bit_depth) - 1) as f64;
        let n = self.latest_state[c].eqns.n;
        let data_c = data[c];
        let den_c = denoised[c];
        let alt = if c > 0 { Some((data[0], denoised[0])) } else { None };
        let mut buffer = vec![0.0f64; num_coords + 1];

        let bsx = (block_size as i32) >> sub[0];
        let bsy = (block_size as i32) >> sub[1];
        let wsub = (w as i32) >> sub[0];
        let hsub = (h as i32) >> sub[1];

        for by in 0..num_blocks_h {
            let y_o = by as i32 * bsy;
            for bx in 0..num_blocks_w {
                let x_o = bx as i32 * bsx;
                if flat_blocks[by * num_blocks_w + bx] == 0 {
                    continue;
                }
                let y_start = if by > 0 && flat_blocks[(by - 1) * num_blocks_w + bx] != 0 { 0 } else { lag };
                let x_start = if bx > 0 && flat_blocks[by * num_blocks_w + bx - 1] != 0 { 0 } else { lag };
                let y_end = (hsub - by as i32 * bsy).min(bsy);
                let x_end = (wsub - bx as i32 * bsx - lag).min(
                    if bx + 1 < num_blocks_w && flat_blocks[by * num_blocks_w + bx + 1] != 0 {
                        bsx
                    } else {
                        bsx - lag
                    },
                );
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let val = extract_ar_row(
                            &self.coords, num_coords, data_c, den_c, stride, sub, alt, alt_stride,
                            x + x_o, y + y_o, &mut buffer,
                        );
                        let a = &mut self.latest_state[c].eqns.a;
                        let b = &mut self.latest_state[c].eqns.b;
                        for i in 0..n {
                            for j in 0..n {
                                a[i * n + j] += (buffer[i] * buffer[j]) / (normalization * normalization);
                            }
                            b[i] += (buffer[i] * val) / (normalization * normalization);
                        }
                        self.latest_state[c].num_observations += 1;
                    }
                }
            }
        }
    }

    /// `add_noise_std_observations` — feed per-block `(mean, adjusted_std)`
    /// measurements into the channel's strength solver.
    #[allow(clippy::too_many_arguments)]
    fn add_noise_std_observations(
        &mut self,
        c: usize,
        coeffs: &[f64],
        w: usize,
        h: usize,
        stride: usize,
        sub: [i32; 2],
        data: [&[u16]; 3],
        denoised: [&[u16]; 3],
        alt_stride: usize,
        flat_blocks: &[u8],
        block_size: usize,
        num_blocks_w: usize,
        num_blocks_h: usize,
    ) {
        let num_coords = self.n;
        let luma_gain = self.latest_state[0].ar_gain;
        let noise_gain = self.latest_state[c].ar_gain;
        let data_c = data[c];
        let den_c = denoised[c];
        let alt_data = if c > 0 { Some(data[0]) } else { None };

        let bsx = (block_size as i32) >> sub[0];
        let bsy = (block_size as i32) >> sub[1];
        let wsub = (w as i32) >> sub[0];
        let hsub = (h as i32) >> sub[1];

        for by in 0..num_blocks_h {
            let y_o = by as i32 * bsy;
            for bx in 0..num_blocks_w {
                let x_o = bx as i32 * bsx;
                if flat_blocks[by * num_blocks_w + bx] == 0 {
                    continue;
                }
                let num_samples_h = (hsub - by as i32 * bsy).min(bsy);
                let num_samples_w = (wsub - bx as i32 * bsx).min(bsx);
                if num_samples_w * num_samples_h > block_size as i32 {
                    let (mean_src, mean_stride, mean_x, mean_y) = match alt_data {
                        Some(ad) => (ad, alt_stride, x_o << sub[0], y_o << sub[1]),
                        None => (data_c, stride, x_o, y_o),
                    };
                    let block_mean =
                        get_block_mean(mean_src, w as i32, h as i32, mean_stride, mean_x, mean_y, block_size as i32);
                    let noise_var = get_noise_var(
                        data_c, den_c, stride, wsub, hsub, x_o, y_o, bsx, bsy,
                    );
                    let luma_strength = if c > 0 {
                        luma_gain * self.latest_state[0].strength_solver.get_value(block_mean)
                    } else {
                        0.0
                    };
                    let corr = if c > 0 { coeffs[num_coords] } else { 0.0 };
                    let uncorr_std =
                        (noise_var / 16.0).max(noise_var - (corr * luma_strength).powi(2)).sqrt();
                    let adjusted_strength = uncorr_std / noise_gain;
                    self.latest_state[c]
                        .strength_solver
                        .add_measurement(block_mean, adjusted_strength);
                }
            }
        }
    }

    /// `is_noise_model_different` — luma-only divergence check (AR
    /// cross-correlation + strength-histogram difference) between the latest
    /// and combined estimates.
    fn is_noise_model_different(&self) -> bool {
        let k_coeff_threshold = 0.9;
        let k_strength_threshold = 0.005 * (1i64 << (self.params.bit_depth - 8)) as f64;
        let c = 0;
        let corr = normalized_cross_correlation(
            &self.latest_state[c].eqns.x,
            &self.combined_state[c].eqns.x,
            self.combined_state[c].eqns.n,
        );
        if corr < k_coeff_threshold {
            return true;
        }
        let dx = 1.0 / self.latest_state[c].strength_solver.num_bins as f64;
        let latest = &self.latest_state[c].strength_solver.eqns;
        let combined = &self.combined_state[c].strength_solver.eqns;
        let ln = latest.n;
        let mut diff = 0.0;
        let mut total_weight = 0.0;
        for j in 0..ln {
            let mut weight = 0.0;
            for i in 0..ln {
                weight += latest.a[i * ln + j];
            }
            weight = weight.sqrt();
            diff += weight * (latest.x[j] - combined.x[j]).abs();
            total_weight += weight;
        }
        diff * dx / total_weight > k_strength_threshold
    }

    /// The fitted combined-state AR coefficients for channel `c`
    /// (`combined_state[c].eqns.x`) — for differential tests / callers.
    pub fn combined_ar_coeffs(&self, c: usize) -> &[f64] {
        &self.combined_state[c].eqns.x
    }

    /// The combined-state AR gain for channel `c`.
    pub fn combined_ar_gain(&self, c: usize) -> f64 {
        self.combined_state[c].ar_gain
    }

    /// The solved combined-state noise-strength curve for channel `c`
    /// (`combined_state[c].strength_solver.eqns.x`).
    pub fn combined_strength_curve(&self, c: usize) -> &[f64] {
        self.combined_state[c].strength_solver.solved()
    }

    /// `combined_state[c].strength_solver.num_equations` — the have-estimate
    /// gate in `aom_denoise_and_model_run`.
    pub fn combined_strength_num_equations(&self, c: usize) -> i32 {
        self.combined_state[c].strength_solver.num_equations
    }

    /// `aom_noise_model_save_latest` — promote the latest estimate to combined.
    pub fn save_latest(&mut self) {
        for c in 0..3 {
            let (le, lse, lne, lno, lg) = {
                let l = &self.latest_state[c];
                (
                    l.eqns.clone(),
                    l.strength_solver.eqns.clone(),
                    l.strength_solver.num_equations,
                    l.num_observations,
                    l.ar_gain,
                )
            };
            self.combined_state[c].eqns.copy_from(&le);
            self.combined_state[c].strength_solver.eqns.copy_from(&lse);
            self.combined_state[c].strength_solver.num_equations = lne;
            self.combined_state[c].num_observations = lno;
            self.combined_state[c].ar_gain = lg;
        }
    }

    /// `aom_noise_model_get_grain_parameters` — quantize the combined estimate
    /// into bitstream film-grain parameters. Returns `false` if `lag > 3` or a
    /// piecewise fit fails. `film_grain.random_seed` is preserved.
    pub fn get_grain_parameters(&self, film_grain: &mut FilmGrainParams) -> bool {
        if self.params.lag > 3 {
            return false;
        }
        let random_seed = film_grain.random_seed;
        *film_grain = FilmGrainParams::default();
        film_grain.random_seed = random_seed;
        film_grain.apply_grain = true;
        film_grain.update_parameters = true;
        film_grain.ar_coeff_lag = self.params.lag;

        // Reduced piecewise scaling LUTs per channel (14 luma / 10 chroma pts).
        let mut scaling: [Vec<[f64; 2]>; 3] = [
            self.combined_state[0].strength_solver.fit_piecewise(14).points,
            self.combined_state[1].strength_solver.fit_piecewise(10).points,
            self.combined_state[2].strength_solver.fit_piecewise(10).points,
        ];

        let strength_divisor = (1i64 << (self.params.bit_depth - 8)) as f64;
        let mut max_scaling_value = 1e-4;
        for c in 0..3 {
            for p in scaling[c].iter_mut() {
                p[0] = (255.0f64).min(p[0] / strength_divisor);
                p[1] = (255.0f64).min(p[1] / strength_divisor);
                max_scaling_value = p[1].max(max_scaling_value);
            }
        }

        let max_scaling_value_log2 = ((max_scaling_value.log2() + 1.0).floor() as i32).clamp(2, 5);
        film_grain.scaling_shift = 5 + (8 - max_scaling_value_log2);
        let scale_factor = (1i64 << (8 - max_scaling_value_log2)) as f64;
        film_grain.num_y_points = scaling[0].len() as i32;
        film_grain.num_cb_points = scaling[1].len() as i32;
        film_grain.num_cr_points = scaling[2].len() as i32;

        for c in 0..3 {
            for (i, p) in scaling[c].iter().enumerate() {
                let x = (p[0] + 0.5) as i32;
                let y = ((scale_factor * p[1] + 0.5) as i32).clamp(0, 255);
                match c {
                    0 => film_grain.scaling_points_y[i] = [x, y],
                    1 => film_grain.scaling_points_cb[i] = [x, y],
                    _ => film_grain.scaling_points_cr[i] = [x, y],
                }
            }
        }

        // Quantize the AR coefficients.
        let n_coeff = self.combined_state[0].eqns.n;
        let mut max_coeff = 1e-4f64;
        let mut min_coeff = -1e-4f64;
        let mut y_corr = [0.0f64; 2];
        let mut avg_luma_strength = 0.0;
        for c in 0..3 {
            let eqns = &self.combined_state[c].eqns;
            for i in 0..n_coeff {
                max_coeff = max_coeff.max(eqns.x[i]);
                min_coeff = min_coeff.min(eqns.x[i]);
            }
            let solver = &self.combined_state[c].strength_solver;
            let sn = solver.eqns.n;
            let mut average_strength = 0.0;
            let mut total_weight = 0.0;
            for i in 0..sn {
                let mut w = 0.0;
                for j in 0..sn {
                    w += solver.eqns.a[i * sn + j];
                }
                w = w.sqrt();
                average_strength += solver.eqns.x[i] * w;
                total_weight += w;
            }
            if total_weight == 0.0 {
                average_strength = 1.0;
            } else {
                average_strength /= total_weight;
            }
            if c == 0 {
                avg_luma_strength = average_strength;
            } else {
                y_corr[c - 1] = avg_luma_strength * eqns.x[n_coeff] / average_strength;
                max_coeff = max_coeff.max(y_corr[c - 1]);
                min_coeff = min_coeff.min(y_corr[c - 1]);
            }
        }
        film_grain.ar_coeff_shift = {
            let m = (1.0 + max_coeff.log2().floor()).max((-min_coeff).log2().ceil());
            (7 - m as i32).clamp(6, 9)
        };
        let scale_ar_coeff = (1i64 << film_grain.ar_coeff_shift) as f64;
        for c in 0..3 {
            let eqns = &self.combined_state[c].eqns;
            for i in 0..n_coeff {
                let v = ((scale_ar_coeff * eqns.x[i]).round() as i32).clamp(-128, 127);
                match c {
                    0 => film_grain.ar_coeffs_y[i] = v,
                    1 => film_grain.ar_coeffs_cb[i] = v,
                    _ => film_grain.ar_coeffs_cr[i] = v,
                }
            }
            if c > 0 {
                let v = ((scale_ar_coeff * y_corr[c - 1]).round() as i32).clamp(-128, 127);
                if c == 1 {
                    film_grain.ar_coeffs_cb[n_coeff] = v;
                } else {
                    film_grain.ar_coeffs_cr[n_coeff] = v;
                }
            }
        }

        film_grain.cb_mult = 128;
        film_grain.cb_luma_mult = 192;
        film_grain.cb_offset = 256;
        film_grain.cr_mult = 128;
        film_grain.cr_luma_mult = 192;
        film_grain.cr_offset = 256;
        film_grain.chroma_scaling_from_luma = false;
        film_grain.grain_scale_shift = 0;
        film_grain.overlap_flag = true;
        true
    }
}

/// `ar_equation_system_solve` — solve the AR system and derive `ar_gain` from
/// the diagonal variance and the fitted covariance.
fn ar_equation_system_solve(state: &mut NoiseState, is_chroma: bool) -> bool {
    let ret = state.eqns.solve();
    state.ar_gain = 1.0;
    if !ret {
        return ret;
    }
    let isc = is_chroma as i32;
    let n = state.eqns.n;
    let ni = n as i32;
    let nobs = state.num_observations as f64;
    let mut var = 0.0;
    for i in 0..(ni - isc) as usize {
        var += state.eqns.a[i * n + i] / nobs;
    }
    var /= (ni - isc) as f64;

    let mut sum_covar = 0.0;
    for i in 0..(ni - isc) as usize {
        let mut bi = state.eqns.b[i];
        if is_chroma {
            bi -= state.eqns.a[i * n + (n - 1)] * state.eqns.x[n - 1];
        }
        sum_covar += (bi * state.eqns.x[i]) / nobs;
    }
    let noise_var = (var - sum_covar).max(1e-6);
    state.ar_gain = (1.0f64).max((var / noise_var).max(1e-6).sqrt());
    ret
}

/// `extract_ar_row` — fill `buffer[0..num_coords]` with the residual
/// (`data - denoised`) at the AR-neighbour offsets around `(x, y)`, and (for
/// chroma, `alt` set) `buffer[num_coords]` with the co-located luma residual
/// averaged over the subsampling block. Returns the residual at `(x, y)`.
#[allow(clippy::too_many_arguments)]
fn extract_ar_row(
    coords: &[[i32; 2]],
    num_coords: usize,
    data: &[u16],
    denoised: &[u16],
    stride: usize,
    sub_log2: [i32; 2],
    alt: Option<(&[u16], &[u16])>,
    alt_stride: usize,
    x: i32,
    y: i32,
    buffer: &mut [f64],
) -> f64 {
    for i in 0..num_coords {
        let x_i = x + coords[i][0];
        let y_i = y + coords[i][1];
        let idx = (y_i as usize) * stride + x_i as usize;
        buffer[i] = data[idx] as f64 - denoised[idx] as f64;
    }
    let idx = (y as usize) * stride + x as usize;
    let val = data[idx] as f64 - denoised[idx] as f64;
    if let Some((alt_data, alt_denoised)) = alt {
        let mut avg_data = 0.0;
        let mut avg_denoised = 0.0;
        let mut num_samples = 0i32;
        for dy_i in 0..(1 << sub_log2[1]) {
            let y_up = (y << sub_log2[1]) + dy_i;
            for dx_i in 0..(1 << sub_log2[0]) {
                let x_up = (x << sub_log2[0]) + dx_i;
                let aidx = (y_up as usize) * alt_stride + x_up as usize;
                avg_data += alt_data[aidx] as f64;
                avg_denoised += alt_denoised[aidx] as f64;
                num_samples += 1;
            }
        }
        buffer[num_coords] = (avg_data - avg_denoised) / num_samples as f64;
    }
    val
}

/// `get_block_mean` (`aom_dsp/noise_model.c`).
fn get_block_mean(data: &[u16], w: i32, h: i32, stride: usize, x_o: i32, y_o: i32, block_size: i32) -> f64 {
    let max_h = (h - y_o).min(block_size);
    let max_w = (w - x_o).min(block_size);
    let mut block_mean = 0.0;
    for y in 0..max_h {
        for x in 0..max_w {
            block_mean += data[((y_o + y) as usize) * stride + (x_o + x) as usize] as f64;
        }
    }
    block_mean / (max_w * max_h) as f64
}

/// `get_noise_var` (`aom_dsp/noise_model.c`).
#[allow(clippy::too_many_arguments)]
fn get_noise_var(
    data: &[u16],
    denoised: &[u16],
    stride: usize,
    w: i32,
    h: i32,
    x_o: i32,
    y_o: i32,
    block_size_x: i32,
    block_size_y: i32,
) -> f64 {
    let max_h = (h - y_o).min(block_size_y);
    let max_w = (w - x_o).min(block_size_x);
    let mut noise_var = 0.0;
    let mut noise_mean = 0.0;
    for y in 0..max_h {
        for x in 0..max_w {
            let idx = ((y_o + y) as usize) * stride + (x_o + x) as usize;
            let noise = data[idx] as f64 - denoised[idx] as f64;
            noise_mean += noise;
            noise_var += noise * noise;
        }
    }
    noise_mean /= (max_w * max_h) as f64;
    noise_var / (max_w * max_h) as f64 - noise_mean * noise_mean
}

/// `aom_normalized_cross_correlation` (`aom_dsp/noise_util.c`).
fn normalized_cross_correlation(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut c = 0.0;
    let mut a_len = 0.0;
    let mut b_len = 0.0;
    for i in 0..n {
        a_len += a[i] * a[i];
        b_len += b[i] * b[i];
        c += a[i] * b[i];
    }
    c / (a_len.sqrt() * b_len.sqrt())
}
