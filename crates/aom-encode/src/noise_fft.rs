//! Film-grain noise **FFT transform** — port of `aom_dsp/fft.c` (the 2-D
//! real FFT/IFFT drivers + unrolled 1-D butterflies) and the `aom_noise_tx_*`
//! wrapper in `aom_dsp/noise_util.c`. This is the Wiener-denoise front end of
//! the `--denoise-noise-level` grain-estimation path.
//!
//! **Byte-exactness (VERIFIED, not assumed).** The transforms use only distinct
//! `add`/`sub`/`mul` roundings — the reference build's `fft.c.o` contains ZERO
//! fma instructions (gcc 15.2 `-std=c11` does not contract), and the SSE2/AVX2
//! SIMD variants the RTCD layer dispatches to use `_mm*_add/sub/mul_ps`
//! intrinsics (also non-fused, parallel across *columns* — same per-element op
//! order as the scalar path). So a strict-`f32` Rust port that mirrors the
//! scalar op order is bit-identical to whichever implementation the reference
//! build selects at runtime. The 1-D butterflies live in `noise_fft_gen.rs`
//! (mechanically transcribed by `xtask/transcribe_noise_fft.py`); this file is
//! the 2-D composition + the noise-transform struct. Proven by
//! `tests/noise_fft_diff.rs` against the real exported `aom_fft*x*_float` /
//! `aom_ifft*x*_float` and `aom_noise_tx_*`.

/// One correctly-rounded `f32` addition (C `add_float`). Distinct from `*`+`+`
/// fusion: kept a separate op so the transcribed butterflies round exactly as C.
#[inline(always)]
fn add(a: f32, b: f32) -> f32 {
    a + b
}
/// One correctly-rounded `f32` subtraction (C `sub_float`).
#[inline(always)]
fn sub(a: f32, b: f32) -> f32 {
    a - b
}
/// One correctly-rounded `f32` multiplication (C `mul_float`).
#[inline(always)]
fn mul(a: f32, b: f32) -> f32 {
    a * b
}

#[allow(clippy::needless_range_loop, clippy::identity_op)]
mod bfly {
    include!("noise_fft_gen.rs");
}

/// A 1-D transform of one column: `(input, output, stride)`.
type Fft1d = fn(&[f32], &mut [f32], usize);

/// `simple_transpose` (`aom_dsp/fft.c`): `B = Aᵀ` (pure data movement).
fn simple_transpose(a: &[f32], b: &mut [f32], n: usize) {
    for y in 0..n {
        for x in 0..n {
            b[y * n + x] = a[x * n + y];
        }
    }
}

/// `unpack_2d_output` (`aom_dsp/fft.c`): assemble the packed column-FFT into the
/// interleaved `[re, im]` 2-D spectrum, exploiting conjugate symmetry. Bit-exact
/// to the SSE2 unpack the oracle runs (identical sums/differences of the same
/// `col_fft` operands, no FMA).
fn unpack_2d_output(col_fft: &[f32], output: &mut [f32], n: usize) {
    for y in 0..=n / 2 {
        let y2 = y + n / 2;
        let y_extra = y2 > n / 2 && y2 < n;
        for x in 0..=n / 2 {
            let x2 = x + n / 2;
            let x_extra = x2 > n / 2 && x2 < n;
            output[2 * (y * n + x)] =
                col_fft[y * n + x] - if x_extra && y_extra { col_fft[y2 * n + x2] } else { 0.0 };
            output[2 * (y * n + x) + 1] = (if y_extra { col_fft[y2 * n + x] } else { 0.0 })
                + (if x_extra { col_fft[y * n + x2] } else { 0.0 });
            if y_extra {
                output[2 * ((n - y) * n + x)] = col_fft[y * n + x]
                    + if x_extra && y_extra { col_fft[y2 * n + x2] } else { 0.0 };
                output[2 * ((n - y) * n + x) + 1] =
                    -(if y_extra { col_fft[y2 * n + x] } else { 0.0 })
                        + (if x_extra { col_fft[y * n + x2] } else { 0.0 });
            }
        }
    }
}

/// `aom_fft_2d_gen` with `vec_size = 1` (the scalar path; the SIMD path is a
/// column-parallel rearrangement of the identical per-element math). Computes
/// the 2-D real FFT of `input` (`n×n`) into `output` (`2×n×n`), `temp` scratch.
fn fft_2d_gen(input: &[f32], temp: &mut [f32], output: &mut [f32], n: usize, tform: Fft1d) {
    for x in 0..n {
        tform(&input[x..], &mut output[x..], n);
    }
    simple_transpose(&output[..n * n], temp, n);
    for x in 0..n {
        tform(&temp[x..], &mut output[x..], n);
    }
    simple_transpose(&output[..n * n], temp, n);
    unpack_2d_output(temp, output, n);
}

/// `aom_ifft_2d_gen` with `vec_size = 1`, `fft_single = fft_multi = fft1d_n`,
/// `ifft_multi = ifft1d_n`. Inverse 2-D transform of the packed `input`
/// (`2×n×n`) into `output` (`n×n`); `temp` scratch (`2×n×n`).
fn ifft_2d_gen(input: &[f32], temp: &mut [f32], output: &mut [f32], n: usize, fft: Fft1d, ifft: Fft1d) {
    // Columns 0 and n/2 have conjugate symmetry -> direct real ifft.
    for y in 0..=n / 2 {
        output[y * n] = input[2 * y * n];
        output[y * n + 1] = input[2 * (y * n + n / 2)];
    }
    for y in (n / 2 + 1)..n {
        output[y * n] = input[2 * (y - n / 2) * n + 1];
        output[y * n + 1] = input[2 * ((y - n / 2) * n + n / 2) + 1];
    }
    for i in 0..2 {
        ifft(&output[i..], &mut temp[i..], n);
    }
    // Split the remaining columns into real then imaginary halves.
    for y in 0..n {
        for x in 1..n / 2 {
            output[y * n + (x + 1)] = input[2 * (y * n + x)];
        }
        for x in 1..n / 2 {
            output[y * n + (x + n / 2)] = input[2 * (y * n + x) + 1];
        }
    }
    // (vec_size = 1: the `fft_single` tail loop `2..vec_size` is empty.)
    for y in 2..n {
        fft(&output[y..], &mut temp[y..], n);
    }
    // Put the 0 and n/2 results in place.
    for x in 0..n {
        output[x] = temp[x * n];
        output[(n / 2) * n + x] = temp[x * n + 1];
    }
    // Rearrange + transpose the interior columns.
    for y in 1..n / 2 {
        for x in 0..=n / 2 {
            output[x + y * n] = temp[(y + 1) + x * n]
                + if x > 0 && x < n / 2 { temp[(y + n / 2) + (x + n / 2) * n] } else { 0.0 };
        }
        for x in (n / 2 + 1)..n {
            output[x + y * n] =
                temp[(y + 1) + (n - x) * n] - temp[(y + n / 2) + ((n - x) + n / 2) * n];
        }
        for x in 0..=n / 2 {
            output[x + (y + n / 2) * n] = temp[(y + n / 2) + x * n]
                - if x > 0 && x < n / 2 { temp[(y + 1) + (x + n / 2) * n] } else { 0.0 };
        }
        for x in (n / 2 + 1)..n {
            output[x + (y + n / 2) * n] =
                temp[(y + 1) + ((n - x) + n / 2) * n] + temp[(y + n / 2) + (n - x) * n];
        }
    }
    for y in 0..n {
        ifft(&output[y..], &mut temp[y..], n);
    }
    simple_transpose(&temp[..n * n], output, n);
}

/// Resolve the `(fft1d, ifft1d)` pair for a supported block size (2/4/8/16/32).
fn transforms_for(block_size: usize) -> Option<(Fft1d, Fft1d)> {
    Some(match block_size {
        2 => (bfly::fft1d_2, bfly::ifft1d_2),
        4 => (bfly::fft1d_4, bfly::ifft1d_4),
        8 => (bfly::fft1d_8, bfly::ifft1d_8),
        16 => (bfly::fft1d_16, bfly::ifft1d_16),
        32 => (bfly::fft1d_32, bfly::ifft1d_32),
        _ => return None,
    })
}

/// `aom_fftNxN_float(input, temp, output)` for `n ∈ {2,4,8,16,32}`. `input` is
/// `n×n`, `output` is `2×n×n`, `temp` scratch (`n×n` used). Returns whether the
/// size is supported.
pub fn fft2d(block_size: usize, input: &[f32], temp: &mut [f32], output: &mut [f32]) -> bool {
    match transforms_for(block_size) {
        Some((fft, _)) => {
            fft_2d_gen(input, temp, output, block_size, fft);
            true
        }
        None => false,
    }
}

/// `aom_ifftNxN_float(input, temp, output)` for `n ∈ {2,4,8,16,32}`. `input` is
/// `2×n×n`, `output` is `n×n`, `temp` scratch (`2×n×n`).
pub fn ifft2d(block_size: usize, input: &[f32], temp: &mut [f32], output: &mut [f32]) -> bool {
    match transforms_for(block_size) {
        Some((fft, ifft)) => {
            ifft_2d_gen(input, temp, output, block_size, fft, ifft);
            true
        }
        None => false,
    }
}

/// `aom_noise_psd_get_default_value` (`aom_dsp/noise_util.c`).
pub fn noise_psd_get_default_value(block_size: usize, factor: f32) -> f32 {
    (factor * factor / 10000.0) * block_size as f32 * block_size as f32 / 8.0
}

/// `aom_noise_tx_t` — the Wiener-denoise transform holder. Owns the transformed
/// spectrum (`tx_block`, `2×n×n`, `[re, im]` interleaved) and a working buffer.
/// Buffers are zero-initialised (some forward outputs are real-only, so their
/// imaginary lanes are never written — mirrors C's up-front `memset`).
pub struct NoiseTx {
    tx_block: Vec<f32>,
    temp: Vec<f32>,
    block_size: usize,
    fft: Fft1d,
    ifft: Fft1d,
}

impl NoiseTx {
    /// `aom_noise_tx_malloc(block_size)` — `None` for an unsupported size.
    pub fn new(block_size: usize) -> Option<Self> {
        let (fft, ifft) = transforms_for(block_size)?;
        let sz = 2 * block_size * block_size;
        Some(NoiseTx {
            tx_block: vec![0.0; sz],
            temp: vec![0.0; sz],
            block_size,
            fft,
            ifft,
        })
    }

    /// `aom_noise_tx_forward(noise_tx, data)`.
    pub fn forward(&mut self, data: &[f32]) {
        fft_2d_gen(data, &mut self.temp, &mut self.tx_block, self.block_size, self.fft);
    }

    /// `aom_noise_tx_filter(noise_tx, psd)` — Wiener-style attenuation of each
    /// spectral coefficient toward the noise power spectral density.
    pub fn filter(&mut self, psd: &[f32]) {
        let block_size = self.block_size;
        const K_BETA: f32 = 1.1;
        const K_EPS: f32 = 1e-6;
        for y in 0..block_size {
            for x in 0..block_size {
                let i = y * block_size + x;
                let c0 = (self.tx_block[2 * i]).abs().max(1e-8);
                let c1 = (self.tx_block[2 * i + 1]).abs().max(1e-8);
                let p = c0 * c0 + c1 * c1;
                if p > K_BETA * psd[i] && p > 1e-6 {
                    self.tx_block[2 * i] *= (p - psd[i]) / p.max(K_EPS);
                    self.tx_block[2 * i + 1] *= (p - psd[i]) / p.max(K_EPS);
                } else {
                    self.tx_block[2 * i] *= (K_BETA - 1.0) / K_BETA;
                    self.tx_block[2 * i + 1] *= (K_BETA - 1.0) / K_BETA;
                }
            }
        }
    }

    /// `aom_noise_tx_inverse(noise_tx, data)` — inverse transform + `/n²` scale.
    pub fn inverse(&mut self, data: &mut [f32]) {
        let n = self.block_size * self.block_size;
        ifft_2d_gen(&self.tx_block, &mut self.temp, data, self.block_size, self.fft, self.ifft);
        for v in data.iter_mut().take(n) {
            *v /= n as f32;
        }
    }

    /// `aom_noise_tx_add_energy(noise_tx, psd)` — accumulate `|coeff|²` of the
    /// low half-plane into `psd`.
    pub fn add_energy(&self, psd: &mut [f32]) {
        let block_size = self.block_size;
        for yb in 0..block_size {
            for xb in 0..=block_size / 2 {
                let c0 = self.tx_block[2 * (yb * block_size + xb)];
                let c1 = self.tx_block[2 * (yb * block_size + xb) + 1];
                psd[yb * block_size + xb] += c0 * c0 + c1 * c1;
            }
        }
    }

    /// Read-only view of the packed spectrum (`2×n×n`) — for differential tests.
    pub fn tx_block(&self) -> &[f32] {
        &self.tx_block
    }
}
