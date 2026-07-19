//! Film-grain ESTIMATION wiring — port of `aom_wiener_denoise_2d` +
//! `aom_denoise_and_model_run` (`aom_dsp/noise_model.c`), the end-to-end
//! `--denoise-noise-level` path: denoise the source with an overlapped-block
//! Wiener filter (FFT), find flat blocks, fit the AR noise model over the
//! (source − denoised) residual, and quantize to `aom_film_grain_t`.
//!
//! **Byte-exactness.** Every ingredient is byte-exact vs C: the FFT noise
//! transform ([`crate::noise_fft`], instruction-level non-FMA proof), the
//! flat-block planar fit + the AR model ([`crate::noise_model`], differential
//! gates), and the only remaining libm dependency here is `cos` (the half-cosine
//! window), which routes to the same glibc as C. The Floyd–Steinberg
//! dither/quantize and the overlapped-block accumulation are deterministic `f32`.
//! Validated by `tests/wiener_denoise_diff.rs` (the denoise) and
//! `tests/denoise_and_model_diff.rs` (the full orchestrator) against the REAL
//! `aom_wiener_denoise_2d` / `aom_denoise_and_model_run`.

use crate::noise_fft::{noise_psd_get_default_value, NoiseTx};
use crate::noise_model::{FlatBlockFinder, NoiseModel, NoiseModelParams, NoiseShape, NoiseStatus};
use aom_dsp::entropy::header::FilmGrainParams;
use core::f64::consts::PI;

/// `pointwise_multiply(a, b)` — `b[i] *= a[i]`.
fn pointwise_multiply(a: &[f32], b: &mut [f32], n: usize) {
    for i in 0..n {
        b[i] *= a[i];
    }
}

/// `get_half_cos_window(block_size)` — the separable half-cosine analysis
/// window. `cos` matches glibc.
fn get_half_cos_window(block_size: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; block_size * block_size];
    for y in 0..block_size {
        let cos_yd = ((0.5 + y as f64) * PI / block_size as f64 - PI / 2.0).cos();
        for x in 0..block_size {
            let cos_xd = ((0.5 + x as f64) * PI / block_size as f64 - PI / 2.0).cos();
            window[y * block_size + x] = (cos_yd * cos_xd) as f32;
        }
    }
    window
}

/// `dither_and_quantize` — Floyd–Steinberg error-diffuse `result` back to
/// integer `denoised` pixels (`u16` storage; `block_normalization` is
/// `(1<<bit_depth)-1`). Reads the `result` region offset by one block.
#[allow(clippy::too_many_arguments)]
fn dither_and_quantize(
    result: &mut [f32],
    result_stride: usize,
    denoised: &mut [u16],
    w: usize,
    h: usize,
    stride: usize,
    chroma_sub_w: i32,
    chroma_sub_h: i32,
    block_size: usize,
    block_normalization: f32,
) {
    let hh = h >> chroma_sub_h;
    let ww = w >> chroma_sub_w;
    let bsw = block_size >> chroma_sub_w;
    let bsh = block_size >> chroma_sub_h;
    for y in 0..hh {
        for x in 0..ww {
            let result_idx = (y + bsh) * result_stride + x + bsw;
            let new_val_f = (result[result_idx] * block_normalization + 0.5)
                .max(0.0)
                .min(block_normalization);
            let new_val = new_val_f as u16; // truncation, as C's (INT_TYPE) cast
            let err = -((new_val as f32) / block_normalization - result[result_idx]);
            denoised[y * stride + x] = new_val;
            if x + 1 < ww {
                result[result_idx + 1] += err * 7.0 / 16.0;
            }
            if y + 1 < hh {
                if x > 0 {
                    result[result_idx + result_stride - 1] += err * 3.0 / 16.0;
                }
                result[result_idx + result_stride] += err * 5.0 / 16.0;
                if x + 1 < ww {
                    result[result_idx + result_stride + 1] += err * 1.0 / 16.0;
                }
            }
        }
    }
}

/// `aom_wiener_denoise_2d` — overlapped-block Wiener denoise. `data` and
/// `denoised` are `u16` planes (chroma sized `(w>>ss)×(h>>ss)`); `strides` in
/// `u16` units; `noise_psd[c]` is `block_size²` (flat, per channel). Returns
/// `false` on unsupported chroma or block size (mirrors C's `init_success`).
/// Requires `chroma_sub[0] == chroma_sub[1]`.
#[allow(clippy::too_many_arguments)]
pub fn wiener_denoise_2d(
    data: [&[u16]; 3],
    denoised: [&mut [u16]; 3],
    w: usize,
    h: usize,
    strides: [usize; 3],
    chroma_sub: [i32; 2],
    noise_psd: [&[f32]; 3],
    block_size: usize,
    bit_depth: i32,
) -> bool {
    if chroma_sub[0] != chroma_sub[1] {
        return false;
    }
    let num_blocks_w = w.div_ceil(block_size);
    let num_blocks_h = h.div_ceil(block_size);
    let result_stride = (num_blocks_w + 2) * block_size;
    let result_height = (num_blocks_h + 2) * block_size;
    let block_normalization = ((1i64 << bit_depth) - 1) as f32;

    let block_finder_full = FlatBlockFinder::new(block_size, bit_depth);
    let window_full = get_half_cos_window(block_size);
    let mut tx_full = match NoiseTx::new(block_size) {
        Some(t) => t,
        None => return false,
    };

    let (block_finder_chroma, window_chroma, mut tx_chroma) = if chroma_sub[0] != 0 {
        let bs_c = block_size >> chroma_sub[0];
        (
            Some(FlatBlockFinder::new(bs_c, bit_depth)),
            Some(get_half_cos_window(bs_c)),
            NoiseTx::new(bs_c),
        )
    } else {
        (None, None, None)
    };
    if chroma_sub[0] != 0 && tx_chroma.is_none() {
        return false;
    }

    let mut result = vec![0.0f32; result_stride * result_height];

    for c in 0..3 {
        if data[c].is_empty() || denoised[c].is_empty() {
            continue;
        }
        let chroma = c > 0 && chroma_sub[0] > 0;
        let chroma_sub_h = if c > 0 { chroma_sub[1] } else { 0 };
        let chroma_sub_w = if c > 0 { chroma_sub[0] } else { 0 };
        let window = if chroma {
            window_chroma.as_ref().unwrap()
        } else {
            &window_full
        };
        let bf = if chroma {
            block_finder_chroma.as_ref().unwrap()
        } else {
            &block_finder_full
        };
        let tx = if chroma { tx_chroma.as_mut().unwrap() } else { &mut tx_full };

        let bsw = block_size >> chroma_sub_w;
        let bsh = block_size >> chroma_sub_h;
        let pixels_per_block = bsw * bsh;
        let cw = w >> chroma_sub_w;
        let ch = h >> chroma_sub_h;

        result.iter_mut().for_each(|v| *v = 0.0);

        let mut block = vec![0.0f32; 2 * block_size * block_size];
        let mut plane = vec![0.0f32; block_size * block_size];

        let mut offsy = 0i32;
        while offsy < bsh as i32 {
            let mut offsx = 0i32;
            while offsx < bsw as i32 {
                for by in -1..num_blocks_h as i32 {
                    for bx in -1..num_blocks_w as i32 {
                        let (plane_d, block_d) = bf.extract_block(
                            data[c],
                            cw,
                            ch,
                            strides[c],
                            bx * bsw as i32 + offsx,
                            by * bsh as i32 + offsy,
                        );
                        for j in 0..pixels_per_block {
                            block[j] = block_d[j] as f32;
                            plane[j] = plane_d[j] as f32;
                        }
                        pointwise_multiply(window, &mut block, pixels_per_block);
                        tx.forward(&block);
                        tx.filter(noise_psd[c]);
                        tx.inverse(&mut block);
                        pointwise_multiply(window, &mut plane, pixels_per_block);

                        for y in 0..bsh {
                            let y_result = y as i32 + (by + 1) * bsh as i32 + offsy;
                            for x in 0..bsw {
                                let x_result = x as i32 + (bx + 1) * bsw as i32 + offsx;
                                let ri = y_result as usize * result_stride + x_result as usize;
                                result[ri] += (block[y * bsw + x] + plane[y * bsw + x])
                                    * window[y * bsw + x];
                            }
                        }
                    }
                }
                offsx += bsw as i32 / 2;
            }
            offsy += bsh as i32 / 2;
        }

        dither_and_quantize(
            &mut result,
            result_stride,
            denoised[c],
            w,
            h,
            strides[c],
            chroma_sub_w,
            chroma_sub_h,
            block_size,
            block_normalization,
        );
    }
    true
}

/// `aom_denoise_and_model_t` — the `--denoise-noise-level` estimator context.
/// Holds the persistent noise model (combined across frames) and per-dimension
/// scratch, reallocated on a size change (mirrors C).
pub struct DenoiseAndModel {
    block_size: usize,
    bit_depth: i32,
    noise_level: f32,
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    subsampling_x: i32,
    subsampling_y: i32,
    noise_psd: [Vec<f32>; 3],
    denoised: [Vec<u16>; 3],
    flat_blocks: Vec<u8>,
    num_blocks_w: usize,
    num_blocks_h: usize,
    flat_block_finder: Option<FlatBlockFinder>,
    noise_model: Option<NoiseModel>,
}

impl DenoiseAndModel {
    /// `aom_denoise_and_model_alloc(bit_depth, block_size, noise_level)`.
    pub fn new(bit_depth: i32, block_size: usize, noise_level: f32) -> Self {
        DenoiseAndModel {
            block_size,
            bit_depth,
            noise_level,
            width: 0,
            height: 0,
            y_stride: 0,
            uv_stride: 0,
            subsampling_x: 0,
            subsampling_y: 0,
            noise_psd: [
                vec![0.0; block_size * block_size],
                vec![0.0; block_size * block_size],
                vec![0.0; block_size * block_size],
            ],
            denoised: [Vec::new(), Vec::new(), Vec::new()],
            flat_blocks: Vec::new(),
            num_blocks_w: 0,
            num_blocks_h: 0,
            flat_block_finder: None,
            noise_model: None,
        }
    }

    /// The denoised plane `c` from the last [`run`](Self::run).
    pub fn denoised(&self, c: usize) -> &[u16] {
        &self.denoised[c]
    }

    /// `denoise_and_model_realloc_if_necessary`.
    fn realloc_if_necessary(
        &mut self,
        w: usize,
        h: usize,
        y_stride: usize,
        uv_stride: usize,
        uv_height: usize,
        ss_x: i32,
        ss_y: i32,
    ) -> bool {
        if self.width == w && self.height == h && self.y_stride == y_stride && self.uv_stride == uv_stride {
            return true;
        }
        self.width = w;
        self.height = h;
        self.y_stride = y_stride;
        self.uv_stride = uv_stride;
        self.subsampling_x = ss_x;
        self.subsampling_y = ss_y;

        self.denoised[0] = vec![0u16; y_stride * h];
        self.denoised[1] = vec![0u16; uv_stride * uv_height];
        self.denoised[2] = vec![0u16; uv_stride * uv_height];

        self.num_blocks_w = w.div_ceil(self.block_size);
        self.num_blocks_h = h.div_ceil(self.block_size);
        self.flat_blocks = vec![0u8; self.num_blocks_w * self.num_blocks_h];

        self.flat_block_finder = Some(FlatBlockFinder::new(self.block_size, self.bit_depth));
        let params = NoiseModelParams {
            shape: NoiseShape::Square,
            lag: 3,
            bit_depth: self.bit_depth,
            use_highbd: self.bit_depth > 8,
        };
        self.noise_model = NoiseModel::new(params);
        if self.noise_model.is_none() {
            return false;
        }

        // Flat PSD (default value from block size + noise level).
        let y_noise = noise_psd_get_default_value(self.block_size, self.noise_level);
        let uv_noise = noise_psd_get_default_value(self.block_size >> ss_x, self.noise_level);
        for i in 0..self.block_size * self.block_size {
            self.noise_psd[0][i] = y_noise;
            self.noise_psd[1][i] = uv_noise;
            self.noise_psd[2][i] = uv_noise;
        }
        true
    }

    /// `aom_denoise_and_model_run` — estimate film grain for one frame. `data`
    /// planes are `u16` (chroma sized `(w>>ss_x)×(h>>ss_y)`, `uv_height` its
    /// height); on success `film_grain` is populated and
    /// `film_grain.apply_grain` set. The denoised planes are available via
    /// [`denoised`](Self::denoised) for the caller to apply if desired.
    /// `film_grain.random_seed` is preserved (defaults to 7391 if zero).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        data: [&[u16]; 3],
        w: usize,
        h: usize,
        y_stride: usize,
        uv_stride: usize,
        uv_height: usize,
        ss_x: i32,
        ss_y: i32,
        monochrome: bool,
        film_grain: &mut FilmGrainParams,
    ) -> bool {
        let block_size = self.block_size;
        if !self.realloc_if_necessary(w, h, y_stride, uv_stride, uv_height, ss_x, ss_y) {
            return false;
        }
        let strides = [y_stride, uv_stride, uv_stride];

        // Flat-block map from luma.
        let (flat, _n) = self.flat_block_finder.as_ref().unwrap().run(data[0], w, h, y_stride);
        self.flat_blocks = flat;

        // Denoise (in-place into self.denoised).
        {
            let [d0, d1, d2] = &mut self.denoised;
            let den = [d0.as_mut_slice(), d1.as_mut_slice(), d2.as_mut_slice()];
            if !wiener_denoise_2d(
                data,
                den,
                w,
                h,
                strides,
                [ss_x, ss_y],
                [&self.noise_psd[0], &self.noise_psd[1], &self.noise_psd[2]],
                block_size,
                self.bit_depth,
            ) {
                return false;
            }
        }

        // Update the AR model over (source, denoised).
        let denoised_ref = [
            self.denoised[0].as_slice(),
            self.denoised[1].as_slice(),
            self.denoised[2].as_slice(),
        ];
        let model = self.noise_model.as_mut().unwrap();
        let status = model.update(data, denoised_ref, w, h, strides, [ss_x, ss_y], &self.flat_blocks, block_size);

        let have_estimate = match status {
            NoiseStatus::Ok => true,
            NoiseStatus::DifferentNoiseType => {
                model.save_latest();
                true
            }
            _ => model.combined_strength_num_equations(0) > 0,
        };

        film_grain.apply_grain = false;
        let _ = monochrome;
        if have_estimate {
            if !model.get_grain_parameters(film_grain) {
                return false;
            }
            if film_grain.random_seed == 0 {
                film_grain.random_seed = 7391;
            }
        }
        true
    }
}

/// One-shot film-grain estimation for a single frame (the still / KEY-frame
/// `--denoise-noise-level` case): construct a [`DenoiseAndModel`], run it once,
/// and return `(grain_params, denoised_planes)` on success. `None` if the model
/// could not be initialised or the estimate failed. The caller applies the
/// denoised planes to the encode source iff it wants `--denoise` behaviour.
#[allow(clippy::too_many_arguments)]
pub fn estimate_film_grain(
    data: [&[u16]; 3],
    w: usize,
    h: usize,
    y_stride: usize,
    uv_stride: usize,
    uv_height: usize,
    ss_x: i32,
    ss_y: i32,
    monochrome: bool,
    bit_depth: i32,
    block_size: usize,
    noise_level: f32,
    random_seed: i32,
) -> Option<(FilmGrainParams, [Vec<u16>; 3])> {
    let mut ctx = DenoiseAndModel::new(bit_depth, block_size, noise_level);
    let mut fg = FilmGrainParams { random_seed, ..Default::default() };
    if !ctx.run(data, w, h, y_stride, uv_stride, uv_height, ss_x, ss_y, monochrome, &mut fg) {
        return None;
    }
    if !fg.apply_grain {
        return None;
    }
    let denoised = [ctx.denoised[0].clone(), ctx.denoised[1].clone(), ctx.denoised[2].clone()];
    Some((fg, denoised))
}
