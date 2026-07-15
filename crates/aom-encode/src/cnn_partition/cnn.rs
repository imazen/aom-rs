//! Port of the CNN conv engine used by `intra_mode_cnn_partition` — the
//! `av1/encoder/cnn.c` `av1_cnn_predict_c` path specialised to THIS model's
//! config (`av1_intra_mode_cnn_partition_cnn_config`): 5 sequential layers,
//! all branch-0, no maxpool / batchnorm / deconvolve / branch-copy / combine,
//! every layer `PADDING_VALID` + `RELU`. So the whole engine reduces to a
//! 5-layer VALID-convolution cascade where layers 1..4 each tap a multi-scale
//! output.
//!
//! Cascade (input 65×65×1, normalised `pixel/255`):
//! | layer | filter | stride | in→out ch | in→out size | output_num → branch |
//! |------|--------|--------|-----------|-------------|---------------------|
//! | 0    | 5×5    | 4      | 1→20      | 65²→16²     | −1 (intermediate)   |
//! | 1    | 2×2    | 2      | 20→20     | 16²→8²      | 3 → branch_3 (8²)   |
//! | 2    | 2×2    | 2      | 20→20     | 8²→4²       | 2 → branch_2 (4²)   |
//! | 3    | 2×2    | 2      | 20→4      | 4²→2²       | 1 → branch_1 (2²)   |
//! | 4    | 2×2    | 2      | 4→20      | 2²→1²       | 0 → branch_0 (1²)   |
//!
//! Every tensor is channel-major, each channel `h*w` contiguous with
//! `stride == width` (matches C's output-buffer + multi-out layout).
//!
//! Validated bit-exactly against the pure C-scalar CNN via
//! [`aom_sys_ref::ref_intra_cnn_run`] `force_cscalar` (which forces the inner
//! `av1_cnn_convolve_no_maxpool_padding_valid` to `_c`).

use super::weights as w;

/// `CNN_OUT_BUF_SIZE` — branch_0[20] + branch_1[16] + branch_2[320] +
/// branch_3[1280].
pub const CNN_OUT_BUF_SIZE: usize = 1636;

/// One convolution layer of the cascade (square filter + square stride).
struct ConvLayer {
    in_ch: usize,
    out_ch: usize,
    filter: usize,
    stride: usize,
    kernel: &'static [f32],
    bias: &'static [f32],
    /// libaom `output_num`: −1 = intermediate; 0..=3 = the multi-out branch.
    output_num: i32,
}

/// `relu(x) = (x < 0) ? 0 : x` (cnn.c). Distinct from ml.c's `x > 0 ? x : 0`
/// only for NaN, which does not occur here; kept faithful to the CNN source.
#[inline]
fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

/// `(branch_offset, spatial_dim)` in the multi-out buffer for an `output_num`.
/// branch_0 at 0 (1×1), branch_1 at 20 (2×2), branch_2 at 36 (4×4),
/// branch_3 at 356 (8×8) — the exact offsets `intra_mode_cnn_partition` reads
/// (`branch_1 = branch_0 + CNN_BRANCH_0_OUT_SIZE`, ...).
fn branch_region(output_num: i32) -> usize {
    match output_num {
        0 => 0,
        1 => 20,
        2 => 36,
        3 => 356,
        _ => unreachable!("intra-CNN has outputs 0..=3"),
    }
}

/// Port of `av1_cnn_convolve_no_maxpool_padding_valid_c` specialised to
/// `start_idx = 0`, `channel_step = 1`. `input` is channel-major
/// (`in_ch × in_h × in_stride`); `output` is channel-major
/// (`out_ch × out_h × out_w`, stride `out_w`). Weight index walks exactly as C:
/// `off = k*out_ch + i`, then `+= cstep (= in_ch*out_ch)` per filter tap in
/// (row-major) order; accumulation order is k → ii → jj (sequential f32 adds).
#[allow(clippy::too_many_arguments)]
fn conv_valid(
    input: &[f32],
    in_ch: usize,
    in_w: usize,
    in_h: usize,
    in_stride: usize,
    layer: &ConvLayer,
    output: &mut [f32],
    out_w: usize,
    out_h: usize,
) {
    let out_ch = layer.out_ch;
    let filter = layer.filter;
    let skip = layer.stride;
    let cstep = in_ch * out_ch;
    let in_ch_len = in_h * in_stride;
    let out_stride = out_w;
    let out_ch_len = out_h * out_stride;
    let weights = layer.kernel;
    let bias = layer.bias;

    for i in 0..out_ch {
        let mut u = 0usize;
        let mut h = 0usize;
        while h + filter <= in_h {
            let out_row = u * out_stride;
            let mut out_index = out_row;
            let mut wcol = 0usize;
            while wcol + filter <= in_w {
                let mut sum = bias[i];
                for k in 0..in_ch {
                    let mut off = k * out_ch + i;
                    for ii in h..h + filter {
                        let row = k * in_ch_len + ii * in_stride;
                        for jj in wcol..wcol + filter {
                            sum += weights[off] * input[row + jj];
                            off += cstep;
                        }
                    }
                }
                output[i * out_ch_len + out_index] = sum;
                out_index += 1;
                wcol += skip;
            }
            h += skip;
            u += 1;
        }
    }
}

/// Run the intra-CNN cascade on the 65×65 luma window `win` (row-major, stride
/// 65, with replicated top/left borders) and return the multi-out buffer in the
/// exact layout `intra_mode_cnn_partition` reads. `win.len() >= 65*65`.
pub fn cnn_predict(win: &[u8]) -> [f32; CNN_OUT_BUF_SIZE] {
    // Layer-0 input: normalise the 65×65 window to `pixel / 255` (max_val =
    // 255.0f; av1_cnn_predict_img_multi_out, ext = 0 so no engine-side border).
    let mut cur: Vec<f32> = win[..65 * 65].iter().map(|&p| f32::from(p) / 255.0).collect();
    let mut cur_ch = 1usize;
    let mut cur_w = 65usize;
    let mut cur_h = 65usize;
    let mut cur_stride = 65usize;

    let layers = [
        ConvLayer { in_ch: 1, out_ch: 20, filter: 5, stride: 4, kernel: &w::CNN_LAYER_0_KERNEL, bias: &w::CNN_LAYER_0_BIAS, output_num: -1 },
        ConvLayer { in_ch: 20, out_ch: 20, filter: 2, stride: 2, kernel: &w::CNN_LAYER_1_KERNEL, bias: &w::CNN_LAYER_1_BIAS, output_num: 3 },
        ConvLayer { in_ch: 20, out_ch: 20, filter: 2, stride: 2, kernel: &w::CNN_LAYER_2_KERNEL, bias: &w::CNN_LAYER_2_BIAS, output_num: 2 },
        ConvLayer { in_ch: 20, out_ch: 4, filter: 2, stride: 2, kernel: &w::CNN_LAYER_3_KERNEL, bias: &w::CNN_LAYER_3_BIAS, output_num: 1 },
        ConvLayer { in_ch: 4, out_ch: 20, filter: 2, stride: 2, kernel: &w::CNN_LAYER_4_KERNEL, bias: &w::CNN_LAYER_4_BIAS, output_num: 0 },
    ];

    let mut cnn_buffer = [0.0f32; CNN_OUT_BUF_SIZE];
    for layer in &layers {
        debug_assert_eq!(cur_ch, layer.in_ch, "cascade channel-count invariant");
        // av1_find_cnn_layer_output_size, PADDING_VALID.
        let out_w = (cur_w - layer.filter + layer.stride) / layer.stride;
        let out_h = (cur_h - layer.filter + layer.stride) / layer.stride;
        let mut out = vec![0.0f32; layer.out_ch * out_h * out_w];
        conv_valid(&cur, cur_ch, cur_w, cur_h, cur_stride, layer, &mut out, out_w, out_h);
        // Non-linearity (all layers RELU), applied over every element.
        for v in out.iter_mut() {
            *v = relu(*v);
        }
        // Output layers land their (relu'd) tensor in the multi-out buffer AND
        // feed forward (C assigns tensor2 to the output buffer, then swaps).
        if layer.output_num >= 0 {
            let off = branch_region(layer.output_num);
            cnn_buffer[off..off + out.len()].copy_from_slice(&out);
        }
        cur = out;
        cur_ch = layer.out_ch;
        cur_w = out_w;
        cur_h = out_h;
        cur_stride = out_w;
    }
    cnn_buffer
}
