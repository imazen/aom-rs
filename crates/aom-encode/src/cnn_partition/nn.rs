//! Port of `av1/encoder/ml.c` `av1_nn_predict_c` + `av1_nn_output_prec_reduce`
//! — the fully-connected DNN forward pass that `intra_mode_cnn_partition`
//! (the speed>=1 intra CNN partition prune) runs on its assembled features to
//! produce the split/no-split logits.
//!
//! Bit-exact transcription of the C: sequential accumulation (`val = bias;
//! val += w*in` in source order, no reassociation), ReLU on hidden layers, a
//! linear output layer, then — when `reduce_prec` — the 1/512 output
//! quantisation libaom uses to keep C and SIMD agreeing. The single caller in
//! this port (the CNN partition prune) passes `reduce_prec = true`, exactly as
//! `av1_nn_predict(dnn_features, dnn_config, 1, logits)` does.
//!
//! Validated against the REAL `av1_nn_predict_c` via `ref_nn_predict`
//! (`rd_shim.c` `shim_nn_predict`) over randomised shapes + weights in
//! `tests/cnn_partition_nn_diff.rs`.

/// `NN_MAX_NODES_PER_LAYER` (ml.h:22). Hidden layers never exceed this.
const NN_MAX_NODES_PER_LAYER: usize = 128;

/// `av1_nn_output_prec_reduce` (ml.c) — quantise each output to `prec_bits = 9`
/// fractional bits. Transcribed with the C's exact float/double promotion:
/// `output[i] * prec` is `float * int` (→ f32), `+ 0.5` promotes to double, the
/// `(int)` cast truncates toward zero, and `* inv_prec` is `int * float` (→ f32).
fn nn_output_prec_reduce(output: &mut [f32]) {
    const PREC: f32 = 512.0; // 1 << 9
    // inv_prec = (float)(1.0 / prec); 1/512 is exactly representable in f32.
    const INV_PREC: f32 = (1.0f64 / 512.0f64) as f32;
    for o in output.iter_mut() {
        // (int)(output[i] * prec + 0.5): the multiply is f32, the +0.5 is f64.
        let q = (f64::from(*o * PREC) + 0.5f64) as i32;
        *o = (q as f32) * INV_PREC;
    }
}

/// One ReLU-activated fully-connected layer: `output[node] = relu(bias[node] +
/// sum_i weights[node*num_in + i] * input[i])`. Accumulation order matches C.
fn relu_layer(input: &[f32], weights: &[f32], bias: &[f32], num_out: usize, output: &mut [f32]) {
    let num_in = input.len();
    for node in 0..num_out {
        let mut val = bias[node];
        let row = &weights[node * num_in..node * num_in + num_in];
        for i in 0..num_in {
            val += row[i] * input[i];
        }
        // ReLU: `val > 0.0f ? val : 0.0f` (ml.c). NaN -> 0.0 either way.
        output[node] = if val > 0.0 { val } else { 0.0 };
    }
}

/// Port of `av1_nn_predict_c` (ml.c). `hidden_nodes[l]` = the node count of
/// hidden layer `l`; `weights[l]` / `biases[l]` are that layer's tables
/// (`weights[l][node*num_in + i]`, `biases[l][node]`), and the final entry
/// (index `hidden_nodes.len()`) is the linear output layer producing
/// `num_outputs` values into `output`. `reduce_prec` applies the 1/512 output
/// quantisation. `weights.len() == biases.len() == hidden_nodes.len() + 1`.
pub fn nn_predict(
    features: &[f32],
    hidden_nodes: &[usize],
    weights: &[&[f32]],
    biases: &[&[f32]],
    num_outputs: usize,
    reduce_prec: bool,
    output: &mut [f32],
) {
    let num_hidden = hidden_nodes.len();
    debug_assert_eq!(weights.len(), num_hidden + 1);
    debug_assert_eq!(biases.len(), num_hidden + 1);

    // Ping-pong buffers (C uses `float buf[2][NN_MAX_NODES_PER_LAYER]`). Two
    // distinct locals so each layer reads one and writes the other without a
    // borrow conflict.
    let mut a = [0.0f32; NN_MAX_NODES_PER_LAYER];
    let mut b = [0.0f32; NN_MAX_NODES_PER_LAYER];
    a[..features.len()].copy_from_slice(features);
    let mut cur_len = features.len();
    let mut input_in_a = true;

    for layer in 0..num_hidden {
        let n_out = hidden_nodes[layer];
        debug_assert!(n_out <= NN_MAX_NODES_PER_LAYER);
        if input_in_a {
            let (inp, out) = (&a, &mut b);
            relu_layer(&inp[..cur_len], weights[layer], biases[layer], n_out, out);
        } else {
            let (inp, out) = (&b, &mut a);
            relu_layer(&inp[..cur_len], weights[layer], biases[layer], n_out, out);
        }
        cur_len = n_out;
        input_in_a = !input_in_a;
    }

    // Final (linear, no ReLU) output layer.
    let input = if input_in_a { &a[..cur_len] } else { &b[..cur_len] };
    let final_w = weights[num_hidden];
    let final_b = biases[num_hidden];
    for node in 0..num_outputs {
        let mut val = final_b[node];
        let row = &final_w[node * cur_len..node * cur_len + cur_len];
        for i in 0..cur_len {
            val += row[i] * input[i];
        }
        output[node] = val;
    }

    if reduce_prec {
        nn_output_prec_reduce(&mut output[..num_outputs]);
    }
}
