//! `prune_tx_2D` (tx_search.c:1541) — the inter var-tx 2D tx-type NN prune.
//! Fires (from `get_tx_mask_inter`'s multi-type arm) when the inter ext-tx set
//! has > `allowed_tx_count` (5 at PRUNE_1) candidates: two hor/ver NNs score the
//! residual's energy-distribution + horver-correlation features, an outer
//! product + fast-softmax + adaptive threshold keep/prune the tx-type mask and
//! reorder the search `txk_map`.
//!
//! Scope: `prune_2d_txfm_mode == TX_TYPE_PRUNE_1` (the witness config); the
//! `>= TX_TYPE_PRUNE_4` extra cumulative-probability pruning is not reached.
//! All arithmetic transcribed bit-exactly from the C (f32 NN eval + f64 `+0.5`
//! prec-reduce + i64 correlation accumulators + the `av1_sort_fi32_*` networks).

use crate::prune_tx_2d_nn_weights::{PRUNE_TX_2D_HOR, PRUNE_TX_2D_VER, PruneTx2dNn};

const TXS_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TXS_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

/// `EXT_TX_SET_DTT9_IDTX_1DDCT` / `EXT_TX_SET_ALL16` (txfm_common.h).
const EXT_TX_SET_DTT9_IDTX_1DDCT: usize = 4;
const EXT_TX_SET_ALL16: usize = 5;

/// `tx_type_table_2D[16]` (tx_search.c:1547) — the NN's 4x4 (row=ver, col=hor)
/// index space -> TX_TYPE enum value (the search order, != enum order).
const TX_TYPE_TABLE_2D: [usize; 16] = [
    0, 2, 5, 10, // DCT_DCT, DCT_ADST, DCT_FLIPADST, V_DCT
    1, 3, 7, 12, // ADST_DCT, ADST_ADST, ADST_FLIPADST, V_ADST
    4, 8, 6, 14, // FLIPADST_DCT, FLIPADST_ADST, FLIPADST_FLIPADST, V_FLIPADST
    11, 13, 15, 9, // H_DCT, H_ADST, H_FLIPADST, IDTX
];

/// `TX_TYPE_INVALID` (255).
pub const TX_TYPE_INVALID: usize = 255;

/// `av1_nn_output_prec_reduce` (ml.c:19, reduce_prec=1): `((int)(x*512 + 0.5))/512`,
/// the `+0.5` a f64 literal (`x*512` f32, promoted for the add), trunc-to-zero.
#[inline]
fn prec_reduce(x: f32) -> f32 {
    let scaled = x * 512.0f32;
    ((f64::from(scaled) + 0.5) as i32) as f32 / 512.0f32
}

/// `approx_exp` (mathutils.h:129) — the reinterpret-cast fast exp used by
/// `av1_nn_fast_softmax_16`. `A = (1<<23)/ln2` in f32, `(int32)(y*A)` truncates,
/// `+ ((127<<23) - 60801)`, reinterpret bits as f32.
#[inline]
fn approx_exp(y: f32) -> f32 {
    let a = (1i32 << 23) as f32 / 0.693_147_18_f32;
    let bits = ((y * a) as i32).wrapping_add((127i32 << 23) - 60801);
    f32::from_bits(bits as u32)
}

/// `av1_nn_fast_softmax_16_c` (ml.c:159): 16-class softmax via [`approx_exp`].
fn nn_fast_softmax_16(input: &[f32; 16], output: &mut [f32; 16]) {
    let mut max_input = input[0];
    for &v in &input[1..16] {
        if v > max_input {
            max_input = v;
        }
    }
    let mut sum_out = 0.0f32;
    for i in 0..16 {
        let d = input[i] - max_input;
        let normalized = if d > -10.0f32 { d } else { -10.0f32 };
        output[i] = approx_exp(normalized);
        sum_out += output[i];
    }
    for o in output.iter_mut() {
        *o /= sum_out;
    }
}

/// `av1_nn_predict_c` (ml.c:31) for a hor/ver tx-type NN: `num_inputs ->
/// num_hidden (ReLU) -> 4`, then `av1_nn_output_prec_reduce` (reduce_prec=1).
/// Node-major weights `w0[node*num_inputs + i]`, `w1[node*num_hidden + i]`.
fn nn_predict_4(features: &[f32], nn: &PruneTx2dNn, out: &mut [f32; 4]) {
    let (ni, nh) = (nn.num_inputs, nn.num_hidden);
    let mut hidden = [0f32; 16]; // max num_hidden = 16
    for node in 0..nh {
        let mut val = nn.b0[node];
        for i in 0..ni {
            val += nn.w0[node * ni + i] * features[i];
        }
        hidden[node] = if val > 0.0 { val } else { 0.0 };
    }
    for (node, o) in out.iter_mut().enumerate() {
        let mut val = nn.b1[node];
        for i in 0..nh {
            val += nn.w1[node * nh + i] * hidden[i];
        }
        *o = prec_reduce(val);
    }
}

/// `get_energy_distribution_finer` (tx_search.c:1465): fills `hordist[0..hf-1]`
/// and `verdist[0..vf-1]` (hf=esq_w, vf=esq_h) with the normalized column/row
/// energy projections; the last slot of each is written by
/// [`get_horver_correlation_full`]. `diff` is `bw x bh`, stride `stride`.
fn get_energy_distribution_finer(
    diff: &[i16],
    stride: usize,
    bw: usize,
    bh: usize,
    hordist: &mut [f32; 16],
    verdist: &mut [f32; 16],
) {
    let w_shift = usize::from(bw > 8);
    let h_shift = usize::from(bh > 8);
    let esq_w = bw >> w_shift;
    let esq_h = bh >> h_shift;
    let esq_sz = esq_w * esq_h;
    let mut esq = [0u32; 256];

    if w_shift == 1 {
        for i in 0..bh {
            let row = (i >> h_shift) * esq_w;
            let drow = i * stride;
            let mut j = 0;
            while j < bw {
                let a = i32::from(diff[drow + j]);
                let b = i32::from(diff[drow + j + 1]);
                esq[row + (j >> 1)] += (a * a + b * b) as u32;
                j += 2;
            }
        }
    } else {
        for i in 0..bh {
            let row = (i >> h_shift) * esq_w;
            let drow = i * stride;
            for j in 0..bw {
                let a = i32::from(diff[drow + j]);
                esq[row + j] += (a * a) as u32;
            }
        }
    }

    let mut total: u64 = 0;
    for &e in &esq[..esq_sz] {
        total += u64::from(e);
    }

    if total == 0 {
        let hor_val = 1.0f32 / esq_w as f32;
        for h in hordist.iter_mut().take(esq_w - 1) {
            *h = hor_val;
        }
        let ver_val = 1.0f32 / esq_h as f32;
        for v in verdist.iter_mut().take(esq_h - 1) {
            *v = ver_val;
        }
        return;
    }

    let e_recip = 1.0f32 / total as f32;
    for h in hordist.iter_mut().take(esq_w - 1) {
        *h = 0.0;
    }
    for v in verdist.iter_mut().take(esq_h - 1) {
        *v = 0.0;
    }
    for i in 0..esq_h - 1 {
        let row = i * esq_w;
        for j in 0..esq_w - 1 {
            hordist[j] += esq[row + j] as f32;
            verdist[i] += esq[row + j] as f32;
        }
        verdist[i] += esq[row + (esq_w - 1)] as f32; // last column
    }
    let last_row = (esq_h - 1) * esq_w;
    for j in 0..esq_w - 1 {
        hordist[j] += esq[last_row + j] as f32; // last row
    }
    for h in hordist.iter_mut().take(esq_w - 1) {
        *h *= e_recip;
    }
    for v in verdist.iter_mut().take(esq_h - 1) {
        *v *= e_recip;
    }
}

/// `av1_get_horver_correlation_full_c` (rdopt.c:527): the last hfeature/vfeature.
/// i64 accumulators (int16 reads, int products), f32 variance-numerators (exact
/// eval order), f32 `sqrtf`; degenerate guard -> `1.0`.
fn get_horver_correlation_full(diff: &[i16], stride: usize, width: usize, height: usize) -> (f32, f32) {
    let mut x_sum: i64 = 0;
    let mut x2_sum: i64 = 0;
    let mut xy_sum: i64 = 0;
    let mut xz_sum: i64 = 0;
    let (mut x_firstrow, mut x_firstcol) = (0i64, 0i64);
    let (mut x2_firstrow, mut x2_firstcol) = (0i64, 0i64);

    // First row horizontally.
    x_sum += i64::from(diff[0]);
    x2_sum += i64::from(diff[0]) * i64::from(diff[0]);
    x_firstrow += i64::from(diff[0]);
    x2_firstrow += i64::from(diff[0]) * i64::from(diff[0]);
    for j in 1..width {
        let x = i32::from(diff[j]);
        let y = i32::from(diff[j - 1]);
        x_sum += i64::from(x);
        x_firstrow += i64::from(x);
        x2_sum += i64::from(x * x);
        x2_firstrow += i64::from(x * x);
        xy_sum += i64::from(x * y);
    }
    // First column vertically.
    x_firstcol += i64::from(diff[0]);
    x2_firstcol += i64::from(diff[0]) * i64::from(diff[0]);
    for i in 1..height {
        let x = i32::from(diff[i * stride]);
        let z = i32::from(diff[(i - 1) * stride]);
        x_sum += i64::from(x);
        x_firstcol += i64::from(x);
        x2_sum += i64::from(x * x);
        x2_firstcol += i64::from(x * x);
        xz_sum += i64::from(x * z);
    }
    // Interior.
    for i in 1..height {
        for j in 1..width {
            let x = i32::from(diff[i * stride + j]);
            let y = i32::from(diff[i * stride + j - 1]);
            let z = i32::from(diff[(i - 1) * stride + j]);
            x_sum += i64::from(x);
            x2_sum += i64::from(x * x);
            xy_sum += i64::from(x * y);
            xz_sum += i64::from(x * z);
        }
    }

    let (mut x_finalrow, mut x_finalcol) = (0i64, 0i64);
    let (mut x2_finalrow, mut x2_finalcol) = (0i64, 0i64);
    for j in 0..width {
        let v = i64::from(diff[(height - 1) * stride + j]);
        x_finalrow += v;
        x2_finalrow += v * v;
    }
    for i in 0..height {
        let v = i64::from(diff[i * stride + width - 1]);
        x_finalcol += v;
        x2_finalcol += v * v;
    }

    let xhor_sum = x_sum - x_finalcol;
    let xver_sum = x_sum - x_finalrow;
    let y_sum = x_sum - x_firstcol;
    let z_sum = x_sum - x_firstrow;
    let x2hor_sum = x2_sum - x2_finalcol;
    let x2ver_sum = x2_sum - x2_finalrow;
    let y2_sum = x2_sum - x2_firstcol;
    let z2_sum = x2_sum - x2_firstrow;

    let num_hor = (height * (width - 1)) as f32;
    let num_ver = ((height - 1) * width) as f32;

    // `x2 - (x*x)/num`: i64 product exact, cast to f32 for /num, then f32 subtract.
    let xhor_var_n = x2hor_sum as f32 - (xhor_sum * xhor_sum) as f32 / num_hor;
    let xver_var_n = x2ver_sum as f32 - (xver_sum * xver_sum) as f32 / num_ver;
    let y_var_n = y2_sum as f32 - (y_sum * y_sum) as f32 / num_hor;
    let z_var_n = z2_sum as f32 - (z_sum * z_sum) as f32 / num_ver;
    let xy_var_n = xy_sum as f32 - (xhor_sum * y_sum) as f32 / num_hor;
    let xz_var_n = xz_sum as f32 - (xver_sum * z_sum) as f32 / num_ver;

    let hcorr = if xhor_var_n > 0.0 && y_var_n > 0.0 {
        let c = xy_var_n / (xhor_var_n * y_var_n).sqrt();
        if c < 0.0 { 0.0 } else { c }
    } else {
        1.0
    };
    let vcorr = if xver_var_n > 0.0 && z_var_n > 0.0 {
        let c = xz_var_n / (xver_var_n * z_var_n).sqrt();
        if c < 0.0 { 0.0 } else { c }
    } else {
        1.0
    };
    (hcorr, vcorr)
}

/// `prune_2D_adaptive_thresholds[TX_SIZES_ALL]` (tx_search.c:1390) — the ragged
/// per-tx-size threshold table (None where no NN); TX_16X16 has 10 columns, the
/// rest 14. Indexed by `pruning_aggressiveness`.
#[rustfmt::skip]
const PRUNE_2D_THRESHOLDS: [Option<&[f32]>; 19] = [
    Some(&[0.00549, 0.01306, 0.02039, 0.02747, 0.03406, 0.04065, 0.04724, 0.05383, 0.06067, 0.06799, 0.07605, 0.08533, 0.09778, 0.11780]), // TX_4X4
    Some(&[0.00037, 0.00183, 0.00525, 0.01038, 0.01697, 0.02502, 0.03381, 0.04333, 0.05286, 0.06287, 0.07434, 0.08850, 0.10803, 0.14124]), // TX_8X8
    Some(&[0.01404, 0.02000, 0.04211, 0.05164, 0.05798, 0.06335, 0.06897, 0.07629, 0.08875, 0.11169]), // TX_16X16 (10)
    None, None, // 32x32, 64x64
    Some(&[0.00183, 0.00745, 0.01428, 0.02185, 0.02966, 0.03723, 0.04456, 0.05188, 0.05920, 0.06702, 0.07605, 0.08704, 0.10168, 0.12585]), // TX_4X8
    Some(&[0.00085, 0.00476, 0.01135, 0.01892, 0.02698, 0.03528, 0.04358, 0.05164, 0.05994, 0.06848, 0.07849, 0.09021, 0.10583, 0.13123]), // TX_8X4
    Some(&[0.00037, 0.00232, 0.00671, 0.01257, 0.01965, 0.02722, 0.03552, 0.04382, 0.05237, 0.06189, 0.07336, 0.08728, 0.10730, 0.14221]), // TX_8X16
    Some(&[0.00061, 0.00330, 0.00818, 0.01453, 0.02185, 0.02966, 0.03772, 0.04578, 0.05383, 0.06262, 0.07288, 0.08582, 0.10339, 0.13464]), // TX_16X8
    None, None, None, None, // 16x32, 32x16, 32x64, 64x32
    Some(&[0.00232, 0.00671, 0.01257, 0.01941, 0.02673, 0.03430, 0.04211, 0.04968, 0.05750, 0.06580, 0.07507, 0.08655, 0.10242, 0.12878]), // TX_4X16
    Some(&[0.00110, 0.00525, 0.01208, 0.01990, 0.02795, 0.03601, 0.04358, 0.05115, 0.05896, 0.06702, 0.07629, 0.08752, 0.10217, 0.12610]), // TX_16X4
    None, None, None, None, // 8x32, 32x8, 16x64, 64x16
];

/// `get_adaptive_thresholds` (tx_search.c:1448). PRUNE_1 -> aggr row 0 = {4,1}:
/// ALL16 -> col 0 = 4, DTT9 -> col 1 = 1.
fn get_adaptive_thresholds(tx_size: usize, tx_set_type: usize, prune_mode: usize) -> f32 {
    const PRUNE_AGGR: [[usize; 2]; 5] = [[4, 1], [6, 3], [9, 6], [9, 6], [12, 9]];
    let row = prune_mode - 1; // TX_TYPE_PRUNE_1 == 1
    let col = usize::from(tx_set_type != EXT_TX_SET_ALL16); // ALL16 -> 0, DTT9 -> 1
    let aggr = PRUNE_AGGR[row][col];
    PRUNE_2D_THRESHOLDS[tx_size].expect("prune_tx_2D only runs on non-NULL nnconfig sizes")[aggr]
}

/// `SWAP(i,j)` (sorting_network.h:25): compare-exchange putting the larger key at
/// `i` (>= tie-break keeps the lower index's element in the max slot), payload
/// tracks its key. `k[i] < k[j]` (== `!(k[i] >= k[j])`) for finite keys.
#[inline]
fn sw(k: &mut [f32], v: &mut [i32], i: usize, j: usize) {
    if k[i] < k[j] {
        k.swap(i, j);
        v.swap(i, j);
    }
}

/// `av1_sort_fi32_8` (sorting_network.h:118) — 19-comparator descending sort.
fn sort_fi32_8(k: &mut [f32], v: &mut [i32]) {
    for &(i, j) in &[
        (0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (1, 2), (5, 6), (0, 4),
        (3, 7), (1, 5), (2, 6), (1, 4), (3, 6), (2, 4), (3, 5), (3, 4),
    ] {
        sw(k, v, i, j);
    }
}

/// `av1_sort_fi32_16` (sorting_network.h:44) — 60-comparator descending sort.
fn sort_fi32_16(k: &mut [f32], v: &mut [i32]) {
    for &(i, j) in &[
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (0, 2), (1, 3),
        (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15), (1, 2), (5, 6), (0, 4), (3, 7),
        (9, 10), (13, 14), (8, 12), (11, 15), (1, 5), (2, 6), (9, 13), (10, 14), (0, 8), (7, 15),
        (1, 4), (3, 6), (9, 12), (11, 14), (2, 4), (3, 5), (10, 12), (11, 13), (1, 9), (6, 14),
        (3, 4), (11, 12), (1, 8), (2, 10), (5, 13), (7, 14), (3, 11), (2, 8), (4, 12), (7, 13),
        (3, 10), (5, 12), (3, 9), (6, 12), (3, 8), (7, 12), (5, 9), (6, 10), (4, 8), (7, 11),
        (5, 8), (7, 10), (6, 8), (7, 9), (7, 8),
    ] {
        sw(k, v, i, j);
    }
}

/// The result of [`prune_tx_2d`]: the pruned mask + the reordered search order
/// (`txk_map`, with `TX_TYPE_INVALID` padding after the allowed types).
pub struct PruneTx2dResult {
    pub allowed_tx_mask: u16,
    pub txk_map: [usize; 16],
}

/// `prune_tx_2D` (tx_search.c:1541) at `TX_TYPE_PRUNE_1`. `diff` is the txb's
/// residual (`bw x bh`, stride `stride`). Returns the pruned mask + reordered
/// txk_map, or `None` when the prune does not apply (set type or NULL nnconfig)
/// — the caller then keeps its mask + the identity order.
pub fn prune_tx_2d(
    diff: &[i16],
    stride: usize,
    tx_size: usize,
    tx_set_type: usize,
    prune_mode: usize,
    allowed_tx_mask: u16,
) -> Option<PruneTx2dResult> {
    if tx_set_type != EXT_TX_SET_ALL16 && tx_set_type != EXT_TX_SET_DTT9_IDTX_1DDCT {
        return None;
    }
    let nn_hor = PRUNE_TX_2D_HOR[tx_size].as_ref()?;
    let nn_ver = PRUNE_TX_2D_VER[tx_size].as_ref()?;

    let (bw, bh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let hfeatures_num = if bw <= 8 { bw } else { bw / 2 };
    let vfeatures_num = if bh <= 8 { bh } else { bh / 2 };

    let mut hfeatures = [0f32; 16];
    let mut vfeatures = [0f32; 16];
    get_energy_distribution_finer(diff, stride, bw, bh, &mut hfeatures, &mut vfeatures);
    let (hc, vc) = get_horver_correlation_full(diff, stride, bw, bh);
    hfeatures[hfeatures_num - 1] = hc;
    vfeatures[vfeatures_num - 1] = vc;

    let mut hscores = [0f32; 4];
    let mut vscores = [0f32; 4];
    nn_predict_4(&hfeatures[..hfeatures_num], nn_hor, &mut hscores);
    nn_predict_4(&vfeatures[..vfeatures_num], nn_ver, &mut vscores);

    let mut scores_2d_raw = [0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            scores_2d_raw[i * 4 + j] = vscores[i] * hscores[j];
        }
    }
    let mut scores_sm = [0f32; 16];
    nn_fast_softmax_16(&scores_2d_raw, &mut scores_sm);

    let score_thresh = get_adaptive_thresholds(tx_size, tx_set_type, prune_mode);

    let mut max_score_i = 0usize;
    let mut max_score = 0.0f32;
    let mut allow_bitmask = 0u16;
    let mut allow_count = 0usize;
    let mut tx_type_allowed = [TX_TYPE_INVALID as i32; 16];
    let mut scores_2d = [-1.0f32; 16];
    for tx_idx in 0..16 {
        let tx_type = TX_TYPE_TABLE_2D[tx_idx];
        if allowed_tx_mask & (1 << tx_type) == 0 {
            continue;
        }
        if scores_sm[tx_idx] > max_score {
            max_score = scores_sm[tx_idx];
            max_score_i = tx_idx;
        }
        if scores_sm[tx_idx] >= score_thresh {
            allow_bitmask |= 1 << tx_type;
            scores_2d[allow_count] = scores_sm[tx_idx];
            tx_type_allowed[allow_count] = tx_type as i32;
            allow_count += 1;
        }
    }

    // If even the max-score type was pruned, force it and end (raw table order).
    if allow_bitmask & (1 << TX_TYPE_TABLE_2D[max_score_i]) == 0 {
        allow_bitmask |= 1 << TX_TYPE_TABLE_2D[max_score_i];
        return Some(PruneTx2dResult { allowed_tx_mask: allow_bitmask, txk_map: TX_TYPE_TABLE_2D });
    }

    if allow_count <= 8 {
        sort_fi32_8(&mut scores_2d, &mut tx_type_allowed);
    } else {
        sort_fi32_16(&mut scores_2d, &mut tx_type_allowed);
    }
    // prune_2d_txfm_mode >= TX_TYPE_PRUNE_4 extra pruning: not reached at PRUNE_1.

    let mut txk_map = [TX_TYPE_INVALID; 16];
    for (i, &t) in tx_type_allowed.iter().enumerate() {
        txk_map[i] = t as usize;
    }
    Some(PruneTx2dResult { allowed_tx_mask: allow_bitmask, txk_map })
}
