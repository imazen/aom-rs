//! `av1_rd_pick_palette_intra_sby` (av1/encoder/palette.c) — the LUMA palette
//! RD search, plus its self-contained machinery: the dim-1/dim-2 k-means
//! kernels (`k_means_template.h`), colour counting
//! (`av1_count_colors[_highbd]`, intra_mode_search.c:320), dominant-colour
//! extraction (`find_top_colors`), the colour-cache bias
//! (`optimize_palette_colors`), palette-colour signalling cost
//! (`av1_palette_color_cost_y/uv` + `delta_encode_cost`), and the colour-map
//! token rate (`av1_cost_color_map`, tokenize.c:257).
//!
//! Scope/faithfulness notes:
//! - The k-means kernels are ports of the `_c` templates. libaom dispatches
//!   SIMD variants via rtcd for `av1_calc_indices_dim*` / `av1_k_means_dim*`;
//!   those are designed to match the C output (same integer arithmetic), so
//!   the `_c` port is the correct reference behaviour.
//! - `cost_and_tokenize_map`'s context derivation uses
//!   `av1_fast_palette_color_index_context`, an optimized twin of
//!   [`aom_dsp::entropy::partition::get_palette_color_index_context`] (the C
//!   asserts they agree); this port reuses the entropy crate's exact-port
//!   version.
//! - `discount_color_cost` (`rt_sf`) is 0 on every non-realtime path this
//!   port models; it is still threaded for shape.
//! - The UV palette search (`av1_rd_pick_palette_intra_sbuv`) lives in
//!   [`crate::intra_uv_rd`]'s orbit but shares this module's machinery
//!   (dim-2 k-means, colour cost, map cost).

use crate::intra_rd::{IntraSbyBest, IntraSbySearchCfg, WinnerModeEntry, store_winner_mode_stats};
use crate::mode_costs::{
    PaletteCosts, block_signals_txsize, intra_mode_info_cost_y, tx_size_cost, write_uniform_cost,
};
use crate::rd;
use crate::tx_search::{
    BLK_H_B, BLK_W_B, MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, PaletteYrd, TxTypeSearchPolicy, TxfmYrdEnv,
    pick_uniform_tx_size_type_yrd_intra,
};
use aom_dsp::entropy::partition::{
    PaletteNbrKf, get_palette_cache, get_palette_color_index_context, index_color_cache,
};

/// `PALETTE_MAX_SIZE` (blockd.h).
pub const PALETTE_MAX_SIZE: usize = 8;
/// `PALETTE_MIN_SIZE` (blockd.h).
pub const PALETTE_MIN_SIZE: usize = 2;

// ---------------------------------------------------------------------------
// k-means (k_means_template.h, AV1_K_MEANS_DIM in {1, 2})
// ---------------------------------------------------------------------------

/// `lcg_next` (av1/encoder/random.h).
#[inline]
fn lcg_next(state: &mut u32) -> u32 {
    *state = (u64::from(*state) * 1103515245u64 + 12345) as u32;
    *state
}

/// `lcg_rand16` (av1/encoder/random.h): `(lcg_next(state) / 65536) % 32768`.
#[inline]
fn lcg_rand16(state: &mut u32) -> u32 {
    (lcg_next(state) / 65536) % 32768
}

/// `DIVIDE_AND_ROUND(x, y)` (aom_ports/mem.h) for the non-negative sums the
/// centroid update produces.
#[inline]
fn divide_and_round(x: i32, y: i32) -> i32 {
    (x + (y >> 1)) / y
}

/// `calc_dist` (k_means_template.h): dim 1 is L1 (squared only for the total),
/// dim 2 is squared L2.
#[inline]
fn calc_dist(p1: &[i16], p2: &[i16], dim: usize) -> i32 {
    if dim == 1 {
        (i32::from(p1[0]) - i32::from(p2[0])).abs()
    } else {
        let mut dist = 0i32;
        for i in 0..dim {
            let diff = i32::from(p1[i]) - i32::from(p2[i]);
            dist += diff * diff;
        }
        dist
    }
}

/// `av1_calc_indices_dim1/_dim2` (`_c`): nearest-centroid assignment; the
/// returned total distortion squares the per-point L1 in dim 1.
pub fn calc_indices(
    data: &[i16],
    centroids: &[i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
    dim: usize,
) -> i64 {
    let mut total: i64 = 0;
    for i in 0..n {
        let p = &data[i * dim..i * dim + dim];
        let mut min_dist = calc_dist(p, &centroids[0..dim], dim);
        indices[i] = 0;
        for j in 1..k {
            let this_dist = calc_dist(p, &centroids[j * dim..j * dim + dim], dim);
            if this_dist < min_dist {
                min_dist = this_dist;
                indices[i] = j as u8;
            }
        }
        if dim == 1 {
            total += i64::from(min_dist) * i64::from(min_dist);
        } else {
            total += i64::from(min_dist);
        }
    }
    total
}

/// `calc_centroids` (k_means_template.h): per-cluster mean
/// (`DIVIDE_AND_ROUND`); an empty cluster re-seeds from a `lcg_rand16`-chosen
/// data point (state seeded with `(unsigned int)data[0]`).
fn calc_centroids(
    data: &[i16],
    centroids: &mut [i16],
    indices: &[u8],
    n: usize,
    k: usize,
    dim: usize,
) {
    let mut count = [0i32; PALETTE_MAX_SIZE];
    let mut sums = [0i32; 2 * PALETTE_MAX_SIZE];
    // (unsigned int)data[0]: int16 -> int (sign extend) -> u32.
    let mut rand_state = i32::from(data[0]) as u32;
    debug_assert!(n <= 32768);

    for i in 0..n {
        let index = indices[i] as usize;
        debug_assert!(index < k);
        count[index] += 1;
        for j in 0..dim {
            sums[index * dim + j] += i32::from(data[i * dim + j]);
        }
    }

    for i in 0..k {
        if count[i] == 0 {
            let src = (lcg_rand16(&mut rand_state) as usize) % n;
            for j in 0..dim {
                centroids[i * dim + j] = data[src * dim + j];
            }
        } else {
            for j in 0..dim {
                centroids[i * dim + j] = divide_and_round(sums[i * dim + j], count[i]) as i16;
            }
        }
    }
}

/// `av1_k_means_dim1/_dim2` (`_c`): alternate assignment/update up to
/// `max_itr`, ping-ponging two centroid/index buffers; stop on convergence
/// (identical centroids) or a distortion regression (keep the previous side).
pub fn k_means(
    data: &[i16],
    centroids: &mut [i16],
    indices: &mut [u8],
    n: usize,
    k: usize,
    dim: usize,
    max_itr: usize,
) {
    let mut centroids_tmp = [0i16; 2 * PALETTE_MAX_SIZE];
    let mut indices_tmp = vec![0u8; n];
    let mut this_dist;

    debug_assert!(n <= 64 * 64);
    this_dist = calc_indices(data, centroids, indices, n, k, dim);

    // meta_centroids[0] = centroids (caller), [1] = tmp; likewise indices.
    let mut l = 0usize; // which side is CURRENT
    let mut best_l = 0usize;
    let mut i = 0usize;
    while i < max_itr {
        let prev_dist = this_dist;
        let prev_l = l;
        l = if l == 1 { 0 } else { 1 };

        // calc_centroids(data, meta_centroids[l], meta_indices[prev_l], ..)
        {
            let (dst, src_prev): (&mut [i16], &[i16]) = if l == 0 {
                (&mut centroids[..k * dim], &centroids_tmp[..k * dim])
            } else {
                (&mut centroids_tmp[..k * dim], &centroids[..k * dim])
            };
            let idx_prev: &[u8] = if prev_l == 0 { indices } else { &indices_tmp };
            calc_centroids(data, dst, idx_prev, n, k, dim);
            // memcmp(meta_centroids[l], meta_centroids[prev_l], ..)
            if dst[..k * dim] == src_prev[..k * dim] {
                break;
            }
        }
        {
            let cen: &[i16] = if l == 0 { centroids } else { &centroids_tmp };
            let idx: &mut [u8] = if l == 0 { indices } else { &mut indices_tmp };
            this_dist = calc_indices(data, &cen[..k * dim], idx, n, k, dim);
        }

        if this_dist > prev_dist {
            best_l = prev_l;
            break;
        }
        i += 1;
    }
    if i == max_itr {
        best_l = l;
    }
    if best_l != 0 {
        centroids[..k * dim].copy_from_slice(&centroids_tmp[..k * dim]);
        indices[..n].copy_from_slice(&indices_tmp[..n]);
    }
}

// ---------------------------------------------------------------------------
// colour counting + dominant colours
// ---------------------------------------------------------------------------

/// `av1_count_colors` (intra_mode_search.c:320) over this port's u16 planes
/// (bd 8): 256-bin histogram + distinct count.
pub fn count_colors(
    src: &[u16],
    off: usize,
    stride: usize,
    rows: usize,
    cols: usize,
    val_count: &mut [i32],
) -> i32 {
    for v in val_count[..256].iter_mut() {
        *v = 0;
    }
    for r in 0..rows {
        for c in 0..cols {
            let this_val = src[off + r * stride + c] as usize;
            debug_assert!(this_val < 256);
            val_count[this_val] += 1;
        }
    }
    val_count[..256].iter().filter(|&&v| v != 0).count() as i32
}

/// `av1_count_colors_highbd` (intra_mode_search.c:338): full-depth histogram
/// (`val_count`, `1 << bit_depth` bins) for `num_colors`, plus the
/// down-converted 8-bit-domain bin count (`num_color_bins`) that gates the
/// palette path consistently across bit depths.
pub fn count_colors_highbd(
    src: &[u16],
    off: usize,
    stride: usize,
    rows: usize,
    cols: usize,
    bit_depth: i32,
    val_count: &mut [i32],
    num_color_bins: &mut i32,
) -> i32 {
    debug_assert!(bit_depth <= 12);
    let max_pix_val = 1usize << bit_depth;
    let mut bin_val_count = [0i32; 256];
    for v in val_count[..max_pix_val].iter_mut() {
        *v = 0;
    }
    for r in 0..rows {
        for c in 0..cols {
            let raw = src[off + r * stride + c];
            let this_val = (raw >> (bit_depth - 8)) as usize;
            if this_val >= 256 {
                continue;
            }
            bin_val_count[this_val] += 1;
            val_count[raw as usize] += 1;
        }
    }
    *num_color_bins = bin_val_count.iter().filter(|&&v| v != 0).count() as i32;
    val_count[..max_pix_val].iter().filter(|&&v| v != 0).count() as i32
}

/// `find_top_colors` (palette.c:502): the `n_colors` highest-count histogram
/// entries, count-descending (ties index-ascending).
pub fn find_top_colors(count_buf: &[i32], bit_depth: i32, n_colors: usize, top_colors: &mut [i16]) {
    // (index, count), kept sorted by count desc / index asc once full.
    let mut top: [(i32, i32); PALETTE_MAX_SIZE] = [(0, 0); PALETTE_MAX_SIZE];
    let mut n_color_count = 0usize;
    for i in 0..(1usize << bit_depth) {
        let cnt = count_buf[i];
        if cnt > 0 {
            if n_color_count < n_colors {
                top[n_color_count] = (i as i32, cnt);
                n_color_count += 1;
                if n_color_count == n_colors {
                    // qsort(color_count_comp): count desc, index asc.
                    top[..n_colors].sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                }
            } else if cnt > top[n_colors - 1].1 {
                let mut j = n_colors - 1;
                while j >= 1 && cnt > top[j - 1].1 {
                    j -= 1;
                }
                top.copy_within(j..n_colors - 1, j + 1);
                top[j] = (i as i32, cnt);
            }
        }
    }
    debug_assert_eq!(n_color_count, n_colors);
    for i in 0..n_colors {
        top_colors[i] = top[i].0 as i16;
    }
}

/// `fill_data_and_get_bounds` (palette.c:448) over u16 planes.
pub fn fill_data_and_get_bounds(
    src: &[u16],
    off: usize,
    stride: usize,
    rows: usize,
    cols: usize,
    data: &mut [i16],
) -> (i32, i32) {
    let mut lower = i32::from(src[off]);
    let mut upper = lower;
    for r in 0..rows {
        for c in 0..cols {
            let val = i32::from(src[off + r * stride + c]);
            data[r * cols + c] = val as i16;
            lower = lower.min(val);
            upper = upper.max(val);
        }
    }
    (lower, upper)
}

// ---------------------------------------------------------------------------
// centroid post-processing
// ---------------------------------------------------------------------------

/// `remove_duplicates` (palette.c:51): sort + dedup rounded centroids.
pub fn remove_duplicates(centroids: &mut [i16], num_centroids: usize) -> usize {
    centroids[..num_centroids].sort_unstable();
    let mut num_unique = 1usize;
    for i in 1..num_centroids {
        if centroids[i] != centroids[i - 1] {
            centroids[num_unique] = centroids[i];
            num_unique += 1;
        }
    }
    num_unique
}

/// `optimize_palette_colors` (palette.c:203): snap centroids to the nearest
/// cache colour when within `4 << (bd - 8)`.
pub fn optimize_palette_colors(
    color_cache: &[u16],
    n_cache: usize,
    n_colors: usize,
    stride: usize,
    centroids: &mut [i16],
    bit_depth: i32,
) {
    if n_cache == 0 {
        return;
    }
    let mut i = 0usize;
    while i < n_colors * stride {
        let mut min_diff = (i32::from(centroids[i]) - i32::from(color_cache[0])).abs();
        let mut idx = 0usize;
        for j in 1..n_cache {
            let this_diff = (i32::from(centroids[i]) - i32::from(color_cache[j])).abs();
            if this_diff < min_diff {
                min_diff = this_diff;
                idx = j;
            }
        }
        let min_threshold = 4 << (bit_depth - 8);
        if min_diff <= min_threshold {
            centroids[i] = color_cache[idx] as i16;
        }
        i += stride;
    }
}

/// `extend_palette_color_map` (palette.c:180): grow `orig_width x orig_height`
/// to `new_width x new_height` by replicating the last column/row (in place,
/// buffer sized `new_width * new_height`, original data packed at
/// `orig_width` stride).
pub fn extend_palette_color_map(
    color_map: &mut [u8],
    orig_width: usize,
    orig_height: usize,
    new_width: usize,
    new_height: usize,
) {
    debug_assert!(new_width >= orig_width && new_height >= orig_height);
    if new_width == orig_width && new_height == orig_height {
        return;
    }
    for j in (0..orig_height).rev() {
        color_map.copy_within(j * orig_width..j * orig_width + orig_width, j * new_width);
        let edge = color_map[j * new_width + orig_width - 1];
        color_map[j * new_width + orig_width..j * new_width + new_width].fill(edge);
    }
    for j in orig_height..new_height {
        color_map.copy_within(
            (orig_height - 1) * new_width..orig_height * new_width,
            j * new_width,
        );
    }
}

// ---------------------------------------------------------------------------
// signalling costs
// ---------------------------------------------------------------------------

/// `aom_ceil_log2` (aom_ports/bitops.h): 0 for n < 2.
fn ceil_log2(n: i32) -> i32 {
    if n < 2 {
        return 0;
    }
    let mut i = 1;
    let mut p = 2;
    while p < n {
        i += 1;
        p <<= 1;
    }
    i
}

/// `av1_cost_literal(n)`: `n << AV1_PROB_COST_SHIFT` (9).
#[inline]
fn cost_literal(n: i32) -> i32 {
    n * 512
}

/// `delta_encode_cost` (palette.c:65): raw bits to delta-code `colors`
/// (ascending) with `min_val` minimum delta.
fn delta_encode_cost(colors: &[i32], num: usize, bit_depth: i32, min_val: i32) -> i32 {
    if num == 0 {
        return 0;
    }
    let mut bits_cost = bit_depth;
    if num == 1 {
        return bits_cost;
    }
    bits_cost += 2;
    let mut max_delta = 0i32;
    let mut deltas = [0i32; PALETTE_MAX_SIZE];
    let min_bits = bit_depth - 3;
    for i in 1..num {
        let delta = colors[i] - colors[i - 1];
        deltas[i - 1] = delta;
        if delta > max_delta {
            max_delta = delta;
        }
    }
    let mut bits_per_delta = ceil_log2(max_delta + 1 - min_val).max(min_bits);
    let mut range = (1 << bit_depth) - colors[0] - min_val;
    for &d in deltas.iter().take(num - 1) {
        bits_cost += bits_per_delta;
        range -= d;
        bits_per_delta = bits_per_delta.min(ceil_log2(range));
    }
    bits_cost
}

/// `av1_palette_color_cost_y` (palette.c:138): the cache-signal bits + the
/// delta-coded out-of-cache colours, as an `av1_cost_literal` rate.
pub fn palette_color_cost_y(colors: &[u16], color_cache: &[u16], bit_depth: i32) -> i32 {
    let (_found, out_cache, n_out) = index_color_cache(color_cache, colors);
    let total_bits = color_cache.len() as i32 + delta_encode_cost(&out_cache, n_out, bit_depth, 1);
    cost_literal(total_bits)
}

/// `av1_get_palette_delta_bits_v` (palette.c:119): the V-channel wrap-delta
/// bit width + zero-delta count (`min_bits = bd - 4`).
fn palette_delta_bits_v(colors_v: &[u16], n: usize, bit_depth: i32) -> (i32, i32, i32) {
    let max_val = 1 << bit_depth;
    let mut max_d = 0i32;
    let min_bits = bit_depth - 4;
    let mut zero_count = 0i32;
    for i in 1..n {
        let delta = i32::from(colors_v[i]) - i32::from(colors_v[i - 1]);
        let v = delta.abs();
        let d = v.min(max_val - v);
        if d > max_d {
            max_d = d;
        }
        if d == 0 {
            zero_count += 1;
        }
    }
    (ceil_log2(max_d + 1).max(min_bits), zero_count, min_bits)
}

/// `av1_palette_color_cost_uv` (palette.c:152): U cache/delta cost + the
/// cheaper of V raw vs V wrap-delta coding.
pub fn palette_color_cost_uv(
    colors_u: &[u16],
    colors_v: &[u16],
    color_cache: &[u16],
    bit_depth: i32,
) -> i32 {
    let n = colors_u.len();
    let mut total_bits = 0i32;
    let (_found, out_cache, n_out) = index_color_cache(color_cache, colors_u);
    total_bits += color_cache.len() as i32 + delta_encode_cost(&out_cache, n_out, bit_depth, 0);
    let (bits_v, zero_count, _min_bits) = palette_delta_bits_v(colors_v, n, bit_depth);
    let bits_using_delta = 2 + bit_depth + (bits_v + 1) * (n as i32 - 1) - zero_count;
    let bits_using_raw = bit_depth * n as i32;
    total_bits += 1 + bits_using_delta.min(bits_using_raw);
    cost_literal(total_bits)
}

/// `av1_cost_color_map` (tokenize.c:257) for `PALETTE_MAP`: the wavefront
/// colour-index token rate over the visible `rows x cols` region of the
/// (extended, `plane_width`-stride) map, using the per-size/ctx colour-index
/// cost table (`palette_y_color_cost` / `palette_uv_color_cost` row for
/// `n - PALETTE_MIN_SIZE`). The first index is uniform-coded and costed
/// separately (`write_uniform_cost` in the mode-info rate).
pub fn cost_color_map(
    color_map: &[u8],
    plane_width: usize,
    rows: usize,
    cols: usize,
    n: usize,
    color_cost: &[[i32; 8]; 5],
) -> i32 {
    let mut this_rate = 0i32;
    for k in 1..(rows + cols - 1) {
        let j_hi = k.min(cols - 1);
        let j_lo = (k + 1).saturating_sub(rows);
        let mut j = j_hi;
        loop {
            let i = k - j;
            let (_order, color_new_idx, color_ctx) =
                get_palette_color_index_context(color_map, plane_width, i, j, n as i32);
            debug_assert!((color_new_idx as usize) < n);
            this_rate += color_cost[color_ctx][color_new_idx as usize];
            if j == j_lo {
                break;
            }
            j -= 1;
        }
    }
    this_rate
}

// ---------------------------------------------------------------------------
// the luma palette RD search
// ---------------------------------------------------------------------------

/// The palette-Y winner state carried on [`IntraSbyBest`] (the
/// `mbmi->palette_mode_info` Y half + `xd->plane[0].color_index_map`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteYInfo {
    /// `palette_size[0]` (2..=8).
    pub size: usize,
    /// `palette_colors[0..size]` (ascending).
    pub colors: [u16; PALETTE_MAX_SIZE],
    /// The winning colour-index map, `block_width x block_height` (bsize
    /// pixel dims), extended from the visible `rows x cols`.
    pub color_map: Vec<u8>,
}

/// Everything [`rd_pick_palette_intra_sby`] needs beyond the yrd `env`.
pub struct PaletteSearchArgs<'a> {
    /// `bmode_costs[DC_PRED]`.
    pub dc_mode_cost: i32,
    /// The `IntraSbySearchCfg` the enclosing sby search runs under (mode
    /// costs, palette contexts, winner-mode config...).
    pub cfg: &'a IntraSbySearchCfg<'a>,
    /// The first-pass tx policy/method (MODE_EVAL under winner-mode).
    pub pass_pol: &'a TxTypeSearchPolicy,
    pub pass_method: usize,
    /// The palette size/colour-index cost tables.
    pub palette_costs: &'a PaletteCosts,
    /// Above/left neighbour palette state (for `av1_get_palette_cache` and
    /// nothing else — the mode ctx is already on `cfg`).
    pub palette_above: Option<&'a PaletteNbrKf>,
    pub palette_left: Option<&'a PaletteNbrKf>,
    /// `intra_sf.prune_palette_search_level` (allintra: 0 at speed 0, 1 at
    /// speed>=1, 2 at speed>=3).
    pub prune_palette_search_level: i32,
    /// `intra_sf.prune_luma_palette_size_search_level` (allintra: 1 at speed
    /// 0, 2 at speed>=1).
    pub prune_luma_palette_size_search_level: i32,
    /// `rt_sf.discount_color_cost` (0 on all modelled paths).
    pub discount_color_cost: bool,
    /// `x->source_variance` — the same live value the enclosing mode loop
    /// used (the intra tx-size depth sweep's low-contrast prune reads it).
    pub source_variance: u32,
}

/// The palette-use extra rate of `intra_mode_info_cost_y` (its
/// `use_palette` branch, intra_mode_search_utils.h:527-543) recomputed from
/// a stored winner: `palette_y_size_cost + write_uniform_cost(first index) +
/// av1_palette_color_cost_y + av1_cost_color_map`. The colour cache is
/// re-derived exactly as the C does on every call (`av1_get_palette_cache`).
/// Used by the winner-mode second pass; `palette_rd_y` computes the same sum
/// inline (with the cache it already holds).
pub fn palette_y_extra_rate(
    env: &TxfmYrdEnv,
    p: &PaletteYInfo,
    pal_cfg: &crate::intra_rd::PaletteModeCfg,
    bsize_ctx: usize,
) -> i32 {
    let bit_depth = i32::from(env.bd);
    let bsize = env.bsize;
    let block_width = BLK_W_B[bsize];
    let bwv = MI_SIZE_WIDE_B[bsize].min((env.mi_cols - env.mi_col).max(0) as usize);
    let bhv = MI_SIZE_HIGH_B[bsize].min((env.mi_rows - env.mi_row).max(0) as usize);
    let (rows, cols) = (bhv * 4, bwv * 4);

    let mut cache_buf = [0u16; 2 * PALETTE_MAX_SIZE];
    let zero_nbr = PaletteNbrKf::default();
    let above = pal_cfg.above.as_ref().unwrap_or(&zero_nbr);
    let left = pal_cfg.left.as_ref().unwrap_or(&zero_nbr);
    let mb_to_top_edge = -((env.mi_row * 4) * 8);
    let n_cache = get_palette_cache(
        &mut cache_buf,
        0,
        mb_to_top_edge,
        pal_cfg.above.is_some(),
        &above.colors,
        above.size[0],
        pal_cfg.left.is_some(),
        &left.colors,
        left.size[0],
    );
    let color_cache = &cache_buf[..n_cache];

    let mut extra = pal_cfg.costs.palette_y_size_cost[bsize_ctx][p.size - PALETTE_MIN_SIZE]
        + write_uniform_cost(p.size as i32, i32::from(p.color_map[0]));
    extra += palette_color_cost_y(&p.colors[..p.size], color_cache, bit_depth);
    extra += cost_color_map(
        &p.color_map,
        block_width,
        rows,
        cols,
        p.size,
        &pal_cfg.costs.palette_y_color_cost[p.size - PALETTE_MIN_SIZE],
    );
    extra
}

/// The per-candidate shared state `palette_rd_y` mutates.
struct PaletteRdState<'a> {
    best_rd: i64,
    best: Option<IntraSbyBest>,
    /// C `beat_best_rd` — set on ANY palette candidate win (also implied by
    /// `best.is_some()` here since the enclosing search seeds `best: None`
    /// into this struct only when nothing had won yet — the caller merges).
    winner_stats: &'a mut Vec<WinnerModeEntry>,
}

/// `palette_rd_y` (palette.c:229): evaluate ONE candidate colour set.
/// Returns `(beat_best_palette_rd, do_header_rd_based_breakout)`.
#[allow(clippy::too_many_arguments)]
fn palette_rd_y(
    env: &mut TxfmYrdEnv,
    recon: &mut [u16],
    args: &PaletteSearchArgs,
    st: &mut PaletteRdState,
    data: &[i16],
    centroids: &mut [i16],
    n: usize,
    color_cache: &[u16],
    do_header_rd_based_gating: bool,
) -> (bool, bool) {
    let bsize = env.bsize;
    let bit_depth = i32::from(env.bd);
    optimize_palette_colors(color_cache, color_cache.len(), n, 1, centroids, bit_depth);
    let num_unique_colors = remove_duplicates(centroids, n);
    if num_unique_colors < PALETTE_MIN_SIZE {
        // Too few unique colours; DC_PRED covers it.
        return (false, false);
    }
    let max_pix = (1i32 << bit_depth) - 1;
    let mut colors = [0u16; PALETTE_MAX_SIZE];
    for i in 0..num_unique_colors {
        // clip_pixel[_highbd]: centroids are already in range for pixels but
        // the C clips defensively.
        colors[i] = i32::from(centroids[i]).clamp(0, max_pix) as u16;
    }

    // av1_get_block_dimensions(bsize, 0): full block dims + visible crop.
    let block_width = BLK_W_B[bsize];
    let block_height = BLK_H_B[bsize];
    let bwv = MI_SIZE_WIDE_B[bsize].min((env.mi_cols - env.mi_col).max(0) as usize);
    let bhv = MI_SIZE_HIGH_B[bsize].min((env.mi_rows - env.mi_row).max(0) as usize);
    let rows = bhv * 4;
    let cols = bwv * 4;

    // av1_calc_indices over the visible data, then extend to block dims.
    let mut color_map = vec![0u8; block_width * block_height];
    calc_indices(
        data,
        &centroids[..num_unique_colors],
        &mut color_map,
        rows * cols,
        num_unique_colors,
        1,
    );
    extend_palette_color_map(&mut color_map, cols, rows, block_width, block_height);

    // The palette-use mode-info rate (intra_mode_info_cost_y's use_palette
    // branch): size + first-index uniform + colour signalling + map tokens.
    let palette_extra = {
        let bctx = args.cfg.palette_bsize_ctx;
        let mut extra = args.palette_costs.palette_y_size_cost[bctx]
            [num_unique_colors - PALETTE_MIN_SIZE]
            + write_uniform_cost(num_unique_colors as i32, i32::from(color_map[0]));
        extra += palette_color_cost_y(&colors[..num_unique_colors], color_cache, bit_depth);
        if !args.discount_color_cost {
            extra += cost_color_map(
                &color_map,
                block_width,
                rows,
                cols,
                num_unique_colors,
                &args.palette_costs.palette_y_color_cost[num_unique_colors - PALETTE_MIN_SIZE],
            );
        }
        extra
    };
    let palette_mode_rate = intra_mode_info_cost_y(
        args.cfg.mode_costs,
        args.dc_mode_cost,
        0, // DC_PRED
        bsize,
        0,
        false,
        0,
        false,
        args.cfg.try_palette,
        args.cfg.palette_bsize_ctx,
        args.cfg.palette_mode_ctx,
        args.cfg.enable_filter_intra,
        args.cfg.allow_intrabc,
        true, // use_palette
        palette_extra,
    );

    // The palette tx search: DC_PRED with map-fill prediction.
    env.mode = 0;
    env.angle_delta = 0;
    env.use_filter_intra = false;
    env.filter_intra_mode = 0;
    let pal = PaletteYrd {
        colors: &colors,
        size: num_unique_colors,
        map: &color_map,
        map_stride: block_width,
    };

    let (tokenonly, this_rate) = if do_header_rd_based_gating {
        let header_rd = rd::rdcost(env.rdmult, palette_mode_rate, 0);
        // Less aggressive pruning at prune_luma_palette_size_search_level == 1.
        let header_rd_shift = if args.prune_luma_palette_size_search_level == 1 {
            1
        } else {
            0
        };
        if (header_rd >> header_rd_shift) > st.best_rd {
            return (false, true);
        }
        let Some(choice) = pick_uniform_tx_size_type_yrd_intra(
            env,
            recon,
            st.best_rd,
            args.pass_pol,
            args.source_variance,
            args.cfg.enable_tx64,
            args.cfg.enable_rect_tx,
            args.pass_method,
            Some(&pal),
        ) else {
            return (false, false);
        };
        let rate = choice.stats.rate + palette_mode_rate;
        (choice, rate)
    } else {
        let Some(choice) = pick_uniform_tx_size_type_yrd_intra(
            env,
            recon,
            st.best_rd,
            args.pass_pol,
            args.source_variance,
            args.cfg.enable_tx64,
            args.cfg.enable_rect_tx,
            args.pass_method,
            Some(&pal),
        ) else {
            return (false, false);
        };
        let rate = choice.stats.rate + palette_mode_rate;
        (choice, rate)
    };

    let this_rd = rd::rdcost(env.rdmult, this_rate, tokenonly.stats.dist);
    // NOTE (C asymmetry, palette.c:300): NO ALLINTRA variance factor here —
    // palette candidates compare their raw RDCOST against the factored
    // mode-loop best.
    let mut this_rate_tokenonly = tokenonly.stats.rate;
    if !env.lossless && block_signals_txsize(bsize) {
        this_rate_tokenonly -= tx_size_cost(
            env.tx_size_costs,
            env.tx_mode_is_select,
            bsize,
            tokenonly.best_tx_size,
            env.tx_size_ctx,
        );
    }

    // store_winner_mode_stats (palette.c:306): palette candidates feed the
    // multi-winner list too (no-op below speed 4).
    if let Some(wm) = args.cfg.winner_mode {
        store_winner_mode_stats(
            st.winner_stats,
            wm.max_winner_count,
            WinnerModeEntry {
                mode: 0,
                angle_delta: 0,
                use_filter_intra: false,
                filter_intra_mode: 0,
                rd: this_rd,
                palette_y: Some(PaletteYInfo {
                    size: num_unique_colors,
                    colors,
                    color_map: color_map.clone(),
                }),
            },
        );
    }

    if this_rd < st.best_rd {
        st.best_rd = this_rd;
        st.best = Some(IntraSbyBest {
            mode: 0,
            angle_delta: 0,
            tx_size: tokenonly.best_tx_size,
            winners: tokenonly.winners,
            rate: this_rate,
            rate_tokenonly: this_rate_tokenonly,
            dist: tokenonly.stats.dist,
            skippable: tokenonly.stats.skip_txfm,
            best_rd: this_rd,
            use_filter_intra: false,
            filter_intra_mode: 0,
            palette_y: Some(PaletteYInfo {
                size: num_unique_colors,
                colors,
                color_map,
            }),
        });
        (true, false)
    } else {
        (false, false)
    }
}

/// `is_iter_over` (palette.c:326).
#[inline]
fn is_iter_over(curr: i32, end: i32, step: i32) -> bool {
    if step > 0 { curr >= end } else { curr <= end }
}

/// `perform_top_color_palette_search` (palette.c:335). Returns the winning n
/// (`end_n` when nothing won) and stores the last n searched.
#[allow(clippy::too_many_arguments)]
fn perform_top_color_palette_search(
    env: &mut TxfmYrdEnv,
    recon: &mut [u16],
    args: &PaletteSearchArgs,
    st: &mut PaletteRdState,
    data: &[i16],
    top_colors: &[i16],
    start_n: i32,
    end_n: i32,
    step_size: i32,
    do_header_rd_based_gating: bool,
    last_n_searched: &mut i32,
    color_cache: &[u16],
) -> i32 {
    let mut centroids = [0i16; PALETTE_MAX_SIZE];
    let mut n = start_n;
    let mut top_color_winner = end_n;
    debug_assert!(step_size != 0);
    while !is_iter_over(n, end_n, step_size) {
        centroids[..n as usize].copy_from_slice(&top_colors[..n as usize]);
        let (beat_best_palette_rd, breakout) = palette_rd_y(
            env,
            recon,
            args,
            st,
            data,
            &mut centroids,
            n as usize,
            color_cache,
            do_header_rd_based_gating,
        );
        *last_n_searched = n;
        if breakout {
            *last_n_searched = end_n;
            break;
        }
        if beat_best_palette_rd {
            top_color_winner = n;
        } else if args.prune_palette_search_level == 2 {
            return top_color_winner;
        }
        n += step_size;
    }
    top_color_winner
}

/// `perform_k_means_palette_search` (palette.c:382). Same contract as the
/// top-colour search, seeding centroids uniformly in `[lower, upper]` and
/// k-means-refining them per n.
#[allow(clippy::too_many_arguments)]
fn perform_k_means_palette_search(
    env: &mut TxfmYrdEnv,
    recon: &mut [u16],
    args: &PaletteSearchArgs,
    st: &mut PaletteRdState,
    data: &[i16],
    lower_bound: i32,
    upper_bound: i32,
    start_n: i32,
    end_n: i32,
    step_size: i32,
    do_header_rd_based_gating: bool,
    last_n_searched: &mut i32,
    color_cache: &[u16],
    color_map_scratch: &mut [u8],
    data_points: usize,
) -> i32 {
    let mut centroids = [0i16; PALETTE_MAX_SIZE];
    let max_itr = 50usize;
    let mut n = start_n;
    let mut top_color_winner = end_n;
    while !is_iter_over(n, end_n, step_size) {
        for i in 0..n {
            centroids[i as usize] =
                (lower_bound + (2 * i + 1) * (upper_bound - lower_bound) / n / 2) as i16;
        }
        k_means(
            data,
            &mut centroids,
            color_map_scratch,
            data_points,
            n as usize,
            1,
            max_itr,
        );
        let (beat_best_palette_rd, breakout) = palette_rd_y(
            env,
            recon,
            args,
            st,
            data,
            &mut centroids,
            n as usize,
            color_cache,
            do_header_rd_based_gating,
        );
        *last_n_searched = n;
        if breakout {
            *last_n_searched = end_n;
            break;
        }
        if beat_best_palette_rd {
            top_color_winner = n;
        } else if args.prune_palette_search_level == 2 {
            return top_color_winner;
        }
        n += step_size;
    }
    top_color_winner
}

/// `set_stage2_params` (palette.c:432).
fn set_stage2_params(winner: i32, end_n: i32) -> (i32, i32, i32) {
    let min_n = if winner == PALETTE_MIN_SIZE as i32 {
        PALETTE_MIN_SIZE as i32 + 1
    } else {
        (winner - 1).max(PALETTE_MIN_SIZE as i32)
    };
    let max_n = if winner == end_n {
        winner - 1
    } else {
        (winner + 1).min(PALETTE_MAX_SIZE as i32)
    };
    let step_size = (max_n - min_n).max(1);
    (min_n, max_n, step_size)
}

/// `av1_rd_pick_palette_intra_sby` (palette.c:540): the full luma palette
/// search. `best_rd`/`best`/`winner_stats` are the enclosing sby search's
/// running state; a palette win replaces `best` (with `palette_y` set) and
/// tightens `best_rd`. Returns whether any palette candidate won (the C
/// `beat_best_rd` contribution).
#[allow(clippy::too_many_arguments)]
pub fn rd_pick_palette_intra_sby(
    env: &mut TxfmYrdEnv,
    recon: &mut [u16],
    args: &PaletteSearchArgs,
    best_rd: &mut i64,
    best: &mut Option<IntraSbyBest>,
    winner_stats: &mut Vec<WinnerModeEntry>,
) -> bool {
    let bsize = env.bsize;
    let bit_depth = i32::from(env.bd);
    let is_hbd = env.bd > 8;

    // Visible block dims (av1_get_block_dimensions).
    let bwv = MI_SIZE_WIDE_B[bsize].min((env.mi_cols - env.mi_col).max(0) as usize);
    let bhv = MI_SIZE_HIGH_B[bsize].min((env.mi_rows - env.mi_row).max(0) as usize);
    let rows = bhv * 4;
    let cols = bwv * 4;

    let mut count_buf = vec![0i32; 1 << 12];
    let mut colors_threshold = 0i32;
    let colors = if is_hbd {
        count_colors_highbd(
            env.src,
            env.src_off,
            env.src_stride,
            rows,
            cols,
            bit_depth,
            &mut count_buf,
            &mut colors_threshold,
        )
    } else {
        let c = count_colors(
            env.src,
            env.src_off,
            env.src_stride,
            rows,
            cols,
            &mut count_buf,
        );
        colors_threshold = c;
        c
    };

    // x->color_palette_thresh (block.h init: 64; rt-only adjustments off).
    let color_thresh_palette = 64i32;

    let mut st = PaletteRdState {
        best_rd: *best_rd,
        best: None,
        winner_stats,
    };

    if colors_threshold > 1 && colors_threshold <= color_thresh_palette {
        let mut data = vec![0i16; rows * cols];
        let (lower_bound, upper_bound) =
            fill_data_and_get_bounds(env.src, env.src_off, env.src_stride, rows, cols, &mut data);

        // uint16_t color_cache[2 * PALETTE_MAX_SIZE].
        let mut cache_buf = [0u16; 2 * PALETTE_MAX_SIZE];
        let zero_nbr = PaletteNbrKf::default();
        let above = args.palette_above.unwrap_or(&zero_nbr);
        let left = args.palette_left.unwrap_or(&zero_nbr);
        let mb_to_top_edge = -((env.mi_row * 4) * 8);
        let n_cache = get_palette_cache(
            &mut cache_buf,
            0,
            mb_to_top_edge,
            args.palette_above.is_some(),
            &above.colors,
            above.size[0],
            args.palette_left.is_some(),
            &left.colors,
            left.size[0],
        );
        let color_cache = &cache_buf[..n_cache];

        // Dominant colours.
        let mut top_colors = [0i16; PALETTE_MAX_SIZE];
        find_top_colors(
            &count_buf,
            bit_depth,
            (colors as usize).min(PALETTE_MAX_SIZE),
            &mut top_colors,
        );

        let do_header_rd_based_gating = args.prune_luma_palette_size_search_level != 0;
        let mut unused = 0i32;
        let mut color_map_scratch = vec![0u8; rows * cols];

        if args.prune_palette_search_level == 1 && colors > PALETTE_MIN_SIZE as i32 {
            // Coarse two-stage search (speed>=1).
            const START_N: [u8; PALETTE_MAX_SIZE + 1] = [0, 0, 0, 3, 3, 2, 3, 3, 2];
            const STEP: [u8; PALETTE_MAX_SIZE + 1] = [0, 0, 0, 3, 3, 3, 3, 3, 3];
            let max_n = (colors).min(PALETTE_MAX_SIZE as i32);
            let min_n = i32::from(START_N[max_n as usize]);
            let step_size = i32::from(STEP[max_n as usize]);
            let top_color_winner = perform_top_color_palette_search(
                env,
                recon,
                args,
                &mut st,
                &data,
                &top_colors,
                min_n,
                max_n + 1,
                step_size,
                do_header_rd_based_gating,
                &mut unused,
                color_cache,
            );
            if top_color_winner <= max_n {
                let (s2_min, s2_max, s2_step) = set_stage2_params(top_color_winner, max_n);
                perform_top_color_palette_search(
                    env,
                    recon,
                    args,
                    &mut st,
                    &data,
                    &top_colors,
                    s2_min,
                    s2_max + 1,
                    s2_step,
                    false,
                    &mut unused,
                    color_cache,
                );
            }
            let k_means_winner = perform_k_means_palette_search(
                env,
                recon,
                args,
                &mut st,
                &data,
                lower_bound,
                upper_bound,
                min_n,
                max_n + 1,
                step_size,
                do_header_rd_based_gating,
                &mut unused,
                color_cache,
                &mut color_map_scratch,
                rows * cols,
            );
            if k_means_winner <= max_n {
                let (s2_min, s2_max, s2_step) = set_stage2_params(k_means_winner, max_n);
                perform_k_means_palette_search(
                    env,
                    recon,
                    args,
                    &mut st,
                    &data,
                    lower_bound,
                    upper_bound,
                    s2_min,
                    s2_max + 1,
                    s2_step,
                    false,
                    &mut unused,
                    color_cache,
                    &mut color_map_scratch,
                    rows * cols,
                );
            }
        } else {
            // Full ascending + descending search (speed 0 and level-2 prune).
            let max_n = (colors).min(PALETTE_MAX_SIZE as i32);
            let min_n = PALETTE_MIN_SIZE as i32;
            let mut last_n_searched = min_n;
            perform_top_color_palette_search(
                env,
                recon,
                args,
                &mut st,
                &data,
                &top_colors,
                min_n,
                max_n + 1,
                1,
                do_header_rd_based_gating,
                &mut last_n_searched,
                color_cache,
            );
            if last_n_searched < max_n {
                perform_top_color_palette_search(
                    env,
                    recon,
                    args,
                    &mut st,
                    &data,
                    &top_colors,
                    max_n,
                    last_n_searched,
                    -1,
                    false,
                    &mut unused,
                    color_cache,
                );
            }
            if colors == PALETTE_MIN_SIZE as i32 {
                // The two colours ARE the centroids.
                let mut centroids = [0i16; PALETTE_MAX_SIZE];
                centroids[0] = lower_bound as i16;
                centroids[1] = upper_bound as i16;
                palette_rd_y(
                    env,
                    recon,
                    args,
                    &mut st,
                    &data,
                    &mut centroids,
                    2,
                    color_cache,
                    false,
                );
            } else {
                let mut last_n = min_n;
                perform_k_means_palette_search(
                    env,
                    recon,
                    args,
                    &mut st,
                    &data,
                    lower_bound,
                    upper_bound,
                    min_n,
                    max_n + 1,
                    1,
                    do_header_rd_based_gating,
                    &mut last_n,
                    color_cache,
                    &mut color_map_scratch,
                    rows * cols,
                );
                if last_n < max_n {
                    perform_k_means_palette_search(
                        env,
                        recon,
                        args,
                        &mut st,
                        &data,
                        lower_bound,
                        upper_bound,
                        max_n,
                        last_n,
                        -1,
                        false,
                        &mut unused,
                        color_cache,
                        &mut color_map_scratch,
                        rows * cols,
                    );
                }
            }
        }
    }

    let won = st.best.is_some();
    if won {
        *best_rd = st.best_rd;
        *best = st.best;
    }
    won
}

// ---------------------------------------------------------------------------
// the chroma (UV) palette RD search
// ---------------------------------------------------------------------------

/// `av1_get_block_dimensions(bsize, plane=1)` (blockd.h): the chroma plane
/// block dims + visible crop, INCLUDING the chroma sub-8x8 correction
/// (`is_chroma_sub8_*` adds 2 — the merged chroma-ref block of a 4-wide/high
/// luma block, reachable for palette via BLOCK_4X16/16X4). Returns
/// `(plane_block_width, plane_block_height, rows, cols)`.
pub fn chroma_block_dims(
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    mi_rows: i32,
    mi_cols: i32,
    ss_x: usize,
    ss_y: usize,
) -> (usize, usize, usize, usize) {
    let (bw_px, bh_px) = (BLK_W_B[bsize] as i32, BLK_H_B[bsize] as i32);
    let mb_to_right_edge = (mi_cols - MI_SIZE_WIDE_B[bsize] as i32 - mi_col) * 4 * 8;
    let mb_to_bottom_edge = (mi_rows - MI_SIZE_HIGH_B[bsize] as i32 - mi_row) * 4 * 8;
    let block_cols_px = if mb_to_right_edge >= 0 {
        bw_px
    } else {
        (mb_to_right_edge >> 3) + bw_px
    };
    let block_rows_px = if mb_to_bottom_edge >= 0 {
        bh_px
    } else {
        (mb_to_bottom_edge >> 3) + bh_px
    };
    let pw = (bw_px >> ss_x) as usize;
    let ph = (bh_px >> ss_y) as usize;
    let sub8_x = usize::from(pw < 4) * 2;
    let sub8_y = usize::from(ph < 4) * 2;
    (
        pw + sub8_x,
        ph + sub8_y,
        (block_rows_px >> ss_y) as usize + sub8_y,
        (block_cols_px >> ss_x) as usize + sub8_x,
    )
}

/// The palette-UV winner state (the `mbmi->palette_mode_info` UV half +
/// `xd->plane[1].color_index_map`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteUvInfo {
    /// `palette_size[1]` (2..=8).
    pub size: usize,
    /// U / V channel colours (`palette_colors[8..8+n]` / `[16..16+n]`),
    /// U-ascending pair order.
    pub colors_u: [u16; PALETTE_MAX_SIZE],
    pub colors_v: [u16; PALETTE_MAX_SIZE],
    /// The winning colour-index map over the CHROMA plane dims
    /// (`plane_block_width x plane_block_height`).
    pub color_map: Vec<u8>,
}

/// Everything `av1_rd_pick_palette_intra_sbuv` needs beyond the [`UvRdEnv`].
pub struct UvPaletteArgs<'a> {
    /// `intra_uv_mode_cost[cfl_allowed][y_mode][UV_DC_PRED]`.
    pub dc_mode_cost: i32,
    /// The palette size/colour-index cost tables.
    pub costs: &'a PaletteCosts,
    /// Above/left neighbour palette state (`av1_get_palette_cache`, plane 1).
    pub above: Option<PaletteNbrKf>,
    pub left: Option<PaletteNbrKf>,
    /// `av1_get_palette_bsize_ctx(bsize)`.
    pub bsize_ctx: usize,
    /// `try_palette` (the flag-cost gate — always true when this struct
    /// exists) + whether the LUMA winner uses palette (the UV flag's ctx).
    pub y_palette_active: bool,
    /// sf `intra_sf.early_term_chroma_palette_size_search` (allintra: 1 at
    /// every speed, :364).
    pub early_term: bool,
    /// The tx search policy the uv loop ran under.
    pub pol: &'a crate::tx_search::TxTypeSearchPolicy,
}

/// `av1_rd_pick_palette_intra_sbuv` (palette.c:763): the chroma palette
/// search — ascending palette-size loop over dim-2 k-means (U,V) pairs.
/// Updates `best` (with `palette_uv` set) and its `best_rd` on a win.
#[allow(clippy::too_many_arguments)]
pub fn rd_pick_palette_intra_sbuv(
    env: &crate::intra_uv_rd::UvRdEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    args: &UvPaletteArgs,
    uv_mode_costs: &crate::mode_costs::IntraModeCosts,
    best: &mut crate::intra_uv_rd::UvModeResult,
) {
    let bsize = env.bsize;
    let bit_depth = i32::from(env.bd);
    let is_hbd = env.bd > 8;

    // av1_get_block_dimensions(bsize, 1): plane dims + the visible crop
    // (incl. the chroma sub-8x8 +2 correction for 4x16/16x4 luma blocks).
    let (plane_block_width, plane_block_height, rows, cols) = chroma_block_dims(
        bsize,
        env.mi_row,
        env.mi_col,
        env.mi_rows,
        env.mi_cols,
        env.ss_x,
        env.ss_y,
    );

    let src_stride = env.src_stride;
    let (off_u, off_v) = (env.src_off[0], env.src_off[1]);

    // Colour counts per channel (the 8-bit-domain threshold for hbd).
    let mut count_buf = vec![0i32; 1 << 12];
    let (mut thr_u, mut thr_v) = (0i32, 0i32);
    let colors_u = if is_hbd {
        count_colors_highbd(
            env.src_u,
            off_u,
            src_stride,
            rows,
            cols,
            bit_depth,
            &mut count_buf,
            &mut thr_u,
        )
    } else {
        let c = count_colors(env.src_u, off_u, src_stride, rows, cols, &mut count_buf);
        thr_u = c;
        c
    };
    let colors_v = if is_hbd {
        count_colors_highbd(
            env.src_v,
            off_v,
            src_stride,
            rows,
            cols,
            bit_depth,
            &mut count_buf,
            &mut thr_v,
        )
    } else {
        let c = count_colors(env.src_v, off_v, src_stride, rows, cols, &mut count_buf);
        thr_v = c;
        c
    };

    // uint16_t color_cache[2 * PALETTE_MAX_SIZE]; av1_get_palette_cache(xd, 1).
    let mut cache_buf = [0u16; 2 * PALETTE_MAX_SIZE];
    let zero_nbr = PaletteNbrKf::default();
    let above = args.above.as_ref().unwrap_or(&zero_nbr);
    let left = args.left.as_ref().unwrap_or(&zero_nbr);
    let mb_to_top_edge = -((env.mi_row * 4) * 8);
    let n_cache = get_palette_cache(
        &mut cache_buf,
        1,
        mb_to_top_edge,
        args.above.is_some(),
        &above.colors,
        above.size[1],
        args.left.is_some(),
        &left.colors,
        left.size[1],
    );
    let color_cache = &cache_buf[..n_cache];

    let colors_threshold = thr_u.max(thr_v);
    if !(colors_threshold > 1 && colors_threshold <= 64) {
        return;
    }

    let max_itr = 50usize;
    // Interleaved (u, v) data + per-channel bounds.
    let mut data = vec![0i16; rows * cols * 2];
    let (mut lb_u, mut ub_u) = (i32::from(env.src_u[off_u]), i32::from(env.src_u[off_u]));
    let (mut lb_v, mut ub_v) = (i32::from(env.src_v[off_v]), i32::from(env.src_v[off_v]));
    for r in 0..rows {
        for c in 0..cols {
            let val_u = i32::from(env.src_u[off_u + r * src_stride + c]);
            let val_v = i32::from(env.src_v[off_v + r * src_stride + c]);
            data[(r * cols + c) * 2] = val_u as i16;
            data[(r * cols + c) * 2 + 1] = val_v as i16;
            if val_u < lb_u {
                lb_u = val_u;
            } else if val_u > ub_u {
                ub_u = val_u;
            }
            if val_v < lb_v {
                lb_v = val_v;
            } else if val_v > ub_v {
                ub_v = val_v;
            }
        }
    }

    let colors = colors_u.max(colors_v);
    let max_colors = colors.min(PALETTE_MAX_SIZE as i32);
    let max_pix = (1i32 << bit_depth) - 1;
    let mut centroids = [0i16; 2 * PALETTE_MAX_SIZE];
    let mut color_map = vec![0u8; (plane_block_width * plane_block_height).max(rows * cols)];

    for n in (PALETTE_MIN_SIZE as i32)..=max_colors {
        let n = n as usize;
        for i in 0..n {
            centroids[i * 2] = (lb_u + (2 * i as i32 + 1) * (ub_u - lb_u) / n as i32 / 2) as i16;
            centroids[i * 2 + 1] =
                (lb_v + (2 * i as i32 + 1) * (ub_v - lb_v) / n as i32 / 2) as i16;
        }
        k_means(
            &data,
            &mut centroids,
            &mut color_map,
            rows * cols,
            n,
            2,
            max_itr,
        );
        optimize_palette_colors(color_cache, n_cache, n, 2, &mut centroids, bit_depth);
        // Sort the U channel colours ascending (selection sort keeping pairs).
        for i in (0..2 * (n - 1)).step_by(2) {
            let mut min_idx = i;
            let mut min_val = centroids[i];
            let mut j = i + 2;
            while j < 2 * n {
                if centroids[j] < min_val {
                    min_val = centroids[j];
                    min_idx = j;
                }
                j += 2;
            }
            if min_idx != i {
                centroids.swap(i, min_idx);
                centroids.swap(i + 1, min_idx + 1);
            }
        }
        calc_indices(
            &data,
            &centroids[..2 * n],
            &mut color_map,
            rows * cols,
            n,
            2,
        );
        extend_palette_color_map(
            &mut color_map,
            cols,
            rows,
            plane_block_width,
            plane_block_height,
        );

        let mut colors_u_arr = [0u16; PALETTE_MAX_SIZE];
        let mut colors_v_arr = [0u16; PALETTE_MAX_SIZE];
        for j in 0..n {
            colors_u_arr[j] = i32::from(centroids[j * 2]).clamp(0, max_pix) as u16;
            colors_v_arr[j] = i32::from(centroids[j * 2 + 1]).clamp(0, max_pix) as u16;
        }

        // intra_mode_info_cost_uv, use_palette arm.
        let palette_extra = args.costs.palette_uv_size_cost[args.bsize_ctx][n - PALETTE_MIN_SIZE]
            + write_uniform_cost(n as i32, i32::from(color_map[0]))
            + palette_color_cost_uv(
                &colors_u_arr[..n],
                &colors_v_arr[..n],
                color_cache,
                bit_depth,
            )
            + cost_color_map(
                &color_map,
                plane_block_width,
                rows,
                cols,
                n,
                &args.costs.palette_uv_color_cost[n - PALETTE_MIN_SIZE],
            );
        let palette_mode_rate = crate::mode_costs::intra_mode_info_cost_uv(
            uv_mode_costs,
            args.dc_mode_cost,
            0, // UV_DC_PRED
            bsize,
            0,
            true, // try_palette (this search only runs under it)
            args.y_palette_active,
            true, // use_palette
            palette_extra,
        );

        let pal_pred = crate::intra_uv_rd::PaletteUvPred {
            colors_u: &colors_u_arr,
            colors_v: &colors_v_arr,
            size: n,
            map: &color_map,
            map_stride: plane_block_width,
        };
        let (tokenonly_rate, tokenonly_dist, tokenonly_skip);
        if args.early_term {
            let header_rd = rd::rdcost(env.rdmult, palette_mode_rate, 0);
            // Terminate further palette_size search (palette.c:906): >= best.
            if header_rd >= best.best_rd {
                break;
            }
            let Some((stats, _wu, _wv)) = crate::intra_uv_rd::txfm_uvrd_p(
                env,
                recon_u,
                recon_v,
                0, // UV_DC_PRED
                0,
                best.best_rd,
                args.pol,
                Some(&pal_pred),
            ) else {
                continue;
            };
            if stats.rate == i32::MAX {
                continue;
            }
            tokenonly_rate = stats.rate;
            tokenonly_dist = stats.dist;
            tokenonly_skip = stats.skip_txfm;
        } else {
            let Some((stats, _wu, _wv)) = crate::intra_uv_rd::txfm_uvrd_p(
                env,
                recon_u,
                recon_v,
                0,
                0,
                best.best_rd,
                args.pol,
                Some(&pal_pred),
            ) else {
                continue;
            };
            if stats.rate == i32::MAX {
                continue;
            }
            tokenonly_rate = stats.rate;
            tokenonly_dist = stats.dist;
            tokenonly_skip = stats.skip_txfm;
        }

        let this_rate = tokenonly_rate + palette_mode_rate;
        let this_rd = rd::rdcost(env.rdmult, this_rate, tokenonly_dist);
        if this_rd < best.best_rd {
            *best = crate::intra_uv_rd::UvModeResult {
                uv_mode: 0, // UV_DC_PRED
                angle_delta_uv: 0,
                cfl_alpha_idx: 0,
                cfl_alpha_signs: 0,
                rate: this_rate,
                rate_tokenonly: tokenonly_rate,
                dist: tokenonly_dist,
                skippable: tokenonly_skip,
                best_rd: this_rd,
                palette_uv: Some(PaletteUvInfo {
                    size: n,
                    colors_u: colors_u_arr,
                    colors_v: colors_v_arr,
                    color_map: color_map[..plane_block_width * plane_block_height].to_vec(),
                }),
            };
        }
    }
}
