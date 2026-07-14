//! Shared C-side chain helpers for the aom-encode differential harnesses —
//! the transcribed loop skeletons driving REAL reference pieces
//! (`c_search_tx_type`, `c_uniform_txfm_yrd`, `c_pick_uniform_tx_size_type_yrd`,
//! `c_intra_model_rd`) plus the common Rng / cost-table / CDF generators.
//! Moved verbatim out of uniform_txfm_yrd_diff.rs / intra_model_rd_diff.rs
//! (each test binary uses a subset).
#![allow(dead_code)]

use aom_encode::tx_search::TX_SIZE_2D_TBL;
use aom_sys_ref as c;
use aom_txb::{scan, txb_high, txb_wide};

pub const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
pub const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
pub const BLK_W: [usize; 22] =
    [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
pub const BLK_H: [usize; 22] =
    [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
pub const VAR_IDX: [usize; 19] = [0, 4, 9, 14, 18, 1, 3, 5, 8, 10, 13, 15, 17, 2, 7, 6, 12, 11, 16];
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    pub fn cost(&mut self) -> i32 {
        self.range(0, 20 << 9)
    }
}
pub fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

pub fn cdf_row4(rng: &mut Rng, nsymbs: usize) -> [u16; 4] {
    let mut row = [0u16; 4];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, 32000 / nsymbs as i32) as u32;
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

pub fn gen_cdfs(rng: &mut Rng, count: usize, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(count * padded);
    for _ in 0..count {
        let mut row = vec![0u16; padded];
        let mut acc: u32 = 0;
        for e in row.iter_mut().take(nsymbs - 1) {
            acc += rng.range(1, (32000 / nsymbs as i32).max(2)) as u32;
            *e = (32768u32.saturating_sub(acc)).max(1) as u16;
        }
        row[nsymbs - 1] = 0;
        v.extend_from_slice(&row);
    }
    v
}
/// C-side search_tx_type for one txb (the chain of REAL pieces; loop control
/// transcribed from tx_search.c 2199-2363). Returns the winner
/// (tx_type, eob, rate, dist, sse, entropy_ctx, dqcoeff, rd).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_search_tx_type(
    residual: &[i16],
    pred: &[u16],
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    tx_size: usize,
    mode: usize,
    use_fi: bool,
    fi_mode: usize,
    lossless: bool,
    reduced: bool,
    bd: u8,
    plane_rows_c: &[i16],
    dequant: [i16; 2],
    t_above: &[i8],
    t_left: &[i8],
    bsize: usize,
    rdmult: i32,
    ref_best_rd: i64,
    coeff_tbls: (&[i32], &[i32], &[i32], &[i32], &[i32], &[i32], &[i32]),
    ttc_tables: (&[i32], &[i32]),
) -> (usize, u16, i32, i64, i64, u8, Vec<i32>, i64) {
    c_search_tx_type_p(
        0, 0, residual, pred, src, src_off, src_stride, tx_size, mode, use_fi, fi_mode,
        lossless, reduced, bd, plane_rows_c, dequant, t_above, t_left, bsize, rdmult,
        ref_best_rd, coeff_tbls, ttc_tables,
    )
}

/// BLOCK_SIZE with the same dims as a TX_SIZE.
pub fn tx_to_bsize(tx_size: usize) -> usize {
    const T: [usize; 19] = [0, 3, 6, 9, 12, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21];
    T[tx_size]
}
/// C-side `uniform_txfm_yrd` for one size: the full walk + intra assembly.
/// Returns `(rd, Some((rate, dist, sse, winners)))` or `(MAX, None)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_uniform_txfm_yrd(
    bsize: usize,
    tx_size: usize,
    geometry: (i32, i32, usize, usize, usize),  // mi_row, mi_col, ref_off, src_off, stride
    recon_c: &mut [u16],
    src: &[u16],
    mode: usize,
    angle_delta: i32,
    use_fi: bool,
    fi_mode: usize,
    lossless: bool,
    reduced: bool,
    bd: u8,
    plane_rows_c: &[i16],
    dequant: [i16; 2],
    above_ctx: &[i8],
    left_ctx: &[i8],
    rdmult: i32,
    ref_best_rd: i64,
    coeff_tbls: (&[i32], &[i32], &[i32], &[i32], &[i32], &[i32], &[i32]),
    ttc_tables: (&[i32], &[i32]),
    skip_costs: &[[i32; 2]; 3],
    skip_ctx: usize,
    ts_flat: &[i32],
    tx_size_ctx: usize,
) -> (i64, Option<(i32, i64, i64, Vec<(usize, u16, u8)>)>) {
    let (mi_row, mi_col, ref_off, src_off, stride) = geometry;
    let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
    // tx_mode_is_select = !lossless (select_tx_mode: lossless => ONLY_4X4).
    let tx_size_rate =
        c::ref_tx_size_cost(ts_flat, !lossless, bsize as i32, tx_size as i32, tx_size_ctx as i32);
    let no_skip_rate = skip_costs[skip_ctx][0];
    let no_this_rd = c::ref_rdcost(rdmult, no_skip_rate + tx_size_rate, 0);
    if no_this_rd > ref_best_rd {
        return (i64::MAX, None);
    }
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let (txwu, txhu) = (txw >> 2, txh >> 2);
    let mut t_above = above_ctx[..bw >> 2].to_vec();
    let mut t_left = left_ctx[..bh >> 2].to_vec();
    let mut rate_sum: i64 = 0;
    let mut dist_sum: i64 = 0;
    let mut sse_sum: i64 = 0;
    let mut winners: Vec<(usize, u16, u8)> = Vec::new();
    let mut current_rd = no_this_rd;
    let mut invalid = false;
    'walk: for blk_row in (0..bh >> 2).step_by(txhu) {
        for blk_col in (0..bw >> 2).step_by(txwu) {
            if invalid {
                break 'walk;
            }
            let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                12, bsize, mi_row, mi_col, true, true, 1 << 16, 1 << 16, 0, tx_size, 0, 0,
                blk_row as i32, blk_col as i32, bw as i32, bh as i32, 512, 512, mode,
                angle_delta * 3, use_fi,
            );
            let txb_off = ref_off + (blk_row * stride + blk_col) * 4;
            let pred = c::ref_hbd_predict_intra(
                recon_c, txb_off, stride, mode, angle_delta * 3, use_fi, fi_mode, false, 0,
                tx_size, txw, txh, n_top, n_tr, n_left, n_bl, bd as i32,
            );
            for r in 0..txh {
                recon_c[txb_off + r * stride..txb_off + r * stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }
            let src_txb_off = src_off + (blk_row * stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            c::ref_highbd_subtract_block(
                txh, txw, &mut residual, txw, &src[src_txb_off..], stride, &pred, txw,
            );
            let (wtype, weob, wrate, wdist, wsse, wctx, wdqc, _wrd) = c_search_tx_type(
                &residual, &pred, src, src_txb_off, stride, tx_size, mode, use_fi, fi_mode,
                lossless, reduced, bd, plane_rows_c, dequant, &t_above[blk_col..],
                &t_left[blk_row..], bsize, rdmult, ref_best_rd - current_rd, coeff_tbls,
                ttc_tables,
            );
            if weob > 0 {
                let mut tight = pred.clone();
                c::ref_inv_txfm2d_add(tx_size, &wdqc, &mut tight, txw, wtype, bd as i32);
                for r in 0..txh {
                    recon_c[txb_off + r * stride..txb_off + r * stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }
            for a in t_above[blk_col..blk_col + txwu].iter_mut() {
                *a = wctx as i8;
            }
            for l in t_left[blk_row..blk_row + txhu].iter_mut() {
                *l = wctx as i8;
            }
            winners.push((wtype, weob, wctx));
            rate_sum += i64::from(wrate);
            dist_sum += wdist;
            sse_sum += wsse;
            current_rd += c::ref_rdcost(rdmult, wrate, wdist);
            if current_rd > ref_best_rd {
                invalid = true;
            }
        }
    }
    if invalid {
        return (i64::MAX, None);
    }
    let rate_total = rate_sum.min(i64::from(i32::MAX)) as i32;
    let rd = c::ref_rdcost(rdmult, rate_total + no_skip_rate + tx_size_rate, dist_sum);
    (rd, Some((rate_total + tx_size_rate, dist_sum, sse_sum, winners)))
}
/// `max_txsize_lookup[BLOCK_SIZES_ALL]` (common_data.h).
pub const MAX_TXSIZE_LOOKUP: [usize; 22] = [
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0, 1, 1, 2, 2,
];
/// C-side `intra_model_rd` (luma, use_hadamard=1) over REAL reference pieces.
#[allow(clippy::too_many_arguments)]
pub fn c_intra_model_rd(
    bsize: usize,
    tx_size: usize,
    recon_c: &mut [u16],
    src: &[u16],
    geometry: (i32, i32, usize, usize, usize), // mi_row, mi_col, ref_off, src_off, stride
    mode: usize,
    angle_delta: i32,
    use_fi: bool,
    fi_mode: usize,
    bd: u8,
) -> i64 {
    let (mi_row, mi_col, ref_off, src_off, stride) = geometry;
    let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let (txwu, txhu) = (txw >> 2, txh >> 2);
    let n = txw; // square
    let mut satd_cost: i64 = 0;
    for blk_row in (0..bh >> 2).step_by(txhu) {
        for blk_col in (0..bw >> 2).step_by(txwu) {
            let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                12,
                bsize,
                mi_row,
                mi_col,
                true,
                true,
                1 << 16,
                1 << 16,
                0,
                tx_size,
                0,
                0,
                blk_row as i32,
                blk_col as i32,
                bw as i32,
                bh as i32,
                512,
                512,
                mode,
                angle_delta * 3,
                use_fi,
            );
            let txb_off = ref_off + (blk_row * stride + blk_col) * 4;
            let pred = c::ref_hbd_predict_intra(
                recon_c,
                txb_off,
                stride,
                mode,
                angle_delta * 3,
                use_fi,
                fi_mode,
                false,
                0,
                tx_size,
                txw,
                txh,
                n_top,
                n_tr,
                n_left,
                n_bl,
                bd as i32,
            );
            for r in 0..txh {
                recon_c[txb_off + r * stride..txb_off + r * stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }
            let src_txb_off = src_off + (blk_row * stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            c::ref_highbd_subtract_block(
                txh,
                txw,
                &mut residual,
                txw,
                &src[src_txb_off..],
                stride,
                &pred,
                txw,
            );
            // av1_quick_txfm use_hadamard=1: wht_fwd_txfm (8-bit buffers) /
            // highbd_wht_fwd_txfm (bd>8: lowbd 4x4, highbd above).
            let coeff = if bd > 8 && n > 4 {
                c::ref_highbd_hadamard(n, &residual, txw)
            } else {
                c::ref_hadamard(n, &residual, txw)
            };
            satd_cost += i64::from(c::ref_satd(&coeff));
        }
    }
    satd_cost
}

/// C-side `av1_pick_uniform_tx_size_type_yrd` (luma intra): the lossless
/// TX_4X4 arm or the `choose_tx_size_type_from_rd` depth sweep (transcribed;
/// speed-0 init depth, low-contrast regression prune) over
/// [`c_uniform_txfm_yrd`]. Returns the winner
/// `(tx_size, rd, rate, dist, sse, winners)` or `None` (rate INT_MAX).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_pick_uniform_tx_size_type_yrd(
    bsize: usize,
    geometry: (i32, i32, usize, usize, usize),
    recon_c: &mut [u16],
    src: &[u16],
    mode: usize,
    angle_delta: i32,
    use_fi: bool,
    fi_mode: usize,
    lossless: bool,
    reduced: bool,
    bd: u8,
    plane_rows_c: &[i16],
    dequant: [i16; 2],
    above_ctx: &[i8],
    left_ctx: &[i8],
    rdmult: i32,
    ref_best_rd: i64,
    coeff_tbls: (&[i32], &[i32], &[i32], &[i32], &[i32], &[i32], &[i32]),
    ttc_tables: (&[i32], &[i32]),
    skip_costs: &[[i32; 2]; 3],
    skip_ctx: usize,
    ts_flat: &[i32],
    tx_size_ctx: usize,
    source_variance: u32,
) -> Option<(usize, i64, i32, i64, i64, Vec<(usize, u16, u8)>)> {
    const MI_W: [usize; 22] =
        [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
    const MI_H: [usize; 22] =
        [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];
    const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] =
        [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];
    const SUB_TX_SIZE_MAP: [usize; 19] =
        [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];

    if lossless {
        let (rd, res) = c_uniform_txfm_yrd(
            bsize, 0, geometry, recon_c, src, mode, angle_delta, use_fi, fi_mode, lossless,
            reduced, bd, plane_rows_c, dequant, above_ctx, left_ctx, rdmult, ref_best_rd,
            coeff_tbls, ttc_tables, skip_costs, skip_ctx, ts_flat, tx_size_ctx,
        );
        return res.map(|(rate, dist, sse, w)| (0, rd, rate, dist, sse, w));
    }
    // get_search_init_depth (intra, speed-0 allintra): sqr = 1, rect = 0.
    let init_depth = if MI_H[bsize] != MI_W[bsize] { 0 } else { 1 };
    let start_tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
    let mut best: Option<(usize, i64, i32, i64, i64, Vec<(usize, u16, u8)>)> = None;
    let mut rd_arr = [i64::MAX; 3];
    let mut best_rd_c = i64::MAX;
    let mut tx = start_tx;
    let mut depth = init_depth;
    while depth <= 2 {
        let (rd, res) = c_uniform_txfm_yrd(
            bsize, tx, geometry, recon_c, src, mode, angle_delta, use_fi, fi_mode, false,
            reduced, bd, plane_rows_c, dequant, above_ctx, left_ctx, rdmult, ref_best_rd,
            coeff_tbls, ttc_tables, skip_costs, skip_ctx, ts_flat, tx_size_ctx,
        );
        rd_arr[depth as usize] = rd;
        if rd < best_rd_c {
            best_rd_c = rd;
            if let Some((rate, dist, sse, w)) = res {
                best = Some((tx, rd, rate, dist, sse, w));
            }
        }
        if tx == 0 {
            break;
        }
        if depth > init_depth && depth != 2 && source_variance < 256 {
            let prev = rd_arr[depth as usize - 1];
            if prev != i64::MAX && rd_arr[depth as usize] > prev {
                break;
            }
        }
        depth += 1;
        tx = SUB_TX_SIZE_MAP[tx];
    }
    best
}

// ---------------------------------------------------------------------------
// Chroma (UV) intra RD C-side chain: plane-aware search_tx_type +
// av1_txfm_rd_in_plane (UV walk, incl. the CfL DC+AC prediction with the
// encoder dc-pred cache) + av1_txfm_uvrd — transcribed control flow over
// REAL reference pieces.
// ---------------------------------------------------------------------------

use aom_encode::tx_search::trellis_rdmult_intra;

pub const MI_W: [usize; 22] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
pub const MI_H: [usize; 22] =
    [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];

/// The C-side encoder CfL DC-prediction cache (cfl_store_dc_pred /
/// cfl_load_dc_pred transcription: first row stored, row-replicated on load).
pub struct CDcCache {
    pub use_cache: bool,
    pub cached: [bool; 2],
    pub row: [[u16; 32]; 2],
}

impl CDcCache {
    pub fn cleared() -> Self {
        CDcCache { use_cache: false, cached: [false; 2], row: [[0; 32]; 2] }
    }
}

/// C-side search_tx_type for one txb of ANY plane (the chain of REAL pieces;
/// loop control transcribed from tx_search.c 2199-2363). `plane_bsize` is the
/// plane's (subsampled) block size; `uv_mode` selects the pinned chroma tx
/// type when `plane > 0`. Returns the winner
/// (tx_type, eob, rate, dist, sse, entropy_ctx, dqcoeff, rd).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_search_tx_type_p(
    plane: usize,
    uv_mode: usize,
    residual: &[i16],
    pred: &[u16],
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    tx_size: usize,
    mode: usize,
    use_fi: bool,
    fi_mode: usize,
    lossless: bool,
    reduced: bool,
    bd: u8,
    plane_rows_c: &[i16],
    dequant: [i16; 2],
    t_above: &[i8],
    t_left: &[i8],
    plane_bsize: usize,
    rdmult: i32,
    ref_best_rd: i64,
    coeff_tbls: (&[i32], &[i32], &[i32], &[i32], &[i32], &[i32], &[i32]),
    ttc_tables: (&[i32], &[i32]),
) -> (usize, u16, i32, i64, i64, u8, Vec<i32>, i64) {
    let (w, _h) = (TX_W[tx_size], TX_H[tx_size]);
    let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
    let (txb_skip, base_eob, base, eob_extra, dc_sign, lps, eob_tbl) = coeff_tbls;
    let (mask_c, _txk) = if plane == 0 {
        c::ref_get_tx_mask_intra(
            tx_size as i32,
            mode as i32,
            use_fi,
            fi_mode as i32,
            lossless,
            reduced,
            1,
            false,
            true,
            false,
        )
    } else {
        let (m, t) = c::ref_get_tx_mask_uv_intra(
            tx_size, uv_mode, mode, use_fi, fi_mode, lossless, reduced, 1, true, false,
        );
        (m, t as i32)
    };
    let tx_bsize_twin = tx_to_bsize(tx_size);
    let (bsse_raw, mut mse_c) = c::ref_pixel_diff_dist(
        residual, tx_bsize_twin as i32, tx_bsize_twin as i32, 0, 0, 0, 0, 0, 0,
    );
    let mut bsse_c = bsse_raw;
    if bd > 8 {
        let s = 2 * (bd as i32 - 8);
        bsse_c = (bsse_c + ((1i64 << s) >> 1)) >> s;
        mse_c = (((mse_c as u64) + ((1u64 << s) >> 1)) >> s) as u32;
    }
    bsse_c *= 16;
    let dequant_shift = if bd > 8 { bd as i32 - 5 } else { 3 };
    let qstep_c = (i32::from(dequant[1]) >> dequant_shift) as u64;
    let skip_trellis_c = !((mse_c as u64) <= 3200u64 * qstep_c * qstep_c);
    let kind_c = if skip_trellis_c { 1 } else { 0 };
    let trellis_rdmult = trellis_rdmult_intra(rdmult, 0, bd, plane);
    let (txb_skip_ctx_c, dc_sign_ctx_c) =
        c::ref_get_txb_ctx(plane_bsize, tx_size, plane, t_above, t_left);

    let mut best_rd_c = i64::MAX;
    let mut best: Option<(usize, u16, i32, i64, i64, u8, Vec<i32>)> = None;
    for tx_type in 0..16usize {
        if mask_c & (1 << tx_type) == 0 {
            continue;
        }
        let coeff = c::ref_fwd_txfm2d(tx_size, residual, w, tx_type);
        let tcoeff = coeff[..n_coeffs].to_vec();
        let mut qc = vec![0i32; n_coeffs];
        let mut dqc = vec![0i32; n_coeffs];
        let eob = c::ref_quant_plane_rows(
            kind_c,
            bd > 8,
            &tcoeff,
            plane_rows_c,
            scan(tx_size, tx_type),
            aom_txb::iscan(tx_size, tx_type),
            aom_encode::tx_scale(tx_size),
            &mut qc,
            &mut dqc,
        ) as usize;
        let ttc = |eob: usize| -> i32 {
            if eob > 0 {
                c::ref_get_tx_type_cost(
                    ttc_tables.0,
                    ttc_tables.1,
                    plane as i32,
                    tx_size as i32,
                    tx_type as i32,
                    false,
                    reduced,
                    lossless,
                    use_fi,
                    fi_mode as i32,
                    mode as i32,
                )
            } else {
                0
            }
        };
        let (eob, rate_c, ctx_c) = if !skip_trellis_c {
            if eob == 0 {
                (0usize, txb_skip[txb_skip_ctx_c as usize * 2 + 1], 0u8)
            } else {
                let (ne, r) = c::ref_optimize_txb(
                    tx_size,
                    tx_type,
                    &mut qc,
                    &mut dqc,
                    &tcoeff,
                    eob,
                    &dequant,
                    trellis_rdmult,
                    dc_sign_ctx_c as usize,
                    txb_skip_ctx_c as usize,
                    0,
                    scan(tx_size, tx_type),
                    txb_skip,
                    base_eob,
                    base,
                    eob_extra,
                    dc_sign,
                    lps,
                    eob_tbl,
                );
                let ctx = c::ref_txb_entropy_context(&qc, tx_size, tx_type, ne);
                (ne, r + ttc(ne), ctx)
            }
        } else {
            let r = c::ref_cost_coeffs_txb(
                &qc,
                eob,
                tx_size,
                tx_type,
                txb_skip_ctx_c as usize,
                dc_sign_ctx_c as usize,
                txb_skip,
                base_eob,
                base,
                eob_extra,
                dc_sign,
                lps,
                eob_tbl,
            ) + ttc(eob);
            let ctx = c::ref_txb_entropy_context(&qc, tx_size, tx_type, eob);
            (eob, r, ctx)
        };
        if c::ref_rdcost(rdmult, rate_c, 0) > best_rd_c {
            continue;
        }
        let (dist_c, sse_c) = if eob == 0 {
            (bsse_c, bsse_c)
        } else {
            let high_energy = bsse_c >= 128 * 128 * TX_SIZE_2D_TBL[tx_size];
            let is_tx64 = tx_size == 4;
            let mut d = i64::MAX;
            let mut s_tx = i64::MAX;
            let mut sse_diff = i64::MAX;
            if is_tx64 || high_energy {
                let (dt, st) = c::ref_dist_block_tx_domain(&tcoeff, &dqc, tx_size, bd);
                d = dt;
                s_tx = st;
                sse_diff = bsse_c - st;
            }
            if !is_tx64 || !high_energy || sse_diff * 2 < s_tx {
                let tx_dom = d;
                let mut recon = pred.to_vec();
                c::ref_inv_txfm2d_add(tx_size, &dqc, &mut recon, w, tx_type, bd as i32);
                let (_v, vf_sse) = c::ref_hbd_variance(
                    VAR_IDX[tx_size],
                    bd,
                    &src[src_off..],
                    src_stride,
                    &recon,
                    w,
                );
                d = 16 * i64::from(vf_sse);
                if high_energy && d < tx_dom {
                    d = tx_dom;
                }
            } else {
                d += sse_diff;
            }
            (d, bsse_c)
        };
        let rd = c::ref_rdcost(rdmult, rate_c, dist_c);
        if rd < best_rd_c {
            best_rd_c = rd;
            best = Some((tx_type, eob as u16, rate_c, dist_c, sse_c, ctx_c, dqc.clone()));
        }
        if (best_rd_c - (best_rd_c >> 1)) > ref_best_rd {
            break;
        }
    }
    let b = best.expect("C search always yields a winner");
    (b.0, b.1, b.2, b.3, b.4, b.5, b.6, best_rd_c)
}

/// The geometry + candidate arguments of the C-side UV walk (shared by both
/// planes; per-plane offsets indexed `[plane - 1]`).
#[allow(clippy::type_complexity)]
pub struct CUvEnv<'a> {
    pub bsize: usize,
    pub mi_row: i32,
    pub mi_col: i32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub ref_off: [usize; 2],
    pub src_off: [usize; 2],
    pub stride: usize,
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    pub luma_mode: usize,
    pub luma_use_fi: bool,
    pub luma_fi_mode: usize,
    pub lossless: bool,
    pub reduced: bool,
    pub bd: u8,
    pub rows_u_c: &'a [i16],
    pub rows_v_c: &'a [i16],
    pub dequant_u: [i16; 2],
    pub dequant_v: [i16; 2],
    pub above_ctx: [&'a [i8]; 2],
    pub left_ctx: [&'a [i8]; 2],
    pub rdmult: i32,
    pub coeff_tbls: (&'a [i32], &'a [i32], &'a [i32], &'a [i32], &'a [i32], &'a [i32], &'a [i32]),
    pub ttc_tables: (&'a [i32], &'a [i32]),
}

/// C-side `av1_txfm_rd_in_plane` for one CHROMA plane (intra): the walk over
/// REAL pieces, incl. the CfL arm (dc pred via ref_hbd_predict_intra — cached
/// per the C dc-pred cache — + the REAL av1_cfl_predict_block).
/// Returns `(rate, dist, sse, winners)` or `None` (exit_early).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_txfm_rd_in_plane_uv(
    env: &CUvEnv,
    recon: &mut [u16],
    plane: usize,
    uv_mode: usize,
    angle_delta_uv: i32,
    cfl: Option<(&mut c::RefCflState, &mut CDcCache, i32, i32)>,
    tx_size: usize,
    ref_best_rd: i64,
    current_rd_in: i64,
) -> Option<(i32, i64, i64, Vec<(usize, u16, u8)>)> {
    if current_rd_in > ref_best_rd {
        return None;
    }
    let plane_bsize = aom_entropy::partition::get_plane_block_size(env.bsize, env.ss_x, env.ss_y);
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let (txwu, txhu) = (txw >> 2, txh >> 2);
    let max_w = MI_W[plane_bsize];
    let max_h = MI_H[plane_bsize];
    let pi = plane - 1;
    let mode = aom_entropy::partition::get_uv_mode(uv_mode) as usize;
    let wpx = ((MI_W[env.bsize] * 4) >> env.ss_x).max(4) as i32;
    let hpx = ((MI_H[env.bsize] * 4) >> env.ss_y).max(4) as i32;
    let src: &[u16] = if plane == 1 { env.src_u } else { env.src_v };
    let (rows_c, dequant) =
        if plane == 1 { (env.rows_u_c, env.dequant_u) } else { (env.rows_v_c, env.dequant_v) };

    let mut t_above = env.above_ctx[pi][..max_w].to_vec();
    let mut t_left = env.left_ctx[pi][..max_h].to_vec();
    let mut rate_sum: i64 = 0;
    let mut dist_sum: i64 = 0;
    let mut sse_sum: i64 = 0;
    let mut winners: Vec<(usize, u16, u8)> = Vec::new();
    let mut current_rd = current_rd_in;
    let mut cfl = cfl;

    let mut blk_row = 0usize;
    while blk_row < max_h {
        let mut blk_col = 0usize;
        while blk_col < max_w {
            let txb_off = env.ref_off[pi] + (blk_row * env.stride + blk_col) * 4;
            if let Some((st, cache, alpha_idx, joint_sign)) = cfl.as_mut() {
                assert_eq!((blk_row, blk_col), (0, 0));
                let pred_plane = plane - 1;
                if !(cache.use_cache && cache.cached[pred_plane]) {
                    let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                        12, env.bsize, env.mi_row, env.mi_col, true, true, 1 << 16, 1 << 16,
                        0, tx_size, env.ss_x as i32, env.ss_y as i32, blk_row as i32,
                        blk_col as i32, wpx, hpx, 512, 512, mode, 0, false,
                    );
                    let pred = c::ref_hbd_predict_intra(
                        recon, txb_off, env.stride, mode, 0, false, 0, false, 0, tx_size,
                        txw, txh, n_top, n_tr, n_left, n_bl, env.bd as i32,
                    );
                    for r in 0..txh {
                        recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                            .copy_from_slice(&pred[r * txw..r * txw + txw]);
                    }
                    if cache.use_cache {
                        cache.row[pred_plane][..txw]
                            .copy_from_slice(&recon[txb_off..txb_off + txw]);
                        cache.cached[pred_plane] = true;
                    }
                } else {
                    for r in 0..txh {
                        recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                            .copy_from_slice(&cache.row[pred_plane][..txw]);
                    }
                }
                c::ref_cfl_predict_block(
                    st, recon, txb_off, env.stride, tx_size, plane, *alpha_idx, *joint_sign,
                    env.bsize, env.lossless, env.ss_x as i32, env.ss_y as i32, env.bd,
                );
            } else {
                let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                    12, env.bsize, env.mi_row, env.mi_col, true, true, 1 << 16, 1 << 16, 0,
                    tx_size, env.ss_x as i32, env.ss_y as i32, blk_row as i32, blk_col as i32,
                    wpx, hpx, 512, 512, mode, angle_delta_uv * 3, false,
                );
                let pred = c::ref_hbd_predict_intra(
                    recon, txb_off, env.stride, mode, angle_delta_uv * 3, false, 0, false, 0,
                    tx_size, txw, txh, n_top, n_tr, n_left, n_bl, env.bd as i32,
                );
                for r in 0..txh {
                    recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                        .copy_from_slice(&pred[r * txw..r * txw + txw]);
                }
            }
            // Snapshot the prediction for subtract + winner recon.
            let mut pred = vec![0u16; txw * txh];
            for r in 0..txh {
                pred[r * txw..r * txw + txw].copy_from_slice(
                    &recon[txb_off + r * env.stride..txb_off + r * env.stride + txw],
                );
            }
            let src_txb_off = env.src_off[pi] + (blk_row * env.stride + blk_col) * 4;
            let mut residual = vec![0i16; txw * txh];
            c::ref_highbd_subtract_block(
                txh, txw, &mut residual, txw, &src[src_txb_off..], env.stride, &pred, txw,
            );
            let (wtype, weob, wrate, wdist, wsse, wctx, wdqc, _wrd) = c_search_tx_type_p(
                plane, uv_mode, &residual, &pred, src, src_txb_off, env.stride, tx_size,
                env.luma_mode, env.luma_use_fi, env.luma_fi_mode, env.lossless, env.reduced,
                env.bd, rows_c, dequant, &t_above[blk_col..], &t_left[blk_row..], plane_bsize,
                env.rdmult, ref_best_rd - current_rd, env.coeff_tbls, env.ttc_tables,
            );
            if weob > 0 {
                let mut tight = pred.clone();
                c::ref_inv_txfm2d_add(tx_size, &wdqc, &mut tight, txw, wtype, env.bd as i32);
                for r in 0..txh {
                    recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }
            for a in t_above[blk_col..blk_col + txwu].iter_mut() {
                *a = wctx as i8;
            }
            for l in t_left[blk_row..blk_row + txhu].iter_mut() {
                *l = wctx as i8;
            }
            winners.push((wtype, weob, wctx));
            rate_sum += i64::from(wrate);
            dist_sum += wdist;
            sse_sum += wsse;
            current_rd += c::ref_rdcost(env.rdmult, wrate, wdist);
            if current_rd > ref_best_rd {
                // exit_early: for intra ANY early exit invalidates — but only
                // if a later txb would run; the last txb setting it still
                // invalidates (tx_search.c:3786 exit_early arm).
                return None;
            }
            blk_col += txwu;
        }
        blk_row += txhu;
    }
    let rate_total = rate_sum.min(i64::from(i32::MAX)) as i32;
    Some((rate_total, dist_sum, sse_sum, winners))
}

/// C-side `av1_txfm_uvrd` (intra arm): both chroma planes at the uniform UV
/// tx size with the merged-min gate. Returns
/// `(rate, dist, sse, winners_u, winners_v)` or `None` (invalid).
#[allow(clippy::type_complexity)]
pub fn c_txfm_uvrd(
    env: &CUvEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    uv_mode: usize,
    angle_delta_uv: i32,
    ref_best_rd: i64,
) -> Option<(i32, i64, i64, Vec<(usize, u16, u8)>, Vec<(usize, u16, u8)>)> {
    if ref_best_rd < 0 {
        return None;
    }
    let uv_tx_size = aom_encode::intra_uv_rd::av1_get_tx_size_uv(
        env.bsize, env.lossless, env.ss_x, env.ss_y,
    );
    let mut rate: i64 = 0;
    let mut dist: i64 = 0;
    let mut sse: i64 = 0;
    let mut winners_u = Vec::new();
    let mut winners_v = Vec::new();
    for plane in 1..=2usize {
        let recon: &mut [u16] = if plane == 1 { recon_u } else { recon_v };
        let r = c_txfm_rd_in_plane_uv(
            env, recon, plane, uv_mode, angle_delta_uv, None, uv_tx_size, ref_best_rd, 0,
        )?;
        let (prate, pdist, psse, winners) = r;
        if prate == i32::MAX {
            return None;
        }
        rate = (rate + i64::from(prate)).min(i64::from(i32::MAX));
        dist += pdist;
        sse += psse;
        if plane == 1 {
            winners_u = winners;
        } else {
            winners_v = winners;
        }
        let this_rd = c::ref_rdcost(env.rdmult, rate as i32, dist);
        let skip_rd = c::ref_rdcost(env.rdmult, 0, sse);
        if this_rd.min(skip_rd) > ref_best_rd {
            return None;
        }
    }
    Some((rate as i32, dist, sse, winners_u, winners_v))
}
