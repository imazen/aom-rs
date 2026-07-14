//! Shared C-side chain helpers for the aom-encode differential harnesses —
//! the transcribed loop skeletons driving REAL reference pieces
//! (`c_search_tx_type`, `c_uniform_txfm_yrd`, `c_pick_uniform_tx_size_type_yrd`,
//! `c_intra_model_rd`) plus the common Rng / cost-table / CDF generators.
//! Moved verbatim out of uniform_txfm_yrd_diff.rs / intra_model_rd_diff.rs
//! (each test binary uses a subset).
#![allow(dead_code)]

use aom_encode::tx_search::TX_SIZE_2D_TBL;
use aom_sys_ref as c;
use aom_txb::{CoeffCostSet, LvMapCoeffCost, scan, txb_high, txb_wide};

pub const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
pub const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
pub const BLK_W: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
pub const BLK_H: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
pub const VAR_IDX: [usize; 19] = [
    0, 4, 9, 14, 18, 1, 3, 5, 8, 10, 13, 15, 17, 2, 7, 6, 12, 11, 16,
];
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

/// Build a [`CoeffCostSet`] by replicating ONE synthetic-but-valid
/// `LV_MAP_COEFF_COST`/`LV_MAP_EOB_COST` table (the 7 flat arrays these
/// harnesses already generate via [`tbl`]) across every `txs_ctx` (0..5) and
/// `eob_multi_size` (0..7) slot. Every C-side oracle in this test suite
/// reads these SAME 7 arrays directly regardless of the tx_size under test
/// (they are not themselves per-txs_ctx-indexed on the C shim side — see
/// e.g. `c_uniform_txfm_yrd`'s `coeff_tbls` param), so
/// `CoeffCostSet::tables(tx_size)` returning these exact values at ANY
/// `tx_size` reproduces the pre-`CoeffCostSet` single-table behaviour
/// byte-for-byte — this is glue for the `TxfmYrdEnv`/`SbEncodeEnv` struct
/// field type, not a change to what values the search/pack sees.
pub fn coeff_cost_set_from_tables(
    txb_skip: &[i32],
    base_eob: &[i32],
    base: &[i32],
    eob_extra: &[i32],
    dc_sign: &[i32],
    lps: &[i32],
    eob: &[i32],
) -> CoeffCostSet {
    let single = LvMapCoeffCost {
        txb_skip: txb_skip.to_vec(),
        base_eob: base_eob.to_vec(),
        base: base.to_vec(),
        eob_extra: eob_extra.to_vec(),
        dc_sign: dc_sign.to_vec(),
        lps: lps.to_vec(),
    };
    let mut eob_row = [0i32; 22];
    eob_row[..eob.len()].copy_from_slice(eob);
    CoeffCostSet {
        by_txs_ctx: core::array::from_fn(|_| single.clone()),
        eob_by_multi_size: [eob_row; 7],
    }
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
    // plane 0: the chroma trellis-table select is irrelevant (luma is 17 in
    // both tables).
    c_search_tx_type_p(
        0,
        0,
        residual,
        pred,
        src,
        src_off,
        src_stride,
        tx_size,
        mode,
        use_fi,
        fi_mode,
        lossless,
        reduced,
        bd,
        plane_rows_c,
        dequant,
        t_above,
        t_left,
        bsize,
        rdmult,
        ref_best_rd,
        coeff_tbls,
        ttc_tables,
        true,
    )
}

/// BLOCK_SIZE with the same dims as a TX_SIZE.
pub fn tx_to_bsize(tx_size: usize) -> usize {
    const T: [usize; 19] = [
        0, 3, 6, 9, 12, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21,
    ];
    T[tx_size]
}
/// C-side `uniform_txfm_yrd` for one size: the full walk + intra assembly.
/// Returns `(rd, Some((rate, dist, sse, winners)))` or `(MAX, None)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_uniform_txfm_yrd(
    bsize: usize,
    tx_size: usize,
    geometry: (i32, i32, usize, usize, usize), // mi_row, mi_col, ref_off, src_off, stride
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
    let tx_size_rate = c::ref_tx_size_cost(
        ts_flat,
        !lossless,
        bsize as i32,
        tx_size as i32,
        tx_size_ctx as i32,
    );
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
            let (wtype, weob, wrate, wdist, wsse, wctx, wdqc, _wrd) = c_search_tx_type(
                &residual,
                &pred,
                src,
                src_txb_off,
                stride,
                tx_size,
                mode,
                use_fi,
                fi_mode,
                lossless,
                reduced,
                bd,
                plane_rows_c,
                dequant,
                &t_above[blk_col..],
                &t_left[blk_row..],
                bsize,
                rdmult,
                ref_best_rd - current_rd,
                coeff_tbls,
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
    (
        rd,
        Some((rate_total + tx_size_rate, dist_sum, sse_sum, winners)),
    )
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
    const MI_W: [usize; 22] = [
        1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
    ];
    const MI_H: [usize; 22] = [
        1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
    ];
    const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] = [
        0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18,
    ];
    const SUB_TX_SIZE_MAP: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];

    if lossless {
        let (rd, res) = c_uniform_txfm_yrd(
            bsize,
            0,
            geometry,
            recon_c,
            src,
            mode,
            angle_delta,
            use_fi,
            fi_mode,
            lossless,
            reduced,
            bd,
            plane_rows_c,
            dequant,
            above_ctx,
            left_ctx,
            rdmult,
            ref_best_rd,
            coeff_tbls,
            ttc_tables,
            skip_costs,
            skip_ctx,
            ts_flat,
            tx_size_ctx,
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
            bsize,
            tx,
            geometry,
            recon_c,
            src,
            mode,
            angle_delta,
            use_fi,
            fi_mode,
            false,
            reduced,
            bd,
            plane_rows_c,
            dequant,
            above_ctx,
            left_ctx,
            rdmult,
            ref_best_rd,
            coeff_tbls,
            ttc_tables,
            skip_costs,
            skip_ctx,
            ts_flat,
            tx_size_ctx,
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

pub const MI_W: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
pub const MI_H: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// The C-side encoder CfL DC-prediction cache (cfl_store_dc_pred /
/// cfl_load_dc_pred transcription: first row stored, row-replicated on load).
pub struct CDcCache {
    pub use_cache: bool,
    pub cached: [bool; 2],
    pub row: [[u16; 32]; 2],
}

impl CDcCache {
    pub fn cleared() -> Self {
        CDcCache {
            use_cache: false,
            cached: [false; 2],
            row: [[0; 32]; 2],
        }
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
    use_chroma_trellis_rd_mult: bool,
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
        residual,
        tx_bsize_twin as i32,
        tx_bsize_twin as i32,
        0,
        0,
        0,
        0,
        0,
        0,
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
    let trellis_rdmult = trellis_rdmult_intra(rdmult, 0, bd, plane, use_chroma_trellis_rd_mult);
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
            best = Some((
                tx_type,
                eob as u16,
                rate_c,
                dist_c,
                sse_c,
                ctx_c,
                dqc.clone(),
            ));
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
    pub coeff_tbls: (
        &'a [i32],
        &'a [i32],
        &'a [i32],
        &'a [i32],
        &'a [i32],
        &'a [i32],
        &'a [i32],
    ),
    pub ttc_tables: (&'a [i32], &'a [i32]),
    /// sf `tx_sf.use_chroma_trellis_rd_mult` — ALLINTRA/RT 1 (chroma trellis
    /// mult 13), usage GOOD 0 (mult 20). Both arms swept by the diffs.
    pub use_chroma_trellis_rd_mult: bool,
    /// `mbmi->partition` (has_top_right/has_bottom_left input; 0 NONE /
    /// 1 HORZ / 2 VERT for the ported tree shapes).
    pub partition: usize,
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
    let (rows_c, dequant) = if plane == 1 {
        (env.rows_u_c, env.dequant_u)
    } else {
        (env.rows_v_c, env.dequant_v)
    };

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
                        12,
                        env.bsize,
                        env.mi_row,
                        env.mi_col,
                        true,
                        true,
                        1 << 16,
                        1 << 16,
                        0,
                        tx_size,
                        env.ss_x as i32,
                        env.ss_y as i32,
                        blk_row as i32,
                        blk_col as i32,
                        wpx,
                        hpx,
                        512,
                        512,
                        mode,
                        0,
                        false,
                    );
                    let pred = c::ref_hbd_predict_intra(
                        recon,
                        txb_off,
                        env.stride,
                        mode,
                        0,
                        false,
                        0,
                        false,
                        0,
                        tx_size,
                        txw,
                        txh,
                        n_top,
                        n_tr,
                        n_left,
                        n_bl,
                        env.bd as i32,
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
                    st,
                    recon,
                    txb_off,
                    env.stride,
                    tx_size,
                    plane,
                    *alpha_idx,
                    *joint_sign,
                    env.bsize,
                    env.lossless,
                    env.ss_x as i32,
                    env.ss_y as i32,
                    env.bd,
                );
            } else {
                let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                    12,
                    env.bsize,
                    env.mi_row,
                    env.mi_col,
                    true,
                    true,
                    1 << 16,
                    1 << 16,
                    0,
                    tx_size,
                    env.ss_x as i32,
                    env.ss_y as i32,
                    blk_row as i32,
                    blk_col as i32,
                    wpx,
                    hpx,
                    512,
                    512,
                    mode,
                    angle_delta_uv * 3,
                    false,
                );
                let pred = c::ref_hbd_predict_intra(
                    recon,
                    txb_off,
                    env.stride,
                    mode,
                    angle_delta_uv * 3,
                    false,
                    0,
                    false,
                    0,
                    tx_size,
                    txw,
                    txh,
                    n_top,
                    n_tr,
                    n_left,
                    n_bl,
                    env.bd as i32,
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
                txh,
                txw,
                &mut residual,
                txw,
                &src[src_txb_off..],
                env.stride,
                &pred,
                txw,
            );
            let (wtype, weob, wrate, wdist, wsse, wctx, wdqc, _wrd) = c_search_tx_type_p(
                plane,
                uv_mode,
                &residual,
                &pred,
                src,
                src_txb_off,
                env.stride,
                tx_size,
                env.luma_mode,
                env.luma_use_fi,
                env.luma_fi_mode,
                env.lossless,
                env.reduced,
                env.bd,
                rows_c,
                dequant,
                &t_above[blk_col..],
                &t_left[blk_row..],
                plane_bsize,
                env.rdmult,
                ref_best_rd - current_rd,
                env.coeff_tbls,
                env.ttc_tables,
                env.use_chroma_trellis_rd_mult,
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
    let uv_tx_size =
        aom_encode::intra_uv_rd::av1_get_tx_size_uv(env.bsize, env.lossless, env.ss_x, env.ss_y);
    let mut rate: i64 = 0;
    let mut dist: i64 = 0;
    let mut sse: i64 = 0;
    let mut winners_u = Vec::new();
    let mut winners_v = Vec::new();
    for plane in 1..=2usize {
        let recon: &mut [u16] = if plane == 1 { recon_u } else { recon_v };
        let r = c_txfm_rd_in_plane_uv(
            env,
            recon,
            plane,
            uv_mode,
            angle_delta_uv,
            None,
            uv_tx_size,
            ref_best_rd,
            0,
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

// ---------------------------------------------------------------------------
// CfL alpha search C-side chain (intra_mode_search.c 586-848 transcription
// over REAL pieces: ref_hbd_predict_intra dc + REAL av1_cfl_predict_block +
// the plane-aware search chain + ref_fwd_txfm2d/ref_satd fast model).
// ---------------------------------------------------------------------------

pub const CFL_MAGS_SIZE: usize = 33;
pub const CFL_INDEX_ZERO: i32 = 16;

pub fn c_cfl_idx_to_sign_and_alpha(cfl_idx: i32) -> (i32, i32) {
    let lin = cfl_idx - CFL_INDEX_ZERO;
    if lin == 0 {
        (0, 0)
    } else {
        (if lin > 0 { 2 } else { 1 }, lin.abs() - 1)
    }
}

/// C-side RD_STATS for the joint scan.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CRdStats {
    pub rate: i32,
    pub dist: i64,
    pub sse: i64,
    pub skip: bool,
    pub zero_rate: i32,
    pub rdcost: i64,
}

impl CRdStats {
    pub fn invalid() -> Self {
        CRdStats {
            rate: i32::MAX,
            dist: i64::MAX,
            sse: i64::MAX,
            skip: false,
            zero_rate: 0,
            rdcost: i64::MAX,
        }
    }
    pub fn merge(&mut self, o: &CRdStats) {
        if self.rate == i32::MAX || o.rate == i32::MAX {
            *self = CRdStats::invalid();
            return;
        }
        self.rate = (i64::from(self.rate) + i64::from(o.rate)).min(i64::from(i32::MAX)) as i32;
        if self.zero_rate == 0 {
            self.zero_rate = o.zero_rate;
        }
        self.dist += o.dist;
        if self.sse < i64::MAX && o.sse < i64::MAX {
            self.sse += o.sse;
        }
        self.skip &= o.skip;
    }
    pub fn rd_cost_update(&mut self, rdmult: i32) {
        if self.rate < i32::MAX && self.dist < i64::MAX && self.rdcost < i64::MAX {
            self.rdcost = c::ref_rdcost(rdmult, self.rate, self.dist);
        } else {
            *self = CRdStats::invalid();
        }
    }
}

/// C-side CfL prediction for one txb (facade CfL arm): dc pred (cache-aware)
/// + REAL av1_cfl_predict_block, into the recon plane.
#[allow(clippy::too_many_arguments)]
fn c_predict_cfl_txb(
    env: &CUvEnv,
    recon: &mut [u16],
    plane: usize,
    st: &mut c::RefCflState,
    cache: &mut CDcCache,
    alpha_idx: i32,
    joint_sign: i32,
    tx_size: usize,
    txb_off: usize,
) {
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let wpx = ((MI_W[env.bsize] * 4) >> env.ss_x).max(4) as i32;
    let hpx = ((MI_H[env.bsize] * 4) >> env.ss_y).max(4) as i32;
    let pred_plane = plane - 1;
    if !(cache.use_cache && cache.cached[pred_plane]) {
        let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
            12,
            env.bsize,
            env.mi_row,
            env.mi_col,
            true,
            true,
            1 << 16,
            1 << 16,
            0,
            tx_size,
            env.ss_x as i32,
            env.ss_y as i32,
            0,
            0,
            wpx,
            hpx,
            512,
            512,
            0,
            0,
            false,
        );
        let pred = c::ref_hbd_predict_intra(
            recon,
            txb_off,
            env.stride,
            0,
            0,
            false,
            0,
            false,
            0,
            tx_size,
            txw,
            txh,
            n_top,
            n_tr,
            n_left,
            n_bl,
            env.bd as i32,
        );
        for r in 0..txh {
            recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                .copy_from_slice(&pred[r * txw..r * txw + txw]);
        }
        if cache.use_cache {
            cache.row[pred_plane][..txw].copy_from_slice(&recon[txb_off..txb_off + txw]);
            cache.cached[pred_plane] = true;
        }
    } else {
        for r in 0..txh {
            recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                .copy_from_slice(&cache.row[pred_plane][..txw]);
        }
    }
    c::ref_cfl_predict_block(
        st,
        recon,
        txb_off,
        env.stride,
        tx_size,
        plane,
        alpha_idx,
        joint_sign,
        env.bsize,
        env.lossless,
        env.ss_x as i32,
        env.ss_y as i32,
        env.bd,
    );
}

/// C-side `intra_model_rd` (chroma, use_hadamard=0): per model txb CfL
/// predict -> subtract -> real DCT_DCT forward -> ref_satd.
#[allow(clippy::too_many_arguments)]
pub fn c_intra_model_rd_uv(
    env: &CUvEnv,
    recon: &mut [u16],
    plane: usize,
    st: &mut c::RefCflState,
    cache: &mut CDcCache,
    alpha_idx: i32,
    joint_sign: i32,
    tx_size: usize,
) -> i64 {
    let plane_bsize = aom_entropy::partition::get_plane_block_size(env.bsize, env.ss_x, env.ss_y);
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let (txwu, txhu) = (txw >> 2, txh >> 2);
    let n = txw * txh;
    let pi = plane - 1;
    let src: &[u16] = if plane == 1 { env.src_u } else { env.src_v };
    let mut satd_cost: i64 = 0;
    for blk_row in (0..MI_H[plane_bsize]).step_by(txhu) {
        for blk_col in (0..MI_W[plane_bsize]).step_by(txwu) {
            let txb_off = env.ref_off[pi] + (blk_row * env.stride + blk_col) * 4;
            c_predict_cfl_txb(
                env, recon, plane, st, cache, alpha_idx, joint_sign, tx_size, txb_off,
            );
            let mut pred = vec![0u16; n];
            for r in 0..txh {
                pred[r * txw..r * txw + txw].copy_from_slice(
                    &recon[txb_off + r * env.stride..txb_off + r * env.stride + txw],
                );
            }
            let src_txb_off = env.src_off[pi] + (blk_row * env.stride + blk_col) * 4;
            let mut residual = vec![0i16; n];
            c::ref_highbd_subtract_block(
                txh,
                txw,
                &mut residual,
                txw,
                &src[src_txb_off..],
                env.stride,
                &pred,
                txw,
            );
            let coeff = c::ref_fwd_txfm2d(tx_size, &residual, txw, 0);
            satd_cost += i64::from(c::ref_satd(&coeff[..n]));
        }
    }
    satd_cost
}

/// C-side `cfl_compute_rd`: fast = the SATD model; full = the CfL UV walk
/// (budget-free) + av1_rd_cost_update.
#[allow(clippy::too_many_arguments)]
pub fn c_cfl_compute_rd(
    env: &CUvEnv,
    recon: &mut [u16],
    plane: usize,
    st: &mut c::RefCflState,
    cache: &mut CDcCache,
    tx_size: usize,
    cfl_idx: i32,
    fast_mode: bool,
) -> (i64, Option<CRdStats>) {
    let pred_plane = plane - 1;
    let (cfl_sign, cfl_alpha) = c_cfl_idx_to_sign_and_alpha(cfl_idx);
    let dummy_sign = 1; // CFL_SIGN_NEG
    let joint_sign = if pred_plane == 0 {
        cfl_sign * 3 + dummy_sign - 1
    } else {
        dummy_sign * 3 + cfl_sign - 1
    };
    let alpha_idx = (cfl_alpha << 4) + cfl_alpha;
    if fast_mode {
        let cost =
            c_intra_model_rd_uv(env, recon, plane, st, cache, alpha_idx, joint_sign, tx_size);
        (cost, None)
    } else {
        let (rate, dist, sse, _w) = c_txfm_rd_in_plane_uv(
            env,
            recon,
            plane,
            13, // UV_CFL_PRED
            0,
            Some((st, cache, alpha_idx, joint_sign)),
            tx_size,
            i64::MAX,
            0,
        )
        .expect("budget-free C walk is valid");
        let mut s = CRdStats {
            rate,
            dist,
            sse,
            skip: false,
            zero_rate: 0,
            rdcost: 0,
        };
        // Walk merges intra txbs as non-skip; init skip=1 AND per-txb 0 -> 0.
        s.rd_cost_update(env.rdmult);
        (s.rdcost, Some(s))
    }
}

/// C-side `cfl_pick_plane_parameter` + `cfl_pick_plane_rd` + the joint scan
/// of `cfl_rd_pick_alpha`. Returns
/// `Some((alpha_idx, joint_sign, stats))` or `None`.
#[allow(clippy::too_many_arguments)]
pub fn c_cfl_rd_pick_alpha(
    env: &CUvEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    st: &mut c::RefCflState,
    tx_size: usize,
    ref_best_rd: i64,
    cfl_search_range: usize,
    cfl_costs: &[i32], // [8][2][16] flat (ref_fill_cfl_costs)
    uv_mode_cost: i32,
) -> Option<(u8, i8, CRdStats)> {
    let mut cache = CDcCache::cleared();
    cache.use_cache = true;

    let pick_parameter = |env: &CUvEnv,
                          recon: &mut [u16],
                          plane: usize,
                          st: &mut c::RefCflState,
                          cache: &mut CDcCache|
     -> i32 {
        if cfl_search_range == CFL_MAGS_SIZE {
            return CFL_INDEX_ZERO;
        }
        let mut est = CFL_INDEX_ZERO;
        let (mut best_cost, _) =
            c_cfl_compute_rd(env, recon, plane, st, cache, tx_size, CFL_INDEX_ZERO, true);
        for dir in [1i32, -1] {
            for i in 1..CFL_MAGS_SIZE as i32 {
                let idx = CFL_INDEX_ZERO + dir * i;
                if !(0..CFL_MAGS_SIZE as i32).contains(&idx) {
                    break;
                }
                let (cost, _) = c_cfl_compute_rd(env, recon, plane, st, cache, tx_size, idx, true);
                if cost < best_cost {
                    best_cost = cost;
                    est = idx;
                } else {
                    break;
                }
            }
        }
        est
    };
    let est_u = pick_parameter(env, recon_u, 1, st, &mut cache);
    let est_v = pick_parameter(env, recon_v, 2, st, &mut cache);

    if cfl_search_range == 1 {
        if est_u == CFL_INDEX_ZERO && est_v == CFL_INDEX_ZERO {
            return None;
        }
        let (su, au) = c_cfl_idx_to_sign_and_alpha(est_u);
        let (sv, av) = c_cfl_idx_to_sign_and_alpha(est_v);
        let js = su * 3 + sv - 1;
        let rate_overhead = cfl_costs[(js as usize * 2) * 16 + au as usize]
            + cfl_costs[(js as usize * 2 + 1) * 16 + av as usize]
            + uv_mode_cost;
        if c::ref_rdcost(env.rdmult, rate_overhead, 0) > ref_best_rd {
            return None;
        }
    }

    let pick_rd = |env: &CUvEnv,
                   recon: &mut [u16],
                   plane: usize,
                   st: &mut c::RefCflState,
                   cache: &mut CDcCache,
                   est: i32|
     -> Vec<CRdStats> {
        let mut arr = vec![CRdStats::invalid(); CFL_MAGS_SIZE];
        let (_, s) = c_cfl_compute_rd(env, recon, plane, st, cache, tx_size, est, false);
        arr[est as usize] = s.unwrap();
        if cfl_search_range == 1 {
            return arr;
        }
        for dir in [1i32, -1] {
            for i in 1..cfl_search_range as i32 {
                let idx = est + dir * i;
                if !(0..CFL_MAGS_SIZE as i32).contains(&idx) {
                    break;
                }
                let (_, s) = c_cfl_compute_rd(env, recon, plane, st, cache, tx_size, idx, false);
                arr[idx as usize] = s.unwrap();
            }
        }
        arr
    };
    let arr_u = pick_rd(env, recon_u, 1, st, &mut cache, est_u);
    let arr_v = pick_rd(env, recon_v, 2, st, &mut cache, est_v);

    let mut best: Option<(u8, i8, CRdStats)> = None;
    let mut best_rdcost = i64::MAX;
    for (ui, ue) in arr_u.iter().enumerate() {
        if ue.rate == i32::MAX {
            continue;
        }
        let (su, au) = c_cfl_idx_to_sign_and_alpha(ui as i32);
        for (vi, ve) in arr_v.iter().enumerate() {
            if ve.rate == i32::MAX {
                continue;
            }
            let (sv, av) = c_cfl_idx_to_sign_and_alpha(vi as i32);
            if su == 0 && sv == 0 {
                continue;
            }
            let js = su * 3 + sv - 1;
            let mut rd_stats = *ue;
            rd_stats.merge(ve);
            if rd_stats.rate != i32::MAX {
                rd_stats.rate += cfl_costs[(js as usize * 2) * 16 + au as usize];
                rd_stats.rate += cfl_costs[(js as usize * 2 + 1) * 16 + av as usize];
            }
            rd_stats.rd_cost_update(env.rdmult);
            if rd_stats.rdcost < best_rdcost {
                best_rdcost = rd_stats.rdcost;
                best = Some((((au << 4) + av) as u8, js as i8, rd_stats));
            }
        }
    }
    match best {
        Some(b) if b.2.rdcost < ref_best_rd => Some(b),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The UV mode loop C-side chain (intra_mode_search.c 496-1029 transcription
// over c_txfm_uvrd / c_cfl_rd_pick_alpha / the REAL-gate
// ref_intra_mode_info_cost_uv).
// ---------------------------------------------------------------------------

pub const UV_RD_SEARCH_MODE_ORDER_C: [usize; 14] = [0, 13, 2, 1, 9, 12, 10, 11, 4, 7, 6, 8, 5, 3];

/// C-side `rd_pick_intra_angle_sbuv` + `pick_intra_angle_routine_sbuv`.
/// Returns `Some((best_angle, rate_tokenonly, dist, skip, best_rd))`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_rd_pick_intra_angle_sbuv(
    env: &CUvEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    uv_mode: usize,
    rate_overhead: i32,
    mut best_rd: i64,
    angle_flat: &[i32],
    pal_flat: &[i32],
    try_palette: bool,
) -> Option<(i32, i32, i64, bool, i64)> {
    let mut best_angle_delta = 0i32;
    let mut best_stats: Option<(i32, i64, bool)> = None;
    let mut rd_cost = [i64::MAX; 10];

    let routine = |ad: i32,
                   best_rd_in: i64,
                   best_rd: &mut i64,
                   best_angle_delta: &mut i32,
                   best_stats: &mut Option<(i32, i64, bool)>,
                   recon_u: &mut [u16],
                   recon_v: &mut [u16]|
     -> i64 {
        let Some((rate, dist, _sse, wu, wv)) =
            c_txfm_uvrd(env, recon_u, recon_v, uv_mode, ad, best_rd_in)
        else {
            return i64::MAX;
        };
        let _ = (wu, wv);
        let this_rate = rate
            + c::ref_intra_mode_info_cost_uv(
                angle_flat,
                pal_flat,
                rate_overhead,
                uv_mode,
                env.bsize,
                ad,
                try_palette,
                false,
            );
        let this_rd = c::ref_rdcost(env.rdmult, this_rate, dist);
        if this_rd < *best_rd {
            *best_rd = this_rd;
            *best_angle_delta = ad;
            *best_stats = Some((rate, dist, false));
        }
        this_rd
    };

    let mut angle_delta = 0i32;
    while angle_delta <= 3 {
        for i in 0..2i32 {
            let best_rd_in = if best_rd == i64::MAX {
                i64::MAX
            } else {
                best_rd + (best_rd >> if angle_delta == 0 { 3 } else { 5 })
            };
            let this_rd = routine(
                (1 - 2 * i) * angle_delta,
                best_rd_in,
                &mut best_rd,
                &mut best_angle_delta,
                &mut best_stats,
                recon_u,
                recon_v,
            );
            rd_cost[(2 * angle_delta + i) as usize] = this_rd;
            if angle_delta == 0 {
                if this_rd == i64::MAX {
                    return None;
                }
                rd_cost[1] = this_rd;
                break;
            }
        }
        angle_delta += 2;
    }
    let mut angle_delta = 1i32;
    while angle_delta <= 3 {
        for i in 0..2i32 {
            let rd_thresh = best_rd + (best_rd >> 5);
            let skip_search = rd_cost[(2 * (angle_delta + 1) + i) as usize] > rd_thresh
                && rd_cost[(2 * (angle_delta - 1) + i) as usize] > rd_thresh;
            if !skip_search {
                routine(
                    (1 - 2 * i) * angle_delta,
                    best_rd,
                    &mut best_rd,
                    &mut best_angle_delta,
                    &mut best_stats,
                    recon_u,
                    recon_v,
                );
            }
        }
        angle_delta += 2;
    }
    best_stats.map(|(rate, dist, skip)| (best_angle_delta, rate, dist, skip, best_rd))
}

/// C-side `av1_rd_pick_intra_sbuv_mode` (non-palette). Returns the winner
/// `(uv_mode, angle, cfl_idx, cfl_signs, rate, rate_tokenonly, dist, skip,
/// best_rd)` + the per-candidate `Option<this_rd>` visit log.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_rd_pick_intra_sbuv_mode(
    env: &CUvEnv,
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    st: &mut c::RefCflState,
    cfl_allowed: bool,
    cfl_search_range: usize,
    uv_mode_costs: &[[i32; 14]; 13],
    angle_flat: &[i32],
    pal_flat: &[i32],
    cfl_costs_c: &[i32],
    try_palette: bool,
) -> (
    (usize, i32, u8, i8, i32, i32, i64, bool, i64),
    Vec<(usize, Option<i64>)>,
) {
    let mut best = (0usize, 0i32, 0u8, 0i8, 0i32, 0i32, 0i64, false, i64::MAX);
    let mut visits: Vec<(usize, Option<i64>)> = Vec::new();
    for &uv_mode in UV_RD_SEARCH_MODE_ORDER_C.iter() {
        let mode_rate = uv_mode_costs[env.luma_mode][uv_mode];
        if c::ref_rdcost(env.rdmult, mode_rate, 0) > best.8 {
            visits.push((uv_mode, None));
            continue;
        }
        let intra_mode = aom_entropy::partition::get_uv_mode(uv_mode);
        let is_directional = (1..=8).contains(&intra_mode);
        // enable flags all ON at aomenc defaults; uv mask ALL; prunes off.
        let mut angle_delta_uv = 0i32;
        let tokenonly: (i32, i64, bool);
        let mut cfl_fields = (0u8, 0i8);
        if uv_mode == 13 {
            if !cfl_allowed {
                visits.push((uv_mode, None));
                continue;
            }
            let uv_tx = aom_encode::intra_uv_rd::av1_get_tx_size_uv(
                env.bsize,
                env.lossless,
                env.ss_x,
                env.ss_y,
            );
            let Some((idx, js, stats)) = c_cfl_rd_pick_alpha(
                env,
                recon_u,
                recon_v,
                st,
                uv_tx,
                best.8,
                cfl_search_range,
                cfl_costs_c,
                uv_mode_costs[env.luma_mode][13],
            ) else {
                visits.push((uv_mode, None));
                continue;
            };
            tokenonly = (stats.rate, stats.dist, stats.skip);
            cfl_fields = (idx, js);
        } else if is_directional && aom_entropy::partition::use_angle_delta(env.bsize) {
            let rate_overhead = uv_mode_costs[env.luma_mode][uv_mode];
            let Some((ba, rate, dist, skip, _nb)) = c_rd_pick_intra_angle_sbuv(
                env,
                recon_u,
                recon_v,
                uv_mode,
                rate_overhead,
                best.8,
                angle_flat,
                pal_flat,
                try_palette,
            ) else {
                visits.push((uv_mode, None));
                continue;
            };
            angle_delta_uv = ba;
            tokenonly = (rate, dist, skip);
        } else {
            let Some((rate, dist, _sse, _wu, _wv)) =
                c_txfm_uvrd(env, recon_u, recon_v, uv_mode, 0, best.8)
            else {
                visits.push((uv_mode, None));
                continue;
            };
            tokenonly = (rate, dist, false);
        }
        let mode_cost = uv_mode_costs[env.luma_mode][uv_mode];
        let this_rate = tokenonly.0
            + c::ref_intra_mode_info_cost_uv(
                angle_flat,
                pal_flat,
                mode_cost,
                uv_mode,
                env.bsize,
                angle_delta_uv,
                try_palette,
                false,
            );
        let this_rd = c::ref_rdcost(env.rdmult, this_rate, tokenonly.1);
        visits.push((uv_mode, Some(this_rd)));
        if this_rd < best.8 {
            best = (
                uv_mode,
                angle_delta_uv,
                cfl_fields.0,
                cfl_fields.1,
                this_rate,
                tokenonly.0,
                tokenonly.1,
                tokenonly.2,
                this_rd,
            );
        }
    }
    assert!(best.8 < i64::MAX);
    (best, visits)
}

// ---- moved verbatim from intra_sby_mode_loop_diff.rs (shared with the
// ---- rd_pick_intra_mode_sb composition diff) --------------------------------
use aom_encode::intra_rd::{INTRA_MODES, MAX_ANGLE_DELTA, TOP_INTRA_MODEL_COUNT};
use aom_encode::mode_costs::{
    BLOCK_SIZE_GROUPS, BLOCK_SIZES_ALL, DIRECTIONAL_MODES, FILTER_INTRA_MODES, IntraModeCosts,
    KF_MODE_CONTEXTS, PALETTE_BSIZE_CTXS, PALETTE_Y_MODE_CONTEXTS, UV_INTRA_MODES,
    fill_intra_mode_costs,
};

/// The C loop's static gate chain (intra_mode_search.c:1555-1594) at the
/// aomenc-default tool flags — an independent transcription (the Rust side
/// gates live in IntraSbyGates::visits). Speed-0: every intra_mode_cfg flag
/// on, disable_smooth_intra off, intra_y_mode_mask all-ones,
/// use_mb_mode_cache off.
pub fn c_gate_visits(
    mode: usize,
    luma_delta_angle: i32,
    bsize: usize,
    skip_mask: &[bool; 13],
) -> bool {
    let is_directional = (1..=8).contains(&mode);
    // enable_diagonal_intra / enable_directional_intra / smooth flags /
    // enable_paeth_intra: all true (CLI defaults) — their `continue`s never
    // fire. directional_mode_skip_mask is the HOG output.
    if is_directional && skip_mask[mode] {
        return false;
    }
    // av1_use_angle_delta(bsize) = bsize >= BLOCK_8X8 (&& enable_angle_delta).
    if is_directional && bsize < 3 && luma_delta_angle != 0 {
        return false;
    }
    true // intra_y_mode_mask = INTRA_ALL
}

#[allow(clippy::type_complexity)]
pub struct CLoopOut {
    /// (mode, delta, tx_size, winners, rate, rate_tokenonly, dist, rd,
    ///  use_filter_intra, filter_intra_mode)
    pub best: Option<(
        usize,
        i32,
        usize,
        Vec<(usize, u16, u8)>,
        i32,
        i32,
        i64,
        i64,
        bool,
        usize,
    )>,
    pub rd_table: [[i64; 9]; 13],
    /// Candidates whose ALLINTRA factor was != 1.0 (coverage signal).
    pub factor_fired: usize,
}

/// The mode-info CDF set + the dual fill (Rust tables + the C reference
/// tables from the SAME CDFs) — the fill path is already differentially
/// validated in intra_mode_cost_diff.rs.
pub struct CdfSet {
    pub kf_y: Vec<u16>,
    pub y_mode: Vec<u16>,
    pub uv: Vec<u16>,
    pub fi_mode: Vec<u16>,
    pub fi: Vec<u16>,
    pub pal_y_mode: Vec<u16>,
    pub angle: Vec<u16>,
    pub intrabc: Vec<u16>,
}

pub fn gen_all_cdfs(rng: &mut Rng) -> CdfSet {
    let mut uv = gen_cdfs(rng, INTRA_MODES, UV_INTRA_MODES - 1, UV_INTRA_MODES + 1);
    uv.extend_from_slice(&gen_cdfs(
        rng,
        INTRA_MODES,
        UV_INTRA_MODES,
        UV_INTRA_MODES + 1,
    ));
    CdfSet {
        kf_y: gen_cdfs(
            rng,
            KF_MODE_CONTEXTS * KF_MODE_CONTEXTS,
            INTRA_MODES,
            INTRA_MODES + 1,
        ),
        y_mode: gen_cdfs(rng, BLOCK_SIZE_GROUPS, INTRA_MODES, INTRA_MODES + 1),
        uv,
        fi_mode: gen_cdfs(rng, 1, FILTER_INTRA_MODES, FILTER_INTRA_MODES + 1),
        fi: gen_cdfs(rng, BLOCK_SIZES_ALL, 2, 3),
        pal_y_mode: gen_cdfs(rng, PALETTE_BSIZE_CTXS * PALETTE_Y_MODE_CONTEXTS, 2, 3),
        angle: gen_cdfs(rng, DIRECTIONAL_MODES, 7, 8),
        intrabc: gen_cdfs(rng, 1, 2, 3),
    }
}

pub fn fill_both(cdfs: &CdfSet, enable_fi: bool) -> (Box<IntraModeCosts>, c::RefIntraModeCosts) {
    let want = c::ref_fill_intra_mode_costs(
        &cdfs.kf_y,
        &cdfs.y_mode,
        &cdfs.uv,
        &cdfs.fi_mode,
        &cdfs.fi,
        &cdfs.pal_y_mode,
        &cdfs.angle,
        &cdfs.intrabc,
        enable_fi,
    );
    let mut costs = IntraModeCosts::zeroed();
    fill_intra_mode_costs(
        &mut costs,
        &cdfs.kf_y,
        &cdfs.y_mode,
        &cdfs.uv,
        &cdfs.fi_mode,
        &cdfs.fi,
        &cdfs.pal_y_mode,
        &cdfs.angle,
        &cdfs.intrabc,
        enable_fi,
    );
    (costs, want)
}

/// The C-side mode loop: an independent transcription of
/// intra_mode_search.c:1545-1661 over REAL reference pieces.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn c_mode_loop(
    bsize: usize,
    geometry: (i32, i32, usize, usize, usize),
    sb_size: usize,
    recon_c: &mut [u16],
    src: &[u16],
    reduced: bool,
    bd: u8,
    plane_rows_c: &[i16],
    dequant: [i16; 2],
    above_ctx: &[i8],
    left_ctx: &[i8],
    rdmult: i32,
    best_rd_in: i64,
    coeff_tbls: (&[i32], &[i32], &[i32], &[i32], &[i32], &[i32], &[i32]),
    ttc_tables: (&[i32], &[i32]),
    skip_costs: &[[i32; 2]; 3],
    skip_ctx: usize,
    ts_flat: &[i32],
    tx_size_ctx: usize,
    source_variance: u32,
    skip_mask: &[bool; 13],
    neigh_modes: (Option<i32>, Option<i32>),
    qindex: i32,
    c_costs: &c::RefIntraModeCosts,
    allintra: bool,
    cvar: &mut [i32],
    clog: &mut [f64],
) -> CLoopOut {
    let (mi_row, mi_col, ref_off, src_off, stride) = geometry;
    let (above_mode, left_mode) = neigh_modes;
    // bmode_costs = y_mode_costs[above_ctx][left_ctx] — the kf ctx pair from
    // intra_mode_context[above/left mode] (absent neighbour = DC_PRED),
    // costed from the SAME kf CDF the Rust tables were filled from, via the
    // real av1_cost_tokens_from_cdf (ref_cost_tokens_from_cdf).
    const INTRA_MODE_CONTEXT: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
    let actx = INTRA_MODE_CONTEXT[above_mode.unwrap_or(0) as usize];
    let lctx = INTRA_MODE_CONTEXT[left_mode.unwrap_or(0) as usize];
    let bmode_costs = &c_costs.y_mode[(actx * 5 + lctx) * 13..(actx * 5 + lctx) * 13 + 13];

    let mut best_rd = best_rd_in;
    #[allow(clippy::type_complexity)]
    let mut best: Option<(
        usize,
        i32,
        usize,
        Vec<(usize, u16, u8)>,
        i32,
        i32,
        i64,
        i64,
        bool,
        usize,
    )> = None;
    let mut best_model_rd = i64::MAX;
    let mut top_model = [i64::MAX; TOP_INTRA_MODEL_COUNT];
    let mut rd_table = [[i64::MAX; 9]; 13];
    let mut factor_fired = 0usize;
    let model_tx = MAX_TXSIZE_LOOKUP[bsize].min(3);

    for mode_idx in 0..61 {
        // REAL exported set_y_mode_and_delta_angle
        // (prune_luma_odd_delta_angles_in_intra = 0 at speed 0).
        let (mode_i, delta) = c::ref_set_y_mode_and_delta_angle(mode_idx, false);
        let mode = mode_i as usize;
        if !c_gate_visits(mode, delta, bsize, skip_mask) {
            continue;
        }
        // prune_luma_odd_delta_angles_using_rd_cost: sf OFF at speed 0 — the
        // C body returns 0 immediately.

        // intra_model_rd (prediction walk mutates the C recon).
        let this_model_rd = c_intra_model_rd(
            bsize,
            model_tx,
            recon_c,
            src,
            (mi_row, mi_col, ref_off, src_off, stride),
            mode,
            delta,
            false,
            0,
            bd,
        );
        let idx = c::ref_get_model_rd_index_for_pruning(
            mode,
            qindex,
            TOP_INTRA_MODEL_COUNT as i32,
            false,
            left_mode.map(|m| m as usize),
            above_mode.map(|m| m as usize),
        );
        if c::ref_prune_intra_y_mode(
            this_model_rd,
            &mut best_model_rd,
            &mut top_model,
            TOP_INTRA_MODEL_COUNT,
            idx as usize,
        ) {
            continue;
        }

        // av1_pick_uniform_tx_size_type_yrd with the RUNNING best_rd.
        let Some((tx_size, _rd_pick, rate_tok_raw, dist, _sse, winners)) =
            c_pick_uniform_tx_size_type_yrd(
                bsize,
                (mi_row, mi_col, ref_off, src_off, stride),
                recon_c,
                src,
                mode,
                delta,
                false,
                0,
                false,
                reduced,
                bd,
                plane_rows_c,
                dequant,
                above_ctx,
                left_ctx,
                rdmult,
                best_rd,
                coeff_tbls,
                ttc_tables,
                skip_costs,
                skip_ctx,
                ts_flat,
                tx_size_ctx,
                source_variance,
            )
        else {
            continue; // rate == INT_MAX
        };

        // tx-size cost subtraction (lossless off; block_signals_txsize =
        // bsize > BLOCK_4X4).
        let mut rate_tokenonly = rate_tok_raw;
        if bsize > 0 {
            rate_tokenonly -= c::ref_tx_size_cost(
                ts_flat,
                true,
                bsize as i32,
                tx_size as i32,
                tx_size_ctx as i32,
            );
        }
        // intra_mode_info_cost_y over the REAL shim (no palette / fi off /
        // no intrabc; enable_filter_intra on -> fi flag bit costed on
        // eligible bsizes; angle-delta rate on directional modes).
        let mode_info_rate = c::ref_intra_mode_info_cost_y(
            c_costs,
            bmode_costs[mode],
            mode as i32,
            bsize as i32,
            delta,
            false,
            0,
            false,
            false,
            0,
            0,
            true,
            false,
        );
        let this_rate = rate_tok_raw + mode_info_rate;
        let mut this_rd = c::ref_rdcost(rdmult, this_rate, dist);
        if allintra && this_rd != i64::MAX {
            let factor = c::ref_intra_rd_variance_factor(
                0,
                src,
                src_off,
                stride,
                recon_c,
                ref_off,
                stride,
                bsize,
                sb_size,
                mi_row,
                mi_col,
                1 << 12,
                1 << 12,
                bd,
                cvar,
                clog,
            );
            if factor != 1.0 {
                factor_fired += 1;
            }
            this_rd = (this_rd as f64 * factor) as i64;
        }
        rd_table[mode][(delta + MAX_ANGLE_DELTA + 1) as usize] = this_rd;
        // store_winner_mode_stats: MULTI_WINNER_MODE_OFF no-op.
        if this_rd < best_rd {
            best_rd = this_rd;
            best = Some((
                mode,
                delta,
                tx_size,
                winners,
                this_rate,
                rate_tokenonly,
                dist,
                this_rd,
                false,
                0,
            ));
        }
    }

    // rd_pick_filter_intra_sby (intra_mode_search.c:1672 + 231): runs when a
    // Y mode beat best_rd_in AND av1_filter_intra_allowed_bsize (the REAL
    // gate — both dims <= 32; kills the loop for the rect-partition
    // 64x32/32x64 leaves; enable_filter_intra on). mbmi carries the STALE
    // angle_delta of loop index 60 (set_y_mode_and_delta_angle mutates
    // before the gates) — mirrored via the last (mode, delta) pair.
    let stale_delta = c::ref_set_y_mode_and_delta_angle(60, false).1;
    if best.is_some() && c::ref_filter_intra_allowed_bsize(true, bsize) {
        let best_mode_so_far = best.as_ref().map_or(0, |b| b.0);
        let _ = best_mode_so_far; // prune_filter_intra_level = 0: no level-1 gate
        for fim in 0..5usize {
            // model_intra_yrd_and_prune: the model walk (fi prediction) +
            // the INTEGER prune on the SHARED best_model_rd.
            let this_model_rd = c_intra_model_rd(
                bsize,
                model_tx,
                recon_c,
                src,
                (mi_row, mi_col, ref_off, src_off, stride),
                0, // DC_PRED
                stale_delta,
                true,
                fim,
                bd,
            );
            if best_model_rd != i64::MAX && this_model_rd > best_model_rd + (best_model_rd >> 2) {
                continue;
            } else if this_model_rd < best_model_rd {
                best_model_rd = this_model_rd;
            }
            let Some((tx_size, _rd_pick, rate_tok_raw, dist, _sse, winners)) =
                c_pick_uniform_tx_size_type_yrd(
                    bsize,
                    (mi_row, mi_col, ref_off, src_off, stride),
                    recon_c,
                    src,
                    0,
                    stale_delta,
                    true,
                    fim,
                    false,
                    reduced,
                    bd,
                    plane_rows_c,
                    dequant,
                    above_ctx,
                    left_ctx,
                    rdmult,
                    best_rd,
                    coeff_tbls,
                    ttc_tables,
                    skip_costs,
                    skip_ctx,
                    ts_flat,
                    tx_size_ctx,
                    source_variance,
                )
            else {
                continue;
            };
            // NOTE: no tx-size-cost subtraction in the filter-intra path —
            // *rate_tokenonly takes the raw tx-search rate.
            let mode_info_rate = c::ref_intra_mode_info_cost_y(
                c_costs,
                bmode_costs[0], // DC_PRED
                0,
                bsize as i32,
                0,
                true,
                fim as i32,
                false,
                false,
                0,
                0,
                true,
                false,
            );
            let this_rate = rate_tok_raw + mode_info_rate;
            let mut this_rd = c::ref_rdcost(rdmult, this_rate, dist);
            if allintra && this_rd != i64::MAX {
                let factor = c::ref_intra_rd_variance_factor(
                    0,
                    src,
                    src_off,
                    stride,
                    recon_c,
                    ref_off,
                    stride,
                    bsize,
                    sb_size,
                    mi_row,
                    mi_col,
                    1 << 12,
                    1 << 12,
                    bd,
                    cvar,
                    clog,
                );
                if factor != 1.0 {
                    factor_fired += 1;
                }
                this_rd = (this_rd as f64 * factor) as i64;
            }
            if this_rd < best_rd {
                best_rd = this_rd;
                best = Some((
                    0,
                    stale_delta,
                    tx_size,
                    winners,
                    this_rate,
                    rate_tok_raw,
                    dist,
                    this_rd,
                    true,
                    fim,
                ));
            }
        }
    }
    CLoopOut {
        best,
        rd_table,
        factor_fired,
    }
}

// ---- the C-side av1_encode_intra_block_plane (luma) walk over REAL pieces
// ---- (shared by encode_intra_plane_diff + the rd_pick composition diff) ----

/// One C-side re-encoded txb: (tx_type, eob, entropy ctx, qcoeff, dqcoeff).
pub type CTxb = (usize, u16, u8, Vec<i32>, Vec<i32>);

/// The seven coefficient cost tables (txb_skip, base_eob, base, eob_extra,
/// dc_sign, lps, eob).
pub type CCoeffTbls<'a> = (
    &'a [i32],
    &'a [i32],
    &'a [i32],
    &'a [i32],
    &'a [i32],
    &'a [i32],
    &'a [i32],
);

/// The C-side `av1_encode_intra_block_plane(AOM_PLANE_Y)` walk arguments.
pub struct CEncPlaneArgs<'a> {
    pub bsize: usize,
    pub tx_size: usize,
    /// (mi_row, mi_col, ref_off, src_off, stride)
    pub geometry: (i32, i32, usize, usize, usize),
    pub sb_size: usize,
    /// `mbmi->partition` — the has_top_right/has_bottom_left availability
    /// input (0 NONE / 1 HORZ / 2 VERT for the ported tree shapes).
    pub partition: usize,
    pub src: &'a [u16],
    pub mode: usize,
    pub angle_delta: i32,
    pub use_fi: bool,
    pub fi_mode: usize,
    pub skip_txfm: bool,
    /// `is_trellis_used(enable_optimize_b, dry_run)`.
    pub use_trellis: bool,
    /// `enable_optimize_b != NO_TRELLIS_OPT` (the ta/tl load gate).
    pub load_ctx: bool,
    pub sharpness: i32,
    pub reduced: bool,
    pub bd: u8,
    pub plane_rows_c: &'a [i16],
    pub dequant: [i16; 2],
    pub above_ctx: &'a [i8],
    pub left_ctx: &'a [i8],
    pub rdmult: i32,
    pub coeff_tbls: CCoeffTbls<'a>,
    /// `xd->cfl.store_y` + the cfl subsampling.
    pub store: bool,
    pub ss: (i32, i32),
}

/// The C-side walk: per txb ref_intra_avail + ref_hbd_predict_intra (into
/// the C recon) -> ref_highbd_subtract_block -> ref_get_tx_type_y (REAL,
/// over the marshalled block-local map) -> ref_fwd_txfm2d +
/// ref_quant_plane_rows (FP when trellis / B when not) -> [trellis:
/// ref_get_txb_ctx + ref_optimize_txb + ref_txb_entropy_context] ->
/// ref_inv_txfm2d_add -> [eob 0: ref_update_txk_array DCT reset] ->
/// [store: ref_cfl_store_tx] -> the av1_set_txb_context stamp.
/// Returns (txbs, ta, tl).
#[allow(clippy::type_complexity)]
pub fn c_encode_intra_block_plane_y(
    a: &CEncPlaneArgs,
    recon_c: &mut [u16],
    map_c: &mut [u8],
    cfl_c: &mut c::RefCflState,
) -> (Vec<CTxb>, Vec<i8>, Vec<i8>) {
    use aom_encode::tx_search::trellis_rdmult_intra;
    const MI_W_B: [usize; 22] = [
        1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
    ];
    const MI_H_B: [usize; 22] = [
        1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
    ];
    let (mi_row, mi_col, ref_off, src_off, stride) = a.geometry;
    let (bw, bh) = (BLK_W[a.bsize], BLK_H[a.bsize]);
    let (txw, txh) = (TX_W[a.tx_size], TX_H[a.tx_size]);
    let (txwu, txhu) = (txw >> 2, txh >> 2);
    let (mbw, mbh) = (MI_W_B[a.bsize], MI_H_B[a.bsize]);
    let map_stride = mbw;
    let (txb_skip, base_eob, base, eob_extra, dc_sign, lps, eob_tbl) = a.coeff_tbls;

    let mut ta_c = vec![0i8; mbw];
    let mut tl_c = vec![0i8; mbh];
    if a.load_ctx {
        ta_c.copy_from_slice(&a.above_ctx[..mbw]);
        tl_c.copy_from_slice(&a.left_ctx[..mbh]);
    }
    let mut txbs: Vec<CTxb> = Vec::new();
    for blk_row in (0..mbh).step_by(txhu) {
        for blk_col in (0..mbw).step_by(txwu) {
            let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                a.sb_size,
                a.bsize,
                mi_row,
                mi_col,
                true,
                true,
                1 << 16,
                1 << 16,
                a.partition,
                a.tx_size,
                0,
                0,
                blk_row as i32,
                blk_col as i32,
                bw as i32,
                bh as i32,
                512,
                512,
                a.mode,
                a.angle_delta * 3,
                a.use_fi,
            );
            let txb_off = ref_off + (blk_row * stride + blk_col) * 4;
            let pred = c::ref_hbd_predict_intra(
                recon_c,
                txb_off,
                stride,
                a.mode,
                a.angle_delta * 3,
                a.use_fi,
                a.fi_mode,
                false,
                0,
                a.tx_size,
                txw,
                txh,
                n_top,
                n_tr,
                n_left,
                n_bl,
                a.bd as i32,
            );
            for r in 0..txh {
                recon_c[txb_off + r * stride..txb_off + r * stride + txw]
                    .copy_from_slice(&pred[r * txw..r * txw + txw]);
            }

            let mut tx_type_c = 0usize;
            let (mut qc, mut dqc): (Vec<i32>, Vec<i32>) = (Vec::new(), Vec::new());
            let (eob_c, ctx_c);
            if a.skip_txfm {
                eob_c = 0usize;
                ctx_c = 0u8;
            } else {
                let src_txb_off = src_off + (blk_row * stride + blk_col) * 4;
                let mut residual = vec![0i16; txw * txh];
                c::ref_highbd_subtract_block(
                    txh,
                    txw,
                    &mut residual,
                    txw,
                    &a.src[src_txb_off..],
                    stride,
                    &pred,
                    txw,
                );
                tx_type_c = c::ref_get_tx_type_y(
                    false, a.tx_size, a.reduced, map_c, map_stride, blk_row, blk_col,
                );
                let n_coeffs = txb_wide(a.tx_size) * txb_high(a.tx_size);
                let coeff = c::ref_fwd_txfm2d(a.tx_size, &residual, txw, tx_type_c);
                let tcoeff = coeff[..n_coeffs].to_vec();
                qc = vec![0i32; n_coeffs];
                dqc = vec![0i32; n_coeffs];
                let kind_c = if a.use_trellis { 0 } else { 1 }; // FP : B
                let eob0 = c::ref_quant_plane_rows(
                    kind_c,
                    a.bd > 8,
                    &tcoeff,
                    a.plane_rows_c,
                    scan(a.tx_size, tx_type_c),
                    aom_txb::iscan(a.tx_size, tx_type_c),
                    aom_encode::tx_scale(a.tx_size),
                    &mut qc,
                    &mut dqc,
                ) as usize;
                if a.use_trellis {
                    let (txb_skip_ctx_c, dc_sign_ctx_c) = c::ref_get_txb_ctx(
                        a.bsize,
                        a.tx_size,
                        0,
                        &ta_c[blk_col..],
                        &tl_c[blk_row..],
                    );
                    if eob0 == 0 {
                        eob_c = 0;
                        ctx_c = 0;
                    } else {
                        let (ne, _rate) = c::ref_optimize_txb(
                            a.tx_size,
                            tx_type_c,
                            &mut qc,
                            &mut dqc,
                            &tcoeff,
                            eob0,
                            &a.dequant,
                            // plane 0: table select irrelevant (17 both).
                            trellis_rdmult_intra(a.rdmult, a.sharpness, a.bd, 0, true),
                            dc_sign_ctx_c as usize,
                            txb_skip_ctx_c as usize,
                            a.sharpness,
                            scan(a.tx_size, tx_type_c),
                            txb_skip,
                            base_eob,
                            base,
                            eob_extra,
                            dc_sign,
                            lps,
                            eob_tbl,
                        );
                        eob_c = ne;
                        ctx_c = c::ref_txb_entropy_context(&qc, a.tx_size, tx_type_c, ne);
                    }
                } else {
                    eob_c = eob0;
                    ctx_c = c::ref_txb_entropy_context(&qc, a.tx_size, tx_type_c, eob0);
                }
            }

            if eob_c > 0 {
                let mut tight = pred.clone();
                c::ref_inv_txfm2d_add(a.tx_size, &dqc, &mut tight, txw, tx_type_c, a.bd as i32);
                for r in 0..txh {
                    recon_c[txb_off + r * stride..txb_off + r * stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }
            if eob_c == 0 {
                c::ref_update_txk_array(map_c, map_stride, blk_row, blk_col, a.tx_size, 0);
            }
            if a.store {
                c::ref_cfl_store_tx(
                    cfl_c,
                    recon_c,
                    ref_off,
                    stride,
                    blk_row as i32,
                    blk_col as i32,
                    a.tx_size,
                    a.bsize,
                    mi_row,
                    mi_col,
                    a.ss.0,
                    a.ss.1,
                    a.bd,
                );
            }
            for x in ta_c[blk_col..blk_col + txwu].iter_mut() {
                *x = ctx_c as i8;
            }
            for x in tl_c[blk_row..blk_row + txhu].iter_mut() {
                *x = ctx_c as i8;
            }
            txbs.push((tx_type_c, eob_c as u16, ctx_c, qc, dqc));
        }
    }
    (txbs, ta_c, tl_c)
}

/// The C-side CHROMA walk of `av1_encode_intra_block_plane` (plane 1/2) over
/// REAL pieces: per txb prediction (plain via ref_hbd_predict_intra with the
/// UV mode/angle, or the CfL arm — fresh DC prediction + the REAL
/// `av1_cfl_predict_block` with the WINNER's alpha_idx/joint_sign; the
/// dc-pred cache is inactive outside cfl_rd_pick_alpha) ->
/// ref_highbd_subtract_block -> the REAL `av1_get_tx_type` UV arm
/// (ref_get_tx_type_uv_intra) -> ref_fwd_txfm2d + ref_quant_plane_rows
/// (FP/B) -> [trellis: ref_get_txb_ctx(plane_bsize, plane) +
/// ref_optimize_txb at the plane trellis rdmult + ref_txb_entropy_context]
/// -> ref_inv_txfm2d_add. NO txk reset, NO cfl store (plane != 0 gates).
#[allow(clippy::too_many_arguments)]
pub fn c_encode_intra_block_plane_uv(
    env: &CUvEnv,
    plane: usize,
    uv_mode: usize,
    angle_delta_uv: i32,
    mut cfl: Option<(&mut c::RefCflState, i32, i32)>,
    tx_size: usize,
    skip_txfm: bool,
    use_trellis: bool,
    load_ctx: bool,
    sharpness: i32,
    recon: &mut [u16],
) -> (Vec<CTxb>, Vec<i8>, Vec<i8>) {
    use aom_encode::tx_search::trellis_rdmult_intra;
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
    let (rows_c, dequant) = if plane == 1 {
        (env.rows_u_c, env.dequant_u)
    } else {
        (env.rows_v_c, env.dequant_v)
    };
    let (txb_skip, base_eob, base, eob_extra, dc_sign, lps, eob_tbl) = env.coeff_tbls;

    let mut ta_c = vec![0i8; max_w];
    let mut tl_c = vec![0i8; max_h];
    if load_ctx {
        ta_c.copy_from_slice(&env.above_ctx[pi][..max_w]);
        tl_c.copy_from_slice(&env.left_ctx[pi][..max_h]);
    }

    let mut txbs: Vec<CTxb> = Vec::new();
    let mut blk_row = 0usize;
    while blk_row < max_h {
        let mut blk_col = 0usize;
        while blk_col < max_w {
            let txb_off = env.ref_off[pi] + (blk_row * env.stride + blk_col) * 4;
            // --- av1_predict_intra_block_facade ---
            if let Some((st, alpha_idx, joint_sign)) = cfl.as_mut() {
                assert_eq!((blk_row, blk_col), (0, 0), "CfL block == tx block");
                // Fresh DC prediction (no dc-pred cache in the encode pass).
                let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                    12,
                    env.bsize,
                    env.mi_row,
                    env.mi_col,
                    true,
                    true,
                    1 << 16,
                    1 << 16,
                    env.partition,
                    tx_size,
                    env.ss_x as i32,
                    env.ss_y as i32,
                    blk_row as i32,
                    blk_col as i32,
                    wpx,
                    hpx,
                    512,
                    512,
                    mode,
                    0,
                    false,
                );
                let pred = c::ref_hbd_predict_intra(
                    recon,
                    txb_off,
                    env.stride,
                    mode,
                    0,
                    false,
                    0,
                    false,
                    0,
                    tx_size,
                    txw,
                    txh,
                    n_top,
                    n_tr,
                    n_left,
                    n_bl,
                    env.bd as i32,
                );
                for r in 0..txh {
                    recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                        .copy_from_slice(&pred[r * txw..r * txw + txw]);
                }
                c::ref_cfl_predict_block(
                    st,
                    recon,
                    txb_off,
                    env.stride,
                    tx_size,
                    plane,
                    *alpha_idx,
                    *joint_sign,
                    env.bsize,
                    env.lossless,
                    env.ss_x as i32,
                    env.ss_y as i32,
                    env.bd,
                );
            } else {
                let (n_top, n_tr, n_left, n_bl) = c::ref_intra_avail(
                    12,
                    env.bsize,
                    env.mi_row,
                    env.mi_col,
                    true,
                    true,
                    1 << 16,
                    1 << 16,
                    env.partition,
                    tx_size,
                    env.ss_x as i32,
                    env.ss_y as i32,
                    blk_row as i32,
                    blk_col as i32,
                    wpx,
                    hpx,
                    512,
                    512,
                    mode,
                    angle_delta_uv * 3,
                    false,
                );
                let pred = c::ref_hbd_predict_intra(
                    recon,
                    txb_off,
                    env.stride,
                    mode,
                    angle_delta_uv * 3,
                    false,
                    0,
                    false,
                    0,
                    tx_size,
                    txw,
                    txh,
                    n_top,
                    n_tr,
                    n_left,
                    n_bl,
                    env.bd as i32,
                );
                for r in 0..txh {
                    recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                        .copy_from_slice(&pred[r * txw..r * txw + txw]);
                }
            }

            let mut tx_type_c = 0usize;
            let (mut qc, mut dqc): (Vec<i32>, Vec<i32>) = (Vec::new(), Vec::new());
            let (eob_c, ctx_c);
            if skip_txfm {
                eob_c = 0usize;
                ctx_c = 0u8;
            } else {
                // Prediction snapshot (recon holds it) for subtract + recon base.
                let mut pred = vec![0u16; txw * txh];
                for r in 0..txh {
                    pred[r * txw..r * txw + txw].copy_from_slice(
                        &recon[txb_off + r * env.stride..txb_off + r * env.stride + txw],
                    );
                }
                let src_txb_off = env.src_off[pi] + (blk_row * env.stride + blk_col) * 4;
                let mut residual = vec![0i16; txw * txh];
                c::ref_highbd_subtract_block(
                    txh,
                    txw,
                    &mut residual,
                    txw,
                    &src[src_txb_off..],
                    env.stride,
                    &pred,
                    txw,
                );
                tx_type_c =
                    c::ref_get_tx_type_uv_intra(uv_mode, env.lossless, tx_size, env.reduced);
                let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
                let coeff = c::ref_fwd_txfm2d(tx_size, &residual, txw, tx_type_c);
                let tcoeff = coeff[..n_coeffs].to_vec();
                qc = vec![0i32; n_coeffs];
                dqc = vec![0i32; n_coeffs];
                let kind_c = if use_trellis { 0 } else { 1 }; // FP : B
                let eob0 = c::ref_quant_plane_rows(
                    kind_c,
                    env.bd > 8,
                    &tcoeff,
                    rows_c,
                    scan(tx_size, tx_type_c),
                    aom_txb::iscan(tx_size, tx_type_c),
                    aom_encode::tx_scale(tx_size),
                    &mut qc,
                    &mut dqc,
                ) as usize;
                if use_trellis {
                    let (txb_skip_ctx_c, dc_sign_ctx_c) = c::ref_get_txb_ctx(
                        plane_bsize,
                        tx_size,
                        plane,
                        &ta_c[blk_col..],
                        &tl_c[blk_row..],
                    );
                    if eob0 == 0 {
                        eob_c = 0;
                        ctx_c = 0;
                    } else {
                        let (ne, _rate) = c::ref_optimize_txb(
                            tx_size,
                            tx_type_c,
                            &mut qc,
                            &mut dqc,
                            &tcoeff,
                            eob0,
                            &dequant,
                            trellis_rdmult_intra(
                                env.rdmult,
                                sharpness,
                                env.bd,
                                plane,
                                env.use_chroma_trellis_rd_mult,
                            ),
                            dc_sign_ctx_c as usize,
                            txb_skip_ctx_c as usize,
                            sharpness,
                            scan(tx_size, tx_type_c),
                            txb_skip,
                            base_eob,
                            base,
                            eob_extra,
                            dc_sign,
                            lps,
                            eob_tbl,
                        );
                        eob_c = ne;
                        ctx_c = c::ref_txb_entropy_context(&qc, tx_size, tx_type_c, ne);
                    }
                } else {
                    eob_c = eob0;
                    ctx_c = c::ref_txb_entropy_context(&qc, tx_size, tx_type_c, eob0);
                }
            }

            if eob_c > 0 {
                let mut tight = vec![0u16; txw * txh];
                for r in 0..txh {
                    tight[r * txw..r * txw + txw].copy_from_slice(
                        &recon[txb_off + r * env.stride..txb_off + r * env.stride + txw],
                    );
                }
                c::ref_inv_txfm2d_add(tx_size, &dqc, &mut tight, txw, tx_type_c, env.bd as i32);
                for r in 0..txh {
                    recon[txb_off + r * env.stride..txb_off + r * env.stride + txw]
                        .copy_from_slice(&tight[r * txw..r * txw + txw]);
                }
            }
            // plane != 0: no update_txk_array, no cfl_store_tx.

            for x in ta_c[blk_col..blk_col + txwu].iter_mut() {
                *x = ctx_c as i8;
            }
            for x in tl_c[blk_row..blk_row + txhu].iter_mut() {
                *x = ctx_c as i8;
            }
            txbs.push((tx_type_c, eob_c as u16, ctx_c, qc, dqc));
            blk_col += txwu;
        }
        blk_row += txhu;
    }
    (txbs, ta_c, tl_c)
}

use aom_encode::encode_sb::{LeafWinner, SbTree};
use aom_encode::intra_uv_rd::{av1_get_tx_size_uv as uv_txsz, chroma_plane_offset as uv_off};

/// The C-side walk: transcribed control flow over REAL leaf chains + REAL
/// context-stamp facades. Mirrors `encode_sb_dry` shape-for-shape; every
/// pixel/context mutation goes through validated `ref_` entry points.
fn c_split_subsize(bsize: usize) -> usize {
    match bsize {
        3 => 0,
        6 => 3,
        9 => 6,
        12 => 9,
        _ => unreachable!(),
    }
}

pub struct COracle<'a> {
    pub ss: (usize, usize),
    pub monochrome: bool,
    pub bd: u8,
    pub reduced: bool,
    pub sharpness: i32,
    pub use_trellis: bool,
    pub load_ctx: bool,
    pub use_chroma_tbl: bool,
    pub mi_rows: i32,
    pub mi_cols: i32,
    pub base_y: usize,
    pub stride: usize,
    pub base_uv: usize,
    pub rdmult: i32,
    pub src_y: &'a [u16],
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    pub plane_rows_y: &'a [i16],
    pub rows_u_c: &'a [i16],
    pub rows_v_c: &'a [i16],
    pub dequant_y: [i16; 2],
    pub dequant_u: [i16; 2],
    pub dequant_v: [i16; 2],
    pub coeff_tbls_y: CCoeffTbls<'a>,
    pub coeff_tbls_uv: CCoeffTbls<'a>,
    pub ttc: (&'a [i32], &'a [i32]),
    // Tile context arrays (the C-side TileCtxState mirror).
    pub above_e: [Vec<i8>; 3],
    pub left_e: [[i8; 32]; 3],
    pub above_p: [i8; 64],
    pub left_p: [i8; 32],
    pub above_t: Vec<u8>,
    pub left_t: [u8; 32],
}

pub type CLeafOut = (
    i32,
    i32,
    usize,
    bool,
    bool,
    Vec<CTxb>,
    Option<Vec<CTxb>>,
    Option<Vec<CTxb>>,
);

impl COracle<'_> {
    #[allow(clippy::too_many_arguments)]
    pub fn encode_b(
        &mut self,
        recon_y: &mut [u16],
        recon_u: &mut [u16],
        recon_v: &mut [u16],
        cfl: &mut c::RefCflState,
        w: &mut LeafWinner,
        mi_row: i32,
        mi_col: i32,
        partition: usize,
        out: &mut Vec<CLeafOut>,
    ) {
        let bsize = w.bsize;
        let (ss_x, ss_y) = self.ss;
        let (mbw, mbh) = (MI_W[bsize], MI_H[bsize]);
        let chroma_ref =
            c::ref_is_chroma_reference(mi_row, mi_col, bsize, ss_x as i32, ss_y as i32);
        let store_y = c::ref_store_cfl_required(self.monochrome, chroma_ref, w.uv_mode);
        let ref_off_y = self.base_y + (mi_row as usize * 4) * self.stride + mi_col as usize * 4;
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;

        // Plane 0 re-encode (REAL chain; ta/tl loaded from the tile arrays).
        let above_y: Vec<i8> = self.above_e[0][a0..a0 + mbw].to_vec();
        let left_y: Vec<i8> = self.left_e[0][l0..l0 + mbh].to_vec();
        let ca = CEncPlaneArgs {
            partition,
            bsize,
            tx_size: w.tx_size,
            geometry: (mi_row, mi_col, ref_off_y, ref_off_y, self.stride),
            sb_size: 12,
            src: self.src_y,
            mode: w.mode,
            angle_delta: w.angle_delta_y,
            use_fi: w.use_filter_intra,
            fi_mode: w.filter_intra_mode,
            skip_txfm: w.skip_txfm,
            use_trellis: self.use_trellis,
            load_ctx: self.load_ctx,
            sharpness: self.sharpness,
            reduced: self.reduced,
            bd: self.bd,
            plane_rows_c: self.plane_rows_y,
            dequant: self.dequant_y,
            above_ctx: &above_y,
            left_ctx: &left_y,
            rdmult: self.rdmult,
            coeff_tbls: self.coeff_tbls_y,
            store: store_y,
            ss: (ss_x as i32, ss_y as i32),
        };
        let mut map_c = std::mem::take(&mut w.tx_type_map);
        let (y_txbs, _ta, _tl) = c_encode_intra_block_plane_y(&ca, recon_y, &mut map_c, cfl);
        w.tx_type_map = map_c;

        // Planes 1/2.
        let uv_tx = uv_txsz(bsize, false, ss_x, ss_y);
        let mut u_txbs = None;
        let mut v_txbs = None;
        if !self.monochrome && chroma_ref {
            let ref_off_uv = uv_off(self.base_uv, self.stride, mi_row, mi_col, bsize, ss_x, ss_y);
            let plane_bsize = aom_entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
            let (pmw, pmh) = (MI_W[plane_bsize], MI_H[plane_bsize]);
            let au = (mi_col >> ss_x) as usize;
            let lu = ((mi_row & 31) >> ss_y) as usize;
            let above_u: Vec<i8> = self.above_e[1][au..au + pmw].to_vec();
            let left_u: Vec<i8> = self.left_e[1][lu..lu + pmh].to_vec();
            let above_v: Vec<i8> = self.above_e[2][au..au + pmw].to_vec();
            let left_v: Vec<i8> = self.left_e[2][lu..lu + pmh].to_vec();
            let cenv = CUvEnv {
                partition,
                bsize,
                mi_row,
                mi_col,
                ss_x,
                ss_y,
                ref_off: [ref_off_uv, ref_off_uv],
                src_off: [ref_off_uv, ref_off_uv],
                stride: self.stride,
                src_u: self.src_u,
                src_v: self.src_v,
                luma_mode: w.mode,
                luma_use_fi: w.use_filter_intra,
                luma_fi_mode: w.filter_intra_mode,
                lossless: false,
                reduced: self.reduced,
                bd: self.bd,
                rows_u_c: self.rows_u_c,
                rows_v_c: self.rows_v_c,
                dequant_u: self.dequant_u,
                dequant_v: self.dequant_v,
                above_ctx: [&above_u, &above_v],
                left_ctx: [&left_u, &left_v],
                rdmult: self.rdmult,
                coeff_tbls: self.coeff_tbls_uv,
                ttc_tables: self.ttc,
                use_chroma_trellis_rd_mult: self.use_chroma_tbl,
            };
            let use_cfl = w.uv_mode == 13;
            for plane in [1usize, 2usize] {
                let recon = if plane == 1 {
                    &mut *recon_u
                } else {
                    &mut *recon_v
                };
                let (txbs, _ta, _tl) = c_encode_intra_block_plane_uv(
                    &cenv,
                    plane,
                    w.uv_mode,
                    w.angle_delta_uv,
                    if use_cfl {
                        Some((cfl, w.cfl_alpha_idx, w.cfl_alpha_signs))
                    } else {
                        None
                    },
                    uv_tx,
                    w.skip_txfm,
                    self.use_trellis,
                    self.load_ctx,
                    self.sharpness,
                    recon,
                );
                if plane == 1 {
                    u_txbs = Some(txbs);
                } else {
                    v_txbs = Some(txbs);
                }
            }
        }

        // av1_update_intra_mb_txb_context at DRY_RUN: REAL tx_type re-read +
        // REAL av1_get_txb_entropy_context + REAL av1_set_entropy_contexts.
        {
            let (txwu, txhu) = (TX_W[w.tx_size] >> 2, TX_H[w.tx_size] >> 2);
            let mut k = 0usize;
            for blk_row in (0..mbh).step_by(txhu) {
                for blk_col in (0..mbw).step_by(txwu) {
                    let tt = c::ref_get_tx_type_y(
                        false,
                        w.tx_size,
                        self.reduced,
                        &w.tx_type_map,
                        mbw,
                        blk_row,
                        blk_col,
                    );
                    let (_, eob, _, qc, _) = &y_txbs[k];
                    let cul = if *eob == 0 {
                        0u8
                    } else {
                        c::ref_txb_entropy_context(qc, w.tx_size, tt, *eob as usize)
                    };
                    c::ref_set_entropy_contexts(
                        &mut self.above_e[0][a0..],
                        &mut self.left_e[0][l0..],
                        0,
                        bsize,
                        w.tx_size,
                        i32::from(cul),
                        blk_col,
                        blk_row,
                    );
                    k += 1;
                }
            }
            if !self.monochrome && chroma_ref {
                let plane_bsize = aom_entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
                let (pmw, pmh) = (MI_W[plane_bsize], MI_H[plane_bsize]);
                let (ptxwu, ptxhu) = (TX_W[uv_tx] >> 2, TX_H[uv_tx] >> 2);
                let au = (mi_col >> ss_x) as usize;
                let lu = ((mi_row & 31) >> ss_y) as usize;
                let uv_tt = c::ref_get_tx_type_uv_intra(w.uv_mode, false, uv_tx, self.reduced);
                for (plane, txbs) in [(1usize, u_txbs.as_ref()), (2usize, v_txbs.as_ref())] {
                    let txbs = txbs.unwrap();
                    let mut k = 0usize;
                    for blk_row in (0..pmh).step_by(ptxhu) {
                        for blk_col in (0..pmw).step_by(ptxwu) {
                            let (_, eob, _, qc, _) = &txbs[k];
                            let cul = if *eob == 0 {
                                0u8
                            } else {
                                c::ref_txb_entropy_context(qc, uv_tx, uv_tt, *eob as usize)
                            };
                            c::ref_set_entropy_contexts(
                                &mut self.above_e[plane][au..],
                                &mut self.left_e[plane][lu..],
                                plane,
                                plane_bsize,
                                uv_tx,
                                i32::from(cul),
                                blk_col,
                                blk_row,
                            );
                            k += 1;
                        }
                    }
                }
            }
        }

        // set_txfm_ctxs (REAL) at the block origin.
        c::ref_set_txfm_ctxs(
            w.tx_size,
            mbw,
            mbh,
            false,
            &mut self.above_t[a0..],
            &mut self.left_t[l0..],
        );

        out.push((
            mi_row, mi_col, bsize, chroma_ref, store_y, y_txbs, u_txbs, v_txbs,
        ));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_sb(
        &mut self,
        recon_y: &mut [u16],
        recon_u: &mut [u16],
        recon_v: &mut [u16],
        cfl: &mut c::RefCflState,
        tree: &mut SbTree,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        out: &mut Vec<CLeafOut>,
    ) {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return;
        }
        let hbs = (MI_W[bsize] / 2) as i32;
        let (partition, subsize) = match tree {
            SbTree::Leaf(_) => (0i32, bsize),
            SbTree::Split(_) => (3i32, c_split_subsize(bsize)),
            // REAL get_partition_subsize for the rect subsizes.
            SbTree::Horz(_) => (1i32, c::ref_get_partition_subsize(bsize as i32, 1) as usize),
            SbTree::Vert(_) => (2i32, c::ref_get_partition_subsize(bsize as i32, 2) as usize),
        };
        match tree {
            SbTree::Leaf(w) => {
                self.encode_b(recon_y, recon_u, recon_v, cfl, w, mi_row, mi_col, 0, out);
            }
            SbTree::Split(kids) => {
                for (idx, child) in kids.iter_mut().enumerate() {
                    let y = mi_row + ((idx as i32) >> 1) * hbs;
                    let x = mi_col + ((idx as i32) & 1) * hbs;
                    self.encode_sb(recon_y, recon_u, recon_v, cfl, child, y, x, subsize, out);
                }
            }
            SbTree::Horz(subs) => {
                // encode_sb PARTITION_HORZ (partition_search.c:1637-1644).
                let [s0, s1] = &mut **subs;
                self.encode_b(recon_y, recon_u, recon_v, cfl, s0, mi_row, mi_col, 1, out);
                if mi_row + hbs < self.mi_rows {
                    self.encode_b(
                        recon_y,
                        recon_u,
                        recon_v,
                        cfl,
                        s1,
                        mi_row + hbs,
                        mi_col,
                        1,
                        out,
                    );
                }
            }
            SbTree::Vert(subs) => {
                // encode_sb PARTITION_VERT (partition_search.c:1629-1636).
                let [s0, s1] = &mut **subs;
                self.encode_b(recon_y, recon_u, recon_v, cfl, s0, mi_row, mi_col, 2, out);
                if mi_col + hbs < self.mi_cols {
                    self.encode_b(
                        recon_y,
                        recon_u,
                        recon_v,
                        cfl,
                        s1,
                        mi_row,
                        mi_col + hbs,
                        2,
                        out,
                    );
                }
            }
        }
        // The REAL update_ext_partition_context (64-entry above window).
        let mut above64 = [0i8; 64];
        above64.copy_from_slice(&self.above_p);
        let (ao, lo) = c::ref_update_ext_partition_context(
            mi_row,
            mi_col,
            subsize as i32,
            bsize as i32,
            partition,
            &above64,
            &self.left_p,
        );
        self.above_p = ao;
        self.left_p = lo;
    }
}
