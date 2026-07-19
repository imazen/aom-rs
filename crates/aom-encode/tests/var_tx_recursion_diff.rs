//! Differential harness for the inter var-tx RECURSION
//! (`var_tx::pick_recursive_tx_size_type_yrd` -> select_tx_size_and_type ->
//! select_tx_block / try_tx_block_no_split / try_tx_block_split) vs an
//! INDEPENDENT transcription of the same quadtree from tx_search.c, using the
//! already-C-locked inter leaf (`search_tx_type_inter`, validated byte-for-byte
//! against the real C kernels in var_tx_leaf_diff.rs) as the per-txb evaluator.
//!
//! The recursion glue (no-split vs recursive-split RD comparison, the pick-skip
//! txfm decision, the txfm_partition split-flag cost, the av1_set_txb_context +
//! txfm_partition_update context threading with no-split-overwrite backtracking,
//! the select_tx_size_and_type unit raster + skip decision) is transcribed here
//! straight from tx_search.c (2406/2454/2601/3433) — a fresh derivation, not a
//! copy of var_tx.rs — so a divergence isolates a recursion-glue transcription
//! bug. Prunes are off on BOTH sides (model_based / ml_tx_split / prune_tx_2D
//! unported); the active speed-0 early-terms (adaptive_txb_search_level=1,
//! txb_split_cap=1) are transcribed on both sides.

use aom_encode::BlockContext;
use aom_encode::rd::rdcost;
use aom_encode::var_tx::{InterLeafInputs, VarTxEnv, pick_recursive_tx_size_type_yrd, search_tx_type_inter};
use aom_entropy::partition::{txfm_partition_context, txfm_partition_update};
use aom_quant::{Dequants, PlaneQuantRows, Quants, av1_build_quantizer, set_q_index};
use aom_txb::{CoeffCostSet, TxTypeCosts, fill_tx_type_costs, get_txb_ctx};

const TXS_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TXS_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
const TX_WU: [usize; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
const TX_HU: [usize; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];
const SUB_TX: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
const MAX_RECT: [usize; 22] = [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];
const BLK_W: [usize; 22] = [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
const BLK_H: [usize; 22] = [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
    fn cost(&mut self) -> i32 {
        self.range(0, 20 << 9)
    }
}
fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}
fn gen_cdf_row(rng: &mut Rng, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut row = vec![0u16; padded];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, (32000 / nsymbs as i32).max(2)) as u32;
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}
fn gen_cdfs(rng: &mut Rng, count: usize, nsymbs: usize, padded: usize) -> Vec<u16> {
    let mut v = Vec::with_capacity(count * padded);
    for _ in 0..count {
        v.extend_from_slice(&gen_cdf_row(rng, nsymbs, padded));
    }
    v
}

/// Facade-side context bundle (immutable per block).
struct Ctx<'a> {
    bsize: usize,
    max_bw: usize,
    max_bh: usize,
    residual: &'a [i16],
    res_stride: usize,
    pred: &'a [u16],
    pred_stride: usize,
    src: &'a [u16],
    src_off: usize,
    src_stride: usize,
    bd: u8,
    rows: &'a PlaneQuantRows<'a>,
    rdmult: i32,
    coeff_costs: &'a CoeffCostSet,
    tx_type_costs: &'a TxTypeCosts,
    txfm_partition_cost: [[i32; 2]; 21],
    reduced: bool,
}

fn extract_i16(s: &[i16], stride: usize, br: usize, bc: usize, w: usize, h: usize) -> Vec<i16> {
    let mut o = vec![0i16; w * h];
    let base = 4 * br * stride + 4 * bc;
    for r in 0..h {
        o[r * w..r * w + w].copy_from_slice(&s[base + r * stride..base + r * stride + w]);
    }
    o
}
fn extract_u16(s: &[u16], stride: usize, br: usize, bc: usize, w: usize, h: usize) -> Vec<u16> {
    let mut o = vec![0u16; w * h];
    let base = 4 * br * stride + 4 * bc;
    for r in 0..h {
        o[r * w..r * w + w].copy_from_slice(&s[base + r * stride..base + r * stride + w]);
    }
    o
}

#[derive(Clone, Copy)]
struct Rd {
    rate: i32,
    dist: i64,
    sse: i64,
    skip: bool,
    zero_rate: i32,
}
impl Rd {
    fn init() -> Self {
        Rd { rate: 0, dist: 0, sse: 0, skip: true, zero_rate: 0 }
    }
    fn invalid() -> Self {
        Rd { rate: i32::MAX, dist: i64::MAX, sse: i64::MAX, skip: false, zero_rate: 0 }
    }
    fn merge(&mut self, s: &Rd) {
        if self.rate == i32::MAX || s.rate == i32::MAX {
            *self = Rd::invalid();
            return;
        }
        self.rate = ((self.rate as i64) + (s.rate as i64)).min(i32::MAX as i64) as i32;
        if self.zero_rate == 0 {
            self.zero_rate = s.zero_rate;
        }
        self.dist += s.dist;
        if self.sse < i64::MAX && s.sse < i64::MAX {
            self.sse += s.sse;
        }
        self.skip &= s.skip;
    }
}
fn rd_of(rm: i32, rate: i32, dist: i64) -> i64 {
    if rate == i32::MAX || dist == i64::MAX {
        i64::MAX
    } else {
        rdcost(rm, rate, dist)
    }
}

/// (rd_stats, no_split rd, txb_ctx, eob, skip). try_tx_block_no_split.
#[allow(clippy::too_many_arguments)]
fn c_try_no_split(
    c: &Ctx,
    br: usize,
    bc: usize,
    tx_size: usize,
    depth: i32,
    ta: &[i8],
    tl: &[i8],
    part_ctx: usize,
    ref_best_rd: i64,
) -> (Rd, i64, u8, u16) {
    let (w, h) = (TXS_W[tx_size], TXS_H[tx_size]);
    let vis_c = (c.max_bw.saturating_sub(bc)).min(TX_WU[tx_size]) * 4;
    let vis_r = (c.max_bh.saturating_sub(br)).min(TX_HU[tx_size]) * 4;
    let residual = extract_i16(c.residual, c.res_stride, br, bc, w, h);
    let pred = extract_u16(c.pred, c.pred_stride, br, bc, w, h);
    let src_off = c.src_off + 4 * br * c.src_stride + 4 * bc;
    let bctx = BlockContext { above: &ta[bc..], left: &tl[br..], plane: 0, plane_bsize: c.bsize };
    let (skip_ctx, _) = get_txb_ctx(c.bsize, tx_size, 0, &ta[bc..], &tl[br..]);
    let zero_blk_rate = c.coeff_costs.tables(tx_size).txb_skip[skip_ctx as usize * 2 + 1];

    let inp = InterLeafInputs {
        residual: &residual,
        pred: &pred,
        src: c.src,
        src_off,
        src_stride: c.src_stride,
        tx_size,
        lossless: false,
        reduced_tx_set_used: c.reduced,
        enable_flip_idtx: true,
        use_inter_dct_only: false,
        bd: c.bd,
        rows: c.rows,
        bctx: &bctx,
        rdmult: c.rdmult,
        coeff_costs: &c.coeff_costs.tables(tx_size),
        tx_type_costs: c.tx_type_costs,
        visible_cols: vis_c.min(w),
        visible_rows: vis_r.min(h),
        qm_level: None,
    };
    let Some(leaf) = search_tx_type_inter(&inp, 0, false, 3200, 1, ref_best_rd) else {
        return (Rd::invalid(), i64::MAX, 0, 0);
    };
    let mut rd = Rd::init();
    rd.zero_rate = zero_blk_rate;
    rd.merge(&Rd { rate: leaf.rate, dist: leaf.dist, sse: leaf.sse, skip: leaf.skip_txfm, zero_rate: 0 });

    let mut eob = leaf.best_eob;
    let pick_skip = rd.skip
        || rd_of(c.rdmult, rd.rate, rd.dist) >= rd_of(c.rdmult, zero_blk_rate, rd.sse);
    if pick_skip {
        rd.rate = zero_blk_rate;
        rd.dist = rd.sse;
        eob = 0;
    }
    rd.skip = pick_skip;
    if tx_size > 0 && depth < 2 {
        rd.rate =
            ((rd.rate as i64) + c.txfm_partition_cost[part_ctx][0] as i64).min(i32::MAX as i64) as i32;
    }
    let txb_ctx = if pick_skip { 0 } else { leaf.best_txb_ctx };
    let no_rd = rd_of(c.rdmult, rd.rate, rd.dist);
    (rd, no_rd, txb_ctx, eob)
}

/// select_tx_block — returns (rd_stats, is_valid). Mutates the contexts to the
/// winner state. `inter_tx_size` accumulates the chosen tx-unit map.
#[allow(clippy::too_many_arguments)]
fn c_select_tx_block(
    c: &Ctx,
    br: usize,
    bc: usize,
    tx_size: usize,
    depth: i32,
    ta: &mut [i8],
    tl: &mut [i8],
    txa: &mut [u8],
    txl: &mut [u8],
    prev_level_rd: i64,
    ref_best_rd: i64,
) -> (Rd, bool) {
    if ref_best_rd < 0 {
        return (Rd::init(), false);
    }
    let part_ctx = txfm_partition_context(txa[bc], txl[br], c.bsize, tx_size);
    let mut try_split = tx_size > 0 && depth < 2;

    let (no_rd_stats, no_rd, no_ctx, no_eob) =
        c_try_no_split(c, br, bc, tx_size, depth, ta, tl, part_ctx, ref_best_rd);
    // adaptive_txb_search_level = 1.
    if no_rd != i64::MAX && (no_rd - (no_rd >> 2)) > ref_best_rd {
        return (Rd::init(), false);
    }
    if no_rd != i64::MAX && (no_rd - (no_rd >> 3)) > prev_level_rd {
        try_split = false;
    }
    // txb_split_cap = 1.
    if no_eob == 0 {
        try_split = false;
    }

    let mut split_rd = Rd::invalid();
    let mut split_rdcost = i64::MAX;
    // Snapshot contexts so a no-split win overwrites cleanly (the split recursion
    // mutates them in place, exactly like C).
    if try_split {
        let sub = SUB_TX[tx_size];
        let (sw, sh) = (TX_WU[sub], TX_HU[sub]);
        let (tw, th) = (TX_WU[tx_size], TX_HU[tx_size]);
        let nblks = ((th / sh) * (tw / sw)) as i64;
        let mut s = Rd::init();
        s.rate = c.txfm_partition_cost[part_ctx][1];
        split_rdcost = rd_of(c.rdmult, s.rate, s.dist);
        let inner_ref = no_rd.min(ref_best_rd);
        let mut ok = true;
        let mut r = 0;
        'outer: while r < th {
            let or = br + r;
            if or >= c.max_bh {
                break;
            }
            let mut col = 0;
            while col < tw {
                let oc = bc + col;
                if oc >= c.max_bw {
                    col += sw;
                    continue;
                }
                let child_prev = if nblks > 0 && no_rd != i64::MAX { no_rd / nblks } else { i64::MAX };
                let child_ref = if inner_ref == i64::MAX { i64::MAX } else { inner_ref - split_rdcost };
                let (cs, cv) =
                    c_select_tx_block(c, or, oc, sub, depth + 1, ta, tl, txa, txl, child_prev, child_ref);
                if !cv {
                    ok = false;
                    break 'outer;
                }
                s.merge(&cs);
                split_rdcost = rd_of(c.rdmult, s.rate, s.dist);
                if split_rdcost > inner_ref {
                    ok = false;
                    break 'outer;
                }
                col += sw;
            }
            r += sh;
        }
        if ok {
            split_rd = s;
        } else {
            split_rd = Rd::invalid();
            split_rdcost = i64::MAX;
        }
    }

    if no_rd < split_rdcost {
        let (tw, th) = (TX_WU[tx_size], TX_HU[tx_size]);
        for a in ta[bc..bc + tw].iter_mut() {
            *a = no_ctx as i8;
        }
        for l in tl[br..br + th].iter_mut() {
            *l = no_ctx as i8;
        }
        txfm_partition_update(&mut txa[bc..], &mut txl[br..], tx_size, tx_size);
        (no_rd_stats, true)
    } else if split_rd.rate == i32::MAX {
        (split_rd, false)
    } else {
        (split_rd, true)
    }
}

/// select_tx_size_and_type — returns (final_rd, rate, dist, sse, skip) or None.
fn c_select_size_type(
    c: &Ctx,
    above: &[i8],
    left: &[i8],
    txa0: &[u8],
    txl0: &[u8],
    skip_cost: [i32; 2],
    ref_best_rd: i64,
) -> Option<(i32, i64, i64, bool)> {
    let max_tx = MAX_RECT[c.bsize];
    let (bw, bh) = (TX_WU[max_tx], TX_HU[max_tx]);
    let mut ta = above[..c.max_bw].to_vec();
    let mut tl = left[..c.max_bh].to_vec();
    let mut txa = txa0[..c.max_bw].to_vec();
    let mut txl = txl0[..c.max_bh].to_vec();
    let mut skip_rd = rdcost(c.rdmult, skip_cost[1], 0);
    let mut no_skip_rd = rdcost(c.rdmult, skip_cost[0], 0);
    let mut rd = Rd::init();
    let mut idy = 0;
    while idy < c.max_bh {
        let mut idx = 0;
        while idx < c.max_bw {
            let best = if ref_best_rd == i64::MAX { i64::MAX } else { ref_best_rd - skip_rd.min(no_skip_rd) };
            let (pn, valid) =
                c_select_tx_block(c, idy, idx, max_tx, 0, &mut ta, &mut tl, &mut txa, &mut txl, i64::MAX, best);
            if !valid || pn.rate == i32::MAX {
                return None;
            }
            rd.merge(&pn);
            skip_rd = rdcost(c.rdmult, skip_cost[1], rd.sse);
            no_skip_rd = rdcost(
                c.rdmult,
                ((rd.rate as i64) + skip_cost[0] as i64).min(i32::MAX as i64) as i32,
                rd.dist,
            );
            idx += bw;
        }
        idy += bh;
    }
    if rd.rate == i32::MAX {
        return None;
    }
    let skip = skip_rd <= no_skip_rd;
    Some((rd.rate, rd.dist, rd.sse, skip))
}

#[test]
fn pick_recursive_tx_size_type_matches_c_recursion() {
    aom_sys_ref::ref_init();
    let mut rng = Rng(0x2026_0713_c0ffee11);
    let mut split_seen = 0usize;
    let mut nosplit_seen = 0usize;
    let mut skip_seen = 0usize;
    let mut coded_seen = 0usize;

    // Block sizes exercising the quadtree (>= 8x8 so the root can split).
    let bsizes = [3usize, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21];
    for &bsize in &bsizes {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let max_bw = bw / 4;
        let max_bh = bh / 4;
        for iter in 0..20 {
            let bd = 8u8;
            // Low qindex so residuals are robustly CODED (not all-skip).
            let qindex = [8, 12, 16, 24, 40][iter % 5] as usize;
            let stride = bw + 8;
            // Flat source (128); a SMOOTH low-frequency residual (compressible ->
            // coding wins, eob>0, no skip) with a coarse period so per-region
            // fine tx competes. The split-flag cost knob below then drives the
            // split-vs-no-split decision so the split recursion + context
            // threading is actually exercised.
            let src: Vec<u16> = vec![128u16; stride * (bh + 8)];
            let src_off = 2 * stride + 3;
            // PIECEWISE-CONSTANT 8x8 tiles, each a distinct DC value. A large tx
            // must code the tile discontinuities with many high-frequency coeffs
            // (expensive); splitting down to 8x8 isolates each tile into a single
            // cheap DC coeff -> split wins strongly (and is coded, eob>0, not
            // skipped). Combined with the split-flag cost knob this drives the
            // full split recursion + depth-2 context threading.
            let tiles = [40i32, -32, 56, -48, 28, -60, 44, -20];
            let res_at = |r: usize, cc: usize| -> i32 {
                tiles[(r / 8 + 3 * (cc / 8) + iter) % 8]
            };
            let pred: Vec<u16> = (0..stride * (bh + 8))
                .map(|i| {
                    let (r, cc) = (i / stride, i % stride);
                    (128 - res_at(r, cc)).clamp(0, 255) as u16
                })
                .collect();
            // Whole-block residual (stride = bw), src - pred at the block origin.
            let residual: Vec<i16> = (0..bw * bh)
                .map(|i| {
                    let (r, cc) = (i / bw, i % bw);
                    (i64::from(src[src_off + r * stride + cc]) - i64::from(pred[src_off + r * stride + cc])) as i16
                })
                .collect();

            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows = set_q_index(&quants, &deq, qindex, 0);

            // Random (but consistent, both sides share it) per-txs_ctx coeff
            // cost set. The differential only needs the SAME tables on both
            // sides; realism is irrelevant to the recursion glue.
            let coeff_set = random_coeff_set(&mut rng);

            const NUM_EXT_TX_SET: [usize; 6] = [1, 2, 5, 7, 12, 16];
            const IDX_TO_TYPE: [[usize; 4]; 2] = [[0, 3, 2, 0], [0, 5, 4, 1]];
            let mut intra_cdf = Vec::new();
            for s in 0..3 {
                let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[0][s]].max(2);
                intra_cdf.extend_from_slice(&gen_cdfs(&mut rng, 4 * 13, ns, 17));
            }
            let mut inter_cdf = Vec::new();
            for s in 0..4 {
                let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[1][s]].max(2);
                inter_cdf.extend_from_slice(&gen_cdfs(&mut rng, 4, ns, 17));
            }
            let mut ttc = TxTypeCosts::zeroed();
            fill_tx_type_costs(&mut ttc, &intra_cdf, &inter_cdf);

            // Tx-partition split-flag cost knob: on ~half the iters make the
            // no-split flag ([0]) EXPENSIVE and the split flag ([1]) cheap, which
            // forces the search down the split recursion to depth 2 (exercising
            // try_tx_block_split + the child context threading); on the others
            // the reverse (favouring no-split). Both sides share the table, so
            // the differential still tests the exact glue.
            let split_favor = iter % 2 == 0;
            let mut tpc = [[0i32; 2]; 21];
            for row in tpc.iter_mut() {
                if split_favor {
                    row[0] = rng.range(3000, 6000); // no-split expensive
                    row[1] = rng.range(1, 40); // split cheap
                } else {
                    row[0] = rng.range(1, 40);
                    row[1] = rng.range(3000, 6000);
                }
            }
            let skip_cost = [rng.range(1, 400), rng.range(1, 400)];
            // Low rdmult (realistic for the low-qindex band above) so CODING wins
            // over skip at the root (no_eob > 0) — otherwise txb_split_cap
            // disables split and the split recursion is never exercised.
            let rdmult = rng.range(20, 2000);
            let reduced = iter % 5 == 2;

            let above = vec![0i8; 64];
            let left = vec![0i8; 64];
            let txa = vec![0u8; 64];
            let txl = vec![0u8; 64];

            // ---- Port side ----
            let env = VarTxEnv {
                bsize,
                max_blocks_wide: max_bw,
                max_blocks_high: max_bh,
                residual: &residual,
                residual_stride: bw,
                pred: &pred,
                pred_stride: stride,
                src: &src,
                src_off,
                src_stride: stride,
                lossless: false,
                reduced_tx_set_used: reduced,
                enable_flip_idtx: true,
                use_inter_dct_only: false,
                bd,
                rows: &rows,
                rdmult,
                coeff_costs: &coeff_set,
                tx_type_costs: &ttc,
                qm_level: None,
                txfm_partition_cost: &tpc,
                skip_txfm_cost: skip_cost,
                above_ctx: &above,
                left_ctx: &left,
                tx_above: &txa,
                tx_left: &txl,
                sharpness: 0,
                iq_tuning: false,
                coeff_opt_dist_threshold: 3200,
                adaptive_txb_search_level: 1,
                txb_split_cap: true,
                ml_tx_split_thresh: -1, // NN off (prunes-off recursion differential)
                init_depth: 0,
            };
            let port = pick_recursive_tx_size_type_yrd(&env, i64::MAX);

            // ---- Facade side ----
            let cctx = Ctx {
                bsize,
                max_bw,
                max_bh,
                residual: &residual,
                res_stride: bw,
                pred: &pred,
                pred_stride: stride,
                src: &src,
                src_off,
                src_stride: stride,
                bd,
                rows: &rows,
                rdmult,
                coeff_costs: &coeff_set,
                tx_type_costs: &ttc,
                txfm_partition_cost: tpc,
                reduced,
            };
            let facade = c_select_size_type(&cctx, &above, &left, &txa, &txl, skip_cost, i64::MAX);

            let m = format!("bsize={bsize} iter={iter} q={qindex}");
            assert!(port.valid, "port invalid {m}");
            let (fr, fd, fs, fskip) = facade.expect(&format!("facade invalid {m}"));
            assert_eq!(port.rate, fr, "rate {m}");
            assert_eq!(port.dist, fd, "dist {m}");
            assert_eq!(port.sse, fs, "sse {m}");
            assert_eq!(port.skip_txfm, fskip, "skip {m}");

            // Coverage.
            let root = MAX_RECT[bsize];
            let all_root = port.leaves.iter().all(|l| l.tx_size == root) && port.leaves.len() == 1;
            if all_root {
                nosplit_seen += 1;
            } else {
                split_seen += 1;
            }
            for l in &port.leaves {
                if l.skip_txfm {
                    skip_seen += 1;
                } else {
                    coded_seen += 1;
                }
            }
        }
    }
    // Non-vacuity: the split recursion, the no-split path, and coded (eob>0)
    // leaves are all exercised (the differential asserts byte-equality on each).
    assert!(split_seen > 4, "split trees: {split_seen}");
    assert!(nosplit_seen > 4, "no-split trees: {nosplit_seen}");
    assert!(coded_seen > 12, "coded leaves: {coded_seen}");
    let _ = skip_seen;
}

/// A random `CoeffCostSet` (5 distinct per-txs_ctx tables + 7 eob tables).
/// Both port + facade share the same instance, so the recursion differential is
/// unaffected by realism — only self-consistency matters.
fn random_coeff_set(rng: &mut Rng) -> CoeffCostSet {
    let mk = |rng: &mut Rng| aom_txb::LvMapCoeffCost {
        txb_skip: tbl(rng, 13 * 2),
        base_eob: tbl(rng, 4 * 3),
        base: tbl(rng, 42 * 8),
        eob_extra: tbl(rng, 9 * 2),
        dc_sign: tbl(rng, 3 * 2),
        lps: tbl(rng, 21 * 26),
    };
    // Array literal evaluates left-to-right; each mk() call releases the borrow
    // before the next.
    let by_txs_ctx = [mk(rng), mk(rng), mk(rng), mk(rng), mk(rng)];
    let mut eob_by_multi_size = [[0i32; 22]; 7];
    for row in eob_by_multi_size.iter_mut() {
        for x in row.iter_mut() {
            *x = rng.cost();
        }
    }
    CoeffCostSet { by_txs_ctx, eob_by_multi_size }
}
