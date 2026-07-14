//! Differential: `rd_pick_intra_sby_mode_y` (the av1_rd_pick_intra_sby_mode
//! 61-candidate luma mode loop, speed-0 all-intra scope) vs an independent
//! transcription of the C loop (intra_mode_search.c:1545-1661) driving REAL
//! reference pieces end-to-end:
//! ref_set_y_mode_and_delta_angle (REAL exported) -> the gate chain
//! (transcribed) -> c_intra_model_rd (REAL predict/subtract/hadamard/satd
//! chain, prediction writes into the C recon plane) ->
//! ref_get_model_rd_index_for_pruning + ref_prune_intra_y_mode (compiled-C
//! transcriptions) -> c_pick_uniform_tx_size_type_yrd (the REAL-piece depth
//! sweep, threaded with the RUNNING best_rd) -> ref_tx_size_cost subtraction
//! -> ref_intra_mode_info_cost_y -> ref_rdcost ->
//! [ALLINTRA: ref_intra_rd_variance_factor over the evolving C recon +
//! C-side var cache] -> intra_modes_rd_cost bookkeeping -> strict `<` best
//! tracking (first winner kept on ties).
//!
//! Asserted per case: the winning (mode, angle_delta, tx_size, per-txb
//! winners, rate, rate_tokenonly, dist, skippable, best_rd) — or both sides
//! agreeing no candidate beat best_rd_in — plus the FULL 13x9
//! intra_modes_rd_cost table, the final recon planes (the cross-candidate
//! prediction/reconstruction state), and the variance-factor caches.

use aom_encode::hog::prune_intra_mode_with_hog_y;
use aom_encode::intra_rd::{
    rd_pick_intra_sby_mode_y, Block4x4VarInfo, IntraSbyGates, IntraSbySearchCfg,
    INTRA_MODES, MAX_ANGLE_DELTA, TOP_INTRA_MODEL_COUNT,
};
use aom_encode::mode_costs::{
    fill_intra_mode_costs, fill_tx_size_costs, IntraModeCosts, TxSizeCosts, BLOCK_SIZES_ALL,
    BLOCK_SIZE_GROUPS, DIRECTIONAL_MODES, FILTER_INTRA_MODES, KF_MODE_CONTEXTS,
    PALETTE_BSIZE_CTXS, PALETTE_Y_MODE_CONTEXTS, UV_INTRA_MODES,
};
use aom_encode::tx_search::{TxTypeSearchPolicy, TxfmYrdEnv};
use aom_quant::{av1_build_quantizer, set_q_index, Dequants, Quants};
use aom_sys_ref as c;
use aom_txb::{fill_tx_type_costs, CoeffCostTables, TxTypeCosts};

mod common;
use common::*;

/// The C loop's static gate chain (intra_mode_search.c:1555-1594) at the
/// aomenc-default tool flags — an independent transcription (the Rust side
/// gates live in IntraSbyGates::visits). Speed-0: every intra_mode_cfg flag
/// on, disable_smooth_intra off, intra_y_mode_mask all-ones,
/// use_mb_mode_cache off.
fn c_gate_visits(mode: usize, luma_delta_angle: i32, bsize: usize, skip_mask: &[bool; 13]) -> bool {
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
struct CLoopOut {
    best: Option<(usize, i32, usize, Vec<(usize, u16, u8)>, i32, i32, i64, i64)>,
    rd_table: [[i64; 9]; 13],
    /// Candidates whose ALLINTRA factor was != 1.0 (coverage signal).
    factor_fired: usize,
}

#[test]
fn rd_pick_intra_sby_mode_matches_c_loop() {
    c::ref_init();
    let mut rng = Rng(0x10de_100b_2026_0714);
    const STRIDE: usize = 256;
    // (bsize, sb_size, mi_row, mi_col): small blocks keep the 61-candidate
    // full-RD loop tractable; one 16x16 exercises the 8x8-init-depth sweep +
    // 4-txb walks, the rects exercise init_depth 0.
    let cases: [(usize, usize, i32, i32); 5] = [
        (0, 12, 8, 8),   // 4x4 (no angle deltas, no tx-size signaling)
        (3, 12, 8, 8),   // 8x8
        (4, 12, 9, 10),  // 8x16 rect
        (6, 12, 20, 24), // 16x16
        (9, 12, 8, 8),   // 32x32 (TX_32X32 model, 3-size depth sweeps)
    ];
    let mut beat_cases = 0usize;
    let mut nobeat_cases = 0usize;
    let mut prune_survivor_variety = std::collections::HashSet::new();
    let mut factor_fired_total = 0usize;
    let mut hog_masked_cases = 0usize;

    for (ci, &(bsize, sb_size, mi_row, mi_col)) in cases.iter().enumerate() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        for iter in 0..8 {
            let bd: u8 = match iter % 3 {
                0 => 8,
                1 => 10,
                _ => 12,
            };
            let amp: i32 = match iter % 4 {
                0 => {
                    if bd > 8 {
                        2048
                    } else {
                        160
                    }
                }
                1 => 24,
                2 => 96,
                _ => 64,
            };
            // High-q + detailed content flattens the reconstruction, driving
            // the ALLINTRA variance factor above 1.0.
            let qindex = [16, 64, 200, 255][iter % 4] as usize;
            let reduced = iter % 4 == 3;
            let allintra = iter % 3 != 1; // ALLINTRA on for most cases
            let source_variance =
                if iter % 3 == 0 { rng.range(0, 256) as u32 } else { 256 + rng.range(0, 4096) as u32 };
            // best_rd_in: mostly MAX; some tight budgets force no-beat + the
            // running-best-rd early exits inside pick_uniform.
            let best_rd_in = if iter == 7 { 1 << 10 } else { i64::MAX };
            let above_mode = if rng.next().is_multiple_of(4) { None } else { Some((rng.next() % 13) as i32) };
            let left_mode = if rng.next().is_multiple_of(4) { None } else { Some((rng.next() % 13) as i32) };

            let ref_off = 32 * STRIDE + 32;
            let src_off = ref_off;
            // High-q iters use a FLAT-ish recon plane (smooth prediction
            // edges) under a heavily-detailed source: the reconstruction
            // flattens (residual quantized away), pushing the ALLINTRA
            // variance factor's src >= recon arm past var_diff > 0.5.
            let flat_recon = iter % 4 >= 2;
            let recon0: Vec<u16> = if flat_recon {
                let base = (rng.next() % ((1u64 << bd) - 16)) as i64;
                (0..STRIDE * 96)
                    .map(|_| (base + i64::from(rng.range(0, 8))) as u16)
                    .collect()
            } else {
                (0..STRIDE * 96).map(|_| (rng.next() % (1u64 << bd)) as u16).collect()
            };
            let src_amp = if flat_recon { (1 << bd) / 2 } else { amp };
            let mut src = recon0.clone();
            for r in 0..bh {
                for cx in 0..bw {
                    let idx = src_off + r * STRIDE + cx;
                    let v = i64::from(recon0[idx]) + i64::from(rng.range(-src_amp, src_amp + 1));
                    src[idx] = v.clamp(0, (1 << bd) - 1) as u16;
                }
            }

            // HOG skip mask: the REAL speed-0 prune (thresh -1.2) computed
            // from the source pixels by BOTH sides (mask equality asserted),
            // exactly as av1_rd_pick_intra_sby_mode fills it before the loop
            // (intra_mode_search.c:1501-1510) — interior blocks, so
            // mb_to_*_edge are large positive.
            let mut skip_mask = [false; INTRA_MODES];
            if iter % 4 == 2 {
                prune_intra_mode_with_hog_y(
                    &src, src_off, STRIDE, bsize, 1 << 12, 1 << 12, -1.2, &mut skip_mask,
                );
                let mask_c = c::ref_prune_intra_mode_with_hog_y(
                    &src, src_off, STRIDE, bsize, 1 << 12, 1 << 12, bd, -1.2,
                );
                assert_eq!(skip_mask, mask_c, "hog mask ci={ci} iter={iter} bd={bd}");
                if skip_mask.iter().any(|&b| b) {
                    hog_masked_cases += 1;
                }
            }

            // Quantizer rows.
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows = set_q_index(&quants, &deq, qindex, 0);
            let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
            let plane_rows_c = &rows_c[0..7 * 8];

            // Coefficient cost tables.
            let txb_skip = tbl(&mut rng, 13 * 2);
            let base_eob = tbl(&mut rng, 4 * 3);
            let base = tbl(&mut rng, 42 * 8);
            let eob_extra = tbl(&mut rng, 9 * 2);
            let dc_sign = tbl(&mut rng, 3 * 2);
            let lps = tbl(&mut rng, 21 * 26);
            let eob_tbl = tbl(&mut rng, 2 * 11);
            let coeff_costs = CoeffCostTables {
                txb_skip: &txb_skip,
                base_eob: &base_eob,
                base: &base,
                eob_extra: &eob_extra,
                dc_sign: &dc_sign,
                lps: &lps,
                eob: &eob_tbl,
            };
            const NUM_EXT_TX_SET: [usize; 6] = [1, 2, 5, 7, 12, 16];
            const IDX_TO_TYPE: [[usize; 4]; 2] = [[0, 3, 2, 0], [0, 5, 4, 1]];
            let mut ttc_intra_cdf = Vec::new();
            for s in 0..3 {
                let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[0][s]].max(2);
                ttc_intra_cdf.extend_from_slice(&gen_cdfs(&mut rng, 4 * 13, ns, 17));
            }
            let mut ttc_inter_cdf = Vec::new();
            for s in 0..4 {
                let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[1][s]].max(2);
                ttc_inter_cdf.extend_from_slice(&gen_cdfs(&mut rng, 4, ns, 17));
            }
            let (c_ttc_intra, c_ttc_inter) =
                c::ref_fill_tx_type_costs(&ttc_intra_cdf, &ttc_inter_cdf);
            let mut tx_type_costs = TxTypeCosts::zeroed();
            fill_tx_type_costs(&mut tx_type_costs, &ttc_intra_cdf, &ttc_inter_cdf);

            // tx-size + skip costs + contexts.
            let mut ts_cdf = Vec::new();
            for cat in 0..4 {
                let ns = if cat == 0 { 2 } else { 3 };
                for _ in 0..3 {
                    ts_cdf.extend_from_slice(&cdf_row4(&mut rng, ns));
                }
            }
            let mut tx_size_costs = TxSizeCosts::zeroed();
            fill_tx_size_costs(&mut tx_size_costs, &ts_cdf);
            let ts_flat: Vec<i32> = tx_size_costs.0.iter().flatten().flatten().copied().collect();
            let skip_costs =
                [[rng.cost(), rng.cost()], [rng.cost(), rng.cost()], [rng.cost(), rng.cost()]];
            let skip_ctx = (rng.next() % 3) as usize;
            let tx_size_ctx = (rng.next() % 3) as usize;
            let above_ctx: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let left_ctx: Vec<i8> =
                (0..32).map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8).collect();
            let rdmult = rng.range(1, 1 << 22);

            // Mode-info cost tables (random CDFs through the dual fill).
            let cdf_set = gen_all_cdfs(&mut rng);
            let (mode_costs, c_mode_costs) = fill_both(&cdf_set, true);
            let pol = TxTypeSearchPolicy::speed0_allintra();
            let gates = IntraSbyGates {
                directional_mode_skip_mask: skip_mask,
                ..IntraSbyGates::speed0([false; INTRA_MODES])
            };

            // ---- Rust side ----
            let mut env = TxfmYrdEnv {
                sb_size,
                bsize,
                mi_row,
                mi_col,
                up_available: true,
                left_available: true,
                tile_col_end: 1 << 16,
                tile_row_end: 1 << 16,
                partition: 0,
                mi_cols: 512,
                mi_rows: 512,
                ref_off,
                ref_stride: STRIDE,
                src: &src,
                src_off,
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                mode: 0,
                angle_delta: 0,
                use_filter_intra: false,
                filter_intra_mode: 0,
                lossless: false,
                reduced_tx_set_used: reduced,
                bd,
                rows: &rows,
                rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                skip_costs: &skip_costs,
                skip_ctx,
                tx_size_costs: &tx_size_costs,
                tx_size_ctx,
                tx_mode_is_select: true,
                above_ctx: &above_ctx,
                left_ctx: &left_ctx,
            };
            let cfg = IntraSbySearchCfg {
                gates: &gates,
                top_intra_model_count_allowed: TOP_INTRA_MODEL_COUNT as i32,
                adapt_top_model_rd_count_using_neighbors: false,
                above_mode,
                left_mode,
                qindex: qindex as i32,
                mode_costs: &mode_costs,
                try_palette: false,
                palette_bsize_ctx: 0,
                palette_mode_ctx: 0,
                enable_filter_intra: true,
                allow_intrabc: false,
                pol: &pol,
                source_variance,
                enable_tx64: true,
                enable_rect_tx: true,
                allintra,
                speed: 0,
                mb_to_right_edge: 1 << 12,
                mb_to_bottom_edge: 1 << 12,
            };
            let mut recon_rust = recon0.clone();
            let mut var_cache_rust = Block4x4VarInfo::sb_cache(sb_size);
            let got =
                rd_pick_intra_sby_mode_y(&mut env, &mut recon_rust, &cfg, &mut var_cache_rust, best_rd_in);

            // ---- C-side loop ----
            let mut recon_c = recon0.clone();
            let n_mi = if sb_size == 15 { 32 * 32 } else { 16 * 16 };
            let mut cvar = vec![-1i32; n_mi];
            let mut clog = vec![-1.0f64; n_mi];
            let want = c_mode_loop(
                bsize,
                (mi_row, mi_col, ref_off, src_off, STRIDE),
                sb_size,
                &mut recon_c,
                &src,
                reduced,
                bd,
                plane_rows_c,
                [rows.dequant[0], rows.dequant[1]],
                &above_ctx,
                &left_ctx,
                rdmult,
                best_rd_in,
                (&txb_skip, &base_eob, &base, &eob_extra, &dc_sign, &lps, &eob_tbl),
                (&c_ttc_intra, &c_ttc_inter),
                &skip_costs,
                skip_ctx,
                &ts_flat,
                tx_size_ctx,
                source_variance,
                &skip_mask,
                (above_mode, left_mode),
                qindex as i32,
                &c_mode_costs,
                allintra,
                &mut cvar,
                &mut clog,
            );

            factor_fired_total += want.factor_fired;
            let m = format!(
                "ci={ci} bsize={bsize} iter={iter} bd={bd} amp={amp} q={qindex} red={reduced} \
                 allintra={allintra} var={source_variance} best_in={best_rd_in}",
            );
            // The full per-(mode, delta) rd table.
            for mode in 0..INTRA_MODES {
                for d in 0..9usize {
                    assert_eq!(
                        got.intra_modes_rd_cost[mode][d], want.rd_table[mode][d],
                        "rd_table[{mode}][{d}] {m}",
                    );
                }
            }
            assert_eq!(recon_rust, recon_c, "final recon plane {m}");
            for k in 0..n_mi {
                assert_eq!(var_cache_rust[k].var, cvar[k], "var cache k={k} {m}");
                assert_eq!(
                    var_cache_rust[k].log_var.to_bits(),
                    clog[k].to_bits(),
                    "log cache k={k} {m}",
                );
            }
            match (got.best, want.best) {
                (Some(g), Some(cb)) => {
                    assert_eq!(g.mode, cb.0, "mode {m}");
                    assert_eq!(g.angle_delta, cb.1, "angle_delta {m}");
                    assert_eq!(g.tx_size, cb.2, "tx_size {m}");
                    let wins: Vec<(usize, u16, u8)> =
                        g.winners.iter().map(|w| (w.tx_type, w.eob, w.txb_ctx)).collect();
                    assert_eq!(wins, cb.3, "winners {m}");
                    assert_eq!(g.rate, cb.4, "rate {m}");
                    assert_eq!(g.rate_tokenonly, cb.5, "rate_tokenonly {m}");
                    assert_eq!(g.dist, cb.6, "dist {m}");
                    assert!(!g.skippable, "intra non-skip {m}");
                    assert_eq!(g.best_rd, cb.7, "best_rd {m}");
                    beat_cases += 1;
                    prune_survivor_variety.insert(g.mode);
                }
                (None, None) => {
                    nobeat_cases += 1;
                }
                (g, cb) => {
                    panic!("presence mismatch {m}: rust={} c={}", g.is_some(), cb.is_some())
                }
            }
        }
    }
    assert!(beat_cases > 24, "winning searches: {beat_cases}");
    assert!(nobeat_cases > 2, "no-beat (tight budget) searches: {nobeat_cases}");
    assert!(prune_survivor_variety.len() > 3, "winner variety: {prune_survivor_variety:?}");
    assert!(factor_fired_total > 20, "variance-factor != 1.0 candidates: {factor_fired_total}");
    assert!(hog_masked_cases > 4, "real-HOG masked cases: {hog_masked_cases}");
}

/// The mode-info CDF set + the dual fill (Rust tables + the C reference
/// tables from the SAME CDFs) — the fill path is already differentially
/// validated in intra_mode_cost_diff.rs.
struct CdfSet {
    kf_y: Vec<u16>,
    y_mode: Vec<u16>,
    uv: Vec<u16>,
    fi_mode: Vec<u16>,
    fi: Vec<u16>,
    pal_y_mode: Vec<u16>,
    angle: Vec<u16>,
    intrabc: Vec<u16>,
}

fn gen_all_cdfs(rng: &mut Rng) -> CdfSet {
    let mut uv = gen_cdfs(rng, INTRA_MODES, UV_INTRA_MODES - 1, UV_INTRA_MODES + 1);
    uv.extend_from_slice(&gen_cdfs(rng, INTRA_MODES, UV_INTRA_MODES, UV_INTRA_MODES + 1));
    CdfSet {
        kf_y: gen_cdfs(rng, KF_MODE_CONTEXTS * KF_MODE_CONTEXTS, INTRA_MODES, INTRA_MODES + 1),
        y_mode: gen_cdfs(rng, BLOCK_SIZE_GROUPS, INTRA_MODES, INTRA_MODES + 1),
        uv,
        fi_mode: gen_cdfs(rng, 1, FILTER_INTRA_MODES, FILTER_INTRA_MODES + 1),
        fi: gen_cdfs(rng, BLOCK_SIZES_ALL, 2, 3),
        pal_y_mode: gen_cdfs(rng, PALETTE_BSIZE_CTXS * PALETTE_Y_MODE_CONTEXTS, 2, 3),
        angle: gen_cdfs(rng, DIRECTIONAL_MODES, 7, 8),
        intrabc: gen_cdfs(rng, 1, 2, 3),
    }
}

fn fill_both(cdfs: &CdfSet, enable_fi: bool) -> (Box<IntraModeCosts>, c::RefIntraModeCosts) {
    let want = c::ref_fill_intra_mode_costs(
        &cdfs.kf_y, &cdfs.y_mode, &cdfs.uv, &cdfs.fi_mode, &cdfs.fi, &cdfs.pal_y_mode,
        &cdfs.angle, &cdfs.intrabc, enable_fi,
    );
    let mut costs = IntraModeCosts::zeroed();
    fill_intra_mode_costs(
        &mut costs, &cdfs.kf_y, &cdfs.y_mode, &cdfs.uv, &cdfs.fi_mode, &cdfs.fi,
        &cdfs.pal_y_mode, &cdfs.angle, &cdfs.intrabc, enable_fi,
    );
    (costs, want)
}

/// The C-side mode loop: an independent transcription of
/// intra_mode_search.c:1545-1661 over REAL reference pieces.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn c_mode_loop(
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
    let mut best: Option<(usize, i32, usize, Vec<(usize, u16, u8)>, i32, i32, i64, i64)> = None;
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
            rate_tokenonly -=
                c::ref_tx_size_cost(ts_flat, true, bsize as i32, tx_size as i32, tx_size_ctx as i32);
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
                0, src, src_off, stride, recon_c, ref_off, stride, bsize, sb_size, mi_row,
                mi_col, 1 << 12, 1 << 12, bd, cvar, clog,
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
            best = Some((mode, delta, tx_size, winners, this_rate, rate_tokenonly, dist, this_rd));
        }
    }
    CLoopOut { best, rd_table, factor_fired }
}
