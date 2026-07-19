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
    Block4x4VarInfo, INTRA_MODES, IntraSbyGates, IntraSbySearchCfg, TOP_INTRA_MODEL_COUNT,
    rd_pick_intra_sby_mode_y,
};
use aom_encode::mode_costs::{TxSizeCosts, fill_tx_size_costs};
use aom_encode::tx_search::{TxTypeSearchPolicy, TxfmYrdEnv};
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_dsp::txb::{TxTypeCosts, fill_tx_type_costs};

mod common;
use common::*;

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
    let mut fi_winners = 0usize;
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
            let source_variance = if iter % 3 == 0 {
                rng.range(0, 256) as u32
            } else {
                256 + rng.range(0, 4096) as u32
            };
            // best_rd_in: mostly MAX; some tight budgets force no-beat + the
            // running-best-rd early exits inside pick_uniform.
            let best_rd_in = if iter == 7 { 1 << 10 } else { i64::MAX };
            let above_mode = if rng.next().is_multiple_of(4) {
                None
            } else {
                Some((rng.next() % 13) as i32)
            };
            let left_mode = if rng.next().is_multiple_of(4) {
                None
            } else {
                Some((rng.next() % 13) as i32)
            };

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
                (0..STRIDE * 96)
                    .map(|_| (rng.next() % (1u64 << bd)) as u16)
                    .collect()
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
                    &src,
                    src_off,
                    STRIDE,
                    bsize,
                    1 << 12,
                    1 << 12,
                    -1.2,
                    &mut skip_mask,
                );
                let mask_c = c::ref_prune_intra_mode_with_hog_y(
                    &src,
                    src_off,
                    STRIDE,
                    bsize,
                    1 << 12,
                    1 << 12,
                    bd,
                    -1.2,
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
            let coeff_costs = coeff_cost_set_from_tables(
                &txb_skip, &base_eob, &base, &eob_extra, &dc_sign, &lps, &eob_tbl,
            );
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
            let ts_flat: Vec<i32> = tx_size_costs
                .0
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            let skip_costs = [
                [rng.cost(), rng.cost()],
                [rng.cost(), rng.cost()],
                [rng.cost(), rng.cost()],
            ];
            let skip_ctx = (rng.next() % 3) as usize;
            let tx_size_ctx = (rng.next() % 3) as usize;
            let above_ctx: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let left_ctx: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let rdmult = rng.range(1, 1 << 22);

            // Mode-info cost tables (random CDFs through the dual fill).
            let cdf_set = gen_all_cdfs(&mut rng);
            let (mode_costs, c_mode_costs) = fill_both(&cdf_set, true);
            let mut pol = TxTypeSearchPolicy::speed0_allintra();
            // C9 `--use-intra-dct-only` sweep (forces the luma mask to DCT).
            let use_intra_dct_only = iter % 4 == 3;
            pol.use_intra_dct_only = use_intra_dct_only;
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
                qindex: qindex as i32,
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
                qm_levels: None,
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
                winner_mode: None,
                palette: None,
            };
            let mut recon_rust = recon0.clone();
            let mut var_cache_rust = Block4x4VarInfo::sb_cache(sb_size);
            let got = rd_pick_intra_sby_mode_y(
                &mut env,
                &mut recon_rust,
                &cfg,
                &mut var_cache_rust,
                best_rd_in,
            );

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
                (
                    &txb_skip, &base_eob, &base, &eob_extra, &dc_sign, &lps, &eob_tbl,
                ),
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
            use_intra_dct_only,
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
                    let wins: Vec<(usize, u16, u8)> = g
                        .winners
                        .iter()
                        .map(|w| (w.tx_type, w.eob, w.txb_ctx))
                        .collect();
                    assert_eq!(wins, cb.3, "winners {m}");
                    assert_eq!(g.rate, cb.4, "rate {m}");
                    assert_eq!(g.rate_tokenonly, cb.5, "rate_tokenonly {m}");
                    assert_eq!(g.dist, cb.6, "dist {m}");
                    assert!(!g.skippable, "intra non-skip {m}");
                    assert_eq!(g.best_rd, cb.7, "best_rd {m}");
                    assert_eq!(g.use_filter_intra, cb.8, "use_filter_intra {m}");
                    assert_eq!(g.filter_intra_mode, cb.9, "filter_intra_mode {m}");
                    if g.use_filter_intra {
                        fi_winners += 1;
                    }
                    beat_cases += 1;
                    prune_survivor_variety.insert(g.mode);
                }
                (None, None) => {
                    nobeat_cases += 1;
                }
                (g, cb) => {
                    panic!(
                        "presence mismatch {m}: rust={} c={}",
                        g.is_some(),
                        cb.is_some()
                    )
                }
            }
        }
    }
    assert!(beat_cases > 24, "winning searches: {beat_cases}");
    assert!(
        nobeat_cases > 2,
        "no-beat (tight budget) searches: {nobeat_cases}"
    );
    assert!(
        prune_survivor_variety.len() > 3,
        "winner variety: {prune_survivor_variety:?}"
    );
    assert!(
        factor_fired_total > 20,
        "variance-factor != 1.0 candidates: {factor_fired_total}"
    );
    assert!(
        hog_masked_cases > 4,
        "real-HOG masked cases: {hog_masked_cases}"
    );
    assert!(fi_winners > 4, "filter-intra winners: {fi_winners}");
}
