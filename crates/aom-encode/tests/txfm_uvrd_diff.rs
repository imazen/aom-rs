//! Differential harness for the chroma intra RD evaluation.
//!
//! `txfm_uvrd` / `txfm_rd_in_plane_uv` (av1_txfm_uvrd, av1_txfm_rd_in_plane
//! and block_rd_txfm; chroma intra, speed-0 policy, interior blocks) vs the
//! same walk over REAL C pieces (ref_intra_avail w/ subsampling, then
//! ref_hbd_predict_intra, then the plane-aware search_tx_type chain w/ the
//! pinned UV tx type + chroma trellis rd mult) — non-CfL UV modes across
//! 420/422/444, bd 8/10/12, sub-8x8 chroma-ref shapes, per-plane u/v
//! quantizer deltas, angle deltas, tight-budget invalidation.
//!
//! Plus the CfL fixed-alpha full-RD path (`txfm_rd_in_plane_uv` with
//! `CflPredict`) vs the C walk over the REAL `av1_cfl_predict_block` and
//! `cfl_store_tx`, incl. the encoder DC-prediction cache row-replication.
//! Both sides start from IDENTICAL recon planes; final planes compared.

use aom_encode::intra_uv_rd::{
    CflDcCache, CflPredict, UV_CFL_PRED, UvRdEnv, av1_get_tx_size_uv, chroma_plane_offset,
    is_chroma_reference, txfm_rd_in_plane_uv, txfm_uvrd,
};
use aom_encode::tx_search::TxTypeSearchPolicy;
use aom_intra::cfl::{CflCtx, cfl_store_tx};
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_txb::{CoeffCostTables, TxTypeCosts, fill_tx_type_costs};

mod common;
use common::*;

const STRIDE: usize = 256;

struct Tables {
    txb_skip: Vec<i32>,
    base_eob: Vec<i32>,
    base: Vec<i32>,
    eob_extra: Vec<i32>,
    dc_sign: Vec<i32>,
    lps: Vec<i32>,
    eob_tbl: Vec<i32>,
    ttc_intra: Vec<i32>,
    ttc_inter: Vec<i32>,
    tx_type_costs: Box<TxTypeCosts>,
}

fn gen_tables(rng: &mut Rng) -> Tables {
    const NUM_EXT_TX_SET: [usize; 6] = [1, 2, 5, 7, 12, 16];
    const IDX_TO_TYPE: [[usize; 4]; 2] = [[0, 3, 2, 0], [0, 5, 4, 1]];
    let mut ttc_intra_cdf = Vec::new();
    for s in 0..3 {
        let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[0][s]].max(2);
        ttc_intra_cdf.extend_from_slice(&gen_cdfs(rng, 4 * 13, ns, 17));
    }
    let mut ttc_inter_cdf = Vec::new();
    for s in 0..4 {
        let ns = NUM_EXT_TX_SET[IDX_TO_TYPE[1][s]].max(2);
        ttc_inter_cdf.extend_from_slice(&gen_cdfs(rng, 4, ns, 17));
    }
    let (ttc_intra, ttc_inter) = c::ref_fill_tx_type_costs(&ttc_intra_cdf, &ttc_inter_cdf);
    let mut tx_type_costs = TxTypeCosts::zeroed();
    fill_tx_type_costs(&mut tx_type_costs, &ttc_intra_cdf, &ttc_inter_cdf);
    Tables {
        txb_skip: tbl(rng, 13 * 2),
        base_eob: tbl(rng, 4 * 3),
        base: tbl(rng, 42 * 8),
        eob_extra: tbl(rng, 9 * 2),
        dc_sign: tbl(rng, 3 * 2),
        lps: tbl(rng, 21 * 26),
        eob_tbl: tbl(rng, 2 * 11),
        ttc_intra,
        ttc_inter,
        tx_type_costs,
    }
}

/// Build one paired (Rust env, C env) chroma scenario. Returns everything the
/// two walks need, with per-plane recon/src planes seeded identically.
#[allow(clippy::type_complexity)]
struct Scenario {
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
    mi_row: i32,
    mi_col: i32,
    ref_off: [usize; 2],
    recon_u0: Vec<u16>,
    recon_v0: Vec<u16>,
    src_u: Vec<u16>,
    src_v: Vec<u16>,
    luma_mode: usize,
    luma_use_fi: bool,
    luma_fi_mode: usize,
    reduced: bool,
    bd: u8,
    qindex: usize,
    rdmult: i32,
    above_u: Vec<i8>,
    left_u: Vec<i8>,
    above_v: Vec<i8>,
    left_v: Vec<i8>,
}

fn build_scenario(
    rng: &mut Rng,
    bsize: usize,
    ss_x: usize,
    ss_y: usize,
    mi_row: i32,
    mi_col: i32,
    iter: usize,
) -> Scenario {
    let bd: u8 = [8, 10, 12][iter % 3];
    let maxv = (1i64 << bd) - 1;
    let amp = match iter % 4 {
        0 => maxv,
        1 => 24,
        2 => 6,
        _ => 96,
    };
    let qindex = [16, 64, 128, 200, 255][iter % 5] as usize;
    let plane_bsize = aom_entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
    let (pw, ph) = (BLK_W[plane_bsize], BLK_H[plane_bsize]);
    let ref_off_u = chroma_plane_offset(0, STRIDE, mi_row, mi_col, bsize, ss_x, ss_y);
    let recon_u0: Vec<u16> = (0..STRIDE * 128)
        .map(|_| (rng.next() % (1u64 << bd)) as u16)
        .collect();
    let recon_v0: Vec<u16> = (0..STRIDE * 128)
        .map(|_| (rng.next() % (1u64 << bd)) as u16)
        .collect();
    let mut src_u = recon_u0.clone();
    let mut src_v = recon_v0.clone();
    for (src, recon) in [(&mut src_u, &recon_u0), (&mut src_v, &recon_v0)] {
        for r in 0..ph {
            for cx in 0..pw {
                let idx = ref_off_u + r * STRIDE + cx;
                let v = i64::from(recon[idx]) + i64::from(rng.range(-(amp as i32), amp as i32 + 1));
                src[idx] = v.clamp(0, maxv) as u16;
            }
        }
    }
    let luma_mode = (rng.next() % 13) as usize;
    let luma_use_fi = rng.next().is_multiple_of(5);
    Scenario {
        bsize,
        ss_x,
        ss_y,
        mi_row,
        mi_col,
        ref_off: [ref_off_u, ref_off_u],
        recon_u0,
        recon_v0,
        src_u,
        src_v,
        luma_mode,
        luma_use_fi,
        luma_fi_mode: (rng.next() % 5) as usize,
        reduced: iter % 4 == 3,
        bd,
        qindex,
        rdmult: rng.range(1, 1 << 22),
        above_u: (0..32)
            .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
            .collect(),
        left_u: (0..32)
            .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
            .collect(),
        above_v: (0..32)
            .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
            .collect(),
        left_v: (0..32)
            .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
            .collect(),
    }
}

#[test]
fn txfm_uvrd_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0x0a11_ab0a_00d1_ff01);
    // (bsize, ss_x, ss_y, mi_row, mi_col): chroma-ref shapes incl. sub-8x8
    // (odd mi at subsampled axes) and the 444 64x64 multi-txb walk.
    let cases: [(usize, usize, usize, i32, i32); 14] = [
        (3, 1, 1, 8, 8),  // 8x8 @420 -> plane 4x4
        (6, 1, 1, 8, 8),  // 16x16 @420 -> 8x8
        (9, 1, 1, 8, 8),  // 32x32 @420 -> 16x16
        (5, 1, 1, 8, 8),  // 16x8 @420 -> 8x4
        (4, 1, 1, 8, 8),  // 8x16 @420 -> 4x8
        (12, 1, 1, 8, 8), // 64x64 @420 -> 32x32
        (0, 1, 1, 9, 9),  // 4x4 @420 sub-8x8 chroma-ref (odd mi both)
        (1, 1, 1, 9, 9),  // 4x8 @420 (odd col)
        (2, 1, 1, 9, 9),  // 8x4 @420 (odd row)
        (6, 1, 0, 8, 8),  // 16x16 @422 -> 8x16
        (2, 1, 0, 8, 8),  // 8x4 @422 -> 4x4
        (6, 0, 0, 8, 8),  // 16x16 @444
        (12, 0, 0, 8, 8), // 64x64 @444 -> uv_tx 32x32, FOUR txbs
        (0, 0, 0, 9, 9),  // 4x4 @444
    ];
    let mut invalid_hits = 0usize;
    let mut multi_txb_hits = 0usize;
    let mut angle_hits = 0usize;

    for (ci, &(bsize, ss_x, ss_y, mi_row, mi_col)) in cases.iter().enumerate() {
        assert!(
            is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y),
            "case {ci} must be a chroma-ref block",
        );
        for iter in 0..8 {
            // Sweep BOTH usage arms (ALLINTRA chroma trellis mult 13 /
            // GOOD 20).
            let pol = if iter % 2 == 0 {
                TxTypeSearchPolicy::speed0_allintra()
            } else {
                TxTypeSearchPolicy::speed0_good()
            };
            let sc = build_scenario(&mut rng, bsize, ss_x, ss_y, mi_row, mi_col, ci + iter);
            let t = gen_tables(&mut rng);
            let coeff_costs = CoeffCostTables {
                txb_skip: &t.txb_skip,
                base_eob: &t.base_eob,
                base: &t.base,
                eob_extra: &t.eob_extra,
                dc_sign: &t.dc_sign,
                lps: &t.lps,
                eob: &t.eob_tbl,
            };
            // Per-plane quantizers with u/v deltas.
            let (udc, uac, vdc, vac) = (
                rng.range(-12, 13),
                rng.range(-12, 13),
                rng.range(-12, 13),
                rng.range(-12, 13),
            );
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(sc.bd, 0, udc, uac, vdc, vac, &mut quants, &mut deq, 0);
            let rows_u = set_q_index(&quants, &deq, sc.qindex, 1);
            let rows_v = set_q_index(&quants, &deq, sc.qindex, 2);
            let rows_c =
                c::ref_set_q_index(sc.bd as i32, 0, udc, uac, vdc, vac, 0, sc.qindex as i32);
            let (rows_u_c, rows_v_c) = (&rows_c[56..112], &rows_c[112..168]);
            let dequant_u = [rows_u_c[48], rows_u_c[49]];
            let dequant_v = [rows_v_c[48], rows_v_c[49]];

            // Candidate: non-CfL UV mode (+ angle for directional).
            let uv_mode = (rng.next() % 13) as usize;
            let im = aom_entropy::partition::get_uv_mode(uv_mode);
            let angle_delta_uv = if (1..=8).contains(&im) {
                rng.range(-3, 4)
            } else {
                0
            };
            if angle_delta_uv != 0 {
                angle_hits += 1;
            }
            let ref_best_rd = if iter == 7 { 1 << 10 } else { i64::MAX };

            let env = UvRdEnv {
                sb_size: 12,
                bsize: sc.bsize,
                mi_row: sc.mi_row,
                mi_col: sc.mi_col,
                chroma_up_available: true,
                chroma_left_available: true,
                tile_col_end: 1 << 16,
                tile_row_end: 1 << 16,
                partition: 0,
                mi_cols: 512,
                mi_rows: 512,
                ss_x: sc.ss_x,
                ss_y: sc.ss_y,
                ref_off: sc.ref_off,
                ref_stride: STRIDE,
                src_u: &sc.src_u,
                src_v: &sc.src_v,
                src_off: sc.ref_off,
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                luma_mode: sc.luma_mode,
                luma_use_fi: sc.luma_use_fi,
                luma_fi_mode: sc.luma_fi_mode,
                lossless: false,
                reduced_tx_set_used: sc.reduced,
                bd: sc.bd,
                rows_u: &rows_u,
                rows_v: &rows_v,
                rdmult: sc.rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &t.tx_type_costs,
                above_ctx: [&sc.above_u, &sc.above_v],
                left_ctx: [&sc.left_u, &sc.left_v],
            };
            let mut recon_u = sc.recon_u0.clone();
            let mut recon_v = sc.recon_v0.clone();
            let rust = txfm_uvrd(
                &env,
                &mut recon_u,
                &mut recon_v,
                uv_mode,
                angle_delta_uv,
                ref_best_rd,
                &pol,
            );

            let cenv = CUvEnv {
                partition: 0,
                bsize: sc.bsize,
                mi_row: sc.mi_row,
                mi_col: sc.mi_col,
                ss_x: sc.ss_x,
                ss_y: sc.ss_y,
                ref_off: sc.ref_off,
                src_off: sc.ref_off,
                stride: STRIDE,
                src_u: &sc.src_u,
                src_v: &sc.src_v,
                luma_mode: sc.luma_mode,
                luma_use_fi: sc.luma_use_fi,
                luma_fi_mode: sc.luma_fi_mode,
                lossless: false,
                reduced: sc.reduced,
                bd: sc.bd,
                rows_u_c,
                rows_v_c,
                dequant_u,
                dequant_v,
                above_ctx: [&sc.above_u, &sc.above_v],
                left_ctx: [&sc.left_u, &sc.left_v],
                rdmult: sc.rdmult,
                coeff_tbls: (
                    &t.txb_skip,
                    &t.base_eob,
                    &t.base,
                    &t.eob_extra,
                    &t.dc_sign,
                    &t.lps,
                    &t.eob_tbl,
                ),
                ttc_tables: (&t.ttc_intra, &t.ttc_inter),
                use_chroma_trellis_rd_mult: pol.use_chroma_trellis_rd_mult,
            };
            let mut recon_u_c = sc.recon_u0.clone();
            let mut recon_v_c = sc.recon_v0.clone();
            let cres = c_txfm_uvrd(
                &cenv,
                &mut recon_u_c,
                &mut recon_v_c,
                uv_mode,
                angle_delta_uv,
                ref_best_rd,
            );

            let ctx = format!(
                "case={ci} iter={iter} bsize={bsize} ss=({ss_x},{ss_y}) uv={uv_mode} ad={angle_delta_uv} bd={} q={}",
                sc.bd, sc.qindex,
            );
            match (rust, cres) {
                (None, None) => invalid_hits += 1,
                (Some((stats, wu, wv)), Some((crate_, cdist, csse, cwu, cwv))) => {
                    assert_eq!(stats.rate, crate_, "{ctx} rate");
                    assert_eq!(stats.dist, cdist, "{ctx} dist");
                    assert_eq!(stats.sse, csse, "{ctx} sse");
                    // Intra chroma always merges non-skip per txb.
                    assert!(!stats.skip_txfm, "{ctx} skip");
                    let wu_t: Vec<(usize, u16, u8)> =
                        wu.iter().map(|w| (w.tx_type, w.eob, w.txb_ctx)).collect();
                    let wv_t: Vec<(usize, u16, u8)> =
                        wv.iter().map(|w| (w.tx_type, w.eob, w.txb_ctx)).collect();
                    assert_eq!(wu_t, cwu, "{ctx} winners U");
                    assert_eq!(wv_t, cwv, "{ctx} winners V");
                    assert_eq!(recon_u, recon_u_c, "{ctx} recon U");
                    assert_eq!(recon_v, recon_v_c, "{ctx} recon V");
                    if wu_t.len() > 1 {
                        multi_txb_hits += 1;
                    }
                }
                (r, c_) => panic!(
                    "{ctx} validity split: rust={:?} c={:?}",
                    r.is_some(),
                    c_.is_some()
                ),
            }
        }
    }
    assert!(
        invalid_hits > 4,
        "tight-budget invalidation under-exercised: {invalid_hits}"
    );
    assert!(
        multi_txb_hits > 4,
        "multi-txb UV walk under-exercised: {multi_txb_hits}"
    );
    assert!(
        angle_hits > 20,
        "UV angle deltas under-exercised: {angle_hits}"
    );
}

/// CfL fixed-alpha full-RD evaluation (`cfl_compute_rd` fast_mode=0 inner):
/// `txfm_rd_in_plane_uv` with `CflPredict` vs the C walk over the REAL
/// `av1_cfl_predict_block`, threading one CfL context (loaded from a luma
/// recon via both sides' `cfl_store_tx`) and the DC-prediction cache across
/// repeated evaluations of the same plane (cache off -> on transition).
#[test]
fn txfm_rd_in_plane_uv_cfl_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0xcf1a_1fa5_0a1c_0002);
    // CfL-legal (<=32x32) chroma-ref shapes.
    let cases: [(usize, usize, usize, i32, i32); 8] = [
        (3, 1, 1, 8, 8),
        (6, 1, 1, 8, 8),
        (9, 1, 1, 8, 8),
        (0, 1, 1, 9, 9),
        (5, 1, 1, 8, 8),
        (6, 1, 0, 8, 8),
        (6, 0, 0, 8, 8),
        (9, 0, 0, 8, 8),
    ];
    let mut cached_evals = 0usize;
    for (ci, &(bsize, ss_x, ss_y, mi_row, mi_col)) in cases.iter().enumerate() {
        for iter in 0..8 {
            // Sweep BOTH usage arms (ALLINTRA chroma trellis mult 13 /
            // GOOD 20).
            let pol = if iter % 2 == 0 {
                TxTypeSearchPolicy::speed0_allintra()
            } else {
                TxTypeSearchPolicy::speed0_good()
            };
            let sc = build_scenario(&mut rng, bsize, ss_x, ss_y, mi_row, mi_col, ci + iter);
            let t = gen_tables(&mut rng);
            let coeff_costs = CoeffCostTables {
                txb_skip: &t.txb_skip,
                base_eob: &t.base_eob,
                base: &t.base,
                eob_extra: &t.eob_extra,
                dc_sign: &t.dc_sign,
                lps: &t.lps,
                eob: &t.eob_tbl,
            };
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(sc.bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows_u = set_q_index(&quants, &deq, sc.qindex, 1);
            let rows_v = set_q_index(&quants, &deq, sc.qindex, 2);
            let rows_c = c::ref_set_q_index(sc.bd as i32, 0, 0, 0, 0, 0, 0, sc.qindex as i32);
            let (rows_u_c, rows_v_c) = (&rows_c[56..112], &rows_c[112..168]);
            let dequant_u = [rows_u_c[48], rows_u_c[49]];
            let dequant_v = [rows_v_c[48], rows_v_c[49]];

            // Load the CfL luma context on BOTH sides from a random luma recon
            // (the av1_encode_intra_block_plane product in the real flow).
            let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
            let luma: Vec<u16> = (0..STRIDE * 96)
                .map(|_| (rng.next() % (1u64 << sc.bd)) as u16)
                .collect();
            let luma_off = 16 * STRIDE + 16;
            const TX_OF_DIMS: fn(usize, usize) -> usize = |w, h| match (w, h) {
                (4, 4) => 0,
                (8, 8) => 1,
                (16, 16) => 2,
                (32, 32) => 3,
                (4, 8) => 5,
                (8, 4) => 6,
                (8, 16) => 7,
                (16, 8) => 8,
                (16, 32) => 9,
                (32, 16) => 10,
                _ => unreachable!(),
            };
            let luma_tx = TX_OF_DIMS(bw, bh);
            let mut cfl_ctx = CflCtx::new(ss_x as i32, ss_y as i32);
            let mut st_c = c::RefCflState::default();
            cfl_store_tx(
                &mut cfl_ctx,
                &luma,
                luma_off,
                STRIDE,
                0,
                0,
                luma_tx,
                bsize,
                mi_row,
                mi_col,
            );
            c::ref_cfl_store_tx(
                &mut st_c,
                &luma,
                luma_off,
                STRIDE,
                0,
                0,
                luma_tx,
                bsize,
                mi_row,
                mi_col,
                ss_x as i32,
                ss_y as i32,
                sc.bd,
            );

            let env = UvRdEnv {
                sb_size: 12,
                bsize: sc.bsize,
                mi_row: sc.mi_row,
                mi_col: sc.mi_col,
                chroma_up_available: true,
                chroma_left_available: true,
                tile_col_end: 1 << 16,
                tile_row_end: 1 << 16,
                partition: 0,
                mi_cols: 512,
                mi_rows: 512,
                ss_x: sc.ss_x,
                ss_y: sc.ss_y,
                ref_off: sc.ref_off,
                ref_stride: STRIDE,
                src_u: &sc.src_u,
                src_v: &sc.src_v,
                src_off: sc.ref_off,
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                luma_mode: sc.luma_mode,
                luma_use_fi: sc.luma_use_fi,
                luma_fi_mode: sc.luma_fi_mode,
                lossless: false,
                reduced_tx_set_used: sc.reduced,
                bd: sc.bd,
                rows_u: &rows_u,
                rows_v: &rows_v,
                rdmult: sc.rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &t.tx_type_costs,
                above_ctx: [&sc.above_u, &sc.above_v],
                left_ctx: [&sc.left_u, &sc.left_v],
            };
            let cenv = CUvEnv {
                partition: 0,
                bsize: sc.bsize,
                mi_row: sc.mi_row,
                mi_col: sc.mi_col,
                ss_x: sc.ss_x,
                ss_y: sc.ss_y,
                ref_off: sc.ref_off,
                src_off: sc.ref_off,
                stride: STRIDE,
                src_u: &sc.src_u,
                src_v: &sc.src_v,
                luma_mode: sc.luma_mode,
                luma_use_fi: sc.luma_use_fi,
                luma_fi_mode: sc.luma_fi_mode,
                lossless: false,
                reduced: sc.reduced,
                bd: sc.bd,
                rows_u_c,
                rows_v_c,
                dequant_u,
                dequant_v,
                above_ctx: [&sc.above_u, &sc.above_v],
                left_ctx: [&sc.left_u, &sc.left_v],
                rdmult: sc.rdmult,
                coeff_tbls: (
                    &t.txb_skip,
                    &t.base_eob,
                    &t.base,
                    &t.eob_extra,
                    &t.dc_sign,
                    &t.lps,
                    &t.eob_tbl,
                ),
                ttc_tables: (&t.ttc_intra, &t.ttc_inter),
                use_chroma_trellis_rd_mult: pol.use_chroma_trellis_rd_mult,
            };

            let uv_tx = av1_get_tx_size_uv(bsize, false, ss_x, ss_y);
            let mut cache = CflDcCache::cleared();
            let mut cache_c = CDcCache::cleared();
            cache.use_cache = iter % 2 == 1; // cfl_rd_pick_alpha turns it on
            cache_c.use_cache = cache.use_cache;

            // Three alpha evaluations per plane on the SAME context (the
            // pick-plane-rd shape): dc cache fills on the first, replays after.
            for round in 0..3 {
                let joint_sign = (rng.next() % 8) as i32;
                let alpha_idx = (rng.next() % 256) as i32;
                let plane = 1 + ((iter + round) % 2);
                let mut recon = if plane == 1 {
                    sc.recon_u0.clone()
                } else {
                    sc.recon_v0.clone()
                };
                let mut recon_c = recon.clone();
                let mut cflp = CflPredict {
                    ctx: &mut cfl_ctx,
                    cache: &mut cache,
                    alpha_idx,
                    joint_sign,
                };
                let rust = txfm_rd_in_plane_uv(
                    &env,
                    &mut recon,
                    plane,
                    UV_CFL_PRED,
                    0,
                    Some(&mut cflp),
                    uv_tx,
                    i64::MAX,
                    0,
                    &pol,
                );
                let cres = c_txfm_rd_in_plane_uv(
                    &cenv,
                    &mut recon_c,
                    plane,
                    UV_CFL_PRED,
                    0,
                    Some((&mut st_c, &mut cache_c, alpha_idx, joint_sign)),
                    uv_tx,
                    i64::MAX,
                    0,
                );
                let ctx = format!(
                    "case={ci} iter={iter} round={round} plane={plane} js={joint_sign} idx={alpha_idx}",
                );
                let (stats, winners) = rust.expect("budget-free walk is valid");
                let (crate_, cdist, csse, cw) = cres.expect("budget-free C walk is valid");
                assert_eq!(
                    (stats.rate, stats.dist, stats.sse),
                    (crate_, cdist, csse),
                    "{ctx}",
                );
                let w_t: Vec<(usize, u16, u8)> = winners
                    .iter()
                    .map(|w| (w.tx_type, w.eob, w.txb_ctx))
                    .collect();
                assert_eq!(w_t, cw, "{ctx} winners");
                assert_eq!(recon, recon_c, "{ctx} recon");
                if cache.use_cache && round > 0 {
                    cached_evals += 1;
                }
            }
            // Both sides' CfL AC state stays in lockstep.
            assert_eq!(
                &cfl_ctx.ac_buf_q3[..],
                &st_c.ac_q3[..],
                "case={ci} iter={iter} ac"
            );
        }
    }
    assert!(
        cached_evals > 20,
        "dc-pred cache replay under-exercised: {cached_evals}"
    );
}
