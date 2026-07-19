//! Differential harness for `encode_intra_block_plane_uv`
//! (`av1_encode_intra_block_plane` + `encode_block_intra_and_set_context`,
//! encodemb.c:702-823, planes 1/2, interior chroma-ref blocks) vs the same
//! walk over REAL C pieces (`common::c_encode_intra_block_plane_uv`): per txb
//! `av1_predict_intra_block_facade` — the plain UV arm (`get_uv_mode` +
//! UV angle via `ref_hbd_predict_intra`) or the signalled-CfL arm (fresh DC
//! prediction, dc-pred cache INACTIVE, + the REAL `av1_cfl_predict_block`
//! with the WINNER's `cfl_alpha_idx`/`cfl_alpha_signs`) ->
//! `ref_highbd_subtract_block` -> the REAL `av1_get_tx_type` UV arm
//! (`ref_get_tx_type_uv_intra`) -> `ref_fwd_txfm2d` + `ref_quant_plane_rows`
//! (FP when trellis / B when not) -> [trellis: `ref_get_txb_ctx` at the
//! SUBSAMPLED plane bsize + plane index + `ref_optimize_txb` at the CHROMA
//! trellis rd multiplier + `ref_txb_entropy_context`] -> `ref_inv_txfm2d_add`
//! -> the `av1_set_txb_context` stamp. NO `update_txk_array` reset and NO
//! `cfl_store_tx` (both `plane == 0`-gated in the C).
//!
//! Sweeps 420/422/444 (incl. the sub-8x8 odd-mi shared-chroma anchor and the
//! 444 64x64 4-txb UV walk), bd 8/10/12, all UV mode shapes (DC, directional
//! with nonzero UV angle, the SMOOTH family, PAETH, and signalled CfL with
//! random alpha and joint-sign), the trellis arms (FULL / NO_TRELLIS /
//! FINAL_PASS at DRY_RUN), the dead-but-modelled skip arm, and BOTH usage
//! arms of the chroma trellis-table sf (`use_chroma_trellis_rd_mult`:
//! ALLINTRA 13 / GOOD 20 — the only speed-0 tx-layer sf delta between the
//! usages).

use aom_encode::encode_intra::{
    TrellisOptType, UvEncodeParams, UvWinner, encode_intra_block_plane_uv, is_trellis_used,
};
use aom_encode::intra_uv_rd::{
    UV_CFL_PRED, UvRdEnv, av1_get_tx_size_uv, chroma_plane_offset, is_chroma_reference,
};
use aom_dsp::intra::cfl::{CflCtx, cfl_store_tx};
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_dsp::txb::{CoeffCostTables, TxTypeCosts};

mod common;
use common::*;

const STRIDE: usize = 256;

const BLK_W_L: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H_L: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

fn tx_of_dims(w: usize, h: usize) -> usize {
    match (w, h) {
        (4, 4) => 0,
        (8, 8) => 1,
        (16, 16) => 2,
        (32, 32) => 3,
        (64, 64) => 4,
        (4, 8) => 5,
        (8, 4) => 6,
        (8, 16) => 7,
        (16, 8) => 8,
        (16, 32) => 9,
        (32, 16) => 10,
        _ => unreachable!(),
    }
}

#[test]
fn encode_intra_block_plane_uv_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0x0c11_ab5e_7e11_a7e5);
    // (bsize, ss_x, ss_y, mi_row, mi_col): all subsamplings, sub-8x8 odd-mi
    // shared-chroma, CfL-legal (<=32) and CfL-forbidden (64) luma sizes, the
    // 444 64x64 4-txb UV walk, rects.
    let cases: [(usize, usize, usize, i32, i32); 10] = [
        (3, 1, 1, 8, 8),  // 8x8 @420 (uv 4x4)
        (6, 1, 1, 8, 8),  // 16x16 @420 (uv 8x8)
        (9, 1, 1, 8, 8),  // 32x32 @420 (uv 16x16)
        (12, 1, 1, 8, 8), // 64x64 @420 (uv 32x32; CfL forbidden: luma > 32)
        (0, 1, 1, 9, 9),  // 4x4 @420 sub-8x8 odd-mi (shared chroma anchor)
        (5, 1, 1, 8, 8),  // 16x8 @420 (uv 8x4)
        (6, 1, 0, 8, 8),  // 16x16 @422 (uv 8x16)
        (3, 1, 0, 8, 8),  // 8x8 @422 (uv 4x8)
        (6, 0, 0, 8, 8),  // 16x16 @444 (uv 16x16)
        (12, 0, 0, 8, 8), // 64x64 @444 (uv 64x64 -> adjusted TX_32X32: 4 txbs)
    ];

    let mut cfl_encodes = 0usize;
    let mut multi_txb = 0usize;
    let mut eob0 = 0usize;
    let mut eob_pos = 0usize;
    let mut angle_hits = 0usize;
    let mut skip_arm = 0usize;
    let mut no_trellis_arm = 0usize;
    let mut good_table_hits = 0usize;
    let mut sub8 = 0usize;

    for (ci, &(bsize, ss_x, ss_y, mi_row, mi_col)) in cases.iter().enumerate() {
        assert!(is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y));
        let (bw, bh) = (BLK_W_L[bsize], BLK_H_L[bsize]);
        let cfl_allowed = bw <= 32 && bh <= 32;
        let plane_bsize = aom_dsp::entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
        let (pw, ph) = (BLK_W_L[plane_bsize], BLK_H_L[plane_bsize]);
        let tx_size = av1_get_tx_size_uv(bsize, false, ss_x, ss_y);
        let n_txbs = (pw / TX_W[tx_size]) * (ph / TX_H[tx_size]);
        let ref_off =
            chroma_plane_offset(32 * STRIDE + 32, STRIDE, mi_row, mi_col, bsize, ss_x, ss_y);

        for iter in 0..12 {
            // BOTH usage arms of the chroma trellis-table sf.
            let use_chroma_trellis_rd_mult = iter % 2 == 0;
            if !use_chroma_trellis_rd_mult {
                good_table_hits += 1;
            }
            let bd: u8 = [8, 10, 12][iter % 3];
            let maxv = (1i64 << bd) - 1;
            let flat = iter % 6 == 5; // bait eob 0
            let qindex: usize = if flat {
                255
            } else {
                [16, 64, 128, 200, 255][iter % 5]
            };
            // UV mode: CfL every 3rd iter on CfL-legal shapes; else the
            // non-CfL modes (13), with UV angle on directional.
            let use_cfl = cfl_allowed && iter % 3 == 1;
            let uv_mode = if use_cfl {
                UV_CFL_PRED
            } else {
                (rng.next() % 13) as usize
            };
            let angle_delta_uv = if !use_cfl && (1..=8).contains(&uv_mode) {
                rng.range(-3, 4)
            } else {
                0
            };
            // Signalled CfL alphas: joint_sign 1..7 (ZERO/ZERO never signalled),
            // per-plane 4-bit indices.
            let joint_sign = rng.range(1, 8);
            let alpha_idx = rng.range(0, 256);
            let skip_txfm = iter == 8;
            let (opt_type, dry_out) = match iter % 8 {
                6 => (TrellisOptType::NoTrellisOpt, false),
                7 => (TrellisOptType::FinalPassTrellisOpt, false),
                _ => (TrellisOptType::FullTrellisOpt, false),
            };
            let use_trellis = is_trellis_used(opt_type, dry_out);
            let sharpness = if iter == 9 { 3 } else { 0 };
            let reduced = iter % 4 == 3;
            let amp = match iter % 4 {
                0 => (maxv / 16).max(8) as i32,
                1 => 24,
                2 => 6,
                _ => 96,
            };

            // CfL context loaded from a random luma recon (both sides) —
            // exactly the state the LUMA re-encode's cfl_store_tx leaves.
            let luma: Vec<u16> = (0..STRIDE * 96)
                .map(|_| (rng.next() % (1u64 << bd)) as u16)
                .collect();
            let luma_off = 16 * STRIDE + 16;
            let mut cfl_rust = CflCtx::new(ss_x as i32, ss_y as i32);
            let mut cfl_c = c::RefCflState::default();
            if use_cfl {
                let luma_tx = tx_of_dims(bw, bh);
                cfl_store_tx(
                    &mut cfl_rust,
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
                    &mut cfl_c,
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
                    bd,
                );
            }

            // Chroma planes: recon = neighbours; src = recon +- amp inside
            // the block (correlated content so eobs vary with q).
            let recon0: Vec<u16> = if flat {
                vec![(1u16 << (bd - 1)) - 5; STRIDE * 128]
            } else {
                (0..STRIDE * 128)
                    .map(|_| (rng.next() % (1u64 << bd)) as u16)
                    .collect()
            };
            let recon_v0: Vec<u16> = if flat {
                vec![(1u16 << (bd - 1)) + 5; STRIDE * 128]
            } else {
                (0..STRIDE * 128)
                    .map(|_| (rng.next() % (1u64 << bd)) as u16)
                    .collect()
            };
            let mut src_u = recon0.clone();
            let mut src_v = recon_v0.clone();
            if !flat {
                for r in 0..ph {
                    for cx in 0..pw {
                        let idx = ref_off + r * STRIDE + cx;
                        let vu = i64::from(src_u[idx]) + i64::from(rng.range(-amp, amp + 1));
                        let vv = i64::from(src_v[idx]) + i64::from(rng.range(-amp, amp + 1));
                        src_u[idx] = vu.clamp(0, maxv) as u16;
                        src_v[idx] = vv.clamp(0, maxv) as u16;
                    }
                }
            }

            // Quantizer rows (plane 1 / plane 2).
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows_u = set_q_index(&quants, &deq, qindex, 1);
            let rows_v = set_q_index(&quants, &deq, qindex, 2);
            let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
            let (rows_u_c, rows_v_c) = (&rows_c[56..112], &rows_c[112..168]);
            let dequant_u = [rows_u_c[48], rows_u_c[49]];
            let dequant_v = [rows_v_c[48], rows_v_c[49]];

            // Coefficient cost tables (trellis rate inputs; rate discarded).
            let t_txb_skip = tbl(&mut rng, 13 * 2);
            let t_base_eob = tbl(&mut rng, 4 * 3);
            let t_base = tbl(&mut rng, 42 * 8);
            let t_eob_extra = tbl(&mut rng, 9 * 2);
            let t_dc_sign = tbl(&mut rng, 3 * 2);
            let t_lps = tbl(&mut rng, 21 * 26);
            let t_eob = tbl(&mut rng, 2 * 11);
            let coeff_costs = CoeffCostTables {
                txb_skip: &t_txb_skip,
                base_eob: &t_base_eob,
                base: &t_base,
                eob_extra: &t_eob_extra,
                dc_sign: &t_dc_sign,
                lps: &t_lps,
                eob: &t_eob,
            };
            let tx_type_costs = TxTypeCosts::zeroed();
            let ttc_intra = vec![0i32; 3 * 4 * 13 * 16];
            let ttc_inter = vec![0i32; 4 * 4 * 16];

            let above_u: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let left_u: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let above_v: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let left_v: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let rdmult = rng.range(1, 1 << 22);
            let luma_mode = (rng.next() % 13) as usize;

            let env = UvRdEnv {
                sb_size: 12,
                bsize,
                mi_row,
                mi_col,
                chroma_up_available: true,
                chroma_left_available: true,
                tile_col_end: 1 << 16,
                tile_row_end: 1 << 16,
                partition: 0,
                mi_cols: 512,
                mi_rows: 512,
                ss_x,
                ss_y,
                ref_off: [ref_off, ref_off],
                ref_stride: STRIDE,
                src_u: &src_u,
                src_v: &src_v,
                src_off: [ref_off, ref_off],
                src_stride: STRIDE,
                disable_edge_filter: false,
                filter_type: 0,
                luma_mode,
                luma_use_fi: false,
                luma_fi_mode: 0,
                luma_palette_active: false,
                lossless: false,
                reduced_tx_set_used: reduced,
                bd,
                rows_u: &rows_u,
                rows_v: &rows_v,
                rdmult,
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                above_ctx: [&above_u, &above_v],
                left_ctx: [&left_u, &left_v],
                qm_levels: None,
            };
            let winner = UvWinner {
                uv_mode,
                angle_delta_uv,
                cfl_alpha_idx: alpha_idx,
                cfl_alpha_signs: joint_sign,
                palette: None,
            };
            let prm = UvEncodeParams {
                tx_size,
                skip_txfm,
                sharpness,
                enable_optimize_b: opt_type,
                dry_run_output_enabled: dry_out,
                use_chroma_trellis_rd_mult,
                tune: Default::default(),
            };

            let cenv = CUvEnv {
                use_intra_dct_only: false,
                partition: 0,
                bsize,
                mi_row,
                mi_col,
                ss_x,
                ss_y,
                ref_off: [ref_off, ref_off],
                src_off: [ref_off, ref_off],
                stride: STRIDE,
                src_u: &src_u,
                src_v: &src_v,
                luma_mode,
                luma_use_fi: false,
                luma_fi_mode: 0,
                lossless: false,
                reduced,
                bd,
                rows_u_c,
                rows_v_c,
                dequant_u,
                dequant_v,
                above_ctx: [&above_u, &above_v],
                left_ctx: [&left_u, &left_v],
                rdmult,
                coeff_tbls: (
                    &t_txb_skip,
                    &t_base_eob,
                    &t_base,
                    &t_eob_extra,
                    &t_dc_sign,
                    &t_lps,
                    &t_eob,
                ),
                ttc_tables: (&ttc_intra, &ttc_inter),
                use_chroma_trellis_rd_mult,
            };

            // Both chroma planes, in the C plane-loop order (1 then 2; the
            // planes are independent in this walk — the shared CfL ctx is
            // read-only after the luma store).
            for plane in [1usize, 2usize] {
                let recon_base = if plane == 1 { &recon0 } else { &recon_v0 };
                let mut recon_rust = recon_base.clone();
                let mut recon_c = recon_base.clone();

                let out = encode_intra_block_plane_uv(
                    &env,
                    &winner,
                    &prm,
                    plane,
                    &mut recon_rust,
                    &mut cfl_rust,
                );
                let (txbs_c, ta_c, tl_c) = c_encode_intra_block_plane_uv(
                    &cenv,
                    plane,
                    uv_mode,
                    angle_delta_uv,
                    if use_cfl {
                        Some((&mut cfl_c, alpha_idx, joint_sign))
                    } else {
                        None
                    },
                    tx_size,
                    skip_txfm,
                    use_trellis,
                    opt_type != TrellisOptType::NoTrellisOpt,
                    sharpness,
                    &mut recon_c,
                );

                let tag = format!(
                    "case {ci} (bs {bsize} ss {ss_x}{ss_y}) plane {plane} iter {iter} bd {bd} \
                     q {qindex} uv_mode {uv_mode} delta {angle_delta_uv} cfl {use_cfl} \
                     (idx {alpha_idx} js {joint_sign}) reduced {reduced} skip {skip_txfm} \
                     opt {opt_type:?} chroma_tbl {use_chroma_trellis_rd_mult} flat {flat}"
                );
                assert_eq!(out.txbs.len(), n_txbs, "txb count: {tag}");
                assert_eq!(out.txbs.len(), txbs_c.len(), "txb count vs C: {tag}");
                for (k, (r, cc)) in out.txbs.iter().zip(txbs_c.iter()).enumerate() {
                    assert_eq!(r.tx_type, cc.0, "txb {k} tx_type: {tag}");
                    assert_eq!(r.eob, cc.1, "txb {k} eob: {tag}");
                    assert_eq!(r.txb_entropy_ctx, cc.2, "txb {k} entropy ctx: {tag}");
                    assert_eq!(r.qcoeff, cc.3, "txb {k} qcoeff: {tag}");
                    assert_eq!(r.dqcoeff, cc.4, "txb {k} dqcoeff: {tag}");
                    if cc.1 == 0 {
                        eob0 += 1;
                    } else {
                        eob_pos += 1;
                    }
                }
                assert_eq!(recon_rust, recon_c, "final recon plane {plane}: {tag}");
                assert_eq!(out.ta, ta_c, "final ta: {tag}");
                assert_eq!(out.tl, tl_c, "final tl: {tag}");

                if use_cfl {
                    cfl_encodes += 1;
                }
                if n_txbs > 1 {
                    multi_txb += 1;
                }
                if angle_delta_uv != 0 {
                    angle_hits += 1;
                }
                if skip_txfm {
                    skip_arm += 1;
                }
                if !use_trellis {
                    no_trellis_arm += 1;
                }
                if mi_row % 2 == 1 || mi_col % 2 == 1 {
                    sub8 += 1;
                }
            }
        }
    }

    // Coverage floors.
    assert!(
        cfl_encodes >= 20,
        "signalled-CfL encodes exercised: {cfl_encodes}"
    );
    assert!(multi_txb >= 20, "multi-txb UV walks exercised: {multi_txb}");
    assert!(eob0 >= 20, "eob-0 txbs exercised: {eob0}");
    assert!(eob_pos >= 150, "eob>0 txbs exercised: {eob_pos}");
    assert!(
        angle_hits >= 15,
        "nonzero UV angle encodes exercised: {angle_hits}"
    );
    assert!(skip_arm >= 10, "skip_txfm arm exercised: {skip_arm}");
    assert!(
        no_trellis_arm >= 20,
        "no-trellis (B) arm exercised: {no_trellis_arm}"
    );
    assert!(
        good_table_hits >= 40,
        "GOOD trellis-table arm exercised: {good_table_hits}"
    );
    assert!(
        sub8 >= 10,
        "sub-8x8 odd-mi chroma-ref shapes exercised: {sub8}"
    );
}
