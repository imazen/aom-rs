//! Differential harness for `encode_intra_block_plane_y`
//! (`av1_encode_intra_block_plane` + `encode_block_intra_and_set_context`,
//! encodemb.c:702-823, luma, interior blocks) vs the same walk over REAL C
//! pieces: per txb `ref_intra_avail` + `ref_hbd_predict_intra` (into the
//! C-side recon plane) -> `ref_highbd_subtract_block` -> the REAL
//! `av1_get_tx_type` Y arm (`ref_get_tx_type_y` over the marshalled
//! block-local tx_type_map) -> `ref_fwd_txfm2d` + `ref_quant_plane_rows`
//! (FP when trellis / B when not, `USE_B_QUANT_NO_TRELLIS == 1`) ->
//! [trellis: `ref_get_txb_ctx` + `ref_optimize_txb` + `ref_txb_entropy_context`]
//! -> `ref_inv_txfm2d_add` into the C recon -> [eob 0: the REAL
//! `update_txk_array` DCT reset via `ref_update_txk_array`] -> [store_y: the
//! REAL exported `cfl_store_tx` via `ref_cfl_store_tx`] -> the
//! `av1_set_txb_context` stamp. Asserts per txb (tx_type, eob, qcoeff,
//! dqcoeff, entropy ctx), the final recon planes, the final tx_type_maps
//! (the eob-0 resets, incl. the TX_64X64 16x16-unit fill), the final local
//! ta/tl, and the threaded CfL state (recon_buf_q3 / extent / invalidation).
//!
//! Trellis gating swept per `is_trellis_used` (encodemb.h:153): FULL always
//! trellis; FINAL_PASS only at OUTPUT_ENABLED; NO_TRELLIS never (B quant) and
//! also leaves ta/tl zeroed (`if (enable_optimize_b)` gate, encodemb.c:817).
//! The skip_txfm arm (dead in the KF intra RD path) is exercised explicitly.

use aom_encode::encode_intra::{
    EncodeIntraYEnv, TrellisOptType, encode_intra_block_plane_y, is_trellis_used,
};
use aom_encode::tx_search::AV1_EXT_TX_USED_FLAG;
use aom_intra::cfl::{CFL_BUF_SQUARE, CflCtx};
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_txb::{CoeffCostTables, ext_tx_set_type};

mod common;
use common::*;

const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_SIZE_HIGH_B: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// In-set tx types for an intra block at (tx_size, reduced) — the
/// `av1_ext_tx_used[set]` membership the map's origin cells must satisfy
/// (the C assert is live). sqr_up > 32 sizes are read as DCT before the map.
fn in_set_types(tx_size: usize, reduced: bool) -> Vec<usize> {
    let set = ext_tx_set_type(tx_size, false, reduced);
    let mask = AV1_EXT_TX_USED_FLAG[set];
    (0..16).filter(|t| mask & (1 << t) != 0).collect()
}

#[test]
fn encode_intra_block_plane_y_matches_c_walk() {
    c::ref_init();
    let mut rng = Rng(0x6e0d_e11a_5b1a_57e6);
    const STRIDE: usize = 256;
    // (bsize, tx_size): multi-txb walks, rects, the 64-side map fill, sub-8x8.
    let pairs: [(usize, usize); 12] = [
        (3, 0),   // 8x8 @ TX_4X4: 4 txbs
        (6, 1),   // 16x16 @ TX_8X8: 4 txbs
        (6, 2),   // 16x16 @ TX_16X16: 1
        (5, 1),   // 16x8 @ TX_8X8: 2
        (9, 2),   // 32x32 @ TX_16X16: 4
        (12, 4),  // 64x64 @ TX_64X64: the update_txk_array 16x16-unit fill
        (12, 3),  // 64x64 @ TX_32X32: 4 txbs
        (0, 0),   // 4x4
        (1, 5),   // 4x8 @ TX_4X8
        (1, 0),   // 4x8 @ TX_4X4: 2 txbs (sub-8x8 cfl store)
        (17, 14), // 16x4 @ TX_16X4
        (16, 13), // 4x16 @ TX_4X16
    ];

    let mut eob0_resets = 0usize;
    let mut eob_pos = 0usize;
    let mut stores = 0usize;
    let mut sub8_stores = 0usize;
    let mut skip_arm = 0usize;
    let mut no_trellis_arm = 0usize;
    let mut fill64 = 0usize;

    for (pi, &(bsize, tx_size)) in pairs.iter().enumerate() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
        let (txwu, txhu) = (txw >> 2, txh >> 2);
        let mbw = MI_SIZE_WIDE_B[bsize];
        let mbh = MI_SIZE_HIGH_B[bsize];
        let n_txbs = (bw / txw) * (bh / txh);

        for iter in 0..12 {
            let bd: u8 = if iter % 3 == 2 { 12 } else { 8 };
            let flat = iter % 6 == 5; // force eob 0 (+ the 64-side map fill)
            let amp = match iter % 4 {
                0 => {
                    if bd > 8 {
                        4095
                    } else {
                        255
                    }
                }
                1 => 24,
                2 => 6,
                _ => 96,
            };
            let qindex: usize = if flat {
                255
            } else {
                [16, 64, 128, 200, 255][iter % 5]
            };
            let mode = (rng.next() % 13) as usize;
            let angle_delta = if (1..=8).contains(&mode) {
                rng.range(-3, 4)
            } else {
                0
            };
            let use_fi = mode == 0 && bw <= 32 && bh <= 32 && iter % 4 == 1;
            let fi_mode = if use_fi { (rng.next() % 5) as usize } else { 0 };
            let reduced = iter % 4 == 3;
            // skip arm on a store-active iteration (predict + cfl store still
            // run under skip; the transform does not).
            let skip_txfm = iter == 8;
            // Trellis arms: mostly speed-0 (FULL + DRY_RUN_NORMAL); iter 6 =
            // NO_TRELLIS (B quant, ta/tl stay zeroed); iter 7 = FINAL_PASS at
            // DRY_RUN_NORMAL (B quant, ta/tl loaded).
            let (opt_type, dry_out) = match iter % 8 {
                6 => (TrellisOptType::NoTrellisOpt, false),
                7 => (TrellisOptType::FinalPassTrellisOpt, false),
                _ => (TrellisOptType::FullTrellisOpt, false),
            };
            let use_trellis = is_trellis_used(opt_type, dry_out);
            let sharpness = if iter == 9 { 3 } else { 0 };

            // store_y: is_cfl_allowed-shaped gate (dims <= 32).
            let store_y = bw <= 32 && bh <= 32 && iter % 3 != 1;
            let (ss_x, ss_y): (i32, i32) = [(1, 1), (0, 0), (1, 0), (1, 1)][iter % 4];
            // Odd mi position exercises the sub-8x8 shared-chroma adjust —
            // only on axes where the partition tree can place the block there
            // (odd mi_row only for bh == 4, odd mi_col only for bw == 4; the
            // C sub8x8_adjust_offset asserts the txb offset is 0 on the
            // adjusted axis, which legal geometry implies).
            let odd = (bw == 4 || bh == 4) && iter % 2 == 1;
            let mi_row = if odd && bh == 4 { 9 } else { 8 };
            let mi_col = if odd && bw == 4 { 9 } else { 8 };

            let ref_off = 32 * STRIDE + 32;
            let src_off = 32 * STRIDE + 32;

            // Planes.
            let recon0: Vec<u16> = if flat {
                vec![(1u16 << (bd - 1)) - 3; STRIDE * 96]
            } else {
                (0..STRIDE * 96)
                    .map(|_| (rng.next() % (1u64 << bd)) as u16)
                    .collect()
            };
            let mut src = recon0.clone();
            if !flat {
                for r in 0..bh {
                    for cx in 0..bw {
                        let idx = src_off + r * STRIDE + cx;
                        let v = i64::from(recon0[idx]) + i64::from(rng.range(-amp, amp + 1));
                        src[idx] = v.clamp(0, (1 << bd) - 1) as u16;
                    }
                }
            }

            // Quantizer rows.
            let mut quants = Quants::zeroed();
            let mut deq = Dequants::zeroed();
            av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
            let rows = set_q_index(&quants, &deq, qindex, 0);
            let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
            let plane_rows_c = &rows_c[0..7 * 8];
            let dequant = [rows.dequant[0], rows.dequant[1]];

            // Coefficient cost tables (trellis rate inputs; rate discarded).
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

            // The block-local tx_type_map: in-set types at txb origins,
            // arbitrary valid types elsewhere (dead cells). Identical sides.
            let map_stride = mbw;
            let allowed = in_set_types(tx_size, reduced);
            let mut map0: Vec<u8> = (0..mbw * mbh).map(|_| (rng.next() % 16) as u8).collect();
            for blk_row in (0..mbh).step_by(txhu) {
                for blk_col in (0..mbw).step_by(txwu) {
                    map0[blk_row * map_stride + blk_col] =
                        allowed[(rng.next() as usize) % allowed.len()] as u8;
                }
            }

            let above_ctx: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let left_ctx: Vec<i8> = (0..32)
                .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                .collect();
            let rdmult = rng.range(1, 1 << 22);

            // ---- Rust side ----
            let env = EncodeIntraYEnv {
                sb_size: 12,
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
                mode,
                angle_delta,
                use_filter_intra: use_fi,
                filter_intra_mode: fi_mode,
                tx_size,
                skip_txfm,
                lossless: false,
                reduced_tx_set_used: reduced,
                bd,
                rows: &rows,
                rdmult,
                sharpness,
                coeff_costs: &coeff_costs,
                enable_optimize_b: opt_type,
                dry_run_output_enabled: dry_out,
                above_ctx: &above_ctx,
                left_ctx: &left_ctx,
                qm_level: None,
                palette: None,
                tune: Default::default(),
            };
            let mut recon_rust = recon0.clone();
            let mut map_rust = map0.clone();
            let mut cfl_rust = CflCtx::new(ss_x, ss_y);
            let out = encode_intra_block_plane_y(
                &env,
                &mut recon_rust,
                &mut map_rust,
                if store_y { Some(&mut cfl_rust) } else { None },
            );

            // ---- C side: the walk over REAL pieces (common) ----
            let mut recon_c = recon0.clone();
            let mut map_c = map0.clone();
            let mut cfl_c = c::RefCflState::default();
            let ca = CEncPlaneArgs {
                partition: 0,
                bsize,
                tx_size,
                geometry: (mi_row, mi_col, ref_off, src_off, STRIDE),
                sb_size: 12,
                src: &src,
                mode,
                angle_delta,
                use_fi,
                fi_mode,
                // Frozen 0: this plane-level fixture drives the port env with
                // filter_type: 0 (above), so the C side matches at 0 too.
                filter_type: 0,
                skip_txfm,
                use_trellis,
                load_ctx: opt_type != TrellisOptType::NoTrellisOpt,
                sharpness,
                reduced,
                bd,
                plane_rows_c,
                dequant,
                above_ctx: &above_ctx,
                left_ctx: &left_ctx,
                rdmult,
                coeff_tbls: (
                    &txb_skip, &base_eob, &base, &eob_extra, &dc_sign, &lps, &eob_tbl,
                ),
                store: store_y,
                ss: (ss_x, ss_y),
            };
            let (txbs_c, ta_c, tl_c) =
                c_encode_intra_block_plane_y(&ca, &mut recon_c, &mut map_c, &mut cfl_c);
            for cc in txbs_c.iter() {
                if cc.1 == 0 {
                    eob0_resets += 1;
                    if txwu == 16 || txhu == 16 {
                        fill64 += 1;
                    }
                } else {
                    eob_pos += 1;
                }
                if store_y {
                    stores += 1;
                    if (bw == 4 || bh == 4) && (mi_row % 2 == 1 || mi_col % 2 == 1) {
                        sub8_stores += 1;
                    }
                }
            }
            if skip_txfm {
                skip_arm += 1;
            }
            if !use_trellis {
                no_trellis_arm += 1;
            }

            // ---- compare ----
            let tag = format!(
                "pair {pi} (bs {bsize} tx {tx_size}) iter {iter} bd {bd} q {qindex} mode {mode} \
                 delta {angle_delta} fi {use_fi} reduced {reduced} skip {skip_txfm} \
                 opt {opt_type:?} dry_out {dry_out} store {store_y} flat {flat}"
            );
            assert_eq!(out.txbs.len(), n_txbs, "txb count: {tag}");
            assert_eq!(out.txbs.len(), txbs_c.len(), "txb count vs C: {tag}");
            for (k, (r, cc)) in out.txbs.iter().zip(txbs_c.iter()).enumerate() {
                assert_eq!(r.tx_type, cc.0, "txb {k} tx_type: {tag}");
                assert_eq!(r.eob, cc.1, "txb {k} eob: {tag}");
                assert_eq!(r.txb_entropy_ctx, cc.2, "txb {k} entropy ctx: {tag}");
                assert_eq!(r.qcoeff, cc.3, "txb {k} qcoeff: {tag}");
                assert_eq!(r.dqcoeff, cc.4, "txb {k} dqcoeff: {tag}");
            }
            assert_eq!(recon_rust, recon_c, "final recon planes: {tag}");
            assert_eq!(map_rust, map_c, "final tx_type_map: {tag}");
            assert_eq!(out.ta, ta_c, "final ta: {tag}");
            assert_eq!(out.tl, tl_c, "final tl: {tag}");
            if store_y {
                assert_eq!(
                    &cfl_rust.recon_buf_q3[..],
                    &cfl_c.recon_q3[..CFL_BUF_SQUARE],
                    "CfL recon_buf_q3: {tag}"
                );
                assert_eq!(cfl_rust.buf_width, cfl_c.buf_w, "CfL buf_width: {tag}");
                assert_eq!(cfl_rust.buf_height, cfl_c.buf_h, "CfL buf_height: {tag}");
                assert_eq!(
                    cfl_rust.are_parameters_computed, cfl_c.params_computed,
                    "CfL params_computed: {tag}"
                );
            }
        }
    }

    // Coverage floors.
    assert!(
        eob0_resets >= 20,
        "eob-0 DCT resets exercised: {eob0_resets}"
    );
    assert!(eob_pos >= 100, "eob>0 txbs exercised: {eob_pos}");
    assert!(
        fill64 >= 1,
        "TX_64X64 16x16-unit map fill exercised: {fill64}"
    );
    assert!(stores >= 40, "CfL stores exercised: {stores}");
    assert!(
        sub8_stores >= 4,
        "sub-8x8 CfL stores exercised: {sub8_stores}"
    );
    assert!(skip_arm >= 10, "skip_txfm arm exercised: {skip_arm}");
    assert!(
        no_trellis_arm >= 20,
        "no-trellis (B) arm exercised: {no_trellis_arm}"
    );
}
