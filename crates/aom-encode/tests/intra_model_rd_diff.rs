//! Differential: `intra_model_rd_y` vs the C chain of REAL reference pieces —
//! per model-txb `ref_intra_avail` + `ref_hbd_predict_intra` (prediction
//! written into the C-side recon plane, as `av1_predict_intra_block_facade`
//! writes `pd->dst` in place) -> `ref_highbd_subtract_block` ->
//! `ref_hadamard` / `ref_highbd_hadamard` (the `av1_quick_txfm use_hadamard=1`
//! dispatch) -> `ref_satd`, accumulated. Asserts the model cost AND the
//! post-walk recon planes (the prediction side effects are caller-visible
//! state for the mode loop).

use aom_encode::mode_costs::TxSizeCosts;
use aom_encode::tx_search::{TxTypeSearchPolicy, TxfmYrdEnv, intra_model_rd_y};
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_txb::{CoeffCostTables, TxTypeCosts};

mod common;
use common::*;




#[test]
fn intra_model_rd_matches_c_chain() {
    c::ref_init();
    let mut rng = Rng(0x1a0d_e1bd_2026_0714);
    const STRIDE: usize = 256;
    // bsize -> model tx = min(TX_32X32, max_txsize_lookup): covers 4x4 (0),
    // 8x8 (1), 16x16 (2), 32x32 (3) models, square + rect blocks, multi-txb
    // walks (64x64 -> 4 32x32 txbs; 16x8 -> 2 8x8 txbs).
    let bsizes = [0usize, 3, 4, 5, 6, 9, 12, 19];
    let mut multi_txb = 0usize;
    let mut nonzero_cost = 0usize;
    let mut recon_mutated = 0usize;

    // Unused-by-model quantizer/cost plumbing to fill TxfmYrdEnv.
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(8, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
    let rows = set_q_index(&quants, &deq, 64, 0);
    let zero_tbl = vec![0i32; 21 * 26];
    let coeff_costs = CoeffCostTables {
        txb_skip: &zero_tbl[..13 * 2],
        base_eob: &zero_tbl[..4 * 3],
        base: &zero_tbl[..42 * 8],
        eob_extra: &zero_tbl[..9 * 2],
        dc_sign: &zero_tbl[..3 * 2],
        lps: &zero_tbl[..21 * 26],
        eob: &zero_tbl[..2 * 11],
    };
    let tx_type_costs = TxTypeCosts::zeroed();
    let tx_size_costs = TxSizeCosts::zeroed();
    let skip_costs = [[0i32; 2]; 3];
    let above_ctx = vec![0i8; 32];
    let left_ctx = vec![0i8; 32];
    let _pol = TxTypeSearchPolicy::speed0_allintra();

    for (bi, &bsize) in bsizes.iter().enumerate() {
        let (bw, bh) = (BLK_W[bsize], BLK_H[bsize]);
        let model_tx = MAX_TXSIZE_LOOKUP[bsize].min(3);
        let n_txbs = (bw / TX_W[model_tx]) * (bh / TX_H[model_tx]);
        for iter in 0..14 {
            let bd: u8 = match iter % 3 {
                0 => 8,
                1 => 10,
                _ => 12,
            };
            let amp: i32 = match iter % 4 {
                0 => {
                    if bd > 8 {
                        4095
                    } else {
                        255
                    }
                }
                1 => 24,
                2 => 2,
                _ => 96,
            };
            let mode = (rng.next() % 13) as usize;
            let angle_delta = if (1..=8).contains(&mode) {
                rng.range(-3, 4)
            } else {
                0
            };
            let (mi_row, mi_col) = (8, 8);
            let ref_off = 32 * STRIDE + 32;
            let src_off = ref_off;

            let recon0: Vec<u16> = (0..STRIDE * 96)
                .map(|_| (rng.next() % (1u64 << bd)) as u16)
                .collect();
            let mut src = recon0.clone();
            for r in 0..bh {
                for cx in 0..bw {
                    let idx = src_off + r * STRIDE + cx;
                    let v = i64::from(recon0[idx]) + i64::from(rng.range(-amp, amp + 1));
                    src[idx] = v.clamp(0, (1 << bd) - 1) as u16;
                }
            }

            let env = TxfmYrdEnv {
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
                use_filter_intra: false,
                filter_intra_mode: 0,
                lossless: false,
                reduced_tx_set_used: false,
                bd,
                rows: &rows,
                rdmult: 1,
                coeff_costs: &coeff_costs,
                tx_type_costs: &tx_type_costs,
                skip_costs: &skip_costs,
                skip_ctx: 0,
                tx_size_costs: &tx_size_costs,
                tx_size_ctx: 0,
                tx_mode_is_select: true,
                above_ctx: &above_ctx,
                left_ctx: &left_ctx,
            };

            let mut recon_rust = recon0.clone();
            let got = intra_model_rd_y(&env, &mut recon_rust, model_tx);

            let mut recon_c = recon0.clone();
            let want = c_intra_model_rd(
                bsize,
                model_tx,
                &mut recon_c,
                &src,
                (mi_row, mi_col, ref_off, src_off, STRIDE),
                mode,
                angle_delta,
                false,
                0,
                bd,
            );

            let m = format!(
                "bi={bi} bsize={bsize} model_tx={model_tx} n_txbs={n_txbs} iter={iter} \
                 bd={bd} amp={amp} mode={mode}/{angle_delta}",
            );
            assert_eq!(got, want, "model rd {m}");
            assert_eq!(recon_rust, recon_c, "recon plane {m}");
            if n_txbs > 1 {
                multi_txb += 1;
            }
            if got != 0 {
                nonzero_cost += 1;
            }
            if recon_rust != recon0 {
                recon_mutated += 1;
            }
        }
    }
    assert!(multi_txb > 30, "multi-txb model walks: {multi_txb}");
    assert!(nonzero_cost > 90, "nonzero model costs: {nonzero_cost}");
    assert!(
        recon_mutated > 100,
        "prediction writes unexercised: {recon_mutated}"
    );
}
