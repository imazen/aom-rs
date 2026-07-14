//! Differential harness for `search_tx_type`'s building blocks:
//! - `get_tx_mask` luma-intra arm (`get_tx_mask_intra`) vs the C transcription
//!   over the REAL `av1_get_ext_tx_set_type` + REAL blockd.h tables;
//! - `av1_pixel_diff_dist` vs the REAL EXPORTED C function (marshalled
//!   MACROBLOCK), including frame-edge-clipped visible dimensions.

use aom_encode::tx_search::{
    TX_TYPES, TxMaskParams, av1_pixel_diff_dist, get_tx_mask_intra, get_txb_visible_dimensions,
};
use aom_sys_ref as c;

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
}

#[test]
fn tx_mask_intra_matches_c() {
    // Exhaustive over the discrete axes: 19 tx sizes x 13 modes x fi (5 modes
    // + off) x lossless x reduced set x sf 0/1/2 x derived x flip_idtx x
    // dct_only = every branch of the intra arm.
    let mut multi = 0usize;
    let mut single = 0usize;
    let mut reduced_hits = 0usize;
    for tx_size in 0..19usize {
        for mode in 0..13usize {
            for fi in 0..6usize {
                let (use_fi, fi_mode) = if fi == 5 { (false, 0) } else { (true, fi) };
                for cfg in 0..48usize {
                    let lossless = cfg & 1 != 0;
                    let reduced = cfg & 2 != 0;
                    let use_reduced_txset = (cfg >> 2) % 3; // 0/1/2
                    let derived = (cfg >> 4) & 1 != 0;
                    let flip_idtx = cfg & 32 == 0; // mostly on
                    let p = TxMaskParams {
                        use_reduced_intra_txset: use_reduced_txset as u8,
                        use_derived_intra_tx_type_set: derived,
                        enable_flip_idtx: flip_idtx,
                        use_intra_dct_only: false,
                    };
                    let (mask, txk) =
                        get_tx_mask_intra(tx_size, mode, use_fi, fi_mode, lossless, reduced, &p);
                    let (mask_c, txk_c) = c::ref_get_tx_mask_intra(
                        tx_size as i32,
                        mode as i32,
                        use_fi,
                        fi_mode as i32,
                        lossless,
                        reduced,
                        use_reduced_txset as i32,
                        derived,
                        flip_idtx,
                        false,
                    );
                    let txk_rust = txk.unwrap_or(TX_TYPES) as i32;
                    assert_eq!(
                        (mask, txk_rust),
                        (mask_c, txk_c),
                        "ts={tx_size} mode={mode} fi={use_fi}/{fi_mode} cfg={cfg}",
                    );
                    if txk.is_none() {
                        multi += 1;
                    } else {
                        single += 1;
                    }
                    if use_reduced_txset > 0 && mask != 0 {
                        reduced_hits += 1;
                    }
                }
            }
        }
    }
    // use_intra_dct_only arm.
    let p = TxMaskParams {
        use_intra_dct_only: true,
        ..TxMaskParams::speed0_allintra()
    };
    let (mask, txk) = get_tx_mask_intra(2, 4, false, 0, false, false, &p);
    let (mask_c, txk_c) =
        c::ref_get_tx_mask_intra(2, 4, false, 0, false, false, 1, false, true, true);
    assert_eq!((mask, txk.unwrap_or(TX_TYPES) as i32), (mask_c, txk_c));
    assert_eq!(mask, 1);
    // Non-vacuity: both single-type and multi-type outcomes heavily exercised.
    assert!(
        multi > 10_000 && single > 10_000,
        "multi={multi} single={single}"
    );
    assert!(reduced_hits > 10_000);
}

#[test]
fn pixel_diff_dist_matches_real_c() {
    c::ref_init();
    let mut rng = Rng(0xd1f_fd15_7000_0001);
    // (plane_bsize, tx_bsize, bw, bh, txw, txh) triples: tx <= plane block.
    // BLOCK_8X8=3(8x8), BLOCK_16X16=6, BLOCK_32X32=9, BLOCK_16X8=5, BLOCK_8X16=4.
    let cases: [(i32, i32, usize, usize, usize, usize); 6] = [
        (3, 3, 8, 8, 8, 8),
        (6, 3, 16, 16, 8, 8),
        (6, 6, 16, 16, 16, 16),
        (9, 6, 32, 32, 16, 16),
        (5, 5, 16, 8, 16, 8),
        (4, 4, 8, 16, 8, 16),
    ];
    let mut clipped = 0usize;
    for case in 0..600 {
        let (pb, tb, bw, bh, txw, txh) = cases[case % cases.len()];
        let diff: Vec<i16> = (0..bw * bh)
            .map(|_| rng.range(-4095, 4096) as i16)
            .collect();
        // blk offsets in MI units, on the txb grid (multiples of the tx unit,
        // as the real foreach-txb walk produces).
        let (txwu, txhu) = (txw >> 2, txh >> 2);
        let n_r = (bh - txh) / txh + 1;
        let n_c = (bw - txw) / txw + 1;
        let blk_row = ((rng.next() as usize) % n_r) * txhu;
        let blk_col = ((rng.next() as usize) % n_c) * txwu;
        // Frame edges: interior (>= 0) or overhanging (negative, 1/8-pel
        // units: -8 per overhanging pixel). Keep >= 1 visible row and column
        // at the sampled txb — the real foreach-txb walk (max_block_wide/high
        // clip) never visits fully-clipped txbs, and the C SSE2 kernel's
        // do-while is undefined-ish for 0-height requests.
        let max_right_cut = (bw - (blk_col << 2) - 1) as i32;
        let max_bottom_cut = (bh - (blk_row << 2) - 1) as i32;
        let (right, bottom) = match case % 3 {
            0 => (0, 0),
            1 => (-(rng.range(1, max_right_cut + 1) * 8), 0),
            _ => (
                -(rng.range(1, max_right_cut + 1) * 8),
                -(rng.range(1, max_bottom_cut + 1) * 8),
            ),
        };
        let (vis_w, vis_h) =
            get_txb_visible_dimensions(bw, bh, txw, txh, blk_row, blk_col, right, bottom, 0, 0);
        let (sse, mse) = av1_pixel_diff_dist(&diff, bw, blk_row, blk_col, vis_w, vis_h);
        let (sse_c, mse_c) = c::ref_pixel_diff_dist(
            &diff,
            pb,
            tb,
            blk_row as i32,
            blk_col as i32,
            right,
            bottom,
            0,
            0,
        );
        assert_eq!(
            (sse as i64, mse),
            (sse_c, mse_c),
            "case={case} pb={pb} tb={tb} blk=({blk_row},{blk_col}) edges=({right},{bottom}) vis=({vis_w},{vis_h})",
        );
        if vis_w != txw || vis_h != txh {
            clipped += 1;
        }
    }
    assert!(clipped > 100, "edge clipping under-exercised: {clipped}");
}
