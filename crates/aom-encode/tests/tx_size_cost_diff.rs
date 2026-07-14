//! Differential harness for the tx-size signaling cost: `fill_tx_size_costs`
//! (the rd.c `av1_fill_mode_rates` tx-size slice, oracle = transcription over
//! the REAL exported `av1_cost_tokens_from_cdf`) and `tx_size_cost`
//! (tx_search.h, oracle = transcription over the REAL `bsize_to_tx_size_cat`
//! / `tx_size_to_depth` / `block_signals_txsize` header statics;
//! `get_tx_size_context` is the caller's, deferred on both sides).

use aom_encode::mode_costs::{TxSizeCosts, fill_tx_size_costs, tx_size_cost};
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

/// Valid `nsymbs`-symbol inverse-CDF row padded to 4 entries.
fn cdf_row(rng: &mut Rng, nsymbs: usize) -> [u16; 4] {
    let mut row = [0u16; 4];
    let mut acc: u32 = 0;
    for e in row.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, 32000 / nsymbs as i32) as u32;
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    row[nsymbs - 1] = 0;
    row
}

/// `max_txsize_rect_lookup[bsize]` (common_data.h) — the depth-0 tx size the
/// depth sweep starts from, for generating valid (bsize, tx_size) pairs.
const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] = [
    0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18,
];
/// `sub_tx_size_map[TX_SIZES_ALL]` (common_data.h): rect sizes halve the
/// LONG side (4x16 -> 4x8, 8x32 -> 8x16, ...), squares halve both.
const SUB_TX_SIZE_MAP: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];

#[test]
fn tx_size_costs_match_c() {
    let mut rng = Rng(0x7c51_2ec0_5757_0001);
    let mut nonzero_gate = 0usize;
    let mut zero_gate = 0usize;
    for case in 0..2000 {
        // Category symbol counts: cat 0 -> 2 symbols, cats 1..3 -> 3.
        let mut cdf = Vec::with_capacity(4 * 3 * 4);
        for cat in 0..4 {
            let ns = if cat == 0 { 2 } else { 3 };
            for _ in 0..3 {
                cdf.extend_from_slice(&cdf_row(&mut rng, ns));
            }
        }
        let cref = c::ref_fill_tx_size_costs(&cdf);
        let mut costs = TxSizeCosts::zeroed();
        fill_tx_size_costs(&mut costs, &cdf);
        for cat in 0..4 {
            for ctx in 0..3 {
                let ns = if cat == 0 { 2 } else { 3 };
                for d in 0..ns {
                    assert_eq!(
                        costs.0[cat][ctx][d],
                        cref[(cat * 3 + ctx) * 3 + d],
                        "fill case={case} cat={cat} ctx={ctx} d={d}",
                    );
                }
            }
        }

        // tx_size_cost over every signaling bsize x its depth chain x ctx.
        let flat: Vec<i32> = costs.0.iter().flatten().flatten().copied().collect();
        #[allow(clippy::needless_range_loop)] // bsize IS the domain being swept
        for bsize in 0..22usize {
            let select = case % 4 != 3;
            let ctx = (rng.next() % 3) as i32;
            let mut tx = MAX_TXSIZE_RECT_LOOKUP[bsize];
            for _depth in 0..=2 {
                let rust = tx_size_cost(&costs, select, bsize, tx, ctx as usize);
                let cval = c::ref_tx_size_cost(&flat, select, bsize as i32, tx as i32, ctx);
                assert_eq!(
                    rust, cval,
                    "cost case={case} bsize={bsize} tx={tx} ctx={ctx} select={select}"
                );
                if rust != 0 {
                    nonzero_gate += 1;
                } else {
                    zero_gate += 1;
                }
                if tx == 0 {
                    break;
                }
                tx = SUB_TX_SIZE_MAP[tx];
            }
        }
    }
    assert!(
        nonzero_gate > 20_000 && zero_gate > 5_000,
        "{nonzero_gate}/{zero_gate}"
    );
}
