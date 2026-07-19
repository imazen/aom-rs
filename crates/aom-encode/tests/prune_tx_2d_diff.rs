//! Differential for `prune_tx_2d` (tx_search.c:1541, the inter var-tx 2D tx-type
//! NN prune) vs the REAL-C `shim_prune_tx_2D` tier-1 oracle (the static helpers +
//! driver copied verbatim, calling the exported av1_nn_predict /
//! av1_nn_fast_softmax_16 / av1_get_horver_correlation_full + the real non-V2
//! nnconfig maps). Over every tx size that has a nnconfig, its determined inter
//! ext-tx set (ALL16 / DTT9_IDTX_1DDCT), and residual amplitudes spanning
//! near-flat -> saturating, comparing the pruned mask + the reordered txk_map
//! (incl. the force-keep-argmax early-return and both sort-network arms).

use aom_encode::prune_tx_2d::prune_tx_2d;
use aom_sys_ref as c;
use aom_dsp::txb::ext_tx_set_type;

const TXS_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TXS_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
/// `av1_ext_tx_used_flag[TxSetType]` (blockd.h): DTT9_IDTX_1DDCT(4)=0x0FFF, ALL16(5)=0xFFFF.
const EXT_TX_USED_FLAG: [u16; 6] = [0x0001, 0x0201, 0x020F, 0x0E0F, 0x0FFF, 0xFFFF];

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
fn prune_tx_2d_matches_c() {
    c::ref_init();
    let mut rng = Rng(0x9e37_2026_0719_00a5);
    // The 9 tx sizes with a non-V2 hor/ver nnconfig (all others early-return).
    let tx_sizes = [0usize, 1, 2, 5, 6, 7, 8, 13, 14];
    let mut fired = 0usize;
    let mut all16 = 0usize;
    let mut dtt9 = 0usize;
    let mut nontrivial = 0usize; // mask actually pruned something

    for &tx_size in &tx_sizes {
        let (bw, bh) = (TXS_W[tx_size], TXS_H[tx_size]);
        let set = ext_tx_set_type(tx_size, true, false);
        assert!(set == 4 || set == 5, "tx_size {tx_size} unexpected set {set}");
        let full_mask = EXT_TX_USED_FLAG[set];

        for iter in 0..64 {
            let amp = [2, 12, 60, 300, 1200][iter % 5];
            let residual: Vec<i16> = (0..bw * bh).map(|_| rng.range(-amp, amp + 1) as i16).collect();
            // in_mask: the full set, or a random subset (still >= a few bits).
            let in_mask = if iter % 3 == 0 {
                full_mask
            } else {
                let drop = (rng.next() as u16) & full_mask;
                let m = full_mask & !drop;
                if m.count_ones() >= 2 { m } else { full_mask }
            };

            let port = prune_tx_2d(&residual, bw, tx_size, set, 1, in_mask)
                .unwrap_or_else(|| panic!("port None for tx_size={tx_size} set={set}"));
            let (c_mask, c_txk) = c::ref_prune_tx_2d(&residual, bw, tx_size, set, 1, in_mask);

            assert_eq!(
                port.allowed_tx_mask, c_mask,
                "mask tx_size={tx_size} iter={iter} set={set} in={in_mask:#06x}"
            );
            let port_txk: [i32; 16] = core::array::from_fn(|i| port.txk_map[i] as i32);
            if port_txk != c_txk {
                eprintln!("MISMATCH tx_size={tx_size} iter={iter} set={set} mask={c_mask:#06x}");
                eprintln!("  port_txk={port_txk:?}");
                eprintln!("  c_txk   ={c_txk:?}");
                panic!("txk_map tx_size={tx_size} iter={iter}");
            }

            fired += 1;
            if set == 5 { all16 += 1 } else { dtt9 += 1 }
            if c_mask.count_ones() < in_mask.count_ones() {
                nontrivial += 1;
            }
        }
    }

    eprintln!("prune_tx_2d_diff: fired={fired} all16={all16} dtt9={dtt9} nontrivial={nontrivial}");
    assert!(fired > 500, "fired: {fired}");
    assert!(all16 > 100, "ALL16 sizes: {all16}");
    // Only TX_16X16 (tx_size_sqr == 16x16) uses DTT9 among the nnconfig sizes.
    assert!(dtt9 >= 60, "DTT9 sizes: {dtt9}");
    // The prune actually shrinks the incoming mask (real work, not a pass-through).
    assert!(nontrivial > 20, "cases where the prune shrank the mask: {nontrivial}");
}
