//! Differential harness for `dist_block_tx_domain` (non-QM path,
//! av1/encoder/tx_search.c) vs C: transform-domain `(dist, sse)` for one txb,
//! i.e. `block_error` followed by the per-tx-size normalization shift
//! `(MAX_TX_SCALE - av1_get_tx_scale(tx_size)) * 2`. Sweeps every tx_size so all
//! three shift magnitudes (right by 2, none, left by 2 for 64-wide) are hit.

use aom_encode::dist_block_tx_domain;
use aom_sys_ref as c;
use aom_dsp::txb::{txb_high, txb_wide};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % ((hi - lo) as u64)) as i32
    }
}

#[test]
fn dist_block_tx_domain_matches_c() {
    let mut rng = Rng(0x00d1_5700_9e37_1111);
    for tx_size in 0..19usize {
        let n = txb_wide(tx_size) * txb_high(tx_size);
        for &bd in &[8u8, 12] {
            // Keep coeff magnitudes in the regime real transform coefficients
            // occupy so the lowbd block_error 32-bit products stay in range
            // (|diff|, |coeff| < 2^15 => diff^2, coeff^2 < 2^31).
            let bound = if bd > 8 { 32768 } else { 16384 };
            for _ in 0..3000 {
                let coeff: Vec<i32> = (0..n).map(|_| rng.range_i32(-bound, bound)).collect();
                let dqcoeff: Vec<i32> = (0..n).map(|_| rng.range_i32(-bound, bound)).collect();
                let got = dist_block_tx_domain(&coeff, &dqcoeff, tx_size, bd);
                let want = c::ref_dist_block_tx_domain(&coeff, &dqcoeff, tx_size, bd);
                assert_eq!(got, want, "rand tx_size={tx_size} bd={bd}");
            }
            // Edge cases: all-zero (dist=sse=0), identical (dist=0), and a
            // saturated +/- split (max energy, exercises the left shift on 64-wide).
            let zeros = vec![0i32; n];
            assert_eq!(
                dist_block_tx_domain(&zeros, &zeros, tx_size, bd),
                c::ref_dist_block_tx_domain(&zeros, &zeros, tx_size, bd),
                "zero tx_size={tx_size} bd={bd}"
            );
            let pos = vec![bound - 1; n];
            let neg = vec![-(bound - 1); n];
            assert_eq!(
                dist_block_tx_domain(&pos, &neg, tx_size, bd),
                c::ref_dist_block_tx_domain(&pos, &neg, tx_size, bd),
                "sat tx_size={tx_size} bd={bd}"
            );
        }
    }
}
