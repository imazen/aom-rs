//! Differential harness for `txb_rd_cost` — the coefficient-level per-txb intra
//! RD cost (av1/encoder/tx_search.c composition). Validates that composing the
//! individually-bit-exact pieces (cost_coeffs_txb for rate, dist_block_tx_domain
//! for distortion, RDCOST to combine) is wired identically to the C chain:
//! `RDCOST(rdmult, ref_cost_coeffs_txb(qcoeff, …), ref_dist_block_tx_domain(coeff, dqcoeff, …))`.
//!
//! The rate is coefficient-coding bits only; block-level mode/tx signaling is out
//! of scope on both sides (as in cost_coeffs_diff.rs).

use aom_encode::txb_rd_cost;
use aom_sys_ref as c;
use aom_dsp::txb::{CoeffCostTables, scan, txb_high, txb_wide};

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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
    /// A plausible cost-table entry: a small non-negative bit cost in the
    /// `1<<9`-per-bit domain.
    fn cost(&mut self) -> i32 {
        self.range(0, 16 << 9) as i32
    }
    fn coeff_val(&mut self, bound: i32) -> i32 {
        (self.next() % (2 * bound as u64 + 1)) as i32 - bound
    }
}

fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

/// Scan-consistent sparse quantized coefficients (matches cost_coeffs_diff.rs):
/// `scan[eob-1]` nonzero, positions `>= eob` zero.
fn gen_qcoeffs(rng: &mut Rng, sc: &[i16], area: usize) -> (Vec<i32>, usize) {
    let mut q = vec![0i32; area];
    let eob = rng.range(1, area as u32 + 1) as usize;
    let nz = |rng: &mut Rng| -> i32 {
        let mag = match rng.range(0, 10) {
            0..=4 => rng.range(1, 3) as i32,
            5..=7 => rng.range(1, 20) as i32,
            _ => rng.range(1, 3000) as i32,
        };
        if rng.next() & 1 == 1 { -mag } else { mag }
    };
    #[allow(clippy::needless_range_loop)]
    for i in 0..eob {
        let pos = sc[i] as usize;
        if i == eob - 1 || rng.range(0, 10) >= 4 {
            q[pos] = nz(rng);
        }
    }
    (q, eob)
}

#[test]
fn txb_rd_cost_matches_c_chain() {
    let mut rng = Rng(0x00d1_c057_9e37_1111);
    const TX_TYPES: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];

    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &tx_type in &TX_TYPES {
            for &bd in &[8u8, 12] {
                let bound = if bd > 8 { 32768 } else { 16384 };
                for _ in 0..60 {
                    // Shared random cost tables.
                    let txb_skip = tbl(&mut rng, 13 * 2);
                    let base_eob = tbl(&mut rng, 4 * 3);
                    let base = tbl(&mut rng, 42 * 8);
                    let eob_extra = tbl(&mut rng, 9 * 2);
                    let dc_sign = tbl(&mut rng, 3 * 2);
                    let lps = tbl(&mut rng, 21 * 26);
                    let eob_c = tbl(&mut rng, 2 * 11);

                    let sc = scan(tx_size, tx_type);
                    // qcoeff (quantized) drives the rate; coeff/dqcoeff drive the
                    // transform-domain distortion (each piece is validated on
                    // arbitrary inputs, so independent draws exercise the wiring).
                    let (qcoeff, eob) = gen_qcoeffs(&mut rng, sc, area);
                    let coeff: Vec<i32> = (0..area).map(|_| rng.coeff_val(bound)).collect();
                    let dqcoeff: Vec<i32> = (0..area).map(|_| rng.coeff_val(bound)).collect();
                    let txb_skip_ctx = rng.range(0, 13) as usize;
                    let dc_sign_ctx = rng.range(0, 3) as usize;
                    let rdmult = rng.range(1, i32::MAX as u32) as i32;

                    // C chain: rate + tx-domain dist -> RDCOST.
                    let rate_c = c::ref_cost_coeffs_txb(
                        &qcoeff,
                        eob,
                        tx_size,
                        tx_type,
                        txb_skip_ctx,
                        dc_sign_ctx,
                        &txb_skip,
                        &base_eob,
                        &base,
                        &eob_extra,
                        &dc_sign,
                        &lps,
                        &eob_c,
                    );
                    let (dist_c, _sse_c) =
                        c::ref_dist_block_tx_domain(&coeff, &dqcoeff, tx_size, bd);
                    let want = c::ref_rdcost(rdmult, rate_c, dist_c);

                    let tables = CoeffCostTables {
                        txb_skip: &txb_skip,
                        base_eob: &base_eob,
                        base: &base,
                        eob_extra: &eob_extra,
                        dc_sign: &dc_sign,
                        lps: &lps,
                        eob: &eob_c,
                    };
                    let got = txb_rd_cost(
                        &coeff,
                        &qcoeff,
                        &dqcoeff,
                        eob,
                        tx_size,
                        tx_type,
                        txb_skip_ctx,
                        dc_sign_ctx,
                        &tables,
                        rdmult,
                        bd,
                    );

                    assert_eq!(
                        got, want,
                        "txb_rd tx_size={tx_size} tx_type={tx_type} bd={bd} eob={eob} rdmult={rdmult}"
                    );
                }
            }
        }
    }
}
