//! Differential harness for `av1_cost_coeffs_txb` (`warehouse_efficients_txb`):
//! the Rust cost function must return the identical integer RD rate as C libaom
//! over identical cost tables and coefficient blocks.
//!
//! Cost tables (`LV_MAP_COEFF_COST` / `LV_MAP_EOB_COST`) are random-but-shared,
//! isolating the cost-summation logic from the separate CDF→cost derivation.
//! `get_tx_type_cost` (plane-0 tx_type) is out of scope on both sides.

use aom_sys_ref as c;
use aom_txb::{cost_coeffs_txb, scan, txb_high, txb_wide, CoeffCostTables};

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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
    /// A plausible cost-table entry: a small non-negative "bit cost" in the
    /// `1<<9`-per-bit domain (0..~16 bits), matching real av1 cost magnitudes.
    fn cost(&mut self) -> i32 {
        self.range(0, 16 << 9) as i32
    }
}

fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

/// Scan-consistent sparse coefficients: `scan[eob-1]` nonzero, positions
/// `>= eob` zero. Magnitudes span base / base-range / golomb.
fn gen_coeffs(rng: &mut Rng, sc: &[i16], area: usize) -> (Vec<i32>, usize) {
    let mut coeff = vec![0i32; area];
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
        // eob position is always nonzero; ~60% of interior positions too.
        if i == eob - 1 || rng.range(0, 10) >= 4 {
            coeff[pos] = nz(rng);
        }
    }
    (coeff, eob)
}

#[test]
fn cost_coeffs_txb_identical() {
    let mut rng = Rng(0x_c057_0000_dead_beef);
    const TX_TYPES: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];

    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &tx_type in &TX_TYPES {
            for _ in 0..120 {
                // Fresh random cost tables (shared by both sides).
                let txb_skip = tbl(&mut rng, 13 * 2);
                let base_eob = tbl(&mut rng, 4 * 3);
                let base = tbl(&mut rng, 42 * 8);
                let eob_extra = tbl(&mut rng, 9 * 2);
                let dc_sign = tbl(&mut rng, 3 * 2);
                let lps = tbl(&mut rng, 21 * 26);
                let eob_c = tbl(&mut rng, 2 * 11);

                let sc = scan(tx_size, tx_type);
                let (coeff, eob) = gen_coeffs(&mut rng, sc, area);
                let txb_skip_ctx = rng.range(0, 13) as usize;
                let dc_sign_ctx = rng.range(0, 3) as usize;

                let want = c::ref_cost_coeffs_txb(
                    &coeff, eob, tx_size, tx_type, txb_skip_ctx, dc_sign_ctx, &txb_skip,
                    &base_eob, &base, &eob_extra, &dc_sign, &lps, &eob_c,
                );

                let tables = CoeffCostTables {
                    txb_skip: &txb_skip,
                    base_eob: &base_eob,
                    base: &base,
                    eob_extra: &eob_extra,
                    dc_sign: &dc_sign,
                    lps: &lps,
                    eob: &eob_c,
                };
                let got =
                    cost_coeffs_txb(&coeff, eob, tx_size, tx_type, txb_skip_ctx, dc_sign_ctx, &tables);

                assert_eq!(
                    got, want,
                    "cost tx_size={tx_size} tx_type={tx_type} eob={eob} \
                     skip_ctx={txb_skip_ctx} dc_ctx={dc_sign_ctx}"
                );
            }
        }
    }
}
