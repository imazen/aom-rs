//! Differential harness for the trellis per-coefficient cost helpers
//! (`get_two_coeff_cost_simple`, `get_coeff_cost_eob`, `get_coeff_cost_general`)
//! vs C libaom. Random cost tables + levels buffers + parameters; every returned
//! cost (and the trellis `cost_low`) must be integer-identical.

use aom_sys_ref as c;

use aom_txb::{coeff_cost_eob, coeff_cost_general, two_coeff_cost_simple, txb_bhl, txb_wide, CoeffCostTables, TxClass};

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
    fn cost(&mut self) -> i32 {
        self.range(0, 20 << 9) as i32
    }
}

const TX_PAD_2D: usize = (32 + 4) * (32 + 4) + 16;

fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

fn class_of(v: u32) -> (TxClass, i32) {
    match v {
        1 => (TxClass::Horiz, 1),
        2 => (TxClass::Vert, 2),
        _ => (TxClass::TwoD, 0),
    }
}

#[test]
fn trellis_cost_helpers_identical() {
    let mut rng = Rng(0x_7e11_0000_c057_1234);
    // Square tx sizes give bhl/width; 3 (TX_32X32) is the max levels footprint.
    for &tx_size in &[0usize, 1, 2, 3, 5, 8, 13] {
        let bhl = txb_bhl(tx_size);
        let width = txb_wide(tx_size);
        let area = width << bhl;
        for _ in 0..300 {
            // Random cost tables (base with valid [4..7] diff slots = arbitrary).
            let base_eob = tbl(&mut rng, 4 * 3);
            let base = tbl(&mut rng, 42 * 8);
            let dc_sign = tbl(&mut rng, 3 * 2);
            let lps = tbl(&mut rng, 21 * 26);
            // Random padded levels buffer (bytes read by get_br_ctx).
            let levels: Vec<u8> = (0..TX_PAD_2D).map(|_| rng.range(0, 16) as u8).collect();
            let t = CoeffCostTables {
                txb_skip: &[],
                base_eob: &base_eob,
                base: &base,
                eob_extra: &[],
                dc_sign: &dc_sign,
                lps: &lps,
                eob: &[],
            };

            let (tx_class, tc) = class_of(rng.range(0, 3));
            let abs_qc = match rng.range(0, 4) {
                0 => rng.range(1, 3) as i32,
                1 => rng.range(1, 20) as i32,
                2 => rng.range(1, 4000) as i32,
                _ => 0,
            };
            let ci = rng.range(0, area as u32) as usize;
            let coeff_ctx_g = rng.range(0, 42) as usize;
            let coeff_ctx_e = rng.range(0, 4) as usize;
            let dc_sign_ctx = rng.range(0, 3) as usize;
            let sign = rng.range(0, 2) as usize;

            // get_two_coeff_cost_simple (assumes ci > 0, abs_qc may be 0..).
            if ci > 0 {
                let (gc, gl) = two_coeff_cost_simple(ci, abs_qc, coeff_ctx_g, &t, bhl, tx_class, &levels);
                let (wc, wl) = c::ref_two_coeff_cost_simple(ci, abs_qc, coeff_ctx_g, &base, &lps, bhl, tc, &levels);
                assert_eq!((gc, gl), (wc, wl), "two_coeff_simple ts={tx_size} ci={ci} qc={abs_qc}");
            }

            // get_coeff_cost_eob (abs_qc >= 1 at the eob position).
            let eob_qc = abs_qc.max(1);
            let ge = coeff_cost_eob(ci, eob_qc, sign, coeff_ctx_e, dc_sign_ctx, &t, bhl, tx_class);
            let we = c::ref_coeff_cost_eob(ci, eob_qc, sign, coeff_ctx_e, dc_sign_ctx, &base_eob, &dc_sign, &lps, bhl, tc);
            assert_eq!(ge, we, "coeff_cost_eob ts={tx_size} ci={ci} qc={eob_qc}");

            // get_coeff_cost_general, both is_last polarities.
            for &is_last in &[false, true] {
                let cc = if is_last { coeff_ctx_e } else { coeff_ctx_g };
                let qc = if is_last { eob_qc } else { abs_qc };
                let gg = coeff_cost_general(is_last, ci, qc, sign, cc, dc_sign_ctx, &t, bhl, tx_class, &levels);
                let wg = c::ref_coeff_cost_general(is_last, ci, qc, sign, cc, dc_sign_ctx, &base_eob, &base, &dc_sign, &lps, bhl, tc, &levels);
                assert_eq!(gg, wg, "coeff_cost_general last={is_last} ts={tx_size} ci={ci} qc={qc}");
            }
        }
    }
}
