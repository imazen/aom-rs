//! Differential harness for `av1_fill_coeff_costs` (per txs_ctx/plane
//! `LV_MAP_COEFF_COST` fill) vs C libaom: the assembled cost tables — including
//! the `base_cost[4..7]` trellis-diff and `lps_cost` cumulation/diff fixups —
//! must be integer-identical.

use aom_sys_ref as c;
use aom_txb::fill_lv_map_coeff_cost;

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
}

/// `count` valid `n`-symbol inverse-CDFs back to back (each `n+1` u16: strictly
/// decreasing to 0 + a zeroed adaptation counter).
fn gen_cdfs(rng: &mut Rng, count: usize, n: usize) -> Vec<u16> {
    let mut out = vec![0u16; count * (n + 1)];
    for slot in 0..count {
        let base = slot * (n + 1);
        let mut acc: u32 = 0;
        for e in out[base..base + n - 1].iter_mut() {
            acc += rng.range(1, (32000 / n as u32).max(2));
            *e = (32768u32.saturating_sub(acc)).max(1) as u16;
        }
        out[base + n - 1] = 0;
    }
    out
}

#[test]
fn fill_coeff_costs_identical() {
    let mut rng = Rng(0x_f111_c057_0000_abcd);
    for _ in 0..4000 {
        let txb_skip = gen_cdfs(&mut rng, 13, 2);
        let base_eob = gen_cdfs(&mut rng, 4, 3);
        let base = gen_cdfs(&mut rng, 42, 4);
        let eob_extra = gen_cdfs(&mut rng, 9, 2);
        let dc_sign = gen_cdfs(&mut rng, 3, 2);
        let br = gen_cdfs(&mut rng, 21, 4);

        let (wts, wbe, wb, wee, wds, wl) =
            c::ref_fill_lv_map(&txb_skip, &base_eob, &base, &eob_extra, &dc_sign, &br);
        let got = fill_lv_map_coeff_cost(&txb_skip, &base_eob, &base, &eob_extra, &dc_sign, &br);

        assert_eq!(got.txb_skip, wts, "txb_skip");
        assert_eq!(got.base_eob, wbe, "base_eob");
        assert_eq!(got.base, wb, "base (incl [4..7] trellis-diff fixup)");
        assert_eq!(got.eob_extra, wee, "eob_extra");
        assert_eq!(got.dc_sign, wds, "dc_sign");
        assert_eq!(got.lps, wl, "lps (incl cumulation + diff fixup)");
    }
}
