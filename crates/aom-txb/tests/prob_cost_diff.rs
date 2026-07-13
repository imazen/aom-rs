//! Differential harness for `av1_cost_tokens_from_cdf` / `av1_cost_symbol`
//! (CDF → per-symbol RD cost tables) vs C libaom v3.14.1.

use aom_sys_ref as c;
use aom_txb::{cost_symbol, cost_tokens_from_cdf};

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

/// Valid `nsymbs`-symbol inverse-CDF: strictly decreasing to 0, then a count.
fn gen_cdf(rng: &mut Rng, nsymbs: usize) -> Vec<u16> {
    let mut cdf = vec![0u16; nsymbs + 1];
    let mut acc: u32 = 0;
    for e in cdf.iter_mut().take(nsymbs - 1) {
        acc += rng.range(1, (32000 / nsymbs as u32).max(2));
        *e = (32768u32.saturating_sub(acc)).max(1) as u16;
    }
    cdf[nsymbs - 1] = 0;
    cdf
}

#[test]
fn cost_symbol_matches_c_full_range() {
    // Every representable Q15 probability.
    for p15 in 1..32768i32 {
        // Rust vs a single-symbol cdf through the C token derivation would be
        // indirect; instead exercise cost_symbol via a 2-symbol cdf whose first
        // probability is exactly p15 (so the derivation calls cost_symbol(p15)).
        let cdf = [(32768 - p15) as u16, 0u16, 0u16];
        let want = c::ref_cost_tokens_from_cdf(2, &cdf, None);
        let mut got = [0i32; 2];
        cost_tokens_from_cdf(&mut got, &cdf, None);
        assert_eq!(got[0], want[0], "cost for p15={p15}");
        // Direct cost_symbol matches the derivation's first entry (p15 >= 4).
        if p15 >= 4 {
            assert_eq!(cost_symbol(p15), want[0], "cost_symbol({p15})");
        }
    }
}

#[test]
fn cost_tokens_from_cdf_matches_c() {
    let mut rng = Rng(0x_c057_abcd_ef01);
    for nsymbs in 2..=13usize {
        for _ in 0..3000 {
            let cdf = gen_cdf(&mut rng, nsymbs);

            // identity map
            let want = c::ref_cost_tokens_from_cdf(nsymbs, &cdf, None);
            let mut got = vec![0i32; nsymbs];
            cost_tokens_from_cdf(&mut got, &cdf, None);
            assert_eq!(got, want, "identity nsymbs={nsymbs}");

            // random inverse permutation
            let mut inv: Vec<i32> = (0..nsymbs as i32).collect();
            for i in (1..nsymbs).rev() {
                let j = rng.range(0, i as u32 + 1) as usize;
                inv.swap(i, j);
            }
            let want_p = c::ref_cost_tokens_from_cdf(nsymbs, &cdf, Some(&inv));
            let mut got_p = vec![0i32; nsymbs];
            cost_tokens_from_cdf(&mut got_p, &cdf, Some(&inv));
            assert_eq!(got_p, want_p, "permuted nsymbs={nsymbs}");
        }
    }
}
