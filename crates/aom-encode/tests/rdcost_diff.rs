//! Differential harness for the RD cost macros `RDCOST` / `RDCOST_NEG_R`
//! (av1/encoder/rd.h) vs C. Integer, bit-exact across the ranges the encoder
//! produces (rate ~ AV1_PROB_COST-scaled bits, dist ~ transform-domain SSE),
//! kept within the i64-non-overflow regime the real code stays in.

use aom_encode::rd::{rdcost, rdcost_neg_r};
use aom_sys_ref as c;

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
    fn range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo) as u64)) as i64
    }
}

#[test]
fn rdcost_matches_c() {
    let mut rng = Rng(0x0dc0_5709_e371_1111);
    // Corner cases first.
    for &(rm, rate, dist) in &[
        (1i32, 0i32, 0i64),
        (1, 1, 1),
        (i32::MAX, 0, 0),
        (i32::MAX, 1, 0),
        (1, i32::MAX, 0),
        (128, 4096, 1_000_000),
        (i32::MAX, 10_000_000, 100_000_000_000),
    ] {
        assert_eq!(
            rdcost(rm, rate, dist),
            c::ref_rdcost(rm, rate, dist),
            "rdcost {rm} {rate} {dist}"
        );
        assert_eq!(
            rdcost_neg_r(rm, rate, dist),
            c::ref_rdcost_neg_r(rm, rate, dist),
            "rdcost_neg_r {rm} {rate} {dist}"
        );
    }
    for _ in 0..500_000 {
        // rm in [1, i32::MAX]; rate up to 1e7 (well past a superblock's bit cost);
        // dist up to 1e13. rate*rm <= ~2.1e16, dist*128 <= ~1.3e15 — no i64 overflow.
        let rm = rng.range_i64(1, i32::MAX as i64 + 1) as i32;
        let rate = rng.range_i64(0, 10_000_001) as i32;
        let dist = rng.range_i64(0, 10_000_000_000_000);
        assert_eq!(
            rdcost(rm, rate, dist),
            c::ref_rdcost(rm, rate, dist),
            "rdcost {rm} {rate} {dist}"
        );
        // RDCOST_NEG_R is used where the rate is subtracted; exercise negative
        // rate too (the macro casts rate to i64 and multiplies).
        let neg_rate = rng.range_i64(-10_000_000, 10_000_001) as i32;
        assert_eq!(
            rdcost_neg_r(rm, neg_rate, dist),
            c::ref_rdcost_neg_r(rm, neg_rate, dist),
            "rdcost_neg_r {rm} {neg_rate} {dist}"
        );
    }
}
