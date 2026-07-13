//! Differential harness for the Laplacian RD model (av1_model_rd_from_var_lapndz)
//! vs C libaom: fixed-point (rate, dist) from (variance, block-area-log2, qstep).
//! Exported oracle — bit-exact across the input ranges the encoder produces.

use aom_encode::rd::model_rd_from_var_lapndz;
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
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo)
    }
}

#[test]
fn model_rd_from_var_lapndz_matches_c() {
    let mut rng = Rng(0x0de1_d000_9e37_1111);
    // n_log2 = log2(block area), 4x4 (16) .. 64x64 (4096) => 4..12.
    for n_log2 in 4u32..=12 {
        for _ in 0..80_000 {
            // var 0 exercises the early-out; otherwise a wide spread.
            let var: i64 = if rng.next() % 50 == 0 {
                0
            } else {
                rng.range(1, 1 << 32) as i64
            };
            let qstep = rng.range(1, 1 << 15) as u32;
            let got = model_rd_from_var_lapndz(var, n_log2, qstep);
            let want = c::ref_model_rd_from_var_lapndz(var, n_log2, qstep);
            assert_eq!(got, want, "model_rd var={var} n_log2={n_log2} qstep={qstep}");
        }
    }
}
