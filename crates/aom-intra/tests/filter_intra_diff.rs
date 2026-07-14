//! Differential harness for `filter_intra_predict_high` — the recursive highbd
//! filter-intra predictor (`highbd_filter_intra_predictor`) — vs C libaom
//! v3.14.1 (`ref_hbd_filter_intra`, driven by the public av1_filter_intra_taps).
//! Filter-intra is luma-only for blocks ≤ 32×32; swept over every eligible tx
//! size × all 5 FILTER_INTRA_MODEs × bitdepths {8,10,12}.

use aom_intra::filter_intra_predict_high;
use aom_sys_ref as c;

const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

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
    fn pixel(&mut self, bd: i32) -> u16 {
        (self.next() % (1u64 << bd)) as u16
    }
}

#[test]
fn filter_intra_matches_c() {
    let mut rng = Rng(0xf117_ea1a_9931_0000);
    let mut checks = 0u64;
    for &bd in &[8i32, 10, 12] {
        for tx_size in 0..19usize {
            let (bw, bh) = (TX_W[tx_size], TX_H[tx_size]);
            if bw > 32 || bh > 32 {
                continue; // filter-intra is only defined for blocks <= 32x32
            }
            for mode in 0..5usize {
                for _ in 0..40 {
                    // above is a [-1..] view: index 0 = corner, 1.. = above samples.
                    let above: Vec<u16> = (0..bw + 2).map(|_| rng.pixel(bd)).collect();
                    let left: Vec<u16> = (0..bh).map(|_| rng.pixel(bd)).collect();

                    let mut got = vec![0u16; bw * bh];
                    filter_intra_predict_high(&mut got, bw, tx_size, &above, &left, mode, bd);
                    let want = c::ref_hbd_filter_intra(tx_size, bw, bh, &above, &left, mode, bd);

                    assert_eq!(
                        got, want,
                        "filter_intra divergence ts={tx_size} ({bw}x{bh}) mode={mode} bd={bd}"
                    );
                    checks += 1;
                }
            }
        }
    }
    assert!(checks > 1500, "expected many checks, got {checks}");
}
