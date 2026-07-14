//! Differential harness for `build_non_directional_intra_high` — the highbd
//! non-directional intra builder (edge assembly + predict), whose assembly is
//! archmage-`#[autoversion]`-vectorized. For a reconstruction buffer with valid
//! neighbours, the predicted block must be byte-identical to C libaom v3.14.1's
//! `highbd_build_non_directional_intra_predictors` (`ref_hbd_build_nd_intra`),
//! swept over the five non-directional modes × all 19 tx sizes × bitdepths
//! {8,10,12} × neighbour-availability combos (full / none / partial / top-only /
//! left-only — exercising DC/DC_TOP/DC_LEFT/DC_128, edge replication, and the
//! PAETH corner).

use aom_intra::build_non_directional_intra_high;
use aom_sys_ref as c;

const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
// AV1 PREDICTION_MODE values for the non-directional family.
const MODES: [usize; 5] = [0, 9, 10, 11, 12]; // DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH

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
    fn upto(&mut self, hi: usize) -> usize {
        (self.next() % (hi as u64 + 1)) as usize
    }
}

#[test]
fn build_non_directional_matches_c() {
    let mut rng = Rng(0x_b0a7_face_1234_5678);
    const STRIDE: usize = 96;
    const ROWS: usize = 96;
    // Block top-left at (2,2): leaves an above row, a left column, and the corner.
    const ROW0: usize = 2;
    const COL0: usize = 2;
    let ref_off = ROW0 * STRIDE + COL0;

    let mut saw_dc128 = false;
    let mut saw_partial = false;
    let mut saw_toponly = false;

    for &bd in &[8i32, 10, 12] {
        // Fresh reconstruction plane for this bitdepth.
        let recon: Vec<u16> = (0..STRIDE * ROWS).map(|_| rng.pixel(bd)).collect();
        for tx_size in 0..19usize {
            let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
            for &mode in &MODES {
                // Availability combos: none / top-only / left-only / full /
                // partial-both / a couple randomized mid-values.
                let mut combos: Vec<(usize, usize)> = vec![
                    (0, 0),
                    (txw, 0),
                    (0, txh),
                    (txw, txh),
                    (1, 1),
                    (txw / 2, txh / 2),
                ];
                for _ in 0..3 {
                    combos.push((rng.upto(txw), rng.upto(txh)));
                }
                for (n_top, n_left) in combos {
                    let mut got = vec![0u16; txw * txh];
                    build_non_directional_intra_high(
                        &recon, ref_off, STRIDE, &mut got, txw, mode, tx_size, n_top, n_left, bd,
                    );
                    let want = c::ref_hbd_build_nd_intra(
                        &recon, ref_off, STRIDE, mode, tx_size, txw, txh, n_top, n_left, bd,
                    );
                    assert_eq!(
                        got, want,
                        "build_nd divergence mode={mode} ts={tx_size} ({txw}x{txh}) bd={bd} n_top={n_top} n_left={n_left}"
                    );
                    saw_dc128 |= mode == 0 && n_top == 0 && n_left == 0;
                    saw_partial |= n_top > 0 && n_top < txw;
                    saw_toponly |= n_top > 0 && n_left == 0;
                }
            }
        }
    }
    assert!(saw_dc128, "never exercised the DC_128 (no-neighbour) path");
    assert!(saw_partial, "never exercised edge replication (partial top)");
    assert!(saw_toponly, "never exercised the top-only availability path");
}
