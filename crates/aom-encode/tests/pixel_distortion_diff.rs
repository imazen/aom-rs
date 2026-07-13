//! Differential for `pixel_distortion` — the reconstruction-domain RD distortion
//! (SSE of `pred + inv_txfm(dqcoeff)` clamped, vs source). Composes the validated
//! inverse transform + SSE; the oracle chains the same C references
//! (ref_inv_txfm2d_add -> ref_hbd_sse). Confirms the reconstruction/SSE wiring
//! (sizing, stride, clamp) matches C across tx sizes/types and bd 8/10/12.

use aom_encode::pixel_distortion;
use aom_sys_ref as c;
use aom_transform::inv_txfm2d::{inv_input_len, inv_txfm_valid};

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
    fn coeff(&mut self) -> i32 {
        // dequantized coefficient range (as inv_txfm2d_diff uses).
        (self.next() % (1 << 17)) as i32 - (1 << 16)
    }
    fn pixel(&mut self, bd: i32) -> u16 {
        (self.next() % (1u64 << bd)) as u16
    }
}

#[test]
fn pixel_distortion_matches_recon_plus_sse() {
    let mut rng = Rng(0x_9143_d157_9e37_79b9);
    const TX_TYPES: [usize; 7] = [0, 1, 2, 3, 9, 10, 11];
    let mut nonzero = 0usize;
    for tx_size in 0..19usize {
        let (w, h) = (TX_W[tx_size], TX_H[tx_size]);
        let n_in = inv_input_len(tx_size);
        for &tx_type in &TX_TYPES {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for &bd in &[8i32, 10, 12] {
                for _ in 0..30 {
                    let dqcoeff: Vec<i32> = (0..n_in).map(|_| rng.coeff()).collect();
                    let pred: Vec<u16> = (0..w * h).map(|_| rng.pixel(bd)).collect();
                    let source: Vec<u16> = (0..w * h).map(|_| rng.pixel(bd)).collect();

                    let got = pixel_distortion(&dqcoeff, tx_size, tx_type, &pred, &source, bd);

                    // Oracle: reconstruct with C inverse transform, then C SSE.
                    let mut recon = pred.clone();
                    c::ref_inv_txfm2d_add(tx_size, &dqcoeff, &mut recon, w, tx_type, bd);
                    let want = c::ref_hbd_sse(&recon, w, &source, w, w, h);

                    assert_eq!(got, want, "pixel_distortion ts={tx_size} tt={tx_type} bd={bd}");
                    nonzero += (got > 0) as usize;
                }
            }
        }
    }
    assert!(nonzero > 0, "distortion was always zero — test is vacuous");
}
