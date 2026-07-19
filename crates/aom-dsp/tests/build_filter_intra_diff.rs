//! Differential harness for `build_filter_intra_high` — the filter-intra builder
//! (edge assembly + recursive filter-intra predictor), the `use_filter_intra`
//! branch of libaom's directional-and-filter builder. End-to-end vs C libaom
//! v3.14.1 (`ref_hbd_build_filter_intra`), over every eligible tx size (≤ 32×32)
//! × all 5 FILTER_INTRA_MODEs × bitdepths {8,10,12} × availability combos.

use aom_dsp::intra::build_filter_intra_high;
use aom_sys_ref as c;

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];

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
}

const STRIDE: usize = 256;
const ROWS: usize = 256;
const ROW0: usize = 4;
const COL0: usize = 4;

#[test]
fn build_filter_intra_matches_c() {
    let mut rng = Rng(0xf117_eb01_14e3_0000);
    let ref_off = ROW0 * STRIDE + COL0;
    for &bd in &[8i32, 10, 12] {
        let recon: Vec<u16> = (0..STRIDE * ROWS)
            .map(|_| (rng.next() % (1u64 << bd)) as u16)
            .collect();
        for tx_size in 0..19usize {
            let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
            if txw > 32 || txh > 32 {
                continue; // filter-intra is luma-only, blocks <= 32x32
            }
            let (tw, th) = (txw as i32, txh as i32);
            // Availability combos (n_top, n_topright, n_left, n_bottomleft); -1 =
            // unavailable. Extension only from a full edge (n_topright>0 => n_top==txwpx;
            // n_bottomleft>0 => n_left==txhpx).
            let combos: [(i32, i32, i32, i32); 5] = [
                (tw, th, th, tw), // full + above-right + below-left
                (tw, -1, th, -1), // full top+left, no extension
                (tw, 0, th, 0),   // extension considered, 0 px (replicate)
                (tw, -1, 0, -1),  // top only
                (0, -1, th, -1),  // left only
            ];
            for mode in 0..5usize {
                for &(n_top, n_topright, n_left, n_bottomleft) in &combos {
                    let mut got = vec![0u16; txw * txh];
                    build_filter_intra_high(
                        &recon,
                        ref_off,
                        STRIDE,
                        &mut got,
                        txw,
                        mode,
                        tx_size,
                        n_top as usize,
                        n_topright,
                        n_left as usize,
                        n_bottomleft,
                        bd,
                    );
                    let want = c::ref_hbd_build_filter_intra(
                        &recon,
                        ref_off,
                        STRIDE,
                        mode as i32,
                        tx_size,
                        txw,
                        txh,
                        n_top,
                        n_topright,
                        n_left,
                        n_bottomleft,
                        bd,
                    );
                    assert_eq!(
                        got, want,
                        "build_filter_intra divergence ts={tx_size} ({txw}x{txh}) mode={mode} combo=({n_top},{n_topright},{n_left},{n_bottomleft}) bd={bd}"
                    );
                }
            }
        }
    }
}
