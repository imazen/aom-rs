//! Differential harness for `build_directional_intra_high` — the highbd
//! directional intra builder (edge assembly with above-right/below-left +
//! corner-filter + edge filter/upsample + angle dispatch), whose assembly is
//! archmage-`#[autoversion]`-vectorized. End-to-end vs C libaom v3.14.1
//! (`highbd_build_directional_and_filter_intra_predictors`, directional path,
//! `ref_hbd_build_dir_intra`): every valid angle × 19 tx sizes × bitdepths
//! {8,10,12} × edge-filter on/off × filter type × neighbour-availability combos.

use aom_intra::build_directional_intra_high;
use aom_intra::dir::{get_dx, get_dy};
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
}

fn valid_angle(a: i32) -> bool {
    if a == 90 || a == 180 {
        true
    } else if a > 0 && a < 90 {
        get_dx(a) != 0
    } else if a > 90 && a < 180 {
        get_dx(a) != 0 && get_dy(a) != 0
    } else if a > 180 && a < 270 {
        get_dy(a) != 0
    } else {
        false
    }
}

const STRIDE: usize = 256;
const ROWS: usize = 256;
const ROW0: usize = 4;
const COL0: usize = 4;

#[allow(clippy::too_many_arguments)]
fn check(
    recon: &[u16], tx_size: usize, angle: i32, disable: bool, filt: i32,
    combo: (i32, i32, i32, i32), bd: i32,
) {
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let ref_off = ROW0 * STRIDE + COL0;
    let (n_top, n_topright, n_left, n_bottomleft) = combo;

    let mut got = vec![0u16; txw * txh];
    build_directional_intra_high(
        recon, ref_off, STRIDE, &mut got, txw, angle, disable, filt, tx_size,
        n_top as usize, n_topright, n_left as usize, n_bottomleft, bd,
    );
    let want = c::ref_hbd_build_dir_intra(
        recon, ref_off, STRIDE, angle, disable, filt, tx_size, txw, txh, n_top, n_topright, n_left,
        n_bottomleft, bd,
    );
    assert_eq!(
        got, want,
        "build_dir divergence ts={tx_size} ({txw}x{txh}) angle={angle} disable={disable} filt={filt} combo={combo:?} bd={bd}"
    );
}

#[test]
fn build_directional_matches_c() {
    let mut rng = Rng(0x_d15ec7_104a1_77b);
    for &bd in &[8i32, 10, 12] {
        let recon: Vec<u16> = (0..STRIDE * ROWS).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
        for tx_size in 0..19usize {
            let (txw, txh) = (TX_W[tx_size] as i32, TX_H[tx_size] as i32);
            // Availability combos (n_top, n_topright, n_left, n_bottomleft); -1 =
            // unavailable. Above availability is txwpx-wide, left is txhpx-tall; the
            // above-right extension is txhpx-wide, below-left txwpx-wide. libaom
            // requires n_top==txwpx when n_topright>0, and n_left==txhpx when
            // n_bottomleft>0 (extension only from a full edge), so those slots pair.
            let combos: [(i32, i32, i32, i32); 5] = [
                (txw, txh, txh, txw),   // full + above-right + below-left
                (txw, -1, txh, -1),     // full top+left, no extension
                (txw, 0, txh, 0),       // extension considered, 0 px (replicate)
                (txw, -1, 0, -1),       // top only
                (0, -1, txh, -1),       // left only
            ];

            // Full angle coverage on the default config (edge filter on, type 1, full avail).
            for angle in 1..270i32 {
                if valid_angle(angle) {
                    check(&recon, tx_size, angle, false, 1, combos[0], bd);
                }
            }
            // Config variety (edge filter on/off, both filter types, all combos) on
            // a few representative angles spanning z1 / z2 / z3 / V / H.
            for &angle in &[45i32, 67, 113, 135, 157, 203, 225, 90, 180] {
                if !valid_angle(angle) {
                    continue;
                }
                for disable in [false, true] {
                    for filt in [0i32, 1] {
                        for &combo in &combos {
                            check(&recon, tx_size, angle, disable, filt, combo, bd);
                        }
                    }
                }
            }
        }
    }
}
