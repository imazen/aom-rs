//! Differential harness for `predict_intra_high` — the highbd intra prediction
//! dispatch (`av1_predict_intra_block` mode routing + p_angle derivation, minus
//! palette / CfL). vs C libaom v3.14.1 (`ref_hbd_predict_intra`, transcribing the
//! same routing to the builder shims). Covers every intra mode, angle-delta,
//! filter-intra, tx size, availability combo, and bitdepth — pinning the mode
//! classification, the mode_to_angle_map table, and the builder selection.

use aom_intra::predict_intra_high;
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

const STRIDE: usize = 256;
const ROWS: usize = 256;
const ROW0: usize = 4;
const COL0: usize = 4;

#[allow(clippy::too_many_arguments)]
fn check(
    recon: &[u16], tx_size: usize, mode: usize, delta: i32, use_fi: bool, fi_mode: usize,
    disable: bool, filt: i32, combo: (i32, i32, i32, i32), bd: i32,
) {
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let ref_off = ROW0 * STRIDE + COL0;
    let (nt, ntr, nl, nbl) = combo;
    let mut got = vec![0u16; txw * txh];
    predict_intra_high(
        recon, ref_off, STRIDE, &mut got, txw, mode, delta, use_fi, fi_mode, disable, filt, tx_size,
        nt as usize, ntr, nl as usize, nbl, bd,
    );
    let want = c::ref_hbd_predict_intra(
        recon, ref_off, STRIDE, mode, delta, use_fi, fi_mode, disable, filt, tx_size, txw, txh, nt,
        ntr, nl, nbl, bd,
    );
    assert_eq!(
        got, want,
        "predict_intra divergence ts={tx_size} ({txw}x{txh}) mode={mode} delta={delta} use_fi={use_fi} fi_mode={fi_mode} disable={disable} filt={filt} combo={combo:?} bd={bd}"
    );
}

#[test]
fn predict_intra_matches_c() {
    let mut rng = Rng(0x9ec7_00d1_5a7c_4000);
    for &bd in &[8i32, 10, 12] {
        let recon: Vec<u16> = (0..STRIDE * ROWS).map(|_| (rng.next() % (1u64 << bd)) as u16).collect();
        for tx_size in 0..19usize {
            let (tw, th) = (TX_W[tx_size] as i32, TX_H[tx_size] as i32);
            let combos: [(i32, i32, i32, i32); 4] = [
                (tw, th, th, tw),   // full + above-right + below-left
                (tw, -1, th, -1),   // full top+left, no extension
                (tw, 0, th, 0),     // extension considered, 0 px
                (0, -1, th, -1),    // left only
            ];
            // Every mode: directional modes (1..=8) sweep angle-delta (pre-scaled
            // by ANGLE_STEP=3), non-directional (0, 9..12) use delta 0.
            for mode in 0..13usize {
                let deltas: &[i32] =
                    if (1..=8).contains(&mode) { &[-9, -6, -3, 0, 3, 6, 9] } else { &[0] };
                for &delta in deltas {
                    for &combo in &combos {
                        check(&recon, tx_size, mode, delta, false, 0, false, 1, combo, bd);
                    }
                }
            }
            // Filter-intra (luma-only, <= 32x32): every filter mode.
            if TX_W[tx_size] <= 32 && TX_H[tx_size] <= 32 {
                for fi_mode in 0..5usize {
                    for &combo in &combos {
                        check(&recon, tx_size, 0, 0, true, fi_mode, false, 1, combo, bd);
                    }
                }
            }
            // Config variety (edge-filter on/off, both filter types) on a few modes.
            for &mode in &[0usize, 1, 3, 9, 12] {
                let delta = if (1..=8).contains(&mode) { 3 } else { 0 };
                for disable in [false, true] {
                    for filt in [0i32, 1] {
                        check(&recon, tx_size, mode, delta, false, 0, disable, filt, combos[0], bd);
                    }
                }
            }
        }
    }
}
