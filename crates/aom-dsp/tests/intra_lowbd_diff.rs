//! Differential harness for the bd8 LOWBD (`u8` pixel) intra prediction dispatch
//! [`aom_dsp::intra::predict_intra_u8`] vs C libaom v3.14.1 AND vs the port's own
//! highbd (`u16`) bd8 path [`aom_dsp::intra::predict_intra_high`]. This is the
//! byte-identity PROOF for the lowbd decode pipeline's intra-prediction lever:
//! the narrower `u8` reconstruction plane must predict the SAME pixel that C (and
//! the u16 port) does at bit depth 8, across every intra mode.
//!
//! Two independent oracles, both asserted per cell:
//!   1. `u8_out[i] as u16 == ref_hbd_predict_intra(bd=8)[i]`  — vs the real
//!      exported C `av1_predict_intra_block` routing (`shim_hbd_predict_intra`).
//!   2. `u8_out[i] as u16 == predict_intra_high(bd=8)[i]`      — vs the port's
//!      already-C-verified highbd path (guards against the two ever drifting).
//!
//! Coverage mirrors `predict_intra_diff.rs` but at bd8 only, over every intra
//! mode, angle-delta (pre-scaled by ANGLE_STEP), filter-intra mode, tx size,
//! neighbour-availability combo, and edge-filter/filter-type toggle — pinning the
//! mode classification, `mode_to_angle_map`, the builder selection, the reference
//! edge assembly, the corner/edge low-pass + upsample conditioning, and the
//! directional / non-directional / filter-intra predictors on `u8`.
//!
//! Run the binary a second time under `AOM_FORCE_SCALAR=1` to exercise the scalar
//! dispatch tier of the highbd oracle's SIMD kernels against the same `u8` path.

use aom_dsp::intra::{predict_intra_high, predict_intra_u8};
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

#[allow(clippy::too_many_arguments)]
fn check(
    recon_u8: &[u8],
    recon_u16: &[u16],
    tx_size: usize,
    mode: usize,
    delta: i32,
    use_fi: bool,
    fi_mode: usize,
    disable: bool,
    filt: i32,
    combo: (i32, i32, i32, i32),
) {
    let (txw, txh) = (TX_W[tx_size], TX_H[tx_size]);
    let ref_off = ROW0 * STRIDE + COL0;
    let (nt, ntr, nl, nbl) = combo;

    // The lowbd path under test: u8 recon in, u8 block out, bd fixed at 8.
    let mut got_u8 = vec![0u8; txw * txh];
    predict_intra_u8(
        recon_u8,
        ref_off,
        STRIDE,
        &mut got_u8,
        txw,
        mode,
        delta,
        use_fi,
        fi_mode,
        disable,
        filt,
        tx_size,
        nt as usize,
        ntr,
        nl as usize,
        nbl,
    );

    // Oracle 1 — the real exported C predictor at bd8.
    let want_c = c::ref_hbd_predict_intra(
        recon_u16, ref_off, STRIDE, mode, delta, use_fi, fi_mode, disable, filt, tx_size, txw, txh,
        nt, ntr, nl, nbl, 8,
    );

    // Oracle 2 — the port's already-C-verified highbd path at bd8.
    let mut want_hi = vec![0u16; txw * txh];
    predict_intra_high(
        recon_u16,
        ref_off,
        STRIDE,
        &mut want_hi,
        txw,
        mode,
        delta,
        use_fi,
        fi_mode,
        disable,
        filt,
        tx_size,
        nt as usize,
        ntr,
        nl as usize,
        nbl,
        8,
    );

    for i in 0..txw * txh {
        assert_eq!(
            got_u8[i] as u16, want_c[i],
            "lowbd intra vs C divergence at px {i}: ts={tx_size} ({txw}x{txh}) mode={mode} \
             delta={delta} use_fi={use_fi} fi_mode={fi_mode} disable={disable} filt={filt} \
             combo={combo:?}"
        );
        assert_eq!(
            got_u8[i] as u16, want_hi[i],
            "lowbd intra vs highbd-port divergence at px {i}: ts={tx_size} ({txw}x{txh}) \
             mode={mode} delta={delta} use_fi={use_fi} fi_mode={fi_mode} disable={disable} \
             filt={filt} combo={combo:?}"
        );
    }
}

/// Full mode / angle-delta / filter-intra / availability / edge-toggle sweep at
/// bd8. Same grid as `predict_intra_diff.rs`, dual-oracle on the `u8` path.
#[test]
fn predict_intra_lowbd_matches_c_and_highbd() {
    let mut rng = Rng(0x9ec7_00d1_5a7c_4000);
    // A single 8-bit reconstruction plane; the u16 mirror is the exact widening
    // (`u8 as u16`), so the two oracles see identical sample values.
    let recon_u8: Vec<u8> = (0..STRIDE * ROWS).map(|_| (rng.next() & 0xff) as u8).collect();
    let recon_u16: Vec<u16> = recon_u8.iter().map(|&p| p as u16).collect();

    for tx_size in 0..19usize {
        let (tw, th) = (TX_W[tx_size] as i32, TX_H[tx_size] as i32);
        let combos: [(i32, i32, i32, i32); 4] = [
            (tw, th, th, tw), // full + above-right + below-left
            (tw, -1, th, -1), // full top+left, no extension
            (tw, 0, th, 0),   // extension considered, 0 px
            (0, -1, th, -1),  // left only
        ];
        // Every mode: directional modes (1..=8) sweep angle-delta (pre-scaled by
        // ANGLE_STEP=3), non-directional (0, 9..12) use delta 0.
        for mode in 0..13usize {
            let deltas: &[i32] = if (1..=8).contains(&mode) {
                &[-9, -6, -3, 0, 3, 6, 9]
            } else {
                &[0]
            };
            for &delta in deltas {
                for &combo in &combos {
                    check(
                        &recon_u8, &recon_u16, tx_size, mode, delta, false, 0, false, 1, combo,
                    );
                }
            }
        }
        // Filter-intra (luma-only, <= 32x32): every filter mode.
        if TX_W[tx_size] <= 32 && TX_H[tx_size] <= 32 {
            for fi_mode in 0..5usize {
                for &combo in &combos {
                    check(
                        &recon_u8, &recon_u16, tx_size, 0, 0, true, fi_mode, false, 1, combo,
                    );
                }
            }
        }
        // Config variety (edge-filter on/off, both filter types) on a few modes.
        for &mode in &[0usize, 1, 3, 9, 12] {
            let delta = if (1..=8).contains(&mode) { 3 } else { 0 };
            for disable in [false, true] {
                for filt in [0i32, 1] {
                    check(
                        &recon_u8, &recon_u16, tx_size, mode, delta, false, 0, disable, filt,
                        combos[0],
                    );
                }
            }
        }
    }
}
