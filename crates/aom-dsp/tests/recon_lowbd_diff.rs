//! Differential harness for the bd8 LOWBD (u8 pixel) residual reconstruction
//! [`reconstruct_txb_u8_into`] — the recon family's lowbd dispatch entry. It
//! composes dequant (+ optional inverse quant matrix) -> inverse 2-D transform
//! -> add-onto-prediction. The narrower `u8` destination must reconstruct the
//! SAME pixel that
//!
//!   1. C libaom v3.14.1 does — `ref_dequant_txb` + `ref_inv_txfm2d_add`
//!      composed identically at bd=8 (the same two bit-exact kernels the
//!      highbd `reconstruct_txb_diff` on the encode side already validates), and
//!   2. the port's own already-C-verified highbd (u16) bd8 path
//!      [`reconstruct_txb_into`] with `bd == 8`
//!
//! reconstructs, pixel for pixel. This is the byte-identity PROOF for the recon
//! family's lowbd lever. Unlike `inv_txfm2d_lowbd_diff` (which feeds raw i32
//! coefficients straight into the transform), this exercises the WHOLE recon
//! composition — the dequant step and the optional inverse quant matrix — on
//! the u8 path, which no other test covers.
//!
//! SAFE-STEP note: the coefficient path is still i32 (the butterfly precision is
//! not narrowed); only the destination pixel storage narrows to u8. At bd == 8
//! the highbd `highbd_clip_pixel_add` already clamps to `(1<<8)-1 == 255`, so a
//! u8 store of the clamped value equals the u16 store — see the crate `lowbd`
//! design notes.
//!
//! Run the scalar variant with `AOM_FORCE_SCALAR=1` to prove the SIMD column
//! pass ([`try_inv_col_pass_u8`]) and the scalar column pass reconstruct
//! identically:
//!
//! ```text
//! AOM_FORCE_SCALAR=1 cargo test -p zenav1-aom-dsp --test recon_lowbd_diff
//! ```

use aom_dsp::recon::{ReconScratch, reconstruct_txb_into, reconstruct_txb_u8_into};
use aom_dsp::transform::inv_txfm2d::{inv_input_len, inv_txfm_valid};
use aom_sys_ref as c;

/// Full (un-repacked) transform dims — the residual/prediction buffer is `w*h`.
const W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

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
    /// A quantized coefficient level: ~40% zeros (realistic sparsity), else a
    /// signed magnitude spread across the small-and-large range so the
    /// dequantized block spans the `dequant_txb` clamp domain `[-(1<<15),
    /// (1<<15)-1]` and drives the full inverse-transform butterfly (incl. the
    /// per-stage normative clamps). Kept under 2^18 so it never wraps the
    /// `& 0xfffff` level mask in `dequant_txb`.
    fn qcoeff(&mut self) -> i32 {
        let r = self.next();
        if r % 5 < 2 {
            return 0;
        }
        let mag = (self.next() % (1 << 17)) as i32;
        if r & 1 == 0 { mag } else { -mag }
    }
    /// A realistic dequant step (DC or AC), matching the encode-side harness's
    /// `[4, 800)` range.
    fn dequant(&mut self) -> i16 {
        (4 + self.next() % 796) as i16
    }
    /// An inverse-quant-matrix weight. libaom QM weights are u8; `get_dqv`
    /// folds them as `(iqm*dqv + 16) >> 5`. Biased away from the degenerate
    /// near-zero end so the residual stays non-trivial while still exercising
    /// the full formula. Fed byte-identically to port and C, so any value is
    /// byte-identity-valid.
    fn iqm(&mut self) -> u8 {
        (16 + (self.next() % 240)) as u8
    }
    fn pixel_u8(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
}

/// One (tx_size, tx_type) cell at a given stride and iqmatrix mode: random
/// quantized coeffs + random dequant (+ optional iqm) + random u8 prediction.
/// The lowbd u8 reconstruction must equal BOTH the C bd8 reconstruction and the
/// port's highbd bd8 reconstruction, pixel for pixel. Returns whether the
/// reconstruction actually altered any pixel (for the vacuity guard).
fn check(
    rng: &mut Rng,
    tx_size: usize,
    tx_type: usize,
    stride: usize,
    use_iqm: bool,
    scratch: &mut ReconScratch,
) -> bool {
    let (w, h) = (W[tx_size], H[tx_size]);
    let area = inv_input_len(tx_size); // == txb_wide*txb_high, the coded region
    let buf_len = (h - 1) * stride + w;

    let qcoeff: Vec<i32> = (0..area).map(|_| rng.qcoeff()).collect();
    let dequant = [rng.dequant(), rng.dequant()];
    let iqm: Option<Vec<u8>> = if use_iqm {
        Some((0..area).map(|_| rng.iqm()).collect())
    } else {
        None
    };
    let iqm_ref: Option<&[u8]> = iqm.as_deref();
    let pred: Vec<u8> = (0..buf_len).map(|_| rng.pixel_u8()).collect();

    // lowbd u8 path (the thing under test)
    let mut got_u8 = pred.clone();
    reconstruct_txb_u8_into(
        &mut got_u8, stride, tx_size, tx_type, &qcoeff, dequant, iqm_ref, scratch,
    );

    // oracle 1 — real exported C: dequant + inverse transform + add, at bd8
    let dq_c = c::ref_dequant_txb(&qcoeff, tx_size, dequant, iqm_ref, 8);
    let mut want_c: Vec<u16> = pred.iter().map(|&p| p as u16).collect();
    c::ref_inv_txfm2d_add(tx_size, &dq_c, &mut want_c, stride, tx_type, 8);

    // oracle 2 — port highbd (u16) path at bd8
    let mut want_hi: Vec<u16> = pred.iter().map(|&p| p as u16).collect();
    reconstruct_txb_into(
        &mut want_hi, stride, tx_size, tx_type, &qcoeff, dequant, iqm_ref, 8, scratch,
    );

    let mut changed = false;
    for i in 0..buf_len {
        assert_eq!(
            got_u8[i] as u16, want_c[i],
            "lowbd vs C: tx_size={tx_size} ({w}x{h}) tt={tx_type} stride={stride} iqm={use_iqm} px={i}"
        );
        assert_eq!(
            got_u8[i] as u16, want_hi[i],
            "lowbd vs highbd port: tx_size={tx_size} ({w}x{h}) tt={tx_type} stride={stride} iqm={use_iqm} px={i}"
        );
        changed |= got_u8[i] != pred[i];
    }
    changed
}

#[test]
fn recon_lowbd_zero_coeff() {
    // All-zero coeffs: residual is 0, the prediction must pass through unchanged
    // (both iqm modes; a strided destination proves padding is untouched).
    let mut rng = Rng(0x_11ec_04c0_ffee);
    let mut scratch = ReconScratch::default();
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            let (w, h) = (W[tx_size], H[tx_size]);
            let area = inv_input_len(tx_size);
            for &(stride, use_iqm) in &[(w, false), (w + 5, true)] {
                let buf_len = (h - 1) * stride + w;
                let qcoeff = vec![0i32; area];
                let dequant = [rng.dequant(), rng.dequant()];
                let iqm: Option<Vec<u8>> = if use_iqm {
                    Some((0..area).map(|_| rng.iqm()).collect())
                } else {
                    None
                };
                let pred: Vec<u8> = (0..buf_len).map(|_| rng.pixel_u8()).collect();
                let mut got = pred.clone();
                reconstruct_txb_u8_into(
                    &mut got, stride, tx_size, tx_type, &qcoeff, dequant, iqm.as_deref(),
                    &mut scratch,
                );
                assert_eq!(
                    got, pred,
                    "zero-coeff must pass through: tx_size={tx_size} stride={stride} iqm={use_iqm}"
                );
            }
        }
    }
}

#[test]
fn recon_lowbd_differential_fuzz() {
    let mut rng = Rng(0x_2ec0_5eed_2026);
    let mut scratch = ReconScratch::default();
    let mut any_changed = false;
    for tx_size in 0..19 {
        for tx_type in 0..16 {
            if !inv_txfm_valid(tx_type, tx_size) {
                continue;
            }
            let (w, _h) = (W[tx_size], H[tx_size]);
            // Tight and two strided destinations; both iqmatrix modes.
            for &stride in &[w, w + 3, w + 16] {
                for &use_iqm in &[false, true] {
                    for _ in 0..120 {
                        any_changed |=
                            check(&mut rng, tx_size, tx_type, stride, use_iqm, &mut scratch);
                    }
                }
            }
        }
    }
    // Vacuity guard: the recon must have actually altered the prediction on at
    // least one cell (i.e. a non-zero residual path was exercised, not just the
    // trivial pass-through).
    assert!(
        any_changed,
        "recon never altered the prediction across the whole fuzz — test is vacuous"
    );
}
