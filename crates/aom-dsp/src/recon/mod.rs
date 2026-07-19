//! aom-recon — shared residual reconstruction for aom-rs.
//!
//! Houses [`reconstruct_txb`] (dequant + inverse transform + add), the residual
//! half of per-block reconstruction. It composes the already-bit-exact
//! [`aom_txb::dequant_txb`] and
//! [`crate::transform::inv_txfm2d::av1_inv_txfm2d_add`] kernels, so both the decoder
//! (`aom-decode`) and the encoder (`aom-encode`) depend on it — the
//! reconstruction primitive lives here rather than in the encoder crate, so the
//! decoder does not have to depend on the encoder to reach it.


use crate::transform::inv_txfm2d::av1_inv_txfm2d_add;
use aom_txb::{dequant_txb, txb_high, txb_wide};

/// Reconstruct one transform block's pixels from its decoded coefficients: the
/// residual half of per-block decode reconstruction. Dequantize `qcoeff` (raster
/// layout, as produced by [`read_coeffs_txb`](aom_txb::read_coeffs_txb) /
/// `read_coding_block_plane`) and add the inverse transform onto the prediction
/// already in `dst`.
///
/// `dst` is the plane pixel buffer (bd-bit samples) with row `stride`; the
/// block's top-left is `dst[0]`. On entry `dst` holds the intra/inter prediction;
/// on return it holds the reconstructed block — the prediction plus the residual
/// clipped to `[0, (1<<bd)-1]` by [`av1_inv_txfm2d_add`]. This is the structural
/// inverse of the encoder's residual path (predict → subtract → `xform_quant`);
/// the predictor that fills `dst` is applied by the caller (intra-edge management
/// and the predictor call are the next reconstruction layer).
///
/// The coefficient layout is consistent end to end: the forward transform's
/// output layout, the scan-indexed storage the entropy coder uses, and this
/// inverse transform's input layout are the same convention (established by the
/// encoder's `av1_fwd_txfm2d` → `aom_quantize_b`(`coeff[scan[i]]`) path and
/// mirrored on decode), so `qcoeff`/`dqcoeff` feed straight in with no transpose.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_txb(
    dst: &mut [u16],
    stride: usize,
    tx_size: usize,
    tx_type: usize,
    qcoeff: &[i32],
    dequant: [i16; 2],
    iqmatrix: Option<&[u8]>,
    bd: i32,
) {
    // area == inv_input_len(tx_size) for every size (the coded region the inverse
    // transform reads), so the dequantized block feeds in directly.
    let area = txb_wide(tx_size) * txb_high(tx_size);
    let mut dqcoeff = vec![0i32; area];
    dequant_txb(qcoeff, &mut dqcoeff, tx_size, dequant, iqmatrix, bd);
    av1_inv_txfm2d_add(&dqcoeff, dst, stride, tx_type, tx_size, bd);
}
