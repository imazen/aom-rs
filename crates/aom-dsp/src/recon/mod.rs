//! aom-recon — shared residual reconstruction for aom-rs.
//!
//! Houses [`reconstruct_txb`] (dequant + inverse transform + add), the residual
//! half of per-block reconstruction. It composes the already-bit-exact
//! [`crate::txb::dequant_txb`] and
//! [`crate::transform::inv_txfm2d::av1_inv_txfm2d_add`] kernels, so both the decoder
//! (`aom-decode`) and the encoder (`aom-encode`) depend on it — the
//! reconstruction primitive lives here rather than in the encoder crate, so the
//! decoder does not have to depend on the encoder to reach it.


use crate::transform::inv_txfm2d::{InvTxfmScratch, av1_inv_txfm2d_add_into};
use crate::txb::{dequant_txb, txb_high, txb_wide};

/// Reconstruct one transform block's pixels from its decoded coefficients: the
/// residual half of per-block decode reconstruction. Dequantize `qcoeff` (raster
/// layout, as produced by [`read_coeffs_txb`](crate::txb::read_coeffs_txb) /
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
    let mut scratch = ReconScratch::default();
    reconstruct_txb_into(
        dst, stride, tx_size, tx_type, qcoeff, dequant, iqmatrix, bd, &mut scratch,
    );
}

/// Reusable per-transform-block scratch for [`reconstruct_txb_into`].
///
/// Holds the dequantized-coefficient block and the inverse transform's row-pass
/// buffer. Both are fully overwritten on every use (`dequant_txb` zero-fills
/// its output; the row pass writes every element of the transform buffer before
/// the column pass reads it), so reuse is byte-for-byte identical to allocating
/// fresh — it only removes allocator traffic.
///
/// Gate 3: the decoder reconstructs one transform block at a time and was
/// heap-allocating both buffers per block. On `dec_mosaic_4k_cq20` that
/// measured 89.5 M Ir/decode of calloc/free (2.8 % of the decode) across
/// ~126 k transform blocks.
#[derive(Default, Clone, Debug)]
pub struct ReconScratch {
    dqcoeff: Vec<i32>,
    txfm: InvTxfmScratch,
}

/// [`reconstruct_txb`] with caller-owned scratch. Byte-identical output; see
/// [`ReconScratch`].
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_txb_into(
    dst: &mut [u16],
    stride: usize,
    tx_size: usize,
    tx_type: usize,
    qcoeff: &[i32],
    dequant: [i16; 2],
    iqmatrix: Option<&[u8]>,
    bd: i32,
    scratch: &mut ReconScratch,
) {
    // area == inv_input_len(tx_size) for every size (the coded region the inverse
    // transform reads), so the dequantized block feeds in directly.
    let area = txb_wide(tx_size) * txb_high(tx_size);
    let dq = &mut scratch.dqcoeff;
    dq.clear();
    dq.resize(area, 0);
    dequant_txb(qcoeff, dq, tx_size, dequant, iqmatrix, bd);
    av1_inv_txfm2d_add_into(
        dq,
        dst,
        stride,
        tx_type,
        tx_size,
        bd,
        &mut scratch.txfm,
    );
}
