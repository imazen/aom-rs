//! Encoder composition layer: `av1_xform_quant` (libaom `av1/encoder/encodemb.c`).
//!
//! The per-block encoder workhorse = `av1_xform` (forward 2-D transform) +
//! `av1_quant` (quantize + entropy-context write). Every sub-step is already
//! bit-exact against C in its own crate:
//! - forward transform: [`aom_transform::txfm2d::av1_fwd_txfm2d`]
//! - quantizers: [`aom_quant`] (fp/b, flat/QM — lowbd 8-bit here)
//! - neighbour context: [`aom_txb::txb_entropy_context`]
//!
//! This crate wires them in the exact order/params libaom uses, so a residual
//! block maps to byte-identical (qcoeff, dqcoeff, eob, txb_entropy_ctx). The
//! forward transform is bd=8 (lowbd); highbd composition follows once the highbd
//! forward transform lands.
#![forbid(unsafe_code)]

use aom_quant::{
    aom_quantize_b_no_qmatrix, aom_quantize_b_qm, av1_quantize_fp_no_qmatrix, av1_quantize_fp_qm,
};
use aom_transform::txfm2d::av1_fwd_txfm2d;
use aom_txb::{
    get_txb_ctx, optimize_txb, optimize_txb_qm, scan, txb_entropy_context, txb_high, txb_wide,
    CoeffCostTables,
};

/// Full (un-adjusted) transform width per `TX_SIZE` — the residual/coeff buffer
/// dimensions the forward transform reads/writes before 64-point repacking.
const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
/// `tx_size_2d[tx_size]` — full pel count, drives `av1_get_tx_scale` (log_scale).
const TX_SIZE_2D: [i32; 19] =
    [16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024];

/// `av1_get_tx_scale(tx_size)` = `(pels > 256) + (pels > 1024)`.
#[inline]
pub fn tx_scale(tx_size: usize) -> i32 {
    let pels = TX_SIZE_2D[tx_size];
    (pels > 256) as i32 + (pels > 1024) as i32
}

/// Which quantizer `av1_quant` dispatches to (`quant_func_list` row). The DC-only
/// fast path (`AV1_XFORM_QUANT_DC`) is not modelled yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantKind {
    /// `AV1_XFORM_QUANT_FP` — the fast (round-only) quantizer.
    Fp,
    /// `AV1_XFORM_QUANT_B` — the dead-zone (`zbin` + `quant`/`quant_shift`) quantizer.
    B,
}

/// The `[dc, ac]` quantizer parameter tables (the `*_QTX` fields libaom's
/// `MACROBLOCK_PLANE` feeds the quantizer). `zbin`/`quant_shift` are only read by
/// [`QuantKind::B`]; `round` is the fp round for FP and the b round for B.
#[derive(Clone, Copy, Debug)]
pub struct QuantParams<'a> {
    pub zbin: &'a [i16; 2],
    pub round: &'a [i16; 2],
    pub quant: &'a [i16; 2],
    pub quant_shift: &'a [i16; 2],
    pub dequant: &'a [i16; 2],
    /// Per-position quant matrix (`qm`) and its inverse (`iqm`), both indexed by
    /// raster position (length = block area). `None` = flat (no quant matrix).
    pub qm: Option<&'a [u8]>,
    pub iqm: Option<&'a [u8]>,
}

/// Output of [`xform_quant`]: the quantized block plus the propagated context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XformQuantResult {
    pub coeff: Vec<i32>,
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    pub eob: u16,
    pub txb_entropy_ctx: u8,
}

/// `av1_xform_quant` for a single (bd=8) transform block: forward-transform the
/// `residual` (full `TX_W*TX_H`, row-major, stride `TX_W`), quantize with `kind`,
/// and (unless `use_optimize_b`, which defers to the trellis) write the neighbour
/// entropy context. Returns byte-identical output to libaom's `av1_xform` +
/// `av1_quant`.
pub fn xform_quant(
    residual: &[i16],
    tx_size: usize,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    use_optimize_b: bool,
) -> XformQuantResult {
    let full = TX_W[tx_size] * TX_H[tx_size];
    assert_eq!(residual.len(), full, "residual must be full TX_W*TX_H");
    let n_coeffs = txb_wide(tx_size) * txb_high(tx_size); // == av1_get_max_eob
    let log_scale = tx_scale(tx_size);
    let sc = scan(tx_size, tx_type);

    // av1_xform: forward 2-D transform into a full-size buffer (64-point sizes
    // repack their valid area into the first n_coeffs entries in-place).
    let mut coeff = vec![0i32; full];
    av1_fwd_txfm2d(residual, &mut coeff, TX_W[tx_size], tx_type, tx_size);

    // av1_quant: quantize the valid coefficient block.
    let mut qcoeff = vec![0i32; n_coeffs];
    let mut dqcoeff = vec![0i32; n_coeffs];
    let src = &coeff[..n_coeffs];
    let eob = match (kind, qp.qm, qp.iqm) {
        (QuantKind::Fp, Some(qm), Some(iqm)) => av1_quantize_fp_qm(
            qp.round, qp.quant, qp.dequant, log_scale, qm, iqm, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::Fp, _, _) => av1_quantize_fp_no_qmatrix(
            qp.quant, qp.dequant, qp.round, log_scale, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, Some(qm), Some(iqm)) => aom_quantize_b_qm(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, qm, iqm, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, _, _) => aom_quantize_b_no_qmatrix(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
    };

    // av1_quant tail: entropy ctx is deferred to optimize_b when it will run.
    let txb_entropy_ctx = if use_optimize_b {
        0
    } else {
        txb_entropy_context(&qcoeff, tx_size, tx_type, eob as usize)
    };

    XformQuantResult { coeff, qcoeff, dqcoeff, eob, txb_entropy_ctx }
}

/// Neighbour entropy contexts + plane geometry for `get_txb_ctx` (the block's
/// above/left `ENTROPY_CONTEXT` bytes; each is `cul_level | dc_sign<<3`).
#[derive(Clone, Copy, Debug)]
pub struct BlockContext<'a> {
    pub above: &'a [i8],
    pub left: &'a [i8],
    pub plane: usize,
    /// Plane block size (BlockSize discriminant).
    pub plane_bsize: usize,
}

/// RD inputs for the coefficient trellis (`av1_optimize_b`).
#[derive(Clone, Copy)]
pub struct OptimizeInputs<'a> {
    pub cost: &'a CoeffCostTables<'a>,
    pub rdmult: i64,
    pub sharpness: i32,
}

/// Output of [`xform_quant_optimize`]: the final (trellis-optimized) block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XformQuantOptResult {
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    pub eob: u16,
    pub txb_entropy_ctx: u8,
    /// Total coefficient rate (the trellis result, or the skip-txb cost at eob 0).
    pub rate: i32,
}

/// The full speed-0 block coefficient pipeline: `av1_xform_quant` (with
/// `use_optimize_b`) + `get_txb_ctx` + `av1_optimize_b` (the trellis) + the final
/// entropy-context write. Returns the trellis-optimized block byte-identical to
/// libaom. At `eob == 0` the trellis is skipped and the rate is `av1_cost_skip_txb`
/// (the `txb_skip` = 1 cost), matching `av1_optimize_b`'s early return.
pub fn xform_quant_optimize(
    residual: &[i16],
    tx_size: usize,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    bctx: &BlockContext,
    opt: &OptimizeInputs,
) -> XformQuantOptResult {
    // av1_xform_quant with use_optimize_b: quantize but defer the entropy ctx.
    let xq = xform_quant(residual, tx_size, tx_type, kind, qp, true);
    let XformQuantResult { coeff, mut qcoeff, mut dqcoeff, eob, .. } = xq;

    // get_txb_ctx: neighbour contexts -> (txb_skip_ctx, dc_sign_ctx).
    let (txb_skip_ctx, dc_sign_ctx) =
        get_txb_ctx(bctx.plane_bsize, tx_size, bctx.plane, bctx.above, bctx.left);
    let txb_skip_ctx = txb_skip_ctx as usize;
    let dc_sign_ctx = dc_sign_ctx as usize;

    // av1_optimize_b: eob 0 -> skip-txb cost; else run the trellis.
    if eob == 0 {
        let rate = opt.cost.txb_skip[txb_skip_ctx * 2 + 1];
        return XformQuantOptResult { qcoeff, dqcoeff, eob: 0, txb_entropy_ctx: 0, rate };
    }

    let dequant = [qp.dequant[0], qp.dequant[1]];
    let sc = scan(tx_size, tx_type);
    let tcoeff = &coeff[..qcoeff.len()];
    let res = match (qp.qm, qp.iqm) {
        (Some(qm), Some(iqm)) => optimize_txb_qm(
            tx_size, tx_type, &mut qcoeff, &mut dqcoeff, tcoeff, eob as usize, dequant, opt.rdmult,
            dc_sign_ctx, txb_skip_ctx, opt.sharpness, sc, opt.cost, iqm, qm,
        ),
        _ => optimize_txb(
            tx_size, tx_type, &mut qcoeff, &mut dqcoeff, tcoeff, eob as usize, dequant, opt.rdmult,
            dc_sign_ctx, txb_skip_ctx, opt.sharpness, sc, opt.cost,
        ),
    };

    // Trellis tail: entropy ctx from the *optimized* qcoeff / eob.
    let txb_entropy_ctx = txb_entropy_context(&qcoeff, tx_size, tx_type, res.eob);
    XformQuantOptResult {
        qcoeff,
        dqcoeff,
        eob: res.eob as u16,
        txb_entropy_ctx,
        rate: res.rate,
    }
}
