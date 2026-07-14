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

pub mod encode_intra;
pub mod hog;
pub mod intra_rd;
pub mod intra_uv_rd;
pub mod mode_costs;
pub mod rd;
pub mod tx_search;

use aom_entropy::dec::OdEcDec;
use aom_entropy::enc::OdEcEnc;
use aom_quant::{
    aom_highbd_quantize_b_no_qmatrix, aom_highbd_quantize_b_qm, aom_quantize_b_no_qmatrix,
    aom_quantize_b_qm, av1_highbd_quantize_dc, av1_highbd_quantize_fp_no_qmatrix,
    av1_highbd_quantize_fp_qm, av1_quantize_dc, av1_quantize_fp_no_qmatrix, av1_quantize_fp_qm,
};
use aom_transform::inv_txfm2d::av1_inv_txfm2d_add;
use aom_transform::txfm2d::av1_fwd_txfm2d;
use aom_txb::{
    dequant_txb, get_txb_ctx, optimize_txb, optimize_txb_qm, scan, txb_entropy_context, txb_high,
    txb_wide, write_coeffs_txb, write_coeffs_txb_full, CoeffCostTables,
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
    /// `AV1_XFORM_QUANT_DC` — DC-only (quantizes coefficient 0, zeroes the rest).
    Dc,
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
    /// Bit depth (8/10/12). `> 8` selects the highbd (64-bit) quantizer variants;
    /// the forward transform, trellis, entropy context, and writer are all
    /// bd-independent (verified: forward output is identical for bd 8/10/12).
    pub bd: u8,
}

impl<'a> QuantParams<'a> {
    /// Build the quantizer parameters for one plane from the per-qindex rows
    /// [`aom_quant::set_q_index`] selects — the bridge from `(qindex, deltas)`
    /// to [`xform_quant`].
    ///
    /// Mirrors how libaom's quantize facades read `MACROBLOCK_PLANE`:
    /// - [`QuantKind::Fp`] (`av1_quantize_fp_facade`): `quant_fp_QTX` +
    ///   `round_fp_QTX`.
    /// - [`QuantKind::B`] (`av1_quantize_b_facade`): `quant_QTX` + `round_QTX`
    ///   (+ `zbin_QTX`/`quant_shift_QTX`, threaded below for every kind).
    /// - [`QuantKind::Dc`] (`av1_quantize_dc_facade`): `quant_fp_QTX[0]` +
    ///   `round_QTX` (the facade passes the B round with the FP multiplier).
    ///
    /// `qm`/`iqm` start as `None` (flat); set them for the quant-matrix path.
    pub fn from_plane_rows(rows: &aom_quant::PlaneQuantRows<'a>, kind: QuantKind, bd: u8) -> Self {
        let pair = |row: &'a [i16; 8]| -> &'a [i16; 2] {
            row[..2].try_into().expect("row has 8 lanes")
        };
        let (quant, round) = match kind {
            QuantKind::Fp => (rows.quant_fp, rows.round_fp),
            QuantKind::B => (rows.quant, rows.round),
            QuantKind::Dc => (rows.quant_fp, rows.round),
        };
        QuantParams {
            zbin: pair(rows.zbin),
            round: pair(round),
            quant: pair(quant),
            quant_shift: pair(rows.quant_shift),
            dequant: pair(rows.dequant),
            qm: None,
            iqm: None,
            bd,
        }
    }
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
    let hbd = qp.bd > 8;
    let eob = match (kind, qp.qm, qp.iqm, hbd) {
        (QuantKind::Fp, Some(qm), Some(iqm), false) => av1_quantize_fp_qm(
            qp.round, qp.quant, qp.dequant, log_scale, qm, iqm, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::Fp, Some(qm), Some(iqm), true) => av1_highbd_quantize_fp_qm(
            qp.round, qp.quant, qp.dequant, log_scale, qm, iqm, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::Fp, _, _, false) => av1_quantize_fp_no_qmatrix(
            qp.quant, qp.dequant, qp.round, log_scale, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::Fp, _, _, true) => av1_highbd_quantize_fp_no_qmatrix(
            qp.quant, qp.dequant, qp.round, log_scale, sc, src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, Some(qm), Some(iqm), false) => aom_quantize_b_qm(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, qm, iqm, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, Some(qm), Some(iqm), true) => aom_highbd_quantize_b_qm(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, qm, iqm, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, _, _, false) => aom_quantize_b_no_qmatrix(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, _, _, true) => aom_highbd_quantize_b_no_qmatrix(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, sc, src,
            &mut qcoeff, &mut dqcoeff,
        ),
        // DC-only: the DC scalars (quant[0]/dequant[0]); qm/iqm handled internally.
        (QuantKind::Dc, _, _, false) => av1_quantize_dc(
            qp.round, qp.quant[0], qp.dequant[0], log_scale, qp.qm, qp.iqm, src, &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::Dc, _, _, true) => av1_highbd_quantize_dc(
            qp.round, qp.quant[0], qp.dequant[0], log_scale, qp.qm, qp.iqm, src, &mut qcoeff,
            &mut dqcoeff,
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
    /// Forward-transform coefficients (pre-quantization) — the transform-domain
    /// distortion reference `dist_block_tx_domain` compares `dqcoeff` against.
    pub coeff: Vec<i32>,
    pub qcoeff: Vec<i32>,
    pub dqcoeff: Vec<i32>,
    pub eob: u16,
    pub txb_entropy_ctx: u8,
    /// Total coefficient rate (the trellis result, or the skip-txb cost at eob 0).
    pub rate: i32,
    /// The `get_txb_ctx` result — the contexts the coefficient writer also needs.
    pub txb_skip_ctx: usize,
    pub dc_sign_ctx: usize,
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
        return XformQuantOptResult {
            coeff,
            qcoeff,
            dqcoeff,
            eob: 0,
            txb_entropy_ctx: 0,
            rate,
            txb_skip_ctx,
            dc_sign_ctx,
        };
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
        coeff,
        qcoeff,
        dqcoeff,
        eob: res.eob as u16,
        txb_entropy_ctx,
        rate: res.rate,
        txb_skip_ctx,
        dc_sign_ctx,
    }
}

/// Encode a residual block all the way to entropy-coded coefficient bytes: the
/// full speed-0 path ([`xform_quant_optimize`]) followed by `av1_write_coeffs_txb`
/// on the range coder. `enc` accumulates the bitstream (call `enc.done()` for the
/// bytes); `cdfs` is the coefficient-CDF arena (`CDF_ARENA_LEN` u16), adapted in
/// place when `allow_update_cdf`. Returns the optimized block (its qcoeff is what
/// was written). `av1_write_tx_type` (plane-0 tx_type) is out of scope, matching
/// the writer. `plane_type` is `0` for luma (`plane == 0`), else `1`.
#[allow(clippy::too_many_arguments)]
pub fn encode_block_coeffs(
    residual: &[i16],
    tx_size: usize,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    bctx: &BlockContext,
    opt: &OptimizeInputs,
    allow_update_cdf: bool,
    enc: &mut OdEcEnc,
    cdfs: &mut [u16],
) -> XformQuantOptResult {
    let r = xform_quant_optimize(residual, tx_size, tx_type, kind, qp, bctx, opt);
    let plane_type = (bctx.plane > 0) as usize;
    write_coeffs_txb(
        enc,
        cdfs,
        &r.qcoeff,
        r.eob as usize,
        tx_size,
        tx_type,
        plane_type,
        r.txb_skip_ctx,
        r.dc_sign_ctx,
        allow_update_cdf,
    );
    r
}

/// Luma `tx_type` signaling context — what the encoder's mbmi/frame state supplies
/// to `av1_write_tx_type` (mode, filter-intra, inter/reduced flags, and the
/// qindex/skip/segment gate that permits transmission).
#[derive(Clone, Copy, Debug)]
pub struct TxTypeContext {
    pub is_inter: bool,
    pub reduced: bool,
    pub use_filter_intra: bool,
    pub fi_mode: usize,
    pub mode: usize,
    pub signal_gate: bool,
}

/// Like [`encode_block_coeffs`] but emits the *complete* txb bitstream: the luma
/// `tx_type` (via `write_coeffs_txb_full`) is written between the txb_skip flag and
/// the coefficients, matching `av1_write_coeffs_txb`. `ext_tx_cdf` is the
/// caller-selected ext-tx CDF slot (adapted in place, only touched for luma).
#[allow(clippy::too_many_arguments)]
pub fn encode_block_coeffs_full(
    residual: &[i16],
    tx_size: usize,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    bctx: &BlockContext,
    opt: &OptimizeInputs,
    ttx: &TxTypeContext,
    allow_update_cdf: bool,
    enc: &mut OdEcEnc,
    cdfs: &mut [u16],
    ext_tx_cdf: &mut [u16],
) -> XformQuantOptResult {
    let r = xform_quant_optimize(residual, tx_size, tx_type, kind, qp, bctx, opt);
    let plane_type = (bctx.plane > 0) as usize;
    write_coeffs_txb_full(
        enc,
        cdfs,
        ext_tx_cdf,
        &r.qcoeff,
        r.eob as usize,
        tx_size,
        tx_type,
        plane_type,
        r.txb_skip_ctx,
        r.dc_sign_ctx,
        allow_update_cdf,
        ttx.is_inter,
        ttx.reduced,
        ttx.use_filter_intra,
        ttx.fi_mode,
        ttx.mode,
        ttx.signal_gate,
    );
    r
}

// block_size_wide / block_size_high (pixels) for BLOCK_SIZES_ALL.
const BLK_W: [usize; 22] = [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
const BLK_H: [usize; 22] = [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
// tx_size_wide_unit / high_unit (units of 4-pel MI).
const TXU_W: [usize; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
const TXU_H: [usize; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];

/// The per-plane above/left `ENTROPY_CONTEXT` arrays after a coding block, for
/// verification / propagation to the next block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockContexts {
    pub above: Vec<i8>,
    pub left: Vec<i8>,
}

/// Encode one plane of a coding block: iterate its transform blocks in raster
/// order (`av1_foreach_transformed_block_in_plane`), threading the above/left
/// `ENTROPY_CONTEXT` arrays — each txb reads its neighbour context via
/// `get_txb_ctx`, then fills its footprint with its own `txb_entropy_context`
/// byte (`av1_set_entropy_contexts`, interior case). `txb_residuals` supplies one
/// contiguous residual per txb in raster order. Uniform `tx_size` tiling only
/// (the tx must divide `plane_bsize` evenly); frame-edge clipping is not modelled.
/// The coefficient bytes accumulate in `enc`; returns the final contexts.
#[allow(clippy::too_many_arguments)]
pub fn encode_coding_block_plane(
    txb_residuals: &[&[i16]],
    plane_bsize: usize,
    tx_size: usize,
    tx_type: usize,
    kind: QuantKind,
    qp: &QuantParams,
    opt: &OptimizeInputs,
    ttx: &TxTypeContext,
    plane: usize,
    allow_update_cdf: bool,
    enc: &mut OdEcEnc,
    cdfs: &mut [u16],
    ext_tx_cdf: &mut [u16],
) -> BlockContexts {
    let uw = BLK_W[plane_bsize] >> 2;
    let uh = BLK_H[plane_bsize] >> 2;
    let txw = TXU_W[tx_size];
    let txh = TXU_H[tx_size];
    assert!(uw.is_multiple_of(txw) && uh.is_multiple_of(txh), "tx_size must tile plane_bsize evenly");
    let mut above = vec![0i8; uw];
    let mut left = vec![0i8; uh];

    let mut idx = 0;
    let mut blk_row = 0;
    while blk_row < uh {
        let mut blk_col = 0;
        while blk_col < uw {
            let cul = {
                let bctx = BlockContext {
                    above: &above[blk_col..],
                    left: &left[blk_row..],
                    plane,
                    plane_bsize,
                };
                let r = encode_block_coeffs_full(
                    txb_residuals[idx], tx_size, tx_type, kind, qp, &bctx, opt, ttx, allow_update_cdf,
                    enc, cdfs, ext_tx_cdf,
                );
                r.txb_entropy_ctx as i8
            };
            above[blk_col..blk_col + txw].fill(cul);
            left[blk_row..blk_row + txh].fill(cul);
            idx += 1;
            blk_col += txw;
        }
        blk_row += txh;
    }
    BlockContexts { above, left }
}

/// The pixel-domain reconstruction distortion for a transform block — the SSE
/// the encoder's final RD uses. Reconstructs `pred + inv_txfm(dqcoeff)` (clamped
/// to the bd pixel range, via [`av1_inv_txfm2d_add`]) and returns the sum of
/// squared differences vs `source`. `dqcoeff` has `inv_input_len(tx_size)`
/// entries; `pred`/`source`/output are the full `TX_W x TX_H` pixel block
/// (contiguous, stride = TX_W). Pixels are u16 (libaom's internal representation
/// for all bit depths; for bd=8 the values are simply <= 255).
pub fn pixel_distortion(
    dqcoeff: &[i32],
    tx_size: usize,
    tx_type: usize,
    pred: &[u16],
    source: &[u16],
    bd: i32,
) -> i64 {
    let w = TX_W[tx_size];
    let h = TX_H[tx_size];
    let mut recon = pred[..w * h].to_vec();
    aom_transform::inv_txfm2d::av1_inv_txfm2d_add(dqcoeff, &mut recon, w, tx_type, tx_size, bd);
    aom_dist::highbd_sse(&recon, w, source, w, w, h)
}

/// `RIGHT_SIGNED_SHIFT(value, n)` (`aom_ports/mem.h`): arithmetic right shift for
/// `n >= 0`, or a left shift by `-n` for `n < 0`.
#[inline]
fn right_signed_shift_i64(value: i64, n: i32) -> i64 {
    if n < 0 {
        value << (-n)
    } else {
        value >> n
    }
}

/// `dist_block_tx_domain` non-QM path (av1/encoder/tx_search.c) — the
/// transform-domain distortion (`dist`) and coefficient energy (`sse`) for one
/// transform block, normalized to the common Q4 scale. This is the distortion
/// term the per-txb intra RD cost feeds to [`rd::rdcost`], computed without an
/// inverse transform (unlike [`pixel_distortion`]).
///
/// Composes the validated [`aom_dist::block_error`] /
/// [`aom_dist::highbd_block_error`] with the exact per-tx-size normalization
/// shift `(MAX_TX_SCALE - av1_get_tx_scale(tx_size)) * 2` (with `MAX_TX_SCALE ==
/// 1`); the shift is **negative** — a left shift — for 64-wide transforms
/// (`tx_scale == 2`). `coeff` / `dqcoeff` must each hold at least
/// `av1_get_max_eob(tx_size) = txb_wide * txb_high` entries; only that prefix is
/// read (matching C's `buffer_length`). Returns `(dist, sse)`.
pub fn dist_block_tx_domain(coeff: &[i32], dqcoeff: &[i32], tx_size: usize, bd: u8) -> (i64, i64) {
    let n = txb_wide(tx_size) * txb_high(tx_size);
    let (dist, sse) = if bd > 8 {
        aom_dist::highbd_block_error(&coeff[..n], &dqcoeff[..n], bd)
    } else {
        aom_dist::block_error(&coeff[..n], &dqcoeff[..n])
    };
    // shift = (MAX_TX_SCALE - av1_get_tx_scale(tx_size)) * 2, MAX_TX_SCALE = 1.
    let shift = (1 - tx_scale(tx_size)) * 2;
    (
        right_signed_shift_i64(dist, shift),
        right_signed_shift_i64(sse, shift),
    )
}

/// Coefficient-level per-transform-block RD cost — the core of the speed-0 intra
/// tx-type RD evaluation (av1/encoder/tx_search.c): given the outputs of
/// [`xform_quant_optimize`] (`coeff` = forward-transform coefficients,
/// `qcoeff` = quantized, `dqcoeff` = dequantized, plus `eob` and the entropy
/// contexts), compute
/// `RDCOST(rdmult, cost_coeffs_txb(qcoeff, …), dist_block_tx_domain(coeff, dqcoeff, …))`.
///
/// The rate here is the coefficient-coding bits only ([`aom_txb::cost_coeffs_txb`]);
/// the block-level mode / tx-type signaling bits are added by the caller and are
/// out of scope for this txb-level primitive. Distortion is transform-domain
/// ([`dist_block_tx_domain`]). `cost_tables` are the derived
/// `LV_MAP_COEFF_COST` / `LV_MAP_EOB_COST` cost tables; `rdmult` is the RD
/// multiplier from [`rd::av1_compute_rd_mult_based_on_qindex`].
#[allow(clippy::too_many_arguments)]
pub fn txb_rd_cost(
    coeff: &[i32],
    qcoeff: &[i32],
    dqcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    cost_tables: &CoeffCostTables,
    rdmult: i32,
    bd: u8,
) -> i64 {
    let rate = aom_txb::cost_coeffs_txb(
        qcoeff,
        eob,
        tx_size,
        tx_type,
        txb_skip_ctx,
        dc_sign_ctx,
        cost_tables,
    );
    let (dist, _sse) = dist_block_tx_domain(coeff, dqcoeff, tx_size, bd);
    rd::rdcost(rdmult, rate, dist)
}

/// `read_coding_block_plane` — decode inverse of [`encode_coding_block_plane`]: iterate a
/// plane's transform blocks in raster order, reading each txb's coefficients via
/// `read_coeffs_txb_full` with the `get_txb_ctx`-derived contexts, threading the above/left
/// entropy context (`txb_entropy_context` cul level) exactly as the encoder does. Returns
/// the per-txb decoded coefficients (raster order) + the final [`BlockContexts`].
/// `tx_type_chroma` is the luma-derived tx_type used for chroma planes (ignored for luma,
/// which reads its own).
#[allow(clippy::too_many_arguments)]
pub fn read_coding_block_plane(
    dec: &mut OdEcDec,
    cdfs: &mut [u16],
    ext_tx_cdf: &mut [u16],
    plane_bsize: usize,
    tx_size: usize,
    tx_type_chroma: usize,
    plane: usize,
    allow_update_cdf: bool,
    ttx: &TxTypeContext,
) -> (Vec<Vec<i32>>, BlockContexts) {
    let uw = BLK_W[plane_bsize] >> 2;
    let uh = BLK_H[plane_bsize] >> 2;
    let txw = TXU_W[tx_size];
    let txh = TXU_H[tx_size];
    assert!(uw.is_multiple_of(txw) && uh.is_multiple_of(txh), "tx_size must tile plane_bsize evenly");
    let area = aom_txb::txb_wide(tx_size) * aom_txb::txb_high(tx_size);
    let mut above = vec![0i8; uw];
    let mut left = vec![0i8; uh];
    let mut coeffs = Vec::new();
    let mut blk_row = 0;
    while blk_row < uh {
        let mut blk_col = 0;
        while blk_col < uw {
            let (txb_skip_ctx, dc_sign_ctx) =
                aom_txb::get_txb_ctx(plane_bsize, tx_size, plane, &above[blk_col..], &left[blk_row..]);
            let mut tcoeff = vec![0i32; area];
            let (eob, tx_type) = aom_txb::read_coeffs_txb_full(
                dec, cdfs, ext_tx_cdf, &mut tcoeff, tx_size, plane, txb_skip_ctx as usize,
                dc_sign_ctx as usize, allow_update_cdf, ttx.is_inter, ttx.reduced, ttx.signal_gate,
                tx_type_chroma,
            );
            let cul = aom_txb::txb_entropy_context(&tcoeff, tx_size, tx_type, eob) as i8;
            above[blk_col..blk_col + txw].fill(cul);
            left[blk_row..blk_row + txh].fill(cul);
            coeffs.push(tcoeff);
            blk_col += txw;
        }
        blk_row += txh;
    }
    (coeffs, BlockContexts { above, left })
}

/// Reconstruct one transform block's pixels from its decoded coefficients: the
/// residual half of per-block decode reconstruction. Dequantize `qcoeff` (raster
/// layout, as produced by [`read_coeffs_txb`](aom_txb::read_coeffs_txb) /
/// [`read_coding_block_plane`]) and add the inverse transform onto the prediction
/// already in `dst`.
///
/// `dst` is the plane pixel buffer (bd-bit samples) with row `stride`; the
/// block's top-left is `dst[0]`. On entry `dst` holds the intra/inter prediction;
/// on return it holds the reconstructed block — the prediction plus the residual
/// clipped to `[0, (1<<bd)-1]` by [`av1_inv_txfm2d_add`]. This is the structural
/// inverse of the encoder's residual path (predict → subtract → [`xform_quant`]);
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
