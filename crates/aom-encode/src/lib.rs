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

pub mod allintra_vis;
pub mod ab_nn_prune;
pub mod ab_nn_weights;
pub mod cnn_partition;
pub mod denoise;
pub mod encode_intra;
pub mod encode_sb;
pub mod grain_table;
pub mod hog;
pub mod intra_rd;
pub mod intra_tx_nn_weights;
pub mod intra_uv_rd;
pub mod intrabc_search;
pub mod lf_search;
pub mod mode_costs;
pub mod noise_fft;
pub mod noise_model;
pub mod nonrd_pickmode;
pub mod obu_assemble;
pub mod pack;
pub mod palette_search;
pub mod part4_nn_weights;
pub mod part4_prune;
pub mod partition;
pub mod partition_pick;
pub mod pickcdef;
pub mod rc;
pub mod rd;
pub mod rd_pick;
pub mod real_costs;
pub mod resize;
pub mod speed_features;
pub mod superres_select;
pub mod tx_search;
pub mod var_part;
pub mod var_tx;

use aom_entropy::dec::OdEcDec;
use aom_entropy::enc::OdEcEnc;
use aom_quant::{
    aom_highbd_quantize_b_adaptive_helper, aom_highbd_quantize_b_no_qmatrix,
    aom_highbd_quantize_b_qm, aom_quantize_b_adaptive_helper, aom_quantize_b_no_qmatrix,
    aom_quantize_b_qm, av1_highbd_quantize_dc, av1_highbd_quantize_fp_no_qmatrix,
    av1_highbd_quantize_fp_qm, av1_quantize_dc, av1_quantize_fp_qm,
};
use aom_transform::inv_txfm2d::av1_fwht4x4;
use aom_transform::txfm2d::av1_fwd_txfm2d;
use aom_txb::{
    CoeffCostTables, get_txb_ctx, iscan, optimize_txb, optimize_txb_qm, scan, txb_entropy_context,
    txb_high, txb_wide, write_coeffs_txb, write_coeffs_txb_full,
};

/// Full (un-adjusted) transform width per `TX_SIZE` — the residual/coeff buffer
/// dimensions the forward transform reads/writes before 64-point repacking.
const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
/// `tx_size_2d[tx_size]` — full pel count, drives `av1_get_tx_scale` (log_scale).
const TX_SIZE_2D: [i32; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];

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

/// The `oxcf.tune_cfg` knobs the block-coefficient pipeline reads (`TuneCfg`,
/// av1/encoder/encoder.h) — threaded from the frame config into the trellis
/// and the tx-search distortion. `Default` = the PSNR-tune envelope (both
/// false), byte-identical to the pre-tune path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TuneKnobs {
    /// `tune_cfg.dist_metric == AOM_DIST_METRIC_QM_PSNR` (`--dist-metric=qm-psnr`,
    /// also installed by `tune=IQ`/`tune=SSIMULACRA2` via `handle_tuning`).
    /// Weights the trellis distortion by the forward QM (`av1_optimize_txb`,
    /// txb_rdopt.c:346-351) and the tx-search transform-domain distortion by
    /// [`aom_dist::block_error_qm`] (tx_search.c:1150/:1159), and forces
    /// transform-domain distortion on (`set_tx_domain_dist_params`,
    /// rdopt_utils.h:516-522).
    pub use_qm_dist_metric: bool,
    /// `tune_cfg.tuning ∈ {AOM_TUNE_IQ, AOM_TUNE_SSIMULACRA2}` — the
    /// `av1_optimize_txb` trellis `rshift = 7` (vs 5) arm (txb_rdopt.c:378-386),
    /// biasing the trellis towards keeping more coefficients.
    pub iq_tuning: bool,
}

/// Frame-level quantization-matrix context for one plane — `av1_setup_qmatrix`'s
/// inputs (`av1_quantize.c:370`). When present on [`QuantParams`], the effective
/// per-transform `qmatrix`/`iqmatrix` slices are selected INSIDE
/// [`xform_quant`] / [`xform_quant_optimize`] per `(tx_size, tx_type)` —
/// mirroring C, where the selection happens per transform block, not per plane
/// (1-D / identity transforms and the flat top level 15 select no matrix).
#[derive(Clone, Copy, Debug)]
pub struct QmCtx {
    /// The frame `qmatrix_level_{y,u,v}` for this plane (`av1_set_quantizer`);
    /// `NUM_QM_LEVELS - 1` (15) is the flat level (selectors return `None`).
    pub qm_level: usize,
    /// Plane index (0=Y, 1=U, 2=V) — the selectors group `plane >= 1` as chroma.
    pub plane: usize,
    /// `tune_cfg.dist_metric == AOM_DIST_METRIC_QM_PSNR`: the trellis weights
    /// its DISTORTION by the forward matrix (C selects `qmatrix` in
    /// `av1_optimize_txb` only under this metric, txb_rdopt.c:346-351; the
    /// production default PSNR metric runs the trellis with `qmatrix = NULL`
    /// while dequant still folds `iqmatrix`). Set via
    /// [`QuantParams::with_qm_dist_metric`].
    pub use_qm_dist_metric: bool,
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
    /// These are the PRE-SELECTED slices (kept for the kernel diff tests, which
    /// drive one explicit matrix); production QM encoding sets [`Self::qm_ctx`]
    /// instead and lets the funnel select per transform.
    pub qm: Option<&'a [u8]>,
    pub iqm: Option<&'a [u8]>,
    /// Frame QM context: `Some` = QM-on (select per `(tx_size, tx_type)` inside
    /// the funnel), `None` = use the explicit `qm`/`iqm` slices (flat when both
    /// `None` — the default, byte-identical to the pre-QM path).
    pub qm_ctx: Option<QmCtx>,
    /// Bit depth (8/10/12). `> 8` selects the highbd (64-bit) quantizer variants;
    /// the forward transform, trellis, entropy context, and writer are all
    /// bd-independent (verified: forward output is identical for bd 8/10/12).
    pub bd: u8,
    /// `xd->lossless[segment_id]` for this block's segment (coded-lossless =>
    /// `true` for every segment at qindex 0 with zero q-deltas). When set, the
    /// forward transform is the reversible 4x4 Walsh–Hadamard ([`av1_fwht4x4`])
    /// instead of `av1_fwd_txfm2d` — libaom's `highbd_fwd_txfm_4x4`
    /// (`hybrid_fwd_txfm.c:83`) makes the same branch. Lossless forces
    /// `tx_size == TX_4X4` and `tx_type == DCT_DCT` upstream.
    pub lossless: bool,
    /// `oxcf.q_cfg.quant_b_adapt` (`--quant-b-adapt`, default false). Read ONLY
    /// by [`QuantKind::B`]: when set, the dead-zone quantizer routes through the
    /// adaptive helper ([`aom_quant::aom_quantize_b_adaptive_helper`] / highbd)
    /// instead of the plain one — the prescan-widened dead-zone + the
    /// SKIP_EOB_FACTOR_ADJUST tail. Inert for [`QuantKind::Fp`]/[`QuantKind::Dc`]
    /// (C only threads `quant_b_adapt` into `av1_quantize_b_facade`). Set via
    /// [`QuantParams::with_adaptive`].
    pub adaptive: bool,
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
    /// `lossless` is the block's segment `xd->lossless` bit (routes the forward
    /// transform to [`av1_fwht4x4`]); pass `false` for the normal path.
    pub fn from_plane_rows(
        rows: &aom_quant::PlaneQuantRows<'a>,
        kind: QuantKind,
        bd: u8,
        lossless: bool,
    ) -> Self {
        let pair =
            |row: &'a [i16; 8]| -> &'a [i16; 2] { row[..2].try_into().expect("row has 8 lanes") };
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
            qm_ctx: None,
            bd,
            lossless,
            adaptive: false,
        }
    }

    /// Set `oxcf.q_cfg.quant_b_adapt` (`--quant-b-adapt`) — routes the
    /// [`QuantKind::B`] dead-zone quantizer through the adaptive helper. No-op
    /// for FP/DC kinds (C threads the flag only into `av1_quantize_b_facade`).
    #[must_use]
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Attach the frame QM context (`av1_setup_qmatrix` inputs) for this plane —
    /// the QM-on path. `qm_level == NUM_QM_LEVELS - 1` (15, the flat level) and
    /// 1-D / identity transforms still resolve to the flat quantizer per
    /// transform, exactly as C.
    pub fn with_qm(mut self, qm_level: usize, plane: usize) -> Self {
        self.qm_ctx = Some(QmCtx {
            qm_level,
            plane,
            use_qm_dist_metric: false,
        });
        self
    }

    /// Mark the QM-PSNR distortion metric on an attached frame QM context
    /// (`tune_cfg.dist_metric == AOM_DIST_METRIC_QM_PSNR`): the trellis then
    /// weights its distortion by the forward matrix (see [`QmCtx`]). A no-op
    /// without [`Self::with_qm`] — exactly C, where `av1_get_qmatrix` resolves
    /// the flat NULL matrix when QM is off (`av1_use_qmatrix` false loads the
    /// `NUM_QM_LEVELS - 1` seg matrices, quant_common.c:243/:256), so the
    /// QM-PSNR metric degrades to unweighted distortion.
    pub fn with_qm_dist_metric(mut self) -> Self {
        if let Some(cx) = &mut self.qm_ctx {
            cx.use_qm_dist_metric = true;
        }
        self
    }
}

/// Resolve the effective per-transform QM slices — `av1_setup_qmatrix`
/// (`av1_quantize.c:370`): with a frame QM context, select forward+inverse for
/// `(level, plane, tx_size, tx_type)` (both `None` for 1-D/identity transforms
/// and the flat level); without one, pass through the explicit pre-selected
/// slices (the kernel-diff path / flat default).
fn resolve_qm<'a>(
    qp: &QuantParams<'a>,
    tx_size: usize,
    tx_type: usize,
) -> (Option<&'a [u8]>, Option<&'a [u8]>) {
    match qp.qm_ctx {
        Some(cx) => (
            aom_quant::qmatrix(cx.qm_level, cx.plane, tx_size, tx_type),
            aom_quant::iqmatrix(cx.qm_level, cx.plane, tx_size, tx_type),
        ),
        None => (qp.qm, qp.iqm),
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
    // repack their valid area into the first n_coeffs entries in-place). For a
    // coded-lossless block libaom's `highbd_fwd_txfm_4x4` (hybrid_fwd_txfm.c:83)
    // swaps the DCT for the reversible 4x4 Walsh–Hadamard; lossless forces
    // TX_4X4 (== 0) and DCT_DCT (== 0) upstream, so this is the only tx here.
    let mut coeff = vec![0i32; full];
    if qp.lossless {
        debug_assert_eq!(tx_size, 0, "lossless forces TX_4X4");
        debug_assert_eq!(tx_type, 0, "lossless forces DCT_DCT");
        av1_fwht4x4(residual, &mut coeff, TX_W[tx_size]);
    } else {
        av1_fwd_txfm2d(residual, &mut coeff, TX_W[tx_size], tx_type, tx_size);
    }

    // av1_quant: quantize the valid coefficient block.
    let mut qcoeff = vec![0i32; n_coeffs];
    let mut dqcoeff = vec![0i32; n_coeffs];
    let src = &coeff[..n_coeffs];
    let hbd = qp.bd > 8;
    // av1_setup_qmatrix: per-(tx_size, tx_type) QM selection (QM-off / explicit
    // slices pass through unchanged — byte-identical to the pre-QM path).
    let (qm_sel, iqm_sel) = resolve_qm(qp, tx_size, tx_type);
    let eob = match (kind, qm_sel, iqm_sel, hbd) {
        // --quant-b-adapt: the dead-zone "b" quantizer routes through the
        // adaptive helper (av1_quantize_b_facade's use_quant_b_adapt arm); qm_sel
        // /iqm_sel carry the per-transform QM (None = flat). FP/DC unaffected.
        (QuantKind::B, _, _, true) if qp.adaptive => aom_highbd_quantize_b_adaptive_helper(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, qm_sel, iqm_sel, sc,
            src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::B, _, _, false) if qp.adaptive => aom_quantize_b_adaptive_helper(
            qp.zbin, qp.round, qp.quant, qp.quant_shift, qp.dequant, log_scale, qm_sel, iqm_sel, sc,
            src, &mut qcoeff, &mut dqcoeff,
        ),
        (QuantKind::Fp, Some(qm), Some(iqm), false) => av1_quantize_fp_qm(
            qp.round,
            qp.quant,
            qp.dequant,
            log_scale,
            qm,
            iqm,
            sc,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::Fp, Some(qm), Some(iqm), true) => av1_highbd_quantize_fp_qm(
            qp.round,
            qp.quant,
            qp.dequant,
            log_scale,
            qm,
            iqm,
            sc,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        // The hottest quantizer (speed-0 search path): SIMD-dispatched, bit-
        // identical to the scalar port at every tier (quantize_fp_simd_diff);
        // AOM_FORCE_SCALAR pins it back to the transcription.
        (QuantKind::Fp, _, _, false) => aom_quant::simd::av1_quantize_fp_no_qmatrix_dispatch(
            qp.quant,
            qp.dequant,
            qp.round,
            log_scale,
            sc,
            iscan(tx_size, tx_type),
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::Fp, _, _, true) => av1_highbd_quantize_fp_no_qmatrix(
            qp.quant,
            qp.dequant,
            qp.round,
            log_scale,
            sc,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::B, Some(qm), Some(iqm), false) => aom_quantize_b_qm(
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
            log_scale,
            qm,
            iqm,
            sc,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::B, Some(qm), Some(iqm), true) => aom_highbd_quantize_b_qm(
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
            log_scale,
            qm,
            iqm,
            sc,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::B, _, _, false) => aom_quantize_b_no_qmatrix(
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
            log_scale,
            sc,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::B, _, _, true) => aom_highbd_quantize_b_no_qmatrix(
            qp.zbin,
            qp.round,
            qp.quant,
            qp.quant_shift,
            qp.dequant,
            log_scale,
            sc,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        // DC-only: the DC scalars (quant[0]/dequant[0]); qm/iqm handled internally.
        (QuantKind::Dc, qm, iqm, false) => av1_quantize_dc(
            qp.round,
            qp.quant[0],
            qp.dequant[0],
            log_scale,
            qm,
            iqm,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
        (QuantKind::Dc, qm, iqm, true) => av1_highbd_quantize_dc(
            qp.round,
            qp.quant[0],
            qp.dequant[0],
            log_scale,
            qm,
            iqm,
            src,
            &mut qcoeff,
            &mut dqcoeff,
        ),
    };

    // av1_quant tail: entropy ctx is deferred to optimize_b when it will run.
    let txb_entropy_ctx = if use_optimize_b {
        0
    } else {
        txb_entropy_context(&qcoeff, tx_size, tx_type, eob as usize)
    };

    XformQuantResult {
        coeff,
        qcoeff,
        dqcoeff,
        eob,
        txb_entropy_ctx,
    }
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
    let XformQuantResult {
        coeff,
        mut qcoeff,
        mut dqcoeff,
        eob,
        ..
    } = xq;

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
    // Same av1_setup_qmatrix selection the quantize above used — the trellis
    // (optimize_txb_qm's get_dqv) must fold the SAME per-position inverse.
    //
    // The trellis DISTORTION qm mirrors C's `av1_optimize_txb` (txb_rdopt.c:
    // 346-351): the forward matrix is selected there ONLY under `dist_metric
    // == AOM_DIST_METRIC_QM_PSNR` (`--dist-metric=qm-psnr`, installed by
    // tune=IQ / SSIMULACRA2 — [`QmCtx::use_qm_dist_metric`]). On the
    // production QM path with the default PSNR metric the trellis distortion
    // is UNWEIGHTED (`qmatrix = NULL` in C), while the dequant still folds
    // the inverse matrix. The explicit-slice path (`qp.qm`/`qp.iqm`,
    // kernel-diff tests) keeps weighting the distortion — it models the
    // QM_PSNR arm the C shim oracle validates.
    let (qm_sel, iqm_sel) = resolve_qm(qp, tx_size, tx_type);
    let trellis_dist_qm = match qp.qm_ctx {
        Some(cx) if cx.use_qm_dist_metric => qm_sel,
        Some(_) => None,
        None => qm_sel,
    };
    let res = match (trellis_dist_qm, iqm_sel) {
        (qm, Some(iqm)) => optimize_txb_qm(
            tx_size,
            tx_type,
            &mut qcoeff,
            &mut dqcoeff,
            tcoeff,
            eob as usize,
            dequant,
            opt.rdmult,
            dc_sign_ctx,
            txb_skip_ctx,
            opt.sharpness,
            sc,
            opt.cost,
            iqm,
            qm,
        ),
        _ => optimize_txb(
            tx_size,
            tx_type,
            &mut qcoeff,
            &mut dqcoeff,
            tcoeff,
            eob as usize,
            dequant,
            opt.rdmult,
            dc_sign_ctx,
            txb_skip_ctx,
            opt.sharpness,
            sc,
            opt.cost,
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
const BLK_W: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLK_H: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
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
    assert!(
        uw.is_multiple_of(txw) && uh.is_multiple_of(txh),
        "tx_size must tile plane_bsize evenly"
    );
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
                    txb_residuals[idx],
                    tx_size,
                    tx_type,
                    kind,
                    qp,
                    &bctx,
                    opt,
                    ttx,
                    allow_update_cdf,
                    enc,
                    cdfs,
                    ext_tx_cdf,
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
    if n < 0 { value << (-n) } else { value >> n }
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

/// `dist_block_tx_domain` (av1/encoder/tx_search.c:1131) with the QM-PSNR
/// distortion-metric arm: when a per-transform forward matrix resolves AND the
/// QM-PSNR metric is active, the block error is QM-weighted
/// ([`aom_dist::block_error_qm`], scan-indexed — C passes `xd->bd` for highbd
/// and a literal 8 for lowbd, which `bd` covers directly); otherwise this is
/// exactly [`dist_block_tx_domain`] (C's `qmatrix == NULL ||
/// !use_qm_dist_metric` fallthrough at :1150/:1159). Same
/// `(MAX_TX_SCALE - tx_scale) * 2` normalization on both arms.
pub fn dist_block_tx_domain_qm(
    coeff: &[i32],
    dqcoeff: &[i32],
    tx_size: usize,
    bd: u8,
    qmatrix: Option<&[u8]>,
    scan: &[i16],
    use_qm_dist_metric: bool,
) -> (i64, i64) {
    let qm = match qmatrix {
        Some(qm) if use_qm_dist_metric => qm,
        _ => return dist_block_tx_domain(coeff, dqcoeff, tx_size, bd),
    };
    let n = txb_wide(tx_size) * txb_high(tx_size);
    let (dist, sse) = aom_dist::block_error_qm(&coeff[..n], &dqcoeff[..n], qm, &scan[..n], bd);
    let shift = (1 - tx_scale(tx_size)) * 2;
    (
        right_signed_shift_i64(dist, shift),
        right_signed_shift_i64(sse, shift),
    )
}

/// Resolve the per-transform FORWARD QM slice for the tx-search distortion —
/// the `quant_param.qmatrix` C threads from `av1_setup_qmatrix` into
/// `dist_block_tx_domain` (tx_search.c:2204-2249): `Some` only when the
/// [`QuantParams`] carries a frame QM context resolving a genuine matrix for
/// `(tx_size, tx_type)` (2-D transforms below the flat level).
pub fn dist_qmatrix<'a>(
    qp: &QuantParams<'a>,
    tx_size: usize,
    tx_type: usize,
) -> Option<&'a [u8]> {
    resolve_qm(qp, tx_size, tx_type).0
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
    assert!(
        uw.is_multiple_of(txw) && uh.is_multiple_of(txh),
        "tx_size must tile plane_bsize evenly"
    );
    let area = aom_txb::txb_wide(tx_size) * aom_txb::txb_high(tx_size);
    let mut above = vec![0i8; uw];
    let mut left = vec![0i8; uh];
    let mut coeffs = Vec::new();
    let mut blk_row = 0;
    while blk_row < uh {
        let mut blk_col = 0;
        while blk_col < uw {
            let (txb_skip_ctx, dc_sign_ctx) = aom_txb::get_txb_ctx(
                plane_bsize,
                tx_size,
                plane,
                &above[blk_col..],
                &left[blk_row..],
            );
            let mut tcoeff = vec![0i32; area];
            let (eob, tx_type) = aom_txb::read_coeffs_txb_full(
                dec,
                cdfs,
                ext_tx_cdf,
                &mut tcoeff,
                tx_size,
                plane,
                txb_skip_ctx as usize,
                dc_sign_ctx as usize,
                allow_update_cdf,
                ttx.is_inter,
                ttx.reduced,
                ttx.signal_gate,
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

/// Residual reconstruction (dequant + inverse transform + add) — moved to the
/// shared [`aom_recon`] crate and re-exported here. See
/// [`aom_recon::reconstruct_txb`].
pub use aom_recon::reconstruct_txb;
