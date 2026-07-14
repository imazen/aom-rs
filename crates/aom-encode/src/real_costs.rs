//! Derive the search's mode/partition/tx-size/cfl/tx-type/coeff RD cost
//! tables from a LIVE `KfFrameContext` (the same CDFs the entropy coder
//! reads/adapts) -- the encoder-gate wiring `av1_fill_mode_rates` +
//! `av1_fill_coeff_costs` (rd.c) perform at KEY frame init, so
//! `rd_pick_partition_real`/the leaf search sees the SAME costs aomenc's RD
//! decisions are actually made against, not the synthetic-but-valid random
//! tables `pack_tile_roundtrip.rs`'s baseline case uses to verify pack glue
//! only.
//!
//! **Scope (labelled):** covers every cost table that is genuinely
//! SINGLE-instance-per-frame in real AV1 (mode/partition/skip/tx-size/cfl/
//! intra-tx-type) -- these match the CURRENT `PickFrameCfg`/`SbEncodeEnv`
//! architecture with zero signature changes -- **plus** the coefficient-
//! coding cost tables (`LV_MAP_COEFF_COST` + `LV_MAP_EOB_COST`,
//! `av1_fill_coeff_costs` in rd.c), which real AV1 keeps PER-(`txs_ctx`,
//! `plane`) -- 5 tx-size categories x 2 plane types for the coeff tables,
//! plus a SEPARATE 7-way `eob_multi_size` x 2 plane axis for the eob-position
//! table (`aom_txb::CoeffCostSet`, `crate::encode_sb::SbEncodeEnv::
//! coeff_costs_y`/`coeff_costs_uv` and every `TxfmYrdEnv`/`UvRdEnv`/
//! `rd_pick_intra_mode_sb` field/parameter that threads them are now
//! `&CoeffCostSet`, selecting the real per-tx_size table at each txb via
//! `CoeffCostSet::tables(tx_size)` -- closing the gap the previous chunk left
//! open (single representative `txs_ctx` + zeroed eob costs).
//!
//! `av1_fill_mode_rates`'s inter-only slices (comp_ref/single_ref/newmv/...,
//! gated `if (!frame_is_intra_only(cm))` in the C) are out of scope for this
//! KEY-frame-only encoder envelope. `mbmode_cost` (`y_mode_cdf`, the non-KEY
//! intra-in-inter-frame Y mode) has no `KfFrameContext` field and is
//! confirmed dead in the current KEY-frame search (`y_mode_costs` is what
//! `rd_pick_intra_sby_mode_y` actually reads) -- filled from a degenerate
//! all-zero placeholder CDF. Likewise `TxTypeCosts::inter` has no consumer
//! in this intra-only pipeline.

use crate::mode_costs::{
    CflCosts, EXT_PARTITION_TYPES, IntraModeCosts, PARTITION_CONTEXTS, SKIP_CONTEXTS, TxSizeCosts,
    fill_cfl_costs, fill_intra_mode_costs, fill_partition_costs, fill_skip_costs,
    fill_tx_size_costs,
};
use aom_entropy::partition::KfFrameContext;
use aom_txb::{CoeffCostSet, TxTypeCosts, fill_coeff_cost_set_from_arena, fill_tx_type_costs};

/// `EXT_TX_SETS_INTRA` / `EXT_TX_SIZES` / `INTRA_MODES` / `TX_TYPES` (aom-txb
/// `ext_tx` module) restated here since they're not re-exported by name.
const EXT_TX_SETS_INTRA: usize = 3;
const EXT_TX_SIZES: usize = 4;
const INTRA_MODES: usize = 13;
const TX_TYPES: usize = 16;
/// `EXT_TX_SETS_INTER` (aom-txb `ext_tx`), for the dummy inter_cdf shape.
const EXT_TX_SETS_INTER: usize = 4;

/// Every real-cost table the KEY-frame intra search + pack stage reads,
/// owned (the fill functions write in place; some are boxed to avoid a large
/// stack frame, matching `IntraModeCosts`/`TxTypeCosts`'s own `zeroed()`).
pub struct RealCosts {
    pub mode_costs: Box<IntraModeCosts>,
    pub tx_size_costs: TxSizeCosts,
    pub tx_type_costs_y: Box<TxTypeCosts>,
    pub cfl_costs: CflCosts,
    pub partition_costs: [[i32; EXT_PARTITION_TYPES]; PARTITION_CONTEXTS],
    pub skip_costs: [[i32; 2]; SKIP_CONTEXTS],
    /// The REAL per-`(txs_ctx, eob_multi_size)` luma coefficient-coding cost
    /// tables (`av1_fill_coeff_costs`'s `plane == PLANE_TYPE_Y` slice).
    pub coeff_costs_y: CoeffCostSet,
    /// The REAL per-`(txs_ctx, eob_multi_size)` chroma coefficient-coding
    /// cost tables (`av1_fill_coeff_costs`'s `plane == PLANE_TYPE_UV` slice;
    /// shared by both U and V, matching real AV1's single UV plane-type).
    pub coeff_costs_uv: CoeffCostSet,
}

/// A degenerate but VALID single-symbol CDF row: `cdf[0] == 0` terminates
/// `cost_tokens_from_cdf`'s scan immediately (AOM_ICDF(0) == CDF_PROB_TOP,
/// the real AV1 "last real entry" convention every valid CDF satisfies at
/// its own N-1), so this is safe filler for slots the current KEY-frame-only
/// intra search never reads (never an out-of-bounds scan).
fn zero_cdf_row(n: usize) -> Vec<u16> {
    vec![0u16; n]
}

/// Repack `KfFrameContext::ext_tx_1ddct` (eset 1, 7-symbol, 8-wide) and
/// `ext_tx_dtt4` (eset 2, 5-symbol, 6-wide) into the uniform-stride
/// `[EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES][TX_TYPES+1]` flat layout
/// `aom_txb::fill_tx_type_costs` expects. eset 0 (DCT-only, codes nothing)
/// is filled with the same zero-row filler -- `fill_tx_type_costs`'s `for s
/// in 1..EXT_TX_SETS_INTRA` loop never reads set index 0. The narrower rows'
/// own terminators (`ext_tx_*_cdf[..][N-1] == 0`) land within the first 8/6
/// of the 17-wide slot, so the zero-padded tail is never scanned either.
fn repack_intra_ext_tx_cdf(kf: &KfFrameContext) -> Vec<u16> {
    let stride = TX_TYPES + 1; // 17
    let mut out = vec![0u16; EXT_TX_SETS_INTRA * EXT_TX_SIZES * INTRA_MODES * stride];
    for tx_idx in 0..EXT_TX_SIZES {
        for mode in 0..INTRA_MODES {
            let base1 = ((EXT_TX_SIZES + tx_idx) * INTRA_MODES + mode) * stride; // eset 1
            out[base1..base1 + 8].copy_from_slice(&kf.ext_tx_1ddct[tx_idx][mode]);
            let base2 = ((2 * EXT_TX_SIZES + tx_idx) * INTRA_MODES + mode) * stride;
            out[base2..base2 + 6].copy_from_slice(&kf.ext_tx_dtt4[tx_idx][mode]);
        }
    }
    out
}

/// `av1_fill_mode_rates` (rd.c), the KEY-frame-relevant slices only: derive
/// every table in [`RealCosts`] from `kf`'s live CDFs.
/// `enable_filter_intra` must equal the sequence header flag the search's
/// `PickFrameCfg::enable_filter_intra` / `filter_intra_allowed_bsize` gate
/// uses (same value threaded to both).
pub fn derive_real_costs(kf: &KfFrameContext, enable_filter_intra: bool) -> RealCosts {
    let mut mode_costs = IntraModeCosts::zeroed();
    let kf_y_cdf: Vec<u16> = kf.kf_y.iter().flatten().flatten().copied().collect();
    let uv_mode_cdf: Vec<u16> = kf.uv_mode.iter().flatten().flatten().copied().collect();
    let filter_intra_cdfs: Vec<u16> = kf.filter_intra.iter().flatten().copied().collect();
    let palette_y_mode_cdf: Vec<u16> = kf
        .palette_y_mode
        .iter()
        .flatten()
        .flatten()
        .copied()
        .collect();
    let angle_delta_cdf: Vec<u16> = kf.angle_delta.iter().flatten().copied().collect();
    // mbmode_cost (non-KEY intra Y mode) has no KfFrameContext CDF and is
    // unread by the KEY-frame search (see module docs) -- degenerate filler.
    let y_mode_cdf = zero_cdf_row(4 * (INTRA_MODES + 1));
    fill_intra_mode_costs(
        &mut mode_costs,
        &kf_y_cdf,
        &y_mode_cdf,
        &uv_mode_cdf,
        &kf.filter_intra_mode,
        &filter_intra_cdfs,
        &palette_y_mode_cdf,
        &angle_delta_cdf,
        &kf.intrabc,
        enable_filter_intra,
    );

    let mut cfl_costs = CflCosts::zeroed();
    let cfl_alpha_cdf: Vec<u16> = kf.cfl_alpha.iter().flatten().copied().collect();
    fill_cfl_costs(&mut cfl_costs, &kf.cfl_sign, &cfl_alpha_cdf);

    let mut tx_size_costs = TxSizeCosts::zeroed();
    let tx_size_cdf: Vec<u16> = kf.tx_size.iter().flatten().flatten().copied().collect();
    fill_tx_size_costs(&mut tx_size_costs, &tx_size_cdf);

    let mut tx_type_costs_y = TxTypeCosts::zeroed();
    let intra_ext_tx_cdf = repack_intra_ext_tx_cdf(kf);
    // Inter tx-type costs have no consumer in this intra-only pipeline.
    let inter_ext_tx_cdf = zero_cdf_row(EXT_TX_SETS_INTER * EXT_TX_SIZES * (TX_TYPES + 1));
    fill_tx_type_costs(&mut tx_type_costs_y, &intra_ext_tx_cdf, &inter_ext_tx_cdf);

    let mut partition_costs = [[0i32; EXT_PARTITION_TYPES]; PARTITION_CONTEXTS];
    let partition_cdf: Vec<u16> = kf.partition.iter().flatten().copied().collect();
    fill_partition_costs(&mut partition_costs, &partition_cdf);

    let mut skip_costs = [[0i32; 2]; SKIP_CONTEXTS];
    let skip_cdf: Vec<u16> = kf.skip.iter().flatten().copied().collect();
    fill_skip_costs(&mut skip_costs, &skip_cdf);

    // av1_fill_coeff_costs: the full per-(txs_ctx, eob_multi_size) real
    // LV_MAP_COEFF_COST / LV_MAP_EOB_COST tables, one CoeffCostSet per plane
    // type, sliced directly from the live coefficient-CDF arena (the SAME
    // bytes write_coeffs_txb/cost_coeffs_txb read/adapt).
    let coeff_costs_y = fill_coeff_cost_set_from_arena(&kf.coeff, 0);
    let coeff_costs_uv = fill_coeff_cost_set_from_arena(&kf.coeff, 1);

    RealCosts {
        mode_costs,
        tx_size_costs,
        tx_type_costs_y,
        cfl_costs,
        partition_costs,
        skip_costs,
        coeff_costs_y,
        coeff_costs_uv,
    }
}
