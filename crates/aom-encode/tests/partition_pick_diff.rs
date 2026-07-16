//! Differential harness for `rd_pick_partition_real` — the NONE-vs-SPLIT
//! recursion with REAL whole-block leaves + the winner-subtree DRY_RUN
//! `encode_sb` propagation — vs an independent transcription of
//! `av1_rd_pick_partition`'s control flow (partition_search.c:5653) where
//! every derivation/primitive the recursion ADDS is the REAL C:
//!
//! - budget math via the REAL `av1_rd_cost_update` /
//!   `av1_rd_stats_subtraction` (`ref_rd_cost_update` /
//!   `ref_rd_stats_subtraction`);
//! - `partition_plane_context` via the REAL facade
//!   (`ref_partition_plane_context`) over the C-side live partition arrays;
//! - per-leaf input derivations via REAL C: `av1_get_perpixel_variance`
//!   (`ref_hbd_variance` + the pel shift), the HOG mask
//!   (`ref_prune_intra_mode_with_hog_y` at the KEY thresh -1.2),
//!   `get_tx_size_context` (`ref_get_tx_size_context`) over the C-side live
//!   txfm arrays;
//! - winner propagation via the item-2 REAL-piece `encode_sb` oracle
//!   (`common::COracle` — REAL leaf chains + REAL context-stamp facades);
//! - `av1_save_context`/`av1_restore_context` + the
//!   `should_do_dry_run_encode_for_current_block` gate transcribed
//!   (simple slice copies / a 4-line predicate, cited in the module).
//!
//! The whole-block leaf EVALUATION itself is the ported
//! `rd_pick_intra_mode_sb` on BOTH sides (bit-exact vs the full C chain by
//! rd_pick_intra_sb_diff) with each side constructing the leaf's INPUTS from
//! its own state mirrors — so a wrong offset/context read on either side
//! diverges. What this harness pins: recursion order + budget threading
//! (the FULL leaf-visit sequence with budgets), the NONE/SPLIT strict-<
//! gates, save/restore placement, the dry-run gating (index-3 skip), and the
//! REAL sibling pixel/context/mode dependency through the propagation walks.
//!
//! Coverage note: cases use `min_partition_size >= 8x8` (every leaf is a
//! chroma reference), so the CfL context is per-leaf scratch on both sides.
//! The sub-8x8 shared-chroma propagation is validated at the encode_sb layer
//! (encode_sb_diff); threading a search-side CfL mirror through the C oracle
//! for 4x4 recursion leaves is the documented next step.
//!
//! Sweeps: 420/444, bd 8, q {64,128,200}, ALLINTRA/GOOD arms (chroma trellis
//! table + the leaf variance-factor arm), min partition {16x16, 8x8},
//! quadrant-mixed content (flat / detailed) so NONE and SPLIT genuinely
//! trade wins.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::{SbEncodeEnv, SbTree, TileCtxState};
use aom_encode::hog::prune_intra_mode_with_hog_y as _rust_hog; // (symmetry doc)
use aom_encode::intra_rd::{Block4x4VarInfo, IntraSbyGates, IntraSbySearchCfg};
use aom_encode::intra_uv_rd::{
    UvLoopPolicy, UvRdEnv, av1_get_tx_size_uv, chroma_plane_offset, is_chroma_reference,
};
use aom_encode::mode_costs::{CflCosts, IntraModeCosts, TxSizeCosts, fill_cfl_costs};
use aom_encode::partition::PartRdStats;
use aom_encode::partition_pick::{
    LeafVisit, ModeGrid, PickFrameCfg, perpixel_variance_y, rd_pick_partition_real,
};
use aom_encode::rd_pick::{RdPickUvArgs, RdPickUvOutcome, ReencodeParams, rd_pick_intra_mode_sb};
use aom_encode::tx_search::{TxTypeSearchPolicy, TxfmYrdEnv};
use aom_intra::cfl::CflCtx;
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_txb::{CoeffCostSet, TxTypeCosts, fill_tx_type_costs};

mod common;
use common::*;

const STRIDE: usize = 256;
/// The diff harness encodes at mi(0,0) in a 512x512 mi frame, so every tested
/// block is interior (`has_rows && has_cols`) and the frame-EDGE partition-cost
/// override never fires — `PickFrameCfg::partition_cdfs` is never read here.
const UNUSED_EDGE_PARTITION_CDF: [[u16; 11]; 20] = [[0u16; 11]; 20];
const MI_WB: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_HB: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
const BLK_1D: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];

fn c_split_sub(bsize: usize) -> usize {
    match bsize {
        3 => 0,
        6 => 3,
        9 => 6,
        12 => 9,
        _ => unreachable!(),
    }
}

/// The C-side recursion state: the item-2 REAL-piece walk oracle + the
/// C-side mode grid + the shared leaf tables.
struct CPick<'a> {
    o: COracle<'a>,
    grid: Vec<u8>,
    /// Parallel UV-mode grid (same stride as `grid`) — the C-recursion
    /// reference for the per-block chroma intra edge filter
    /// (`get_intra_edge_filter_type(xd, plane=1)`). Stamped with each winner's
    /// `uv_mode` wherever `grid` is stamped with `mode`, so a chroma-ref
    /// leaf's SMOOTH-UV-neighbour check mirrors production exactly.
    uv_grid: Vec<u8>,
    grid_stride: usize,
    // Leaf-search tables (SHARED values with the Rust side — the tables are
    // fixture inputs; the DERIVED inputs are computed per side).
    mode_costs: &'a IntraModeCosts,
    tx_size_costs: &'a TxSizeCosts,
    skip_costs: &'a [[i32; 2]; 3],
    tx_type_costs_y: &'a TxTypeCosts,
    pol: &'a TxTypeSearchPolicy,
    uv_lp: &'a UvLoopPolicy,
    intra_uv_mode_cost: &'a [[[i32; 14]; 13]; 2],
    cfl_costs: &'a CflCosts,
    partition_costs: &'a [[i32; 10]; 20],
    allintra: bool,
    speed: i32,
    qindex_cfg: i32,
    // Rust-side quant rows (the leaf search consumes the Rust rows; the
    // encode oracle consumes the C rows inside `o`).
    rows_y: &'a aom_quant::PlaneQuantRows<'a>,
    rows_u: &'a aom_quant::PlaneQuantRows<'a>,
    rows_v: &'a aom_quant::PlaneQuantRows<'a>,
    // Frame-level (multi-txs_ctx) sets, mirroring SbEncodeEnv::coeff_costs_y/
    // _uv -- the recursion visits many leaf bsizes/tx_sizes, so this cannot
    // be a single pre-selected CoeffCostTables (luma) the way EncodeIntraYEnv/
    // the winner re-encode can be. UvRdEnv's per-leaf construction below
    // still pre-selects its OWN single CoeffCostTables via
    // `coeff_costs_uv.tables(uv_tx_size)`, matching `UvRdEnv::coeff_costs`'s
    // pre-selected-by-caller contract (chroma has no tx-size depth search).
    coeff_costs_y: &'a CoeffCostSet,
    coeff_costs_uv: &'a CoeffCostSet,
    ttc_dummy: &'a TxTypeCosts,
    max_partition_size: usize,
    min_partition_size: usize,
    sb_size: usize,
    monochrome: bool,
    lossless: bool,
    enable_optimize_b: TrellisOptType,
    enable_rect_partitions: bool,
    /// sf less_rectangular_check_level (ALLINTRA 1 / GOOD 0 at speed 0).
    less_rectangular_check_level: i32,
    /// Stage-coverage counters (C-side instrumentation; outputs are
    /// separately asserted equal to the Rust side).
    stats: CPickStats,
}

/// C-side coverage counters for the stage arms this harness must
/// genuinely exercise.
#[derive(Default)]
struct CPickStats {
    /// ALLINTRA var arm force-split firings (:5814-5818).
    var_force_split: usize,
    /// less_rectangular_check rect kills (:4630-4640).
    less_rect_kills: usize,
    /// rect types entered (is_rect_part_allowed passed).
    rect_evals: usize,
    /// rect mid-stage sub-0 propagation encodes (:3613-3616).
    rect_mid_encodes: usize,
    /// HORZ / VERT winners picked.
    rect_wins: [usize; 2],
}

impl CPick<'_> {
    /// pick_sb_modes over the C-side mirrors: REAL derivations
    /// (variance/HOG/tx-size-ctx) + the validated whole-block leaf.
    /// `partition` = the `mbmi->partition = partition` install (:887).
    #[allow(clippy::too_many_arguments)]
    fn leaf(
        &mut self,
        recon_y: &mut [u16],
        recon_u: &mut [u16],
        recon_v: &mut [u16],
        cfl: &mut CflCtx,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
        best_remain: (i32, i64, i64),
    ) -> (PartRdStats, Option<aom_encode::encode_sb::LeafWinner>) {
        // av1_rd_cost_update on entry (REAL).
        let (_r, _d, budget) =
            c::ref_rd_cost_update(self.o.rdmult, best_remain.0, best_remain.1, best_remain.2);

        let mi_w = MI_WB[bsize];
        let mi_h = MI_HB[bsize];
        let up = mi_row > 0;
        let left = mi_col > 0;
        let is_cref = c::ref_is_chroma_reference(
            mi_row,
            mi_col,
            bsize,
            self.o.ss.0 as i32,
            self.o.ss.1 as i32,
        );
        let ref_off_y = self.o.base_y + (mi_row as usize * 4) * STRIDE + mi_col as usize * 4;
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;

        // REAL av1_get_perpixel_variance: vf vs the flat offs buffer.
        // BSIZE_VAR_IDX: AV1 BLOCK_SIZE -> the dims-sorted shim_hbd_var
        // kernel index (fixes the chunk-7 harness slip of passing the enum
        // straight through — wrong dims for every size above 4x8; the
        // tx-depth low-contrast gate `source_variance < 256` is live).
        const BSIZE_VAR_IDX: [usize; 22] = [
            0, 1, 3, 4, 5, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 21, 2, 7, 6, 12, 11, 16,
        ];
        let offs = vec![128u16 << (self.o.bd - 8); 128];
        let (var, _sse) = c::ref_hbd_variance(
            BSIZE_VAR_IDX[bsize],
            self.o.bd,
            &self.o.src_y[ref_off_y..],
            STRIDE,
            &offs,
            0,
        );
        const NUM_PELS_LOG2: [u32; 22] = [
            4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10,
        ];
        let bits = NUM_PELS_LOG2[bsize];
        let source_variance = (var + (1 << (bits - 1))) >> bits;

        // REAL HOG mask at the KEY thresh -1.2.
        let mb_right = (self.o.mi_cols - mi_w as i32 - mi_col) * 4 * 8;
        let mb_bottom = (self.o.mi_rows - mi_h as i32 - mi_row) * 4 * 8;
        let skip_mask = c::ref_prune_intra_mode_with_hog_y(
            self.o.src_y,
            ref_off_y,
            STRIDE,
            bsize,
            mb_right,
            mb_bottom,
            self.o.bd,
            -1.2,
        );
        let gates = IntraSbyGates::speed0(skip_mask);

        let above_mode = if up {
            Some(i32::from(
                self.grid[(mi_row - 1) as usize * self.grid_stride + mi_col as usize],
            ))
        } else {
            None
        };
        let left_mode = if left {
            Some(i32::from(
                self.grid[mi_row as usize * self.grid_stride + (mi_col - 1) as usize],
            ))
        } else {
            None
        };

        // Luma intra edge filter type (reconintra.c get_intra_edge_filter_type):
        // 1 iff the above OR left neighbour is a SMOOTH mode (9/10/11). Mirrors
        // partition_pick.rs's production recompute so this C-recursion reference
        // stays faithful to C (which re-derives it per block from the live grid).
        let is_smooth_luma = |m: Option<i32>| m.is_some_and(|md| (9..=11).contains(&md));
        let luma_edge_filter_type =
            i32::from(is_smooth_luma(above_mode) || is_smooth_luma(left_mode));

        // REAL get_tx_size_context over the C-side live txfm arrays
        // (KEY intra: no inter neighbours).
        let tx_size_ctx = c::ref_get_tx_size_context(
            bsize,
            self.o.above_t[a0],
            self.o.left_t[l0],
            up,
            left,
            0,
            false,
            0,
            false,
        ) as usize;

        let above_y: Vec<i8> = self.o.above_e[0][a0..a0 + mi_w].to_vec();
        let left_y: Vec<i8> = self.o.left_e[0][l0..l0 + mi_h].to_vec();
        let mut y_env = TxfmYrdEnv {
            sb_size: self.sb_size,
            bsize,
            mi_row,
            mi_col,
            up_available: up,
            left_available: left,
            tile_col_end: 1 << 16,
            tile_row_end: 1 << 16,
            partition,
            mi_cols: self.o.mi_cols,
            mi_rows: self.o.mi_rows,
            ref_off: ref_off_y,
            ref_stride: STRIDE,
            src: self.o.src_y,
            src_off: ref_off_y,
            src_stride: STRIDE,
            disable_edge_filter: false,
            filter_type: luma_edge_filter_type,
            mode: 0,
            angle_delta: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            lossless: self.lossless,
            reduced_tx_set_used: self.o.reduced,
            bd: self.o.bd,
            rows: self.rows_y,
            rdmult: self.o.rdmult,
            coeff_costs: self.coeff_costs_y,
            tx_type_costs: self.tx_type_costs_y,
            skip_costs: self.skip_costs,
            skip_ctx: 0,
            tx_size_costs: self.tx_size_costs,
            tx_size_ctx,
            tx_mode_is_select: true,
            above_ctx: &above_y,
            left_ctx: &left_y,
        };
        let sby_cfg = IntraSbySearchCfg {
            gates: &gates,
            top_intra_model_count_allowed: 4,
            adapt_top_model_rd_count_using_neighbors: false,
            above_mode,
            left_mode,
            qindex: self.qindex_cfg,
            mode_costs: self.mode_costs,
            try_palette: false,
            palette_bsize_ctx: 0,
            palette_mode_ctx: 0,
            enable_filter_intra: true,
            allow_intrabc: false,
            pol: self.pol,
            source_variance,
            enable_tx64: true,
            enable_rect_tx: true,
            allintra: self.allintra,
            speed: self.speed,
            mb_to_right_edge: mb_right,
            mb_to_bottom_edge: mb_bottom,
            winner_mode: None,
        };
        let mut var_cache = Block4x4VarInfo::sb_cache(self.sb_size);

        let ss_x = self.o.ss.0;
        let ss_y = self.o.ss.1;
        let ref_off_uv =
            chroma_plane_offset(self.o.base_uv, STRIDE, mi_row, mi_col, bsize, ss_x, ss_y);
        let mut c_up = up;
        let mut c_left = left;
        if ss_x != 0 && mi_w < 2 {
            c_left = (mi_col - 1) > 0;
        }
        if ss_y != 0 && mi_h < 2 {
            c_up = (mi_row - 1) > 0;
        }
        let plane_bsize = aom_entropy::partition::get_plane_block_size(bsize, ss_x, ss_y);
        let (pmw, pmh) = (MI_WB[plane_bsize], MI_HB[plane_bsize]);
        let au = (mi_col >> ss_x) as usize;
        let lu = ((mi_row & 31) >> ss_y) as usize;
        let above_u: Vec<i8> = self.o.above_e[1][au..au + pmw].to_vec();
        let left_u: Vec<i8> = self.o.left_e[1][lu..lu + pmh].to_vec();
        let above_v: Vec<i8> = self.o.above_e[2][au..au + pmw].to_vec();
        let left_v: Vec<i8> = self.o.left_e[2][lu..lu + pmh].to_vec();
        let cfl_allowed = !self.lossless && BLK_1D[bsize] <= 32 && BLK_1D[bsize] <= 32 && {
            const BLK_H: [usize; 22] = [
                4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
            ];
            BLK_H[bsize] <= 32
        };
        // Chroma has no tx-size depth search -- pre-select the ONE real
        // per-txs_ctx table THIS leaf's uv_tx_size uses (mirrors
        // partition_pick.rs::leaf_pick_sb_modes's fix).
        let uv_tx_size = av1_get_tx_size_uv(bsize, self.lossless, ss_x, ss_y);
        let uv_coeff_tables = self.coeff_costs_uv.tables(uv_tx_size);
        // Per-block chroma intra edge filter type — the C-recursion analogue of
        // the luma `luma_edge_filter_type` above, mirroring
        // partition_pick.rs::leaf_pick_sb_modes exactly: 1 iff the available
        // above/left chroma neighbour's `uv_mode` is SMOOTH (9/10/11). Chroma
        // neighbour mi (av1_common_int.h:1400-1416): from `base = (mi_row -
        // (mi_row & ss_y), mi_col - (mi_col & ss_x))`, above = base + (-1,+ss_x),
        // left = base + (+ss_y,-1).
        let is_smooth_uv = |uvm: u8| (9..=11).contains(&uvm);
        let base_row = mi_row - (mi_row & ss_y as i32);
        let base_col = mi_col - (mi_col & ss_x as i32);
        let uv_mode_at = |r: i32, cc: i32| -> u8 {
            if r >= 0 && cc >= 0 && r < self.o.mi_rows && cc < self.o.mi_cols {
                self.uv_grid[r as usize * self.grid_stride + cc as usize]
            } else {
                0
            }
        };
        let chroma_edge_filter_type = i32::from(
            (c_up && is_smooth_uv(uv_mode_at(base_row - 1, base_col + ss_x as i32)))
                || (c_left && is_smooth_uv(uv_mode_at(base_row + ss_y as i32, base_col - 1))),
        );
        let mut uv_env = UvRdEnv {
            sb_size: self.sb_size,
            bsize,
            mi_row,
            mi_col,
            chroma_up_available: c_up,
            chroma_left_available: c_left,
            tile_col_end: 1 << 16,
            tile_row_end: 1 << 16,
            partition,
            mi_cols: self.o.mi_cols,
            mi_rows: self.o.mi_rows,
            ss_x,
            ss_y,
            ref_off: [ref_off_uv, ref_off_uv],
            ref_stride: STRIDE,
            src_u: self.o.src_u,
            src_v: self.o.src_v,
            src_off: [ref_off_uv, ref_off_uv],
            src_stride: STRIDE,
            disable_edge_filter: false,
            filter_type: chroma_edge_filter_type,
            luma_mode: 0,
            luma_use_fi: false,
            luma_fi_mode: 0,
            lossless: self.lossless,
            reduced_tx_set_used: self.o.reduced,
            bd: self.o.bd,
            rows_u: self.rows_u,
            rows_v: self.rows_v,
            rdmult: self.o.rdmult,
            coeff_costs: &uv_coeff_tables,
            tx_type_costs: self.ttc_dummy,
            above_ctx: [&above_u, &above_v],
            left_ctx: [&left_u, &left_v],
        };
        let re = ReencodeParams {
            sharpness: self.o.sharpness,
            enable_optimize_b: self.enable_optimize_b,
        };
        let outcome = rd_pick_intra_mode_sb(
            &mut y_env,
            recon_y,
            &sby_cfg,
            &mut var_cache,
            budget,
            self.coeff_costs_y,
            re,
            if self.monochrome {
                None
            } else {
                Some(RdPickUvArgs {
                    env: &mut uv_env,
                    recon_u,
                    recon_v,
                    cfl,
                    is_chroma_ref: is_cref,
                    cfl_allowed,
                    intra_uv_mode_cost: self.intra_uv_mode_cost,
                    costs: self.mode_costs,
                    cfl_costs: self.cfl_costs,
                    lp: self.uv_lp,
                })
            },
        );
        match outcome.best {
            None => (PartRdStats::invalid(), None),
            Some(best) => {
                let stats = PartRdStats {
                    rate: best.rate,
                    dist: best.dist,
                    rdcost: best.rdcost,
                };
                let (uv_mode, ad_uv, ci, cs) = match &best.uv {
                    RdPickUvOutcome::Searched(w, _) => (
                        w.uv_mode,
                        w.angle_delta_uv,
                        i32::from(w.cfl_alpha_idx),
                        i32::from(w.cfl_alpha_signs),
                    ),
                    _ => (0, 0, 0, 0),
                };
                let winner = aom_encode::encode_sb::LeafWinner {
                    bsize,
                    mode: best.y.mode,
                    angle_delta_y: best.y.angle_delta,
                    use_filter_intra: best.y.use_filter_intra,
                    filter_intra_mode: best.y.filter_intra_mode,
                    tx_size: best.y.tx_size,
                    luma_edge_filter_type,
                    uv_mode,
                    angle_delta_uv: ad_uv,
                    cfl_alpha_idx: ci,
                    cfl_alpha_signs: cs,
                    uv_edge_filter_type: chroma_edge_filter_type,
                    tx_type_map: best.tx_type_map,
                    skip_txfm: false,
                    raw_rdstats: stats,
                };
                (stats, Some(winner))
            }
        }
    }

    /// The transcribed av1_rd_pick_partition (NONE + SPLIT + HORZ + VERT)
    /// over the C-side mirrors — every ADDED derivation through REAL C
    /// primitives (module docs).
    #[allow(clippy::too_many_arguments)]
    fn pick(
        &mut self,
        recon_y: &mut [u16],
        recon_u: &mut [u16],
        recon_v: &mut [u16],
        cfl_search: &mut CflCtx,
        cfl_enc: &mut c::RefCflState,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        mut best_rdc: PartRdStats,
        pc_index: usize,
        visits: &mut Vec<LeafVisit>,
    ) -> (Option<SbTree>, PartRdStats, bool) {
        if best_rdc.rdcost < 0 {
            return (None, PartRdStats::invalid(), false);
        }
        let mi_w = MI_WB[bsize];
        let mi_step = (mi_w / 2) as i32;
        let bsize_at_least_8x8 = bsize >= 3;
        let has_rows = mi_row + mi_step < self.o.mi_rows;
        let has_cols = mi_col + mi_step < self.o.mi_cols;
        let mut partition_none_allowed = has_rows && has_cols;
        let mut do_square_split = bsize_at_least_8x8;
        // Rect flag init (:3382-3399): REAL get_partition_subsize + REAL
        // get_plane_block_size (the chroma-validity guard).
        let mut do_rectangular_split = self.enable_rect_partitions && bsize_at_least_8x8;
        let mut partition_rect_allowed = [false; 2];
        if do_rectangular_split {
            let horz_sub = c::ref_get_partition_subsize(bsize as i32, 1) as usize;
            let vert_sub = c::ref_get_partition_subsize(bsize as i32, 2) as usize;
            partition_rect_allowed[0] =
                has_cols && c::ref_get_plane_block_size(horz_sub, self.o.ss.0, self.o.ss.1) != 255;
            partition_rect_allowed[1] =
                has_rows && c::ref_get_plane_block_size(vert_sub, self.o.ss.0, self.o.ss.1) != 255;
        }
        if BLK_1D[bsize] > BLK_1D[self.max_partition_size] {
            // av1_set_square_split_only.
            partition_none_allowed = false;
            do_square_split = true;
            do_rectangular_split = false;
            partition_rect_allowed = [false, false];
        } else if BLK_1D[bsize] <= BLK_1D[self.min_partition_size] {
            // av1_disable_rect_partitions + the le-min square clamp.
            do_rectangular_split = false;
            partition_rect_allowed = [false, false];
            if has_rows && has_cols {
                do_square_split = false;
            }
            partition_none_allowed = !do_square_split;
        }

        // REAL partition_plane_context over the live C-side arrays.
        let pl_ctx = if bsize_at_least_8x8 {
            let mut above64 = [0i8; 64];
            above64.copy_from_slice(&self.o.above_p);
            c::ref_partition_plane_context(&above64, &self.o.left_p, mi_row, mi_col, bsize as i32)
                as usize
        } else {
            0
        };
        let partition_cost = &self.partition_costs[pl_ctx];

        // REAL av1_rd_cost_update.
        let (r, d, cst) =
            c::ref_rd_cost_update(self.o.rdmult, best_rdc.rate, best_rdc.dist, best_rdc.rdcost);
        best_rdc = PartRdStats {
            rate: r,
            dist: d,
            rdcost: cst,
        };

        // av1_save_context (transcribed slice copies).
        let saved_e: Vec<Vec<i8>> = (0..3)
            .map(|p| {
                let (s0, w) = if p == 0 {
                    (mi_col as usize, mi_w)
                } else {
                    ((mi_col >> self.o.ss.0) as usize, mi_w >> self.o.ss.0)
                };
                self.o.above_e[p][s0..s0 + w].to_vec()
            })
            .collect();
        let saved_le: Vec<Vec<i8>> = (0..3)
            .map(|p| {
                let (s0, h) = if p == 0 {
                    ((mi_row & 31) as usize, MI_HB[bsize])
                } else {
                    (
                        ((mi_row & 31) >> self.o.ss.1) as usize,
                        MI_HB[bsize] >> self.o.ss.1,
                    )
                };
                self.o.left_e[p][s0..s0 + h].to_vec()
            })
            .collect();
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;
        let saved_p: Vec<i8> = self.o.above_p[a0..a0 + mi_w].to_vec();
        let saved_lp: Vec<i8> = self.o.left_p[l0..l0 + MI_HB[bsize]].to_vec();
        let saved_t: Vec<u8> = self.o.above_t[a0..a0 + mi_w].to_vec();
        let saved_lt: Vec<u8> = self.o.left_t[l0..l0 + MI_HB[bsize]].to_vec();
        let restore = |o: &mut COracle| {
            for p in 0..3 {
                let (s0, _) = if p == 0 { (a0, 0) } else { (a0 >> o.ss.0, 0) };
                o.above_e[p][s0..s0 + saved_e[p].len()].copy_from_slice(&saved_e[p]);
                let (t0, _) = if p == 0 { (l0, 0) } else { (l0 >> o.ss.1, 0) };
                o.left_e[p][t0..t0 + saved_le[p].len()].copy_from_slice(&saved_le[p]);
            }
            o.above_p[a0..a0 + saved_p.len()].copy_from_slice(&saved_p);
            o.left_p[l0..l0 + saved_lp.len()].copy_from_slice(&saved_lp);
            o.above_t[a0..a0 + saved_t.len()].copy_from_slice(&saved_t);
            o.left_t[l0..l0 + saved_lt.len()].copy_from_slice(&saved_lt);
        };

        // The per-node ALLINTRA variance arm (:5791-5827) via the REAL
        // log_sub_block_var facade — at speed 0 only the >= 16x16
        // force-split branch is live.
        if self.allintra && bsize >= 6 {
            let ref_off_y = self.o.base_y + (mi_row as usize * 4) * STRIDE + mi_col as usize * 4;
            let mb_right = (self.o.mi_cols - mi_w as i32 - mi_col) * 4 * 8;
            let mb_bottom = (self.o.mi_rows - MI_HB[bsize] as i32 - mi_row) * 4 * 8;
            let (var_min, var_max) = c::ref_log_sub_block_var(
                self.o.src_y,
                ref_off_y,
                STRIDE,
                bsize,
                mb_right,
                mb_bottom,
                self.o.bd,
            );
            if var_min < 0.272 && (var_max - var_min) > 3.0 {
                partition_none_allowed = false;
                do_square_split = true;
                self.stats.var_force_split += 1;
            }
        }

        let mut found = false;
        let mut best_tree: Option<SbTree> = None;
        // part_search_state->none_rd (:3366; stored PRE-pt_cost at :4458).
        let mut none_rd: i64 = 0;

        if partition_none_allowed {
            let mut pt_cost = 0i32;
            if bsize_at_least_8x8 {
                pt_cost = if partition_cost[0] < i32::MAX {
                    partition_cost[0]
                } else {
                    0
                };
            }
            let (_pr, _pd, ptc) = c::ref_rd_cost_update(self.o.rdmult, pt_cost, 0, 0);
            let _ = ptc;
            let best_remain = c::ref_rd_stats_subtraction(
                self.o.rdmult,
                (best_rdc.rate, best_rdc.dist, best_rdc.rdcost),
                (pt_cost, 0, ptc),
            );
            let (mut this_rdc, winner) = self.leaf(
                recon_y,
                recon_u,
                recon_v,
                cfl_search,
                mi_row,
                mi_col,
                bsize,
                0,
                best_remain,
            );
            visits.push(LeafVisit {
                mi_row,
                mi_col,
                bsize,
                budget: best_remain.2,
                rate: this_rdc.rate,
                dist: this_rdc.dist,
                rdcost: this_rdc.rdcost,
            });
            none_rd = this_rdc.rdcost;
            if this_rdc.rate != i32::MAX {
                if bsize_at_least_8x8 {
                    this_rdc.rate += pt_cost;
                    this_rdc.rdcost =
                        aom_encode::rd::rdcost(self.o.rdmult, this_rdc.rate, this_rdc.dist);
                }
                if this_rdc.rdcost < best_rdc.rdcost {
                    best_rdc = this_rdc;
                    found = true;
                    best_tree = Some(SbTree::Leaf(winner.unwrap()));
                }
            }
            restore(&mut self.o);
        }

        if do_square_split {
            let subsize = c_split_sub(bsize);
            let mut sum_rdc = PartRdStats::init();
            sum_rdc.rate = partition_cost[3];
            sum_rdc.rdcost = aom_encode::rd::rdcost(self.o.rdmult, sum_rdc.rate, 0);
            let mut children: Vec<Option<SbTree>> = Vec::new();
            let mut idx = 0usize;
            while idx < 4 && sum_rdc.rdcost < best_rdc.rdcost {
                let y = mi_row + ((idx as i32) >> 1) * mi_step;
                let x = mi_col + ((idx as i32) & 1) * mi_step;
                if y >= self.o.mi_rows || x >= self.o.mi_cols {
                    children.push(None);
                    idx += 1;
                    continue;
                }
                let best_remain = c::ref_rd_stats_subtraction(
                    self.o.rdmult,
                    (best_rdc.rate, best_rdc.dist, best_rdc.rdcost),
                    (sum_rdc.rate, sum_rdc.dist, sum_rdc.rdcost),
                );
                let (t, crdc, cfound) = self.pick(
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl_search,
                    cfl_enc,
                    y,
                    x,
                    subsize,
                    PartRdStats {
                        rate: best_remain.0,
                        dist: best_remain.1,
                        rdcost: best_remain.2,
                    },
                    idx,
                    visits,
                );
                if !cfound {
                    sum_rdc = PartRdStats::invalid();
                    children.push(t);
                    break;
                }
                sum_rdc.rate += crdc.rate;
                sum_rdc.dist += crdc.dist;
                let (nr, nd, nc) = c::ref_rd_cost_update(
                    self.o.rdmult,
                    sum_rdc.rate,
                    sum_rdc.dist,
                    sum_rdc.rdcost,
                );
                sum_rdc = PartRdStats {
                    rate: nr,
                    dist: nd,
                    rdcost: nc,
                };
                children.push(t);
                idx += 1;
            }
            if idx == 4 && sum_rdc.rdcost < best_rdc.rdcost {
                sum_rdc.rdcost = aom_encode::rd::rdcost(self.o.rdmult, sum_rdc.rate, sum_rdc.dist);
                if sum_rdc.rdcost < best_rdc.rdcost {
                    best_rdc = sum_rdc;
                    found = true;
                    let kids: Vec<SbTree> = children.into_iter().map(|t| t.unwrap()).collect();
                    best_tree = Some(SbTree::Split(Box::new(
                        <[SbTree; 4]>::try_from(kids).ok().unwrap(),
                    )));
                }
            } else if self.less_rectangular_check_level > 0 {
                // :4630-4640 (ALLINTRA level 1 at speed 0): kill rect when
                // the PRE-pt_cost NONE rdcost beat the split-stage sum.
                if self.less_rectangular_check_level == 2 || idx <= 2 {
                    let partition_none_valid = none_rd > 0;
                    let partition_none_better = none_rd < sum_rdc.rdcost;
                    if partition_none_valid && partition_none_better {
                        do_rectangular_split = false;
                        self.stats.less_rect_kills += 1;
                    }
                }
            }
            restore(&mut self.o);
        }

        // ---- rectangular partition stage (rectangular_partition_search,
        // :3520): REAL subsize/plane-size/budget/propagation primitives ----
        #[allow(clippy::needless_range_loop)] // C-transcription index loop
        for i in 0..2usize {
            // is_rect_part_allowed (:3506) + av1_active_h/v_edge at the
            // one-pass shape (top/left edge 0, bottom/right edge mi dims).
            let (mi_pos, dim_end) = if i == 0 {
                (mi_row, self.o.mi_rows)
            } else {
                (mi_col, self.o.mi_cols)
            };
            let active_edge = (0 >= mi_pos && 0 < mi_pos + mi_step)
                || (dim_end >= mi_pos && dim_end < mi_pos + mi_step);
            if !partition_rect_allowed[i] || !(do_rectangular_split || active_edge) {
                continue;
            }
            self.stats.rect_evals += 1;
            let partition_type = 1 + i; // PARTITION_HORZ / PARTITION_VERT
            let subsize =
                c::ref_get_partition_subsize(bsize as i32, partition_type as i32) as usize;
            let mut sum_rdc = PartRdStats::init();
            sum_rdc.rate = partition_cost[partition_type];
            sum_rdc.rdcost = aom_encode::rd::rdcost(self.o.rdmult, sum_rdc.rate, 0);

            // Sub-block 0 at the origin (rd_pick_rect_partition :3471).
            let mut w0 = {
                let best_remain = c::ref_rd_stats_subtraction(
                    self.o.rdmult,
                    (best_rdc.rate, best_rdc.dist, best_rdc.rdcost),
                    (sum_rdc.rate, sum_rdc.dist, sum_rdc.rdcost),
                );
                let (this_rdc, w) = self.leaf(
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl_search,
                    mi_row,
                    mi_col,
                    subsize,
                    partition_type,
                    best_remain,
                );
                visits.push(LeafVisit {
                    mi_row,
                    mi_col,
                    bsize: subsize,
                    budget: best_remain.2,
                    rate: this_rdc.rate,
                    dist: this_rdc.dist,
                    rdcost: this_rdc.rdcost,
                });
                if this_rdc.rate == i32::MAX {
                    sum_rdc.rdcost = i64::MAX;
                } else {
                    sum_rdc.rate += this_rdc.rate;
                    sum_rdc.dist += this_rdc.dist;
                    let (nr, nd, nc) = c::ref_rd_cost_update(
                        self.o.rdmult,
                        sum_rdc.rate,
                        sum_rdc.dist,
                        sum_rdc.rdcost,
                    );
                    sum_rdc = PartRdStats {
                        rate: nr,
                        dist: nd,
                        rdcost: nc,
                    };
                }
                w
            };

            let is_not_edge_block = if i == 0 { has_rows } else { has_cols };
            let mut w1: Option<aom_encode::encode_sb::LeafWinner> = None;
            if sum_rdc.rdcost < best_rdc.rdcost && is_not_edge_block {
                // Mid-stage propagation (:3613-3616): av1_update_state +
                // encode_superblock(DRY_RUN_NORMAL) of sub 0 through the
                // REAL-piece encode_b + the mi-grid stamp.
                let w0m = w0.as_mut().unwrap();
                let mut outs: Vec<CLeafOut> = Vec::new();
                self.o.encode_b(
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl_enc,
                    w0m,
                    mi_row,
                    mi_col,
                    partition_type,
                    &mut outs,
                    false, // DRY_RUN_NORMAL: tx_type_map resets persist (C ctx alias)
                );
                for rr in 0..MI_HB[subsize] {
                    let base = (mi_row as usize + rr) * self.grid_stride + mi_col as usize;
                    self.grid[base..base + MI_WB[subsize]].fill(w0m.mode as u8);
                    self.uv_grid[base..base + MI_WB[subsize]].fill(w0m.uv_mode as u8);
                }
                self.stats.rect_mid_encodes += 1;
                // Sub-block 1 at the edge position (+ mi_step).
                let (r1, c1) = if i == 0 {
                    (mi_row + mi_step, mi_col)
                } else {
                    (mi_row, mi_col + mi_step)
                };
                let best_remain = c::ref_rd_stats_subtraction(
                    self.o.rdmult,
                    (best_rdc.rate, best_rdc.dist, best_rdc.rdcost),
                    (sum_rdc.rate, sum_rdc.dist, sum_rdc.rdcost),
                );
                let (this_rdc, w) = self.leaf(
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl_search,
                    r1,
                    c1,
                    subsize,
                    partition_type,
                    best_remain,
                );
                visits.push(LeafVisit {
                    mi_row: r1,
                    mi_col: c1,
                    bsize: subsize,
                    budget: best_remain.2,
                    rate: this_rdc.rate,
                    dist: this_rdc.dist,
                    rdcost: this_rdc.rdcost,
                });
                if this_rdc.rate == i32::MAX {
                    sum_rdc.rdcost = i64::MAX;
                } else {
                    sum_rdc.rate += this_rdc.rate;
                    sum_rdc.dist += this_rdc.dist;
                    let (nr, nd, nc) = c::ref_rd_cost_update(
                        self.o.rdmult,
                        sum_rdc.rate,
                        sum_rdc.dist,
                        sum_rdc.rdcost,
                    );
                    sum_rdc = PartRdStats {
                        rate: nr,
                        dist: nd,
                        rdcost: nc,
                    };
                }
                w1 = w;
            }
            // Best update (:3626-3632).
            if sum_rdc.rdcost < best_rdc.rdcost {
                sum_rdc.rdcost = aom_encode::rd::rdcost(self.o.rdmult, sum_rdc.rate, sum_rdc.dist);
                if sum_rdc.rdcost < best_rdc.rdcost {
                    best_rdc = sum_rdc;
                    found = true;
                    self.stats.rect_wins[i] += 1;
                    let pair = Box::new([w0.take().unwrap(), w1.take().unwrap()]);
                    best_tree = Some(if i == 0 {
                        SbTree::Horz(pair)
                    } else {
                        SbTree::Vert(pair)
                    });
                }
            }
            // av1_restore_context at EACH type's loop tail (:3644).
            restore(&mut self.o);
        }

        if found {
            let tree = best_tree.as_mut().unwrap();
            let do_encode = if bsize == self.sb_size {
                true
            } else if bsize > self.max_partition_size {
                false
            } else if pc_index != 3 {
                true
            } else {
                bsize == self.max_partition_size
                    && c_split_sub(self.sb_size) != self.max_partition_size
            };
            if do_encode {
                let mut outs: Vec<CLeafOut> = Vec::new();
                // OUTPUT_ENABLED at the SB root (:6010): tx_type_map resets go
                // to the frame-map copy, winner maps stay pristine; DRY at
                // non-SB nodes (:6023): resets persist (C ctx alias) — the
                // exact split the port's winner encode applies.
                self.o.encode_sb(
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl_enc,
                    tree,
                    mi_row,
                    mi_col,
                    bsize,
                    &mut outs,
                    bsize == self.sb_size,
                );
                stamp_grid(
                    &mut self.grid,
                    &mut self.uv_grid,
                    self.grid_stride,
                    tree,
                    mi_row,
                    mi_col,
                    bsize,
                );
            }
        }
        if found {
            (best_tree, best_rdc, true)
        } else {
            (None, best_rdc, false)
        }
    }
}

fn stamp_grid(
    grid: &mut [u8],
    uv_grid: &mut [u8],
    stride: usize,
    tree: &SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) {
    match tree {
        SbTree::Leaf(w) => {
            for r in 0..MI_HB[bsize] {
                let base = (mi_row as usize + r) * stride + mi_col as usize;
                grid[base..base + MI_WB[bsize]].fill(w.mode as u8);
                uv_grid[base..base + MI_WB[bsize]].fill(w.uv_mode as u8);
            }
        }
        SbTree::Split(kids) => {
            let sub = c_split_sub(bsize);
            let hbs = (MI_WB[bsize] / 2) as i32;
            for (idx, t) in kids.iter().enumerate() {
                stamp_grid(
                    grid,
                    uv_grid,
                    stride,
                    t,
                    mi_row + ((idx as i32) >> 1) * hbs,
                    mi_col + ((idx as i32) & 1) * hbs,
                    sub,
                );
            }
        }
        SbTree::Horz(subs) => {
            let sub = c::ref_get_partition_subsize(bsize as i32, 1) as usize;
            let hbs = (MI_WB[bsize] / 2) as i32;
            for (k, w) in subs.iter().enumerate() {
                let r = mi_row + (k as i32) * hbs;
                for rr in 0..MI_HB[sub] {
                    let base = (r as usize + rr) * stride + mi_col as usize;
                    grid[base..base + MI_WB[sub]].fill(w.mode as u8);
                    uv_grid[base..base + MI_WB[sub]].fill(w.uv_mode as u8);
                }
            }
        }
        SbTree::Vert(subs) => {
            let sub = c::ref_get_partition_subsize(bsize as i32, 2) as usize;
            let hbs = (MI_WB[bsize] / 2) as i32;
            for (k, w) in subs.iter().enumerate() {
                let cc = mi_col + (k as i32) * hbs;
                for rr in 0..MI_HB[sub] {
                    let base = (mi_row as usize + rr) * stride + cc as usize;
                    grid[base..base + MI_WB[sub]].fill(w.mode as u8);
                    uv_grid[base..base + MI_WB[sub]].fill(w.uv_mode as u8);
                }
            }
        }
        SbTree::Horz4(_) | SbTree::Vert4(_) => {
            panic!(
                "partition_pick_diff.rs's NONE/SPLIT/HORZ/VERT-only harness never produces a 4-way tree"
            )
        }
        SbTree::HorzA(_) | SbTree::HorzB(_) | SbTree::VertA(_) | SbTree::VertB(_) => {
            panic!(
                "partition_pick_diff.rs's NONE/SPLIT/HORZ/VERT-only harness never produces an AB tree"
            )
        }
        // Off-frame SPLIT-child placeholder (interior fixtures never produce it).
        SbTree::Absent => {}
    }
}

fn leaf_eq(
    x: &aom_encode::encode_sb::LeafWinner,
    y: &aom_encode::encode_sb::LeafWinner,
    tag: &str,
) {
    assert_eq!(x.bsize, y.bsize, "leaf bsize: {tag}");
    assert_eq!(x.mode, y.mode, "leaf mode: {tag}");
    assert_eq!(x.angle_delta_y, y.angle_delta_y, "leaf angle: {tag}");
    assert_eq!(x.use_filter_intra, y.use_filter_intra, "leaf fi: {tag}");
    assert_eq!(
        x.filter_intra_mode, y.filter_intra_mode,
        "leaf fi mode: {tag}"
    );
    assert_eq!(x.tx_size, y.tx_size, "leaf tx: {tag}");
    assert_eq!(
        x.luma_edge_filter_type, y.luma_edge_filter_type,
        "leaf luma edge filter: {tag}"
    );
    assert_eq!(x.uv_mode, y.uv_mode, "leaf uv_mode: {tag}");
    assert_eq!(x.angle_delta_uv, y.angle_delta_uv, "leaf uv angle: {tag}");
    assert_eq!(x.cfl_alpha_idx, y.cfl_alpha_idx, "leaf cfl idx: {tag}");
    assert_eq!(
        x.cfl_alpha_signs, y.cfl_alpha_signs,
        "leaf cfl signs: {tag}"
    );
    assert_eq!(
        x.uv_edge_filter_type, y.uv_edge_filter_type,
        "leaf uv edge filter: {tag}"
    );
    assert_eq!(x.tx_type_map, y.tx_type_map, "leaf map: {tag}");
}

fn tree_eq(a: &SbTree, b: &SbTree, tag: &str) {
    match (a, b) {
        (SbTree::Leaf(x), SbTree::Leaf(y)) => {
            leaf_eq(x, y, tag);
        }
        (SbTree::Split(xs), SbTree::Split(ys)) => {
            for (x, y) in xs.iter().zip(ys.iter()) {
                tree_eq(x, y, tag);
            }
        }
        (SbTree::Horz(xs), SbTree::Horz(ys)) | (SbTree::Vert(xs), SbTree::Vert(ys)) => {
            for (x, y) in xs.iter().zip(ys.iter()) {
                leaf_eq(x, y, tag);
            }
        }
        _ => panic!("tree SHAPE divergence: {tag}"),
    }
}

#[test]
fn rd_pick_partition_real_matches_c_recursion() {
    c::ref_init();
    let mut rng = Rng(0x9a57_11c1_0000_0007);
    // SB roots are mib-aligned in the real encoder (the per-SB variance
    // cache + left-context indexing rely on it) — (16,16) is aligned AND
    // interior (up/left available).
    let (mi_row0, mi_col0) = (16i32, 16i32);
    let sb = 12usize; // 64x64

    let mut split_roots = 0usize;
    let mut none_nodes = 0usize;
    let mut split_nodes = 0usize;
    let mut horz_nodes = 0usize;
    let mut vert_nodes = 0usize;
    let mut total_visits = 0usize;
    let mut tot = CPickStats::default();

    // (ss, qindex, allintra, min_partition, enable_rect_partitions, banded)
    #[allow(clippy::type_complexity)]
    let cases: [((usize, usize), usize, bool, usize, bool, bool); 8] = [
        // Chunk-7 regression baselines (rect off).
        ((1, 1), 64, true, 6, false, false), // 420, ALLINTRA arm, min 16x16
        ((0, 0), 128, false, 6, false, false), // 444, GOOD arm, min 16x16
        // Rect ON over the flat-ish quadrant content: NONE dominates ->
        // the ALLINTRA less_rectangular_check arm gets to kill rect.
        ((1, 1), 200, true, 6, true, false), // 420 high q, min 16x16
        ((1, 1), 128, false, 3, false, false), // 420, min 8x8 (3-level recursion)
        // Rect-structured (banded) content: HORZ/VERT genuinely win.
        ((1, 1), 128, true, 3, true, true), // 420 ALLINTRA min 8x8
        ((0, 0), 128, false, 3, true, true), // 444 GOOD min 8x8
        ((1, 1), 64, true, 6, true, true),  // 420 ALLINTRA min 16x16
        ((1, 1), 200, false, 6, true, true), // 420 GOOD high q min 16x16
    ];

    for (case, &((ss_x, ss_y), qindex, allintra, min_part, rect_on, banded)) in
        cases.iter().enumerate()
    {
        let bd: u8 = 8;
        let maxv = 255i64;
        let reduced = false;
        let use_chroma_tbl = allintra;
        let pol = if allintra {
            TxTypeSearchPolicy::speed0_allintra()
        } else {
            TxTypeSearchPolicy::speed0_good()
        };

        // Content: quadrant-mixed — flat-ish TL, detailed BR, gradients
        // elsewhere; neighbours seeded (the rows above / cols left of the SB
        // hold "previously coded" content).
        let recon_y0: Vec<u16> = (0..STRIDE * 128)
            .map(|_| (rng.next() % 256) as u16)
            .collect();
        let recon_u0: Vec<u16> = (0..STRIDE * 128)
            .map(|_| (rng.next() % 256) as u16)
            .collect();
        let recon_v0: Vec<u16> = (0..STRIDE * 128)
            .map(|_| (rng.next() % 256) as u16)
            .collect();
        let mut src_y = recon_y0.clone();
        let mut src_u = recon_u0.clone();
        let mut src_v = recon_v0.clone();
        let base_y = 0usize;
        let base_uv = 0usize;
        let y_org = base_y + (mi_row0 as usize * 4) * STRIDE + mi_col0 as usize * 4;
        for r in 0..64usize {
            for cx in 0..64usize {
                let i = y_org + r * STRIDE + cx;
                let (qr, qc) = (r / 32, cx / 32);
                let v: i64 = if banded {
                    // Rect-structured: sharp band boundaries at half-block
                    // offsets (NONE mispredicts the step, SPLIT overpays,
                    // HORZ/VERT split exactly on it).
                    match (qr, qc) {
                        // TL: horizontal bands split at r%32 == 16 (the
                        // 32x32 nodes' HORZ line).
                        (0, 0) => {
                            if r % 32 < 16 {
                                70 + (cx as i64 / 16)
                            } else {
                                185 + (cx as i64 / 16)
                            }
                        }
                        // TR: vertical bands split at cx%32 == 16 (VERT).
                        (0, 1) => {
                            if cx % 32 < 16 {
                                55 + (r as i64 / 16)
                            } else {
                                205 - (r as i64 / 16)
                            }
                        }
                        // BL: height-8 horizontal bands (HORZ at 16x16).
                        (1, 0) => {
                            if r % 16 < 8 {
                                90 + (cx as i64 % 3)
                            } else {
                                160 + (cx as i64 % 3)
                            }
                        }
                        // BR: width-8 vertical bands + dither (VERT bait;
                        // the dither keeps a high-variance 4x4 in the SB
                        // for the ALLINTRA var arm).
                        _ => {
                            if cx % 16 < 8 {
                                100 + i64::from(rng.range(0, 6))
                            } else {
                                30 + i64::from(rng.range(0, 6))
                            }
                        }
                    }
                } else {
                    match (qr, qc) {
                        (0, 0) => 96 + (r as i64 / 8),       // near-flat
                        (0, 1) => 40 + 3 * (cx as i64 % 24), // vertical texture
                        (1, 0) => 200 - 2 * (r as i64 % 32), // horizontal ramp
                        _ => i64::from(rng.range(0, 255)),   // noise (split bait)
                    }
                };
                src_y[i] = v.clamp(0, maxv) as u16;
            }
        }
        let uv_org = chroma_plane_offset(base_uv, STRIDE, mi_row0, mi_col0, sb, ss_x, ss_y);
        let (cw, ch) = (64 >> ss_x, 64 >> ss_y);
        for r in 0..ch {
            for cx in 0..cw {
                let i = uv_org + r * STRIDE + cx;
                // Luma-correlated chroma (CfL bait) + noise quadrant.
                let ly = y_org + (r << ss_y) * STRIDE + (cx << ss_x);
                src_u[i] = ((i64::from(src_y[ly]) * 3 / 5 + 60).clamp(0, maxv)) as u16;
                src_v[i] = ((200 - i64::from(src_y[ly]) / 3).clamp(0, maxv)) as u16;
            }
        }

        // Quantizers (Rust + C rows).
        let mut quants = Quants::zeroed();
        let mut deq = Dequants::zeroed();
        av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
        let rows_y = set_q_index(&quants, &deq, qindex, 0);
        let rows_u = set_q_index(&quants, &deq, qindex, 1);
        let rows_v = set_q_index(&quants, &deq, qindex, 2);
        let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
        let plane_rows_y = &rows_c[0..56];
        let (rows_u_c, rows_v_c) = (&rows_c[56..112], &rows_c[112..168]);
        let dequant_y = [rows_y.dequant[0], rows_y.dequant[1]];
        let dequant_u = [rows_u_c[48], rows_u_c[49]];
        let dequant_v = [rows_v_c[48], rows_v_c[49]];

        // Shared cost tables (identical fixture values both sides).
        let y_tbls: Vec<Vec<i32>> = [13 * 2, 4 * 3, 42 * 8, 9 * 2, 3 * 2, 21 * 26, 2 * 11]
            .iter()
            .map(|&n| tbl(&mut rng, n))
            .collect();
        let u_tbls: Vec<Vec<i32>> = [13 * 2, 4 * 3, 42 * 8, 9 * 2, 3 * 2, 21 * 26, 2 * 11]
            .iter()
            .map(|&n| tbl(&mut rng, n))
            .collect();
        // CPick::coeff_costs_y/_uv (+ the SbEncodeEnv built below, which
        // shares these same values) are the full per-txs_ctx CoeffCostSet;
        // the C oracle (COracle::coeff_tbls_y/_uv) still takes the 7 flat
        // arrays directly at any tx_size, so replicating them across every
        // txs_ctx/eob_multi_size slot reproduces the exact "identical
        // fixture values both sides" this harness relies on (see
        // coeff_cost_set_from_tables' doc comment).
        let coeff_costs_y = coeff_cost_set_from_tables(
            &y_tbls[0], &y_tbls[1], &y_tbls[2], &y_tbls[3], &y_tbls[4], &y_tbls[5], &y_tbls[6],
        );
        let coeff_costs_uv = coeff_cost_set_from_tables(
            &u_tbls[0], &u_tbls[1], &u_tbls[2], &u_tbls[3], &u_tbls[4], &u_tbls[5], &u_tbls[6],
        );
        // Real ext-tx cost fill (both sides share the values).
        let mut rng2 = Rng(rng.next() | 1);
        let mut ext_cdfs: Vec<u16> = Vec::new();
        for _ in 0..(3 * 4 * 13) {
            let mut acc = 0u32;
            let mut row = [0u16; 17];
            for e in row.iter_mut().take(15) {
                acc += rng2.range(1, 1800) as u32;
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            ext_cdfs.extend_from_slice(&row);
        }
        let mut ttc = TxTypeCosts::zeroed();
        {
            // fill only the intra sets we exercise via the real filler if
            // available; otherwise leave zeroed (identical both sides).
            let _ = &mut ttc;
        }
        let _ = fill_tx_type_costs; // referenced; full fill exercised in tx-search diffs
        let ttc_dummy = TxTypeCosts::zeroed();

        // Mode-cost tables: randomized but identical both sides.
        let mut mode_costs = IntraModeCosts::zeroed();
        for row in mode_costs.y_mode_costs.iter_mut().flatten() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        for row in mode_costs.angle_delta_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 8 << 9);
            }
        }
        for row in mode_costs.filter_intra_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        for row in mode_costs.filter_intra_mode_cost.iter_mut() {
            *row = rng.range(0, 4 << 9);
        }
        for row in mode_costs.palette_y_mode_cost.iter_mut().flatten() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        for row in mode_costs.palette_uv_mode_cost.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 4 << 9);
            }
        }
        let mut uv_mode_cost = [[[0i32; 14]; 13]; 2];
        for t in uv_mode_cost.iter_mut() {
            for row in t.iter_mut() {
                for e in row.iter_mut() {
                    *e = rng.range(0, 4 << 9);
                }
            }
        }
        let sign_cdf = {
            let mut row = vec![0u16; 9];
            let mut acc = 0u32;
            for e in row.iter_mut().take(7) {
                acc += rng.range(1, 3600) as u32;
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            row
        };
        let mut alpha_cdf = Vec::new();
        for _ in 0..6 {
            let mut row = vec![0u16; 17];
            let mut acc = 0u32;
            for e in row.iter_mut().take(15) {
                acc += rng.range(1, 1900) as u32;
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            alpha_cdf.extend(row);
        }
        let mut cfl_costs = CflCosts::zeroed();
        fill_cfl_costs(&mut cfl_costs, &sign_cdf, &alpha_cdf);
        let mut tx_size_costs = TxSizeCosts::zeroed();
        for row in tx_size_costs.0.iter_mut().flatten() {
            for e in row.iter_mut() {
                *e = rng.range(0, 2 << 9);
            }
        }
        let skip_costs = [[rng.range(0, 4 << 9), rng.range(0, 4 << 9)]; 3];
        let mut partition_costs = [[0i32; 10]; 20];
        for row in partition_costs.iter_mut() {
            for e in row.iter_mut() {
                *e = rng.range(0, 6 << 9);
            }
        }
        let uv_lp = UvLoopPolicy::speed0_allintra();
        let rdmult = 4000 + rng.range(0, 1 << 16);
        let sharpness = 0;

        // Pre-walk tile state: fresh-tile init (entropy/partition 0, txfm
        // 64) + seeded "previous SB" history above/left of this SB.
        let mut tile = TileCtxState::zeroed(64);
        for p in 0..3 {
            for x in tile.above_ectx[p][..(mi_col0 as usize >> [0, ss_x, ss_x][p])].iter_mut() {
                *x = (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8;
            }
        }
        let mut grid_rust = ModeGrid::dc(64, 64);
        for m in grid_rust.modes.iter_mut() {
            *m = (rng.next() % 13) as u8;
        }
        // Randomize the pre-SB UV-neighbour context too (values 0..12 include
        // the SMOOTH UV modes 9/10/11), so the per-block chroma edge filter is
        // actually exercised as a differential witness — not left trivially 0.
        for m in grid_rust.uv_modes.iter_mut() {
            *m = (rng.next() % 13) as u8;
        }
        let grid_c = grid_rust.modes.clone();
        let uv_grid_c = grid_rust.uv_modes.clone();
        let tile0 = tile.clone();

        let env = SbEncodeEnv {
            sb_size: sb,
            mi_rows: 512,
            mi_cols: 512,
            tile_row_start: 0,
            tile_col_start: 0,
            tile_row_end: 1 << 16,
            tile_col_end: 1 << 16,
            monochrome: false,
            ss_x,
            ss_y,
            bd,
            lossless: false,
            reduced_tx_set_used: reduced,
            disable_edge_filter: false,
            filter_type: 0,
            stride: STRIDE,
            src_y: &src_y,
            src_u: &src_u,
            src_v: &src_v,
            base_y,
            base_uv,
            rows_y: &rows_y,
            rows_u: &rows_u,
            rows_v: &rows_v,
            rdmult,
            sharpness,
            enable_optimize_b: TrellisOptType::FullTrellisOpt,
            use_chroma_trellis_rd_mult: use_chroma_tbl,
            coeff_costs_y: &coeff_costs_y,
            coeff_costs_uv: &coeff_costs_uv,
            tx_type_costs: &ttc_dummy,
        };
        let cfg = PickFrameCfg {
            mode_costs: &mode_costs,
            tx_size_costs: &tx_size_costs,
            skip_costs: &skip_costs,
            tx_type_costs_y: &ttc,
            pol: &pol,
            uv_lp: &uv_lp,
            intra_uv_mode_cost: &uv_mode_cost,
            cfl_costs: &cfl_costs,
            partition_costs: &partition_costs,
            partition_cdfs: &UNUSED_EDGE_PARTITION_CDF,
            allintra,
            speed: 0,
            qindex: qindex as i32,
            enable_filter_intra: true,
            enable_tx64: true,
            enable_rect_tx: true,
            intra_pruning_with_hog: true,
            enable_rect_partitions: rect_on,
            less_rectangular_check_level: if allintra { 1 } else { 0 },
            max_partition_size: 15, // BLOCK_128X128 (KEY default)
            min_partition_size: min_part,
            // This harness's own tree-shape counters (`stamp_grid`/`count`
            // above) are NONE/SPLIT/HORZ/VERT-only -- keep 4-way off so
            // rd_pick_partition_real never produces a shape they can't
            // handle. Not yet cross-checked against the C reference here.
            enable_1to4_partitions: false,
            // Same reasoning: AB shapes are not yet cross-checked here either.
            enable_ab_partitions: false,
            allow_screen_content_tools: false,
        };

        // ---- Rust recursion ----
        let mut ry = recon_y0.clone();
        let mut ru = recon_u0.clone();
        let mut rv = recon_v0.clone();
        let mut cfl_rust = CflCtx::new(ss_x as i32, ss_y as i32);
        let mut visits = Vec::new();
        let mut last_source_variance = 0u32;
        let (tree, best, found) = rd_pick_partition_real(
            &env,
            &cfg,
            &mut tile,
            &mut grid_rust,
            &mut ry,
            &mut ru,
            &mut rv,
            &mut cfl_rust,
            mi_row0,
            mi_col0,
            sb,
            PartRdStats::invalid(),
            0,
            0, // quad_tree_idx: 0 at the SB root
            None,
            &mut visits,
            &mut last_source_variance,
        );

        // ---- C-side transcribed recursion ----
        let mut cy = recon_y0.clone();
        let mut cu = recon_u0.clone();
        let mut cv = recon_v0.clone();
        let mut cfl_c_search = CflCtx::new(ss_x as i32, ss_y as i32);
        let mut cfl_c_enc = c::RefCflState::default();
        let mut above_p = [0i8; 64];
        above_p.copy_from_slice(&tile0.above_pctx);
        let mut cp = CPick {
            o: COracle {
                ss: (ss_x, ss_y),
                monochrome: false,
                bd,
                reduced,
                sharpness,
                use_trellis: true,
                load_ctx: true,
                use_chroma_tbl,
                mi_rows: 512,
                mi_cols: 512,
                base_y,
                stride: STRIDE,
                base_uv,
                rdmult,
                src_y: &src_y,
                src_u: &src_u,
                src_v: &src_v,
                plane_rows_y,
                rows_u_c,
                rows_v_c,
                dequant_y,
                dequant_u,
                dequant_v,
                coeff_tbls_y: (
                    &y_tbls[0], &y_tbls[1], &y_tbls[2], &y_tbls[3], &y_tbls[4], &y_tbls[5],
                    &y_tbls[6],
                ),
                coeff_tbls_uv: (
                    &u_tbls[0], &u_tbls[1], &u_tbls[2], &u_tbls[3], &u_tbls[4], &u_tbls[5],
                    &u_tbls[6],
                ),
                ttc: (&[], &[]),
                above_e: [
                    tile0.above_ectx[0].clone(),
                    tile0.above_ectx[1].clone(),
                    tile0.above_ectx[2].clone(),
                ],
                left_e: tile0.left_ectx,
                above_p,
                left_p: tile0.left_pctx,
                above_t: tile0.above_tctx.clone(),
                left_t: tile0.left_tctx,
            },
            grid: grid_c,
            uv_grid: uv_grid_c,
            grid_stride: 64,
            mode_costs: &mode_costs,
            tx_size_costs: &tx_size_costs,
            skip_costs: &skip_costs,
            tx_type_costs_y: &ttc,
            pol: &pol,
            uv_lp: &uv_lp,
            intra_uv_mode_cost: &uv_mode_cost,
            cfl_costs: &cfl_costs,
            partition_costs: &partition_costs,
            allintra,
            speed: 0,
            qindex_cfg: qindex as i32,
            rows_y: &rows_y,
            rows_u: &rows_u,
            rows_v: &rows_v,
            coeff_costs_y: &coeff_costs_y,
            coeff_costs_uv: &coeff_costs_uv,
            ttc_dummy: &ttc_dummy,
            max_partition_size: 15,
            min_partition_size: min_part,
            sb_size: sb,
            monochrome: false,
            lossless: false,
            enable_optimize_b: TrellisOptType::FullTrellisOpt,
            enable_rect_partitions: rect_on,
            less_rectangular_check_level: if allintra { 1 } else { 0 },
            stats: CPickStats::default(),
        };
        let mut c_visits = Vec::new();
        let (c_tree, c_best, c_found) = cp.pick(
            &mut cy,
            &mut cu,
            &mut cv,
            &mut cfl_c_search,
            &mut cfl_c_enc,
            mi_row0,
            mi_col0,
            sb,
            PartRdStats::invalid(),
            0,
            &mut c_visits,
        );

        // ---- compare ----
        let tag =
            format!("case {case} ss {ss_x}{ss_y} q {qindex} allintra {allintra} min {min_part}");
        assert_eq!(found, c_found, "found: {tag}");
        assert_eq!(visits.len(), c_visits.len(), "visit count: {tag}");
        for (k, (a, b)) in visits.iter().zip(c_visits.iter()).enumerate() {
            assert_eq!(a, b, "leaf visit {k}: {tag}");
        }
        assert_eq!(
            (best.rate, best.dist, best.rdcost),
            (c_best.rate, c_best.dist, c_best.rdcost),
            "best stats: {tag}"
        );
        let (tree, c_tree) = (tree.expect("found"), c_tree.expect("found"));
        tree_eq(&tree, &c_tree, &tag);
        assert_eq!(ry, cy, "recon Y: {tag}");
        assert_eq!(ru, cu, "recon U: {tag}");
        assert_eq!(rv, cv, "recon V: {tag}");
        assert_eq!(tile.above_ectx[0], cp.o.above_e[0], "above ectx Y: {tag}");
        assert_eq!(tile.above_ectx[1], cp.o.above_e[1], "above ectx U: {tag}");
        assert_eq!(tile.above_ectx[2], cp.o.above_e[2], "above ectx V: {tag}");
        assert_eq!(tile.left_ectx, cp.o.left_e, "left ectx: {tag}");
        assert_eq!(&tile.above_pctx[..], &cp.o.above_p[..], "above pctx: {tag}");
        assert_eq!(tile.left_pctx, cp.o.left_p, "left pctx: {tag}");
        assert_eq!(tile.above_tctx, cp.o.above_t, "above tctx: {tag}");
        assert_eq!(tile.left_tctx, cp.o.left_t, "left tctx: {tag}");
        assert_eq!(grid_rust.modes, cp.grid, "mode grid: {tag}");
        assert_eq!(grid_rust.uv_modes, cp.uv_grid, "uv mode grid: {tag}");

        // Shape coverage accounting.
        fn count(t: &SbTree, none_n: &mut usize, split_n: &mut usize, rect_n: &mut [usize; 2]) {
            match t {
                SbTree::Leaf(_) => *none_n += 1,
                SbTree::Split(kids) => {
                    *split_n += 1;
                    for k in kids.iter() {
                        count(k, none_n, split_n, rect_n);
                    }
                }
                SbTree::Horz(_) => rect_n[0] += 1,
                SbTree::Vert(_) => rect_n[1] += 1,
                SbTree::Horz4(_) | SbTree::Vert4(_) => {
                    panic!("this NONE/SPLIT/HORZ/VERT-only harness never produces a 4-way tree")
                }
                SbTree::HorzA(_) | SbTree::HorzB(_) | SbTree::VertA(_) | SbTree::VertB(_) => {
                    panic!("this NONE/SPLIT/HORZ/VERT-only harness never produces an AB tree")
                }
                // Off-frame SPLIT-child placeholder (interior fixtures never produce it).
                SbTree::Absent => {}
            }
        }
        let (mut n, mut s) = (0usize, 0usize);
        let mut r2 = [0usize; 2];
        count(&tree, &mut n, &mut s, &mut r2);
        horz_nodes += r2[0];
        vert_nodes += r2[1];
        none_nodes += n;
        split_nodes += s;
        if matches!(tree, SbTree::Split(_)) {
            split_roots += 1;
        }
        total_visits += visits.len();
        tot.var_force_split += cp.stats.var_force_split;
        tot.less_rect_kills += cp.stats.less_rect_kills;
        tot.rect_evals += cp.stats.rect_evals;
        tot.rect_mid_encodes += cp.stats.rect_mid_encodes;
        tot.rect_wins[0] += cp.stats.rect_wins[0];
        tot.rect_wins[1] += cp.stats.rect_wins[1];
    }

    // Coverage floors: both partition outcomes must genuinely occur.
    assert!(
        split_roots >= 1,
        "at least one SB picked SPLIT: {split_roots}"
    );
    assert!(none_nodes >= 6, "NONE leaves across cases: {none_nodes}");
    assert!(split_nodes >= 2, "SPLIT nodes across cases: {split_nodes}");
    assert!(
        total_visits >= 40,
        "leaf evaluations exercised: {total_visits}"
    );
    // Rect-stage coverage floors: HORZ and VERT must both genuinely win
    // somewhere, the mid-stage sub-0 propagation must run, and the ALLINTRA
    // arms (less_rect kill + var force-split) must fire.
    assert!(horz_nodes >= 1, "HORZ winners across cases: {horz_nodes}");
    assert!(vert_nodes >= 1, "VERT winners across cases: {vert_nodes}");
    assert!(
        tot.rect_evals >= 8,
        "rect types evaluated: {}",
        tot.rect_evals
    );
    assert!(
        tot.rect_mid_encodes >= 4,
        "rect mid-stage encodes: {}",
        tot.rect_mid_encodes
    );
    assert!(
        tot.rect_wins[0] >= 1 && tot.rect_wins[1] >= 1,
        "rect wins: {:?}",
        tot.rect_wins
    );
    assert!(
        tot.less_rect_kills >= 1,
        "less_rect kills: {}",
        tot.less_rect_kills
    );
    assert!(
        tot.var_force_split >= 1,
        "ALLINTRA var force-splits: {}",
        tot.var_force_split
    );
    eprintln!(
        "coverage: none={none_nodes} split={split_nodes} horz={horz_nodes} vert={vert_nodes} \
         visits={total_visits} rect_evals={} mid_encodes={} wins={:?} less_rect={} var_split={}",
        tot.rect_evals,
        tot.rect_mid_encodes,
        tot.rect_wins,
        tot.less_rect_kills,
        tot.var_force_split
    );
    let _ = (_rust_hog, perpixel_variance_y, is_chroma_reference);
}
