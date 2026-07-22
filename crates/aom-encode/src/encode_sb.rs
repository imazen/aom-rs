//! `encode_sb` / `encode_b` (av1/encoder/partition_search.c:1581/1419) — the
//! partition-tree walk that re-encodes a PICKED tree's winners, at
//! `DRY_RUN_NORMAL` semantics: the pass `av1_rd_pick_partition` runs on the
//! winner subtree below SB level (`should_do_dry_run_encode_for_current_block`,
//! partition_search.c:5556) so siblings and parents see the winner's
//! reconstruction and contexts.
//!
//! # `encode_b` at DRY_RUN for a KEY intra leaf (verified against the C)
//!
//! `encode_b` (:1419): `av1_set_offsets_without_segment_id` (geometry +
//! neighbour flags + the tile-level context pointers) -> `setup_block_rdmult`
//! (GOOD/KEY/NO_AQ: constant `cpi->rd.RDMULT`; ALLINTRA: the per-SB
//! `intra_sb_rdmult_modifier` fold, `>>7`, floor 1 — partition_search.c:652,
//! set at the SB root from `log_sub_block_var` :5710) -> `mbmi->partition =
//! partition` -> `av1_update_state` (:176, encodeframe_utils.c): install the
//! winner `ctx->mic` into the mi grid, point `xd->tx_type_map` at the
//! BLOCK-LOCAL winner map (`stride = mi_size_wide[bsize]`; the frame-level
//! copy happens ONLY at `!dry_run`), adopt the pick ctx's coefficient
//! buffers; segmentation/AQ arms all off in the envelope -> `encode_superblock`
//! -> [`!dry_run` ONLY: cb offsets, delta-q state, `update_stats` symbol
//! counts — the pack stage] -> rdmult restore.
//!
//! `encode_superblock` (:395), intra arm, at ANY dry_run:
//! 1. `xd->cfl.store_y = store_cfl_required(cm, xd)` — the NON-rdo gate
//!    ([`store_cfl_required`]): monochrome -> 0; `!is_chroma_ref` -> 1
//!    (ALWAYS store — a later chroma-ref sibling may pick CfL);
//!    else `!is_inter && uv_mode == UV_CFL_PRED`.
//! 2. `av1_encode_intra_block_plane(plane, dry_run, optimize_seg_arr[seg])`
//!    for ALL planes ([`encode_intra_block_plane_y`] +
//!    [`encode_intra_block_plane_uv`]; the chroma arms early-return when
//!    `!is_chroma_ref`).
//! 3. lossless-segment skip force + palette: envelope-excluded (no
//!    segmentation; palette off).
//! 4. `av1_update_intra_mb_txb_context` (encodetxb.c:871): per plane a
//!    `foreach_txb` walk at `av1_get_tx_size(plane)`; at DRY_RUN both
//!    visitors (`av1_update_and_record_txb_context` / `av1_record_txb_context`
//!    — `allow_update_cdf` selects, but their OUTPUT_ENABLED block is the
//!    only difference) reduce to: `tx_type = av1_get_tx_type` (Y: the map
//!    read AFTER the encode's eob-0 DCT resets; UV: the uv-mode arm) ->
//!    `cul_level = av1_get_txb_entropy_context(qcoeff, scan, eob)` ->
//!    `av1_set_entropy_contexts` — the TILE-LEVEL entropy-context stamp
//!    (edge-clipped in general; full-footprint for interior blocks). This is
//!    the stamp sibling RD walks load via `av1_get_entropy_contexts`.
//!    At OUTPUT_ENABLED it ADDS the coefficient-CDF adaptation + tcoeff
//!    recording + `update_tx_type_count` (plane-0 ext-tx CDF) — the pack
//!    stage, NOT ported here.
//! 5. `!dry_run` ONLY: the tx-size CDF count/update block — pack stage.
//! 6. At ANY dry_run (the `else` of the inter vartx-context arm): intra
//!    `mbmi->tx_size = (bsize > BLOCK_4X4) ? tx_size : TX_4X4` (a no-op for
//!    picked winners) + `set_txfm_ctxs(tx_size, mi_w, mi_h, 0, xd)` — the
//!    above/left TXFM-context stamp (`tx_size_wide/high` bytes) that later
//!    blocks' `get_tx_size_context` reads.
//! 7. `cfl_store_block` (:583): `is_inter && !is_chroma_ref` — dead for
//!    intra (intra !chroma-ref blocks store per-txb inside the plane-0 walk
//!    via `store_y == CFL_ALLOWED`).
//!
//! `encode_sb` (:1581): out-of-frame/invalid-subsize outs; [`!dry_run` ONLY:
//! the partition-CDF update at partition roots with rows+cols — pack stage];
//! the partition switch (NONE -> `encode_b`; SPLIT -> 4 recursive
//! `encode_sb`; HORZ -> `encode_b(subsize, HORZ)` at the origin + [`mi_row +
//! hbs < mi_rows`:] `encode_b` at `(mi_row + hbs, mi_col)` (:1637-1644);
//! VERT -> the mirrored column pair (:1629-1636); AB/4-way sequences for
//! those tree types); then ALWAYS `update_ext_partition_context` (the
//! partition-context stamp; ported in aom-entropy). The rect leaves receive
//! `partition = PARTITION_HORZ/VERT` — `mbmi->partition` feeds the
//! `has_top_right`/`has_bottom_left` availability tables (reconintra.c:182/
//! 367 branch only on VERT_A/VERT_B, so HORZ/VERT read the default table;
//! threaded for exactness + the AB stage).
//!
//! # Scope
//!
//! NONE + SPLIT + HORZ + VERT tree shapes (4 of 10 partition types —
//! matching the ported partition search slice; the AB/4-way `encode_b`
//! sequences are mechanical extensions of the same leaf composition). KEY
//! intra leaves, interior blocks, no segmentation, non-lossless-segment
//! envelope, block sizes <= 64x64. MISSING: AB/4-way tree shapes; the
//! OUTPUT_ENABLED adds (partition/coeff/tx-size CDF adaptation, tcoeff
//! recording, cb offsets — the pack stage, documented per step above);
//! frame-edge clipped walks (the rect arms carry the C's sub-1 frame-bound
//! guards but interior fixtures never take them); SB128.

use crate::encode_intra::{
    EncodeIntraPlaneOutcome, EncodeIntraYEnv, TrellisOptType, UvEncodeParams, UvWinner,
    encode_intra_block_plane_uv, encode_intra_block_plane_y,
};
use crate::intra_uv_rd::{
    UV_CFL_PRED, UvRdEnv, av1_get_tx_size_uv, chroma_plane_offset, is_chroma_reference,
};
use crate::partition::PartRdStats;
use crate::encode_intra::TxbEncode;
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TXS_H, TXS_W, max_block_units};
use aom_dsp::entropy::partition::{get_plane_block_size, update_ext_partition_context};
use aom_dsp::intra::cfl::CflCtx;
use aom_dsp::txb::{CoeffCostSet, get_txb_ctx, txb_entropy_context};

/// `store_cfl_required` (cfl.h:38) — the NON-rdo `store_y` gate of
/// `encode_superblock`, intra arm (`is_inter_block == 0`).
pub fn store_cfl_required(monochrome: bool, is_chroma_ref: bool, uv_mode: usize) -> bool {
    if monochrome {
        return false;
    }
    if !is_chroma_ref {
        // Always store luma: the corresponding chroma-ref block may use CfL.
        return true;
    }
    uv_mode == UV_CFL_PRED
}

/// The tile-level context arrays the dry-run walk stamps (the
/// `RD_SEARCH_MACROBLOCK_CONTEXT`-visible state + what
/// `av1_get_entropy_contexts` / `partition_plane_context` /
/// `get_tx_size_context` read for later blocks). `above_*` are tile-width
/// arrays indexed by ABSOLUTE `mi_col` (`>> ss_x` for chroma entropy);
/// `left_*` are the per-SB-strip 32-entry arrays indexed
/// `mi_row & MAX_MIB_MASK` (`>> ss_y` for chroma entropy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileCtxState {
    pub above_ectx: [Vec<i8>; 3],
    pub left_ectx: [[i8; 32]; 3],
    pub above_pctx: Vec<i8>,
    pub left_pctx: [i8; 32],
    pub above_tctx: Vec<u8>,
    pub left_tctx: [u8; 32],
}

/// `ALIGN_POWER_OF_TWO(mi_cols, MAX_MIB_SIZE_LOG2)` (aom_ports/mem.h:68;
/// `MAX_MIB_SIZE_LOG2` = 5): round the tile MI width up to a whole superblock
/// (a 32-mi multiple — the COMPILE-TIME max SB, 128px, used unconditionally by
/// `av1_alloc_above_context_buffers` regardless of the frame's actual SB size).
/// The above-context arrays are allocated to this width (alloccommon.c:414-448)
/// so a partial edge superblock — whose full-extent context save/restore covers
/// a straddling `BLOCK_64X64` candidate reaching past `mi_cols` to the SB
/// boundary — stays in bounds. `mi_cols` alone under-allocates and the search's
/// `save_context` panics on the first frame whose width is not a whole number
/// of superblocks (e.g. 196px -> mi_cols=50, aligned=64).
pub fn aligned_mi_cols(mi_cols: usize) -> usize {
    const MAX_MIB_SIZE_LOG2: usize = 5;
    (mi_cols + (1 << MAX_MIB_SIZE_LOG2) - 1) & !((1usize << MAX_MIB_SIZE_LOG2) - 1)
}

impl TileCtxState {
    /// Zeroed contexts for a tile of `mi_cols` (the av1_zero tile init). The
    /// ABOVE arrays are sized to [`aligned_mi_cols`] (a whole-SB multiple), NOT
    /// bare `mi_cols`, matching `av1_alloc_above_context_buffers`
    /// (alloccommon.c:414) so a frame-edge partial superblock's context
    /// save/restore never overruns the array. The LEFT arrays are per-SB-strip
    /// (32 = `MAX_MIB_SIZE`) and already whole-SB.
    pub fn zeroed(mi_cols: usize) -> Self {
        let aligned = aligned_mi_cols(mi_cols);
        TileCtxState {
            above_ectx: [vec![0; aligned], vec![0; aligned], vec![0; aligned]],
            left_ectx: [[0; 32]; 3],
            above_pctx: vec![0; aligned],
            left_pctx: [0; 32],
            // The C tile init memsets the txfm-context arrays to
            // tx_size_wide[TX_SIZES_LARGEST] == 64, NOT 0
            // (av1_zero_above_context / av1_zero_left_context;
            // aom_dsp::entropy::partition::TXFM_CTX_INIT).
            above_tctx: vec![aom_dsp::entropy::partition::TXFM_CTX_INIT; aligned],
            left_tctx: [aom_dsp::entropy::partition::TXFM_CTX_INIT; 32],
        }
    }
}

/// The picked winner of one leaf (`ctx->mic` + `ctx->tx_type_map` — what
/// `av1_update_state` installs before `encode_superblock`).
#[derive(Clone, Debug)]
pub struct LeafWinner {
    /// `mbmi->bsize` (must equal the tree position's subsize).
    pub bsize: usize,
    // Luma winner.
    pub mode: usize,
    pub angle_delta_y: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: usize,
    /// `mbmi->tx_size` (the uniform luma winner size).
    pub tx_size: usize,
    /// Per-block LUMA intra edge filter type (`get_intra_edge_filter_type` on
    /// plane 0, reconintra.c:974) — 1 iff an available above/left neighbour is a
    /// SMOOTH luma mode (SMOOTH_PRED=9/SMOOTH_V=10/SMOOTH_H=11), else 0.
    /// Recomputed from the neighbour Y modes during the search (KB-2) and frozen
    /// onto the winner so the dry-run re-encode (`encode_b_intra_dry`, which has
    /// no grid) predicts LUMA with the SAME edge filter the search used. KB-2
    /// fixed only the search; this closes the luma RE-ENCODE gap (KB-6) — the
    /// exact luma analogue of `uv_edge_filter_type` (#26). Without it an angled
    /// luma leaf with a SMOOTH neighbour re-encodes with filter_type 0 instead of
    /// 1 → wrong residual → a per-txb eob flip in the coded bytes.
    pub luma_edge_filter_type: i32,
    // Chroma winner (read only when the leaf is a chroma reference).
    pub uv_mode: usize,
    pub angle_delta_uv: i32,
    pub cfl_alpha_idx: i32,
    pub cfl_alpha_signs: i32,
    /// Per-block chroma intra edge filter type (`get_intra_edge_filter_type`
    /// on the UV plane, reconintra.c:974) — 1 iff an available above/left
    /// chroma neighbour is a SMOOTH UV mode (9/10/11), else 0. Recomputed
    /// from the chroma neighbours' UV modes during the search (where the mode
    /// grid is live) and frozen onto the winner so the dry-run re-encode
    /// (`encode_b_intra_dry`, which has no grid) predicts chroma with the SAME
    /// edge filter the search's UV RD used. Replaces the frozen SB-level
    /// `SbEncodeEnv::filter_type` on the chroma re-encode path (the KB-2 luma
    /// fix's chroma analogue).
    pub uv_edge_filter_type: i32,
    /// The block-local winner tx_type_map (stride `mi_size_wide[bsize]`),
    /// mutated in place by the luma re-encode's eob-0 resets (the state
    /// `ctx->tx_type_map` holds after the walk).
    pub tx_type_map: Vec<u8>,
    /// `mbmi->palette_mode_info` (Y half) + the winning colour-index map —
    /// `Some` when the leaf's luma winner is palette mode (`mode` is DC_PRED
    /// then). Drives the re-encode's map-fill prediction AND the pack's
    /// palette syntax + colour-map tokens.
    pub palette_y: Option<crate::palette_search::PaletteYInfo>,
    /// The UV half (`palette_size[1]` + U/V colours + the chroma-plane map)
    /// — `Some` when the chroma winner is UV palette (`uv_mode` is
    /// UV_DC_PRED then).
    pub palette_uv: Option<crate::palette_search::PaletteUvInfo>,
    /// `mbmi->skip_txfm` (0 throughout the KEY intra path; may be 1 on an
    /// intrabc leaf whose RD picked the skip arm).
    pub skip_txfm: bool,
    /// `mbmi->use_intrabc` — the leaf is an intra-block-copy block (screen
    /// content only, gated on `allow_intrabc`). When set, `mode`/`uv_mode` are
    /// DC_PRED/UV_DC_PRED (dead) and the block copies from the recon at the DV.
    pub use_intrabc: bool,
    /// `mbmi->inter_tx_size[16]` — the var-tx quadtree the intrabc COEFF arm
    /// picked (`av1_pick_recursive_tx_size_type_yrd` →
    /// [`crate::var_tx::VarTxResult::inter_tx_size`]). Read ONLY when
    /// `use_intrabc && !skip_txfm`: it drives both the re-encode walk
    /// (`encode_block_inter`'s recursion, encodemb.c:495-533, which compares
    /// the walk's `tx_size` against `inter_tx_size[get_txb_size_index(..)]`)
    /// and the pack's `write_tx_size_vartx` (bitstream.c:1542-1552). Every
    /// other winner leaves it at `[0; 16]` (dead — the intra path signals a
    /// UNIFORM `tx_size` instead).
    pub inter_tx_size: [usize; 16],
    /// `mbmi->mv[0]` (1/8-pel, full-pel multiples of 8): the winning DV.
    pub dv_row: i32,
    pub dv_col: i32,
    /// The ref DV the mode-rate was computed against — the pack writes the
    /// signalled diff `dv - dv_ref` (`write_intrabc_info`).
    pub dv_ref_row: i32,
    pub dv_ref_col: i32,
    /// `is_inter_block(mbmi)` — the leaf is an INTER block in a P/B frame
    /// (distinct from `use_intrabc`, which is intra-frame block copy). When
    /// set, the pack takes `pack_inter_mode_mvs`' inter arm
    /// (`av1/encoder/bitstream.c:1092`) instead of the intra prediction modes,
    /// and the recon is built from the REFERENCE frame rather than predicted
    /// intra. INTER-ENCODE chunk 2.
    pub is_inter: bool,
    /// `mbmi->ref_frame[0]` — `LAST_FRAME` (1) throughout the §3 single-
    /// reference envelope; `INTRA_FRAME` (0) on a non-inter leaf.
    pub ref_frame0: i8,
    /// `mbmi->ref_frame[1]` — `NONE_FRAME` (-1); compound is out of scope.
    pub ref_frame1: i8,
    /// `mbmi->mode` for an inter leaf (`NEARESTMV`/`NEARMV`/`GLOBALMV`/`NEWMV`,
    /// enums.h:337-349). Read only when `is_inter`; the intra `mode` field is
    /// dead then, exactly as C leaves it.
    pub inter_mode: i32,
    /// `mbmi->mv[0]` in 1/8-pel units. `(0, 0)` for the zero-MV target.
    pub mv_row: i32,
    pub mv_col: i32,
    /// The `mode_context` the inter mode rate was costed against
    /// (`av1_mode_context_analyzer` over `find_inter_mv_refs`' result). Frozen
    /// onto the winner so the pack writes `write_inter_mode` on the SAME
    /// context slices the RD priced — the search has the neighbour grid live,
    /// the pack re-walk does not.
    pub inter_mode_context: i32,
    /// `ctx->rd_stats` (the PICK_MODE_CONTEXT's own raw mode-search RD,
    /// BEFORE any enclosing stage adds its partition-type `pt_cost` —
    /// `leaf_pick_sb_modes`'s own returned [`PartRdStats`], unconditionally
    /// stored here too). Needed by the AB `reuse_prev_rd_results_for_part_ab`
    /// mechanism (`pick_sb_modes`'s `rd_mode_is_ready` early-return copies
    /// exactly `ctx->rd_stats`, partition_search.c:854-861) — a NONE-leaf
    /// SPLIT child or a rect stage's sub-0 winner exposes this so a later AB
    /// sub-block at the SAME position/size can be seeded from it verbatim
    /// instead of re-searching under a different (tighter) budget.
    pub raw_rdstats: PartRdStats,
}

impl LeafWinner {
    /// A never-coded placeholder for a rectangular partition's second
    /// sub-block whose origin is off-frame at a partial edge superblock — the
    /// C's `is_not_edge_block[i]`-false case where `rd_pick_rect_partition`
    /// searches only sub-0 (partition_search.c:3604). Every tree walker guards
    /// the sub-1 frame bound (`if mi_row/col + hbs < mi_rows/cols`) before
    /// touching it, so this is inert; `bsize` carries the nominal rect subsize
    /// only to satisfy the in-guard `debug_assert_eq!(s1.bsize, subsize)` in
    /// the (unreached) coded branch.
    pub fn off_frame_placeholder(bsize: usize) -> Self {
        LeafWinner {
            bsize,
            mode: 0,
            angle_delta_y: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            tx_size: 0,
            luma_edge_filter_type: 0,
            uv_mode: 0,
            angle_delta_uv: 0,
            cfl_alpha_idx: 0,
            cfl_alpha_signs: 0,
            uv_edge_filter_type: 0,
            tx_type_map: Vec::new(),
            palette_y: None,
            palette_uv: None,
            skip_txfm: false,
            use_intrabc: false,
            inter_tx_size: [0; 16],
            dv_row: 0,
            dv_col: 0,
            dv_ref_row: 0,
            dv_ref_col: 0,
            is_inter: false,
            ref_frame0: 0,
            ref_frame1: -1,
            inter_mode: 0,
            mv_row: 0,
            mv_col: 0,
            inter_mode_context: 0,
            raw_rdstats: PartRdStats::invalid(),
        }
    }

    /// The block's per-mi DV projection for the search-side [`ModeGrid`] DV
    /// grid (the `find_dv_ref_mvs` neighbour source + skip context). Intra
    /// winners project `use_intrabc = false`, dv = 0.
    pub fn dv_cell(&self) -> crate::intrabc_search::DvCell {
        crate::intrabc_search::DvCell {
            bsize: self.bsize as u8,
            // An INTER leaf projects its inter mode (NEARESTMV..NEWMV) — the
            // ref-MV scan's `add_ref_mv_candidate` matches on it. Otherwise
            // DC_PRED for an intrabc block (dead), else the intra mode.
            mode: if self.is_inter {
                self.inter_mode as u8
            } else if self.use_intrabc {
                0
            } else {
                self.mode as u8
            },
            use_intrabc: self.use_intrabc,
            skip_txfm: self.skip_txfm,
            // The 1/8-pel motion vector for an inter leaf, the DV for intrabc.
            dv_row: if self.is_inter {
                self.mv_row as i16
            } else {
                self.dv_row as i16
            },
            dv_col: if self.is_inter {
                self.mv_col as i16
            } else {
                self.dv_col as i16
            },
            ref_frame0: if self.is_inter { self.ref_frame0 } else { 0 },
            ref_frame1: -1,
        }
    }
}

/// The frame/tile environment shared by every leaf of one dry-run walk.
pub struct SbEncodeEnv<'a> {
    pub sb_size: usize,
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// Tile bounds (mi units; interior fixtures use 0 / large ends).
    pub tile_row_start: i32,
    pub tile_col_start: i32,
    pub tile_row_end: i32,
    pub tile_col_end: i32,
    pub monochrome: bool,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: u8,
    pub lossless: bool,
    pub reduced_tx_set_used: bool,
    pub disable_edge_filter: bool,
    pub filter_type: i32,
    /// One stride shared by all planes (fixture convenience — the walk only
    /// ever derives per-plane offsets from it).
    pub stride: usize,
    /// Source planes + the pixel offset of mi (0,0) in each.
    pub src_y: &'a [u16],
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    pub base_y: usize,
    pub base_uv: usize,
    // Quantizer rows per plane.
    pub rows_y: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    pub rows_u: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    pub rows_v: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    /// `x->rdmult` as `setup_block_rdmult` leaves it — frame-constant at
    /// GOOD/KEY/NO_AQ; the per-SB ALLINTRA modifier fold is the SB-level
    /// caller's (constant across one SB's recursion either way).
    pub rdmult: i32,
    pub sharpness: i32,
    pub enable_optimize_b: TrellisOptType,
    /// sf `tx_sf.use_chroma_trellis_rd_mult` (ALLINTRA 1 / GOOD 0).
    pub use_chroma_trellis_rd_mult: bool,
    /// Coefficient cost tables: the full REAL per-(txs_ctx, eob_multi_size)
    /// luma (PLANE_TYPE_Y) and chroma (PLANE_TYPE_UV) sets — the trellis rate
    /// inputs. Callers select the per-tx_size view via `CoeffCostSet::tables`
    /// at each construction site that knows the txb's actual tx_size.
    pub coeff_costs_y: &'a CoeffCostSet,
    pub coeff_costs_uv: &'a CoeffCostSet,
    /// `txfm_partition_cost` for the per-SB (INTERNAL_COST_UPD_SB) refresh —
    /// the var-tx split-flag costs from the LIVE txfm_partition CDF. Read only
    /// by the intrabc COEFF-arm var-tx (`IntrabcLeafArgs.txfm_partition_costs`);
    /// zeroed on the intra envelope where the var-tx never runs (byte-inert).
    pub txfm_partition_costs: [[i32; 2]; 21],
    /// UV tx-type cost tables — REQUIRED by [`UvRdEnv`] but never read by
    /// the encode arm (chroma codes no tx-type bits); zeroed tables are
    /// fine.
    pub tx_type_costs: &'a aom_dsp::txb::TxTypeCosts,
    /// Frame QM levels (`qmatrix_level_{y,u,v}`), `None` = QM off — threaded
    /// into every leaf re-encode (search context-prop AND pack output).
    pub qm_levels: Option<[usize; 3]>,
    /// `oxcf.tune_cfg` knobs ([`crate::TuneKnobs`]): QM-PSNR trellis/search
    /// distortion metric + the IQ/SSIMULACRA2 trellis rshift. `Default` =
    /// the PSNR envelope (byte-identical to the pre-tune path).
    pub tune: crate::TuneKnobs,
    /// `--deltaq-mode=6` (DELTA_Q_VARIANCE_BOOST) frame context — `Some`
    /// makes [`crate::pack::pack_tile`] derive a per-superblock qindex from
    /// source variance (`setup_delta_q`, encodeframe.c:341) and re-select
    /// quantizer rows + rdmult per SB. `None` (default) = the proven
    /// fixed-qindex envelope, byte-identical.
    pub deltaq: Option<DeltaQFrameCtx<'a>>,
}

/// The frame-level inputs of the per-SB Variance Boost delta-q derivation
/// (`cm->delta_q_info` + the quantizer tables `av1_init_plane_quantizers`
/// re-selects rows from at each SB's adjusted qindex).
#[derive(Clone, Copy)]
pub struct DeltaQFrameCtx<'a> {
    /// The frame quantizer tables ([`aom_dsp::quant::av1_build_quantizer`] output)
    /// — per-SB rows are re-selected from these at the adjusted qindex.
    pub quants: &'a aom_dsp::quant::Quants,
    pub deq: &'a aom_dsp::quant::Dequants,
    /// `quant_params->base_qindex`.
    pub base_qindex: i32,
    /// `delta_q_info.delta_q_res` ([`crate::allintra_vis::variance_boost_delta_q_res`]
    /// for mode 6; [`crate::allintra_vis::DELTA_Q_RES_PERCEPTUAL`] = 4 for mode 3).
    pub delta_q_res: i32,
    /// `--deltaq-strength` percent (default 100) — Variance-Boost (mode 6) only.
    pub deltaq_strength: u32,
    /// `Some` selects `--deltaq-mode=3` (`DELTA_Q_PERCEPTUAL_AI`): the per-SB
    /// qindex comes from this precomputed wiener-variance map
    /// ([`crate::allintra_vis::av1_get_sbq_perceptual_ai`]) instead of the
    /// source-variance boost. `None` = `--deltaq-mode=6` (Variance Boost, the
    /// [`deltaq_strength`](Self::deltaq_strength) path) unless
    /// [`perceptual_wavelet`](Self::perceptual_wavelet) selects mode 2.
    pub perceptual_ai: Option<&'a crate::allintra_vis::WeberVarMap>,
    /// `Some(is_screen_content)` selects `--deltaq-mode=2`
    /// (`DELTA_Q_PERCEPTUAL`, wavelet AC energy): the per-SB qindex comes from
    /// the SB source wavelet energy ([`crate::allintra_vis::setup_delta_q_perceptual`]).
    /// The bool is `cpi->is_screen_content_type` (the rate-model enumerator).
    /// Mutually exclusive with [`perceptual_ai`](Self::perceptual_ai) (mode 3);
    /// when both are `None` the ctx is `--deltaq-mode=6` (Variance Boost).
    pub perceptual_wavelet: Option<bool>,
    /// `mi_size_wide[sb_size]` — the SB mi extent the mode-3 per-SB wiener
    /// window uses (unused when `perceptual_ai` is `None`).
    pub sb_mi: i32,
    /// `--delta-lf-mode=1` (`enable_deltalf_mode`, gated on `deltaq_mode !=
    /// NO_DELTA_Q`): when true the per-SB `delta_lf_from_base` is derived from
    /// each SB's `delta_qindex` (`setup_delta_q`, encodeframe.c:377-399) and
    /// coded alongside the delta-qindex. `delta_lf_res = DEFAULT_DELTA_LF_RES`
    /// (2), `delta_lf_multi = DEFAULT_DELTA_LF_MULTI` (0/single).
    pub delta_lf_present: bool,
}

/// One leaf's re-encode outputs (differential visibility).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafEncodeOut {
    pub mi_row: i32,
    pub mi_col: i32,
    pub bsize: usize,
    pub is_chroma_ref: bool,
    pub store_y: bool,
    pub y: EncodeIntraPlaneOutcome,
    pub u: Option<EncodeIntraPlaneOutcome>,
    pub v: Option<EncodeIntraPlaneOutcome>,
}

/// A picked partition tree (`pc_tree` slice): NONE leaves + SPLIT nodes +
/// HORZ/VERT rect pairs.
#[derive(Clone, Debug)]
pub enum SbTree {
    /// PARTITION_NONE at this node — the leaf winner.
    Leaf(LeafWinner),
    /// PARTITION_SPLIT — 4 children in raster order.
    Split(Box<[SbTree; 4]>),
    /// PARTITION_HORZ — `pc_tree->horizontal[2]`: the top sub-block at the
    /// node origin + the bottom at `mi_row + hbs` (both winners' bsize is
    /// the HORZ subsize). Interior envelope: both present (an edge HORZ
    /// with only sub-0 coded — `!has_rows` — is out of envelope).
    Horz(Box<[LeafWinner; 2]>),
    /// PARTITION_VERT — `pc_tree->vertical[2]`: left + right sub-blocks.
    Vert(Box<[LeafWinner; 2]>),
    /// PARTITION_HORZ_4 — `pc_tree->horizontal4[4]`: 4 equal-height
    /// horizontal strips in top-to-bottom order (module docs on
    /// [`crate::partition_pick::rd_pick_partition_real`]'s 4-way stage).
    /// Interior envelope: all 4 present (a frame-edge HORZ_4 with fewer
    /// than 4 coded strips is out of envelope, matching the existing
    /// Horz/Vert interior-only scope).
    Horz4(Box<[LeafWinner; 4]>),
    /// PARTITION_VERT_4 — `pc_tree->vertical4[4]`: 4 equal-width vertical
    /// strips in left-to-right order.
    Vert4(Box<[LeafWinner; 4]>),
    /// PARTITION_HORZ_A — `pc_tree->horizontala[3]`: top-left quarter,
    /// top-right quarter (both `bsize2` = SPLIT subsize), then the full-width
    /// bottom half (`subsize` = HORZ subsize) — `ab_subsize[HORZ_A]`/
    /// `ab_mi_pos[HORZ_A]`, partition_search.c:3805-3821. Interior envelope
    /// only (matches `allow_ab_partition_search`'s own `has_rows && has_cols`
    /// requirement — module docs on `rd_pick_partition_real`'s AB stage): all
    /// 3 sub-blocks always present, no frame-bound gating (the C's own
    /// `encode_sb` AB arms carry none either, partition_search.c:1652-1673).
    HorzA(Box<[LeafWinner; 3]>),
    /// PARTITION_HORZ_B — `pc_tree->horizontalb[3]`: full-width top half
    /// (`subsize`), then bottom-left quarter, bottom-right quarter (both
    /// `bsize2`) — `ab_subsize[HORZ_B]`, partition_search.c:3808-3813.
    HorzB(Box<[LeafWinner; 3]>),
    /// PARTITION_VERT_A — `pc_tree->verticala[3]`: top-left quarter,
    /// bottom-left quarter (both `bsize2`), then the full-height right half
    /// (`subsize` = VERT subsize) — `ab_subsize[VERT_A]`,
    /// partition_search.c:3810-3814 (column-axis mirror of HORZ_A).
    VertA(Box<[LeafWinner; 3]>),
    /// PARTITION_VERT_B — `pc_tree->verticalb[3]`: full-height left half
    /// (`subsize`), then top-right quarter, bottom-right quarter (both
    /// `bsize2`) — `ab_subsize[VERT_B]` (column-axis mirror of HORZ_B).
    VertB(Box<[LeafWinner; 3]>),
    /// A SPLIT child whose origin (`mi_row`/`mi_col`) is entirely off-frame at
    /// a partial edge superblock — the C's `pc_tree->split[idx]` for an
    /// out-of-bounds quadrant, which `write_modes_sb`/`encode_sb` never code
    /// (partition_search.c:1583, the `mi_row >= mi_rows || mi_col >= mi_cols`
    /// out). It carries no winner and is NEVER traversed: every walker
    /// (`encode_sb_dry`, `pack_sb`, `stamp_grid_from_tree`, `stamp_lf_tree`)
    /// tests the same frame-bound guard at entry and returns before inspecting
    /// it. It exists only so `Split`'s `[SbTree; 4]` can hold a placeholder for
    /// the trimmed quadrant instead of a bogus leaf.
    Absent,
}

/// `PARTITION_NONE` / `PARTITION_HORZ` / `PARTITION_VERT` /
/// `PARTITION_SPLIT` / `PARTITION_HORZ_4` / `PARTITION_VERT_4` C values.
const PARTITION_NONE: i32 = 0;
const PARTITION_HORZ: i32 = 1;
const PARTITION_VERT: i32 = 2;
const PARTITION_SPLIT: i32 = 3;
const PARTITION_HORZ_A: i32 = 4;
const PARTITION_HORZ_B: i32 = 5;
const PARTITION_VERT_A: i32 = 6;
const PARTITION_VERT_B: i32 = 7;
const PARTITION_HORZ_4: i32 = 8;
const PARTITION_VERT_4: i32 = 9;

/// `get_partition_subsize(bsize, PARTITION_SPLIT)` for the square sizes.
fn split_subsize(bsize: usize) -> usize {
    match bsize {
        3 => 0,
        6 => 3,
        9 => 6,
        12 => 9,
        15 => 12,
        _ => panic!("split_subsize: non-splittable square bsize {bsize}"),
    }
}

/// `encode_b` for one KEY intra leaf — see the module docs for the exact C
/// sequence. Mutates the recon planes, the CfL context (per-txb stores when
/// `store_y`), and stamps `state`'s entropy + txfm contexts. The partition
/// context stamp is [`encode_sb`]'s (`update_ext_partition_context` runs at
/// the node, not the leaf).
///
/// `output_enabled` mirrors C's `RUN_TYPE` through `av1_update_state`'s
/// tx_type_map plumbing (encodeframe_utils.c:217-231):
/// - `false` (C `DRY_RUN_NORMAL` — the search's context-propagation walks):
///   `xd->tx_type_map` ALIASES `ctx->tx_type_map`, so the per-txb
///   eob-0 -> DCT_DCT resets (encodemb.c:770-779) PERSIST into the stored
///   winner map — a later dry walk of the same leaf re-quantizes those txbs
///   as DCT_DCT, exactly as C does. Do not "fix" this by cloning: the
///   persistence is C behaviour.
/// - `true` (C `OUTPUT_ENABLED` — the SB-root winner walk, and the pack's
///   re-walk of the same tree): `av1_update_state` COPIES the ctx map into
///   the frame-level `mi_params.tx_type_map` and points `xd` at THAT, so the
///   resets land in the frame map and the winner's (ctx) map is left
///   untouched. Modelled with a transient clone. The port runs C's single
///   OUTPUT_ENABLED walk TWICE (the SB-root context/recon walk + the pack
///   re-walk); without the copy semantics the first walk's resets leaked
///   into the second walk's re-quant input, and a skip-winning txb (non-DCT
///   winner, eob 0) re-quantized as DCT_DCT with eob > 0 — the KB-4
///   bd10/bd12 mono coded-eob divergence.
/// `encode_superblock`'s inter-path CfL luma store (partition_search.c:580-583,
/// `!CONFIG_REALTIME_ONLY`):
/// `if (is_inter_block(mbmi) && !xd->is_chroma_ref && is_cfl_allowed(xd))
///  cfl_store_block(xd, mbmi->bsize, mbmi->tx_size);`
///
/// `is_inter_block` is TRUE for intra-block-copy (blockd.h:372:
/// `is_intrabc_block(mbmi) || ref_frame[0] > INTRA_FRAME`), so on a
/// screen-content frame an intrabc block that is NOT a chroma reference must
/// still publish its reconstructed luma to the CfL buffer — the later
/// chroma-reference sibling covering that luma reads it when it evaluates
/// `UV_CFL_PRED`. The intra path never reaches here (an intra non-chroma-ref
/// block stores per-txb inside the plane-0 walk, via `store_cfl_required`'s
/// `!is_chroma_ref => CFL_ALLOWED` arm), which is why this site was previously
/// documented as dead — true until the intrabc arm landed.
fn cfl_store_block_for_inter(
    env: &SbEncodeEnv,
    cfl: &mut aom_dsp::intra::cfl::CflCtx,
    recon_y: &[u16],
    ref_off_y: usize,
    winner: &LeafWinner,
    mi_row: i32,
    mi_col: i32,
) {
    let bsize = winner.bsize;
    let is_chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    if env.monochrome
        || is_chroma_ref
        || !aom_dsp::entropy::partition::is_cfl_allowed(bsize, env.lossless, env.ss_x, env.ss_y)
    {
        return;
    }
    aom_dsp::intra::cfl::cfl_store_block(
        cfl,
        recon_y,
        ref_off_y,
        env.stride,
        bsize,
        winner.tx_size,
        mi_row,
        mi_col,
        env.mi_rows,
        env.mi_cols,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn encode_b_intra_dry(
    env: &SbEncodeEnv,
    state: &mut TileCtxState,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    winner: &mut LeafWinner,
    mi_row: i32,
    mi_col: i32,
    partition: usize,
    output_enabled: bool,
) -> LeafEncodeOut {
    let bsize = winner.bsize;
    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_h = MI_SIZE_HIGH_B[bsize];
    // set_mi_row_col: neighbour availability (interior fixtures: all true).
    let up_available = mi_row > env.tile_row_start;
    let left_available = mi_col > env.tile_col_start;
    let mut chroma_up_available = up_available;
    let mut chroma_left_available = left_available;
    if env.ss_x != 0 && mi_w < 2 {
        chroma_left_available = (mi_col - 1) > env.tile_col_start;
    }
    if env.ss_y != 0 && mi_h < 2 {
        chroma_up_available = (mi_row - 1) > env.tile_row_start;
    }
    let is_chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, env.ss_x, env.ss_y);

    // Intra-block-copy leaf: predict from the recon at the DV, then (skip arm,
    // the only regime this port codes intrabc in — see rd_pick_intrabc_mode_sb)
    // reset the coeff entropy context to 0 and stamp the skip txfm context.
    // No residual, no coeff txbs. C: encode_superblock's inter arm +
    // av1_reset_entropy_context (skip) + set_txfm_ctxs(skip).
    if winner.use_intrabc {
        let bw = crate::tx_search::BLK_W_B[bsize];
        let bh = crate::tx_search::BLK_H_B[bsize];
        let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;
        let (fm_r, fm_c) = (winner.dv_row / 8, winner.dv_col / 8);
        // Luma: predict into scratch (reads recon at the DV), then commit.
        let mut pred = vec![0u16; bw * bh];
        crate::intrabc_search::intrabc_predict_luma(
            recon_y, ref_off_y, env.stride, fm_r, fm_c, &mut pred, bw, bw, bh,
        );
        for r in 0..bh {
            recon_y[ref_off_y + r * env.stride..ref_off_y + r * env.stride + bw]
                .copy_from_slice(&pred[r * bw..r * bw + bw]);
        }
        // --- COEFF arm (`!skip_txfm`): av1_encode_sb's inter path. Luma
        //     recurses on `inter_tx_size`, chroma is uniform; both reconstruct
        //     into the recon planes and stamp the real entropy contexts.
        if !winner.skip_txfm {
            let out = encode_b_intrabc_coeff(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                winner,
                mi_row,
                mi_col,
                is_chroma_ref,
                output_enabled,
            );
            cfl_store_block_for_inter(env, cfl, recon_y, ref_off_y, winner, mi_row, mi_col);
            return out;
        }

        // Reset luma coeff entropy context (skip → cul 0).
        state.above_ectx[0][a0..a0 + mi_w].fill(0);
        state.left_ectx[0][l0..l0 + mi_h].fill(0);

        let mut u_out = None;
        let mut v_out = None;
        if !env.monochrome && is_chroma_ref {
            let ref_off_uv =
                chroma_plane_offset(env.base_uv, env.stride, mi_row, mi_col, bsize, env.ss_x, env.ss_y);
            let plane_bsize = get_plane_block_size(bsize, env.ss_x, env.ss_y);
            let (pmw, pmh) = (MI_SIZE_WIDE_B[plane_bsize], MI_SIZE_HIGH_B[plane_bsize]);
            // The chroma PLANE block is padded to a 4x4 minimum, so for sub-8x8
            // luma this is LARGER than `bw >> ss_x`: a 4x8 intrabc chroma-ref
            // covers the full 4x4 chroma, not a 2x4 strip. Sizing by `bw >> ss_x`
            // left the right chroma columns unwritten (the "128 default" island),
            // corrupting the DC-pred of the CfL block below it. Mirror the COEFF
            // arm (`encode_b_intrabc_coeff`), which already uses `plane_bsize`.
            let (cw, ch) = (
                crate::tx_search::BLK_W_B[plane_bsize],
                crate::tx_search::BLK_H_B[plane_bsize],
            );
            let au = (mi_col >> env.ss_x) as usize;
            let lu = ((mi_row & 31) >> env.ss_y) as usize;
            for (plane, recon) in [(1usize, &mut *recon_u), (2usize, &mut *recon_v)] {
                let mut cpred = vec![0u16; cw * ch];
                crate::intrabc_search::intrabc_predict_chroma(
                    recon, ref_off_uv, env.stride, winner.dv_row, winner.dv_col, env.ss_x, env.ss_y,
                    &mut cpred, cw, cw, ch, i32::from(env.bd),
                );
                for r in 0..ch {
                    recon[ref_off_uv + r * env.stride..ref_off_uv + r * env.stride + cw]
                        .copy_from_slice(&cpred[r * cw..r * cw + cw]);
                }
                state.above_ectx[plane][au..au + pmw].fill(0);
                state.left_ectx[plane][lu..lu + pmh].fill(0);
            }
            u_out = Some(EncodeIntraPlaneOutcome {
                txbs: Vec::new(),
                ta: Vec::new(),
                tl: Vec::new(),
            });
            v_out = Some(EncodeIntraPlaneOutcome {
                txbs: Vec::new(),
                ta: Vec::new(),
                tl: Vec::new(),
            });
        }
        // set_txfm_ctxs skip convention: ctx = block width/height in pixels.
        state.above_tctx[a0..a0 + mi_w].fill((mi_w * 4) as u8);
        state.left_tctx[l0..l0 + mi_h].fill((mi_h * 4) as u8);

        cfl_store_block_for_inter(env, cfl, recon_y, ref_off_y, winner, mi_row, mi_col);

        return LeafEncodeOut {
            mi_row,
            mi_col,
            bsize,
            is_chroma_ref,
            store_y: false,
            y: EncodeIntraPlaneOutcome {
                txbs: Vec::new(),
                ta: Vec::new(),
                tl: Vec::new(),
            },
            u: u_out,
            v: v_out,
        };
    }

    // encode_superblock step 1: store_y = store_cfl_required(cm, xd).
    let store_y = store_cfl_required(env.monochrome, is_chroma_ref, winner.uv_mode);

    // Step 2, plane 0: av1_encode_intra_block_plane(AOM_PLANE_Y).
    let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;
    let above_y: Vec<i8> = state.above_ectx[0][a0..a0 + mi_w].to_vec();
    let left_y: Vec<i8> = state.left_ectx[0][l0..l0 + mi_h].to_vec();
    // The real per-txs_ctx table for THIS leaf's winner tx_size (uniform tx
    // only, so one lookup covers the whole leaf's luma plane).
    let y_tables = env.coeff_costs_y.tables(winner.tx_size);
    let y_env = EncodeIntraYEnv {
        sb_size: env.sb_size,
        bsize,
        mi_row,
        mi_col,
        up_available,
        left_available,
        tile_col_end: env.tile_col_end,
        tile_row_end: env.tile_row_end,
        partition,
        mi_cols: env.mi_cols,
        mi_rows: env.mi_rows,
        ref_off: ref_off_y,
        ref_stride: env.stride,
        src: env.src_y,
        src_off: ref_off_y,
        src_stride: env.stride,
        disable_edge_filter: env.disable_edge_filter,
        // Per-block luma edge filter (`get_intra_edge_filter_type(xd, plane=0)`),
        // recomputed at search time from the neighbour Y modes and carried on the
        // winner (KB-6) — NOT the frozen SB-level `env.filter_type`. KB-2 fixed
        // the luma SEARCH; this fixes the luma RE-ENCODE (the coded-bytes path),
        // mirroring the #26 chroma fix.
        filter_type: winner.luma_edge_filter_type,
        mode: winner.mode,
        angle_delta: winner.angle_delta_y,
        use_filter_intra: winner.use_filter_intra,
        filter_intra_mode: winner.filter_intra_mode,
        tx_size: winner.tx_size,
        skip_txfm: winner.skip_txfm,
        lossless: env.lossless,
        reduced_tx_set_used: env.reduced_tx_set_used,
        bd: env.bd,
        rows: env.rows_y,
        rdmult: env.rdmult,
        sharpness: env.sharpness,
        coeff_costs: &y_tables,
        enable_optimize_b: env.enable_optimize_b,
        // C's `dry_run == OUTPUT_ENABLED` (tokenize.h) — the pack/winner walk
        // passes `output_enabled = true`, the search's DRY_RUN_NORMAL context
        // propagation passes `false`. Byte-inert for NO/FULL/NO_ESTIMATE_YRD
        // trellis (is_trellis_used is flag-independent there — every existing
        // gate); load-bearing ONLY for FINAL_PASS_TRELLIS_OPT
        // (--disable-trellis-quant=2), where the final pack must trellis and
        // the search must not (init_rd_sf, is_trellis_used, encodemb.h:153).
        dry_run_output_enabled: output_enabled,
        above_ctx: &above_y,
        left_ctx: &left_y,
        qm_level: env.qm_levels.map(|l| l[0]),
        palette: winner
            .palette_y
            .as_ref()
            .map(|p| crate::tx_search::PaletteYrd {
                colors: &p.colors,
                size: p.size,
                map: &p.color_map,
                map_stride: crate::tx_search::BLK_W_B[bsize],
            }),
        tune: env.tune,
    };
    let mut y_out = if output_enabled {
        // OUTPUT_ENABLED: the eob-0 -> DCT_DCT resets land in the frame-map
        // copy (transient here — the pack writes tx_type syntax only for
        // eob > 0 txbs, whose entries the reset never touches), and the
        // winner's map keeps the state the search left it in. See the fn doc.
        let mut frame_tx_type_map = winner.tx_type_map.clone();
        encode_intra_block_plane_y(
            &y_env,
            recon_y,
            &mut frame_tx_type_map,
            if store_y { Some(cfl) } else { None },
        )
    } else {
        // DRY_RUN_NORMAL: alias — the resets persist into the winner map.
        encode_intra_block_plane_y(
            &y_env,
            recon_y,
            &mut winner.tx_type_map,
            if store_y { Some(cfl) } else { None },
        )
    };

    // Step 2, planes 1/2 (early return inside the C when !is_chroma_ref).
    let mut u_out = None;
    let mut v_out = None;
    let uv_tx = av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
    if !env.monochrome && is_chroma_ref {
        let ref_off_uv = chroma_plane_offset(
            env.base_uv,
            env.stride,
            mi_row,
            mi_col,
            bsize,
            env.ss_x,
            env.ss_y,
        );
        let plane_bsize = get_plane_block_size(bsize, env.ss_x, env.ss_y);
        let (pmw, pmh) = (MI_SIZE_WIDE_B[plane_bsize], MI_SIZE_HIGH_B[plane_bsize]);
        let au = (mi_col >> env.ss_x) as usize;
        let lu = ((mi_row & 31) >> env.ss_y) as usize;
        let above_u: Vec<i8> = state.above_ectx[1][au..au + pmw].to_vec();
        let left_u: Vec<i8> = state.left_ectx[1][lu..lu + pmh].to_vec();
        let above_v: Vec<i8> = state.above_ectx[2][au..au + pmw].to_vec();
        let left_v: Vec<i8> = state.left_ectx[2][lu..lu + pmh].to_vec();
        // The real per-txs_ctx table for THIS leaf's uniform UV tx_size.
        let uv_tables = env.coeff_costs_uv.tables(uv_tx);
        let uv_env = UvRdEnv {
            sb_size: env.sb_size,
            bsize,
            mi_row,
            mi_col,
            chroma_up_available,
            chroma_left_available,
            tile_col_end: env.tile_col_end,
            tile_row_end: env.tile_row_end,
            partition,
            mi_cols: env.mi_cols,
            mi_rows: env.mi_rows,
            ss_x: env.ss_x,
            ss_y: env.ss_y,
            ref_off: [ref_off_uv, ref_off_uv],
            ref_stride: env.stride,
            src_u: env.src_u,
            src_v: env.src_v,
            src_off: [ref_off_uv, ref_off_uv],
            src_stride: env.stride,
            disable_edge_filter: env.disable_edge_filter,
            // Per-block chroma edge filter (`get_intra_edge_filter_type(xd,
            // plane=1)`), recomputed at search time from the chroma
            // neighbours' UV modes and carried on the winner — NOT the frozen
            // SB-level `env.filter_type` (which only ever held 0 and mis-
            // predicted chroma when an above/left UV neighbour was SMOOTH).
            filter_type: winner.uv_edge_filter_type,
            // The luma-winner fields feed the RD tx-type MASK only — the
            // encode arm derives its tx type from uv_mode alone. Threaded
            // for completeness.
            luma_mode: winner.mode,
            luma_use_fi: winner.use_filter_intra,
            luma_fi_mode: winner.filter_intra_mode,
            luma_palette_active: winner.palette_y.is_some(),
            lossless: env.lossless,
            reduced_tx_set_used: env.reduced_tx_set_used,
            bd: env.bd,
            rows_u: env.rows_u,
            rows_v: env.rows_v,
            rdmult: env.rdmult,
            coeff_costs: &uv_tables,
            tx_type_costs: env.tx_type_costs,
            above_ctx: [&above_u, &above_v],
            left_ctx: [&left_u, &left_v],
            qm_levels: env.qm_levels,
        };
        let uv_winner = UvWinner {
            uv_mode: winner.uv_mode,
            angle_delta_uv: winner.angle_delta_uv,
            cfl_alpha_idx: winner.cfl_alpha_idx,
            cfl_alpha_signs: winner.cfl_alpha_signs,
            palette: winner.palette_uv.as_ref().map(|p| {
                let (pw, _, _, _) = crate::palette_search::chroma_block_dims(
                    bsize,
                    mi_row,
                    mi_col,
                    env.mi_rows,
                    env.mi_cols,
                    env.ss_x,
                    env.ss_y,
                );
                crate::intra_uv_rd::PaletteUvPred {
                    colors_u: &p.colors_u,
                    colors_v: &p.colors_v,
                    size: p.size,
                    map: &p.color_map,
                    map_stride: pw,
                }
            }),
        };
        let prm = UvEncodeParams {
            tx_size: uv_tx,
            skip_txfm: winner.skip_txfm,
            sharpness: env.sharpness,
            enable_optimize_b: env.enable_optimize_b,
            // See the luma `dry_run_output_enabled` note above: `output_enabled`
            // is C's OUTPUT_ENABLED flag; only FINAL_PASS_TRELLIS_OPT reads it
            // (the chroma final pack must trellis, the search must not).
            dry_run_output_enabled: output_enabled,
            use_chroma_trellis_rd_mult: env.use_chroma_trellis_rd_mult,
            tune: env.tune,
        };
        u_out = Some(encode_intra_block_plane_uv(
            &uv_env, &uv_winner, &prm, 1, recon_u, cfl,
        ));
        v_out = Some(encode_intra_block_plane_uv(
            &uv_env, &uv_winner, &prm, 2, recon_v, cfl,
        ));
    }

    // Step 4: av1_update_intra_mb_txb_context at DRY_RUN — the tile-level
    // entropy-context stamps (cul_level recomputed from the final qcoeff,
    // tx types re-read per av1_get_tx_type: the Y map AFTER the eob-0
    // resets / the UV arm). skip_txfm reset arm dead (KEY intra skip == 0).
    debug_assert!(!winner.skip_txfm, "KEY intra skip_txfm is 0");
    {
        // Plane 0.
        let (txw_u, txh_u) = (TXS_W[winner.tx_size] >> 2, TXS_H[winner.tx_size] >> 2);
        let map_stride = mi_w;
        // The Y encode produced txbs for the VISIBLE tx blocks only (the
        // frame-edge `max_block_wide/high` clip); iterate the same range so
        // `k` stays in `y_out.txbs` bounds and the entropy-context stamp
        // covers exactly the coded (in-frame) tx blocks, matching C's
        // edge-clipped `av1_set_entropy_contexts`. Map stride stays full.
        let bwv = mi_w.min((env.mi_cols - mi_col).max(0) as usize);
        let bhv = mi_h.min((env.mi_rows - mi_row).max(0) as usize);
        // mu-64 chunk walk (the tokenize `av1_tokenize_sb_vartx` is plane-outer
        // but 64-chunked within each plane): `k` must advance in the SAME
        // chunk-major order `encode_intra_block_plane_y` produced the txbs, so
        // the persistent-context stamp + the cached-ctx overwrite land on the
        // right txb. Luma mu = 16; one chunk (byte-identical) for bsize <= 64.
        let mu_w = MI_SIZE_WIDE_B[12].min(bwv); // BLOCK_64X64
        let mu_h = MI_SIZE_HIGH_B[12].min(bhv);
        let mut k = 0usize;
        let mut chunk_r = 0usize;
        while chunk_r < bhv {
            let unit_h = (chunk_r + mu_h).min(bhv);
            let mut chunk_c = 0usize;
            while chunk_c < bwv {
                let unit_w = (chunk_c + mu_w).min(bwv);
                let mut blk_row = chunk_r;
                while blk_row < unit_h {
                    let mut blk_col = chunk_c;
                    while blk_col < unit_w {
                let tt = crate::encode_intra::get_tx_type_y(
                    env.lossless,
                    winner.tx_size,
                    &winner.tx_type_map,
                    map_stride,
                    blk_row,
                    blk_col,
                );
                // C's tokenize (`av1_update_and_record_txb_context`,
                // encodetxb.c, OUTPUT_ENABLED arm) derives the WRITE-side
                // `(txb_skip_ctx, dc_sign_ctx)` from the PERSISTENT
                // above/left entropy arrays — whose within-leaf state at this
                // point carries the earlier txbs' edge-CLIPPED
                // `av1_set_entropy_contexts` stamps — and caches it in
                // `cb_coef_buff->entropy_ctx` for `av1_write_coeffs_txb`. The
                // encode walk's local ta/tl (full-footprint
                // `av1_set_txb_context` stamps) feed ONLY the trellis. For
                // interior txbs the two derivations agree (no clip); at a
                // frame-edge txb whose within-leaf neighbour footprint spans
                // the visible boundary they can differ — the tail-zeroed
                // cells drop out of the dc-sign SUM (and the skip-ctx OR).
                // KB-6 196² cq48: txb blk(8,0) of the mi(0,48) 32×64 leaf read
                // dc_sign = left(-4) + above(+4 full-stamp) = 0 → ctx 0 in the
                // port vs C's left(-4) + above(+2 clipped) = -2 → ctx 1 — the
                // DC-sign symbol went to a different cdf row, diverging the
                // bits mid-tile with IDENTICAL search decisions. Overwrite the
                // cached pair with the tokenize-derived one; the pack writes
                // with these rows (pack.rs `pack_plane_coeffs`).
                let (tok_tsc, tok_dsc) = get_txb_ctx(
                    bsize,
                    winner.tx_size,
                    0,
                    &state.above_ectx[0][a0 + blk_col..],
                    &state.left_ectx[0][l0 + blk_row..],
                );
                {
                    let t = &mut y_out.txbs[k];
                    t.txb_skip_ctx = tok_tsc as usize;
                    // C caches dc_sign_ctx only when the DC (tcoeff[0]) is
                    // nonzero (`entropy_ctx[block] |= dc_sign_ctx << 4`);
                    // otherwise the cached field stays 0 (and the writer
                    // never reads it — no DC sign symbol is coded).
                    t.dc_sign_ctx = if t.qcoeff.first().copied().unwrap_or(0) != 0 {
                        tok_dsc as usize
                    } else {
                        0
                    };
                }
                let txb = &y_out.txbs[k];
                let cul = txb_entropy_context(&txb.qcoeff, winner.tx_size, tt, txb.eob as usize);
                // The recompute equals the encode walk's stored ctx: eob==0
                // gives 0 under any scan, and eob>0 txbs kept their map type
                // (only eob-0 origins reset to DCT).
                debug_assert_eq!(cul, txb.txb_entropy_ctx, "tokenize cul == encode ctx");
                // av1_set_entropy_contexts (blockd.c:29): interior txbs memset
                // the FULL tx footprint with the cul; at a frame edge the
                // beyond-visible TAIL of the footprint is memset to ZERO
                // (`memset(a + above_contexts, 0, txs_wide - above_contexts)`,
                // both the mb_to_right_edge/above and mb_to_bottom_edge/left
                // arms — with cul==0 the else arm zeroes the full width, same
                // result). Stamping the cul across the tail instead poisons
                // out-of-frame columns/rows that a LATER edge block's
                // full-footprint `get_txb_ctx` read ORs in, flipping its
                // txb_skip_ctx (KB-6 196x196: phantom cul at mi cols 50-51
                // shifted SB(32,48)'s skip ctx 1->3, +3 bits, stream desync).
                let vis_w = txw_u.min(bwv - blk_col);
                let vis_h = txh_u.min(bhv - blk_row);
                for (i, x) in state.above_ectx[0][a0 + blk_col..a0 + blk_col + txw_u]
                    .iter_mut()
                    .enumerate()
                {
                    *x = if i < vis_w { cul as i8 } else { 0 };
                }
                for (i, x) in state.left_ectx[0][l0 + blk_row..l0 + blk_row + txh_u]
                    .iter_mut()
                    .enumerate()
                {
                    *x = if i < vis_h { cul as i8 } else { 0 };
                }
                k += 1;
                blk_col += txw_u;
                    }
                    blk_row += txh_u;
                }
                chunk_c += mu_w;
            }
            chunk_r += mu_h;
        }
        debug_assert_eq!(k, y_out.txbs.len());
        // Planes 1/2 (the C loop breaks when !is_chroma_ref).
        if !env.monochrome && is_chroma_ref {
            let plane_bsize = get_plane_block_size(bsize, env.ss_x, env.ss_y);
            // Iterate only the VISIBLE chroma tx blocks (frame-edge
            // `max_block_wide/high` clip WITH chroma subsampling), matching the
            // UV encode's txb count so `k` stays in `out.txbs` bounds and the
            // entropy-context stamp covers exactly the coded (in-frame) tx
            // blocks — mirrors `encode_intra_block_plane_uv`. Interior blocks
            // return the full plane block; the context arrays stay full size.
            let (pmw, pmh, _, _) = max_block_units(
                env.mi_cols,
                env.mi_rows,
                mi_col,
                mi_row,
                MI_SIZE_WIDE_B[bsize] as i32,
                MI_SIZE_HIGH_B[bsize] as i32,
                MI_SIZE_WIDE_B[plane_bsize] * 4,
                MI_SIZE_HIGH_B[plane_bsize] * 4,
                env.ss_x,
                env.ss_y,
            );
            let (ptxw_u, ptxh_u) = (TXS_W[uv_tx] >> 2, TXS_H[uv_tx] >> 2);
            let au = (mi_col >> env.ss_x) as usize;
            let lu = ((mi_row & 31) >> env.ss_y) as usize;
            let uv_tt = crate::tx_search::uv_intra_tx_type(
                winner.uv_mode,
                env.lossless,
                uv_tx,
                env.reduced_tx_set_used,
            );
            // Chroma mu-64 chunk unit (get_plane_block_size(BLOCK_64X64, ss),
            // encodemb.c:560-561): chunk-major to match the chunk-major chroma
            // txb arrays. One chunk (byte-identical) for a chroma block <= 64.
            let uv_unit_bsize = get_plane_block_size(12, env.ss_x, env.ss_y);
            let cmu_w = MI_SIZE_WIDE_B[uv_unit_bsize].min(pmw);
            let cmu_h = MI_SIZE_HIGH_B[uv_unit_bsize].min(pmh);
            for (plane, out) in [(1usize, u_out.as_mut()), (2usize, v_out.as_mut())] {
                let out = out.expect("chroma-ref leaf has uv outcomes");
                let mut k = 0usize;
                let mut chunk_r = 0usize;
                while chunk_r < pmh {
                    let unit_h = (chunk_r + cmu_h).min(pmh);
                    let mut chunk_c = 0usize;
                    while chunk_c < pmw {
                        let unit_w = (chunk_c + cmu_w).min(pmw);
                        let mut blk_row = chunk_r;
                        while blk_row < unit_h {
                        let mut blk_col = chunk_c;
                        while blk_col < unit_w {
                        // Tokenize-derived WRITE ctx from the persistent
                        // (clipped-stamp) arrays — see the plane-0 stamp
                        // above for the C mapping + KB-6 mechanism.
                        let (tok_tsc, tok_dsc) = get_txb_ctx(
                            plane_bsize,
                            uv_tx,
                            plane,
                            &state.above_ectx[plane][au + blk_col..],
                            &state.left_ectx[plane][lu + blk_row..],
                        );
                        {
                            let t = &mut out.txbs[k];
                            t.txb_skip_ctx = tok_tsc as usize;
                            t.dc_sign_ctx = if t.qcoeff.first().copied().unwrap_or(0) != 0 {
                                tok_dsc as usize
                            } else {
                                0
                            };
                        }
                        let txb = &out.txbs[k];
                        let cul = txb_entropy_context(&txb.qcoeff, uv_tx, uv_tt, txb.eob as usize);
                        debug_assert_eq!(cul, txb.txb_entropy_ctx, "uv tokenize cul == encode ctx");
                        // av1_set_entropy_contexts edge arms (see the luma
                        // stamp above): the beyond-visible tail of the tx
                        // footprint is ZERO, not the cul.
                        let vis_w = ptxw_u.min(pmw - blk_col);
                        let vis_h = ptxh_u.min(pmh - blk_row);
                        for (i, x) in state.above_ectx[plane][au + blk_col..au + blk_col + ptxw_u]
                            .iter_mut()
                            .enumerate()
                        {
                            *x = if i < vis_w { cul as i8 } else { 0 };
                        }
                        for (i, x) in state.left_ectx[plane][lu + blk_row..lu + blk_row + ptxh_u]
                            .iter_mut()
                            .enumerate()
                        {
                            *x = if i < vis_h { cul as i8 } else { 0 };
                        }
                        k += 1;
                        blk_col += ptxw_u;
                        }
                        blk_row += ptxh_u;
                        }
                        chunk_c += cmu_w;
                    }
                    chunk_r += cmu_h;
                }
                debug_assert_eq!(k, out.txbs.len());
            }
        }
    }

    // Step 6: mbmi->tx_size stays (intra: (bsize > 4x4) ? tx_size : TX_4X4 —
    // a no-op for picked winners) + set_txfm_ctxs(tx_size, mi_w, mi_h, 0).
    // C: `(bsize > BLOCK_4X4) ? tx_size : TX_4X4` (BLOCK_4X4 == 0); a 4x4
    // winner's tx_size is TX_4X4 already, so this is a no-op for picked
    // winners (asserted).
    let final_tx = if bsize > 0 { winner.tx_size } else { 0 };
    debug_assert_eq!(
        final_tx, winner.tx_size,
        "picked winners already satisfy the stamp"
    );
    for x in state.above_tctx[a0..a0 + mi_w].iter_mut() {
        *x = TXS_W[final_tx] as u8;
    }
    for x in state.left_tctx[l0..l0 + mi_h].iter_mut() {
        *x = TXS_H[final_tx] as u8;
    }

    LeafEncodeOut {
        mi_row,
        mi_col,
        bsize,
        is_chroma_ref,
        store_y,
        y: y_out,
        u: u_out,
        v: v_out,
    }
}

/// `encode_sb` over a NONE/SPLIT tree — see the module docs. Appends each
/// leaf's outputs to `leaves` in walk order. `output_enabled` selects C's
/// `RUN_TYPE` tx_type_map semantics per leaf — see [`encode_b_intra_dry`]:
/// `false` for the search's DRY_RUN context-propagation walks (winner-map
/// resets persist, C's ctx alias), `true` for the SB-root winner walk and
/// the pack re-walk (C's OUTPUT_ENABLED frame-map copy; winner maps stay
/// as the search left them).
#[allow(clippy::too_many_arguments)]
pub fn encode_sb_dry(
    env: &SbEncodeEnv,
    state: &mut TileCtxState,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    tree: &mut SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    leaves: &mut Vec<LeafEncodeOut>,
    output_enabled: bool,
) {
    if mi_row >= env.mi_rows || mi_col >= env.mi_cols {
        return;
    }
    let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
    let (partition, subsize) = match tree {
        SbTree::Leaf(_) => (PARTITION_NONE, bsize),
        SbTree::Split(_) => (PARTITION_SPLIT, split_subsize(bsize)),
        SbTree::Horz(_) => (
            PARTITION_HORZ,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ) as usize,
        ),
        SbTree::Vert(_) => (
            PARTITION_VERT,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT) as usize,
        ),
        SbTree::Horz4(_) => (
            PARTITION_HORZ_4,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_4) as usize,
        ),
        SbTree::Vert4(_) => (
            PARTITION_VERT_4,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_4) as usize,
        ),
        SbTree::HorzA(_) => (
            PARTITION_HORZ_A,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_A) as usize,
        ),
        SbTree::HorzB(_) => (
            PARTITION_HORZ_B,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_B) as usize,
        ),
        SbTree::VertA(_) => (
            PARTITION_VERT_A,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_A) as usize,
        ),
        SbTree::VertB(_) => (
            PARTITION_VERT_B,
            aom_dsp::entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_B) as usize,
        ),
        // Off-frame placeholder: unreachable here (the entry guard returned
        // for its off-frame origin), but the match must be exhaustive.
        SbTree::Absent => return,
    };
    debug_assert_ne!(subsize, 255, "tree subsize is valid by construction");
    // !dry_run partition-CDF update: pack stage (skipped at DRY_RUN).
    match tree {
        SbTree::Leaf(w) => {
            debug_assert_eq!(w.bsize, bsize, "leaf winner bsize == tree subsize");
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                w,
                mi_row,
                mi_col,
                PARTITION_NONE as usize,
                output_enabled,
            );
            leaves.push(out);
        }
        SbTree::Split(children) => {
            for (idx, child) in children.iter_mut().enumerate() {
                let y = mi_row + ((idx as i32) >> 1) * hbs;
                let x = mi_col + ((idx as i32) & 1) * hbs;
                encode_sb_dry(
                    env,
                    state,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    child,
                    y,
                    x,
                    subsize,
                    leaves,
                    output_enabled,
                );
            }
        }
        SbTree::Horz(subs) => {
            // encode_sb PARTITION_HORZ (:1637-1644): sub-0 at the origin,
            // sub-1 at (mi_row + hbs, mi_col) gated by the frame bound.
            let [s0, s1] = &mut **subs;
            debug_assert_eq!(s0.bsize, subsize, "horz winner bsize == subsize");
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_HORZ as usize,
                output_enabled,
            );
            leaves.push(out);
            if mi_row + hbs < env.mi_rows {
                debug_assert_eq!(s1.bsize, subsize, "horz winner bsize == subsize");
                let out = encode_b_intra_dry(
                    env,
                    state,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s1,
                    mi_row + hbs,
                    mi_col,
                    PARTITION_HORZ as usize,
                    output_enabled,
                );
                leaves.push(out);
            }
        }
        SbTree::Vert(subs) => {
            // encode_sb PARTITION_VERT (:1629-1636): sub-0 at the origin,
            // sub-1 at (mi_row, mi_col + hbs) gated by the frame bound.
            let [s0, s1] = &mut **subs;
            debug_assert_eq!(s0.bsize, subsize, "vert winner bsize == subsize");
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_VERT as usize,
                output_enabled,
            );
            leaves.push(out);
            if mi_col + hbs < env.mi_cols {
                debug_assert_eq!(s1.bsize, subsize, "vert winner bsize == subsize");
                let out = encode_b_intra_dry(
                    env,
                    state,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s1,
                    mi_row,
                    mi_col + hbs,
                    PARTITION_VERT as usize,
                    output_enabled,
                );
                leaves.push(out);
            }
        }
        SbTree::Horz4(subs) => {
            // encode_sb PARTITION_HORZ_4 (:1690-1697): 4 strips at
            // mi_row + i*quarter_step, i>0 gated by the frame bound.
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, s) in subs.iter_mut().enumerate() {
                let this_mi_row = mi_row + (i as i32) * quarter_step;
                if i > 0 && this_mi_row >= env.mi_rows {
                    break;
                }
                debug_assert_eq!(s.bsize, subsize, "horz4 winner bsize == subsize");
                let out = encode_b_intra_dry(
                    env,
                    state,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    this_mi_row,
                    mi_col,
                    PARTITION_HORZ_4 as usize,
                    output_enabled,
                );
                leaves.push(out);
            }
        }
        SbTree::Vert4(subs) => {
            // encode_sb PARTITION_VERT_4 (:1699-1705): 4 strips at
            // mi_col + i*quarter_step, i>0 gated by the frame bound.
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, s) in subs.iter_mut().enumerate() {
                let this_mi_col = mi_col + (i as i32) * quarter_step;
                if i > 0 && this_mi_col >= env.mi_cols {
                    break;
                }
                debug_assert_eq!(s.bsize, subsize, "vert4 winner bsize == subsize");
                let out = encode_b_intra_dry(
                    env,
                    state,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    mi_row,
                    this_mi_col,
                    PARTITION_VERT_4 as usize,
                    output_enabled,
                );
                leaves.push(out);
            }
        }
        SbTree::HorzA(subs) => {
            // encode_sb PARTITION_HORZ_A (:1652-1660): no frame-bound gating
            // on any sub-block (AB is interior-only by construction — module
            // docs on the SbTree::HorzA variant).
            let bsize2 = split_subsize(bsize);
            let [s0, s1, s2] = &mut **subs;
            debug_assert_eq!(s0.bsize, bsize2);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_HORZ_A as usize,
                output_enabled,
            );
            leaves.push(out);
            debug_assert_eq!(s1.bsize, bsize2);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s1,
                mi_row,
                mi_col + hbs,
                PARTITION_HORZ_A as usize,
                output_enabled,
            );
            leaves.push(out);
            debug_assert_eq!(s2.bsize, subsize);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s2,
                mi_row + hbs,
                mi_col,
                PARTITION_HORZ_A as usize,
                output_enabled,
            );
            leaves.push(out);
        }
        SbTree::HorzB(subs) => {
            // encode_sb PARTITION_HORZ_B (:1661-1667).
            let bsize2 = split_subsize(bsize);
            let [s0, s1, s2] = &mut **subs;
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_HORZ_B as usize,
                output_enabled,
            );
            debug_assert_eq!(s0.bsize, subsize);
            leaves.push(out);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s1,
                mi_row + hbs,
                mi_col,
                PARTITION_HORZ_B as usize,
                output_enabled,
            );
            debug_assert_eq!(s1.bsize, bsize2);
            leaves.push(out);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s2,
                mi_row + hbs,
                mi_col + hbs,
                PARTITION_HORZ_B as usize,
                output_enabled,
            );
            debug_assert_eq!(s2.bsize, bsize2);
            leaves.push(out);
        }
        SbTree::VertA(subs) => {
            // encode_sb PARTITION_VERT_A (:1668-1676): column-axis mirror of
            // HORZ_A.
            let bsize2 = split_subsize(bsize);
            let [s0, s1, s2] = &mut **subs;
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_VERT_A as usize,
                output_enabled,
            );
            debug_assert_eq!(s0.bsize, bsize2);
            leaves.push(out);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s1,
                mi_row + hbs,
                mi_col,
                PARTITION_VERT_A as usize,
                output_enabled,
            );
            debug_assert_eq!(s1.bsize, bsize2);
            leaves.push(out);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s2,
                mi_row,
                mi_col + hbs,
                PARTITION_VERT_A as usize,
                output_enabled,
            );
            debug_assert_eq!(s2.bsize, subsize);
            leaves.push(out);
        }
        SbTree::VertB(subs) => {
            // encode_sb PARTITION_VERT_B (:1677-1684): column-axis mirror of
            // HORZ_B.
            let bsize2 = split_subsize(bsize);
            let [s0, s1, s2] = &mut **subs;
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_VERT_B as usize,
                output_enabled,
            );
            debug_assert_eq!(s0.bsize, subsize);
            leaves.push(out);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s1,
                mi_row,
                mi_col + hbs,
                PARTITION_VERT_B as usize,
                output_enabled,
            );
            debug_assert_eq!(s1.bsize, bsize2);
            leaves.push(out);
            let out = encode_b_intra_dry(
                env,
                state,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s2,
                mi_row + hbs,
                mi_col + hbs,
                PARTITION_VERT_B as usize,
                output_enabled,
            );
            debug_assert_eq!(s2.bsize, bsize2);
            leaves.push(out);
        }
        // Off-frame placeholder — unreachable past the entry frame-bound guard.
        SbTree::Absent => {}
    }
    update_ext_partition_context(
        &mut state.above_pctx,
        &mut state.left_pctx,
        mi_row,
        mi_col,
        subsize,
        bsize,
        partition,
    );
}

// ===========================================================================
// IntraBC COEFF arm re-encode — `av1_encode_sb`'s inter path (encodemb.c:636)
// over `encode_block_inter` (:482).
//
// Plane-outer (Y, U, V), each plane mu-64 chunked, and within a chunk a raster
// over `max_tx_size` units. LUMA recurses on `mbmi->inter_tx_size` until
// `tx_size == plane_tx_size`; CHROMA is UNIFORM (`av1_get_max_uv_txsize`) and
// takes `encode_block` directly (:505 `if (tx_size == plane_tx_size || plane)`).
// ===========================================================================

/// Shared per-txb state for the intrabc coeff-arm re-encode.
struct IbcTxbCtx<'a> {
    plane: usize,
    plane_bsize: usize,
    tx_size_uniform_uv: usize,
    bd: u8,
    lossless: bool,
    stride: usize,
    ref_off: usize,
    src: &'a [u16],
    src_off: usize,
    rows: &'a aom_dsp::quant::PlaneQuantRows<'a>,
    coeff_costs: &'a CoeffCostSet,
    rdmult: i32,
    sharpness: i32,
    iq_tuning: bool,
    qm_level: Option<usize>,
    use_trellis: bool,
    max_blocks_wide: usize,
    max_blocks_high: usize,
}

/// One txb of the coeff-arm re-encode: transform + quantize (+ trellis),
/// reconstruct into `recon`, and stamp the local entropy context.
/// Mirrors `encode_block` (encodemb.c:390) on the inter path — no intra
/// prediction (the DV copy already wrote `recon`), and the residual is taken
/// against the CURRENT recon contents (the prediction).
#[allow(clippy::too_many_arguments)]
fn ibc_encode_txb(
    c: &IbcTxbCtx,
    recon: &mut [u16],
    ta: &mut [i8],
    tl: &mut [i8],
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    tx_type: usize,
) -> TxbEncode {
    let (txw, txh) = (TXS_W[tx_size], TXS_H[tx_size]);
    let (txw_u, txh_u) = (txw >> 2, txh >> 2);
    let txb_off = c.ref_off + (blk_row * c.stride + blk_col) * 4;
    let src_txb_off = c.src_off + (blk_row * c.stride + blk_col) * 4;

    // The prediction currently occupying the recon plane at this txb.
    let mut pred = vec![0u16; txw * txh];
    for r in 0..txh {
        pred[r * txw..r * txw + txw]
            .copy_from_slice(&recon[txb_off + r * c.stride..txb_off + r * c.stride + txw]);
    }
    // av1_subtract_txb.
    let mut residual = vec![0i16; txw * txh];
    for r in 0..txh {
        for col in 0..txw {
            residual[r * txw + col] = (i32::from(c.src[src_txb_off + r * c.stride + col])
                - i32::from(pred[r * txw + col])) as i16;
        }
    }

    let kind = if c.use_trellis {
        crate::QuantKind::Fp
    } else {
        crate::QuantKind::B
    };
    let mut qp = crate::QuantParams::from_plane_rows(c.rows, kind, c.bd, c.lossless);
    if let Some(level) = c.qm_level {
        qp = qp.with_qm(level, 0);
    }
    let tables = c.coeff_costs.tables(tx_size);
    let (qcoeff, dqcoeff, eob, ent_ctx, txb_skip_ctx, dc_sign_ctx);
    if c.use_trellis {
        let bctx = crate::BlockContext {
            above: &ta[blk_col..],
            left: &tl[blk_row..],
            plane: c.plane,
            plane_bsize: c.plane_bsize,
        };
        let opt = crate::OptimizeInputs {
            cost: &tables,
            // The INTER trellis rd-mult rows (encodetxb.h:266-273): luma 16,
            // chroma 10 under the allintra `use_chroma_trellis_rd_mult`.
            rdmult: if c.plane > 0 {
                crate::var_tx::trellis_rdmult_inter_uv(c.rdmult, c.sharpness, c.bd, c.iq_tuning)
            } else {
                crate::var_tx::trellis_rdmult_inter_y(c.rdmult, c.sharpness, c.bd, c.iq_tuning)
            },
            sharpness: c.sharpness,
        };
        let r = crate::xform_quant_optimize(&residual, tx_size, tx_type, kind, &qp, &bctx, &opt);
        qcoeff = r.qcoeff;
        dqcoeff = r.dqcoeff;
        eob = r.eob;
        ent_ctx = r.txb_entropy_ctx;
        txb_skip_ctx = r.txb_skip_ctx;
        dc_sign_ctx = r.dc_sign_ctx;
    } else {
        let r = crate::xform_quant(&residual, tx_size, tx_type, kind, &qp, false);
        qcoeff = r.qcoeff;
        dqcoeff = r.dqcoeff;
        eob = r.eob;
        ent_ctx = r.txb_entropy_ctx;
        let (sc, dc) = get_txb_ctx(
            c.plane_bsize,
            tx_size,
            c.plane,
            &ta[blk_col..],
            &tl[blk_row..],
        );
        txb_skip_ctx = sc as usize;
        dc_sign_ctx = dc as usize;
    }

    // if (*eob) av1_inverse_transform_block -> recon.
    if eob > 0 {
        let mut tight = pred.clone();
        aom_dsp::transform::inv_txfm2d::av1_inverse_transform_add(
            &dqcoeff,
            &mut tight,
            txw,
            tx_type,
            tx_size,
            i32::from(c.bd),
            eob as usize,
            c.lossless,
        );
        for r in 0..txh {
            recon[txb_off + r * c.stride..txb_off + r * c.stride + txw]
                .copy_from_slice(&tight[r * txw..r * txw + txw]);
        }
    }

    // av1_set_txb_context (full txb footprint).
    let a_end = (blk_col + txw_u).min(ta.len());
    let l_end = (blk_row + txh_u).min(tl.len());
    for a in ta[blk_col..a_end].iter_mut() {
        *a = ent_ctx as i8;
    }
    for l in tl[blk_row..l_end].iter_mut() {
        *l = ent_ctx as i8;
    }
    let _ = (c.tx_size_uniform_uv, c.max_blocks_wide, c.max_blocks_high);

    TxbEncode {
        tx_type,
        eob,
        txb_entropy_ctx: ent_ctx,
        qcoeff,
        dqcoeff,
        txb_skip_ctx,
        dc_sign_ctx,
    }
}

/// `encode_block_inter` (encodemb.c:482) LUMA recursion: descend until
/// `tx_size == inter_tx_size[get_txb_size_index(bsize, blk_row, blk_col)]`,
/// then encode that leaf. Appends txbs in C's depth-first order.
#[allow(clippy::too_many_arguments)]
fn ibc_encode_block_inter_y(
    c: &IbcTxbCtx,
    recon: &mut [u16],
    ta: &mut [i8],
    tl: &mut [i8],
    pers_a: &mut [i8],
    pers_l: &mut [i8],
    out: &mut Vec<TxbEncode>,
    tx_type_map: &mut [u8],
    map_stride: usize,
    inter_tx_size: &[usize; 16],
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
) {
    if blk_row >= c.max_blocks_high || blk_col >= c.max_blocks_wide {
        return;
    }
    let idx = crate::var_tx::get_txb_size_index(c.plane_bsize, blk_row, blk_col);
    let plane_tx_size = inter_tx_size[idx];
    if tx_size == plane_tx_size {
        // av1_get_tx_type(PLANE_TYPE_Y): the map entry at this txb origin.
        // av1_get_tx_type(PLANE_TYPE_Y) — DCT_DCT at lossless / sqr_up > TX_32X32,
        // else the map entry at this txb origin.
        let tx_type = crate::encode_intra::get_tx_type_y(
            c.lossless,
            tx_size,
            tx_type_map,
            map_stride,
            blk_row,
            blk_col,
        );
        // C's tokenize (`av1_update_and_record_txb_context`) derives the
        // WRITE-side (txb_skip_ctx, dc_sign_ctx) from the PERSISTENT arrays,
        // read BEFORE this txb's own (edge-clipped) stamp — KB-6's write-ctx
        // root. The local ta/tl (full-footprint) feed only the trellis.
        let (tok_tsc, tok_dsc) = get_txb_ctx(
            c.plane_bsize,
            tx_size,
            c.plane,
            &pers_a[blk_col..],
            &pers_l[blk_row..],
        );
        let mut txb = ibc_encode_txb(c, recon, ta, tl, blk_row, blk_col, tx_size, tx_type);
        txb.txb_skip_ctx = tok_tsc as usize;
        txb.dc_sign_ctx = if txb.qcoeff.first().copied().unwrap_or(0) != 0 {
            tok_dsc as usize
        } else {
            0
        };
        // av1_set_entropy_contexts (blockd.c:29): full footprint inside the
        // visible area, ZERO beyond it.
        ibc_stamp_persistent(
            pers_a,
            pers_l,
            blk_row,
            blk_col,
            tx_size,
            txb.txb_entropy_ctx as i8,
            c.max_blocks_wide,
            c.max_blocks_high,
        );
        // `if (*eob == 0 && plane == 0) update_txk_array(.., DCT_DCT)`.
        if txb.eob == 0 {
            crate::encode_intra::update_txk_array(
                tx_type_map,
                map_stride,
                blk_row,
                blk_col,
                tx_size,
                0,
            );
        }
        out.push(txb);
        return;
    }
    let sub_txs = crate::var_tx::SUB_TX_SIZE_MAP[tx_size];
    let bsw = crate::var_tx::TX_SIZE_WIDE_UNIT[sub_txs];
    let bsh = crate::var_tx::TX_SIZE_HIGH_UNIT[sub_txs];
    let row_end =
        crate::var_tx::TX_SIZE_HIGH_UNIT[tx_size].min(c.max_blocks_high - blk_row);
    let col_end =
        crate::var_tx::TX_SIZE_WIDE_UNIT[tx_size].min(c.max_blocks_wide - blk_col);
    let mut row = 0usize;
    while row < row_end {
        let mut col = 0usize;
        while col < col_end {
            ibc_encode_block_inter_y(
                c,
                recon,
                ta,
                tl,
                pers_a,
                pers_l,
                out,
                tx_type_map,
                map_stride,
                inter_tx_size,
                blk_row + row,
                blk_col + col,
                sub_txs,
            );
            col += bsw;
        }
        row += bsh;
    }
}


/// `av1_set_entropy_contexts` (blockd.c:29) for ONE txb: stamp the cul across
/// the tx footprint, but ZERO the portion beyond the block's visible extent
/// (`memset(a + above_contexts, 0, txs_wide - above_contexts)`) — the
/// frame-edge tail-zero KB-6 pinned as a write-side probability defect.
#[allow(clippy::too_many_arguments)]
fn ibc_stamp_persistent(
    pers_a: &mut [i8],
    pers_l: &mut [i8],
    blk_row: usize,
    blk_col: usize,
    tx_size: usize,
    cul: i8,
    max_blocks_wide: usize,
    max_blocks_high: usize,
) {
    let txw_u = TXS_W[tx_size] >> 2;
    let txh_u = TXS_H[tx_size] >> 2;
    let above_contexts = txw_u.min(max_blocks_wide.saturating_sub(blk_col));
    let left_contexts = txh_u.min(max_blocks_high.saturating_sub(blk_row));
    for i in 0..txw_u {
        if blk_col + i < pers_a.len() {
            pers_a[blk_col + i] = if i < above_contexts { cul } else { 0 };
        }
    }
    for i in 0..txh_u {
        if blk_row + i < pers_l.len() {
            pers_l[blk_row + i] = if i < left_contexts { cul } else { 0 };
        }
    }
}

/// The intrabc COEFF-arm leaf re-encode (`av1_encode_sb`'s inter path,
/// encodemb.c:636). Called from [`encode_b_intra_dry`]'s intrabc arm AFTER the
/// luma DV prediction is committed to `recon_y`; builds the chroma DV
/// prediction itself, then walks Y (var-tx recursion) / U / V (uniform) in C's
/// plane-outer, mu-64-chunked order.
#[allow(clippy::too_many_arguments)]
fn encode_b_intrabc_coeff(
    env: &SbEncodeEnv,
    state: &mut TileCtxState,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    winner: &mut LeafWinner,
    mi_row: i32,
    mi_col: i32,
    is_chroma_ref: bool,
    output_enabled: bool,
) -> LeafEncodeOut {
    let bsize = winner.bsize;
    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_h = MI_SIZE_HIGH_B[bsize];
    let a0 = mi_col as usize;
    let l0 = (mi_row & 31) as usize;
    let ref_off_y = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
    let use_trellis = crate::encode_intra::is_trellis_used(env.enable_optimize_b, output_enabled);
    // Frame-edge-clipped extents via the validated `max_block_wide/high` port
    // (av1_common_int.h:1567/1581) — hand-rolled mi-difference clips are wrong
    // for chroma and were the shape of the KB-6 edge defects.
    let (bwv, bhv, mb_to_right, mb_to_bottom) = crate::tx_search::max_block_units(
        env.mi_cols,
        env.mi_rows,
        mi_col,
        mi_row,
        mi_w as i32,
        mi_h as i32,
        crate::tx_search::BLK_W_B[bsize],
        crate::tx_search::BLK_H_B[bsize],
        0,
        0,
    );

    // ---- plane 0 (Y): the var-tx quadtree ----
    let mut ta: Vec<i8> = state.above_ectx[0][a0..a0 + mi_w].to_vec();
    let mut tl: Vec<i8> = state.left_ectx[0][l0..l0 + mi_h].to_vec();
    let mut pers_a: Vec<i8> = state.above_ectx[0][a0..a0 + mi_w].to_vec();
    let mut pers_l: Vec<i8> = state.left_ectx[0][l0..l0 + mi_h].to_vec();
    let yc = IbcTxbCtx {
        plane: 0,
        plane_bsize: bsize,
        tx_size_uniform_uv: 0,
        bd: env.bd,
        lossless: env.lossless,
        stride: env.stride,
        ref_off: ref_off_y,
        src: env.src_y,
        src_off: ref_off_y,
        rows: env.rows_y,
        coeff_costs: env.coeff_costs_y,
        rdmult: env.rdmult,
        sharpness: env.sharpness,
        iq_tuning: env.tune.iq_tuning,
        qm_level: env.qm_levels.map(|l| l[0]),
        use_trellis,
        max_blocks_wide: bwv,
        max_blocks_high: bhv,
    };
    let max_tx = crate::tx_search::MAX_TXSIZE_RECT_LOOKUP[bsize];
    let bw_u = crate::var_tx::TX_SIZE_WIDE_UNIT[max_tx];
    let bh_u = crate::var_tx::TX_SIZE_HIGH_UNIT[max_tx];
    // KB-4 tx_type_map semantics: OUTPUT_ENABLED copies (the eob-0 -> DCT_DCT
    // resets land in a transient frame map, the winner keeps the search state);
    // DRY aliases (resets persist).
    let mut frame_map;
    let map: &mut Vec<u8> = if output_enabled {
        frame_map = winner.tx_type_map.clone();
        &mut frame_map
    } else {
        &mut winner.tx_type_map
    };
    let mut y_txbs: Vec<TxbEncode> = Vec::new();
    let mu_w = MI_SIZE_WIDE_B[12].min(mi_w); // BLOCK_64X64
    let mu_h = MI_SIZE_HIGH_B[12].min(mi_h);
    let mut idy = 0usize;
    while idy < mi_h {
        let mut idx = 0usize;
        while idx < mi_w {
            let unit_h = (idy + mu_h).min(mi_h);
            let unit_w = (idx + mu_w).min(mi_w);
            let mut blk_row = idy;
            while blk_row < unit_h {
                let mut blk_col = idx;
                while blk_col < unit_w {
                    ibc_encode_block_inter_y(
                        &yc,
                        recon_y,
                        &mut ta,
                        &mut tl,
                        &mut pers_a,
                        &mut pers_l,
                        &mut y_txbs,
                        map,
                        mi_w,
                        &winner.inter_tx_size,
                        blk_row,
                        blk_col,
                        max_tx,
                    );
                    blk_col += bw_u;
                }
                blk_row += bh_u;
            }
            idx += mu_w;
        }
        idy += mu_h;
    }
    state.above_ectx[0][a0..a0 + mi_w].copy_from_slice(&pers_a);
    state.left_ectx[0][l0..l0 + mi_h].copy_from_slice(&pers_l);

    // ---- planes 1/2 (U, V): DV prediction + the UNIFORM uv tx walk ----
    let mut u_out = None;
    let mut v_out = None;
    if !env.monochrome && is_chroma_ref {
        let ref_off_uv = chroma_plane_offset(
            env.base_uv,
            env.stride,
            mi_row,
            mi_col,
            bsize,
            env.ss_x,
            env.ss_y,
        );
        let plane_bsize = get_plane_block_size(bsize, env.ss_x, env.ss_y);
        let (pmw, pmh) = (MI_SIZE_WIDE_B[plane_bsize], MI_SIZE_HIGH_B[plane_bsize]);
        // The chroma plane block is padded to a 4x4 minimum, so for sub-8x8
        // luma this is LARGER than `bw >> ss_x` (see the RD-side note).
        let (cw, ch) = (
            crate::tx_search::BLK_W_B[plane_bsize],
            crate::tx_search::BLK_H_B[plane_bsize],
        );
        let uv_tx = av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
        let au = (mi_col >> env.ss_x) as usize;
        let lu = ((mi_row & 31) >> env.ss_y) as usize;
        let (cvis_w, cvis_h, _, _) = crate::tx_search::max_block_units(
            env.mi_cols,
            env.mi_rows,
            mi_col,
            mi_row,
            mi_w as i32,
            mi_h as i32,
            crate::tx_search::BLK_W_B[plane_bsize],
            crate::tx_search::BLK_H_B[plane_bsize],
            env.ss_x,
            env.ss_y,
        );
        let txw_u = crate::var_tx::TX_SIZE_WIDE_UNIT[uv_tx];
        let txh_u = crate::var_tx::TX_SIZE_HIGH_UNIT[uv_tx];

        for (pi, (plane, recon)) in [(1usize, &mut *recon_u), (2usize, &mut *recon_v)]
            .into_iter()
            .enumerate()
        {
            // av1_enc_build_inter_predictor (rdopt.c:3601) already built this
            // in the SEARCH; the re-encode rebuilds it from the recon at the DV.
            let mut cpred = vec![0u16; cw * ch];
            crate::intrabc_search::intrabc_predict_chroma(
                recon,
                ref_off_uv,
                env.stride,
                winner.dv_row,
                winner.dv_col,
                env.ss_x,
                env.ss_y,
                &mut cpred,
                cw,
                cw,
                ch,
                i32::from(env.bd),
            );
            for r in 0..ch {
                recon[ref_off_uv + r * env.stride..ref_off_uv + r * env.stride + cw]
                    .copy_from_slice(&cpred[r * cw..r * cw + cw]);
            }

            let mut cta: Vec<i8> = state.above_ectx[plane][au..au + pmw].to_vec();
            let mut ctl: Vec<i8> = state.left_ectx[plane][lu..lu + pmh].to_vec();
            let mut cpers_a: Vec<i8> = cta.clone();
            let mut cpers_l: Vec<i8> = ctl.clone();
            let cc = IbcTxbCtx {
                plane,
                plane_bsize,
                tx_size_uniform_uv: uv_tx,
                bd: env.bd,
                lossless: env.lossless,
                stride: env.stride,
                ref_off: ref_off_uv,
                src: if plane == 1 { env.src_u } else { env.src_v },
                src_off: ref_off_uv,
                rows: if plane == 1 { env.rows_u } else { env.rows_v },
                coeff_costs: env.coeff_costs_uv,
                rdmult: env.rdmult,
                sharpness: env.sharpness,
                iq_tuning: env.tune.iq_tuning,
                qm_level: env.qm_levels.map(|l| l[plane]),
                use_trellis,
                max_blocks_wide: cvis_w,
                max_blocks_high: cvis_h,
            };
            let mut txbs: Vec<TxbEncode> = Vec::new();
            let cmu_w = (MI_SIZE_WIDE_B[12] >> env.ss_x).min(pmw);
            let cmu_h = (MI_SIZE_HIGH_B[12] >> env.ss_y).min(pmh);
            let mut cidy = 0usize;
            while cidy < pmh {
                let mut cidx = 0usize;
                while cidx < pmw {
                    let uh = (cidy + cmu_h).min(pmh);
                    let uw = (cidx + cmu_w).min(pmw);
                    let mut br = cidy;
                    while br < uh {
                        let mut bc = cidx;
                        while bc < uw {
                            if br < cvis_h && bc < cvis_w {
                                // Chroma inherits the co-located LUMA tx type.
                                let tt = crate::var_tx::uv_tx_type_inter(
                                    uv_tx,
                                    env.lossless,
                                    env.reduced_tx_set_used,
                                    map,
                                    mi_w,
                                    br,
                                    bc,
                                    env.ss_x,
                                    env.ss_y,
                                );
                                let (tsc, dsc) = get_txb_ctx(
                                    plane_bsize,
                                    uv_tx,
                                    plane,
                                    &cpers_a[bc..],
                                    &cpers_l[br..],
                                );
                                let mut txb = ibc_encode_txb(
                                    &cc, recon, &mut cta, &mut ctl, br, bc, uv_tx, tt,
                                );
                                txb.txb_skip_ctx = tsc as usize;
                                txb.dc_sign_ctx =
                                    if txb.qcoeff.first().copied().unwrap_or(0) != 0 {
                                        dsc as usize
                                    } else {
                                        0
                                    };
                                ibc_stamp_persistent(
                                    &mut cpers_a,
                                    &mut cpers_l,
                                    br,
                                    bc,
                                    uv_tx,
                                    txb.txb_entropy_ctx as i8,
                                    cvis_w,
                                    cvis_h,
                                );
                                txbs.push(txb);
                            }
                            bc += txw_u;
                        }
                        br += txh_u;
                    }
                    cidx += cmu_w;
                }
                cidy += cmu_h;
            }
            state.above_ectx[plane][au..au + pmw].copy_from_slice(&cpers_a);
            state.left_ectx[plane][lu..lu + pmh].copy_from_slice(&cpers_l);
            let outcome = EncodeIntraPlaneOutcome {
                txbs,
                ta: cta,
                tl: ctl,
            };
            if pi == 0 {
                u_out = Some(outcome);
            } else {
                v_out = Some(outcome);
            }
        }
    }

    // txfm-partition contexts: C stamps them HERE only on a DRY run
    // (partition_search.c:559-562 `if (dry_run) tx_partition_set_contexts`);
    // on the OUTPUT path the pack's `write_tx_size_vartx` does it instead.
    if !output_enabled {
        aom_dsp::entropy::partition::tx_partition_set_contexts(
            bsize,
            &winner.inter_tx_size,
            mb_to_right,
            mb_to_bottom,
            &mut state.above_tctx[a0..a0 + mi_w],
            &mut state.left_tctx[l0..l0 + mi_h],
        );
    }

    LeafEncodeOut {
        mi_row,
        mi_col,
        bsize,
        is_chroma_ref,
        store_y: false,
        y: EncodeIntraPlaneOutcome {
            txbs: y_txbs,
            ta,
            tl,
        },
        u: u_out,
        v: v_out,
    }
}
