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
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TXS_H, TXS_W};
use aom_entropy::partition::{get_plane_block_size, update_ext_partition_context};
use aom_intra::cfl::CflCtx;
use aom_txb::{CoeffCostSet, txb_entropy_context};

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

impl TileCtxState {
    /// Zeroed contexts for a tile of `mi_cols` (the av1_zero tile init).
    pub fn zeroed(mi_cols: usize) -> Self {
        TileCtxState {
            above_ectx: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_ectx: [[0; 32]; 3],
            above_pctx: vec![0; mi_cols],
            left_pctx: [0; 32],
            // The C tile init memsets the txfm-context arrays to
            // tx_size_wide[TX_SIZES_LARGEST] == 64, NOT 0
            // (av1_zero_above_context / av1_zero_left_context;
            // aom_entropy::partition::TXFM_CTX_INIT).
            above_tctx: vec![aom_entropy::partition::TXFM_CTX_INIT; mi_cols],
            left_tctx: [aom_entropy::partition::TXFM_CTX_INIT; 32],
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
    /// `mbmi->skip_txfm` (0 throughout the KEY intra path).
    pub skip_txfm: bool,
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
    pub rows_y: &'a aom_quant::PlaneQuantRows<'a>,
    pub rows_u: &'a aom_quant::PlaneQuantRows<'a>,
    pub rows_v: &'a aom_quant::PlaneQuantRows<'a>,
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
    /// UV tx-type cost tables — REQUIRED by [`UvRdEnv`] but never read by
    /// the encode arm (chroma codes no tx-type bits); zeroed tables are
    /// fine.
    pub tx_type_costs: &'a aom_txb::TxTypeCosts,
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

/// `encode_b` at DRY_RUN for one KEY intra leaf — see the module docs for
/// the exact C sequence. Mutates the recon planes, the winner's
/// `tx_type_map` (eob-0 resets), the CfL context (per-txb stores when
/// `store_y`), and stamps `state`'s entropy + txfm contexts. The partition
/// context stamp is [`encode_sb`]'s (`update_ext_partition_context` runs at
/// the node, not the leaf).
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
        // DRY_RUN_NORMAL.
        dry_run_output_enabled: false,
        above_ctx: &above_y,
        left_ctx: &left_y,
    };
    let y_out = encode_intra_block_plane_y(
        &y_env,
        recon_y,
        &mut winner.tx_type_map,
        if store_y { Some(cfl) } else { None },
    );

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
        };
        let uv_winner = UvWinner {
            uv_mode: winner.uv_mode,
            angle_delta_uv: winner.angle_delta_uv,
            cfl_alpha_idx: winner.cfl_alpha_idx,
            cfl_alpha_signs: winner.cfl_alpha_signs,
        };
        let prm = UvEncodeParams {
            tx_size: uv_tx,
            skip_txfm: winner.skip_txfm,
            sharpness: env.sharpness,
            enable_optimize_b: env.enable_optimize_b,
            dry_run_output_enabled: false,
            use_chroma_trellis_rd_mult: env.use_chroma_trellis_rd_mult,
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
        let mut k = 0usize;
        let mut blk_row = 0usize;
        while blk_row < mi_h {
            let mut blk_col = 0usize;
            while blk_col < mi_w {
                let tt = crate::encode_intra::get_tx_type_y(
                    env.lossless,
                    winner.tx_size,
                    &winner.tx_type_map,
                    map_stride,
                    blk_row,
                    blk_col,
                );
                let txb = &y_out.txbs[k];
                let cul = txb_entropy_context(&txb.qcoeff, winner.tx_size, tt, txb.eob as usize);
                // The recompute equals the encode walk's stored ctx: eob==0
                // gives 0 under any scan, and eob>0 txbs kept their map type
                // (only eob-0 origins reset to DCT).
                debug_assert_eq!(cul, txb.txb_entropy_ctx, "tokenize cul == encode ctx");
                // av1_set_entropy_contexts (interior: full-footprint memset).
                for x in state.above_ectx[0][a0 + blk_col..a0 + blk_col + txw_u].iter_mut() {
                    *x = cul as i8;
                }
                for x in state.left_ectx[0][l0 + blk_row..l0 + blk_row + txh_u].iter_mut() {
                    *x = cul as i8;
                }
                k += 1;
                blk_col += txw_u;
            }
            blk_row += txh_u;
        }
        debug_assert_eq!(k, y_out.txbs.len());
        // Planes 1/2 (the C loop breaks when !is_chroma_ref).
        if !env.monochrome && is_chroma_ref {
            let plane_bsize = get_plane_block_size(bsize, env.ss_x, env.ss_y);
            let (pmw, pmh) = (MI_SIZE_WIDE_B[plane_bsize], MI_SIZE_HIGH_B[plane_bsize]);
            let (ptxw_u, ptxh_u) = (TXS_W[uv_tx] >> 2, TXS_H[uv_tx] >> 2);
            let au = (mi_col >> env.ss_x) as usize;
            let lu = ((mi_row & 31) >> env.ss_y) as usize;
            let uv_tt = crate::tx_search::uv_intra_tx_type(
                winner.uv_mode,
                env.lossless,
                uv_tx,
                env.reduced_tx_set_used,
            );
            for (plane, out) in [(1usize, u_out.as_ref()), (2usize, v_out.as_ref())] {
                let out = out.expect("chroma-ref leaf has uv outcomes");
                let mut k = 0usize;
                let mut blk_row = 0usize;
                while blk_row < pmh {
                    let mut blk_col = 0usize;
                    while blk_col < pmw {
                        let txb = &out.txbs[k];
                        let cul = txb_entropy_context(&txb.qcoeff, uv_tx, uv_tt, txb.eob as usize);
                        debug_assert_eq!(cul, txb.txb_entropy_ctx, "uv tokenize cul == encode ctx");
                        for x in
                            state.above_ectx[plane][au + blk_col..au + blk_col + ptxw_u].iter_mut()
                        {
                            *x = cul as i8;
                        }
                        for x in
                            state.left_ectx[plane][lu + blk_row..lu + blk_row + ptxh_u].iter_mut()
                        {
                            *x = cul as i8;
                        }
                        k += 1;
                        blk_col += ptxw_u;
                    }
                    blk_row += ptxh_u;
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

/// `encode_sb` at DRY_RUN over a NONE/SPLIT tree — see the module docs.
/// Appends each leaf's outputs to `leaves` in walk order.
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
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ) as usize,
        ),
        SbTree::Vert(_) => (
            PARTITION_VERT,
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT) as usize,
        ),
        SbTree::Horz4(_) => (
            PARTITION_HORZ_4,
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_4) as usize,
        ),
        SbTree::Vert4(_) => (
            PARTITION_VERT_4,
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_4) as usize,
        ),
        SbTree::HorzA(_) => (
            PARTITION_HORZ_A,
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_A) as usize,
        ),
        SbTree::HorzB(_) => (
            PARTITION_HORZ_B,
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_B) as usize,
        ),
        SbTree::VertA(_) => (
            PARTITION_VERT_A,
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_A) as usize,
        ),
        SbTree::VertB(_) => (
            PARTITION_VERT_B,
            aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_B) as usize,
        ),
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
            );
            leaves.push(out);
        }
        SbTree::Split(children) => {
            for (idx, child) in children.iter_mut().enumerate() {
                let y = mi_row + ((idx as i32) >> 1) * hbs;
                let x = mi_col + ((idx as i32) & 1) * hbs;
                encode_sb_dry(
                    env, state, recon_y, recon_u, recon_v, cfl, child, y, x, subsize, leaves,
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
            );
            debug_assert_eq!(s2.bsize, bsize2);
            leaves.push(out);
        }
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
