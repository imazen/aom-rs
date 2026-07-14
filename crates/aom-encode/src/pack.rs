//! The pack stage: `OUTPUT_ENABLED` — walk a *picked* partition tree
//! ([`crate::partition_pick::rd_pick_partition_real`]'s winner) a second
//! time, this time driving a real entropy coder, to emit partition symbols,
//! per-leaf KEY-frame mode-info, and per-leaf coefficient bytes. This is the
//! SB/tile-walk composition + coefficient-write half of the encoder gate's
//! "compose the per-SB partition RDO across a whole frame" chunk.
//!
//! Mirrors libaom's actual architecture: `av1_rd_pick_partition` (the
//! DRY_RUN search, decides `pc_tree`) is followed by a **separate**
//! `encode_sb(cpi, td, tile_data, tp, mi_row, mi_col, OUTPUT_ENABLED, bsize,
//! pc_tree, NULL)` call (`av1/encoder/partition_search.c:2073/5133/5458/
//! 5545/6433`) that is the one true real encode. This module's [`pack_sb`]
//! IS that second call; [`pack_tile`] is the per-SB-row/col loop
//! (`av1/encoder/encodeframe.c`'s `encode_sb_row`) that calls search then
//! pack for every superblock in raster order.
//!
//! # Why re-running `encode_b_intra_dry` is correct, not a shortcut
//!
//! [`crate::encode_intra::encode_intra_block_plane_y`]/`_uv`'s only
//! `dry_run_output_enabled`-gated behavior is [`crate::encode_intra::is_trellis_used`]'s
//! `FinalPassTrellisOpt` check (`encodemb.h`); at speed 0, non-lossless, the
//! trellis mode is always `FullTrellisOpt` (`encode_intra.rs` module docs:
//! "the speed-0 final encode is ALWAYS AV1_XFORM_QUANT_FP + av1_optimize_b"),
//! for which `is_trellis_used` returns `true` **regardless** of the flag. So
//! re-running [`crate::encode_sb::encode_b_intra_dry`] (which hardcodes
//! `dry_run_output_enabled: false`) over the SAME winning leaf, from the
//! SAME starting context state, reproduces byte-identical
//! qcoeff/eob/tx_type/dqcoeff to what a true `OUTPUT_ENABLED` call would —
//! this module reuses that validated code path instead of a parallel copy,
//! and only adds the two things `OUTPUT_ENABLED` actually changes: symbol
//! emission (partition / mode-info / tx-size / coefficients) and CDF
//! adaptation. Both search and pack walks visit the SAME winning leaves in
//! the SAME order starting from the SAME zeroed initial tile state (search's
//! own `av1_save_context`/`av1_restore_context` rollback ensures only the
//! winning subtree's contribution survives in its context arrays by the time
//! [`crate::partition_pick::rd_pick_partition_real`] returns) — so the two
//! independently-progressing [`TileCtxState`] instances ([`pack_tile`] keeps
//! one for search, one for pack) stay in lockstep across the whole tile
//! without needing to snapshot/restore between them.
//!
//! # Scope
//!
//! Matches [`crate::partition_pick::rd_pick_partition_real`]'s envelope:
//! `NONE`/`SPLIT`/`HORZ`/`VERT` (4 of 10 partition types), KEY intra,
//! interior SBs, `sb_size <= 64`, no segmentation, no delta-q/delta-lf, no
//! palette, `allow_intrabc = false`, uniform (`TX_MODE_SELECT`) luma tx
//! size. `cdef_bits = 0` models `enable_cdef = 0` (the CDEF-strength literal
//! is zero-width, so `write_literal` is a no-op regardless of the value
//! passed — matches the frame-header `cdef_bits` derivation when the search
//! never finds more than one strength, which is what "off" collapses to).
//! MISSING (mechanical extensions once needed): AB/4-way partition shapes,
//! segmentation-driven per-block qindex, delta-q/delta-lf signaling,
//! palette, intrabc, SB128, multi-tile.

use crate::encode_intra::EncodeIntraPlaneOutcome;
use crate::encode_sb::{LeafWinner, SbEncodeEnv, SbTree, TileCtxState, encode_b_intra_dry};
use crate::intra_uv_rd::{av1_get_tx_size_uv, is_chroma_reference};
use crate::partition::{PartRdStats, split_subsize};
use crate::partition_pick::{ModeGrid, PickFrameCfg, rd_pick_partition_real};
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B};
use aom_entropy::enc::OdEcEnc;
use aom_entropy::partition::{
    KfBlockState, KfFrameContext, MbModeInfoKf, MiNbrKf, bsize_to_max_depth, bsize_to_tx_size_cat,
    get_partition_subsize, get_tx_size_context, is_cfl_allowed, partition_cdf_length,
    partition_plane_context, tx_size_to_depth, update_ext_partition_context, write_mb_modes_kf_fc,
    write_partition, write_selected_tx_size,
};
use aom_intra::cfl::CflCtx;
use aom_txb::{ext_tx_derive, write_coeffs_txb_full};

/// `PARTITION_NONE`/`HORZ`/`VERT`/`SPLIT` C values (matches
/// [`crate::encode_sb`]'s private copies — duplicated here since they're not
/// exported).
const PARTITION_NONE: i32 = 0;
const PARTITION_HORZ: i32 = 1;
const PARTITION_VERT: i32 = 2;
const PARTITION_SPLIT: i32 = 3;
const PARTITION_HORZ_4: i32 = 8;
const PARTITION_VERT_4: i32 = 9;

/// Frame-level pack-stage constants beyond what [`SbEncodeEnv`] already
/// carries for the residual recompute.
#[derive(Clone, Copy, Debug)]
pub struct PackCfg {
    /// `seq_params->enable_filter_intra` — threaded to
    /// [`write_mb_modes_kf_fc`] and MUST equal the value
    /// [`PickFrameCfg::enable_filter_intra`] used during the search (same
    /// seq-header flag).
    pub enable_filter_intra: bool,
    /// `cm->features.tx_mode == TX_MODE_SELECT` (speed-0 default: true —
    /// `tx_search.rs`'s `tx_mode_is_select` doc: "speed-0 all-intra: true").
    pub tx_mode_is_select: bool,
    /// The frame's `av1_write_tx_type` gate (`bitstream.c:815-819`):
    /// `((!seg.enabled && base_qindex > 0) || (seg.enabled &&
    /// qindex[segment_id] > 0)) && !skip_txfm && !seg_skip`. No
    /// segmentation and `skip_txfm` always 0 in this envelope, so this
    /// reduces to `base_qindex > 0` — a frame constant.
    pub signal_gate: bool,
    /// `!cm->features.disable_cdf_update` — whether symbol writes adapt
    /// their CDFs.
    pub allow_update_cdf: bool,
    /// The frame's `current_base_qindex` (no delta-q in this envelope, so
    /// every block's `current_qindex` is this constant).
    pub base_qindex: i32,
}

/// Per-MI-position neighbour tracking for [`write_mb_modes_kf_fc`]'s
/// `above`/`left: Option<MiNbrKf>` — the same shape/reset discipline as
/// [`TileCtxState`]'s other above/left arrays (above indexed by absolute
/// `mi_col`, zeroed once per tile; left indexed by `mi_row & 31`, zeroed at
/// each SB row).
pub struct MiNbrGrid {
    above: Vec<Option<MiNbrKf>>,
    left: [Option<MiNbrKf>; 32],
}

impl MiNbrGrid {
    /// All-absent neighbours (tile start: `av1_zero_above_context`'s
    /// mode-info analogue — no MI has been coded yet).
    pub fn zeroed(mi_cols: usize) -> Self {
        MiNbrGrid {
            above: vec![None; mi_cols],
            left: [None; 32],
        }
    }
    /// `av1_zero_left_context`'s mode-info analogue: called at each SB row
    /// start.
    pub fn zero_left(&mut self) {
        self.left = [None; 32];
    }
    fn stamp(&mut self, mi_row: i32, mi_col: i32, mi_w: usize, mi_h: usize, nbr: MiNbrKf) {
        let a0 = mi_col as usize;
        for x in self.above[a0..a0 + mi_w].iter_mut() {
            *x = Some(nbr);
        }
        let l0 = (mi_row & 31) as usize;
        for x in self.left[l0..l0 + mi_h].iter_mut() {
            *x = Some(nbr);
        }
    }
}

/// A frame-constant [`KfBlockState`] (segmentation/palette/intrabc/delta-q
/// all off, matching [`rd_pick_partition_real`]'s stated envelope) — the
/// per-leaf fields (`mi_row`/`mi_col`/`bsize`/`is_chroma_ref`/`cfl_allowed`/
/// `has_above`/`has_left`) are overwritten by [`pack_leaf`] every call; the
/// mutable carries (`cdef_transmitted`/`current_base_qindex`/`xd_delta_lf*`)
/// self-reset at each SB's upper-left position (`write_cdef`'s own
/// `mi_row/col & sb_mask == 0` check) so one instance can be reused for the
/// whole tile.
pub fn kf_block_state(cfg: &PackCfg, env: &SbEncodeEnv, mib_size: i32) -> KfBlockState {
    KfBlockState {
        segid_preskip: false,
        seg_enabled: false,
        update_map: false,
        seg_pred: 0,
        seg_cdf_num: 0,
        last_active_segid: 0,
        seg_skip_feature: [false; 8],
        mi_row: 0,
        mi_col: 0,
        mib_size,
        sb_size: env.sb_size,
        bsize: env.sb_size,
        coded_lossless: env.lossless,
        allow_intrabc: false,
        cdef_bits: 0,
        dq_present: false,
        dlf_present: false,
        dlf_multi: false,
        num_planes: if env.monochrome { 1 } else { 3 },
        dq_res: 0,
        dlf_res: 0,
        monochrome: env.monochrome,
        is_chroma_ref: true,
        cfl_allowed: false,
        allow_palette: false,
        bit_depth: i32::from(env.bd),
        filter_allowed: false,
        mb_to_top_edge: 0,
        has_above: false,
        has_left: false,
        cdef_transmitted: [false; 4],
        current_base_qindex: cfg.base_qindex,
        xd_delta_lf: [0; 4],
        xd_delta_lf_from_base: 0,
    }
}

/// Pack (`OUTPUT_ENABLED`) one `PARTITION_NONE`/rect leaf: mode-info, the
/// luma tx-size symbol (if signaled), then every coded plane's coefficient
/// bytes — libaom's exact `write_modes_b` order (`write_mbmi_b` -> [palette,
/// excluded] -> tx_size -> `write_tokens_b`, `bitstream.c:1516-1567`).
#[allow(clippy::too_many_arguments)]
pub fn pack_leaf(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    cfg: &PackCfg,
    kf: &mut KfFrameContext,
    kfs: &mut KfBlockState,
    tile: &mut TileCtxState,
    nbr: &mut MiNbrGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    winner: &mut LeafWinner,
    mi_row: i32,
    mi_col: i32,
    partition: usize,
) {
    let bsize = winner.bsize;
    let mi_w = MI_SIZE_WIDE_B[bsize];
    let mi_h = MI_SIZE_HIGH_B[bsize];
    let is_chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, env.ss_x, env.ss_y);
    let cfl_allowed = is_cfl_allowed(bsize, env.lossless, env.ss_x, env.ss_y);
    let has_above = mi_row > env.tile_row_start;
    let has_left = mi_col > env.tile_col_start;

    // ---- 1. write_mbmi_b: mode-info (write_mb_modes_kf_fc). ----
    let above_nbr = nbr.above[mi_col as usize];
    let left_nbr = nbr.left[(mi_row & 31) as usize];
    let info = MbModeInfoKf {
        segment_id: 0,
        skip: i32::from(winner.skip_txfm),
        cdef_strength: 0,
        current_qindex: cfg.base_qindex,
        delta_lf: [0; 4],
        delta_lf_from_base: 0,
        use_intrabc: 0,
        dv_row: 0,
        dv_col: 0,
        y_mode: winner.mode as i32,
        angle_delta_y: winner.angle_delta_y,
        uv_mode: winner.uv_mode as i32,
        cfl_alpha_idx: winner.cfl_alpha_idx,
        cfl_joint_sign: winner.cfl_alpha_signs,
        angle_delta_uv: winner.angle_delta_uv,
        palette_size: [0, 0],
        // No palette in this envelope (module docs: "no palette") -- all-zero
        // colors, matching palette_size == [0, 0] (3 * PALETTE_MAX_SIZE == 24,
        // PALETTE_MAX_SIZE == 8 is aom-entropy-private).
        palette_colors: [0u16; 24],
        use_filter_intra: i32::from(winner.use_filter_intra),
        filter_intra_mode: winner.filter_intra_mode as i32,
    };
    kfs.mi_row = mi_row;
    kfs.mi_col = mi_col;
    kfs.bsize = bsize;
    kfs.is_chroma_ref = is_chroma_ref;
    kfs.cfl_allowed = cfl_allowed;
    kfs.has_above = has_above;
    kfs.has_left = has_left;
    write_mb_modes_kf_fc(
        enc,
        &info,
        kf,
        kfs,
        cfg.enable_filter_intra,
        above_nbr,
        left_nbr,
        // No palette in this envelope -- no palette neighbour context either.
        None,
        None,
    );

    // ---- 2. tx_size symbol (write_selected_tx_size), gated exactly as
    //     write_modes_b's TX_MODE_SELECT branch (bitstream.c:1538-1548); for
    //     intra `is_inter_tx` is always false so that branch collapses to
    //     `tx_mode_is_select && block_signals_txsize(bsize) && !lossless`.
    //     Reads the PRE-block above/left txfm context (before this leaf's
    //     own stamp) -- the stamp itself happens inside encode_b_intra_dry
    //     below (its step 6), unconditionally, matching set_txfm_ctxs being
    //     called with the same args on both branch sides for intra. ----
    if cfg.tx_mode_is_select && bsize > 0 && !env.lossless {
        let a0 = mi_col as usize;
        let l0 = (mi_row & 31) as usize;
        let ctx = get_tx_size_context(
            bsize,
            tile.above_tctx[a0],
            tile.left_tctx[l0],
            has_above,
            has_left,
            None,
            None,
        );
        let cat = bsize_to_tx_size_cat(bsize) as usize;
        let depth = tx_size_to_depth(winner.tx_size, bsize);
        let max_depths = bsize_to_max_depth(bsize);
        write_selected_tx_size(enc, &mut kf.tx_size[cat][ctx], bsize, depth, max_depths);
    }

    // ---- 3. residual/coefficient recompute (reuses the validated dry-run
    //     leaf encode -- see module docs for why this reproduces the true
    //     OUTPUT_ENABLED result in this envelope). ----
    let out = encode_b_intra_dry(
        env, tile, recon_y, recon_u, recon_v, cfl, winner, mi_row, mi_col, partition,
    );

    // ---- 4. write_tokens_b: coefficient bytes, gated on !skip_txfm (always
    //     true in the KEY intra envelope, asserted by encode_b_intra_dry). ----
    if !winner.skip_txfm {
        pack_plane_coeffs(enc, kf, cfg, env, winner, &out.y, winner.tx_size, 0);
        if let Some(u) = &out.u {
            let uv_tx = av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
            pack_plane_coeffs(enc, kf, cfg, env, winner, u, uv_tx, 1);
        }
        if let Some(v) = &out.v {
            let uv_tx = av1_get_tx_size_uv(bsize, env.lossless, env.ss_x, env.ss_y);
            pack_plane_coeffs(enc, kf, cfg, env, winner, v, uv_tx, 2);
        }
    }

    // ---- 5. neighbour-grid stamp for the next block's Y-mode/skip ctx. ----
    let nbr_kf = MiNbrKf {
        y_mode: winner.mode as i32,
        skip_txfm: i32::from(winner.skip_txfm),
    };
    nbr.stamp(mi_row, mi_col, mi_w, mi_h, nbr_kf);
}

/// Pack-stage per-plane coefficient loop: walk `out.txbs` (already in raster
/// order — see [`crate::encode_intra::encode_intra_block_plane_y`]/`_uv`'s
/// doc) and emit each txb's bytes via `write_coeffs_txb_full`, reusing the
/// `txb_skip_ctx`/`dc_sign_ctx` the residual recompute already derived (the
/// SAME pair the trellis used to select its rate tables).
#[allow(clippy::too_many_arguments)]
fn pack_plane_coeffs(
    enc: &mut OdEcEnc,
    kf: &mut KfFrameContext,
    cfg: &PackCfg,
    env: &SbEncodeEnv,
    winner: &LeafWinner,
    out: &EncodeIntraPlaneOutcome,
    tx_size: usize,
    plane: usize,
) {
    let plane_type = usize::from(plane > 0);
    for txb in &out.txbs {
        let mut dummy = [0u16; 8];
        let ext_tx_cdf: &mut [u16] = if plane_type == 0 {
            let d = ext_tx_derive(
                tx_size,
                false, // is_inter
                env.reduced_tx_set_used,
                txb.tx_type,
                winner.use_filter_intra,
                winner.filter_intra_mode,
                winner.mode,
            );
            match d.eset {
                1 => &mut kf.ext_tx_1ddct[d.square as usize][d.intra_dir as usize],
                2 => &mut kf.ext_tx_dtt4[d.square as usize][d.intra_dir as usize],
                _ => &mut dummy[..],
            }
        } else {
            &mut dummy[..]
        };
        write_coeffs_txb_full(
            enc,
            &mut kf.coeff,
            ext_tx_cdf,
            &txb.qcoeff,
            txb.eob as usize,
            tx_size,
            txb.tx_type,
            plane_type,
            txb.txb_skip_ctx,
            txb.dc_sign_ctx,
            cfg.allow_update_cdf,
            false, // is_inter
            env.reduced_tx_set_used,
            winner.use_filter_intra,
            winner.filter_intra_mode,
            winner.mode,
            cfg.signal_gate,
        );
    }
}

/// Pack (`OUTPUT_ENABLED`) walk over a picked [`SbTree`]: write the
/// partition symbol at each node (`write_modes_sb`'s exact recursion,
/// `bitstream.c`, extended to the `HORZ`/`VERT` rect shapes this port's
/// search also produces — the upstream `write_modes_b`/`write_modes_sb` in
/// `aom-entropy` only handle `NONE`/`SPLIT`), then dispatch [`pack_leaf`] at
/// each `PARTITION_NONE`/rect sub-block. Mirrors
/// [`crate::encode_sb::encode_sb_dry`] shape-for-shape (frame-bound gating
/// for `HORZ`'s/`VERT`'s second sub-block included).
#[allow(clippy::too_many_arguments)]
pub fn pack_sb(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    cfg: &PackCfg,
    kf: &mut KfFrameContext,
    kfs: &mut KfBlockState,
    tile: &mut TileCtxState,
    nbr: &mut MiNbrGrid,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    cfl: &mut CflCtx,
    tree: &mut SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) {
    if mi_row >= env.mi_rows || mi_col >= env.mi_cols {
        return;
    }
    let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
    let mi_step = hbs;
    let has_rows = mi_row + mi_step < env.mi_rows;
    let has_cols = mi_col + mi_step < env.mi_cols;

    let (p, subsize) = match tree {
        SbTree::Leaf(_) => (PARTITION_NONE, bsize),
        SbTree::Split(_) => (PARTITION_SPLIT, split_subsize(bsize)),
        SbTree::Horz(_) => (
            PARTITION_HORZ,
            get_partition_subsize(bsize, PARTITION_HORZ) as usize,
        ),
        SbTree::Vert(_) => (
            PARTITION_VERT,
            get_partition_subsize(bsize, PARTITION_VERT) as usize,
        ),
        SbTree::Horz4(_) => (
            PARTITION_HORZ_4,
            get_partition_subsize(bsize, PARTITION_HORZ_4) as usize,
        ),
        SbTree::Vert4(_) => (
            PARTITION_VERT_4,
            get_partition_subsize(bsize, PARTITION_VERT_4) as usize,
        ),
    };

    if bsize >= 3 {
        // BLOCK_8X8: write_partition's own internal size gate makes this
        // redundant for smaller sizes, but partition_plane_context's
        // neighbour read is only meaningful (and only computed by the
        // search) from 8x8 up -- matches rd_pick_partition_real's own
        // `bsize_at_least_8x8` gate.
        let ctx = partition_plane_context(
            &tile.above_pctx,
            &tile.left_pctx,
            mi_row as usize,
            mi_col as usize,
            bsize,
        ) as usize;
        write_partition(
            enc,
            &mut kf.partition[ctx],
            partition_cdf_length(bsize),
            p,
            has_rows,
            has_cols,
            bsize,
        );
    }

    match tree {
        SbTree::Leaf(w) => {
            pack_leaf(
                enc,
                env,
                cfg,
                kf,
                kfs,
                tile,
                nbr,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                w,
                mi_row,
                mi_col,
                PARTITION_NONE as usize,
            );
        }
        SbTree::Split(children) => {
            for (idx, child) in children.iter_mut().enumerate() {
                let y = mi_row + ((idx as i32) >> 1) * hbs;
                let x = mi_col + ((idx as i32) & 1) * hbs;
                pack_sb(
                    enc, env, cfg, kf, kfs, tile, nbr, recon_y, recon_u, recon_v, cfl, child, y, x,
                    subsize,
                );
            }
        }
        SbTree::Horz(subs) => {
            let [s0, s1] = &mut **subs;
            pack_leaf(
                enc,
                env,
                cfg,
                kf,
                kfs,
                tile,
                nbr,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_HORZ as usize,
            );
            if mi_row + hbs < env.mi_rows {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s1,
                    mi_row + hbs,
                    mi_col,
                    PARTITION_HORZ as usize,
                );
            }
        }
        SbTree::Vert(subs) => {
            let [s0, s1] = &mut **subs;
            pack_leaf(
                enc,
                env,
                cfg,
                kf,
                kfs,
                tile,
                nbr,
                recon_y,
                recon_u,
                recon_v,
                cfl,
                s0,
                mi_row,
                mi_col,
                PARTITION_VERT as usize,
            );
            if mi_col + hbs < env.mi_cols {
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s1,
                    mi_row,
                    mi_col + hbs,
                    PARTITION_VERT as usize,
                );
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
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    this_mi_row,
                    mi_col,
                    PARTITION_HORZ_4 as usize,
                );
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
                pack_leaf(
                    enc,
                    env,
                    cfg,
                    kf,
                    kfs,
                    tile,
                    nbr,
                    recon_y,
                    recon_u,
                    recon_v,
                    cfl,
                    s,
                    mi_row,
                    this_mi_col,
                    PARTITION_VERT_4 as usize,
                );
            }
        }
    }

    update_ext_partition_context(
        &mut tile.above_pctx,
        &mut tile.left_pctx,
        mi_row,
        mi_col,
        subsize,
        bsize,
        p,
    );
}

/// Pack a whole tile: search ([`rd_pick_partition_real`]) then pack
/// ([`pack_sb`]) each SB in raster order, threading two independently-
/// progressing [`TileCtxState`]s (search's own save/restore keeps it
/// winners-only by the time it returns, so pack's separate forward-only walk
/// stays in lockstep — see the module docs) and the running
/// [`MiNbrGrid`]/[`KfFrameContext`] — libaom's `write_modes`/`encode_sb_row`
/// tile loop: above context zeroed once per tile, left zeroed at each SB row
/// start. `(mi_row0, mi_col0)` is the tile's first SB position (frame-
/// absolute mi units; `0, 0` for a frame-first single tile). Returns the
/// winning trees (one per SB, row-major) for differential visibility; the
/// entropy-coded bytes accumulate in `enc`.
#[allow(clippy::too_many_arguments)]
pub fn pack_tile(
    enc: &mut OdEcEnc,
    env: &SbEncodeEnv,
    pick_cfg: &PickFrameCfg,
    pack_cfg: &PackCfg,
    kf: &mut KfFrameContext,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    mi_row0: i32,
    mi_col0: i32,
    n_sb_rows: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
) -> Vec<SbTree> {
    let mi_cols = env.mi_cols as usize;
    let mut search_tile = TileCtxState::zeroed(mi_cols);
    let mut pack_tile_ctx = TileCtxState::zeroed(mi_cols);
    let mut grid = ModeGrid::dc(env.mi_rows as usize, mi_cols);
    let mut nbr = MiNbrGrid::zeroed(mi_cols);
    let mut kfs = kf_block_state(pack_cfg, env, sb_mi);
    let mut trees = Vec::new();

    for r in 0..n_sb_rows {
        search_tile.left_ectx = [[0; 32]; 3];
        search_tile.left_pctx = [0; 32];
        search_tile.left_tctx = [aom_entropy::partition::TXFM_CTX_INIT; 32];
        pack_tile_ctx.left_ectx = [[0; 32]; 3];
        pack_tile_ctx.left_pctx = [0; 32];
        pack_tile_ctx.left_tctx = [aom_entropy::partition::TXFM_CTX_INIT; 32];
        nbr.zero_left();
        for c in 0..n_sb_cols {
            let mi_row = mi_row0 + r * sb_mi;
            let mi_col = mi_col0 + c * sb_mi;

            // ALLINTRA SB-root rdmult modifier (setup_block_rdmult,
            // partition_search.c:652/5710-5722): computed ONCE per SB from
            // the whole-SB source variance, then held constant for every
            // node/leaf below it (both the search and the pack walk use the
            // SAME folded env for this SB).
            let sb_rdmult = if pick_cfg.allintra {
                let mi_w = MI_SIZE_WIDE_B[sb_size] as i32;
                let mi_h = MI_SIZE_HIGH_B[sb_size] as i32;
                let ref_off_y =
                    env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
                let mb_to_right_edge = (env.mi_cols - mi_w - mi_col) * 4 * 8;
                let mb_to_bottom_edge = (env.mi_rows - mi_h - mi_row) * 4 * 8;
                let (var_min, var_max) = crate::partition_pick::log_sub_block_var(
                    env.src_y,
                    ref_off_y,
                    env.stride,
                    sb_size,
                    mb_to_right_edge,
                    mb_to_bottom_edge,
                    env.bd,
                );
                let modifier = crate::partition_pick::intra_sb_rdmult_modifier(var_min, var_max);
                crate::partition_pick::fold_intra_sb_rdmult(env.rdmult, modifier)
            } else {
                env.rdmult
            };
            let sb_env = SbEncodeEnv {
                rdmult: sb_rdmult,
                ..*env
            };

            let mut cfl_search = CflCtx::new(env.ss_x as i32, env.ss_y as i32);
            let mut visits = Vec::new();
            // x->source_variance: 0 at the top of a fresh SB in this
            // single-SB-frame-scoped harness (no prior-SB carry-over
            // modelled — see rd_pick_partition_real's own module docs on
            // this in/out param). By the time any AB stage actually reads
            // it (bsize >= 16x16), an earlier leaf search within THIS SAME
            // node's own NONE/SPLIT/RECT stages has always already
            // overwritten it in every case this port's envelope reaches.
            let mut last_source_variance = 0u32;
            let (tree, _stats, found) = rd_pick_partition_real(
                &sb_env,
                pick_cfg,
                &mut search_tile,
                &mut grid,
                recon_y,
                recon_u,
                recon_v,
                &mut cfl_search,
                mi_row,
                mi_col,
                sb_size,
                PartRdStats::invalid(),
                0,
                None,
                &mut visits,
                &mut last_source_variance,
            );
            assert!(
                found,
                "partition search must find a valid tree at ({mi_row}, {mi_col})"
            );
            let mut tree = tree.expect("found implies a winning tree");

            let mut cfl_pack = CflCtx::new(env.ss_x as i32, env.ss_y as i32);
            pack_sb(
                enc,
                &sb_env,
                pack_cfg,
                kf,
                &mut kfs,
                &mut pack_tile_ctx,
                &mut nbr,
                recon_y,
                recon_u,
                recon_v,
                &mut cfl_pack,
                &mut tree,
                mi_row,
                mi_col,
                sb_size,
            );
            trees.push(tree);
        }
    }
    trees
}
