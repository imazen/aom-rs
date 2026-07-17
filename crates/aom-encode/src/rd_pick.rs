//! `av1_rd_pick_intra_mode_sb` (av1/encoder/rdopt.c:3636-3698) — the
//! whole-block intra RD composition for a KEY-frame coding block:
//!
//! 1. mbmi bookkeeping (`ref_frame = INTRA_FRAME`, `use_intrabc = 0`,
//!    `mv = 0`, `skip_mode = 0`, `ctx->rd_stats.skip_txfm = 0`) — modelled
//!    implicitly (this port's environments carry no inter state).
//! 2. `intra_yrd = av1_rd_pick_intra_sby_mode(..)` — the luma search
//!    ([`rd_pick_intra_sby_mode_y`], 61-candidate loop + filter-intra tail;
//!    its own tail restored the WINNER's tx_type_map into `xd->tx_type_map`,
//!    intra_mode_search.c:1739).
//! 3. `set_mode_eval_params(cpi, x, DEFAULT_EVAL)` (rdopt.c:3659) — a STATE
//!    NO-OP at speed 0 for everything this port models: with every
//!    `winner_mode_sf` flag 0 and `fast_intra_tx_type_search = 0` /
//!    `use_intra_default_tx_only = 0`, the DEFAULT_EVAL arm
//!    (rdopt_utils.h:556-577) writes the same `use_default_intra_tx_type=0`,
//!    `skip_txfm_level[0]=0`, `predict_dc_level[0]=0`, tx-domain-dist /
//!    coeff-opt-threshold / tx-size-method / tx-type-prune values MODE_EVAL
//!    installed before the loop (both arms pass `enable_winner_mode = 0`);
//!    the `mode_eval_type` change only resets the mb-rd hash record, unused
//!    at speed 0 (`use_mb_rd_hash = 0`, speed_features.c:2499).
//! 4. If `intra_yrd < best_rd` and `num_planes > 1`:
//!    - `if (xd->is_chroma_ref && store_cfl_required_rdo)` restore the
//!      winner map from `ctx->tx_type_map` (rdopt.c:3666-68; redundant with
//!      the sby tail restore — same source, kept for shape).
//!    - `max_uv_tx_size = av1_get_tx_size(AOM_PLANE_U, xd)`
//!      ([`av1_get_tx_size_uv`]).
//!    - `av1_rd_pick_intra_sbuv_mode(..)`, whose PREAMBLE
//!      (intra_mode_search.c:877-901) owns: `init_sbuv_mode`; the
//!      `!xd->is_chroma_ref` early return (`rate/rate_tokenonly/dist = 0`,
//!      `skippable = 1`); `xd->cfl.store_y = store_cfl_required_rdo(cm, x)`
//!      = `!monochrome && is_chroma_ref && is_cfl_allowed(xd)` (the
//!      rdopt_utils.h:653-671 non-chroma-ref CFL_ALLOWED arm is dead code —
//!      the first return already catches it); when store_y, THE LUMA WINNER
//!      RE-ENCODE `av1_encode_intra_block_plane(cpi, x, bsize, AOM_PLANE_Y,
//!      DRY_RUN_NORMAL, cpi->optimize_seg_arr[segment_id])`
//!      ([`encode_intra_block_plane_y`] — loads the CfL context from the
//!      re-reconstructed winner luma, per txb) then `store_y = 0`; then the
//!      14-candidate UV loop ([`rd_pick_intra_sbuv_mode`]).
//! 5. Assembly (rdopt.c:3675-3684): intra is always coded non-skip —
//!    `rate = rate_y + rate_uv + skip_txfm_cost[skip_ctx][0]`,
//!    `dist = dist_y + dist_uv`, `rdcost = RDCOST(rdmult, rate, dist)`,
//!    `skip_txfm = 0`; else (`intra_yrd >= best_rd`) `rate = INT_MAX`
//!    (modelled as `best: None`).
//! 6. `rd_pick_intrabc_mode_sb` (rdopt.c:3688) — ENVELOPE-EXCLUDED hard
//!    no-op: it returns INT64_MAX immediately when `!av1_allow_intrabc(cm)
//!    || !cpi->oxcf.kf_cfg.enable_intrabc` (rdopt.c:3432-3434); the encoder
//!    gate envelope passes `--enable-intrabc=0` and keeps screen-content
//!    tools off (crates/aom-decode/tests/real_bitstream.rs flags).
//! 7. `ctx->mic = *mbmi` + the `ctx->tx_type_map` copy (rdopt.c:3694-3697)
//!    — the returned winner tuple + [`RdPickIntraBest::tx_type_map`]. The
//!    tail copies the map AFTER the luma re-encode's `eob == 0` DCT resets,
//!    so those resets flow into `ctx->tx_type_map` (what packing reads).
//!
//! ## Palette: ENVELOPE-EXCLUDED (decision + evidence, not ported)
//!
//! `av1_rd_pick_palette_intra_sby` runs iff `try_palette =
//! cpi->oxcf.tool_cfg.enable_palette &&
//! av1_allow_palette(allow_screen_content_tools, bsize)`
//! (intra_mode_search.c:1485-1488) — a DOUBLE gate. The encoder gate
//! envelope (crates/aom-decode/tests/real_bitstream.rs:8-10) passes
//! `--enable-palette=0` explicitly, AND generates photographic content
//! precisely so the encoder's screen-content detection keeps
//! `allow_screen_content_tools = 0` (ibid:42-45; screen-content streams are
//! rejected by the decode envelope). Either zero kills the gate, so the
//! palette search — luma AND the chroma `av1_rd_pick_palette_intra_sbuv`,
//! which shares the gate — is legitimately out of the first encoder-gate
//! envelope and is NOT ported (dead code under the envelope, not a
//! simplification of live behaviour). The same evidence covers intrabc
//! (`--enable-intrabc=0`).
//!
//! ## Scope
//!
//! KEY-frame intra blocks at speed-0 all-intra, interior blocks, no
//! segmentation (segment 0), non-lossless. MISSING vs the full C function:
//! palette + intrabc (envelope-excluded above), the inter-frame callers'
//! state (`av1_copy_mbmi_ext_to_mbmi_ext_frame` ref-mv bookkeeping — no-op
//! content for intra), frame-edge clipped walks (as everywhere upstream).

use crate::encode_intra::{
    EncodeIntraPlaneOutcome, EncodeIntraYEnv, TrellisOptType, encode_intra_block_plane_y,
    update_txk_array,
};
use crate::intra_rd::{Block4x4VarInfo, IntraSbyBest, IntraSbySearchCfg, rd_pick_intra_sby_mode_y};
use crate::intra_uv_rd::{
    UvLoopPolicy, UvModeResult, UvModeVisit, UvRdEnv, av1_get_tx_size_uv, rd_pick_intra_sbuv_mode,
};
use crate::mode_costs::IntraModeCosts;
use crate::rd::rdcost;
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B, TxfmYrdEnv};
use aom_intra::cfl::CflCtx;
use aom_txb::CoeffCostSet;

/// The chroma-side arguments of [`rd_pick_intra_mode_sb`] (present when
/// `num_planes > 1`; `None` models monochrome, where the C never enters the
/// uv block and `rate_uv/dist_uv` stay 0).
pub struct RdPickUvArgs<'a, 'b> {
    /// The UV walk environment. `luma_mode`/`luma_use_fi`/`luma_fi_mode` are
    /// OVERWRITTEN with the luma winner before the uv search (the C reads
    /// them from the winner-restored `mbmi`).
    pub env: &'b mut UvRdEnv<'a>,
    pub recon_u: &'b mut [u16],
    pub recon_v: &'b mut [u16],
    /// The CfL context (`xd->cfl`): loaded by the luma winner re-encode when
    /// `store_y`, then consumed by the CfL alpha search.
    pub cfl: &'b mut CflCtx,
    /// `xd->is_chroma_ref`.
    pub is_chroma_ref: bool,
    /// `is_cfl_allowed(xd)` (aom_entropy::partition::is_cfl_allowed).
    pub cfl_allowed: bool,
    /// `mode_costs.intra_uv_mode_cost` — BOTH cfl rows (`[2][13][14]`); the
    /// composition selects `[cfl_allowed]` (intra_mode_search.c:915).
    pub intra_uv_mode_cost: &'b [[[i32; 14]; 13]; 2],
    pub costs: &'b IntraModeCosts,
    pub cfl_costs: &'b crate::mode_costs::CflCosts,
    pub lp: &'b UvLoopPolicy,
    /// The UV palette-search slice (`av1_rd_pick_palette_intra_sbuv` runs iff
    /// `enable_palette && av1_allow_palette(..)`): `None` under
    /// `--enable-palette=0`. `dc_mode_cost`/`y_palette_active` are
    /// placeholders overwritten from the luma winner below (the C reads the
    /// winner-restored mbmi).
    pub palette: Option<crate::palette_search::UvPaletteArgs<'b>>,
}

/// The luma winner re-encode inputs beyond what [`TxfmYrdEnv`] carries.
#[derive(Clone, Copy)]
pub struct ReencodeParams {
    /// `cpi->oxcf.algo_cfg.sharpness` (0 default).
    pub sharpness: i32,
    /// `cpi->optimize_seg_arr[mbmi->segment_id]` (FULL_TRELLIS_OPT at
    /// speed-0 non-lossless).
    pub enable_optimize_b: TrellisOptType,
    /// `oxcf.tune_cfg` knobs ([`crate::TuneKnobs`]): QM-PSNR trellis
    /// distortion metric + the IQ/SSIMULACRA2 trellis rshift.
    pub tune: crate::TuneKnobs,
}

/// The chroma outcome of one [`rd_pick_intra_mode_sb`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RdPickUvOutcome {
    /// `num_planes == 1`: the uv block never runs; `rate_uv = dist_uv = 0`,
    /// `uv_skip_txfm` keeps its 0 init (rdopt.c:3644).
    Monochrome,
    /// `!xd->is_chroma_ref`: the sbuv preamble's early return —
    /// `rate/rate_tokenonly/dist = 0`, `skippable = 1`
    /// (intra_mode_search.c:880-886).
    NotChromaRef,
    /// The uv search ran; the winner + per-candidate visit log.
    Searched(UvModeResult, Vec<UvModeVisit>),
}

impl RdPickUvOutcome {
    /// `rate_uv` in the assembly.
    pub fn rate(&self) -> i32 {
        match self {
            RdPickUvOutcome::Searched(w, _) => w.rate,
            _ => 0,
        }
    }
    /// `dist_uv` in the assembly.
    pub fn dist(&self) -> i64 {
        match self {
            RdPickUvOutcome::Searched(w, _) => w.dist,
            _ => 0,
        }
    }
}

/// The winner state of one [`rd_pick_intra_mode_sb`] (`ctx->mic` +
/// `ctx->tx_type_map` + `rd_cost`).
#[derive(Clone, Debug)]
pub struct RdPickIntraBest {
    /// The luma winner (`best_mbmi` Y fields + rate/dist outputs).
    pub y: IntraSbyBest,
    /// The chroma outcome.
    pub uv: RdPickUvOutcome,
    /// `xd->cfl.store_y` as computed by the sbuv preamble (whether the luma
    /// re-encode ran).
    pub store_y: bool,
    /// The luma winner re-encode outputs (`Some` iff `store_y`): per-txb
    /// eob/qcoeff/dqcoeff/entropy-ctx — the state
    /// `p->qcoeff/dqcoeff/eobs/txb_entropy_ctx` hold afterwards.
    pub reencode: Option<EncodeIntraPlaneOutcome>,
    /// The block-local winner tx_type_map (stride `mi_size_wide[bsize]`)
    /// AFTER the re-encode's eob-0 DCT resets — the `ctx->tx_type_map` copy
    /// at rdopt.c:3697. Origin cells of the winner tx layout are exact; the
    /// dead non-origin cells are canonicalized to DCT_DCT (the C leaves
    /// stale per-candidate history there; nothing on the KEY-frame path
    /// reads them — `av1_get_tx_type`/packing read txb origins only).
    pub tx_type_map: Vec<u8>,
    /// `rd_cost->rate` = `rate_y + rate_uv + skip_txfm_cost[skip_ctx][0]`.
    pub rate: i32,
    /// `rd_cost->dist` = `dist_y + dist_uv`.
    pub dist: i64,
    /// `rd_cost->rdcost`.
    pub rdcost: i64,
}

/// The [`rd_pick_intra_mode_sb`] outcome: `best: None` models the
/// `intra_yrd >= best_rd` path (`rd_cost->rate = INT_MAX`, rdopt.c:3683,
/// and the function returns without touching `ctx`).
pub struct RdPickIntraOutcome {
    pub best: Option<RdPickIntraBest>,
    /// The luma loop's per-(mode, delta) rd table (diff visibility).
    pub intra_modes_rd_cost: [[i64; 9]; 13],
}

/// Build the winner's block-local tx_type_map from the sby winner: stamp
/// each txb origin with its winner tx type via [`update_txk_array`] — the
/// state `xd->tx_type_map` holds after the sby tail's ctx restore
/// (intra_mode_search.c:1739). Non-origin cells default to DCT_DCT (dead
/// state; see [`RdPickIntraBest::tx_type_map`]).
pub fn winner_tx_type_map(
    bsize: usize,
    mi_row: i32,
    mi_col: i32,
    mi_cols: i32,
    mi_rows: i32,
    y: &IntraSbyBest,
) -> Vec<u8> {
    let mbw = MI_SIZE_WIDE_B[bsize];
    let mbh = MI_SIZE_HIGH_B[bsize];
    // The tx search only produced winners for the VISIBLE tx blocks (the
    // frame-edge clip, `max_block_wide/high`); iterate the same range so `k`
    // stays in `y.winners` bounds. Map stride is the FULL block width.
    let bwv = mbw.min((mi_cols - mi_col).max(0) as usize);
    let bhv = mbh.min((mi_rows - mi_row).max(0) as usize);
    let (txwu, txhu) = (
        crate::tx_search::TXS_W[y.tx_size] >> 2,
        crate::tx_search::TXS_H[y.tx_size] >> 2,
    );
    let mut map = vec![0u8; mbw * mbh];
    let mut k = 0usize;
    let mut blk_row = 0usize;
    while blk_row < bhv {
        let mut blk_col = 0usize;
        while blk_col < bwv {
            update_txk_array(
                &mut map,
                mbw,
                blk_row,
                blk_col,
                y.tx_size,
                y.winners[k].tx_type,
            );
            k += 1;
            blk_col += txwu;
        }
        blk_row += txhu;
    }
    debug_assert_eq!(k, y.winners.len());
    map
}

/// `av1_rd_pick_intra_mode_sb` (rdopt.c:3636) — see the module docs for the
/// exact composition and the envelope-excluded pieces. `recon_y` and the
/// chroma recons are mutated exactly as the C leaves them (search state for
/// !store_y luma; winner reconstruction after the re-encode; last-candidate
/// chroma state after the uv loop).
#[allow(clippy::too_many_arguments)]
pub fn rd_pick_intra_mode_sb(
    y_env: &mut TxfmYrdEnv,
    recon_y: &mut [u16],
    sby_cfg: &IntraSbySearchCfg,
    var_cache: &mut [Block4x4VarInfo],
    best_rd: i64,
    coeff_costs_y: &CoeffCostSet,
    re: ReencodeParams,
    mut uv: Option<RdPickUvArgs>,
) -> RdPickIntraOutcome {
    // (2) the luma search.
    let outcome = rd_pick_intra_sby_mode_y(y_env, recon_y, sby_cfg, var_cache, best_rd);
    let rd_table = outcome.intra_modes_rd_cost;
    let Some(y) = outcome.best else {
        // !beat_best_rd => intra_yrd = INT64_MAX >= best_rd => rate INT_MAX.
        return RdPickIntraOutcome {
            best: None,
            intra_modes_rd_cost: rd_table,
        };
    };
    // (3) set_mode_eval_params(DEFAULT_EVAL): state no-op at speed 0 (docs).

    // The sby tail restored the winner's tx_type_map into xd->tx_type_map.
    let mut tx_type_map = winner_tx_type_map(
        y_env.bsize,
        y_env.mi_row,
        y_env.mi_col,
        y_env.mi_cols,
        y_env.mi_rows,
        &y,
    );

    // (4) the uv side.
    let mut store_y = false;
    let mut reencode = None;
    let uv_outcome = match uv.as_mut() {
        None => RdPickUvOutcome::Monochrome,
        Some(args) => {
            if !args.is_chroma_ref {
                // init_sbuv_mode + the early return (rate 0 / skip 1).
                RdPickUvOutcome::NotChromaRef
            } else {
                // store_y = store_cfl_required_rdo(cm, x): !monochrome (we
                // are in the Some arm) && is_chroma_ref && is_cfl_allowed.
                store_y = args.cfl_allowed;
                if store_y {
                    // THE LUMA WINNER RE-ENCODE (DRY_RUN_NORMAL,
                    // optimize_seg_arr[seg]) — loads args.cfl per txb. The
                    // real per-txs_ctx table for the WINNER's tx_size (a
                    // single fixed size — the search already picked it).
                    let coeff_tables_y = coeff_costs_y.tables(y.tx_size);
                    let enc_env = EncodeIntraYEnv {
                        sb_size: y_env.sb_size,
                        bsize: y_env.bsize,
                        mi_row: y_env.mi_row,
                        mi_col: y_env.mi_col,
                        up_available: y_env.up_available,
                        left_available: y_env.left_available,
                        tile_col_end: y_env.tile_col_end,
                        tile_row_end: y_env.tile_row_end,
                        partition: y_env.partition,
                        mi_cols: y_env.mi_cols,
                        mi_rows: y_env.mi_rows,
                        ref_off: y_env.ref_off,
                        ref_stride: y_env.ref_stride,
                        src: y_env.src,
                        src_off: y_env.src_off,
                        src_stride: y_env.src_stride,
                        disable_edge_filter: y_env.disable_edge_filter,
                        filter_type: y_env.filter_type,
                        mode: y.mode,
                        angle_delta: y.angle_delta,
                        use_filter_intra: y.use_filter_intra,
                        filter_intra_mode: y.filter_intra_mode,
                        tx_size: y.tx_size,
                        // mbmi->skip_txfm == 0 on the KF intra RD path
                        // (pick_sb_modes zeroes it; intra never sets it).
                        skip_txfm: false,
                        lossless: y_env.lossless,
                        reduced_tx_set_used: y_env.reduced_tx_set_used,
                        bd: y_env.bd,
                        rows: y_env.rows,
                        rdmult: y_env.rdmult,
                        sharpness: re.sharpness,
                        coeff_costs: &coeff_tables_y,
                        enable_optimize_b: re.enable_optimize_b,
                        // DRY_RUN_NORMAL at this call site
                        // (intra_mode_search.c:897-899).
                        dry_run_output_enabled: false,
                        above_ctx: y_env.above_ctx,
                        left_ctx: y_env.left_ctx,
                        qm_level: y_env.qm_levels.map(|l| l[0]),
                        palette: y.palette_y.as_ref().map(|p| crate::tx_search::PaletteYrd {
                            colors: &p.colors,
                            size: p.size,
                            map: &p.color_map,
                            map_stride: MI_SIZE_WIDE_B[y_env.bsize] * 4,
                        }),
                        tune: re.tune,
                    };
                    reencode = Some(encode_intra_block_plane_y(
                        &enc_env,
                        recon_y,
                        &mut tx_type_map,
                        Some(args.cfl),
                    ));
                    // xd->cfl.store_y = 0 (intra_mode_search.c:900).
                }
                // max_uv_tx_size = av1_get_tx_size(AOM_PLANE_U, xd).
                let max_uv_tx_size =
                    av1_get_tx_size_uv(y_env.bsize, y_env.lossless, args.env.ss_x, args.env.ss_y);
                // The uv loop reads the LUMA WINNER's mode fields from mbmi.
                args.env.luma_mode = y.mode;
                args.env.luma_use_fi = y.use_filter_intra;
                args.env.luma_fi_mode = y.filter_intra_mode;
                args.env.luma_palette_active = y.palette_y.is_some();
                let uv_mode_costs = &args.intra_uv_mode_cost[args.cfl_allowed as usize];
                // The UV palette args read the LUMA WINNER's state (the C
                // reads the winner-restored mbmi): the UV-DC mode cost row +
                // whether Y palette is active (the UV flag's cdf-row ctx).
                if let Some(pal) = args.palette.as_mut() {
                    pal.dc_mode_cost = uv_mode_costs[y.mode][0];
                    pal.y_palette_active = y.palette_y.is_some();
                }
                let (win, visits) = rd_pick_intra_sbuv_mode(
                    args.env,
                    args.recon_u,
                    args.recon_v,
                    args.cfl,
                    max_uv_tx_size,
                    args.cfl_allowed,
                    uv_mode_costs,
                    args.costs,
                    args.cfl_costs,
                    sby_cfg.pol,
                    args.lp,
                    args.palette.as_ref(),
                );
                RdPickUvOutcome::Searched(win, visits)
            }
        }
    };

    // (5) assembly: intra is always coded non-skip.
    let rate = y.rate + uv_outcome.rate() + y_env.skip_costs[y_env.skip_ctx][0];
    let dist = y.dist + uv_outcome.dist();
    let rd = rdcost(y_env.rdmult, rate, dist);
    // (6) rd_pick_intrabc_mode_sb: envelope-excluded no-op (module docs).
    // (7) ctx->mic / ctx->tx_type_map: the returned winner state.
    RdPickIntraOutcome {
        best: Some(RdPickIntraBest {
            y,
            uv: uv_outcome,
            store_y,
            reencode,
            tx_type_map,
            rate,
            dist,
            rdcost: rd,
        }),
        intra_modes_rd_cost: rd_table,
    }
}
