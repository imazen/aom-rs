//! Loop-filter-level RD search — port of `av1/encoder/picklpf.c`'s
//! `av1_pick_filter_level`, for the envelope this port targets: a single
//! shown KEY frame (`frame_is_intra_only`), ALLINTRA (usage=2) or GOOD
//! (usage=0) one-pass encode, `method == LPF_PICK_FROM_FULL_IMAGE` (the
//! speed-0 default for BOTH usages — `av1/encoder/speed_features.c`'s
//! `init_lpf_sf` sets this; `set_allintra_speed_feature_framesize_independent`
//! only overrides `lpf_pick` at `speed >= 4` (`NON_DUAL`) / `speed >= 6`
//! (`FROM_Q`), never at speed 0; `set_good_speed_features_framesize_independent`
//! likewise only touches it at `speed >= 5`/`speed >= 7` — verified by
//! grepping every `lpf_pick =` assignment site in speed_features.c).
//!
//! # Envelope simplifications (each individually verified against the C, not
//! assumed)
//!
//! - **`last_frame_filter_level` is always `[0,0,0,0]`.** `av1_pick_filter_level`
//!   only populates it `if !frame_is_intra_only(cm)` — always false for our
//!   single-KEY-frame envelope — so every [`search_filter_level`] search
//!   starts at `filt_mid = 0`, hence `filter_step = 4` (`0 < 16`) always.
//! - **`max_filter_level` is always [`MAX_LOOP_FILTER`] (63).**
//!   `get_max_filter_level` only shrinks it `if
//!   is_stat_consumption_stage_twopass(cpi)` — never true for a one-pass
//!   `--cq-level` encode.
//! - **`use_coarse_filter_level_search` is always `0`** (`init_lpf_sf`
//!   default; neither the allintra nor good speed-feature setter touches it
//!   below `speed >= 5`), so `min_filter_step_thesh = 0` — the search always
//!   runs to `filter_step == 0`, never stopping early at `2`.
//! - **`skip_loop_filter_using_filt_error` / `adaptive_luma_loop_filter_skip`
//!   are always `0`** (dead) at speed 0, both usages — grepped every
//!   assignment site in speed_features.c; the only non-zero writes are in
//!   `set_good_speed_features_lc_dec_framesize_{dependent,independent}`
//!   (a "low complexity decode" variant this envelope never reaches) or
//!   `speed >= 5`/`speed >= 6` blocks. The corresponding C branches
//!   (`av1_pick_filter_level`'s zero/best-filter-SSE reset logic) are
//!   therefore omitted here, not reachable in this envelope.
//! - **`loopfilter_control == LOOPFILTER_ALL`** (default; `disable_filter_rt_screen`
//!   is always 0 — no cyclic-refresh screen-content AQ in this envelope), so
//!   the early `[0,0]` return never fires.
//! - **`cm->features.tx_mode != ONLY_4X4` always holds where this result is
//!   used.** The frame header only calls `encode_loopfilter` (hence only
//!   needs this derivation) `if !coded_lossless`
//!   (`write_frame_header_obu`); `ONLY_4X4` only arises together with
//!   lossless / very old profiles, so the bias-halving branch in
//!   [`search_filter_level`] is applied unconditionally.
//! - **`mode_ref_delta_enabled` is `true` with the textbook default deltas**
//!   `ref_deltas = [1,0,0,0,-1,0,-1,-1]` / `mode_deltas = [0,0]`
//!   (`av1_setup_past_independence` → `set_default_lf_deltas`, called for
//!   every KEY frame) — VERIFIED against the real parsed header (both a
//!   flat and a textured case show `mode_ref_delta_enabled: true` with
//!   exactly these deltas). Since every block in this envelope is intra
//!   (`ref0 == INTRA_FRAME == 0`, `mode_lf == 0` — `MODE_LF_LUT` maps every
//!   intra mode to 0), the ONLY `(ref, mode)` cell of the per-level table
//!   ever read is `[0][0]`, which resolves to `clamp(base_level +
//!   ref_deltas[0] * scale, 0, 63)` — a uniform `+1` (or `+2` once
//!   `base_level >= 32`, since `scale = 1 << (base_level >> 5)`) added to
//!   EVERY block's level, not a per-block-varying adjustment. Getting this
//!   wrong (e.g. assuming `mode_ref_delta_enabled == false`, a reasonable
//!   but WRONG first guess) would silently pick a different filter level
//!   than real aomenc.
//! - **`sharpness_level`**: ALLINTRA → `cpi->oxcf.algo_cfg.sharpness` (this
//!   port's callers never set `--sharpness`, so 0 in every case actually
//!   exercised); GOOD/other → 0 unconditionally (only nonzero for
//!   `AOM_TUNE_IQ`/`AOM_TUNE_SSIMULACRA2`, and this port's tune is always
//!   `TuneMetric::Psnr` — `rd.rs`).
//!
//! Trial deblocking reuses the ALREADY bit-exact
//! [`aom_loopfilter::frame::loop_filter_frame`] verbatim (never
//! reimplemented) — a fresh clone of the plane under test is filtered at
//! each candidate level and compared against the source via [`sse_plane`].

use crate::encode_sb::{LeafWinner, SbTree};
use crate::tx_search::{MI_SIZE_HIGH_B, MI_SIZE_WIDE_B};
use aom_loopfilter::frame::{
    LfFrameBuf, LfMi, LfMiGrid, LfParams, MAX_LOOP_FILTER, loop_filter_frame,
};

/// `av1_set_default_ref_deltas`/`av1_set_default_mode_deltas`
/// (`av1/common/entropymode.c`) — the KEY-frame reset values, applied via
/// `av1_setup_past_independence` -> `set_default_lf_deltas` (which ALSO sets
/// `mode_ref_delta_enabled = 1`, verified against the real parsed header —
/// see module docs).
const DEFAULT_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const DEFAULT_MODE_DELTAS: [i8; 2] = [0, 0];

/// `aom_get_{y,u,v}_sse` / `highbd_get_sse` (`aom_dsp/psnr.c`): sum of
/// squared per-pixel differences over `w x h`, u16 samples at every bit
/// depth (bd==8 values fit in 0..255 and are numerically identical whether
/// the C reads them as u8 or u16 — the same insight
/// `aom-loopfilter/src/frame.rs`'s own module docs already rely on). The C
/// tiles this sum in 16x16 blocks via `aom_sse`/`aom_highbd_sse` (RTCD/SIMD
/// dispatch) for performance; summing non-negative i64 terms is associative
/// with no overflow risk at any realistic frame size (even 8192x8192 x
/// 65535^2 ~= 2.9e14, far under `i64::MAX`), so this straight row-major sum
/// reproduces the bit-identical i64 total regardless of tiling order — no
/// need to replicate the tiling itself.
pub fn sse_plane(
    a: &[u16],
    a_stride: usize,
    b: &[u16],
    b_stride: usize,
    w: usize,
    h: usize,
) -> i64 {
    let mut total: i64 = 0;
    for row in 0..h {
        let ar = &a[row * a_stride..row * a_stride + w];
        let br = &b[row * b_stride..row * b_stride + w];
        for (av, bv) in ar.iter().zip(br) {
            let diff = i64::from(*av) - i64::from(*bv);
            total += diff * diff;
        }
    }
    total
}

/// The frame state one [`pick_filter_level`] call searches over: this
/// port's OWN reconstruction (pre-deblock) + the original source, both
/// planes sharing one `stride` (matching [`crate::encode_sb::SbEncodeEnv`]'s
/// own contract), plus the mi grid the filter reads.
#[derive(Clone, Copy)]
pub struct LfSearchFrame<'a> {
    pub recon_y: &'a [u16],
    pub recon_u: &'a [u16],
    pub recon_v: &'a [u16],
    pub src_y: &'a [u16],
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    pub stride: usize,
    /// Luma CROP dims (the coded frame size — SSE is measured over crop
    /// dims, matching `aom_get_y_sse`'s use of `y_crop_width/height`, NOT
    /// the mi-aligned dims).
    pub crop_width: u32,
    pub crop_height: u32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: i32,
    pub monochrome: bool,
    /// The mi grid (luma mi units; `aom_loopfilter::frame` derives chroma
    /// positions from `ss_x`/`ss_y` internally — one grid serves all 3
    /// planes).
    pub mi: &'a [LfMi],
    pub mi_rows: i32,
    pub mi_cols: i32,
}

/// The derived per-frame loop-filter state (`struct loopfilter`'s
/// search-relevant fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopFilterLevels {
    /// `[vert, horz]` Y-plane levels.
    pub filter_level: [i32; 2],
    pub filter_level_u: i32,
    pub filter_level_v: i32,
    pub sharpness: i32,
}

/// One loop-filter trial (`try_filter_frame`, picklpf.c, MINUS the
/// MT/YV12-buffer plumbing): clone only the plane under test (the other two
/// are never dereferenced — see the `plane_start`/`plane_end` gating note
/// below), apply [`loop_filter_frame`] at the given levels, return the SSE
/// vs source. This port always starts each trial from the untouched
/// original reconstruction (no persistent "restore the unfiltered frame"
/// dance is needed since nothing here mutates shared state between trials).
///
/// Passing a same-sized real slice for the OTHER two planes (rather than a
/// tiny dummy) would also be correct but wasteful; a 1-element dummy is
/// safe because [`loop_filter_frame`]'s `plane_start`/`plane_end` gate
/// (`planes_to_lf`) skips the entire per-plane loop body — including the
/// `match plane { .. }` buffer selection — for any plane outside
/// `[plane_start, plane_end)`, so the dummy is provably never indexed.
fn try_filter_plane(
    f: &LfSearchFrame,
    plane: usize,
    filter_level: [i32; 2],
    filter_level_u: i32,
    filter_level_v: i32,
    sharpness: i32,
) -> i64 {
    let p = LfParams {
        filter_level,
        filter_level_u,
        filter_level_v,
        sharpness,
        mode_ref_delta_enabled: true,
        ref_deltas: DEFAULT_REF_DELTAS,
        mode_deltas: DEFAULT_MODE_DELTAS,
        delta_lf_present: false,
        delta_lf_multi: false,
        lossless: [false; 8],
        seg: Default::default(),
    };
    let grid = LfMiGrid {
        mi: f.mi,
        stride: f.mi_cols as usize,
        mi_rows: f.mi_rows,
        mi_cols: f.mi_cols,
    };
    let uv_w = ((f.crop_width + f.ss_x as u32) >> f.ss_x) as usize;
    let uv_h = ((f.crop_height + f.ss_y as u32) >> f.ss_y) as usize;
    // Two separate dummy buffers (never dereferenced -- see the doc comment
    // above) since `LfFrameBuf` needs distinct mutable refs for its 3 plane
    // fields even when only one plane is under test.
    let mut dummy_a = [0u16; 1];
    let mut dummy_b = [0u16; 1];
    match plane {
        0 => {
            let mut y = f.recon_y.to_vec();
            let mut buf = LfFrameBuf {
                y: &mut y,
                y_stride: f.stride,
                u: &mut dummy_a,
                v: &mut dummy_b,
                uv_stride: f.stride,
                crop_width: f.crop_width,
                crop_height: f.crop_height,
                ss_x: f.ss_x,
                ss_y: f.ss_y,
                bd: f.bd,
            };
            loop_filter_frame(&mut buf, &grid, &p, 0, 1);
            sse_plane(
                f.src_y,
                f.stride,
                &y,
                f.stride,
                f.crop_width as usize,
                f.crop_height as usize,
            )
        }
        1 => {
            let mut u = f.recon_u.to_vec();
            let mut buf = LfFrameBuf {
                y: &mut dummy_a,
                y_stride: f.stride,
                u: &mut u,
                v: &mut dummy_b,
                uv_stride: f.stride,
                crop_width: f.crop_width,
                crop_height: f.crop_height,
                ss_x: f.ss_x,
                ss_y: f.ss_y,
                bd: f.bd,
            };
            loop_filter_frame(&mut buf, &grid, &p, 1, 2);
            sse_plane(f.src_u, f.stride, &u, f.stride, uv_w, uv_h)
        }
        _ => {
            let mut v = f.recon_v.to_vec();
            let mut buf = LfFrameBuf {
                y: &mut dummy_a,
                y_stride: f.stride,
                u: &mut dummy_b,
                v: &mut v,
                uv_stride: f.stride,
                crop_width: f.crop_width,
                crop_height: f.crop_height,
                ss_x: f.ss_x,
                ss_y: f.ss_y,
                bd: f.bd,
            };
            loop_filter_frame(&mut buf, &grid, &p, 2, 3);
            sse_plane(f.src_v, f.stride, &v, f.stride, uv_w, uv_h)
        }
    }
}

/// `filter_level`/`filter_level_u`/`filter_level_v` for one trial candidate
/// (`try_filter_frame`'s `filter_level[2]` construction, picklpf.c:71-84).
/// `dir`: 2 = combined Y search (both directions equal the candidate), 0 =
/// vertical-only (horizontal held at `held_horiz`), 1 = horizontal-only
/// (vertical held at `held_vert`); irrelevant for chroma (`plane != 0`).
fn filt_pair(
    plane: usize,
    dir: usize,
    level: i32,
    held_vert: i32,
    held_horiz: i32,
) -> ([i32; 2], i32, i32) {
    if plane == 0 {
        let fl = match dir {
            0 => [level, held_horiz],
            1 => [held_vert, level],
            _ => [level, level], // dir == 2: combined
        };
        (fl, 0, 0)
    } else if plane == 1 {
        ([0, 0], level, 0)
    } else {
        ([0, 0], 0, level)
    }
}

/// `search_filter_level` (picklpf.c) for this envelope: `last_frame_filter_level`
/// is always 0 (KEY / intra-only — module docs), so `filt_mid` always starts
/// at 0, `filter_step` always starts at 4, and the search always runs to
/// `filter_step == 0` (`min_filter_step_thesh == 0`, `use_coarse_filter_level_search
/// == 0` default). `dir` per [`filt_pair`]'s doc; `held_vert`/`held_horiz`
/// are the ALREADY-COMMITTED other-axis levels from a prior stage of
/// [`pick_filter_level`] (only one is read per `dir`).
#[allow(clippy::too_many_arguments)]
fn search_filter_level(
    f: &LfSearchFrame,
    plane: usize,
    dir: usize,
    held_vert: i32,
    held_horiz: i32,
    sharpness: i32,
) -> i32 {
    const MIN_FILTER_LEVEL: i32 = 0;
    let max_filter_level = MAX_LOOP_FILTER; // one-pass envelope: always 63 (module docs)
    let mut filt_mid = 0i32; // last_frame_filter_level always 0 in this envelope
    let mut filter_step = if filt_mid < 16 { 4 } else { filt_mid / 4 };
    let mut filt_direction = 0i32;
    let mut ss_err = [-1i64; (MAX_LOOP_FILTER + 1) as usize];

    let trial = |level: i32, ss_err: &mut [i64; (MAX_LOOP_FILTER + 1) as usize]| -> i64 {
        if ss_err[level as usize] < 0 {
            let (fl, flu, flv) = filt_pair(plane, dir, level, held_vert, held_horiz);
            ss_err[level as usize] = try_filter_plane(f, plane, fl, flu, flv, sharpness);
        }
        ss_err[level as usize]
    };

    let mut best_err = trial(filt_mid, &mut ss_err);
    let mut filt_best = filt_mid;

    while filter_step > 0 {
        let filt_high = (filt_mid + filter_step).min(max_filter_level);
        let filt_low = (filt_mid - filter_step).max(MIN_FILTER_LEVEL);

        // Bias against raising the loop filter in favor of lowering it.
        // (one-pass envelope: no twopass section_intra_rating scaling.)
        let mut bias = (best_err >> (15 - (filt_mid / 8))) * i64::from(filter_step);
        // cm.features.tx_mode != ONLY_4X4 always holds where this result is
        // used (module docs) -- unconditional halving.
        bias >>= 1;

        if filt_direction <= 0 && filt_low != filt_mid {
            let e = trial(filt_low, &mut ss_err);
            if e < best_err + bias {
                if e < best_err {
                    best_err = e;
                }
                filt_best = filt_low;
            }
        }
        if filt_direction >= 0 && filt_high != filt_mid {
            let e = trial(filt_high, &mut ss_err);
            if e < best_err - bias {
                best_err = e;
                filt_best = filt_high;
            }
        }

        if filt_best == filt_mid {
            filter_step /= 2;
            filt_direction = 0;
        } else {
            filt_direction = if filt_best < filt_mid { -1 } else { 1 };
            filt_mid = filt_best;
        }
    }
    filt_best
}

/// `av1_pick_filter_level` (picklpf.c) for this port's envelope — see the
/// module docs for every simplification and its justification. `allintra`
/// selects the ALLINTRA-vs-GOOD sharpness derivation (`sharpness` is this
/// port's `--sharpness`-equivalent input, 0 for every case actually
/// exercised so far).
pub fn pick_filter_level(
    f: &LfSearchFrame,
    allintra: bool,
    sharpness_cfg: i32,
) -> LoopFilterLevels {
    let sharpness = if allintra { sharpness_cfg } else { 0 };

    // Y: combined search first (both directions equal), then refine vertical
    // alone (horizontal held at the combined winner), then horizontal alone
    // (vertical held at the just-refined value) -- exactly
    // av1_pick_filter_level's dir=2 -> dir=0 -> dir=1 sequence.
    let combined = search_filter_level(f, 0, 2, 0, 0, sharpness);
    let mut filter_level = [combined, combined];
    filter_level[0] = search_filter_level(f, 0, 0, 0, filter_level[1], sharpness);
    filter_level[1] = search_filter_level(f, 0, 1, filter_level[0], 0, sharpness);

    let (filter_level_u, filter_level_v) = if f.monochrome {
        (0, 0)
    } else {
        (
            search_filter_level(f, 1, 0, 0, 0, sharpness),
            search_filter_level(f, 2, 0, 0, 0, sharpness),
        )
    };

    LoopFilterLevels {
        filter_level,
        filter_level_u,
        filter_level_v,
        sharpness,
    }
}

// ---- mi-grid construction from this port's OWN picked+packed trees -------

/// Build the [`LfMi`] grid the loop filter reads, from this port's OWN
/// picked+packed [`SbTree`]s (a post-hoc walk over [`crate::pack::pack_tile`]'s
/// return value, deliberately NOT a `pack_sb` modification — mirrors the
/// existing `stamp_grid_from_tree`/`ModeGrid` pattern in
/// `partition_pick.rs`, duplicated here for a different per-cell payload,
/// consistent with that pattern already existing 3x in this crate). All-intra
/// KEY envelope: `segment_id`/`ref0`/`mode_lf`/`is_inter`/`delta_lf*` are
/// frame constants (no segmentation; every block is intra, `ref0 ==
/// INTRA_FRAME == 0`; every intra mode maps to `mode_lf == 0` via
/// `MODE_LF_LUT`; no delta-lf signaling in this envelope).
pub fn build_lf_mi_grid(
    trees: &[SbTree],
    mi_rows: i32,
    mi_cols: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
) -> Vec<LfMi> {
    let mut mi = vec![LfMi::default(); mi_rows as usize * mi_cols as usize];
    let stride = mi_cols as usize;
    for (idx, tree) in trees.iter().enumerate() {
        let r = idx as i32 / n_sb_cols;
        let c = idx as i32 % n_sb_cols;
        stamp_lf_tree(
            &mut mi,
            stride,
            tree,
            r * sb_mi,
            c * sb_mi,
            sb_size,
            mi_rows,
            mi_cols,
        );
    }
    mi
}

fn lf_cell(bsize: usize, w: &LeafWinner) -> LfMi {
    LfMi {
        bsize: bsize as u8,
        tx_size: w.tx_size as u8,
        segment_id: 0,
        ref0: 0,
        mode_lf: 0,
        is_inter: false,
        skip_txfm: w.skip_txfm,
        delta_lf_from_base: 0,
        delta_lf: [0; 4],
    }
}

#[allow(clippy::too_many_arguments)]
fn stamp_lf(
    mi: &mut [LfMi],
    stride: usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    w: &LeafWinner,
    mi_rows: i32,
    mi_cols: i32,
) {
    let rows = (MI_SIZE_HIGH_B[bsize] as i32).min(mi_rows - mi_row) as usize;
    let cols = (MI_SIZE_WIDE_B[bsize] as i32).min(mi_cols - mi_col) as usize;
    let cell = lf_cell(bsize, w);
    for r in 0..rows {
        let base = (mi_row as usize + r) * stride + mi_col as usize;
        mi[base..base + cols].fill(cell);
    }
}

/// PARTITION_HORZ/VERT/HORZ_4/VERT_4/SPLIT constants matching
/// `aom_entropy::partition::get_partition_subsize`'s expected values
/// (duplicated per-file, matching this crate's established convention —
/// see e.g. `pack.rs`'s own copies).
const PARTITION_HORZ: i32 = 1;
const PARTITION_VERT: i32 = 2;
const PARTITION_HORZ_A: i32 = 4;
const PARTITION_HORZ_B: i32 = 5;
const PARTITION_VERT_A: i32 = 6;
const PARTITION_VERT_B: i32 = 7;
const PARTITION_HORZ_4: i32 = 8;
const PARTITION_VERT_4: i32 = 9;

/// Mirrors `partition_pick.rs`'s private `stamp_grid_from_tree` /
/// `pack.rs`'s `pack_sb` recursion exactly (frame-bound gating on every
/// rect/4-way second+ sub-block), stamping [`LfMi`] cells instead of a mode
/// byte.
#[allow(clippy::too_many_arguments)]
fn stamp_lf_tree(
    mi: &mut [LfMi],
    stride: usize,
    tree: &SbTree,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    match tree {
        SbTree::Leaf(w) => stamp_lf(mi, stride, mi_row, mi_col, bsize, w, mi_rows, mi_cols),
        SbTree::Split(kids) => {
            let sub = crate::partition::split_subsize(bsize);
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            for (idx, child) in kids.iter().enumerate() {
                stamp_lf_tree(
                    mi,
                    stride,
                    child,
                    mi_row + ((idx as i32) >> 1) * hbs,
                    mi_col + ((idx as i32) & 1) * hbs,
                    sub,
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Horz(subs) => {
            let sub = aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, sub, &subs[0], mi_rows, mi_cols);
            if mi_row + hbs < mi_rows {
                stamp_lf(
                    mi,
                    stride,
                    mi_row + hbs,
                    mi_col,
                    sub,
                    &subs[1],
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Vert(subs) => {
            let sub = aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT) as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, sub, &subs[0], mi_rows, mi_cols);
            if mi_col + hbs < mi_cols {
                stamp_lf(
                    mi,
                    stride,
                    mi_row,
                    mi_col + hbs,
                    sub,
                    &subs[1],
                    mi_rows,
                    mi_cols,
                );
            }
        }
        SbTree::Horz4(subs) => {
            let sub =
                aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_4) as usize;
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, w) in subs.iter().enumerate() {
                let this_mi_row = mi_row + (i as i32) * quarter_step;
                if i > 0 && this_mi_row >= mi_rows {
                    break;
                }
                stamp_lf(mi, stride, this_mi_row, mi_col, sub, w, mi_rows, mi_cols);
            }
        }
        SbTree::Vert4(subs) => {
            let sub =
                aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_4) as usize;
            let quarter_step = (MI_SIZE_WIDE_B[bsize] / 4) as i32;
            for (i, w) in subs.iter().enumerate() {
                let this_mi_col = mi_col + (i as i32) * quarter_step;
                if i > 0 && this_mi_col >= mi_cols {
                    break;
                }
                stamp_lf(mi, stride, mi_row, this_mi_col, sub, w, mi_rows, mi_cols);
            }
        }
        SbTree::HorzA(subs) => {
            // AB is interior-only (module docs on encode_sb.rs's SbTree::
            // HorzA) -- no frame-bound gating on any of the 3 sub-blocks.
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize = aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_A)
                as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, bsize2, &subs[0], mi_rows, mi_cols);
            stamp_lf(
                mi,
                stride,
                mi_row,
                mi_col + hbs,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col,
                subsize,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
        SbTree::HorzB(subs) => {
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize = aom_entropy::partition::get_partition_subsize(bsize, PARTITION_HORZ_B)
                as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, subsize, &subs[0], mi_rows, mi_cols);
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertA(subs) => {
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize = aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_A)
                as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, bsize2, &subs[0], mi_rows, mi_cols);
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row,
                mi_col + hbs,
                subsize,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
        SbTree::VertB(subs) => {
            let bsize2 = crate::partition::split_subsize(bsize);
            let subsize = aom_entropy::partition::get_partition_subsize(bsize, PARTITION_VERT_B)
                as usize;
            let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
            stamp_lf(mi, stride, mi_row, mi_col, subsize, &subs[0], mi_rows, mi_cols);
            stamp_lf(
                mi,
                stride,
                mi_row,
                mi_col + hbs,
                bsize2,
                &subs[1],
                mi_rows,
                mi_cols,
            );
            stamp_lf(
                mi,
                stride,
                mi_row + hbs,
                mi_col + hbs,
                bsize2,
                &subs[2],
                mi_rows,
                mi_cols,
            );
        }
    }
}
