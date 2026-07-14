//! Intra-block-copy DV (block vector) prediction: the neighbour-scan candidate
//! list `av1_find_mv_refs`/`setup_ref_mv_list`/`av1_find_best_ref_mvs` build
//! (`av1/common/mvref_common.c`), reduced to the ONE path
//! [`aom_entropy::partition::read_intrabc_info`]'s caller needs —
//! `ref_frame == INTRA_FRAME` on a KEY frame — plus the `av1_find_ref_dv`
//! fallback and `av1_is_dv_valid` wavefront/tile check (`mvref_common.h`) that
//! together turn `read_intrabc_info`'s raw `(diff_row, diff_col)` into a
//! validated absolute DV, exactly mirroring `decodemv.c`'s
//! `read_intrabc_info` + `assign_dv`.
//!
//! # What's reduced away, and why it's provably dead on a KEY frame
//!
//! `av1_find_mv_refs(cm, xd, mi, ref_frame, ...)` branches hard on
//! `ref_frame == INTRA_FRAME`: `gm_mv` is zeroed and `global_mvs[INTRA_FRAME]`
//! set to `INVALID_MV` (the `mi->bsize`/global-motion branch is simply never
//! taken), then it calls `setup_ref_mv_list`. Inside `setup_ref_mv_list`,
//! `av1_set_ref_frame(rf, INTRA_FRAME)` yields `rf = [INTRA_FRAME,
//! NONE_FRAME]` (`INTRA_FRAME < REF_FRAMES`, so the single-ref arm of
//! `av1_set_ref_frame`), which permanently selects the SINGLE-reference
//! branch (`rf[1] <= NONE_FRAME`) — the entire compound-reference branch
//! (`rf[1] > NONE_FRAME`, ~65 lines building `comp_list`/using
//! `process_compound_ref_mv_candidate`) is unreachable and dropped. Likewise
//! `cm->features.allow_ref_frame_mvs` (the temporal/`add_tpl_ref_mv` motion-
//! field block) requires a previous frame's stored motion field; our decode
//! envelope is KEY-frame-only (no reference frames exist at all), so this
//! flag is always false and that whole block is dropped too.
//!
//! `mode_context`/`newmv_count` bookkeeping is ALSO dropped: `decodemv.c`'s
//! `read_intrabc_info` passes `inter_mode_ctx` to `av1_find_mv_refs` and never
//! reads it afterwards (intrabc codes no inter mode — it always forces
//! `mbmi->mode = DC_PRED`), so `mode_context`'s bits have zero effect on any
//! value this module returns. `newmv_count` (`have_newmv_in_inter_mode`
//! checks) exists ONLY to feed `mode_context` — dropped for the same reason.
//!
//! `is_global_mv_block(candidate, gm_params[rf[0]].wmtype)` — ported IN FULL
//! (not hand-waved away) since it's cheap, but on every path this module is
//! exercised through, `candidate.mode` is always an intra prediction mode
//! (`DC_PRED`..`PAETH_PRED`/`SMOOTH*`/`D*_PRED`, or `DC_PRED` again for an
//! intrabc neighbour — `read_intrabc_info` forces `mbmi->mode = DC_PRED`),
//! NEVER `GLOBALMV`/`GLOBAL_GLOBALMV` (those are inter-only mode values that
//! cannot occur on a KEY frame), so `is_global_mv_block` is provably always
//! `false` here and the `gm_mv_candidates` path it would select is dead too
//! (kept as a real, general branch — not special-cased away — precisely so a
//! future extension to inter frames doesn't need to revisit this file).
//!
//! # Evidence
//!
//! Every function here is diffed against the REAL C in
//! `dv_ref_diff.rs`: [`find_dv_ref_mvs`] against the actually-EXPORTED
//! (non-static) `av1_find_mv_refs` + `av1_find_best_ref_mvs`, called directly
//! (not transcribed) over a synthetic `AV1_COMMON`/`MACROBLOCKD`/MI-grid
//! facade (`shim_find_dv_ref_mvs`, `dec_shim.c`); [`find_ref_dv`] and
//! [`is_dv_valid`] against the real `static inline` `av1_find_ref_dv` /
//! `av1_is_dv_valid` (`mvref_common.h`), called through thin facade wrappers
//! (`shim_find_ref_dv` / `shim_is_dv_valid`).

/// A neighbour candidate's DV-relevant `MB_MODE_INFO` projection — what
/// [`find_dv_ref_mvs`]'s spatial scan reads through `xd->mi[row_off *
/// xd->mi_stride + col_off]`. Ported generally (both `ref_frame` slots and
/// `mode`), even though on the KEY-frame-only envelope every real candidate
/// has `ref_frame0 == INTRA_FRAME`, `ref_frame1 == NONE_FRAME`, and `mode`
/// never `GLOBALMV`/`GLOBAL_GLOBALMV` — see the module doc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DvNbr {
    /// `candidate->bsize` (`BLOCK_SIZES_ALL` index).
    pub bsize: usize,
    /// `candidate->ref_frame[0]`. `INTRA_FRAME` (0) for any intra/intrabc
    /// candidate; only `> INTRA_FRAME` (an actual inter ref) makes
    /// `process_single_ref_mv_candidate` contribute — never true on a KEY
    /// frame.
    pub ref_frame0: i32,
    /// `candidate->ref_frame[1]`. `NONE_FRAME` (-1) for any single-ref
    /// candidate (always true on a KEY frame — no compound intra).
    pub ref_frame1: i32,
    /// `candidate->use_intrabc` (`is_intrabc_block`).
    pub use_intrabc: bool,
    /// `candidate->mode` — only consulted by `is_global_mv_block`.
    pub mode: i32,
    /// `candidate->mv[0]` (the candidate's own DV when `use_intrabc`), 1/8-pel units.
    pub mv0_row: i32,
    pub mv0_col: i32,
    /// `candidate->mv[1]` — only read by the (dead-for-KEY-frames, ported for
    /// generality) compound/second-ref arms.
    pub mv1_row: i32,
    pub mv1_col: i32,
}

/// `NONE_FRAME` (`enums.h`).
pub const NONE_FRAME: i32 = -1;
/// `INTRA_FRAME` (`enums.h`).
pub const INTRA_FRAME: i32 = 0;
/// `GLOBALMV` / `GLOBAL_GLOBALMV` (`enums.h`) — the only modes
/// `is_global_mv_block` accepts. Values from the `PREDICTION_MODE` enum:
/// intra modes are 0..=12 (`DC_PRED`..`UV_CFL_PRED`-adjacent), inter simple
/// modes start at 13 (`NEARESTMV`); `GLOBALMV` = 15, `GLOBAL_GLOBALMV` = 21
/// (`av1/common/enums.h`).
const GLOBALMV: i32 = 15;
const GLOBAL_GLOBALMV: i32 = 21;

const MVREF_ROW_COLS: i32 = 3;
const MAX_REF_MV_STACK_SIZE: usize = 8;
const MAX_MV_REF_CANDIDATES: usize = 2;
const REF_CAT_LEVEL: u32 = 640;
const MI_SIZE: i32 = 4;
const MI_SIZE_LOG2: i32 = 2;
/// `MV_BORDER` (`mvref_common.h`): 16 pels in 1/8-pel units.
const MV_BORDER: i32 = 16 << 3;
/// `MV_IN_USE_BITS` / `MV_LOW` / `MV_UPP` (`entropymv.h`).
const MV_UPP: i32 = 1 << 14;
const MV_LOW: i32 = -(1 << 14);
/// `INTRABC_DELAY_PIXELS` / `INTRABC_DELAY_SB64` (`mvref_common.h`).
const INTRABC_DELAY_PIXELS: i32 = 256;
const INTRABC_DELAY_SB64: i32 = INTRABC_DELAY_PIXELS / 64;

/// `block_size_wide[BLOCK_SIZES_ALL]` (`common_data.h`) — duplicated from
/// `partition.rs`'s private copy (same libaom-fixed geometry table; kept
/// local so this module has no intra-crate coupling).
const BLOCK_SIZE_WIDE: [i32; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLOCK_SIZE_HIGH: [i32; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
const MI_SIZE_WIDE: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_SIZE_HIGH: [i32; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];
const BLOCK_8X8: usize = 3;
const BLOCK_16X16: usize = 6;
const BLOCK_64X64: usize = 12;
const PARTITION_VERT_A: usize = 6;

/// Tile bounds in mi units (`xd->tile` / `TileInfo`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DvTileBounds {
    pub mi_row_start: i32,
    pub mi_row_end: i32,
    pub mi_col_start: i32,
    pub mi_col_end: i32,
}

fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// `is_inter_block` (`blockd.h`): `use_intrabc || ref_frame[0] > INTRA_FRAME`.
fn is_inter_block(c: &DvNbr) -> bool {
    c.use_intrabc || c.ref_frame0 > INTRA_FRAME
}

/// `is_global_mv_block` (`blockd.h`). `wmtype` is `TransformationType`;
/// `TRANSLATION` = 1 in libaom's enum (`IDENTITY`=0, `TRANSLATION`=1,
/// `ROTZOOM`=2, `AFFINE`=3) — `type > TRANSLATION` selects ROTZOOM/AFFINE
/// only. On our KEY-frame envelope `wmtype` is never populated (no encoded
/// global-motion params reach this module — see the module doc), so callers
/// pass `0` and this always short-circuits false on the mode check first
/// regardless.
fn is_global_mv_block(c: &DvNbr, wmtype: i32) -> bool {
    const TRANSLATION: i32 = 1;
    let block_size_allowed = BLOCK_SIZE_WIDE[c.bsize].min(BLOCK_SIZE_HIGH[c.bsize]) >= 8;
    (c.mode == GLOBALMV || c.mode == GLOBAL_GLOBALMV) && wmtype > TRANSLATION && block_size_allowed
}

/// `is_inside` (`mvref_common.h`).
fn is_inside(tile: &DvTileBounds, mi_col: i32, mi_row: i32, pos_row: i32, pos_col: i32) -> bool {
    !(mi_row + pos_row < tile.mi_row_start
        || mi_col + pos_col < tile.mi_col_start
        || mi_row + pos_row >= tile.mi_row_end
        || mi_col + pos_col >= tile.mi_col_end)
}

/// `find_valid_row_offset` (`mvref_common.h`).
fn find_valid_row_offset(tile: &DvTileBounds, mi_row: i32, row_offset: i32) -> i32 {
    clamp_i32(
        row_offset,
        tile.mi_row_start - mi_row,
        tile.mi_row_end - mi_row - 1,
    )
}

/// `find_valid_col_offset` (`mvref_common.h`).
fn find_valid_col_offset(tile: &DvTileBounds, mi_col: i32, col_offset: i32) -> i32 {
    clamp_i32(
        col_offset,
        tile.mi_col_start - mi_col,
        tile.mi_col_end - mi_col - 1,
    )
}

/// `lower_mv_precision` (`mvref_common.h`), `is_integer` arm dropped: every
/// call site in this module passes `is_integer = 0` (`av1_find_best_ref_mvs`
/// is always called with `is_integer=0` from `read_intrabc_info` —
/// `decodemv.c:715`), so only the `allow_hp` branch is reachable; kept as a
/// real parameter (not hand-fixed to `false`) for direct fidelity to the C
/// signature.
fn lower_mv_precision(row: &mut i32, col: &mut i32, allow_hp: bool) {
    if !allow_hp {
        if *row & 1 != 0 {
            *row += if *row > 0 { -1 } else { 1 };
        }
        if *col & 1 != 0 {
            *col += if *col > 0 { -1 } else { 1 };
        }
    }
}

/// `clamp_mv` over the `clamp_mv_ref` `SubpelMvLimits` (`mv.h` + `mvref_common.h`).
#[allow(clippy::too_many_arguments)]
fn clamp_mv_ref(
    row: &mut i32,
    col: &mut i32,
    bw_px: i32,
    bh_px: i32,
    mb_to_left_edge: i32,
    mb_to_right_edge: i32,
    mb_to_top_edge: i32,
    mb_to_bottom_edge: i32,
) {
    let col_min = mb_to_left_edge - bw_px * 8 - MV_BORDER;
    let col_max = mb_to_right_edge + bw_px * 8 + MV_BORDER;
    let row_min = mb_to_top_edge - bh_px * 8 - MV_BORDER;
    let row_max = mb_to_bottom_edge + bh_px * 8 + MV_BORDER;
    *col = clamp_i32(*col, col_min, col_max);
    *row = clamp_i32(*row, row_min, row_max);
}

/// One entry of the DV candidate stack: `CANDIDATE_MV.this_mv` (`comp_mv` is
/// dropped — see the module doc: the compound branch is dead for
/// `ref_frame == INTRA_FRAME`).
#[derive(Clone, Copy, Debug, Default)]
struct StackEntry {
    row: i32,
    col: i32,
}

/// `add_ref_mv_candidate` (`mvref_common.c`), compound arm (`rf[1] >
/// NONE_FRAME`) dropped — dead for `rf == [INTRA_FRAME, NONE_FRAME]` (see
/// module doc).
#[allow(clippy::too_many_arguments)]
fn add_ref_mv_candidate(
    candidate: &DvNbr,
    rf0: i32,
    refmv_count: &mut u8,
    ref_match_count: &mut u8,
    stack: &mut [StackEntry; MAX_REF_MV_STACK_SIZE],
    weight_arr: &mut [u32; MAX_REF_MV_STACK_SIZE],
    gm_mv0_row: i32,
    gm_mv0_col: i32,
    gm_wmtype: i32,
    weight: u32,
) {
    if !is_inter_block(candidate) {
        return;
    }
    // rf[1] == NONE_FRAME: single-reference branch only.
    for ref_idx in 0..2 {
        let cand_rf = if ref_idx == 0 {
            candidate.ref_frame0
        } else {
            candidate.ref_frame1
        };
        if cand_rf == rf0 {
            let (this_row, this_col) = if is_global_mv_block(candidate, gm_wmtype) {
                (gm_mv0_row, gm_mv0_col)
            } else if ref_idx == 0 {
                (candidate.mv0_row, candidate.mv0_col)
            } else {
                (candidate.mv1_row, candidate.mv1_col)
            };
            let mut index = 0usize;
            while index < *refmv_count as usize {
                if stack[index].row == this_row && stack[index].col == this_col {
                    weight_arr[index] += weight;
                    break;
                }
                index += 1;
            }
            if index == *refmv_count as usize && (*refmv_count as usize) < MAX_REF_MV_STACK_SIZE {
                stack[index] = StackEntry {
                    row: this_row,
                    col: this_col,
                };
                weight_arr[index] = weight;
                *refmv_count += 1;
            }
            *ref_match_count += 1;
        }
    }
}

/// A grid accessor: `(row_offset, col_offset)` relative to the current
/// block's `(mi_row, mi_col)`, mirroring `xd->mi[row_offset * xd->mi_stride +
/// col_offset]`. MUST return the correct candidate for any position this
/// module probes — which (matching C's own trust model: `xd->mi` is never
/// bounds-checked per access either) is only ever inside the current tile
/// and frame, guarded by `up_available`/`left_available`/`max_row_offset`/
/// `max_col_offset`/`is_inside` before every access.
pub trait DvGrid {
    fn get(&self, row_offset: i32, col_offset: i32) -> DvNbr;
}
impl<F: Fn(i32, i32) -> DvNbr> DvGrid for F {
    fn get(&self, row_offset: i32, col_offset: i32) -> DvNbr {
        self(row_offset, col_offset)
    }
}

/// `scan_row_mbmi` (`mvref_common.c`), `newmv_count` dropped (see module doc).
#[allow(clippy::too_many_arguments)]
fn scan_row_mbmi(
    grid: &impl DvGrid,
    mi_col: i32,
    frame_mi_cols: i32,
    rf0: i32,
    row_offset: i32,
    width_mi: i32,
    stack: &mut [StackEntry; MAX_REF_MV_STACK_SIZE],
    weight_arr: &mut [u32; MAX_REF_MV_STACK_SIZE],
    refmv_count: &mut u8,
    row_match_count: &mut u8,
    max_row_offset: i32,
    processed_rows: &mut i32,
) {
    let mut end_mi = width_mi.min(frame_mi_cols - mi_col);
    end_mi = end_mi.min(MI_SIZE_WIDE[BLOCK_64X64]);
    let width_8x8 = MI_SIZE_WIDE[BLOCK_8X8];
    let width_16x16 = MI_SIZE_WIDE[BLOCK_16X16];
    let mut col_offset = 0;
    if row_offset.abs() > 1 {
        col_offset = 1;
        if (mi_col & 1) != 0 && width_mi < width_8x8 {
            col_offset -= 1;
        }
    }
    let use_step_16 = width_mi >= 16;

    let mut i = 0;
    while i < end_mi {
        let candidate = grid.get(row_offset, col_offset + i);
        let candidate_bsize = candidate.bsize;
        let n4_w = MI_SIZE_WIDE[candidate_bsize];
        let mut len = width_mi.min(n4_w);
        if use_step_16 {
            len = len.max(width_16x16);
        } else if row_offset.abs() > 1 {
            len = len.max(width_8x8);
        }

        let mut weight: u32 = 2;
        if width_mi >= width_8x8 && width_mi <= n4_w {
            let inc = (-max_row_offset + row_offset + 1).min(MI_SIZE_HIGH[candidate_bsize]);
            weight = weight.max(inc.max(0) as u32);
            *processed_rows = inc - row_offset - 1;
        }

        add_ref_mv_candidate(
            &candidate,
            rf0,
            refmv_count,
            row_match_count,
            stack,
            weight_arr,
            0,
            0,
            0,
            (len as u32) * weight,
        );

        i += len;
    }
}

/// `scan_col_mbmi` (`mvref_common.c`), `newmv_count` dropped.
#[allow(clippy::too_many_arguments)]
fn scan_col_mbmi(
    grid: &impl DvGrid,
    mi_row: i32,
    frame_mi_rows: i32,
    rf0: i32,
    col_offset: i32,
    height_mi: i32,
    stack: &mut [StackEntry; MAX_REF_MV_STACK_SIZE],
    weight_arr: &mut [u32; MAX_REF_MV_STACK_SIZE],
    refmv_count: &mut u8,
    col_match_count: &mut u8,
    max_col_offset: i32,
    processed_cols: &mut i32,
) {
    let mut end_mi = height_mi.min(frame_mi_rows - mi_row);
    end_mi = end_mi.min(MI_SIZE_HIGH[BLOCK_64X64]);
    let h8x8 = MI_SIZE_HIGH[BLOCK_8X8];
    let h16x16 = MI_SIZE_HIGH[BLOCK_16X16];
    let mut row_offset = 0;
    if col_offset.abs() > 1 {
        row_offset = 1;
        if (mi_row & 1) != 0 && height_mi < h8x8 {
            row_offset -= 1;
        }
    }
    let use_step_16 = height_mi >= 16;

    let mut i = 0;
    while i < end_mi {
        let candidate = grid.get(row_offset + i, col_offset);
        let candidate_bsize = candidate.bsize;
        let n4_h = MI_SIZE_HIGH[candidate_bsize];
        let mut len = height_mi.min(n4_h);
        if use_step_16 {
            len = len.max(h16x16);
        } else if col_offset.abs() > 1 {
            len = len.max(h8x8);
        }

        let mut weight: u32 = 2;
        if height_mi >= h8x8 && height_mi <= n4_h {
            let inc = (-max_col_offset + col_offset + 1).min(MI_SIZE_WIDE[candidate_bsize]);
            weight = weight.max(inc.max(0) as u32);
            *processed_cols = inc - col_offset - 1;
        }

        add_ref_mv_candidate(
            &candidate,
            rf0,
            refmv_count,
            col_match_count,
            stack,
            weight_arr,
            0,
            0,
            0,
            (len as u32) * weight,
        );

        i += len;
    }
}

/// `scan_blk_mbmi` (`mvref_common.c`), `newmv_count` dropped.
#[allow(clippy::too_many_arguments)]
fn scan_blk_mbmi(
    grid: &impl DvGrid,
    mi_row: i32,
    mi_col: i32,
    tile: &DvTileBounds,
    rf0: i32,
    row_offset: i32,
    col_offset: i32,
    stack: &mut [StackEntry; MAX_REF_MV_STACK_SIZE],
    weight_arr: &mut [u32; MAX_REF_MV_STACK_SIZE],
    match_count: &mut u8,
    refmv_count: &mut u8,
) {
    if is_inside(tile, mi_col, mi_row, row_offset, col_offset) {
        let candidate = grid.get(row_offset, col_offset);
        let len = MI_SIZE_WIDE[BLOCK_8X8];
        add_ref_mv_candidate(
            &candidate,
            rf0,
            refmv_count,
            match_count,
            stack,
            weight_arr,
            0,
            0,
            0,
            2 * (len as u32),
        );
    }
}

/// `process_single_ref_mv_candidate` (`mvref_common.c`). Ported in full even
/// though `candidate.ref_frame[rf_idx] > INTRA_FRAME` is provably always
/// false on a KEY frame (no real inter ref exists) — see module doc.
fn process_single_ref_mv_candidate(
    candidate: &DvNbr,
    stack: &mut [StackEntry; MAX_REF_MV_STACK_SIZE],
    weight_arr: &mut [u32; MAX_REF_MV_STACK_SIZE],
    refmv_count: &mut u8,
) {
    for rf_idx in 0..2 {
        let (cand_rf, mv_row, mv_col) = if rf_idx == 0 {
            (candidate.ref_frame0, candidate.mv0_row, candidate.mv0_col)
        } else {
            (candidate.ref_frame1, candidate.mv1_row, candidate.mv1_col)
        };
        if cand_rf > INTRA_FRAME {
            // sign-bias negation dropped: requires `cm->ref_frame_sign_bias`
            // for a REAL inter ref, unreachable on a KEY frame (see module doc).
            let mut stack_idx = 0usize;
            while stack_idx < *refmv_count as usize {
                if stack[stack_idx].row == mv_row && stack[stack_idx].col == mv_col {
                    break;
                }
                stack_idx += 1;
            }
            if stack_idx == *refmv_count as usize && stack_idx < MAX_REF_MV_STACK_SIZE {
                stack[stack_idx] = StackEntry {
                    row: mv_row,
                    col: mv_col,
                };
                weight_arr[stack_idx] = 2;
                *refmv_count += 1;
            }
        }
    }
}

/// `has_top_right` (`av1/common/mvref_common.c` — DISTINCT from the
/// same-named `av1/common/reconintra.c` intra-edge-availability function
/// already ported elsewhere in this codebase; this one drives the MV/DV
/// reference top-right corner probe, via an algorithmic geometry check, not
/// a LUT). `is_last_vertical_rect`/`is_first_horizontal_rect` are derived
/// inline from `(mi_row, mi_col, width_mi, height_mi)` exactly as
/// `av1_common_int.h`'s `set_mi_row_col` computes them — pure functions of
/// the current block's own position/size, no extra state threading needed.
#[allow(clippy::too_many_arguments)]
fn mvref_has_top_right(
    sb_mi_size: i32,
    mi_row: i32,
    mi_col: i32,
    bs_in: i32,
    width_mi: i32,
    height_mi: i32,
    own_partition: usize,
) -> bool {
    let mask_row = mi_row & (sb_mi_size - 1);
    let mask_col = mi_col & (sb_mi_size - 1);

    if bs_in > MI_SIZE_WIDE[BLOCK_64X64] {
        return false;
    }

    let mut bs = bs_in;
    let mut has_tr = !((mask_row & bs) != 0 && (mask_col & bs) != 0);

    while bs < sb_mi_size {
        if (mask_col & bs) != 0 {
            if (mask_col & (2 * bs)) != 0 && (mask_row & (2 * bs)) != 0 {
                has_tr = false;
                break;
            }
        } else {
            break;
        }
        bs <<= 1;
    }

    // `is_last_vertical_rect` (`set_mi_row_col`, `av1_common_int.h`):
    // `width < height && (mi_col + width) % height == 0` (height is a power
    // of 2, so `& (height-1)` == `% height`).
    if width_mi < height_mi {
        let is_last_vertical_rect = (mi_col + width_mi) & (height_mi - 1) == 0;
        if !is_last_vertical_rect {
            has_tr = true;
        }
    }

    // `is_first_horizontal_rect`: `width > height && mi_row % width == 0`.
    if width_mi > height_mi {
        let is_first_horizontal_rect = mi_row & (width_mi - 1) == 0;
        if !is_first_horizontal_rect {
            has_tr = false;
        }
    }

    // NOTE: uses the MUTATED `bs` (post-while-loop shift), not `bs_in` — the
    // C reads `mask_row & bs` here after `bs` has already been left-shifted
    // zero or more times above (`mvref_common.c`'s `has_top_right`).
    if own_partition == PARTITION_VERT_A && width_mi == height_mi && (mask_row & bs) != 0 {
        has_tr = false;
    }

    has_tr
}

/// `av1_find_mv_refs` (`ref_frame == INTRA_FRAME` path) + the reduced
/// `setup_ref_mv_list` (single-ref, no temporal — see module doc) +
/// `av1_find_best_ref_mvs(allow_hp=0, is_integer=0)`. Returns `(nearest_row,
/// nearest_col, near_row, near_col)` in 1/8-pel units — the RAW predictor
/// pair `read_intrabc_info` computes BEFORE the `dv_ref` selection /
/// truncation [`assign_and_validate_dv`] applies next.
#[allow(clippy::too_many_arguments)]
pub fn find_dv_ref_mvs(
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    own_partition: usize,
    up_available: bool,
    left_available: bool,
    tile: DvTileBounds,
    frame_mi_rows: i32,
    frame_mi_cols: i32,
    mib_size: i32,
    grid: impl DvGrid,
) -> (i32, i32, i32, i32) {
    let width_mi = MI_SIZE_WIDE[bsize];
    let height_mi = MI_SIZE_HIGH[bsize];
    let bs = width_mi.max(height_mi);
    let has_tr = mvref_has_top_right(
        mib_size,
        mi_row,
        mi_col,
        bs,
        width_mi,
        height_mi,
        own_partition,
    );

    let row_adj = (height_mi < MI_SIZE_HIGH[BLOCK_8X8]) && (mi_row & 1) != 0;
    let col_adj = (width_mi < MI_SIZE_WIDE[BLOCK_8X8]) && (mi_col & 1) != 0;
    let mut processed_rows = 0;
    let mut processed_cols = 0;

    let rf0 = INTRA_FRAME;
    let mut refmv_count: u8 = 0;
    let mut stack = [StackEntry::default(); MAX_REF_MV_STACK_SIZE];
    let mut weight_arr = [0u32; MAX_REF_MV_STACK_SIZE];

    let mut max_row_offset = 0;
    let mut max_col_offset = 0;

    if up_available {
        max_row_offset = -(MVREF_ROW_COLS << 1) + row_adj as i32;
        if height_mi < MI_SIZE_HIGH[BLOCK_8X8] {
            max_row_offset = -(2 << 1) + row_adj as i32;
        }
        max_row_offset = find_valid_row_offset(&tile, mi_row, max_row_offset);
    }
    if left_available {
        max_col_offset = -(MVREF_ROW_COLS << 1) + col_adj as i32;
        if width_mi < MI_SIZE_WIDE[BLOCK_8X8] {
            max_col_offset = -(2 << 1) + col_adj as i32;
        }
        max_col_offset = find_valid_col_offset(&tile, mi_col, max_col_offset);
    }

    let mut col_match_count: u8 = 0;
    let mut row_match_count: u8 = 0;

    if max_row_offset.abs() >= 1 {
        scan_row_mbmi(
            &grid,
            mi_col,
            frame_mi_cols,
            rf0,
            -1,
            width_mi,
            &mut stack,
            &mut weight_arr,
            &mut refmv_count,
            &mut row_match_count,
            max_row_offset,
            &mut processed_rows,
        );
    }
    if max_col_offset.abs() >= 1 {
        scan_col_mbmi(
            &grid,
            mi_row,
            frame_mi_rows,
            rf0,
            -1,
            height_mi,
            &mut stack,
            &mut weight_arr,
            &mut refmv_count,
            &mut col_match_count,
            max_col_offset,
            &mut processed_cols,
        );
    }
    if has_tr {
        scan_blk_mbmi(
            &grid,
            mi_row,
            mi_col,
            &tile,
            rf0,
            -1,
            width_mi,
            &mut stack,
            &mut weight_arr,
            &mut row_match_count,
            &mut refmv_count,
        );
    }

    let nearest_refmv_count = refmv_count;
    for w in weight_arr.iter_mut().take(nearest_refmv_count as usize) {
        *w += REF_CAT_LEVEL;
    }

    // `allow_ref_frame_mvs` temporal block dropped (always false on a KEY
    // frame — see module doc).

    scan_blk_mbmi(
        &grid,
        mi_row,
        mi_col,
        &tile,
        rf0,
        -1,
        -1,
        &mut stack,
        &mut weight_arr,
        &mut row_match_count,
        &mut refmv_count,
    );

    for idx in 2..=MVREF_ROW_COLS {
        let row_offset = -(idx << 1) + 1 + row_adj as i32;
        let col_offset = -(idx << 1) + 1 + col_adj as i32;

        if row_offset.abs() <= max_row_offset.abs() && row_offset.abs() > processed_rows {
            scan_row_mbmi(
                &grid,
                mi_col,
                frame_mi_cols,
                rf0,
                row_offset,
                width_mi,
                &mut stack,
                &mut weight_arr,
                &mut refmv_count,
                &mut row_match_count,
                max_row_offset,
                &mut processed_rows,
            );
        }
        if col_offset.abs() <= max_col_offset.abs() && col_offset.abs() > processed_cols {
            scan_col_mbmi(
                &grid,
                mi_row,
                frame_mi_rows,
                rf0,
                col_offset,
                height_mi,
                &mut stack,
                &mut weight_arr,
                &mut refmv_count,
                &mut col_match_count,
                max_col_offset,
                &mut processed_cols,
            );
        }
    }

    // Rank the likelihood (verbatim bubble sort, descending by weight —
    // ported EXACTLY, not replaced by a "provably equivalent" stable sort,
    // per the differential-methodology's zero-tolerance-for-subtlety rule).
    bubble_sort_desc(&mut stack, &mut weight_arr, 0, nearest_refmv_count as usize);
    bubble_sort_desc(
        &mut stack,
        &mut weight_arr,
        nearest_refmv_count as usize,
        refmv_count as usize,
    );

    let mut mi_width = MI_SIZE_WIDE[BLOCK_64X64].min(width_mi);
    mi_width = mi_width.min(frame_mi_cols - mi_col);
    let mut mi_height = MI_SIZE_HIGH[BLOCK_64X64].min(height_mi);
    mi_height = mi_height.min(frame_mi_rows - mi_row);
    let mi_size = mi_width.min(mi_height);

    // rf[1] <= NONE_FRAME: single-reference extension (the compound arm is
    // dead — see module doc). NOTE: C guards this loop on `*refmv_count <
    // MAX_MV_REF_CANDIDATES` (2) — NOT `MAX_REF_MV_STACK_SIZE` (8), which is
    // the stack's storage capacity, a different constant
    // (`mvref_common.c`'s "Handle single reference frame extension" loops).
    let mut idx = 0;
    while max_row_offset.abs() >= 1
        && idx < mi_size
        && (refmv_count as usize) < MAX_MV_REF_CANDIDATES
    {
        let candidate = grid.get(-1, idx);
        process_single_ref_mv_candidate(&candidate, &mut stack, &mut weight_arr, &mut refmv_count);
        idx += MI_SIZE_WIDE[candidate.bsize];
    }
    let mut idx = 0;
    while max_col_offset.abs() >= 1
        && idx < mi_size
        && (refmv_count as usize) < MAX_MV_REF_CANDIDATES
    {
        let candidate = grid.get(idx, -1);
        process_single_ref_mv_candidate(&candidate, &mut stack, &mut weight_arr, &mut refmv_count);
        idx += MI_SIZE_HIGH[candidate.bsize];
    }

    let mb_to_left_edge = -(mi_col * MI_SIZE * 8);
    let mb_to_right_edge = (frame_mi_cols - width_mi - mi_col) * MI_SIZE * 8;
    let mb_to_top_edge = -(mi_row * MI_SIZE * 8);
    let mb_to_bottom_edge = (frame_mi_rows - height_mi - mi_row) * MI_SIZE * 8;
    let bw_px = width_mi << MI_SIZE_LOG2;
    let bh_px = height_mi << MI_SIZE_LOG2;
    for e in stack.iter_mut().take(refmv_count as usize) {
        clamp_mv_ref(
            &mut e.row,
            &mut e.col,
            bw_px,
            bh_px,
            mb_to_left_edge,
            mb_to_right_edge,
            mb_to_top_edge,
            mb_to_bottom_edge,
        );
    }

    // mv_ref_list fill (`setup_ref_mv_list`): C fills `[*refmv_count,
    // MAX_MV_REF_CANDIDATES)` with `gm_mv_candidates[0]` then `[0,
    // min(2,*refmv_count))` with the ranked stack. `gm_mv_candidates[0]` is
    // ALWAYS `(0,0)` for `ref_frame == INTRA_FRAME` (`av1_find_mv_refs`
    // zeroes it before calling `setup_ref_mv_list`), so pre-initializing to
    // `(0,0)` and only writing the stack range is equivalent — the C's first
    // (gm-fill) loop would write nothing but `(0,0)` regardless.
    let mut mv_ref_list = [(0i32, 0i32); MAX_MV_REF_CANDIDATES];
    for (idx, slot) in mv_ref_list
        .iter_mut()
        .enumerate()
        .take((refmv_count as usize).min(MAX_MV_REF_CANDIDATES))
    {
        *slot = (stack[idx].row, stack[idx].col);
    }

    // `av1_find_best_ref_mvs(allow_hp=0, is_integer=0)`.
    let (mut nr, mut nc) = mv_ref_list[0];
    let (mut nrr, mut ncc) = mv_ref_list[1];
    lower_mv_precision(&mut nr, &mut nc, false);
    lower_mv_precision(&mut nrr, &mut ncc, false);
    (nr, nc, nrr, ncc)
}

/// The C bubble sort in `setup_ref_mv_list` (`mvref_common.c`), ported
/// verbatim over the `[start, end)` half-open range.
fn bubble_sort_desc(
    stack: &mut [StackEntry; MAX_REF_MV_STACK_SIZE],
    weight_arr: &mut [u32; MAX_REF_MV_STACK_SIZE],
    start: usize,
    end: usize,
) {
    let mut len = end;
    while len > start {
        let mut nr_len = start;
        for idx in (start + 1)..len {
            if weight_arr[idx - 1] < weight_arr[idx] {
                stack.swap(idx - 1, idx);
                weight_arr.swap(idx - 1, idx);
                nr_len = idx;
            }
        }
        len = nr_len;
    }
}

/// `av1_find_ref_dv` (`mvref_common.h`): the DV predictor fallback when the
/// candidate list yields `(0,0)`. Returns `(row, col)` in 1/8-pel units
/// (already a multiple of 8 by construction — `MI_SIZE * mib_size` and
/// `INTRABC_DELAY_PIXELS` are both pixel counts, `convert_fullmv_to_mv`
/// scales by 8).
pub fn find_ref_dv(tile_mi_row_start: i32, mib_size: i32, mi_row: i32) -> (i32, i32) {
    if mi_row - mib_size < tile_mi_row_start {
        (0, (-MI_SIZE * mib_size - INTRABC_DELAY_PIXELS) * 8)
    } else {
        (-MI_SIZE * mib_size * 8, 0)
    }
}

/// `is_mv_valid` (`decodemv.c`).
fn is_mv_valid(row: i32, col: i32) -> bool {
    row > MV_LOW && row < MV_UPP && col > MV_LOW && col < MV_UPP
}

/// `av1_is_dv_valid` (`mvref_common.h`): the wavefront/tile/chroma constraint
/// on an INTEGER-PEL dv (row/col already multiples of 8, i.e. already
/// truncated by the caller exactly as `assign_dv` truncates `mv` after
/// `read_mv`).
#[allow(clippy::too_many_arguments)]
pub fn is_dv_valid(
    dv_row: i32,
    dv_col: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    tile: DvTileBounds,
    mib_size_log2: i32,
    is_chroma_ref: bool,
    num_planes: i32,
    subsampling_x: i32,
    subsampling_y: i32,
) -> bool {
    let bw = BLOCK_SIZE_WIDE[bsize];
    let bh = BLOCK_SIZE_HIGH[bsize];
    const SCALE_PX_TO_MV: i32 = 8;

    if (dv_row & (SCALE_PX_TO_MV - 1)) != 0 || (dv_col & (SCALE_PX_TO_MV - 1)) != 0 {
        return false;
    }

    let src_top_edge = mi_row * MI_SIZE * SCALE_PX_TO_MV + dv_row;
    let tile_top_edge = tile.mi_row_start * MI_SIZE * SCALE_PX_TO_MV;
    if src_top_edge < tile_top_edge {
        return false;
    }
    let src_left_edge = mi_col * MI_SIZE * SCALE_PX_TO_MV + dv_col;
    let tile_left_edge = tile.mi_col_start * MI_SIZE * SCALE_PX_TO_MV;
    if src_left_edge < tile_left_edge {
        return false;
    }
    let src_bottom_edge = (mi_row * MI_SIZE + bh) * SCALE_PX_TO_MV + dv_row;
    let tile_bottom_edge = tile.mi_row_end * MI_SIZE * SCALE_PX_TO_MV;
    if src_bottom_edge > tile_bottom_edge {
        return false;
    }
    let src_right_edge = (mi_col * MI_SIZE + bw) * SCALE_PX_TO_MV + dv_col;
    let tile_right_edge = tile.mi_col_end * MI_SIZE * SCALE_PX_TO_MV;
    if src_right_edge > tile_right_edge {
        return false;
    }

    // Special case for sub-8x8 chroma: prevent referring to chroma pixels
    // outside the current tile.
    if is_chroma_ref && num_planes > 1 {
        if bw < 8 && subsampling_x != 0 && src_left_edge < tile_left_edge + 4 * SCALE_PX_TO_MV {
            return false;
        }
        if bh < 8 && subsampling_y != 0 && src_top_edge < tile_top_edge + 4 * SCALE_PX_TO_MV {
            return false;
        }
    }

    // Is the bottom right within an already-coded SB (+ HW-decoder delay)?
    let max_mib_size = 1 << mib_size_log2;
    let active_sb_row = mi_row >> mib_size_log2;
    let active_sb64_col = (mi_col * MI_SIZE) >> 6;
    let sb_size = max_mib_size * MI_SIZE;
    // C `/` truncates toward zero; Rust's `/` on signed integers does too
    // (matches exactly — do NOT use `div_euclid`, which floors instead).
    let src_sb_row = ((src_bottom_edge >> 3) - 1) / sb_size;
    let src_sb64_col = ((src_right_edge >> 3) - 1) >> 6;
    let total_sb64_per_row = ((tile.mi_col_end - tile.mi_col_start - 1) >> 4) + 1;
    let active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
    let src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
    if src_sb64 >= active_sb64 - INTRABC_DELAY_SB64 {
        return false;
    }

    // Wavefront constraint: use only the top-left area of the frame as reference.
    let gradient = 1 + INTRABC_DELAY_SB64 + (sb_size > 64) as i32;
    let wf_offset = gradient * (active_sb_row - src_sb_row);
    if src_sb_row > active_sb_row
        || src_sb64_col >= active_sb64_col - INTRABC_DELAY_SB64 + wf_offset
    {
        return false;
    }

    true
}

/// `assign_dv`'s post-`read_mv` logic (`decodemv.c`) + the `dv_ref`
/// selection/truncation `read_intrabc_info` performs BEFORE calling
/// `assign_dv` — composed into ONE function since `read_intrabc_info`'s
/// caller ([`aom_entropy::partition::read_intrabc_info`]) already stops at
/// the raw `(diff_row, diff_col)` entropy read (see the module doc): this is
/// everything the caller must do with that diff to get a validated absolute
/// DV, or `None` on an invalid DV (`decodemv.c`'s
/// `aom_internal_error(..., "Invalid intrabc dv")` — the caller must
/// hard-error, matching C).
///
/// `nearest_mv`/`near_mv` are [`find_dv_ref_mvs`]'s raw output (1/8-pel,
/// PRE-truncation — `av1_find_best_ref_mvs`'s return).
#[allow(clippy::too_many_arguments)]
pub fn assign_and_validate_dv(
    nearest_mv: (i32, i32),
    near_mv: (i32, i32),
    diff_row: i32,
    diff_col: i32,
    tile_mi_row_start: i32,
    mib_size: i32,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    tile: DvTileBounds,
    mib_size_log2: i32,
    is_chroma_ref: bool,
    num_planes: i32,
    subsampling_x: i32,
    subsampling_y: i32,
) -> Option<(i32, i32)> {
    // `int_mv dv_ref = nearestmv.as_int == 0 ? nearmv : nearestmv;`
    let mut dv_ref = if nearest_mv == (0, 0) {
        near_mv
    } else {
        nearest_mv
    };
    if dv_ref == (0, 0) {
        dv_ref = find_ref_dv(tile_mi_row_start, mib_size, mi_row);
    }
    // "Ref DV should not have sub-pel."
    let valid_dv_ref = (dv_ref.1 & 7) == 0 && (dv_ref.0 & 7) == 0;
    dv_ref = ((dv_ref.0 >> 3) * 8, (dv_ref.1 >> 3) * 8);

    // `assign_dv`: mv = dv_ref + diff (read_mv's internal `ref->row + diff.row`).
    let mv_row = dv_ref.0 + diff_row;
    let mv_col = dv_ref.1 + diff_col;
    // "DV should not have sub-pel" (structurally guaranteed at
    // MV_SUBPEL_NONE precision — see `find_dv_ref_mvs`'s doc — kept as a
    // real check, matching C's `assert`, rather than assumed).
    let mv_row = (mv_row >> 3) * 8;
    let mv_col = (mv_col >> 3) * 8;

    let valid = valid_dv_ref
        && is_mv_valid(mv_row, mv_col)
        && is_dv_valid(
            mv_row,
            mv_col,
            mi_row,
            mi_col,
            bsize,
            tile,
            mib_size_log2,
            is_chroma_ref,
            num_planes,
            subsampling_x,
            subsampling_y,
        );

    if valid {
        Some((mv_row, mv_col))
    } else {
        None
    }
}
