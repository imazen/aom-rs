//! CDEF strength RD search — port of `av1/encoder/pickcdef.c`
//! (`av1_cdef_search` + its static helpers) for the single-frame KEY
//! envelope, driven only when the caller explicitly enables CDEF
//! (`--enable-cdef=1`; CDEF is OFF by default in allintra — the byte-exact
//! default envelope never reaches this module).
//!
//! # What is ported (against libaom v3.14.1 source, not assumed)
//!
//! - The full per-64x64-filter-block MSE grid (`av1_cdef_mse_calc_block` /
//!   `cdef_mse_calc_frame`): per (plane, fb, strength-index) SSE between the
//!   ORIGINAL source and the CDEF-filtered deblocked reconstruction, with
//!   the search-specific `av1_cdef_filter_fb` shape (`dirinit` caching of
//!   the per-8x8 luma directions across all strength indices AND planes of
//!   one fb; the packed 16-bit output layout for the high-bitdepth path;
//!   the direct 8-bit `aom_sse` paths for bd8).
//! - `get_cdef_filter_strengths` for ALL six search methods
//!   (`CDEF_FULL_SEARCH` + `CDEF_FAST_SEARCH_LVL1..5`, the
//!   `priconv/secconv` tables).
//! - The greedy + refinement joint strength-set selection
//!   (`search_one[_dual]`, `joint_strength_search[_dual]`): mono uses the
//!   single-plane search (refinement gated on `!fast`), color the dual
//!   (refinement unconditional) — exactly the C gates.
//! - The signaling-bits RD loop (`i = 0..=3` bits,
//!   `RDCOST(rdmult, av1_cost_literal(sb_count*i + nb*6*(1|2)),
//!   tot_mse*16)`), per-SB best-index assignment, and the fast-method
//!   strength re-mapping (`STORE_CDEF_FILTER_STRENGTH`).
//!
//! # What is intentionally NOT ported (all verified dead for this
//! envelope's config: one-pass `--enable-cdef=1` (CDEF_ALL), AOM_Q, no rtc)
//!
//! - `CDEF_PICK_FROM_Q` / `av1_pick_cdef_from_qp` (allintra speed >= 7 rt
//!   path) — out of the current cpu-used 0..=6 scope.
//! - Every `apply_adaptive_cdef` arm (`cdef_control == CDEF_ADAPTIVE` only:
//!   the qindex<=32 early-off, chroma-MSE zeroing, strength
//!   reduction/zeroing, the luma pct-improvement disable): `--enable-cdef=1`
//!   maps to `CDEF_ALL` (`av1_cx_iface.c:1272`), so `apply_adaptive_cdef`
//!   is false. `lpf_sf.adaptive_cdef_mode` is likewise 0 outside the
//!   "low-complexity decode" video presets (`init_lpf_sf:2543`).
//! - `rtc_external_ratectrl` / `skip_cdef_sb` (rt-only; 0 here).
//! - The multi-threaded frame walk (oracle build is CONFIG_MULTITHREAD=0).
//! - The dual/quad `aom_mse_16xh_16bit` and multi-unit `aom_sse` merge
//!   optimizations (`is_dual_or_quad_applicable` /
//!   `get_error_calc_width_in_filt_units`): pure summation regroupings of
//!   the same u64 terms — bit-identical totals without them.
//!
//! # Inputs
//!
//! The search reads the DEBLOCKED reconstruction (`cm->cur_frame->buf`
//! after `av1_loop_filter_frame`) and the ORIGINAL source (`cpi->source`),
//! plus the per-mi `skip_txfm` facts (an 8x8 unit is filtered iff ANY of
//! its 2x2 mi is non-skip; an fb with no such unit is skipped entirely and
//! never signalled). The rdmult is the FRAME-level `cpi->rd.RDMULT` —
//! `loopfilter_frame` (encoder.c:2867) re-loads `cpi->td.mb.rdmult =
//! cpi->rd.RDMULT` right before the LF+CDEF stage, so no per-SB rdmult
//! modifier leaks into the CDEF RD.
//!
//! Building blocks reused from `aom-cdef` (both already differentially
//! validated vs the real C): [`cdef_find_dir`] (600k-case diff) and
//! [`cdef_filter_block_16`] (the u16 store is value-identical to the C u8
//! store for bd8 — see that crate's docs).

use aom_dsp::cdef::{CDEF_BSTRIDE, CDEF_VERY_LARGE, cdef_filter_block_16, cdef_find_dir};
use aom_dsp::loopfilter::frame::LfMi;

/// `CDEF_SEC_STRENGTHS` (cdef.h).
pub const CDEF_SEC_STRENGTHS: i32 = 4;
/// `CDEF_PRI_STRENGTHS * CDEF_SEC_STRENGTHS` (pickcdef.h).
pub const TOTAL_STRENGTHS: usize = 64;
/// `CDEF_STRENGTH_BITS` (cdef.h) — bits per signalled strength value.
const CDEF_STRENGTH_BITS: i32 = 6;
/// `CDEF_MAX_STRENGTHS` (av1_common_int.h) — `best_lev*` scratch size.
const CDEF_MAX_STRENGTHS: usize = 16;

const MI_SIZE_64X64: i32 = 16;
const MI_SIZE_128X128: i32 = 32;
/// `CDEF_VBORDER` / `CDEF_HBORDER` (cdef_block.h).
const VB: usize = 2;
const HB: usize = 8;
/// `CDEF_NBLOCKS` = 128/8.
const NB: usize = 16;
/// `CDEF_INBUF_SIZE` = CDEF_BSTRIDE * (128 + 2*CDEF_VBORDER).
const INBUF_SIZE: usize = CDEF_BSTRIDE * (128 + 2 * VB);
/// The interior origin (`in` in the C) inside the inbuf.
const IN_BASE: usize = VB * CDEF_BSTRIDE + HB;
/// `1 << MAX_SB_SIZE_LOG2` — the search tmp buffer stride/extent.
const MAX_SB: usize = 128;

// `BLOCK_*` values used by the >64 filter-block arms (enums.h order).
const BLOCK_64X128: u8 = 13;
const BLOCK_128X64: u8 = 14;
const BLOCK_128X128: u8 = 15;
const BLOCK_64X64: u8 = 12;

/// `block_size_wide[BLOCK_SIZES_ALL]` (common_data.h).
const BLOCK_SIZE_WIDE: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
/// `block_size_high[BLOCK_SIZES_ALL]` (common_data.h).
const BLOCK_SIZE_HIGH: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
const BLOCK_INVALID: u8 = 255;
/// `av1_ss_size_lookup[BLOCK_SIZES_ALL][ss_x][ss_y]` (common_data.c) — the
/// same table `aom-loopfilter`/`lf_search` carry privately (per-file
/// duplication is this workspace's established convention).
#[rustfmt::skip]
const SS_SIZE_LOOKUP: [[[u8; 2]; 2]; 22] = [
    [[0, 0], [0, 0]],                 // 4x4
    [[1, 0], [BLOCK_INVALID, 0]],     // 4x8
    [[2, BLOCK_INVALID], [0, 0]],     // 8x4
    [[3, 2], [1, 0]],                 // 8x8
    [[4, 3], [BLOCK_INVALID, 1]],     // 8x16
    [[5, BLOCK_INVALID], [3, 2]],     // 16x8
    [[6, 5], [4, 3]],                 // 16x16
    [[7, 6], [BLOCK_INVALID, 4]],     // 16x32
    [[8, BLOCK_INVALID], [6, 5]],     // 32x16
    [[9, 8], [7, 6]],                 // 32x32
    [[10, 9], [BLOCK_INVALID, 7]],    // 32x64
    [[11, BLOCK_INVALID], [9, 8]],    // 64x32
    [[12, 11], [10, 9]],              // 64x64
    [[13, 12], [BLOCK_INVALID, 10]],  // 64x128
    [[14, BLOCK_INVALID], [12, 11]],  // 128x64
    [[15, 14], [13, 12]],             // 128x128
    [[16, 1], [BLOCK_INVALID, 1]],    // 4x16
    [[17, BLOCK_INVALID], [2, 2]],    // 16x4
    [[18, 4], [BLOCK_INVALID, 16]],   // 8x32
    [[19, BLOCK_INVALID], [5, 17]],   // 32x8
    [[20, 7], [BLOCK_INVALID, 18]],   // 16x64
    [[21, BLOCK_INVALID], [8, 19]],   // 64x16
];

// ---- pick-method strength tables (pickcdef.h) -----------------------------

/// `nb_cdef_strengths[CDEF_PICK_METHODS]`: total strength indices evaluated
/// per method (FULL, LVL1..LVL5, PICK_FROM_Q).
pub const NB_CDEF_STRENGTHS: [usize; 7] = [64, 32, 20, 10, 4, 2, 64];

const PRICONV_LVL1: [i32; 8] = [0, 1, 2, 3, 5, 7, 10, 13];
const PRICONV_LVL2: [i32; 5] = [0, 2, 4, 8, 14];
const PRICONV_LVL4: [i32; 2] = [0, 11];
const PRICONV_LVL5: [i32; 2] = [0, 5];
const SECCONV_LVL3: [i32; 2] = [0, 2];
const SECCONV_LVL5: [i32; 1] = [0];

/// `get_cdef_filter_strengths` (pickcdef.c:29): decompose a strength index
/// into (pri, sec) for the given pick method. `pick_method` uses the
/// [`crate::speed_features`] constants (`CDEF_FULL_SEARCH` = 0,
/// `CDEF_FAST_SEARCH_LVL1..5` = 1..5).
pub fn get_cdef_filter_strengths(pick_method: i32, strength_idx: i32) -> (i32, i32) {
    let tot_sec_filter = if pick_method == 5 {
        1 // REDUCED_SEC_STRENGTHS_LVL5
    } else if pick_method >= 3 {
        2 // REDUCED_SEC_STRENGTHS_LVL3
    } else {
        CDEF_SEC_STRENGTHS
    };
    let pri_idx = (strength_idx / tot_sec_filter) as usize;
    let sec_idx = strength_idx % tot_sec_filter;
    match pick_method {
        0 | 6 => (pri_idx as i32, sec_idx), // FULL (and FROM_Q's table shape)
        1 => (PRICONV_LVL1[pri_idx], sec_idx),
        2 => (PRICONV_LVL2[pri_idx], sec_idx),
        3 => (PRICONV_LVL2[pri_idx], SECCONV_LVL3[sec_idx as usize]),
        4 => (PRICONV_LVL4[pri_idx], SECCONV_LVL3[sec_idx as usize]),
        5 => (PRICONV_LVL5[pri_idx], SECCONV_LVL5[sec_idx as usize]),
        _ => unreachable!("invalid CDEF pick method {pick_method}"),
    }
}

// ---- joint strength-set selection (pickcdef.c:84-224) ---------------------

/// `search_one`: best strength to ADD given `nb_strengths` already-selected
/// options in `lev[..nb_strengths]`; writes the winner to
/// `lev[nb_strengths]`, returns its total mse.
fn search_one(
    lev: &mut [i32; CDEF_MAX_STRENGTHS],
    nb_strengths: usize,
    mse: &[[u64; TOTAL_STRENGTHS]],
    total_strengths: usize,
) -> u64 {
    let mut tot_mse = [0u64; TOTAL_STRENGTHS];
    for row in mse {
        let mut best_mse = 1u64 << 63;
        for &gi in &lev[..nb_strengths] {
            if row[gi as usize] < best_mse {
                best_mse = row[gi as usize];
            }
        }
        for (j, t) in tot_mse[..total_strengths].iter_mut().enumerate() {
            *t += best_mse.min(row[j]);
        }
    }
    let mut best_tot_mse = 1u64 << 63;
    let mut best_id = 0usize;
    for (j, &t) in tot_mse[..total_strengths].iter().enumerate() {
        if t < best_tot_mse {
            best_tot_mse = t;
            best_id = j;
        }
    }
    lev[nb_strengths] = best_id as i32;
    best_tot_mse
}

/// `search_one_dual`: the luma+chroma pair variant.
fn search_one_dual(
    lev0: &mut [i32; CDEF_MAX_STRENGTHS],
    lev1: &mut [i32; CDEF_MAX_STRENGTHS],
    nb_strengths: usize,
    mse: [&[[u64; TOTAL_STRENGTHS]]; 2],
    total_strengths: usize,
) -> u64 {
    // (TOTAL_STRENGTHS^2 u64 = 32 KiB — matches the C stack array.)
    let mut tot_mse = vec![[0u64; TOTAL_STRENGTHS]; TOTAL_STRENGTHS];
    for (row0, row1) in mse[0].iter().zip(mse[1]) {
        let mut best_mse = 1u64 << 63;
        for gi in 0..nb_strengths {
            let curr = row0[lev0[gi] as usize] + row1[lev1[gi] as usize];
            if curr < best_mse {
                best_mse = curr;
            }
        }
        for (j, tot_row) in tot_mse[..total_strengths].iter_mut().enumerate() {
            for (k, tot) in tot_row[..total_strengths].iter_mut().enumerate() {
                let curr = row0[j] + row1[k];
                *tot += best_mse.min(curr);
            }
        }
    }
    let mut best_tot_mse = 1u64 << 63;
    let (mut best_id0, mut best_id1) = (0usize, 0usize);
    for (j, tot_row) in tot_mse[..total_strengths].iter().enumerate() {
        for (k, &tot) in tot_row[..total_strengths].iter().enumerate() {
            if tot < best_tot_mse {
                best_tot_mse = tot;
                best_id0 = j;
                best_id1 = k;
            }
        }
    }
    lev0[nb_strengths] = best_id0 as i32;
    lev1[nb_strengths] = best_id1 as i32;
    best_tot_mse
}

/// `joint_strength_search` (mono / single plane-pair): greedy add-one, then
/// (full search only) 4*nb rounds of drop-oldest + re-search refinement.
fn joint_strength_search(
    best_lev: &mut [i32; CDEF_MAX_STRENGTHS],
    nb_strengths: usize,
    mse: &[[u64; TOTAL_STRENGTHS]],
    total_strengths: usize,
    fast: bool,
) -> u64 {
    let mut best_tot_mse = 1u64 << 63;
    for i in 0..nb_strengths {
        best_tot_mse = search_one(best_lev, i, mse, total_strengths);
    }
    if !fast {
        for _ in 0..4 * nb_strengths {
            for j in 0..nb_strengths - 1 {
                best_lev[j] = best_lev[j + 1];
            }
            best_tot_mse = search_one(best_lev, nb_strengths - 1, mse, total_strengths);
        }
    }
    best_tot_mse
}

/// `joint_strength_search_dual`: refinement runs UNCONDITIONALLY (no `fast`
/// gate — pickcdef.c:214, verified different from the mono variant).
fn joint_strength_search_dual(
    best_lev0: &mut [i32; CDEF_MAX_STRENGTHS],
    best_lev1: &mut [i32; CDEF_MAX_STRENGTHS],
    nb_strengths: usize,
    mse: [&[[u64; TOTAL_STRENGTHS]]; 2],
    total_strengths: usize,
) -> u64 {
    let mut best_tot_mse = 1u64 << 63;
    for i in 0..nb_strengths {
        best_tot_mse = search_one_dual(best_lev0, best_lev1, i, mse, total_strengths);
    }
    for _ in 0..4 * nb_strengths {
        for j in 0..nb_strengths - 1 {
            best_lev0[j] = best_lev0[j + 1];
            best_lev1[j] = best_lev1[j + 1];
        }
        best_tot_mse =
            search_one_dual(best_lev0, best_lev1, nb_strengths - 1, mse, total_strengths);
    }
    best_tot_mse
}

// ---- frame view ------------------------------------------------------------

/// The frame state one CDEF search reads: the DEBLOCKED reconstruction +
/// the ORIGINAL source (both `u16` planes sharing one `stride`, the same
/// contract as [`crate::lf_search::LfSearchFrame`]) plus the per-mi facts.
#[derive(Clone, Copy)]
pub struct CdefSearchFrame<'a> {
    /// Post-loop-filter reconstruction (`cm->cur_frame->buf` after
    /// `av1_loop_filter_frame`) — the CDEF filter input.
    pub recon_y: &'a [u16],
    pub recon_u: &'a [u16],
    pub recon_v: &'a [u16],
    /// Original source (`cpi->source`) — the MSE reference. Must be
    /// edge-replicated to the mi-aligned extent (the search measures over
    /// mi dims, exactly like C's border-extended source).
    pub src_y: &'a [u16],
    pub src_u: &'a [u16],
    pub src_v: &'a [u16],
    pub stride: usize,
    /// Per-mi facts (`skip_txfm` + `bsize`), `mi_rows x mi_cols`, row
    /// stride `mi_cols` — [`crate::lf_search::build_lf_mi_grid`]'s output.
    pub mi: &'a [LfMi],
    pub mi_rows: i32,
    pub mi_cols: i32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub monochrome: bool,
    pub bd: u8,
    pub base_qindex: i32,
    /// Frame-level `cpi->rd.RDMULT` (see module docs).
    pub rdmult: i32,
}

impl<'a> CdefSearchFrame<'a> {
    fn recon(&self, pli: usize) -> &'a [u16] {
        match pli {
            0 => self.recon_y,
            1 => self.recon_u,
            _ => self.recon_v,
        }
    }
    fn src(&self, pli: usize) -> &'a [u16] {
        match pli {
            0 => self.src_y,
            1 => self.src_u,
            _ => self.src_v,
        }
    }
    fn num_planes(&self) -> usize {
        if self.monochrome { 1 } else { 3 }
    }
}

/// `av1_cdef_search`'s output — `cm->cdef_info` plus the per-64x64-unit
/// strength indices the encode stamps on the mi grid
/// (`mbmi->cdef_strength = best_gi`).
#[derive(Clone, Debug)]
pub struct CdefSearchResult {
    /// `cdef_info.cdef_bits` (0..=3).
    pub cdef_bits: i32,
    /// `cdef_info.nb_cdef_strengths` = `1 << cdef_bits`.
    pub nb_cdef_strengths: usize,
    /// `cdef_info.cdef_strengths` (luma), `pri * 4 + sec` packed.
    pub cdef_strengths: [i32; 8],
    /// `cdef_info.cdef_uv_strengths` (chroma).
    pub cdef_uv_strengths: [i32; 8],
    /// `cdef_info.cdef_damping` = `3 + (base_qindex >> 6)`.
    pub cdef_damping: i32,
    /// Per-64x64-unit strength INDEX (`nvfb x nhfb` raster). Units the
    /// search skipped (all-skip, or the shadowed odd unit of a >64 block)
    /// stay 0 — matching C, where their `mbmi->cdef_strength` is never
    /// stamped and `write_cdef` never emits a literal for them (every
    /// block in such a unit has `skip_txfm == 1`).
    pub unit_strength: Vec<i32>,
    pub nvfb: i32,
    pub nhfb: i32,
}

// ---- skip logic (pickcdef.h:175-215, cdef.c:29-73) -------------------------

fn mi_at(f: &CdefSearchFrame, mi_row: i32, mi_col: i32) -> LfMi {
    f.mi[(mi_row * f.mi_cols + mi_col) as usize]
}

/// `sb_all_skip`: every mi of the 64x64 unit has `skip_txfm` set.
fn sb_all_skip(f: &CdefSearchFrame, mi_row: i32, mi_col: i32) -> bool {
    let maxr = (f.mi_rows - mi_row).min(MI_SIZE_64X64);
    let maxc = (f.mi_cols - mi_col).min(MI_SIZE_64X64);
    for r in 0..maxr {
        for c in 0..maxc {
            if !mi_at(f, mi_row + r, mi_col + c).skip_txfm {
                return false;
            }
        }
    }
    true
}

/// `cdef_sb_skip`: all-skip units, plus the odd-row/col units shadowed by a
/// >64 block (filtered at the covering block's own call).
fn cdef_sb_skip(f: &CdefSearchFrame, fbr: i32, fbc: i32) -> bool {
    let m = mi_at(f, MI_SIZE_64X64 * fbr, MI_SIZE_64X64 * fbc);
    if sb_all_skip(f, fbr * MI_SIZE_64X64, fbc * MI_SIZE_64X64) {
        return true;
    }
    if ((fbc & 1) != 0 && (m.bsize == BLOCK_128X128 || m.bsize == BLOCK_128X64))
        || ((fbr & 1) != 0 && (m.bsize == BLOCK_128X128 || m.bsize == BLOCK_64X128))
    {
        return true;
    }
    false
}

/// `is_8x8_block_skip`: ALL 2x2 mi of the 8x8 unit are skip.
fn is_8x8_block_skip(f: &CdefSearchFrame, mi_row: i32, mi_col: i32) -> bool {
    for r in 0..2 {
        for c in 0..2 {
            if !mi_at(f, mi_row + r, mi_col + c).skip_txfm {
                return false;
            }
        }
    }
    true
}

/// `av1_cdef_compute_sb_list`: the fb's non-skip 8x8 units as `(by, bx)`.
fn compute_sb_list(
    f: &CdefSearchFrame,
    mi_row: i32,
    mi_col: i32,
    dlist: &mut [(u8, u8); 256],
    bs: u8,
) -> usize {
    let mut maxc = f.mi_cols - mi_col;
    let mut maxr = f.mi_rows - mi_row;
    if bs == BLOCK_128X128 || bs == BLOCK_128X64 {
        maxc = maxc.min(MI_SIZE_128X128);
    } else {
        maxc = maxc.min(MI_SIZE_64X64);
    }
    if bs == BLOCK_128X128 || bs == BLOCK_64X128 {
        maxr = maxr.min(MI_SIZE_128X128);
    } else {
        maxr = maxr.min(MI_SIZE_64X64);
    }
    let mut count = 0usize;
    let mut r = 0;
    while r < maxr {
        let mut c = 0;
        while c < maxc {
            if !is_8x8_block_skip(f, mi_row + r, mi_col + c) {
                dlist[count] = ((r >> 1) as u8, (c >> 1) as u8);
                count += 1;
            }
            c += 2;
        }
        r += 2;
    }
    count
}

// ---- filtering + error (pickcdef.c:226-506, cdef_block.c:283-426) ----------

fn fill_rect(dst: &mut [u16], off: usize, dstride: usize, v: usize, h: usize, x: u16) {
    for i in 0..v {
        dst[off + i * dstride..off + i * dstride + h].fill(x);
    }
}

/// `fill_borders_for_fbs_on_frame_boundary` (pickcdef.c:317): CDEF_VERY_LARGE
/// into every border region that lies outside the frame.
#[allow(clippy::too_many_arguments)]
fn fill_borders_for_fbs_on_frame_boundary(
    inbuf: &mut [u16],
    hfilt_size: usize,
    vfilt_size: usize,
    on_left: bool,
    on_right: bool,
    on_top: bool,
    on_bottom: bool,
) {
    if !on_left && !on_right && !on_top && !on_bottom {
        return;
    }
    let very = CDEF_VERY_LARGE as u16;
    if on_bottom {
        let off = (vfilt_size + VB) * CDEF_BSTRIDE + HB;
        fill_rect(inbuf, off, CDEF_BSTRIDE, VB, hfilt_size, very);
    }
    if on_bottom || on_left {
        let off = (vfilt_size + VB) * CDEF_BSTRIDE;
        fill_rect(inbuf, off, CDEF_BSTRIDE, VB, HB, very);
    }
    if on_bottom || on_right {
        let off = (vfilt_size + VB) * CDEF_BSTRIDE + hfilt_size + HB;
        fill_rect(inbuf, off, CDEF_BSTRIDE, VB, HB, very);
    }
    if on_top {
        fill_rect(inbuf, HB, CDEF_BSTRIDE, VB, hfilt_size, very);
    }
    if on_top || on_left {
        fill_rect(inbuf, 0, CDEF_BSTRIDE, VB, HB, very);
    }
    if on_top || on_right {
        fill_rect(inbuf, hfilt_size + HB, CDEF_BSTRIDE, VB, HB, very);
    }
    if on_left {
        let off = VB * CDEF_BSTRIDE;
        fill_rect(inbuf, off, CDEF_BSTRIDE, vfilt_size, HB, very);
    }
    if on_right {
        let off = VB * CDEF_BSTRIDE + hfilt_size + HB;
        fill_rect(inbuf, off, CDEF_BSTRIDE, vfilt_size, HB, very);
    }
}

/// `adjust_strength` (cdef_block.c:289) — variance-adaptive luma primary
/// strength (private copy, same as `aom-cdef/src/frame.rs`).
fn adjust_strength(strength: i32, var: i32) -> i32 {
    let i = if var >> 6 != 0 {
        (31 - ((var >> 6) as u32).leading_zeros() as i32).min(12)
    } else {
        0
    };
    if var != 0 {
        (strength * (4 + i) + 8) >> 4
    } else {
        0
    }
}

/// `av1_cdef_filter_fb` in its SEARCH shape (`dirinit != NULL`,
/// cdef_block.c:323): luma directions found once per fb and cached in
/// `dir`/`var` across every strength index and plane; the 4:2:2/4:4:0
/// remap mutates the shared `dir` on EVERY plane-1 call (a real shipped-C
/// property of the search — the remap is not gated on `dirinit`); output
/// written either at frame positions with `dstride` (`packed = false`,
/// C's `dst8` arm used by the bd8 search) or packed per dlist entry with
/// stride `1 << bw_log2` (`packed = true`, C's `dst16`+`dirinit` arm used
/// by the high-bitdepth search). The `pri == 0 && sec == 0` early COPY arm
/// (only reachable packed) copies the input without touching `dirinit`.
#[allow(clippy::too_many_arguments)]
fn cdef_filter_fb_search(
    dst: &mut [u16],
    dstride: usize,
    packed: bool,
    inbuf: &[u16],
    xdec: usize,
    ydec: usize,
    dir: &mut [[i32; NB]; NB],
    dirinit: &mut bool,
    var: &mut [[i32; NB]; NB],
    pli: usize,
    dlist: &[(u8, u8)],
    cdef_count: usize,
    level: i32,
    sec_strength_in: i32,
    damping_in: i32,
    coeff_shift: i32,
) {
    let pri_strength = level << coeff_shift;
    let sec_strength = sec_strength_in << coeff_shift;
    let damping = damping_in + coeff_shift - i32::from(pli != 0);
    let bw_log2 = 3 - xdec;
    let bh_log2 = 3 - ydec;

    if packed && pri_strength == 0 && sec_strength == 0 {
        // The av1_cdef_search-only copy arm (cdef_block.c:337-353).
        for (bi, &(by, bx)) in dlist[..cdef_count].iter().enumerate() {
            let (by, bx) = (by as usize, bx as usize);
            for iy in 0..1usize << bh_log2 {
                let s = IN_BASE + ((by << bh_log2) + iy) * CDEF_BSTRIDE + (bx << bw_log2);
                let d = (bi << (bw_log2 + bh_log2)) + (iy << bw_log2);
                dst[d..d + (1 << bw_log2)].copy_from_slice(&inbuf[s..s + (1 << bw_log2)]);
            }
        }
        return;
    }

    if pli == 0 && !*dirinit {
        // aom_cdef_find_dir over the dlist (the C dual call == two singles).
        for &(by, bx) in &dlist[..cdef_count] {
            let (by, bx) = (by as usize, bx as usize);
            let pos = IN_BASE + 8 * by * CDEF_BSTRIDE + 8 * bx;
            let (d, v) = cdef_find_dir(&inbuf[pos..], CDEF_BSTRIDE, coeff_shift);
            dir[by][bx] = d;
            var[by][bx] = v;
        }
        *dirinit = true;
    }
    if pli == 1 && xdec != ydec {
        const CONV422: [i32; 8] = [7, 0, 2, 4, 5, 6, 6, 6];
        const CONV440: [i32; 8] = [1, 2, 2, 2, 3, 4, 6, 0];
        for &(by, bx) in &dlist[..cdef_count] {
            let (by, bx) = (by as usize, bx as usize);
            let d = dir[by][bx] as usize;
            dir[by][bx] = if xdec != 0 { CONV422[d] } else { CONV440[d] };
        }
    }

    let block_width = 8 >> xdec;
    let block_height = 8 >> ydec;
    for (bi, &(by, bx)) in dlist[..cdef_count].iter().enumerate() {
        let (by, bx) = (by as usize, bx as usize);
        let t = if pli != 0 {
            pri_strength
        } else {
            adjust_strength(pri_strength, var[by][bx])
        };
        let (dst_off, eff_stride) = if packed {
            (bi << (bw_log2 + bh_log2), 1usize << bw_log2)
        } else {
            ((by << bh_log2) * dstride + (bx << bw_log2), dstride)
        };
        cdef_filter_block_16(
            dst,
            dst_off,
            eff_stride,
            inbuf,
            IN_BASE + ((by * CDEF_BSTRIDE) << bh_log2) + (bx << bw_log2),
            t,
            sec_strength,
            if pri_strength != 0 { dir[by][bx] } else { 0 },
            damping,
            damping,
            coeff_shift,
            block_width,
            block_height,
            t != 0,
            sec_strength != 0,
        );
    }
}

/// `aom_sse` over u16 samples (values <= 255 in the bd8 path where this is
/// used, numerically identical to the C u8 reads); u64 sum of squared
/// diffs — summation regrouping of C's 16x16-tiled SIMD is exact.
#[allow(clippy::too_many_arguments)]
fn sse_rect(
    a: &[u16],
    a_off: usize,
    a_stride: usize,
    b: &[u16],
    b_off: usize,
    b_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sum = 0u64;
    for r in 0..h {
        let ar = &a[a_off + r * a_stride..a_off + r * a_stride + w];
        let br = &b[b_off + r * b_stride..b_off + r * b_stride + w];
        for (&x, &y) in ar.iter().zip(br) {
            let d = i64::from(x) - i64::from(y);
            sum += (d * d) as u64;
        }
    }
    sum
}

/// `compute_cdef_dist_highbd` (pickcdef.c:237): per-dlist-block
/// `aom_mse_wxh_16bit_highbd` between the source and the PACKED filtered
/// blocks, total `>> 2*coeff_shift`.
#[allow(clippy::too_many_arguments)]
fn compute_cdef_dist_highbd(
    src_plane: &[u16],
    src_off: usize,
    src_stride: usize,
    tmp: &[u16],
    dlist: &[(u8, u8)],
    cdef_count: usize,
    bw_log2: usize,
    bh_log2: usize,
    coeff_shift: i32,
) -> u64 {
    let w = 1usize << bw_log2;
    let h = 1usize << bh_log2;
    let mut sum = 0u64;
    for (bi, &(by, bx)) in dlist[..cdef_count].iter().enumerate() {
        let (by, bx) = (by as usize, bx as usize);
        sum += sse_rect(
            src_plane,
            src_off + (by << bh_log2) * src_stride + (bx << bw_log2),
            src_stride,
            tmp,
            bi << (bh_log2 + bw_log2),
            w,
            w,
            h,
        );
    }
    sum >> (2 * coeff_shift)
}

/// `get_filt_error` (pickcdef.c:401): the (plane, fb, strength) error.
#[allow(clippy::too_many_arguments)]
fn get_filt_error(
    f: &CdefSearchFrame,
    pli: usize,
    use_hbd: bool,
    dlist: &[(u8, u8)],
    cdef_count: usize,
    dir: &mut [[i32; NB]; NB],
    dirinit: &mut bool,
    var: &mut [[i32; NB]; NB],
    inbuf: &[u16],
    tmp: &mut [u16],
    row: usize,
    col: usize,
    pri_strength: i32,
    sec_strength: i32,
    damping: i32,
    coeff_shift: i32,
    bs: u8,
) -> u64 {
    let (ss_x, ss_y) = if pli == 0 { (0, 0) } else { (f.ss_x, f.ss_y) };
    let plane_bsize = SS_SIZE_LOOKUP[bs as usize][ss_x][ss_y] as usize;
    let bw_log2 = 3 - ss_x;
    let bh_log2 = 3 - ss_y;
    let recon = f.recon(pli);
    let src = f.src(pli);
    let stride = f.stride;
    let fb_off = row * stride + col;

    if !use_hbd {
        let bsw = BLOCK_SIZE_WIDE[plane_bsize];
        let bsh = BLOCK_SIZE_HIGH[plane_bsize];
        let tot_blk_count = (bsw * bsh) >> (bw_log2 + bh_log2);
        if cdef_count == tot_blk_count {
            if pri_strength == 0 && sec_strength == 0 {
                // Zero strength: error vs the unfiltered reconstruction.
                sse_rect(src, fb_off, stride, recon, fb_off, stride, bsw, bsh)
            } else {
                cdef_filter_fb_search(
                    tmp,
                    MAX_SB,
                    false,
                    inbuf,
                    ss_x,
                    ss_y,
                    dir,
                    dirinit,
                    var,
                    pli,
                    dlist,
                    cdef_count,
                    pri_strength,
                    sec_strength + i32::from(sec_strength == 3),
                    damping,
                    coeff_shift,
                );
                sse_rect(src, fb_off, stride, tmp, 0, MAX_SB, bsw, bsh)
            }
        } else if pri_strength == 0 && sec_strength == 0 {
            let mut sum = 0u64;
            for &(by, bx) in &dlist[..cdef_count] {
                let o = fb_off + ((by as usize) << bh_log2) * stride + ((bx as usize) << bw_log2);
                sum += sse_rect(src, o, stride, recon, o, stride, 1 << bw_log2, 1 << bh_log2);
            }
            sum
        } else {
            cdef_filter_fb_search(
                tmp,
                MAX_SB,
                false,
                inbuf,
                ss_x,
                ss_y,
                dir,
                dirinit,
                var,
                pli,
                dlist,
                cdef_count,
                pri_strength,
                sec_strength + i32::from(sec_strength == 3),
                damping,
                coeff_shift,
            );
            let mut sum = 0u64;
            for &(by, bx) in &dlist[..cdef_count] {
                let (by, bx) = (by as usize, bx as usize);
                sum += sse_rect(
                    src,
                    fb_off + (by << bh_log2) * stride + (bx << bw_log2),
                    stride,
                    tmp,
                    (by << bh_log2) * MAX_SB + (bx << bw_log2),
                    MAX_SB,
                    1 << bw_log2,
                    1 << bh_log2,
                );
            }
            sum
        }
    } else {
        // High bitdepth: filter (or copy, at zero strength) into the PACKED
        // tmp layout, then the per-block distance >> 2*coeff_shift.
        cdef_filter_fb_search(
            tmp,
            CDEF_BSTRIDE,
            true,
            inbuf,
            ss_x,
            ss_y,
            dir,
            dirinit,
            var,
            pli,
            dlist,
            cdef_count,
            pri_strength,
            sec_strength + i32::from(sec_strength == 3),
            damping,
            coeff_shift,
        );
        compute_cdef_dist_highbd(
            src,
            fb_off,
            stride,
            tmp,
            dlist,
            cdef_count,
            bw_log2,
            bh_log2,
            coeff_shift,
        )
    }
}

// ---- per-fb MSE walk (pickcdef.c:517-647) -----------------------------------

/// `av1_cdef_mse_calc_block`: fill `mse_y[gi]` / `mse_uv[gi]` for one fb.
#[allow(clippy::too_many_arguments)]
fn cdef_mse_calc_block(
    f: &CdefSearchFrame,
    pick_method: i32,
    total_strengths: usize,
    coeff_shift: i32,
    damping: i32,
    nvfb: i32,
    nhfb: i32,
    fbr: i32,
    fbc: i32,
    mse_y: &mut [u64; TOTAL_STRENGTHS],
    mse_uv: &mut [u64; TOTAL_STRENGTHS],
    inbuf: &mut [u16],
    tmp: &mut [u16],
) {
    let use_hbd = f.bd > 8;
    let mut dlist = [(0u8, 0u8); 256];
    let mut dir = [[0i32; NB]; NB];
    let mut var = [[0i32; NB]; NB];

    let mut nhb = (f.mi_cols - MI_SIZE_64X64 * fbc).min(MI_SIZE_64X64);
    let mut nvb = (f.mi_rows - MI_SIZE_64X64 * fbr).min(MI_SIZE_64X64);
    let mut hb_step = 1;
    let mut vb_step = 1;
    let mbmi_bsize = mi_at(f, MI_SIZE_64X64 * fbr, MI_SIZE_64X64 * fbc).bsize;
    let bs = if mbmi_bsize == BLOCK_128X128
        || mbmi_bsize == BLOCK_128X64
        || mbmi_bsize == BLOCK_64X128
    {
        if mbmi_bsize == BLOCK_128X128 || mbmi_bsize == BLOCK_128X64 {
            nhb = (f.mi_cols - MI_SIZE_64X64 * fbc).min(MI_SIZE_128X128);
            hb_step = 2;
        }
        if mbmi_bsize == BLOCK_128X128 || mbmi_bsize == BLOCK_64X128 {
            nvb = (f.mi_rows - MI_SIZE_64X64 * fbr).min(MI_SIZE_128X128);
            vb_step = 2;
        }
        mbmi_bsize
    } else {
        BLOCK_64X64
    };

    let cdef_count = compute_sb_list(f, fbr * MI_SIZE_64X64, fbc * MI_SIZE_64X64, &mut dlist, bs);

    let on_left = fbc == 0;
    let on_right = fbc + hb_step == nhfb;
    let on_top = fbr == 0;
    let on_bottom = fbr + vb_step == nvfb;
    let yoff = VB * usize::from(!on_top);
    let xoff = HB * usize::from(!on_left);
    let mut dirinit = false;

    for pli in 0..f.num_planes() {
        let (ss_x, ss_y) = if pli == 0 { (0, 0) } else { (f.ss_x, f.ss_y) };
        let mi_wide_l2 = 2 - ss_x;
        let mi_high_l2 = 2 - ss_y;
        let hfilt_size = (nhb as usize) << mi_wide_l2;
        let vfilt_size = (nvb as usize) << mi_high_l2;
        let ysize = vfilt_size + VB * usize::from(!on_bottom) + yoff;
        let xsize = hfilt_size + HB * usize::from(!on_right) + xoff;
        let row = ((fbr * MI_SIZE_64X64) as usize) << mi_high_l2;
        let col = ((fbc * MI_SIZE_64X64) as usize) << mi_wide_l2;
        let recon = f.recon(pli);

        // av1_cdef_copy_sb8_16: the deblocked recon rect (incl. the live
        // 2-row/8-col neighbourhood on non-boundary sides) into the work
        // buffer at (VB - yoff, HB - xoff).
        {
            let doff = (VB - yoff) * CDEF_BSTRIDE + (HB - xoff);
            let soff = (row - yoff) * f.stride + (col - xoff);
            for r in 0..ysize {
                inbuf[doff + r * CDEF_BSTRIDE..doff + r * CDEF_BSTRIDE + xsize]
                    .copy_from_slice(&recon[soff + r * f.stride..soff + r * f.stride + xsize]);
            }
        }
        fill_borders_for_fbs_on_frame_boundary(
            inbuf, hfilt_size, vfilt_size, on_left, on_right, on_top, on_bottom,
        );

        for gi in 0..total_strengths {
            let (pri_strength, sec_strength) = get_cdef_filter_strengths(pick_method, gi as i32);
            let curr_mse = get_filt_error(
                f,
                pli,
                use_hbd,
                &dlist,
                cdef_count,
                &mut dir,
                &mut dirinit,
                &mut var,
                inbuf,
                tmp,
                row,
                col,
                pri_strength,
                sec_strength,
                damping,
                coeff_shift,
                bs,
            );
            if pli < 2 {
                if pli == 0 {
                    mse_y[gi] = curr_mse;
                } else {
                    mse_uv[gi] = curr_mse;
                }
            } else {
                mse_uv[gi] += curr_mse;
            }
        }
    }
}

// ---- the search (pickcdef.c:838-1100) ---------------------------------------

/// `get_msb` (31 - clz), n > 0.
fn get_msb(n: u32) -> i32 {
    31 - n.leading_zeros() as i32
}

/// `av1_cdef_search` for the non-adaptive (`CDEF_ALL`) one-pass envelope:
/// full per-fb MSE grid, joint strength-set selection over 0..=3 signaling
/// bits, per-unit best-index assignment, fast-method strength re-mapping.
/// `pick_method` = `CDEF_FULL_SEARCH`(0) .. `CDEF_FAST_SEARCH_LVL5`(5)
/// (the [`crate::speed_features`] `cdef_pick_method` value; speed 0 = FULL).
pub fn av1_cdef_search(f: &CdefSearchFrame, pick_method: i32) -> CdefSearchResult {
    assert!(
        (0..=5).contains(&pick_method),
        "CDEF_PICK_FROM_Q (speed >= 7 rt) is out of this port's envelope"
    );
    let num_planes = f.num_planes();
    let damping = 3 + (f.base_qindex >> 6);
    let fast = (1..=5).contains(&pick_method);
    let coeff_shift = (i32::from(f.bd) - 8).max(0);
    let total_strengths = NB_CDEF_STRENGTHS[pick_method as usize];
    let nvfb = (f.mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    let nhfb = (f.mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;

    // Frame-level MSE grid (cdef_mse_calc_frame).
    let mut mse_y: Vec<[u64; TOTAL_STRENGTHS]> = Vec::new();
    let mut mse_uv: Vec<[u64; TOTAL_STRENGTHS]> = Vec::new();
    let mut sb_units: Vec<usize> = Vec::new();
    let mut inbuf = vec![0u16; INBUF_SIZE];
    let mut tmp = vec![0u16; MAX_SB * MAX_SB];
    for fbr in 0..nvfb {
        for fbc in 0..nhfb {
            if cdef_sb_skip(f, fbr, fbc) {
                continue;
            }
            let mut row_y = [0u64; TOTAL_STRENGTHS];
            let mut row_uv = [0u64; TOTAL_STRENGTHS];
            cdef_mse_calc_block(
                f,
                pick_method,
                total_strengths,
                coeff_shift,
                damping,
                nvfb,
                nhfb,
                fbr,
                fbc,
                &mut row_y,
                &mut row_uv,
                &mut inbuf,
                &mut tmp,
            );
            mse_y.push(row_y);
            mse_uv.push(row_uv);
            sb_units.push((fbr * nhfb + fbc) as usize);
        }
    }
    let sb_count = mse_y.len();

    // Signaling-bits RD loop.
    let joint_strengths = if num_planes > 1 {
        total_strengths * total_strengths
    } else {
        total_strengths
    };
    let max_signaling_bits = if joint_strengths == 1 {
        0
    } else {
        get_msb(joint_strengths as u32 - 1) + 1
    };
    let mut nb_strength_bits = 0i32;
    let mut best_rd = u64::MAX;
    let mut cdef_strengths = [0i32; 8];
    let mut cdef_uv_strengths = [0i32; 8];
    for i in 0..=3i32 {
        if i > max_signaling_bits {
            break;
        }
        let mut best_lev0 = [0i32; CDEF_MAX_STRENGTHS];
        let mut best_lev1 = [0i32; CDEF_MAX_STRENGTHS];
        let nb_strengths = 1usize << i;
        let tot_mse = if num_planes > 1 {
            joint_strength_search_dual(
                &mut best_lev0,
                &mut best_lev1,
                nb_strengths,
                [&mse_y, &mse_uv],
                total_strengths,
            )
        } else {
            joint_strength_search(&mut best_lev0, nb_strengths, &mse_y, total_strengths, fast)
        };
        let total_bits = sb_count as i32 * i
            + nb_strengths as i32 * CDEF_STRENGTH_BITS * if num_planes > 1 { 2 } else { 1 };
        // av1_cost_literal + RDCOST (cost.h:29, rd.h:32): rate*rdmult
        // rounded down 9 bits, plus dist << RDDIV_BITS(7); dist = mse*16.
        let rate_cost = i64::from(total_bits) * 512;
        let dist = tot_mse * 16;
        let rd = ((rate_cost * i64::from(f.rdmult) + 256) >> 9) as u64 + dist * 128;
        if rd < best_rd {
            best_rd = rd;
            nb_strength_bits = i;
            cdef_strengths[..nb_strengths].copy_from_slice(&best_lev0[..nb_strengths]);
            if num_planes > 1 {
                cdef_uv_strengths[..nb_strengths].copy_from_slice(&best_lev1[..nb_strengths]);
            }
        }
    }

    let nb_cdef_strengths = 1usize << nb_strength_bits;

    // Per-unit best strength index (mbmi->cdef_strength stamping).
    let mut unit_strength = vec![0i32; (nvfb * nhfb) as usize];
    for i in 0..sb_count {
        let mut best_mse = u64::MAX;
        let mut best_gi = 0i32;
        for gi in 0..nb_cdef_strengths {
            let mut curr = mse_y[i][cdef_strengths[gi] as usize];
            if num_planes > 1 {
                curr += mse_uv[i][cdef_uv_strengths[gi] as usize];
            }
            if curr < best_mse {
                best_gi = gi as i32;
                best_mse = curr;
            }
        }
        unit_strength[sb_units[i]] = best_gi;
    }

    // Fast methods: convert table indices to real (pri*4 + sec) strengths.
    if fast {
        for j in 0..nb_cdef_strengths {
            let (pri, sec) = get_cdef_filter_strengths(pick_method, cdef_strengths[j]);
            cdef_strengths[j] = pri * CDEF_SEC_STRENGTHS + sec;
            if num_planes > 1 {
                let (pri, sec) = get_cdef_filter_strengths(pick_method, cdef_uv_strengths[j]);
                cdef_uv_strengths[j] = pri * CDEF_SEC_STRENGTHS + sec;
            }
        }
    }

    CdefSearchResult {
        cdef_bits: nb_strength_bits,
        nb_cdef_strengths,
        cdef_strengths,
        cdef_uv_strengths,
        cdef_damping: damping,
        unit_strength,
        nvfb,
        nhfb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strength decomposition spot-checks, hand-computed from the C tables.
    #[test]
    fn filter_strengths_match_c_tables() {
        // FULL: gi = pri*4 + sec.
        assert_eq!(get_cdef_filter_strengths(0, 0), (0, 0));
        assert_eq!(get_cdef_filter_strengths(0, 13), (3, 1));
        assert_eq!(get_cdef_filter_strengths(0, 63), (15, 3));
        // LVL1: priconv_lvl1 = {0,1,2,3,5,7,10,13}, sec 0..3.
        assert_eq!(get_cdef_filter_strengths(1, 13), (3, 1));
        assert_eq!(get_cdef_filter_strengths(1, 30), (13, 2));
        // LVL2: priconv_lvl2 = {0,2,4,8,14}.
        assert_eq!(get_cdef_filter_strengths(2, 17), (14, 1));
        // LVL3: sec via secconv_lvl3 = {0,2}, 2 per pri.
        assert_eq!(get_cdef_filter_strengths(3, 7), (8, 2));
        // LVL4: priconv_lvl4 = {0,11}.
        assert_eq!(get_cdef_filter_strengths(4, 3), (11, 2));
        // LVL5: priconv_lvl5 = {0,5}, single sec 0.
        assert_eq!(get_cdef_filter_strengths(5, 1), (5, 0));
    }

    /// `search_one` at nb_strengths=0 is a plain argmin over summed rows.
    #[test]
    fn search_one_is_argmin_for_first_strength() {
        let mut rows: Vec<[u64; TOTAL_STRENGTHS]> = Vec::new();
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) & 0xffff
        };
        for _ in 0..17 {
            let mut r = [0u64; TOTAL_STRENGTHS];
            for x in &mut r {
                *x = rnd();
            }
            rows.push(r);
        }
        let mut lev = [0i32; CDEF_MAX_STRENGTHS];
        let got = search_one(&mut lev, 0, &rows, TOTAL_STRENGTHS);
        // Brute force.
        let mut best = (u64::MAX, 0usize);
        for j in 0..TOTAL_STRENGTHS {
            let tot: u64 = rows.iter().map(|r| r[j]).sum();
            if tot < best.0 {
                best = (tot, j);
            }
        }
        assert_eq!((got, lev[0] as usize), best);
    }

    /// The greedy dual search with nb=1 must equal the exhaustive joint
    /// argmin over (luma, chroma) pairs.
    #[test]
    fn search_one_dual_nb1_is_joint_argmin() {
        let mut seed = 0x243f6a8885a308d3u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) & 0xffff
        };
        let gen_rows = |rnd: &mut dyn FnMut() -> u64| {
            let mut rows: Vec<[u64; TOTAL_STRENGTHS]> = Vec::new();
            for _ in 0..9 {
                let mut r = [0u64; TOTAL_STRENGTHS];
                for x in &mut r {
                    *x = rnd();
                }
                rows.push(r);
            }
            rows
        };
        let m0 = gen_rows(&mut rnd);
        let m1 = gen_rows(&mut rnd);
        let mut lev0 = [0i32; CDEF_MAX_STRENGTHS];
        let mut lev1 = [0i32; CDEF_MAX_STRENGTHS];
        let got = search_one_dual(&mut lev0, &mut lev1, 0, [&m0, &m1], TOTAL_STRENGTHS);
        let mut best = (u64::MAX, 0usize, 0usize);
        for j in 0..TOTAL_STRENGTHS {
            for k in 0..TOTAL_STRENGTHS {
                let tot: u64 = (0..m0.len()).map(|i| m0[i][j] + m1[i][k]).sum();
                if tot < best.0 {
                    best = (tot, j, k);
                }
            }
        }
        assert_eq!((got, lev0[0] as usize, lev1[0] as usize), best);
    }
}
