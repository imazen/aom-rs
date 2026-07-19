//! Inter-frame motion estimation — the net-new subpel search machinery
//! (INTER-ENCODE-ROADMAP.md chunk 2d).
//!
//! The full-pel diamond/mesh search is the shared intrabc/inter core in
//! [`crate::intrabc_search`] (retargeted to a reference frame). This module
//! holds the pieces that are net-new for inter: the **upsampled subpel
//! predictor** ([`upsampled_pred`], the cost primitive of
//! `av1_find_best_sub_pixel_tree`) and — as they land — the subpel tree search
//! itself.
//!
//! All lowbd (bd = 8). The port stores planes as `u16` (bd8 values `0..=255`),
//! matching the rest of the codebase; the arithmetic is byte-identical to
//! libaom's `u8` kernels since every value fits in a byte.

use aom_convolve::SUB_PEL_FILTERS_8;
use aom_entropy::partition::get_mv_joint;

const FILTER_BITS: i32 = 7;
const SUBPEL_TAPS: usize = 8;
/// `SUBPEL_TAPS / 2 - 1` — the 8-tap filter's left/top origin offset.
const FILTER_OFF: usize = SUBPEL_TAPS / 2 - 1; // 3

#[inline]
fn round_pow2(v: i32, n: i32) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

#[inline]
fn clip_pixel(v: i32) -> u16 {
    v.clamp(0, 255) as u16
}

/// One horizontal 8-tap pass (`aom_convolve8_horiz_c` with `x_step_q4 ==
/// SUBPEL_SHIFTS`, i.e. the fixed-phase `aom_upsampled_pred` use): for each
/// output `(y, x)`, `dst = clip(round(Σ_k kernel[k]·src[y·stride + x - 3 + k],
/// FILTER_BITS))`. `src_off` is the block origin; the tap reads `x-3 .. x+4`, so
/// `src` needs `>= 3` samples of left border and `>= 4` of right.
#[allow(clippy::too_many_arguments)]
fn convolve8_horiz(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    kernel: &[i16; 8],
) {
    for y in 0..h {
        let row = src_off as isize + (y * src_stride) as isize - FILTER_OFF as isize;
        for x in 0..w {
            let base = row + x as isize;
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                sum += kernel[k] as i32 * src[(base + k as isize) as usize] as i32;
            }
            dst[y * dst_stride + x] = clip_pixel(round_pow2(sum, FILTER_BITS));
        }
    }
}

/// One vertical 8-tap pass (`aom_convolve8_vert_c`, fixed-phase): for each
/// output `(y, x)`, `dst = clip(round(Σ_k kernel[k]·src[(y - 3 + k)·stride + x],
/// FILTER_BITS))`. `src` needs `>= 3` samples of top border and `>= 4` of bottom.
#[allow(clippy::too_many_arguments)]
fn convolve8_vert(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
    w: usize,
    h: usize,
    kernel: &[i16; 8],
) {
    for y in 0..h {
        for x in 0..w {
            let base = src_off as isize
                + (y as isize - FILTER_OFF as isize) * src_stride as isize
                + x as isize;
            let mut sum = 0i32;
            for k in 0..SUBPEL_TAPS {
                sum += kernel[k] as i32 * src[(base + (k as isize) * src_stride as isize) as usize] as i32;
            }
            dst[y * dst_stride + x] = clip_pixel(round_pow2(sum, FILTER_BITS));
        }
    }
}

/// `aom_upsampled_pred_c` (av1/encoder/reconinter_enc.c:462), lowbd, unscaled,
/// `subpel_search == USE_8_TAPS` (`av1_get_filter(USE_8_TAPS)` =
/// `EIGHTTAP_REGULAR`). The fixed-phase 8-tap subpel predictor the speed-0
/// subpel motion search builds (`upsampled_pref_error` ->
/// `check_better`/`upsampled_setup_center_error`).
///
/// The C kernel selects `av1_get_interp_filter_subpel_kernel(filter,
/// subpel_q3 << 1)` — the `EIGHTTAP_REGULAR` row at the doubled 1/16-pel phase,
/// which is [`SUB_PEL_FILTERS_8`]`[subpel_q3 << 1]`. Dispatch matches C:
/// - `(0, 0)` → block copy;
/// - `(x, 0)` → single horizontal pass;
/// - `(0, y)` → single vertical pass;
/// - `(x, y)` → horizontal into a `(h + 7)`-row intermediate (u8-clipped, as the
///   C 2-D path clips between passes), then vertical.
///
/// `refb`/`ref_off`/`ref_stride` describe the reference plane; `ref_off` is the
/// fullpel block origin with `>= 3` samples of border before and `>= 4` after in
/// every subpel-filtered direction (the caller's `get_buf_from_mv` position on a
/// border-extended reference frame). `subpel_x_q3`/`subpel_y_q3` are 1/8-pel
/// phases in `0..=7`. Returns the `w`×`h` predictor (u16 bd8, tight stride `w`).
///
/// Differentially locked vs the REAL `aom_upsampled_pred_c` in
/// `tests/upsampled_pred_diff.rs`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn upsampled_pred(
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
    subpel_x_q3: usize,
    subpel_y_q3: usize,
) -> Vec<u16> {
    debug_assert!(subpel_x_q3 <= 7 && subpel_y_q3 <= 7);
    let mut dst = vec![0u16; w * h];
    let need_x = subpel_x_q3 != 0;
    let need_y = subpel_y_q3 != 0;

    if !need_x && !need_y {
        for y in 0..h {
            let s = ref_off + y * ref_stride;
            dst[y * w..y * w + w].copy_from_slice(&refb[s..s + w]);
        }
    } else if !need_y {
        let kx = &SUB_PEL_FILTERS_8[subpel_x_q3 << 1];
        convolve8_horiz(refb, ref_off, ref_stride, &mut dst, w, w, h, kx);
    } else if !need_x {
        let ky = &SUB_PEL_FILTERS_8[subpel_y_q3 << 1];
        convolve8_vert(refb, ref_off, ref_stride, &mut dst, w, w, h, ky);
    } else {
        // 2-D separable: horizontal into an (h + 7)-row intermediate starting 3
        // rows above the block origin, then vertical. The intermediate is
        // u8-clipped per pass (round to FILTER_BITS + clip), byte-identical to
        // aom_convolve8_horiz_c writing its uint8_t temp.
        let kx = &SUB_PEL_FILTERS_8[subpel_x_q3 << 1];
        let ky = &SUB_PEL_FILTERS_8[subpel_y_q3 << 1];
        let inter_h = h + SUBPEL_TAPS - 1; // h + 7
        let mut temp = vec![0u16; inter_h * w];
        let horiz_off = ref_off - FILTER_OFF * ref_stride;
        convolve8_horiz(refb, horiz_off, ref_stride, &mut temp, w, w, inter_h, kx);
        // The block origin sits at intermediate row FILTER_OFF (= 3); the
        // vertical pass reads temp[(y - 3 + k) + 3] = temp[y + k].
        convolve8_vert(&temp, FILTER_OFF * w, w, &mut dst, w, w, h, ky);
    }
    dst
}

// ===================================================================
// Subpel motion search — av1_find_best_sub_pixel_tree (mcomp.c:3266),
// the SUBPEL_TREE (full) variant with USE_8_TAPS accuracy — the speed-0
// allintra/GOOD path. Lowbd, unscaled, single-ref translational.
// ===================================================================

/// `MV_MAX` (entropymv.h): `(1 << (MV_CLASSES + CLASS0_BITS + 2)) - 1` = 16383.
/// The per-component MV cost tables ([`SubpelSearchParams::mvcost0`] /
/// `mvcost1`) are centred here: `mvcost[MV_MAX + v]` is the cost of component
/// value `v`.
pub const MV_MAX: i32 = (1 << 14) - 1;
/// `INIT_SUBPEL_STEP_SIZE` (mcomp.c:2466): the half-pel starting step (4/8).
const INIT_SUBPEL_STEP_SIZE: i32 = 4;
/// `FULL_PEL` (SUBPEL_FORCE_STOP, mcomp.h:280).
const FULL_PEL: i32 = 3;
/// `INT_MAX` — the out-of-range / initial `besterr` sentinel. libaom uses the
/// SIGNED `INT_MAX` (`0x7FFF_FFFF`) for the unsigned `besterr`, NOT `UINT_MAX`.
const SUBPEL_INT_MAX: u32 = i32::MAX as u32;

/// `SubpelMvLimits` (mv.h): the 1/8-pel MV range the search may not leave.
#[derive(Clone, Copy, Debug)]
pub struct SubpelMvLimits {
    pub row_min: i32,
    pub row_max: i32,
    pub col_min: i32,
    pub col_max: i32,
}

/// Inputs to [`find_best_sub_pixel_tree`]. `src`/`ref` planes are `u16` bd8
/// (values `0..=255`). `ref_origin` is the reference `buf_2d` origin for a
/// zero MV (`get_buf_from_mv` offsets it by `mv >> 3` per component); it needs
/// enough border for the search excursion plus the 8-tap margin. MVs are
/// 1/8-pel `(row, col)`.
pub struct SubpelSearchParams<'a> {
    pub src: &'a [u16],
    pub src_off: usize,
    pub src_stride: usize,
    pub refb: &'a [u16],
    pub ref_origin: usize,
    pub ref_stride: usize,
    pub w: usize,
    pub h: usize,
    /// Fullpel search result promoted to 1/8-pel (`get_mv_from_fullmv`).
    pub start_mv: (i32, i32),
    /// Predicted MV the cost is measured against.
    pub ref_mv: (i32, i32),
    pub mvjcost: [i32; 4],
    /// Per-component MV cost tables, centred at [`MV_MAX`] (`mvcost[MV_MAX + v]`).
    pub mvcost0: &'a [i32],
    pub mvcost1: &'a [i32],
    pub error_per_bit: i32,
    pub allow_hp: bool,
    /// `SUBPEL_FORCE_STOP`: 0 = EIGHTH_PEL … 3 = FULL_PEL.
    pub forced_stop: i32,
    pub iters_per_step: i32,
    pub limits: SubpelMvLimits,
}

/// Output of [`find_best_sub_pixel_tree`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubpelResult {
    pub best_mv: (i32, i32),
    pub distortion: i32,
    pub sse: u32,
    /// Return value of `av1_find_best_sub_pixel_tree` (distortion + mv cost).
    pub besterr: u32,
}

#[inline]
fn round_pow2_i64(v: i64, n: i32) -> i64 {
    (v + (1i64 << (n - 1))) >> n
}

/// `mv_err_cost` (mcomp.c:317) at `MV_COST_ENTROPY`: the MV rate→distortion cost
/// of `mv` (1/8-pel) relative to `ref_mv`, with per-component cost tables centred
/// at [`MV_MAX`]. Shift = `RDDIV_BITS(7) + AV1_PROB_COST_SHIFT(9) - RD_EPB_SHIFT(6)
/// + PIXEL_TRANSFORM_ERROR_SCALE(4) = 14`.
#[inline]
fn mv_err_cost_entropy(
    mv: (i32, i32),
    ref_mv: (i32, i32),
    mvjcost: &[i32; 4],
    mvcost0: &[i32],
    mvcost1: &[i32],
    error_per_bit: i32,
) -> i32 {
    let dr = mv.0 - ref_mv.0;
    let dc = mv.1 - ref_mv.1;
    let joint = get_mv_joint(dr, dc) as usize;
    let mvc = mvjcost[joint] as i64
        + mvcost0[(MV_MAX + dr) as usize] as i64
        + mvcost1[(MV_MAX + dc) as usize] as i64;
    round_pow2_i64(mvc * error_per_bit as i64, 14) as i32
}

/// Mutable subpel-search state — mirrors the in-place `bestmv`/`besterr`/
/// `distortion`/`sse1` pointers `av1_find_best_sub_pixel_tree` threads through
/// `check_better`.
struct Search<'a> {
    p: &'a SubpelSearchParams<'a>,
    /// Source block (`w`×`h`) as a tight u8 buffer (the `vf` `b` operand).
    src8: Vec<u8>,
    best_mv: (i32, i32),
    besterr: u32,
    distortion: i32,
    sse: u32,
}

impl<'a> Search<'a> {
    fn new(p: &'a SubpelSearchParams<'a>) -> Self {
        let mut src8 = vec![0u8; p.w * p.h];
        for y in 0..p.h {
            for x in 0..p.w {
                src8[y * p.w + x] = p.src[p.src_off + y * p.src_stride + x] as u8;
            }
        }
        Search {
            p,
            src8,
            best_mv: p.start_mv,
            besterr: SUBPEL_INT_MAX,
            distortion: 0,
            sse: 0,
        }
    }

    /// `mv_err_cost_` (mcomp.c:343) at `MV_COST_ENTROPY`.
    fn mv_err_cost(&self, mv: (i32, i32)) -> i32 {
        mv_err_cost_entropy(
            mv,
            self.p.ref_mv,
            &self.p.mvjcost,
            self.p.mvcost0,
            self.p.mvcost1,
            self.p.error_per_bit,
        )
    }

    /// `upsampled_pref_error` (mcomp.c:2521), lowbd USE_8_TAPS: build the
    /// upsampled subpel predictor at `mv` and score it against the source with
    /// the plain variance. Returns `(besterr = variance, sse)`.
    fn pref_error(&self, mv: (i32, i32)) -> (u32, u32) {
        // get_subpel_part(x) = x & 7 (correct floor decomposition for negatives:
        // Rust `&`/`>>` on i32 match C's two's-complement `& 7` / arithmetic `>> 3`).
        let sx = (mv.1 & 7) as usize;
        let sy = (mv.0 & 7) as usize;
        // get_buf_from_mv: ref_origin + (mv.row>>3)*stride + (mv.col>>3).
        let ref_ptr = (self.p.ref_origin as isize
            + (mv.0 >> 3) as isize * self.p.ref_stride as isize
            + (mv.1 >> 3) as isize) as usize;
        let pred = upsampled_pred(
            self.p.refb,
            ref_ptr,
            self.p.ref_stride,
            self.p.w,
            self.p.h,
            sx,
            sy,
        );
        let pred8: Vec<u8> = pred.iter().map(|&v| v as u8).collect();
        // vfp->vf(pred, pred_stride=w, src, src_stride, &sse) -> (variance, sse).
        aom_dist::variance(&pred8, self.p.w, &self.src8, self.p.w, self.p.w, self.p.h)
    }

    fn in_range(&self, mv: (i32, i32)) -> bool {
        mv.1 >= self.p.limits.col_min
            && mv.1 <= self.p.limits.col_max
            && mv.0 >= self.p.limits.row_min
            && mv.0 <= self.p.limits.row_max
    }

    /// `check_better` (mcomp.c:2647): score `this_mv`; if it beats `besterr`,
    /// adopt it. Returns `(cost, improved)`.
    fn check_better(&mut self, this_mv: (i32, i32)) -> (u32, bool) {
        if self.in_range(this_mv) {
            let (var, sse) = self.pref_error(this_mv);
            let cost = (self.mv_err_cost(this_mv) as u32).wrapping_add(var);
            let mut improved = false;
            if cost < self.besterr {
                self.besterr = cost;
                self.best_mv = this_mv;
                self.distortion = var as i32;
                self.sse = sse;
                improved = true;
            }
            (cost, improved)
        } else {
            (SUBPEL_INT_MAX, false)
        }
    }

    /// `first_level_check` (mcomp.c:2808): the 4 cardinal ±hstep probes + the
    /// best diagonal. Returns `diag_step`.
    fn first_level_check(&mut self, this_mv: (i32, i32), hstep: i32) -> (i32, i32) {
        let (left, _) = self.check_better((this_mv.0, this_mv.1 - hstep));
        let (right, _) = self.check_better((this_mv.0, this_mv.1 + hstep));
        let (up, _) = self.check_better((this_mv.0 - hstep, this_mv.1));
        let (down, _) = self.check_better((this_mv.0 + hstep, this_mv.1));
        // get_best_diag_step (mcomp.c:2672).
        let diag_step = (
            if up <= down { -hstep } else { hstep },
            if left <= right { -hstep } else { hstep },
        );
        let diag_mv = (this_mv.0 + diag_step.0, this_mv.1 + diag_step.1);
        self.check_better(diag_mv);
        diag_step
    }

    /// `second_level_check_v2` (mcomp.c:2847), `subpel_search_type > USE_2_TAPS`
    /// arm: refine in the winning quadrant.
    fn second_level_check_v2(&mut self, this_mv: (i32, i32), mut diag_step: (i32, i32)) {
        if this_mv == self.best_mv {
            return;
        } else if this_mv.0 == self.best_mv.0 {
            diag_step.0 = -diag_step.0;
        } else if this_mv.1 == self.best_mv.1 {
            diag_step.1 = -diag_step.1;
        }
        let bm = self.best_mv;
        let row_bias = (bm.0 + diag_step.0, bm.1);
        let col_bias = (bm.0, bm.1 + diag_step.1);
        let diag_bias = (bm.0 + diag_step.0, bm.1 + diag_step.1);
        let (_, i1) = self.check_better(row_bias);
        let (_, i2) = self.check_better(col_bias);
        if i1 || i2 {
            self.check_better(diag_bias);
        }
    }
}

/// `av1_find_best_sub_pixel_tree` (mcomp.c:3266), lowbd single-ref translational,
/// unscaled, `subpel_search_type == USE_8_TAPS` (the speed-0 allintra/GOOD
/// path). Refines the fullpel [`SubpelSearchParams::start_mv`] to 1/8-pel by an
/// iterated cardinal+diagonal tree search over the upsampled-predictor variance
/// plus the MV rate cost. `start_mv_stats`/`last_mv_search_list` are not modelled
/// (the differential passes them NULL — the full center-error computation +
/// no repeat guard).
///
/// Differentially locked vs the REAL exported `av1_find_best_sub_pixel_tree` in
/// `tests/subpel_tree_diff.rs`.
#[must_use]
pub fn find_best_sub_pixel_tree(p: &SubpelSearchParams) -> SubpelResult {
    // round = AOMMIN(FULL_PEL - forced_stop, 3 - !allow_hp).
    let round = (FULL_PEL - p.forced_stop).min(3 - (!p.allow_hp) as i32);
    let mut hstep = INIT_SUBPEL_STEP_SIZE;
    let mut s = Search::new(p);

    // upsampled_setup_center_error (mcomp.c:2962): besterr = pref_error;
    // *distortion = besterr; besterr += mv_err_cost.
    let (var, sse) = s.pref_error(p.start_mv);
    s.distortion = var as i32;
    s.sse = sse;
    s.besterr = (var as i64 + s.mv_err_cost(p.start_mv) as i64) as u32;

    if round == 0 {
        return SubpelResult {
            best_mv: s.best_mv,
            distortion: s.distortion,
            sse: s.sse,
            besterr: s.besterr,
        };
    }

    for _iter in 0..round {
        let iter_center = s.best_mv;
        // check_repeated_mv_and_update with a NULL list is a no-op (returns 0).
        let diag = s.first_level_check(iter_center, hstep);
        if iter_center != s.best_mv && p.iters_per_step > 1 {
            s.second_level_check_v2(iter_center, diag);
        }
        hstep >>= 1;
    }

    SubpelResult {
        best_mv: s.best_mv,
        distortion: s.distortion,
        sse: s.sse,
        besterr: s.besterr,
    }
}

/// `av1_get_mvpred_sse` (mcomp.c:3963): the score `av1_single_motion_search`
/// assigns a full-pel search result — the plain (non-upsampled) predictor SSE at
/// `best_full_mv` plus the coded-MV rate cost. `pre`/`pre_origin` locate the
/// reference `buf_2d` origin (fullmv 0); `get_buf_from_fullmv` offsets it by the
/// integer MV. Cost is on the 1/8-pel MV (`best_full_mv * 8`). Returns
/// `sse + mv_err_cost` (libaom's `int` return; both terms non-negative here).
///
/// Differentially locked vs the REAL exported `av1_get_mvpred_sse` in
/// `tests/subpel_tree_diff.rs`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn get_mvpred_sse(
    best_full_mv: (i32, i32),
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    pre: &[u16],
    pre_origin: usize,
    pre_stride: usize,
    w: usize,
    h: usize,
    ref_mv: (i32, i32),
    mvjcost: &[i32; 4],
    mvcost0: &[i32],
    mvcost1: &[i32],
    error_per_bit: i32,
) -> u32 {
    // get_buf_from_fullmv(pre, best_mv) = pre + row*stride + col.
    let pre_at = (pre_origin as isize
        + best_full_mv.0 as isize * pre_stride as isize
        + best_full_mv.1 as isize) as usize;
    let mut src8 = vec![0u8; w * h];
    let mut pre8 = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            src8[y * w + x] = src[src_off + y * src_stride + x] as u8;
            pre8[y * w + x] = pre[pre_at + y * pre_stride + x] as u8;
        }
    }
    // vfp->vf(src->buf, src->stride, pre_at, pre->stride, &sse) -> (var, sse).
    let (_var, sse) = aom_dist::variance(&src8, w, &pre8, w, w, h);
    // get_mv_from_fullmv(best_mv) = best_mv * 8 (full-pel -> 1/8-pel).
    let mv = (best_full_mv.0 * 8, best_full_mv.1 * 8);
    let cost = mv_err_cost_entropy(mv, ref_mv, mvjcost, mvcost0, mvcost1, error_per_bit);
    sse.wrapping_add(cost as u32)
}

/// `av1_mv_bit_cost` (mcomp.c:307): the rate to code MV `mv` (1/8-pel) relative
/// to `ref_mv` — `ROUND_POWER_OF_TWO(mv_cost(diff) * weight, 7)`. `weight` is
/// `MV_COST_WEIGHT` (108) for the RD rate or `MV_COST_WEIGHT_SUB` (120) for the
/// coded DV. This is the coded-MV rate (NOT the motion-search variance-metric
/// cost [`mv_err_cost_entropy`]).
///
/// Differentially locked vs the REAL exported `av1_mv_bit_cost` in
/// `tests/subpel_tree_diff.rs`.
#[must_use]
pub fn mv_bit_cost(
    mv: (i32, i32),
    ref_mv: (i32, i32),
    mvjcost: &[i32; 4],
    mvcost0: &[i32],
    mvcost1: &[i32],
    weight: i32,
) -> i32 {
    let dr = mv.0 - ref_mv.0;
    let dc = mv.1 - ref_mv.1;
    let joint = get_mv_joint(dr, dc) as usize;
    let mvc = mvjcost[joint] + mvcost0[(MV_MAX + dr) as usize] + mvcost1[(MV_MAX + dc) as usize];
    round_pow2(mvc * weight, 7)
}
