//! Composition test for [`aom_encode::inter_me::single_motion_search`] — the
//! `av1_single_motion_search` glue (motion_search_facade.c:120) reduced to the
//! §3 single-ref SIMPLE_TRANSLATION speed-0 config.
//!
//! Both halves are already differential-locked vs the REAL exported C
//! (`full_pixel_search_inter` in `full_pixel_search_diff.rs`,
//! `find_best_sub_pixel_tree`/`mv_bit_cost` in `subpel_tree_diff.rs`), so per the
//! INTER-CHUNK2 handoff (a real-C `av1_single_motion_search` differential needs a
//! full `MACROBLOCK`/`AV1_COMP` shim) this validates the **composition** two ways:
//!
//! 1. **Glue faithfulness** — `single_motion_search` reproduces a manual
//!    `set_mv_search_range → full_pixel_search_inter → set_subpel_mv_search_range
//!    → find_best_sub_pixel_tree → mv_bit_cost` (and the `force_integer_mv` short
//!    circuit), locking the limit derivation + the full→subpel handoff + the rate.
//! 2. **Convergence** — on translational content (`src` = the reference shifted
//!    by a known integer MV) the search converges to that exact MV.

use aom_encode::inter_me::{
    find_best_sub_pixel_tree, mv_bit_cost, single_motion_search, SingleMotionSearchParams,
    SubpelMvLimits, SubpelSearchParams, MV_COST_WEIGHT,
};
use aom_encode::intrabc_search::{
    fill_nmv_costs, full_pixel_search_inter, set_mv_search_range, DvCosts, FullMvLimits,
    MV_SUBPEL_LOW,
};
use aom_entropy::default_cdfs::{DEFAULT_NMV_COMPS, DEFAULT_NMV_JOINTS};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_u64() >> 33) as i32 % (hi - lo + 1)
    }
}

const BORDER: usize = 96;

/// A bordered reference plane (`(w+2B)×(h+2B)`, random u8) with the MV(0,0)
/// origin at (BORDER, BORDER). Returns `(u16_buf, origin_off, stride)`.
fn ref_plane(rng: &mut Rng, w: usize, h: usize) -> (Vec<u16>, usize, usize) {
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let mut u16b = vec![0u16; stride * rows];
    for b in u16b.iter_mut() {
        *b = rng.byte() as u16;
    }
    let origin = BORDER * stride + BORDER;
    (u16b, origin, stride)
}

fn default_dv_costs() -> DvCosts {
    fill_nmv_costs(
        MV_SUBPEL_LOW,
        &DEFAULT_NMV_JOINTS,
        &DEFAULT_NMV_COMPS[0],
        &DEFAULT_NMV_COMPS[1],
    )
}

/// Replicate the reduced `av1_single_motion_search` composition by hand from the
/// two C-locked halves — the reference `single_motion_search` must reproduce.
#[allow(clippy::too_many_arguments)]
fn manual_compose(p: &SingleMotionSearchParams) -> ((i32, i32), i32, bool) {
    let mut full_limits = p.mv_limits;
    set_mv_search_range(&mut full_limits, p.ref_mv.0, p.ref_mv.1);
    let (bestsme, brow, bcol) = full_pixel_search_inter(
        p.src,
        p.src_off,
        p.src_stride,
        p.refb,
        p.ref_origin,
        p.ref_stride,
        p.w,
        p.h,
        p.ref_mv.0,
        p.ref_mv.1,
        p.dv,
        p.error_per_bit,
        p.sad_per_bit,
        full_limits,
        p.step_param,
    );
    if bestsme >= i64::from(i32::MAX) {
        return (p.ref_mv, 0, false);
    }
    if p.force_integer_mv {
        let mv = (brow * 8, bcol * 8);
        let rate = mv_bit_cost(
            mv,
            p.ref_mv,
            &p.dv.joint_mv,
            &p.dv.dv_costs[0],
            &p.dv.dv_costs[1],
            MV_COST_WEIGHT,
        );
        return (mv, rate, true);
    }
    // set_subpel_mv_search_range(subpel, x->mv_limits, ref_mv).
    const MAX_FULL_PEL_VAL: i32 = (1 << 10) - 1;
    const MV_LOW: i32 = -(1 << 14);
    const MV_UPP: i32 = 1 << 14;
    let max_mv = MAX_FULL_PEL_VAL * 8;
    let minc = (p.mv_limits.col_min * 8).max(p.ref_mv.1 - max_mv);
    let maxc = ((p.mv_limits.col_max * 8).min(p.ref_mv.1 + max_mv)).max(minc);
    let minr = (p.mv_limits.row_min * 8).max(p.ref_mv.0 - max_mv);
    let maxr = ((p.mv_limits.row_max * 8).min(p.ref_mv.0 + max_mv)).max(minr);
    let subpel_limits = SubpelMvLimits {
        col_min: (MV_LOW + 1).max(minc),
        col_max: (MV_UPP - 1).min(maxc),
        row_min: (MV_LOW + 1).max(minr),
        row_max: (MV_UPP - 1).min(maxr),
    };
    let sr = find_best_sub_pixel_tree(&SubpelSearchParams {
        src: p.src,
        src_off: p.src_off,
        src_stride: p.src_stride,
        refb: p.refb,
        ref_origin: p.ref_origin,
        ref_stride: p.ref_stride,
        w: p.w,
        h: p.h,
        start_mv: (brow * 8, bcol * 8),
        ref_mv: p.ref_mv,
        mvjcost: p.dv.joint_mv,
        mvcost0: &p.dv.dv_costs[0],
        mvcost1: &p.dv.dv_costs[1],
        error_per_bit: p.error_per_bit,
        allow_hp: p.allow_hp,
        forced_stop: p.forced_stop,
        iters_per_step: p.iters_per_step,
        limits: subpel_limits,
    });
    let rate = mv_bit_cost(
        sr.best_mv,
        p.ref_mv,
        &p.dv.joint_mv,
        &p.dv.dv_costs[0],
        &p.dv.dv_costs[1],
        MV_COST_WEIGHT,
    );
    (sr.best_mv, rate, true)
}

/// The glue reproduces the hand-composed pipeline across sizes, ref-MVs, step
/// params, precision, and the `force_integer_mv` short circuit.
#[test]
fn single_motion_search_matches_manual_composition() {
    let sizes = [(8usize, 8usize), (16, 16), (16, 8), (32, 32), (16, 32)];
    let dv = default_dv_costs();
    let mut rng = Rng::new(0x51A9_3ED0_2026);
    let limits = FullMvLimits {
        col_min: -48,
        col_max: 48,
        row_min: -48,
        row_max: 48,
    };
    let mut n = 0;
    for &(w, h) in &sizes {
        let (refb, origin, stride) = ref_plane(&mut rng, w, h);
        for _ in 0..8 {
            // A mix of random and converging (shifted) source content.
            let dy = rng.range(-16, 16);
            let dx = rng.range(-16, 16);
            let mut src = vec![0u16; w * h];
            for i in 0..h {
                for j in 0..w {
                    let p = (origin as i64
                        + (dy + i as i32) as i64 * stride as i64
                        + (dx + j as i32) as i64) as usize;
                    src[i * w + j] = refb[p];
                }
            }
            for &ref_mv in &[(0, 0), (8, -8), (5, 3)] {
                for &sp in &[0usize, 3, 6] {
                    for &force_int in &[false, true] {
                        let p = SingleMotionSearchParams {
                            src: &src,
                            src_off: 0,
                            src_stride: w,
                            refb: &refb,
                            ref_origin: origin,
                            ref_stride: stride,
                            w,
                            h,
                            ref_mv,
                            dv: &dv,
                            error_per_bit: 180,
                            sad_per_bit: 12,
                            mv_limits: limits,
                            step_param: sp,
                            allow_hp: true,
                            force_integer_mv: force_int,
                            forced_stop: 0,
                            iters_per_step: 2,
                        };
                        let got = single_motion_search(&p);
                        let (want_mv, want_rate, want_valid) = manual_compose(&p);
                        assert_eq!(
                            got.valid, want_valid,
                            "valid w{w}xh{h} ref_mv{ref_mv:?} sp{sp} fi{force_int}"
                        );
                        assert_eq!(
                            got.best_mv, want_mv,
                            "best_mv w{w}xh{h} ref_mv{ref_mv:?} sp{sp} fi{force_int}"
                        );
                        assert_eq!(
                            got.rate_mv, want_rate,
                            "rate_mv w{w}xh{h} ref_mv{ref_mv:?} sp{sp} fi{force_int}"
                        );
                        n += 1;
                    }
                }
            }
        }
    }
    assert!(n >= 200, "expected a broad composition sweep, ran {n}");
}

/// A smooth L1-cone reference (`(w+2B)×(h+2B)`), unimodal so the local NSTEP
/// diamond reliably descends to the true shift. `val` peaks at the plane centre.
fn cone_plane(w: usize, h: usize) -> (Vec<u16>, usize, usize) {
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let cy = (rows / 2) as i32;
    let cx = (stride / 2) as i32;
    let mut b = vec![0u16; stride * rows];
    for y in 0..rows {
        for x in 0..stride {
            let d = (y as i32 - cy).abs() + (x as i32 - cx).abs();
            b[y * stride + x] = (255 - d.min(255)) as u16;
        }
    }
    (b, BORDER * stride + BORDER, stride)
}

/// Convergence: on translational content (`src` = the reference shifted by an
/// integer MV) with a smooth unimodal (L1-cone) reference, the search converges
/// to that exact MV. The zero-shift case (the byte-exact P target: `frame1 ==
/// frame0`) is guaranteed regardless of content — the search starts at ref_mv
/// with variance 0 and cannot improve.
#[test]
fn single_motion_search_converges_to_true_shift() {
    let sizes = [(16usize, 16usize), (32, 32), (16, 32)];
    let dv = default_dv_costs();
    let limits = FullMvLimits {
        col_min: -48,
        col_max: 48,
        row_min: -48,
        row_max: 48,
    };
    let mut n = 0;
    for &(w, h) in &sizes {
        let (refb, origin, stride) = cone_plane(w, h);
        // Zero shift + small integer shifts on the smooth unimodal basin.
        for &(dy, dx) in &[(0, 0), (1, 0), (0, -1), (2, 3), (-3, 2), (-4, -5)] {
            let mut src = vec![0u16; w * h];
            for i in 0..h {
                for j in 0..w {
                    let p = (origin as i64
                        + (dy + i as i32) as i64 * stride as i64
                        + (dx + j as i32) as i64) as usize;
                    src[i * w + j] = refb[p];
                }
            }
            let p = SingleMotionSearchParams {
                src: &src,
                src_off: 0,
                src_stride: w,
                refb: &refb,
                ref_origin: origin,
                ref_stride: stride,
                w,
                h,
                ref_mv: (0, 0),
                dv: &dv,
                error_per_bit: 180,
                sad_per_bit: 12,
                mv_limits: limits,
                step_param: 0,
                allow_hp: true,
                force_integer_mv: false,
                forced_stop: 0,
                iters_per_step: 2,
            };
            let got = single_motion_search(&p);
            assert!(got.valid, "search should succeed w{w}xh{h} dy{dy} dx{dx}");
            // Exact match at the integer MV (dy, dx) → the zero-variance point;
            // subpel keeps the integer (a fractional phase filters the pixels →
            // nonzero variance). MV is 1/8-pel.
            assert_eq!(
                got.best_mv,
                (dy * 8, dx * 8),
                "converged MV w{w}xh{h} dy{dy} dx{dx} (got {:?})",
                got.best_mv
            );
            assert_eq!(got.distortion, 0, "zero distortion at exact match");
            n += 1;
        }
    }
    assert!(n >= 18, "ran {n}");
}
