//! SIMD deblock loop-filter kernels (Gate 3) — bit-identical to the highbd
//! scalar core, at every dispatch tier (`tests/lpf_simd_diff.rs`).
//!
//! Same aom-rs SIMD pattern as `aom_cdef` / `aom_txb`: ONE magetypes generic
//! kernel (`#[magetypes(define(i32x4), v3, neon, wasm128, -scalar)]`), a
//! hand-written `_scalar` tier that calls the untouched highbd transcription,
//! `incant!` dispatch, `aom_dispatch::scalar_forced()` pin at the entry.
//!
//! # Layout
//!
//! Each `aom_(highbd_)lpf_*` call filters **4 edge positions** — AV1's 4-px
//! edge segment (`aom_dsp/loopfilter.c`, the `for (i = 0; i < 4; ++i)` loops).
//! Those 4 positions are the 4 SIMD lanes. Tap `k` for lane `l` lives at
//! `center + l*step + k*ts`, where `ts` = tap stride and `step` = position
//! advance (horizontal: `ts` = pitch, `step` = 1; vertical: `ts` = 1,
//! `step` = pitch). Taps are gathered per lane into an `i32x4`; the filter
//! math runs once across the 4 lanes instead of the scalar core's 4 sequential
//! iterations.
//!
//! # Branchless width selection
//!
//! The scalar core branches per position: `filter6`/`filter8` apply the wide
//! (flat-region) filter iff `flat && mask`, else fall back to `filter4`; those
//! branches differ per lane. This kernel computes BOTH the `filter4` result and
//! the wide result for all 4 lanes and blends per lane on the `flat & mask`
//! lane mask (`i32x4::blend`) — the standard libaom SIMD structure. `filter4`
//! itself is already branchless (its `& mask` / `& hev` gates zero out the
//! contribution for unfiltered lanes, reproducing the scalar identity when
//! `mask == 0`). Taps that only the wide filter writes (`p2`/`q2` in the 8-tap
//! filter) blend against the ORIGINAL sample, matching the scalar core leaving
//! them untouched on the `filter4` fallback.
//!
//! # Bit-exactness (full highbd domain, bd 8/10/12)
//!
//! The highbd scalar core does the `filter4` math in `i16` (`filter`, `hev`,
//! `filter1/2`, `f`) and the WIDE (6/8/14-tap) sums in `i32`. This kernel runs
//! EVERYTHING in `i32` lanes, which reproduces the scalar result lane-for-lane:
//!
//! * `scc` (`signed_char_clamp_high`) clamps to `[-(128<<sh), (128<<sh)-1]`
//!   ⊆ `[-2048, 2047]` (bd ≤ 12) — always inside `i16`, so the `i32` clamp and
//!   the scalar `i16` clamp produce the same number.
//! * The `& hev` / `& mask` / `& !hev` gates are `0`/`-1` lane masks; ANDing a
//!   value already in the `i16` range with an all-ones/all-zero `i32` mask
//!   yields the same value the scalar `i16` AND does.
//! * `>> 3` / `>> 1` are arithmetic shifts on values in the `i16` range; the
//!   wide-sum rounding (`rpo2`) shifts a NON-NEGATIVE sum, so `>> n` is a plain
//!   unsigned divide. `i32` and `i16`/scalar agree in both cases.
//! * `iabs(a,b) = |a-b|` with `a,b` ∈ `[0, 4095]`, so `.abs()` never hits the
//!   `i32::MIN` corner; `iabs/2` is `>> 1` on a non-negative value.
//! * The wide 14-tap weighted sums reach ~`4095*16 = 65520` at bd 12 — beyond
//!   `i16`, which is exactly why the accumulation is `i32` here (and in the
//!   scalar core). Comparisons (`filter_mask*` / `flat_mask*` / `hev_mask`)
//!   use `simd_gt`, matching the scalar `iabs > threshold`.
//!
//! `bd == 8` runs this same path with `sh = 0` (the decoder keeps `u16`
//! samples at every bit depth); the differential proves it against the REAL C
//! lowbd kernels via `hbd_lpf_diff.rs` (dispatch-vs-C) plus the SIMD-vs-scalar
//! `lpf_simd_diff.rs` at every token tier.

use archmage::prelude::*;

/// Dispatch entry for one 4-position deblock edge segment (highbd `u16` path).
/// `ts` = tap stride, `step` = position advance; the axis is encoded by the
/// caller ([`crate::highbd::horizontal`] / [`crate::highbd::vertical`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lpf(
    width: u32,
    buf: &mut [u16],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
) {
    let _ = aom_dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        lpf_impl(width, buf, center, ts, step, bl, li, th, bd),
        [v3, neon, wasm128, scalar]
    )
}

/// Scalar tier = the untouched highbd transcription, verbatim.
#[allow(clippy::too_many_arguments)]
fn lpf_impl_scalar(
    _t: archmage::ScalarToken,
    width: u32,
    buf: &mut [u16],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
) {
    crate::highbd::lpf_scalar(width, buf, center, ts, step, bl, li, th, bd);
}

#[magetypes(define(i32x4), v3, neon, wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn lpf_impl(
    token: Token,
    width: u32,
    buf: &mut [u16],
    center: usize,
    ts: isize,
    step: isize,
    bl: u8,
    li: u8,
    th: u8,
    bd: i32,
) {
    // Widths not yet vectorized fall through to the scalar transcription.
    if width == 14 {
        crate::highbd::lpf_scalar(width, buf, center, ts, step, bl, li, th, bd);
        return;
    }

    let shift = bd - 8;
    let bias: i32 = 0x80 << shift;
    let lim: i32 = 128 << shift;
    let neg_lim = i32x4::splat(token, -lim);
    let lim_hi = i32x4::splat(token, lim - 1);
    let l = i32x4::splat(token, (li as i32) << shift); // limit
    let blv = i32x4::splat(token, (bl as i32) << shift); // blimit
    let ft = i32x4::splat(token, 1 << shift); // flat_mask thresh (== 1)

    // signed_char_clamp_high
    let scc = |v: i32x4| v.clamp(neg_lim, lim_hi);
    // |a - b| (a,b are u16 pixels widened to i32, so abs() is exact)
    let iabs = |a: i32x4, b: i32x4| (a - b).abs();
    // rpo2(v, 3): round-power-of-two on a non-negative sum
    let rpo3 = |v: i32x4| (v + 4).shr_logical_const::<3>();

    let c = center as isize;
    // Gather tap `k` (offset k*ts from center) across the 4 edge positions
    // (offset l*step, l in 0..4) into one i32x4 lane vector.
    let load = |k: isize| -> i32x4 {
        i32x4::from_array(
            token,
            [
                buf[(c + k * ts) as usize] as i32,
                buf[(c + step + k * ts) as usize] as i32,
                buf[(c + 2 * step + k * ts) as usize] as i32,
                buf[(c + 3 * step + k * ts) as usize] as i32,
            ],
        )
    };

    // filter4: taps p1(-2) p0(-1) q0(0) q1(1); `mask` is the filter_mask lane
    // mask (all-ones = filter). Returns (op1', op0', oq0', oq1').
    let filter4 = |op1: i32x4,
                   op0: i32x4,
                   oq0: i32x4,
                   oq1: i32x4,
                   mask: i32x4|
     -> (i32x4, i32x4, i32x4, i32x4) {
        let ps1 = op1 - bias;
        let ps0 = op0 - bias;
        let qs0 = oq0 - bias;
        let qs1 = oq1 - bias;
        let t_hev = i32x4::splat(token, (th as i32) << shift);
        let hev = iabs(op1, op0).simd_gt(t_hev) | iabs(oq1, oq0).simd_gt(t_hev);

        let mut filter = scc(ps1 - qs1) & hev;
        filter = scc(filter + (qs0 - ps0) * 3) & mask;
        let filter1 = scc(filter + 4).shr_arithmetic_const::<3>();
        let filter2 = scc(filter + 3).shr_arithmetic_const::<3>();
        let n_oq0 = scc(qs0 - filter1) + bias;
        let n_op0 = scc(ps0 + filter2) + bias;
        let f = ((filter1 + 1).shr_arithmetic_const::<1>()) & hev.not();
        let n_oq1 = scc(qs1 - f) + bias;
        let n_op1 = scc(ps1 + f) + bias;
        (n_op1, n_op0, n_oq0, n_oq1)
    };

    // filter_mask2(limit, blimit, p1, p0, q0, q1) -> filter mask (post-NOT)
    let fmask2 = |p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4| -> i32x4 {
        (iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    // filter_mask3_chroma(limit, blimit, p2,p1,p0,q0,q1,q2)
    let fmask6 = |p2: i32x4, p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4, q2: i32x4| -> i32x4 {
        (iabs(p2, p1).simd_gt(l)
            | iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | iabs(q2, q1).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    // filter_mask(limit, blimit, p3,p2,p1,p0,q0,q1,q2,q3)
    let fmask8 = |p3: i32x4,
                  p2: i32x4,
                  p1: i32x4,
                  p0: i32x4,
                  q0: i32x4,
                  q1: i32x4,
                  q2: i32x4,
                  q3: i32x4|
     -> i32x4 {
        (iabs(p3, p2).simd_gt(l)
            | iabs(p2, p1).simd_gt(l)
            | iabs(p1, p0).simd_gt(l)
            | iabs(q1, q0).simd_gt(l)
            | iabs(q2, q1).simd_gt(l)
            | iabs(q3, q2).simd_gt(l)
            | (iabs(p0, q0) * 2 + iabs(p1, q1).shr_logical_const::<1>()).simd_gt(blv))
        .not()
    };
    // flat_mask3_chroma(1, p2,p1,p0,q0,q1,q2)
    let flat3 = |p2: i32x4, p1: i32x4, p0: i32x4, q0: i32x4, q1: i32x4, q2: i32x4| -> i32x4 {
        (iabs(p1, p0).simd_gt(ft)
            | iabs(q1, q0).simd_gt(ft)
            | iabs(p2, p0).simd_gt(ft)
            | iabs(q2, q0).simd_gt(ft))
        .not()
    };
    // flat_mask4(1, p3,p2,p1,p0,q0,q1,q2,q3)
    let flat4 = |p3: i32x4,
                 p2: i32x4,
                 p1: i32x4,
                 p0: i32x4,
                 q0: i32x4,
                 q1: i32x4,
                 q2: i32x4,
                 q3: i32x4|
     -> i32x4 {
        (iabs(p1, p0).simd_gt(ft)
            | iabs(q1, q0).simd_gt(ft)
            | iabs(p2, p0).simd_gt(ft)
            | iabs(q2, q0).simd_gt(ft)
            | iabs(p3, p0).simd_gt(ft)
            | iabs(q3, q0).simd_gt(ft))
        .not()
    };

    // Scatter helper values into `buf` for taps `ks` — direct indexing so the
    // (immutable) `load` closure's borrow has ended by this point.
    macro_rules! store {
        ($($k:expr => $v:expr),+ $(,)?) => {{
            $(
                let a = ($v).to_array();
                buf[(c + ($k) * ts) as usize] = a[0] as u16;
                buf[(c + step + ($k) * ts) as usize] = a[1] as u16;
                buf[(c + 2 * step + ($k) * ts) as usize] = a[2] as u16;
                buf[(c + 3 * step + ($k) * ts) as usize] = a[3] as u16;
            )+
        }};
    }

    match width {
        4 => {
            // taps p1(-2) p0(-1) q0(0) q1(1)
            let op1 = load(-2);
            let op0 = load(-1);
            let oq0 = load(0);
            let oq1 = load(1);
            let mask = fmask2(op1, op0, oq0, oq1);
            let (n1, n0, m0, m1) = filter4(op1, op0, oq0, oq1, mask);
            store!(-2 => n1, -1 => n0, 0 => m0, 1 => m1);
        }
        6 => {
            // taps p2(-3) p1(-2) p0(-1) q0(0) q1(1) q2(2)
            let p2 = load(-3);
            let p1 = load(-2);
            let p0 = load(-1);
            let q0 = load(0);
            let q1 = load(1);
            let q2 = load(2);
            let mask = fmask6(p2, p1, p0, q0, q1, q2);
            let flat = flat3(p2, p1, p0, q0, q1, q2);
            let use_wide = flat & mask;
            // wide 6-tap (writes p1,p0,q0,q1)
            let w_p1 = rpo3(p2 * 3 + p1 * 2 + p0 * 2 + q0);
            let w_p0 = rpo3(p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1);
            let w_q0 = rpo3(p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2);
            let w_q1 = rpo3(p0 + q0 * 2 + q1 * 2 + q2 * 3);
            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            let o_p1 = i32x4::blend(use_wide, w_p1, f_p1);
            let o_p0 = i32x4::blend(use_wide, w_p0, f_p0);
            let o_q0 = i32x4::blend(use_wide, w_q0, f_q0);
            let o_q1 = i32x4::blend(use_wide, w_q1, f_q1);
            store!(-2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1);
        }
        8 => {
            // taps p3(-4) p2(-3) p1(-2) p0(-1) q0(0) q1(1) q2(2) q3(3)
            let p3 = load(-4);
            let p2 = load(-3);
            let p1 = load(-2);
            let p0 = load(-1);
            let q0 = load(0);
            let q1 = load(1);
            let q2 = load(2);
            let q3 = load(3);
            let mask = fmask8(p3, p2, p1, p0, q0, q1, q2, q3);
            let flat = flat4(p3, p2, p1, p0, q0, q1, q2, q3);
            let use_wide = flat & mask;
            // wide 8-tap (writes p2,p1,p0,q0,q1,q2)
            let w_p2 = rpo3(p3 * 3 + p2 * 2 + p1 + p0 + q0);
            let w_p1 = rpo3(p3 * 2 + p2 + p1 * 2 + p0 + q0 + q1);
            let w_p0 = rpo3(p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2);
            let w_q0 = rpo3(p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3);
            let w_q1 = rpo3(p1 + p0 + q0 + q1 * 2 + q2 + q3 * 2);
            let w_q2 = rpo3(p0 + q0 + q1 + q2 * 2 + q3 * 3);
            let (f_p1, f_p0, f_q0, f_q1) = filter4(p1, p0, q0, q1, mask);
            // p2/q2 fall back to the ORIGINAL sample (filter4 leaves them).
            let o_p2 = i32x4::blend(use_wide, w_p2, p2);
            let o_p1 = i32x4::blend(use_wide, w_p1, f_p1);
            let o_p0 = i32x4::blend(use_wide, w_p0, f_p0);
            let o_q0 = i32x4::blend(use_wide, w_q0, f_q0);
            let o_q1 = i32x4::blend(use_wide, w_q1, f_q1);
            let o_q2 = i32x4::blend(use_wide, w_q2, q2);
            store!(-3 => o_p2, -2 => o_p1, -1 => o_p0, 0 => o_q0, 1 => o_q1, 2 => o_q2);
        }
        _ => crate::highbd::lpf_scalar(width, buf, center, ts, step, bl, li, th, bd),
    }
}
