//! INTER-ENCODE chunk 2d differential — `av1_find_best_sub_pixel_tree` (lowbd,
//! USE_8_TAPS, the speed-0 allintra/GOOD subpel search) vs the REAL exported C.
//!
//! Locks the port's [`aom_encode::inter_me::find_best_sub_pixel_tree`]
//! byte-for-byte against `av1_find_best_sub_pixel_tree` (mcomp.c:3266): the
//! refined `bestmv`, the `distortion`, the `sse`, and the function's `besterr`
//! return value all match, across every subpel-stop / precision / iters-per-step
//! knob, a sweep of block sizes, and both converging (src = subpel-shifted ref)
//! and arbitrary (random src) content. The oracle drives the real tree over a
//! minimal MACROBLOCKD; the same caller-supplied MV cost tables and
//! `aom_variance{W}x{H}_c` feed both sides.

use aom_encode::inter_me::{
    find_best_sub_pixel_tree, upsampled_pred, MV_MAX, SubpelMvLimits, SubpelSearchParams,
};
use aom_sys_ref::ref_find_best_sub_pixel_tree;

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
}

/// Full per-component MV cost table (length `2*MV_MAX+1`, cost of value `v` at
/// index `MV_MAX + v`). A monotone, plausible bit-cost model; the exact values
/// are irrelevant to the differential (both sides use the same table) — only
/// that they vary with `|v|` and are in a non-overflowing range.
fn mvcost_table() -> Vec<i32> {
    let n = (2 * MV_MAX + 1) as usize;
    let mut t = vec![0i32; n];
    for (i, e) in t.iter_mut().enumerate() {
        let v = i as i32 - MV_MAX;
        *e = v.abs() * 48 + 96;
    }
    t
}

const BORDER: usize = 24;

/// A reference plane `(w+2*BORDER)×(h+2*BORDER)` (u8, random) with the buf_2d
/// origin (MV 0) at (BORDER, BORDER). Returns `(buf, ref_origin, ref_stride)`.
fn ref_plane(rng: &mut Rng, w: usize, h: usize) -> (Vec<u8>, usize, usize) {
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let mut buf = vec![0u8; stride * rows];
    for b in buf.iter_mut() {
        *b = rng.byte();
    }
    (buf, BORDER * stride + BORDER, stride)
}

#[allow(clippy::too_many_arguments)]
fn one_case(
    rng: &mut Rng,
    w: usize,
    h: usize,
    start_mv: (i32, i32),
    ref_mv: (i32, i32),
    error_per_bit: i32,
    allow_hp: bool,
    forced_stop: i32,
    iters_per_step: i32,
    src_from_ref_subpel: Option<(usize, usize)>,
) {
    let (ref8, ref_origin, ref_stride) = ref_plane(rng, w, h);
    let ref16: Vec<u16> = ref8.iter().map(|&b| b as u16).collect();

    // Source block: either a subpel-shifted crop of the reference (so a nonzero
    // subpel MV genuinely wins and the tree traverses), or independent random.
    let src8: Vec<u8> = match src_from_ref_subpel {
        Some((sx, sy)) => {
            // Crop at the fullpel start position, shifted by (sx, sy) 1/8-pel.
            let base = (ref_origin as isize
                + (start_mv.0 >> 3) as isize * ref_stride as isize
                + (start_mv.1 >> 3) as isize) as usize;
            upsampled_pred(&ref16, base, ref_stride, w, h, sx, sy)
                .iter()
                .map(|&v| v as u8)
                .collect()
        }
        None => (0..w * h).map(|_| rng.byte()).collect(),
    };
    let src16: Vec<u16> = src8.iter().map(|&b| b as u16).collect();

    let mvcost0 = mvcost_table();
    let mvcost1 = mvcost_table();
    let mvjcost = [0i32, 240, 240, 480];
    let limits = (-4096, 4096, -4096, 4096);

    let got = find_best_sub_pixel_tree(&SubpelSearchParams {
        src: &src16,
        src_off: 0,
        src_stride: w,
        refb: &ref16,
        ref_origin,
        ref_stride,
        w,
        h,
        start_mv,
        ref_mv,
        mvjcost,
        mvcost0: &mvcost0,
        mvcost1: &mvcost1,
        error_per_bit,
        allow_hp,
        forced_stop,
        iters_per_step,
        limits: SubpelMvLimits {
            row_min: limits.0,
            row_max: limits.1,
            col_min: limits.2,
            col_max: limits.3,
        },
    });

    let want = ref_find_best_sub_pixel_tree(
        &src8, w, &ref8, ref_origin, ref_stride, w, h, start_mv, ref_mv, &mvjcost, &mvcost0,
        &mvcost1, error_per_bit, allow_hp, forced_stop, iters_per_step, limits,
    );

    let label = format!(
        "w={w} h={h} start={start_mv:?} ref_mv={ref_mv:?} epb={error_per_bit} hp={allow_hp} \
         stop={forced_stop} iters={iters_per_step} srcsub={src_from_ref_subpel:?}"
    );
    assert_eq!(got.best_mv, want.best_mv, "best_mv: {label}");
    assert_eq!(got.distortion, want.distortion, "distortion: {label}");
    assert_eq!(got.sse, want.sse, "sse: {label}");
    assert_eq!(got.besterr, want.besterr, "besterr: {label}");
}

#[test]
fn subpel_tree_matches_real_c() {
    let mut rng = Rng::new(0x5AB9_E179_EEDD_1FF0 ^ 0xDEAD_BEEF);
    let sizes = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (8, 4),
        (16, 8),
        (8, 16),
        (16, 64),
    ];
    let starts = [(0, 0), (8, 0), (0, 8), (8, 8), (-8, -8), (16, -8)];
    for &(w, h) in &sizes {
        for &start in &starts {
            for &(allow_hp, forced_stop, iters) in
                &[(true, 0, 2), (false, 0, 2), (true, 2, 2), (true, 0, 1)]
            {
                // Converging case: src is a subpel-shifted reference crop.
                one_case(
                    &mut rng, w, h, start, start, 256, allow_hp, forced_stop, iters,
                    Some((3, 5)),
                );
                // Arbitrary case: independent random source, ref_mv away from start.
                one_case(
                    &mut rng, w, h, start, (0, 0), 384, allow_hp, forced_stop, iters, None,
                );
            }
        }
    }
}

#[test]
fn get_mvpred_sse_matches_real_c() {
    use aom_encode::inter_me::get_mvpred_sse;
    use aom_sys_ref::ref_get_mvpred_sse;

    let mut rng = Rng::new(0x6E7D_9AED_5A5E_0F11 ^ 0x1234_5678);
    let mvcost0 = mvcost_table();
    let mvcost1 = mvcost_table();
    let mvjcost = [0i32, 240, 240, 480];
    let sizes = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (8, 4),
        (16, 8),
        (8, 16),
        (16, 64),
    ];
    // Full-pel MVs (the reference is offset by mv * stride, within BORDER).
    let mvs = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (2, -3), (-4, 5)];
    let ref_mv = (0, 0);
    for &(w, h) in &sizes {
        for &mv in &mvs {
            for &epb in &[128, 384] {
                let (pre8, pre_origin, pre_stride) = ref_plane(&mut rng, w, h);
                let pre16: Vec<u16> = pre8.iter().map(|&b| b as u16).collect();
                let src8: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
                let src16: Vec<u16> = src8.iter().map(|&b| b as u16).collect();

                let got = get_mvpred_sse(
                    mv, &src16, 0, w, &pre16, pre_origin, pre_stride, w, h, ref_mv, &mvjcost,
                    &mvcost0, &mvcost1, epb,
                );
                let want = ref_get_mvpred_sse(
                    mv, &src8, 0, w, &pre8, pre_origin, pre_stride, w, h, ref_mv, &mvjcost,
                    &mvcost0, &mvcost1, epb,
                );
                assert_eq!(got, want, "w={w} h={h} mv={mv:?} epb={epb}");
            }
        }
    }
}

#[test]
fn mv_bit_cost_matches_real_c() {
    use aom_encode::inter_me::mv_bit_cost;
    use aom_sys_ref::ref_mv_bit_cost;

    let mvcost0 = mvcost_table();
    let mvcost1 = mvcost_table();
    let mvjcost = [0i32, 240, 240, 480];
    // MV_COST_WEIGHT (108, RD rate) + MV_COST_WEIGHT_SUB (120, coded DV).
    let mut rng = Rng::new(0x11B_C057_ABCD_1234 ^ 0x5A5A_5A5A);
    for _ in 0..4000 {
        let mv = (
            (rng.next_u64() % 1024) as i32 - 512,
            (rng.next_u64() % 1024) as i32 - 512,
        );
        let ref_mv = (
            (rng.next_u64() % 256) as i32 - 128,
            (rng.next_u64() % 256) as i32 - 128,
        );
        for &weight in &[108, 120] {
            let got = mv_bit_cost(mv, ref_mv, &mvjcost, &mvcost0, &mvcost1, weight);
            let want = ref_mv_bit_cost(mv, ref_mv, &mvjcost, &mvcost0, &mvcost1, weight);
            assert_eq!(got, want, "mv={mv:?} ref={ref_mv:?} w={weight}");
        }
    }
}
