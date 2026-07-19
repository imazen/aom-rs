//! INTER-ENCODE chunk 2 differential — the full-pixel motion search
//! (`av1_full_pixel_search`, mcomp.c:1768) retargeted from the current frame
//! (intrabc) to a reference frame, vs the REAL exported C.
//!
//! Locks [`aom_encode::intrabc_search::full_pixel_search_inter`] — the intrabc
//! NSTEP diamond with the stride split (separate src/ref planes) + the real
//! inter MV cost tables ([`fill_nmv_costs`]) — against `av1_full_pixel_search`
//! for the inter SIMPLE_TRANSLATION speed-0 path with the mesh disabled. Both
//! sides get the SAME source/reference pixels (u8 for C, the identical 8-bit
//! values as u16 for the port — SAD/variance are exact), the SAME centred MV
//! cost tables, error/sad-per-bit, mv_limits and step_param. The `(var_cost,
//! best_row, best_col)` triple must match across a sweep of block sizes, ref MVs
//! (integer AND subpel), step params, and both converging (src = a shifted ref)
//! and arbitrary (random) content.
//!
//! This is the FIRST real-C validation of the port's full-pel diamond (it was
//! previously geometry-unit-locked only), so it hardens the shared intrabc/inter
//! search path as well.

use aom_encode::intrabc_search::{fill_nmv_costs, full_pixel_search_inter, FullMvLimits, MV_SUBPEL_HIGH};
use aom_entropy::default_cdfs::{DEFAULT_NMV_COMPS, DEFAULT_NMV_JOINTS};
use aom_sys_ref::ref_full_pixel_search;

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
/// origin at (BORDER, BORDER). Returns `(u8_buf, u16_buf, origin_off, stride)`.
fn ref_plane(rng: &mut Rng, w: usize, h: usize) -> (Vec<u8>, Vec<u16>, usize, usize) {
    let stride = w + 2 * BORDER;
    let rows = h + 2 * BORDER;
    let mut u8b = vec![0u8; stride * rows];
    for b in u8b.iter_mut() {
        *b = rng.byte();
    }
    let u16b: Vec<u16> = u8b.iter().map(|&x| x as u16).collect();
    let origin = BORDER * stride + BORDER;
    (u8b, u16b, origin, stride)
}

/// Compare the port and C full-pel search for one configuration. `src8`/`src16`
/// are the tight `w×h` source block (stride = w).
#[allow(clippy::too_many_arguments)]
fn run_case(
    tag: &str,
    src8: &[u8],
    src16: &[u16],
    w: usize,
    h: usize,
    refu8: &[u8],
    refu16: &[u16],
    ref_origin: usize,
    ref_stride: usize,
    ref_mv: (i32, i32),
    epb: i32,
    spb: i32,
    step_param: usize,
    limits: FullMvLimits,
) {
    let dv = fill_nmv_costs(
        MV_SUBPEL_HIGH,
        &DEFAULT_NMV_JOINTS,
        &DEFAULT_NMV_COMPS[0],
        &DEFAULT_NMV_COMPS[1],
    );

    let (pvar, pr, pc) = full_pixel_search_inter(
        src16,
        0,
        w,
        refu16,
        ref_origin,
        ref_stride,
        w,
        h,
        ref_mv.0,
        ref_mv.1,
        &dv,
        epb,
        spb,
        limits,
        step_param,
    );

    let (cvar, cr, cc) = ref_full_pixel_search(
        src8,
        w as i32,
        refu8,
        ref_origin,
        ref_stride as i32,
        w as i32,
        h as i32,
        ref_mv,
        &dv.joint_mv,
        &dv.dv_costs[0],
        &dv.dv_costs[1],
        epb,
        spb,
        step_param as i32,
        (limits.row_min, limits.row_max, limits.col_min, limits.col_max),
    );

    assert_eq!(
        (pr, pc),
        (cr, cc),
        "{tag}: best MV differs — port ({pr},{pc}) var {pvar} vs C ({cr},{cc}) var {cvar}"
    );
    assert_eq!(
        pvar, cvar as i64,
        "{tag}: var cost differs at MV ({pr},{pc}) — port {pvar} vs C {cvar}"
    );
}

/// Random (arbitrary) content across block sizes, ref MVs, step params.
#[test]
fn full_pixel_search_random_content_matches_real_c() {
    let sizes = [
        (8usize, 8usize),
        (16, 16),
        (8, 16),
        (16, 8),
        (32, 32),
        (16, 32),
        (32, 16),
        (64, 64),
    ];
    let mut rng = Rng::new(0xF017_9EED_2026);
    let limits = FullMvLimits {
        col_min: -40,
        col_max: 40,
        row_min: -40,
        row_max: 40,
    };
    let mut n = 0;
    for &(w, h) in &sizes {
        let (refu8, refu16, origin, stride) = ref_plane(&mut rng, w, h);
        for _ in 0..6 {
            // random source block
            let src8: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
            let src16: Vec<u16> = src8.iter().map(|&x| x as u16).collect();
            // ref MV with subpel bits (exercises get_fullmv rounding + centring)
            let ref_mv = (rng.range(-80, 80), rng.range(-80, 80));
            for &sp in &[0usize, 2, 4, 6] {
                let epb = rng.range(1, 400);
                let spb = rng.range(1, 40);
                run_case(
                    &format!("rand w{w}xh{h} sp{sp}"),
                    &src8,
                    &src16,
                    w,
                    h,
                    &refu8,
                    &refu16,
                    origin,
                    stride,
                    ref_mv,
                    epb,
                    spb,
                    sp,
                    limits,
                );
                n += 1;
            }
        }
    }
    assert!(n >= 100, "expected a broad sweep, ran {n}");
}

/// Converging content: `src` is the reference shifted by a known integer MV, so
/// the search moves toward it. Exercises the diamond actually walking (not just
/// staying at center) — the non-trivial path where port/C tie-breaking matters.
#[test]
fn full_pixel_search_converging_content_matches_real_c() {
    let sizes = [(8usize, 8usize), (16, 16), (16, 8), (32, 32), (64, 64)];
    let mut rng = Rng::new(0x0C0A_1CE5_0E42);
    let limits = FullMvLimits {
        col_min: -48,
        col_max: 48,
        row_min: -48,
        row_max: 48,
    };
    let mut n = 0;
    for &(w, h) in &sizes {
        let (refu8, refu16, origin, stride) = ref_plane(&mut rng, w, h);
        for _ in 0..8 {
            // true shift within the limits; the perfect match is at this MV.
            let dy = rng.range(-20, 20);
            let dx = rng.range(-20, 20);
            // src[i,j] = ref at MV (dy, dx)
            let mut src8 = vec![0u8; w * h];
            for i in 0..h {
                for j in 0..w {
                    let p = (origin as i64 + (dy + i as i32) as i64 * stride as i64 + (dx + j as i32) as i64) as usize;
                    src8[i * w + j] = refu8[p];
                }
            }
            let src16: Vec<u16> = src8.iter().map(|&x| x as u16).collect();
            // ref MV near zero (start of search); include subpel variants.
            for &ref_mv in &[(0, 0), (8, -8), (3, 5), (-13, 21)] {
                for &sp in &[0usize, 3, 6] {
                    run_case(
                        &format!("conv w{w}xh{h} dy{dy} dx{dx} sp{sp}"),
                        &src8,
                        &src16,
                        w,
                        h,
                        &refu8,
                        &refu16,
                        origin,
                        stride,
                        ref_mv,
                        180,
                        12,
                        sp,
                        limits,
                    );
                    n += 1;
                }
            }
        }
    }
    assert!(n >= 100, "expected a broad sweep, ran {n}");
}
