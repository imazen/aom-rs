//! Differential harness: [`aom_cdef::frame::cdef_frame`] vs the REAL
//! `av1_cdef_frame` (driven whole through `shim_cdef_frame` — the exported
//! single-threaded decoder walk incl. `av1_cdef_init_fb_row`,
//! `cdef_prepare_fb` border priming and `av1_cdef_filter_fb`).
//!
//! Sweep axes: frame shapes incl. non-multiple-of-64 and 8-aligned-crop
//! sizes, 4:2:0 / 4:2:2 / 4:4:4 / monochrome, bd 8/10/12 (bd 8 exercises the
//! REAL lowbd u8 path in C against our u16 store), damping 3..=6, random
//! per-slot strength grids incl. zero-Y / zero-UV / all-zero, per-64x64 unit
//! strength indices incl. the -1 (nothing-read) arm, and skip patterns from
//! all-skip to none-skip (mixed patterns produce skipped fbs mid-frame,
//! exercising the unfiltered-left-border `cstart = -CDEF_HBORDER` path).
//!
//! The two sides get DIFFERENT padding fills beyond the mi-aligned visible
//! width: byte-identity of the visible region then also PROVES the walk's
//! output never depends on the out-of-frame columns the line-buffer copies
//! read (in production C those are YV12 border bytes).

use aom_cdef::frame::{cdef_frame, CdefFrameParams};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn upto(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

fn align16(v: usize) -> usize {
    (v + 15) & !15
}

/// Padding fills (both <= 255 so the bd-8 u16->u8->u16 roundtrip on the C
/// side is lossless and the invariance check stays exact).
const RUST_PAD: u16 = 0xAB;
const C_PAD: u16 = 0x5C;

struct Case {
    w: usize,
    h: usize,
    bd: i32,
    num_planes: usize,
    ss: (usize, usize),
    damping: i32,
    strengths: [i32; 8],
    uv_strengths: [i32; 8],
    /// 0 = none skip, 1 = mixed p=1/2, 2 = heavy skip p=15/16, 3 = all skip.
    skip_kind: u32,
}

/// Returns (any_pixel_changed, any_fb_minus_one, any_fb_skipped_by_allskip).
fn run_case(rng: &mut Rng, cs: &Case) -> (bool, bool, bool) {
    let mi_cols = (((cs.w + 7) & !7) >> 2) as i32;
    let mi_rows = (((cs.h + 7) & !7) >> 2) as i32;
    let nvfb = (mi_rows as usize).div_ceil(16);
    let nhfb = (mi_cols as usize).div_ceil(16);
    let (ss_x, ss_y) = cs.ss;

    // Plane geometry: EXACT minimum stride the walk requires (stresses the
    // aligned-row line-buffer reads at the right edge).
    let yw = mi_cols as usize * 4;
    let yh = mi_rows as usize * 4;
    let y_stride = align16(yw);
    let (uvw, uvh, uv_stride) = if cs.num_planes > 1 {
        (yw >> ss_x, yh >> ss_y, y_stride >> ss_x)
    } else {
        (0, 0, 0)
    };

    let maxv = (1u32 << cs.bd) - 1;
    let mut mk_plane = |w: usize, h: usize, stride: usize, pad: u16| -> Vec<u16> {
        let mut p = vec![pad; stride * h];
        for r in 0..h {
            for col in 0..w {
                p[r * stride + col] = rng.upto(maxv + 1) as u16;
            }
        }
        p
    };
    let y0 = mk_plane(yw, yh, y_stride, RUST_PAD);
    let u0 = mk_plane(uvw, uvh, uv_stride, RUST_PAD);
    let v0 = mk_plane(uvw, uvh, uv_stride, RUST_PAD);

    // Per-mi skip pattern.
    let ncells = (mi_rows * mi_cols) as usize;
    let skip: Vec<bool> = (0..ncells)
        .map(|_| match cs.skip_kind {
            0 => false,
            1 => rng.upto(2) == 0,
            2 => rng.upto(16) != 0,
            _ => true,
        })
        .collect();
    let skip_i32: Vec<i32> = skip.iter().map(|&s| s as i32).collect();

    // Per-fb strength indices; ~1/8 get the -1 (nothing-read) arm.
    let unit_strength: Vec<i32> = (0..nvfb * nhfb)
        .map(|_| {
            if rng.upto(8) == 0 {
                -1
            } else {
                rng.upto(8) as i32
            }
        })
        .collect();
    let any_minus_one = unit_strength.contains(&-1);
    // An fb skipped purely by all-skip aggregation (mid-frame cstart=-8 arms).
    let mut any_allskip_fb = false;
    for fbr in 0..nvfb as i32 {
        'fb: for fbc in 0..nhfb as i32 {
            if unit_strength[fbr as usize * nhfb + fbc as usize] < 0 {
                continue;
            }
            let maxr = (mi_rows - fbr * 16).min(16);
            let maxc = (mi_cols - fbc * 16).min(16);
            for r in 0..maxr {
                for c2 in 0..maxc {
                    if !skip[((fbr * 16 + r) * mi_cols + fbc * 16 + c2) as usize] {
                        continue 'fb;
                    }
                }
            }
            any_allskip_fb = true;
        }
    }

    // Rust side.
    let (mut y_r, mut u_r, mut v_r) = (y0.clone(), u0.clone(), v0.clone());
    let params = CdefFrameParams {
        mi_rows,
        mi_cols,
        num_planes: cs.num_planes,
        ss_x,
        ss_y,
        bit_depth: cs.bd,
        damping: cs.damping,
        cdef_strengths: cs.strengths,
        cdef_uv_strengths: cs.uv_strengths,
        skip_txfm: &skip,
        unit_strength: &unit_strength,
    };
    cdef_frame(&mut y_r, y_stride, &mut u_r, &mut v_r, uv_stride, &params);

    // C side: same visible content, DIFFERENT padding.
    let repad = |src: &[u16], w: usize, h: usize, stride: usize| -> Vec<u16> {
        let mut p = vec![C_PAD; src.len()];
        for r in 0..h {
            p[r * stride..r * stride + w].copy_from_slice(&src[r * stride..r * stride + w]);
        }
        p
    };
    let mut y_c = repad(&y0, yw, yh, y_stride);
    let mut u_c = repad(&u0, uvw, uvh, uv_stride);
    let mut v_c = repad(&v0, uvw, uvh, uv_stride);
    c::ref_cdef_frame(
        &mut y_c,
        y_stride,
        &mut u_c,
        &mut v_c,
        uv_stride,
        mi_rows,
        mi_cols,
        cs.num_planes,
        ss_x,
        ss_y,
        cs.bd,
        cs.damping,
        &cs.strengths,
        &cs.uv_strengths,
        &skip_i32,
        &unit_strength,
    );

    // Visible-region byte identity + padding invariance on BOTH sides.
    let ctx = format!(
        "{}x{} bd{} planes={} ss={:?} damp={} skip_kind={} y={:?} uv={:?}",
        cs.w,
        cs.h,
        cs.bd,
        cs.num_planes,
        cs.ss,
        cs.damping,
        cs.skip_kind,
        cs.strengths,
        cs.uv_strengths
    );
    let mut changed = false;
    let mut check = |name: &str,
                     rust: &[u16],
                     cref: &[u16],
                     orig: &[u16],
                     w: usize,
                     h: usize,
                     stride: usize| {
        for r in 0..h {
            assert_eq!(
                &rust[r * stride..r * stride + w],
                &cref[r * stride..r * stride + w],
                "{name} row {r} mismatch [{ctx}]"
            );
            if rust[r * stride..r * stride + w] != orig[r * stride..r * stride + w] {
                changed = true;
            }
            // Padding: untouched on both sides (proves no out-of-frame writes
            // and, with the differing fills, no out-of-frame reads either).
            assert!(
                rust[r * stride + w..(r + 1) * stride]
                    .iter()
                    .all(|&x| x == RUST_PAD),
                "{name} rust padding touched row {r} [{ctx}]"
            );
            assert!(
                cref[r * stride + w..(r + 1) * stride]
                    .iter()
                    .all(|&x| x == C_PAD),
                "{name} C padding touched row {r} [{ctx}]"
            );
        }
    };
    check("Y", &y_r, &y_c, &y0, yw, yh, y_stride);
    if cs.num_planes > 1 {
        check("U", &u_r, &u_c, &u0, uvw, uvh, uv_stride);
        check("V", &v_r, &v_c, &v0, uvw, uvh, uv_stride);
    }
    (changed, any_minus_one, any_allskip_fb)
}

#[test]
fn cdef_frame_matches_real_c_walk() {
    let mut rng = Rng(0x_cdef_f2a3_e001_4d57);
    let sizes = [
        (64usize, 64usize),
        (128, 128),
        (96, 80),
        (100, 76),
        (192, 64),
        (64, 192),
        (24, 16),
        (176, 144),
    ];
    let modes = [
        (3usize, (1usize, 1usize)), // 4:2:0
        (3, (0, 0)),                // 4:4:4
        (3, (1, 0)),                // 4:2:2
        (1, (1, 1)),                // monochrome
    ];
    let mut n = 0u32;
    let mut n_changed = 0u32;
    let mut n_minus_one = 0u32;
    let mut n_allskip_fb = 0u32;
    let mut skip_kind_seen = [0u32; 4];
    let mk_strengths = |rng: &mut Rng, force_zero: bool| -> [i32; 8] {
        core::array::from_fn(|_| {
            if force_zero || rng.upto(4) == 0 {
                0
            } else {
                rng.upto(64) as i32
            }
        })
    };
    for iter in 0..420 {
        let (w, h) = sizes[(iter % sizes.len() as u32) as usize];
        let (num_planes, ss) = modes[rng.upto(4) as usize];
        let bd = [8, 10, 12][rng.upto(3) as usize];
        let skip_kind = rng.upto(4);
        // Strength shapes: 0 = both random, 1 = Y all-zero (chroma-only),
        // 2 = UV all-zero (luma-only), 3 = ALL zero (whole walk a no-op).
        let shape = rng.upto(8);
        let strengths = mk_strengths(&mut rng, shape == 1 || shape == 3);
        let uv_strengths = mk_strengths(&mut rng, shape == 2 || shape == 3);
        let cs = Case {
            w,
            h,
            bd,
            num_planes,
            ss,
            damping: 3 + rng.upto(4) as i32,
            strengths,
            uv_strengths,
            skip_kind,
        };
        let (changed, minus_one, allskip_fb) = run_case(&mut rng, &cs);
        n += 1;
        n_changed += changed as u32;
        n_minus_one += minus_one as u32;
        n_allskip_fb += allskip_fb as u32;
        skip_kind_seen[skip_kind as usize] += 1;
    }
    println!(
        "cdef_frame diff: {n} cases, {n_changed} changed pixels, \
         {n_minus_one} had -1 fbs, {n_allskip_fb} had all-skip fbs, kinds {skip_kind_seen:?}"
    );
    // Coverage floors: the walk must genuinely filter in most cases, and the
    // skip/-1 arms must all have run.
    assert!(n_changed * 2 > n, "pixel-changing floor: {n_changed}/{n}");
    assert!(
        n_minus_one > 20,
        "-1 strength arm underexercised ({n_minus_one})"
    );
    assert!(
        n_allskip_fb > 20,
        "all-skip fb arm underexercised ({n_allskip_fb})"
    );
    assert!(
        skip_kind_seen.iter().all(|&k| k > 40),
        "skip kinds {skip_kind_seen:?}"
    );
}
