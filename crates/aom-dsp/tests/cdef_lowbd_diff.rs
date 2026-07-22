//! Differential harness for the bd8 LOWBD (u8 pixel) CDEF frame walk
//! ([`aom_dsp::cdef::frame::cdef_frame_u8`]) vs C libaom v3.14.1 AND vs the
//! port's own highbd (u16) bd8 walk. This is the byte-identity PROOF for the
//! lowbd decode pipeline's CDEF lever: the narrower u8 recon plane must filter
//! to the SAME pixel every C (lowbd) and the u16 port produce at bit depth 8.
//!
//! Two independent oracles, both asserted per visible pixel:
//!   1. `u8_out[i] as u16 == ref_cdef_frame(bd=8)[i]` — vs the REAL exported C
//!      single-threaded decoder walk. At `bd == 8` `shim_cdef_frame` converts
//!      the planes to `u8` and runs `av1_cdef_frame` with `use_highbitdepth=0`,
//!      i.e. the genuine C LOWBD `cdef_filter_8_*` path (dec_shim.c:1326-1345).
//!   2. `u8_out[i] as u16 == cdef_frame(bd=8)[i]` — vs the port's already
//!      C-verified highbd (u16) walk (`cdef_frame_diff.rs`); guards against the
//!      two paths ever drifting.
//!
//! The three sides get DIFFERENT out-of-visible padding fills, so visible-region
//! byte-identity ALSO proves the u8 walk (like the u16 walk) never depends on the
//! out-of-frame columns the line-buffer copies read.
//!
//! Sweep axes mirror `cdef_frame_diff.rs` (frame shapes incl. non-multiple-of-64
//! and 8-aligned crops, 4:2:0 / 4:2:2 / 4:4:4 / monochrome, damping 3..=6, random
//! per-slot strength grids incl. zero-Y / zero-UV / all-zero, per-64x64 unit
//! strength incl. the -1 nothing-read arm, and skip patterns all-skip..none-skip)
//! but with the bit depth fixed at 8 — the only depth the u8 path serves.
//!
//! An `AOM_FORCE_SCALAR=1` run of this binary exercises the same asserts with the
//! CDEF SIMD dispatch pinned to the scalar core (the u8 filter reuses the u16
//! SIMD+scalar dispatch via a per-block scratch — see `cdef_filter_block_u8`).

use aom_dsp::cdef::frame::{cdef_frame, cdef_frame_u8, CdefFrameParams};
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

/// Distinct out-of-visible padding fills, one per side, so visible-region
/// identity proves neither walk reads out-of-frame columns. All `<= 255` so the
/// C side's `u16 -> u8 -> u16` shim roundtrip is lossless.
const U8_PAD: u8 = 0xAB;
const PORT_PAD: u16 = 0x33;
const C_PAD: u16 = 0x5C;

struct Case {
    w: usize,
    h: usize,
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

    // Plane geometry: EXACT minimum stride the walk requires.
    let yw = mi_cols as usize * 4;
    let yh = mi_rows as usize * 4;
    let y_stride = align16(yw);
    let (uvw, uvh, uv_stride) = if cs.num_planes > 1 {
        (yw >> ss_x, yh >> ss_y, y_stride >> ss_x)
    } else {
        (0, 0, 0)
    };

    // Visible bd8 pixels (0..=255) as u8; padding = U8_PAD.
    let mut mk_u8_plane = |w: usize, h: usize, stride: usize| -> Vec<u8> {
        let mut p = vec![U8_PAD; stride * h];
        for r in 0..h {
            for col in 0..w {
                p[r * stride + col] = rng.upto(256) as u8;
            }
        }
        p
    };
    let y8 = mk_u8_plane(yw, yh, y_stride);
    let u8p = mk_u8_plane(uvw, uvh, uv_stride);
    let v8 = mk_u8_plane(uvw, uvh, uv_stride);

    // The two u16 sides share the SAME visible pixels (widened), DIFFERENT pad.
    let widen = |src: &[u8], w: usize, h: usize, stride: usize, pad: u16| -> Vec<u16> {
        let mut p = vec![pad; src.len()];
        for r in 0..h {
            for col in 0..w {
                p[r * stride + col] = src[r * stride + col] as u16;
            }
        }
        p
    };

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
    // An fb skipped purely by all-skip aggregation.
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

    // Shared params (borrows `skip` + `unit_strength`; the calls take `&`).
    let params = CdefFrameParams {
        mi_rows,
        mi_cols,
        num_planes: cs.num_planes,
        ss_x,
        ss_y,
        bit_depth: 8,
        damping: cs.damping,
        cdef_strengths: cs.strengths,
        cdef_uv_strengths: cs.uv_strengths,
        skip_txfm: &skip,
        unit_strength: &unit_strength,
    };

    // Side 1: lowbd u8 walk (the thing under test).
    let (mut y_lo, mut u_lo, mut v_lo) = (y8.clone(), u8p.clone(), v8.clone());
    cdef_frame_u8(&mut y_lo, y_stride, &mut u_lo, &mut v_lo, uv_stride, &params);

    // Side 2: port highbd u16 walk (same visible pixels, PORT_PAD).
    let mut y_hi = widen(&y8, yw, yh, y_stride, PORT_PAD);
    let mut u_hi = widen(&u8p, uvw, uvh, uv_stride, PORT_PAD);
    let mut v_hi = widen(&v8, uvw, uvh, uv_stride, PORT_PAD);
    cdef_frame(&mut y_hi, y_stride, &mut u_hi, &mut v_hi, uv_stride, &params);

    // Side 3: REAL C lowbd walk (same visible pixels, C_PAD).
    let mut y_c = widen(&y8, yw, yh, y_stride, C_PAD);
    let mut u_c = widen(&u8p, uvw, uvh, uv_stride, C_PAD);
    let mut v_c = widen(&v8, uvw, uvh, uv_stride, C_PAD);
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
        8,
        cs.damping,
        &cs.strengths,
        &cs.uv_strengths,
        &skip_i32,
        &unit_strength,
    );

    let ctx = format!(
        "{}x{} planes={} ss={:?} damp={} skip_kind={} y={:?} uv={:?}",
        cs.w,
        cs.h,
        cs.num_planes,
        cs.ss,
        cs.damping,
        cs.skip_kind,
        cs.strengths,
        cs.uv_strengths
    );
    let mut changed = false;
    let mut check = |name: &str,
                     lo: &[u8],
                     hi: &[u16],
                     cref: &[u16],
                     orig: &[u8],
                     w: usize,
                     h: usize,
                     stride: usize| {
        for r in 0..h {
            for col in 0..w {
                let idx = r * stride + col;
                assert_eq!(
                    lo[idx] as u16, cref[idx],
                    "{name} vs C lowbd: row {r} col {col} [{ctx}]"
                );
                assert_eq!(
                    lo[idx] as u16, hi[idx],
                    "{name} vs highbd port: row {r} col {col} [{ctx}]"
                );
                if lo[idx] != orig[idx] {
                    changed = true;
                }
            }
            // Padding untouched on the u8 side (proves no out-of-frame writes;
            // with the differing fills across the three sides, visible identity
            // above already proves no out-of-frame reads).
            assert!(
                lo[r * stride + w..(r + 1) * stride]
                    .iter()
                    .all(|&x| x == U8_PAD),
                "{name} u8 padding touched row {r} [{ctx}]"
            );
            assert!(
                hi[r * stride + w..(r + 1) * stride]
                    .iter()
                    .all(|&x| x == PORT_PAD),
                "{name} port padding touched row {r} [{ctx}]"
            );
            assert!(
                cref[r * stride + w..(r + 1) * stride]
                    .iter()
                    .all(|&x| x == C_PAD),
                "{name} C padding touched row {r} [{ctx}]"
            );
        }
    };
    check("Y", &y_lo, &y_hi, &y_c, &y8, yw, yh, y_stride);
    if cs.num_planes > 1 {
        check("U", &u_lo, &u_hi, &u_c, &u8p, uvw, uvh, uv_stride);
        check("V", &v_lo, &v_hi, &v_c, &v8, uvw, uvh, uv_stride);
    }
    (changed, any_minus_one, any_allskip_fb)
}

#[test]
fn cdef_frame_u8_matches_c_lowbd_and_highbd_port() {
    let mut rng = Rng(0x_cdef_10bd_2026_4d57);
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
        let skip_kind = rng.upto(4);
        // Strength shapes: 0 = both random, 1 = Y all-zero (chroma-only),
        // 2 = UV all-zero (luma-only), 3 = ALL zero (whole walk a no-op).
        let shape = rng.upto(8);
        let strengths = mk_strengths(&mut rng, shape == 1 || shape == 3);
        let uv_strengths = mk_strengths(&mut rng, shape == 2 || shape == 3);
        let cs = Case {
            w,
            h,
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
        "cdef_frame_u8 diff: {n} cases, {n_changed} changed pixels, \
         {n_minus_one} had -1 fbs, {n_allskip_fb} had all-skip fbs, kinds {skip_kind_seen:?}"
    );
    // Coverage floors: the u8 walk must genuinely filter in most cases, and the
    // skip/-1 arms must all have run.
    assert!(n_changed * 2 > n, "pixel-changing floor: {n_changed}/{n}");
    assert!(n_minus_one > 20, "-1 strength arm underexercised ({n_minus_one})");
    assert!(n_allskip_fb > 20, "all-skip fb arm underexercised ({n_allskip_fb})");
    assert!(
        skip_kind_seen.iter().all(|&k| k > 40),
        "skip kinds {skip_kind_seen:?}"
    );
}
