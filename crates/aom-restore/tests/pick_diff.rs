//! Loop-restoration ENCODER-SEARCH numeric core vs the REAL C `_c` exports
//! (av1/encoder/pickrst.c):
//!
//! - `compute_stats` vs `av1_compute_stats_c` (lowbd, incl. the downsampled
//!   mode) and `compute_stats_highbd` vs `av1_compute_stats_highbd_c`
//!   (bd 8/10/12): M/H byte-identical.
//! - `pixel_proj_error` vs `av1_{lowbd,highbd}_pixel_proj_error_c`, all
//!   three radius classes.
//! - `calc_proj_params` / `get_proj_subspace` vs
//!   `av1_calc_proj_params[_high_bd]_c` (H/C accumulators identical; the
//!   xq solve is composed on top of the identical accumulators).
//! - `sgr::selfguided_restoration` (the search's flt0/flt1 producer) vs the
//!   EXPORTED `av1_selfguided_restoration_c`, both arms.
//!
//! The Wiener decompose/solve chain (`wiener_decompose_sep_sym` /
//! `linsolve_wiener` / `compute_score` / `finalize_sym_filter`) has NO C
//! export (static): it is transcribed and covered here only by range/
//! symmetry invariants on stats from real-shaped content; its true
//! validation is the end-to-end encoder gate (later chunk).

use aom_restore::pick;
use aom_restore::sgr;
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u64) as i32
    }
}

fn rand_plane_u16(rng: &mut Rng, len: usize, bd: i32) -> Vec<u16> {
    let mask = (1u64 << bd) - 1;
    (0..len).map(|_| (rng.next() & mask) as u16).collect()
}

/// Smooth-ish content (random walk) — exercises non-degenerate stats the way
/// real reconstructions do, unlike white noise.
fn walk_plane_u16(rng: &mut Rng, w: usize, h: usize, bd: i32) -> Vec<u16> {
    let maxv = (1i32 << bd) - 1;
    let mut out = vec![0u16; w * h];
    let mut v = rng.range(0, maxv);
    for r in 0..h {
        for cx in 0..w {
            v = (v + rng.range(-6, 6)).clamp(0, maxv);
            out[r * w + cx] = v as u16;
        }
    }
    out
}

#[test]
fn compute_stats_lowbd_matches_c() {
    let mut rng = Rng(0x57A7_5EED_0001);
    // Rect sizes incl. the odd/tiny shapes the 150% last-unit tiling makes.
    let rects: &[(i32, i32)] = &[(8, 8), (13, 9), (64, 64), (96, 40), (37, 64), (5, 3)];
    for case in 0..40 {
        for &(rw, rh) in rects {
            for &win in &[7usize, 5] {
                for &downsampled in &[false, true] {
                    // Buffer with >= 3 px margin around the stats rect.
                    let margin = 8i32;
                    let stride = rw + 2 * margin;
                    let bh = rh + 2 * margin;
                    let (dgd16, src16) = if case % 2 == 0 {
                        (
                            rand_plane_u16(&mut rng, (stride * bh) as usize, 8),
                            rand_plane_u16(&mut rng, (stride * bh) as usize, 8),
                        )
                    } else {
                        let d = walk_plane_u16(&mut rng, stride as usize, bh as usize, 8);
                        // src = dgd + small noise (the real relationship).
                        let s = d
                            .iter()
                            .map(|&v| (v as i32 + rng.range(-4, 4)).clamp(0, 255) as u16)
                            .collect();
                        (d, s)
                    };
                    let dgd8: Vec<u8> = dgd16.iter().map(|&v| v as u8).collect();
                    let src8: Vec<u8> = src16.iter().map(|&v| v as u8).collect();

                    let (h_start, h_end) = (margin, margin + rw);
                    let (v_start, v_end) = (margin, margin + rh);
                    let win2 = win * win;
                    let mut m = vec![0i64; win2];
                    let mut h = vec![0i64; win2 * win2];
                    pick::compute_stats(
                        win, &dgd16, 0, &src16, h_start, h_end, v_start, v_end, stride, stride,
                        &mut m, &mut h, downsampled,
                    );
                    let (cm, ch) = c::ref_compute_stats(
                        win, &dgd8, &src8, h_start, h_end, v_start, v_end, stride, stride,
                        downsampled,
                    );
                    assert_eq!(m, cm, "M case{case} {rw}x{rh} win{win} ds{downsampled}");
                    assert_eq!(h, ch, "H case{case} {rw}x{rh} win{win} ds{downsampled}");
                }
            }
        }
    }
}

#[test]
fn compute_stats_highbd_matches_c() {
    let mut rng = Rng(0x57A7_5EED_0002);
    let rects: &[(i32, i32)] = &[(8, 8), (13, 9), (64, 64), (48, 20)];
    for case in 0..24 {
        for &(rw, rh) in rects {
            for &win in &[7usize, 5] {
                for &bd in &[8, 10, 12] {
                    let margin = 8i32;
                    let stride = rw + 2 * margin;
                    let bh = rh + 2 * margin;
                    let dgd = if case % 2 == 0 {
                        rand_plane_u16(&mut rng, (stride * bh) as usize, bd)
                    } else {
                        walk_plane_u16(&mut rng, stride as usize, bh as usize, bd)
                    };
                    let maxv = (1i32 << bd) - 1;
                    let src: Vec<u16> = dgd
                        .iter()
                        .map(|&v| (v as i32 + rng.range(-9, 9)).clamp(0, maxv) as u16)
                        .collect();
                    let (h_start, h_end) = (margin, margin + rw);
                    let (v_start, v_end) = (margin, margin + rh);
                    let win2 = win * win;
                    let mut m = vec![0i64; win2];
                    let mut h = vec![0i64; win2 * win2];
                    pick::compute_stats_highbd(
                        win, &dgd, 0, &src, h_start, h_end, v_start, v_end, stride, stride,
                        &mut m, &mut h, bd,
                    );
                    let (cm, ch) = c::ref_compute_stats_highbd(
                        win, &dgd, &src, h_start, h_end, v_start, v_end, stride, stride, bd,
                    );
                    assert_eq!(m, cm, "M case{case} {rw}x{rh} win{win} bd{bd}");
                    assert_eq!(h, ch, "H case{case} {rw}x{rh} win{win} bd{bd}");
                }
            }
        }
    }
}

/// flt planes shaped like real SGR pass output: `pixel << 4` plus bounded
/// deviation (keeps the lowbd C asserts `|flt| < 1<<15` satisfied).
fn flt_like(rng: &mut Rng, dat: &[u16], off: usize, w: usize, h: usize, stride: usize) -> Vec<i32> {
    let mut f = vec![0i32; w * h];
    for i in 0..h {
        for j in 0..w {
            let u = (dat[off + i * stride + j] as i32) << 4;
            f[i * w + j] = u + rng.range(-256, 256);
        }
    }
    f
}

#[test]
fn pixel_proj_error_matches_c() {
    let mut rng = Rng(0x57A7_5EED_0003);
    // ep classes: 0..=9 r0+r1, 10..=13 r1 only, 14..=15 r0 only.
    let eps = [0usize, 5, 9, 10, 13, 14, 15];
    for case in 0..48 {
        for &(w, h) in &[(16usize, 16usize), (33, 17), (64, 64), (7, 5)] {
            for &ep in &eps {
                for &(bd, highbd) in &[(8, false), (8, true), (10, true), (12, true)] {
                    let stride = w + 11;
                    let bh = h + 4;
                    let dat = if case % 2 == 0 {
                        rand_plane_u16(&mut rng, stride * bh, bd)
                    } else {
                        walk_plane_u16(&mut rng, stride, bh, bd)
                    };
                    let maxv = (1i32 << bd) - 1;
                    let src: Vec<u16> = dat
                        .iter()
                        .map(|&v| (v as i32 + rng.range(-12, 12)).clamp(0, maxv) as u16)
                        .collect();
                    let off = 2 * stride + 3;
                    let mut flt0 = flt_like(&mut rng, &dat, off, w, h, stride);
                    let mut flt1 = flt_like(&mut rng, &dat, off, w, h, stride);
                    let xq = [rng.range(-96, 96), rng.range(-64, 64)];

                    let got = pick::pixel_proj_error(
                        &src, off, w, h, stride, &dat, off, stride, &flt0, w, &flt1, w, xq, ep,
                        highbd,
                    );
                    let dat8: Vec<u8> = dat.iter().map(|&v| v as u8).collect();
                    let src8: Vec<u8> = src.iter().map(|&v| v as u8).collect();
                    let want = c::ref_pixel_proj_error(
                        &src8[if highbd { 0 } else { off }..],
                        &src[if highbd { off } else { 0 }..],
                        w as i32,
                        h as i32,
                        stride as i32,
                        &dat8[if highbd { 0 } else { off }..],
                        &dat[if highbd { off } else { 0 }..],
                        stride as i32,
                        &mut flt0,
                        w as i32,
                        &mut flt1,
                        w as i32,
                        xq,
                        ep as i32,
                        highbd,
                    );
                    assert_eq!(
                        got, want,
                        "proj_err case{case} {w}x{h} ep{ep} bd{bd} hbd{highbd}"
                    );
                }
            }
        }
    }
}

#[test]
fn calc_proj_params_and_subspace_match_c() {
    let mut rng = Rng(0x57A7_5EED_0004);
    let eps = [0usize, 3, 9, 10, 12, 14, 15];
    for case in 0..48 {
        for &(w, h) in &[(16usize, 16usize), (64, 64), (40, 24)] {
            for &ep in &eps {
                for &(bd, highbd) in &[(8, false), (10, true), (12, true)] {
                    let stride = w + 7;
                    let bh = h + 2;
                    let dat = if case % 2 == 0 {
                        rand_plane_u16(&mut rng, stride * bh, bd)
                    } else {
                        walk_plane_u16(&mut rng, stride, bh, bd)
                    };
                    let maxv = (1i32 << bd) - 1;
                    let src: Vec<u16> = dat
                        .iter()
                        .map(|&v| (v as i32 + rng.range(-12, 12)).clamp(0, maxv) as u16)
                        .collect();
                    let off = stride + 1;
                    let mut flt0 = flt_like(&mut rng, &dat, off, w, h, stride);
                    let mut flt1 = flt_like(&mut rng, &dat, off, w, h, stride);

                    let (hh, cc) = pick::calc_proj_params(
                        &src, off, w, h, stride, &dat, off, stride, &flt0, w, &flt1, w, ep,
                    );
                    let dat8: Vec<u8> = dat.iter().map(|&v| v as u8).collect();
                    let src8: Vec<u8> = src.iter().map(|&v| v as u8).collect();
                    let (ch, ccc) = c::ref_calc_proj_params(
                        &src8[if highbd { 0 } else { off }..],
                        &src[if highbd { off } else { 0 }..],
                        w as i32,
                        h as i32,
                        stride as i32,
                        &dat8[if highbd { 0 } else { off }..],
                        &dat[if highbd { off } else { 0 }..],
                        stride as i32,
                        &mut flt0,
                        w as i32,
                        &mut flt1,
                        w as i32,
                        ep as i32,
                        highbd,
                    );
                    assert_eq!(
                        [hh[0][0], hh[0][1], hh[1][0], hh[1][1]],
                        ch,
                        "H case{case} {w}x{h} ep{ep} bd{bd}"
                    );
                    assert_eq!(cc, ccc, "C case{case} {w}x{h} ep{ep} bd{bd}");

                    // Composed: the xq solve on identical accumulators is a
                    // deterministic function; sanity-run it (ranges only —
                    // the true gate is e2e).
                    let xq = pick::get_proj_subspace(
                        &src, off, w, h, stride, &dat, off, stride, &flt0, w, &flt1, w, ep,
                    );
                    let xqd = pick::encode_xq(xq, ep);
                    assert!((-96..=31).contains(&xqd[0]), "xqd0 range: {xqd:?}");
                    assert!((-32..=95).contains(&xqd[1]), "xqd1 range: {xqd:?}");
                }
            }
        }
    }
}

#[test]
fn selfguided_flt_producer_matches_c() {
    let mut rng = Rng(0x57A7_5EED_0005);
    let eps = [0usize, 4, 9, 10, 13, 14, 15];
    for case in 0..24 {
        for &(w, h) in &[(32usize, 32usize), (64, 64), (24, 40), (16, 8)] {
            for &ep in &eps {
                for &(bd, highbd) in &[(8, false), (8, true), (10, true), (12, true)] {
                    let margin = 4usize;
                    let stride = w + 2 * margin;
                    let bh = h + 2 * margin;
                    let dgd = if case % 2 == 0 {
                        rand_plane_u16(&mut rng, stride * bh, bd)
                    } else {
                        walk_plane_u16(&mut rng, stride, bh, bd)
                    };
                    let off = margin * stride + margin;
                    let flt_stride = ((w + 7) & !7) + 8;
                    let mut flt0 = vec![0i32; flt_stride * h];
                    let mut flt1 = vec![0i32; flt_stride * h];
                    sgr::selfguided_restoration(
                        &dgd, off, stride, w, h, &mut flt0, &mut flt1, flt_stride, ep, bd,
                    );
                    let dgd8: Vec<u8> = dgd.iter().map(|&v| v as u8).collect();
                    let mut cflt0 = vec![0i32; flt_stride * h];
                    let mut cflt1 = vec![0i32; flt_stride * h];
                    let rc = c::ref_selfguided_restoration(
                        &dgd8,
                        &dgd,
                        off,
                        w as i32,
                        h as i32,
                        stride as i32,
                        &mut cflt0,
                        &mut cflt1,
                        flt_stride as i32,
                        ep as i32,
                        bd,
                        highbd,
                    );
                    assert_eq!(rc, 0, "C selfguided failed");
                    // Compare only the written w x h block per row (the tail
                    // of each flt row is scratch).
                    let (rads, _) = sgr::SGR_PARAMS[ep];
                    for i in 0..h {
                        if rads[0] > 0 {
                            assert_eq!(
                                &flt0[i * flt_stride..i * flt_stride + w],
                                &cflt0[i * flt_stride..i * flt_stride + w],
                                "flt0 row{i} case{case} {w}x{h} ep{ep} bd{bd} hbd{highbd}"
                            );
                        }
                        if rads[1] > 0 {
                            assert_eq!(
                                &flt1[i * flt_stride..i * flt_stride + w],
                                &cflt1[i * flt_stride..i * flt_stride + w],
                                "flt1 row{i} case{case} {w}x{h} ep{ep} bd{bd} hbd{highbd}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Transcription invariants for the export-less Wiener solve chain: on
/// real-shaped stats the decomposed + finalized taps are in the coded
/// ranges, symmetric, with the constrained centre — for both windows.
#[test]
fn wiener_solver_invariants() {
    let mut rng = Rng(0x57A7_5EED_0006);
    for case in 0..60 {
        for &win in &[7usize, 5] {
            let (rw, rh) = (64i32, 64i32);
            let margin = 8i32;
            let stride = rw + 2 * margin;
            let bh = rh + 2 * margin;
            let dgd = walk_plane_u16(&mut rng, stride as usize, bh as usize, 8);
            let src: Vec<u16> = dgd
                .iter()
                .map(|&v| (v as i32 + rng.range(-6, 6)).clamp(0, 255) as u16)
                .collect();
            let win2 = win * win;
            let mut m = vec![0i64; win2];
            let mut h = vec![0i64; win2 * win2];
            pick::compute_stats(
                win, &dgd, 0, &src, margin, margin + rw, margin, margin + rh, stride, stride,
                &mut m, &mut h, false,
            );
            let mut a = [0i32; 7];
            let mut b = [0i32; 7];
            pick::wiener_decompose_sep_sym(win, &m, &h, &mut a, &mut b);
            let mut vf = [0i16; 8];
            let mut hf = [0i16; 8];
            pick::finalize_sym_filter(win, &a, &mut vf);
            pick::finalize_sym_filter(win, &b, &mut hf);
            for f in [&vf, &hf] {
                assert!((-5..=10).contains(&f[0]), "case{case} tap0 {f:?}");
                assert!((-23..=8).contains(&f[1]), "case{case} tap1 {f:?}");
                assert!((-17..=46).contains(&f[2]), "case{case} tap2 {f:?}");
                assert_eq!(f[3], -2 * (f[0] + f[1] + f[2]), "case{case} centre");
                assert_eq!([f[4], f[5], f[6]], [f[2], f[1], f[0]], "case{case} mirror");
                assert_eq!(f[7], 0, "case{case} slot 7");
                if win == 5 {
                    assert_eq!(f[0], 0, "case{case} chroma tap0");
                }
            }
            // compute_score is finite and deterministic on the same input.
            let s1 = pick::compute_score(win, &m, &h, &vf, &hf);
            let s2 = pick::compute_score(win, &m, &h, &vf, &hf);
            assert_eq!(s1, s2);
        }
    }
}
