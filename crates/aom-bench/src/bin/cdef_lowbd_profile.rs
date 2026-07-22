//! Callgrind microbench for the bd8 CDEF lowbd lever. On a fixed CDEF-on,
//! filter-heavy (q32-shaped) frame workload, run N iterations of EITHER:
//!   * `u8`       — `cdef_frame_u8` directly on the u8 recon planes, or
//!   * `delegate` — what a bd8 tile does WITHOUT the u8 kernel: widen the whole
//!                  plane `u8 -> u16`, run the highbd `cdef_frame`, narrow the
//!                  whole plane `u16 -> u8`.
//!
//! so a callgrind Ir profile compares the two directly. The delta is the
//! whole-plane widen+narrow the u8 entry AVOIDS at the (pending) tile-plane
//! flip — the CDEF family's contribution to the lowbd pipeline. (The internal
//! filter is the SAME u16 SIMD in both, reused via `cdef_filter_block_u8`'s
//! per-block scratch, so this is not a "faster filter" claim.)
//!
//! Usage: cdef_lowbd_profile <u8|delegate> <iters>
//!
//! The first setup cross-checks u8 vs delegate byte-identity (a corrupt build
//! must never be profiled).

use aom_dsp::cdef::frame::{cdef_frame, cdef_frame_u8, CdefFrameParams};

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
}

fn align16(v: usize) -> usize {
    (v + 15) & !15
}

/// One CDEF-on frame's inputs (bd8, 4:2:0).
struct Frame {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
    mi_rows: i32,
    mi_cols: i32,
    ss_x: usize,
    ss_y: usize,
    damping: i32,
    strengths: [i32; 8],
    uv_strengths: [i32; 8],
    skip: Vec<bool>,
    unit_strength: Vec<i32>,
}

impl Frame {
    fn params(&self) -> CdefFrameParams<'_> {
        CdefFrameParams {
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            num_planes: 3,
            ss_x: self.ss_x,
            ss_y: self.ss_y,
            bit_depth: 8,
            damping: self.damping,
            cdef_strengths: self.strengths,
            cdef_uv_strengths: self.uv_strengths,
            skip_txfm: &self.skip,
            unit_strength: &self.unit_strength,
        }
    }
}

/// A representative CDEF pass: a few frame sizes decode spends CDEF time on,
/// 4:2:0, CDEF ON with mixed nonzero strengths, filter-heavy skip (~1/16 units
/// skipped — the q32 "filter-dominated" regime), every 64x64 unit strength-
/// enabled so most 8x8 units actually filter.
fn workload() -> Vec<Frame> {
    let mut rng = Rng(0x_cdef_9317_2026);
    let sizes = [(256usize, 256usize), (512, 512), (384, 320)];
    let mut frames = Vec::new();
    for (w, h) in sizes {
        let mi_cols = (((w + 7) & !7) >> 2) as i32;
        let mi_rows = (((h + 7) & !7) >> 2) as i32;
        let (ss_x, ss_y) = (1usize, 1usize);
        let yw = mi_cols as usize * 4;
        let yh = mi_rows as usize * 4;
        let y_stride = align16(yw);
        let uv_stride = y_stride >> ss_x;
        let (uvw, uvh) = (yw >> ss_x, yh >> ss_y);
        let mut mk = |w: usize, h: usize, stride: usize| -> Vec<u8> {
            let mut p = vec![0xABu8; stride * h];
            for r in 0..h {
                for c in 0..w {
                    p[r * stride + c] = (rng.next() & 0xff) as u8;
                }
            }
            p
        };
        let y = mk(yw, yh, y_stride);
        let u = mk(uvw, uvh, uv_stride);
        let v = mk(uvw, uvh, uv_stride);
        let ncells = (mi_rows * mi_cols) as usize;
        // filter-heavy: ~1/16 mi skip.
        let skip: Vec<bool> = (0..ncells).map(|_| (rng.next() % 16) == 0).collect();
        let nvfb = (mi_rows as usize).div_ceil(16);
        let nhfb = (mi_cols as usize).div_ceil(16);
        // every unit CDEF-on, strength index cycling through nonzero levels.
        let unit_strength: Vec<i32> = (0..nvfb * nhfb).map(|i| (1 + (i % 7)) as i32).collect();
        frames.push(Frame {
            y,
            u,
            v,
            y_stride,
            uv_stride,
            mi_rows,
            mi_cols,
            ss_x,
            ss_y,
            damping: 4,
            strengths: [1, 5, 9, 13, 17, 21, 25, 29],
            uv_strengths: [2, 6, 10, 14, 18, 22, 26, 30],
            skip,
            unit_strength,
        });
    }
    frames
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: cdef_lowbd_profile <u8|delegate> <iters>");
        std::process::exit(2);
    }
    let side = args[1].as_str();
    let iters: usize = args[2].parse().expect("iters must be a number");
    let frames = workload();

    // Byte-identity cross-check: the two sides must agree on every frame.
    for f in &frames {
        let (mut y1, mut u1, mut v1) = (f.y.clone(), f.u.clone(), f.v.clone());
        cdef_frame_u8(&mut y1, f.y_stride, &mut u1, &mut v1, f.uv_stride, &f.params());
        let widen = |p: &[u8]| -> Vec<u16> { p.iter().map(|&x| x as u16).collect() };
        let (mut y2, mut u2, mut v2) = (widen(&f.y), widen(&f.u), widen(&f.v));
        cdef_frame(&mut y2, f.y_stride, &mut u2, &mut v2, f.uv_stride, &f.params());
        let narrow = |p: &[u16]| -> Vec<u8> { p.iter().map(|&x| x as u8).collect() };
        assert_eq!(y1, narrow(&y2), "u8 vs delegate luma divergence");
        assert_eq!(u1, narrow(&u2), "u8 vs delegate U divergence");
        assert_eq!(v1, narrow(&v2), "u8 vs delegate V divergence");
    }

    let mut sink = 0u64;
    match side {
        "u8" => {
            // Reusable u8 dst planes (mirroring the decoder's plane reuse).
            let mut dsts: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
                frames.iter().map(|f| (f.y.clone(), f.u.clone(), f.v.clone())).collect();
            for _ in 0..iters {
                for (f, (y, u, v)) in frames.iter().zip(dsts.iter_mut()) {
                    y.copy_from_slice(&f.y);
                    u.copy_from_slice(&f.u);
                    v.copy_from_slice(&f.v);
                    cdef_frame_u8(y, f.y_stride, u, v, f.uv_stride, &f.params());
                    sink = sink.wrapping_add(y[0] as u64);
                }
            }
        }
        "delegate" => {
            // Reusable u16 tmp planes (the widen target) + u8 dst planes.
            let mut tmp: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)> = frames
                .iter()
                .map(|f| (vec![0u16; f.y.len()], vec![0u16; f.u.len()], vec![0u16; f.v.len()]))
                .collect();
            let mut dsts: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> =
                frames.iter().map(|f| (f.y.clone(), f.u.clone(), f.v.clone())).collect();
            for _ in 0..iters {
                for ((f, (ty, tu, tv)), (dy, du, dv)) in
                    frames.iter().zip(tmp.iter_mut()).zip(dsts.iter_mut())
                {
                    // widen u8 -> u16 (whole plane)
                    for (d, s) in ty.iter_mut().zip(f.y.iter()) {
                        *d = *s as u16;
                    }
                    for (d, s) in tu.iter_mut().zip(f.u.iter()) {
                        *d = *s as u16;
                    }
                    for (d, s) in tv.iter_mut().zip(f.v.iter()) {
                        *d = *s as u16;
                    }
                    cdef_frame(ty, f.y_stride, tu, tv, f.uv_stride, &f.params());
                    // narrow u16 -> u8 (whole plane)
                    for (d, s) in dy.iter_mut().zip(ty.iter()) {
                        *d = *s as u8;
                    }
                    for (d, s) in du.iter_mut().zip(tu.iter()) {
                        *d = *s as u8;
                    }
                    for (d, s) in dv.iter_mut().zip(tv.iter()) {
                        *d = *s as u8;
                    }
                    sink = sink.wrapping_add(dy[0] as u64);
                }
            }
        }
        other => panic!("side must be u8|delegate, got {other}"),
    }
    eprintln!("{side} x{iters}: sink={sink}");
}
