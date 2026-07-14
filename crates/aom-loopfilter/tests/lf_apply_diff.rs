//! Differential harness for the deblock APPLICATION layer
//! (`aom_loopfilter::frame`) vs REAL libaom v3.14.1:
//!
//! - `lf_frame_init` vs the real exported `av1_loop_filter_init` +
//!   `av1_loop_filter_frame_init` (threshold + level tables).
//! - `loop_filter_frame` (whole-frame strip/plane/dir walk +
//!   `set_lpf_parameters` per-edge derivation + kernels) vs the real exported
//!   `av1_filter_block_plane_vert`/`_horz` driven in the exact single-threaded
//!   `loop_filter_rows` order, over structurally-valid random mi grids
//!   (recursive partitions incl. rect + 1:4 shapes, intra/inter/intrabc/skip
//!   cells, per-cell tx variation, segmentation LF features, delta-lf,
//!   mode/ref deltas, sharpness, lossless segments) and gradient/flat/noisy
//!   plane content, at bd 8 (REAL LOWBD C path — proves the hbd-at-bd8
//!   equivalence the Rust side relies on), 10, and 12.
//!
//! 4:2:2 chroma filtering is intentionally NOT swept: with `ss = (1, 0)`
//! libaom's `av1_get_max_uv_txsize(mbmi->bsize, 1, 0)` hits
//! `BLOCK_INVALID = 255` for tall blocks (av1_ss_size_lookup[.][1][0],
//! common_data.c:17) — an out-of-bounds `max_txsize_rect_lookup` read in the
//! NDEBUG production build, which is not portable behavior. The decoder
//! envelope rejects 4:2:2 streams with nonzero chroma filter levels; 4:2:2
//! LUMA deblocking (plane 0 never derives a chroma bsize) IS swept here.

use aom_loopfilter::frame::{
    lf_frame_init, loop_filter_frame, LfFrameBuf, LfMi, LfMiGrid, LfParams, LfSeg, FRAME_LF_COUNT,
    MODE_LF_LUT,
};
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
    fn range_i(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.upto((hi - lo + 1) as u32) as i32
    }
    fn chance(&mut self, num: u32, den: u32) -> bool {
        self.upto(den) < num
    }
}

// ---- spec tables the generator needs (common_data.h) ----------------------------

/// `mi_size_wide` / `mi_size_high` per bsize.
const MI_W: [usize; 22] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
const MI_H: [usize; 22] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];
/// `max_txsize_rect_lookup[bsize]`.
const MAX_TX_RECT: [usize; 22] = [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];
/// `sub_tx_size_map[tx]` (common_data.h:165).
const SUB_TX: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];

/// (mi_w, mi_h) -> bsize index.
fn bsize_of(mi_w: usize, mi_h: usize) -> usize {
    (0..22)
        .find(|&b| MI_W[b] == mi_w && MI_H[b] == mi_h)
        .unwrap_or_else(|| panic!("no bsize {mi_w}x{mi_h}"))
}

// ---- random structurally-valid frame model --------------------------------------

struct Frame {
    mi_rows: usize,
    mi_cols: usize,
    mi: Vec<LfMi>,
    // Flattened parallel i32/i8 arrays for the C facade.
    bsize: Vec<i32>,
    txsize: Vec<i32>,
    seg: Vec<i32>,
    ref0: Vec<i32>,
    mode: Vec<i32>,
    skip: Vec<i32>,
    intrabc: Vec<i32>,
    dlf_base: Vec<i8>,
    dlf: Vec<i8>,
}

impl Frame {
    fn new(mi_rows: usize, mi_cols: usize) -> Self {
        let n = mi_rows * mi_cols;
        Frame {
            mi_rows,
            mi_cols,
            mi: vec![LfMi::default(); n],
            bsize: vec![0; n],
            txsize: vec![0; n],
            seg: vec![0; n],
            ref0: vec![0; n],
            mode: vec![0; n],
            skip: vec![0; n],
            intrabc: vec![0; n],
            dlf_base: vec![0; n],
            dlf: vec![0; 4 * n],
        }
    }

    /// Stamp one leaf block (bsize at mi position), clipped to the frame.
    fn stamp(&mut self, rng: &mut Rng, mi_r: usize, mi_c: usize, bsize: usize) {
        if mi_r >= self.mi_rows || mi_c >= self.mi_cols {
            return; // fully out of frame: not in the mi grid
        }
        // Block identity fields (uniform across the block's cells).
        let kind = rng.upto(10);
        let (is_inter, intrabc, ref0, mode) = if kind < 6 {
            (false, false, 0u8, rng.upto(13) as i32) // intra
        } else if kind < 7 {
            (true, true, 0u8, 0) // intrabc (DC_PRED)
        } else {
            (true, false, 1 + rng.upto(7) as u8, 13 + rng.upto(12) as i32) // inter
        };
        let skip = is_inter && rng.chance(1, 3);
        let segment_id = rng.upto(8) as u8;
        let dlf_base = rng.range_i(-63, 63) as i8;
        let dlf: [i8; FRAME_LF_COUNT] =
            core::array::from_fn(|_| rng.range_i(-63, 63) as i8);
        // Block tx depth chain (intra blocks are uniform; inter blocks vary
        // per cell to exercise the flattened vartx contract).
        let max_tx = MAX_TX_RECT[bsize];
        let chain = [max_tx, SUB_TX[max_tx], SUB_TX[SUB_TX[max_tx]]];
        let block_tx = chain[rng.upto(3) as usize];

        let h = MI_H[bsize].min(self.mi_rows - mi_r);
        let w = MI_W[bsize].min(self.mi_cols - mi_c);
        for r in 0..h {
            for cc in 0..w {
                let i = (mi_r + r) * self.mi_cols + (mi_c + cc);
                let tx = if is_inter && !skip {
                    chain[rng.upto(3) as usize] // per-cell (vartx flattening)
                } else {
                    block_tx
                };
                self.mi[i] = LfMi {
                    bsize: bsize as u8,
                    tx_size: tx as u8,
                    segment_id,
                    ref0,
                    mode_lf: MODE_LF_LUT[mode as usize],
                    is_inter,
                    skip_txfm: skip,
                    delta_lf_from_base: dlf_base,
                    delta_lf: dlf,
                };
                self.bsize[i] = bsize as i32;
                self.txsize[i] = tx as i32;
                self.seg[i] = segment_id as i32;
                self.ref0[i] = ref0 as i32;
                self.mode[i] = mode;
                self.skip[i] = skip as i32;
                self.intrabc[i] = intrabc as i32;
                self.dlf_base[i] = dlf_base;
                self.dlf[i * FRAME_LF_COUNT..(i + 1) * FRAME_LF_COUNT].copy_from_slice(&dlf);
            }
        }
    }

    /// Recursive partition of a `sq`-mi square node (like the AV1 tree; rect +
    /// 1:4 leaves included so every block shape appears).
    fn part(&mut self, rng: &mut Rng, mi_r: usize, mi_c: usize, sq: usize) {
        let can_split = sq > 1;
        let choice = rng.upto(10);
        if can_split && (choice < 4 || sq > 8) {
            let h = sq / 2;
            self.part(rng, mi_r, mi_c, h);
            self.part(rng, mi_r, mi_c + h, h);
            self.part(rng, mi_r + h, mi_c, h);
            self.part(rng, mi_r + h, mi_c + h, h);
        } else if can_split && choice < 6 {
            // HORZ: two sq x sq/2.
            let b = bsize_of(sq, sq / 2);
            self.stamp(rng, mi_r, mi_c, b);
            self.stamp(rng, mi_r + sq / 2, mi_c, b);
        } else if can_split && choice < 8 {
            // VERT.
            let b = bsize_of(sq / 2, sq);
            self.stamp(rng, mi_r, mi_c, b);
            self.stamp(rng, mi_r, mi_c + sq / 2, b);
        } else if sq >= 4 && choice == 8 {
            // HORZ_4 / VERT_4.
            if rng.chance(1, 2) {
                let b = bsize_of(sq, sq / 4);
                for k in 0..4 {
                    self.stamp(rng, mi_r + k * (sq / 4), mi_c, b);
                }
            } else {
                let b = bsize_of(sq / 4, sq);
                for k in 0..4 {
                    self.stamp(rng, mi_r, mi_c + k * (sq / 4), b);
                }
            }
        } else {
            self.stamp(rng, mi_r, mi_c, bsize_of(sq, sq));
        }
    }

    fn generate(rng: &mut Rng, mi_rows: usize, mi_cols: usize, sb_mi: usize) -> Self {
        let mut f = Frame::new(mi_rows, mi_cols);
        let mut r = 0;
        while r < mi_rows {
            let mut c0 = 0;
            while c0 < mi_cols {
                f.part(rng, r, c0, sb_mi);
                c0 += sb_mi;
            }
            r += sb_mi;
        }
        f
    }
}

/// Recon-like plane content: smooth gradient + structure + amplitude-chosen
/// noise (small amp exercises flat/flat2, large amp the mask-off path).
fn gen_plane(rng: &mut Rng, w: usize, h: usize, stride: usize, bd: i32) -> Vec<u16> {
    let maxv = (1u32 << bd) - 1;
    let amp = [2u32, 8, 24, 96][rng.upto(4) as usize].min(maxv / 2);
    let base = rng.upto(maxv - 2 * amp) + amp;
    let dx = rng.range_i(-3, 3);
    let dy = rng.range_i(-3, 3);
    let mut p = vec![0u16; stride * h];
    for r in 0..h {
        for col in 0..w {
            let g = base as i64 + (dx as i64 * col as i64 + dy as i64 * r as i64) / 4
                + rng.upto(2 * amp + 1) as i64
                - amp as i64;
            p[r * stride + col] = g.clamp(0, maxv as i64) as u16;
        }
    }
    p
}

fn random_params(rng: &mut Rng, luma_on: bool, chroma_on: bool) -> LfParams {
    let mut p = LfParams {
        filter_level: if luma_on {
            [rng.range_i(0, 63), rng.range_i(0, 63)]
        } else {
            [0, 0]
        },
        filter_level_u: if chroma_on { rng.range_i(0, 63) } else { 0 },
        filter_level_v: if chroma_on { rng.range_i(0, 63) } else { 0 },
        sharpness: rng.range_i(0, 7),
        mode_ref_delta_enabled: rng.chance(3, 4),
        ref_deltas: core::array::from_fn(|_| rng.range_i(-63, 63) as i8),
        mode_deltas: core::array::from_fn(|_| rng.range_i(-63, 63) as i8),
        delta_lf_present: rng.chance(1, 3),
        delta_lf_multi: rng.chance(1, 2),
        lossless: core::array::from_fn(|_| rng.chance(1, 8)),
        seg: LfSeg::default(),
    };
    if rng.chance(1, 3) {
        p.seg.enabled = true;
        for s in 0..8 {
            for f in 0..4 {
                if rng.chance(1, 3) {
                    p.seg.active[s][f] = true;
                    p.seg.data[s][f] = rng.range_i(-63, 63);
                }
            }
        }
    }
    p
}

fn ref_params(p: &LfParams) -> c::RefLfParams {
    c::RefLfParams {
        filter_level: [p.filter_level[0], p.filter_level[1], p.filter_level_u, p.filter_level_v],
        sharpness: p.sharpness,
        mode_ref_delta_enabled: p.mode_ref_delta_enabled,
        ref_deltas: p.ref_deltas,
        mode_deltas: p.mode_deltas,
        delta_lf_present: p.delta_lf_present,
        delta_lf_multi: p.delta_lf_multi,
        lossless: p.lossless,
        seg_enabled: p.seg.enabled,
        seg_active: p.seg.active,
        seg_data: p.seg.data,
    }
}

#[test]
fn frame_init_tables_match_c() {
    c::ref_init();
    let mut rng = Rng(0x1F2E_3D4C_5B6A_7988);
    let mut n = 0u32;
    for case in 0..400 {
        let (luma_on, chroma_on) = match case % 4 {
            0 => (true, true),
            1 => (true, false),
            2 => (false, true), // zero luma: plane loop breaks like C
            _ => (false, false),
        };
        let p = random_params(&mut rng, luma_on, chroma_on);
        let plane_end = 1 + rng.upto(3) as i32; // 1..=3
        let (c_lfthr, c_lvl) = c::ref_lf_frame_init_tables(&ref_params(&p), 0, plane_end);
        let lfi = lf_frame_init(&p, 0, plane_end as usize);
        for (l, &(mblim, lim, hev)) in lfi.lfthr.iter().enumerate() {
            assert_eq!(
                [mblim, lim, hev],
                c_lfthr[l],
                "lfthr[{l}] sharpness={}",
                p.sharpness
            );
        }
        let mut idx = 0usize;
        for plane in 0..3 {
            for seg in 0..8 {
                for dir in 0..2 {
                    for r in 0..8 {
                        for m in 0..2 {
                            assert_eq!(
                                lfi.lvl[plane][seg][dir][r][m], c_lvl[idx],
                                "lvl[{plane}][{seg}][{dir}][{r}][{m}] case {case}"
                            );
                            idx += 1;
                        }
                    }
                }
            }
        }
        n += 1;
    }
    assert_eq!(n, 400);
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    rng: &mut Rng,
    w: usize,
    h: usize,
    ss_x: usize,
    ss_y: usize,
    mono: bool,
    bd: i32,
    sb_mi: usize,
    force_zero_levels: bool,
) -> bool {
    // 8px-aligned mi dims (set_mb_mi).
    let mi_cols = ((w + 7) & !7) >> 2;
    let mi_rows = ((h + 7) & !7) >> 2;
    let f = Frame::generate(rng, mi_rows, mi_cols, sb_mi);

    let luma_on = !force_zero_levels;
    // 4:2:2 chroma deblocking is out of scope (see module doc).
    let chroma_on = luma_on && !mono && !(ss_x == 1 && ss_y == 0) && rng.chance(3, 4);
    let p = random_params(rng, luma_on, chroma_on);

    let y_stride = mi_cols * 4 + [0, 8, 36][rng.upto(3) as usize]; // strided too
    let y_rows = mi_rows * 4;
    let (uv_stride, uv_rows) = if mono {
        (0usize, 0usize)
    } else {
        (y_stride >> ss_x, y_rows >> ss_y)
    };
    let mut y = gen_plane(rng, mi_cols * 4, y_rows, y_stride, bd);
    let (mut u, mut v) = if mono {
        (Vec::new(), Vec::new())
    } else {
        (
            gen_plane(rng, (mi_cols * 4) >> ss_x, uv_rows, uv_stride, bd),
            gen_plane(rng, (mi_cols * 4) >> ss_x, uv_rows, uv_stride, bd),
        )
    };
    let (y0, u0, v0) = (y.clone(), u.clone(), v.clone());

    // C reference (real av1_filter_block_plane_vert/horz walk).
    let (mut cy, mut cu, mut cv) = (y0.clone(), u0.clone(), v0.clone());
    let grid = c::RefLfGrid {
        mi_rows: mi_rows as i32,
        mi_cols: mi_cols as i32,
        grid_stride: mi_cols as i32,
        bsize: &f.bsize,
        txsize: &f.txsize,
        seg: &f.seg,
        ref0: &f.ref0,
        mode: &f.mode,
        skip: &f.skip,
        intrabc: &f.intrabc,
        dlf_base: &f.dlf_base,
        dlf: &f.dlf,
    };
    let num_planes: usize = if mono { 1 } else { 3 };
    c::ref_lf_filter_frame(
        &mut cy, y_stride, &mut cu, &mut cv, uv_stride,
        w as i32, h as i32, ss_x as i32, ss_y as i32, bd,
        &grid, &ref_params(&p), 0, num_planes as i32,
    );

    // Rust.
    let mi_grid = LfMiGrid {
        mi: &f.mi,
        stride: mi_cols,
        mi_rows: mi_rows as i32,
        mi_cols: mi_cols as i32,
    };
    {
        let mut buf = LfFrameBuf {
            y: &mut y,
            y_stride,
            u: &mut u,
            v: &mut v,
            uv_stride,
            crop_width: w as u32,
            crop_height: h as u32,
            ss_x,
            ss_y,
            bd,
        };
        loop_filter_frame(&mut buf, &mi_grid, &p, 0, num_planes);
    }

    assert_eq!(y, cy, "LUMA {w}x{h} ss=({ss_x},{ss_y}) bd{bd} mono={mono}");
    assert_eq!(u, cu, "U {w}x{h} ss=({ss_x},{ss_y}) bd{bd}");
    assert_eq!(v, cv, "V {w}x{h} ss=({ss_x},{ss_y}) bd{bd}");
    if force_zero_levels {
        // Zero levels: both sides must be exact no-ops.
        assert_eq!(y, y0, "zero-level walk must not touch luma");
        assert_eq!(u, u0);
        assert_eq!(v, v0);
    }
    // Coverage signal: did the C filter change anything?
    y != y0 || u != u0 || v != v0
}

#[test]
fn filter_frame_matches_c() {
    c::ref_init();
    let mut rng = Rng(0xA1B2_C3D4_E5F6_0718);
    // (w, h): SB-multiple, non-SB-multiple, non-8-multiple crops, multi-strip
    // (>128 needs more than one 32-mi strip), tiny.
    let shapes = [
        (64usize, 64usize),
        (96, 80),
        (100, 76),
        (160, 144),
        (32, 32),
        (196, 132),
    ];
    // (ss_x, ss_y, mono): 420, 444, 422 (luma-only — see module doc), mono.
    let formats = [
        (1usize, 1usize, false),
        (0, 0, false),
        (1, 0, false),
        (1, 1, true),
    ];
    let mut n = 0u32;
    let mut changed = 0u32;
    for &(w, h) in &shapes {
        for &(ss_x, ss_y, mono) in &formats {
            for &bd in &[8i32, 10, 12] {
                for rep in 0..3 {
                    // sb_mi 16 = 64x64 SBs (the envelope); one rep with 32 =
                    // 128x128 blocks to cover the biggest bsizes.
                    let sb_mi = if rep == 2 { 32 } else { 16 };
                    changed += run_one(&mut rng, w, h, ss_x, ss_y, mono, bd, sb_mi, false) as u32;
                    n += 1;
                }
            }
        }
    }
    // Zero-level no-op arm (the lf==0 envelope streams take this path).
    for &(w, h) in &[(64usize, 64usize), (100, 76)] {
        for &(ss_x, ss_y, mono) in &formats {
            run_one(&mut rng, w, h, ss_x, ss_y, mono, 8, 16, true);
            n += 1;
        }
    }
    assert_eq!(n, 6 * 4 * 3 * 3 + 8);
    // The sweep must actually exercise filtering (not degenerate to no-ops).
    assert!(
        changed > n * 3 / 5,
        "only {changed}/{n} runs changed pixels — coverage collapsed"
    );
}
