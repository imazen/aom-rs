//! The CDEF frame walk — bit-exact port of `av1_cdef_frame` (libaom v3.14.1
//! `av1/common/cdef.c`, the single-threaded decoder path via
//! `av1_cdef_init_fb_row` / `av1_cdef_fb_row` / `cdef_fb_col` /
//! `cdef_prepare_fb` / `av1_cdef_filter_fb`).
//!
//! Semantics carried over from the C (each verified in source):
//! - per 64x64 filter block (fb): the strength INDEX comes from the mi grid's
//!   top-left `MB_MODE_INFO::cdef_strength` (the decoder stored the literal
//!   read at the fb's first non-skip block there; `-1`/missing skips the fb —
//!   cdef.c:315-320). Y strength = `cdef_strengths[idx]`, UV =
//!   `cdef_uv_strengths[idx]`; split `level = s / 4`, `sec = s % 4` with
//!   `sec += (sec == 3)` (so sec in {0,1,2,4}) — cdef.c:323-338.
//! - skip aggregation: an 8x8 unit joins the fb's dlist iff ANY of its 2x2 mi
//!   has `skip_txfm == 0` (`is_8x8_block_skip` requires all-skip to skip —
//!   cdef.c:29-39); an fb whose dlist is empty is skipped entirely.
//! - luma is prepared+filtered even at Y level 0 when UV is nonzero (dirs are
//!   computed on luma; the Y filter call degenerates to a copy) — cdef.c:354-365.
//! - source priming (`cdef_prepare_fb`): the 16-bit `src` block (stride
//!   [`CDEF_BSTRIDE`], origin at (`CDEF_VBORDER`, `CDEF_HBORDER`)) reads the
//!   deblocked-but-not-yet-CDEF'd frame: current fb + 8 px to the right
//!   directly from the frame; the left 8 px from `colbuf` when the left fb
//!   was filtered (saved UNFILTERED there before filtering), else from the
//!   frame; the top/bottom 2 rows from the `linebuf` ping-pong line stores
//!   (saved pre-filter in `init_fb_row`); frame edges fill
//!   [`CDEF_VERY_LARGE`]. Corner copies gate on neighbour existence.
//! - per-8x8 strengths (`av1_cdef_filter_fb`): luma primary is
//!   variance-adjusted (`adjust_strength`), chroma is used as-is; the filter
//!   variant drops the primary/secondary taps when the respective strength is
//!   0; `dir` passed 0 when the PLANE-level primary strength is 0; chroma
//!   damping is one less than luma (`damping += coeff_shift - (pli != 0)`).
//! - 4:2:2 / 4:4:0 chroma remaps luma dirs via `conv422`/`conv440` ONCE (on
//!   plane 1, mutating the shared per-fb dir array; plane 2 reuses the
//!   remapped values) — cdef_block.c:361-369.
//!
//! Plane buffers are u16 at every bit depth. The walk requires the padded
//! layout the aom-decode tile driver produces: `stride >=
//! align16(mi_cols * 4) >> ss_x` and at least `(mi_rows * 4) >> ss_y` rows —
//! the line-buffer copies read full aligned-stride rows exactly like the C
//! (which reads into the YV12 border; the beyond-`mi_cols` columns never
//! influence output, a property cdef_frame_diff.rs pins by giving the two
//! sides DIFFERENT padding).

use crate::{cdef_filter_block_16, cdef_find_dir, CDEF_BSTRIDE, CDEF_VERY_LARGE};

pub const CDEF_SEC_STRENGTHS: i32 = 4;
const MI_SIZE_64X64: i32 = 16;
const MI_SIZE_LOG2: i32 = 2;
/// `CDEF_VBORDER` / `CDEF_HBORDER` (cdef_block.h).
const VB: usize = 2;
const HB: usize = 8;
/// `CDEF_NBLOCKS` = 128/8 (dir/var grids are sized for 128 SBs).
const NB: usize = 16;
/// `CDEF_INBUF_SIZE` = CDEF_BSTRIDE * (128 + 2*CDEF_VBORDER).
const INBUF_SIZE: usize = CDEF_BSTRIDE * (128 + 2 * VB);

/// Frame-level CDEF inputs (frame header + decoded per-mi facts).
pub struct CdefFrameParams<'a> {
    pub mi_rows: i32,
    pub mi_cols: i32,
    /// 1 (monochrome) or 3.
    pub num_planes: usize,
    /// Chroma subsampling (planes 1/2; plane 0 is never subsampled).
    pub ss_x: usize,
    pub ss_y: usize,
    pub bit_depth: i32,
    /// `cdef_info.cdef_damping` (3..=6).
    pub damping: i32,
    pub cdef_strengths: [i32; 8],
    pub cdef_uv_strengths: [i32; 8],
    /// Per-mi `skip_txfm`, `mi_rows x mi_cols`, row stride `mi_cols`.
    pub skip_txfm: &'a [bool],
    /// Per-64x64-fb decoded `cdef_strength` index, `nvfb x nhfb` raster
    /// (`nvfb/nhfb = ceil(mi_rows/16) / ceil(mi_cols/16)`); `-1` for an fb
    /// where no strength was read (all-skip) — the C NULL/-1 early-out arm.
    pub unit_strength: &'a [i32],
}

/// `ALIGN_POWER_OF_TWO(v, 4)` — the aligned luma row width the line buffers
/// use (`luma_stride` in cdef.c).
fn align16(v: i32) -> i32 {
    (v + 15) & !15
}

struct Planes<'p> {
    y: &'p mut [u16],
    y_stride: usize,
    u: &'p mut [u16],
    v: &'p mut [u16],
    uv_stride: usize,
}

impl Planes<'_> {
    fn get(&mut self, plane: usize) -> (&mut [u16], usize) {
        match plane {
            0 => (&mut *self.y, self.y_stride),
            1 => (&mut *self.u, self.uv_stride),
            _ => (&mut *self.v, self.uv_stride),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_rect(
    dst: &mut [u16],
    doff: usize,
    dstride: usize,
    src: &[u16],
    soff: usize,
    sstride: usize,
    v: usize,
    h: usize,
) {
    for i in 0..v {
        dst[doff + i * dstride..doff + i * dstride + h]
            .copy_from_slice(&src[soff + i * sstride..soff + i * sstride + h]);
    }
}

fn fill_rect(dst: &mut [u16], off: usize, dstride: usize, v: usize, h: usize, x: u16) {
    for i in 0..v {
        dst[off + i * dstride..off + i * dstride + h].fill(x);
    }
}

/// `usize` from a computed non-negative i32 offset.
fn uz(v: i32) -> usize {
    usize::try_from(v).expect("negative buffer offset")
}

/// Per-fb-row state (`CdefBlockInfo` + the row-level fields of the C walk).
struct FbInfo {
    /// Offsets into `linebuf[plane]` of the top (read slot) / bottom lines.
    top_off: [usize; 3],
    bot_off: [usize; 3],
    frame_boundary: [bool; 4], // TOP, LEFT, BOTTOM, RIGHT
    damping: i32,
    coeff_shift: i32,
    dir: [[i32; NB]; NB],
    var: [[i32; NB]; NB],
    /// dlist: the fb's non-skip 8x8 units as (by, bx).
    dlist: [(u8, u8); MI_SIZE_64X64 as usize * MI_SIZE_64X64 as usize / 4],
    cdef_count: usize,
}

const TOP: usize = 0;
const LEFT: usize = 1;
const BOTTOM: usize = 2;
const RIGHT: usize = 3;

/// `is_8x8_block_skip`: ALL mi of the 8x8 unit (2x2 mi at (mi_row, mi_col))
/// must have skip_txfm set.
fn is_8x8_block_skip(p: &CdefFrameParams, mi_row: i32, mi_col: i32) -> bool {
    for r in 0..2 {
        for c in 0..2 {
            if !p.skip_txfm[uz((mi_row + r) * p.mi_cols + mi_col + c)] {
                return false;
            }
        }
    }
    true
}

/// `av1_cdef_compute_sb_list` for BLOCK_64X64 at (16*fbr, 16*fbc).
fn compute_sb_list(p: &CdefFrameParams, fbr: i32, fbc: i32, fb: &mut FbInfo) -> usize {
    let mi_row = fbr * MI_SIZE_64X64;
    let mi_col = fbc * MI_SIZE_64X64;
    let maxc = (p.mi_cols - mi_col).min(MI_SIZE_64X64);
    let maxr = (p.mi_rows - mi_row).min(MI_SIZE_64X64);
    let mut count = 0usize;
    let mut r = 0;
    while r < maxr {
        let mut c = 0;
        while c < maxc {
            if !is_8x8_block_skip(p, mi_row + r, mi_col + c) {
                fb.dlist[count] = ((r >> 1) as u8, (c >> 1) as u8);
                count += 1;
            }
            c += 2;
        }
        r += 2;
    }
    count
}

/// `av1_cdef_init_fb_row` (single-threaded): row boundaries, damping/shift,
/// dir/var reset, and the ping-pong top/bottom line-buffer copies (saved
/// PRE-filter pixels of the last/first 2 rows around the next fb row).
#[allow(clippy::too_many_arguments)]
fn init_fb_row(
    p: &CdefFrameParams,
    planes: &mut Planes,
    linebuf: &mut [Vec<u16>; 3],
    fb: &mut FbInfo,
    fbr: i32,
    nvfb: i32,
    luma_stride: i32,
) {
    fb.frame_boundary[TOP] = fbr == 0;
    fb.frame_boundary[BOTTOM] = if fbr != nvfb - 1 {
        MI_SIZE_64X64 * (fbr + 1) == p.mi_rows
    } else {
        true
    };
    fb.damping = p.damping;
    fb.coeff_shift = (p.bit_depth - 8).max(0);
    fb.dir = [[0; NB]; NB];
    fb.var = [[0; NB]; NB];

    let ping_pong = (fbr & 1) as usize;
    // Indexed like the C per-plane loop (linebuf/top_off/bot_off in step).
    #[allow(clippy::needless_range_loop)]
    for plane in 0..p.num_planes {
        let (ss_x, ss_y) = if plane == 0 { (0, 0) } else { (p.ss_x, p.ss_y) };
        let mi_high_l2 = MI_SIZE_LOG2 - ss_y as i32;
        let offset = (MI_SIZE_64X64 * (fbr + 1)) << mi_high_l2;
        let stride = uz(luma_stride >> ss_x);
        let (buf, pstride) = planes.get(plane);
        // Write slot for the NEXT row's top border; read slot is the other one.
        let write_top = ping_pong * VB * stride;
        fb.top_off[plane] = (1 - ping_pong) * VB * stride;
        fb.bot_off[plane] = 2 * VB * stride;
        if fbr != nvfb - 1 {
            // top line buffer copy (last 2 rows of the current fb row).
            copy_rect(
                &mut linebuf[plane],
                write_top,
                stride,
                buf,
                uz(offset - VB as i32) * pstride,
                pstride,
                VB,
                stride,
            );
            // bottom line buffer copy (first 2 rows of the next fb row).
            let bot = fb.bot_off[plane];
            copy_rect(
                &mut linebuf[plane],
                bot,
                stride,
                buf,
                uz(offset) * pstride,
                pstride,
                VB,
                stride,
            );
        }
    }
}

/// Per-fb-col, per-plane geometry (`cdef_init_fb_col`).
struct ColInfo {
    level: i32,
    sec_strength: i32,
    xdec: usize,
    ydec: usize,
    mi_wide_l2: i32,
    mi_high_l2: i32,
    roffset: i32,
    coffset: i32,
}

/// `cdef_prepare_fb`: prime the 16-bit `src` block with the fb + its 5px
/// borders (8 left/right, 2 top/bottom) from frame / linebuf / colbuf, with
/// CDEF_VERY_LARGE at frame edges. Saves this fb's right columns into colbuf.
#[allow(clippy::too_many_arguments)]
fn prepare_fb(
    p: &CdefFrameParams,
    planes: &mut Planes,
    linebuf: &[Vec<u16>; 3],
    colbuf: &mut [Vec<u16>; 3],
    src: &mut [u16],
    fb: &FbInfo,
    ci: &ColInfo,
    cdef_left: bool,
    fbc: i32,
    fbr: i32,
    plane: usize,
    luma_stride: i32,
) {
    let nvfb = (p.mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    let nhfb = (p.mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    let cstart: i32 = if cdef_left { 0 } else { -(HB as i32) };
    let nhb = (p.mi_cols - MI_SIZE_64X64 * fbc).min(MI_SIZE_64X64);
    let nvb = (p.mi_rows - MI_SIZE_64X64 * fbr).min(MI_SIZE_64X64);
    let hsize = uz(nhb << ci.mi_wide_l2);
    let vsize = uz(nvb << ci.mi_high_l2);
    let bot_offset = (vsize + VB) * CDEF_BSTRIDE;
    let stride = uz(luma_stride >> if plane == 0 { 0 } else { p.ss_x });
    let cend = if fbc == nhfb - 1 {
        hsize as i32
    } else {
        (hsize + HB) as i32
    };
    let rend = if fbr == nvfb - 1 { vsize } else { vsize + VB };
    let very = CDEF_VERY_LARGE as u16;

    let (buf, pstride) = planes.get(plane);
    // Current superblock pixels (and the unfiltered right border; the left
    // border too when the left fb was NOT filtered).
    copy_rect(
        src,
        uz((VB * CDEF_BSTRIDE + HB) as i32 + cstart),
        CDEF_BSTRIDE,
        buf,
        uz(ci.roffset) * pstride + uz(ci.coffset + cstart),
        pstride,
        vsize,
        uz(cend - cstart),
    );

    // Bottom rows from the bottom line buffer (pre-filter pixels of the next
    // fb row), frame edge -> CDEF_VERY_LARGE.
    if fbr < nvfb - 1 {
        copy_rect(
            src,
            bot_offset + HB,
            CDEF_BSTRIDE,
            &linebuf[plane],
            fb.bot_off[plane] + uz(ci.coffset),
            stride,
            VB,
            hsize,
        );
    } else {
        fill_rect(src, bot_offset + HB, CDEF_BSTRIDE, VB, hsize, very);
    }
    if fbr < nvfb - 1 && fbc > 0 {
        copy_rect(
            src,
            bot_offset,
            CDEF_BSTRIDE,
            &linebuf[plane],
            fb.bot_off[plane] + uz(ci.coffset) - HB,
            stride,
            VB,
            HB,
        );
    } else {
        fill_rect(src, bot_offset, CDEF_BSTRIDE, VB, HB, very);
    }
    if fbr < nvfb - 1 && fbc < nhfb - 1 {
        copy_rect(
            src,
            bot_offset + hsize + HB,
            CDEF_BSTRIDE,
            &linebuf[plane],
            fb.bot_off[plane] + uz(ci.coffset) + hsize,
            stride,
            VB,
            HB,
        );
    } else {
        fill_rect(src, bot_offset + hsize + HB, CDEF_BSTRIDE, VB, HB, very);
    }

    // Top rows from the top line buffer (pre-filter pixels saved while
    // processing the row above), frame edge -> CDEF_VERY_LARGE.
    if fbr > 0 {
        copy_rect(
            src,
            HB,
            CDEF_BSTRIDE,
            &linebuf[plane],
            fb.top_off[plane] + uz(ci.coffset),
            stride,
            VB,
            hsize,
        );
    } else {
        fill_rect(src, HB, CDEF_BSTRIDE, VB, hsize, very);
    }
    if fbr > 0 && fbc > 0 {
        copy_rect(
            src,
            0,
            CDEF_BSTRIDE,
            &linebuf[plane],
            fb.top_off[plane] + uz(ci.coffset) - HB,
            stride,
            VB,
            HB,
        );
    } else {
        fill_rect(src, 0, CDEF_BSTRIDE, VB, HB, very);
    }
    if fbr > 0 && fbc < nhfb - 1 {
        copy_rect(
            src,
            hsize + HB,
            CDEF_BSTRIDE,
            &linebuf[plane],
            fb.top_off[plane] + uz(ci.coffset) + hsize,
            stride,
            VB,
            HB,
        );
    } else {
        fill_rect(src, hsize + HB, CDEF_BSTRIDE, VB, HB, very);
    }

    if cdef_left {
        // The left fb was filtered: its pre-filter right columns were saved
        // in colbuf — restore them as our left border.
        copy_rect(src, 0, CDEF_BSTRIDE, &colbuf[plane], 0, HB, rend + VB, HB);
    }
    // Save THIS fb's (still unfiltered) right columns for the fb to our right.
    copy_rect(
        &mut colbuf[plane],
        0,
        HB,
        src,
        hsize,
        CDEF_BSTRIDE,
        rend + VB,
        HB,
    );

    if fb.frame_boundary[LEFT] {
        fill_rect(src, 0, CDEF_BSTRIDE, vsize + 2 * VB, HB, very);
    }
    if fb.frame_boundary[RIGHT] {
        fill_rect(src, hsize + HB, CDEF_BSTRIDE, vsize + 2 * VB, HB, very);
    }
}

/// `adjust_strength` (cdef_block.c): variance-adaptive luma primary strength.
fn adjust_strength(strength: i32, var: i32) -> i32 {
    let i = if var >> 6 != 0 {
        (31 - ((var >> 6) as u32).leading_zeros() as i32).min(12)
    } else {
        0
    };
    if var != 0 {
        (strength * (4 + i) + 8) >> 4
    } else {
        0
    }
}

/// `av1_cdef_filter_fb` (decoder shape: the dst16/dirinit encoder-search arms
/// folded out; the u16 store is value-identical to the C u8 store for bd 8 —
/// see [`cdef_filter_block_16`]). `src` is the FULL work buffer; the interior
/// origin (`in` in C) sits at `VB*CDEF_BSTRIDE + HB` and the filter taps
/// reach backwards from it into the borders.
#[allow(clippy::too_many_arguments)]
fn filter_fb(
    dst: &mut [u16],
    dst_off: usize,
    dstride: usize,
    src: &[u16],
    fb: &mut FbInfo,
    ci: &ColInfo,
    plane: usize,
    cdef_count: usize,
) {
    let in_base = VB * CDEF_BSTRIDE + HB;
    let coeff_shift = fb.coeff_shift;
    let pri_strength = ci.level << coeff_shift;
    let sec_strength = ci.sec_strength << coeff_shift;
    let damping = fb.damping + coeff_shift - i32::from(plane != 0);
    let bw_log2 = 3 - ci.xdec;
    let bh_log2 = 3 - ci.ydec;

    if plane == 0 {
        // aom_cdef_find_dir over the dlist (dual = two independent singles).
        for bi in 0..cdef_count {
            let (by, bx) = (fb.dlist[bi].0 as usize, fb.dlist[bi].1 as usize);
            let pos = in_base + 8 * by * CDEF_BSTRIDE + 8 * bx;
            let (dir, var) = cdef_find_dir(&src[pos..], CDEF_BSTRIDE, coeff_shift);
            fb.dir[by][bx] = dir;
            fb.var[by][bx] = var;
        }
    }
    if plane == 1 && ci.xdec != ci.ydec {
        // 4:2:2 / 4:4:0: remap luma dirs ONCE (plane 2 sees the remapped values).
        const CONV422: [i32; 8] = [7, 0, 2, 4, 5, 6, 6, 6];
        const CONV440: [i32; 8] = [1, 2, 2, 2, 3, 4, 6, 0];
        for bi in 0..cdef_count {
            let (by, bx) = (fb.dlist[bi].0 as usize, fb.dlist[bi].1 as usize);
            let d = fb.dir[by][bx] as usize;
            fb.dir[by][bx] = if ci.xdec != 0 { CONV422[d] } else { CONV440[d] };
        }
    }

    let block_width = 8 >> ci.xdec;
    let block_height = 8 >> ci.ydec;
    for bi in 0..cdef_count {
        let (by, bx) = (fb.dlist[bi].0 as usize, fb.dlist[bi].1 as usize);
        let t = if plane != 0 {
            pri_strength
        } else {
            adjust_strength(pri_strength, fb.var[by][bx])
        };
        // strength_index selects which tap families are live:
        //   0: pri+sec, 1: pri only, 2: sec only, 3: neither.
        let enable_primary = t != 0;
        let enable_secondary = sec_strength != 0;
        cdef_filter_block_16(
            dst,
            dst_off + (by << bh_log2) * dstride + (bx << bw_log2),
            dstride,
            src,
            in_base + ((by * CDEF_BSTRIDE) << bh_log2) + (bx << bw_log2),
            t,
            sec_strength,
            if pri_strength != 0 { fb.dir[by][bx] } else { 0 },
            damping,
            damping,
            coeff_shift,
            block_width,
            block_height,
            enable_primary,
            enable_secondary,
        );
    }
}

/// `cdef_fb_col`: strength selection, skip-list, then per-plane
/// prepare+filter. Updates `cdef_left` per plane.
#[allow(clippy::too_many_arguments)]
fn fb_col(
    p: &CdefFrameParams,
    planes: &mut Planes,
    linebuf: &[Vec<u16>; 3],
    colbuf: &mut [Vec<u16>; 3],
    src: &mut [u16],
    fb: &mut FbInfo,
    cdef_left: &mut [bool; 3],
    fbc: i32,
    fbr: i32,
    luma_stride: i32,
) {
    let nhfb = (p.mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    let strength_idx = p.unit_strength[uz(fbr * nhfb + fbc)];
    if strength_idx < 0 {
        cdef_left[..p.num_planes].fill(false);
        return;
    }
    let si = strength_idx as usize;

    // level/sec split, PLANE_TYPE_Y then _UV; sec 3 promotes to 4.
    let mut level = [0i32; 2];
    let mut sec = [0i32; 2];
    let mut is_zero = [true; 2];
    level[0] = p.cdef_strengths[si] / CDEF_SEC_STRENGTHS;
    sec[0] = p.cdef_strengths[si] % CDEF_SEC_STRENGTHS;
    sec[0] += i32::from(sec[0] == 3);
    is_zero[0] = level[0] == 0 && sec[0] == 0;
    if p.num_planes > 1 {
        level[1] = p.cdef_uv_strengths[si] / CDEF_SEC_STRENGTHS;
        sec[1] = p.cdef_uv_strengths[si] % CDEF_SEC_STRENGTHS;
        sec[1] += i32::from(sec[1] == 3);
        is_zero[1] = level[1] == 0 && sec[1] == 0;
    }
    if is_zero[0] && is_zero[1] {
        cdef_left[..p.num_planes].fill(false);
        return;
    }

    fb.cdef_count = compute_sb_list(p, fbr, fbc, fb);
    if fb.cdef_count == 0 {
        cdef_left[..p.num_planes].fill(false);
        return;
    }
    let cdef_count = fb.cdef_count;

    // Indexed like the C per-plane loop (cdef_left updated in step).
    #[allow(clippy::needless_range_loop)]
    for plane in 0..p.num_planes {
        let plane_type = usize::from(plane > 0);
        // Luma always runs (directions are computed on it); zero-level chroma
        // is skipped.
        if plane != 0 && is_zero[plane_type] {
            cdef_left[plane] = false;
            continue;
        }
        let (ss_x, ss_y) = if plane == 0 { (0, 0) } else { (p.ss_x, p.ss_y) };
        let ci = ColInfo {
            level: level[plane_type],
            sec_strength: sec[plane_type],
            xdec: ss_x,
            ydec: ss_y,
            mi_wide_l2: MI_SIZE_LOG2 - ss_x as i32,
            mi_high_l2: MI_SIZE_LOG2 - ss_y as i32,
            roffset: (MI_SIZE_64X64 * fbr) << (MI_SIZE_LOG2 - ss_y as i32),
            coffset: (MI_SIZE_64X64 * fbc) << (MI_SIZE_LOG2 - ss_x as i32),
        };
        prepare_fb(
            p,
            planes,
            linebuf,
            colbuf,
            src,
            fb,
            &ci,
            cdef_left[plane],
            fbc,
            fbr,
            plane,
            luma_stride,
        );
        let (buf, pstride) = planes.get(plane);
        filter_fb(
            buf,
            uz(ci.roffset) * pstride + uz(ci.coffset),
            pstride,
            src,
            fb,
            &ci,
            plane,
            cdef_count,
        );
        cdef_left[plane] = true;
    }
}

/// `av1_cdef_frame` — apply CDEF to the (deblocked) frame planes in place.
///
/// `y`/`u`/`v` are the mi-aligned reconstruction planes (u16 at every bit
/// depth); `u`/`v` are ignored when `num_planes == 1`. Buffer requirements
/// (asserted): `y_stride >= align16(mi_cols * 4)`, `uv_stride >= align16(
/// mi_cols * 4) >> ss_x`, with at least `mi_rows * 4` (>> ss_y) rows.
pub fn cdef_frame(
    y: &mut [u16],
    y_stride: usize,
    u: &mut [u16],
    v: &mut [u16],
    uv_stride: usize,
    p: &CdefFrameParams,
) {
    let nvfb = (p.mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    let nhfb = (p.mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    let luma_stride = align16(p.mi_cols << MI_SIZE_LOG2);

    assert!(p.num_planes == 1 || p.num_planes == 3);
    assert_eq!(p.skip_txfm.len(), uz(p.mi_rows * p.mi_cols));
    assert_eq!(p.unit_strength.len(), uz(nvfb * nhfb));
    assert!(y_stride >= uz(luma_stride));
    assert!(y.len() >= y_stride * uz(p.mi_rows << MI_SIZE_LOG2));
    if p.num_planes > 1 {
        assert!(uv_stride >= uz(luma_stride >> p.ss_x));
        let uv_rows = uz(p.mi_rows << MI_SIZE_LOG2) >> p.ss_y;
        assert!(u.len() >= uv_stride * uv_rows && v.len() >= uv_stride * uv_rows);
    }

    let mut planes = Planes {
        y,
        y_stride,
        u,
        v,
        uv_stride,
    };
    // Work buffers, laid out as the single-threaded C allocates them:
    // linebuf = ping-pong top slots + the bottom slot (3 x VB lines of the
    // plane's aligned stride); colbuf = (64 + 2*VB + VB) x HB is enough for
    // rend + VB rows; srcbuf = CDEF_INBUF_SIZE. Initial contents are never
    // consumed (every read region is written first — same property as the C's
    // malloc'd buffers).
    let mut linebuf: [Vec<u16>; 3] = [const { Vec::new() }; 3];
    let mut colbuf: [Vec<u16>; 3] = [const { Vec::new() }; 3];
    for plane in 0..p.num_planes {
        let ss_x = if plane == 0 { 0 } else { p.ss_x };
        linebuf[plane] = vec![0u16; 3 * VB * uz(luma_stride >> ss_x)];
        colbuf[plane] = vec![0u16; (64 + 3 * VB) * HB];
    }
    let mut src = vec![0u16; INBUF_SIZE];
    let mut fb = FbInfo {
        top_off: [0; 3],
        bot_off: [0; 3],
        frame_boundary: [false; 4],
        damping: 0,
        coeff_shift: 0,
        dir: [[0; NB]; NB],
        var: [[0; NB]; NB],
        dlist: [(0, 0); MI_SIZE_64X64 as usize * MI_SIZE_64X64 as usize / 4],
        cdef_count: 0,
    };

    for fbr in 0..nvfb {
        // av1_cdef_fb_row: cdef_left starts TRUE each row (the fbc==0 colbuf
        // restore it triggers is then fully overwritten by the LEFT edge fill).
        let mut cdef_left = [true; 3];
        init_fb_row(
            p,
            &mut planes,
            &mut linebuf,
            &mut fb,
            fbr,
            nvfb,
            luma_stride,
        );
        for fbc in 0..nhfb {
            fb.frame_boundary[LEFT] = fbc == 0;
            fb.frame_boundary[RIGHT] = if fbc != nhfb - 1 {
                MI_SIZE_64X64 * (fbc + 1) == p.mi_cols
            } else {
                true
            };
            fb_col(
                p,
                &mut planes,
                &linebuf,
                &mut colbuf,
                &mut src,
                &mut fb,
                &mut cdef_left,
                fbc,
                fbr,
                luma_stride,
            );
        }
    }
}
