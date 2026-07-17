//! The whole-frame loop-restoration walk (single-threaded decoder path):
//! `av1_loop_restoration_filter_frame` and
//! `av1_loop_restoration_save_boundary_lines` (av1/common/restoration.c) on
//! u16 planes.
//!
//! Striped processing: each plane filters in 64-luma-pixel stripes offset up
//! by `RESTORATION_UNIT_OFFSET = 8` (matching CDEF's block grid). The 3
//! context rows above/below each stripe are NOT the frame's neighbouring rows
//! but saved boundary lines: 2 rows of DEBLOCKED (pre-CDEF) pixels at
//! internal stripe boundaries (expanded 0,0,1 / 0,1,1 to 3 rows), and CDEF
//! output rows at the frame's top/bottom edges. `save_boundary_lines` runs
//! TWICE in the decoder (decodeframe.c:5437-5460): `after_cdef=0` on the
//! deblocked frame before CDEF (internal boundaries), `after_cdef=1` on the
//! CDEF output (frame edges) — then `filter_frame` temporarily swaps those
//! rows into the source around each stripe (`setup/restore_processing_stripe_
//! boundary`). With no CDEF and no superres the decoder takes the
//! `optimized_lr` arm instead: no boundary saves; the stripe context is the
//! frame's own rows with only the ±3rd row duplicated from the ±2nd.
//!
//! The C operates on bordered YV12 buffers; this port stages each plane into
//! a padded working buffer (the filters read up to 4 samples past a unit and
//! the Wiener stripe filter rounds its width up to a multiple of 16, reading
//! /writing up to 18/15 columns past the plane's right edge — dead values,
//! but the memory must exist).

use crate::sgr::apply_selfguided_restoration;
use crate::wiener::wiener_convolve_add_src;
use aom_entropy::lr::{
    LrFrameConfig, LrUnitInfo, RESTORATION_PROC_UNIT_SIZE, RESTORATION_UNIT_OFFSET, RESTORE_NONE,
    RESTORE_SGRPROJ, RESTORE_WIENER,
};

/// `RESTORATION_BORDER` / `RESTORATION_CTX_VERT` / `RESTORATION_EXTRA_HORZ`.
const RESTORATION_BORDER: usize = 3;
const RESTORATION_CTX_VERT: usize = 2;
const RESTORATION_EXTRA_HORZ: usize = 4;

/// Working-buffer margins. Horizontal covers the Wiener rounded-width
/// overhang (reads to plane col +18, writes to +14, both dead) plus the ±4
/// stripe-swap columns; vertical covers the ±3 context rows + the tap-7 row.
pub(crate) const MARGIN_H: usize = 32;
pub(crate) const MARGIN_V: usize = 8;

/// One plane's restoration inputs: the CDEF output (filter source), the
/// deblocked pre-CDEF pixels (internal stripe boundary context), the decoded
/// per-RU parameters, and the plane geometry.
pub struct LrPlaneInput<'a> {
    /// Post-CDEF (current) plane, `stride`-strided, filtered IN PLACE.
    pub cur: &'a mut [u16],
    /// Deblocked pre-CDEF plane (same dims; may alias `cur`'s content when
    /// CDEF did not run). Only read when `!optimized_lr`.
    pub deblocked: &'a [u16],
    pub stride: usize,
    /// Per-RU parameters in unit-grid raster order (`horz*vert` entries).
    pub units: &'a [LrUnitInfo],
}

/// `av1_loop_restoration_save_boundary_lines` + `av1_loop_restoration_filter_
/// frame` in the decoder's exact ordering, for all restored planes.
/// `optimized_lr` = the decoder's `!do_cdef && !do_superres`.
pub fn loop_restoration_filter_frame(
    planes: &mut [LrPlaneInput<'_>],
    lr: &LrFrameConfig,
    ss_x: usize,
    ss_y: usize,
    bit_depth: i32,
    optimized_lr: bool,
) {
    for (plane, p) in planes.iter_mut().enumerate() {
        if lr.frame_restoration_type[plane] == RESTORE_NONE {
            continue;
        }
        filter_plane(p, lr, plane, ss_x, ss_y, bit_depth, optimized_lr);
    }
}

pub(crate) struct StripeBoundaries {
    pub(crate) above: Vec<u16>,
    pub(crate) below: Vec<u16>,
    pub(crate) stride: usize,
}

/// `save_boundary_lines` geometry + both passes for one plane, into u16
/// boundary buffers (`av1_alloc_restoration_buffers` sizing: stripes counted
/// on the LUMA extent, stride 32-aligned incl. the ±4 extension columns).
pub(crate) fn save_boundary_lines(
    b: &mut StripeBoundaries,
    src: &[u16],
    src_stride: usize,
    plane_w: usize,
    plane_h: usize,
    ss_y: usize,
    after_cdef: bool,
) {
    let stripe_height = (RESTORATION_PROC_UNIT_SIZE >> ss_y) as usize;
    let stripe_off = (RESTORATION_UNIT_OFFSET >> ss_y) as usize;
    let mut stripe_idx = 0usize;
    loop {
        let y0 = (stripe_idx * stripe_height).saturating_sub(stripe_off);
        if y0 >= plane_h {
            break;
        }
        let y1 = ((stripe_idx + 1) * stripe_height - stripe_off).min(plane_h);
        // Deblocked context at internal boundaries, CDEF context at frame
        // edges (plane_height == the crop height here — superres unscaled).
        let use_deblock_above = stripe_idx > 0;
        let use_deblock_below = y1 < plane_h;
        if !after_cdef {
            if use_deblock_above {
                save_deblock_lines(
                    b,
                    src,
                    src_stride,
                    plane_w,
                    plane_h,
                    y0 - RESTORATION_CTX_VERT,
                    stripe_idx,
                    true,
                );
            }
            if use_deblock_below {
                save_deblock_lines(b, src, src_stride, plane_w, plane_h, y1, stripe_idx, false);
            }
        } else {
            if !use_deblock_above {
                save_cdef_lines(b, src, src_stride, plane_w, y0, stripe_idx, true);
            }
            if !use_deblock_below {
                save_cdef_lines(b, src, src_stride, plane_w, y1 - 1, stripe_idx, false);
            }
        }
        stripe_idx += 1;
    }
}

/// `extend_lines`: replicate `RESTORATION_EXTRA_HORZ` columns on each side of
/// `RESTORATION_CTX_VERT` boundary rows starting at `row0`, columns offset by
/// `RESTORATION_EXTRA_HORZ` in the buffer.
fn extend_lines(buf: &mut [u16], row0: usize, stride: usize, width: usize) {
    for r in 0..RESTORATION_CTX_VERT {
        let base = (row0 + r) * stride + RESTORATION_EXTRA_HORZ;
        let first = buf[base];
        let last = buf[base + width - 1];
        for e in 1..=RESTORATION_EXTRA_HORZ {
            buf[base - e] = first;
            buf[base + width - 1 + e] = last;
        }
    }
}

/// `save_deblock_boundary_lines` (superres-unscaled arm): up to 2 deblocked
/// rows at `row`, clamped against the crop bottom (a stripe can end 1px above
/// it — then the one row is duplicated), extended ±4.
#[allow(clippy::too_many_arguments)]
fn save_deblock_lines(
    b: &mut StripeBoundaries,
    src: &[u16],
    src_stride: usize,
    plane_w: usize,
    plane_h: usize,
    row: usize,
    stripe: usize,
    is_above: bool,
) {
    let buf = if is_above { &mut b.above } else { &mut b.below };
    let row0 = RESTORATION_CTX_VERT * stripe;
    let lines_to_save = RESTORATION_CTX_VERT.min(plane_h - row);
    debug_assert!(lines_to_save == 1 || lines_to_save == 2);
    for i in 0..lines_to_save {
        let d = (row0 + i) * b.stride + RESTORATION_EXTRA_HORZ;
        buf[d..d + plane_w].copy_from_slice(&src[(row + i) * src_stride..][..plane_w]);
    }
    if lines_to_save == 1 {
        let s0 = row0 * b.stride + RESTORATION_EXTRA_HORZ;
        let d1 = (row0 + 1) * b.stride + RESTORATION_EXTRA_HORZ;
        buf.copy_within(s0..s0 + plane_w, d1);
    }
    extend_lines(buf, row0, b.stride, plane_w);
}

/// `save_cdef_boundary_lines`: the single CDEF row at `row` copied into both
/// context lines, extended ±4.
fn save_cdef_lines(
    b: &mut StripeBoundaries,
    src: &[u16],
    src_stride: usize,
    plane_w: usize,
    row: usize,
    stripe: usize,
    is_above: bool,
) {
    let buf = if is_above { &mut b.above } else { &mut b.below };
    let row0 = RESTORATION_CTX_VERT * stripe;
    for i in 0..RESTORATION_CTX_VERT {
        let d = (row0 + i) * b.stride + RESTORATION_EXTRA_HORZ;
        buf[d..d + plane_w].copy_from_slice(&src[row * src_stride..][..plane_w]);
    }
    extend_lines(buf, row0, b.stride, plane_w);
}

/// Working-buffer coordinates: plane (row, col) — both possibly negative /
/// past the plane — to a padded-buffer index.
#[inline]
pub(crate) fn at(w_stride: usize, row: isize, col: isize) -> usize {
    ((row + MARGIN_V as isize) * w_stride as isize + col + MARGIN_H as isize) as usize
}

/// `av1_extend_frame`: replicate a `RESTORATION_BORDER`-pixel border around
/// the `[0, w) x [0, h)` plane in the working buffer.
pub(crate) fn extend_frame(buf: &mut [u16], w: usize, h: usize, w_stride: usize) {
    const B: isize = RESTORATION_BORDER as isize;
    for r in 0..h as isize {
        let first = buf[at(w_stride, r, 0)];
        let last = buf[at(w_stride, r, w as isize - 1)];
        for e in 1..=B {
            buf[at(w_stride, r, -e)] = first;
            buf[at(w_stride, r, w as isize - 1 + e)] = last;
        }
    }
    for e in 1..=B {
        let top_src = at(w_stride, 0, -B);
        let top_dst = at(w_stride, -e, -B);
        buf.copy_within(top_src..top_src + w + 2 * B as usize, top_dst);
        let bot_src = at(w_stride, h as isize - 1, -B);
        let bot_dst = at(w_stride, h as isize - 1 + e, -B);
        buf.copy_within(bot_src..bot_src + w + 2 * B as usize, bot_dst);
    }
}

/// One plane: stage into the padded working buffer, build boundaries, run
/// the unit walk into a padded dst, copy the crop back.
fn filter_plane(
    p: &mut LrPlaneInput<'_>,
    lr: &LrFrameConfig,
    plane: usize,
    ss_x: usize,
    ss_y: usize,
    bit_depth: i32,
    optimized_lr: bool,
) {
    let (pw, ph) = lr.plane_size(plane, ss_x, ss_y);
    let (pw, ph) = (pw as usize, ph as usize);
    let (sx, sy) = if plane > 0 { (ss_x, ss_y) } else { (0, 0) };

    // --- boundary buffers (av1_alloc_restoration_buffers geometry) ---
    // Stripes are counted on the luma extent: ext_h = 8 + mi-aligned height.
    let mi_h = (lr.crop_height + 7) & !7; // set_mb_mi 8-px alignment, in px
    let ext_h = RESTORATION_UNIT_OFFSET + mi_h;
    let num_stripes = ((ext_h + 63) / 64) as usize;
    let b_stride = (pw + 2 * RESTORATION_EXTRA_HORZ + 31) & !31;
    let mut bnd = StripeBoundaries {
        above: vec![0; num_stripes * RESTORATION_CTX_VERT * b_stride],
        below: vec![0; num_stripes * RESTORATION_CTX_VERT * b_stride],
        stride: b_stride,
    };
    if !optimized_lr {
        // decodeframe.c ordering: deblocked pixels (pre-CDEF) feed internal
        // stripe boundaries; the CDEF output feeds the frame-edge context.
        save_boundary_lines(&mut bnd, p.deblocked, p.stride, pw, ph, sy, false);
        save_boundary_lines(&mut bnd, p.cur, p.stride, pw, ph, sy, true);
    }

    // --- working src (extended) + dst ---
    let w_stride = pw + 2 * MARGIN_H;
    let mut src = vec![0u16; w_stride * (ph + 2 * MARGIN_V)];
    for r in 0..ph {
        src[at(w_stride, r as isize, 0)..at(w_stride, r as isize, pw as isize)]
            .copy_from_slice(&p.cur[r * p.stride..][..pw]);
    }
    extend_frame(&mut src, pw, ph, w_stride);
    let mut dst = vec![0u16; w_stride * (ph + 2 * MARGIN_V)];

    // --- the unit walk (foreach_rest_unit_in_plane) ---
    let unit_size = lr.unit_size[plane];
    let (hu, _vu) = lr.plane_units(plane, ss_x, ss_y);
    let ext_size = unit_size * 3 / 2;
    let voffset = RESTORATION_UNIT_OFFSET >> sy;
    let mut y0 = 0i32;
    let mut row_number = 0i32;
    while y0 < ph as i32 {
        let remaining_h = ph as i32 - y0;
        let h = if remaining_h < ext_size {
            remaining_h
        } else {
            unit_size
        };
        let mut v_start = y0;
        let mut v_end = y0 + h;
        debug_assert!(v_end <= ph as i32);
        // Offset upwards to align with the restoration processing stripe.
        v_start = (v_start - voffset).max(0);
        if v_end < ph as i32 {
            v_end -= voffset;
        }

        // av1_foreach_rest_unit_in_row
        let mut x0 = 0i32;
        let mut j = 0i32;
        while x0 < pw as i32 {
            let remaining_w = pw as i32 - x0;
            let w = if remaining_w < ext_size {
                remaining_w
            } else {
                unit_size
            };
            let unit_idx = (row_number * hu + j) as usize;
            filter_unit(
                &mut src,
                &mut dst,
                w_stride,
                &p.units[unit_idx],
                &bnd,
                ph,
                sx,
                sy,
                bit_depth,
                (v_start, v_end, x0, x0 + w),
                optimized_lr,
            );
            x0 += w;
            j += 1;
        }
        y0 += h;
        row_number += 1;
    }

    // loop_restoration_copy_planes: the filtered crop back to the caller.
    for r in 0..ph {
        p.cur[r * p.stride..][..pw].copy_from_slice(
            &dst[at(w_stride, r as isize, 0)..at(w_stride, r as isize, pw as isize)],
        );
    }
}

/// `av1_loop_restoration_filter_unit`: the per-unit stripe loop with boundary
/// row swapping. `limits = (v_start, v_end, h_start, h_end)` in plane coords.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_unit(
    src: &mut [u16],
    dst: &mut [u16],
    w_stride: usize,
    rui: &LrUnitInfo,
    rsb: &StripeBoundaries,
    plane_h: usize,
    ss_x: usize,
    ss_y: usize,
    bit_depth: i32,
    limits: (i32, i32, i32, i32),
    optimized_lr: bool,
) {
    let (v_start, v_end, h_start, h_end) = limits;
    let unit_h = (v_end - v_start) as usize;
    let unit_w = (h_end - h_start) as usize;

    if rui.restoration_type == RESTORE_NONE {
        // copy_rest_unit
        for r in 0..unit_h {
            let s = at(w_stride, (v_start + r as i32) as isize, h_start as isize);
            let row = src[s..s + unit_w].to_vec();
            dst[s..s + unit_w].copy_from_slice(&row);
        }
        return;
    }

    let procunit_width = (RESTORATION_PROC_UNIT_SIZE >> ss_x) as usize;
    let full_stripe_height = RESTORATION_PROC_UNIT_SIZE >> ss_y;
    let runit_offset = RESTORATION_UNIT_OFFSET >> ss_y;

    let mut i = 0i32;
    while i < unit_h as i32 {
        let rs_v_start = v_start + i;

        // get_stripe_boundary_info
        let first_stripe_in_plane = rs_v_start == 0;
        let this_stripe_height = full_stripe_height
            - if first_stripe_in_plane {
                runit_offset
            } else {
                0
            };
        let last_stripe_in_plane = rs_v_start + this_stripe_height >= plane_h as i32;
        let copy_above = !first_stripe_in_plane;
        let copy_below = !last_stripe_in_plane;

        // This stripe's slot in the boundary buffers.
        let frame_stripe = (rs_v_start + runit_offset) / full_stripe_height;
        let rsb_row = RESTORATION_CTX_VERT * frame_stripe as usize;

        // Topmost frame stripe is 8 luma px shorter; never past the unit.
        let nominal_stripe_height =
            full_stripe_height - if frame_stripe == 0 { runit_offset } else { 0 };
        let h = nominal_stripe_height.min(v_end - rs_v_start) as usize;

        // setup_processing_stripe_boundary: swap the context rows
        // above/below the stripe for the saved boundary lines (optimized:
        // duplicate the ±2nd row into the ±3rd), saving the originals.
        let line_width = unit_w + 2 * RESTORATION_EXTRA_HORZ;
        let data_x0 = h_start as isize - RESTORATION_EXTRA_HORZ as isize;
        let mut tmp_above: [Vec<u16>; RESTORATION_BORDER] = Default::default();
        let mut tmp_below: [Vec<u16>; RESTORATION_BORDER] = Default::default();
        let stripe_end = rs_v_start + h as i32;
        if !optimized_lr {
            if copy_above {
                for (bi, tmp) in tmp_above.iter_mut().enumerate() {
                    let i_off = bi as isize - RESTORATION_BORDER as isize; // -3..-1
                                                                           // buf_row: the 2 saved rows expand 0,0,1 over rows -3..-1.
                    let buf_row = (rsb_row as isize
                        + (i_off + RESTORATION_CTX_VERT as isize).max(0))
                        as usize;
                    // Boundary-buffer origin is offset EXTRA_HORZ: plane col
                    // (h_start - 4) is buffer index h_start + row*stride.
                    let buf0 = buf_row * rsb.stride + h_start as usize;
                    let d = at(w_stride, rs_v_start as isize + i_off, data_x0);
                    *tmp = src[d..d + line_width].to_vec();
                    src[d..d + line_width].copy_from_slice(&rsb.above[buf0..buf0 + line_width]);
                }
            }
            if copy_below {
                for (bi, tmp) in tmp_below.iter_mut().enumerate() {
                    // The second saved row repeats: 0,1,1 over rows 0..2.
                    let buf_row = rsb_row + bi.min(RESTORATION_CTX_VERT - 1);
                    let buf0 = buf_row * rsb.stride + h_start as usize;
                    let d = at(w_stride, (stripe_end + bi as i32) as isize, data_x0);
                    *tmp = src[d..d + line_width].to_vec();
                    src[d..d + line_width].copy_from_slice(&rsb.below[buf0..buf0 + line_width]);
                }
            }
        } else {
            if copy_above {
                let d = at(w_stride, rs_v_start as isize - 3, data_x0);
                let s = at(w_stride, rs_v_start as isize - 2, data_x0);
                tmp_above[0] = src[d..d + line_width].to_vec();
                let row = src[s..s + line_width].to_vec();
                src[d..d + line_width].copy_from_slice(&row);
            }
            if copy_below {
                let d = at(w_stride, stripe_end as isize + 2, data_x0);
                let s = at(w_stride, stripe_end as isize + 1, data_x0);
                tmp_below[2] = src[d..d + line_width].to_vec();
                let row = src[s..s + line_width].to_vec();
                src[d..d + line_width].copy_from_slice(&row);
            }
        }

        // stripe_filter: Wiener rounds each chunk's width up to a multiple
        // of 16 (over-writing into the next unit / padding — dead values);
        // SGR uses the exact remaining width.
        let row0 = at(w_stride, rs_v_start as isize, h_start as isize);
        match rui.restoration_type {
            RESTORE_WIENER => {
                let mut j = 0usize;
                while j < unit_w {
                    let w = procunit_width.min((unit_w - j + 15) & !15);
                    wiener_convolve_add_src(
                        src,
                        row0 + j,
                        w_stride,
                        dst,
                        row0 + j,
                        w_stride,
                        &rui.wiener.hfilter,
                        &rui.wiener.vfilter,
                        w,
                        h,
                        bit_depth,
                    );
                    j += procunit_width;
                }
            }
            _ => {
                debug_assert_eq!(rui.restoration_type, RESTORE_SGRPROJ);
                let mut j = 0usize;
                while j < unit_w {
                    let w = procunit_width.min(unit_w - j);
                    apply_selfguided_restoration(
                        src,
                        row0 + j,
                        w_stride,
                        w,
                        h,
                        rui.sgrproj.ep as usize,
                        &rui.sgrproj.xqd,
                        dst,
                        row0 + j,
                        w_stride,
                        bit_depth,
                    );
                    j += procunit_width;
                }
            }
        }

        // restore_processing_stripe_boundary
        if !optimized_lr {
            if copy_above {
                for (bi, tmp) in tmp_above.iter().enumerate() {
                    let i_off = bi as isize - RESTORATION_BORDER as isize;
                    let d = at(w_stride, rs_v_start as isize + i_off, data_x0);
                    src[d..d + line_width].copy_from_slice(tmp);
                }
            }
            if copy_below {
                for (bi, tmp) in tmp_below.iter().enumerate() {
                    if stripe_end + bi as i32 >= v_end + RESTORATION_BORDER as i32 {
                        break;
                    }
                    let d = at(w_stride, (stripe_end + bi as i32) as isize, data_x0);
                    src[d..d + line_width].copy_from_slice(tmp);
                }
            }
        } else {
            if copy_above {
                let d = at(w_stride, rs_v_start as isize - 3, data_x0);
                src[d..d + line_width].copy_from_slice(&tmp_above[0]);
            }
            if copy_below && stripe_end + 2 < v_end + RESTORATION_BORDER as i32 {
                let d = at(w_stride, stripe_end as isize + 2, data_x0);
                src[d..d + line_width].copy_from_slice(&tmp_below[2]);
            }
        }

        i += h as i32;
    }
}
