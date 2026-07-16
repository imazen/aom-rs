//! Loop-restoration per-unit parameter syntax — the tile-data side of AV1
//! loop restoration (`av1/decoder/decodeframe.c`): the finite-subexponential
//! read primitives on the arithmetic coder (`aom_dsp/binary_codes_reader.c` /
//! `aom_dsp/recenter.h`), `read_wiener_filter`, `read_sgrproj_filter`, and
//! `loop_restoration_read_sb_coeffs` (the per-restoration-unit dispatch the
//! tile decoder runs at each superblock root).
//!
//! Everything here is decode-side; the values feed `aom-restore`'s kernels.

use crate::cdf::{read_bit, read_literal, read_symbol, update_cdf, write_literal};
use crate::dec::OdEcDec;
use crate::enc::OdEcEnc;

/// `RESTORE_*` (av1/common/enums.h `RestorationType`).
pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_WIENER: u8 = 1;
pub const RESTORE_SGRPROJ: u8 = 2;
pub const RESTORE_SWITCHABLE: u8 = 3;
/// `RESTORE_SWITCHABLE_TYPES` — the per-unit alphabet in a SWITCHABLE frame.
pub const RESTORE_SWITCHABLE_TYPES: usize = 3;

/// `WIENER_WIN` / `WIENER_WIN_CHROMA` / `WIENER_HALFWIN` (restoration.h).
pub const WIENER_WIN: usize = 7;
pub const WIENER_WIN_CHROMA: usize = 5;
pub const WIENER_HALFWIN: usize = 3;

// Wiener tap bounds (restoration.h): MIDV 3/-7/15, BITS 4/5/6.
pub const WIENER_FILT_TAP0_MINV: i32 = 3 - (1 << 4) / 2; // -5
pub const WIENER_FILT_TAP0_MAXV: i32 = 3 - 1 + (1 << 4) / 2; // 10
pub const WIENER_FILT_TAP1_MINV: i32 = -7 - (1 << 5) / 2; // -23
pub const WIENER_FILT_TAP1_MAXV: i32 = -7 - 1 + (1 << 5) / 2; // 8
pub const WIENER_FILT_TAP2_MINV: i32 = 15 - (1 << 6) / 2; // -17
pub const WIENER_FILT_TAP2_MAXV: i32 = 15 - 1 + (1 << 6) / 2; // 46
pub const WIENER_FILT_TAP0_SUBEXP_K: u16 = 1;
pub const WIENER_FILT_TAP1_SUBEXP_K: u16 = 2;
pub const WIENER_FILT_TAP2_SUBEXP_K: u16 = 3;

/// `SGRPROJ_*` coding constants (restoration.h).
pub const SGRPROJ_PARAMS_BITS: u32 = 4;
pub const SGRPROJ_PRJ_BITS: i32 = 7;
pub const SGRPROJ_PRJ_MIN0: i32 = -(1 << SGRPROJ_PRJ_BITS) * 3 / 4; // -96
pub const SGRPROJ_PRJ_MAX0: i32 = SGRPROJ_PRJ_MIN0 + (1 << SGRPROJ_PRJ_BITS) - 1; // 31
pub const SGRPROJ_PRJ_MIN1: i32 = -(1 << SGRPROJ_PRJ_BITS) / 4; // -32
pub const SGRPROJ_PRJ_MAX1: i32 = SGRPROJ_PRJ_MIN1 + (1 << SGRPROJ_PRJ_BITS) - 1; // 95
pub const SGRPROJ_PRJ_SUBEXP_K: u16 = 4;

/// `av1_sgr_params[SGRPROJ_PARAMS].r` — the two box radii per `ep` index
/// (restoration.c). Radius 0 disables that pass; the reader only needs the
/// radii (the `s` values live with the kernel in aom-restore).
pub const SGR_PARAMS_R: [[i32; 2]; 16] = [
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [2, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [2, 0],
    [2, 0],
];

/// `WienerInfo` (blockd.h): 8-slot symmetric taps; slot 3 is the centre
/// (implicit `+WIENER_FILT_STEP`), slot 7 always 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WienerInfoLr {
    pub vfilter: [i16; 8],
    pub hfilter: [i16; 8],
}

impl Default for WienerInfoLr {
    /// `set_default_wiener` (restoration.h): the per-tile reference reset.
    fn default() -> Self {
        let taps = [3i16, -7, 15, -2 * (3 - 7 + 15), 15, -7, 3, 0];
        WienerInfoLr {
            vfilter: taps,
            hfilter: taps,
        }
    }
}

/// `SgrprojInfo` (blockd.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SgrprojInfoLr {
    pub ep: i32,
    pub xqd: [i32; 2],
}

impl Default for SgrprojInfoLr {
    /// `set_default_sgrproj` (restoration.h): the per-tile reference reset.
    fn default() -> Self {
        SgrprojInfoLr {
            ep: 0,
            xqd: [
                (SGRPROJ_PRJ_MIN0 + SGRPROJ_PRJ_MAX0) / 2,
                (SGRPROJ_PRJ_MIN1 + SGRPROJ_PRJ_MAX1) / 2,
            ],
        }
    }
}

/// One restoration unit's decoded parameters (`RestorationUnitInfo`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LrUnitInfo {
    /// `RESTORE_NONE` / `RESTORE_WIENER` / `RESTORE_SGRPROJ` (never
    /// SWITCHABLE at unit level).
    pub restoration_type: u8,
    pub wiener: WienerInfoLr,
    pub sgrproj: SgrprojInfoLr,
}

/// `inv_recenter_nonneg` (aom_dsp/recenter.h).
fn inv_recenter_nonneg(r: u16, v: u16) -> u16 {
    if v > (r << 1) {
        v
    } else if (v & 1) == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}

/// `inv_recenter_finite_nonneg` (aom_dsp/recenter.h): inverse-recenter a
/// value `v` in `[0, n-1]` around a reference `r` also in `[0, n-1]`.
pub fn inv_recenter_finite_nonneg(n: u16, r: u16, v: u16) -> u16 {
    if (r << 1) <= n {
        inv_recenter_nonneg(r, v)
    } else {
        n - 1 - inv_recenter_nonneg(n - 1 - r, v)
    }
}

/// `read_primitive_quniform` (aom_dsp/binary_codes_reader.c): quasi-uniform
/// value in `[0, n-1]` on the arithmetic coder.
pub fn read_primitive_quniform(dec: &mut OdEcDec, n: u16) -> u16 {
    if n <= 1 {
        return 0;
    }
    let l = (15 - n.leading_zeros() as i32) + 1; // get_msb(n) + 1
    let m = (1i32 << l) - n as i32;
    let v = read_literal(dec, (l - 1) as u32);
    if v < m {
        v as u16
    } else {
        ((v << 1) - m + read_bit(dec)) as u16
    }
}

/// `read_primitive_subexpfin` (aom_dsp/binary_codes_reader.c): finite
/// subexponential code for a symbol in `[0, n-1]` with parameter `k`.
pub fn read_primitive_subexpfin(dec: &mut OdEcDec, n: u16, k: u16) -> u16 {
    let mut i: i32 = 0;
    let mut mk: i32 = 0;
    loop {
        let b = if i != 0 { k as i32 + i - 1 } else { k as i32 };
        let a = 1i32 << b;
        if (n as i32) <= mk + 3 * a {
            return read_primitive_quniform(dec, (n as i32 - mk) as u16) + mk as u16;
        }
        if read_bit(dec) == 0 {
            return (read_literal(dec, b as u32) + mk) as u16;
        }
        i += 1;
        mk += a;
    }
}

/// `aom_read_primitive_refsubexpfin` (aom_dsp/binary_codes_reader.c).
pub fn read_primitive_refsubexpfin(dec: &mut OdEcDec, n: u16, k: u16, r: u16) -> u16 {
    inv_recenter_finite_nonneg(n, r, read_primitive_subexpfin(dec, n, k))
}

/// `read_wiener_filter` (decodeframe.c): the three coded taps per direction
/// (tap 0 skipped/zeroed for the 5-tap chroma window), centre = `-2 * sum`,
/// symmetric mirror in slots 4..6, slot 7 zero. Updates `ref` in place.
pub fn read_wiener_filter(
    dec: &mut OdEcDec,
    wiener_win: usize,
    r: &mut WienerInfoLr,
) -> WienerInfoLr {
    let mut w = WienerInfoLr {
        vfilter: [0; 8],
        hfilter: [0; 8],
    };
    for dir in 0..2 {
        let (out, rf): (&mut [i16; 8], &[i16; 8]) = if dir == 0 {
            (&mut w.vfilter, &r.vfilter)
        } else {
            (&mut w.hfilter, &r.hfilter)
        };
        if wiener_win == WIENER_WIN {
            let v = read_primitive_refsubexpfin(
                dec,
                (WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1) as u16,
                WIENER_FILT_TAP0_SUBEXP_K,
                (rf[0] as i32 - WIENER_FILT_TAP0_MINV) as u16,
            ) as i32
                + WIENER_FILT_TAP0_MINV;
            out[0] = v as i16;
            out[WIENER_WIN - 1] = v as i16;
        } else {
            out[0] = 0;
            out[WIENER_WIN - 1] = 0;
        }
        let v1 = read_primitive_refsubexpfin(
            dec,
            (WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1) as u16,
            WIENER_FILT_TAP1_SUBEXP_K,
            (rf[1] as i32 - WIENER_FILT_TAP1_MINV) as u16,
        ) as i32
            + WIENER_FILT_TAP1_MINV;
        out[1] = v1 as i16;
        out[WIENER_WIN - 2] = v1 as i16;
        let v2 = read_primitive_refsubexpfin(
            dec,
            (WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1) as u16,
            WIENER_FILT_TAP2_SUBEXP_K,
            (rf[2] as i32 - WIENER_FILT_TAP2_MINV) as u16,
        ) as i32
            + WIENER_FILT_TAP2_MINV;
        out[2] = v2 as i16;
        out[WIENER_WIN - 3] = v2 as i16;
        // The central element has an implicit +WIENER_FILT_STEP.
        out[WIENER_HALFWIN] = -2 * (out[0] + out[1] + out[2]);
    }
    *r = w;
    w
}

/// `read_sgrproj_filter` (decodeframe.c): the 4-bit `ep` then the projection
/// weights, coded per the parameter set's radii. Updates `ref` in place.
pub fn read_sgrproj_filter(dec: &mut OdEcDec, r: &mut SgrprojInfoLr) -> SgrprojInfoLr {
    let ep = read_literal(dec, SGRPROJ_PARAMS_BITS);
    let rad = SGR_PARAMS_R[ep as usize];
    let mut s = SgrprojInfoLr { ep, xqd: [0; 2] };
    let n0 = (SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1) as u16;
    let n1 = (SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1) as u16;
    if rad[0] == 0 {
        s.xqd[0] = 0;
        s.xqd[1] = read_primitive_refsubexpfin(
            dec,
            n1,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
        ) as i32
            + SGRPROJ_PRJ_MIN1;
    } else if rad[1] == 0 {
        s.xqd[0] = read_primitive_refsubexpfin(
            dec,
            n0,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
        ) as i32
            + SGRPROJ_PRJ_MIN0;
        s.xqd[1] = ((1 << SGRPROJ_PRJ_BITS) - s.xqd[0]).clamp(SGRPROJ_PRJ_MIN1, SGRPROJ_PRJ_MAX1);
    } else {
        s.xqd[0] = read_primitive_refsubexpfin(
            dec,
            n0,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
        ) as i32
            + SGRPROJ_PRJ_MIN0;
        s.xqd[1] = read_primitive_refsubexpfin(
            dec,
            n1,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
        ) as i32
            + SGRPROJ_PRJ_MIN1;
    }
    *r = s;
    s
}

/// Per-tile loop-restoration reference state (`xd->wiener_info` /
/// `xd->sgrproj_info`), reset by `av1_reset_loop_restoration` at tile start.
#[derive(Clone, Debug, Default)]
pub struct LrRefState {
    pub wiener: [WienerInfoLr; 3],
    pub sgrproj: [SgrprojInfoLr; 3],
}

/// `loop_restoration_read_sb_coeffs` (decodeframe.c): one restoration unit's
/// parameters, dispatched on the plane's `frame_restoration_type`. `cdf` is
/// the matching per-tile CDF instance (switchable: 3-symbol; wiener/sgrproj
/// gates: 2-symbol), adapted in place like `aom_read_symbol`.
pub fn read_lr_unit(
    dec: &mut OdEcDec,
    frame_restoration_type: u8,
    plane: usize,
    refs: &mut LrRefState,
    switchable_cdf: &mut [u16],
    wiener_cdf: &mut [u16],
    sgrproj_cdf: &mut [u16],
) -> LrUnitInfo {
    debug_assert_ne!(frame_restoration_type, RESTORE_NONE);
    let wiener_win = if plane > 0 {
        WIENER_WIN_CHROMA
    } else {
        WIENER_WIN
    };
    let mut u = LrUnitInfo::default();
    match frame_restoration_type {
        RESTORE_SWITCHABLE => {
            u.restoration_type = read_symbol(dec, switchable_cdf, RESTORE_SWITCHABLE_TYPES) as u8;
            match u.restoration_type {
                RESTORE_WIENER => {
                    u.wiener = read_wiener_filter(dec, wiener_win, &mut refs.wiener[plane]);
                }
                RESTORE_SGRPROJ => {
                    u.sgrproj = read_sgrproj_filter(dec, &mut refs.sgrproj[plane]);
                }
                _ => debug_assert_eq!(u.restoration_type, RESTORE_NONE),
            }
        }
        RESTORE_WIENER => {
            if read_symbol(dec, wiener_cdf, 2) != 0 {
                u.restoration_type = RESTORE_WIENER;
                u.wiener = read_wiener_filter(dec, wiener_win, &mut refs.wiener[plane]);
            } else {
                u.restoration_type = RESTORE_NONE;
            }
        }
        _ => {
            debug_assert_eq!(frame_restoration_type, RESTORE_SGRPROJ);
            if read_symbol(dec, sgrproj_cdf, 2) != 0 {
                u.restoration_type = RESTORE_SGRPROJ;
                u.sgrproj = read_sgrproj_filter(dec, &mut refs.sgrproj[plane]);
            } else {
                u.restoration_type = RESTORE_NONE;
            }
        }
    }
    u
}

/// `RESTORATION_UNITSIZE_MAX` / `RESTORATION_PROC_UNIT_SIZE` /
/// `RESTORATION_UNIT_OFFSET` (restoration.h).
pub const RESTORATION_UNITSIZE_MAX: i32 = 256;
pub const RESTORATION_PROC_UNIT_SIZE: i32 = 64;
pub const RESTORATION_UNIT_OFFSET: i32 = 8;

/// `av1_lr_count_units` (restoration.c): units along one axis — round the
/// plane size to nearest (a right/bottom unit may extend to 150%), min 1.
pub fn lr_count_units(unit_size: i32, plane_size: i32) -> i32 {
    ((plane_size + (unit_size >> 1)) / unit_size).max(1)
}

/// The loop-restoration frame geometry the tile parse and the frame walk
/// share: per-plane `frame_restoration_type` + unit size (from
/// `read_restoration_mode`) and the frame crop dims that size the unit grid.
/// `Default` (all `RESTORE_NONE`) codes and applies nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct LrFrameConfig {
    /// Per-plane `RESTORE_*` frame restoration type.
    pub frame_restoration_type: [u8; 3],
    /// Per-plane restoration unit size (64/128/256; luma-scale values —
    /// chroma planes use the coded chroma size directly).
    pub unit_size: [i32; 3],
    /// Frame crop dims — the superres-UPSCALED (display) width/height. The RU
    /// grid is always upscaled-domain (`av1_get_upsampled_plane_size`).
    pub crop_width: i32,
    pub crop_height: i32,
    /// Superres `SuperresDenom` (`cm->superres_scale_denominator`): `SCALE_NUMERATOR`
    /// (8) or `0` means unscaled; `[9,16]` means the frame was coded downscaled,
    /// so `lr_corners_in_sb` scales the downscaled mi position into the upscaled
    /// RU grid. `Default` (0) is unscaled.
    pub superres_denom: i32,
}

impl LrFrameConfig {
    /// Any plane restored (`do_loop_restoration`, decodeframe.c:5424).
    pub fn any_enabled(&self) -> bool {
        self.frame_restoration_type
            .iter()
            .any(|&t| t != RESTORE_NONE)
    }

    /// This plane's subsampled dims (`av1_get_upsampled_plane_size`).
    pub fn plane_size(&self, plane: usize, ss_x: usize, ss_y: usize) -> (i32, i32) {
        let (sx, sy) = if plane > 0 { (ss_x, ss_y) } else { (0, 0) };
        (
            (self.crop_width + (1 << sx) - 1) >> sx,
            (self.crop_height + (1 << sy) - 1) >> sy,
        )
    }

    /// This plane's `(horz_units, vert_units)` (`av1_alloc_restoration_struct`).
    pub fn plane_units(&self, plane: usize, ss_x: usize, ss_y: usize) -> (i32, i32) {
        let (pw, ph) = self.plane_size(plane, ss_x, ss_y);
        (
            lr_count_units(self.unit_size[plane], pw),
            lr_count_units(self.unit_size[plane], ph),
        )
    }
}

/// `av1_loop_restoration_corners_in_sb` (restoration.c) for the unscaled-
/// superres envelope: the half-open RU rectangle `[rcol0, rcol1) x
/// [rrow0, rrow1)` whose top-left corners fall inside the superblock at
/// `(mi_row, mi_col)` of `mi_size` mi units — the units whose parameters this
/// superblock codes. `None` when empty. The caller applies the
/// `bsize == sb_size` gate (the C returns 0 for any smaller bsize).
#[allow(clippy::too_many_arguments)]
pub fn lr_corners_in_sb(
    lr: &LrFrameConfig,
    plane: usize,
    ss_x: usize,
    ss_y: usize,
    mi_row: i32,
    mi_col: i32,
    mi_size_high: i32,
    mi_size_wide: i32,
) -> Option<(i32, i32, i32, i32)> {
    let (sx, sy) = if plane > 0 {
        (ss_x as i32, ss_y as i32)
    } else {
        (0, 0)
    };
    let size = lr.unit_size[plane];
    let (horz_units, vert_units) = lr.plane_units(plane, ss_x, ss_y);

    // MI_SIZE = 4. `av1_loop_restoration_corners_in_sb`: the mi position is in
    // the DOWNSCALED grid but the RU grid is upscaled, so for scaled superres
    // the X mapping is `u = D * MI_SIZE * m / N` — mi_to_num_x = mi_size_x * D,
    // denom_x = size * SCALE_NUMERATOR(N). Y is never scaled (superres is
    // horizontal only).
    let mi_size_x = 4 >> sx;
    let mi_size_y = 4 >> sy;
    const SCALE_NUMERATOR: i32 = 8;
    let superres_scaled = lr.superres_denom > SCALE_NUMERATOR;
    let (mi_to_num_x, denom_x) = if superres_scaled {
        (mi_size_x * lr.superres_denom, size * SCALE_NUMERATOR)
    } else {
        (mi_size_x, size)
    };
    let (mi_to_num_y, denom_y) = (mi_size_y, size);
    let (rnd_x, rnd_y) = (denom_x - 1, denom_y - 1);

    let rcol0 = (mi_col * mi_to_num_x + rnd_x) / denom_x;
    let rrow0 = (mi_row * mi_to_num_y + rnd_y) / denom_y;
    let rcol1 = (((mi_col + mi_size_wide) * mi_to_num_x + rnd_x) / denom_x).min(horz_units);
    let rrow1 = (((mi_row + mi_size_high) * mi_to_num_y + rnd_y) / denom_y).min(vert_units);
    (rcol0 < rcol1 && rrow0 < rrow1).then_some((rcol0, rcol1, rrow0, rrow1))
}

// ---------------------------------------------------------------------------
// Write side (encoder): `aom_dsp/binary_codes_writer.c` primitives, the
// per-unit writers of `av1/encoder/bitstream.c` (`write_wiener_filter` /
// `write_sgrproj_filter` / `loop_restoration_write_sb_coeffs`), and the bit
// counters the encoder's restoration search costs parameters with
// (`aom_count_primitive_refsubexpfin`, used by pickrst.c's
// `count_wiener_bits` / `count_sgrproj_bits`).
// ---------------------------------------------------------------------------

/// `recenter_nonneg` (aom_dsp/recenter.h): forward recentering of `v` around
/// reference `r` (both nonnegative).
fn recenter_nonneg(r: u16, v: u16) -> u16 {
    if v > (r << 1) {
        v
    } else if v >= r {
        (v - r) << 1
    } else {
        ((r - v) << 1) - 1
    }
}

/// `recenter_finite_nonneg` (aom_dsp/recenter.h): forward recentering of a
/// value `v` in `[0, n-1]` around a reference `r` also in `[0, n-1]` —
/// inverse of [`inv_recenter_finite_nonneg`].
pub fn recenter_finite_nonneg(n: u16, r: u16, v: u16) -> u16 {
    if (r << 1) <= n {
        recenter_nonneg(r, v)
    } else {
        recenter_nonneg(n - 1 - r, n - 1 - v)
    }
}

/// `write_primitive_quniform` (aom_dsp/binary_codes_writer.c): quasi-uniform
/// code for `v` in `[0, n-1]`.
pub fn write_primitive_quniform(enc: &mut OdEcEnc, n: u16, v: u16) {
    if n <= 1 {
        return;
    }
    let l = (15 - n.leading_zeros() as i32) + 1; // get_msb(n) + 1
    let m = (1i32 << l) - n as i32;
    if (v as i32) < m {
        write_literal(enc, v as i32, (l - 1) as u32);
    } else {
        write_literal(enc, m + ((v as i32 - m) >> 1), (l - 1) as u32);
        write_literal(enc, (v as i32 - m) & 1, 1);
    }
}

/// `count_primitive_quniform` (aom_dsp/binary_codes_writer.c).
pub fn count_primitive_quniform(n: u16, v: u16) -> i32 {
    if n <= 1 {
        return 0;
    }
    let l = (15 - n.leading_zeros() as i32) + 1;
    let m = (1i32 << l) - n as i32;
    if (v as i32) < m { l - 1 } else { l }
}

/// `write_primitive_subexpfin` (aom_dsp/binary_codes_writer.c): finite
/// subexponential code for `v` in `[0, n-1]` with parameter `k`.
pub fn write_primitive_subexpfin(enc: &mut OdEcEnc, n: u16, k: u16, v: u16) {
    let mut i: i32 = 0;
    let mut mk: i32 = 0;
    loop {
        let b = if i != 0 { k as i32 + i - 1 } else { k as i32 };
        let a = 1i32 << b;
        if (n as i32) <= mk + 3 * a {
            write_primitive_quniform(enc, (n as i32 - mk) as u16, (v as i32 - mk) as u16);
            return;
        }
        let t = (v as i32) >= mk + a;
        write_literal(enc, t as i32, 1);
        if t {
            i += 1;
            mk += a;
        } else {
            write_literal(enc, v as i32 - mk, b as u32);
            return;
        }
    }
}

/// `count_primitive_subexpfin` (aom_dsp/binary_codes_writer.c).
pub fn count_primitive_subexpfin(n: u16, k: u16, v: u16) -> i32 {
    let mut count: i32 = 0;
    let mut i: i32 = 0;
    let mut mk: i32 = 0;
    loop {
        let b = if i != 0 { k as i32 + i - 1 } else { k as i32 };
        let a = 1i32 << b;
        if (n as i32) <= mk + 3 * a {
            count += count_primitive_quniform((n as i32 - mk) as u16, (v as i32 - mk) as u16);
            return count;
        }
        count += 1;
        if (v as i32) >= mk + a {
            i += 1;
            mk += a;
        } else {
            count += b;
            return count;
        }
    }
}

/// `aom_write_primitive_refsubexpfin` (aom_dsp/binary_codes_writer.c).
pub fn write_primitive_refsubexpfin(enc: &mut OdEcEnc, n: u16, k: u16, r: u16, v: u16) {
    write_primitive_subexpfin(enc, n, k, recenter_finite_nonneg(n, r, v));
}

/// `aom_count_primitive_refsubexpfin` (aom_dsp/binary_codes_writer.c): the
/// coded bit count of [`write_primitive_refsubexpfin`].
pub fn count_primitive_refsubexpfin(n: u16, k: u16, r: u16, v: u16) -> i32 {
    count_primitive_subexpfin(n, k, recenter_finite_nonneg(n, r, v))
}

/// `write_wiener_filter` (av1/encoder/bitstream.c): the three coded taps per
/// direction (tap 0 skipped for the 5-tap chroma window), delta-coded against
/// `ref`, which is updated in place.
pub fn write_wiener_filter(
    enc: &mut OdEcEnc,
    wiener_win: usize,
    w: &WienerInfoLr,
    r: &mut WienerInfoLr,
) {
    for dir in 0..2 {
        let (f, rf): (&[i16; 8], &[i16; 8]) = if dir == 0 {
            (&w.vfilter, &r.vfilter)
        } else {
            (&w.hfilter, &r.hfilter)
        };
        if wiener_win == WIENER_WIN {
            write_primitive_refsubexpfin(
                enc,
                (WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1) as u16,
                WIENER_FILT_TAP0_SUBEXP_K,
                (rf[0] as i32 - WIENER_FILT_TAP0_MINV) as u16,
                (f[0] as i32 - WIENER_FILT_TAP0_MINV) as u16,
            );
        } else {
            debug_assert!(f[0] == 0 && f[WIENER_WIN - 1] == 0);
        }
        write_primitive_refsubexpfin(
            enc,
            (WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1) as u16,
            WIENER_FILT_TAP1_SUBEXP_K,
            (rf[1] as i32 - WIENER_FILT_TAP1_MINV) as u16,
            (f[1] as i32 - WIENER_FILT_TAP1_MINV) as u16,
        );
        write_primitive_refsubexpfin(
            enc,
            (WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1) as u16,
            WIENER_FILT_TAP2_SUBEXP_K,
            (rf[2] as i32 - WIENER_FILT_TAP2_MINV) as u16,
            (f[2] as i32 - WIENER_FILT_TAP2_MINV) as u16,
        );
    }
    *r = *w;
}

/// `count_wiener_bits` (av1/encoder/pickrst.c): the syntax bit count
/// [`write_wiener_filter`] would code for `w` against reference `r`
/// (reference NOT updated).
pub fn count_wiener_bits(wiener_win: usize, w: &WienerInfoLr, r: &WienerInfoLr) -> i32 {
    let mut bits = 0;
    for dir in 0..2 {
        let (f, rf): (&[i16; 8], &[i16; 8]) = if dir == 0 {
            (&w.vfilter, &r.vfilter)
        } else {
            (&w.hfilter, &r.hfilter)
        };
        if wiener_win == WIENER_WIN {
            bits += count_primitive_refsubexpfin(
                (WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1) as u16,
                WIENER_FILT_TAP0_SUBEXP_K,
                (rf[0] as i32 - WIENER_FILT_TAP0_MINV) as u16,
                (f[0] as i32 - WIENER_FILT_TAP0_MINV) as u16,
            );
        }
        bits += count_primitive_refsubexpfin(
            (WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1) as u16,
            WIENER_FILT_TAP1_SUBEXP_K,
            (rf[1] as i32 - WIENER_FILT_TAP1_MINV) as u16,
            (f[1] as i32 - WIENER_FILT_TAP1_MINV) as u16,
        );
        bits += count_primitive_refsubexpfin(
            (WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1) as u16,
            WIENER_FILT_TAP2_SUBEXP_K,
            (rf[2] as i32 - WIENER_FILT_TAP2_MINV) as u16,
            (f[2] as i32 - WIENER_FILT_TAP2_MINV) as u16,
        );
    }
    bits
}

/// `write_sgrproj_filter` (av1/encoder/bitstream.c): the 4-bit `ep` then the
/// projection weights coded per the parameter set's radii, delta-coded
/// against `ref`, which is updated in place.
pub fn write_sgrproj_filter(enc: &mut OdEcEnc, s: &SgrprojInfoLr, r: &mut SgrprojInfoLr) {
    write_literal(enc, s.ep, SGRPROJ_PARAMS_BITS);
    let rad = SGR_PARAMS_R[s.ep as usize];
    let n0 = (SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1) as u16;
    let n1 = (SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1) as u16;
    if rad[0] == 0 {
        debug_assert_eq!(s.xqd[0], 0);
        write_primitive_refsubexpfin(
            enc,
            n1,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
            (s.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
        );
    } else if rad[1] == 0 {
        write_primitive_refsubexpfin(
            enc,
            n0,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
            (s.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
        );
    } else {
        write_primitive_refsubexpfin(
            enc,
            n0,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
            (s.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
        );
        write_primitive_refsubexpfin(
            enc,
            n1,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
            (s.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
        );
    }
    *r = *s;
}

/// `count_sgrproj_bits` (av1/encoder/pickrst.c): the syntax bit count
/// [`write_sgrproj_filter`] would code for `s` against reference `r`
/// (reference NOT updated). Includes the `SGRPROJ_PARAMS_BITS` literal.
pub fn count_sgrproj_bits(s: &SgrprojInfoLr, r: &SgrprojInfoLr) -> i32 {
    let mut bits = SGRPROJ_PARAMS_BITS as i32;
    let rad = SGR_PARAMS_R[s.ep as usize];
    if rad[0] > 0 {
        bits += count_primitive_refsubexpfin(
            (SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1) as u16,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
            (s.xqd[0] - SGRPROJ_PRJ_MIN0) as u16,
        );
    }
    if rad[1] > 0 {
        bits += count_primitive_refsubexpfin(
            (SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1) as u16,
            SGRPROJ_PRJ_SUBEXP_K,
            (r.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
            (s.xqd[1] - SGRPROJ_PRJ_MIN1) as u16,
        );
    }
    bits
}

/// `loop_restoration_write_sb_coeffs` (av1/encoder/bitstream.c): one
/// restoration unit's parameters, dispatched on the plane's
/// `frame_restoration_type`. The mirror of [`read_lr_unit`]; `cdf` is the
/// matching per-tile CDF instance, adapted in place like `aom_write_symbol`
/// iff `allow_update_cdf` (the writer's `w->allow_update_cdf` gate).
#[allow(clippy::too_many_arguments)]
pub fn write_lr_unit(
    enc: &mut OdEcEnc,
    u: &LrUnitInfo,
    frame_restoration_type: u8,
    plane: usize,
    refs: &mut LrRefState,
    switchable_cdf: &mut [u16],
    wiener_cdf: &mut [u16],
    sgrproj_cdf: &mut [u16],
    allow_update_cdf: bool,
) {
    debug_assert_ne!(frame_restoration_type, RESTORE_NONE);
    let wiener_win = if plane > 0 {
        WIENER_WIN_CHROMA
    } else {
        WIENER_WIN
    };
    let mut sym = |enc: &mut OdEcEnc, symb: i32, cdf: &mut [u16], nsymbs: usize| {
        enc.encode_cdf_q15(symb, &cdf[..nsymbs], nsymbs as i32);
        if allow_update_cdf {
            update_cdf(cdf, symb, nsymbs);
        }
    };
    match frame_restoration_type {
        RESTORE_SWITCHABLE => {
            sym(
                enc,
                u.restoration_type as i32,
                switchable_cdf,
                RESTORE_SWITCHABLE_TYPES,
            );
            match u.restoration_type {
                RESTORE_WIENER => {
                    write_wiener_filter(enc, wiener_win, &u.wiener, &mut refs.wiener[plane]);
                }
                RESTORE_SGRPROJ => {
                    write_sgrproj_filter(enc, &u.sgrproj, &mut refs.sgrproj[plane]);
                }
                _ => debug_assert_eq!(u.restoration_type, RESTORE_NONE),
            }
        }
        RESTORE_WIENER => {
            sym(
                enc,
                (u.restoration_type == RESTORE_WIENER) as i32,
                wiener_cdf,
                2,
            );
            if u.restoration_type == RESTORE_WIENER {
                write_wiener_filter(enc, wiener_win, &u.wiener, &mut refs.wiener[plane]);
            }
        }
        _ => {
            debug_assert_eq!(frame_restoration_type, RESTORE_SGRPROJ);
            sym(
                enc,
                (u.restoration_type == RESTORE_SGRPROJ) as i32,
                sgrproj_cdf,
                2,
            );
            if u.restoration_type == RESTORE_SGRPROJ {
                write_sgrproj_filter(enc, &u.sgrproj, &mut refs.sgrproj[plane]);
            }
        }
    }
}
