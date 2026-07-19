//! aom-inter — bit-exact AV1 **interintra** prediction blend for the decoder
//! (lowbd, bd = 8). Port of libaom v3.14.1's `av1_combine_interintra`
//! (`av1/common/reconinter.c:1138`) and its two mask families:
//!
//! - **smooth** interintra masks (`build_smooth_interintra_mask`,
//!   reconinter.c:540) from the 1-D weight table `ii_weights1d` (:524), used for
//!   `II_DC/V/H/SMOOTH` when `use_wedge_interintra == 0`;
//! - **wedge** interintra masks (the compound wedge codebook, built once at init
//!   by `init_wedge_master_masks` / `init_wedge_masks`, reconinter.c:449/494),
//!   used when `use_wedge_interintra == 1` (fixed `wedge_sign = 0`).
//!
//! The blend itself is `aom_blend_a64_mask_c` (`aom_dsp/blend_a64_mask.c:229`):
//! `dst = round(mask*intra + (64-mask)*inter, 6)` — note the mask weights the
//! **intra** predictor (`src0`), `(64-mask)` the **inter** predictor (`src1`),
//! with 2×2 / 2×1 / 1×2 mask subsampling for the chroma plane on the wedge path.
//!
//! The inter predictor is produced by the crate's existing translational /
//! warp / OBMC path; this module only builds the intra-vs-inter mask blend that
//! overlays it. The intra predictor comes from the caller (aom-intra). Both the
//! blend arithmetic and the wedge codebook are differentially locked vs the real
//! exported C in `tests/interintra_diff.rs`.

// --- BLOCK_SIZES_ALL tables (common_data.h) ---
const BLOCK_SIZE_WIDE: [usize; 22] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BLOCK_SIZE_HIGH: [usize; 22] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];
const MI_SIZE_WIDE: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_SIZE_HIGH: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

/// `interintra_to_intra_mode[INTERINTRA_MODES]` (reconintra.h): II_DC→DC_PRED(0),
/// II_V→V_PRED(1), II_H→H_PRED(2), II_SMOOTH→SMOOTH_PRED(9).
pub const INTERINTRA_TO_INTRA_MODE: [usize; 4] = [0, 1, 2, 9];

const AOM_BLEND_A64_ROUND_BITS: i32 = 6;
const AOM_BLEND_A64_MAX_ALPHA: i32 = 1 << AOM_BLEND_A64_ROUND_BITS; // 64

#[inline]
fn round_pow2(v: i32, n: i32) -> i32 {
    (v + ((1 << n) >> 1)) >> n
}

/// `AOM_BLEND_A64(m, v0, v1)` = `round(m*v0 + (64-m)*v1, 6)` (aom_dsp/blend.h).
#[inline]
fn blend_a64(m: i32, v0: u16, v1: u16) -> u16 {
    round_pow2(
        m * v0 as i32 + (AOM_BLEND_A64_MAX_ALPHA - m) * v1 as i32,
        AOM_BLEND_A64_ROUND_BITS,
    ) as u16
}

/// `aom_blend_a64_mask_c` (aom_dsp/blend_a64_mask.c:229). `src0` is weighted by
/// `mask`, `src1` by `64-mask`. `subw`/`subh` average a luma-resolution mask down
/// to a subsampled (chroma) block: 2×2, 1×2, or 2×1 box-average of the mask.
#[allow(clippy::too_many_arguments)]
pub fn blend_a64_mask(
    dst: &mut [u16],
    dst_stride: usize,
    src0: &[u16],
    src0_stride: usize,
    src1: &[u16],
    src1_stride: usize,
    mask: &[u8],
    mask_stride: usize,
    w: usize,
    h: usize,
    subw: bool,
    subh: bool,
) {
    let m_at = |r: usize, c: usize| mask[r * mask_stride + c] as i32;
    for i in 0..h {
        for j in 0..w {
            let m = match (subw, subh) {
                (false, false) => m_at(i, j),
                (true, true) => round_pow2(
                    m_at(2 * i, 2 * j)
                        + m_at(2 * i + 1, 2 * j)
                        + m_at(2 * i, 2 * j + 1)
                        + m_at(2 * i + 1, 2 * j + 1),
                    2,
                ),
                (true, false) => round_pow2(m_at(i, 2 * j) + m_at(i, 2 * j + 1), 1),
                (false, true) => round_pow2(m_at(2 * i, j) + m_at(2 * i + 1, j), 1),
            };
            dst[i * dst_stride + j] =
                blend_a64(m, src0[i * src0_stride + j], src1[i * src1_stride + j]);
        }
    }
}

// ===================================================================
// Smooth interintra masks (build_smooth_interintra_mask, reconinter.c:540).
// ===================================================================

/// `ii_weights1d[MAX_SB_SIZE = 128]` (reconinter.c:524) — the raised falloff
/// weight table (0..60) for the smooth interintra mask.
#[rustfmt::skip]
const II_WEIGHTS_1D: [u8; 128] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32,
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16,
    16, 15, 15, 14, 14, 13, 13, 12, 12, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8,
    8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4,
    4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// `ii_size_scales[BLOCK_SIZES_ALL]` (reconinter.c:533): the `ii_weights1d`
/// index stride per (plane) block size.
const II_SIZE_SCALES: [usize; 22] = [
    32, 16, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2, 1, 1, 1, 8, 8, 4, 4, 2, 2,
];

/// `build_smooth_interintra_mask` (reconinter.c:540): the 2-D smooth mask for
/// `mode` (II_DC/V/H/SMOOTH) at `plane_bsize`, contiguous with `stride = bw`.
/// II_V varies down rows, II_H across cols, II_SMOOTH by `min(i,j)`, II_DC flat 32.
pub fn build_smooth_interintra_mask(mode: usize, plane_bsize: usize) -> Vec<u8> {
    let bw = BLOCK_SIZE_WIDE[plane_bsize];
    let bh = BLOCK_SIZE_HIGH[plane_bsize];
    let scale = II_SIZE_SCALES[plane_bsize];
    let mut mask = vec![0u8; bw * bh];
    match mode {
        1 => {
            // II_V_PRED
            for i in 0..bh {
                let v = II_WEIGHTS_1D[i * scale];
                for j in 0..bw {
                    mask[i * bw + j] = v;
                }
            }
        }
        2 => {
            // II_H_PRED
            for i in 0..bh {
                for j in 0..bw {
                    mask[i * bw + j] = II_WEIGHTS_1D[j * scale];
                }
            }
        }
        3 => {
            // II_SMOOTH_PRED
            for i in 0..bh {
                for j in 0..bw {
                    mask[i * bw + j] = II_WEIGHTS_1D[i.min(j) * scale];
                }
            }
        }
        _ => {
            // II_DC_PRED (and default)
            mask.fill(32);
        }
    }
    mask
}

// ===================================================================
// Wedge interintra masks (compound wedge codebook; sign fixed to 0).
// ===================================================================

const MASK_MASTER_SIZE: usize = 64;
const MASK_MASTER_STRIDE: usize = 64;
const WEDGE_WEIGHT_BITS: i32 = 6;
const MAX_WEDGE_TYPES: usize = 16;

// Direction enum (reconinter.h:49-57).
const WEDGE_HORIZONTAL: usize = 0;
const WEDGE_VERTICAL: usize = 1;
const WEDGE_OBLIQUE27: usize = 2;
const WEDGE_OBLIQUE63: usize = 3;
const WEDGE_OBLIQUE117: usize = 4;
const WEDGE_OBLIQUE153: usize = 5;

#[rustfmt::skip]
const WEDGE_MASTER_OBLIQUE_ODD: [u8; 64] = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,1,2,6,18,
    37,53,60,63,64,64,64,64,64,64,64,64,64,64,64,64, 64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
];
#[rustfmt::skip]
const WEDGE_MASTER_OBLIQUE_EVEN: [u8; 64] = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,1,4,11,27,
    46,58,62,63,64,64,64,64,64,64,64,64,64,64,64,64, 64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
];
#[rustfmt::skip]
const WEDGE_MASTER_VERTICAL: [u8; 64] = [
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,2,7,21,
    43,57,62,64,64,64,64,64,64,64,64,64,64,64,64,64, 64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
];

// (direction, x_offset, y_offset) codebooks (reconinter.c:203-234).
type WedgeCode = (usize, i32, i32);
#[rustfmt::skip]
const WEDGE_CODEBOOK_16_HGTW: [WedgeCode; 16] = [
    (WEDGE_OBLIQUE27,4,4),(WEDGE_OBLIQUE63,4,4),(WEDGE_OBLIQUE117,4,4),(WEDGE_OBLIQUE153,4,4),
    (WEDGE_HORIZONTAL,4,2),(WEDGE_HORIZONTAL,4,4),(WEDGE_HORIZONTAL,4,6),(WEDGE_VERTICAL,4,4),
    (WEDGE_OBLIQUE27,4,2),(WEDGE_OBLIQUE27,4,6),(WEDGE_OBLIQUE153,4,2),(WEDGE_OBLIQUE153,4,6),
    (WEDGE_OBLIQUE63,2,4),(WEDGE_OBLIQUE63,6,4),(WEDGE_OBLIQUE117,2,4),(WEDGE_OBLIQUE117,6,4),
];
#[rustfmt::skip]
const WEDGE_CODEBOOK_16_HLTW: [WedgeCode; 16] = [
    (WEDGE_OBLIQUE27,4,4),(WEDGE_OBLIQUE63,4,4),(WEDGE_OBLIQUE117,4,4),(WEDGE_OBLIQUE153,4,4),
    (WEDGE_VERTICAL,2,4),(WEDGE_VERTICAL,4,4),(WEDGE_VERTICAL,6,4),(WEDGE_HORIZONTAL,4,4),
    (WEDGE_OBLIQUE27,4,2),(WEDGE_OBLIQUE27,4,6),(WEDGE_OBLIQUE153,4,2),(WEDGE_OBLIQUE153,4,6),
    (WEDGE_OBLIQUE63,2,4),(WEDGE_OBLIQUE63,6,4),(WEDGE_OBLIQUE117,2,4),(WEDGE_OBLIQUE117,6,4),
];
#[rustfmt::skip]
const WEDGE_CODEBOOK_16_HEQW: [WedgeCode; 16] = [
    (WEDGE_OBLIQUE27,4,4),(WEDGE_OBLIQUE63,4,4),(WEDGE_OBLIQUE117,4,4),(WEDGE_OBLIQUE153,4,4),
    (WEDGE_HORIZONTAL,4,2),(WEDGE_HORIZONTAL,4,6),(WEDGE_VERTICAL,2,4),(WEDGE_VERTICAL,6,4),
    (WEDGE_OBLIQUE27,4,2),(WEDGE_OBLIQUE27,4,6),(WEDGE_OBLIQUE153,4,2),(WEDGE_OBLIQUE153,4,6),
    (WEDGE_OBLIQUE63,2,4),(WEDGE_OBLIQUE63,6,4),(WEDGE_OBLIQUE117,2,4),(WEDGE_OBLIQUE117,6,4),
];

/// `wedge_signflip_lookup[BLOCK_SIZES_ALL][MAX_WEDGE_TYPES]` (reconinter.c:159).
#[rustfmt::skip]
const WEDGE_SIGNFLIP: [[u8; 16]; 22] = [
    [0;16],[0;16],[0;16],
    [1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1], // BLOCK_8X8
    [1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1], // BLOCK_8X16
    [1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1], // BLOCK_16X8
    [1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1], // BLOCK_16X16
    [1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1], // BLOCK_16X32
    [1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1], // BLOCK_32X16
    [1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1], // BLOCK_32X32
    [0;16],[0;16],[0;16],[0;16],[0;16],[0;16],[0;16],[0;16],
    [1,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1], // BLOCK_8X32
    [1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1], // BLOCK_32X8
    [0;16],[0;16],
];

/// `av1_wedge_params_lookup[bsize].codebook` — the codebook per bsize (or `None`
/// when the bsize has no wedge types).
fn wedge_codebook(bsize: usize) -> Option<&'static [WedgeCode; 16]> {
    match bsize {
        3 | 6 | 9 => Some(&WEDGE_CODEBOOK_16_HEQW),       // 8X8, 16X16, 32X32 (h==w)
        4 | 7 | 18 => Some(&WEDGE_CODEBOOK_16_HGTW),      // 8X16, 16X32, 8X32 (h>w)
        5 | 8 | 19 => Some(&WEDGE_CODEBOOK_16_HLTW),      // 16X8, 32X16, 32X8 (h<w)
        _ => None,
    }
}

/// `av1_is_wedge_used(bsize)` (reconinter.h:329): whether the block size has a
/// nonzero wedge type count.
pub fn is_wedge_used(bsize: usize) -> bool {
    wedge_codebook(bsize).is_some()
}

fn shift_copy(src: &[u8; 64], dst: &mut [u8], shift: i32) {
    let w = MASK_MASTER_SIZE;
    if shift >= 0 {
        let s = shift as usize;
        dst[s..w].copy_from_slice(&src[..w - s]);
        for d in dst.iter_mut().take(s) {
            *d = src[0];
        }
    } else {
        let s = (-shift) as usize;
        dst[..w - s].copy_from_slice(&src[s..w]);
        for d in dst.iter_mut().take(w).skip(w - s) {
            *d = src[w - 1];
        }
    }
}

/// `wedge_mask_obl[2][WEDGE_DIRECTIONS=6][64*64]`, built by
/// `init_wedge_master_masks` (reconinter.c:449).
fn build_wedge_mask_obl() -> Vec<Vec<Vec<u8>>> {
    let sz = MASK_MASTER_SIZE * MASK_MASTER_SIZE;
    let mut obl = vec![vec![vec![0u8; sz]; 6]; 2];
    let stride = MASK_MASTER_STRIDE;
    let h = MASK_MASTER_SIZE;
    let w = MASK_MASTER_SIZE;

    // Prototype rows: OBLIQUE63 from the two oblique masters (shifted), VERTICAL
    // from the vertical master.
    let mut shift = (h / 4) as i32;
    let mut i = 0;
    while i < h {
        shift_copy(
            &WEDGE_MASTER_OBLIQUE_EVEN,
            &mut obl[0][WEDGE_OBLIQUE63][i * stride..i * stride + w],
            shift,
        );
        shift -= 1;
        shift_copy(
            &WEDGE_MASTER_OBLIQUE_ODD,
            &mut obl[0][WEDGE_OBLIQUE63][(i + 1) * stride..(i + 1) * stride + w],
            shift,
        );
        obl[0][WEDGE_VERTICAL][i * stride..i * stride + w].copy_from_slice(&WEDGE_MASTER_VERTICAL);
        obl[0][WEDGE_VERTICAL][(i + 1) * stride..(i + 1) * stride + w]
            .copy_from_slice(&WEDGE_MASTER_VERTICAL);
        i += 2;
    }

    // Derive the remaining directions + the [1] (negated) set by transposition
    // and complement.
    let comp = (1i32 << WEDGE_WEIGHT_BITS) as u8; // 64
    for i in 0..h {
        for j in 0..w {
            let msk = obl[0][WEDGE_OBLIQUE63][i * stride + j];
            obl[0][WEDGE_OBLIQUE27][j * stride + i] = msk;
            let cmsk = comp - msk;
            obl[0][WEDGE_OBLIQUE117][i * stride + w - 1 - j] = cmsk;
            obl[0][WEDGE_OBLIQUE153][(w - 1 - j) * stride + i] = cmsk;
            obl[1][WEDGE_OBLIQUE63][i * stride + j] = cmsk;
            obl[1][WEDGE_OBLIQUE27][j * stride + i] = cmsk;
            obl[1][WEDGE_OBLIQUE117][i * stride + w - 1 - j] = msk;
            obl[1][WEDGE_OBLIQUE153][(w - 1 - j) * stride + i] = msk;
            let mskx = obl[0][WEDGE_VERTICAL][i * stride + j];
            obl[0][WEDGE_HORIZONTAL][j * stride + i] = mskx;
            let cmskx = comp - mskx;
            obl[1][WEDGE_VERTICAL][i * stride + j] = cmskx;
            obl[1][WEDGE_HORIZONTAL][j * stride + i] = cmskx;
        }
    }
    obl
}

fn wedge_obl() -> &'static Vec<Vec<Vec<u8>>> {
    use std::sync::OnceLock;
    static OBL: OnceLock<Vec<Vec<Vec<u8>>>> = OnceLock::new();
    OBL.get_or_init(build_wedge_mask_obl)
}

/// `av1_get_contiguous_soft_mask(index, sign=0, bsize)` = the baked wedge mask —
/// a contiguous `bw*bh` buffer (stride `bw`), values 0..64. Matches
/// `av1_wedge_params_lookup[bsize].masks[0][index]` (reconinter.c:494 init).
/// Returns `None` for a bsize with no wedge types.
pub fn wedge_mask(bsize: usize, index: usize) -> Option<Vec<u8>> {
    let cb = wedge_codebook(bsize)?;
    let (direction, x_off, y_off) = cb[index];
    let bw = BLOCK_SIZE_WIDE[bsize];
    let bh = BLOCK_SIZE_HIGH[bsize];
    let wsignflip = WEDGE_SIGNFLIP[bsize][index] as usize;
    let woff = ((x_off * bw as i32) >> 3) as usize;
    let hoff = ((y_off * bh as i32) >> 3) as usize;
    let obl = wedge_obl();
    let plane = wsignflip; // neg = 0 for interintra
    let base_row = MASK_MASTER_SIZE / 2 - hoff;
    let base_col = MASK_MASTER_SIZE / 2 - woff;
    let src = &obl[plane][direction];
    let mut out = vec![0u8; bw * bh];
    for r in 0..bh {
        for c in 0..bw {
            out[r * bw + c] = src[(base_row + r) * MASK_MASTER_STRIDE + base_col + c];
        }
    }
    Some(out)
}

// ===================================================================
// combine_interintra (reconinter.c:1059).
// ===================================================================

/// `combine_interintra` (reconinter.c:1059): blend the inter predictor
/// (`inter_pred`) with the intra predictor (`intra_pred`) into `comp` using the
/// smooth or wedge interintra mask. `mask` weights **intra**, `64-mask` weights
/// **inter**. `bsize` is the luma block size, `plane_bsize` the (possibly
/// subsampled) plane block size.
#[allow(clippy::too_many_arguments)]
pub fn combine_interintra(
    mode: usize,
    use_wedge_interintra: bool,
    wedge_index: usize,
    bsize: usize,
    plane_bsize: usize,
    comp: &mut [u16],
    comp_stride: usize,
    inter_pred: &[u16],
    inter_stride: usize,
    intra_pred: &[u16],
    intra_stride: usize,
) {
    let bw = BLOCK_SIZE_WIDE[plane_bsize];
    let bh = BLOCK_SIZE_HIGH[plane_bsize];

    if use_wedge_interintra {
        if let Some(mask) = wedge_mask(bsize, wedge_index) {
            // The wedge mask is at LUMA resolution (stride block_size_wide[bsize]);
            // for a subsampled plane it is box-averaged in the blend (subw/subh).
            let subw = 2 * MI_SIZE_WIDE[bsize] == bw;
            let subh = 2 * MI_SIZE_HIGH[bsize] == bh;
            blend_a64_mask(
                comp,
                comp_stride,
                intra_pred,
                intra_stride,
                inter_pred,
                inter_stride,
                &mask,
                BLOCK_SIZE_WIDE[bsize],
                bw,
                bh,
                subw,
                subh,
            );
        }
        return;
    }

    // Smooth path: the mask is prebuilt at PLANE resolution (stride = bw), blended
    // 1:1 (no subsampling).
    let mask = build_smooth_interintra_mask(mode, plane_bsize);
    blend_a64_mask(
        comp,
        comp_stride,
        intra_pred,
        intra_stride,
        inter_pred,
        inter_stride,
        &mask,
        bw,
        bw,
        bh,
        false,
        false,
    );
}
