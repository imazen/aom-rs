//! Partition-symbol CDF primitives (libaom `av1/common/av1_common_int.h`) — the
//! per-block-size partition CDF length and the edge-block CDF "gather" transforms
//! that reduce the full partition CDF to a 2-way split-vs-not distribution when a
//! superblock is clipped by the frame boundary. Byte-identical to C.

/// `CDF_PROB_TOP` (`aom_dsp/prob.h`): `1 << CDF_PROB_BITS`, `CDF_PROB_BITS = 15`.
const CDF_PROB_TOP: i32 = 1 << 15;

// PARTITION_TYPE indices (`av1/common/enums.h`).
const PARTITION_HORZ: usize = 1;
const PARTITION_VERT: usize = 2;
const PARTITION_SPLIT: usize = 3;
const PARTITION_HORZ_A: usize = 4;
const PARTITION_HORZ_B: usize = 5;
const PARTITION_VERT_A: usize = 6;
const PARTITION_VERT_B: usize = 7;
const PARTITION_HORZ_4: usize = 8;
const PARTITION_VERT_4: usize = 9;

// BLOCK_SIZE indices (`av1/common/enums.h`).
const BLOCK_8X8: usize = 3;
const BLOCK_128X128: usize = 15;

/// `mi_size_wide_log2[BLOCK_SIZES_ALL]` (`common_data.h`): log2 of a block's width in
/// mode-info (4x4) units.
const MI_SIZE_WIDE_LOG2: [u8; 22] =
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4];
/// `MAX_MIB_MASK` = `MAX_MIB_SIZE - 1` = 31 (128-wide superblock in mi units).
const MAX_MIB_MASK: usize = 31;
/// `PARTITION_PLOFFSET`: probability models per block size.
const PARTITION_PLOFFSET: i32 = 4;

/// `partition_plane_context` (`av1_common_int.h`): the partition CDF context for a
/// block, from the above/left partition-context bits at the block's size level
/// (`bsl`) — `(left*2 + above) + bsl * PARTITION_PLOFFSET`.
pub fn partition_plane_context(
    above_ctx: &[i8],
    left_ctx: &[i8],
    mi_row: usize,
    mi_col: usize,
    bsize: usize,
) -> i32 {
    let bsl = MI_SIZE_WIDE_LOG2[bsize] as i32 - MI_SIZE_WIDE_LOG2[BLOCK_8X8] as i32;
    let above = (above_ctx[mi_col] as i32 >> bsl) & 1;
    let left = (left_ctx[mi_row & MAX_MIB_MASK] as i32 >> bsl) & 1;
    (left * 2 + above) + bsl * PARTITION_PLOFFSET
}

/// `partition_cdf_length` (`av1_common_int.h`): the number of partition symbols a
/// block of `bsize` codes — `PARTITION_TYPES`(4) at 8x8, `EXT_PARTITION_TYPES`(10)
/// generally, and `EXT_PARTITION_TYPES - 2`(8) at 128x128 (no 4:1 splits).
pub fn partition_cdf_length(bsize: usize) -> usize {
    if bsize <= BLOCK_8X8 {
        4
    } else if bsize == BLOCK_128X128 {
        8
    } else {
        10
    }
}

/// `cdf_element_prob` (`aom_dsp/prob.h`): the probability mass of symbol `element`
/// in an inverse-cumulative CDF — `(element>0 ? cdf[element-1] : CDF_PROB_TOP) -
/// cdf[element]`.
fn cdf_element_prob(cdf: &[u16], element: usize) -> i32 {
    let hi = if element > 0 { cdf[element - 1] as i32 } else { CDF_PROB_TOP };
    hi - cdf[element] as i32
}

/// `partition_gather_vert_alike` (`av1_common_int.h`): reduce the full partition CDF
/// to a 2-way distribution of "codes a vertical-alike split" vs not, for a block
/// with columns but no rows at the frame edge. `out[0] = AOM_ICDF(TOP - Σ probs)`,
/// `out[1] = 0`.
pub fn partition_gather_vert_alike(cdf_in: &[u16], bsize: usize) -> [u16; 2] {
    let mut o = CDF_PROB_TOP;
    o -= cdf_element_prob(cdf_in, PARTITION_VERT);
    o -= cdf_element_prob(cdf_in, PARTITION_SPLIT);
    o -= cdf_element_prob(cdf_in, PARTITION_HORZ_A);
    o -= cdf_element_prob(cdf_in, PARTITION_VERT_A);
    o -= cdf_element_prob(cdf_in, PARTITION_VERT_B);
    if bsize != BLOCK_128X128 {
        o -= cdf_element_prob(cdf_in, PARTITION_VERT_4);
    }
    [(CDF_PROB_TOP - o) as u16, 0]
}

/// `partition_gather_horz_alike` (`av1_common_int.h`): the horizontal-edge companion
/// of [`partition_gather_vert_alike`].
pub fn partition_gather_horz_alike(cdf_in: &[u16], bsize: usize) -> [u16; 2] {
    let mut o = CDF_PROB_TOP;
    o -= cdf_element_prob(cdf_in, PARTITION_HORZ);
    o -= cdf_element_prob(cdf_in, PARTITION_SPLIT);
    o -= cdf_element_prob(cdf_in, PARTITION_HORZ_A);
    o -= cdf_element_prob(cdf_in, PARTITION_HORZ_B);
    o -= cdf_element_prob(cdf_in, PARTITION_VERT_A);
    if bsize != BLOCK_128X128 {
        o -= cdf_element_prob(cdf_in, PARTITION_HORZ_4);
    }
    [(CDF_PROB_TOP - o) as u16, 0]
}

use crate::cdf::write_symbol;
use crate::enc::OdEcEnc;

/// `write_partition` (`av1/encoder/bitstream.c`): code the partition symbol `p` for a
/// block. When the block has both rows and columns in-frame, the full partition CDF is
/// used (with adaptation, `aom_write_symbol`); at a frame edge the CDF is gathered to a
/// 2-way split-vs-not distribution and coded without adaptation (`aom_write_cdf`); when
/// neither rows nor columns remain the partition is forced `PARTITION_SPLIT` and nothing
/// is coded. `partition_cdf` is the (context-selected) CDF, adapted in place.
pub fn write_partition(
    enc: &mut OdEcEnc,
    partition_cdf: &mut [u16],
    cdf_len: usize,
    p: i32,
    has_rows: bool,
    has_cols: bool,
    bsize: usize,
) {
    if bsize < BLOCK_8X8 {
        return; // not a partition point
    }
    if has_rows && has_cols {
        write_symbol(enc, p, partition_cdf, cdf_len);
    } else if !has_rows && has_cols {
        let cdf = partition_gather_vert_alike(partition_cdf, bsize);
        enc.encode_cdf_q15((p == PARTITION_SPLIT as i32) as i32, &cdf, 2);
    } else if has_rows && !has_cols {
        let cdf = partition_gather_horz_alike(partition_cdf, bsize);
        enc.encode_cdf_q15((p == PARTITION_SPLIT as i32) as i32, &cdf, 2);
    }
    // !has_rows && !has_cols => PARTITION_SPLIT, nothing coded.
}

/// `av1_get_skip_txfm_context` (`av1/common/*.h`): the transform-skip CDF context —
/// the sum of the above and left neighbours' `skip_txfm` flags (each 0 when the
/// neighbour is off-frame), giving a context in `{0, 1, 2}`.
pub fn skip_txfm_context(above_skip_txfm: i32, left_skip_txfm: i32) -> i32 {
    above_skip_txfm + left_skip_txfm
}

/// `write_skip` (`av1/encoder/bitstream.c`): the per-block transform-skip flag. When
/// segment-level skip is active the flag is implied (returns 1, nothing coded);
/// otherwise the `skip_txfm` bit is coded on the (context-selected) 2-symbol skip CDF
/// with adaptation. Returns the coded skip value.
pub fn write_skip(enc: &mut OdEcEnc, skip_cdf: &mut [u16], seg_skip_active: bool, skip_txfm: i32) -> i32 {
    if seg_skip_active {
        return 1;
    }
    write_symbol(enc, skip_txfm, skip_cdf, 2);
    skip_txfm
}

use crate::cdf::{read_bit, read_literal, write_bit, write_literal};

const DELTA_Q_SMALL: i32 = 3;
const DELTA_Q_PROBS: usize = 3;

/// `get_msb`: index of the most-significant set bit (`floor(log2(n))`), `n > 0`.
fn get_msb(n: u32) -> u32 {
    31 - n.leading_zeros()
}

/// `write_delta_qindex` (`av1/encoder/bitstream.c`): the per-superblock delta-q — the
/// clamped magnitude symbol `min(|dq|, DELTA_Q_SMALL)` on the 4-symbol delta-q CDF
/// (adapted), then for large magnitudes the exp-Golomb remainder (`rem_bits-1` in 3
/// bits + `|dq|-thr` in `rem_bits`), and the sign bit when nonzero.
pub fn write_delta_qindex(enc: &mut OdEcEnc, delta_q_cdf: &mut [u16], delta_qindex: i32) {
    let sign = delta_qindex < 0;
    let abs = delta_qindex.abs();
    let smallval = abs < DELTA_Q_SMALL;
    write_symbol(enc, abs.min(DELTA_Q_SMALL), delta_q_cdf, DELTA_Q_PROBS + 1);
    if !smallval {
        let rem_bits = get_msb((abs - 1) as u32) as i32;
        let thr = (1 << rem_bits) + 1;
        write_literal(enc, rem_bits - 1, 3);
        write_literal(enc, abs - thr, rem_bits as u32);
    }
    if abs > 0 {
        write_bit(enc, sign as i32);
    }
}

const DELTA_LF_SMALL: i32 = 3;
const DELTA_LF_PROBS: usize = 3;

/// `write_delta_lflevel` (`av1/encoder/bitstream.c`): the per-superblock delta
/// loop-filter level — same exp-Golomb delta coding as [`write_delta_qindex`]
/// (`DELTA_LF_SMALL == DELTA_Q_SMALL == 3`), on the caller-selected delta-lf CDF
/// (the single `delta_lf_cdf` or, for `delta_lf_multi`, `delta_lf_multi_cdf[lf_id]`).
pub fn write_delta_lflevel(enc: &mut OdEcEnc, delta_lf_cdf: &mut [u16], delta_lflevel: i32) {
    let sign = delta_lflevel < 0;
    let abs = delta_lflevel.abs();
    let smallval = abs < DELTA_LF_SMALL;
    write_symbol(enc, abs.min(DELTA_LF_SMALL), delta_lf_cdf, DELTA_LF_PROBS + 1);
    if !smallval {
        let rem_bits = get_msb((abs - 1) as u32) as i32;
        let thr = (1 << rem_bits) + 1;
        write_literal(enc, rem_bits - 1, 3);
        write_literal(enc, abs - thr, rem_bits as u32);
    }
    if abs > 0 {
        write_bit(enc, sign as i32);
    }
}

const CFL_JOINT_SIGNS: usize = 8;
const CFL_ALPHABET_SIZE: usize = 16;
const CFL_SIGNS: i32 = 3;

fn cfl_sign_u(js: i32) -> i32 {
    ((js + 1) * 11) >> 5
}
fn cfl_sign_v(js: i32) -> i32 {
    (js + 1) - CFL_SIGNS * cfl_sign_u(js)
}
fn cfl_context_u(js: i32) -> i32 {
    js + 1 - CFL_SIGNS
}
fn cfl_context_v(js: i32) -> i32 {
    cfl_sign_v(js) * CFL_SIGNS + cfl_sign_u(js) - CFL_SIGNS
}

/// `write_cfl_alphas` (`av1/encoder/bitstream.c`): the chroma-from-luma alpha coding —
/// the joint-sign symbol on `cfl_sign_cdf` (8 symbols), then, for each plane whose sign
/// is nonzero, the 4-bit alpha magnitude (`CFL_IDX_U/V(idx)`) on `cfl_alpha_cdf` at the
/// plane's derived context. `cfl_alpha_cdf` holds the 6 context CDFs (17 entries each),
/// all adapted in place.
pub fn write_cfl_alphas(
    enc: &mut OdEcEnc,
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
    idx: i32,
    joint_sign: i32,
) {
    write_symbol(enc, joint_sign, cfl_sign_cdf, CFL_JOINT_SIGNS);
    if cfl_sign_u(joint_sign) != 0 {
        let ctx = cfl_context_u(joint_sign) as usize;
        write_symbol(enc, idx >> 4, &mut cfl_alpha_cdf[ctx], CFL_ALPHABET_SIZE);
    }
    if cfl_sign_v(joint_sign) != 0 {
        let ctx = cfl_context_v(joint_sign) as usize;
        write_symbol(enc, idx & 15, &mut cfl_alpha_cdf[ctx], CFL_ALPHABET_SIZE);
    }
}

const INTRA_MODES: usize = 13;
/// `intra_mode_context[INTRA_MODES]` (`common_data.h`): maps a Y prediction mode to
/// its keyframe Y-mode CDF context.
const INTRA_MODE_CONTEXT: [usize; INTRA_MODES] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `get_y_mode_cdf` context (`av1_common_int.h`): `(intra_mode_context[above_mode],
/// intra_mode_context[left_mode])` selecting `kf_y_cdf[above_ctx][left_ctx]`. An absent
/// neighbour resolves to `DC_PRED` (0).
pub fn get_y_mode_ctx(above_mode: Option<i32>, left_mode: Option<i32>) -> (usize, usize) {
    let a = above_mode.unwrap_or(0) as usize;
    let l = left_mode.unwrap_or(0) as usize;
    (INTRA_MODE_CONTEXT[a], INTRA_MODE_CONTEXT[l])
}

/// `write_intra_y_mode_kf` (`av1/encoder/bitstream.c`): the keyframe intra luma mode —
/// `aom_write_symbol(mode, kf_y_cdf[above_ctx][left_ctx], INTRA_MODES)` (adapted). The
/// caller selects the CDF via [`get_y_mode_ctx`].
pub fn write_intra_y_mode_kf(enc: &mut OdEcEnc, kf_y_cdf: &mut [u16], mode: i32) {
    write_symbol(enc, mode, kf_y_cdf, INTRA_MODES);
}

const UV_INTRA_MODES: usize = 14;
/// `size_group_lookup[BLOCK_SIZES_ALL]` (`common_data.h`): the non-keyframe Y-mode CDF
/// context (one of 4 size groups) for a block size.
const SIZE_GROUP_LOOKUP: [usize; 22] =
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2];

/// `size_group_lookup[bsize]` — selects `y_mode_cdf[size_group]` for non-keyframe intra.
pub fn y_mode_size_group(bsize: usize) -> usize {
    SIZE_GROUP_LOOKUP[bsize]
}

/// `write_intra_y_mode_nonkf` (`av1/encoder/bitstream.c`): the non-keyframe intra luma
/// mode — `aom_write_symbol(mode, y_mode_cdf[size_group_lookup[bsize]], INTRA_MODES)`
/// (adapted). Same symbol write as the keyframe variant on a size-group-selected CDF.
pub fn write_intra_y_mode_nonkf(enc: &mut OdEcEnc, y_mode_cdf: &mut [u16], mode: i32) {
    write_symbol(enc, mode, y_mode_cdf, INTRA_MODES);
}

/// `write_intra_uv_mode` (`av1/encoder/bitstream.c`): the intra chroma mode on the
/// (cfl-allowed, y-mode)-selected CDF — `UV_INTRA_MODES` symbols when CFL is allowed,
/// one fewer (no CFL_PRED) when not.
pub fn write_intra_uv_mode(enc: &mut OdEcEnc, uv_mode_cdf: &mut [u16], uv_mode: i32, cfl_allowed: bool) {
    let n = UV_INTRA_MODES - (!cfl_allowed) as usize;
    write_symbol(enc, uv_mode, uv_mode_cdf, n);
}

const NEARESTMV: i32 = 13;
const NEARMV: i32 = 14;
const GLOBALMV: i32 = 15;
const NEWMV: i32 = 16;

/// `write_inter_mode` (`av1/encoder/bitstream.c`): the single-reference inter mode as a
/// cascade of three binary symbols keyed off `mode_ctx` — is-not-NEWMV on
/// `newmv_cdf[mode_ctx & 7]`, then (if not NEWMV) is-not-GLOBALMV on
/// `zeromv_cdf[(mode_ctx>>3) & 1]`, then (if not GLOBALMV) is-not-NEARESTMV on
/// `refmv_cdf[(mode_ctx>>4) & 15]`. Only the CDFs on the taken path adapt.
pub fn write_inter_mode(
    enc: &mut OdEcEnc,
    newmv_cdf: &mut [[u16; 3]; 6],
    zeromv_cdf: &mut [[u16; 3]; 2],
    refmv_cdf: &mut [[u16; 3]; 6],
    mode: i32,
    mode_ctx: i32,
) {
    let newmv_ctx = (mode_ctx & 7) as usize;
    write_symbol(enc, (mode != NEWMV) as i32, &mut newmv_cdf[newmv_ctx], 2);
    if mode != NEWMV {
        let zeromv_ctx = ((mode_ctx >> 3) & 1) as usize;
        write_symbol(enc, (mode != GLOBALMV) as i32, &mut zeromv_cdf[zeromv_ctx], 2);
        if mode != GLOBALMV {
            let refmv_ctx = ((mode_ctx >> 4) & 15) as usize;
            write_symbol(enc, (mode != NEARESTMV) as i32, &mut refmv_cdf[refmv_ctx], 2);
        }
    }
}

const REF_CAT_LEVEL: u16 = 640;
const NEW_NEWMV: i32 = 24;

/// `av1_drl_ctx` (`mvref_common.h`): the DRL CDF context from the two candidate ref-mv
/// weights around `ref_idx` relative to `REF_CAT_LEVEL`.
fn av1_drl_ctx(weight: &[u16], ref_idx: usize) -> usize {
    let a = weight[ref_idx] >= REF_CAT_LEVEL;
    let b = weight[ref_idx + 1] >= REF_CAT_LEVEL;
    if a && b {
        0
    } else if a && !b {
        1
    } else if !a && !b {
        2
    } else {
        0
    }
}

/// `have_nearmv_in_inter_mode` (`blockd.h`): the mode uses a NEAR mv component.
pub fn have_nearmv_in_inter_mode(mode: i32) -> bool {
    // NEARMV=14, NEAR_NEARMV=18, NEAR_NEWMV=21, NEW_NEARMV=22
    mode == 14 || mode == 18 || mode == 21 || mode == 22
}

/// `write_drl_idx` (`av1/encoder/bitstream.c`): the dynamic-ref-list index — up to two
/// binary symbols selecting `ref_mv_idx` among the candidate ref MVs, on the
/// weight-derived DRL CDF context. NEWMV modes scan idx 0..1; NEAR modes scan idx 1..2
/// (offset by the NEARESTMV slot). Stops once the chosen index is coded.
pub fn write_drl_idx(
    enc: &mut OdEcEnc,
    drl_cdf: &mut [[u16; 3]; 3],
    mode: i32,
    ref_mv_idx: i32,
    ref_mv_count: i32,
    weight: &[u16],
) {
    let new_mv = mode == NEWMV || mode == NEW_NEWMV;
    if new_mv {
        for idx in 0..2 {
            if ref_mv_count > idx + 1 {
                let ctx = av1_drl_ctx(weight, idx as usize);
                write_symbol(enc, (ref_mv_idx != idx) as i32, &mut drl_cdf[ctx], 2);
                if ref_mv_idx == idx {
                    return;
                }
            }
        }
        return;
    }
    if have_nearmv_in_inter_mode(mode) {
        for idx in 1..3 {
            if ref_mv_count > idx + 1 {
                let ctx = av1_drl_ctx(weight, idx as usize);
                write_symbol(enc, (ref_mv_idx != idx - 1) as i32, &mut drl_cdf[ctx], 2);
                if ref_mv_idx == idx - 1 {
                    return;
                }
            }
        }
    }
}

const CLASS0_SIZE_MV: i32 = 2;

/// `av1_get_mv_joint` (`encodemv.h`): the MV joint type from which components are
/// nonzero — `(col!=0) | ((row!=0)<<1)` (ZERO=0, HNZVZ=1, HZVNZ=2, HNZVNZ=3).
pub fn get_mv_joint(row: i32, col: i32) -> i32 {
    (col != 0) as i32 | (((row != 0) as i32) << 1)
}

/// `av1_mv_class_base` (`encodemv.h`): `c ? CLASS0_SIZE << (c+2) : 0`.
fn mv_class_base(c: i32) -> i32 {
    if c != 0 { CLASS0_SIZE_MV << (c + 2) } else { 0 }
}

/// `av1_get_mv_class` (`encodemv.h`): the magnitude class of `z` (= |mv_diff|-1) and its
/// offset within the class — `class = log2(z>>3)` (0 when `z>>3 == 0`),
/// `offset = z - av1_mv_class_base(class)`.
pub fn get_mv_class(z: i32) -> (i32, i32) {
    let zz = (z >> 3) as u32;
    let c = if zz == 0 { 0 } else { 31 - zz.leading_zeros() as i32 };
    (c, z - mv_class_base(c))
}

const MV_CLASSES: usize = 11;
const MV_FP_SIZE: usize = 4;
const MV_SUBPEL_NONE: i32 = -1;
const MV_SUBPEL_LOW: i32 = 0;

// nmv_component CDFs packed in one 69-u16 blob:
//   sign     0..3    (2-sym)
//   classes  3..15   (11-sym)
//   class0   15..18  (2-sym)
//   bits[10] 18..48  (2-sym each, 10 CDFs)
//   class0_fp[2] 48..58 (4-sym each)
//   fp       58..63  (4-sym)
//   class0_hp 63..66 (2-sym)
//   hp       66..69  (2-sym)
/// `encode_mv_component` (`av1/encoder/encodemv.c`): one MV-difference component —
/// sign, magnitude class, then the class-0 integer symbol or the per-class integer
/// bits, then (precision-gated) the fractional bits and the high-precision bit, all on
/// the component's adapted nmv CDFs.
pub fn encode_mv_component(enc: &mut OdEcEnc, cdf: &mut [u16; 69], comp: i32, precision: i32) {
    let sign = (comp < 0) as i32;
    let mag = comp.abs();
    let (mv_class, offset) = get_mv_class(mag - 1);
    let d = offset >> 3;
    let fr = (offset >> 1) & 3;
    let hp = offset & 1;

    write_symbol(enc, sign, &mut cdf[0..3], 2);
    write_symbol(enc, mv_class, &mut cdf[3..15], MV_CLASSES);
    if mv_class == 0 {
        write_symbol(enc, d, &mut cdf[15..18], 2);
    } else {
        let n = mv_class; // mv_class + CLASS0_BITS(1) - 1
        for i in 0..n {
            let off = 18 + (i as usize) * 3;
            write_symbol(enc, (d >> i) & 1, &mut cdf[off..off + 3], 2);
        }
    }
    if precision > MV_SUBPEL_NONE {
        if mv_class == 0 {
            let off = 48 + (d as usize) * 5;
            write_symbol(enc, fr, &mut cdf[off..off + 5], MV_FP_SIZE);
        } else {
            write_symbol(enc, fr, &mut cdf[58..63], MV_FP_SIZE);
        }
    }
    if precision > MV_SUBPEL_LOW {
        if mv_class == 0 {
            write_symbol(enc, hp, &mut cdf[63..66], 2);
        } else {
            write_symbol(enc, hp, &mut cdf[66..69], 2);
        }
    }
}

/// `av1_encode_mv` (`av1/encoder/encodemv.c`): a motion-vector difference — the MV
/// joint symbol on `joints_cdf` (MV_JOINTS=4), then the vertical component (when the
/// joint has a nonzero row) and the horizontal component (nonzero col), each via
/// [`encode_mv_component`] at precision `usehp` (the caller forces `MV_SUBPEL_NONE`
/// under integer-mv). `diff_*` are the already-computed mv-minus-ref components.
pub fn encode_mv(
    enc: &mut OdEcEnc,
    joints_cdf: &mut [u16],
    comp0: &mut [u16; 69],
    comp1: &mut [u16; 69],
    diff_row: i32,
    diff_col: i32,
    usehp: i32,
) {
    let j = get_mv_joint(diff_row, diff_col);
    write_symbol(enc, j, joints_cdf, 4); // MV_JOINTS
    if j == 2 || j == 3 {
        // mv_joint_vertical
        encode_mv_component(enc, comp0, diff_row, usehp);
    }
    if j == 1 || j == 3 {
        // mv_joint_horizontal
        encode_mv_component(enc, comp1, diff_col, usehp);
    }
}

const MAX_ANGLE_DELTA: i32 = 3;

/// `write_angle_delta` (`av1/encoder/bitstream.c`): the intra directional-mode angle
/// delta — `aom_write_symbol(angle_delta + MAX_ANGLE_DELTA, cdf, 2*MAX_ANGLE_DELTA+1)`
/// (7 symbols, adapted) on the caller-selected per-mode angle CDF.
pub fn write_angle_delta(enc: &mut OdEcEnc, cdf: &mut [u16], angle_delta: i32) {
    write_symbol(enc, angle_delta + MAX_ANGLE_DELTA, cdf, (2 * MAX_ANGLE_DELTA + 1) as usize);
}

/// `bsize_to_max_depth` (`blockd.h`): the max TX-split depth signalled for a block size.
const BSIZE_TO_MAX_DEPTH: [usize; 22] =
    [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
/// `bsize_to_tx_size_cat` table (`blockd.h`) before the `-1`.
const BSIZE_TO_TX_SIZE_DEPTH: [i32; 22] =
    [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4];

/// `bsize_to_max_depth`.
pub fn bsize_to_max_depth(bsize: usize) -> usize {
    BSIZE_TO_MAX_DEPTH[bsize]
}

/// `bsize_to_tx_size_cat` (`blockd.h`): `bsize_to_tx_size_depth_table[bsize] - 1` — the
/// TX-size CDF category (only meaningful for `bsize > BLOCK_4X4`).
pub fn bsize_to_tx_size_cat(bsize: usize) -> i32 {
    BSIZE_TO_TX_SIZE_DEPTH[bsize] - 1
}

/// `write_selected_tx_size` (`av1/encoder/bitstream.c`): for a block that signals its
/// TX size (`bsize > BLOCK_4X4`), the split `depth` on the (category, context)-selected
/// `tx_size_cdf` at `max_depths + 1` symbols. `depth` / the CDF selection
/// (`tx_size_to_depth` + `get_tx_size_context` + `bsize_to_tx_size_cat`) are resolved by
/// the caller.
pub fn write_selected_tx_size(
    enc: &mut OdEcEnc,
    tx_size_cdf: &mut [u16],
    bsize: usize,
    depth: i32,
    max_depths: usize,
) {
    if bsize > 0 {
        // block_signals_txsize: bsize > BLOCK_4X4
        write_symbol(enc, depth, tx_size_cdf, max_depths + 1);
    }
}

// ---- tx-size selection: context + txfm-context maintenance ------------------------
//
// The `get_tx_size_context` facade and the `above_txfm_context` /
// `left_txfm_context` array maintenance (`av1/common/av1_common_int.h` +
// `av1/common/pred_common.h`). NOTE: a direct differential test of these against
// a C shim is DEFERRED — `aom-sys-ref` had live encoder-track WIP when this
// landed, so the port is validated by (a) unit tests against hand-traced C
// semantics (`tests/tx_size_ctx.rs`) and (b) the encode/decode tile roundtrip in
// `aom-decode`, whose write side uses this same facade (lockstep catches any
// decode/encode asymmetry; a *shared* misreading of the C would not be caught —
// hence the deferred C diff is tracked for the coordinator).

/// `TX_MODE` (`enums.h`): the frame transform mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TxMode {
    /// `ONLY_4X4`: use only the 4x4 transform.
    Only4x4 = 0,
    /// `TX_MODE_LARGEST`: the largest transform for the block size, no bits coded.
    Largest = 1,
    /// `TX_MODE_SELECT`: the transform size is coded per block.
    Select = 2,
}

/// `max_txsize_lookup[BLOCK_SIZES_ALL]` (`common_data.h`): the largest **square**
/// transform fitting the block.
const MAX_TXSIZE_LOOKUP: [usize; 22] =
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 0, 1, 1, 2, 2];
/// `txsize_sqr_map[TX_SIZES_ALL]` (`common_data.h`): the largest square transform
/// **contained in** the transform.
const TXSIZE_SQR_MAP: [usize; 19] = [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
/// `tx_mode_to_biggest_tx_size[TX_MODES]` (`common_data.h`): TX_4X4 / TX_64X64 /
/// TX_64X64.
const TX_MODE_TO_BIGGEST_TX_SIZE: [usize; 3] = [0, 4, 4];

/// `tx_size_from_tx_mode` (`blockd.h`): the block's transform size when the frame
/// tx mode does not code per-block sizes (also the non-SELECT fallback inside
/// `read_tx_size`). For `TX_MODE_LARGEST` this is `max_txsize_rect_lookup[bsize]`
/// for every block size. For `ONLY_4X4` the C function returns the rect max for
/// the 2:1/4:1 4-row/4-col blocks (their `txsize_sqr_map` is `TX_4X4`, passing
/// the `<=` gate) and `TX_4X4` otherwise — but `ONLY_4X4` occurs only with
/// `coded_lossless`, where `read_tx_size`'s lossless branch preempts with
/// `TX_4X4` before this function is reached (its `assert(IMPLIES(ONLY_4X4,
/// bsize == BLOCK_4X4))` documents that); ported verbatim regardless.
pub fn tx_size_from_tx_mode(bsize: usize, tx_mode: TxMode) -> usize {
    let largest_tx_size = TX_MODE_TO_BIGGEST_TX_SIZE[tx_mode as usize];
    let max_rect_tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
    if bsize == 0 {
        // BLOCK_4X4: AOMMIN on the TX_SIZE enum values
        return MAX_TXSIZE_LOOKUP[bsize].min(largest_tx_size);
    }
    if TXSIZE_SQR_MAP[max_rect_tx_size] <= largest_tx_size {
        max_rect_tx_size
    } else {
        largest_tx_size
    }
}

/// `depth_to_tx_size` (`blockd.h`): walk `depth` `sub_tx_size_map` steps down from
/// the block's `max_txsize_rect_lookup` size — the inverse of the coded tx-size
/// depth symbol.
pub fn depth_to_tx_size(depth: i32, bsize: usize) -> usize {
    let mut tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
    for _ in 0..depth {
        tx_size = SUB_TX_SIZE_MAP[tx_size];
    }
    tx_size
}

/// `tx_size_to_depth` (`av1/encoder/block.h`): the number of `sub_tx_size_map`
/// steps from `max_txsize_rect_lookup[bsize]` down to `tx_size` — the coded
/// depth symbol on the write side.
pub fn tx_size_to_depth(tx_size: usize, bsize: usize) -> i32 {
    let mut ctx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
    let mut depth = 0;
    while tx_size != ctx_size {
        depth += 1;
        ctx_size = SUB_TX_SIZE_MAP[ctx_size];
        debug_assert!(depth <= MAX_VARTX_DEPTH); // MAX_TX_DEPTH == 2
    }
    depth
}

/// The txfm-context array reset value (`av1_zero_above_context` /
/// `av1_zero_left_context` in `av1_common_int.h`): both memset their array to
/// `tx_size_wide[TX_SIZES_LARGEST]` == `tx_size_high[TX_SIZES_LARGEST]` == 64
/// (`TX_SIZES_LARGEST` is `TX_64X64`, `aom_dsp/txfm_common.h`). The above array
/// is reset once per tile over the superblock-aligned tile width; the left array
/// (`MAX_MIB_SIZE` = 32 entries) once per superblock row.
pub const TXFM_CTX_INIT: u8 = 64;

/// `get_tx_size_context` (`av1/common/pred_common.h`): the tx-size symbol context
/// (0..=2) from the above/left txfm-context bytes (the entries at the block's
/// `mi_col` / `mi_row & 31` — pixel width/height of the neighbouring transform,
/// maintained by [`set_txfm_ctxs`]) compared against this block's max rect
/// transform dims. An **inter** neighbour substitutes its block dims for its
/// txfm-context byte; `above_inter_bsize` / `left_inter_bsize` carry that
/// neighbour's `bsize` when it exists **and** `is_inter_block(..)` (pass `None`
/// for an intra neighbour or when unavailable — always `None` on a KEY frame
/// without intrabc). The context is the sum/copy of the available directions.
pub fn get_tx_size_context(
    bsize: usize,
    above_txfm_ctx: u8,
    left_txfm_ctx: u8,
    has_above: bool,
    has_left: bool,
    above_inter_bsize: Option<usize>,
    left_inter_bsize: Option<usize>,
) -> usize {
    let max_tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
    let max_tx_wide = TX_SIZE_WIDE[max_tx_size];
    let max_tx_high = TX_SIZE_HIGH[max_tx_size];

    let mut above = (i32::from(above_txfm_ctx) >= max_tx_wide) as usize;
    let mut left = (i32::from(left_txfm_ctx) >= max_tx_high) as usize;

    if has_above {
        if let Some(ab) = above_inter_bsize {
            above = (BLOCK_SIZE_WIDE[ab] >= max_tx_wide) as usize;
        }
    }
    if has_left {
        if let Some(lb) = left_inter_bsize {
            left = (BLOCK_SIZE_HIGH[lb] >= max_tx_high) as usize;
        }
    }

    if has_above && has_left {
        above + left
    } else if has_above {
        above
    } else if has_left {
        left
    } else {
        0
    }
}

/// `set_txfm_ctxs` (`av1_common_int.h`): after the block's `tx_size` is
/// read/written, stamp the above/left txfm-context arrays over the block's
/// **full** (not frame-clipped) mi footprint — `above[0..n4_w]` with the tx
/// width in pixels and `left[0..n4_h]` with the tx height; a **skipped inter**
/// block (`skip` = `skip_txfm && is_inter_block`, always false for intra)
/// stamps the block's pixel dims instead. `above` starts at the block's
/// `mi_col` entry of the tile array; `left` at `mi_row & 31`.
pub fn set_txfm_ctxs(
    above: &mut [u8],
    left: &mut [u8],
    tx_size: usize,
    n4_w: usize,
    n4_h: usize,
    skip: bool,
) {
    let mut bw = TX_SIZE_WIDE[tx_size] as u8;
    let mut bh = TX_SIZE_HIGH[tx_size] as u8;
    if skip {
        bw = (n4_w * 4) as u8; // n4_w * MI_SIZE
        bh = (n4_h * 4) as u8;
    }
    // set_txfm_ctx(above_txfm_context, bw, n4_w) / (left_txfm_context, bh, n4_h)
    above[..n4_w].fill(bw);
    left[..n4_h].fill(bh);
}

const FILTER_INTRA_MODES: usize = 5;

/// `write_filter_intra_mode_info` (`av1/encoder/bitstream.c`): when filter-intra is
/// allowed for the block, the use-filter-intra flag on `filter_intra_cdfs[bsize]`
/// (2 symbols), then, if used, the filter-intra mode on `filter_intra_mode_cdf`
/// (`FILTER_INTRA_MODES`=5). The `allowed` gate (`av1_filter_intra_allowed`) and the
/// bsize CDF selection are the caller's.
pub fn write_filter_intra_mode_info(
    enc: &mut OdEcEnc,
    use_cdf: &mut [u16],
    mode_cdf: &mut [u16],
    allowed: bool,
    use_filter_intra: i32,
    mode: i32,
) {
    if allowed {
        write_symbol(enc, use_filter_intra, use_cdf, 2);
        if use_filter_intra != 0 {
            write_symbol(enc, mode, mode_cdf, FILTER_INTRA_MODES);
        }
    }
}

const NEAREST_NEARESTMV: i32 = 17;
const INTER_COMPOUND_MODES: usize = 8;

/// `write_inter_compound_mode` (`av1/encoder/bitstream.c`): the compound inter mode —
/// `aom_write_symbol(mode - NEAREST_NEARESTMV, inter_compound_mode_cdf[mode_ctx],
/// INTER_COMPOUND_MODES=8)` (adapted) on the caller-selected CDF.
pub fn write_inter_compound_mode(enc: &mut OdEcEnc, cdf: &mut [u16], mode: i32) {
    write_symbol(enc, mode - NEAREST_NEARESTMV, cdf, INTER_COMPOUND_MODES);
}

/// `write_is_inter` (`av1/encoder/bitstream.c`): the intra/inter flag — coded on the
/// context-selected `intra_inter_cdf` (2 symbols) unless the segment fixes the
/// reference frame (nothing coded) or forces global-mv (implied inter, nothing coded).
pub fn write_is_inter(
    enc: &mut OdEcEnc,
    intra_inter_cdf: &mut [u16],
    seg_ref_frame_active: bool,
    seg_globalmv_active: bool,
    is_inter: i32,
) {
    if !seg_ref_frame_active {
        if seg_globalmv_active {
            return;
        }
        write_symbol(enc, is_inter, intra_inter_cdf, 2);
    }
}

const MOTION_MODES: usize = 3;

/// `write_motion_mode` (`av1/encoder/bitstream.c`): the block motion mode, gated by the
/// caller-resolved `last_motion_mode_allowed` — nothing for SIMPLE_TRANSLATION(0), a
/// 2-symbol OBMC flag on `obmc_cdf[bsize]` when OBMC_CAUSAL(1) is the ceiling, else the
/// full mode on `motion_mode_cdf[bsize]` (MOTION_MODES=3).
pub fn write_motion_mode(
    enc: &mut OdEcEnc,
    obmc_cdf: &mut [u16],
    motion_mode_cdf: &mut [u16],
    last_motion_mode_allowed: i32,
    motion_mode: i32,
) {
    match last_motion_mode_allowed {
        0 => {}
        1 => write_symbol(enc, (motion_mode == 1) as i32, obmc_cdf, 2),
        _ => write_symbol(enc, motion_mode, motion_mode_cdf, MOTION_MODES),
    }
}

const SWITCHABLE_FILTERS: usize = 3;

/// `write_mb_interp_filter` (`av1/encoder/bitstream.c`): the per-block interpolation
/// filter. Nothing is coded when interp isn't needed or the frame filter isn't
/// SWITCHABLE; otherwise the horizontal filter on the ctx-selected
/// `switchable_interp_cdf` and, when dual-filter is enabled, the vertical filter on its
/// own ctx-selected CDF (`SWITCHABLE_FILTERS`=3 each). Contexts + gates are the caller's.
#[allow(clippy::too_many_arguments)]
pub fn write_mb_interp_filter(
    enc: &mut OdEcEnc,
    cdf0: &mut [u16],
    cdf1: &mut [u16],
    interp_needed: bool,
    is_switchable: bool,
    enable_dual_filter: bool,
    filter0: i32,
    filter1: i32,
) {
    if !interp_needed {
        return;
    }
    if is_switchable {
        write_symbol(enc, filter0, cdf0, SWITCHABLE_FILTERS);
        if !enable_dual_filter {
            return;
        }
        write_symbol(enc, filter1, cdf1, SWITCHABLE_FILTERS);
    }
}

/// `av1_get_intra_inter_context` (`av1/common/pred_common.c`): the intra/inter CDF
/// context from the above/left neighbours' inter-ness (`is_inter_block`) and edge
/// availability. Returns 0..3.
pub fn get_intra_inter_context(
    has_above: bool,
    above_inter: bool,
    has_left: bool,
    left_inter: bool,
) -> i32 {
    if has_above && has_left {
        let ai = !above_inter;
        let li = !left_inter;
        if li && ai {
            3
        } else {
            (li || ai) as i32
        }
    } else if has_above || has_left {
        let nbr_inter = if has_above { above_inter } else { left_inter };
        2 * (!nbr_inter) as i32
    } else {
        0
    }
}

#[inline]
fn is_backward_ref(ref_frame: i32) -> bool {
    ref_frame >= 5 // BWDREF_FRAME
}
#[inline]
fn has_second_ref(ref1: i32) -> bool {
    ref1 > 0 // > INTRA_FRAME
}
#[inline]
fn nbr_is_inter(use_intrabc: bool, ref0: i32) -> bool {
    use_intrabc || ref0 > 0
}

/// `av1_get_reference_mode_context` (`av1/common/pred_common.c`): the single-vs-compound
/// reference-mode CDF context from the above/left neighbours' comp-pred use, backward-ref
/// direction, and inter-ness. Returns 0..4.
#[allow(clippy::too_many_arguments)]
pub fn get_reference_mode_context(
    has_above: bool,
    a_r0: i32,
    a_r1: i32,
    a_ibc: bool,
    has_left: bool,
    l_r0: i32,
    l_r1: i32,
    l_ibc: bool,
) -> i32 {
    if has_above && has_left {
        if !has_second_ref(a_r1) && !has_second_ref(l_r1) {
            (is_backward_ref(a_r0) ^ is_backward_ref(l_r0)) as i32
        } else if !has_second_ref(a_r1) {
            2 + (is_backward_ref(a_r0) || !nbr_is_inter(a_ibc, a_r0)) as i32
        } else if !has_second_ref(l_r1) {
            2 + (is_backward_ref(l_r0) || !nbr_is_inter(l_ibc, l_r0)) as i32
        } else {
            4
        }
    } else if has_above || has_left {
        let (r0, r1) = if has_above { (a_r0, a_r1) } else { (l_r0, l_r1) };
        if !has_second_ref(r1) {
            is_backward_ref(r0) as i32
        } else {
            3
        }
    } else {
        1
    }
}

#[inline]
fn has_uni_comp_refs_h(r0: i32, r1: i32) -> bool {
    has_second_ref(r1) && (is_backward_ref(r0) == is_backward_ref(r1))
}

/// `av1_get_comp_reference_type_context` (`av1/common/pred_common.c`): the
/// unidirectional-vs-bidirectional compound-reference CDF context from the above/left
/// neighbours' compound-ref structure. Returns 0..4.
#[allow(clippy::too_many_arguments)]
pub fn get_comp_reference_type_context(
    ha: bool,
    a_r0: i32,
    a_r1: i32,
    a_ibc: bool,
    hl: bool,
    l_r0: i32,
    l_r1: i32,
    l_ibc: bool,
) -> i32 {
    if ha && hl {
        let above_intra = !nbr_is_inter(a_ibc, a_r0);
        let left_intra = !nbr_is_inter(l_ibc, l_r0);
        if above_intra && left_intra {
            2
        } else if above_intra || left_intra {
            let (r0, r1) = if above_intra { (l_r0, l_r1) } else { (a_r0, a_r1) };
            if !has_second_ref(r1) {
                2
            } else {
                1 + 2 * has_uni_comp_refs_h(r0, r1) as i32
            }
        } else {
            let a_sg = !has_second_ref(a_r1);
            let l_sg = !has_second_ref(l_r1);
            if a_sg && l_sg {
                1 + 2 * (!(is_backward_ref(a_r0) ^ is_backward_ref(l_r0))) as i32
            } else if l_sg || a_sg {
                let uni_rfc = if a_sg {
                    has_uni_comp_refs_h(l_r0, l_r1)
                } else {
                    has_uni_comp_refs_h(a_r0, a_r1)
                };
                if !uni_rfc {
                    1
                } else {
                    3 + (!(is_backward_ref(a_r0) ^ is_backward_ref(l_r0))) as i32
                }
            } else {
                let a_uni = has_uni_comp_refs_h(a_r0, a_r1);
                let l_uni = has_uni_comp_refs_h(l_r0, l_r1);
                if !a_uni && !l_uni {
                    0
                } else if !a_uni || !l_uni {
                    2
                } else {
                    // exact == BWDREF_FRAME here (not >=)
                    3 + (!((a_r0 == 5) ^ (l_r0 == 5))) as i32
                }
            }
        }
    } else if ha || hl {
        let (r0, r1, ibc) = if ha { (a_r0, a_r1, a_ibc) } else { (l_r0, l_r1, l_ibc) };
        // intra edge, or inter single-pred -> 2 (merged; the C writes them separately)
        if !nbr_is_inter(ibc, r0) || !has_second_ref(r1) {
            2
        } else {
            4 * has_uni_comp_refs_h(r0, r1) as i32
        }
    } else {
        2
    }
}

/// `av1_get_pred_context_single_ref_p1` (`pred_common.c`): the P1 single-ref CDF context
/// from the neighbours' forward vs backward reference counts — 1 if equal, 0 if forward
/// < backward, else 2. `ref_counts` is `neighbors_ref_counts[REF_FRAMES]`.
pub fn single_ref_p1_context(ref_counts: &[u8; 8]) -> i32 {
    let fwd = ref_counts[1] as i32 + ref_counts[2] as i32 + ref_counts[3] as i32 + ref_counts[4] as i32;
    let bwd = ref_counts[5] as i32 + ref_counts[6] as i32 + ref_counts[7] as i32;
    if fwd == bwd {
        1
    } else if fwd < bwd {
        0
    } else {
        2
    }
}

#[inline]
fn ref_count_ctx(a: i32, b: i32) -> i32 {
    if a == b {
        1
    } else if a < b {
        0
    } else {
        2
    }
}

/// `get_pred_context_brfarf2_or_arf` (`pred_common.c`): (BWDREF+ALTREF2) vs ALTREF counts.
/// = single_ref P2 / comp_bwdref P0.
pub fn pred_ctx_brfarf2_or_arf(rc: &[u8; 8]) -> i32 {
    ref_count_ctx(rc[5] as i32 + rc[6] as i32, rc[7] as i32)
}
/// `get_pred_context_ll2_or_l3gld`: (LAST+LAST2) vs (LAST3+GOLDEN). = single_ref P3 / comp_ref P0.
pub fn pred_ctx_ll2_or_l3gld(rc: &[u8; 8]) -> i32 {
    ref_count_ctx(rc[1] as i32 + rc[2] as i32, rc[3] as i32 + rc[4] as i32)
}
/// `get_pred_context_last_or_last2`: LAST vs LAST2. = single_ref P4 / comp_ref P1.
pub fn pred_ctx_last_or_last2(rc: &[u8; 8]) -> i32 {
    ref_count_ctx(rc[1] as i32, rc[2] as i32)
}
/// `get_pred_context_last3_or_gld`: LAST3 vs GOLDEN. = single_ref P5 / comp_ref P2.
pub fn pred_ctx_last3_or_gld(rc: &[u8; 8]) -> i32 {
    ref_count_ctx(rc[3] as i32, rc[4] as i32)
}
/// `get_pred_context_brf_or_arf2`: BWDREF vs ALTREF2. = single_ref P6 / comp_bwdref P1.
pub fn pred_ctx_brf_or_arf2(rc: &[u8; 8]) -> i32 {
    ref_count_ctx(rc[5] as i32, rc[6] as i32)
}

/// `av1_get_pred_context_uni_comp_ref_p1`: LAST2 vs (LAST3+GOLDEN) counts. The other two
/// uni-comp-ref contexts reuse [`single_ref_p1_context`] (fwd vs bwd) and
/// [`pred_ctx_last3_or_gld`] (LAST3 vs GOLDEN) — same count groupings.
pub fn pred_ctx_last2_or_l3gld(rc: &[u8; 8]) -> i32 {
    ref_count_ctx(rc[2] as i32, rc[3] as i32 + rc[4] as i32)
}

// write_ref_frames CDF blob slots (each a 3-entry 2-symbol CDF):
//  0 reference_mode(comp_inter) | 1 comp_ref_type | 2..4 uni_comp_ref p/p1/p2
//  5..7 comp_ref p/p1/p2 | 8..9 comp_bwdref p/p1 | 10..15 single_ref p1..p6
/// `write_ref_frames` (`av1/encoder/bitstream.c`): the block reference-frame signalling
/// cascade. Seg-fixed refs code nothing; otherwise the reference-mode SELECT bit (when
/// allowed), then for compound the uni/bi split and its ref bits, or for single the
/// backward/forward ref bit-tree — each `WRITE_REF_BIT` an `aom_write_symbol` on the
/// pred-context-selected 2-symbol CDF (the caller pre-selects each via the validated
/// context helpers). Only the CDFs on the taken path adapt.
#[allow(clippy::too_many_arguments)]
pub fn write_ref_frames(
    enc: &mut OdEcEnc,
    cdfs: &mut [[u16; 3]; 16],
    seg_ref_active: bool,
    seg_skipgmv_active: bool,
    reference_mode_is_select: bool,
    is_comp_ref_allowed: bool,
    is_compound: bool,
    comp_ref_type: i32,
    ref0: i32,
    ref1: i32,
) {
    if seg_ref_active || seg_skipgmv_active {
        return;
    }
    if reference_mode_is_select && is_comp_ref_allowed {
        write_symbol(enc, is_compound as i32, &mut cdfs[0], 2);
    }
    if is_compound {
        write_symbol(enc, comp_ref_type, &mut cdfs[1], 2);
        if comp_ref_type == 0 {
            // UNIDIR_COMP_REFERENCE
            let bit = (ref0 == 5) as i32; // BWDREF
            write_symbol(enc, bit, &mut cdfs[2], 2);
            if bit == 0 {
                let bit1 = (ref1 == 3 || ref1 == 4) as i32; // LAST3 || GOLDEN
                write_symbol(enc, bit1, &mut cdfs[3], 2);
                if bit1 != 0 {
                    write_symbol(enc, (ref1 == 4) as i32, &mut cdfs[4], 2); // GOLDEN
                }
            }
            return;
        }
        // BIDIR_COMP_REFERENCE
        let bit = (ref0 == 4 || ref0 == 3) as i32; // GOLDEN || LAST3
        write_symbol(enc, bit, &mut cdfs[5], 2);
        if bit == 0 {
            write_symbol(enc, (ref0 == 2) as i32, &mut cdfs[6], 2); // LAST2
        } else {
            write_symbol(enc, (ref0 == 4) as i32, &mut cdfs[7], 2); // GOLDEN
        }
        let bit_bwd = (ref1 == 7) as i32; // ALTREF
        write_symbol(enc, bit_bwd, &mut cdfs[8], 2);
        if bit_bwd == 0 {
            write_symbol(enc, (ref1 == 6) as i32, &mut cdfs[9], 2); // ALTREF2
        }
    } else {
        let bit0 = (5..=7).contains(&ref0) as i32; // BWDREF..ALTREF
        write_symbol(enc, bit0, &mut cdfs[10], 2); // single_ref_p1
        if bit0 != 0 {
            let bit1 = (ref0 == 7) as i32; // ALTREF
            write_symbol(enc, bit1, &mut cdfs[11], 2); // single_ref_p2
            if bit1 == 0 {
                write_symbol(enc, (ref0 == 6) as i32, &mut cdfs[15], 2); // single_ref_p6, ALTREF2
            }
        } else {
            let bit2 = (ref0 == 3 || ref0 == 4) as i32; // LAST3 || GOLDEN
            write_symbol(enc, bit2, &mut cdfs[12], 2); // single_ref_p3
            if bit2 == 0 {
                write_symbol(enc, (ref0 != 1) as i32, &mut cdfs[13], 2); // single_ref_p4, != LAST
            } else {
                write_symbol(enc, (ref0 != 3) as i32, &mut cdfs[14], 2); // single_ref_p5, != LAST3
            }
        }
    }
}

const MAX_SEGMENTS_MI: usize = 8;

/// `av1_neg_interleave` (`av1/encoder/bitstream.c`): the segment-id coding transform that
/// recenters `x` around the prediction `ref` into `[0, max)` for entropy coding.
pub fn neg_interleave(x: i32, ref_: i32, max: i32) -> i32 {
    let diff = x - ref_;
    if ref_ == 0 {
        return x;
    }
    if ref_ >= max - 1 {
        return -x + max - 1;
    }
    if 2 * ref_ < max {
        if diff.abs() <= ref_ {
            if diff > 0 {
                (diff << 1) - 1
            } else {
                (-diff) << 1
            }
        } else {
            x
        }
    } else if diff.abs() < max - ref_ {
        if diff > 0 {
            (diff << 1) - 1
        } else {
            (-diff) << 1
        }
    } else {
        (max - x) - 1
    }
}

/// `write_segment_id` (`av1/encoder/bitstream.c`): the per-block segment id. Nothing is
/// coded when segmentation is off or the map isn't updated, or when `skip_txfm` (the id
/// is then set to the spatial prediction). Otherwise the neg-interleaved id is coded on
/// the (cdf_num-selected) `spatial_pred_seg_cdf` (`MAX_SEGMENTS`=8). The spatial
/// prediction `pred` + `cdf_num` (av1_get_spatial_seg_pred) are the caller's.
#[allow(clippy::too_many_arguments)]
pub fn write_segment_id(
    enc: &mut OdEcEnc,
    pred_cdf: &mut [u16],
    seg_enabled: bool,
    update_map: bool,
    skip_txfm: bool,
    segment_id: i32,
    pred: i32,
    last_active_segid: i32,
) {
    if !seg_enabled || !update_map || skip_txfm {
        return;
    }
    let coded_id = neg_interleave(segment_id, pred, last_active_segid + 1);
    write_symbol(enc, coded_id, pred_cdf, MAX_SEGMENTS_MI);
}

/// `write_intrabc_info` (`av1/encoder/bitstream.c`): the intra-block-copy flag on
/// `intrabc_cdf` (2 symbols), then, when used, the block delta (motion) vector via
/// `av1_encode_dv` — which is [`encode_mv`] at `MV_SUBPEL_NONE` precision on the DV nmv
/// context. `diff_*` are the DV minus its reference.
#[allow(clippy::too_many_arguments)]
pub fn write_intrabc_info(
    enc: &mut OdEcEnc,
    intrabc_cdf: &mut [u16],
    ndvc_joints: &mut [u16],
    ndvc_comp0: &mut [u16; 69],
    ndvc_comp1: &mut [u16; 69],
    use_intrabc: i32,
    diff_row: i32,
    diff_col: i32,
) {
    write_symbol(enc, use_intrabc, intrabc_cdf, 2);
    if use_intrabc != 0 {
        encode_mv(enc, ndvc_joints, ndvc_comp0, ndvc_comp1, diff_row, diff_col, MV_SUBPEL_NONE);
    }
}

/// `av1_get_skip_mode_context` (`pred_common.h`): above+left neighbours' `skip_mode`
/// flags (0 when absent), context in {0,1,2}.
pub fn skip_mode_context(above_skip_mode: i32, left_skip_mode: i32) -> i32 {
    above_skip_mode + left_skip_mode
}

/// `write_skip_mode` (`av1/encoder/bitstream.c`): the skip-mode flag on the
/// context-selected `skip_mode_cdfs` (2 symbols). Nothing is coded unless the frame
/// enables skip mode, compound refs are allowed for the block, and no segment feature
/// (SKIP / REF_FRAME / GLOBALMV) forces the mode. All those gates are the caller's.
pub fn write_skip_mode(
    enc: &mut OdEcEnc,
    skip_mode_cdf: &mut [u16],
    frame_skip_mode_flag: bool,
    seg_skip_active: bool,
    is_comp_ref_allowed: bool,
    seg_ref_or_gmv_active: bool,
    skip_mode: i32,
) {
    if !frame_skip_mode_flag || seg_skip_active || !is_comp_ref_allowed || seg_ref_or_gmv_active {
        return;
    }
    write_symbol(enc, skip_mode, skip_mode_cdf, 2);
}

// --- variable-transform-size (var-tx) neighbour context (av1_common_int.h) ---

// TX_SIZE indices (aom_dsp/txfm_common.h): square sizes 0..4, then rects 5..18.
const TX_4X4: usize = 0;
const TX_8X8: usize = 1;
const TX_SIZES: usize = 5; // number of square tx sizes
/// `TXFM_PARTITION_CONTEXTS` = `(TX_SIZES - TX_8X8) * 6 - 3` = 21 (`enums.h`).
const TXFM_PARTITION_CONTEXTS: usize = (TX_SIZES - TX_8X8) * 6 - 3;

/// `tx_size_wide[TX_SIZES_ALL]` (`common_data.h`): transform width in pixels.
const TX_SIZE_WIDE: [i32; 19] =
    [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
/// `tx_size_high[TX_SIZES_ALL]` (`common_data.h`): transform height in pixels.
const TX_SIZE_HIGH: [i32; 19] =
    [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
/// `block_size_wide[BLOCK_SIZES_ALL]` (`common_data.h`): block width in pixels.
const BLOCK_SIZE_WIDE: [i32; 22] =
    [4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64];
/// `block_size_high[BLOCK_SIZES_ALL]` (`common_data.h`): block height in pixels.
const BLOCK_SIZE_HIGH: [i32; 22] =
    [4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16];
/// `mi_size_wide[BLOCK_SIZES_ALL]` (`common_data.h`): block width in 4x4 units.
const MI_SIZE_WIDE: [i32; 22] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
/// `mi_size_high[BLOCK_SIZES_ALL]` (`common_data.h`): block height in 4x4 units.
const MI_SIZE_HIGH: [i32; 22] =
    [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];
/// `txsize_sqr_up_map[TX_SIZES_ALL]` (`common_data.h`): map each tx size to the
/// smallest square tx size that contains it.
const TXSIZE_SQR_UP_MAP: [usize; 19] =
    [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];
/// `txsize_to_bsize[TX_SIZES_ALL]` (`common_data.h`): the block size equal to a
/// transform block's dimensions.
const TXSIZE_TO_BSIZE: [usize; 19] =
    [0, 3, 6, 9, 12, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21];

/// `get_sqr_tx_size` (`av1_common_int.h`): largest square tx size fitting `tx_dim`.
fn get_sqr_tx_size(tx_dim: i32) -> usize {
    match tx_dim {
        128 | 64 => 4, // TX_64X64
        32 => 3,       // TX_32X32
        16 => 2,       // TX_16X16
        8 => 1,        // TX_8X8
        _ => 0,        // TX_4X4
    }
}

/// `txfm_partition_context` (`av1_common_int.h`): the CDF context (0..20) for the
/// var-tx split flag, from the above/left neighbour txfm-context values (the single
/// element each pointer addresses) plus this block's `bsize` and current `tx_size`.
pub fn txfm_partition_context(above_ctx: u8, left_ctx: u8, bsize: usize, tx_size: usize) -> usize {
    let txw = TX_SIZE_WIDE[tx_size];
    let txh = TX_SIZE_HIGH[tx_size];
    let above = (i32::from(above_ctx) < txw) as usize;
    let left = (i32::from(left_ctx) < txh) as usize;

    // dummy return, not used by others. C writes `tx_size <= TX_4X4`; TX_4X4 is the
    // minimum TX_SIZE so on an unsigned index that is exactly `== TX_4X4`.
    if tx_size == TX_4X4 {
        return 0;
    }

    let max_tx_size = get_sqr_tx_size(BLOCK_SIZE_WIDE[bsize].max(BLOCK_SIZE_HIGH[bsize]));
    let mut category = TXFM_PARTITION_CONTEXTS;
    if max_tx_size >= TX_8X8 {
        category = usize::from(TXSIZE_SQR_UP_MAP[tx_size] != max_tx_size && max_tx_size > TX_8X8)
            + (TX_SIZES - 1 - max_tx_size) * 2;
    }
    debug_assert_ne!(category, TXFM_PARTITION_CONTEXTS);
    category * 3 + above + left
}

/// `txfm_partition_update` (`av1_common_int.h`): after coding a var-tx split flag,
/// stamp the neighbour txfm-context arrays — `above[0..bw] = txw`, `left[0..bh] = txh`,
/// where `bw`/`bh` come from `txb_size`'s block dimensions in mi units.
pub fn txfm_partition_update(above_ctx: &mut [u8], left_ctx: &mut [u8], tx_size: usize, txb_size: usize) {
    let bsize = TXSIZE_TO_BSIZE[txb_size];
    let bh = MI_SIZE_HIGH[bsize] as usize;
    let bw = MI_SIZE_WIDE[bsize] as usize;
    let txw = TX_SIZE_WIDE[tx_size] as u8;
    let txh = TX_SIZE_HIGH[tx_size] as u8;
    for l in left_ctx.iter_mut().take(bh) {
        *l = txh;
    }
    for a in above_ctx.iter_mut().take(bw) {
        *a = txw;
    }
}

/// `tx_size_wide_unit[TX_SIZES_ALL]` (`common_data.h`): tx width in 4x4 units.
const TX_SIZE_WIDE_UNIT: [i32; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
/// `tx_size_high_unit[TX_SIZES_ALL]` (`common_data.h`): tx height in 4x4 units.
const TX_SIZE_HIGH_UNIT: [i32; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];
/// `sub_tx_size_map[TX_SIZES_ALL]` (`common_data.h`): one var-tx split step.
const SUB_TX_SIZE_MAP: [usize; 19] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
/// `MAX_VARTX_DEPTH` (`enums.h`).
const MAX_VARTX_DEPTH: i32 = 2;

/// `max_block_wide` (`av1_common_int.h`) for luma (plane 0, subsampling_x = 0): the
/// block width in 4x4 tx units, clipped to the frame's right edge.
fn max_block_wide(bsize: usize, mb_to_right_edge: i32) -> i32 {
    let mut max_blocks_wide = BLOCK_SIZE_WIDE[bsize];
    if mb_to_right_edge < 0 {
        max_blocks_wide += mb_to_right_edge >> 3; // 3 + subsampling_x(=0)
    }
    max_blocks_wide >> 2 // MI_SIZE_LOG2
}

/// `max_block_high` (`av1_common_int.h`) for luma (plane 0, subsampling_y = 0).
fn max_block_high(bsize: usize, mb_to_bottom_edge: i32) -> i32 {
    let mut max_blocks_high = BLOCK_SIZE_HIGH[bsize];
    if mb_to_bottom_edge < 0 {
        max_blocks_high += mb_to_bottom_edge >> 3; // 3 + subsampling_y(=0)
    }
    max_blocks_high >> 2 // MI_SIZE_LOG2
}

/// `av1_get_txb_size_index` (`blockd.h`): index into `inter_tx_size[]` for the txb at
/// (blk_row, blk_col) within a block of size `bsize`.
fn get_txb_size_index(bsize: usize, blk_row: i32, blk_col: i32) -> usize {
    const TW_W_LOG2: [i32; 22] = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0, 1, 1, 2, 2, 3];
    const TW_H_LOG2: [i32; 22] = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 0, 2, 1, 3, 2];
    const STRIDE_LOG2: [i32; 22] = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1];
    let index = ((blk_row >> TW_H_LOG2[bsize]) << STRIDE_LOG2[bsize]) + (blk_col >> TW_W_LOG2[bsize]);
    index as usize
}

/// `write_tx_size_vartx` (`av1/encoder/bitstream.c`): the recursive variable-transform
/// -size split coder for an inter block. Starting from the block's top `tx_size`, it
/// walks the quadtree — at each node either coding a "no further split" flag (0) into
/// `txfm_partition_cdf[ctx]` and stamping the neighbour context, or coding a "split"
/// flag (1) and recursing into the four `sub_tx_size_map` children (down to
/// `MAX_VARTX_DEPTH`). `inter_tx_size[]` (indexed by `av1_get_txb_size_index`) holds
/// the chosen per-txb sizes that decide each node. `above_ctx`/`left_ctx` are the
/// per-superblock neighbour txfm-context arrays (mutated in place).
#[allow(clippy::too_many_arguments)]
pub fn write_tx_size_vartx(
    enc: &mut OdEcEnc,
    txfm_partition_cdf: &mut [[u16; 3]],
    bsize: usize,
    inter_tx_size: &[usize; 16],
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    above_ctx: &mut [u8],
    left_ctx: &mut [u8],
    tx_size: usize,
    depth: i32,
    blk_row: i32,
    blk_col: i32,
) {
    let max_blocks_high = max_block_high(bsize, mb_to_bottom_edge);
    let max_blocks_wide = max_block_wide(bsize, mb_to_right_edge);
    if blk_row >= max_blocks_high || blk_col >= max_blocks_wide {
        return;
    }

    let (bc, br) = (blk_col as usize, blk_row as usize);
    if depth == MAX_VARTX_DEPTH {
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
        return;
    }

    let ctx = txfm_partition_context(above_ctx[bc], left_ctx[br], bsize, tx_size);
    let txb_size_index = get_txb_size_index(bsize, blk_row, blk_col);
    let write_txfm_partition = tx_size == inter_tx_size[txb_size_index];
    if write_txfm_partition {
        write_symbol(enc, 0, &mut txfm_partition_cdf[ctx], 2);
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
    } else {
        let sub_txs = SUB_TX_SIZE_MAP[tx_size];
        let bsw = TX_SIZE_WIDE_UNIT[sub_txs];
        let bsh = TX_SIZE_HIGH_UNIT[sub_txs];
        write_symbol(enc, 1, &mut txfm_partition_cdf[ctx], 2);
        if sub_txs == TX_4X4 {
            txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], sub_txs, tx_size);
            return;
        }
        let mut row = 0;
        while row < TX_SIZE_HIGH_UNIT[tx_size] {
            let offsetr = blk_row + row;
            let mut col = 0;
            while col < TX_SIZE_WIDE_UNIT[tx_size] {
                let offsetc = blk_col + col;
                write_tx_size_vartx(
                    enc, txfm_partition_cdf, bsize, inter_tx_size, mb_to_right_edge,
                    mb_to_bottom_edge, above_ctx, left_ctx, sub_txs, depth + 1, offsetr, offsetc,
                );
                col += bsw;
            }
            row += bsh;
        }
    }
}

// --- palette signalling: contexts + flag/size symbols (av1/encoder/bitstream.c) ---

const PALETTE_MIN_SIZE: i32 = 2;
const PALETTE_SIZES: usize = 7;
/// `num_pels_log2_lookup[BLOCK_SIZES_ALL]` (`common_data.h`): log2 of a block's pixel
/// count (`log2(w*h)`).
const NUM_PELS_LOG2_LOOKUP: [i32; 22] =
    [4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10];

/// `av1_get_palette_bsize_ctx` (`pred_common.h`): the palette size-CDF context —
/// `num_pels_log2_lookup[bsize] - num_pels_log2_lookup[BLOCK_8X8]` (BLOCK_8X8 = 6).
pub fn palette_bsize_ctx(bsize: usize) -> i32 {
    NUM_PELS_LOG2_LOOKUP[bsize] - NUM_PELS_LOG2_LOOKUP[BLOCK_8X8]
}

/// `av1_get_palette_mode_ctx` (`pred_common.h`): the palette-Y mode-CDF context — the
/// count of present neighbours (above, left) that themselves use a Y palette, in
/// `{0, 1, 2}`.
pub fn palette_mode_ctx(has_above: bool, above_palette_size0: i32, has_left: bool, left_palette_size0: i32) -> i32 {
    let mut ctx = 0;
    if has_above {
        ctx += i32::from(above_palette_size0 > 0);
    }
    if has_left {
        ctx += i32::from(left_palette_size0 > 0);
    }
    ctx
}

/// `write_palette_mode_info` (`av1/encoder/bitstream.c`) — the flag/size portion (the
/// colour payload is coded separately). For a DC_PRED luma block, code the Y-palette
/// on/off flag (`n_y > 0`) on the (bsize_ctx, mode_ctx)-selected `palette_y_mode_cdf`,
/// and when on the size symbol (`n_y - PALETTE_MIN_SIZE`) on `palette_y_size_cdf`
/// (`PALETTE_SIZES` symbols). The chroma-reference UV_DC_PRED path is the analogue on
/// the UV CDFs. All CDFs adapt in place.
#[allow(clippy::too_many_arguments)]
pub fn write_palette_mode_info_flags(
    enc: &mut OdEcEnc,
    mode_is_dc_pred: bool,
    n_y: i32,
    palette_y_mode_cdf: &mut [u16],
    palette_y_size_cdf: &mut [u16],
    uv_dc_pred: bool,
    n_uv: i32,
    palette_uv_mode_cdf: &mut [u16],
    palette_uv_size_cdf: &mut [u16],
) {
    if mode_is_dc_pred {
        write_symbol(enc, i32::from(n_y > 0), palette_y_mode_cdf, 2);
        if n_y > 0 {
            write_symbol(enc, n_y - PALETTE_MIN_SIZE, palette_y_size_cdf, PALETTE_SIZES);
        }
    }
    if uv_dc_pred {
        write_symbol(enc, i32::from(n_uv > 0), palette_uv_mode_cdf, 2);
        if n_uv > 0 {
            write_symbol(enc, n_uv - PALETTE_MIN_SIZE, palette_uv_size_cdf, PALETTE_SIZES);
        }
    }
}

const PALETTE_MAX_SIZE: usize = 8;

/// `aom_ceil_log2` (`aom_ports/bitops.h`): `ceil(log2(n))` — `0` for `n < 2`, else
/// `get_msb(n - 1) + 1`.
fn aom_ceil_log2(n: i32) -> i32 {
    if n < 2 {
        0
    } else {
        get_msb((n - 1) as u32) as i32 + 1
    }
}

/// `delta_encode_palette_colors` (`av1/encoder/bitstream.c`): code an ascending list
/// of `num` palette colours not found in the neighbour cache. The first colour is a
/// raw `bit_depth`-bit literal; the rest are coded as deltas (`>= min_val`) with a
/// bit-width that starts at `max(ceil_log2(max_delta + 1 - min_val), bit_depth - 3)`
/// (the excess over `bit_depth - 3` sent in 2 bits) and shrinks as the remaining
/// `range` narrows. `min_val` is 1 for luma, 0 for chroma-U.
pub fn delta_encode_palette_colors(enc: &mut OdEcEnc, colors: &[i32], bit_depth: i32, min_val: i32) {
    let num = colors.len();
    if num == 0 {
        return;
    }
    write_literal(enc, colors[0], bit_depth as u32);
    if num == 1 {
        return;
    }
    let mut max_delta = 0;
    let mut deltas = [0i32; PALETTE_MAX_SIZE];
    for i in 1..num {
        let delta = colors[i] - colors[i - 1];
        deltas[i - 1] = delta;
        if delta > max_delta {
            max_delta = delta;
        }
    }
    let min_bits = bit_depth - 3;
    let mut bits = aom_ceil_log2(max_delta + 1 - min_val).max(min_bits);
    let mut range = (1 << bit_depth) - colors[0] - min_val;
    write_literal(enc, bits - min_bits, 2);
    for &delta in deltas.iter().take(num - 1) {
        write_literal(enc, delta - min_val, bits as u32);
        range -= delta;
        bits = bits.min(aom_ceil_log2(range));
    }
}

/// `av1_get_palette_delta_bits_v` (`av1/encoder/palette.c`): the per-delta bit-width
/// for the V palette plane and `zero_count` / `min_bits`. Unlike Y/U, the V colours are
/// not sorted, so the "delta" wraps modulo `2^bit_depth` (`d = min(|Δ|, 2^bd - |Δ|)`).
pub fn palette_delta_bits_v(colors_v: &[u16], bit_depth: i32) -> (i32, i32, i32) {
    let n = colors_v.len();
    let max_val = 1 << bit_depth;
    let mut max_d = 0;
    let min_bits = bit_depth - 4;
    let mut zero_count = 0;
    for i in 1..n {
        let delta = colors_v[i] as i32 - colors_v[i - 1] as i32;
        let v = delta.abs();
        let d = v.min(max_val - v);
        if d > max_d {
            max_d = d;
        }
        if d == 0 {
            zero_count += 1;
        }
    }
    (aom_ceil_log2(max_d + 1).max(min_bits), zero_count, min_bits)
}

/// `write_palette_colors_uv` V-plane portion (`av1/encoder/bitstream.c`): the V palette
/// colours, coded either as wrap-around deltas (a 1 flag, the `bits_v - min_bits_v`
/// excess in 2 bits, the first colour raw, then each `|Δ|` — or `2^bd - |Δ|`, whichever
/// is smaller — in `bits_v` bits plus a sign bit) or as raw `bit_depth`-bit literals
/// (a 0 flag), whichever the rate estimate prefers. No colour cache (V is unsorted).
pub fn write_palette_colors_v(enc: &mut OdEcEnc, colors_v: &[u16], bit_depth: i32) {
    let n = colors_v.len() as i32;
    let max_val = 1 << bit_depth;
    let (bits_v, zero_count, min_bits_v) = palette_delta_bits_v(colors_v, bit_depth);
    let rate_using_delta = 2 + bit_depth + (bits_v + 1) * (n - 1) - zero_count;
    let rate_using_raw = bit_depth * n;
    if rate_using_delta < rate_using_raw {
        write_bit(enc, 1);
        write_literal(enc, bits_v - min_bits_v, 2);
        write_literal(enc, colors_v[0] as i32, bit_depth as u32);
        for i in 1..n as usize {
            if colors_v[i] == colors_v[i - 1] {
                write_literal(enc, 0, bits_v as u32);
                continue;
            }
            let delta = (colors_v[i] as i32 - colors_v[i - 1] as i32).abs();
            let sign_bit = i32::from(colors_v[i] < colors_v[i - 1]);
            if delta <= max_val - delta {
                write_literal(enc, delta, bits_v as u32);
                write_bit(enc, sign_bit);
            } else {
                write_literal(enc, max_val - delta, bits_v as u32);
                write_bit(enc, (sign_bit == 0) as i32);
            }
        }
    } else {
        write_bit(enc, 0);
        for &c in colors_v.iter() {
            write_literal(enc, c as i32, bit_depth as u32);
        }
    }
}

const MIN_SB_SIZE_LOG2: i32 = 6;

/// `palette_add_to_cache` (`pred_common.c`): append `val` to `cache`, skipping it when
/// it equals the current last entry (the merged lists are sorted, so this dedups).
fn palette_add_to_cache(cache: &mut [u16], n: &mut usize, val: u16) {
    if *n > 0 && val == cache[*n - 1] {
        return;
    }
    cache[*n] = val;
    *n += 1;
}

/// `av1_get_palette_cache` (`pred_common.c`): build the neighbour colour cache for
/// `plane` by merging the above and left neighbours' (individually sorted) palettes
/// into one sorted, deduplicated list. The above neighbour is dropped when the block
/// sits on a superblock-row boundary (`row % (1<<MIN_SB_SIZE_LOG2) == 0`, `row =
/// -mb_to_top_edge >> 3`). Each neighbour is passed as its full `3*PALETTE_MAX_SIZE`
/// `palette_colors` layout with the plane's colour count. Returns the cache length.
#[allow(clippy::too_many_arguments)]
pub fn get_palette_cache(
    cache: &mut [u16],
    plane: usize,
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    mut above_n: i32,
    has_left: bool,
    left_colors: &[u16],
    mut left_n: i32,
) -> usize {
    let row = -mb_to_top_edge >> 3;
    // Do not refer to above SB row when on SB boundary.
    let use_above = has_above && (row % (1 << MIN_SB_SIZE_LOG2)) != 0;
    if !use_above {
        above_n = 0;
    }
    if !has_left {
        left_n = 0;
    }
    if above_n == 0 && left_n == 0 {
        return 0;
    }
    let mut above_idx = plane * PALETTE_MAX_SIZE;
    let mut left_idx = plane * PALETTE_MAX_SIZE;
    let mut n = 0;
    // Merge the sorted lists of base colors from above and left.
    while above_n > 0 && left_n > 0 {
        let v_above = above_colors[above_idx];
        let v_left = left_colors[left_idx];
        if v_left < v_above {
            palette_add_to_cache(cache, &mut n, v_left);
            left_idx += 1;
            left_n -= 1;
        } else {
            palette_add_to_cache(cache, &mut n, v_above);
            above_idx += 1;
            above_n -= 1;
            if v_left == v_above {
                left_idx += 1;
                left_n -= 1;
            }
        }
    }
    while above_n > 0 {
        palette_add_to_cache(cache, &mut n, above_colors[above_idx]);
        above_idx += 1;
        above_n -= 1;
    }
    while left_n > 0 {
        palette_add_to_cache(cache, &mut n, left_colors[left_idx]);
        left_idx += 1;
        left_n -= 1;
    }
    n
}

/// `av1_index_color_cache` (`av1/encoder/palette.c`): mark which cache entries appear in
/// the block's `colors` (`cache_color_found`), and collect the colours *not* in the
/// cache (`out_cache_colors`, preserving order). Returns `n_out_cache`. With an empty
/// cache every colour is out-of-cache.
pub fn index_color_cache(cache: &[u16], colors: &[u16]) -> (Vec<u8>, Vec<i32>, usize) {
    let n_cache = cache.len();
    let n_colors = colors.len();
    if n_cache == 0 {
        let out: Vec<i32> = colors.iter().map(|&c| c as i32).collect();
        return (Vec::new(), out, n_colors);
    }
    let mut cache_color_found = vec![0u8; n_cache];
    let mut in_cache_flags = [0u8; PALETTE_MAX_SIZE];
    let mut n_in_cache = 0;
    for i in 0..n_cache {
        if n_in_cache >= n_colors {
            break;
        }
        for j in 0..n_colors {
            if colors[j] == cache[i] {
                in_cache_flags[j] = 1;
                cache_color_found[i] = 1;
                n_in_cache += 1;
                break;
            }
        }
    }
    let mut out = Vec::with_capacity(n_colors - n_in_cache);
    for i in 0..n_colors {
        if in_cache_flags[i] == 0 {
            out.push(colors[i] as i32);
        }
    }
    let n_out = out.len();
    (cache_color_found, out, n_out)
}

/// `write_palette_colors_y` / the U half of `write_palette_colors_uv`
/// (`av1/encoder/bitstream.c`): for a palette plane, signal which of the neighbour
/// cache's colours the block reuses (one bit each, until all `n` are placed), then
/// delta-code the remaining out-of-cache colours. `min_val` is 1 for luma, 0 for U.
#[allow(clippy::too_many_arguments)]
fn write_palette_colors_plane(
    enc: &mut OdEcEnc,
    colors: &[u16],
    n: usize,
    plane: usize,
    bit_depth: i32,
    min_val: i32,
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_n_plane: i32,
    has_left: bool,
    left_colors: &[u16],
    left_n_plane: i32,
) {
    let mut cache = [0u16; 16];
    let n_cache = get_palette_cache(
        &mut cache, plane, mb_to_top_edge, has_above, above_colors, above_n_plane, has_left,
        left_colors, left_n_plane,
    );
    let (found, out_colors, _n_out) = index_color_cache(&cache[..n_cache], &colors[..n]);
    let mut n_in_cache = 0;
    for &f in found.iter().take(n_cache) {
        if n_in_cache >= n {
            break;
        }
        write_bit(enc, i32::from(f));
        n_in_cache += f as usize;
    }
    delta_encode_palette_colors(enc, &out_colors, bit_depth, min_val);
}

/// `write_palette_mode_info` (`av1/encoder/bitstream.c`), complete: the Y-palette
/// on/off flag + size + colours for a DC_PRED block, then the UV_DC_PRED analogue
/// (U colours through the cache, V colours through the unsorted delta/raw coder). The
/// four CDFs are the caller's (bsize/mode)-selected slices, adapted in place. Neighbour
/// palettes (full `3*PALETTE_MAX_SIZE` layout) feed the colour cache.
#[allow(clippy::too_many_arguments)]
pub fn write_palette_mode_info(
    enc: &mut OdEcEnc,
    mode_is_dc_pred: bool,
    uv_dc_pred: bool,
    bit_depth: i32,
    palette_size: [i32; 2],
    palette_colors: &[u16],
    y_mode_cdf: &mut [u16],
    y_size_cdf: &mut [u16],
    uv_mode_cdf: &mut [u16],
    uv_size_cdf: &mut [u16],
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_size: [i32; 2],
    has_left: bool,
    left_colors: &[u16],
    left_size: [i32; 2],
) {
    if mode_is_dc_pred {
        let n = palette_size[0];
        write_symbol(enc, i32::from(n > 0), y_mode_cdf, 2);
        if n > 0 {
            write_symbol(enc, n - PALETTE_MIN_SIZE, y_size_cdf, PALETTE_SIZES);
            write_palette_colors_plane(
                enc, palette_colors, n as usize, 0, bit_depth, 1, mb_to_top_edge, has_above,
                above_colors, above_size[0], has_left, left_colors, left_size[0],
            );
        }
    }
    if uv_dc_pred {
        let n = palette_size[1];
        write_symbol(enc, i32::from(n > 0), uv_mode_cdf, 2);
        if n > 0 {
            write_symbol(enc, n - PALETTE_MIN_SIZE, uv_size_cdf, PALETTE_SIZES);
            let colors_u = &palette_colors[PALETTE_MAX_SIZE..];
            let vbase = 2 * PALETTE_MAX_SIZE;
            let colors_v = &palette_colors[vbase..vbase + n as usize];
            write_palette_colors_plane(
                enc, colors_u, n as usize, 1, bit_depth, 0, mb_to_top_edge, has_above,
                above_colors, above_size[1], has_left, left_colors, left_size[1],
            );
            write_palette_colors_v(enc, colors_v, bit_depth);
        }
    }
}

const INTERINTRA_MODES: usize = 4;
const MAX_WEDGE_TYPES: usize = 16;

/// The interintra sub-symbols of `write_mbmi_b` (`av1/encoder/bitstream.c`): when
/// interintra compound is allowed for the block, code the interintra on/off flag; if
/// on, the `interintra_mode` (`INTERINTRA_MODES` symbols on the size-group CDF) and,
/// when wedges are usable for `bsize`, the `use_wedge_interintra` flag and (if set) the
/// `wedge_idx` (`MAX_WEDGE_TYPES` symbols). The outer allow-gate (reference mode,
/// sequence flag, `is_interintra_allowed`) and `wedge_used` (`av1_is_wedge_used`) are
/// the caller's; the four CDFs are the caller's size-group/`bsize`-selected slices.
#[allow(clippy::too_many_arguments)]
pub fn write_interintra_info(
    enc: &mut OdEcEnc,
    allowed: bool,
    interintra: i32,
    interintra_cdf: &mut [u16],
    interintra_mode: i32,
    interintra_mode_cdf: &mut [u16],
    wedge_used: bool,
    use_wedge_interintra: i32,
    wedge_interintra_cdf: &mut [u16],
    interintra_wedge_index: i32,
    wedge_idx_cdf: &mut [u16],
) {
    if !allowed {
        return;
    }
    write_symbol(enc, interintra, interintra_cdf, 2);
    if interintra != 0 {
        write_symbol(enc, interintra_mode, interintra_mode_cdf, INTERINTRA_MODES);
        if wedge_used {
            write_symbol(enc, use_wedge_interintra, wedge_interintra_cdf, 2);
            if use_wedge_interintra != 0 {
                write_symbol(enc, interintra_wedge_index, wedge_idx_cdf, MAX_WEDGE_TYPES);
            }
        }
    }
}

const ALTREF_FRAME: i32 = 7;
const COMPOUND_WEDGE: i32 = 2;
const MASKED_COMPOUND_TYPES: usize = 2;
const MAX_DIFFWTD_MASK_BITS: u32 = 1;

/// One neighbour's contribution to `get_comp_group_idx_context`: its `comp_group_idx`
/// when it is a compound block, else 3 when it is a single-ref block pointing at ALTREF,
/// else 0.
fn comp_group_idx_nbr(ref_frame0: i32, ref_frame1: i32, comp_group_idx: i32) -> i32 {
    if ref_frame1 > 0 {
        // has_second_ref (INTRA_FRAME == 0)
        comp_group_idx
    } else if ref_frame0 == ALTREF_FRAME {
        3
    } else {
        0
    }
}

/// `get_comp_group_idx_context` (`pred_common.h`): the compound-group-index CDF context,
/// the sum of the above and left neighbours' contributions capped at 5.
#[allow(clippy::too_many_arguments)]
pub fn get_comp_group_idx_context(
    has_above: bool, a_rf0: i32, a_rf1: i32, a_cgi: i32,
    has_left: bool, l_rf0: i32, l_rf1: i32, l_cgi: i32,
) -> i32 {
    let above = if has_above { comp_group_idx_nbr(a_rf0, a_rf1, a_cgi) } else { 0 };
    let left = if has_left { comp_group_idx_nbr(l_rf0, l_rf1, l_cgi) } else { 0 };
    (above + left).min(5)
}

/// The compound-type coding of `write_mbmi_b` (`av1/encoder/bitstream.c`) for a
/// two-reference block. When masked compound is available, code `comp_group_idx`. Group
/// 0 (average / distance-weighted): code `compound_idx` when distance-weighted compound
/// is enabled. Group 1 (masked): code the `compound_type` (wedge vs diffwtd) when wedge
/// is usable, then either the wedge index + sign, or the diffwtd `mask_type` literal.
/// The outer `has_second_ref` gate, the two CDF contexts, and the wedge/dist-wtd/masked
/// gates are the caller's; CDFs are pre-selected.
#[allow(clippy::too_many_arguments)]
pub fn write_compound_type_info(
    enc: &mut OdEcEnc,
    masked_compound_used: bool,
    comp_group_idx: i32,
    comp_group_idx_cdf: &mut [u16],
    enable_dist_wtd_comp: bool,
    compound_idx: i32,
    compound_index_cdf: &mut [u16],
    wedge_used: bool,
    comp_type: i32,
    compound_type_cdf: &mut [u16],
    wedge_index: i32,
    wedge_idx_cdf: &mut [u16],
    wedge_sign: i32,
    mask_type: i32,
) {
    if masked_compound_used {
        write_symbol(enc, comp_group_idx, comp_group_idx_cdf, 2);
    }
    if comp_group_idx == 0 {
        if enable_dist_wtd_comp {
            write_symbol(enc, compound_idx, compound_index_cdf, 2);
        }
    } else {
        if wedge_used {
            write_symbol(enc, comp_type - COMPOUND_WEDGE, compound_type_cdf, MASKED_COMPOUND_TYPES);
        }
        if comp_type == COMPOUND_WEDGE {
            write_symbol(enc, wedge_index, wedge_idx_cdf, MAX_WEDGE_TYPES);
            write_bit(enc, wedge_sign);
        } else {
            write_literal(enc, mask_type, MAX_DIFFWTD_MASK_BITS);
        }
    }
}

/// `get_relative_dist` (`mvref_common.h`): the signed order-hint distance `a - b`
/// wrapped into `[-2^(bits-1), 2^(bits-1))`, where `bits = order_hint_bits_minus_1 + 1`.
/// Zero when order hints are disabled.
pub fn get_relative_dist(enable_order_hint: bool, order_hint_bits_minus_1: i32, a: i32, b: i32) -> i32 {
    if !enable_order_hint {
        return 0;
    }
    let bits = order_hint_bits_minus_1 + 1;
    let diff = a - b;
    let m = 1 << (bits - 1);
    (diff & (m - 1)) - (diff & m)
}

/// `get_comp_index_context` (`pred_common.h`): the distance-weighted-compound index CDF
/// context. `offset = (fwd == bck)` compares the forward/backward order-hint distances
/// (`fwd_order_hint`/`bck_order_hint` are the ref buffers' order hints, 0 when absent);
/// each present neighbour adds its `compound_idx` (compound block) or 1 (single-ref
/// ALTREF). Result is `above + left + 3 * offset`.
#[allow(clippy::too_many_arguments)]
pub fn get_comp_index_context(
    enable_order_hint: bool,
    order_hint_bits_minus_1: i32,
    cur_order_hint: i32,
    fwd_order_hint: i32,
    bck_order_hint: i32,
    has_above: bool,
    a_has_second_ref: bool,
    a_compound_idx: i32,
    a_ref_frame0: i32,
    has_left: bool,
    l_has_second_ref: bool,
    l_compound_idx: i32,
    l_ref_frame0: i32,
) -> i32 {
    let fwd = get_relative_dist(enable_order_hint, order_hint_bits_minus_1, fwd_order_hint, cur_order_hint).abs();
    let bck = get_relative_dist(enable_order_hint, order_hint_bits_minus_1, cur_order_hint, bck_order_hint).abs();
    let offset = i32::from(fwd == bck);
    let mut above_ctx = 0;
    let mut left_ctx = 0;
    if has_above {
        if a_has_second_ref {
            above_ctx = a_compound_idx;
        } else if a_ref_frame0 == ALTREF_FRAME {
            above_ctx = 1;
        }
    }
    if has_left {
        if l_has_second_ref {
            left_ctx = l_compound_idx;
        } else if l_ref_frame0 == ALTREF_FRAME {
            left_ctx = 1;
        }
    }
    above_ctx + left_ctx + 3 * offset
}

// --- intra-prediction-mode driver gates (av1/common/*.h) ---

const V_PRED: i32 = 1;
const D67_PRED: i32 = 8;
/// `uv2y` (`get_uv_mode`, blockd.h): UV prediction mode -> the Y mode it maps to
/// (UV_CFL_PRED -> DC_PRED).
const UV2Y: [i32; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0];
/// `av1_ss_size_lookup[BLOCK_SIZES_ALL][2][2]` (`common_data.c`), flattened as
/// `[bsize][ssx*2 + ssy]`; `BLOCK_INVALID` (255) where subsampling is illegal.
const SS_SIZE_LOOKUP: [[u8; 4]; 22] = [
    [0, 0, 0, 0], [1, 0, 255, 0], [2, 255, 0, 0], [3, 2, 1, 0], [4, 3, 255, 1],
    [5, 255, 3, 2], [6, 5, 4, 3], [7, 6, 255, 4], [8, 255, 6, 5], [9, 8, 7, 6],
    [10, 9, 255, 7], [11, 255, 9, 8], [12, 11, 10, 9], [13, 12, 255, 10],
    [14, 255, 12, 11], [15, 14, 13, 12], [16, 1, 255, 1], [17, 255, 2, 2],
    [18, 4, 255, 16], [19, 255, 5, 17], [20, 7, 255, 18], [21, 255, 8, 19],
];

/// `av1_use_angle_delta` (`reconintra.h`): angle deltas apply at `BLOCK_8X8` and larger.
pub fn use_angle_delta(bsize: usize) -> bool {
    bsize >= BLOCK_8X8
}

/// `av1_is_directional_mode` (`reconintra.h`): the mode is one of the directional
/// predictors (`V_PRED..=D67_PRED`).
pub fn is_directional_mode(mode: i32) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

/// `get_uv_mode` (`blockd.h`): the Y prediction mode a UV mode maps to.
pub fn get_uv_mode(uv_mode: usize) -> i32 {
    UV2Y[uv_mode]
}

/// `av1_allow_palette` (`blockd.h`): palette is available for screen-content-enabled
/// frames on blocks in `[BLOCK_8X8, 64x64]`.
pub fn allow_palette(allow_screen_content_tools: bool, bsize: usize) -> bool {
    allow_screen_content_tools
        && BLOCK_SIZE_WIDE[bsize] <= 64
        && BLOCK_SIZE_HIGH[bsize] <= 64
        && bsize >= BLOCK_8X8
}

/// `get_plane_block_size` (`blockd.h`): the block size of a chroma plane with the given
/// subsampling, via `av1_ss_size_lookup` (`BLOCK_INVALID` when illegal).
pub fn get_plane_block_size(bsize: usize, ssx: usize, ssy: usize) -> usize {
    SS_SIZE_LOOKUP[bsize][ssx * 2 + ssy] as usize
}

/// `is_cfl_allowed` (`cfl.h`): whether chroma-from-luma is available. In lossless the
/// chroma plane block size must be `BLOCK_4X4`; otherwise the luma block must be
/// `<= 32x32`.
pub fn is_cfl_allowed(bsize: usize, lossless: bool, ssx: usize, ssy: usize) -> bool {
    const BLOCK_4X4: usize = 0;
    if lossless {
        get_plane_block_size(bsize, ssx, ssy) == BLOCK_4X4
    } else {
        BLOCK_SIZE_WIDE[bsize] <= 32 && BLOCK_SIZE_HIGH[bsize] <= 32
    }
}

/// `write_intra_prediction_modes` piece 1 (`av1/encoder/bitstream.c`): the intra luma
/// mode followed by its angle delta. The Y mode is a plain `INTRA_MODES` symbol on the
/// caller-selected CDF (`kf_y_cdf[above_ctx][left_ctx]` for a keyframe, else
/// `y_mode_cdf[size_group]` — identical write, different CDF). The angle delta is coded
/// only for a directional mode on a `>= BLOCK_8X8` block, on the caller-selected
/// per-mode `angle_delta_cdf[mode - V_PRED]`. First composition of the mode-info driver.
pub fn write_intra_y_and_angle_delta(
    enc: &mut OdEcEnc,
    y_cdf: &mut [u16],
    mode: i32,
    bsize: usize,
    angle_delta_y: i32,
    y_angle_cdf: &mut [u16],
) {
    write_symbol(enc, mode, y_cdf, INTRA_MODES);
    if use_angle_delta(bsize) && is_directional_mode(mode) {
        write_angle_delta(enc, y_angle_cdf, angle_delta_y);
    }
}

const UV_CFL_PRED: i32 = 13;

/// `write_intra_prediction_modes` piece 2 (`av1/encoder/bitstream.c`): the chroma
/// prediction signalling. For a chroma-reference block on a non-monochrome frame, the
/// intra UV mode (on the caller-selected `uv_mode_cdf`, `UV_INTRA_MODES` symbols when
/// CFL is allowed, one fewer otherwise), then — for `UV_CFL_PRED` — the CFL alphas, then
/// the UV angle delta when the mapped intra mode (`get_uv_mode`) is directional on a
/// `>= BLOCK_8X8` block. Second composition of the mode-info driver.
#[allow(clippy::too_many_arguments)]
pub fn write_intra_uv_and_angle_delta(
    enc: &mut OdEcEnc,
    monochrome: bool,
    is_chroma_ref: bool,
    uv_mode: i32,
    cfl_allowed: bool,
    bsize: usize,
    cfl_alpha_idx: i32,
    cfl_joint_sign: i32,
    angle_delta_uv: i32,
    uv_mode_cdf: &mut [u16],
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
    uv_angle_cdf: &mut [u16],
) {
    if !monochrome && is_chroma_ref {
        write_intra_uv_mode(enc, uv_mode_cdf, uv_mode, cfl_allowed);
        if uv_mode == UV_CFL_PRED {
            write_cfl_alphas(enc, cfl_sign_cdf, cfl_alpha_cdf, cfl_alpha_idx, cfl_joint_sign);
        }
        let intra_mode = get_uv_mode(uv_mode as usize);
        if use_angle_delta(bsize) && is_directional_mode(intra_mode) {
            write_angle_delta(enc, uv_angle_cdf, angle_delta_uv);
        }
    }
}

const DC_PRED: i32 = 0;
const UV_DC_PRED: i32 = 0;

/// `write_intra_prediction_modes` (`av1/encoder/bitstream.c`), complete: the entire
/// intra mode-info fragment for one block over a single range coder — Y mode + Y angle
/// delta, then the chroma mode/cfl/angle, then (when allowed) the palette mode info, then
/// the filter-intra mode info. This is the first full per-block mode-info driver
/// composition; all four constituents are already bit-exact, this fixes their ordering
/// and per-block gating. CDFs are the caller's context-selected slices; the palette /
/// filter-intra allow gates and neighbour palettes are the caller's.
#[allow(clippy::too_many_arguments)]
pub fn write_intra_prediction_modes(
    enc: &mut OdEcEnc,
    // Y
    mode: i32,
    bsize: usize,
    y_cdf: &mut [u16],
    angle_delta_y: i32,
    y_angle_cdf: &mut [u16],
    // UV
    monochrome: bool,
    is_chroma_ref: bool,
    uv_mode: i32,
    cfl_allowed: bool,
    cfl_alpha_idx: i32,
    cfl_joint_sign: i32,
    angle_delta_uv: i32,
    uv_mode_cdf: &mut [u16],
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
    uv_angle_cdf: &mut [u16],
    // Palette
    allow_palette: bool,
    bit_depth: i32,
    palette_size: [i32; 2],
    palette_colors: &[u16],
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_size: [i32; 2],
    has_left: bool,
    left_colors: &[u16],
    left_size: [i32; 2],
    pal_y_mode_cdf: &mut [u16],
    pal_y_size_cdf: &mut [u16],
    pal_uv_mode_cdf: &mut [u16],
    pal_uv_size_cdf: &mut [u16],
    // Filter intra
    filter_allowed: bool,
    use_filter_intra: i32,
    filter_intra_mode: i32,
    fi_use_cdf: &mut [u16],
    fi_mode_cdf: &mut [u16],
) {
    write_intra_y_and_angle_delta(enc, y_cdf, mode, bsize, angle_delta_y, y_angle_cdf);
    write_intra_uv_and_angle_delta(
        enc, monochrome, is_chroma_ref, uv_mode, cfl_allowed, bsize, cfl_alpha_idx,
        cfl_joint_sign, angle_delta_uv, uv_mode_cdf, cfl_sign_cdf, cfl_alpha_cdf, uv_angle_cdf,
    );
    if allow_palette {
        let mode_is_dc_pred = mode == DC_PRED;
        let uv_dc_pred = !monochrome && uv_mode == UV_DC_PRED && is_chroma_ref;
        write_palette_mode_info(
            enc, mode_is_dc_pred, uv_dc_pred, bit_depth, palette_size, palette_colors,
            pal_y_mode_cdf, pal_y_size_cdf, pal_uv_mode_cdf, pal_uv_size_cdf, mb_to_top_edge,
            has_above, above_colors, above_size, has_left, left_colors, left_size,
        );
    }
    write_filter_intra_mode_info(enc, fi_use_cdf, fi_mode_cdf, filter_allowed, use_filter_intra, filter_intra_mode);
}

const FRAME_LF_COUNT: usize = 4;

/// `write_delta_q_params` (`av1/encoder/bitstream.c`): the per-superblock delta-Q (and
/// optional delta-loop-filter) driver. When delta-Q is present and this is the
/// superblock's upper-left block (and either the block is smaller than the SB or it is
/// not skipped), code the reduced delta-qindex on `delta_q_cdf` and advance
/// `current_base_qindex`. When delta-LF is present, code either per-plane multi deltas
/// (`FRAME_LF_COUNT` or `FRAME_LF_COUNT - 2` ids on `delta_lf_multi_cdf[id]`) or a single
/// delta (`delta_lf_cdf`), each advancing the corresponding `xd` running value. The
/// reduced values are `(target - running) / res`. State is updated in place.
#[allow(clippy::too_many_arguments)]
pub fn write_delta_q_params_sb(
    enc: &mut OdEcEnc,
    delta_q_present: bool,
    delta_lf_present: bool,
    delta_lf_multi: bool,
    num_planes: i32,
    bsize: usize,
    sb_size: usize,
    skip: i32,
    super_block_upper_left: bool,
    current_qindex: i32,
    current_base_qindex: &mut i32,
    delta_q_res: i32,
    mbmi_delta_lf: &[i32; FRAME_LF_COUNT],
    xd_delta_lf: &mut [i32; FRAME_LF_COUNT],
    mbmi_delta_lf_from_base: i32,
    xd_delta_lf_from_base: &mut i32,
    delta_lf_res: i32,
    delta_q_cdf: &mut [u16],
    delta_lf_multi_cdf: &mut [[u16; 5]; FRAME_LF_COUNT],
    delta_lf_cdf: &mut [u16],
) {
    if !delta_q_present {
        return;
    }
    if (bsize != sb_size || skip == 0) && super_block_upper_left {
        let reduced_delta_qindex = (current_qindex - *current_base_qindex) / delta_q_res;
        write_delta_qindex(enc, delta_q_cdf, reduced_delta_qindex);
        *current_base_qindex = current_qindex;
        if delta_lf_present {
            if delta_lf_multi {
                let frame_lf_count =
                    if num_planes > 1 { FRAME_LF_COUNT } else { FRAME_LF_COUNT - 2 };
                for lf_id in 0..frame_lf_count {
                    let reduced = (mbmi_delta_lf[lf_id] - xd_delta_lf[lf_id]) / delta_lf_res;
                    write_delta_lflevel(enc, &mut delta_lf_multi_cdf[lf_id], reduced);
                    xd_delta_lf[lf_id] = mbmi_delta_lf[lf_id];
                }
            } else {
                let reduced = (mbmi_delta_lf_from_base - *xd_delta_lf_from_base) / delta_lf_res;
                write_delta_lflevel(enc, delta_lf_cdf, reduced);
                *xd_delta_lf_from_base = mbmi_delta_lf_from_base;
            }
        }
    }
}

const MI_SIZE_LOG2: i32 = 2;

/// `write_cdef` (`av1/encoder/bitstream.c`): the per-CDEF-unit strength. Skipped for
/// coded-lossless / intrabc frames. CDEF units are 64x64 (16 mi units); at the
/// superblock's upper-left block the per-unit `cdef_transmitted` flags reset. The unit
/// index within a 128x128 SB is `col + 2*row` (else 0). On the first non-skip block of
/// an untransmitted unit, a `cdef_bits`-bit strength literal is written and the unit is
/// marked transmitted. `cdef_transmitted` is threaded across blocks (updated in place).
#[allow(clippy::too_many_arguments)]
pub fn write_cdef(
    enc: &mut OdEcEnc,
    coded_lossless: bool,
    allow_intrabc: bool,
    mi_row: i32,
    mi_col: i32,
    mib_size: i32,
    sb_size: usize,
    skip: i32,
    cdef_transmitted: &mut [bool; 4],
    cdef_bits: u32,
    cdef_strength: i32,
) {
    if coded_lossless || allow_intrabc {
        return;
    }
    let sb_mask = mib_size - 1;
    if (mi_row & sb_mask) == 0 && (mi_col & sb_mask) == 0 {
        *cdef_transmitted = [false; 4];
    }
    let cdef_size = 1 << (6 - MI_SIZE_LOG2); // 64x64 CDEF unit => 16 mi units
    let index_mask = cdef_size;
    let cdef_unit_row = i32::from((mi_row & index_mask) != 0);
    let cdef_unit_col = i32::from((mi_col & index_mask) != 0);
    let index = if sb_size == BLOCK_128X128 {
        (cdef_unit_col + 2 * cdef_unit_row) as usize
    } else {
        0
    };
    if !cdef_transmitted[index] && skip == 0 {
        write_literal(enc, cdef_strength, cdef_bits);
        cdef_transmitted[index] = true;
    }
}

/// `write_mb_modes_kf` prefix (`av1/encoder/bitstream.c:1267`): the KEY-frame per-block
/// driver up to (not including) the intrabc + intra-prediction-modes tail. Codes the
/// segment id before the skip flag when `segid_preskip` (with `skip_txfm=0`), the skip
/// flag, the segment id after skip otherwise (with `skip_txfm=skip`), then the CDEF
/// strength and the per-SB delta-Q params — the last two receiving `write_skip`'s return.
/// Returns the coded `skip`. State (segment/skip/cdef/delta CDFs, cdef_transmitted,
/// delta running values, base qindex) is updated in place.
#[allow(clippy::too_many_arguments)]
pub fn write_mb_modes_kf_prefix(
    enc: &mut OdEcEnc,
    segid_preskip: bool,
    seg_enabled: bool,
    update_map: bool,
    segment_id: i32,
    seg_pred: i32,
    last_active_segid: i32,
    seg_cdf: &mut [u16],
    seg_skip_active: bool,
    skip_txfm: i32,
    skip_cdf: &mut [u16],
    coded_lossless: bool,
    allow_intrabc: bool,
    mi_row: i32,
    mi_col: i32,
    mib_size: i32,
    sb_size: usize,
    cdef_transmitted: &mut [bool; 4],
    cdef_bits: u32,
    cdef_strength: i32,
    dq_present: bool,
    dlf_present: bool,
    dlf_multi: bool,
    num_planes: i32,
    bsize: usize,
    cur_qindex: i32,
    current_base_qindex: &mut i32,
    dq_res: i32,
    mbmi_delta_lf: &[i32; FRAME_LF_COUNT],
    xd_delta_lf: &mut [i32; FRAME_LF_COUNT],
    mbmi_delta_lf_from_base: i32,
    xd_delta_lf_from_base: &mut i32,
    dlf_res: i32,
    delta_q_cdf: &mut [u16],
    delta_lf_multi_cdf: &mut [[u16; 5]; FRAME_LF_COUNT],
    delta_lf_cdf: &mut [u16],
) -> i32 {
    if segid_preskip && update_map {
        write_segment_id(enc, seg_cdf, seg_enabled, update_map, false, segment_id, seg_pred, last_active_segid);
    }
    let skip = write_skip(enc, skip_cdf, seg_skip_active, skip_txfm);
    if !segid_preskip && update_map {
        write_segment_id(enc, seg_cdf, seg_enabled, update_map, skip != 0, segment_id, seg_pred, last_active_segid);
    }
    write_cdef(enc, coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size, skip, cdef_transmitted, cdef_bits, cdef_strength);
    let super_block_upper_left = (mi_row & (mib_size - 1)) == 0 && (mi_col & (mib_size - 1)) == 0;
    write_delta_q_params_sb(
        enc, dq_present, dlf_present, dlf_multi, num_planes, bsize, sb_size, skip,
        super_block_upper_left, cur_qindex, current_base_qindex, dq_res, mbmi_delta_lf,
        xd_delta_lf, mbmi_delta_lf_from_base, xd_delta_lf_from_base, dlf_res, delta_q_cdf,
        delta_lf_multi_cdf, delta_lf_cdf,
    );
    skip
}

/// `write_mb_modes_kf` tail (`av1/encoder/bitstream.c`): the intrabc + intra half. When
/// intrabc is allowed, code the intrabc flag (and DV via the MV coder); if the block is
/// an intrabc block, nothing further is coded (early return). Otherwise the full intra
/// prediction modes follow. Composes write_intrabc_info + write_intra_prediction_modes.
#[allow(clippy::too_many_arguments)]
pub fn write_kf_tail(
    enc: &mut OdEcEnc,
    allow_intrabc: bool,
    intrabc_cdf: &mut [u16],
    ndvc_joints: &mut [u16],
    ndvc_comp0: &mut [u16; 69],
    ndvc_comp1: &mut [u16; 69],
    use_intrabc: i32,
    diff_row: i32,
    diff_col: i32,
    // write_intra_prediction_modes state
    mode: i32,
    bsize: usize,
    y_cdf: &mut [u16],
    angle_delta_y: i32,
    y_angle_cdf: &mut [u16],
    monochrome: bool,
    is_chroma_ref: bool,
    uv_mode: i32,
    cfl_allowed: bool,
    cfl_alpha_idx: i32,
    cfl_joint_sign: i32,
    angle_delta_uv: i32,
    uv_mode_cdf: &mut [u16],
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
    uv_angle_cdf: &mut [u16],
    allow_palette: bool,
    bit_depth: i32,
    palette_size: [i32; 2],
    palette_colors: &[u16],
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_size: [i32; 2],
    has_left: bool,
    left_colors: &[u16],
    left_size: [i32; 2],
    pal_y_mode_cdf: &mut [u16],
    pal_y_size_cdf: &mut [u16],
    pal_uv_mode_cdf: &mut [u16],
    pal_uv_size_cdf: &mut [u16],
    filter_allowed: bool,
    use_filter_intra: i32,
    filter_intra_mode: i32,
    fi_use_cdf: &mut [u16],
    fi_mode_cdf: &mut [u16],
) {
    if allow_intrabc {
        write_intrabc_info(enc, intrabc_cdf, ndvc_joints, ndvc_comp0, ndvc_comp1, use_intrabc, diff_row, diff_col);
        if use_intrabc != 0 {
            return; // is_intrabc_block
        }
    }
    write_intra_prediction_modes(
        enc, mode, bsize, y_cdf, angle_delta_y, y_angle_cdf, monochrome, is_chroma_ref, uv_mode,
        cfl_allowed, cfl_alpha_idx, cfl_joint_sign, angle_delta_uv, uv_mode_cdf, cfl_sign_cdf,
        cfl_alpha_cdf, uv_angle_cdf, allow_palette, bit_depth, palette_size, palette_colors,
        mb_to_top_edge, has_above, above_colors, above_size, has_left, left_colors, left_size,
        pal_y_mode_cdf, pal_y_size_cdf, pal_uv_mode_cdf, pal_uv_size_cdf, filter_allowed,
        use_filter_intra, filter_intra_mode, fi_use_cdf, fi_mode_cdf,
    );
}

/// `av1_get_pred_context_seg_id` (`pred_common.h`): the segment-id-predicted CDF context
/// — the sum of the above and left neighbours' `seg_id_predicted` flags (0 when absent).
pub fn get_pred_context_seg_id(has_above: bool, above_sip: i32, has_left: bool, left_sip: i32) -> i32 {
    let a = if has_above { above_sip } else { 0 };
    let l = if has_left { left_sip } else { 0 };
    a + l
}

/// `write_inter_segment_id` (`av1/encoder/bitstream.c:920`): the inter-frame per-block
/// segment id. `preskip` selects the before-skip call (coded only when `segid_preskip`)
/// vs the after-skip call (coded only when `!segid_preskip`; a skipped block codes
/// nothing). When coded and `temporal_update` is on, a `seg_id_predicted` flag is coded
/// on the (caller-selected) prediction CDF, and the spatial segment id follows only when
/// the flag is 0; otherwise the spatial id is coded directly (via [`write_segment_id`]).
#[allow(clippy::too_many_arguments)]
pub fn write_inter_segment_id(
    enc: &mut OdEcEnc,
    update_map: bool,
    preskip: bool,
    segid_preskip: bool,
    skip: bool,
    temporal_update: bool,
    seg_id_predicted: i32,
    pred_cdf: &mut [u16],
    seg_cdf: &mut [u16],
    seg_enabled: bool,
    segment_id: i32,
    seg_pred: i32,
    last_active_segid: i32,
) {
    if !update_map {
        return;
    }
    let mut do_seg_block = false;
    if preskip {
        if segid_preskip {
            do_seg_block = true;
        }
    } else if !segid_preskip {
        if skip {
            // write_segment_id(skip_txfm=true): sets the seg id, codes nothing.
            write_segment_id(enc, seg_cdf, seg_enabled, update_map, true, segment_id, seg_pred, last_active_segid);
        } else {
            do_seg_block = true;
        }
    }
    if do_seg_block {
        if temporal_update {
            write_symbol(enc, seg_id_predicted, pred_cdf, 2);
            if seg_id_predicted == 0 {
                write_segment_id(enc, seg_cdf, seg_enabled, update_map, false, segment_id, seg_pred, last_active_segid);
            }
        } else {
            write_segment_id(enc, seg_cdf, seg_enabled, update_map, false, segment_id, seg_pred, last_active_segid);
        }
    }
}

/// `pack_inter_mode_mvs` prefix (`av1/encoder/bitstream.c:1092`): the INTER per-block
/// driver up to the inter/intra mode split — inter_segment_id (before skip) -> skip_mode
/// -> skip (= 1 when skip_mode, else write_skip) -> inter_segment_id (after skip) -> cdef
/// -> delta_q_params -> is_inter (coded only when not a skip-mode block). Returns
/// `(skip, skip_mode)`; the caller returns before the mode coding when `skip_mode`. All
/// state (segment/skip-mode/skip/cdef/delta/intra-inter CDFs + running values) updated in
/// place.
#[allow(clippy::too_many_arguments)]
pub fn write_inter_prefix(
    enc: &mut OdEcEnc,
    update_map: bool,
    segid_preskip: bool,
    temporal_update: bool,
    seg_id_predicted: i32,
    pred_cdf: &mut [u16],
    seg_cdf: &mut [u16],
    seg_enabled: bool,
    segment_id: i32,
    seg_pred: i32,
    last_active_segid: i32,
    skip_mode_cdf: &mut [u16],
    frame_skip_mode_flag: bool,
    sm_seg_skip: bool,
    sm_comp_allowed: bool,
    sm_seg_ref_gmv: bool,
    skip_mode: i32,
    skip_cdf: &mut [u16],
    skip_seg_active: bool,
    skip_txfm: i32,
    coded_lossless: bool,
    allow_intrabc: bool,
    mi_row: i32,
    mi_col: i32,
    mib_size: i32,
    sb_size: usize,
    cdef_transmitted: &mut [bool; 4],
    cdef_bits: u32,
    cdef_strength: i32,
    dq_present: bool,
    dlf_present: bool,
    dlf_multi: bool,
    num_planes: i32,
    bsize: usize,
    cur_qindex: i32,
    current_base_qindex: &mut i32,
    dq_res: i32,
    mbmi_delta_lf: &[i32; FRAME_LF_COUNT],
    xd_delta_lf: &mut [i32; FRAME_LF_COUNT],
    mbmi_delta_lf_from_base: i32,
    xd_delta_lf_from_base: &mut i32,
    dlf_res: i32,
    delta_q_cdf: &mut [u16],
    delta_lf_multi_cdf: &mut [[u16; 5]; FRAME_LF_COUNT],
    delta_lf_cdf: &mut [u16],
    intra_inter_cdf: &mut [u16],
    seg_ref_frame_active: bool,
    seg_globalmv_active: bool,
    is_inter: i32,
) -> (i32, i32) {
    write_inter_segment_id(enc, update_map, true, segid_preskip, false, temporal_update, seg_id_predicted, pred_cdf, seg_cdf, seg_enabled, segment_id, seg_pred, last_active_segid);
    write_skip_mode(enc, skip_mode_cdf, frame_skip_mode_flag, sm_seg_skip, sm_comp_allowed, sm_seg_ref_gmv, skip_mode);
    let skip = if skip_mode != 0 { 1 } else { write_skip(enc, skip_cdf, skip_seg_active, skip_txfm) };
    write_inter_segment_id(enc, update_map, false, segid_preskip, skip != 0, temporal_update, seg_id_predicted, pred_cdf, seg_cdf, seg_enabled, segment_id, seg_pred, last_active_segid);
    write_cdef(enc, coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size, skip, cdef_transmitted, cdef_bits, cdef_strength);
    let super_block_upper_left = (mi_row & (mib_size - 1)) == 0 && (mi_col & (mib_size - 1)) == 0;
    write_delta_q_params_sb(enc, dq_present, dlf_present, dlf_multi, num_planes, bsize, sb_size, skip, super_block_upper_left, cur_qindex, current_base_qindex, dq_res, mbmi_delta_lf, xd_delta_lf, mbmi_delta_lf_from_base, xd_delta_lf_from_base, dlf_res, delta_q_cdf, delta_lf_multi_cdf, delta_lf_cdf);
    if skip_mode == 0 {
        write_is_inter(enc, intra_inter_cdf, seg_ref_frame_active, seg_globalmv_active, is_inter);
    }
    (skip, skip_mode)
}

// --- inter-mode-body gates + mode-context analysis (mvref_common.h) ---

const MB_MODE_COUNT: i32 = 25;
/// `compound_mode_ctx_map[3][COMP_NEWMV_CTXS]` (`mvref_common.h`).
const COMPOUND_MODE_CTX_MAP: [[i32; 5]; 3] = [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]];

/// `is_inter_compound_mode` (`blockd.h`): a compound inter mode (`NEAREST_NEARESTMV`..).
pub fn is_inter_compound_mode(mode: i32) -> bool {
    (NEAREST_NEARESTMV..MB_MODE_COUNT).contains(&mode)
}

/// `is_inter_singleref_mode` (`blockd.h`): a single-ref inter mode (`NEARESTMV`..`NEWMV`).
pub fn is_inter_singleref_mode(mode: i32) -> bool {
    (NEARESTMV..NEAREST_NEARESTMV).contains(&mode)
}

/// `av1_mode_context_analyzer` (`mvref_common.h`): the inter-mode CDF context. For a
/// single-ref block it is the raw `mode_context` value; for a compound block it combines
/// the new-mv and ref-mv sub-contexts via `compound_mode_ctx_map`. `mode_context_val` is
/// `mode_context[av1_ref_frame_type(rf)]`; `is_compound` is `rf[1] > INTRA_FRAME`.
pub fn mode_context_analyzer(mode_context_val: i32, is_compound: bool) -> i32 {
    if !is_compound {
        return mode_context_val;
    }
    let newmv_ctx = mode_context_val & 7; // NEWMV_CTX_MASK
    let refmv_ctx = (mode_context_val >> 4) & 15; // REFMV_OFFSET=4, REFMV_CTX_MASK=15
    COMPOUND_MODE_CTX_MAP[(refmv_ctx >> 1) as usize][newmv_ctx.min(4) as usize]
}

// PREDICTION_MODE inter values (av1/common/enums.h); NEWMV/NEW_NEWMV defined above.
const NEAREST_NEWMV: i32 = 19;
const NEW_NEARESTMV: i32 = 20;
const NEAR_NEWMV: i32 = 21;
const NEW_NEARMV: i32 = 22;

/// The mode-dependent inter-block MV coding of `pack_inter_mode_mvs`
/// (`av1/encoder/bitstream.c`): NEWMV / NEW_NEWMV code one MV per reference (0..1+
/// is_compound); NEAREST_NEWMV / NEAR_NEWMV code the second reference's MV; NEW_NEARESTMV
/// / NEW_NEARMV code the first reference's. All share one `nmvc` (joints + both component
/// CDFs adapt across the two references). `usehp` is the caller's resolved precision
/// (`allow_high_precision_mv`, or `MV_SUBPEL_NONE` under `cur_frame_force_integer_mv`).
/// Each coded `diff = mv - ref_mv` must be non-zero (a zero diff would use a NEAR/NEAREST
/// mode instead).
#[allow(clippy::too_many_arguments)]
pub fn write_inter_block_mvs(
    enc: &mut OdEcEnc,
    mode: i32,
    is_compound: bool,
    diff_row: [i32; 2],
    diff_col: [i32; 2],
    usehp: i32,
    joints_cdf: &mut [u16],
    comp0: &mut [u16; 69],
    comp1: &mut [u16; 69],
) {
    if mode == NEWMV || mode == NEW_NEWMV {
        let refs = 1 + is_compound as usize;
        for r in 0..refs {
            encode_mv(enc, joints_cdf, comp0, comp1, diff_row[r], diff_col[r], usehp);
        }
    } else if mode == NEAREST_NEWMV || mode == NEAR_NEWMV {
        encode_mv(enc, joints_cdf, comp0, comp1, diff_row[1], diff_col[1], usehp);
    } else if mode == NEW_NEARESTMV || mode == NEW_NEARMV {
        encode_mv(enc, joints_cdf, comp0, comp1, diff_row[0], diff_col[0], usehp);
    }
}

/// The inter mode + drl coding of `pack_inter_mode_mvs` (`av1/encoder/bitstream.c`):
/// unless segment-skip forces the mode, code the compound-mode symbol (for a compound
/// mode) or the single-ref inter-mode cascade (for a single-ref mode), then — for NEWMV /
/// NEW_NEWMV / a NEAR mode — the dynamic-ref-list index. `inter_compound_mode_cdf` is the
/// caller's `[mode_ctx]`-selected slice; the single-ref CDFs are the full tables indexed
/// by `mode_ctx` internally.
#[allow(clippy::too_many_arguments)]
pub fn write_inter_mode_drl(
    enc: &mut OdEcEnc,
    seg_skip: bool,
    mode: i32,
    mode_ctx: i32,
    inter_compound_mode_cdf: &mut [u16],
    newmv_cdf: &mut [[u16; 3]; 6],
    zeromv_cdf: &mut [[u16; 3]; 2],
    refmv_cdf: &mut [[u16; 3]; 6],
    drl_cdf: &mut [[u16; 3]; 3],
    ref_mv_idx: i32,
    ref_mv_count: i32,
    weight: &[u16],
) {
    if seg_skip {
        return;
    }
    if is_inter_compound_mode(mode) {
        write_inter_compound_mode(enc, inter_compound_mode_cdf, mode);
    } else if is_inter_singleref_mode(mode) {
        write_inter_mode(enc, newmv_cdf, zeromv_cdf, refmv_cdf, mode, mode_ctx);
    }
    if mode == NEWMV || mode == NEW_NEWMV || have_nearmv_in_inter_mode(mode) {
        write_drl_idx(enc, drl_cdf, mode, ref_mv_idx, ref_mv_count, weight);
    }
}

/// The inter mode-body tail of `pack_inter_mode_mvs` (`av1/encoder/bitstream.c`):
/// interintra info (when allowed) -> motion mode (when `ref_frame[1] != INTRA_FRAME`)
/// -> compound type (when `has_second_ref`) -> the interpolation filter. interintra and
/// compound-type are mutually exclusive in practice and share the one `wedge_idx_cdf`.
/// All gates + CDFs are the caller's.
#[allow(clippy::too_many_arguments)]
pub fn write_inter_mode_tail(
    enc: &mut OdEcEnc,
    // interintra
    interintra_allowed: bool,
    interintra: i32,
    ii_cdf: &mut [u16],
    ii_mode: i32,
    ii_mode_cdf: &mut [u16],
    wedge_used_ii: bool,
    use_wedge_ii: i32,
    wedge_ii_cdf: &mut [u16],
    ii_wedge_index: i32,
    wedge_idx_cdf: &mut [u16],
    // motion mode
    motion_mode_present: bool,
    obmc_cdf: &mut [u16],
    mm_cdf: &mut [u16],
    last_motion_mode_allowed: i32,
    motion_mode: i32,
    // compound type
    has_second_ref: bool,
    masked_used: bool,
    comp_group_idx: i32,
    cgi_cdf: &mut [u16],
    dist_wtd: bool,
    compound_idx: i32,
    cidx_cdf: &mut [u16],
    wedge_used_ct: bool,
    comp_type: i32,
    ctype_cdf: &mut [u16],
    ct_wedge_index: i32,
    wedge_sign: i32,
    mask_type: i32,
    // interp filter
    interp_needed: bool,
    is_switchable: bool,
    enable_dual: bool,
    f0: i32,
    f1: i32,
    interp_cdf0: &mut [u16],
    interp_cdf1: &mut [u16],
) {
    write_interintra_info(
        enc, interintra_allowed, interintra, ii_cdf, ii_mode, ii_mode_cdf, wedge_used_ii,
        use_wedge_ii, wedge_ii_cdf, ii_wedge_index, wedge_idx_cdf,
    );
    if motion_mode_present {
        write_motion_mode(enc, obmc_cdf, mm_cdf, last_motion_mode_allowed, motion_mode);
    }
    if has_second_ref {
        write_compound_type_info(
            enc, masked_used, comp_group_idx, cgi_cdf, dist_wtd, compound_idx, cidx_cdf,
            wedge_used_ct, comp_type, ctype_cdf, ct_wedge_index, wedge_idx_cdf, wedge_sign, mask_type,
        );
    }
    write_mb_interp_filter(enc, interp_cdf0, interp_cdf1, interp_needed, is_switchable, enable_dual, f0, f1);
}

/// `av1_collect_neighbors_ref_counts` (`mvref_common.h`): tally the above and left
/// inter neighbours' reference frames into an 8-entry count array (used by the
/// reference-frame prediction contexts). A neighbour contributes when it is present and
/// an inter block (`use_intrabc` or `ref_frame[0] > INTRA_FRAME`): its `ref_frame[0]`,
/// and `ref_frame[1]` when it has a second reference (`ref_frame[1] > INTRA_FRAME`).
#[allow(clippy::too_many_arguments)]
pub fn collect_neighbors_ref_counts(
    has_above: bool,
    a_use_intrabc: bool,
    a_ref_frame0: i32,
    a_ref_frame1: i32,
    has_left: bool,
    l_use_intrabc: bool,
    l_ref_frame0: i32,
    l_ref_frame1: i32,
) -> [u8; 8] {
    let mut counts = [0u8; 8];
    if has_above && (a_use_intrabc || a_ref_frame0 > 0) {
        counts[a_ref_frame0 as usize] += 1;
        if a_ref_frame1 > 0 {
            counts[a_ref_frame1 as usize] += 1;
        }
    }
    if has_left && (l_use_intrabc || l_ref_frame0 > 0) {
        counts[l_ref_frame0 as usize] += 1;
        if l_ref_frame1 > 0 {
            counts[l_ref_frame1 as usize] += 1;
        }
    }
    counts
}

/// `subsize_lookup[EXT_PARTITION_TYPES][SQR_BLOCK_SIZES]` (`common_data.h`): the sub-block
/// size for a (partition, square-block-size-index) pair (`BLOCK_INVALID` = 255 where the
/// partition is illegal for that size).
const SUBSIZE_LOOKUP: [[u8; 6]; 10] = [
    [0, 3, 6, 9, 12, 15],        // PARTITION_NONE
    [255, 2, 5, 8, 11, 14],      // PARTITION_HORZ
    [255, 1, 4, 7, 10, 13],      // PARTITION_VERT
    [255, 0, 3, 6, 9, 12],       // PARTITION_SPLIT
    [255, 255, 5, 8, 11, 14],    // PARTITION_HORZ_A
    [255, 255, 5, 8, 11, 14],    // PARTITION_HORZ_B
    [255, 255, 4, 7, 10, 13],    // PARTITION_VERT_A
    [255, 255, 4, 7, 10, 13],    // PARTITION_VERT_B
    [255, 255, 17, 19, 21, 255], // PARTITION_HORZ_4
    [255, 255, 16, 18, 20, 255], // PARTITION_VERT_4
];

/// `get_sqr_bsize_idx` (`enums.h`/`blockd.h`): the 0..5 index of a square block size, or
/// `SQR_BLOCK_SIZES` (6) for a non-square block.
fn get_sqr_bsize_idx(bsize: usize) -> usize {
    match bsize {
        0 => 0,   // BLOCK_4X4
        3 => 1,   // BLOCK_8X8
        6 => 2,   // BLOCK_16X16
        9 => 3,   // BLOCK_32X32
        12 => 4,  // BLOCK_64X64
        15 => 5,  // BLOCK_128X128
        _ => 6,   // SQR_BLOCK_SIZES
    }
}

/// `get_partition_subsize` (`common_data.h`): the block size of a partition's sub-blocks
/// (`BLOCK_INVALID` = 255 for `PARTITION_INVALID` or an illegal partition/size). Used by
/// the partition-tree recursion (`write_modes_sb`).
pub fn get_partition_subsize(bsize: usize, partition: i32) -> i32 {
    if partition == 255 {
        return 255; // PARTITION_INVALID -> BLOCK_INVALID
    }
    let idx = get_sqr_bsize_idx(bsize);
    if idx >= 6 {
        return 255; // BLOCK_INVALID
    }
    SUBSIZE_LOOKUP[partition as usize][idx] as i32
}

/// `partition_context_lookup[BLOCK_SIZES_ALL]` (`common_data.h`): the (above, left)
/// partition-context bytes a block of each size stamps into the neighbour context.
const PARTITION_CONTEXT_LOOKUP: [(i8, i8); 22] = [
    (31, 31), (31, 30), (30, 31), (30, 30), (30, 28), (28, 30), (28, 28), (28, 24),
    (24, 28), (24, 24), (24, 16), (16, 24), (16, 16), (16, 0), (0, 16), (0, 0),
    (31, 28), (28, 31), (30, 24), (24, 30), (28, 16), (16, 28),
];

/// `update_partition_context` (`av1_common_int.h`): stamp `subsize`'s context bytes over
/// `above[mi_col..+bw]` and `left[(mi_row & MAX_MIB_MASK)..+bh]` (bw/bh from `bsize`).
fn update_partition_context(above: &mut [i8], left: &mut [i8], mi_row: i32, mi_col: i32, subsize: usize, bsize: usize) {
    let (a, l) = PARTITION_CONTEXT_LOOKUP[subsize];
    let bw = MI_SIZE_WIDE[bsize] as usize;
    let bh = MI_SIZE_HIGH[bsize] as usize;
    let ac = mi_col as usize;
    let lc = (mi_row & MAX_MIB_MASK as i32) as usize;
    for x in above[ac..ac + bw].iter_mut() {
        *x = a;
    }
    for x in left[lc..lc + bh].iter_mut() {
        *x = l;
    }
}

/// `update_ext_partition_context` (`av1_common_int.h`): after coding a partition, update
/// the neighbour partition context for the sub-blocks it created — one stamp for the
/// simple splits, two for the extended (HORZ_A/B, VERT_A/B) types.
pub fn update_ext_partition_context(above: &mut [i8], left: &mut [i8], mi_row: i32, mi_col: i32, subsize: usize, bsize: usize, partition: i32) {
    if bsize < BLOCK_8X8 {
        return;
    }
    let hbs = MI_SIZE_WIDE[bsize] / 2;
    let bsize2 = get_partition_subsize(bsize, PARTITION_SPLIT as i32) as usize;
    match partition {
        3 => {
            // PARTITION_SPLIT: only BLOCK_8X8 falls through to the stamp.
            if bsize == BLOCK_8X8 {
                update_partition_context(above, left, mi_row, mi_col, subsize, bsize);
            }
        }
        0 | 1 | 2 | 8 | 9 => {
            // NONE / HORZ / VERT / HORZ_4 / VERT_4
            update_partition_context(above, left, mi_row, mi_col, subsize, bsize);
        }
        4 => {
            // HORZ_A
            update_partition_context(above, left, mi_row, mi_col, bsize2, subsize);
            update_partition_context(above, left, mi_row + hbs, mi_col, subsize, subsize);
        }
        5 => {
            // HORZ_B
            update_partition_context(above, left, mi_row, mi_col, subsize, subsize);
            update_partition_context(above, left, mi_row + hbs, mi_col, bsize2, subsize);
        }
        6 => {
            // VERT_A
            update_partition_context(above, left, mi_row, mi_col, bsize2, subsize);
            update_partition_context(above, left, mi_row, mi_col + hbs, subsize, subsize);
        }
        7 => {
            // VERT_B
            update_partition_context(above, left, mi_row, mi_col, subsize, subsize);
            update_partition_context(above, left, mi_row, mi_col + hbs, bsize2, subsize);
        }
        _ => {}
    }
}

/// `write_modes_sb` per-node partition step (`av1/encoder/bitstream.c`): select the
/// partition CDF by the (threaded) above/left partition context, code the partition (full
/// CDF when the block is in-frame; the 2-way gather at a partial edge; nothing when the
/// block is fully off both edges), then update the neighbour partition context for the
/// sub-blocks. This is the per-node operation the partition-tree recursion repeats.
#[allow(clippy::too_many_arguments)]
pub fn write_partition_node(
    enc: &mut OdEcEnc,
    above: &mut [i8],
    left: &mut [i8],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    partition: i32,
    mi_rows: i32,
    mi_cols: i32,
    partition_cdf_arena: &mut [[u16; 11]; 20],
) {
    if bsize >= BLOCK_8X8 {
        let hbs = MI_SIZE_WIDE[bsize] / 2;
        let has_rows = (mi_row + hbs) < mi_rows;
        let has_cols = (mi_col + hbs) < mi_cols;
        let ctx = partition_plane_context(above, left, mi_row as usize, mi_col as usize, bsize) as usize;
        write_partition(enc, &mut partition_cdf_arena[ctx], partition_cdf_length(bsize), partition, has_rows, has_cols, bsize);
    }
    let subsize = get_partition_subsize(bsize, partition) as usize;
    update_ext_partition_context(above, left, mi_row, mi_col, subsize, bsize, partition);
}

const PARTITION_SPLIT_MODE: i32 = 3;

#[allow(clippy::too_many_arguments)]
fn write_modes_sb_recurse(
    enc: &mut OdEcEnc,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    tree: &[i8],
    idx: &mut usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) {
    if bsize < BLOCK_8X8 {
        return; // 4X4 leaf under an 8X8 split: no partition, no context
    }
    let p = tree[*idx] as i32;
    *idx += 1;
    let subsize = get_partition_subsize(bsize, p) as usize;
    let hbs = MI_SIZE_WIDE[bsize] / 2;
    let ctx = partition_plane_context(above, left, mi_row as usize, mi_col as usize, bsize) as usize;
    write_partition(enc, &mut arena[ctx], partition_cdf_length(bsize), p, true, true, bsize);
    if p == PARTITION_SPLIT_MODE && bsize > BLOCK_8X8 {
        write_modes_sb_recurse(enc, above, left, arena, tree, idx, mi_row, mi_col, subsize);
        write_modes_sb_recurse(enc, above, left, arena, tree, idx, mi_row, mi_col + hbs, subsize);
        write_modes_sb_recurse(enc, above, left, arena, tree, idx, mi_row + hbs, mi_col, subsize);
        write_modes_sb_recurse(enc, above, left, arena, tree, idx, mi_row + hbs, mi_col + hbs, subsize);
    }
    update_ext_partition_context(above, left, mi_row, mi_col, subsize, bsize, p);
}

/// `write_modes_sb` (`av1/encoder/bitstream.c`): the partition-tree recursion for a
/// superblock. At each node it codes the partition (context threaded through the tree
/// via the neighbour partition context) and either recurses into four quadrants
/// (`PARTITION_SPLIT`) or writes the sub-blocks. This fully-in-frame form takes the
/// per-node partitions as a pre-order sequence and stubs the block content (which is
/// coded by the separately-validated block drivers); it validates the partition symbol
/// stream, the cross-node context threading, and the CDF-arena adaptation. Returns the
/// number of `tree` entries consumed.
#[allow(clippy::too_many_arguments)]
pub fn write_modes_sb(
    enc: &mut OdEcEnc,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    tree: &[i8],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> usize {
    let mut idx = 0usize;
    write_modes_sb_recurse(enc, above, left, arena, tree, &mut idx, mi_row, mi_col, bsize);
    idx
}

/// `write_modes` (`av1/encoder/bitstream.c`), partition-only form: the tile loop over
/// superblocks. The above partition context is zeroed once for the tile and threads
/// vertically across SB rows; the left context is zeroed at the start of each SB row and
/// threads horizontally across the row's SBs. Each SB is walked by the partition-tree
/// recursion, consuming the concatenated (row-major) pre-order partition sequence. Block
/// content is stubbed (coded by the block drivers). Returns the entries consumed.
#[allow(clippy::too_many_arguments)]
pub fn write_modes_tile(
    enc: &mut OdEcEnc,
    above: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    tree: &[i8],
    n_sb_rows: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
) -> usize {
    for a in above.iter_mut() {
        *a = 0; // av1_zero_above_context
    }
    let mut idx = 0usize;
    for r in 0..n_sb_rows {
        let mut left = [0i8; 32]; // av1_zero_left_context per SB row
        for c in 0..n_sb_cols {
            write_modes_sb_recurse(enc, above, &mut left, arena, tree, &mut idx, r * sb_mi, c * sb_mi, sb_size);
        }
    }
    idx
}

/// `max_txsize_rect_lookup[BLOCK_SIZES_ALL]` (`common_data.h`): the block's max
/// (rectangular) transform size — the starting size for the inter var-tx loop.
const MAX_TXSIZE_RECT_LOOKUP: [usize; 22] =
    [0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18];

/// `get_vartx_max_txsize` (`blockd.h`) for luma, non-lossless: `max_txsize_rect_lookup
/// [bsize]` (lossless would be `TX_4X4`, but the var-tx tx-size coding is gated off in
/// lossless).
pub fn get_vartx_max_txsize_luma(bsize: usize) -> usize {
    MAX_TXSIZE_RECT_LOOKUP[bsize]
}

/// The block-level inter transform-size loop of `write_modes_b`
/// (`av1/encoder/bitstream.c`): drive [`write_tx_size_vartx`] across the block's transform
/// grid — stepping `blk_row` by the max tx's height in units and `blk_col` by its width
/// in units — threading the above/left txfm context through the whole block. `max_tx` is
/// [`get_vartx_max_txsize_luma`].
#[allow(clippy::too_many_arguments)]
pub fn write_inter_txfm_size(
    enc: &mut OdEcEnc,
    txfm_partition_cdf: &mut [[u16; 3]],
    bsize: usize,
    inter_tx_size: &[usize; 16],
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    above_ctx: &mut [u8],
    left_ctx: &mut [u8],
    max_tx: usize,
) {
    let txbh = TX_SIZE_HIGH_UNIT[max_tx];
    let txbw = TX_SIZE_WIDE_UNIT[max_tx];
    let width = MI_SIZE_WIDE[bsize];
    let height = MI_SIZE_HIGH[bsize];
    let mut idy = 0;
    while idy < height {
        let mut idx = 0;
        while idx < width {
            write_tx_size_vartx(
                enc, txfm_partition_cdf, bsize, inter_tx_size, mb_to_right_edge, mb_to_bottom_edge,
                above_ctx, left_ctx, max_tx, 0, idy, idx,
            );
            idx += txbw;
        }
        idy += txbh;
    }
}

/// `get_unsigned_bits` (`common.h`): bits to represent `n` values (`get_msb(n) + 1`, 0 for 0).
fn get_unsigned_bits(n: u32) -> u32 {
    if n > 0 { get_msb(n) + 1 } else { 0 }
}

/// `write_uniform` (`av1/encoder/bitstream.c`) on the arithmetic coder: a near-uniform
/// code for `v` in `0..n` — `l-1` low values take `l-1` bits, the rest `l` bits.
fn write_uniform_arith(enc: &mut OdEcEnc, n: i32, v: i32) {
    let l = get_unsigned_bits(n as u32) as i32;
    let m = (1 << l) - n;
    if l == 0 {
        return;
    }
    if v < m {
        write_literal(enc, v, (l - 1) as u32);
    } else {
        write_literal(enc, m + ((v - m) >> 1), (l - 1) as u32);
        write_bit(enc, (v - m) & 1);
    }
}

/// `pack_map_tokens` (`av1/encoder/bitstream.c`): the palette colour-index map. The first
/// index is coded uniformly (`write_uniform`); each subsequent index is coded on the
/// `map_pb_cdf[palette_size_idx][color_ctx]` CDF (`n` symbols). `tokens[i]` is the colour
/// index and `color_ctxs[i]` its neighbour context (`color_ctxs[0]` unused). `map_cdf` is
/// the `[PALETTE_COLOR_INDEX_CONTEXTS]` slice already selected for this palette size.
pub fn pack_map_tokens(enc: &mut OdEcEnc, n: i32, tokens: &[i32], color_ctxs: &[usize], map_cdf: &mut [[u16; 9]; 5]) {
    write_uniform_arith(enc, n, tokens[0]);
    for i in 1..tokens.len() {
        write_symbol(enc, tokens[i], &mut map_cdf[color_ctxs[i]], n as usize);
    }
}

use crate::cdf::read_symbol;
use crate::dec::OdEcDec;

/// `read_partition` (`av1/decoder/decodeframe.c`): the decode-side inverse of
/// [`write_partition`]. With both rows and columns in-frame it reads the partition symbol
/// from the (context-selected) partition CDF (adapting it); at a frame edge it reads the
/// 2-way split-vs-alike bit from the gathered CDF (no adaptation); when neither rows nor
/// columns remain the partition is the forced `PARTITION_SPLIT`. `partition_cdf` is the
/// caller's `[ctx]`-selected slice (adapted in place, mirroring the encoder).
pub fn read_partition(
    dec: &mut OdEcDec,
    partition_cdf: &mut [u16],
    cdf_len: usize,
    has_rows: bool,
    has_cols: bool,
    bsize: usize,
) -> i32 {
    if !has_rows && !has_cols {
        return PARTITION_SPLIT_MODE;
    }
    if has_rows && has_cols {
        read_symbol(dec, partition_cdf, cdf_len)
    } else if !has_rows && has_cols {
        let cdf = partition_gather_vert_alike(partition_cdf, bsize);
        if dec.decode_cdf_q15(&cdf, 2) != 0 {
            PARTITION_SPLIT_MODE
        } else {
            PARTITION_HORZ as i32
        }
    } else {
        let cdf = partition_gather_horz_alike(partition_cdf, bsize);
        if dec.decode_cdf_q15(&cdf, 2) != 0 {
            PARTITION_SPLIT_MODE
        } else {
            PARTITION_VERT as i32
        }
    }
}

// --- decoder-side mode-info symbol readers (av1/decoder/decodemv.c), inverses of the
// corresponding write_* symbols; validated by encode->decode roundtrip. ---

/// `read_skip` — inverse of [`write_skip`]. Segment-level skip implies 1 (nothing read);
/// otherwise read the transform-skip bit from the 2-symbol skip CDF.
pub fn read_skip(dec: &mut OdEcDec, skip_cdf: &mut [u16], seg_skip_active: bool) -> i32 {
    if seg_skip_active {
        return 1;
    }
    read_symbol(dec, skip_cdf, 2)
}

/// `read_intra_y_mode` — inverse of [`write_intra_y_mode_kf`] / [`write_intra_y_mode_nonkf`]:
/// the luma prediction mode on the caller-selected CDF (`INTRA_MODES` symbols).
pub fn read_intra_y_mode(dec: &mut OdEcDec, y_cdf: &mut [u16]) -> i32 {
    read_symbol(dec, y_cdf, INTRA_MODES)
}

/// `read_intra_uv_mode` — inverse of [`write_intra_uv_mode`]: the chroma mode
/// (`UV_INTRA_MODES` symbols with CFL, one fewer without).
pub fn read_intra_uv_mode(dec: &mut OdEcDec, uv_mode_cdf: &mut [u16], cfl_allowed: bool) -> i32 {
    let n = UV_INTRA_MODES - usize::from(!cfl_allowed);
    read_symbol(dec, uv_mode_cdf, n)
}

/// `read_inter_compound_mode` — inverse of [`write_inter_compound_mode`]: the compound
/// inter mode (`INTER_COMPOUND_MODES` symbols, offset by `NEAREST_NEARESTMV`).
pub fn read_inter_compound_mode(dec: &mut OdEcDec, cdf: &mut [u16]) -> i32 {
    read_symbol(dec, cdf, INTER_COMPOUND_MODES) + NEAREST_NEARESTMV
}

/// `read_angle_delta` — inverse of [`write_angle_delta`]: the directional-mode angle delta
/// (`2*MAX_ANGLE_DELTA+1` symbols, offset by `-MAX_ANGLE_DELTA`).
pub fn read_angle_delta(dec: &mut OdEcDec, cdf: &mut [u16]) -> i32 {
    read_symbol(dec, cdf, (2 * MAX_ANGLE_DELTA + 1) as usize) - MAX_ANGLE_DELTA
}

/// `read_inter_mode` — inverse of [`write_inter_mode`]: the single-ref inter mode via the
/// 3-symbol cascade (is-not-NEWMV on `newmv_cdf[ctx&7]`, then is-not-GLOBALMV on
/// `zeromv_cdf[(ctx>>3)&1]`, then is-not-NEARESTMV on `refmv_cdf[(ctx>>4)&15]`).
pub fn read_inter_mode(
    dec: &mut OdEcDec,
    newmv_cdf: &mut [[u16; 3]; 6],
    zeromv_cdf: &mut [[u16; 3]; 2],
    refmv_cdf: &mut [[u16; 3]; 6],
    mode_ctx: i32,
) -> i32 {
    let newmv_ctx = (mode_ctx & 7) as usize;
    if read_symbol(dec, &mut newmv_cdf[newmv_ctx], 2) == 0 {
        return NEWMV;
    }
    let zeromv_ctx = ((mode_ctx >> 3) & 1) as usize;
    if read_symbol(dec, &mut zeromv_cdf[zeromv_ctx], 2) == 0 {
        return GLOBALMV;
    }
    let refmv_ctx = ((mode_ctx >> 4) & 15) as usize;
    if read_symbol(dec, &mut refmv_cdf[refmv_ctx], 2) != 0 {
        NEARMV
    } else {
        NEARESTMV
    }
}

/// `read_mv_component` — inverse of [`encode_mv_component`]: reads sign, class, the
/// class-0 or bit-tree integer part, and (per `precision`) the fractional and high-
/// precision bits, then reconstructs the signed component. When a part is not coded the
/// decoder uses the spec default (`fr = 3`, `hp = 1`).
pub fn read_mv_component(dec: &mut OdEcDec, cdf: &mut [u16; 69], precision: i32) -> i32 {
    let sign = read_symbol(dec, &mut cdf[0..3], 2);
    let mv_class = read_symbol(dec, &mut cdf[3..15], MV_CLASSES);
    let d = if mv_class == 0 {
        read_symbol(dec, &mut cdf[15..18], 2)
    } else {
        let mut dd = 0;
        for i in 0..mv_class {
            let off = 18 + (i as usize) * 3;
            dd |= read_symbol(dec, &mut cdf[off..off + 3], 2) << i;
        }
        dd
    };
    let fr = if precision > MV_SUBPEL_NONE {
        if mv_class == 0 {
            let off = 48 + (d as usize) * 5;
            read_symbol(dec, &mut cdf[off..off + 5], MV_FP_SIZE)
        } else {
            read_symbol(dec, &mut cdf[58..63], MV_FP_SIZE)
        }
    } else {
        3
    };
    let hp = if precision > MV_SUBPEL_LOW {
        if mv_class == 0 {
            read_symbol(dec, &mut cdf[63..66], 2)
        } else {
            read_symbol(dec, &mut cdf[66..69], 2)
        }
    } else {
        1
    };
    let offset = (d << 3) | (fr << 1) | hp;
    let mag = mv_class_base(mv_class) + offset + 1;
    if sign != 0 {
        -mag
    } else {
        mag
    }
}

/// `read_mv` — inverse of [`encode_mv`]: reads the MV joint, then the vertical and/or
/// horizontal components as the joint indicates. Returns `(diff_row, diff_col)`.
pub fn read_mv(
    dec: &mut OdEcDec,
    joints_cdf: &mut [u16],
    comp0: &mut [u16; 69],
    comp1: &mut [u16; 69],
    precision: i32,
) -> (i32, i32) {
    let j = read_symbol(dec, joints_cdf, 4); // MV_JOINTS
    let row = if j & 2 != 0 { read_mv_component(dec, comp0, precision) } else { 0 };
    let col = if j & 1 != 0 { read_mv_component(dec, comp1, precision) } else { 0 };
    (row, col)
}

/// `read_drl_idx` — inverse of [`write_drl_idx`] (`av1/decoder/decodemv.c`): reconstructs
/// `ref_mv_idx` from the DRL bit cascade. NEWMV/NEW_NEWMV walk idx 0..2 with
/// `ref_mv_idx = idx + drl`; the have-nearmv modes walk idx 1..3 with
/// `ref_mv_idx = idx + drl - 1`; each bit uses the `av1_drl_ctx(weight, idx)` CDF and the
/// walk stops at the first `drl == 0`. Modes with no DRL signaling return 0.
pub fn read_drl_idx(
    dec: &mut OdEcDec,
    drl_cdf: &mut [[u16; 3]; 3],
    mode: i32,
    ref_mv_count: i32,
    weight: &[u16],
) -> i32 {
    let mut ref_mv_idx = 0;
    let new_mv = mode == NEWMV || mode == NEW_NEWMV;
    if new_mv {
        for idx in 0..2 {
            if ref_mv_count > idx + 1 {
                let ctx = av1_drl_ctx(weight, idx as usize);
                let drl = read_symbol(dec, &mut drl_cdf[ctx], 2);
                ref_mv_idx = idx + drl;
                if drl == 0 {
                    break;
                }
            }
        }
        return ref_mv_idx;
    }
    if have_nearmv_in_inter_mode(mode) {
        for idx in 1..3 {
            if ref_mv_count > idx + 1 {
                let ctx = av1_drl_ctx(weight, idx as usize);
                let drl = read_symbol(dec, &mut drl_cdf[ctx], 2);
                ref_mv_idx = idx + drl - 1;
                if drl == 0 {
                    break;
                }
            }
        }
    }
    ref_mv_idx
}

/// `read_ref_frames` — inverse of [`write_ref_frames`] (`av1/decoder/decodemv.c`).
/// Derives `(is_compound, comp_ref_type, ref0, ref1)` from the reference cascade:
/// the compound flag (only when reference-mode-select and compound is allowed), then per
/// mode the single-ref tree (cdfs[10..16]) or the unidir/bidir compound trees
/// (cdfs[1..10]). Reference-frame ids follow the enum (LAST=1..ALTREF=7); `ref1 = -1`
/// (NONE) for single. Returns `(false, -1, -1, -1)` when segment features suppress coding.
#[allow(clippy::type_complexity)]
pub fn read_ref_frames(
    dec: &mut OdEcDec,
    cdfs: &mut [[u16; 3]; 16],
    seg_ref_active: bool,
    seg_skipgmv_active: bool,
    reference_mode_is_select: bool,
    is_comp_ref_allowed: bool,
) -> (bool, i32, i32, i32) {
    if seg_ref_active || seg_skipgmv_active {
        return (false, -1, -1, -1);
    }
    let is_compound = if reference_mode_is_select && is_comp_ref_allowed {
        read_symbol(dec, &mut cdfs[0], 2) != 0
    } else {
        false
    };
    if is_compound {
        let comp_ref_type = read_symbol(dec, &mut cdfs[1], 2);
        if comp_ref_type == 0 {
            // UNIDIR_COMP_REFERENCE
            if read_symbol(dec, &mut cdfs[2], 2) != 0 {
                return (true, 0, 5, 7); // {BWDREF, ALTREF}
            }
            if read_symbol(dec, &mut cdfs[3], 2) != 0 {
                let bit2 = read_symbol(dec, &mut cdfs[4], 2);
                return (true, 0, 1, if bit2 != 0 { 4 } else { 3 }); // {LAST, GOLDEN|LAST3}
            }
            return (true, 0, 1, 2); // {LAST, LAST2}
        }
        // BIDIR_COMP_REFERENCE
        let ref0 = if read_symbol(dec, &mut cdfs[5], 2) == 0 {
            if read_symbol(dec, &mut cdfs[6], 2) != 0 { 2 } else { 1 }
        } else if read_symbol(dec, &mut cdfs[7], 2) != 0 {
            4
        } else {
            3
        };
        let ref1 = if read_symbol(dec, &mut cdfs[8], 2) == 0 {
            if read_symbol(dec, &mut cdfs[9], 2) != 0 { 6 } else { 5 }
        } else {
            7
        };
        (true, 1, ref0, ref1)
    } else {
        let ref0 = if read_symbol(dec, &mut cdfs[10], 2) != 0 {
            if read_symbol(dec, &mut cdfs[11], 2) == 0 {
                if read_symbol(dec, &mut cdfs[15], 2) != 0 { 6 } else { 5 }
            } else {
                7
            }
        } else if read_symbol(dec, &mut cdfs[12], 2) != 0 {
            if read_symbol(dec, &mut cdfs[14], 2) != 0 { 4 } else { 3 }
        } else if read_symbol(dec, &mut cdfs[13], 2) != 0 {
            2
        } else {
            1
        };
        (false, -1, ref0, -1)
    }
}

/// `read_selected_tx_size` — inverse of [`write_selected_tx_size`]
/// (`av1/decoder/decodemv.c` `read_selected_tx_size`): the intra tx-depth. Coded only
/// when the block signals a tx size (`bsize > BLOCK_4X4`) on the `(max_depths+1)`-symbol
/// CDF; otherwise depth 0.
pub fn read_selected_tx_size(
    dec: &mut OdEcDec,
    tx_size_cdf: &mut [u16],
    bsize: usize,
    max_depths: usize,
) -> i32 {
    if bsize > 0 {
        read_symbol(dec, tx_size_cdf, max_depths + 1)
    } else {
        0
    }
}

/// `read_filter_intra_mode_info` — inverse of [`write_filter_intra_mode_info`]
/// (`av1/decoder/decodemv.c`): the use-filter-intra flag (when allowed), then the
/// filter-intra mode if used. Returns `(use_filter_intra, mode)`; `(0, 0)` when not
/// allowed or not used.
pub fn read_filter_intra_mode_info(
    dec: &mut OdEcDec,
    use_cdf: &mut [u16],
    mode_cdf: &mut [u16],
    allowed: bool,
) -> (i32, i32) {
    if allowed {
        let use_filter_intra = read_symbol(dec, use_cdf, 2);
        let mode = if use_filter_intra != 0 {
            read_symbol(dec, mode_cdf, FILTER_INTRA_MODES)
        } else {
            0
        };
        (use_filter_intra, mode)
    } else {
        (0, 0)
    }
}

/// `read_tx_size_vartx` — inverse of [`write_tx_size_vartx`] (`av1/decoder/decodemv.c`
/// `read_tx_size_vartx`): the recursive inter var-tx tree. Reads one txfm-partition bit
/// per interior node (0 = leaf, 1 = split) on the `txfm_partition_context`-selected CDF,
/// recursing into `SUB_TX_SIZE_MAP` children on a split. At every leaf (a coded `0`, a
/// `TX_4X4` sub-split, or the forced `MAX_VARTX_DEPTH` leaf) it records the leaf tx size
/// at the block's `get_txb_size_index` slot and updates the above/left txfm context — the
/// same slot the encoder reads, so the reconstructed `inter_tx_size` re-encodes identically.
#[allow(clippy::too_many_arguments)]
pub fn read_tx_size_vartx(
    dec: &mut OdEcDec,
    txfm_partition_cdf: &mut [[u16; 3]],
    bsize: usize,
    inter_tx_size: &mut [usize; 16],
    mb_to_right_edge: i32,
    mb_to_bottom_edge: i32,
    above_ctx: &mut [u8],
    left_ctx: &mut [u8],
    tx_size: usize,
    depth: i32,
    blk_row: i32,
    blk_col: i32,
) {
    let max_blocks_high = max_block_high(bsize, mb_to_bottom_edge);
    let max_blocks_wide = max_block_wide(bsize, mb_to_right_edge);
    if blk_row >= max_blocks_high || blk_col >= max_blocks_wide {
        return;
    }

    let (bc, br) = (blk_col as usize, blk_row as usize);
    if depth == MAX_VARTX_DEPTH {
        inter_tx_size[get_txb_size_index(bsize, blk_row, blk_col)] = tx_size;
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
        return;
    }

    let ctx = txfm_partition_context(above_ctx[bc], left_ctx[br], bsize, tx_size);
    let is_split = read_symbol(dec, &mut txfm_partition_cdf[ctx], 2);
    if is_split == 0 {
        inter_tx_size[get_txb_size_index(bsize, blk_row, blk_col)] = tx_size;
        txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], tx_size, tx_size);
    } else {
        let sub_txs = SUB_TX_SIZE_MAP[tx_size];
        let bsw = TX_SIZE_WIDE_UNIT[sub_txs];
        let bsh = TX_SIZE_HIGH_UNIT[sub_txs];
        if sub_txs == TX_4X4 {
            inter_tx_size[get_txb_size_index(bsize, blk_row, blk_col)] = sub_txs;
            txfm_partition_update(&mut above_ctx[bc..], &mut left_ctx[br..], sub_txs, tx_size);
            return;
        }
        let mut row = 0;
        while row < TX_SIZE_HIGH_UNIT[tx_size] {
            let offsetr = blk_row + row;
            let mut col = 0;
            while col < TX_SIZE_WIDE_UNIT[tx_size] {
                let offsetc = blk_col + col;
                read_tx_size_vartx(
                    dec, txfm_partition_cdf, bsize, inter_tx_size, mb_to_right_edge,
                    mb_to_bottom_edge, above_ctx, left_ctx, sub_txs, depth + 1, offsetr, offsetc,
                );
                col += bsw;
            }
            row += bsh;
        }
    }
}

/// `read_uniform` (`av1/decoder/decodemv.c`) on the arithmetic coder — inverse of
/// [`write_uniform_arith`]: a near-uniform value in `[0, n)`. Reads `l-1` MSB-first bits;
/// values below `m = (1<<l) - n` use those bits directly, otherwise one extra bit
/// disambiguates the doubled range.
fn read_uniform_arith(dec: &mut OdEcDec, n: i32) -> i32 {
    let l = get_unsigned_bits(n as u32) as i32;
    let m = (1 << l) - n;
    if l == 0 {
        return 0;
    }
    let v = read_literal(dec, (l - 1) as u32);
    if v < m {
        v
    } else {
        (v << 1) - m + read_bit(dec)
    }
}

/// `read_map_tokens` — inverse of [`pack_map_tokens`] (`av1/decoder/decodemv.c` palette
/// colour-index map). The first index is read uniformly; each subsequent index is read on
/// the `n`-symbol `color_map_cdf[color_ctxs[i]]`. `out.len()` is the map size; the caller
/// supplies the running colour contexts (computed incrementally from the decoded map in a
/// full decode, precomputed here to mirror the encoder's token/context layout).
pub fn read_map_tokens(
    dec: &mut OdEcDec,
    n: i32,
    color_ctxs: &[usize],
    map_cdf: &mut [[u16; 9]; 5],
    out: &mut [i32],
) {
    if out.is_empty() {
        return;
    }
    out[0] = read_uniform_arith(dec, n);
    for i in 1..out.len() {
        out[i] = read_symbol(dec, &mut map_cdf[color_ctxs[i]], n as usize);
    }
}

/// `read_delta_palette_colors` — inverse of [`delta_encode_palette_colors`]
/// (`av1/decoder/decodemv.c`): the ascending non-cached palette colours. The first is a
/// raw `bit_depth`-bit literal; the excess bit-width over `bit_depth - 3` is read in 2
/// bits; each subsequent colour is `prev + (delta_raw + min_val)` with the same
/// range-driven bit-width shrink as the encoder. `min_val` is 1 for luma, 0 for chroma-U.
pub fn read_delta_palette_colors(dec: &mut OdEcDec, num: usize, bit_depth: i32, min_val: i32) -> Vec<i32> {
    let mut colors = vec![0i32; num];
    if num == 0 {
        return colors;
    }
    colors[0] = read_literal(dec, bit_depth as u32);
    if num == 1 {
        return colors;
    }
    let min_bits = bit_depth - 3;
    let mut bits = read_literal(dec, 2) + min_bits;
    let mut range = (1 << bit_depth) - colors[0] - min_val;
    for i in 1..num {
        let delta = read_literal(dec, bits as u32) + min_val;
        colors[i] = colors[i - 1] + delta;
        range -= delta;
        bits = bits.min(aom_ceil_log2(range));
    }
    colors
}

/// `read_palette_colors_v` — inverse of [`write_palette_colors_v`]
/// (`av1/decoder/decodemv.c`): the V palette plane. A leading flag selects delta vs raw
/// coding. Delta mode reads the `bits_v - (bit_depth-4)` excess in 2 bits, the first
/// colour raw, then each `|Δ|` (or its `2^bd` complement) plus a sign bit, reconstructing
/// `colors[i] = wrap(colors[i-1] ± Δ)` modulo `2^bit_depth` (a zero literal repeats the
/// previous colour, no sign). Raw mode reads `n` raw `bit_depth`-bit literals.
pub fn read_palette_colors_v(dec: &mut OdEcDec, n: usize, bit_depth: i32) -> Vec<u16> {
    let max_val = 1i32 << bit_depth;
    let mut colors = vec![0u16; n];
    if read_bit(dec) != 0 {
        let min_bits_v = bit_depth - 4;
        let bits_v = read_literal(dec, 2) + min_bits_v;
        colors[0] = read_literal(dec, bit_depth as u32) as u16;
        for i in 1..n {
            let d = read_literal(dec, bits_v as u32);
            if d == 0 {
                colors[i] = colors[i - 1];
                continue;
            }
            let delta = if read_bit(dec) != 0 { -d } else { d };
            let mut val = colors[i - 1] as i32 + delta;
            if val < 0 {
                val += max_val;
            }
            if val >= max_val {
                val -= max_val;
            }
            colors[i] = val as u16;
        }
    } else {
        for c in colors.iter_mut() {
            *c = read_literal(dec, bit_depth as u32) as u16;
        }
    }
    colors
}

/// `read_is_inter` — inverse of [`write_is_inter`]: the intra/inter flag. Coded only
/// when neither the segment ref-frame nor global-mv feature forces the block; the
/// forced cases infer inter (global-mv) or the caller's segment ref (ref feature).
pub fn read_is_inter(
    dec: &mut OdEcDec,
    intra_inter_cdf: &mut [u16],
    seg_ref_frame_active: bool,
    seg_globalmv_active: bool,
) -> i32 {
    if seg_ref_frame_active || seg_globalmv_active {
        return 1; // inferred inter (caller overrides for a ref-to-intra segment)
    }
    read_symbol(dec, intra_inter_cdf, 2)
}

/// `read_motion_mode` — inverse of [`write_motion_mode`]: SIMPLE when no other mode is
/// allowed, the OBMC on/off bit when only OBMC is allowed, else the full motion mode.
pub fn read_motion_mode(
    dec: &mut OdEcDec,
    obmc_cdf: &mut [u16],
    motion_mode_cdf: &mut [u16],
    last_motion_mode_allowed: i32,
) -> i32 {
    match last_motion_mode_allowed {
        0 => 0, // SIMPLE
        1 => read_symbol(dec, obmc_cdf, 2),
        _ => read_symbol(dec, motion_mode_cdf, MOTION_MODES),
    }
}

/// `read_mb_interp_filter` — inverse of [`write_mb_interp_filter`]: the (dual)
/// interpolation filters. Coded only when needed and switchable; without dual-filter
/// the vertical filter mirrors the horizontal. Returns `(filter0, filter1)`.
pub fn read_mb_interp_filter(
    dec: &mut OdEcDec,
    cdf0: &mut [u16],
    cdf1: &mut [u16],
    interp_needed: bool,
    is_switchable: bool,
    enable_dual_filter: bool,
) -> (i32, i32) {
    if !interp_needed || !is_switchable {
        return (0, 0); // inferred (the frame's fixed filter)
    }
    let f0 = read_symbol(dec, cdf0, SWITCHABLE_FILTERS);
    let f1 = if enable_dual_filter {
        read_symbol(dec, cdf1, SWITCHABLE_FILTERS)
    } else {
        f0
    };
    (f0, f1)
}

/// Read the exp-Golomb delta magnitude+sign shared by delta-q / delta-lf: the small
/// value on the `probs+1`-symbol CDF, then (when it saturates) the get_msb remainder,
/// then the sign. Inverse of the `write_delta_*` tail.
fn read_delta_value(dec: &mut OdEcDec, cdf: &mut [u16], probs: usize, small: i32) -> i32 {
    let s = read_symbol(dec, cdf, probs + 1);
    let mut abs = s;
    if s == small {
        let rem_bits = read_literal(dec, 3) + 1;
        let thr = (1 << rem_bits) + 1;
        abs = read_literal(dec, rem_bits as u32) + thr;
    }
    if abs > 0 && read_bit(dec) != 0 {
        -abs
    } else {
        abs
    }
}

/// `read_delta_qindex` — inverse of [`write_delta_qindex`].
pub fn read_delta_qindex(dec: &mut OdEcDec, delta_q_cdf: &mut [u16]) -> i32 {
    read_delta_value(dec, delta_q_cdf, DELTA_Q_PROBS, DELTA_Q_SMALL)
}

/// `read_delta_lflevel` — inverse of [`write_delta_lflevel`].
pub fn read_delta_lflevel(dec: &mut OdEcDec, delta_lf_cdf: &mut [u16]) -> i32 {
    read_delta_value(dec, delta_lf_cdf, DELTA_LF_PROBS, DELTA_LF_SMALL)
}

/// `av1_neg_deinterleave` (`av1/common/seg_common.c`): inverse of [`neg_interleave`] —
/// recover the segment id from its recentred code.
pub fn neg_deinterleave(diff: i32, ref_: i32, max: i32) -> i32 {
    if ref_ == 0 {
        return diff;
    }
    if ref_ >= max - 1 {
        return max - diff - 1;
    }
    if 2 * ref_ < max {
        if diff <= 2 * ref_ {
            if diff & 1 != 0 {
                ref_ + ((diff + 1) >> 1)
            } else {
                ref_ - (diff >> 1)
            }
        } else {
            diff
        }
    } else if diff <= 2 * (max - ref_ - 1) {
        if diff & 1 != 0 {
            ref_ + ((diff + 1) >> 1)
        } else {
            ref_ - (diff >> 1)
        }
    } else {
        max - (diff + 1)
    }
}

/// `read_segment_id` — inverse of [`write_segment_id`]: the spatial-pred segment id
/// (neg-deinterleaved around `pred`). The caller applies the coding gate (segmentation
/// enabled + map update + not skip-inferred).
pub fn read_segment_id(dec: &mut OdEcDec, pred_cdf: &mut [u16], pred: i32, last_active_segid: i32) -> i32 {
    let coded_id = read_symbol(dec, pred_cdf, MAX_SEGMENTS_MI);
    neg_deinterleave(coded_id, pred, last_active_segid + 1)
}

/// `read_cfl_alphas` — inverse of [`write_cfl_alphas`]: the CfL joint sign, then the
/// per-plane alpha magnitudes for whichever planes are signed. Returns
/// `(joint_sign, idx)` with `idx = (u_alpha << 4) | v_alpha`.
pub fn read_cfl_alphas(
    dec: &mut OdEcDec,
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
) -> (i32, i32) {
    let joint_sign = read_symbol(dec, cfl_sign_cdf, CFL_JOINT_SIGNS);
    let mut idx = 0;
    if cfl_sign_u(joint_sign) != 0 {
        let ctx = cfl_context_u(joint_sign) as usize;
        idx |= read_symbol(dec, &mut cfl_alpha_cdf[ctx], CFL_ALPHABET_SIZE) << 4;
    }
    if cfl_sign_v(joint_sign) != 0 {
        let ctx = cfl_context_v(joint_sign) as usize;
        idx |= read_symbol(dec, &mut cfl_alpha_cdf[ctx], CFL_ALPHABET_SIZE);
    }
    (joint_sign, idx)
}

/// `read_skip_mode` — inverse of [`write_skip_mode`]: the skip-mode flag, coded only
/// when the frame allows it and no segment feature suppresses it; 0 otherwise.
#[allow(clippy::too_many_arguments)]
pub fn read_skip_mode(
    dec: &mut OdEcDec,
    skip_mode_cdf: &mut [u16],
    frame_skip_mode_flag: bool,
    seg_skip_active: bool,
    is_comp_ref_allowed: bool,
    seg_ref_or_gmv_active: bool,
) -> i32 {
    if !frame_skip_mode_flag || seg_skip_active || !is_comp_ref_allowed || seg_ref_or_gmv_active {
        return 0;
    }
    read_symbol(dec, skip_mode_cdf, 2)
}

/// `read_intrabc_info` — inverse of [`write_intrabc_info`]: the intrabc flag, then (if
/// set) the block vector via the MV reader at `MV_SUBPEL_NONE` (integer-pel DV).
/// Returns `(use_intrabc, diff_row, diff_col)`.
pub fn read_intrabc_info(
    dec: &mut OdEcDec,
    intrabc_cdf: &mut [u16],
    ndvc_joints: &mut [u16],
    ndvc_comp0: &mut [u16; 69],
    ndvc_comp1: &mut [u16; 69],
) -> (i32, i32, i32) {
    let use_intrabc = read_symbol(dec, intrabc_cdf, 2);
    if use_intrabc != 0 {
        let (r, c) = read_mv(dec, ndvc_joints, ndvc_comp0, ndvc_comp1, MV_SUBPEL_NONE);
        (use_intrabc, r, c)
    } else {
        (use_intrabc, 0, 0)
    }
}

/// `read_cdef` — inverse of [`write_cdef`]: the per-64x64-unit CDEF strength literal,
/// once per unit (tracked in `cdef_transmitted`, reset at the SB upper-left) and only
/// for a non-skip block. Returns the strength, or `-1` when nothing is read.
#[allow(clippy::too_many_arguments)]
pub fn read_cdef(
    dec: &mut OdEcDec,
    coded_lossless: bool,
    allow_intrabc: bool,
    mi_row: i32,
    mi_col: i32,
    mib_size: i32,
    sb_size: usize,
    skip: i32,
    cdef_transmitted: &mut [bool; 4],
    cdef_bits: u32,
) -> i32 {
    if coded_lossless || allow_intrabc {
        return -1;
    }
    let sb_mask = mib_size - 1;
    if (mi_row & sb_mask) == 0 && (mi_col & sb_mask) == 0 {
        *cdef_transmitted = [false; 4];
    }
    let index_mask = 1 << (6 - MI_SIZE_LOG2);
    let cdef_unit_row = i32::from((mi_row & index_mask) != 0);
    let cdef_unit_col = i32::from((mi_col & index_mask) != 0);
    let index = if sb_size == BLOCK_128X128 {
        (cdef_unit_col + 2 * cdef_unit_row) as usize
    } else {
        0
    };
    if !cdef_transmitted[index] && skip == 0 {
        let strength = read_literal(dec, cdef_bits);
        cdef_transmitted[index] = true;
        strength
    } else {
        -1
    }
}

/// `read_interintra_info` — inverse of [`write_interintra_info`]: the inter-intra flag,
/// then (if set) the inter-intra mode and the optional wedge flag + index. Returns
/// `(interintra, mode, use_wedge, wedge_index)`.
#[allow(clippy::too_many_arguments)]
pub fn read_interintra_info(
    dec: &mut OdEcDec,
    allowed: bool,
    interintra_cdf: &mut [u16],
    interintra_mode_cdf: &mut [u16],
    wedge_used: bool,
    wedge_interintra_cdf: &mut [u16],
    wedge_idx_cdf: &mut [u16],
) -> (i32, i32, i32, i32) {
    if !allowed {
        return (0, 0, 0, 0);
    }
    let interintra = read_symbol(dec, interintra_cdf, 2);
    let (mut mode, mut use_wedge, mut widx) = (0, 0, 0);
    if interintra != 0 {
        mode = read_symbol(dec, interintra_mode_cdf, INTERINTRA_MODES);
        if wedge_used {
            use_wedge = read_symbol(dec, wedge_interintra_cdf, 2);
            if use_wedge != 0 {
                widx = read_symbol(dec, wedge_idx_cdf, MAX_WEDGE_TYPES);
            }
        }
    }
    (interintra, mode, use_wedge, widx)
}

/// `read_compound_type_info` — inverse of [`write_compound_type_info`]: the compound
/// group idx, then either the dist-wtd compound idx (average group) or the masked
/// compound type + wedge index/sign or diffwtd mask type. Returns
/// `(comp_group_idx, compound_idx, comp_type, wedge_index, wedge_sign, mask_type)`.
/// Non-coded fields take their inferred values (compound_idx=1, COMPOUND_AVERAGE=0 /
/// COMPOUND_DIFFWTD=3).
#[allow(clippy::too_many_arguments)]
pub fn read_compound_type_info(
    dec: &mut OdEcDec,
    masked_compound_used: bool,
    comp_group_idx_cdf: &mut [u16],
    enable_dist_wtd_comp: bool,
    compound_index_cdf: &mut [u16],
    wedge_used: bool,
    compound_type_cdf: &mut [u16],
    wedge_idx_cdf: &mut [u16],
) -> (i32, i32, i32, i32, i32, i32) {
    let comp_group_idx = if masked_compound_used {
        read_symbol(dec, comp_group_idx_cdf, 2)
    } else {
        0
    };
    let (mut compound_idx, mut comp_type) = (1, 0); // COMPOUND_AVERAGE
    let (mut wedge_index, mut wedge_sign, mut mask_type) = (0, 0, 0);
    if comp_group_idx == 0 {
        if enable_dist_wtd_comp {
            compound_idx = read_symbol(dec, compound_index_cdf, 2);
        }
    } else {
        comp_type = if wedge_used {
            read_symbol(dec, compound_type_cdf, MASKED_COMPOUND_TYPES) + COMPOUND_WEDGE
        } else {
            COMPOUND_WEDGE + 1 // COMPOUND_DIFFWTD
        };
        if comp_type == COMPOUND_WEDGE {
            wedge_index = read_symbol(dec, wedge_idx_cdf, MAX_WEDGE_TYPES);
            wedge_sign = read_bit(dec);
        } else {
            mask_type = read_literal(dec, MAX_DIFFWTD_MASK_BITS);
        }
    }
    (comp_group_idx, compound_idx, comp_type, wedge_index, wedge_sign, mask_type)
}

/// `read_palette_mode_info_flags` — inverse of [`write_palette_mode_info_flags`]: the
/// Y (DC-pred) and UV (chroma-ref DC-pred) palette on/off flags + sizes. Returns
/// `(n_y, n_uv)` (0 when the plane has no palette).
#[allow(clippy::too_many_arguments)]
pub fn read_palette_mode_info_flags(
    dec: &mut OdEcDec,
    mode_is_dc_pred: bool,
    palette_y_mode_cdf: &mut [u16],
    palette_y_size_cdf: &mut [u16],
    uv_dc_pred: bool,
    palette_uv_mode_cdf: &mut [u16],
    palette_uv_size_cdf: &mut [u16],
) -> (i32, i32) {
    let n_y = if mode_is_dc_pred && read_symbol(dec, palette_y_mode_cdf, 2) != 0 {
        read_symbol(dec, palette_y_size_cdf, PALETTE_SIZES) + PALETTE_MIN_SIZE
    } else {
        0
    };
    let n_uv = if uv_dc_pred && read_symbol(dec, palette_uv_mode_cdf, 2) != 0 {
        read_symbol(dec, palette_uv_size_cdf, PALETTE_SIZES) + PALETTE_MIN_SIZE
    } else {
        0
    };
    (n_y, n_uv)
}

/// `read_partition_node` — inverse of [`write_partition_node`] (the per-node partition
/// decode with frame-edge gating): when the block signals a partition (`bsize >=
/// BLOCK_8X8`) read it on the `partition_plane_context`-selected CDF (with the has-rows/
/// has-cols edge form), then update the neighbour partition context for the sub-blocks.
/// Returns the decoded partition (`PARTITION_NONE` for a non-signalling block).
#[allow(clippy::too_many_arguments)]
pub fn read_partition_node(
    dec: &mut OdEcDec,
    above: &mut [i8],
    left: &mut [i8],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
    partition_cdf_arena: &mut [[u16; 11]; 20],
) -> i32 {
    let mut partition = 0; // PARTITION_NONE
    if bsize >= BLOCK_8X8 {
        let hbs = MI_SIZE_WIDE[bsize] / 2;
        let has_rows = (mi_row + hbs) < mi_rows;
        let has_cols = (mi_col + hbs) < mi_cols;
        let ctx = partition_plane_context(above, left, mi_row as usize, mi_col as usize, bsize) as usize;
        partition = read_partition(
            dec, &mut partition_cdf_arena[ctx], partition_cdf_length(bsize), has_rows, has_cols, bsize,
        );
    }
    let subsize = get_partition_subsize(bsize, partition) as usize;
    update_ext_partition_context(above, left, mi_row, mi_col, subsize, bsize, partition);
    partition
}

#[allow(clippy::too_many_arguments)]
fn read_modes_sb_recurse(
    dec: &mut OdEcDec,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    out: &mut Vec<i8>,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) {
    if bsize < BLOCK_8X8 {
        return;
    }
    let hbs = MI_SIZE_WIDE[bsize] / 2;
    let ctx = partition_plane_context(above, left, mi_row as usize, mi_col as usize, bsize) as usize;
    let p = read_partition(dec, &mut arena[ctx], partition_cdf_length(bsize), true, true, bsize);
    out.push(p as i8);
    let subsize = get_partition_subsize(bsize, p) as usize;
    if p == PARTITION_SPLIT_MODE && bsize > BLOCK_8X8 {
        read_modes_sb_recurse(dec, above, left, arena, out, mi_row, mi_col, subsize);
        read_modes_sb_recurse(dec, above, left, arena, out, mi_row, mi_col + hbs, subsize);
        read_modes_sb_recurse(dec, above, left, arena, out, mi_row + hbs, mi_col, subsize);
        read_modes_sb_recurse(dec, above, left, arena, out, mi_row + hbs, mi_col + hbs, subsize);
    }
    update_ext_partition_context(above, left, mi_row, mi_col, subsize, bsize, p);
}

/// `read_modes_sb` — inverse of [`write_modes_sb`]: decode one superblock's partition
/// tree (pre-order), returning the reconstructed partition sequence.
pub fn read_modes_sb(
    dec: &mut OdEcDec,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> Vec<i8> {
    let mut out = Vec::new();
    read_modes_sb_recurse(dec, above, left, arena, &mut out, mi_row, mi_col, bsize);
    out
}

/// `read_modes_tile` — inverse of [`write_modes_tile`]: the tile superblock loop. Zeroes
/// the above partition context once (threads vertically across SB rows), zeroes the left
/// context at each SB row, and decodes each superblock's partition tree in row-major
/// order. Returns the concatenated pre-order partition sequence for the whole tile.
pub fn read_modes_tile(
    dec: &mut OdEcDec,
    above: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    n_sb_rows: i32,
    n_sb_cols: i32,
    sb_mi: i32,
    sb_size: usize,
) -> Vec<i8> {
    for a in above.iter_mut() {
        *a = 0; // av1_zero_above_context
    }
    let mut out = Vec::new();
    for r in 0..n_sb_rows {
        let mut left = [0i8; 32]; // av1_zero_left_context per SB row
        for c in 0..n_sb_cols {
            read_modes_sb_recurse(dec, above, &mut left, arena, &mut out, r * sb_mi, c * sb_mi, sb_size);
        }
    }
    out
}

/// `read_intra_y_and_angle_delta` — inverse of [`write_intra_y_and_angle_delta`]: the
/// nonkf luma intra mode, then the luma angle delta when the block uses angle deltas and
/// the mode is directional. Returns `(mode, angle_delta_y)`.
pub fn read_intra_y_and_angle_delta(
    dec: &mut OdEcDec,
    y_cdf: &mut [u16],
    bsize: usize,
    y_angle_cdf: &mut [u16],
) -> (i32, i32) {
    let mode = read_intra_y_mode(dec, y_cdf);
    let angle_delta_y = if use_angle_delta(bsize) && is_directional_mode(mode) {
        read_angle_delta(dec, y_angle_cdf)
    } else {
        0
    };
    (mode, angle_delta_y)
}

/// `read_intra_uv_and_angle_delta` — inverse of [`write_intra_uv_and_angle_delta`]: for a
/// chroma-reference non-monochrome block, the UV intra mode, then the CfL alphas (when
/// UV_CFL_PRED) and the UV angle delta (when the mapped UV mode is directional). Returns
/// `(uv_mode, cfl_alpha_idx, cfl_joint_sign, angle_delta_uv)`.
#[allow(clippy::too_many_arguments)]
pub fn read_intra_uv_and_angle_delta(
    dec: &mut OdEcDec,
    monochrome: bool,
    is_chroma_ref: bool,
    cfl_allowed: bool,
    bsize: usize,
    uv_mode_cdf: &mut [u16],
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
    uv_angle_cdf: &mut [u16],
) -> (i32, i32, i32, i32) {
    if !monochrome && is_chroma_ref {
        let uv_mode = read_intra_uv_mode(dec, uv_mode_cdf, cfl_allowed);
        let (cfl_joint_sign, cfl_alpha_idx) = if uv_mode == UV_CFL_PRED {
            read_cfl_alphas(dec, cfl_sign_cdf, cfl_alpha_cdf)
        } else {
            (0, 0)
        };
        let intra_mode = get_uv_mode(uv_mode as usize);
        let angle_delta_uv = if use_angle_delta(bsize) && is_directional_mode(intra_mode) {
            read_angle_delta(dec, uv_angle_cdf)
        } else {
            0
        };
        (uv_mode, cfl_alpha_idx, cfl_joint_sign, angle_delta_uv)
    } else {
        (0, 0, 0, 0)
    }
}

/// `read_palette_colors_plane` — inverse of [`write_palette_colors_plane`]: rebuild the
/// neighbour colour cache, read the per-cache-entry use bits (the cached palette colours),
/// delta-decode the remaining non-cached colours, and merge both sorted subsets back into
/// the ascending palette. `min_val` is 1 for luma, 0 for chroma-U.
#[allow(clippy::too_many_arguments)]
pub fn read_palette_colors_plane(
    dec: &mut OdEcDec,
    n: usize,
    plane: usize,
    bit_depth: i32,
    min_val: i32,
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_n_plane: i32,
    has_left: bool,
    left_colors: &[u16],
    left_n_plane: i32,
) -> Vec<u16> {
    let mut cache = [0u16; 16];
    let n_cache = get_palette_cache(
        &mut cache, plane, mb_to_top_edge, has_above, above_colors, above_n_plane, has_left,
        left_colors, left_n_plane,
    );
    let mut merged: Vec<u16> = Vec::with_capacity(n);
    for &cv in cache.iter().take(n_cache) {
        if merged.len() >= n {
            break;
        }
        if read_bit(dec) != 0 {
            merged.push(cv);
        }
    }
    let n_delta = n - merged.len();
    for &d in read_delta_palette_colors(dec, n_delta, bit_depth, min_val).iter() {
        merged.push(d as u16);
    }
    merged.sort_unstable();
    merged
}

/// `read_palette_mode_info` — inverse of [`write_palette_mode_info`]: the Y (DC-pred) and
/// UV (chroma-ref DC-pred) palette on/off flags + sizes + colours (Y/U cache-merged, V
/// raw/delta). Returns `(palette_size, palette_colors)` with `palette_colors` laid out
/// `[Y..PALETTE_MAX_SIZE, U.., V..]`.
#[allow(clippy::too_many_arguments)]
pub fn read_palette_mode_info(
    dec: &mut OdEcDec,
    mode_is_dc_pred: bool,
    uv_dc_pred: bool,
    bit_depth: i32,
    y_mode_cdf: &mut [u16],
    y_size_cdf: &mut [u16],
    uv_mode_cdf: &mut [u16],
    uv_size_cdf: &mut [u16],
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_size: [i32; 2],
    has_left: bool,
    left_colors: &[u16],
    left_size: [i32; 2],
) -> ([i32; 2], Vec<u16>) {
    let mut palette_size = [0i32; 2];
    let mut colors = vec![0u16; 3 * PALETTE_MAX_SIZE];
    if mode_is_dc_pred && read_symbol(dec, y_mode_cdf, 2) != 0 {
        let n = read_symbol(dec, y_size_cdf, PALETTE_SIZES) + PALETTE_MIN_SIZE;
        palette_size[0] = n;
        let yc = read_palette_colors_plane(
            dec, n as usize, 0, bit_depth, 1, mb_to_top_edge, has_above, above_colors,
            above_size[0], has_left, left_colors, left_size[0],
        );
        colors[..n as usize].copy_from_slice(&yc);
    }
    if uv_dc_pred && read_symbol(dec, uv_mode_cdf, 2) != 0 {
        let n = read_symbol(dec, uv_size_cdf, PALETTE_SIZES) + PALETTE_MIN_SIZE;
        palette_size[1] = n;
        let uc = read_palette_colors_plane(
            dec, n as usize, 1, bit_depth, 0, mb_to_top_edge, has_above, above_colors,
            above_size[1], has_left, left_colors, left_size[1],
        );
        colors[PALETTE_MAX_SIZE..PALETTE_MAX_SIZE + n as usize].copy_from_slice(&uc);
        let vc = read_palette_colors_v(dec, n as usize, bit_depth);
        let vbase = 2 * PALETTE_MAX_SIZE;
        colors[vbase..vbase + n as usize].copy_from_slice(&vc);
    }
    (palette_size, colors)
}

/// `read_intra_prediction_modes` — inverse of [`write_intra_prediction_modes`]: the full
/// per-block intra mode-info fragment. Composes read_intra_y_and_angle_delta +
/// read_intra_uv_and_angle_delta + read_palette_mode_info (when palette is allowed) +
/// read_filter_intra_mode_info. Returns
/// `(y_mode, angle_y, uv_mode, cfl_idx, cfl_sign, angle_uv, palette_size, palette_colors,
/// use_filter_intra, filter_intra_mode)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn read_intra_prediction_modes(
    dec: &mut OdEcDec,
    bsize: usize,
    y_cdf: &mut [u16],
    y_angle_cdf: &mut [u16],
    monochrome: bool,
    is_chroma_ref: bool,
    cfl_allowed: bool,
    uv_mode_cdf: &mut [u16],
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
    uv_angle_cdf: &mut [u16],
    allow_palette: bool,
    bit_depth: i32,
    pal_y_mode_cdf: &mut [u16],
    pal_y_size_cdf: &mut [u16],
    pal_uv_mode_cdf: &mut [u16],
    pal_uv_size_cdf: &mut [u16],
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_size: [i32; 2],
    has_left: bool,
    left_colors: &[u16],
    left_size: [i32; 2],
    filter_allowed: bool,
    fi_use_cdf: &mut [u16],
    fi_mode_cdf: &mut [u16],
) -> (i32, i32, i32, i32, i32, i32, [i32; 2], Vec<u16>, i32, i32) {
    let (mode, angle_y) = read_intra_y_and_angle_delta(dec, y_cdf, bsize, y_angle_cdf);
    let (uv_mode, cfl_idx, cfl_sign, angle_uv) = read_intra_uv_and_angle_delta(
        dec, monochrome, is_chroma_ref, cfl_allowed, bsize, uv_mode_cdf, cfl_sign_cdf,
        cfl_alpha_cdf, uv_angle_cdf,
    );
    let (palette_size, palette_colors) = if allow_palette {
        let mode_is_dc_pred = mode == DC_PRED;
        let uv_dc_pred = !monochrome && uv_mode == UV_DC_PRED && is_chroma_ref;
        read_palette_mode_info(
            dec, mode_is_dc_pred, uv_dc_pred, bit_depth, pal_y_mode_cdf, pal_y_size_cdf,
            pal_uv_mode_cdf, pal_uv_size_cdf, mb_to_top_edge, has_above, above_colors, above_size,
            has_left, left_colors, left_size,
        )
    } else {
        ([0, 0], vec![0u16; 3 * PALETTE_MAX_SIZE])
    };
    let (use_fi, fi_mode) = read_filter_intra_mode_info(dec, fi_use_cdf, fi_mode_cdf, filter_allowed);
    (mode, angle_y, uv_mode, cfl_idx, cfl_sign, angle_uv, palette_size, palette_colors, use_fi, fi_mode)
}

/// Decoded KEY-frame block tail (`read_kf_tail` output): either an intrabc block (with
/// its block vector) or the intra prediction mode-info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KfTailResult {
    pub use_intrabc: i32,
    pub diff_row: i32,
    pub diff_col: i32,
    pub mode: i32,
    pub angle_delta_y: i32,
    pub uv_mode: i32,
    pub cfl_alpha_idx: i32,
    pub cfl_joint_sign: i32,
    pub angle_delta_uv: i32,
    pub palette_size: [i32; 2],
    pub palette_colors: Vec<u16>,
    pub use_filter_intra: i32,
    pub filter_intra_mode: i32,
}

/// `read_kf_tail` — inverse of [`write_kf_tail`]: the intrabc flag + block vector (an
/// intrabc block returns immediately), otherwise the full intra prediction mode-info.
/// Composes [`read_intrabc_info`] + [`read_intra_prediction_modes`].
#[allow(clippy::too_many_arguments)]
pub fn read_kf_tail(
    dec: &mut OdEcDec,
    allow_intrabc: bool,
    intrabc_cdf: &mut [u16],
    ndvc_joints: &mut [u16],
    ndvc_comp0: &mut [u16; 69],
    ndvc_comp1: &mut [u16; 69],
    bsize: usize,
    y_cdf: &mut [u16],
    y_angle_cdf: &mut [u16],
    monochrome: bool,
    is_chroma_ref: bool,
    cfl_allowed: bool,
    uv_mode_cdf: &mut [u16],
    cfl_sign_cdf: &mut [u16],
    cfl_alpha_cdf: &mut [[u16; 17]; 6],
    uv_angle_cdf: &mut [u16],
    allow_palette: bool,
    bit_depth: i32,
    pal_y_mode_cdf: &mut [u16],
    pal_y_size_cdf: &mut [u16],
    pal_uv_mode_cdf: &mut [u16],
    pal_uv_size_cdf: &mut [u16],
    mb_to_top_edge: i32,
    has_above: bool,
    above_colors: &[u16],
    above_size: [i32; 2],
    has_left: bool,
    left_colors: &[u16],
    left_size: [i32; 2],
    filter_allowed: bool,
    fi_use_cdf: &mut [u16],
    fi_mode_cdf: &mut [u16],
) -> KfTailResult {
    if allow_intrabc {
        let (use_intrabc, dr, dc) =
            read_intrabc_info(dec, intrabc_cdf, ndvc_joints, ndvc_comp0, ndvc_comp1);
        if use_intrabc != 0 {
            return KfTailResult {
                use_intrabc,
                diff_row: dr,
                diff_col: dc,
                mode: 0,
                angle_delta_y: 0,
                uv_mode: 0,
                cfl_alpha_idx: 0,
                cfl_joint_sign: 0,
                angle_delta_uv: 0,
                palette_size: [0, 0],
                palette_colors: vec![0u16; 3 * PALETTE_MAX_SIZE],
                use_filter_intra: 0,
                filter_intra_mode: 0,
            };
        }
    }
    let (mode, angle_y, uv_mode, cfl_idx, cfl_sign, angle_uv, palette_size, palette_colors, use_fi, fi_mode) =
        read_intra_prediction_modes(
            dec, bsize, y_cdf, y_angle_cdf, monochrome, is_chroma_ref, cfl_allowed, uv_mode_cdf,
            cfl_sign_cdf, cfl_alpha_cdf, uv_angle_cdf, allow_palette, bit_depth, pal_y_mode_cdf,
            pal_y_size_cdf, pal_uv_mode_cdf, pal_uv_size_cdf, mb_to_top_edge, has_above, above_colors,
            above_size, has_left, left_colors, left_size, filter_allowed, fi_use_cdf, fi_mode_cdf,
        );
    KfTailResult {
        use_intrabc: 0,
        diff_row: 0,
        diff_col: 0,
        mode,
        angle_delta_y: angle_y,
        uv_mode,
        cfl_alpha_idx: cfl_idx,
        cfl_joint_sign: cfl_sign,
        angle_delta_uv: angle_uv,
        palette_size,
        palette_colors,
        use_filter_intra: use_fi,
        filter_intra_mode: fi_mode,
    }
}

/// `MAXQ` (`av1/common/quant_common.h`) — the top of the qindex range.
const MAXQ: i32 = 255;
/// `MAX_LOOP_FILTER` (`av1/common/av1_loopfilter.h`).
const MAX_LOOP_FILTER: i32 = 63;

/// `read_delta_q_params_sb` — inverse of [`write_delta_q_params_sb`] (the per-SB delta-q /
/// delta-lf driver). At an SB-upper-left non-skip block it reads the reduced delta-qindex
/// (reconstructing `current_qindex = clamp(base + reduced * delta_q_res, 1, MAXQ)` — the
/// normative `[1, 255]` clamp of `read_delta_q_params`, av1/decoder/decodemv.c: "Clamp to
/// [1,MAXQ] to not interfere with lossless mode" — updating the base carry) and, when
/// delta-lf is present, the per-plane multi deltas or the single from-base delta (each
/// updating its carry, clamped to `[-MAX_LOOP_FILTER, MAX_LOOP_FILTER]` as in C). Returns
/// the (possibly unchanged) `current_qindex`.
///
/// The clamps only fire on streams a libaom encoder never emits (its write side asserts
/// `current_qindex > 0` and derives targets via `av1_adjust_q_from_delta_q_res`, so
/// `base + reduced * res` is always in range and the carries stay lockstep); they are
/// normative decoder behaviour for arbitrary conformant-syntax input.
#[allow(clippy::too_many_arguments)]
pub fn read_delta_q_params_sb(
    dec: &mut OdEcDec,
    delta_q_present: bool,
    delta_lf_present: bool,
    delta_lf_multi: bool,
    num_planes: i32,
    bsize: usize,
    sb_size: usize,
    skip: i32,
    super_block_upper_left: bool,
    current_base_qindex: &mut i32,
    delta_q_res: i32,
    xd_delta_lf: &mut [i32; FRAME_LF_COUNT],
    xd_delta_lf_from_base: &mut i32,
    delta_lf_res: i32,
    delta_q_cdf: &mut [u16],
    delta_lf_multi_cdf: &mut [[u16; 5]; FRAME_LF_COUNT],
    delta_lf_cdf: &mut [u16],
) -> i32 {
    if !delta_q_present {
        return *current_base_qindex;
    }
    if (bsize != sb_size || skip == 0) && super_block_upper_left {
        let reduced = read_delta_qindex(dec, delta_q_cdf);
        // "Normative: Clamp to [1,MAXQ] to not interfere with lossless mode"
        // (read_delta_q_params, av1/decoder/decodemv.c).
        let current_qindex = (*current_base_qindex + reduced * delta_q_res).clamp(1, MAXQ);
        *current_base_qindex = current_qindex;
        if delta_lf_present {
            if delta_lf_multi {
                let frame_lf_count = if num_planes > 1 { FRAME_LF_COUNT } else { FRAME_LF_COUNT - 2 };
                for lf_id in 0..frame_lf_count {
                    let r = read_delta_lflevel(dec, &mut delta_lf_multi_cdf[lf_id]);
                    xd_delta_lf[lf_id] = (xd_delta_lf[lf_id] + r * delta_lf_res)
                        .clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER);
                }
            } else {
                let r = read_delta_lflevel(dec, delta_lf_cdf);
                *xd_delta_lf_from_base = (*xd_delta_lf_from_base + r * delta_lf_res)
                    .clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER);
            }
        }
        return current_qindex;
    }
    *current_base_qindex
}

/// `read_mb_modes_kf_prefix` — inverse of [`write_mb_modes_kf_prefix`]: the KEY-frame
/// block prefix — segment id (pre- or post-skip), the skip flag, CDEF strength, and the
/// per-SB delta-q/lf. Returns `(skip, segment_id, cdef_strength, current_qindex)`;
/// `cdef_transmitted` + the delta carries are updated in place.
#[allow(clippy::too_many_arguments)]
pub fn read_mb_modes_kf_prefix(
    dec: &mut OdEcDec,
    segid_preskip: bool,
    seg_enabled: bool,
    update_map: bool,
    seg_pred: i32,
    last_active_segid: i32,
    seg_cdf: &mut [u16],
    seg_skip_active: bool,
    skip_cdf: &mut [u16],
    coded_lossless: bool,
    allow_intrabc: bool,
    mi_row: i32,
    mi_col: i32,
    mib_size: i32,
    sb_size: usize,
    cdef_transmitted: &mut [bool; 4],
    cdef_bits: u32,
    dq_present: bool,
    dlf_present: bool,
    dlf_multi: bool,
    num_planes: i32,
    bsize: usize,
    current_base_qindex: &mut i32,
    dq_res: i32,
    xd_delta_lf: &mut [i32; FRAME_LF_COUNT],
    xd_delta_lf_from_base: &mut i32,
    dlf_res: i32,
    delta_q_cdf: &mut [u16],
    delta_lf_multi_cdf: &mut [[u16; 5]; FRAME_LF_COUNT],
    delta_lf_cdf: &mut [u16],
) -> (i32, i32, i32, i32) {
    let seg_coded = seg_enabled && update_map;
    let mut segment_id = 0;
    if segid_preskip && seg_coded {
        segment_id = read_segment_id(dec, seg_cdf, seg_pred, last_active_segid);
    }
    let skip = read_skip(dec, skip_cdf, seg_skip_active);
    // post-skip segment id is gated on skip==0 (write_segment_id's skip_txfm=skip).
    if !segid_preskip && seg_coded && skip == 0 {
        segment_id = read_segment_id(dec, seg_cdf, seg_pred, last_active_segid);
    }
    let cdef_strength = read_cdef(
        dec, coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size, skip,
        cdef_transmitted, cdef_bits,
    );
    let super_block_upper_left = (mi_row & (mib_size - 1)) == 0 && (mi_col & (mib_size - 1)) == 0;
    let current_qindex = read_delta_q_params_sb(
        dec, dq_present, dlf_present, dlf_multi, num_planes, bsize, sb_size, skip,
        super_block_upper_left, current_base_qindex, dq_res, xd_delta_lf, xd_delta_lf_from_base,
        dlf_res, delta_q_cdf, delta_lf_multi_cdf, delta_lf_cdf,
    );
    (skip, segment_id, cdef_strength, current_qindex)
}

// ===================== KEY-frame block struct model =====================
// Bundles the ~70 flat args of the KF block driver into typed structs so
// read_modes_b / write_modes_b can dispatch through one interface. The CDFs are
// pre-selected per context by the caller (read_modes_b computes the neighbour
// contexts); this struct model owns the MB_MODE_INFO output + the block state carry.

/// A KEY-frame block's decoded/decided mode info (`MB_MODE_INFO`, intra subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbModeInfoKf {
    pub segment_id: i32,
    pub skip: i32,
    pub cdef_strength: i32,
    pub current_qindex: i32,
    /// Per-plane loop-filter deltas after this block (the updated carry).
    pub delta_lf: [i32; FRAME_LF_COUNT],
    pub delta_lf_from_base: i32,
    pub use_intrabc: i32,
    pub dv_row: i32,
    pub dv_col: i32,
    pub y_mode: i32,
    pub angle_delta_y: i32,
    pub uv_mode: i32,
    pub cfl_alpha_idx: i32,
    pub cfl_joint_sign: i32,
    pub angle_delta_uv: i32,
    pub palette_size: [i32; 2],
    pub use_filter_intra: i32,
    pub filter_intra_mode: i32,
}

/// The (context-pre-selected) CDFs a KEY-frame block touches. Adapted in place.
#[derive(Debug, Clone)]
pub struct KfCdfs {
    pub seg: [u16; 9],
    pub skip: [u16; 3],
    pub delta_q: [u16; 5],
    pub delta_lf_multi: [[u16; 5]; FRAME_LF_COUNT],
    pub delta_lf: [u16; 5],
    pub intrabc: [u16; 3],
    pub ndvc_joints: [u16; 5],
    pub ndvc_comp0: [u16; 69],
    pub ndvc_comp1: [u16; 69],
    pub y_mode: [u16; 14],
    pub y_angle: [u16; 8],
    pub uv_mode: [u16; 15],
    pub cfl_sign: [u16; 9],
    pub cfl_alpha: [[u16; 17]; 6],
    pub uv_angle: [u16; 8],
    pub pal_y_mode: [u16; 3],
    pub pal_y_size: [u16; 8],
    pub pal_uv_mode: [u16; 3],
    pub pal_uv_size: [u16; 8],
    pub fi_use: [u16; 3],
    pub fi_mode: [u16; 6],
}

/// Per-block + per-frame state for the KEY-frame driver: the coding gates/config
/// plus the mutable carries (cdef-transmitted mask, running base qindex, delta-lf).
#[derive(Debug, Clone)]
pub struct KfBlockState {
    // segmentation
    pub segid_preskip: bool,
    pub seg_enabled: bool,
    pub update_map: bool,
    pub seg_pred: i32,
    pub last_active_segid: i32,
    pub seg_skip_active: bool,
    // block position / sb
    pub mi_row: i32,
    pub mi_col: i32,
    pub mib_size: i32,
    pub sb_size: usize,
    pub bsize: usize,
    // cdef / delta gates
    pub coded_lossless: bool,
    pub allow_intrabc: bool,
    pub cdef_bits: u32,
    pub dq_present: bool,
    pub dlf_present: bool,
    pub dlf_multi: bool,
    pub num_planes: i32,
    pub dq_res: i32,
    pub dlf_res: i32,
    // intra / palette / filter gates
    pub monochrome: bool,
    pub is_chroma_ref: bool,
    pub cfl_allowed: bool,
    pub allow_palette: bool,
    pub bit_depth: i32,
    pub filter_allowed: bool,
    pub mb_to_top_edge: i32,
    pub has_above: bool,
    pub has_left: bool,
    // mutable carries
    pub cdef_transmitted: [bool; 4],
    pub current_base_qindex: i32,
    pub xd_delta_lf: [i32; FRAME_LF_COUNT],
    pub xd_delta_lf_from_base: i32,
}

/// `read_mb_modes_kf` — the KEY-frame per-block decode driver, bundling
/// [`read_mb_modes_kf_prefix`] + [`read_kf_tail`] over the [`KfCdfs`] + [`KfBlockState`]
/// structs (palette colours not modelled in this cut: `allow_palette` must be false).
/// Mutates the CDFs + the state carries in place; returns the decoded [`MbModeInfoKf`].
pub fn read_mb_modes_kf(dec: &mut OdEcDec, c: &mut KfCdfs, s: &mut KfBlockState) -> MbModeInfoKf {
    let (skip, segment_id, cdef_strength, current_qindex) = read_mb_modes_kf_prefix(
        dec, s.segid_preskip, s.seg_enabled, s.update_map, s.seg_pred, s.last_active_segid, &mut c.seg, s.seg_skip_active, &mut c.skip, s.coded_lossless, s.allow_intrabc, s.mi_row,
        s.mi_col, s.mib_size, s.sb_size, &mut s.cdef_transmitted, s.cdef_bits, s.dq_present,
        s.dlf_present, s.dlf_multi, s.num_planes, s.bsize, &mut s.current_base_qindex, s.dq_res,
        &mut s.xd_delta_lf, &mut s.xd_delta_lf_from_base, s.dlf_res, &mut c.delta_q,
        &mut c.delta_lf_multi, &mut c.delta_lf,
    );
    let tail = read_kf_tail(
        dec, s.allow_intrabc, &mut c.intrabc, &mut c.ndvc_joints, &mut c.ndvc_comp0,
        &mut c.ndvc_comp1, s.bsize, &mut c.y_mode, &mut c.y_angle, s.monochrome, s.is_chroma_ref,
        s.cfl_allowed, &mut c.uv_mode, &mut c.cfl_sign, &mut c.cfl_alpha, &mut c.uv_angle,
        s.allow_palette, s.bit_depth, &mut c.pal_y_mode, &mut c.pal_y_size, &mut c.pal_uv_mode,
        &mut c.pal_uv_size, s.mb_to_top_edge, s.has_above, &[], [0, 0], s.has_left, &[], [0, 0],
        s.filter_allowed, &mut c.fi_use, &mut c.fi_mode,
    );
    MbModeInfoKf {
        segment_id,
        skip,
        cdef_strength,
        current_qindex,
        delta_lf: s.xd_delta_lf,
        delta_lf_from_base: s.xd_delta_lf_from_base,
        use_intrabc: tail.use_intrabc,
        dv_row: tail.diff_row,
        dv_col: tail.diff_col,
        y_mode: tail.mode,
        angle_delta_y: tail.angle_delta_y,
        uv_mode: tail.uv_mode,
        cfl_alpha_idx: tail.cfl_alpha_idx,
        cfl_joint_sign: tail.cfl_joint_sign,
        angle_delta_uv: tail.angle_delta_uv,
        palette_size: tail.palette_size,
        use_filter_intra: tail.use_filter_intra,
        filter_intra_mode: tail.filter_intra_mode,
    }
}

/// `write_mb_modes_kf` — the encode-side counterpart to [`read_mb_modes_kf`]: encode a
/// KEY-frame block's [`MbModeInfoKf`] via the [`KfCdfs`] + [`KfBlockState`] structs.
/// `seg_skip_active`'s skip and the delta targets come from `info` / `state`.
pub fn write_mb_modes_kf(enc: &mut OdEcEnc, info: &MbModeInfoKf, c: &mut KfCdfs, s: &mut KfBlockState) {
    write_mb_modes_kf_prefix(
        enc, s.segid_preskip, s.seg_enabled, s.update_map, info.segment_id, s.seg_pred,
        s.last_active_segid, &mut c.seg, s.seg_skip_active, info.skip, &mut c.skip, s.coded_lossless,
        s.allow_intrabc, s.mi_row, s.mi_col, s.mib_size, s.sb_size, &mut s.cdef_transmitted,
        s.cdef_bits, info.cdef_strength, s.dq_present, s.dlf_present, s.dlf_multi, s.num_planes,
        s.bsize, info.current_qindex, &mut s.current_base_qindex, s.dq_res, &info.delta_lf,
        &mut s.xd_delta_lf, info.delta_lf_from_base, &mut s.xd_delta_lf_from_base, s.dlf_res,
        &mut c.delta_q, &mut c.delta_lf_multi, &mut c.delta_lf,
    );
    write_kf_tail(
        enc, s.allow_intrabc, &mut c.intrabc, &mut c.ndvc_joints, &mut c.ndvc_comp0,
        &mut c.ndvc_comp1, info.use_intrabc, info.dv_row, info.dv_col, info.y_mode, s.bsize,
        &mut c.y_mode, info.angle_delta_y, &mut c.y_angle, s.monochrome, s.is_chroma_ref,
        info.uv_mode, s.cfl_allowed, info.cfl_alpha_idx, info.cfl_joint_sign, info.angle_delta_uv,
        &mut c.uv_mode, &mut c.cfl_sign, &mut c.cfl_alpha, &mut c.uv_angle, s.allow_palette,
        s.bit_depth, info.palette_size, &[], s.mb_to_top_edge, s.has_above, &[], [0, 0], s.has_left,
        &[], [0, 0], &mut c.pal_y_mode, &mut c.pal_y_size, &mut c.pal_uv_mode, &mut c.pal_uv_size,
        s.filter_allowed, info.use_filter_intra, info.filter_intra_mode, &mut c.fi_use, &mut c.fi_mode,
    );
}

// ===================== FRAME_CONTEXT context selection (KEY-frame intra scope) ==========
// The struct driver above takes CONTEXT-PRE-SELECTED CDFs (one shared instance per
// symbol — the KfCdfs simplification). libaom's real model keeps full per-context CDF
// arrays in FRAME_CONTEXT (av1/common/entropymode.h) and selects each symbol's instance
// from neighbour/block state at the moment the symbol is coded. KfFrameContext + the
// `_fc` entry points implement that model for every symbol the KEY-frame intra path
// codes, with the selection rules of read_intra_frame_mode_info (av1/decoder/decodemv.c)
// / write_intra_prediction_modes (av1/encoder/bitstream.c).

/// `KF_MODE_CONTEXTS` (entropymode.h).
const KF_MODE_CONTEXTS: usize = 5;
/// `DIRECTIONAL_MODES` (enums.h).
const DIRECTIONAL_MODES: usize = 8;
/// `SKIP_CONTEXTS` (enums.h).
const SKIP_CONTEXTS: usize = 3;
/// `SPATIAL_PREDICTION_PROBS` (seg_common.h).
const SPATIAL_PREDICTION_PROBS: usize = 3;
/// `PALATTE_BSIZE_CTXS` (entropymode.h — libaom's spelling).
const PALATTE_BSIZE_CTXS: usize = 7;
/// `PALETTE_Y_MODE_CONTEXTS` / `PALETTE_UV_MODE_CONTEXTS` (entropymode.h).
const PALETTE_Y_MODE_CONTEXTS: usize = 3;
const PALETTE_UV_MODE_CONTEXTS: usize = 2;
/// `MAX_TX_CATS` (blockd.h) / `TX_SIZE_CONTEXTS` (enums.h).
const MAX_TX_CATS: usize = 4;
const TX_SIZE_CONTEXTS: usize = 3;
/// `EXT_TX_SIZES` (enums.h): square tx-size classes indexing the ext-tx CDFs.
const EXT_TX_SIZES: usize = 4;
/// `BLOCK_SIZES_ALL` (enums.h).
const BLOCK_SIZES_ALL: usize = 22;

/// The neighbour mode-info projection the KEY-frame context selection reads from
/// `xd->above_mbmi` / `xd->left_mbmi`: the neighbour block's Y prediction mode
/// (`kf_y_cdf` context + the smooth-mode edge-filter check) and its transform-skip
/// flag (`av1_get_skip_txfm_context`). `None` = neighbour unavailable (off tile).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MiNbrKf {
    pub y_mode: i32,
    pub skip_txfm: i32,
}

/// The KEY-frame-intra slice of libaom's `FRAME_CONTEXT` (av1/common/entropymode.h):
/// every CDF array that path codes with, at C's per-context dimensions. Each leaf CDF
/// is sized exactly `nsymbols + 1` (adaptation-count slot last). C pads several arrays
/// to `CDF_SIZE(TX_TYPES)`-style maxima; only the leading `nsymbols + 1` slots ever
/// participate, so exact sizing is behaviour-identical.
///
/// NOT modelled (not coded by the KEY-frame intra path): the inter-prediction CDFs
/// (newmv/zeromv/refmv/drl, compound/wedge/interintra/motion-mode/obmc, ref-frame
/// trees, `nmvc`, skip_mode, intra_inter, `inter_ext_tx_cdf`, switchable_interp,
/// txfm_partition), the non-keyframe `y_mode_cdf[BLOCK_SIZE_GROUPS]`, the
/// loop-restoration CDFs, the segmentation `tree_cdf`/`pred_cdf` (temporal prediction
/// is inter-only; the intra path codes on `spatial_pred_seg_cdf`), and the palette
/// colour-index CDFs (`palette_*_color_index_cdf` — colour-map tokens are coded in the
/// token path, which the KEY driver does not model).
#[derive(Clone, Debug)]
pub struct KfFrameContext {
    /// `kf_y_cdf[KF_MODE_CONTEXTS][KF_MODE_CONTEXTS]` (13 symbols) — selected by
    /// `(intra_mode_context[above_mode], intra_mode_context[left_mode])`
    /// ([`get_y_mode_ctx`]; an absent neighbour resolves to `DC_PRED`).
    pub kf_y: [[[u16; 14]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS],
    /// `uv_mode_cdf[CFL_ALLOWED_TYPES][INTRA_MODES]` — selected by
    /// `(cfl_allowed, y_mode)`; the `[1][..]` plane codes 14 symbols, `[0][..]`
    /// codes 13 (an instance never mixes alphabets, so its count slot is fixed).
    pub uv_mode: [[[u16; 15]; INTRA_MODES]; 2],
    /// `angle_delta_cdf[DIRECTIONAL_MODES]` (7 symbols) — selected by
    /// `mode - V_PRED`; ONE array shared by the Y and UV angle deltas.
    pub angle_delta: [[u16; 8]; DIRECTIONAL_MODES],
    /// `skip_txfm_cdfs[SKIP_CONTEXTS]` — selected by [`skip_txfm_context`]
    /// (above_skip + left_skip).
    pub skip: [[u16; 3]; SKIP_CONTEXTS],
    /// `seg.spatial_pred_seg_cdf[SPATIAL_PREDICTION_PROBS]` (8 symbols). Held at C
    /// dims; the `cdf_num` selection (`av1_get_spatial_seg_pred`) reads the
    /// segment-id map, which the KEY driver does not model — the `_fc` entry
    /// points require segmentation coding off.
    pub seg_spatial: [[u16; 9]; SPATIAL_PREDICTION_PROBS],
    /// `partition_cdf[PARTITION_CONTEXTS]` — ns-symbol per block-size level
    /// (4/8/10); selected by [`partition_plane_context`].
    pub partition: [[u16; 11]; 20],
    /// `palette_y_mode_cdf[PALATTE_BSIZE_CTXS][PALETTE_Y_MODE_CONTEXTS]`,
    /// `palette_uv_mode_cdf[PALETTE_UV_MODE_CONTEXTS]`,
    /// `palette_{y,uv}_size_cdf[PALATTE_BSIZE_CTXS]` (7 symbols). Held at C dims;
    /// the `_fc` entry points require palette off (the colour coder needs the
    /// neighbour palette colour caches, which are not modelled).
    pub palette_y_mode: [[[u16; 3]; PALETTE_Y_MODE_CONTEXTS]; PALATTE_BSIZE_CTXS],
    pub palette_uv_mode: [[u16; 3]; PALETTE_UV_MODE_CONTEXTS],
    pub palette_y_size: [[u16; 8]; PALATTE_BSIZE_CTXS],
    pub palette_uv_size: [[u16; 8]; PALATTE_BSIZE_CTXS],
    /// `filter_intra_cdfs[BLOCK_SIZES_ALL]` — selected by bsize.
    pub filter_intra: [[u16; 3]; BLOCK_SIZES_ALL],
    /// `filter_intra_mode_cdf` (5 symbols; single instance).
    pub filter_intra_mode: [u16; 6],
    /// `cfl_sign_cdf` (8 symbols) + `cfl_alpha_cdf[CFL_ALPHA_CONTEXTS]` (16 symbols;
    /// the per-plane context derives from the joint sign inside
    /// [`read_cfl_alphas`] / [`write_cfl_alphas`]).
    pub cfl_sign: [u16; 9],
    pub cfl_alpha: [[u16; 17]; 6],
    /// `delta_q_cdf` / `delta_lf_multi_cdf[FRAME_LF_COUNT]` / `delta_lf_cdf`
    /// (single instances, as in C).
    pub delta_q: [u16; 5],
    pub delta_lf_multi: [[u16; 5]; FRAME_LF_COUNT],
    pub delta_lf: [u16; 5],
    /// `intrabc_cdf` + `ndvc` (single instances).
    pub intrabc: [u16; 3],
    pub ndvc_joints: [u16; 5],
    pub ndvc_comp0: [u16; 69],
    pub ndvc_comp1: [u16; 69],
    /// `tx_size_cdf[MAX_TX_CATS][TX_SIZE_CONTEXTS]` (3 symbols). Held at C dims;
    /// `TX_MODE_LARGEST` codes no tx-size symbols, so the `get_tx_size_context`
    /// neighbour selection lands with the TX_MODE_SELECT layer.
    pub tx_size: [[[u16; 4]; TX_SIZE_CONTEXTS]; MAX_TX_CATS],
    /// `intra_ext_tx_cdf[eset][EXT_TX_SIZES][INTRA_MODES]`: eset 1 =
    /// `EXT_TX_SET_DTT4_IDTX_1DDCT` (7 symbols), eset 2 = `EXT_TX_SET_DTT4_IDTX`
    /// (5 symbols); eset 0 is DCT-only and codes nothing. Selected by
    /// `(square tx size, intra direction)` in the coefficient loop
    /// (`av1_read_tx_type`); the intra direction is `fimode_to_intradir[..]` for a
    /// filter-intra block, else the Y mode.
    pub ext_tx_1ddct: [[[u16; 8]; INTRA_MODES]; EXT_TX_SIZES],
    pub ext_tx_dtt4: [[[u16; 6]; INTRA_MODES]; EXT_TX_SIZES],
    /// The coefficient-CDF region (`txb_skip`/`eob*`/`dc_sign`/`coeff_base*`/
    /// `coeff_br`) as aom-txb's flat arena; its per-symbol context selection
    /// happens inside the coefficient reader/writer.
    pub coeff: Vec<u16>,
}

impl KfFrameContext {
    /// An all-zero instance with a `coeff_len`-slot coefficient arena (aom-txb's
    /// `CDF_ARENA_LEN`). Zero CDFs are not valid to code with — fill every region
    /// before use (harnesses fill random-valid CDFs; libaom-default-table init is
    /// a separate layer, not yet ported).
    pub fn zeroed(coeff_len: usize) -> Self {
        KfFrameContext {
            kf_y: [[[0; 14]; KF_MODE_CONTEXTS]; KF_MODE_CONTEXTS],
            uv_mode: [[[0; 15]; INTRA_MODES]; 2],
            angle_delta: [[0; 8]; DIRECTIONAL_MODES],
            skip: [[0; 3]; SKIP_CONTEXTS],
            seg_spatial: [[0; 9]; SPATIAL_PREDICTION_PROBS],
            partition: [[0; 11]; 20],
            palette_y_mode: [[[0; 3]; PALETTE_Y_MODE_CONTEXTS]; PALATTE_BSIZE_CTXS],
            palette_uv_mode: [[0; 3]; PALETTE_UV_MODE_CONTEXTS],
            palette_y_size: [[0; 8]; PALATTE_BSIZE_CTXS],
            palette_uv_size: [[0; 8]; PALATTE_BSIZE_CTXS],
            filter_intra: [[0; 3]; BLOCK_SIZES_ALL],
            filter_intra_mode: [0; 6],
            cfl_sign: [0; 9],
            cfl_alpha: [[0; 17]; 6],
            delta_q: [0; 5],
            delta_lf_multi: [[0; 5]; FRAME_LF_COUNT],
            delta_lf: [0; 5],
            intrabc: [0; 3],
            ndvc_joints: [0; 5],
            ndvc_comp0: [0; 69],
            ndvc_comp1: [0; 69],
            tx_size: [[[0; 4]; TX_SIZE_CONTEXTS]; MAX_TX_CATS],
            ext_tx_1ddct: [[[0; 8]; INTRA_MODES]; EXT_TX_SIZES],
            ext_tx_dtt4: [[[0; 6]; INTRA_MODES]; EXT_TX_SIZES],
            coeff: vec![0; coeff_len],
        }
    }
}

/// `av1_filter_intra_allowed` (av1/common/blockd.h), in full: DC mode, no Y palette,
/// and `av1_filter_intra_allowed_bsize` (the sequence `enable_filter_intra` flag +
/// block within 32x32).
pub fn filter_intra_allowed(
    enable_filter_intra: bool,
    bsize: usize,
    y_mode: i32,
    palette_size_y: i32,
) -> bool {
    enable_filter_intra
        && y_mode == DC_PRED
        && palette_size_y == 0
        && BLOCK_SIZE_WIDE[bsize] <= 32
        && BLOCK_SIZE_HIGH[bsize] <= 32
}

/// Shared `_fc` gate checks: the contexts this cut cannot select yet must be off.
fn assert_fc_scope(s: &KfBlockState) {
    assert!(
        !(s.seg_enabled && s.update_map),
        "segment-id spatial context selection (av1_get_spatial_seg_pred) needs the segment-id map — not modelled"
    );
    assert!(
        !s.allow_palette,
        "palette coding needs the neighbour palette colour caches — not modelled"
    );
    assert!(
        !s.filter_allowed,
        "the filter-intra gate is internal to the _fc driver — pass enable_filter_intra instead"
    );
}

/// [`read_mb_modes_kf`] with libaom's FRAME_CONTEXT context selection: every symbol's
/// CDF instance is picked from the [`KfFrameContext`] arrays at the moment it is read,
/// exactly as `read_intra_frame_mode_info` (av1/decoder/decodemv.c) selects from
/// `xd->tile_ctx`:
///
/// - skip on `skip[above_skip + left_skip]`;
/// - Y mode on `kf_y[intra_mode_context[above_mode]][intra_mode_context[left_mode]]`;
/// - Y angle delta on `angle_delta[y_mode - V_PRED]`;
/// - UV mode on `uv_mode[cfl_allowed][y_mode]`, UV angle delta on
///   `angle_delta[get_uv_mode(uv_mode) - V_PRED]` (the same array as Y);
/// - the filter-intra flag on `filter_intra[bsize]`, gated by the full
///   [`filter_intra_allowed`] (the mode-dependent gate needs no follow-up call —
///   pass the sequence flag as `enable_filter_intra`; `s.filter_allowed` must be
///   `false`);
/// - the single-instance CDFs (delta-q/lf, intrabc + ndvc, cfl sign/alpha,
///   filter-intra mode) as in C.
///
/// Segmentation coding and palette must be off (their selection needs the
/// segment-id map / neighbour palette colours — see [`KfFrameContext`]).
pub fn read_mb_modes_kf_fc(
    dec: &mut OdEcDec,
    f: &mut KfFrameContext,
    s: &mut KfBlockState,
    enable_filter_intra: bool,
    above: Option<MiNbrKf>,
    left: Option<MiNbrKf>,
) -> MbModeInfoKf {
    assert_fc_scope(s);
    let skip_ctx = skip_txfm_context(
        above.map_or(0, |m| m.skip_txfm),
        left.map_or(0, |m| m.skip_txfm),
    ) as usize;
    let (skip, segment_id, cdef_strength, current_qindex) = read_mb_modes_kf_prefix(
        dec, s.segid_preskip, s.seg_enabled, s.update_map, s.seg_pred, s.last_active_segid,
        &mut f.seg_spatial[0], s.seg_skip_active, &mut f.skip[skip_ctx], s.coded_lossless,
        s.allow_intrabc, s.mi_row, s.mi_col, s.mib_size, s.sb_size, &mut s.cdef_transmitted,
        s.cdef_bits, s.dq_present, s.dlf_present, s.dlf_multi, s.num_planes, s.bsize,
        &mut s.current_base_qindex, s.dq_res, &mut s.xd_delta_lf, &mut s.xd_delta_lf_from_base,
        s.dlf_res, &mut f.delta_q, &mut f.delta_lf_multi, &mut f.delta_lf,
    );
    let mut info = MbModeInfoKf {
        segment_id,
        skip,
        cdef_strength,
        current_qindex,
        delta_lf: s.xd_delta_lf,
        delta_lf_from_base: s.xd_delta_lf_from_base,
        use_intrabc: 0,
        dv_row: 0,
        dv_col: 0,
        y_mode: 0,
        angle_delta_y: 0,
        uv_mode: 0,
        cfl_alpha_idx: 0,
        cfl_joint_sign: 0,
        angle_delta_uv: 0,
        palette_size: [0, 0],
        use_filter_intra: 0,
        filter_intra_mode: 0,
    };
    if s.allow_intrabc {
        let (use_intrabc, dr, dc) = read_intrabc_info(
            dec, &mut f.intrabc, &mut f.ndvc_joints, &mut f.ndvc_comp0, &mut f.ndvc_comp1,
        );
        if use_intrabc != 0 {
            info.use_intrabc = use_intrabc;
            info.dv_row = dr;
            info.dv_col = dc;
            return info;
        }
    }
    let (ac, lc) = get_y_mode_ctx(above.map(|m| m.y_mode), left.map(|m| m.y_mode));
    let mode = read_intra_y_mode(dec, &mut f.kf_y[ac][lc]);
    info.y_mode = mode;
    if use_angle_delta(s.bsize) && is_directional_mode(mode) {
        info.angle_delta_y = read_angle_delta(dec, &mut f.angle_delta[(mode - V_PRED) as usize]);
    }
    if !s.monochrome && s.is_chroma_ref {
        let uv = read_intra_uv_mode(
            dec,
            &mut f.uv_mode[s.cfl_allowed as usize][mode as usize],
            s.cfl_allowed,
        );
        info.uv_mode = uv;
        if uv == UV_CFL_PRED {
            let (joint_sign, idx) = read_cfl_alphas(dec, &mut f.cfl_sign, &mut f.cfl_alpha);
            info.cfl_alpha_idx = idx;
            info.cfl_joint_sign = joint_sign;
        }
        let intra_mode = get_uv_mode(uv as usize);
        if use_angle_delta(s.bsize) && is_directional_mode(intra_mode) {
            info.angle_delta_uv =
                read_angle_delta(dec, &mut f.angle_delta[(intra_mode - V_PRED) as usize]);
        }
    }
    // (Palette would be coded here; `allow_palette` is asserted off above.)
    let fi_allowed =
        filter_intra_allowed(enable_filter_intra, s.bsize, mode, info.palette_size[0]);
    let (use_fi, fi_mode) = read_filter_intra_mode_info(
        dec, &mut f.filter_intra[s.bsize], &mut f.filter_intra_mode, fi_allowed,
    );
    info.use_filter_intra = use_fi;
    info.filter_intra_mode = fi_mode;
    info
}

/// The encode-side counterpart to [`read_mb_modes_kf_fc`]: [`write_mb_modes_kf`] with
/// the same per-symbol FRAME_CONTEXT selection (`write_intra_prediction_modes`,
/// av1/encoder/bitstream.c, selects identically to the decoder). Same scope gates:
/// segmentation coding, palette, and `s.filter_allowed` must be off.
pub fn write_mb_modes_kf_fc(
    enc: &mut OdEcEnc,
    info: &MbModeInfoKf,
    f: &mut KfFrameContext,
    s: &mut KfBlockState,
    enable_filter_intra: bool,
    above: Option<MiNbrKf>,
    left: Option<MiNbrKf>,
) {
    assert_fc_scope(s);
    let skip_ctx = skip_txfm_context(
        above.map_or(0, |m| m.skip_txfm),
        left.map_or(0, |m| m.skip_txfm),
    ) as usize;
    write_mb_modes_kf_prefix(
        enc, s.segid_preskip, s.seg_enabled, s.update_map, info.segment_id, s.seg_pred,
        s.last_active_segid, &mut f.seg_spatial[0], s.seg_skip_active, info.skip,
        &mut f.skip[skip_ctx], s.coded_lossless, s.allow_intrabc, s.mi_row, s.mi_col, s.mib_size,
        s.sb_size, &mut s.cdef_transmitted, s.cdef_bits, info.cdef_strength, s.dq_present,
        s.dlf_present, s.dlf_multi, s.num_planes, s.bsize, info.current_qindex,
        &mut s.current_base_qindex, s.dq_res, &info.delta_lf, &mut s.xd_delta_lf,
        info.delta_lf_from_base, &mut s.xd_delta_lf_from_base, s.dlf_res, &mut f.delta_q,
        &mut f.delta_lf_multi, &mut f.delta_lf,
    );
    if s.allow_intrabc {
        write_intrabc_info(
            enc, &mut f.intrabc, &mut f.ndvc_joints, &mut f.ndvc_comp0, &mut f.ndvc_comp1,
            info.use_intrabc, info.dv_row, info.dv_col,
        );
        if info.use_intrabc != 0 {
            return;
        }
    }
    let (ac, lc) = get_y_mode_ctx(above.map(|m| m.y_mode), left.map(|m| m.y_mode));
    write_intra_y_mode_kf(enc, &mut f.kf_y[ac][lc], info.y_mode);
    if use_angle_delta(s.bsize) && is_directional_mode(info.y_mode) {
        write_angle_delta(
            enc,
            &mut f.angle_delta[(info.y_mode - V_PRED) as usize],
            info.angle_delta_y,
        );
    }
    if !s.monochrome && s.is_chroma_ref {
        write_intra_uv_mode(
            enc,
            &mut f.uv_mode[s.cfl_allowed as usize][info.y_mode as usize],
            info.uv_mode,
            s.cfl_allowed,
        );
        if info.uv_mode == UV_CFL_PRED {
            write_cfl_alphas(
                enc, &mut f.cfl_sign, &mut f.cfl_alpha, info.cfl_alpha_idx, info.cfl_joint_sign,
            );
        }
        let intra_mode = get_uv_mode(info.uv_mode as usize);
        if use_angle_delta(s.bsize) && is_directional_mode(intra_mode) {
            write_angle_delta(
                enc,
                &mut f.angle_delta[(intra_mode - V_PRED) as usize],
                info.angle_delta_uv,
            );
        }
    }
    // (Palette would be coded here; `allow_palette` is asserted off above.)
    let fi_allowed =
        filter_intra_allowed(enable_filter_intra, s.bsize, info.y_mode, info.palette_size[0]);
    write_filter_intra_mode_info(
        enc,
        &mut f.filter_intra[s.bsize],
        &mut f.filter_intra_mode,
        fi_allowed,
        info.use_filter_intra,
        info.filter_intra_mode,
    );
}

// ===================== tile-content dispatch (partition walk + block driver) =====================
// Walks a superblock's square (NONE/SPLIT) partition tree and dispatches the KEY-frame
// block driver at each PARTITION_NONE leaf. This is the first end-to-end tile-content
// path: partition structure + per-block mode-info over one coder. The block driver takes
// context-pre-selected CDFs (KfCdfs); per-neighbour mode-context selection is a later
// layer. Rectangular partitions (HORZ/VERT/AB/4) are handled by the full version to come.

#[allow(clippy::too_many_arguments)]
fn write_modes_b_recurse(
    enc: &mut OdEcEnc,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    cdfs: &mut KfCdfs,
    state: &mut KfBlockState,
    tree: &[i8],
    t_idx: &mut usize,
    infos: &[MbModeInfoKf],
    i_idx: &mut usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) {
    if bsize < BLOCK_8X8 {
        return;
    }
    let p = tree[*t_idx] as i32;
    *t_idx += 1;
    let subsize = get_partition_subsize(bsize, p) as usize;
    let hbs = MI_SIZE_WIDE[bsize] / 2;
    let ctx = partition_plane_context(above, left, mi_row as usize, mi_col as usize, bsize) as usize;
    write_partition(enc, &mut arena[ctx], partition_cdf_length(bsize), p, true, true, bsize);
    if p == PARTITION_SPLIT_MODE && bsize > BLOCK_8X8 {
        write_modes_b_recurse(enc, above, left, arena, cdfs, state, tree, t_idx, infos, i_idx, mi_row, mi_col, subsize);
        write_modes_b_recurse(enc, above, left, arena, cdfs, state, tree, t_idx, infos, i_idx, mi_row, mi_col + hbs, subsize);
        write_modes_b_recurse(enc, above, left, arena, cdfs, state, tree, t_idx, infos, i_idx, mi_row + hbs, mi_col, subsize);
        write_modes_b_recurse(enc, above, left, arena, cdfs, state, tree, t_idx, infos, i_idx, mi_row + hbs, mi_col + hbs, subsize);
    } else {
        // PARTITION_NONE leaf: one block of `subsize`.
        state.mi_row = mi_row;
        state.mi_col = mi_col;
        state.bsize = subsize;
        write_mb_modes_kf(enc, &infos[*i_idx], cdfs, state);
        *i_idx += 1;
    }
    update_ext_partition_context(above, left, mi_row, mi_col, subsize, bsize, p);
}

/// `write_modes_b` (square partitions) — encode one superblock's partition tree +
/// per-leaf KEY-frame block mode-info. `infos` supplies one [`MbModeInfoKf`] per
/// PARTITION_NONE leaf in pre-order. Returns the number of blocks written.
#[allow(clippy::too_many_arguments)]
pub fn write_modes_b(
    enc: &mut OdEcEnc,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    cdfs: &mut KfCdfs,
    state: &mut KfBlockState,
    tree: &[i8],
    infos: &[MbModeInfoKf],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> usize {
    let mut t_idx = 0;
    let mut i_idx = 0;
    write_modes_b_recurse(enc, above, left, arena, cdfs, state, tree, &mut t_idx, infos, &mut i_idx, mi_row, mi_col, bsize);
    i_idx
}

#[allow(clippy::too_many_arguments)]
fn read_modes_b_recurse(
    dec: &mut OdEcDec,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    cdfs: &mut KfCdfs,
    state: &mut KfBlockState,
    tree: &mut Vec<i8>,
    infos: &mut Vec<MbModeInfoKf>,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) {
    if bsize < BLOCK_8X8 {
        return;
    }
    let hbs = MI_SIZE_WIDE[bsize] / 2;
    let ctx = partition_plane_context(above, left, mi_row as usize, mi_col as usize, bsize) as usize;
    let p = read_partition(dec, &mut arena[ctx], partition_cdf_length(bsize), true, true, bsize);
    tree.push(p as i8);
    let subsize = get_partition_subsize(bsize, p) as usize;
    if p == PARTITION_SPLIT_MODE && bsize > BLOCK_8X8 {
        read_modes_b_recurse(dec, above, left, arena, cdfs, state, tree, infos, mi_row, mi_col, subsize);
        read_modes_b_recurse(dec, above, left, arena, cdfs, state, tree, infos, mi_row, mi_col + hbs, subsize);
        read_modes_b_recurse(dec, above, left, arena, cdfs, state, tree, infos, mi_row + hbs, mi_col, subsize);
        read_modes_b_recurse(dec, above, left, arena, cdfs, state, tree, infos, mi_row + hbs, mi_col + hbs, subsize);
    } else {
        state.mi_row = mi_row;
        state.mi_col = mi_col;
        state.bsize = subsize;
        infos.push(read_mb_modes_kf(dec, cdfs, state));
    }
    update_ext_partition_context(above, left, mi_row, mi_col, subsize, bsize, p);
}

/// `read_modes_b` (square partitions) — inverse of [`write_modes_b`]: decode one
/// superblock's partition tree, dispatching [`read_mb_modes_kf`] at each PARTITION_NONE
/// leaf. Returns `(partition_tree, per_leaf_mode_info)`.
#[allow(clippy::too_many_arguments)]
pub fn read_modes_b(
    dec: &mut OdEcDec,
    above: &mut [i8],
    left: &mut [i8],
    arena: &mut [[u16; 11]; 20],
    cdfs: &mut KfCdfs,
    state: &mut KfBlockState,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
) -> (Vec<i8>, Vec<MbModeInfoKf>) {
    let mut tree = Vec::new();
    let mut infos = Vec::new();
    read_modes_b_recurse(dec, above, left, arena, cdfs, state, &mut tree, &mut infos, mi_row, mi_col, bsize);
    (tree, infos)
}

// ---- intra neighbour availability (has_top_right / has_bottom_left) ----------
// Port of av1/common/reconintra.c: are the top-right / bottom-left reference
// pixels available (coded before this block)? Table-driven by coding order.

/// `mi_size_high_log2[BLOCK_SIZES_ALL]` (`common_data.h`): log2 of a block's
/// height in mode-info (4x4) units.
const MI_SIZE_HIGH_LOG2: [u8; 22] =
    [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2];
/// `MAX_MIB_SIZE_LOG2` = `MAX_SB_SIZE_LOG2 - MI_SIZE_LOG2` = 7 - 2 = 5.
const MAX_MIB_SIZE_LOG2: i32 = 5;
/// `BLOCK_64X64` block-size index.
const BLOCK_64X64: usize = 12;

#[rustfmt::skip]
const HAS_TR_4X4: [u8; 128] = [
    255, 255, 255, 255, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
    127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
    255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
    127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
    255, 255, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
    127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
    255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
    127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
];
#[rustfmt::skip]
const HAS_TR_4X8: [u8; 64] = [
    255, 255, 255, 255, 119, 119, 119, 119, 127, 127, 127, 127, 119,
    119, 119, 119, 255, 127, 255, 127, 119, 119, 119, 119, 127, 127,
    127, 127, 119, 119, 119, 119, 255, 255, 255, 127, 119, 119, 119,
    119, 127, 127, 127, 127, 119, 119, 119, 119, 255, 127, 255, 127,
    119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119,
];
#[rustfmt::skip]
const HAS_TR_8X4: [u8; 64] = [
    255, 255, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
    127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
    255, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
    127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
];
#[rustfmt::skip]
const HAS_TR_8X8: [u8; 32] = [
    255, 255, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
    255, 127, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
];
#[rustfmt::skip]
const HAS_TR_8X16: [u8; 16] = [
    255, 255, 119, 119, 127, 127, 119, 119, 255, 127, 119, 119, 127, 127, 119, 119,
];
#[rustfmt::skip]
const HAS_TR_16X8: [u8; 16] = [255, 0, 85, 0, 119, 0, 85, 0, 127, 0, 85, 0, 119, 0, 85, 0];
#[rustfmt::skip]
const HAS_TR_16X16: [u8; 8] = [255, 85, 119, 85, 127, 85, 119, 85];
const HAS_TR_16X32: [u8; 4] = [255, 119, 127, 119];
const HAS_TR_32X16: [u8; 4] = [15, 5, 7, 5];
const HAS_TR_32X32: [u8; 2] = [95, 87];
const HAS_TR_32X64: [u8; 1] = [127];
const HAS_TR_64X32: [u8; 1] = [19];
const HAS_TR_64X64: [u8; 1] = [7];
const HAS_TR_64X128: [u8; 1] = [3];
const HAS_TR_128X64: [u8; 1] = [1];
const HAS_TR_128X128: [u8; 1] = [1];
#[rustfmt::skip]
const HAS_TR_4X16: [u8; 32] = [
    255, 255, 255, 255, 127, 127, 127, 127, 255, 127, 255,
    127, 127, 127, 127, 127, 255, 255, 255, 127, 127, 127,
    127, 127, 255, 127, 255, 127, 127, 127, 127, 127,
];
#[rustfmt::skip]
const HAS_TR_16X4: [u8; 32] = [
    255, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
    127, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
];
const HAS_TR_8X32: [u8; 8] = [255, 255, 127, 127, 255, 127, 127, 127];
const HAS_TR_32X8: [u8; 8] = [15, 0, 5, 0, 7, 0, 5, 0];
const HAS_TR_16X64: [u8; 2] = [255, 127];
const HAS_TR_64X16: [u8; 2] = [3, 1];

const HAS_TR_TABLES: [&[u8]; 22] = [
    &HAS_TR_4X4, &HAS_TR_4X8, &HAS_TR_8X4, &HAS_TR_8X8, &HAS_TR_8X16, &HAS_TR_16X8,
    &HAS_TR_16X16, &HAS_TR_16X32, &HAS_TR_32X16, &HAS_TR_32X32, &HAS_TR_32X64,
    &HAS_TR_64X32, &HAS_TR_64X64, &HAS_TR_64X128, &HAS_TR_128X64, &HAS_TR_128X128,
    &HAS_TR_4X16, &HAS_TR_16X4, &HAS_TR_8X32, &HAS_TR_32X8, &HAS_TR_16X64, &HAS_TR_64X16,
];
#[rustfmt::skip]
const HAS_TR_VERT_8X8: [u8; 32] = [
    255, 255, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
    255, 127, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
];
const HAS_TR_VERT_16X16: [u8; 8] = [255, 0, 119, 0, 127, 0, 119, 0];
const HAS_TR_VERT_32X32: [u8; 2] = [15, 7];
const HAS_TR_VERT_64X64: [u8; 1] = [3];
const HAS_TR_VERT_TABLES: [Option<&[u8]>; 16] = [
    None, Some(&HAS_TR_4X8), None, Some(&HAS_TR_VERT_8X8), Some(&HAS_TR_8X16), None,
    Some(&HAS_TR_VERT_16X16), Some(&HAS_TR_16X32), None, Some(&HAS_TR_VERT_32X32),
    Some(&HAS_TR_32X64), None, Some(&HAS_TR_VERT_64X64), Some(&HAS_TR_64X128), None,
    Some(&HAS_TR_128X128),
];

fn get_has_tr_table(partition: usize, bsize: usize) -> &'static [u8] {
    if partition == PARTITION_VERT_A || partition == PARTITION_VERT_B {
        HAS_TR_VERT_TABLES[bsize].expect("has_tr vert table is non-null for a VERT partition")
    } else {
        HAS_TR_TABLES[bsize]
    }
}

/// `has_top_right` (reconintra.c): are the block's top-right reference pixels
/// available (coded before this block)? `mi_row`/`mi_col` are the block's
/// mode-info position, `row_off`/`col_off` the transform-unit offset within the
/// block (in mi units), `txsz` the transform size, `ss_x`/`ss_y` the plane
/// subsampling. Returns 0 or 1.
#[allow(clippy::too_many_arguments)]
pub fn has_top_right(
    sb_size: usize, bsize: usize, mi_row: i32, mi_col: i32, top_available: bool,
    right_available: bool, partition: usize, txsz: usize, row_off: i32, col_off: i32,
    ss_x: i32, ss_y: i32,
) -> i32 {
    if !top_available || !right_available {
        return 0;
    }
    let bw_unit = MI_SIZE_WIDE[bsize];
    let plane_bw_unit = (bw_unit >> ss_x).max(1);
    let top_right_count_unit = TX_SIZE_WIDE_UNIT[txsz];
    if row_off > 0 {
        if BLOCK_SIZE_WIDE[bsize] > BLOCK_SIZE_WIDE[BLOCK_64X64] {
            if row_off == MI_SIZE_HIGH[BLOCK_64X64] >> ss_y
                && col_off + top_right_count_unit == MI_SIZE_WIDE[BLOCK_64X64] >> ss_x
            {
                return 1;
            }
            let plane_bw_unit_64 = MI_SIZE_WIDE[BLOCK_64X64] >> ss_x;
            let col_off_64 = col_off % plane_bw_unit_64;
            return (col_off_64 + top_right_count_unit < plane_bw_unit_64) as i32;
        }
        (col_off + top_right_count_unit < plane_bw_unit) as i32
    } else {
        if col_off + top_right_count_unit < plane_bw_unit {
            return 1;
        }
        let bw_in_mi_log2 = MI_SIZE_WIDE_LOG2[bsize] as i32;
        let bh_in_mi_log2 = MI_SIZE_HIGH_LOG2[bsize] as i32;
        let sb_mi_size = MI_SIZE_HIGH[sb_size];
        let blk_row_in_sb = (mi_row & (sb_mi_size - 1)) >> bh_in_mi_log2;
        let blk_col_in_sb = (mi_col & (sb_mi_size - 1)) >> bw_in_mi_log2;
        if blk_row_in_sb == 0 {
            return 1;
        }
        if ((blk_col_in_sb + 1) << bw_in_mi_log2) >= sb_mi_size {
            return 0;
        }
        let this_blk_index = (blk_row_in_sb << (MAX_MIB_SIZE_LOG2 - bw_in_mi_log2)) + blk_col_in_sb;
        let idx1 = (this_blk_index / 8) as usize;
        let idx2 = this_blk_index % 8;
        let table = get_has_tr_table(partition, bsize);
        ((table[idx1] >> idx2) & 1) as i32
    }
}

#[rustfmt::skip]
const HAS_BL_4X4: [u8; 128] = [
    84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1, 84, 85, 85,
    85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 1, 0, 84, 85, 85, 85, 16, 17,
    17, 17, 84, 85, 85, 85, 0, 1, 1, 1, 84, 85, 85, 85, 16, 17, 17, 17, 84,
    85, 85, 85, 0, 0, 0, 0, 84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85,
    0, 1, 1, 1, 84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 1,
    0, 84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1, 84, 85,
    85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 0, 0,
];
#[rustfmt::skip]
const HAS_BL_4X8: [u8; 64] = [
    16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0,
    16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 0, 0,
    16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0,
    16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 0, 0,
];
#[rustfmt::skip]
const HAS_BL_8X4: [u8; 64] = [
    254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 1,
    254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0,
    254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 1,
    254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0,
];
#[rustfmt::skip]
const HAS_BL_8X8: [u8; 32] = [
    84, 85, 16, 17, 84, 85, 0, 1, 84, 85, 16, 17, 84, 85, 0, 0,
    84, 85, 16, 17, 84, 85, 0, 1, 84, 85, 16, 17, 84, 85, 0, 0,
];
#[rustfmt::skip]
const HAS_BL_8X16: [u8; 16] = [16, 17, 0, 1, 16, 17, 0, 0, 16, 17, 0, 1, 16, 17, 0, 0];
#[rustfmt::skip]
const HAS_BL_16X8: [u8; 16] = [254, 84, 254, 16, 254, 84, 254, 0, 254, 84, 254, 16, 254, 84, 254, 0];
const HAS_BL_16X16: [u8; 8] = [84, 16, 84, 0, 84, 16, 84, 0];
const HAS_BL_16X32: [u8; 4] = [16, 0, 16, 0];
const HAS_BL_32X16: [u8; 4] = [78, 14, 78, 14];
const HAS_BL_32X32: [u8; 2] = [4, 4];
const HAS_BL_32X64: [u8; 1] = [0];
const HAS_BL_64X32: [u8; 1] = [34];
const HAS_BL_64X64: [u8; 1] = [0];
const HAS_BL_64X128: [u8; 1] = [0];
const HAS_BL_128X64: [u8; 1] = [0];
const HAS_BL_128X128: [u8; 1] = [0];
#[rustfmt::skip]
const HAS_BL_4X16: [u8; 32] = [
    0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
];
#[rustfmt::skip]
const HAS_BL_16X4: [u8; 32] = [
    254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
    254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
];
const HAS_BL_8X32: [u8; 8] = [0, 1, 0, 0, 0, 1, 0, 0];
const HAS_BL_32X8: [u8; 8] = [238, 78, 238, 14, 238, 78, 238, 14];
const HAS_BL_16X64: [u8; 2] = [0, 0];
const HAS_BL_64X16: [u8; 2] = [42, 42];
const HAS_BL_TABLES: [&[u8]; 22] = [
    &HAS_BL_4X4, &HAS_BL_4X8, &HAS_BL_8X4, &HAS_BL_8X8, &HAS_BL_8X16, &HAS_BL_16X8,
    &HAS_BL_16X16, &HAS_BL_16X32, &HAS_BL_32X16, &HAS_BL_32X32, &HAS_BL_32X64,
    &HAS_BL_64X32, &HAS_BL_64X64, &HAS_BL_64X128, &HAS_BL_128X64, &HAS_BL_128X128,
    &HAS_BL_4X16, &HAS_BL_16X4, &HAS_BL_8X32, &HAS_BL_32X8, &HAS_BL_16X64, &HAS_BL_64X16,
];
#[rustfmt::skip]
const HAS_BL_VERT_8X8: [u8; 32] = [
    254, 255, 16, 17, 254, 255, 0, 1, 254, 255, 16, 17, 254, 255, 0, 0,
    254, 255, 16, 17, 254, 255, 0, 1, 254, 255, 16, 17, 254, 255, 0, 0,
];
const HAS_BL_VERT_16X16: [u8; 8] = [254, 16, 254, 0, 254, 16, 254, 0];
const HAS_BL_VERT_32X32: [u8; 2] = [14, 14];
const HAS_BL_VERT_64X64: [u8; 1] = [2];
const HAS_BL_VERT_TABLES: [Option<&[u8]>; 16] = [
    None, Some(&HAS_BL_4X8), None, Some(&HAS_BL_VERT_8X8), Some(&HAS_BL_8X16), None,
    Some(&HAS_BL_VERT_16X16), Some(&HAS_BL_16X32), None, Some(&HAS_BL_VERT_32X32),
    Some(&HAS_BL_32X64), None, Some(&HAS_BL_VERT_64X64), Some(&HAS_BL_64X128), None,
    Some(&HAS_BL_128X128),
];
fn get_has_bl_table(partition: usize, bsize: usize) -> &'static [u8] {
    if partition == PARTITION_VERT_A || partition == PARTITION_VERT_B {
        HAS_BL_VERT_TABLES[bsize].expect("has_bl vert table is non-null for a VERT partition")
    } else {
        HAS_BL_TABLES[bsize]
    }
}

/// `has_bottom_left` (reconintra.c): are the block's bottom-left reference pixels
/// available (coded before this block)? Args mirror [`has_top_right`]. Returns 0/1.
#[allow(clippy::too_many_arguments)]
pub fn has_bottom_left(
    sb_size: usize, bsize: usize, mi_row: i32, mi_col: i32, bottom_available: bool,
    left_available: bool, partition: usize, txsz: usize, row_off: i32, col_off: i32,
    ss_x: i32, ss_y: i32,
) -> i32 {
    if !bottom_available || !left_available {
        return 0;
    }
    if BLOCK_SIZE_WIDE[bsize] > BLOCK_SIZE_WIDE[BLOCK_64X64] && col_off > 0 {
        let plane_bw_unit_64 = MI_SIZE_WIDE[BLOCK_64X64] >> ss_x;
        let col_off_64 = col_off % plane_bw_unit_64;
        if col_off_64 == 0 {
            let plane_bh_unit_64 = MI_SIZE_HIGH[BLOCK_64X64] >> ss_y;
            let row_off_64 = row_off % plane_bh_unit_64;
            let plane_bh_unit = (MI_SIZE_HIGH[bsize] >> ss_y).min(plane_bh_unit_64);
            return (row_off_64 + TX_SIZE_HIGH_UNIT[txsz] < plane_bh_unit) as i32;
        }
    }
    if col_off > 0 {
        0
    } else {
        let bh_unit = MI_SIZE_HIGH[bsize];
        let plane_bh_unit = (bh_unit >> ss_y).max(1);
        let bottom_left_count_unit = TX_SIZE_HIGH_UNIT[txsz];
        if row_off + bottom_left_count_unit < plane_bh_unit {
            return 1;
        }
        let bw_in_mi_log2 = MI_SIZE_WIDE_LOG2[bsize] as i32;
        let bh_in_mi_log2 = MI_SIZE_HIGH_LOG2[bsize] as i32;
        let sb_mi_size = MI_SIZE_HIGH[sb_size];
        let blk_row_in_sb = (mi_row & (sb_mi_size - 1)) >> bh_in_mi_log2;
        let blk_col_in_sb = (mi_col & (sb_mi_size - 1)) >> bw_in_mi_log2;
        if blk_col_in_sb == 0 {
            let blk_start_row_off = (blk_row_in_sb << bh_in_mi_log2) >> ss_y;
            let row_off_in_sb = blk_start_row_off + row_off;
            let sb_height_unit = sb_mi_size >> ss_y;
            return (row_off_in_sb + bottom_left_count_unit < sb_height_unit) as i32;
        }
        if ((blk_row_in_sb + 1) << bh_in_mi_log2) >= sb_mi_size {
            return 0;
        }
        let this_blk_index = (blk_row_in_sb << (MAX_MIB_SIZE_LOG2 - bw_in_mi_log2)) + blk_col_in_sb;
        let idx1 = (this_blk_index / 8) as usize;
        let idx2 = this_blk_index % 8;
        let table = get_has_bl_table(partition, bsize);
        ((table[idx1] >> idx2) & 1) as i32
    }
}

/// Intra neighbour-availability composition — the reference-pixel counts
/// `av1_predict_intra_block` (reconintra.c) computes before predicting: how many
/// above / above-right / left / below-left samples are available given the
/// block's grid position, the frame/tile bounds, and the mode. Composes
/// [`has_top_right`] / [`has_bottom_left`] with the have_top/have_left/xr/yd
/// geometry. Returns `(n_top_px, n_topright_px, n_left_px, n_bottomleft_px)` — the
/// exact inputs `aom_intra::predict_intra_high` needs. Shared by the decode
/// reconstruction driver and the encoder's intra RD search.
///
/// `up_available`/`left_available` are the plane's neighbour flags
/// (`mi_row > tile_row_start` / `mi_col > tile_col_start` for luma; caller supplies
/// the chroma-adjusted values). `wpx`/`hpx` are the plane block dimensions;
/// `row_off`/`col_off` the transform-unit offset (mi units) within the block;
/// `angle_delta` is pre-scaled by `ANGLE_STEP`.
#[allow(clippy::too_many_arguments)]
pub fn intra_avail(
    sb_size: usize, bsize: usize, mi_row: i32, mi_col: i32, up_available: bool,
    left_available: bool, tile_col_end: i32, tile_row_end: i32, partition: usize,
    tx_size: usize, ss_x: i32, ss_y: i32, row_off: i32, col_off: i32, wpx: i32,
    hpx: i32, mi_cols: i32, mi_rows: i32, mode: usize, angle_delta: i32,
    use_filter_intra: bool,
) -> (i32, i32, i32, i32) {
    const M2A: [i32; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];
    let txwpx = TX_SIZE_WIDE[tx_size];
    let txhpx = TX_SIZE_HIGH[tx_size];
    let txw = TX_SIZE_WIDE_UNIT[tx_size];
    let txh = TX_SIZE_HIGH_UNIT[tx_size];
    let bw_mi = MI_SIZE_WIDE[bsize];
    let bh_mi = MI_SIZE_HIGH[bsize];
    // mb_to_*_edge = GET_MV_SUBPEL((mi_dim - blk_mi - mi_pos) * MI_SIZE) = *4*8.
    let mb_to_right_edge = (mi_cols - bw_mi - mi_col) * 4 * 8;
    let mb_to_bottom_edge = (mi_rows - bh_mi - mi_row) * 4 * 8;
    let x = col_off << 2; // MI_SIZE_LOG2
    let y = row_off << 2;
    let have_top = row_off != 0 || up_available;
    let have_left = col_off != 0 || left_available;
    let xr = (mb_to_right_edge >> (3 + ss_x)) + wpx - x - txwpx;
    let yd = (mb_to_bottom_edge >> (3 + ss_y)) + hpx - y - txhpx;
    let n_top_px = if have_top { txwpx.min(xr + txwpx) } else { 0 };
    let n_left_px = if have_left { txhpx.min(yd + txhpx) } else { 0 };
    let right_available = mi_col + ((col_off + txw) << ss_x) < tile_col_end;
    let bottom_available = yd > 0 && mi_row + ((row_off + txh) << ss_y) < tile_row_end;
    let is_dr = (1..=8).contains(&mode);
    let p_angle = if is_dr { M2A[mode] + angle_delta } else { 0 };
    let need_tr = !use_filter_intra && is_dr && p_angle < 90;
    let need_bl = !use_filter_intra && is_dr && p_angle > 180;
    let have_tr = if need_tr {
        has_top_right(
            sb_size, bsize, mi_row, mi_col, have_top, right_available, partition, tx_size, row_off,
            col_off, ss_x, ss_y,
        )
    } else {
        -1
    };
    let have_bl = if need_bl {
        has_bottom_left(
            sb_size, bsize, mi_row, mi_col, bottom_available, have_left, partition, tx_size, row_off,
            col_off, ss_x, ss_y,
        )
    } else {
        -1
    };
    let n_topright_px = if have_tr > 0 { txwpx.min(xr) } else { have_tr };
    let n_bottomleft_px = if have_bl > 0 { txhpx.min(yd) } else { have_bl };
    (n_top_px, n_topright_px, n_left_px, n_bottomleft_px)
}
