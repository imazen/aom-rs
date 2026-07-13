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

use crate::cdf::{write_bit, write_literal};

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

fn have_nearmv_in_inter_mode(mode: i32) -> bool {
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
