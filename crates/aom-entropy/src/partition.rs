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
