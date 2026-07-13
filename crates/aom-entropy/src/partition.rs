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
