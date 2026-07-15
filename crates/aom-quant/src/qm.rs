//! Quantization-matrix (QM) selection for the *forward* quantizer — the
//! encode-side of `av1/common/quant_common.c`'s `av1_get_qmatrix` /
//! `av1_qm_init`.
//!
//! When a frame turns on `using_qmatrix` (via `--enable-qm` / `tune=IQ` /
//! `tune=SSIMULACRA2`; it is OFF by default in allintra), each 2-D-transform
//! coefficient is quantized with a per-position forward-QM weight `wt` instead
//! of the flat step: the `av1_quantize_*_qm` kernels fold `wt = qmatrix[pos]`
//! into the quant chain (`abs_coeff * wt * quant >> (16 - log_scale +
//! AOM_QM_BITS)`) and `iwt = iqmatrix[pos]` into the dequant. This module
//! supplies the forward `qmatrix` slice for a given (qm level, plane, tx size,
//! tx type); the inverse `iqmatrix` is selected identically from the decode-side
//! `iwt_matrix_ref` bases.
//!
//! The forward-QM bases `wt_matrix_ref[NUM_QM_LEVELS-1][2][QM_TOTAL_SIZE]` are
//! ported verbatim from libaom (generated into [`crate::qm_fwd_tables`]);
//! `av1_qm_init` packs each tx size's matrix contiguously in the 3344-entry
//! array in `TX_SIZES_ALL` order, reusing the `av1_get_adjusted_tx_size` matrix
//! for the 64-point sizes. The per-tx offsets [`QM_OFFSET`] reproduce that
//! packing — byte-identical to the decode-side `iqmatrix` selector (both come
//! from the same `av1_qm_init` loop, differing only in the source table).

use crate::qm_fwd_tables::WT_MATRIX_REF;

/// `NUM_QM_LEVELS` (`quant_common.h`): 16 QM sets. Level `NUM_QM_LEVELS - 1`
/// (15) is the flat/no-op matrix and is signalled as "no matrix" (`None`).
pub const NUM_QM_LEVELS: usize = 16;

/// `IDTX` (`enums.h`): the identity 2-D transform. `is_2d_transform(tx_type)`
/// is `tx_type < IDTX`; 1-D and identity transforms take the flat matrix.
const IDTX: usize = 9;

/// Byte offset into a `wt_matrix_ref[q][c]` row (`QM_TOTAL_SIZE` = 3344) of each
/// `TX_SIZE`'s matrix, indexed by the RAW tx size. Reproduces `av1_qm_init`'s
/// packing: iterate `TX_SIZES_ALL` in enum order, appending `tx_size_2d[t]`
/// bytes only for the canonical sizes (`t == av1_get_adjusted_tx_size(t)`); the
/// 64-point sizes reuse their adjusted (32-capped) matrix's offset. Identical to
/// the decode-side `QM_OFFSET` (same init loop). Accumulates to exactly 3344.
#[rustfmt::skip]
const QM_OFFSET: [usize; 19] = [
    0, 16, 80, 336, 336, 1360, 1392, 1424, 1552, 1680, 2192, 336, 336, 2704,
    2768, 2832, 3088, 1680, 2192,
];

/// Matrix length per RAW tx size == `tx_size_2d[av1_get_adjusted_tx_size(t)]`
/// (`txb_wide(t) * txb_high(t)`): the coded region the forward transform writes,
/// with 64-point sizes capped to their 32-point adjusted area. Inlined (rather
/// than depending on `aom-txb`) to keep this quant-kernel crate pure-Rust and
/// dependency-free; the differential test against C `av1_qm_init` validates
/// every entry, and [`QM_OFFSET`] + `QM_LEN` tile the 3344-entry row exactly.
#[rustfmt::skip]
const QM_LEN: [usize; 19] = [
    16, 64, 256, 1024, 1024, 32, 32, 128, 128, 512, 512, 1024, 1024, 64, 64,
    256, 256, 512, 512,
];

/// `av1_get_qmatrix` (`quant_common.c`): the forward quantization matrix for a
/// coefficient block, or `None` for the flat (no-weighting) case.
///
/// Returns `None` — flat quant, matching libaom's `gqmatrix[NUM_QM_LEVELS-1]`
/// NULL rows — when the level is the flat top level (QM off, or a lossless
/// segment) or the transform is 1-D / identity (`!is_2d_transform`). Otherwise
/// the `wt_matrix_ref` slice for `(qm_level, plane>=1 ? chroma : luma, adjusted
/// tx size)`, laid out in raster order over the adjusted transform block —
/// exactly what the `av1_quantize_*_qm` kernels index by coefficient position.
///
/// Mirrors the decode-side [`aom-decode`'s `iqmatrix`] byte-for-byte in
/// structure (same offsets, same None cases, same plane grouping); only the
/// source table differs (`wt_matrix_ref` vs `iwt_matrix_ref`).
pub fn qmatrix(
    qm_level: usize,
    plane: usize,
    tx_size: usize,
    tx_type: usize,
) -> Option<&'static [u8]> {
    // Flat matrix (i.e. no weighting) for the top level and for 1-D / Identity
    // transforms (av1_get_qmatrix: gqmatrix[NUM_QM_LEVELS-1][0][.] == NULL).
    if qm_level >= NUM_QM_LEVELS - 1 || tx_type >= IDTX {
        return None;
    }
    // Plane group: luma (0) vs chroma (both U and V share the c>=1 bases).
    let c = usize::from(plane >= 1);
    let off = QM_OFFSET[tx_size];
    // Length == tx_size_2d[av1_get_adjusted_tx_size(tx_size)] — the coded
    // region the forward transform writes (adjusted for the 64-point sizes).
    let len = QM_LEN[tx_size];
    Some(&WT_MATRIX_REF[qm_level][c][off..off + len])
}

#[cfg(test)]
mod tests {
    use super::*;

    // TX_SIZE indices used below (enums.h order).
    const TX_4X4: usize = 0;
    const TX_8X8: usize = 1;
    const TX_16X16: usize = 2;
    const TX_32X32: usize = 3;
    const TX_64X64: usize = 4;
    // TX_TYPE indices (enums.h order).
    const DCT_DCT: usize = 0;
    const V_DCT: usize = 11; // a 1-D transform (>= IDTX)

    /// The ported forward table anchors to the exact C `wt_matrix_ref` bytes,
    /// independently parsed from `reference/libaom/av1/common/quant_common.c`.
    /// Nine (level, plane, tx-size) spot-checks across luma+chroma and levels
    /// 0/7/14 guard against an extraction shift or a plane/level index bug.
    /// (Values verified against the C source, 2026-07-15.)
    #[test]
    fn table_anchors_to_c_source() {
        // Level 0, luma, across tx sizes.
        assert_eq!(
            &qmatrix(0, 0, TX_4X4, DCT_DCT).unwrap()[..4],
            &[32, 24, 14, 11]
        );
        assert_eq!(
            &qmatrix(0, 0, TX_8X8, DCT_DCT).unwrap()[..4],
            &[32, 32, 27, 20]
        );
        assert_eq!(
            &qmatrix(0, 0, TX_16X16, DCT_DCT).unwrap()[..4],
            &[32, 33, 33, 30]
        );
        assert_eq!(
            &qmatrix(0, 0, TX_32X32, DCT_DCT).unwrap()[..4],
            &[32, 33, 33, 33]
        );
        // Level 0, chroma (plane group c>=1). NOTE: forward chroma-4x4 DC is 29,
        // NOT 32 — the forward DC weight is only 32 for certain (plane, size).
        assert_eq!(
            &qmatrix(0, 1, TX_4X4, DCT_DCT).unwrap()[..4],
            &[29, 22, 18, 16]
        );
        assert_eq!(
            &qmatrix(0, 2, TX_16X16, DCT_DCT).unwrap()[..4],
            &[32, 34, 31, 25]
        );
        // U and V share the c>=1 base.
        assert_eq!(
            qmatrix(0, 1, TX_16X16, DCT_DCT),
            qmatrix(0, 2, TX_16X16, DCT_DCT)
        );
        // Mid + top-nonflat levels.
        assert_eq!(
            &qmatrix(7, 0, TX_16X16, DCT_DCT).unwrap()[..4],
            &[32, 33, 33, 33]
        );
        // Level 14 is the near-flat top-nonflat matrix (~33 everywhere).
        assert_eq!(
            &qmatrix(14, 0, TX_4X4, DCT_DCT).unwrap()[..4],
            &[33, 33, 33, 33]
        );
        assert_eq!(
            &qmatrix(14, 1, TX_4X4, DCT_DCT).unwrap()[..4],
            &[33, 33, 33, 33]
        );
    }

    /// `av1_get_qmatrix` flat cases: the top level and 1-D / identity transforms
    /// take no matrix (`None`), a real matrix otherwise.
    #[test]
    fn flat_cases_select_none() {
        // Level NUM_QM_LEVELS-1 (15) is the flat no-op.
        assert!(qmatrix(15, 0, TX_16X16, DCT_DCT).is_none());
        assert!(qmatrix(15, 1, TX_8X8, DCT_DCT).is_none());
        // 1-D / identity transforms are always flat, even at a steep level.
        assert!(qmatrix(0, 0, TX_16X16, V_DCT).is_none());
        assert!(qmatrix(0, 0, TX_16X16, IDTX).is_none());
        // 2-D transform at a non-flat level: a real matrix.
        assert!(qmatrix(0, 0, TX_16X16, DCT_DCT).is_some());
        // Level 14 (near-flat) is still a REAL matrix, not None.
        assert!(qmatrix(14, 0, TX_16X16, DCT_DCT).is_some());
    }

    /// Slice length equals the coded-region area, and the 64-point sizes reuse
    /// their adjusted (32-capped) matrix — i.e. `av1_qm_init`'s aliasing.
    #[test]
    fn lengths_and_64pt_aliasing() {
        assert_eq!(qmatrix(3, 0, TX_4X4, DCT_DCT).unwrap().len(), 16);
        assert_eq!(qmatrix(3, 0, TX_16X16, DCT_DCT).unwrap().len(), 256);
        assert_eq!(qmatrix(3, 0, TX_32X32, DCT_DCT).unwrap().len(), 1024);
        // TX_64X64 aliases TX_32X32 (same slice — offset 336, length 1024).
        let a = qmatrix(3, 0, TX_32X32, DCT_DCT).unwrap();
        let b = qmatrix(3, 0, TX_64X64, DCT_DCT).unwrap();
        assert_eq!(a, b, "TX_64X64 must reuse the TX_32X32 matrix");
    }

    /// Luma and chroma use distinct bases (`c >= 1`), so their matrices differ.
    #[test]
    fn luma_chroma_bases_differ() {
        let luma = qmatrix(0, 0, TX_16X16, DCT_DCT).unwrap();
        let chroma = qmatrix(0, 1, TX_16X16, DCT_DCT).unwrap();
        assert_ne!(luma, chroma, "luma and chroma bases must differ");
    }
}
