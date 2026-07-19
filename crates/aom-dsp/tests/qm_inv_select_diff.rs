//! Differential: the Rust INVERSE-QM selector [`aom_dsp::quant::iqmatrix`] vs the
//! REAL libaom `av1_qm_init` packing (`giqmatrix[q][c][t]` pointers into the
//! file-static `iwt_matrix_ref`), read back through
//! [`aom_sys_ref::ref_iqm_giqmatrix`].
//!
//! Mirror of `qm_fwd_select_diff` for the inverse table that #23's encoder
//! threading needs in-crate (`aom-encode` cannot depend on `aom-decode`'s
//! validated copy — library-graph cycle). Validates the generated
//! `qm_inv_tables.rs` bytes, the per-tx packing offsets, the plane grouping,
//! the level indexing, and the 64-point matrix aliasing against the genuine C
//! init loop.

use aom_dsp::quant::iqmatrix;
use aom_sys_ref as c;

// enums.h order.
const DCT_DCT: usize = 0;
const IDTX: usize = 9;
const V_DCT: usize = 11;
const NUM_QM_LEVELS: usize = 16;
const TX_SIZES_ALL: usize = 19;

/// Full sweep: every (qm level, plane, tx size) cell, DCT_DCT (2-D) so a real
/// matrix is selected. The Rust selector must byte-match C's `giqmatrix[q][c][t]`
/// for every non-flat cell, and both must agree on `None` at the flat top level.
#[test]
fn inverse_qmatrix_matches_c_av1_qm_init() {
    c::ref_init();
    let mut some_cells = 0usize; // anti-vacuous: count real matrix comparisons
    let mut none_cells = 0usize;
    let mut max_len = 0usize;
    for q in 0..NUM_QM_LEVELS {
        for plane in 0..3usize {
            for t in 0..TX_SIZES_ALL {
                let mine = iqmatrix(q, plane, t, DCT_DCT);
                let theirs = c::ref_iqm_giqmatrix(q, plane, t);
                match (mine, &theirs) {
                    (Some(m), Some(cm)) => {
                        assert_eq!(
                            m,
                            cm.as_slice(),
                            "inverse QM mismatch at (level={q}, plane={plane}, tx_size={t})"
                        );
                        some_cells += 1;
                        max_len = max_len.max(m.len());
                    }
                    (None, None) => none_cells += 1,
                    (mine, theirs) => panic!(
                        "None/Some disagreement at (level={q}, plane={plane}, tx_size={t}): \
                         rust={:?} c={:?}",
                        mine.map(<[u8]>::len),
                        theirs.as_ref().map(Vec::len),
                    ),
                }
            }
        }
    }
    // 15 non-flat levels x 3 planes x 19 tx sizes = 855 real matrices; the flat
    // top level (q=15) x 3 x 19 = 57 None cells.
    assert_eq!(
        some_cells,
        15 * 3 * TX_SIZES_ALL,
        "expected 855 real-matrix cells"
    );
    assert_eq!(none_cells, 3 * TX_SIZES_ALL, "expected 57 flat (None) cells");
    assert_eq!(max_len, 1024, "largest matrix must be a full 32x32 (1024)");
}

/// The `tx_type` gating (`av1_get_iqmatrix` returns the flat matrix for 1-D /
/// identity transforms) lives in the selector, not the packing. Confirm the
/// Rust selector returns `None` for those even at a steep level, while DCT_DCT
/// at the same cell selects a real matrix identical to C. Also anchor the
/// inverse selector against the forward one: same Some/None shape everywhere
/// (the invariant `xform_quant`'s QM resolution relies on).
#[test]
fn one_d_and_identity_transforms_are_flat() {
    c::ref_init();
    for &t in &[0usize, 2, 3, 4] {
        assert!(iqmatrix(0, 0, t, IDTX).is_none(), "IDTX must be flat");
        assert!(iqmatrix(0, 0, t, V_DCT).is_none(), "1-D V_DCT must be flat");
        // and the 2-D transform at the same cell is a real, C-matching matrix.
        let mine = iqmatrix(0, 0, t, DCT_DCT).unwrap();
        let theirs = c::ref_iqm_giqmatrix(0, 0, t).unwrap();
        assert_eq!(mine, theirs.as_slice());
    }
    // fwd/inv selector agreement on Some/None across the full grid.
    for q in 0..NUM_QM_LEVELS {
        for plane in 0..3usize {
            for t in 0..TX_SIZES_ALL {
                for tt in 0..16usize {
                    assert_eq!(
                        aom_dsp::quant::qmatrix(q, plane, t, tt).is_some(),
                        iqmatrix(q, plane, t, tt).is_some(),
                        "fwd/inv Some-ness must agree at (q={q},p={plane},t={t},tt={tt})"
                    );
                }
            }
        }
    }
}
