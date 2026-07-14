//! Direct C differentials for the chroma decode facades — the real
//! `is_chroma_reference` / `av1_get_max_uv_txsize` / `scale_chroma_bsize` /
//! `av1_get_tx_type` (intra UV arm) / `intra_mode_to_tx_type` static inlines
//! (via dec_shim.c) vs the aom-decode ports. Previously roundtrip-covered
//! only (both sides shared the Rust facade).

use aom_decode::{is_chroma_reference, max_uv_txsize, scale_chroma_bsize, uv_tx_type};
use aom_entropy::partition::get_plane_block_size;
use aom_sys_ref as c;

const BLOCK_SIZES_ALL: usize = 22;
const SS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

#[test]
fn is_chroma_reference_matches_c() {
    let mut cases = 0u64;
    for bsize in 0..BLOCK_SIZES_ALL {
        for &(ss_x, ss_y) in &SS {
            for mi_row in 0..16 {
                for mi_col in 0..16 {
                    assert_eq!(
                        is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y),
                        c::ref_is_chroma_reference(mi_row, mi_col, bsize, ss_x as i32, ss_y as i32),
                        "bsize={bsize} ss=({ss_x},{ss_y}) mi=({mi_row},{mi_col})"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 22 * 4 * 256);
}

#[test]
fn max_uv_txsize_matches_c() {
    // Domain: (bsize, ss) combos with a real chroma plane block size — the C
    // asserts (and the decoder rejects as corrupt, decodeframe.c:393) the rest
    // (e.g. tall sub-8x8 shapes in 4:2:2).
    let mut valid = 0u32;
    for bsize in 0..BLOCK_SIZES_ALL {
        for &(ss_x, ss_y) in &SS {
            if get_plane_block_size(bsize, ss_x, ss_y) == 255 {
                continue;
            }
            assert_eq!(
                max_uv_txsize(bsize, ss_x, ss_y),
                c::ref_get_max_uv_txsize(bsize, ss_x as i32, ss_y as i32),
                "bsize={bsize} ss=({ss_x},{ss_y})"
            );
            valid += 1;
        }
    }
    assert!(valid >= 70, "suspiciously small valid domain ({valid})");
}

#[test]
fn scale_chroma_bsize_matches_c() {
    for bsize in 0..BLOCK_SIZES_ALL {
        for &(ss_x, ss_y) in &SS {
            assert_eq!(
                scale_chroma_bsize(bsize, ss_x, ss_y),
                c::ref_scale_chroma_bsize(bsize, ss_x as i32, ss_y as i32),
                "bsize={bsize} ss=({ss_x},{ss_y})"
            );
        }
    }
}

/// Chroma-real tx sizes (both dims ≤ 32 — `av1_get_max_uv_txsize` is
/// 64-clamped, so these are exactly the sizes the UV arm can see).
const CHROMA_TX: [usize; 14] = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16];

#[test]
fn uv_tx_type_matches_c() {
    // The intra UV arm: mode-implied type demoted by ext-tx-set membership.
    // lossless=false (the Rust port carries no lossless arm; the driver's
    // lossless streams are out of scope).
    let mut cases = 0u64;
    for uv_mode in 0..14 {
        // 13 UV intra modes + UV_CFL_PRED(13)
        for &tx in &CHROMA_TX {
            for reduced in [false, true] {
                let r = uv_tx_type(uv_mode as i32, tx, reduced);
                // y_mode is irrelevant to the UV arm; pass a rotating one to
                // prove it.
                let cr = c::ref_av1_get_tx_type_uv_intra(uv_mode % 13, uv_mode, tx, reduced, false);
                assert_eq!(r, cr, "uv_mode={uv_mode} tx={tx} reduced={reduced}");
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 14 * 14 * 2);
}

#[test]
fn intra_mode_to_tx_type_matches_c() {
    // The bare mode->type table through both plane arms.
    for mode in 0..13 {
        for uv_mode in 0..14 {
            let cy = c::ref_intra_mode_to_tx_type(mode, uv_mode, 0);
            let cuv = c::ref_intra_mode_to_tx_type(mode, uv_mode, 1);
            // Y arm keys on mode only; UV arm on get_uv_mode(uv_mode). The
            // Rust table is INTRA_MODE_TO_TX_TYPE (private) — validate through
            // uv_tx_type's full-set case (all types available at TX_4X4,
            // reduced=false, where no demotion happens).
            assert_eq!(cuv, uv_tx_type(uv_mode as i32, 0, false), "uv arm uv_mode={uv_mode}");
            // And the Y arm equals the UV arm when uv_mode == mode (mode < 13
            // never hits the CFL mapping).
            assert_eq!(cy, c::ref_intra_mode_to_tx_type(0, mode, 1), "y arm mode={mode}");
        }
    }
}
