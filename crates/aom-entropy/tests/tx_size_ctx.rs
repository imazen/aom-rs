//! Unit validation of the tx-size selection layer against hand-traced C
//! semantics (`av1/common/pred_common.h` `get_tx_size_context`,
//! `av1/common/av1_common_int.h` `set_txfm_ctxs`, `av1/common/blockd.h`
//! `tx_size_from_tx_mode` / `depth_to_tx_size` / `bsize_to_max_depth` /
//! `bsize_to_tx_size_cat`, `av1/encoder/block.h` `tx_size_to_depth`).
//!
//! A direct differential test against a C shim is DEFERRED (aom-sys-ref had
//! live encoder-track WIP when this landed); until that lands, these tests pin
//! the port to independently re-transcribed C tables and hand-computed context
//! values, and the aom-decode tile roundtrip pins the decode/encode symmetry.

use aom_entropy::partition::{
    bsize_to_max_depth, bsize_to_tx_size_cat, depth_to_tx_size, get_tx_size_context, set_txfm_ctxs,
    tx_size_from_tx_mode, tx_size_to_depth, TxMode, TXFM_CTX_INIT,
};

// ---- independent transcriptions of the C tables (common_data.h) ------------------
// (deliberately re-typed from the C source, NOT imported from the crate, so a
// transposition typo in either copy makes these tests fail)

const BLOCK_SIZES_ALL: usize = 22;
const TX_SIZES_ALL: usize = 19;

/// `max_txsize_rect_lookup[BLOCK_SIZES_ALL]`.
const MAX_TXSIZE_RECT: [usize; BLOCK_SIZES_ALL] = [
    0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4, 4, 13, 14, 15, 16, 17, 18,
];
/// `sub_tx_size_map[TX_SIZES_ALL]`.
const SUB_TX: [usize; TX_SIZES_ALL] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
/// `tx_size_wide[TX_SIZES_ALL]` / `tx_size_high[TX_SIZES_ALL]`.
const TXW: [i32; TX_SIZES_ALL] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TXH: [i32; TX_SIZES_ALL] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
/// `block_size_wide` / `block_size_high` `[BLOCK_SIZES_ALL]`.
const BW: [i32; BLOCK_SIZES_ALL] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64,
];
const BH: [i32; BLOCK_SIZES_ALL] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16,
];

const TX_4X4: usize = 0;
const BLOCK_4X4: usize = 0;
const BLOCK_8X8: usize = 3;
const BLOCK_64X64: usize = 12;
const MAX_TX_DEPTH: i32 = 2;

// ---- table-generation loops from the C comments -----------------------------------

/// `bsize_to_max_depth`'s documented generation loop (blockd.h comment).
fn gen_max_depth(bsize: usize) -> usize {
    let mut tx_size = MAX_TXSIZE_RECT[bsize];
    let mut depth = 0usize;
    while (depth as i32) < MAX_TX_DEPTH && tx_size != TX_4X4 {
        depth += 1;
        tx_size = SUB_TX[tx_size];
    }
    depth
}

/// `bsize_to_tx_size_cat`'s documented generation loop (blockd.h comment),
/// including the `- 1`.
fn gen_tx_size_cat(bsize: usize) -> i32 {
    let mut tx_size = MAX_TXSIZE_RECT[bsize];
    assert_ne!(tx_size, TX_4X4, "cat undefined for BLOCK_4X4");
    let mut depth = 0i32;
    while tx_size != TX_4X4 {
        depth += 1;
        tx_size = SUB_TX[tx_size];
        assert!(depth < 10);
    }
    depth - 1
}

#[test]
fn max_depth_and_cat_tables_match_generation_loops() {
    for bsize in 0..BLOCK_SIZES_ALL {
        assert_eq!(
            bsize_to_max_depth(bsize),
            gen_max_depth(bsize),
            "bsize_to_max_depth[{bsize}]"
        );
        if bsize > BLOCK_4X4 {
            assert_eq!(
                bsize_to_tx_size_cat(bsize),
                gen_tx_size_cat(bsize),
                "bsize_to_tx_size_cat[{bsize}]"
            );
        }
    }
    // Per-category symbol count is uniform (C default_tx_size_cdf shapes:
    // cat 0 is AOM_CDF2, cats 1..=3 AOM_CDF3): every bsize in cat 0 has
    // max_depth 1, every bsize in cats 1..=3 has max_depth 2.
    for bsize in 1..BLOCK_SIZES_ALL {
        let cat = bsize_to_tx_size_cat(bsize);
        let expect_depth = if cat == 0 { 1 } else { 2 };
        assert_eq!(
            bsize_to_max_depth(bsize),
            expect_depth,
            "cat {cat} bsize {bsize} max_depth"
        );
    }
}

#[test]
fn depth_tx_size_roundtrip_all_bsizes() {
    for (bsize, &max_rect) in MAX_TXSIZE_RECT.iter().enumerate() {
        // depth 0 is always the max rect size
        assert_eq!(depth_to_tx_size(0, bsize), max_rect);
        for depth in 0..=(bsize_to_max_depth(bsize) as i32) {
            let tx = depth_to_tx_size(depth, bsize);
            assert_eq!(
                tx_size_to_depth(tx, bsize),
                depth,
                "depth<->tx_size roundtrip bsize {bsize} depth {depth}"
            );
            // each step halves along one axis: strictly smaller area unless depth 0
            if depth > 0 {
                let prev = depth_to_tx_size(depth - 1, bsize);
                assert!(
                    TXW[tx] * TXH[tx] < TXW[prev] * TXH[prev],
                    "sub step must shrink: bsize {bsize} depth {depth}"
                );
            }
        }
    }
}

#[test]
fn tx_size_from_tx_mode_matches_c() {
    // txsize_sqr_map[TX_SIZES_ALL] (common_data.h), independent copy.
    const SQR: [usize; TX_SIZES_ALL] = [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
    for bsize in 0..BLOCK_SIZES_ALL {
        // ONLY_4X4, traced from the C function verbatim: 4x4 ->
        // AOMMIN(max_txsize_lookup[4x4], TX_4X4) = TX_4X4; otherwise the
        // sqr_map[max_rect] <= TX_4X4 gate passes exactly when the rect max
        // has a 4-px side pair (TX_4X8/TX_8X4/TX_4X16/TX_16X4), returning the
        // RECT max, else TX_4X4. (Unreachable for bsize > 4x4 in a conformant
        // stream: ONLY_4X4 requires coded_lossless, and read_tx_size's
        // lossless branch preempts with TX_4X4 — the port still matches the C
        // function exactly.)
        let expect_4x4_mode = if bsize == BLOCK_4X4 {
            TX_4X4
        } else if SQR[MAX_TXSIZE_RECT[bsize]] == TX_4X4 {
            MAX_TXSIZE_RECT[bsize]
        } else {
            TX_4X4
        };
        assert_eq!(
            tx_size_from_tx_mode(bsize, TxMode::Only4x4),
            expect_4x4_mode,
            "ONLY_4X4 bsize {bsize}"
        );
        // TX_MODE_LARGEST / TX_MODE_SELECT (biggest = TX_64X64):
        // txsize_sqr_map[..] <= TX_64X64 always -> max rect; 4x4 -> min(TX_4X4,
        // TX_64X64) = TX_4X4 which IS its max rect.
        assert_eq!(
            tx_size_from_tx_mode(bsize, TxMode::Largest),
            MAX_TXSIZE_RECT[bsize],
            "LARGEST bsize {bsize}"
        );
        assert_eq!(
            tx_size_from_tx_mode(bsize, TxMode::Select),
            MAX_TXSIZE_RECT[bsize],
            "SELECT fallback bsize {bsize}"
        );
    }
}

/// An independent transcription of `get_tx_size_context` (pred_common.h) over
/// explicit neighbour state, exercised against the port on a full sweep of
/// inputs.
#[allow(clippy::too_many_arguments)]
fn c_get_tx_size_context(
    bsize: usize,
    above_byte: u8,
    left_byte: u8,
    has_above: bool,
    has_left: bool,
    above_inter: Option<usize>,
    left_inter: Option<usize>,
) -> usize {
    let max_tx_size = MAX_TXSIZE_RECT[bsize];
    let (max_tx_wide, max_tx_high) = (TXW[max_tx_size], TXH[max_tx_size]);
    let mut above = (i32::from(above_byte) >= max_tx_wide) as usize;
    let mut left = (i32::from(left_byte) >= max_tx_high) as usize;
    if has_above {
        if let Some(ab) = above_inter {
            above = (BW[ab] >= max_tx_wide) as usize;
        }
    }
    if has_left {
        if let Some(lb) = left_inter {
            left = (BH[lb] >= max_tx_high) as usize;
        }
    }
    match (has_above, has_left) {
        (true, true) => above + left,
        (true, false) => above,
        (false, true) => left,
        (false, false) => 0,
    }
}

#[test]
fn tx_size_context_hand_traced_vectors() {
    // Fresh tile: both txfm-context bytes at the 64 reset value.
    // 8x8 block (max rect TX_8X8: 8 wide, 8 high): 64 >= 8 both ways.
    assert_eq!(
        get_tx_size_context(
            BLOCK_8X8,
            TXFM_CTX_INIT,
            TXFM_CTX_INIT,
            true,
            true,
            None,
            None
        ),
        2
    );
    // Tile origin: nothing available -> 0 regardless of bytes.
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 64, 64, false, false, None, None),
        0
    );
    // First SB row: only left available -> the left bit alone.
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 64, 64, false, true, None, None),
        1
    );
    // Above neighbour stamped TX_4X4 (4 px): 4 >= 8 is false -> above 0, left 1.
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 4, 64, true, true, None, None),
        1
    );
    // Both stamped 4: 0.
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 4, 4, true, true, None, None),
        0
    );
    // 64x64 block (max rect TX_64X64): reset 64 >= 64 -> 1 per direction.
    assert_eq!(
        get_tx_size_context(BLOCK_64X64, 64, 64, true, true, None, None),
        2
    );
    // ... but a 32-stamped neighbour (TX_32X32) fails 32 >= 64.
    assert_eq!(
        get_tx_size_context(BLOCK_64X64, 32, 64, true, true, None, None),
        1
    );
    // Rect block 4x8 (bsize 1, max rect TX_4X8: 4 wide, 8 high): width compares
    // against 4, height against 8 -> above byte 4 passes, left byte 4 fails.
    assert_eq!(get_tx_size_context(1, 4, 4, true, true, None, None), 1);
    // Inter above neighbour substitutes its block WIDTH for the byte:
    // 64x64 inter above (block_size_wide 64 >= 8) overrides a 4-stamped byte...
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 4, 4, true, true, Some(BLOCK_64X64), None),
        1
    );
    // ...and a 4x4 inter above (width 4 < 8) overrides a 64 byte to 0.
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 64, 4, true, true, Some(BLOCK_4X4), None),
        0
    );
    // Inter left neighbour uses block HEIGHT: 16x4 inter left (height 4 < 8) -> 0.
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 4, 64, true, true, None, Some(17)),
        0
    );
    // Unavailable direction ignores the inter hint by construction (has_above
    // gates the substitution in C; callers pass None there anyway).
    assert_eq!(
        get_tx_size_context(BLOCK_8X8, 64, 64, false, true, Some(BLOCK_64X64), None),
        1
    );
}

#[test]
fn tx_size_context_sweep_matches_independent_transcription() {
    // Full sweep: every bsize x plausible byte values x availability x
    // inter-neighbour hints, against the independently-typed transcription.
    let bytes: [u8; 7] = [4, 8, 16, 32, 64, 128, 0];
    let inters: [Option<usize>; 4] = [None, Some(BLOCK_4X4), Some(BLOCK_64X64), Some(17)];
    let mut n = 0usize;
    for bsize in 0..BLOCK_SIZES_ALL {
        for &ab in &bytes {
            for &lb in &bytes {
                for ha in [false, true] {
                    for hl in [false, true] {
                        for &ai in &inters {
                            for &li in &inters {
                                assert_eq!(
                                    get_tx_size_context(bsize, ab, lb, ha, hl, ai, li),
                                    c_get_tx_size_context(bsize, ab, lb, ha, hl, ai, li),
                                    "bsize {bsize} ab {ab} lb {lb} ha {ha} hl {hl} ai {ai:?} li {li:?}"
                                );
                                n += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(n, 22 * 7 * 7 * 2 * 2 * 4 * 4);
}

#[test]
fn set_txfm_ctxs_stamps_tx_dims_or_skip_block_dims() {
    // Non-skip: stamps the TX dims (width above, height left) over exactly
    // n4_w / n4_h entries, leaving the rest untouched.
    let mut above = [0xAAu8; 20];
    let mut left = [0xBBu8; 20];
    // TX_16X8 (index 8): 16 wide, 8 high; block 32x16 (n4 8x4).
    set_txfm_ctxs(&mut above, &mut left, 8, 8, 4, false);
    assert_eq!(&above[..8], &[16u8; 8]);
    assert_eq!(&above[8..], &[0xAAu8; 12]);
    assert_eq!(&left[..4], &[8u8; 4]);
    assert_eq!(&left[4..], &[0xBBu8; 16]);

    // Skip (inter): stamps the BLOCK pixel dims instead (n4 * MI_SIZE).
    let mut above = [0u8; 16];
    let mut left = [0u8; 16];
    set_txfm_ctxs(&mut above, &mut left, 8, 8, 4, true);
    assert_eq!(&above[..8], &[32u8; 8]);
    assert_eq!(&left[..4], &[16u8; 4]);

    // The reset value is what a fresh tile / SB row compares against.
    assert_eq!(TXFM_CTX_INIT, 64);
}
