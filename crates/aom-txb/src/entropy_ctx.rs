//! Per-transform-block entropy-context propagation (libaom `txb_common.h`,
//! `encodetxb.c`): `get_txb_ctx` reads the neighbour entropy contexts into the
//! `txb_skip_ctx` / `dc_sign_ctx` that gate the writer / cost / trellis, and
//! `av1_get_txb_entropy_context` writes this block's packed context for its
//! neighbours. Both are used by the encoder *and* decoder tracks. Byte-exact.

use crate::scan::scan;

const COEFF_CONTEXT_BITS: u32 = 3;
const COEFF_CONTEXT_MASK: i32 = (1 << COEFF_CONTEXT_BITS) - 1; // 7

// tx unit dims (width/4, height/4) per TX_SIZES_ALL.
const TX_WIDE_UNIT: [usize; 19] = [1, 2, 4, 8, 16, 1, 2, 2, 4, 4, 8, 8, 16, 1, 4, 2, 8, 4, 16];
const TX_HIGH_UNIT: [usize; 19] = [1, 2, 4, 8, 16, 2, 1, 4, 2, 8, 4, 16, 8, 4, 1, 8, 2, 16, 4];

// txsize_to_bsize[TX_SIZES_ALL] -> BLOCK_SIZE (matches our BlockSize discriminants).
const TXSIZE_TO_BSIZE: [usize; 19] = [
    0, 3, 6, 9, 12, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21,
];

// num_pels_log2_lookup[BLOCK_SIZES_ALL].
const NUM_PELS_LOG2: [i32; 22] = [
    4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 6, 6, 8, 8, 10, 10,
];

// dc_sign_contexts[4*16+1] = 65 entries: 32x 1, then 0, then 32x 2.
#[inline]
fn dc_sign_context(dc_sign: i32) -> i32 {
    let idx = dc_sign + 32;
    if idx < 32 {
        1
    } else if idx == 32 {
        0
    } else {
        2
    }
}

// skip_contexts[5][5].
const SKIP_CONTEXTS: [[i32; 5]; 5] = [
    [1, 2, 2, 2, 3],
    [2, 4, 4, 4, 5],
    [2, 4, 4, 4, 5],
    [2, 4, 4, 4, 5],
    [3, 5, 5, 5, 6],
];

/// `get_entropy_context`: `(any above nonzero) + (any left nonzero)`.
fn get_entropy_context(tx_size: usize, a: &[i8], l: &[i8]) -> i32 {
    let above = a[..TX_WIDE_UNIT[tx_size]].iter().any(|&x| x != 0);
    let left = l[..TX_HIGH_UNIT[tx_size]].iter().any(|&x| x != 0);
    above as i32 + left as i32
}

/// `get_txb_ctx`: neighbour entropy contexts (`a` above, `l` left; packed bytes
/// `cul_level | dc_sign<<3`) -> `(txb_skip_ctx, dc_sign_ctx)` for `plane`
/// (0 = luma). `plane_bsize` is the plane block size (BlockSize discriminant).
pub fn get_txb_ctx(
    plane_bsize: usize,
    tx_size: usize,
    plane: usize,
    a: &[i8],
    l: &[i8],
) -> (i32, i32) {
    let w_unit = TX_WIDE_UNIT[tx_size];
    let h_unit = TX_HIGH_UNIT[tx_size];
    const SIGNS: [i32; 3] = [0, -1, 1];
    let mut dc_sign = 0;
    for &x in &a[..w_unit] {
        dc_sign += SIGNS[((x as u8) >> COEFF_CONTEXT_BITS) as usize];
    }
    for &x in &l[..h_unit] {
        dc_sign += SIGNS[((x as u8) >> COEFF_CONTEXT_BITS) as usize];
    }
    let dc_sign_ctx = dc_sign_context(dc_sign);

    let txb_skip_ctx = if plane == 0 {
        if plane_bsize == TXSIZE_TO_BSIZE[tx_size] {
            0
        } else {
            let mut top = 0i32;
            for &x in &a[..w_unit] {
                top |= x as i32;
            }
            top = (top & COEFF_CONTEXT_MASK).min(4);
            let mut left = 0i32;
            for &x in &l[..h_unit] {
                left |= x as i32;
            }
            left = (left & COEFF_CONTEXT_MASK).min(4);
            SKIP_CONTEXTS[top as usize][left as usize]
        }
    } else {
        let ctx_base = get_entropy_context(tx_size, a, l);
        let ctx_offset = if NUM_PELS_LOG2[plane_bsize] > NUM_PELS_LOG2[TXSIZE_TO_BSIZE[tx_size]] {
            10
        } else {
            7
        };
        ctx_base + ctx_offset
    };
    (txb_skip_ctx, dc_sign_ctx)
}

/// `av1_get_txb_entropy_context`: pack this block's context (culminative level,
/// capped at 7, OR the DC-sign code) for its right/below neighbours.
pub fn txb_entropy_context(qcoeff: &[i32], tx_size: usize, tx_type: usize, eob: usize) -> u8 {
    if eob == 0 {
        return 0;
    }
    let sc = scan(tx_size, tx_type);
    let mut cul_level = 0i32;
    for &s in &sc[..eob] {
        let v = qcoeff[s as usize];
        if v == 0 {
            continue;
        }
        cul_level += v.abs();
        if cul_level > COEFF_CONTEXT_MASK {
            break;
        }
    }
    cul_level = cul_level.min(COEFF_CONTEXT_MASK);
    // set_dc_sign
    let dc = qcoeff[0];
    if dc < 0 {
        cul_level |= 1 << COEFF_CONTEXT_BITS;
    } else if dc > 0 {
        cul_level += 2 << COEFF_CONTEXT_BITS;
    }
    cul_level as u8
}
