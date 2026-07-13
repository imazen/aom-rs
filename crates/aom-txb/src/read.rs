//! `av1_read_coeffs_txb` (libaom `av1/decoder/decodetxb.c`): the inverse of
//! [`crate::write::write_coeffs_txb`] — unpack one transform block's quantized
//! coefficients from the entropy bitstream. Reconstructs `tcoeff` (transposed
//! layout) and the `eob`, adapting the same flat CDF arena the encoder mutates.
//!
//! The coefficient contexts are computed incrementally: the reverse (high→low
//! frequency) base/br scan reads only already-decoded higher-frequency
//! neighbours, so `get_lower_levels_ctx` / `get_br_ctx` see the same values the
//! encoder computed from the full block. Levels are stored capped at base+br
//! (≤ 15); `get_nz_mag` (min-3) and `get_br_ctx` (saturating) yield identical
//! contexts to the encoder's full-magnitude levels buffer.

use crate::write::{
    A_BASE, A_BASE_EOB, A_BR, A_DC_SIGN, A_EOB_EXTRA, A_TXB_SKIP, EOB_OFF, TXSIZE_LOG2_MINUS4,
};
use crate::{
    get_br_ctx, get_lower_levels_ctx, get_lower_levels_ctx_eob, padded_idx, txb_bhl, txb_high,
    txb_wide, txsize_entropy_ctx, TxClass, EOB_GROUP_START, EOB_OFFSET_BITS, TX_TYPE_TO_CLASS,
};
use aom_entropy::cdf::{read_bit, read_symbol};
use aom_entropy::dec::OdEcDec;

use crate::scan::scan;

/// Read one CDF symbol at arena offset `off` (`n` symbols), adapting the CDF when
/// `upd`. Mirrors [`crate::write::sym`].
fn rsym(dec: &mut OdEcDec, cdfs: &mut [u16], off: usize, n: i32, upd: bool) -> i32 {
    let cdf = &mut cdfs[off..off + n as usize + 1];
    if upd {
        read_symbol(dec, cdf, n as usize)
    } else {
        dec.decode_cdf_q15(&cdf[..n as usize], n)
    }
}

/// `read_golomb` (decodetxb.c): inverse of [`crate::write::write_golomb`] — an
/// exp-Golomb value on the od_ec coder (leading zeros give the length, then the
/// mantissa MSB-first, minus one).
fn read_golomb(dec: &mut OdEcDec) -> i32 {
    let mut length = 0;
    while read_bit(dec) == 0 {
        length += 1;
        if length >= 32 {
            break;
        }
    }
    let mut x = 1i32;
    for _ in 0..length {
        x = (x << 1) | read_bit(dec);
    }
    x - 1
}

/// `av1_read_coeffs_txb` — inverse of [`crate::write::write_coeffs_txb`]. Fills
/// `tcoeff` (transposed raster layout, zeroed for uncoded positions) and returns
/// the `eob`. `tcoeff.len()` must be ≥ `txb_wide * txb_high`.
#[allow(clippy::too_many_arguments)]
pub fn read_coeffs_txb(
    dec: &mut OdEcDec,
    cdfs: &mut [u16],
    tcoeff: &mut [i32],
    tx_size: usize,
    tx_type: usize,
    plane_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    allow_update_cdf: bool,
) -> usize {
    let txs_ctx = txsize_entropy_ctx(tx_size);
    let upd = allow_update_cdf;

    let all_zero = rsym(dec, cdfs, A_TXB_SKIP + (txs_ctx * 13 + txb_skip_ctx) * 3, 2, upd);
    let width = txb_wide(tx_size);
    let height = txb_high(tx_size);
    tcoeff[..width * height].fill(0);
    if all_zero != 0 {
        return 0;
    }
    read_txb_body(dec, cdfs, tcoeff, tx_size, tx_type, plane_type, dc_sign_ctx, upd)
}

/// The txb payload after the `txb_skip` flag: the eob token + extra bits, the
/// reverse-scan base/br levels, and the forward-scan sign/golomb pass. Inverse of
/// [`crate::write::write_txb_body`]. Returns the decoded `eob`.
#[allow(clippy::too_many_arguments)]
fn read_txb_body(
    dec: &mut OdEcDec,
    cdfs: &mut [u16],
    tcoeff: &mut [i32],
    tx_size: usize,
    tx_type: usize,
    plane_type: usize,
    dc_sign_ctx: usize,
    upd: bool,
) -> usize {
    let txs_ctx = txsize_entropy_ctx(tx_size);
    let tx_class = TX_TYPE_TO_CLASS[tx_type];
    let eob_multi_size = TXSIZE_LOG2_MINUS4[tx_size] as usize;
    let eob_multi_ctx = if tx_class == TxClass::TwoD { 0 } else { 1 };
    let nsy = 5 + eob_multi_size;
    let eob_pt = rsym(
        dec,
        cdfs,
        EOB_OFF[eob_multi_size] + (plane_type * 2 + eob_multi_ctx) * (nsy + 1),
        nsy as i32,
        upd,
    ) + 1;

    let eob_offset_bits = EOB_OFFSET_BITS[eob_pt as usize] as i32;
    let mut eob = EOB_GROUP_START[eob_pt as usize] as i32;
    if eob_offset_bits > 0 {
        let eob_ctx = (eob_pt - 3) as usize;
        let mut eob_shift = eob_offset_bits - 1;
        let bit = rsym(dec, cdfs, A_EOB_EXTRA + ((txs_ctx * 2 + plane_type) * 9 + eob_ctx) * 3, 2, upd);
        if bit != 0 {
            eob += 1 << eob_shift;
        }
        for i in 1..eob_offset_bits {
            eob_shift = eob_offset_bits - 1 - i;
            if read_bit(dec) != 0 {
                eob += 1 << eob_shift;
            }
        }
    }
    let eob = eob as usize;

    let sc = scan(tx_size, tx_type);
    let bhl = txb_bhl(tx_size);
    let width = txb_wide(tx_size);
    let mut levels_buf = [0u8; crate::TX_PAD_2D];

    // Reverse scan (high→low frequency): base level, then base-range refinement.
    // Magnitudes are stashed in `tcoeff[pos]`; the forward pass applies signs.
    for c in (0..eob).rev() {
        let pos = sc[c] as usize;
        let mut level = if c == eob - 1 {
            let ctx = get_lower_levels_ctx_eob(bhl, width, c) as usize;
            rsym(dec, cdfs, A_BASE_EOB + ((txs_ctx * 2 + plane_type) * 4 + ctx) * 4, 3, upd) + 1
        } else {
            let ctx = get_lower_levels_ctx(&levels_buf, pos, bhl, tx_size, tx_class) as usize;
            rsym(dec, cdfs, A_BASE + ((txs_ctx * 2 + plane_type) * 42 + ctx) * 5, 4, upd)
        };
        if level > 2 {
            // NUM_BASE_LEVELS
            let br_ctx = get_br_ctx(&levels_buf, pos, bhl, tx_class) as usize;
            let mts = txs_ctx.min(3);
            let cdf_off = A_BR + ((mts * 2 + plane_type) * 21 + br_ctx) * 5;
            let mut idx = 0;
            while idx < 12 {
                // COEFF_BASE_RANGE
                let k = rsym(dec, cdfs, cdf_off, 4, upd);
                level += k;
                if k < 3 {
                    break;
                }
                idx += 3; // BR_CDF_SIZE - 1
            }
        }
        levels_buf[padded_idx(pos, bhl)] = level.min(i8::MAX as i32) as u8;
        tcoeff[pos] = level;
    }

    // Forward scan (low→high frequency): sign + golomb, finalize signed coeffs.
    #[allow(clippy::needless_range_loop)]
    for c in 0..eob {
        let pos = sc[c] as usize;
        let mut level = tcoeff[pos];
        if level != 0 {
            let sign = if c == 0 {
                rsym(dec, cdfs, A_DC_SIGN + (plane_type * 3 + dc_sign_ctx) * 3, 2, upd)
            } else {
                read_bit(dec)
            };
            if level > 14 {
                // COEFF_BASE_RANGE + NUM_BASE_LEVELS
                level += read_golomb(dec);
            }
            tcoeff[pos] = if sign != 0 { -level } else { level };
        }
    }
    eob
}
