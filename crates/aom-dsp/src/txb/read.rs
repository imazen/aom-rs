//! `av1_read_coeffs_txb` (libaom `av1/decoder/decodetxb.c`): the inverse of
//! [`crate::txb::write::write_coeffs_txb`] — unpack one transform block's quantized
//! coefficients from the entropy bitstream. Reconstructs `tcoeff` (transposed
//! layout) and the `eob`, adapting the same flat CDF arena the encoder mutates.
//!
//! The coefficient contexts are computed incrementally: the reverse (high→low
//! frequency) base/br scan reads only already-decoded higher-frequency
//! neighbours, so `get_lower_levels_ctx` / `get_br_ctx` see the same values the
//! encoder computed from the full block. Levels are stored capped at base+br
//! (≤ 15); `get_nz_mag` (min-3) and `get_br_ctx` (saturating) yield identical
//! contexts to the encoder's full-magnitude levels buffer.

use crate::txb::write::{
    A_BASE, A_BASE_EOB, A_BR, A_DC_SIGN, A_EOB_EXTRA, A_TXB_SKIP, EOB_OFF, TXSIZE_LOG2_MINUS4,
};
use crate::txb::{
    get_br_ctx, get_lower_levels_ctx, get_lower_levels_ctx_eob, padded_idx, txb_bhl, txb_high,
    txb_wide, txsize_entropy_ctx, TxClass, EOB_GROUP_START, EOB_OFFSET_BITS, TX_TYPE_TO_CLASS,
};
use crate::entropy::cdf::{read_bit, read_symbol};
use crate::entropy::dec::OdEcDec;

use crate::txb::scan::scan;

/// Read one CDF symbol at arena offset `off` (`n` symbols), adapting the CDF when
/// `upd`. Mirrors [`crate::txb::write::sym`].
fn rsym(dec: &mut OdEcDec, cdfs: &mut [u16], off: usize, n: i32, upd: bool) -> i32 {
    let cdf = &mut cdfs[off..off + n as usize + 1];
    if upd {
        read_symbol(dec, cdf, n as usize)
    } else {
        dec.decode_cdf_q15(&cdf[..n as usize], n)
    }
}

/// `read_golomb` (decodetxb.c): inverse of [`crate::txb::write::write_golomb`] — an
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

/// `av1_read_coeffs_txb` — inverse of [`crate::txb::write::write_coeffs_txb`]. Fills
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

    let all_zero = rsym(
        dec,
        cdfs,
        A_TXB_SKIP + (txs_ctx * 13 + txb_skip_ctx) * 3,
        2,
        upd,
    );
    let width = txb_wide(tx_size);
    let height = txb_high(tx_size);
    tcoeff[..width * height].fill(0);
    if all_zero != 0 {
        return 0;
    }
    read_txb_body(
        dec,
        cdfs,
        tcoeff,
        tx_size,
        tx_type,
        plane_type,
        dc_sign_ctx,
        upd,
    )
}

/// The txb payload after the `txb_skip` flag: the eob token + extra bits, the
/// reverse-scan base/br levels, and the forward-scan sign/golomb pass. Inverse of
/// [`crate::txb::write::write_txb_body`]. Returns the decoded `eob`.
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
        let bit = rsym(
            dec,
            cdfs,
            A_EOB_EXTRA + ((txs_ctx * 2 + plane_type) * 9 + eob_ctx) * 3,
            2,
            upd,
        );
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
    let mut levels_buf = [0u8; crate::txb::TX_PAD_2D];

    // Reverse scan (high→low frequency): base level, then base-range refinement.
    // Magnitudes are stashed in `tcoeff[pos]`; the forward pass applies signs.
    for c in (0..eob).rev() {
        let pos = sc[c] as usize;
        let mut level = if c == eob - 1 {
            let ctx = get_lower_levels_ctx_eob(bhl, width, c) as usize;
            rsym(
                dec,
                cdfs,
                A_BASE_EOB + ((txs_ctx * 2 + plane_type) * 4 + ctx) * 4,
                3,
                upd,
            ) + 1
        } else {
            let ctx = get_lower_levels_ctx(&levels_buf, pos, bhl, tx_size, tx_class) as usize;
            rsym(
                dec,
                cdfs,
                A_BASE + ((txs_ctx * 2 + plane_type) * 42 + ctx) * 5,
                4,
                upd,
            )
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
                rsym(
                    dec,
                    cdfs,
                    A_DC_SIGN + (plane_type * 3 + dc_sign_ctx) * 3,
                    2,
                    upd,
                )
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

/// `tx_size_2d` (enums.h): full pel count per transform size — drives
/// `av1_get_tx_scale`. Note this is the FULL area (64x64 → 4096), not the clamped
/// 32×32 coded region, so `tx_scale(TX_64X64) == 2`.
const TX_SIZE_2D: [i32; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];

/// `av1_get_tx_scale(tx_size)` = `(pels > 256) + (pels > 1024)`: the extra
/// right-shift applied to dequantized coefficients for large transforms.
#[inline]
pub fn tx_scale(tx_size: usize) -> i32 {
    let pels = TX_SIZE_2D[tx_size];
    (pels > 256) as i32 + (pels > 1024) as i32
}

/// Decoder dequant — the `qcoeff → dqcoeff` step fused into `av1_read_coeffs_txb`
/// (decodetxb.c). Given the signed decoded `qcoeff` (raster layout, as produced
/// by [`read_coeffs_txb`]/[`read_coeffs_txb_full`]), fill `dqcoeff` (raster) with
/// the reconstructed transform-domain residuals ready for the inverse transform.
///
/// Per coefficient: mask the magnitude to 20 bits, multiply by the per-position
/// dequant step (`get_dqv`, folding an inverse quant matrix when present), mask
/// the product to 24 bits, right-shift by [`tx_scale`], reapply the sign, then
/// clamp to the bitdepth range `[-(1<<(7+bd)), (1<<(7+bd))-1]`. `iqmatrix` is
/// indexed by raster position; `None` selects the flat (no-QM) dequant.
///
/// This is the structural inverse of the encoder's dqcoeff computation in
/// `crate::quant::aom_quantize_b_no_qmatrix`: for a
/// conformant stream the two agree (the masks/clamp are no-ops on in-range
/// coefficients); the masks/clamp additionally bound malformed input.
pub fn dequant_txb(
    qcoeff: &[i32],
    dqcoeff: &mut [i32],
    tx_size: usize,
    dequant: [i16; 2],
    iqmatrix: Option<&[u8]>,
    bd: i32,
) {
    let area = txb_wide(tx_size) * txb_high(tx_size);
    let shift = tx_scale(tx_size);
    let max_value = (1i32 << (7 + bd)) - 1;
    let min_value = -(1i32 << (7 + bd));
    dqcoeff[..area].fill(0);
    for pos in 0..area {
        let q = qcoeff[pos];
        if q == 0 {
            continue;
        }
        let sign = q < 0;
        let level = (q.unsigned_abs() as i64) & 0xfffff;
        let dqv = crate::txb::optimize::get_dqv(dequant, pos, iqmatrix) as i64;
        let mut dq = ((level * dqv) & 0xffffff) as i32;
        dq >>= shift;
        if sign {
            dq = -dq;
        }
        dqcoeff[pos] = dq.clamp(min_value, max_value);
    }
}

/// `read_coeffs_txb_full` — inverse of [`crate::txb::write::write_coeffs_txb_full`]: the
/// txb_skip flag, then (luma, `plane_type == 0`) the `tx_type` via `read_tx_type` on the
/// caller-selected `ext_tx_cdf` slot, then the coefficient body. Chroma inherits
/// `tx_type_in` (derived from the luma block). Returns `(eob, tx_type)`.
#[allow(clippy::too_many_arguments)]
pub fn read_coeffs_txb_full(
    dec: &mut OdEcDec,
    cdfs: &mut [u16],
    ext_tx_cdf: &mut [u16],
    tcoeff: &mut [i32],
    tx_size: usize,
    plane_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    allow_update_cdf: bool,
    is_inter: bool,
    reduced: bool,
    signal_gate: bool,
    tx_type_in: usize,
) -> (usize, usize) {
    let txs_ctx = txsize_entropy_ctx(tx_size);
    let upd = allow_update_cdf;
    let all_zero = rsym(
        dec,
        cdfs,
        A_TXB_SKIP + (txs_ctx * 13 + txb_skip_ctx) * 3,
        2,
        upd,
    );
    let (width, height) = (txb_wide(tx_size), txb_high(tx_size));
    tcoeff[..width * height].fill(0);
    if all_zero != 0 {
        return (0, 0);
    }
    let tx_type = if plane_type == 0 {
        crate::txb::read_tx_type(dec, ext_tx_cdf, tx_size, is_inter, reduced, signal_gate)
    } else {
        tx_type_in
    };
    let eob = read_txb_body(
        dec,
        cdfs,
        tcoeff,
        tx_size,
        tx_type,
        plane_type,
        dc_sign_ctx,
        upd,
    );
    (eob, tx_type)
}
