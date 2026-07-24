//! `av1_write_coeffs_txb` (libaom `av1/encoder/encodetxb.c`): pack one transform
//! block's quantized coefficients into the entropy bitstream. Produces
//! byte-identical output to C libaom.
//!
//! The `av1_write_tx_type` step (plane-0 tx_type signaling) is intentionally out
//! of scope here — it depends on `mbmi` state; a caller wires it separately.
//!
//! All CDFs live in a single caller-owned flat `u16` arena (`CdfArena`) whose
//! layout mirrors the differential shim exactly, so both sides mutate identical
//! adaptive state. The producing quantizer supplies `tcoeff` (transposed
//! layout), `eob`, and the two entropy contexts (`txb_skip_ctx`, `dc_sign_ctx`).

use crate::txb::scan::scan;
use crate::txb::{
    get_br_ctx, get_eob_pos_token, get_nz_map_contexts, txb_high, txb_init_levels, txb_wide,
    TxClass, EOB_OFFSET_BITS, TX_PAD_2D, TX_TYPE_TO_CLASS,
};
use crate::entropy::cdf::write_symbol;
use crate::entropy::enc::OdEcEnc;

// Header-static index tables (common_data.h / entropy.h).
pub(crate) const TXSIZE_LOG2_MINUS4: [i32; 19] =
    [0, 2, 4, 6, 6, 1, 1, 3, 3, 5, 5, 6, 6, 2, 2, 4, 4, 5, 5];
const TXS_SQR: [usize; 19] = [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
const TXS_SQR_UP: [usize; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];

/// `get_txsize_entropy_ctx`.
#[inline]
pub fn txsize_entropy_ctx(tx_size: usize) -> usize {
    (TXS_SQR[tx_size] + TXS_SQR_UP[tx_size] + 1) >> 1
}

// --- CDF arena: one flat u16 buffer; offsets/strides match the shim exactly ---
pub(crate) const A_TXB_SKIP: usize = 0; // [5][13] n2  s3
const A_EOB16: usize = 195; // [2][2] n5  s6
const A_EOB32: usize = 219; // [2][2] n6  s7
const A_EOB64: usize = 247; // [2][2] n7  s8
const A_EOB128: usize = 279; // [2][2] n8  s9
const A_EOB256: usize = 315; // [2][2] n9  s10
const A_EOB512: usize = 355; // [2][2] n10 s11
const A_EOB1024: usize = 399; // [2][2] n11 s12
pub(crate) const A_EOB_EXTRA: usize = 447; // [5][2][9] n2 s3
pub(crate) const A_BASE_EOB: usize = 717; // [5][2][4] n3 s4
pub(crate) const A_BASE: usize = 877; // [5][2][42] n4 s5
pub(crate) const A_BR: usize = 2977; // [5][2][21] n4 s5
pub(crate) const A_DC_SIGN: usize = 4027; // [2][3] n2 s3
/// Total u16 length of the coefficient-coding CDF arena.
pub const CDF_ARENA_LEN: usize = 4045;

/// `av1_reset_cdf_symbol_counters`' coefficient slice (entropy.c:85-98): zero
/// the per-row adaptation COUNTER slot (index `nsymbs` within each
/// `CDF_SIZE(nsymbs) = nsymbs+1`-stride row) of every coefficient CDF in the
/// arena. C runs this when a frame's end-of-frame context is SAVED
/// (`REFRESH_FRAME_CONTEXT_BACKWARD`, decodeframe.c:5492), so a later frame
/// inheriting the context via `primary_ref_frame` restarts the update-rate
/// ramp — carrying the counters adapts too slowly and drifts off C within a
/// few frames (found by the animated-AVIF alpha-track chain, 2026-07-23).
pub fn reset_arena_cdf_counters(cdfs: &mut [u16]) {
    // (offset, rows, nsymbs, stride) per segment — the layout constants above.
    const SEGS: [(usize, usize, usize); 13] = [
        (A_TXB_SKIP, 5 * 13, 2),
        (A_EOB16, 4, 5),
        (A_EOB32, 4, 6),
        (A_EOB64, 4, 7),
        (A_EOB128, 4, 8),
        (A_EOB256, 4, 9),
        (A_EOB512, 4, 10),
        (A_EOB1024, 4, 11),
        (A_EOB_EXTRA, 5 * 2 * 9, 2),
        (A_BASE_EOB, 5 * 2 * 4, 3),
        (A_BASE, 5 * 2 * 42, 4),
        (A_BR, 5 * 2 * 21, 4),
        (A_DC_SIGN, 2 * 3, 2),
    ];
    for (off, rows, nsymbs) in SEGS {
        let stride = nsymbs + 1;
        for r in 0..rows {
            cdfs[off + r * stride + nsymbs] = 0;
        }
    }
}

pub(crate) const EOB_OFF: [usize; 7] = [
    A_EOB16, A_EOB32, A_EOB64, A_EOB128, A_EOB256, A_EOB512, A_EOB1024,
];

/// Pack one transform block. Appends symbols to `enc`; mutates the adaptive
/// CDFs in `cdfs` in place when `allow_update_cdf`.
///
/// `tcoeff` is the block's quantized coefficients in transposed raster layout
/// (`pos = col*height + row`), length ≥ `txb_wide*txb_high`.
#[allow(clippy::too_many_arguments)]
pub fn write_coeffs_txb(
    enc: &mut OdEcEnc,
    cdfs: &mut [u16],
    tcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    plane_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    allow_update_cdf: bool,
) {
    let txs_ctx = txsize_entropy_ctx(tx_size);
    let upd = allow_update_cdf;

    sym(
        enc,
        cdfs,
        A_TXB_SKIP + (txs_ctx * 13 + txb_skip_ctx) * 3,
        (eob == 0) as i32,
        2,
        upd,
    );
    if eob == 0 {
        return;
    }
    // tx_type (plane-0) is written by write_coeffs_txb_full; here just the payload.
    write_txb_body(
        enc,
        cdfs,
        tcoeff,
        eob,
        tx_size,
        tx_type,
        plane_type,
        dc_sign_ctx,
        upd,
    );
}

/// The txb payload written after the `txb_skip` flag (and, for luma, the
/// `tx_type`): the eob position + eob-extra bits, the coefficient base/br levels,
/// and the sign/golomb pass. Shared by [`write_coeffs_txb`] and
/// [`write_coeffs_txb_full`]. `upd` = `allow_update_cdf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_txb_body(
    enc: &mut OdEcEnc,
    cdfs: &mut [u16],
    tcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    plane_type: usize,
    dc_sign_ctx: usize,
    upd: bool,
) {
    let txs_ctx = txsize_entropy_ctx(tx_size);
    let tx_class = TX_TYPE_TO_CLASS[tx_type];
    let (eob_pt, eob_extra) = get_eob_pos_token(eob as i32);
    let eob_multi_size = TXSIZE_LOG2_MINUS4[tx_size] as usize;
    let eob_multi_ctx = if tx_class == TxClass::TwoD { 0 } else { 1 };
    let nsy = 5 + eob_multi_size;
    sym(
        enc,
        cdfs,
        EOB_OFF[eob_multi_size] + (plane_type * 2 + eob_multi_ctx) * (nsy + 1),
        eob_pt - 1,
        nsy as i32,
        upd,
    );

    let eob_offset_bits = EOB_OFFSET_BITS[eob_pt as usize] as i32;
    if eob_offset_bits > 0 {
        let eob_ctx = (eob_pt - 3) as usize;
        let mut eob_shift = eob_offset_bits - 1;
        let bit = i32::from(eob_extra & (1 << eob_shift) != 0);
        sym(
            enc,
            cdfs,
            A_EOB_EXTRA + ((txs_ctx * 2 + plane_type) * 9 + eob_ctx) * 3,
            bit,
            2,
            upd,
        );
        for i in 1..eob_offset_bits {
            eob_shift = eob_offset_bits - 1 - i;
            let bit = i32::from(eob_extra & (1 << eob_shift) != 0);
            write_bit(enc, bit);
        }
    }

    let width = txb_wide(tx_size);
    let height = txb_high(tx_size);
    let mut levels_buf = [0u8; TX_PAD_2D];
    txb_init_levels(tcoeff, width, height, &mut levels_buf); // TX_PAD_TOP == 0
    let sc = scan(tx_size, tx_type);
    let mut coeff_contexts = [0i8; 32 * 32];
    get_nz_map_contexts(&levels_buf, sc, eob, tx_size, tx_class, &mut coeff_contexts);

    let bhl = crate::txb::txb_bhl(tx_size);
    for c in (0..eob).rev() {
        let pos = sc[c] as usize;
        let coeff_ctx = coeff_contexts[pos] as usize;
        let v = tcoeff[pos];
        let level = v.unsigned_abs();

        if c == eob - 1 {
            let s = level.min(3) as i32 - 1;
            sym(
                enc,
                cdfs,
                A_BASE_EOB + ((txs_ctx * 2 + plane_type) * 4 + coeff_ctx) * 4,
                s,
                3,
                upd,
            );
        } else {
            let s = level.min(3) as i32;
            sym(
                enc,
                cdfs,
                A_BASE + ((txs_ctx * 2 + plane_type) * 42 + coeff_ctx) * 5,
                s,
                4,
                upd,
            );
        }
        if level > 2 {
            // NUM_BASE_LEVELS
            let base_range = level as i32 - 1 - 2;
            let br_ctx = get_br_ctx(&levels_buf, pos, bhl, tx_class) as usize;
            let mts = txs_ctx.min(3); // AOMMIN(txs_ctx, TX_32X32)
            let cdf_off = A_BR + ((mts * 2 + plane_type) * 21 + br_ctx) * 5;
            let mut idx = 0;
            while idx < 12 {
                // COEFF_BASE_RANGE
                let k = (base_range - idx).min(3);
                sym(enc, cdfs, cdf_off, k, 4, upd);
                if k < 3 {
                    break;
                }
                idx += 3; // BR_CDF_SIZE - 1
            }
        }
    }

    for c in 0..eob {
        let v = tcoeff[sc[c] as usize];
        let level = v.unsigned_abs();
        let sign = i32::from(v < 0);
        if level != 0 {
            if c == 0 {
                sym(
                    enc,
                    cdfs,
                    A_DC_SIGN + (plane_type * 3 + dc_sign_ctx) * 3,
                    sign,
                    2,
                    upd,
                );
            } else {
                write_bit(enc, sign);
            }
            if level as i32 > 12 + 2 {
                write_golomb(enc, level as i32 - 12 - 1 - 2);
            }
        }
    }
}

/// The complete txb writer: `txb_skip` flag, then (for luma, `plane_type == 0`,
/// when `signal_gate`) the `tx_type` via [`write_tx_type`], then the coefficient
/// payload — matching the full `av1_write_coeffs_txb` order. `ext_tx_cdf` is the
/// caller-selected `[intra|inter]_ext_tx_cdf[eset][square_tx_size][intra_dir]`
/// slot (separate from the coefficient `cdfs` arena); it is only touched on the
/// luma tx_type path. The `tx_type` context (`is_inter`, `reduced`,
/// `use_filter_intra`, `fi_mode`, `mode`, `signal_gate`) is what the encoder's
/// mbmi/frame state would supply.
#[allow(clippy::too_many_arguments)]
pub fn write_coeffs_txb_full(
    enc: &mut OdEcEnc,
    cdfs: &mut [u16],
    ext_tx_cdf: &mut [u16],
    tcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    plane_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    allow_update_cdf: bool,
    is_inter: bool,
    reduced: bool,
    use_filter_intra: bool,
    fi_mode: usize,
    mode: usize,
    signal_gate: bool,
) {
    let txs_ctx = txsize_entropy_ctx(tx_size);
    let upd = allow_update_cdf;
    sym(
        enc,
        cdfs,
        A_TXB_SKIP + (txs_ctx * 13 + txb_skip_ctx) * 3,
        (eob == 0) as i32,
        2,
        upd,
    );
    if eob == 0 {
        return;
    }
    if plane_type == 0 {
        // Only luma transmits tx_type.
        crate::txb::write_tx_type(
            enc,
            ext_tx_cdf,
            tx_size,
            is_inter,
            reduced,
            tx_type,
            use_filter_intra,
            fi_mode,
            mode,
            signal_gate,
        );
    }
    write_txb_body(
        enc,
        cdfs,
        tcoeff,
        eob,
        tx_size,
        tx_type,
        plane_type,
        dc_sign_ctx,
        upd,
    );
}

#[inline]
fn sym(enc: &mut OdEcEnc, cdfs: &mut [u16], off: usize, symb: i32, n: i32, upd: bool) {
    let cdf = &mut cdfs[off..off + n as usize + 1];
    if upd {
        write_symbol(enc, symb, cdf, n as usize);
    } else {
        // aom_write_cdf without the CDF adaptation.
        enc.encode_cdf_q15(symb, &cdf[..n as usize], n);
    }
}

/// `aom_write_bit`: `aom_write(w, bit, 128)`.
#[inline]
fn write_bit(enc: &mut OdEcEnc, bit: i32) {
    let p = ((0x7FFFFF - (128 << 15) + 128) >> 8) as u32;
    enc.encode_bool_q15(bit, p);
}

/// `write_golomb` (encodetxb.c).
fn write_golomb(enc: &mut OdEcEnc, level: i32) {
    let x = level + 1;
    let mut i = x;
    let mut length = 0;
    while i != 0 {
        i >>= 1;
        length += 1;
    }
    for _ in 0..length - 1 {
        write_bit(enc, 0);
    }
    for i in (0..length).rev() {
        write_bit(enc, (x >> i) & 0x01);
    }
}
