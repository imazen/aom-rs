//! `av1_cost_coeffs_txb` (libaom `av1/encoder/txb_rdopt.c`,
//! `warehouse_efficients_txb`): the RD-cost twin of `write_coeffs_txb` — the
//! single hottest speed-0 function (evaluated for every candidate txb during
//! mode / tx-type search). Same symbol chain, but sums precomputed cost-table
//! entries instead of emitting bits. Byte-exact integer result vs C libaom.
//!
//! Cost tables (`LV_MAP_COEFF_COST` / `LV_MAP_EOB_COST`) are inputs — derived
//! from the frame's CDFs by a separate step (`av1_cost_tokens_from_cdf`), out of
//! scope here. `av1_cost_literal(n) = n << 9`. `get_tx_type_cost` (plane-0
//! tx_type) is out of scope, matching `write_coeffs_txb`.

use crate::scan::scan;
use crate::{
    get_br_ctx, get_eob_pos_token, get_nz_map_contexts, txb_bhl, txb_high, txb_wide,
    txb_init_levels, TxClass, TX_PAD_2D, TX_TYPE_TO_CLASS, EOB_OFFSET_BITS,
};

const NUM_BASE_LEVELS: u32 = 2;
const COEFF_BASE_RANGE: i32 = 12;
const COST_LIT1: i32 = 1 << 9; // av1_cost_literal(1)

/// Borrowed cost tables for one `(txs_ctx, plane_type)` (`LV_MAP_COEFF_COST`)
/// plus the eob costs (`LV_MAP_EOB_COST`), flat row-major as in C.
pub struct CoeffCostTables<'a> {
    /// `txb_skip_cost[13][2]`
    pub txb_skip: &'a [i32],
    /// `base_eob_cost[4][3]`
    pub base_eob: &'a [i32],
    /// `base_cost[42][8]`
    pub base: &'a [i32],
    /// `eob_extra_cost[9][2]`
    pub eob_extra: &'a [i32],
    /// `dc_sign_cost[3][2]`
    pub dc_sign: &'a [i32],
    /// `lps_cost[21][26]`
    pub lps: &'a [i32],
    /// `eob_cost[2][11]`
    pub eob: &'a [i32],
}

/// `get_br_ctx_eob` (txb_common.h).
#[inline]
pub(crate) fn get_br_ctx_eob(c: usize, bhl: u32, tx_class: TxClass) -> usize {
    if c == 0 {
        return 0;
    }
    let col = c >> bhl;
    let row = c - (col << bhl);
    let hit = match tx_class {
        TxClass::TwoD => row < 2 && col < 2,
        TxClass::Horiz => col == 0,
        TxClass::Vert => row == 0,
    };
    if hit {
        7
    } else {
        14
    }
}

/// `get_golomb_cost`.
#[inline]
pub(crate) fn golomb_cost(abs_qc: i32) -> i32 {
    if abs_qc >= 1 + NUM_BASE_LEVELS as i32 + COEFF_BASE_RANGE {
        let r = abs_qc - COEFF_BASE_RANGE - NUM_BASE_LEVELS as i32;
        let length = (31 - (r as u32).leading_zeros()) as i32 + 1; // get_msb(r)+1
        COST_LIT1 * (2 * length - 1)
    } else {
        0
    }
}

/// `get_br_cost`: `lps[base_range] + golomb`.
#[inline]
fn br_cost(level: i32, lps: &[i32]) -> i32 {
    let base_range = (level - 1 - NUM_BASE_LEVELS as i32).min(COEFF_BASE_RANGE);
    lps[base_range as usize] + golomb_cost(level)
}

const LPS_STRIDE: usize = (COEFF_BASE_RANGE as usize + 1) * 2; // 26

/// `get_eob_cost` (crate-visible wrapper for the trellis).
pub(crate) fn eob_cost_pub(eob: usize, t: &CoeffCostTables, tx_class: TxClass) -> i32 { eob_cost(eob, t, tx_class) }

/// `get_eob_cost`.
fn eob_cost(eob: usize, t: &CoeffCostTables, tx_class: TxClass) -> i32 {
    let (eob_pt, eob_extra) = get_eob_pos_token(eob as i32);
    let eob_multi_ctx = if tx_class == TxClass::TwoD { 0 } else { 1 };
    let mut cost = t.eob[eob_multi_ctx * 11 + (eob_pt as usize - 1)];
    let offset_bits = EOB_OFFSET_BITS[eob_pt as usize] as i32;
    if offset_bits > 0 {
        let eob_ctx = (eob_pt - 3) as usize;
        let eob_shift = offset_bits - 1;
        let bit = usize::from(eob_extra & (1 << eob_shift) != 0);
        cost += t.eob_extra[eob_ctx * 2 + bit];
        if offset_bits > 1 {
            cost += COST_LIT1 * (offset_bits - 1);
        }
    }
    cost
}

/// `av1_cost_coeffs_txb`: RD rate (in `1<<9`-scaled bits) of coding this txb's
/// quantized coefficients. `qcoeff` is the transposed-layout coefficient block.
pub fn cost_coeffs_txb(
    qcoeff: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    t: &CoeffCostTables,
) -> i32 {
    if eob == 0 {
        return t.txb_skip[txb_skip_ctx * 2 + 1];
    }
    let tx_class = TX_TYPE_TO_CLASS[tx_type];
    let bhl = txb_bhl(tx_size);
    let width = txb_wide(tx_size);
    let height = txb_high(tx_size);
    let sc = scan(tx_size, tx_type);

    let mut levels_buf = [0u8; TX_PAD_2D];
    if eob > 1 {
        txb_init_levels(qcoeff, width, height, &mut levels_buf);
    }

    let mut cost = t.txb_skip[txb_skip_ctx * 2]; // [txb_skip_ctx][0]
    cost += eob_cost(eob, t, tx_class);

    let mut coeff_contexts = [0i8; 32 * 32];
    get_nz_map_contexts(&levels_buf, sc, eob, tx_size, tx_class, &mut coeff_contexts);

    // c == eob - 1 (the EOB coefficient)
    let mut c = eob - 1;
    {
        let pos = sc[c] as usize;
        let v = qcoeff[pos];
        if v != 0 {
            let level = v.unsigned_abs() as i32;
            let coeff_ctx = coeff_contexts[pos] as usize;
            cost += t.base_eob[coeff_ctx * 3 + (level.min(3) - 1) as usize];
            if level > NUM_BASE_LEVELS as i32 {
                let ctx = get_br_ctx_eob(pos, bhl, tx_class);
                cost += br_cost(level, &t.lps[ctx * LPS_STRIDE..ctx * LPS_STRIDE + LPS_STRIDE]);
            }
            if c != 0 {
                cost += COST_LIT1;
            } else {
                let sign01 = usize::from(v < 0);
                cost += t.dc_sign[dc_sign_ctx * 2 + sign01];
                return cost;
            }
        }
    }

    // c from eob-2 down to 1
    c = eob.wrapping_sub(2);
    while (c as isize) >= 1 {
        let pos = sc[c] as usize;
        let coeff_ctx = coeff_contexts[pos] as usize;
        let v = qcoeff[pos];
        if v == 0 {
            cost += t.base[coeff_ctx * 8];
        } else {
            let level = v.unsigned_abs() as i32;
            cost += t.base[coeff_ctx * 8 + level.min(3) as usize];
            cost += COST_LIT1;
            if level > NUM_BASE_LEVELS as i32 {
                let ctx = get_br_ctx(&levels_buf, pos, bhl, tx_class) as usize;
                cost += br_cost(level, &t.lps[ctx * LPS_STRIDE..ctx * LPS_STRIDE + LPS_STRIDE]);
            }
        }
        c -= 1;
    }

    // c == 0 (DC)
    {
        let pos = sc[0] as usize;
        let v = qcoeff[pos];
        let coeff_ctx = coeff_contexts[pos] as usize;
        if v == 0 {
            cost += t.base[coeff_ctx * 8];
        } else {
            let level = v.unsigned_abs() as i32;
            cost += t.base[coeff_ctx * 8 + level.min(3) as usize];
            let sign01 = usize::from(v < 0);
            cost += t.dc_sign[dc_sign_ctx * 2 + sign01];
            if level > NUM_BASE_LEVELS as i32 {
                let ctx = get_br_ctx(&levels_buf, pos, bhl, tx_class) as usize;
                cost += br_cost(level, &t.lps[ctx * LPS_STRIDE..ctx * LPS_STRIDE + LPS_STRIDE]);
            }
        }
    }
    cost
}
