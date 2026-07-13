//! Per-coefficient RD-cost helpers used by the coefficient trellis
//! (`av1_optimize_txb`), from libaom `av1/encoder/txb_rdopt_utils.h`. Pure
//! integer functions over the `LV_MAP_COEFF_COST` tables; byte-exact vs C. These
//! are the building blocks the trellis calls per candidate coefficient level.

use crate::cost::{get_br_ctx_eob, golomb_cost, CoeffCostTables};
use crate::{get_br_ctx, TxClass};

const COST_LIT1: i32 = 1 << 9;
const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_RANGE: i32 = 12;
const LPS_STRIDE: usize = (COEFF_BASE_RANGE as usize + 1) * 2; // 26

/// `golomb_bits_cost[32]`.
#[rustfmt::skip]
const GOLOMB_BITS_COST: [i32; 32] = [
    0, 512, 512*3, 512*3, 512*5, 512*5, 512*5, 512*5,
    512*7, 512*7, 512*7, 512*7, 512*7, 512*7, 512*7, 512*7,
    512*9, 512*9, 512*9, 512*9, 512*9, 512*9, 512*9, 512*9,
    512*9, 512*9, 512*9, 512*9, 512*9, 512*9, 512*9, 512*9,
];
/// `golomb_cost_diff[32]`.
#[rustfmt::skip]
const GOLOMB_COST_DIFF: [i32; 32] = [
    0, 512, 512*2, 0, 512*2, 0, 0, 0, 512*2, 0, 0, 0, 0, 0, 0, 0,
    512*2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[inline]
fn lps_row<'a>(t: &'a CoeffCostTables, ctx: usize) -> &'a [i32] {
    &t.lps[ctx * LPS_STRIDE..ctx * LPS_STRIDE + LPS_STRIDE]
}

/// `get_br_cost`: `lps[base_range] + golomb`.
#[inline]
fn br_cost(level: i32, lps: &[i32]) -> i32 {
    let base_range = (level - 1 - NUM_BASE_LEVELS).min(COEFF_BASE_RANGE);
    lps[base_range as usize] + golomb_cost(level)
}

/// `get_br_cost_with_diff`: returns the base-range cost and accumulates the
/// trellis `diff` (cost of coding `level-1` instead of `level`).
pub fn br_cost_with_diff(level: i32, lps: &[i32], diff: &mut i32) -> i32 {
    let base_range = (level - 1 - NUM_BASE_LEVELS).min(COEFF_BASE_RANGE);
    let mut golomb_bits = 0;
    if level <= COEFF_BASE_RANGE + 1 + NUM_BASE_LEVELS {
        *diff += lps[(base_range + COEFF_BASE_RANGE + 1) as usize];
    }
    if level >= COEFF_BASE_RANGE + 1 + NUM_BASE_LEVELS {
        let r = level - COEFF_BASE_RANGE - NUM_BASE_LEVELS;
        if r < 32 {
            golomb_bits = GOLOMB_BITS_COST[r as usize];
            *diff += GOLOMB_COST_DIFF[r as usize];
        } else {
            golomb_bits = golomb_cost(level);
            *diff += if r & (r - 1) == 0 { 1024 } else { 0 };
        }
    }
    lps[base_range as usize] + golomb_bits
}

/// `get_two_coeff_cost_simple` (scan_idx not DC and not eob-1). Returns
/// `(cost, cost_low)` where `cost_low` is the cost of coding `abs_qc-1`.
#[allow(clippy::too_many_arguments)]
pub fn two_coeff_cost_simple(
    ci: usize,
    abs_qc: i32,
    coeff_ctx: usize,
    t: &CoeffCostTables,
    bhl: u32,
    tx_class: TxClass,
    levels: &[u8],
) -> (i32, i32) {
    let mut cost = t.base[coeff_ctx * 8 + abs_qc.min(3) as usize];
    let mut diff = 0;
    if abs_qc <= 3 {
        diff = t.base[coeff_ctx * 8 + (abs_qc + 4) as usize];
    }
    if abs_qc != 0 {
        cost += COST_LIT1;
        if abs_qc > NUM_BASE_LEVELS {
            let br_ctx = get_br_ctx(levels, ci, bhl, tx_class) as usize;
            let mut brdiff = 0;
            cost += br_cost_with_diff(abs_qc, lps_row(t, br_ctx), &mut brdiff);
            diff += brdiff;
        }
    }
    (cost, cost - diff)
}

/// `get_coeff_cost_eob`.
#[allow(clippy::too_many_arguments)]
pub fn coeff_cost_eob(
    ci: usize,
    abs_qc: i32,
    sign: usize,
    coeff_ctx: usize,
    dc_sign_ctx: usize,
    t: &CoeffCostTables,
    bhl: u32,
    tx_class: TxClass,
) -> i32 {
    let mut cost = t.base_eob[coeff_ctx * 3 + (abs_qc.min(3) - 1) as usize];
    if abs_qc != 0 {
        if ci == 0 {
            cost += t.dc_sign[dc_sign_ctx * 2 + sign];
        } else {
            cost += COST_LIT1;
        }
        if abs_qc > NUM_BASE_LEVELS {
            let br_ctx = get_br_ctx_eob(ci, bhl, tx_class);
            cost += br_cost(abs_qc, lps_row(t, br_ctx));
        }
    }
    cost
}

/// `get_coeff_cost_general`.
#[allow(clippy::too_many_arguments)]
pub fn coeff_cost_general(
    is_last: bool,
    ci: usize,
    abs_qc: i32,
    sign: usize,
    coeff_ctx: usize,
    dc_sign_ctx: usize,
    t: &CoeffCostTables,
    bhl: u32,
    tx_class: TxClass,
    levels: &[u8],
) -> i32 {
    let mut cost = if is_last {
        t.base_eob[coeff_ctx * 3 + (abs_qc.min(3) - 1) as usize]
    } else {
        t.base[coeff_ctx * 8 + abs_qc.min(3) as usize]
    };
    if abs_qc != 0 {
        if ci == 0 {
            cost += t.dc_sign[dc_sign_ctx * 2 + sign];
        } else {
            cost += COST_LIT1;
        }
        if abs_qc > NUM_BASE_LEVELS {
            let br_ctx = if is_last {
                get_br_ctx_eob(ci, bhl, tx_class)
            } else {
                get_br_ctx(levels, ci, bhl, tx_class) as usize
            };
            cost += br_cost(abs_qc, lps_row(t, br_ctx));
        }
    }
    cost
}
