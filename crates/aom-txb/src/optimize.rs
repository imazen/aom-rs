//! `av1_optimize_txb` (libaom `av1/encoder/txb_rdopt.c`): the coefficient
//! trellis — RD-optimal rounding of quantized coefficients (the core of
//! speed-0 transform-block encoding). Non-QM path (iqmatrix/qmatrix = NULL).
//! Byte-identical optimized qcoeff/dqcoeff/eob + rate vs C libaom.
//!
//! Every per-coefficient cost is one of the already-bit-exact helpers
//! (`coeff_cost_general`/`_eob`, `two_coeff_cost_simple`, `get_eob_cost` via the
//! cost tables); this module ports the trellis control flow (update_coeff_general
//! / _eob / _simple / update_skip) around them. `get_tx_type_cost` (plane-0
//! tx_type rate) is out of scope, added as 0 by both sides.

use crate::cost::CoeffCostTables;
use crate::trellis_cost::{coeff_cost_eob, coeff_cost_general, two_coeff_cost_simple};
use crate::{
    get_lower_levels_ctx, get_lower_levels_ctx_eob, get_lower_levels_ctx_general, padded_idx,
    txb_bhl, txb_high, txb_init_levels, txb_wide, TxClass, TX_PAD_2D, TX_TYPE_TO_CLASS,
};

const TX_2D: [i64; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];
const AV1_PROB_COST_SHIFT: i64 = 9;
const RDDIV_BITS: i64 = 7;
const INT8_MAX: i32 = 127;

/// `RDCOST(rdmult, rate, dist)`.
#[inline]
fn rdcost(rdmult: i64, rate: i64, dist: i64) -> i64 {
    ((rate * rdmult + (1 << (AV1_PROB_COST_SHIFT - 1))) >> AV1_PROB_COST_SHIFT) + (dist << RDDIV_BITS)
}

/// `get_coeff_dist` (non-QM): `((t - dq) << shift)^2`.
#[inline]
fn coeff_dist(tcoeff: i32, dqcoeff: i32, shift: i32) -> i64 {
    let diff = (tcoeff as i64 - dqcoeff as i64) * (1i64 << shift);
    diff * diff
}

/// `get_qc_dqc_low`: the "coded one lower" candidate.
#[inline]
fn qc_dqc_low(abs_qc: i32, sign: i32, dqv: i32, shift: i32) -> (i32, i32) {
    let abs_qc_low = abs_qc - 1;
    let qc_low = (-sign ^ abs_qc_low) + sign;
    let abs_dqc_low = (abs_qc_low * dqv) >> shift;
    let dqc_low = (-sign ^ abs_dqc_low) + sign;
    (qc_low, dqc_low)
}

/// Result of the trellis: the (possibly reduced) eob and the accumulated rate.
pub struct OptimizeResult {
    pub eob: usize,
    pub rate: i32,
}

/// `av1_optimize_txb`: optimize `qcoeff`/`dqcoeff` in place. `dequant[0]` is the
/// DC step, `dequant[1]` the AC step. Returns the new eob + rate.
#[allow(clippy::too_many_arguments)]
pub fn optimize_txb(
    tx_size: usize,
    tx_type: usize,
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    tcoeff: &[i32],
    eob_in: usize,
    dequant: [i16; 2],
    rdmult: i64,
    dc_sign_ctx: usize,
    txb_skip_ctx: usize,
    sharpness: i32,
    scan: &[i16],
    t: &CoeffCostTables,
) -> OptimizeResult {
    let tx_class = TX_TYPE_TO_CLASS[tx_type];
    let bhl = txb_bhl(tx_size);
    let width = txb_wide(tx_size);
    let height = txb_high(tx_size);
    let pels = TX_2D[tx_size];
    let shift = ((pels > 256) as i32) + ((pels > 1024) as i32);

    let mut eob = eob_in;
    let mut levels = [0u8; TX_PAD_2D];
    if eob > 1 {
        txb_init_levels(qcoeff, width, height, &mut levels);
    }
    let dqv = |ci: usize| -> i32 { dequant[(ci != 0) as usize] as i32 };
    let base0 = |ctx: usize| -> i32 { t.base[ctx * 8] };

    let non_skip_cost = t.txb_skip[txb_skip_ctx * 2];
    let skip_cost = t.txb_skip[txb_skip_ctx * 2 + 1];
    let mut accu_rate = crate::cost::eob_cost_pub(eob, t, tx_class);
    let mut accu_dist: i64 = 0;

    let mut si = eob as isize - 1;
    let ci0 = scan[si as usize] as usize;
    let qc0 = qcoeff[ci0];
    let abs_qc0 = qc0.abs();
    let sign0 = (qc0 < 0) as i32;
    let max_nz_num = 2;
    let mut nz_num = 1usize;
    let mut nz_ci = [ci0, 0usize, 0usize];

    if abs_qc0 >= 2 {
        update_coeff_general(
            &mut accu_rate, &mut accu_dist, si as usize, true, tx_size, tx_class, bhl, width, shift,
            rdmult, dc_sign_ctx, &dqv, scan, t, tcoeff, qcoeff, dqcoeff, &mut levels,
        );
        si -= 1;
    } else {
        let coeff_ctx = get_lower_levels_ctx_eob(bhl, width, si as usize) as usize;
        accu_rate += coeff_cost_eob(ci0, abs_qc0, sign0 as usize, coeff_ctx, dc_sign_ctx, t, bhl, tx_class);
        let (tqc, dqc) = (tcoeff[ci0], dqcoeff[ci0]);
        accu_dist += coeff_dist(tqc, dqc, shift) - coeff_dist(tqc, 0, shift);
        si -= 1;
    }

    // update_coeff_eob loop
    while si >= 0 && nz_num <= max_nz_num {
        let s = si as usize;
        let ci = scan[s] as usize;
        let qc = qcoeff[ci];
        let coeff_ctx = get_lower_levels_ctx(&levels, ci, bhl, tx_size, tx_class) as usize;
        if qc == 0 {
            accu_rate += base0(coeff_ctx);
            si -= 1;
            continue;
        }
        let v = dqv(scan[s] as usize);
        let mut lower_level = false;
        let abs_qc = qc.abs();
        let (tqc, dqc) = (tcoeff[ci], dqcoeff[ci]);
        let sign = (qc < 0) as i32;
        let dist0 = coeff_dist(tqc, 0, shift);
        let mut dist = coeff_dist(tqc, dqc, shift) - dist0;
        let mut rate = coeff_cost_general(false, ci, abs_qc, sign as usize, coeff_ctx, dc_sign_ctx, t, bhl, tx_class, &levels);
        let mut rd = rdcost(rdmult, (accu_rate + rate) as i64, accu_dist + dist);

        let (qc_low, dqc_low, abs_qc_low, dist_low, rate_low, rd_low);
        if abs_qc == 1 {
            abs_qc_low = 0;
            qc_low = 0;
            dqc_low = 0;
            dist_low = 0;
            rate_low = base0(coeff_ctx);
            rd_low = rdcost(rdmult, (accu_rate + rate_low) as i64, accu_dist);
        } else {
            let (ql, dql) = qc_dqc_low(abs_qc, sign, v, shift);
            qc_low = ql;
            dqc_low = dql;
            abs_qc_low = abs_qc - 1;
            dist_low = coeff_dist(tqc, dqc_low, shift) - dist0;
            rate_low = coeff_cost_general(false, ci, abs_qc_low, sign as usize, coeff_ctx, dc_sign_ctx, t, bhl, tx_class, &levels);
            rd_low = rdcost(rdmult, (accu_rate + rate_low) as i64, accu_dist + dist_low);
        }

        let mut lower_level_new_eob = false;
        let new_eob = s + 1;
        let coeff_ctx_new_eob = get_lower_levels_ctx_eob(bhl, width, s) as usize;
        let new_eob_cost = crate::cost::eob_cost_pub(new_eob, t, tx_class);
        let mut rate_coeff_eob =
            new_eob_cost + coeff_cost_eob(ci, abs_qc, sign as usize, coeff_ctx_new_eob, dc_sign_ctx, t, bhl, tx_class);
        let mut dist_new_eob = dist;
        let mut rd_new_eob = rdcost(rdmult, rate_coeff_eob as i64, dist_new_eob);
        if abs_qc_low > 0 {
            let rate_coeff_eob_low = new_eob_cost
                + coeff_cost_eob(ci, abs_qc_low, sign as usize, coeff_ctx_new_eob, dc_sign_ctx, t, bhl, tx_class);
            let rd_new_eob_low = rdcost(rdmult, rate_coeff_eob_low as i64, dist_low);
            if rd_new_eob_low < rd_new_eob {
                lower_level_new_eob = true;
                rd_new_eob = rd_new_eob_low;
                rate_coeff_eob = rate_coeff_eob_low;
                dist_new_eob = dist_low;
            }
        }
        let qc_threshold = if s <= 5 { 2 } else { 1 };
        let allow_lower_qc = if sharpness != 0 { abs_qc > qc_threshold } else { true };
        if allow_lower_qc && rd_low < rd {
            lower_level = true;
            rd = rd_low;
            rate = rate_low;
            dist = dist_low;
        }
        if (sharpness == 0 || new_eob >= 5) && rd_new_eob < rd {
            for &lc in nz_ci.iter().take(nz_num) {
                levels[padded_idx(lc, bhl)] = 0;
                qcoeff[lc] = 0;
                dqcoeff[lc] = 0;
            }
            eob = new_eob;
            nz_num = 0;
            accu_rate = rate_coeff_eob;
            accu_dist = dist_new_eob;
            lower_level = lower_level_new_eob;
        } else {
            accu_rate += rate;
            accu_dist += dist;
        }
        if lower_level {
            qcoeff[ci] = qc_low;
            dqcoeff[ci] = dqc_low;
            levels[padded_idx(ci, bhl)] = abs_qc_low.min(INT8_MAX) as u8;
        }
        if qcoeff[ci] != 0 {
            nz_ci[nz_num] = ci;
            nz_num += 1;
        }
        si -= 1;
    }

    // update_skip
    if si == -1 && nz_num <= max_nz_num && sharpness == 0 {
        let rd = rdcost(rdmult, (accu_rate + non_skip_cost) as i64, accu_dist);
        let rd_new_eob = rdcost(rdmult, skip_cost as i64, 0);
        if rd_new_eob < rd {
            for &ci in nz_ci.iter().take(nz_num) {
                qcoeff[ci] = 0;
                dqcoeff[ci] = 0;
            }
            accu_rate = 0;
            eob = 0;
        }
    }

    // update_coeff_simple loop
    while si >= 1 {
        let s = si as usize;
        let ci = scan[s] as usize;
        let qc = qcoeff[ci];
        let coeff_ctx = get_lower_levels_ctx(&levels, ci, bhl, tx_size, tx_class) as usize;
        if qc == 0 {
            accu_rate += base0(coeff_ctx);
            si -= 1;
            continue;
        }
        let abs_qc = qc.abs();
        let abs_tqc = tcoeff[ci].abs();
        let abs_dqc = dqcoeff[ci].abs();
        let (rate, rate_low) = two_coeff_cost_simple(ci, abs_qc, coeff_ctx, t, bhl, tx_class, &levels);
        if abs_dqc < abs_tqc {
            accu_rate += rate;
            si -= 1;
            continue;
        }
        let v = dqv(scan[s] as usize);
        let dist = coeff_dist(abs_tqc, abs_dqc, shift);
        let rd = rdcost(rdmult, rate as i64, dist);
        let abs_qc_low = abs_qc - 1;
        let abs_dqc_low = (abs_qc_low * v) >> shift;
        let dist_low = coeff_dist(abs_tqc, abs_dqc_low, shift);
        let rd_low = rdcost(rdmult, rate_low as i64, dist_low);
        let allow_lower_qc = if sharpness != 0 { abs_qc > 1 } else { true };
        if rd_low < rd && allow_lower_qc {
            let sign = (qc < 0) as i32;
            qcoeff[ci] = (-sign ^ abs_qc_low) + sign;
            dqcoeff[ci] = (-sign ^ abs_dqc_low) + sign;
            levels[padded_idx(ci, bhl)] = abs_qc_low.min(INT8_MAX) as u8;
            accu_rate += rate_low;
        } else {
            accu_rate += rate;
        }
        si -= 1;
    }

    // DC position
    if si == 0 {
        let mut dummy = 0i64;
        update_coeff_general(
            &mut accu_rate, &mut dummy, 0, false, tx_size, tx_class, bhl, width, shift, rdmult,
            dc_sign_ctx, &dqv, scan, t, tcoeff, qcoeff, dqcoeff, &mut levels,
        );
    }

    if eob == 0 {
        accu_rate += skip_cost;
    } else {
        accu_rate += non_skip_cost; // + tx_type_cost (out of scope)
    }
    OptimizeResult { eob, rate: accu_rate }
}

/// `update_coeff_general` (used at the eob coefficient and the DC position).
#[allow(clippy::too_many_arguments)]
fn update_coeff_general(
    accu_rate: &mut i32,
    accu_dist: &mut i64,
    si: usize,
    is_last: bool,
    tx_size: usize,
    tx_class: TxClass,
    bhl: u32,
    width: usize,
    shift: i32,
    rdmult: i64,
    dc_sign_ctx: usize,
    dqv: &dyn Fn(usize) -> i32,
    scan: &[i16],
    t: &CoeffCostTables,
    tcoeff: &[i32],
    qcoeff: &mut [i32],
    dqcoeff: &mut [i32],
    levels: &mut [u8],
) {
    let ci = scan[si] as usize;
    let qc = qcoeff[ci];
    let coeff_ctx =
        get_lower_levels_ctx_general(is_last, si, bhl, width, levels, ci, tx_size, tx_class) as usize;
    if qc == 0 {
        *accu_rate += t.base[coeff_ctx * 8];
        return;
    }
    let v = dqv(scan[si] as usize);
    let sign = (qc < 0) as i32;
    let abs_qc = qc.abs();
    let (tqc, dqc) = (tcoeff[ci], dqcoeff[ci]);
    let dist = coeff_dist(tqc, dqc, shift);
    let dist0 = coeff_dist(tqc, 0, shift);
    let rate = coeff_cost_general(is_last, ci, abs_qc, sign as usize, coeff_ctx, dc_sign_ctx, t, bhl, tx_class, levels);
    let rd = rdcost(rdmult, rate as i64, dist);

    let (qc_low, dqc_low, abs_qc_low, dist_low, rate_low);
    if abs_qc == 1 {
        abs_qc_low = 0;
        qc_low = 0;
        dqc_low = 0;
        dist_low = dist0;
        rate_low = t.base[coeff_ctx * 8];
    } else {
        let (ql, dql) = qc_dqc_low(abs_qc, sign, v, shift);
        qc_low = ql;
        dqc_low = dql;
        abs_qc_low = abs_qc - 1;
        dist_low = coeff_dist(tqc, dqc_low, shift);
        rate_low = coeff_cost_general(is_last, ci, abs_qc_low, sign as usize, coeff_ctx, dc_sign_ctx, t, bhl, tx_class, levels);
    }
    let rd_low = rdcost(rdmult, rate_low as i64, dist_low);
    if rd_low < rd {
        qcoeff[ci] = qc_low;
        dqcoeff[ci] = dqc_low;
        levels[padded_idx(ci, bhl)] = abs_qc_low.min(INT8_MAX) as u8;
        *accu_rate += rate_low;
        *accu_dist += dist_low - dist0;
    } else {
        *accu_rate += rate;
        *accu_dist += dist - dist0;
    }
}
