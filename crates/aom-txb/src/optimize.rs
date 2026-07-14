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
    ((rate * rdmult + (1 << (AV1_PROB_COST_SHIFT - 1))) >> AV1_PROB_COST_SHIFT)
        + (dist << RDDIV_BITS)
}

const AOM_QM_BITS: i32 = 5;

/// `get_dqv`: per-position dequant step. With `iqmatrix`, folds the inverse
/// quant-matrix weight `(iqm[ci]*dqv + 16) >> 5`; otherwise `dequant[ci!=0]`.
#[inline]
pub(crate) fn get_dqv(dequant: [i16; 2], coeff_idx: usize, iqmatrix: Option<&[u8]>) -> i32 {
    let dqv = dequant[(coeff_idx != 0) as usize] as i32;
    match iqmatrix {
        Some(iqm) => (iqm[coeff_idx] as i32 * dqv + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS,
        None => dqv,
    }
}

/// `get_coeff_dist`: squared-error distortion `((t - dq) << shift)^2`. With
/// `qmatrix`, weights the diff by `qm[ci]` then rounds `>> (2*AOM_QM_BITS)`.
#[inline]
pub(crate) fn get_coeff_dist(
    tcoeff: i32,
    dqcoeff: i32,
    shift: i32,
    qmatrix: Option<&[u8]>,
    coeff_idx: usize,
) -> i64 {
    let diff = (tcoeff as i64 - dqcoeff as i64) * (1i64 << shift);
    match qmatrix {
        None => diff * diff,
        Some(qm) => {
            let diff = diff * qm[coeff_idx] as i64;
            (diff * diff + (1 << (2 * AOM_QM_BITS - 1))) >> (2 * AOM_QM_BITS)
        }
    }
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

/// `av1_optimize_txb` (non-QM): optimize `qcoeff`/`dqcoeff` in place. `dequant[0]`
/// is the DC step, `dequant[1]` the AC step. Returns the new eob + rate.
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
    optimize_txb_core(
        tx_size,
        tx_type,
        qcoeff,
        dqcoeff,
        tcoeff,
        eob_in,
        dequant,
        rdmult,
        dc_sign_ctx,
        txb_skip_ctx,
        sharpness,
        scan,
        t,
        None,
        None,
    )
}

/// `av1_optimize_txb` *with* a quant matrix: `iqmatrix` folds into the
/// per-position dequant (`get_dqv`), `qmatrix` into the distortion
/// (`get_coeff_dist`). Both are indexed by raster position (length = block area).
#[allow(clippy::too_many_arguments)]
pub fn optimize_txb_qm(
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
    iqmatrix: &[u8],
    qmatrix: &[u8],
) -> OptimizeResult {
    optimize_txb_core(
        tx_size,
        tx_type,
        qcoeff,
        dqcoeff,
        tcoeff,
        eob_in,
        dequant,
        rdmult,
        dc_sign_ctx,
        txb_skip_ctx,
        sharpness,
        scan,
        t,
        Some(iqmatrix),
        Some(qmatrix),
    )
}

#[allow(clippy::too_many_arguments)]
fn optimize_txb_core(
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
    iqmatrix: Option<&[u8]>,
    qmatrix: Option<&[u8]>,
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
    let dqv = |ci: usize| -> i32 { get_dqv(dequant, ci, iqmatrix) };
    let cdist =
        |tqc: i32, dqc: i32, ci: usize| -> i64 { get_coeff_dist(tqc, dqc, shift, qmatrix, ci) };
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
            &mut accu_rate,
            &mut accu_dist,
            si as usize,
            true,
            tx_size,
            tx_class,
            bhl,
            width,
            shift,
            rdmult,
            dc_sign_ctx,
            &dqv,
            scan,
            t,
            tcoeff,
            qcoeff,
            dqcoeff,
            &mut levels,
            qmatrix,
        );
        si -= 1;
    } else {
        let coeff_ctx = get_lower_levels_ctx_eob(bhl, width, si as usize) as usize;
        accu_rate += coeff_cost_eob(
            ci0,
            abs_qc0,
            sign0 as usize,
            coeff_ctx,
            dc_sign_ctx,
            t,
            bhl,
            tx_class,
        );
        let (tqc, dqc) = (tcoeff[ci0], dqcoeff[ci0]);
        accu_dist += cdist(tqc, dqc, ci0) - cdist(tqc, 0, ci0);
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
        let dist0 = cdist(tqc, 0, ci);
        let mut dist = cdist(tqc, dqc, ci) - dist0;
        let mut rate = coeff_cost_general(
            false,
            ci,
            abs_qc,
            sign as usize,
            coeff_ctx,
            dc_sign_ctx,
            t,
            bhl,
            tx_class,
            &levels,
        );
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
            dist_low = cdist(tqc, dqc_low, ci) - dist0;
            rate_low = coeff_cost_general(
                false,
                ci,
                abs_qc_low,
                sign as usize,
                coeff_ctx,
                dc_sign_ctx,
                t,
                bhl,
                tx_class,
                &levels,
            );
            rd_low = rdcost(rdmult, (accu_rate + rate_low) as i64, accu_dist + dist_low);
        }

        let mut lower_level_new_eob = false;
        let new_eob = s + 1;
        let coeff_ctx_new_eob = get_lower_levels_ctx_eob(bhl, width, s) as usize;
        let new_eob_cost = crate::cost::eob_cost_pub(new_eob, t, tx_class);
        let mut rate_coeff_eob = new_eob_cost
            + coeff_cost_eob(
                ci,
                abs_qc,
                sign as usize,
                coeff_ctx_new_eob,
                dc_sign_ctx,
                t,
                bhl,
                tx_class,
            );
        let mut dist_new_eob = dist;
        let mut rd_new_eob = rdcost(rdmult, rate_coeff_eob as i64, dist_new_eob);
        if abs_qc_low > 0 {
            let rate_coeff_eob_low = new_eob_cost
                + coeff_cost_eob(
                    ci,
                    abs_qc_low,
                    sign as usize,
                    coeff_ctx_new_eob,
                    dc_sign_ctx,
                    t,
                    bhl,
                    tx_class,
                );
            let rd_new_eob_low = rdcost(rdmult, rate_coeff_eob_low as i64, dist_low);
            if rd_new_eob_low < rd_new_eob {
                lower_level_new_eob = true;
                rd_new_eob = rd_new_eob_low;
                rate_coeff_eob = rate_coeff_eob_low;
                dist_new_eob = dist_low;
            }
        }
        let qc_threshold = if s <= 5 { 2 } else { 1 };
        let allow_lower_qc = if sharpness != 0 {
            abs_qc > qc_threshold
        } else {
            true
        };
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
        let (rate, rate_low) =
            two_coeff_cost_simple(ci, abs_qc, coeff_ctx, t, bhl, tx_class, &levels);
        if abs_dqc < abs_tqc {
            accu_rate += rate;
            si -= 1;
            continue;
        }
        let v = dqv(scan[s] as usize);
        let dist = cdist(abs_tqc, abs_dqc, ci);
        let rd = rdcost(rdmult, rate as i64, dist);
        let abs_qc_low = abs_qc - 1;
        let abs_dqc_low = (abs_qc_low * v) >> shift;
        let dist_low = cdist(abs_tqc, abs_dqc_low, ci);
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
            &mut accu_rate,
            &mut dummy,
            0,
            false,
            tx_size,
            tx_class,
            bhl,
            width,
            shift,
            rdmult,
            dc_sign_ctx,
            &dqv,
            scan,
            t,
            tcoeff,
            qcoeff,
            dqcoeff,
            &mut levels,
            qmatrix,
        );
    }

    if eob == 0 {
        accu_rate += skip_cost;
    } else {
        accu_rate += non_skip_cost; // + tx_type_cost (out of scope)
    }
    OptimizeResult {
        eob,
        rate: accu_rate,
    }
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
    qmatrix: Option<&[u8]>,
) {
    let ci = scan[si] as usize;
    let qc = qcoeff[ci];
    let coeff_ctx =
        get_lower_levels_ctx_general(is_last, si, bhl, width, levels, ci, tx_size, tx_class)
            as usize;
    if qc == 0 {
        *accu_rate += t.base[coeff_ctx * 8];
        return;
    }
    let v = dqv(scan[si] as usize);
    let sign = (qc < 0) as i32;
    let abs_qc = qc.abs();
    let (tqc, dqc) = (tcoeff[ci], dqcoeff[ci]);
    let dist = get_coeff_dist(tqc, dqc, shift, qmatrix, ci);
    let dist0 = get_coeff_dist(tqc, 0, shift, qmatrix, ci);
    let rate = coeff_cost_general(
        is_last,
        ci,
        abs_qc,
        sign as usize,
        coeff_ctx,
        dc_sign_ctx,
        t,
        bhl,
        tx_class,
        levels,
    );
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
        dist_low = get_coeff_dist(tqc, dqc_low, shift, qmatrix, ci);
        rate_low = coeff_cost_general(
            is_last,
            ci,
            abs_qc_low,
            sign as usize,
            coeff_ctx,
            dc_sign_ctx,
            t,
            bhl,
            tx_class,
            levels,
        );
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

#[cfg(test)]
mod qm_primitive_tests {
    //! Focused differential for the two QM primitives against the real C static
    //! inlines (`get_dqv` / `get_coeff_dist` via the shim). Inputs are bounded to
    //! the non-overflow regime real encoder coefficients live in (the squared
    //! `diff*qm` must stay within i64), matching C's own assumption.
    use super::{get_coeff_dist, get_dqv};
    use aom_sys_ref as c;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo) as u64) as i64
        }
    }

    #[test]
    fn get_dqv_matches_c() {
        let mut rng = Rng(0xd90c_0de0_0000_0001);
        for _ in 0..200_000 {
            let dequant = [rng.range(1, 8000) as i16, rng.range(1, 8000) as i16];
            let ci = rng.range(0, 1024) as usize;
            let iqm: Vec<u8> = (0..1024).map(|_| rng.range(1, 243) as u8).collect();
            // None (flat) and Some (weighted).
            assert_eq!(
                get_dqv(dequant, ci, None),
                c::ref_get_dqv(&dequant, ci, None)
            );
            assert_eq!(
                get_dqv(dequant, ci, Some(&iqm)),
                c::ref_get_dqv(&dequant, ci, Some(&iqm)),
                "get_dqv qm dequant={dequant:?} ci={ci} iqm[ci]={}",
                iqm[ci]
            );
        }
    }

    #[test]
    fn get_coeff_dist_matches_c() {
        let mut rng = Rng(0xd90c_0de0_0000_0002);
        for _ in 0..200_000 {
            // Bound |tcoeff-dqcoeff| so (diff<<shift)*qm stays < 2^31 => diff^2 < 2^62.
            let tcoeff = rng.range(-(1 << 19), 1 << 19) as i32;
            let dqcoeff = rng.range(-(1 << 19), 1 << 19) as i32;
            let shift = rng.range(0, 3) as i32;
            let ci = rng.range(0, 1024) as usize;
            let qm: Vec<u8> = (0..1024).map(|_| rng.range(1, 243) as u8).collect();
            assert_eq!(
                get_coeff_dist(tcoeff, dqcoeff, shift, None, ci),
                c::ref_get_coeff_dist(tcoeff, dqcoeff, shift, None, ci)
            );
            assert_eq!(
                get_coeff_dist(tcoeff, dqcoeff, shift, Some(&qm), ci),
                c::ref_get_coeff_dist(tcoeff, dqcoeff, shift, Some(&qm), ci),
                "coeff_dist qm t={tcoeff} dq={dqcoeff} shift={shift} qm[ci]={}",
                qm[ci]
            );
        }
    }
}
