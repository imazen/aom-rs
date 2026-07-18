//! Differential harness for the adaptive dead-zone "b" quantizer
//! (`--quant-b-adapt`) vs C libaom v3.14.1:
//! [`aom_quantize_b_adaptive_helper`] / [`aom_highbd_quantize_b_adaptive_helper`]
//! must produce byte-identical qcoeff / dqcoeff / eob to the exported
//! `aom_quantize_b_adaptive_helper_c` / `aom_highbd_quantize_b_adaptive_helper_c`
//! across log_scale {0,1,2} (32x32 / 64x64), bit depth {lowbd, highbd},
//! no-matrix AND quant-matrix, and both the large-coeff and small-coeff (sparse
//! single-±1) regimes — the latter exercises the `SKIP_EOB_FACTOR_ADJUST` tail.

use aom_quant::{aom_highbd_quantize_b_adaptive_helper, aom_quantize_b_adaptive_helper};
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
    /// Large-coeff regime: full ±2^18 dynamic range.
    fn coeff_big(&mut self) -> i32 {
        (self.next() % (1 << 19)) as i32 - (1 << 18)
    }
    /// Small-coeff regime: mostly deep in the dead-zone with the occasional
    /// near-boundary value — drives sparse eobs + lone ±1 (the skip tail).
    fn coeff_small(&mut self) -> i32 {
        (self.next() % 121) as i32 - 60
    }
    fn pos_i16(&mut self, lo: i32, hi: i32) -> i16 {
        (lo + (self.next() % (hi - lo) as u64) as i32) as i16
    }
    fn qm(&mut self) -> u8 {
        (1 + self.next() % 32) as u8 // 1..=32 (qm_val_t range)
    }
}

fn perm(rng: &mut Rng, n: usize) -> Vec<i16> {
    let mut v: Vec<i16> = (0..n as i16).collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

struct Counts {
    eob_zero: u64,
    eob_pos: u64,
}

#[allow(clippy::too_many_arguments)]
fn check(
    rng: &mut Rng,
    hbd: bool,
    use_qm: bool,
    small: bool,
    log_scale: i32,
    n: usize,
    cnt: &mut Counts,
) {
    let scan = perm(rng, n);
    let coeff: Vec<i32> = (0..n)
        .map(|_| {
            if small {
                rng.coeff_small()
            } else {
                rng.coeff_big()
            }
        })
        .collect();
    // Small-coeff regime pairs with a larger dequant so most coeffs land in the
    // dead-zone (sparse survivors → the skip tail is reachable).
    let dq_hi = if small { 4000 } else { 8000 };
    let zbin = [rng.pos_i16(1, 1000), rng.pos_i16(1, 1000)];
    let round = [rng.pos_i16(1, 2000), rng.pos_i16(1, 2000)];
    let quant = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
    let quant_shift = [rng.pos_i16(1, 32767), rng.pos_i16(1, 32767)];
    let dequant = [rng.pos_i16(1, dq_hi), rng.pos_i16(1, dq_hi)];
    let (qm, iqm): (Option<Vec<u8>>, Option<Vec<u8>>) = if use_qm {
        (
            Some((0..n).map(|_| rng.qm()).collect()),
            Some((0..n).map(|_| rng.qm()).collect()),
        )
    } else {
        (None, None)
    };
    let qm_ref = qm.as_deref();
    let iqm_ref = iqm.as_deref();

    let mut q_got = vec![0i32; n];
    let mut dq_got = vec![0i32; n];
    let eob_got = if hbd {
        aom_highbd_quantize_b_adaptive_helper(
            &zbin,
            &round,
            &quant,
            &quant_shift,
            &dequant,
            log_scale,
            qm_ref,
            iqm_ref,
            &scan,
            &coeff,
            &mut q_got,
            &mut dq_got,
        )
    } else {
        aom_quantize_b_adaptive_helper(
            &zbin,
            &round,
            &quant,
            &quant_shift,
            &dequant,
            log_scale,
            qm_ref,
            iqm_ref,
            &scan,
            &coeff,
            &mut q_got,
            &mut dq_got,
        )
    };

    let (q_want, dq_want, eob_want) = c::ref_quantize_b_adaptive(
        hbd,
        log_scale,
        &coeff,
        &zbin,
        &round,
        &quant,
        &quant_shift,
        &dequant,
        qm_ref,
        iqm_ref,
        &scan,
    );

    assert_eq!(
        eob_got, eob_want,
        "eob mismatch hbd={hbd} qm={use_qm} small={small} log_scale={log_scale} n={n}"
    );
    assert_eq!(q_got, q_want, "qcoeff mismatch hbd={hbd} qm={use_qm} small={small} log_scale={log_scale} n={n}\ncoeff={coeff:?}");
    assert_eq!(
        dq_got, dq_want,
        "dqcoeff mismatch hbd={hbd} qm={use_qm} small={small} log_scale={log_scale} n={n}"
    );

    if eob_got == 0 {
        cnt.eob_zero += 1;
    } else {
        cnt.eob_pos += 1;
    }
}

#[test]
fn quantize_b_adaptive_differential_fuzz() {
    let mut rng = Rng(0x_ada9_71be_b0b0_5555);
    let sizes = [16usize, 64, 256, 1024];
    let mut cnt = Counts {
        eob_zero: 0,
        eob_pos: 0,
    };
    for &hbd in &[false, true] {
        for &use_qm in &[false, true] {
            for &small in &[false, true] {
                for log_scale in 0..=2i32 {
                    for &n in &sizes {
                        for _ in 0..2_000 {
                            check(&mut rng, hbd, use_qm, small, log_scale, n, &mut cnt);
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "quantize_b_adaptive: {} eob==0 cells, {} eob>0 cells",
        cnt.eob_zero, cnt.eob_pos
    );
    // Anti-vacuity: the sweep must exercise both the all-zeroed (dead-zone /
    // skip-tail) and the surviving-coefficient paths.
    assert!(
        cnt.eob_zero > 0,
        "no fully-zeroed blocks — skip/dead-zone path unexercised"
    );
    assert!(
        cnt.eob_pos > 0,
        "no surviving coefficients — quant path unexercised"
    );
}
