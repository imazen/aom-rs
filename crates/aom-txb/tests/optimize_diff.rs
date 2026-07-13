//! Differential harness for `av1_optimize_txb` (the coefficient trellis, non-QM
//! path) vs C libaom: identical random (tcoeff, qcoeff, dqcoeff, eob, cost
//! tables, dequant, rdmult, dc_sign/skip ctx, sharpness) inputs must produce
//! byte-identical optimized qcoeff / dqcoeff, the same reduced eob, and the same
//! rate. `get_tx_type_cost` (plane-0 tx_type rate) is out of scope on both sides.

use aom_sys_ref as c;
use aom_txb::{optimize_txb, scan, txb_high, txb_wide, CoeffCostTables};

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
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
    fn cost(&mut self) -> i32 {
        self.range(0, 20 << 9) as i32
    }
}

fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

/// Build a self-consistent (tcoeff, qcoeff, dqcoeff, eob) block: qcoeff is the
/// quantized value, dqcoeff = qcoeff*dqv>>shift with the right sign, tcoeff a
/// nearby transform coefficient. `scan[eob-1]` is forced nonzero.
fn gen_block(
    rng: &mut Rng,
    sc: &[i16],
    area: usize,
    dqv: [i32; 2],
    shift: i32,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, usize) {
    let mut qcoeff = vec![0i32; area];
    let mut dqcoeff = vec![0i32; area];
    let mut tcoeff = vec![0i32; area];
    let eob = rng.range(1, area as u32 + 1) as usize;
    for i in 0..eob {
        let pos = sc[i] as usize;
        let nz = i == eob - 1 || rng.range(0, 10) >= 4;
        if nz {
            let mag = match rng.range(0, 10) {
                0..=5 => rng.range(1, 3) as i32,
                6..=8 => rng.range(1, 20) as i32,
                _ => rng.range(1, 400) as i32,
            };
            let sign = if rng.next() & 1 == 1 { -1 } else { 1 };
            let q = sign * mag;
            qcoeff[pos] = q;
            let d = dqv[(pos != 0) as usize];
            let abs_dq = (mag * d) >> shift;
            dqcoeff[pos] = sign * abs_dq;
            // tcoeff near dqcoeff (so both keep-low and drop are exercised).
            let jitter = rng.range(0, (d.max(2)) as u32) as i32 - d / 2;
            tcoeff[pos] = sign * abs_dq + jitter;
        }
    }
    (tcoeff, qcoeff, dqcoeff, eob)
}

#[test]
fn optimize_txb_round_trip_identical() {
    let mut rng = Rng(0x_0971_1112_e57a_b1e5);
    const TX_TYPES: [usize; 7] = [0, 3, 9, 10, 14, 11, 15];
    for tx_size in 0..19usize {
        let area = txb_wide(tx_size) * txb_high(tx_size);
        for &tx_type in &TX_TYPES {
            for _ in 0..60 {
                let txb_skip = tbl(&mut rng, 13 * 2);
                let base_eob = tbl(&mut rng, 4 * 3);
                let base = tbl(&mut rng, 42 * 8);
                let eob_extra = tbl(&mut rng, 9 * 2);
                let dc_sign = tbl(&mut rng, 3 * 2);
                let lps = tbl(&mut rng, 21 * 26);
                let eob_c = tbl(&mut rng, 2 * 11);

                let dequant: [i16; 2] =
                    [rng.range(4, 4000) as i16, rng.range(4, 4000) as i16];
                let dqv = [dequant[0] as i32, dequant[1] as i32];
                let pels: [i64; 19] = [
                    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256,
                    256, 1024, 1024,
                ];
                let shift = (pels[tx_size] > 256) as i32 + (pels[tx_size] > 1024) as i32;

                let sc = scan(tx_size, tx_type);
                let (tcoeff, qcoeff0, dqcoeff0, eob) = gen_block(&mut rng, sc, area, dqv, shift);
                let rdmult = rng.range(1, 1 << 20) as i64;
                let dc_sign_ctx = rng.range(0, 3) as usize;
                let txb_skip_ctx = rng.range(0, 13) as usize;
                let sharpness = rng.range(0, 8) as i32;

                // C reference (mutates copies).
                let mut qc_c = qcoeff0.clone();
                let mut dqc_c = dqcoeff0.clone();
                let (eob_wc, rate_wc) = c::ref_optimize_txb(
                    tx_size, tx_type, &mut qc_c, &mut dqc_c, &tcoeff, eob, &dequant, rdmult,
                    dc_sign_ctx, txb_skip_ctx, sharpness, sc, &txb_skip, &base_eob, &base, &eob_extra,
                    &dc_sign, &lps, &eob_c,
                );

                // Rust.
                let mut qc_r = qcoeff0.clone();
                let mut dqc_r = dqcoeff0.clone();
                let t = CoeffCostTables {
                    txb_skip: &txb_skip,
                    base_eob: &base_eob,
                    base: &base,
                    eob_extra: &eob_extra,
                    dc_sign: &dc_sign,
                    lps: &lps,
                    eob: &eob_c,
                };
                let r = optimize_txb(
                    tx_size, tx_type, &mut qc_r, &mut dqc_r, &tcoeff, eob, dequant, rdmult,
                    dc_sign_ctx, txb_skip_ctx, sharpness, sc, &t,
                );

                let ctx = format!("ts={tx_size} tt={tx_type} eob={eob} sharp={sharpness}");
                assert_eq!(r.eob, eob_wc, "eob {ctx}");
                assert_eq!(r.rate, rate_wc, "rate {ctx}");
                assert_eq!(qc_r, qc_c, "qcoeff {ctx}");
                assert_eq!(dqc_r, dqc_c, "dqcoeff {ctx}");
            }
        }
    }
}
