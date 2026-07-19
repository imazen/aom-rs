//! Differential harness for the DC/AC quantizer lookups
//! (`av1_dc_quant_QTX` / `av1_ac_quant_QTX`, av1/common/quant_common.c) vs C.
//! Exhaustive over the full qindex range plus clamp edges and random deltas.

use aom_dsp::quant::{av1_ac_quant_qtx, av1_dc_quant_qtx};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % ((hi - lo) as u64)) as i32
    }
}

#[test]
fn dc_ac_quant_qtx_matches_c() {
    let mut rng = Rng(0x00dc_ac00_9e37_1111);
    for &bd in &[8u8, 10, 12] {
        // Exhaustive over the valid qindex range with delta 0 (the rd-mult use).
        for qindex in 0..=255i32 {
            assert_eq!(
                av1_dc_quant_qtx(qindex, 0, bd) as i32,
                c::ref_dc_quant_qtx(qindex, 0, bd as i32),
                "dc qindex={qindex} bd={bd}"
            );
            assert_eq!(
                av1_ac_quant_qtx(qindex, 0, bd) as i32,
                c::ref_ac_quant_qtx(qindex, 0, bd as i32),
                "ac qindex={qindex} bd={bd}"
            );
        }
        // Clamp behaviour: qindex + delta is clamped to [0, 255]. Sweep over and
        // under the range with fixed deltas, plus random (qindex, delta) pairs.
        for qindex in -40..=295i32 {
            for &delta in &[-400, -256, -100, -32, -1, 0, 1, 32, 100, 256, 400] {
                assert_eq!(
                    av1_dc_quant_qtx(qindex, delta, bd) as i32,
                    c::ref_dc_quant_qtx(qindex, delta, bd as i32),
                    "dc clamp qindex={qindex} delta={delta} bd={bd}"
                );
                assert_eq!(
                    av1_ac_quant_qtx(qindex, delta, bd) as i32,
                    c::ref_ac_quant_qtx(qindex, delta, bd as i32),
                    "ac clamp qindex={qindex} delta={delta} bd={bd}"
                );
            }
        }
        for _ in 0..50_000 {
            let qindex = rng.range(-300, 600);
            let delta = rng.range(-300, 300);
            assert_eq!(
                av1_dc_quant_qtx(qindex, delta, bd) as i32,
                c::ref_dc_quant_qtx(qindex, delta, bd as i32),
                "dc rand qindex={qindex} delta={delta} bd={bd}"
            );
            assert_eq!(
                av1_ac_quant_qtx(qindex, delta, bd) as i32,
                c::ref_ac_quant_qtx(qindex, delta, bd as i32),
                "ac rand qindex={qindex} delta={delta} bd={bd}"
            );
        }
    }
}
