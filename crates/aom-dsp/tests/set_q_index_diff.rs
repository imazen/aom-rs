//! Differential harness for `set_q_index` (av1/encoder/av1_quantize.c, the
//! per-qindex row selection into `MACROBLOCK_PLANE`) and `av1_get_qindex`
//! (av1/common/quant_common.c, the segment-adjusted effective qindex).
//!
//! Oracles: `shim_set_q_index` transcribes the (static) row assignments over
//! tables filled by the REAL exported `av1_build_quantizer`;
//! `shim_get_qindex` calls the REAL exported `av1_get_qindex` over a
//! marshalled `struct segmentation`.

use aom_dsp::quant::{
    av1_build_quantizer, av1_get_qindex, set_q_index, Dequants, Quants, Segmentation,
    QINDEX_RANGE, SEG_LVL_ALT_Q,
};
use aom_sys_ref::{ref_get_qindex, ref_set_q_index};

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

/// For one quantizer configuration, check every qindex x plane: the seven
/// 8-lane rows the Rust selection returns equal the rows C's `set_q_index`
/// installs into `x->plane[plane]`.
fn check_rows(bd: u8, ydc: i32, udc: i32, uac: i32, vdc: i32, vac: i32, sharpness: i32) {
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(bd, ydc, udc, uac, vdc, vac, &mut quants, &mut deq, sharpness);

    for qindex in 0..QINDEX_RANGE {
        let cref = ref_set_q_index(bd as i32, ydc, udc, uac, vdc, vac, sharpness, qindex as i32);
        for plane in 0..3usize {
            let rows = set_q_index(&quants, &deq, qindex, plane);
            // Row order documented in rd_shim.c (the C assignment order).
            let pairs: [(&str, &[i16; 8]); 7] = [
                ("quant", rows.quant),
                ("quant_fp", rows.quant_fp),
                ("round_fp", rows.round_fp),
                ("quant_shift", rows.quant_shift),
                ("zbin", rows.zbin),
                ("round", rows.round),
                ("dequant", rows.dequant),
            ];
            for (r, (name, rust_row)) in pairs.iter().enumerate() {
                let c_row = &cref[(plane * 7 + r) * 8..(plane * 7 + r) * 8 + 8];
                assert_eq!(
                    &rust_row[..],
                    c_row,
                    "row mismatch: bd={bd} deltas=({ydc},{udc},{uac},{vdc},{vac}) \
                     sharp={sharpness} qindex={qindex} plane={plane} table={name}",
                );
            }
        }
    }
}

#[test]
fn set_q_index_rows_match_c() {
    // Zero deltas (the all-intra default) for every bd, all 256 qindex.
    for bd in [8u8, 10, 12] {
        check_rows(bd, 0, 0, 0, 0, 0, 0);
    }
    // Distinct per-axis deltas so a swapped table/plane cannot cancel out:
    // each of y-dc/u-dc/u-ac/v-dc/v-ac gets a different value.
    check_rows(8, -12, 7, -3, 21, 9, 0);
    check_rows(10, 15, -8, 30, -25, 4, 3);
    check_rows(12, -63, 63, -32, 17, -5, 7);
    // Random tuples x random sharpness.
    let mut rng = Rng(0x5e70_11de_0a0b_0c0d);
    for _ in 0..6 {
        let bd = [8u8, 10, 12][(rng.next() % 3) as usize];
        check_rows(
            bd,
            rng.range(-64, 64),
            rng.range(-64, 64),
            rng.range(-64, 64),
            rng.range(-64, 64),
            rng.range(-64, 64),
            rng.range(0, 8),
        );
    }
}

#[test]
fn get_qindex_matches_c() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut checked = 0u32;
    for _case in 0..4000 {
        let enabled = rng.next() & 1 == 1;
        let mut mask = [0u32; 8];
        let mut data = [[0i16; 8]; 8];
        let mut altq = [0i16; 8];
        for s in 0..8 {
            // Bias toward the ALT_Q bit but exercise other bits too (they must
            // not affect the result).
            mask[s] = (rng.next() & 0xff) as u32;
            let d = rng.range(-255, 256) as i16;
            data[s][SEG_LVL_ALT_Q] = d;
            altq[s] = d;
        }
        let seg = Segmentation { enabled, feature_mask: mask, feature_data: data };
        for segment_id in 0..8usize {
            let base = rng.range(0, 256);
            let rust = av1_get_qindex(&seg, segment_id, base);
            let c = ref_get_qindex(enabled, &mask, &altq, segment_id as i32, base);
            assert_eq!(
                rust, c,
                "get_qindex: enabled={enabled} mask={:#x} altq={} seg={segment_id} base={base}",
                mask[segment_id], altq[segment_id],
            );
            checked += 1;
        }
    }
    // Clamp edges: data pushing past both ends.
    for (base, d) in [(0, -255), (255, 255), (128, -200), (128, 200), (0, 0), (255, -1)] {
        let mut mask = [0u32; 8];
        mask[3] = 1 << SEG_LVL_ALT_Q;
        let mut data = [[0i16; 8]; 8];
        data[3][SEG_LVL_ALT_Q] = d;
        let mut altq = [0i16; 8];
        altq[3] = d;
        let seg = Segmentation { enabled: true, feature_mask: mask, feature_data: data };
        assert_eq!(
            av1_get_qindex(&seg, 3, base),
            ref_get_qindex(true, &mask, &altq, 3, base),
            "clamp edge base={base} d={d}",
        );
        checked += 1;
    }
    assert!(checked >= 32_000, "coverage: {checked}");
}
