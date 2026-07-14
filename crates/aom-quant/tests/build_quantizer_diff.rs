//! Differential harness for `av1_build_quantizer` (av1/encoder/av1_quantize.c)
//! vs the REAL exported C function (via `shim_build_quantizer`, a pure
//! marshalling wrapper — no transcription). Every call covers the full
//! `QINDEX_RANGE` (256 qindex values) x 21 tables x 8 lanes; the sweeps cover
//! bit depth {8,10,12}, sharpness 0..=7, and the delta-q axes over the full
//! signaled range [-64, 63] (equal / per-axis / random tuples).

use aom_quant::{av1_build_quantizer, Dequants, Quants, QINDEX_RANGE};
use aom_sys_ref::{ref_build_quantizer, BUILD_QUANTIZER_OUT_LEN};

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

/// Run both sides for one parameter set and compare every table entry.
fn check(bd: u8, ydc: i32, udc: i32, uac: i32, vdc: i32, vac: i32, sharpness: i32) {
    let mut cbuf = vec![0i16; BUILD_QUANTIZER_OUT_LEN];
    ref_build_quantizer(bd as i32, ydc, udc, uac, vdc, vac, sharpness, &mut cbuf);

    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(bd, ydc, udc, uac, vdc, vac, &mut quants, &mut deq, sharpness);

    // Table order must match the flat layout documented in rd_shim.c.
    let tables: [(&str, &[[i16; 8]; QINDEX_RANGE]); 21] = [
        ("y_quant", &quants.y_quant),
        ("y_quant_shift", &quants.y_quant_shift),
        ("y_zbin", &quants.y_zbin),
        ("y_round", &quants.y_round),
        ("y_quant_fp", &quants.y_quant_fp),
        ("u_quant_fp", &quants.u_quant_fp),
        ("v_quant_fp", &quants.v_quant_fp),
        ("y_round_fp", &quants.y_round_fp),
        ("u_round_fp", &quants.u_round_fp),
        ("v_round_fp", &quants.v_round_fp),
        ("u_quant", &quants.u_quant),
        ("v_quant", &quants.v_quant),
        ("u_quant_shift", &quants.u_quant_shift),
        ("v_quant_shift", &quants.v_quant_shift),
        ("u_zbin", &quants.u_zbin),
        ("v_zbin", &quants.v_zbin),
        ("u_round", &quants.u_round),
        ("v_round", &quants.v_round),
        ("y_dequant_qtx", &deq.y_dequant_qtx),
        ("u_dequant_qtx", &deq.u_dequant_qtx),
        ("v_dequant_qtx", &deq.v_dequant_qtx),
    ];
    let per_table = QINDEX_RANGE * 8;
    for (t, (name, table)) in tables.iter().enumerate() {
        let c_table = &cbuf[t * per_table..(t + 1) * per_table];
        for (q, (rust_row, c_row)) in table.iter().zip(c_table.chunks_exact(8)).enumerate() {
            assert_eq!(
                rust_row.as_slice(),
                c_row,
                "{name}[{q}] bd={bd} deltas=({ydc},{udc},{uac},{vdc},{vac}) sharpness={sharpness}"
            );
        }
    }
}

/// Baseline: zero deltas, every bit depth, every sharpness (0..=7 is the
/// aomenc --sharpness range; non-zero flips the rounding-factor branch).
#[test]
fn zero_deltas_all_bd_all_sharpness() {
    for &bd in &[8u8, 10, 12] {
        for sharpness in 0..=7 {
            check(bd, 0, 0, 0, 0, 0, sharpness);
        }
    }
}

/// All five deltas equal, swept over the full signaled delta-q range.
#[test]
fn equal_deltas_full_range() {
    for &bd in &[8u8, 10, 12] {
        for &sharpness in &[0, 3, 7] {
            for d in -64..=63 {
                check(bd, d, d, d, d, d, sharpness);
            }
        }
    }
}

/// One delta axis at a time over the full range (others zero) — isolates which
/// table family each delta feeds (y dc / u dc / u ac / v dc / v ac).
#[test]
fn per_axis_delta_sweep() {
    for &bd in &[8u8, 10, 12] {
        for axis in 0..5 {
            for d in (-64..=63).step_by(3) {
                let mut deltas = [0i32; 5];
                deltas[axis] = d;
                let [ydc, udc, uac, vdc, vac] = deltas;
                check(bd, ydc, udc, uac, vdc, vac, 0);
            }
        }
    }
}

/// Random delta 5-tuples with random sharpness.
#[test]
fn random_delta_tuples() {
    let mut rng = Rng(0xB1DC_0DE5_9E37_79B9);
    for &bd in &[8u8, 10, 12] {
        for _ in 0..150 {
            let ydc = rng.range(-64, 64);
            let udc = rng.range(-64, 64);
            let uac = rng.range(-64, 64);
            let vdc = rng.range(-64, 64);
            let vac = rng.range(-64, 64);
            let sharpness = rng.range(0, 8);
            check(bd, ydc, udc, uac, vdc, vac, sharpness);
        }
    }
}
