//! Differential harness for the `(qindex, deltas)` -> quantizer-setup layer:
//! - `av1_set_error_per_bit` (rd.h) and the sad-per-bit lut entry
//!   (`init_me_luts_bd`, rd.c) vs the C shims (real inline / real
//!   `av1_convert_qindex_to_q`);
//! - `init_plane_quantizers` (the qindex-derivation slice of
//!   `av1_init_plane_quantizers`, av1/encoder/av1_quantize.c) vs the same C
//!   chain (REAL `av1_get_qindex` -> REAL `av1_compute_rd_mult` -> real
//!   per-bit helpers);
//! - `QuantParams::from_plane_rows` (the `MACROBLOCK_PLANE` row-choice bridge)
//!   vs the REAL exported quantize facades fed a full plane whose rows are
//!   installed exactly as `set_q_index` installs them — the facade, not the
//!   harness, picks the rows each quantizer kind reads.

use aom_encode::rd::{
    EncMode, FrameType, FrameUpdateType, TuneMetric, av1_set_error_per_bit, av1_set_sad_per_bit,
    init_plane_quantizers,
};
use aom_encode::{QuantKind, QuantParams, xform_quant};
use aom_quant::{Dequants, Quants, SEG_LVL_ALT_Q, Segmentation, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_transform::txfm2d::fwd_txfm_valid;
use aom_txb::{txb_high, txb_wide};

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];

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
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

#[test]
fn error_per_bit_matches_c() {
    let mut rng = Rng(0xe9b1_7d3f_0451_9c2b);
    for rdmult in [1, 2, 63, 64, 65, 127, 128, 1 << 20, i32::MAX] {
        assert_eq!(
            av1_set_error_per_bit(rdmult),
            c::ref_error_per_bit(rdmult),
            "rdmult={rdmult}"
        );
    }
    for _ in 0..100_000 {
        let rdmult = rng.range(1, i32::MAX);
        assert_eq!(
            av1_set_error_per_bit(rdmult),
            c::ref_error_per_bit(rdmult),
            "rdmult={rdmult}"
        );
    }
}

#[test]
fn sad_per_bit_matches_c() {
    for bd in [8, 10, 12] {
        for qindex in 0..256 {
            assert_eq!(
                av1_set_sad_per_bit(qindex, bd),
                c::ref_sad_per_bit(qindex, bd as i32),
                "qindex={qindex} bd={bd}",
            );
        }
    }
}

/// The composition: Rust `init_plane_quantizers` vs the C chain of REAL
/// functions over the same inputs (the one-line `clamp(base [+ delta], 0, 255)`
/// prefix is shared per av1_quantize.c:806-810; everything downstream —
/// `av1_get_qindex`, `av1_compute_rd_mult`, both per-bit helpers — is C).
#[test]
fn init_plane_quantizers_matches_c_chain() {
    let mut rng = Rng(0x1d1e_5c0d_e0f1_a2b3);
    // Fixed frame-level params: the speed-0 all-intra KEY shape...
    let allintra = (
        FrameUpdateType::Kf,
        FrameType::Key,
        TuneMetric::Psnr,
        EncMode::Allintra,
        false,
        false,
    );
    // ...plus a two-pass inter shape so the layer/boost path stays honest.
    let twopass = (
        FrameUpdateType::Arf,
        FrameType::NonKey,
        TuneMetric::Psnr,
        EncMode::Good,
        false,
        true,
    );
    for case in 0..60_000 {
        let (update_type, frame_type, tuning, mode, use_fixed, is_stat) =
            if case % 4 == 3 { twopass } else { allintra };
        let bd = [8u8, 10, 12][case % 3];
        let base_qindex = rng.range(0, 256);
        let delta_q_present = rng.next() & 1 == 1;
        let delta_qindex = rng.range(-80, 81);
        let y_dc_delta_q = rng.range(-64, 64);
        let segment_id = (rng.next() % 8) as usize;
        let enabled = rng.next() & 1 == 1;
        let mut mask = [0u32; 8];
        let mut data = [[0i16; 8]; 8];
        let mut altq = [0i16; 8];
        for s in 0..8 {
            mask[s] = (rng.next() & 0xff) as u32;
            let d = rng.range(-128, 129) as i16;
            data[s][SEG_LVL_ALT_Q] = d;
            altq[s] = d;
        }
        let seg = Segmentation {
            enabled,
            feature_mask: mask,
            feature_data: data,
        };
        let layer_depth = rng.range(0, 7);
        let boost_index = rng.range(0, 16);

        let got = init_plane_quantizers(
            &seg,
            segment_id,
            base_qindex,
            delta_qindex,
            delta_q_present,
            y_dc_delta_q,
            bd,
            update_type,
            layer_depth,
            boost_index,
            frame_type,
            use_fixed,
            is_stat,
            tuning,
            mode,
        );

        // C chain (shared clamp prefix, then real C for every step).
        let current_qindex = if delta_q_present {
            base_qindex + delta_qindex
        } else {
            base_qindex
        }
        .clamp(0, 255);
        let qindex_c = c::ref_get_qindex(enabled, &mask, &altq, segment_id as i32, current_qindex);
        let rdmult_c = c::ref_compute_rd_mult(
            qindex_c + y_dc_delta_q,
            bd as i32,
            update_type as i32,
            layer_depth,
            boost_index,
            frame_type as i32,
            use_fixed as i32,
            is_stat as i32,
            tuning as i32,
            mode as i32,
        );
        let m = format!(
            "case={case} bd={bd} base={base_qindex} dq={delta_q_present}/{delta_qindex} \
             ydc={y_dc_delta_q} seg={segment_id} en={enabled}",
        );
        assert_eq!(got.qindex, qindex_c, "qindex {m}");
        assert_eq!(got.rdmult, rdmult_c, "rdmult {m}");
        assert_eq!(
            got.errorperbit,
            c::ref_error_per_bit(rdmult_c),
            "errorperbit {m}"
        );
        assert_eq!(
            got.sadperbit,
            c::ref_sad_per_bit(qindex_c, bd as i32),
            "sadperbit {m}"
        );
        let skip_c = enabled && (mask[segment_id] & (1 << aom_quant::SEG_LVL_SKIP)) != 0;
        assert_eq!(got.seg_skip_block, skip_c, "seg_skip_block {m}");
    }
}

/// The row-choice bridge: residual -> `xform_quant` with
/// `QuantParams::from_plane_rows(set_q_index(...))` vs the C chain
/// `ref_fwd_txfm2d` -> REAL quantize facade over a `MACROBLOCK_PLANE` holding
/// ALL seven rows (the facade picks per kind). A wrong row choice in the
/// bridge (e.g. `round` vs `round_fp`, `quant` vs `quant_fp`) diverges here.
#[test]
fn from_plane_rows_matches_real_facades() {
    c::ref_init();
    let mut rng = Rng(0xface_cafe_0000_5eed);
    const TX_TYPES: [usize; 4] = [0, 1, 2, 3];
    let mut nonzero_eob = [0usize; 3];
    let mut total = [0usize; 3];
    for tx_size in 0..19usize {
        let full = TX_W[tx_size] * TX_H[tx_size];
        let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
        let ls = aom_encode::tx_scale(tx_size);
        for &tx_type in &TX_TYPES {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for iter in 0..18 {
                // bd and kind on independent axes (all 9 combos, twice each).
                let bd: u8 = [8, 8, 12][iter % 3];
                let rmax = if bd > 8 { 4096 } else { 256 };
                let kind = match (iter / 3) % 3 {
                    0 => QuantKind::Fp,
                    1 => QuantKind::B,
                    _ => QuantKind::Dc,
                };
                // Real quantizer tables from (bd, deltas, sharpness); a LOW
                // qindex band keeps quant steps small so eob > 0 dominates.
                let (ydc, udc, uac, vdc, vac) = (
                    rng.range(-16, 17),
                    rng.range(-16, 17),
                    rng.range(-16, 17),
                    rng.range(-16, 17),
                    rng.range(-16, 17),
                );
                let sharpness = if iter % 4 == 0 { rng.range(0, 8) } else { 0 };
                let qindex = if iter % 2 == 0 {
                    rng.range(1, 96) as usize
                } else {
                    rng.range(0, 256) as usize
                };
                let plane = (rng.next() % 3) as usize;

                let mut quants = Quants::zeroed();
                let mut deq = Dequants::zeroed();
                av1_build_quantizer(
                    bd,
                    ydc,
                    udc,
                    uac,
                    vdc,
                    vac,
                    &mut quants,
                    &mut deq,
                    sharpness,
                );
                let rows = set_q_index(&quants, &deq, qindex, plane);
                let qp = QuantParams::from_plane_rows(&rows, kind, bd);

                let residual: Vec<i16> = (0..full)
                    .map(|_| rng.range(-(rmax - 1), rmax) as i16)
                    .collect();
                let got = xform_quant(&residual, tx_size, tx_type, kind, &qp, false);

                // C side: real fwd txfm, then the REAL facade over the full plane rows.
                let coeff_c = c::ref_fwd_txfm2d(tx_size, &residual, TX_W[tx_size], tx_type);
                let rows_c = c::ref_set_q_index(
                    bd as i32,
                    ydc,
                    udc,
                    uac,
                    vdc,
                    vac,
                    sharpness,
                    qindex as i32,
                );
                let plane_rows = &rows_c[plane * 7 * 8..(plane + 1) * 7 * 8];
                let kind_c = match kind {
                    QuantKind::Fp => 0,
                    QuantKind::B => 1,
                    QuantKind::Dc => 2,
                };
                let mut qc = vec![0i32; n_coeffs];
                let mut dqc = vec![0i32; n_coeffs];
                let eob_c = c::ref_quant_plane_rows(
                    kind_c,
                    bd > 8,
                    &coeff_c[..n_coeffs],
                    plane_rows,
                    aom_txb::scan(tx_size, tx_type),
                    aom_txb::iscan(tx_size, tx_type),
                    ls,
                    &mut qc,
                    &mut dqc,
                );

                let m = format!(
                    "ts={tx_size} tt={tx_type} kind={kind:?} bd={bd} qindex={qindex} plane={plane}",
                );
                assert_eq!(got.qcoeff, qc, "qcoeff {m}");
                assert_eq!(got.dqcoeff, dqc, "dqcoeff {m}");
                assert_eq!(got.eob, eob_c, "eob {m}");
                total[kind_c as usize] += 1;
                nonzero_eob[kind_c as usize] += (got.eob > 0) as usize;
            }
        }
    }
    // Non-vacuity per kind: each quantizer must see plenty of coded blocks
    // (a wrong-row bridge that zeroes everything would otherwise pass).
    for k in 0..3 {
        assert!(
            nonzero_eob[k] * 4 >= total[k],
            "too few nonzero-eob blocks for kind {k}: {}/{}",
            nonzero_eob[k],
            total[k],
        );
    }
}
