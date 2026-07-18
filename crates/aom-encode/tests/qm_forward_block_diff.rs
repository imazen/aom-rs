//! Forward-QM block-level differential — parts A+B of the QM-on forward-quant
//! port, composed end-to-end and validated against real libaom:
//!
//! - `inverse_qmatrix_reuse_matches_c`: the decode-side `aom_decode::qm::iqmatrix`
//!   selector the encoder now REUSES for the inverse weights must byte-match C's
//!   `av1_qm_init` `giqmatrix[q][c][t]` across all 16 levels x 3 planes x 19 tx
//!   sizes (symmetric to `aom-quant`'s forward `qm_fwd_select_diff`).
//! - `forward_qm_block_realistic_matches_c`: the realistic path — qindex ->
//!   `aom_get_qmlevel_allintra` -> select forward (`aom_quant::qmatrix`) + inverse
//!   (`aom_decode::qm::iqmatrix`) -> `xform_quant` -> byte-match C's forward
//!   transform + `av1_quantize_fp` (+_qm). Ties the qmlevel derivation, both
//!   selectors, and the quantizer kernel together for a real block, and asserts
//!   the forward/inverse selectors agree on the QM-vs-flat (tx_type) gating.

use aom_encode::{QuantKind, QuantParams, xform_quant};
use aom_quant::{aom_get_qmlevel_allintra, qmatrix};
use aom_sys_ref as c;
use aom_transform::txfm2d::fwd_txfm_valid;
use aom_txb::{scan, txb_high, txb_wide};

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
const TX_SIZE_2D: [i32; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];

const NUM_QM_LEVELS: usize = 16;
const TX_SIZES_ALL: usize = 19;
const DCT_DCT: usize = 0;
const V_DCT: usize = 11; // a 1-D transform (>= IDTX) -> flat QM

fn log_scale(tx_size: usize) -> i32 {
    let p = TX_SIZE_2D[tx_size];
    (p > 256) as i32 + (p > 1024) as i32
}

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

/// The reused inverse selector must byte-match real C `av1_qm_init` packing.
#[test]
fn inverse_qmatrix_reuse_matches_c() {
    let mut some_cells = 0usize;
    let mut none_cells = 0usize;
    for q in 0..NUM_QM_LEVELS {
        for plane in 0..3usize {
            for t in 0..TX_SIZES_ALL {
                let mine = aom_decode::qm::iqmatrix(q, plane, t, DCT_DCT);
                let theirs = c::ref_iqm_giqmatrix(q, plane, t);
                match (mine, &theirs) {
                    (Some(m), Some(cm)) => {
                        assert_eq!(
                            m,
                            cm.as_slice(),
                            "inverse QM mismatch at (level={q}, plane={plane}, tx_size={t})"
                        );
                        some_cells += 1;
                    }
                    (None, None) => none_cells += 1,
                    (a, b) => panic!(
                        "None/Some disagreement at (level={q}, plane={plane}, tx_size={t}): \
                         rust={:?} c={:?}",
                        a.map(<[u8]>::len),
                        b.as_ref().map(Vec::len)
                    ),
                }
            }
        }
    }
    assert_eq!(
        some_cells,
        15 * 3 * TX_SIZES_ALL,
        "expected 855 real-matrix cells"
    );
    assert_eq!(
        none_cells,
        3 * TX_SIZES_ALL,
        "expected 57 flat (None) cells"
    );
}

/// Realistic forward-QM block path: qindex -> allintra qmlevel -> select fwd+inv
/// -> xform_quant -> byte-match C. Exercises DCT_DCT (QM applied) and a 1-D
/// transform (QM flat), asserting the fwd/inv selectors agree on the gating.
#[test]
fn forward_qm_block_realistic_matches_c() {
    let mut rng = Rng(0x5eed_00b1_0c4d_2345);
    // qm_min/qm_max = 4/10 are the allintra override defaults.
    let (qm_min, qm_max) = (4i32, 10i32);
    // A spread of base qindex across the allintra step boundaries (<=40, <=100,
    // <=160, <=200, <=220, <=240, else) so multiple qmlevels are exercised.
    let qindices = [10i32, 60, 130, 180, 210, 230, 250];
    // Representative tx sizes incl. a 64-point size (aliasing) and rect.
    let tx_sizes = [0usize, 1, 2, 3, 4, 9, 17];
    let tx_types = [DCT_DCT, V_DCT];

    let mut qm_applied = 0usize; // anti-vacuous: real QM matrices exercised
    let mut flat_gated = 0usize; // 1-D transforms took the flat path
    let mut nonzero_eob = 0usize;

    for &qindex in &qindices {
        let qm_level = aom_get_qmlevel_allintra(qindex, qm_min, qm_max) as usize;
        for plane in 0..2usize {
            for &tx_size in &tx_sizes {
                let full = TX_W[tx_size] * TX_H[tx_size];
                let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
                let ls = log_scale(tx_size);
                for &tx_type in &tx_types {
                    if !fwd_txfm_valid(tx_type, tx_size) {
                        continue;
                    }
                    let qm = qmatrix(qm_level, plane, tx_size, tx_type);
                    let iqm = aom_decode::qm::iqmatrix(qm_level, plane, tx_size, tx_type);
                    // Forward and inverse selectors MUST agree on QM-vs-flat.
                    assert_eq!(
                        qm.is_some(),
                        iqm.is_some(),
                        "fwd/inv gating disagreement: qm_level={qm_level} plane={plane} \
                         tx_size={tx_size} tx_type={tx_type}"
                    );

                    // Residual + a small-dequant/large-quant setup so plenty of
                    // coefficients survive (exercises eob / dqcoeff).
                    let residual: Vec<i16> =
                        (0..full).map(|_| rng.range(-255, 256) as i16).collect();
                    let round = [rng.range(1, 400) as i16, rng.range(1, 400) as i16];
                    let quant = [
                        rng.range(12000, 32767) as i16,
                        rng.range(12000, 32767) as i16,
                    ];
                    let dequant = [rng.range(4, 200) as i16, rng.range(4, 200) as i16];

                    let qp = QuantParams {
                        zbin: &round,
                        round: &round,
                        quant: &quant,
                        quant_shift: &quant,
                        dequant: &dequant,
                        qm,
                        iqm,
                        bd: 8,
                        lossless: false,
                        qm_ctx: None,
                        adaptive: false,
                    };
                    let got = xform_quant(&residual, tx_size, tx_type, QuantKind::Fp, &qp, false);

                    // C oracle: same forward transform, then av1_quantize_fp with
                    // C's OWN selected matrices (ref_qm_gqmatrix/ref_iqm_giqmatrix)
                    // when QM applies, else the flat quantizer.
                    let coeff_c = c::ref_fwd_txfm2d(tx_size, &residual, TX_W[tx_size], tx_type);
                    let src = &coeff_c[..n_coeffs];
                    let (qc, dqc, eob) = if qm.is_some() {
                        let qm_c = c::ref_qm_gqmatrix(qm_level, plane, tx_size).unwrap();
                        let iqm_c = c::ref_iqm_giqmatrix(qm_level, plane, tx_size).unwrap();
                        qm_applied += 1;
                        c::ref_quantize_fp_qm(
                            ls,
                            src,
                            &round,
                            &quant,
                            &dequant,
                            &qm_c,
                            &iqm_c,
                            scan(tx_size, tx_type),
                            &vec![0i16; n_coeffs],
                        )
                    } else {
                        flat_gated += 1;
                        c::ref_quantize_fp(
                            ls,
                            src,
                            &round,
                            &quant,
                            &dequant,
                            scan(tx_size, tx_type),
                        )
                    };

                    let m = format!(
                        "qidx={qindex} lvl={qm_level} plane={plane} ts={tx_size} tt={tx_type}"
                    );
                    assert_eq!(&got.coeff[..n_coeffs], src, "coeff {m}");
                    assert_eq!(got.qcoeff, qc, "qcoeff {m}");
                    assert_eq!(got.dqcoeff, dqc, "dqcoeff {m}");
                    assert_eq!(got.eob, eob, "eob {m}");
                    nonzero_eob += (got.eob > 0) as usize;
                }
            }
        }
    }
    // Anti-vacuous: real QM matrices AND the 1-D flat path were both exercised,
    // and blocks actually produced coefficients.
    assert!(qm_applied > 0, "no QM-applied cells exercised");
    assert!(flat_gated > 0, "no flat-gated (1-D) cells exercised");
    assert!(
        nonzero_eob > 0,
        "no nonzero-eob blocks — assertions would be vacuous"
    );
}
