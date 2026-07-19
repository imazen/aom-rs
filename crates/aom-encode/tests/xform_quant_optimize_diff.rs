//! End-to-end differential for the full speed-0 block coefficient pipeline vs C:
//! av1_xform_quant + get_txb_ctx + av1_optimize_b (the trellis) + entropy-ctx
//! write. A random residual + quant tables + cost tables + neighbour contexts,
//! run through `xform_quant_optimize`, must match the same steps chained through
//! the C oracle: ref_fwd_txfm2d -> ref_quantize_* -> ref_get_txb_ctx ->
//! ref_optimize_txb[_qm] -> ref_txb_entropy_context (or the txb-skip cost at
//! eob 0). Locks the trellis + context wiring on top of xform_quant.

use aom_encode::{BlockContext, OptimizeInputs, QuantKind, QuantParams, xform_quant_optimize};
use aom_sys_ref as c;
use aom_dsp::transform::txfm2d::fwd_txfm_valid;
use aom_dsp::txb::{CoeffCostTables, scan, txb_high, txb_wide};

const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];
const TX_SIZE_2D: [i32; 19] = [
    16, 64, 256, 1024, 4096, 32, 32, 128, 128, 512, 512, 2048, 2048, 64, 64, 256, 256, 1024, 1024,
];
const PLANE_BSIZES: [usize; 8] = [0, 3, 6, 9, 12, 4, 7, 10];

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
    fn cost(&mut self) -> i32 {
        self.range(0, 20 << 9)
    }
    fn qm(&mut self) -> u8 {
        self.range(1, 256) as u8
    }
}

fn tbl(rng: &mut Rng, n: usize) -> Vec<i32> {
    (0..n).map(|_| rng.cost()).collect()
}

#[test]
fn xform_quant_optimize_end_to_end_identical() {
    let mut rng = Rng(0x0f01_c0de_a11b_1005);
    const TX_TYPES: [usize; 7] = [0, 1, 2, 3, 9, 10, 11];
    let (mut total, mut nonzero_eob, mut reduced_eob) = (0usize, 0usize, 0usize);
    for tx_size in 0..19usize {
        let full = TX_W[tx_size] * TX_H[tx_size];
        let n_coeffs = txb_wide(tx_size) * txb_high(tx_size);
        let ls = log_scale(tx_size);
        for &tx_type in &TX_TYPES {
            if !fwd_txfm_valid(tx_type, tx_size) {
                continue;
            }
            for iter in 0..40 {
                let residual: Vec<i16> = (0..full).map(|_| rng.range(-255, 256) as i16).collect();
                // Reciprocal quant/dequant (quant ~ 2^16/dequant), as real encoder
                // tables are: this keeps dqcoeff ~ tcoeff so the QM trellis'
                // squared (diff*qm) in get_coeff_dist can't overflow i64 (over-
                // quantized dqcoeff, not realistic, is what blows it up).
                let recip = |rng: &mut Rng| -> ([i16; 2], [i16; 2]) {
                    let dq = [rng.range(16, 800), rng.range(16, 800)];
                    let q = [
                        (65536 / dq[0]).clamp(1, 32767),
                        (65536 / dq[1]).clamp(1, 32767),
                    ];
                    ([dq[0] as i16, dq[1] as i16], [q[0] as i16, q[1] as i16])
                };
                let (dequant, quant) = recip(&mut rng);
                let zbin = [rng.range(1, 100) as i16, rng.range(1, 100) as i16];
                let round = [rng.range(1, 400) as i16, rng.range(1, 400) as i16];
                let quant_shift = [rng.range(8000, 32767) as i16, rng.range(8000, 32767) as i16];
                // Real quant matrices satisfy qm[i]*iqm[i] ~ 2^(2*AOM_QM_BITS)=1024,
                // so the quantizer's qm-inflation cancels the dequant's iqm-inflation
                // (dqcoeff reconstructs ~tcoeff). Independent random qm/iqm would
                // instead compound and blow up the QM trellis' squared get_coeff_dist.
                let qm_v: Vec<u8> = (0..n_coeffs).map(|_| rng.qm()).collect();
                let iqm_v: Vec<u8> = qm_v
                    .iter()
                    .map(|&w| (1024 / w as u32).clamp(1, 255) as u8)
                    .collect();

                // Cost tables (same layout as optimize_diff).
                let txb_skip = tbl(&mut rng, 13 * 2);
                let base_eob = tbl(&mut rng, 4 * 3);
                let base = tbl(&mut rng, 42 * 8);
                let eob_extra = tbl(&mut rng, 9 * 2);
                let dc_sign = tbl(&mut rng, 3 * 2);
                let lps = tbl(&mut rng, 21 * 26);
                let eob_c = tbl(&mut rng, 2 * 11);

                // Neighbour ENTROPY_CONTEXT bytes (cul_level | dc_sign<<3).
                let mk = |rng: &mut Rng| -> Vec<i8> {
                    (0..16)
                        .map(|_| (rng.range(0, 8) | (rng.range(0, 3) << 3)) as i8)
                        .collect()
                };
                let above = mk(&mut rng);
                let left = mk(&mut rng);
                let plane = (rng.range(0, 3)) as usize;
                let plane_bsize = PLANE_BSIZES[rng.range(0, 8) as usize];
                let rdmult = rng.range(1, 1 << 20) as i64;
                let sharpness = rng.range(0, 8);

                let use_qm = iter % 2 == 0;
                let kind = match iter % 5 {
                    0 | 1 => QuantKind::Fp,
                    2 | 3 => QuantKind::B,
                    _ => QuantKind::Dc,
                };
                let (qm, iqm) = if use_qm {
                    (Some(&qm_v[..]), Some(&iqm_v[..]))
                } else {
                    (None, None)
                };

                let cost = CoeffCostTables {
                    txb_skip: &txb_skip,
                    base_eob: &base_eob,
                    base: &base,
                    eob_extra: &eob_extra,
                    dc_sign: &dc_sign,
                    lps: &lps,
                    eob: &eob_c,
                };
                let qp = QuantParams {
                    zbin: &zbin,
                    round: &round,
                    quant: &quant,
                    quant_shift: &quant_shift,
                    dequant: &dequant,
                    qm,
                    iqm,
                    bd: 8,
                    lossless: false,
                    qm_ctx: None,
                    adaptive: false,
                };
                let bctx = BlockContext {
                    above: &above,
                    left: &left,
                    plane,
                    plane_bsize,
                };
                let opt = OptimizeInputs {
                    cost: &cost,
                    rdmult,
                    sharpness,
                };
                let got = xform_quant_optimize(&residual, tx_size, tx_type, kind, &qp, &bctx, &opt);

                // Oracle: fwd -> quant -> get_txb_ctx -> optimize (or skip cost).
                let coeff_c = c::ref_fwd_txfm2d(tx_size, &residual, TX_W[tx_size], tx_type);
                let src = &coeff_c[..n_coeffs];
                let sc = scan(tx_size, tx_type);
                let iscan = vec![0i16; n_coeffs];
                let (mut qc, mut dqc, eob0) = match (kind, use_qm) {
                    (QuantKind::Fp, true) => c::ref_quantize_fp_qm(
                        ls, src, &round, &quant, &dequant, &qm_v, &iqm_v, sc, &iscan,
                    ),
                    (QuantKind::Fp, false) => {
                        c::ref_quantize_fp(ls, src, &round, &quant, &dequant, sc)
                    }
                    (QuantKind::B, true) => c::ref_quantize_b_qm(
                        ls,
                        src,
                        &zbin,
                        &round,
                        &quant,
                        &quant_shift,
                        &dequant,
                        &qm_v,
                        &iqm_v,
                        sc,
                    ),
                    (QuantKind::B, false) => c::ref_quantize_b(
                        ls,
                        src,
                        &zbin,
                        &round,
                        &quant,
                        &quant_shift,
                        &dequant,
                        sc,
                    ),
                    (QuantKind::Dc, _) => {
                        c::ref_quantize_dc(ls, src, &round, quant[0], dequant[0], qm, iqm)
                    }
                };
                let (skip_ctx, dc_sign_ctx) =
                    c::ref_get_txb_ctx(plane_bsize, tx_size, plane, &above, &left);

                let m = format!("ts={tx_size} tt={tx_type} kind={kind:?} qm={use_qm} eob0={eob0}");
                total += 1;
                if eob0 == 0 {
                    let rate = txb_skip[skip_ctx as usize * 2 + 1];
                    assert_eq!(got.rate, rate, "skip rate {m}");
                    assert_eq!(got.eob, 0, "skip eob {m}");
                    assert_eq!(got.txb_entropy_ctx, 0, "skip ctx {m}");
                    assert_eq!(got.qcoeff, qc, "skip qcoeff {m}");
                    continue;
                }
                nonzero_eob += 1;
                let (eob_w, rate_w) = if use_qm {
                    c::ref_optimize_txb_qm(
                        tx_size,
                        tx_type,
                        &mut qc,
                        &mut dqc,
                        src,
                        eob0 as usize,
                        &dequant,
                        rdmult,
                        dc_sign_ctx as usize,
                        skip_ctx as usize,
                        sharpness,
                        sc,
                        &txb_skip,
                        &base_eob,
                        &base,
                        &eob_extra,
                        &dc_sign,
                        &lps,
                        &eob_c,
                        &iqm_v,
                        &qm_v,
                    )
                } else {
                    c::ref_optimize_txb(
                        tx_size,
                        tx_type,
                        &mut qc,
                        &mut dqc,
                        src,
                        eob0 as usize,
                        &dequant,
                        rdmult,
                        dc_sign_ctx as usize,
                        skip_ctx as usize,
                        sharpness,
                        sc,
                        &txb_skip,
                        &base_eob,
                        &base,
                        &eob_extra,
                        &dc_sign,
                        &lps,
                        &eob_c,
                    )
                };
                let ctx_w = c::ref_txb_entropy_context(&qc, tx_size, tx_type, eob_w);

                reduced_eob += (eob_w < eob0 as usize) as usize;
                assert_eq!(got.eob as usize, eob_w, "opt eob {m}");
                assert_eq!(got.rate, rate_w, "opt rate {m}");
                assert_eq!(got.qcoeff, qc, "opt qcoeff {m}");
                assert_eq!(got.dqcoeff, dqc, "opt dqcoeff {m}");
                assert_eq!(got.txb_entropy_ctx, ctx_w, "opt ctx {m}");
            }
        }
    }
    // The trellis path (eob>0) and its eob-reduction branch must both fire.
    assert!(
        nonzero_eob * 4 >= total,
        "too few nonzero-eob blocks: {nonzero_eob}/{total}"
    );
    assert!(
        reduced_eob > 0,
        "trellis never reduced an eob ({nonzero_eob} nonzero blocks)"
    );
}
