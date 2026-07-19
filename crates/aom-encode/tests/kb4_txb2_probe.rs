//! KB-4 bug#2 DECISIVE PROBE — real-C-leaf per-tx_type table for the exact
//! bd10 cq8 ramp mi(2,4) txb2 near-tie.
//!
//! Feeds txb2's EXACT captured residual + context (bd10, TX_4X4, mode=D45,
//! plane_bsize=BLOCK_16X4, bctx above=[0,23]/left=[0], qindex=32) into BOTH the
//! port `search_tx_type_intra` AND a real-C-leaf per-tx_type chain (the SAME
//! REAL C pieces `search_tx_type_diff.rs` uses: ref_get_tx_mask_intra,
//! ref_pixel_diff_dist, ref_fwd_txfm2d, ref_quant_plane_rows, ref_optimize_txb,
//! ref_get_tx_type_cost, ref_hbd_variance, ref_rdcost). REAL frame costs from
//! `derive_real_costs(KfFrameContext::default_for_qindex(32))` — identical to
//! the encode. Dumps both per-tx_type tables to pin the divergent
//! (tx_type, rate-or-dist, value). Report-only (asserts nothing).

use aom_encode::BlockContext;
use aom_encode::real_costs::derive_real_costs;
use aom_encode::tx_search::{
    TxTypeSearchInputs, TxTypeSearchPolicy, search_tx_type_intra, trellis_rdmult_intra_y,
};
use aom_dsp::entropy::partition::KfFrameContext;
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use aom_dsp::txb::{iscan, scan};

// ---- Captured txb2 inputs (from KB4_CAP instrumentation) ----
const RESIDUAL: [i16; 16] = [1, -2, -5, -4, -2, -5, -4, -5, -5, -4, -5, -3, -4, -5, -3, 2];
const PRED: [u16; 16] = [
    395, 364, 349, 450, 364, 349, 450, 297, 349, 450, 297, 533, 450, 297, 533, 750,
];
const SRC: [u16; 16] = [
    396, 362, 344, 446, 362, 344, 446, 292, 344, 446, 292, 530, 446, 292, 530, 752,
];

/// sadvar_shim variance size index per TX_SIZE (TX_4X4 -> 0).
const VAR_IDX_4X4: usize = 0;

/// Replicate `real_costs::repack_intra_ext_tx_cdf` (private) so the C-side ttc
/// uses the SAME KfFrameContext CDFs the port's `tx_type_costs_y` was filled
/// from.
fn repack_intra_ext_tx_cdf(kf: &KfFrameContext) -> Vec<u16> {
    const EXT_TX_SETS_INTRA: usize = 3;
    const EXT_TX_SIZES: usize = 4;
    const INTRA_MODES: usize = 13;
    const TX_TYPES: usize = 16;
    let stride = TX_TYPES + 1;
    let mut out = vec![0u16; EXT_TX_SETS_INTRA * EXT_TX_SIZES * INTRA_MODES * stride];
    for tx_idx in 0..EXT_TX_SIZES {
        for mode in 0..INTRA_MODES {
            let base1 = ((EXT_TX_SIZES + tx_idx) * INTRA_MODES + mode) * stride;
            out[base1..base1 + 8].copy_from_slice(&kf.ext_tx_1ddct[tx_idx][mode]);
            let base2 = ((2 * EXT_TX_SIZES + tx_idx) * INTRA_MODES + mode) * stride;
            out[base2..base2 + 6].copy_from_slice(&kf.ext_tx_dtt4[tx_idx][mode]);
        }
    }
    out
}

#[test]
fn kb4_txb2_real_c_leaf_probe() {
    c::ref_init();
    let bd = 10u8;
    let tx_size = 0usize; // TX_4X4
    let w = 4usize;
    let h = 4usize;
    let plane_bsize = 17usize; // BLOCK_16X4 (the real block; drives get_txb_ctx)
    let tx_bsize = 0usize; // BLOCK_4X4 (drives block_sse over the txb)
    let mode = 3usize; // D45_PRED
    let reduced = false;
    let use_fi = false;
    let fi_mode = 0usize;
    let lossless = false;
    let qindex = 32usize;
    let rdmult = 2572i32;
    let above: Vec<i8> = vec![0, 23];
    let left: Vec<i8> = vec![0];

    // Quantizer rows (port + C) at qindex=32, bd10, KEY (deltas 0).
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(bd, 0, 0, 0, 0, 0, &mut quants, &mut deq, 0);
    let rows = set_q_index(&quants, &deq, qindex, 0);
    let rows_c = c::ref_set_q_index(bd as i32, 0, 0, 0, 0, 0, 0, qindex as i32);
    let plane_rows_c = &rows_c[0..7 * 8];

    // REAL frame costs — identical to the encode (same kf@qindex32).
    let kf = KfFrameContext::default_for_qindex(qindex as i32);
    let real = derive_real_costs(&kf, true);
    let ct = real.coeff_costs_y.tables(tx_size);
    let intra_cdf = repack_intra_ext_tx_cdf(&kf);
    let inter_cdf = vec![0u16; 4 * 4 * 17];
    let (c_ttc_intra, c_ttc_inter) = c::ref_fill_tx_type_costs(&intra_cdf, &inter_cdf);

    // ---- PORT side ----
    let bctx = BlockContext {
        plane_bsize,
        plane: 0,
        above: &above,
        left: &left,
    };
    let inp = TxTypeSearchInputs {
        residual: &RESIDUAL,
        src: &SRC,
        src_off: 0,
        src_stride: w,
        pred: &PRED,
        tx_size,
        plane: 0,
        uv_mode: 0,
        mode,
        use_filter_intra: use_fi,
        filter_intra_mode: fi_mode,
        lossless,
        reduced_tx_set_used: reduced,
        bd,
        rows: &rows,
        bctx: &bctx,
        rdmult,
        coeff_costs: &ct,
        tx_type_costs: &real.tx_type_costs_y,
        // Interior txb (the captured txb2 is fully inside the frame): the
        // visible area is the full TX_4X4 (see TxTypeSearchInputs docs).
        visible_cols: 4,
        visible_rows: 4,
        qm_level: None,
        predict_skip_zero_blk_rate: 0,
    };
    let pol = TxTypeSearchPolicy::speed0_allintra();
    let got = search_tx_type_intra(&inp, &pol, i64::MAX);
    eprintln!(
        "=== PORT winner @ref_best_rd=MAX (full eval): {:?}",
        got.as_ref()
            .map(|g| (g.best_tx_type, g.best_eob, g.rate, g.dist, g.rd))
    );
    // Sweep ref_best_rd: the adaptive_txb_search break (level=1) fires after
    // tt0 when best_rd/2 = 16775 > ref_best_rd. Below the threshold the port
    // truncates after DCT_DCT and picks tt0 (matching aomenc); above it it
    // over-searches to tt2.
    for &rbr in &[i64::MAX, 20000, 16775, 16774, 16000, 8000] {
        let g = search_tx_type_intra(&inp, &pol, rbr);
        eprintln!(
            "    PORT @ref_best_rd={rbr}: winner={:?}",
            g.as_ref().map(|g| (g.best_tx_type, g.best_eob, g.rd))
        );
    }

    // ---- C real-leaf per-tx_type chain (loop transcribed, LEAVES real C) ----
    let (mask_c, _txk) = c::ref_get_tx_mask_intra(
        tx_size as i32,
        mode as i32,
        use_fi,
        fi_mode as i32,
        lossless,
        reduced,
        1,     // use_reduced_intra_txset (speed-0 allintra)
        false, // use_derived_intra_tx_type_set
        true,  // enable_flip_idtx
        false, // use_intra_dct_only
        false, // use_default_intra_tx_type (speed-0: OFF — winner-mode/speed>=1)
        false, // use_screen_content_tools (mono synthetic HF cell: not screen)
        0,     // prune_tx_type_using_stats (speed-0 sub-480p: off)
    );
    let (bsse_raw, mut mse_c) = c::ref_pixel_diff_dist(
        &RESIDUAL,
        tx_bsize as i32,
        tx_bsize as i32,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    let mut bsse_c = bsse_raw;
    let s = 2 * (bd as i32 - 8);
    bsse_c = (bsse_c + ((1i64 << s) >> 1)) >> s;
    mse_c = (((mse_c as u64) + ((1u64 << s) >> 1)) >> s) as u32;
    bsse_c *= 16;
    let dequant_shift = bd as i32 - 5;
    let qstep_c = (i32::from(plane_rows_c[6 * 8 + 1]) >> dequant_shift) as u64;
    let skip_trellis_c = !((mse_c as u64) <= 3200u64 * qstep_c * qstep_c);
    let kind_c = if skip_trellis_c { 1 } else { 0 }; // B : FP
    let trellis_rdmult = trellis_rdmult_intra_y(rdmult, 0, bd, false);
    let (txb_skip_ctx_c, dc_sign_ctx_c) =
        c::ref_get_txb_ctx(plane_bsize, tx_size, 0, &above, &left);
    eprintln!(
        "=== C setup: mask={mask_c:#06x} block_sse={bsse_c} mse={mse_c} qstep={qstep_c} skip_trellis={skip_trellis_c} txb_skip_ctx={txb_skip_ctx_c} dc_sign_ctx={dc_sign_ctx_c}"
    );

    for tx_type in 0..16usize {
        if mask_c & (1 << tx_type) == 0 {
            continue;
        }
        let coeff = c::ref_fwd_txfm2d(tx_size, &RESIDUAL, w, tx_type);
        let tcoeff = coeff[..w * h].to_vec();
        let mut qc = vec![0i32; w * h];
        let mut dqc = vec![0i32; w * h];
        let eob = c::ref_quant_plane_rows(
            kind_c,
            bd > 8,
            &tcoeff,
            plane_rows_c,
            scan(tx_size, tx_type),
            iscan(tx_size, tx_type),
            aom_encode::tx_scale(tx_size),
            &mut qc,
            &mut dqc,
        ) as usize;
        let ttc = |eob: usize| -> i32 {
            if eob > 0 {
                c::ref_get_tx_type_cost(
                    &c_ttc_intra,
                    &c_ttc_inter,
                    0,
                    tx_size as i32,
                    tx_type as i32,
                    false,
                    reduced,
                    lossless,
                    use_fi,
                    fi_mode as i32,
                    mode as i32,
                )
            } else {
                0
            }
        };
        let (eob_f, rate_c) = if !skip_trellis_c {
            if eob == 0 {
                (0usize, ct.txb_skip[txb_skip_ctx_c as usize * 2 + 1])
            } else {
                let (new_eob, r) = c::ref_optimize_txb(
                    tx_size,
                    tx_type,
                    &mut qc,
                    &mut dqc,
                    &tcoeff,
                    eob,
                    &[rows.dequant[0], rows.dequant[1]],
                    trellis_rdmult,
                    dc_sign_ctx_c as usize,
                    txb_skip_ctx_c as usize,
                    0,
                    scan(tx_size, tx_type),
                    ct.txb_skip,
                    ct.base_eob,
                    ct.base,
                    ct.eob_extra,
                    ct.dc_sign,
                    ct.lps,
                    ct.eob,
                );
                (new_eob, r + ttc(new_eob))
            }
        } else {
            let r = c::ref_cost_coeffs_txb(
                &qc,
                eob,
                tx_size,
                tx_type,
                txb_skip_ctx_c as usize,
                dc_sign_ctx_c as usize,
                ct.txb_skip,
                ct.base_eob,
                ct.base,
                ct.eob_extra,
                ct.dc_sign,
                ct.lps,
                ct.eob,
            ) + ttc(eob);
            (eob, r)
        };

        // Distortion: eob==0 -> block_sse; else pixel-domain (low-energy 4x4).
        let dist_c: i64 = if eob_f == 0 {
            bsse_c
        } else {
            let mut recon = PRED.to_vec();
            c::ref_inv_txfm2d_add(tx_size, &dqc, &mut recon, w, tx_type, bd as i32);
            let (_v, vf_sse) = c::ref_hbd_variance(VAR_IDX_4X4, bd, &SRC, w, &recon, w);
            16 * i64::from(vf_sse)
        };
        let rd = c::ref_rdcost(rdmult, rate_c, dist_c);
        eprintln!("C tt{tx_type}: eob={eob_f} rate={rate_c} dist={dist_c} rd={rd}");
    }
}
