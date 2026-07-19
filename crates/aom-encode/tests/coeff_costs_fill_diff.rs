//! Differential: the port's CDF->cost table derivation
//! (`derive_real_costs` -> `fill_coeff_cost_set_from_arena`, the encoder-gate
//! reproduction of `av1_fill_coeff_costs`) vs the REAL C `av1_fill_coeff_costs`
//! over the KF-default coefficient CDFs at a given qindex. Every
//! `LV_MAP_COEFF_COST` entry (`txb_skip / base_eob / base / eob_extra /
//! dc_sign / lps`) for every `(txs_ctx, plane)` and every `LV_MAP_EOB_COST`
//! (`eob_cost`) for every `(eob_multi_size, plane)` must be integer-identical.
//! This pins the one unvalidated link in the trellis-input chain (the coeff
//! cost tables the trellis + `av1_cost_coeffs_txb` consume) that
//! `cost_coeffs_diff` / `trellis_cost_diff` deliberately fed random tables to
//! isolate.

use aom_encode::real_costs::derive_real_costs;
use aom_dsp::entropy::partition::KfFrameContext;
use aom_sys_ref as c;

fn assert_slice(label: &str, ctx: &str, got: &[i32], want: &[i32]) {
    assert_eq!(
        got,
        want,
        "{label} mismatch {ctx}\n  first diff: {:?}",
        got.iter()
            .zip(want.iter())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
    );
}

/// Compare every derived cost table (coeff + eob) vs real C for the qindexes
/// that select each of the four default-CDF q-contexts (get_q_ctx bins:
/// <=20, <=60, <=120, else) -- including 128 (the multi-SB gate's cq32/cq48
/// qindex).
#[test]
fn fill_coeff_costs_matches_real_c() {
    c::ref_init();
    for &qindex in &[12i32, 40, 100, 128, 200, 255] {
        let kf = KfFrameContext::default_for_qindex(qindex);
        let real = derive_real_costs(&kf, true);
        for (plane, set) in [
            (0usize, &real.coeff_costs_y),
            (1usize, &real.coeff_costs_uv),
        ] {
            for txs_ctx in 0..5usize {
                let port = &set.by_txs_ctx[txs_ctx];
                let (txb_skip, base_eob, base, eob_extra, dc_sign, lps, _eob) =
                    c::ref_fill_coeff_costs(qindex, txs_ctx, plane, 0);
                let cx = format!("q={qindex} txs_ctx={txs_ctx} plane={plane}");
                assert_slice("txb_skip", &cx, &port.txb_skip, &txb_skip);
                assert_slice("base_eob", &cx, &port.base_eob, &base_eob);
                assert_slice("base", &cx, &port.base, &base);
                assert_slice("eob_extra", &cx, &port.eob_extra, &eob_extra);
                assert_slice("dc_sign", &cx, &port.dc_sign, &dc_sign);
                assert_slice("lps", &cx, &port.lps, &lps);
            }
            for eob_multi_size in 0..7usize {
                let (_ts, _be, _b, _ee, _ds, _lps, eob) =
                    c::ref_fill_coeff_costs(qindex, 0, plane, eob_multi_size);
                let got = &set.eob_by_multi_size[eob_multi_size];
                assert_eq!(
                    got.as_slice(),
                    eob.as_slice(),
                    "eob_cost q={qindex} eob_multi_size={eob_multi_size} plane={plane}\n  first diff: {:?}",
                    got.iter()
                        .zip(eob.iter())
                        .enumerate()
                        .find(|(_, (a, b))| a != b)
                        .map(|(i, (a, b))| (i, *a, *b))
                );
            }
        }
    }
}
