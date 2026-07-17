//! Smoke + anti-vacuous witness for `shim_encode_av1_kf_tune` (C4 family):
//! the knob-explicit encode entry must (a) produce a decodable-length stream
//! for stock knobs byte-identical to `ref_encode_av1_kf` (same controls), and
//! (b) produce a DIFFERENT stream when tune=IQ is installed (the bundle must
//! bite) and when --dist-metric=qm-psnr rides on a QM-on encode (the
//! chunk-2 metric arm must bite) — otherwise the port-side byte gates could
//! pass vacuously.

use aom_sys_ref as c;

fn textured(w: usize, h: usize) -> Vec<u16> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (((x * 7 + y * 13) ^ (x.wrapping_mul(y)) ^ (x * x + y)) & 0xff) as u16
        })
        .collect()
}

#[test]
fn tune_shim_stock_matches_base_and_knobs_bite() {
    c::ref_init();
    let (w, h) = (64usize, 64);
    let y = textured(w, h);
    let empty: Vec<u16> = Vec::new();
    for &cq in &[20i32, 40] {
        // Stock knobs (deltaq_mode pinned 0 like the base shim) == base shim.
        let base = c::ref_encode_av1_kf(
            &y, &empty, &empty, w, h, 8, true, 1, 1, cq, 0, false, false, 2, 0, false,
        );
        let stock = c::ref_encode_av1_kf_tune(
            &y,
            &empty,
            &empty,
            w,
            h,
            8,
            true,
            1,
            1,
            cq,
            0,
            2,
            &c::RefTuneKnobs {
                deltaq_mode: 0,
                enable_cdef: 0,
                ..Default::default()
            },
        );
        assert!(!base.is_empty() && !stock.is_empty());
        assert_eq!(base, stock, "stock tune-shim must reproduce the base shim (cq={cq})");

        // tune=IQ (cdef/deltaq arms left ON) must change the stream.
        let iq = c::ref_encode_av1_kf_tune(
            &y,
            &empty,
            &empty,
            w,
            h,
            8,
            true,
            1,
            1,
            cq,
            0,
            2,
            &c::RefTuneKnobs {
                tuning: c::AOM_TUNE_IQ,
                ..Default::default()
            },
        );
        assert_ne!(base, iq, "tune=IQ must change the C bitstream (cq={cq})");

        // QM-on + qm-psnr dist metric vs QM-on + psnr: the metric must bite.
        let qm_psnr = |metric: i32| {
            c::ref_encode_av1_kf_tune(
                &y,
                &empty,
                &empty,
                w,
                h,
                8,
                true,
                1,
                1,
                cq,
                0,
                2,
                &c::RefTuneKnobs {
                    dist_metric: metric,
                    enable_qm: 1,
                    qm_min: 2,
                    qm_max: 10,
                    deltaq_mode: 0,
                    enable_cdef: 0,
                    ..Default::default()
                },
            )
        };
        let m_psnr = qm_psnr(c::AOM_DIST_METRIC_PSNR);
        let m_qm = qm_psnr(c::AOM_DIST_METRIC_QM_PSNR);
        assert_ne!(
            m_psnr, m_qm,
            "--dist-metric=qm-psnr must change a QM-on C bitstream (cq={cq})"
        );
    }
}
