//! Contract smoke tests for the dec_shim oracles: the default-KF-FRAME_CONTEXT
//! dump produces the documented layout, the real codec API round-trips a KEY
//! frame in-process, and the MACROBLOCKD facades answer on their domains.

use aom_sys_ref::*;

#[test]
fn dump_default_kf_fc_layout_and_band_boundaries() {
    // Non-degenerate, CDF-shaped output: every dumped instance's first value
    // slot is a probability in (0, 32768) except padding/count slots.
    let d = ref_dump_default_kf_fc(60);
    assert_eq!(d.len(), DUMP_KF_FC_LEN);
    // kf_y[0][0] first slot is a real ICDF value.
    assert!(d[0] > 0 && d[0] < 32768, "kf_y[0][0][0] = {}", d[0]);
    // The coefficient arena is the trailing 4045: its first slot (txb_skip
    // [0][0]) is a real ICDF value too.
    let cf = &d[DUMP_KF_FC_LEN - 4045..];
    assert!(cf[0] > 0 && cf[0] < 32768, "txb_skip[0][0][0] = {}", cf[0]);

    // av1_default_coef_probs selects by qindex band (<=20, <=60, <=120, >120):
    // coeff regions differ across bands, mode regions are qindex-independent.
    let bands = [0, 20, 21, 60, 61, 120, 121, 255];
    let dumps: Vec<Vec<u16>> = bands.iter().map(|&q| ref_dump_default_kf_fc(q)).collect();
    let mode_len = DUMP_KF_FC_LEN - 4045;
    for w in dumps.windows(2) {
        assert_eq!(
            w[0][..mode_len],
            w[1][..mode_len],
            "mode defaults must not depend on qindex"
        );
    }
    // Same band -> identical coeff region; adjacent bands -> different.
    assert_eq!(dumps[0][mode_len..], dumps[1][mode_len..]); // 0 vs 20
    assert_ne!(dumps[1][mode_len..], dumps[2][mode_len..]); // 20 vs 21
    assert_eq!(dumps[2][mode_len..], dumps[3][mode_len..]); // 21 vs 60
    assert_ne!(dumps[3][mode_len..], dumps[4][mode_len..]); // 60 vs 61
    assert_eq!(dumps[4][mode_len..], dumps[5][mode_len..]); // 61 vs 120
    assert_ne!(dumps[5][mode_len..], dumps[6][mode_len..]); // 120 vs 121
    assert_eq!(dumps[6][mode_len..], dumps[7][mode_len..]); // 121 vs 255
}

#[test]
fn codec_api_encode_decode_roundtrip_smoke() {
    // A tiny 4:2:0 8-bit KEY frame through the REAL encoder + decoder.
    let (w, h) = (64usize, 64usize);
    let mut y = vec![0u16; w * h];
    for (i, p) in y.iter_mut().enumerate() {
        *p = ((i * 7) % 200) as u16 + 20;
    }
    let u = vec![100u16; (w / 2) * (h / 2)];
    let v = vec![160u16; (w / 2) * (h / 2)];
    let bytes = ref_encode_av1_kf(&y, &u, &v, w, h, 8, false, 1, 1, 30, 3, false, false, 0);
    assert!(
        bytes.len() > 20,
        "suspiciously small bitstream: {}",
        bytes.len()
    );
    let dec = ref_decode_av1_kf(&bytes, w, h);
    assert_eq!(dec.info[0], 8);
    assert_eq!(dec.info[1], 0);
    assert_eq!((dec.info[2], dec.info[3]), (1, 1));
    assert_eq!(dec.y.len(), w * h);
    assert_eq!(dec.u.len(), (w / 2) * (h / 2));
    // Lossy but sane: mean abs error under 32 at cq 30.
    let mae: u64 = y
        .iter()
        .zip(&dec.y)
        .map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs())
        .sum::<u64>()
        / (w * h) as u64;
    assert!(mae < 32, "decoded luma wildly off (mae {mae})");
}

#[test]
fn dec_facades_answer_on_domain() {
    // Spot vectors; the exhaustive diffs live in aom-entropy / aom-decode.
    assert!(!ref_is_chroma_reference(0, 0, 0 /*4x4*/, 1, 1));
    assert!(ref_is_chroma_reference(1, 1, 0 /*4x4*/, 1, 1));
    assert!(ref_is_chroma_reference(0, 0, 3 /*8x8*/, 1, 1));
    assert_eq!(ref_scale_chroma_bsize(0 /*4x4*/, 1, 1), 3 /*8x8*/);
    assert_eq!(ref_get_max_uv_txsize(3 /*8x8*/, 1, 1), 0 /*TX_4X4*/);
    // DC everywhere -> DCT_DCT; H_PRED (uv_mode 2) -> V/H flip rule.
    assert_eq!(ref_intra_mode_to_tx_type(0, 0, 1), 0);
    // tx_size_from_tx_mode: TX_MODE_LARGEST(1) on 64x64 (bsize 12) = TX_64X64(4).
    assert_eq!(ref_tx_size_from_tx_mode(12, 1), 4);
    assert_eq!(ref_depth_to_tx_size(0, 12), 4);
    // get_tx_size_context: no neighbours -> 0.
    assert_eq!(
        ref_get_tx_size_context(12, 64, 64, false, false, 0, false, 0, false),
        0
    );
    // set_txfm_ctxs stamps tx dims (not skip): TX_8X8 over a 16x16 block.
    let (mut a, mut l) = ([0u8; 4], [0u8; 4]);
    ref_set_txfm_ctxs(1 /*TX_8X8*/, 4, 4, false, &mut a, &mut l);
    assert_eq!(a, [8; 4]);
    assert_eq!(l, [8; 4]);
    ref_set_txfm_ctxs(1, 4, 4, true, &mut a, &mut l);
    assert_eq!(a, [16; 4]); // skip -> block dims (4 mi * 4 px)
    assert_eq!(l, [16; 4]);
}

#[test]
fn cfl_kernels_reachable() {
    // 8x8 luma 4:2:0 subsample -> 4x4 q3 (each = 2x2 sum << 1).
    let input = [64u16; 8 * 8];
    let mut out = [0u16; 1024];
    ref_cfl_subsample_hbd((1, 1), 1 /*TX_8X8*/, &input, 8, &mut out);
    assert_eq!(out[0], 64 * 4 * 2);
    assert_eq!(out[3], 64 * 4 * 2);
    assert_eq!(out[4], 0); // outside the 4-wide output row
    let mut ac = [0i16; 1024];
    // subtract_average over the 4x4 chroma tx (TX_4X4=0): flat -> all zeros.
    let mut q3 = [0u16; 1024];
    for r in 0..4 {
        for c in 0..4 {
            q3[r * 32 + c] = 512;
        }
    }
    ref_cfl_subtract_average(0, &q3, &mut ac);
    assert!(ac.iter().take(4).all(|&x| x == 0));
    // predict adds alpha*ac to the DC pred; ac==0 keeps dst.
    let mut dst = vec![77u16; 16];
    ref_cfl_predict_hbd(0, &ac, &mut dst, 4, 3, 8);
    assert!(dst.iter().all(|&x| x == 77));
}
