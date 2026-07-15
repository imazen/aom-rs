//! Anti-vacuous baseline for the (staged) #23 QM-on e2e gate: prove that the
//! real libaom encoder produces a DIFFERENT bitstream with QM on vs off for the
//! same content — otherwise a future "port QM-on bytes == C QM-on bytes" gate
//! could pass vacuously (QM doing nothing). Also exercises the new
//! `ref_encode_av1_kf_qm` FFI wrapper (AV1E_SET_ENABLE_QM/QM_MIN/QM_MAX) end to
//! end. This is the C-reference half of the e2e gate; the port half is blocked
//! on the RD-search QM threading staged in docs/qm_rd_threading_staged.md.

use aom_sys_ref as c;

/// Textured monochrome content — high AC energy so quantization (and thus the
/// QM reweighting) materially affects the coded coefficients.
fn textured(w: usize, h: usize) -> Vec<u16> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (((x * 7 + y * 13) ^ (x.wrapping_mul(y)) ^ (x * x + y)) & 0xff) as u16
        })
        .collect()
}

#[test]
fn qm_on_differs_from_qm_off_in_c() {
    let (w, h) = (64usize, 64);
    let y = textured(w, h);
    let empty: Vec<u16> = Vec::new();
    // qm_min == qm_max == 6 pins a genuine non-flat QM level for every plane.
    // Sweep a few quality levels so at least one keeps enough coefficients for
    // the QM to bite (and assert it bites at every one).
    let mut differed = 0usize;
    for &cq in &[16i32, 32, 48] {
        let off = c::ref_encode_av1_kf(
            &y, &empty, &empty, w, h, 8, true, 1, 1, cq, 0, false, false, 2, 0, false,
        );
        let on = c::ref_encode_av1_kf_qm(
            &y, &empty, &empty, w, h, 8, true, 1, 1, cq, 0, false, false, 2, 0, false, 6, 6,
        );
        assert!(!off.is_empty(), "QM-off encode empty (cq={cq})");
        assert!(!on.is_empty(), "QM-on encode empty (cq={cq})");
        assert_ne!(
            off, on,
            "QM-on must change the C bitstream vs QM-off at cq={cq} (anti-vacuous)"
        );
        differed += 1;
    }
    assert_eq!(
        differed, 3,
        "all three cq levels must show a QM-on/off difference"
    );
}
