//! `--min-q` / `--max-q` (task: the qindex clamp bounds) differential:
//! prove the port DERIVES a KEY frame's `base_qindex` from `(--cq-level,
//! --min-q, --max-q)` byte-identically to the real encoder, extending #8
//! (`qindex_from_cq_diff.rs`, the unclamped `--min-q=0 --max-q=63` case) with
//! the `rc_min/max_quantizer` clamp.
//!
//! Method (differential, top-tier evidence = real exported C path): for a sweep
//! of `(cq, min_q, max_q)` the real encoder encodes a small KEY frame with those
//! `cfg.rc_min_quantizer` / `rc_max_quantizer`; the port parses the resulting
//! stream and reads the real `quant.base_qindex`; the test asserts
//! [`aom_encode::rc::base_qindex_from_cq_clamped`] reproduces it exactly. The
//! sweep is chosen so the clamp fires DOWN (cq above max_q), fires UP (cq below
//! min_q), and stays inert — each observed at least once (anti-vacuity).

use aom_encode::rc::{base_qindex_from_cq, base_qindex_from_cq_clamped};
use aom_sys_ref as c;

fn gen_planes(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = (((r * 5 + col * 3) % 200) + 20) as u16;
        }
    }
    let (cw, ch) = ((w + 1) >> 1, (h + 1) >> 1);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = (128 + ((r + col) % 7)) as u16;
            v[r * cw + col] = (128u16).saturating_sub(((r + col) % 5) as u16);
        }
    }
    (y, u, v)
}

fn real_base_qindex(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    cq: i32,
    min_q: i32,
    max_q: i32,
) -> i32 {
    let bytes = c::ref_encode_av1_kf_minmaxq(y, u, v, w, h, 8, false, 1, 1, cq, 0, 2, min_q, max_q);
    assert!(
        !bytes.is_empty(),
        "C encode produced no bytes (cq={cq} min_q={min_q} max_q={max_q})"
    );
    let (_dec, _cfg, hdr) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("decode failed (cq={cq} min_q={min_q} max_q={max_q}): {e}"));
    hdr.quant.base_qindex
}

#[test]
fn base_qindex_min_max_q_clamp_matches_c() {
    let (w, h) = (64usize, 64usize);
    let (y, u, v) = gen_planes(w, h);
    let mut checks = 0usize;
    let (mut fired_down, mut fired_up, mut inert) = (false, false, false);

    // (cq, min_q, max_q) triples covering: default (inert), clamp-down (cq above
    // max), clamp-up (cq below min), and equal bounds (fixed qindex).
    let cases: &[(i32, i32, i32)] = &[
        (32, 0, 63),  // default bounds -> inert
        (63, 0, 63),  // endpoint, inert
        (0, 0, 63),   // endpoint, inert
        (50, 0, 20),  // cq50 (q200) clamped DOWN to max_q20 (q80)
        (40, 0, 30),  // cq40 (q160) clamped DOWN to max_q30 (q120)
        (63, 0, 40),  // cq63 (q255) clamped DOWN to max_q40 (q160)
        (5, 20, 63),  // cq5 (q20) clamped UP to min_q20 (q80)
        (0, 10, 63),  // cq0 (q0) clamped UP to min_q10 (q40)
        (12, 30, 63), // cq12 (q48) clamped UP to min_q30 (q120)
        (32, 32, 32), // min==max==cq -> fixed q128 (inert here)
        (10, 32, 32), // min==max, cq below -> clamped UP to q128
        (60, 32, 32), // min==max, cq above -> clamped DOWN to q128
        (25, 15, 45), // interior, inert (q100 within [q60,q180])
    ];

    for &(cq, min_q, max_q) in cases {
        let real = real_base_qindex(&y, &u, &v, w, h, cq, min_q, max_q);
        let derived = base_qindex_from_cq_clamped(cq, min_q, max_q);
        assert_eq!(
            derived, real,
            "base_qindex mismatch cq={cq} min_q={min_q} max_q={max_q}: derived {derived} != real {real}"
        );
        let unclamped = base_qindex_from_cq(cq);
        if real < unclamped {
            fired_down = true;
        } else if real > unclamped {
            fired_up = true;
        } else {
            inert = true;
        }
        checks += 1;
    }

    // The unclamped derivation must equal the (0,63) clamped one (consistency).
    for cq in 0..=63 {
        assert_eq!(
            base_qindex_from_cq(cq),
            base_qindex_from_cq_clamped(cq, 0, 63)
        );
    }

    assert_eq!(checks, cases.len());
    assert!(fired_down, "no clamp-DOWN case observed (max_q must bite)");
    assert!(fired_up, "no clamp-UP case observed (min_q must bite)");
    assert!(inert, "no inert case observed");
}
