//! #8 qindex-from-cq: prove the port DERIVES a KEY frame's `base_qindex` from
//! the user `--cq-level` byte-identically to the real encoder, instead of
//! reading it off the parsed header.
//!
//! Method (differential, top-tier evidence = real exported C fn): for a sweep
//! of `(cq_level, usage, bit_depth, subsampling)` the real encoder
//! (`ref_encode_av1_kf` == `shim_encode_av1_kf`, the constant-quality `AOM_Q`
//! single-KEY-frame envelope the encoder gates use) encodes a small frame; the
//! port parses the resulting stream and reads the real `quant.base_qindex`; the
//! test asserts [`aom_encode::rc::base_qindex_from_cq`] reproduces it exactly.
//!
//! The derivation under test is the pure `quantizer_to_qindex[cq]` table lookup
//! (`av1/encoder/av1_quantize.c:1033`); the C control flow that makes it exact
//! for this envelope — `av1_rc_pick_q_and_bounds` →
//! `rc_pick_q_and_bounds_q_mode` → `get_intra_q_and_bounds` branch
//! `ratectrl.c:1832` (`active_best = active_worst = cq_level`), then
//! `av1_set_quantizer`'s `AOMMAX(delta_q_present_flag, q)`
//! (`av1_quantize.c:884`) with delta-q off — is documented on
//! `base_qindex_from_cq`. Sweeping `usage {ALLINTRA, GOOD}` × `bd {8, 10, 12}` ×
//! `{4:2:0, 4:4:4, mono}` proves the claimed invariance to all of them.

use aom_encode::rc::base_qindex_from_cq;
use aom_sys_ref as c;

/// Mid-range deterministic texture sized EXACTLY as `ref_encode_av1_kf` asserts
/// (`y.len() == w*h`, `u/v.len() == cw*ch` with `cw = (w + ss_x) >> ss_x`).
fn gen_planes(
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let maxv = (1u32 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let base = ((r * 5 + col * 3) as u32) % (maxv / 2 + 1);
            y[r * w + col] = (base + maxv / 4).min(maxv) as u16;
        }
    }
    if mono {
        return (y, Vec::new(), Vec::new());
    }
    let cw = (w + ss_x as usize) >> ss_x;
    let ch = (h + ss_y as usize) >> ss_y;
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = (maxv / 2 + ((r + col) as u32 % 7)).min(maxv) as u16;
            v[r * cw + col] = (maxv / 2).saturating_sub((r + col) as u32 % 5) as u16;
        }
    }
    (y, u, v)
}

/// Encode with the real C encoder in the primary envelope (cpu-used=0, cdef &
/// restoration off, aq_mode 0, one pass) and return the real header's
/// `base_qindex` as the port parses it back.
#[allow(clippy::too_many_arguments)]
fn real_base_qindex(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss_x: i32,
    ss_y: i32,
    cq: i32,
    usage: u32,
) -> i32 {
    let bytes = c::ref_encode_av1_kf(
        y, u, v, w, h, bd, mono, ss_x, ss_y, cq, 0, false, false, usage, 0, false,
    );
    assert!(
        !bytes.is_empty(),
        "C encode produced no bytes (bd={bd} usage={usage} cq={cq} mono={mono})"
    );
    let (_dec, _cfg, hdr) = aom_decode::frame::decode_frame_obus_prefilter(&bytes).unwrap_or_else(
        |e| panic!("decode failed (bd={bd} usage={usage} cq={cq} mono={mono} ss=({ss_x},{ss_y})): {e}"),
    );
    hdr.quant.base_qindex
}

#[test]
fn base_qindex_derived_from_cq_matches_c() {
    let (w, h) = (64usize, 64usize);
    let mut checks = 0usize;
    let mut distinct = std::collections::BTreeSet::new();
    let (mut saw0, mut saw249, mut saw255) = (false, false, false);

    // Exhaustive over the whole cq table × usage × bit depth on 4:2:0 — this is
    // the axis under test (the table) plus the two invariance axes.
    for bd in [8i32, 10, 12] {
        let (y, u, v) = gen_planes(w, h, bd, false, 1, 1);
        for usage in [2u32, 0] {
            for cq in 0..=63i32 {
                let real = real_base_qindex(&y, &u, &v, w, h, bd, false, 1, 1, cq, usage);
                let derived = base_qindex_from_cq(cq);
                assert_eq!(
                    derived, real,
                    "base_qindex mismatch bd={bd} usage={usage} cq={cq}: derived {derived} != real {real}"
                );
                distinct.insert(real);
                saw0 |= real == 0;
                saw249 |= real == 249;
                saw255 |= real == 255;
                checks += 1;
            }
        }
    }

    // Subsampling / monochrome invariance spot-check (usage=2, bd=8), including
    // both table endpoints (cq 0 -> lossless qindex 0, cq 63 -> 255).
    for (mono, ss_x, ss_y) in [(false, 0i32, 0i32), (true, 1i32, 1i32)] {
        let (y, u, v) = gen_planes(w, h, 8, mono, ss_x, ss_y);
        for cq in [0i32, 32, 62, 63] {
            let real = real_base_qindex(&y, &u, &v, w, h, 8, mono, ss_x, ss_y, cq, 2);
            assert_eq!(
                base_qindex_from_cq(cq),
                real,
                "geom mono={mono} ss=({ss_x},{ss_y}) cq={cq}: derived != real {real}"
            );
            checks += 1;
        }
    }

    // Anti-vacuity: the sweep must genuinely exercise the whole table and the
    // derivation must be a real (non-identity) transform.
    assert_eq!(checks, 64 * 2 * 3 + 2 * 4, "unexpected check count {checks}");
    assert!(
        distinct.len() >= 60,
        "cq 0..=63 should map to ~64 distinct qindices, saw {} ({distinct:?})",
        distinct.len()
    );
    assert!(
        saw0 && saw249 && saw255,
        "must observe table endpoints qindex 0/249/255 (saw0={saw0} saw249={saw249} saw255={saw255})"
    );
    assert_ne!(
        base_qindex_from_cq(62),
        62,
        "the cq->qindex table must not be the identity"
    );
}
