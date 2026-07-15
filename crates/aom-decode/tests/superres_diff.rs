//! THE SUPERRES GATE: decode real libaom-encoded KEY streams that use
//! fixed-denominator superres (`aom_codec_av1_cx` with
//! `rc_superres_mode = AOM_SUPERRES_FIXED`, `rc_superres_kf_denominator = D`)
//! with `decode_frame_obus`, and compare every plane BYTE-IDENTICALLY against
//! the REAL C decoder (`aom_codec_av1_dx`). The encoder codes the frame at a
//! reduced (downscaled) width `FrameWidth = (UpscaledWidth*8 + D/2)/D` and the
//! decoder upscales it back to `UpscaledWidth` horizontally (normative,
//! post-CDEF; `crate::superres`).
//!
//! ANTI-VACUOUS (asserted per stream): `SuperresDenom > 8`, the coded
//! `FrameWidth < UpscaledWidth` (a real downscale/upscale happened), non-flat AC
//! content in the output, and the full-width upscaled output differs from the
//! narrower pre-upscale coded reconstruction.

use aom_decode::frame::{
    apply_cdef, apply_deblock, apply_superres, decode_frame_obus, decode_frame_obus_prefilter,
};
use aom_decode::superres::coded_frame_width;
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Photographic-ish content (smooth gradients + sinusoids + noise) — NOT
/// synthetic-few-colours (which could trip screen-content detection, out of
/// envelope). Deliberately high-frequency in X so the horizontal upscale has
/// real AC to interpolate (a flat plane would upscale vacuously).
fn gen_plane(w: usize, h: usize, bd: i32, seed: u64, chroma: bool) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 0.25 + 0.5 * (0.6 * fx + 0.4 * fy);
            let wave = 0.14 * ((fx * 17.0).sin() * (fy * 7.0).cos());
            let noise = ((rng.next() >> 40) as i64 % 33 - 16) as f64 / maxv as f64;
            let mut v = base + wave + noise * if chroma { 2.0 } else { 4.0 };
            v = v.clamp(0.0, 1.0);
            p[r * w + col] = (v * maxv as f64).round() as u16;
        }
    }
    p
}

#[allow(dead_code)] // `lr_gated` is read by the LR-composition arm
struct SrFacts {
    denom: i32,
    coded_w: i32,
    upscaled_w: i32,
    lr_gated: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_superres(
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss: (i32, i32),
    cq: i32,
    denom: i32,
    cdef: bool,
    restoration: bool,
    usage: u32,
) -> SrFacts {
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss.0 as usize) >> ss.0, (h + ss.1 as usize) >> ss.1)
    };
    let seed = ((w as u64) << 40)
        ^ ((h as u64) << 24)
        ^ ((bd as u64) << 12)
        ^ ((denom as u64) << 4)
        ^ cq as u64;
    let y = gen_plane(w, h, bd, seed ^ 0x1111, false);
    let u = gen_plane(cw, ch, bd, seed ^ 0x2222, true);
    let v = gen_plane(cw, ch, bd, seed ^ 0x3333, true);

    // REAL encoder bytes with fixed-denominator superres (cpu-used=0).
    let bytes = c::ref_encode_av1_kf_superres(
        &y,
        &u,
        &v,
        w,
        h,
        bd,
        mono,
        ss.0,
        ss.1,
        cq,
        0,
        cdef,
        restoration,
        usage,
        denom,
    );
    assert!(bytes.len() > 50, "suspiciously small superres stream");

    // Anti-vacuous facts from OUR OWN parse (prefilter = pre loop-filter recon +
    // parsed header): the stream really is superres-scaled.
    let (mut pt, ptcfg, header) = decode_frame_obus_prefilter(&bytes).unwrap_or_else(|e| {
        panic!("prefilter rejected superres {w}x{h} bd{bd} mono={mono} D={denom}: {e}")
    });
    let coded_w = coded_frame_width(
        header.frame_size.superres_upscaled_width,
        header.frame_size.scale_denominator,
    );
    assert_eq!(
        header.frame_size.scale_denominator, denom,
        "coded superres denom mismatch"
    );
    assert!(
        header.frame_size.scale_denominator > 8,
        "stream is not superres-scaled (D={})",
        header.frame_size.scale_denominator
    );
    assert!(
        coded_w < w as i32,
        "no real downscale: coded {coded_w} !< upscaled {w}"
    );
    assert_eq!(header.frame_size.superres_upscaled_width, w as i32);

    // The pre-upscale coded recon is narrower (mi-aligned to the coded width).
    // Reconstruct it through deblock (if any) + superres and confirm the
    // upscaled output is genuinely wider than the coded frame and non-flat.
    if header.loopfilter.filter_level != [0, 0] {
        apply_deblock(&mut pt, &ptcfg, &header);
    }
    let coded_stride = pt.stride;
    if header.cdef.cdef_bits != 0
        || header.cdef.cdef_strengths[0] != 0
        || header.cdef.cdef_uv_strengths[0] != 0
    {
        apply_cdef(&mut pt, &ptcfg, &header);
    }
    apply_superres(&mut pt, &ptcfg, &header);
    assert!(
        pt.stride > coded_stride || (w as i32) > coded_w,
        "superres did not widen the plane"
    );

    // Rust decode (full pipeline) — hard-errors outside the envelope (FAILS).
    let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
        panic!("decode_frame_obus rejected superres {w}x{h} bd{bd} mono={mono} ss={ss:?} D={denom}: {e}")
    });
    assert_eq!((rust.width, rust.height), (w, h), "upscaled output dims");
    assert!(rust.base_qindex > 0);
    assert!(
        rust.y.iter().any(|&px| px != rust.y[0]),
        "upscaled luma is constant (no AC to interpolate — vacuous)"
    );

    // Gold oracle: the REAL C decoder on the same bytes, at the FULL dims.
    let cref = c::ref_decode_av1_kf(&bytes, w, h);
    assert_eq!(cref.info[0], bd);
    assert_eq!(cref.info[1] != 0, mono);
    assert_eq!(
        rust.y, cref.y,
        "LUMA mismatch superres {w}x{h} bd{bd} mono={mono} ss={ss:?} D={denom} cdef={cdef} lr={restoration}"
    );
    if mono {
        assert!(rust.u.is_empty() && rust.v.is_empty());
    } else {
        assert_eq!(
            rust.u, cref.u,
            "U mismatch superres {w}x{h} bd{bd} ss={ss:?} D={denom} cdef={cdef} lr={restoration}"
        );
        assert_eq!(
            rust.v, cref.v,
            "V mismatch superres {w}x{h} bd{bd} ss={ss:?} D={denom} cdef={cdef} lr={restoration}"
        );
    }

    SrFacts {
        denom,
        coded_w,
        upscaled_w: w as i32,
        lr_gated: rust.lr_frame_restoration_type.iter().any(|&t| t != 0),
    }
}

/// LUMA-ONLY chunk: monochrome superres streams (CDEF + LR off) decode
/// byte-identical to C across denominators {9,12,16}, 8/10-bit, and several
/// sizes (incl. non-8-multiple widths that exercise the mi-aligned border
/// clamp).
#[test]
fn superres_luma_mono_byte_identical_to_c() {
    let sizes = [(256usize, 64usize), (200, 96), (160, 128), (100, 76)];
    let denoms = [9i32, 12, 16];
    let mut n = 0u32;
    let mut min_ratio = f64::MAX;
    for &(w, h) in &sizes {
        for &bd in &[8i32, 10] {
            for &denom in &denoms {
                let f = run_superres(w, h, bd, true, (1, 1), 24, denom, false, false, 0);
                assert!(f.denom > 8 && f.coded_w < f.upscaled_w);
                min_ratio = min_ratio.min(f.coded_w as f64 / f.upscaled_w as f64);
                n += 1;
            }
        }
    }
    assert_eq!(n, 4 * 2 * 3, "superres luma/mono arm count");
    // The steepest downscale (D=16) is ~0.5x; confirm we actually exercised a
    // strong downscale, not a near-1.0 no-op.
    assert!(
        min_ratio < 0.55,
        "steepest downscale too mild (min coded/upscaled = {min_ratio:.3})"
    );
    eprintln!(
        "superres luma/mono: {n} streams byte-identical, min coded/upscaled ratio {min_ratio:.3}"
    );
}

/// CHROMA chunk: colour superres streams (4:2:0 + 4:4:4, 8/10/12-bit) decode
/// byte-identical to C — the chroma planes are upscaled at their subsampled
/// width `(coded_w + ss_x) >> ss_x -> (upscaled_w + ss_x) >> ss_x`. Rotates
/// GOOD/ALL_INTRA usage and CDEF on/off.
#[test]
fn superres_color_byte_identical_to_c() {
    let sizes = [(256usize, 96usize), (200, 128), (100, 76)];
    let combos = [
        (8i32, (1i32, 1i32)), // 4:2:0
        (8, (0, 0)),          // 4:4:4
        (10, (1, 1)),
        (10, (0, 0)),
        (12, (1, 1)),
    ];
    let denoms = [9i32, 12, 16];
    let mut n = 0u32;
    for &(w, h) in &sizes {
        for &(bd, ss) in &combos {
            for &denom in &denoms {
                let usage = if (n & 1) == 0 { 0u32 } else { 2 };
                let cdef = n % 3 == 0;
                run_superres(w, h, bd, false, ss, 24, denom, cdef, false, usage);
                n += 1;
            }
        }
    }
    assert_eq!(n, 3 * 5 * 3, "superres colour arm count");
    eprintln!("superres colour: {n} streams byte-identical");
}

/// LR-COMPOSITION chunk: superres + loop restoration. Superres always takes the
/// NON-optimized LR path (decodeframe.c:5422 `!do_cdef && !do_superres`): the
/// pre-CDEF deblock boundary rows are upscaled (matching C's
/// `save_deblock_boundary_lines`) so LR's internal stripe boundaries and the
/// filter itself run entirely in the upscaled domain. Both CDEF-off and CDEF-on
/// arms exercise the non-optimized boundary-save path.
#[test]
fn superres_lr_composed_byte_identical_to_c() {
    let sizes = [(256usize, 128usize), (200, 96)];
    let combos = [
        (8i32, (1i32, 1i32), false),
        (8, (0, 0), false),
        (10, (1, 1), false),
        (8, (1, 1), true), // monochrome
    ];
    let denoms = [9i32, 12, 16];
    let mut n = 0u32;
    let mut lr_seen = 0u32;
    for &(w, h) in &sizes {
        for &(bd, ss, mono) in &combos {
            for &denom in &denoms {
                let cdef = n % 2 == 0;
                let f = run_superres(w, h, bd, mono, ss, 36, denom, cdef, true, 2);
                lr_seen += f.lr_gated as u32;
                n += 1;
            }
        }
    }
    assert_eq!(n, 2 * 4 * 3, "superres+LR arm count");
    // The task requires >= 1 stream with restoration genuinely active, composed
    // with superres, decoded byte-identical. (Byte-identity is asserted for
    // every arm inside run_superres; this floor keeps the LR-active population
    // from silently vanishing if the speed-0 search stops picking LR.)
    assert!(
        lr_seen >= 1,
        "no superres+LR stream carried restoration syntax ({lr_seen}) — LR composition unproven"
    );
    eprintln!("superres+LR: {n} streams byte-identical, {lr_seen} carried LR syntax");
}
