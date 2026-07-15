//! THE `disable_cdf_update` DECODE GATE: decode real libaom v3.14.1 bitstreams
//! whose uncompressed header carries `disable_cdf_update = 1` (the encoder's
//! `--cdf-update-mode=0` / `AV1E_SET_CDF_UPDATE_MODE=0`, which forces the flag
//! on for every frame) and compare every plane BYTE-IDENTICALLY against the REAL
//! C decoder (`aom_codec_av1_dx`).
//!
//! WHAT THE FLAG DOES (traced): `aom_dsp/bitreader.h`'s `aom_read_symbol_`
//! adapts the CDF only `if (r->allow_update_cdf)`. The decoder derives
//! `allow_update_cdf = (!tiles.large_scale) && !features.disable_cdf_update`
//! (`av1/decoder/decodeframe.c`) and stores it on the reader
//! (`r->allow_update_cdf = allow_update_cdf`). A single-tile KEY frame is never
//! `large_scale`, so with `disable_cdf_update = 1` the reader NEVER adapts: every
//! symbol read leaves its CDF at the loaded/initial value for the whole tile.
//! The port mirrors this exactly — `OdEcDec::allow_update_cdf` set to
//! `!disable_cdf_update` in `decode_frame_tiles_kf`, gated in
//! `aom_entropy::read_symbol`, which is the sole `update_cdf` site (both the
//! mode-info reads and the `aom_txb` coefficient reads route through it).
//!
//! ANTI-VACUOUS (this gate is not a no-op):
//! 1. Each decoded stream's parsed header is asserted to carry
//!    `disable_cdf_update = 1` (from the port's OWN parse), so the non-adapting
//!    path is genuinely exercised.
//! 2. Real AC content is asserted present (the decoded luma is far from flat),
//!    so the streams actually code coefficients through the gated reader.
//! 3. The same source image re-encoded WITHOUT the flag produces a DIFFERENT
//!    bitstream, confirming `disable_cdf_update` materially changes the coded
//!    representation the decoder must handle.
//! 4. `disable_cdf_update_gate_is_load_bearing` proves at the symbol layer that
//!    flipping `allow_update_cdf` changes both the adapted CDF state AND the
//!    decoded symbol values on an adapting-coded stream — i.e. the gate is real.
//!
//! ENVELOPE: one shown KEY frame per stream, single tile, 64x64 SBs, no
//! superres / film grain / screen-content tools / qm / lossless. 4:2:0 + 4:4:4 +
//! monochrome, bit depths 8/10/12. (4:2:2 is excluded — its chroma-deblock path
//! is rejected upstream, unrelated to this flag.)

use aom_decode::frame::{decode_frame_obus, decode_frame_obus_prefilter};
use aom_entropy::{OdEcDec, OdEcEnc, read_symbol, write_symbol};
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

/// Photographic-ish content: smooth gradients + sinusoidal structure + noise
/// (NOT flat/few-colors, which could trip the encoder's screen-content
/// detection — screen-content tools are out of the decode envelope). Identical
/// in spirit to the real_bitstream.rs generator.
fn gen_plane(w: usize, h: usize, bd: i32, seed: u64, chroma: bool) -> Vec<u16> {
    let mut rng = Rng(seed | 1);
    let maxv = (1i64 << bd) - 1;
    let mut p = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let fx = col as f64 / w.max(1) as f64;
            let fy = r as f64 / h.max(1) as f64;
            let base = 0.25 + 0.5 * (0.6 * fx + 0.4 * fy);
            let wave = 0.12 * ((fx * 9.0).sin() * (fy * 7.0).cos());
            let noise = ((rng.next() >> 40) as i64 % 33 - 16) as f64 / maxv as f64;
            let mut v = base + wave + noise * if chroma { 2.0 } else { 4.0 };
            v = v.clamp(0.0, 1.0);
            p[r * w + col] = (v * maxv as f64).round() as u16;
        }
    }
    p
}

/// A luma plane has real AC content when it is far from flat: many distinct
/// values AND non-trivial local (horizontal) gradient energy. This rules out a
/// degenerate all-DC / all-skip stream where the gated coefficient reader would
/// barely fire.
fn has_ac_content(y: &[u16], w: usize, h: usize) -> bool {
    let (mut lo, mut hi) = (u16::MAX, 0u16);
    for &p in y {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    if hi - lo < 8 {
        return false;
    }
    let mut grad: u64 = 0;
    for r in 0..h {
        for col in 1..w {
            grad += (y[r * w + col] as i64 - y[r * w + col - 1] as i64).unsigned_abs();
        }
    }
    // Average |dx| of at least ~1 code value across the frame.
    grad >= (w.saturating_sub(1) * h) as u64
}

struct Cell {
    w: usize,
    h: usize,
    bd: i32,
    mono: bool,
    ss: (i32, i32),
    cq: i32,
    fmt: &'static str,
}

fn run_cell(cell: &Cell) {
    let (cw, ch) = if cell.mono {
        (0, 0)
    } else {
        (
            (cell.w + cell.ss.0 as usize) >> cell.ss.0,
            (cell.h + cell.ss.1 as usize) >> cell.ss.1,
        )
    };
    let seed = ((cell.w as u64) << 32)
        ^ ((cell.h as u64) << 20)
        ^ ((cell.bd as u64) << 12)
        ^ ((cell.mono as u64) << 8)
        ^ cell.cq as u64;
    let y = gen_plane(cell.w, cell.h, cell.bd, seed ^ 0x1111, false);
    let u = gen_plane(cw, ch, cell.bd, seed ^ 0x2222, true);
    let v = gen_plane(cw, ch, cell.bd, seed ^ 0x3333, true);

    // REAL libaom encode with AV1E_SET_CDF_UPDATE_MODE=0 -> disable_cdf_update=1.
    let bytes = c::ref_encode_av1_kf_disable_cdf(
        &y, &u, &v, cell.w, cell.h, cell.bd, cell.mono, cell.ss.0, cell.ss.1, cell.cq, 0,
        /*enable_cdef=*/ true, /*enable_restoration=*/ false, /*usage GOOD=*/ 0,
    );

    // Port full-envelope decode (hard-errors outside the envelope -> FAILS).
    let rust = decode_frame_obus(&bytes).unwrap_or_else(|e| {
        panic!(
            "decode_frame_obus rejected {} {}x{} bd{} cq{}: {e}",
            cell.fmt, cell.w, cell.h, cell.bd, cell.cq
        )
    });

    // Anti-vacuous #1: the flag is genuinely set (port's own parse). Confirm via
    // both the FrameDecode fact AND the lower-level parsed header + config.
    assert!(
        rust.disable_cdf_update,
        "{} {}x{} bd{} cq{}: FrameDecode.disable_cdf_update must be set",
        cell.fmt, cell.w, cell.h, cell.bd, cell.cq
    );
    let (_pre, cfg, header) = decode_frame_obus_prefilter(&bytes).unwrap();
    assert!(
        header.prefix.disable_cdf_update && cfg.disable_cdf_update,
        "{} {}x{} bd{} cq{}: parsed header/config must carry disable_cdf_update=1",
        cell.fmt, cell.w, cell.h, cell.bd, cell.cq
    );
    assert!(rust.base_qindex > 0, "not coded-lossless (base_qindex > 0)");
    assert_eq!((rust.width, rust.height), (cell.w, cell.h));
    assert_eq!(rust.bit_depth, cell.bd);
    assert_eq!(rust.monochrome, cell.mono);

    // Anti-vacuous #2: real AC content present.
    assert!(
        has_ac_content(&rust.y, cell.w, cell.h),
        "{} {}x{} bd{} cq{}: decoded luma has no AC content (degenerate stream)",
        cell.fmt, cell.w, cell.h, cell.bd, cell.cq
    );

    // MAIN GATE: byte-identical to the REAL C decoder on every plane.
    let cref = c::ref_decode_av1_kf(&bytes, cell.w, cell.h);
    assert_eq!(cref.info[0], cell.bd, "C bit depth");
    assert_eq!(cref.info[1] != 0, cell.mono, "C monochrome");
    assert_eq!(
        rust.y, cref.y,
        "{} {}x{} bd{} cq{}: LUMA differs from C",
        cell.fmt, cell.w, cell.h, cell.bd, cell.cq
    );
    if !cell.mono {
        assert_eq!(
            rust.u, cref.u,
            "{} {}x{} bd{} cq{}: U differs from C",
            cell.fmt, cell.w, cell.h, cell.bd, cell.cq
        );
        assert_eq!(
            rust.v, cref.v,
            "{} {}x{} bd{} cq{}: V differs from C",
            cell.fmt, cell.w, cell.h, cell.bd, cell.cq
        );
    }

    // Anti-vacuous #3: the SAME image re-encoded WITHOUT the flag (the ordinary
    // adapting encoder) is a genuinely different bitstream, so disable_cdf_update
    // materially changes the coded representation the decoder handles.
    let bytes_adapt = c::ref_encode_av1_kf(
        &y, &u, &v, cell.w, cell.h, cell.bd, cell.mono, cell.ss.0, cell.ss.1, cell.cq, 0, true,
        false, 0, 0, false,
    );
    // The adapting encode must NOT carry the flag (sanity on the control split).
    let adapt = decode_frame_obus(&bytes_adapt).unwrap();
    assert!(
        !adapt.disable_cdf_update,
        "{} {}x{} bd{}: adapting encode must have disable_cdf_update=0",
        cell.fmt, cell.w, cell.h, cell.bd
    );
    assert_ne!(
        bytes, bytes_adapt,
        "{} {}x{} bd{} cq{}: disable vs adapt encodes must differ",
        cell.fmt, cell.w, cell.h, cell.bd, cell.cq
    );
}

#[test]
fn disable_cdf_update_streams_decode_byte_identical_to_c() {
    // 4:2:0, 4:4:4, monochrome. (4:2:2 excluded — chroma-deblock reject.)
    let formats: [(bool, (i32, i32), &str); 3] = [
        (false, (1, 1), "420"),
        (false, (0, 0), "444"),
        (true, (1, 1), "mono"),
    ];
    let sizes: [(usize, usize); 3] = [(64, 64), (96, 80), (128, 128)];
    let cqs: [i32; 2] = [20, 48];

    let mut n = 0usize;
    let mut bd_seen = [false; 3]; // 8,10,12
    let mut fmt_seen = [false; 3]; // 420,444,mono
    for &(mono, ss, fmt) in &formats {
        // 8/10-bit for every format; 12-bit too (highbd across chroma types).
        for &bd in &[8, 10, 12] {
            for &(w, h) in &sizes {
                for &cq in &cqs {
                    run_cell(&Cell {
                        w,
                        h,
                        bd,
                        mono,
                        ss,
                        cq,
                        fmt,
                    });
                    n += 1;
                    bd_seen[match bd {
                        8 => 0,
                        10 => 1,
                        _ => 2,
                    }] = true;
                    fmt_seen[match fmt {
                        "420" => 0,
                        "444" => 1,
                        _ => 2,
                    }] = true;
                }
            }
        }
    }

    // Coverage floors: every bit depth and every format actually ran, and the
    // stream count is well above the 8-stream minimum.
    assert!(bd_seen.iter().all(|&b| b), "all bit depths 8/10/12 covered");
    assert!(fmt_seen.iter().all(|&f| f), "all formats 420/444/mono covered");
    assert_eq!(n, 3 * 3 * 3 * 2, "cell count");
    assert!(n >= 8, "at least 8 real streams");
}

/// Proves `read_symbol`'s `allow_update_cdf` gate is load-bearing: on a stream
/// coded by an ADAPTING encoder, decoding with the flag SET reproduces the
/// symbols and adapts the CDF, while decoding the SAME bytes with the flag
/// CLEARED both freezes the CDF (deterministic state divergence) AND decodes
/// different symbol values (observable output divergence). This is the exact
/// mechanism `disable_cdf_update` toggles in the real decode path.
#[test]
fn disable_cdf_update_gate_is_load_bearing() {
    const N: usize = 4; // symbols per CDF
    // A valid AV1 icdf (Q15, length N+1, cdf[N-1]==0, cdf[N]==adaptation count):
    // a non-uniform initial distribution so adaptation visibly skews boundaries.
    let init: [u16; N + 1] = [32768 - 4096, 32768 - 12288, 32768 - 26624, 0, 0];

    // A strongly-biased, deterministic symbol sequence: adaptation skews the CDF
    // hard toward symbol 3, so a frozen decoder desynchronises quickly.
    let mut seq = Vec::new();
    let mut r = Rng(0x00C0_FFEE_1234_5678);
    for _ in 0..96 {
        let v = (r.next() >> 33) % 8;
        seq.push(if v < 5 { 3 } else { (v % 4) as i32 });
    }

    // Encode with adaptation ON (the ordinary write_symbol path).
    let mut enc = OdEcEnc::new();
    let mut cdf_e = init;
    for &s in &seq {
        write_symbol(&mut enc, s, &mut cdf_e, N);
    }
    let bytes = enc.done().to_vec();

    // Decode with the flag SET (allow_update_cdf = true): must reproduce + adapt.
    let mut d_on = OdEcDec::new(&bytes);
    d_on.allow_update_cdf = true;
    let mut cdf_on = init;
    let out_on: Vec<i32> = (0..seq.len())
        .map(|_| read_symbol(&mut d_on, &mut cdf_on, N))
        .collect();
    assert_eq!(out_on, seq, "flag-on decode must reproduce the adapting stream");
    assert_eq!(
        cdf_on, cdf_e,
        "flag-on decoder CDF must track the encoder's adapted CDF"
    );

    // Decode the SAME bytes with the flag CLEARED (allow_update_cdf = false).
    let mut d_off = OdEcDec::new(&bytes);
    d_off.allow_update_cdf = false;
    let mut cdf_off = init;
    let out_off: Vec<i32> = (0..seq.len())
        .map(|_| read_symbol(&mut d_off, &mut cdf_off, N))
        .collect();

    // Deterministic CDF-state divergence: the frozen CDF stays at its init value.
    assert_eq!(cdf_off, init, "flag-off decoder must NOT adapt its CDF");
    assert_ne!(
        cdf_off, cdf_on,
        "adapted vs frozen CDF state must differ (gate changes state)"
    );
    // Observable output divergence: frozen CDFs mis-decode the adapting stream.
    assert_ne!(
        out_off, seq,
        "flag-off decode of an adapting stream must diverge (gate changes output)"
    );
}
