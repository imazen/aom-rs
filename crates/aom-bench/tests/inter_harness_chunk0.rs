//! INTER-ENCODE Chunk 0 GATE — the multi-frame encode harness + decode-both
//! localizer (INTER-ENCODE-ROADMAP.md chunk 0).
//!
//! Chunk 0 is test/verification INFRASTRUCTURE that unblocks the inter-encode
//! skeleton (chunk 2). The gate here is that the infra WORKS — a real (not
//! stubbed) `aomenc` 2-frame `[KEY, P]` reference, a decode-both localizer that
//! pins divergences, and a `MultiFrameEncodeCell` carrying the source — NOT a
//! byte-exact inter ENCODE (that is chunk 2). What this asserts:
//!
//! 1. The C-encode reference ([`MultiFrameEncodeCell::c_encode_inter`]) produces
//!    a valid KEY + single-ref-translational-P stream the port decoder decodes,
//!    with frame 0 (KEY) byte-exact vs C (the regression control).
//! 2. The harness verifies a byte-exact P END TO END on the cases inside the
//!    port inter decoder's envelope (mono luma-inter, zero-MV 4:2:0) — a real
//!    positive gate, proving the infra can confirm inter byte-exactness.
//! 3. The localizer reports ZERO divergence on identical streams and the EXACT
//!    offset on a corrupted copy.
//! 4. The decoder-envelope map (which `--cpu-used` levels `aomenc`'s simplest P
//!    stays byte-exact-decodable at) — the finding that BOUNDS the chunk-2
//!    target config.

use aom_bench::inter_localize::{
    Divergence, FrameView, Plane, SB64_PX, decode_both, first_frameset_divergence,
    try_decode_frames,
};
use aom_bench::{EncodeCell, MultiFrameEncodeCell};
use aom_dsp::entropy::obu::read_obu_header;

// ---------------------------------------------------------------------------
// Content + stream helpers
// ---------------------------------------------------------------------------

/// A textured base frame (frame 0 of a cell), usage = GOOD (the inter context).
fn textured_base(label: &str, w: usize, h: usize, mono: bool, cq: i32, speed: i32) -> EncodeCell {
    let content = |r: usize, c: usize| -> u16 { (40 + ((r * 3 + c * 5) % 160)) as u16 };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = content(r, c);
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + 1) >> 1, (h + 1) >> 1)
    };
    let cont_uv = |r: usize, c: usize| -> u16 { (110 + ((r * 2 + c) % 40)) as u16 };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = cont_uv(r, c);
                v[r * cw + c] = cont_uv(r, c) + 3;
            }
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x: if mono { 1 } else { 1 },
        ss_y: if mono { 1 } else { 1 },
        usage: 0, // GOOD_QUALITY (inter context; not all-intra)
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

const OBU_TEMPORAL_DELIMITER: u32 = 2;
const OBU_FRAME_HEADER: u32 = 3;
const OBU_FRAME: u32 = 6;

/// Per-temporal-unit coded frame type (KEY=0, INTER=1, INTRA_ONLY=2, SWITCH=3),
/// read from the first byte of each TU's frame OBU (the chunk-0 seq header has
/// no `show_existing`/`frame_id` prefix bits, so `frame_type` is bits 6-5).
fn tu_frame_types(stream: &[u8]) -> Vec<u8> {
    let mut pos = 0usize;
    let mut types: Vec<u8> = Vec::new();
    let mut cur: Option<u8> = None;
    while pos < stream.len() {
        let h = read_obu_header(&stream[pos..]).expect("valid OBU header");
        let after = pos + h.header_len;
        let (size, sb) = aom_dsp::entropy::leb128::uleb_decode(&stream[after..]).expect("leb128");
        if h.obu_type == OBU_TEMPORAL_DELIMITER
            && let Some(t) = cur.take()
        {
            types.push(t);
        }
        if h.obu_type == OBU_FRAME || h.obu_type == OBU_FRAME_HEADER {
            cur = Some((stream[after + sb] >> 5) & 0x3);
        }
        pos = after + sb + size as usize;
    }
    if let Some(t) = cur.take() {
        types.push(t);
    }
    types
}

/// The port's multi-frame decode of `stream`, panic-safe.
fn port_frames(stream: &[u8]) -> Vec<aom_decode::frame::FrameDecode> {
    try_decode_frames(stream).unwrap_or_else(|e| panic!("port decode_frames failed: {e}"))
}

/// Localize the port's decode of `cell`'s `aomenc` stream against the C decode
/// (per shown frame) — the chunk-0 measurement of the port inter decoder's
/// envelope vs the reference. Returns `(stream, port_frame_count, divergence)`.
fn port_vs_c(
    cell: &MultiFrameEncodeCell,
    cdef: bool,
    lr: bool,
) -> (Vec<u8>, usize, Option<Divergence>) {
    aom_sys_ref::ref_init();
    let stream = cell.c_encode_inter(cdef, lr);
    let pf = port_frames(&stream);
    let cf: Vec<aom_sys_ref::RefDecodedFrame> = (0..pf.len())
        .map(|i| aom_sys_ref::ref_decode_av1_stream_frame(&stream, i, cell.w, cell.h))
        .collect();
    let pv: Vec<FrameView> = pf.iter().map(FrameView::of_decode).collect();
    let cv: Vec<FrameView> = cf.iter().map(FrameView::of_ref_decoded).collect();
    let div = first_frameset_divergence(&pv, &cv, SB64_PX);
    (stream, pf.len(), div)
}

// ---------------------------------------------------------------------------
// 1. C-encode reference: valid KEY+P stream, port-decodable, frame 0 byte-exact
// ---------------------------------------------------------------------------

#[test]
fn chunk0_c_encode_reference_produces_decodable_key_p_stream() {
    aom_sys_ref::ref_init();
    let base = textured_base("tex_420_64", 64, 64, false, 60, 0);
    let cell = MultiFrameEncodeCell::translational(&base, 3, 0);
    let stream = cell.c_encode_inter(/*cdef=*/ false, /*lr=*/ false);
    assert!(!stream.is_empty(), "shim produced an empty stream");

    // The reference must be a real KEY + single-ref translational P (not a stub,
    // not KEY+KEY).
    let types = tu_frame_types(&stream);
    assert_eq!(
        types,
        vec![0u8, 1u8],
        "expected [KEY, INTER]; got frame types {types:?}"
    );

    // The port's multi-frame decoder decodes both shown frames.
    let pf = port_frames(&stream);
    assert_eq!(pf.len(), 2, "port decoded {} frames, expected 2", pf.len());

    // Frame 0 (KEY) is byte-exact vs C through the 2-frame harness — the
    // regression control (single-frame KEY decode is Gate-1 byte-exact).
    let c0 = aom_sys_ref::ref_decode_av1_stream_frame(&stream, 0, cell.w, cell.h);
    let p0 = FrameView::of_decode(&pf[0]);
    let v0 = FrameView::of_ref_decoded(&c0);
    assert_eq!(
        first_frameset_divergence(&[p0], &[v0], SB64_PX),
        None,
        "frame 0 (KEY) regressed vs C through the 2-frame harness"
    );
}

// ---------------------------------------------------------------------------
// 2. In-envelope P byte-exact END TO END — the harness CAN verify inter parity
// ---------------------------------------------------------------------------

#[test]
fn chunk0_in_envelope_p_frame_byte_exact_end_to_end() {
    // Mono 64x64, real MV (dx=3): luma-only single-ref translational P. The port
    // inter decoder reconstructs it byte-exact vs C (probed).
    let mono_base = textured_base("tex_mono_64", 64, 64, true, 60, 0);
    let mono_cell = MultiFrameEncodeCell::translational(&mono_base, 3, 0);
    let (_s, n, div) = port_vs_c(&mono_cell, false, false);
    assert_eq!(n, 2, "mono cell decoded {n} frames");
    assert_eq!(
        div,
        None,
        "mono luma-inter P must decode byte-exact vs C, got: {}",
        div.as_ref().map(|d| d.to_string()).unwrap_or_default()
    );

    // 4:2:0 64x64, zero MV (dx=0): a degenerate near-skip P (both planes). The
    // decoder is byte-exact here (probed).
    let base420 = textured_base("tex_420_64_zmv", 64, 64, false, 60, 0);
    let zmv_cell = MultiFrameEncodeCell::translational(&base420, 0, 0);
    let (_s2, n2, div2) = port_vs_c(&zmv_cell, false, false);
    assert_eq!(n2, 2, "zero-MV cell decoded {n2} frames");
    assert_eq!(
        div2,
        None,
        "zero-MV 4:2:0 P must decode byte-exact vs C, got: {}",
        div2.as_ref().map(|d| d.to_string()).unwrap_or_default()
    );
}

// ---------------------------------------------------------------------------
// 3a. Localizer: ZERO divergence on identical streams (decode-both soundness)
// ---------------------------------------------------------------------------

#[test]
fn chunk0_localizer_zero_divergence_on_identical_streams() {
    aom_sys_ref::ref_init();
    let base = textured_base("tex_420_64", 64, 64, false, 60, 0);
    let cell = MultiFrameEncodeCell::translational(&base, 3, 0);
    let stream = cell.c_encode_inter(false, false);

    // decode-both of the SAME stream: deterministic decode -> identical
    // frame-sets -> no divergence.
    assert_eq!(
        decode_both(&stream, &stream, SB64_PX).expect("both decode"),
        None,
        "decode-both of one stream must report zero divergence"
    );

    // The pure comparator over identical frame-sets is also None.
    let pf = port_frames(&stream);
    let pv: Vec<FrameView> = pf.iter().map(FrameView::of_decode).collect();
    assert_eq!(
        first_frameset_divergence(&pv, &pv, SB64_PX),
        None,
        "comparator over identical frame-sets must be None"
    );
}

// ---------------------------------------------------------------------------
// 3b. Localizer: pins a corrupted copy at the EXACT known offset
// ---------------------------------------------------------------------------

#[test]
fn chunk0_localizer_pins_known_offset_corruption() {
    aom_sys_ref::ref_init();
    // 128x128 4:2:0 so the SB grid has 4 superblocks (non-trivial SB mapping).
    let base = textured_base("tex_420_128", 128, 128, false, 60, 0);
    let cell = MultiFrameEncodeCell::translational(&base, 3, 0);
    let stream = cell.c_encode_inter(false, false);
    let orig = port_frames(&stream);
    assert_eq!(orig.len(), 2);

    // (a) Corrupt a LUMA sample in frame 1 at a KNOWN offset in SB(1,0).
    {
        let mut corrupted = orig.clone();
        let (row, col) = (70usize, 10usize); // luma -> SB(70/64, 10/64) = (1, 0)
        let idx = row * corrupted[1].width + col;
        corrupted[1].y[idx] ^= 0x1; // flip one bit -> guaranteed !=
        let ov: Vec<FrameView> = orig.iter().map(FrameView::of_decode).collect();
        let cv: Vec<FrameView> = corrupted.iter().map(FrameView::of_decode).collect();
        match first_frameset_divergence(&ov, &cv, SB64_PX) {
            Some(Divergence::Sample {
                frame,
                plane,
                row: r,
                col: c,
                sb_row,
                sb_col,
                ..
            }) => {
                assert_eq!(
                    (frame, plane, r, c, sb_row, sb_col),
                    (1, Plane::Y, row, col, 1, 0),
                    "luma corruption localized to the wrong place"
                );
            }
            other => panic!("expected a luma Sample divergence, got {other:?}"),
        }
    }

    // (b) Corrupt a CHROMA (U) sample in frame 1 at a KNOWN offset; the SB label
    // maps through 4:2:0 subsampling to luma coords.
    {
        let mut corrupted = orig.clone();
        let (crow, ccol) = (40usize, 5usize); // chroma -> luma (80,10) -> SB(1,0)
        let idx = crow * corrupted[1].width_uv + ccol;
        corrupted[1].u[idx] ^= 0x1;
        let ov: Vec<FrameView> = orig.iter().map(FrameView::of_decode).collect();
        let cv: Vec<FrameView> = corrupted.iter().map(FrameView::of_decode).collect();
        match first_frameset_divergence(&ov, &cv, SB64_PX) {
            Some(Divergence::Sample {
                frame,
                plane,
                row: r,
                col: c,
                sb_row,
                sb_col,
                ..
            }) => {
                assert_eq!(
                    (frame, plane, r, c, sb_row, sb_col),
                    (1, Plane::U, crow, ccol, 1, 0),
                    "chroma corruption localized to the wrong place"
                );
            }
            other => panic!("expected a chroma Sample divergence, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 3c. Localizer: an asymmetric decode failure is itself a divergence
// ---------------------------------------------------------------------------

#[test]
fn chunk0_localizer_reports_decode_failure_as_divergence() {
    aom_sys_ref::ref_init();
    let base = textured_base("tex_420_64", 64, 64, false, 60, 0);
    let cell = MultiFrameEncodeCell::translational(&base, 3, 0);
    let good = cell.c_encode_inter(false, false);
    let garbage = vec![0xFFu8; 32]; // not a decodable stream

    // A stream that decodes vs one that does not -> DecodeError divergence.
    match decode_both(&good, &garbage, SB64_PX).expect("A decodes") {
        Some(Divergence::DecodeError { a_ok, b_ok, .. }) => {
            assert!(a_ok && !b_ok, "expected A ok, B failed");
        }
        other => panic!("expected a DecodeError divergence, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 4. Decoder inter-envelope map — bounds the chunk-2 target config
// ---------------------------------------------------------------------------

#[test]
fn chunk0_decoder_inter_envelope_report_bounds_chunk2() {
    aom_sys_ref::ref_init();
    println!(
        "\n=== port inter-decoder envelope on aomenc's simplest-config P (4:2:0 64x64, dx=3, cq60, cdef/lr off) ==="
    );
    println!("cpu | frame0 KEY | frame1 P (port-decode vs C)");
    // BYTE-EXACT GATE (was a pinned-divergence probe): the port inter decoder now
    // decodes aomenc's simplest-config P byte-exact at EVERY cpu level, luma AND
    // chroma. The former cpu-0/3/4/6 chroma (and cpu-1 luma) ± divergences were
    // the loop-filter grid treating inter blocks as intra (`build_lf_inputs`
    // hardcoded ref_frame[0]=INTRA_FRAME + is_inter=use_intrabc-only), so the
    // per-block deblock level/skip decision was wrong for inter blocks — fixed by
    // carrying `ref_frame[0]`/mode/is_inter into the LF grid (DecodedBlockKf::
    // inter_lf). Any future frame-1 divergence here is a regression, NOT a probe.
    for cpu in 0..=6 {
        let base = textured_base(&format!("tex_420_64_cpu{cpu}"), 64, 64, false, 60, cpu);
        let cell = MultiFrameEncodeCell::translational(&base, 3, 0);
        let (stream, n, div) = port_vs_c(&cell, false, false);
        assert_eq!(n, 2, "cpu{cpu}: expected 2 decoded frames");

        // Frame 0 (KEY) must always be byte-exact (regression control across the
        // sweep). Localize frame 0 alone.
        let c0 = aom_sys_ref::ref_decode_av1_stream_frame(&stream, 0, cell.w, cell.h);
        let pf = port_frames(&stream);
        let f0_div = first_frameset_divergence(
            &[FrameView::of_decode(&pf[0])],
            &[FrameView::of_ref_decoded(&c0)],
            SB64_PX,
        );
        assert_eq!(f0_div, None, "cpu{cpu}: frame 0 (KEY) diverged");

        let verdict = match &div {
            None => "byte-exact".to_string(),
            Some(d) => d.to_string(),
        };
        println!("{cpu:>3} | exact      | {verdict}");

        // Frame 1 (the P) must be byte-exact too — luma AND chroma — at every cpu.
        assert_eq!(
            div, None,
            "cpu{cpu}: frame 1 (P) diverged from the C decode — {verdict}"
        );
    }
    println!(
        "FINDING: the port INTER DECODER decodes aomenc's simplest-config P BYTE-EXACT at every \n\
         cpu 0..6 (luma AND chroma). The former chroma/luma ± divergences were the loop-filter grid \n\
         mislabelling inter blocks as intra; fixed by threading ref/mode/is_inter into the LF grid."
    );
}
