//! RD-closeness gate for the PALETTE-Y RD search (`av1_rd_pick_palette_intra_sby`)
//! — the first bulk-ported screen-content stills feature (PARITY.md section B).
//!
//! Cells encode few-colour screen content (text / UI / checker patterns that
//! trip the encoder's ANTIALIASING_AWARE screen-content auto-detection) with
//! `--enable-palette=1 --enable-intrabc=0` on BOTH sides:
//! - C: `ref_encode_av1_kf_screen_content(enable_palette=1, enable_intrabc=0)`
//! - port: `EncodeCell::port_encode_with(bootstrap, enable_palette=true)`
//!   (the palette search + palette pack path, everything else stock).
//!
//! Gate = `aom_bench::rd_close` bands (|Δsize| ≤ 5%, Δzensim ≤ 0.5;
//! byte-identical fast-paths as EXACT).
//!
//! Anti-vacuous witnesses (a palette gate that never codes a palette block
//! proves nothing):
//! 1. every screen cell asserts the real header signalled
//!    `allow_screen_content_tools` (detection actually fired);
//! 2. the C reference with palette ON must produce a DIFFERENT stream than
//!    with palette OFF on at least one screen cell (C used palette here);
//! 3. the PORT with palette ON must produce a DIFFERENT stream than the port
//!    with palette OFF on at least one screen cell (the port's search
//!    actually picked palette somewhere — not just tolerated the knob).
//!
//! A real-content (photographic, KB-6 recipe) cell rides along as a CONTROL:
//! screen detection does NOT fire there, palette is never searched by either
//! side, and the cell must stay in band (it is byte-exact today).

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_bench::rd_close::{self, RdBands};
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::rb::ReadBitBuffer;
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;

/// Parse `allow_screen_content_tools` out of a reference stream's headers
/// (sequence + uncompressed frame header), so the gate can assert the screen
/// detection actually fired on its cells.
fn stream_allow_screen_content(stream: &[u8]) -> bool {
    let mut pos = 0usize;
    let mut seq_payload: Option<&[u8]> = None;
    let mut frame_payload: Option<&[u8]> = None;
    while pos < stream.len() {
        let hdr = read_obu_header(&stream[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_entropy::leb128::uleb_decode(&stream[after_header..]).expect("leb128");
        let start = after_header + size_bytes;
        let end = start + size as usize;
        match hdr.obu_type {
            t if t == OBU_SEQUENCE_HEADER => seq_payload = Some(&stream[start..end]),
            t if t == OBU_FRAME => frame_payload = Some(&stream[start..end]),
            _ => {}
        }
        pos = end;
    }
    let seq_payload = seq_payload.expect("no sequence header OBU");
    let frame_payload = frame_payload.expect("no frame OBU");
    let mut rb = ReadBitBuffer::new(seq_payload);
    let seq = aom_entropy::header::read_sequence_header_obu(&mut rb);
    let s = &seq.seq_header;
    let cfg = FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time_length: seq.decoder_model_info.frame_presentation_time_length
                as u32,
            frame_id_numbers_present_flag: s.frame_id_numbers_present_flag,
            frame_id_length: s.frame_id_length as u32,
            force_screen_content_tools: s.force_screen_content_tools,
            force_integer_mv: s.force_integer_mv,
            max_frame_width: s.max_frame_width,
            max_frame_height: s.max_frame_height,
            enable_order_hint: s.enable_order_hint,
            order_hint_bits_minus_1: s.order_hint_bits_minus_1,
            operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
            operating_point_idc: seq.operating_point_idc,
            op_decoder_model_param_present: seq.op_decoder_model_param_present,
            buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length as u32,
            temporal_layer_id: 0,
            spatial_layer_id: 0,
            ..Default::default()
        },
        frame_size: FrameSizeHeader {
            num_bits_width: s.num_bits_width,
            num_bits_height: s.num_bits_height,
            superres_upscaled_width: s.max_frame_width,
            superres_upscaled_height: s.max_frame_height,
            enable_superres: s.enable_superres,
            ..Default::default()
        },
        num_planes: if seq.color_config.monochrome { 1 } else { 3 },
        separate_uv_delta_q: seq.color_config.separate_uv_delta_q,
        loopfilter: LoopfilterHeader::default(),
        cdef: CdefHeader {
            enable_cdef: s.enable_cdef,
            ..Default::default()
        },
        restoration: RestorationHeader {
            enable_restoration: s.enable_restoration,
            sb_size_128: s.sb_size_128,
            subsampling_x: seq.color_config.subsampling_x,
            subsampling_y: seq.color_config.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    let mut rb = ReadBitBuffer::new(frame_payload);
    let p = aom_entropy::header::read_uncompressed_header(&mut rb, &cfg);
    p.prefix.allow_screen_content_tools
}

/// Few-colour "terminal text" luma: 8-px glyph rows on a flat background,
/// blocky 2-colour glyphs + a third accent colour. Period-8 exact repeats
/// horizontally + large flat runs → screen-content detection fires; 3-4
/// distinct luma values per 64×64 block → the palette search has real wins.
fn text_luma(r: usize, c: usize) -> u16 {
    let row_in_line = r % 10;
    if row_in_line >= 7 {
        return 235; // inter-line background
    }
    let glyph = (c / 8 + (r / 10) * 5) % 4;
    let col_in_glyph = c % 8;
    match glyph {
        0 => {
            if col_in_glyph < 5 && row_in_line % 2 == 0 {
                32
            } else {
                235
            }
        }
        1 => {
            if col_in_glyph % 3 == 0 || row_in_line == 3 {
                32
            } else {
                235
            }
        }
        2 => {
            if col_in_glyph < 2 || col_in_glyph >= 6 {
                96
            } else {
                235
            }
        }
        _ => 235, // space
    }
}

/// Flat few-colour chroma "syntax highlighting" panels (period-16 vertical
/// bands over 3 chroma values — exact repeats, zero gradients).
fn ui_chroma(r: usize, c: usize) -> u16 {
    match (c / 16 + r / 24) % 3 {
        0 => 84,
        1 => 128,
        _ => 170,
    }
}

/// UI panels + 1-px borders luma: nested flat rectangles with sharp borders
/// (period-32 layout, 4 luma values).
fn ui_luma(r: usize, c: usize) -> u16 {
    let (pr, pc) = (r % 32, c % 32);
    if pr == 0 || pc == 0 {
        16 // grid border
    } else if pr < 6 {
        70 // title bar
    } else if (8..28).contains(&pc) && (10..26).contains(&pr) {
        200 // content well
    } else {
        140 // panel body
    }
}

fn screen_cell(
    label: &str,
    w: usize,
    h: usize,
    mono: bool,
    cq: i32,
    luma: impl Fn(usize, usize) -> u16,
    chroma: impl Fn(usize, usize) -> u16,
) -> EncodeCell {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = luma(r, c);
        }
    }
    let (cw, ch) = if mono { (0, 0) } else { (w / 2, h / 2) };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = chroma(r, c);
                v[r * cw + c] = chroma(r + 5, c + 7);
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
        usage: 2, // ALLINTRA
        cq_level: cq,
        speed: 0,
        bd: 8,
        y,
        u,
        v,
    }
}

#[test]
fn palette_y_rd_close_gate() {
    c::ref_init();
    let cells: Vec<EncodeCell> = vec![
        screen_cell("text_mono_64_cq32", 64, 64, true, 32, text_luma, |_, _| 0),
        screen_cell("text_420_64_cq12", 64, 64, false, 12, text_luma, ui_chroma),
        screen_cell("text_420_64_cq32", 64, 64, false, 32, text_luma, ui_chroma),
        screen_cell("text_420_64_cq63", 64, 64, false, 63, text_luma, ui_chroma),
        screen_cell("ui_420_128_cq32", 128, 128, false, 32, ui_luma, ui_chroma),
        screen_cell(
            "text_420_128_cq20",
            128,
            128,
            false,
            20,
            text_luma,
            ui_chroma,
        ),
    ];

    let mut results = Vec::new();
    let mut c_palette_active_somewhere = false;
    let mut port_palette_active_somewhere = false;

    for cell in &cells {
        // Both sides: --enable-palette=1 --enable-intrabc=0.
        let c_tu = cell.c_encode_screen(true, false);
        assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
        // Anti-vacuous witness 1: the screen-content detection must fire on
        // these cells or the palette gate exercises nothing.
        assert!(
            stream_allow_screen_content(&c_tu),
            "{}: screen-content detection did not fire — cell content needs rework",
            cell.label
        );

        // Anti-vacuous witnesses 2+3: palette-ON vs palette-OFF must change
        // the coded bytes somewhere (C side proves the reference uses
        // palette; port side proves OUR search picks it).
        let c_tu_off = cell.c_encode_screen(false, false);
        if c_tu != c_tu_off {
            c_palette_active_somewhere = true;
        }
        let port_on = cell.port_encode_with(
            &c_tu,
            &ToggleKnobs {
                enable_palette: true,
                ..Default::default()
            },
        );
        // The OFF-side port run bootstraps from the palette-OFF C stream (its
        // own same-knob reference).
        let port_off = cell.port_encode_with(&c_tu_off, &ToggleKnobs::default());
        if port_on != port_off {
            port_palette_active_somewhere = true;
        }

        let port_tu = rd_close::splice_frame_obu(&c_tu, &port_on);
        results.push(rd_close::compare_cell(&cell.label, cell, &port_tu, &c_tu));
    }

    // CONTROL: real photographic content (KB-6 recipe) — detection off,
    // palette never searched, must stay in band (byte-exact today).
    let control =
        EncodeCell::real_content("control_real64_cq32", "av1-1-b8-01-size-64x64", None, 32, 0);
    let control_tu = control.c_encode_screen(true, false);
    assert!(
        !stream_allow_screen_content(&control_tu),
        "control cell unexpectedly detected as screen content"
    );
    let control_port = control.port_encode_with(
        &control_tu,
        &ToggleKnobs {
            enable_palette: true,
            ..Default::default()
        },
    );
    let control_spliced = rd_close::splice_frame_obu(&control_tu, &control_port);
    results.push(rd_close::compare_cell(
        &control.label,
        &control,
        &control_spliced,
        &control_tu,
    ));

    assert!(
        c_palette_active_somewhere,
        "vacuous gate: real aomenc never coded a palette on any screen cell"
    );
    assert!(
        port_palette_active_somewhere,
        "vacuous gate: the port's palette search never changed the coded bytes \
         on any screen cell (knob dead?)"
    );

    rd_close::assert_rd_close(&results, &RdBands::default());
}
