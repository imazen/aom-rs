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

use aom_bench::rd_close::{self, RdBands};
use aom_bench::{EncodeCell, ToggleKnobs};
use aom_dsp::entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader,
};
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::partition::get_partition_subsize;
use aom_dsp::entropy::rb::ReadBitBuffer;
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
            aom_dsp::entropy::leb128::uleb_decode(&stream[after_header..]).expect("leb128");
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
    let seq = aom_dsp::entropy::header::read_sequence_header_obu(&mut rb);
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
    let p = aom_dsp::entropy::header::read_uncompressed_header(&mut rb, &cfg);
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

    // Graduate the byte-exact cells to a HARD byte-identity assert (PARITY
    // Section A): these 4 screen cells + the real-content control are
    // byte-identical to real aomenc and MUST stay so — a regression to
    // merely-RD-close now FAILS loudly instead of passing on the RD band.
    // The two 128² cells (`ui_420_128_cq32`, `text_420_128_cq20`) are PINNED
    // as genuine palette-induced AB/4-way partition near-ties (see
    // `decode_diff_palette_close_cells` + PARITY.md C3 / the KB-P palette
    // near-tie note): they stay RD-close only.
    const BYTE_EXACT_CELLS: &[&str] = &[
        "text_mono_64_cq32",
        "text_420_64_cq12",
        "text_420_64_cq32",
        "text_420_64_cq63",
        "control_real64_cq32",
    ];
    for r in &results {
        if BYTE_EXACT_CELLS.contains(&r.label.as_str()) {
            assert!(
                r.bit_identical,
                "{}: expected BYTE-IDENTICAL (PARITY Section A) but diverged \
                 ({:+.2}% size, {:+.3} zensim) — a byte-exact palette cell regressed",
                r.label, r.size_delta_pct, r.zensim_drop
            );
        }
    }

    rd_close::assert_rd_close(&results, &RdBands::default());
}

// ---------------------------------------------------------------------------
// DECODE-BOTH localizer (diagnostic) for the two CLOSE 128² near-tie cells.
// Encodes each cell with real aomenc (--enable-palette=1) AND the port, decodes
// BOTH with the (bit-exact) port decoder, and finds the FIRST divergence:
// partition node → leaf mode/tx/PALETTE field → reconstruction pixel. Prints
// the exact (mi_row, mi_col) so the divergence can be root-caused C-faithfully.
// Reuses the same cell generators as the gate (no drift). Not a hard gate.
// ---------------------------------------------------------------------------

const SB_BSIZE: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4
const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const PARTITION_NAMES: [&str; 10] = [
    "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
];

fn replay_tree(
    tree: &[i8],
    cursor: &mut usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
    out: &mut Vec<(i32, i32, usize, i8)>,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    let p = tree[*cursor];
    out.push((mi_row, mi_col, bsize, p));
    *cursor += 1;
    if p as usize == 3 {
        // PARTITION_SPLIT — the only type that recurses.
        let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
        let subsize = get_partition_subsize(bsize, p as i32) as usize;
        for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
            replay_tree(
                tree,
                cursor,
                mi_row + dr,
                mi_col + dc,
                subsize,
                mi_rows,
                mi_cols,
                out,
            );
        }
    }
}

/// Returns `true` when the cell is byte-identical to real aomenc (a PINNED
/// near-tie has been closed → promote it), `false` while it still diverges.
fn localize_palette_cell(cell: &EncodeCell) -> bool {
    // Palette-OFF control: is the divergence purely palette-induced?
    let c_off = cell.c_encode_screen(false, false);
    let port_off = cell.port_encode_with(&c_off, &ToggleKnobs::default());
    let port_off_tu = rd_close::splice_frame_obu(&c_off, &port_off);
    eprintln!(
        "--- {} palette-OFF: c_off={}B port_off={}B bit_identical={}",
        cell.label,
        c_off.len(),
        port_off_tu.len(),
        port_off_tu == c_off
    );

    let c_tu = cell.c_encode_screen(true, false);
    let port_on = cell.port_encode_with(
        &c_tu,
        &ToggleKnobs {
            enable_palette: true,
            ..Default::default()
        },
    );
    let port_tu = rd_close::splice_frame_obu(&c_tu, &port_on);
    eprintln!(
        "\n=== {} ===  c_tu={}B  port_tu={}B  bit_identical={}",
        cell.label,
        c_tu.len(),
        port_tu.len(),
        port_tu == c_tu
    );
    if port_tu == c_tu {
        eprintln!("  already BYTE-EXACT — nothing to localize");
        return true;
    }

    let (t_real, _, _) = aom_decode::frame::decode_frame_obus_prefilter(&c_tu)
        .expect("decode of REAL aomenc bytes failed");
    let (t_ours, _, _) = aom_decode::frame::decode_frame_obus_prefilter(&port_tu)
        .expect("decode of PORT bytes failed");

    let mi_cols = (cell.w as i32 + 3) >> 2;
    let mi_rows = (cell.h as i32 + 3) >> 2;

    // ---- partition trees, all SB roots in raster order ----
    let mut real_seq = Vec::new();
    let mut ours_seq = Vec::new();
    let (mut cr, mut co) = (0usize, 0usize);
    let mut sr = 0;
    while sr < mi_rows {
        let mut sc = 0;
        while sc < mi_cols {
            replay_tree(
                &t_real.tree,
                &mut cr,
                sr,
                sc,
                SB_BSIZE,
                mi_rows,
                mi_cols,
                &mut real_seq,
            );
            replay_tree(
                &t_ours.tree,
                &mut co,
                sr,
                sc,
                SB_BSIZE,
                mi_rows,
                mi_cols,
                &mut ours_seq,
            );
            sc += SB_MI;
        }
        sr += SB_MI;
    }
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        assert_eq!(
            (r.0, r.1, r.2),
            (o.0, o.1, o.2),
            "positions must stay locked until the first partition divergence"
        );
        if r.3 != o.3 {
            eprintln!(
                ">>> FIRST PARTITION DIVERGENCE at (mi_row={}, mi_col={}, bsize={}): real=PARTITION_{} ({}) ours=PARTITION_{} ({})",
                r.0,
                r.1,
                r.2,
                PARTITION_NAMES[r.3 as usize],
                r.3,
                PARTITION_NAMES[o.3 as usize],
                o.3
            );
            return false;
        }
    }
    eprintln!(
        "partition trees agree (real_seq={} ours_seq={}); scanning leaves incl. palette",
        real_seq.len(),
        ours_seq.len()
    );

    // ---- leaf fields incl. palette size/colours, matched on (mi_row,mi_col) ----
    for rb in &t_real.blocks {
        if let Some(ob) = t_ours
            .blocks
            .iter()
            .find(|b| b.mi_row == rb.mi_row && b.mi_col == rb.mi_col)
        {
            let modes_differ = ob.bsize != rb.bsize
                || ob.partition != rb.partition
                || ob.info.y_mode != rb.info.y_mode
                || ob.info.angle_delta_y != rb.info.angle_delta_y
                || ob.info.use_filter_intra != rb.info.use_filter_intra
                || ob.tx_size != rb.tx_size
                || ob.info.uv_mode != rb.info.uv_mode
                || ob.info.cfl_alpha_idx != rb.info.cfl_alpha_idx
                || ob.info.cfl_joint_sign != rb.info.cfl_joint_sign;
            let palette_differs = ob.info.palette_size != rb.info.palette_size
                || ob.info.palette_colors != rb.info.palette_colors;
            let txbs_differ = ob.txbs != rb.txbs || ob.txbs_uv != rb.txbs_uv;
            if modes_differ || palette_differs || txbs_differ {
                eprintln!(
                    ">>> FIRST LEAF MISMATCH at (mi_row={}, mi_col={}) [modes={modes_differ} palette={palette_differs} txbs={txbs_differ}]:\n    \
                     real bsize={} part={} y={} adly={} uv={} cfl=({},{}) fi={} tx={} pal={:?} txbs={:?} txbs_uv={:?}\n    \
                     ours bsize={} part={} y={} adly={} uv={} cfl=({},{}) fi={} tx={} pal={:?} txbs={:?} txbs_uv={:?}",
                    rb.mi_row,
                    rb.mi_col,
                    rb.bsize,
                    rb.partition,
                    rb.info.y_mode,
                    rb.info.angle_delta_y,
                    rb.info.uv_mode,
                    rb.info.cfl_alpha_idx,
                    rb.info.cfl_joint_sign,
                    rb.info.use_filter_intra,
                    rb.tx_size,
                    rb.info.palette_size,
                    rb.txbs,
                    rb.txbs_uv,
                    ob.bsize,
                    ob.partition,
                    ob.info.y_mode,
                    ob.info.angle_delta_y,
                    ob.info.uv_mode,
                    ob.info.cfl_alpha_idx,
                    ob.info.cfl_joint_sign,
                    ob.info.use_filter_intra,
                    ob.tx_size,
                    ob.info.palette_size,
                    ob.txbs,
                    ob.txbs_uv,
                );
                if palette_differs {
                    eprintln!(
                        "    real pal colours Y={:?} U={:?} V={:?}",
                        &rb.info.palette_colors[0..8],
                        &rb.info.palette_colors[8..16],
                        &rb.info.palette_colors[16..24]
                    );
                    eprintln!(
                        "    ours pal colours Y={:?} U={:?} V={:?}",
                        &ob.info.palette_colors[0..8],
                        &ob.info.palette_colors[8..16],
                        &ob.info.palette_colors[16..24]
                    );
                }
                return false;
            }
        }
    }
    eprintln!("all shared leaves agree on modes/palette/txb — scanning reconstruction");

    // ---- reconstruction diff (luma then chroma) ----
    for (name, (rr, rstride, rw, rh), (orr, ostride, ow, oh)) in [
        (
            "luma",
            (&t_real.recon, t_real.stride, t_real.width, t_real.height),
            (&t_ours.recon, t_ours.stride, t_ours.width, t_ours.height),
        ),
        (
            "U",
            (
                &t_real.recon_u,
                t_real.stride_uv,
                t_real.width_uv,
                t_real.height_uv,
            ),
            (
                &t_ours.recon_u,
                t_ours.stride_uv,
                t_ours.width_uv,
                t_ours.height_uv,
            ),
        ),
        (
            "V",
            (
                &t_real.recon_v,
                t_real.stride_uv,
                t_real.width_uv,
                t_real.height_uv,
            ),
            (
                &t_ours.recon_v,
                t_ours.stride_uv,
                t_ours.width_uv,
                t_ours.height_uv,
            ),
        ),
    ] {
        for row in 0..rh.min(oh) {
            for col in 0..rw.min(ow) {
                let rv = rr.px(row * rstride + col);
                let ov = orr.px(row * ostride + col);
                if rv != ov {
                    eprintln!(
                        ">>> FIRST {name} RECON DIVERGENCE at (row={row}, col={col}): real={rv} ours={ov}"
                    );
                    return false;
                }
            }
        }
    }
    eprintln!(
        "reconstruction planes IDENTICAL — byte divergence is pure entropy coding (unexpected)"
    );
    false
}

/// PINNED near-tie guard + localizer for the two CLOSE 128² palette cells.
///
/// Both cells are byte-exact with palette OFF; palette ON tips a genuine
/// AB/4-way partition RD near-tie (the palette machinery — `av1_allow_palette`,
/// `av1_get_palette_bsize_ctx`/`_mode_ctx`, k-means, neighbour cache/ctx
/// stamping — is all verified C-faithful, and the 64² palette cells are
/// byte-exact, so this is NOT a palette-cost bug). Localized (decode-both):
///   - `ui_420_128_cq32`   : (mi 0,0)  BLOCK_32X32 real HORZ_B vs port HORZ_4
///   - `text_420_128_cq20` : (mi 8,20) BLOCK_16X16 real VERT   vs port VERT_A
/// Same class as the KB-10/KB-11 pinned near-ties; closing it needs a sibling-C
/// per-candidate partition-RD dump (the deferred next step). The test ASSERTS
/// the divergence is still present — if either cell becomes byte-exact, this
/// FAILS so it gets promoted into `BYTE_EXACT_CELLS` above.
#[test]
fn decode_diff_palette_close_cells() {
    c::ref_init();
    let ui_exact = localize_palette_cell(&screen_cell(
        "ui_420_128_cq32",
        128,
        128,
        false,
        32,
        ui_luma,
        ui_chroma,
    ));
    let text_exact = localize_palette_cell(&screen_cell(
        "text_420_128_cq20",
        128,
        128,
        false,
        20,
        text_luma,
        ui_chroma,
    ));
    assert!(
        !ui_exact && !text_exact,
        "a PINNED palette near-tie became byte-exact — promote it into \
         BYTE_EXACT_CELLS in palette_y_rd_close_gate: ui_exact={ui_exact} text_exact={text_exact}"
    );
}
