//! Byte-match [`aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile`]
//! against the REAL tile-group bytes that follow the frame header in the SAME
//! `OBU_FRAME` payload real aomenc produces (`shim_encode_av1_kf`) -- Task 2's
//! deliverable: "the last piece for a genuine minimal end-to-end byte match"
//! per STATUS.md's own next-chunk note on `frame_header_matches_real_encoder
//! .rs`.
//!
//! Method: identical setup to `frame_header_matches_real_encoder.rs` (same
//! cases, same seq/frame-header parse), extended one step further: instead of
//! only comparing the frame-header's own bit prefix, this test extracts the
//! REAL raw tile bytes that follow it (`frame_payload[tile_data_start..]`,
//! `tile_data_start` = the byte position immediately after the frame header's
//! byte-aligned end) and feeds them BACK into
//! [`aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile`] along
//! with the REAL parsed frame header `p` -- proving the ASSEMBLY (byte-align +
//! `write_tile_group_header` + concatenation) reproduces the COMPLETE real
//! `frame_payload` byte-for-byte, given the same header values and the same
//! tile bytes. This is deliberately Assembly-verified, matching
//! `frame_header_matches_real_encoder.rs`'s own honesty framing: it proves the
//! WRAPPING is correct, not that this port's own search DERIVES those tile
//! bytes (that is the separate, harder Task 3 claim --
//! `encoder_gate_e2e_byte_match.rs`).
//!
//! `tiles_log2` is asserted 0 (single tile) for every case here -- the only
//! arm `assemble_frame_obu_payload_single_tile` implements; a future multi-
//! tile case must extend that function first, not this test.

use aom_entropy::header::{
    read_sequence_header_obu, read_uncompressed_header, CdefHeader, FrameHeaderObu,
    FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader, RestorationHeader, TileInfoHeader,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::rb::ReadBitBuffer;
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME_HEADER: u32 = 3;
const OBU_FRAME: u32 = 6;

/// Split a real AV1 byte stream into `(obu_type, payload)` pairs. Duplicated
/// from `frame_header_matches_real_encoder.rs` (itself duplicated from the
/// sequence-header test) -- small test-only helper, not worth a shared-crate
/// dependency for 15 lines (established convention in this test family).
fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        assert!(hdr.obu_has_size_field, "shim_encode_av1_kf always sets has_size_field");
        let (size, size_bytes) =
            aom_entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}

/// `tile_log2` (tile_common.c). Duplicated from
/// `frame_header_matches_real_encoder.rs` (post the 2026-07-14 `2*k` bug fix
/// documented there).
fn tile_log2(blk_size: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// `av1_get_tile_limits`. Duplicated from `frame_header_matches_real_encoder
/// .rs` (itself transcribed from the decoder-owned, private
/// `aom-decode/src/frame.rs::tile_limits`).
fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    const MAX_TILE_WIDTH: i32 = 4096;
    const MAX_TILE_AREA: i32 = 4096 * 2304;
    const MAX_TILE_COLS: i32 = 64;
    const MAX_TILE_ROWS: i32 = 64;
    let sb_cols = (mi_cols + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_rows = (mi_rows + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_size_log2 = mib_size_log2 as i32 + 2; // MI_SIZE_LOG2
    let max_width_sb = MAX_TILE_WIDTH >> sb_size_log2;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size_log2);
    let min_log2_cols = tile_log2(max_width_sb, sb_cols);
    let max_log2_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = tile_log2(max_tile_area_sb, sb_cols * sb_rows).max(min_log2_cols);
    TileInfoHeader {
        mi_cols,
        mi_rows,
        mib_size_log2,
        min_log2_cols,
        max_log2_cols,
        min_log2_rows: (min_log2_tiles - min_log2_cols).max(0),
        max_log2_rows,
        max_width_sb,
        max_height_sb: (max_tile_area_sb / max_width_sb.max(1)).max(1),
        ..Default::default()
    }
}

/// `set_mb_mi`. Duplicated from `frame_header_matches_real_encoder.rs`.
fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// `av1_setup_past_independence` KEY-frame defaults. Duplicated from
/// `frame_header_matches_real_encoder.rs`.
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

#[test]
fn tile_group_obu_assembly_matches_real_aomenc_output() {
    use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;

    // Identical case list to frame_header_matches_real_encoder.rs (this test
    // extends the SAME real streams one step further into the tile-group
    // bytes, so reusing the case list keeps the two tests directly
    // comparable). enable_cdef=false, enable_restoration=false: the task's
    // "no-post-filter envelope".
    let cases: &[(usize, usize, bool, usize, usize, u32, i32)] = &[
        (64, 64, false, 1, 1, 2, 32),  // 420, ALLINTRA, single 64x64 SB
        (64, 64, true, 1, 1, 2, 32),   // mono, ALLINTRA
        (64, 64, false, 1, 1, 0, 32),  // 420, GOOD
        (128, 64, false, 1, 1, 2, 40), // 2 SBs wide, ALLINTRA
    ];

    for &(w, h, mono, ss_x, ss_y, usage, cq_level) in cases {
        let y = vec![128u16; w * h];
        let (cw, ch) = if mono { (0, 0) } else { ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y) };
        let u = vec![128u16; cw * ch];
        let v = vec![128u16; cw * ch];

        let bytes = c::ref_encode_av1_kf(
            &y, &u, &v, w, h, 8, mono, ss_x as i32, ss_y as i32, cq_level, 0, false, false, usage,
            0, false,
        );
        assert!(!bytes.is_empty(), "shim_encode_av1_kf must produce a real stream");

        let obus = walk_obus(&bytes);
        let seq_payload = obus
            .iter()
            .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
            .map(|(_, p)| *p)
            .unwrap_or_else(|| panic!("no sequence-header OBU (w={w} h={h})"));
        let mut seq_rb = ReadBitBuffer::new(seq_payload);
        let seq = read_sequence_header_obu(&mut seq_rb);

        let (frame_obu_type, frame_payload) = obus
            .iter()
            .find(|(t, _)| *t == OBU_FRAME_HEADER || *t == OBU_FRAME)
            .map(|(t, p)| (*t, *p))
            .unwrap_or_else(|| panic!("no frame/frame-header OBU (w={w} h={h})"));
        assert_eq!(
            frame_obu_type, OBU_FRAME,
            "w={w} h={h}: real aomenc's default num_tg==1 config must combine frame header + \
             tile group into one OBU_FRAME (a standalone OBU_FRAME_HEADER would carry no tile \
             bytes to compare against)"
        );

        // Mirrors frame_header_matches_real_encoder.rs's `cfg` construction
        // (transcription, not a call -- see that file's own comment).
        let s = &seq.seq_header;
        let cc = &seq.color_config;
        let num_planes = if cc.monochrome { 1 } else { 3 };
        let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
        let mi_cols = mi_dim(s.max_frame_width);
        let mi_rows = mi_dim(s.max_frame_height);

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
            tile_info: tile_limits(mi_cols, mi_rows, mib_size_log2),
            num_planes,
            separate_uv_delta_q: cc.separate_uv_delta_q,
            loopfilter: LoopfilterHeader {
                last_ref_deltas: KF_REF_DELTAS,
                last_mode_deltas: KF_MODE_DELTAS,
                ..Default::default()
            },
            cdef: CdefHeader { enable_cdef: s.enable_cdef, ..Default::default() },
            restoration: RestorationHeader {
                enable_restoration: s.enable_restoration,
                sb_size_128: s.sb_size_128,
                subsampling_x: cc.subsampling_x,
                subsampling_y: cc.subsampling_y,
                ..Default::default()
            },
            film_grain_params_present: seq.film_grain_params_present,
            ..Default::default()
        };

        let mut rb = ReadBitBuffer::new(frame_payload);
        let p = read_uncompressed_header(&mut rb, &cfg);
        let real_bit_len = rb.bit_position();

        assert!(!p.prefix.show_existing_frame, "w={w} h={h}: show_existing_frame unexpected");
        assert_eq!(p.prefix.frame_type, 0, "w={w} h={h}: frame_type must be KEY");

        let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
        let ctx = format!(
            "w={w} h={h} mono={mono} ss=({ss_x},{ss_y}) usage={usage} cq={cq_level} \
             bit_len={real_bit_len} tiles=({}x{}, log2={tiles_log2}) frame_payload.len()={}",
            p.tile_info.cols,
            p.tile_info.rows,
            frame_payload.len(),
        );
        eprintln!("{ctx}");
        assert_eq!(
            tiles_log2, 0,
            "{ctx}: this test (and assemble_frame_obu_payload_single_tile) only implements the \
             single-tile envelope -- a case that genuinely needs >1 tile must extend the \
             assembly function first, not silently skip here"
        );

        // The REAL tile bytes: everything from the first byte-aligned
        // position after the frame header's content onward. byte_alignment()
        // zero-pads any partial trailing byte (verified below, not assumed).
        let tile_data_start = real_bit_len.div_ceil(8);
        let real_tile_bytes = &frame_payload[tile_data_start..];
        assert!(
            !real_tile_bytes.is_empty(),
            "{ctx}: a coded KEY frame must carry non-empty tile data"
        );

        // If the frame header did not end on a byte boundary, its trailing
        // partial byte's LOW bits must be zero (byte_alignment() padding,
        // per the AV1 spec's frame_obu() -- see obu_assemble.rs's module
        // docs) -- verify this explicitly rather than assuming it.
        let rem_bits = real_bit_len % 8;
        if rem_bits > 0 {
            let full_bytes = real_bit_len / 8;
            let low_mask = 0xFFu8 >> rem_bits;
            assert_eq!(
                frame_payload[full_bytes] & low_mask,
                0,
                "{ctx}: frame header's trailing partial byte's low {} bits must be \
                 byte_alignment() zero padding",
                8 - rem_bits
            );
        }

        // ---- the actual assembly-under-test ----
        let assembled = assemble_frame_obu_payload_single_tile(&p, tiles_log2, real_tile_bytes);
        assert_eq!(
            assembled, frame_payload,
            "{ctx}: assemble_frame_obu_payload_single_tile(real frame header, real tile bytes) \
             must reproduce the COMPLETE real aomenc OBU_FRAME payload byte-for-byte"
        );
    }
}
