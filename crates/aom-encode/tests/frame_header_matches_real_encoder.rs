//! Byte-match [`aom_entropy::header::write_frame_header_obu`] against the
//! REAL frame-header bits produced by `shim_encode_av1_kf` (real aomenc, the
//! same oracle [`seq_header_matches_real_encoder`] validates the
//! sequence-header OBU against) -- the next slice of the encoder gate's
//! step 3 ("byte-match the smallest frame vs shim_encode_av1_kf").
//!
//! Method: same shape as the sequence-header test. Encode a real minimal KEY
//! frame via `ref_encode_av1_kf` with `enable_cdef=false, enable_restoration
//! =false` (the task's "no-post-filter envelope" -- sidesteps needing the
//! unported CDEF-strength / loop-restoration searches; loop-filter level is
//! whatever the real encoder chose, since this test round-trips it rather
//! than deriving it -- see the module-level fraction note below), walk the
//! OBU stream for the sequence-header OBU (parsed first, since the frame
//! header's `cfg` template needs its fields -- mirrors
//! `aom-decode/src/frame.rs::parse_frame_header`'s own `cfg` construction,
//! REPLICATED here rather than called: that function plus its `tile_limits`/
//! `mi_dim` helpers are private to the decoder-owned `aom-decode` crate) and
//! the frame OBU (`OBU_FRAME_HEADER` standalone, or the head of a combined
//! `OBU_FRAME`), parse it with the ALREADY-VALIDATED
//! `read_uncompressed_header` (aom-entropy, decoder-owned; the same reader
//! `aom-decode`'s KEY-frame driver uses, gated on 336+ byte-identical real
//! streams per STATUS.md), then re-serialize with `write_frame_header_obu`
//! and assert the bits are identical to the real OBU's frame-header prefix.
//!
//! **Honest fraction**: this is ASSEMBLY-verified (option (a) in the task
//! brief), NOT derivation-verified. The parsed `FrameHeaderObu` (`p`) holds
//! real aomenc's OWN chosen field values (quant, loop-filter level/deltas,
//! tile info, ...) -- this test proves `write_frame_header_obu`'s
//! ordering/gating serializes those values back byte-for-byte, not that
//! this port can DERIVE them from an RDO search (loop-filter-level search,
//! CDEF-strength search, and real per-txs_ctx coeff-cost-driven mode/
//! partition decisions are all still separately-tracked gaps -- see
//! STATUS.md). Comparison is BIT-exact via `ReadBitBuffer::bit_position()`
//! (the frame header does not generally end on a byte boundary -- what
//! follows in the same payload, tile-group header or the next OBU, owns the
//! remaining bits of the last partial byte).

use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
    write_frame_header_obu, write_sequence_header_obu,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::rb::ReadBitBuffer;
use aom_entropy::wb::WriteBitBuffer;
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME_HEADER: u32 = 3;
const OBU_FRAME: u32 = 6;

/// Split a real AV1 byte stream into `(obu_type, payload)` pairs (OBU
/// header plus leb128 size framing only, no payload interpretation).
/// Duplicated from the sequence-header test since it is a small test-only
/// helper, not worth a shared-crate dependency for 15 lines.
fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        assert!(
            hdr.obu_has_size_field,
            "shim_encode_av1_kf always sets has_size_field"
        );
        let (size, size_bytes) =
            aom_entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}

/// `tile_log2` (tile_common.c): smallest k with `blk_size << k >= target`.
/// Transcribed from `aom-decode/src/frame.rs::tile_limits` (private to that
/// crate) -- see that function's own comment for the 2026-07-14 `2*k` bug
/// history; this copy is post-fix.
fn tile_log2(blk_size: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// `av1_get_tile_limits` (av1/encoder/bitstream.c / decodeframe.c
/// `av1_get_tile_limits` share the same math): transcribed verbatim from
/// `aom-decode/src/frame.rs::tile_limits` (private to that crate, decoder-
/// owned -- replicated here rather than made `pub` to avoid touching it).
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

/// `set_mb_mi` (av1/common/alloccommon.c): frame mi dims, 8-pixel aligned.
/// Transcribed from `aom-decode/src/frame.rs::mi_dim` (private to that
/// crate).
fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// `av1_set_default_ref_deltas` / `_mode_deltas` (KEY-frame
/// `av1_setup_past_independence` defaults) -- transcribed from
/// `aom-decode/src/frame.rs::KF_REF_DELTAS`/`KF_MODE_DELTAS`.
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

#[test]
fn frame_header_matches_real_aomenc_output() {
    c::ref_init();

    // usage=2 (ALLINTRA, the zenavif/avifenc primary path per the ALLINTRA
    // directive) first; usage=0 (GOOD) second. enable_cdef=false,
    // enable_restoration=false: the task's "no-post-filter envelope".
    let cases: &[(usize, usize, bool, usize, usize, u32, i32)] = &[
        (64, 64, false, 1, 1, 2, 32),  // 420, ALLINTRA, single 64x64 SB
        (64, 64, true, 1, 1, 2, 32),   // mono, ALLINTRA
        (64, 64, false, 1, 1, 0, 32),  // 420, GOOD
        (128, 64, false, 1, 1, 2, 40), // 2 SBs wide, ALLINTRA
    ];

    for &(w, h, mono, ss_x, ss_y, usage, cq_level) in cases {
        let y = vec![128u16; w * h];
        let (cw, ch) = if mono {
            (0, 0)
        } else {
            ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
        };
        let u = vec![128u16; cw * ch];
        let v = vec![128u16; cw * ch];

        let bytes = c::ref_encode_av1_kf(
            &y,
            &u,
            &v,
            w,
            h,
            8,
            mono,
            ss_x as i32,
            ss_y as i32,
            cq_level,
            0,
            false,
            false,
            usage,
            0,
            false,
        );
        assert!(
            !bytes.is_empty(),
            "shim_encode_av1_kf must produce a real stream"
        );

        let obus = walk_obus(&bytes);
        let seq_payload = obus
            .iter()
            .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
            .map(|(_, p)| *p)
            .unwrap_or_else(|| panic!("no sequence-header OBU (w={w} h={h})"));
        let mut seq_rb = ReadBitBuffer::new(seq_payload);
        let seq = read_sequence_header_obu(&mut seq_rb);

        // Cross-check our own seq-header writer reproduces the SAME real
        // bytes here too (already proven by seq_header_matches_real_encoder
        // .rs independently; re-asserting costs nothing and pins that the
        // `seq` we're about to build the frame cfg from is genuinely the
        // real encoder's own choice, not a misparse).
        let mut seq_wb = WriteBitBuffer::new();
        write_sequence_header_obu(&mut seq_wb, &seq);
        assert_eq!(
            seq_wb.bytes(),
            seq_payload,
            "w={w} h={h}: seq-header sanity re-check"
        );

        let (frame_obu_type, frame_payload) = obus
            .iter()
            .find(|(t, _)| *t == OBU_FRAME_HEADER || *t == OBU_FRAME)
            .map(|(t, p)| (*t, *p))
            .unwrap_or_else(|| panic!("no frame/frame-header OBU (w={w} h={h})"));

        // Mirrors aom-decode/src/frame.rs::parse_frame_header's `cfg`
        // construction (that function is private; this is a transcription,
        // not a call).
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
                frame_presentation_time_length: seq
                    .decoder_model_info
                    .frame_presentation_time_length
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
                buffer_removal_time_length: seq.decoder_model_info.buffer_removal_time_length
                    as u32,
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
            cdef: CdefHeader {
                enable_cdef: s.enable_cdef,
                ..Default::default()
            },
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

        assert!(
            !p.prefix.show_existing_frame,
            "w={w} h={h}: show_existing_frame unexpected"
        );
        assert_eq!(
            p.prefix.frame_type, 0,
            "w={w} h={h}: frame_type must be KEY"
        );

        let mut wb = WriteBitBuffer::new();
        write_frame_header_obu(&mut wb, &p);

        let full_bytes = real_bit_len / 8;
        let rem_bits = real_bit_len % 8;
        let ctx = format!(
            "w={w} h={h} mono={mono} ss=({ss_x},{ss_y}) usage={usage} cq={cq_level} \
             frame_obu_type={frame_obu_type} bit_len={real_bit_len}"
        );
        eprintln!(
            "{ctx}: frame_payload.len()={} qindex={} lf_level={:?} cdef_on={} \
             restoration_on={} tile_cols={} tile_rows={}",
            frame_payload.len(),
            p.quant.base_qindex,
            p.loopfilter.filter_level,
            p.cdef.enable_cdef,
            p.restoration.enable_restoration,
            p.tile_info.cols,
            p.tile_info.rows,
        );
        assert!(
            real_bit_len > 32,
            "{ctx}: suspiciously short frame header (near-empty parse?)"
        );
        assert_eq!(
            wb.bytes()[..full_bytes],
            frame_payload[..full_bytes],
            "{ctx}: write_frame_header_obu must reproduce the real aomenc frame-header bytes \
             (full-byte prefix)"
        );
        if rem_bits > 0 {
            let mask = 0xFFu8 << (8 - rem_bits);
            assert_eq!(
                wb.bytes()[full_bytes] & mask,
                frame_payload[full_bytes] & mask,
                "{ctx}: write_frame_header_obu must reproduce the real aomenc frame-header bits \
                 (final partial byte, high {rem_bits} bits)"
            );
        }
    }
}
