//! INTER-ENCODE chunk 2, sub-step 2a GATE — the low-delay P frame-header
//! DERIVATION (`aom_encode::inter_frame::derive_lowdelay_p_frame_header`) is
//! byte-exact vs a real `aomenc` frame-1 header.
//!
//! The KEY path bootstraps its header by re-parsing the reference stream; the P
//! path must DERIVE the values from the sequence header + the §3 config. This
//! gate proves the derivation reproduces the real frame-1 header byte-for-byte:
//! it encodes a real 2-frame `[KEY, P]` clip with `aomenc`, parses frame-1's
//! uncompressed header, derives a `FrameHeaderObu` from the §3 config (qindex
//! from `base_qindex_lowdelay_p_from_cq`, the ref bookkeeping constants, the
//! parsed recon-DEPENDENT loop-filter/CDEF — those are sub-step 2f's job), and
//! asserts:
//!   1. the individual DERIVED (recon-independent) field values equal the parse;
//!   2. `write_frame_header_obu(derived)` == `write_frame_header_obu(parsed)`
//!      byte-for-byte (the whole-header serialization proof 2g rides on).
//! Swept over cq {20,40,60,63} x {64x64, 128x128} x {mono, 4:2:0}.

use aom_bench::{EncodeCell, MultiFrameEncodeCell};
use aom_encode::inter_frame::{
    LowDelayPHeaderParams, PRIMARY_REF_NONE, TWO_FRAME_P_REF_MAP_IDX, TWO_FRAME_P_REFRESH_FLAGS,
    derive_lowdelay_p_frame_header,
};
use aom_encode::rc::base_qindex_lowdelay_p_from_cq;
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
    write_frame_header_obu,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::rb::ReadBitBuffer;
use aom_entropy::wb::WriteBitBuffer;

const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

fn base(label: &str, w: usize, h: usize, mono: bool, cq: i32) -> EncodeCell {
    let content = |r: usize, c: usize| -> u16 { (40 + ((r * 3 + c * 5) % 160)) as u16 };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = content(r, c);
        }
    }
    let (cw, ch) = if mono { (0, 0) } else { ((w + 1) >> 1, (h + 1) >> 1) };
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
        ss_x: 1,
        ss_y: 1,
        usage: 0,
        cq_level: cq,
        speed: 0,
        bd: 8,
        y,
        u,
        v,
    }
}

fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}
fn tile_log2(blk: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}
fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    let sb_cols = (mi_cols + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_rows = (mi_rows + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_size_log2 = mib_size_log2 as i32 + 2;
    let max_width_sb = 4096 >> sb_size_log2;
    let max_tile_area_sb = (4096 * 2304) >> (2 * sb_size_log2);
    let min_log2_cols = tile_log2(max_width_sb, sb_cols);
    let min_log2_tiles = tile_log2(max_tile_area_sb, sb_cols * sb_rows).max(min_log2_cols);
    TileInfoHeader {
        mi_cols,
        mi_rows,
        mib_size_log2,
        min_log2_cols,
        max_log2_cols: tile_log2(1, sb_cols.min(64)),
        min_log2_rows: (min_log2_tiles - min_log2_cols).max(0),
        max_log2_rows: tile_log2(1, sb_rows.min(64)),
        max_width_sb,
        max_height_sb: (max_tile_area_sb / max_width_sb.max(1)).max(1),
        ..Default::default()
    }
}

fn walk(bytes: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("obu");
        let after = pos + hdr.header_len;
        let (size, sb) = aom_entropy::leb128::uleb_decode(&bytes[after..]).expect("leb");
        let ps = after + sb;
        let pe = ps + size as usize;
        out.push((hdr.obu_type, bytes[ps..pe].to_vec()));
        pos = pe;
    }
    out
}

/// Build the sequence-derived `FrameHeaderObu` template (the SAME shape the KEY
/// path builds before `read_uncompressed_header`) + parse frame-1's real header,
/// returning also frame-1's raw OBU payload and the header BIT length the parse
/// consumed (so a caller can slice the byte-aligned real header off the stream —
/// the ground-truth bytes, avoiding the lossy reader->writer round-trip).
fn seq_template_and_real_p_header(
    stream: &[u8],
) -> (FrameHeaderObu, FrameHeaderObu, Vec<u8>, usize) {
    let obus = walk(stream);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == 1)
        .map(|(_, p)| p.clone())
        .expect("seq");
    let mut seq_rb = ReadBitBuffer::new(&seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let num_planes = if c.monochrome { 1 } else { 3 };
    let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
    let mi_cols = mi_dim(s.max_frame_width);
    let mi_rows = mi_dim(s.max_frame_height);

    let mut cfg = FrameHeaderObu {
        prefix: FrameHeaderPrefix {
            reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
            decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
            equal_picture_interval: seq.timing_info.equal_picture_interval,
            frame_presentation_time_length: seq
                .decoder_model_info
                .frame_presentation_time_length as u32,
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
        separate_uv_delta_q: c.separate_uv_delta_q,
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
            subsampling_x: c.subsampling_x,
            subsampling_y: c.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    cfg.might_allow_ref_frame_mvs = s.enable_ref_frame_mvs && s.enable_order_hint;
    cfg.might_allow_warped_motion = s.enable_warped_motion;

    // Parse frame 1 (the 2nd frame OBU) into a real header.
    let frames: Vec<&(u32, Vec<u8>)> = obus.iter().filter(|(t, _)| *t == 6 || *t == 3).collect();
    assert_eq!(frames.len(), 2, "expected [KEY, P] frame OBUs");
    let f1_payload = frames[1].1.clone();
    let mut rb = ReadBitBuffer::new(&f1_payload);
    let real = read_uncompressed_header(&mut rb, &cfg);
    let header_bits = rb.bit_position();
    assert_eq!(real.prefix.frame_type, 1, "frame 1 must be INTER");
    (cfg, real, f1_payload, header_bits)
}

fn header_bytes(h: &FrameHeaderObu) -> Vec<u8> {
    let mut wb = WriteBitBuffer::new();
    write_frame_header_obu(&mut wb, h);
    wb.byte_align_zeros(); // frame_obu()'s byte_alignment() after frame_header_obu()
    wb.bytes().to_vec()
}

#[test]
fn derive_lowdelay_p_header_byte_exact_vs_aomenc() {
    aom_sys_ref::ref_init();
    let mut cells = 0;
    for cq in [20, 40, 60, 63] {
        for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
            for mono in [false, true] {
                let label = format!("cq{cq}_{w}x{h}_{}", if mono { "mono" } else { "420" });
                let cell = MultiFrameEncodeCell::translational(&base(&label, w, h, mono, cq), 0, 0);
                let stream = cell.c_encode_inter(false, false);
                let (seq_cfg, real, f1_payload, header_bits) =
                    seq_template_and_real_p_header(&stream);

                // DERIVE base_qindex from cq (sub-step 2b) — must match the parse.
                let derived_qindex = base_qindex_lowdelay_p_from_cq(cq);
                assert_eq!(
                    derived_qindex, real.quant.base_qindex,
                    "{label}: derived base_qindex {derived_qindex} != real {}",
                    real.quant.base_qindex
                );

                let params = LowDelayPHeaderParams {
                    base_qindex: derived_qindex,
                    order_hint: 1,
                    refresh_frame_flags: TWO_FRAME_P_REFRESH_FLAGS,
                    ref_map_idx: TWO_FRAME_P_REF_MAP_IDX,
                    disable_cdf_update: real.prefix.disable_cdf_update,
                    reduced_tx_set_used: real.reduced_tx_set_used,
                    // Recon/RD-DEPENDENT (sub-step 2f); sourced from the parse here.
                    // interp_filter is the per-frame filter RD; loopfilter/cdef
                    // need the P recon.
                    interp_filter: real.interp_filter,
                    loopfilter: real.loopfilter.clone(),
                    cdef: real.cdef.clone(),
                };
                let derived = derive_lowdelay_p_frame_header(&seq_cfg, &params);

                // 1. individual DERIVED (recon-independent) field values.
                assert_eq!(derived.prefix.frame_type, 1, "{label}: frame_type");
                assert_eq!(
                    derived.prefix.order_hint, real.prefix.order_hint,
                    "{label}: order_hint (derived {} vs real {})",
                    derived.prefix.order_hint, real.prefix.order_hint
                );
                assert_eq!(
                    derived.prefix.primary_ref_frame, PRIMARY_REF_NONE,
                    "{label}: primary_ref_frame"
                );
                assert_eq!(
                    derived.prefix.primary_ref_frame, real.prefix.primary_ref_frame,
                    "{label}: primary_ref_frame vs real"
                );
                assert_eq!(
                    derived.prefix.refresh_frame_flags, real.prefix.refresh_frame_flags,
                    "{label}: refresh_frame_flags (derived {:#x} vs real {:#x})",
                    derived.prefix.refresh_frame_flags, real.prefix.refresh_frame_flags
                );
                assert_eq!(
                    derived.inter_ref.ref_map_idx, real.inter_ref.ref_map_idx,
                    "{label}: ref_map_idx"
                );
                assert_eq!(
                    derived.inter_ref.frame_refs_short_signaling,
                    real.inter_ref.frame_refs_short_signaling,
                    "{label}: frame_refs_short_signaling"
                );
                assert_eq!(
                    derived.interp_filter, real.interp_filter,
                    "{label}: interp_filter (derived {} vs real {})",
                    derived.interp_filter, real.interp_filter
                );
                assert_eq!(
                    derived.allow_high_precision_mv, real.allow_high_precision_mv,
                    "{label}: allow_high_precision_mv"
                );
                assert_eq!(
                    derived.switchable_motion_mode, real.switchable_motion_mode,
                    "{label}: switchable_motion_mode"
                );
                assert_eq!(
                    derived.allow_ref_frame_mvs, real.allow_ref_frame_mvs,
                    "{label}: allow_ref_frame_mvs"
                );
                assert_eq!(
                    derived.tx_mode_select, real.tx_mode_select,
                    "{label}: tx_mode_select"
                );
                assert_eq!(
                    derived.reference_mode_select, real.reference_mode_select,
                    "{label}: reference_mode_select"
                );
                assert_eq!(
                    derived.reduced_tx_set_used, real.reduced_tx_set_used,
                    "{label}: reduced_tx_set_used"
                );
                assert_eq!(
                    derived.allow_warped_motion, real.allow_warped_motion,
                    "{label}: allow_warped_motion"
                );
                assert_eq!(
                    derived.prefix.disable_cdf_update, real.prefix.disable_cdf_update,
                    "{label}: disable_cdf_update"
                );

                // 2. whole-header byte-serialization proof vs the REAL STREAM
                // BYTES (not a reader->writer round-trip, which is lossy for
                // inter: the reader leaves inter_ref.enable_order_hint at its
                // default and the writer then drops frame_refs_short_signaling).
                // write_frame_header_obu(derived) + byte_align == the byte-aligned
                // real header prefix frames[1].payload[..header_byte_len].
                let db = header_bytes(&derived); // already byte-aligned
                let header_byte_len = header_bits.div_ceil(8);
                assert_eq!(
                    db.len(),
                    header_byte_len,
                    "{label}: derived header byte length {} != real {header_byte_len} (bits {header_bits})",
                    db.len()
                );
                let real_hdr = &f1_payload[..header_byte_len];
                assert_eq!(
                    db.as_slice(),
                    real_hdr,
                    "{label}: derived P frame-header bytes != real stream (derived {db:02x?} vs real {real_hdr:02x?})"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 16, "expected 16 swept cells");
    println!("2a: derived low-delay P header byte-exact vs aomenc on {cells} cells");
}
