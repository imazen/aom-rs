//! Characterization tool: walk a raw low-overhead AV1 OBU stream and print
//! every frame header's inter-relevant fields (frame type, show/showable,
//! show_existing_frame, refresh_frame_flags, ref map indices, primary_ref,
//! order hints, CDF-update flags, MC filter/precision flags, global motion).
//!
//! Built for the animated-AVIF inter-decode envelope work: run it over the
//! per-track streams extracted by `tools/avif-extract` to inventory exactly
//! which inter tools a target corpus uses (INTER_DECODE_ENVELOPE.md).
//!
//! ```text
//! cargo run -p zenav1-aom-decode --example inspect_headers -- <stream.obu>
//! ```
//!
//! Probe-quality parse: the header fields printed here all precede the quant
//! params, so the coded-lossless / superres two-phase re-parse the real
//! decode driver performs is unnecessary for this inventory (tail fields
//! after `quant` may be inexact for a coded-lossless frame — the inventory
//! fields are exact either way).

use aom_dsp::entropy::header::{
    read_sequence_header_obu, read_uncompressed_header, FrameHeaderObu, FrameHeaderPrefix,
    FrameSizeHeader, SequenceHeaderObu, TileInfoHeader,
};
use aom_dsp::entropy::leb128::uleb_decode;
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::rb::ReadBitBuffer;

/// `av1_get_tile_limits` (tile_common.c) — copy of the decode driver's
/// private helper (probe use only).
fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    const MAX_TILE_WIDTH: i32 = 4096;
    const MAX_TILE_AREA: i32 = 4096 * 2304;
    const MAX_TILE_COLS: i32 = 64;
    const MAX_TILE_ROWS: i32 = 64;
    fn tile_log2(blk_size: i32, target: i32) -> i32 {
        let mut k = 0;
        while (blk_size << k) < target {
            k += 1;
        }
        k
    }
    let sb_cols = (mi_cols + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_rows = (mi_rows + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_size_log2 = mib_size_log2 as i32 + 2;
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

/// `set_mb_mi`: frame mi dims, 8-pixel aligned.
fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// Build the reader-context `FrameHeaderObu` exactly as the decode driver's
/// `parse_frame_header_ext` does (inter gate inputs included).
fn reader_cfg(seq: &SequenceHeaderObu) -> FrameHeaderObu {
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
    let mi_cols = mi_dim(s.max_frame_width);
    let mi_rows = mi_dim(s.max_frame_height);
    let mut cfg = FrameHeaderObu {
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
        num_planes: if c.monochrome { 1 } else { 3 },
        separate_uv_delta_q: c.separate_uv_delta_q,
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    cfg.might_allow_ref_frame_mvs = s.enable_ref_frame_mvs && s.enable_order_hint;
    cfg.might_allow_warped_motion = s.enable_warped_motion;
    cfg
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <stream.obu> [...]", args[0]);
        std::process::exit(1);
    }
    for path in &args[1..] {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        println!("==== {path} ({} bytes)", data.len());
        let mut pos = 0usize;
        let mut seq: Option<SequenceHeaderObu> = None;
        let mut frame_no = 0usize;
        while pos < data.len() {
            let h = match read_obu_header(&data[pos..]) {
                Some(h) => h,
                None => {
                    println!("  !! bad OBU header at {pos}");
                    break;
                }
            };
            assert!(h.obu_has_size_field, "OBU without size field at {pos}");
            let (size, size_len) = uleb_decode(&data[pos + h.header_len..]).expect("leb128");
            let body = pos + h.header_len + size_len;
            let end = body + size as usize;
            let payload = &data[body..end];
            match h.obu_type {
                2 => println!("  -- temporal delimiter"),
                1 => {
                    let mut rb = ReadBitBuffer::new(payload);
                    let sh = read_sequence_header_obu(&mut rb);
                    let s = &sh.seq_header;
                    let c = &sh.color_config;
                    println!(
                        "  SEQ: profile={} {}x{} bd={} mono={} ss={}{} order_hint={} (bits {}) \
                         ref_frame_mvs={} dual_filter={} jnt_comp={} warp={} superres={} \
                         cdef={} restoration={} sb128={} intra_edge={} filter_intra={} \
                         screen_tools={} force_int_mv={} film_grain={} reduced_still={}",
                        seq_profile(&sh),
                        s.max_frame_width,
                        s.max_frame_height,
                        c.bit_depth,
                        c.monochrome as u8,
                        c.subsampling_x,
                        c.subsampling_y,
                        s.enable_order_hint as u8,
                        s.order_hint_bits_minus_1 + 1,
                        s.enable_ref_frame_mvs as u8,
                        s.enable_dual_filter as u8,
                        s.enable_dist_wtd_comp as u8,
                        s.enable_warped_motion as u8,
                        s.enable_superres as u8,
                        s.enable_cdef as u8,
                        s.enable_restoration as u8,
                        s.sb_size_128 as u8,
                        s.enable_intra_edge_filter as u8,
                        s.enable_filter_intra as u8,
                        s.force_screen_content_tools,
                        s.force_integer_mv,
                        sh.film_grain_params_present as u8,
                        sh.reduced_still_picture_hdr as u8,
                    );
                    seq = Some(sh);
                }
                3 | 6 | 7 => {
                    let sh = seq.as_ref().expect("frame before sequence header");
                    let cfg = reader_cfg(sh);
                    let mut rb = ReadBitBuffer::new(payload);
                    let p = read_uncompressed_header(&mut rb, &cfg);
                    let kind = match h.obu_type {
                        3 => "FRAME_HDR",
                        6 => "FRAME",
                        _ => "REDUNDANT_FH",
                    };
                    if p.prefix.show_existing_frame {
                        println!(
                            "  [{frame_no}] {kind}: SHOW_EXISTING idx={}",
                            p.prefix.existing_fb_idx_to_show
                        );
                        frame_no += 1;
                        pos = end;
                        continue;
                    }
                    let ft = ["KEY", "INTER", "INTRA_ONLY", "SWITCH"]
                        [p.prefix.frame_type.clamp(0, 3) as usize];
                    println!(
                        "  [{frame_no}] {kind}: {ft} show={} showable={} err_res={} \
                         disable_cdf_update={} order_hint={} refresh=0x{:02x} primary_ref={}",
                        p.prefix.show_frame as u8,
                        p.prefix.showable_frame as u8,
                        p.prefix.error_resilient_mode as u8,
                        p.prefix.disable_cdf_update as u8,
                        p.prefix.order_hint,
                        p.prefix.refresh_frame_flags,
                        p.prefix.primary_ref_frame,
                    );
                    println!(
                        "        size={}x{} (upscaled {}x{}, superres_denom={}) qidx={} \
                         seg={} tiles={}x{} refresh_ctx_disabled={} txmode_select={} reduced_tx={}",
                        p.frame_size.superres_upscaled_width,
                        p.frame_size.superres_upscaled_height,
                        p.frame_size.superres_upscaled_width,
                        p.frame_size.superres_upscaled_height,
                        p.frame_size.scale_denominator,
                        p.quant.base_qindex,
                        p.segmentation.enabled as u8,
                        p.tile_info.cols,
                        p.tile_info.rows,
                        p.refresh_frame_context_disabled as u8,
                        p.tx_mode_select as u8,
                        p.reduced_tx_set_used as u8,
                    );
                    if p.prefix.frame_type == 1 || p.prefix.frame_type == 3 {
                        let gm: Vec<u8> =
                            p.global_motion.iter().map(|g| g.wmtype).collect();
                        println!(
                            "        INTER: ref_map={:?} hp_mv={} force_int_mv={} interp={} \
                             switchable_mm={} ref_frame_mvs={} ref_mode_select={} skip_mode=[{},{}] \
                             warp={} gm_types={:?}",
                            &p.inter_ref.ref_map_idx,
                            p.allow_high_precision_mv as u8,
                            p.cur_frame_force_integer_mv as u8,
                            p.interp_filter,
                            p.switchable_motion_mode as u8,
                            p.allow_ref_frame_mvs as u8,
                            p.reference_mode_select as u8,
                            p.skip_mode_allowed as u8,
                            p.skip_mode_flag as u8,
                            p.allow_warped_motion as u8,
                            gm,
                        );
                    }
                    frame_no += 1;
                }
                4 => println!("  -- tile group ({size} B)"),
                5 => println!("  -- metadata ({size} B)"),
                15 => println!("  -- padding ({size} B)"),
                t => println!("  -- OBU type {t} ({size} B)"),
            }
            pos = end;
        }
    }
}

fn seq_profile(sh: &SequenceHeaderObu) -> i32 {
    sh.profile
}
