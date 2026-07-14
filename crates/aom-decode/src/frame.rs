//! `decode_frame_obus` — decode a real AV1 KEY-frame bitstream (aomenc /
//! `aom_codec_av1_cx` output) to pixels: OBU walk (temporal delimiter /
//! sequence header / frame header / tile group, incl. the combined OBU_FRAME),
//! header parsing through the validated aom-entropy readers, default
//! FRAME_CONTEXT init by `base_qindex`, and the KEY-frame tile decode driver.
//!
//! ENVELOPE — the feature set [`decode_tile_kf`] / [`decode_frame_tiles_kf`]
//! models. Anything outside it is a hard [`Err`], never a mis-decode:
//! - KEY frame, shown, not show-existing; `error_resilient` accepted.
//! - 64x64 AND 128x128 superblocks (`use_128x128_superblock`): the
//!   sequence-header flag drives `mib_size_log2` (4 or 5), the partition-tree
//!   root bsize (`BLOCK_64X64` / `BLOCK_128X128`), the CDEF
//!   per-64x64-unit strength indexing, and the loop-restoration
//!   corners-in-sb SB extent. Gated by `sb128_streams_decode_byte_identical_to_c`.
//! - ANY tile grid (`TileInfoHeader::{cols,rows}`, uniform spacing — the only
//!   shape `AV1E_SET_TILE_COLUMNS`/`_ROWS` produces): each tile independently
//!   decoded ([`split_tiles`] + [`decode_frame_tiles_kf`] — per-tile context
//!   resets, tile-relative neighbour availability, a fresh `KfFrameContext`
//!   per tile) into one shared frame reconstruction. Gated by
//!   `multi_tile_streams_decode_byte_identical_to_c`. ONE tile GROUP per
//!   frame only (`--num-tile-groups=1`, the default — a real encoder splits
//!   into `OBU_TILE_GROUP`s only when explicitly configured otherwise);
//!   [`read_full_tile_group`] hard-errors on a partial tile-group range.
//!   Large-scale tile mode (`enable_large_scale_tile`) is not modelled —
//!   structurally unreachable here since nothing in this parse path signals
//!   or sets `large_scale`.
//! - PALETTE mode IS in the envelope (`allow_screen_content_tools` ON):
//!   per-block palette flags/sizes (`av1_allow_palette` — DC_PRED luma /
//!   UV_DC_PRED chroma, bsize 8x8..64x64), the neighbour colour-cache-aware
//!   colour coding (Y/U cache-bits + ascending delta, V raw/delta;
//!   [`aom_entropy::partition::read_palette_colors_plane`] /
//!   [`aom_entropy::partition::get_palette_cache`]), and the colour-index MAP
//!   (wavefront-order tokens on [`aom_entropy::partition::get_palette_color_index_context`],
//!   [`aom_entropy::partition::decode_color_map_tokens`]) — reconstruction
//!   bypasses ordinary intra prediction for a palette plane's tx blocks (the
//!   map indexes the palette directly; residual add is unaffected). Intra
//!   BLOCK COPY — the other screen-content tool `allow_screen_content_tools`
//!   gates — stays OUT of the envelope: `p.allow_intrabc` (only ever true
//!   when screen-content-tools is also on) is still a hard reject, so this is
//!   a clean palette-only envelope.
//! - film grain disabled at the sequence level; superres not scaled; no
//!   frame-size override (frame == sequence max dims).
//! - loop restoration IS applied ([`aom_restore::frame`], C-diffed against
//!   the real `av1_loop_restoration_filter_frame` + boundary-line saves in
//!   frame_walk_diff.rs): per-RU Wiener/SGR params decoded interleaved in
//!   the tile SB walk, deblocked-pre-CDEF stripe boundary context, both the
//!   boundary-swapped (with CDEF) and optimized (no CDEF) decoder arms.
//! - CDEF IS applied ([`aom_cdef::frame::cdef_frame`], C-diffed against the
//!   real `av1_cdef_frame` walk in cdef_frame_diff.rs): any damping /
//!   strength grids / per-64x64 strength indices, with the same gate as the
//!   C decoder (`cdef_bits || cdef_strengths[0] || cdef_uv_strengths[0]`,
//!   after deblocking).
//! - deblocking IS applied ([`aom_loopfilter::frame::loop_filter_frame`],
//!   C-diffed against the real `av1_filter_block_plane_vert/horz` walk) —
//!   any filter levels, sharpness, mode/ref deltas, and per-block delta-lf
//!   are in the envelope. ONE exception: 4:2:2 streams with nonzero CHROMA
//!   levels are rejected — libaom's chroma path reads
//!   `max_txsize_rect_lookup[BLOCK_INVALID]` out of bounds for tall blocks
//!   at `ss = (1,0)` (av1_ss_size_lookup, common_data.c:17), which is not
//!   portable behavior. 4:2:2 luma-only deblocking is in the envelope.
//! - segmentation IS in the envelope: per-block segment ids (spatial-pred
//!   symbols over the current-frame segment map), `SEG_LVL_ALT_Q` per-block
//!   dequant shifts (composing with delta-q), `SEG_LVL_SKIP` forced skips,
//!   and `SEG_LVL_ALT_LF_*` loop-filter level deltas (threaded into the
//!   deblock stage's per-segment level derivation). LOSSLESS SEGMENTS
//!   (`xd->lossless[i]` — effective qindex 0 with all plane deltas zero,
//!   which flips the block transform path to forced TX_4X4 + WHT) are
//!   rejected; the C encoder never emits them (av1_vaq_frame_setup clamps
//!   `base + delta` to >= 1).
//! - no quantization matrices, not (coded-)lossless.
//! - `disable_cdf_update` off (the driver always adapts).
//! - delta-q / delta-lf ARE in the envelope (per-block dequant recompute).
//!
//! The gold test (tests/real_bitstream.rs) compares the output planes
//! byte-identically against the REAL C decoder (`aom_codec_av1_dx`) on
//! bitstreams produced by the REAL encoder at `--cpu-used=0 --end-usage=q`.

use crate::{
    KfTileConfig, KfTileDecode, MI_SIZE_HIGH, MI_SIZE_WIDE, TileBoundsKf, TileBytesKf,
    decode_frame_tiles_kf,
};
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, SequenceHeaderObu, TileInfoHeader, read_sequence_header_obu,
    read_tile_group_header, read_uncompressed_header,
};
use aom_entropy::leb128::uleb_decode;
use aom_entropy::obu::read_obu_header;
use aom_entropy::partition::TxMode;
use aom_entropy::rb::ReadBitBuffer;
use aom_quant::av1_get_qindex;

/// aom-txb's `CDF_ARENA_LEN` (the coefficient region length
/// `KfFrameContext::default_for_qindex` fills).
const _: () = assert!(aom_txb::CDF_ARENA_LEN == 4045);

/// A decoded KEY frame: cropped planes + the header facts harnesses assert on.
#[derive(Clone, Debug)]
pub struct FrameDecode {
    /// Cropped luma, tight `width`-strided rows, u16 at every bit depth.
    pub y: Vec<u16>,
    /// Cropped chroma (empty when monochrome), tight `width_uv`-strided.
    pub u: Vec<u16>,
    pub v: Vec<u16>,
    pub width: usize,
    pub height: usize,
    pub width_uv: usize,
    pub height_uv: usize,
    pub bit_depth: i32,
    pub monochrome: bool,
    pub subsampling_x: usize,
    pub subsampling_y: usize,
    /// Frame quantizer facts (for harness assertions).
    pub base_qindex: i32,
    /// `[y0, y1, u, v]` loop-filter levels as coded — deblocking was applied
    /// with them (a gated no-op when both luma levels are 0, like C).
    pub filter_level: [i32; 4],
    /// CDEF params as coded — CDEF was applied with them when the C decoder
    /// gate (`cdef_bits || cdef_strengths[0] || cdef_uv_strengths[0]`) holds.
    pub cdef_damping: i32,
    pub cdef_bits: i32,
    pub cdef_strengths: [i32; 8],
    pub cdef_uv_strengths: [i32; 8],
    /// `features.tx_mode` was TX_MODE_SELECT (vs LARGEST).
    pub tx_mode_select: bool,
    pub reduced_tx_set: bool,
    pub delta_q_present: bool,
    /// Segmentation as coded: when enabled, per-block segment ids were
    /// decoded, `SEG_LVL_ALT_Q` shifted the per-block dequant, and
    /// `SEG_LVL_ALT_LF_*` deltas fed the deblock level derivation.
    /// `seg_last_active_segid` is `av1_calculate_segdata`'s highest segment
    /// with any active feature (the coded id alphabet bound).
    pub seg_enabled: bool,
    pub seg_last_active_segid: i32,
    /// Per-plane `frame_restoration_type` as coded (`RESTORE_*`); loop
    /// restoration was applied when any is non-NONE.
    pub lr_frame_restoration_type: [u8; 3],
    /// Restoration-unit populations actually decoded+applied across all
    /// planes: `(wiener, sgrproj, none)` counts (for harness floors).
    pub lr_unit_counts: (usize, usize, usize),
    /// `TileInfoHeader::{cols,rows}` as coded — the tile grid this frame was
    /// decoded with (independently-decoded tiles when either exceeds 1).
    pub tile_cols: usize,
    pub tile_rows: usize,
}

/// `av1_get_tile_limits` (av1/common/tile_common.c): the min/max tile log2
/// bounds the tile-info reader consumes (needed to correctly parse the
/// unary log2_cols/log2_rows increment bits for ANY tile count, not just the
/// single-tile envelope this was first written for — `min_log2_rows` is
/// `max(min_log2_tiles - log2_cols, 0)` in C with the CODED `log2_cols`, and
/// `log2_cols = min_log2_cols` is exact whenever the stream codes the
/// minimum, e.g. a 1x1 tiling always does).
fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    const MAX_TILE_WIDTH: i32 = 4096;
    const MAX_TILE_AREA: i32 = 4096 * 2304;
    const MAX_TILE_COLS: i32 = 64;
    const MAX_TILE_ROWS: i32 = 64;
    // `tile_log2` (tile_common.c): smallest k with `blk_size << k >= target`.
    // NOTE (2026-07-14): this previously read `blk_size << (2 * k)` — a
    // latent bug from the single-tile-only era, invisible because every
    // prior test image had <= 2 SBs per axis (`tile_log2(1, sb_cols)` only
    // diverges from the correct value once `sb_cols >= 4`, since the wrong
    // exponent `2*k` first differs from the right one `k` at k=1: `1<<2=4`
    // vs `1<<1=2`, both still `< 2` — or rather both still failing `< 2` —
    // so k=1 is reached identically for sb_cols<=2 either way). Multi-tile
    // streams on 4+ SB-wide/tall images (`multi_tile_streams_decode_
    // byte_identical_to_c`) exposed it: a too-small `max_log2_cols`/
    // `max_log2_rows` truncates the unary increment read loop early,
    // misaligning every bit read after it (`context_update_tile_id`,
    // `tile_size_bytes`, and the entire tile-group payload).
    fn tile_log2(blk_size: i32, target: i32) -> i32 {
        let mut k = 0;
        while (blk_size << k) < target {
            k += 1;
        }
        k
    }
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
fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// `read_tile_group_header`'s caller in `obu.c` (`read_one_tile_group_obu` /
/// the `is_obu_frame` inline in [`parse_frame_header`]): parse the
/// `tile_start_and_end_present_flag` + optional explicit `tiles_log2`-bit
/// start/end from `rb` (continuing wherever it is — mid-payload for a
/// combined OBU_FRAME, fresh for a standalone OBU_TILE_GROUP), inferring the
/// full tile range when the flag is absent (`num_tiles == 1` or the flag
/// reads 0), then byte-aligning — same as the real decoder's
/// `read_tile_group_header` + `byte_alignment` pair (obu.c). Hard-errors when
/// the group doesn't cover every tile: a frame whose tiles are split across
/// MORE THAN ONE tile-group OBU (`--num-tile-groups>1`) is out of envelope —
/// aomenc only emits that when explicitly configured; the default (and every
/// stream this driver's own oracle produces) is one tile group per frame.
fn read_full_tile_group(rb: &mut ReadBitBuffer, ti: &TileInfoHeader) -> Result<(), String> {
    let num_tiles = ti.cols as i32 * ti.rows as i32;
    let tiles_log2 = ti.log2_cols + ti.log2_rows;
    let (ts, te, present) = read_tile_group_header(rb, tiles_log2);
    let (start_tile, end_tile) = if present {
        (ts, te)
    } else {
        (0, num_tiles - 1)
    };
    if (start_tile, end_tile) != (0, num_tiles - 1) {
        return Err(format!(
            "partial tile group [{start_tile}..={end_tile}] of {num_tiles} tiles \
             (multiple tile groups per frame not supported)"
        ));
    }
    rb.byte_align();
    Ok(())
}

/// `mem_get_varsize` (`aom_ports/mem_ops.h`): an `n`-byte (1..=4) little-endian
/// unsigned read — the width `read_tile_info` parsed as `tile_size_bytes`.
fn mem_get_varsize(data: &[u8], n: usize) -> usize {
    let mut v = 0usize;
    for (i, &b) in data.iter().take(n).enumerate() {
        v |= (b as usize) << (8 * i);
    }
    v
}

/// `get_tile_buffers` / `get_tile_buffer` (decodeframe.c, the non-large-scale
/// path): split the tile-group payload into `ti.cols * ti.rows` per-tile
/// byte slices + their mi-space bounds (`av1_tile_set_row` / `_col`,
/// tile_common.c — `row_start_sb[row] << mib_size_log2`, clamped to the
/// frame's `mi_rows`/`mi_cols`, UNCONDITIONALLY — matches C's `AOMMIN` being
/// applied to every tile, not just the last), in raster (`tile_row`-major)
/// order. Every tile except the LAST is prefixed by a `tile_size_bytes`-byte
/// little-endian `tile_size_minus_1`; actual size = decoded value +
/// `AV1_MIN_TILE_SIZE_BYTES` (1). The last tile takes the remainder.
fn split_tiles<'a>(
    tile_data: &'a [u8],
    ti: &TileInfoHeader,
    tile_size_bytes: i32,
) -> Result<Vec<TileBytesKf<'a>>, String> {
    let num_tiles = ti.cols * ti.rows;
    let n = tile_size_bytes as usize;
    let mut out = Vec::with_capacity(num_tiles);
    let mut data = tile_data;
    let mut tc = 0usize;
    for row in 0..ti.rows {
        for col in 0..ti.cols {
            let is_last = tc == num_tiles - 1;
            let bounds = TileBoundsKf {
                mi_row_start: ti.row_start_sb[row] << ti.mib_size_log2,
                mi_row_end: (ti.row_start_sb[row + 1] << ti.mib_size_log2).min(ti.mi_rows),
                mi_col_start: ti.col_start_sb[col] << ti.mib_size_log2,
                mi_col_end: (ti.col_start_sb[col + 1] << ti.mib_size_log2).min(ti.mi_cols),
            };
            let size = if is_last {
                data.len()
            } else {
                if data.len() < n {
                    return Err("truncated tile-size prefix".into());
                }
                let sz = mem_get_varsize(data, n) + 1; // AV1_MIN_TILE_SIZE_BYTES
                data = &data[n..];
                sz
            };
            if data.len() < size {
                return Err("truncated tile payload".into());
            }
            let (bytes, rest) = data.split_at(size);
            out.push(TileBytesKf { bytes, bounds });
            data = rest;
            tc += 1;
        }
    }
    Ok(out)
}

/// Bridge the parsed segmentation frame header into the quantizer-layer
/// `cm->seg` shape ([`aom_quant::Segmentation`]) the block layer consumes.
/// Feature data is post-clamp (`|data| <= 255`), so the i16 narrowing is exact.
fn bridge_segmentation(h: &aom_entropy::header::SegmentationHeader) -> aom_quant::Segmentation {
    let mut feature_data = [[0i16; aom_quant::SEG_LVL_MAX]; aom_quant::MAX_SEGMENTS];
    for (dst, src) in feature_data.iter_mut().zip(h.feature_data.iter()) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = *s as i16;
        }
    }
    aom_quant::Segmentation {
        enabled: h.enabled,
        feature_mask: h.feature_mask,
        feature_data,
    }
}

/// KF `av1_setup_past_independence` loop-filter delta defaults
/// (`av1_set_default_ref_deltas` / `_mode_deltas`) — the "previous" deltas the
/// loop-filter reader diffs against on a keyframe.
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

struct ParsedFrame {
    header: FrameHeaderObu,
    /// Byte offset of the tile data within the SAME payload (OBU_FRAME), or
    /// `None` when the header stood alone (tile group arrives as its own OBU).
    tile_data_off: Option<usize>,
}

/// Parse a frame header (standalone FRAME_HEADER OBU or the head of an
/// OBU_FRAME payload) and run every envelope gate. The gates run in stream
/// order so an early out-of-envelope field can never be masked by a later
/// mis-parse (`allow_screen_content_tools` precedes the frame size in the
/// bitstream, so it is checked first).
fn parse_frame_header(
    seq: &SequenceHeaderObu,
    payload: &[u8],
    is_obu_frame: bool,
) -> Result<ParsedFrame, String> {
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let num_planes = if c.monochrome { 1 } else { 3 };
    // `set_sb_size` (av1_common_int.h): mib_size_log2 = mi_size_wide_log2[sb_size]
    // — 4 for BLOCK_64X64 (16 mi/side), 5 for BLOCK_128X128 (32 mi/side).
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
            num_bits_width: seq.seq_header.num_bits_width,
            num_bits_height: seq.seq_header.num_bits_height,
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
        // coded_lossless / all_lossless are inputs to this writer-mirror
        // reader but are stream facts on the decode side: parse as
        // non-lossless and hard-error afterwards if the quant params imply
        // lossless (the parse of a lossless stream is invalid).
        ..Default::default()
    };

    let mut rb = ReadBitBuffer::new(payload);
    let p = read_uncompressed_header(&mut rb, &cfg);

    // --- envelope gates, in bitstream order ---
    if p.prefix.show_existing_frame {
        return Err("show_existing_frame".into());
    }
    if p.prefix.frame_type != 0 {
        return Err(format!("frame_type {} (KEY only)", p.prefix.frame_type));
    }
    if !p.prefix.show_frame {
        return Err("unshown frame".into());
    }
    if p.prefix.disable_cdf_update {
        return Err("disable_cdf_update (driver always adapts)".into());
    }
    // `allow_screen_content_tools` (PALETTE mode) is in the envelope; intra block
    // copy — the OTHER screen-content tool it gates — stays rejected below via
    // `p.allow_intrabc` (the per-frame syntax flag; the spec only ever reads it
    // when `allow_screen_content_tools` holds — see `read_uncompressed_header`,
    // `p.allow_intrabc = p.prefix.allow_screen_content_tools && !cfg.superres_scaled
    // && rb.read_bit() != 0`). That single check is a clean palette-only envelope:
    // it can never be true unless screen-content-tools is also on, so accepting
    // palette streams here does not admit any intrabc stream.
    if p.frame_size.superres_upscaled_width != s.max_frame_width
        || p.frame_size.superres_upscaled_height != s.max_frame_height
    {
        return Err("frame_size_override (frame != sequence max dims)".into());
    }
    if p.frame_size.scale_denominator != 8 {
        return Err("superres scaled".into());
    }
    // Intra block copy: monochrome intrabc KEY frames are in the envelope
    // (luma is an integer block copy from the already-decoded region). Colour
    // intrabc stays rejected pending chroma reconstruction (the chroma DV is
    // luma-derived and can land at half-pel, needing the 2-tap intrabc
    // convolve + the co-located-luma chroma tx-type).
    if p.allow_intrabc && !c.monochrome {
        return Err("intrabc (colour; chroma reconstruction pending)".into());
    }
    if p.quant.using_qmatrix {
        return Err("quantization matrices".into());
    }
    let q = &p.quant;
    if q.base_qindex == 0
        && q.y_dc_delta_q == 0
        && q.u_dc_delta_q == 0
        && q.u_ac_delta_q == 0
        && q.v_dc_delta_q == 0
        && q.v_ac_delta_q == 0
    {
        // Also invalidates this parse (cfg.coded_lossless was false).
        return Err("(coded-)lossless stream".into());
    }
    if p.segmentation.enabled {
        // KEY frame, no primary ref: the parse forces map+data updates on and
        // temporal prediction off (setup_segmentation's PRIMARY_REF_NONE arm).
        debug_assert!(
            p.segmentation.update_map
                && p.segmentation.update_data
                && !p.segmentation.temporal_update
        );
        // xd->lossless[i] (decodeframe.c:5166): a lossless SEGMENT switches
        // the block transform path (forced TX_4X4 + WHT) — out of envelope.
        // Only ids 0..=last_active_segid are decodable (read_segment_id
        // bounds the alphabet), so only those rows gate.
        let seg = bridge_segmentation(&p.segmentation);
        let (_, last_active) = crate::calculate_segdata(&seg);
        for i in 0..=last_active as usize {
            if av1_get_qindex(&seg, i, q.base_qindex) == 0
                && q.y_dc_delta_q == 0
                && q.u_dc_delta_q == 0
                && q.u_ac_delta_q == 0
                && q.v_dc_delta_q == 0
                && q.v_ac_delta_q == 0
            {
                return Err(format!("lossless segment {i} (forced-WHT path)"));
            }
        }
    }
    let lf = &p.loopfilter;
    if c.subsampling_x == 1
        && c.subsampling_y == 0
        && (lf.filter_level_u != 0 || lf.filter_level_v != 0)
    {
        // libaom's 4:2:2 chroma deblock path indexes
        // max_txsize_rect_lookup[BLOCK_INVALID] out of bounds for tall
        // blocks; not portable — see aom-loopfilter/tests/lf_apply_diff.rs.
        return Err(format!(
            "4:2:2 chroma deblocking (levels u={} v={}) out of envelope",
            lf.filter_level_u, lf.filter_level_v
        ));
    }

    let tile_data_off = if is_obu_frame {
        rb.byte_align();
        read_full_tile_group(&mut rb, &p.tile_info)?;
        Some(rb.bytes_read())
    } else {
        None
    };
    Ok(ParsedFrame {
        header: p,
        tile_data_off,
    })
}

/// Decode a full AV1 KEY-frame bitstream (a temporal unit as emitted by
/// aomenc / `aom_codec_av1_cx`: temporal delimiter + sequence header + frame)
/// to cropped planes. Hard-errors on anything outside the documented envelope.
pub fn decode_frame_obus(data: &[u8]) -> Result<FrameDecode, String> {
    let (mut t, cfg, header) = decode_frame_obus_prefilter(data)?;
    if header.loopfilter.filter_level != [0, 0] {
        apply_deblock(&mut t, &cfg, &header);
    }
    // The C decoder's do_cdef gate (decodeframe.c:5417): !skip_loop_filter
    // (a decoder option, always off here) && !coded_lossless (rejected
    // upstream) && any CDEF syntax present. allow_intrabc (which would force
    // cdef_bits == 0) and multi-tile large-scale decoding are rejected too.
    let cd = &header.cdef;
    let do_cdef = cd.cdef_bits != 0 || cd.cdef_strengths[0] != 0 || cd.cdef_uv_strengths[0] != 0;
    let do_lr = cfg.lr.any_enabled();
    // decodeframe.c:5423: optimized_loop_restoration = !do_cdef &&
    // !do_superres (superres is rejected upstream, so always unscaled). The
    // non-optimized arm saves the DEBLOCKED rows (pre-CDEF) as internal
    // stripe boundary context before CDEF runs, and the CDEF output rows as
    // frame-edge context after; the optimized arm saves nothing (the frame's
    // own rows are the context).
    let optimized_lr = !do_cdef;
    let pre_cdef =
        (do_lr && !optimized_lr).then(|| (t.recon.clone(), t.recon_u.clone(), t.recon_v.clone()));
    if do_cdef {
        apply_cdef(&mut t, &cfg, &header);
    }
    if do_lr {
        apply_restoration(&mut t, &cfg, pre_cdef.as_ref(), optimized_lr);
    }
    Ok(finish_frame(t, &cfg, &header))
}

/// Run [`aom_restore::frame::loop_restoration_filter_frame`] over the
/// (mi-aligned, deblocked+CDEF'd) recon planes, exactly as the C decoder does
/// after CDEF (decodeframe.c:5437-5482). `pre_cdef` is the deblocked
/// pre-CDEF snapshot feeding internal stripe boundaries (`None` on the
/// optimized no-CDEF path, which reads no boundary context). Hidden: harness
/// entry so tests can recompose the filter pipeline stage by stage.
#[doc(hidden)]
pub fn apply_restoration(
    t: &mut KfTileDecode,
    cfg: &KfTileConfig,
    pre_cdef: Option<&(Vec<u16>, Vec<u16>, Vec<u16>)>,
    optimized_lr: bool,
) {
    use aom_restore::frame::{LrPlaneInput, loop_restoration_filter_frame};
    let empty: (Vec<u16>, Vec<u16>, Vec<u16>) = (Vec::new(), Vec::new(), Vec::new());
    let (dy, du, dv) = pre_cdef.unwrap_or(&empty);
    let mut planes = Vec::new();
    planes.push(LrPlaneInput {
        cur: &mut t.recon,
        deblocked: dy,
        stride: t.stride,
        units: &t.lr_units[0],
    });
    if !cfg.monochrome {
        planes.push(LrPlaneInput {
            cur: &mut t.recon_u,
            deblocked: du,
            stride: t.stride_uv,
            units: &t.lr_units[1],
        });
        planes.push(LrPlaneInput {
            cur: &mut t.recon_v,
            deblocked: dv,
            stride: t.stride_uv,
            units: &t.lr_units[2],
        });
    }
    loop_restoration_filter_frame(
        &mut planes,
        &cfg.lr,
        cfg.subsampling_x,
        cfg.subsampling_y,
        cfg.bd,
        optimized_lr,
    );
}

/// Everything [`decode_frame_obus`] does up to (but not including) the loop
/// filter: OBU walk, header parse + envelope gates, tile decode. Returns the
/// mi-aligned pre-filter reconstruction + the tile config + the parsed frame
/// header. Hidden: harness entry so differential tests can drive the C
/// reference filter over the exact same pre-filter state.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn decode_frame_obus_prefilter(
    data: &[u8],
) -> Result<(KfTileDecode, KfTileConfig, FrameHeaderObu), String> {
    let mut pos = 0usize;
    let mut seq: Option<SequenceHeaderObu> = None;
    let mut pending_header: Option<FrameHeaderObu> = None;
    let mut decoded: Option<(KfTileDecode, KfTileConfig, FrameHeaderObu)> = None;

    while pos < data.len() {
        let h = read_obu_header(&data[pos..]).ok_or("bad OBU header")?;
        if !h.obu_has_size_field {
            return Err("OBU without size field".into());
        }
        let (size, size_len) =
            uleb_decode(&data[pos + h.header_len..]).ok_or("bad OBU size leb128")?;
        let body = pos + h.header_len + size_len;
        let end = body + size as usize;
        if end > data.len() {
            return Err("OBU size past end of data".into());
        }
        let payload = &data[body..end];

        match h.obu_type {
            2 => {} // OBU_TEMPORAL_DELIMITER (empty)
            1 => {
                // OBU_SEQUENCE_HEADER
                let mut rb = ReadBitBuffer::new(payload);
                let sh = read_sequence_header_obu(&mut rb);
                let s = &sh.seq_header;
                if sh.film_grain_params_present {
                    return Err("film grain enabled (sequence)".into());
                }
                if s.force_screen_content_tools == 1 {
                    return Err("screen content tools forced on (sequence)".into());
                }
                let c = &sh.color_config;
                if !c.monochrome {
                    let ss = (c.subsampling_x, c.subsampling_y);
                    if !matches!(ss, (0, 0) | (1, 0) | (1, 1)) {
                        return Err(format!("unsupported subsampling {ss:?}"));
                    }
                }
                seq = Some(sh);
            }
            3 => {
                // OBU_FRAME_HEADER
                let sh = seq.as_ref().ok_or("frame header before sequence header")?;
                let pf = parse_frame_header(sh, payload, false)?;
                pending_header = Some(pf.header);
            }
            4 | 6 => {
                // OBU_TILE_GROUP | OBU_FRAME
                if decoded.is_some() {
                    return Err("second frame in stream (single KEY frame only)".into());
                }
                let sh = seq.as_ref().ok_or("frame before sequence header")?;
                let (header, tile_data) = if h.obu_type == 6 {
                    let pf = parse_frame_header(sh, payload, true)?;
                    let off = pf.tile_data_off.unwrap();
                    (pf.header, &payload[off..])
                } else {
                    let header = pending_header
                        .take()
                        .ok_or("tile group without frame header")?;
                    // A standalone OBU_TILE_GROUP payload starts fresh (no
                    // preceding uncompressed-header bytes to continue from).
                    let mut rb = ReadBitBuffer::new(payload);
                    read_full_tile_group(&mut rb, &header.tile_info)?;
                    let off = rb.bytes_read();
                    (header, &payload[off..])
                };
                decoded = Some(decode_tile_payload(sh, &header, tile_data)?);
            }
            5 | 15 => {} // OBU_METADATA | OBU_PADDING — content-neutral
            t => return Err(format!("unsupported OBU type {t}")),
        }
        pos = end;
    }

    decoded.ok_or_else(|| "no frame in stream".into())
}

/// Run the tile decoder over the (single) tile payload — the pre-filter
/// stage: builds the tile config and decodes; no loop filter, no crop.
#[allow(clippy::type_complexity)]
fn decode_tile_payload(
    seq: &SequenceHeaderObu,
    p: &FrameHeaderObu,
    tile_data: &[u8],
) -> Result<(KfTileDecode, KfTileConfig, FrameHeaderObu), String> {
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let (ss_x, ss_y) = if c.monochrome {
        (1usize, 1usize)
    } else {
        (c.subsampling_x as usize, c.subsampling_y as usize)
    };

    let cfg = KfTileConfig {
        mi_rows: mi_dim(s.max_frame_height),
        mi_cols: mi_dim(s.max_frame_width),
        bd: c.bit_depth,
        monochrome: c.monochrome,
        subsampling_x: ss_x,
        subsampling_y: ss_y,
        cdef_bits: p.cdef.cdef_bits as u32,
        disable_edge_filter: !s.enable_intra_edge_filter,
        enable_filter_intra: s.enable_filter_intra,
        tx_mode: if p.tx_mode_select {
            TxMode::Select
        } else {
            TxMode::Largest
        },
        reduced_tx_set: p.reduced_tx_set_used,
        base_qindex: p.quant.base_qindex,
        y_dc_delta_q: p.quant.y_dc_delta_q,
        u_dc_delta_q: p.quant.u_dc_delta_q,
        u_ac_delta_q: p.quant.u_ac_delta_q,
        v_dc_delta_q: p.quant.v_dc_delta_q,
        v_ac_delta_q: p.quant.v_ac_delta_q,
        delta_q_present: p.delta_q.delta_q_present,
        delta_q_res: p.delta_q.delta_q_res,
        delta_lf_present: p.delta_q.delta_lf_present,
        delta_lf_multi: p.delta_q.delta_lf_multi,
        delta_lf_res: p.delta_q.delta_lf_res,
        lr: aom_entropy::lr::LrFrameConfig {
            frame_restoration_type: p.restoration.frame_restoration_type,
            unit_size: p.restoration.restoration_unit_size,
            crop_width: p.frame_size.superres_upscaled_width,
            crop_height: p.frame_size.superres_upscaled_height,
        },
        seg: bridge_segmentation(&p.segmentation),
        sb_size_128: s.sb_size_128,
        allow_screen_content_tools: p.prefix.allow_screen_content_tools,
        allow_intrabc: p.allow_intrabc,
    };
    let tiles = split_tiles(tile_data, &p.tile_info, p.tile_size_bytes)?;
    let t = decode_frame_tiles_kf(&tiles, &cfg, 0);
    Ok((t, cfg, p.clone()))
}

/// Crop the (post-filter) mi-aligned recon to the frame dims and assemble the
/// output facts. The deblocking gate ran in [`decode_frame_obus`].
fn finish_frame(t: KfTileDecode, cfg: &KfTileConfig, p: &FrameHeaderObu) -> FrameDecode {
    // The coded frame (crop) dims — superres is unscaled and frame-size
    // override rejected in this envelope, so the upscaled size IS the size.
    let width = p.frame_size.superres_upscaled_width as usize;
    let height = p.frame_size.superres_upscaled_height as usize;
    let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);

    let mut y = vec![0u16; width * height];
    for r in 0..height {
        y[r * width..(r + 1) * width].copy_from_slice(&t.recon[r * t.stride..r * t.stride + width]);
    }
    let (width_uv, height_uv) = if cfg.monochrome {
        (0, 0)
    } else {
        ((width + ss_x) >> ss_x, (height + ss_y) >> ss_y)
    };
    let mut u = vec![0u16; width_uv * height_uv];
    let mut v = vec![0u16; width_uv * height_uv];
    for r in 0..height_uv {
        u[r * width_uv..(r + 1) * width_uv]
            .copy_from_slice(&t.recon_u[r * t.stride_uv..r * t.stride_uv + width_uv]);
        v[r * width_uv..(r + 1) * width_uv]
            .copy_from_slice(&t.recon_v[r * t.stride_uv..r * t.stride_uv + width_uv]);
    }

    FrameDecode {
        y,
        u,
        v,
        width,
        height,
        width_uv,
        height_uv,
        bit_depth: cfg.bd,
        monochrome: cfg.monochrome,
        subsampling_x: ss_x,
        subsampling_y: ss_y,
        base_qindex: p.quant.base_qindex,
        filter_level: [
            p.loopfilter.filter_level[0],
            p.loopfilter.filter_level[1],
            p.loopfilter.filter_level_u,
            p.loopfilter.filter_level_v,
        ],
        cdef_damping: p.cdef.cdef_damping,
        cdef_bits: p.cdef.cdef_bits,
        cdef_strengths: p.cdef.cdef_strengths,
        cdef_uv_strengths: p.cdef.cdef_uv_strengths,
        tx_mode_select: p.tx_mode_select,
        reduced_tx_set: p.reduced_tx_set_used,
        delta_q_present: p.delta_q.delta_q_present,
        seg_enabled: p.segmentation.enabled,
        seg_last_active_segid: crate::calculate_segdata(&bridge_segmentation(&p.segmentation)).1,
        lr_frame_restoration_type: p.restoration.frame_restoration_type,
        lr_unit_counts: {
            let mut c = (0, 0, 0);
            for units in &t.lr_units {
                for u in units {
                    match u.restoration_type {
                        aom_entropy::lr::RESTORE_WIENER => c.0 += 1,
                        aom_entropy::lr::RESTORE_SGRPROJ => c.1 += 1,
                        _ => c.2 += 1,
                    }
                }
            }
            c
        },
        tile_cols: p.tile_info.cols,
        tile_rows: p.tile_info.rows,
    }
}

/// Build the loop-filter mi grid + params from the decoded leaf blocks and
/// frame header — the inputs [`apply_deblock`] filters with. Exposed (hidden)
/// so harnesses can drive the C reference walk over the exact same inputs.
///
/// KEY all-intra flattening: every cell of a block carries the block's
/// `tx_size` (intra tx is uniform — the `LfMi::tx_size` contract), `ref0 =
/// INTRA_FRAME`, `mode_lf = MODE_LF_LUT[y_mode]` (0 for every intra mode),
/// `is_inter = use_intrabc` (rejected upstream, so false), and the block's
/// post-update delta-lf carries.
#[doc(hidden)]
pub fn build_lf_inputs(
    t: &KfTileDecode,
    cfg: &KfTileConfig,
    p: &FrameHeaderObu,
) -> (
    Vec<aom_loopfilter::frame::LfMi>,
    aom_loopfilter::frame::LfParams,
) {
    use aom_loopfilter::frame::{LfMi, LfParams, MODE_LF_LUT};

    let mi_rows = cfg.mi_rows as usize;
    let mi_cols = cfg.mi_cols as usize;
    let mut mi = vec![LfMi::default(); mi_rows * mi_cols];
    for b in &t.blocks {
        let cell = LfMi {
            bsize: b.bsize as u8,
            tx_size: b.tx_size as u8,
            segment_id: b.info.segment_id as u8,
            ref0: 0, // INTRA_FRAME
            mode_lf: MODE_LF_LUT[b.info.y_mode as usize],
            is_inter: b.info.use_intrabc != 0,
            skip_txfm: b.info.skip != 0,
            delta_lf_from_base: b.info.delta_lf_from_base as i8,
            delta_lf: [
                b.info.delta_lf[0] as i8,
                b.info.delta_lf[1] as i8,
                b.info.delta_lf[2] as i8,
                b.info.delta_lf[3] as i8,
            ],
        };
        let h = (MI_SIZE_HIGH[b.bsize] as usize).min(mi_rows - b.mi_row as usize);
        let w = (MI_SIZE_WIDE[b.bsize] as usize).min(mi_cols - b.mi_col as usize);
        for r in 0..h {
            let row0 = (b.mi_row as usize + r) * mi_cols + b.mi_col as usize;
            mi[row0..row0 + w].fill(cell);
        }
    }

    let lfh = &p.loopfilter;
    // Segmentation LF inputs: the SEG_LVL_ALT_LF_* features (C ids 1..=4)
    // re-based to LfSeg's 0..4, exactly what av1_loop_filter_frame_init's
    // per-segment level derivation (and the per-block delta-lf path) read;
    // xd->lossless[i] via the C formula (decodeframe.c:5166) — always false
    // here since whole-frame and per-segment lossless are rejected upstream.
    let seg = bridge_segmentation(&p.segmentation);
    let mut lf_seg = aom_loopfilter::frame::LfSeg {
        enabled: seg.enabled,
        ..Default::default()
    };
    for i in 0..aom_quant::MAX_SEGMENTS {
        for f in 0..4 {
            lf_seg.active[i][f] = seg.feature_mask[i] & (1 << (1 + f)) != 0;
            lf_seg.data[i][f] = i32::from(seg.feature_data[i][1 + f]);
        }
    }
    let q = &p.quant;
    let plane_deltas_zero = q.y_dc_delta_q == 0
        && q.u_dc_delta_q == 0
        && q.u_ac_delta_q == 0
        && q.v_dc_delta_q == 0
        && q.v_ac_delta_q == 0;
    let lossless =
        std::array::from_fn(|i| av1_get_qindex(&seg, i, q.base_qindex) == 0 && plane_deltas_zero);
    let params = LfParams {
        filter_level: lfh.filter_level,
        filter_level_u: lfh.filter_level_u,
        filter_level_v: lfh.filter_level_v,
        sharpness: lfh.sharpness_level,
        mode_ref_delta_enabled: lfh.mode_ref_delta_enabled,
        ref_deltas: lfh.ref_deltas,
        mode_deltas: lfh.mode_deltas,
        delta_lf_present: p.delta_q.delta_lf_present,
        delta_lf_multi: p.delta_q.delta_lf_multi,
        lossless,
        seg: lf_seg,
    };
    (mi, params)
}

/// Run [`aom_loopfilter::frame::loop_filter_frame`] over the (mi-aligned)
/// recon planes, exactly as the C decoder does after tile decode. Hidden:
/// harnesses recompose the filter pipeline stage by stage.
#[doc(hidden)]
pub fn apply_deblock(t: &mut KfTileDecode, cfg: &KfTileConfig, p: &FrameHeaderObu) {
    use aom_loopfilter::frame::{LfFrameBuf, LfMiGrid, loop_filter_frame};

    let (mi, params) = build_lf_inputs(t, cfg, p);
    let grid = LfMiGrid {
        mi: &mi,
        stride: cfg.mi_cols as usize,
        mi_rows: cfg.mi_rows,
        mi_cols: cfg.mi_cols,
    };
    let num_planes = if cfg.monochrome { 1 } else { 3 };
    let mut buf = LfFrameBuf {
        y: &mut t.recon,
        y_stride: t.stride,
        u: &mut t.recon_u,
        v: &mut t.recon_v,
        uv_stride: t.stride_uv,
        // CROP dims (dst.width/height in C — set_lpf_parameters skips edges
        // at/past them). KfTileDecode.width is the mi-ALIGNED width; the
        // coded frame size lives in the header (superres unscaled here).
        crop_width: p.frame_size.superres_upscaled_width as u32,
        crop_height: p.frame_size.superres_upscaled_height as u32,
        ss_x: cfg.subsampling_x,
        ss_y: cfg.subsampling_y,
        bd: cfg.bd,
    };
    loop_filter_frame(&mut buf, &grid, &params, 0, num_planes);
}

/// Run [`aom_cdef::frame::cdef_frame`] over the (mi-aligned, deblocked)
/// recon planes, exactly as the C decoder does after deblocking.
///
/// Input flattening mirrors the C mi grid the walk reads:
/// - per-mi `skip_txfm` from each block's footprint (frame-cropped stamps);
/// - per-64x64-unit strength index from the ONE block per unit whose
///   `read_cdef` returned the literal (the first non-skip block; the C
///   stores it on the unit's top-left grid mbmi and the frame walk reads it
///   back from there — cdef.c:304-308). Units where no block read a
///   strength (all-skip) keep `-1`: they are skipped either way (empty
///   dlist / the -1 arm), matching the C's stale-field-never-consumed
///   behavior. Hidden: harness entry.
#[doc(hidden)]
pub fn apply_cdef(t: &mut KfTileDecode, cfg: &KfTileConfig, p: &FrameHeaderObu) {
    use aom_cdef::frame::{CdefFrameParams, cdef_frame};

    let mi_rows = cfg.mi_rows as usize;
    let mi_cols = cfg.mi_cols as usize;
    let nhfb = mi_cols.div_ceil(16);
    let nvfb = mi_rows.div_ceil(16);
    let mut skip = vec![false; mi_rows * mi_cols];
    let mut unit_strength = vec![-1i32; nvfb * nhfb];
    for b in &t.blocks {
        let h = (MI_SIZE_HIGH[b.bsize] as usize).min(mi_rows - b.mi_row as usize);
        let w = (MI_SIZE_WIDE[b.bsize] as usize).min(mi_cols - b.mi_col as usize);
        let sk = b.info.skip != 0;
        for r in 0..h {
            let row0 = (b.mi_row as usize + r) * mi_cols + b.mi_col as usize;
            skip[row0..row0 + w].fill(sk);
        }
        if b.info.cdef_strength >= 0 {
            unit_strength[(b.mi_row as usize / 16) * nhfb + b.mi_col as usize / 16] =
                b.info.cdef_strength;
        }
    }
    let params = CdefFrameParams {
        mi_rows: cfg.mi_rows,
        mi_cols: cfg.mi_cols,
        num_planes: if cfg.monochrome { 1 } else { 3 },
        ss_x: cfg.subsampling_x,
        ss_y: cfg.subsampling_y,
        bit_depth: cfg.bd,
        damping: p.cdef.cdef_damping,
        cdef_strengths: p.cdef.cdef_strengths,
        cdef_uv_strengths: p.cdef.cdef_uv_strengths,
        skip_txfm: &skip,
        unit_strength: &unit_strength,
    };
    cdef_frame(
        &mut t.recon,
        t.stride,
        &mut t.recon_u,
        &mut t.recon_v,
        t.stride_uv,
        &params,
    );
}
