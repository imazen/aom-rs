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
//!   BLOCK COPY — the OTHER screen-content tool `allow_screen_content_tools`
//!   gates — IS ALSO in the envelope (monochrome and colour): `p.allow_intrabc`
//!   is threaded into the tile driver, block vectors are read at
//!   `MV_SUBPEL_NONE`, and luma/chroma reconstruct via an integer block copy (a
//!   2-tap intrabc bilinear at half-pel on a subsampled chroma axis). Gated by
//!   `intrabc_{monochrome,colour}_streams_decode_byte_identical_to_c`.
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
//!   are in the envelope, for ALL subsampling modes INCLUDING 4:2:2 chroma.
//!   (Past sessions rejected 4:2:2 nonzero-chroma-level deblock believing
//!   libaom's chroma path reads `max_txsize_rect_lookup[BLOCK_INVALID]` OOB
//!   for tall blocks at `ss = (1,0)`; VERIFIED FALSE — the portrait luma
//!   sizes that map to BLOCK_INVALID never reach the chroma loop filter for
//!   a conformant 4:2:2 stream, so that branch is dead. 4:2:2 chroma deblock
//!   is byte-exact vs the C decoder — see real_bitstream.rs's 4:2:2 gate.)
//! - segmentation IS in the envelope: per-block segment ids (spatial-pred
//!   symbols over the current-frame segment map), `SEG_LVL_ALT_Q` per-block
//!   dequant shifts (composing with delta-q), `SEG_LVL_SKIP` forced skips,
//!   and `SEG_LVL_ALT_LF_*` loop-filter level deltas (threaded into the
//!   deblock stage's per-segment level derivation). A MIXED lossless frame —
//!   segmentation on with some but not all segments lossless — stays OUT (see
//!   the coded-lossless bullet).
//! - CODED-LOSSLESS IS in the envelope (`--lossless=1`: `base_qindex == 0` with
//!   zero plane deltas, or every segment lossless). `xd->lossless[i]` flips the
//!   block transform path to forced `TX_4X4` + the 4x4 Walsh–Hadamard (WHT,
//!   [`aom_transform::inv_txfm2d::av1_highbd_iwht4x4_add`]) with the qindex-0
//!   dequant, and `is_cfl_allowed` narrows to `BLOCK_4X4`. The header parse
//!   gates loop-filter / CDEF / restoration / tx-mode off (a two-phase parse:
//!   probe -> compute `coded_lossless` -> re-parse). Only a genuinely MIXED
//!   frame (some-but-not-all segments lossless — never emitted by the real
//!   encoder, not differentially testable) is rejected. Gated by
//!   `lossless_streams_decode_byte_identical_to_c`.
//! - no quantization matrices.
//! - `disable_cdf_update` off (the driver always adapts).
//! - delta-q / delta-lf ARE in the envelope (per-block dequant recompute).
//!
//! The gold test (tests/real_bitstream.rs) compares the output planes
//! byte-identically against the REAL C decoder (`aom_codec_av1_dx`) on
//! bitstreams produced by the REAL encoder at `--cpu-used=0 --end-usage=q`.

use crate::superres;
use crate::{
    KfTileConfig, KfTileDecode, MI_SIZE_HIGH, MI_SIZE_WIDE, TileBoundsKf, TileBytesKf,
    decode_frame_tiles_kf,
};
use aom_entropy::header::{
    CdefHeader, FilmGrainParams, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader,
    LoopfilterHeader, RestorationHeader, SequenceHeaderObu, TileInfoHeader,
    read_sequence_header_obu, read_tile_group_header, read_uncompressed_header,
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
    /// `features.disable_cdf_update` as coded — when true the tile symbol
    /// reader did NOT adapt CDFs (every `read_symbol` left its CDF at the
    /// loaded value). For harness assertions.
    pub disable_cdf_update: bool,
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
/// C's `is_coded_lossless` (`decodeframe.c`): the frame is coded-lossless iff
/// EVERY segment is lossless — effective qindex 0 with all plane dc/ac deltas
/// zero (`xd->lossless[i]`). Segmentation off reduces to `base_qindex == 0`.
/// Drives the forced-`TX_4X4` + WHT block transform path and gates the header's
/// loop-filter / CDEF / restoration / tx-mode reads off.
fn frame_coded_lossless(fh: &FrameHeaderObu) -> bool {
    let q = &fh.quant;
    let plane_deltas_zero = q.y_dc_delta_q == 0
        && q.u_dc_delta_q == 0
        && q.u_ac_delta_q == 0
        && q.v_dc_delta_q == 0
        && q.v_ac_delta_q == 0;
    if !plane_deltas_zero {
        return false;
    }
    if fh.segmentation.enabled {
        let seg = bridge_segmentation(&fh.segmentation);
        // is_coded_lossless loops ALL MAX_SEGMENTS, not just the reachable ids.
        (0..aom_quant::MAX_SEGMENTS).all(|i| av1_get_qindex(&seg, i, q.base_qindex) == 0)
    } else {
        q.base_qindex == 0
    }
}

fn parse_frame_header(
    seq: &SequenceHeaderObu,
    payload: &[u8],
    is_obu_frame: bool,
) -> Result<ParsedFrame, String> {
    parse_frame_header_ext(seq, payload, is_obu_frame, false)
}

/// [`parse_frame_header`] with an `allow_inter` gate. When `false` (the
/// single-KEY-frame path), an inter frame is rejected as before. When `true`
/// (the multi-frame [`decode_frames`] path), an INTER frame is accepted: the
/// ref-mv gate inputs `read_uncompressed_header` consumes
/// (`might_allow_ref_frame_mvs`, `cur_frame_force_integer_mv`) are derived from
/// the sequence header (the `01-size-*` envelope: `enable_ref_frame_mvs` +
/// `enable_order_hint`, `force_integer_mv == 0`, non-error-resilient).
fn parse_frame_header_ext(
    seq: &SequenceHeaderObu,
    payload: &[u8],
    is_obu_frame: bool,
    allow_inter: bool,
) -> Result<ParsedFrame, String> {
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let num_planes = if c.monochrome { 1 } else { 3 };
    // `set_sb_size` (av1_common_int.h): mib_size_log2 = mi_size_wide_log2[sb_size]
    // — 4 for BLOCK_64X64 (16 mi/side), 5 for BLOCK_128X128 (32 mi/side).
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
        // Reader context for `read_film_grain_params`: the chroma-absent gate
        // reads monochrome + subsampling from the color config.
        film_grain: FilmGrainParams {
            monochrome: c.monochrome,
            subsampling_x: c.subsampling_x,
            subsampling_y: c.subsampling_y,
            ..Default::default()
        },
        // coded_lossless / all_lossless are inputs to this writer-mirror
        // reader but are stream facts on the decode side: the probe pass below
        // parses as non-lossless (correct for the quant + segmentation, which
        // precede the lossless-gated tail), then recomputes them and re-parses
        // when the stream is coded-lossless.
        ..Default::default()
    };
    if allow_inter {
        // Ref-mv gate input `read_uncompressed_header`'s inter branch consumes.
        // `use_ref_frame_mvs = enable_ref_frame_mvs && enable_order_hint &&
        // !error_resilient_mode && !FrameIsIntra` (spec 5.9.2); the envelope is
        // non-error-resilient, so derive from the sequence flags.
        // (`cur_frame_force_integer_mv` is read from the stream by the prefix,
        // so it is NOT set here.)
        cfg.might_allow_ref_frame_mvs = s.enable_ref_frame_mvs && s.enable_order_hint;
        // `frame_might_allow_warped_motion` needs the sequence's
        // `enable_warped_motion`; `read_uncompressed_header` combines it with the
        // parsed `!FrameIsIntra && !error_resilient_mode` gate. Without this the
        // reader defaulted it to `false`, SKIPPING the `allow_warped_motion` bit and
        // shifting `reduced_tx_set` (which then read the warped bit) — the 1-bit
        // slip is hidden by the header's trailing byte-alignment, so modes still
        // parsed but the tx-type set was wrong.
        cfg.might_allow_warped_motion = s.enable_warped_motion;
    }

    // Two-phase header parse for coded-lossless. `read_uncompressed_header` gates
    // its loop-filter / CDEF / restoration / tx-mode reads on
    // `cfg.coded_lossless` / `cfg.all_lossless` — a writer-mirror INPUT (the
    // minimal-header anchor tests rely on that), whereas the decoder only learns
    // the lossless status by computing it from the parsed quant + segmentation
    // (exactly what `decodeframe.c` does mid-header). Those are parsed BEFORE the
    // gated sections, so a first pass with `coded_lossless = false` yields exact
    // quant/segmentation regardless of the (then-misread) tail; recompute
    // `coded_lossless` from them and, when the stream IS coded-lossless (what
    // `--lossless=1` produces), re-parse with the correct gating so the tail is
    // read right. Only a shown KEY frame reaches the lossless path (everything
    // else is rejected just below), so guard the recompute + re-parse on that.
    let mut rb = ReadBitBuffer::new(payload);
    let probe = read_uncompressed_header(&mut rb, &cfg);
    let key_shown = !probe.prefix.show_existing_frame
        && probe.prefix.frame_type == 0
        && probe.prefix.show_frame;
    // Superres codes the tile info (and everything after the frame size) over the
    // DOWNSCALED mi grid — `compute_image_size` derives MiCols from the reduced
    // FrameWidth. The frame size (including the superres denom) is read correctly
    // on the first pass: it precedes the tile info and doesn't depend on the mi
    // grid. So when superres is active, re-parse with a `tile_info` built on the
    // downscaled mi grid (composed with the coded-lossless re-parse).
    let superres_denom = probe.frame_size.scale_denominator;
    let superres = crate::superres::superres_scaled(superres_denom);
    // Under superres the probe's full (upscaled-width) tile_info is OVER-SIZED, so
    // every field the probe reads after it — the quant params included — comes off
    // a shifted bit position and is garbage. `coded_lossless` MUST therefore be
    // decided from quant read with the CORRECT (downscaled) grid; detecting it from
    // the probe's garbage FALSE-POSITIVES (a normal frame reads as qindex 0), which
    // would then drop the loop-filter + CDEF header sections and desync the
    // tile-data offset (KB-14: the denom-16 exact-2:1 corner, where the coded frame
    // is a single 64-wide superblock, was the first stream to trip this). So when
    // superres is active, parse a probe on the downscaled grid FIRST and reuse it as
    // the final header (re-parsing only if it turns out to be coded-lossless).
    let sr_probe = superres.then(|| {
        let coded_w = crate::superres::coded_frame_width(
            probe.frame_size.superres_upscaled_width,
            superres_denom,
        );
        let mut cfg_sr = cfg.clone();
        cfg_sr.superres_scaled = true;
        cfg_sr.tile_info = tile_limits(mi_dim(coded_w), mi_rows, mib_size_log2);
        let mut rb_sr = ReadBitBuffer::new(payload);
        let p_sr = read_uncompressed_header(&mut rb_sr, &cfg_sr);
        (cfg_sr, p_sr, rb_sr)
    });
    let coded_lossless =
        key_shown && frame_coded_lossless(sr_probe.as_ref().map_or(&probe, |(_, p_sr, _)| p_sr));
    let (p, mut rb) = match sr_probe {
        // Superres. `cfg_sr`/`p_sr` already use the downscaled grid; `p_sr` was
        // parsed with coded_lossless=false (all_lossless is always false under
        // superres = coded_lossless && !superres_scaled). Reuse it unless the frame
        // is coded-lossless, which gates the loop-filter/CDEF sections off and needs
        // one more parse with the flag set.
        Some((mut cfg_sr, p_sr, rb_sr)) => {
            if coded_lossless {
                cfg_sr.coded_lossless = true;
                let mut rb2 = ReadBitBuffer::new(payload);
                let p2 = read_uncompressed_header(&mut rb2, &cfg_sr);
                (p2, rb2)
            } else {
                (p_sr, rb_sr)
            }
        }
        // No superres: the probe is authoritative. Re-parse only for coded-lossless
        // (all_lossless = coded_lossless && !superres_scaled = coded_lossless).
        None if coded_lossless => {
            let mut cfg2 = cfg.clone();
            cfg2.coded_lossless = true;
            cfg2.all_lossless = true;
            let mut rb2 = ReadBitBuffer::new(payload);
            let p2 = read_uncompressed_header(&mut rb2, &cfg2);
            (p2, rb2)
        }
        None => (probe, rb),
    };

    // --- envelope gates, in bitstream order ---
    if p.prefix.show_existing_frame {
        return Err("show_existing_frame".into());
    }
    // frame_type: 0=KEY, 1=INTER, 2=INTRA_ONLY, 3=SWITCH. The single-frame path
    // accepts only KEY; the multi-frame path additionally accepts INTER (the
    // inter walking-skeleton envelope).
    if p.prefix.frame_type != 0 && !(allow_inter && p.prefix.frame_type == 1) {
        return Err(format!("frame_type {} (unsupported)", p.prefix.frame_type));
    }
    if !p.prefix.show_frame {
        return Err("unshown frame".into());
    }
    // `disable_cdf_update` IS in the envelope: when set (the encoder's
    // `cdf_update_mode == 0`, and forced on by `error_resilient_mode` for
    // reference frames), the tile symbol reader leaves every CDF at its
    // loaded/initial value instead of adapting after each `read_symbol`. Carried
    // as `cfg.disable_cdf_update` -> `dec.allow_update_cdf = !disable_cdf_update`
    // in `decode_frame_tiles_kf`; the flag-off path stays byte-identical.
    // `disable_frame_end_update_cdf` (which `error_resilient_mode` also forces)
    // is decode-irrelevant here: it only governs whether this frame's END adapted
    // CDFs are saved as a frame context for LATER frames that reference it, and
    // this driver decodes exactly one shown KEY frame with no forward reference
    // chain (`REFRESH_FRAME_CONTEXT` save is never read back).
    // `allow_screen_content_tools` gates BOTH screen-content tools that ARE in
    // the envelope: PALETTE mode and intra block copy. `p.allow_intrabc` (the
    // per-frame syntax flag the spec reads only when `allow_screen_content_tools`
    // holds — `p.allow_intrabc = p.prefix.allow_screen_content_tools &&
    // !cfg.superres_scaled && rb.read_bit() != 0`) is threaded into the tile
    // driver (see `allow_intrabc` in the KfTileConfig below), not rejected:
    // intrabc luma is an integer block copy from the already-decoded region and
    // chroma reuses the scaled DV (integer copy or a 2-tap bilinear at half-pel).
    if p.frame_size.superres_upscaled_width != s.max_frame_width
        || p.frame_size.superres_upscaled_height != s.max_frame_height
    {
        return Err("frame_size_override (frame != sequence max dims)".into());
    }
    // Superres (SuperresDenom in [9,16]) IS in the envelope: the frame is coded
    // at a reduced width and upscaled back to the full UpscaledWidth
    // horizontally, as a normative post-CDEF stage ([`crate::superres`], spliced
    // between CDEF and loop-restoration in `decode_frame_obus`). Only the
    // single-tile-column upscale is implemented — the AVIF-still / KEY superres
    // path is single-tile; multi-tile superres would need the per-tile-column
    // convolve loop (`av1_upscale_normative_rows`'s tile walk).
    if superres::superres_scaled(p.frame_size.scale_denominator) && p.tile_info.cols > 1 {
        return Err("multi-tile superres (out of envelope)".into());
    }
    // Quantization matrices (`using_qmatrix`): each 2-D-transform coefficient is
    // dequantized with a per-position inverse-QM weight
    // (`av1_get_iqmatrix`/[`crate::qm`]) instead of the flat step. Decoded via
    // the per-block `block_qm_level` + `qm::iqmatrix` threading below; no reject.
    let q = &p.quant;
    // Coded-lossless (`--lossless=1`: base_qindex 0 + zero deltas, or every
    // segment lossless) IS in the envelope: `coded_lossless` above drove the
    // re-parse's LF/CDEF/restoration/tx-mode gating, the tile driver forces
    // TX_4X4 + WHT per block (`st.coded_lossless`), and the frame-level
    // deblock/CDEF/LR gates below see everything off. What stays OUT is a MIXED
    // frame — segmentation enabled with SOME but not ALL segments lossless (C's
    // coded_lossless is false, yet a block in a lossless segment still forces the
    // WHT): the real encoder never emits it (av1_vaq clamps base+delta >= 1 on
    // non-lossless frames; --lossless disables segmentation), it isn't
    // differentially testable, and its per-segment `cfl_allowed` would diverge
    // from the value the driver precomputes once. Reject it cleanly.
    if p.segmentation.enabled {
        // KEY frame, no primary ref: the parse forces map+data updates on and
        // temporal prediction off (setup_segmentation's PRIMARY_REF_NONE arm).
        debug_assert!(
            p.segmentation.update_map
                && p.segmentation.update_data
                && !p.segmentation.temporal_update
        );
        let seg = bridge_segmentation(&p.segmentation);
        let (_, last_active) = crate::calculate_segdata(&seg);
        let plane_deltas_zero = q.y_dc_delta_q == 0
            && q.u_dc_delta_q == 0
            && q.u_ac_delta_q == 0
            && q.v_dc_delta_q == 0
            && q.v_ac_delta_q == 0;
        // A reachable (decodable) segment is lossless but the frame is not
        // coded-lossless -> a genuine mix. `coded_lossless` already checked ALL
        // MAX_SEGMENTS, so this fires only for the true mixed case.
        let any_reachable_lossless = plane_deltas_zero
            && (0..=last_active as usize).any(|i| av1_get_qindex(&seg, i, q.base_qindex) == 0);
        if any_reachable_lossless && !coded_lossless {
            return Err("mixed lossless/non-lossless segments (out of envelope)".into());
        }
    }
    // 4:2:2 chroma deblocking IS in the envelope. libaom's
    // `av1_ss_size_lookup[bsize][1][0]` marks the portrait (height>width)
    // luma sizes BLOCK_INVALID — the ss_x-only chroma-plane-size that
    // `av1_get_max_uv_txsize` would index `max_txsize_rect_lookup` with — but
    // those portrait sizes never reach the chroma loop filter for a conformant
    // 4:2:2 stream (they are never chroma-reference blocks; measured: zero OOB
    // hits across a wide encode/decode sweep — see the 4:2:2 deblock gate in
    // real_bitstream.rs). So the BLOCK_INVALID branch is dead, and the wide/
    // square chroma tx sizes that DO occur deblock byte-exact vs the C decoder.

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
    run_post_filters(&mut t, &cfg, &header);
    Ok(finish_and_grain(t, &cfg, &header))
}

/// Crop the filtered reconstruction to the display planes and apply film grain
/// (the final post-reconstruction output stage). Shared by [`decode_frame_obus`]
/// and the multi-frame [`decode_frames`] path.
fn finish_and_grain(t: KfTileDecode, cfg: &KfTileConfig, header: &FrameHeaderObu) -> FrameDecode {
    let mut fd = finish_frame(t, cfg, header);
    // Film grain is applied at the decoder as the final post-reconstruction
    // output stage (av1_add_film_grain), on the cropped display planes, exactly
    // as `aom_codec_get_frame` does. Byte-identical to C (film_grain_diff.rs).
    if header.film_grain_params_present && header.film_grain.apply_grain {
        // AOM_CICP_MC_IDENTITY (0) selects the luma legal range for chroma
        // under clip_to_restricted_range.
        let mc_identity = cfg.matrix_coefficients == 0;
        let (gy, gu, gv) = crate::film_grain::add_film_grain(
            &header.film_grain,
            fd.bit_depth,
            fd.monochrome,
            cfg.subsampling_x as i32,
            cfg.subsampling_y as i32,
            mc_identity,
            fd.width,
            fd.height,
            &fd.y,
            &fd.u,
            &fd.v,
        );
        fd.y = gy;
        fd.u = gu;
        fd.v = gv;
    }
    fd
}

/// Decode an AV1 bitstream that may contain INTER frames after the KEY frame,
/// returning one [`FrameDecode`] per shown frame in decode order. The inter
/// walking-skeleton envelope (`av1-1-b8-01-size-*`): a KEY frame 0 whose
/// filtered reconstruction becomes the single `LAST` reference for the inter
/// frames that follow (`primary_ref = NONE`, single reference, no forward CDF
/// chain). The single-KEY-frame [`decode_frame_obus`] is unchanged.
pub fn decode_frames(data: &[u8]) -> Result<Vec<FrameDecode>, String> {
    let mut pos = 0usize;
    let mut seq: Option<SequenceHeaderObu> = None;
    let mut pending_header: Option<FrameHeaderObu> = None;
    let mut out: Vec<FrameDecode> = Vec::new();
    // The reference DPB: for this envelope a single stored `LAST` reference is
    // sufficient (frame 0's filtered recon feeds every following inter frame).
    let mut last_ref: Option<crate::RefFrame> = None;

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
            2 => {} // OBU_TEMPORAL_DELIMITER
            1 => {
                let mut rb = ReadBitBuffer::new(payload);
                let sh = read_sequence_header_obu(&mut rb);
                let cc = &sh.color_config;
                if !cc.monochrome {
                    let ss = (cc.subsampling_x, cc.subsampling_y);
                    if !matches!(ss, (0, 0) | (1, 0) | (1, 1)) {
                        return Err(format!("unsupported subsampling {ss:?}"));
                    }
                }
                seq = Some(sh);
            }
            3 => {
                let sh = seq.as_ref().ok_or("frame header before sequence header")?;
                let pf = parse_frame_header_ext(sh, payload, false, true)?;
                pending_header = Some(pf.header);
            }
            4 | 6 => {
                let sh = seq.as_ref().ok_or("frame before sequence header")?;
                let (header, tile_data) = if h.obu_type == 6 {
                    let pf = parse_frame_header_ext(sh, payload, true, true)?;
                    let off = pf.tile_data_off.unwrap();
                    (pf.header, &payload[off..])
                } else {
                    let header = pending_header
                        .take()
                        .ok_or("tile group without frame header")?;
                    let mut rb = ReadBitBuffer::new(payload);
                    read_full_tile_group(&mut rb, &header.tile_info)?;
                    let off = rb.bytes_read();
                    (header, &payload[off..])
                };
                let (mut t, cfg, hdr) = if header.prefix.frame_type == 1 {
                    let last = last_ref
                        .as_ref()
                        .ok_or("inter frame decoded before any reference frame")?;
                    decode_inter_tile_payload(sh, &header, tile_data, last)?
                } else {
                    decode_tile_payload(sh, &header, tile_data)?
                };
                run_post_filters(&mut t, &cfg, &hdr);
                // Store the FILTERED reconstruction (pre-crop, pre-film-grain) as
                // the reference for later frames. Every `01-size-*` frame refreshes
                // at least one slot; a single stored `LAST` covers this envelope.
                if hdr.prefix.refresh_frame_flags != 0 {
                    // MC clamps reference reads to the VISIBLE (crop) dims — the
                    // reference's `y_crop_width/height` / `uv_crop_width/height`
                    // (the coded post-superres-upscale visible size), matching
                    // `finish_frame`'s crop derivation. On a partial-edge frame
                    // these are smaller than the mi-aligned recon extent.
                    let crop_w = hdr.frame_size.superres_upscaled_width as usize;
                    let crop_h = hdr.frame_size.superres_upscaled_height as usize;
                    let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
                    let crop_w_uv = (crop_w + ss_x) >> ss_x;
                    let crop_h_uv = (crop_h + ss_y) >> ss_y;
                    last_ref = Some(crate::RefFrame::from_filtered(
                        &t,
                        hdr.prefix.order_hint,
                        crop_w,
                        crop_h,
                        crop_w_uv,
                        crop_h_uv,
                    ));
                }
                out.push(finish_and_grain(t, &cfg, &hdr));
            }
            5 | 15 => {} // OBU_METADATA | OBU_PADDING
            t => return Err(format!("unsupported OBU type {t}")),
        }
        pos = end;
    }

    if out.is_empty() {
        return Err("no frame in stream".into());
    }
    Ok(out)
}

/// The INTER analogue of [`decode_tile_payload`]: build the shared tile config +
/// the inter frame-level config (single `LAST` reference) and drive the inter
/// tile decode.
fn decode_inter_tile_payload(
    seq: &SequenceHeaderObu,
    p: &FrameHeaderObu,
    tile_data: &[u8],
    last: &crate::RefFrame,
) -> Result<(KfTileDecode, KfTileConfig, FrameHeaderObu), String> {
    let cfg = build_tile_cfg(seq, p);
    let inter = crate::InterFrameCfg {
        last,
        allow_high_precision_mv: p.allow_high_precision_mv,
        cur_frame_force_integer_mv: p.cur_frame_force_integer_mv,
        interp_filter: p.interp_filter,
        switchable_motion_mode: p.switchable_motion_mode,
        allow_ref_frame_mvs: p.allow_ref_frame_mvs,
        reference_mode_select: p.reference_mode_select,
        enable_dual_filter: seq.seq_header.enable_dual_filter,
        allow_warped_motion: p.allow_warped_motion,
        skip_mode_allowed: p.skip_mode_allowed,
        enable_interintra_compound: seq.seq_header.enable_interintra_compound,
        gm_wmtype: std::array::from_fn(|i| p.global_motion[i].wmtype),
        order_hint: p.prefix.order_hint,
    };
    let tiles = split_tiles(tile_data, &p.tile_info, p.tile_size_bytes)?;
    let t = crate::decode_frame_tiles_inter(&tiles, &cfg, &inter, 0);
    Ok((t, cfg, p.clone()))
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

/// Upscale a downscaled `(y, u, v)` plane triplet horizontally to the full
/// UpscaledWidth (`av1_upscale_normative_rows`, single tile column), returning
/// `(y, stride, u, v, stride_uv)` at the upscaled width (tight strides). Height
/// and row count are unchanged — superres is horizontal only. Shared by
/// [`apply_superres`] (the post-CDEF recon) and the deblocked-snapshot upscale
/// that feeds LR's internal stripe boundaries. `src_stride`/`src_stride_uv` are
/// the DOWNSCALED strides; `cfg.mi_cols` is the downscaled mi grid.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn superres_upscale_planes(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    src_stride: usize,
    src_stride_uv: usize,
    cfg: &KfTileConfig,
    p: &FrameHeaderObu,
) -> (Vec<u16>, usize, Vec<u16>, Vec<u16>, usize) {
    let denom = p.frame_size.scale_denominator;
    let upscaled_w = p.frame_size.superres_upscaled_width;
    let coded_w = superres::coded_frame_width(upscaled_w, denom);
    let (ss_x, ss_y) = (cfg.subsampling_x as i32, cfg.subsampling_y as i32);
    let bd = cfg.bd;
    // mi-aligned downscaled luma width — the border-extension clamp bound
    // (libaom's `mi_col_end << MI_SIZE_LOG2` for a single tile spanning mi_cols).
    // The recon stride is SB-aligned (>= this); pixels past it are the decoded
    // SB overhang that libaom's normative upscale does NOT read.
    let mi_w_luma = cfg.mi_cols * 4;

    // Luma.
    let rows = y.len() / src_stride;
    let dst_stride = upscaled_w as usize;
    let mut y_up = vec![0u16; dst_stride * rows];
    superres::upscale_plane(
        y, src_stride, &mut y_up, dst_stride, coded_w, upscaled_w, mi_w_luma, rows, bd,
    );

    // Chroma (absent for monochrome).
    let (u_up, v_up, dst_stride_uv) = if cfg.monochrome || u.is_empty() {
        (Vec::new(), Vec::new(), 0usize)
    } else {
        let rows_uv = u.len() / src_stride_uv;
        let coded_w_uv = (coded_w + ss_x) >> ss_x;
        let upscaled_w_uv = (upscaled_w + ss_x) >> ss_x;
        let mi_w_uv = mi_w_luma >> ss_x;
        let dst_stride_uv = upscaled_w_uv as usize;
        let _ = ss_y; // superres does not touch the vertical axis
        let mut u2 = vec![0u16; dst_stride_uv * rows_uv];
        let mut v2 = vec![0u16; dst_stride_uv * rows_uv];
        superres::upscale_plane(
            u,
            src_stride_uv,
            &mut u2,
            dst_stride_uv,
            coded_w_uv,
            upscaled_w_uv,
            mi_w_uv,
            rows_uv,
            bd,
        );
        superres::upscale_plane(
            v,
            src_stride_uv,
            &mut v2,
            dst_stride_uv,
            coded_w_uv,
            upscaled_w_uv,
            mi_w_uv,
            rows_uv,
            bd,
        );
        (u2, v2, dst_stride_uv)
    };
    (y_up, dst_stride, u_up, v_up, dst_stride_uv)
}

/// Apply normative superres upscale in place on `t` (the post-CDEF recon):
/// widen every plane back to UpscaledWidth and update the strides. Loop
/// restoration (upscaled domain) and `finish_frame` (crop to UpscaledWidth) then
/// run on the result. Hidden: harness entry so tests can drive the stage.
#[doc(hidden)]
pub fn apply_superres(t: &mut KfTileDecode, cfg: &KfTileConfig, p: &FrameHeaderObu) {
    let (y, s, u, v, s_uv) = superres_upscale_planes(
        &t.recon,
        &t.recon_u,
        &t.recon_v,
        t.stride,
        t.stride_uv,
        cfg,
        p,
    );
    t.recon = y;
    t.stride = s;
    t.recon_u = u;
    t.recon_v = v;
    t.stride_uv = s_uv;
}

/// The in-place post-reconstruction filter pipeline (deblock -> CDEF ->
/// superres -> loop restoration), mutating `t` from the tile-decoded pre-filter
/// state to the final FILTERED reconstruction (still mi-aligned, pre-crop,
/// pre-film-grain). Shared by [`decode_frame_obus`] (single KEY frame) and the
/// multi-frame [`crate::inter::decode_frames`] path (where the filtered result
/// also becomes a reference frame).
pub(crate) fn run_post_filters(t: &mut KfTileDecode, cfg: &KfTileConfig, header: &FrameHeaderObu) {
    if header.loopfilter.filter_level != [0, 0] {
        apply_deblock(t, cfg, header);
    }
    // The C decoder's do_cdef gate (decodeframe.c:5417): !skip_loop_filter
    // (a decoder option, always off here) && !coded_lossless (rejected
    // upstream) && any CDEF syntax present. allow_intrabc (which would force
    // cdef_bits == 0) and multi-tile large-scale decoding are rejected too.
    let cd = &header.cdef;
    let do_cdef = cd.cdef_bits != 0 || cd.cdef_strengths[0] != 0 || cd.cdef_uv_strengths[0] != 0;
    let do_lr = cfg.lr.any_enabled();
    let do_superres = superres::superres_scaled(header.frame_size.scale_denominator);
    // decodeframe.c:5422: optimized_loop_restoration = !do_cdef && !do_superres.
    // The non-optimized arm saves the DEBLOCKED rows (pre-CDEF) as internal
    // stripe boundary context before CDEF runs, and the CDEF output rows as
    // frame-edge context after; the optimized arm saves nothing (the frame's own
    // rows are the context). Superres ALWAYS takes the non-optimized arm (even
    // with CDEF off) because the boundary rows must be upscaled.
    let optimized_lr = !do_cdef && !do_superres;
    let mut pre_cdef =
        (do_lr && !optimized_lr).then(|| (t.recon.clone(), t.recon_u.clone(), t.recon_v.clone()));
    if do_cdef {
        apply_cdef(t, cfg, header);
    }
    // Superres upscale (decodeframe.c:5451 `superres_post_decode`): AFTER CDEF,
    // BEFORE loop restoration. Horizontal-only; widens the coded (downscaled)
    // recon back to UpscaledWidth. The pre-CDEF deblocked snapshot that feeds
    // LR's internal stripe boundaries is upscaled the same way — matching C's
    // `save_deblock_boundary_lines`, which runs `av1_upscale_normative_rows` on
    // those boundary rows — so LR runs entirely in the upscaled domain.
    if do_superres {
        let (ds_stride, ds_stride_uv) = (t.stride, t.stride_uv);
        apply_superres(t, cfg, header);
        if let Some((dy, du, dv)) = pre_cdef.take() {
            let (uy, _, uu, uv, _) =
                superres_upscale_planes(&dy, &du, &dv, ds_stride, ds_stride_uv, cfg, header);
            pre_cdef = Some((uy, uu, uv));
        }
    }
    if do_lr {
        apply_restoration(t, cfg, pre_cdef.as_ref(), optimized_lr);
    }
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
                // Film grain IS in the envelope: `film_grain_params_present`
                // enables the per-frame grain params (parsed in the frame
                // header) and byte-exact post-reconstruction synthesis
                // ([`crate::film_grain::add_film_grain`], applied in
                // [`decode_frame_obus`]). Gated by
                // `film_grain_streams_decode_byte_identical_to_c`.
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
    let cfg = build_tile_cfg(seq, p);
    let tiles = split_tiles(tile_data, &p.tile_info, p.tile_size_bytes)?;
    let t = decode_frame_tiles_kf(&tiles, &cfg, 0);
    Ok((t, cfg, p.clone()))
}

/// Build the per-frame tile config shared by the KEY ([`decode_tile_payload`])
/// and INTER ([`decode_inter_tile_payload`]) tile-decode entries — one source
/// of truth so the two paths cannot drift.
fn build_tile_cfg(seq: &SequenceHeaderObu, p: &FrameHeaderObu) -> KfTileConfig {
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let (ss_x, ss_y) = if c.monochrome {
        (1usize, 1usize)
    } else {
        (c.subsampling_x as usize, c.subsampling_y as usize)
    };

    // The tile decode / deblock / CDEF all run in the CODED (downscaled) domain.
    // With superres the coded FrameWidth is the reduced width; the mi grid,
    // strides, and per-block reconstruction are sized to it. Height is never
    // scaled (superres is horizontal only). For unscaled frames
    // `coded_frame_width` returns the width unchanged, so this is a no-op.
    let coded_width = crate::superres::coded_frame_width(
        p.frame_size.superres_upscaled_width,
        p.frame_size.scale_denominator,
    );
    KfTileConfig {
        mi_rows: mi_dim(s.max_frame_height),
        mi_cols: mi_dim(coded_width),
        bd: c.bit_depth,
        monochrome: c.monochrome,
        subsampling_x: ss_x,
        subsampling_y: ss_y,
        matrix_coefficients: c.matrix_coefficients,
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
            // The RU grid is always in the UPSCALED domain
            // (`av1_get_upsampled_plane_size`); `superres_denom` lets
            // `lr_corners_in_sb` map the downscaled mi position into it.
            crop_width: p.frame_size.superres_upscaled_width,
            crop_height: p.frame_size.superres_upscaled_height,
            superres_denom: p.frame_size.scale_denominator,
        },
        seg: bridge_segmentation(&p.segmentation),
        sb_size_128: s.sb_size_128,
        allow_screen_content_tools: p.prefix.allow_screen_content_tools,
        allow_intrabc: p.allow_intrabc,
        using_qmatrix: p.quant.using_qmatrix,
        qm_y: p.quant.qmatrix_level_y as usize,
        qm_u: p.quant.qmatrix_level_u as usize,
        qm_v: p.quant.qmatrix_level_v as usize,
        disable_cdf_update: p.prefix.disable_cdf_update,
    }
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
        disable_cdf_update: cfg.disable_cdf_update,
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
        // at/past them). Deblock runs in the CODED (downscaled) domain, BEFORE
        // superres upscaling, so the crop is the coded FrameWidth — NOT the
        // upscaled width (using the upscaled width would filter the mi-overhang
        // columns past the coded edge as if they were real content, corrupting
        // the right-edge pixels the upscale then samples). Height is never
        // scaled. `coded_frame_width` is a no-op for unscaled frames.
        crop_width: superres::coded_frame_width(
            p.frame_size.superres_upscaled_width,
            p.frame_size.scale_denominator,
        ) as u32,
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
            // C reads the CDEF strength once per 64x64 unit and stores it on the
            // block's shared MB_MODE_INFO (decodemv.c read_cdef); the frame walk
            // reads it back from each 64x64 unit's top-left mi (cdef.c:304). A
            // block LARGER than 64x64 (128x128 / 128x64 / 64x128, only chosen at
            // very high qindex) covers several 64x64 units, all sharing that one
            // mbmi, so EVERY covered unit gets the strength — not just the
            // top-left one. Stamp all covered units (in-frame extent h x w).
            let (ur0, uc0) = (b.mi_row as usize / 16, b.mi_col as usize / 16);
            let ur1 = (b.mi_row as usize + h - 1) / 16;
            let uc1 = (b.mi_col as usize + w - 1) / 16;
            for ur in ur0..=ur1 {
                for uc in uc0..=uc1 {
                    unit_strength[ur * nhfb + uc] = b.info.cdef_strength;
                }
            }
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
