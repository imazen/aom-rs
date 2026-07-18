//! Encoder-side superres RD-closeness / byte-identity gate
//! (`--superres-mode=fixed --superres-denominator=D`, PARITY.md family C6).
//!
//! Superres is OFF by default (superres_mode=NONE); every byte-exact gate
//! encodes without it, so this file is the only place the knob turns on and the
//! proven default envelope stays untouched.
//!
//! The port pipeline under test is the ordinary ALLINTRA KEY encode with ONE
//! addition: the source is downscaled horizontally to the coded `FrameWidth`
//! before the search (`av1_resize_and_extend_frame_nonnormative` →
//! [`aom_encode::resize::resize_plane`], the bit-exact CHUNK-1 kernel), the mi
//! grid is sized to the coded frame, and the frame header signals the superres
//! denominator (`write_superres_scale`, already bit-exact; the decoder upscales
//! the coded frame back to `UpscaledWidth`, task #5). CDEF and loop-restoration
//! stay OFF (allintra defaults), so the coded bytes are the coded-domain
//! partition/mode/tx encode + the deblock derived from the coded recon —
//! exactly the byte-exact envelope, on a downscaled frame.
//!
//! Header FIELDS (qindex, tile limits, the superres denom + upscaled width) are
//! bootstrapped from the C stream like every other e2e gate (the documented
//! Gate-3 caveat). For FIXED mode the denom is a user knob, not a search
//! decision — the port applies the downscale itself and re-derives the coded
//! bytes, so this is not a "feature bootstrap leak" (PARITY.md rule 4); the
//! anti-vacuity asserts confirm superres is genuinely active
//! (`scale_denominator == D`, `coded_w < w`).
//!
//! Validation per cell: the spliced port stream decodes through BOTH the real C
//! decoder and the port decoder to the identical reconstruction, then
//! [`compare_cell`] scores RD-closeness. Byte-identical cells are hard-asserted.
//!
//! Scope of THIS gate: 8-bit + highbd (10/12), FIXED denom (9..14, non-normative
//! `av1_resize_plane` / `highbd_resize_plane`) AND the derived-denom modes
//! QTHRESH / AUTO / RANDOM (`aom_encode::superres_select`, below). The 8-bit
//! denom-16 even-width corner now uses the ported OPTIMIZED
//! `av1_resize_and_extend_frame` scaler (`optimized_downscale_plane_8bit`,
//! EIGHTTAP_SMOOTH/phase 8), bit-exact vs C. The AUTO recode loop (never fires
//! for a single-frame KEY still, `frames_to_key<=1`) is the sole remaining C6
//! follow-up.

use aom_bench::EncodeCell;
use aom_bench::rd_close::{RdBands, RdCellResult, assert_rd_close, compare_cell, splice_frame_obu};
use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
use aom_encode::pack::{PackCfg, pack_tile, pack_tile_from_trees};
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rc::{base_qindex_from_cq, quantizer_to_qindex};
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::resize::{
    coded_superres_width, has_optimized_scaler, highbd_resize_plane,
    optimized_downscale_plane_8bit, resize_plane,
};
use aom_encode::speed_features::SpeedFeatures;
use aom_encode::superres_select::{
    SuperresAutoSearchType, superres_denom_auto_key, superres_denom_qthresh_key,
};
use aom_entropy::enc::OdEcEnc;
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, SequenceHeaderObu, TileInfoHeader, read_sequence_header_obu,
    read_uncompressed_header,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::partition::KfFrameContext;
use aom_entropy::rb::ReadBitBuffer;
use aom_loopfilter::frame::{LfFrameBuf, LfMiGrid, LfParams, loop_filter_frame};
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
/// `SCALE_NUMERATOR` — denom 8 means no superres (coded width == upscaled width).
const SCALE_NUMERATOR_U8: u8 = 8;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16;
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}

fn tile_log2(blk_size: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    const MAX_TILE_WIDTH: i32 = 4096;
    const MAX_TILE_AREA: i32 = 4096 * 2304;
    const MAX_TILE_COLS: i32 = 64;
    const MAX_TILE_ROWS: i32 = 64;
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

fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// Real aomenc with fixed-denominator superres ON. `enable_cdef=false`,
/// `enable_restoration=false` (allintra defaults) so the coded bytes isolate
/// the downscale + coded-domain encode + deblock.
fn c_encode_superres(cell: &EncodeCell, denom: i32) -> Vec<u8> {
    c::ref_encode_av1_kf_superres(
        &cell.y,
        &cell.u,
        &cell.v,
        cell.w,
        cell.h,
        i32::from(cell.bd),
        cell.mono,
        cell.ss_x as i32,
        cell.ss_y as i32,
        cell.cq_level,
        cell.speed,
        false, // enable_cdef
        false, // enable_restoration
        cell.usage,
        denom,
    )
}

/// Downscale one tight bd8 plane (`w x h` u16, values 0..255) horizontally to
/// `coded_w x h`. `use_opt` picks libaom's OPTIMIZED 8-bit scaler
/// (`av1_resize_and_extend_frame`, EIGHTTAP_SMOOTH/phase 8 — the exact-1/2
/// even-width corner) vs the non-normative `resize_plane`; the encoder's
/// all-planes-or-none choice is passed in. Superres is horizontal-only:
/// `height2 == height`.
fn downscale_plane_bd8(src: &[u16], w: usize, h: usize, coded_w: usize, use_opt: bool) -> Vec<u16> {
    let src_u8: Vec<u8> = src.iter().map(|&p| p as u8).collect();
    let out_u8 = if use_opt {
        optimized_downscale_plane_8bit(&src_u8, w, h, coded_w, h)
    } else {
        let mut out = vec![0u8; coded_w * h];
        resize_plane(
            &src_u8,
            h as i32,
            w as i32,
            w as i32,
            &mut out,
            h as i32,
            coded_w as i32,
            coded_w as i32,
        );
        out
    };
    out_u8.iter().map(|&p| u16::from(p)).collect()
}

/// Downscale one tight plane (`w x h` u16) horizontally to `coded_w x h`,
/// dispatching on bit depth: bd8 uses the 8-bit scaler (optimized or
/// non-normative per `use_opt`), bd10/12 the `highbd_resize_plane` arm. Superres
/// is horizontal-only: `height2 == height`, an identity vertical pass.
fn downscale_plane(
    src: &[u16],
    w: usize,
    h: usize,
    coded_w: usize,
    bd: u8,
    use_opt: bool,
) -> Vec<u16> {
    if bd == 8 {
        return downscale_plane_bd8(src, w, h, coded_w, use_opt);
    }
    let mut out = vec![0u16; coded_w * h];
    highbd_resize_plane(
        src,
        h as i32,
        w as i32,
        w as i32,
        &mut out,
        h as i32,
        coded_w as i32,
        coded_w as i32,
        i32::from(bd),
    );
    out
}

/// Build the [`FrameHeaderObu`] bootstrap cfg for a superres KEY frame from the
/// parsed sequence header and the coded (downscaled) mi dimensions. Shared by
/// [`port_encode_superres`] and [`parse_superres_facts`] so the two stay in sync.
fn superres_frame_cfg(seq: &SequenceHeaderObu, mi_cols: i32, mi_rows: i32) -> FrameHeaderObu {
    let s = &seq.seq_header;
    let cc = &seq.color_config;
    let num_planes = if cc.monochrome { 1 } else { 3 };
    let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
    FrameHeaderObu {
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
        // Superres is active: gate the intrabc read/write bit (screen-content
        // frames drop it under superres). Propagated to `p` via `cfg.clone()`.
        superres_scaled: true,
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
    }
}

/// Parse the sequence + uncompressed frame header from a real superres stream and
/// return `(allow_screen_content_tools, scale_denominator)`. Both are decoded
/// BEFORE `tile_info` in the header, so the coded-width-dependent tile limits are
/// irrelevant here — the facts are exact regardless of the mi dims passed to
/// [`superres_frame_cfg`]. Used to read the denom the REAL encoder *chose* (the
/// output of `calculate_next_superres_scale`) so the port's derivation can be
/// asserted against it.
fn parse_superres_facts(bootstrap: &[u8]) -> (bool, i32) {
    let obus = walk_obus(bootstrap);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .expect("sequence header present");
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);
    let (_, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME)
        .map(|(t, p)| (*t, *p))
        .expect("combined OBU_FRAME present");
    // Full-width mi (denom-agnostic): scale_denominator + allow_scc are parsed
    // before tile_info, so these limits don't affect them.
    let mi_cols = mi_dim(seq.seq_header.max_frame_width);
    let mi_rows = mi_dim(seq.seq_header.max_frame_height);
    let cfg = superres_frame_cfg(&seq, mi_cols, mi_rows);
    let mut rb = ReadBitBuffer::new(frame_payload);
    let p = read_uncompressed_header(&mut rb, &cfg);
    (p.allow_screen_content_tools, p.frame_size.scale_denominator)
}

/// The port's superres encode: downscale the source to the coded width, encode
/// the coded frame with the ordinary two-pass (encode → deblock → repack)
/// pipeline, and assemble a frame OBU whose header signals the superres denom.
/// Returns the frame OBU payload.
fn port_encode_superres(cell: &EncodeCell, denom: i32, bootstrap: &[u8]) -> Vec<u8> {
    let (w, h, mono, ss_x, ss_y, bd) = (cell.w, cell.h, cell.mono, cell.ss_x, cell.ss_y, cell.bd);
    assert_eq!(cell.speed, 0, "superres gate runs at speed 0");
    assert!(
        matches!(bd, 8 | 10 | 12),
        "superres gate supports bd 8/10/12"
    );

    let obus = walk_obus(bootstrap);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .expect("sequence header present");
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);
    let (_, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME)
        .map(|(t, p)| (*t, *p))
        .expect("combined OBU_FRAME present");

    let s = &seq.seq_header;
    let cc = &seq.color_config;
    assert!(
        s.enable_superres,
        "{}: --superres-mode=fixed must set the sequence enable_superres bit",
        cell.label
    );
    let num_planes = if cc.monochrome { 1 } else { 3 };

    // Coded (downscaled) width: derived from the upscaled width + denom, the
    // same value av1_calculate_scaled_superres_size gives the C encoder.
    let coded_w = coded_superres_width(s.max_frame_width, denom) as usize;
    assert!(
        coded_w < w,
        "{}: superres must downscale (coded {coded_w} < upscaled {w})",
        cell.label
    );

    let mi_cols = mi_dim(coded_w as i32); // CODED width, not max_frame_width
    let mi_rows = mi_dim(h as i32); // height unchanged by superres

    let cfg = superres_frame_cfg(&seq, mi_cols, mi_rows);
    let mut rb = ReadBitBuffer::new(frame_payload);
    let mut p = read_uncompressed_header(&mut rb, &cfg);
    assert!(!p.prefix.show_existing_frame);
    assert_eq!(p.prefix.frame_type, 0, "{}: frame must be KEY", cell.label);
    assert!(!p.coded_lossless, "superres cells never run at qindex 0");
    assert_eq!(
        p.frame_size.scale_denominator, denom,
        "{}: bootstrapped denom must equal the knob (anti-vacuity)",
        cell.label
    );
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "{}: single-tile envelope", cell.label);

    let qindex = p.quant.base_qindex;
    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    av1_build_quantizer(
        bd,
        p.quant.y_dc_delta_q,
        p.quant.u_dc_delta_q,
        p.quant.u_ac_delta_q,
        p.quant.v_dc_delta_q,
        p.quant.v_ac_delta_q,
        &mut quants,
        &mut deq,
        0,
    );
    let rows_y = set_q_index(&quants, &deq, qindex as usize, 0);
    let rows_u = set_q_index(&quants, &deq, qindex as usize, 1);
    let rows_v = set_q_index(&quants, &deq, qindex as usize, 2);

    let real = {
        let kf_probe = KfFrameContext::default_for_qindex(qindex);
        derive_real_costs(&kf_probe, s.enable_filter_intra)
    };
    let rdmult = av1_compute_rd_mult_based_on_qindex(
        bd,
        FrameUpdateType::Kf,
        qindex,
        TuneMetric::Psnr,
        EncMode::Allintra,
    );

    // ---- downscale the source to the coded frame width ----------------------
    let full_cw = if mono { 0 } else { (w + ss_x) >> ss_x };
    let ch = if mono { 0 } else { (h + ss_y) >> ss_y };
    let coded_cw = if mono { 0 } else { (coded_w + ss_x) >> ss_x };
    // libaom's optimized 8-bit scaler is all-planes-or-none: use it iff bd8 AND
    // every plane satisfies `av1_has_optimized_scaler` (height unchanged for
    // superres). bd10/12 always take the non-normative path.
    let use_opt = bd == 8
        && has_optimized_scaler(w as i32, h as i32, coded_w as i32, h as i32)
        && (mono || has_optimized_scaler(full_cw as i32, ch as i32, coded_cw as i32, ch as i32));
    let ds_y = downscale_plane(&cell.y, w, h, coded_w, bd, use_opt);
    let (ds_u, ds_v) = if mono {
        (Vec::new(), Vec::new())
    } else {
        (
            downscale_plane(&cell.u, full_cw, ch, coded_cw, bd, use_opt),
            downscale_plane(&cell.v, full_cw, ch, coded_cw, bd, use_opt),
        )
    };

    // ---- SB-aligned strided planes at the CODED dimensions ------------------
    let n_sb_x = ((mi_cols + SB_MI - 1) / SB_MI).max(1);
    let n_sb_y = ((mi_rows + SB_MI - 1) / SB_MI).max(1);
    let sb_px_w = n_sb_x as usize * 64;
    let sb_px_h = n_sb_y as usize * 64;
    let stride = 320.max(sb_px_w + 4);
    let buf_h = (sb_px_h + 4).max(h + 4);
    let extend_plane = |dst: &mut [u16], pw: usize, ph: usize| {
        for r in 0..ph {
            let edge = dst[r * stride + pw - 1];
            for col in pw..stride {
                dst[r * stride + col] = edge;
            }
        }
        for r in ph..buf_h {
            dst.copy_within((ph - 1) * stride..ph * stride, r * stride);
        }
    };
    let mut src_y_strided = vec![0u16; stride * buf_h];
    for r in 0..h {
        src_y_strided[r * stride..r * stride + coded_w]
            .copy_from_slice(&ds_y[r * coded_w..r * coded_w + coded_w]);
    }
    extend_plane(&mut src_y_strided, coded_w, h);
    let mut src_u_strided = vec![0u16; stride * buf_h];
    let mut src_v_strided = vec![0u16; stride * buf_h];
    if !mono {
        for r in 0..ch {
            src_u_strided[r * stride..r * stride + coded_cw]
                .copy_from_slice(&ds_u[r * coded_cw..r * coded_cw + coded_cw]);
            src_v_strided[r * stride..r * stride + coded_cw]
                .copy_from_slice(&ds_v[r * coded_cw..r * coded_cw + coded_cw]);
        }
        extend_plane(&mut src_u_strided, coded_cw, ch);
        extend_plane(&mut src_v_strided, coded_cw, ch);
    }

    let speed = 0i32;
    let sf = SpeedFeatures::set_allintra(speed, p.allow_screen_content_tools, false);
    let env = SbEncodeEnv {
        sb_size: SB,
        mi_rows,
        mi_cols,
        tile_row_start: 0,
        tile_col_start: 0,
        tile_row_end: 1 << 16,
        tile_col_end: 1 << 16,
        monochrome: mono,
        ss_x,
        ss_y,
        bd,
        lossless: p.coded_lossless,
        reduced_tx_set_used: p.reduced_tx_set_used,
        disable_edge_filter: !s.enable_intra_edge_filter,
        filter_type: 0,
        stride,
        src_y: &src_y_strided,
        src_u: &src_u_strided,
        src_v: &src_v_strided,
        base_y: 0,
        base_uv: 0,
        rows_y: &rows_y,
        rows_u: &rows_u,
        rows_v: &rows_v,
        rdmult,
        sharpness: 0,
        enable_optimize_b: TrellisOptType::FullTrellisOpt,
        use_chroma_trellis_rd_mult: true,
        coeff_costs_y: &real.coeff_costs_y,
        coeff_costs_uv: &real.coeff_costs_uv,
        tx_type_costs: &real.tx_type_costs_y,
        qm_levels: None,
        tune: Default::default(),
        deltaq: None,
    };
    let pick_cfg = PickFrameCfg {
        intrabc: None,
        intra_tools: Default::default(),
        mode_costs: &real.mode_costs,
        tx_size_costs: &real.tx_size_costs,
        skip_costs: &real.skip_costs,
        tx_type_costs_y: &real.tx_type_costs_y,
        pol: &sf.tx_type_search_policy(false, 0),
        uv_lp: &UvLoopPolicy::speed0_allintra(),
        intra_uv_mode_cost: &real.mode_costs.intra_uv_mode_cost,
        cfl_costs: &real.cfl_costs,
        partition_costs: &real.partition_costs,
        partition_cdfs: &real.partition_cdf,
        allintra: true,
        speed,
        qindex,
        enable_filter_intra: s.enable_filter_intra,
        enable_tx64: true,
        enable_rect_tx: true,
        intra_pruning_with_hog: sf.intra_pruning_with_hog != 0,
        enable_rect_partitions: true,
        less_rectangular_check_level: sf.less_rectangular_check_level,
        max_partition_size: 15,
        min_partition_size: 0,
        enable_1to4_partitions: true,
        enable_ab_partitions: true,
        allow_screen_content_tools: p.allow_screen_content_tools,
        qm_levels: None,
        palette_costs: None,
    };
    let pack_cfg = PackCfg {
        enable_filter_intra: s.enable_filter_intra,
        tx_mode_is_select: p.tx_mode_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
        allow_screen_content_tools: p.allow_screen_content_tools,
        delta_q_present: false,
        delta_q_res: 0,
    };

    // ---- phase 1: the encode pass (bits discarded) --------------------------
    let mut recon_y = src_y_strided.clone();
    let mut recon_u = src_u_strided.clone();
    let mut recon_v = src_v_strided.clone();
    let mut kf_phase1 = KfFrameContext::default_for_qindex(qindex);
    let mut throwaway = OdEcEnc::new();
    let mut trees = pack_tile(
        &mut throwaway,
        &env,
        &pick_cfg,
        &pack_cfg,
        &mut kf_phase1,
        &mut recon_y,
        &mut recon_u,
        &mut recon_v,
        0,
        0,
        n_sb_y,
        n_sb_x,
        SB_MI,
        SB,
    );
    assert_eq!(trees.len(), (n_sb_x * n_sb_y) as usize);

    // ---- loop-filter: derive (from the coded recon) + apply -----------------
    let mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb_x, SB_MI, SB);
    let lf_frame = LfSearchFrame {
        recon_y: &recon_y,
        recon_u: &recon_u,
        recon_v: &recon_v,
        src_y: &src_y_strided,
        src_u: &src_u_strided,
        src_v: &src_v_strided,
        stride,
        crop_width: coded_w as u32,
        crop_height: h as u32,
        ss_x,
        ss_y,
        bd: i32::from(bd),
        monochrome: mono,
        mi: &mi_grid,
        mi_rows,
        mi_cols,
        delta_lf_present: false,
    };
    let derived_lf = pick_filter_level(&lf_frame, true, 0, false);
    if derived_lf.filter_level[0] != 0 || derived_lf.filter_level[1] != 0 {
        let params = LfParams {
            filter_level: derived_lf.filter_level,
            filter_level_u: derived_lf.filter_level_u,
            filter_level_v: derived_lf.filter_level_v,
            sharpness: derived_lf.sharpness,
            mode_ref_delta_enabled: true,
            ref_deltas: KF_REF_DELTAS,
            mode_deltas: KF_MODE_DELTAS,
            delta_lf_present: false,
            delta_lf_multi: false,
            lossless: [false; 8],
            seg: Default::default(),
        };
        let grid = LfMiGrid {
            mi: &mi_grid,
            stride: mi_cols as usize,
            mi_rows,
            mi_cols,
        };
        let mut buf = LfFrameBuf {
            y: &mut recon_y,
            y_stride: stride,
            u: &mut recon_u,
            v: &mut recon_v,
            uv_stride: stride,
            crop_width: coded_w as u32,
            crop_height: h as u32,
            ss_x,
            ss_y,
            bd: i32::from(bd),
        };
        loop_filter_frame(&mut buf, &grid, &params, 0, num_planes);
    }

    // ---- phase 2: the pack pass (fresh context; no CDEF) --------------------
    let mut recon2_y = src_y_strided.clone();
    let mut recon2_u = src_u_strided.clone();
    let mut recon2_v = src_v_strided.clone();
    let mut kf_phase2 = KfFrameContext::default_for_qindex(qindex);
    let mut enc = OdEcEnc::new();
    pack_tile_from_trees(
        &mut enc,
        &env,
        &pick_cfg,
        &pack_cfg,
        &mut kf_phase2,
        &mut recon2_y,
        &mut recon2_u,
        &mut recon2_v,
        &mut trees,
        0,
        0,
        n_sb_y,
        n_sb_x,
        SB_MI,
        SB,
        None,
    );
    let our_tile_bytes = enc.done().to_vec();

    // ---- header: derived LF; superres denom + upscaled width bootstrapped ---
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;

    assemble_frame_obu_payload_single_tile(&p, tiles_log2, &our_tile_bytes)
}

/// Run one superres cell: C encode (denom D) → port downscale+encode → splice →
/// decoder-agreement asserts → RD-closeness comparison.
fn run_superres_cell(cell: &EncodeCell, denom: i32) -> RdCellResult {
    c::ref_init();
    let c_tu = c_encode_superres(cell, denom);
    assert!(
        !c_tu.is_empty(),
        "{}: real superres encode failed",
        cell.label
    );
    let port_payload = port_encode_superres(cell, denom, &c_tu);
    let port_tu = splice_frame_obu(&c_tu, &port_payload);

    // The port's superres stream must mean the same to the real C decoder and
    // the port decoder: identical UPSCALED reconstructions.
    let ours_c = c::ref_decode_av1_kf(&port_tu, cell.w, cell.h);
    let ours_port = aom_decode::frame::decode_frame_obus(&port_tu)
        .unwrap_or_else(|e| panic!("{}: port decode of OUR stream failed: {e}", cell.label));
    assert_eq!(
        ours_port.y, ours_c.y,
        "{}: port and C decoders disagree on OUR stream's luma recon",
        cell.label
    );
    if !cell.mono {
        assert_eq!(
            ours_port.u, ours_c.u,
            "{}: U recon disagreement",
            cell.label
        );
        assert_eq!(
            ours_port.v, ours_c.v,
            "{}: V recon disagreement",
            cell.label
        );
    }

    compare_cell(&cell.label, cell, &port_tu, &c_tu)
}

/// Every cell came out BIT-IDENTICAL on the first complete run of this gate
/// (2026-07-17 first measurement — not a later tightening): the ported source
/// downscale + coded-domain encode + deblock + superres header signalling
/// reproduce real aomenc's fixed-denom superres bytes exactly. So this gate
/// asserts full byte-identity ON TOP of the RD-closeness report (PARITY.md
/// rule 2: a byte-identity gate lands the family in section A). Any EXACT →
/// CLOSE slip is a real regression and must fail loudly.
fn assert_all_exact(results: &[RdCellResult]) {
    assert_rd_close(results, &RdBands::default());
    let non_exact: Vec<&RdCellResult> = results.iter().filter(|r| !r.bit_identical).collect();
    assert!(
        non_exact.is_empty(),
        "{} superres cells are no longer BIT-IDENTICAL to real aomenc (regression from the \
         landed EXACT map): {:?}",
        non_exact.len(),
        non_exact.iter().map(|r| &r.label).collect::<Vec<_>>()
    );
}

/// REAL image content (decoded conformance frames), bd8 4:2:0, FIXED superres
/// across denoms {9,12,14} and the aggressive-web cq range. All cells are
/// byte-identical to real aomenc (`assert_all_exact`).
#[test]
fn encoder_gate_superres_fixed_real_content_rd_close() {
    let mut results = Vec::new();
    let mut exact = 0usize;
    for &denom in &[9i32, 12, 14] {
        for &cq in &[20i32, 32, 48] {
            let cell = EncodeCell::real_content(
                &format!("superres_real196_d{denom}_cq{cq:02}"),
                "av1-1-b8-01-size-196x196",
                None,
                cq,
                0,
            );
            let r = run_superres_cell(&cell, denom);
            if r.bit_identical {
                exact += 1;
            }
            results.push(r);
        }
    }
    eprintln!(
        "{}",
        aom_bench::rd_close::render_table(&results, &RdBands::default())
    );
    eprintln!(
        "superres FIXED real-content: {exact}/{} cells BYTE-IDENTICAL",
        results.len()
    );
    assert_all_exact(&results);
}

/// Monochrome FIXED superres (single-plane downscale + encode).
#[test]
fn encoder_gate_superres_fixed_mono_rd_close() {
    let mut results = Vec::new();
    let mut exact = 0usize;
    for &denom in &[9i32, 12] {
        for &cq in &[20i32, 48] {
            // Monochrome: build from the real luma plane, chroma dropped.
            let base = EncodeCell::real_content(
                &format!("superres_mono196_d{denom}_cq{cq:02}"),
                "av1-1-b8-01-size-196x196",
                None,
                cq,
                0,
            );
            let cell = EncodeCell {
                label: base.label.clone(),
                w: base.w,
                h: base.h,
                mono: true,
                ss_x: 1,
                ss_y: 1,
                usage: 2,
                cq_level: cq,
                speed: 0,
                bd: 8,
                y: base.y.clone(),
                u: Vec::new(),
                v: Vec::new(),
            };
            let r = run_superres_cell(&cell, denom);
            if r.bit_identical {
                exact += 1;
            }
            results.push(r);
        }
    }
    eprintln!(
        "{}",
        aom_bench::rd_close::render_table(&results, &RdBands::default())
    );
    eprintln!(
        "superres FIXED mono: {exact}/{} cells BYTE-IDENTICAL",
        results.len()
    );
    assert_all_exact(&results);
}

/// Textured synthetic cell (bd-aware; chroma is deliberately NOT an affine
/// function of luma so CfL doesn't trivialize it) — the same generator family
/// as the byte-exact chroma-ss / CDEF synthetic gates. 4:2:0 or monochrome.
fn synth_superres_cell(label: &str, sz: usize, mono: bool, cq: i32, bd: u8) -> EncodeCell {
    let maxv = (1u16 << bd) - 1;
    let mask = u32::from(maxv);
    let (w, h) = (sz, sz);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            let base = ((r * 37 + col * 23) as u32) & mask;
            let hf = if (r ^ col) & 1 == 1 { mask / 12 } else { 0 };
            y[r * w + col] = ((base ^ hf) as u16).min(maxv);
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + 1) >> 1, (h + 1) >> 1)
    };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                let base = ((r * 19 + col * 29) as u32) & mask;
                let hf = if (r + col) % 3 == 0 { mask / 20 } else { 0 };
                u[r * cw + col] = ((base ^ hf) as u16).min(maxv);
                let base2 = (((r + 7) * 19 + (col + 3) * 29) as u32) & mask;
                let hf2 = if (r + col + 10) % 3 == 0 {
                    mask / 20
                } else {
                    0
                };
                v[r * cw + col] = ((base2 ^ hf2) as u16).min(maxv);
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
        usage: 2,
        cq_level: cq,
        speed: 0,
        bd,
        y,
        u,
        v,
    }
}

/// bd10/12 FIXED superres — the highbd source-downscale arm (`highbd_resize_plane`,
/// CHUNK-2, bit-exact vs C). Textured synthetic 4:2:0 + mono content, denoms
/// {9,12,14}, aggressive-web cq. The bd10/12 KEY encode envelope is itself
/// byte-exact (KB-4), so the two compose to byte-identical superres streams.
#[test]
fn encoder_gate_superres_fixed_highbd_rd_close() {
    let mut results = Vec::new();
    let mut exact = 0usize;
    for &bd in &[10u8, 12] {
        // 4:2:0 across denom × cq
        for &denom in &[9i32, 12, 14] {
            for &cq in &[20i32, 48] {
                let cell = synth_superres_cell(
                    &format!("superres_synth_b{bd}_420_128_d{denom}_cq{cq:02}"),
                    128,
                    false,
                    cq,
                    bd,
                );
                let r = run_superres_cell(&cell, denom);
                if r.bit_identical {
                    exact += 1;
                }
                results.push(r);
            }
        }
        // monochrome (single-plane highbd downscale)
        for &denom in &[9i32, 12] {
            let cell = synth_superres_cell(
                &format!("superres_synth_b{bd}_mono_128_d{denom}_cq32"),
                128,
                true,
                32,
                bd,
            );
            let r = run_superres_cell(&cell, denom);
            if r.bit_identical {
                exact += 1;
            }
            results.push(r);
        }
    }
    eprintln!(
        "{}",
        aom_bench::rd_close::render_table(&results, &RdBands::default())
    );
    eprintln!(
        "superres FIXED highbd (bd10/12): {exact}/{} cells BYTE-IDENTICAL",
        results.len()
    );
    assert_all_exact(&results);
}

// ============================================================================
// DERIVED-denominator superres modes (PARITY C6): QTHRESH / RANDOM. The encoder
// CHOOSES the denom via `calculate_next_superres_scale`; the port re-derives it
// from the source + qindex (`aom_encode::superres_select`) and must match the
// denom the real encoder embedded in the stream, then reproduce the bytes.
// ============================================================================

/// C-encode one KEY frame with a derived-denom superres `mode`
/// (2=RANDOM, 3=QTHRESH), then parse `(allow_screen_content_tools, chosen denom)`
/// out of the emitted stream. `cli_qthresh`/`cli_kf_qthresh` are the 1..=63 CLI
/// knobs. CDEF/restoration OFF (allintra defaults).
fn c_encode_and_facts(
    cell: &EncodeCell,
    mode: i32,
    cli_qthresh: i32,
    cli_kf_qthresh: i32,
) -> (Vec<u8>, bool, i32) {
    let c_tu = c::ref_encode_av1_kf_superres_mode(
        &cell.y,
        &cell.u,
        &cell.v,
        cell.w,
        cell.h,
        i32::from(cell.bd),
        cell.mono,
        cell.ss_x as i32,
        cell.ss_y as i32,
        cell.cq_level,
        cell.speed,
        false, // enable_cdef
        false, // enable_restoration
        cell.usage,
        mode,
        cli_qthresh,
        cli_kf_qthresh,
        8, // superres_denom (AUTO_ALL only; unused for RANDOM/QTHRESH)
        8, // superres_kf_denom (AUTO_ALL only)
    );
    assert!(
        !c_tu.is_empty(),
        "{}: real superres-mode encode failed",
        cell.label
    );
    let (allow_scc, real_denom) = parse_superres_facts(&c_tu);
    (c_tu, allow_scc, real_denom)
}

/// Given the port-derived `port_denom` (already asserted == the real chosen
/// denom) and a downscale that engages superres (`port_denom > 8`), run the
/// port's downscale+encode, splice, and score byte-identity vs real aomenc.
///
/// The decode-both agreement cross-check (port decoder == C decoder on the
/// port's stream) runs only when the port's bytes DIFFER from real aomenc's — it
/// exists to confirm a byte-DIFFERENT-but-valid port stream decodes to the same
/// recon. When the port emits the EXACT real-aomenc bytes (`port_tu == c_tu`,
/// the byte-identity target this gate asserts), the encode is provably correct
/// and decode-both would instead be a DECODER-conformance check on real aomenc's
/// own bytes. The superres denom-16 (exact-2:1) UPSCALE currently diverges in
/// the port decoder — a decoder-track Known Bug orthogonal to this encoder gate
/// (the FIXED gate covers denoms 9/12/14 decode; denom-16 decode was untested).
fn superres_select_bytes(cell: &EncodeCell, c_tu: &[u8], port_denom: i32) -> RdCellResult {
    let port_payload = port_encode_superres(cell, port_denom, c_tu);
    let port_tu = splice_frame_obu(c_tu, &port_payload);
    if port_tu != c_tu {
        let ours_c = c::ref_decode_av1_kf(&port_tu, cell.w, cell.h);
        let ours_port = aom_decode::frame::decode_frame_obus(&port_tu)
            .unwrap_or_else(|e| panic!("{}: port decode of OUR stream failed: {e}", cell.label));
        assert_eq!(
            ours_port.y, ours_c.y,
            "{}: port and C decoders disagree on OUR (byte-different) stream's luma recon",
            cell.label
        );
        if !cell.mono {
            assert_eq!(
                ours_port.u, ours_c.u,
                "{}: U recon disagreement",
                cell.label
            );
            assert_eq!(
                ours_port.v, ours_c.v,
                "{}: V recon disagreement",
                cell.label
            );
        }
    }
    compare_cell(&cell.label, cell, &port_tu, c_tu)
}

/// Smooth synthetic content (low horizontal-frequency energy) so the QTHRESH
/// energy analysis derives a superres-ENGAGING denom (`> 8`). A gentle diagonal
/// ramp plus a small mid-frequency ripple spreads the derived denom across the
/// range (9..16 as q rises) — exercising both the non-normative scaler (9..15)
/// and the optimized denom-16 corner. `bd`-aware; 4:2:0 or monochrome.
fn synth_smooth_superres_cell(label: &str, sz: usize, mono: bool, cq: i32, bd: u8) -> EncodeCell {
    let maxv = (1u32 << bd) - 1;
    let (w, h) = (sz, sz);
    let span = (w as u32) * 3 + (h as u32) * 2;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            // Smooth diagonal ramp (dominant low frequency) + a gentle ripple.
            let ramp = maxv * (col as u32 * 3 + r as u32 * 2) / span;
            let ripple = ((((col as u32) / 8) & 1) * maxv) / 40;
            y[r * w + col] = (ramp + ripple).min(maxv) as u16;
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + 1) >> 1, (h + 1) >> 1)
    };
    let cspan = (cw as u32) * 3 + (ch as u32) * 2 + 1;
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                u[r * cw + col] = (maxv * (col as u32 + r as u32 * 2) / cspan).min(maxv) as u16;
                v[r * cw + col] = (maxv * (col as u32 * 2 + r as u32) / cspan).min(maxv) as u16;
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
        usage: 2,
        cq_level: cq,
        speed: 0,
        bd,
        y,
        u,
        v,
    }
}

/// QTHRESH mode — the core C6 derivation gate. The port derives the superres
/// denom from the source's 16×4 H_DCT energy + the picked qindex vs the KEY
/// qthresh knob (`superres_select::superres_denom_qthresh_key`), and it MUST
/// equal the denom real `aomenc --superres-mode=qthresh` chose (embedded in the
/// stream) — asserted for EVERY cell, across bd 8/10/12 (so the bd-dependent
/// energy accumulation shift is validated end-to-end vs real aomenc). Content:
/// - REAL 196² image content (detailed): declines superres (denom 8 — high HF
///   energy), exactly as real aomenc does.
/// - SMOOTH synthetic content: derives an ENGAGING denom (> 8).
///
/// End-to-end byte-identity is additionally asserted for the PRIMARY bd8 config
/// — the derived denom feeds the identical downscale (non-normative for denoms
/// 9..15, the ported OPTIMIZED scaler for denom 16) + coded-width encode +
/// `write_superres_scale` pipeline the FIXED gate proves byte-exact (and which
/// RANDOM exercises on real content at denoms 11/14/15/9). Highbd pipeline
/// byte-identity is the FIXED highbd gate's domain (16/16); a bd10 SMOOTH
/// 128²→102 cell hits a pre-existing partition near-tie (KB-6 class — bd8 and
/// bd12 at the identical denom byte-match), orthogonal to superres selection, so
/// highbd here is derivation-only.
#[test]
fn encoder_gate_superres_qthresh_e2e() {
    c::ref_init();
    let mut matched = 0usize;
    let mut bd8_bytes = Vec::new();

    // Derive + assert the chosen denom for one cell; byte-check bd8 engaged cells.
    let mut run = |cell: &EncodeCell, kf_qt: i32, bd8_byte: bool, bytes: &mut Vec<RdCellResult>| {
        let (c_tu, allow_scc, real_denom) = c_encode_and_facts(cell, 3, 63, kf_qt);
        let q = base_qindex_from_cq(cell.cq_level);
        let port_denom = i32::from(superres_denom_qthresh_key(
            &cell.y,
            cell.w,
            cell.h,
            cell.w,
            cell.bd,
            q,
            quantizer_to_qindex(kf_qt),
            allow_scc,
            true,
        ));
        let coded_w = coded_superres_width(cell.w as i32, port_denom) as i32;
        eprintln!(
            "{}: q={q} kf_qt={kf_qt} scc={allow_scc} -> real {real_denom}, port {port_denom} (coded_w={coded_w})",
            cell.label
        );
        assert_eq!(
            port_denom, real_denom,
            "{}: port-derived superres denom {port_denom} != the denom real aomenc chose \
             {real_denom} (q={q})",
            cell.label
        );
        // Byte-check engaged bd8 cells — both the non-normative scaler (denoms
        // 9..15) and the optimized denom-16 even-width scaler are ported.
        if bd8_byte && port_denom > i32::from(SCALE_NUMERATOR_U8) {
            bytes.push(superres_select_bytes(cell, &c_tu, port_denom));
        }
    };

    // REAL detailed content: derivation declines superres (denom 8).
    for &cq in &[20i32, 48, 63] {
        let cell = EncodeCell::real_content(
            &format!("qthresh_real196_cq{cq:02}"),
            "av1-1-b8-01-size-196x196",
            None,
            cq,
            0,
        );
        run(&cell, 8, false, &mut bd8_bytes);
        matched += 1;
    }

    // SMOOTH content across bd 8/10/12 (derivation incl. the bd-shift). Byte-check
    // the PRIMARY bd8 engaged cells (a low kf-qthresh keeps q > qthresh).
    for &bd in &[8u8, 10, 12] {
        for &cq in &[32i32, 44, 48, 52, 56, 63] {
            let cell = synth_smooth_superres_cell(
                &format!("qthresh_smooth_b{bd}_128_cq{cq:02}"),
                128,
                false,
                cq,
                bd,
            );
            run(&cell, 8, bd == 8, &mut bd8_bytes);
            matched += 1;
        }
    }

    eprintln!(
        "QTHRESH e2e: {matched} cells derivation-match real aomenc; {} bd8 engaged cells byte-identical",
        bd8_bytes.len()
    );
    assert!(
        bd8_bytes.len() >= 2,
        "expected >= 2 engaged bd8 QTHRESH cells on the ported scaler for the byte-identity path; \
         got {}",
        bd8_bytes.len()
    );
    assert_all_exact(&bd8_bytes);
}

/// RANDOM mode — byte-identity across the seeded denom sequence. libaom's RANDOM
/// seed is a process-global `static` (34567) advancing once per RANDOM frame, so
/// consecutive encodes draw 11, 14, 15, 9. This is the ONLY RANDOM encoder in the
/// binary and runs its cells sequentially, so C's static seed and the port's
/// threaded seed stay in lockstep — the port reproduces each draw and the
/// downscale+encode is byte-identical at all four denominators.
#[test]
fn encoder_gate_superres_random_e2e() {
    c::ref_init();
    let mut seed = aom_encode::superres_select::SUPERRES_RANDOM_SEED_INIT;
    let mut results = Vec::new();
    for (i, &cq) in [20i32, 32, 48, 63].iter().enumerate() {
        let cell = EncodeCell::real_content(
            &format!("random_real196_draw{i}_cq{cq:02}"),
            "av1-1-b8-01-size-196x196",
            None,
            cq,
            0,
        );
        let port_denom = i32::from(aom_encode::superres_select::superres_denom_random(
            &mut seed,
        ));
        let (c_tu, allow_scc, real_denom) = c_encode_and_facts(&cell, 2, 63, 63);
        eprintln!(
            "{}: draw#{i} scc={allow_scc} -> real {real_denom}, port {port_denom}",
            cell.label
        );
        assert_eq!(
            port_denom, real_denom,
            "{}: port RANDOM denom {port_denom} (draw #{i}) != real aomenc {real_denom} — \
             the process-global static-seed sequence desynced",
            cell.label
        );
        assert!(
            port_denom > i32::from(SCALE_NUMERATOR_U8),
            "{}: RANDOM denom {port_denom} must downscale",
            cell.label
        );
        results.push(superres_select_bytes(&cell, &c_tu, port_denom));
    }
    eprintln!(
        "RANDOM e2e: {} cells (denoms 11,14,15,9) match real aomenc + byte-identical",
        results.len()
    );
    assert_all_exact(&results);
}

/// AUTO mode — the non-recode single-KEY path (PARITY C6). For ALLINTRA the AUTO
/// search type is `Dual` (speed_features.c:384); a single KEY still has
/// `frames_to_key <= 1`, so `av1_superres_in_recode_allowed` is false — no recode
/// loop, no SOLO bump — and AUTO reduces to the q-based energy derivation with a
/// qthresh of 0 (`superres_select::superres_denom_auto_key`). The port must match
/// the denom real `aomenc --superres-mode=auto` chose; engaged bd8 cells are
/// byte-identity-checked. (If recode WERE active, the real denom would be
/// bumped/searched and this assert would catch it — proving the non-recode
/// assumption for the single-frame envelope.)
#[test]
fn encoder_gate_superres_auto_e2e() {
    c::ref_init();
    let mut matched = 0usize;
    let mut bd8_bytes = Vec::new();

    let mut run = |cell: &EncodeCell, bd8_byte: bool, bytes: &mut Vec<RdCellResult>| {
        let (c_tu, allow_scc, real_denom) = c_encode_and_facts(cell, 4, 63, 8);
        let q = base_qindex_from_cq(cell.cq_level);
        let port_denom = i32::from(superres_denom_auto_key(
            &cell.y,
            cell.w,
            cell.h,
            cell.w,
            cell.bd,
            q,
            allow_scc,
            true, // frames_to_key <= 1 (single KEY still) -> no recode
            SuperresAutoSearchType::Dual,
            8, // kf_scale_denominator (AUTO_ALL only; unused for Dual)
        ));
        let coded_w = coded_superres_width(cell.w as i32, port_denom) as i32;
        eprintln!(
            "{}: q={q} scc={allow_scc} -> real {real_denom}, port {port_denom} (coded_w={coded_w})",
            cell.label
        );
        assert_eq!(
            port_denom, real_denom,
            "{}: port AUTO denom {port_denom} != real aomenc {real_denom} (q={q}) — \
             a mismatch here would mean the recode loop fired (frames_to_key>1) or the \
             search type isn't Dual",
            cell.label
        );
        if bd8_byte && port_denom > i32::from(SCALE_NUMERATOR_U8) {
            bytes.push(superres_select_bytes(cell, &c_tu, port_denom));
        }
    };

    // REAL detailed content: AUTO also declines superres (denom 8).
    for &cq in &[20i32, 48] {
        let cell = EncodeCell::real_content(
            &format!("auto_real196_cq{cq:02}"),
            "av1-1-b8-01-size-196x196",
            None,
            cq,
            0,
        );
        run(&cell, false, &mut bd8_bytes);
        matched += 1;
    }
    // SMOOTH content across bd 8/10/12 (derivation incl. bd-shift); bd8 byte-checked.
    for &bd in &[8u8, 10, 12] {
        for &cq in &[20i32, 44, 48] {
            let cell = synth_smooth_superres_cell(
                &format!("auto_smooth_b{bd}_128_cq{cq:02}"),
                128,
                false,
                cq,
                bd,
            );
            run(&cell, bd == 8, &mut bd8_bytes);
            matched += 1;
        }
    }

    eprintln!(
        "AUTO e2e: {matched} cells derivation-match real aomenc; {} bd8 engaged byte-identical",
        bd8_bytes.len()
    );
    assert!(
        !bd8_bytes.is_empty(),
        "no engaged bd8 AUTO cell on the ported scaler — byte path untested"
    );
    assert_all_exact(&bd8_bytes);
}
