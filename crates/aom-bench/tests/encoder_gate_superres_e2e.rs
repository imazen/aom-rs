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
//! Scope of THIS gate: 8-bit, FIXED denom, denoms whose down-ratio is not a
//! 1/16 multiple (9..14 here) — those use the non-normative `av1_resize_plane`
//! the CHUNK-1 kernel ports. The 8-bit denom-16-even-width corner (libaom's
//! OPTIMIZED `av1_resize_and_extend_frame` scaler), the highbd downscale, and
//! AUTO/QTHRESH/RANDOM denom selection are documented follow-ups (PARITY C6).

use aom_bench::EncodeCell;
use aom_bench::rd_close::{RdBands, RdCellResult, assert_rd_close, compare_cell, splice_frame_obu};
use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
use aom_encode::pack::{PackCfg, pack_tile, pack_tile_from_trees};
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::resize::{coded_superres_width, resize_plane};
use aom_encode::speed_features::SpeedFeatures;
use aom_entropy::enc::OdEcEnc;
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::partition::KfFrameContext;
use aom_entropy::rb::ReadBitBuffer;
use aom_loopfilter::frame::{LfFrameBuf, LfMiGrid, LfParams, loop_filter_frame};
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
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

/// libaom `av1_has_optimized_scaler` restricted to the superres case (height
/// unchanged): true iff the 8-bit optimized scaler would be used instead of the
/// non-normative `av1_resize_plane`. Used to keep this gate on the ported path.
fn superres_uses_optimized_scaler_8bit(w: i32, coded_w: i32) -> bool {
    // Height is unchanged (dst_h == src_h), so its two conditions hold trivially.
    coded_w * 4 >= w && coded_w <= w * 16 && (16 * coded_w) % w == 0 && (16 * w) % coded_w == 0
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
/// `coded_w x h` via the ported non-normative resize. Superres is
/// horizontal-only: `height2 == height`, so the vertical pass is an identity
/// copy inside `resize_plane`.
fn downscale_plane_bd8(src: &[u16], w: usize, h: usize, coded_w: usize) -> Vec<u16> {
    let src_u8: Vec<u8> = src.iter().map(|&p| p as u8).collect();
    let mut out_u8 = vec![0u8; coded_w * h];
    resize_plane(
        &src_u8,
        h as i32,
        w as i32,
        w as i32,
        &mut out_u8,
        h as i32,
        coded_w as i32,
        coded_w as i32,
    );
    out_u8.iter().map(|&p| u16::from(p)).collect()
}

/// The port's superres encode: downscale the source to the coded width, encode
/// the coded frame with the ordinary two-pass (encode → deblock → repack)
/// pipeline, and assemble a frame OBU whose header signals the superres denom.
/// Returns the frame OBU payload.
fn port_encode_superres(cell: &EncodeCell, denom: i32, bootstrap: &[u8]) -> Vec<u8> {
    let (w, h, mono, ss_x, ss_y, bd) = (cell.w, cell.h, cell.mono, cell.ss_x, cell.ss_y, cell.bd);
    assert_eq!(cell.speed, 0, "superres gate runs at speed 0");
    assert_eq!(
        bd, 8,
        "this gate is 8-bit (highbd downscale is a follow-up)"
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
    let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };

    // Coded (downscaled) width: derived from the upscaled width + denom, the
    // same value av1_calculate_scaled_superres_size gives the C encoder.
    let coded_w = coded_superres_width(s.max_frame_width, denom) as usize;
    assert!(
        coded_w < w,
        "{}: superres must downscale (coded {coded_w} < upscaled {w})",
        cell.label
    );
    assert!(
        !superres_uses_optimized_scaler_8bit(w as i32, coded_w as i32),
        "{}: this cell hits the 8-bit optimized scaler (denom-16-even corner); \
         out of scope for the non-normative kernel gate",
        cell.label
    );

    let mi_cols = mi_dim(coded_w as i32); // CODED width, not max_frame_width
    let mi_rows = mi_dim(h as i32); // height unchanged by superres

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
    };
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
    let ds_y = downscale_plane_bd8(&cell.y, w, h, coded_w);
    let (ds_u, ds_v) = if mono {
        (Vec::new(), Vec::new())
    } else {
        (
            downscale_plane_bd8(&cell.u, full_cw, ch, coded_cw),
            downscale_plane_bd8(&cell.v, full_cw, ch, coded_cw),
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
