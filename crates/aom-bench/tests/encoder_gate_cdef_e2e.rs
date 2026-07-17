//! CDEF-strength RD search RD-closeness gate (`--enable-cdef=1`, task #7) —
//! the first stills bulk-port family through the shared
//! [`aom_bench::rd_close`] harness (PARITY.md section B).
//!
//! CDEF is OFF by default in allintra; every byte-exact gate encodes with
//! `enable_cdef=false`, so this file is the only place the knob turns on and
//! the proven default envelope stays untouched.
//!
//! Port pipeline under test (the two-pass frame encode mirroring C's actual
//! architecture: encode pass → deblock → `av1_cdef_search` → pack pass):
//!  1. phase 1 = [`pack_tile`] into a THROWAWAY entropy coder (C's
//!     `encode_sb(.., OUTPUT_ENABLED, ..)` pass: recon + adapted tile ctx);
//!  2. loop-filter level derivation ([`pick_filter_level`]) + application
//!     ([`loop_filter_frame`], gated on nonzero luma levels — encoder.c:2887);
//!  3. [`av1_cdef_search`] over (source, deblocked recon) → damping / bits /
//!     strength set + per-64x64-unit indices — derived by the PORT, never
//!     copied from the C header (PARITY.md rule 4: no bootstrap leak of the
//!     feature under test);
//!  4. phase 2 = [`pack_tile_from_trees`] with a fresh frame context over the
//!     phase-1 trees, interleaving the `write_cdef` strength literals;
//!  5. frame header carries the derived LF + CDEF params; OBU assembly.
//!
//! Validation per cell (all hard-asserted):
//!  - the spliced port stream decodes through the REAL C decoder AND the port
//!    decoder to the IDENTICAL reconstruction (the port's CDEF syntax means
//!    to C exactly what it means to us);
//!  - [`compare_cell`] + [`assert_rd_close`] with the default bands
//!    (|size| <= 5%, zensim drop <= 0.5) — AND, because every cell measured
//!    BIT-IDENTICAL on the first complete run (2026-07-17), full
//!    byte-identity per cell (`assert_all_exact`): the family lands directly
//!    in PARITY section A.
//!
//! Content: REAL conformance-decoded frames (the KB-6 lesson: synthetic-only
//! gates miss real-statistics divergences) across the aggressive-web cq
//! range, plus synthetic mono / 4:4:4 / bd10 cells for the single-plane
//! search, no-subsampling chroma, and 16-bit MSE paths.

use aom_bench::EncodeCell;
use aom_bench::rd_close::{
    RdBands, RdCellResult, assert_rd_close, compare_cell, splice_frame_obu,
};
use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
use aom_encode::pack::{CdefPackState, PackCfg, pack_tile, pack_tile_from_trees};
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::pickcdef::{CdefSearchFrame, av1_cdef_search};
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
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

/// Real aomenc with the CDEF knob ON (`--enable-cdef=1`) — the reference
/// stream for every cell here. Same shim/config as `EncodeCell::c_encode`
/// otherwise.
fn c_encode_cdef(cell: &EncodeCell) -> Vec<u8> {
    c::ref_encode_av1_kf(
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
        true, // --enable-cdef=1: THE knob under test
        false,
        cell.usage,
        0,
        false,
    )
}

/// The port's CDEF-enabled two-pass encode (see module docs). Returns the
/// assembled frame OBU payload. Header FIELDS are bootstrapped from the C
/// stream exactly like the byte-exact gates (the documented caveat); the
/// CDEF params + per-unit strengths + LF levels are PORT-derived.
fn port_encode_cdef(cell: &EncodeCell, bootstrap: &[u8]) -> Vec<u8> {
    let (w, h, mono, ss_x, ss_y, bd) =
        (cell.w, cell.h, cell.mono, cell.ss_x, cell.ss_y, cell.bd);
    assert_eq!(cell.speed, 0, "CDEF gate cells run at speed 0 (FULL search)");
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
        s.enable_cdef,
        "{}: --enable-cdef=1 must set the sequence enable_cdef bit",
        cell.label
    );
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
    assert!(!p.coded_lossless, "CDEF cells never run at qindex 0");
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "{}: single-tile envelope", cell.label);
    let real_cdef = p.cdef.clone();

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

    // SB-aligned strided planes with replicate edge extension (identical to
    // the byte-exact harnesses — encoder_gate_chroma_ss_e2e.rs::run_case).
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
    };
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
        src_y_strided[r * stride..r * stride + w].copy_from_slice(&cell.y[r * w..r * w + w]);
    }
    extend_plane(&mut src_y_strided, w, h);
    let mut src_u_strided = vec![0u16; stride * buf_h];
    let mut src_v_strided = vec![0u16; stride * buf_h];
    if !mono {
        for r in 0..ch {
            src_u_strided[r * stride..r * stride + cw]
                .copy_from_slice(&cell.u[r * cw..r * cw + cw]);
            src_v_strided[r * stride..r * stride + cw]
                .copy_from_slice(&cell.v[r * cw..r * cw + cw]);
        }
        extend_plane(&mut src_u_strided, cw, ch);
        extend_plane(&mut src_v_strided, cw, ch);
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
    };
    let pick_cfg = PickFrameCfg {
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
    };
    let pack_cfg = PackCfg {
        enable_filter_intra: s.enable_filter_intra,
        tx_mode_is_select: p.tx_mode_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
        allow_screen_content_tools: p.allow_screen_content_tools,
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

    // ---- loop-filter: derive + apply (loopfilter_frame, encoder.c) ----------
    let mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb_x, SB_MI, SB);
    let lf_frame = LfSearchFrame {
        recon_y: &recon_y,
        recon_u: &recon_u,
        recon_v: &recon_v,
        src_y: &src_y_strided,
        src_u: &src_u_strided,
        src_v: &src_v_strided,
        stride,
        crop_width: w as u32,
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
        // The C apply gate (encoder.c:2887). All planes; per-plane zero
        // levels no-op inside, exactly like av1_loop_filter_frame.
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
            crop_width: w as u32,
            crop_height: h as u32,
            ss_x,
            ss_y,
            bd: i32::from(bd),
        };
        loop_filter_frame(&mut buf, &grid, &params, 0, num_planes);
    }

    // ---- CDEF search (speed 0 => CDEF_FULL_SEARCH) ---------------------------
    let cdef_res = av1_cdef_search(
        &CdefSearchFrame {
            recon_y: &recon_y,
            recon_u: &recon_u,
            recon_v: &recon_v,
            src_y: &src_y_strided,
            src_u: &src_u_strided,
            src_v: &src_v_strided,
            stride,
            mi: &mi_grid,
            mi_rows,
            mi_cols,
            ss_x,
            ss_y,
            monochrome: mono,
            bd,
            base_qindex: qindex,
            rdmult,
        },
        sf.cdef_pick_method,
    );

    // ---- phase 2: the pack pass (fresh context, CDEF literals live) ---------
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
        Some(CdefPackState {
            cdef_bits: cdef_res.cdef_bits as u32,
            unit_strength: cdef_res.unit_strength.clone(),
            nhfb: cdef_res.nhfb,
        }),
    );
    let our_tile_bytes = enc.done().to_vec();

    // ---- header: derived LF + derived CDEF (never the C stream's) -----------
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;
    p.cdef.cdef_damping = cdef_res.cdef_damping;
    p.cdef.cdef_bits = cdef_res.cdef_bits;
    p.cdef.nb_cdef_strengths = cdef_res.nb_cdef_strengths;
    p.cdef.cdef_strengths = cdef_res.cdef_strengths;
    p.cdef.cdef_uv_strengths = cdef_res.cdef_uv_strengths;

    let n = p.cdef.nb_cdef_strengths;
    let rn = real_cdef.nb_cdef_strengths;
    let params_match = p.cdef.cdef_damping == real_cdef.cdef_damping
        && p.cdef.cdef_bits == real_cdef.cdef_bits
        && n == rn
        && p.cdef.cdef_strengths[..n] == real_cdef.cdef_strengths[..rn.min(8)]
        && (mono || p.cdef.cdef_uv_strengths[..n] == real_cdef.cdef_uv_strengths[..rn.min(8)]);
    eprintln!(
        "{}: derived CDEF damping={} bits={} y={:?} uv={:?} | real damping={} bits={} y={:?} uv={:?} | params_match={}",
        cell.label,
        p.cdef.cdef_damping,
        p.cdef.cdef_bits,
        &p.cdef.cdef_strengths[..n],
        &p.cdef.cdef_uv_strengths[..if mono { 0 } else { n }],
        real_cdef.cdef_damping,
        real_cdef.cdef_bits,
        &real_cdef.cdef_strengths[..rn],
        &real_cdef.cdef_uv_strengths[..if mono { 0 } else { rn }],
        params_match,
    );

    assemble_frame_obu_payload_single_tile(&p, tiles_log2, &our_tile_bytes)
}

/// Run one CDEF cell: C encode (knob on) → port two-pass encode → splice →
/// decoder-agreement asserts → RD-closeness comparison.
fn run_cdef_cell(cell: &EncodeCell) -> RdCellResult {
    c::ref_init();
    let c_tu = c_encode_cdef(cell);
    assert!(!c_tu.is_empty(), "{}: real encode failed", cell.label);
    let port_payload = port_encode_cdef(cell, &c_tu);
    let port_tu = splice_frame_obu(&c_tu, &port_payload);

    // The port's CDEF signalling must mean to the REAL C decoder exactly what
    // it means to the port decoder: identical reconstructions + round-tripped
    // CDEF params. (compare_cell only uses the port decoder; this closes the
    // loop through C.)
    let ours_c = c::ref_decode_av1_kf(&port_tu, cell.w, cell.h);
    let ours_port = aom_decode::frame::decode_frame_obus(&port_tu)
        .unwrap_or_else(|e| panic!("{}: port decode of OUR stream failed: {e}", cell.label));
    assert_eq!(
        ours_port.y, ours_c.y,
        "{}: port and C decoders disagree on OUR stream's luma recon",
        cell.label
    );
    if !cell.mono {
        assert_eq!(ours_port.u, ours_c.u, "{}: U recon disagreement", cell.label);
        assert_eq!(ours_port.v, ours_c.v, "{}: V recon disagreement", cell.label);
    }

    compare_cell(&cell.label, cell, &port_tu, &c_tu)
}

/// Synthetic textured cell over the mono/4:4:4/bd10 axes (the same generator
/// family as the byte-exact chroma-ss gates; chroma deliberately NOT an
/// affine function of luma so CfL doesn't trivialize it).
fn synth_cell(label: &str, sz: usize, mono: bool, ss_x: usize, ss_y: usize, cq: i32, bd: u8) -> EncodeCell {
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
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
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
                let hf2 = if (r + col + 10) % 3 == 0 { mask / 20 } else { 0 };
                v[r * cw + col] = ((base2 ^ hf2) as u16).min(maxv);
            }
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x,
        ss_y,
        usage: 2,
        cq_level: cq,
        speed: 0,
        bd,
        y,
        u,
        v,
    }
}

/// Every cell came out BIT-IDENTICAL on the first complete run of this gate
/// (2026-07-17, first measurement — not a later tightening): the derived
/// damping/bits/strength sets, the per-unit literals, and the re-packed tile
/// bytes all reproduce real aomenc exactly. So this gate asserts full
/// byte-identity ON TOP of the RD-closeness report (PARITY.md rule 2: a
/// byte-identity gate moves the family to section A). Any future EXACT →
/// CLOSE slip is a real regression and must fail loudly.
fn assert_all_exact(results: &[RdCellResult]) {
    assert_rd_close(results, &RdBands::default());
    let non_exact: Vec<&RdCellResult> = results.iter().filter(|r| !r.bit_identical).collect();
    assert!(
        non_exact.is_empty(),
        "{} CDEF cells are no longer BIT-IDENTICAL to real aomenc (regression from the \
         landed EXACT map): {:?}",
        non_exact.len(),
        non_exact.iter().map(|r| &r.label).collect::<Vec<_>>()
    );
}

/// REAL image content (decoded conformance frames), bd8 4:2:0, across the
/// aggressive-web quality range — the primary CDEF validation axis. The
/// 196x196 cells derive cdef_bits=2 (four-strength joint sets) at low cq, so
/// the per-64x64-unit strength literals are genuinely coded and matched.
#[test]
fn encoder_gate_cdef_real_content_rd_close() {
    let mut results = Vec::new();
    for &cq in &[5i32, 12, 20, 32, 48, 63] {
        let cell = EncodeCell::real_content(
            &format!("cdef_real196_cq{cq:02}"),
            "av1-1-b8-01-size-196x196",
            None,
            cq,
            0,
        );
        results.push(run_cdef_cell(&cell));
    }
    for &cq in &[12i32, 32, 63] {
        let cell = EncodeCell::real_content(
            &format!("cdef_real64_cq{cq:02}"),
            "av1-1-b8-01-size-64x64",
            None,
            cq,
            0,
        );
        results.push(run_cdef_cell(&cell));
    }
    assert_all_exact(&results);
}

/// Synthetic cells covering the mono (single-plane search), 4:4:4
/// (no-subsampling chroma), 4:2:0, and bd10 (16-bit MSE path) axes.
#[test]
fn encoder_gate_cdef_synthetic_axes_rd_close() {
    let cells = [
        // mono carries ss (1,1) — profile 0; monochrome is invalid in the
        // 4:4:4 profile 1 the (0,0) sampling would select.
        synth_cell("cdef_mono128_cq12", 128, true, 1, 1, 12, 8),
        synth_cell("cdef_mono128_cq48", 128, true, 1, 1, 48, 8),
        synth_cell("cdef_444_128_cq32", 128, false, 0, 0, 32, 8),
        synth_cell("cdef_420_128_cq32", 128, false, 1, 1, 32, 8),
        synth_cell("cdef_bd10_420_128_cq32", 128, false, 1, 1, 32, 10),
    ];
    let results: Vec<RdCellResult> = cells.iter().map(run_cdef_cell).collect();
    assert_all_exact(&results);
}
