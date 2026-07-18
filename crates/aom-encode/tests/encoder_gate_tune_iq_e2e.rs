//! Full-frame ENCODE byte-match gates for the **tune=IQ / tune=SSIMULACRA2
//! family** (PARITY.md C4): the knob-isolated pieces of the `handle_tuning`
//! bundle (av1_cx_iface.c:1938), each validated against real aomenc driven
//! with the SAME explicit knobs (`ref_encode_av1_kf_tune` — tuning first,
//! explicit overrides after, aomenc CLI ordering).
//!
//! Structure mirrors `encoder_gate_chroma_ss_e2e.rs`'s `run_case_ext`: encode
//! the reference with real aomenc, bootstrap the frame-header FIELDS from that
//! parse (the documented Gate-3 caveat — qindex mapping, tile limits), run THIS
//! PORT's `pack_tile` over the identical source pixels with the tune knobs
//! threaded, derive the loop-filter level, assemble the OBU payload, and
//! compare byte-for-byte. The FEATURE UNDER TEST never flows through the
//! bootstrap (PARITY.md rule 4): QM levels, chroma delta-q values, and the
//! sharpness LF derivation are computed by the port and CROSS-CHECKED against
//! the real header (a wiring witness that fails loudly before any byte
//! comparison can).
//!
//! This file is OWNED by the C4 tune-family track. Every test asserts real
//! byte-identity — no `#[ignore]`, no weakened asserts, no graceful skips.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
use aom_encode::pack::pack_tile;
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::speed_features::SpeedFeatures;
use aom_encode::TuneKnobs;
use aom_entropy::enc::OdEcEnc;
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::partition::KfFrameContext;
use aom_entropy::rb::ReadBitBuffer;
use aom_quant::{
    Dequants, QuantTuning, Quants, av1_build_quantizer, av1_set_quantizer, set_q_index,
};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        assert!(hdr.obu_has_size_field, "shim always sets has_size_field");
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

/// The PORT-side tune configuration of one case — the piece(s) of the
/// `handle_tuning` bundle under test. Everything defaults to the stock
/// envelope; the C side must be driven with the matching `RefTuneKnobs`.
#[derive(Clone, Copy, Debug)]
struct PortTune {
    /// QM on with this `(qm_min, qm_max)`; levels derived per
    /// `av1_set_quantizer`'s formula selection (the `is_allintra` arm here —
    /// PSNR-family tuning) and cross-checked against the real header.
    qm: Option<(i32, i32)>,
    /// `--dist-metric=qm-psnr` (`TuneKnobs::use_qm_dist_metric`).
    use_qm_dist_metric: bool,
    /// The tuning arm: IQ/SSIMULACRA2 flip the rdmult weight
    /// (`av1_compute_rd_mult_based_on_qindex`), the trellis rshift-7
    /// (`TuneKnobs::iq_tuning`), and `av1_set_quantizer`'s chroma-delta/QM
    /// formula arms.
    tuning: QuantTuning,
    /// `--enable-chroma-deltaq` (`av1_set_quantizer` chroma delta-q arms).
    chroma_deltaq: bool,
    /// `--sharpness` 0..=7: `av1_build_quantizer` rounding bias + trellis
    /// `(8 - sharpness)` scaling + eob>=5 guard + LF sharpness_level.
    sharpness: i32,
    /// `--enable-adaptive-sharpness`: qindex-adaptive LF sharpness cap
    /// (picklpf.c:232-247).
    adaptive_sharpness: bool,
    /// `--deltaq-mode=6` (DELTA_Q_VARIANCE_BOOST): per-SB source-variance
    /// qindex modulation.
    deltaq_mode6: bool,
    /// `--deltaq-strength` percent (only read under `deltaq_mode6`).
    deltaq_strength: u32,
}

impl Default for PortTune {
    fn default() -> Self {
        PortTune {
            qm: None,
            use_qm_dist_metric: false,
            tuning: QuantTuning::Psnr,
            chroma_deltaq: false,
            sharpness: 0,
            adaptive_sharpness: false,
            deltaq_mode6: false,
            deltaq_strength: 100,
        }
    }
}

/// Result of one case: whether the port matched real aomenc byte-for-byte.
struct CaseResult {
    matched: bool,
}

/// Encode one case with the C side driven by `ref_encode_av1_kf_tune(knobs)`
/// and the port by the matching `PortTune`, then compare frame OBU payloads
/// byte-for-byte. See the module docs for the bootstrap discipline.
#[allow(clippy::too_many_arguments)]
fn run_tune_case(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    cq_level: i32,
    bd: u8,
    content: impl Fn(usize, usize) -> u16,
    u_content: impl Fn(usize, usize) -> u16,
    v_content: impl Fn(usize, usize) -> u16,
    knobs: &c::RefTuneKnobs,
    port: &PortTune,
) -> CaseResult {
    c::ref_init();
    let usage = 2u32; // ALLINTRA — the stills envelope
    let maxv = (1u16 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = content(r, col).min(maxv);
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
                u[r * cw + col] = u_content(r, col).min(maxv);
                v[r * cw + col] = v_content(r, col).min(maxv);
            }
        }
    }

    let bytes = c::ref_encode_av1_kf_tune(
        &y,
        &u,
        &v,
        w,
        h,
        i32::from(bd),
        mono,
        ss_x as i32,
        ss_y as i32,
        cq_level,
        0,
        usage,
        knobs,
    );
    assert!(!bytes.is_empty(), "real tune encode must produce a stream");

    let obus = walk_obus(&bytes);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .unwrap_or_else(|| panic!("no sequence-header OBU (bd{bd} w={w} h={h})"));
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);

    let (frame_obu_type, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME || *t == 3)
        .map(|(t, p)| (*t, *p))
        .unwrap_or_else(|| panic!("no frame OBU (bd{bd})"));
    assert_eq!(frame_obu_type, OBU_FRAME, "expected combined OBU_FRAME");

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
    assert_eq!(p.prefix.frame_type, 0, "frame_type must be KEY");
    assert!(
        p.quant.base_qindex > 0,
        "tune gates run lossy cells only (qindex > 0)"
    );
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "single-tile envelope: expected exactly 1 tile");

    let allintra = true;
    let fmt = if mono {
        "mono".to_string()
    } else {
        format!("ss={ss_x}{ss_y}")
    };
    let ctx = format!(
        "bd{bd} {w}x{h} {fmt} cq={cq_level} qindex={} {port:?}",
        p.quant.base_qindex
    );
    eprintln!("{ctx}");

    // ---- port pipeline, tune knobs threaded, header bootstrapped ----------
    let qindex = p.quant.base_qindex;

    // Wiring witness: the sharpness knob must round-trip through the real
    // header's LF sharpness_level derivation (picklpf.c:220-247) — checked
    // after the port derives its own LF below.

    // `av1_set_quantizer` (av1_quantize.c:878): the port DERIVES the chroma
    // delta-q values + QM levels from its own knobs (the qindex itself stays
    // bootstrapped — the documented Gate-3 caveat), then CROSS-CHECKS every
    // derived field against the real header (bootstrap-leak guard).
    let (qm_min, qm_max) = port.qm.unwrap_or((4, 10)); // allintra default range
    let settings = av1_set_quantizer(
        qm_min,
        qm_max,
        qindex,
        port.chroma_deltaq,
        /*is_allintra=*/ true,
        port.tuning,
        if mono { 1 } else { ss_x as i32 },
        if mono { 1 } else { ss_y as i32 },
        cc.separate_uv_delta_q,
        /*delta_q_present=*/ false,
    );
    assert_eq!(settings.base_qindex, qindex, "{ctx}: base qindex passthrough");
    if !mono {
        // Mono streams never code chroma deltas (num_planes == 1) — the C
        // values exist but are unobservable; skip the cross-check there.
        assert_eq!(
            (
                settings.u_dc_delta_q,
                settings.u_ac_delta_q,
                settings.v_dc_delta_q,
                settings.v_ac_delta_q
            ),
            (
                p.quant.u_dc_delta_q,
                p.quant.u_ac_delta_q,
                p.quant.v_dc_delta_q,
                p.quant.v_ac_delta_q
            ),
            "{ctx}: derived chroma delta-q must match the real header"
        );
    }
    let qm_levels = if port.qm.is_some() {
        assert!(
            p.quant.using_qmatrix,
            "{ctx}: harness asked for QM but the real stream did not signal using_qmatrix"
        );
        assert_eq!(
            [
                settings.qmatrix_level_y,
                settings.qmatrix_level_u,
                settings.qmatrix_level_v
            ],
            [
                p.quant.qmatrix_level_y,
                p.quant.qmatrix_level_u,
                p.quant.qmatrix_level_v
            ],
            "{ctx}: derived qmatrix_level_{{y,u,v}} must match the real header"
        );
        Some([
            settings.qmatrix_level_y as usize,
            settings.qmatrix_level_u as usize,
            settings.qmatrix_level_v as usize,
        ])
    } else {
        assert!(
            !p.quant.using_qmatrix,
            "{ctx}: real stream signals using_qmatrix but the harness did not request QM"
        );
        None
    };

    let mut quants = Quants::zeroed();
    let mut deq = Dequants::zeroed();
    // `av1_build_quantizer(..., sharpness)` from the DERIVED deltas — the
    // --sharpness rounding bias (`sharpness_adjustment`, av1_quantize.c:607).
    let (u_dc, u_ac, v_dc, v_ac) = if mono {
        (0, 0, 0, 0)
    } else {
        (
            settings.u_dc_delta_q,
            settings.u_ac_delta_q,
            settings.v_dc_delta_q,
            settings.v_ac_delta_q,
        )
    };
    av1_build_quantizer(
        bd,
        settings.y_dc_delta_q,
        u_dc,
        u_ac,
        v_dc,
        v_ac,
        &mut quants,
        &mut deq,
        port.sharpness,
    );
    let rows_y = set_q_index(&quants, &deq, qindex as usize, 0);
    let rows_u = set_q_index(&quants, &deq, qindex as usize, 1);
    let rows_v = set_q_index(&quants, &deq, qindex as usize, 2);

    let mut kf_write = KfFrameContext::default_for_qindex(qindex);
    let real = derive_real_costs(&kf_write, s.enable_filter_intra);
    // The rdmult tuning arm: IQ/SSIMULACRA2 share the SSIM weight
    // (av1_compute_rd_mult_based_on_qindex) — PSNR-family cells pass Psnr.
    let rd_tuning = match port.tuning {
        QuantTuning::Iq => TuneMetric::Iq,
        QuantTuning::Ssimulacra2 => TuneMetric::Ssimulacra2,
        QuantTuning::Psnr => TuneMetric::Psnr,
    };
    let rdmult = av1_compute_rd_mult_based_on_qindex(
        bd,
        FrameUpdateType::Kf,
        qindex,
        rd_tuning,
        EncMode::Allintra,
    );

    let n_sb_x = ((mi_cols + SB_MI - 1) / SB_MI).max(1);
    let n_sb_y = ((mi_rows + SB_MI - 1) / SB_MI).max(1);
    let sb_px_w = n_sb_x as usize * 64;
    let sb_px_h = n_sb_y as usize * 64;
    let stride = 320.max(sb_px_w + 4);
    let buf_h = (sb_px_h + 4).max(h + 4);
    let extend_plane = |dst: &mut [u16], pw: usize, ph: usize| {
        for r in 0..ph {
            let edge = dst[r * stride + pw - 1];
            for cx in pw..stride {
                dst[r * stride + cx] = edge;
            }
        }
        for r in ph..buf_h {
            dst.copy_within((ph - 1) * stride..ph * stride, r * stride);
        }
    };
    let mut src_y_strided = vec![0u16; stride * buf_h];
    for r in 0..h {
        src_y_strided[r * stride..r * stride + w].copy_from_slice(&y[r * w..r * w + w]);
    }
    extend_plane(&mut src_y_strided, w, h);
    let mut src_u_strided = vec![0u16; stride * buf_h];
    let mut src_v_strided = vec![0u16; stride * buf_h];
    if !mono {
        for r in 0..ch {
            src_u_strided[r * stride..r * stride + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
            src_v_strided[r * stride..r * stride + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
        }
        extend_plane(&mut src_u_strided, cw, ch);
        extend_plane(&mut src_v_strided, cw, ch);
    }

    // --deltaq-mode=6 (DELTA_Q_VARIANCE_BOOST): derive `delta_q_res` from the
    // base qindex (encodeframe.c:2297) and the frame `delta_q_present` flag
    // from a source-only precompute of every SB's adjusted qindex — exact
    // because the running `current_base_qindex` advances unconditionally on
    // KEY intra (SB-root skip_txfm is structurally 0), so the per-SB chain
    // depends only on source pixels. C zeroes the flag post-encode when no SB
    // used a nonzero delta (encodeframe.c:2450) and the searches are then
    // identical (adjusted == base everywhere), which the `None` arm
    // reproduces. Both derived values are CROSS-CHECKED against the real
    // header, then written into the port's header (bootstrap-leak-free).
    let (delta_q_present, delta_q_res) = if port.deltaq_mode6 {
        let res = aom_encode::allintra_vis::variance_boost_delta_q_res(qindex);
        let mut running = qindex;
        let mut used = false;
        for r in 0..n_sb_y {
            for c in 0..n_sb_x {
                let off = (r as usize * 64) * stride + c as usize * 64;
                let adj = aom_encode::allintra_vis::setup_delta_q_variance_boost(
                    &src_y_strided,
                    off,
                    stride,
                    bd,
                    qindex,
                    port.deltaq_strength,
                    res,
                    running,
                );
                used |= adj != qindex;
                running = adj;
            }
        }
        (used && qindex > 0, res)
    } else {
        (false, 0)
    };
    assert_eq!(
        p.delta_q.delta_q_present, delta_q_present,
        "{ctx}: derived delta_q_present must match the real header"
    );
    if delta_q_present {
        assert_eq!(
            p.delta_q.delta_q_res, delta_q_res,
            "{ctx}: derived delta_q_res must match the real header"
        );
        p.delta_q.delta_q_res = delta_q_res;
    }
    p.delta_q.delta_q_present = delta_q_present;

    let tune = TuneKnobs {
        use_qm_dist_metric: port.use_qm_dist_metric,
        iq_tuning: port.tuning != QuantTuning::Psnr,
    };
    let speed = 0i32;
    let sf = SpeedFeatures::set_allintra(speed, p.allow_screen_content_tools, false);
    let pol = sf
        .tx_type_search_policy(false, port.sharpness)
        .with_tune_knobs(tune);
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
        sharpness: port.sharpness,
        enable_optimize_b: if p.coded_lossless {
            TrellisOptType::NoTrellisOpt
        } else {
            TrellisOptType::FullTrellisOpt
        },
        use_chroma_trellis_rd_mult: allintra,
        coeff_costs_y: &real.coeff_costs_y,
        coeff_costs_uv: &real.coeff_costs_uv,
        tx_type_costs: &real.tx_type_costs_y,
        qm_levels,
        deltaq: delta_q_present.then_some(aom_encode::encode_sb::DeltaQFrameCtx {
            quants: &quants,
            deq: &deq,
            base_qindex: qindex,
            delta_q_res,
            deltaq_strength: port.deltaq_strength,
            perceptual_ai: None, // Variance Boost (mode 6), not Perceptual-AI
            perceptual_wavelet: None, // nor Perceptual wavelet (mode 2)
            sb_mi: 0,
        }),
        tune,
    };
    let pick_cfg = PickFrameCfg {
        mode_costs: &real.mode_costs,
        tx_size_costs: &real.tx_size_costs,
        skip_costs: &real.skip_costs,
        tx_type_costs_y: &real.tx_type_costs_y,
        pol: &pol,
        uv_lp: &UvLoopPolicy::speed0_allintra(),
        intra_uv_mode_cost: &real.mode_costs.intra_uv_mode_cost,
        cfl_costs: &real.cfl_costs,
        partition_costs: &real.partition_costs,
        partition_cdfs: &real.partition_cdf,
        palette_costs: None,
        intra_tools: Default::default(),
        allintra,
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
        qm_levels,
    };
    let pack_cfg = aom_encode::pack::PackCfg {
        enable_filter_intra: s.enable_filter_intra,
        tx_mode_is_select: p.tx_mode_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
        delta_q_present,
        delta_q_res,
        allow_screen_content_tools: p.allow_screen_content_tools,
    };

    let mut recon_y = src_y_strided.clone();
    let mut recon_u = src_u_strided.clone();
    let mut recon_v = src_v_strided.clone();

    let mut enc = OdEcEnc::new();
    let trees = pack_tile(
        &mut enc,
        &env,
        &pick_cfg,
        &pack_cfg,
        &mut kf_write,
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
    assert_eq!(
        trees.len(),
        (n_sb_x * n_sb_y) as usize,
        "{ctx}: pack_tile must walk every SB"
    );
    let our_tile_bytes = enc.done().to_vec();

    // Port-derived loop-filter level with the sharpness knob threaded
    // (picklpf.c: sharpness_level = algo_cfg.sharpness under ALLINTRA, then
    // the optional qindex-adaptive cap).
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
    let lf_sharpness = aom_encode::lf_search::frame_lf_sharpness(
        true,
        port.tuning != QuantTuning::Psnr,
        port.sharpness,
        port.adaptive_sharpness,
        qindex,
    );
    let derived_lf = pick_filter_level(&lf_frame, allintra, lf_sharpness, false);
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;
    // Wiring witness: the port's derived LF sharpness_level must equal the
    // real header's (the picklpf derivation — NOT bootstrapped), then land in
    // the written header.
    assert_eq!(
        derived_lf.sharpness, p.loopfilter.sharpness_level,
        "{ctx}: port LF sharpness_level must match the real header"
    );
    p.loopfilter.sharpness_level = derived_lf.sharpness;

    let our_payload = assemble_frame_obu_payload_single_tile(&p, tiles_log2, &our_tile_bytes);
    let matched = our_payload == frame_payload;
    if matched {
        eprintln!("{ctx}: TRUE END-TO-END BYTE MATCH");
    } else {
        let first_diff = our_payload
            .iter()
            .zip(frame_payload.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(our_payload.len().min(frame_payload.len()));
        eprintln!(
            "{ctx}: MISMATCH at byte {first_diff} (ours={:?} real={:?}); our_tile.len()={} \
             real_frame.len()={}",
            our_payload.get(first_diff),
            frame_payload.get(first_diff),
            our_tile_bytes.len(),
            frame_payload.len(),
        );
    }
    CaseResult { matched }
}

/// Textured luma: two-axis gradient XOR checkerboard HF (never all-skip).
fn tex_luma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        let base = ((r * 37 + cc * 23) as u32) & mask;
        let hf = if (r ^ cc) & 1 == 1 { mask / 12 } else { 0 };
        (base ^ hf) as u16
    }
}

/// Textured chroma, deliberately NOT an affine function of luma (defeats CfL
/// trivialization).
fn tex_chroma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        let base = ((r * 19 + cc * 29) as u32) & mask;
        let hf = if (r + cc) % 3 == 0 { mask / 20 } else { 0 };
        (base ^ hf) as u16
    }
}

fn report_and_assert(label: &str, results: &[(String, bool)]) {
    eprintln!("\n=== {label} results ===");
    for (name, ok) in results {
        eprintln!("  {name}: {}", if *ok { "MATCH" } else { "MISMATCH" });
    }
    let failed: Vec<&String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| n)
        .collect();
    assert!(
        failed.is_empty(),
        "{}/{} {label} cells diverged from real aomenc: {:?}",
        failed.len(),
        results.len(),
        failed
    );
}

/// **C4 piece 2 — `--dist-metric=qm-psnr`** (QM-PSNR distortion metric in the
/// tx search + trellis, tx_search.c:1150 / txb_rdopt.c:347): QM-on encodes
/// (`--enable-qm=1 --qm-min=2 --qm-max=10`, the tune=IQ/SSIMULACRA2 QM range)
/// with the QM-PSNR metric, vs real aomenc with the same knobs. Everything
/// else stock (deltaq/cdef off, PSNR-family tuning, so the `is_allintra` QM
/// formula arm is selected on both sides). Sizes 64/128/192 (1x1 / 2x2 / 3x3
/// SB incl. partial-SB frame edge) x cq {12, 32, 50}, mono + 4:2:0 + 4:4:4.
/// The anti-vacuous companion witness (`tune_shim_smoke` in aom-sys-ref)
/// proves the metric changes the C stream on this exact content.
#[test]
fn encoder_gate_qm_psnr_dist_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let knobs = c::RefTuneKnobs {
        dist_metric: c::AOM_DIST_METRIC_QM_PSNR,
        enable_qm: 1,
        qm_min: 2,
        qm_max: 10,
        deltaq_mode: 0,
        enable_cdef: 0,
        ..Default::default()
    };
    let port = PortTune {
        qm: Some((2, 10)),
        use_qm_dist_metric: true,
        ..Default::default()
    };
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(mono, ss_x, ss_y, tag) in &[
        (true, 1usize, 1usize, "mono"),
        (false, 1, 1, "420"),
        (false, 0, 0, "444"),
    ] {
        for &sz in &[64usize, 128, 192] {
            for &cq in &[12i32, 32, 50] {
                let res = run_tune_case(
                    sz, sz, mono, ss_x, ss_y, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                );
                results.push((format!("qm-psnr {tag} {sz}x{sz} cq{cq:>2}"), res.matched));
            }
        }
    }
    report_and_assert("qm-psnr dist metric", &results);
}

/// Anti-vacuous witnesses for the sharpness gates: on this content,
/// `--sharpness=7` must change the real C stream vs `--sharpness=0` (the
/// quantizer rounding bias + trellis scaling + LF sharpness must bite), and
/// `--enable-adaptive-sharpness=1` must change a `--sharpness=7` stream at a
/// qindex where the cap kicks in (base_qindex 200 -> cap 0).
#[test]
fn sharpness_witness_knobs_bite() {
    c::ref_init();
    let luma = tex_luma(0xff);
    let (w, h) = (64usize, 64);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = luma(r, col);
        }
    }
    let empty: Vec<u16> = Vec::new();
    let enc = |sharpness: i32, adaptive: i32, cq: i32| {
        c::ref_encode_av1_kf_tune(
            &y,
            &empty,
            &empty,
            w,
            h,
            8,
            true,
            1,
            1,
            cq,
            0,
            2,
            &c::RefTuneKnobs {
                sharpness,
                enable_adaptive_sharpness: adaptive,
                deltaq_mode: 0,
                enable_cdef: 0,
                ..Default::default()
            },
        )
    };
    let s0 = enc(0, -1, 32);
    let s7 = enc(7, -1, 32);
    assert_ne!(s0, s7, "--sharpness=7 must change the C bitstream at cq32");
    // cq50 -> qindex 200: adaptive cap = 0 (base_qindex > 160), so adaptive
    // must undo sharpness-7's LF sharpness (and only that — quantizer/trellis
    // keep the CLI value).
    let hi7 = enc(7, -1, 50);
    let hi7_adaptive = enc(7, 1, 50);
    assert_ne!(
        hi7, hi7_adaptive,
        "--enable-adaptive-sharpness must change a sharpness-7 stream at qindex 200"
    );
}

/// **C4 piece 3 — `--sharpness` e2e** (the tune bundle installs `sharpness=7`):
/// the `av1_build_quantizer` rounding bias (`sharpness_adjustment`,
/// av1_quantize.c:607-621), the trellis `(8 - sharpness)` rdmult scaling +
/// eob>=5 / no-lower-qc guards (txb_rdopt.c), and the LF `sharpness_level`
/// derivation (picklpf.c:220-230, ALLINTRA arm) — all threaded through one
/// knob and byte-matched vs real `aomenc --sharpness=N`. Sharpness 1/3/7
/// (7 = the tune=IQ value; 3 additionally exercises C's GOOD-only sf arms,
/// which must stay inert in allintra), mono + 4:2:0, 64² + 128², cq 12/32/50.
#[test]
fn encoder_gate_sharpness_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &sharp in &[1i32, 3, 7] {
        let knobs = c::RefTuneKnobs {
            sharpness: sharp,
            deltaq_mode: 0,
            enable_cdef: 0,
            ..Default::default()
        };
        let port = PortTune {
            sharpness: sharp,
            ..Default::default()
        };
        for &(mono, tag) in &[(true, "mono"), (false, "420")] {
            for &sz in &[64usize, 128] {
                for &cq in &[12i32, 32, 50] {
                    let res = run_tune_case(
                        sz, sz, mono, 1, 1, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                    );
                    results.push((format!("sharp{sharp} {tag} {sz}x{sz} cq{cq:>2}"), res.matched));
                }
            }
        }
    }
    report_and_assert("--sharpness e2e", &results);
}

/// **C4 piece 5 — `--enable-adaptive-sharpness`** (tune=IQ additionally
/// installs it): the qindex-adaptive LF sharpness cap (picklpf.c:232-247,
/// `frame_lf_sharpness`) on top of `--sharpness=7`. cq12 -> qindex 48 (cap 7,
/// sharpness kept), cq32 -> qindex 128 (cap 1), cq50 -> qindex 200 (cap 0) —
/// all three cap arms exercised; the quantizer/trellis keep the CLI
/// sharpness (only the LF level is capped), which the byte match proves.
#[test]
fn encoder_gate_adaptive_sharpness_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let knobs = c::RefTuneKnobs {
        sharpness: 7,
        enable_adaptive_sharpness: 1,
        deltaq_mode: 0,
        enable_cdef: 0,
        ..Default::default()
    };
    let port = PortTune {
        sharpness: 7,
        adaptive_sharpness: true,
        ..Default::default()
    };
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(mono, tag) in &[(true, "mono"), (false, "420")] {
        for &sz in &[64usize, 128] {
            for &cq in &[12i32, 32, 50] {
                let res = run_tune_case(
                    sz, sz, mono, 1, 1, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                );
                results.push((format!("adaptive7 {tag} {sz}x{sz} cq{cq:>2}"), res.matched));
            }
        }
    }
    report_and_assert("--enable-adaptive-sharpness e2e", &results);
}

/// **C4 piece 4 — `--enable-chroma-deltaq`** (the tune bundle installs it):
/// the frame chroma delta-q derivation (`av1_set_quantizer`,
/// av1_quantize.c:886-966) byte-matched vs real aomenc. Two arms:
/// - PSNR-family tuning: the constant `2/2/2/2` arm (4:2:0 + 4:4:4).
/// - tune=IQ / tune=SSIMULACRA2 with every OTHER bundle piece explicitly
///   overridden OFF (qm/dist-metric/sharpness/deltaq6/cdef — aomenc CLI
///   ordering lets explicit knobs override the bundle): the empirically
///   derived ramps — 4:2:0 `-clamp(base/2-14, 0, 16|20)` (IQ|SSIMULACRA2),
///   4:2:2 `+clamp(base/2, 0, 6)` AC, 4:4:4 `+clamp(base/2, 0, 24)` AC.
///   These cells also run the IQ/SSIM2 rdmult weight + trellis rshift-7 arms
///   LIVE e2e for the first time (both sides), and the port derives the
///   deltas itself (cross-checked against the real header before comparing).
#[test]
fn encoder_gate_chroma_deltaq_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();

    // Arm 1: PSNR-family (no tune) — the 2/2/2/2 arm.
    let knobs = c::RefTuneKnobs {
        enable_chroma_deltaq: 1,
        deltaq_mode: 0,
        enable_cdef: 0,
        ..Default::default()
    };
    let port = PortTune {
        chroma_deltaq: true,
        ..Default::default()
    };
    for &(ss_x, ss_y, tag) in &[(1usize, 1usize, "420"), (0, 0, "444")] {
        for &sz in &[64usize, 128] {
            for &cq in &[12i32, 32, 50] {
                let res = run_tune_case(
                    sz, sz, false, ss_x, ss_y, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                );
                results.push((format!("cdq-psnr {tag} {sz}x{sz} cq{cq:>2}"), res.matched));
            }
        }
    }

    // Arm 2: tune=IQ / tune=SSIMULACRA2 ramps, bundle otherwise disarmed.
    for &(tuning, ctune, ttag) in &[
        (QuantTuning::Iq, c::AOM_TUNE_IQ, "iq"),
        (QuantTuning::Ssimulacra2, c::AOM_TUNE_SSIMULACRA2, "ssim2"),
    ] {
        let knobs = c::RefTuneKnobs {
            tuning: ctune,
            sharpness: 0,
            enable_adaptive_sharpness: 0,
            dist_metric: c::AOM_DIST_METRIC_PSNR,
            deltaq_mode: 0,
            enable_qm: 0,
            enable_cdef: 0,
            // chroma deltaq stays at the bundle's 1
            ..Default::default()
        };
        let port = PortTune {
            tuning,
            chroma_deltaq: true,
            ..Default::default()
        };
        for &(ss_x, ss_y, tag) in &[
            (1usize, 1usize, "420"),
            (1, 0, "422"),
            (0, 0, "444"),
        ] {
            for &cq in &[12i32, 32, 50] {
                let res = run_tune_case(
                    64, 64, false, ss_x, ss_y, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                );
                results.push((format!("cdq-{ttag} {tag} 64x64 cq{cq:>2}"), res.matched));
            }
        }
    }
    report_and_assert("--enable-chroma-deltaq e2e", &results);
}

/// **C4 pieces 1+2+4 composite — the tune=IQ / tune=SSIMULACRA2 QM stack**:
/// tune installed with only deltaq6/cdef/sharpness disarmed, keeping the
/// bundle's `enable_qm=1 qm 2..10` + `dist_metric=QM_PSNR` +
/// `enable_chroma_deltaq=1`. This runs the tune QM-level formulas LIVE:
/// SSIMULACRA2 luma (`aom_get_qmlevel_luma_ssimulacra2`), the 4:4:4 chroma
/// formula (`aom_get_qmlevel_444_chroma`), and the allintra formulas at the
/// chroma-delta-SHIFTED qindexes — plus the QM-PSNR metric and the IQ rdmult
/// / trellis-rshift arms, all interacting. 4:2:0 + 4:4:4 × both tunes.
#[test]
fn encoder_gate_tune_qm_stack_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(tuning, ctune, ttag) in &[
        (QuantTuning::Iq, c::AOM_TUNE_IQ, "iq"),
        (QuantTuning::Ssimulacra2, c::AOM_TUNE_SSIMULACRA2, "ssim2"),
    ] {
        let knobs = c::RefTuneKnobs {
            tuning: ctune,
            sharpness: 0,
            enable_adaptive_sharpness: 0,
            deltaq_mode: 0,
            enable_cdef: 0,
            // bundle keeps: enable_qm=1 qm 2..10, dist=QM_PSNR, chroma deltaq
            ..Default::default()
        };
        let port = PortTune {
            qm: Some((2, 10)),
            use_qm_dist_metric: true,
            tuning,
            chroma_deltaq: true,
            ..Default::default()
        };
        for &(ss_x, ss_y, tag) in &[(1usize, 1usize, "420"), (0, 0, "444")] {
            for &sz in &[64usize, 128] {
                for &cq in &[12i32, 32, 50] {
                    let res = run_tune_case(
                        sz, sz, false, ss_x, ss_y, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                    );
                    results.push((format!("qmstack-{ttag} {tag} {sz}x{sz} cq{cq:>2}"), res.matched));
                }
            }
        }
    }
    report_and_assert("tune QM stack (formulas+metric+chroma-deltaq)", &results);
}

/// Mixed-variance luma for the Variance Boost gates: quadrant-structured 64px
/// tiles alternating near-flat gradients (low variance -> strong boost) and
/// dense texture (high variance -> no boost), so a multi-SB frame codes
/// genuinely DIFFERENT per-SB delta-q values.
fn mixed_variance_luma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        let flat_tile = ((r / 64) + (cc / 64)) % 2 == 0;
        if flat_tile {
            // gentle gradient: sub-block variance near zero
            (((r + cc) as u32 * mask / 640) & mask) as u16
        } else {
            let base = ((r * 37 + cc * 23) as u32) & mask;
            let hf = if (r ^ cc) & 1 == 1 { mask / 8 } else { 0 };
            (base ^ hf) as u16
        }
    }
}

/// Anti-vacuous witness for the Variance Boost gate: `--deltaq-mode=6` must
/// change the real C stream vs `--deltaq-mode=0` on mixed-variance content
/// (per-SB delta-q symbols + per-SB quantization must bite), and
/// `--deltaq-strength=200` must differ from the default 100.
#[test]
fn variance_boost_witness_knobs_bite() {
    c::ref_init();
    let luma = mixed_variance_luma(0xff);
    let (w, h) = (128usize, 128);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = luma(r, col);
        }
    }
    let empty: Vec<u16> = Vec::new();
    let enc = |deltaq_mode: i32, strength: i32| {
        c::ref_encode_av1_kf_tune(
            &y,
            &empty,
            &empty,
            w,
            h,
            8,
            true,
            1,
            1,
            32,
            0,
            2,
            &c::RefTuneKnobs {
                deltaq_mode,
                deltaq_strength: strength,
                enable_cdef: 0,
                ..Default::default()
            },
        )
    };
    let off = enc(0, -1);
    let vb = enc(6, -1);
    assert_ne!(off, vb, "--deltaq-mode=6 must change the C bitstream");
    let vb200 = enc(6, 200);
    assert_ne!(vb, vb200, "--deltaq-strength=200 must change a deltaq-6 stream");
}

/// **C4 piece 6 — `--deltaq-mode=6` DELTA_Q_VARIANCE_BOOST** (the tune
/// bundle's deltaq mode): per-SB source-variance qindex modulation —
/// `av1_get_variance_boost_block_variance` (octile-sampled 8x8 variances) →
/// `av1_get_sbq_variance_boost` (the still-picture boost curve) →
/// `av1_adjust_q_from_delta_q_res` against the running base — with per-SB
/// quantizer-row re-selection + rdmult recompute in the search, the per-SB
/// `delta_q` symbol in the pack, and the derived `delta_q_present`/`res`
/// header fields. Mixed-variance multi-SB content (2x2 / 3x3 SBs) so deltas
/// genuinely vary; cq 12/32/50 crosses the delta_q_res 1/2/8 arms; strength
/// 100 (default) + 200; mono + 4:2:0. Byte-matched vs real
/// `aomenc --deltaq-mode=6`.
#[test]
fn encoder_gate_variance_boost_deltaq_e2e() {
    let luma = mixed_variance_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(strength, cstrength, stag) in &[(100u32, -1i32, "s100"), (200, 200, "s200")] {
        let knobs = c::RefTuneKnobs {
            deltaq_mode: 6,
            deltaq_strength: cstrength,
            enable_cdef: 0,
            ..Default::default()
        };
        let port = PortTune {
            deltaq_mode6: true,
            deltaq_strength: strength,
            ..Default::default()
        };
        for &(mono, tag) in &[(true, "mono"), (false, "420")] {
            for &sz in &[128usize, 192] {
                for &cq in &[12i32, 32, 50] {
                    let res = run_tune_case(
                        sz, sz, mono, 1, 1, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                    );
                    results.push((format!(
                        "vboost-{stag} {tag} {sz}x{sz} cq{cq:>2}"
                    ), res.matched));
                }
            }
        }
    }
    report_and_assert("--deltaq-mode=6 variance boost", &results);
}

/// The **full `--tune=iq` / `--tune=ssimulacra2` bundle**, every knob live at
/// once (`handle_tuning`, av1_cx_iface.c:1938-1978): QM 2..10 + QM-PSNR dist
/// metric + `--sharpness=7` + `--enable-chroma-deltaq` + Variance-Boost
/// `--deltaq-mode=6` (+ `--enable-adaptive-sharpness` for IQ). CDEF is the ONE
/// bundle member overridden off (`enable_cdef=0`): it is a separate,
/// already-bit-exact track (STATUS.md #7 `av1_cdef_search`) and is applied
/// post-reconstruction, so it never touches the coded tile symbols — the tune
/// family port deliberately does not own it. This gate is the real product
/// config the whole C4 family exists for: it proves the knobs *compose*, not
/// merely that each works in isolation (the six per-piece gates above). The
/// harness's per-field cross-checks (chroma delta-q, QM levels, LF sharpness,
/// delta-q res/present) fire before the byte compare if any knob's port
/// derivation disagrees with the real header.
#[test]
fn encoder_gate_tune_composite_full_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(qtune, ctune, ttag, adaptive) in &[
        (QuantTuning::Iq, c::AOM_TUNE_IQ, "iq", true),
        (QuantTuning::Ssimulacra2, c::AOM_TUNE_SSIMULACRA2, "ssim2", false),
    ] {
        // Install the WHOLE handle_tuning bundle (AOME_SET_TUNING FIRST),
        // override ONLY CDEF off (port models no CDEF; CDEF is symbol-inert).
        let knobs = c::RefTuneKnobs {
            tuning: ctune,
            enable_cdef: 0,
            ..Default::default()
        };
        // The port mirror: every bundle knob (adaptive sharpness IQ-only).
        let port = PortTune {
            qm: Some((2, 10)),
            use_qm_dist_metric: true,
            tuning: qtune,
            chroma_deltaq: true,
            sharpness: 7,
            adaptive_sharpness: adaptive,
            deltaq_mode6: true,
            deltaq_strength: 100,
        };
        // mono + 420 + 444 × single-SB and multi-SB (192 = 3×3 sb64, exercises
        // per-SB Variance-Boost qindex variation) × the low..high-q web regime.
        for &(mono, ss_x, ss_y, tag) in &[
            (true, 1usize, 1usize, "mono"),
            (false, 1, 1, "420"),
            (false, 0, 0, "444"),
        ] {
            for &sz in &[64usize, 128, 192] {
                for &cq in &[12i32, 32, 50] {
                    let res = run_tune_case(
                        sz, sz, mono, ss_x, ss_y, cq, 8, &luma, &chroma, &chroma, &knobs, &port,
                    );
                    results.push((
                        format!("composite-{ttag} {tag} {sz}x{sz} cq{cq:>2}"),
                        res.matched,
                    ));
                }
            }
        }
    }
    report_and_assert("full tune=IQ/SSIMULACRA2 bundle (cdef off)", &results);
}
