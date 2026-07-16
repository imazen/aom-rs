//! High-bit-depth (bd10 / bd12) full-frame ENCODE gate — the port's own
//! search+pack pipeline (`pack_tile` + `assemble_frame_obu_payload_single_tile`)
//! must byte-match real aomenc at bit depth 10 and 12, for the same 10/12-bit
//! source. The pre-existing e2e harness (`encoder_gate_e2e_byte_match.rs`) is
//! bd8-only, so the entire high-bit-depth encode path was previously unvalidated
//! end-to-end (a primary use case: zenavif/avifenc encode 10-bit stills).
//!
//! Same structure as the bd8 gate: encode the reference with real aomenc
//! (`ref_encode_av1_kf` at `bd`), bootstrap the frame header from that parse,
//! run THIS PORT's pack over the identical source pixels, assemble, and compare
//! byte-for-byte. Starts MONOCHROME (removes the chroma path entirely — the
//! simplest RDO landscape, and avoids the encoder track's active chroma work).
//! The block-level `xform_quant` is already bd12-validated (`xform_quant_diff`);
//! this locks the *frame-level* highbd path (intra predict / recon / pack / LF).
//!
//! This test file is OWNED by the bd10 track (#28); it does not touch the
//! encoder track's `encoder_gate_e2e_byte_match.rs`.

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
use aom_entropy::enc::OdEcEnc;
use aom_entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
};
use aom_entropy::obu::read_obu_header;
use aom_entropy::partition::KfFrameContext;
use aom_entropy::rb::ReadBitBuffer;
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
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

/// Result of one bd case: whether the port matched C byte-for-byte, and the two
/// full assembled OBU_FRAME payloads (ours + the real reference).
struct CaseResult {
    matched: bool,
    real_payload: Vec<u8>,
}

/// Encode one case at bit depth `bd` with `content(row,col) -> u16` luma and
/// `uv_content(row,col) -> u16` chroma (values in `[0, (1<<bd)-1]`; chroma
/// ignored when `mono`), bootstrap the header from real aomenc, run this port's
/// `pack_tile`, assemble, and compare byte-for-byte. Mirrors the bd8 harness's
/// `attempt_case_content_uv`, parameterized on `bd`. `mono=true` removes the
/// chroma path entirely; `mono=false` with `ss_x=ss_y=1` is 4:2:0.
#[allow(clippy::too_many_arguments)]
fn run_case(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    bd: u8,
    content: impl Fn(usize, usize) -> u16,
    uv_content: impl Fn(usize, usize) -> u16,
) -> CaseResult {
    c::ref_init();
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
                let val = uv_content(r, col).min(maxv);
                u[r * cw + col] = val;
                v[r * cw + col] = val;
            }
        }
    }

    let bytes = c::ref_encode_av1_kf(
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
        false,
        false,
        usage,
        0,
        false,
    );
    assert!(
        !bytes.is_empty(),
        "ref_encode_av1_kf (bd{bd}) must produce a real stream"
    );

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
    assert_eq!(
        frame_obu_type, OBU_FRAME,
        "expected combined OBU_FRAME (bd{bd})"
    );

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
    let p = read_uncompressed_header(&mut rb, &cfg);
    let real_bit_len = rb.bit_position();
    assert!(!p.prefix.show_existing_frame);
    assert_eq!(p.prefix.frame_type, 0, "frame_type must be KEY");
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "single-tile envelope only");
    let allintra = usage == 2;
    let fmt = if mono {
        "mono".to_string()
    } else {
        format!("4:2:0(ss={ss_x},{ss_y})")
    };
    let ctx = format!(
        "bd{bd} w={w} h={h} {fmt} usage={usage} cq={cq_level} qindex={} lossless={}",
        p.quant.base_qindex, p.coded_lossless
    );
    eprintln!("{ctx}");

    // ---- port pipeline, header bootstrapped, coeffs/modes/partitions derived --
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

    let mut kf_write = KfFrameContext::default_for_qindex(qindex);
    let real = derive_real_costs(&kf_write, s.enable_filter_intra);
    let rdmult = av1_compute_rd_mult_based_on_qindex(
        bd,
        FrameUpdateType::Kf,
        qindex,
        TuneMetric::Psnr,
        if allintra {
            EncMode::Allintra
        } else {
            EncMode::Good
        },
    );

    let stride = 320.max(w + 4);
    let mut src_y_strided = vec![0u16; stride * (h + 4)];
    for r in 0..h {
        src_y_strided[r * stride..r * stride + w].copy_from_slice(&y[r * w..r * w + w]);
    }
    let mut src_u_strided = vec![0u16; stride * (h + 4)];
    let mut src_v_strided = vec![0u16; stride * (h + 4)];
    if !mono {
        for r in 0..ch {
            src_u_strided[r * stride..r * stride + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
            src_v_strided[r * stride..r * stride + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
        }
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
        enable_optimize_b: if p.coded_lossless {
            TrellisOptType::NoTrellisOpt
        } else {
            TrellisOptType::FullTrellisOpt
        },
        use_chroma_trellis_rd_mult: allintra,
        coeff_costs_y: &real.coeff_costs_y,
        coeff_costs_uv: &real.coeff_costs_uv,
        tx_type_costs: &real.tx_type_costs_y,
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
        allintra,
        speed,
        qindex,
        enable_filter_intra: s.enable_filter_intra,
        enable_tx64: true,
        enable_rect_tx: true,
        intra_pruning_with_hog: if allintra {
            sf.intra_pruning_with_hog != 0
        } else {
            true
        },
        enable_rect_partitions: true,
        less_rectangular_check_level: if allintra {
            sf.less_rectangular_check_level
        } else {
            i32::from(allintra)
        },
        max_partition_size: 15,
        min_partition_size: 0,
        enable_1to4_partitions: true,
        enable_ab_partitions: true,
        allow_screen_content_tools: p.allow_screen_content_tools,
    };
    let pack_cfg = aom_encode::pack::PackCfg {
        enable_filter_intra: s.enable_filter_intra,
        tx_mode_is_select: p.tx_mode_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
        allow_screen_content_tools: p.allow_screen_content_tools,
    };

    let mut recon_y = src_y_strided.clone();
    let mut recon_u = src_u_strided.clone();
    let mut recon_v = src_v_strided.clone();
    let mut enc = OdEcEnc::new();
    let n_sb = (mi_cols / SB_MI).max(1);
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
        n_sb,
        n_sb,
        SB_MI,
        SB,
    );
    assert_eq!(
        trees.len(),
        (n_sb * n_sb) as usize,
        "{ctx}: pack_tile must walk every SB"
    );
    let our_tile_bytes = enc.done().to_vec();

    // Port-derived loop-filter level (same as the bd8 harness).
    let mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb, SB_MI, SB);
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
    let derived_lf = pick_filter_level(&lf_frame, allintra, 0, false);
    let mut p = p;
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;

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
        let tile_data_start = real_bit_len.div_ceil(8);
        eprintln!(
            "{ctx}: MISMATCH at byte {first_diff} (ours={:?} real={:?}); header-region={} \
             our_tile.len()={} real_frame.len()={}",
            our_payload.get(first_diff),
            frame_payload.get(first_diff),
            tile_data_start,
            our_tile_bytes.len(),
            frame_payload.len(),
        );
    }
    CaseResult {
        matched,
        real_payload: frame_payload.to_vec(),
    }
}

/// Thin monochrome wrapper: `mono=true`, `ss=(1,1)`, no chroma content.
fn run_mono_case(
    w: usize,
    h: usize,
    usage: u32,
    cq_level: i32,
    bd: u8,
    content: impl Fn(usize, usize) -> u16,
) -> CaseResult {
    run_case(w, h, true, 1, 1, usage, cq_level, bd, content, |_, _| 0)
}

/// bd10 monochrome ALLINTRA KEY frames: flat + textured content, byte-identical
/// to real aomenc. Flat is the trivial DC case; textured exercises the highbd
/// intra-prediction / residual / quant / recon / coeff-pack path.
#[test]
fn encoder_gate_bd10_mono() {
    // Flat mid-grey (bd10 range) — the simplest highbd case.
    let flat = run_mono_case(64, 64, 2, 32, 10, |_, _| 512);
    assert!(flat.matched, "bd10 mono flat must byte-match real aomenc");

    // Textured — a smooth 10-bit gradient with high-frequency detail so the
    // highbd residual/quant path is genuinely exercised (not an all-skip frame).
    let tex = run_mono_case(64, 64, 2, 32, 10, |r, ccol| {
        let base = ((r * 12 + ccol * 9) as u16) & 0x3ff;
        let hf = if (r + ccol) % 2 == 0 { 40 } else { 0 };
        (base ^ (hf as u16)).min(1023)
    });
    assert!(
        tex.matched,
        "bd10 mono textured must byte-match real aomenc"
    );
}

/// bd12 monochrome ALLINTRA KEY frames: flat + textured content, byte-identical
/// to real aomenc. Exercises the deepest bit-depth quantizer / transform range.
#[test]
fn encoder_gate_bd12_mono() {
    // Flat mid-grey (bd12 range).
    let flat = run_mono_case(64, 64, 2, 32, 12, |_, _| 2048);
    assert!(flat.matched, "bd12 mono flat must byte-match real aomenc");

    // Textured 12-bit gradient with high-frequency detail.
    let tex = run_mono_case(64, 64, 2, 32, 12, |r, ccol| {
        let base = ((r * 48 + ccol * 36) as u16) & 0xfff;
        let hf = if (r + ccol) % 2 == 0 { 160 } else { 0 };
        (base ^ (hf as u16)).min(4095)
    });
    assert!(
        tex.matched,
        "bd12 mono textured must byte-match real aomenc"
    );
}

/// Anti-vacuous: the bd10 gate (`port == C` at bd10) is only meaningful if the
/// bd10 reference is genuinely a 10-bit stream, not a bd8 stream mislabeled /
/// silently downcast. Prove it by encoding the SAME numeric content (values in
/// `[0,255]`, so both bit depths see identical samples with no differing clamp)
/// at bd10 and bd8 with real aomenc, and asserting the two reference streams
/// differ. Since the port byte-matches the bd10 reference (`encoder_gate_bd10_mono`
/// / the `r10.matched` check here), and that reference is provably distinct from
/// its bd8 counterpart, the port genuinely reproduces 10-bit output.
#[test]
fn bd10_stream_is_really_10bit() {
    // Smooth ramp + checkerboard HF, all values <= 156 so nothing clamps at
    // either bit depth — a fair same-content, different-bit-depth comparison.
    let content = |r: usize, cc: usize| (r + cc) as u16 + if (r ^ cc) & 1 == 1 { 30 } else { 0 };
    let r10 = run_mono_case(64, 64, 2, 32, 10, content);
    let r8 = run_mono_case(64, 64, 2, 32, 8, content);
    assert!(
        r10.matched,
        "bd10 port must byte-match the real aomenc bd10 stream"
    );
    assert_ne!(
        r10.real_payload, r8.real_payload,
        "bd10 vs bd8 reference streams must differ for identical content — else the \
         bd10 reference is not genuinely 10-bit and the gate would pass vacuously"
    );
}

/// bd10 / bd12 **4:2:0** ALLINTRA KEY frames: separate luma + chroma content,
/// byte-identical to real aomenc. Adds the highbd chroma path (intra UV mode
/// search, chroma residual/quant/recon, CfL) on top of the mono coverage.
/// Only runs because mono is byte-clean at both bit depths.
#[test]
fn encoder_gate_bd10_bd12_420() {
    // Distinct luma vs chroma textures so the chroma path is genuinely exercised
    // (not a copy of luma). bd10.
    let y10 = |r: usize, cc: usize| {
        let base = ((r * 12 + cc * 9) as u16) & 0x3ff;
        let hf = if (r + cc) % 2 == 0 { 40 } else { 0 };
        (base ^ (hf as u16)).min(1023)
    };
    let uv10 = |r: usize, cc: usize| (((r * 7 + cc * 5) as u16) & 0x3ff).min(1023);
    let c10 = run_case(64, 64, false, 1, 1, 2, 32, 10, y10, uv10);
    assert!(c10.matched, "bd10 4:2:0 must byte-match real aomenc");

    // bd12.
    let y12 = |r: usize, cc: usize| {
        let base = ((r * 48 + cc * 36) as u16) & 0xfff;
        let hf = if (r + cc) % 2 == 0 { 160 } else { 0 };
        (base ^ (hf as u16)).min(4095)
    };
    let uv12 = |r: usize, cc: usize| (((r * 28 + cc * 20) as u16) & 0xfff).min(4095);
    let c12 = run_case(64, 64, false, 1, 1, 2, 32, 12, y12, uv12);
    assert!(c12.matched, "bd12 4:2:0 must byte-match real aomenc");
}

/// Multi-size / multi-qindex bd10+bd12 gate with REPRESENTABLE-magnitude content
/// (luma/chroma values in `[0,255]`, fed identically at every bit depth). This
/// locks the high-bit-depth PLUMBING — quantizer build/selection, bit-depth
/// threading through the whole pack pipeline, header signaling — across
/// superblock boundaries (128 = 2x2 SB, 192 = 3x3 SB force real partition
/// recursion) and the aggressive-web qindex range (cq {12,32,63} -> qindex
/// ~48/128/232). Every cell's result is collected in one pass so a regression
/// shows the full grid, not just the first failing cell.
///
/// Genuine full-dynamic-range (>8-bit) coefficient magnitudes are covered
/// separately and exhaustively at the BLOCK level — `xform_quant_diff`,
/// `encode_block_coeffs_diff`, and `xform_quant_optimize_highbd_diff` all drive
/// ±4095 bd12 — so quantize / coeff-coding / trellis are byte-exact at high
/// magnitude. See KNOWN DIVERGENCE below for the one frame-level gap that remains.
///
/// KNOWN DIVERGENCE (staged for the encoder track, not asserted here): with
/// FULL-dynamic-range (>8-bit) aggressive high-frequency content at LOW qindex,
/// the port's tile bytes diverge from aomenc — different mode/tx/partition RD
/// winners, i.e. different byte counts, not a coding-of-fixed-decision bug. By
/// elimination (all block-level coefficient kernels are byte-exact at ±4095 bd12;
/// representable-magnitude frames are byte-exact here across all sizes/qindex; no
/// int64 overflow fires in debug), the divergence is in the RD *decision* layer
/// (tx_search / intra_rd / partition_pick), whose large-coefficient RD-cost
/// accumulation differs at high bit depth. Those are the encoder track's ACTIVE
/// files, so this bd10-track test does not touch them — the finding is reported.
#[test]
fn encoder_gate_bd10_bd12_multisize() {
    // Representable content (<=255): the SAME samples encode at bd8/10/12, so the
    // only high-bd variable is the quantizer/plumbing, not the coefficient range.
    let luma = |r: usize, cc: usize| {
        let base = ((r * 37 + cc * 23) as u16) & 0xff;
        let hf = if (r ^ cc) & 1 == 1 { 21 } else { 0 };
        base ^ hf
    };
    let uv = |r: usize, cc: usize| {
        let base = ((r * 19 + cc * 29) as u16) & 0xff;
        let hf = if (r + cc) % 3 == 0 { 11 } else { 0 };
        base ^ hf
    };
    let mut results: Vec<(String, bool)> = Vec::new();
    let mut run = |w: usize, h: usize, mono: bool, cq: i32, bd: u8| {
        let res = run_case(w, h, mono, 1, 1, 2, cq, bd, luma, uv);
        let kind = if mono { "mono" } else { "420 " };
        results.push((format!("bd{bd} {w}x{h} {kind} cq{cq:>2}"), res.matched));
    };

    // bd10 mono: three sizes (64/128/192 = 1x1 / 2x2 / 3x3 SB) x three qindex.
    for &(w, h) in &[(64usize, 64usize), (128, 128), (192, 192)] {
        for &cq in &[12i32, 32, 63] {
            run(w, h, true, cq, 10);
        }
    }
    // bd12 mono: two sizes x low+high qindex.
    for &(w, h) in &[(64usize, 64usize), (128, 128)] {
        for &cq in &[12i32, 63] {
            run(w, h, true, cq, 12);
        }
    }
    // Multi-SB 4:2:0 at both bit depths.
    run(128, 128, false, 32, 10);
    run(128, 128, false, 32, 12);

    eprintln!("\n=== bd10/bd12 multisize gate ===");
    for (label, ok) in &results {
        eprintln!("  {label}: {}", if *ok { "MATCH" } else { "MISMATCH" });
    }
    let failed: Vec<&String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(l, _)| l)
        .collect();
    assert!(
        failed.is_empty(),
        "{}/{} cells diverged from real aomenc: {:?}",
        failed.len(),
        results.len(),
        failed
    );
}

/// Cross-bit-depth plumbing guard: feed byte-IDENTICAL content (values in
/// `[0,255]`) to bd8, bd10 AND bd12, and require the port to byte-match aomenc at
/// every bit depth. Bit depth is the ONLY variable, so this proves the highbd
/// quantizer/threading is correct (not silently downcast) independent of content
/// range — and, with `bd10_stream_is_really_10bit`, that the streams genuinely
/// differ by bit depth.
#[test]
fn encoder_gate_bd_plumbing_8_10_12() {
    let luma = |r: usize, cc: usize| {
        let base = ((r * 37 + cc * 23) as u16) & 0xff;
        let hf = if (r ^ cc) & 1 == 1 { 21 } else { 0 };
        base ^ hf
    };
    let uv = |_r: usize, _cc: usize| 0u16;
    for &bd in &[8u8, 10, 12] {
        for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
            for &cq in &[12i32, 32] {
                let r = run_case(w, h, true, 1, 1, 2, cq, bd, luma, uv);
                assert!(
                    r.matched,
                    "bd{bd} {w}x{h} cq{cq} must byte-match real aomenc"
                );
            }
        }
    }
}
