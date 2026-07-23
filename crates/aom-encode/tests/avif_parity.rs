//! AVIF-parity harness (Gate 4, zenavif integration).
//!
//! The port already emits a byte-identical AV1 `OBU_FRAME` payload vs real
//! aomenc across a large ALLINTRA KEY envelope (`encoder_gate_e2e_*`). This
//! harness takes that proven, byte-exact payload one step further: it MUXES it
//! into an AVIF still (via `zenavif-serialize`) and closes the loop two ways.
//!
//!  1. **Round-trip (muxer preserves the codec bytes).** Parse the AVIF
//!     container, extract the coded AV1 bytes stored in `mdat`, and assert
//!     they byte-equal exactly what we handed the muxer. `zenavif-serialize`
//!     stores `color_av1_data` verbatim as the primary item's coded data, so
//!     a mismatch here is a muxer corruption bug.
//!  2. **Decode (the muxed stream is a valid, correct AV1 bitstream).** Feed
//!     the extracted AV1 stream to the port's OWN byte-exact AV1 decoder
//!     (`aom-decode`) and assert the decoded pixels match a decode of real
//!     aomenc's full stream (universal), and — for the lossless flat cells —
//!     the source pixels exactly.
//!
//! The muxed AV1 payload is produced by the SAME machinery the e2e byte gate
//! uses (`rd_pick_partition_real` + `pack_tile`, header bootstrapped from a
//! real parse, loop-filter level self-derived via `pick_filter_level`). This
//! harness ALSO re-asserts, per cell, that the port's frame-OBU payload
//! byte-equals real aomenc's — so the bytes it muxes are genuinely the port's
//! own proven output, not the C reference's. The chosen cells are all
//! asserted byte-exact by the `encoder_gate_e2e_*` gates (flat mono/420
//! ALLINTRA cq32; the strong-LF textured `diag+vbars16+ripple` 256² cq63):
//! any parity regression fails the precondition assert here too.
//!
//! **Envelope:** `enable_cdef=false, enable_restoration=false` (the same
//! bootstrap boundary the `encoder_gate_e2e_*` gates use — i.e. the
//! `--enable-restoration=0` config, NOT the plain-default `aomenc --allintra`
//! which is restoration-ON; that default-config parity is separately proven
//! in `aom-bench`'s `lr_default_parity.rs`). The AVIF plumbing this harness
//! validates is independent of that choice.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_obu_frame_single_tile;
use aom_encode::pack::pack_tile;
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::speed_features::SpeedFeatures;
use aom_encode::tx_search::TxTypeSearchPolicy;
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
};
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::partition::KfFrameContext;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;
use zenavif_serialize::{Aviffy, ChromaSubsampling};

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

// ---- OBU / tile helpers (duplicated per this test family's convention; see
//      e.g. `decode_diff_multisb.rs`'s own comment). ----

fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_dsp::entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_start = after_header + size_bytes;
        let payload_end = payload_start + size as usize;
        out.push((hdr.obu_type, &bytes[payload_start..payload_end]));
        pos = payload_end;
    }
    out
}

fn raw_obu_span(bytes: &[u8], want_type: u32) -> &[u8] {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_dsp::entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
        let payload_end = after_header + size_bytes + size as usize;
        if hdr.obu_type == want_type {
            return &bytes[pos..payload_end];
        }
        pos = payload_end;
    }
    panic!("no OBU of type {want_type} found");
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

/// Diagonal gradient + vertical bars (period 16) + fine ripple. Copied
/// verbatim from `encoder_gate_e2e_byte_match.rs`'s `lf_diag_vbars16_ripple`
/// (the strong-LF generator asserted byte-exact at 256² cq63 by
/// `encoder_gate_e2e_rich_content_strong_lf`).
fn lf_diag_vbars16_ripple(r: usize, c: usize) -> u8 {
    let grad = 32 + (r + c) * 150 / 256;
    let bar = if (c / 16) % 2 == 0 { 0 } else { 45 };
    let ripple = if (r + c) % 2 == 0 { 14 } else { -14 };
    (grad as i32 + bar + ripple).clamp(0, 255) as u8
}

/// Everything a cell produces for the AVIF-parity checks.
struct Produced {
    /// `raw_seq_hdr_obu ++ port_frame_obu` — a self-contained AV1 elementary
    /// stream (the exact bytes we hand the muxer).
    av1_stream: Vec<u8>,
    /// The full real-aomenc stream (TD + seq + frame), for the decode
    /// cross-check.
    real_full_stream: Vec<u8>,
    /// The port's own `OBU_FRAME` PAYLOAD (frame header + tile data), for the
    /// per-cell byte-exactness precondition.
    our_frame_payload: Vec<u8>,
    /// Real aomenc's `OBU_FRAME` payload.
    real_frame_payload: Vec<u8>,
    w: usize,
    h: usize,
    mono: bool,
    /// Tight `w`-strided source luma (u16 at bd8).
    src_y: Vec<u16>,
    /// Tight `cw`-strided source chroma (empty when mono).
    src_u: Vec<u16>,
    src_v: Vec<u16>,
    /// Whether the source is trivially lossless (flat) so decode == source.
    lossless_flat: bool,
}

/// Run the port's OWN encode pipeline for one still, exactly as
/// `encoder_gate_e2e_byte_match::attempt_case_content_uv_sep` does (header
/// bootstrapped from the real parse, coefficients/modes/partitions/tx +
/// loop-filter level all port-derived), and package the artifacts the AVIF
/// checks need.
#[allow(clippy::too_many_arguments)]
fn produce(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    lossless_flat: bool,
    content: impl Fn(usize, usize) -> u8,
) -> Produced {
    c::ref_init();
    let mut y = vec![128u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = u16::from(content(r, col));
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
    };
    // Chroma stays flat mid-grey 128 (matches the e2e gate's
    // `attempt_case_content`): only the luma decision space is stressed.
    let u = vec![128u16; cw * ch];
    let v = vec![128u16; cw * ch];

    let bytes = c::ref_encode_av1_kf(
        &y, &u, &v, w, h, 8, mono, ss_x as i32, ss_y as i32, cq_level, 0, false, false, usage, 0,
        false,
    );
    assert!(!bytes.is_empty(), "shim_encode_av1_kf must produce a stream");

    let obus = walk_obus(&bytes);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .unwrap_or_else(|| panic!("no sequence-header OBU (w={w} h={h})"));
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);

    let (_frame_obu_type, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME)
        .map(|(t, p)| (*t, *p))
        .unwrap_or_else(|| panic!("no OBU_FRAME (w={w} h={h})"));

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
    assert_eq!(p.prefix.frame_type, 0, "KEY frame only");
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "single-tile envelope only");
    let _tile_data_start = real_bit_len.div_ceil(8);

    // ---- port pipeline (config off the bootstrapped header; payload derived) ----
    let bd: u8 = 8;
    let allintra = usage == 2;
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

    let sf = SpeedFeatures::set_allintra(0, p.allow_screen_content_tools, false);
    let env = SbEncodeEnv {
            ref_frame: None,
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
        txfm_partition_costs: [[0i32; 2]; 21],
        tx_type_costs: &real.tx_type_costs_y,
        qm_levels: None,
        tune: Default::default(),
        deltaq: None,
    };
    let pick_cfg = PickFrameCfg {
            inter: None,
        intrabc: None,
        intra_tools: Default::default(),
        mode_costs: &real.mode_costs,
        tx_size_costs: &real.tx_size_costs,
        skip_costs: &real.skip_costs,
        tx_type_costs_y: &real.tx_type_costs_y,
        pol: &if allintra {
            sf.tx_type_search_policy(false, 0)
        } else {
            TxTypeSearchPolicy::speed0_good()
        },
        uv_lp: &UvLoopPolicy::speed0_allintra(),
        intra_uv_mode_cost: &real.mode_costs.intra_uv_mode_cost,
        cfl_costs: &real.cfl_costs,
        partition_costs: &real.partition_costs,
        partition_cdfs: &real.partition_cdf,
        allintra,
        speed: 0,
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
        max_partition_size: if allintra {
            sf.default_max_partition_size.min(15).min(SB)
        } else {
            15
        },
        min_partition_size: 0,
        enable_1to4_partitions: true,
        enable_ab_partitions: true,
        allow_screen_content_tools: p.allow_screen_content_tools,
        qm_levels: None,
        palette_costs: None,
    };
    let pack_cfg = aom_encode::pack::PackCfg {
        enable_filter_intra: s.enable_filter_intra,
        tx_mode_is_select: p.tx_mode_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
        delta_q_present: false,
        delta_q_res: 0,
        allow_screen_content_tools: p.allow_screen_content_tools,
        allow_intrabc: false,
    };

    let mut recon_y = src_y_strided.clone();
    let mut recon_u = src_u_strided.clone();
    let mut recon_v = src_v_strided.clone();
    let mut enc = OdEcEnc::new();
    let n_sb_x = (mi_cols / SB_MI).max(1);
    let n_sb_y = (mi_rows / SB_MI).max(1);
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
    let our_tile_bytes = enc.done().to_vec();

    // loop-filter-level: TRUE DERIVATION from OUR reconstruction (speed-0 dual).
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
        delta_lf_present: false,
    };
    let derived_lf = pick_filter_level(&lf_frame, allintra, 0, false);
    let mut p = p;
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;

    // Assemble the port's full frame OBU (header byte + leb128 size + payload).
    let our_frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &our_tile_bytes, false, 0);
    // Its inner payload (for the byte-exactness precondition vs real).
    let our_frame_payload = {
        let obus = walk_obus(&our_frame_obu);
        obus.iter()
            .find(|(t, _)| *t == OBU_FRAME)
            .map(|(_, pl)| pl.to_vec())
            .expect("assembled a frame OBU")
    };

    // Materialize the borrows of `bytes` into owned Vecs BEFORE moving `bytes`
    // into the returned struct (ends the `walk_obus`/`raw_obu_span` borrows).
    let real_frame_payload = frame_payload.to_vec();
    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER).to_vec();
    let mut av1_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    av1_stream.extend_from_slice(&seq_hdr_raw);
    av1_stream.extend_from_slice(&our_frame_obu);

    Produced {
        av1_stream,
        real_full_stream: bytes,
        our_frame_payload,
        real_frame_payload,
        w,
        h,
        mono,
        src_y: y,
        src_u: u,
        src_v: v,
        lossless_flat,
    }
}

/// Minimal ISO-BMFF box walker: return the payload of the top-level `mdat`
/// box. For a color-only AVIF still (no alpha, no Exif) `zenavif-serialize`'s
/// `mdat` contains exactly the color AV1 data as a single extent, so this IS
/// the primary item's coded data. Handles the 64-bit `largesize` form.
fn extract_mdat_payload(avif: &[u8]) -> Vec<u8> {
    let mut pos = 0usize;
    while pos + 8 <= avif.len() {
        let size32 = u32::from_be_bytes(avif[pos..pos + 4].try_into().unwrap()) as u64;
        let typ = &avif[pos + 4..pos + 8];
        let (payload_start, box_end) = if size32 == 1 {
            // 64-bit largesize follows the type.
            assert!(pos + 16 <= avif.len(), "truncated 64-bit box header");
            let large = u64::from_be_bytes(avif[pos + 8..pos + 16].try_into().unwrap());
            (pos + 16, pos + large as usize)
        } else if size32 == 0 {
            // Box extends to end of file.
            (pos + 8, avif.len())
        } else {
            (pos + 8, pos + size32 as usize)
        };
        assert!(box_end <= avif.len(), "box size past end of file");
        if typ == b"mdat" {
            return avif[payload_start..box_end].to_vec();
        }
        pos = box_end;
    }
    panic!("no mdat box found in AVIF output");
}

/// Confirm the muxer stores the color AV1 data verbatim and the port's stream
/// decodes correctly through the container -- across flat mono/420 (lossless)
/// and strong-LF textured 420 (lossy) cells, all asserted byte-exact vs real
/// aomenc by the `encoder_gate_e2e_*` gates.
#[test]
fn avif_parity_roundtrip_and_decode() {
    // (name, w, h, mono, ss_x, ss_y, usage, cq, lossless_flat, content)
    #[allow(clippy::type_complexity)]
    let cells: &[(&str, usize, usize, bool, usize, usize, u32, i32, bool, fn(usize, usize) -> u8)] = &[
        ("flat128 mono 64x64 allintra cq32", 64, 64, true, 1, 1, 2, 32, true, |_, _| 128),
        ("flat128 420 64x64 allintra cq32", 64, 64, false, 1, 1, 2, 32, true, |_, _| 128),
        (
            "diag+vbars16+ripple 256x256 420 allintra cq63",
            256,
            256,
            false,
            1,
            1,
            2,
            63,
            false,
            lf_diag_vbars16_ripple,
        ),
    ];

    let mut passed = 0usize;
    for &(name, w, h, mono, ss_x, ss_y, usage, cq, lossless_flat, content) in cells {
        let prod = produce(w, h, mono, ss_x, ss_y, usage, cq, lossless_flat, content);

        // (0) precondition: the muxed payload IS the port's proven byte-exact
        //     frame OBU (== real aomenc). A failure here is an encoder parity
        //     regression, surfaced honestly rather than hidden.
        assert_eq!(
            prod.our_frame_payload, prod.real_frame_payload,
            "{name}: port frame-OBU payload must byte-equal real aomenc (e2e-gate property)"
        );

        // Mux the port's AV1 stream into an AVIF still.
        let subsampling = if mono {
            ChromaSubsampling::NONE
        } else if ss_x == 1 && ss_y == 1 {
            ChromaSubsampling::YUV420
        } else if ss_x == 1 && ss_y == 0 {
            ChromaSubsampling::YUV422
        } else {
            ChromaSubsampling::NONE
        };
        let avif = Aviffy::new()
            .set_monochrome(mono)
            .set_chroma_subsampling(subsampling)
            .try_to_vec(&prod.av1_stream, None, w as u32, h as u32, 8)
            .expect("muxing a valid AVIF");

        // (1) ROUND-TRIP: the container preserves the codec bytes verbatim.
        let extracted = extract_mdat_payload(&avif);
        assert_eq!(
            extracted, prod.av1_stream,
            "{name}: AVIF mdat must byte-equal the muxed AV1 stream (muxer preservation)"
        );

        // (2a) DECODE the extracted stream via the port's own AV1 decoder.
        let dec = aom_decode::frame::decode_frame_obus(&extracted)
            .unwrap_or_else(|e| panic!("{name}: decode of the AVIF-extracted AV1 failed: {e}"));
        // Decode the real full aomenc stream for a universal cross-check.
        let dec_real = aom_decode::frame::decode_frame_obus(&prod.real_full_stream)
            .unwrap_or_else(|e| panic!("{name}: decode of real aomenc stream failed: {e}"));

        assert_eq!(dec.width, prod.w, "{name}: decoded width");
        assert_eq!(dec.height, prod.h, "{name}: decoded height");
        assert_eq!(
            (dec.y.clone(), dec.u.clone(), dec.v.clone()),
            (dec_real.y.clone(), dec_real.u.clone(), dec_real.v.clone()),
            "{name}: decode(muxed) must pixel-equal decode(real aomenc) -- the container \
             round-trip is pixel-lossless and the port's stream decodes identically to real"
        );

        // (2b) For the trivially-lossless flat cells, the round-tripped decode
        //      must reproduce the SOURCE exactly (encode -> mux -> demux ->
        //      decode is the identity on flat content).
        if prod.lossless_flat {
            assert_eq!(
                dec.y, prod.src_y,
                "{name}: lossless-flat decode(muxed).y must equal the source luma"
            );
            if !prod.mono {
                assert_eq!(dec.u, prod.src_u, "{name}: lossless-flat decode(muxed).u == source");
                assert_eq!(dec.v, prod.src_v, "{name}: lossless-flat decode(muxed).v == source");
            }
        }

        eprintln!(
            "avif_parity {name}: OK -- av1_stream={} bytes, avif={} bytes, mdat round-trip byte-exact, \
             decode(muxed)==decode(real){}",
            prod.av1_stream.len(),
            avif.len(),
            if prod.lossless_flat { " and ==source" } else { "" },
        );
        passed += 1;
    }
    assert_eq!(passed, cells.len(), "every AVIF-parity cell must pass");
    eprintln!("avif_parity_roundtrip_and_decode: {passed}/{} cells OK", cells.len());
}
