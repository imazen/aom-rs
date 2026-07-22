//! KB-6 (task #34) real-content bd8 4:2:0 RD-decision divergence LOCALIZER
//! (diagnostic).
//!
//! `encoder_gate_real_image_e2e_kb6_repro` (encoder_gate_chroma_ss_e2e.rs)
//! discovered that GENUINE image content (decoded from the AV1 conformance
//! corpus, re-encoded) diverges from real aomenc on the PRIMARY bd8 4:2:0
//! speed-0 KEY path, while every synthetic gate is byte-exact. This file
//! localizes the FIRST divergent superblock on the most robustly-divergent
//! cell (196x196 cq20) so the flipped RD decision can be root-caused.
//!
//! Technique (identical to `kb4_bd10_rd_localize.rs::localize_mono`, retargeted
//! to bd8 4:2:0 with REAL corpus content): decode the vector's first KEY frame
//! to genuine YUV via the C decode oracle, encode it with real aomenc, run THIS
//! PORT's `pack_tile` over the identical pixels, re-wrap the port's tile bytes
//! into a real OBU stream, then DECODE BOTH the aomenc stream and the port's
//! stream with the (bit-exact vs C) port decoder and diff the per-block records:
//!   1. partition tree, node-for-node (`replay_tree`) -> first divergent
//!      `(mi_row, mi_col, bsize)` partition decision;
//!   2. else every shared leaf's mode/tx fields + per-txb `(eob, tx_type)`;
//!   3. else the first divergent reconstruction pixel.
//! The first divergence + which field pins the divergent RD decision.
//!
//! This file is OWNED by the encoder track (KB-6); it does NOT touch the bd10
//! track's `encoder_gate_bd10_diff.rs`.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::obu_assemble::assemble_obu_frame_single_tile;
use aom_encode::pack::pack_tile;
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::speed_features::SpeedFeatures;
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader, read_sequence_header_obu, read_uncompressed_header,
};
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::partition::{KfFrameContext, get_partition_subsize};
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4

const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_SIZE_HIGH_B: [usize; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

const PARTITION_NAMES: [&str; 10] = [
    "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
];

const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

fn corpus_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("AOM_CONFORMANCE_DIR") {
        return std::path::PathBuf::from(d);
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("conformance")
        .join("data")
}

fn ivf_hdr_dims(data: &[u8]) -> (usize, usize) {
    (
        u16::from_le_bytes([data[12], data[13]]) as usize,
        u16::from_le_bytes([data[14], data[15]]) as usize,
    )
}

fn ivf_temporal_units(data: &[u8]) -> Vec<Vec<u8>> {
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 12;
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

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

#[allow(clippy::too_many_arguments)]
fn replay_tree(
    tree: &[i8],
    cursor: &mut usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
    out: &mut Vec<(i32, i32, usize, i8)>,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    let p = tree[*cursor];
    out.push((mi_row, mi_col, bsize, p));
    *cursor += 1;
    if p as usize == 3 {
        let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
        let subsize = get_partition_subsize(bsize, p as i32) as usize;
        replay_tree(tree, cursor, mi_row, mi_col, subsize, mi_rows, mi_cols, out);
        replay_tree(
            tree,
            cursor,
            mi_row,
            mi_col + hbs,
            subsize,
            mi_rows,
            mi_cols,
            out,
        );
        replay_tree(
            tree,
            cursor,
            mi_row + hbs,
            mi_col,
            subsize,
            mi_rows,
            mi_cols,
            out,
        );
        replay_tree(
            tree,
            cursor,
            mi_row + hbs,
            mi_col + hbs,
            subsize,
            mi_rows,
            mi_cols,
            out,
        );
    }
}

/// Decode a real conformance vector's first KEY frame, encode it with real
/// aomenc AND the port, decode both with the bit-exact port decoder, and diff
/// the per-block decisions. bd8 4:2:0 (the vector's native format). Returns
/// `true` if decisions + recon are identical (byte-exact); prints the first
/// divergence otherwise.
/// Localize one cell. `crop_w == 0` => the full decoded frame; otherwise an
/// SB-aligned (mult-of-64) crop at (`off_x`, `off_y`) — same crop convention as
/// the e2e gate, so a crop cell here matches a crop cell there byte-for-byte.
fn localize_real(
    name: &str,
    cq_level: i32,
    crop_w: usize,
    crop_h: usize,
    off_x: usize,
    off_y: usize,
) -> bool {
    localize_real_speed(name, cq_level, crop_w, crop_h, off_x, off_y, 0)
}

/// Speed-parameterized core of [`localize_real`]. `cpu_used` selects BOTH the
/// aomenc reference speed and this port's `SpeedFeatures::set_allintra(cpu_used)`
/// — used by the TASK #39 real-content speed>=1 localization. `localize_real`
/// (above) == speed 0. Faithful to `attempt_case_content_uv_sep`'s speed-N
/// config for cpu_used 1..=4 (rd_pick_partition path; the sf-derived pol/hog/
/// rect fields all read `speed`; `max_partition_size` stays BLOCK_64X64 through
/// speed 5 so the literal 15 is correct here).
#[allow(clippy::too_many_arguments)]
fn localize_real_speed(
    name: &str,
    cq_level: i32,
    crop_w: usize,
    crop_h: usize,
    off_x: usize,
    off_y: usize,
    cpu_used: i32,
) -> bool {
    c::ref_init();
    let dir = corpus_dir();
    let path = dir.join(format!("{name}.ivf"));
    let ivf = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{name}: conformance vector missing at {path:?} ({e}); fetch via \
             `python3 xtask/conformance.py --fetch --scope intra`"
        )
    });
    let (dw, dh) = ivf_hdr_dims(&ivf);
    let tus = ivf_temporal_units(&ivf);
    let frame = c::ref_decode_av1_kf(&tus[0], dw, dh);
    let bd = frame.info[0] as u8;
    let mono = frame.info[1] != 0;
    let ss_x = frame.info[2] as usize;
    let ss_y = frame.info[3] as usize;
    let (w, h) = if crop_w == 0 {
        (dw, dh)
    } else {
        (crop_w, crop_h)
    };
    assert!(
        off_x + w <= dw && off_y + h <= dh && off_x % 2 == 0 && off_y % 2 == 0,
        "{name}: crop {w}x{h}@{off_x},{off_y} out of bounds / not chroma-aligned"
    );
    let dcw = (dw + ss_x) >> ss_x; // full-frame chroma stride
    let cw = (w + ss_x) >> ss_x;
    let ch = (h + ss_y) >> ss_y;
    let (cox, coy) = (off_x >> ss_x, off_y >> ss_y);
    let usage = 2u32;
    let maxv = (1u16 << bd) - 1;

    // Tight planes for the aomenc encode (cropped from the decoded frame).
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = frame.y[(r + off_y) * dw + (col + off_x)].min(maxv);
        }
    }
    let (mut u, mut v) = (vec![0u16; cw * ch], vec![0u16; cw * ch]);
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                u[r * cw + col] = frame.u[(r + coy) * dcw + (col + cox)].min(maxv);
                v[r * cw + col] = frame.v[(r + coy) * dcw + (col + cox)].min(maxv);
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
        cpu_used,
        false,
        false,
        usage,
        0,
        false,
    );
    assert!(!bytes.is_empty());

    let obus = walk_obus(&bytes);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .unwrap();
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);

    let (frame_obu_type, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME)
        .map(|(t, p)| (*t, *p))
        .unwrap();
    assert_eq!(frame_obu_type, OBU_FRAME);

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
    assert!(!p.prefix.show_existing_frame);
    assert_eq!(p.prefix.frame_type, 0);
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "single-tile envelope only");

    // ---- port pipeline (EXACT config parity with run_case in the e2e gate) ----
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
    let allintra = usage == 2;
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

    // Partial-SB (frame-edge) source setup: size the planes to the SB-aligned
    // extent (CEIL(mi/16) SBs) and replicate the crop edge into the off-frame
    // overhang, matching aom_extend_frame_borders (C reads the FULL block/tx from
    // the bordered source). SB-aligned crops keep the usual 320 / h+4 envelope.
    let n_sb_x = ((mi_cols + SB_MI - 1) / SB_MI).max(1);
    let n_sb_y = ((mi_rows + SB_MI - 1) / SB_MI).max(1);
    let sb_px_w = n_sb_x as usize * 64;
    let sb_px_h = n_sb_y as usize * 64;
    let stride = 320.max(sb_px_w + 4);
    let buf_h = (sb_px_h + 4).max(h + 4);
    let extend_plane = |dst: &mut [u16], pw: usize, ph: usize| {
        for r in 0..ph {
            let edge = dst[r * stride + pw - 1];
            for c in pw..stride {
                dst[r * stride + c] = edge;
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

    let speed = cpu_used;
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
        txfm_partition_costs: [[0i32; 2]; 21],
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
    // n_sb_x/n_sb_y computed above (from the plane sizing); pass (rows, cols).
    let _trees = pack_tile(
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

    // ---- rewrap OUR bytes into a real OBU stream and decode BOTH ----
    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER);
    let our_frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &our_tile_bytes, false, 0);
    let mut our_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    our_stream.extend_from_slice(seq_hdr_raw);
    our_stream.extend_from_slice(&our_frame_obu);

    let (t_real, _c1, _h1) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("decode of REAL aomenc bytes failed: {e}"));
    let (t_ours, _c2, _h2) = aom_decode::frame::decode_frame_obus_prefilter(&our_stream)
        .unwrap_or_else(|e| panic!("decode of OUR OWN rewrapped bytes failed: {e}"));

    let mut real_seq = Vec::new();
    let mut ours_seq = Vec::new();
    replay_tree(
        &t_real.tree,
        &mut 0,
        0,
        0,
        SB,
        mi_rows,
        mi_cols,
        &mut real_seq,
    );
    replay_tree(
        &t_ours.tree,
        &mut 0,
        0,
        0,
        SB,
        mi_rows,
        mi_cols,
        &mut ours_seq,
    );

    let mut partition_div: Option<(i32, i32, usize, i8, i8)> = None;
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        if (r.0, r.1, r.2) != (o.0, o.1, o.2) {
            break;
        }
        if r.3 != o.3 {
            partition_div = Some((r.0, r.1, r.2, r.3, o.3));
            break;
        }
    }

    let mut leaf_div: Option<String> = None;
    for rbk in &t_real.blocks {
        if let Some(ob) = t_ours
            .blocks
            .iter()
            .find(|b| b.mi_row == rbk.mi_row && b.mi_col == rbk.mi_col)
        {
            let modes_differ = ob.bsize != rbk.bsize
                || ob.partition != rbk.partition
                || ob.info.y_mode != rbk.info.y_mode
                || ob.info.angle_delta_y != rbk.info.angle_delta_y
                || ob.info.use_filter_intra != rbk.info.use_filter_intra
                || ob.tx_size != rbk.tx_size
                || ob.info.uv_mode != rbk.info.uv_mode
                || ob.info.angle_delta_uv != rbk.info.angle_delta_uv;
            let txbs_differ = ob.txbs != rbk.txbs || ob.txbs_uv != rbk.txbs_uv;
            if modes_differ || txbs_differ {
                leaf_div = Some(format!(
                    "(mi_row={}, mi_col={}) [modes_differ={modes_differ} txbs_differ={txbs_differ}]\n\
                     \x20    real bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} aduv={} txbs(eob,tt)={:?} txbs_uv={:?}\n\
                     \x20    ours bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} aduv={} txbs(eob,tt)={:?} txbs_uv={:?}",
                    rbk.mi_row,
                    rbk.mi_col,
                    rbk.bsize,
                    rbk.partition,
                    rbk.info.y_mode,
                    rbk.info.angle_delta_y,
                    rbk.info.use_filter_intra,
                    rbk.tx_size,
                    rbk.info.uv_mode,
                    rbk.info.angle_delta_uv,
                    rbk.txbs,
                    rbk.txbs_uv,
                    ob.bsize,
                    ob.partition,
                    ob.info.y_mode,
                    ob.info.angle_delta_y,
                    ob.info.use_filter_intra,
                    ob.tx_size,
                    ob.info.uv_mode,
                    ob.info.angle_delta_uv,
                    ob.txbs,
                    ob.txbs_uv
                ));
                break;
            }
        }
    }

    // Diagnostic: when the partition tree flips, dump the real vs port leaves
    // covering that node's mi extent, so the flipped sub-partition's per-leaf
    // modes/eobs are visible (e.g. C's HORZ_4 strips vs the port's SPLIT leaves).
    if let Some((dr, dc, dbs, rp, op)) = partition_div {
        let (nw, nh) = (MI_SIZE_WIDE_B[dbs] as i32, MI_SIZE_HIGH_B[dbs] as i32);
        eprintln!(
            "  --- divergent node ({dr},{dc}) bsize={dbs} {nw}x{nh}mi: real={} vs ours={} ---",
            PARTITION_NAMES[rp as usize], PARTITION_NAMES[op as usize]
        );
        for (tag, blocks) in [("real", &t_real.blocks), ("ours", &t_ours.blocks)] {
            for b in blocks.iter().filter(|b| {
                b.mi_row >= dr && b.mi_row < dr + nh && b.mi_col >= dc && b.mi_col < dc + nw
            }) {
                eprintln!(
                    "    {tag} ({},{}) bsize={} part={} y_mode={} adly={} use_fi={} tx={} uv_mode={} eobs={:?}",
                    b.mi_row,
                    b.mi_col,
                    b.bsize,
                    b.partition,
                    b.info.y_mode,
                    b.info.angle_delta_y,
                    b.info.use_filter_intra,
                    b.tx_size,
                    b.info.uv_mode,
                    b.txbs.iter().map(|t| t.0).collect::<Vec<_>>()
                );
            }
        }
    }

    let mut recon_div: Option<(usize, usize, u16, u16)> = None;
    'rec: for row in 0..t_real.height.min(t_ours.height) {
        for col in 0..t_real.width.min(t_ours.width) {
            let rv = t_real.recon.px(row * t_real.stride + col);
            let ovv = t_ours.recon.px(row * t_ours.stride + col);
            if rv != ovv {
                recon_div = Some((row, col, rv, ovv));
                break 'rec;
            }
        }
    }

    let decisions_match = partition_div.is_none() && leaf_div.is_none() && recon_div.is_none();
    let fmt = match (mono, ss_x, ss_y) {
        (true, _, _) => "mono",
        (false, 1, 1) => "4:2:0",
        (false, 1, 0) => "4:2:2",
        (false, 0, 0) => "4:4:4",
        _ => "chroma",
    };
    eprintln!(
        "\n=== {name} bd{bd} {w}x{h} {fmt} cq{cq_level} (qindex={qindex}) === {}",
        if decisions_match {
            "MATCH (decisions + recon identical)".to_string()
        } else {
            format!(
                "DIVERGE (real frame_payload={} ours tile={})",
                frame_payload.len(),
                our_tile_bytes.len()
            )
        }
    );
    if decisions_match {
        return true;
    }

    if let Some((mi_row, mi_col, bsize, pr, po)) = partition_div {
        eprintln!(
            "  >>> FIRST PARTITION DIVERGENCE at (mi_row={mi_row}, mi_col={mi_col}, bsize={bsize}) \
             SB(mi_row={},mi_col={}): real=PARTITION_{} ({pr}) ours=PARTITION_{} ({po})",
            (mi_row / SB_MI) * SB_MI,
            (mi_col / SB_MI) * SB_MI,
            PARTITION_NAMES[pr as usize],
            PARTITION_NAMES[po as usize]
        );
    }
    if let Some(d) = &leaf_div {
        eprintln!("  >>> FIRST LEAF MISMATCH at {d}");
    }
    if let Some((row, col, rv, ov)) = recon_div {
        eprintln!(
            "  >>> FIRST RECON PIXEL DIVERGENCE at luma (row={row}, col={col}) -> \
             SB(mi_row={}, mi_col={}): real={rv} ours={ov}",
            (row / 64) * 16,
            (col / 64) * 16,
        );
    }
    false
}

/// KB-6 localizer: pin the first divergent SB on the CLEAN, SB-aligned real cell
/// (64x64 = exactly 1 superblock, so no partial-edge-SB confound). This cell
/// diverges at cq5/12/32/48 (matches only at the coincidental cq20/cq63), so
/// cq12 is a robust RD-near-tie repro. Report-only diagnostic — surfaces the
/// first partition/leaf/recon divergence for the encoder track to root-cause.
/// The gate is `encoder_gate_real_image_e2e_kb6_repro`.
///
/// NOTE: the 196x196 cell is NOT used here — 196px is not a multiple of 64, so
/// it has a partial edge superblock (mi_cols=50, not 16-aligned). The port's
/// tile context arrays are sized to mi_cols, so a full-SB context read at the
/// edge SB (mi_col=48 -> above_ectx[48..64]) panics (partition_pick.rs:858);
/// run_case's `n_sb = mi_cols/SB_MI` floors to 3, silently encoding only
/// 192x192. That partial-SB gap is a SEPARATE real bug (tracked in KB-6), so the
/// clean RD-near-tie is isolated here on the 1-SB cell.
#[test]
fn kb6_localize_real_64_cq12() {
    let matched = localize_real("av1-1-b8-01-size-64x64", 12, 0, 0, 0, 0);
    eprintln!(
        "\n=== KB-6 localize 64x64 cq12: {} ===",
        if matched {
            "MATCH"
        } else {
            "DIVERGE (see first divergence above)"
        }
    );
}

/// Localize the **cq5 (qindex 20, aggressive low-q) near-tie** on real
/// photographic content — the highest-value KB-6 target after the luma re-encode
/// fix. It diverges across ALL THREE crop contents (quantizer-64²/128², film-64²)
/// so it is content-independent. This uses the SAME 00-quantizer 64x64@96,64 crop
/// the e2e gate flags MISMATCH at cq5, isolated to a single SB. Report-only
/// diagnostic (the gate is `encoder_gate_real_image_e2e_kb6_repro`).
#[test]
fn kb6_localize_quantizer_64_cq5() {
    let matched = localize_real("av1-1-b8-00-quantizer-00", 5, 64, 64, 96, 64);
    eprintln!(
        "\n=== KB-6 localize quantizer 64x64@96,64 cq5: {} ===",
        if matched {
            "MATCH"
        } else {
            "DIVERGE (see first divergence above)"
        }
    );
}

/// Second cq5 content (film grain) to confirm the low-q near-tie is
/// CONTENT-INDEPENDENT: if the FIRST divergent decision here is the same CLASS as
/// the quantizer crop (a 4-way HORZ_4/VERT_4 vs SPLIT partition flip), the root is
/// one shared speed-0 RD decision, not per-content noise. Report-only diagnostic.
#[test]
fn kb6_localize_film_64_cq5() {
    let matched = localize_real("av1-1-b8-23-film_grain-50", 5, 64, 64, 96, 64);
    eprintln!(
        "\n=== KB-6 localize film 64x64@96,64 cq5: {} ===",
        if matched {
            "MATCH"
        } else {
            "DIVERGE (see first divergence above)"
        }
    );
}

/// Localize the **cq32 (qindex 128, mid-q) near-tie** on the SAME 00-quantizer
/// 64x64@96,64 photographic crop. NOTE (2026-07-15 finding): this crop's RD
/// decisions + recon are IDENTICAL to aomenc at cq32 (this localizer reports
/// MATCH); the e2e byte divergence, if any, is header-layer (derived LF level),
/// not RD — the localizer compares PREFILTER recon + modes and leaves LF at the
/// real parsed value, so it cannot see an LF-level header mismatch. Report-only.
#[test]
fn kb6_localize_quantizer_64_cq32() {
    let matched = localize_real("av1-1-b8-00-quantizer-00", 32, 64, 64, 96, 64);
    eprintln!(
        "\n=== KB-6 localize quantizer 64x64@96,64 cq32: {} ===",
        if matched {
            "MATCH"
        } else {
            "DIVERGE (see first divergence above)"
        }
    );
}

/// Localize the **size-64x64 cq32 (qindex 128) near-tie** — the documented KB-6
/// "SECOND near-tie" (CLAUDE.md). The e2e gate flags this cell MISMATCH at tile
/// byte ~390 (a TILE-PAYLOAD divergence, header-region=7), so the coded symbols
/// differ — an RD decision the localizer CAN pin (partition/leaf/recon compare).
/// Report-only diagnostic; the gate is `encoder_gate_real_image_e2e_kb6_repro`.
#[test]
fn kb6_localize_size_64_cq32() {
    let matched = localize_real("av1-1-b8-01-size-64x64", 32, 0, 0, 0, 0);
    eprintln!(
        "\n=== KB-6 localize size-64x64 cq32: {} ===",
        if matched {
            "MATCH"
        } else {
            "DIVERGE (see first divergence above)"
        }
    );
}

/// Localize the last open **196x196 partial-SB divergence** (cq48). 196px is
/// not a multiple of 64 → mi_cols = mi_dim(196) = 50: partial edge superblocks
/// along the right column and bottom row (mi 48, a 2-mi/8px visible strip).
///
/// FIXED history (each previously pinned here or in the map): the harness
/// true-frame + border-extend (CHUNK 0), luma + chroma visible-distortion
/// clips (CHUNKs 1-2), the edge partition-cost override (CHUNK 3), the KB-4
/// OUTPUT_ENABLED tx_type_map reset-leak (a2dd28e — closed cq63), and the
/// **frame-edge entropy-context stamp tail-zero** (`av1_set_entropy_contexts`,
/// blockd.c:29 — the port stamped the txb's cul across the FULL tx footprint
/// at edge txbs where C zeroes the beyond-visible tail; the phantom nonzero
/// culs at out-of-frame columns fed later edge blocks' full-footprint
/// `get_txb_ctx` reads → wrong txb_skip_ctx → same symbols on
/// different-probability cdf rows → stream desync at the bottom SB row; the
/// apparent mi(48,0) "16x8-vs-8x4 over-split" was that desync's decode
/// artifact, not a search decision) + the edge partition-cost gather reading
/// the frame-init `cm->fc` table (not the adapted one) — which together
/// closed cq12/20/32. cq32/cq12/cq20/cq5/cq63 are asserted byte-match gates
/// in `encoder_gate_real_image_e2e_kb6_repro`.
///
/// FIXED 2026-07-16 (the LAST cell — map 30/30): **cq48** (qindex 192) was
/// NOT a search near-tie — decode-both + full pass-marker dumps proved the
/// port's search decisions, per-pass tx-type evals, winner stores, and both
/// OUTPUT_ENABLED requant walks were ALREADY identical to C at the divergent
/// leaf (mi(0,48) 32×64 SMOOTH, txbs blk(8,0)/(12,0)). The divergence was a
/// WRITE-side cdf-row defect (the entropy-stamp-fix class, dc-sign flavour):
/// C's pack writes each txb's coefficients with the `(txb_skip_ctx,
/// dc_sign_ctx)` cached by the TOKENIZE walk
/// (`av1_update_and_record_txb_context`, encodetxb.c — derived from the
/// PERSISTENT entropy arrays whose within-leaf stamps are edge-CLIPPED
/// `av1_set_entropy_contexts`), while the TRELLIS uses the encode walk's
/// local full-footprint `av1_set_txb_context` stamps. The port used the
/// trellis pair for both; at txb blk(8,0) (vis 8×16 of a 16×16 txb) the
/// above-footprint tail-zero flipped the dc-sign SUM (C: −4+2 = −2 → ctx 1;
/// port: −4+4 = 0 → ctx 0) → the DC-sign symbol went to a different cdf row
/// → bits diverged at tile byte ~253 with IDENTICAL symbols, and the decoded
/// "txb4 = (eob4,tt2)" was that desync's artifact, not a decision. Fix:
/// encode_sb.rs Step 4 (the tokenize-equivalent stamp loop) derives the
/// pack's write ctx from the persistent arrays per txb — before that txb's
/// clipped stamp — exactly C's tokenize; interior txbs derive identical
/// values. Promoted: 196² cq48 is an asserted byte-match cell in
/// `encoder_gate_real_image_e2e_kb6_repro` (30/30); this pin now asserts the
/// cell KEEPS matching (decisions + recon identical via decode-both).
#[test]
fn kb6_characterize_196_partial_sb() {
    let matched = localize_real("av1-1-b8-01-size-196x196", 48, 0, 0, 0, 0);
    eprintln!("\n=== KB-6 196x196 partial-SB cq48: encode OK, matched={matched} ===");
    assert!(
        matched,
        "KB-6 196x196 cq48 REGRESSED: the tokenize write-ctx fix (encode_sb.rs \
         Step 4) previously made this cell byte-match real aomenc (map 30/30)."
    );
}

// ============================================================================
// TASK #39 — real-content speed>=1 localization. Decode-both (port decoder is
// bit-exact) pins the FIRST divergent SB/leaf for representative DIFF cells
// from `encoder_gate_real_content_speed1to4_e2e`'s map, classifying each
// (partition flip / leaf mode-tx / recon) so the encoder track can root-cause.
// Report-only diagnostics — the gate is encoder_gate_real_content_speed1to4_e2e.
// ============================================================================

/// The 196x196 PARTIAL-SB cluster (0/12 in the map — all cpu 1..4 × cq diverge).
/// KB-6 made this cell byte-exact at speed 0; here at speed>=1 it regresses. This
/// pins the first divergence at three (cpu,cq) points to see whether it's ONE
/// shared root across speeds or per-speed near-ties. Report-only.
#[test]
fn task39_localize_196_partial_sb_speed() {
    for (cpu, cq) in [(1i32, 32i32), (1, 12), (4, 32)] {
        eprintln!("\n########## TASK#39 196x196 cpu{cpu} cq{cq} ##########");
        let matched = localize_real_speed("av1-1-b8-01-size-196x196", cq, 0, 0, 0, 0, cpu);
        eprintln!(
            "=== TASK#39 196x196 cpu{cpu} cq{cq}: {} ===",
            if matched { "MATCH" } else { "DIVERGE (see first divergence above)" }
        );
    }
}

/// Interior multi-SB (quantizer 128x128@64,64, 4 aligned SBs — no partial edge)
/// + interior 1-SB near-ties (quantizer/film 64x64 crops). These isolate the
/// speed>=1 INTERIOR partition/mode near-ties (KB-2/10/11/12 family) from the
/// 196 partial-SB edge path. Report-only.
#[test]
fn task39_localize_interior_speed() {
    let cells: &[(&str, i32, usize, usize, usize, usize, i32)] = &[
        ("av1-1-b8-00-quantizer-00", 32, 128, 128, 64, 64, 1), // 4-SB interior
        ("av1-1-b8-00-quantizer-00", 12, 128, 128, 64, 64, 1),
        ("av1-1-b8-00-quantizer-00", 12, 64, 64, 96, 64, 1), // 1-SB interior
        ("av1-1-b8-00-quantizer-00", 32, 64, 64, 96, 64, 2),
        ("av1-1-b8-23-film_grain-50", 12, 64, 64, 96, 64, 1),
    ];
    for &(name, cq, cw, ch, ox, oy, cpu) in cells {
        eprintln!("\n########## TASK#39 {name} {cw}x{ch}@{ox},{oy} cpu{cpu} cq{cq} ##########");
        let matched = localize_real_speed(name, cq, cw, ch, ox, oy, cpu);
        eprintln!(
            "=== TASK#39 {name} {cw}x{ch}@{ox},{oy} cpu{cpu} cq{cq}: {} ===",
            if matched { "MATCH" } else { "DIVERGE (see first divergence above)" }
        );
    }
}
