//! KB-7 speed-3/4 cq12 4:2:0 partition-flip LOCALIZER — now a fixed-root
//! regression witness.
//!
//! The speed-3 gate pinned 3 cq12 4:2:0 cells (the speed-4 gate 5) that
//! diverged from real aomenc via a SB-root partition flip (real =
//! BLOCK_64X64 NONE, port = split). This file reproduces the simplest cell
//! (`two-tone 64x64 420 cq12`, exactly 1 SB) at cpu-used 3 AND 4 with the
//! SAME derivation the gates use, byte-compares the port payload vs real
//! aomenc, and — on a mismatch — decodes BOTH streams with the (bit-exact
//! vs C) port decoder and prints the partition-tree + leaf-mode diff
//! (kb6_real_rd_localize technique).
//!
//! How the roots were found (2026-07-16, sibling-C RD dump at
//! /root/kb7-instr — removed after localization, per KB-2/KB-3
//! methodology): temp dumps of every pick_sb_modes leaf (rate_y/dist_y +
//! rate_uv/dist_uv), the NONE/SPLIT stage totals, the 4-way ML prune
//! inputs/scores, and the chroma-HOG mask, port vs instrumented C. Every
//! leaf RD matched C to the unit; the flips were TWO speed-feature-port
//! gaps:
//!   1. speed>=3: `av1_ml_prune_4_partition`'s OLD (`ml_model_index == 0`)
//!      NN branch was unported — C prunes HORZ_4/VERT_4 at the 32x32 nodes
//!      (old-NN int-scores, e.g. [530,-348,0,-392] thresh=max-500 -> only
//!      label 0 => both pruned); the port searched them and found a cheaper
//!      HORZ_4 (12.9M vs NONE 16.5M at child 0) -> root flipped NONE->SPLIT.
//!   2. speed>=4: the tail of set_allintra_speed_features_framesize_
//!      independent (speed_features.c:608-616) force-disables
//!      `chroma_intra_pruning_with_hog` when
//!      `prune_chroma_modes_using_luma_winner` is on; the port kept the
//!      chroma HOG live and HOG-pruned UV_V_PRED where C evaluates + picks
//!      it (58469617 vs the port's SMOOTH 58779332).
//!
//! With both roots fixed these single-cell repros byte-match and are
//! asserted; the 64-cell gates assert the full grids.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::{
    assemble_frame_obu_payload_single_tile, assemble_obu_frame_single_tile,
};
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
use aom_entropy::partition::{KfFrameContext, get_partition_subsize};
use aom_entropy::rb::ReadBitBuffer;
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16;

const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const PARTITION_NAMES: [&str; 10] = [
    "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
];
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

fn raw_obu_span(bytes: &[u8], want_type: u32) -> &[u8] {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        let (size, size_bytes) =
            aom_entropy::leb128::uleb_decode(&bytes[after_header..]).expect("valid leb128 size");
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

/// The gate's exact cell derivation (attempt_case_content_uv,
/// encoder_gate_e2e_byte_match.rs) for one synthetic 4:2:0 cell at the given
/// speed, PLUS the decode-both partition/leaf diff. Returns `true` on a full
/// e2e byte match.
fn localize_cell(w: usize, h: usize, cq_level: i32, speed: i32, name: &str) -> bool {
    c::ref_init();
    let content = |r: usize, c: usize| -> u8 {
        match name {
            "two-tone" => {
                if c < w / 2 {
                    72
                } else {
                    168
                }
            }
            "vgrad" => (32 + c * 190 / w) as u8,
            "diag" => (32 + (r + c) * 190 / (w + h)) as u8,
            other => panic!("unknown {other:?}"),
        }
    };
    let uv_content = |r: usize, c: usize| -> u8 { (60 + (r * 7 + c * 3) % 80) as u8 };
    let (ss_x, ss_y) = (1usize, 1usize);
    let usage = 2u32;
    let cpu_used = speed;

    let mut y = vec![128u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = u16::from(content(r, col));
        }
    }
    let (cw, ch) = ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y);
    let mut u = vec![128u16; cw * ch];
    let mut v = vec![128u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            let val = u16::from(uv_content(r, col));
            u[r * cw + col] = val;
            v[r * cw + col] = val;
        }
    }

    let bytes = c::ref_encode_av1_kf(
        &y,
        &u,
        &v,
        w,
        h,
        8,
        false,
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
    // Persist the real stream for the sibling-C harness byte-check.
    if let Ok(dir) = std::env::var("KB7_OUT_DIR") {
        let tag = format!("{name}{w}_cq{cq_level}_cpu{speed}");
        std::fs::write(format!("{dir}/kb7_real_{tag}.av1"), &bytes).unwrap();
    }

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
    let real_bit_len = rb.bit_position();
    assert_eq!(p.prefix.frame_type, 0);
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0);
    let qindex = p.quant.base_qindex;
    eprintln!(
        "=== {name} {w}x{h} 420 cq{cq_level} cpu{speed}: qindex={qindex} \
         screen_content={} ===",
        p.prefix.allow_screen_content_tools
    );

    let tile_data_start = real_bit_len.div_ceil(8);
    let real_tile_bytes = &frame_payload[tile_data_start..];

    // ---- port pipeline (parity with attempt_case_content_uv) ----
    let bd: u8 = 8;
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

    let stride = 320.max(w + 4);
    let mut src_y_strided = vec![0u16; stride * (h + 4)];
    for r in 0..h {
        src_y_strided[r * stride..r * stride + w].copy_from_slice(&y[r * w..r * w + w]);
    }
    let mut src_u_strided = vec![0u16; stride * (h + 4)];
    let mut src_v_strided = vec![0u16; stride * (h + 4)];
    for r in 0..ch {
        src_u_strided[r * stride..r * stride + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
        src_v_strided[r * stride..r * stride + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
    }

    let sf = SpeedFeatures::set_allintra(speed, p.allow_screen_content_tools, false);
    let env = SbEncodeEnv {
        sb_size: SB,
        mi_rows,
        mi_cols,
        tile_row_start: 0,
        tile_col_start: 0,
        tile_row_end: 1 << 16,
        tile_col_end: 1 << 16,
        monochrome: false,
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
    assert_eq!(trees.len(), (n_sb * n_sb) as usize);
    let our_tile_bytes = enc.done().to_vec();

    // LF derivation (parity with the gate; speed>=4 => non-dual).
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
        monochrome: false,
        mi: &mi_grid,
        mi_rows,
        mi_cols,
        delta_lf_present: false,
    };
    let derived_lf = pick_filter_level(&lf_frame, allintra, 0, allintra && speed >= 4);
    let mut p2 = p.clone();
    p2.loopfilter.filter_level = derived_lf.filter_level;
    p2.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p2.loopfilter.filter_level_v = derived_lf.filter_level_v;
    let our_payload = assemble_frame_obu_payload_single_tile(&p2, tiles_log2, &our_tile_bytes);

    let matched = our_payload == frame_payload;
    eprintln!(
        "  real tile={}B ours tile={}B -> {}",
        real_tile_bytes.len(),
        our_tile_bytes.len(),
        if matched { "BYTE MATCH" } else { "DIVERGE" }
    );

    // ---- decode BOTH streams; diff partition trees + leaf modes ----
    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER);
    let our_frame_obu = assemble_obu_frame_single_tile(&p2, tiles_log2, &our_tile_bytes, false, 0);
    let mut our_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    our_stream.extend_from_slice(seq_hdr_raw);
    our_stream.extend_from_slice(&our_frame_obu);
    if let Ok(dir) = std::env::var("KB7_OUT_DIR") {
        let tag = format!("{name}{w}_cq{cq_level}_cpu{speed}");
        std::fs::write(format!("{dir}/kb7_port_{tag}.av1"), &our_stream).unwrap();
    }

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
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        if (r.0, r.1, r.2) != (o.0, o.1, o.2) {
            break;
        }
        if r.3 != o.3 {
            eprintln!(
                "  >>> FIRST PARTITION DIVERGENCE at (mi_row={}, mi_col={}, bsize={}): \
                 real=PARTITION_{} ours=PARTITION_{}",
                r.0, r.1, r.2, PARTITION_NAMES[r.3 as usize], PARTITION_NAMES[o.3 as usize]
            );
            break;
        }
    }
    // Leaf-level dump over the whole frame (both trees' leaves) for context.
    for (tag, blocks) in [("real", &t_real.blocks), ("ours", &t_ours.blocks)] {
        for b in blocks.iter() {
            eprintln!(
                "  {tag} leaf ({},{}) bsize={} part={} y_mode={} adly={} uv_mode={} aduv={} \
                 tx={} eobs_y={:?} eobs_uv={:?}",
                b.mi_row,
                b.mi_col,
                b.bsize,
                b.partition,
                b.info.y_mode,
                b.info.angle_delta_y,
                b.info.uv_mode,
                b.info.angle_delta_uv,
                b.tx_size,
                b.txbs.iter().map(|t| t.0).collect::<Vec<_>>(),
                b.txbs_uv.iter().map(|t| t.0).collect::<Vec<_>>()
            );
        }
    }
    matched
}

/// The simplest former KB-7 pin: `two-tone 64x64 420 cq12` at cpu-used=3
/// (1 SB). Byte-match asserted (regression witness for the level-3
/// OLD-model 4-way ML prune); on failure the decode-both partition/leaf
/// diff above the assertion localizes the flip.
#[test]
fn kb7_localize_twotone64_cq12_speed3() {
    let matched = localize_cell(64, 64, 12, 3, "two-tone");
    assert!(
        matched,
        "KB-7 speed-3 repro (two-tone 64x64 420 cq12) re-diverged — the level-3 \
         OLD-model 4-way ML prune (part4_prune.rs) regressed; see the decode-both \
         diff above"
    );
}

/// The same cell at cpu-used=4 — exercises BOTH KB-7 roots (the 4-way ML
/// prune AND the speed>=4 chroma-HOG disable). Byte-match asserted.
#[test]
fn kb7_localize_twotone64_cq12_speed4() {
    let matched = localize_cell(64, 64, 12, 4, "two-tone");
    assert!(
        matched,
        "KB-7 speed-4 repro (two-tone 64x64 420 cq12) re-diverged — the level-3 \
         OLD-model 4-way ML prune (part4_prune.rs) or the speed>=4 \
         chroma_intra_pruning_with_hog disable tail (partition_pick.rs / \
         speed_features.rs) regressed; see the decode-both diff above"
    );
}
