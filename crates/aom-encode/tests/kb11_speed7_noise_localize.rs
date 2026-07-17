//! KB-11 pinned-open characterization + localization: the speed-7
//! (`--cpu-used=7`, VAR_BASED_PARTITION) **noise 64x64 cq63** divergence —
//! the same two cells (mono + 4:2:0) that are pinned open at speed 6, and
//! (LOCALIZED below) the SAME root: KB-10's multi-feature winner-pass
//! tx-size near-tie at qindex 255.
//!
//! Method (the decode_diff_noise_case.rs shape): encode the SAME noise
//! content with real `aomenc --cpu-used=7` and with this port's own speed-7
//! pipeline (`choose_var_based_partitioning_key` + `rd_use_partition_real`
//! via `pack_tile`), decode BOTH streams with the (bit-exact vs C) port
//! decoder, and compare the decoded partition `tree` symbol sequences and
//! per-leaf `blocks` records — the true first divergent DECISION, not the
//! first divergent byte (the range coder smears symbols across bytes) —
//! plus the search's own INTENDED SbTree winners (what the pack was given).
//!
//! **LOCALIZED (2026-07-17, this harness's own output):**
//! - mono cq63: the variance tree fixes the SAME shape real uses (SPLIT +
//!   four NONE-32x32s — decoded trees IDENTICAL, 5 nodes) and every decoded
//!   leaf's mode record matches (DC, adly 0, skip 0) EXCEPT the (mi 8,0)
//!   leaf's txbs: real codes its single TX_32X32 txb as eob 0; the port's
//!   intended winner (SbTree dump) carries **tx_size TX_16X16 where real
//!   keeps TX_32X32** — children 0/1/3 tx_size 3 (32x32), child 2 tx_size 2.
//!   This is EXACTLY KB-10's pinned-open speed-6 near-tie ("the (mi 8,0)
//!   32x32 leaf's WINNER-pass uniform tx-size sweep picks TX_16X16 over
//!   TX_32X32 by a 0.19% rd margin where real keeps 32"): the speed-7 leaf
//!   mode search is the SAME machinery (only the partition source changed,
//!   and both produce the same fixed tree here), so the near-tie reproduces
//!   verbatim. The frame codes tx_mode LARGEST post-hoc (real's
//!   txb_split_count==0 collapse), so the 16x16 tx plan desyncs the parse —
//!   the decoded "eob 50" on our stream is the desync artifact, as is the
//!   4:2:0 cell's apparent "tree divergence at (mi 8,8)" (the mis-parse
//!   trips before that partition symbol; mono's leaf diff is the true
//!   signal).
//! - NOT a speed-7 (VBP/rd_use_partition) defect: cq32 on the same content
//!   byte-matches through a 1560-byte deep tree (the control below), and
//!   the KB-10 next step (sibling-C RD dump of the winner sweep at (8,0))
//!   closes BOTH speeds' cells at once.
//!
//! The pinned test asserts the byte divergence is still PRESENT (fails the
//! moment the cells start matching -> promote to byte-match asserts in
//! `encoder_gate_speed7_noise_flatuv_allintra`). cq12/cq32/cq48 on the same
//! content are hard-asserted byte-matches in that gate — qindex 255 is the
//! only diverging quality on this content, exactly the KB-10 pattern.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::obu_assemble::assemble_obu_frame_single_tile;
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
const OBU_FRAME_HEADER: u32 = 3;
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
        assert!(hdr.obu_has_size_field);
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
    2 * ((px + 7) >> 3)
}

/// The speed-6 noise-extension content (encoder_gate_e2e_byte_match.rs).
fn noise_content(r: usize, c: usize) -> u8 {
    let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (64 + (x % 129)) as u8
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

/// Returns true when the port's speed-7 encode byte-matches real
/// `aomenc --cpu-used=7` on noise/flat-uv 64x64 at `cq_level`; on a byte
/// divergence, decodes both streams and prints the structural localization
/// (first divergent partition decision + first differing leaf record).
fn run_and_localize(cq_level: i32, mono: bool) -> bool {
    c::ref_init();
    let (w, h, ss_x, ss_y, usage) = (64usize, 64usize, 1usize, 1usize, 2u32);
    let speed = 7i32;

    let mut y = vec![128u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = u16::from(noise_content(r, col));
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
    };
    let u = vec![128u16; cw * ch];
    let v = vec![128u16; cw * ch];

    let bytes = c::ref_encode_av1_kf(
        &y,
        &u,
        &v,
        w,
        h,
        8,
        mono,
        ss_x as i32,
        ss_y as i32,
        cq_level,
        /*cpu_used=*/ 7,
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
        .find(|(t, _)| *t == OBU_FRAME_HEADER || *t == OBU_FRAME)
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
    assert_eq!(tiles_log2, 0, "single-tile envelope only");

    // ---- the port's own speed-7 pipeline (identical derivation to
    //      encoder_gate_e2e_byte_match.rs's attempt_case_content_uv) ----
    let bd: u8 = 8;
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
        EncMode::Allintra,
    );

    const STRIDE: usize = 320;
    let mut src_y_strided = vec![0u16; STRIDE * (h + 4)];
    for r in 0..h {
        src_y_strided[r * STRIDE..r * STRIDE + w].copy_from_slice(&y[r * w..r * w + w]);
    }
    let mut src_u_strided = vec![0u16; STRIDE * (h + 4)];
    let mut src_v_strided = vec![0u16; STRIDE * (h + 4)];
    if !mono {
        for r in 0..ch {
            src_u_strided[r * STRIDE..r * STRIDE + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
            src_v_strided[r * STRIDE..r * STRIDE + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
        }
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
        monochrome: mono,
        ss_x,
        ss_y,
        bd,
        lossless: p.coded_lossless,
        reduced_tx_set_used: p.reduced_tx_set_used,
        disable_edge_filter: !s.enable_intra_edge_filter,
        filter_type: 0,
        stride: STRIDE,
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
        intra_pruning_with_hog: sf.intra_pruning_with_hog != 0,
        enable_rect_partitions: true,
        less_rectangular_check_level: sf.less_rectangular_check_level,
        max_partition_size: sf.default_max_partition_size.min(15).min(SB),
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
        allow_screen_content_tools: p.allow_screen_content_tools,
        delta_q_present: false,
        delta_q_res: 0,
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
    let our_tile_bytes = enc.done().to_vec();
    // Dump the search's own intended winners for the SB (KB-11 evidence:
    // what the pack was GIVEN, vs what the coded bytes decode back to).
    if cq_level == 63 {
        for t in &trees {
            if let aom_encode::encode_sb::SbTree::Split(kids) = t {
                for (i, k) in kids.iter().enumerate() {
                    if let aom_encode::encode_sb::SbTree::Leaf(w) = k {
                        eprintln!(
                            "kb11 intended child {i}: bsize {} mode {} adly {} tx_size {} \
                             skip_txfm {:?} uv {}",
                            w.bsize, w.mode, w.angle_delta_y, w.tx_size, w.skip_txfm, w.uv_mode
                        );
                    } else {
                        eprintln!("kb11 intended child {i}: non-leaf {k:?}");
                    }
                }
            } else {
                eprintln!("kb11 intended root: non-split");
            }
        }
    }

    let tile_data_start = real_bit_len.div_ceil(8);
    let real_tile_bytes = &frame_payload[tile_data_start..];
    let fmt = if mono { "mono" } else { "420" };
    eprintln!(
        "kb11 noise 64x64 {fmt} cq{cq_level} (qindex {qindex}): real tile {}B, ours {}B",
        real_tile_bytes.len(),
        our_tile_bytes.len()
    );
    if real_tile_bytes == our_tile_bytes.as_slice() {
        eprintln!("kb11 noise {fmt} cq{cq_level}: TILE BYTES MATCH");
        return true;
    }

    // ---- structural localization: decode both, diff tree + blocks ----
    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER);
    let our_frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &our_tile_bytes, false, 0);
    let mut our_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    our_stream.extend_from_slice(seq_hdr_raw);
    our_stream.extend_from_slice(&our_frame_obu);

    let (t_real, _, _) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("decode of REAL aomenc bytes failed: {e}"));
    let (t_ours, _, _) = aom_decode::frame::decode_frame_obus_prefilter(&our_stream)
        .unwrap_or_else(|e| panic!("decode of OUR rewrapped bytes failed: {e}"));

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
    let mut tree_diverged = false;
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        if (r.0, r.1, r.2) != (o.0, o.1, o.2) || r.3 != o.3 {
            eprintln!(
                "kb11 {fmt} cq{cq_level}: FIRST TREE DIVERGENCE at (mi {},{}) bsize {}: \
                 real PARTITION_{} vs ours PARTITION_{}",
                r.0, r.1, r.2, PARTITION_NAMES[r.3 as usize], PARTITION_NAMES[o.3 as usize]
            );
            tree_diverged = true;
            break;
        }
    }
    if !tree_diverged {
        eprintln!(
            "kb11 {fmt} cq{cq_level}: partition trees IDENTICAL ({} nodes; lens {}/{}) — \
             the divergence is a LEAF decision",
            real_seq.len(),
            real_seq.len(),
            ours_seq.len()
        );
        eprintln!(
            "kb11 {fmt} cq{cq_level}: block counts real {} vs ours {}",
            t_real.blocks.len(),
            t_ours.blocks.len()
        );
        for (rb, ob) in t_real.blocks.iter().zip(t_ours.blocks.iter()) {
            if rb != ob {
                eprintln!(
                    "kb11 {fmt} cq{cq_level}: FIRST LEAF DIFF\n  real: {rb:?}\n  ours: {ob:?}"
                );
                break;
            }
        }
    }
    false
}

/// The two KB-11 pinned-open cells (mono + 4:2:0 noise 64x64 cq63) must
/// still DIVERGE — this test FAILS the moment either starts matching
/// (promote it + the noise gate's cq63 arm to hard byte-match asserts).
/// The localization prints above document the current structural diff.
#[test]
fn kb11_speed7_noise_cq63_pinned_open() {
    let mono_match = run_and_localize(63, true);
    let s420_match = run_and_localize(63, false);
    assert!(
        !mono_match && !s420_match,
        "the KB-11 pinned-open speed-7 noise cq63 cell(s) started BYTE-MATCHING real aomenc \
         (mono match: {mono_match}, 420 match: {s420_match}) — promote the characterization \
         to full byte-match asserts and close the KB-11 open item"
    );
}

/// Control: the same content at cq32 byte-matches (the localization helper
/// itself is sound — divergence at cq63 is content+qindex-specific, not a
/// harness artifact).
#[test]
fn kb11_speed7_noise_cq32_control_matches() {
    assert!(
        run_and_localize(32, true),
        "noise mono cq32 must byte-match at speed 7 (it does in the noise gate)"
    );
}
