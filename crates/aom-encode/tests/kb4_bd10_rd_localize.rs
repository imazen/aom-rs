//! KB-4 (task #31) bd10/bd12 RD-decision divergence LOCALIZER (diagnostic).
//!
//! The bd10 track's `encoder_gate_bd10_diff.rs` documents a KNOWN DIVERGENCE
//! (its `encoder_gate_bd10_bd12_multisize` docstring): with FULL-dynamic-range
//! (>8-bit) aggressive high-frequency content at LOW qindex, the port's tile
//! bytes diverge from real aomenc -- different mode/tx/partition RD winners at
//! high bit depth. That gate only exercises REPRESENTABLE (<=255) content, so
//! the divergence is described but never reproduced. This file reproduces it
//! with genuine >8-bit content and localizes the FIRST divergent block.
//!
//! Technique (identical to `decode_diff_multisb.rs::localize`, retargeted to
//! bd10/bd12 with u16 full-range content): encode the source with real aomenc,
//! run THIS PORT's `pack_tile` over the identical pixels, re-wrap the port's
//! tile bytes into a real OBU stream, then DECODE BOTH the aomenc stream and
//! the port's stream with the (bit-exact vs C) port decoder and diff the
//! per-block records:
//!   1. partition tree, node-for-node (`replay_tree`) -> first divergent
//!      `(mi_row, mi_col, bsize)` partition decision;
//!   2. else every shared leaf's mode/tx fields + per-txb `(eob, tx_type)`;
//!   3. else the first divergent reconstruction pixel (coefficient VALUES).
//! The first divergence + which field pins the divergent RD decision, which
//! narrows the RD-cost INPUT to trace.
//!
//! This file is OWNED by the encoder track (KB-4); it does NOT touch the bd10
//! track's `encoder_gate_bd10_diff.rs` or the chroma-ss track's file. Content
//! is MONOCHROME first (isolates the luma RD path: tx_search / intra_rd /
//! partition_pick) -- if mono full-range reproduces the divergence, the bug is
//! in luma RD, independent of chroma.

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
const OBU_FRAME_HEADER: u32 = 3;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4

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
        // PARTITION_SPLIT -- the only type that recurses.
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

/// Encode one MONOCHROME ALLINTRA KEY case at bit depth `bd` and `cq_level`
/// with the given u16 luma content (clamped to `[0, (1<<bd)-1]`), run the
/// port's own `pack_tile`, and return `(bytes_match, first_divergence_report)`.
/// Localizes the first divergent block when the streams diverge.
fn localize_mono(
    w: usize,
    h: usize,
    bd: u8,
    cq_level: i32,
    content: impl Fn(usize, usize) -> u16,
) -> bool {
    c::ref_init();
    let (mono, ss_x, ss_y, usage) = (true, 1usize, 1usize, 2u32);
    let maxv = (1u16 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = content(r, col).min(maxv);
        }
    }
    let u: Vec<u16> = Vec::new();
    let v: Vec<u16> = Vec::new();

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
    assert!(!p.prefix.show_existing_frame);
    assert_eq!(p.prefix.frame_type, 0);
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "single-tile envelope only");

    // ---- OUR OWN pipeline (identical config to the e2e gate) ----
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
    // EXACT config parity with the byte-exact bd10 gate
    // (`encoder_gate_bd10_diff.rs`): SpeedFeatures-derived tx-type policy +
    // hog/rect levels, so any divergence here is a real RD divergence, not a
    // config artifact (validated by the representable-content control below,
    // which MUST byte-match).
    let sf = SpeedFeatures::set_allintra(0, p.allow_screen_content_tools, false);
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
    let src_u_strided = vec![0u16; stride * (h + 4)];
    let src_v_strided = vec![0u16; stride * (h + 4)];

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
    let n_sb = (mi_cols / SB_MI).max(1);
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
        n_sb,
        n_sb,
        SB_MI,
        SB,
    );
    let our_tile_bytes = enc.done().to_vec();

    // ---- rewrap OUR bytes into a real OBU stream and decode BOTH ----
    // Correct divergence signal: decode BOTH streams with the bit-exact port
    // decoder and diff the per-block DECISIONS + reconstruction. (A raw
    // tile-bytes vs frame-OBU-payload compare is apples-to-oranges -- the
    // frame_payload carries the uncompressed header; the decode-both diff is
    // the ground truth for "did the RD decisions/coeffs diverge".)
    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER);
    let our_frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &our_tile_bytes, false, 0);
    let mut our_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    our_stream.extend_from_slice(seq_hdr_raw);
    our_stream.extend_from_slice(&our_frame_obu);

    let (t_real, _cfg_real, _hdr_real) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("decode of REAL aomenc bytes failed: {e}"));
    let (t_ours, _cfg_ours, _hdr_ours) =
        aom_decode::frame::decode_frame_obus_prefilter(&our_stream)
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

    // First divergent leaf FIELD (partition/mode/tx_size/uv_mode) or per-txb
    // (eob, tx_type), in decode-block order.
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
                || ob.info.uv_mode != rbk.info.uv_mode;
            let txbs_differ = ob.txbs != rbk.txbs || ob.txbs_uv != rbk.txbs_uv;
            if modes_differ || txbs_differ {
                leaf_div = Some(format!(
                    "(mi_row={}, mi_col={}) [modes_differ={modes_differ} txbs_differ={txbs_differ}]\n     real bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} txbs(eob,tt)={:?}\n     ours bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} txbs(eob,tt)={:?}",
                    rbk.mi_row,
                    rbk.mi_col,
                    rbk.bsize,
                    rbk.partition,
                    rbk.info.y_mode,
                    rbk.info.angle_delta_y,
                    rbk.info.use_filter_intra,
                    rbk.tx_size,
                    rbk.info.uv_mode,
                    rbk.txbs,
                    ob.bsize,
                    ob.partition,
                    ob.info.y_mode,
                    ob.info.angle_delta_y,
                    ob.info.use_filter_intra,
                    ob.tx_size,
                    ob.info.uv_mode,
                    ob.txbs
                ));
                break;
            }
        }
    }

    // First divergent reconstruction pixel (ground truth: identical dequant
    // coeffs <=> identical recon).
    let mut recon_div: Option<(usize, usize, u16, u16)> = None;
    'rec: for row in 0..t_real.height.min(t_ours.height) {
        for col in 0..t_real.width.min(t_ours.width) {
            let rv = t_real.recon[row * t_real.stride + col];
            let ovv = t_ours.recon[row * t_ours.stride + col];
            if rv != ovv {
                recon_div = Some((row, col, rv, ovv));
                break 'rec;
            }
        }
    }

    let decisions_match = partition_div.is_none() && leaf_div.is_none() && recon_div.is_none();
    eprintln!(
        "\n=== bd{bd} {w}x{h} mono cq{cq_level} (qindex={qindex}) === {}",
        if decisions_match {
            "MATCH (decisions + recon identical)".to_string()
        } else {
            format!(
                "DIVERGE (real tile bytes={} ours={})",
                frame_payload.len(),
                our_tile_bytes.len()
            )
        }
    );
    if decisions_match {
        return true;
    }

    // DECISIVE prediction-vs-encoder-decision test: decode C's stream with BOTH
    // the port decoder (full decode) AND the C decoder, and compare. If they
    // agree, the port's intra-prediction/reconstruction path is bit-exact vs C
    // for this exact directional/high-bd/high-gradient corner -> the divergence
    // is a pure ENCODER RD decision (coeff/quant/trellis/assembly). If they
    // DISagree, the port's prediction itself diverges from C (the residual is
    // NOT identical), which would be the root cause.
    let port_full = aom_decode::frame::decode_frame_obus(&bytes)
        .unwrap_or_else(|e| panic!("port full-decode of C stream failed: {e}"));
    let c_full = c::ref_decode_av1_kf(&bytes, w, h);
    let mut dec_div: Option<(usize, usize, u16, u16)> = None;
    'd: for row in 0..h {
        for col in 0..w {
            let pv = port_full.y[row * port_full.width + col];
            let cv = c_full.y[row * w + col];
            if pv != cv {
                dec_div = Some((row, col, pv, cv));
                break 'd;
            }
        }
    }
    match dec_div {
        Some((row, col, pv, cv)) => eprintln!(
            "  [DECODER CHECK] port-decode(C-stream) != C-decode(C-stream) at ({row},{col}): port={pv} C={cv}  <== PORT DECODER/PREDICTION diverges from C (residual NOT identical)"
        ),
        None => eprintln!(
            "  [DECODER CHECK] port-decode(C-stream) == C-decode(C-stream): port prediction/recon is bit-exact vs C for this stream -> divergence is a pure ENCODER RD decision"
        ),
    }

    if let Some((mi_row, mi_col, bsize, pr, po)) = partition_div {
        eprintln!(
            "  >>> FIRST PARTITION DIVERGENCE at (mi_row={mi_row}, mi_col={mi_col}, bsize={bsize}): \
             real=PARTITION_{} ({pr}) ours=PARTITION_{} ({po})",
            PARTITION_NAMES[pr as usize], PARTITION_NAMES[po as usize]
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

/// Full-dynamic-range aggressive-HF luma generator masked to `bd` bits: a
/// two-axis gradient (spans the whole range) XORed with a large-amplitude
/// checkerboard high-frequency term, so the intra predictor leaves LARGE
/// residuals -> large (>8-bit) coefficients -- the KNOWN-DIVERGENCE regime.
fn hf_luma(bd: u8) -> impl Fn(usize, usize) -> u16 {
    let mask = (1u32 << bd) - 1;
    move |r, cc| {
        let base = ((r * 37 + cc * 23) as u32) & mask;
        let hf = if (r ^ cc) & 1 == 1 { mask / 3 } else { 0 };
        (base ^ hf) as u16
    }
}

/// Steep diagonal ramp spanning the full bd range with a period-2 HF ripple.
fn ramp_luma(bd: u8) -> impl Fn(usize, usize) -> u16 {
    let mask = (1u32 << bd) - 1;
    move |r, cc| {
        let ramp = (((r + cc) as u32).wrapping_mul(mask / 24)) & mask;
        let hf = if (r + cc) % 2 == 0 { mask / 5 } else { 0 };
        (ramp ^ hf) as u16
    }
}

/// bd10/bd12 MONO KB-4 byte-match GATE (promoted from the localizer that
/// root-caused the KB-4 mono divergence): sweep full-dynamic-range
/// aggressive-HF content across qindex, print the per-cell MATCH/MISMATCH
/// grid, and localize the FIRST divergent block on any MISMATCH.
///
/// The formerly-diverging cells (bd10 cq12 hf, bd12 cq8 hf, bd12 cq20 hf)
/// were fixed by the OUTPUT_ENABLED tx_type_map copy semantics in
/// `encode_b_intra_dry` (encode_sb.rs): the SB-root winner walk + the pack
/// re-walk model C's single OUTPUT_ENABLED pass, whose eob-0 -> DCT_DCT
/// resets go to the frame map and never back into the stored winner maps
/// (encodeframe_utils.c:217-231). A regression here means the reset leak is
/// back: a skip-winning txb (non-DCT winner, eob 0) re-quantizes as DCT_DCT
/// with eob > 0 in the pack.
#[test]
fn kb4_gate_bd10_bd12_mono_hf_byte_match() {
    let mut any_mismatch = false;
    for &bd in &[10u8, 12] {
        for &(w, h) in &[(64usize, 64usize)] {
            for &cq in &[8i32, 12, 20] {
                let m1 = localize_mono(w, h, bd, cq, hf_luma(bd));
                let m2 = localize_mono(w, h, bd, cq, ramp_luma(bd));
                any_mismatch |= !m1 || !m2;
            }
        }
    }
    assert!(
        !any_mismatch,
        "KB-4 bd10/bd12 mono full-range aggressive-HF sweep must byte-match \
         real aomenc (the localizer output above pins the first divergent \
         block). Fixed by encode_b_intra_dry's OUTPUT_ENABLED tx_type_map \
         copy semantics — a mismatch means the eob-0 reset leak regressed."
    );
}

/// HARNESS VALIDATION CONTROL: the SAME localizer harness with REPRESENTABLE
/// (<=255) content -- byte-IDENTICAL samples encode at bd10/bd12, exactly the
/// regime the byte-exact `encoder_gate_bd10_diff.rs::encoder_gate_bd10_bd12_multisize`
/// gate asserts. This MUST byte-match: if it does, the harness config is sound
/// and the full-range MISMATCH above is a REAL RD divergence (only the content
/// range changed between this control and the sweep), not a config artifact. If
/// this control ever fails, the localizer's pack config drifted from the gate.
#[test]
fn kb4_localize_representable_control_matches() {
    // Representable generator (values in [0,255], same shape as the gate's).
    let rep_luma = |r: usize, cc: usize| -> u16 {
        let base = ((r * 37 + cc * 23) as u16) & 0xff;
        let hf = if (r ^ cc) & 1 == 1 { 21 } else { 0 };
        base ^ hf
    };
    let mut all_match = true;
    for &bd in &[10u8, 12] {
        for &(w, h) in &[(64usize, 64usize)] {
            for &cq in &[12i32, 32] {
                all_match &= localize_mono(w, h, bd, cq, rep_luma);
            }
        }
    }
    assert!(
        all_match,
        "CONTROL FAILED: representable-content bd10/12 must byte-match (like the \
         bd10 gate) -- the localizer's pack config has drifted from the gate; the \
         full-range divergence cannot be trusted until this passes"
    );
}
