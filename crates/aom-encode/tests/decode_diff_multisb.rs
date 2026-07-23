//! Multi-SB DECODE-DIFF localizer (diagnostic). This test localized the
//! steep-content cq32 divergence that the multi-SB scale gate
//! (`encoder_gate_e2e_byte_match.rs::encoder_gate_e2e_multi_sb_scale`) once
//! excluded -- now CLOSED. The tool encodes identical content with real aomenc
//! AND with this port's own pipeline (the EXACT `attempt_case_content` config
//! -- AB partitions ON, screen-content-tools from the real header), decodes
//! BOTH with the (bit-exact vs C) decoder, then compares the decoded partition
//! `tree`s node-for-node (`replay_tree`) to find the first divergent
//! `(mi_row, mi_col, bsize)` partition decision; if the trees match, it walks
//! every shared leaf's mode/tx fields and then the reconstruction pixels.
//!
//! It was this tool that pinned the last two 512x512 diagonal cells to an
//! intra-MODE divergence (real picked D45/PAETH where the port picked DC), and
//! the per-mode C-oracle trace off it showed the root cause was the stale
//! frame-init mode-rate tables (`av1_fill_mode_rates`): the port was missing
//! the `INTERNAL_COST_UPD_SB` per-superblock mode-cost update that C applies at
//! every SB, so on adapted later superblocks the RD rate for the directional
//! modes was over-charged and DC won a near-tie it should have lost. Fixed by a
//! per-SB `derive_real_costs(kf, ..)` in `pack_tile` (which now re-derives the
//! coeff AND mode tables together); all 16 multi-SB cells byte-match.
//!
//! DIAGNOSTIC (not a hard byte-match gate -- that role is the scale gate's): it
//! asserts only the structural invariants it relies on (positions stay locked
//! until the first partition-value divergence) and PRINTS the first divergence;
//! it stays in tree as the localization tool for any future multi-SB regression
//! and now reports "reconstruction planes are IDENTICAL" on the cases it once
//! flagged.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::obu_assemble::assemble_obu_frame_single_tile;
use aom_encode::pack::pack_tile;
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::tx_search::TxTypeSearchPolicy;
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

const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

/// The EXACT smooth-diagonal-ramp content from the multi-SB scale gate's
/// `families(w, h)[0]` (must reproduce the same source pixels to hit the same
/// divergence).
fn diagonal_ramp(w: usize, h: usize) -> impl Fn(usize, usize) -> u8 {
    move |r, c| (32 + (r + c) * 190 / (w + h)) as u8
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
        // PARTITION_SPLIT -- the only type that recurses into more decode_partition calls
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

/// Localize the first divergence for one (w, h, cq) mono ALLINTRA case with
/// the given luma content. Prints the first divergent partition node (or the
/// first leaf mode/tx mismatch if the trees are identical).
fn localize(w: usize, h: usize, cq_level: i32, content: impl Fn(usize, usize) -> u8) {
    c::ref_init();
    let (mono, ss_x, ss_y, usage) = (true, 1usize, 1usize, 2u32);
    let mut y = vec![128u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = u16::from(content(r, col));
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
        8,
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

    // ---- OUR OWN pipeline (identical config to attempt_case_content) ----
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
            TxTypeSearchPolicy::speed0_allintra()
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
        intra_pruning_with_hog: true,
        enable_rect_partitions: true,
        less_rectangular_check_level: i32::from(allintra),
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

    // ---- rewrap OUR bytes into a real OBU stream (real seq header, verbatim) ----
    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER);
    let our_frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &our_tile_bytes, false, 0);
    let mut our_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    our_stream.extend_from_slice(seq_hdr_raw);
    our_stream.extend_from_slice(&our_frame_obu);

    // ---- decode BOTH with the (bit-exact vs C) decoder ----
    let (t_real, _cfg_real, _hdr_real) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("decode of REAL aomenc bytes failed: {e}"));
    let (t_ours, _cfg_ours, _hdr_ours) =
        aom_decode::frame::decode_frame_obus_prefilter(&our_stream)
            .unwrap_or_else(|e| panic!("decode of OUR OWN rewrapped bytes failed: {e}"));

    eprintln!(
        "[{w}x{h} cq{cq_level} qindex={qindex}] real tile bytes={} ours={} | real tree len={} blocks={} | ours tree len={} blocks={}",
        frame_payload.len(),
        our_tile_bytes.len(),
        t_real.tree.len(),
        t_real.blocks.len(),
        t_ours.tree.len(),
        t_ours.blocks.len()
    );

    let ab_4way: Vec<(usize, i8)> = t_real
        .tree
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p >= 4)
        .map(|(i, &p)| (i, p))
        .collect();
    eprintln!(
        "SCAN real tree AB/4-way nodes: {}",
        if ab_4way.is_empty() {
            "none (only NONE/HORZ/VERT/SPLIT)".to_string()
        } else {
            format!("{ab_4way:?}")
        }
    );

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

    let mut first_div: Option<(i32, i32, usize, i8, i8)> = None;
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        assert_eq!(
            (r.0, r.1, r.2),
            (o.0, o.1, o.2),
            "positions must stay locked until the first partition divergence"
        );
        if r.3 != o.3 {
            first_div = Some((r.0, r.1, r.2, r.3, o.3));
            break;
        }
    }

    match first_div {
        Some((mi_row, mi_col, bsize, pr, po)) => {
            eprintln!(
                ">>> FIRST PARTITION DIVERGENCE at (mi_row={mi_row}, mi_col={mi_col}, bsize={bsize}): \
                 real=PARTITION_{} ({pr}) ours=PARTITION_{} ({po})",
                PARTITION_NAMES[pr as usize], PARTITION_NAMES[po as usize]
            );
        }
        None => {
            eprintln!(
                "partition trees agree on shared prefix (real_seq={} ours_seq={}); scanning leaves",
                real_seq.len(),
                ours_seq.len()
            );
            // Compare every shared leaf's mode/tx fields AND its per-txb
            // (eob, tx_type) records (`DecodedBlockKf.txbs`/`txbs_uv`, in
            // raster order) -- the latter pins the divergence to the tx_type
            // vs the coefficient VALUES (which don't appear in the record but
            // show up in the reconstruction, compared below).
            let mut found = false;
            for rb in &t_real.blocks {
                if let Some(ob) = t_ours
                    .blocks
                    .iter()
                    .find(|b| b.mi_row == rb.mi_row && b.mi_col == rb.mi_col)
                {
                    let modes_differ = ob.bsize != rb.bsize
                        || ob.partition != rb.partition
                        || ob.info.y_mode != rb.info.y_mode
                        || ob.info.angle_delta_y != rb.info.angle_delta_y
                        || ob.info.use_filter_intra != rb.info.use_filter_intra
                        || ob.tx_size != rb.tx_size
                        || ob.info.uv_mode != rb.info.uv_mode;
                    let txbs_differ = ob.txbs != rb.txbs || ob.txbs_uv != rb.txbs_uv;
                    if modes_differ || txbs_differ {
                        eprintln!(
                            ">>> FIRST LEAF MISMATCH at (mi_row={}, mi_col={}) [modes_differ={modes_differ} txbs_differ={txbs_differ}]: \
                             real bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} txbs(eob,tt)={:?} txbs_uv={:?} | \
                             ours bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} txbs(eob,tt)={:?} txbs_uv={:?}",
                            rb.mi_row,
                            rb.mi_col,
                            rb.bsize,
                            rb.partition,
                            rb.info.y_mode,
                            rb.info.angle_delta_y,
                            rb.info.use_filter_intra,
                            rb.tx_size,
                            rb.info.uv_mode,
                            rb.txbs,
                            rb.txbs_uv,
                            ob.bsize,
                            ob.partition,
                            ob.info.y_mode,
                            ob.info.angle_delta_y,
                            ob.info.use_filter_intra,
                            ob.tx_size,
                            ob.info.uv_mode,
                            ob.txbs,
                            ob.txbs_uv
                        );
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                eprintln!(
                    "no partition/leaf-field/txb (eob,tx_type) divergence found -- same partition, \
                     intra mode, tx_size AND same per-txb eob+tx_type; the byte divergence is in \
                     the coefficient VALUES (quantized-level/trellis choice) for identical \
                     tx_type -- see the reconstruction diff below for the exact block."
                );
            }

            // Reconstruction diff: identical dequantized coeffs <=> identical
            // recon. First differing luma pixel localizes the exact block whose
            // coefficient VALUES diverge (real vs ours).
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
            match recon_div {
                Some((row, col, rv, ov)) => eprintln!(
                    ">>> FIRST RECON PIXEL DIVERGENCE at luma (row={row}, col={col}) -> \
                     SB(mi_row={}, mi_col={}): real={rv} ours={ov} (coefficient-VALUE divergence \
                     in that block's txb -- trellis/rounding, since modes+tx_size+eob+tx_type all \
                     agree)",
                    (row / 64) * 16,
                    (col / 64) * 16,
                ),
                None => eprintln!(
                    "reconstruction planes are IDENTICAL (real == ours) -- if bytes still differ \
                     the divergence is purely in entropy coding of identical coeffs (unexpected)"
                ),
            }
        }
    }
}

/// Localize the cq32 continuous-tone multi-SB divergences (diagonal ramp +
/// vertical gradient, 256x256). DIAGNOSTIC -- prints the first divergent node.
#[test]
fn decode_diff_multisb_cq32() {
    eprintln!("=== diagonal ramp 256x256 cq32 ===");
    localize(256, 256, 32, diagonal_ramp(256, 256));
    eprintln!("=== vertical gradient 256x256 cq32 ===");
    localize(256, 256, 32, |_r, c| (32 + c * 190 / 256) as u8);
}
