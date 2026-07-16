//! Decode-diff investigation for the two AB-probe cases
//! (`encoder_gate_e2e_ab_attempt` in `encoder_gate_e2e_byte_match.rs`) whose
//! loop-filter level EXACTLY agrees with real aomenc ("top split (2 freqs) /
//! bottom flat" and "left flat / right split (2 freqs)") -- i.e. the two
//! cases where the header-region byte-0 confound (fixed, `aom-entropy`
//! commit `0d144b6`) AND the separate LF-level near-miss (STATUS.md's
//! "Loop-filter-level RD search ported" milestone, unrelated to AB) are BOTH
//! absent, so a mismatch can only come from tile-group (partition/mode/tx)
//! data. The other two AB-probe cases ("top flat / bottom split" and "left
//! split / right flat") still have an LF-level near-miss and mismatch INSIDE
//! the header (byte 2, before `tile_data_start`) -- not a clean read, not
//! attempted here.
//!
//! Method: identical to `decode_diff_noise_case.rs`'s (the VERT_4-finding
//! precedent) -- encode with real aomenc, bootstrap the frame header from the
//! real parse, run THIS PORT'S OWN `pack_tile` over the identical source
//! pixels, derive loop-filter level with `pick_filter_level` (confirmed
//! EXACT agreement for both cases below via `encoder_gate_e2e_ab_attempt`'s
//! own eprintln output), decode BOTH bitstreams with the already bit-exact
//! decoder (`aom_decode::frame::decode_frame_obus_prefilter`), and diff
//! `KfTileDecode::tree` (the pre-order partition-symbol sequence) index-by-
//! index -- byte offset alone doesn't localize the true first divergent
//! symbol because range-coder carry propagation can shift the visible effect
//! later than the actual diverging decision.
//!
//! **NOT asserted (diagnostic only, matching decode_diff_noise_case.rs's OWN
//! precedent before its underlying gap was fixed).** AB partitions
//! (HORZ_A/HORZ_B/VERT_A/VERT_B) are unported -- a divergence here is
//! EXPECTED, not a bug. This file exists to answer, with direct decode-side
//! evidence (not a guess), whether real aomenc's own RDO actually picked an
//! AB type on this content, and exactly where.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::{SbEncodeEnv, SbTree};
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_obu_frame_single_tile;
use aom_encode::pack::pack_tile;
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::tx_search::TxTypeSearchPolicy;
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
const SB_MI: i32 = 16; // 64px / 4

// `MI_SIZE_WIDE_B` (common_data.h) -- duplicated locally, matching
// `decode_diff_noise_case.rs`'s own established convention (not reachable
// from an external test binary).
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
        assert!(
            hdr.obu_has_size_field,
            "shim_encode_av1_kf always sets has_size_field"
        );
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

const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

/// EXACT content closures from `encoder_gate_e2e_byte_match.rs`'s
/// `encoder_gate_e2e_ab_attempt` -- must reproduce the same source pixels
/// bit-for-bit to hit the same divergence. Only the two LF-clean cases
/// (indices 0 and 3 in that test's `cases` array) are included here.
fn top_split_bottom_flat(r: usize, c: usize) -> u8 {
    if r < 32 {
        let period = if c < 32 { 4 } else { 6 };
        if (r / period + c / period) % 2 == 0 {
            80
        } else {
            176
        }
    } else {
        128
    }
}

fn left_flat_right_split(r: usize, c: usize) -> u8 {
    if c >= 32 {
        let period = if r < 32 { 4 } else { 6 };
        if (r / period + c / period) % 2 == 0 {
            80
        } else {
            176
        }
    } else {
        128
    }
}

/// Serialize the port's OWN search [`SbTree`] into the same
/// `(mi_row, mi_col, bsize, partition)` sequence the decoded-tree
/// [`replay_tree`] produces, so the search tree and the decoded tree can be
/// compared node-for-node. The two walks are deliberately structural twins:
/// each emits one entry per visited node and recurses ONLY on `SPLIT`, with the
/// identical partition-code encoding (0=NONE/Leaf, 1=HORZ, 2=VERT, 3=SPLIT,
/// 4..=7=AB, 8=HORZ_4, 9=VERT_4). Equal sequences therefore mean the packed
/// bytes decode back to exactly the partition structure the search chose --
/// the self-consistency invariant the palette-usage-flag fix restores.
#[allow(clippy::too_many_arguments)]
fn sbtree_seq(
    tree: &SbTree,
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
    let p: i8 = match tree {
        SbTree::Leaf(_) => 0,
        SbTree::Horz(_) => 1,
        SbTree::Vert(_) => 2,
        SbTree::Split(_) => 3,
        SbTree::HorzA(_) => 4,
        SbTree::HorzB(_) => 5,
        SbTree::VertA(_) => 6,
        SbTree::VertB(_) => 7,
        SbTree::Horz4(_) => 8,
        SbTree::Vert4(_) => 9,
        // Off-frame SPLIT-child placeholder — unreachable past the entry guard.
        SbTree::Absent => return,
    };
    out.push((mi_row, mi_col, bsize, p));
    if let SbTree::Split(kids) = tree {
        let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
        let subsize = get_partition_subsize(bsize, 3) as usize;
        sbtree_seq(&kids[0], mi_row, mi_col, subsize, mi_rows, mi_cols, out);
        sbtree_seq(
            &kids[1],
            mi_row,
            mi_col + hbs,
            subsize,
            mi_rows,
            mi_cols,
            out,
        );
        sbtree_seq(
            &kids[2],
            mi_row + hbs,
            mi_col,
            subsize,
            mi_rows,
            mi_cols,
            out,
        );
        sbtree_seq(
            &kids[3],
            mi_row + hbs,
            mi_col + hbs,
            subsize,
            mi_rows,
            mi_cols,
            out,
        );
    }
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
        // PARTITION_SPLIT
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

/// Runs one AB-probe case end-to-end (real-encode, bootstrap header, our own
/// pack_tile + pick_filter_level, decode both with the shared decoder) and
/// prints the first partition-tree divergence, if any. Mirrors
/// `decode_diff_noise_case.rs::decode_diff_pseudo_random_noise_case` almost
/// verbatim -- see that file for the full method writeup.
fn run_one(name: &str, content: impl Fn(usize, usize) -> u8) {
    eprintln!("=== decode-diff AB-probe case: {name} ===");
    c::ref_init();
    let (w, h, mono, ss_x, ss_y, usage, cq_level) =
        (64usize, 64usize, true, 1usize, 1usize, 2u32, 32i32);

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
    let real_bit_len = rb.bit_position();
    assert!(!p.prefix.show_existing_frame);
    assert_eq!(p.prefix.frame_type, 0);
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    assert_eq!(tiles_log2, 0, "single-tile envelope only");

    // ---- OUR OWN pipeline (identical to encoder_gate_e2e_byte_match.rs) ----
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

    const STRIDE: usize = 320;
    let mut src_y_strided = vec![0u16; STRIDE * (h + 4)];
    for r in 0..h {
        src_y_strided[r * STRIDE..r * STRIDE + w].copy_from_slice(&y[r * w..r * w + w]);
    }
    let src_u_strided = vec![0u16; STRIDE * (h + 4)];
    let src_v_strided = vec![0u16; STRIDE * (h + 4)];

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
    };
    let pick_cfg = PickFrameCfg {
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
    };
    eprintln!(
        "{name}: allow_screen_content_tools={}",
        p.allow_screen_content_tools
    );
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
    assert_eq!(trees.len(), (n_sb * n_sb) as usize);
    let our_tile_bytes = enc.done().to_vec();

    let tile_data_start = real_bit_len.div_ceil(8);
    let real_tile_bytes = &frame_payload[tile_data_start..];

    // ---- loop-filter-level: same TRUE DERIVATION the e2e harness performs
    //      (confirmed via encoder_gate_e2e_ab_attempt's own eprintln output
    //      to EXACTLY agree with the real value for both cases run here --
    //      re-derived here rather than reusing the bootstrapped value so this
    //      diagnostic exercises the identical code path production does). ----
    let mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb, SB_MI, SB);
    let lf_frame = LfSearchFrame {
        recon_y: &recon_y,
        recon_u: &recon_u,
        recon_v: &recon_v,
        src_y: &src_y_strided,
        src_u: &src_u_strided,
        src_v: &src_v_strided,
        stride: STRIDE,
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
    eprintln!(
        "{name}: DERIVED lf_level={:?} -- REAL(bootstrapped) lf_level={:?} -- {}",
        derived_lf.filter_level,
        p.loopfilter.filter_level,
        if derived_lf.filter_level == p.loopfilter.filter_level {
            "LF-LEVEL AGREES (clean read, as established by encoder_gate_e2e_ab_attempt)"
        } else {
            "LF-LEVEL DISAGREES (NOT a clean read -- this case should not have been run here)"
        }
    );
    assert_eq!(
        derived_lf.filter_level, p.loopfilter.filter_level,
        "{name}: this case was selected BECAUSE encoder_gate_e2e_ab_attempt found exact LF \
         agreement -- a disagreement here means something upstream changed; re-run the AB-probe \
         to pick a still-clean case before trusting this decode-diff's localization"
    );
    let mut p = p;
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;

    eprintln!(
        "{name}: real_tile_bytes.len()={} our_tile_bytes.len()={}",
        real_tile_bytes.len(),
        our_tile_bytes.len()
    );

    // ---- rewrap OUR bytes into a real OBU stream (real seq header, verbatim) ----
    let seq_hdr_raw = raw_obu_span(&bytes, OBU_SEQUENCE_HEADER);
    let our_frame_obu = assemble_obu_frame_single_tile(&p, tiles_log2, &our_tile_bytes, false, 0);
    let mut our_stream = Vec::with_capacity(seq_hdr_raw.len() + our_frame_obu.len());
    our_stream.extend_from_slice(seq_hdr_raw);
    our_stream.extend_from_slice(&our_frame_obu);

    // ---- decode BOTH with the (already bit-exact vs C, real_bitstream.rs) decoder ----
    let (t_real, _cfg_real, _hdr_real) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .unwrap_or_else(|e| panic!("{name}: decode of REAL aomenc bytes failed: {e}"));
    let (t_ours, _cfg_ours, _hdr_ours) =
        aom_decode::frame::decode_frame_obus_prefilter(&our_stream)
            .unwrap_or_else(|e| panic!("{name}: decode of OUR OWN rewrapped bytes failed: {e}"));

    eprintln!(
        "{name}: real tree len={} blocks={} | ours tree len={} blocks={}",
        t_real.tree.len(),
        t_real.blocks.len(),
        t_ours.tree.len(),
        t_ours.blocks.len()
    );

    // Smoking-gun scan: did real aomenc use ANY partition type this port's
    // search cannot produce (AB: 4-7; 4-way (8/9) IS ported) anywhere in the
    // tree?
    let unported_nodes: Vec<(usize, i8)> = t_real
        .tree
        .iter()
        .enumerate()
        .filter(|&(_, &p)| (4..=7).contains(&p))
        .map(|(i, &p)| (i, p))
        .collect();
    if unported_nodes.is_empty() {
        eprintln!(
            "{name}: SCAN: real aomenc's tree uses NO AB partition type anywhere -- a \
             divergence (if any) is NOT explained by missing AB support."
        );
    } else {
        eprintln!(
            "{name}: SCAN: real aomenc's tree uses {} AB node(s) this port's search cannot \
             produce: {:?}",
            unported_nodes.len(),
            unported_nodes
                .iter()
                .map(|&(i, p)| format!("tree[{i}]={}", PARTITION_NAMES[p as usize]))
                .collect::<Vec<_>>()
        );
    }

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

    eprintln!(
        "{name}: replayed real_seq.len()={} ours_seq.len()={}",
        real_seq.len(),
        ours_seq.len()
    );
    eprintln!(
        "{name}: FULL real tree dump: {}",
        real_seq
            .iter()
            .enumerate()
            .map(|(i, &(r, c, b, p))| format!(
                "[{i}](mr={r},mc={c},bs={b},{})",
                PARTITION_NAMES[p as usize]
            ))
            .collect::<Vec<_>>()
            .join(" ")
    );

    // ---- SELF-CONSISTENCY REGRESSION (palette-usage-flag desync) ----
    // The port's packed bytes MUST decode back to exactly the partition tree its
    // OWN search chose. Walk the search tree (`trees[0]`) into the same flat
    // sequence `ours_seq` was built from, then require equality.
    //
    // This is the permanent guard for the (mi_row=0, mi_col=8, BLOCK_32X32)
    // NONE-vs-SPLIT bug. This checkerboard content turns
    // `allow_screen_content_tools` ON, and `pack.rs` used to hardcode
    // `allow_palette = false` -- omitting the palette-usage flag the decoder
    // reads unconditionally for every DC-predicted 8x8..64x64 block. That
    // desynced the arithmetic coder from the very first (0,0) 32x32 leaf, so the
    // decoded tree collapsed (search len 25 vs decoded len 5) and the
    // decode-diff misread (0,8) as NONE. Threading SCT into `PackCfg` + setting
    // `kfs.allow_palette` per block (matching the decoder's `av1_allow_palette`)
    // restores the flag; the two sequences now match exactly. A regression here
    // fires the instant the write side drops/adds a symbol the read side expects.
    //
    // `trees[0]` is the whole frame: this probe is hard-wired to 64x64 == one
    // 64x64 superblock, and `ours_seq` was likewise replayed from the single
    // decoded SB tree rooted at (0,0). Assert that so a future frame-size bump
    // can't silently make this a partial-frame check.
    assert_eq!(
        trees.len(),
        1,
        "{name}: self-consistency check assumes a single 64x64 superblock",
    );
    let mut search_seq = Vec::new();
    sbtree_seq(&trees[0], 0, 0, SB, mi_rows, mi_cols, &mut search_seq);
    let fmt_seq = |seq: &[(i32, i32, usize, i8)]| {
        seq.iter()
            .enumerate()
            .map(|(i, &(r, c, b, p))| {
                format!(
                    "[{i}](mr={r},mc={c},bs={b},{})",
                    PARTITION_NAMES[p as usize]
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    eprintln!(
        "{name}: PORT SEARCH tree (len={}): {}",
        search_seq.len(),
        fmt_seq(&search_seq)
    );
    eprintln!(
        "{name}: DECODED ours tree (len={}): {}",
        ours_seq.len(),
        fmt_seq(&ours_seq)
    );
    assert_eq!(
        search_seq,
        ours_seq,
        "{name}: SELF-CONSISTENCY: the port's packed bytes did NOT decode back to the partition \
         tree its own search chose -- the palette-usage-flag desync signature (see comment \
         above): the write side dropped/added a symbol the decoder expects. search_len={} \
         decoded_len={}",
        search_seq.len(),
        ours_seq.len()
    );

    let mut first_divergence: Option<(i32, i32, usize, i8, i8)> = None;
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        if (r.0, r.1, r.2) != (o.0, o.1, o.2) {
            // Positions diverged before the partition VALUE did -- can only
            // happen once an earlier `p` divergence already changed the
            // recursion shape; report the SAME earlier index instead.
            break;
        }
        if r.3 != o.3 {
            first_divergence = Some((r.0, r.1, r.2, r.3, o.3));
            break;
        }
    }

    if let Some((mi_row, mi_col, bsize, p_real, p_ours)) = first_divergence {
        eprintln!(
            "{name}: FIRST DIVERGENCE at (mi_row={mi_row}, mi_col={mi_col}, bsize={bsize}): \
             real aomenc chose PARTITION_{} ({p_real}), ours chose PARTITION_{} ({p_ours}).",
            PARTITION_NAMES[p_real as usize], PARTITION_NAMES[p_ours as usize]
        );
    } else if real_seq.len() != ours_seq.len() {
        eprintln!(
            "{name}: partition VALUES agree on the shared prefix but tree LENGTH differs \
             (real_seq.len()={} ours_seq.len()={}).",
            real_seq.len(),
            ours_seq.len()
        );
    }

    // PORT-VS-REAL FULL-TREE GATE (ratcheted 2026-07-14). Both LF-clean AB-probe
    // cases now produce a partition tree BYTE-FOR-STRUCTURE identical to real
    // aomenc's. This asserts it permanently: the fix was the intra tx-size
    // search's `recon_intra` guard (tx_search.rs -- reconstruct a txb into the
    // recon plane ONLY when it is NOT the last/bottom-right txb, matching C's
    // `recon_intra` at tx_search.c:930-932). Leaving the LAST txb as the raw
    // PREDICTION (not the reconstruction) is what the ALLINTRA
    // `intra_rd_variance_factor` reads: for a DC-predicted (flat) small block it
    // makes the recon variance read as ~0 -> factor up to 3.0, inflating that
    // leaf's rd exactly as C does. Pre-fix the port read the reconstruction
    // (high variance -> factor 1.0), under-costing small screen-content leaves
    // and over-selecting HORZ over NONE/SPLIT at (2,12,BLOCK_8X8) (case 1, real
    // NONE) and (8,12,BLOCK_16X16) (case 2, real SPLIT). If this fires, that
    // small-block screen-content RD parity regressed.
    assert_eq!(
        ours_seq,
        real_seq,
        "{name}: PORT-VS-REAL partition tree must match real aomenc exactly \
         (recon_intra last-txb variance-factor parity). ours.len={} real.len={}",
        ours_seq.len(),
        real_seq.len()
    );
}

/// ASSERTS, for both LF-clean AB-probe cases, BOTH:
/// 1. self-consistency (the port's bytes decode back to the port's own search
///    tree) -- the permanent regression for the palette-usage-flag desync that
///    made (0,8,BLOCK_32X32) read as NONE; and
/// 2. FULL port-vs-real-aomenc partition-tree equality (`ours_seq ==
///    real_seq`).
///
/// (2) was diagnostic-only until 2026-07-14, when the trees still diverged at
/// DEEP small-block nodes (top-split: (2,12,8x8) real=NONE/ours=HORZ; left-flat:
/// (8,12,16x16) real=SPLIT/ours=HORZ). Root cause: the intra tx-size search
/// reconstructed EVERY txb into the recon plane, but C's `recon_intra`
/// (tx_search.c:930-932) leaves the LAST/bottom-right txb as the raw prediction.
/// The ALLINTRA `intra_rd_variance_factor` reads that plane; on a flat DC small
/// block the leftover reconstruction (vs C's flat prediction) collapsed the
/// factor from ~3.0 to 1.0, under-costing small screen-content leaves. With the
/// guard added (tx_search.rs) both cases' trees now match real aomenc exactly,
/// so this is a hard gate.
#[test]
fn decode_diff_ab_probe_clean_cases() {
    run_one("top split (2 freqs) / bottom flat", top_split_bottom_flat);
    run_one("left flat / right split (2 freqs)", left_flat_right_split);
}
