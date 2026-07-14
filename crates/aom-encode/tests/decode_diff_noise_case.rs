//! Task 1 (decode-diff), ORIGINAL FINDING (now fixed, kept as a regression
//! gate): this test originally isolated the exact first divergent encode
//! DECISION in the `encoder_gate_e2e_textured_attempt` "pseudo-random
//! noise" case (the one textured case out of 7 that did NOT byte-match real
//! aomenc -- first mismatched byte at offset 1139 of ~1520-1536 total
//! tile-group bytes). The decode-diff below found the first divergence at
//! (mi_row=8, mi_col=8, bsize=BLOCK_16X16): real aomenc chose
//! `PARTITION_VERT_4`, this port's search chose `PARTITION_VERT` (VERT_4
//! wasn't in its candidate set) -- confirming the "missing AB/4-way
//! partition types" gap documented in STATUS.md. HORZ_4/VERT_4 (with their
//! real ML prune, `av1_ml_prune_4_partition`) are now ported
//! (`crates/aom-encode/src/partition_pick.rs`'s 4-way stage,
//! `part4_prune.rs`) and this exact case now byte-matches end-to-end (see
//! `encoder_gate_e2e_byte_match.rs`'s `encoder_gate_e2e_textured_attempt`:
//! 7/7, up from 6/7). AB (HORZ_A/B, VERT_A/B) is still unported --
//! `STATUS.md`'s MISSING list is the honest source of truth on remaining
//! partition-type coverage.
//!
//! Method: encode the SAME noise content with real aomenc (`ref_encode_av1_kf`)
//! and with this port's own pipeline (`rd_pick_partition_real` + `pack_tile`,
//! exactly as `encoder_gate_e2e_byte_match.rs` does), then DECODE BOTH
//! bitstreams -- real bytes as-is, and our own tile bytes rewrapped into a
//! real OBU_FRAME (reusing the real, already-verified-byte-identical
//! sequence-header OBU) -- with the SAME (already bit-exact vs the C
//! decoder, `real_bitstream.rs`) decoder,
//! `aom_decode::frame::decode_frame_obus_prefilter`. Both decodes expose
//! `KfTileDecode::tree` (the pre-order partition-symbol sequence, EVERY
//! visited node) and `KfTileDecode::blocks` (per-leaf mode/tx records).
//!
//! Byte offset 1139 does NOT directly localize the divergent DECISION: the
//! arithmetic range coder mixes many symbols into each byte and carry
//! propagation can shift the visible byte effect later than the true first
//! diverging symbol. Comparing the two decodes' `tree` sequences index-by-
//! index instead finds the true first divergent partition decision
//! structurally: both trees start at the identical SB root (mi_row=0,
//! mi_col=0, bsize=BLOCK_64X64) and — by construction of the pre-order DFS
//! `decode_partition` performs (verified by direct reading,
//! `aom-decode/src/lib.rs:1847-1965`) — stay position-locked entry-for-entry
//! for as long as the partition VALUES keep agreeing (only `PARTITION_SPLIT`
//! recurses into further `decode_partition` calls that push more `tree`
//! entries; every other type is a `tree` leaf). The first index where the
//! partition values differ is therefore the true first divergent decision,
//! unambiguously, with its exact (mi_row, mi_col, bsize) spatial position.
//!
//! NOW a hard regression gate (was a diagnostic while the divergence was
//! still open): asserts the two decoded partition `tree`s are IDENTICAL
//! (not just non-divergent in some prefix) -- if a future change reopens
//! this gap (e.g. an AB port accidentally changing the 4-way RD budget so
//! VERT_4 stops winning here), this test fails loudly instead of silently
//! reverting to "diagnostic, not asserted."

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
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

// `MI_SIZE_WIDE_B` (common_data.h) -- duplicated locally (pub(crate) in
// aom_encode::tx_search, not reachable from an external test binary; see
// `encoder_gate_e2e_byte_match.rs`'s own `walk_obus` comment on this test
// family's established convention of small local duplication).
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

/// Same OBU walk, but returns the RAW byte span (header + leb128 size +
/// payload) for the first OBU of the given type -- what's needed to splice a
/// real sequence-header OBU verbatim in front of a reassembled OBU_FRAME.
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

/// The EXACT pseudo-random noise content from
/// `encoder_gate_e2e_textured_attempt` (must reproduce the same source
/// pixels bit-for-bit to hit the same divergence).
fn noise_content(r: usize, c: usize) -> u8 {
    let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (64 + (x % 129)) as u8
}

/// One `KfTileDecode.tree` entry replayed with its spatial position: the
/// exact recursion `decode_partition` performs (traced directly from
/// `aom-decode/src/lib.rs:1847-1965`) -- only `PARTITION_SPLIT` recurses
/// into further `tree`-pushing calls; every other partition type is a
/// `tree` leaf (its sub-blocks are coded via `decode_block`, which does not
/// push to `tree`).
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

#[test]
fn decode_diff_pseudo_random_noise_case() {
    c::ref_init();
    let (w, h, mono, ss_x, ss_y, usage, cq_level) =
        (64usize, 64usize, true, 1usize, 1usize, 2u32, 32i32);

    let mut y = vec![128u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = u16::from(noise_content(r, col));
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
        enable_1to4_partitions: true, // the true aomenc default (unset --enable-1to4-partitions)
    };
    let pack_cfg = aom_encode::pack::PackCfg {
        enable_filter_intra: s.enable_filter_intra,
        tx_mode_is_select: p.tx_mode_select,
        signal_gate: qindex > 0,
        allow_update_cdf: !p.prefix.disable_cdf_update,
        base_qindex: qindex,
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
    eprintln!(
        "noise case: real_tile_bytes.len()={} our_tile_bytes.len()={}",
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
        .unwrap_or_else(|e| panic!("decode of REAL aomenc bytes failed: {e}"));
    let (t_ours, _cfg_ours, _hdr_ours) =
        aom_decode::frame::decode_frame_obus_prefilter(&our_stream)
            .unwrap_or_else(|e| panic!("decode of OUR OWN rewrapped bytes failed: {e}"));

    eprintln!(
        "real tree len={} blocks={} | ours tree len={} blocks={}",
        t_real.tree.len(),
        t_real.blocks.len(),
        t_ours.tree.len(),
        t_ours.blocks.len()
    );

    // Smoking-gun scan: did real aomenc use ANY partition type this port's
    // search cannot produce (AB: 4-7, 4-way: 8-9) anywhere in the tree?
    let ab_4way_nodes: Vec<(usize, i8)> = t_real
        .tree
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p >= 4)
        .map(|(i, &p)| (i, p))
        .collect();
    if ab_4way_nodes.is_empty() {
        eprintln!(
            "SCAN: real aomenc's tree uses ONLY NONE/SPLIT/HORZ/VERT (no AB/4-way anywhere) \
             -- the divergence is NOT explained by missing partition types alone."
        );
    } else {
        eprintln!(
            "SCAN: real aomenc's tree uses {} AB/4-way node(s) this port's search cannot \
             produce: {:?}",
            ab_4way_nodes.len(),
            ab_4way_nodes
                .iter()
                .map(|&(i, p)| format!("tree[{i}]={}", PARTITION_NAMES[p as usize]))
                .collect::<Vec<_>>()
        );
    }

    // Replay both trees to (mi_row, mi_col, bsize, partition) with position.
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
        "replayed real_seq.len()={} ours_seq.len()={}",
        real_seq.len(),
        ours_seq.len()
    );

    let mut first_divergence: Option<(i32, i32, usize, i8, i8)> = None;
    for (r, o) in real_seq.iter().zip(ours_seq.iter()) {
        assert_eq!(
            (r.0, r.1, r.2),
            (o.0, o.1, o.2),
            "positions must stay locked until the first `p` divergence (replay bug if not)"
        );
        if r.3 != o.3 {
            first_divergence = Some((r.0, r.1, r.2, r.3, o.3));
            break;
        }
    }

    match first_divergence {
        None => {
            eprintln!(
                "NO partition-tree divergence found (real_seq.len()={} ours_seq.len()={}) -- \
                 cross-checking per-leaf mode/tx for full confirmation.",
                real_seq.len(),
                ours_seq.len()
            );
            // Hard gate (module docs): the trees must be the SAME LENGTH, not
            // just non-divergent in the shared prefix -- a length mismatch
            // with an otherwise-matching prefix would mean one side's tree is
            // a strict subset (e.g. this port stopped recursing early), which
            // `first_divergence == None` alone would NOT catch.
            assert_eq!(
                real_seq.len(),
                ours_seq.len(),
                "partition tree LENGTH must match exactly, not just agree on a shared prefix"
            );
            assert_eq!(
                t_real.blocks.len(),
                t_ours.blocks.len(),
                "decoded leaf-block COUNT must match exactly"
            );
            let mut leaf_mismatch: Option<(i32, i32)> = None;
            #[allow(clippy::collapsible_if)]
            'outer: for rb in &t_real.blocks {
                if let Some(ob) = t_ours
                    .blocks
                    .iter()
                    .find(|b| b.mi_row == rb.mi_row && b.mi_col == rb.mi_col)
                {
                    if ob.bsize != rb.bsize
                        || ob.partition != rb.partition
                        || ob.info.y_mode != rb.info.y_mode
                        || ob.info.angle_delta_y != rb.info.angle_delta_y
                        || ob.info.use_filter_intra != rb.info.use_filter_intra
                        || ob.tx_size != rb.tx_size
                        || ob.info.uv_mode != rb.info.uv_mode
                    {
                        eprintln!(
                            "LEAF MISMATCH at (mi_row={}, mi_col={}): real bsize={} partition={} \
                             y_mode={} angle_delta_y={} use_fi={} tx_size={} uv_mode={} | \
                             ours bsize={} partition={} y_mode={} angle_delta_y={} use_fi={} \
                             tx_size={} uv_mode={}",
                            rb.mi_row,
                            rb.mi_col,
                            rb.bsize,
                            rb.partition,
                            rb.info.y_mode,
                            rb.info.angle_delta_y,
                            rb.info.use_filter_intra,
                            rb.tx_size,
                            rb.info.uv_mode,
                            ob.bsize,
                            ob.partition,
                            ob.info.y_mode,
                            ob.info.angle_delta_y,
                            ob.info.use_filter_intra,
                            ob.tx_size,
                            ob.info.uv_mode
                        );
                        leaf_mismatch = Some((rb.mi_row, rb.mi_col));
                        break 'outer;
                    }
                }
            }
            assert!(
                leaf_mismatch.is_none(),
                "leaf-level (mode/tx) mismatch found at {leaf_mismatch:?} even though the \
                 partition trees matched exactly -- see the LEAF MISMATCH eprintln above for \
                 the field-by-field diff; this is a regression, not the expected state \
                 (encoder_gate_e2e_textured_attempt's pseudo-random-noise case now byte-matches \
                 end-to-end, so this decode-diff must find zero divergence of any kind)"
            );
            eprintln!(
                "CONFIRMED: partition trees AND every shared leaf's mode/tx fields match \
                 exactly -- the 4-way partition port (HORZ_4/VERT_4 + av1_ml_prune_4_partition) \
                 fully resolves this case's original VERT_4 divergence."
            );
        }
        Some((mi_row, mi_col, bsize, p_real, p_ours)) => {
            panic!(
                "REGRESSION: partition-tree divergence reappeared at (mi_row={mi_row}, \
                 mi_col={mi_col}, bsize={bsize}) -- real chose PARTITION_{} ({p_real}), ours \
                 chose PARTITION_{} ({p_ours}). This case was fixed by the 4-way partition port \
                 (HORZ_4/VERT_4 + av1_ml_prune_4_partition, crates/aom-encode/src/\
                 partition_pick.rs + part4_prune.rs) -- if this fires, something (e.g. a later \
                 AB-partition change to the RD budget flowing into the 4-way stage) reopened the \
                 gap. Re-run the SCAN/decode-diff logic above for a fresh root-cause read.",
                PARTITION_NAMES[p_real as usize], PARTITION_NAMES[p_ours as usize]
            );
        }
    }
}
