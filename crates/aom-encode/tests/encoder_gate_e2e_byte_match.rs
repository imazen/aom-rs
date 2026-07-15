//! Task 3: the headline encoder-gate deliverable -- attempt a TRUE end-to-end
//! byte match against real aomenc (`shim_encode_av1_kf`) for the smallest
//! single-SB all-intra frame, where the coded PAYLOAD (partitions, modes,
//! tx, coefficients) comes from THIS PORT'S OWN search + pack pipeline
//! (`rd_pick_partition_real` + `pack_tile`, driven by `derive_real_costs`'s
//! now-FULL per-txs_ctx coeff costs -- the Task 1 deliverable), not copied
//! from the real stream. Only the frame header is bootstrapped verbatim from
//! the real parse (loop-filter-level search / CDEF-strength search / the
//! qindex-from-cq-level mapping are not ported -- see the module docs on
//! `frame_header_matches_real_encoder.rs` for the same, already-documented,
//! bootstrap boundary). The wrapping (`assemble_frame_obu_payload_single_tile`)
//! is Task 2's already-verified assembly.
//!
//! Method: encode a real minimal flat KEY frame via `ref_encode_av1_kf`
//! (`enable_cdef=false, enable_restoration=false`), parse its sequence +
//! frame header (same transcription as `frame_header_matches_real_encoder
//! .rs`/`tile_group_obu_matches_real_encoder.rs`), then build THIS PORT'S
//! OWN encode pipeline from values read off that REAL parsed header
//! (qindex, tile info, tx-mode-select, cdf-update flag, ...) and the REAL
//! sequence header (filter-intra enable, edge-filter enable, ...) -- run
//! `pack_tile` over the IDENTICAL flat source pixels real aomenc encoded,
//! assemble the result, and compare byte-for-byte against the complete real
//! `OBU_FRAME` payload.
//!
//! **Result: [`encoder_gate_e2e_attempt`] achieves a TRUE end-to-end byte
//! match on all 3 flat-content cases (mono/4:2:0 ALLINTRA + 4:2:0 GOOD).**
//! This is the smallest possible case (near-empty 1-byte tile payload --
//! every txb is EOB=0) and does not by itself demonstrate coeff-cost
//! decision parity on frames with real texture; [`encoder_gate_e2e_textured_attempt`]
//! is a harder, unasserted, exploratory attempt at that.
//!
//! **Honest labelling is mandatory here — see the per-case `eprintln!` and
//! the final assertion message for exactly what's bootstrapped vs derived.**

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
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
use aom_entropy::partition::KfFrameContext;
use aom_entropy::rb::ReadBitBuffer;
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME_HEADER: u32 = 3;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4

/// Split a real AV1 byte stream into `(obu_type, payload)` pairs. Duplicated
/// per this test family's established convention (see
/// `frame_header_matches_real_encoder.rs`'s own comment on why).
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

/// Deterministic pseudo-random "noise" (xorshift, no external RNG
/// dependency) -- the SAME content family as
/// `encoder_gate_e2e_textured_attempt`'s "pseudo-random noise" case
/// (duplicated as a named `fn` here so [`encoder_gate_e2e_nonzero_lf_sweep`]
/// can reuse it across multiple `cq_level`s in a `&[(.., fn(..) -> u8)]`
/// table).
fn noise_content(r: usize, c: usize) -> u8 {
    let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (64 + (x % 129)) as u8
}

/// The 2-frequency checkerboard AB-probe content family (shared by
/// [`encoder_gate_e2e_ab_attempt`] and
/// [`encoder_gate_lf_level_bit_exact_vs_real`]). Real aomenc auto-detects
/// screen content on these patterns and picks GENUINELY NONZERO loop-filter
/// levels ([8,8] / [8,3] / [7,16] / [8,8] as coded at cq32), which is exactly
/// the nonzero LF-search path the flat/textured e2e cases never reach.
fn ab_top_split_bottom_flat(r: usize, c: usize) -> u8 {
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
fn ab_top_flat_bottom_split(r: usize, c: usize) -> u8 {
    if r >= 32 {
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
fn ab_left_split_right_flat(r: usize, c: usize) -> u8 {
    if c < 32 {
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
fn ab_left_flat_right_split(r: usize, c: usize) -> u8 {
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

/// Attempt the full derivation for one (w, h, mono, ss_x, ss_y, usage,
/// cq_level) case with FLAT constant-128 source content. Returns `true` iff
/// the assembled bytes matched the real stream byte-for-byte end to end.
#[allow(clippy::too_many_arguments)]
fn attempt_case(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
) -> bool {
    attempt_case_content(w, h, mono, ss_x, ss_y, usage, cq_level, |_r, _c| 128)
}

/// Same as [`attempt_case`] but with caller-supplied luma content
/// (`content(row, col) -> u8`); chroma is a flat mid-grey 128 regardless (so
/// only the LUMA search's decision space is stressed). Genuine texture (not
/// flat) exercises real partition/mode/tx-type competition -- a harder,
/// more meaningful test of coeff-cost decision parity than the trivial flat
/// case, at real risk of NOT matching (this port's search omits AB/4-way
/// partitions and doesn't replicate every candidate-order/pruning subtlety
/// of real aomenc's RDO).
#[allow(clippy::too_many_arguments)]
fn attempt_case_content(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    content: impl Fn(usize, usize) -> u8,
) -> bool {
    attempt_case_content_uv(
        w,
        h,
        mono,
        ss_x,
        ss_y,
        usage,
        cq_level,
        content,
        |_r, _c| 128,
    )
}

/// Same as [`attempt_case_content`] but ALSO accepts caller-supplied chroma
/// content (`uv_content(row, col) -> u8`, same value used for both U and V)
/// instead of flat mid-grey 128 -- lets a case stress ONLY the chroma
/// search's decision space (luma stays whatever `content` produces,
/// typically flat) while keeping the luma partition/mode landscape trivial.
#[allow(clippy::too_many_arguments)]
fn attempt_case_content_uv(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    content: impl Fn(usize, usize) -> u8,
    uv_content: impl Fn(usize, usize) -> u8,
) -> bool {
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
    let mut u = vec![128u16; cw * ch];
    let mut v = vec![128u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                let val = u16::from(uv_content(r, col));
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
    assert!(
        !bytes.is_empty(),
        "shim_encode_av1_kf must produce a real stream"
    );

    let obus = walk_obus(&bytes);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .unwrap_or_else(|| panic!("no sequence-header OBU (w={w} h={h})"));
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);

    let (frame_obu_type, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME_HEADER || *t == OBU_FRAME)
        .map(|(t, p)| (*t, *p))
        .unwrap_or_else(|| panic!("no frame/frame-header OBU (w={w} h={h})"));
    assert_eq!(
        frame_obu_type, OBU_FRAME,
        "w={w} h={h}: expected the combined num_tg==1 OBU_FRAME"
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
    // p: the REAL frame header, BOOTSTRAPPED (not derived) -- qindex, tile
    // info, tx-mode-select, cdf-update flag, etc. all come from real
    // aomenc's OWN choice. loop-filter LEVEL is overwritten below with this
    // port's own derivation (`pick_filter_level`) once the reconstruction is
    // available; every other loopfilter field (sharpness/deltas) stays
    // bootstrapped (out of this mission's scope, and already correct for
    // this envelope -- see `lf_search.rs` module docs).
    let p = read_uncompressed_header(&mut rb, &cfg);
    let real_bit_len = rb.bit_position();
    assert!(
        !p.prefix.show_existing_frame,
        "w={w} h={h}: show_existing_frame unexpected"
    );
    assert_eq!(
        p.prefix.frame_type, 0,
        "w={w} h={h}: frame_type must be KEY"
    );

    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    let allintra = usage == 2;
    let ctx = format!(
        "w={w} h={h} mono={mono} ss=({ss_x},{ss_y}) usage={usage} cq={cq_level} \
         qindex={} lf_level={:?} tiles_log2={tiles_log2} tx_mode_select={} lossless={} \
         screen_content={}",
        p.quant.base_qindex,
        p.loopfilter.filter_level,
        p.tx_mode_select,
        p.coded_lossless,
        p.prefix.allow_screen_content_tools,
    );
    eprintln!("{ctx}");
    assert_eq!(tiles_log2, 0, "{ctx}: single-tile envelope only");

    let tile_data_start = real_bit_len.div_ceil(8);
    let real_tile_bytes = &frame_payload[tile_data_start..];

    // ---- OUR OWN pipeline, config values read off the REAL (bootstrapped)
    //      header/seq-header, coefficients/modes/partitions TRUE-DERIVED ----
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

    // Row stride for OUR pipeline's source/recon buffers. `320` for every
    // frame up to 316px wide (so all existing <=256px cases are byte-for-byte
    // unchanged), widened to `w + 4` beyond that so 512px frames fit. The
    // stride is buffer padding only -- the encoded bytes depend solely on the
    // [0,w)x[0,h) crop, never on the padding columns -- so widening it cannot
    // perturb any case's output.
    let stride = 320.max(w + 4);
    let src_y = &y;
    // Pad the source buffers the same way the other pack.rs harnesses do
    // (a few extra rows of headroom; stride > w so row-major indexing below
    // matches SbEncodeEnv's stride contract).
    let mut src_y_strided = vec![0u16; stride * (h + 4)];
    for r in 0..h {
        src_y_strided[r * stride..r * stride + w].copy_from_slice(&src_y[r * w..r * w + w]);
    }
    let mut src_u_strided = vec![0u16; stride * (h + 4)];
    let mut src_v_strided = vec![0u16; stride * (h + 4)];
    if !mono {
        for r in 0..ch {
            src_u_strided[r * stride..r * stride + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
            src_v_strided[r * stride..r * stride + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
        }
    }

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
        filter_type: 0, // av1_get_filt_type (neighbour-derived) not ported -- matches existing pipeline's established simplification (pack_tile_roundtrip.rs, partition_pick_diff.rs).
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
        max_partition_size: 15, // BLOCK_64X64 == sb_size for this envelope
        min_partition_size: 0,  // BLOCK_4X4: the true aomenc default (unset --min-partition-size)
        enable_1to4_partitions: true, // the true aomenc default (unset --enable-1to4-partitions)
        enable_ab_partitions: true, // the true aomenc default (unset --disable-ab-partition-type)
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

    // ---- loop-filter-level: TRUE DERIVATION from OUR OWN reconstruction +
    //      the original source, via `pick_filter_level` (lf_search.rs) --
    //      replaces the bootstrapped `p.loopfilter.filter_level*` with this
    //      port's own av1_pick_filter_level-equivalent search. Every other
    //      loopfilter field (sharpness/deltas) stays bootstrapped -- see
    //      lf_search.rs module docs for why that's correct in this envelope. ----
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
    let derived_lf = pick_filter_level(&lf_frame, allintra, 0);
    eprintln!(
        "{ctx}: DERIVED lf_level={:?} lf_u={} lf_v={} sharpness={} -- REAL(bootstrapped) \
         lf_level={:?} lf_u={} lf_v={} sharpness={} -- {}",
        derived_lf.filter_level,
        derived_lf.filter_level_u,
        derived_lf.filter_level_v,
        derived_lf.sharpness,
        p.loopfilter.filter_level,
        p.loopfilter.filter_level_u,
        p.loopfilter.filter_level_v,
        p.loopfilter.sharpness_level,
        if derived_lf.filter_level == p.loopfilter.filter_level
            && derived_lf.filter_level_u == p.loopfilter.filter_level_u
            && derived_lf.filter_level_v == p.loopfilter.filter_level_v
        {
            "LF-LEVEL AGREES"
        } else {
            "LF-LEVEL DISAGREES"
        }
    );
    let mut p = p;
    p.loopfilter.filter_level = derived_lf.filter_level;
    p.loopfilter.filter_level_u = derived_lf.filter_level_u;
    p.loopfilter.filter_level_v = derived_lf.filter_level_v;

    let our_payload = assemble_frame_obu_payload_single_tile(&p, tiles_log2, &our_tile_bytes);

    eprintln!(
        "{ctx}: real_tile_bytes.len()={} our_tile_bytes.len()={} real_payload.len()={} \
         our_payload.len()={}",
        real_tile_bytes.len(),
        our_tile_bytes.len(),
        frame_payload.len(),
        our_payload.len()
    );

    if our_payload == frame_payload {
        eprintln!("{ctx}: TRUE END-TO-END BYTE MATCH");
        true
    } else {
        let first_diff = our_payload
            .iter()
            .zip(frame_payload.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(our_payload.len().min(frame_payload.len()));
        eprintln!(
            "{ctx}: MISMATCH at byte {first_diff} (our_payload[{first_diff}]={:?} \
             real_payload[{first_diff}]={:?}, header-region bytes = {})",
            our_payload.get(first_diff),
            frame_payload.get(first_diff),
            tile_data_start,
        );
        false
    }
}

/// The headline attempt: the smallest single-SB ALLINTRA (usage=2, the
/// zenavif/avifenc primary path) KEY frame, monochrome first (removes
/// chroma search entirely -- the simplest possible RDO landscape), then
/// 4:2:0, then GOOD usage. A flat constant-128 source is used throughout:
/// the RD-optimal choice at every candidate size/mode is DC prediction with
/// essentially zero residual, which is the best chance this port's
/// (narrower, 4-of-10-partition-type, no-AB/4-way) search has of reaching
/// the SAME decisions real aomenc's full search does.
///
/// **VERIFIED: all 3 cases achieve a TRUE end-to-end byte match** -- every
/// byte of the assembled `OBU_FRAME` payload (frame header, bootstrapped
/// from the real parse per the module docs, PLUS the tile-group payload
/// this port's OWN search+pack pipeline produces) equals real aomenc's own
/// output. Asserted as a hard regression gate (`assert_eq!(matched,
/// cases.len())`) -- this is genuinely the smallest possible case (a 1-byte
/// tile payload: EOB=0/txb_skip=1 for every plane, so it does NOT exercise
/// the coefficient-cost tables Task 1 fixed) and should not be read as
/// evidence of decision parity on frames with real texture -- see
/// [`encoder_gate_e2e_textured_attempt`] for that (harder, unasserted,
/// exploratory) attempt.
#[test]
fn encoder_gate_e2e_attempt() {
    let cases: &[(usize, usize, bool, usize, usize, u32, i32)] = &[
        (64, 64, true, 1, 1, 2, 32),  // mono, ALLINTRA -- simplest possible
        (64, 64, false, 1, 1, 2, 32), // 420, ALLINTRA
        (64, 64, false, 1, 1, 0, 32), // 420, GOOD
    ];
    let mut matched = 0usize;
    for &(w, h, mono, ss_x, ss_y, usage, cq_level) in cases {
        if attempt_case(w, h, mono, ss_x, ss_y, usage, cq_level) {
            matched += 1;
        }
    }
    eprintln!(
        "encoder_gate_e2e_attempt: {matched}/{} cases byte-identical end-to-end",
        cases.len()
    );
    assert_eq!(
        matched,
        cases.len(),
        "the flat-content envelope must be fully derived"
    );
}

/// Stretch goal beyond the trivial flat-content case above: genuinely
/// textured 64x64 mono ALLINTRA content, which forces real (nonzero-
/// residual) coefficient coding and gives coeff-cost decision parity
/// (Task 1) an actual chance to matter -- the flat case's near-empty
/// 1-byte tile payload doesn't exercise the coefficient-cost tables at all
/// (txb_skip=1 everywhere).
///
/// **VERIFIED 7/7, ASSERTED as a hard regression gate.** The "pseudo-random
/// noise" case was the one exception (6/7) until the 4-way partition port
/// (`PARTITION_HORZ_4`/`VERT_4` + the real `av1_ml_prune_4_partition` NN,
/// `crates/aom-encode/src/partition_pick.rs` + `part4_prune.rs`):
/// decode-diffing (`decode_diff_noise_case.rs`) isolated the divergence to
/// (mi_row=8, mi_col=8, bsize=16x16) where real aomenc chose
/// `PARTITION_VERT_4`, a type this port's search didn't have; with 4-way
/// ported, all 7 cases (including noise) now byte-match end-to-end and
/// `decode_diff_noise_case.rs` independently confirms the decoded partition
/// trees AND every leaf's mode/tx fields are identical, not just the raw
/// bytes. AB (`HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B`) is now ported --
/// [`encoder_gate_e2e_ab_attempt`] is the honest, unasserted probe for
/// content that needs it.
#[test]
fn encoder_gate_e2e_textured_attempt() {
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, fn(usize, usize) -> u8)] = &[
        ("horizontal gradient", |r, _c| (96 + r) as u8),
        ("vertical gradient", |_r, c| (96 + c) as u8),
        ("diagonal ramp", |r, c| (64 + (r + c) / 2) as u8),
        (
            "two-tone left/right split",
            |_r, c| if c < 32 { 90 } else { 160 },
        ),
        (
            "two-tone top/bottom split",
            |r, _c| if r < 32 { 90 } else { 160 },
        ),
        ("checkerboard (16px)", |r, c| {
            if (r / 16 + c / 16) % 2 == 0 { 80 } else { 176 }
        }),
        // Deterministic pseudo-random "noise" (xorshift, no external RNG
        // dependency): the hardest case for decision parity -- forces many
        // small nonzero residuals across many txbs, maximizing the chance
        // that a candidate-order/pruning difference between this port's
        // search and real aomenc's actually surfaces.
        ("pseudo-random noise", |r, c| {
            let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            (64 + (x % 129)) as u8
        }),
    ];
    let mut matched = 0usize;
    for &(name, content) in cases {
        eprintln!("--- textured case: {name} ---");
        if attempt_case_content(64, 64, true, 1, 1, 2, 32, content) {
            matched += 1;
        }
    }
    eprintln!(
        "encoder_gate_e2e_textured_attempt: {matched}/{} textured cases byte-identical end-to-end",
        cases.len()
    );
    assert_eq!(
        matched,
        cases.len(),
        "all 7 textured cases (incl. pseudo-random noise, which needs the 4-way partition port) \
         must byte-match end-to-end -- see module docs for the VERT_4 finding this gate pins"
    );
}

/// AB-partition probe (`PARTITION_HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B`, now
/// PORTED). Content is deliberately engineered to make an AB split
/// RD-attractive: one half of a block is uniform (an AB type's "whole" side
/// wants zero/near-zero residual, which only a genuinely flat region gives),
/// the other half has TWO distinct sharp-edged sub-regions (the AB type's
/// "split" side wants separate handling for each). E.g.
/// [`ab_top_split_bottom_flat`]: rows 0..32 hold a checkerboard on the left
/// 32 cols and a different-frequency checkerboard on the right 32 cols
/// (asymmetric detail -- rewards splitting the TOP into two quarters, i.e.
/// HORZ_A's shape); rows 32..64 are flat.
///
/// These patterns turn real aomenc's screen-content auto-detection ON and
/// drive its independent loop-filter search to GENUINELY NONZERO levels
/// (`[8,8]` / `[8,3]` / `[7,16]` / `[8,8]` as coded at cq32) -- so this is the
/// only e2e family that exercises the nonzero LF-level path end to end. The
/// loop-filter level is this port's OWN derivation here
/// ([`aom_encode::lf_search::pick_filter_level`]), NOT bootstrapped.
///
/// **State (2026-07-15): ALL 4 byte-match end to end -- ASSERTED.** Every case
/// proves the nonzero-LF header-assembly path AND the LF-level search are
/// correct on byte-identical reconstruction. The last mismatch,
/// "top flat / bottom split", was closed by a partition-RD fix: its coded tile
/// used to diverge from real's at a single node -- `(mi_row=8, mi_col=12,
/// BLOCK_16X16)`, where real picked `PARTITION_HORZ` and this port picked
/// `PARTITION_SPLIT`. Root cause (C-oracle traced, term-by-term): the 8x8
/// `PARTITION_NONE` leaf at `(8,12)` under that node's SPLIT was under-costing
/// `DC_PRED` by exactly 26 bits, flipping a near-tie against `V_PRED` (which
/// real picks). The 26 bits are the screen-content palette-Y "no-palette" flag
/// (`av1_allow_palette(allow_screen_content_tools, bsize)` +
/// `palette_y_mode_cost[bsize_ctx][mode_ctx][0]`), which real signals on every
/// `DC_PRED` block at `bsize >= BLOCK_8X8` in a screen-content frame -- this
/// port had hardcoded `try_palette: false` and omitted it. With DC correctly
/// costed, `V_PRED` wins the leaf (matching real), the SPLIT children match,
/// the node picks `PARTITION_HORZ`, and the whole tile is byte-identical. Fix:
/// `partition_pick.rs` (`PickFrameCfg::allow_screen_content_tools` threaded to
/// the leaf's `intra_mode_info_cost_y`).
#[test]
fn encoder_gate_e2e_ab_attempt() {
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, fn(usize, usize) -> u8)] = &[
        // Detail split across the TOP two quadrants (different checkerboard
        // periods left/right), uniform BOTTOM -- HORZ_A's shape (top-left +
        // top-right split, bottom whole).
        (
            "top split (2 freqs) / bottom flat",
            ab_top_split_bottom_flat,
        ),
        // Mirror: uniform TOP, detail split across the BOTTOM two quadrants
        // -- HORZ_B's shape.
        (
            "top flat / bottom split (2 freqs)",
            ab_top_flat_bottom_split,
        ),
        // Detail split across the LEFT two quadrants, uniform RIGHT --
        // VERT_A's shape.
        (
            "left split (2 freqs) / right flat",
            ab_left_split_right_flat,
        ),
        // Mirror: uniform LEFT, detail split across the RIGHT two quadrants
        // -- VERT_B's shape.
        (
            "left flat / right split (2 freqs)",
            ab_left_flat_right_split,
        ),
    ];
    let mut matched = 0usize;
    for &(name, content) in cases {
        eprintln!("--- AB-probe case: {name} ---");
        let ok = attempt_case_content(64, 64, true, 1, 1, 2, 32, content);
        assert!(
            ok,
            "AB-probe case {name:?} must byte-match real aomenc end-to-end \
             (64x64 mono cq32 all-intra). All four cases -- including \
             'top flat / bottom split', whose (mi_row=8,mi_col=12,BLOCK_16X16) \
             node was closed by the palette-Y no-palette flag-cost fix in \
             partition_pick.rs -- are now byte-identical."
        );
        matched += 1;
    }
    assert_eq!(
        matched,
        cases.len(),
        "all {} AB-probe cases must byte-match",
        cases.len()
    );
    eprintln!(
        "encoder_gate_e2e_ab_attempt: {matched}/{} AB-probe cases byte-identical end-to-end \
         (AB ported; the (mi_row=8,mi_col=12,BLOCK_16X16) real=HORZ/ours=SPLIT partition-RD \
         gap is CLOSED -- it was a missing screen-content palette-Y flag cost in the DC_PRED \
         leaf RD; see STATUS.md)",
        cases.len()
    );
}

/// Run this port's loop-filter-LEVEL search
/// ([`aom_encode::lf_search::pick_filter_level`], the `av1_pick_filter_level`
/// port) on real aomenc's OWN reconstruction and return
/// `(derived, real-coded)` `([v,h], u, v)` levels.
///
/// Method: encode with the real encoder; decode its OWN bytes to recover
/// real's EXACT pre-loop-filter reconstruction + mi grid
/// (`aom_decode::frame::{decode_frame_obus_prefilter, build_lf_inputs}`, both
/// already bit-exact vs C -- decoder track); run pick_filter_level on THAT.
/// Feeding real's own reconstruction removes the encoder-side reconstruction
/// (partition/mode/coeff RD, which for one AB case still diverges on a
/// separate gap) as a variable, so the returned pair isolates the LF-level
/// derivation itself. This is the same isolation the end-to-end AB-probe's 3
/// byte-identical cases demonstrate implicitly; here it is made explicit and
/// assert-grade for the nonzero levels.
#[allow(clippy::too_many_arguments)]
fn lf_derived_vs_real_on_real_recon(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    content: impl Fn(usize, usize) -> u8,
) -> (([i32; 2], i32, i32), ([i32; 2], i32, i32)) {
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
        0,
        false,
        false,
        usage,
        0,
        false,
    );
    assert!(
        !bytes.is_empty(),
        "shim_encode_av1_kf must produce a stream"
    );

    let allintra = usage == 2;
    let (t_real, cfg_real, hdr_real) = aom_decode::frame::decode_frame_obus_prefilter(&bytes)
        .expect("decode of real aomenc bytes must succeed");
    let (mi_real, _prm) = aom_decode::frame::build_lf_inputs(&t_real, &cfg_real, &hdr_real);

    const STRIDE: usize = 320;
    let mut src_y = vec![0u16; STRIDE * (h + 4)];
    for r in 0..h {
        src_y[r * STRIDE..r * STRIDE + w].copy_from_slice(&y[r * w..r * w + w]);
    }
    let mut src_u = vec![0u16; STRIDE * (ch + 4).max(1)];
    let mut src_v = vec![0u16; STRIDE * (ch + 4).max(1)];
    if !mono {
        for r in 0..ch {
            src_u[r * STRIDE..r * STRIDE + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
            src_v[r * STRIDE..r * STRIDE + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
        }
    }
    // Real's pre-filter reconstruction, re-strided to share STRIDE with src.
    let mut ry = vec![0u16; STRIDE * (t_real.height + 4)];
    for r in 0..t_real.height {
        ry[r * STRIDE..r * STRIDE + t_real.width]
            .copy_from_slice(&t_real.recon[r * t_real.stride..r * t_real.stride + t_real.width]);
    }
    let mut ru = vec![0u16; STRIDE * (t_real.height_uv + 4).max(1)];
    let mut rv = vec![0u16; STRIDE * (t_real.height_uv + 4).max(1)];
    if !mono {
        for r in 0..t_real.height_uv {
            ru[r * STRIDE..r * STRIDE + t_real.width_uv].copy_from_slice(
                &t_real.recon_u[r * t_real.stride_uv..r * t_real.stride_uv + t_real.width_uv],
            );
            rv[r * STRIDE..r * STRIDE + t_real.width_uv].copy_from_slice(
                &t_real.recon_v[r * t_real.stride_uv..r * t_real.stride_uv + t_real.width_uv],
            );
        }
    }
    let lf = LfSearchFrame {
        recon_y: &ry,
        recon_u: &ru,
        recon_v: &rv,
        src_y: &src_y,
        src_u: &src_u,
        src_v: &src_v,
        stride: STRIDE,
        crop_width: w as u32,
        crop_height: h as u32,
        ss_x,
        ss_y,
        bd: 8,
        monochrome: mono,
        mi: &mi_real,
        mi_rows: cfg_real.mi_rows,
        mi_cols: cfg_real.mi_cols,
    };
    let d = pick_filter_level(&lf, allintra, 0);
    (
        (d.filter_level, d.filter_level_u, d.filter_level_v),
        (
            hdr_real.loopfilter.filter_level,
            hdr_real.loopfilter.filter_level_u,
            hdr_real.loopfilter.filter_level_v,
        ),
    )
}

/// ASSERTED loop-filter-LEVEL bit-exactness gate. The AB-probe checkerboard
/// content drives real aomenc to GENUINELY NONZERO loop-filter levels
/// (`[8,8]` / `[8,3]` / `[7,16]` / `[8,8]` at cq32) -- the nonzero search path
/// the flat/textured e2e gates (all `[0,0]`) never reach. For each case, fed
/// real aomenc's OWN decoded reconstruction + mi grid, this port's
/// `pick_filter_level` MUST reproduce real's coded level EXACTLY.
///
/// This is the honest promotion of the previously-failing nonzero-LF case to
/// an asserted gate: it proves the loop-filter-LEVEL DERIVATION is bit-exact
/// vs the real encoder on real's own pixels -- including "top flat / bottom
/// split", whose end-to-end byte match is blocked ONLY by a separate
/// partition-RD reconstruction gap (`(8,12,BLOCK_16X16)` real=HORZ/ours=SPLIT,
/// see [`encoder_gate_e2e_ab_attempt`] + STATUS.md), NOT by the LF search.
#[test]
fn encoder_gate_lf_level_bit_exact_vs_real() {
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, fn(usize, usize) -> u8)] = &[
        ("top split / bottom flat", ab_top_split_bottom_flat),
        ("top flat / bottom split", ab_top_flat_bottom_split),
        ("left split / right flat", ab_left_split_right_flat),
        ("left flat / right split", ab_left_flat_right_split),
    ];
    let mut saw_nonzero = false;
    for &(name, content) in cases {
        let (derived, real) = lf_derived_vs_real_on_real_recon(64, 64, true, 1, 1, 2, 32, content);
        eprintln!(
            "lf-level gate [{name}]: derived (fl={:?} u={} v={}) -- real coded (fl={:?} u={} v={})",
            derived.0, derived.1, derived.2, real.0, real.1, real.2
        );
        saw_nonzero |= real.0 != [0, 0] || real.1 != 0 || real.2 != 0;
        assert_eq!(
            derived.0, real.0,
            "{name}: derived luma LF level must equal real aomenc's coded level on real's own \
             pre-filter reconstruction (av1_pick_filter_level bit-exactness)"
        );
        assert_eq!(
            derived.1, real.1,
            "{name}: derived U LF level must equal real aomenc's coded level"
        );
        assert_eq!(
            derived.2, real.2,
            "{name}: derived V LF level must equal real aomenc's coded level"
        );
    }
    assert!(
        saw_nonzero,
        "this gate must exercise at least one genuinely NONZERO real LF level, else it proves \
         nothing about the nonzero search path"
    );
}

/// EXPLORATORY, unasserted: sweep content/cq_level candidates looking for a
/// case that (a) drives a genuinely NONZERO luma LF level, (b) does NOT
/// trigger real aomenc's screen-content-tools auto-detection (avoids the
/// separate, unrelated `aom-entropy` bug documented in STATUS.md — see the
/// "Loop-filter-level RD search ported" milestone), and (c) stays within
/// this port's ported partition types (no AB needed — checked via
/// `real_tile_bytes.len() == our_tile_bytes.len()` as a necessary, not
/// sufficient, proxy). Print-only; the follow-up chunk promotes any winner
/// into an asserted regression gate. NOT part of the AB-probe family
/// (deliberately avoids short-period repeating patterns, which the AB-probe
/// findings show trip screen-content-tools).
#[test]
fn encoder_gate_e2e_nonzero_lf_sweep() {
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, i32, fn(usize, usize) -> u8)] = &[
        ("steep gradient cq32", 32, |r, _c| (r * 4).min(255) as u8),
        ("steep gradient cq48", 48, |r, _c| (r * 4).min(255) as u8),
        ("steep gradient cq60", 60, |r, _c| (r * 4).min(255) as u8),
        ("high-contrast two-tone split cq32", 32, |_r, c| {
            if c < 32 { 16 } else { 235 }
        }),
        ("high-contrast two-tone split cq48", 48, |_r, c| {
            if c < 32 { 16 } else { 235 }
        }),
        ("high-contrast two-tone split cq60", 60, |_r, c| {
            if c < 32 { 16 } else { 235 }
        }),
        ("bright bar on dark cq32", 32, |_r, c| {
            if (28..36).contains(&c) { 230 } else { 30 }
        }),
        ("bright bar on dark cq48", 48, |_r, c| {
            if (28..36).contains(&c) { 230 } else { 30 }
        }),
        ("bright bar on dark cq60", 60, |_r, c| {
            if (28..36).contains(&c) { 230 } else { 30 }
        }),
        ("radial blob cq32", 32, |r, c| {
            let dr = r as i32 - 32;
            let dc = c as i32 - 32;
            let d = ((dr * dr + dc * dc) as f64).sqrt();
            (255.0 - d * 5.0).clamp(0.0, 255.0) as u8
        }),
        ("radial blob cq48", 48, |r, c| {
            let dr = r as i32 - 32;
            let dc = c as i32 - 32;
            let d = ((dr * dr + dc * dc) as f64).sqrt();
            (255.0 - d * 5.0).clamp(0.0, 255.0) as u8
        }),
        ("steep gradient both axes cq48", 48, |r, c| {
            ((r + c) * 2).min(255) as u8
        }),
        ("steep gradient both axes cq60", 60, |r, c| {
            ((r + c) * 2).min(255) as u8
        }),
        ("noise cq48", 48, noise_content),
        ("noise cq60", 60, noise_content),
        ("noise cq50", 50, noise_content),
        ("noise cq63", 63, noise_content),
        ("fine non-repeating ripple cq48", 48, |r, c| {
            // Amplitude-varying "ripple": period 3 but the AMPLITUDE itself
            // drifts across the block (not an exact repeat -- avoids
            // screen-content-tools' exact-repeat heuristic while keeping
            // high spatial frequency for quantization sensitivity).
            let base = 128i32 + (r as i32 - 32);
            let amp = 20 + (c as i32 / 8) * 10;
            let ripple = if (r + c) % 3 == 0 { amp } else { -amp };
            (base + ripple).clamp(0, 255) as u8
        }),
        ("fine non-repeating ripple cq60", 60, |r, c| {
            let base = 128i32 + (r as i32 - 32);
            let amp = 20 + (c as i32 / 8) * 10;
            let ripple = if (r + c) % 3 == 0 { amp } else { -amp };
            (base + ripple).clamp(0, 255) as u8
        }),
    ];
    let mut winners = Vec::new();
    for &(name, cq, content) in cases {
        eprintln!("--- nonzero-LF sweep case: {name} (cq={cq}) ---");
        if attempt_case_content(64, 64, true, 1, 1, 2, cq, content) {
            winners.push(name);
        }
    }
    // Larger frames (more SBs -> more internal edges -> more potential
    // deblocking benefit) with 1-D stripes (not a 2-D checkerboard --
    // avoids the AB-probe's exact "screen content" shape) at fine periods.
    #[allow(clippy::type_complexity)]
    let big_cases: &[(&str, usize, usize, i32, fn(usize, usize) -> u8)] = &[
        ("128x128 vert stripes p4 cq48", 128, 128, 48, |_r, c| {
            if (c / 4) % 2 == 0 { 70 } else { 186 }
        }),
        ("128x128 vert stripes p6 cq48", 128, 128, 48, |_r, c| {
            if (c / 6) % 2 == 0 { 70 } else { 186 }
        }),
        ("128x128 vert stripes p4 cq32", 128, 128, 32, |_r, c| {
            if (c / 4) % 2 == 0 { 70 } else { 186 }
        }),
        ("128x128 noise cq48", 128, 128, 48, noise_content),
        ("256x256 vert stripes p4 cq48", 256, 256, 48, |_r, c| {
            if (c / 4) % 2 == 0 { 70 } else { 186 }
        }),
        ("256x256 noise cq32", 256, 256, 32, noise_content),
    ];
    let mut big_winners = Vec::new();
    for &(name, w, h, cq, content) in big_cases {
        eprintln!("--- nonzero-LF sweep (multi-SB) case: {name} ---");
        if attempt_case_content(w, h, true, 1, 1, 2, cq, content) {
            big_winners.push(name);
        }
    }
    eprintln!(
        "encoder_gate_e2e_nonzero_lf_sweep: {}/{} single-SB cases + {}/{} multi-SB cases \
         byte-identical end-to-end (exploratory -- see per-case eprintln output for \
         lf_level/screen-content-tools/tile-byte-length diagnostics; not asserted, this test \
         is a discovery tool)",
        winners.len(),
        cases.len(),
        big_winners.len(),
        big_cases.len(),
    );
}

/// EXPLORATORY, unasserted: keep LUMA flat (128 -- trivial partition/mode
/// landscape, matches the already-asserted flat cases) and vary ONLY
/// chroma, looking for content that drives `filter_level_u`/`filter_level_v`
/// nonzero while (a) staying clear of screen-content-tools (which appears
/// to key off luma, so should stay false here regardless of chroma
/// content -- verified per-case below, not assumed) and (b) not needing AB
/// partitions (flat luma trivially avoids that on the luma side; chroma has
/// no tx-depth/partition search of its own to diverge on -- `av1_get_tx_size_uv`
/// is a pure function of bsize/lossless/subsampling, module docs on
/// `intra_uv_rd.rs`). This is the only chunk this session exercises the
/// [`aom_encode::lf_search::pick_filter_level`] chroma path against a
/// genuinely nonzero real value (every other case tried, single- or
/// multi-SB, only ever reached `filter_level_u == filter_level_v == 0`).
#[test]
fn encoder_gate_e2e_nonzero_lf_chroma_sweep() {
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, i32, fn(usize, usize) -> u8)] = &[
        ("chroma checkerboard p4 cq32", 32, |r, c| {
            if (r / 4 + c / 4) % 2 == 0 { 90 } else { 166 }
        }),
        ("chroma checkerboard p4 cq48", 48, |r, c| {
            if (r / 4 + c / 4) % 2 == 0 { 90 } else { 166 }
        }),
        ("chroma checkerboard p2 cq32", 32, |r, c| {
            if (r / 2 + c / 2) % 2 == 0 { 90 } else { 166 }
        }),
        ("chroma checkerboard p2 cq48", 48, |r, c| {
            if (r / 2 + c / 2) % 2 == 0 { 90 } else { 166 }
        }),
        ("chroma stripes p2 cq32", 32, |_r, c| {
            if (c / 2) % 2 == 0 { 90 } else { 166 }
        }),
        ("chroma stripes p2 cq48", 48, |_r, c| {
            if (c / 2) % 2 == 0 { 90 } else { 166 }
        }),
        ("chroma noise cq32", 32, |r, c| noise_content(r, c)),
        ("chroma noise cq48", 48, |r, c| noise_content(r, c)),
    ];
    let mut winners = Vec::new();
    for &(name, cq, uv_content) in cases {
        eprintln!("--- chroma nonzero-LF sweep case: {name} (cq={cq}) ---");
        if attempt_case_content_uv(64, 64, false, 1, 1, 2, cq, |_r, _c| 128, uv_content) {
            winners.push(name);
        }
    }
    eprintln!(
        "encoder_gate_e2e_nonzero_lf_chroma_sweep: {}/{} cases byte-identical end-to-end \
         (exploratory -- see per-case eprintln output for lf_level/screen-content-tools/\
         tile-byte-length diagnostics; not asserted, this test is a discovery tool)",
        winners.len(),
        cases.len()
    );
}

/// **Multi-SB SCALE gate (primary task-3 deliverable), ASSERTED.** The SAME
/// single-tile search+pack pipeline that byte-matches real aomenc at 64x64
/// (`encoder_gate_e2e_attempt` / `_textured_attempt`, each a SINGLE superblock)
/// ALSO byte-matches end to end at **256x256 (16 SB64)** and **512x512 (64
/// SB64)** -- the first cases that exercise the multi-superblock path:
/// above/left neighbour-context threading ACROSS SB boundaries, one adapting
/// CDF shared across all SBs of the tile, and deblock/LF-level search over
/// interior SB edges. Content spans flat (pure structural proof), a hard-edged
/// two-tone split, and a real continuous-tone gradient, across aggressive
/// (cq48) and mid (cq32) quantization.
///
/// **Coverage: 12 of 16 swept (w, content, cq) cells byte-match, ASSERTED
/// here.** The 4 that diverge are all the STEEPEST content (the smooth
/// diagonal ramp -- energy along both axes -- at every cq, plus the steep
/// 256px vertical gradient) at the higher-quality cq32/cq48 end, where enough
/// coefficients survive quantization to surface an RD *decision* divergence.
/// It is NOT a structural multi-SB bug: FLAT 256x256/512x512 (which exercises
/// the identical cross-SB structure) byte-matches, and the gentler gradients
/// match. The divergence is localized structurally by
/// `decode_diff_multisb.rs` (a committed diagnostic) to its first divergent
/// `(mi_row, mi_col, bsize)` decision -- see STATUS.md for the exact node/term
/// and the smallest next chunk. Those 4 cells are deliberately EXCLUDED from
/// the asserted set (not silently non-asserting): they are the honest gap, and
/// the decode-diff test documents exactly where they diverge.
///
/// All content is continuous-tone or wide-flat and prints `screen_content=false`
/// -- staying clear of the two other, size-independent divergence regimes the
/// sweep documents (screen-content auto-detection on short-period repeats; the
/// noise-at-cq48/cq50 coeff-RD gap that also mismatches at 64x64).
#[test]
fn encoder_gate_e2e_multi_sb_scale() {
    // Size-scaled content by name (boxed so each closure can capture the frame
    // dimensions -- the `fn`-pointer tables the other gates use can't scale to
    // size). Luma only; chroma is flat mid-grey (mono frames here).
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "flat 128" => Box::new(|_r, _c| 128u8),
            "soft wide two-tone L/R split" => {
                Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 })
            }
            "smooth vertical gradient" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "smooth diagonal ramp" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            other => panic!("unknown content family {other:?}"),
        }
    }

    // The confirmed multi-SB byte-matching set (256x256 = 16 SB64, 512x512 = 64
    // SB64), empirically determined by the discovery sweep (see module doc).
    // Excludes the 4 steep-content/high-quality divergences (documented above +
    // in `decode_diff_multisb.rs`).
    let winners: &[(usize, usize, &str, i32)] = &[
        // 256x256 (16 SB64)
        (256, 256, "flat 128", 32),
        (256, 256, "flat 128", 48),
        (256, 256, "soft wide two-tone L/R split", 32),
        (256, 256, "soft wide two-tone L/R split", 48),
        (256, 256, "smooth vertical gradient", 48),
        (256, 256, "smooth diagonal ramp", 48),
        // 512x512 (64 SB64)
        (512, 512, "flat 128", 32),
        (512, 512, "flat 128", 48),
        (512, 512, "soft wide two-tone L/R split", 32),
        (512, 512, "soft wide two-tone L/R split", 48),
        (512, 512, "smooth vertical gradient", 32),
        (512, 512, "smooth vertical gradient", 48),
    ];
    let mut matched = 0usize;
    for &(w, h, name, cq) in winners {
        eprintln!("--- multi-SB {w}x{h} [{name}] cq{cq} ---");
        let content = content_for(w, h, name);
        if attempt_case_content(w, h, true, 1, 1, 2, cq, |r, c| content(r, c)) {
            matched += 1;
        }
    }
    eprintln!(
        "encoder_gate_e2e_multi_sb_scale: {matched}/{} multi-SB cases byte-identical end-to-end",
        winners.len()
    );
    assert_eq!(
        matched,
        winners.len(),
        "every confirmed multi-SB case (256x256 + 512x512, flat / two-tone / gradient, cq32 + \
         cq48) must byte-match real aomenc end-to-end -- a mismatch here is a genuine regression \
         (the 4 known steep-content cq32/cq48 divergences are excluded + localized by \
         decode_diff_multisb.rs; see this fn's module doc + STATUS.md)"
    );
}
