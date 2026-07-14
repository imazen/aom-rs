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
    // p: the REAL frame header, BOOTSTRAPPED (not derived) -- qindex,
    // loop-filter level/deltas, tile info, tx-mode-select, cdf-update flag,
    // etc. all come from real aomenc's OWN choice. See module docs.
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
         qindex={} lf_level={:?} tiles_log2={tiles_log2} tx_mode_select={} lossless={}",
        p.quant.base_qindex, p.loopfilter.filter_level, p.tx_mode_select, p.coded_lossless,
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

    const STRIDE: usize = 320;
    let src_y = &y;
    // Pad the source buffers the same way the other pack.rs harnesses do
    // (a few extra rows of headroom; STRIDE > w so row-major indexing below
    // matches SbEncodeEnv's stride contract).
    let mut src_y_strided = vec![0u16; STRIDE * (h + 4)];
    for r in 0..h {
        src_y_strided[r * STRIDE..r * STRIDE + w].copy_from_slice(&src_y[r * w..r * w + w]);
    }
    let mut src_u_strided = vec![0u16; STRIDE * (h + 4)];
    let mut src_v_strided = vec![0u16; STRIDE * (h + 4)];
    if !mono {
        for r in 0..ch {
            src_u_strided[r * STRIDE..r * STRIDE + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
            src_v_strided[r * STRIDE..r * STRIDE + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
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
        max_partition_size: 15, // BLOCK_64X64 == sb_size for this envelope
        min_partition_size: 0,  // BLOCK_4X4: the true aomenc default (unset --min-partition-size)
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
    assert_eq!(
        trees.len(),
        (n_sb * n_sb) as usize,
        "{ctx}: pack_tile must walk every SB"
    );
    let our_tile_bytes = enc.done().to_vec();

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
/// bytes. AB (`HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B`) is still unported --
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

/// AB-partition probe (`PARTITION_HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B`, still
/// unported -- see STATUS.md's MISSING list). Content is deliberately
/// engineered to make an AB split RD-attractive: one half of a block is
/// uniform (an AB type's "whole" side wants zero/near-zero residual, which
/// only a genuinely flat region gives), the other half has TWO distinct
/// sharp-edged sub-regions (the AB type's "split" side wants separate
/// handling for each). E.g. `top_split_bottom_flat`: rows 0..32 hold a
/// checkerboard on the left 32 cols and a different-frequency checkerboard
/// on the right 32 cols (asymmetric detail -- rewards splitting the TOP
/// into two quarters, i.e. HORZ_A's shape); rows 32..64 are flat.
///
/// Honestly NOT asserted (mission requirement: do not assert-pass a case
/// that isn't proven byte-identical). Each case prints match / first-
/// divergent-byte; the final line states the honest fraction. If a case
/// diverges, that is EXPECTED (not a bug) until AB is ported -- this test
/// exists to give forward visibility into whether real aomenc's own RDO
/// actually picks an AB type on this content (via the SAME kind of
/// decode-diff `decode_diff_noise_case.rs` used to find the VERT_4 gap),
/// not to claim AB support that doesn't exist.
///
/// **CONFOUNDED, honestly reported: this specific content family does NOT
/// currently isolate the AB question.** All 4 cases mismatch at byte 0 --
/// inside the bootstrapped frame HEADER, before any partition/mode/
/// coefficient data -- because this content's edge sharpness (even softened
/// to the checkerboard-16px contrast that keeps the OTHER textured cases at
/// `lf_level=[0,0]`) still pushes real aomenc's independent loop-filter
/// search to a NONZERO level (`[7,8]..[8,16]` observed). This port's e2e
/// pipeline bootstraps `lf_level` verbatim from the real parsed header (no
/// LF-level search is ported -- see this file's module docs and STATUS.md's
/// MISSING list: "every e2e case observed lf_level=[0,0] ... A nonzero-LF
/// case is untested end-to-end"), so a header-assembly path that has NEVER
/// been exercised end-to-end with nonzero lf_level is now confirmed broken
/// somewhere in that untested path -- separate from, and prior to, whatever
/// AB partitions would or wouldn't explain. Root-causing the nonzero-LF
/// header path is out of THIS mission's scope (partition types); flagged
/// here as a genuine, newly-confirmed gap for whichever session picks up
/// loop-filter-level search next, not fixed or worked around.
#[test]
fn encoder_gate_e2e_ab_attempt() {
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, fn(usize, usize) -> u8)] = &[
        // Detail split across the TOP two quadrants (different checkerboard
        // periods left/right), uniform BOTTOM -- HORZ_A's shape (top-left +
        // top-right split, bottom whole).
        ("top split (2 freqs) / bottom flat", |r, c| {
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
        }),
        // Mirror: uniform TOP, detail split across the BOTTOM two quadrants
        // -- HORZ_B's shape.
        ("top flat / bottom split (2 freqs)", |r, c| {
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
        }),
        // Detail split across the LEFT two quadrants, uniform RIGHT --
        // VERT_A's shape.
        ("left split (2 freqs) / right flat", |r, c| {
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
        }),
        // Mirror: uniform LEFT, detail split across the RIGHT two quadrants
        // -- VERT_B's shape.
        ("left flat / right split (2 freqs)", |r, c| {
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
        }),
    ];
    let mut matched = 0usize;
    for &(name, content) in cases {
        eprintln!("--- AB-probe case: {name} ---");
        if attempt_case_content(64, 64, true, 1, 1, 2, 32, content) {
            matched += 1;
        }
    }
    eprintln!(
        "encoder_gate_e2e_ab_attempt: {matched}/{} AB-probe cases byte-identical end-to-end \
         (exploratory -- AB partitions unported, see module docs; not asserted)",
        cases.len()
    );
}
