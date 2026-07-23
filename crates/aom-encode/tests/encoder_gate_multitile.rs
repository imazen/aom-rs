//! **Multi-tile end-to-end byte-match gate.** Extends the single-tile e2e
//! proof (`encoder_gate_e2e_byte_match.rs`) to `tile_cols`/`tile_rows > 1`:
//! this port's OWN per-tile `pack_tile` + the new
//! `obu_assemble::assemble_frame_obu_payload_multi_tile` produce a byte-
//! identical `OBU_FRAME` payload vs real aomenc encoded with
//! `AV1E_SET_TILE_COLUMNS`/`AV1E_SET_TILE_ROWS` (via the committed append-only
//! oracle `ref_encode_av1_kf_tiles`).
//!
//! Each AV1 tile is entropy-INDEPENDENT: a fresh `KfFrameContext` +
//! `OdEcEnc` per tile, and an `SbEncodeEnv` whose `tile_row_start`/
//! `tile_col_start`/`tile_row_end`/`tile_col_end` are the tile's own MI bounds
//! (from the parsed header's `col_start_sb`/`row_start_sb`) so intra
//! prediction / tx-size context / the RD search treat the tile edges as
//! unavailable and never read the adjacent tile's reconstruction. The per-tile
//! raw bytes are then length-prefixed (`tile_size_bytes`-byte LE
//! `tile_size_minus_1`) for every tile except the last, per the AV1 tile-group
//! layout (`num_tg == 1` => one `OBU_FRAME`, `tile_start_and_end_present_flag
//! == 0`).
//!
//! The frame header (including the multi-tile `tile_info`:
//! `context_update_tile_id`, `tile_size_bytes`, the uniform tile spacing) is
//! bootstrapped verbatim from the real parse -- the SAME bootstrap boundary as
//! the single-tile e2e gates -- and the loop-filter LEVEL is bootstrapped too
//! (this gate isolates the TILE machinery; `pick_filter_level` bit-exactness is
//! already proven separately by `encoder_gate_lf_level_bit_exact_vs_real`).
//!
//! DISCOVERY MODE (this revision): sweeps a small (size, tile-config, content,
//! cq) matrix and PRINTS which cells byte-match, so the confirmed-matching set
//! can be locked as an assertion. The single-tile gate showed the coefficient
//! trellis diverges on steep content at higher quality; the matching subset
//! here is chosen to isolate the multi-tile machinery from that separate gap.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::obu_assemble::assemble_multitile_frame_obu_payload;
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
use aom_dsp::entropy::partition::KfFrameContext;
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_dsp::quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME_HEADER: u32 = 3;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4

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

/// Attempt one multi-tile case. Returns `true` iff the assembled `OBU_FRAME`
/// payload equals real aomenc's byte-for-byte. `tile_columns_log2`/
/// `tile_rows_log2` are the `AV1E_SET_TILE_COLUMNS`/`_ROWS` values (the CODED
/// log2 tile counts). Monochrome, ALLINTRA-or-GOOD per `usage`, speed 0.
#[allow(clippy::too_many_arguments)]
fn attempt_multitile_case(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    tile_columns_log2: i32,
    tile_rows_log2: i32,
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

    let bytes = c::ref_encode_av1_kf_tiles(
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
        false, // sb_size_128
        tile_columns_log2,
        tile_rows_log2,
    );
    assert!(!bytes.is_empty(), "oracle must produce a real stream");

    let obus = walk_obus(&bytes);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .expect("sequence-header OBU");
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);

    let (frame_obu_type, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME_HEADER || *t == OBU_FRAME)
        .map(|(t, p)| (*t, *p))
        .expect("frame/frame-header OBU");
    assert_eq!(
        frame_obu_type, OBU_FRAME,
        "num_tg==1 => combined OBU_FRAME expected"
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
    // p: the REAL frame header, bootstrapped verbatim (same boundary as the
    // single-tile e2e gates), including the multi-tile tile_info +
    // loop-filter level. This gate isolates the TILE machinery.
    let p = read_uncompressed_header(&mut rb, &cfg);
    // The byte-aligned frame-header span (frame_header_obu() bits +
    // frame_obu()'s byte_alignment()). Spliced VERBATIM (the header is
    // bootstrapped anyway) rather than re-serialized: aom-entropy's
    // `write_tile_info` multi-tile branch currently hardcodes
    // context_update_tile_id / tile_size_bytes_minus_1, so re-serializing a
    // multi-tile header does not round-trip. This gate proves the TILE
    // machinery (per-tile pack + tile-group length-prefix assembly), not the
    // multi-tile header serialization.
    let header_end = rb.bit_position().div_ceil(8);
    let frame_header_bytes = &frame_payload[0..header_end];
    assert!(!p.prefix.show_existing_frame);
    assert_eq!(p.prefix.frame_type, 0, "frame_type must be KEY");

    let ti = &p.tile_info;
    let tiles_log2 = ti.log2_cols + ti.log2_rows;
    let n_tile_cols = ti.cols;
    let n_tile_rows = ti.rows;
    let ctx = format!(
        "w={w} h={h} mono={mono} usage={usage} cq={cq_level} tiles={n_tile_cols}x{n_tile_rows} \
         (log2 {}+{}) qindex={} tx_mode_select={} lossless={} sc={}",
        ti.log2_cols,
        ti.log2_rows,
        p.quant.base_qindex,
        p.tx_mode_select,
        p.coded_lossless,
        p.prefix.allow_screen_content_tools,
    );
    // This gate is for genuine multi-tile frames only.
    assert!(tiles_log2 > 0, "{ctx}: expected multi-tile (tiles_log2>0)");

    // ---- OUR pipeline: config from the bootstrapped header (single set of
    //      real-derived costs; each tile starts from the SAME initial CDF) ----
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

    let kf_init = KfFrameContext::default_for_qindex(qindex);
    let real = derive_real_costs(&kf_init, s.enable_filter_intra);

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
    if !mono {
        for r in 0..ch {
            src_u_strided[r * stride..r * stride + cw].copy_from_slice(&u[r * cw..r * cw + cw]);
            src_v_strided[r * stride..r * stride + cw].copy_from_slice(&v[r * cw..r * cw + cw]);
        }
    }

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

    // Shared full-frame reconstruction buffers -- each tile writes only its own
    // region; the tile bounds keep intra prediction from reading across tiles.
    let mut recon_y = src_y_strided.clone();
    let mut recon_u = src_u_strided.clone();
    let mut recon_v = src_v_strided.clone();

    // Per-tile pack, raster (tile-row-major) order.
    let mut tile_payloads: Vec<Vec<u8>> = Vec::with_capacity(n_tile_cols * n_tile_rows);
    for trow in 0..n_tile_rows {
        let mi_row_start = ti.row_start_sb[trow] << ti.mib_size_log2;
        let mi_row_end = (ti.row_start_sb[trow + 1] << ti.mib_size_log2).min(mi_rows);
        let n_sb_rows = ti.row_start_sb[trow + 1] - ti.row_start_sb[trow];
        for tcol in 0..n_tile_cols {
            let mi_col_start = ti.col_start_sb[tcol] << ti.mib_size_log2;
            let mi_col_end = (ti.col_start_sb[tcol + 1] << ti.mib_size_log2).min(mi_cols);
            let n_sb_cols = ti.col_start_sb[tcol + 1] - ti.col_start_sb[tcol];

            let env = SbEncodeEnv {
            ref_frame: None,
                sb_size: SB,
                mi_rows,
                mi_cols,
                // The tile's OWN mi bounds -- gates intra-pred / tx-size ctx /
                // RD-search neighbour availability at the tile edges.
                tile_row_start: mi_row_start,
                tile_col_start: mi_col_start,
                tile_row_end: mi_row_end,
                tile_col_end: mi_col_end,
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

            // Fresh entropy state per tile (tiles are entropy-independent).
            let mut kf_write = KfFrameContext::default_for_qindex(qindex);
            let mut enc = OdEcEnc::new();
            let trees = pack_tile(
                &mut enc,
                &env,
                &pick_cfg,
                &pack_cfg,
                &mut kf_write,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                mi_row_start,
                mi_col_start,
                n_sb_rows,
                n_sb_cols,
                SB_MI,
                SB,
            );
            assert_eq!(
                trees.len(),
                (n_sb_rows * n_sb_cols) as usize,
                "{ctx}: tile ({trow},{tcol}) must walk every SB"
            );
            tile_payloads.push(enc.done().to_vec());
        }
    }

    let our_payload =
        assemble_multitile_frame_obu_payload(frame_header_bytes, p.tile_size_bytes, &tile_payloads);

    if our_payload == frame_payload {
        eprintln!("{ctx}: TRUE MULTI-TILE END-TO-END BYTE MATCH");
        true
    } else {
        let first_diff = our_payload
            .iter()
            .zip(frame_payload.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(our_payload.len().min(frame_payload.len()));
        eprintln!(
            "{ctx}: MISMATCH at byte {first_diff} (our={:?} real={:?}); our_payload.len()={} \
             real_payload.len()={} tile_size_bytes={} tile_lens={:?}",
            our_payload.get(first_diff),
            frame_payload.get(first_diff),
            our_payload.len(),
            frame_payload.len(),
            p.tile_size_bytes,
            tile_payloads.iter().map(|t| t.len()).collect::<Vec<_>>(),
        );
        false
    }
}

/// **Multi-tile end-to-end byte-match gate, ASSERTED.** Every case: this port's
/// per-tile `pack_tile` (tile bounds set to the tile's own MI edges, fresh
/// `KfFrameContext` + `OdEcEnc` per tile) + `assemble_multitile_frame_obu_payload`
/// produce a byte-identical `OBU_FRAME` payload vs real aomenc encoded with
/// `AV1E_SET_TILE_COLUMNS`/`_ROWS`.
///
/// Coverage: `tile_cols x tile_rows` of `2x1`, `2x2`, and `4x1`, at 128x128 /
/// 256x256 / 512x512, cq48 + cq32. Content includes flat AND gradients that
/// vary ACROSS the tile-column/row boundary (hgrad/vgrad) -- the latter would
/// diverge if a tile's intra prediction leaked the adjacent tile's
/// reconstruction, so they specifically prove the tile-boundary isolation
/// (`SbEncodeEnv::tile_{row,col}_{start,end}` gating intra-pred / tx-size ctx /
/// RD-search neighbour availability). Content stays in the single-tile gate's
/// confirmed-matching regime (no steep-content coeff-trellis divergence), so a
/// mismatch isolates the multi-tile machinery.
///
/// The frame header (incl. the multi-tile `tile_info`) is bootstrapped verbatim
/// from the real parse (same boundary as the single-tile e2e gates) because
/// aom-entropy's `write_tile_info` multi-tile branch currently hardcodes
/// `context_update_tile_id`/`tile_size_bytes` -- see the module doc. This gate
/// therefore asserts the TILE machinery + tile-group length-prefix assembly,
/// not the multi-tile header serialization (tracked separately).
#[test]
fn encoder_gate_multitile_byte_match() {
    let flat = |_r: usize, _c: usize| 128u8;
    // Content whose value varies ACROSS the tile column/row boundary, so a
    // mis-set tile bound (intra pred leaking the adjacent tile's recon) would
    // change the coded bytes.
    fn hgrad(w: usize) -> impl Fn(usize, usize) -> u8 {
        move |_r, c| (40 + c * 150 / w) as u8
    }
    fn vgrad(h: usize) -> impl Fn(usize, usize) -> u8 {
        move |r, _c| (40 + r * 150 / h) as u8
    }

    #[allow(clippy::type_complexity)]
    let cases: Vec<(
        usize,
        usize,
        i32,
        i32,
        i32,
        &str,
        Box<dyn Fn(usize, usize) -> u8>,
    )> = vec![
        // (w, h, tile_cols_log2, tile_rows_log2, cq, name, content)
        (128, 128, 1, 0, 48, "flat 2x1", Box::new(flat)),
        (256, 256, 1, 0, 48, "flat 2x1", Box::new(flat)),
        (256, 256, 1, 1, 48, "flat 2x2", Box::new(flat)),
        (256, 256, 2, 0, 48, "flat 4x1", Box::new(flat)),
        (256, 256, 1, 0, 48, "hgrad 2x1", Box::new(hgrad(256))),
        (256, 256, 1, 1, 48, "hgrad 2x2", Box::new(hgrad(256))),
        (256, 256, 1, 1, 48, "vgrad 2x2", Box::new(vgrad(256))),
        (256, 256, 1, 0, 32, "flat 2x1 cq32", Box::new(flat)),
        (256, 256, 1, 1, 32, "flat 2x2 cq32", Box::new(flat)),
        // scale x tiles: 512x512 (64 SB64) split into 2x1 and 2x2 tiles
        (512, 512, 1, 0, 48, "flat 2x1", Box::new(flat)),
        (512, 512, 1, 1, 48, "flat 2x2", Box::new(flat)),
    ];
    let mut matched = 0usize;
    for (w, h, tcl, trl, cq, name, content) in &cases {
        eprintln!("--- multitile {name} {w}x{h} tiles(log2 {tcl}+{trl}) cq{cq} ---");
        if attempt_multitile_case(*w, *h, true, 1, 1, 2, *cq, *tcl, *trl, |r, c| content(r, c)) {
            matched += 1;
        }
    }
    eprintln!(
        "encoder_gate_multitile_byte_match: {matched}/{} multi-tile cases byte-identical",
        cases.len()
    );
    assert_eq!(
        matched,
        cases.len(),
        "every multi-tile case (2x1 / 2x2 / 4x1, flat + boundary-crossing gradients, cq32 + cq48, \
         128/256/512) must byte-match real aomenc end-to-end -- a mismatch is a genuine multi-tile \
         machinery regression (per-tile pack bounds / fresh entropy / length-prefix assembly)"
    );
}
