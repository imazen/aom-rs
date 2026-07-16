//! Full-frame ENCODE byte-match gate at **4:2:2 and 4:4:4** (8-bit) — closes a
//! real single-frame coverage gap. The existing end-to-end gates
//! (`encoder_gate_e2e_byte_match.rs`, `encoder_gate_bd10_diff.rs`) validate the
//! port's own search+pack pipeline (`pack_tile` +
//! `assemble_frame_obu_payload_single_tile` + port-derived loop-filter) only for
//! **mono + 4:2:0**. The block/plane module diffs
//! (`encode_intra_plane_uv_diff`, `cfl_alpha_search_diff`,
//! `intra_sbuv_mode_loop_diff`) already prove 4:2:2 / 4:4:4 chroma *block* ops are
//! byte-exact, and `encode_sb_diff` covers the SB pipeline at 4:4:4 — but NO gate
//! proved the **full-frame integration** (multi-SB pack + frame OBU assemble +
//! frame loop-filter) is byte-exact at 4:2:2 or 4:4:4. This gate does, at 8-bit,
//! across sizes 64/128/192 (1×1 / 2×2 / 3×3 SB) and cq 12/32/63.
//!
//! Structure mirrors `encoder_gate_bd10_diff.rs` exactly: encode the reference
//! with real aomenc (`ref_encode_av1_kf`), bootstrap the frame header from that
//! parse, run THIS PORT's `pack_tile` over the identical source pixels, derive the
//! loop-filter level, assemble the OBU payload, and compare byte-for-byte. Every
//! test asserts real byte-identity — no `#[ignore]`, no weakened asserts, no
//! graceful skips. Content is textured (never all-flat → avoids an all-skip frame)
//! and chroma is genuinely distinct high-frequency content, NOT a scaled copy of
//! luma (else CfL would trivialize the chroma path).
//!
//! High-bit-depth (bd10/bd12) chroma-subsampling and lossless (cq0) full-frame
//! coverage land here alongside their respective port fixes; this file starts with
//! the 8-bit 4:2:2 / 4:4:4 cells that are byte-exact today.
//!
//! This file is OWNED by the chroma-subsampling e2e track; it does not touch the
//! bd10 track's `encoder_gate_bd10_diff.rs` nor the encoder track's
//! `encoder_gate_e2e_byte_match.rs`.

use aom_encode::encode_intra::TrellisOptType;
use aom_encode::encode_sb::SbEncodeEnv;
use aom_encode::intra_uv_rd::UvLoopPolicy;
use aom_encode::lf_search::{LfSearchFrame, build_lf_mi_grid, pick_filter_level};
use aom_encode::obu_assemble::{
    assemble_frame_obu_payload_single_tile, assemble_multitile_frame_obu_payload_derived,
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
use aom_entropy::partition::KfFrameContext;
use aom_entropy::rb::ReadBitBuffer;
use aom_quant::{Dequants, Quants, av1_build_quantizer, set_q_index};
use aom_sys_ref as c;

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64px / 4
const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

fn walk_obus(bytes: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let hdr = read_obu_header(&bytes[pos..]).expect("valid OBU header");
        let after_header = pos + hdr.header_len;
        assert!(hdr.obu_has_size_field, "shim always sets has_size_field");
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

/// Result of one case: whether the port matched real aomenc byte-for-byte.
struct CaseResult {
    matched: bool,
}

/// Encode one case at bit depth `bd` with `content(row,col) -> u16` luma and
/// `uv_content(row,col) -> u16` chroma (values in `[0, (1<<bd)-1]`; chroma ignored
/// when `mono`), bootstrap the header from real aomenc, run this port's
/// `pack_tile`, assemble, and compare byte-for-byte. Verbatim copy of the bd10
/// harness's `run_case` (parameterized on `bd`, `ss_x`, `ss_y`, `mono`), with the
/// subsampling label corrected (the bd10 harness only ever printed "4:2:0") and
/// the returned `CaseResult` extended to carry the lossless-confirming header facts.
/// `mono=true` removes the chroma path entirely; non-mono `ss_x/ss_y` select
/// 4:4:4 (0,0) / 4:2:2 (1,0) / 4:2:0 (1,1).
#[allow(clippy::too_many_arguments)]
fn run_case(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    bd: u8,
    content: impl Fn(usize, usize) -> u16,
    u_content: impl Fn(usize, usize) -> u16,
    v_content: impl Fn(usize, usize) -> u16,
    // Tile grid: (tile_columns_log2, tile_rows_log2). (0, 0) => a single tile
    // (the whole frame) via `ref_encode_av1_kf` — the original behaviour, byte
    // for byte. Anything non-zero drives the real MULTI-TILE encoder
    // (`ref_encode_av1_kf_tiles`) and this port's per-tile search+pack loop
    // (fresh entropy coder + reset CDF context per tile, tile-boundary neighbour
    // availability via env.tile_*), then the derived multi-tile OBU assembler.
    tile_cols_log2: i32,
    tile_rows_log2: i32,
) -> CaseResult {
    c::ref_init();
    let maxv = (1u16 << bd) - 1;
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = content(r, col).min(maxv);
        }
    }
    let (cw, ch) = if mono {
        (0, 0)
    } else {
        ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y)
    };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for col in 0..cw {
                u[r * cw + col] = u_content(r, col).min(maxv);
                v[r * cw + col] = v_content(r, col).min(maxv);
            }
        }
    }

    let bytes = if tile_cols_log2 == 0 && tile_rows_log2 == 0 {
        c::ref_encode_av1_kf(
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
        )
    } else {
        c::ref_encode_av1_kf_tiles(
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
            false, // sb_size_128
            tile_cols_log2,
            tile_rows_log2,
        )
    };
    assert!(
        !bytes.is_empty(),
        "ref_encode_av1_kf (bd{bd}) must produce a real stream"
    );

    let obus = walk_obus(&bytes);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, p)| *p)
        .unwrap_or_else(|| panic!("no sequence-header OBU (bd{bd} w={w} h={h})"));
    let mut seq_rb = ReadBitBuffer::new(seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);

    let (frame_obu_type, frame_payload) = obus
        .iter()
        .find(|(t, _)| *t == OBU_FRAME || *t == 3)
        .map(|(t, p)| (*t, *p))
        .unwrap_or_else(|| panic!("no frame OBU (bd{bd})"));
    assert_eq!(
        frame_obu_type, OBU_FRAME,
        "expected combined OBU_FRAME (bd{bd})"
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

    // Two-pass header parse for coded-lossless, mirroring the decoder
    // (aom-decode/src/frame.rs:466-490). `read_uncompressed_header` gates its
    // loop-filter / CDEF / restoration / tx-mode tail reads on `cfg.coded_lossless`
    // / `cfg.all_lossless` (a writer-mirror INPUT), but quant + segmentation
    // precede every gated read, so a probe pass with `coded_lossless = false`
    // yields exact quant regardless of the (then-misread) tail. Recompute
    // `coded_lossless` from the probe's quant — this harness encodes segmentation-
    // off KEY frames, so `frame_coded_lossless` reduces to base_qindex == 0 with
    // all 5 plane q-deltas 0 — and, when the stream IS coded-lossless (cq0 /
    // `--lossless=1`), re-parse with the correct gating so the tail is read right.
    // No superres here => `all_lossless == coded_lossless`.
    let mut rb = ReadBitBuffer::new(frame_payload);
    let probe = read_uncompressed_header(&mut rb, &cfg);
    let key_shown = !probe.prefix.show_existing_frame
        && probe.prefix.frame_type == 0
        && probe.prefix.show_frame;
    let coded_lossless = key_shown
        && probe.quant.base_qindex == 0
        && probe.quant.y_dc_delta_q == 0
        && probe.quant.u_dc_delta_q == 0
        && probe.quant.u_ac_delta_q == 0
        && probe.quant.v_dc_delta_q == 0
        && probe.quant.v_ac_delta_q == 0;
    let (p, real_bit_len) = if coded_lossless {
        let mut cfg2 = cfg.clone();
        cfg2.coded_lossless = true;
        cfg2.all_lossless = true;
        let mut rb2 = ReadBitBuffer::new(frame_payload);
        let mut p2 = read_uncompressed_header(&mut rb2, &cfg2);
        p2.coded_lossless = true;
        p2.all_lossless = true;
        (p2, rb2.bit_position())
    } else {
        (probe, rb.bit_position())
    };
    assert!(!p.prefix.show_existing_frame);
    assert_eq!(p.prefix.frame_type, 0, "frame_type must be KEY");
    let tiles_log2 = p.tile_info.log2_cols + p.tile_info.log2_rows;
    if tile_cols_log2 == 0 && tile_rows_log2 == 0 {
        assert_eq!(tiles_log2, 0, "single-tile envelope: expected exactly 1 tile");
    } else {
        // Multi-tile: the real encoder must have produced the REQUESTED grid, or
        // the per-tile SB boundaries the port packs against won't line up.
        assert_eq!(
            (p.tile_info.log2_cols, p.tile_info.log2_rows),
            (tile_cols_log2, tile_rows_log2),
            "real encoder tile grid must match requested (cols_log2, rows_log2)"
        );
    }
    let allintra = usage == 2;
    let fmt = if mono {
        "mono".to_string()
    } else {
        let name = match (ss_x, ss_y) {
            (0, 0) => "4:4:4",
            (1, 0) => "4:2:2",
            (1, 1) => "4:2:0",
            _ => "chroma",
        };
        format!("{name}(ss={ss_x},{ss_y})")
    };
    let ctx = format!(
        "bd{bd} w={w} h={h} {fmt} usage={usage} cq={cq_level} qindex={} lossless={}",
        p.quant.base_qindex, p.coded_lossless
    );
    eprintln!("{ctx}");

    // ---- port pipeline, header bootstrapped, coeffs/modes/partitions derived --
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

    // Frame-edge (partial-SB) support: the SB walk covers CEIL(mi/16) SBs, so
    // a non-64-aligned frame (e.g. 196 -> mi_cols=50, 4 SBs) has partial right/
    // bottom edge SBs. C reads the FULL block/tx from the border-EXTENDED source
    // (av1_get_perpixel_variance / av1_subtract_txb read the whole extent incl.
    // the off-frame overhang, which aom_extend_frame_borders replicate-fills from
    // the crop edge). So size the planes to the SB-aligned extent and replicate the
    // crop edge into the overhang, matching what aomenc encodes. SB-aligned frames
    // keep the usual 320 / h+4 envelope (sb_px_* == w/h rounded to the same SBs),
    // so interior cells stay byte-identical; the overhang region is never read there.
    let n_sb_x = ((mi_cols + SB_MI - 1) / SB_MI).max(1);
    let n_sb_y = ((mi_rows + SB_MI - 1) / SB_MI).max(1);
    let sb_px_w = n_sb_x as usize * 64;
    let sb_px_h = n_sb_y as usize * 64;
    let stride = 320.max(sb_px_w + 4);
    let buf_h = (sb_px_h + 4).max(h + 4);
    // Replicate col (pw-1) into cols pw..stride, then row (ph-1) into rows ph..buf_h.
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

    let speed = 0i32;
    let sf = SpeedFeatures::set_allintra(speed, p.allow_screen_content_tools, false);
    let mut env = SbEncodeEnv {
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
        tx_type_costs: &real.tx_type_costs_y,
    };
    let pick_cfg = PickFrameCfg {
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
    let mut p = p;
    let (our_payload, our_len) = if tile_cols_log2 == 0 && tile_rows_log2 == 0 {
        // ---- SINGLE TILE (original path): port derives the loop-filter level ----
        let mut enc = OdEcEnc::new();
        // CEIL to cover partial edge SBs (n_sb_x/n_sb_y computed above from the
        // planes; == mi/16 exactly when the frame is 64-aligned).
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
            n_sb_y,
            n_sb_x,
            SB_MI,
            SB,
        );
        assert_eq!(
            trees.len(),
            (n_sb_x * n_sb_y) as usize,
            "{ctx}: pack_tile must walk every SB"
        );
        let our_tile_bytes = enc.done().to_vec();

        // Port-derived loop-filter level (same as the bd10 / bd8 harness).
        let mi_grid = build_lf_mi_grid(&trees, mi_rows, mi_cols, n_sb_x, SB_MI, SB);
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
        let derived_lf = pick_filter_level(&lf_frame, allintra, 0, false);
        p.loopfilter.filter_level = derived_lf.filter_level;
        p.loopfilter.filter_level_u = derived_lf.filter_level_u;
        p.loopfilter.filter_level_v = derived_lf.filter_level_v;

        let len = our_tile_bytes.len();
        (
            assemble_frame_obu_payload_single_tile(&p, tiles_log2, &our_tile_bytes),
            len,
        )
    } else {
        // ---- MULTI-TILE: the port encodes EACH tile itself (fresh entropy coder
        // + reset CDF context per tile, tile-boundary neighbour availability via
        // env.tile_*), then the derived multi-tile OBU assembler. The tile grid
        // (col_start_sb / row_start_sb, uniform spacing) comes from the real
        // parsed header. Loop-filter is left at the real parsed value: LF is a
        // post-filter that does NOT change the entropy-coded tile bytes, so this
        // isolates the per-tile search+pack+assembly (multi-tile LF derivation is
        // a follow-up). A tile-encode bug shows as differing tile bytes -> differing
        // tile-size prefixes -> a byte mismatch in the assembled frame.
        let ti = p.tile_info.clone();
        let mut tile_bytes: Vec<Vec<u8>> = Vec::with_capacity(ti.rows * ti.cols);
        for tr in 0..ti.rows {
            for tc in 0..ti.cols {
                let mi_col0 = ti.col_start_sb[tc] << ti.mib_size_log2;
                let mi_row0 = ti.row_start_sb[tr] << ti.mib_size_log2;
                let n_sb_cols = ti.col_start_sb[tc + 1] - ti.col_start_sb[tc];
                let n_sb_rows = ti.row_start_sb[tr + 1] - ti.row_start_sb[tr];
                env.tile_col_start = mi_col0;
                env.tile_row_start = mi_row0;
                env.tile_col_end = (ti.col_start_sb[tc + 1] << ti.mib_size_log2).min(mi_cols);
                env.tile_row_end = (ti.row_start_sb[tr + 1] << ti.mib_size_log2).min(mi_rows);
                let mut enc = OdEcEnc::new();
                let mut kf = KfFrameContext::default_for_qindex(qindex);
                let trees = pack_tile(
                    &mut enc,
                    &env,
                    &pick_cfg,
                    &pack_cfg,
                    &mut kf,
                    &mut recon_y,
                    &mut recon_u,
                    &mut recon_v,
                    mi_row0,
                    mi_col0,
                    n_sb_rows,
                    n_sb_cols,
                    SB_MI,
                    SB,
                );
                assert_eq!(
                    trees.len(),
                    (n_sb_rows * n_sb_cols) as usize,
                    "{ctx}: tile ({tr},{tc}) pack must walk every SB"
                );
                tile_bytes.push(enc.done().to_vec());
            }
        }
        let len: usize = tile_bytes.iter().map(|t| t.len()).sum();
        (
            assemble_multitile_frame_obu_payload_derived(&p, &tile_bytes),
            len,
        )
    };
    let matched = our_payload == frame_payload;
    if matched {
        eprintln!("{ctx}: TRUE END-TO-END BYTE MATCH");
    } else {
        let first_diff = our_payload
            .iter()
            .zip(frame_payload.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(our_payload.len().min(frame_payload.len()));
        let tile_data_start = real_bit_len.div_ceil(8);
        eprintln!(
            "{ctx}: MISMATCH at byte {first_diff} (ours={:?} real={:?}); header-region={} \
             our_tile.len()={} real_frame.len()={}",
            our_payload.get(first_diff),
            frame_payload.get(first_diff),
            tile_data_start,
            our_len,
            frame_payload.len(),
        );
    }
    CaseResult { matched }
}

/// Textured luma generator masked to `bd` bits: a two-axis gradient XORed with a
/// checkerboard high-frequency term so the residual / quant / recon path sees
/// genuine detail at every size (never an all-flat, all-skip frame).
fn tex_luma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        let base = ((r * 37 + cc * 23) as u32) & mask;
        let hf = if (r ^ cc) & 1 == 1 { mask / 12 } else { 0 };
        (base ^ hf) as u16
    }
}

/// Textured chroma generator masked to `bd` bits — **deliberately independent** of
/// [`tex_luma`]: different gradient slopes (19/29 vs 37/23) and a different HF
/// pattern (period-3 vs checkerboard) so the chroma is NOT an affine function of
/// luma. That defeats CfL's scaled-luma prediction and forces the real chroma
/// intra-mode / residual / recon / pack path to be exercised.
fn tex_chroma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        let base = ((r * 19 + cc * 29) as u32) & mask;
        let hf = if (r + cc) % 3 == 0 { mask / 20 } else { 0 };
        (base ^ hf) as u16
    }
}

/// Luma companion for the chroma-edge-filter witness: gentle, low-detail content
/// so the *luma* partition stays coarse and the chroma mode search (not luma) is
/// what drives the block layout the witness depends on.
fn witness_luma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| (((r + cc) as u32 * mask / 200) & mask) as u16
}

/// Chroma engineered to FIRE the per-block chroma intra edge filter
/// (`get_intra_edge_filter_type(xd, plane=1)`): 16-wide vertical bands alternate
/// between a SMOOTH bilinear gradient (the SMOOTH_PRED / SMOOTH_V / SMOOTH_H
/// family wins those blocks with near-zero residual) and a strong diagonal-stripe
/// field (a *directional* uv_mode wins). Every stripe band borders a SMOOTH band
/// on its left, so a chroma block in a stripe band is exactly the "directional
/// block whose above/left chroma neighbour is SMOOTH" case: C derives
/// `filter_type = 1` there, while the pre-#26 frozen SB-level value was 0 — the
/// two edge filters give different predicted chroma, hence different coded bytes.
/// This is the byte-exact witness for the re-encode (pack) path the
/// `partition_pick_diff` unit witness (search path) cannot reach.
fn smooth_dir_bands_chroma(mask: u32) -> impl Fn(usize, usize) -> u16 {
    move |r, cc| {
        if (cc / 16) & 1 == 0 {
            // Smooth bilinear gradient -> SMOOTH family (no high-frequency term).
            (((r + cc) as u32 * mask / 128) & mask) as u16
        } else {
            // Strong diagonal stripes -> a directional uv_mode.
            let s = ((r as u32).wrapping_add((cc as u32).wrapping_mul(3))) % 16;
            ((s * mask / 15) & mask) as u16
        }
    }
}

/// Print the per-cell match grid and assert every cell byte-matched real aomenc.
/// A mismatch fails loudly with the full grid and the failing cells — the
/// per-cell first-diff byte offset is already printed by `run_case`.
fn report_and_assert(label: &str, results: &[(String, bool)]) {
    eprintln!("\n=== {label} results ===");
    for (name, ok) in results {
        eprintln!("  {name}: {}", if *ok { "MATCH" } else { "MISMATCH" });
    }
    let failed: Vec<&String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| n)
        .collect();
    assert!(
        failed.is_empty(),
        "{}/{} {label} cells diverged from real aomenc: {:?}",
        failed.len(),
        results.len(),
        failed
    );
}

/// **4:4:4** (ss 0,0) bd8 ALLINTRA KEY: full-resolution chroma with distinct
/// textured luma AND chroma. Sizes 64/128/192 = 1×1 / 2×2 / 3×3 SB force real
/// multi-SB `pack_tile` + partition recursion; cq 12/32/63 spans the aggressive-web
/// qindex range (~48/128/232). This is the first gate proving the full-frame 4:4:4
/// integration (multi-SB pack + OBU assemble + frame loop-filter) is byte-exact.
#[test]
fn encoder_gate_444_bd8_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &sz in &[64usize, 128, 192] {
        for &cq in &[12i32, 32, 63] {
            let res = run_case(sz, sz, false, 0, 0, 2, cq, 8, &luma, &chroma, &chroma, 0, 0);
            results.push((format!("444 {sz}x{sz} cq{cq:>2}"), res.matched));
        }
    }
    report_and_assert("4:4:4 bd8", &results);
}

/// **MULTI-TILE** bd8 ALLINTRA KEY — exercises the single- AND multi-tile encode
/// paths. The single-tile gates above prove the port's own search+pack byte-matches
/// real aomenc for a 1-tile frame; this proves the SAME pipeline byte-matches when
/// the frame is a real MULTI-TILE grid: the port encodes EACH tile itself (fresh
/// entropy coder + reset CDF context per tile, tile-boundary neighbour availability
/// via env.tile_*), then the derived multi-tile OBU assembler stitches the tiles.
/// Content is the proven-byte-exact 4:4:4 `tex` pattern at 128x128 (= 2x2 SBs), so a
/// divergence here isolates a TILE-path bug, not a content near-tie.
/// (`obu_assemble_multitile_diff` proves the ASSEMBLER on C's own tile bytes; this
/// proves the port's per-tile ENCODE feeding that assembler.) Grid shapes
/// (tile_columns_log2, tile_rows_log2): (1,0) two 64x128 column tiles, (0,1) two
/// 128x64 row tiles, (1,1) four 64x64 tiles.
#[test]
fn encoder_gate_multitile_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(tcl, trl, shape) in &[(1i32, 0i32, "2x1"), (0, 1, "1x2"), (1, 1, "2x2")] {
        for &cq in &[12i32, 32, 63] {
            let res = run_case(128, 128, false, 0, 0, 2, cq, 8, &luma, &chroma, &chroma, tcl, trl);
            results.push((format!("444 128x128 tiles={shape} cq{cq:>2}"), res.matched));
        }
    }
    report_and_assert("multi-tile 4:4:4 bd8", &results);
}

/// **4:2:2** (ss 1,0) bd8 ALLINTRA KEY: horizontally-subsampled chroma, distinct
/// textured luma AND chroma. Same 64/128/192 × cq{12,32,63} grid as the 4:4:4 gate.
/// Proves the full-frame 4:2:2 integration is byte-exact — the 4:2:2 chroma has a
/// different block-geometry footprint (half-width, full-height planes) than 4:2:0,
/// so this exercises a distinct multi-SB pack path.
#[test]
fn encoder_gate_422_bd8_e2e() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &sz in &[64usize, 128, 192] {
        for &cq in &[12i32, 32, 63] {
            let res = run_case(sz, sz, false, 1, 0, 2, cq, 8, &luma, &chroma, &chroma, 0, 0);
            results.push((format!("422 {sz}x{sz} cq{cq:>2}"), res.matched));
        }
    }
    report_and_assert("4:2:2 bd8", &results);
}

/// **Witness — per-block chroma intra edge filter (#26).** 4:4:4 bd8 ALLINTRA KEY
/// with [`smooth_dir_bands_chroma`]: SMOOTH chroma bands abut strong directional
/// chroma bands, so many directional chroma blocks border a SMOOTH above/left
/// chroma neighbour — the case where C derives `get_intra_edge_filter_type(xd,
/// plane=1) = 1`. Before #26 the port froze the SB-level chroma `filter_type` at
/// 0 and predicted those blocks with the wrong intra edge filter, diverging the
/// coded chroma bytes. With the per-block recompute wired through BOTH the UV RD
/// search (`leaf_pick_sb_modes`) AND the pack re-encode (`encode_b_intra_dry`,
/// via `LeafWinner::uv_edge_filter_type`), every cell is byte-identical to real
/// aomenc. Sizes 64/128 (1×1 / 2×2 SB) × cq 12/32 keep partitions fine enough
/// (high-quality qindex) that directional chroma modes actually win alongside the
/// SMOOTH bands.
#[test]
fn encoder_gate_444_bd8_chroma_edge_filter_witness() {
    let luma = witness_luma(0xff);
    let chroma = smooth_dir_bands_chroma(0xff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &sz in &[64usize, 128] {
        for &cq in &[12i32, 32] {
            let res = run_case(sz, sz, false, 0, 0, 2, cq, 8, &luma, &chroma, &chroma, 0, 0);
            results.push((format!("444-edgefilter {sz}x{sz} cq{cq:>2}"), res.matched));
        }
    }
    report_and_assert("4:4:4 chroma edge filter witness", &results);
}

// ---- Real-image encoder gate ------------------------------------------------
// The synthetic generators above are hand-tuned to stress specific code paths;
// this gate feeds GENUINE image content — a small KEY frame decoded from the AV1
// conformance corpus (the same real vectors the decoder track is anchored on) —
// through the port's full encode pipeline vs real aomenc, byte-for-byte. It
// guards every landed encoder fix against real photographic/screen statistics
// that synthetic patterns may not cover. Corpus provisioning mirrors the decoder
// conformance gate (`AOM_CONFORMANCE_DIR` or `conformance/data`, CI-fetched at
// `--scope intra`); a missing vector FAILS LOUD with the fetch command — never a
// silent skip.

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
    assert!(data.len() >= 32 && &data[0..4] == b"DKIF", "not an IVF file");
    let hdr_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut off = hdr_len;
    let mut tus = Vec::new();
    while off + 12 <= data.len() {
        let sz =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 12; // 4-byte size + 8-byte timestamp
        assert!(off + sz <= data.len(), "IVF frame runs past end of file");
        tus.push(data[off..off + sz].to_vec());
        off += sz;
    }
    tus
}

/// **KB-6 repro (OPEN) — real-content encoder RD divergence at bd8 4:2:0.**
/// Decodes the first KEY frame of small real conformance vectors (`01-size`
/// family, intra scope) to genuine YUV via the C decode oracle, then runs the
/// port's full encode (`pack_tile` + assemble + LF) vs real aomenc byte-for-byte
/// on those real pixels at the vector's native subsampling/bit depth.
///
/// FINDING (2026-07-15): every synthetic e2e gate is byte-exact, but GENUINE
/// image content diverges across the whole quality range — the multi-SB 196x196
/// frame diverges at every cq (5..63) and the 1-SB 64x64 diverges except at two
/// coincidental cq points. The port codes FEWER symbols than aomenc (the KB-2
/// near-tie signature): real content triggers partition/mode/tx RD decisions the
/// hand-tuned synthetic patterns never did. Tracked as KB-6 (see CLAUDE.md).
///
/// PROMOTED 2026-07-16 to a FULL byte-match gate: all 30 real-content cells
/// (5 vectors/crops × cq 5/12/20/32/48/63) byte-match real aomenc. The
/// per-group asserts (1)/(1b)/(1c) keep the per-fix history + specific
/// regression messages; (2) is the all-cells `report_and_assert`. The last
/// cell (196² cq48) closed with the tokenize WRITE-ctx fix (encode_sb.rs —
/// C caches the pack's `(txb_skip_ctx, dc_sign_ctx)` from the persistent
/// clipped-stamp arrays in `av1_update_and_record_txb_context`, not from the
/// trellis's full-footprint local ta/tl).
#[test]
fn encoder_gate_real_image_e2e_kb6_repro() {
    let dir = corpus_dir();
    // Cell = (vector, crop_w, crop_h, off_x, off_y). crop_w == 0 means the FULL
    // frame; otherwise an SB-aligned (mult-of-64) crop at (off_x, off_y). Two
    // deliberate axes of coverage on the PRIMARY bd8 4:2:0 speed-0 KEY path:
    //   * CONTENT DIVERSITY: `01-size` is a frame-size test pattern; `00-quantizer`
    //     and `23-film` are real photographic / film-grain statistics — different
    //     content tips DIFFERENT speed-0 RD near-ties (exactly what surfaced KB-6).
    //     All three are FAMILY_SCOPE "intra" in xtask/conformance.py, so CI fetches
    //     them (`--scope intra`).
    //   * LOW PIXEL COUNT + CLEAN vs PARTIAL-SB: 64x64 (1 SB) and 128x128 (4 SB)
    //     crops are SB-aligned, so they exercise multi-SB RD near-ties WITHOUT the
    //     separate frame-edge partial-SB gap (the port's partition search does not
    //     yet model edge blocks; see KB-6 notes). The full 196x196 stays as the
    //     partial-SB characterization. Offsets are even (4:2:0 chroma alignment)
    //     and land in the textured interior, away from flat borders.
    let cells: &[(&str, usize, usize, usize, usize)] = &[
        ("av1-1-b8-01-size-64x64", 0, 0, 0, 0), // 1 SB, aligned — primary clean signal
        ("av1-1-b8-01-size-196x196", 0, 0, 0, 0), // partial-SB (documents the edge gap)
        ("av1-1-b8-00-quantizer-00", 64, 64, 96, 64), // photo, 1-SB aligned crop
        ("av1-1-b8-00-quantizer-00", 128, 128, 64, 64), // photo, 4-SB aligned crop
        ("av1-1-b8-23-film_grain-50", 64, 64, 96, 64), // film grain, 1-SB aligned crop
    ];
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(name, crop_w, crop_h, off_x, off_y) in cells {
        let path = dir.join(format!("{name}.ivf"));
        let ivf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{name}: conformance vector missing at {path:?} ({e}); fetch via \
                 `python3 xtask/conformance.py --fetch --scope intra`"
            )
        });
        let (fw, fh) = ivf_hdr_dims(&ivf);
        let tus = ivf_temporal_units(&ivf);
        let frame = c::ref_decode_av1_kf(&tus[0], fw, fh);
        let bd = frame.info[0] as u8;
        let mono = frame.info[1] != 0;
        let ss_x = frame.info[2] as usize;
        let ss_y = frame.info[3] as usize;
        let fcw = (fw + ss_x) >> ss_x; // full-frame chroma stride (tight row-major)
        let (y, u, v) = (frame.y, frame.u, frame.v);
        let (w, h) = if crop_w == 0 { (fw, fh) } else { (crop_w, crop_h) };
        assert!(
            off_x + w <= fw && off_y + h <= fh,
            "{name}: crop {w}x{h}@{off_x},{off_y} exceeds frame {fw}x{fh}"
        );
        assert!(
            off_x % 2 == 0 && off_y % 2 == 0,
            "{name}: crop offset must be chroma-aligned (even)"
        );
        let (cox, coy) = (off_x >> ss_x, off_y >> ss_y);
        let y_c = |r: usize, c: usize| y[(r + off_y) * fw + (c + off_x)];
        let u_c = |r: usize, c: usize| u[(r + coy) * fcw + (c + cox)];
        let v_c = |r: usize, c: usize| v[(r + coy) * fcw + (c + cox)];
        let fmt = match (mono, ss_x, ss_y) {
            (true, _, _) => "mono",
            (false, 1, 1) => "420",
            (false, 1, 0) => "422",
            (false, 0, 0) => "444",
            _ => "chroma",
        };
        let tag = if crop_w == 0 {
            format!("{name} {fmt}")
        } else {
            format!("{name} {fmt} {w}x{h}@{off_x},{off_y}")
        };
        for &cq in &[5i32, 12, 20, 32, 48, 63] {
            let res = run_case(w, h, mono, ss_x, ss_y, 2, cq, bd, &y_c, &u_c, &v_c, 0, 0);
            results.push((format!("{tag} cq{cq}"), res.matched));
        }
    }

    eprintln!("\n=== KB-6 real-image e2e map (MATCH = byte-exact vs real aomenc) ===");
    for (label, ok) in &results {
        eprintln!("  {label}: {}", if *ok { "MATCH" } else { "MISMATCH (KB-6)" });
    }
    let matched: Vec<&String> = results.iter().filter(|(_, ok)| *ok).map(|(n, _)| n).collect();
    let diverged: Vec<&String> = results.iter().filter(|(_, ok)| !*ok).map(|(n, _)| n).collect();
    eprintln!(
        "KB-6: {}/{} real-content cells byte-exact; {} diverge {:?}",
        matched.len(),
        results.len(),
        diverged.len(),
        diverged
    );

    // (1) PROMOTED byte-match gates — the real size-64x64 cells byte-match real
    // aomenc at EVERY cq (5/12/20/32/48/63). cq5/12/20/48/63 landed with the KB-6
    // luma re-encode edge-filter fix (per-block get_intra_edge_filter_type,
    // mirroring #26 for chroma); **cq32 landed 2026-07-16 with the AB-partition
    // HORZ_A nested-reuse fix** (partition_pick.rs: sub-block 1's
    // reuse_prev_rd_results_for_part_ab is nested under sub-block 0's readiness,
    // matching partition_search.c:3858-3868 — the port was reusing the split-context
    // DC+filter_intra winner for the HORZ_A top-right 8x8 where C re-searches it and
    // picks SMOOTH_H_PRED). Assert every one: a regression here is NOT the open KB-6
    // divergence (the OTHER cells) — it means a previously byte-exact real cell
    // broke, or the harness broke.
    for cq in [5, 12, 20, 32, 48, 63] {
        let label = format!("av1-1-b8-01-size-64x64 420 cq{cq}");
        let ok = results
            .iter()
            .find(|(n, _)| *n == label)
            .map(|(_, ok)| *ok)
            .expect("promoted cell present");
        assert!(
            ok,
            "regression: real cell `{label}` must byte-match real aomenc \
             (KB-6: cq5/12/20/48/63 = luma re-encode fix; cq32 = AB HORZ_A nested-reuse fix)"
        );
    }

    // (1b) NEWLY-PROMOTED byte-match gates — the KB-6 chroma has_top_right fix
    // (sub-8x8 chroma-reference DIRECTIONAL prediction fed the availability walk
    // the raw luma bsize/mi instead of scale_chroma_bsize + the chroma-ref adj mi;
    // reconintra.c:1637/1783, matching the bit-exact decoder aom-decode:2534/2044).
    // That took these three real cells byte-exact. A regression here is a real
    // cell breaking, NOT the open KB-6 divergence (the remaining cells below).
    for label in [
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cq5",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cq20",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cq20",
    ] {
        let ok = results
            .iter()
            .find(|(n, _)| n.as_str() == label)
            .map(|(_, ok)| *ok)
            .expect("promoted cell present");
        assert!(
            ok,
            "regression: real cell `{label}` must byte-match real aomenc \
             (KB-6 chroma has_top_right / scale_chroma_bsize fix)"
        );
    }

    // (1c) PROMOTED byte-match gates — PARTIAL-SB (frame-edge) full-frame
    // matches. 196 = 3*64+4, so the right column + bottom row are partial
    // superblocks; these cells prove the edge encode path end-to-end. Landed
    // in sequence: cq5 with the KB-6 frame-edge chunk series (true-196x196
    // harness, luma + chroma visible-area distortion clips, edge
    // partition-cost override); cq63 with the KB-4 OUTPUT_ENABLED tx_type_map
    // reset-leak fix (a2dd28e); **cq12/20/32 with the frame-edge
    // entropy-context stamp fix** — `av1_set_entropy_contexts` (blockd.c:29)
    // zeroes the beyond-visible TAIL of an edge txb's above/left footprint,
    // but the port's tile stamp (encode_sb.rs) wrote the cul across the full
    // footprint, poisoning out-of-frame columns that later edge blocks'
    // full-footprint `get_txb_ctx` reads OR in (wrong txb_skip_ctx → same
    // symbols on different-probability cdf rows → bottom-row stream desync) —
    // plus the edge partition-cost gather reading the frame-init `cm->fc`
    // table (partition_search.c:3415), not the per-SB-adapted one; **cq48
    // (the LAST cell) with the tokenize WRITE-ctx fix** — C's pack writes
    // each txb's coefficients with the `(txb_skip_ctx, dc_sign_ctx)` cached
    // by the tokenize walk (`av1_update_and_record_txb_context`,
    // encodetxb.c) from the PERSISTENT entropy arrays (edge-CLIPPED
    // `av1_set_entropy_contexts` stamps), while the trellis uses the encode
    // walk's local full-footprint stamps; the port used the trellis pair for
    // both, so an edge txb whose within-leaf neighbour footprint spans the
    // visible boundary could write its DC-sign on a different cdf row
    // (196² cq48: identical search decisions, bits diverged mid-tile).
    for cq in [5, 12, 20, 32, 48, 63] {
        let label = format!("av1-1-b8-01-size-196x196 420 cq{cq}");
        let ok = results
            .iter()
            .find(|(n, _)| *n == label)
            .map(|(_, ok)| *ok)
            .expect("promoted cell present");
        assert!(
            ok,
            "regression: real cell `{label}` must byte-match real aomenc \
             (KB-6 partial-SB frame-edge fixes: visible-area distortion clips + \
             edge partition cost + entropy-stamp tail-zero + tx_type_map reset leak)"
        );
    }

    // (2) PROMOTED 2026-07-16 — the full 30-cell real-content map byte-matches
    // real aomenc (the last cell, 196² cq48, closed with the tokenize
    // WRITE-ctx fix — see (1c)). This is now a FULL byte-match gate: any
    // real-content cell diverging is a regression.
    report_and_assert("KB-6 real-image e2e (30/30 promoted)", &results);
}

/// Print a per-cell map and assert an OPEN bug still reproduces. Shared by the
/// KB-4 / KB-5 characterization repros below: a committed reproduction that is
/// CI-green while the bug is open and FAILS (loudly, with a promote instruction)
/// the moment a fix makes the cells byte-match — never a weakened or skipped test.
fn assert_open_divergence(kb: &str, results: &[(String, bool)]) {
    eprintln!("\n=== {kb} repro map (MATCH = byte-exact vs real aomenc) ===");
    for (label, ok) in results {
        eprintln!("  {label}: {}", if *ok { "MATCH" } else { "MISMATCH" });
    }
    let diverged: Vec<&String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| n)
        .collect();
    eprintln!(
        "{kb}: {}/{} cells diverge {:?}",
        diverged.len(),
        results.len(),
        diverged
    );
    assert!(
        !diverged.is_empty(),
        "{kb} appears FIXED: all cells now byte-match real aomenc. Promote this repro \
         to an asserting byte-match gate (report_and_assert) and close {kb} in CLAUDE.md."
    );
}

/// **KB-4 bd10 non-4:2:0 chroma — byte-match gate (FIXED 2026-07-16).** At bit
/// depth 10 the port's encoded bitstream for 4:4:4 / 4:2:2 chroma (the non-4:2:0
/// subsamplings) now byte-matches real aomenc. The divergence was NOT a bd10
/// large-coefficient RD-scaling issue (the original KB-4 hypothesis) — it was the
/// **AB-partition HORZ_A nested-reuse bug** (partition_pick.rs: sub-block 1's
/// `reuse_prev_rd_results_for_part_ab` must be nested under sub-block 0's readiness,
/// partition_search.c:3858-3868). The same fix that closed the size-64x64 cq32 KB-6
/// near-tie made all four of these cells byte-exact. Promoted from a characterization
/// repro to an ASSERTING byte-match gate: a regression here means the AB nested-reuse
/// fix broke. The byte-exact bd10 mono+4:2:0 regime is covered by
/// `encoder_gate_bd10_diff`.
#[test]
fn encoder_gate_bd10_non420_e2e_kb4_repro() {
    // Full bd10-range textured luma+chroma (values 0..=1023), distinct so CfL
    // cannot trivialize chroma — the large-coefficient regime KB-4 lived in.
    let luma = tex_luma(0x3ff);
    let chroma = tex_chroma(0x3ff);
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(ss_x, ss_y, fmt) in &[(0usize, 0usize, "444"), (1usize, 0usize, "422")] {
        for &sz in &[64usize, 128] {
            let res = run_case(sz, sz, false, ss_x, ss_y, 2, 32, 10, &luma, &chroma, &chroma, 0, 0);
            results.push((format!("bd10-{fmt} {sz}x{sz} cq32"), res.matched));
        }
    }
    report_and_assert("KB-4 bd10 non-420", &results);
}

/// **KB-5 repro (OPEN) — lossless (cq0 / qindex 0) KEY encode divergence.** A
/// lossless allintra KEY frame diverges badly: (1) `run_case`'s single-pass
/// `read_uncompressed_header` skips the two-pass `coded_lossless` probe the parser
/// contract requires, so `p.coded_lossless=false` → the port runs its NON-lossless
/// encoder at qindex 0 (full DCT), and (2) there is no forward Walsh–Hadamard
/// transform in the encode path (the decoder has the inverse WHT; the encoder
/// applies `av1_fwd_txfm2d` unconditionally). Both must be fixed for byte-match —
/// see KB-5 in CLAUDE.md. Committed characterization: green while KB-5 is open.
#[test]
fn encoder_gate_lossless_cq0_e2e_kb5_repro() {
    let luma = tex_luma(0xff);
    let chroma = tex_chroma(0xff);
    // MONO lossless (cq0 → qindex 0, coded_lossless): FIXED — byte-exact vs real
    // aomenc. Closed by the forward WHT (av1_fwht4x4, routed through xform_quant for
    // coded-lossless TX_4X4) PLUS the entropy-context-propagation fix: the WRITTEN
    // txb_skip_ctx / dc_sign_ctx must derive from the REAL above/left neighbour
    // context even when the trellis is off (coded-lossless uses USE_B_QUANT_NO_TRELLIS),
    // matching C's write path (encodetxb.c:596-598 always reads pd->*_entropy_context).
    // The port had gated that seed on the trellis, so the right half of a VERT split
    // wrote ctx 1/0 instead of the real 3/1 and desynced the decoder. Hard-assert as a
    // regression guard — mono must stay byte-exact.
    let mono = run_case(64, 64, true, 1, 1, 2, 0, 8, &luma, &chroma, &chroma, 0, 0);
    assert!(
        mono.matched,
        "KB-5 mono lossless cq0 regressed — the forward-WHT + entropy-context fix must keep it byte-exact"
    );
    // 4:2:0 lossless: a residual chroma RD near-tie remains — the first 16×16 node
    // splits PARTITION_SPLIT in aomenc but PARTITION_NONE in the port, SPLIT losing by
    // ≤1 rdcost unit on the chroma cost (child-3 lands exactly at its remaining budget).
    // Tracked OPEN in KB-5 (CLAUDE.md). Kept as a characterization that FAILS (prompting
    // promotion to a byte-match gate) once the chroma near-tie is closed.
    let c420 = run_case(64, 64, false, 1, 1, 2, 0, 8, &luma, &chroma, &chroma, 0, 0);
    assert_open_divergence(
        "KB-5 (4:2:0 residual)",
        &[("lossless-420 64x64 cq0".to_string(), c420.matched)],
    );
}
