//! INTER-ENCODE chunk 2, sub-step 2f/2g: BYTE GATE on the inter-frame tile
//! bytes for the §3 low-delay zero-MV P.
//!
//! This is the first end-to-end byte-identity claim on the inter ENCODE side.
//! It proves the port's inter mode-info pack — symbol order, prediction
//! contexts, CDF selection and the arithmetic coder's finalization — reproduces
//! `aomenc`'s **actual coded tile bytes**, not a re-derivation of its own
//! decisions.
//!
//! ## What is being compared
//!
//! `aomenc` at the INTER-ENCODE-ROADMAP §3 config on a zero-MV translational
//! P (`MultiFrameEncodeCell::translational(base, 0, 0)`) codes frame 1 as ONE
//! `PARTITION_NONE` 64x64 block: `NEARESTMV`, reference `(LAST, NONE)`,
//! `SIMPLE_TRANSLATION`, `skip = 1`, no residual. That is not an assumption —
//! it was READ OUT of the stream with the instrumented libaom decoder
//! (`/root/aom-inspect/examples/inspect -bs -ts -m -r -mm`, plus `-a` for the
//! per-symbol accounting). The dump tool that produces the IVF for that is
//! `aom-bench`'s `dump_inter_stream` example.
//!
//! The test:
//! 1. runs the REAL `aomenc` on the 2-frame `[KEY, P]` clip;
//! 2. splits frame 1's OBU payload into (frame header | tile data) using the
//!    header bit-length from the port's own byte-exact header parse;
//! 3. re-encodes the tile from scratch with the port's writers, deriving every
//!    prediction context rather than copying it;
//! 4. asserts BYTE IDENTITY against `aomenc`'s tile.
//!
//! Nothing about the tile is bootstrapped: the CDFs are the spec defaults (the
//! P codes `primary_ref_frame = PRIMARY_REF_NONE`), the contexts come from the
//! port's context functions, and the mode/reference/MV come from the port's
//! ref-MV scan.
//!
//! ## Scope
//!
//! This gates the PACK layer for a single-block P. It does NOT yet gate the RD
//! loop choosing that block (`inter_rd::rd_pick_inter_mode_sb` is separately
//! unit-tested) nor multi-block CDF adaptation — both are the next rungs.

use aom_bench::{EncodeCell, MultiFrameEncodeCell};
use aom_dsp::entropy::dv_ref::{find_inter_mv_refs, DvNbr, DvTileBounds};
use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::header::{
    CdefHeader, FrameHeaderObu, FrameHeaderPrefix, FrameSizeHeader, LoopfilterHeader,
    RestorationHeader, TileInfoHeader,
};
use aom_dsp::entropy::header::{read_sequence_header_obu, read_uncompressed_header};
use aom_dsp::entropy::partition::{
    get_intra_inter_context, partition_plane_context, single_ref_p1_context, skip_txfm_context,
    write_partition, KfFrameContext,
};
use aom_dsp::entropy::rb::ReadBitBuffer;
use aom_encode::inter_costs::{InterFrameCdfs, SingleRefCtx, LAST_FRAME, NEARESTMV};
use aom_encode::inter_pack::{write_inter_leaf_mode_info, InterLeafCtx, InterLeafSyntax};

const KF_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
const KF_MODE_DELTAS: [i8; 2] = [0, 0];

/// `BLOCK_64X64`.
const BLOCK_64X64: usize = 12;
/// `PARTITION_NONE`.
const PARTITION_NONE: i32 = 0;

fn base(label: &str, w: usize, h: usize, mono: bool, cq: i32) -> EncodeCell {
    let content = |r: usize, c: usize| -> u16 { (40 + ((r * 3 + c * 5) % 160)) as u16 };
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = content(r, c);
        }
    }
    let (cw, ch) = if mono { (0, 0) } else { ((w + 1) >> 1, (h + 1) >> 1) };
    let cont_uv = |r: usize, c: usize| -> u16 { (110 + ((r * 2 + c) % 40)) as u16 };
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    if !mono {
        for r in 0..ch {
            for c in 0..cw {
                u[r * cw + c] = cont_uv(r, c);
                v[r * cw + c] = cont_uv(r, c) + 3;
            }
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono,
        ss_x: 1,
        ss_y: 1,
        usage: 0,
        cq_level: cq,
        speed: 0,
        bd: 8,
        y,
        u,
        v,
    }
}

fn walk(bytes: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let b0 = bytes[pos];
        let obu_type = u32::from((b0 >> 3) & 0xF);
        let ext = (b0 >> 2) & 1;
        let has_size = (b0 >> 1) & 1;
        assert_eq!(has_size, 1, "shim always sets obu_has_size_field");
        let mut p = pos + 1 + usize::from(ext == 1);
        let mut size = 0u64;
        let mut shift = 0;
        loop {
            let b = bytes[p];
            size |= u64::from(b & 0x7f) << shift;
            p += 1;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        let end = p + size as usize;
        out.push((obu_type, bytes[p..end].to_vec()));
        pos = end;
    }
    out
}

fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

fn tile_log2(blk: i32, target: i32) -> i32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

fn tile_limits(mi_cols: i32, mi_rows: i32, mib_size_log2: u32) -> TileInfoHeader {
    let sb_cols = (mi_cols + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_rows = (mi_rows + (1 << mib_size_log2) - 1) >> mib_size_log2;
    let sb_size_log2 = mib_size_log2 as i32 + 2;
    let max_width_sb = 4096 >> sb_size_log2;
    let max_tile_area_sb = (4096 * 2304) >> (2 * sb_size_log2);
    let min_log2_cols = tile_log2(max_width_sb, sb_cols);
    let min_log2_tiles = tile_log2(max_tile_area_sb, sb_cols * sb_rows).max(min_log2_cols);
    TileInfoHeader {
        mi_cols,
        mi_rows,
        mib_size_log2,
        min_log2_cols,
        max_log2_cols: tile_log2(1, sb_cols.min(64)),
        min_log2_rows: (min_log2_tiles - min_log2_cols).max(0),
        max_log2_rows: tile_log2(1, sb_rows.min(64)),
        max_width_sb,
        max_height_sb: (max_tile_area_sb / max_width_sb.max(1)).max(1),
        ..Default::default()
    }
}

/// Parse the real 2-frame stream: return frame 1's OBU payload and the exact
/// bit-length of its frame header, so the tile data can be split off without
/// hardcoding an offset.
fn frame1_payload_and_header_bits(stream: &[u8]) -> (Vec<u8>, usize, FrameHeaderObu, FrameHeaderObu) {
    let obus = walk(stream);
    let seq_payload = obus
        .iter()
        .find(|(t, _)| *t == 1)
        .map(|(_, p)| p.clone())
        .expect("sequence header OBU");
    let mut seq_rb = ReadBitBuffer::new(&seq_payload);
    let seq = read_sequence_header_obu(&mut seq_rb);
    let s = &seq.seq_header;
    let c = &seq.color_config;
    let num_planes = if c.monochrome { 1 } else { 3 };
    let mib_size_log2 = if s.sb_size_128 { 5u32 } else { 4u32 };
    let mi_cols = mi_dim(s.max_frame_width);
    let mi_rows = mi_dim(s.max_frame_height);

    let mut cfg = FrameHeaderObu {
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
        separate_uv_delta_q: c.separate_uv_delta_q,
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
            subsampling_x: c.subsampling_x,
            subsampling_y: c.subsampling_y,
            ..Default::default()
        },
        film_grain_params_present: seq.film_grain_params_present,
        ..Default::default()
    };
    cfg.might_allow_ref_frame_mvs = s.enable_ref_frame_mvs && s.enable_order_hint;
    cfg.might_allow_warped_motion = s.enable_warped_motion;

    let frames: Vec<&(u32, Vec<u8>)> = obus.iter().filter(|(t, _)| *t == 6 || *t == 3).collect();
    assert_eq!(frames.len(), 2, "expected [KEY, P] frame OBUs");
    let f1 = frames[1].1.clone();
    let mut rb = ReadBitBuffer::new(&f1);
    let real = read_uncompressed_header(&mut rb, &cfg);
    assert_eq!(real.prefix.frame_type, 1, "frame 1 must be INTER");
    let bits = rb.bit_position();
    (f1, bits, real, cfg)
}

/// Encode the §3 zero-MV P tile with the port: one `PARTITION_NONE` 64x64
/// inter leaf. Every context is DERIVED (no neighbours exist at (0,0), which the
/// context functions handle), and the CDFs start at the spec defaults because
/// the P codes `primary_ref_frame = PRIMARY_REF_NONE`.
fn port_encode_zero_mv_p_tile(qindex: i32, mi_rows: i32, mi_cols: i32) -> (Vec<u8>, i32, i32) {
    port_encode_p_tile_with_mode(qindex, mi_rows, mi_cols, NEARESTMV)
}

/// The same encode with an explicit inter mode, so a mutation test can prove the
/// byte gate DISCRIMINATES (a wrong mode must produce different bytes).
fn port_encode_p_tile_with_mode(
    qindex: i32,
    mi_rows: i32,
    mi_cols: i32,
    mode: i32,
) -> (Vec<u8>, i32, i32) {
    let mut kf = KfFrameContext::default_for_qindex(qindex);
    let mut inter_cdfs = InterFrameCdfs::defaults();
    let mut enc = OdEcEnc::new();

    let (mi_row, mi_col) = (0i32, 0i32);
    let bsize = BLOCK_64X64;

    // --- the ref-MV scan: gives NEARESTMV's motion vector AND the mode_context
    //     the inter-mode symbol is coded on. No neighbours at (0,0).
    let tile = DvTileBounds {
        mi_row_start: 0,
        mi_row_end: mi_rows,
        mi_col_start: 0,
        mi_col_end: mi_cols,
    };
    let refs = find_inter_mv_refs(
        LAST_FRAME,
        mi_row,
        mi_col,
        bsize,
        PARTITION_NONE as usize,
        false, // up_available
        false, // left_available
        tile,
        mi_rows,
        mi_cols,
        16,    // mib_size (SB64)
        false, // allow_ref_frame_mvs — §3 disables it
        None,
        (0, 0),
        0, // gm_wmtype = IDENTITY
        [0i8; 8],
        false, // allow_high_precision_mv (qindex 240 >= 128)
        false, // is_integer_mv
        |_ro: i32, _co: i32| -> DvNbr { DvNbr::default() },
    );
    assert_eq!(
        refs.nearest,
        (0, 0),
        "with no neighbours the ref-MV scan must yield a zero NEARESTMV"
    );

    // --- partition symbol: PARTITION_NONE at the 64x64 SB root ---
    let above_ctx = [0i8; 64];
    let left_ctx = [0i8; 64];
    let pctx = partition_plane_context(&above_ctx, &left_ctx, mi_row as usize, mi_col as usize, bsize);
    let cdf_len = aom_dsp::entropy::partition::partition_cdf_length(bsize);
    write_partition(
        &mut enc,
        &mut kf.partition[pctx as usize],
        cdf_len,
        PARTITION_NONE,
        true, // has_rows
        true, // has_cols
        bsize,
    );

    // --- the block's mode info ---
    let skip_ctx = skip_txfm_context(0, 0) as usize;
    let intra_inter_ctx = get_intra_inter_context(false, false, false, false);
    // No neighbours ⇒ all reference counts are zero ⇒ every single-ref context
    // takes the "equal counts" branch (1).
    let ref_counts = [0u8; 8];
    let p_equal = single_ref_p1_context(&ref_counts);
    let single_ref = SingleRefCtx {
        p1: p_equal,
        p2: p_equal,
        p3: p_equal,
        p4: p_equal,
        p5: p_equal,
        p6: p_equal,
    };
    let ctx = InterLeafCtx {
        skip_ctx,
        intra_inter_ctx,
        single_ref,
    };
    let info = InterLeafSyntax {
        skip_txfm: 1,
        is_inter: 1,
        ref_frame0: LAST_FRAME as i8,
        inter_mode: mode,
        mode_context: refs.mode_context,
    };
    write_inter_leaf_mode_info(
        &mut enc,
        &mut inter_cdfs,
        &mut kf.skip[skip_ctx],
        &ctx,
        &info,
    );

    (enc.done().to_vec(), pctx, refs.mode_context)
}

/// THE GATE: the port's coded tile bytes for the §3 zero-MV P must be
/// byte-identical to `aomenc`'s.
#[test]
fn zero_mv_p_tile_bytes_byte_exact_vs_aomenc() {
    let (w, h, cq) = (64usize, 64usize, 60i32);
    let cell = MultiFrameEncodeCell::translational(&base("zero_mv", w, h, false, cq), 0, 0);
    let stream = cell.c_encode_inter(false, false);

    let (f1, header_bits, real, _seq_cfg) = frame1_payload_and_header_bits(&stream);
    let header_bytes = header_bits.div_ceil(8);
    assert!(
        header_bytes < f1.len(),
        "frame 1 must carry tile data after its header"
    );
    let c_tile = &f1[header_bytes..];

    // Anti-vacuity: the reference really is the §3 low-delay P we think it is.
    assert_eq!(real.prefix.primary_ref_frame, 7, "P must use DEFAULT CDFs");
    assert!(!real.allow_high_precision_mv);
    assert_eq!(real.interp_filter, 0, "EIGHTTAP_REGULAR (non-switchable)");
    assert!(!real.switchable_motion_mode);
    assert!(!real.allow_ref_frame_mvs);

    let mi_rows = mi_dim(h as i32);
    let mi_cols = mi_dim(w as i32);
    let (port_tile, pctx, mode_ctx) =
        port_encode_zero_mv_p_tile(real.quant.base_qindex, mi_rows, mi_cols);

    assert_eq!(
        port_tile,
        c_tile,
        "\ninter P tile bytes differ.\n  port: {:02x?} ({} bytes)\n  aomenc: {:02x?} ({} bytes)\n\
         \n  qindex={} partition_ctx={} mode_context={}\n\
         \n  Expected symbol sequence (measured with the instrumented libaom decoder):\n\
         \n    partition(NONE) -> skip(1) -> is_inter(1) -> ref_frames(LAST) -> inter_mode(NEARESTMV)\n",
        port_tile,
        port_tile.len(),
        c_tile,
        c_tile.len(),
        real.quant.base_qindex,
        pctx,
        mode_ctx,
    );
}

/// Anti-vacuity for the gate above: `aomenc` must really be coding frame 1 as a
/// single-block all-skip INTER frame with a tiny tile. If a future config change
/// made the P frame something else, the byte gate would be testing the wrong
/// thing, so pin the shape.
#[test]
fn zero_mv_p_is_a_single_skip_block_frame() {
    let (w, h, cq) = (64usize, 64usize, 60i32);
    let cell = MultiFrameEncodeCell::translational(&base("zero_mv", w, h, false, cq), 0, 0);
    let stream = cell.c_encode_inter(false, false);
    let (f1, header_bits, real, _seq_cfg) = frame1_payload_and_header_bits(&stream);
    let tile_len = f1.len() - header_bits.div_ceil(8);
    assert_eq!(real.prefix.frame_type, 1);
    assert!(
        tile_len <= 4,
        "the all-skip zero-MV P tile must be tiny (got {tile_len} bytes) — a larger \
         tile means aomenc is coding residual or splitting, and the byte gate's \
         single-block model no longer describes the stream"
    );
    assert!(
        !real.tx_mode_select,
        "the §3 P codes TX_MODE_LARGEST, so no var-tx quadtree symbol is written"
    );
}

/// ANTI-VACUITY for the byte gate: it must be capable of FAILING. Coding the
/// same block as `GLOBALMV` instead of the measured `NEARESTMV` changes one
/// symbol of the inter-mode cascade, and the coded tile bytes must differ.
///
/// Without this, a gate that compared (say) two empty slices would pass and
/// prove nothing.
#[test]
fn tile_byte_gate_discriminates_a_wrong_mode() {
    use aom_encode::inter_costs::GLOBALMV;
    let (w, h, cq) = (64usize, 64usize, 60i32);
    let cell = MultiFrameEncodeCell::translational(&base("zero_mv", w, h, false, cq), 0, 0);
    let stream = cell.c_encode_inter(false, false);
    let (f1, header_bits, real, _seq_cfg) = frame1_payload_and_header_bits(&stream);
    let c_tile = &f1[header_bits.div_ceil(8)..];
    assert!(
        c_tile.len() >= 2,
        "the compared tile must carry real coded bytes (got {})",
        c_tile.len()
    );

    let mi_rows = mi_dim(h as i32);
    let mi_cols = mi_dim(w as i32);
    let (right, _, _) =
        port_encode_p_tile_with_mode(real.quant.base_qindex, mi_rows, mi_cols, NEARESTMV);
    let (wrong, _, _) =
        port_encode_p_tile_with_mode(real.quant.base_qindex, mi_rows, mi_cols, GLOBALMV);
    assert_eq!(right, c_tile, "control: NEARESTMV must match aomenc");
    assert_ne!(
        wrong, c_tile,
        "GLOBALMV must NOT match — otherwise the gate cannot detect a wrong mode"
    );
}

/// THE FULL-FRAME GATE: the port assembles frame 1's ENTIRE OBU payload —
/// derived frame header (sub-step 2a) + the byte-exact tile above — and it must
/// equal `aomenc`'s frame-1 payload byte for byte.
///
/// This composes two independently-verified pieces into one bitstream claim:
/// `inter_frame::derive_lowdelay_p_frame_header` (landed + gated by
/// `inter_header_derive_diff.rs`) and `inter_pack::write_inter_leaf_mode_info`
/// (gated above).
///
/// ## HONEST BOOTSTRAP — read before trusting this
///
/// Three header fields are RECON-DEPENDENT and are taken from the reference
/// stream rather than derived, exactly as `LowDelayPHeaderParams` documents:
/// the loop-filter levels/deltas, the CDEF header, and the frame `interp_filter`
/// (a per-frame RD decision, not a config constant). Their derivation is a
/// later sub-step. Everything else — frame type, order hint, primary_ref_frame,
/// refresh flags, ref_map_idx, allow_high_precision_mv, tx_mode, quant,
/// segmentation, and the whole tile — is port-derived.
///
/// The KEY-frame path bootstraps its header the same way (`port_encode` parses
/// the reference header), so this is the established shape for the project, not
/// a new concession. The tile bytes — the part this chunk is about — are
/// derived from nothing.
#[test]
fn zero_mv_p_frame_payload_byte_exact_vs_aomenc() {
    use aom_encode::inter_frame::{
        derive_lowdelay_p_frame_header, LowDelayPHeaderParams, TWO_FRAME_P_REF_MAP_IDX,
        TWO_FRAME_P_REFRESH_FLAGS,
    };
    use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
    use aom_encode::rc::base_qindex_lowdelay_p_from_cq;

    let (w, h, cq) = (64usize, 64usize, 60i32);
    let cell = MultiFrameEncodeCell::translational(&base("zero_mv", w, h, false, cq), 0, 0);
    let stream = cell.c_encode_inter(false, false);
    let (f1, header_bits, real, seq_cfg) = frame1_payload_and_header_bits(&stream);

    // The port's own qindex for this P — DERIVED from the cq, not parsed.
    let qindex = base_qindex_lowdelay_p_from_cq(cq);
    assert_eq!(
        qindex, real.quant.base_qindex,
        "the port's low-delay P qindex must match the coded one"
    );

    let p = LowDelayPHeaderParams {
        base_qindex: qindex,
        order_hint: 1,
        refresh_frame_flags: TWO_FRAME_P_REFRESH_FLAGS,
        ref_map_idx: TWO_FRAME_P_REF_MAP_IDX,
        disable_cdf_update: real.prefix.disable_cdf_update,
        reduced_tx_set_used: real.reduced_tx_set_used,
        // BOOTSTRAPPED (recon-dependent) — see the doc comment.
        interp_filter: real.interp_filter,
        loopfilter: real.loopfilter.clone(),
        cdef: real.cdef.clone(),
    };
    let derived = derive_lowdelay_p_frame_header(&seq_cfg, &p);

    let mi_rows = mi_dim(h as i32);
    let mi_cols = mi_dim(w as i32);
    let (port_tile, _, _) = port_encode_zero_mv_p_tile(qindex, mi_rows, mi_cols);
    let port_payload = assemble_frame_obu_payload_single_tile(&derived, 0, &port_tile);

    assert_eq!(
        port_payload.len(),
        f1.len(),
        "frame-1 payload length differs: port {} vs aomenc {} (header_bits {})",
        port_payload.len(),
        f1.len(),
        header_bits
    );
    assert_eq!(
        port_payload, f1,
        "\nframe-1 OBU payload differs.\n  port:   {:02x?}\n  aomenc: {:02x?}\n",
        port_payload, f1
    );
}

// MULTI-BLOCK (64x128): SUPERSEDED — the "two-superblock tile" premise was
// WRONG. Measured 2026-07-23 (instrumented sibling libaom, byte-inert): the
// §3 GOOD-mode stream codes **SB128**, so a 64x128 frame is ONE
// column-cropped 128x128 superblock — its root codes a gathered 2-way
// SPLIT/VERT symbol before the two visible 64x64 children. The hand-rolled
// SB64 walk this file used to pin could therefore never match aomenc (it
// omits the root symbol), and its per-symbol "block-1 CDF row" suspect list
// is obsolete. The live pin for the remaining divergence — a SEARCH-space
// gap (the port models no switchable interp-filter rate; C's
// `use_more_sharp_interp` biases the 64x128 VERT candidate onto SHARP and
// prices it out) — is
// `inter_e2e_search.rs::zero_mv_p_own_search_64x128_cropped_sb128_pinned_divergent`,
// which runs the REAL search+pack path end-to-end. The single-block gates
// above remain valid (a 64x64 frame is walk-degenerate: SB64 and SB128 emit
// identical symbols).
