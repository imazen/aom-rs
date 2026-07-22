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
use aom_encode::lf_search::{
    LfSearchFrame, build_lf_mi_grid, pick_filter_level, pick_filter_level_from_q,
};
use aom_encode::obu_assemble::assemble_frame_obu_payload_single_tile;
use aom_encode::pack::pack_tile;
use aom_encode::partition_pick::PickFrameCfg;
use aom_encode::rd::{EncMode, FrameUpdateType, TuneMetric, av1_compute_rd_mult_based_on_qindex};
use aom_encode::real_costs::derive_real_costs;
use aom_encode::speed_features::SpeedFeatures;
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

// ---- STRONG loop-filter-level content generators -------------------------
//
// Combined high-texture content (smooth gradient base + hard block-boundary
// edges + high-frequency texture) that drives real aomenc to STRONG nonzero
// loop-filter levels (>= 12) -- the regime the AB-probe checkerboards
// (`[8,8]`/`[8,3]`/`[7,16]`) never reach and the regime "HONEST GAP #2" once
// flagged. Amplitude/period tuned (see the discovery sweep in the git history
// of this file) so real deterministically codes a strong level; shared by
// `encoder_gate_lf_level_bit_exact_vs_real` (isolation on real's own recon)
// and `encoder_gate_e2e_rich_content_strong_lf` (full end-to-end byte match).
// `filter_level[0]` (vertical) responds to HORIZONTAL edges (h-stripes),
// `filter_level[1]` (horizontal) to VERTICAL edges (v-stripes) -- verified
// empirically by the discovery sweep.

/// Dominant horizontal stripes (period 4) + secondary vertical stripes
/// (period 12): both LF axes exercised, `filter_level[0]` strong (~15 at
/// cq58) -- the exact "15" of the `[15,6]`-shape the honest gap documents.
fn lf_hstripes4_vstripes12(r: usize, c: usize) -> u8 {
    let hbar = if (r / 4) % 2 == 0 { 40 } else { 200 };
    let vbar = if (c / 12) % 2 == 0 { 0 } else { 24 };
    (hbar + vbar).clamp(0, 255) as u8
}

/// Strong horizontal stripes (period 6) + secondary vertical stripes
/// (period 16) + gradient: drives BOTH axes nonzero and strong (`[6,20]` at
/// 128 cq58) -- a genuine both-axes-strong `[15,6]`-shape.
fn lf_hstripes6_vstripes16_grad(r: usize, c: usize) -> u8 {
    let grad = 18 + r * 90 / 256;
    let hbar = if (r / 6) % 2 == 0 { 0 } else { 80 };
    let vbar = if (c / 16) % 2 == 0 { 0 } else { 30 };
    (grad as i32 + hbar + vbar).clamp(0, 255) as u8
}

/// Strong vertical stripes (period 4) + secondary horizontal stripes
/// (period 12) + gradient: `filter_level[1]` strong (`[0,15]` at cq60).
fn lf_plaid_v4_h12_grad(r: usize, c: usize) -> u8 {
    let grad = 20 + r * 100 / 256;
    let vbar = if (c / 4) % 2 == 0 { 0 } else { 70 };
    let hbar = if (r / 12) % 2 == 0 { 0 } else { 22 };
    (grad as i32 + vbar + hbar).clamp(0, 255) as u8
}

/// Radial blob + diagonal hard bars + noise: drives a very strong
/// `filter_level[0]` (`[26,0]` at 256 cq63).
fn lf_radial_diagbars_noise(r: usize, c: usize) -> u8 {
    let dr = r as i32 - 64;
    let dc = c as i32 - 64;
    let d = ((dr * dr + dc * dc) as f64).sqrt();
    let base = (200.0 - d * 2.0).clamp(0.0, 255.0) as i32;
    let bar = if ((r + c) / 12) % 2 == 0 { 0 } else { 40 };
    let noise = ((r * 131 + c * 977) % 41) as i32 - 20;
    (base + bar + noise).clamp(0, 255) as u8
}

/// Diagonal gradient + vertical bars (period 16) + fine ripple: very strong
/// `filter_level[1]` (`[0,25]` at 256 cq63).
fn lf_diag_vbars16_ripple(r: usize, c: usize) -> u8 {
    let grad = 32 + (r + c) * 150 / 256;
    let bar = if (c / 16) % 2 == 0 { 0 } else { 45 };
    let ripple = if (r + c) % 2 == 0 { 14 } else { -14 };
    (grad as i32 + bar + ripple).clamp(0, 255) as u8
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
        0,
        0,
        content,
        |_r, _c| 128,
    )
}

/// Same as [`attempt_case_content`] but ALSO accepts caller-supplied chroma
/// content (`uv_content(row, col) -> u8`, same value used for both U and V)
/// instead of flat mid-grey 128 -- lets a case stress ONLY the chroma
/// search's decision space (luma stays whatever `content` produces,
/// typically flat) while keeping the luma partition/mode landscape trivial.
///
/// Thin wrapper over [`attempt_case_content_uv_sep`] feeding `uv_content` to
/// BOTH chroma planes (U == V) -- byte-identical to every existing caller.
#[allow(clippy::too_many_arguments)]
fn attempt_case_content_uv(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    cpu_used: i32,
    speed: i32,
    content: impl Fn(usize, usize) -> u8,
    uv_content: impl Fn(usize, usize) -> u8,
) -> bool {
    attempt_case_content_uv_sep(
        w, h, mono, ss_x, ss_y, usage, cq_level, cpu_used, speed, content, &uv_content,
        &uv_content,
    )
}

/// Core of the content-driven e2e attempt with SEPARATE caller-supplied U and V
/// chroma closures. Real decoded image content has distinct U/V planes (unlike
/// the synthetic U==V callers routed through [`attempt_case_content_uv`]), which
/// is what the real-content speed>=1 gate needs. Runs real aomenc
/// `--cpu-used=cpu_used` for the reference bytes, then this port's own
/// `SpeedFeatures::set_allintra(speed)` search+pack, and returns whether the
/// assembled OBU_FRAME payload byte-matches. bd8 only (matches the harness).
#[allow(clippy::too_many_arguments)]
fn attempt_case_content_uv_sep(
    w: usize,
    h: usize,
    mono: bool,
    ss_x: usize,
    ss_y: usize,
    usage: u32,
    cq_level: i32,
    cpu_used: i32,
    speed: i32,
    content: impl Fn(usize, usize) -> u8,
    u_content: impl Fn(usize, usize) -> u8,
    v_content: impl Fn(usize, usize) -> u8,
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
                u[r * cw + col] = u16::from(u_content(r, col));
                v[r * cw + col] = u16::from(v_content(r, col));
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
        cpu_used,
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

    // Speed features for this cpu-used level (all-intra path). At speed 0 this
    // reproduces the frozen hardcoded values EXACTLY; at speed >= 1 it applies
    // the transcribed `set_allintra_*` deltas. GOOD usage keeps the frozen
    // speed-0 policy (the GOOD setter is out of the all-intra slice).
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
        pol: &if allintra {
            sf.tx_type_search_policy(false, 0)
        } else {
            assert_eq!(speed, 0, "speed>0 e2e harness is all-intra only");
            TxTypeSearchPolicy::speed0_good()
        },
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
        // set_max_min_partition_size (partition_strategy.h:214): AOMMIN(sf
        // default_max, CLI cap BLOCK_128X128 (unset --max-partition-size),
        // sb_size). Through speed 5 this resolves to BLOCK_64X64 == SB — the
        // same consumer outcomes as the former hardcoded 15 on this 64-SB
        // envelope (nothing exceeds SB; the pc_index==3 equality arm needs
        // bsize == max, impossible at 12 and 15 alike for sub-root nodes).
        // At allintra speed>=6, `default_max_partition_size = BLOCK_32X32`
        // (:546) makes av1_prune_partitions_by_max_min_bsize force
        // square-split-only at the 64x64 SB root.
        max_partition_size: if allintra {
            sf.default_max_partition_size.min(15).min(SB)
        } else {
            15
        },
        min_partition_size: 0, // BLOCK_4X4: the true aomenc default (unset --min-partition-size)
        enable_1to4_partitions: true, // the true aomenc default (unset --enable-1to4-partitions)
        enable_ab_partitions: true, // the true aomenc default (unset --disable-ab-partition-type)
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
        delta_lf_present: false,
    };
    // allintra `lpf_pick` is DUAL (`LPF_PICK_FROM_FULL_IMAGE`) for speed 0..=3,
    // NON_DUAL for speed 4/5 (speed_features.c:496), and the closed-form
    // LPF_PICK_FROM_Q at speed >= 6 (:559 -- `pick_filter_level_from_q`,
    // oracle-validated vs real cpu-6 header levels by
    // `speed6_prep_lf_from_q_matches_real_aomenc`). See lf_search docs.
    let derived_lf = if allintra && speed >= 6 {
        pick_filter_level_from_q(qindex, bd, allintra, 0)
    } else {
        pick_filter_level(&lf_frame, allintra, 0, allintra && speed >= 4)
    };
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
        delta_lf_present: false,
    };
    let d = pick_filter_level(&lf, allintra, 0, false);
    (
        (d.filter_level, d.filter_level_u, d.filter_level_v),
        (
            hdr_real.loopfilter.filter_level,
            hdr_real.loopfilter.filter_level_u,
            hdr_real.loopfilter.filter_level_v,
        ),
    )
}

/// ASSERTED loop-filter-LEVEL bit-exactness gate — WEAK **and** STRONG levels.
/// For each case, fed real aomenc's OWN decoded pre-filter reconstruction + mi
/// grid (via [`lf_derived_vs_real_on_real_recon`]), this port's
/// `pick_filter_level` MUST reproduce real's coded level EXACTLY. Feeding
/// real's own pixels isolates the loop-filter-LEVEL SEARCH from the
/// encoder-side reconstruction: any residual e2e divergence on the same
/// content is therefore reconstruction, not the LF search.
///
/// Two regimes, both asserted:
/// - **Weak** (AB-probe checkerboards, 64x64 cq32): `[8,8]` / `[8,3]` /
///   `[7,16]` / `[8,8]` — the low-level nonzero search path.
/// - **Strong** (combined high-texture, 128/256 cq58-63): `[15,0]` / `[6,20]`
///   (both axes) / `[0,15]` / `[0,25]` / `[26,0]` — the strong-filter regime
///   "HONEST GAP #2" once flagged as unverified. This port derives every one
///   of them bit-exactly on real's own recon, so the strong-LF search is
///   CORRECT; the earlier honest-stop was reconstruction (since fixed by the
///   coeff-trellis + partition-RDO + `INTERNAL_COST_UPD_SB` per-SB cost
///   updates), not an LF-search bug. `encoder_gate_e2e_rich_content_strong_lf`
///   confirms that same rich content now byte-matches END-TO-END.
///
/// A `saw_strong` guard (a real level >= 12 actually exercised) keeps the
/// strong regime anti-vacuous; `saw_nonzero` keeps the weak regime honest.
#[test]
fn encoder_gate_lf_level_bit_exact_vs_real() {
    // (name, w, h, cq, content). Weak AB cases stay at 64x64 cq32; strong
    // cases at their discovered (size, cq) operating points.
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, usize, usize, i32, fn(usize, usize) -> u8)] = &[
        // --- WEAK regime (AB-probe checkerboards) ---
        (
            "top split / bottom flat",
            64,
            64,
            32,
            ab_top_split_bottom_flat,
        ),
        (
            "top flat / bottom split",
            64,
            64,
            32,
            ab_top_flat_bottom_split,
        ),
        (
            "left split / right flat",
            64,
            64,
            32,
            ab_left_split_right_flat,
        ),
        (
            "left flat / right split",
            64,
            64,
            32,
            ab_left_flat_right_split,
        ),
        // --- STRONG regime (combined high-texture) ---
        (
            "hstripes4+vstripes12 (vert~15)",
            128,
            128,
            58,
            lf_hstripes4_vstripes12,
        ),
        (
            "hstripes6+vstripes16+grad (both axes)",
            128,
            128,
            58,
            lf_hstripes6_vstripes16_grad,
        ),
        (
            "plaid v4/h12+grad (horz~15)",
            128,
            128,
            60,
            lf_plaid_v4_h12_grad,
        ),
        (
            "diag+vbars16+ripple (horz~25)",
            256,
            256,
            63,
            lf_diag_vbars16_ripple,
        ),
        (
            "radial+diagbars+noise (vert~26)",
            256,
            256,
            63,
            lf_radial_diagbars_noise,
        ),
    ];
    let mut saw_nonzero = false;
    let mut saw_strong = false;
    for &(name, w, h, cq, content) in cases {
        let (derived, real) = lf_derived_vs_real_on_real_recon(w, h, true, 1, 1, 2, cq, content);
        eprintln!(
            "lf-level gate [{name} {w}x{h} cq{cq}]: derived (fl={:?} u={} v={}) -- real coded \
             (fl={:?} u={} v={})",
            derived.0, derived.1, derived.2, real.0, real.1, real.2
        );
        saw_nonzero |= real.0 != [0, 0] || real.1 != 0 || real.2 != 0;
        saw_strong |= real.0[0] >= 12 || real.0[1] >= 12;
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
    assert!(
        saw_strong,
        "this gate must exercise at least one STRONG real LF level (>= 12), else it proves nothing \
         about the strong-filter search path (HONEST GAP #2's regime)"
    );
}

/// **ASSERTED rich-content STRONG-LF end-to-end byte match (promotion of the
/// multi-SB agent's honest-stopped variant).** The SAME combined high-texture
/// generators that drive real aomenc to STRONG loop-filter levels (asserted in
/// [`encoder_gate_lf_level_bit_exact_vs_real`]: `[15,0]` / `[6,20]` / `[0,15]`
/// / `[0,25]` / `[26,0]`) now produce a BYTE-IDENTICAL `OBU_FRAME` payload
/// end-to-end vs real aomenc — the port's OWN search + reconstruction + its
/// OWN `pick_filter_level` LF derivation, all the way to the coded bytes.
///
/// This was "HONEST GAP #2": a combined-content probe drove real to strong LF
/// levels where the port appeared to disagree. The isolation gate above proves
/// the LF SEARCH was never wrong (it derives real's strong level exactly on
/// real's own recon); the earlier divergence was the port's reconstruction,
/// which the coeff-trellis + partition-RDO + per-SB `INTERNAL_COST_UPD_SB` cost
/// updates have since made accurate enough that this rich content byte-matches.
///
/// Anti-vacuous by construction: every generator here is the SAME `fn` the LF
/// gate asserts drives a real level >= 12, so a flat/weak regression would fail
/// that gate; here it would additionally have to still byte-match, which it
/// won't if reconstruction drifts.
#[test]
fn encoder_gate_e2e_rich_content_strong_lf() {
    // (name, w, h, cq, content) — the strong-LF generators that ALSO byte-match
    // end-to-end (discovery sweep: 15/16 strong cells matched e2e; these 5 are
    // the robust subset, spanning vert-strong / both-axes / horz-strong / very
    // strong on each axis, at 128 and 256).
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, usize, usize, i32, fn(usize, usize) -> u8)] = &[
        (
            "hstripes4+vstripes12 (vert~15)",
            128,
            128,
            58,
            lf_hstripes4_vstripes12,
        ),
        (
            "hstripes6+vstripes16+grad (both axes)",
            128,
            128,
            58,
            lf_hstripes6_vstripes16_grad,
        ),
        (
            "plaid v4/h12+grad (horz~15)",
            128,
            128,
            60,
            lf_plaid_v4_h12_grad,
        ),
        // KB-2: same generator as the cq63 case below, at cq62 (qindex 249,
        // screen_content auto-detected, real LF [1,17]). This cell exposed the
        // frozen-`filter_type` bug — the port never re-derived the intra edge
        // filter type (get_intra_edge_filter_type) per block, so a SMOOTH
        // VERT_4 strip-0 neighbour did not raise strip-1's angled-prediction
        // edge-filter strength; the resulting model-RD over-pruned V_PRED
        // angle_delta=-1 and flipped the SB(32,32) partition. Fixed in
        // partition_pick.rs (per-block filter_type recompute).
        (
            "diag+vbars16+ripple cq62 (KB-2)",
            256,
            256,
            62,
            lf_diag_vbars16_ripple,
        ),
        (
            "diag+vbars16+ripple (horz~25)",
            256,
            256,
            63,
            lf_diag_vbars16_ripple,
        ),
        (
            "radial+diagbars+noise (vert~26)",
            256,
            256,
            63,
            lf_radial_diagbars_noise,
        ),
    ];
    let mut matched = 0usize;
    for &(name, w, h, cq, content) in cases {
        eprintln!("--- rich strong-LF e2e {w}x{h} [{name}] cq{cq} ---");
        if attempt_case_content(w, h, true, 1, 1, 2, cq, content) {
            matched += 1;
        }
    }
    eprintln!(
        "encoder_gate_e2e_rich_content_strong_lf: {matched}/{} rich strong-LF cases byte-identical \
         end-to-end",
        cases.len()
    );
    assert_eq!(
        matched,
        cases.len(),
        "every rich strong-LF case must byte-match real aomenc end-to-end -- these are the \
         promotion of HONEST GAP #2's honest-stopped rich-content variant; a mismatch here is a \
         genuine reconstruction regression (the strong-LF SEARCH itself is separately proven \
         bit-exact by encoder_gate_lf_level_bit_exact_vs_real)"
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
/// **ASSERTED low-qindex speed-0 end-to-end byte match — closes the low-q
/// coverage gap.** Every other `encoder_gate_e2e_*` speed-0 gate runs at high
/// qindex (cq58–63 → qindex 232–255); this one sweeps the AGGRESSIVE-WEB range
/// (cq8–30 → qindex 32–120 via the `q*4` map; cq17→68 is under the boosted-KEY
/// `qindex_thresh=70`, cq35→140 the non-boosted thresh) that CLAUDE.md weights
/// most, across 3 partition-diverse textured generators, <720p, monochrome,
/// speed 0.
///
/// Motivated by task #27: `av1_set_speed_features_qindex_dependent` sets
/// `model_based_prune_tx_search_level = 0` for `{<720p, base_qindex ≤ thresh}`
/// while the port keeps 1 — but that field is INERT on the all-intra KEY path
/// (the C gate is inside `av1_pick_recursive_tx_size_type_yrd`, `is_inter_block`
/// only; the port never reads it), so it cannot cause a divergence. This gate
/// proves the point empirically AND, more usefully, gives the previously-absent
/// low-q regime a real regression guard. Anti-vacuous: the generators drive
/// genuine partition/tx/coeff decisions (they are the same content family the
/// strong-LF gate uses), so a low-q reconstruction/search regression fails here.
/// (Tiny-size and 4:2:0 low-q are follow-ups; 4:2:0 waits on the #26 chroma
/// `filter_type` item.)
#[test]
fn encoder_gate_e2e_low_qindex_speed0() {
    // 4 cq points spanning the range (qindex 32/64/96/120 — cq8/16 under the
    // boosted-KEY thresh 70, all under the non-boosted 140) x 3 partition-diverse
    // generators = 12 cells. A proven subset of the 24-cell probe (all matched);
    // trimmed from 6x4 to keep per-platform CI runtime reasonable.
    #[allow(clippy::type_complexity)]
    let content: &[(&str, fn(usize, usize) -> u8)] = &[
        ("diag+vbars16+ripple", lf_diag_vbars16_ripple),
        ("hstripes6+vstripes16+grad", lf_hstripes6_vstripes16_grad),
        ("radial+diagbars+noise", lf_radial_diagbars_noise),
    ];
    let mut matched = 0usize;
    let mut total = 0usize;
    for &cq in &[8, 16, 24, 30] {
        for &(name, gen_fn) in content {
            total += 1;
            eprintln!("--- low-q speed-0 e2e 256x256 [{name}] cq{cq} ---");
            if attempt_case_content(256, 256, true, 1, 1, 2, cq, gen_fn) {
                matched += 1;
            }
        }
    }
    eprintln!("encoder_gate_e2e_low_qindex_speed0: {matched}/{total} low-q cases byte-identical");
    assert_eq!(
        matched, total,
        "every low-qindex (cq8-30 / qindex 32-120) speed-0 all-intra case must byte-match real \
         aomenc end-to-end -- this is the aggressive-web low-q regime that was previously untested \
         (all other speed-0 gates are qindex>=232); a mismatch here is a genuine low-q \
         search/reconstruction divergence"
    );
}

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
        if attempt_case_content_uv(64, 64, false, 1, 1, 2, cq, 0, 0, |_r, _c| 128, uv_content) {
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
/// **Coverage: ALL 16 of 16 swept (w, content, cq) cells byte-match, ASSERTED
/// here.** The 4 steepest-content cells that once diverged (smooth diagonal
/// ramp -- energy along both axes -- at cq32/cq48, plus the 256px vertical
/// gradient at cq32) are now byte-exact too: their divergence was the missing
/// `INTERNAL_COST_UPD_SB` per-superblock cost update (speed 0's default). Real
/// libaom re-derives BOTH the coefficient cost tables (`av1_fill_coeff_costs`)
/// AND the mode-rate tables (`av1_fill_mode_rates`) from the adapting tile CDF
/// at the start of every SB; the port now does the same via a per-SB
/// `derive_real_costs(kf, ..)` in `pack_tile`. On the later superblocks of
/// steep content -- whose CDFs have adapted enough to move the tables -- the
/// stale frame-init costs had flipped near-tie coefficient (code-vs-skip) and
/// intra-mode (DC vs directional/PAETH) RD decisions; the per-SB update tracks
/// the adaptation and both flavors now match. `decode_diff_multisb.rs` (a
/// committed diagnostic) confirms the decoded partition trees + every leaf's
/// mode/tx fields are identical, not just the raw bytes.
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

    // The FULL multi-SB grid (256x256 = 16 SB64, 512x512 = 64 SB64): 4 content
    // families x 2 sizes x cq32/cq48 = 16 cells, EVERY one byte-identical
    // end-to-end. The 4 steep-content cq32/cq48 cells this gate once excluded
    // (256x256 vgrad+diag cq32; 512x512 diag cq32+cq48) now match too, once the
    // `INTERNAL_COST_UPD_SB` per-superblock cost update was ported for BOTH the
    // coefficient tables (`av1_fill_coeff_costs`) AND the mode-rate tables
    // (`av1_fill_mode_rates`) -- see `pack_tile`'s per-SB `derive_real_costs(kf,
    // ..)` and `decode_diff_multisb.rs`. On steep diagonal/gradient content the
    // stale frame-init tables had flipped near-tie coefficient (code-vs-skip) and
    // intra-mode (DC vs directional/PAETH) decisions on the later superblocks,
    // whose adapting CDFs the per-SB update tracks.
    let winners: &[(usize, usize, &str, i32)] = &[
        // 256x256 (16 SB64)
        (256, 256, "flat 128", 32),
        (256, 256, "flat 128", 48),
        (256, 256, "soft wide two-tone L/R split", 32),
        (256, 256, "soft wide two-tone L/R split", 48),
        (256, 256, "smooth vertical gradient", 32),
        (256, 256, "smooth vertical gradient", 48),
        (256, 256, "smooth diagonal ramp", 32),
        (256, 256, "smooth diagonal ramp", 48),
        // 512x512 (64 SB64)
        (512, 512, "flat 128", 32),
        (512, 512, "flat 128", 48),
        (512, 512, "soft wide two-tone L/R split", 32),
        (512, 512, "soft wide two-tone L/R split", 48),
        (512, 512, "smooth vertical gradient", 32),
        (512, 512, "smooth vertical gradient", 48),
        (512, 512, "smooth diagonal ramp", 32),
        (512, 512, "smooth diagonal ramp", 48),
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
        "every multi-SB case (256x256 + 512x512, flat / two-tone / gradient / diagonal, cq32 + \
         cq48 -- all 16) must byte-match real aomenc end-to-end -- a mismatch here is a genuine \
         regression (the steep-content cq32/cq48 cells that once diverged are fixed by the per-SB \
         `INTERNAL_COST_UPD_SB` coeff+mode cost update in `pack_tile`; see this fn's module doc + \
         STATUS.md)"
    );
}

/// Gate 2 (`aomenc --cpu-used=1`) — the all-intra KEY speed-1 path. Reuses the
/// full e2e derivation but with `ref_encode_av1_kf(cpu_used=1)` and port config
/// from `SpeedFeatures::set_allintra(1, ..)`.
///
/// FLAT content is asserted: on EOB=0 blocks the speed-1 sf deltas (partition
/// CNN prune, top-N intra model, 2D tx-type prune, coeff-opt level, tx-domain
/// distortion, ...) are all no-ops, so a byte match here proves the oracle
/// (cpu-used=1) + the `SpeedFeatures` wiring are correct end-to-end.
#[test]
fn encoder_gate_speed1_flat_allintra() {
    let sizes = [(64usize, 64usize), (128, 128), (256, 256)];
    let mut matched = 0usize;
    let mut total = 0usize;
    for &(w, h) in &sizes {
        for &cq in &[32i32, 48] {
            total += 1;
            let ok =
                attempt_case_content_uv(w, h, true, 1, 1, 2, cq, 1, 1, |_r, _c| 128, |_r, _c| 128);
            eprintln!(
                "speed1 FLAT {w}x{h} cq{cq}: {}",
                if ok { "MATCH" } else { "DIFF" }
            );
            if ok {
                matched += 1;
            }
        }
    }
    assert_eq!(
        matched, total,
        "every flat cpu-used=1 all-intra case must byte-match real aomenc"
    );
}

/// Gate 2 (`aomenc --cpu-used=1`) — gentle-slope textured all-intra content
/// (the families that byte-match at speed 0 in `encoder_gate_e2e_multi_sb_scale`),
/// re-run with `ref_encode_av1_kf(cpu_used=1)` + `SpeedFeatures::set_allintra(1)`.
/// These carry REAL coefficients, so they exercise the speed-1 tx path:
/// `adaptive_txb_search_level` 1→2, `skip_tx_search` 0→1, `perform_coeff_opt`
/// 1→2 (coeff-opt dist threshold 3200→1728), and `tx_domain_dist_level` 0→1
/// (transform-domain distortion during the tx-type search + the
/// `calc_pixel_domain_distortion_final` pixel-domain recompute of the winner).
///
/// The `winners` below byte-match end-to-end; the excluded steep cell is the
/// next localization target (documented under the list), exactly as the speed-0
/// `encoder_gate_e2e_multi_sb_scale` gate once excluded its steep cq32 cells
/// before the per-SB cost update fixed them.
#[test]
fn encoder_gate_speed1_textured_allintra() {
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
            "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            other => panic!("unknown {other:?}"),
        }
    }
    // Byte-identical to real aomenc --cpu-used=1 end to end.
    let winners: &[(usize, usize, &str, i32)] = &[
        (128, 128, "two-tone", 48),
        (128, 128, "vgrad", 48),
        (128, 128, "diag", 48),
        (128, 128, "vgrad", 32),
        (256, 256, "two-tone", 48),
        (256, 256, "vgrad", 48),
        (256, 256, "diag", 48),
        // PROMOTED — the last excluded cpu-used=1 cell now byte-matches (KB-3
        // FIXED). Root cause (found via isolated sibling-libaom RD instrumentation
        // dumping C's per-candidate RD at SB(0,0) 64×64): a missing speed-1
        // partition prune, NOT a learned-model prune. C's NONE/SPLIT RD matched
        // the port EXACTLY (NONE rdcost 7427690), but C never evaluated the
        // rectangular partitions while the port did — and the port's HORZ
        // (rdcost 7058801) beat NONE, so the port wrongly picked PARTITION_HORZ.
        // C disables rect here via the "square-partition-only" rect kill
        // (partition_search.c:5749): `if (bsize > use_square_partition_only
        // _threshold) { partition_rect_allowed[HORZ] &= !has_rows; [VERT] &=
        // !has_cols; }`. That threshold is a framesize-DEPENDENT ALLINTRA speed
        // feature = BLOCK_64X64 sub-480p at speed 0 (so `bsize>64X64` never holds
        // in a <=64X64 SB — the reason speed-0 never needed it) but BLOCK_32X32
        // at speed >= 1, which kills rect on the 64X64 SB. Now wired in
        // rd_pick_partition_real (use_square_partition_only_threshold_allintra).
        (256, 256, "vgrad", 32),
    ];
    let mut matched = 0usize;
    for &(w, h, name, cq) in winners {
        let content = content_for(w, h, name);
        let ok = attempt_case_content_uv(
            w,
            h,
            true,
            1,
            1,
            2,
            cq,
            1,
            1,
            |r, c| content(r, c),
            |_r, _c| 128,
        );
        eprintln!(
            "speed1 {name} {w}x{h} cq{cq}: {}",
            if ok { "MATCH" } else { "DIFF" }
        );
        if ok {
            matched += 1;
        }
    }
    assert_eq!(
        matched,
        winners.len(),
        "every gentle-slope textured cpu-used=1 all-intra winner must byte-match real aomenc \
         (the tx-policy speed-1 deltas + calc_pixel_domain_distortion_final recompute)"
    );
}

/// **ASSERTED speed-1 (cpu-used=1) witness for the two #25 latent fixes.**
/// Before #25 the port carried the speed-0 tx/partition speed-feature values
/// into the speed-1 search: the rectangular tx-size depth-search floor
/// `intra_tx_size_search_init_depth_rect` stayed 0 (C sets 1 at speed >= 1,
/// speed_features.c:409) and the 4-partition ML-prune row
/// `ml_4_partition_search_level_index` stayed 0 (C sets 1 at speed >= 1,
/// speed_features.c:210). Both are now threaded from the resolved
/// `SpeedFeatures`: Feature 2 via `TxTypeSearchPolicy::intra_tx_size_init_depth_
/// {rect,sqr}`, Feature 1 via `predict_4partition_prune`'s `level_index` param.
///
/// These three ALLINTRA cells diverged from real aomenc pre-#25 and byte-match
/// after. Per-fix isolation (reverting one fix at a time) attributes them:
///   - `hstripes6+vstripes16 cq8` — needs BOTH fixes: re-diverges if EITHER the
///     rect init-depth (F2) OR the 4-partition LEVEL (F1) is reverted.
///   - `diag+vbars16 cq32`, `hstripes6+vstripes16 cq56` — need the rect
///     init-depth fix (F2): re-diverge if F2 is reverted; F1 alone leaves them
///     matching. vbars16/vstripes16 produce VERT_4 -> 16x64 rect leaves whose
///     tx-size winner moves when the extra smallest sub-tx (TX_16X16, which C
///     skips at speed >= 1) is or isn't searched.
///
/// Together the three give per-feature revert coverage: this gate FAILS if
/// either #25 fix is reverted. Anti-vacuous — three real end-to-end encodes vs
/// aomenc, each confirmed DIFF before the fix. (17 of the 24 swept
/// vbars/vstripes speed-1 cells still diverge for reasons outside #25's two
/// fixes — separate later speed-1 / #10 work.)
#[test]
fn encoder_gate_speed1_rect_and_4way_25() {
    // (generator, cq) — the #25 witness cells; per-fix attribution in the doc.
    #[allow(clippy::type_complexity)]
    let cells: &[(&str, fn(usize, usize) -> u8, i32)] = &[
        ("hstripes6+vstripes16+grad", lf_hstripes6_vstripes16_grad, 8), // needs F1 AND F2
        ("diag+vbars16+ripple", lf_diag_vbars16_ripple, 32),            // needs F2
        (
            "hstripes6+vstripes16+grad",
            lf_hstripes6_vstripes16_grad,
            56,
        ), // needs F2
    ];
    let mut matched = 0usize;
    for &(name, g, cq) in cells {
        let ok = attempt_case_content_uv(256, 256, true, 1, 1, 2, cq, 1, 1, g, |_r, _c| 128);
        eprintln!(
            "speed1 #25 [{name}] cq{cq}: {}",
            if ok { "MATCH" } else { "DIFF" }
        );
        if ok {
            matched += 1;
        }
    }
    assert_eq!(
        matched,
        cells.len(),
        "every speed-1 (cpu-used=1) ALLINTRA #25 witness cell must byte-match real aomenc \
         end-to-end -- these cells re-diverge if either the rect tx-size init-depth fix \
         (tx_search + speed_features) or the 4-partition ml_4_partition_search_level_index fix \
         (part4_prune + partition_pick) is reverted"
    );
}

/// ISOLATION (Gate 2, cpu-used=1) of the `vgrad 256x256 cq32` byte-5
/// divergence — COMPLETE. Conclusion: the divergence is NOT an unported
/// learned-model prune. The prime suspect, `intra_cnn_based_part_prune_level`
/// 0->2 (the CNN split-vs-nonsplit partition prune), is now fully ported +
/// wired into `rd_pick_partition_real`, and its four flags are bit-exact vs C
/// (`cnn_partition_decision_diff`). Because port and C compute IDENTICAL CNN
/// flags, the CNN constrains both searches the same way and CANNOT be the
/// source of a port-vs-C divergence — confirmed empirically: wiring the CNN in
/// left this cell's byte-5 value (157 vs 8) UNCHANGED. The other candidate,
/// `prune_2d_txfm_mode` PRUNE_1->PRUNE_2, does not affect intra at speed 1
/// (see the NOTE on the excluded cell in `encoder_gate_speed1_textured_allintra`
/// and KB-3 in CLAUDE.md for the full elimination). The real cause — since FIXED
/// — was a missing speed-1 `use_square_partition_only_threshold` rect kill: C
/// disables rectangular partitions on the 64x64 SB sub-480p at speed>=1 (thresh
/// drops to BLOCK_32X32) while the port evaluated HORZ and wrongly won on it.
/// Root-caused via isolated sibling-libaom RD instrumentation; see KB-3.
///
/// This test is retained as a DIAGNOSTIC of the CNN's decisions on SB(0,0):
/// it runs the REAL libaom CNN + DNN inference (via
/// `ref_intra_cnn_partition_decision`, an `rd_shim.c` oracle that reproduces
/// `av1/encoder/partition_strategy.c intra_mode_cnn_partition` verbatim over the
/// real exported inference + real weights) on an IDEALIZED vgrad-256 window and
/// asserts the per-bsize square-split-disable pattern. (Note: this uses the
/// synthetic column-gradient window, not the exact source-buffer pixels the
/// encode extracts, so the 64x64-root logit here is neutral (-3.4) whereas the
/// real encode window yields a mild disable; the sub-block pattern is stable and
/// is what the assertions pin.)
///
/// The decision is checked across a qindex bracket (cq32 -> base_qindex 128;
/// cq48 -> 192 per STATUS.md) because `log_q` is only 1 of the 37 DNN features
/// -- a decision stable across the bracket does not depend on the exact qindex.
#[test]
fn isolate_vgrad256_cq32_cnn_partition_prune() {
    c::ref_init();
    // The EXACT speed-1 textured `vgrad` content at 256px (see `content_for`
    // in `encoder_gate_speed1_textured_allintra`): value depends only on column.
    let w = 256usize;
    let vgrad = |_r: usize, col: usize| -> u8 { (32 + col * 190 / w) as u8 };

    // SB(0,0)'s 64x64 CNN input = the 65x65 luma window at frame(-1,-1) with
    // replicated top/left borders (lookahead.c av1_copy_and_extend_frame ->
    // extend_plane edge-replicates). window[i][j] = src(max(i-1,0), max(j-1,0)).
    let mut win = vec![0u8; 65 * 65];
    for i in 0..65 {
        for j in 0..65 {
            let fr = (i as i32 - 1).max(0) as usize;
            let fc = (j as i32 - 1).max(0) as usize;
            win[i * 65 + j] = vgrad(fr, fc);
        }
    }

    // lowres tier thresholds (min(256,256) < 480), from partition_cnn_weights.h
    // split_thresh_lowres / no_split_thresh_lowres, indexed by bsize_idx.
    const SPLIT_LOWRES: [f32; 5] = [100.0, 1.890757, 2.658417, 1.450626, 1.833180];
    const NO_SPLIT_LOWRES: [f32; 5] = [-100.0, -4.100921, -4.564202, -5.695176, -1.483546];

    // The REAL qindex for vgrad-256 cq32 is 128 (confirmed: flat-256 cq32 in
    // encoder_gate_speed1_flat_allintra prints qindex=128; base_qindex is
    // content-independent for a single CQ-mode KEY frame). cq48 -> 192.
    let qindex = 128i32;

    // Sweep EVERY block in SB(0,0)'s quad-tree that the CNN prune visits
    // (bsize <= 64X64, >= 8X8). The CNN runs once on the 64x64 window; each
    // sub-block selects a spatial slice of the cached branch outputs via
    // quad_tree_idx (the shim recomputes the CNN from the same window, then
    // slices -- bit-identical to the cached path). quad_tree_idx layout:
    // 64x64 root = 0; 32x32 = 1..=4; 16x16 = 5..=20; 8x8 = 21..=84.
    // (bsize_idx, label, quad_tree_idx range)
    let blocks: &[(i32, &str, std::ops::RangeInclusive<i32>)] = &[
        (1, "64x64", 0..=0),
        (2, "32x32", 1..=4),
        (3, "16x16", 5..=20),
        (4, "8x8", 21..=84),
    ];
    // Per-bsize: how many sub-blocks disable square-split (flags[3]==1).
    let mut sqsplit_disabled = [0usize; 5];
    let mut counts = [0usize; 5];
    for (bsize_idx, label, qt_range) in blocks {
        let bi = *bsize_idx as usize;
        let mut logit_min = f32::INFINITY;
        let mut logit_max = f32::NEG_INFINITY;
        for qt in qt_range.clone() {
            let (logits, flags) = c::ref_intra_cnn_partition_decision(
                &win, qindex, 8, w as i32, w as i32, *bsize_idx, qt, 2, false,
            );
            logit_min = logit_min.min(logits[0]);
            logit_max = logit_max.max(logits[0]);
            counts[bi] += 1;
            if flags[3] == 1 {
                sqsplit_disabled[bi] += 1;
            }
        }
        eprintln!(
            "vgrad256 SB(0,0) {label} (bsize_idx={bsize_idx}, split>{} no_split<{}): \
             logits[0] in [{logit_min:.4}, {logit_max:.4}], {}/{} disable square-split",
            SPLIT_LOWRES[bi], NO_SPLIT_LOWRES[bi], sqsplit_disabled[bi], counts[bi]
        );
    }

    // DIAGNOSTIC (real libaom CNN+DNN inference, synthetic vgrad window): on
    // this idealized gradient the 64x64 root is CNN-neutral and the CNN DISABLES
    // PARTITION_SPLIT for every 32x32/16x16/8x8 sub-block. This is now a fact
    // the PORT reproduces bit-exactly (the CNN prune is wired into
    // rd_pick_partition_real and flag-matched vs C in cnn_partition_decision
    // _diff) — so it does NOT explain the port-vs-C divergence (both apply the
    // SAME CNN constraints). See KB-3: the residual is a partition RD near-tie
    // (the port picks PARTITION_HORZ for SB(0,0), C picks otherwise). Decisions
    // (flags) are asserted -- not the exact logits, which vary by SIMD tier --
    // because the prec-reduced margins here are large and the flags are what
    // constrain the bitstream-affecting partition search.
    assert_eq!(
        sqsplit_disabled[1], 0,
        "the 64x64 root must be CNN-neutral at qindex=128"
    );
    assert_eq!(
        sqsplit_disabled[2], 4,
        "all four 32x32 sub-blocks must be CNN square-split-disabled"
    );
    assert_eq!(
        sqsplit_disabled[3], 16,
        "all sixteen 16x16 sub-blocks must be CNN square-split-disabled"
    );
    assert_eq!(
        sqsplit_disabled[4], 64,
        "all sixty-four 8x8 sub-blocks must be CNN square-split-disabled"
    );
}

/// Gate 2 (`aomenc --cpu-used=2`) — the all-intra KEY **speed-2** path. Reuses the
/// full e2e derivation with `ref_encode_av1_kf(cpu_used=2)` +
/// `SpeedFeatures::set_allintra(2)`. Byte-identical to real aomenc `--cpu-used=2`
/// end to end across a size × quality × content × subsampling grid: 64/128 (1×1 /
/// 2×2 SB) × cq{12,32,48,63} (qindex ~48/128/184/232, spanning the aggressive-web
/// to high-q range) × {flat, two-tone, vgrad, diag} × {mono, 4:2:0}.
///
/// The speed-2 deltas over speed 1 that are LIVE on the bd8 4:2:0 all-intra KEY
/// path (`set_allintra_speed_features_*`, speed_features.c speed>=2 blocks):
///   - `disable_smooth_intra = 1` (:429) — prunes SMOOTH_H_PRED / SMOOTH_V_PRED
///     from the luma intra-mode search (SMOOTH_PRED survives, since
///     `prune_filter_intra_level != 0`);
///   - `prune_filter_intra_level = 1` (:431) — restricts the filter-intra search
///     to the FILTER variants derived from the best-so-far Y mode;
///   - `perform_coeff_opt = 3` (:433) — coeff-opt dist threshold 1728 -> 864;
///   - `ml_4_partition_search_level_index = 2` (:237) — the 4-way ML-prune row (a
///     byte no-op: `av1_partition4_{search,not_search}_thresh` is byte-identical
///     for levels 1 and 2);
///   - `intra_pruning_with_hog = 2` (:430) — a byte no-op (the HOG threshold table
///     `{-1.2,-1.2,-0.6,0.4}` maps level 1 and 2 to the same -1.2);
///   - `top_intra_model_count_allowed = 3` (:404) — actually a speed>=1 delta the
///     port had left hardcoded at 4. It drives the `prune_intra_y_mode` model-RD
///     slot (index = count-1). Inert at speed 0/1 for every existing gate (the
///     mode set kept fewer than 4 competitive models), but the speed-2
///     `disable_smooth_intra` prune shrinks the mode set enough that the 4-vs-3
///     slot difference tips the winner. **Witness:** `diag 128x128 cq48`
///     byte-matches with `top_intra_model_count_allowed = 3` and re-diverges with
///     4 (proven by env-gated bisect during development), while it byte-matched
///     at speed 0/1 both ways — i.e. this cell diverged from aomenc `--cpu-used=2`
///     before the fix and byte-matches after.
///
/// All other speed-2 deltas are inert here: inter/motion (`auto_mv_step_size`,
/// `simple_motion_search_prune_agg`), CDEF / loop-restoration search (off in the
/// allintra default envelope), the INTER-only `partition_search_breakout_*`
/// thresholds, and the `!boosted`-gated qindex `ext_partition_eval_thresh` (a KEY
/// frame is always boosted). Every cell asserts real end-to-end byte-identity.
#[test]
fn encoder_gate_speed2_textured_allintra() {
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
            "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            "flat" => Box::new(move |_r, _c| 128u8),
            other => panic!("unknown {other:?}"),
        }
    }
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        2,
                        2,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    eprintln!(
                        "speed2 {name} {w}x{h} {fmt} cq{cq}: {}",
                        if ok { "MATCH" } else { "DIFF" }
                    );
                    results.push((format!("{name} {w}x{h} {fmt} cq{cq}"), ok));
                }
            }
        }
    }
    let failed: Vec<&String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| n)
        .collect();
    assert!(
        failed.is_empty(),
        "{}/{} cpu-used=2 all-intra cells diverged from real aomenc: {:?}",
        failed.len(),
        results.len(),
        failed
    );
}

/// Gate 2 (`aomenc --cpu-used=3`) — the all-intra KEY **speed-3** path. Reuses the
/// full e2e derivation with `ref_encode_av1_kf(cpu_used=3)` +
/// `SpeedFeatures::set_allintra(3)`. Byte-identical to real aomenc `--cpu-used=3`
/// end to end across the same size × quality × content × subsampling grid as the
/// speed-2 gate: 64/128 (1×1 / 2×2 SB) × cq{12,32,48,63} × {flat, two-tone, vgrad,
/// diag} × {mono, 4:2:0} — **ALL 64/64 cells byte-identical** (the former 3-cell
/// KB-7 pin was resolved by porting the level-3 OLD-model 4-way ML prune; see the
/// KB-7 note at the assertion at the bottom).
///
/// Anti-vacuous witness (the chroma HOG delta is load-bearing): BEFORE this fix
/// (i.e. `set_allintra(3)` returning the speed-2 sf values, chroma HOG OFF) **16**
/// of these cells diverged from aomenc `--cpu-used=3` — every 4:2:0 cell at cq12
/// AND cq32 (env-gated bisect during development; all 16 byte-MATCH aomenc
/// `--cpu-used=2`, proving they are speed-3-specific). Turning ON the speed-3
/// `chroma_intra_pruning_with_hog = 2` chroma HOG prune fixes **13** of them —
/// e.g. `flat/diag 64x64 420 cq12`, `flat/vgrad/diag 128x128 420 cq32` byte-match
/// ONLY with the chroma HOG applied. So the gate's assertion on those 13 cells is
/// the witness: it fails if the chroma HOG is reverted.
///
/// The speed-3 deltas over speed 2 that are LIVE on the bd8 4:2:0 all-intra KEY
/// path (`set_allintra_speed_features_*`, speed_features.c speed>=3 blocks):
///   - `intra_pruning_with_hog = 3` (:455) — luma HOG prune threshold 1->2 was a
///     no-op (`thresh[]={-1.2,-1.2,-0.6,0.4}` maps levels 1,2 to -1.2), but level
///     3 -> `thresh[2] = -0.6`, a strictly more aggressive directional-mode prune.
///     LIVE: wired via `PickFrameCfg::intra_hog_thresh`.
///   - `chroma_intra_pruning_with_hog = 2` (:454) — default 0 (chroma HOG prune
///     OFF at speed 0/1/2); speed>=3 turns it ON with intraframe `thresh[1][1] =
///     -1.2`. LIVE on 4:2:0 cells: wired via `UvLoopPolicy::chroma_hog_thresh`.
///   - `less_rectangular_check_level = 2` (:444) — base 1; at level 2 the SPLIT
///     stage's rect-kill runs for every `idx` (not only `idx <= 2`). LIVE: already
///     consumed at `rd_pick_partition`'s post-split arm; threaded via the sf value.
///
/// All other speed-3 deltas are inert on this path (verified against source):
///   - framesize-DEPENDENT (:269-290): `ml_early_term_after_part_split_level = 0`
///     (both consumers `!frame_is_intra_only`, partition_search.c:4322/4335),
///     `max_intra_bsize = BLOCK_32X32` (only `init_mode_skip_mask`'s INTER
///     ref-frame mask, rdopt.c:4217), `partition_search_breakout_{dist,rate}_thr`
///     (INTER), `prune_tx_size_level` (gated on `use_hbd`, false at bd8);
///   - framesize-INDEPENDENT (:439-469): `high_precision_mv_usage`, `search_method`,
///     `full_pixel_search_level`, `simple_motion_search_prune_agg` (all
///     motion/INTER), `recode_loop`/`screen_detection_mode2_fast_detection`
///     (high-level; the harness bootstraps the parsed header + fixed-q allintra
///     path does not recode), the four `lpf_sf` wiener/sgr fields
///     (loop-restoration search OFF in the allintra envelope), `prune_palette_
///     search_level = 2` (`av1_allow_palette` requires `allow_screen_content_tools`
///     -> palette search never runs on these non-screen cells),
///     `use_skip_flag_prediction = 2` (vestigial: indexes `predict_skip_levels`
///     into `winner_mode_params->skip_txfm_level`, whose DEFAULT_EVAL column the
///     port's non-winner-mode intra tx path does not consume), `adaptive_txb_search
///     _level = 2` (already 2 since speed 1), `prune_ext_part_using_split_info = 1`
///     / `ml_4_partition_search_level_index = 3` (4-way HORZ_4/VERT_4 prune only;
///     empirically no-op on this grid — the ML 4-way prune isn't the deciding
///     factor on any diverging cell), `use_rd_based_breakout_for_intra_tx_search`
///     (intra tx-size-search early-exit; empirically no-op on this grid).
///
/// The luma HOG level-3 threshold (-0.6) and `less_rectangular_check_level = 2`
/// are BYTE-NO-OPS on this grid (every mono cell byte-matches, and mono exercises
/// the luma HOG + luma partition rect-kill without any chroma) — they are ported
/// for source faithfulness and verified inert here.
#[test]
fn encoder_gate_speed3_textured_allintra() {
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
            "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            "flat" => Box::new(move |_r, _c| 128u8),
            other => panic!("unknown {other:?}"),
        }
    }
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        3,
                        3,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    eprintln!(
                        "speed3 {name} {w}x{h} {fmt} cq{cq}: {}",
                        if ok { "MATCH" } else { "DIFF" }
                    );
                    results.push((format!("{name} {w}x{h} {fmt} cq{cq}"), ok));
                }
            }
        }
    }

    // KB-7 RESOLVED (was: 3 pinned cq12 4:2:0 residuals — two-tone 64/128,
    // vgrad 128). The flips were NOT a latent chroma-RD arithmetic near-tie —
    // decode-both + the sibling-C RD dump showed every leaf RD (NONE / HORZ /
    // VERT, luma AND chroma parts) matching C to the unit — but a
    // partition-search-SPACE gap: at speed>=3 C's `av1_ml_prune_4_partition`
    // switches to the OLD (`ml_model_index == 0`, LABEL_SIZE=4,
    // UNnormalized-features, int-score max-minus-thresh) NN variant
    // (partition_strategy.c:1359/1472-1497), which on these cells prunes
    // HORZ_4/VERT_4 at every 32x32 node; the port's old level>=3 guard skipped
    // the ML prune entirely, so it searched HORZ_4, found a cheaper 4-way
    // (two-tone 64x64 cq12: child-0 HORZ_4 rdcost 12.9M vs NONE 16.5M) and the
    // SB root flipped NONE->SPLIT. With the old-model branch ported
    // (part4_prune.rs OLD_* weights + int-score decision), ALL 64 cells are
    // byte-identical; this gate now asserts full byte-identity.
    let mut residual: Vec<String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| n.clone())
        .collect();
    residual.sort();
    let expected_residual: [String; 0] = [];
    let matched = results.iter().filter(|(_, ok)| *ok).count();
    eprintln!(
        "encoder_gate_speed3_textured_allintra: {matched}/{} cells byte-identical vs aomenc \
         --cpu-used=3",
        results.len()
    );
    assert_eq!(
        residual, expected_residual,
        "cpu-used=3 divergence set changed: got {residual:?} (expected FULL 64/64 \
         byte-identity) — a speed-3 regression was introduced -> investigate."
    );
}

/// Gate 2 (`aomenc --cpu-used=4`) — the all-intra KEY **speed-4** path. Reuses
/// the full e2e derivation with `ref_encode_av1_kf(cpu_used=4)` +
/// `SpeedFeatures::set_allintra(4)`, over the same grid as the speed-2/3 gates:
/// 64/128 × cq{12,32,48,63} × {flat, two-tone, vgrad, diag} × {mono, 4:2:0} —
/// **ALL 64/64 cells byte-identical** vs real aomenc `--cpu-used=4` (51/64
/// after the two wired deltas below, 59/64 after the KB-8 chunk series,
/// 64/64 after the two KB-7 roots — see the note at the assertion).
///
/// This is a PARTIAL speed-4 port (see `speed_features.rs`'s `if speed >= 4`
/// block). Two speed-4 deltas are LIVE and wired here; the rest are KB-8:
///   1. `prune_chroma_modes_using_luma_winner = 1` (:480) — chroma-mode prune
///      keyed on the luma winner (`av1_derived_chroma_intra_mode_used_flag`,
///      consumer intra_uv_rd.rs:1497).
///   2. `lpf_pick = LPF_PICK_FROM_FULL_IMAGE_NON_DUAL` (:496) — the Y
///      loop-filter search drops the two single-direction refine passes
///      (lf_search::pick_filter_level `non_dual`).
///
/// Anti-vacuous witness (BOTH deltas are load-bearing): starting from
/// `set_allintra(4)` == the speed-3 sf (i.e. neither delta applied), 35/64 cells
/// matched — exactly the cells where aomenc itself emits identical bytes at
/// `--cpu-used=3` and `--cpu-used=4` (env-gated bisect during development). The
/// chroma prune then fixed 10 cells that byte-matched at `--cpu-used=3` but
/// diverged at 4 pre-fix (e.g. `flat 128x128 420 cq32/48/63`, `diag 128x128 420
/// cq32/48` — pure-chroma, their mono siblings never diverged) AND resolved the
/// KB-7 `two-tone 128x128 420 cq12` residual; the NON_DUAL loop-filter fixed 6
/// more (`vgrad 128x128 cq32/48`, `diag 128x128 cq63`, each mono AND 420 — the
/// divergence there was a header-only LF-level byte, tile data byte-identical).
/// Reverting either delta re-diverges its cells, so this gate's byte-match
/// assertion on them is the witness.
///
/// The KB-8 chunk series then ported the FULL remaining speed-4 delta set (all
/// live at speed 4 in this gate's path): the SATD trellis-skip
/// (`skip_trellis_opt_based_on_satd`), the stage-aware MODE_EVAL /
/// WINNER_MODE_EVAL policies (`perform_coeff_opt=5`,
/// `tx_domain_dist_thres_level=3`), the LUMA winner-mode two-pass
/// (`multi_winner_mode_type=MULTI_WINNER_MODE_DEFAULT`: MODE_EVAL loop at
/// USE_LARGESTALL + default-tx-type → top-3 `store_winner_mode_stats` →
/// WINNER_MODE_EVAL re-eval at USE_FULL_RD), the est-rd tx-type prune +
/// txk_map reorder (`prune_tx_type_est_rd`), `use_rd_based_breakout_for_
/// intra_tx_search`, and both split-info prunes (`prune_ext_part_using_
/// split_info` — the AB evaluate at level 2 and the 4-way rect-win prune at
/// level 1). That took the gate 51/64 → 59/64: ALL 32 mono cells byte-match,
/// i.e. the speed-4 LUMA path is byte-exact.
///
/// The former 5-cell 4:2:0 cq12/cq32 pin (KB-7) was then resolved by TWO
/// speed-feature-port fixes (see the note at the assertion): the level-3
/// OLD-model 4-way ML prune (shared with speed 3) and the
/// `chroma_intra_pruning_with_hog`-disable tail (speed_features.c:608-616,
/// live at speed>=4 via `prune_chroma_modes_using_luma_winner`) — taking the
/// gate 59/64 → 64/64 full byte-identity.
#[test]
fn encoder_gate_speed4_textured_allintra() {
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
            "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            "flat" => Box::new(move |_r, _c| 128u8),
            other => panic!("unknown {other:?}"),
        }
    }
    let mut results: Vec<(String, bool)> = Vec::new();
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        4,
                        4,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    eprintln!(
                        "speed4 {name} {w}x{h} {fmt} cq{cq}: {}",
                        if ok { "MATCH" } else { "DIFF" }
                    );
                    results.push((format!("{name} {w}x{h} {fmt} cq{cq}"), ok));
                }
            }
        }
    }

    let mut residual: Vec<String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| n.clone())
        .collect();
    residual.sort();
    // KB-7 RESOLVED (was: 5 pinned 4:2:0 cq12/cq32 residuals). TWO distinct
    // speed-feature-port gaps, no RD-arithmetic bug:
    //   1. (shared with speed 3) `av1_ml_prune_4_partition`'s OLD
    //      (`ml_model_index == 0`) NN variant at level>=3 was unported — the
    //      port searched HORZ_4/VERT_4 where C prunes them (see the speed-3
    //      gate's KB-7 note).
    //   2. (speed-4-specific) the UNCONDITIONAL TAIL of
    //      set_allintra_speed_features_framesize_independent
    //      (speed_features.c:608-616) force-disables
    //      `chroma_intra_pruning_with_hog` whenever
    //      `prune_chroma_modes_using_luma_winner` is on (allintra speed>=4).
    //      The port kept the chroma HOG live at speed 4, HOG-pruning a
    //      directional uv mode C evaluates and picks (two-tone 64x64 cq12:
    //      C picks UV_V_PRED 58469617, the port had HOG-masked V and picked
    //      UV_SMOOTH 58779332) — the leaf then coded different chroma bytes.
    // With the old-model 4-way branch ported + the HOG-disable tail modeled,
    // ALL 64 cells are byte-identical; this gate now asserts full
    // byte-identity.
    let expected_residual: [String; 0] = [];
    let matched = results.iter().filter(|(_, ok)| *ok).count();
    eprintln!(
        "encoder_gate_speed4_textured_allintra: {matched}/{} cells byte-identical vs aomenc \
         --cpu-used=4",
        results.len()
    );
    assert_eq!(
        residual, expected_residual,
        "cpu-used=4 divergence set changed: got {residual:?} (expected FULL 64/64 \
         byte-identity) — a speed-4 regression was introduced -> investigate."
    );
}

/// Gate 2 (`aomenc --cpu-used=5`) — the all-intra KEY **speed-5** path.
/// {64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,4:2:0}
/// vs REAL `aomenc --cpu-used=5`: **64/64 cells byte-identical** (asserted).
///
/// The speed-5 deltas over speed 4 (speed_features.c:508-525 + the
/// qindex-dependent aggr=3 arm; the framesize-dependent setter has no
/// speed-5 block), all sourced from `SpeedFeatures::set_allintra(5)` +
/// `ext_partition_eval_thresh_allintra_key`:
/// - `multi_winner_mode_type = MULTI_WINNER_MODE_FAST` (:524): the luma
///   winner-mode two-pass keeps the top-2 (not top-3) winners.
/// - `ext_partition_eval_thresh` → BLOCK_128X128 on sub-480p KEY frames
///   (:510 + :2952): AB + 4-way partitions are never evaluated on this grid.
/// - `chroma_intra_pruning_with_hog = 3` (:515) — zeroed right back by the
///   :608-615 override (`prune_chroma_modes_using_luma_winner` on since
///   speed 4), so chroma HOG stays OFF exactly as at speed 4.
/// - Everything else carries the speed-4 values (top_intra_model_count 3,
///   prune_filter_intra_level 1, NON_DUAL LF, the whole winner-mode/tx
///   policy set — `speed5_allintra_deltas_match_source` pins the fields).
///
/// Anti-vacuity: `encoder_gate_speed5_vs_speed4_sf_witness` proves the
/// speed-5 feature set is NOT byte-equivalent to speed-4's on this grid
/// (port@speed-4-features vs aomenc --cpu-used=5 diverges on the witness
/// cells; 4 cells full-grid at landing time), and per-delta bisects during
/// landing showed each LIVE delta flips cells on its own: multi-winner FAST
/// flips `two-tone 64² mono/420 cq63`; the ext-partition disable flips
/// `two-tone 64² 420 cq12` + `vgrad 128² 420 cq12` (two former KB-7 cells —
/// resolved at speed 4 by the KB-7 landing's level-3 old-model 4-way NN
/// prune; at speed 5 the AB/4-way disable removes those candidates from the
/// search space before any NN runs).
#[test]
fn encoder_gate_speed5_textured_allintra() {
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
            "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            "flat" => Box::new(move |_r, _c| 128u8),
            other => panic!("unknown {other:?}"),
        }
    }
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        5,
                        5,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    eprintln!(
                        "speed5 {name} {w}x{h} {fmt} cq{cq}: {}",
                        if ok { "MATCH" } else { "DIFF" }
                    );
                    if !ok {
                        failures.push(format!("{name} {w}x{h} {fmt} cq{cq}"));
                    }
                    total += 1;
                }
            }
        }
    }
    eprintln!(
        "encoder_gate_speed5_textured_allintra: {}/{total} cells byte-identical vs aomenc \
         --cpu-used=5",
        total - failures.len()
    );
    assert!(
        failures.is_empty(),
        "every cpu-used=5 all-intra cell must byte-match real aomenc; diverging: {failures:?}"
    );
}

/// Anti-vacuous witness for [`encoder_gate_speed5_textured_allintra`]: the
/// port runs with the **speed-4** feature set against REAL `aomenc
/// --cpu-used=5` reference bytes and must DIVERGE — proving the speed-5
/// deltas (`multi_winner_mode_type` FAST + the `ext_partition_eval_thresh`
/// AB/4-way disable) are load-bearing, i.e. the 64/64 speed-5 gate is not
/// vacuously matching a speed-4-equivalent search. The mono cell makes the
/// luma-side (winner-mode) delta's effect visible with no chroma involved;
/// the paired speed-5 run on the same cells re-asserts the true match so
/// this witness can never "pass" via a harness that ignores `speed`.
#[test]
fn encoder_gate_speed5_vs_speed4_sf_witness() {
    let two_tone_64 = |_r: usize, c: usize| if c < 32 { 72u8 } else { 168u8 };
    let uv = |r: usize, c: usize| (60 + (r * 7 + c * 3) % 80) as u8;
    // (mono, cq): the two cq63 cells the multi-winner FAST delta flips.
    for &(mono, cq) in &[(true, 63i32), (false, 63)] {
        let fmt = if mono { "mono" } else { "420" };
        let cross = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 5, 4, two_tone_64, uv);
        assert!(
            !cross,
            "two-tone 64x64 {fmt} cq{cq}: port with SPEED-4 features unexpectedly byte-matched \
             aomenc --cpu-used=5 — the speed-5 witness cell has gone sf-equivalent; pick a new \
             witness cell (or a speed-4 delta leaked into the speed-5 derivation)"
        );
        let true_match = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 5, 5, two_tone_64, uv);
        assert!(
            true_match,
            "two-tone 64x64 {fmt} cq{cq}: the speed-5 witness cell must byte-match with the \
             speed-5 features (it does in the full gate)"
        );
    }
}

/// Gate 2 (`aomenc --cpu-used=6`) — the all-intra KEY **speed-6** path.
/// {64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,4:2:0}
/// vs REAL `aomenc --cpu-used=6`: **64/64 cells byte-identical** (asserted).
///
/// Speed 6 is NEW MACHINERY on top of speed 5 (speed_features.c:527-564 +
/// framesize-dep :304-316 + qindex-dep aggr=4; the KB-10 chunk series):
/// - `lpf_pick = LPF_PICK_FROM_Q` (:559): closed-form LF levels
///   (`pick_filter_level_from_q`) replace the reconstruction search.
/// - Partition: `default_max_partition_size = BLOCK_32X32` (:546, square-
///   split-only at the 64² root), `use_square_partition_only_threshold =
///   BLOCK_16X16` (:315), `ext_partition_eval_thresh = BLOCK_128X128` for
///   ALL sizes (aggr=4), `prune_rectangular_split_based_on_qidx = 2`
///   (:537), `prune_rect_part_using_{4x4_var_deviation,none_pred_mode}`
///   (:539-540), `prune_sub_8x8_partition_level = 1` (:541).
/// - Intra: `top_intra_model_count_allowed = 2` (:533) +
///   `adapt_top_model_rd_count_using_neighbors` (:534),
///   `prune_luma_odd_delta_angles_in_intra` (:535, evens-first sweep + RD
///   prune), `intra_pruning_with_hog = 4` (:531, thresh 0.4),
///   `prune_filter_intra_level = 2` (:529, no filter-intra search),
///   `prune_smooth_intra_mode_for_chroma` (:528), `cfl_search_range = 1`
///   (:532).
/// - Tx: `prune_intra_tx_depths_using_nn` (:553, the 8x8 NN depth prune),
///   `dc_blk_pred_level = 1` (:563, skip-block prediction in DEFAULT/MODE
///   eval), `perform_coeff_opt = 6` + `tx_domain_dist_level = 3` (:555-556,
///   the winner pass moves to tx-domain distortion),
///   `winner_mode_tx_type_pruning = 3` + `prune_tx_type_est_rd = 0`
///   (:551-552, the est-rd prune turns OFF again).
/// - Winner mode: `multi_winner_mode_type = OFF` (:561, single best_mbmi
///   re-eval), `prune_winner_mode_eval_level = 1` (:562, source-variance
///   bypass).
#[test]
fn encoder_gate_speed6_textured_allintra() {
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
            "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            "flat" => Box::new(move |_r, _c| 128u8),
            other => panic!("unknown {other:?}"),
        }
    }
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        6,
                        6,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    eprintln!(
                        "speed6 {name} {w}x{h} {fmt} cq{cq}: {}",
                        if ok { "MATCH" } else { "DIFF" }
                    );
                    if !ok {
                        failures.push(format!("{name} {w}x{h} {fmt} cq{cq}"));
                    }
                    total += 1;
                }
            }
        }
    }
    eprintln!(
        "encoder_gate_speed6_textured_allintra: {}/{total} cells byte-identical vs aomenc \
         --cpu-used=6",
        total - failures.len()
    );
    assert!(
        failures.is_empty(),
        "every cpu-used=6 all-intra cell must byte-match real aomenc; diverging: {failures:?}"
    );
}

/// Anti-vacuous witness for [`encoder_gate_speed6_textured_allintra`]: the
/// port runs with the **speed-5** feature set against REAL `aomenc
/// --cpu-used=6` reference bytes and must DIVERGE — proving the speed-6
/// deltas are load-bearing (the gate is not vacuously matching a
/// speed-5-equivalent search). The paired speed-6 run on the same cells
/// re-asserts the true match so this witness can never "pass" via a harness
/// that ignores `speed`.
#[test]
fn encoder_gate_speed6_vs_speed5_sf_witness() {
    let vgrad_64 = |_r: usize, c: usize| (32 + c * 190 / 64) as u8;
    let uv = |r: usize, c: usize| (60 + (r * 7 + c * 3) % 80) as u8;
    for &(mono, cq) in &[(true, 32i32), (false, 32)] {
        let fmt = if mono { "mono" } else { "420" };
        let cross = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 6, 5, vgrad_64, uv);
        assert!(
            !cross,
            "vgrad 64x64 {fmt} cq{cq}: port with SPEED-5 features unexpectedly byte-matched \
             aomenc --cpu-used=6 — the speed-6 witness cell has gone sf-equivalent; pick a new \
             witness cell (or a speed-5 delta leaked into the speed-6 derivation)"
        );
        let true_match = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 6, 6, vgrad_64, uv);
        assert!(
            true_match,
            "vgrad 64x64 {fmt} cq{cq}: the speed-6 witness cell must byte-match with the \
             speed-6 features (it does in the full gate)"
        );
    }
}

/// Speed-6 coverage extension beyond the textured grid: content engineered
/// to reach speed-6 machinery the {flat,two-tone,vgrad,diag} cells never
/// exercise (probe-verified):
/// - **pseudo-random noise luma** forces small partitions so the WINNER-pass
///   depth sweep runs on 8x8 blocks — the `prune_intra_tx_depths_using_nn`
///   NN's only live shape (bd8 / TX_8X8 / whole-block; 96 live Split
///   verdicts measured on these cells, byte-exact at cq32/48 — the model is
///   additionally pinned by `intra_tx_nn_diff`'s 4000-decision REAL-C
///   differential);
/// - **flat chroma** (constant 128: per-pixel variance 0 < 20) arms
///   `prune_smooth_intra_mode_for_chroma` (its uv-loop consumer stays
///   unreached even here — UV_SMOOTH only survives the speed>=4 luma-winner
///   mask when the LUMA winner is SMOOTH-family, which noise content never
///   yields — carried as transcription-faithful) and drives the CHROMA
///   `predict_dc_only_block` skip predictor (plane-1/2 fires measured).
///
/// cq32/cq48 (mono + 4:2:0) must BYTE-MATCH real `aomenc --cpu-used=6` —
/// hard-asserted like the main gate.
///
/// **KB-10 OPEN CHARACTERIZATION — cq63 (both mono + 4:2:0) DIVERGES**, and
/// this test asserts the divergence is still PRESENT (it FAILS the moment
/// the cells start matching → promote to a full byte-match assert).
/// Localized 2026-07-16 (decode-both + search-tree + sweep-rd probes):
/// - the port's SEARCH partition tree matches real exactly (forced SPLIT of
///   the 64² root — `default_max_partition_size = BLOCK_32X32` — plus four
///   32×32 DC NONE leaves);
/// - the divergence is the (mi 8,0) leaf's WINNER-pass uniform tx-size
///   sweep: the port picks TX_16X16 (rd 3048875660) over TX_32X32
///   (rd 3054700138) — a 0.19% margin — where real aomenc keeps TX_32X32;
///   the tx-plan difference desyncs the coded coefficients (the frame codes
///   tx_mode LARGEST post-hoc via encodeframe.c:2797's txb_split_count==0
///   collapse, so the parse derives 32×32 txbs).
/// - NOT closed by single-feature reverts (NN off / predict_dc off / rd
///   tables at speed-5 values / winner-OFF arm at speed-5 shape / intra-loop
///   prunes off / partition prunes off — each still diverges) ⇒ a
///   multi-feature near-tie interaction at qindex 255 (KB-2 class), not one
///   missing delta. The same content byte-matches at speed 5 and at speed-6
///   cq32/cq48.
/// - Next step: sibling-C RD dump (KB-2/KB-3/KB-7 method) of the
///   winner-pass 16-vs-32 sweep at (8,0) — per-depth rd, the MODE_EVAL
///   best_rd, and `intra_block_yrd`'s unfactored-vs-factored comparison.
#[test]
fn encoder_gate_speed6_noise_flatuv_allintra() {
    let noise = |r: usize, c: usize| {
        let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        (64 + (x % 129)) as u8
    };
    let flat_uv = |_r: usize, _c: usize| 128u8;
    let mut failures: Vec<String> = Vec::new();
    let mut open_matches: Vec<String> = Vec::new();
    for &cq in &[32i32, 48, 63] {
        for &mono in &[true, false] {
            let ok = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 6, 6, noise, flat_uv);
            let fmt = if mono { "mono" } else { "420" };
            eprintln!(
                "speed6-noise-flatuv 64x64 {fmt} cq{cq}: {}",
                if ok { "MATCH" } else { "DIFF" }
            );
            if cq == 63 {
                // The pinned-open KB-10 near-tie cells (docs above).
                if ok {
                    open_matches.push(format!("64x64 {fmt} cq{cq}"));
                }
            } else if !ok {
                failures.push(format!("64x64 {fmt} cq{cq}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "every asserted speed-6 noise/flat-uv cell must byte-match real aomenc; \
         diverging: {failures:?}"
    );
    assert!(
        open_matches.is_empty(),
        "the pinned-open KB-10 cq63 near-tie cells started BYTE-MATCHING real aomenc \
         ({open_matches:?}) — promote this characterization to a full byte-match assert \
         and close the KB-10 open item"
    );
}

/// Gate 2 (`aomenc --cpu-used=7`) — the all-intra KEY **speed-7** path.
/// {64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,4:2:0}
/// vs REAL `aomenc --cpu-used=7`: **64/64 cells byte-identical** (asserted).
///
/// Speed 7 is STRUCTURALLY NEW (speed_features.c:569-575, KB-11) — not a
/// prune re-parameterization: `partition_search_type = VAR_BASED_PARTITION`
/// replaces the whole RD partition search with a variance-threshold FIXED
/// tree + an RD mode search over it (fast-allintra = the avifenc-style
/// fast-stills path, so this is stills-parity scope):
/// - `av1_choose_var_based_partitioning` (var_part.rs): 4x4-downsampled
///   source variance tree, `120 * ac_q(qindex)`-based thresholds
///   (`set_vbp_thresholds_key_frame`), KEY rules (64x64+ always split;
///   16/32 split on threshold exceed; NONE under; edge-fit rect pairs) —
///   stamped as mi-grid bsizes.
/// - `av1_rd_use_partition` (partition_pick.rs `rd_use_partition_real`):
///   walks the fixed tree with the NORMAL full-RD leaf mode search
///   (`use_nonrd_pick_mode` stays 0 until speed 8) — HORZ/VERT mid-strip
///   dry-encode propagation, SPLIT recursion with `do_recon = i != 3`,
///   save/restore + the root OUTPUT_ENABLED winner walk, exactly the pick
///   path's context shape. The allintra SB rdmult modifier is IDENTITY on
///   this path (only av1_rd_pick_partition's root recomputes it —
///   pack_tile skips the fold at speed >= 7).
/// - `default_min_partition_size = BLOCK_8X8` (assertion-only: the KEY tree
///   never stamps below 8x8); `cdef_pick_method = CDEF_PICK_FROM_Q` (CDEF
///   off in allintra — inert); `mode_search_skip_flags |=
///   FLAG_SKIP_INTRA_DIRMISMATCH` (consumer is the INTER-frame intra search
///   only — inert); `var_part_split_threshold_shift = 7` (dead while
///   `force_large_partition_blocks_intra == 0` — inert on KEY).
/// - Everything else (the whole speed-6 feature set: intra loop, tx policy,
///   winner-mode OFF restructure, LPF_PICK_FROM_Q) carries over unchanged.
#[test]
fn encoder_gate_speed7_textured_allintra() {
    fn content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
        match name {
            "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
            "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
            "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
            "flat" => Box::new(move |_r, _c| 128u8),
            other => panic!("unknown {other:?}"),
        }
    }
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        7,
                        7,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    eprintln!(
                        "speed7 {name} {w}x{h} {fmt} cq{cq}: {}",
                        if ok { "MATCH" } else { "DIFF" }
                    );
                    if !ok {
                        failures.push(format!("{name} {w}x{h} {fmt} cq{cq}"));
                    }
                    total += 1;
                }
            }
        }
    }
    eprintln!(
        "encoder_gate_speed7_textured_allintra: {}/{total} cells byte-identical vs aomenc \
         --cpu-used=7",
        total - failures.len()
    );
    assert!(
        failures.is_empty(),
        "every cpu-used=7 all-intra cell must byte-match real aomenc; diverging: {failures:?}"
    );
}

/// Anti-vacuous witness for [`encoder_gate_speed7_textured_allintra`]: the
/// port runs with the **speed-6** feature set (the full RD partition
/// search) against REAL `aomenc --cpu-used=7` reference bytes and must
/// DIVERGE — proving the speed-7 structural flip (VAR_BASED_PARTITION +
/// rd_use_partition) is load-bearing, i.e. the 64/64 speed-7 gate is not
/// vacuously matching a speed-6-equivalent search. vgrad content splits
/// differently under the variance tree than under the RD search on these
/// cells; the mono cell isolates the luma side. The paired speed-7 run on
/// the same cells re-asserts the true match so this witness can never
/// "pass" via a harness that ignores `speed`.
#[test]
fn encoder_gate_speed7_vs_speed6_sf_witness() {
    let vgrad_64 = |_r: usize, c: usize| (32 + c * 190 / 64) as u8;
    let uv = |r: usize, c: usize| (60 + (r * 7 + c * 3) % 80) as u8;
    for &(mono, cq) in &[(true, 32i32), (false, 32)] {
        let fmt = if mono { "mono" } else { "420" };
        let cross = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 7, 6, vgrad_64, uv);
        assert!(
            !cross,
            "vgrad 64x64 {fmt} cq{cq}: port with SPEED-6 features unexpectedly byte-matched \
             aomenc --cpu-used=7 — the speed-7 witness cell has gone sf-equivalent; pick a new \
             witness cell (or a speed-6 delta leaked into the speed-7 derivation)"
        );
        let true_match = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 7, 7, vgrad_64, uv);
        assert!(
            true_match,
            "vgrad 64x64 {fmt} cq{cq}: the speed-7 witness cell must byte-match with the \
             speed-7 features (it does in the full gate)"
        );
    }
}

/// Speed-7 coverage extension beyond the textured grid (the speed-6
/// extension's exact content): pseudo-random noise luma + flat chroma. On
/// this content the variance tree goes DEEP — 16x16s exceed thresholds[3]
/// and force-split to 8x8 leaves (cq32's tile is ~1560 bytes vs the canon
/// cells' tens) — so this exercises `rd_use_partition_real`'s full
/// recursion: 8x8 NONE leaves, the `do_recon = i != 3` split-child
/// propagation at every level, and the deep save/restore + root
/// OUTPUT_ENABLED walk. cq12/cq32/cq48 (mono + 4:2:0) must BYTE-MATCH real
/// `aomenc --cpu-used=7` — hard-asserted like the main gate.
///
/// **KB-11 OPEN CHARACTERIZATION — cq63 (both mono + 4:2:0) DIVERGES**, and
/// this test asserts the divergence is still PRESENT (it FAILS the moment
/// the cells start matching → promote to a full byte-match assert). This is
/// the SAME two cells and the SAME root as KB-10's pinned-open speed-6
/// near-tie — localized by `kb11_speed7_noise_localize.rs` (decode-both +
/// intended-winner dump): the variance tree fixes the same SPLIT + four
/// NONE-32x32 shape real uses (decoded trees IDENTICAL), every mode record
/// matches, and the port's (mi 8,0) leaf carries tx_size TX_16X16 where
/// real keeps TX_32X32 — KB-10's "(mi 8,0) 32x32 WINNER-pass uniform
/// tx-size sweep picks TX_16X16 over TX_32X32 by 0.19%" near-tie verbatim
/// (the leaf mode search is the same machinery at speeds 6 and 7; only the
/// partition source changed, and both produce this same tree here). KB-10's
/// next step (sibling-C RD dump of the winner sweep at (8,0)) closes both
/// speeds' cells at once.
#[test]
fn encoder_gate_speed7_noise_flatuv_allintra() {
    let noise = |r: usize, c: usize| {
        let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        (64 + (x % 129)) as u8
    };
    let flat_uv = |_r: usize, _c: usize| 128u8;
    let mut failures: Vec<String> = Vec::new();
    let mut open_matches: Vec<String> = Vec::new();
    for &cq in &[12i32, 32, 48, 63] {
        for &mono in &[true, false] {
            let ok = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 7, 7, noise, flat_uv);
            let fmt = if mono { "mono" } else { "420" };
            eprintln!(
                "speed7-noise-flatuv 64x64 {fmt} cq{cq}: {}",
                if ok { "MATCH" } else { "DIFF" }
            );
            if cq == 63 {
                // The pinned-open KB-11 cells (== the KB-10 near-tie; docs above).
                if ok {
                    open_matches.push(format!("64x64 {fmt} cq{cq}"));
                }
            } else if !ok {
                failures.push(format!("64x64 {fmt} cq{cq}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "every asserted speed-7 noise/flat-uv cell must byte-match real aomenc; \
         diverging: {failures:?}"
    );
    assert!(
        open_matches.is_empty(),
        "the pinned-open KB-11 cq63 near-tie cells started BYTE-MATCHING real aomenc \
         ({open_matches:?}) — promote this characterization (and \
         kb11_speed7_noise_localize.rs) to full byte-match asserts and close the KB-11 \
         open item (KB-10's speed-6 twin likely closed too)"
    );
}

/// **Speed-6 prep (chunk 1 building block):** `pick_filter_level_from_q`
/// (lf_search.rs) — the `LPF_PICK_FROM_Q` closed-form KEY-frame LF-level
/// derivation (picklpf.c:266-330, `lpf_pick` at allintra speed >= 6 per
/// speed_features.c:559) — validated against REAL `aomenc --cpu-used=6`:
/// the LF levels aomenc codes in its cpu-6 frame header ARE the C formula's
/// output, so parsing them from a real cpu-6 stream is a true-oracle check
/// of the formula, its fixed-point constants, AND the `av1_ac_quant_QTX`
/// input, across the quality range and both mono/4:2:0. Content-independence
/// (a FROM_Q property — no reconstruction search) is asserted by comparing
/// two different contents at the same cq. No speed 0..=5 behavior is touched
/// (the function has no caller below speed 6 yet — the speed-6 flip wires it
/// into the harness LF derivation like the `non_dual` flag was for 4/5).
#[test]
fn speed6_prep_lf_from_q_matches_real_aomenc() {
    c::ref_init();
    for &cq in &[5i32, 12, 32, 48, 63] {
        for &mono in &[true, false] {
            let mut parsed: Vec<(i32, [i32; 2], i32, i32)> = Vec::new();
            for content_id in 0..2 {
                let (w, h) = (64usize, 64usize);
                let content = |r: usize, cc: usize| -> u8 {
                    if content_id == 0 {
                        if cc < 32 { 72 } else { 168 } // two-tone
                    } else {
                        (32 + (r + cc) * 190 / (w + h)) as u8 // diag
                    }
                };
                let mut y = vec![128u16; w * h];
                for r in 0..h {
                    for cc in 0..w {
                        y[r * w + cc] = u16::from(content(r, cc));
                    }
                }
                let (cw, ch) = if mono { (0, 0) } else { (32usize, 32usize) };
                let mut u = vec![128u16; cw * ch];
                let mut v = vec![128u16; cw * ch];
                for r in 0..ch {
                    for cc in 0..cw {
                        let val = (60 + (r * 7 + cc * 3) % 80) as u16;
                        u[r * cw + cc] = val;
                        v[r * cw + cc] = val;
                    }
                }
                let bytes = c::ref_encode_av1_kf(
                    &y, &u, &v, w, h, 8, mono, 1, 1, cq, /*cpu_used=*/ 6, false, false,
                    /*usage=*/ 2, 0, false,
                );
                assert!(!bytes.is_empty(), "cpu6 encode must produce a stream");
                let obus = walk_obus(&bytes);
                let seq_payload = obus
                    .iter()
                    .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
                    .map(|(_, pl)| *pl)
                    .expect("no sequence header");
                let mut seq_rb = ReadBitBuffer::new(seq_payload);
                let seq = read_sequence_header_obu(&mut seq_rb);
                let (_, frame_payload) = obus
                    .iter()
                    .find(|(t, _)| *t == OBU_FRAME_HEADER || *t == OBU_FRAME)
                    .map(|(t, pl)| (*t, *pl))
                    .expect("no frame OBU");
                let st = &seq.seq_header;
                let cc2 = &seq.color_config;
                let mib_size_log2 = if st.sb_size_128 { 5u32 } else { 4u32 };
                let mi_cols = mi_dim(st.max_frame_width);
                let mi_rows = mi_dim(st.max_frame_height);
                let cfg = FrameHeaderObu {
                    prefix: FrameHeaderPrefix {
                        reduced_still_picture_hdr: seq.reduced_still_picture_hdr,
                        decoder_model_info_present_flag: seq.decoder_model_info_present_flag,
                        equal_picture_interval: seq.timing_info.equal_picture_interval,
                        frame_presentation_time_length: seq
                            .decoder_model_info
                            .frame_presentation_time_length
                            as u32,
                        frame_id_numbers_present_flag: st.frame_id_numbers_present_flag,
                        frame_id_length: st.frame_id_length as u32,
                        force_screen_content_tools: st.force_screen_content_tools,
                        force_integer_mv: st.force_integer_mv,
                        max_frame_width: st.max_frame_width,
                        max_frame_height: st.max_frame_height,
                        enable_order_hint: st.enable_order_hint,
                        order_hint_bits_minus_1: st.order_hint_bits_minus_1,
                        operating_points_cnt_minus_1: seq.operating_points_cnt_minus_1,
                        operating_point_idc: seq.operating_point_idc,
                        op_decoder_model_param_present: seq.op_decoder_model_param_present,
                        buffer_removal_time_length: seq
                            .decoder_model_info
                            .buffer_removal_time_length
                            as u32,
                        temporal_layer_id: 0,
                        spatial_layer_id: 0,
                        ..Default::default()
                    },
                    frame_size: FrameSizeHeader {
                        num_bits_width: st.num_bits_width,
                        num_bits_height: st.num_bits_height,
                        superres_upscaled_width: st.max_frame_width,
                        superres_upscaled_height: st.max_frame_height,
                        enable_superres: st.enable_superres,
                        ..Default::default()
                    },
                    tile_info: tile_limits(mi_cols, mi_rows, mib_size_log2),
                    num_planes: if cc2.monochrome { 1 } else { 3 },
                    separate_uv_delta_q: cc2.separate_uv_delta_q,
                    loopfilter: LoopfilterHeader {
                        last_ref_deltas: KF_REF_DELTAS,
                        last_mode_deltas: KF_MODE_DELTAS,
                        ..Default::default()
                    },
                    cdef: CdefHeader {
                        enable_cdef: st.enable_cdef,
                        ..Default::default()
                    },
                    restoration: RestorationHeader {
                        enable_restoration: st.enable_restoration,
                        sb_size_128: st.sb_size_128,
                        subsampling_x: cc2.subsampling_x,
                        subsampling_y: cc2.subsampling_y,
                        ..Default::default()
                    },
                    film_grain_params_present: seq.film_grain_params_present,
                    ..Default::default()
                };
                let mut rb = ReadBitBuffer::new(frame_payload);
                let p = read_uncompressed_header(&mut rb, &cfg);
                parsed.push((
                    p.quant.base_qindex,
                    p.loopfilter.filter_level,
                    p.loopfilter.filter_level_u,
                    p.loopfilter.filter_level_v,
                ));
            }
            // Content-independence: identical qindex ⇒ identical levels.
            assert_eq!(
                parsed[0], parsed[1],
                "FROM_Q must be content-independent (cq{cq})"
            );
            let (qindex, lf_y, lf_u, lf_v) = parsed[0];
            let ours = pick_filter_level_from_q(qindex, 8, true, 0);
            let fmt = if mono { "mono" } else { "420" };
            eprintln!(
                "speed6-prep lf_from_q cq{cq} {fmt}: qindex={qindex} real Y={lf_y:?} U={lf_u} \
                 V={lf_v} ours Y={:?} U={} V={}",
                ours.filter_level, ours.filter_level_u, ours.filter_level_v
            );
            assert_eq!(ours.filter_level, lf_y, "cq{cq} {fmt}: Y levels");
            if !mono {
                assert_eq!(ours.filter_level_u, lf_u, "cq{cq} {fmt}: U level");
                assert_eq!(ours.filter_level_v, lf_v, "cq{cq} {fmt}: V level");
            }
        }
    }
}

// ===========================================================================
// KB-12 — Gate 2, cpu-used=8 (nonrd PICKMODE, allintra KEY).
// ===========================================================================

fn textured_content_for(w: usize, h: usize, name: &str) -> Box<dyn Fn(usize, usize) -> u8> {
    match name {
        "two-tone" => Box::new(move |_r, c| if c < w / 2 { 72 } else { 168 }),
        "vgrad" => Box::new(move |_r, c| (32 + c * 190 / w) as u8),
        "diag" => Box::new(move |r, c| (32 + (r + c) * 190 / (w + h)) as u8),
        "flat" => Box::new(move |_r, _c| 128u8),
        other => panic!("unknown {other:?}"),
    }
}

/// Gate 2 (`aomenc --cpu-used=8`) — the all-intra KEY **speed-8** path: the
/// nonrd PICKMODE (`use_nonrd_pick_mode = 1`, speed_features.c:578). The same
/// `av1_choose_var_based_partitioning` KEY tree the speed-7 gate uses now
/// drives `av1_nonrd_use_partition` (partition_pick.rs `nonrd_use_partition_
/// real`) — a SINGLE-PASS walk: per leaf `hybrid_intra_mode_search`
/// (`hybrid_intra_pickmode = 2` → full-RD `av1_rd_pick_intra_mode_sb` for
/// `bsize < BLOCK_16X16 && source_variance >= 101`, else the estimate arm
/// `av1_nonrd_pick_intra_mode`), then encode immediately. Chroma is Y-only DC
/// on the estimate arm (nonrd_pickmode.c:1735). Same canon grid as speeds
/// 4-7 ({64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,420}).
///
/// **60/64 byte-identical vs real `aomenc --cpu-used=8`. FOUR pinned-open
/// near-tie cells** (KB-12, the same near-tie CLASS as KB-10/KB-11):
/// `diag {64² cq12, 128² cq32}` × {mono,420}. Localized (via a decode-both
/// diff, the `kb11_speed7_noise_localize.rs` shape) to a single BLOCK_8X8
/// **estimate-arm** leaf (mi 2,2 on the 64² cells): the partition trees are
/// IDENTICAL and every earlier leaf byte-matches, but `av1_nonrd_pick_intra_
/// mode` picks **V_PRED** where real codes **H_PRED** — a directional-mode
/// near-tie decided by ~0.7 % rdcost (V 624968 vs H 629535; identical dist,
/// bmode_costs V 1596 / H 1385). The whole traced estimate chain (the LP
/// Hadamard/`quantize_lp`/`satd_lp`/`block_error_lp` kernels + the mode loop)
/// matches libaom to the line, so the exact tip needs a sibling-C per-mode
/// RD dump at that leaf (the KB-10/KB-11 next step). Speed 9 (all-estimate,
/// with the speed-9 mode prunes) byte-matches the SAME cells 64/64, i.e. the
/// prunes mask this near-tie; only speed 8's unpruned mixed hybrid exposes it.
/// The mono cells prove it is luma-side. The test FAILS if a pinned cell
/// starts matching (→ promote) or if any non-pinned cell diverges.
#[test]
fn encoder_gate_speed8_textured_allintra() {
    // KB-12 pinned-open near-tie cells (see the fn doc).
    let pinned: &[&str] = &[
        "diag 64x64 mono cq12",
        "diag 64x64 420 cq12",
        "diag 128x128 mono cq32",
        "diag 128x128 420 cq32",
    ];
    let mut failures: Vec<String> = Vec::new();
    let mut pinned_now_matching: Vec<String> = Vec::new();
    let mut total = 0usize;
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = textured_content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        8,
                        8,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    let cell = format!("{name} {w}x{h} {fmt} cq{cq}");
                    eprintln!("speed8 {cell}: {}", if ok { "MATCH" } else { "DIFF" });
                    if pinned.contains(&cell.as_str()) {
                        if ok {
                            pinned_now_matching.push(cell);
                        }
                    } else if !ok {
                        failures.push(cell);
                    }
                    total += 1;
                }
            }
        }
    }
    eprintln!(
        "encoder_gate_speed8_textured_allintra: {}/{total} cells byte-identical vs aomenc \
         --cpu-used=8 ({} pinned-open near-tie cells)",
        total - failures.len() - pinned.len() + pinned_now_matching.len(),
        pinned.len()
    );
    assert!(
        failures.is_empty(),
        "every non-pinned cpu-used=8 all-intra cell must byte-match real aomenc; \
         diverging: {failures:?}"
    );
    assert!(
        pinned_now_matching.is_empty(),
        "KB-12 pinned-open speed-8 near-tie cell(s) started BYTE-MATCHING real aomenc \
         ({pinned_now_matching:?}) — promote them to hard byte-match asserts and close the \
         KB-12 estimate-arm V/H near-tie"
    );
}

/// Gate 2 (`aomenc --cpu-used=9`) — the all-intra KEY **speed-9** path: the
/// nonrd PICKMODE with `hybrid_intra_pickmode = 0` (every leaf uses the
/// estimate arm `av1_nonrd_pick_intra_mode`; the full-RD arm dies), the three
/// estimate-loop prunes (`prune_h_pred_using_best_mode_so_far`,
/// `enable_intra_mode_pruning_using_neighbors`,
/// `prune_intra_mode_using_best_sad_so_far`), and `INTERNAL_COST_UPD_OFF`
/// (<4k) — every SB reads the FRAME-INIT cost tables (byte-visible on the 128²
/// multi-SB cells). The `vbp_prune_16x16_split_using_min_max_sub_blk_var`
/// speed-9 flag is threaded but inert <720p (force_large-gated). Same canon
/// grid, byte-identical to real `aomenc --cpu-used=9`.
#[test]
fn encoder_gate_speed9_textured_allintra() {
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;
    for &(w, h) in &[(64usize, 64usize), (128usize, 128usize)] {
        for &name in &["flat", "two-tone", "vgrad", "diag"] {
            for &cq in &[12i32, 32, 48, 63] {
                for &mono in &[true, false] {
                    let content = textured_content_for(w, h, name);
                    let ok = attempt_case_content_uv(
                        w,
                        h,
                        mono,
                        1,
                        1,
                        2,
                        cq,
                        9,
                        9,
                        |r, c| content(r, c),
                        |r, c| (60 + (r * 7 + c * 3) % 80) as u8,
                    );
                    let fmt = if mono { "mono" } else { "420" };
                    eprintln!(
                        "speed9 {name} {w}x{h} {fmt} cq{cq}: {}",
                        if ok { "MATCH" } else { "DIFF" }
                    );
                    if !ok {
                        failures.push(format!("{name} {w}x{h} {fmt} cq{cq}"));
                    }
                    total += 1;
                }
            }
        }
    }
    eprintln!(
        "encoder_gate_speed9_textured_allintra: {}/{total} cells byte-identical vs aomenc \
         --cpu-used=9",
        total - failures.len()
    );
    assert!(
        failures.is_empty(),
        "every cpu-used=9 all-intra cell must byte-match real aomenc; diverging: {failures:?}"
    );
}

/// Anti-vacuous witness for [`encoder_gate_speed8_textured_allintra`]: the port
/// runs with the **speed-7** feature set (the full-RD `rd_use_partition` walk
/// over the VBP tree) against REAL `aomenc --cpu-used=8` reference bytes and
/// must DIVERGE — proving the speed-8 nonrd-pickmode flip (`nonrd_use_
/// partition` + `av1_nonrd_pick_intra_mode`) is load-bearing, i.e. the gate is
/// not vacuously matching a speed-7-equivalent search. The paired speed-8 run
/// on the same cells re-asserts the true match. The mono cell isolates luma.
#[test]
fn encoder_gate_speed8_vs_speed7_sf_witness() {
    let uv = |r: usize, c: usize| (60 + (r * 7 + c * 3) % 80) as u8;
    let vgrad_64 = |_r: usize, c: usize| (32 + c * 190 / 64) as u8;
    for &mono in &[true, false] {
        let fmt = if mono { "mono" } else { "420" };
        let cross = attempt_case_content_uv(64, 64, mono, 1, 1, 2, 32, 8, 7, vgrad_64, uv);
        assert!(
            !cross,
            "vgrad 64x64 {fmt} cq32: port with SPEED-7 features unexpectedly byte-matched \
             aomenc --cpu-used=8 — the speed-8 witness cell has gone sf-equivalent; pick a new \
             witness cell (or a speed-7 delta leaked into the speed-8 derivation)"
        );
        let true_match = attempt_case_content_uv(64, 64, mono, 1, 1, 2, 32, 8, 8, vgrad_64, uv);
        assert!(
            true_match,
            "vgrad 64x64 {fmt} cq32: the speed-8 witness cell must byte-match with the speed-8 \
             features (it does in the full gate)"
        );
    }
}

/// Anti-vacuous witness for [`encoder_gate_speed9_textured_allintra`]: the port
/// runs with the **speed-8** feature set against REAL `aomenc --cpu-used=9`
/// reference bytes and must DIVERGE — proving the speed-9 deltas
/// (`hybrid_intra_pickmode = 0` all-estimate, the three estimate-loop prunes,
/// and `INTERNAL_COST_UPD_OFF` on multi-SB frames) are load-bearing. Uses a
/// 128² cell so the cost-update-off delta is byte-visible (4 SBs), plus a 64²
/// cell for the prune deltas. The paired speed-9 run re-asserts the true match.
#[test]
fn encoder_gate_speed9_vs_speed8_sf_witness() {
    let uv = |r: usize, c: usize| (60 + (r * 7 + c * 3) % 80) as u8;
    // Diverging cells discovered empirically (port@speed-8 vs aomenc cpu-9);
    // each must match at true speed 9 (it does in the full gate).
    let candidates: &[(usize, &str, i32, bool)] = &[
        (128, "vgrad", 12, true),
        (128, "vgrad", 12, false),
        (128, "diag", 12, true),
        (64, "diag", 12, true),
    ];
    let mut any_diverged = false;
    for &(sz, name, cq, mono) in candidates {
        let content = textured_content_for(sz, sz, name);
        let cross =
            attempt_case_content_uv(sz, sz, mono, 1, 1, 2, cq, 9, 8, |r, c| content(r, c), uv);
        if !cross {
            any_diverged = true;
            // The same cell must byte-match with the true speed-9 features.
            let content = textured_content_for(sz, sz, name);
            let true_match =
                attempt_case_content_uv(sz, sz, mono, 1, 1, 2, cq, 9, 9, |r, c| content(r, c), uv);
            let fmt = if mono { "mono" } else { "420" };
            assert!(
                true_match,
                "{name} {sz}x{sz} {fmt} cq{cq}: diverges at port@speed-8 vs cpu-9 (good) but \
                 must byte-match at true speed 9"
            );
        }
    }
    assert!(
        any_diverged,
        "no speed-9 witness cell diverged with the speed-8 feature set vs aomenc --cpu-used=9 — \
         the speed-9 deltas have gone sf-equivalent on the candidate cells; pick new witnesses"
    );
}

/// Speed-8 coverage extension beyond the textured grid: pseudo-random noise
/// luma + flat chroma (the speed-6/7 noise-extension content). The variance
/// tree goes DEEP (16×16s force-split to 8×8), exercising the nonrd walk's
/// full SPLIT recursion AND the `hybrid_intra_mode_search` full-RD arm on the
/// high-variance 8×8 leaves. cq12/32/48/63 (mono + 4:2:0) byte-match real
/// `aomenc --cpu-used=8` — hard-asserted (the KB-10/KB-11 speed-6/7 cq63 tx
/// near-tie does NOT reproduce at speed 8: the estimate arm codes tx_size =
/// max-square directly, with no winner-pass tx sweep to flip).
#[test]
fn encoder_gate_speed8_noise_flatuv_allintra() {
    let noise = |r: usize, c: usize| {
        let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        (64 + (x % 129)) as u8
    };
    let flat_uv = |_r: usize, _c: usize| 128u8;
    let mut failures: Vec<String> = Vec::new();
    for &cq in &[12i32, 32, 48, 63] {
        for &mono in &[true, false] {
            let ok = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 8, 8, noise, flat_uv);
            let fmt = if mono { "mono" } else { "420" };
            eprintln!(
                "speed8-noise-flatuv 64x64 {fmt} cq{cq}: {}",
                if ok { "MATCH" } else { "DIFF" }
            );
            if !ok {
                failures.push(format!("64x64 {fmt} cq{cq}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "every speed-8 noise/flat-uv cell must byte-match real aomenc; diverging: {failures:?}"
    );
}

/// Speed-9 coverage extension (same noise content as the speed-8 extension).
/// Every leaf is the estimate arm (`hybrid_intra_pickmode = 0`) with the
/// speed-9 mode prunes; cq12/32/48/63 (mono + 4:2:0) byte-match real
/// `aomenc --cpu-used=9` — hard-asserted.
#[test]
fn encoder_gate_speed9_noise_flatuv_allintra() {
    let noise = |r: usize, c: usize| {
        let mut x = (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (c as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        (64 + (x % 129)) as u8
    };
    let flat_uv = |_r: usize, _c: usize| 128u8;
    let mut failures: Vec<String> = Vec::new();
    for &cq in &[12i32, 32, 48, 63] {
        for &mono in &[true, false] {
            let ok = attempt_case_content_uv(64, 64, mono, 1, 1, 2, cq, 9, 9, noise, flat_uv);
            let fmt = if mono { "mono" } else { "420" };
            eprintln!(
                "speed9-noise-flatuv 64x64 {fmt} cq{cq}: {}",
                if ok { "MATCH" } else { "DIFF" }
            );
            if !ok {
                failures.push(format!("64x64 {fmt} cq{cq}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "every speed-9 noise/flat-uv cell must byte-match real aomenc; diverging: {failures:?}"
    );
}

// ============================================================================
// TASK #39 — REAL-content encode byte-parity at speed >= 1.
//
// The encoder byte-matches aomenc on SYNTHETIC content across all 10 speeds
// (the `encoder_gate_speed{1..9}_textured_allintra` gates), and on REAL content
// only at speed 0 (KB-6's `encoder_gate_real_image_e2e_kb6_repro` = 30/30).
// This gate feeds the SAME real conformance-decoded YUV KB-6 uses through the
// PROVEN speed-N encode path (`attempt_case_content_uv_sep`, the exact function
// every synthetic speed gate rides) at cpu-used 1..4, mapping which real cells
// hold and which diverge once speed features engage on real statistics.
// ============================================================================

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
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF",
        "not an IVF file"
    );
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

/// **TASK #39 — real-content encoder byte-parity at cpu-used 1..4 (MAP).**
///
/// Decodes the same real conformance vectors/crops KB-6 uses (`01-size`,
/// `00-quantizer`, `23-film_grain` — a frame-size test pattern plus real
/// photographic + film-grain statistics) to genuine bd8 4:2:0 YUV via the C
/// decode oracle, then runs this port's full `set_allintra(speed)` search +
/// pack vs real aomenc `--cpu-used=speed` byte-for-byte, at every
/// cpu-used ∈ {1,2,3,4} × cq ∈ {12,32,63}. Prints the MATCH/DIFF map.
///
/// GRADUATE + PIN gate (was report-only on the landing commit; the map = 24/60).
/// Three assertions: (1) an anti-vacuous harness-faithfulness control — the real
/// 64×64 cq20 cell at **speed 0** MUST byte-match (KB-6 proved 30/30 at speed 0
/// through the sibling `run_case`; proves `attempt_case_content_uv_sep`
/// reproduces that speed-0 encode on real separate-U/V content); (2) the 24
/// `byte_exact` cells MUST byte-match (regression guard); (3) every OTHER cell is
/// a PINNED near-tie and MUST diverge (self-promoting, KB-P29/KB-12 pattern). The
/// 36 pinned DIFFs are all interior BLOCK_16X16/8X8 partition RD near-ties
/// (KB-13: the port over-picks AB/SPLIT where C picks simpler) — genuine, not a
/// shared root (the AB prune is C-faithful for allintra). When a fix closes a
/// pinned cell this test FAILS on it → move it into `byte_exact` (never relax).
#[test]
fn encoder_gate_real_content_speed1to4_e2e() {
    let dir = corpus_dir();
    // (vector, crop_w, crop_h, off_x, off_y). crop_w == 0 => FULL frame; else an
    // even-offset crop. Mirrors KB-6's cell family so the speed-0 (KB-6 30/30)
    // and speed-1..4 (here) maps are directly comparable on identical pixels.
    let cells: &[(&str, usize, usize, usize, usize)] = &[
        ("av1-1-b8-01-size-64x64", 0, 0, 0, 0), // 1 SB, aligned — primary clean signal
        ("av1-1-b8-01-size-196x196", 0, 0, 0, 0), // partial-SB, multi-SB
        ("av1-1-b8-00-quantizer-00", 64, 64, 96, 64), // photo, 1-SB aligned crop
        ("av1-1-b8-00-quantizer-00", 128, 128, 64, 64), // photo, 4-SB aligned crop
        ("av1-1-b8-23-film_grain-50", 64, 64, 96, 64), // film grain, 1-SB aligned crop
    ];
    // Cells byte-exact at speed 1..4 (graduated 2026-07-17 from the map run) —
    // hard byte-match asserts (regression guard). Every OTHER (cpu,cq) below is a
    // pinned near-tie (asserted DIVERGE, self-promoting: a DIFF->MATCH flip fails
    // the pin => promote it here; a MATCH->DIFF flip on these is a regression).
    let byte_exact: &[&str] = &[
        // 01-size-64x64 (1-SB): 12/12 byte-exact at speed 1..4.
        "av1-1-b8-01-size-64x64 420 cpu1 cq12",
        "av1-1-b8-01-size-64x64 420 cpu1 cq32",
        "av1-1-b8-01-size-64x64 420 cpu1 cq63",
        "av1-1-b8-01-size-64x64 420 cpu2 cq12",
        "av1-1-b8-01-size-64x64 420 cpu2 cq32",
        "av1-1-b8-01-size-64x64 420 cpu2 cq63",
        "av1-1-b8-01-size-64x64 420 cpu3 cq12",
        "av1-1-b8-01-size-64x64 420 cpu3 cq32",
        "av1-1-b8-01-size-64x64 420 cpu3 cq63",
        "av1-1-b8-01-size-64x64 420 cpu4 cq12",
        "av1-1-b8-01-size-64x64 420 cpu4 cq32",
        "av1-1-b8-01-size-64x64 420 cpu4 cq63",
        // quantizer-00 64x64@96,64 (photo, 1-SB): 11 byte-exact (4 promoted
        // 2026-07-19 by the AB mode-cache root, KB-13/task #39).
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu1 cq12",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu1 cq32",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu1 cq63",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu2 cq12",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu2 cq32",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu2 cq63",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu3 cq12",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu3 cq32",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu3 cq63",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu4 cq12",
        "av1-1-b8-00-quantizer-00 420 64x64@96,64 cpu4 cq63",
        // quantizer-00 128x128@64,64 (photo, 4-SB): 8 byte-exact (6 promoted
        // 2026-07-19, AB mode-cache).
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu1 cq12",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu1 cq32",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu1 cq63",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu2 cq12",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu2 cq32",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu2 cq63",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu3 cq12",
        "av1-1-b8-00-quantizer-00 420 128x128@64,64 cpu3 cq32",
        // 23-film_grain-50 64x64@96,64 (film, 1-SB): 10 byte-exact (7 promoted
        // 2026-07-19, AB mode-cache).
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu1 cq12",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu1 cq32",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu1 cq63",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu2 cq12",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu2 cq32",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu2 cq63",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu3 cq12",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu3 cq32",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu4 cq12",
        "av1-1-b8-23-film_grain-50 420 64x64@96,64 cpu4 cq63",
    ];

    let mut results: Vec<(String, bool)> = Vec::new();
    let mut control_ok = false;
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
        assert_eq!(bd, 8, "{name}: this harness (attempt_case_content_uv_sep) is bd8-only");
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
        let y_c = |r: usize, col: usize| y[(r + off_y) * fw + (col + off_x)] as u8;
        let u_c = |r: usize, col: usize| u[(r + coy) * fcw + (col + cox)] as u8;
        let v_c = |r: usize, col: usize| v[(r + coy) * fcw + (col + cox)] as u8;
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
        // Anti-vacuous harness control (speed 0, NOT the speed>=1 subject).
        if name == "av1-1-b8-01-size-64x64" {
            control_ok =
                attempt_case_content_uv_sep(w, h, mono, ss_x, ss_y, 2, 20, 0, 0, &y_c, &u_c, &v_c);
        }
        for &cpu in &[1i32, 2, 3, 4] {
            for &cq in &[12i32, 32, 63] {
                let ok = attempt_case_content_uv_sep(
                    w, h, mono, ss_x, ss_y, 2, cq, cpu, cpu, &y_c, &u_c, &v_c,
                );
                results.push((format!("{tag} cpu{cpu} cq{cq}"), ok));
            }
        }
    }

    eprintln!("\n=== TASK#39 real-content speed 1-4 map (MATCH = byte-exact vs real aomenc) ===");
    for (label, ok) in &results {
        eprintln!("  {label}: {}", if *ok { "MATCH" } else { "DIFF (speed>=1 real divergence)" });
    }
    let matched = results.iter().filter(|(_, ok)| *ok).count();
    let diverged: Vec<&String> = results
        .iter()
        .filter(|(_, ok)| !*ok)
        .map(|(n, _)| n)
        .collect();
    eprintln!(
        "TASK#39: {}/{} real-content speed 1-4 cells byte-exact; {} diverge {:?}",
        matched,
        results.len(),
        diverged.len(),
        diverged
    );

    // Anti-vacuous harness-faithfulness control (KB-6 = 30/30 at speed 0).
    assert!(
        control_ok,
        "harness faithfulness: real av1-1-b8-01-size-64x64 cq20 at SPEED 0 must byte-match \
         (KB-6 proved 30/30 at speed 0 via the sibling run_case) — if this fails, \
         attempt_case_content_uv_sep's speed-0 config diverges from run_case's and the \
         speed>=1 map below cannot be trusted"
    );

    // Graduated byte-match asserts: each cell in `byte_exact` MUST match
    // (regression guard). Every OTHER cell is a PINNED near-tie and MUST diverge
    // (self-promoting, KB-P29/KB-12 pattern): the DIFFs are all interior
    // BLOCK_16X16/8X8 partition RD near-ties (KB-13; the port over-picks AB/SPLIT)
    // — genuine, not a shared root (the AB prune is C-faithful for allintra). When
    // a fix closes one, this test FAILS on the flipped cell -> move it into
    // `byte_exact`. A byte_exact cell that regresses to DIVERGE also FAILS here.
    for (label, ok) in &results {
        let graduated = byte_exact.contains(&label.as_str());
        if graduated {
            assert!(
                *ok,
                "regression: graduated real-content cell `{label}` must byte-match real aomenc"
            );
        } else {
            assert!(
                !*ok,
                "PROMOTE: pinned real-content near-tie `{label}` now byte-matches real aomenc — \
                 move it into `byte_exact` (a root was fixed; see KB-13)"
            );
        }
    }
}
