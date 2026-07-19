//! INTER-ENCODE chunk 2, sub-step 2a: encode-side reference management +
//! low-delay P frame-header VALUE derivation.
//!
//! Two pieces the inter-encode skeleton needs before any P tile can be coded:
//!
//! 1. [`RefFrame`] — the encode-side stored reference (frame 0's reconstructed,
//!    border-extended Y/U/V + order_hint). Shape mirrors the decoder's
//!    `aom_decode::RefFrame` (planes + strides + crop dims + order_hint) so the
//!    inter predictor ([`aom_dsp::inter::build_inter_predictor`], sub-step 2e) reads
//!    it exactly as the decoder does.
//!
//! 2. [`derive_lowdelay_p_frame_header`] — DERIVE (not parse) the
//!    [`FrameHeaderObu`] for the INTER-ENCODE-ROADMAP §3 low-delay P (frame 1 →
//!    frame 0). The KEY path bootstraps its header by re-parsing the reference
//!    stream (`aom-bench`'s `port_encode`); the P path must derive the values
//!    from the sequence header + the §3 config. The recon-DEPENDENT tail
//!    (loop-filter levels, CDEF) is derived once the P recon exists (sub-step
//!    2f) and is passed in here; every other field is derived from the config.
//!
//! Values verified against a real `aomenc --end-usage=q --lag-in-frames=0
//! --limit=2` frame-1 header (bd8 4:2:0 64x64 cq60): `frame_type=INTER(1)`,
//! `order_hint=1`, `error_resilient_mode=0`, `primary_ref_frame=PRIMARY_REF_NONE
//! (7)`, `frame_refs_short_signaling=0`, `ref_map_idx=[0;7]` (all point at
//! frame-0's slot 0), `refresh_frame_flags=0x02` (frame 1 writes its recon to
//! slot 1), `allow_high_precision_mv=0`, `interp_filter=EIGHTTAP_REGULAR(0)`
//! (the §3 config disables dual-filter / switchable), `switchable_motion_mode=0`,
//! `allow_ref_frame_mvs=0`, `reference_mode_select=0`, `skip_mode_allowed=0`,
//! `tx_mode_select=0` (TX_MODE_LARGEST), global-motion all identity. The prior
//! handoff's "interp_filter=SHARP, allow_high_precision_mv=1" note was WRONG for
//! this config (measured `0`/`false`).

use aom_dsp::entropy::header::{
    CdefHeader, FrameHeaderObu, InterRefSignaling, LoopfilterHeader, WarpedMotionParams,
};

/// `PRIMARY_REF_NONE` (av1/common/enums.h): no reference frame supplies the
/// entropy-context / loop-filter-delta primer — the P uses default CDFs.
pub const PRIMARY_REF_NONE: i32 = 7;

/// `EIGHTTAP_REGULAR` interpolation filter (av1/common/filter.h).
pub const EIGHTTAP_REGULAR: i32 = 0;

/// `HIGH_PRECISION_MV_QTHRESH` (av1/encoder/mv_prec.h): a frame codes
/// `allow_high_precision_mv` iff its `base_qindex` is below this (and integer-MV
/// is not forced).
pub const HIGH_PRECISION_MV_QTHRESH: i32 = 128;

/// The encode-side stored reference frame: frame 0's reconstructed, filtered,
/// border-extended planes + its order_hint. Shape mirrors the decoder's
/// `aom_decode::RefFrame` (lib.rs) so [`aom_dsp::inter::build_inter_predictor`] reads
/// it identically. The recon buffers/strides stay SB/mi-aligned; `width*` /
/// `height*` carry the post-superres VISIBLE (crop) dims used for MC edge
/// replication.
#[derive(Clone, Debug)]
pub struct RefFrame {
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
    pub stride: usize,
    pub stride_uv: usize,
    /// Plane VISIBLE (crop) dims for MC edge replication — coded post-superres
    /// VISIBLE dims, NOT SB/mi extent.
    pub width: usize,
    pub height: usize,
    pub width_uv: usize,
    pub height_uv: usize,
    pub order_hint: i32,
}

impl RefFrame {
    /// Build a reference from reconstructed, border-extended planes + strides +
    /// crop dims + order_hint. Callers pass the FILTERED recon (post loop-filter
    /// / CDEF / restoration), i.e. what the decoder stores as the reference.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        y: Vec<u16>,
        u: Vec<u16>,
        v: Vec<u16>,
        stride: usize,
        stride_uv: usize,
        width: usize,
        height: usize,
        width_uv: usize,
        height_uv: usize,
        order_hint: i32,
    ) -> Self {
        RefFrame {
            y,
            u,
            v,
            stride,
            stride_uv,
            width,
            height,
            width_uv,
            height_uv,
            order_hint,
        }
    }
}

/// The per-P inputs [`derive_lowdelay_p_frame_header`] needs beyond the sequence
/// template. All are derived from the §3 config + the encode-side ref
/// bookkeeping — none are parsed from a reference stream.
#[derive(Clone, Debug)]
pub struct LowDelayPHeaderParams {
    /// The coded frame `base_qindex` (`aom_encode::rc::base_qindex_lowdelay_p_from_cq`).
    pub base_qindex: i32,
    /// Display / decode order hint of this P (frame index; 1 for the first P).
    pub order_hint: i32,
    /// `get_refresh_frame_flags` (encode_strategy.c): the ref-buffer slot(s) this
    /// P refreshes. For the 2-frame low-delay clip: `0x02` (slot 1; frame 0 —
    /// the KEY — populated all 8 slots with `refresh_frame_flags=0xff`).
    pub refresh_frame_flags: i32,
    /// `remapped_ref_idx[LAST..ALTREF]` = the map slot each ref points at. For the
    /// 2-frame low-delay clip every ref resolves to frame 0's slot 0 → `[0; 7]`.
    pub ref_map_idx: [i32; 7],
    /// `disable_cdf_update` (`--cdf-update-mode==0`).
    pub disable_cdf_update: bool,
    /// `reduced_tx_set_used` (`--reduced-tx-type-set`).
    pub reduced_tx_set_used: bool,
    /// Frame-level `interp_filter` (av1/common/enums.h `InterpFilter`). This is a
    /// per-frame RD/heuristic decision (`av1_encode_frame` picks the fixed frame
    /// filter — measured `EIGHTTAP_REGULAR(0)` at cq60, `MULTITAP_SHARP(2)` at
    /// cq20 for the same content), NOT a config constant, so it is passed in (its
    /// standalone derivation — the frame interp-filter selection — is a sub-step
    /// 2f follow-up; bootstrapped from the reference until then).
    pub interp_filter: i32,
    /// The recon-DEPENDENT loop-filter header (levels + deltas), derived from the
    /// P recon by the loop-filter-level search (sub-step 2f). Carries
    /// `mode_ref_delta_enabled` + `ref_deltas`/`mode_deltas`.
    pub loopfilter: LoopfilterHeader,
    /// The recon-DEPENDENT CDEF header (off in the §3 envelope).
    pub cdef: CdefHeader,
}

/// Derive the [`FrameHeaderObu`] for the §3 low-delay P (frame 1 → frame 0),
/// given the sequence-header-derived template `seq_cfg` (the SAME template the
/// KEY path builds before `read_uncompressed_header` — carrying prefix seq
/// fields, `frame_size`, `tile_info`, `num_planes`, `separate_uv_delta_q`,
/// `restoration.enable/ss`, `film_grain_params_present`, and the KF loop-filter
/// deltas) plus the per-P [`LowDelayPHeaderParams`].
///
/// The header is DERIVED, not parsed: every field is set from the config /
/// sequence header / ref bookkeeping. The result feeds
/// [`crate::obu_assemble::assemble_obu_frame_single_tile`] +
/// [`aom_dsp::entropy::header::write_frame_header_obu`] to serialize a byte-exact P
/// frame header.
pub fn derive_lowdelay_p_frame_header(
    seq_cfg: &FrameHeaderObu,
    p: &LowDelayPHeaderParams,
) -> FrameHeaderObu {
    let identity = WarpedMotionParams {
        wmtype: 0, // IDENTITY
        wmmat: [0; 6],
    };
    let enable_order_hint = seq_cfg.prefix.enable_order_hint;
    // `frame_might_allow_ref_frame_mvs` = `enable_ref_frame_mvs && enable_order_hint`
    // — the seq template does not carry `enable_ref_frame_mvs` directly, so it is
    // folded into `might_allow_ref_frame_mvs` by the caller (mirrors the decoder's
    // frame.rs seeding). The §3 config keeps ref-frame-mvs off, so it is false.
    let might_allow_ref_frame_mvs = seq_cfg.might_allow_ref_frame_mvs;
    let might_allow_warped_motion = seq_cfg.might_allow_warped_motion;

    let mut out = seq_cfg.clone();

    // --- prefix (frame-type + inter order/ref bookkeeping) ---
    out.prefix.frame_type = 1; // INTER_FRAME
    out.prefix.show_frame = true;
    // A shown non-KEY frame infers `showable_frame = frame_type != KEY_FRAME`
    // (spec 5.9.2); for INTER this is true.
    out.prefix.showable_frame = true;
    out.prefix.error_resilient_mode = false;
    out.prefix.disable_cdf_update = p.disable_cdf_update;
    out.prefix.allow_screen_content_tools = false;
    out.prefix.cur_frame_force_integer_mv = false;
    out.prefix.order_hint = p.order_hint;
    out.prefix.primary_ref_frame = PRIMARY_REF_NONE;
    out.prefix.refresh_frame_flags = p.refresh_frame_flags;
    // `frame_size_override_flag` is derived by the prefix writer as
    // `superres_upscaled_{w,h} != max_frame_{w,h}` — for a frame at max size (no
    // resize) these must be the max dims so the flag codes 0.
    out.prefix.superres_upscaled_width = out.prefix.max_frame_width;
    out.prefix.superres_upscaled_height = out.prefix.max_frame_height;

    // --- top-level inter body ---
    out.allow_screen_content_tools = false;
    out.superres_scaled = false;
    out.allow_intrabc = false;
    out.inter_ref = InterRefSignaling {
        enable_order_hint,
        frame_refs_short_signaling: false,
        ref_map_idx: p.ref_map_idx,
        set_ref_frame_config: false,
        rtc_reference: [0; 7],
        rtc_ref_idx: [0; 7],
        number_spatial_layers: 1,
        frame_id_numbers_present_flag: false,
        frame_id_length: 0,
        current_frame_id: 0,
        ref_frame_id: [0; 8],
        delta_frame_id_length: 0,
    };
    out.cur_frame_force_integer_mv = false;
    // `av1_pick_and_set_high_precision_mv` (mv_prec.c): `use_hp = qindex <
    // HIGH_PRECISION_MV_QTHRESH`, gated off when `cur_frame_force_integer_mv`.
    out.allow_high_precision_mv =
        !out.cur_frame_force_integer_mv && p.base_qindex < HIGH_PRECISION_MV_QTHRESH;
    out.interp_filter = p.interp_filter;
    out.switchable_motion_mode = false;
    out.might_allow_ref_frame_mvs = might_allow_ref_frame_mvs;
    out.allow_ref_frame_mvs = false;

    // --- quant / delta-q ---
    out.quant.base_qindex = p.base_qindex;
    out.quant.y_dc_delta_q = 0;
    out.quant.u_dc_delta_q = 0;
    out.quant.u_ac_delta_q = 0;
    out.quant.v_dc_delta_q = 0;
    out.quant.v_ac_delta_q = 0;
    out.quant.using_qmatrix = false;
    out.delta_q.base_qindex = p.base_qindex;
    out.delta_q.delta_q_present = false;
    out.delta_q.delta_q_res = 1;
    out.delta_q.delta_lf_present = false;
    out.delta_q.delta_lf_res = 1;
    out.delta_q.delta_lf_multi = false;

    // --- segmentation (off) ---
    out.segmentation.enabled = false;

    // --- lossless / recon-dependent loop-filter + CDEF (from sub-step 2f) ---
    out.coded_lossless = false;
    out.all_lossless = false;
    out.loopfilter = p.loopfilter.clone();
    out.cdef = p.cdef.clone();
    // restoration.enable_restoration / sb_size_128 / ss come from seq_cfg;
    // the §3 envelope codes restoration off.

    // --- mode / tx tail ---
    out.tx_mode_select = false; // TX_MODE_LARGEST
    out.reference_mode_select = false; // SINGLE_REFERENCE
    out.skip_mode_allowed = false;
    out.skip_mode_flag = false;
    out.might_allow_warped_motion = might_allow_warped_motion;
    out.allow_warped_motion = false;
    out.reduced_tx_set_used = p.reduced_tx_set_used;
    out.global_motion = [identity; 7];
    out.ref_global_motion = [identity; 7];

    // Resolve the SINGLE-TILE layout (`tiles_log2 == 0` envelope): the seq
    // template carries only the tile LIMITS (from `tile_limits`); the coded tile
    // info the writer reads is the resolved uniform single tile. `read_uncompressed
    // _header` resolves this on the KEY path; the P path derives it. Multi-tile is
    // out of the single-tile envelope (asserted by the OBU assembler).
    out.tile_info.uniform_spacing = true;
    out.tile_info.log2_cols = 0;
    out.tile_info.log2_rows = 0;
    out.tile_info.cols = 1;
    out.tile_info.rows = 1;

    // tile_info LIMITS, num_planes, separate_uv_delta_q, frame_size,
    // film_grain_params_present all inherited from seq_cfg.
    out
}

/// Convenience: the §3 low-delay-P ref bookkeeping for the 2-frame `[KEY, P]`
/// clip. Frame 0 (the KEY, `refresh_frame_flags=0xff`) populated all 8 ref
/// slots, so frame 1 references slot 0 (`ref_map_idx=[0; 7]`) and refreshes the
/// next slot (`refresh_frame_flags=0x02`). Generalizing beyond 2 frames needs
/// the full `get_refresh_frame_flags` / `get_ref_frame_map_idx` ref-buffer
/// manager (deferred; the 2-frame skeleton only ever refreshes slot 1).
pub const TWO_FRAME_P_REFRESH_FLAGS: i32 = 0x02;
pub const TWO_FRAME_P_REF_MAP_IDX: [i32; 7] = [0; 7];
