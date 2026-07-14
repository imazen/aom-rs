//! Full-tile encode→decode roundtrip for the KEY-frame luma decode driver.
//!
//! DEBLOCKING IS DELIBERATELY NOT MODELED HERE: the mirror mini-encoder has
//! no loop filter, so this roundtrip pins the PRE-FILTER reconstruction
//! (exactly what C's tile decode produces before `av1_loop_filter_frame`).
//! The deblock application is validated separately — synthetically against
//! the real C walk in `aom-loopfilter/tests/lf_apply_diff.rs`, and
//! end-to-end on real deblocked streams in `tests/real_bitstream.rs`.
//!
//! A mirror mini-encoder performs the identical tile walk with the write-side
//! counterparts (`write_partition` / `write_mb_modes_kf_fc` /
//! `write_coeffs_txb_full`) and its own reconstruction feedback loop: per txb
//! it predicts from *its* recon-so-far (same `intra_avail` +
//! `predict_intra_high`), computes the residual against a synthetic source,
//! forward-transforms + quantizes (`xform_quant`, `QuantKind::B`,
//! `invert_quant`-derived params), writes the coefficients, and reconstructs
//! through the same `reconstruct_txb`. Because every write-side piece is
//! byte-identical to C libaom, a clean roundtrip (byte-identical
//! reconstruction planes + lockstep CDF state + per-leaf mode-info equality)
//! pins the decode driver to the C decoder.
//!
//! Both sides run the full FRAME_CONTEXT context selection: each keeps its own
//! per-mi mode-info grid (`MiNbrKf`) and selects every symbol's CDF instance
//! from the `KfFrameContext` arrays by neighbour/block state. The sweep
//! asserts the selection is NON-VACUOUS — many distinct kf_y cells / skip
//! contexts / angle-delta instances / uv_mode instances / filter-intra bsizes
//! / ext-tx (square, intra-dir) cells must actually adapt.
//!
//! Encoder and decoder reconstruction planes start from *different* fill values:
//! a conformant walk never reads an unwritten pixel, so any neighbour-
//! availability bug becomes a hard plane mismatch instead of silently agreeing.
//!
//! Sweep: 4 frame sizes (one SB / 2x2 SBs / non-multiple-of-SB 80x96 px with
//! partial superblocks / 3x3 SBs with a fully-interior SB) × 13 configs
//! (monochrome + 4:4:4 + 4:2:0 + 4:2:2, bd 8/10/12, filter intra on/off,
//! intra edge filter on/off, reduced tx set, tx-type gate off, cdef bits
//! 0..3, per-plane dc/ac dequant deltas, and **delta-q** — 4 configs with
//! `delta_q_present` at `delta_q_res` 1/2/4/8, where the mirror decides a
//! per-SB `current_qindex` target, codes the reduced delta at each SB's
//! upper-left coded block, and both sides recompute every block's per-plane
//! dequant rows from the running carry exactly as `parse_decode_block`
//! does; two of them also code **delta-lf** — multi per-plane and single
//! from-base, carried per block with no reconstruction effect) × 6 seeds × 2
//! frame tx modes (`TX_MODE_LARGEST` — the original sweep, no tx-size bits —
//! and `TX_MODE_SELECT`, where the mirror codes a pseudo-random tx-size
//! depth per signalling block through `write_selected_tx_size` on the
//! `get_tx_size_context`-selected CDF and the decoder must reproduce it,
//! driving real multi-txb grids whose later txbs predict from earlier txbs'
//! reconstruction *inside* the block), with pseudo-random partition trees
//! over all 10 partition types, all 13 intra modes, angle deltas,
//! filter-intra, and skip blocks; coverage of each is asserted at the end
//! (including distinct-tx-size, multi-txb-grid, tx_size_cdf cell-diversity,
//! distinct-effective-qindex, and delta-sign/exp-Golomb floors).

use aom_decode::{
    ANGLE_STEP, BLOCK_8X8, BLOCK_64X64, BLOCK_SIZE_HIGH, BLOCK_SIZE_WIDE, DecodedBlockKf,
    KfTileConfig, MAX_TXSIZE_RECT_LOOKUP, MI_SIZE_HIGH, MI_SIZE_WIDE, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, TX_SIZE_HIGH, TX_SIZE_HIGH_UNIT, TX_SIZE_WIDE,
    TX_SIZE_WIDE_UNIT, UV_CFL_PRED, decode_tile_kf, intra_ext_tx_cdf, is_chroma_reference,
    max_block_units, max_block_units_ss, max_uv_txsize, plane_dequants, scale_chroma_bsize,
    uv_tx_type,
};
use aom_encode::{QuantKind, QuantParams, xform_quant};
use aom_entropy::dec::OdEcDec;
use aom_entropy::enc::OdEcEnc;
use aom_entropy::partition::{
    KfBlockState, KfFrameContext, MbModeInfoKf, MiNbrKf, TXFM_CTX_INIT, TxMode, bsize_to_max_depth,
    bsize_to_tx_size_cat, depth_to_tx_size, filter_intra_allowed, get_partition_subsize,
    get_plane_block_size, get_tx_size_context, get_uv_mode, intra_avail, is_cfl_allowed,
    is_directional_mode, partition_cdf_length, partition_plane_context, set_txfm_ctxs,
    tx_size_from_tx_mode, tx_size_to_depth, update_ext_partition_context, use_angle_delta,
    write_mb_modes_kf_fc, write_partition, write_selected_tx_size,
};
use aom_intra::cfl::{CflCtx, cfl_predict_block, cfl_store_tx};
use aom_intra::predict_intra_high;
use aom_txb::{CDF_ARENA_LEN, ext_tx_set_type, get_txb_ctx, write_coeffs_txb_full};

// ---- deterministic rng + CDF fixtures (repo pattern) -----------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo) as u64) as u32
    }
}

/// Random valid ns-symbol CDF (count slot at `out[n]` left 0).
fn mk_ns_cdf(rng: &mut Rng, n: usize, out: &mut [u16]) {
    for v in out.iter_mut() {
        *v = 0;
    }
    let mut vals = [0i32; 16];
    for v in vals.iter_mut().take(n - 1) {
        *v = 1 + (rng.next() % 32766) as i32;
    }
    vals[..n - 1].sort_unstable();
    vals[..n - 1].reverse();
    let mut prev = 32768i32;
    for i in 0..n - 1 {
        let v = vals[i].min(prev - 1).max((n - 1 - i) as i32);
        out[i] = v as u16;
        prev = v;
    }
}

fn mk_comp(rng: &mut Rng) -> [u16; 69] {
    let mut c = [0u16; 69];
    mk_ns_cdf(rng, 2, &mut c[0..3]);
    mk_ns_cdf(rng, 11, &mut c[3..15]);
    mk_ns_cdf(rng, 2, &mut c[15..18]);
    for i in 0..10 {
        let o = 18 + i * 3;
        mk_ns_cdf(rng, 2, &mut c[o..o + 3]);
    }
    for i in 0..2 {
        let o = 48 + i * 5;
        mk_ns_cdf(rng, 4, &mut c[o..o + 5]);
    }
    mk_ns_cdf(rng, 4, &mut c[58..63]);
    mk_ns_cdf(rng, 2, &mut c[63..66]);
    mk_ns_cdf(rng, 2, &mut c[66..69]);
    c
}

/// Random-valid fill for every CDF region of the frame context (the real
/// coefficient arena included — `mk_coeff_arena`).
fn mk_frame_ctx(rng: &mut Rng) -> KfFrameContext {
    let mut f = KfFrameContext::zeroed(CDF_ARENA_LEN);
    for row in f.kf_y.iter_mut() {
        for cell in row.iter_mut() {
            mk_ns_cdf(rng, 13, cell);
        }
    }
    for (cfl, plane) in f.uv_mode.iter_mut().enumerate() {
        // ns = 14 with CfL / 13 without; slice covers ns+1 slots (count last).
        for cell in plane.iter_mut() {
            mk_ns_cdf(rng, 13 + cfl, &mut cell[..14 + cfl]);
        }
    }
    for a in f.angle_delta.iter_mut() {
        mk_ns_cdf(rng, 7, a);
    }
    for s in f.skip.iter_mut() {
        mk_ns_cdf(rng, 2, s);
    }
    for s in f.seg_spatial.iter_mut() {
        mk_ns_cdf(rng, 8, s);
    }
    for (c, slot) in f.partition.iter_mut().enumerate() {
        let bsl = c / 4;
        let ns = if bsl == 0 {
            4
        } else if bsl == 4 {
            8
        } else {
            10
        };
        mk_ns_cdf(rng, ns, slot);
    }
    for b in f.palette_y_mode.iter_mut() {
        for c in b.iter_mut() {
            mk_ns_cdf(rng, 2, c);
        }
    }
    for c in f.palette_uv_mode.iter_mut() {
        mk_ns_cdf(rng, 2, c);
    }
    for c in f.palette_y_size.iter_mut() {
        mk_ns_cdf(rng, 7, c);
    }
    for c in f.palette_uv_size.iter_mut() {
        mk_ns_cdf(rng, 7, c);
    }
    for c in f.filter_intra.iter_mut() {
        mk_ns_cdf(rng, 2, c);
    }
    mk_ns_cdf(rng, 5, &mut f.filter_intra_mode);
    mk_ns_cdf(rng, 8, &mut f.cfl_sign);
    for a in f.cfl_alpha.iter_mut() {
        mk_ns_cdf(rng, 16, a);
    }
    mk_ns_cdf(rng, 4, &mut f.delta_q);
    for m in f.delta_lf_multi.iter_mut() {
        mk_ns_cdf(rng, 4, m);
    }
    mk_ns_cdf(rng, 4, &mut f.delta_lf);
    mk_ns_cdf(rng, 2, &mut f.intrabc);
    mk_ns_cdf(rng, 4, &mut f.ndvc_joints);
    f.ndvc_comp0 = mk_comp(rng);
    f.ndvc_comp1 = mk_comp(rng);
    for (cat, cells) in f.tx_size.iter_mut().enumerate() {
        // Per-category symbol count (matches C default_tx_size_cdf shapes):
        // cat 0 codes max_depth+1 = 2 symbols, cats 1..=3 code 3.
        let ns = if cat == 0 { 2 } else { 3 };
        for c in cells.iter_mut() {
            mk_ns_cdf(rng, ns, &mut c[..ns + 1]);
        }
    }
    for sq in f.ext_tx_1ddct.iter_mut() {
        for c in sq.iter_mut() {
            mk_ns_cdf(rng, 7, c);
        }
    }
    for sq in f.ext_tx_dtt4.iter_mut() {
        for c in sq.iter_mut() {
            mk_ns_cdf(rng, 5, c);
        }
    }
    f.coeff = mk_coeff_arena(rng);
    f
}

/// Coefficient-arena regions `(offset, slot_count, symbols)` — the same layout the
/// aom-txb/aom-encode roundtrip harnesses use.
const COEFF_REGIONS: [(usize, usize, usize); 13] = [
    (0, 5 * 13, 2),
    (195, 4, 5),
    (219, 4, 6),
    (247, 4, 7),
    (279, 4, 8),
    (315, 4, 9),
    (355, 4, 10),
    (399, 4, 11),
    (447, 5 * 2 * 9, 2),
    (717, 5 * 2 * 4, 3),
    (877, 5 * 2 * 42, 4),
    (2977, 5 * 2 * 21, 4),
    (4027, 2 * 3, 2),
];

fn mk_coeff_arena(rng: &mut Rng) -> Vec<u16> {
    let mut a = vec![0u16; CDF_ARENA_LEN];
    for &(off, count, n) in &COEFF_REGIONS {
        for slot in 0..count {
            let base = off + slot * (n + 1);
            let mut acc: u32 = 0;
            for e in a[base..base + n - 1].iter_mut() {
                acc += rng.range(1, (32000 / n as u32).max(2));
                *e = (32768u32.saturating_sub(acc)).max(1) as u16;
            }
            a[base + n - 1] = 0;
            a[base + n] = 0;
        }
    }
    a
}

/// KfFrameContext has no PartialEq (public-API discipline); compare field by
/// field so a mismatch names the desynced symbol.
fn assert_fc_eq(e: &KfFrameContext, d: &KfFrameContext, what: &str) {
    assert_eq!(e.kf_y, d.kf_y, "{what}: kf_y cdf");
    assert_eq!(e.uv_mode, d.uv_mode, "{what}: uv_mode cdf");
    assert_eq!(e.angle_delta, d.angle_delta, "{what}: angle_delta cdf");
    assert_eq!(e.skip, d.skip, "{what}: skip cdf");
    assert_eq!(e.seg_spatial, d.seg_spatial, "{what}: seg_spatial cdf");
    assert_eq!(e.partition, d.partition, "{what}: partition arena");
    assert_eq!(
        e.palette_y_mode, d.palette_y_mode,
        "{what}: palette_y_mode cdf"
    );
    assert_eq!(
        e.palette_uv_mode, d.palette_uv_mode,
        "{what}: palette_uv_mode cdf"
    );
    assert_eq!(
        e.palette_y_size, d.palette_y_size,
        "{what}: palette_y_size cdf"
    );
    assert_eq!(
        e.palette_uv_size, d.palette_uv_size,
        "{what}: palette_uv_size cdf"
    );
    assert_eq!(e.filter_intra, d.filter_intra, "{what}: filter_intra cdf");
    assert_eq!(
        e.filter_intra_mode, d.filter_intra_mode,
        "{what}: filter_intra_mode cdf"
    );
    assert_eq!(e.cfl_sign, d.cfl_sign, "{what}: cfl_sign cdf");
    assert_eq!(e.cfl_alpha, d.cfl_alpha, "{what}: cfl_alpha cdf");
    assert_eq!(e.delta_q, d.delta_q, "{what}: delta_q cdf");
    assert_eq!(
        e.delta_lf_multi, d.delta_lf_multi,
        "{what}: delta_lf_multi cdf"
    );
    assert_eq!(e.delta_lf, d.delta_lf, "{what}: delta_lf cdf");
    assert_eq!(e.intrabc, d.intrabc, "{what}: intrabc cdf");
    assert_eq!(e.ndvc_joints, d.ndvc_joints, "{what}: ndvc_joints cdf");
    assert_eq!(e.ndvc_comp0, d.ndvc_comp0, "{what}: ndvc_comp0 cdf");
    assert_eq!(e.ndvc_comp1, d.ndvc_comp1, "{what}: ndvc_comp1 cdf");
    assert_eq!(e.tx_size, d.tx_size, "{what}: tx_size cdf");
    assert_eq!(e.ext_tx_1ddct, d.ext_tx_1ddct, "{what}: ext_tx_1ddct cdf");
    assert_eq!(e.ext_tx_dtt4, d.ext_tx_dtt4, "{what}: ext_tx_dtt4 cdf");
    assert_eq!(e.coeff, d.coeff, "{what}: coeff arena");
}

/// libaom `invert_quant` (av1/encoder/av1_quantize.c): the (quant, shift) pair
/// inverting dequant step `d` — realistic qcoeff/eob structure for the mirror.
fn invert_quant(d: i32) -> (i16, i16) {
    let l = 31 - (d as u32).leading_zeros() as i32;
    let m = 1 + (1i64 << (16 + l)) / d as i64;
    ((m - (1 << 16)) as i16, (1i32 << (16 - l)) as i16)
}

/// A B-quant parameter row derived from a `[dc, ac]` dequant pair (the
/// mirror's synthetic-but-valid quantizer: `invert_quant` reciprocals plus
/// simple round/zbin). With delta-q the mirror recomputes this per block from
/// its live dequant rows, so the quantize and the decoder's dequant always
/// use the same block-effective steps.
#[derive(Clone, Copy)]
struct BQuant {
    quant: [i16; 2],
    quant_shift: [i16; 2],
    round: [i16; 2],
    zbin: [i16; 2],
}

fn bquant_for(dequant: [i16; 2]) -> BQuant {
    let mut q = BQuant {
        quant: [0; 2],
        quant_shift: [0; 2],
        round: [0; 2],
        zbin: [0; 2],
    };
    for (i, &d) in dequant.iter().enumerate() {
        let (qq, qs) = invert_quant(d as i32);
        q.quant[i] = qq;
        q.quant_shift[i] = qs;
        q.round[i] = d / 8 + 1;
        q.zbin[i] = d / 2 + 1;
    }
    q
}

/// The used tx_types of the two intra ext-tx sets (av1_ext_tx_used):
/// DTT4_IDTX (5): DCT_DCT/ADST_DCT/DCT_ADST/ADST_ADST/IDTX;
/// DTT4_IDTX_1DDCT (7): + V_DCT/H_DCT.
const EXT_USED_DTT4_IDTX: [usize; 5] = [0, 1, 2, 3, 9];
const EXT_USED_DTT4_IDTX_1DDCT: [usize; 7] = [0, 1, 2, 3, 9, 10, 11];

// ---- coverage accounting -----------------------------------------------------------

#[derive(Default)]
struct Coverage {
    y_modes: [usize; 13],
    partitions: [usize; 10],
    fi_used: usize,
    angle_nonzero: usize,
    skip_blocks: usize,
    eob_zero: usize,
    eob_pos: usize,
    cfl_uv_blocks: usize,
    ext5_signaled: usize,
    ext7_signaled: usize,
    dct_only_txbs: usize,
    edge_clipped_txb_blocks: usize,
    // Chroma reconstruction accounting: CfL blocks that actually PREDICT
    // (chroma-reference, UV_CFL_PRED) per subsampling; non-CfL UV predictions;
    // sub-8x8 shared-chroma reference blocks (a 1-mi dimension on a subsampled
    // axis); joint-sign + alpha-index diversity among predicting CfL blocks;
    // chroma coefficient eobs; and non-DCT chroma transform types.
    cfl_predicted_420: usize,
    cfl_predicted_444: usize,
    cfl_predicted_422: usize,
    uv_non_cfl_blocks: usize,
    shared_chroma_blocks: usize,
    cfl_js_seen: [bool; 8],
    /// 256-entry bitset over observed `cfl_alpha_idx` values.
    cfl_alpha_idx_seen: [u64; 4],
    uv_eob_pos: usize,
    uv_eob_zero: usize,
    uv_non_dct_txbs: usize,
    uv_angle_nonzero: usize,
    // TX_MODE_SELECT accounting (from the DECODER's output): which of the 19
    // tx sizes were actually decoded, how many blocks decoded a non-max depth,
    // and how many blocks ran a real multi-txb grid (>1 txb).
    tx_sizes_decoded: [bool; 19],
    tx_depth_nonzero: usize,
    multi_txb_blocks: usize,
    max_txbs_in_block: usize,
    // tx_size_cdf (cat, ctx) instances that adapted anywhere in the sweep.
    tx_cells: [[bool; 3]; 4],
    // Delta-q accounting (delta-q sweep cases only, from the DECODER's
    // records): distinct effective qindexes decoded; distinct effective
    // qindexes among blocks that actually reconstructed luma coefficients
    // (different dequant rows genuinely exercised in the recon); written
    // reduced-delta stats (both signs + the |reduced| >= 3 exp-Golomb
    // remainder path); the sb-sized-skip gate-out arm; delta CDF adaptation;
    // non-zero delta-lf carries observed.
    /// 256-entry bitsets over observed effective qindex values.
    dq_qindex_seen: [u64; 4],
    dq_qindex_recon_seen: [u64; 4],
    dq_pos: usize,
    dq_neg: usize,
    dq_golomb: usize,
    dq_gated_sb_skip: usize,
    dq_cdf_adapted: bool,
    dlf_cdf_adapted: bool,
    dlf_multi_adapted: [bool; 4],
    dlf_nonzero_carries: usize,
    // FRAME_CONTEXT selection diversity: which context instances adapted
    // (final decoder CDFs differ from the initial fill) anywhere in the sweep.
    kf_y_cells: [[bool; 5]; 5],
    skip_ctxs: [bool; 3],
    angle_insts: [bool; 8],
    uv_insts: [[bool; 13]; 2],
    fi_bsizes: [bool; 22],
    ext7_cells: [[bool; 13]; 4],
    ext5_cells: [[bool; 13]; 4],
}

// ---- the mirror mini-encoder --------------------------------------------------------

struct Mirror<'a> {
    cfg: &'a KfTileConfig,
    src: &'a [u16],
    src_u: &'a [u16],
    src_v: &'a [u16],
    recon: Vec<u16>,
    stride: usize,
    recon_u: Vec<u16>,
    recon_v: Vec<u16>,
    stride_uv: usize,
    above_e: [Vec<i8>; 3],
    left_e: [[i8; 32]; 3],
    above_p: Vec<i8>,
    left_p: [i8; 32],
    /// Txfm-context byte arrays (`above_txfm_context`/`left_txfm_context`),
    /// mirroring the decoder's: init 64, stamped by `set_txfm_ctxs` per block.
    above_t: Vec<u8>,
    left_t: [u8; 32],
    /// Per-mi mode-info grid — the encoder's own `xd->above_mbmi/left_mbmi`
    /// source for every context selection (mirrors the decoder's grid).
    mi: Vec<MiNbrKf>,
    /// Per-mi uv-mode grid (chroma edge-filter-type neighbours).
    mi_uv: Vec<i8>,
    /// The mirror's own CfL store, fed from ITS reconstruction feedback.
    cfl: CflCtx,
    st: KfBlockState,
    /// The live per-plane `[dc, ac]` dequant rows (segment 0) — the mirror's
    /// counterpart of the decoder's `parse_decode_block` recompute: frame
    /// constant from `base_qindex` without delta-q, else refreshed per block
    /// from the write-side `current_base_qindex` carry.
    dequants: [[i16; 2]; 3],
    /// The SB-level delta-q target the mirror decided for the current
    /// superblock (`mbmi->current_qindex` of every block in it whose delta
    /// gate fires; the C encoder decides it once per SB in setup_delta_q).
    sb_target_qindex: i32,
    /// Per-SB delta-lf targets (multi per-plane + single from-base).
    sb_target_lf: [i32; 4],
    sb_target_lf_base: i32,
    sb_cdef_strength: i32,
    sb_cdef_done: bool,
    tree: Vec<i8>,
    blocks: Vec<DecodedBlockKf>,
}

impl<'a> Mirror<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &'a KfTileConfig,
        src: &'a [u16],
        src_u: &'a [u16],
        src_v: &'a [u16],
        stride: usize,
        recon_init: u16,
    ) -> Self {
        let aligned_rows = (cfg.mi_rows as usize).div_ceil(16) * 16;
        let stride_uv = if cfg.monochrome {
            0
        } else {
            stride >> cfg.subsampling_x
        };
        let uv_len = if cfg.monochrome {
            0
        } else {
            stride_uv * ((aligned_rows * 4) >> cfg.subsampling_y)
        };
        let st = KfBlockState {
            segid_preskip: false,
            seg_enabled: false,
            update_map: false,
            seg_pred: 0,
            last_active_segid: 0,
            seg_skip_active: false,
            mi_row: 0,
            mi_col: 0,
            mib_size: 16,
            sb_size: BLOCK_64X64,
            bsize: BLOCK_64X64,
            coded_lossless: false,
            allow_intrabc: false,
            cdef_bits: cfg.cdef_bits,
            dq_present: cfg.delta_q_present,
            dlf_present: cfg.delta_lf_present,
            dlf_multi: cfg.delta_lf_multi,
            num_planes: if cfg.monochrome { 1 } else { 3 },
            dq_res: cfg.delta_q_res,
            dlf_res: cfg.delta_lf_res,
            monochrome: cfg.monochrome,
            is_chroma_ref: !cfg.monochrome,
            cfl_allowed: false,
            allow_palette: false,
            bit_depth: cfg.bd,
            filter_allowed: false, // real gate applied via the follow-up write
            mb_to_top_edge: 0,
            has_above: false,
            has_left: false,
            cdef_transmitted: [false; 4],
            current_base_qindex: cfg.base_qindex,
            xd_delta_lf: [0; 4],
            xd_delta_lf_from_base: 0,
        };
        Mirror {
            cfg,
            src,
            src_u,
            src_v,
            recon: vec![recon_init; stride * aligned_rows * 4],
            stride,
            recon_u: vec![recon_init; uv_len],
            recon_v: vec![recon_init; uv_len],
            stride_uv,
            above_e: [
                vec![0; stride / 4],
                vec![0; stride / 4],
                vec![0; stride / 4],
            ],
            left_e: [[0; 32]; 3],
            above_p: vec![0; stride / 4],
            left_p: [0; 32],
            above_t: vec![TXFM_CTX_INIT; stride / 4],
            left_t: [TXFM_CTX_INIT; 32],
            mi: vec![
                MiNbrKf {
                    y_mode: 0,
                    skip_txfm: 0
                };
                (cfg.mi_rows * cfg.mi_cols) as usize
            ],
            mi_uv: vec![0; (cfg.mi_rows * cfg.mi_cols) as usize],
            cfl: CflCtx::new(cfg.subsampling_x as i32, cfg.subsampling_y as i32),
            st,
            dequants: plane_dequants(cfg, cfg.base_qindex),
            sb_target_qindex: cfg.base_qindex,
            sb_target_lf: [0; 4],
            sb_target_lf_base: 0,
            sb_cdef_strength: 0,
            sb_cdef_done: false,
            tree: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// The `xd->above_mbmi` / `xd->left_mbmi` neighbours of the block at
    /// `(mi_row, mi_col)` — identical semantics to the decode driver's grid.
    fn neighbours(&self, mi_row: i32, mi_col: i32) -> (Option<MiNbrKf>, Option<MiNbrKf>) {
        let cols = self.cfg.mi_cols;
        let above = (mi_row > 0).then(|| self.mi[((mi_row - 1) * cols + mi_col) as usize]);
        let left = (mi_col > 0).then(|| self.mi[(mi_row * cols + mi_col - 1) as usize]);
        (above, left)
    }

    #[allow(clippy::too_many_arguments)]
    fn set_entropy_ctx(
        &mut self,
        plane: usize,
        cul: i8,
        a_base: usize,
        l_base: usize,
        blk_row: usize,
        blk_col: usize,
        txw: usize,
        txh: usize,
        blocks_wide: usize,
        blocks_high: usize,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
    ) {
        let a0 = a_base + blk_col;
        if cul != 0 && mb_to_right_edge < 0 {
            let n = txw.min(blocks_wide.saturating_sub(blk_col));
            self.above_e[plane][a0..a0 + n].fill(cul);
            self.above_e[plane][a0 + n..a0 + txw].fill(0);
        } else {
            self.above_e[plane][a0..a0 + txw].fill(cul);
        }
        let l0 = l_base + blk_row;
        if cul != 0 && mb_to_bottom_edge < 0 {
            let n = txh.min(blocks_high.saturating_sub(blk_row));
            self.left_e[plane][l0..l0 + n].fill(cul);
            self.left_e[plane][l0 + n..l0 + txh].fill(0);
        } else {
            self.left_e[plane][l0..l0 + txh].fill(cul);
        }
    }

    /// Choose + write + reconstruct one leaf block; record the expected decode.
    #[allow(clippy::too_many_arguments)]
    fn encode_block(
        &mut self,
        enc: &mut OdEcEnc,
        cdfs: &mut KfFrameContext,
        rng: &mut Rng,
        cov: &mut Coverage,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
        partition: usize,
    ) {
        let cfg = self.cfg;
        let up_available = mi_row > 0;
        let left_available = mi_col > 0;
        let (ss_x, ss_y) = (cfg.subsampling_x, cfg.subsampling_y);
        let chroma_ref = is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y);
        let cfl_allowed = !cfg.monochrome && is_cfl_allowed(bsize, false, ss_x, ss_y);
        let (above, left) = self.neighbours(mi_row, mi_col);

        // --- choose the block's mode info ---
        let y_mode = (rng.next() % 13) as i32;
        let angle_y = if use_angle_delta(bsize) && is_directional_mode(y_mode) {
            (rng.next() % 7) as i32 - 3
        } else {
            0
        };
        // UV fields exist only on chroma-reference blocks (the decoder's
        // read_kf_tail else-branch reports zeros for the rest).
        let (uv_mode, cfl_idx, js, angle_uv) = if !cfg.monochrome && chroma_ref {
            let n = if cfl_allowed { 14 } else { 13 };
            let uv = (rng.next() % n) as i32;
            let (idx, sign) = if uv == 13 {
                let js = (rng.next() % 8) as i32;
                let (su, sv) = ((js + 1) / 3, (js + 1) % 3);
                let u = if su != 0 { (rng.next() % 16) as i32 } else { 0 };
                let v = if sv != 0 { (rng.next() % 16) as i32 } else { 0 };
                ((u << 4) | v, js)
            } else {
                (0, 0)
            };
            let ang = if use_angle_delta(bsize) && is_directional_mode(get_uv_mode(uv as usize)) {
                (rng.next() % 7) as i32 - 3
            } else {
                0
            };
            (uv, idx, sign, ang)
        } else {
            (0, 0, 0, 0)
        };
        let skip = rng.next().is_multiple_of(8) as i32;
        let fi_allowed = filter_intra_allowed(cfg.enable_filter_intra, bsize, y_mode, 0);
        let use_fi = if fi_allowed && rng.next() & 1 == 1 {
            1
        } else {
            0
        };
        let fi_mode = if use_fi != 0 {
            (rng.next() % 5) as i32
        } else {
            0
        };
        // write_delta_q_params gate (bitstream.c): the SB-upper-left block
        // codes the SB's delta targets unless it is an sb-sized skip block;
        // every block's mbmi then carries the (possibly unchanged) running
        // values — which is exactly what the decoder reports back.
        let sb_upper_left = (mi_row & 15) == 0 && (mi_col & 15) == 0;
        let dq_coded = cfg.delta_q_present && sb_upper_left && (bsize != BLOCK_64X64 || skip == 0);
        let (cur_q, dlf, dlfb) = if !cfg.delta_q_present {
            (
                cfg.base_qindex,
                self.st.xd_delta_lf,
                self.st.xd_delta_lf_from_base,
            )
        } else if dq_coded {
            let reduced = (self.sb_target_qindex - self.st.current_base_qindex) / cfg.delta_q_res;
            if reduced >= 3 || reduced <= -3 {
                cov.dq_golomb += 1;
            }
            if reduced > 0 {
                cov.dq_pos += 1;
            } else if reduced < 0 {
                cov.dq_neg += 1;
            }
            (
                self.sb_target_qindex,
                if cfg.delta_lf_present && cfg.delta_lf_multi {
                    self.sb_target_lf
                } else {
                    self.st.xd_delta_lf
                },
                if cfg.delta_lf_present && !cfg.delta_lf_multi {
                    self.sb_target_lf_base
                } else {
                    self.st.xd_delta_lf_from_base
                },
            )
        } else {
            if cfg.delta_q_present && sb_upper_left {
                cov.dq_gated_sb_skip += 1; // sb-sized skip block: nothing coded
            }
            (
                self.st.current_base_qindex,
                self.st.xd_delta_lf,
                self.st.xd_delta_lf_from_base,
            )
        };
        let info = MbModeInfoKf {
            segment_id: 0,
            skip,
            cdef_strength: self.sb_cdef_strength,
            current_qindex: cur_q,
            delta_lf: dlf,
            delta_lf_from_base: dlfb,
            use_intrabc: 0,
            dv_row: 0,
            dv_col: 0,
            y_mode,
            angle_delta_y: angle_y,
            uv_mode,
            cfl_alpha_idx: cfl_idx,
            cfl_joint_sign: js,
            angle_delta_uv: angle_uv,
            palette_size: [0, 0],
            use_filter_intra: use_fi,
            filter_intra_mode: fi_mode,
        };
        cov.y_modes[y_mode as usize] += 1;
        if use_fi != 0 {
            cov.fi_used += 1;
        }
        if angle_y != 0 {
            cov.angle_nonzero += 1;
        }
        if skip != 0 {
            cov.skip_blocks += 1;
        }
        if uv_mode == 13 {
            cov.cfl_uv_blocks += 1;
        }

        // --- write the mode info (write_mb_modes_kf_fc: full per-symbol
        // FRAME_CONTEXT selection from the neighbour grid) ---
        self.st.mi_row = mi_row;
        self.st.mi_col = mi_col;
        self.st.bsize = bsize;
        self.st.is_chroma_ref = chroma_ref;
        self.st.cfl_allowed = cfl_allowed;
        self.st.mb_to_top_edge = -(mi_row * 32);
        self.st.has_above = up_available;
        self.st.has_left = left_available;
        write_mb_modes_kf_fc(
            enc,
            &info,
            cdfs,
            &mut self.st,
            cfg.enable_filter_intra,
            above,
            left,
        );
        // parse_decode_block counterpart: with delta-q present, refresh the
        // live per-plane dequant rows from the (possibly just advanced)
        // current_base_qindex carry — the quantize AND the reconstruction
        // below both use the block-effective steps, like the decoder.
        if cfg.delta_q_present {
            debug_assert_eq!(self.st.current_base_qindex, cur_q);
            self.dequants = plane_dequants(cfg, self.st.current_base_qindex);
        }
        // What the decoder will report for cdef: coded at the first non-skip
        // block of the SB, -1 elsewhere (write_cdef threads cdef_transmitted).
        let cdef_coded = skip == 0;
        let expected_cdef = if cdef_coded && !self.sb_cdef_done {
            self.sb_cdef_done = true;
            self.sb_cdef_strength
        } else {
            -1
        };

        // --- choose + write the block's transform size (write_modes_b order:
        // after the mode info, before any coefficient symbols); intra blocks
        // write it even when skipped (`!(is_inter_tx && skip_txfm)`) ---
        let bw = MI_SIZE_WIDE[bsize] as usize;
        let bh = MI_SIZE_HIGH[bsize] as usize;
        let tx_size = if bsize > 0 {
            // block_signals_txsize
            if cfg.tx_mode == TxMode::Select {
                let max_depths = bsize_to_max_depth(bsize);
                // pseudo-random depth drives varied per-block tx sizes
                let depth = (rng.next() % (max_depths as u64 + 1)) as i32;
                let tx = depth_to_tx_size(depth, bsize);
                let cat = bsize_to_tx_size_cat(bsize) as usize;
                let ctx = get_tx_size_context(
                    bsize,
                    self.above_t[mi_col as usize],
                    self.left_t[(mi_row & 31) as usize],
                    up_available,
                    left_available,
                    None,
                    None,
                );
                // write_selected_tx_size (bitstream.c): the encoder-side
                // depth recomputation (tx_size_to_depth) round-trips the choice
                write_selected_tx_size(
                    enc,
                    &mut cdfs.tx_size[cat][ctx],
                    bsize,
                    tx_size_to_depth(tx, bsize),
                    max_depths,
                );
                tx
            } else {
                tx_size_from_tx_mode(bsize, cfg.tx_mode)
            }
        } else {
            MAX_TXSIZE_RECT_LOOKUP[bsize]
        };
        // set_txfm_ctxs, skip arg 0 for intra (C passes literal 0 on the
        // write_selected_tx_size path and `skip && is_inter` on the other).
        set_txfm_ctxs(
            &mut self.above_t[mi_col as usize..],
            &mut self.left_t[(mi_row & 31) as usize..],
            tx_size,
            bw,
            bh,
            false,
        );

        // The chroma-side geometry (shared-chroma group origin + context bases),
        // mirroring the decoder.
        let adj_row = if ss_y != 0 && (mi_row & 1) != 0 && MI_SIZE_HIGH[bsize] == 1 {
            mi_row - 1
        } else {
            mi_row
        };
        let adj_col = if ss_x != 0 && (mi_col & 1) != 0 && MI_SIZE_WIDE[bsize] == 1 {
            mi_col - 1
        } else {
            mi_col
        };
        let uv_a_base = (adj_col >> ss_x) as usize;
        let uv_l_base = ((adj_row & 31) >> ss_y) as usize;

        // --- skip blocks reset their entropy-context footprint (all planes) ---
        if skip != 0 {
            let a0 = mi_col as usize;
            self.above_e[0][a0..a0 + bw].fill(0);
            let l0 = (mi_row & 31) as usize;
            self.left_e[0][l0..l0 + bh].fill(0);
            if !cfg.monochrome && chroma_ref {
                let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
                let (uw, uh) = (
                    MI_SIZE_WIDE[plane_bsize] as usize,
                    MI_SIZE_HIGH[plane_bsize] as usize,
                );
                for plane in 1..=2 {
                    self.above_e[plane][uv_a_base..uv_a_base + uw].fill(0);
                    self.left_e[plane][uv_l_base..uv_l_base + uh].fill(0);
                }
            }
        }

        // --- per-txb: predict -> residual -> quantize -> write -> reconstruct ---
        let (txw, txh) = (TX_SIZE_WIDE_UNIT[tx_size], TX_SIZE_HIGH_UNIT[tx_size]);
        let (txwpx, txhpx) = (TX_SIZE_WIDE[tx_size], TX_SIZE_HIGH[tx_size]);
        let mb_to_right_edge = (cfg.mi_cols - MI_SIZE_WIDE[bsize] - mi_col) * 32;
        let mb_to_bottom_edge = (cfg.mi_rows - MI_SIZE_HIGH[bsize] - mi_row) * 32;
        let max_blocks_wide = max_block_units(BLOCK_SIZE_WIDE[bsize], mb_to_right_edge);
        let max_blocks_high = max_block_units(BLOCK_SIZE_HIGH[bsize], mb_to_bottom_edge);
        if max_blocks_wide < BLOCK_SIZE_WIDE[bsize] as usize / 4
            || max_blocks_high < BLOCK_SIZE_HIGH[bsize] as usize / 4
        {
            cov.edge_clipped_txb_blocks += 1;
        }
        // get_filt_type from the same neighbours the mode contexts used.
        let is_smooth = |m: Option<MiNbrKf>| m.is_some_and(|n| (9..=11).contains(&n.y_mode));
        let filt_type = (is_smooth(above) || is_smooth(left)) as i32;
        // av1_read_tx_type gate: the FRAME-level qindex (not the delta-q carry).
        let signal_gate = cfg.base_qindex > 0 && skip == 0;
        let set_type = ext_tx_set_type(tx_size, false, cfg.reduced_tx_set);
        let mut scratch = vec![0u16; txwpx * txhpx];
        let mut residual = vec![0i16; txwpx * txhpx];
        let mut txbs = Vec::new();
        let dequant = self.dequants[0];
        let bq = bquant_for(dequant);
        let qp = QuantParams {
            zbin: &bq.zbin,
            round: &bq.round,
            quant: &bq.quant,
            quant_shift: &bq.quant_shift,
            dequant: &dequant,
            qm: None,
            iqm: None,
            bd: cfg.bd as u8,
        };

        let mut blk_row = 0usize;
        while blk_row < max_blocks_high {
            let mut blk_col = 0usize;
            while blk_col < max_blocks_wide {
                // predict from the encoder's own recon-so-far (the feedback loop)
                let (n_top, n_tr, n_left, n_bl) = intra_avail(
                    BLOCK_64X64,
                    bsize,
                    mi_row,
                    mi_col,
                    up_available,
                    left_available,
                    cfg.mi_cols,
                    cfg.mi_rows,
                    partition,
                    tx_size,
                    0,
                    0,
                    blk_row as i32,
                    blk_col as i32,
                    BLOCK_SIZE_WIDE[bsize],
                    BLOCK_SIZE_HIGH[bsize],
                    cfg.mi_cols,
                    cfg.mi_rows,
                    info.y_mode as usize,
                    info.angle_delta_y * ANGLE_STEP,
                    info.use_filter_intra != 0,
                );
                let off = ((mi_row * 4) as usize + blk_row * 4) * self.stride
                    + (mi_col * 4) as usize
                    + blk_col * 4;
                predict_intra_high(
                    &self.recon,
                    off,
                    self.stride,
                    &mut scratch,
                    txwpx,
                    info.y_mode as usize,
                    info.angle_delta_y * ANGLE_STEP,
                    info.use_filter_intra != 0,
                    info.filter_intra_mode as usize,
                    cfg.disable_edge_filter,
                    filt_type,
                    tx_size,
                    usize::try_from(n_top).expect("n_top_px"),
                    n_tr,
                    usize::try_from(n_left).expect("n_left_px"),
                    n_bl,
                    cfg.bd,
                );
                for r in 0..txhpx {
                    let d = off + r * self.stride;
                    self.recon[d..d + txwpx].copy_from_slice(&scratch[r * txwpx..(r + 1) * txwpx]);
                }

                if skip == 0 {
                    // residual = source − prediction over the full tx rect
                    for r in 0..txhpx {
                        let s = off + r * self.stride;
                        for c in 0..txwpx {
                            residual[r * txwpx + c] =
                                self.src[s + c] as i16 - scratch[r * txwpx + c] as i16;
                        }
                    }
                    // per-txb tx_type: uniform over the set when signalled
                    let tx_type = if signal_gate {
                        match set_type {
                            2 => EXT_USED_DTT4_IDTX[(rng.next() % 5) as usize],
                            3 => EXT_USED_DTT4_IDTX_1DDCT[(rng.next() % 7) as usize],
                            _ => 0,
                        }
                    } else {
                        0
                    };
                    if signal_gate && set_type == 2 {
                        cov.ext5_signaled += 1;
                    } else if signal_gate && set_type == 3 {
                        cov.ext7_signaled += 1;
                    } else {
                        cov.dct_only_txbs += 1;
                    }
                    let a0 = mi_col as usize + blk_col;
                    let l0 = (mi_row & 31) as usize + blk_row;
                    let (tsc, dsc) = get_txb_ctx(
                        bsize,
                        tx_size,
                        0,
                        &self.above_e[0][a0..],
                        &self.left_e[0][l0..],
                    );
                    let r = xform_quant(&residual, tx_size, tx_type, QuantKind::B, &qp, false);
                    let ext = intra_ext_tx_cdf(
                        &mut cdfs.ext_tx_1ddct,
                        &mut cdfs.ext_tx_dtt4,
                        tx_size,
                        cfg.reduced_tx_set,
                        info.use_filter_intra != 0,
                        info.filter_intra_mode as usize,
                        info.y_mode as usize,
                    );
                    write_coeffs_txb_full(
                        enc,
                        &mut cdfs.coeff,
                        ext,
                        &r.qcoeff,
                        r.eob as usize,
                        tx_size,
                        tx_type,
                        0,
                        tsc as usize,
                        dsc as usize,
                        true,
                        false,
                        cfg.reduced_tx_set,
                        info.use_filter_intra != 0,
                        info.filter_intra_mode as usize,
                        info.y_mode as usize,
                        signal_gate,
                    );
                    self.set_entropy_ctx(
                        0,
                        r.txb_entropy_ctx as i8,
                        mi_col as usize,
                        (mi_row & 31) as usize,
                        blk_row,
                        blk_col,
                        txw,
                        txh,
                        max_blocks_wide,
                        max_blocks_high,
                        mb_to_right_edge,
                        mb_to_bottom_edge,
                    );
                    if r.eob > 0 {
                        cov.eob_pos += 1;
                        aom_encode::reconstruct_txb(
                            &mut self.recon[off..],
                            self.stride,
                            tx_size,
                            tx_type,
                            &r.qcoeff,
                            dequant,
                            None,
                            cfg.bd,
                        );
                        txbs.push((r.eob as usize, tx_type));
                    } else {
                        cov.eob_zero += 1;
                        // decoder infers DCT_DCT for an all-zero txb
                        txbs.push((0, 0));
                    }
                } else {
                    txbs.push((0, 0));
                }
                // CfL luma store from the MIRROR's reconstruction feedback
                // (store_cfl_required; skip blocks store their prediction).
                if !cfg.monochrome && (!chroma_ref || uv_mode == UV_CFL_PRED) {
                    let block_off = (mi_row * 4) as usize * self.stride + (mi_col * 4) as usize;
                    cfl_store_tx(
                        &mut self.cfl,
                        &self.recon,
                        block_off,
                        self.stride,
                        blk_row as i32,
                        blk_col as i32,
                        tx_size,
                        bsize,
                        mi_row,
                        mi_col,
                    );
                }
                blk_col += txw;
            }
            blk_row += txh;
        }

        // --- chroma planes: predict (+CfL) -> residual -> quantize -> write ->
        // reconstruct, mirroring the decoder's plane loop exactly ---
        let mut txbs_uv = Vec::new();
        if !cfg.monochrome && chroma_ref {
            let plane_bsize = get_plane_block_size(bsize, ss_x, ss_y);
            assert_ne!(plane_bsize, 255, "mirror produced an invalid chroma bsize");
            let uv_tx = max_uv_txsize(bsize, ss_x, ss_y);
            let (uv_txw, uv_txh) = (TX_SIZE_WIDE_UNIT[uv_tx], TX_SIZE_HIGH_UNIT[uv_tx]);
            let (uv_txwpx, uv_txhpx) = (TX_SIZE_WIDE[uv_tx], TX_SIZE_HIGH[uv_tx]);
            let unit_width = ((max_blocks_wide as i32 + ss_x as i32) >> ss_x) as usize;
            let unit_height = ((max_blocks_high as i32 + ss_y as i32) >> ss_y) as usize;
            let blocks_wide_uv =
                max_block_units_ss(BLOCK_SIZE_WIDE[plane_bsize], mb_to_right_edge, ss_x);
            let blocks_high_uv =
                max_block_units_ss(BLOCK_SIZE_HIGH[plane_bsize], mb_to_bottom_edge, ss_y);
            let wpx = ((MI_SIZE_WIDE[bsize] * 4) >> ss_x).max(4);
            let hpx = ((MI_SIZE_HIGH[bsize] * 4) >> ss_y).max(4);
            let bsize_uv = scale_chroma_bsize(bsize, ss_x, ss_y);
            let up_uv = adj_row > 0;
            let left_uv = adj_col > 0;
            let cols = cfg.mi_cols;
            let base_row = mi_row - (mi_row & ss_y as i32);
            let base_col = mi_col - (mi_col & ss_x as i32);
            let uv_smooth = |m: i8| (9..=11).contains(&m);
            let ab_sm = up_uv
                && uv_smooth(self.mi_uv[((base_row - 1) * cols + base_col + ss_x as i32) as usize]);
            let le_sm = left_uv
                && uv_smooth(self.mi_uv[((base_row + ss_y as i32) * cols + base_col - 1) as usize]);
            let filt_type_uv = (ab_sm || le_sm) as i32;
            let uv_org = ((adj_row * 4) >> ss_y) as usize * self.stride_uv
                + ((adj_col * 4) >> ss_x) as usize;
            let tt_uv = uv_tx_type(uv_mode, uv_tx, cfg.reduced_tx_set);
            let mode_uv = get_uv_mode(uv_mode as usize) as usize;
            let mut scratch_uv = vec![0u16; uv_txwpx * uv_txhpx];
            let mut residual_uv = vec![0i16; uv_txwpx * uv_txhpx];
            let mut no_ext: [u16; 0] = [];

            // coverage
            if uv_mode == UV_CFL_PRED {
                match (ss_x, ss_y) {
                    (1, 1) => cov.cfl_predicted_420 += 1,
                    (1, 0) => cov.cfl_predicted_422 += 1,
                    _ => cov.cfl_predicted_444 += 1,
                }
                cov.cfl_js_seen[js as usize] = true;
                cov.cfl_alpha_idx_seen[(cfl_idx >> 6) as usize] |= 1u64 << (cfl_idx & 63);
            } else {
                cov.uv_non_cfl_blocks += 1;
            }
            if (ss_x != 0 && MI_SIZE_WIDE[bsize] == 1) || (ss_y != 0 && MI_SIZE_HIGH[bsize] == 1) {
                cov.shared_chroma_blocks += 1;
            }
            if angle_uv != 0 {
                cov.uv_angle_nonzero += 1;
            }
            if tt_uv != 0 {
                cov.uv_non_dct_txbs += 1;
            }

            for plane in 1..=2usize {
                let dequant_p = self.dequants[plane];
                let bq_uv = bquant_for(dequant_p);
                let qp_uv = QuantParams {
                    zbin: &bq_uv.zbin,
                    round: &bq_uv.round,
                    quant: &bq_uv.quant,
                    quant_shift: &bq_uv.quant_shift,
                    dequant: &dequant_p,
                    qm: None,
                    iqm: None,
                    bd: cfg.bd as u8,
                };
                let src_uv = if plane == 1 { self.src_u } else { self.src_v };
                let mut blk_row = 0usize;
                while blk_row < unit_height {
                    let mut blk_col = 0usize;
                    while blk_col < unit_width {
                        // (1) predict from the mirror's own chroma recon (+CfL
                        // AC from the mirror's own luma store).
                        let (n_top, n_tr, n_left, n_bl) = intra_avail(
                            BLOCK_64X64,
                            bsize_uv,
                            adj_row,
                            adj_col,
                            up_uv,
                            left_uv,
                            cfg.mi_cols,
                            cfg.mi_rows,
                            partition,
                            uv_tx,
                            ss_x as i32,
                            ss_y as i32,
                            blk_row as i32,
                            blk_col as i32,
                            wpx,
                            hpx,
                            cfg.mi_cols,
                            cfg.mi_rows,
                            mode_uv,
                            angle_uv * ANGLE_STEP,
                            false,
                        );
                        let off_uv = uv_org + (blk_row * 4) * self.stride_uv + blk_col * 4;
                        {
                            let plane_recon = if plane == 1 {
                                &self.recon_u
                            } else {
                                &self.recon_v
                            };
                            predict_intra_high(
                                plane_recon,
                                off_uv,
                                self.stride_uv,
                                &mut scratch_uv,
                                uv_txwpx,
                                mode_uv,
                                angle_uv * ANGLE_STEP,
                                false,
                                0,
                                cfg.disable_edge_filter,
                                filt_type_uv,
                                uv_tx,
                                usize::try_from(n_top).expect("n_top_px"),
                                n_tr,
                                usize::try_from(n_left).expect("n_left_px"),
                                n_bl,
                                cfg.bd,
                            );
                        }
                        if uv_mode == UV_CFL_PRED {
                            cfl_predict_block(
                                &mut self.cfl,
                                &mut scratch_uv,
                                0,
                                uv_txwpx,
                                uv_tx,
                                plane,
                                cfl_idx,
                                js,
                                cfg.bd,
                            );
                        }
                        {
                            let plane_recon = if plane == 1 {
                                &mut self.recon_u
                            } else {
                                &mut self.recon_v
                            };
                            for r in 0..uv_txhpx {
                                let d = off_uv + r * self.stride_uv;
                                plane_recon[d..d + uv_txwpx]
                                    .copy_from_slice(&scratch_uv[r * uv_txwpx..(r + 1) * uv_txwpx]);
                            }
                        }

                        if skip == 0 {
                            // (2) residual -> quantize -> write (plane_type 1:
                            // no tx_type symbol) -> entropy contexts.
                            for r in 0..uv_txhpx {
                                let s = off_uv + r * self.stride_uv;
                                for c in 0..uv_txwpx {
                                    residual_uv[r * uv_txwpx + c] =
                                        src_uv[s + c] as i16 - scratch_uv[r * uv_txwpx + c] as i16;
                                }
                            }
                            let (tsc, dsc) = get_txb_ctx(
                                plane_bsize,
                                uv_tx,
                                plane,
                                &self.above_e[plane][uv_a_base + blk_col..],
                                &self.left_e[plane][uv_l_base + blk_row..],
                            );
                            let r = xform_quant(
                                &residual_uv,
                                uv_tx,
                                tt_uv,
                                QuantKind::B,
                                &qp_uv,
                                false,
                            );
                            write_coeffs_txb_full(
                                enc,
                                &mut cdfs.coeff,
                                &mut no_ext,
                                &r.qcoeff,
                                r.eob as usize,
                                uv_tx,
                                tt_uv,
                                1,
                                tsc as usize,
                                dsc as usize,
                                true,
                                false,
                                cfg.reduced_tx_set,
                                false,
                                0,
                                0,
                                false,
                            );
                            self.set_entropy_ctx(
                                plane,
                                r.txb_entropy_ctx as i8,
                                uv_a_base,
                                uv_l_base,
                                blk_row,
                                blk_col,
                                uv_txw,
                                uv_txh,
                                blocks_wide_uv,
                                blocks_high_uv,
                                mb_to_right_edge,
                                mb_to_bottom_edge,
                            );
                            // (3) reconstruct through the same path.
                            if r.eob > 0 {
                                cov.uv_eob_pos += 1;
                                let plane_recon = if plane == 1 {
                                    &mut self.recon_u
                                } else {
                                    &mut self.recon_v
                                };
                                aom_encode::reconstruct_txb(
                                    &mut plane_recon[off_uv..],
                                    self.stride_uv,
                                    uv_tx,
                                    tt_uv,
                                    &r.qcoeff,
                                    dequant_p,
                                    None,
                                    cfg.bd,
                                );
                                txbs_uv.push((r.eob as usize, tt_uv));
                            } else {
                                cov.uv_eob_zero += 1;
                                txbs_uv.push((0, 0));
                            }
                        } else {
                            txbs_uv.push((0, 0));
                        }
                        blk_col += uv_txw;
                    }
                    blk_row += uv_txh;
                }
            }
        }

        // mode-info grid stamp (frame-cropped), for later blocks' context
        // selection + filt_type
        let x_mis = MI_SIZE_WIDE[bsize].min(cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * cfg.mi_cols + mi_col) as usize;
            self.mi[base..base + x_mis as usize].fill(MiNbrKf {
                y_mode,
                skip_txfm: skip,
            });
            self.mi_uv[base..base + x_mis as usize].fill(uv_mode as i8);
        }

        let mut expected_info = info;
        expected_info.cdef_strength = expected_cdef;
        self.blocks.push(DecodedBlockKf {
            mi_row,
            mi_col,
            bsize,
            partition,
            info: expected_info,
            tx_size,
            txbs,
            txbs_uv,
        });
    }

    /// Is this partition conformant for the frame subsampling? A partition is
    /// illegal when a leaf block size it produces has no valid chroma plane
    /// size (`av1_ss_size_lookup` = BLOCK_INVALID) — the C decoder rejects
    /// the 8x8-and-larger cases as corrupt and relies on conformance for the
    /// sub-8x8 ones (e.g. no 4xN chroma-reference blocks in 4:2:2). SPLIT
    /// only recurses into square nodes (always valid).
    fn partition_legal(&self, bsize: usize, p: usize) -> bool {
        if self.cfg.monochrome || (self.cfg.subsampling_x == 0 && self.cfg.subsampling_y == 0) {
            return true;
        }
        if p == PARTITION_SPLIT {
            return true;
        }
        let (ss_x, ss_y) = (self.cfg.subsampling_x, self.cfg.subsampling_y);
        let subsize = get_partition_subsize(bsize, p as i32) as usize;
        let bsize2 = get_partition_subsize(bsize, PARTITION_SPLIT as i32) as usize;
        let leaves: &[usize] = match p {
            4..=7 => &[subsize, bsize2], // HORZ/VERT_A/B mix both sizes
            _ => &[subsize],
        };
        leaves
            .iter()
            .all(|&b| get_plane_block_size(b, ss_x, ss_y) != 255)
    }

    /// Choose a legal partition for the node (mirrors the C decoder's edge rules:
    /// forced NONE below 8x8, HORZ/SPLIT at a bottom edge, VERT/SPLIT at a right
    /// edge, forced SPLIT off both, the full set in frame), filtered to the
    /// subsampling-conformant set.
    fn choose_partition(
        &self,
        rng: &mut Rng,
        bsize: usize,
        has_rows: bool,
        has_cols: bool,
    ) -> usize {
        if bsize < BLOCK_8X8 {
            return PARTITION_NONE;
        }
        match (has_rows, has_cols) {
            (false, false) => PARTITION_SPLIT,
            (false, true) => {
                let p = [PARTITION_HORZ, PARTITION_SPLIT][(rng.next() & 1) as usize];
                if self.partition_legal(bsize, p) {
                    p
                } else {
                    PARTITION_SPLIT
                }
            }
            (true, false) => {
                let p = [PARTITION_VERT, PARTITION_SPLIT][(rng.next() & 1) as usize];
                if self.partition_legal(bsize, p) {
                    p
                } else {
                    PARTITION_SPLIT
                }
            }
            (true, true) => {
                let n = partition_cdf_length(bsize);
                // bias splits a little so the sweep reaches small blocks
                if bsize > BLOCK_8X8 && rng.next() % 100 < 30 {
                    PARTITION_SPLIT
                } else {
                    loop {
                        let p = (rng.next() % n as u64) as usize;
                        if self.partition_legal(bsize, p) {
                            break p;
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_partition(
        &mut self,
        enc: &mut OdEcEnc,
        cdfs: &mut KfFrameContext,
        rng: &mut Rng,
        cov: &mut Coverage,
        mi_row: i32,
        mi_col: i32,
        bsize: usize,
    ) {
        if mi_row >= self.cfg.mi_rows || mi_col >= self.cfg.mi_cols {
            return;
        }
        let hbs = MI_SIZE_WIDE[bsize] / 2;
        let quarter_step = MI_SIZE_WIDE[bsize] / 4;
        let has_rows = (mi_row + hbs) < self.cfg.mi_rows;
        let has_cols = (mi_col + hbs) < self.cfg.mi_cols;
        let p = self.choose_partition(rng, bsize, has_rows, has_cols);
        if bsize >= BLOCK_8X8 {
            let ctx = partition_plane_context(
                &self.above_p,
                &self.left_p,
                mi_row as usize,
                mi_col as usize,
                bsize,
            ) as usize;
            write_partition(
                enc,
                &mut cdfs.partition[ctx],
                partition_cdf_length(bsize),
                p as i32,
                has_rows,
                has_cols,
                bsize,
            );
        }
        self.tree.push(p as i8);
        cov.partitions[p] += 1;
        let subsize = get_partition_subsize(bsize, p as i32) as usize;
        assert_ne!(subsize, 255, "mirror chose an invalid partition");
        let bsize2 = get_partition_subsize(bsize, PARTITION_SPLIT as i32) as usize;
        match p {
            PARTITION_NONE => self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col, subsize, p),
            PARTITION_HORZ => {
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col, subsize, p);
                if has_rows {
                    self.encode_block(enc, cdfs, rng, cov, mi_row + hbs, mi_col, subsize, p);
                }
            }
            PARTITION_VERT => {
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col, subsize, p);
                if has_cols {
                    self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col + hbs, subsize, p);
                }
            }
            PARTITION_SPLIT => {
                self.encode_partition(enc, cdfs, rng, cov, mi_row, mi_col, subsize);
                self.encode_partition(enc, cdfs, rng, cov, mi_row, mi_col + hbs, subsize);
                self.encode_partition(enc, cdfs, rng, cov, mi_row + hbs, mi_col, subsize);
                self.encode_partition(enc, cdfs, rng, cov, mi_row + hbs, mi_col + hbs, subsize);
            }
            4 => {
                // HORZ_A
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col, bsize2, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col + hbs, bsize2, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row + hbs, mi_col, subsize, p);
            }
            5 => {
                // HORZ_B
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col, subsize, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row + hbs, mi_col, bsize2, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row + hbs, mi_col + hbs, bsize2, p);
            }
            6 => {
                // VERT_A
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col, bsize2, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row + hbs, mi_col, bsize2, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col + hbs, subsize, p);
            }
            7 => {
                // VERT_B
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col, subsize, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row, mi_col + hbs, bsize2, p);
                self.encode_block(enc, cdfs, rng, cov, mi_row + hbs, mi_col + hbs, bsize2, p);
            }
            8 => {
                // HORZ_4
                for i in 0..4 {
                    let rr = mi_row + i * quarter_step;
                    if i > 0 && rr >= self.cfg.mi_rows {
                        break;
                    }
                    self.encode_block(enc, cdfs, rng, cov, rr, mi_col, subsize, p);
                }
            }
            9 => {
                // VERT_4
                for i in 0..4 {
                    let cc = mi_col + i * quarter_step;
                    if i > 0 && cc >= self.cfg.mi_cols {
                        break;
                    }
                    self.encode_block(enc, cdfs, rng, cov, mi_row, cc, subsize, p);
                }
            }
            _ => unreachable!(),
        }
        update_ext_partition_context(
            &mut self.above_p,
            &mut self.left_p,
            mi_row,
            mi_col,
            subsize,
            bsize,
            p as i32,
        );
    }

    fn encode_tile(
        &mut self,
        enc: &mut OdEcEnc,
        cdfs: &mut KfFrameContext,
        rng: &mut Rng,
        cov: &mut Coverage,
    ) {
        let mut mi_row = 0;
        while mi_row < self.cfg.mi_rows {
            self.left_e = [[0; 32]; 3];
            self.left_p = [0; 32];
            self.left_t = [TXFM_CTX_INIT; 32];
            let mut mi_col = 0;
            while mi_col < self.cfg.mi_cols {
                // new SB: cdef strength not yet transmitted for it
                self.sb_cdef_done = false;
                self.sb_cdef_strength = if self.cfg.cdef_bits > 0 {
                    (rng.next() % (1u64 << self.cfg.cdef_bits)) as i32
                } else {
                    0
                };
                // New SB: decide its delta-q/delta-lf targets (the C encoder
                // decides once per SB in setup_delta_q). Encoder-valid only:
                // carry + k*res kept inside [1, 255] / [-63, 63] by clamping
                // k, mirroring av1_adjust_q_from_delta_q_res's alignment —
                // the decoder's normative clamps stay no-ops and the carries
                // lockstep. |k| >= 3 exercises the exp-Golomb remainder path.
                if self.cfg.delta_q_present {
                    let res = self.cfg.delta_q_res;
                    let carry = self.st.current_base_qindex;
                    let k = ((rng.next() % 41) as i32 - 20)
                        .clamp(-((carry - 1) / res), (255 - carry) / res);
                    self.sb_target_qindex = carry + k * res;
                    if self.cfg.delta_lf_present {
                        let lres = self.cfg.delta_lf_res;
                        if self.cfg.delta_lf_multi {
                            // frame_lf_count ids are coded; the rest must
                            // report the untouched carries.
                            let n = if self.cfg.monochrome { 2 } else { 4 };
                            for id in 0..4 {
                                let c = self.st.xd_delta_lf[id];
                                self.sb_target_lf[id] = if id < n {
                                    let k = ((rng.next() % 11) as i32 - 5)
                                        .clamp(-((63 + c) / lres), (63 - c) / lres);
                                    c + k * lres
                                } else {
                                    c
                                };
                            }
                        } else {
                            let c = self.st.xd_delta_lf_from_base;
                            let k = ((rng.next() % 11) as i32 - 5)
                                .clamp(-((63 + c) / lres), (63 - c) / lres);
                            self.sb_target_lf_base = c + k * lres;
                        }
                    }
                }
                self.encode_partition(enc, cdfs, rng, cov, mi_row, mi_col, BLOCK_64X64);
                mi_col += 16;
            }
            mi_row += 16;
        }
    }
}

// ---- the roundtrip ------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SweepCase {
    mi_rows: i32,
    mi_cols: i32,
    bd: i32,
    monochrome: bool,
    ss_x: usize,
    ss_y: usize,
    cdef_bits: u32,
    disable_edge_filter: bool,
    enable_filter_intra: bool,
    reduced_tx_set: bool,
    /// `base_qindex > 0` (the tx-type signalling gate): true derives a
    /// per-seed base qindex in [20, 250]; false uses base_qindex = 0 (the
    /// gate-off path — combined with a non-zero y_dc delta so the frame is
    /// not coded_lossless in C terms).
    base_qindex_gt0: bool,
    /// Derive per-seed non-zero per-plane dc/ac deltas (y_dc; u_dc/u_ac;
    /// v_dc/v_ac — clearly distinct U vs V so a plane swap or shared-dequant
    /// bug hard-fails, as the old independent random dequants did).
    plane_deltas: bool,
    /// 0 = delta-q off; 1/2/4/8 = delta_q_present with this delta_q_res.
    dq_res: i32,
    /// Delta-lf (multi, res); requires dq_res > 0. No reconstruction effect
    /// (no loop filters) — validates the coded symbols + carried values.
    dlf: Option<(bool, i32)>,
    tx_mode: TxMode,
}

fn run_roundtrip(case: &SweepCase, seed: u64, cov: &mut Coverage) {
    let mut rng = Rng(seed);
    // Per-seed frame quant point: base qindex + per-plane dc/ac deltas
    // (dequants now DERIVE from these through av1_{dc,ac}_quant_QTX, as in
    // C). U and V deltas are drawn from disjoint sign/magnitude bands so a
    // plane swap or shared-dequant bug hard-fails; y_dc is non-zero whenever
    // plane deltas are on (keeps the base_qindex = 0 gate-off configs out of
    // C's coded_lossless condition).
    let base_qindex = if case.base_qindex_gt0 {
        rng.range(20, 251) as i32
    } else {
        0
    };
    let deltas = if case.plane_deltas {
        [
            (rng.range(1, 25) as i32) * if rng.next() & 1 == 1 { 1 } else { -1 }, // y_dc
            -(rng.range(8, 40) as i32),                                           // u_dc
            rng.range(8, 40) as i32,                                              // u_ac
            rng.range(41, 63) as i32,                                             // v_dc
            -(rng.range(41, 63) as i32),                                          // v_ac
        ]
    } else {
        [0; 5]
    };
    let cfg = KfTileConfig {
        mi_rows: case.mi_rows,
        mi_cols: case.mi_cols,
        bd: case.bd,
        monochrome: case.monochrome,
        subsampling_x: case.ss_x,
        subsampling_y: case.ss_y,
        cdef_bits: case.cdef_bits,
        disable_edge_filter: case.disable_edge_filter,
        enable_filter_intra: case.enable_filter_intra,
        tx_mode: case.tx_mode,
        reduced_tx_set: case.reduced_tx_set,
        base_qindex,
        y_dc_delta_q: deltas[0],
        u_dc_delta_q: deltas[1],
        u_ac_delta_q: deltas[2],
        v_dc_delta_q: deltas[3],
        v_ac_delta_q: deltas[4],
        delta_q_present: case.dq_res > 0,
        delta_q_res: case.dq_res.max(1),
        delta_lf_present: case.dlf.is_some(),
        delta_lf_multi: case.dlf.is_some_and(|(m, _)| m),
        delta_lf_res: case.dlf.map_or(1, |(_, r)| r),
        lr: Default::default(),
    };
    let aligned_cols = (cfg.mi_cols as usize).div_ceil(16) * 16;
    let aligned_rows = (cfg.mi_rows as usize).div_ceil(16) * 16;
    let stride = aligned_cols * 4;
    let stride_uv = stride >> cfg.subsampling_x;
    let uv_rows = (aligned_rows * 4) >> cfg.subsampling_y;
    let mask = (1u64 << cfg.bd) - 1;
    let mut src: Vec<u16> = (0..stride * aligned_rows * 4)
        .map(|_| (rng.next() & mask) as u16)
        .collect();
    let uv_len = if cfg.monochrome {
        0
    } else {
        stride_uv * uv_rows
    };
    let mut src_u: Vec<u16> = (0..uv_len).map(|_| (rng.next() & mask) as u16).collect();
    let mut src_v: Vec<u16> = (0..uv_len).map(|_| (rng.next() & mask) as u16).collect();
    // Carve some flat 64x64 regions (all planes): blocks there predict
    // near-perfectly, so the quantizer produces genuine all-zero txbs (the
    // txb_skip=1 decode path, luma and chroma).
    for sbr in 0..aligned_rows / 16 {
        for sbc in 0..aligned_cols / 16 {
            if rng.next().is_multiple_of(3) {
                let v = (rng.next() & mask) as u16;
                for r in 0..64 {
                    let base = (sbr * 64 + r) * stride + sbc * 64;
                    src[base..base + 64].fill(v);
                }
                if !cfg.monochrome {
                    let (cw, ch) = (64 >> cfg.subsampling_x, 64 >> cfg.subsampling_y);
                    let vu = (rng.next() & mask) as u16;
                    let vv = (rng.next() & mask) as u16;
                    for r in 0..ch {
                        let base = (sbr * ch + r) * stride_uv + sbc * cw;
                        src_u[base..base + cw].fill(vu);
                        src_v[base..base + cw].fill(vv);
                    }
                }
            }
        }
    }
    let cdfs0 = mk_frame_ctx(&mut rng);

    // encode (mirror), recon initialised to 0
    let mut enc_cdfs = cdfs0.clone();
    let mut mirror = Mirror::new(&cfg, &src, &src_u, &src_v, stride, 0);
    let mut enc = OdEcEnc::new();
    mirror.encode_tile(&mut enc, &mut enc_cdfs, &mut rng, cov);
    let bytes = enc.done().to_vec();

    // decode, recon initialised to the max pixel value (divergent on purpose,
    // chroma planes included)
    let mut dec_cdfs = cdfs0.clone();
    let mut dec = OdEcDec::new(&bytes);
    let got = decode_tile_kf(&mut dec, &cfg, &mut dec_cdfs, mask as u16);

    let what = format!(
        "case mi={}x{} bd={} mono={} ss={}{} cdef={} fi={} reduced={} q={base_qindex} dq_res={} dlf={:?} tx={:?} seed={seed:#x}",
        case.mi_rows,
        case.mi_cols,
        case.bd,
        case.monochrome,
        case.ss_x,
        case.ss_y,
        case.cdef_bits,
        case.enable_filter_intra,
        case.reduced_tx_set,
        case.dq_res,
        case.dlf,
        case.tx_mode,
    );
    // (a) partition tree + per-leaf decode records (mode info, per-txb eob/tx_type)
    assert_eq!(got.tree, mirror.tree, "{what}: partition tree");
    assert_eq!(got.blocks.len(), mirror.blocks.len(), "{what}: leaf count");
    for (i, (g, w)) in got.blocks.iter().zip(&mirror.blocks).enumerate() {
        assert_eq!(g, w, "{what}: block {i}");
    }
    // (b) byte-identical reconstruction over the frame crop, all planes
    assert_eq!(got.stride, stride, "{what}: stride");
    for row in 0..got.height {
        assert_eq!(
            got.recon[row * stride..row * stride + got.width],
            mirror.recon[row * stride..row * stride + got.width],
            "{what}: recon row {row}"
        );
    }
    if !cfg.monochrome {
        assert_eq!(got.stride_uv, stride_uv, "{what}: uv stride");
        assert_eq!(
            got.width_uv,
            (cfg.mi_cols as usize * 4) >> cfg.subsampling_x
        );
        assert_eq!(
            got.height_uv,
            (cfg.mi_rows as usize * 4) >> cfg.subsampling_y
        );
        for row in 0..got.height_uv {
            let s = row * stride_uv;
            assert_eq!(
                got.recon_u[s..s + got.width_uv],
                mirror.recon_u[s..s + got.width_uv],
                "{what}: recon U row {row}"
            );
            assert_eq!(
                got.recon_v[s..s + got.width_uv],
                mirror.recon_v[s..s + got.width_uv],
                "{what}: recon V row {row}"
            );
        }
    }
    // (c) every CDF in lockstep — the whole frame context
    assert_fc_eq(&enc_cdfs, &dec_cdfs, &what);
    // (d) tally which context instances adapted (vs the initial fill) for the
    // sweep-wide selection-diversity assertions
    for ((flag, new), old) in cov
        .kf_y_cells
        .iter_mut()
        .flatten()
        .zip(dec_cdfs.kf_y.iter().flatten())
        .zip(cdfs0.kf_y.iter().flatten())
    {
        *flag |= new != old;
    }
    for ((flag, new), old) in cov
        .skip_ctxs
        .iter_mut()
        .zip(&dec_cdfs.skip)
        .zip(&cdfs0.skip)
    {
        *flag |= new != old;
    }
    for ((flag, new), old) in cov
        .angle_insts
        .iter_mut()
        .zip(&dec_cdfs.angle_delta)
        .zip(&cdfs0.angle_delta)
    {
        *flag |= new != old;
    }
    for ((flag, new), old) in cov
        .uv_insts
        .iter_mut()
        .flatten()
        .zip(dec_cdfs.uv_mode.iter().flatten())
        .zip(cdfs0.uv_mode.iter().flatten())
    {
        *flag |= new != old;
    }
    for ((flag, new), old) in cov
        .fi_bsizes
        .iter_mut()
        .zip(&dec_cdfs.filter_intra)
        .zip(&cdfs0.filter_intra)
    {
        *flag |= new != old;
    }
    for ((flag, new), old) in cov
        .ext7_cells
        .iter_mut()
        .flatten()
        .zip(dec_cdfs.ext_tx_1ddct.iter().flatten())
        .zip(cdfs0.ext_tx_1ddct.iter().flatten())
    {
        *flag |= new != old;
    }
    for ((flag, new), old) in cov
        .ext5_cells
        .iter_mut()
        .flatten()
        .zip(dec_cdfs.ext_tx_dtt4.iter().flatten())
        .zip(cdfs0.ext_tx_dtt4.iter().flatten())
    {
        *flag |= new != old;
    }
    // TX_MODE_SELECT accounting, from the DECODER's records: distinct decoded
    // tx sizes, blocks whose coded depth left the max-rect default, and blocks
    // whose txb grid was genuinely multi-txb (the within-block interleave).
    if case.tx_mode == TxMode::Select {
        for b in &got.blocks {
            cov.tx_sizes_decoded[b.tx_size] = true;
            if b.tx_size != MAX_TXSIZE_RECT_LOOKUP[b.bsize] {
                cov.tx_depth_nonzero += 1;
            }
            if b.txbs.len() > 1 {
                cov.multi_txb_blocks += 1;
            }
            cov.max_txbs_in_block = cov.max_txbs_in_block.max(b.txbs.len());
        }
    }
    for ((flag, new), old) in cov
        .tx_cells
        .iter_mut()
        .flatten()
        .zip(dec_cdfs.tx_size.iter().flatten())
        .zip(cdfs0.tx_size.iter().flatten())
    {
        *flag |= new != old;
    }
    // Delta-q accounting from the DECODER's records: which effective
    // qindexes were decoded, which of them reconstructed real luma
    // coefficients (their per-block dequant rows demonstrably drove the
    // byte-identical recon), delta CDF adaptation, delta-lf carries.
    if cfg.delta_q_present {
        for b in &got.blocks {
            let q = b.info.current_qindex;
            cov.dq_qindex_seen[(q >> 6) as usize] |= 1u64 << (q & 63);
            if b.txbs.iter().any(|&(eob, _)| eob > 0) {
                cov.dq_qindex_recon_seen[(q >> 6) as usize] |= 1u64 << (q & 63);
            }
            if b.info.delta_lf_from_base != 0 || b.info.delta_lf.iter().any(|&x| x != 0) {
                cov.dlf_nonzero_carries += 1;
            }
        }
        cov.dq_cdf_adapted |= dec_cdfs.delta_q != cdfs0.delta_q;
        cov.dlf_cdf_adapted |= dec_cdfs.delta_lf != cdfs0.delta_lf;
        for (flag, (new, old)) in cov.dlf_multi_adapted.iter_mut().zip(
            dec_cdfs
                .delta_lf_multi
                .iter()
                .zip(cdfs0.delta_lf_multi.iter()),
        ) {
            *flag |= new != old;
        }
    }
}

#[test]
fn kf_luma_tile_roundtrips() {
    // (mi_rows, mi_cols): one 64x64 SB; 2x2 SBs; non-multiple-of-SB 80x96 px
    // (partial SBs on the right and bottom edges); 3x3 SBs (a fully-interior
    // superblock, exercising cross-SB top-right availability).
    let sizes = [(16, 16), (32, 32), (20, 24), (48, 48)];
    // The delta-q-off axes keep the original 432-tile sweep semantics; chroma
    // configs derive distinct-band per-plane deltas so U/V dequants stay
    // clearly distinct (plane-swap hard-fail, as the old independent random
    // dequants gave). The gate-off config carries a non-zero y_dc delta so
    // base_qindex = 0 is not C's coded_lossless.
    let configs = [
        // bd, mono, cdef, edge_off, fi, reduced, gate
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 2,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: false,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: false,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 0,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 10,
            monochrome: true,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 3,
            disable_edge_filter: true,
            enable_filter_intra: false,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: false,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 1,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: true,
            base_qindex_gt0: true,
            plane_deltas: false,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 0,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: false,
            plane_deltas: true,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 12,
            monochrome: false,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 2,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        // 4:2:0 — shared sub-8x8 chroma rules + 420 CfL subsampling.
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: false,
            ss_x: 1,
            ss_y: 1,
            cdef_bits: 1,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        // 4:2:0 high bit depth + reduced tx set (chroma tx-type demotion).
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 10,
            monochrome: false,
            ss_x: 1,
            ss_y: 1,
            cdef_bits: 2,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: true,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        // 4:2:2 — horizontal-only subsampling (tall shapes are non-conformant
        // for it and filtered out of the mirror's partition choices).
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: false,
            ss_x: 1,
            ss_y: 0,
            cdef_bits: 0,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 0,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        // ---- delta-q present: per-SB qindex deltas drive per-block dequant
        // recompute (decodeframe.c parse_decode_block). ----
        // res 1, monochrome: pure luma delta-q, finest step.
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 1,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 1,
            dlf: None,
            tx_mode: TxMode::Largest,
        },
        // res 4 + multi delta-lf (all 4 lf ids), 4:2:0 bd10: chroma dequant
        // rows recomputed with per-plane deltas folded in.
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 10,
            monochrome: false,
            ss_x: 1,
            ss_y: 1,
            cdef_bits: 2,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 4,
            dlf: Some((true, 2)),
            tx_mode: TxMode::Largest,
        },
        // res 8 + single from-base delta-lf, 4:4:4 bd12 reduced set.
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 12,
            monochrome: false,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 0,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: true,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 8,
            dlf: Some((false, 4)),
            tx_mode: TxMode::Largest,
        },
        // res 2 + multi delta-lf on monochrome (the FRAME_LF_COUNT - 2 arm).
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            ss_x: 0,
            ss_y: 0,
            cdef_bits: 3,
            disable_edge_filter: false,
            enable_filter_intra: false,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            plane_deltas: true,
            dq_res: 2,
            dlf: Some((true, 8)),
            tx_mode: TxMode::Largest,
        },
    ];
    let seeds: [u64; 6] = [
        0x0dec_0dea_11ce_0001,
        0x0dec_0dea_11ce_0002,
        0x0dec_0dea_11ce_0003,
        0x0dec_0dea_11ce_0004,
        0x0dec_0dea_11ce_0005,
        0x0dec_0dea_11ce_0006,
    ];

    let mut cov = Coverage::default();
    for c in &configs {
        for &(mi_rows, mi_cols) in &sizes {
            for &seed in &seeds {
                // Every config runs under BOTH frame tx modes: LARGEST keeps
                // the original 144-tile sweep green (no tx-size bits), SELECT
                // adds per-block tx-size signalling + real multi-txb grids.
                for tx_mode in [TxMode::Largest, TxMode::Select] {
                    let case = SweepCase {
                        mi_rows,
                        mi_cols,
                        tx_mode,
                        ..*c
                    };
                    run_roundtrip(&case, seed, &mut cov);
                }
            }
        }
    }

    // Coverage: the sweep must actually have exercised every mode-family path.
    for (m, &n) in cov.y_modes.iter().enumerate() {
        assert!(n > 0, "intra y mode {m} never exercised");
    }
    for p in [
        PARTITION_NONE,
        PARTITION_HORZ,
        PARTITION_VERT,
        PARTITION_SPLIT,
        4,
        5,
        6,
        7,
        8,
        9,
    ] {
        assert!(cov.partitions[p] > 0, "partition type {p} never exercised");
    }
    assert!(cov.fi_used > 0, "filter-intra never used");
    assert!(cov.angle_nonzero > 0, "no non-zero angle delta");
    assert!(cov.skip_blocks > 0, "no skip blocks");
    assert!(cov.eob_zero > 0, "no all-zero txbs");
    assert!(cov.eob_pos > 0, "no coded txbs");
    assert!(cov.cfl_uv_blocks > 0, "no UV CfL-mode blocks");
    assert!(cov.ext5_signaled > 0, "5-symbol ext-tx set never signalled");
    assert!(cov.ext7_signaled > 0, "7-symbol ext-tx set never signalled");
    assert!(cov.dct_only_txbs > 0, "no DCT-only txbs");
    assert!(
        cov.edge_clipped_txb_blocks > 0,
        "no frame-edge-clipped blocks"
    );

    // Chroma reconstruction coverage: CfL must actually predict in every
    // subsampling, with varied joint signs and alpha indices; non-CfL UV modes
    // and the sub-8x8 shared-chroma path must be exercised; chroma coefficients
    // must hit both the coded and the all-zero paths, non-DCT chroma transform
    // types, and non-zero UV angle deltas.
    let js_n = cov.cfl_js_seen.iter().filter(|&&x| x).count();
    let alpha_n: u32 = cov.cfl_alpha_idx_seen.iter().map(|w| w.count_ones()).sum();
    eprintln!(
        "chroma: cfl 420 {} / 422 {} / 444 {}, non-cfl uv {}, shared sub-8x8 {}, \
         js {js_n}/8, alpha idx {alpha_n}/256, uv eob (pos {}, zero {}), \
         uv non-DCT {}, uv angle!=0 {}",
        cov.cfl_predicted_420,
        cov.cfl_predicted_422,
        cov.cfl_predicted_444,
        cov.uv_non_cfl_blocks,
        cov.shared_chroma_blocks,
        cov.uv_eob_pos,
        cov.uv_eob_zero,
        cov.uv_non_dct_txbs,
        cov.uv_angle_nonzero,
    );
    assert!(cov.cfl_predicted_420 > 0, "no CfL predictions at 4:2:0");
    assert!(cov.cfl_predicted_422 > 0, "no CfL predictions at 4:2:2");
    assert!(cov.cfl_predicted_444 > 0, "no CfL predictions at 4:4:4");
    assert!(cov.uv_non_cfl_blocks > 0, "no non-CfL UV predictions");
    assert!(
        cov.shared_chroma_blocks > 0,
        "no sub-8x8 shared-chroma reference blocks"
    );
    assert!(js_n == 8, "cfl joint-sign diversity too low: {js_n}/8");
    assert!(
        alpha_n >= 32,
        "cfl alpha-index diversity too low: {alpha_n}/256"
    );
    assert!(cov.uv_eob_pos > 0, "no coded chroma txbs");
    assert!(cov.uv_eob_zero > 0, "no all-zero chroma txbs");
    assert!(cov.uv_non_dct_txbs > 0, "no non-DCT chroma transform types");
    assert!(cov.uv_angle_nonzero > 0, "no non-zero UV angle deltas");

    // FRAME_CONTEXT selection diversity: the per-context arrays must have been
    // exercised across many DISTINCT instances — a regression back to one
    // shared CDF per symbol collapses these counts to 1.
    let kf_y_n: usize = cov.kf_y_cells.iter().flatten().filter(|&&x| x).count();
    let skip_n = cov.skip_ctxs.iter().filter(|&&x| x).count();
    let angle_n = cov.angle_insts.iter().filter(|&&x| x).count();
    let uv_n: usize = cov.uv_insts.iter().flatten().filter(|&&x| x).count();
    let fi_n = cov.fi_bsizes.iter().filter(|&&x| x).count();
    let ext7_n: usize = cov.ext7_cells.iter().flatten().filter(|&&x| x).count();
    let ext5_n: usize = cov.ext5_cells.iter().flatten().filter(|&&x| x).count();
    eprintln!(
        "ctx diversity: kf_y {kf_y_n}/25 skip {skip_n}/3 angle {angle_n}/8 \
         uv {uv_n}/26 fi {fi_n}/22 ext7 {ext7_n}/52 ext5 {ext5_n}/52"
    );
    assert!(kf_y_n >= 20, "kf_y context diversity too low: {kf_y_n}/25");
    assert!(skip_n == 3, "skip context diversity too low: {skip_n}/3");
    assert!(
        angle_n == 8,
        "angle_delta instance diversity too low: {angle_n}/8"
    );
    assert!(uv_n >= 18, "uv_mode instance diversity too low: {uv_n}/26");
    assert!(fi_n >= 4, "filter_intra bsize diversity too low: {fi_n}/22");
    assert!(
        ext7_n >= 10,
        "ext-tx 7-symbol cell diversity too low: {ext7_n}/52"
    );
    assert!(
        ext5_n >= 10,
        "ext-tx 5-symbol cell diversity too low: {ext5_n}/52"
    );

    // TX_MODE_SELECT: the sweep must have decoded genuinely varied tx sizes,
    // exercised the within-block multi-txb interleave, and adapted the
    // (category, context)-selected tx_size_cdf instances.
    let tx_distinct = cov.tx_sizes_decoded.iter().filter(|&&x| x).count();
    let tx_cells_n: usize = cov.tx_cells.iter().flatten().filter(|&&x| x).count();
    eprintln!(
        "tx-size: {tx_distinct}/19 distinct sizes decoded, {} non-max-depth blocks, \
         {} multi-txb blocks (max {} txbs/block), tx_size_cdf cells {tx_cells_n}/12",
        cov.tx_depth_nonzero, cov.multi_txb_blocks, cov.max_txbs_in_block
    );
    // Floors set from the deterministic sweep (observed: 19/19 distinct,
    // 4210 non-max-depth, 4210 multi-txb, max grid 16, 12/12 cells) with
    // headroom only where a minor sweep edit could legitimately shave counts.
    assert!(
        tx_distinct >= 16,
        "too few distinct tx sizes decoded under SELECT: {tx_distinct}/19"
    );
    assert!(
        cov.tx_depth_nonzero >= 1000,
        "too few non-max tx depths decoded (SELECT barely varied): {}",
        cov.tx_depth_nonzero
    );
    assert!(
        cov.multi_txb_blocks >= 1000,
        "too few within-block multi-txb grids: {}",
        cov.multi_txb_blocks
    );
    // 16 = the structural max for this scope: a 64x64 block at depth 2
    // (TX_16X16) is a 4x4 txb grid; the sweep must reach it.
    assert!(
        cov.max_txbs_in_block >= 16,
        "largest decoded txb grid too small: {} txbs",
        cov.max_txbs_in_block
    );
    assert!(
        tx_cells_n == 12,
        "every tx_size_cdf (cat, ctx) instance must adapt: {tx_cells_n}/12"
    );

    // Delta-q: the sweep must have decoded genuinely varied effective
    // qindexes (floor >= 3 distinct, per the chunk spec — observed far more),
    // and blocks with DIFFERENT effective qindexes must have reconstructed
    // real luma coefficients (their recomputed dequant rows drove the
    // byte-identical reconstruction, so a dequant-recompute bug cannot hide
    // behind skip/eob-0 blocks). Both delta signs, the |reduced| >= 3
    // exp-Golomb remainder path, and the sb-sized-skip gate-out arm must
    // occur; the delta-q/delta-lf CDFs must adapt (they are also asserted in
    // lockstep per tile by assert_fc_eq); multi-mode must adapt every lf id
    // and non-zero delta-lf carries must reach the block records.
    let dq_distinct: u32 = cov.dq_qindex_seen.iter().map(|w| w.count_ones()).sum();
    let dq_recon_distinct: u32 = cov
        .dq_qindex_recon_seen
        .iter()
        .map(|w| w.count_ones())
        .sum();
    let dlf_ids = cov.dlf_multi_adapted.iter().filter(|&&x| x).count();
    eprintln!(
        "delta-q: {dq_distinct} distinct effective qindexes ({dq_recon_distinct} with \
         reconstructed luma coeffs), reduced-delta pos {} / neg {} / golomb {}, \
         sb-skip-gated {}, dq cdf adapted {}, dlf cdf adapted {} (multi ids {dlf_ids}/4), \
         nonzero dlf carries {}",
        cov.dq_pos,
        cov.dq_neg,
        cov.dq_golomb,
        cov.dq_gated_sb_skip,
        cov.dq_cdf_adapted,
        cov.dlf_cdf_adapted,
        cov.dlf_nonzero_carries,
    );
    assert!(
        dq_distinct >= 3,
        "too few distinct effective qindexes decoded: {dq_distinct}"
    );
    assert!(
        dq_recon_distinct >= 3,
        "too few distinct effective qindexes among blocks with reconstructed \
         luma coefficients: {dq_recon_distinct}"
    );
    assert!(
        cov.dq_pos > 0 && cov.dq_neg > 0,
        "both delta-q signs must be written (pos {}, neg {})",
        cov.dq_pos,
        cov.dq_neg
    );
    assert!(
        cov.dq_golomb > 0,
        "no |reduced| >= 3 delta-q (exp-Golomb remainder path never exercised)"
    );
    assert!(
        cov.dq_gated_sb_skip > 0,
        "no sb-sized skip block skipped the delta-q read (gate arm never exercised)"
    );
    assert!(cov.dq_cdf_adapted, "delta_q cdf never adapted");
    assert!(cov.dlf_cdf_adapted, "single delta_lf cdf never adapted");
    assert!(
        dlf_ids == 4,
        "every delta_lf_multi id must adapt (4:2:0 multi codes all 4): {dlf_ids}/4"
    );
    assert!(
        cov.dlf_nonzero_carries > 0,
        "no non-zero delta-lf carries reached the block records"
    );
}
