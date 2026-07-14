//! Full-tile encode→decode roundtrip for the KEY-frame luma decode driver.
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
//! partial superblocks / 3x3 SBs with a fully-interior SB) × 6 configs
//! (monochrome + 4:4:4, bd 8/10/12, filter intra on/off, intra edge filter
//! on/off, reduced tx set, tx-type gate off,
//! cdef bits 0..3) × 6 seeds × 2 frame tx modes (`TX_MODE_LARGEST` — the
//! original 144-tile sweep, no tx-size bits — and `TX_MODE_SELECT`, where the
//! mirror codes a pseudo-random tx-size depth per signalling block through
//! `write_selected_tx_size` on the `get_tx_size_context`-selected CDF and the
//! decoder must reproduce it, driving real multi-txb grids whose later txbs
//! predict from earlier txbs' reconstruction *inside* the block), with
//! pseudo-random partition trees over all 10 partition types, all 13 intra
//! modes, angle deltas, filter-intra, and skip blocks; coverage of each is
//! asserted at the end (including distinct-tx-size, multi-txb-grid, and
//! tx_size_cdf cell-diversity floors).

use aom_decode::{
    ANGLE_STEP, BLOCK_8X8, BLOCK_64X64, BLOCK_SIZE_HIGH, BLOCK_SIZE_WIDE, DecodedBlockKf,
    KfTileConfig, MAX_TXSIZE_RECT_LOOKUP, MI_SIZE_HIGH, MI_SIZE_WIDE, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, TX_SIZE_HIGH, TX_SIZE_HIGH_UNIT, TX_SIZE_WIDE,
    TX_SIZE_WIDE_UNIT, decode_tile_kf, intra_ext_tx_cdf, max_block_units,
};
use aom_encode::{QuantKind, QuantParams, xform_quant};
use aom_entropy::dec::OdEcDec;
use aom_entropy::enc::OdEcEnc;
use aom_entropy::partition::{
    KfBlockState, KfFrameContext, MbModeInfoKf, MiNbrKf, TXFM_CTX_INIT, TxMode,
    bsize_to_max_depth, bsize_to_tx_size_cat, depth_to_tx_size, filter_intra_allowed,
    get_partition_subsize, get_tx_size_context, get_uv_mode, intra_avail, is_cfl_allowed,
    is_directional_mode, partition_cdf_length, partition_plane_context, set_txfm_ctxs,
    tx_size_from_tx_mode, tx_size_to_depth, update_ext_partition_context, use_angle_delta,
    write_mb_modes_kf_fc, write_partition, write_selected_tx_size,
};
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
    assert_eq!(e.palette_y_mode, d.palette_y_mode, "{what}: palette_y_mode cdf");
    assert_eq!(e.palette_uv_mode, d.palette_uv_mode, "{what}: palette_uv_mode cdf");
    assert_eq!(e.palette_y_size, d.palette_y_size, "{what}: palette_y_size cdf");
    assert_eq!(e.palette_uv_size, d.palette_uv_size, "{what}: palette_uv_size cdf");
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
    // TX_MODE_SELECT accounting (from the DECODER's output): which of the 19
    // tx sizes were actually decoded, how many blocks decoded a non-max depth,
    // and how many blocks ran a real multi-txb grid (>1 txb).
    tx_sizes_decoded: [bool; 19],
    tx_depth_nonzero: usize,
    multi_txb_blocks: usize,
    max_txbs_in_block: usize,
    // tx_size_cdf (cat, ctx) instances that adapted anywhere in the sweep.
    tx_cells: [[bool; 3]; 4],
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
    recon: Vec<u16>,
    stride: usize,
    above_e: Vec<i8>,
    left_e: [i8; 32],
    above_p: Vec<i8>,
    left_p: [i8; 32],
    /// Txfm-context byte arrays (`above_txfm_context`/`left_txfm_context`),
    /// mirroring the decoder's: init 64, stamped by `set_txfm_ctxs` per block.
    above_t: Vec<u8>,
    left_t: [u8; 32],
    /// Per-mi mode-info grid — the encoder's own `xd->above_mbmi/left_mbmi`
    /// source for every context selection (mirrors the decoder's grid).
    mi: Vec<MiNbrKf>,
    st: KfBlockState,
    quant: [i16; 2],
    quant_shift: [i16; 2],
    round: [i16; 2],
    zbin: [i16; 2],
    sb_cdef_strength: i32,
    sb_cdef_done: bool,
    tree: Vec<i8>,
    blocks: Vec<DecodedBlockKf>,
}

impl<'a> Mirror<'a> {
    fn new(cfg: &'a KfTileConfig, src: &'a [u16], stride: usize, recon_init: u16) -> Self {
        let aligned_rows = (cfg.mi_rows as usize).div_ceil(16) * 16;
        let (q0, s0) = invert_quant(cfg.dequant[0] as i32);
        let (q1, s1) = invert_quant(cfg.dequant[1] as i32);
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
            dq_present: false,
            dlf_present: false,
            dlf_multi: false,
            num_planes: if cfg.monochrome { 1 } else { 3 },
            dq_res: 1,
            dlf_res: 1,
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
            current_base_qindex: 0,
            xd_delta_lf: [0; 4],
            xd_delta_lf_from_base: 0,
        };
        Mirror {
            cfg,
            src,
            recon: vec![recon_init; stride * aligned_rows * 4],
            stride,
            above_e: vec![0; stride / 4],
            left_e: [0; 32],
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
            st,
            quant: [q0, q1],
            quant_shift: [s0, s1],
            round: [cfg.dequant[0] / 8 + 1, cfg.dequant[1] / 8 + 1],
            zbin: [cfg.dequant[0] / 2 + 1, cfg.dequant[1] / 2 + 1],
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
        cul: i8,
        mi_row: i32,
        mi_col: i32,
        blk_row: usize,
        blk_col: usize,
        txw: usize,
        txh: usize,
        blocks_wide: usize,
        blocks_high: usize,
        mb_to_right_edge: i32,
        mb_to_bottom_edge: i32,
    ) {
        let a0 = mi_col as usize + blk_col;
        if cul != 0 && mb_to_right_edge < 0 {
            let n = txw.min(blocks_wide - blk_col);
            self.above_e[a0..a0 + n].fill(cul);
            self.above_e[a0 + n..a0 + txw].fill(0);
        } else {
            self.above_e[a0..a0 + txw].fill(cul);
        }
        let l0 = (mi_row & 31) as usize + blk_row;
        if cul != 0 && mb_to_bottom_edge < 0 {
            let n = txh.min(blocks_high - blk_row);
            self.left_e[l0..l0 + n].fill(cul);
            self.left_e[l0 + n..l0 + txh].fill(0);
        } else {
            self.left_e[l0..l0 + txh].fill(cul);
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
        let cfl_allowed = !cfg.monochrome && is_cfl_allowed(bsize, false, 0, 0);
        let (above, left) = self.neighbours(mi_row, mi_col);

        // --- choose the block's mode info ---
        let y_mode = (rng.next() % 13) as i32;
        let angle_y = if use_angle_delta(bsize) && is_directional_mode(y_mode) {
            (rng.next() % 7) as i32 - 3
        } else {
            0
        };
        let (uv_mode, cfl_idx, js, angle_uv) = if !cfg.monochrome {
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
        let info = MbModeInfoKf {
            segment_id: 0,
            skip,
            cdef_strength: self.sb_cdef_strength,
            current_qindex: 0,
            delta_lf: [0; 4],
            delta_lf_from_base: 0,
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

        // --- skip blocks reset their entropy-context footprint ---
        if skip != 0 {
            let a0 = mi_col as usize;
            self.above_e[a0..a0 + bw].fill(0);
            let l0 = (mi_row & 31) as usize;
            self.left_e[l0..l0 + bh].fill(0);
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
        let is_smooth =
            |m: Option<MiNbrKf>| m.is_some_and(|n| (9..=11).contains(&n.y_mode));
        let filt_type = (is_smooth(above) || is_smooth(left)) as i32;
        let signal_gate = cfg.base_qindex_gt0 && skip == 0;
        let set_type = ext_tx_set_type(tx_size, false, cfg.reduced_tx_set);
        let mut scratch = vec![0u16; txwpx * txhpx];
        let mut residual = vec![0i16; txwpx * txhpx];
        let mut txbs = Vec::new();
        let quant = self.quant;
        let quant_shift = self.quant_shift;
        let round = self.round;
        let zbin = self.zbin;
        let dequant = cfg.dequant;
        let qp = QuantParams {
            zbin: &zbin,
            round: &round,
            quant: &quant,
            quant_shift: &quant_shift,
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
                    let (tsc, dsc) =
                        get_txb_ctx(bsize, tx_size, 0, &self.above_e[a0..], &self.left_e[l0..]);
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
                        r.txb_entropy_ctx as i8,
                        mi_row,
                        mi_col,
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
                            cfg.dequant,
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
                blk_col += txw;
            }
            blk_row += txh;
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
        });
    }

    /// Choose a legal partition for the node (mirrors the C decoder's edge rules:
    /// forced NONE below 8x8, HORZ/SPLIT at a bottom edge, VERT/SPLIT at a right
    /// edge, forced SPLIT off both, the full set in frame).
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
            (false, true) => [PARTITION_HORZ, PARTITION_SPLIT][(rng.next() & 1) as usize],
            (true, false) => [PARTITION_VERT, PARTITION_SPLIT][(rng.next() & 1) as usize],
            (true, true) => {
                let n = partition_cdf_length(bsize);
                // bias splits a little so the sweep reaches small blocks
                if bsize > BLOCK_8X8 && rng.next() % 100 < 30 {
                    PARTITION_SPLIT
                } else {
                    (rng.next() % n as u64) as usize
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
            self.left_e = [0; 32];
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
    cdef_bits: u32,
    disable_edge_filter: bool,
    enable_filter_intra: bool,
    reduced_tx_set: bool,
    base_qindex_gt0: bool,
    tx_mode: TxMode,
}

fn run_roundtrip(case: &SweepCase, seed: u64, cov: &mut Coverage) {
    let mut rng = Rng(seed);
    let dequant = [rng.range(4, 800) as i16, rng.range(4, 800) as i16];
    let cfg = KfTileConfig {
        mi_rows: case.mi_rows,
        mi_cols: case.mi_cols,
        bd: case.bd,
        monochrome: case.monochrome,
        cdef_bits: case.cdef_bits,
        disable_edge_filter: case.disable_edge_filter,
        enable_filter_intra: case.enable_filter_intra,
        tx_mode: case.tx_mode,
        reduced_tx_set: case.reduced_tx_set,
        base_qindex_gt0: case.base_qindex_gt0,
        dequant,
    };
    let aligned_cols = (cfg.mi_cols as usize).div_ceil(16) * 16;
    let aligned_rows = (cfg.mi_rows as usize).div_ceil(16) * 16;
    let stride = aligned_cols * 4;
    let mask = (1u64 << cfg.bd) - 1;
    let mut src: Vec<u16> = (0..stride * aligned_rows * 4)
        .map(|_| (rng.next() & mask) as u16)
        .collect();
    // Carve some flat 64x64 regions: blocks there predict near-perfectly, so the
    // quantizer produces genuine all-zero txbs (the txb_skip=1 decode path).
    for sbr in 0..aligned_rows / 16 {
        for sbc in 0..aligned_cols / 16 {
            if rng.next().is_multiple_of(3) {
                let v = (rng.next() & mask) as u16;
                for r in 0..64 {
                    let base = (sbr * 64 + r) * stride + sbc * 64;
                    src[base..base + 64].fill(v);
                }
            }
        }
    }
    let cdfs0 = mk_frame_ctx(&mut rng);

    // encode (mirror), recon initialised to 0
    let mut enc_cdfs = cdfs0.clone();
    let mut mirror = Mirror::new(&cfg, &src, stride, 0);
    let mut enc = OdEcEnc::new();
    mirror.encode_tile(&mut enc, &mut enc_cdfs, &mut rng, cov);
    let bytes = enc.done().to_vec();

    // decode, recon initialised to the max pixel value (divergent on purpose)
    let mut dec_cdfs = cdfs0.clone();
    let mut dec = OdEcDec::new(&bytes);
    let got = decode_tile_kf(&mut dec, &cfg, &mut dec_cdfs, mask as u16);

    let what = format!(
        "case mi={}x{} bd={} mono={} cdef={} fi={} reduced={} gate={} tx={:?} seed={seed:#x}",
        case.mi_rows,
        case.mi_cols,
        case.bd,
        case.monochrome,
        case.cdef_bits,
        case.enable_filter_intra,
        case.reduced_tx_set,
        case.base_qindex_gt0,
        case.tx_mode,
    );
    // (a) partition tree + per-leaf decode records (mode info, per-txb eob/tx_type)
    assert_eq!(got.tree, mirror.tree, "{what}: partition tree");
    assert_eq!(got.blocks.len(), mirror.blocks.len(), "{what}: leaf count");
    for (i, (g, w)) in got.blocks.iter().zip(&mirror.blocks).enumerate() {
        assert_eq!(g, w, "{what}: block {i}");
    }
    // (b) byte-identical reconstruction over the frame crop
    assert_eq!(got.stride, stride, "{what}: stride");
    for row in 0..got.height {
        assert_eq!(
            got.recon[row * stride..row * stride + got.width],
            mirror.recon[row * stride..row * stride + got.width],
            "{what}: recon row {row}"
        );
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
    for ((flag, new), old) in cov.skip_ctxs.iter_mut().zip(&dec_cdfs.skip).zip(&cdfs0.skip) {
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
}

#[test]
fn kf_luma_tile_roundtrips() {
    // (mi_rows, mi_cols): one 64x64 SB; 2x2 SBs; non-multiple-of-SB 80x96 px
    // (partial SBs on the right and bottom edges); 3x3 SBs (a fully-interior
    // superblock, exercising cross-SB top-right availability).
    let sizes = [(16, 16), (32, 32), (20, 24), (48, 48)];
    let configs = [
        // bd, mono, cdef, edge_off, fi, reduced, gate
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            cdef_bits: 2,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: false,
            cdef_bits: 0,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 10,
            monochrome: true,
            cdef_bits: 3,
            disable_edge_filter: true,
            enable_filter_intra: false,
            reduced_tx_set: false,
            base_qindex_gt0: true,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            cdef_bits: 1,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: true,
            base_qindex_gt0: true,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 8,
            monochrome: true,
            cdef_bits: 0,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: false,
            tx_mode: TxMode::Largest,
        },
        SweepCase {
            mi_rows: 0,
            mi_cols: 0,
            bd: 12,
            monochrome: false,
            cdef_bits: 2,
            disable_edge_filter: false,
            enable_filter_intra: true,
            reduced_tx_set: false,
            base_qindex_gt0: true,
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
    assert!(angle_n == 8, "angle_delta instance diversity too low: {angle_n}/8");
    assert!(uv_n >= 18, "uv_mode instance diversity too low: {uv_n}/26");
    assert!(fi_n >= 4, "filter_intra bsize diversity too low: {fi_n}/22");
    assert!(ext7_n >= 10, "ext-tx 7-symbol cell diversity too low: {ext7_n}/52");
    assert!(ext5_n >= 10, "ext-tx 5-symbol cell diversity too low: {ext5_n}/52");

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
}
