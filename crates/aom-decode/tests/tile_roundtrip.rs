//! Full-tile encode→decode roundtrip for the KEY-frame luma decode driver.
//!
//! A mirror mini-encoder performs the identical tile walk with the write-side
//! counterparts (`write_partition` / `write_mb_modes_kf` +
//! `write_filter_intra_mode_info` / `write_coeffs_txb_full`) and its own
//! reconstruction feedback loop: per txb it predicts from *its* recon-so-far
//! (same `intra_avail` + `predict_intra_high`), computes the residual against a
//! synthetic source, forward-transforms + quantizes (`xform_quant`,
//! `QuantKind::B`, `invert_quant`-derived params), writes the coefficients, and
//! reconstructs through the same `reconstruct_txb`. Because every write-side
//! piece is byte-identical to C libaom, a clean roundtrip (byte-identical
//! reconstruction planes + lockstep CDF state + per-leaf mode-info equality)
//! pins the decode driver to the C decoder.
//!
//! Encoder and decoder reconstruction planes start from *different* fill values:
//! a conformant walk never reads an unwritten pixel, so any neighbour-
//! availability bug becomes a hard plane mismatch instead of silently agreeing.
//!
//! Sweep: 3 frame sizes (one SB / 2x2 SBs / non-multiple-of-SB 80x96 px with
//! partial superblocks) × 6 configs (monochrome + 4:4:4, bd 8/10/12, filter
//! intra on/off, intra edge filter on/off, reduced tx set, tx-type gate off,
//! cdef bits 0..3) × 3 seeds, with pseudo-random partition trees over all 10
//! partition types, all 13 intra modes, angle deltas, filter-intra, and skip
//! blocks; coverage of each is asserted at the end.

use aom_decode::{
    ANGLE_STEP, BLOCK_8X8, BLOCK_64X64, BLOCK_SIZE_HIGH, BLOCK_SIZE_WIDE, DecodedBlockKf,
    KfTileCdfs, KfTileConfig, MAX_TXSIZE_RECT_LOOKUP, MI_SIZE_HIGH, MI_SIZE_WIDE, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, TX_SIZE_HIGH, TX_SIZE_HIGH_UNIT, TX_SIZE_WIDE,
    TX_SIZE_WIDE_UNIT, decode_tile_kf, filter_intra_allowed, intra_ext_tx_cdf, max_block_units,
};
use aom_encode::{QuantKind, QuantParams, xform_quant};
use aom_entropy::dec::OdEcDec;
use aom_entropy::enc::OdEcEnc;
use aom_entropy::partition::{
    KfBlockState, KfCdfs, MbModeInfoKf, get_partition_subsize, get_uv_mode, intra_avail,
    is_cfl_allowed, is_directional_mode, partition_cdf_length, partition_plane_context,
    update_ext_partition_context, use_angle_delta, write_filter_intra_mode_info, write_mb_modes_kf,
    write_partition,
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

fn mk_kf_cdfs(rng: &mut Rng) -> KfCdfs {
    let mut c = KfCdfs {
        seg: [0; 9],
        skip: [0; 3],
        delta_q: [0; 5],
        delta_lf_multi: [[0; 5]; 4],
        delta_lf: [0; 5],
        intrabc: [0; 3],
        ndvc_joints: [0; 5],
        ndvc_comp0: mk_comp(rng),
        ndvc_comp1: mk_comp(rng),
        y_mode: [0; 14],
        y_angle: [0; 8],
        uv_mode: [0; 15],
        cfl_sign: [0; 9],
        cfl_alpha: [[0; 17]; 6],
        uv_angle: [0; 8],
        pal_y_mode: [0; 3],
        pal_y_size: [0; 8],
        pal_uv_mode: [0; 3],
        pal_uv_size: [0; 8],
        fi_use: [0; 3],
        fi_mode: [0; 6],
    };
    mk_ns_cdf(rng, 8, &mut c.seg);
    mk_ns_cdf(rng, 2, &mut c.skip);
    mk_ns_cdf(rng, 4, &mut c.delta_q);
    for m in c.delta_lf_multi.iter_mut() {
        mk_ns_cdf(rng, 4, m);
    }
    mk_ns_cdf(rng, 4, &mut c.delta_lf);
    mk_ns_cdf(rng, 2, &mut c.intrabc);
    mk_ns_cdf(rng, 4, &mut c.ndvc_joints);
    mk_ns_cdf(rng, 13, &mut c.y_mode);
    mk_ns_cdf(rng, 7, &mut c.y_angle);
    mk_ns_cdf(rng, 14, &mut c.uv_mode); // scratch slot; real ones live in KfTileCdfs
    mk_ns_cdf(rng, 8, &mut c.cfl_sign);
    for a in c.cfl_alpha.iter_mut() {
        mk_ns_cdf(rng, 16, a);
    }
    mk_ns_cdf(rng, 7, &mut c.uv_angle);
    mk_ns_cdf(rng, 2, &mut c.pal_y_mode);
    mk_ns_cdf(rng, 7, &mut c.pal_y_size);
    mk_ns_cdf(rng, 2, &mut c.pal_uv_mode);
    mk_ns_cdf(rng, 7, &mut c.pal_uv_size);
    mk_ns_cdf(rng, 2, &mut c.fi_use);
    mk_ns_cdf(rng, 5, &mut c.fi_mode);
    c
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

fn mk_tile_cdfs(rng: &mut Rng) -> KfTileCdfs {
    let mut uv_cfl = [0u16; 15];
    let mut uv_nocfl = [0u16; 15];
    mk_ns_cdf(rng, 14, &mut uv_cfl);
    mk_ns_cdf(rng, 13, &mut uv_nocfl[..14]);
    let mut ext5 = [0u16; 6];
    let mut ext7 = [0u16; 8];
    mk_ns_cdf(rng, 5, &mut ext5);
    mk_ns_cdf(rng, 7, &mut ext7);
    let mut partition = [[0u16; 11]; 20];
    for (c, slot) in partition.iter_mut().enumerate() {
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
    KfTileCdfs {
        kf: mk_kf_cdfs(rng),
        uv_mode_cfl: uv_cfl,
        uv_mode_nocfl: uv_nocfl,
        coeff: mk_coeff_arena(rng),
        ext_tx_dtt4_idtx: ext5,
        ext_tx_dtt4_idtx_1ddct: ext7,
        partition,
    }
}

/// KfCdfs has no PartialEq (public-API discipline); compare field by field so a
/// mismatch names the desynced symbol.
fn assert_kf_cdfs_eq(e: &KfCdfs, d: &KfCdfs, what: &str) {
    assert_eq!(e.seg, d.seg, "{what}: seg cdf");
    assert_eq!(e.skip, d.skip, "{what}: skip cdf");
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
    assert_eq!(e.y_mode, d.y_mode, "{what}: y_mode cdf");
    assert_eq!(e.y_angle, d.y_angle, "{what}: y_angle cdf");
    assert_eq!(e.uv_mode, d.uv_mode, "{what}: uv_mode scratch slot");
    assert_eq!(e.cfl_sign, d.cfl_sign, "{what}: cfl_sign cdf");
    assert_eq!(e.cfl_alpha, d.cfl_alpha, "{what}: cfl_alpha cdf");
    assert_eq!(e.uv_angle, d.uv_angle, "{what}: uv_angle cdf");
    assert_eq!(e.pal_y_mode, d.pal_y_mode, "{what}: pal_y_mode cdf");
    assert_eq!(e.pal_y_size, d.pal_y_size, "{what}: pal_y_size cdf");
    assert_eq!(e.pal_uv_mode, d.pal_uv_mode, "{what}: pal_uv_mode cdf");
    assert_eq!(e.pal_uv_size, d.pal_uv_size, "{what}: pal_uv_size cdf");
    assert_eq!(e.fi_use, d.fi_use, "{what}: fi_use cdf");
    assert_eq!(e.fi_mode, d.fi_mode, "{what}: fi_mode cdf");
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
    smooth: Vec<u8>,
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
            smooth: vec![0; (cfg.mi_rows * cfg.mi_cols) as usize],
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

    fn filt_type(&self, mi_row: i32, mi_col: i32, up: bool, left: bool) -> i32 {
        let cols = self.cfg.mi_cols;
        let ab = up && self.smooth[((mi_row - 1) * cols + mi_col) as usize] != 0;
        let le = left && self.smooth[(mi_row * cols + mi_col - 1) as usize] != 0;
        (ab || le) as i32
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
        cdfs: &mut KfTileCdfs,
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
        let fi_allowed = filter_intra_allowed(cfg.enable_filter_intra, bsize, y_mode);
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

        // --- write the mode info (write_mb_modes_kf + filter-intra follow-up) ---
        self.st.mi_row = mi_row;
        self.st.mi_col = mi_col;
        self.st.bsize = bsize;
        self.st.cfl_allowed = cfl_allowed;
        self.st.mb_to_top_edge = -(mi_row * 32);
        self.st.has_above = up_available;
        self.st.has_left = left_available;
        let saved_uv = cdfs.kf.uv_mode;
        cdfs.kf.uv_mode = if cfl_allowed {
            cdfs.uv_mode_cfl
        } else {
            cdfs.uv_mode_nocfl
        };
        write_mb_modes_kf(enc, &info, &mut cdfs.kf, &mut self.st);
        if cfl_allowed {
            cdfs.uv_mode_cfl = cdfs.kf.uv_mode;
        } else {
            cdfs.uv_mode_nocfl = cdfs.kf.uv_mode;
        }
        cdfs.kf.uv_mode = saved_uv;
        write_filter_intra_mode_info(
            enc,
            &mut cdfs.kf.fi_use,
            &mut cdfs.kf.fi_mode,
            fi_allowed,
            use_fi,
            fi_mode,
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

        // --- skip blocks reset their entropy-context footprint ---
        let bw = MI_SIZE_WIDE[bsize] as usize;
        let bh = MI_SIZE_HIGH[bsize] as usize;
        if skip != 0 {
            let a0 = mi_col as usize;
            self.above_e[a0..a0 + bw].fill(0);
            let l0 = (mi_row & 31) as usize;
            self.left_e[l0..l0 + bh].fill(0);
        }

        // --- per-txb: predict -> residual -> quantize -> write -> reconstruct ---
        let tx_size = MAX_TXSIZE_RECT_LOOKUP[bsize];
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
        let filt_type = self.filt_type(mi_row, mi_col, up_available, left_available);
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
                        &mut cdfs.ext_tx_dtt4_idtx,
                        &mut cdfs.ext_tx_dtt4_idtx_1ddct,
                        tx_size,
                        cfg.reduced_tx_set,
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

        // smooth-mode grid stamp (frame-cropped), for later blocks' filt_type
        let sm = ((9..=11).contains(&y_mode)) as u8;
        let x_mis = MI_SIZE_WIDE[bsize].min(cfg.mi_cols - mi_col);
        let y_mis = MI_SIZE_HIGH[bsize].min(cfg.mi_rows - mi_row);
        for r in 0..y_mis {
            let base = ((mi_row + r) * cfg.mi_cols + mi_col) as usize;
            self.smooth[base..base + x_mis as usize].fill(sm);
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
        cdfs: &mut KfTileCdfs,
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
        cdfs: &mut KfTileCdfs,
        rng: &mut Rng,
        cov: &mut Coverage,
    ) {
        let mut mi_row = 0;
        while mi_row < self.cfg.mi_rows {
            self.left_e = [0; 32];
            self.left_p = [0; 32];
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
    let cdfs0 = mk_tile_cdfs(&mut rng);

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
        "case mi={}x{} bd={} mono={} cdef={} fi={} reduced={} gate={} seed={seed:#x}",
        case.mi_rows,
        case.mi_cols,
        case.bd,
        case.monochrome,
        case.cdef_bits,
        case.enable_filter_intra,
        case.reduced_tx_set,
        case.base_qindex_gt0,
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
    // (c) every CDF in lockstep
    assert_kf_cdfs_eq(&enc_cdfs.kf, &dec_cdfs.kf, &what);
    assert_eq!(
        enc_cdfs.uv_mode_cfl, dec_cdfs.uv_mode_cfl,
        "{what}: uv cfl cdf"
    );
    assert_eq!(
        enc_cdfs.uv_mode_nocfl, dec_cdfs.uv_mode_nocfl,
        "{what}: uv nocfl cdf"
    );
    assert_eq!(enc_cdfs.coeff, dec_cdfs.coeff, "{what}: coeff arena");
    assert_eq!(
        enc_cdfs.ext_tx_dtt4_idtx, dec_cdfs.ext_tx_dtt4_idtx,
        "{what}: ext-tx 5 cdf"
    );
    assert_eq!(
        enc_cdfs.ext_tx_dtt4_idtx_1ddct, dec_cdfs.ext_tx_dtt4_idtx_1ddct,
        "{what}: ext-tx 7 cdf"
    );
    assert_eq!(
        enc_cdfs.partition, dec_cdfs.partition,
        "{what}: partition arena"
    );
}

#[test]
fn kf_luma_tile_roundtrips() {
    // (mi_rows, mi_cols): one 64x64 SB; 2x2 SBs; non-multiple-of-SB 80x96 px
    // (partial SBs on the right and bottom edges).
    let sizes = [(16, 16), (32, 32), (20, 24)];
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
        },
    ];
    let seeds: [u64; 3] = [
        0x0dec_0dea_11ce_0001,
        0x0dec_0dea_11ce_0002,
        0x0dec_0dea_11ce_0003,
    ];

    let mut cov = Coverage::default();
    for c in &configs {
        for &(mi_rows, mi_cols) in &sizes {
            for &seed in &seeds {
                let case = SweepCase {
                    mi_rows,
                    mi_cols,
                    ..*c
                };
                run_roundtrip(&case, seed, &mut cov);
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
}
