//! KB-12 — speed >= 8 nonrd pickmode (allintra KEY): `av1_nonrd_pick_intra_mode`
//! (nonrd_pickmode.c:1582) + its `av1_block_yrd` Hadamard estimator
//! (nonrd_opt.c:126) + the LP kernel set they stand on.
//!
//! HANDOFF STATUS (2026-07-17, written under kill-order — code is traced from
//! source but NEVER COMPILED OR TESTED; expect mechanical fixes):
//! every function carries its exact C provenance; `// HANDOFF:` marks the
//! spots that still need verification or wiring. See HANDOFF-SPEED89.md at
//! the worktree root for the full state + validation recipe.
//!
//! ## The chroma answer (the KB-11 flagged unknown — RESOLVED)
//! `av1_nonrd_pick_intra_mode` is Y-only and hard-sets
//! `mi->uv_mode = UV_DC_PRED` (nonrd_pickmode.c:1735, comment "Keep DC for UV
//! since mode test is based on Y channel only"). There is NO uv mode search
//! and NO uv rate/dist estimate on the estimate arm; chroma is coded as DC by
//! the ordinary leaf encode (`encode_superblock` — port: `encode_b_intra_dry`
//! consuming `LeafWinner::uv_mode = 0`). The full-RD arm
//! (`av1_rd_pick_intra_mode_sb` via `hybrid_intra_mode_search`,
//! partition_search.c:755-772) picks uv with the EXISTING ported machinery
//! (`leaf_pick_sb_modes`). Palette/CfL: `init_mbmi_nonrd` (nonrd_opt.h:516)
//! zeroes palette sizes + filter_intra; CfL is never a candidate (uv fixed
//! DC), so `cfl_alpha_* = 0` on estimate leaves.
//!
//! ## Speed-8 allintra sf deltas (speed_features.c:577-590, verified)
//! - `hybrid_intra_pickmode = 2` → full-RD arm for `bsize < BLOCK_16X16 &&
//!   source_variance >= var_thresh[1] = 101` (partition_search.c:762-766;
//!   `var_thresh = {0, 101, 201}`, index `hybrid_intra_pickmode - 1`).
//! - `use_nonrd_pick_mode = 1` → `encode_nonrd_sb` + `av1_nonrd_use_partition`.
//! - `nonrd_check_partition_merge_mode = 1` (framesize-dependent :157-160
//!   raises it to 2 below 480p) — `try_merge` is `!frame_is_intra_only`-gated
//!   (partition_search.c:3089) → INERT on KEY.
//! - `var_part_split_threshold_shift = 8` — read only under
//!   `force_large_partition_blocks_intra`, which needs speed>=8 AND 720p+
//!   (:161-163) → inert on the canon grid (<720p).
//! - `prune_palette_search_nonrd = 1` — palette arm still needs
//!   `enable_palette && av1_allow_palette(allow_screen_content_tools, bsize)`;
//!   canon grid runs `allow_screen_content_tools = 0` → dead (guarded below).
//! - `intra_y_mode_bsize_mask_nrd[...]` = INTRA_DC (>=32x32) / INTRA_DC_H_V —
//!   consumed ONLY by `is_prune_intra_mode` (nonrd_opt.c:570) on the INTER
//!   frame path → INERT on KEY (verified: `av1_nonrd_pick_intra_mode` loops
//!   `intra_mode_list` directly with no mask check).
//!
//! ## Speed-9 allintra sf deltas (speed_features.c:592-607 + :166-177, verified)
//! - `hybrid_intra_pickmode = 0` → the full-RD arm DIES; every leaf uses the
//!   estimate loop below.
//! - `nonrd_check_partition_merge_mode = 0` (still KEY-inert either way).
//! - `var_part_split_threshold_shift = 7` (still force_large-gated → inert
//!   <720p).
//! - `vbp_prune_16x16_split_using_min_max_sub_blk_var = true` → LIVE in the
//!   KEY VBP tree: the 16x16 force-split rule (var_based_part.c:1804-1809)
//!   becomes `get_part_eval_based_on_sub_blk_var(vtemp, thresholds[3])`
//!   (:1530): max/min over the four 8x8 sub-variances; `(max - min) >
//!   (threshold16 << 2)` → ONLY_SPLIT else ONLY_NONE (instead of
//!   unconditional ONLY_SPLIT). Port: thread the existing
//!   `vbp_prune_16x16_split_using_min_max_sub_blk_var` param of
//!   [`crate::var_part::choose_var_based_partitioning_key`] as
//!   `speed >= 9` — the param already exists (passed `false` today) but
//!   HANDOFF: verify var_part.rs implements the ONLY_NONE arm (3-state
//!   PART_EVAL semantics), not just a bool force-split.
//! - `prune_h_pred_using_best_mode_so_far = true` → estimate-loop prune (live).
//! - `enable_intra_mode_pruning_using_neighbors = true` → estimate-loop prune
//!   (live).
//! - `prune_intra_mode_using_best_sad_so_far = true` → per-mode SAD prune in
//!   `av1_estimate_block_intra` (live; `bsize == tx_bsize` always holds for
//!   our square single-txb leaves).
//! - `coeff_cost_upd_level = mode_cost_upd_level = INTERNAL_COST_UPD_SBROW`
//!   (framesize-independent :593-594) then **INTERNAL_COST_UPD_OFF for <4k**
//!   (framesize-DEPENDENT :166-177, runs later and wins) → on the whole canon
//!   grid the per-SB `derive_real_costs` refresh STOPS at speed 9: every SB
//!   uses the FRAME-INIT cost tables (visible on 128² cells = 4 SBs;
//!   64² = 1 SB is inert). Port: gate the `derive_real_costs` call in
//!   `pack::pack_tile` on `speed <= 8`.
//! - SB size: `av1_select_sb_size` (encoder_utils.c:958) already returns
//!   BLOCK_64X64 for `speed >= 1 && <= 480p` — the speed-9 allintra <4k rule
//!   (:1035-1037) adds nothing on the canon grid.
//!
//! ## Structural notes (encode side)
//! - `pick_sb_modes_nonrd` (partition_search.c:2254): recomputes
//!   `x->source_variance = av1_get_perpixel_variance_facade(bsize)` per leaf
//!   whenever `bsize < sb_size` OR it is UINT_MAX (:2306-2311); the SB-level
//!   value `choose_var_based_partitioning` computes (var_based_part.c:1724-1731,
//!   gated `use_nonrd_pick_mode && source_sad_nonrd > kLowSad`; KEY inits
//!   source_sad_nonrd = kMedSad, encodeframe.c:1289) is the SAME
//!   perpixel-variance of the same pixels → per-leaf recompute is exact.
//!   `get_force_zeromv_skip_flag_for_blk` (:2182) returns
//!   `force_zeromv_skip_for_sb` when < 2 — 0 on KEY → no gating.
//! - `encode_b_nonrd` (partition_search.c:2089): set_offsets_without_segment_id
//!   → `setup_block_rdmult(.., NO_AQ, NULL)` (identity here: aq NONE +
//!   VBP leaves the ALLINTRA sb modifier at 128, the KB-11 fact) →
//!   `av1_update_state` → `if (!is_inter_block) mi->skip_txfm = 0` →
//!   `encode_superblock(dry_run=0)` → cb_offsets/update_stats. In this port's
//!   split architecture the recon+context walk is `encode_b_intra_dry` and the
//!   bit-writing is `pack_sb` over the finished tree — the SAME split already
//!   proven byte-exact for speeds 0-7 (the symbol stream determines the CDF
//!   adaptation, and the nonrd walk's symbol stream is the same tree replay).
//! - rd costs of the walk are DECISION-INERT (`dummy_cost` is invalid and
//!   never compared — av1_nonrd_use_partition:2983); only the per-leaf
//!   estimate numerics decide `best_mode`, which is why `av1_block_yrd` must
//!   be bit-exact.
//!
//! ## Lowbd-only
//! The canon grid is 8-bit; `use_hbd` (`is_cur_buf_hbd`) is FALSE there, so
//! only the `_lp` kernel family is live: `aom_hadamard_lp_8x8/16x16`,
//! `aom_fdct4x4_lp`, `av1_quantize_lp`, `aom_satd_lp`, `av1_block_error_lp`,
//! with the `*_lp_*_transpose` scans. The hbd arm (aom_hadamard_16x16 +
//! av1_quantize_fp + `fp_16x16_transpose` scans, nonrd_opt.c:199-215) is NOT
//! ported — guarded by an assert below. HANDOFF: port the hbd arm before any
//! bd10/bd12 speed-8 gate.

use crate::encode_sb::SbEncodeEnv;
use crate::partition::PartRdStats;
use aom_dist::highbd_subtract_block;
use aom_intra::predict_intra_high;

/// `MI_SIZE_WIDE`/`HIGH` for the square sizes used here (port-wide numbering:
/// BLOCK_8X8=3, BLOCK_16X16=6, BLOCK_32X32=9, BLOCK_64X64=12).
const MI_W: [usize; 22] = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
const MI_H: [usize; 22] = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];

/// `intra_mode_context[]` (av1_common_int.h) — KF y-mode cost context per
/// neighbour PREDICTION_MODE. HANDOFF: dedupe with the copy the full-RD leaf
/// uses (intra_rd.rs derives above_ctx/left_ctx somewhere — same table).
const INTRA_MODE_CONTEXT: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `intra_mode_list[]` (nonrd_opt.h:121): DC, V, H, SMOOTH.
const INTRA_MODE_LIST: [usize; 4] = [0, 1, 2, 9];

/// `AV1_PROB_COST_SHIFT` (av1/encoder/cost.h).
const AV1_PROB_COST_SHIFT: i32 = 9;

// ---------------------------------------------------------------------------
// LP kernels (aom_dsp/avg.c, aom_dsp/fwd_txfm.c, av1/encoder/av1_quantize.c,
// av1/encoder/rdopt.c) — C-scalar ports, all wrapping-i16 where C wraps.
// ---------------------------------------------------------------------------

/// `get_msb` (aom_dsp/bitops.h): index of the highest set bit; UB at 0 in C,
/// callers always pass `n >= 1`.
#[inline]
fn get_msb(n: u32) -> i32 {
    debug_assert!(n != 0);
    31 - n.leading_zeros() as i32
}

/// `hadamard_col8` (aom_dsp/avg.c:149). C does the arithmetic in int16_t —
/// intermediate sums are allowed to wrap (the dynamic-range comments bound
/// REAL inputs away from wrap, but bit-exactness demands wrapping semantics).
#[inline]
fn hadamard_col8(src: &[i16], stride: usize, coeff: &mut [i16; 8]) {
    let b0 = src[0].wrapping_add(src[stride]);
    let b1 = src[0].wrapping_sub(src[stride]);
    let b2 = src[2 * stride].wrapping_add(src[3 * stride]);
    let b3 = src[2 * stride].wrapping_sub(src[3 * stride]);
    let b4 = src[4 * stride].wrapping_add(src[5 * stride]);
    let b5 = src[4 * stride].wrapping_sub(src[5 * stride]);
    let b6 = src[6 * stride].wrapping_add(src[7 * stride]);
    let b7 = src[6 * stride].wrapping_sub(src[7 * stride]);

    let c0 = b0.wrapping_add(b2);
    let c1 = b1.wrapping_add(b3);
    let c2 = b0.wrapping_sub(b2);
    let c3 = b1.wrapping_sub(b3);
    let c4 = b4.wrapping_add(b6);
    let c5 = b5.wrapping_add(b7);
    let c6 = b4.wrapping_sub(b6);
    let c7 = b5.wrapping_sub(b7);

    coeff[0] = c0.wrapping_add(c4);
    coeff[7] = c1.wrapping_add(c5);
    coeff[3] = c2.wrapping_add(c6);
    coeff[4] = c3.wrapping_add(c7);
    coeff[2] = c0.wrapping_sub(c4);
    coeff[6] = c1.wrapping_sub(c5);
    coeff[1] = c2.wrapping_sub(c6);
    coeff[5] = c3.wrapping_sub(c7);
}

/// `aom_hadamard_lp_8x8_c` (aom_dsp/avg.c:209): 8x8 2D Hadamard, int16 out.
/// `coeff` receives 64 values in the C's transposed-output order (which is
/// why the `*_transpose` scans exist).
pub fn hadamard_lp_8x8(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    let mut buffer = [0i16; 64];
    let mut buffer2 = [0i16; 64];
    for idx in 0..8 {
        let mut col = [0i16; 8];
        hadamard_col8(&src_diff[idx..], src_stride, &mut col);
        buffer[idx * 8..idx * 8 + 8].copy_from_slice(&col);
    }
    for idx in 0..8 {
        let mut col = [0i16; 8];
        hadamard_col8(&buffer[idx..], 8, &mut col);
        buffer2[idx * 8..idx * 8 + 8].copy_from_slice(&col);
    }
    coeff[..64].copy_from_slice(&buffer2);
}

/// `aom_hadamard_lp_8x8_dual_c` (avg.c:240): two adjacent 8x8s. UNREACHABLE
/// from the intra estimate arm (needs `tx_size == TX_8X8 && block_width >=
/// 16`, but a square single-txb leaf clamps 8x8 only when the txb IS 8x8) —
/// kept for completeness / the inter path.
pub fn hadamard_lp_8x8_dual(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    for i in 0..2 {
        hadamard_lp_8x8(&src_diff[i * 8..], src_stride, &mut coeff[i * 64..]);
    }
}

/// `aom_hadamard_lp_16x16_c` (avg.c:291): four 8x8 stages + a cross-combine
/// with `>> 1` normalization. int16 wrapping.
pub fn hadamard_lp_16x16(src_diff: &[i16], src_stride: usize, coeff: &mut [i16]) {
    for idx in 0..4 {
        let src_off = (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        hadamard_lp_8x8(&src_diff[src_off..], src_stride, &mut coeff[idx * 64..]);
    }
    for idx in 0..64 {
        let a0 = coeff[idx];
        let a1 = coeff[idx + 64];
        let a2 = coeff[idx + 128];
        let a3 = coeff[idx + 192];

        let b0 = a0.wrapping_add(a1) >> 1;
        let b1 = a0.wrapping_sub(a1) >> 1;
        let b2 = a2.wrapping_add(a3) >> 1;
        let b3 = a2.wrapping_sub(a3) >> 1;

        coeff[idx] = b0.wrapping_add(b2);
        coeff[idx + 64] = b1.wrapping_add(b3);
        coeff[idx + 128] = b0.wrapping_sub(b2);
        coeff[idx + 192] = b1.wrapping_sub(b3);
    }
}

/// `aom_fdct4x4_lp_c` (aom_dsp/fwd_txfm.c:85). Reachable only at lossless
/// (TX_4X4) — outside the canon envelope, ported for completeness.
pub fn fdct4x4_lp(input: &[i16], output: &mut [i16], stride: usize) {
    // cospi constants (aom_dsp/txfm_common.h).
    const COSPI_16_64: i32 = 11585;
    const COSPI_24_64: i32 = 6270;
    const COSPI_8_64: i32 = 15137;
    const DCT_CONST_BITS: i32 = 14;
    #[inline]
    fn fdct_round_shift(v: i32) -> i32 {
        (v + (1 << (DCT_CONST_BITS - 1))) >> DCT_CONST_BITS
    }
    let mut intermediate = [0i16; 16];
    for pass in 0..2 {
        for i in 0..4 {
            let mut in_high = [0i32; 4];
            if pass == 0 {
                in_high[0] = i32::from(input[i]) * 16;
                in_high[1] = i32::from(input[stride + i]) * 16;
                in_high[2] = i32::from(input[2 * stride + i]) * 16;
                in_high[3] = i32::from(input[3 * stride + i]) * 16;
                if i == 0 && in_high[0] != 0 {
                    in_high[0] += 1;
                }
            } else {
                in_high[0] = i32::from(intermediate[i]);
                in_high[1] = i32::from(intermediate[4 + i]);
                in_high[2] = i32::from(intermediate[8 + i]);
                in_high[3] = i32::from(intermediate[12 + i]);
            }
            let step0 = in_high[0] + in_high[3];
            let step1 = in_high[1] + in_high[2];
            let step2 = in_high[1] - in_high[2];
            let step3 = in_high[0] - in_high[3];
            let t0 = fdct_round_shift((step0 + step1) * COSPI_16_64) as i16;
            let t2 = fdct_round_shift((step0 - step1) * COSPI_16_64) as i16;
            let t1 = fdct_round_shift(step2 * COSPI_24_64 + step3 * COSPI_8_64) as i16;
            let t3 = fdct_round_shift(-step2 * COSPI_8_64 + step3 * COSPI_24_64) as i16;
            if pass == 0 {
                intermediate[i * 4] = t0;
                intermediate[i * 4 + 1] = t1;
                intermediate[i * 4 + 2] = t2;
                intermediate[i * 4 + 3] = t3;
            } else {
                output[i] = t0;
                output[4 + i] = t1;
                output[8 + i] = t2;
                output[12 + i] = t3;
            }
        }
    }
    // C post-pass: output[j] = (output[j] + 1) >> 2 (fwd_txfm.c:150-ish).
    // HANDOFF: verify the final rounding loop of aom_fdct4x4_lp_c —
    // read past line 145; the fdct4x4 (non-lp) does
    // `(out + 1) >> 2`; confirm the lp variant matches before ANY lossless
    // speed-8 use. (Unreachable on the canon grid.)
    for v in output[..16].iter_mut() {
        *v = (*v + 1) >> 2;
    }
}

/// `av1_quantize_lp_c` (av1/encoder/av1_quantize.c:214): the low-precision FP
/// quantizer. `scan` orders the eob computation; qcoeff/dqcoeff are written at
/// RAW (`rc`) positions. round/quant/dequant use row lane `[rc != 0]`.
#[allow(clippy::too_many_arguments)]
pub fn quantize_lp(
    coeff: &[i16],
    n_coeffs: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    qcoeff: &mut [i16],
    dqcoeff: &mut [i16],
    dequant: &[i16; 8],
    scan: &[i16],
) -> u16 {
    let mut eob: i32 = -1;
    qcoeff[..n_coeffs].fill(0);
    dqcoeff[..n_coeffs].fill(0);
    for (i, &sc) in scan[..n_coeffs].iter().enumerate() {
        let rc = sc as usize;
        let c = i32::from(coeff[rc]);
        let coeff_sign = c >> 31; // AOMSIGN
        let abs_coeff = (c ^ coeff_sign) - coeff_sign;
        let lane = usize::from(rc != 0);
        let mut tmp = (abs_coeff + i32::from(round_fp[lane])).clamp(i16::MIN as i32, i16::MAX as i32);
        tmp = (tmp * i32::from(quant_fp[lane])) >> 16;
        qcoeff[rc] = ((tmp ^ coeff_sign) - coeff_sign) as i16;
        dqcoeff[rc] = qcoeff[rc].wrapping_mul(dequant[lane]);
        if tmp != 0 {
            eob = i as i32;
        }
    }
    (eob + 1) as u16
}

/// `aom_satd_lp_c` (avg.c:520).
pub fn satd_lp(coeff: &[i16], length: usize) -> i32 {
    coeff[..length].iter().map(|&c| i32::from(c).abs()).sum()
}

/// `av1_block_error_lp_c` (rdopt.c:907).
pub fn block_error_lp(coeff: &[i16], dqcoeff: &[i16], block_size: usize) -> i64 {
    let mut error: i64 = 0;
    for i in 0..block_size {
        let diff = i64::from(coeff[i]) - i64::from(dqcoeff[i]);
        error += diff * diff;
    }
    error
}

// ---------------------------------------------------------------------------
// Transposed scan orders (nonrd_opt.h:212-300) — used ONLY with the lp
// Hadamard outputs (whose coefficient order is the C transposed layout).
// ---------------------------------------------------------------------------

/// `default_scan_8x8_transpose` (nonrd_opt.h:212).
pub const DEFAULT_SCAN_8X8_TRANSPOSE: [i16; 64] = [
    0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34,
    41, 48, 56, 49, 42, 35, 28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 31,
    38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
];

/// `default_scan_lp_16x16_transpose` (nonrd_opt.h:238).
pub const DEFAULT_SCAN_LP_16X16_TRANSPOSE: [i16; 256] = [
    0, 8, 2, 4, 10, 16, 24, 18, 12, 6, 64, 14, 20, 26, 32, 40, 34, 28, 22, 72, 66, 68, 74, 80, 30,
    36, 42, 48, 56, 50, 44, 38, 88, 82, 76, 70, 128, 78, 84, 90, 96, 46, 52, 58, 1, 9, 3, 60, 54,
    104, 98, 92, 86, 136, 130, 132, 138, 144, 94, 100, 106, 112, 62, 5, 11, 17, 25, 19, 13, 7,
    120, 114, 108, 102, 152, 146, 140, 134, 192, 142, 148, 154, 160, 110, 116, 122, 65, 15, 21,
    27, 33, 41, 35, 29, 23, 73, 67, 124, 118, 168, 162, 156, 150, 200, 194, 196, 202, 208, 158,
    164, 170, 176, 126, 69, 75, 81, 31, 37, 43, 49, 57, 51, 45, 39, 89, 83, 77, 71, 184, 178, 172,
    166, 216, 210, 204, 198, 206, 212, 218, 224, 174, 180, 186, 129, 79, 85, 91, 97, 47, 53, 59,
    61, 55, 105, 99, 93, 87, 137, 131, 188, 182, 232, 226, 220, 214, 222, 228, 234, 240, 190, 133,
    139, 145, 95, 101, 107, 113, 63, 121, 115, 109, 103, 153, 147, 141, 135, 248, 242, 236, 230,
    238, 244, 250, 193, 143, 149, 155, 161, 111, 117, 123, 125, 119, 169, 163, 157, 151, 201, 195,
    252, 246, 254, 197, 203, 209, 159, 165, 171, 177, 127, 185, 179, 173, 167, 217, 211, 205, 199,
    207, 213, 219, 225, 175, 181, 187, 189, 183, 233, 227, 221, 215, 223, 229, 235, 241, 191, 249,
    243, 237, 231, 239, 245, 251, 253, 247, 255,
];

// NOTE: the `av1_default_iscan_*_transpose` tables are NOT needed —
// `av1_quantize_lp_c` ignores its iscan argument entirely ((void)iscan,
// av1_quantize.c:219). The fp (hbd) quantizer DOES use iscan; port those
// tables with the hbd arm if it's ever needed.

// ---------------------------------------------------------------------------
// av1_block_yrd (nonrd_opt.c:126) — lowbd arm.
// ---------------------------------------------------------------------------

/// One txb's Hadamard-estimate RD, `av1_block_yrd` with `use_hbd == 0`.
///
/// `diff` is the residual for the WHOLE txb (`bsize_tx`), stride `bw` (4 *
/// mi-width of the txb bsize) — the caller has already run
/// `aom_subtract_block` (here: [`highbd_subtract_block`] on the 8-bit-valued
/// u16 planes, identical arithmetic).
///
/// `tx_size` is the CLAMPED loop size (`AOMMIN(mi->tx_size, TX_16X16)`,
/// nonrd_opt.c:660): 0=4x4, 1=8x8, 2=16x16 sub-blocks over the txb.
/// Returns `(rate, dist, skippable)` where rate is the pre-shift SATD
/// accumulation already folded per C (`rate <<= 2 + AV1_PROB_COST_SHIFT;
/// rate += eob_cost << AV1_PROB_COST_SHIFT`).
///
/// `max_blocks_wide/high`: the C edge clamps (`num_4x4 + (mb_to_edge >> 5)`
/// when negative) — pass the full mi counts for interior leaves.
#[allow(clippy::too_many_arguments)]
pub fn block_yrd_lowbd(
    diff: &[i16],
    bw4: usize, // num_4x4_w of the txb bsize (diff stride = 4 * bw4)
    bh4: usize, // num_4x4_h
    max_blocks_wide: usize,
    max_blocks_high: usize,
    tx_size: usize,
    round_fp: &[i16; 8],
    quant_fp: &[i16; 8],
    dequant: &[i16; 8],
) -> (i32, i64, bool) {
    debug_assert!(tx_size <= 2, "clamped to <= TX_16X16 (nonrd_opt.c:660)");
    let diff_stride = 4 * bw4;
    let block_step = 1usize << tx_size;
    let step = 1usize << (tx_size << 1); // 4x4 units per sub-block
    let _ = bh4;

    let mut rate: i32 = 0;
    let mut dist: i64 = 0;
    let mut eob_cost: i32 = 0;
    let mut temp_skippable = true;

    let mut coeff = [0i16; 256];
    let mut qcoeff = [0i16; 256];
    let mut dqcoeff = [0i16; 256];

    let mut r = 0usize;
    while r < max_blocks_high {
        let mut c = 0usize;
        while c < max_blocks_wide {
            let src_diff = &diff[(r * diff_stride + c) * 4..];
            let eob: u16 = match tx_size {
                2 => {
                    hadamard_lp_16x16(src_diff, diff_stride, &mut coeff);
                    quantize_lp(
                        &coeff,
                        256,
                        round_fp,
                        quant_fp,
                        &mut qcoeff,
                        &mut dqcoeff,
                        dequant,
                        &DEFAULT_SCAN_LP_16X16_TRANSPOSE,
                    )
                }
                1 => {
                    hadamard_lp_8x8(src_diff, diff_stride, &mut coeff);
                    quantize_lp(
                        &coeff,
                        64,
                        round_fp,
                        quant_fp,
                        &mut qcoeff,
                        &mut dqcoeff,
                        dequant,
                        &DEFAULT_SCAN_8X8_TRANSPOSE,
                    )
                }
                _ => {
                    // TX_4X4: aom_fdct4x4_lp + the NORMAL default 4x4 scan
                    // (av1_scan_orders[TX_4X4][DCT_DCT] — no transpose,
                    // nonrd_opt.c:252 comment). Lossless-only.
                    // HANDOFF: wire av1_scan_orders[TX_4X4][DCT_DCT].scan from
                    // aom-entropy's scan tables (default_scan_4x4) if a
                    // lossless speed-8 envelope ever opens.
                    unimplemented!("TX_4X4 block_yrd (lossless) — out of canon envelope")
                }
            };
            // update_yrd_loop_vars (nonrd_opt.c:43).
            let ncoeffs = eob as usize;
            let is_txfm_skip = ncoeffs == 0;
            temp_skippable &= is_txfm_skip;
            // x->txfm_search_info.blk_skip[r * num_blk_skip_w + c] write:
            // decision-inert for KEY intra (consumed by the inter var-tx
            // path only) — not modelled. HANDOFF: verify nothing on the
            // allintra pack path reads blk_skip (speeds 0-7 never set it
            // from here; the full-RD arm has its own).
            eob_cost += get_msb(ncoeffs as u32 + 1);
            if ncoeffs == 1 {
                rate += i32::from(qcoeff[0]).abs();
            } else if ncoeffs > 1 {
                rate += satd_lp(&qcoeff, step << 4);
            }
            dist += block_error_lp(&coeff, &dqcoeff, step << 4) >> 2;
            c += block_step;
        }
        r += block_step;
    }

    // (nonrd_opt.c:322-336): this_rdc->sse is INT64_MAX from the caller's
    // av1_invalid_rd_stats → the `sse < INT64_MAX` skippable-dist arm never
    // fires on the intra estimate path; rate gets the final shifts.
    let rate = (rate << (2 + AV1_PROB_COST_SHIFT)) + (eob_cost << AV1_PROB_COST_SHIFT);
    (rate, dist, temp_skippable)
}

// ---------------------------------------------------------------------------
// av1_nonrd_pick_intra_mode (nonrd_pickmode.c:1582) — the estimate arm.
// ---------------------------------------------------------------------------

/// Per-leaf inputs the estimate arm needs beyond [`SbEncodeEnv`].
pub struct NonrdIntraLeafCtx<'a> {
    /// `y_mode_costs[above_ctx][left_ctx]` KF table (13 modes) — from
    /// `mode_costs.y_mode_costs[intra_mode_context[A]][intra_mode_context[L]]`.
    pub bmode_costs: &'a [i32; 13],
    /// `skip_txfm_cost[skip_ctx]` — skip_ctx is 0 on the KEY intra path
    /// (every neighbour mi carries skip_txfm 0; the leaf_pick_sb_modes
    /// invariant, verified 64/64 across speeds 0-7).
    pub skip_cost: &'a [i32; 2],
    /// Above/left neighbour Y modes (A/L; DC=0 when unavailable) + their
    /// availability — the neighbour-prune inputs.
    pub above_mode: usize,
    pub left_mode: usize,
    pub up_available: bool,
    pub left_available: bool,
    /// x->source_variance for THIS leaf (perpixel_variance_y at leaf bsize).
    pub source_variance: u32,
    /// intra_avail geometry.
    pub partition: usize,
    /// Speed-9 sf gates (all false at speed 8).
    pub prune_h_pred_using_best_mode_so_far: bool,
    pub enable_intra_mode_pruning_using_neighbors: bool,
    pub prune_intra_mode_using_best_sad_so_far: bool,
    /// `prune_palette_search_nonrd` level (1 at speed>=8) + the palette
    /// enable inputs — used ONLY to assert the palette arm stays dead.
    pub allow_screen_content_tools: bool,
    /// Edge filter type for directional prediction (V/H are directional):
    /// `get_intra_edge_filter_type(xd, 0)` — smooth above/left neighbour.
    pub luma_edge_filter_type: i32,
}

/// The estimate-arm result: the winner Y mode + the ctx snapshot fields
/// `store_coding_context_nonrd` (nonrd_opt.h:576-597) preserves that the
/// encode consumes. uv is ALWAYS DC (the chroma answer).
pub struct NonrdIntraPick {
    pub mode: usize,
    /// `mi->tx_size` = `AOMMIN(max_txsize_lookup[bsize],
    /// tx_mode_to_biggest_tx_size[TX_MODE_SELECT])` (nonrd_pickmode.c:1591) —
    /// the max square tx for the leaf (TX_64X64 cap).
    pub tx_size: usize,
    pub rd: PartRdStats,
}

/// `max_txsize_lookup[bsize]` for the square sizes, capped TX_64X64
/// (tx_mode_to_biggest_tx_size[TX_MODE_SELECT] = TX_64X64; allintra speed 8/9
/// keeps tx_size_search_method != USE_LARGEST_TX_SIZE → cm tx_mode =
/// TX_MODE_SELECT — HANDOFF: re-verify select_tx_mode at speed 8/9 allintra
/// (av1/encoder/encodeframe_utils/rdopt_utils select_tx_mode); if it were
/// TX_MODE_LARGEST the biggest is TX_64X64 anyway, same value — only
/// ONLY_4X4/lossless differs and that's out of envelope).
pub fn nonrd_leaf_tx_size(bsize: usize) -> usize {
    match bsize {
        3 => 1,  // BLOCK_8X8  -> TX_8X8
        6 => 2,  // BLOCK_16X16 -> TX_16X16
        9 => 3,  // BLOCK_32X32 -> TX_32X32
        12 => 4, // BLOCK_64X64 -> TX_64X64
        _ => panic!("nonrd leaf bsize {bsize}: KEY VBP tree stamps squares 8x8..64x64 only"),
    }
}

/// `should_prune_intra_modes_using_neighbors` (nonrd_pickmode.c:1566).
fn should_prune_intra_modes_using_neighbors(
    enable: bool,
    this_mode: usize,
    above_mode: usize,
    left_mode: usize,
    up_available: bool,
    left_available: bool,
) -> bool {
    if !enable {
        return false;
    }
    if this_mode == 0 {
        return false; // DC never pruned
    }
    up_available && this_mode != above_mode && left_available && this_mode != left_mode
}

/// `av1_nonrd_pick_intra_mode` (nonrd_pickmode.c:1582), Y estimate loop.
///
/// The prediction step (`av1_estimate_block_intra` → `av1_predict_intra_block
/// _facade`) writes INTO the recon plane at the leaf position (the C facade's
/// dst IS pd->dst), scribbling only inside the block — the winner encode
/// (`encode_b_intra_dry`) re-predicts + adds residual afterwards, exactly like
/// C's encode_superblock.
///
/// Single-txb invariant: `mi->tx_size` is the max square tx of the leaf, so
/// `av1_foreach_transformed_block_in_plane` visits exactly ONE txb (the whole
/// block) — the loop below is that one visit inlined.
#[allow(clippy::too_many_arguments)]
pub fn nonrd_pick_intra_mode(
    env: &SbEncodeEnv,
    lctx: &NonrdIntraLeafCtx,
    recon_y: &mut [u16],
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    rdmult: i32,
) -> NonrdIntraPick {
    assert!(env.bd == 8, "HANDOFF: hbd estimate arm (av1_quantize_fp + fp scans) not ported");
    let mi_w = MI_W[bsize];
    let mi_h = MI_H[bsize];
    let bw = mi_w * 4;
    let bh = mi_h * 4;
    let tx_size_full = nonrd_leaf_tx_size(bsize); // mi->tx_size (signalled)
    let tx_clamped = tx_size_full.min(2); // AOMMIN(tx_size, TX_16X16) for block_yrd

    // Edge clamps (block_yrd's max_blocks_wide/high; av1_block_yrd:141-144):
    // mb_to_right_edge = (mi_cols - mi_w - mi_col) * 4 * 8 (in 1/8 pel).
    let mb_right = (env.mi_cols - mi_w as i32 - mi_col) * 32;
    let mb_bottom = (env.mi_rows - mi_h as i32 - mi_row) * 32;
    let max_blocks_wide = mi_w as i32 + if mb_right >= 0 { 0 } else { mb_right >> 5 };
    let max_blocks_high = mi_h as i32 + if mb_bottom >= 0 { 0 } else { mb_bottom >> 5 };
    let (max_blocks_wide, max_blocks_high) = (max_blocks_wide as usize, max_blocks_high as usize);

    let ref_off = env.base_y + (mi_row as usize * 4) * env.stride + mi_col as usize * 4;
    let src_off = ref_off; // src and recon share layout in this port

    let above_ctx = INTRA_MODE_CONTEXT[lctx.above_mode.min(12)];
    let left_ctx = INTRA_MODE_CONTEXT[lctx.left_mode.min(12)];
    let _ = (above_ctx, left_ctx); // bmode_costs already row-selected by caller
    // HANDOFF: caller must select bmode_costs with the SAME ctx pair —
    // the row selection is the caller's so it can reuse leaf_pick_sb_modes'
    // existing neighbour reads; assert parity there.

    let mut best_rdc = PartRdStats::invalid();
    let mut best_mode = 0usize; // DC_PRED
    let mut best_sad = u32::MAX;
    let prune_mode_based_on_sad = lctx.prune_intra_mode_using_best_sad_so_far; // bsize == tx_bsize always
    let allow_skip_nondc = true; // flat_blocks_screen is REALTIME+SCREEN only → const true (ALLINTRA)

    let mut diff = vec![0i16; bw * bh];
    let mut pred = vec![0u16; bw * bh];

    for &this_mode in INTRA_MODE_LIST.iter() {
        // Force DC for spatially flat block at top-left, bsize >= 32x32
        // (nonrd_pickmode.c:1636-1640) — LIVE on the flat canon cells.
        if lctx.source_variance == 0 && mi_col == 0 && mi_row == 0 && bsize >= 9 && this_mode > 0 {
            continue;
        }
        // prune_h_pred_using_best_mode_so_far (:1648-1650), speed 9.
        if lctx.prune_h_pred_using_best_mode_so_far
            && this_mode == 2
            && best_mode == 1
            && allow_skip_nondc
        {
            continue;
        }
        if should_prune_intra_modes_using_neighbors(
            lctx.enable_intra_mode_pruning_using_neighbors,
            this_mode,
            lctx.above_mode,
            lctx.left_mode,
            lctx.up_available,
            lctx.left_available,
        ) {
            // (:1656-1668), speed 9.
            if (this_mode == 1 || this_mode == 2)
                && lctx.source_variance <= 50
                && allow_skip_nondc
            {
                continue;
            }
            if best_mode == 0 && this_mode == 9 && allow_skip_nondc {
                continue;
            }
        }

        // --- av1_estimate_block_intra, single txb == whole block ---
        // Predict with the leaf's SIGNALLED tx_size (prediction granularity
        // is mi->tx_size, NOT the clamped block_yrd loop size).
        let (n_top, n_topright, n_left, n_bottomleft) = aom_entropy::partition::intra_avail(
            env.sb_size,
            bsize,
            mi_row,
            mi_col,
            lctx.up_available,
            lctx.left_available,
            env.tile_col_end,
            env.tile_row_end,
            lctx.partition,
            tx_size_full,
            0,
            0,
            0, // blk_row
            0, // blk_col
            bw as i32,
            bh as i32,
            env.mi_cols,
            env.mi_rows,
            this_mode,
            0,     // angle_delta * ANGLE_STEP
            false, // use_filter_intra
        );
        predict_intra_high(
            recon_y,
            ref_off,
            env.stride,
            &mut pred,
            bw,
            this_mode,
            0,
            false,
            0,
            env.disable_edge_filter,
            lctx.luma_edge_filter_type,
            tx_size_full,
            n_top as usize,
            n_topright,
            n_left as usize,
            n_bottomleft,
            i32::from(env.bd),
        );
        // Facade writes prediction into the recon plane (dst) — mirror that.
        for r in 0..bh {
            recon_y[ref_off + r * env.stride..ref_off + r * env.stride + bw]
                .copy_from_slice(&pred[r * bw..r * bw + bw]);
        }

        // Speed-9 SAD prune (av1_estimate_block_intra:646-668).
        if prune_mode_based_on_sad {
            let mut this_sad: u32 = 0;
            for r in 0..bh {
                for c in 0..bw {
                    let s = env.src_y[src_off + r * env.stride + c] as i32;
                    let p = pred[r * bw + c] as i32;
                    this_sad += (s - p).unsigned_abs();
                }
            }
            let sad_threshold = if best_sad != u32::MAX {
                best_sad + (best_sad >> 4)
            } else {
                u32::MAX
            };
            if this_sad > sad_threshold {
                // rate INT_MAX → the caller-side `if (this_rdc.rate == INT_MAX)
                // continue` (:1674).
                continue;
            }
            if this_sad < best_sad {
                best_sad = this_sad;
            }
        }

        // av1_subtract_block over the whole txb.
        highbd_subtract_block(
            bh,
            bw,
            &mut diff,
            bw,
            &env.src_y[src_off..],
            env.stride,
            &pred,
            bw,
        );
        let (rate_yrd, dist_yrd, skippable) = block_yrd_lowbd(
            &diff,
            mi_w,
            mi_h,
            max_blocks_wide,
            max_blocks_high,
            tx_clamped,
            env.rows_y.round_fp,
            env.rows_y.quant_fp,
            env.rows_y.dequant,
        );

        // (:1676-1687): skip-cost fold (skip_ctx 0 on KEY intra — module docs)
        // + the KF y-mode cost.
        let mut rate = if skippable {
            lctx.skip_cost[1] // '=' — clobbers the SATD rate (C :1678)
        } else {
            rate_yrd + lctx.skip_cost[0]
        };
        rate += lctx.bmode_costs[this_mode];
        let rdc = crate::rd::rdcost(rdmult, rate, dist_yrd);
        if rdc < best_rdc.rdcost {
            best_rdc = PartRdStats { rate, dist: dist_yrd, rdcost: rdc };
            best_mode = this_mode;
        }
        // flat_blocks_screen / allow_skip_nondc mutation: dead at ALLINTRA
        // (cpi->oxcf.mode == REALTIME gate, :1620-1623).
    }

    // Palette arm (:1698-1731): requires enable_palette &&
    // av1_allow_palette(allow_screen_content_tools, bsize) — the canon grid
    // encodes with allow_screen_content_tools = 0 → dead. Guarded:
    debug_assert!(
        !lctx.allow_screen_content_tools,
        "HANDOFF: av1_search_palette_mode_luma (palette.c) not ported — required \
         before any screen-content (allow_screen_content_tools=1) speed-8 cell"
    );

    // mi->mode = best_mode; mi->uv_mode = UV_DC_PRED (:1734-1735) — the
    // chroma answer. store_coding_context_nonrd's ctx->mic snapshot maps to
    // the LeafWinner the caller builds from this pick.
    NonrdIntraPick { mode: best_mode, tx_size: tx_size_full, rd: best_rdc }
}

/// `hybrid_intra_mode_search` (partition_search.c:755): the speed-8 dispatch.
/// `hybrid_intra_pickmode`: 2 at speed 8, 0 at speed 9 (allintra).
/// Returns true → run the full-RD leaf (`leaf_pick_sb_modes`); false → the
/// estimate arm above. `var_thresh = {0, 101, 201}[hybrid - 1]`.
pub fn hybrid_use_rdopt(hybrid_intra_pickmode: i32, bsize: usize, source_variance: u32) -> bool {
    debug_assert!((0..=3).contains(&hybrid_intra_pickmode));
    if hybrid_intra_pickmode == 0 {
        return false;
    }
    // bsize < BLOCK_16X16 (port numbering: < 6 — 8x8 and the sub-8x8 rects;
    // the KEY VBP tree stamps nothing below 8x8).
    if bsize >= 6 {
        return false;
    }
    let var_thresh: [u32; 3] = [0, 101, 201];
    source_variance >= var_thresh[(hybrid_intra_pickmode - 1) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// quantize_lp against hand-computed values (C:214 semantics: round/quant
    /// lane [rc!=0], eob over the scan order, dq = q * dequant).
    #[test]
    fn quantize_lp_basic() {
        let mut coeff = [0i16; 64];
        coeff[0] = 100; // DC
        coeff[8] = -40; // first AC in the transposed scan (scan[1] = 8)
        let round_fp = [48i16, 24, 24, 24, 24, 24, 24, 24];
        let quant_fp = [2048i16, 1024, 1024, 1024, 1024, 1024, 1024, 1024];
        let dequant = [32i16, 64, 64, 64, 64, 64, 64, 64];
        let mut q = [0i16; 64];
        let mut dq = [0i16; 64];
        let eob = quantize_lp(
            &coeff,
            64,
            &round_fp,
            &quant_fp,
            &mut q,
            &mut dq,
            &dequant,
            &DEFAULT_SCAN_8X8_TRANSPOSE,
        );
        // DC: (100+48)*2048 >> 16 = 4; dq = 4*32 = 128.
        assert_eq!(q[0], 4);
        assert_eq!(dq[0], 128);
        // AC at rc=8: (40+24)*1024 >> 16 = 1, negative → -1; dq = -64.
        assert_eq!(q[8], -1);
        assert_eq!(dq[8], -64);
        // scan[1] == 8 → eob index 1 → eob = 2.
        assert_eq!(eob, 2);
    }

    /// Hadamard lp 8x8: DC-only input → coeff[0] = 64 * v (sum), others 0.
    #[test]
    fn hadamard_lp_8x8_flat() {
        let src = [3i16; 64];
        let mut coeff = [0i16; 64];
        hadamard_lp_8x8(&src, 8, &mut coeff);
        assert_eq!(coeff[0], 64 * 3);
        assert!(coeff[1..].iter().all(|&c| c == 0));
    }

    /// Hadamard lp 16x16: flat input → DC = 256*v/4 (the >>1 stages halve
    /// twice), others 0.
    #[test]
    fn hadamard_lp_16x16_flat() {
        let src = [2i16; 256];
        let mut coeff = [0i16; 256];
        hadamard_lp_16x16(&src, 16, &mut coeff);
        // per-8x8 DC = 128; combine: b0 = (128+128)>>1 = 128; c0 = 128+128 = 256.
        assert_eq!(coeff[0], 256);
        assert!(coeff[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn hybrid_gate_matches_source() {
        // speed 8: hybrid=2 → threshold 101, only below 16x16.
        assert!(hybrid_use_rdopt(2, 3, 101));
        assert!(!hybrid_use_rdopt(2, 3, 100));
        assert!(!hybrid_use_rdopt(2, 6, 5000)); // 16x16: estimate arm
        assert!(!hybrid_use_rdopt(2, 9, 5000));
        // speed 9: hybrid=0 → never.
        assert!(!hybrid_use_rdopt(0, 3, 5000));
    }
}
