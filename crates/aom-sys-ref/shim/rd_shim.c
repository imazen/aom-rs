/* Shim over the exported RD-multiplier functions (av1/encoder/rd.c) and the
 * RDCOST / RDCOST_NEG_R macros (av1/encoder/rd.h), plus the exported
 * av1_{dc,ac}_quant_QTX accessors (av1/common/quant_common.c).
 *
 * av1_compute_rd_mult, av1_compute_rd_mult_based_on_qindex, av1_dc_quant_QTX and
 * av1_ac_quant_QTX are all non-static exported symbols in libaom.a. These thin
 * wrappers take plain `int` params to sidestep enum-ABI width questions
 * (FRAME_UPDATE_TYPE / MODE are UENUM1BYTE = uint8_t) and expose the real macros
 * from the real header, so any misreading shows up as a value mismatch in the
 * differential harness. Pure integer/table/float math — no RTCD needed. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "av1/common/blockd.h" /* av1_get_ext_tx_set_type, ext_tx_set_index,
                                  av1_num_ext_tx_set, fimode_to_intradir */
#include "av1/common/reconintra.h" /* av1_filter_intra_allowed_bsize,
                                      av1_is_directional_mode, av1_use_angle_delta */
#include "av1/common/common_data.h" /* txsize_sqr_map */
#include "av1/common/entropymode.h" /* av1_ext_tx_inv */
#include "av1/common/cfl.h" /* cfl_store_tx, av1_cfl_predict_block */
#include "av1/common/quant_common.h" /* av1_get_qindex */
#include "av1/common/seg_common.h" /* struct segmentation, SEG_LVL_ALT_Q */
#include "av1/common/idct.h" /* MAX_TX_SCALE, av1_get_tx_scale */
#include "av1/encoder/av1_quantize.h" /* QUANTS, Dequants, av1_build_quantizer */
#include "av1/encoder/cost.h" /* av1_cost_tokens_from_cdf */
#include "av1/encoder/ratectrl.h" /* av1_convert_qindex_to_q */
#include "av1/encoder/rd.h" /* av1_set_error_per_bit */
#include "aom_ports/mem.h" /* RIGHT_SIGNED_SHIFT */

/* Exported (RTCD `_c`) transform-domain distortion primitives; hand-declared
 * (they live in the generated av1_rtcd.h, not a plain header the shim pulls). */
int64_t av1_block_error_c(const int32_t *coeff, const int32_t *dqcoeff,
                          intptr_t block_size, int64_t *ssz);
int64_t av1_highbd_block_error_c(const int32_t *coeff, const int32_t *dqcoeff,
                                 intptr_t block_size, int64_t *ssz, int bd);

int shim_compute_rd_mult_based_on_qindex(int bit_depth, int update_type,
                                         int qindex, int tuning, int mode) {
  return av1_compute_rd_mult_based_on_qindex(
      (aom_bit_depth_t)bit_depth, (FRAME_UPDATE_TYPE)update_type, qindex,
      (aom_tune_metric)tuning, (MODE)mode);
}

int shim_compute_rd_mult(int qindex, int bit_depth, int update_type,
                         int layer_depth, int boost_index, int frame_type,
                         int use_fixed_qp_offsets, int is_stat_consumption_stage,
                         int tuning, int mode) {
  return av1_compute_rd_mult(qindex, (aom_bit_depth_t)bit_depth,
                             (FRAME_UPDATE_TYPE)update_type, layer_depth,
                             boost_index, (FRAME_TYPE)frame_type,
                             use_fixed_qp_offsets, is_stat_consumption_stage,
                             (aom_tune_metric)tuning, (MODE)mode);
}

int shim_dc_quant_qtx(int qindex, int delta, int bit_depth) {
  return av1_dc_quant_QTX(qindex, delta, (aom_bit_depth_t)bit_depth);
}

int shim_ac_quant_qtx(int qindex, int delta, int bit_depth) {
  return av1_ac_quant_QTX(qindex, delta, (aom_bit_depth_t)bit_depth);
}

int64_t shim_rdcost(int rm, int rate, int64_t dist) {
  return RDCOST(rm, rate, dist);
}

int64_t shim_rdcost_neg_r(int rm, int rate, int64_t dist) {
  return RDCOST_NEG_R(rm, rate, dist);
}

/* dist_block_tx_domain non-QM path (av1/encoder/tx_search.c), transcribed over
 * the real exported av1_block_error_c / av1_highbd_block_error_c: buffer_length
 * = av1_get_max_eob(tx_size); shift = (MAX_TX_SCALE - av1_get_tx_scale) * 2;
 * dist/sse right-signed-shifted to the common Q4 scale. */
void shim_dist_block_tx_domain(const int32_t *coeff, const int32_t *dqcoeff,
                               int tx_size, int bd, int64_t *out_dist,
                               int64_t *out_sse) {
  const int buffer_length = av1_get_max_eob((TX_SIZE)tx_size);
  const int shift = (MAX_TX_SCALE - av1_get_tx_scale((TX_SIZE)tx_size)) * 2;
  int64_t sse = 0, dist;
  if (bd > 8) {
    dist = av1_highbd_block_error_c(coeff, dqcoeff, buffer_length, &sse, bd);
  } else {
    dist = av1_block_error_c(coeff, dqcoeff, buffer_length, &sse);
  }
  *out_dist = RIGHT_SIGNED_SHIFT(dist, shift);
  *out_sse = RIGHT_SIGNED_SHIFT(sse, shift);
}

/* ---- av1_build_quantizer oracle --------------------------------------------
 * Marshals the REAL exported av1_build_quantizer (av1/encoder/av1_quantize.c)
 * into one flat int16 buffer so the Rust harness needs no C struct layout
 * knowledge. Output layout: 21 tables x QINDEX_RANGE x 8 lanes, QUANTS
 * declaration order then Dequants declaration order:
 *   [ 0] y_quant        [ 1] y_quant_shift   [ 2] y_zbin        [ 3] y_round
 *   [ 4] y_quant_fp     [ 5] u_quant_fp      [ 6] v_quant_fp
 *   [ 7] y_round_fp     [ 8] u_round_fp      [ 9] v_round_fp
 *   [10] u_quant        [11] v_quant         [12] u_quant_shift [13] v_quant_shift
 *   [14] u_zbin         [15] v_zbin          [16] u_round       [17] v_round
 *   [18] y_dequant_QTX  [19] u_dequant_QTX   [20] v_dequant_QTX
 * Returns 0 on success, -1 on allocation failure. */
int shim_build_quantizer(int bit_depth, int y_dc_delta_q, int u_dc_delta_q,
                         int u_ac_delta_q, int v_dc_delta_q, int v_ac_delta_q,
                         int sharpness, int16_t *out) {
  QUANTS *quants = (QUANTS *)malloc(sizeof(QUANTS));
  Dequants *deq = (Dequants *)malloc(sizeof(Dequants));
  if (!quants || !deq) {
    free(quants);
    free(deq);
    return -1;
  }
  av1_build_quantizer((aom_bit_depth_t)bit_depth, y_dc_delta_q, u_dc_delta_q,
                      u_ac_delta_q, v_dc_delta_q, v_ac_delta_q, quants, deq,
                      sharpness);
  {
    const size_t n = QINDEX_RANGE * 8;
    const int16_t *src[21] = {
      &quants->y_quant[0][0],       &quants->y_quant_shift[0][0],
      &quants->y_zbin[0][0],        &quants->y_round[0][0],
      &quants->y_quant_fp[0][0],    &quants->u_quant_fp[0][0],
      &quants->v_quant_fp[0][0],    &quants->y_round_fp[0][0],
      &quants->u_round_fp[0][0],    &quants->v_round_fp[0][0],
      &quants->u_quant[0][0],       &quants->v_quant[0][0],
      &quants->u_quant_shift[0][0], &quants->v_quant_shift[0][0],
      &quants->u_zbin[0][0],        &quants->v_zbin[0][0],
      &quants->u_round[0][0],       &quants->v_round[0][0],
      &deq->y_dequant_QTX[0][0],    &deq->u_dequant_QTX[0][0],
      &deq->v_dequant_QTX[0][0],
    };
    for (int a = 0; a < 21; ++a) {
      memcpy(out + (size_t)a * n, src[a], n * sizeof(int16_t));
    }
  }
  free(quants);
  free(deq);
  return 0;
}

/* ---- tx-type signaling cost ------------------------------------------------
 * (1) shim_fill_tx_type_costs: transcribes the tx-type slice of
 *     av1_fill_mode_rates (av1/encoder/rd.c) over the REAL exported
 *     av1_cost_tokens_from_cdf and the REAL av1_ext_tx_inv (entropymode.h).
 *     The three rd.c file-local gating tables below are transcribed verbatim
 *     (they are static in rd.c and not reachable any other way).
 * (2) shim_get_tx_type_cost: transcribes get_tx_type_cost
 *     (av1/encoder/txb_rdopt.c, static) over the REAL av1_get_ext_tx_set_type
 *     / av1_num_ext_tx_set / ext_tx_set_index / fimode_to_intradir /
 *     txsize_sqr_map (all header statics = pristine C recompiled); the
 *     MACROBLOCK(D) derefs are passed as scalars, the cost tables as the flat
 *     outputs of shim_fill_tx_type_costs. */
static const int sh_use_intra_ext_tx_for_txsize[EXT_TX_SETS_INTRA]
                                               [EXT_TX_SIZES] = {
                                                 { 1, 1, 1, 1 },  // unused
                                                 { 1, 1, 0, 0 },
                                                 { 0, 0, 1, 0 },
                                               };

static const int sh_use_inter_ext_tx_for_txsize[EXT_TX_SETS_INTER]
                                               [EXT_TX_SIZES] = {
                                                 { 1, 1, 1, 1 },  // unused
                                                 { 1, 1, 0, 0 },
                                                 { 0, 0, 1, 0 },
                                                 { 0, 1, 1, 1 },
                                               };

static const int sh_ext_tx_set_idx_to_type[2][4] = {
  {
      // Intra
      EXT_TX_SET_DCTONLY,
      EXT_TX_SET_DTT4_IDTX_1DDCT,
      EXT_TX_SET_DTT4_IDTX,
  },
  {
      // Inter
      EXT_TX_SET_DCTONLY,
      EXT_TX_SET_ALL16,
      EXT_TX_SET_DTT9_IDTX_1DDCT,
      EXT_TX_SET_DCT_IDTX,
  },
};

/* intra_cdf: flat [EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES][TX_TYPES+1];
 * inter_cdf: flat [EXT_TX_SETS_INTER][EXT_TX_SIZES][TX_TYPES+1].
 * out_intra: flat [EXT_TX_SETS_INTRA][EXT_TX_SIZES][INTRA_MODES][TX_TYPES];
 * out_inter: flat [EXT_TX_SETS_INTER][EXT_TX_SIZES][TX_TYPES].
 * Both outputs must be caller-zeroed (ungated combos stay 0 on both sides). */
void shim_fill_tx_type_costs(const uint16_t *intra_cdf,
                             const uint16_t *inter_cdf, int *out_intra,
                             int *out_inter) {
  for (int i = TX_4X4; i < EXT_TX_SIZES; ++i) {
    for (int s = 1; s < EXT_TX_SETS_INTER; ++s) {
      if (sh_use_inter_ext_tx_for_txsize[s][i]) {
        av1_cost_tokens_from_cdf(
            out_inter + (s * EXT_TX_SIZES + i) * TX_TYPES,
            inter_cdf + (s * EXT_TX_SIZES + i) * (TX_TYPES + 1),
            av1_ext_tx_inv[sh_ext_tx_set_idx_to_type[1][s]]);
      }
    }
    for (int s = 1; s < EXT_TX_SETS_INTRA; ++s) {
      if (sh_use_intra_ext_tx_for_txsize[s][i]) {
        for (int j = 0; j < INTRA_MODES; ++j) {
          av1_cost_tokens_from_cdf(
              out_intra + ((s * EXT_TX_SIZES + i) * INTRA_MODES + j) * TX_TYPES,
              intra_cdf +
                  ((s * EXT_TX_SIZES + i) * INTRA_MODES + j) * (TX_TYPES + 1),
              av1_ext_tx_inv[sh_ext_tx_set_idx_to_type[0][s]]);
        }
      }
    }
  }
}

/* ---- intra mode-info signaling costs ---------------------------------------
 * (1) shim_fill_intra_mode_costs: transcribes the intra-mode slices of
 *     av1_fill_mode_rates (rd.c) over the REAL exported av1_cost_tokens_from_cdf
 *     and the REAL av1_filter_intra_allowed_bsize (reconintra.h, driven by a
 *     minimal heap AV1_COMMON+SequenceHeader carrying enable_filter_intra).
 * (2) shim_intra_mode_info_cost_y: transcribes intra_mode_info_cost_y
 *     (av1/encoder/intra_mode_search_utils.h, static inline) for the
 *     palette_size[0]==0 path, over the REAL av1_is_directional_mode /
 *     av1_use_angle_delta / av1_filter_intra_allowed_bsize gates; the
 *     try_palette / palette ctx / allow_intrabc cm+xd derefs are scalars.
 *     The live assert mirrors the C's exclusive-mode-flag assert. */
static int sh_filter_intra_allowed_bsize(int enable_filter_intra, int bsize) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(AV1_COMMON));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(SequenceHeader));
  assert(cm && seq);
  seq->enable_filter_intra = (uint8_t)enable_filter_intra;
  cm->seq_params = seq;
  const int r = av1_filter_intra_allowed_bsize(cm, (BLOCK_SIZE)bsize);
  free(seq);
  free(cm);
  return r;
}

/* CDF inputs flat (rows are nsymbs-terminated inverse-CDFs, padded):
 *   kf_y_cdf [5][5][14], y_mode_cdf [4][14], uv_mode_cdf [2][13][15],
 *   filter_intra_mode_cdf [6], filter_intra_cdfs [22][3],
 *   palette_y_mode_cdf [7][3][3], angle_delta_cdf [8][8], intrabc_cdf [3].
 * Cost outputs flat, caller-zeroed:
 *   y_mode_costs [5][5][13], mbmode [4][13], uv [2][13][14], fi_mode [5],
 *   fi [22][2], pal_y_mode [7][3][2], angle [8][7], intrabc [2]. */
void shim_fill_intra_mode_costs(
    const uint16_t *kf_y_cdf, const uint16_t *y_mode_cdf,
    const uint16_t *uv_mode_cdf, const uint16_t *filter_intra_mode_cdf,
    const uint16_t *filter_intra_cdfs, const uint16_t *palette_y_mode_cdf,
    const uint16_t *angle_delta_cdf, const uint16_t *intrabc_cdf,
    int enable_filter_intra, int *out_y_mode, int *out_mbmode, int *out_uv,
    int *out_fi_mode, int *out_fi, int *out_pal_y_mode, int *out_angle,
    int *out_intrabc) {
  int i, j;
  for (i = 0; i < KF_MODE_CONTEXTS; ++i)
    for (j = 0; j < KF_MODE_CONTEXTS; ++j)
      av1_cost_tokens_from_cdf(
          out_y_mode + (i * KF_MODE_CONTEXTS + j) * INTRA_MODES,
          kf_y_cdf + (i * KF_MODE_CONTEXTS + j) * (INTRA_MODES + 1), NULL);

  for (i = 0; i < BLOCK_SIZE_GROUPS; ++i)
    av1_cost_tokens_from_cdf(out_mbmode + i * INTRA_MODES,
                             y_mode_cdf + i * (INTRA_MODES + 1), NULL);
  for (i = 0; i < CFL_ALLOWED_TYPES; ++i)
    for (j = 0; j < INTRA_MODES; ++j)
      av1_cost_tokens_from_cdf(
          out_uv + (i * INTRA_MODES + j) * UV_INTRA_MODES,
          uv_mode_cdf + (i * INTRA_MODES + j) * (UV_INTRA_MODES + 1), NULL);

  av1_cost_tokens_from_cdf(out_fi_mode, filter_intra_mode_cdf, NULL);
  for (i = 0; i < BLOCK_SIZES_ALL; ++i) {
    if (sh_filter_intra_allowed_bsize(enable_filter_intra, i))
      av1_cost_tokens_from_cdf(out_fi + i * 2, filter_intra_cdfs + i * 3,
                               NULL);
  }

  for (i = 0; i < PALATTE_BSIZE_CTXS; ++i) {
    for (j = 0; j < PALETTE_Y_MODE_CONTEXTS; ++j) {
      av1_cost_tokens_from_cdf(
          out_pal_y_mode + (i * PALETTE_Y_MODE_CONTEXTS + j) * 2,
          palette_y_mode_cdf + (i * PALETTE_Y_MODE_CONTEXTS + j) * 3, NULL);
    }
  }

  for (i = 0; i < DIRECTIONAL_MODES; ++i)
    av1_cost_tokens_from_cdf(out_angle + i * (2 * MAX_ANGLE_DELTA + 1),
                             angle_delta_cdf + i * (2 * MAX_ANGLE_DELTA + 2),
                             NULL);
  av1_cost_tokens_from_cdf(out_intrabc, intrabc_cdf, NULL);
}

int shim_intra_mode_info_cost_y(
    const int *filter_intra_cost, const int *filter_intra_mode_cost,
    const int *angle_delta_cost, const int *intrabc_cost,
    const int *palette_y_mode_cost, int mode_cost, int mode, int bsize,
    int angle_delta_y, int use_filter_intra, int filter_intra_mode,
    int use_intrabc, int try_palette, int palette_bsize_ctx,
    int palette_mode_ctx, int enable_filter_intra, int allow_intrabc) {
  int total_rate = mode_cost;
  const int use_palette = 0; /* scope: palette_size[0] == 0 */
  /* Can only activate one mode. */
  assert(((mode != DC_PRED) + use_palette + use_intrabc + use_filter_intra) <=
         1);
  if (try_palette && mode == DC_PRED) {
    total_rate +=
        palette_y_mode_cost[(palette_bsize_ctx * PALETTE_Y_MODE_CONTEXTS +
                             palette_mode_ctx) *
                                2 +
                            use_palette];
  }
  /* av1_filter_intra_allowed(cm, mbmi) with palette_size[0]==0 */
  if (mode == DC_PRED &&
      sh_filter_intra_allowed_bsize(enable_filter_intra, bsize)) {
    total_rate += filter_intra_cost[bsize * 2 + use_filter_intra];
    if (use_filter_intra) {
      total_rate += filter_intra_mode_cost[filter_intra_mode];
    }
  }
  if (av1_is_directional_mode((PREDICTION_MODE)mode)) {
    if (av1_use_angle_delta((BLOCK_SIZE)bsize)) {
      total_rate += angle_delta_cost[(mode - V_PRED) *
                                         (2 * MAX_ANGLE_DELTA + 1) +
                                     MAX_ANGLE_DELTA + angle_delta_y];
    }
  }
  if (allow_intrabc) total_rate += intrabc_cost[use_intrabc];
  return total_rate;
}

int shim_get_tx_type_cost(const int *intra_costs, const int *inter_costs,
                          int plane, int tx_size, int tx_type, int is_inter,
                          int reduced_tx_set_used, int lossless,
                          int use_filter_intra, int filter_intra_mode,
                          int mode) {
  if (plane > 0) return 0;

  const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];

  const TxSetType set_type = av1_get_ext_tx_set_type(
      (TX_SIZE)tx_size, is_inter, reduced_tx_set_used);
  if (av1_num_ext_tx_set[set_type] > 1 && !lossless) {
    const int ext_tx_set = ext_tx_set_index[is_inter][set_type];
    if (is_inter) {
      if (ext_tx_set > 0)
        return inter_costs[(ext_tx_set * EXT_TX_SIZES + square_tx_size) *
                               TX_TYPES +
                           tx_type];
    } else {
      if (ext_tx_set > 0) {
        PREDICTION_MODE intra_dir;
        if (use_filter_intra)
          intra_dir = fimode_to_intradir[filter_intra_mode];
        else
          intra_dir = (PREDICTION_MODE)mode;
        return intra_costs[((ext_tx_set * EXT_TX_SIZES + square_tx_size) *
                                INTRA_MODES +
                            intra_dir) *
                               TX_TYPES +
                           tx_type];
      }
    }
  }
  return 0;
}

/* ---- set_q_index / av1_init_plane_quantizers -------------------------------
 * (1) shim_set_q_index: transcribes the static set_q_index
 *     (av1/encoder/av1_quantize.c) — the per-qindex row selection out of
 *     QUANTS/Dequants into the MACROBLOCK plane fields — over tables filled by
 *     the REAL exported av1_build_quantizer. Out layout: [plane 0..3][7][8],
 *     row order = the C assignment order per plane:
 *       [0] quant_QTX  [1] quant_fp_QTX [2] round_fp_QTX [3] quant_shift_QTX
 *       [4] zbin_QTX   [5] round_QTX    [6] dequant_QTX
 * (2) shim_get_qindex: the REAL exported av1_get_qindex over a stack
 *     struct segmentation (enabled + feature_mask/feature_data marshalled).
 * (3) shim_error_per_bit: the REAL av1_set_error_per_bit (rd.h static inline =
 *     pristine C recompiled in this TU).
 * (4) shim_sad_per_bit: transcribes the sad_per_bit_lut_* entry formula
 *     (init_me_luts_bd, av1/encoder/rd.c — file-static tables) over the REAL
 *     exported av1_convert_qindex_to_q. */
int shim_set_q_index(int bit_depth, int y_dc_delta_q, int u_dc_delta_q,
                     int u_ac_delta_q, int v_dc_delta_q, int v_ac_delta_q,
                     int sharpness, int qindex, int16_t *out) {
  QUANTS *quants = (QUANTS *)malloc(sizeof(QUANTS));
  Dequants *deq = (Dequants *)malloc(sizeof(Dequants));
  if (!quants || !deq) {
    free(quants);
    free(deq);
    return -1;
  }
  av1_build_quantizer((aom_bit_depth_t)bit_depth, y_dc_delta_q, u_dc_delta_q,
                      u_ac_delta_q, v_dc_delta_q, v_ac_delta_q, quants, deq,
                      sharpness);
  {
    /* set_q_index body (av1_quantize.c), Y/U/V blocks verbatim: the seven
     * row pointers each plane's MACROBLOCK_PLANE receives for `qindex`. */
    const int16_t *rows[3][7] = {
      { quants->y_quant[qindex], quants->y_quant_fp[qindex],
        quants->y_round_fp[qindex], quants->y_quant_shift[qindex],
        quants->y_zbin[qindex], quants->y_round[qindex],
        deq->y_dequant_QTX[qindex] },
      { quants->u_quant[qindex], quants->u_quant_fp[qindex],
        quants->u_round_fp[qindex], quants->u_quant_shift[qindex],
        quants->u_zbin[qindex], quants->u_round[qindex],
        deq->u_dequant_QTX[qindex] },
      { quants->v_quant[qindex], quants->v_quant_fp[qindex],
        quants->v_round_fp[qindex], quants->v_quant_shift[qindex],
        quants->v_zbin[qindex], quants->v_round[qindex],
        deq->v_dequant_QTX[qindex] },
    };
    for (int p = 0; p < 3; ++p) {
      for (int r = 0; r < 7; ++r) {
        memcpy(out + ((size_t)p * 7 + r) * 8, rows[p][r], 8 * sizeof(int16_t));
      }
    }
  }
  free(quants);
  free(deq);
  return 0;
}

int shim_get_qindex(int enabled, const uint32_t *feature_mask,
                    const int16_t *altq_data, int segment_id,
                    int base_qindex) {
  struct segmentation seg;
  memset(&seg, 0, sizeof(seg));
  seg.enabled = (uint8_t)enabled;
  for (int i = 0; i < MAX_SEGMENTS; ++i) {
    seg.feature_mask[i] = feature_mask[i];
    seg.feature_data[i][SEG_LVL_ALT_Q] = altq_data[i];
  }
  return av1_get_qindex(&seg, segment_id, base_qindex);
}

int shim_error_per_bit(int rdmult) {
  int errorperbit = 0;
  av1_set_error_per_bit(&errorperbit, rdmult);
  return errorperbit;
}

int shim_sad_per_bit(int qindex, int bit_depth) {
  /* init_me_luts_bd entry (rd.c): (int)(0.0418 * q + 2.4107) over the REAL
   * av1_convert_qindex_to_q; av1_set_sad_per_bit is a plain lut lookup. */
  const double q = av1_convert_qindex_to_q(qindex, (aom_bit_depth_t)bit_depth);
  return (int)(0.0418 * q + 2.4107);
}

/* shim_quant_plane_rows: quantize through the REAL exported quantize facades
 * (av1_[highbd_]quantize_{fp,b,dc}_facade) with a MACROBLOCK_PLANE whose seven
 * QTX rows are installed exactly as set_q_index installs them — so the FACADE
 * (not the caller) picks which rows each quantizer kind reads. This is the
 * oracle for the Rust QuantParams::from_plane_rows row-choice bridge.
 * `rows` = the [7][8] one-plane blob in shim_set_q_index order; `kind` matches
 * quant_func_list: 0 = AV1_XFORM_QUANT_FP, 1 = _B, 2 = _DC. Flat (no qmatrix).
 * The FP/B facades dispatch RTCD kernels — call ref_init() first. */
uint16_t shim_quant_plane_rows(int kind, int is_hbd, const int32_t *coeff,
                               int n, const int16_t *rows, const int16_t *scan,
                               const int16_t *iscan, int log_scale,
                               int32_t *qcoeff, int32_t *dqcoeff) {
  /* The AVX2 quantizers RTCD dispatches to use ALIGNED 32-byte loads/stores
   * (DECLARE_ALIGNED buffers in libaom); bounce through 32-byte-aligned
   * copies so the caller's buffers can have any alignment. */
  const size_t bytes = ((size_t)n * sizeof(int32_t) + 31) & ~(size_t)31;
  int32_t *acoeff = (int32_t *)aligned_alloc(32, bytes);
  int32_t *aq = (int32_t *)aligned_alloc(32, bytes);
  int32_t *adq = (int32_t *)aligned_alloc(32, bytes);
  if (!acoeff || !aq || !adq) {
    free(acoeff);
    free(aq);
    free(adq);
    return 0xffff;
  }
  memcpy(acoeff, coeff, (size_t)n * sizeof(int32_t));
  memset(aq, 0, bytes);
  memset(adq, 0, bytes);
  MACROBLOCK_PLANE p;
  memset(&p, 0, sizeof(p));
  p.quant_QTX = rows + 0 * 8;
  p.quant_fp_QTX = rows + 1 * 8;
  p.round_fp_QTX = rows + 2 * 8;
  p.quant_shift_QTX = rows + 3 * 8;
  p.zbin_QTX = rows + 4 * 8;
  p.round_QTX = rows + 5 * 8;
  p.dequant_QTX = rows + 6 * 8;
  SCAN_ORDER sc = { scan, iscan };
  QUANT_PARAM qparam;
  memset(&qparam, 0, sizeof(qparam));
  qparam.log_scale = log_scale;
  qparam.qmatrix = NULL;
  qparam.iqmatrix = NULL;
  uint16_t eob = 0;
  if (is_hbd) {
    switch (kind) {
      case 0:
        av1_highbd_quantize_fp_facade(acoeff, (intptr_t)n, &p, aq, adq,
                                      &eob, &sc, &qparam);
        break;
      case 1:
        av1_highbd_quantize_b_facade(acoeff, (intptr_t)n, &p, aq, adq,
                                     &eob, &sc, &qparam);
        break;
      default:
        av1_highbd_quantize_dc_facade(acoeff, (intptr_t)n, &p, aq, adq,
                                      &eob, &sc, &qparam);
        break;
    }
  } else {
    switch (kind) {
      case 0:
        av1_quantize_fp_facade(acoeff, (intptr_t)n, &p, aq, adq, &eob,
                               &sc, &qparam);
        break;
      case 1:
        av1_quantize_b_facade(acoeff, (intptr_t)n, &p, aq, adq, &eob,
                              &sc, &qparam);
        break;
      default:
        av1_quantize_dc_facade(acoeff, (intptr_t)n, &p, aq, adq, &eob,
                               &sc, &qparam);
        break;
    }
  }
  memcpy(qcoeff, aq, (size_t)n * sizeof(int32_t));
  memcpy(dqcoeff, adq, (size_t)n * sizeof(int32_t));
  free(acoeff);
  free(aq);
  free(adq);
  return eob;
}

/* ---- av1_rd_pick_intra_sby_mode candidate loop head -------------------------
 * (1) shim_set_y_mode_and_delta_angle: the REAL EXPORTED
 *     set_y_mode_and_delta_angle (av1/encoder/intra_mode_search.c) over a
 *     stack MB_MODE_INFO; returns mode and writes the delta.
 * (2) shim_intra_sby_visits: transcribes the candidate loop's static skip
 *     chain (intra_mode_search.c 1555-1594) over the REAL header statics
 *     av1_is_diagonal_mode / av1_is_directional_mode / av1_use_angle_delta
 *     (reconintra.h) and max_txsize_lookup (common_data.h). The dynamic
 *     model-RD prune is separate; use_mb_mode_cache is modelled off.
 * (3) shim_prune_odd_delta: transcribes
 *     prune_luma_odd_delta_angles_using_rd_cost over a passed rd-cost array
 *     (the fn is static and pure). */
void set_y_mode_and_delta_angle(const int mode_idx, MB_MODE_INFO *const mbmi,
                                int reorder_delta_angle_eval);

int shim_set_y_mode_and_delta_angle(int mode_idx, int reorder,
                                    int *out_delta) {
  MB_MODE_INFO mbmi;
  memset(&mbmi, 0, sizeof(mbmi));
  set_y_mode_and_delta_angle(mode_idx, &mbmi, reorder);
  *out_delta = mbmi.angle_delta[PLANE_TYPE_Y];
  return mbmi.mode;
}

int shim_intra_sby_visits(int mode, int luma_delta_angle, int bsize,
                          int enable_diagonal_intra,
                          int enable_directional_intra, int enable_smooth_intra,
                          int enable_paeth_intra, int enable_angle_delta,
                          int disable_smooth_intra, int prune_filter_intra_level,
                          const uint16_t *intra_y_mode_mask /*[5]*/,
                          const uint8_t *directional_mode_skip_mask /*[13]*/) {
  const PREDICTION_MODE m = (PREDICTION_MODE)mode;
  const int is_diagonal_mode = av1_is_diagonal_mode(m);
  if (is_diagonal_mode && !enable_diagonal_intra) return 0;
  if (av1_is_directional_mode(m) && !enable_directional_intra) return 0;
  if ((!enable_smooth_intra || disable_smooth_intra) &&
      (m == SMOOTH_H_PRED || m == SMOOTH_V_PRED))
    return 0;
  if (!enable_smooth_intra && m == SMOOTH_PRED) return 0;
  if (disable_smooth_intra && prune_filter_intra_level == 0 &&
      m == SMOOTH_PRED)
    return 0;
  if (!enable_paeth_intra && m == PAETH_PRED) return 0;
  const int is_directional_mode = av1_is_directional_mode(m);
  if (is_directional_mode && directional_mode_skip_mask[m]) return 0;
  if (is_directional_mode &&
      !(av1_use_angle_delta((BLOCK_SIZE)bsize) && enable_angle_delta) &&
      luma_delta_angle != 0)
    return 0;
  if (!(intra_y_mode_mask[max_txsize_lookup[bsize]] & (1 << m))) return 0;
  return 1;
}

int shim_prune_odd_delta(int mode, int luma_delta_angle,
                         const int64_t *intra_modes_rd_cost /*[9]*/,
                         int64_t best_rd,
                         int prune_luma_odd_delta_angles_in_intra) {
  if (!prune_luma_odd_delta_angles_in_intra ||
      !av1_is_directional_mode((PREDICTION_MODE)mode) ||
      !(abs(luma_delta_angle) & 1) || best_rd == INT64_MAX)
    return 0;
  const int64_t rd_thresh = best_rd + (best_rd >> 3);
  return intra_modes_rd_cost[luma_delta_angle + MAX_ANGLE_DELTA] > rd_thresh &&
         intra_modes_rd_cost[luma_delta_angle + MAX_ANGLE_DELTA + 2] >
             rd_thresh;
}

/* ---- search_tx_type building blocks -----------------------------------------
 * (1) shim_get_tx_mask_intra: transcribes the LUMA INTRA arm of get_tx_mask
 *     (av1/encoder/tx_search.c, static) over the REAL exported
 *     av1_get_ext_tx_set_type and the REAL blockd.h tables
 *     (av1_ext_tx_used_flag / av1_reduced_intra_tx_used_flag /
 *     av1_derived_intra_tx_used_flag / fimode_to_intradir — header statics =
 *     pristine C recompiled). Structurally-off arms (inter forcing, stats
 *     prune, est-rd prune, prune_tx_2D, use_default_intra_tx_type,
 *     LOW_TXFM_RD) are omitted per the speed-0 intra contract.
 *     Returns the mask; *out_txk_allowed = TX_TYPES or the single type.
 * (2) shim_pixel_diff_dist: the REAL EXPORTED av1_pixel_diff_dist over a
 *     calloc'd MACROBLOCK (pure marshalling: src_diff pointer, frame-edge
 *     fields, plane subsampling). aom_sum_squares_2d_i16 is RTCD-dispatched:
 *     call ref_init() first. */
int shim_get_tx_mask_intra(int tx_size, int mode, int use_filter_intra,
                           int filter_intra_mode, int lossless,
                           int reduced_tx_set_used,
                           int use_reduced_intra_txset,
                           int use_derived_intra_tx_type_set,
                           int enable_flip_idtx, int use_intra_dct_only,
                           int *out_txk_allowed) {
  TX_TYPE txk_allowed = TX_TYPES;
  const TxSetType tx_set_type = av1_get_ext_tx_set_type(
      (TX_SIZE)tx_size, /*is_inter=*/0, reduced_tx_set_used);

  PREDICTION_MODE intra_dir;
  if (use_filter_intra)
    intra_dir = fimode_to_intradir[filter_intra_mode];
  else
    intra_dir = (PREDICTION_MODE)mode;
  uint16_t ext_tx_used_flag =
      use_reduced_intra_txset != 0 && tx_set_type == EXT_TX_SET_DTT4_IDTX_1DDCT
          ? av1_reduced_intra_tx_used_flag[intra_dir]
          : av1_ext_tx_used_flag[tx_set_type];
  if (use_reduced_intra_txset == 2)
    ext_tx_used_flag &= av1_derived_intra_tx_used_flag[intra_dir];

  if (lossless || txsize_sqr_up_map[tx_size] > TX_32X32 ||
      ext_tx_used_flag == 0x0001 || use_intra_dct_only) {
    txk_allowed = DCT_DCT;
  }
  if (enable_flip_idtx == 0) ext_tx_used_flag &= DCT_ADST_TX_MASK;

  uint16_t allowed_tx_mask = 0;
  if (txk_allowed < TX_TYPES) {
    allowed_tx_mask = 1 << txk_allowed;
    allowed_tx_mask &= ext_tx_used_flag;
  } else if (use_derived_intra_tx_type_set) {
    allowed_tx_mask = av1_derived_intra_tx_used_flag[intra_dir];
    allowed_tx_mask &= ext_tx_used_flag;
  } else {
    allowed_tx_mask = ext_tx_used_flag;
  }

  if (allowed_tx_mask == 0) {
    txk_allowed = DCT_DCT;
    allowed_tx_mask = (1 << txk_allowed);
  }
  *out_txk_allowed = txk_allowed;
  return allowed_tx_mask;
}

int64_t av1_pixel_diff_dist(const MACROBLOCK *x, int plane, int blk_row,
                            int blk_col, const BLOCK_SIZE plane_bsize,
                            const BLOCK_SIZE tx_bsize,
                            unsigned int *block_mse_q8);

int64_t shim_pixel_diff_dist(const int16_t *src_diff, int n_diff,
                             int plane_bsize, int tx_bsize, int blk_row,
                             int blk_col, int mb_to_right_edge,
                             int mb_to_bottom_edge, int subsampling_x,
                             int subsampling_y, uint32_t *out_mse_q8) {
  MACROBLOCK *x = (MACROBLOCK *)calloc(1, sizeof(MACROBLOCK));
  /* libaom's src_diff lives in a DECLARE_ALIGNED(32) buffer and the SSE2
   * sum-squares kernel uses ALIGNED loads (xx_load_128) — bounce through a
   * 32-byte-aligned copy so the caller's buffer can have any alignment. */
  const size_t bytes = ((size_t)n_diff * sizeof(int16_t) + 31) & ~(size_t)31;
  int16_t *adiff = (int16_t *)aligned_alloc(32, bytes);
  if (!x || !adiff) {
    free(x);
    free(adiff);
    return -1;
  }
  memcpy(adiff, src_diff, (size_t)n_diff * sizeof(int16_t));
  x->plane[0].src_diff = adiff;
  x->e_mbd.mb_to_right_edge = mb_to_right_edge;
  x->e_mbd.mb_to_bottom_edge = mb_to_bottom_edge;
  x->e_mbd.plane[0].subsampling_x = subsampling_x;
  x->e_mbd.plane[0].subsampling_y = subsampling_y;
  unsigned int mse = 0;
  int64_t sse =
      av1_pixel_diff_dist(x, /*plane=*/0, blk_row, blk_col,
                          (BLOCK_SIZE)plane_bsize, (BLOCK_SIZE)tx_bsize, &mse);
  *out_mse_q8 = mse;
  free(adiff);
  free(x);
  return sse;
}

/* ---- tx-size signaling cost -------------------------------------------------
 * shim_fill_tx_size_costs: the tx-size slice of av1_fill_mode_rates (rd.c
 * 175-178) over the REAL exported av1_cost_tokens_from_cdf.
 * shim_tx_size_cost: transcribes tx_size_cost (tx_search.h) over the REAL
 * header statics bsize_to_tx_size_cat / tx_size_to_depth /
 * block_signals_txsize; the get_tx_size_context neighbour facade is the
 * caller's (tx_size_ctx param), mirroring the Rust deferral. */
void shim_fill_tx_size_costs(const uint16_t *tx_size_cdf /*[4][3][4]*/,
                             int *out /*[4][3][3]*/) {
  for (int cat = 0; cat < MAX_TX_CATS; ++cat) {
    for (int ctx = 0; ctx < TX_SIZE_CONTEXTS; ++ctx) {
      av1_cost_tokens_from_cdf(out + (cat * TX_SIZE_CONTEXTS + ctx) * 3,
                               tx_size_cdf + (cat * TX_SIZE_CONTEXTS + ctx) * 4,
                               NULL);
    }
  }
}

int shim_tx_size_cost(const int *costs /*[4][3][3]*/, int tx_mode_is_select,
                      int bsize, int tx_size, int tx_size_ctx) {
  if (!tx_mode_is_select || !block_signals_txsize((BLOCK_SIZE)bsize)) return 0;
  const int32_t tx_size_cat = bsize_to_tx_size_cat((BLOCK_SIZE)bsize);
  const int depth = tx_size_to_depth((TX_SIZE)tx_size, (BLOCK_SIZE)bsize);
  return costs[(tx_size_cat * TX_SIZE_CONTEXTS + tx_size_ctx) * 3 + depth];
}

/* ---- model-rd prune (intra_mode_search.c statics) ---------------------------
 * get_model_rd_index_for_pruning + prune_intra_y_mode are static; both bodies
 * are transcribed VERBATIM (intra_mode_search.c:423-492) so the double
 * threshold math (1.50 / 1.00 * int64 model rd) is compiled as C. Neighbour
 * state is passed as plain ints (avail flag + mode); left/above mode
 * comparisons default 0 when unavailable, as the C's guarded reads do. */
int shim_get_model_rd_index_for_pruning(int cur_mode, int qindex,
                                        int top_intra_model_count_allowed,
                                        int adapt_top_model_rd_count_using_neighbors,
                                        int left_available, int left_mode,
                                        int up_available, int above_mode) {
  if (!adapt_top_model_rd_count_using_neighbors)
    return top_intra_model_count_allowed - 1;

  const int mode = cur_mode;
  int model_rd_index_for_pruning = top_intra_model_count_allowed - 1;
  int is_left_mode_neq_cur_mode = 0, is_above_mode_neq_cur_mode = 0;
  if (left_available) is_left_mode_neq_cur_mode = left_mode != mode;
  if (up_available) is_above_mode_neq_cur_mode = above_mode != mode;
  if (qindex <= 127) {
    if (is_left_mode_neq_cur_mode || is_above_mode_neq_cur_mode)
      model_rd_index_for_pruning = AOMMAX(model_rd_index_for_pruning - 1, 0);
  } else {
    if (is_left_mode_neq_cur_mode && is_above_mode_neq_cur_mode)
      model_rd_index_for_pruning = AOMMAX(model_rd_index_for_pruning - 1, 0);
  }
  return model_rd_index_for_pruning;
}

int shim_prune_intra_y_mode(int64_t this_model_rd, int64_t *best_model_rd,
                            int64_t top_intra_model_rd[],
                            int max_model_cnt_allowed,
                            int model_rd_index_for_pruning) {
  const double thresh_best = 1.50;
  const double thresh_top = 1.00;
  for (int i = 0; i < max_model_cnt_allowed; i++) {
    if (this_model_rd < top_intra_model_rd[i]) {
      for (int j = max_model_cnt_allowed - 1; j > i; j--) {
        top_intra_model_rd[j] = top_intra_model_rd[j - 1];
      }
      top_intra_model_rd[i] = this_model_rd;
      break;
    }
  }
  if (top_intra_model_rd[model_rd_index_for_pruning] != INT64_MAX &&
      this_model_rd >
          thresh_top * top_intra_model_rd[model_rd_index_for_pruning])
    return 1;

  if (this_model_rd != INT64_MAX &&
      this_model_rd > thresh_best * (*best_model_rd))
    return 1;
  if (this_model_rd < *best_model_rd) *best_model_rd = this_model_rd;
  return 0;
}

/* ---- intra_rd_variance_factor (intra_mode_search.c statics) -----------------
 * intra_rd_variance_factor + compute_avg_log_variance + the
 * av1_calc_normalized_variance per-4x4 kernel are transcribed VERBATIM over
 * the REAL aom_variance4x4_c / aom_highbd_{8,10,12}_variance4x4_c kernels and
 * libm log1p, so every double op comes from the C compiler. fn_ptr resolution:
 * the production encoder's fn_ptr[BLOCK_4X4].vf is aom_variance4x4 for 8-bit
 * streams (lowbd u8 planes) and aom_highbd_<bd>_variance4x4 for bd > 8
 * (CONVERT_TO_BYTEPTR'd u16 planes) — mirrored here by bd. The 8-bit path
 * copies the u16 block windows into u8 planes at the same stride (the
 * production layout). The 4x4 source-var cache (var / log_var per mi in the
 * superblock, init -1 / -1.0) is caller state, in/out. */
#include <math.h>

uint32_t aom_variance4x4_c(const uint8_t *, int, const uint8_t *, int,
                           uint32_t *);
uint32_t aom_highbd_8_variance4x4_c(const uint8_t *, int, const uint8_t *, int,
                                    uint32_t *);
uint32_t aom_highbd_10_variance4x4_c(const uint8_t *, int, const uint8_t *,
                                     int, uint32_t *);
uint32_t aom_highbd_12_variance4x4_c(const uint8_t *, int, const uint8_t *,
                                     int, uint32_t *);

static const uint8_t shim_var_all_zeros[128] = { 0 };
static const uint16_t shim_var_hbd_all_zeros[128] = { 0 };

static int shim_calc_normalized_variance_u8(const uint8_t *buf, int stride) {
  unsigned int sse;
  return aom_variance4x4_c(buf, stride, shim_var_all_zeros, 0, &sse);
}

static int shim_calc_normalized_variance_hbd(const uint16_t *buf, int stride,
                                             int bd) {
  unsigned int sse;
  const uint8_t *p = CONVERT_TO_BYTEPTR(buf);
  const uint8_t *z = CONVERT_TO_BYTEPTR(shim_var_hbd_all_zeros);
  if (bd == 12) return aom_highbd_12_variance4x4_c(p, stride, z, 0, &sse);
  if (bd == 10) return aom_highbd_10_variance4x4_c(p, stride, z, 0, &sse);
  return aom_highbd_8_variance4x4_c(p, stride, z, 0, &sse);
}

double shim_intra_rd_variance_factor(
    int speed, const uint16_t *src, int src_off, int src_stride,
    const uint16_t *recon, int ref_off, int ref_stride, int bsize, int sb_size,
    int mi_row, int mi_col, int mb_to_right_edge, int mb_to_bottom_edge,
    int bd, int *cache_var, double *cache_log_var) {
  double threshold = 1.0 - (0.25 * speed); /* INTRA_RD_VAR_THRESH */
  if (threshold <= 0) return 1.0;

  double variance_rd_factor = 1.0;
  double avg_log_src_variance = 0.0;
  double avg_log_recon_variance = 0.0;
  double var_diff = 0.0;

  /* compute_avg_log_variance */
  const int mi_row_in_sb = mi_row & (mi_size_high[sb_size] - 1);
  const int mi_col_in_sb = mi_col & (mi_size_wide[sb_size] - 1);
  const int right_overflow =
      (mb_to_right_edge < 0) ? ((-mb_to_right_edge) >> 3) : 0;
  const int bottom_overflow =
      (mb_to_bottom_edge < 0) ? ((-mb_to_bottom_edge) >> 3) : 0;
  const int bw = (MI_SIZE * mi_size_wide[bsize] - right_overflow);
  const int bh = (MI_SIZE * mi_size_high[bsize] - bottom_overflow);
  const int is_hbd = bd > 8;

  uint8_t *src8 = NULL, *rec8 = NULL;
  if (!is_hbd) {
    src8 = (uint8_t *)calloc((size_t)bh * src_stride, 1);
    rec8 = (uint8_t *)calloc((size_t)bh * ref_stride, 1);
    for (int i = 0; i < bh; i++) {
      for (int j = 0; j < bw; j++) {
        src8[i * src_stride + j] = (uint8_t)src[src_off + i * src_stride + j];
        rec8[i * ref_stride + j] = (uint8_t)recon[ref_off + i * ref_stride + j];
      }
    }
  }

  for (int i = 0; i < bh; i += MI_SIZE) {
    const int r = mi_row_in_sb + (i >> MI_SIZE_LOG2);
    for (int j = 0; j < bw; j += MI_SIZE) {
      const int c = mi_col_in_sb + (j >> MI_SIZE_LOG2);
      const int mi_offset = r * mi_size_wide[sb_size] + c;
      int src_var = cache_var[mi_offset];
      double log_src_var = cache_log_var[mi_offset];
      if (src_var < 0) {
        src_var = is_hbd ? shim_calc_normalized_variance_hbd(
                               src + src_off + i * src_stride + j, src_stride,
                               bd)
                         : shim_calc_normalized_variance_u8(
                               src8 + i * src_stride + j, src_stride);
        cache_var[mi_offset] = src_var;
        log_src_var = log1p(src_var / 16.0);
        cache_log_var[mi_offset] = log_src_var;
      } else {
        if (log_src_var < 0) {
          log_src_var = log1p(src_var / 16.0);
          cache_log_var[mi_offset] = log_src_var;
        }
      }
      avg_log_src_variance += log_src_var;

      const int recon_var =
          is_hbd ? shim_calc_normalized_variance_hbd(
                       recon + ref_off + i * ref_stride + j, ref_stride, bd)
                 : shim_calc_normalized_variance_u8(rec8 + i * ref_stride + j,
                                                    ref_stride);
      avg_log_recon_variance += log1p(recon_var / 16.0);
    }
  }

  const int blocks = (bw * bh) / 16;
  avg_log_src_variance /= (double)blocks;
  avg_log_recon_variance /= (double)blocks;
  free(src8);
  free(rec8);

  /* intra_rd_variance_factor tail */
  avg_log_src_variance += 0.000001;
  avg_log_recon_variance += 0.000001;

  if (avg_log_src_variance >= avg_log_recon_variance) {
    var_diff = (avg_log_src_variance - avg_log_recon_variance);
    if ((var_diff > 0.5) && (avg_log_recon_variance < threshold)) {
      variance_rd_factor = 1.0 + ((var_diff * 2) / avg_log_src_variance);
    }
  } else {
    var_diff = (avg_log_recon_variance - avg_log_src_variance);
    if ((var_diff > 0.5) && (avg_log_src_variance < threshold)) {
      variance_rd_factor = 1.0 + (var_diff / (2 * avg_log_src_variance));
    }
  }

  variance_rd_factor = AOMMIN(3.0, variance_rd_factor);

  return variance_rd_factor;
}

/* ---- chroma intra RD: UV tx type / tx mask / CfL costs / UV mode-info cost --
 * (1) shim_get_tx_type_uv_intra: the REAL av1_get_tx_type (blockd.h static
 *     inline = pristine C recompiled in this TU) over a calloc'd MACROBLOCKD
 *     stub — PLANE_TYPE_UV, intra block (ref_frame[0] = INTRA_FRAME,
 *     use_intrabc = 0). Marshals uv_mode + lossless only (the intra UV arm
 *     reads nothing else; blk_row/col and tx_type_map are inter-only).
 * (2) shim_get_tx_mask_uv_intra: the CHROMA arm of get_tx_mask (tx_search.c,
 *     static) transcribed over the REAL av1_get_tx_type + real header statics
 *     (av1_ext_tx_used_flag / av1_reduced_intra_tx_used_flag /
 *     fimode_to_intradir). Structurally-off arms omitted per the speed-0
 *     intra contract (inter forcing, stats/est-rd/2D prunes,
 *     use_default_intra_tx_type, LOW_TXFM_RD). NOTE the chroma empty-mask
 *     reset keeps uv_tx_type (tx_search.c:1942-46), unlike luma's DCT_DCT.
 * (3) shim_fill_cfl_costs: the CfL slice of av1_fill_mode_rates (rd.c
 *     154-172) over the REAL av1_cost_tokens_from_cdf + real CFL_* macros.
 *     out layout [CFL_JOINT_SIGNS=8][CFL_PRED_PLANES=2][CFL_ALPHABET_SIZE=16];
 *     sign_cdf one padded row (9), alpha_cdf flat [6][17].
 * (4) shim_fill_palette_uv_mode_costs: rd.c palette_uv_mode_cost fill
 *     ([2][3] cdf -> [2][2]).
 * (5) shim_intra_mode_info_cost_uv: transcribed intra_mode_info_cost_uv
 *     (intra_mode_search_utils.h) for the palette_size[1]==0 path over the
 *     REAL header gates (get_uv_mode / av1_is_directional_mode /
 *     av1_use_angle_delta). */
int shim_get_tx_type_uv_intra(int uv_mode, int lossless, int tx_size,
                              int reduced_tx_set_used) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(MACROBLOCKD));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(MB_MODE_INFO));
  if (!xd || !mbmi) {
    free(xd);
    free(mbmi);
    return -1;
  }
  MB_MODE_INFO *mi_ptr = mbmi;
  xd->mi = &mi_ptr;
  mbmi->uv_mode = (UV_PREDICTION_MODE)uv_mode;
  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->segment_id = 0;
  xd->lossless[0] = lossless;
  const TX_TYPE t = av1_get_tx_type(xd, PLANE_TYPE_UV, /*blk_row=*/0,
                                    /*blk_col=*/0, (TX_SIZE)tx_size,
                                    reduced_tx_set_used);
  free(xd);
  free(mbmi);
  return (int)t;
}

int shim_get_tx_mask_uv_intra(int tx_size, int uv_mode, int luma_mode,
                              int luma_use_filter_intra,
                              int luma_filter_intra_mode, int lossless,
                              int reduced_tx_set_used,
                              int use_reduced_intra_txset,
                              int enable_flip_idtx, int use_intra_dct_only,
                              int *out_txk_allowed) {
  const int uv_tx_type =
      shim_get_tx_type_uv_intra(uv_mode, lossless, tx_size,
                                reduced_tx_set_used);
  TX_TYPE txk_allowed = (TX_TYPE)uv_tx_type;
  const TxSetType tx_set_type = av1_get_ext_tx_set_type(
      (TX_SIZE)tx_size, /*is_inter=*/0, reduced_tx_set_used);

  PREDICTION_MODE intra_dir;
  if (luma_use_filter_intra)
    intra_dir = fimode_to_intradir[luma_filter_intra_mode];
  else
    intra_dir = (PREDICTION_MODE)luma_mode;
  uint16_t ext_tx_used_flag =
      use_reduced_intra_txset != 0 && tx_set_type == EXT_TX_SET_DTT4_IDTX_1DDCT
          ? av1_reduced_intra_tx_used_flag[intra_dir]
          : av1_ext_tx_used_flag[tx_set_type];
  if (use_reduced_intra_txset == 2)
    ext_tx_used_flag &= av1_derived_intra_tx_used_flag[intra_dir];

  if (lossless || txsize_sqr_up_map[tx_size] > TX_32X32 ||
      ext_tx_used_flag == 0x0001 || use_intra_dct_only) {
    txk_allowed = DCT_DCT;
  }
  if (enable_flip_idtx == 0) ext_tx_used_flag &= DCT_ADST_TX_MASK;

  uint16_t allowed_tx_mask = (1 << txk_allowed) & ext_tx_used_flag;
  if (allowed_tx_mask == 0) {
    txk_allowed = (TX_TYPE)uv_tx_type; /* plane ? uv_tx_type : DCT_DCT */
    allowed_tx_mask = (1 << txk_allowed);
  }
  *out_txk_allowed = txk_allowed;
  return allowed_tx_mask;
}

void shim_fill_cfl_costs(const uint16_t *cfl_sign_cdf,
                         const uint16_t *cfl_alpha_cdf, int *out) {
  int sign_cost[CFL_JOINT_SIGNS];
  av1_cost_tokens_from_cdf(sign_cost, cfl_sign_cdf, NULL);
  for (int joint_sign = 0; joint_sign < CFL_JOINT_SIGNS; joint_sign++) {
    int *cost_u = out + (joint_sign * CFL_PRED_PLANES + CFL_PRED_U) *
                            CFL_ALPHABET_SIZE;
    int *cost_v = out + (joint_sign * CFL_PRED_PLANES + CFL_PRED_V) *
                            CFL_ALPHABET_SIZE;
    if (CFL_SIGN_U(joint_sign) == CFL_SIGN_ZERO) {
      memset(cost_u, 0, CFL_ALPHABET_SIZE * sizeof(*cost_u));
    } else {
      const aom_cdf_prob *cdf_u =
          cfl_alpha_cdf + CFL_CONTEXT_U(joint_sign) * (CFL_ALPHABET_SIZE + 1);
      av1_cost_tokens_from_cdf(cost_u, cdf_u, NULL);
    }
    if (CFL_SIGN_V(joint_sign) == CFL_SIGN_ZERO) {
      memset(cost_v, 0, CFL_ALPHABET_SIZE * sizeof(*cost_v));
    } else {
      const aom_cdf_prob *cdf_v =
          cfl_alpha_cdf + CFL_CONTEXT_V(joint_sign) * (CFL_ALPHABET_SIZE + 1);
      av1_cost_tokens_from_cdf(cost_v, cdf_v, NULL);
    }
    for (int u = 0; u < CFL_ALPHABET_SIZE; u++)
      cost_u[u] += sign_cost[joint_sign];
  }
}

void shim_fill_palette_uv_mode_costs(const uint16_t *palette_uv_mode_cdf,
                                     int *out) {
  for (int i = 0; i < PALETTE_UV_MODE_CONTEXTS; ++i)
    av1_cost_tokens_from_cdf(out + i * 2, palette_uv_mode_cdf + i * 3, NULL);
}

int shim_intra_mode_info_cost_uv(const int *angle_delta_cost,
                                 const int *palette_uv_mode_cost,
                                 int mode_cost, int uv_mode, int bsize,
                                 int angle_delta_uv, int try_palette,
                                 int y_palette_active) {
  int total_rate = mode_cost;
  const int use_palette = 0; /* scope: palette_size[1] == 0 */
  assert(((uv_mode != UV_DC_PRED) + use_palette) <= 1);
  if (try_palette && uv_mode == UV_DC_PRED) {
    total_rate += palette_uv_mode_cost[(y_palette_active ? 1 : 0) * 2 +
                                       use_palette];
  }
  const PREDICTION_MODE intra_mode = get_uv_mode((UV_PREDICTION_MODE)uv_mode);
  if (av1_is_directional_mode(intra_mode)) {
    if (av1_use_angle_delta((BLOCK_SIZE)bsize)) {
      total_rate += angle_delta_cost[(intra_mode - V_PRED) *
                                         (2 * MAX_ANGLE_DELTA + 1) +
                                     angle_delta_uv + MAX_ANGLE_DELTA];
    }
  }
  return total_rate;
}

/* ---- encoder-side CfL facades (REAL exported cfl_store_tx /
 *      av1_cfl_predict_block over a calloc'd MACROBLOCKD stub) --------------
 * Buffers are u16 at every bit depth (CONVERT_TO_BYTEPTR world; xd->cur_buf
 * flags YV12_FLAG_HIGHBITDEPTH so is_cur_buf_hbd(xd) holds, matching every
 * other hbd shim in this crate). CfL state (recon_buf_q3 / ac_buf_q3 /
 * buf_width / buf_height / are_parameters_computed) is copied in and out so
 * callers can thread it across calls exactly as the embedded xd->cfl would
 * be (store invalidates, first predict computes lazily, later predicts
 * reuse). */
void shim_cfl_store_tx(const uint16_t *luma, int block_off, int stride,
                       int row, int col, int tx_size, int bsize, int mi_row,
                       int mi_col, int ss_x, int ss_y, int bd,
                       uint16_t *recon_q3 /* [1024] */, int *buf_w, int *buf_h,
                       int *params_computed) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(MACROBLOCKD));
  YV12_BUFFER_CONFIG *cb =
      (YV12_BUFFER_CONFIG *)calloc(1, sizeof(YV12_BUFFER_CONFIG));
  if (!xd || !cb) {
    free(xd);
    free(cb);
    return;
  }
  cb->flags = YV12_FLAG_HIGHBITDEPTH;
  xd->cur_buf = cb;
  xd->bd = bd;
  xd->cfl.subsampling_x = ss_x;
  xd->cfl.subsampling_y = ss_y;
  xd->mi_row = mi_row;
  xd->mi_col = mi_col;
  memcpy(xd->cfl.recon_buf_q3, recon_q3, sizeof(xd->cfl.recon_buf_q3));
  xd->cfl.buf_width = *buf_w;
  xd->cfl.buf_height = *buf_h;
  xd->cfl.are_parameters_computed = *params_computed;
  xd->plane[AOM_PLANE_Y].dst.buf =
      (uint8_t *)CONVERT_TO_BYTEPTR(luma + block_off);
  xd->plane[AOM_PLANE_Y].dst.stride = stride;
  cfl_store_tx(xd, row, col, (TX_SIZE)tx_size, (BLOCK_SIZE)bsize);
  memcpy(recon_q3, xd->cfl.recon_buf_q3, sizeof(xd->cfl.recon_buf_q3));
  *buf_w = xd->cfl.buf_width;
  *buf_h = xd->cfl.buf_height;
  *params_computed = xd->cfl.are_parameters_computed;
  free(xd);
  free(cb);
}

void shim_cfl_predict_block(uint16_t *recon_q3 /* [1024] */,
                            int16_t *ac_q3 /* [1024] */, int *buf_w,
                            int *buf_h, int *params_computed,
                            uint16_t *dst, int dst_off, int dst_stride,
                            int tx_size, int plane, int cfl_alpha_idx,
                            int cfl_alpha_signs, int bsize, int lossless,
                            int ss_x, int ss_y, int bd) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(MACROBLOCKD));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(MB_MODE_INFO));
  YV12_BUFFER_CONFIG *cb =
      (YV12_BUFFER_CONFIG *)calloc(1, sizeof(YV12_BUFFER_CONFIG));
  if (!xd || !mbmi || !cb) {
    free(xd);
    free(mbmi);
    free(cb);
    return;
  }
  cb->flags = YV12_FLAG_HIGHBITDEPTH;
  xd->cur_buf = cb;
  xd->bd = bd;
  MB_MODE_INFO *mi_ptr = mbmi;
  xd->mi = &mi_ptr;
  mbmi->bsize = (BLOCK_SIZE)bsize;
  mbmi->segment_id = 0;
  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->uv_mode = UV_CFL_PRED;
  mbmi->cfl_alpha_idx = (uint8_t)cfl_alpha_idx;
  mbmi->cfl_alpha_signs = (int8_t)cfl_alpha_signs;
  xd->lossless[0] = lossless;
  /* is_cfl_allowed(xd) (assert) reads plane[AOM_PLANE_U] subsampling. */
  xd->plane[AOM_PLANE_U].subsampling_x = ss_x;
  xd->plane[AOM_PLANE_U].subsampling_y = ss_y;
  xd->cfl.subsampling_x = ss_x;
  xd->cfl.subsampling_y = ss_y;
  memcpy(xd->cfl.recon_buf_q3, recon_q3, sizeof(xd->cfl.recon_buf_q3));
  memcpy(xd->cfl.ac_buf_q3, ac_q3, sizeof(xd->cfl.ac_buf_q3));
  xd->cfl.buf_width = *buf_w;
  xd->cfl.buf_height = *buf_h;
  xd->cfl.are_parameters_computed = *params_computed;
  av1_cfl_predict_block(xd, (uint8_t *)CONVERT_TO_BYTEPTR(dst + dst_off),
                        dst_stride, (TX_SIZE)tx_size, plane);
  memcpy(recon_q3, xd->cfl.recon_buf_q3, sizeof(xd->cfl.recon_buf_q3));
  memcpy(ac_q3, xd->cfl.ac_buf_q3, sizeof(xd->cfl.ac_buf_q3));
  *buf_w = xd->cfl.buf_width;
  *buf_h = xd->cfl.buf_height;
  *params_computed = xd->cfl.are_parameters_computed;
  free(xd);
  free(mbmi);
  free(cb);
}

/* ---- winner re-encode (av1_encode_intra_block_plane) LUMA map oracles -----
 * (1) shim_get_tx_type_y: the REAL av1_get_tx_type (blockd.h:1283 static
 *     inline, pristine C recompiled in this TU) for PLANE_TYPE_Y on an intra
 *     block over a calloc'd MACROBLOCKD stub. Marshals lossless + the
 *     RDO-time BLOCK-LOCAL tx_type_map (xd->tx_type_map /
 *     xd->tx_type_map_stride = mi_size_wide[bsize],
 *     partition_search.c:895-896). The in-set assert on the returned type is
 *     LIVE in this -O2-without-NDEBUG build — callers must keep map origin
 *     cells in-set for (tx_size, reduced_tx_set), the real encoder invariant.
 * (2) shim_update_txk_array: the REAL update_txk_array (blockd.h:1260 static
 *     inline) over the same stub — the eob==0 DCT_DCT reset write
 *     (encodemb.c:770-779) incl. the 64-side 16x16-unit fill. */
int shim_get_tx_type_y(int lossless, int tx_size, int reduced_tx_set_used,
                       const uint8_t *tx_type_map, int map_stride, int blk_row,
                       int blk_col) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(MACROBLOCKD));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(MB_MODE_INFO));
  if (!xd || !mbmi) {
    free(xd);
    free(mbmi);
    return -1;
  }
  MB_MODE_INFO *mi_ptr = mbmi;
  xd->mi = &mi_ptr;
  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->segment_id = 0;
  xd->lossless[0] = lossless;
  xd->tx_type_map = (uint8_t *)tx_type_map; /* Y arm reads only */
  xd->tx_type_map_stride = map_stride;
  const TX_TYPE t = av1_get_tx_type(xd, PLANE_TYPE_Y, blk_row, blk_col,
                                    (TX_SIZE)tx_size, reduced_tx_set_used);
  free(xd);
  free(mbmi);
  return (int)t;
}

int shim_update_txk_array(uint8_t *tx_type_map, int map_stride, int blk_row,
                          int blk_col, int tx_size, int tx_type) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(MACROBLOCKD));
  if (!xd) return -1;
  xd->tx_type_map = tx_type_map;
  xd->tx_type_map_stride = map_stride;
  update_txk_array(xd, blk_row, blk_col, (TX_SIZE)tx_size, (TX_TYPE)tx_type);
  free(xd);
  return 0;
}

/* ---- partition RDO primitives: the REAL rd.h static inlines ---------------
 * av1_rd_cost_update (rd.h:201) + av1_rd_stats_subtraction (rd.h:210) over
 * marshalled RD_STATS rate/dist/rdcost slices (av1_init_rd_stats fills the
 * remaining fields; only these three participate). av1_calculate_rd_cost's
 * negative-rate RDCOST_NEG_R arm is reached through both. */
void shim_rd_cost_update(int mult, int32_t *rate, int64_t *dist,
                         int64_t *rdcost) {
  RD_STATS s;
  av1_init_rd_stats(&s);
  s.rate = *rate;
  s.dist = *dist;
  s.rdcost = *rdcost;
  av1_rd_cost_update(mult, &s);
  *rate = s.rate;
  *dist = s.dist;
  *rdcost = s.rdcost;
}

void shim_rd_stats_subtraction(int mult, int32_t l_rate, int64_t l_dist,
                               int64_t l_rdcost, int32_t r_rate,
                               int64_t r_dist, int64_t r_rdcost,
                               int32_t *o_rate, int64_t *o_dist,
                               int64_t *o_rdcost) {
  RD_STATS left, right, out;
  av1_init_rd_stats(&left);
  av1_init_rd_stats(&right);
  av1_init_rd_stats(&out);
  left.rate = l_rate;
  left.dist = l_dist;
  left.rdcost = l_rdcost;
  right.rate = r_rate;
  right.dist = r_dist;
  right.rdcost = r_rdcost;
  av1_rd_stats_subtraction(mult, &left, &right, &out);
  *o_rate = out.rate;
  *o_dist = out.dist;
  *o_rdcost = out.rdcost;
}
