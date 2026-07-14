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
#include <stdint.h>
#include "av1/common/quant_common.h"
#include "av1/common/idct.h" /* MAX_TX_SCALE, av1_get_tx_scale */
#include "av1/encoder/rd.h"
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
