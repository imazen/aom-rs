/* Oracle shims for the inter-encoder motion search (crate aom-encode,
 * INTER-ENCODE chunk 2d). Oracle use only.
 *
 *  - shim_upsampled_pred wraps the REAL libaom `aom_upsampled_pred_c`
 *    (av1/encoder/reconinter_enc.c:462) for the lowbd, unscaled,
 *    USE_8_TAPS (EIGHTTAP_REGULAR) subpel-predictor path used by the speed-0
 *    subpel motion search (`av1_find_best_sub_pixel_tree` ->
 *    `upsampled_pref_error`). `xd == NULL` takes the unscaled branch directly
 *    (`aom_upsampled_pred_scaled` returns false for a NULL xd — "expect xd ==
 *    NULL only in tests", reconinter_enc.c:427-428), so cm/mi_row/mi_col/mv
 *    are unused and the output is purely the fixed-phase 8-tap convolution of
 *    the reference at (subpel_x_q3, subpel_y_q3).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "config/aom_dsp_rtcd.h"
/* Canonical order (mirrors motion_search_facade.c): reconinter.h (MV, filter.h /
 * USE_8_TAPS) + encoder.h (the umbrella that resolves the mcomp.h <-> speed_
 * features.h SUBPEL_FORCE_STOP circular include) BEFORE mcomp.h. */
#include "av1/common/reconinter.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/mcomp.h"
#include "av1/encoder/reconinter_enc.h"
#include "av1/common/scale.h"
#include "aom_dsp/variance.h"

void shim_upsampled_pred(const uint8_t *ref, int ref_stride, int width,
                         int height, int subpel_x_q3, int subpel_y_q3,
                         uint8_t *dst) {
  MV mv = { 0, 0 };
  aom_upsampled_pred_c(NULL, NULL, 0, 0, &mv, dst, width, height, subpel_x_q3,
                       subpel_y_q3, ref, ref_stride, USE_8_TAPS);
}

/* ---- shim_find_best_sub_pixel_tree ------------------------------------
 * Drives the REAL exported `av1_find_best_sub_pixel_tree` (mcomp.c:3266) for
 * the lowbd, unscaled, single-ref, USE_8_TAPS (speed-0 allintra/GOOD) subpel
 * search. Constructs a minimal MACROBLOCKD (calloc'd mbmi with use_intrabc=0,
 * identity block_ref_scale_factors, lowbd cur_buf, tmp_upsample_pred scratch)
 * + AV1_COMMON + SUBPEL_MOTION_SEARCH_PARAMS field-by-field, exactly the state
 * the tree + upsampled_pref_error read. Cost tables (mvjcost/mvcost) are the
 * caller's (centred at MV_MAX); vfp->vf is the real aom_variance{W}x{H}_c. No
 * start_mv_stats (full center-error) and no repeat list.
 */

extern unsigned int aom_variance4x4_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance4x8_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance8x4_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance8x8_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance8x16_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance16x8_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance16x16_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance16x32_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance16x64_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance32x16_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance32x32_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance64x16_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);
extern unsigned int aom_variance64x64_c(const uint8_t *, int, const uint8_t *, int, unsigned int *);

static aom_variance_fn_t shim_pick_vf(int w, int h) {
  if (w == 4 && h == 4) return aom_variance4x4_c;
  if (w == 4 && h == 8) return aom_variance4x8_c;
  if (w == 8 && h == 4) return aom_variance8x4_c;
  if (w == 8 && h == 8) return aom_variance8x8_c;
  if (w == 8 && h == 16) return aom_variance8x16_c;
  if (w == 16 && h == 8) return aom_variance16x8_c;
  if (w == 16 && h == 16) return aom_variance16x16_c;
  if (w == 16 && h == 32) return aom_variance16x32_c;
  if (w == 16 && h == 64) return aom_variance16x64_c;
  if (w == 32 && h == 16) return aom_variance32x16_c;
  if (w == 32 && h == 32) return aom_variance32x32_c;
  if (w == 64 && h == 16) return aom_variance64x16_c;
  if (w == 64 && h == 64) return aom_variance64x64_c;
  return NULL;
}

int shim_find_best_sub_pixel_tree(
    const uint8_t *src, int src_stride, const uint8_t *ref_at_origin,
    int ref_stride, int w, int h, int start_row, int start_col, int ref_mv_row,
    int ref_mv_col, const int *mvjcost, const int *mvcost0, const int *mvcost1,
    int error_per_bit, int allow_hp, int forced_stop, int iters_per_step,
    int row_min, int row_max, int col_min, int col_max, int *out_best_row,
    int *out_best_col, int *out_distortion, unsigned int *out_sse) {
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(MACROBLOCKD));
  MB_MODE_INFO *mbmi = (MB_MODE_INFO *)calloc(1, sizeof(MB_MODE_INFO));
  YV12_BUFFER_CONFIG *cb = (YV12_BUFFER_CONFIG *)calloc(1, sizeof(YV12_BUFFER_CONFIG));
  struct scale_factors *sf = (struct scale_factors *)calloc(1, sizeof(struct scale_factors));
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(AV1_COMMON));
  uint8_t *tmp_pred = (uint8_t *)calloc((size_t)MAX_SB_SIZE * MAX_SB_SIZE, 1);
  if (!xd || !mbmi || !cb || !sf || !cm || !tmp_pred) {
    free(xd); free(mbmi); free(cb); free(sf); free(cm); free(tmp_pred);
    return -1;
  }

  MB_MODE_INFO *mi_ptr = mbmi; /* use_intrabc = 0 from calloc */
  xd->mi = &mi_ptr;
  cb->flags = 0; /* lowbd */
  xd->cur_buf = cb;
  xd->bd = 8;
  xd->mi_row = 0;
  xd->mi_col = 0;
  sf->x_scale_fp = REF_NO_SCALE;
  sf->y_scale_fp = REF_NO_SCALE;
  sf->x_step_q4 = 16;
  sf->y_step_q4 = 16;
  xd->block_ref_scale_factors[0] = sf;
  xd->block_ref_scale_factors[1] = sf;
  xd->tmp_upsample_pred = tmp_pred;

  struct buf_2d src_buf;
  memset(&src_buf, 0, sizeof(src_buf));
  src_buf.buf = (uint8_t *)src;
  src_buf.stride = src_stride;
  struct buf_2d ref_buf;
  memset(&ref_buf, 0, sizeof(ref_buf));
  ref_buf.buf = (uint8_t *)ref_at_origin;
  ref_buf.stride = ref_stride;

  aom_variance_fn_ptr_t vfp;
  memset(&vfp, 0, sizeof(vfp));
  vfp.vf = shim_pick_vf(w, h);

  MV ref_mv = { (int16_t)ref_mv_row, (int16_t)ref_mv_col };

  SUBPEL_MOTION_SEARCH_PARAMS ms;
  memset(&ms, 0, sizeof(ms));
  ms.allow_hp = allow_hp;
  ms.cost_list = NULL;
  ms.forced_stop = (SUBPEL_FORCE_STOP)forced_stop;
  ms.iters_per_step = iters_per_step;
  ms.mv_limits.row_min = row_min;
  ms.mv_limits.row_max = row_max;
  ms.mv_limits.col_min = col_min;
  ms.mv_limits.col_max = col_max;
  ms.mv_cost_params.ref_mv = &ref_mv;
  ms.mv_cost_params.mv_cost_type = MV_COST_ENTROPY;
  ms.mv_cost_params.mvjcost = mvjcost;
  ms.mv_cost_params.mvcost[0] = (int *)mvcost0;
  ms.mv_cost_params.mvcost[1] = (int *)mvcost1;
  ms.mv_cost_params.error_per_bit = error_per_bit;
  ms.mv_cost_params.sad_per_bit = 0;
  ms.var_params.vfp = &vfp;
  ms.var_params.subpel_search_type = USE_8_TAPS;
  ms.var_params.ms_buffers.src = &src_buf;
  ms.var_params.ms_buffers.ref = &ref_buf;
  ms.var_params.ms_buffers.second_pred = NULL;
  ms.var_params.ms_buffers.mask = NULL;
  ms.var_params.w = w;
  ms.var_params.h = h;

  MV start = { (int16_t)start_row, (int16_t)start_col };
  MV best;
  int distortion = 0;
  unsigned int sse = 0;
  int besterr = av1_find_best_sub_pixel_tree(xd, cm, &ms, start, NULL, &best,
                                             &distortion, &sse, NULL);

  *out_best_row = best.row;
  *out_best_col = best.col;
  *out_distortion = distortion;
  *out_sse = sse;

  free(xd); free(mbmi); free(cb); free(sf); free(cm); free(tmp_pred);
  return besterr;
}

/* ---- shim_get_mvpred_sse ---------------------------------------------
 * Drives the REAL exported av1_get_mvpred_sse (mcomp.c:3963): the full-pel
 * predictor SSE + coded-MV rate cost the motion-search facade scores a full-pel
 * result with. Reuses shim_pick_vf; caller-supplied MV cost tables (centred).
 */
int shim_get_mvpred_sse(int best_row, int best_col, const uint8_t *src,
                        int src_stride, const uint8_t *pre_at_origin,
                        int pre_stride, int w, int h, int ref_mv_row,
                        int ref_mv_col, const int *mvjcost, const int *mvcost0,
                        const int *mvcost1, int error_per_bit) {
  aom_variance_fn_ptr_t vfp;
  memset(&vfp, 0, sizeof(vfp));
  vfp.vf = shim_pick_vf(w, h);

  MV ref_mv = { (int16_t)ref_mv_row, (int16_t)ref_mv_col };
  MV_COST_PARAMS mcp;
  memset(&mcp, 0, sizeof(mcp));
  mcp.ref_mv = &ref_mv;
  mcp.mv_cost_type = MV_COST_ENTROPY;
  mcp.mvjcost = mvjcost;
  mcp.mvcost[0] = (int *)mvcost0;
  mcp.mvcost[1] = (int *)mvcost1;
  mcp.error_per_bit = error_per_bit;
  mcp.sad_per_bit = 0;

  struct buf_2d src_buf;
  memset(&src_buf, 0, sizeof(src_buf));
  src_buf.buf = (uint8_t *)src;
  src_buf.stride = src_stride;
  struct buf_2d pre_buf;
  memset(&pre_buf, 0, sizeof(pre_buf));
  pre_buf.buf = (uint8_t *)pre_at_origin;
  pre_buf.stride = pre_stride;

  FULLPEL_MV best = { (int16_t)best_row, (int16_t)best_col };
  return av1_get_mvpred_sse(&mcp, best, &vfp, &src_buf, &pre_buf);
}

/* ---- shim_mv_bit_cost — the REAL av1_mv_bit_cost (mcomp.c:307). */
int shim_mv_bit_cost(int mv_row, int mv_col, int ref_row, int ref_col,
                     const int *mvjcost, const int *mvcost0,
                     const int *mvcost1, int weight) {
  MV mv = { (int16_t)mv_row, (int16_t)mv_col };
  MV ref_mv = { (int16_t)ref_row, (int16_t)ref_col };
  int *mvcost[2] = { (int *)mvcost0, (int *)mvcost1 };
  return av1_mv_bit_cost(&mv, &ref_mv, mvjcost, mvcost, weight);
}
