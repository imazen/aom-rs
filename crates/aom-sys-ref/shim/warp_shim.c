/* Oracle shims for the decoder local-warped-motion core (crate aom-inter,
 * warp module — chunk 5). Oracle use only.
 *
 *  - shim_warp_affine wraps the REAL libaom `av1_warp_affine_c`
 *    (av1/common/warped_motion.c:518) — the bd8 non-compound affine warp
 *    filter — with the decoder's fixed single-ref ConvolveParams
 *    (`get_conv_params_no_round(0, 0, NULL, 0, 0, 8)` => round_0 = 3,
 *    round_1 = 11, is_compound = 0). ref/pred are u8 planes.
 *
 *  - shim_find_projection wraps `av1_find_projection`
 *    (warped_motion.c:906): the caller sets wm.wmtype = AFFINE (as decodemv.c
 *    does), the model is derived, and the result fields are returned out.
 *
 *  - shim_get_shear_params wraps `av1_get_shear_params` (warped_motion.c:243)
 *    for the standalone alpha/beta/gamma/delta + validity check.
 */
#include <string.h>
#include "config/av1_rtcd.h"
#include "av1/common/warped_motion.h"
#include "av1/common/convolve.h"

void shim_warp_affine(const int32_t *mat, const uint8_t *ref, int width,
                      int height, int stride, uint8_t *pred, int p_col,
                      int p_row, int p_width, int p_height, int p_stride,
                      int subsampling_x, int subsampling_y, int16_t alpha,
                      int16_t beta, int16_t gamma, int16_t delta) {
  ConvolveParams conv_params = get_conv_params_no_round(0, 0, NULL, 0, 0, 8);
  av1_warp_affine_c(mat, ref, width, height, stride, pred, p_col, p_row,
                    p_width, p_height, p_stride, subsampling_x, subsampling_y,
                    &conv_params, alpha, beta, gamma, delta);
}

/* Returns av1_find_projection's return value (1 = invalid model); on success
 * fills wmmat_out[0..5] + the four shear params. */
int shim_find_projection(int np, const int *pts1, const int *pts2, int bsize,
                         int mvy, int mvx, int mi_row, int mi_col,
                         int32_t *wmmat_out, int16_t *alpha_out,
                         int16_t *beta_out, int16_t *gamma_out,
                         int16_t *delta_out) {
  WarpedMotionParams wm;
  memset(&wm, 0, sizeof(wm));
  wm.wmtype = AFFINE;
  wm.invalid = 0;
  int ret = av1_find_projection(np, pts1, pts2, (BLOCK_SIZE)bsize, mvy, mvx, &wm,
                                mi_row, mi_col);
  for (int i = 0; i < 6; ++i) wmmat_out[i] = wm.wmmat[i];
  *alpha_out = wm.alpha;
  *beta_out = wm.beta;
  *gamma_out = wm.gamma;
  *delta_out = wm.delta;
  return ret;
}

/* Wraps av1_get_shear_params: caller supplies wmmat[0..5]; returns the C
 * return (1 = usable model), fills the four shear params. */
int shim_get_shear_params(const int32_t *wmmat, int16_t *alpha_out,
                          int16_t *beta_out, int16_t *gamma_out,
                          int16_t *delta_out) {
  WarpedMotionParams wm;
  memset(&wm, 0, sizeof(wm));
  for (int i = 0; i < 6; ++i) wm.wmmat[i] = wmmat[i];
  wm.wmtype = AFFINE;
  int ret = av1_get_shear_params(&wm);
  *alpha_out = wm.alpha;
  *beta_out = wm.beta;
  *gamma_out = wm.gamma;
  *delta_out = wm.delta;
  return ret;
}
