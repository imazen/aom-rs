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

/* --- warp neighbour-sample gather oracles (av1_findSamples / selectSamples) --- */
#include <stdlib.h>
#include "av1/common/av1_common_int.h"
#include "av1/common/mvref_common.h"

/* Build a synthetic per-cell MB_MODE_INFO grid + MACROBLOCKD and call the real
 * av1_findSamples. Returns np; writes pts / pts_inref (each [SAMPLES_ARRAY_SIZE]). */
int shim_find_samples(int mi_rows, int mi_cols, int grid_stride, int sb_size,
                      const int32_t *g_bsize, const int32_t *g_ref0,
                      const int32_t *g_ref1, const int32_t *g_mv_row,
                      const int32_t *g_mv_col, int mi_row, int mi_col,
                      int cur_bsize, int cur_ref0, int partition,
                      int up_available, int left_available, int tile_row_start,
                      int tile_row_end, int tile_col_start, int tile_col_end,
                      int32_t *pts_out, int32_t *pts_inref_out) {
  const int ncells = mi_rows * grid_stride;
  MB_MODE_INFO *cells = (MB_MODE_INFO *)calloc(ncells, sizeof(*cells));
  MB_MODE_INFO **grid = (MB_MODE_INFO **)calloc(ncells, sizeof(*grid));
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  if (!cells || !grid || !cm || !seq || !xd) return -1;

  for (int r = 0; r < mi_rows; r++) {
    for (int c = 0; c < mi_cols; c++) {
      const int i = r * grid_stride + c;
      MB_MODE_INFO *mi = &cells[i];
      mi->bsize = (BLOCK_SIZE)g_bsize[i];
      mi->ref_frame[0] = (MV_REFERENCE_FRAME)g_ref0[i];
      mi->ref_frame[1] = (MV_REFERENCE_FRAME)g_ref1[i];
      mi->mv[0].as_mv.row = (int16_t)g_mv_row[i];
      mi->mv[0].as_mv.col = (int16_t)g_mv_col[i];
      grid[i] = mi;
    }
  }

  cm->mi_params.mi_grid_base = grid;
  cm->mi_params.mi_stride = grid_stride;
  cm->mi_params.mi_rows = mi_rows;
  cm->mi_params.mi_cols = mi_cols;
  seq->sb_size = (BLOCK_SIZE)sb_size;
  cm->seq_params = seq;

  xd->mi = grid + (mi_row * grid_stride + mi_col);
  xd->mi_stride = grid_stride;
  xd->mi_row = mi_row;
  xd->mi_col = mi_col;
  xd->up_available = up_available;
  xd->left_available = left_available;
  xd->width = mi_size_wide[cur_bsize];
  xd->height = mi_size_high[cur_bsize];
  /* current block mbmi (findSamples reads xd->mi[0]->ref_frame[0] + partition) */
  xd->mi[0]->bsize = (BLOCK_SIZE)cur_bsize;
  xd->mi[0]->ref_frame[0] = (MV_REFERENCE_FRAME)cur_ref0;
  xd->mi[0]->partition = (PARTITION_TYPE)partition;
  /* set_mi_row_col rect flags (av1_common_int.h:1421) */
  xd->is_last_vertical_rect = 0;
  if (xd->width < xd->height && !((mi_col + xd->width) & (xd->height - 1)))
    xd->is_last_vertical_rect = 1;
  xd->is_first_horizontal_rect = 0;
  if (xd->width > xd->height && !(mi_row & (xd->width - 1)))
    xd->is_first_horizontal_rect = 1;
  xd->tile.mi_row_start = tile_row_start;
  xd->tile.mi_row_end = tile_row_end;
  xd->tile.mi_col_start = tile_col_start;
  xd->tile.mi_col_end = tile_col_end;

  int pts[SAMPLES_ARRAY_SIZE], pts_inref[SAMPLES_ARRAY_SIZE];
  int np = av1_findSamples(cm, xd, pts, pts_inref);
  for (int i = 0; i < SAMPLES_ARRAY_SIZE; i++) {
    pts_out[i] = pts[i];
    pts_inref_out[i] = pts_inref[i];
  }

  free(cells);
  free(grid);
  free(cm);
  free(seq);
  free(xd);
  return np;
}

/* Wraps av1_selectSamples: mv (row,col), in/out pts + pts_inref [len*2], len,
 * bsize; returns the kept count and compacts pts/pts_inref in place. */
int shim_select_samples(int mv_row, int mv_col, int32_t *pts, int32_t *pts_inref,
                        int len, int bsize) {
  MV mv;
  mv.row = (int16_t)mv_row;
  mv.col = (int16_t)mv_col;
  int cp[SAMPLES_ARRAY_SIZE], cpr[SAMPLES_ARRAY_SIZE];
  for (int i = 0; i < len * 2; i++) {
    cp[i] = pts[i];
    cpr[i] = pts_inref[i];
  }
  int ret = av1_selectSamples(&mv, cp, cpr, len, (BLOCK_SIZE)bsize);
  for (int i = 0; i < len * 2; i++) {
    pts[i] = cp[i];
    pts_inref[i] = cpr[i];
  }
  return ret;
}
