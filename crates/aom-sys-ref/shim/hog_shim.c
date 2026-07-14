/* HOG intra-mode prune oracle (av1/encoder/intra_mode_search_utils.h).
 *
 * Includes the REAL header so the shim gets the header's own static tables
 * (av1_intra_hog_model_weights / bias / nnconfig, the bin thresholds) and the
 * REAL static-inline lowbd/highbd_generate_hog bodies — nothing re-typed.
 *
 * NN dispatch: the production encoder calls av1_nn_predict through RTCD; on
 * AVX2-capable x86-64 (the reference environment) that resolves to
 * av1_nn_predict_avx2, whose f32 accumulation ORDER differs from the C/SSE3
 * variants — shim_hog_nn_predict calls av1_nn_predict_avx2 explicitly and
 * shim_hog_nn_predict_dispatched calls the RTCD pointer (after ref_init) so
 * the harness can prove they agree on the running machine.
 *
 * shim_prune_intra_mode_with_hog_y transcribes ONLY the thin
 * collect_hog_data edge-clip + chroma-scale + threshold wrapper over those
 * REAL pieces (prune_intra_mode_with_hog itself needs a full MACROBLOCK). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "av1/encoder/intra_mode_search_utils.h"
#define HOG_BINS 32

void shim_hog_nn_predict(const float *hist, int reduce_prec, float *scores) {
  av1_nn_predict_avx2(hist, &av1_intra_hog_model_nnconfig, reduce_prec,
                      scores);
}

void shim_hog_nn_predict_dispatched(const float *hist, int reduce_prec,
                                    float *scores) {
  av1_nn_predict(hist, &av1_intra_hog_model_nnconfig, reduce_prec, scores);
}

/* generate_hog over the REAL static inlines. Lowbd takes an 8-bit copy of
 * the u16 plane window (the production u8 layout); highbd uses
 * CONVERT_TO_BYTEPTR as the real caller does. `rows`/`cols` are the
 * edge-clipped dims; the walk reads rows 0..rows-1 x cols 0..cols-1. */
void shim_generate_hog(const uint16_t *src, int src_off, int stride, int rows,
                       int cols, int bd, float *hist /*[32]*/) {
  memset(hist, 0, HOG_BINS * sizeof(*hist));
  if (bd > 8) {
    highbd_generate_hog(CONVERT_TO_BYTEPTR(src + src_off), stride, rows, cols,
                        hist);
  } else {
    uint8_t *p8 = (uint8_t *)calloc((size_t)rows * stride, 1);
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        p8[r * stride + c] = (uint8_t)src[src_off + r * stride + c];
    lowbd_generate_hog(p8, stride, rows, cols, hist);
    free(p8);
  }
}

/* prune_intra_mode_with_hog, luma (is_chroma = 0, ss_x = ss_y = 0):
 * collect_hog_data's frame-edge clip (mb_to_*_edge in 1/8 pel) + the
 * *= (1+ss_x)*(1+ss_y) scale (a *1 no-op for luma, kept verbatim) + the NN
 * scores + the <= th mask fill over modes V_PRED..D67_PRED (1..8). */
void shim_prune_intra_mode_with_hog_y(const uint16_t *src, int src_off,
                                      int stride, int bsize,
                                      int mb_to_right_edge,
                                      int mb_to_bottom_edge, int bd, float th,
                                      uint8_t *mask /*[13]*/) {
  const int bh = block_size_high[bsize];
  const int bw = block_size_wide[bsize];
  const int rows =
      (mb_to_bottom_edge >= 0) ? bh : (mb_to_bottom_edge >> 3) + bh;
  const int cols = (mb_to_right_edge >= 0) ? bw : (mb_to_right_edge >> 3) + bw;

  float hog[HOG_BINS];
  shim_generate_hog(src, src_off, stride, rows, cols, bd, hog);
  for (int b = 0; b < HOG_BINS; ++b) {
    hog[b] *= (1 + 0) * (1 + 0);
  }

  float scores[DIRECTIONAL_MODES] = { 0.0f };
  av1_nn_predict_avx2(hog, &av1_intra_hog_model_nnconfig, 1, scores);
  for (int mode = V_PRED; mode <= D67_PRED; mode++) {
    if (scores[mode - V_PRED] <= th) {
      mask[mode] = 1;
    }
  }
}
