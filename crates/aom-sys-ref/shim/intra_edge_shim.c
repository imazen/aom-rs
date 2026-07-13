/* Shim for the static-inline intra-edge strength / upsample decisions +
 * direct wrappers for the exported edge filter/upsample kernels. Oracle only. */
#include <stdint.h>
#include <string.h>
#include "av1/common/reconintra.h"

/* intra_edge_filter_strength is static in reconintra.c; transcribe verbatim so
 * the oracle exercises the same LUT the decoder/encoder use. */
int shim_intra_edge_strength(int bs0, int bs1, int delta, int type) {
  const int d = delta < 0 ? -delta : delta;
  int strength = 0;
  const int blk_wh = bs0 + bs1;
  if (type == 0) {
    if (blk_wh <= 8) { if (d >= 56) strength = 1; }
    else if (blk_wh <= 12) { if (d >= 40) strength = 1; }
    else if (blk_wh <= 16) { if (d >= 40) strength = 1; }
    else if (blk_wh <= 24) { if (d >= 8) strength = 1; if (d >= 16) strength = 2; if (d >= 32) strength = 3; }
    else if (blk_wh <= 32) { if (d >= 1) strength = 1; if (d >= 4) strength = 2; if (d >= 32) strength = 3; }
    else { if (d >= 1) strength = 3; }
  } else {
    if (blk_wh <= 8) { if (d >= 40) strength = 1; if (d >= 64) strength = 2; }
    else if (blk_wh <= 16) { if (d >= 20) strength = 1; if (d >= 48) strength = 2; }
    else if (blk_wh <= 24) { if (d >= 4) strength = 3; }
    else { if (d >= 1) strength = 3; }
  }
  return strength;
}

int shim_use_intra_edge_upsample(int bs0, int bs1, int delta, int type) {
  return av1_use_intra_edge_upsample(bs0, bs1, delta, type);
}

void av1_filter_intra_edge_c(uint8_t *p, int sz, int strength);
void av1_upsample_intra_edge_c(uint8_t *p, int sz);

/* p_off points at logical index 0; the kernel reads p[-1..] / writes p[-2..].
 * Caller passes a buffer with >= 2 leading pad bytes. */
void shim_filter_intra_edge(uint8_t *p, int sz, int strength) {
  av1_filter_intra_edge_c(p, sz, strength);
}
void shim_upsample_intra_edge(uint8_t *p, int sz) {
  av1_upsample_intra_edge_c(p, sz);
}
