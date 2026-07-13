/* Oracles for the mode-info partition CDF primitives — the real static-inline
 * partition_cdf_length / partition_gather_{vert,horz}_alike from av1_common_int.h. */
#include <stdint.h>
#include "av1/common/av1_common_int.h"

int shim_partition_cdf_length(int bsize) {
  return partition_cdf_length((BLOCK_SIZE)bsize);
}

void shim_partition_gather_vert(uint16_t *out, const uint16_t *in, int bsize) {
  partition_gather_vert_alike(out, in, (BLOCK_SIZE)bsize);
}

void shim_partition_gather_horz(uint16_t *out, const uint16_t *in, int bsize) {
  partition_gather_horz_alike(out, in, (BLOCK_SIZE)bsize);
}

/* Facade: set the two partition-context pointers on a stack MACROBLOCKD and call
 * the real partition_plane_context (it reads only those two fields). */
#include "av1/common/blockd.h"
int shim_partition_plane_context(const signed char *above, const signed char *left,
                                 int mi_row, int mi_col, int bsize) {
  MACROBLOCKD xd;
  xd.above_partition_context = (PARTITION_CONTEXT *)above; /* pointer field */
  for (int i = 0; i < MAX_MIB_SIZE; i++)
    xd.left_partition_context[i] = left[i]; /* inline array field */
  return partition_plane_context(&xd, mi_row, mi_col, (BLOCK_SIZE)bsize);
}
