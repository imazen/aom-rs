/* Oracle shims for OBMC (overlapped block motion compensation) kernels
 * (crate aom-inter, chunk 4). Oracle use only.
 *
 *  - shim_get_obmc_mask wraps the REAL libaom `av1_get_obmc_mask`
 *    (av1/common/reconinter.c:774) — the raised-cosine feather mask table.
 *  - shim_blend_a64_vmask / shim_blend_a64_hmask wrap the REAL exported
 *    `aom_blend_a64_vmask_c` / `aom_blend_a64_hmask_c`
 *    (aom_dsp/blend_a64_{v,h}mask.c) — the per-row / per-column A64 blends used
 *    by `build_obmc_inter_pred_{above,left}` (reconinter.c:852/:891).
 */
#include "av1/common/reconinter.h"
#include "aom_dsp/blend.h"
#include "config/aom_dsp_rtcd.h"

const unsigned char *shim_get_obmc_mask(int length) {
  return av1_get_obmc_mask(length);
}

void shim_blend_a64_vmask(uint8_t *dst, uint32_t dst_stride,
                          const uint8_t *src0, uint32_t src0_stride,
                          const uint8_t *src1, uint32_t src1_stride,
                          const uint8_t *mask, int w, int h) {
  aom_blend_a64_vmask_c(dst, dst_stride, src0, src0_stride, src1, src1_stride,
                        mask, w, h);
}

void shim_blend_a64_hmask(uint8_t *dst, uint32_t dst_stride,
                          const uint8_t *src0, uint32_t src0_stride,
                          const uint8_t *src1, uint32_t src1_stride,
                          const uint8_t *mask, int w, int h) {
  aom_blend_a64_hmask_c(dst, dst_stride, src0, src0_stride, src1, src1_stride,
                        mask, w, h);
}
