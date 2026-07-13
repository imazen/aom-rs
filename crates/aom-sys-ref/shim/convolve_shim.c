/* Shim over av1_convolve_{x,y}_sr_c using libaom's real EIGHTTAP_REGULAR
 * kernel + SR ConvolveParams. Oracle use only. */
#include <string.h>
#include "av1/common/convolve.h"
#include "av1/common/filter.h"

void av1_convolve_x_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                         int dst_stride, int w, int h,
                         const InterpFilterParams *filter_params_x,
                         const int subpel_x_qn, ConvolveParams *conv_params);
void av1_convolve_y_sr_c(const uint8_t *src, int src_stride, uint8_t *dst,
                         int dst_stride, int w, int h,
                         const InterpFilterParams *filter_params_y,
                         const int subpel_y_qn);

void shim_convolve_x_sr(const uint8_t *src, int ss, uint8_t *dst, int ds,
                        int w, int h, int subpel) {
  InterpFilterParams fp = { (const int16_t *)av1_sub_pel_filters_8,
                            SUBPEL_TAPS, EIGHTTAP_REGULAR };
  ConvolveParams cp;
  memset(&cp, 0, sizeof(cp));
  cp.round_0 = 3;                     // ROUND0_BITS
  cp.round_1 = 2 * FILTER_BITS - 3;   // 11
  av1_convolve_x_sr_c(src, ss, dst, ds, w, h, &fp, subpel, &cp);
}

void shim_convolve_y_sr(const uint8_t *src, int ss, uint8_t *dst, int ds,
                        int w, int h, int subpel) {
  InterpFilterParams fp = { (const int16_t *)av1_sub_pel_filters_8,
                            SUBPEL_TAPS, EIGHTTAP_REGULAR };
  av1_convolve_y_sr_c(src, ss, dst, ds, w, h, &fp, subpel);
}
