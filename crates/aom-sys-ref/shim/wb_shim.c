/* Oracle for aom_write_bit_buffer (aom_dsp/bitwriter_buffer.c): apply a sequence
 * of write_literal / write_unsigned_literal ops and return the produced bytes. */
#include <stdint.h>
#include "aom_dsp/bitwriter_buffer.h"

/* kind[i]: 0 = write_literal (signed src), 1 = write_unsigned_literal,
 *          2 = write_inv_signed_literal. */
uint32_t shim_wb_apply(const uint32_t *data, const int *bits, const int *kind,
                       int n, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  for (int i = 0; i < n; i++) {
    switch (kind[i]) {
      case 1: aom_wb_write_unsigned_literal(&wb, data[i], bits[i]); break;
      case 2: aom_wb_write_inv_signed_literal(&wb, (int)data[i], bits[i]); break;
      default: aom_wb_write_literal(&wb, (int)data[i], bits[i]); break;
    }
  }
  return aom_wb_bytes_written(&wb);
}
