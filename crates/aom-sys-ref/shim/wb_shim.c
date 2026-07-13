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
      case 3: /* add_trailing_bits, via the real aom_wb primitives */
        if (aom_wb_is_byte_aligned(&wb)) aom_wb_write_literal(&wb, 0x80, 8);
        else aom_wb_write_bit(&wb, 1);
        break;
      default: aom_wb_write_literal(&wb, (int)data[i], bits[i]); break;
    }
  }
  return aom_wb_bytes_written(&wb);
}

/* Transcribed verbatim from av1_write_obu_header (av1/encoder/bitstream.c) byte
 * output — the function is not cleanly exported and pulls in AV1LevelParams; the
 * byte layout it writes is copied here. Level-stats side effect omitted (no byte
 * effect). obu_type in bits 6..3, ext flag bit 2, has_size_field bit 1. */
uint32_t shim_write_obu_header(int obu_type, int has_nonzero_op,
                               int is_layer_specific, int obu_extension,
                               uint8_t *dst) {
  const int obu_extension_flag = has_nonzero_op && is_layer_specific;
  const int obu_has_size_field = 1;
  uint32_t size = 0;
  dst[0] = (obu_type << 3) | (obu_extension_flag << 2) | (obu_has_size_field << 1);
  size++;
  if (obu_extension_flag) {
    dst[1] = obu_extension & 0xFF;
    size++;
  }
  return size;
}
