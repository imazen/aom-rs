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

/* Transcribed control flow of encode_quantization (av1/encoder/bitstream.c),
 * driven through the real aom_wb primitives (validated separately). */
static void wb_write_delta_q(struct aom_write_bit_buffer *wb, int delta_q) {
  if (delta_q != 0) {
    aom_wb_write_bit(wb, 1);
    aom_wb_write_inv_signed_literal(wb, delta_q, 6);
  } else {
    aom_wb_write_bit(wb, 0);
  }
}
uint32_t shim_encode_quantization(int base_qindex, int y_dc, int u_dc, int u_ac,
                                  int v_dc, int v_ac, int using_qm, int qm_y,
                                  int qm_u, int qm_v, int num_planes,
                                  int separate_uv, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  aom_wb_write_literal(&wb, base_qindex, 8);
  wb_write_delta_q(&wb, y_dc);
  if (num_planes > 1) {
    int diff_uv = (u_dc != v_dc) || (u_ac != v_ac);
    if (separate_uv) aom_wb_write_bit(&wb, diff_uv);
    wb_write_delta_q(&wb, u_dc);
    wb_write_delta_q(&wb, u_ac);
    if (diff_uv) { wb_write_delta_q(&wb, v_dc); wb_write_delta_q(&wb, v_ac); }
  }
  aom_wb_write_bit(&wb, using_qm);
  if (using_qm) {
    aom_wb_write_literal(&wb, qm_y, 4);
    aom_wb_write_literal(&wb, qm_u, 4);
    if (separate_uv) aom_wb_write_literal(&wb, qm_v, 4);
  }
  return aom_wb_bytes_written(&wb);
}

/* Transcribed control flow of encode_loopfilter over the real aom_wb. */
uint32_t shim_encode_loopfilter(int allow_intrabc, int fl0, int fl1, int flu,
                                int flv, int sharpness, int mode_ref_enabled,
                                int mode_ref_update, const signed char *ref_deltas,
                                const signed char *mode_deltas,
                                const signed char *last_ref,
                                const signed char *last_mode, int num_planes,
                                uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  if (allow_intrabc) return aom_wb_bytes_written(&wb);
  aom_wb_write_literal(&wb, fl0, 6);
  aom_wb_write_literal(&wb, fl1, 6);
  if (num_planes > 1 && (fl0 || fl1)) {
    aom_wb_write_literal(&wb, flu, 6);
    aom_wb_write_literal(&wb, flv, 6);
  }
  aom_wb_write_literal(&wb, sharpness, 3);
  aom_wb_write_bit(&wb, mode_ref_enabled);
  int meaningful = 0;
  if (mode_ref_update) {
    for (int i = 0; i < 8; i++) if (ref_deltas[i] != last_ref[i]) meaningful = 1;
    for (int i = 0; i < 2; i++) if (mode_deltas[i] != last_mode[i]) meaningful = 1;
  }
  aom_wb_write_bit(&wb, meaningful);
  if (!meaningful) return aom_wb_bytes_written(&wb);
  for (int i = 0; i < 8; i++) {
    int changed = ref_deltas[i] != last_ref[i];
    aom_wb_write_bit(&wb, changed);
    if (changed) aom_wb_write_inv_signed_literal(&wb, ref_deltas[i], 6);
  }
  for (int i = 0; i < 2; i++) {
    int changed = mode_deltas[i] != last_mode[i];
    aom_wb_write_bit(&wb, changed);
    if (changed) aom_wb_write_inv_signed_literal(&wb, mode_deltas[i], 6);
  }
  return aom_wb_bytes_written(&wb);
}
