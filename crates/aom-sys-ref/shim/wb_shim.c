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

/* Transcribed control flow of encode_cdef over the real aom_wb. */
uint32_t shim_encode_cdef(int enable_cdef, int allow_intrabc, int damping,
                          int cdef_bits, int nb, const int *y, const int *uv,
                          int num_planes, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  if (!enable_cdef || allow_intrabc) return aom_wb_bytes_written(&wb);
  aom_wb_write_literal(&wb, damping - 3, 2);
  aom_wb_write_literal(&wb, cdef_bits, 2);
  for (int i = 0; i < nb; i++) {
    aom_wb_write_literal(&wb, y[i], 6);
    if (num_planes > 1) aom_wb_write_literal(&wb, uv[i], 6);
  }
  return aom_wb_bytes_written(&wb);
}

/* Segmentation / frame-size frame-header components, transcribed control flow
 * over the real aom_wb (and the real exported seg-feature tables). */
#include "av1/common/seg_common.h"
#include "av1/common/common.h"  /* get_unsigned_bits, clamp */
#include "av1/common/filter.h"  /* SWITCHABLE, LOG_SWITCHABLE_FILTERS */
#include "av1/common/scale.h"   /* SCALE_NUMERATOR */
#include "av1/common/enums.h"   /* SUPERRES_SCALE_BITS, SUPERRES_SCALE_DENOMINATOR_MIN */

uint32_t shim_encode_segmentation(int enabled, int has_primary_ref, int update_map,
                                  int temporal_update, int update_data,
                                  const uint32_t *feature_mask,
                                  const int *feature_data, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  aom_wb_write_bit(&wb, enabled);
  if (!enabled) return aom_wb_bytes_written(&wb);
  if (has_primary_ref) {
    aom_wb_write_bit(&wb, update_map);
    if (update_map) aom_wb_write_bit(&wb, temporal_update);
    aom_wb_write_bit(&wb, update_data);
  }
  if (update_data) {
    for (int i = 0; i < MAX_SEGMENTS; i++) {
      for (int j = 0; j < SEG_LVL_MAX; j++) {
        const int active = (feature_mask[i] & (1u << j)) != 0;
        aom_wb_write_bit(&wb, active);
        if (active) {
          const int data_max = av1_seg_feature_data_max(j);
          const int data_min = -data_max;
          const int ubits = get_unsigned_bits(data_max);
          const int data = clamp(feature_data[i * SEG_LVL_MAX + j], data_min, data_max);
          if (av1_is_segfeature_signed(j))
            aom_wb_write_inv_signed_literal(&wb, data, ubits);
          else
            aom_wb_write_literal(&wb, data, ubits);
        }
      }
    }
  }
  return aom_wb_bytes_written(&wb);
}

uint32_t shim_write_frame_interp_filter(int filter, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  aom_wb_write_bit(&wb, filter == SWITCHABLE);
  if (filter != SWITCHABLE) aom_wb_write_literal(&wb, filter, LOG_SWITCHABLE_FILTERS);
  return aom_wb_bytes_written(&wb);
}

uint32_t shim_write_superres_scale(int enable_superres, int denom, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  if (!enable_superres) return aom_wb_bytes_written(&wb);
  if (denom == SCALE_NUMERATOR) {
    aom_wb_write_bit(&wb, 0);
  } else {
    aom_wb_write_bit(&wb, 1);
    aom_wb_write_literal(&wb, denom - SUPERRES_SCALE_DENOMINATOR_MIN, SUPERRES_SCALE_BITS);
  }
  return aom_wb_bytes_written(&wb);
}

uint32_t shim_write_render_size(int scaling_active, int rw, int rh, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  aom_wb_write_bit(&wb, scaling_active);
  if (scaling_active) {
    aom_wb_write_literal(&wb, rw - 1, 16);
    aom_wb_write_literal(&wb, rh - 1, 16);
  }
  return aom_wb_bytes_written(&wb);
}

uint32_t shim_write_frame_size(int frame_size_override, int num_bits_width,
                               int num_bits_height, int up_w, int up_h,
                               int enable_superres, int denom, int scaling_active,
                               int rw, int rh, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  const int coded_width = up_w - 1;
  const int coded_height = up_h - 1;
  if (frame_size_override) {
    aom_wb_write_literal(&wb, coded_width, num_bits_width);
    aom_wb_write_literal(&wb, coded_height, num_bits_height);
  }
  if (enable_superres) {
    if (denom == SCALE_NUMERATOR) aom_wb_write_bit(&wb, 0);
    else { aom_wb_write_bit(&wb, 1); aom_wb_write_literal(&wb, denom - SUPERRES_SCALE_DENOMINATOR_MIN, SUPERRES_SCALE_BITS); }
  }
  aom_wb_write_bit(&wb, scaling_active);
  if (scaling_active) {
    aom_wb_write_literal(&wb, rw - 1, 16);
    aom_wb_write_literal(&wb, rh - 1, 16);
  }
  return aom_wb_bytes_written(&wb);
}
