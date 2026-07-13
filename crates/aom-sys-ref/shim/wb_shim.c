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

/* Tile-info frame-header component, transcribed control flow over the real
 * aom_wb (assert(width_sb==0) omitted — debug-only, no byte effect). */
static void shim_wb_write_uniform(struct aom_write_bit_buffer *wb, int n, int v) {
  const int l = get_unsigned_bits(n);
  const int m = (1 << l) - n;
  if (l == 0) return;
  if (v < m) {
    aom_wb_write_literal(wb, v, l - 1);
  } else {
    aom_wb_write_literal(wb, m + ((v - m) >> 1), l - 1);
    aom_wb_write_literal(wb, (v - m) & 1, 1);
  }
}

uint32_t shim_write_tile_info(int mi_cols, int mi_rows, int mib_size_log2,
                              int uniform_spacing, int log2_cols, int min_log2_cols,
                              int max_log2_cols, int log2_rows, int min_log2_rows,
                              int max_log2_rows, int cols, int rows,
                              const int *col_start_sb, const int *row_start_sb,
                              int max_width_sb, int max_height_sb, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  int width_sb = CEIL_POWER_OF_TWO(mi_cols, mib_size_log2);
  int height_sb = CEIL_POWER_OF_TWO(mi_rows, mib_size_log2);
  aom_wb_write_bit(&wb, uniform_spacing);
  if (uniform_spacing) {
    int ones = log2_cols - min_log2_cols;
    while (ones--) aom_wb_write_bit(&wb, 1);
    if (log2_cols < max_log2_cols) aom_wb_write_bit(&wb, 0);
    ones = log2_rows - min_log2_rows;
    while (ones--) aom_wb_write_bit(&wb, 1);
    if (log2_rows < max_log2_rows) aom_wb_write_bit(&wb, 0);
  } else {
    for (int i = 0; i < cols; i++) {
      int size_sb = col_start_sb[i + 1] - col_start_sb[i];
      shim_wb_write_uniform(&wb, AOMMIN(width_sb, max_width_sb), size_sb - 1);
      width_sb -= size_sb;
    }
    for (int i = 0; i < rows; i++) {
      int size_sb = row_start_sb[i + 1] - row_start_sb[i];
      shim_wb_write_uniform(&wb, AOMMIN(height_sb, max_height_sb), size_sb - 1);
      height_sb -= size_sb;
    }
  }
  /* write_tile_info trailing (saved_wb copy has no byte effect) */
  if (rows * cols > 1) {
    aom_wb_write_literal(&wb, 0, log2_cols + log2_rows);
    aom_wb_write_literal(&wb, 3, 2);
  }
  return aom_wb_bytes_written(&wb);
}

/* Transcribed control flow of encode_restoration_mode over the real aom_wb
 * (debug-only asserts omitted — no byte effect). RESTORE_* = 0..3. */
uint32_t shim_encode_restoration_mode(int enable_restoration, int allow_intrabc,
                                      const int *frame_restoration_type, int sb_size_128,
                                      const int *restoration_unit_size, int ssx, int ssy,
                                      int num_planes, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  if (!enable_restoration || allow_intrabc) return aom_wb_bytes_written(&wb);
  int all_none = 1, chroma_none = 1;
  for (int p = 0; p < num_planes; ++p) {
    int ft = frame_restoration_type[p];
    if (ft != 0) { all_none = 0; chroma_none &= p == 0; }
    switch (ft) {
      case 0: aom_wb_write_bit(&wb, 0); aom_wb_write_bit(&wb, 0); break;
      case 1: aom_wb_write_bit(&wb, 1); aom_wb_write_bit(&wb, 0); break;
      case 2: aom_wb_write_bit(&wb, 1); aom_wb_write_bit(&wb, 1); break;
      case 3: aom_wb_write_bit(&wb, 0); aom_wb_write_bit(&wb, 1); break;
      default: break;
    }
  }
  if (!all_none) {
    int sb_size = sb_size_128 ? 128 : 64;
    int rus = restoration_unit_size[0];
    if (sb_size == 64) aom_wb_write_bit(&wb, rus > 64);
    if (rus > 64) aom_wb_write_bit(&wb, rus > 128);
  }
  if (num_planes > 1) {
    int s = AOMMIN(ssx, ssy);
    if (s && !chroma_none)
      aom_wb_write_bit(&wb, restoration_unit_size[1] != restoration_unit_size[0]);
  }
  return aom_wb_bytes_written(&wb);
}

/* Transcribed control flow of the frame-header delta-q/delta-lf block +
 * write_tx_mode over the real aom_wb (debug-only asserts + xd side effects
 * omitted — no byte effect). */
uint32_t shim_write_delta_q_params(int base_qindex, int delta_q_present, int delta_q_res,
                                   int allow_intrabc, int delta_lf_present, int delta_lf_res,
                                   int delta_lf_multi, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  if (base_qindex > 0) {
    aom_wb_write_bit(&wb, delta_q_present);
    if (delta_q_present) {
      aom_wb_write_literal(&wb, get_msb(delta_q_res), 2);
      if (!allow_intrabc) aom_wb_write_bit(&wb, delta_lf_present);
      if (delta_lf_present) {
        aom_wb_write_literal(&wb, get_msb(delta_lf_res), 2);
        aom_wb_write_bit(&wb, delta_lf_multi);
      }
    }
  }
  return aom_wb_bytes_written(&wb);
}

uint32_t shim_write_tx_mode(int coded_lossless, int tx_mode_select, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  if (!coded_lossless) aom_wb_write_bit(&wb, tx_mode_select);
  return aom_wb_bytes_written(&wb);
}

/* write_film_grain_params, transcribed control flow over the real aom_wb. The
 * !update_parameters ref search is encoder logic (no byte effect beyond ref_idx),
 * so ref_idx is passed in. Scalars are packed in s[] (see the Rust binding). */
uint32_t shim_write_film_grain_params(const int *s, const int *spy, const int *spcb,
                                      const int *spcr, const int *ary, const int *arcb,
                                      const int *arcr, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  const int apply_grain = s[0], random_seed = s[1], is_inter = s[2];
  const int update_parameters = s[3], ref_idx = s[4], num_y_points = s[5];
  const int monochrome = s[6], chroma_from_luma = s[7], ssx = s[8], ssy = s[9];
  const int num_cb_points = s[10], num_cr_points = s[11], scaling_shift = s[12];
  const int ar_coeff_lag = s[13], ar_coeff_shift = s[14], grain_scale_shift = s[15];
  const int cb_mult = s[16], cb_luma_mult = s[17], cb_offset = s[18];
  const int cr_mult = s[19], cr_luma_mult = s[20], cr_offset = s[21];
  const int overlap_flag = s[22], clip_to_restricted_range = s[23];

  aom_wb_write_bit(&wb, apply_grain);
  if (!apply_grain) return aom_wb_bytes_written(&wb);
  aom_wb_write_literal(&wb, random_seed, 16);
  if (is_inter) aom_wb_write_bit(&wb, update_parameters);
  if (!update_parameters) {
    aom_wb_write_literal(&wb, ref_idx, 3);
    return aom_wb_bytes_written(&wb);
  }
  aom_wb_write_literal(&wb, num_y_points, 4);
  for (int i = 0; i < num_y_points; i++) {
    aom_wb_write_literal(&wb, spy[i * 2 + 0], 8);
    aom_wb_write_literal(&wb, spy[i * 2 + 1], 8);
  }
  if (!monochrome) aom_wb_write_bit(&wb, chroma_from_luma);
  if (monochrome || chroma_from_luma || (ssx == 1 && ssy == 1 && num_y_points == 0)) {
    /* chroma points absent */
  } else {
    aom_wb_write_literal(&wb, num_cb_points, 4);
    for (int i = 0; i < num_cb_points; i++) {
      aom_wb_write_literal(&wb, spcb[i * 2 + 0], 8);
      aom_wb_write_literal(&wb, spcb[i * 2 + 1], 8);
    }
    aom_wb_write_literal(&wb, num_cr_points, 4);
    for (int i = 0; i < num_cr_points; i++) {
      aom_wb_write_literal(&wb, spcr[i * 2 + 0], 8);
      aom_wb_write_literal(&wb, spcr[i * 2 + 1], 8);
    }
  }
  aom_wb_write_literal(&wb, scaling_shift - 8, 2);
  aom_wb_write_literal(&wb, ar_coeff_lag, 2);
  int num_pos_luma = 2 * ar_coeff_lag * (ar_coeff_lag + 1);
  int num_pos_chroma = num_pos_luma;
  if (num_y_points > 0) ++num_pos_chroma;
  if (num_y_points)
    for (int i = 0; i < num_pos_luma; i++) aom_wb_write_literal(&wb, ary[i] + 128, 8);
  if (num_cb_points || chroma_from_luma)
    for (int i = 0; i < num_pos_chroma; i++) aom_wb_write_literal(&wb, arcb[i] + 128, 8);
  if (num_cr_points || chroma_from_luma)
    for (int i = 0; i < num_pos_chroma; i++) aom_wb_write_literal(&wb, arcr[i] + 128, 8);
  aom_wb_write_literal(&wb, ar_coeff_shift - 6, 2);
  aom_wb_write_literal(&wb, grain_scale_shift, 2);
  if (num_cb_points) {
    aom_wb_write_literal(&wb, cb_mult, 8);
    aom_wb_write_literal(&wb, cb_luma_mult, 8);
    aom_wb_write_literal(&wb, cb_offset, 9);
  }
  if (num_cr_points) {
    aom_wb_write_literal(&wb, cr_mult, 8);
    aom_wb_write_literal(&wb, cr_luma_mult, 8);
    aom_wb_write_literal(&wb, cr_offset, 9);
  }
  aom_wb_write_bit(&wb, overlap_flag);
  aom_wb_write_bit(&wb, clip_to_restricted_range);
  return aom_wb_bytes_written(&wb);
}

/* Directly exercises the real aom_wb_write_signed_primitive_refsubexpfin so the
 * Rust port of the subexpfin primitive chain is validated on its own. */
uint32_t shim_wb_signed_subexpfin(int n, int k, int ref, int v, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  aom_wb_write_signed_primitive_refsubexpfin(&wb, (uint16_t)n, (uint16_t)k,
                                             (int16_t)ref, (int16_t)v);
  return aom_wb_bytes_written(&wb);
}

/* write_global_motion (+ _params), transcribed control flow over the real aom_wb
 * (the assert(type!=TRANSLATION) spec-bug workaround is omitted). Loops the 7
 * inter refs; wmmat/refmat are flat [7*6], wmtype is [7]. */
#include "av1/common/mv.h" /* GM_* / SUBEXPFIN_K / prec constants */
uint32_t shim_write_global_motion(const int *wmtype, const int *wmmat,
                                  const int *refmat, int allow_hp, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  for (int f = 0; f < 7; f++) {
    const int type = wmtype[f];
    const int *m = wmmat + f * 6;
    const int *r = refmat + f * 6;
    aom_wb_write_bit(&wb, type != IDENTITY);
    if (type != IDENTITY) {
      aom_wb_write_bit(&wb, type == ROTZOOM);
      if (type != ROTZOOM) aom_wb_write_bit(&wb, type == TRANSLATION);
    }
    if (type >= ROTZOOM) {
      aom_wb_write_signed_primitive_refsubexpfin(
          &wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K,
          (r[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS),
          (m[2] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
      aom_wb_write_signed_primitive_refsubexpfin(
          &wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K, (r[3] >> GM_ALPHA_PREC_DIFF),
          (m[3] >> GM_ALPHA_PREC_DIFF));
    }
    if (type >= AFFINE) {
      aom_wb_write_signed_primitive_refsubexpfin(
          &wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K, (r[4] >> GM_ALPHA_PREC_DIFF),
          (m[4] >> GM_ALPHA_PREC_DIFF));
      aom_wb_write_signed_primitive_refsubexpfin(
          &wb, GM_ALPHA_MAX + 1, SUBEXPFIN_K,
          (r[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS),
          (m[5] >> GM_ALPHA_PREC_DIFF) - (1 << GM_ALPHA_PREC_BITS));
    }
    if (type >= TRANSLATION) {
      const int trans_bits = (type == TRANSLATION)
                                 ? GM_ABS_TRANS_ONLY_BITS - !allow_hp
                                 : GM_ABS_TRANS_BITS;
      const int trans_prec_diff = (type == TRANSLATION)
                                      ? GM_TRANS_ONLY_PREC_DIFF + !allow_hp
                                      : GM_TRANS_PREC_DIFF;
      aom_wb_write_signed_primitive_refsubexpfin(
          &wb, (1 << trans_bits) + 1, SUBEXPFIN_K, (r[0] >> trans_prec_diff),
          (m[0] >> trans_prec_diff));
      aom_wb_write_signed_primitive_refsubexpfin(
          &wb, (1 << trans_bits) + 1, SUBEXPFIN_K, (r[1] >> trans_prec_diff),
          (m[1] >> trans_prec_diff));
    }
  }
  return aom_wb_bytes_written(&wb);
}

/* write_sequence_header (+ write_sb_size), transcribed control flow over the real
 * aom_wb. Scalars packed in s[] (see the Rust binding order). */
uint32_t shim_write_sequence_header(const int *s, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  const int num_bits_width = s[0], num_bits_height = s[1];
  const int max_frame_width = s[2], max_frame_height = s[3];
  const int reduced = s[4], frame_id_present = s[5];
  const int delta_frame_id_length = s[6], frame_id_length = s[7], sb_size_128 = s[8];
  const int en_filter_intra = s[9], en_intra_edge = s[10], en_interintra = s[11];
  const int en_masked = s[12], en_warped = s[13], en_dual = s[14], en_order_hint = s[15];
  const int en_dist_wtd = s[16], en_ref_mvs = s[17], force_sct = s[18];
  const int force_int_mv = s[19], order_hint_bits_m1 = s[20];
  const int en_superres = s[21], en_cdef = s[22], en_restoration = s[23];

  aom_wb_write_literal(&wb, num_bits_width - 1, 4);
  aom_wb_write_literal(&wb, num_bits_height - 1, 4);
  aom_wb_write_literal(&wb, max_frame_width - 1, num_bits_width);
  aom_wb_write_literal(&wb, max_frame_height - 1, num_bits_height);
  if (!reduced) {
    aom_wb_write_bit(&wb, frame_id_present);
    if (frame_id_present) {
      aom_wb_write_literal(&wb, delta_frame_id_length - 2, 4);
      aom_wb_write_literal(&wb, frame_id_length - delta_frame_id_length - 1, 3);
    }
  }
  aom_wb_write_bit(&wb, sb_size_128);
  aom_wb_write_bit(&wb, en_filter_intra);
  aom_wb_write_bit(&wb, en_intra_edge);
  if (!reduced) {
    aom_wb_write_bit(&wb, en_interintra);
    aom_wb_write_bit(&wb, en_masked);
    aom_wb_write_bit(&wb, en_warped);
    aom_wb_write_bit(&wb, en_dual);
    aom_wb_write_bit(&wb, en_order_hint);
    if (en_order_hint) {
      aom_wb_write_bit(&wb, en_dist_wtd);
      aom_wb_write_bit(&wb, en_ref_mvs);
    }
    if (force_sct == 2) {
      aom_wb_write_bit(&wb, 1);
    } else {
      aom_wb_write_bit(&wb, 0);
      aom_wb_write_bit(&wb, force_sct);
    }
    if (force_sct > 0) {
      if (force_int_mv == 2) {
        aom_wb_write_bit(&wb, 1);
      } else {
        aom_wb_write_bit(&wb, 0);
        aom_wb_write_bit(&wb, force_int_mv);
      }
    }
    if (en_order_hint) aom_wb_write_literal(&wb, order_hint_bits_m1, 3);
  }
  aom_wb_write_bit(&wb, en_superres);
  aom_wb_write_bit(&wb, en_cdef);
  aom_wb_write_bit(&wb, en_restoration);
  return aom_wb_bytes_written(&wb);
}

/* write_ext_tile_info, transcribed control flow over the real aom_wb. pre_bits
 * zero bits are written first so the byte-alignment padding is exercised from an
 * arbitrary starting offset (the saved_wb snapshot has no byte effect). */
uint32_t shim_write_ext_tile_info(int pre_bits, int rows, int cols, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  for (int i = 0; i < pre_bits; i++) aom_wb_write_bit(&wb, 0);
  int mod = wb.bit_offset % 8;
  if (mod > 0) aom_wb_write_literal(&wb, 0, 8 - mod);
  if (rows * cols > 1) {
    aom_wb_write_literal(&wb, 0, 2);
    aom_wb_write_literal(&wb, 0, 2);
  }
  return aom_wb_bytes_written(&wb);
}

/* write_color_config (+ write_bitdepth), transcribed control flow over the real
 * aom_wb (spec asserts omitted — no byte effect). Scalars packed in c[]. */
uint32_t shim_write_color_config(const int *c, uint8_t *out) {
  struct aom_write_bit_buffer wb = { out, 0 };
  const int bit_depth = c[0], profile = c[1], monochrome = c[2];
  const int cp = c[3], tc = c[4], mc = c[5], color_range = c[6];
  const int ssx = c[7], ssy = c[8], chroma_pos = c[9], sep_uv = c[10];

  /* write_bitdepth */
  aom_wb_write_bit(&wb, bit_depth != 8);
  if (profile == 2 && bit_depth != 8) aom_wb_write_bit(&wb, bit_depth == 12);

  if (profile != 1) aom_wb_write_bit(&wb, monochrome);
  if (cp == 2 && tc == 2 && mc == 2) {
    aom_wb_write_bit(&wb, 0);
  } else {
    aom_wb_write_bit(&wb, 1);
    aom_wb_write_literal(&wb, cp, 8);
    aom_wb_write_literal(&wb, tc, 8);
    aom_wb_write_literal(&wb, mc, 8);
  }
  if (monochrome) {
    aom_wb_write_bit(&wb, color_range);
    return aom_wb_bytes_written(&wb);
  }
  int is_srgb = (cp == 1 && tc == 13 && mc == 0);
  if (!is_srgb) {
    aom_wb_write_bit(&wb, color_range);
    if (profile == 2 && bit_depth == 12) {
      aom_wb_write_bit(&wb, ssx);
      if (ssx != 0) aom_wb_write_bit(&wb, ssy);
    }
    if (ssx == 1 && ssy == 1) aom_wb_write_literal(&wb, chroma_pos, 2);
  }
  aom_wb_write_bit(&wb, sep_uv);
  return aom_wb_bytes_written(&wb);
}
