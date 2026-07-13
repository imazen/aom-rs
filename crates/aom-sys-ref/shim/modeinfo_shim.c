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

/* Transcribed body of write_partition over the pristine C od_ec + update_cdf: every
 * symbol write is aom_write_symbol (encode + adapt) or aom_write_cdf (encode only, on
 * the gathered edge CDF). Returns the coded bytes + the adapted partition CDF. */
#include "aom_dsp/entenc.h"
#include "aom_dsp/prob.h"
uint32_t shim_write_partition(uint16_t *partition_cdf, int cdf_len, int p, int has_rows,
                              int has_cols, int bsize, uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 1024);
  if (bsize >= BLOCK_8X8) {
    if (has_rows && has_cols) {
      od_ec_encode_cdf_q15(&ec, p, partition_cdf, cdf_len);
      update_cdf(partition_cdf, p, cdf_len);
    } else if (!has_rows && has_cols) {
      aom_cdf_prob cdf[2];
      partition_gather_vert_alike(cdf, partition_cdf, (BLOCK_SIZE)bsize);
      od_ec_encode_cdf_q15(&ec, p == PARTITION_SPLIT, cdf, 2);
    } else if (has_rows && !has_cols) {
      aom_cdf_prob cdf[2];
      partition_gather_horz_alike(cdf, partition_cdf, (BLOCK_SIZE)bsize);
      od_ec_encode_cdf_q15(&ec, p == PARTITION_SPLIT, cdf, 2);
    }
  }
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < cdf_len + 1; i++) out_cdf[i] = partition_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

#include "av1/common/pred_common.h" /* av1_get_skip_txfm_context */
/* Facade for av1_get_skip_txfm_context: two stack MB_MODE_INFO neighbours (present
 * flags gate the NULL case) with their skip_txfm set, called through the real fn. */
int shim_skip_txfm_context(int above_present, int above_skip, int left_present,
                           int left_skip) {
  MB_MODE_INFO above_mi, left_mi;
  MACROBLOCKD xd;
  above_mi.skip_txfm = above_skip;
  left_mi.skip_txfm = left_skip;
  xd.above_mbmi = above_present ? &above_mi : (MB_MODE_INFO *)0;
  xd.left_mbmi = left_present ? &left_mi : (MB_MODE_INFO *)0;
  return av1_get_skip_txfm_context(&xd);
}

/* Transcribed write_skip symbol over the pristine C od_ec + update_cdf. seg_skip
 * active returns 1 with nothing coded. */
uint32_t shim_write_skip(uint16_t *skip_cdf, int seg_skip_active, int skip_txfm,
                         uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  if (!seg_skip_active) {
    od_ec_encode_cdf_q15(&ec, skip_txfm, skip_cdf, 2);
    update_cdf(skip_cdf, skip_txfm, 2);
  }
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) out_cdf[i] = skip_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

/* Transcribed write_delta_qindex over pristine C od_ec: symbol + exp-Golomb literals
 * + sign, matching aom_write_symbol / aom_write_literal / aom_write_bit. */
#include "aom_ports/bitops.h" /* get_msb */
static void mi_bit(od_ec_enc *ec, int bit) {
  int p = (0x7FFFFF - (128 << 15) + 128) >> 8;
  od_ec_encode_bool_q15(ec, bit, p);
}
static void mi_literal(od_ec_enc *ec, int data, int bits) {
  for (int b = bits - 1; b >= 0; b--) mi_bit(ec, (data >> b) & 1);
}
uint32_t shim_write_delta_qindex(uint16_t *delta_q_cdf, int delta_qindex, uint8_t *out,
                                 uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  int sign = delta_qindex < 0;
  int a = sign ? -delta_qindex : delta_qindex;
  int smallval = a < DELTA_Q_SMALL;
  int sym = a < DELTA_Q_SMALL ? a : DELTA_Q_SMALL;
  od_ec_encode_cdf_q15(&ec, sym, delta_q_cdf, DELTA_Q_PROBS + 1);
  update_cdf(delta_q_cdf, sym, DELTA_Q_PROBS + 1);
  if (!smallval) {
    int rem_bits = get_msb(a - 1);
    int thr = (1 << rem_bits) + 1;
    mi_literal(&ec, rem_bits - 1, 3);
    mi_literal(&ec, a - thr, rem_bits);
  }
  if (a > 0) mi_bit(&ec, sign);
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < DELTA_Q_PROBS + 2; i++) out_cdf[i] = delta_q_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

/* Transcribed write_delta_lflevel over pristine C od_ec (DELTA_LF_* constants).
 * The multi/single CDF selection is the caller's; the selected CDF is passed in. */
uint32_t shim_write_delta_lflevel(uint16_t *delta_lf_cdf, int delta_lflevel, uint8_t *out,
                                  uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  int sign = delta_lflevel < 0;
  int a = sign ? -delta_lflevel : delta_lflevel;
  int smallval = a < DELTA_LF_SMALL;
  int sym = a < DELTA_LF_SMALL ? a : DELTA_LF_SMALL;
  od_ec_encode_cdf_q15(&ec, sym, delta_lf_cdf, DELTA_LF_PROBS + 1);
  update_cdf(delta_lf_cdf, sym, DELTA_LF_PROBS + 1);
  if (!smallval) {
    int rem_bits = get_msb(a - 1);
    int thr = (1 << rem_bits) + 1;
    mi_literal(&ec, rem_bits - 1, 3);
    mi_literal(&ec, a - thr, rem_bits);
  }
  if (a > 0) mi_bit(&ec, sign);
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < DELTA_LF_PROBS + 2; i++) out_cdf[i] = delta_lf_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

/* Transcribed write_cfl_alphas over pristine C od_ec, using the real CFL_* macros.
 * cfl_alpha_cdf is passed flat [6][17]; sign + up-to-two magnitude CDFs adapt. */
uint32_t shim_write_cfl_alphas(uint16_t *cfl_sign_cdf, uint16_t *cfl_alpha_cdf, int idx,
                               int joint_sign, uint8_t *out, uint16_t *out_sign_cdf,
                               uint16_t *out_alpha_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  od_ec_encode_cdf_q15(&ec, joint_sign, cfl_sign_cdf, CFL_JOINT_SIGNS);
  update_cdf(cfl_sign_cdf, joint_sign, CFL_JOINT_SIGNS);
  if (CFL_SIGN_U(joint_sign) != CFL_SIGN_ZERO) {
    uint16_t *cdf_u = cfl_alpha_cdf + CFL_CONTEXT_U(joint_sign) * 17;
    od_ec_encode_cdf_q15(&ec, CFL_IDX_U(idx), cdf_u, CFL_ALPHABET_SIZE);
    update_cdf(cdf_u, CFL_IDX_U(idx), CFL_ALPHABET_SIZE);
  }
  if (CFL_SIGN_V(joint_sign) != CFL_SIGN_ZERO) {
    uint16_t *cdf_v = cfl_alpha_cdf + CFL_CONTEXT_V(joint_sign) * 17;
    od_ec_encode_cdf_q15(&ec, CFL_IDX_V(idx), cdf_v, CFL_ALPHABET_SIZE);
    update_cdf(cdf_v, CFL_IDX_V(idx), CFL_ALPHABET_SIZE);
  }
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < 9; i++) out_sign_cdf[i] = cfl_sign_cdf[i];
  for (int i = 0; i < 6 * 17; i++) out_alpha_cdf[i] = cfl_alpha_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

/* get_y_mode_cdf context via the real intra_mode_context table + the block-mode
 * neighbour rule (absent => DC_PRED). Returns (above_ctx<<8)|left_ctx. */
#include "av1/common/common_data.h" /* intra_mode_context */
int shim_get_y_mode_ctx(int above_present, int above_mode, int left_present,
                        int left_mode) {
  int a = above_present ? above_mode : 0; /* DC_PRED */
  int l = left_present ? left_mode : 0;
  return (intra_mode_context[a] << 8) | intra_mode_context[l];
}

/* write_intra_y_mode_kf symbol (INTRA_MODES) over pristine C od_ec. */
uint32_t shim_write_intra_y_mode_kf(uint16_t *kf_y_cdf, int mode, uint8_t *out,
                                    uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  od_ec_encode_cdf_q15(&ec, mode, kf_y_cdf, INTRA_MODES);
  update_cdf(kf_y_cdf, mode, INTRA_MODES);
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < INTRA_MODES + 1; i++) out_cdf[i] = kf_y_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

int shim_size_group_lookup(int bsize) { return size_group_lookup[bsize]; }

/* write_intra_uv_mode symbol (UV_INTRA_MODES - !cfl_allowed) over pristine C od_ec. */
uint32_t shim_write_intra_uv_mode(uint16_t *uv_mode_cdf, int uv_mode, int cfl_allowed,
                                  uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  int n = UV_INTRA_MODES - !cfl_allowed;
  od_ec_encode_cdf_q15(&ec, uv_mode, uv_mode_cdf, n);
  update_cdf(uv_mode_cdf, uv_mode, n);
  uint32_t nb = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < UV_INTRA_MODES + 1; i++) out_cdf[i] = uv_mode_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* write_inter_mode: 3-symbol cascade over pristine C od_ec. CDFs flat:
 * newmv[6][3], zeromv[2][3], refmv[6][3]. */
uint32_t shim_write_inter_mode(uint16_t *newmv_cdf, uint16_t *zeromv_cdf,
                               uint16_t *refmv_cdf, int mode, int mode_ctx, uint8_t *out,
                               uint16_t *out_newmv, uint16_t *out_zeromv,
                               uint16_t *out_refmv) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  int newmv_ctx = mode_ctx & 7;
  uint16_t *nc = newmv_cdf + newmv_ctx * 3;
  od_ec_encode_cdf_q15(&ec, mode != NEWMV, nc, 2);
  update_cdf(nc, mode != NEWMV, 2);
  if (mode != NEWMV) {
    int zeromv_ctx = (mode_ctx >> 3) & 1;
    uint16_t *zc = zeromv_cdf + zeromv_ctx * 3;
    od_ec_encode_cdf_q15(&ec, mode != GLOBALMV, zc, 2);
    update_cdf(zc, mode != GLOBALMV, 2);
    if (mode != GLOBALMV) {
      int refmv_ctx = (mode_ctx >> 4) & 15;
      uint16_t *rc = refmv_cdf + refmv_ctx * 3;
      od_ec_encode_cdf_q15(&ec, mode != NEARESTMV, rc, 2);
      update_cdf(rc, mode != NEARESTMV, 2);
    }
  }
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < 6 * 3; i++) out_newmv[i] = newmv_cdf[i];
  for (int i = 0; i < 2 * 3; i++) out_zeromv[i] = zeromv_cdf[i];
  for (int i = 0; i < 6 * 3; i++) out_refmv[i] = refmv_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

/* write_drl_idx over pristine C od_ec, calling the REAL av1_drl_ctx +
 * have_nearmv_in_inter_mode. drl_cdf flat [3][3]; weight[4]. */
#include "av1/common/mvref_common.h"
uint32_t shim_write_drl_idx(uint16_t *drl_cdf, int mode, int ref_mv_idx, int ref_mv_count,
                            const uint16_t *weight, uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  int new_mv = (mode == NEWMV || mode == NEW_NEWMV);
  if (new_mv) {
    for (int idx = 0; idx < 2; ++idx) {
      if (ref_mv_count > idx + 1) {
        uint8_t ctx = av1_drl_ctx(weight, idx);
        uint16_t *c = drl_cdf + ctx * 3;
        od_ec_encode_cdf_q15(&ec, ref_mv_idx != idx, c, 2);
        update_cdf(c, ref_mv_idx != idx, 2);
        if (ref_mv_idx == idx) goto done;
      }
    }
    goto done;
  }
  if (have_nearmv_in_inter_mode(mode)) {
    for (int idx = 1; idx < 3; ++idx) {
      if (ref_mv_count > idx + 1) {
        uint8_t ctx = av1_drl_ctx(weight, idx);
        uint16_t *c = drl_cdf + ctx * 3;
        od_ec_encode_cdf_q15(&ec, ref_mv_idx != (idx - 1), c, 2);
        update_cdf(c, ref_mv_idx != (idx - 1), 2);
        if (ref_mv_idx == (idx - 1)) goto done;
      }
    }
  }
done:;
  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  for (uint32_t i = 0; i < n; i++) out[i] = buf[i];
  for (int i = 0; i < 3 * 3; i++) out_cdf[i] = drl_cdf[i];
  od_ec_enc_clear(&ec);
  return n;
}

/* MV class/joint math via the real av1_get_mv_class / av1_get_mv_joint. */
#include "av1/encoder/encodemv.h"
int shim_get_mv_joint(int row, int col) {
  MV mv = { (int16_t)row, (int16_t)col };
  return av1_get_mv_joint(&mv);
}
/* returns (class << 20) | (offset & 0xFFFFF) */
int shim_get_mv_class(int z) {
  int offset = 0;
  int c = av1_get_mv_class(z, &offset);
  return (c << 20) | (offset & 0xFFFFF);
}

/* encode_mv_component over pristine C od_ec + update_cdf. CDF blob layout matches the
 * Rust: sign(3)/classes(12)/class0(3)/bits[10](30)/class0_fp[2](10)/fp(5)/class0_hp(3)/
 * hp(3) = 69. Uses the real av1_get_mv_class. */
uint32_t shim_encode_mv_component(uint16_t *cdf, int comp, int precision, uint8_t *out,
                                  uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  int sign = comp < 0;
  int mag = sign ? -comp : comp;
  int offset;
  int mv_class = av1_get_mv_class(mag - 1, &offset);
  int d = offset >> 3, fr = (offset >> 1) & 3, hp = offset & 1;
#define SYM(base, sy, n) do { od_ec_encode_cdf_q15(&ec, sy, cdf + (base), n); update_cdf(cdf + (base), sy, n); } while (0)
  SYM(0, sign, 2);
  SYM(3, mv_class, 11);
  if (mv_class == 0) {
    SYM(15, d, 2);
  } else {
    int n = mv_class;
    for (int i = 0; i < n; ++i) SYM(18 + i * 3, (d >> i) & 1, 2);
  }
  if (precision > MV_SUBPEL_NONE) {
    if (mv_class == 0) SYM(48 + d * 5, fr, 4);
    else SYM(58, fr, 4);
  }
  if (precision > MV_SUBPEL_LOW_PRECISION) {
    if (mv_class == 0) SYM(63, hp, 2);
    else SYM(66, hp, 2);
  }
#undef SYM
  uint32_t nb = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 69; i++) out_cdf[i] = cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* av1_encode_mv transcribed over pristine C od_ec: joint symbol + the two components,
 * using the REAL av1_get_mv_joint / mv_joint_vertical|horizontal / av1_get_mv_class. */
static void mv_comp_wb(od_ec_enc *ec, uint16_t *cdf, int comp, int precision) {
  int sign = comp < 0;
  int mag = sign ? -comp : comp;
  int offset;
  int mv_class = av1_get_mv_class(mag - 1, &offset);
  int d = offset >> 3, fr = (offset >> 1) & 3, hp = offset & 1;
#define S(base, sy, n) do { od_ec_encode_cdf_q15(ec, sy, cdf + (base), n); update_cdf(cdf + (base), sy, n); } while (0)
  S(0, sign, 2);
  S(3, mv_class, 11);
  if (mv_class == 0) S(15, d, 2);
  else { for (int i = 0; i < mv_class; ++i) S(18 + i * 3, (d >> i) & 1, 2); }
  if (precision > MV_SUBPEL_NONE) { if (mv_class == 0) S(48 + d * 5, fr, 4); else S(58, fr, 4); }
  if (precision > MV_SUBPEL_LOW_PRECISION) { if (mv_class == 0) S(63, hp, 2); else S(66, hp, 2); }
#undef S
}
uint32_t shim_encode_mv(uint16_t *joints_cdf, uint16_t *comp0, uint16_t *comp1,
                        int diff_row, int diff_col, int usehp, uint8_t *out,
                        uint16_t *out_joints, uint16_t *out_comp0, uint16_t *out_comp1) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  MV diff = { (int16_t)diff_row, (int16_t)diff_col };
  int j = av1_get_mv_joint(&diff);
  od_ec_encode_cdf_q15(&ec, j, joints_cdf, 4);
  update_cdf(joints_cdf, j, 4);
  if (mv_joint_vertical(j)) mv_comp_wb(&ec, comp0, diff.row, usehp);
  if (mv_joint_horizontal(j)) mv_comp_wb(&ec, comp1, diff.col, usehp);
  uint32_t nb = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 5; i++) out_joints[i] = joints_cdf[i];
  for (int i = 0; i < 69; i++) out_comp0[i] = comp0[i];
  for (int i = 0; i < 69; i++) out_comp1[i] = comp1[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* write_angle_delta symbol (2*MAX_ANGLE_DELTA+1=7) over pristine C od_ec. */
uint32_t shim_write_angle_delta(uint16_t *cdf, int angle_delta, uint8_t *out,
                                uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  int n = 2 * MAX_ANGLE_DELTA + 1;
  od_ec_encode_cdf_q15(&ec, angle_delta + MAX_ANGLE_DELTA, cdf, n);
  update_cdf(cdf, angle_delta + MAX_ANGLE_DELTA, n);
  uint32_t nb = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 8; i++) out_cdf[i] = cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

int shim_bsize_to_max_depth(int bsize) { return bsize_to_max_depth(bsize); }
int shim_bsize_to_tx_size_cat(int bsize) { return bsize_to_tx_size_cat(bsize); }

/* write_selected_tx_size symbol (max_depths+1) over pristine C od_ec. */
uint32_t shim_write_selected_tx_size(uint16_t *cdf, int bsize, int depth, int max_depths,
                                     uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  if (bsize > BLOCK_4X4) {
    od_ec_encode_cdf_q15(&ec, depth, cdf, max_depths + 1);
    update_cdf(cdf, depth, max_depths + 1);
  }
  uint32_t nb = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < max_depths + 2; i++) out_cdf[i] = cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* write_filter_intra_mode_info over pristine C od_ec. use_cdf(3) + mode_cdf(6). */
uint32_t shim_write_filter_intra(uint16_t *use_cdf, uint16_t *mode_cdf, int allowed,
                                 int use_fi, int mode, uint8_t *out, uint16_t *out_use,
                                 uint16_t *out_mode) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  if (allowed) {
    od_ec_encode_cdf_q15(&ec, use_fi, use_cdf, 2);
    update_cdf(use_cdf, use_fi, 2);
    if (use_fi) {
      od_ec_encode_cdf_q15(&ec, mode, mode_cdf, FILTER_INTRA_MODES);
      update_cdf(mode_cdf, mode, FILTER_INTRA_MODES);
    }
  }
  uint32_t nb = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) out_use[i] = use_cdf[i];
  for (int i = 0; i < 6; i++) out_mode[i] = mode_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

uint32_t shim_write_inter_compound_mode(uint16_t *cdf, int mode, uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  od_ec_encode_cdf_q15(&ec, mode - NEAREST_NEARESTMV, cdf, INTER_COMPOUND_MODES);
  update_cdf(cdf, mode - NEAREST_NEARESTMV, INTER_COMPOUND_MODES);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < INTER_COMPOUND_MODES + 1; i++) out_cdf[i] = cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

uint32_t shim_write_is_inter(uint16_t *cdf, int seg_ref, int seg_gmv, int is_inter, uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (!seg_ref) {
    if (!seg_gmv) { od_ec_encode_cdf_q15(&ec, is_inter, cdf, 2); update_cdf(cdf, is_inter, 2); }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) out_cdf[i] = cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

uint32_t shim_write_motion_mode(uint16_t *obmc_cdf, uint16_t *mm_cdf, int last_allowed,
                                int mm, uint8_t *out, uint16_t *out_obmc, uint16_t *out_mm) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  switch (last_allowed) {
    case 0: break; /* SIMPLE_TRANSLATION */
    case 1: /* OBMC_CAUSAL */
      od_ec_encode_cdf_q15(&ec, mm == 1, obmc_cdf, 2);
      update_cdf(obmc_cdf, mm == 1, 2);
      break;
    default:
      od_ec_encode_cdf_q15(&ec, mm, mm_cdf, MOTION_MODES);
      update_cdf(mm_cdf, mm, MOTION_MODES);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) out_obmc[i] = obmc_cdf[i];
  for (int i = 0; i < 4; i++) out_mm[i] = mm_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

uint32_t shim_write_mb_interp_filter(uint16_t *cdf0, uint16_t *cdf1, int interp_needed,
                                     int is_switchable, int enable_dual, int f0, int f1,
                                     uint8_t *out, uint16_t *out0, uint16_t *out1) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (interp_needed && is_switchable) {
    od_ec_encode_cdf_q15(&ec, f0, cdf0, SWITCHABLE_FILTERS);
    update_cdf(cdf0, f0, SWITCHABLE_FILTERS);
    if (enable_dual) {
      od_ec_encode_cdf_q15(&ec, f1, cdf1, SWITCHABLE_FILTERS);
      update_cdf(cdf1, f1, SWITCHABLE_FILTERS);
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 4; i++) { out0[i] = cdf0[i]; out1[i] = cdf1[i]; }
  od_ec_enc_clear(&ec);
  return nb;
}

/* Facade for av1_get_intra_inter_context: two stack MB_MODE_INFO neighbours whose
 * ref_frame[0] (+ use_intrabc=0) drives is_inter_block, called through the real fn. */
#include "av1/common/pred_common.h"
int shim_get_intra_inter_context(int has_above, int above_inter, int has_left, int left_inter) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.use_intrabc = 0;
  lmi.use_intrabc = 0;
  ami.ref_frame[0] = above_inter ? LAST_FRAME : INTRA_FRAME;
  lmi.ref_frame[0] = left_inter ? LAST_FRAME : INTRA_FRAME;
  xd.above_mbmi = &ami;
  xd.left_mbmi = &lmi;
  xd.up_available = has_above;
  xd.left_available = has_left;
  return av1_get_intra_inter_context(&xd);
}

/* Facade for av1_get_reference_mode_context: two stack MB_MODE_INFO with ref_frame[0/1]
 * + use_intrabc, called through the real exported fn. */
int shim_get_reference_mode_context(int ha, int a_r0, int a_r1, int a_ibc, int hl,
                                    int l_r0, int l_r1, int l_ibc) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.ref_frame[0] = a_r0; ami.ref_frame[1] = a_r1; ami.use_intrabc = a_ibc;
  lmi.ref_frame[0] = l_r0; lmi.ref_frame[1] = l_r1; lmi.use_intrabc = l_ibc;
  xd.above_mbmi = &ami; xd.left_mbmi = &lmi;
  xd.up_available = ha; xd.left_available = hl;
  return av1_get_reference_mode_context(&xd);
}

int shim_get_comp_reference_type_context(int ha, int a_r0, int a_r1, int a_ibc, int hl,
                                         int l_r0, int l_r1, int l_ibc) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.ref_frame[0] = a_r0; ami.ref_frame[1] = a_r1; ami.use_intrabc = a_ibc;
  lmi.ref_frame[0] = l_r0; lmi.ref_frame[1] = l_r1; lmi.use_intrabc = l_ibc;
  xd.above_mbmi = &ami; xd.left_mbmi = &lmi;
  xd.up_available = ha; xd.left_available = hl;
  return av1_get_comp_reference_type_context(&xd);
}

/* Facade for the count-based single/comp ref contexts: set neighbors_ref_counts. */
int shim_single_ref_p1_context(const uint8_t *ref_counts) {
  MACROBLOCKD xd;
  for (int i = 0; i < 8; i++) xd.neighbors_ref_counts[i] = ref_counts[i];
  return av1_get_pred_context_single_ref_p1(&xd);
}

#define SINGLE_REF_SHIM(N) \
  int shim_single_ref_p##N##_context(const uint8_t *rc) { \
    MACROBLOCKD xd; \
    for (int i = 0; i < 8; i++) xd.neighbors_ref_counts[i] = rc[i]; \
    return av1_get_pred_context_single_ref_p##N(&xd); \
  }
SINGLE_REF_SHIM(2)
SINGLE_REF_SHIM(3)
SINGLE_REF_SHIM(4)
SINGLE_REF_SHIM(5)
SINGLE_REF_SHIM(6)
#undef SINGLE_REF_SHIM

#define UNI_REF_SHIM(N, FN) \
  int shim_##FN(const uint8_t *rc) { \
    MACROBLOCKD xd; \
    for (int i = 0; i < 8; i++) xd.neighbors_ref_counts[i] = rc[i]; \
    return av1_get_pred_context_uni_comp_ref_##N(&xd); \
  }
UNI_REF_SHIM(p, uni_comp_ref_p_context)
UNI_REF_SHIM(p1, uni_comp_ref_p1_context)
UNI_REF_SHIM(p2, uni_comp_ref_p2_context)
#undef UNI_REF_SHIM

/* write_ref_frames cascade over pristine C od_ec. cdfs flat [16][3]. */
static void rref_sym(od_ec_enc *ec, uint16_t *cdfs, int slot, int sym) {
  uint16_t *c = cdfs + slot * 3;
  od_ec_encode_cdf_q15(ec, sym, c, 2);
  update_cdf(c, sym, 2);
}
uint32_t shim_write_ref_frames(uint16_t *cdfs, int seg_ref, int seg_skipgmv,
                               int rmode_select, int comp_allowed, int is_compound,
                               int comp_ref_type, int ref0, int ref1, uint8_t *out,
                               uint16_t *out_cdfs) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (!(seg_ref || seg_skipgmv)) {
    if (rmode_select && comp_allowed) rref_sym(&ec, cdfs, 0, is_compound);
    if (is_compound) {
      rref_sym(&ec, cdfs, 1, comp_ref_type);
      if (comp_ref_type == 0) {
        int bit = (ref0 == 5);
        rref_sym(&ec, cdfs, 2, bit);
        if (!bit) {
          int bit1 = (ref1 == 3 || ref1 == 4);
          rref_sym(&ec, cdfs, 3, bit1);
          if (bit1) rref_sym(&ec, cdfs, 4, (ref1 == 4));
        }
        goto done;
      }
      int bit = (ref0 == 4 || ref0 == 3);
      rref_sym(&ec, cdfs, 5, bit);
      if (!bit) rref_sym(&ec, cdfs, 6, (ref0 == 2));
      else rref_sym(&ec, cdfs, 7, (ref0 == 4));
      int bit_bwd = (ref1 == 7);
      rref_sym(&ec, cdfs, 8, bit_bwd);
      if (!bit_bwd) rref_sym(&ec, cdfs, 9, (ref1 == 6));
    } else {
      int bit0 = (ref0 <= 7 && ref0 >= 5);
      rref_sym(&ec, cdfs, 10, bit0);
      if (bit0) {
        int bit1 = (ref0 == 7);
        rref_sym(&ec, cdfs, 11, bit1);
        if (!bit1) rref_sym(&ec, cdfs, 15, (ref0 == 6));
      } else {
        int bit2 = (ref0 == 3 || ref0 == 4);
        rref_sym(&ec, cdfs, 12, bit2);
        if (!bit2) rref_sym(&ec, cdfs, 13, (ref0 != 1));
        else rref_sym(&ec, cdfs, 14, (ref0 != 3));
      }
    }
  }
done:;
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 48; i++) out_cdfs[i] = cdfs[i];
  od_ec_enc_clear(&ec);
  return nb;
}

int av1_neg_interleave(int x, int ref, int max);
int shim_neg_interleave(int x, int ref, int max) { return av1_neg_interleave(x, ref, max); }

uint32_t shim_write_segment_id(uint16_t *cdf, int seg_enabled, int update_map,
                               int skip_txfm, int segment_id, int pred,
                               int last_active_segid, uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (seg_enabled && update_map && !skip_txfm) {
    int coded_id = av1_neg_interleave(segment_id, pred, last_active_segid + 1);
    od_ec_encode_cdf_q15(&ec, coded_id, cdf, MAX_SEGMENTS);
    update_cdf(cdf, coded_id, MAX_SEGMENTS);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 9; i++) out_cdf[i] = cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* write_intrabc_info: flag + av1_encode_dv (= encode_mv at MV_SUBPEL_NONE). */
uint32_t shim_write_intrabc_info(uint16_t *intrabc_cdf, uint16_t *joints, uint16_t *comp0,
                                 uint16_t *comp1, int use_intrabc, int diff_row,
                                 int diff_col, uint8_t *out, uint16_t *out_ibc,
                                 uint16_t *out_joints, uint16_t *out_c0, uint16_t *out_c1) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  od_ec_encode_cdf_q15(&ec, use_intrabc, intrabc_cdf, 2);
  update_cdf(intrabc_cdf, use_intrabc, 2);
  if (use_intrabc) {
    MV diff = { (int16_t)diff_row, (int16_t)diff_col };
    int j = av1_get_mv_joint(&diff);
    od_ec_encode_cdf_q15(&ec, j, joints, 4);
    update_cdf(joints, j, 4);
    if (mv_joint_vertical(j)) mv_comp_wb(&ec, comp0, diff.row, MV_SUBPEL_NONE);
    if (mv_joint_horizontal(j)) mv_comp_wb(&ec, comp1, diff.col, MV_SUBPEL_NONE);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) out_ibc[i] = intrabc_cdf[i];
  for (int i = 0; i < 5; i++) out_joints[i] = joints[i];
  for (int i = 0; i < 69; i++) { out_c0[i] = comp0[i]; out_c1[i] = comp1[i]; }
  od_ec_enc_clear(&ec);
  return nb;
}
