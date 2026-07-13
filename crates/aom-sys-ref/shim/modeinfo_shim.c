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

int shim_get_skip_mode_context(int ha, int a_sm, int hl, int l_sm) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.skip_mode = a_sm; lmi.skip_mode = l_sm;
  xd.above_mbmi = ha ? &ami : (MB_MODE_INFO *)0;
  xd.left_mbmi = hl ? &lmi : (MB_MODE_INFO *)0;
  return av1_get_skip_mode_context(&xd);
}

uint32_t shim_write_skip_mode(uint16_t *cdf, int frame_flag, int seg_skip, int comp_allowed,
                              int seg_ref_gmv, int skip_mode, uint8_t *out, uint16_t *out_cdf) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (frame_flag && !seg_skip && comp_allowed && !seg_ref_gmv) {
    od_ec_encode_cdf_q15(&ec, skip_mode, cdf, 2);
    update_cdf(cdf, skip_mode, 2);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) out_cdf[i] = cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- var-tx neighbour context helpers (av1_common_int.h) --- */

/* txfm_partition_context reads only *above_ctx / *left_ctx (single value each). */
int shim_txfm_partition_context(uint8_t above, uint8_t left, int bsize, int tx_size) {
  TXFM_CONTEXT a = (TXFM_CONTEXT)above, l = (TXFM_CONTEXT)left;
  return txfm_partition_context(&a, &l, (BLOCK_SIZE)bsize, (TX_SIZE)tx_size);
}

/* txfm_partition_update fills above_ctx[0..bw]=txw and left_ctx[0..bh]=txh. */
void shim_txfm_partition_update(uint8_t *above_ctx, uint8_t *left_ctx, int tx_size,
                                int txb_size) {
  txfm_partition_update((TXFM_CONTEXT *)above_ctx, (TXFM_CONTEXT *)left_ctx,
                        (TX_SIZE)tx_size, (TX_SIZE)txb_size);
}

/* --- write_tx_size_vartx recursion (av1/encoder/bitstream.c) --- */
/* Body copied verbatim from write_tx_size_vartx; aom_write_symbol(w,s,cdf,2) is
 * expanded to od_ec_encode_cdf_q15 + update_cdf (exactly what aom_write_symbol does
 * when CDF adaptation is on). All decisions call the real libaom helpers. */
static void shim_wtsv_rec(MACROBLOCKD *xd, const MB_MODE_INFO *mbmi, TX_SIZE tx_size,
                          int depth, int blk_row, int blk_col, od_ec_enc *ec,
                          uint16_t (*txfm_partition_cdf)[3]) {
  const int max_blocks_high = max_block_high(xd, mbmi->bsize, 0);
  const int max_blocks_wide = max_block_wide(xd, mbmi->bsize, 0);
  if (blk_row >= max_blocks_high || blk_col >= max_blocks_wide) return;

  if (depth == MAX_VARTX_DEPTH) {
    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size, tx_size);
    return;
  }

  const int ctx = txfm_partition_context(xd->above_txfm_context + blk_col,
                                         xd->left_txfm_context + blk_row,
                                         mbmi->bsize, tx_size);
  const int txb_size_index = av1_get_txb_size_index(mbmi->bsize, blk_row, blk_col);
  const int write_txfm_partition = tx_size == mbmi->inter_tx_size[txb_size_index];
  if (write_txfm_partition) {
    od_ec_encode_cdf_q15(ec, 0, txfm_partition_cdf[ctx], 2);
    update_cdf(txfm_partition_cdf[ctx], 0, 2);
    txfm_partition_update(xd->above_txfm_context + blk_col,
                          xd->left_txfm_context + blk_row, tx_size, tx_size);
  } else {
    const TX_SIZE sub_txs = sub_tx_size_map[tx_size];
    const int bsw = tx_size_wide_unit[sub_txs];
    const int bsh = tx_size_high_unit[sub_txs];
    od_ec_encode_cdf_q15(ec, 1, txfm_partition_cdf[ctx], 2);
    update_cdf(txfm_partition_cdf[ctx], 1, 2);
    if (sub_txs == TX_4X4) {
      txfm_partition_update(xd->above_txfm_context + blk_col,
                            xd->left_txfm_context + blk_row, sub_txs, tx_size);
      return;
    }
    for (int row = 0; row < tx_size_high_unit[tx_size]; row += bsh) {
      const int offsetr = blk_row + row;
      for (int col = 0; col < tx_size_wide_unit[tx_size]; col += bsw) {
        const int offsetc = blk_col + col;
        shim_wtsv_rec(xd, mbmi, sub_txs, depth + 1, offsetr, offsetc, ec,
                      txfm_partition_cdf);
      }
    }
  }
}

/* above/left are 32-slot (MAX_MIB_SIZE) neighbour txfm-context arrays; inter_tx_size
 * is the 16-entry per-txb chosen tx sizes; cdf is 21x3 flattened txfm_partition_cdf. */
uint32_t shim_write_tx_size_vartx(int bsize, int top_tx_size, const uint8_t *inter_tx_size,
                                  int mb_to_right_edge, int mb_to_bottom_edge,
                                  const uint8_t *above_in, const uint8_t *left_in,
                                  uint16_t *cdf, uint8_t *out, uint8_t *above_out,
                                  uint8_t *left_out, uint16_t *cdf_out) {
  MACROBLOCKD xd;
  MB_MODE_INFO mbmi;
  uint8_t above[32], left[32];
  uint16_t local_cdf[21][3];
  for (int i = 0; i < 32; i++) { above[i] = above_in[i]; left[i] = left_in[i]; }
  for (int i = 0; i < 21; i++) for (int j = 0; j < 3; j++) local_cdf[i][j] = cdf[i * 3 + j];
  mbmi.bsize = (BLOCK_SIZE)bsize;
  for (int i = 0; i < 16; i++) mbmi.inter_tx_size[i] = (TX_SIZE)inter_tx_size[i];
  xd.mb_to_right_edge = mb_to_right_edge;
  xd.mb_to_bottom_edge = mb_to_bottom_edge;
  xd.plane[0].subsampling_x = 0;
  xd.plane[0].subsampling_y = 0;
  xd.above_txfm_context = (TXFM_CONTEXT *)above;
  xd.left_txfm_context = (TXFM_CONTEXT *)left;

  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  shim_wtsv_rec(&xd, &mbmi, (TX_SIZE)top_tx_size, 0, 0, 0, &ec, local_cdf);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 32; i++) { above_out[i] = above[i]; left_out[i] = left[i]; }
  for (int i = 0; i < 21; i++) for (int j = 0; j < 3; j++) cdf_out[i * 3 + j] = local_cdf[i][j];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- palette signalling: contexts + flag/size symbols (no colours) --- */
#include "av1/common/pred_common.h" /* av1_get_palette_bsize_ctx / _mode_ctx */

int shim_get_palette_bsize_ctx(int bsize) {
  return av1_get_palette_bsize_ctx((BLOCK_SIZE)bsize);
}

int shim_get_palette_mode_ctx(int ha, int a_psize, int hl, int l_psize) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.palette_mode_info.palette_size[0] = (uint8_t)a_psize;
  lmi.palette_mode_info.palette_size[0] = (uint8_t)l_psize;
  xd.above_mbmi = ha ? &ami : (MB_MODE_INFO *)0;
  xd.left_mbmi = hl ? &lmi : (MB_MODE_INFO *)0;
  return av1_get_palette_mode_ctx(&xd);
}

/* Codes the palette Y/UV mode + size symbols exactly as write_palette_mode_info,
 * but omitting the colour payload (that is a separate port). PALETTE_MIN_SIZE=2,
 * PALETTE_SIZES=7. mode_dc / uv_dc gate the two planes. */
uint32_t shim_write_palette_flags_sizes(int mode_dc, int n_y, uint16_t *y_mode_cdf,
                                        uint16_t *y_size_cdf, int uv_dc, int n_uv,
                                        uint16_t *uv_mode_cdf, uint16_t *uv_size_cdf,
                                        uint8_t *out, uint16_t *o_ym, uint16_t *o_ys,
                                        uint16_t *o_um, uint16_t *o_us) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (mode_dc) {
    od_ec_encode_cdf_q15(&ec, n_y > 0, y_mode_cdf, 2);
    update_cdf(y_mode_cdf, n_y > 0, 2);
    if (n_y > 0) {
      od_ec_encode_cdf_q15(&ec, n_y - PALETTE_MIN_SIZE, y_size_cdf, PALETTE_SIZES);
      update_cdf(y_size_cdf, n_y - PALETTE_MIN_SIZE, PALETTE_SIZES);
    }
  }
  if (uv_dc) {
    od_ec_encode_cdf_q15(&ec, n_uv > 0, uv_mode_cdf, 2);
    update_cdf(uv_mode_cdf, n_uv > 0, 2);
    if (n_uv > 0) {
      od_ec_encode_cdf_q15(&ec, n_uv - PALETTE_MIN_SIZE, uv_size_cdf, PALETTE_SIZES);
      update_cdf(uv_size_cdf, n_uv - PALETTE_MIN_SIZE, PALETTE_SIZES);
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) { o_ym[i] = y_mode_cdf[i]; o_um[i] = uv_mode_cdf[i]; }
  for (int i = 0; i < 8; i++) { o_ys[i] = y_size_cdf[i]; o_us[i] = uv_size_cdf[i]; }
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- delta_encode_palette_colors (av1/encoder/bitstream.c) --- */
/* Body copied verbatim; aom_write_literal -> mi_literal. aom_ceil_log2 is the real
 * one from aom_ports/bitops.h (already included above). colors is ascending, deltas
 * >= min_val (caller-guaranteed). */
uint32_t shim_delta_encode_palette_colors(const int *colors, int num, int bit_depth,
                                          int min_val, uint8_t *out) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (num > 0) {
    mi_literal(&ec, colors[0], bit_depth);
    if (num > 1) {
      int max_delta = 0;
      int deltas[PALETTE_MAX_SIZE];
      memset(deltas, 0, sizeof(deltas));
      for (int i = 1; i < num; ++i) {
        const int delta = colors[i] - colors[i - 1];
        deltas[i - 1] = delta;
        if (delta > max_delta) max_delta = delta;
      }
      const int min_bits = bit_depth - 3;
      int cl = aom_ceil_log2(max_delta + 1 - min_val);
      int bits = cl > min_bits ? cl : min_bits;
      int range = (1 << bit_depth) - colors[0] - min_val;
      mi_literal(&ec, bits - min_bits, 2);
      for (int i = 0; i < num - 1; ++i) {
        mi_literal(&ec, deltas[i] - min_val, bits);
        range -= deltas[i];
        int clr = aom_ceil_log2(range);
        bits = bits < clr ? bits : clr;
      }
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- palette V-channel colours (av1/encoder/bitstream.c, write_palette_colors_uv) --- */
#include "av1/encoder/palette.h" /* av1_get_palette_delta_bits_v (exported) */

/* The V-plane portion of write_palette_colors_uv copied verbatim; aom_write_bit/
 * literal -> mi_bit/mi_literal; bits_v/zero_count/min_bits_v come from the real
 * av1_get_palette_delta_bits_v. colors_v need not be sorted. */
uint32_t shim_write_palette_colors_v(const uint16_t *colors_v, int n, int bit_depth,
                                     uint8_t *out) {
  PALETTE_MODE_INFO pmi;
  pmi.palette_size[1] = (uint8_t)n;
  for (int i = 0; i < n; i++) pmi.palette_colors[2 * PALETTE_MAX_SIZE + i] = colors_v[i];

  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  const int max_val = 1 << bit_depth;
  int zero_count = 0, min_bits_v = 0;
  int bits_v = av1_get_palette_delta_bits_v(&pmi, bit_depth, &zero_count, &min_bits_v);
  const int rate_using_delta = 2 + bit_depth + (bits_v + 1) * (n - 1) - zero_count;
  const int rate_using_raw = bit_depth * n;
  if (rate_using_delta < rate_using_raw) {
    mi_bit(&ec, 1);
    mi_literal(&ec, bits_v - min_bits_v, 2);
    mi_literal(&ec, colors_v[0], bit_depth);
    for (int i = 1; i < n; ++i) {
      if (colors_v[i] == colors_v[i - 1]) {
        mi_literal(&ec, 0, bits_v);
        continue;
      }
      const int delta = abs((int)colors_v[i] - colors_v[i - 1]);
      const int sign_bit = colors_v[i] < colors_v[i - 1];
      if (delta <= max_val - delta) {
        mi_literal(&ec, delta, bits_v);
        mi_bit(&ec, sign_bit);
      } else {
        mi_literal(&ec, max_val - delta, bits_v);
        mi_bit(&ec, !sign_bit);
      }
    }
  } else {
    mi_bit(&ec, 0);
    for (int i = 0; i < n; ++i) mi_literal(&ec, colors_v[i], bit_depth);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- palette colour cache: av1_get_palette_cache + av1_index_color_cache --- */
/* av1_get_palette_cache is a facade over the real fn (reads xd->above/left_mbmi
 * palettes, applies the SB-row-boundary gate on above via mb_to_top_edge, and merges
 * the two sorted neighbour lists into a sorted+deduped cache). */
int shim_get_palette_cache(int plane, int mb_to_top_edge, int ha, const uint16_t *a_colors,
                           int a_size0, int a_size1, int hl, const uint16_t *l_colors,
                           int l_size0, int l_size1, uint16_t *out_cache) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  for (int i = 0; i < 3 * PALETTE_MAX_SIZE; i++) {
    ami.palette_mode_info.palette_colors[i] = a_colors[i];
    lmi.palette_mode_info.palette_colors[i] = l_colors[i];
  }
  ami.palette_mode_info.palette_size[0] = (uint8_t)a_size0;
  ami.palette_mode_info.palette_size[1] = (uint8_t)a_size1;
  lmi.palette_mode_info.palette_size[0] = (uint8_t)l_size0;
  lmi.palette_mode_info.palette_size[1] = (uint8_t)l_size1;
  xd.above_mbmi = ha ? &ami : (MB_MODE_INFO *)0;
  xd.left_mbmi = hl ? &lmi : (MB_MODE_INFO *)0;
  xd.mb_to_top_edge = mb_to_top_edge;
  return av1_get_palette_cache(&xd, plane, out_cache);
}

/* av1_index_color_cache — real (exported) fn. Returns n_out; fills found[] + out_colors[]. */
int shim_index_color_cache(const uint16_t *cache, int n_cache, const uint16_t *colors,
                           int n_colors, uint8_t *found, int *out_colors) {
  return av1_index_color_cache(cache, n_cache, colors, n_colors, found, out_colors);
}

/* --- full write_palette_mode_info end-to-end (av1/encoder/bitstream.c) --- */
/* Inline helpers mirroring delta_encode_palette_colors + the V-plane coder, writing
 * into an existing od_ec (so the whole palette payload shares one bitstream). */
static void de_pal_inline(od_ec_enc *ec, const int *colors, int num, int bit_depth, int min_val) {
  if (num <= 0) return;
  mi_literal(ec, colors[0], bit_depth);
  if (num == 1) return;
  int max_delta = 0, deltas[PALETTE_MAX_SIZE];
  memset(deltas, 0, sizeof(deltas));
  for (int i = 1; i < num; ++i) {
    const int delta = colors[i] - colors[i - 1];
    deltas[i - 1] = delta;
    if (delta > max_delta) max_delta = delta;
  }
  const int min_bits = bit_depth - 3;
  int cl = aom_ceil_log2(max_delta + 1 - min_val);
  int bits = cl > min_bits ? cl : min_bits;
  int range = (1 << bit_depth) - colors[0] - min_val;
  mi_literal(ec, bits - min_bits, 2);
  for (int i = 0; i < num - 1; ++i) {
    mi_literal(ec, deltas[i] - min_val, bits);
    range -= deltas[i];
    int clr = aom_ceil_log2(range);
    bits = bits < clr ? bits : clr;
  }
}

static void wpc_v_inline(od_ec_enc *ec, const uint16_t *colors_v, int n, int bit_depth) {
  PALETTE_MODE_INFO pmi;
  pmi.palette_size[1] = (uint8_t)n;
  for (int i = 0; i < n; i++) pmi.palette_colors[2 * PALETTE_MAX_SIZE + i] = colors_v[i];
  const int max_val = 1 << bit_depth;
  int zero_count = 0, min_bits_v = 0;
  int bits_v = av1_get_palette_delta_bits_v(&pmi, bit_depth, &zero_count, &min_bits_v);
  const int rate_using_delta = 2 + bit_depth + (bits_v + 1) * (n - 1) - zero_count;
  const int rate_using_raw = bit_depth * n;
  if (rate_using_delta < rate_using_raw) {
    mi_bit(ec, 1);
    mi_literal(ec, bits_v - min_bits_v, 2);
    mi_literal(ec, colors_v[0], bit_depth);
    for (int i = 1; i < n; ++i) {
      if (colors_v[i] == colors_v[i - 1]) { mi_literal(ec, 0, bits_v); continue; }
      const int delta = abs((int)colors_v[i] - colors_v[i - 1]);
      const int sign_bit = colors_v[i] < colors_v[i - 1];
      if (delta <= max_val - delta) { mi_literal(ec, delta, bits_v); mi_bit(ec, sign_bit); }
      else { mi_literal(ec, max_val - delta, bits_v); mi_bit(ec, !sign_bit); }
    }
  } else {
    mi_bit(ec, 0);
    for (int i = 0; i < n; ++i) mi_literal(ec, colors_v[i], bit_depth);
  }
}

/* write_palette_colors_y/uv Y/U cache-bits + delta portion, over the real
 * av1_get_palette_cache + av1_index_color_cache. plane 0 (min_val 1) / plane 1 U
 * (min_val 0). Reuses the caller's MACROBLOCKD for the neighbour cache. */
static void wpc_plane_inline(od_ec_enc *ec, MACROBLOCKD *xd, const uint16_t *colors, int n,
                             int plane, int bit_depth, int min_val) {
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, plane, color_cache);
  int out_cache_colors[PALETTE_MAX_SIZE];
  uint8_t cache_color_found[2 * PALETTE_MAX_SIZE];
  const int n_out_cache =
      av1_index_color_cache(color_cache, n_cache, colors, n, cache_color_found, out_cache_colors);
  int n_in_cache = 0;
  for (int i = 0; i < n_cache && n_in_cache < n; ++i) {
    const int found = cache_color_found[i];
    mi_bit(ec, found);
    n_in_cache += found;
  }
  de_pal_inline(ec, out_cache_colors, n_out_cache, bit_depth, min_val);
}

uint32_t shim_write_palette_mode_info(int mode_dc, int uv_dc, int bit_depth, int bsize_ctx,
                                      int y_mode_ctx, int uv_mode_ctx,
                                      const uint8_t *palette_size, const uint16_t *palette_colors,
                                      int mb_to_top_edge, int ha, const uint16_t *a_colors,
                                      int a_s0, int a_s1, int hl, const uint16_t *l_colors,
                                      int l_s0, int l_s1, uint16_t *y_mode_cdf, uint16_t *y_size_cdf,
                                      uint16_t *uv_mode_cdf, uint16_t *uv_size_cdf, uint8_t *out,
                                      uint16_t *o_ym, uint16_t *o_ys, uint16_t *o_um, uint16_t *o_us) {
  (void)bsize_ctx; (void)y_mode_ctx; (void)uv_mode_ctx;
  MB_MODE_INFO ami, lmi, mbmi;
  MACROBLOCKD xd;
  for (int i = 0; i < 3 * PALETTE_MAX_SIZE; i++) {
    ami.palette_mode_info.palette_colors[i] = a_colors[i];
    lmi.palette_mode_info.palette_colors[i] = l_colors[i];
    mbmi.palette_mode_info.palette_colors[i] = palette_colors[i];
  }
  ami.palette_mode_info.palette_size[0] = (uint8_t)a_s0;
  ami.palette_mode_info.palette_size[1] = (uint8_t)a_s1;
  lmi.palette_mode_info.palette_size[0] = (uint8_t)l_s0;
  lmi.palette_mode_info.palette_size[1] = (uint8_t)l_s1;
  mbmi.palette_mode_info.palette_size[0] = palette_size[0];
  mbmi.palette_mode_info.palette_size[1] = palette_size[1];
  xd.above_mbmi = ha ? &ami : (MB_MODE_INFO *)0;
  xd.left_mbmi = hl ? &lmi : (MB_MODE_INFO *)0;
  xd.mb_to_top_edge = mb_to_top_edge;

  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  const PALETTE_MODE_INFO *const pmi = &mbmi.palette_mode_info;
  if (mode_dc) {
    const int n = pmi->palette_size[0];
    od_ec_encode_cdf_q15(&ec, n > 0, y_mode_cdf, 2);
    update_cdf(y_mode_cdf, n > 0, 2);
    if (n > 0) {
      od_ec_encode_cdf_q15(&ec, n - PALETTE_MIN_SIZE, y_size_cdf, PALETTE_SIZES);
      update_cdf(y_size_cdf, n - PALETTE_MIN_SIZE, PALETTE_SIZES);
      wpc_plane_inline(&ec, &xd, pmi->palette_colors, n, 0, bit_depth, 1);
    }
  }
  if (uv_dc) {
    const int n = pmi->palette_size[1];
    od_ec_encode_cdf_q15(&ec, n > 0, uv_mode_cdf, 2);
    update_cdf(uv_mode_cdf, n > 0, 2);
    if (n > 0) {
      od_ec_encode_cdf_q15(&ec, n - PALETTE_MIN_SIZE, uv_size_cdf, PALETTE_SIZES);
      update_cdf(uv_size_cdf, n - PALETTE_MIN_SIZE, PALETTE_SIZES);
      const uint16_t *colors_u = pmi->palette_colors + PALETTE_MAX_SIZE;
      const uint16_t *colors_v = pmi->palette_colors + 2 * PALETTE_MAX_SIZE;
      wpc_plane_inline(&ec, &xd, colors_u, n, 1, bit_depth, 0);
      wpc_v_inline(&ec, colors_v, n, bit_depth);
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) { o_ym[i] = y_mode_cdf[i]; o_um[i] = uv_mode_cdf[i]; }
  for (int i = 0; i < 8; i++) { o_ys[i] = y_size_cdf[i]; o_us[i] = uv_size_cdf[i]; }
  od_ec_enc_clear(&ec);
  return nb;
}
