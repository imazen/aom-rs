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
