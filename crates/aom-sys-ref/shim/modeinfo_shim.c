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
/* write_inter_mode body writing into an existing od_ec (reused by the inter mode+drl). */
static void inter_mode_into(od_ec_enc *ec, uint16_t *newmv_cdf, uint16_t *zeromv_cdf,
                            uint16_t *refmv_cdf, int mode, int mode_ctx) {
  int newmv_ctx = mode_ctx & 7;
  uint16_t *nc = newmv_cdf + newmv_ctx * 3;
  od_ec_encode_cdf_q15(ec, mode != NEWMV, nc, 2);
  update_cdf(nc, mode != NEWMV, 2);
  if (mode != NEWMV) {
    int zeromv_ctx = (mode_ctx >> 3) & 1;
    uint16_t *zc = zeromv_cdf + zeromv_ctx * 3;
    od_ec_encode_cdf_q15(ec, mode != GLOBALMV, zc, 2);
    update_cdf(zc, mode != GLOBALMV, 2);
    if (mode != GLOBALMV) {
      int refmv_ctx = (mode_ctx >> 4) & 15;
      uint16_t *rc = refmv_cdf + refmv_ctx * 3;
      od_ec_encode_cdf_q15(ec, mode != NEARESTMV, rc, 2);
      update_cdf(rc, mode != NEARESTMV, 2);
    }
  }
}
/* write_drl_idx body writing into an existing od_ec. */
static void drl_into(od_ec_enc *ec, uint16_t *drl_cdf, int mode, int ref_mv_idx, int ref_mv_count,
                     const uint16_t *weight) {
  int new_mv = (mode == NEWMV || mode == NEW_NEWMV);
  if (new_mv) {
    for (int idx = 0; idx < 2; ++idx) {
      if (ref_mv_count > idx + 1) {
        uint8_t ctx = av1_drl_ctx(weight, idx);
        uint16_t *c = drl_cdf + ctx * 3;
        od_ec_encode_cdf_q15(ec, ref_mv_idx != idx, c, 2);
        update_cdf(c, ref_mv_idx != idx, 2);
        if (ref_mv_idx == idx) return;
      }
    }
    return;
  }
  if (have_nearmv_in_inter_mode(mode)) {
    for (int idx = 1; idx < 3; ++idx) {
      if (ref_mv_count > idx + 1) {
        uint8_t ctx = av1_drl_ctx(weight, idx);
        uint16_t *c = drl_cdf + ctx * 3;
        od_ec_encode_cdf_q15(ec, ref_mv_idx != (idx - 1), c, 2);
        update_cdf(c, ref_mv_idx != (idx - 1), 2);
        if (ref_mv_idx == (idx - 1)) return;
      }
    }
  }
}

uint32_t shim_write_inter_mode(uint16_t *newmv_cdf, uint16_t *zeromv_cdf,
                               uint16_t *refmv_cdf, int mode, int mode_ctx, uint8_t *out,
                               uint16_t *out_newmv, uint16_t *out_zeromv,
                               uint16_t *out_refmv) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  inter_mode_into(&ec, newmv_cdf, zeromv_cdf, refmv_cdf, mode, mode_ctx);
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
  drl_into(&ec, drl_cdf, mode, ref_mv_idx, ref_mv_count, weight);
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
/* av1_encode_mv body writing into an existing od_ec (reused by the inter-mode MVs). */
static void encode_mv_into(od_ec_enc *ec, uint16_t *joints_cdf, uint16_t *comp0, uint16_t *comp1,
                           int diff_row, int diff_col, int usehp) {
  MV diff = { (int16_t)diff_row, (int16_t)diff_col };
  int j = av1_get_mv_joint(&diff);
  od_ec_encode_cdf_q15(ec, j, joints_cdf, 4);
  update_cdf(joints_cdf, j, 4);
  if (mv_joint_vertical(j)) mv_comp_wb(ec, comp0, diff.row, usehp);
  if (mv_joint_horizontal(j)) mv_comp_wb(ec, comp1, diff.col, usehp);
}

uint32_t shim_encode_mv(uint16_t *joints_cdf, uint16_t *comp0, uint16_t *comp1,
                        int diff_row, int diff_col, int usehp, uint8_t *out,
                        uint16_t *out_joints, uint16_t *out_comp0, uint16_t *out_comp1) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 256);
  encode_mv_into(&ec, joints_cdf, comp0, comp1, diff_row, diff_col, usehp);
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

/* write_motion_mode / write_mb_interp_filter bodies writing into an existing od_ec
 * (reused by the inter mode-body tail). */
static void motion_mode_into(od_ec_enc *ec, uint16_t *obmc_cdf, uint16_t *mm_cdf,
                             int last_allowed, int mm) {
  switch (last_allowed) {
    case 0: break; /* SIMPLE_TRANSLATION */
    case 1: /* OBMC_CAUSAL */
      od_ec_encode_cdf_q15(ec, mm == 1, obmc_cdf, 2);
      update_cdf(obmc_cdf, mm == 1, 2);
      break;
    default:
      od_ec_encode_cdf_q15(ec, mm, mm_cdf, MOTION_MODES);
      update_cdf(mm_cdf, mm, MOTION_MODES);
  }
}
static void interp_filter_into(od_ec_enc *ec, uint16_t *cdf0, uint16_t *cdf1, int interp_needed,
                               int is_switchable, int enable_dual, int f0, int f1) {
  if (interp_needed && is_switchable) {
    od_ec_encode_cdf_q15(ec, f0, cdf0, SWITCHABLE_FILTERS);
    update_cdf(cdf0, f0, SWITCHABLE_FILTERS);
    if (enable_dual) {
      od_ec_encode_cdf_q15(ec, f1, cdf1, SWITCHABLE_FILTERS);
      update_cdf(cdf1, f1, SWITCHABLE_FILTERS);
    }
  }
}

uint32_t shim_write_motion_mode(uint16_t *obmc_cdf, uint16_t *mm_cdf, int last_allowed,
                                int mm, uint8_t *out, uint16_t *out_obmc, uint16_t *out_mm) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  motion_mode_into(&ec, obmc_cdf, mm_cdf, last_allowed, mm);
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
  interp_filter_into(&ec, cdf0, cdf1, interp_needed, is_switchable, enable_dual, f0, f1);
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

/* Facade for av1_get_pred_context_switchable_interp: reads the current block's
 * ref_frame[0/1] (ctx offset + which frame the neighbours must match) and the
 * above/left neighbours' ref_frame[0/1] + interp_filters (y/x) through the mi
 * grid (xd->mi[-1] / xd->mi[-xd->mi_stride]). Laid out on a small pointer grid so
 * mi[0]=cur, mi[-1]=left, mi[-stride]=above; unavailable neighbours are gated by
 * xd->{up,left}_available (their cells are never dereferenced then). */
int shim_get_pred_context_switchable_interp(int dir, int cur_r0, int cur_r1,
                                            int has_above, int a_r0, int a_r1,
                                            int a_yf, int a_xf, int has_left,
                                            int l_r0, int l_r1, int l_yf,
                                            int l_xf) {
  MB_MODE_INFO cur, ami, lmi;
  memset(&cur, 0, sizeof cur);
  memset(&ami, 0, sizeof ami);
  memset(&lmi, 0, sizeof lmi);
  cur.ref_frame[0] = (MV_REFERENCE_FRAME)cur_r0;
  cur.ref_frame[1] = (MV_REFERENCE_FRAME)cur_r1;
  ami.ref_frame[0] = (MV_REFERENCE_FRAME)a_r0;
  ami.ref_frame[1] = (MV_REFERENCE_FRAME)a_r1;
  ami.interp_filters.as_filters.y_filter = (uint16_t)a_yf;
  ami.interp_filters.as_filters.x_filter = (uint16_t)a_xf;
  lmi.ref_frame[0] = (MV_REFERENCE_FRAME)l_r0;
  lmi.ref_frame[1] = (MV_REFERENCE_FRAME)l_r1;
  lmi.interp_filters.as_filters.y_filter = (uint16_t)l_yf;
  lmi.interp_filters.as_filters.x_filter = (uint16_t)l_xf;
  const int stride = 4;
  MB_MODE_INFO *grid[8];
  memset(grid, 0, sizeof grid);
  grid[stride + 1] = &cur; /* mi[0]     */
  grid[stride + 0] = &lmi; /* mi[-1]    */
  grid[1] = &ami;          /* mi[-stride] */
  MACROBLOCKD xd;
  memset(&xd, 0, sizeof xd);
  xd.mi = &grid[stride + 1];
  xd.mi_stride = stride;
  xd.up_available = has_above;
  xd.left_available = has_left;
  return av1_get_pred_context_switchable_interp(&xd, dir);
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
/* write_intrabc_info body writing into an existing od_ec (reused by write_mb_modes_kf). */
static void intrabc_into(od_ec_enc *ec, uint16_t *intrabc_cdf, uint16_t *joints,
                        uint16_t *comp0, uint16_t *comp1, int use_intrabc, int diff_row,
                        int diff_col) {
  od_ec_encode_cdf_q15(ec, use_intrabc, intrabc_cdf, 2);
  update_cdf(intrabc_cdf, use_intrabc, 2);
  if (use_intrabc) {
    MV diff = { (int16_t)diff_row, (int16_t)diff_col };
    int j = av1_get_mv_joint(&diff);
    od_ec_encode_cdf_q15(ec, j, joints, 4);
    update_cdf(joints, j, 4);
    if (mv_joint_vertical(j)) mv_comp_wb(ec, comp0, diff.row, MV_SUBPEL_NONE);
    if (mv_joint_horizontal(j)) mv_comp_wb(ec, comp1, diff.col, MV_SUBPEL_NONE);
  }
}

uint32_t shim_write_intrabc_info(uint16_t *intrabc_cdf, uint16_t *joints, uint16_t *comp0,
                                 uint16_t *comp1, int use_intrabc, int diff_row,
                                 int diff_col, uint8_t *out, uint16_t *out_ibc,
                                 uint16_t *out_joints, uint16_t *out_c0, uint16_t *out_c1) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  intrabc_into(&ec, intrabc_cdf, joints, comp0, comp1, use_intrabc, diff_row, diff_col);
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

/* --- interintra sub-symbols (av1/encoder/bitstream.c, in write_mbmi_b) --- */
#include "av1/common/reconinter.h" /* MAX_WEDGE_TYPES */
/* Transcribed verbatim over od_ec. INTERINTRA_MODES=4, MAX_WEDGE_TYPES=16. The outer
 * gate (reference_mode / enable_interintra_compound / is_interintra_allowed) and
 * av1_is_wedge_used(bsize) are the caller's; CDFs are pre-selected by bsize_group/bsize. */
/* interintra coding body (the outer allow-gate is the caller's); reused by the tail. */
static void interintra_into(od_ec_enc *ec, int interintra, uint16_t *ii_cdf, int ii_mode,
                            uint16_t *ii_mode_cdf, int wedge_used, int use_wedge,
                            uint16_t *wedge_ii_cdf, int wedge_index, uint16_t *wedge_idx_cdf) {
  od_ec_encode_cdf_q15(ec, interintra, ii_cdf, 2);
  update_cdf(ii_cdf, interintra, 2);
  if (interintra) {
    od_ec_encode_cdf_q15(ec, ii_mode, ii_mode_cdf, INTERINTRA_MODES);
    update_cdf(ii_mode_cdf, ii_mode, INTERINTRA_MODES);
    if (wedge_used) {
      od_ec_encode_cdf_q15(ec, use_wedge, wedge_ii_cdf, 2);
      update_cdf(wedge_ii_cdf, use_wedge, 2);
      if (use_wedge) {
        od_ec_encode_cdf_q15(ec, wedge_index, wedge_idx_cdf, MAX_WEDGE_TYPES);
        update_cdf(wedge_idx_cdf, wedge_index, MAX_WEDGE_TYPES);
      }
    }
  }
}

uint32_t shim_write_interintra_info(int interintra, uint16_t *ii_cdf, int ii_mode,
                                    uint16_t *ii_mode_cdf, int wedge_used, int use_wedge,
                                    uint16_t *wedge_ii_cdf, int wedge_index,
                                    uint16_t *wedge_idx_cdf, uint8_t *out, uint16_t *o_ii,
                                    uint16_t *o_iim, uint16_t *o_wii, uint16_t *o_wix) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  interintra_into(&ec, interintra, ii_cdf, ii_mode, ii_mode_cdf, wedge_used, use_wedge,
                  wedge_ii_cdf, wedge_index, wedge_idx_cdf);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) { o_ii[i] = ii_cdf[i]; o_wii[i] = wedge_ii_cdf[i]; }
  for (int i = 0; i < 5; i++) o_iim[i] = ii_mode_cdf[i];   /* INTERINTRA_MODES+1 */
  for (int i = 0; i < 17; i++) o_wix[i] = wedge_idx_cdf[i]; /* MAX_WEDGE_TYPES+1 */
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- compound-type group (av1/encoder/bitstream.c, in write_mbmi_b) --- */
/* get_comp_group_idx_context facade: above/left neighbours' comp_group_idx (or 3 when
 * a single-ref neighbour points at ALTREF), summed and capped at 5. */
int shim_get_comp_group_idx_context(int ha, int a_rf0, int a_rf1, int a_cgi, int hl,
                                    int l_rf0, int l_rf1, int l_cgi) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.ref_frame[0] = (MV_REFERENCE_FRAME)a_rf0;
  ami.ref_frame[1] = (MV_REFERENCE_FRAME)a_rf1;
  ami.comp_group_idx = (uint8_t)a_cgi;
  lmi.ref_frame[0] = (MV_REFERENCE_FRAME)l_rf0;
  lmi.ref_frame[1] = (MV_REFERENCE_FRAME)l_rf1;
  lmi.comp_group_idx = (uint8_t)l_cgi;
  xd.above_mbmi = ha ? &ami : (MB_MODE_INFO *)0;
  xd.left_mbmi = hl ? &lmi : (MB_MODE_INFO *)0;
  return get_comp_group_idx_context(&xd);
}

/* The compound-type coding portion of write_mbmi_b, transcribed verbatim over od_ec.
 * The has_second_ref outer gate + is_interinter_compound_used(WEDGE) + the two CDF
 * contexts are the caller's; CDFs are pre-selected. MASKED_COMPOUND_TYPES=2,
 * MAX_WEDGE_TYPES=16, MAX_DIFFWTD_MASK_BITS=1. comp_type is COMPOUND_WEDGE(2)/DIFFWTD(3). */
/* compound-type coding body (the has_second_ref gate is the caller's); reused by the tail. */
static void compound_type_into(od_ec_enc *ec, int masked_used, int comp_group_idx,
    uint16_t *cgi_cdf, int dist_wtd, int compound_idx, uint16_t *cidx_cdf, int wedge_used,
    int comp_type, uint16_t *ctype_cdf, int wedge_index, uint16_t *wedge_idx_cdf, int wedge_sign,
    int mask_type) {
  if (masked_used) {
    od_ec_encode_cdf_q15(ec, comp_group_idx, cgi_cdf, 2);
    update_cdf(cgi_cdf, comp_group_idx, 2);
  }
  if (comp_group_idx == 0) {
    if (dist_wtd) {
      od_ec_encode_cdf_q15(ec, compound_idx, cidx_cdf, 2);
      update_cdf(cidx_cdf, compound_idx, 2);
    }
  } else {
    if (wedge_used) {
      od_ec_encode_cdf_q15(ec, comp_type - COMPOUND_WEDGE, ctype_cdf, MASKED_COMPOUND_TYPES);
      update_cdf(ctype_cdf, comp_type - COMPOUND_WEDGE, MASKED_COMPOUND_TYPES);
    }
    if (comp_type == COMPOUND_WEDGE) {
      od_ec_encode_cdf_q15(ec, wedge_index, wedge_idx_cdf, MAX_WEDGE_TYPES);
      update_cdf(wedge_idx_cdf, wedge_index, MAX_WEDGE_TYPES);
      mi_bit(ec, wedge_sign);
    } else {
      mi_literal(ec, mask_type, MAX_DIFFWTD_MASK_BITS);
    }
  }
}

uint32_t shim_write_compound_type_info(int masked_used, int comp_group_idx, uint16_t *cgi_cdf,
                                       int dist_wtd, int compound_idx, uint16_t *cidx_cdf,
                                       int wedge_used, int comp_type, uint16_t *ctype_cdf,
                                       int wedge_index, uint16_t *wedge_idx_cdf, int wedge_sign,
                                       int mask_type, uint8_t *out, uint16_t *o_cgi,
                                       uint16_t *o_cidx, uint16_t *o_ctype, uint16_t *o_wix) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  compound_type_into(&ec, masked_used, comp_group_idx, cgi_cdf, dist_wtd, compound_idx, cidx_cdf,
                     wedge_used, comp_type, ctype_cdf, wedge_index, wedge_idx_cdf, wedge_sign, mask_type);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) { o_cgi[i] = cgi_cdf[i]; o_cidx[i] = cidx_cdf[i]; o_ctype[i] = ctype_cdf[i]; }
  for (int i = 0; i < 17; i++) o_wix[i] = wedge_idx_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- order-hint distance + compound_idx context --- */
#include "av1/common/mvref_common.h" /* get_relative_dist */

int shim_get_relative_dist(int enable, int bits_minus_1, int a, int b) {
  OrderHintInfo oh;
  oh.enable_order_hint = enable;
  oh.order_hint_bits_minus_1 = bits_minus_1;
  return get_relative_dist(&oh, a, b);
}

/* get_comp_index_context body transcribed verbatim, but the two ref-buffer order hints
 * (normally get_ref_frame_buf(cm, ref)->order_hint, or 0 when the buffer is absent) are
 * passed directly — the CONTEXT logic (real get_relative_dist + has_second_ref + the ctx
 * arithmetic) is otherwise identical to libaom. */
int shim_get_comp_index_context(int enable, int bits_minus_1, int cur_order_hint,
                                int fwd_order_hint, int bck_order_hint,
                                int ha, int a_has2, int a_cidx, int a_rf0,
                                int hl, int l_has2, int l_cidx, int l_rf0) {
  OrderHintInfo oh;
  oh.enable_order_hint = enable;
  oh.order_hint_bits_minus_1 = bits_minus_1;
  int fwd = abs(get_relative_dist(&oh, fwd_order_hint, cur_order_hint));
  int bck = abs(get_relative_dist(&oh, cur_order_hint, bck_order_hint));
  int above_ctx = 0, left_ctx = 0;
  const int offset = (fwd == bck);
  if (ha) {
    if (a_has2) above_ctx = a_cidx;
    else if (a_rf0 == ALTREF_FRAME) above_ctx = 1;
  }
  if (hl) {
    if (l_has2) left_ctx = l_cidx;
    else if (l_rf0 == ALTREF_FRAME) left_ctx = 1;
  }
  return above_ctx + left_ctx + 3 * offset;
}

/* --- intra-prediction-mode driver gates (av1/common/*.h) --- */
#include "av1/common/reconintra.h" /* av1_use_angle_delta / av1_is_directional_mode */
#include "av1/common/cfl.h"        /* is_cfl_allowed */

int shim_use_angle_delta(int bsize) { return av1_use_angle_delta((BLOCK_SIZE)bsize); }
int shim_is_directional_mode(int mode) { return av1_is_directional_mode((PREDICTION_MODE)mode); }
int shim_get_uv_mode(int uv_mode) { return get_uv_mode((UV_PREDICTION_MODE)uv_mode); }
int shim_allow_palette(int allow_sct, int bsize) {
  return av1_allow_palette(allow_sct, (BLOCK_SIZE)bsize);
}
int shim_is_cfl_allowed(int bsize, int seg_id, int lossless, int ssx, int ssy) {
  MB_MODE_INFO mbmi;
  MB_MODE_INFO *miptr = &mbmi;
  MACROBLOCKD xd;
  mbmi.bsize = (BLOCK_SIZE)bsize;
  mbmi.segment_id = (int8_t)seg_id;
  xd.mi = &miptr;
  xd.lossless[seg_id] = lossless;
  xd.plane[AOM_PLANE_U].subsampling_x = ssx;
  xd.plane[AOM_PLANE_U].subsampling_y = ssy;
  return (int)is_cfl_allowed(&xd);
}

/* --- write_intra_prediction_modes composition, piece 1: Y mode + Y angle delta --- */
/* The first two steps of write_intra_prediction_modes (av1/encoder/bitstream.c),
 * transcribed over od_ec. is_keyframe only picks the Y CDF upstream (kf_y_cdf[a][l] vs
 * y_mode_cdf[sg]) so it is pre-selected here. The angle delta is gated on the REAL
 * av1_use_angle_delta + av1_is_directional_mode. INTRA_MODES=13, MAX_ANGLE_DELTA=3. */
uint32_t shim_write_intra_y_and_angle(int mode, int bsize, uint16_t *y_cdf,
                                      int angle_delta_y, uint16_t *y_angle_cdf,
                                      uint8_t *out, uint16_t *o_ycdf, uint16_t *o_acdf) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  od_ec_encode_cdf_q15(&ec, mode, y_cdf, INTRA_MODES);
  update_cdf(y_cdf, mode, INTRA_MODES);
  if (av1_use_angle_delta((BLOCK_SIZE)bsize) && av1_is_directional_mode((PREDICTION_MODE)mode)) {
    const int sym = angle_delta_y + MAX_ANGLE_DELTA;
    const int n = 2 * MAX_ANGLE_DELTA + 1;
    od_ec_encode_cdf_q15(&ec, sym, y_angle_cdf, n);
    update_cdf(y_angle_cdf, sym, n);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 14; i++) o_ycdf[i] = y_cdf[i];
  for (int i = 0; i < 8; i++) o_acdf[i] = y_angle_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_intra_prediction_modes composition, piece 2: UV mode + cfl + UV angle --- */
/* The chroma steps of write_intra_prediction_modes, transcribed over od_ec. Gated on
 * !monochrome && is_chroma_ref; cfl_allowed (is_cfl_allowed upstream) sets the UV-mode
 * symbol count; the cfl alphas follow UV_CFL_PRED(13); the UV angle delta is gated on
 * the REAL av1_use_angle_delta + av1_is_directional_mode(get_uv_mode(uv_mode)). */
uint32_t shim_write_intra_uv_and_angle(int monochrome, int is_chroma_ref, int uv_mode,
    int cfl_allowed, int bsize, int cfl_idx, int cfl_joint_sign, int angle_delta_uv,
    uint16_t *uv_mode_cdf, uint16_t *cfl_sign_cdf, uint16_t *cfl_alpha_cdf, uint16_t *uv_angle_cdf,
    uint8_t *out, uint16_t *o_uvcdf, uint16_t *o_signcdf, uint16_t *o_alphacdf, uint16_t *o_uvacdf) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (!monochrome && is_chroma_ref) {
    const int n = UV_INTRA_MODES - !cfl_allowed;
    od_ec_encode_cdf_q15(&ec, uv_mode, uv_mode_cdf, n);
    update_cdf(uv_mode_cdf, uv_mode, n);
    if (uv_mode == UV_CFL_PRED) {
      od_ec_encode_cdf_q15(&ec, cfl_joint_sign, cfl_sign_cdf, CFL_JOINT_SIGNS);
      update_cdf(cfl_sign_cdf, cfl_joint_sign, CFL_JOINT_SIGNS);
      if (CFL_SIGN_U(cfl_joint_sign) != CFL_SIGN_ZERO) {
        uint16_t *cdf_u = cfl_alpha_cdf + CFL_CONTEXT_U(cfl_joint_sign) * 17;
        od_ec_encode_cdf_q15(&ec, CFL_IDX_U(cfl_idx), cdf_u, CFL_ALPHABET_SIZE);
        update_cdf(cdf_u, CFL_IDX_U(cfl_idx), CFL_ALPHABET_SIZE);
      }
      if (CFL_SIGN_V(cfl_joint_sign) != CFL_SIGN_ZERO) {
        uint16_t *cdf_v = cfl_alpha_cdf + CFL_CONTEXT_V(cfl_joint_sign) * 17;
        od_ec_encode_cdf_q15(&ec, CFL_IDX_V(cfl_idx), cdf_v, CFL_ALPHABET_SIZE);
        update_cdf(cdf_v, CFL_IDX_V(cfl_idx), CFL_ALPHABET_SIZE);
      }
    }
    const int intra_mode = get_uv_mode((UV_PREDICTION_MODE)uv_mode);
    if (av1_use_angle_delta((BLOCK_SIZE)bsize) && av1_is_directional_mode((PREDICTION_MODE)intra_mode)) {
      const int sym = angle_delta_uv + MAX_ANGLE_DELTA;
      const int an = 2 * MAX_ANGLE_DELTA + 1;
      od_ec_encode_cdf_q15(&ec, sym, uv_angle_cdf, an);
      update_cdf(uv_angle_cdf, sym, an);
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 15; i++) o_uvcdf[i] = uv_mode_cdf[i];
  for (int i = 0; i < 9; i++) o_signcdf[i] = cfl_sign_cdf[i];
  for (int i = 0; i < 6 * 17; i++) o_alphacdf[i] = cfl_alpha_cdf[i];
  for (int i = 0; i < 8; i++) o_uvacdf[i] = uv_angle_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- full write_intra_prediction_modes over ONE coder (av1/encoder/bitstream.c) --- */
/* Sequences Y mode + Y angle, UV mode + cfl + UV angle, palette, filter-intra over a
 * single od_ec — the whole intra-mode-info per-block fragment. Reuses the palette inline
 * helpers (wpc_plane_inline/wpc_v_inline). allow_palette / filter_allowed are the
 * caller's resolved gates; mode_dc/uv_dc are derived per write_palette_mode_info. */
/* write_intra_prediction_modes body writing into an existing od_ec + xd (reused by
 * write_mb_modes_kf). CDFs pre-selected; xd carries the palette neighbour cache. */
static void wipm_into(od_ec_enc *ec, MACROBLOCKD *xd,
    int mode, int bsize, uint16_t *y_cdf, int angle_delta_y, uint16_t *y_angle_cdf,
    int monochrome, int is_chroma_ref, int uv_mode, int cfl_allowed, int cfl_idx,
    int cfl_joint_sign, int angle_delta_uv, uint16_t *uv_mode_cdf, uint16_t *cfl_sign_cdf,
    uint16_t *cfl_alpha_cdf, uint16_t *uv_angle_cdf, int allow_palette, int bit_depth,
    const uint8_t *palette_size, const uint16_t *palette_colors, uint16_t *pal_y_mode_cdf,
    uint16_t *pal_y_size_cdf, uint16_t *pal_uv_mode_cdf, uint16_t *pal_uv_size_cdf,
    int filter_allowed, int use_filter_intra, int filter_intra_mode, uint16_t *fi_use_cdf,
    uint16_t *fi_mode_cdf) {
  /* Y mode + Y angle. */
  od_ec_encode_cdf_q15(ec, mode, y_cdf, INTRA_MODES);
  update_cdf(y_cdf, mode, INTRA_MODES);
  if (av1_use_angle_delta((BLOCK_SIZE)bsize) && av1_is_directional_mode((PREDICTION_MODE)mode)) {
    const int s = angle_delta_y + MAX_ANGLE_DELTA, n = 2 * MAX_ANGLE_DELTA + 1;
    od_ec_encode_cdf_q15(ec, s, y_angle_cdf, n); update_cdf(y_angle_cdf, s, n);
  }
  /* UV mode + cfl + UV angle. */
  if (!monochrome && is_chroma_ref) {
    const int n = UV_INTRA_MODES - !cfl_allowed;
    od_ec_encode_cdf_q15(ec, uv_mode, uv_mode_cdf, n); update_cdf(uv_mode_cdf, uv_mode, n);
    if (uv_mode == UV_CFL_PRED) {
      od_ec_encode_cdf_q15(ec, cfl_joint_sign, cfl_sign_cdf, CFL_JOINT_SIGNS);
      update_cdf(cfl_sign_cdf, cfl_joint_sign, CFL_JOINT_SIGNS);
      if (CFL_SIGN_U(cfl_joint_sign) != CFL_SIGN_ZERO) {
        uint16_t *c = cfl_alpha_cdf + CFL_CONTEXT_U(cfl_joint_sign) * 17;
        od_ec_encode_cdf_q15(ec, CFL_IDX_U(cfl_idx), c, CFL_ALPHABET_SIZE);
        update_cdf(c, CFL_IDX_U(cfl_idx), CFL_ALPHABET_SIZE);
      }
      if (CFL_SIGN_V(cfl_joint_sign) != CFL_SIGN_ZERO) {
        uint16_t *c = cfl_alpha_cdf + CFL_CONTEXT_V(cfl_joint_sign) * 17;
        od_ec_encode_cdf_q15(ec, CFL_IDX_V(cfl_idx), c, CFL_ALPHABET_SIZE);
        update_cdf(c, CFL_IDX_V(cfl_idx), CFL_ALPHABET_SIZE);
      }
    }
    const int im = get_uv_mode((UV_PREDICTION_MODE)uv_mode);
    if (av1_use_angle_delta((BLOCK_SIZE)bsize) && av1_is_directional_mode((PREDICTION_MODE)im)) {
      const int s = angle_delta_uv + MAX_ANGLE_DELTA, n2 = 2 * MAX_ANGLE_DELTA + 1;
      od_ec_encode_cdf_q15(ec, s, uv_angle_cdf, n2); update_cdf(uv_angle_cdf, s, n2);
    }
  }
  /* Palette. */
  if (allow_palette) {
    const int mode_dc = (mode == DC_PRED);
    const int uv_dc = (!monochrome && uv_mode == UV_DC_PRED && is_chroma_ref);
    if (mode_dc) {
      const int n = palette_size[0];
      od_ec_encode_cdf_q15(ec, n > 0, pal_y_mode_cdf, 2); update_cdf(pal_y_mode_cdf, n > 0, 2);
      if (n > 0) {
        od_ec_encode_cdf_q15(ec, n - PALETTE_MIN_SIZE, pal_y_size_cdf, PALETTE_SIZES);
        update_cdf(pal_y_size_cdf, n - PALETTE_MIN_SIZE, PALETTE_SIZES);
        wpc_plane_inline(ec, xd, palette_colors, n, 0, bit_depth, 1);
      }
    }
    if (uv_dc) {
      const int n = palette_size[1];
      od_ec_encode_cdf_q15(ec, n > 0, pal_uv_mode_cdf, 2); update_cdf(pal_uv_mode_cdf, n > 0, 2);
      if (n > 0) {
        od_ec_encode_cdf_q15(ec, n - PALETTE_MIN_SIZE, pal_uv_size_cdf, PALETTE_SIZES);
        update_cdf(pal_uv_size_cdf, n - PALETTE_MIN_SIZE, PALETTE_SIZES);
        wpc_plane_inline(ec, xd, palette_colors + PALETTE_MAX_SIZE, n, 1, bit_depth, 0);
        wpc_v_inline(ec, palette_colors + 2 * PALETTE_MAX_SIZE, n, bit_depth);
      }
    }
  }
  /* Filter intra. */
  if (filter_allowed) {
    od_ec_encode_cdf_q15(ec, use_filter_intra, fi_use_cdf, 2);
    update_cdf(fi_use_cdf, use_filter_intra, 2);
    if (use_filter_intra) {
      od_ec_encode_cdf_q15(ec, filter_intra_mode, fi_mode_cdf, FILTER_INTRA_MODES);
      update_cdf(fi_mode_cdf, filter_intra_mode, FILTER_INTRA_MODES);
    }
  }
}

/* Sets up a MACROBLOCKD with the palette neighbour state for wipm_into. */
static void wipm_setup_xd(MACROBLOCKD *xd, MB_MODE_INFO *ami, MB_MODE_INFO *lmi,
    int mb_to_top_edge, int ha, const uint16_t *a_colors, int a_s0, int a_s1, int hl,
    const uint16_t *l_colors, int l_s0, int l_s1) {
  for (int i = 0; i < 3 * PALETTE_MAX_SIZE; i++) {
    ami->palette_mode_info.palette_colors[i] = a_colors[i];
    lmi->palette_mode_info.palette_colors[i] = l_colors[i];
  }
  ami->palette_mode_info.palette_size[0] = (uint8_t)a_s0;
  ami->palette_mode_info.palette_size[1] = (uint8_t)a_s1;
  lmi->palette_mode_info.palette_size[0] = (uint8_t)l_s0;
  lmi->palette_mode_info.palette_size[1] = (uint8_t)l_s1;
  xd->above_mbmi = ha ? ami : (MB_MODE_INFO *)0;
  xd->left_mbmi = hl ? lmi : (MB_MODE_INFO *)0;
  xd->mb_to_top_edge = mb_to_top_edge;
}

uint32_t shim_write_intra_pred_modes(
    int mode, int bsize, uint16_t *y_cdf, int angle_delta_y, uint16_t *y_angle_cdf,
    int monochrome, int is_chroma_ref, int uv_mode, int cfl_allowed, int cfl_idx,
    int cfl_joint_sign, int angle_delta_uv, uint16_t *uv_mode_cdf, uint16_t *cfl_sign_cdf,
    uint16_t *cfl_alpha_cdf, uint16_t *uv_angle_cdf,
    int allow_palette, int bit_depth, const uint8_t *palette_size, const uint16_t *palette_colors,
    int mb_to_top_edge, int ha, const uint16_t *a_colors, int a_s0, int a_s1, int hl,
    const uint16_t *l_colors, int l_s0, int l_s1, uint16_t *pal_y_mode_cdf, uint16_t *pal_y_size_cdf,
    uint16_t *pal_uv_mode_cdf, uint16_t *pal_uv_size_cdf,
    int filter_allowed, int use_filter_intra, int filter_intra_mode, uint16_t *fi_use_cdf,
    uint16_t *fi_mode_cdf, uint8_t *out, uint16_t *o_all) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  wipm_setup_xd(&xd, &ami, &lmi, mb_to_top_edge, ha, a_colors, a_s0, a_s1, hl, l_colors, l_s0, l_s1);

  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  wipm_into(&ec, &xd, mode, bsize, y_cdf, angle_delta_y, y_angle_cdf, monochrome, is_chroma_ref,
            uv_mode, cfl_allowed, cfl_idx, cfl_joint_sign, angle_delta_uv, uv_mode_cdf, cfl_sign_cdf,
            cfl_alpha_cdf, uv_angle_cdf, allow_palette, bit_depth, palette_size, palette_colors,
            pal_y_mode_cdf, pal_y_size_cdf, pal_uv_mode_cdf, pal_uv_size_cdf, filter_allowed,
            use_filter_intra, filter_intra_mode, fi_use_cdf, fi_mode_cdf);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  /* Pack every adapted CDF into o_all in a fixed layout (see the Rust side). */
  int k = 0;
  for (int i = 0; i < 14; i++) o_all[k++] = y_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = y_angle_cdf[i];
  for (int i = 0; i < 15; i++) o_all[k++] = uv_mode_cdf[i];
  for (int i = 0; i < 9; i++) o_all[k++] = cfl_sign_cdf[i];
  for (int i = 0; i < 6 * 17; i++) o_all[k++] = cfl_alpha_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = uv_angle_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = pal_y_mode_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = pal_y_size_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = pal_uv_mode_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = pal_uv_size_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = fi_use_cdf[i];
  for (int i = 0; i < 6; i++) o_all[k++] = fi_mode_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_delta_q_params driver (av1/encoder/bitstream.c) --- */
/* Extracted delta_qindex / delta_lflevel bodies writing into an existing od_ec. */
static void dq_body(od_ec_enc *ec, uint16_t *cdf, int delta) {
  int sign = delta < 0, a = sign ? -delta : delta;
  int sym = a < DELTA_Q_SMALL ? a : DELTA_Q_SMALL;
  od_ec_encode_cdf_q15(ec, sym, cdf, DELTA_Q_PROBS + 1);
  update_cdf(cdf, sym, DELTA_Q_PROBS + 1);
  if (a >= DELTA_Q_SMALL) {
    int rem_bits = get_msb(a - 1), thr = (1 << rem_bits) + 1;
    mi_literal(ec, rem_bits - 1, 3);
    mi_literal(ec, a - thr, rem_bits);
  }
  if (a > 0) mi_bit(ec, sign);
}
static void dlf_body(od_ec_enc *ec, uint16_t *cdf, int delta) {
  int sign = delta < 0, a = sign ? -delta : delta;
  int sym = a < DELTA_LF_SMALL ? a : DELTA_LF_SMALL;
  od_ec_encode_cdf_q15(ec, sym, cdf, DELTA_LF_PROBS + 1);
  update_cdf(cdf, sym, DELTA_LF_PROBS + 1);
  if (a >= DELTA_LF_SMALL) {
    int rem_bits = get_msb(a - 1), thr = (1 << rem_bits) + 1;
    mi_literal(ec, rem_bits - 1, 3);
    mi_literal(ec, a - thr, rem_bits);
  }
  if (a > 0) mi_bit(ec, sign);
}

/* write_delta_q_params transcribed verbatim. Reduced deltas + gating + xd state update
 * (current_base_qindex / delta_lf[] / delta_lf_from_base) all mirrored; state returned. */
uint32_t shim_write_delta_q_params_sb(int dq_present, int dlf_present, int dlf_multi, int num_planes,
    int bsize, int sb_size, int skip, int sbul, int cur_qindex, int cur_base_qindex, int dq_res,
    const int *mbmi_dlf, const int *xd_dlf_in, int mbmi_dlf_base, int xd_dlf_base_in, int dlf_res,
    uint16_t *dq_cdf, uint16_t *dlf_multi_cdf, uint16_t *dlf_cdf, uint8_t *out, uint16_t *o_dqcdf,
    uint16_t *o_dlfmcdf, uint16_t *o_dlfcdf, int *o_base, int *o_xd_dlf, int *o_xd_dlf_base) {
  int xd_dlf[FRAME_LF_COUNT];
  for (int i = 0; i < FRAME_LF_COUNT; i++) xd_dlf[i] = xd_dlf_in[i];
  int base = cur_base_qindex, xd_dlf_base = xd_dlf_base_in;
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (dq_present) {
    if ((bsize != sb_size || skip == 0) && sbul) {
      int reduced_dq = (cur_qindex - base) / dq_res;
      dq_body(&ec, dq_cdf, reduced_dq);
      base = cur_qindex;
      if (dlf_present) {
        if (dlf_multi) {
          int flc = num_planes > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
          for (int lf = 0; lf < flc; ++lf) {
            int r = (mbmi_dlf[lf] - xd_dlf[lf]) / dlf_res;
            dlf_body(&ec, dlf_multi_cdf + lf * (DELTA_LF_PROBS + 2), r);
            xd_dlf[lf] = mbmi_dlf[lf];
          }
        } else {
          int r = (mbmi_dlf_base - xd_dlf_base) / dlf_res;
          dlf_body(&ec, dlf_cdf, r);
          xd_dlf_base = mbmi_dlf_base;
        }
      }
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < DELTA_Q_PROBS + 2; i++) o_dqcdf[i] = dq_cdf[i];
  for (int i = 0; i < FRAME_LF_COUNT * (DELTA_LF_PROBS + 2); i++) o_dlfmcdf[i] = dlf_multi_cdf[i];
  for (int i = 0; i < DELTA_LF_PROBS + 2; i++) o_dlfcdf[i] = dlf_cdf[i];
  *o_base = base; *o_xd_dlf_base = xd_dlf_base;
  for (int i = 0; i < FRAME_LF_COUNT; i++) o_xd_dlf[i] = xd_dlf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_cdef (av1/encoder/bitstream.c) --- */
/* Transcribed verbatim. The per-CDEF-unit first-block grid lookup just fetches
 * cdef_strength, so it is passed directly. cdef_transmitted[4] is the per-SB state
 * (reset at the SB upper-left block); it is threaded in/out. MI_SIZE_LOG2=2 ->
 * cdef_size=16; BLOCK_128X128=15. */
uint32_t shim_write_cdef(int coded_lossless, int allow_intrabc, int mi_row, int mi_col,
                         int mib_size, int sb_size, int skip, const int *transmitted_in,
                         int cdef_bits, int cdef_strength, uint8_t *out, int *transmitted_out) {
  int transmitted[4];
  for (int i = 0; i < 4; i++) transmitted[i] = transmitted_in[i];
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (!(coded_lossless || allow_intrabc)) {
    const int sb_mask = mib_size - 1;
    const int mi_row_in_sb = mi_row & sb_mask;
    const int mi_col_in_sb = mi_col & sb_mask;
    if (mi_row_in_sb == 0 && mi_col_in_sb == 0)
      transmitted[0] = transmitted[1] = transmitted[2] = transmitted[3] = 0;
    const int cdef_size = 1 << (6 - MI_SIZE_LOG2);
    const int index_mask = cdef_size;
    const int cdef_unit_row_in_sb = ((mi_row & index_mask) != 0);
    const int cdef_unit_col_in_sb = ((mi_col & index_mask) != 0);
    const int index =
        (sb_size == BLOCK_128X128) ? cdef_unit_col_in_sb + 2 * cdef_unit_row_in_sb : 0;
    if (!transmitted[index] && !skip) {
      mi_literal(&ec, cdef_strength, cdef_bits);
      transmitted[index] = 1;
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 4; i++) transmitted_out[i] = transmitted[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_mb_modes_kf prefix (av1/encoder/bitstream.c:1267) --- */
/* Sequences segment_id(preskip) -> skip -> segment_id(postskip) -> cdef -> delta_q_params
 * over ONE od_ec, threading write_skip's return into cdef/delta_q. The intrabc + intra
 * tail is validated separately. Reuses dq_body/dlf_body; seg/skip/cdef inlined. */
static int seg_body(od_ec_enc *ec, uint16_t *cdf, int seg_enabled, int update_map,
                    int skip_txfm, int segment_id, int pred, int last_active_segid) {
  if (seg_enabled && update_map && !skip_txfm) {
    int coded_id = av1_neg_interleave(segment_id, pred, last_active_segid + 1);
    od_ec_encode_cdf_q15(ec, coded_id, cdf, MAX_SEGMENTS);
    update_cdf(cdf, coded_id, MAX_SEGMENTS);
  }
  return 0;
}
static int skip_body(od_ec_enc *ec, uint16_t *cdf, int seg_skip_active, int skip_txfm) {
  if (seg_skip_active) return 1;
  od_ec_encode_cdf_q15(ec, skip_txfm, cdf, 2);
  update_cdf(cdf, skip_txfm, 2);
  return skip_txfm;
}
static void cdef_body(od_ec_enc *ec, int coded_lossless, int allow_intrabc, int mi_row,
                      int mi_col, int mib_size, int sb_size, int skip, int *transmitted,
                      int cdef_bits, int cdef_strength) {
  if (coded_lossless || allow_intrabc) return;
  int sb_mask = mib_size - 1;
  if ((mi_row & sb_mask) == 0 && (mi_col & sb_mask) == 0)
    transmitted[0] = transmitted[1] = transmitted[2] = transmitted[3] = 0;
  int cdef_size = 1 << (6 - MI_SIZE_LOG2), index_mask = cdef_size;
  int r = ((mi_row & index_mask) != 0), c = ((mi_col & index_mask) != 0);
  int index = (sb_size == BLOCK_128X128) ? c + 2 * r : 0;
  if (!transmitted[index] && !skip) {
    mi_literal(ec, cdef_strength, cdef_bits);
    transmitted[index] = 1;
  }
}
static void dqparams_body(od_ec_enc *ec, int dq_present, int dlf_present, int dlf_multi,
    int num_planes, int bsize, int sb_size, int skip, int sbul, int cur_qindex, int *base,
    int dq_res, const int *mbmi_dlf, int *xd_dlf, int mbmi_dlf_base, int *xd_dlf_base, int dlf_res,
    uint16_t *dq_cdf, uint16_t *dlf_multi_cdf, uint16_t *dlf_cdf) {
  if (!dq_present) return;
  if ((bsize != sb_size || skip == 0) && sbul) {
    dq_body(ec, dq_cdf, (cur_qindex - *base) / dq_res);
    *base = cur_qindex;
    if (dlf_present) {
      if (dlf_multi) {
        int flc = num_planes > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
        for (int lf = 0; lf < flc; ++lf) {
          dlf_body(ec, dlf_multi_cdf + lf * (DELTA_LF_PROBS + 2), (mbmi_dlf[lf] - xd_dlf[lf]) / dlf_res);
          xd_dlf[lf] = mbmi_dlf[lf];
        }
      } else {
        dlf_body(ec, dlf_cdf, (mbmi_dlf_base - *xd_dlf_base) / dlf_res);
        *xd_dlf_base = mbmi_dlf_base;
      }
    }
  }
}

uint32_t shim_write_mb_modes_kf_prefix(
    int segid_preskip, int seg_enabled, int update_map, int segment_id, int seg_pred,
    int last_active_segid, uint16_t *seg_cdf, int seg_skip_active, int skip_txfm, uint16_t *skip_cdf,
    int coded_lossless, int allow_intrabc, int mi_row, int mi_col, int mib_size, int sb_size,
    const int *cdef_trans_in, int cdef_bits, int cdef_strength, int dq_present, int dlf_present,
    int dlf_multi, int num_planes, int bsize, int cur_qindex, int cur_base_qindex, int dq_res,
    const int *mbmi_dlf, const int *xd_dlf_in, int mbmi_dlf_base, int xd_dlf_base_in, int dlf_res,
    uint16_t *dq_cdf, uint16_t *dlf_multi_cdf, uint16_t *dlf_cdf, uint8_t *out, int *out_skip,
    uint16_t *o_segcdf, uint16_t *o_skipcdf, int *o_cdef_trans, uint16_t *o_dqcdf, uint16_t *o_dlfmcdf,
    uint16_t *o_dlfcdf, int *o_base, int *o_xd_dlf, int *o_xd_dlf_base) {
  int cdef_trans[4], xd_dlf[FRAME_LF_COUNT];
  for (int i = 0; i < 4; i++) cdef_trans[i] = cdef_trans_in[i];
  for (int i = 0; i < FRAME_LF_COUNT; i++) xd_dlf[i] = xd_dlf_in[i];
  int base = cur_base_qindex, xd_dlf_base = xd_dlf_base_in;
  const int sbul = ((mi_row & (mib_size - 1)) == 0) && ((mi_col & (mib_size - 1)) == 0);
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (segid_preskip && update_map)
    seg_body(&ec, seg_cdf, seg_enabled, update_map, 0, segment_id, seg_pred, last_active_segid);
  int skip = skip_body(&ec, skip_cdf, seg_skip_active, skip_txfm);
  if (!segid_preskip && update_map)
    seg_body(&ec, seg_cdf, seg_enabled, update_map, skip, segment_id, seg_pred, last_active_segid);
  cdef_body(&ec, coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size, skip, cdef_trans,
            cdef_bits, cdef_strength);
  dqparams_body(&ec, dq_present, dlf_present, dlf_multi, num_planes, bsize, sb_size, skip, sbul,
                cur_qindex, &base, dq_res, mbmi_dlf, xd_dlf, mbmi_dlf_base, &xd_dlf_base, dlf_res,
                dq_cdf, dlf_multi_cdf, dlf_cdf);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  *out_skip = skip; *o_base = base; *o_xd_dlf_base = xd_dlf_base;
  for (int i = 0; i < 9; i++) o_segcdf[i] = seg_cdf[i];
  for (int i = 0; i < 3; i++) o_skipcdf[i] = skip_cdf[i];
  for (int i = 0; i < 4; i++) o_cdef_trans[i] = cdef_trans[i];
  for (int i = 0; i < DELTA_Q_PROBS + 2; i++) o_dqcdf[i] = dq_cdf[i];
  for (int i = 0; i < FRAME_LF_COUNT * (DELTA_LF_PROBS + 2); i++) o_dlfmcdf[i] = dlf_multi_cdf[i];
  for (int i = 0; i < DELTA_LF_PROBS + 2; i++) o_dlfcdf[i] = dlf_cdf[i];
  for (int i = 0; i < FRAME_LF_COUNT; i++) o_xd_dlf[i] = xd_dlf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_mb_modes_kf tail: intrabc + is_intrabc_block early-return + intra --- */
/* if (allow_intrabc) { write_intrabc_info; if is_intrabc_block RETURN }  write_intra_
 * prediction_modes.  Over ONE od_ec, reusing intrabc_into + wipm_into. */
uint32_t shim_kf_tail(
    int allow_intrabc, uint16_t *intrabc_cdf, uint16_t *joints, uint16_t *comp0, uint16_t *comp1,
    int use_intrabc, int diff_row, int diff_col,
    int mode, int bsize, uint16_t *y_cdf, int angle_delta_y, uint16_t *y_angle_cdf, int monochrome,
    int is_chroma_ref, int uv_mode, int cfl_allowed, int cfl_idx, int cfl_joint_sign,
    int angle_delta_uv, uint16_t *uv_mode_cdf, uint16_t *cfl_sign_cdf, uint16_t *cfl_alpha_cdf,
    uint16_t *uv_angle_cdf, int allow_palette, int bit_depth, const uint8_t *palette_size,
    const uint16_t *palette_colors, int mb_to_top_edge, int ha, const uint16_t *a_colors, int a_s0,
    int a_s1, int hl, const uint16_t *l_colors, int l_s0, int l_s1, uint16_t *pal_y_mode_cdf,
    uint16_t *pal_y_size_cdf, uint16_t *pal_uv_mode_cdf, uint16_t *pal_uv_size_cdf,
    int filter_allowed, int use_filter_intra, int filter_intra_mode, uint16_t *fi_use_cdf,
    uint16_t *fi_mode_cdf, uint8_t *out, uint16_t *o_intrabc, uint16_t *o_joints, uint16_t *o_c0,
    uint16_t *o_c1, uint16_t *o_all) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  wipm_setup_xd(&xd, &ami, &lmi, mb_to_top_edge, ha, a_colors, a_s0, a_s1, hl, l_colors, l_s0, l_s1);
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  int early = 0;
  if (allow_intrabc) {
    intrabc_into(&ec, intrabc_cdf, joints, comp0, comp1, use_intrabc, diff_row, diff_col);
    if (use_intrabc) early = 1; /* is_intrabc_block */
  }
  if (!early) {
    wipm_into(&ec, &xd, mode, bsize, y_cdf, angle_delta_y, y_angle_cdf, monochrome, is_chroma_ref,
              uv_mode, cfl_allowed, cfl_idx, cfl_joint_sign, angle_delta_uv, uv_mode_cdf, cfl_sign_cdf,
              cfl_alpha_cdf, uv_angle_cdf, allow_palette, bit_depth, palette_size, palette_colors,
              pal_y_mode_cdf, pal_y_size_cdf, pal_uv_mode_cdf, pal_uv_size_cdf, filter_allowed,
              use_filter_intra, filter_intra_mode, fi_use_cdf, fi_mode_cdf);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) o_intrabc[i] = intrabc_cdf[i];
  for (int i = 0; i < 5; i++) o_joints[i] = joints[i];
  for (int i = 0; i < 69; i++) { o_c0[i] = comp0[i]; o_c1[i] = comp1[i]; }
  int k = 0;
  for (int i = 0; i < 14; i++) o_all[k++] = y_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = y_angle_cdf[i];
  for (int i = 0; i < 15; i++) o_all[k++] = uv_mode_cdf[i];
  for (int i = 0; i < 9; i++) o_all[k++] = cfl_sign_cdf[i];
  for (int i = 0; i < 6 * 17; i++) o_all[k++] = cfl_alpha_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = uv_angle_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = pal_y_mode_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = pal_y_size_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = pal_uv_mode_cdf[i];
  for (int i = 0; i < 8; i++) o_all[k++] = pal_uv_size_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = fi_use_cdf[i];
  for (int i = 0; i < 6; i++) o_all[k++] = fi_mode_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_inter_segment_id (av1/encoder/bitstream.c:920) — inter-driver seg id --- */
/* av1_get_pred_context_seg_id facade: above+left neighbours' seg_id_predicted. */
int shim_get_pred_context_seg_id(int ha, int a_sip, int hl, int l_sip) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.seg_id_predicted = (uint8_t)a_sip;
  lmi.seg_id_predicted = (uint8_t)l_sip;
  xd.above_mbmi = ha ? &ami : (MB_MODE_INFO *)0;
  xd.left_mbmi = hl ? &lmi : (MB_MODE_INFO *)0;
  return av1_get_pred_context_seg_id(&xd);
}

/* write_inter_segment_id transcribed over od_ec. pred_cdf is the seg-id-predicted CDF
 * (segp->pred_cdf[ctx], caller-selected); seg_cdf is the spatial_pred_seg_cdf[cdf_num]
 * for the actual seg id (via seg_body). Only the CODED output is reproduced (the
 * set_spatial_segment_id side effects have no byte effect). */
/* write_inter_segment_id body writing into an existing od_ec (reused by the inter prefix). */
static void inter_seg_id_into(od_ec_enc *ec, int update_map, int preskip, int segid_preskip,
    int skip, int temporal_update, int seg_id_predicted, uint16_t *pred_cdf, uint16_t *seg_cdf,
    int seg_enabled, int segment_id, int seg_pred, int last_active_segid) {
  if (!update_map) return;
  int do_seg_block = 0;
  if (preskip) {
    if (segid_preskip) do_seg_block = 1;
  } else {
    if (!segid_preskip) {
      if (skip) {
        seg_body(ec, seg_cdf, seg_enabled, update_map, 1, segment_id, seg_pred, last_active_segid);
      } else {
        do_seg_block = 1;
      }
    }
  }
  if (do_seg_block) {
    if (temporal_update) {
      od_ec_encode_cdf_q15(ec, seg_id_predicted, pred_cdf, 2);
      update_cdf(pred_cdf, seg_id_predicted, 2);
      if (!seg_id_predicted)
        seg_body(ec, seg_cdf, seg_enabled, update_map, 0, segment_id, seg_pred, last_active_segid);
    } else {
      seg_body(ec, seg_cdf, seg_enabled, update_map, 0, segment_id, seg_pred, last_active_segid);
    }
  }
}
/* write_skip_mode / write_is_inter bodies (reused by the inter prefix). */
static void skip_mode_body(od_ec_enc *ec, uint16_t *cdf, int frame_flag, int seg_skip,
                           int comp_allowed, int seg_ref_gmv, int skip_mode) {
  if (!frame_flag || seg_skip || !comp_allowed || seg_ref_gmv) return;
  od_ec_encode_cdf_q15(ec, skip_mode, cdf, 2);
  update_cdf(cdf, skip_mode, 2);
}
static void is_inter_body(od_ec_enc *ec, uint16_t *cdf, int seg_ref_active, int seg_gmv_active,
                          int is_inter) {
  if (!seg_ref_active) {
    if (seg_gmv_active) return;
    od_ec_encode_cdf_q15(ec, is_inter, cdf, 2);
    update_cdf(cdf, is_inter, 2);
  }
}

uint32_t shim_write_inter_segment_id(int update_map, int preskip, int segid_preskip, int skip,
    int temporal_update, int seg_id_predicted, uint16_t *pred_cdf, uint16_t *seg_cdf,
    int seg_enabled, int segment_id, int seg_pred, int last_active_segid, uint8_t *out,
    uint16_t *o_predcdf, uint16_t *o_segcdf) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  inter_seg_id_into(&ec, update_map, preskip, segid_preskip, skip, temporal_update, seg_id_predicted,
                    pred_cdf, seg_cdf, seg_enabled, segment_id, seg_pred, last_active_segid);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 3; i++) o_predcdf[i] = pred_cdf[i];
  for (int i = 0; i < 9; i++) o_segcdf[i] = seg_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- pack_inter_mode_mvs prefix (av1/encoder/bitstream.c:1092) --- */
/* inter_segment_id(preskip) -> skip_mode -> skip -> inter_segment_id(postskip) -> cdef
 * -> delta_q_params -> is_inter(if !skip_mode), over ONE od_ec. Returns skip + skip_mode
 * (the caller returns early on skip_mode). Reuses all the extracted inline bodies. */
uint32_t shim_write_inter_prefix(
    int update_map, int segid_preskip, int temporal_update, int seg_id_predicted, uint16_t *pred_cdf,
    uint16_t *seg_cdf, int seg_enabled, int segment_id, int seg_pred, int last_active_segid,
    uint16_t *skip_mode_cdf, int frame_skip_mode_flag, int sm_seg_skip, int sm_comp_allowed,
    int sm_seg_ref_gmv, int skip_mode, uint16_t *skip_cdf, int skip_seg_active, int skip_txfm,
    int coded_lossless, int allow_intrabc, int mi_row, int mi_col, int mib_size, int sb_size,
    const int *cdef_trans_in, int cdef_bits, int cdef_strength, int dq_present, int dlf_present,
    int dlf_multi, int num_planes, int bsize, int cur_qindex, int cur_base_qindex, int dq_res,
    const int *mbmi_dlf, const int *xd_dlf_in, int mbmi_dlf_base, int xd_dlf_base_in, int dlf_res,
    uint16_t *dq_cdf, uint16_t *dlf_multi_cdf, uint16_t *dlf_cdf, uint16_t *intra_inter_cdf,
    int seg_ref_frame_active, int seg_globalmv_active, int is_inter, uint8_t *out, int *out_skip,
    int *out_skip_mode, uint16_t *o_predcdf, uint16_t *o_segcdf, uint16_t *o_smcdf, uint16_t *o_skipcdf,
    int *o_cdef_trans, uint16_t *o_dqcdf, uint16_t *o_dlfmcdf, uint16_t *o_dlfcdf, int *o_base,
    int *o_xd_dlf, int *o_xd_dlf_base, uint16_t *o_iicdf) {
  int cdef_trans[4], xd_dlf[FRAME_LF_COUNT];
  for (int i = 0; i < 4; i++) cdef_trans[i] = cdef_trans_in[i];
  for (int i = 0; i < FRAME_LF_COUNT; i++) xd_dlf[i] = xd_dlf_in[i];
  int base = cur_base_qindex, xd_dlf_base = xd_dlf_base_in;
  const int sbul = ((mi_row & (mib_size - 1)) == 0) && ((mi_col & (mib_size - 1)) == 0);
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  inter_seg_id_into(&ec, update_map, 1, segid_preskip, 0, temporal_update, seg_id_predicted,
                    pred_cdf, seg_cdf, seg_enabled, segment_id, seg_pred, last_active_segid);
  skip_mode_body(&ec, skip_mode_cdf, frame_skip_mode_flag, sm_seg_skip, sm_comp_allowed,
                 sm_seg_ref_gmv, skip_mode);
  int skip = skip_mode ? 1 : skip_body(&ec, skip_cdf, skip_seg_active, skip_txfm);
  inter_seg_id_into(&ec, update_map, 0, segid_preskip, skip, temporal_update, seg_id_predicted,
                    pred_cdf, seg_cdf, seg_enabled, segment_id, seg_pred, last_active_segid);
  cdef_body(&ec, coded_lossless, allow_intrabc, mi_row, mi_col, mib_size, sb_size, skip, cdef_trans,
            cdef_bits, cdef_strength);
  dqparams_body(&ec, dq_present, dlf_present, dlf_multi, num_planes, bsize, sb_size, skip, sbul,
                cur_qindex, &base, dq_res, mbmi_dlf, xd_dlf, mbmi_dlf_base, &xd_dlf_base, dlf_res,
                dq_cdf, dlf_multi_cdf, dlf_cdf);
  if (!skip_mode) is_inter_body(&ec, intra_inter_cdf, seg_ref_frame_active, seg_globalmv_active, is_inter);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  *out_skip = skip; *out_skip_mode = skip_mode; *o_base = base; *o_xd_dlf_base = xd_dlf_base;
  for (int i = 0; i < 3; i++) { o_predcdf[i] = pred_cdf[i]; o_smcdf[i] = skip_mode_cdf[i]; o_skipcdf[i] = skip_cdf[i]; o_iicdf[i] = intra_inter_cdf[i]; }
  for (int i = 0; i < 9; i++) o_segcdf[i] = seg_cdf[i];
  for (int i = 0; i < 4; i++) o_cdef_trans[i] = cdef_trans[i];
  for (int i = 0; i < DELTA_Q_PROBS + 2; i++) o_dqcdf[i] = dq_cdf[i];
  for (int i = 0; i < FRAME_LF_COUNT * (DELTA_LF_PROBS + 2); i++) o_dlfmcdf[i] = dlf_multi_cdf[i];
  for (int i = 0; i < DELTA_LF_PROBS + 2; i++) o_dlfcdf[i] = dlf_cdf[i];
  for (int i = 0; i < FRAME_LF_COUNT; i++) o_xd_dlf[i] = xd_dlf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- inter-mode-body gates + mode_context_analyzer (mvref_common.h) --- */
int shim_is_inter_compound_mode(int mode) { return is_inter_compound_mode((PREDICTION_MODE)mode); }
int shim_is_inter_singleref_mode(int mode) { return is_inter_singleref_mode((PREDICTION_MODE)mode); }
int shim_have_nearmv_in_inter_mode(int mode) { return have_nearmv_in_inter_mode((PREDICTION_MODE)mode); }

/* av1_mode_context_analyzer facade: rf=[rf0,rf1]; mode_context[av1_ref_frame_type(rf)]=mc_val. */
int shim_mode_context_analyzer(int rf0, int rf1, int mc_val) {
  MV_REFERENCE_FRAME rf[2] = { (MV_REFERENCE_FRAME)rf0, (MV_REFERENCE_FRAME)rf1 };
  int16_t mode_context[MODE_CTX_REF_FRAMES];
  for (int i = 0; i < MODE_CTX_REF_FRAMES; i++) mode_context[i] = 0;
  const int8_t idx = av1_ref_frame_type(rf);
  mode_context[idx] = (int16_t)mc_val;
  return av1_mode_context_analyzer(mode_context, rf);
}

/* --- inter-block MV coding (pack_inter_mode_mvs, bitstream.c) --- */
/* The mode-dependent av1_encode_mv calls: NEWMV/NEW_NEWMV code one MV per ref (0..1+
 * is_compound); NEAREST_NEWMV/NEAR_NEWMV code ref 1; NEW_NEARESTMV/NEW_NEARMV code ref 0.
 * All share one nmvc (joints/comps adapt across the two refs). usehp is the caller's
 * resolved precision (allow_hp / MV_SUBPEL_NONE under force_integer_mv). */
uint32_t shim_write_inter_block_mvs(int mode, int is_compound, int diff_row0, int diff_col0,
    int diff_row1, int diff_col1, int usehp, uint16_t *joints, uint16_t *comp0, uint16_t *comp1,
    uint8_t *out, uint16_t *o_joints, uint16_t *o_c0, uint16_t *o_c1) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (mode == NEWMV || mode == NEW_NEWMV) {
    for (int ref = 0; ref < 1 + is_compound; ++ref) {
      if (ref == 0) encode_mv_into(&ec, joints, comp0, comp1, diff_row0, diff_col0, usehp);
      else encode_mv_into(&ec, joints, comp0, comp1, diff_row1, diff_col1, usehp);
    }
  } else if (mode == NEAREST_NEWMV || mode == NEAR_NEWMV) {
    encode_mv_into(&ec, joints, comp0, comp1, diff_row1, diff_col1, usehp);
  } else if (mode == NEW_NEARESTMV || mode == NEW_NEARMV) {
    encode_mv_into(&ec, joints, comp0, comp1, diff_row0, diff_col0, usehp);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 5; i++) o_joints[i] = joints[i];
  for (int i = 0; i < 69; i++) { o_c0[i] = comp0[i]; o_c1[i] = comp1[i]; }
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- inter mode + drl (pack_inter_mode_mvs, bitstream.c) --- */
/* if !seg_skip: compound-mode symbol OR the single-ref inter-mode cascade, then the drl
 * index (for NEWMV/NEW_NEWMV/near modes). inter_compound_mode_cdf is pre-selected
 * [mode_ctx] (8-sym); newmv/zeromv/refmv are full tables indexed by mode_ctx. */
uint32_t shim_write_inter_mode_drl(int seg_skip, int mode, int mode_ctx,
    uint16_t *inter_compound_mode_cdf, uint16_t *newmv_cdf, uint16_t *zeromv_cdf,
    uint16_t *refmv_cdf, uint16_t *drl_cdf, int ref_mv_idx, int ref_mv_count,
    const uint16_t *weight, uint8_t *out, uint16_t *o_icm, uint16_t *o_newmv, uint16_t *o_zeromv,
    uint16_t *o_refmv, uint16_t *o_drl) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (!seg_skip) {
    if (is_inter_compound_mode((PREDICTION_MODE)mode)) {
      od_ec_encode_cdf_q15(&ec, mode - NEAREST_NEARESTMV, inter_compound_mode_cdf, INTER_COMPOUND_MODES);
      update_cdf(inter_compound_mode_cdf, mode - NEAREST_NEARESTMV, INTER_COMPOUND_MODES);
    } else if (is_inter_singleref_mode((PREDICTION_MODE)mode)) {
      inter_mode_into(&ec, newmv_cdf, zeromv_cdf, refmv_cdf, mode, mode_ctx);
    }
    if (mode == NEWMV || mode == NEW_NEWMV || have_nearmv_in_inter_mode((PREDICTION_MODE)mode)) {
      drl_into(&ec, drl_cdf, mode, ref_mv_idx, ref_mv_count, weight);
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 9; i++) o_icm[i] = inter_compound_mode_cdf[i];
  for (int i = 0; i < 6 * 3; i++) { o_newmv[i] = newmv_cdf[i]; o_refmv[i] = refmv_cdf[i]; }
  for (int i = 0; i < 2 * 3; i++) o_zeromv[i] = zeromv_cdf[i];
  for (int i = 0; i < 3 * 3; i++) o_drl[i] = drl_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- inter mode-body tail: interintra + motion_mode + compound_type + interp_filter --- */
/* Over ONE od_ec. interintra + compound_type are mutually exclusive (ref[1]==INTRA vs
 * ref[1]>INTRA) and share the one wedge_idx_cdf. Gates are the caller's. */
uint32_t shim_write_inter_mode_tail(
    int interintra_allowed, int interintra, uint16_t *ii_cdf, int ii_mode, uint16_t *ii_mode_cdf,
    int wedge_used_ii, int use_wedge_ii, uint16_t *wedge_ii_cdf, int ii_wedge_index,
    uint16_t *wedge_idx_cdf, int motion_mode_present, uint16_t *obmc_cdf, uint16_t *mm_cdf,
    int last_motion_mode_allowed, int motion_mode, int has_second_ref, int masked_used,
    int comp_group_idx, uint16_t *cgi_cdf, int dist_wtd, int compound_idx, uint16_t *cidx_cdf,
    int wedge_used_ct, int comp_type, uint16_t *ctype_cdf, int ct_wedge_index, int wedge_sign,
    int mask_type, int interp_needed, int is_switchable, int enable_dual, int f0, int f1,
    uint16_t *interp_cdf0, uint16_t *interp_cdf1, uint8_t *out, uint16_t *o_all) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  if (interintra_allowed)
    interintra_into(&ec, interintra, ii_cdf, ii_mode, ii_mode_cdf, wedge_used_ii, use_wedge_ii,
                    wedge_ii_cdf, ii_wedge_index, wedge_idx_cdf);
  if (motion_mode_present)
    motion_mode_into(&ec, obmc_cdf, mm_cdf, last_motion_mode_allowed, motion_mode);
  if (has_second_ref)
    compound_type_into(&ec, masked_used, comp_group_idx, cgi_cdf, dist_wtd, compound_idx, cidx_cdf,
                       wedge_used_ct, comp_type, ctype_cdf, ct_wedge_index, wedge_idx_cdf, wedge_sign,
                       mask_type);
  interp_filter_into(&ec, interp_cdf0, interp_cdf1, interp_needed, is_switchable, enable_dual, f0, f1);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  int k = 0;
  for (int i = 0; i < 3; i++) o_all[k++] = ii_cdf[i];
  for (int i = 0; i < 5; i++) o_all[k++] = ii_mode_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = wedge_ii_cdf[i];
  for (int i = 0; i < 17; i++) o_all[k++] = wedge_idx_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = obmc_cdf[i];
  for (int i = 0; i < 4; i++) o_all[k++] = mm_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = cgi_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = cidx_cdf[i];
  for (int i = 0; i < 3; i++) o_all[k++] = ctype_cdf[i];
  for (int i = 0; i < 4; i++) o_all[k++] = interp_cdf0[i];
  for (int i = 0; i < 4; i++) o_all[k++] = interp_cdf1[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- av1_collect_neighbors_ref_counts (mvref_common.h) --- */
/* Facade: tally the above/left inter neighbours' ref frames into neighbors_ref_counts[8]. */
void shim_collect_neighbors_ref_counts(int ha, int a_intrabc, int a_rf0, int a_rf1,
    int hl, int l_intrabc, int l_rf0, int l_rf1, unsigned char *out_counts) {
  MB_MODE_INFO ami, lmi;
  MACROBLOCKD xd;
  ami.use_intrabc = (uint8_t)a_intrabc;
  ami.ref_frame[0] = (MV_REFERENCE_FRAME)a_rf0;
  ami.ref_frame[1] = (MV_REFERENCE_FRAME)a_rf1;
  lmi.use_intrabc = (uint8_t)l_intrabc;
  lmi.ref_frame[0] = (MV_REFERENCE_FRAME)l_rf0;
  lmi.ref_frame[1] = (MV_REFERENCE_FRAME)l_rf1;
  xd.above_mbmi = &ami;
  xd.left_mbmi = &lmi;
  xd.up_available = ha;
  xd.left_available = hl;
  av1_collect_neighbors_ref_counts(&xd);
  for (int i = 0; i < REF_FRAMES; i++) out_counts[i] = xd.neighbors_ref_counts[i];
}

/* --- get_partition_subsize (av1/common/common_data.h) --- */
int shim_get_partition_subsize(int bsize, int partition) {
  return get_partition_subsize((BLOCK_SIZE)bsize, (PARTITION_TYPE)partition);
}

/* --- update_ext_partition_context (av1_common_int.h) --- */
/* Facade over the real fn: above_partition_context is a pointer (64-slot buffer here),
 * left_partition_context is inline char[MAX_MIB_SIZE=32]. Threads both in/out. */
void shim_update_ext_partition_context(int mi_row, int mi_col, int subsize, int bsize,
    int partition, const signed char *above_in, const signed char *left_in,
    signed char *above_out, signed char *left_out) {
  MACROBLOCKD xd;
  static signed char above[64];
  for (int i = 0; i < 64; i++) above[i] = above_in[i];
  for (int i = 0; i < 32; i++) xd.left_partition_context[i] = left_in[i];
  xd.above_partition_context = (PARTITION_CONTEXT *)above;
  update_ext_partition_context(&xd, mi_row, mi_col, (BLOCK_SIZE)subsize, (BLOCK_SIZE)bsize,
                               (PARTITION_TYPE)partition);
  for (int i = 0; i < 64; i++) above_out[i] = above[i];
  for (int i = 0; i < 32; i++) left_out[i] = xd.left_partition_context[i];
}

/* --- write_modes_sb per-node partition step (av1/encoder/bitstream.c) --- */
/* One partition node over ONE od_ec: ctx = partition_plane_context(threaded above/left)
 * -> write_partition on partition_cdf_arena[ctx] (full CDF adapts; edge gathers don't)
 * -> update_ext_partition_context. Reuses the real partition_plane_context / gather /
 * update fns + partition_cdf_length. arena is [PARTITION_CONTEXTS=20][11]. */
uint32_t shim_write_partition_node(const signed char *above_in, const signed char *left_in,
    int mi_row, int mi_col, int bsize, int partition, int mi_rows, int mi_cols,
    uint16_t *arena, uint8_t *out, signed char *above_out, signed char *left_out,
    uint16_t *arena_out) {
  MACROBLOCKD xd;
  static signed char above[64];
  for (int i = 0; i < 64; i++) above[i] = above_in[i];
  for (int i = 0; i < 32; i++) xd.left_partition_context[i] = left_in[i];
  xd.above_partition_context = (PARTITION_CONTEXT *)above;

  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  const int hbs = mi_size_wide[bsize] / 2;
  const int subsize = get_partition_subsize((BLOCK_SIZE)bsize, (PARTITION_TYPE)partition);
  if (bsize >= BLOCK_8X8) {
    const int has_rows = (mi_row + hbs) < mi_rows;
    const int has_cols = (mi_col + hbs) < mi_cols;
    const int ctx = partition_plane_context(&xd, mi_row, mi_col, (BLOCK_SIZE)bsize);
    uint16_t *cdf = arena + ctx * 11;
    if (!has_rows && !has_cols) {
      /* forced SPLIT, nothing coded */
    } else if (has_rows && has_cols) {
      int n = partition_cdf_length((BLOCK_SIZE)bsize);
      od_ec_encode_cdf_q15(&ec, partition, cdf, n);
      update_cdf(cdf, partition, n);
    } else if (!has_rows && has_cols) {
      aom_cdf_prob g[2];
      partition_gather_vert_alike(g, cdf, (BLOCK_SIZE)bsize);
      od_ec_encode_cdf_q15(&ec, partition == PARTITION_SPLIT, g, 2);
    } else {
      aom_cdf_prob g[2];
      partition_gather_horz_alike(g, cdf, (BLOCK_SIZE)bsize);
      od_ec_encode_cdf_q15(&ec, partition == PARTITION_SPLIT, g, 2);
    }
  }
  update_ext_partition_context(&xd, mi_row, mi_col, (BLOCK_SIZE)subsize, (BLOCK_SIZE)bsize,
                               (PARTITION_TYPE)partition);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 64; i++) above_out[i] = above[i];
  for (int i = 0; i < 32; i++) left_out[i] = xd.left_partition_context[i];
  for (int i = 0; i < PARTITION_CONTEXTS * 11; i++) arena_out[i] = arena[i];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_modes_sb partition-tree recursion (av1/encoder/bitstream.c) --- */
/* Fully-in-frame variant, block content stubbed (write_modes_b validated separately):
 * per node (bsize>=8X8) consume the pre-order partition, write_partition on
 * arena[ctx] (context threaded through the tree), recurse 4x on SPLIT, then
 * update_ext_partition_context. bsize<8X8 nodes (4X4 leaves under 8X8 SPLIT) do nothing. */
static void wms_recurse(od_ec_enc *ec, MACROBLOCKD *xd, const signed char *tree, int *idx,
    int mi_row, int mi_col, int bsize, uint16_t *arena) {
  if (bsize < BLOCK_8X8) return; /* 4X4 leaf: no partition, no context */
  const int p = tree[(*idx)++];
  const int subsize = get_partition_subsize((BLOCK_SIZE)bsize, (PARTITION_TYPE)p);
  const int hbs = mi_size_wide[bsize] / 2;
  const int ctx = partition_plane_context(xd, mi_row, mi_col, (BLOCK_SIZE)bsize);
  uint16_t *cdf = arena + ctx * 11;
  const int n = partition_cdf_length((BLOCK_SIZE)bsize);
  od_ec_encode_cdf_q15(ec, p, cdf, n);
  update_cdf(cdf, p, n);
  if (p == PARTITION_SPLIT && bsize > BLOCK_8X8) {
    wms_recurse(ec, xd, tree, idx, mi_row, mi_col, subsize, arena);
    wms_recurse(ec, xd, tree, idx, mi_row, mi_col + hbs, subsize, arena);
    wms_recurse(ec, xd, tree, idx, mi_row + hbs, mi_col, subsize, arena);
    wms_recurse(ec, xd, tree, idx, mi_row + hbs, mi_col + hbs, subsize, arena);
  }
  /* other partitions: stub blocks (no bytes) */
  update_ext_partition_context(xd, mi_row, mi_col, (BLOCK_SIZE)subsize, (BLOCK_SIZE)bsize,
                               (PARTITION_TYPE)p);
}

uint32_t shim_write_modes_sb(const signed char *above_in, const signed char *left_in,
    int mi_row, int mi_col, int bsize, const signed char *tree, int tree_len, uint16_t *arena,
    uint8_t *out, signed char *above_out, signed char *left_out, uint16_t *arena_out,
    int *tree_consumed) {
  MACROBLOCKD xd;
  static signed char above[64];
  for (int i = 0; i < 64; i++) above[i] = above_in[i];
  for (int i = 0; i < 32; i++) xd.left_partition_context[i] = left_in[i];
  xd.above_partition_context = (PARTITION_CONTEXT *)above;
  (void)tree_len;
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  int idx = 0;
  wms_recurse(&ec, &xd, tree, &idx, mi_row, mi_col, bsize, arena);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 64; i++) above_out[i] = above[i];
  for (int i = 0; i < 32; i++) left_out[i] = xd.left_partition_context[i];
  for (int i = 0; i < PARTITION_CONTEXTS * 11; i++) arena_out[i] = arena[i];
  *tree_consumed = idx;
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- write_modes tile loop (av1/encoder/bitstream.c) --- */
/* Partition-only (stubbed blocks): zero the above partition context once for the tile,
 * then per SB row zero the left context and walk each SB (row-major) via wms_recurse,
 * consuming the concatenated pre-order tree. above threads vertically across the tile;
 * left threads horizontally within a row. tile is n_sb_rows x n_sb_cols SBs of sb_mi mi. */
uint32_t shim_write_modes_tile(int n_sb_rows, int n_sb_cols, int sb_mi, int sb_size,
    const signed char *tree, uint16_t *arena, uint8_t *out, signed char *above_out,
    uint16_t *arena_out, int *tree_consumed) {
  MACROBLOCKD xd;
  static signed char above[128];
  for (int i = 0; i < 128; i++) above[i] = 0; /* av1_zero_above_context */
  xd.above_partition_context = (PARTITION_CONTEXT *)above;
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  int idx = 0;
  for (int r = 0; r < n_sb_rows; r++) {
    for (int i = 0; i < 32; i++) xd.left_partition_context[i] = 0; /* av1_zero_left_context per SB row */
    for (int c = 0; c < n_sb_cols; c++) {
      wms_recurse(&ec, &xd, tree, &idx, r * sb_mi, c * sb_mi, sb_size, arena);
    }
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 128; i++) above_out[i] = above[i];
  for (int i = 0; i < PARTITION_CONTEXTS * 11; i++) arena_out[i] = arena[i];
  *tree_consumed = idx;
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- block-level inter var-tx-size loop (write_modes_b, av1/encoder/bitstream.c) --- */
/* Drives write_tx_size_vartx across the block's txb grid (idy/idx step max_tx units),
 * threading the above/left txfm context through the loop. max_tx = get_vartx_max_txsize
 * (luma non-lossless = max_txsize_rect_lookup[bsize]). */
uint32_t shim_write_inter_txfm_size(int bsize, int max_tx, const uint8_t *inter_tx_size,
                                    int mb_to_right_edge, int mb_to_bottom_edge,
                                    const uint8_t *above_in, const uint8_t *left_in, uint16_t *cdf,
                                    uint8_t *out, uint8_t *above_out, uint8_t *left_out,
                                    uint16_t *cdf_out) {
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
  const int txbh = tx_size_high_unit[max_tx];
  const int txbw = tx_size_wide_unit[max_tx];
  const int width = mi_size_wide[bsize];
  const int height = mi_size_high[bsize];
  for (int idy = 0; idy < height; idy += txbh)
    for (int idx = 0; idx < width; idx += txbw)
      shim_wtsv_rec(&xd, &mbmi, (TX_SIZE)max_tx, 0, idy, idx, &ec, local_cdf);
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 32; i++) { above_out[i] = above[i]; left_out[i] = left[i]; }
  for (int i = 0; i < 21; i++) for (int j = 0; j < 3; j++) cdf_out[i * 3 + j] = local_cdf[i][j];
  od_ec_enc_clear(&ec);
  return nb;
}

/* --- pack_map_tokens (palette colour-index map, av1/encoder/bitstream.c) --- */
/* First index via write_uniform(n); the rest via aom_write_symbol on map_pb_cdf
 * [size_idx][color_ctx] (n symbols). map_cdf is the [PALETTE_COLOR_INDEX_CONTEXTS=5][9]
 * slice for this palette size. tokens[i] is the colour index, color_ctxs[i] its context
 * (unused for i==0). Transcribed over od_ec (aom_write_literal -> mi_literal). */
uint32_t shim_pack_map_tokens(int n, int num, const uint8_t *tokens, const uint8_t *color_ctxs,
                              uint16_t *map_cdf, uint8_t *out, uint16_t *map_cdf_out) {
  od_ec_enc ec; od_ec_enc_init(&ec, 256);
  const int l = n > 0 ? get_msb(n) + 1 : 0;
  const int m = (1 << l) - n;
  const int v = tokens[0];
  if (l != 0) {
    if (v < m) {
      mi_literal(&ec, v, l - 1);
    } else {
      mi_literal(&ec, m + ((v - m) >> 1), l - 1);
      mi_bit(&ec, (v - m) & 1);
    }
  }
  for (int i = 1; i < num; i++) {
    uint16_t *cdf = map_cdf + color_ctxs[i] * 9;
    od_ec_encode_cdf_q15(&ec, tokens[i], cdf, n);
    update_cdf(cdf, tokens[i], n);
  }
  uint32_t nb = 0; const unsigned char *buf = od_ec_enc_done(&ec, &nb);
  for (uint32_t i = 0; i < nb; i++) out[i] = buf[i];
  for (int i = 0; i < 5 * 9; i++) map_cdf_out[i] = map_cdf[i];
  od_ec_enc_clear(&ec);
  return nb;
}
