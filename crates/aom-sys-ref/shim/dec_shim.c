/* dec_shim.c — decoder-track oracles.
 *
 *  1. MACROBLOCKD facades over the REAL static inlines (pred_common.h /
 *     av1_common_int.h / blockd.h): get_tx_size_context, set_txfm_ctxs,
 *     is_chroma_reference, av1_get_max_uv_txsize, av1_get_tx_type (intra UV
 *     arm), intra_mode_to_tx_type, tx_size_from_tx_mode, depth_to_tx_size,
 *     bsize_to_max_depth, bsize_to_tx_size_cat.
 *     scale_chroma_bsize is the one VERBATIM TRANSCRIPTION here (static in
 *     reconintra.c, not reachable from a header; body copied unchanged).
 *
 *  2. shim_dump_default_kf_fc — drive the REAL exported
 *     av1_setup_past_independence over a minimal heap AV1_COMMON (fc /
 *     default_frame_context / cur_frame / base_qindex are the only fields the
 *     call chain touches: av1_clearall_segfeatures(&cm->seg), the cur_frame
 *     seg_map memset (skipped: NULL), ref/mode deltas, av1_default_coef_probs
 *     (reads quant_params.base_qindex), av1_init_mode_probs(cm->fc),
 *     av1_init_mv_probs(cm), av1_setup_frame_contexts (copies *fc to
 *     *default_frame_context; large_scale=0 skips the buffer-pool arm)), then
 *     memcpy the KF-path FRAME_CONTEXT fields to a flat u16 layout mirroring
 *     aom-entropy's KfFrameContext field order (coeff arena LAST, in aom-txb's
 *     CdfArena region layout). Total DUMP_KF_FC_LEN = 6421 u16.
 *
 *  3. shim_encode_av1_kf / shim_decode_av1_kf — the REAL public codec API
 *     (aom_codec_av1_cx / aom_codec_av1_dx): produce a production KEY-frame
 *     bitstream in-process (the same library+path the aomenc CLI drives) and
 *     the gold C-decoder pixel oracle for arbitrary AV1 bytes.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "av1/common/av1_common_int.h"
#include "av1/common/blockd.h"
#include "av1/common/pred_common.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/entropymv.h"

/* ------------------------------------------------------------------ */
/* 1. MACROBLOCKD facades                                              */
/* ------------------------------------------------------------------ */

/* get_tx_size_context (pred_common.h): reads xd->mi[0]->bsize, the first
 * above/left txfm-context bytes, availability, and the neighbour mbmi
 * inter-ness/bsize (inter neighbours override with their block dims). */
int shim_get_tx_size_context(int bsize, uint8_t above_txfm, uint8_t left_txfm,
                             int up_available, int left_available,
                             int above_bsize, int above_inter, int left_bsize,
                             int left_inter) {
  MB_MODE_INFO mi, above_mi, left_mi;
  memset(&mi, 0, sizeof(mi));
  memset(&above_mi, 0, sizeof(above_mi));
  memset(&left_mi, 0, sizeof(left_mi));
  mi.bsize = (BLOCK_SIZE)bsize;
  above_mi.bsize = (BLOCK_SIZE)above_bsize;
  above_mi.ref_frame[0] = above_inter ? LAST_FRAME : INTRA_FRAME;
  left_mi.bsize = (BLOCK_SIZE)left_bsize;
  left_mi.ref_frame[0] = left_inter ? LAST_FRAME : INTRA_FRAME;

  MB_MODE_INFO *mip = &mi;
  uint8_t atc = above_txfm, ltc = left_txfm;
  MACROBLOCKD xd;
  memset(&xd, 0, sizeof(xd));
  xd.mi = &mip;
  xd.above_mbmi = up_available ? &above_mi : NULL;
  xd.left_mbmi = left_available ? &left_mi : NULL;
  xd.up_available = up_available;
  xd.left_available = left_available;
  xd.above_txfm_context = &atc;
  xd.left_txfm_context = &ltc;
  return get_tx_size_context(&xd);
}

/* set_txfm_ctxs (av1_common_int.h): stamps bw into above[0..n4_w) and bh into
 * left[0..n4_h) (block dims instead when skip). */
void shim_set_txfm_ctxs(int tx_size, int n4_w, int n4_h, int skip,
                        uint8_t *above, uint8_t *left) {
  MACROBLOCKD xd;
  memset(&xd, 0, sizeof(xd));
  xd.above_txfm_context = above;
  xd.left_txfm_context = left;
  set_txfm_ctxs((TX_SIZE)tx_size, n4_w, n4_h, skip, &xd);
}

/* is_chroma_reference (av1_common_int.h) — pure. */
int shim_is_chroma_reference(int mi_row, int mi_col, int bsize, int ss_x,
                             int ss_y) {
  return is_chroma_reference(mi_row, mi_col, (BLOCK_SIZE)bsize, ss_x, ss_y);
}

/* av1_get_max_uv_txsize (blockd.h) — pure (asserts the (bsize,ss) combo maps
 * to a valid plane bsize; callers stay in the valid domain). */
int shim_get_max_uv_txsize(int bsize, int ss_x, int ss_y) {
  return (int)av1_get_max_uv_txsize((BLOCK_SIZE)bsize, ss_x, ss_y);
}

/* intra_mode_to_tx_type (blockd.h): Y arm keys on mbmi->mode, UV arm on
 * get_uv_mode(mbmi->uv_mode). */
int shim_intra_mode_to_tx_type(int y_mode, int uv_mode, int plane_type) {
  MB_MODE_INFO mi;
  memset(&mi, 0, sizeof(mi));
  mi.mode = (PREDICTION_MODE)y_mode;
  mi.uv_mode = (UV_PREDICTION_MODE)uv_mode;
  return (int)intra_mode_to_tx_type(&mi, (PLANE_TYPE)plane_type);
}

/* av1_get_tx_type (blockd.h), intra UV arm: lossless/size gates ->
 * intra_mode_to_tx_type(UV) -> ext-tx-set membership downgrade to DCT_DCT.
 * The intra path never reads blk_row/blk_col/tx_type_map. */
int shim_av1_get_tx_type_uv_intra(int y_mode, int uv_mode, int uv_tx_size,
                                  int reduced_tx_set, int lossless) {
  MB_MODE_INFO mi;
  memset(&mi, 0, sizeof(mi));
  mi.mode = (PREDICTION_MODE)y_mode;
  mi.uv_mode = (UV_PREDICTION_MODE)uv_mode;
  mi.ref_frame[0] = INTRA_FRAME;
  MB_MODE_INFO *mip = &mi;
  MACROBLOCKD xd;
  memset(&xd, 0, sizeof(xd));
  xd.mi = &mip;
  xd.lossless[0] = lossless;
  return (int)av1_get_tx_type(&xd, PLANE_TYPE_UV, 0, 0, (TX_SIZE)uv_tx_size,
                              reduced_tx_set);
}

/* tx_size_from_tx_mode / depth_to_tx_size (blockd.h) — pure.
 * (bsize_to_max_depth / bsize_to_tx_size_cat live in modeinfo_shim.c.) */
int shim_tx_size_from_tx_mode(int bsize, int tx_mode) {
  return (int)tx_size_from_tx_mode((BLOCK_SIZE)bsize, (TX_MODE)tx_mode);
}
int shim_depth_to_tx_size(int depth, int bsize) {
  return (int)depth_to_tx_size(depth, (BLOCK_SIZE)bsize);
}

/* scale_chroma_bsize — VERBATIM TRANSCRIPTION of the static inline in
 * av1/common/reconintra.c:1637 (not reachable from any header; including the
 * .c would redefine its exported symbols against libaom.a). Body unchanged. */
static inline BLOCK_SIZE dec_shim_scale_chroma_bsize(BLOCK_SIZE bsize,
                                                     int subsampling_x,
                                                     int subsampling_y) {
  assert(subsampling_x >= 0 && subsampling_x < 2);
  assert(subsampling_y >= 0 && subsampling_y < 2);
  BLOCK_SIZE bs = bsize;
  switch (bsize) {
    case BLOCK_4X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X4;
      else if (subsampling_y == 1)
        bs = BLOCK_4X8;
      break;
    case BLOCK_4X8:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X8;
      else if (subsampling_y == 1)
        bs = BLOCK_4X8;
      break;
    case BLOCK_8X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X8;
      else if (subsampling_x == 1)
        bs = BLOCK_8X4;
      else if (subsampling_y == 1)
        bs = BLOCK_8X8;
      break;
    case BLOCK_4X16:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_8X16;
      else if (subsampling_x == 1)
        bs = BLOCK_8X16;
      else if (subsampling_y == 1)
        bs = BLOCK_4X16;
      break;
    case BLOCK_16X4:
      if (subsampling_x == 1 && subsampling_y == 1)
        bs = BLOCK_16X8;
      else if (subsampling_x == 1)
        bs = BLOCK_16X4;
      else if (subsampling_y == 1)
        bs = BLOCK_16X8;
      break;
    default: break;
  }
  return bs;
}

int shim_scale_chroma_bsize(int bsize, int ss_x, int ss_y) {
  return (int)dec_shim_scale_chroma_bsize((BLOCK_SIZE)bsize, ss_x, ss_y);
}

/* ------------------------------------------------------------------ */
/* 2. Default KF FRAME_CONTEXT dump                                    */
/* ------------------------------------------------------------------ */

/* Flat u16 layout — MUST mirror aom-entropy KfFrameContext field order.
 * Mode fields first (exact-sized: ext-tx instances sliced to nsym+1 leading
 * slots), then the aom-txb coefficient arena (4045). */
#define DUMP_KF_FC_LEN 6421

static uint16_t *dump_nmv_comp(const nmv_component *c, uint16_t *p) {
  /* aom-entropy 69-u16 nmv_component packing:
   *   sign 0..3 / classes 3..15 / class0 15..18 / bits[10] 18..48 /
   *   class0_fp[2] 48..58 / fp 58..63 / class0_hp 63..66 / hp 66..69 */
  memcpy(p, c->sign_cdf, 3 * sizeof(uint16_t));
  p += 3;
  memcpy(p, c->classes_cdf, 12 * sizeof(uint16_t));
  p += 12;
  memcpy(p, c->class0_cdf, 3 * sizeof(uint16_t));
  p += 3;
  memcpy(p, c->bits_cdf, 30 * sizeof(uint16_t));
  p += 30;
  memcpy(p, c->class0_fp_cdf, 10 * sizeof(uint16_t));
  p += 10;
  memcpy(p, c->fp_cdf, 5 * sizeof(uint16_t));
  p += 5;
  memcpy(p, c->class0_hp_cdf, 3 * sizeof(uint16_t));
  p += 3;
  memcpy(p, c->hp_cdf, 3 * sizeof(uint16_t));
  p += 3;
  return p;
}

int shim_dump_default_kf_fc(int base_qindex, uint16_t *out) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(AV1_COMMON));
  FRAME_CONTEXT *fc = (FRAME_CONTEXT *)calloc(1, sizeof(FRAME_CONTEXT));
  FRAME_CONTEXT *dfc = (FRAME_CONTEXT *)calloc(1, sizeof(FRAME_CONTEXT));
  RefCntBuffer *rcb = (RefCntBuffer *)calloc(1, sizeof(RefCntBuffer));
  if (!cm || !fc || !dfc || !rcb) return 1;
  cm->fc = fc;
  cm->default_frame_context = dfc;
  cm->cur_frame = rcb; /* seg_map NULL -> the memset arm is skipped */
  cm->quant_params.base_qindex = base_qindex;
  /* tiles.large_scale = 0 (calloc) -> av1_setup_frame_contexts copies only. */
  av1_setup_past_independence(cm);

  uint16_t *p = out;
#define CP(field)                                    \
  do {                                               \
    memcpy(p, (field), sizeof(field));               \
    p += sizeof(field) / sizeof(uint16_t);           \
  } while (0)

  CP(fc->kf_y_cdf);              /* [5][5][14]  350 */
  CP(fc->uv_mode_cdf);           /* [2][13][15] 390 */
  CP(fc->angle_delta_cdf);       /* [8][8]       64 */
  CP(fc->skip_txfm_cdfs);        /* [3][3]        9 */
  CP(fc->seg.spatial_pred_seg_cdf); /* [3][9]     27 */
  CP(fc->partition_cdf);         /* [20][11]    220 */
  CP(fc->palette_y_mode_cdf);    /* [7][3][3]    63 */
  CP(fc->palette_uv_mode_cdf);   /* [2][3]        6 */
  CP(fc->palette_y_size_cdf);    /* [7][8]       56 */
  CP(fc->palette_uv_size_cdf);   /* [7][8]       56 */
  CP(fc->filter_intra_cdfs);     /* [22][3]      66 */
  CP(fc->filter_intra_mode_cdf); /* [6]           6 */
  CP(fc->cfl_sign_cdf);          /* [9]           9 */
  CP(fc->cfl_alpha_cdf);         /* [6][17]     102 */
  CP(fc->delta_q_cdf);           /* [5]           5 */
  CP(fc->delta_lf_multi_cdf);    /* [4][5]       20 */
  CP(fc->delta_lf_cdf);          /* [5]           5 */
  CP(fc->intrabc_cdf);           /* [3]           3 */
  memcpy(p, fc->ndvc.joints_cdf, 5 * sizeof(uint16_t)); /* 5 */
  p += 5;
  p = dump_nmv_comp(&fc->ndvc.comps[0], p); /* 69 */
  p = dump_nmv_comp(&fc->ndvc.comps[1], p); /* 69 */
  CP(fc->tx_size_cdf);           /* [4][3][4]    48 */
  /* intra_ext_tx_cdf[set][EXT_TX_SIZES][INTRA_MODES][CDF_SIZE(16)]: slice the
   * leading nsym+1 slots (set 1 = 7-sym -> 8, set 2 = 5-sym -> 6). */
  for (int sz = 0; sz < EXT_TX_SIZES; sz++)
    for (int m = 0; m < INTRA_MODES; m++) {
      memcpy(p, fc->intra_ext_tx_cdf[1][sz][m], 8 * sizeof(uint16_t));
      p += 8; /* 416 total */
    }
  for (int sz = 0; sz < EXT_TX_SIZES; sz++)
    for (int m = 0; m < INTRA_MODES; m++) {
      memcpy(p, fc->intra_ext_tx_cdf[2][sz][m], 6 * sizeof(uint16_t));
      p += 6; /* 312 total */
    }

  /* Coefficient arena (aom-txb CdfArena layout, 4045 u16). */
  uint16_t *cf = p;
  memcpy(cf + 0, fc->txb_skip_cdf, sizeof(fc->txb_skip_cdf));         /* 195 */
  memcpy(cf + 195, fc->eob_flag_cdf16, sizeof(fc->eob_flag_cdf16));   /* 24 */
  memcpy(cf + 219, fc->eob_flag_cdf32, sizeof(fc->eob_flag_cdf32));   /* 28 */
  memcpy(cf + 247, fc->eob_flag_cdf64, sizeof(fc->eob_flag_cdf64));   /* 32 */
  memcpy(cf + 279, fc->eob_flag_cdf128, sizeof(fc->eob_flag_cdf128)); /* 36 */
  memcpy(cf + 315, fc->eob_flag_cdf256, sizeof(fc->eob_flag_cdf256)); /* 40 */
  memcpy(cf + 355, fc->eob_flag_cdf512, sizeof(fc->eob_flag_cdf512)); /* 44 */
  memcpy(cf + 399, fc->eob_flag_cdf1024, sizeof(fc->eob_flag_cdf1024)); /* 48 */
  memcpy(cf + 447, fc->eob_extra_cdf, sizeof(fc->eob_extra_cdf));     /* 270 */
  memcpy(cf + 717, fc->coeff_base_eob_cdf, sizeof(fc->coeff_base_eob_cdf)); /* 160 */
  memcpy(cf + 877, fc->coeff_base_cdf, sizeof(fc->coeff_base_cdf));   /* 2100 */
  memcpy(cf + 2977, fc->coeff_br_cdf, sizeof(fc->coeff_br_cdf));      /* 1050 */
  memcpy(cf + 4027, fc->dc_sign_cdf, sizeof(fc->dc_sign_cdf));        /* 18 */
  p = cf + 4045;
#undef CP

  int rc = (p - out) == DUMP_KF_FC_LEN ? 0 : 2;
  free(rcb);
  free(dfc);
  free(fc);
  free(cm);
  return rc;
}

/* ------------------------------------------------------------------ */
/* 3. Real codec-API encode / decode                                   */
/* ------------------------------------------------------------------ */

#include "aom/aom_decoder.h"
#include "aom/aom_encoder.h"
#include "aom/aomcx.h"
#include "aom/aomdx.h"

/* Encode one KEY frame through the REAL aom_codec_av1_cx public API — the
 * same encoder+path the aomenc CLI drives, with the CLI-flag-equivalent
 * controls: --cpu-used --end-usage=q --cq-level --enable-cdef=0
 * --enable-restoration=0 --sb-size=64 --deltaq-mode=0 --aq-mode=0
 * --enable-palette=0 --enable-intrabc=0.
 * Planes are u16 at every depth (bd=8 downshifts into the 8-bit image).
 * Returns the bitstream length, or a negative error code. */
long shim_encode_av1_kf(const uint16_t *y, const uint16_t *u,
                        const uint16_t *v, int w, int h, int bd, int mono,
                        int ss_x, int ss_y, int cq_level, int cpu_used,
                        int enable_cdef, uint8_t *out, size_t out_cap) {
  aom_codec_iface_t *iface = aom_codec_av1_cx();
  aom_codec_enc_cfg_t cfg;
  if (aom_codec_enc_config_default(iface, &cfg, AOM_USAGE_GOOD_QUALITY))
    return -1;
  cfg.g_w = w;
  cfg.g_h = h;
  cfg.g_limit = 1;
  cfg.g_lag_in_frames = 0;
  cfg.g_threads = 1;
  cfg.g_pass = AOM_RC_ONE_PASS;
  cfg.rc_end_usage = AOM_Q;
  cfg.monochrome = mono;
  cfg.g_input_bit_depth = bd;
  if (bd == 8) {
    cfg.g_bit_depth = AOM_BITS_8;
    cfg.g_profile = (ss_x == 0 && ss_y == 0) ? 1 : 0;
  } else if (bd == 10) {
    cfg.g_bit_depth = AOM_BITS_10;
    cfg.g_profile = (ss_x == 0 && ss_y == 0) ? 1 : 0;
  } else {
    cfg.g_bit_depth = AOM_BITS_12;
    cfg.g_profile = 2;
  }
  if (!mono && ss_x == 1 && ss_y == 0) cfg.g_profile = 2; /* 4:2:2 */

  aom_codec_ctx_t ctx;
  aom_codec_flags_t flags = bd > 8 ? AOM_CODEC_USE_HIGHBITDEPTH : 0;
  if (aom_codec_enc_init(&ctx, iface, &cfg, flags)) return -2;

#define TRYCTRL(id, val)                          \
  do {                                            \
    if (aom_codec_control(&ctx, (id), (val))) {   \
      aom_codec_destroy(&ctx);                    \
      return -3;                                  \
    }                                             \
  } while (0)
  TRYCTRL(AOME_SET_CPUUSED, cpu_used);
  TRYCTRL(AOME_SET_CQ_LEVEL, cq_level);
  TRYCTRL(AV1E_SET_ENABLE_CDEF, enable_cdef);
  TRYCTRL(AV1E_SET_ENABLE_RESTORATION, 0);
  TRYCTRL(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_64X64);
  TRYCTRL(AV1E_SET_DELTAQ_MODE, 0);
  TRYCTRL(AV1E_SET_AQ_MODE, 0);
  TRYCTRL(AV1E_SET_ENABLE_PALETTE, 0);
  TRYCTRL(AV1E_SET_ENABLE_INTRABC, 0);
#undef TRYCTRL

  aom_img_fmt_t fmt;
  if (mono || (ss_x == 1 && ss_y == 1))
    fmt = AOM_IMG_FMT_I420;
  else if (ss_x == 1)
    fmt = AOM_IMG_FMT_I422;
  else
    fmt = AOM_IMG_FMT_I444;
  if (bd > 8) fmt |= AOM_IMG_FMT_HIGHBITDEPTH;
  aom_image_t *img = aom_img_alloc(NULL, fmt, w, h, 32);
  if (!img) {
    aom_codec_destroy(&ctx);
    return -4;
  }
  img->monochrome = mono;
  img->bit_depth = bd;
  const int cw = mono ? 0 : (w + ss_x) >> ss_x;
  const int ch = mono ? 0 : (h + ss_y) >> ss_y;
  for (int plane = 0; plane < (mono ? 1 : 3); plane++) {
    const uint16_t *src = plane == 0 ? y : (plane == 1 ? u : v);
    const int pw = plane == 0 ? w : cw;
    const int ph = plane == 0 ? h : ch;
    for (int r = 0; r < ph; r++) {
      if (bd > 8) {
        uint16_t *row =
            (uint16_t *)(img->planes[plane] + (size_t)r * img->stride[plane]);
        for (int c = 0; c < pw; c++) row[c] = src[(size_t)r * pw + c];
      } else {
        uint8_t *row = img->planes[plane] + (size_t)r * img->stride[plane];
        for (int c = 0; c < pw; c++) row[c] = (uint8_t)src[(size_t)r * pw + c];
      }
    }
  }
  /* Chroma planes of a monochrome image are left as aom_img_alloc gave them
   * (the encoder ignores them when cfg.monochrome). */

  long total = 0;
  int rc = 0;
  for (int pass = 0; pass < 2 && rc == 0; pass++) {
    if (aom_codec_encode(&ctx, pass == 0 ? img : NULL, 0, 1,
                         pass == 0 ? AOM_EFLAG_FORCE_KF : 0)) {
      rc = -5;
      break;
    }
    aom_codec_iter_t iter = NULL;
    const aom_codec_cx_pkt_t *pkt;
    while ((pkt = aom_codec_get_cx_data(&ctx, &iter)) != NULL) {
      if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) continue;
      if ((size_t)total + pkt->data.frame.sz > out_cap) {
        rc = -6;
        break;
      }
      memcpy(out + total, pkt->data.frame.buf, pkt->data.frame.sz);
      total += (long)pkt->data.frame.sz;
    }
  }
  aom_img_free(img);
  aom_codec_destroy(&ctx);
  return rc ? rc : total;
}

/* Decode AV1 bytes through the REAL aom_codec_av1_dx public API and copy the
 * (cropped) planes out as u16 row-major with tight strides. info_out:
 *   [0]=bit_depth [1]=monochrome [2]=ss_x [3]=ss_y [4]=d_w [5]=d_h.
 * The y/u/v buffers must hold w*h resp. cw*ch samples for the EXPECTED dims
 * (mismatch errors out). Returns 0 on success. */
int shim_decode_av1_kf(const uint8_t *data, size_t len, int expect_w,
                       int expect_h, uint16_t *y, uint16_t *u, uint16_t *v,
                       int32_t *info_out) {
  aom_codec_ctx_t ctx;
  if (aom_codec_dec_init(&ctx, aom_codec_av1_dx(), NULL, 0)) return 1;
  if (aom_codec_decode(&ctx, data, len, NULL)) {
    aom_codec_destroy(&ctx);
    return 2;
  }
  aom_codec_iter_t iter = NULL;
  aom_image_t *img = aom_codec_get_frame(&ctx, &iter);
  if (!img) {
    aom_codec_destroy(&ctx);
    return 3;
  }
  if ((int)img->d_w != expect_w || (int)img->d_h != expect_h) {
    aom_codec_destroy(&ctx);
    return 4;
  }
  const int mono = img->monochrome;
  const int ss_x = img->x_chroma_shift, ss_y = img->y_chroma_shift;
  const int highbd = (img->fmt & AOM_IMG_FMT_HIGHBITDEPTH) != 0;
  info_out[0] = (int32_t)img->bit_depth;
  info_out[1] = mono;
  info_out[2] = ss_x;
  info_out[3] = ss_y;
  info_out[4] = (int32_t)img->d_w;
  info_out[5] = (int32_t)img->d_h;
  const int cw = mono ? 0 : ((int)img->d_w + ss_x) >> ss_x;
  const int ch = mono ? 0 : ((int)img->d_h + ss_y) >> ss_y;
  for (int plane = 0; plane < (mono ? 1 : 3); plane++) {
    uint16_t *dst = plane == 0 ? y : (plane == 1 ? u : v);
    const int pw = plane == 0 ? (int)img->d_w : cw;
    const int ph = plane == 0 ? (int)img->d_h : ch;
    for (int r = 0; r < ph; r++) {
      const uint8_t *row = img->planes[plane] + (size_t)r * img->stride[plane];
      if (highbd) {
        const uint16_t *row16 = (const uint16_t *)row;
        for (int c = 0; c < pw; c++) dst[(size_t)r * pw + c] = row16[c];
      } else {
        for (int c = 0; c < pw; c++) dst[(size_t)r * pw + c] = row[c];
      }
    }
  }
  /* Exactly one frame expected. */
  int extra = aom_codec_get_frame(&ctx, &iter) != NULL;
  aom_codec_destroy(&ctx);
  return extra ? 5 : 0;
}

/* ------------------------------------------------------------------ */
/* 4. Loop-filter application oracles                                  */
/*                                                                     */
/* Facades over the REAL exported av1_loop_filter_frame_init and       */
/* av1_filter_block_plane_vert / _horz (av1/common/av1_loopfilter.c),  */
/* plus the real static-inline check_planes_to_loop_filter /           */
/* skip_loop_filter_plane from av1/common/thread_common.h, driven in   */
/* the exact loop_filter_rows order (thread_common.c:467, the          */
/* single-worker lpf_opt_level==0 path the decoder takes). No filter   */
/* logic is transcribed: a synthetic AV1_COMMON + per-cell MB_MODE_INFO*/
/* grid is built from flat arrays and the real functions do the work.  */
/*                                                                     */
/* Per-cell flattening: every mi cell gets its OWN MB_MODE_INFO whose  */
/* tx_size AND all inter_tx_size[] entries hold the cell's tx value,   */
/* so get_transform_size resolves to it on every branch (intra, inter  */
/* vartx, skip). bd==8 runs the real LOWBD path (u8 planes,            */
/* use_highbitdepth=0 — what the production decoder does for 8-bit     */
/* streams); bd>8 the real highbd path via CONVERT_TO_BYTEPTR.         */
/* ------------------------------------------------------------------ */

#include "av1/common/av1_loopfilter.h"
#include "av1/common/thread_common.h"

void shim_lf_frame_init_tables(
    const int32_t *filter_level /*[4]: y_v, y_h, u, v*/, int sharpness,
    int mode_ref_delta_enabled, const int8_t *ref_deltas /*[8]*/,
    const int8_t *mode_deltas /*[2]*/, int seg_enabled,
    const int32_t *seg_active /*[8*4] LF features Y_V,Y_H,U,V*/,
    const int32_t *seg_data /*[8*4]*/, int plane_start, int plane_end,
    uint8_t *lfthr_out /*[64*3]: mblim,lim,hev_thr per level*/,
    uint8_t *lvl_out /*[3*8*2*8*2]*/) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  cm->lf.filter_level[0] = filter_level[0];
  cm->lf.filter_level[1] = filter_level[1];
  cm->lf.filter_level_u = filter_level[2];
  cm->lf.filter_level_v = filter_level[3];
  cm->lf.sharpness_level = sharpness;
  cm->lf.mode_ref_delta_enabled = (uint8_t)mode_ref_delta_enabled;
  memcpy(cm->lf.ref_deltas, ref_deltas, REF_FRAMES);
  memcpy(cm->lf.mode_deltas, mode_deltas, MAX_MODE_LF_DELTAS);
  cm->seg.enabled = (uint8_t)seg_enabled;
  for (int s = 0; s < MAX_SEGMENTS; s++) {
    for (int f = 0; f < 4; f++) { /* SEG_LVL_ALT_LF_Y_V..SEG_LVL_ALT_LF_V */
      if (seg_active[s * 4 + f]) {
        cm->seg.feature_mask[s] |= 1 << (SEG_LVL_ALT_LF_Y_V + f);
        cm->seg.feature_data[s][SEG_LVL_ALT_LF_Y_V + f] =
            (int16_t)seg_data[s * 4 + f];
      }
    }
  }
  /* hev_thr comes from av1_loop_filter_init (decoder does it at alloc). */
  av1_loop_filter_init(cm);
  av1_loop_filter_frame_init(cm, plane_start, plane_end);
  for (int l = 0; l <= MAX_LOOP_FILTER; l++) {
    lfthr_out[l * 3 + 0] = cm->lf_info.lfthr[l].mblim[0];
    lfthr_out[l * 3 + 1] = cm->lf_info.lfthr[l].lim[0];
    lfthr_out[l * 3 + 2] = cm->lf_info.lfthr[l].hev_thr[0];
  }
  memcpy(lvl_out, cm->lf_info.lvl, sizeof(cm->lf_info.lvl));
  free(cm);
}

int shim_lf_filter_frame(
    uint16_t *y, int y_stride, uint16_t *u, uint16_t *v, int uv_stride,
    int crop_w, int crop_h, int ss_x, int ss_y, int bd, int mi_rows,
    int mi_cols, int grid_stride, const int32_t *g_bsize,
    const int32_t *g_txsize, const int32_t *g_seg, const int32_t *g_ref0,
    const int32_t *g_mode, const int32_t *g_skip, const int32_t *g_intrabc,
    const int8_t *g_dlf_base, const int8_t *g_dlf /*[4] per cell*/,
    const int32_t *filter_level, int sharpness, int mode_ref_delta_enabled,
    const int8_t *ref_deltas, const int8_t *mode_deltas, int delta_lf_present,
    int delta_lf_multi, const int32_t *lossless /*[8]*/, int seg_enabled,
    const int32_t *seg_active, const int32_t *seg_data, int plane_start,
    int plane_end) {
  const int ncells = mi_rows * grid_stride;
  MB_MODE_INFO *cells = (MB_MODE_INFO *)calloc(ncells, sizeof(*cells));
  MB_MODE_INFO **grid = (MB_MODE_INFO **)calloc(ncells, sizeof(*grid));
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  if (!cells || !grid || !cm || !seq || !xd) return -1;

  for (int r = 0; r < mi_rows; r++) {
    for (int c = 0; c < mi_cols; c++) {
      const int i = r * grid_stride + c;
      MB_MODE_INFO *mi = &cells[i];
      mi->bsize = (BLOCK_SIZE)g_bsize[i];
      mi->tx_size = (TX_SIZE)g_txsize[i];
      for (int k = 0; k < INTER_TX_SIZE_BUF_LEN; k++)
        mi->inter_tx_size[k] = (TX_SIZE)g_txsize[i];
      mi->segment_id = (uint8_t)g_seg[i];
      mi->ref_frame[0] = (MV_REFERENCE_FRAME)g_ref0[i];
      mi->ref_frame[1] = NONE_FRAME;
      mi->mode = (PREDICTION_MODE)g_mode[i];
      mi->skip_txfm = (uint8_t)g_skip[i];
      mi->use_intrabc = (uint8_t)g_intrabc[i];
      mi->delta_lf_from_base = g_dlf_base[i];
      for (int k = 0; k < FRAME_LF_COUNT; k++)
        mi->delta_lf[k] = g_dlf[i * FRAME_LF_COUNT + k];
      grid[i] = mi;
    }
  }

  cm->mi_params.mi_grid_base = grid;
  cm->mi_params.mi_stride = grid_stride;
  cm->mi_params.mi_rows = mi_rows;
  cm->mi_params.mi_cols = mi_cols;
  cm->lf.filter_level[0] = filter_level[0];
  cm->lf.filter_level[1] = filter_level[1];
  cm->lf.filter_level_u = filter_level[2];
  cm->lf.filter_level_v = filter_level[3];
  cm->lf.sharpness_level = sharpness;
  cm->lf.mode_ref_delta_enabled = (uint8_t)mode_ref_delta_enabled;
  memcpy(cm->lf.ref_deltas, ref_deltas, REF_FRAMES);
  memcpy(cm->lf.mode_deltas, mode_deltas, MAX_MODE_LF_DELTAS);
  cm->delta_q_info.delta_lf_present_flag = delta_lf_present;
  cm->delta_q_info.delta_lf_multi = delta_lf_multi;
  cm->seg.enabled = (uint8_t)seg_enabled;
  for (int s = 0; s < MAX_SEGMENTS; s++) {
    for (int f = 0; f < 4; f++) {
      if (seg_active[s * 4 + f]) {
        cm->seg.feature_mask[s] |= 1 << (SEG_LVL_ALT_LF_Y_V + f);
        cm->seg.feature_data[s][SEG_LVL_ALT_LF_Y_V + f] =
            (int16_t)seg_data[s * 4 + f];
      }
    }
  }
  seq->bit_depth = (aom_bit_depth_t)bd;
  seq->use_highbitdepth = bd > 8;
  cm->seq_params = seq;
  for (int s = 0; s < MAX_SEGMENTS; s++) xd->lossless[s] = lossless[s];

  /* Plane buffers: logical mi-aligned area. bd==8 -> real lowbd path on u8
   * copies; bd>8 -> highbd path on the u16 buffers via CONVERT_TO_BYTEPTR. */
  const int y_rows = mi_rows * MI_SIZE;
  const int uv_rows = y_rows >> ss_y;
  const long uv_len = (long)uv_stride * uv_rows; /* 0 for monochrome */
  uint8_t *y8 = NULL, *u8b = NULL, *v8b = NULL;
  if (bd == 8) {
    y8 = (uint8_t *)malloc((size_t)y_stride * y_rows);
    u8b = (uint8_t *)malloc(uv_len ? (size_t)uv_len : 1);
    v8b = (uint8_t *)malloc(uv_len ? (size_t)uv_len : 1);
    if (!y8 || !u8b || !v8b) return -2;
    for (long i = 0; i < (long)y_stride * y_rows; i++) y8[i] = (uint8_t)y[i];
    for (long i = 0; i < uv_len; i++) {
      u8b[i] = (uint8_t)u[i];
      v8b[i] = (uint8_t)v[i];
    }
  }

  int planes_to_lf[MAX_MB_PLANE];
  if (check_planes_to_loop_filter(&cm->lf, planes_to_lf, plane_start,
                                  plane_end)) {
    av1_loop_filter_init(cm);
    av1_loop_filter_frame_init(cm, plane_start, plane_end);

    struct macroblockd_plane pd[MAX_MB_PLANE];
    memset(pd, 0, sizeof(pd));
    for (int mi_row = 0; mi_row < mi_rows; mi_row += MAX_MIB_SIZE) {
      for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
        if (skip_loop_filter_plane(planes_to_lf, plane, 0)) continue;
        const int sx = plane ? ss_x : 0, sy = plane ? ss_y : 0;
        for (int dir = 0; dir < 2; dir++) {
          for (int mi_col = 0; mi_col < mi_cols; mi_col += MAX_MIB_SIZE) {
            /* av1_setup_dst_planes for this plane+SB position. */
            struct macroblockd_plane *p = &pd[plane];
            p->subsampling_x = sx;
            p->subsampling_y = sy;
            const int px = (MI_SIZE * mi_col) >> sx;
            const int py = (MI_SIZE * mi_row) >> sy;
            const int stride = plane ? uv_stride : y_stride;
            p->dst.stride = stride;
            p->dst.width = plane ? (crop_w + ss_x) >> ss_x : crop_w;
            p->dst.height = plane ? (crop_h + ss_y) >> ss_y : crop_h;
            if (bd == 8) {
              uint8_t *base = plane == 0 ? y8 : (plane == 1 ? u8b : v8b);
              p->dst.buf = base + (ptrdiff_t)py * stride + px;
            } else {
              uint16_t *base = plane == 0 ? y : (plane == 1 ? u : v);
              p->dst.buf = CONVERT_TO_BYTEPTR(base) + (ptrdiff_t)py * stride + px;
            }
            if (dir == 0)
              av1_filter_block_plane_vert(cm, xd, plane, p, mi_row, mi_col);
            else
              av1_filter_block_plane_horz(cm, xd, plane, p, mi_row, mi_col);
          }
        }
      }
    }
  }

  if (bd == 8) {
    for (long i = 0; i < (long)y_stride * y_rows; i++) y[i] = y8[i];
    for (long i = 0; i < uv_len; i++) {
      u[i] = u8b[i];
      v[i] = v8b[i];
    }
    free(y8);
    free(u8b);
    free(v8b);
  }
  free(xd);
  free(seq);
  free(cm);
  free(grid);
  free(cells);
  return 0;
}
