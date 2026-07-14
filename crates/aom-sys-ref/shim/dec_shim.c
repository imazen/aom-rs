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
 *     CdfArena region layout). Total DUMP_KF_FC_LEN = 7061 u16.
 *
 *  3. shim_encode_av1_kf / shim_encode_av1_kf_sb128 / shim_encode_av1_kf_tiles
 *     / shim_decode_av1_kf — the REAL public codec API (aom_codec_av1_cx /
 *     aom_codec_av1_dx): produce a production KEY-frame bitstream in-process
 *     (the same library+path the aomenc CLI drives, at --sb-size=64 or =128,
 *     optionally --tile-columns/--tile-rows) and the gold C-decoder pixel
 *     oracle for arbitrary AV1 bytes (SB size / tile grid are stream facts
 *     the real decoder reads itself — no sb-size- or tile-specific decode
 *     entry point is needed).
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

/* av1_get_spatial_seg_pred (pred_common.h) facade: a 2x2-mi frame whose
 * segment-id map holds the three neighbour cells of the block at (1,1) —
 * [0]=up-left, [1]=up, [2]=left — with the given availability
 * (skip_over4x4 = 0, the decoder's step size). Returns pred | (cdf_num << 8). */
int shim_spatial_seg_pred(int up_available, int left_available, int ul, int u,
                          int l) {
  AV1_COMMON cm;
  memset(&cm, 0, sizeof(cm));
  RefCntBuffer buf;
  memset(&buf, 0, sizeof(buf));
  uint8_t map[4];
  map[0] = (uint8_t)ul;
  map[1] = (uint8_t)u;
  map[2] = (uint8_t)l;
  map[3] = 0;
  cm.mi_params.mi_cols = 2;
  cm.mi_params.mi_rows = 2;
  buf.seg_map = map;
  cm.cur_frame = &buf;
  MACROBLOCKD xd;
  memset(&xd, 0, sizeof(xd));
  xd.mi_row = 1;
  xd.mi_col = 1;
  xd.up_available = up_available;
  xd.left_available = left_available;
  int cdf_num = -1;
  const uint8_t pred = av1_get_spatial_seg_pred(&cm, &xd, &cdf_num, 0);
  return (int)pred | (cdf_num << 8);
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
#define DUMP_KF_FC_LEN 7061

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
  CP(fc->palette_y_color_index_cdf);  /* [7][5][9]  315 */
  CP(fc->palette_uv_color_index_cdf); /* [7][5][9]  315 */
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
  CP(fc->switchable_restore_cdf); /* [4]        4 */
  CP(fc->wiener_restore_cdf);     /* [3]        3 */
  CP(fc->sgrproj_restore_cdf);    /* [3]        3 */

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

/* One encoder run over the single prepared image: init a fresh context on
 * cfg (whose g_pass the caller set), apply the CLI-equivalent controls,
 * encode img + flush, and collect either the FRAME packets (into out) or the
 * STATS packets (into out — the pass-1 firstpass stats blob a LAST_PASS run
 * consumes via rc_twopass_stats_in, exactly the aomenc 2-pass dance).
 * Returns bytes collected or a negative error code. */
static long encode_kf_pass(aom_codec_iface_t *iface, aom_codec_enc_cfg_t *cfg,
                           int bd, int cq_level, int cpu_used, int enable_cdef,
                           int enable_restoration, int aq_mode,
                           int sb_size_128, int tile_columns_log2,
                           int tile_rows_log2, int enable_palette,
                           int enable_intrabc, aom_image_t *img,
                           int collect_stats, uint8_t *out, size_t out_cap) {
  aom_codec_ctx_t ctx;
  aom_codec_flags_t flags = bd > 8 ? AOM_CODEC_USE_HIGHBITDEPTH : 0;
  if (aom_codec_enc_init(&ctx, iface, cfg, flags)) return -2;

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
  TRYCTRL(AV1E_SET_ENABLE_RESTORATION, enable_restoration);
  TRYCTRL(AV1E_SET_SUPERBLOCK_SIZE, sb_size_128 ? AOM_SUPERBLOCK_SIZE_128X128
                                                : AOM_SUPERBLOCK_SIZE_64X64);
  /* --tile-columns=<log2> / --tile-rows=<log2> (av1_cx_iface.c): BOTH are the
   * LOG2 tile count (a decoder-track multi-tile addition — 0 (default) is
   * the pre-existing single-tile behavior, unchanged for the two callers
   * below that pass 0,0). */
  TRYCTRL(AV1E_SET_TILE_COLUMNS, tile_columns_log2);
  TRYCTRL(AV1E_SET_TILE_ROWS, tile_rows_log2);
  TRYCTRL(AV1E_SET_DELTAQ_MODE, 0);
  /* aq_mode: 0 = off, 1 = VARIANCE_AQ, 2 = COMPLEXITY_VARIANCE_AQ, 3 =
   * CYCLIC_REFRESH_AQ. 1/2 enable SEGMENTATION on intra frames — 8 segments
   * of SEG_LVL_ALT_Q via av1_vaq_frame_setup / av1_setup_in_frame_q_adj —
   * but ONLY inside encode_with_recode_loop: a ONE-pass encode takes
   * encode_without_recode (speed_features.c "No recode for 1 pass") and
   * never runs the aq setup, so segmented KEY streams REQUIRE two_pass. */
  TRYCTRL(AV1E_SET_AQ_MODE, aq_mode);
  TRYCTRL(AV1E_SET_ENABLE_PALETTE, enable_palette);
  TRYCTRL(AV1E_SET_ENABLE_INTRABC, enable_intrabc);
#undef TRYCTRL

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
      const void *buf;
      size_t sz;
      if (collect_stats && pkt->kind == AOM_CODEC_STATS_PKT) {
        buf = pkt->data.twopass_stats.buf;
        sz = pkt->data.twopass_stats.sz;
      } else if (!collect_stats && pkt->kind == AOM_CODEC_CX_FRAME_PKT) {
        buf = pkt->data.frame.buf;
        sz = pkt->data.frame.sz;
      } else {
        continue;
      }
      if ((size_t)total + sz > out_cap) {
        rc = -6;
        break;
      }
      memcpy(out + total, buf, sz);
      total += (long)sz;
    }
  }
  aom_codec_destroy(&ctx);
  return rc ? rc : total;
}

/* Shared implementation of shim_encode_av1_kf / shim_encode_av1_kf_sb128 /
 * shim_encode_av1_kf_tiles / shim_encode_av1_kf_screen_content (see their doc
 * comments below) — the REAL aom_codec_av1_cx public API, the same
 * encoder+path the aomenc CLI drives, with the CLI-flag-equivalent controls:
 * --cpu-used --end-usage=q --cq-level --enable-cdef --enable-restoration
 * --sb-size={64,128} --tile-columns=<log2> --tile-rows=<log2>
 * --deltaq-mode=0 --aq-mode=<aq_mode> --enable-palette=<enable_palette>
 * --enable-intrabc=<enable_intrabc> --passes=<1|2>. two_pass runs the full
 * first-pass-stats + last-pass sequence (rc_twopass_stats_in) — required for
 * aq_mode 1/2 to actually segment (see encode_kf_pass). Planes are u16 at
 * every depth (bd=8 downshifts into the 8-bit image). Returns the bitstream
 * length, or a negative error code. */
static long encode_av1_kf_impl(const uint16_t *y, const uint16_t *u,
                               const uint16_t *v, int w, int h, int bd,
                               int mono, int ss_x, int ss_y, int cq_level,
                               int cpu_used, int enable_cdef,
                               int enable_restoration, int usage, int aq_mode,
                               int two_pass, int sb_size_128,
                               int tile_columns_log2, int tile_rows_log2,
                               int enable_palette, int enable_intrabc,
                               uint8_t *out, size_t out_cap) {
  aom_codec_iface_t *iface = aom_codec_av1_cx();
  aom_codec_enc_cfg_t cfg;
  /* usage: AOM_USAGE_GOOD_QUALITY (0) or AOM_USAGE_ALL_INTRA (2 — the
   * zenavif/avifenc still-image mode; different speed-feature + default
   * arms in the encoder, same decode-side syntax). */
  if (aom_codec_enc_config_default(iface, &cfg, (unsigned int)usage))
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

  aom_img_fmt_t fmt;
  if (mono || (ss_x == 1 && ss_y == 1))
    fmt = AOM_IMG_FMT_I420;
  else if (ss_x == 1)
    fmt = AOM_IMG_FMT_I422;
  else
    fmt = AOM_IMG_FMT_I444;
  if (bd > 8) fmt |= AOM_IMG_FMT_HIGHBITDEPTH;
  aom_image_t *img = aom_img_alloc(NULL, fmt, w, h, 32);
  if (!img) return -4;
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

  long total;
  if (two_pass) {
    /* Pass 1: firstpass stats (per-frame packets + the flush-time totals
     * packet, concatenated — the aomenc --passes=2 sequence). One frame's
     * stats are a few hundred bytes; 64 KiB is generous. */
    static const size_t STATS_CAP = 65536;
    uint8_t *stats = (uint8_t *)malloc(STATS_CAP);
    if (!stats) {
      aom_img_free(img);
      return -4;
    }
    cfg.g_pass = AOM_RC_FIRST_PASS;
    long stats_len = encode_kf_pass(iface, &cfg, bd, cq_level, cpu_used,
                                    enable_cdef, enable_restoration, aq_mode,
                                    sb_size_128, tile_columns_log2,
                                    tile_rows_log2, enable_palette,
                                    enable_intrabc, img, 1, stats, STATS_CAP);
    if (stats_len <= 0) {
      free(stats);
      aom_img_free(img);
      return stats_len == 0 ? -7 : stats_len;
    }
    cfg.g_pass = AOM_RC_LAST_PASS;
    cfg.rc_twopass_stats_in.buf = stats;
    cfg.rc_twopass_stats_in.sz = (size_t)stats_len;
    total = encode_kf_pass(iface, &cfg, bd, cq_level, cpu_used, enable_cdef,
                           enable_restoration, aq_mode, sb_size_128,
                           tile_columns_log2, tile_rows_log2, enable_palette,
                           enable_intrabc, img, 0, out, out_cap);
    free(stats);
  } else {
    total = encode_kf_pass(iface, &cfg, bd, cq_level, cpu_used, enable_cdef,
                           enable_restoration, aq_mode, sb_size_128,
                           tile_columns_log2, tile_rows_log2, enable_palette,
                           enable_intrabc, img, 0, out, out_cap);
  }
  aom_img_free(img);
  return total;
}

/* Encode one KEY frame with --sb-size=64 (AOM_SUPERBLOCK_SIZE_64X64) — see
 * encode_av1_kf_impl's doc comment for the full control list. Unchanged
 * signature/behavior (decoder-track SB128 work added encode_av1_kf_impl +
 * shim_encode_av1_kf_sb128 alongside this, append-only). */
long shim_encode_av1_kf(const uint16_t *y, const uint16_t *u,
                        const uint16_t *v, int w, int h, int bd, int mono,
                        int ss_x, int ss_y, int cq_level, int cpu_used,
                        int enable_cdef, int enable_restoration, int usage,
                        int aq_mode, int two_pass, uint8_t *out,
                        size_t out_cap) {
  return encode_av1_kf_impl(y, u, v, w, h, bd, mono, ss_x, ss_y, cq_level,
                            cpu_used, enable_cdef, enable_restoration, usage,
                            aq_mode, two_pass, /*sb_size_128=*/0,
                            /*tile_columns_log2=*/0, /*tile_rows_log2=*/0,
                            /*enable_palette=*/0, /*enable_intrabc=*/0, out,
                            out_cap);
}

/* SB128 variant of shim_encode_av1_kf: same controls plus explicit
 * sb_size_128 (0 = --sb-size=64 / AOM_SUPERBLOCK_SIZE_64X64, nonzero =
 * --sb-size=128 / AOM_SUPERBLOCK_SIZE_128X128, av1_cx_iface.c's
 * ctrl_set_superblock_size). See encode_av1_kf_impl's doc comment for the
 * full control list. */
long shim_encode_av1_kf_sb128(const uint16_t *y, const uint16_t *u,
                              const uint16_t *v, int w, int h, int bd,
                              int mono, int ss_x, int ss_y, int cq_level,
                              int cpu_used, int enable_cdef,
                              int enable_restoration, int usage, int aq_mode,
                              int two_pass, int sb_size_128, uint8_t *out,
                              size_t out_cap) {
  return encode_av1_kf_impl(y, u, v, w, h, bd, mono, ss_x, ss_y, cq_level,
                            cpu_used, enable_cdef, enable_restoration, usage,
                            aq_mode, two_pass, sb_size_128,
                            /*tile_columns_log2=*/0, /*tile_rows_log2=*/0,
                            /*enable_palette=*/0, /*enable_intrabc=*/0, out,
                            out_cap);
}

/* Multi-tile variant of shim_encode_av1_kf: same controls plus explicit
 * sb_size_128 AND tile_columns_log2/tile_rows_log2 (av1_cx_iface.c's
 * AV1E_SET_TILE_COLUMNS/AV1E_SET_TILE_ROWS — the CODED value IS the log2 tile
 * count: 0 = --tile-columns=0 -> 1 column (single-tile default, matching the
 * two callers above), 1 = 2 columns, 2 = 4 columns, ...). See
 * encode_av1_kf_impl's doc comment for the full control list. Decoder-track
 * multi-tile work, append-only (shim_encode_av1_kf / _sb128 above untouched). */
long shim_encode_av1_kf_tiles(const uint16_t *y, const uint16_t *u,
                              const uint16_t *v, int w, int h, int bd,
                              int mono, int ss_x, int ss_y, int cq_level,
                              int cpu_used, int enable_cdef,
                              int enable_restoration, int usage, int aq_mode,
                              int two_pass, int sb_size_128,
                              int tile_columns_log2, int tile_rows_log2,
                              uint8_t *out, size_t out_cap) {
  return encode_av1_kf_impl(y, u, v, w, h, bd, mono, ss_x, ss_y, cq_level,
                            cpu_used, enable_cdef, enable_restoration, usage,
                            aq_mode, two_pass, sb_size_128, tile_columns_log2,
                            tile_rows_log2, /*enable_palette=*/0,
                            /*enable_intrabc=*/0, out, out_cap);
}

/* Screen-content variant of shim_encode_av1_kf: same controls as
 * shim_encode_av1_kf (--sb-size=64, single tile) plus explicit
 * enable_palette/enable_intrabc (AV1E_SET_ENABLE_PALETTE /
 * AV1E_SET_ENABLE_INTRABC — 0/1). Decoder-track palette-gate addition,
 * append-only (shim_encode_av1_kf / _sb128 / _tiles above untouched, still
 * hardcoding both off). See encode_av1_kf_impl's doc comment for the full
 * control list. */
long shim_encode_av1_kf_screen_content(const uint16_t *y, const uint16_t *u,
                                       const uint16_t *v, int w, int h,
                                       int bd, int mono, int ss_x, int ss_y,
                                       int cq_level, int cpu_used,
                                       int enable_cdef,
                                       int enable_restoration, int usage,
                                       int aq_mode, int two_pass,
                                       int enable_palette, int enable_intrabc,
                                       uint8_t *out, size_t out_cap) {
  return encode_av1_kf_impl(y, u, v, w, h, bd, mono, ss_x, ss_y, cq_level,
                            cpu_used, enable_cdef, enable_restoration, usage,
                            aq_mode, two_pass, /*sb_size_128=*/0,
                            /*tile_columns_log2=*/0, /*tile_rows_log2=*/0,
                            enable_palette, enable_intrabc, out, out_cap);
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

/* ------------------------------------------------------------------ */
/* 5. CDEF frame-application oracle                                    */
/*                                                                     */
/* Drives the REAL exported av1_cdef_frame (av1/common/cdef.c) —       */
/* including av1_cdef_init_fb_row, cdef_fb_col, cdef_prepare_fb and    */
/* av1_cdef_filter_fb — over a synthetic AV1_COMMON + per-cell         */
/* MB_MODE_INFO grid + YV12 frame. No CDEF logic is transcribed.       */
/*                                                                     */
/* Per-cell flattening: skip_txfm per mi; cdef_strength stamped on     */
/* EVERY cell of its 64x64 unit (the walk reads only the unit's        */
/* top-left grid pointer, cdef.c:304-308; the decoder stores the       */
/* per-unit literal there). unit_strength -1 exercises the skip arm.   */
/* bd==8 runs the real LOWBD path (u8 planes, use_highbitdepth=0);     */
/* bd>8 the real highbd path via CONVERT_TO_BYTEPTR.                   */
/*                                                                     */
/* Work buffers (linebuf/colbuf/srcbuf) are malloc'd per the           */
/* av1_alloc_cdef_buffers single-worker formulas (alloccommon.c).      */
/* Plane buffers MUST be >= ALIGN_POWER_OF_TWO(mi_cols*4,4) >> ss      */
/* wide: the line-buffer copies read full aligned rows (into the YV12  */
/* border in production).                                              */
/* ------------------------------------------------------------------ */

#include "av1/common/cdef.h"

int shim_cdef_frame(uint16_t *y, int y_stride, uint16_t *u, uint16_t *v,
                    int uv_stride, int mi_rows, int mi_cols, int num_planes,
                    int ss_x, int ss_y, int bd, int damping,
                    const int32_t *strengths /*[8]*/,
                    const int32_t *uv_strengths /*[8]*/,
                    const int32_t *skip /*mi_rows*mi_cols*/,
                    const int32_t *unit_strength /*nvfb*nhfb*/) {
  const int nvfb = (mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
  const int nhfb = (mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
  const int luma_stride = ALIGN_POWER_OF_TWO(mi_cols << MI_SIZE_LOG2, 4);
  const int ncells = mi_rows * mi_cols;
  MB_MODE_INFO *cells = (MB_MODE_INFO *)calloc(ncells, sizeof(*cells));
  MB_MODE_INFO **grid = (MB_MODE_INFO **)calloc(ncells, sizeof(*grid));
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  MACROBLOCKD *xd = (MACROBLOCKD *)calloc(1, sizeof(*xd));
  if (!cells || !grid || !cm || !seq || !xd) return -1;

  for (int r = 0; r < mi_rows; r++) {
    for (int c = 0; c < mi_cols; c++) {
      const int i = r * mi_cols + c;
      cells[i].skip_txfm = (uint8_t)skip[i];
      cells[i].cdef_strength =
          (int8_t)unit_strength[(r / MI_SIZE_64X64) * nhfb + (c / MI_SIZE_64X64)];
      grid[i] = &cells[i];
    }
  }
  cm->mi_params.mi_grid_base = grid;
  cm->mi_params.mi_stride = mi_cols;
  cm->mi_params.mi_rows = mi_rows;
  cm->mi_params.mi_cols = mi_cols;

  CdefInfo *ci = &cm->cdef_info;
  ci->cdef_damping = damping;
  ci->nb_cdef_strengths = 8;
  for (int i = 0; i < 8; i++) {
    ci->cdef_strengths[i] = strengths[i];
    ci->cdef_uv_strengths[i] = uv_strengths[i];
  }
  /* av1_alloc_cdef_buffers single-worker (num_bufs=3) formulas. */
  ci->srcbuf = (uint16_t *)malloc(sizeof(uint16_t) * CDEF_INBUF_SIZE);
  if (!ci->srcbuf) return -2;
  for (int plane = 0; plane < num_planes; plane++) {
    const int shift = plane == 0 ? 0 : ss_x;
    ci->linebuf[plane] = (uint16_t *)malloc(
        sizeof(uint16_t) * 3 * (CDEF_VBORDER << 1) * (luma_stride >> shift));
    const int block_height =
        (CDEF_BLOCKSIZE << (MI_SIZE_LOG2 - shift)) * 2 * CDEF_VBORDER;
    ci->colbuf[plane] =
        (uint16_t *)malloc(sizeof(uint16_t) * block_height * CDEF_HBORDER);
    if (!ci->linebuf[plane] || !ci->colbuf[plane]) return -2;
  }

  seq->monochrome = num_planes == 1;
  seq->subsampling_x = ss_x;
  seq->subsampling_y = ss_y;
  seq->bit_depth = (aom_bit_depth_t)bd;
  seq->use_highbitdepth = bd > 8;
  seq->sb_size = BLOCK_64X64;
  cm->seq_params = seq;

  for (int plane = 0; plane < MAX_MB_PLANE; plane++) {
    xd->plane[plane].subsampling_x = plane == 0 ? 0 : ss_x;
    xd->plane[plane].subsampling_y = plane == 0 ? 0 : ss_y;
  }

  /* Plane buffers: bd==8 -> real lowbd path on u8 copies; bd>8 -> highbd
   * path on the u16 buffers via CONVERT_TO_BYTEPTR. */
  const int y_rows = mi_rows * MI_SIZE;
  const int uv_rows = num_planes > 1 ? (y_rows >> ss_y) : 0;
  const long uv_len = (long)uv_stride * uv_rows;
  uint8_t *y8 = NULL, *u8b = NULL, *v8b = NULL;

  YV12_BUFFER_CONFIG frame;
  memset(&frame, 0, sizeof(frame));
  frame.crop_widths[0] = mi_cols * MI_SIZE;
  frame.crop_heights[0] = y_rows;
  frame.crop_widths[1] = (mi_cols * MI_SIZE) >> ss_x;
  frame.crop_heights[1] = uv_rows;
  frame.strides[0] = y_stride;
  frame.strides[1] = uv_stride;
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
    frame.buffers[0] = y8;
    frame.buffers[1] = u8b;
    frame.buffers[2] = v8b;
  } else {
    frame.buffers[0] = CONVERT_TO_BYTEPTR(y);
    frame.buffers[1] = CONVERT_TO_BYTEPTR(u);
    frame.buffers[2] = CONVERT_TO_BYTEPTR(v);
  }

  av1_cdef_frame(&frame, cm, xd, av1_cdef_init_fb_row);

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
  for (int plane = 0; plane < num_planes; plane++) {
    free(ci->linebuf[plane]);
    free(ci->colbuf[plane]);
  }
  free(ci->srcbuf);
  free(xd);
  free(seq);
  free(cm);
  free(grid);
  free(cells);
  return 0;
}

/* ------------------------------------------------------------------ */
/* 6. Loop-restoration oracles                                         */
/*                                                                     */
/* (a) shim_lr_units_roundtrip — RU-params syntax: writes a sequence   */
/*     of restoration-unit parameter sets with the REAL arithmetic     */
/*     writer primitives (aom_write_symbol over the REAL default LR    */
/*     CDFs + EXPORTED aom_write_primitive_refsubexpfin), mirroring    */
/*     the encoder's loop_restoration_write_sb_coeffs, then reads them */
/*     back with the REAL reader primitives mirroring the decoder's    */
/*     loop_restoration_read_sb_coeffs. Returns the bitstream, the     */
/*     read-back values and the reader's final adapted CDFs.           */
/* (b) shim_wiener_convolve / shim_apply_sgr — the REAL exported       */
/*     kernel _c functions over caller-padded buffers (bd==8 runs the  */
/*     lowbd u8 kernels on converted copies — the production path for  */
/*     8-bit streams; bd>8 the highbd kernels via CONVERT_TO_BYTEPTR). */
/* (c) shim_lr_corners_in_sb — REAL av1_loop_restoration_corners_in_sb */
/*     over a minimal AV1_COMMON.                                      */
/* (d) shim_lr_filter_frame — the REAL whole-frame application:        */
/*     av1_loop_restoration_save_boundary_lines (before/after-CDEF     */
/*     passes over TWO frame states, exactly the decoder's ordering)   */
/*     + av1_loop_restoration_filter_frame over real bordered YV12     */
/*     buffers, real av1_alloc_restoration_struct/_buffers geometry.   */
/*     NEEDS ref_init() (RTCD: wiener/sgr kernels are fn pointers).    */
/* ------------------------------------------------------------------ */

#include "aom_dsp/bitwriter.h"
#include "aom_dsp/bitreader.h"
#include "aom_dsp/binary_codes_writer.h"
#include "aom_dsp/binary_codes_reader.h"
#include "aom_scale/yv12config.h"
#include "av1/common/restoration.h"

/* Unit intent/result packing, 10 i32 per unit:
 * [0]=plane [1]=frame_rtype [2]=unit_rtype
 * [3..6)=wiener v0,v1,v2  [6..9)=h0,h1,h2 -- wait 3 v + 3 h = [3..9)
 * [9]=ep  -- and xqd packed after: see LRU_* below. 12 i32 per unit. */
#define LRU_WORDS 12
#define LRU_PLANE 0
#define LRU_FRTYPE 1
#define LRU_RTYPE 2
#define LRU_V0 3 /* v0 v1 v2 h0 h1 h2 = 3..9 */
#define LRU_EP 9
#define LRU_XQD0 10
#define LRU_XQD1 11

static void lr_fill_wiener(WienerInfo *wi, const int32_t *u) {
  memset(wi, 0, sizeof(*wi));
  for (int d = 0; d < 2; d++) {
    int16_t *f = d == 0 ? wi->vfilter : wi->hfilter;
    const int32_t *t = u + LRU_V0 + 3 * d;
    f[0] = (int16_t)t[0];
    f[1] = (int16_t)t[1];
    f[2] = (int16_t)t[2];
    f[3] = (int16_t)(-2 * (t[0] + t[1] + t[2]));
    f[4] = (int16_t)t[2];
    f[5] = (int16_t)t[1];
    f[6] = (int16_t)t[0];
  }
}

/* Transcribed write_wiener_filter (encoder/bitstream.c) over the REAL
 * exported aom_write_primitive_refsubexpfin. */
static void lr_write_wiener(int wiener_win, const WienerInfo *wi,
                            WienerInfo *ref, aom_writer *wb) {
  if (wiener_win == WIENER_WIN)
    aom_write_primitive_refsubexpfin(
        wb, WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
        WIENER_FILT_TAP0_SUBEXP_K, ref->vfilter[0] - WIENER_FILT_TAP0_MINV,
        wi->vfilter[0] - WIENER_FILT_TAP0_MINV);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
      WIENER_FILT_TAP1_SUBEXP_K, ref->vfilter[1] - WIENER_FILT_TAP1_MINV,
      wi->vfilter[1] - WIENER_FILT_TAP1_MINV);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
      WIENER_FILT_TAP2_SUBEXP_K, ref->vfilter[2] - WIENER_FILT_TAP2_MINV,
      wi->vfilter[2] - WIENER_FILT_TAP2_MINV);
  if (wiener_win == WIENER_WIN)
    aom_write_primitive_refsubexpfin(
        wb, WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
        WIENER_FILT_TAP0_SUBEXP_K, ref->hfilter[0] - WIENER_FILT_TAP0_MINV,
        wi->hfilter[0] - WIENER_FILT_TAP0_MINV);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
      WIENER_FILT_TAP1_SUBEXP_K, ref->hfilter[1] - WIENER_FILT_TAP1_MINV,
      wi->hfilter[1] - WIENER_FILT_TAP1_MINV);
  aom_write_primitive_refsubexpfin(
      wb, WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
      WIENER_FILT_TAP2_SUBEXP_K, ref->hfilter[2] - WIENER_FILT_TAP2_MINV,
      wi->hfilter[2] - WIENER_FILT_TAP2_MINV);
  *ref = *wi;
}

/* Transcribed write_sgrproj_filter (encoder/bitstream.c). */
static void lr_write_sgrproj(const SgrprojInfo *si, SgrprojInfo *ref,
                             aom_writer *wb) {
  aom_write_literal(wb, si->ep, SGRPROJ_PARAMS_BITS);
  const sgr_params_type *params = &av1_sgr_params[si->ep];
  if (params->r[0] == 0) {
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref->xqd[1] - SGRPROJ_PRJ_MIN1, si->xqd[1] - SGRPROJ_PRJ_MIN1);
  } else if (params->r[1] == 0) {
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref->xqd[0] - SGRPROJ_PRJ_MIN0, si->xqd[0] - SGRPROJ_PRJ_MIN0);
  } else {
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref->xqd[0] - SGRPROJ_PRJ_MIN0, si->xqd[0] - SGRPROJ_PRJ_MIN0);
    aom_write_primitive_refsubexpfin(
        wb, SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1, SGRPROJ_PRJ_SUBEXP_K,
        ref->xqd[1] - SGRPROJ_PRJ_MIN1, si->xqd[1] - SGRPROJ_PRJ_MIN1);
  }
  *ref = *si;
}

/* Transcribed read_wiener_filter (decoder/decodeframe.c) over the REAL
 * exported aom_read_primitive_refsubexpfin. */
static void lr_read_wiener(int wiener_win, WienerInfo *wi, WienerInfo *ref,
                           aom_reader *rb) {
  memset(wi->vfilter, 0, sizeof(wi->vfilter));
  memset(wi->hfilter, 0, sizeof(wi->hfilter));
  if (wiener_win == WIENER_WIN)
    wi->vfilter[0] = wi->vfilter[WIENER_WIN - 1] =
        aom_read_primitive_refsubexpfin(
            rb, WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
            WIENER_FILT_TAP0_SUBEXP_K,
            ref->vfilter[0] - WIENER_FILT_TAP0_MINV, NULL) +
        WIENER_FILT_TAP0_MINV;
  else
    wi->vfilter[0] = wi->vfilter[WIENER_WIN - 1] = 0;
  wi->vfilter[1] = wi->vfilter[WIENER_WIN - 2] =
      aom_read_primitive_refsubexpfin(
          rb, WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
          WIENER_FILT_TAP1_SUBEXP_K, ref->vfilter[1] - WIENER_FILT_TAP1_MINV,
          NULL) +
      WIENER_FILT_TAP1_MINV;
  wi->vfilter[2] = wi->vfilter[WIENER_WIN - 3] =
      aom_read_primitive_refsubexpfin(
          rb, WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
          WIENER_FILT_TAP2_SUBEXP_K, ref->vfilter[2] - WIENER_FILT_TAP2_MINV,
          NULL) +
      WIENER_FILT_TAP2_MINV;
  wi->vfilter[WIENER_HALFWIN] =
      -2 * (wi->vfilter[0] + wi->vfilter[1] + wi->vfilter[2]);
  if (wiener_win == WIENER_WIN)
    wi->hfilter[0] = wi->hfilter[WIENER_WIN - 1] =
        aom_read_primitive_refsubexpfin(
            rb, WIENER_FILT_TAP0_MAXV - WIENER_FILT_TAP0_MINV + 1,
            WIENER_FILT_TAP0_SUBEXP_K,
            ref->hfilter[0] - WIENER_FILT_TAP0_MINV, NULL) +
        WIENER_FILT_TAP0_MINV;
  else
    wi->hfilter[0] = wi->hfilter[WIENER_WIN - 1] = 0;
  wi->hfilter[1] = wi->hfilter[WIENER_WIN - 2] =
      aom_read_primitive_refsubexpfin(
          rb, WIENER_FILT_TAP1_MAXV - WIENER_FILT_TAP1_MINV + 1,
          WIENER_FILT_TAP1_SUBEXP_K, ref->hfilter[1] - WIENER_FILT_TAP1_MINV,
          NULL) +
      WIENER_FILT_TAP1_MINV;
  wi->hfilter[2] = wi->hfilter[WIENER_WIN - 3] =
      aom_read_primitive_refsubexpfin(
          rb, WIENER_FILT_TAP2_MAXV - WIENER_FILT_TAP2_MINV + 1,
          WIENER_FILT_TAP2_SUBEXP_K, ref->hfilter[2] - WIENER_FILT_TAP2_MINV,
          NULL) +
      WIENER_FILT_TAP2_MINV;
  wi->hfilter[WIENER_HALFWIN] =
      -2 * (wi->hfilter[0] + wi->hfilter[1] + wi->hfilter[2]);
  *ref = *wi;
}

/* Transcribed read_sgrproj_filter (decoder/decodeframe.c). */
static void lr_read_sgrproj(SgrprojInfo *si, SgrprojInfo *ref, aom_reader *rb) {
  si->ep = aom_read_literal(rb, SGRPROJ_PARAMS_BITS, NULL);
  const sgr_params_type *params = &av1_sgr_params[si->ep];
  if (params->r[0] == 0) {
    si->xqd[0] = 0;
    si->xqd[1] = aom_read_primitive_refsubexpfin(
                     rb, SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1,
                     SGRPROJ_PRJ_SUBEXP_K, ref->xqd[1] - SGRPROJ_PRJ_MIN1,
                     NULL) +
                 SGRPROJ_PRJ_MIN1;
  } else if (params->r[1] == 0) {
    si->xqd[0] = aom_read_primitive_refsubexpfin(
                     rb, SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1,
                     SGRPROJ_PRJ_SUBEXP_K, ref->xqd[0] - SGRPROJ_PRJ_MIN0,
                     NULL) +
                 SGRPROJ_PRJ_MIN0;
    si->xqd[1] = clamp((1 << SGRPROJ_PRJ_BITS) - si->xqd[0], SGRPROJ_PRJ_MIN1,
                       SGRPROJ_PRJ_MAX1);
  } else {
    si->xqd[0] = aom_read_primitive_refsubexpfin(
                     rb, SGRPROJ_PRJ_MAX0 - SGRPROJ_PRJ_MIN0 + 1,
                     SGRPROJ_PRJ_SUBEXP_K, ref->xqd[0] - SGRPROJ_PRJ_MIN0,
                     NULL) +
                 SGRPROJ_PRJ_MIN0;
    si->xqd[1] = aom_read_primitive_refsubexpfin(
                     rb, SGRPROJ_PRJ_MAX1 - SGRPROJ_PRJ_MIN1 + 1,
                     SGRPROJ_PRJ_SUBEXP_K, ref->xqd[1] - SGRPROJ_PRJ_MIN1,
                     NULL) +
                 SGRPROJ_PRJ_MIN1;
  }
  *ref = *si;
}

/* Write units[n] (LRU_WORDS i32 each) with the REAL writer over the REAL
 * default LR CDFs, read them back with the REAL reader, return the stream
 * (out/out_cap), the read-back unit params (readback, LRU_WORDS each) and
 * the reader's final CDFs: sw[4], wn[3], sg[3]. Returns stream length or <0. */
long shim_lr_units_roundtrip(const int32_t *units, int n, uint8_t *out,
                             long out_cap, int32_t *readback,
                             uint16_t *cdf_out /*[10]*/) {
  /* REAL default CDFs via av1_setup_past_independence (section 2 pattern). */
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(AV1_COMMON));
  FRAME_CONTEXT *fc = (FRAME_CONTEXT *)calloc(1, sizeof(FRAME_CONTEXT));
  FRAME_CONTEXT *dfc = (FRAME_CONTEXT *)calloc(1, sizeof(FRAME_CONTEXT));
  RefCntBuffer *rcb = (RefCntBuffer *)calloc(1, sizeof(RefCntBuffer));
  if (!cm || !fc || !dfc || !rcb) return -1;
  cm->fc = fc;
  cm->default_frame_context = dfc;
  cm->cur_frame = rcb;
  av1_setup_past_independence(cm);

  uint8_t *buf = (uint8_t *)malloc(1 << 20);
  if (!buf) return -1;
  aom_writer w;
  memset(&w, 0, sizeof(w));
  w.allow_update_cdf = 1;
  aom_start_encode(&w, buf);

  WienerInfo wref[3];
  SgrprojInfo sref[3];
  for (int p = 0; p < 3; p++) {
    set_default_wiener(&wref[p]);
    set_default_sgrproj(&sref[p]);
  }
  /* Writer-side CDF copies (adapt in lockstep with the reader). */
  aom_cdf_prob wsw[4], wwn[3], wsg[3];
  memcpy(wsw, fc->switchable_restore_cdf, sizeof(wsw));
  memcpy(wwn, fc->wiener_restore_cdf, sizeof(wwn));
  memcpy(wsg, fc->sgrproj_restore_cdf, sizeof(wsg));

  for (int i = 0; i < n; i++) {
    const int32_t *u = units + (long)i * LRU_WORDS;
    const int plane = u[LRU_PLANE];
    const int frt = u[LRU_FRTYPE];
    const int rt = u[LRU_RTYPE];
    const int win = plane > 0 ? WIENER_WIN_CHROMA : WIENER_WIN;
    WienerInfo wi;
    SgrprojInfo si;
    lr_fill_wiener(&wi, u);
    si.ep = u[LRU_EP];
    si.xqd[0] = u[LRU_XQD0];
    si.xqd[1] = u[LRU_XQD1];
    if (frt == RESTORE_SWITCHABLE) {
      aom_write_symbol(&w, rt, wsw, RESTORE_SWITCHABLE_TYPES);
      if (rt == RESTORE_WIENER)
        lr_write_wiener(win, &wi, &wref[plane], &w);
      else if (rt == RESTORE_SGRPROJ)
        lr_write_sgrproj(&si, &sref[plane], &w);
    } else if (frt == RESTORE_WIENER) {
      aom_write_symbol(&w, rt != RESTORE_NONE, wwn, 2);
      if (rt != RESTORE_NONE) lr_write_wiener(win, &wi, &wref[plane], &w);
    } else {
      aom_write_symbol(&w, rt != RESTORE_NONE, wsg, 2);
      if (rt != RESTORE_NONE) lr_write_sgrproj(&si, &sref[plane], &w);
    }
  }
  if (aom_stop_encode(&w) < 0) return -2;
  long len = w.pos;
  if (len > out_cap) return -3;
  memcpy(out, buf, len);

  /* Read back with the REAL reader + fresh refs + fresh default CDFs. */
  aom_reader r;
  memset(&r, 0, sizeof(r));
  if (aom_reader_init(&r, buf, len)) return -4;
  r.allow_update_cdf = 1;
  for (int p = 0; p < 3; p++) {
    set_default_wiener(&wref[p]);
    set_default_sgrproj(&sref[p]);
  }
  aom_cdf_prob rsw[4], rwn[3], rsg[3];
  memcpy(rsw, fc->switchable_restore_cdf, sizeof(rsw));
  memcpy(rwn, fc->wiener_restore_cdf, sizeof(rwn));
  memcpy(rsg, fc->sgrproj_restore_cdf, sizeof(rsg));

  for (int i = 0; i < n; i++) {
    const int32_t *u = units + (long)i * LRU_WORDS;
    int32_t *o = readback + (long)i * LRU_WORDS;
    const int plane = u[LRU_PLANE];
    const int frt = u[LRU_FRTYPE];
    const int win = plane > 0 ? WIENER_WIN_CHROMA : WIENER_WIN;
    memset(o, 0, LRU_WORDS * sizeof(int32_t));
    o[LRU_PLANE] = plane;
    o[LRU_FRTYPE] = frt;
    int rt = RESTORE_NONE;
    WienerInfo wi;
    SgrprojInfo si;
    memset(&wi, 0, sizeof(wi));
    memset(&si, 0, sizeof(si));
    if (frt == RESTORE_SWITCHABLE) {
      rt = aom_read_symbol(&r, rsw, RESTORE_SWITCHABLE_TYPES, NULL);
      if (rt == RESTORE_WIENER)
        lr_read_wiener(win, &wi, &wref[plane], &r);
      else if (rt == RESTORE_SGRPROJ)
        lr_read_sgrproj(&si, &sref[plane], &r);
    } else if (frt == RESTORE_WIENER) {
      if (aom_read_symbol(&r, rwn, 2, NULL)) {
        rt = RESTORE_WIENER;
        lr_read_wiener(win, &wi, &wref[plane], &r);
      }
    } else {
      if (aom_read_symbol(&r, rsg, 2, NULL)) {
        rt = RESTORE_SGRPROJ;
        lr_read_sgrproj(&si, &sref[plane], &r);
      }
    }
    o[LRU_RTYPE] = rt;
    if (rt == RESTORE_WIENER) {
      o[LRU_V0 + 0] = wi.vfilter[0];
      o[LRU_V0 + 1] = wi.vfilter[1];
      o[LRU_V0 + 2] = wi.vfilter[2];
      o[LRU_V0 + 3] = wi.hfilter[0];
      o[LRU_V0 + 4] = wi.hfilter[1];
      o[LRU_V0 + 5] = wi.hfilter[2];
    } else if (rt == RESTORE_SGRPROJ) {
      o[LRU_EP] = si.ep;
      o[LRU_XQD0] = si.xqd[0];
      o[LRU_XQD1] = si.xqd[1];
    }
  }
  memcpy(cdf_out + 0, rsw, sizeof(rsw));
  memcpy(cdf_out + 4, rwn, sizeof(rwn));
  memcpy(cdf_out + 7, rsg, sizeof(rsg));

  free(buf);
  free(rcb);
  free(dfc);
  free(fc);
  free(cm);
  return len;
}

/* REAL av1_alloc_restoration_struct unit-grid geometry + REAL
 * av1_loop_restoration_corners_in_sb for one (plane, SB). out[6] =
 * { horz_units, vert_units, rcol0, rcol1, rrow0, rrow1 }; returns the C
 * corners_in_sb return (0 = no unit corners in this SB / not sb_size). */
int shim_lr_corners_in_sb(int w, int h, int ss_x, int ss_y,
                          const int32_t *unit_size /*[3]*/, int plane,
                          int mi_row, int mi_col, int bsize, int32_t *out) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  struct aom_internal_error_info *err =
      (struct aom_internal_error_info *)calloc(1, sizeof(*err));
  if (!cm || !seq || !err) return -1;
  seq->subsampling_x = ss_x;
  seq->subsampling_y = ss_y;
  seq->sb_size = BLOCK_64X64;
  cm->seq_params = seq;
  cm->error = err;
  cm->width = w;
  cm->height = h;
  cm->superres_upscaled_width = w;
  cm->superres_upscaled_height = h;
  cm->superres_scale_denominator = SCALE_NUMERATOR;
  cm->mi_params.mi_rows = ((h + 7) & ~7) >> 2;
  cm->mi_params.mi_cols = ((w + 7) & ~7) >> 2;
  for (int p = 0; p < 3; p++) {
    cm->rst_info[p].restoration_unit_size = unit_size[p];
    cm->rst_info[p].frame_restoration_type = RESTORE_WIENER;
    av1_alloc_restoration_struct(cm, &cm->rst_info[p], p > 0);
  }
  out[0] = cm->rst_info[plane].horz_units;
  out[1] = cm->rst_info[plane].vert_units;
  int rcol0 = 0, rcol1 = 0, rrow0 = 0, rrow1 = 0;
  const int hit = av1_loop_restoration_corners_in_sb(
      cm, plane, mi_row, mi_col, (BLOCK_SIZE)bsize, &rcol0, &rcol1, &rrow0,
      &rrow1);
  out[2] = rcol0;
  out[3] = rcol1;
  out[4] = rrow0;
  out[5] = rrow1;
  for (int p = 0; p < 3; p++) av1_free_restoration_struct(&cm->rst_info[p]);
  free(err);
  free(seq);
  free(cm);
  return hit;
}

/* REAL av1_wiener_convolve_add_src_c (bd==8, on u8 copies — the production
 * lowbd path) / av1_highbd_wiener_convolve_add_src_c (bd>8) over a
 * caller-padded buffer of buf_w x buf_h u16 samples: filters the w x h block
 * whose top-left is (off_x, off_y), writing the same region of dst (whole
 * dst buffer copied back). Filters are copied into a 16-byte-aligned
 * WienerInfo (the C subpel machinery requires InterpKernel alignment). */
#include "av1/common/convolve.h"
int shim_wiener_convolve(const uint16_t *src, uint16_t *dst, int buf_w,
                         int buf_h, int off_x, int off_y, int w, int h,
                         const int16_t *hf, const int16_t *vf, int bd) {
  WienerInfo wi;
  memset(&wi, 0, sizeof(wi));
  memcpy(wi.hfilter, hf, 8 * sizeof(int16_t));
  memcpy(wi.vfilter, vf, 8 * sizeof(int16_t));
  const WienerConvolveParams conv_params = get_conv_params_wiener(bd);
  const long n = (long)buf_w * buf_h;
  const long off = (long)off_y * buf_w + off_x;
  if (bd == 8) {
    uint8_t *s8 = (uint8_t *)malloc(n);
    uint8_t *d8 = (uint8_t *)malloc(n);
    if (!s8 || !d8) return -1;
    for (long i = 0; i < n; i++) {
      s8[i] = (uint8_t)src[i];
      d8[i] = (uint8_t)dst[i];
    }
    av1_wiener_convolve_add_src_c(s8 + off, buf_w, d8 + off, buf_w, wi.hfilter,
                                  16, wi.vfilter, 16, w, h, &conv_params);
    for (long i = 0; i < n; i++) dst[i] = d8[i];
    free(s8);
    free(d8);
  } else {
    uint16_t *s16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!s16) return -1;
    memcpy(s16, src, n * sizeof(uint16_t));
    av1_highbd_wiener_convolve_add_src_c(
        CONVERT_TO_BYTEPTR(s16) + off, buf_w, CONVERT_TO_BYTEPTR(dst) + off,
        buf_w, wi.hfilter, 16, wi.vfilter, 16, w, h, &conv_params, bd);
    free(s16);
  }
  return 0;
}

/* REAL av1_apply_selfguided_restoration_c over the same buffer convention. */
int shim_apply_sgr(const uint16_t *src, uint16_t *dst, int buf_w, int buf_h,
                   int off_x, int off_y, int w, int h, int ep, int xqd0,
                   int xqd1, int bd) {
  int32_t *tmpbuf = (int32_t *)malloc(RESTORATION_TMPBUF_SIZE);
  if (!tmpbuf) return -1;
  int xqd[2] = { xqd0, xqd1 };
  const long n = (long)buf_w * buf_h;
  const long off = (long)off_y * buf_w + off_x;
  int rc;
  if (bd == 8) {
    uint8_t *s8 = (uint8_t *)malloc(n);
    uint8_t *d8 = (uint8_t *)malloc(n);
    if (!s8 || !d8) return -1;
    for (long i = 0; i < n; i++) {
      s8[i] = (uint8_t)src[i];
      d8[i] = (uint8_t)dst[i];
    }
    rc = av1_apply_selfguided_restoration_c(s8 + off, w, h, buf_w, ep, xqd,
                                            d8 + off, buf_w, tmpbuf, bd, 0);
    for (long i = 0; i < n; i++) dst[i] = d8[i];
    free(s8);
    free(d8);
  } else {
    uint16_t *s16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!s16) return -1;
    memcpy(s16, src, n * sizeof(uint16_t));
    rc = av1_apply_selfguided_restoration_c(CONVERT_TO_BYTEPTR(s16) + off, w,
                                            h, buf_w, ep, xqd,
                                            CONVERT_TO_BYTEPTR(dst) + off,
                                            buf_w, tmpbuf, bd, 1);
    free(s16);
  }
  free(tmpbuf);
  return rc;
}

/* The REAL whole-frame loop-restoration application over real bordered YV12
 * buffers + a real AV1_COMMON: av1_alloc_restoration_struct (unit grids) +
 * av1_alloc_restoration_buffers (stripe boundaries + rlbs + tmpbuf) +
 * av1_loop_restoration_save_boundary_lines in the DECODER's ordering
 * (after_cdef=0 on the deblocked frame, after_cdef=1 on the current frame;
 * skipped entirely on the optimized no-cdef path, decodeframe.c:5437-5482) +
 * av1_loop_restoration_filter_frame. bd==8 runs the production lowbd u8
 * frame; bd>8 highbd. units are 10 i32 each:
 * [rtype, v0,v1,v2, h0,h1,h2, ep, xqd0, xqd1]. NEEDS ref_init() (the walk's
 * wiener/sgr kernels are RTCD fn pointers). Returns 0 on success. */
#define LRF_WORDS 10
static void lrf_fill_unit(RestorationUnitInfo *rui, const int32_t *u) {
  memset(rui, 0, sizeof(*rui));
  rui->restoration_type = (RestorationType)u[0];
  for (int d = 0; d < 2; d++) {
    int16_t *f = d == 0 ? rui->wiener_info.vfilter : rui->wiener_info.hfilter;
    const int32_t *t = u + 1 + 3 * d;
    f[0] = (int16_t)t[0];
    f[1] = (int16_t)t[1];
    f[2] = (int16_t)t[2];
    f[3] = (int16_t)(-2 * (t[0] + t[1] + t[2]));
    f[4] = (int16_t)t[2];
    f[5] = (int16_t)t[1];
    f[6] = (int16_t)t[0];
  }
  rui->sgrproj_info.ep = u[7];
  rui->sgrproj_info.xqd[0] = u[8];
  rui->sgrproj_info.xqd[1] = u[9];
}

static void lrf_load_plane(YV12_BUFFER_CONFIG *f, int plane, const uint16_t *s,
                           int stride, int highbd) {
  const int is_uv = plane > 0;
  const int pw = f->crop_widths[is_uv];
  const int ph = f->crop_heights[is_uv];
  for (int r = 0; r < ph; r++) {
    if (highbd) {
      uint16_t *row = CONVERT_TO_SHORTPTR(f->buffers[plane]) +
                      (ptrdiff_t)r * f->strides[is_uv];
      memcpy(row, s + (ptrdiff_t)r * stride, pw * sizeof(uint16_t));
    } else {
      uint8_t *row = f->buffers[plane] + (ptrdiff_t)r * f->strides[is_uv];
      for (int c = 0; c < pw; c++) row[c] = (uint8_t)s[(ptrdiff_t)r * stride + c];
    }
  }
}

static void lrf_store_plane(const YV12_BUFFER_CONFIG *f, int plane,
                            uint16_t *d, int stride, int highbd) {
  const int is_uv = plane > 0;
  const int pw = f->crop_widths[is_uv];
  const int ph = f->crop_heights[is_uv];
  for (int r = 0; r < ph; r++) {
    if (highbd) {
      const uint16_t *row = CONVERT_TO_SHORTPTR(f->buffers[plane]) +
                            (ptrdiff_t)r * f->strides[is_uv];
      memcpy(d + (ptrdiff_t)r * stride, row, pw * sizeof(uint16_t));
    } else {
      const uint8_t *row = f->buffers[plane] + (ptrdiff_t)r * f->strides[is_uv];
      for (int c = 0; c < pw; c++) d[(ptrdiff_t)r * stride + c] = row[c];
    }
  }
}

int shim_lr_filter_frame(uint16_t *y, uint16_t *u, uint16_t *v,
                         const uint16_t *dy, const uint16_t *du,
                         const uint16_t *dv, int w, int h, int y_stride,
                         int uv_stride, int num_planes, int ss_x, int ss_y,
                         int bd, int optimized,
                         const int32_t *frame_rtype /*[3]*/,
                         const int32_t *unit_size /*[3]*/,
                         const int32_t *units0, const int32_t *units1,
                         const int32_t *units2) {
  const int highbd = bd > 8;
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(*cm));
  SequenceHeader *seq = (SequenceHeader *)calloc(1, sizeof(*seq));
  struct aom_internal_error_info *err =
      (struct aom_internal_error_info *)calloc(1, sizeof(*err));
  if (!cm || !seq || !err) return -1;
  seq->monochrome = num_planes == 1;
  seq->subsampling_x = ss_x;
  seq->subsampling_y = ss_y;
  seq->bit_depth = (aom_bit_depth_t)bd;
  seq->use_highbitdepth = highbd;
  seq->sb_size = BLOCK_64X64;
  cm->seq_params = seq;
  cm->error = err;
  cm->width = w;
  cm->height = h;
  cm->superres_upscaled_width = w;
  cm->superres_upscaled_height = h;
  cm->superres_scale_denominator = SCALE_NUMERATOR;
  cm->mi_params.mi_rows = ((h + 7) & ~7) >> 2;
  cm->mi_params.mi_cols = ((w + 7) & ~7) >> 2;

  int is_sgr = 0;
  const int32_t *unit_arrs[3] = { units0, units1, units2 };
  for (int p = 0; p < 3; p++) {
    RestorationInfo *rsi = &cm->rst_info[p];
    rsi->frame_restoration_type = (RestorationType)frame_rtype[p];
    rsi->restoration_unit_size = unit_size[p];
    av1_alloc_restoration_struct(cm, rsi, p > 0);
    if (frame_rtype[p] != RESTORE_NONE) {
      for (int i = 0; i < rsi->num_rest_units; i++)
        lrf_fill_unit(&rsi->unit_info[i], unit_arrs[p] + (long)i * LRF_WORDS);
      is_sgr = 1; /* allocate the sgr tmpbuf whenever anything restores */
    }
  }
  av1_alloc_restoration_buffers(cm, is_sgr || 1);

  YV12_BUFFER_CONFIG frame;
  memset(&frame, 0, sizeof(frame));
  if (aom_alloc_frame_buffer(&frame, w, h, ss_x, ss_y, highbd, 32, 0, false,
                             0))
    return -2;

  const uint16_t *cur[3] = { y, u, v };
  const uint16_t *deb[3] = { dy, du, dv };
  const int strides[3] = { y_stride, uv_stride, uv_stride };
  if (!optimized) {
    for (int p = 0; p < num_planes; p++)
      lrf_load_plane(&frame, p, deb[p], strides[p], highbd);
    av1_loop_restoration_save_boundary_lines(&frame, cm, 0);
    for (int p = 0; p < num_planes; p++)
      lrf_load_plane(&frame, p, cur[p], strides[p], highbd);
    av1_loop_restoration_save_boundary_lines(&frame, cm, 1);
  } else {
    for (int p = 0; p < num_planes; p++)
      lrf_load_plane(&frame, p, cur[p], strides[p], highbd);
  }

  AV1LrStruct lr_ctxt;
  memset(&lr_ctxt, 0, sizeof(lr_ctxt));
  av1_loop_restoration_filter_frame(&frame, cm, optimized, &lr_ctxt);

  uint16_t *outp[3] = { y, u, v };
  for (int p = 0; p < num_planes; p++)
    lrf_store_plane(&frame, p, outp[p], strides[p], highbd);

  aom_free_frame_buffer(&frame);
  aom_free_frame_buffer(&cm->rst_frame);
  av1_free_restoration_buffers(cm);
  free(err);
  free(seq);
  free(cm);
  return 0;
}

/* ------------------------------------------------------------------ */
/* 4. Palette colour-index map — REAL av1_decode_palette_tokens facade */
/* ------------------------------------------------------------------ */

#include "av1/decoder/detokenize.h"

/* av1_get_palette_color_index_context (av1/common/entropymode.c) is EXPORTED and
 * directly bound from Rust (no facade needed) — declared in aom-sys-ref/src/lib.rs. */

/* av1_get_block_dimensions facade (av1/common/blockd.h, static inline): a minimal
 * MACROBLOCKD carrying only the fields the real function reads
 * (mb_to_right_edge/mb_to_bottom_edge, plane[plane].subsampling_x/y). */
void shim_get_block_dimensions(int bsize, int plane, int ss_x, int ss_y,
                               int mb_to_right_edge, int mb_to_bottom_edge, int *width,
                               int *height, int *rows, int *cols) {
  MACROBLOCKD xd;
  memset(&xd, 0, sizeof(xd));
  xd.mb_to_right_edge = mb_to_right_edge;
  xd.mb_to_bottom_edge = mb_to_bottom_edge;
  xd.plane[plane].subsampling_x = plane == 0 ? 0 : ss_x;
  xd.plane[plane].subsampling_y = plane == 0 ? 0 : ss_y;
  av1_get_block_dimensions((BLOCK_SIZE)bsize, plane, &xd, width, height, rows, cols);
}

/* av1_decode_palette_tokens facade (av1/decoder/detokenize.c, exported): decode ONE
 * plane's colour-index map from a REAL aom_reader byte stream, driving the REAL
 * function end-to-end (av1_get_block_dimensions -> the wavefront loop ->
 * av1_get_palette_color_index_context) — so both av1_get_block_dimensions and the
 * token decode itself are cross-checked against Rust, not just transcribed. A minimal
 * MACROBLOCKD + MB_MODE_INFO + FRAME_CONTEXT stand in for the fields the call chain
 * reads (mirrors the shim_lr_units_roundtrip aom_reader setup above). `map_cdf_in` /
 * `map_cdf_out` are the PALETTE_SIZES x PALETTE_COLOR_INDEX_CONTEXTS x
 * CDF_SIZE(PALETTE_COLORS) CDF array (the plane's tile_ctx instance; `_out` is
 * post-adaptation). `color_map_out` (>= MAX_PALETTE_BLOCK_WIDTH*MAX_PALETTE_BLOCK_HEIGHT
 * bytes) is the decoded map, MAX_PALETTE_BLOCK_WIDTH-strided. Returns 0 on success. */
int shim_decode_palette_tokens(const uint8_t *data, size_t len, int plane, int bsize,
                               int n_colors, int ss_x, int ss_y, int mb_to_right_edge,
                               int mb_to_bottom_edge, const uint16_t *map_cdf_in,
                               uint16_t *map_cdf_out, uint8_t *color_map_out) {
  MACROBLOCKD xd;
  MB_MODE_INFO mi;
  MB_MODE_INFO *mi_ptr = &mi;
  FRAME_CONTEXT fc;
  memset(&xd, 0, sizeof(xd));
  memset(&mi, 0, sizeof(mi));
  memset(&fc, 0, sizeof(fc));
  mi.bsize = (BLOCK_SIZE)bsize;
  mi.palette_mode_info.palette_size[plane] = n_colors;
  xd.mi = &mi_ptr;
  xd.plane[0].subsampling_x = 0;
  xd.plane[0].subsampling_y = 0;
  xd.plane[1].subsampling_x = ss_x;
  xd.plane[1].subsampling_y = ss_y;
  xd.mb_to_right_edge = mb_to_right_edge;
  xd.mb_to_bottom_edge = mb_to_bottom_edge;

  uint8_t *map_buf =
      (uint8_t *)calloc((size_t)MAX_PALETTE_BLOCK_WIDTH * MAX_PALETTE_BLOCK_HEIGHT, 1);
  if (!map_buf) return 1;
  xd.plane[plane].color_index_map = map_buf;
  xd.color_index_map_offset[plane] = 0;

  const size_t cdf_bytes = (size_t)PALETTE_SIZES * PALETTE_COLOR_INDEX_CONTEXTS *
                           CDF_SIZE(PALETTE_COLORS) * sizeof(uint16_t);
  if (plane == 0) {
    memcpy(fc.palette_y_color_index_cdf, map_cdf_in, cdf_bytes);
  } else {
    memcpy(fc.palette_uv_color_index_cdf, map_cdf_in, cdf_bytes);
  }
  xd.tile_ctx = &fc;

  aom_reader r;
  memset(&r, 0, sizeof(r));
  if (aom_reader_init(&r, data, len)) {
    free(map_buf);
    return 2;
  }
  r.allow_update_cdf = 1;

  av1_decode_palette_tokens(&xd, plane, &r);

  memcpy(color_map_out, map_buf,
         (size_t)MAX_PALETTE_BLOCK_WIDTH * MAX_PALETTE_BLOCK_HEIGHT);
  memcpy(map_cdf_out,
         plane == 0 ? fc.palette_y_color_index_cdf : fc.palette_uv_color_index_cdf,
         cdf_bytes);
  free(map_buf);
  return 0;
}

/* ===================== intrabc DV prediction facades ===========================
 * Facades for av1_find_mv_refs + av1_find_best_ref_mvs (av1/common/mvref_common.c,
 * BOTH real exported/non-static functions -- driven DIRECTLY, not transcribed) at
 * ref_frame == INTRA_FRAME (the read_intrabc_info DV-predictor path,
 * av1/decoder/decodemv.c), plus the real `static inline` av1_find_ref_dv /
 * av1_is_dv_valid (av1/common/mvref_common.h). Rust's dv_ref.rs module is diffed
 * against these three entry points. */
#include "av1/common/mvref_common.h"

/* Window size for the synthetic MI grid: mi rows/cols [0, DV_GRID_DIM). The Rust
 * harness places the current block at a fixed (mi_row, mi_col) with enough margin
 * on every side for the scan's maximum reach (~8 mi units up/left; up to
 * BLOCK_128X128 = 32 mi units of the block's own footprint down/right) -- see
 * dv_ref_diff.rs. */
#define DV_GRID_DIM 128

/* Facade for av1_find_mv_refs(ref_frame=INTRA_FRAME) + av1_find_best_ref_mvs:
 * builds a real MB_MODE_INFO pool + MB_MODE_INFO* grid from flat per-cell arrays
 * (row-major, DV_GRID_DIM-strided) and calls the REAL exported functions. Only
 * the fields setup_ref_mv_list's INTRA_FRAME path reads are populated. */
void shim_find_dv_ref_mvs(
    int mi_row, int mi_col, int bsize, int own_partition, int up_available,
    int left_available, int tile_mi_row_start, int tile_mi_row_end,
    int tile_mi_col_start, int tile_mi_col_end, int frame_mi_rows, int frame_mi_cols,
    int mib_size, const uint8_t *g_bsize, const int8_t *g_ref_frame0,
    const int8_t *g_ref_frame1, const uint8_t *g_use_intrabc, const uint8_t *g_mode,
    const int16_t *g_mv0_row, const int16_t *g_mv0_col, const int16_t *g_mv1_row,
    const int16_t *g_mv1_col, int *out_nearest_row, int *out_nearest_col,
    int *out_near_row, int *out_near_col) {
  const size_t n = (size_t)DV_GRID_DIM * (size_t)DV_GRID_DIM;
  MB_MODE_INFO *pool = (MB_MODE_INFO *)calloc(n, sizeof(MB_MODE_INFO));
  MB_MODE_INFO **grid = (MB_MODE_INFO **)calloc(n, sizeof(MB_MODE_INFO *));
  for (size_t i = 0; i < n; ++i) {
    pool[i].bsize = (BLOCK_SIZE)g_bsize[i];
    pool[i].ref_frame[0] = (MV_REFERENCE_FRAME)g_ref_frame0[i];
    pool[i].ref_frame[1] = (MV_REFERENCE_FRAME)g_ref_frame1[i];
    pool[i].use_intrabc = g_use_intrabc[i] ? 1 : 0;
    pool[i].mode = (PREDICTION_MODE)g_mode[i];
    pool[i].mv[0].as_mv.row = g_mv0_row[i];
    pool[i].mv[0].as_mv.col = g_mv0_col[i];
    pool[i].mv[1].as_mv.row = g_mv1_row[i];
    pool[i].mv[1].as_mv.col = g_mv1_col[i];
    grid[i] = &pool[i];
  }
  /* The current block's own cell (xd->mi[0]) carries its own bsize/partition,
   * overriding whatever the flat neighbour arrays supplied there. */
  size_t self_idx = (size_t)mi_row * DV_GRID_DIM + (size_t)mi_col;
  pool[self_idx].bsize = (BLOCK_SIZE)bsize;
  pool[self_idx].partition = (PARTITION_TYPE)own_partition;

  AV1_COMMON cm;
  MACROBLOCKD xd;
  SequenceHeader sp;
  memset(&cm, 0, sizeof(cm));
  memset(&xd, 0, sizeof(xd));
  memset(&sp, 0, sizeof(sp));
  sp.sb_size = (mib_size >= 32) ? BLOCK_128X128 : BLOCK_64X64;
  cm.seq_params = &sp;
  cm.mi_params.mi_rows = frame_mi_rows;
  cm.mi_params.mi_cols = frame_mi_cols;
  cm.features.allow_ref_frame_mvs = 0;

  xd.mi_row = mi_row;
  xd.mi_col = mi_col;
  xd.mi_stride = DV_GRID_DIM;
  xd.mi = &grid[self_idx];
  xd.width = mi_size_wide[bsize];
  xd.height = mi_size_high[bsize];
  xd.up_available = up_available;
  xd.left_available = left_available;
  xd.tile.mi_row_start = tile_mi_row_start;
  xd.tile.mi_row_end = tile_mi_row_end;
  xd.tile.mi_col_start = tile_mi_col_start;
  xd.tile.mi_col_end = tile_mi_col_end;
  /* set_mi_row_col (av1_common_int.h): the frame-edge distances clamp_mv_ref
   * clamps ref_mv_stack candidates against. MI_SIZE=4 px/mi,
   * GET_MV_SUBPEL(x)=x*8 (1/8-pel units). Missing this population previously
   * left these at 0 (memset), which silently clamped every candidate to a
   * tiny +-(bw_px*8+128)-ish window centered on zero instead of the real
   * frame-relative window -- a facade bug, not a Rust port bug. */
  xd.mb_to_top_edge = -(mi_row * 4 * 8);
  xd.mb_to_bottom_edge = (frame_mi_rows - mi_size_high[bsize] - mi_row) * 4 * 8;
  xd.mb_to_left_edge = -(mi_col * 4 * 8);
  xd.mb_to_right_edge = (frame_mi_cols - mi_size_wide[bsize] - mi_col) * 4 * 8;
  /* set_mi_row_col (av1_common_int.h): has_top_right (mvref_common.c) reads
   * these two flags directly (not re-derived from width/height itself), so
   * leaving them at their memset-0 default silently forces has_top_right's
   * `if (!xd->is_last_vertical_rect) has_tr = 1;` / `... has_tr = 0;` arms to
   * fire incorrectly whenever the real (derived) value would be true --
   * another facade-population gap, not a Rust port bug. */
  xd.is_last_vertical_rect = 0;
  if (xd.width < xd.height) {
    if (!((mi_col + xd.width) & (xd.height - 1))) xd.is_last_vertical_rect = 1;
  }
  xd.is_first_horizontal_rect = 0;
  if (xd.width > xd.height) {
    if (!(mi_row & (xd.width - 1))) xd.is_first_horizontal_rect = 1;
  }

  uint8_t ref_mv_count[MODE_CTX_REF_FRAMES];
  CANDIDATE_MV ref_mv_stack[MODE_CTX_REF_FRAMES][MAX_REF_MV_STACK_SIZE];
  uint16_t ref_mv_weight[MODE_CTX_REF_FRAMES][MAX_REF_MV_STACK_SIZE];
  int_mv mv_ref_list[MODE_CTX_REF_FRAMES][MAX_MV_REF_CANDIDATES];
  int_mv global_mvs[MODE_CTX_REF_FRAMES];
  int16_t mode_context[MODE_CTX_REF_FRAMES];
  memset(ref_mv_count, 0, sizeof(ref_mv_count));
  memset(ref_mv_stack, 0, sizeof(ref_mv_stack));
  memset(ref_mv_weight, 0, sizeof(ref_mv_weight));
  memset(mv_ref_list, 0, sizeof(mv_ref_list));
  memset(global_mvs, 0, sizeof(global_mvs));
  memset(mode_context, 0, sizeof(mode_context));

  av1_find_mv_refs(&cm, &xd, &pool[self_idx], INTRA_FRAME, ref_mv_count, ref_mv_stack,
                   ref_mv_weight, mv_ref_list, global_mvs, mode_context);

  int_mv nearest_mv, near_mv;
  av1_find_best_ref_mvs(0, mv_ref_list[INTRA_FRAME], &nearest_mv, &near_mv, 0);

  *out_nearest_row = nearest_mv.as_mv.row;
  *out_nearest_col = nearest_mv.as_mv.col;
  *out_near_row = near_mv.as_mv.row;
  *out_near_col = near_mv.as_mv.col;

  free(pool);
  free(grid);
}

/* Facade for the real `static inline` av1_find_ref_dv (mvref_common.h): only
 * `tile->mi_row_start` is read. */
void shim_find_ref_dv(int mi_row, int mib_size, int tile_mi_row_start, int *out_row,
                      int *out_col) {
  TileInfo tile;
  memset(&tile, 0, sizeof(tile));
  tile.mi_row_start = tile_mi_row_start;
  int_mv ref_dv;
  memset(&ref_dv, 0, sizeof(ref_dv));
  av1_find_ref_dv(&ref_dv, &tile, mib_size, mi_row);
  *out_row = ref_dv.as_mv.row;
  *out_col = ref_dv.as_mv.col;
}

/* Facade for the real `static inline` av1_is_dv_valid (mvref_common.h): a minimal
 * AV1_COMMON/MACROBLOCKD carrying only the fields it reads (xd->tile,
 * xd->is_chroma_ref, xd->plane[1].subsampling_{x,y}, cm->seq_params->monochrome via
 * av1_num_planes). Returns 1/0. */
int shim_is_dv_valid(int dv_row, int dv_col, int mi_row, int mi_col, int bsize,
                     int tile_mi_row_start, int tile_mi_row_end, int tile_mi_col_start,
                     int tile_mi_col_end, int mib_size_log2, int is_chroma_ref,
                     int num_planes, int ss_x, int ss_y) {
  AV1_COMMON cm;
  MACROBLOCKD xd;
  SequenceHeader sp;
  memset(&cm, 0, sizeof(cm));
  memset(&xd, 0, sizeof(xd));
  memset(&sp, 0, sizeof(sp));
  sp.monochrome = (num_planes <= 1);
  cm.seq_params = &sp;
  xd.tile.mi_row_start = tile_mi_row_start;
  xd.tile.mi_row_end = tile_mi_row_end;
  xd.tile.mi_col_start = tile_mi_col_start;
  xd.tile.mi_col_end = tile_mi_col_end;
  xd.is_chroma_ref = is_chroma_ref ? true : false;
  xd.plane[1].subsampling_x = ss_x;
  xd.plane[1].subsampling_y = ss_y;
  MV dv;
  dv.row = (int16_t)dv_row;
  dv.col = (int16_t)dv_col;
  return av1_is_dv_valid(dv, &cm, &xd, mi_row, mi_col, (BLOCK_SIZE)bsize, mib_size_log2);
}

/* shim_dump_default_inter_ext_tx — dump the compiled default
 * fc->inter_ext_tx_cdf[EXT_TX_SETS_INTER][EXT_TX_SIZES][CDF_SIZE(TX_TYPES)]
 * (the full padded [4][4][17] = 272 u16 table) from the real
 * av1_setup_past_independence default frame context. inter_ext_tx_cdf is
 * qindex-independent (av1_init_mode_probs); base_qindex is accepted only to
 * mirror shim_dump_default_kf_fc. Verifies aom-entropy's DEFAULT_INTER_EXT_TX. */
int shim_dump_default_inter_ext_tx(int base_qindex, uint16_t *out) {
  AV1_COMMON *cm = (AV1_COMMON *)calloc(1, sizeof(AV1_COMMON));
  FRAME_CONTEXT *fc = (FRAME_CONTEXT *)calloc(1, sizeof(FRAME_CONTEXT));
  FRAME_CONTEXT *dfc = (FRAME_CONTEXT *)calloc(1, sizeof(FRAME_CONTEXT));
  RefCntBuffer *rcb = (RefCntBuffer *)calloc(1, sizeof(RefCntBuffer));
  if (!cm || !fc || !dfc || !rcb) return 1;
  cm->fc = fc;
  cm->default_frame_context = dfc;
  cm->cur_frame = rcb; /* seg_map NULL -> the memset arm is skipped */
  cm->quant_params.base_qindex = base_qindex;
  av1_setup_past_independence(cm);
  memcpy(out, fc->inter_ext_tx_cdf, sizeof(fc->inter_ext_tx_cdf)); /* 272 u16 */
  free(cm);
  free(fc);
  free(dfc);
  free(rcb);
  return 0;
}
