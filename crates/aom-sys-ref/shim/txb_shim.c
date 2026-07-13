/* Shim over av1 txb coefficient-coding kernels + scan/ctx-offset data.
 * Oracle use only. Include the real libaom headers so the kernel prototypes,
 * scan orders, cost helpers, and tx tables are the pristine declarations. */
#include <stdint.h>
#include <string.h>
#include "av1/common/enums.h"
#include "av1/common/scan.h"
#include "av1/common/txb_common.h"
#include "av1/encoder/block.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/encodetxb.h"
#include "av1/encoder/txb_rdopt_utils.h"

void shim_txb_init_levels(const int32_t *coeff, int width, int height,
                          uint8_t *levels) {
  av1_txb_init_levels_c(coeff, width, height, levels);
}

void shim_get_nz_map_contexts(const uint8_t *levels, const int16_t *scan,
                              int eob, int tx_size, int tx_class,
                              int8_t *out) {
  av1_get_nz_map_contexts_c(levels, scan, eob, tx_size, tx_class, out);
}

int shim_eob_pos_token(int eob, int *extra) {
  return av1_get_eob_pos_token(eob, extra);
}

const int8_t *shim_nz_ctx_offset(int tx_size) {
  return av1_nz_map_ctx_offset[tx_size];
}

const int16_t *shim_scan(int tx_size, int tx_type) {
  return av1_scan_orders[tx_size][tx_type].scan;
}

const int16_t *shim_iscan(int tx_size, int tx_type) {
  return av1_scan_orders[tx_size][tx_type].iscan;
}

/* ---- av1_write_coeffs_txb bitstream harness -------------------------------
 * Transcribed body of av1_write_coeffs_txb (encodetxb.c) with the encoder
 * state (MACROBLOCK / FRAME_CONTEXT lookups) replaced by parameters:
 *  - tcoeff/eob/tx_size/tx_type/plane_type/txb_skip_ctx/dc_sign_ctx inputs;
 *  - all CDFs in a caller-owned flat u16 arena (layout below, mirrored in
 *    aom-txb); mutated in place when allow_update_cdf.
 * Every symbol write + helper is the pristine C (od_ec, update_cdf,
 * av1_txb_init_levels_c, av1_get_nz_map_contexts_c, av1_scan_orders).
 * av1_write_tx_type (plane=0 tx_type signaling) is intentionally out of
 * scope. */
#include "aom_dsp/entenc.h"

/* Flat CDF arena offsets (u16 units). MUST match aom-txb::cdf_arena. */
#define A_TXB_SKIP 0            /* [5][13] n=2  (stride 3)  */
#define A_EOB16 195             /* [2][2]  n=5  (stride 6)  */
#define A_EOB32 219             /* [2][2]  n=6  (stride 7)  */
#define A_EOB64 247             /* [2][2]  n=7  (stride 8)  */
#define A_EOB128 279            /* [2][2]  n=8  (stride 9)  */
#define A_EOB256 315            /* [2][2]  n=9  (stride 10) */
#define A_EOB512 355            /* [2][2]  n=10 (stride 11) */
#define A_EOB1024 399           /* [2][2]  n=11 (stride 12) */
#define A_EOB_EXTRA 447         /* [5][2][9]  n=2 (stride 3) */
#define A_BASE_EOB 717          /* [5][2][4]  n=3 (stride 4) */
#define A_BASE 877              /* [5][2][42] n=4 (stride 5) */
#define A_BR 2977               /* [5][2][21] n=4 (stride 5) */
#define A_DC_SIGN 4027          /* [2][3]     n=2 (stride 3) */
#define A_TOTAL 4045

/* Header-static index tables (common_data.h/entropy.h values). */
static const int8_t k_txsize_log2_minus4[19] = { 0, 2, 4, 6, 6, 1, 1, 3, 3,
                                                 5, 5, 6, 6, 2, 2, 4, 4, 5, 5 };
static const int8_t k_txs_sqr[19] = { 0, 1, 2, 3, 4, 0, 0, 1, 1,
                                      2, 2, 3, 3, 0, 0, 1, 1, 2, 2 };
static const int8_t k_txs_sqr_up[19] = { 0, 1, 2, 3, 4, 1, 1, 2, 2,
                                         3, 3, 4, 4, 2, 2, 3, 3, 4, 4 };
static const int8_t k_tx_type_to_class[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 2, 1, 2, 1, 2, 1 };
static const int8_t k_txb_bhl[19] = { 2, 3, 4, 5, 5, 3, 2, 4, 3,
                                      5, 4, 5, 5, 4, 2, 5, 3, 5, 4 };
static const int8_t k_txb_wide[19] = { 4, 8, 16, 32, 32, 4, 8, 8, 16,
                                       16, 32, 32, 32, 4, 16, 8, 32, 16, 32 };
static const int8_t k_txb_high[19] = { 4, 8, 16, 32, 32, 8, 4, 16, 8,
                                       32, 16, 32, 32, 16, 4, 32, 8, 32, 16 };
extern const int16_t av1_eob_offset_bits[12];

static void tw_symbol(od_ec_enc *ec, uint16_t *cdf, int symb, int n, int upd) {
  od_ec_encode_cdf_q15(ec, symb, cdf, n);
  if (upd) update_cdf(cdf, symb, n);
}
static void tw_bit(od_ec_enc *ec, int bit) {
  /* aom_write_bit == aom_write(w, bit, 128) */
  int p = (0x7FFFFF - (128 << 15) + 128) >> 8;
  od_ec_encode_bool_q15(ec, bit, p);
}
static void tw_golomb(od_ec_enc *ec, int level) {
  int x = level + 1;
  int i = x;
  int length = 0;
  while (i) { i >>= 1; ++length; }
  for (i = 0; i < length - 1; ++i) tw_bit(ec, 0);
  for (i = length - 1; i >= 0; --i) tw_bit(ec, (x >> i) & 0x01);
}
/* get_br_ctx (txb_common.h), transposed layout. */
static int tw_get_br_ctx(const uint8_t *levels, int c, int bhl, int tx_class) {
  const int col = c >> bhl;
  const int row = c - (col << bhl);
  const int stride = (1 << bhl) + 4;
  const int pos = col * stride + row;
  int mag = levels[pos + 1];
  mag += levels[pos + stride];
  switch (tx_class) {
    case 0:
      mag += levels[pos + stride + 1];
      mag = (mag + 1) >> 1; if (mag > 6) mag = 6;
      if (c == 0) return mag;
      if ((row < 2) && (col < 2)) return mag + 7;
      break;
    case 1:
      mag += levels[pos + (stride << 1)];
      mag = (mag + 1) >> 1; if (mag > 6) mag = 6;
      if (c == 0) return mag;
      if (col == 0) return mag + 7;
      break;
    case 2:
      mag += levels[pos + 2];
      mag = (mag + 1) >> 1; if (mag > 6) mag = 6;
      if (c == 0) return mag;
      if (row == 0) return mag + 7;
      break;
  }
  return mag + 14;
}

int shim_write_coeffs_txb(const int32_t *tcoeff, int eob, int tx_size,
                          int tx_type, int plane_type, int txb_skip_ctx,
                          int dc_sign_ctx, int allow_update_cdf,
                          uint16_t *cdfs, unsigned char *out, int out_cap) {
  od_ec_enc ec;
  od_ec_enc_init(&ec, 65536);
  const int txs_ctx = (k_txs_sqr[tx_size] + k_txs_sqr_up[tx_size] + 1) >> 1;

  tw_symbol(&ec, cdfs + A_TXB_SKIP + (txs_ctx * 13 + txb_skip_ctx) * 3,
            eob == 0, 2, allow_update_cdf);
  if (eob != 0) {
    /* (av1_write_tx_type intentionally skipped) */
    const int tx_class = k_tx_type_to_class[tx_type];
    int eob_extra;
    const int eob_pt = av1_get_eob_pos_token(eob, &eob_extra);
    const int eob_multi_size = k_txsize_log2_minus4[tx_size];
    const int eob_multi_ctx = (tx_class == 0) ? 0 : 1;
    static const int eob_off[7] = { A_EOB16, A_EOB32, A_EOB64, A_EOB128,
                                    A_EOB256, A_EOB512, A_EOB1024 };
    const int nsy = 5 + eob_multi_size;
    tw_symbol(&ec,
              cdfs + eob_off[eob_multi_size] +
                  (plane_type * 2 + eob_multi_ctx) * (nsy + 1),
              eob_pt - 1, nsy, allow_update_cdf);

    const int eob_offset_bits = av1_eob_offset_bits[eob_pt];
    if (eob_offset_bits > 0) {
      const int eob_ctx = eob_pt - 3;
      int eob_shift = eob_offset_bits - 1;
      int bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
      tw_symbol(&ec,
                cdfs + A_EOB_EXTRA + ((txs_ctx * 2 + plane_type) * 9 + eob_ctx) * 3,
                bit, 2, allow_update_cdf);
      for (int i = 1; i < eob_offset_bits; i++) {
        eob_shift = eob_offset_bits - 1 - i;
        bit = (eob_extra & (1 << eob_shift)) ? 1 : 0;
        tw_bit(&ec, bit);
      }
    }

    const int width = k_txb_wide[tx_size];
    const int height = k_txb_high[tx_size];
    uint8_t levels_buf[(32 + 4) * (32 + 4) + 16];
    uint8_t *const levels = levels_buf; /* TX_PAD_TOP == 0 */
    av1_txb_init_levels_c(tcoeff, width, height, levels);
    const int16_t *const scan = av1_scan_orders[tx_size][tx_type].scan;
    int8_t coeff_contexts[32 * 32];
    av1_get_nz_map_contexts_c(levels, scan, eob, tx_size, tx_class,
                              coeff_contexts);

    const int bhl = k_txb_bhl[tx_size];
    for (int c = eob - 1; c >= 0; --c) {
      const int pos = scan[c];
      const int coeff_ctx = coeff_contexts[pos];
      const int32_t v = tcoeff[pos];
      const int32_t level = v < 0 ? -v : v;

      if (c == eob - 1) {
        int s = (level < 3 ? level : 3) - 1;
        tw_symbol(&ec,
                  cdfs + A_BASE_EOB + ((txs_ctx * 2 + plane_type) * 4 + coeff_ctx) * 4,
                  s, 3, allow_update_cdf);
      } else {
        int s = level < 3 ? level : 3;
        tw_symbol(&ec,
                  cdfs + A_BASE + ((txs_ctx * 2 + plane_type) * 42 + coeff_ctx) * 5,
                  s, 4, allow_update_cdf);
      }
      if (level > 2 /* NUM_BASE_LEVELS */) {
        const int base_range = level - 1 - 2;
        const int br_ctx = tw_get_br_ctx(levels, pos, bhl, tx_class);
        const int mts = txs_ctx < 3 ? txs_ctx : 3; /* AOMMIN(txs_ctx, TX_32X32) */
        uint16_t *cdf = cdfs + A_BR + ((mts * 2 + plane_type) * 21 + br_ctx) * 5;
        for (int idx = 0; idx < 12 /* COEFF_BASE_RANGE */; idx += 3) {
          const int k = (base_range - idx) < 3 ? (base_range - idx) : 3;
          tw_symbol(&ec, cdf, k, 4, allow_update_cdf);
          if (k < 3) break;
        }
      }
    }

    for (int c = 0; c < eob; ++c) {
      const int32_t v = tcoeff[scan[c]];
      const int32_t level = v < 0 ? -v : v;
      const int sign = (v < 0) ? 1 : 0;
      if (level) {
        if (c == 0) {
          tw_symbol(&ec, cdfs + A_DC_SIGN + (plane_type * 3 + dc_sign_ctx) * 3,
                    sign, 2, allow_update_cdf);
        } else {
          tw_bit(&ec, sign);
        }
        if (level > 12 + 2)
          tw_golomb(&ec, level - 12 - 1 - 2);
      }
    }
  }

  uint32_t n = 0;
  const unsigned char *buf = od_ec_enc_done(&ec, &n);
  if ((int)n > out_cap) n = 0;
  else memcpy(out, buf, n);
  od_ec_enc_clear(&ec);
  return (int)n;
}

/* ---- av1_cost_coeffs_txb (warehouse_efficients_txb) RD-cost harness --------
 * Transcribes the warehouse_efficients_txb loop (txb_rdopt.c) but drops the
 * get_tx_type_cost term (plane-0 tx_type is out of scope, matching the writer).
 * The LV_MAP_COEFF_COST / LV_MAP_EOB_COST tables are supplied as flat int
 * arrays (identical on both sides) so this isolates the cost-summation logic
 * from the separate CDF->cost derivation. Uses the pristine static-inline cost
 * helpers (get_eob_cost, get_br_cost, get_br_ctx[_eob]) + av1_txb_init_levels_c
 * + av1_get_nz_map_contexts_c. */
int shim_cost_coeffs_txb(const int32_t *qcoeff, int eob, int tx_size,
                         int tx_type, int txb_skip_ctx, int dc_sign_ctx,
                         const int *txb_skip_cost, const int *base_eob_cost,
                         const int *base_cost, const int *eob_extra_cost,
                         const int *dc_sign_cost, const int *lps_cost,
                         const int *eob_cost_tbl) {
  LV_MAP_COEFF_COST cc;
  memcpy(cc.txb_skip_cost, txb_skip_cost, sizeof(cc.txb_skip_cost));
  memcpy(cc.base_eob_cost, base_eob_cost, sizeof(cc.base_eob_cost));
  memcpy(cc.base_cost, base_cost, sizeof(cc.base_cost));
  memcpy(cc.eob_extra_cost, eob_extra_cost, sizeof(cc.eob_extra_cost));
  memcpy(cc.dc_sign_cost, dc_sign_cost, sizeof(cc.dc_sign_cost));
  memcpy(cc.lps_cost, lps_cost, sizeof(cc.lps_cost));
  LV_MAP_EOB_COST ec;
  memcpy(ec.eob_cost, eob_cost_tbl, sizeof(ec.eob_cost));

  const TX_CLASS tx_class = tx_type_to_class[tx_type];
  const int bhl = get_txb_bhl(tx_size);
  const int width = get_txb_wide(tx_size);
  const int height = get_txb_high(tx_size);
  const int16_t *const scan = av1_scan_orders[tx_size][tx_type].scan;
  uint8_t levels_buf[TX_PAD_2D];
  uint8_t *const levels = set_levels(levels_buf, height);
  int8_t coeff_contexts[MAX_TX_SQUARE];

  int cost = cc.txb_skip_cost[txb_skip_ctx][0];
  if (eob > 1) av1_txb_init_levels_c(qcoeff, width, height, levels);
  /* get_tx_type_cost intentionally omitted */
  cost += get_eob_cost(eob, &ec, &cc, tx_class);
  av1_get_nz_map_contexts_c(levels, scan, eob, tx_size, tx_class,
                            coeff_contexts);

  const int(*lps)[COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1] = cc.lps_cost;
  int c = eob - 1;
  {
    const int pos = scan[c];
    const int32_t v = qcoeff[pos];
    if (v) {
      const int sign = AOMSIGN(v);
      const int level = (v ^ sign) - sign;
      const int coeff_ctx = coeff_contexts[pos];
      cost += cc.base_eob_cost[coeff_ctx][AOMMIN(level, 3) - 1];
      if (level > NUM_BASE_LEVELS) {
        const int ctx = get_br_ctx_eob(pos, bhl, tx_class);
        cost += get_br_cost(level, lps[ctx]);
      }
      if (c) {
        cost += av1_cost_literal(1);
      } else {
        const int sign01 = (sign ^ sign) - sign;
        cost += cc.dc_sign_cost[dc_sign_ctx][sign01];
        return cost;
      }
    }
  }
  const int(*base_c)[8] = cc.base_cost;
  for (c = eob - 2; c >= 1; --c) {
    const int pos = scan[c];
    const int coeff_ctx = coeff_contexts[pos];
    const int32_t v = qcoeff[pos];
    if (!v) {
      cost += base_c[coeff_ctx][0];
      continue;
    }
    const int level = abs(v);
    cost += base_c[coeff_ctx][AOMMIN(level, 3)];
    cost += av1_cost_literal(1);
    if (level > NUM_BASE_LEVELS) {
      const int ctx = get_br_ctx(levels, pos, bhl, tx_class);
      cost += get_br_cost(level, lps[ctx]);
    }
  }
  {
    const int pos = scan[c];
    const int32_t v = qcoeff[pos];
    const int coeff_ctx = coeff_contexts[pos];
    if (!v) {
      cost += base_c[coeff_ctx][0];
    } else {
      const int sign = AOMSIGN(v);
      const int level = (v ^ sign) - sign;
      cost += base_c[coeff_ctx][AOMMIN(level, 3)];
      const int sign01 = (sign ^ sign) - sign;
      cost += cc.dc_sign_cost[dc_sign_ctx][sign01];
      if (level > NUM_BASE_LEVELS) {
        const int ctx = get_br_ctx(levels, pos, bhl, tx_class);
        cost += get_br_cost(level, lps[ctx]);
      }
    }
  }
  return cost;
}

/* ---- av1_cost_tokens_from_cdf (CDF -> per-symbol cost table) --------------- */
void av1_cost_tokens_from_cdf(int *costs, const uint16_t *cdf,
                              const int *inv_map);

void shim_cost_tokens_from_cdf(int *costs, const uint16_t *cdf,
                               const int *inv_map) {
  av1_cost_tokens_from_cdf(costs, cdf, inv_map);
}

/* ---- av1_fill_coeff_costs per-(txs_ctx, plane) LV_MAP_COEFF_COST fill -------
 * Transcribes the inner body of av1_fill_coeff_costs (rd.c): the caller supplies
 * the 6 coeff CDF groups already selected for this (txs_ctx, plane) as flat
 * buffers; we fill the LV_MAP_COEFF_COST tables via the real
 * av1_cost_tokens_from_cdf + the base_cost[4..7] and lps_cost cumulation/diff
 * fixups. Outputs are flat, matching the Rust LvMapCoeffCost layout. */
void shim_fill_lv_map(const uint16_t *txb_skip_cdf, const uint16_t *base_eob_cdf,
                      const uint16_t *base_cdf, const uint16_t *eob_extra_cdf,
                      const uint16_t *dc_sign_cdf, const uint16_t *br_cdf,
                      int *o_txb_skip, int *o_base_eob, int *o_base,
                      int *o_eob_extra, int *o_dc_sign, int *o_lps) {
  for (int ctx = 0; ctx < TXB_SKIP_CONTEXTS; ++ctx)
    av1_cost_tokens_from_cdf(o_txb_skip + ctx * 2, txb_skip_cdf + ctx * 3, NULL);
  for (int ctx = 0; ctx < SIG_COEF_CONTEXTS_EOB; ++ctx)
    av1_cost_tokens_from_cdf(o_base_eob + ctx * 3, base_eob_cdf + ctx * 4, NULL);
  for (int ctx = 0; ctx < SIG_COEF_CONTEXTS; ++ctx)
    av1_cost_tokens_from_cdf(o_base + ctx * 8, base_cdf + ctx * 5, NULL);
  for (int ctx = 0; ctx < SIG_COEF_CONTEXTS; ++ctx) {
    o_base[ctx * 8 + 4] = 0;
    o_base[ctx * 8 + 5] =
        o_base[ctx * 8 + 1] + av1_cost_literal(1) - o_base[ctx * 8 + 0];
    o_base[ctx * 8 + 6] = o_base[ctx * 8 + 2] - o_base[ctx * 8 + 1];
    o_base[ctx * 8 + 7] = o_base[ctx * 8 + 3] - o_base[ctx * 8 + 2];
  }
  for (int ctx = 0; ctx < EOB_COEF_CONTEXTS; ++ctx)
    av1_cost_tokens_from_cdf(o_eob_extra + ctx * 2, eob_extra_cdf + ctx * 3,
                             NULL);
  for (int ctx = 0; ctx < DC_SIGN_CONTEXTS; ++ctx)
    av1_cost_tokens_from_cdf(o_dc_sign + ctx * 2, dc_sign_cdf + ctx * 3, NULL);
  for (int ctx = 0; ctx < LEVEL_CONTEXTS; ++ctx) {
    int *lps = o_lps + ctx * (COEFF_BASE_RANGE + 1 + COEFF_BASE_RANGE + 1);
    int br_rate[BR_CDF_SIZE];
    int prev_cost = 0, i, j;
    av1_cost_tokens_from_cdf(br_rate, br_cdf + ctx * 5, NULL);
    for (i = 0; i < COEFF_BASE_RANGE; i += BR_CDF_SIZE - 1) {
      for (j = 0; j < BR_CDF_SIZE - 1; j++) lps[i + j] = prev_cost + br_rate[j];
      prev_cost += br_rate[j];
    }
    lps[i] = prev_cost;
    lps[0 + COEFF_BASE_RANGE + 1] = lps[0];
    for (i = 1; i <= COEFF_BASE_RANGE; ++i)
      lps[i + COEFF_BASE_RANGE + 1] = lps[i] - lps[i - 1];
  }
}

/* ---- ext-tx derivation for av1_write_tx_type (real functions/tables) ------- */
#include "av1/common/blockd.h"
#include "av1/common/entropymode.h"

/* out[0]=set_type out[1]=num out[2]=eset out[3]=square_tx_size
 * out[4]=symb(av1_ext_tx_ind) out[5]=used(av1_ext_tx_used) out[6]=intra_dir */
void shim_ext_tx_derive(int tx_size, int is_inter, int reduced, int tx_type,
                        int use_fi, int fi_mode, int mode, int *out) {
  const TxSetType st = av1_get_ext_tx_set_type(tx_size, is_inter, reduced);
  out[0] = (int)st;
  out[1] = av1_num_ext_tx_set[st];
  out[2] = get_ext_tx_set(tx_size, is_inter, reduced);
  out[3] = txsize_sqr_map[tx_size];
  out[4] = av1_ext_tx_ind[st][tx_type];
  out[5] = av1_ext_tx_used[st][tx_type];
  out[6] = use_fi ? fimode_to_intradir[fi_mode] : mode;
}

/* ---- trellis per-coefficient cost helpers (real txb_rdopt_utils.h) --------- */
static void tc_build(LV_MAP_COEFF_COST *cc, const int *base_eob,
                     const int *base, const int *dc_sign, const int *lps) {
  memcpy(cc->base_eob_cost, base_eob, sizeof(cc->base_eob_cost));
  memcpy(cc->base_cost, base, sizeof(cc->base_cost));
  memcpy(cc->dc_sign_cost, dc_sign, sizeof(cc->dc_sign_cost));
  memcpy(cc->lps_cost, lps, sizeof(cc->lps_cost));
}

int shim_two_coeff_cost_simple(int ci, int abs_qc, int coeff_ctx,
                               const int *base, const int *lps, int bhl,
                               int tx_class, const uint8_t *levels,
                               int *cost_low) {
  LV_MAP_COEFF_COST cc;
  memset(&cc, 0, sizeof(cc));
  memcpy(cc.base_cost, base, sizeof(cc.base_cost));
  memcpy(cc.lps_cost, lps, sizeof(cc.lps_cost));
  return get_two_coeff_cost_simple(ci, abs_qc, coeff_ctx, &cc, bhl,
                                   (TX_CLASS)tx_class, levels, cost_low);
}

int shim_coeff_cost_eob(int ci, int abs_qc, int sign, int coeff_ctx,
                        int dc_sign_ctx, const int *base_eob, const int *dc_sign,
                        const int *lps, int bhl, int tx_class) {
  LV_MAP_COEFF_COST cc;
  memset(&cc, 0, sizeof(cc));
  tc_build(&cc, base_eob, cc.base_cost /*unused*/, dc_sign, lps);
  memcpy(cc.base_eob_cost, base_eob, sizeof(cc.base_eob_cost));
  memcpy(cc.dc_sign_cost, dc_sign, sizeof(cc.dc_sign_cost));
  memcpy(cc.lps_cost, lps, sizeof(cc.lps_cost));
  return get_coeff_cost_eob(ci, abs_qc, sign, coeff_ctx, dc_sign_ctx, &cc, bhl,
                            (TX_CLASS)tx_class);
}

int shim_coeff_cost_general(int is_last, int ci, int abs_qc, int sign,
                            int coeff_ctx, int dc_sign_ctx, const int *base_eob,
                            const int *base, const int *dc_sign, const int *lps,
                            int bhl, int tx_class, const uint8_t *levels) {
  LV_MAP_COEFF_COST cc;
  memset(&cc, 0, sizeof(cc));
  memcpy(cc.base_eob_cost, base_eob, sizeof(cc.base_eob_cost));
  memcpy(cc.base_cost, base, sizeof(cc.base_cost));
  memcpy(cc.dc_sign_cost, dc_sign, sizeof(cc.dc_sign_cost));
  memcpy(cc.lps_cost, lps, sizeof(cc.lps_cost));
  return get_coeff_cost_general(is_last, ci, abs_qc, sign, coeff_ctx,
                                dc_sign_ctx, &cc, bhl, (TX_CLASS)tx_class,
                                levels);
}
