# Inter var-tx coeff arm (KB-15 / INTER-ENCODE chunk 1) — working notes

Config target (KB-15 witness): intrabc, ALLINTRA speed-0, bd8, screen KEY, cq48 -> base_qindex 192, sub-720p (196x196). intrabc is is_inter -> av1_pick_recursive_tx_size_type_yrd.

## C call tree (reference/libaom/av1/encoder/tx_search.c)
av1_txfm_search(3795) -> av1_pick_recursive_tx_size_type_yrd(3553)
  -> [predict_skip_txfm(3596) skip arm]  OR  select_tx_size_and_type(3433)
     -> loop max-tx-size units -> select_tx_block(2601) recursion
        -> try_tx_block_no_split(2406): leaf tx_type_rd(2393)->search_tx_type(2079) + skip decision + txfm_partition cost
        -> try_tx_block_split(2454): sub_txs, recurse depth+1, sum + split cost
        -> pick min(no_split.rd, split.rdcost); update inter_tx_size[16] + tx_type_map + contexts
leaf search_tx_type(2079): fwd tx+quant+optimize+cost+dist per tx-type; recon_intra is !is_inter-gated (NO recon feedback for inter -> residual FIXED).
MAX_VARTX_DEPTH=2 (enums.h:56). init_depth=0 (sub-720p spd0).

## ACTIVE prunes for witness (MUST port faithfully):
- USE_FULL_RD / FTXS_NONE (full search).
- model_based_prune_tx_search_level=1: early-return at 3563-3565 when ref_best_rd!=MAX and (model_rd*3)>>3 > ref_best_rd. model_rd_sb_fn[MODELRD_TYPE_TX_SEARCH_PRUNE].
- adaptive_txb_search_level=1: select_tx_block 2652-2661 (invalidate if no_split.rd-(>>2)>ref_best_rd; try_split=0 if no_split.rd-(>>3)>prev_level_rd) + tx-type break 2353-2357.
- txb_split_cap=1: eob==0 after no-split -> try_split=0 (2662-2664).
- ml_tx_split_thresh=8500 (bd8): ml_predict_tx_split(1755) NN; try_split=0 if score < -8500. weights av1_tx_split_nnconfig_map (tx_prune_model_weights.h). get_mean_dev_features(1709) EXISTS in port (tx_search.rs).
- prune_2d_txfm_mode=TX_TYPE_PRUNE_1: prune_tx_2D(1541) NN fires for is_inter when num_allowed>5 (ALL16 on <=8x8-sqr, DTT9 on 16x16-sqr). weights tx_prune_model_weights.h.
- FULL inter ext-tx set (reduced_tx_set_used=0): av1_get_ext_tx_set_type(is_inter=1): 64->DCTONLY(eset0); 32x32-sqr->DCT_IDTX(eset1); 16x16-sqr->DTT9_IDTX_1DDCT(eset2); <=8x8-sqr->ALL16(eset3).
- enable_flip_idtx/tx64/rect_tx all ON. default_inter_tx_type_prob_thresh=INT_MAX (full set, no forced type). rd_model=FULL_TXFM_RD.

## INERT (skip): mb_rd_hash=0, prune_tx_size_level=0, prune_inter_tx_split_rd_eval_lvl=0(+intrabc hard-skip), skip_tx_search=0, refine_fast(fast only), use_reduced_intra_txset(intra-set only), fast_intra/inter(MODE_EVAL only).

## Rust reuse surface
- Leaf primitives (SHARED): xform_quant (lib.rs:296), xform_quant_optimize (lib.rs:526), cost_coeffs_txb (aom-txb cost.rs:107), get_tx_type_cost(...,is_inter,...) (aom-txb ext_tx.rs:300), dist_block_tx_domain_qm / dist_block_px_domain (tx_search.rs). ext_tx_set_type/get_ext_tx_set/EXT_TX_USED (aom-txb ext_tx.rs). AV1_EXT_TX_USED_FLAG[6]=[0x0001,0x0201,0x020F,0x0E0F,0x0FFF,0xFFFF] (tx_search.rs:33). DCT_ADST_TX_MASK=0x000F (tx_search.rs:55).
- Intra leaf template: search_tx_type_intra (tx_search.rs:1123). Plan: write search_tx_type_inter (inter subset: inter mask + is_inter cost, NO predict_dc/palette/filter_intra) in new var_tx.rs -> zero risk to intra path.
- Contexts: txfm_partition_context/_update (aom-entropy partition.rs:1303/1328), get_txb_size_index (partition.rs:1377), get_txb_ctx (aom-txb entropy_ctx.rs:58), txb_entropy_context (entropy_ctx.rs:107), get_search_init_depth_intra (tx_search.rs:2414). TXFM_CONTEXT = &mut[u8].
- Pack: write_tx_size_vartx (aom-entropy partition.rs:1401, consumes inter_tx_size[16] + TXFM_CONTEXT arrays) VALIDATED (ref_write_tx_size_vartx, partition_diff.rs). write_inter_txfm_size (partition.rs:3498). write_coeffs_txb_full (aom-txb write.rs:239, is_inter=true routes inter ext-tx). Wire at pack.rs:499 (tx-size) + 531 (coeff).
- derive_real_costs inter cost fix: DONE (real_costs.rs ~155, flatten kf.inter_ext_tx).

## IntraBC wiring points
- intrabc_search.rs:1890 `if !luma_skip || chroma_sse!=0 { continue; }` -> coeff-arm candidate plugs in.
- rd_pick.rs:422-474 carries ibc_skip (always true today) -> carry skip_txfm=false + var-tx data.
- encode_sb.rs:555-620 encode_b_intra_dry intrabc arm (skip: empty txbs) -> coeff arm produces real txbs in var-tx order.
- pack.rs:499 (skip uniform tx-size for intrabc) + 531 (coeff write) -> inter var-tx write.

## Differential plan
- NN kernels (ml_predict_tx_split, prune_tx_2D): export small real-C shims (light marshal: src_diff+tx_size+scalars) -> tier-1 diff. Homes: rd_shim.c. NNs = highest transcription risk.
- model_rd tx-search-prune: real-C shim or reuse model_rd port.
- Recursion glue + leaf: e2e witness vs real aomenc (tier-1) + optional facade c_select_tx_block (tests/common) for localization.
- Real pick_recursive shim (heavy) optional gold-standard.

## Landing sequence
1a. DONE (db90148) — inter leaf search_tx_type_inter + get_tx_mask_inter + trellis_rdmult_inter_y;
    leaf differential-locked vs REAL C kernels (var_tx_leaf_diff.rs, all 19 tx sizes).
1e. DONE (3b9278f) — recursion (select_tx_size_and_type/select_tx_block/try_no_split/try_split);
    differential-locked vs independent C transcription (var_tx_recursion_diff.rs), depth-2 splits.
    [prunes gated OFF on both sides.]
0.  DONE (44bc51c) — derive_real_costs inter ext-tx cost fill (§5 #C).
--- REMAINING (prunes gated off in var_tx.rs today -> recursion over-searches vs C on the witness) ---
1b. ml_predict_tx_split NN + real-C diff. weights: reference/libaom/av1/encoder/tx_prune_model
    _weights.h av1_tx_split_nnconfig_{4x8,8x8,8x16,16x16,32x32,64x64,4x16,16x32,32x64,8x32,...}.
    get_mean_dev_features EXISTS (tx_search.rs). Wire: select_tx_block try_split gate (var_tx.rs,
    the "ml_predict_tx_split: NOT yet ported" comment). C: tx_search.c:2673-2680 + :1755.
1c. prune_tx_2D NN + real-C diff. C: tx_search.c:1541. Two NNs (av1_tx_type_nnconfig_map_hor/ver)
    + get_energy_distribution_finer + av1_get_horver_correlation_full + av1_nn_fast_softmax_16 +
    get_adaptive_thresholds + av1_sort_fi32_8/16. Wire: get_tx_mask_inter multi-type arm (fires
    num_allowed>5 -> ALL16 on <=8x8-sqr, DTT9 on 16x16-sqr). Reorders txk_map + prunes mask.
1d. model_based_tx_search_prune + diff. C: tx_search.c:3532/3563. model_rd_sb_fn[MODELRD_TYPE_TX
    _SEARCH_PRUNE]. Wire: pick_recursive_tx_size_type_yrd early-return (only ref_best_rd!=MAX).
2.  pack wiring: write_tx_size_vartx (partition.rs:1401, consumes inter_tx_size[16]; ref-validated)
    + per-leaf inter coeff write (write_coeffs_txb_full is_inter=true) at pack.rs:499/531.
3.  intrabc integration (intrabc_search.rs:1890 coeff-arm candidate; rd_pick.rs:422-474 carry
    skip_txfm=false + var-tx; encode_sb.rs encode_b_intra_dry intrabc arm real txbs; chroma-eob-0
    skip check) + e2e witness gate (rd_close_intrabc::intrabc_dv_search_pinned) -> resolve KB-15.

## Reuse pointers for the NN ports
- get_mean_dev_features: crates/aom-encode/src/tx_search.rs (from KB-10 intra-tx NN).
- NN eval (av1_nn_predict): KB-10 ported an intra-tx-depth NN (intra_tx_nn_weights.rs +
  ml_predict_intra_tx_depth_prune in tx_search.rs) — reuse the nn_predict + prec-reduce pattern.
- ref_nn_predict exists in aom-sys-ref (KB-10 intra_tx_nn_diff uses it) — reuse for the NN diffs.
- The recursion's try_split hook + get_tx_mask_inter multi-type arm are where the prunes wire in
  (both have "NOT yet ported"/"applied by the caller" comments marking the insertion points).

## 2026-07-19 — integration findings (chunk 2/3: pack + intrabc)

### `model_based_tx_search_prune` is PROVABLY INERT for intrabc (item 1 closed, no port needed)
`rd_pick_intrabc_mode_sb` calls `av1_txfm_search(..., ref_best_rd = INT64_MAX)` — the constant is
HARDCODED at rdopt.c:3611, NOT the incoming `best_rd` (which is only used for the post-hoc
`rd_stats_yuv.rdcost < best_rd` compare at :3615). `av1_pick_recursive_tx_size_type_yrd` then gates
the prune on `ref_best_rd != INT64_MAX` (tx_search.c:3562-3565), so it can never fire on the intrabc
path. Same reasoning voids EVERY early-exit inside `av1_txfm_search` (:3811, :3844) and
`av1_txfm_uvrd` (:3737) for intrabc. This is a source proof, not an empirical guess.

### CHROMA for an inter/intrabc block
- **UNIFORM tx, never var-tx.** `encode_block_inter` short-circuits `plane != 0` to
  `av1_get_max_uv_txsize(bsize, ssx, ssy)` (encodemb.c:495-505); `mbmi->inter_tx_size[]` is luma-only.
  There is no `mbmi->uv_tx_size` field in v3.14.1 — chroma's size is purely derived.
- **NO tx-type search.** `get_tx_mask`'s `if (plane)` arm (tx_search.c:1841-1847) pins
  `txk_allowed = av1_get_tx_type(...)`, collapsing `allowed_tx_mask` to one bit (:1872-1874). The
  full-set branch literally asserts `plane == 0` (:1882). The type is INHERITED from the co-located
  LUMA txb: `xd->tx_type_map[(blk_row << ssy) * stride + (blk_col << ssx)]` (blockd.h:1296-1301),
  DCT_DCT if not in the chroma set.
- **`rd_stats_uv->skip_txfm` is EOB-based**, not SSE-based: `this_rd_stats.skip_txfm &=
  !x->plane[plane].eobs[block]` (tx_search.c:3126), AND-reduced by `av1_merge_rd_stats`. This is the
  "chroma eob-0" check KB-15 called for; the old `chroma_sse == 0` gate was a strict SUBSET of it.
- **Trellis rd-mult differs on chroma**: `plane_rd_mult_chroma[is_inter=1][PLANE_TYPE_UV] = 10`
  (encodetxb.h:266-269) under the allintra `use_chroma_trellis_rd_mult = 1` (speed_features.c:370),
  vs 20 without it. Inter LUMA is 16 either way — which is why the sf looked like a no-op.
- `perform_best_rd_based_gating_for_chroma` is **0** at speed-0 ALLINTRA (init_inter_sf,
  speed_features.c:2391; raised only at GOOD speed >= 3, :1311) so `chroma_ref_best_rd` never tightens.
- The chroma PREDICTOR is the intrabc DV copy, built ONCE before the RD call by
  `av1_enc_build_inter_predictor(..., 0, num_planes-1)` (rdopt.c:3601). `mbmi->uv_mode = UV_DC_PRED`
  (:3596) is a bitstream placeholder only — it never drives a predictor because `block_rd_txfm`'s
  intra-predict branch is `!is_inter`-gated (:3084).

### Walk ORDERS (they differ between encode and write — both ported)
- ENCODE (`av1_encode_sb`, encodemb.c:657-699): **plane-outer** (Y, U, V), each plane mu-64 chunked,
  raster over `max_tx_size` units, luma recursing via `encode_block_inter`.
- WRITE (`write_tokens_b` inter arm, bitstream.c:1444-1471): **64-chunk outer in LUMA mi units, planes
  INTERLEAVED per chunk** (same shape as the intra `av1_write_intra_coeffs_mb`), each plane's sub-walk
  being `write_inter_txb_coeff` (:1395). Identical to the encode order for blocks <= 64x64 (one chunk).
- The tx-SIZE syntax (`write_tx_size_vartx`, bitstream.c:1542-1552) both READS and UPDATES the
  txfm-partition contexts. The encode walk therefore stamps them ONLY on a dry run
  (`if (dry_run) tx_partition_set_contexts`, partition_search.c:559-562) — double-stamping on the
  OUTPUT path would corrupt the contexts.

### `set_skip_txfm` has a NONZERO intermediate rate
tx_search.c:245-281: besides `skip_txfm = 1` and `dist = sse = ROUND_POWER_OF_TWO(dist, 2*(bd-8)) << 4`,
it sets `rd_stats->rate = zero_blk_rate * n_max_tx_blocks` (zero_blk_rate = `txb_skip_cost[ctx][1]` at
the BLOCK-ORIGIN ctx). That rate is discarded when the final decision is skip, but it is an INPUT to
the skip-vs-non-skip comparison (:3863-3868), so a luma-predict-skip block can still come out non-skip
once chroma is costed.
