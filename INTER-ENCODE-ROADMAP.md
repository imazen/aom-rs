# aom-rs — Inter-Frame ENCODE Roadmap

Scope: the **inter-frame ENCODER** — byte-exact bitstream vs `aomenc` for inter (P/B)
frames, the Gate-2 analog for "the rest". Single-frame KEY/intra ENCODE is byte-complete
across `--cpu-used 0..9` (Gate-2 for intra; KB-6 real content 30/30, QM/CDEF/LR done). This
document maps the gap to inter-frame encode and decomposes it into ordered,
smallest-demoable-first chunks.

Reference tree: `/root/aom-rs/reference/libaom` (v3.14.1, git 03087864). Port crates:
`crates/aom-{encode,entropy,convolve,txb,transform,quant,dist,intra,cdef,loopfilter,restore,decode}`.
Companion doc: the inter-**decode** roadmap (`INTER-ROADMAP.md` in the agent worktrees) — the
encoder's byte-exact verification *depends on* the inter decoder it maps (see §3, §6).

Priority of evidence (project methodology): real exported C fn > synthetic-facade-over-real-fn
> verbatim transcription. Every new numeric kernel lands with a differential vs the REAL C.

---

## 0. Executive summary

**This is the largest remaining track** — motion estimation + full inter RD mode decision +
reference-frame management + compound/OBMC/warp/interintra + rate-control/GOP/TPL/2-pass. It is
a multi-month effort. Three facts make it tractable:

1. **The head-start is enormous on the entropy/symbol/ME-full-pel side** (much larger than the
   inter *decoder's* head-start was):
   - **Motion estimation full-pel is DONE** — the KB-15 intrabc landing shipped the NSTEP
     `full_pixel_diamond`, `diamond_search_sad` (coarse→fine SAD), `full_pixel_exhaustive` mesh,
     `set_mv_search_range`, the MV cost model (`mv_cost`/`mvsad_err_cost`/`mv_err_cost`), `sad_wxh`,
     `variance_wxh`, and the source-frame hash — all in `crates/aom-encode/src/intrabc_search.rs`
     (1921 LOC). IntraBC *is* "motion search against the current frame"; the same diamond/mesh
     drives inter ME.
   - **The entire inter SYMBOL layer is byte-exact** in `aom-entropy` (`partition` module):
     `write_inter_mode`, `write_ref_frames` (single + compound cascade), `write_drl_mode`,
     the **MV coder** (`av1_encode_mv`/`encode_mv_component`), `write_motion_mode`, compound-type
     group, interintra, `write_mb_interp_filter`, `write_tx_size_vartx`, `write_skip_mode`,
     `write_is_inter`, + the full neighbour pred-context layer (`av1_get_reference_mode_context`,
     `single_ref_p*`, `comp_*`, `intra_inter`, etc.). All od_ec-driven, diffed vs pristine-C.
   - **MV-candidate scan** (`dv_ref.rs`) ports `setup_ref_mv_list`/`scan_{row,col,blk}_mbmi`/
     `add_ref_mv_candidate`/`av1_find_best_ref_mvs`, byte-exact vs exported C, reduced to the
     single-ref path — the same generalization the inter *decoder* needs.
   - **Inter ext-tx CDF is already in the frame context** — `KfFrameContext.inter_ext_tx`
     (`partition.rs:5849`) is populated with `DEFAULT_INTER_EXT_TX`; only `derive_real_costs`
     (`real_costs.rs:155`) still stubs it with a zero row.
   - **The full intra RD machinery** (partition search, mode search, tx-type search,
     rd_pick_partition, speed features 0-9, the two-pass winner-mode structure, `encode_b*` /
     `pack_*` search==pack split, `av1_txfm_search` dispatch, `block_error`/`model_rd`/
     `pixel_distortion` RD primitives, mode-cost tables) all exist and are byte-exact.

2. **The simplest inter config collapses the search** (see §3): `--lag-in-frames=0` +
   `--end-usage=q` + disabling OBMC/warp/interintra/compound/dual-filter/global-motion via CLI
   reduces the inter mode search to **single-ref translational NEWMV/NEAREST/NEAR/GLOBALMV +
   a reduced interp-filter search + inter var-tx**. `lag=0` structurally kills alt-ref, TPL,
   temporal filtering, and 2-pass.

3. **Decode-both verification** localizes divergence precisely: the inter DECODER decodes BOTH
   the port's re-encoded stream AND `aomenc`'s stream of the *same invocation*, isolating the
   first differing SB/mode/MV/coeff — plus per-kernel differentials vs exported C for each new
   numeric kernel. **The inter decoder's single-ref translational skeleton is already byte-exact
   on `origin/main`** — `93b92ec` (inter walking skeleton, frame 1 byte-exact,
   `av1-1-b8-01-size-64x64`), `bc2d1bd` (16×16 ratchet), `50816e5` (partial-edge 64×66),
   `cfd39e0` (4-tap sub-8 interp), `835b0c0` (switchable-interp neighbour ctx), `cdba774`
   (generalized single-ref inter MV scan `find_inter_mv_refs`, byte-exact vs C). So the encode
   chunk-2 verification dependency (§6) is **already met**, and the shared ref-mv-list
   generalization (§5 #B) has **already landed** — the encoder can reuse both immediately.

**What is genuinely NET-NEW** (concentrated, not diffuse): (a) **subpel** motion search
(`av1_find_best_sub_pixel_tree`) — full-pel is done, subpel is absent; (b) the **encoder-side
inter MC wiring** (`av1_enc_build_inter_predictor` → the shared `reconinter_template` chain →
`aom-convolve`); (c) the **inter var-tx coeff arm** (`av1_pick_recursive_tx_size_type_yrd` + the
`select_tx_block` recursion — already scoped as KB-15's next step); (d) the **`handle_inter_mode`
RD orchestration** + `av1_rd_pick_inter_mode_sb` driver + ref-frame RD loop (the "brain"); (e)
**reference-frame management** on the encode side (GOP structure, ref selection, RD ref buffers);
(f) **multi-frame RC** (CQ inter q — the port's RC is single-KEY only); (g) the **multi-frame
encode harness** (the current one asserts a single frame OBU). Then the tool chunks (GM
estimation, OBMC RD, warp RD, compound search, interintra, skip_mode) and the big optional
subsystems (alt-ref/GOP/temporal-filter, TPL, 2-pass, nonrd-inter for speeds 8/9).

**First byte-exact target:** a 2-frame `[KEY, P]` low-delay clip at a controlled `aomenc`
invocation, single-ref translational, all optional inter tools CLI-disabled — see §3.

---

## 1. Head-start inventory (encode side — what already exists, byte-exact-or-not)

| Building block | Location | State | Byte-exact? |
|---|---|---|---|
| **Full-pel motion search** (diamond NSTEP + mesh) | `aom-encode/src/intrabc_search.rs` — `full_pixel_diamond` (:1319), `diamond_search_sad` (:1252), `full_pixel_exhaustive` (:1346), `set_mv_search_range` (:1099), `FullPelSearch` (:1166) | Landed for intrabc DV search; SAD-metric | **YES** — geometry unit-locked (`nstep_config_matches_c`, `diamond_finds_exact_repeat`); ME is intrabc's proven core |
| **MV cost model** | intrabc_search.rs — `mv_cost` (:556), `mvsad_err_cost` (:593), `mv_err_cost` (:579), `DvCosts`/`fill_dv_costs` (:418/:536) | Landed | **YES** (intrabc gate) — generalizes to the inter `MV_COST_PARAMS` |
| **SAD / variance kernels** | `aom-dist` (SAD/variance/subpel-variance 22 sizes, lowbd+highbd) + intrabc_search.rs `sad_wxh`/`variance_wxh` | Landed | **YES** — full SAD/distortion family diffed vs exported C (both bit depths) |
| **MV coder** (bitstream) | `aom-entropy/src/partition.rs` — `av1_encode_mv`, `encode_mv_component` (:466), `av1_encode_dv` | Landed | **YES** — 300k-case diffs vs pristine-C od_ec |
| **Inter mode-info symbols** | partition.rs — `write_inter_mode`, `write_inter_compound_mode`, `write_ref_frames`, `write_drl_mode`, `write_motion_mode`, compound-type, interintra, `write_mb_interp_filter`, `write_tx_size_vartx`, `write_skip_mode`, `write_is_inter` | Landed | **YES** — each diffed vs pristine-C od_ec + update_cdf |
| **Neighbour pred-context layer** | partition.rs — `get_intra_inter_context`, `get_reference_mode_context`, `single_ref_p1..p6`, `comp_ref/comp_bwdref/uni_comp` contexts, `av1_collect_neighbors_ref_counts` | Landed | **YES** — facade oracles over exported C |
| **MV-candidate / ref-mv list** | `aom-entropy/src/dv_ref.rs` — `setup_ref_mv_list`, `scan_{row,col,blk}_mbmi`, `add_ref_mv_candidate`, `av1_find_best_ref_mvs` | Ported, reduced to single-ref (compound/temporal/mode_context arms dropped as dead-for-intrabc) | **YES** (single-ref) — diffed vs exported `av1_find_mv_refs`; the SAME generalization inter-decode needs |
| **Inter ext-tx CDF** | `partition.rs:5849` `KfFrameContext.inter_ext_tx` = `DEFAULT_INTER_EXT_TX` | Present in frame ctx | Table present; **`derive_real_costs` (real_costs.rs:155) still uses a zero stub** — one-liner to source it |
| **Inter-prediction convolve** (single-ref SR) | `aom-convolve/src/lib.rs` — `convolve_x_sr` (:75), `convolve_2d_sr` (:98), `convolve_y_sr` (:145); EIGHTTAP_REGULAR lowbd | Standalone crate, **NOT wired** into encode/decode; SMOOTH/SHARP + highbd + compound (dist-wtd) variants absent | Transcription of `av1_convolve_*_sr_c`; **needs a differential vs exported C** before trust (shared item with inter-decode 1d) |
| **Inter var-tx WRITE** | partition.rs `write_tx_size_vartx` (recursive) | Landed, **unused by `pack`** | **YES** (symbol) — the pack wiring is net-new |
| **Full intra RD engine** | `aom-encode` (36.8k LOC) — `rd_pick_partition_real`, `leaf_pick_sb_modes`, `av1_txfm_search`, tx-type search, speed features 0-9, `encode_b*`/`pack_*`, `block_error`/`model_rd`/`pixel_distortion`, mode-cost tables | Byte-complete for intra KEY, cpu 0-9 | **YES** (Gate-2 intra) — the inter mode loop plugs into this |
| **Fixed-Q (CQ) qindex, lone KEY** | `aom-encode/src/rc.rs` — `base_qindex_from_cq` | Single-KEY `AOM_Q` branch only | **YES** (`qindex_from_cq_diff`) — the multi-frame inter CQ path is net-new |

**What is NOT present at all (encode side):** subpel motion search
(`av1_find_best_sub_pixel_tree` + upsampled-pred); the encoder-side inter predictor build
(`av1_enc_build_inter_predictor` wiring + buffers); the `handle_inter_mode` RD orchestration +
`av1_rd_pick_inter_mode_sb` driver + `set_params_rd_pick_inter_mode` (ref-frame RD loop); the
inter var-tx coeff arm (`av1_pick_recursive_tx_size_type_yrd` recursion + `prune_tx_2D` +
`ml_predict_tx_split` + the var-tx pack wiring); `av1_single_motion_search` /
`av1_joint_motion_search` / `av1_compound_single_motion_search`; the interp-filter RD search
(`av1_interpolation_filter_search`); motion-mode RD (`motion_mode_rd` OBMC/WARP arms);
compound-type RD (`av1_compound_type_rd` + wedge/diffwtd search); interintra RD; reference-frame
management (GOP structure, ref selection, RD ref buffers, refresh); multi-frame RC
(`rc_pick_q_and_bounds_no_stats_cq` inter path); global-motion *estimation*
(`av1_compute_global_motion_facade`); temporal filtering (`av1_temporal_filter`); TPL
(`tpl_model.c`); 2-pass (`firstpass.c` + `pass2_strategy.c`); the multi-frame encode harness;
the nonrd inter pickmode (`av1_nonrd_pick_inter_mode_sb`, speeds 8/9).

---

## 2. C-path gap map (subsystem → C functions → port need)

All C refs under `reference/libaom/av1/encoder` unless noted. Line numbers verified v3.14.1.

### 2.1 Encode driver / frame structure / reference management
- `av1_encode_strategy` (encode_strategy.c:1250) — top-level per-frame driver: frame-type/params,
  ref-frame flags, primary-ref choice, then `denoise_and_encode` (:728) / RT params.
- `get_ref_frame_flags` (called encode_strategy.c:1657), `choose_primary_ref_frame` (:168),
  `is_altref_enabled(lag_in_frames, enable_auto_arf)` (encoder.h:4110, needs `lag>=ALT_MIN_LAG`),
  the low-delay predicate (`has_no_stats_stage && lag_in_frames==0`, encoder.h:4174).
- GOP: `av1_gop_setup_structure` (gop_structure.c:896) → `construct_multi_layer_gf_structure`
  (:540); `define_gf_group` (pass2_strategy.c:2604) / `define_gf_group_pass0` (:2211, the
  no-lookahead path). For `lag=0` the GF group is trivial (each P references the last decoded).
- **Port need:** a `RefFrame` buffer + `ref_frame_map[8]` + refresh on the encode side (RD refs =
  border-extended recon of prior frames); a minimal frame loop that codes KEY then P; the
  frame-header inter-field **WRITE** path (`write_uncompressed_header_obu` inter branch — the port
  has the read side in `header.rs` + the write pieces anchored per STATUS; the inter WRITE
  assembly + ref-signaling values are the net-new part). For `lag=0`, skip GOP/alt-ref entirely.

### 2.2 Rate control (fixed-Q inter)
- `av1_rc_pick_q_and_bounds` (ratectrl.c:2350) → `rc_pick_q_and_bounds_no_stats_cq`
  (:1791, the `--end-usage=q` CQ path) / `_no_stats` (:1588); `av1_rc_regulate_q` (:1138);
  `av1_rc_postencode_update` (:2444).
- **Port need:** extend `rc.rs` beyond the lone-KEY `AOM_Q` branch to the multi-frame CQ path
  (per-frame `active_best/worst_quality`, the P-frame qindex derivation). For a pinned `--cq-level`
  this is small and differential-testable vs `av1_rc_pick_q_and_bounds`. Full CBR/VBR RC is a
  separate large item (out of the first-target envelope).

### 2.3 Inter MV prediction (encode side)
- `av1_find_mv_refs` (mvref_common.c:788) — called from `set_params_rd_pick_inter_mode`
  (rdopt.c:4403) and per-ref in `handle_inter_mode`; `av1_mv_pred` (mcomp.c) for the search start
  MV; `av1_mode_context_analyzer` (mvref_common.h) for the inter-mode CDF context; `av1_drl_ctx`;
  `av1_collect_neighbors_ref_counts`.
- **Port need:** generalize `dv_ref.rs` to inter single-ref — restore `mode_context`/`newmv_count`
  (dropped as dead), add sign-bias + global-MV candidates. **Shared verbatim with inter-decode
  §2.5** — build once, both tracks consume. Compound + temporal extensions defer to later chunks.

### 2.4 Motion estimation (the search)
- Full-pel: `av1_full_pixel_search` (mcomp.c:1768) → `full_pixel_diamond` / `full_pixel_exhaustive`
  (:1615) / `av1_refining_search_8p_c` (:1696). **PORT HAS THIS** (intrabc_search.rs) — needs the
  inter wiring (ref-frame reads instead of current-frame reads; the inter `MV_COST_PARAMS`).
- **Subpel (NET-NEW): `av1_find_best_sub_pixel_tree` (mcomp.c:3266)** + the pruned variants
  (`_pruned` :3120, `_pruned_more` :3026); `av1_get_mvpred_sse` (:3963); the upsampled-prediction
  path (`aom_upsampled_pred` / the subpel-variance cost). Uses the subpel-variance kernels (port
  HAS these in aom-dist, both bit depths).
- Orchestration: `av1_single_motion_search` (motion_search_facade.c:120), `av1_mv_pred` (mcomp.c),
  `av1_set_mv_search_range`.
- **Port need:** wire the existing full-pel search to reference frames; **port the subpel tree
  search** (the single biggest ME gap) + `av1_single_motion_search`. **First differential:
  `av1_find_best_sub_pixel_tree` vs exported C.**

### 2.5 Inter mode decision / RD (the brain)
- `av1_rd_pick_inter_mode_sb` (rdopt.c, entry ~6180; `set_params_rd_pick_inter_mode` call at
  :6202) — the per-SB inter RD entry; sibling of the ported `av1_rd_pick_intra_mode_sb`.
- `set_params_rd_pick_inter_mode` (rdopt.c:4331) — ref-frame loop setup, mv-refs per ref
  (:4403), mode-skip mask.
- `handle_inter_mode` (rdopt.c:3063) — the per-mode RD. Documented 6-step structure
  (rdopt.c:3153-3161): (1) get/create MV (`handle_newmv` :1317 → motion search), (2) compound-type
  search (`av1_compound_type_rd` compound_type.c:1234), (3) interp-filter search
  (`av1_interpolation_filter_search` interp_search.c:674), (4) build inter predictor, (5) motion
  mode RD (`motion_mode_rd` rdopt.c:1539 → SIMPLE/OBMC/WARP), (6) update best. DRL loop over
  `ref_set` ref-mv indices (`get_drl_refmv_count`, `ref_mv_idx_to_search`).
- Inter mode set: `NEARESTMV/NEARMV/GLOBALMV/NEWMV` (single) + `NEAREST_NEARESTMV..NEW_NEWMV`
  (compound) (enums.h:337-349); `INTER_MODES = 4`, `INTER_COMPOUND_MODES = 8`.
- Cost: `cost_mv_ref`, `get_drl_cost`, `ref_frame_cost` — the port has mode-cost table machinery
  (`mode_costs.rs`); the inter mode/drl/ref costs plug in.
- **Port need:** ALL net-new orchestration, but every *leaf* it calls either exists (symbols,
  MV coder, ref-mv list, RD primitives, tx search dispatch) or is a named chunk below. This is the
  integration center of gravity. Start with single-ref, SIMPLE-motion-mode-only, no compound.

### 2.6 Inter block encode: prediction → residual → var-tx → coeff
- **Encoder-side inter MC:** `av1_enc_build_inter_predictor` (reconinter_enc.c:111) →
  `enc_build_inter_predictors` (:54) → the shared `reconinter_template.inc` chain (same as
  decode, `#include`d with the encoder `IS_DEC=0` switch) → `av1_make_inter_predictor`
  (reconinter.c) → `av1_convolve_*_sr`. `av1_build_inter_predictors_for_planes_single_buf` (:271)
  for the search's tmp buffers.
- **Inter var-tx (the KB-15 shared blocker):** `av1_txfm_search` (tx_search.c:3795) dispatches
  `av1_pick_recursive_tx_size_type_yrd` (:3553, inter) vs `av1_pick_uniform_tx_size_type_yrd`
  (:3628, intra — PORTED). Recursion: `select_tx_size_and_type` (:3433) → `select_tx_block`
  (:2601) → `try_tx_block_no_split` (:2406) / `try_tx_block_split` (:2454, recurse to
  `MAX_VARTX_DEPTH`); pruning `prune_tx_2D` (:1541), `ml_predict_tx_split` (:1755).
- **Residual + coeff:** reuse the byte-exact `xform_quant`/`optimize_txb`/`write_coeffs_txb_full`
  pipeline (already inter-capable — bd-independent forward tx, inter ext-tx set); the var-tx WRITE
  is `write_tx_size_vartx` (ported symbol, needs pack wiring); the inter tx-type cost sources from
  `KfFrameContext.inter_ext_tx` (fix the `derive_real_costs` zero stub, real_costs.rs:155).
- **Port need:** wire `aom-convolve` into an encoder MC path (shared build with inter-decode 1d);
  port the var-tx recursion + prunes + pack wiring (**= KB-15's documented next step** — landing it
  unblocks both intrabc-coeff real content AND inter). Differentials: convolve vs
  `av1_convolve_*_sr_c`; `av1_pick_recursive_tx_size_type_yrd` vs exported C.

### 2.7 Interp-filter search
- `av1_interpolation_filter_search` (interp_search.c:674) → `find_best_interp_rd_facade` (:314) →
  `interpolation_filter_rd` (:153); `av1_is_interp_needed`. Dual-filter gate
  (`--enable-dual-filter`). Needs the SMOOTH/SHARP convolve kernels (aom-convolve has only REGULAR).
- **Port need:** the reduced (dual-filter-off) 1-D search over {REGULAR, SMOOTH, SHARP} — folds
  into chunk 2; SMOOTH/SHARP convolve params are a dependency. Full switchable/dual = later chunk.

### 2.8 Compound prediction (≥2 refs)
- Mode: compound arm of `handle_inter_mode`, `av1_joint_motion_search` (motion_search_facade.c:548),
  `av1_compound_single_motion_search` (:757). Type RD: `av1_compound_type_rd` (compound_type.c:1234),
  `pick_interinter_wedge` (:302), `pick_interinter_seg` (:332), `calc_masked_type_cost` (:920);
  contexts `get_comp_group_idx_context`, `get_comp_index_context`.
- MC: `av1_dist_wtd_convolve_*` (16-bit CONV_BUF), masked blend, wedge/diffwtd mask tables (shared
  with inter-decode §2.9/§2.10). **Port need:** net-new; needs a GOP with ≥2 distinct refs (a
  fwd+bwd or longer low-delay). Split average (chunk) from masked (chunk).

### 2.9 Motion modes: OBMC / warped-causal
- `motion_mode_rd` (rdopt.c:1539) OBMC/WARP arms; encoder OBMC build
  (`av1_build_obmc_inter_prediction` + above/left pred), `av1_findSamples` +
  `av1_find_projection` (local warp fit), the warp kernel (`av1_warp_affine`, shared with
  inter-decode §2.10/§2.11). **Port need:** net-new; CLI-disabled in the first target.

### 2.10 Global motion (encode side = ESTIMATION, net-new + large)
- `av1_compute_global_motion_facade` (global_motion_facade.c:406) →
  `compute_global_motion_for_ref_frame` (:79) — feature matching + RANSAC + model fit
  (`av1_compute_global_motion`, global_motion.c); the warp kernel for GLOBALMV MC. `read/write
  _global_motion` (parse/pack) already anchored. **Port need:** net-new estimation; CLI-disabled
  (`--enable-global-motion=0`) in the first target so GLOBALMV = identity only.

### 2.11 Interintra, skip_mode, segmentation-inter
- Interintra RD (`av1_handle_inter_intra_mode`, reconinter_enc build); skip_mode
  (`av1_setup_skip_mode_allowed` + the skip_mode RD — needs fwd+bwd refs);
  `read/write_inter_segment_id` temporal_update. **Port need:** net-new; CLI-disabled / structurally
  absent in the first (single-ref 2-frame) target.

### 2.12 Large optional subsystems (default good-quality parity, NOT the first target)
- **Alt-ref / GOP / lookahead** (`lag>0`): `av1_gop_setup_structure`, `define_gf_group`,
  `av1_temporal_filter` (temporal_filter.c:1616) + `tf_setup_filtering_buffer` (:1258) — the ARF
  is a temporally-filtered synthetic frame. Large.
- **TPL** (`tpl_model.c`): `av1_init_tpl_stats` (:1839), `mc_flow_dispenser` (:1589),
  `av1_mc_flow_dispenser_row` (:1525) — the lookahead mode/MV/SATD propagation that informs
  deltaq + `prune_inter_modes_based_on_tpl`. Large; gated on lookahead.
- **2-pass** (`firstpass.c` + `pass2_strategy.c`): `av1_first_pass` (firstpass.c:1321),
  `av1_get_second_pass_params` (pass2_strategy.c:3949), `av1_twopass_postencode_update` (:4394).
  Large; only for `--pass=2` parity.
- **NonRD inter pickmode** (speeds 8/9): `av1_nonrd_pick_inter_mode_sb` (nonrd_pickmode.c:3278),
  `search_new_mv` (:311), `find_predictors` (:2406) — a SAD/variance model mode search, distinct
  from the rdopt path. Needed for Gate-2 inter cpu-used 8/9.

---

## 3. Simplest inter-encode config (first byte-exact target)

Unlike the inter *decoder* (which only handles tools that APPEAR in a stream), the *encoder* must
SEARCH every tool that COULD appear to reproduce `aomenc`'s RD decision — even tools that lose.
So the first target must **restrict the search via `aomenc` config**, not just pick easy content.

**Verified levers (all real `aomenc` args — `apps/aomenc.c` + `av1/arg_defs.c`):**
- `--lag-in-frames=0` → low-delay one-pass (`has_no_stats_stage && lag_in_frames==0`,
  encoder.h:4174). Structurally kills **alt-ref** (`is_altref_enabled` needs `lag>=ALT_MIN_LAG`,
  encoder.h:4110), **TPL** (needs lookahead), **temporal filtering**, and **2-pass**. Each P-frame
  references the previously decoded frame (forward refs → frame 0 for frame 1).
- `--end-usage=q --cq-level=N` → the fixed-CQ RC path (`rc_pick_q_and_bounds_no_stats_cq`), no
  rate feedback.
- `--enable-obmc=0 --enable-warped-motion=0` → motion mode collapses to `SIMPLE_TRANSLATION`
  only (enums.h:398; `motion_mode_rd` searches only SIMPLE).
- `--enable-global-motion=0` → GLOBALMV = identity (no GM estimation, no warp).
- `--enable-interintra-comp=0 --enable-masked-comp=0 --enable-diff-wtd-comp=0` → no interintra,
  no wedge/diffwtd compound.
- `--enable-dual-filter=0` → interp filter search is 1-D (single filter for both axes).
- `--enable-ref-frame-mvs=0` → no temporal (motion-field) MV — the ref-mv list stays spatial-only.
- A **2-frame input** (`--limit=2`) → frame 1 has exactly ONE decoded reference (frame 0). All ref
  slots resolve to frame-0's buffer → the encoder codes **single-reference** (compound needs 2
  distinct refs); **`skip_mode` is disallowed** (needs fwd+bwd → `skip_mode_allowed=0`). This kills
  compound / masked-compound / skip_mode from the search entirely.

**Resulting first-target search surface:** single-ref translational
NEWMV/NEARESTMV/NEARMV/GLOBALMV(identity), spatial-only ref-mv list, a reduced interp-filter
search over {REGULAR, SMOOTH, SHARP}, and the inter var-tx coeff arm. That is exactly chunk-2
scope. **The `--cq-level` sweep is a built-in difficulty ladder** (mirroring the decode roadmap's
q63→q00): highest cq → near-all GLOBALMV/skip, integer MVs, fewest coeffs, minimal subpel — the
easiest first frame; ratchet down to add NEWMV subpel + coefficient volume.

**Recommended first target:** `aomenc --end-usage=q --cq-level=60 --lag-in-frames=0 --cpu-used=0
--enable-obmc=0 --enable-warped-motion=0 --enable-global-motion=0 --enable-interintra-comp=0
--enable-masked-comp=0 --enable-diff-wtd-comp=0 --enable-dual-filter=0 --enable-ref-frame-mvs=0
--limit=2 <2-frame input>` — frame 0 KEY (already byte-exact), frame 1 the single-ref P.
Then ratchet cq 60 → 48 → 32 → 12 (more NEWMV, subpel, coeffs), then re-enable each disabled tool
as its own later chunk.

**Speed alternative (even smaller first spike, optional):** `--cpu-used=9` routes inter through
the **nonrd pickmode** (`av1_nonrd_pick_inter_mode_sb`) — a SAD/variance-model mode search that is
structurally simpler than the rdopt path (fewer modes, no full tx-type RD). It is a *different*
(realtime) code path, needed anyway for Gate-2 cpu 8/9. The main spine below targets the **rdopt
path (speed 0)** first because it is the reusable quality path and mirrors the decode roadmap;
nonrd-inter is chunk-scoped separately (§4 chunk 15).

**Two caveats to resolve empirically** (via sibling-C encoder instrumentation, the KB-2/3/7
method):
1. Confirm the chosen cq's P-frame uses only chunk-2 tools (no NEWMV subpel at the very highest
   cq is ideal for the first spike; if subpel appears, it is chunk-2's subpel sub-step).
2. Confirm `allow_ref_frame_mvs` is actually off with `--enable-ref-frame-mvs=0` for the frame
   (it should be; verify by instrumenting the C encoder's frame-header derivation).

---

## 4. Ordered chunk decomposition (smallest-demoable-first)

Each chunk: **{C funcs → port target → byte-exact test → deps → size}**. Sizes S/M/L/XL.
Every kernel lands with a differential vs the REAL exported C. **Cross-track dependency:** the
end-to-end byte-exact gate for chunk 2 needs the inter DECODER (inter-decode chunks 1a-1e) green —
see §6. Kernels can be built in parallel against C differentials before the decoder lands.

### Chunk 0 — Infra: multi-frame encode harness + decode-both localizer — LANDED ✅
- **C funcs:** n/a (port infra).
- **Port:** the current encode harness is single-frame (`aom-bench/src/rd_close.rs:164` asserts one
  frame OBU). Add a 2-frame `[KEY, P]` driver that (a) runs `aomenc` at a controlled inter config,
  (b) runs the port's encode, (c) decode-both localizes the first divergent frame/SB/mode/MV
  (generalize `kb6_real_rd_localize.rs` / `decode_diff_*` across ≥2 frames, driving the
  in-progress inter decoder). Reuse `attempt_case_content_uv_sep`.
- **Test:** frame-0 KEY still byte-exact through the 2-frame harness (regression control).
- **Deps:** inter-decode chunk 1 (for the decode-both leg). **Size:** M. Prereq for verifying any
  inter encode.
- **STATUS (LANDED):** the 2-frame `[KEY, P]` harness + decode-both localizer are in tree, gated by
  `aom-bench/tests/inter_harness_chunk0.rs` (6 tests, all green). Pieces:
  - **Multi-frame C-encode reference** — `shim_encode_av1_inter_2frame` (aom-sys-ref `dec_shim.c`,
    append-only) drives the REAL `aom_codec_av1_cx` API at the §3 simplest inter config
    (`--end-usage=q --lag-in-frames=0 --cpu-used=<N> --limit=2` with obmc/warp/global-motion/
    interintra/masked/diff-wtd/dual-filter/ref-frame-mvs disabled, GOOD usage). Rust:
    `aom_sys_ref::ref_encode_av1_inter_2frame`. **GOTCHA (root-caused):** forcing `AOM_EFLAG_FORCE_KF`
    on frame 0 makes `aomenc` code frame 1 as a KEY too (KEY+KEY, seq-header-per-TU) — the CLI relies
    on AUTO-key of the first frame; dropping FORCE_KF (+ a 1/30 timebase) yields the correct KEY+INTER.
    Verified against the real `aomenc` CLI with the identical flags.
  - **Full-stream N-th-frame C decode oracle** — `shim_decode_av1_stream_frame` /
    `aom_sys_ref::ref_decode_av1_stream_frame` (append-only): splits the stream into TUs and decodes
    per-TU on a persistent ctx (one `aom_codec_decode` on a multi-TU buffer only surfaces the first
    TU), so any frame index cross-checks the port frame-by-frame.
  - **Decode-both localizer** — `aom-bench/src/inter_localize.rs`: `first_frameset_divergence`
    (pure comparator over decoded frame-sets → first `(frame, tile, SB, sample)`), `decode_both`
    (drives `aom_decode::frame::decode_frames`, panic-safe, maps an asymmetric decode failure to a
    `DecodeError` divergence), `FrameView` (works over both a port `FrameDecode` and a C
    `RefDecodedFrame`). Validated: 0-divergence on decode-both of one stream; exact-offset on a
    hand-corrupted luma AND chroma sample (SB label maps chroma → luma via subsampling); pins a real
    port-vs-C divergence.
  - **`MultiFrameEncodeCell`** (aom-bench `lib.rs`) — carries the 2-frame `[KEY, P]` source + config;
    `translational(&EncodeCell, dx, dy)` builds frame 1 as a shifted frame 0 (clean single-ref P);
    `c_encode_inter(cdef, lr)` produces the aomenc stream; `frame0_cell()` reuses `EncodeCell`. Ready
    for chunk 2 to plug the port's inter encoder into.
- **KEY FINDING — the port INTER DECODER's envelope on `aomenc`'s simplest-config P (bounds chunk 2):**
  measured port-decode vs C-decode of the same stream, 4:2:0 64×64 dx=3 cq60 (cdef/lr off), per cpu:
  frame 0 (KEY) is **byte-exact at every cpu**; frame 1 (P) is **byte-exact in LUMA at every cpu
  except 1**, and **fully byte-exact at cpu 2 and 5**; at cpu 0/3/4/6 **CHROMA inter reconstruction
  diverges** by small ± (≤5, 2–6 samples) and at cpu 1 luma diverges (±1). Mono (luma-only) and
  zero-MV 4:2:0 are fully byte-exact; the faithful cdef/lr-ON config also decodes byte-exact at cpu 2.
  ⇒ The port inter decoder (validated only on the minimal skip-P `av1-1-b8-01-size-64x64`) does **not
  yet** reconstruct arbitrary-content CHROMA inter (subpel MC / inter residual) byte-exact — a
  **decoder-track gap (aom-decode/aom-inter)**, NOT a harness bug. **Implication for chunk 2:** the
  first byte-exact encode target must stay where the decoder is byte-exact — luma-inter / mono, or the
  in-envelope 4:2:0 cases (zero-MV; cpu 2/5) — OR the decoder chroma-inter path is completed first.
  The localizer is the tool that pins each such divergence to `(frame, SB, sample)` for the drill-down.

### Chunk 1 — Inter var-tx coeff arm (SHARED with KB-15 intrabc)
- **C funcs:** `av1_txfm_search` inter dispatch (tx_search.c:3795),
  `av1_pick_recursive_tx_size_type_yrd` (:3553), `select_tx_size_and_type` (:3433),
  `select_tx_block` (:2601), `try_tx_block_no_split`/`_split` (:2406/:2454), `prune_tx_2D`
  (:1541), `ml_predict_tx_split` (:1755).
- **Port:** the var-tx quadtree recursion + prunes; wire `write_tx_size_vartx` (ported symbol)
  into `pack`; fix `derive_real_costs` to source `inter_ext_tx` from `KfFrameContext.inter_ext_tx`
  (real_costs.rs:155, one-liner). Reuse the byte-exact `xform_quant`/`optimize_txb`/coeff pack.
- **Test:** differential vs exported `av1_pick_recursive_tx_size_type_yrd`; **closes the KB-15
  intrabc coeff-arm cells** (real screen content) as a first byte-exact witness — no full inter
  frame needed. **Deps:** none (builds on existing coeff pipeline). **Size:** L. *(High value:
  unblocks intrabc real content AND inter; already scoped in KB-15.)*

### Chunk 2 — WALKING SKELETON: encode ONE single-ref translational P-frame byte-exact
The vertical slice at the §3 simplest config. Land sub-steps 2a-2c (structure/plumbing) then
2d-2g (search + integration).
- **2a. Encode-side ref management + 2-frame low-delay structure + inter frame-header WRITE.**
  C: `av1_encode_strategy` low-delay path (encode_strategy.c:1250), `get_ref_frame_flags`,
  `choose_primary_ref_frame` (:168), `define_gf_group_pass0` (pass2_strategy.c:2211, trivial for
  lag=0). Port: a `RefFrame` buffer (border-extended recon of frame 0) + `ref_frame_map`; frame 1
  references frame 0; the inter branch of `write_uncompressed_header_obu` (ref-signaling,
  `frame_size_with_refs`, interp/mv-precision/ref-frame-mvs flags — read side in `header.rs`, the
  WRITE assembly + values are net-new). **Shared with inter-decode 1a/1b.**
- **2b. Fixed-Q inter RC.** C: `rc_pick_q_and_bounds_no_stats_cq` (ratectrl.c:1791),
  `av1_rc_pick_q_and_bounds` (:2350). Port: extend `rc.rs` to the multi-frame CQ P-frame qindex.
  Differential vs `av1_rc_pick_q_and_bounds`.
- **2c. Inter ref-mv list (single-ref, spatial).** C: `av1_find_mv_refs` (mvref_common.c:788),
  `av1_mode_context_analyzer`, `av1_mv_pred` (search start MV), `av1_drl_ctx`. Port: generalize
  `dv_ref.rs` — restore `mode_context`/`newmv_count`, sign-bias, identity-GM candidate. **Shared
  with inter-decode 1c.**
- **2d. Single-ref motion estimation.** C: `av1_single_motion_search`
  (motion_search_facade.c:120) → `av1_full_pixel_search` (mcomp.c:1768, PORT HAS full-pel) +
  **`av1_find_best_sub_pixel_tree` (mcomp.c:3266, NET-NEW subpel)** + upsampled-pred. Port: wire
  the intrabc diamond/mesh to reference-frame reads; port the subpel tree search + upsampled pred
  (uses the ported subpel-variance kernels). **First differential: `av1_find_best_sub_pixel_tree`
  vs exported C.**
- **2e. Encoder-side inter MC.** C: `av1_enc_build_inter_predictor` (reconinter_enc.c:111) → the
  shared `reconinter_template` chain → `av1_convolve_*_sr`. Port: wire `aom-convolve` (first
  consumer, **shared build with inter-decode 1d**); per-plane subpel + chroma subsampling; SR
  round/shift (round_0=3, round_1=11). **Differential: `aom-convolve` vs `av1_convolve_*_sr_c`.**
  Add SMOOTH/SHARP filter params for the interp search (2f).
- **2f. `handle_inter_mode` RD (single-ref, SIMPLE motion mode).** C: `av1_rd_pick_inter_mode_sb`
  (rdopt.c ~6180) + `set_params_rd_pick_inter_mode` (:4331) + `handle_inter_mode` (:3063) reduced
  to NEWMV/NEAREST/NEAR/GLOBALMV single-ref, SIMPLE-only motion mode, no compound; interp search
  (`av1_interpolation_filter_search` :674, dual-filter-off 1-D); inter var-tx (chunk 1); mode/drl/
  ref costs (`cost_mv_ref`, `get_drl_cost`). Port: the RD orchestration plugging into the existing
  partition/leaf search + RD primitives + inter symbols + MV coder. Add the missing inter CDF
  default tables the costs consume (inter_mode/newmv/zeromv/refmv, drl, single_ref, intra_inter,
  switchable_interp — several already in `default_cdfs.rs`).
- **2g. Integrate + gate.** Wire the P-frame into the 2-frame harness (chunk 0); decode-both
  byte-exact vs `aomenc` at the §3 config.
- **Test:** frame 1 of the §3 target byte-identical (decode-both + golden) at cq60; frame 0 still
  matches. Plus per-kernel differentials (subpel, MC, var-tx, ref-mv-inter).
- **Deps:** 0, 1. **Size:** XL (land as 2a→2g).

### Chunk 3 — NEWMV robustness + subpel across the cq sweep
- **C:** full `av1_find_best_sub_pixel_tree` precision, `av1_find_best_ref_mvs`, `read/write_mv`
  full class/fp/hp, `av1_drl_ctx`, the DRL ref-mv loop in `handle_inter_mode`.
- **Port:** exact NEWMV + DRL + subpel across cq48/32/12. **Test:** ratchet cq down, byte-exact.
  **Deps:** 2. **Size:** M.

### Chunk 4 — Switchable / dual interp filter
- **C:** `av1_interpolation_filter_search` (interp_search.c:674) full switchable + dual-filter
  (`--enable-dual-filter=1`), `av1_get_pred_context_switchable_interp`, SMOOTH/SHARP convolve.
- **Port:** per-direction interp selection + the dual-filter RD. **Test:** a frame coding
  switchable/dual filters byte-exact. **Deps:** 2. **Size:** M.

### Chunk 5 — bd10 / bd12 single-ref inter
- **C:** `av1_highbd_convolve_*_sr`, highbd enc build_inter_predictor, highbd subpel search
  (subpel-variance highbd PORTED).
- **Port:** highbd convolve variants (the port is u16-plane throughout). **Test:** bd10/bd12
  single-ref P byte-exact. **Deps:** 2. **Size:** M.

### Chunk 6 — Multi-reference selection (LAST/LAST2/LAST3/GOLDEN, single-prediction)
- **C:** `set_params_rd_pick_inter_mode` ref-frame loop (rdopt.c:4331), `read/write_ref_frames`
  single-ref tree over multiple slots, `av1_get_reference_mode_context`; needs a >2-frame
  low-delay GOP so multiple distinct forward refs exist.
- **Port:** the ref-frame RD loop when >1 distinct reference exists. **Test:** a 3-4 frame
  low-delay clip, single-pred blocks byte-exact. **Deps:** 2, 3. **Size:** M.

### Chunk 7 — Global motion (estimation + GLOBALMV RD)
- **C:** `av1_compute_global_motion_facade` (global_motion_facade.c:406),
  `compute_global_motion_for_ref_frame` (:79), `av1_compute_global_motion` (global_motion.c) —
  feature matching + RANSAC + model fit; the affine warp kernel for GLOBALMV MC; `write_global
  _motion` (anchored). Re-enable `--enable-global-motion=1`.
- **Port:** the GM estimation pipeline + non-identity GLOBALMV MC. **Test:** a panning clip coding
  non-identity global motion byte-exact. **Deps:** 2 (+ warp kernel, shared with chunk 9).
  **Size:** L.

### Chunk 8 — OBMC motion mode RD
- **C:** `motion_mode_rd` OBMC arm (rdopt.c:1539), `av1_build_obmc_inter_prediction` (encode side)
  + above/left pred, `av1_count_overlappable_neighbors`, `read/write_motion_mode` OBMC.
  Re-enable `--enable-obmc=1`.
- **Port:** the OBMC RD candidate + overlapped blend. **Test:** a frame coding OBMC byte-exact.
  **Deps:** 2 (+4). **Size:** L.

### Chunk 9 — Local warped motion RD
- **C:** `motion_mode_rd` WARP arm, `av1_findSamples` + `av1_find_projection` (neighbour-sample
  affine fit), the warp kernel (shared with chunk 7). Re-enable `--enable-warped-motion=1`.
- **Port:** neighbour-sample projection → per-block affine → warp MC RD. **Deps:** 7. **Size:** L.

### Chunk 10 — Temporal / motion-field MV (encode side)
- **C:** `av1_setup_motion_field`, `add_tpl_ref_mv` in the list build, `av1_copy_frame_mvs`;
  re-enable `--enable-ref-frame-mvs=1`. **Port:** the projection + `add_tpl_ref_mv` in the
  encode-side ref-mv list (shared with inter-decode §2.6). **Deps:** 2, 6. **Size:** L.

### Chunk 11 — Compound: reference_select + average/dist-wtd + joint motion search
- **C:** compound arm of `handle_inter_mode`, `av1_joint_motion_search`
  (motion_search_facade.c:548), `av1_compound_single_motion_search` (:757),
  `av1_compound_type_rd` (compound_type.c:1234, AVERAGE/DISTWTD), `av1_dist_wtd_convolve_*`
  (16-bit CONV_BUF), `get_comp_index_context`. Needs a GOP with ≥2 distinct refs (fwd+bwd, i.e.
  an alt-ref or a longer structure) → depends on chunk 12's GOP or a crafted config.
- **Port:** compound-ref RD + 2-predictor average/dist-wtd MC + joint MV search. Add the compound
  CDF tables (comp_inter, comp_ref/uni_comp/comp_ref_type, comp_group_idx, compound_idx,
  inter_compound_mode). **Deps:** 6. **Size:** L.

### Chunk 12 — Masked compound (wedge + diffwtd) RD
- **C:** `av1_compound_type_rd` masked arm, `pick_interinter_wedge` (compound_type.c:302),
  `pick_interinter_seg` (:332), `calc_masked_type_cost` (:920), wedge/diffwtd mask tables +
  masked blend (shared with inter-decode §2.10). Re-enable `--enable-masked-comp=1`.
- **Port:** wedge + diffwtd mask search + masked-compound MC RD. **Deps:** 11. **Size:** L.

### Chunk 13 — Interintra RD + skip_mode
- **C:** interintra RD (`av1_handle_inter_intra_mode`, reconinter_enc build; `read/write
  _interintra_info` present); skip_mode RD (`av1_setup_skip_mode_allowed` + the skip_mode arm —
  needs fwd+bwd refs). Re-enable `--enable-interintra-comp=1`.
- **Port:** the remaining inter tools. **Deps:** 11, 12. **Size:** L.

### Chunk 14 — Alt-ref / GOP / lookahead / temporal filtering (`lag>0`)
- **C:** `av1_gop_setup_structure` (gop_structure.c:896), `define_gf_group`
  (pass2_strategy.c:2604), `construct_multi_layer_gf_structure` (:540), `av1_temporal_filter`
  (temporal_filter.c:1616) + `tf_setup_filtering_buffer` (:1258); the lookahead buffer.
- **Port:** the GF-group structure + the ARF (temporally-filtered synthetic frame) + the B-frame
  (fwd+bwd) coding this unlocks. **Test:** a `--lag-in-frames=N` clip with an alt-ref byte-exact.
  **Deps:** 11. **Size:** XL. *(Enables the default good-quality structure.)*

### Chunk 15 — NonRD inter pickmode (speeds 8/9, Gate-2 inter)
- **C:** `av1_nonrd_pick_inter_mode_sb` (nonrd_pickmode.c:3278), `search_new_mv` (:311),
  `find_predictors` (:2406) — the SAD/variance-model realtime inter mode search.
- **Port:** the nonrd inter path (parallels the existing nonrd intra path, speeds 8/9). **Test:**
  inter cpu-used 8/9 byte-exact. **Deps:** 2. **Size:** L.

### Chunk 16 — TPL (lookahead mode/MV/SATD propagation)
- **C:** `av1_init_tpl_stats` (tpl_model.c:1839), `mc_flow_dispenser` (:1589),
  `av1_mc_flow_dispenser_row` (:1525) → deltaq + `prune_inter_modes_based_on_tpl`.
- **Port:** the TPL model. **Deps:** 14. **Size:** XL. *(Needed for default good-quality parity;
  gated on lookahead.)*

### Chunk 17 — 2-pass RC
- **C:** `av1_first_pass` (firstpass.c:1321), `av1_get_second_pass_params`
  (pass2_strategy.c:3949), `av1_twopass_postencode_update` (:4394).
- **Port:** the first-pass stats + second-pass q/bit allocation. **Deps:** 14. **Size:** XL.
  *(Only for `--pass=2` parity.)*

### Chunk 18 — Reference scaling / superres-with-refs / SVC
- **C:** `av1_setup_scale_factors_for_frame`, `av1_convolve_2d_scale`, `setup_frame_size_with_refs`,
  operating-point / spatial-temporal layers. **Port:** scaled-MC + ref-size mismatch + layers.
  **Deps:** 6. **Size:** L.

**Gate-2-inter definition of done:** for every `--cpu-used 0..9` and the standard inter configs
(low-delay CQ, then alt-ref/2-pass), the port's encoded inter bitstream is byte-identical to
`aomenc`, verified by decode-both (the port's inter decoder) + golden. Reached incrementally by
the cq/tool/speed ladders above.

---

## 5. Refactor / infra needs

### #A — Shared MC crate (`aom-inter` or wire `aom-convolve` into a shared MC module)
Inter prediction is needed by BOTH the inter decoder (decode 1d) and the inter encoder (encode
2e). Build the MC once: `aom-convolve` (wire it in — currently unused; add SMOOTH/SHARP + highbd +
dist-wtd variants) + the `reconinter_template`-equivalent single-block build (subpel params,
chroma subsampling, border reads). Recommend a shared `aom-inter` crate (deps `aom-convolve`
+ MV/ref types) that both `aom-decode` and `aom-encode` consume — keeps MC out of both large
`lib.rs` files and guarantees the encoder's prediction == the decoder's (byte-exactness
depends on this identity).

### #B — Shared ref-mv-list generalization (`dv_ref.rs` → inter) — ALREADY LANDED
The single-ref inter ref-mv list (mode_context/newmv_count/sign-bias/GM restore) is identical for
encode and decode. **This landed on `origin/main` as `find_inter_mv_refs` (commit `cdba774`,
"generalized single-ref inter MV scan — byte-exact vs C").** The encoder reuses it directly for
chunk 2c; no further port work — only wire it into the encode ref-frame loop.

### #C — `derive_real_costs` inter ext-tx (one-liner)
`real_costs.rs:155` stubs `inter_ext_tx_cdf` with a zero row; source it from
`KfFrameContext.inter_ext_tx` (`partition.rs:5849`, already `DEFAULT_INTER_EXT_TX`). Unblocks
inter tx-type cost (chunk 1 / KB-15).

### #D — Multi-frame encode harness
The current `aom-bench` harness is single-frame (`rd_close.rs:164`). Add the 2-frame `[KEY, P]`
driver + decode-both localizer (chunk 0). Reuse `attempt_case_content_uv_sep`.

### #E — Encode-side reference-frame buffer pool + frame loop
A `RefFrame` (border-extended recon Y/U/V + order_hint + saved CDFs + global_motion + per-8×8
mvs) + `ref_frame_map[8]` + refresh, driven by a minimal encode frame loop. Encode-only state
(mirrors the decoder's pool from inter-decode 1a but on the encode side). Belongs in `aom-encode`.

---

## 6. Cross-track dependency on the inter DECODER (verification gating)

The encoder's byte-exact gate uses **decode-both** — so the inter DECODER must be far enough along
to decode the target. **This dependency is already MET for chunk 2:** the inter-decode single-ref
translational skeleton is byte-exact on `origin/main` (`93b92ec`/`bc2d1bd`/`50816e5`/`cfd39e0`/
`835b0c0`), so the encoder can verify a single-ref P-frame by decode-both today. Later encode
chunks still gate on their decode siblings per the table below.

Which inter-decode chunks gate which inter-encode chunks:

| inter-encode chunk | needs inter-decode |
|---|---|
| 2 (single-ref translational P) | decode chunk 1 (single-ref translational decode) |
| 4 (switchable/dual interp) | decode chunk 3 (switchable interp) |
| 5 (bd10/bd12) | decode chunk 4 (bd10 inter) |
| 6 (multi-ref) | decode chunk 5 (multi-ref) |
| 7 (global motion) | decode chunk 10 (global-motion warp) |
| 8 (OBMC) | decode chunk 8 (OBMC) |
| 9 (local warp) | decode chunk 11 (local warp) |
| 10 (temporal MV) | decode chunk 9 (temporal MV) |
| 11-13 (compound/masked/interintra/skip) | decode chunks 6/7/13 |

**Shared builds** (do once, both tracks): the MC crate (§5 #A = decode 1d + encode 2e), the
ref-mv-list generalization (§5 #B = decode §2.5 + encode 2c), the compound/wedge/OBMC/warp mask
+ kernel tables (decode chunks 6-11 = encode chunks 7-13). Kernels can be developed against C
differentials in parallel with the decoder; only the end-to-end byte gate blocks on it.

**Recommended sequencing:** land inter-decode chunk 1 and inter-encode chunks 0+1 in parallel
(chunk 1 = inter var-tx has its own intrabc-coeff witness, needs no decoder), then inter-encode
chunk 2 once the decoder can decode a single-ref P. From there the cq/tool ladders on both tracks
advance in lockstep, each encode chunk gated by its decode sibling.

---

## Appendix A — Key C entry points (verified v3.14.1, `av1/encoder/`)

**Motion estimation (mcomp.c):** `av1_full_pixel_search` (1768), `full_pixel_exhaustive` (1615),
`av1_refining_search_8p_c` (1696), `av1_find_best_sub_pixel_tree` (3266), `_pruned` (3120),
`_pruned_more` (3026), `av1_get_mvpred_sse` (3963). Facade: `av1_single_motion_search`
(motion_search_facade.c:120), `av1_joint_motion_search` (:548), `av1_compound_single_motion
_search` (:757), `av1_simple_motion_search_sse_var` (:989).

**Inter mode RD (rdopt.c):** `av1_rd_pick_inter_mode_sb` (~6180; set_params call :6202),
`set_params_rd_pick_inter_mode` (4331), `handle_inter_mode` (3063; 6-step structure :3153-3161),
`handle_newmv` (1317), `motion_mode_rd` (1539), `av1_rd_pick_inter_mode_sb_seg_skip` (6611).
Inter mode enums (common/enums.h): NEARESTMV/NEARMV/GLOBALMV/NEWMV (337-340),
NEAREST_NEARESTMV..NEW_NEWMV (342-350), motion modes SIMPLE_TRANSLATION/OBMC_CAUSAL/WARPED_CAUSAL
(398-400).

**Inter var-tx (tx_search.c):** `av1_txfm_search` (3795, dispatch), `av1_pick_recursive_tx_size
_type_yrd` (3553), `av1_pick_uniform_tx_size_type_yrd` (3628, intra, PORTED), `select_tx_size_and
_type` (3433), `select_tx_block` (2601), `try_tx_block_no_split` (2406), `try_tx_block_split`
(2454), `prune_tx_2D` (1541), `ml_predict_tx_split` (1755).

**Interp + compound (interp_search.c / compound_type.c):** `av1_interpolation_filter_search`
(674), `interpolation_filter_rd` (153), `find_best_interp_rd_facade` (314); `av1_compound_type_rd`
(1234), `pick_interinter_wedge` (302), `pick_interinter_seg` (332), `calc_masked_type_cost` (920).

**Encoder MC (reconinter_enc.c):** `av1_enc_build_inter_predictor` (111), `_y` (61), `_y_nonrd`
(85), `enc_build_inter_predictors` (54), `av1_build_inter_predictors_for_planes_single_buf` (271).
Shares `reconinter_template.inc` + `av1_make_inter_predictor` (reconinter.c) + `av1_convolve_*_sr`
(convolve.c) with the decoder (see inter-decode Appendix A for the convolve/round-shift refs).

**Frame mgmt / RC / GOP:** `av1_encode_strategy` (encode_strategy.c:1250), `denoise_and_encode`
(:728), `choose_primary_ref_frame` (:168), `get_ref_frame_flags` (:1657 call); `av1_rc_pick_q_and
_bounds` (ratectrl.c:2350), `rc_pick_q_and_bounds_no_stats_cq` (:1791), `_no_stats` (:1588),
`av1_rc_regulate_q` (:1138), `av1_rc_postencode_update` (:2444); `av1_gop_setup_structure`
(gop_structure.c:896), `construct_multi_layer_gf_structure` (:540), `define_gf_group`
(pass2_strategy.c:2604), `define_gf_group_pass0` (:2211); low-delay predicate encoder.h:4174,
`is_altref_enabled` encoder.h:4110.

**Large optional subsystems:** `av1_compute_global_motion_facade` (global_motion_facade.c:406),
`compute_global_motion_for_ref_frame` (:79); `av1_temporal_filter` (temporal_filter.c:1616),
`tf_setup_filtering_buffer` (:1258); `av1_init_tpl_stats` (tpl_model.c:1839), `mc_flow_dispenser`
(:1589), `av1_mc_flow_dispenser_row` (:1525); `av1_first_pass` (firstpass.c:1321),
`av1_get_second_pass_params` (pass2_strategy.c:3949); `av1_nonrd_pick_inter_mode_sb`
(nonrd_pickmode.c:3278), `search_new_mv` (:311), `find_predictors` (:2406).

**Port head-start locations:** ME full-pel + MV cost + SAD + hash =
`aom-encode/src/intrabc_search.rs` (1921 LOC); MV coder + inter symbols + pred-contexts + ref-mv
list = `aom-entropy/src/partition.rs` + `dv_ref.rs`; inter ext-tx CDF =
`partition.rs:5849`; convolve = `aom-convolve/src/lib.rs`; RD engine + costs + tx dispatch =
`aom-encode/src/{rd_pick,partition_pick,tx_search,mode_costs,real_costs,rd}.rs`; CQ qindex =
`aom-encode/src/rc.rs`.
