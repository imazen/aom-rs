# aom-rs — Inter-Frame DECODE Roadmap

Scope: DECODER first (bit-exact-verifiable against the AV1 conformance corpus; the
foundation the inter *encoder* will later mirror). Single-frame KEY/intra decode is
DONE and Gate-1-clean; this document maps the gap to **inter-frame decode** and
decomposes it into ordered, smallest-demoable-first chunks.

Reference tree: `/root/aom-rs/reference/libaom` (libaom v3.14.1). Port crates:
`crates/aom-{decode,entropy,convolve,txb,transform,quant,intra,cdef,loopfilter,restore,...}`.

Priority of evidence (per project methodology): real exported C fn > synthetic-facade-over-real-fn
> verbatim transcription. Every new numeric kernel lands with a differential vs the REAL C.

---

## 0. Executive summary

The inter gap is smaller than it looks on the *parse/entropy* side and larger on the
*prediction* side:

- **Frame-header inter PARSE is essentially done.** `read_uncompressed_header`
  (`crates/aom-entropy/src/header.rs:2952`) already parses the entire non-intra branch:
  inter ref signaling, `frame_size_with_refs`, `allow_high_precision_mv`, `interp_filter`,
  `switchable_motion_mode`, `allow_ref_frame_mvs`, `reference_mode`, `skip_mode`,
  `allow_warped_motion`, `reduced_tx_set`, and `read_global_motion` (header.rs:3080-3093).
  It is a shared read/write module, partially exercised by encoder round-trips.
- **MV-candidate scan is 80% there and byte-exact.** `crates/aom-entropy/src/dv_ref.rs`
  ports `av1_find_mv_refs`/`setup_ref_mv_list`/`scan_{row,col,blk}_mbmi`/`add_ref_mv_candidate`/
  `has_top_right`/bubble-sort/`av1_find_best_ref_mvs`, **verified against the exported
  `av1_find_mv_refs` + `av1_find_best_ref_mvs`** (`dv_ref_diff.rs`), reduced to the
  single-ref INTRA path. It generalizes directly to inter single-ref.
- **Inter mode symbol readers + MV CDFs exist.** `read_inter_mode` (partition.rs:3697),
  `read_inter_compound_mode` (:3684), `read_interintra_info` (:4599), `read_mv_component`
  (:3720, used by intrabc `read_mv`); `nmv_context` default CDFs present in `default_cdfs.rs`.
- **The `is_inter` TX/coefficient plumbing is partly wired via intrabc** (KEY-frame
  `is_inter_block`): vartx read, inter ext-tx set, `is_inter` `get_tx_size_context`.
- **`reconstruct_txb` (dequant+inv-txfm+add) is reused verbatim** for inter residual —
  but it lives in `aom-encode` (`crates/aom-encode/src/lib.rs:1015`) and `aom-decode`
  reverse-depends on it (`crates/aom-decode/src/lib.rs:159`). See §5 (refactor #24).

What is **genuinely missing**: (a) the multi-frame decode loop + reference-frame buffer
pool + refresh; (b) the inter CDF default tables (mode/ref/mv beyond nmvc); (c) inter
**prediction** (motion compensation) — single-ref, compound, OBMC, warp, global motion,
reference scaling; (d) the `read_inter_block_mode_info` orchestration; (e) temporal
(motion-field) MV; (f) the post-header ref-state setup (sign bias, motion field,
skip_mode_allowed, scale factors).

**First byte-exact target:** frame 1 of a `av1-1-b8-00-quantizer-*` vector (2-frame
KEY+INTER; all 128 already local) — see §3.

---

## 1. Head-start inventory (what already exists, byte-exact-or-not)

| Building block | Location | State | Byte-exact? |
|---|---|---|---|
| **Inter-prediction convolution** (single-ref) | `crates/aom-convolve/src/lib.rs:75/98/145` — `convolve_x_sr`, `convolve_2d_sr`, `convolve_y_sr`; EIGHTTAP regular/smooth/sharp, **lowbd single-ref SR only** | Standalone crate, **NOT wired into decode or encode**; no deps | Transcription of `av1_convolve_{x,y,2d}_sr_c`; **needs a differential vs the exported C** before trust |
| **MV-candidate scan / ref-mv list** | `crates/aom-entropy/src/dv_ref.rs` — full `setup_ref_mv_list` machinery, reduced to `ref_frame==INTRA_FRAME` single-ref | Ported; compound + temporal + mode_context + global-MV arms explicitly dropped (documented dead-for-KEY) | **YES** — diffed vs exported `av1_find_mv_refs`+`av1_find_best_ref_mvs` (`dv_ref_diff.rs`) |
| **Inter frame-header parse** | `crates/aom-entropy/src/header.rs:2952` `read_uncompressed_header` (non-intra branch 2982-3093) | Reads all inter header fields incl. global motion | Shared r/w module; exercised for encoder-written fields; **pure-inter fields need decode-path validation** |
| **Inter mode symbol readers** | partition.rs — `read_inter_mode:3697`, `read_inter_compound_mode:3684`, `read_interintra_info:4599`, `read_mv_component:3720` | Leaf readers present; **orchestration absent** | Present (encoder/intrabc-adjacent); confirm each vs C when wired |
| **MV / DV CDF defaults** | `crates/aom-entropy/src/default_cdfs.rs` — `nmv_context` (nmvc/ndvc), `joints_cdf`, `sign_cdf`, `class0*`, `intrabc_cdf`, `inter_ext_tx_cdf` | Present (drive intrabc DV read + inter ext-tx) | Used on KEY path (intrabc); reusable for inter nmvc |
| **`is_inter` tx/coeff plumbing** | partition.rs + `crates/aom-txb/src/{read,ext_tx}.rs` + aom-decode/lib.rs — vartx read, inter ext-tx set, `is_inter` `get_tx_size_context` | Exercised by intrabc (KEY `is_inter_block`) | Byte-exact on the intrabc path in Gate-1 |
| **Residual reconstruct** | `crates/aom-encode/src/lib.rs:1015` `reconstruct_txb` (+ `reconstruct_txb_wht` aom-decode/lib.rs:168) | dequant → `av1_inv_txfm2d_add`; **inter reuses verbatim** | Byte-exact (Gate-1 intra recon) — but wrong crate (see §5) |
| **Frame border extension** | present in aom-convolve, aom-decode/frame.rs, superres.rs (for intrabc/superres) | Partial; needs generalization to reference-frame border extension for MC | n/a |
| **OBU / temporal-unit parse** | `crates/aom-entropy/src/obu.rs` (`read_obu_header`), `crates/aom-decode/src/frame.rs` (seq-header + uncompressed-header + tile-group parse) | Present, but the decoder driver stops after ONE shown KEY frame | Byte-exact for the single-frame path |

**What is NOT present at all:** reference-frame buffer pool / `ref_frame_map` / refcount /
refresh; multi-frame decode loop; inter CDF *default tables* (inter_mode/newmv/zeromv/refmv,
drl, single_ref/comp_ref/uni_comp_ref/comp_ref_type, comp_inter/intra_inter, switchable
interp, motion_mode/obmc, compound_type/wedge/comp_group_idx/compound_idx, interintra,
skip_mode — nmvc classes/bits/fp/hp exist, the rest do not); inter **prediction**
(`av1_build_inter_predictors` and everything under it); temporal/motion-field MV; the
`read_inter_block_mode_info` orchestration; ref-state setup (`av1_setup_frame_buf_refs`,
`av1_setup_frame_sign_bias`, `av1_setup_motion_field`, `av1_setup_skip_mode_allowed`,
`av1_setup_scale_factors_for_frame`).

---

## 2. C-path gap map (subsystem → C functions → port need)

All C refs `av1/...` under `reference/libaom`. Line numbers verified v3.14.1.

### 2.1 Reference-frame management & the multi-frame loop
- Driver chain: `aom_decode_frame_from_obus` (decoder/obu.c:867) → `av1_decode_frame_headers_and_setup`
  (decodeframe.c:5258) → `read_uncompressed_header` (decodeframe.c:4569) →
  `av1_decode_tg_tiles_and_wrapup` (decodeframe.c:5359) → **`update_frame_buffers`** (decoder.c:365).
- Ref refresh/swap: `update_frame_buffers` (decoder.c:365-423, refresh loop 379-386) — for each
  set bit of `refresh_frame_flags`: `decrease_ref_count(ref_frame_map[i])` then
  `ref_frame_map[i]=cur_frame; ++cur_frame->ref_count`. Called at decoder.c:500.
- `reset_ref_frame_map` (decodeframe.c:4498), `update_ref_frame_id` (:4509),
  `reset_frame_buffers` (:4547), `release_current_frame` (decoder.c:349).
- **Port need:** a `RefFrame`/buffer-pool type (currently absent) holding: full uncropped
  border-extended recon planes (Y/U/V), `order_hint`, `ref_order_hints[7]`, `base_qindex`,
  saved frame-context CDFs, `global_motion[7]`, per-8×8 `mvs[]` (`MV_REF`), `showable`,
  refcount; an `ref_frame_map[8]`; a temporal-unit loop in the decode entry
  (`decode_frame_obus`, frame.rs:613) that decodes each frame, installs it per
  `refresh_frame_flags`, and outputs shown frames.

### 2.2 Frame-header inter fields (parse-order)
`read_uncompressed_header` (decodeframe.c:4569-5230). Inter-specific steps, all under the
non-KEY/non-INTRA_ONLY branch (4947-5061) or `!frame_is_intra_only`-gated (5217):
- `show_existing_frame` (4599) + show-existing block (4602-4671: existing_frame_idx,
  buffer assign, order_hint/film_grain copy, return).
- `frame_type` (4673), `show_frame`/`showable_frame` (4686/4699/4706), `error_resilient` (4709).
- `frame_size_override_flag` (4787); `order_hint` (4789, OrderHintBits); `primary_ref_frame`
  (4795, PRIMARY_REF_BITS — never read on intra).
- `refresh_frame_flags` (4824-4853: 0xFF for shown-KEY, read_literal(8) otherwise); missing-ref
  order-hints + neutral-grey buffers under error_resilient (4855-4928).
- Inter ref-idx block (4947-5033): `frame_refs_short_signaling` (4948) + `av1_set_frame_refs`
  (4975); per-ref `remapped_ref_idx[i]` (4992), `ref_frame_sign_bias` init (5001), delta ids.
- `setup_frame_size_with_refs` vs `setup_frame_size` (5019-5024).
- `allow_high_precision_mv` (5026), `interp_filter` (5031, `read_frame_interp_filter`),
  `switchable_motion_mode` (5032).
- `prev_frame = get_primary_ref_frame_buf` (5035); `allow_ref_frame_mvs` (5043).
- per-ref `av1_setup_scale_factors_for_frame` + `av1_is_valid_scale` (5050-5060).
- `av1_setup_frame_buf_refs` (5064), `av1_setup_frame_sign_bias` (5066), `update_ref_frame_id` (5070).
- CDF load: `av1_setup_past_independence` iff `primary_ref==PRIMARY_REF_NONE` (5138), else load
  from primary ref (5324-5329).
- `read_frame_reference_mode` (decodeframe.c:145; call 5199); `av1_setup_skip_mode_allowed`
  (5201) + `skip_mode_flag` (5202); `allow_warped_motion` (5205); `reduced_tx_set` (5210);
  `read_global_motion` (5217).
- **Port state:** the PARSE (§1) is done. Missing = the ref-derived `cfg` inputs the port's
  `read_uncompressed_header` takes (ref crop sizes for `frame_size_with_refs`, order hints,
  `skip_mode_allowed`, `might_allow_{warped_motion,ref_frame_mvs}` gates, `ref_global_motion`)
  and the post-parse state setup (2.1 + sign bias + motion field + scale factors).

### 2.3 Ref-state setup (post-header, pre-tile)
- `av1_setup_frame_buf_refs` (mvref_common.c:843-859), `av1_setup_frame_sign_bias`
  (mvref_common.c:861-876), `av1_calculate_ref_frame_side` (mvref_common.c:994-1013, call
  decodeframe.c:5319), `av1_setup_motion_field` (mvref_common.c:1015-1075, call 5320),
  `av1_setup_skip_mode_allowed` (mvref_common.c:1246-1345), `av1_set_frame_refs`
  (mvref_common.c:1346, short-signaling ref derivation), `get_relative_dist`
  (mvref_common.h:37). **Port need:** all net-new; order-hint math is small + easily
  differential-tested.

### 2.4 Frame-size / ref-scaling
- `setup_frame_size` (decodeframe.c:2025), `setup_frame_size_with_refs` (:2065,
  `valid_ref_frame_size`:2116), `setup_superres` (:1929), `av1_setup_scale_factors_for_frame`
  (scale.c:44-57), `av1_is_valid_scale`/`av1_is_scaled`. **Port need:** frame-size + superres
  parse mostly present (KEY path); the ref-scaling scale-factor attach + the scaled-MC dispatch
  are net-new (needed only when a ref differs in size — deferrable; §4 chunk 12).

### 2.5 Inter MV prediction (candidate list)
- `av1_find_mv_refs` (mvref_common.c:788), `setup_ref_mv_list` (:479, the bit-exact scan order),
  `scan_{row,col,blk}_mbmi` (:143/:191/:239), `add_ref_mv_candidate` (:75), `has_top_right` (:264),
  `av1_find_best_ref_mvs` (:832), `av1_mode_context_analyzer` (mvref_common.h:170), `av1_drl_ctx`
  (:185), `av1_collect_neighbors_ref_counts` (:209), `clamp_mv_ref` (:52).
- Compound extension: `process_compound_ref_mv_candidate` (mvref_common.c:423) + comp_list build
  (setup_ref_mv_list:685-752). Single extension: `process_single_ref_mv_candidate` (:448).
- Global-motion MV: `gm_get_motion_vector` (mv.h:231).
- **Port need:** generalize `dv_ref.rs` — (a) restore the compound branch (currently dropped),
  (b) restore `mode_context`/`newmv_count` (needed for inter mode CDF context — dropped as dead),
  (c) add `gm_mv_candidates` via `gm_get_motion_vector`, (d) add sign-bias negation in
  `process_single_ref_mv_candidate`, (e) add the temporal block (2.6). The spatial scan itself is
  byte-exact and reusable.

### 2.6 Temporal / motion-field MV
- `av1_setup_motion_field` (mvref_common.c:1015), `motion_field_projection` (:919, the
  per-block projection is inlined — no separate `motion_field_estimation` symbol),
  `add_tpl_ref_mv` (:329, consumes `tpl_mvs` during list build), `get_block_position` (:881),
  `av1_get_mv_projection` (:27), `av1_copy_frame_mvs` (:41, per-8×8 MV store into `cur_frame->mvs`).
- Structs: `TPL_MV_REF` (av1_common_int.h:106), `MV_REF` (:111), MFMV_STACK_SIZE=3.
- **Port need:** net-new; gated on `allow_ref_frame_mvs` (enable_order_hint). If the first target
  frame has it off, defer to a dedicated chunk (§4 chunk 9); if on, chunk 1 must include
  `add_tpl_ref_mv` + the store.

### 2.7 Inter mode / ref / MV parse (per block)
- `av1_read_mode_info` (decodemv.c:1571) → `read_inter_frame_mode_info` (:1512):
  `read_inter_segment_id` (:363), `read_skip_mode` (:421), `read_skip_txfm` (:446),
  `read_cdef`, `read_delta_q_params`, `read_is_inter_block` (:1223) →
  `read_inter_block_mode_info` (:1273).
- `read_inter_block_mode_info` (decodemv.c:1273-1510) order-critical: `read_ref_frames` (:938),
  `av1_find_mv_refs` (:1298), mode read (`read_inter_compound_mode`:249 / `read_inter_mode`:177),
  `read_drl_idx` (:197), `av1_find_best_ref_mvs` (:1330), `assign_mv` (:1114) →
  `read_mv` (:886) → `read_mv_component` (:846); inter-intra (:1383), scale factors (:1409),
  `av1_findSamples` (:1417) + `av1_count_overlappable_neighbors`, `read_motion_mode` (:227),
  compound-type read (:1425-1479: comp_group_idx / compound_idx / wedge / diffwtd),
  `read_mb_interp_filter` (:1034), warp model `av1_find_projection` (:1495).
- Ref-frame parse: `read_ref_frames` (:938) — single-ref tree (`single_ref_p1..p6`) vs compound
  (uni/bidir `comp_ref_p*`/`comp_bwdref_p*`); contexts in pred_common.c
  (`av1_get_reference_mode_context`:145, `av1_get_comp_reference_type_context`:187, the
  `single_ref_p*`/`comp_*` context fns 421-499).
- **Port need:** the `read_inter_block_mode_info` orchestration is net-new (leaf symbol readers
  exist, §1). The ref-bit context fns (`av1_collect_neighbors_ref_counts` + the pred_common
  contexts) are net-new. `assign_mv`/`read_mv`/`read_mv_component` largely reusable from the
  intrabc DV path.

### 2.8 Inter prediction — single-ref translational (MC core)
**Key structural fact:** there is NO public `av1_build_inter_predictor(s)` symbol in v3.14.1 —
the per-block build lives in `reconinter_template.inc`, `#include`d twice with an `IS_DEC`
switch (decoder instance compiled at decodeframe.c:668-670). The translational single-ref chain:
- `dec_build_inter_predictor` (decodeframe.c:681, decoder entry per block) →
  `dec_build_inter_predictors` (:672, passes `mc_buf`) →
  `build_inter_predictors` (reconinter_template.inc:242) →
  `build_inter_predictors_8x8_and_bigger` (:165, ref loop; sub-8×8 chroma =
  `build_inter_predictors_sub8x8` :87) → `build_one_inter_predictor` (:16) →
  **`av1_make_inter_predictor` (reconinter.c:77)** [the "single block" core; TRANSLATION_PRED →
  `inter_predictor` reconinter.h:255 / highbd :275] → `av1_convolve_2d_facade` (convolve.c:638) →
  `av1_convolve_{2d,x,y}_sr` (convolve.c:78/158/137) / `aom_convolve_copy`.
- Subpel params: `dec_calc_subpel_params` (decodeframe.c:565) + `dec_calc_subpel_params_and_extend`
  (:651, appends `extend_mc_border` for out-of-frame reads); shared math `init_subpel_params`
  (reconinter.h:131). Params structs: `InterPredParams` (reconinter.h:107), `SubpelParams` (:77),
  filled by `av1_init_inter_params` (reconinter.h:216).
- **Round/shift (bit-exact critical), single-ref SR:** `ConvolveParams` via
  `get_conv_params_no_round` (convolve.h:68): `round_0 = ROUND0_BITS = 3`,
  `round_1 = 2*FILTER_BITS − round_0 = 11`. `av1_convolve_2d_sr` horiz seed `1<<(bd+FILTER_BITS−1)`
  → `>>3`; vert `offset_bits=19`, `>>11` − `((1<<8)+(1<<7))`, final `bits=0`. `_x_sr` total shift 7
  (`>>3` then `>>4`); `_y_sr` takes no ConvolveParams, pure `>>FILTER_BITS(7)`. The port's
  `aom-convolve` (convolve_2d_sr/x_sr/y_sr) already implements exactly these — confirm via a
  differential vs the exported `av1_convolve_*_sr_c`, then wire it in.
- Residual add: inter arm of `decode_token_recon_block` (decodeframe.c:966-1041) predicts FIRST
  (973), then adds residual per 64×64-chunked plane loop via `decode_reconstruct_tx` (1024).
- **Port need:** wire `aom-convolve` in; build the template chain (single-ref, TRANSLATION) with
  per-plane subpel-x/y + integer offset (`init_subpel_params`, chroma subsampling), the sub-8×8
  chroma path, `extend_mc_border` reference reads. Reuse `reconstruct_txb` for residual. **Chunk 1
  core.** Full internal refs in Appendix A.

### 2.9 Compound prediction
- `av1_dist_wtd_comp_weight_assign` (distance-weighted avg), `build_masked_compound` +
  `av1_build_compound_diffwtd_mask` (DIFFWTD), wedge (`av1_get_compound_type_mask`,
  `av1_init_wedge_masks`, `wedge_params_lookup`), `make_masked_inter_predictor`,
  `av1_dist_wtd_convolve_2d`/`_x`/`_y` (convolve.c). Contexts: `get_comp_group_idx_context`
  (pred_common.h:141), `get_comp_index_context` (:102).
- **Port need:** net-new; the compound convolve round/shift (16-bit intermediate `CONV_BUF`),
  averaging, and the mask tables. Split average-compound (chunk 6) from masked-compound
  (wedge/diffwtd, chunk 7).

### 2.10 OBMC
- `av1_build_obmc_inter_prediction`, `av1_setup_obmc_mask`,
  `av1_setup_build_prediction_by_{above,left}_pred`, `av1_count_overlappable_neighbors`;
  decode-side `dec_build_obmc_inter_predictors_sb` (decodeframe.c:818-842, above/left preds
  706-816). **Port need:** net-new (chunk 8).

### 2.11 Warped / global motion
- `av1_warp_plane` / `av1_warp_affine` (warped_motion.c), `av1_find_projection`,
  `av1_get_shear_params`, `div_lut`/`warped_filter` tables; `gm_get_motion_vector` (mv.h:231),
  `read_global_motion` (parse already ported, header.rs:3092). Consts WARPEDMODEL_PREC_BITS,
  WARP_PARAM_REDUCE_BITS. **Port need:** net-new (global-motion warp = chunk 10; local
  warped_causal = chunk 11). *(warped_motion.c internal refs — Appendix A.)*

### 2.12 Interintra, skip_mode, segmentation-inter
- Interintra: `av1_build_interintra_predictors*`, `av1_combine_interintra` (reconinter.c);
  parse `read_interintra_info` present (partition.rs:4599). Chunk 13.
- skip_mode: `av1_setup_skip_mode_allowed` (2.3) + `read_skip_mode` (2.7) — implies bidirectional
  compound; likely NOT allowed on the 2-frame single-ref target (needs 2 refs). Chunk 13.
- Segmentation temporal_update (`read_inter_segment_id` preskip/postskip, seg pred CDF): fold into
  chunk 1's mode-info parse (small).

---

## 3. Simplest inter conformance vector (first byte-exact target)

**Finding (verified by OBU frame-type parse over the local corpus):** every
`av1-1-b8-00-quantizer-NN` and `av1-1-b10-00-quantizer-NN` vector is **2 frames: `[KEY, INTER]`**.
The `scope_hint:"intra"` in `conformance/vectors.json` is a *family heuristic* — the intra Gate-1
test bounds these mixed streams to frame 0 (`scope_for`, `conformance_corpus.rs:273`). **All 128
`00-quantizer` vectors are already local** in `conformance/data/` (gitignored), and each `.md5`
carries **2 golden lines** — line 2 is the byte-exact target for the decoded INTER frame's i420 output.

Only 5 vectors are *explicitly* inter-scoped (`05-mv`, `06-mfmv`, `22-svc-{L1T2,L2T1,L2T2}`) and
**none are local yet** — fetch with `python3 xtask/conformance.py --fetch --scope inter`
(FAMILY_SCOPE maps these three families to "inter", conformance.py:63-66). These are multi-frame
(8 frames each) and *designed to stress* MVs / motion-field / SVC layering — **not** the simplest
first target.

**Why `00-quantizer` frame 1 is the ideal first inter frame:** it is a natural 2-frame clip, so
frame 1 has exactly **one** decoded reference available (frame 0). All 7 ref slots resolve to
frame-0's buffer, so the encoder codes it **single-reference** (compound needs 2 distinct refs);
**`skip_mode` is disallowed** (needs forward+backward refs → `skip_mode_allowed=0`). That
eliminates compound / masked-compound / skip_mode from the first frame, leaving single-ref
translational MC + (some) NEWMV/NEARESTMV/NEARMV/GLOBALMV — exactly chunk-1 scope.

**The qindex sweep IS a built-in difficulty ladder.** Measured INTER-frame payload (bytes):

| vector | KEY | INTER | note |
|---|---|---|---|
| `av1-1-b8-00-quantizer-63` | 393 | **157** | highest-q: near-all-skip + GLOBALMV/NEARESTMV, fewest coeffs |
| `av1-1-b8-00-quantizer-62` | 513 | 170 | |
| `av1-1-b8-00-quantizer-60` | 757 | 232 | |
| `av1-1-b8-00-quantizer-48` | 2228 | 715 | |
| `av1-1-b8-00-quantizer-40` | 4470 | 1183 | |
| `av1-1-b8-00-quantizer-20` | 16793 | 4029 | mid — richer modes/coeffs |
| `av1-1-b8-00-quantizer-00` | 74506 | 54381 | lossless-ish, densest |

**Recommended first target: `av1-1-b8-00-quantizer-63` (frame 1).** Smallest inter payload →
smallest tool surface → fastest to bit-exact. Then ratchet down the sweep (63 → 60 → 48 → 40 →
20 → … → 00), each step adding coefficient volume and eventually NEWMV / switchable interp /
(possibly) warp / temporal-MV — which map onto the later chunks.

**Two caveats the implementer must resolve empirically (via the established sibling-C
instrumentation methodology):**
1. Confirm `allow_ref_frame_mvs` for the chosen frame. If ON, chunk 1 must include the temporal
   `add_tpl_ref_mv` (2.6) OR pick a higher-q frame where it is off. (Verify by instrumenting the
   C decoder's header parse.)
2. Confirm the chosen frame uses only chunk-1 tools (no local-warp / OBMC / interintra). If a tool
   sneaks in at q63, either pick a neighbouring q or pull that tool's chunk forward.

**Infra task:** extend `conformance_corpus.rs`'s `scope_for` (and, optionally, reclassify
`00-quantizer` in `conformance.py` FAMILY_SCOPE to a new `key+inter` scope) so the Gate accepts
frame 1 of the 2-frame families as inter targets, comparing each frame's i420 md5 to the golden
`.md5` lines. Do this in chunk 1.

---

## 4. Ordered chunk decomposition (smallest-demoable-first)

Each chunk: **{C funcs → port target → byte-exact test → deps → size}**. Sizes S/M/L/XL.
Every kernel lands with a differential vs the REAL exported C (project methodology).

### Chunk 0 — Refactor: relocate `reconstruct_txb` to a shared crate (prereq-lite)
- **C funcs:** n/a (port-internal hygiene).
- **Port:** move `reconstruct_txb` + `dequant_txb` wrapper out of `aom-encode` (lib.rs:1015)
  into a shared home both decode & (future inter) encode depend on — see §5. Rewire
  `aom-decode/src/lib.rs:159`.
- **Test:** existing Gate-1 intra recon stays byte-identical; `aom-decode` no longer
  depends on `aom-encode`.
- **Deps:** none. **Size:** S. *(Optional-but-recommended before inter grows; inter can
  technically reuse the existing import, so this can also land in parallel.)*

### Chunk 1 — Walking skeleton: decode ONE single-ref translational inter frame byte-exact
The vertical slice. Decomposed into sub-steps 1a-1f; land 1a-1c first (parse plumbing), then
1d-1f (prediction+integration).
- **1a. Reference-frame buffers + multi-frame loop.** C: `update_frame_buffers` (decoder.c:365),
  `av1_setup_frame_buf_refs` (mvref_common.c:843). Port: a `RefFrame` type (full uncropped
  border-extended Y/U/V recon + order_hint + ref_order_hints + saved CDFs + global_motion +
  per-8×8 `mvs` + refcount) and `ref_frame_map[8]`; a temporal-unit loop in `decode_frame_obus`
  (frame.rs:613); lift the KEY-only gates (frame.rs:504-511). Border-extend each decoded frame
  (generalize the existing intrabc/superres border code).
- **1b. Inter frame-header state setup.** C: `av1_setup_frame_sign_bias` (mvref_common.c:861),
  `av1_calculate_ref_frame_side` (:994), `av1_setup_skip_mode_allowed` (:1246), `get_relative_dist`
  (mvref_common.h:37), `read_frame_reference_mode` (decodeframe.c:145). Port: feed the
  already-ported `read_uncompressed_header` (header.rs:2952) its ref-derived `cfg` inputs (order
  hints, ref crop sizes, `skip_mode_allowed`, `might_allow_*` gates, `ref_global_motion`); run the
  post-parse sign-bias + ref-side. Generalize the two-phase probe (frame.rs:455-501) for inter.
- **1c. Inter mode-info parse (single-ref, translational).** C: `read_inter_frame_mode_info`
  (decodemv.c:1512), `read_is_inter_block` (:1223), `read_ref_frames` single-ref tree (:938),
  `read_inter_mode` (:177), `read_drl_idx` (:197), `assign_mv`/`read_mv`/`read_mv_component`
  (:1114/:886/:846), `read_mb_interp_filter` (:1034); MV list via generalized `dv_ref.rs`
  (single-ref inter: add `mode_context`/`newmv_count`, sign-bias, `gm_mv` — NO compound/temporal
  yet). Contexts: `av1_collect_neighbors_ref_counts` (mvref_common.h:209),
  `av1_get_intra_inter_context` (pred_common.c:124), the `single_ref_p*` contexts (:455-499).
  **Add the missing inter CDF default tables** (inter_mode/newmv/zeromv/refmv, drl, single_ref,
  intra_inter, switchable_interp, motion_mode(→SIMPLE only here)) transcribed from
  `entropymode.c` `default_*_cdf`. Per-block MV store `av1_copy_frame_mvs` (mvref_common.c:41).
- **1d. Single-ref translational MC.** C: the template chain `dec_build_inter_predictor`
  (decodeframe.c:681) → … → `av1_make_inter_predictor` (reconinter.c:77, TRANSLATION) →
  `inter_predictor` (reconinter.h:255) → `av1_convolve_2d_facade` (convolve.c:638) →
  `av1_convolve_{2d,x,y}_sr` (Appendix A); subpel via `dec_calc_subpel_params_and_extend`
  (decodeframe.c:651, `extend_mc_border`); sub-8×8 chroma `build_inter_predictors_sub8x8`
  (template:87). Port: wire in `aom-convolve` (first consumer); per-plane subpel-x/y +
  integer-offset derivation with chroma subsampling; border reads; SR round/shift (round_0=3,
  round_1=11). **First differential: `aom-convolve` vs exported `av1_convolve_*_sr_c`.**
- **1e. Inter block reconstruct.** C: inter arm of `decode_token_recon_block`
  (decodeframe.c:966-1041) — predict then residual per 64×64 plane chunk. Port: predict (1d) into
  the recon plane, then `reconstruct_txb` (Chunk 0) for the residual add; vartx/inter-ext-tx
  reuse the intrabc-wired readers.
- **1f. Integrate + gate.** Extend `scope_for` (conformance_corpus.rs:273) to accept frame 1.
- **Test:** frame 1 of `av1-1-b8-00-quantizer-63` i420 md5 == golden `.md5` line 2 (+ frame 0 still
  matches). Plus differentials for each new kernel (MV list inter, MC, mode parse).
- **Deps:** Chunk 0 (or reuse existing import). **Size:** XL (the skeleton; land as 1a→1f).

### Chunk 2 — NEWMV robustness + full single-ref MV-ref list
- **C:** the complete `setup_ref_mv_list` single-ref extension (mvref_common.c:753-785),
  `av1_find_best_ref_mvs` precision, `read_mv` full class/fp/hp (decodemv.c:846), `av1_drl_ctx`.
- **Port:** ensure NEWMV + DRL + all single-ref MV-list ranking is exact across the lower-q sweep.
- **Test:** ratchet to `quantizer-40`/`-20` frame 1 byte-exact. **Deps:** 1. **Size:** M.

### Chunk 3 — Switchable interpolation filter (per-block, dual)
- **C:** `read_mb_interp_filter` (decodemv.c:1034) SWITCHABLE path,
  `av1_get_pred_context_switchable_interp` (pred_common.c:30), `av1_is_interp_needed`
  (reconinter.h:420), the SMOOTH/SHARP kernels (already in aom-convolve), dual-filter gate.
- **Port:** per-direction interp filter selection + the smooth/sharp convolve variants.
- **Test:** a sweep frame that codes switchable filters byte-exact. **Deps:** 1. **Size:** M.

### Chunk 4 — bd10 (high bit depth) single-ref inter
- **C:** `av1_highbd_convolve_*_sr` (convolve.c), highbd build_inter_predictor, highbd
  reconstruct (already bd-generic in `reconstruct_txb`).
- **Port:** highbd convolve variants (u16 pipeline; the port is already u16-plane throughout).
- **Test:** frame 1 of `av1-1-b10-00-quantizer-*` byte-exact. **Deps:** 1. **Size:** M.

### Chunk 5 — Multi-reference selection (still single-prediction per block)
- **C:** `read_ref_frames` compound-vs-single decision + full single-ref tree over multiple ref
  slots; `av1_get_reference_mode_context` (pred_common.c:145); needs the multi-frame families.
- **Port:** ref-frame selection when >1 distinct reference exists (fetch `05-mv`).
- **Test:** `av1-1-b8-05-mv` early frames (single-pred blocks) byte-exact. **Deps:** 1,2.
  **Size:** M.

### Chunk 6 — Average compound (`reference_select`, compound modes, dist-wtd convolve)
- **C:** compound path of `read_inter_block_mode_info` (decodemv.c:1425), `read_inter_compound_mode`
  (:249), `av1_dist_wtd_comp_weight_assign`, `av1_dist_wtd_convolve_2d`/`_x`/`_y` (convolve.c),
  `get_comp_index_context` (pred_common.h:102), the `CONV_BUF` 16-bit intermediate + averaging.
- **Port:** compound-ref parse + 2-predictor average/dist-wtd MC. **Add compound CDF tables**
  (comp_inter, comp_ref/uni_comp/comp_ref_type, comp_group_idx, compound_idx, inter_compound_mode).
- **Test:** `05-mv`/`06-mfmv` compound frames byte-exact. **Deps:** 5. **Size:** L.

### Chunk 7 — Masked compound (wedge + diffwtd)
- **C:** `build_masked_compound`, `av1_build_compound_diffwtd_mask`, `av1_get_compound_type_mask`,
  `av1_init_wedge_masks`, `wedge_params_lookup`, `make_masked_inter_predictor`; parse
  `compound_type`/`wedge_idx`/`mask_type` (decodemv.c:1429-1479).
- **Port:** wedge mask tables + diffwtd mask + masked blend. **Deps:** 6. **Size:** L.

### Chunk 8 — OBMC
- **C:** `av1_build_obmc_inter_prediction`, `av1_setup_obmc_mask`,
  `av1_setup_build_prediction_by_{above,left}_pred`, `dec_build_obmc_inter_predictors_sb`
  (decodeframe.c:818), `read_motion_mode` OBMC arm (decodemv.c:227), `motion_mode_cdf`/`obmc_cdf`.
- **Port:** overlapped blend from above/left neighbour predictions + motion_mode parse.
- **Test:** an `05-mv`/`06-mfmv` frame that codes OBMC. **Deps:** 1 (+3). **Size:** L.

### Chunk 9 — Temporal / motion-field MV (`allow_ref_frame_mvs`)
- **C:** `av1_setup_motion_field` (mvref_common.c:1015), `motion_field_projection` (:919),
  `add_tpl_ref_mv` (:329), `get_block_position` (:881), `av1_get_mv_projection` (:27),
  `av1_copy_frame_mvs` (:41) + `tpl_mvs` buffer.
- **Port:** the projection pre-tile + `add_tpl_ref_mv` in the list build + the per-8×8 MV store
  the projection reads. **Deps:** 1,5. **Size:** L. *(Pull into chunk 1 if the first target has
  `allow_ref_frame_mvs` on — see §3 caveat.)*

### Chunk 10 — Global motion (warped)
- **C:** `read_global_motion` (parsed already), `gm_get_motion_vector` (mv.h:231),
  `av1_warp_plane`/`av1_warp_affine` (warped_motion.c), `av1_get_shear_params`, `div_lut`,
  `warped_filter`; `is_global_mv_block`/`is_nontrans_global_motion` gating.
- **Port:** the affine warp kernel + global-motion MC dispatch. **Deps:** 1. **Size:** L.

### Chunk 11 — Local warped motion (`WARPED_CAUSAL`)
- **C:** `read_motion_mode` WARPED arm, `av1_findSamples` (mvref_common.c:1118),
  `av1_find_projection`, `av1_selectSamples`, warp kernel (shared with chunk 10).
- **Port:** neighbour-sample projection → per-block affine params → warp. **Deps:** 10.
  **Size:** L.

### Chunk 12 — Reference scaling (refs differ in size) + superres-with-refs
- **C:** `av1_setup_scale_factors_for_frame` (scale.c:44), `av1_convolve_2d_scale` (convolve.c),
  `setup_frame_size_with_refs` (decodeframe.c:2065), `av1_is_scaled` dispatch.
- **Port:** scaled-MC path (SCALE_SUBPEL_BITS) + `frame_size_with_refs`. **Deps:** 1.
  **Size:** M (needed for `22-svc` + some `05-mv`).

### Chunk 13 — Interintra + skip_mode + reduced_tx_set-inter + delta segmentation-inter
- **C:** interintra (`av1_build_interintra_predictors*`, `av1_combine_interintra`;
  parse `read_interintra_info` present); `read_skip_mode` (decodemv.c:421) +
  `av1_setup_skip_mode_allowed`; `read_inter_segment_id` temporal_update.
- **Port:** the remaining inter tools. **Deps:** 6,7. **Size:** L.

### Chunk 14 — SVC / multi-layer (`22-svc`) + `06-mfmv` full + film-grain/monochrome inter frames
- **C:** operating points / `operating_point_idc`, spatial/temporal layer handling,
  `show_existing_frame` path (decodeframe.c:4602), the full `06-mfmv` motion-field stress.
- **Port:** layer selection + show-existing + the film_grain/monochrome multi-frame families
  (`23-film`/`24-monochrome` are 10-frame KEY+INTER with `show_existing_frame`).
- **Test:** all 5 inter-scoped vectors + the film/mono tails byte-exact. **Deps:** 1-13.
  **Size:** L.

**Gate-1-inter definition of done:** every `00-quantizer` (128) frame 1, every `01-size`/
`04-cdfupdate` frame 1, the `05-mv`/`06-mfmv`/`22-svc` full streams, and the `23-film`/
`24-monochrome` 10-frame streams reproduce all golden `.md5` lines byte-identically.

---

## 5. Refactor needs (decode/encode coupling inter forces)

### #24 — `reconstruct_txb` lives in `aom-encode`; `aom-decode` reverse-depends on it
- Today: `aom-decode/src/lib.rs:159` `use aom_encode::reconstruct_txb;`
  (`aom-encode/src/lib.rs:1015`). The dependency edge `aom-decode → aom-encode` is backwards
  and inter makes it worse (inter reconstruct is the SAME dequant+inv-txfm+add, just onto an MC
  prediction). `reconstruct_txb` is cleanly self-contained: `(dst:&mut[u16], stride, tx_size,
  tx_type, qcoeff, dequant, iqmatrix, bd)` → `dequant_txb` (aom-quant) + `av1_inv_txfm2d_add`
  (aom-transform).
- **Recommendation:** relocate `reconstruct_txb` (+ the `reconstruct_txb_wht` lossless variant,
  aom-decode/lib.rs:168, and the `dequant_txb` wrapper) into a shared crate that both
  `aom-decode` and `aom-encode` depend on. Options:
  - **(a) new `aom-recon` crate** (deps: aom-quant + aom-transform + aom-txb) — cleanest; also
    the natural home for the per-txb "predict-then-reconstruct" glue shared by intra & inter.
  - **(b) extend `aom-txb`** to depend on aom-quant + aom-transform and host it there (aom-txb
    already owns coefficient read/optimize). Slightly muddies aom-txb's charter.
  - Pick (a). Result: `aom-decode → aom-encode` edge is deleted.

### Where inter-prediction + inter-recon code should live
- **New `aom-inter` crate** for motion compensation: `build_inter_predictor` (single-ref),
  compound (average/dist-wtd/wedge/diffwtd), OBMC, warp/global-motion, reference scaling.
  Deps: `aom-convolve` (wire it in — currently unused), `aom-quant`/`aom-transform` (scale),
  and the MV/ref types. `aom-decode → aom-inter`. This keeps MC out of the already-large
  `aom-decode/src/lib.rs` and makes it reusable by the future inter *encoder*.
- **MV prediction:** the inter generalization of `dv_ref.rs` (compound stack, temporal
  `add_tpl_ref_mv`, `mode_context`, global-motion MV, sign-bias) can either extend `dv_ref.rs`
  in `aom-entropy` (it already owns the byte-exact spatial scan) or move to a new
  `mvref` module. Prefer extending in-place in `aom-entropy` — the spatial scan, ref-mv stack,
  and CDF context all live there already; only rename `dv_ref` → `mvref` to reflect the wider
  charter.
- **Reference-frame buffer pool + multi-frame loop** belongs in `aom-decode` (the frame driver),
  not a shared crate — it is decode-only state.

### Inter CDF default tables
- Transcribe the missing `default_*_cdf` tables (§1) from `entropymode.c`/`entropymv.c` into
  `default_cdfs.rs`, and extend the frame-context struct the decoder threads. Each new table is
  differential-checkable by round-tripping symbols against the exported C reader. This is
  spread across chunks 1/6/8 as each tool is wired (add only the tables that chunk consumes).

---

## Appendix A — MC-internal C refs (reconinter.c / reconinter_template.inc / convolve.c / warped_motion.c / scale.c)
Verified v3.14.1. Constants: `FILTER_BITS=7`, `SUBPEL_BITS=4`/`SUBPEL_MASK=15`,
`SCALE_SUBPEL_BITS=10`, `ROUND0_BITS=3`, `COMPOUND_ROUND1_BITS=7`, `DIST_PRECISION_BITS=4`,
`DIFF_FACTOR=16`, `REF_SCALE_SHIFT=14`/`REF_NO_SCALE=1<<14`.

**Template mechanism:** per-block build is in `reconinter_template.inc`, `#include`d twice with
`IS_DEC` (decoder at decodeframe.c:668-670, encoder at reconinter_enc.c:44-46). No public
`av1_build_inter_predictor(s)` symbol.

**Single-block build chain:** `dec_build_inter_predictor` (decodeframe.c:681) →
`dec_build_inter_predictors` (:672) → `build_inter_predictors` (template:242) →
`build_inter_predictors_8x8_and_bigger` (:165) | `build_inter_predictors_sub8x8` (:87, 4:2:0
chroma over >1 luma block, asserts !compound) → `build_one_inter_predictor` (:16) →
`av1_make_inter_predictor` (reconinter.c:77; TRANSLATION_PRED→`inter_predictor` reconinter.h:255 /
highbd :275, WARP_PRED→`av1_warp_plane`). Params: `av1_init_inter_params` (reconinter.h:216),
`InterPredParams` (:107), `SubpelParams` (:77). Subpel: `dec_calc_subpel_params` (decodeframe.c:565),
`_and_extend` (:651, `extend_mc_border`), `init_subpel_params` (reconinter.h:131).

**Convolve dispatch:** `av1_convolve_2d_facade` (convolve.c:638; order: 2-tap intrabc →
scaled `convolve_2d_scale_wrapper` :578 → compound `convolve_2d_facade_compound` :592 → single
`convolve_2d_facade_single` :616). `ConvolveParams` (convolve.h:21) via `get_conv_params_no_round`
(convolve.h:68). Filter params `av1_get_interp_filter_params_with_block_size` (filter.h:249),
subpel kernel `av1_get_interp_filter_subpel_kernel` (filter.h:301).

**SR kernels (final 8-bit in one shot):** `av1_convolve_2d_sr_c` (convolve.c:78, round_0=3,
round_1=11, offset_bits=19, final bits=0), `av1_convolve_x_sr_c` (:158, total shift 7),
`av1_convolve_y_sr_c` (:137, no ConvolveParams, `>>7`). Highbd: `_x_sr_c:689`, `_y_sr_c:717`,
`_2d_sr_c:737`; facade `av1_highbd_convolve_2d_facade:1246`.

**Compound (dist-wtd) kernels (16-bit CONV_BUF, blend on ref1):** `av1_dist_wtd_convolve_2d_c`
(convolve.c:293, round_1=7 → 4 extra fractional bits; ref0 writes `dst16`, ref1 blends
`tmp*fwd + res*bck >> DIST_PRECISION_BITS(4)` or `(tmp+res)>>1`, subtract offset composite, final
round_bits=4), `_x_c:408`, `_y_c:361`, `_2d_copy_c:455`. Highbd `_2d_c:906`/`_x:975`/`_y:1023`/
`_copy:1071`. **SR vs compound = round_1 (11 vs 7) + the offset-subtraction; the classic mismatch.**

**Compound modes/masks:** `av1_dist_wtd_comp_weight_assign` (reconinter.c:669,
`quant_dist_lookup_table`, MAX_FRAME_DISTANCE=31), `av1_make_masked_inter_predictor` (:629),
`build_masked_compound_no_round` (:602, `aom_lowbd_blend_a64_d16_mask`),
`av1_get_compound_type_mask` (:290), DIFFWTD `av1_build_compound_diffwtd_mask_d16_c` (:319, from
CONV_BUF, mask_base=38) / `_c` (:351, from 8-bit), wedge `av1_wedge_params_lookup` (reconinter.h:75),
`av1_init_wedge_masks` (reconinter.c:600), `get_wedge_mask_inplace` (:270). Interintra
`av1_build_interintra_predictor` (:1162), `av1_combine_interintra` (:1138 → `combine_interintra`
:1059 / highbd :1086), `av1_build_intra_predictors_for_interintra` (:1115).

**OBMC:** `av1_build_obmc_inter_prediction` (reconinter.c:935; visitors `build_obmc_inter_pred_above`
:852 / `_left` :891, `aom_blend_a64_vmask`/`hmask`), `av1_get_obmc_mask` (:774, tables `obmc_mask_*`
:752-772 — no `av1_setup_obmc_mask`), `av1_count_overlappable_neighbors` (:801),
`av1_setup_build_prediction_by_above_pred` (:980) / `_left_pred` (:1018); decoder
`dec_build_obmc_inter_predictors_sb` (decodeframe.c:818), `dec_build_prediction_by_above/left`
(:706/:736/:762/:791), `av1_setup_obmc_dst_bufs` (reconinter.c:955); iterators
`foreach_overlappable_nb_above/left` (obmc.h:20/:57).

**Warp:** consts mv.h:96-107 (WARPEDMODEL_PREC_BITS=16, WARPEDPIXEL_PREC_BITS=6,
WARPEDDIFF_PREC_BITS=10, WARP_PARAM_REDUCE_BITS=6). `TransformationType`
(flow_estimation.h:28: IDENTITY=0/TRANSLATION=1/ROTZOOM=2/AFFINE=3), `WarpedMotionParams` (mv.h:119).
`av1_warp_plane` (warped_motion.c:662) → `warp_plane` (:648) / `highbd_warp_plane` (:415) →
**`av1_warp_affine_c` (:518, the scalar 8×8-block target; highbd `av1_highbd_warp_affine_c` :287)**,
`av1_warped_filter[]` (:30). `av1_get_shear_params` (:243, `div_lut` :143, `resolve_divisor_32`
:189), `av1_find_projection` (:906, local-warp fit), `av1_init_warp_params` (reconinter.c:58,
sets WARP_PRED via `allow_warp` :31). Decoder global-MV `gm_get_motion_vector` (mv.h:231, called
decodemv.c:1142/1199/1204), frame-header parse `read_global_motion_params` (decodeframe.c:4383) /
`read_global_motion` (:4453, `!frame_is_intra_only`) → `cm->global_motion[]`. (No decoder
`av1_setup_global_motion` — decoder only parses params.)

**Reference scaling:** `scale_factors` (scale.h:28), `av1_setup_scale_factors_for_frame` (scale.c:44),
`av1_scale_mv` (:33), predicates `av1_is_scaled` (scale.h:70)/`av1_is_valid_scale` (:64)/
`av1_scaled_x/y` (:36/:45), runtime `has_scale` (reconinter.h:240). Scaled path = the `scaled=1`
branch of `av1_convolve_2d_facade` → `av1_convolve_2d_scale(_c)` (convolve.c:490, walks
`x_qn += x_step_qn` at SCALE_SUBPEL_BITS, filter idx `(x_qn & MASK) >> SCALE_EXTRA_BITS`). No
`av1_scaled_convolve` symbol. Taken when a ref's dims differ from the current frame.
