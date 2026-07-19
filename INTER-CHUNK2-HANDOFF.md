# INTER-ENCODE Chunk 2 — Handoff (encode skeleton)

Status snapshot for the inter-encode walking skeleton (INTER-ENCODE-ROADMAP.md §"chunk 2",
sub-steps 2a–2g). Goal: encode ONE single-ref translational P-frame **byte-exact** vs `aomenc`,
verified by decode-both.

## What LANDED on origin/main (verified, differential-locked)

| Sub-step | Commit | What | Gate |
|---|---|---|---|
| **2b** fixed-Q inter RC | `dfc6c58` | `aom_encode::rc::base_qindex_lowdelay_p_from_cq` — the low-delay P (inter leaf) frame `base_qindex`. Traced: `rc_pick_q_and_bounds_q_mode` → `get_active_best_quality` `is_leaf_frame && AOM_Q` returns `cq_level` (ratectrl.c:2092), i.e. `quantizer_to_qindex(cq)` (NOT the dead `rc_pick_q_and_bounds_no_stats_cq`). | `aom-bench/tests/inter_rc_qindex_diff.rs` — frame-1 coded qindex byte-matches across cq {8,12,20,32,48,60,63}; anti-vacuity: KEY qindex is boosted lower. |
| **2d.1** subpel predictor | `ad99442` | `aom_encode::inter_me::upsampled_pred` — `aom_upsampled_pred` (lowbd, USE_8_TAPS): the 8-tap fixed-phase subpel predictor; the subpel-search cost primitive. | `aom-encode/tests/upsampled_pred_diff.rs` — byte-matches real `aom_upsampled_pred_c` (2304 cells). |
| **2d.2** subpel search | `654614f` | `aom_encode::inter_me::find_best_sub_pixel_tree` — `av1_find_best_sub_pixel_tree` (SUBPEL_TREE / USE_8_TAPS, the speed-0 path). The biggest net-new ME kernel. | `aom-encode/tests/subpel_tree_diff.rs` — `(best_mv, distortion, sse, besterr)` byte-match real C (432 cells). |
| **2d.3** full-pel score | `dd59677` | `aom_encode::inter_me::get_mvpred_sse` — `av1_get_mvpred_sse` (mcomp.c:3963): the full-pel predictor SSE + coded-MV cost `av1_single_motion_search` scores the full-pel result with. | `subpel_tree_diff.rs::get_mvpred_sse_matches_real_c` (126 cells). |
| **2d.4** coded-MV rate | `dc8ae93` | `aom_encode::inter_me::mv_bit_cost` — `av1_mv_bit_cost` (mcomp.c:307): the NEWMV RD rate (weight 108/120). `mv_err_cost_entropy` (the motion-search variance-metric cost) is a shared free fn. | `subpel_tree_diff.rs::mv_bit_cost_matches_real_c` (8000 cells). |

New oracle: `aom-sys-ref/shim/me_shim.c` (`shim_upsampled_pred`, `shim_find_best_sub_pixel_tree`,
`shim_get_mvpred_sse`, `shim_mv_bit_cost`) + the `ref_*` wrappers in `aom-sys-ref/src/lib.rs`.
`me_shim` registered in `aom-sys-ref/build.rs`. `aom-encode` gained an `aom-convolve` dep (filter
tables). The tree oracle's MACROBLOCKD/SUBPEL_MOTION_SEARCH_PARAMS construction (me_shim.c) is the
template for the full-pel `av1_full_pixel_search` shim.

**So 2d — the subpel motion search + its cost primitives — is DONE and real-C-locked (the biggest
net-new inter ME work). 2b (RC) is DONE.** All leaves are real-C-locked: `upsampled_pred` (2d.1),
`aom_dist::variance` (pre-locked), `mv_err_cost_entropy` (via the tree diff), `mv_bit_cost` (2d.4).
The only ME piece left is the **full-pel search inter retarget + its differential** (below) — then
`av1_single_motion_search` is just glue (full-pel → `get_mvpred_sse` → `find_best_sub_pixel_tree`).

## Head-start inventory (REUSE — do not rebuild)

- **Full-pel ME** (`aom-encode/src/intrabc_search.rs`, 1921 LOC): `FullPelSearch` (:1166) is
  ALREADY a generic reference-plane abstraction (`refb`/`ref_off` fields); `diamond_search_sad`
  (:1252), `full_pixel_diamond` (:1319), `full_pixel_exhaustive` (:1397), `set_mv_search_range`
  (:1099). **NO C oracle yet — geometry-unit-locked only.** MV cost model: `mv_cost` (:556),
  `mv_err_cost` (:579), `mvsad_err_cost` (:593), `DvCosts`/`fill_dv_costs` (:418/:536) — forms
  generalize to inter, but tables are `MV_SUBPEL_NONE` (integer-pel); inter needs the
  subpel-precision `av1_build_nmv_cost_table` build (fp/hp arrays).
- **Encoder inter MC is ALREADY built + byte-exact**: `aom-inter::build_inter_predictor` (single-ref
  translational, lowbd, 4-tap/8-tap, dual filters, border) — the SAME `reconinter` chain the
  decoder uses (proven vs `inter_predictor` + decoder MD5). `aom-decode` already consumes it. For
  2e the encoder just needs to depend on `aom-inter` and call `build_inter_predictor`; the kernel
  is done (roadmap §5 #A satisfied).
- **Inter ref-mv list** (`aom-entropy::dv_ref::find_inter_mv_refs`, :989, commit `cdba774`) —
  byte-exact vs C, single-ref. Oracle: `shim_find_dv_ref_mvs` at a single inter ref (dec_shim.c).
- **Inter symbol WRITE layer** (`aom-entropy` partition module): `write_inter_mode`,
  `write_ref_frames`, MV coder (`av1_encode_mv`), `write_tx_size_vartx`, `write_is_inter`, all
  neighbour pred-contexts — byte-exact.
- **Inter var-tx coeff arm** (chunk 1, `aom-encode/src/var_tx.rs`): recursion + inter leaf
  differential-locked (`db90148`, `3b9278f`); prunes + pack wiring in progress (KB-15).
- **Intra RD engine** (`aom-encode`, cpu 0-9) — the inter mode loop plugs into this.
- **2-frame harness** (chunk 0, `453d145`): `aom-bench::MultiFrameEncodeCell::{translational,
  c_encode_inter, frame0_cell}` + `inter_localize::{decode_both, first_frameset_divergence}`.

## REMAINING (integration-coupled — none independently byte-testable without the RD loop)

Ordered as the roadmap suggests (structure → search wiring → RD → gate):

- **2a — encode-side ref management + inter frame-header WRITE.** NET-NEW structural. Need a
  `RefFrame` (border-extended recon Y/U/V + order_hint + saved CDFs + per-8×8 mvs) +
  `ref_frame_map[8]` + a 2-frame low-delay loop (frame 0 KEY via existing `port_encode`; frame 1
  references frame 0). The inter branch of `write_uncompressed_header_obu` (ref-signaling,
  `frame_size_with_refs`, interp/mv-precision/ref-frame-mvs flags) — the READ side is in
  `aom-entropy/src/header.rs`; the WRITE assembly + values are net-new (STATUS.md has the anchored
  write pieces). C: `av1_encode_strategy` low-delay path, `choose_primary_ref_frame`,
  `define_gf_group_pass0`. **Belongs in `aom-encode`.**
- **2c — wire `find_inter_mv_refs` into the encode ref-frame loop.** The port fn exists + is
  byte-exact; only the RD-loop call site is missing (needs 2f to exist). Restore
  `mode_context`/`newmv_count`/sign-bias/identity-GM if the reduced single-ref path dropped them
  (roadmap §2.3).
- **2e — wire `aom-inter` MC into `aom-encode`.** Add `aom-inter = { path = "../aom-inter" }` to
  `aom-encode/Cargo.toml`; call `aom_inter::build_inter_predictor` to build a candidate's inter
  predictor (per plane, chroma subsampling). Kernel is proven; only the caller (in 2f) is new. A
  confirming differential vs `av1_enc_build_inter_predictor` is optional (MC already proven via the
  decoder). **Add SMOOTH/SHARP filter params to `aom-inter` for the interp-filter search.**
- **2f — `handle_inter_mode` RD (single-ref, SIMPLE motion mode).** The integration center of
  gravity. C: `av1_rd_pick_inter_mode_sb` (rdopt.c ~6180) + `set_params_rd_pick_inter_mode` (:4331)
  + `handle_inter_mode` (:3063), reduced to NEWMV/NEAREST/NEAR/GLOBALMV single-ref, SIMPLE-only, no
  compound; interp search (`av1_interpolation_filter_search`, dual-filter-off); inter var-tx (chunk
  1). Wire the ported ME (`inter_me::find_best_sub_pixel_tree` + the full-pel search) +
  `find_inter_mv_refs` + `build_inter_predictor` + var-tx + the inter symbol writers + the MV coder
  into the existing partition/leaf search. Add the missing inter CDF default tables the costs
  consume (several already in `default_cdfs.rs`). `av1_single_motion_search`
  (motion_search_facade.c:120) is the glue that runs full-pel then subpel — mirror it: build the
  full-pel `FULLPEL_MOTION_SEARCH_PARAMS`, run the diamond (retarget `FullPelSearch` to the ref
  frame — split its single `stride` into src/ref strides; the SAD/variance kernels already take
  both), then `find_best_sub_pixel_tree` with the fullpel start MV.
- **2g — decode-both byte-exact gate.** Wire the P-frame into `MultiFrameEncodeCell`; a
  `port_encode_inter` (frame 0 KEY + frame 1 P), then `decode_both(port_stream, c_encode_inter())`
  == 0 divergence at the §3 config. **Stay in the decoder's byte-exact envelope** (chunk-0 finding:
  mono / luma-inter / zero-MV 4:2:0 / cpu 2,5 4:2:0; arbitrary-content chroma-inter decode is a
  concurrent decoder-track fix).

## Suggested next kernel (if continuing before 2f integration)

- **Full-pel search inter retarget + first real-C differential** (`av1_full_pixel_search`,
  mcomp.c:1768). Would lock the ENTIRE motion search (full-pel + subpel already done). Needs:
  split `FullPelSearch.stride`, expose the diamond/mesh `pub(crate)`, and a
  `shim_full_pixel_search` (FULLPEL_MOTION_SEARCH_PARAMS + a calloc'd MACROBLOCKD + the SAD fn-ptr
  `sdf`; same shim shape as `shim_find_best_sub_pixel_tree` in `me_shim.c`). NOTE: `intrabc_search.rs`
  is also touched by the concurrent KB-15 agent (~line 1890) — keep edits additive (the struct
  region ~1166 and a new inter entry are far from 1890).
- **`av1_build_nmv_cost_table`** (full precision, encodemv.c) — the real MV cost tables the subpel
  tree currently takes as synthetic input. Differentiable vs exported C; self-contained.

## Coordination

Work off origin/main; own `aom-encode` (ME/MC/RD) + `aom-bench` (harness) + `aom-sys-ref`
(me_shim). Concurrent agents touch `aom-decode`/`aom-inter`/`aom-entropy`(read) + `aom-encode`
(var-tx). Rebase-additive. Author `aom-rs <lilith@imazen.io>`, trailer `Co-Authored-By: Claude
Opus 4.8`. Push `HEAD:main`, verify `git merge-base --is-ancestor HEAD origin/main`. Symlink
`reference/libaom` + `conformance/data` from `/root/aom-rs/`.
