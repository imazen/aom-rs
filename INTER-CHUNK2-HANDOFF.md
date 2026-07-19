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
| **2d.5** MV cost tables | `54dd141` | `aom_encode::intrabc_search::fill_nmv_costs(precision, joints, comp0, comp1)` — `av1_build_nmv_cost_table` (encodemv.c:294): the REAL per-frame inter MV cost tables (`x->mv_costs`) the motion search consumes, at LOW/HIGH precision. Generalizes the intrabc `fill_dv_costs` (which is now this at `MV_SUBPEL_NONE`) with the fp/hp cost fills. | `aom-encode/tests/nmv_cost_table_diff.rs` — default + 24 random contexts × NONE/LOW/HIGH byte-match the 4 joint costs + both full magnitude tables; anti-vacuity + `fill_dv_costs` tie. |
| **2d.6** full-pel search | `7188476` | `aom_encode::intrabc_search::full_pixel_search_inter(...)` — `av1_full_pixel_search` (mcomp.c:1768) inter SIMPLE_TRANSLATION speed-0 NSTEP diamond, mesh off. Retargets the intrabc `FullPelSearch` (stride split into src/ref) + the real 2d.5 nmv tables + `get_fullmv_from_mv` rounding. **First real-C validation of the port's full-pel diamond.** | `aom-encode/tests/full_pixel_search_diff.rs` — `(var_cost, best_row, best_col)` byte-match real C across ~670 cells (sizes × random + converging content × integer/subpel ref MVs × step params). |

New oracle: `aom-sys-ref/shim/me_shim.c` (`shim_upsampled_pred`, `shim_find_best_sub_pixel_tree`,
`shim_get_mvpred_sse`, `shim_mv_bit_cost`, **`shim_build_nmv_cost_table`**, **`shim_full_pixel_search`**)
+ the `ref_*` wrappers in `aom-sys-ref/src/lib.rs`. `me_shim` registered in `aom-sys-ref/build.rs`.
`aom-encode` gained an `aom-convolve` dep (filter tables). The full-pel shim builds a
`FULLPEL_MOTION_SEARCH_PARAMS` field-by-field — the NSTEP `search_site_config` via the real
`av1_init_motion_compensation[NSTEP]` (level 0, ref stride), per-size `aom_*_c` SAD/variance fn
ptrs, mesh forced off (`force_mesh_thresh = INT_MAX`).

**So 2d — the ENTIRE single-ref motion search — is DONE and real-C-locked.** All primitives:
full-pel (`full_pixel_search_inter`, 2d.6), subpel tree (2d.2), `upsampled_pred` (2d.1),
`get_mvpred_sse` (2d.3), `mv_bit_cost` (2d.4), the real MV cost tables (`fill_nmv_costs`, 2d.5),
`aom_dist::variance`/SAD (pre-locked). 2b (RC) is DONE. **The only ME piece not built is the
composition glue `av1_single_motion_search`** (full-pel → subpel) — but both halves are C-locked,
so it is pure glue (no new kernel). Follow-ups deferred as speed≥1 / later chunks: the inter
exhaustive mesh (needs `mv_sf->mesh_patterns`, distinct from intrabc's), the full-pel `cost_list`
(`calc_int_cost_list`, only used by the pruned-subpel/DRL paths — the speed-0 SUBPEL_TREE does not
read it), and `second_best_mv`.

## Head-start inventory (REUSE — do not rebuild)

- **Full-pel ME** (`aom-encode/src/intrabc_search.rs`): `FullPelSearch` now carries separate
  `src_stride`/`ref_stride` (equal for intrabc); `diamond_search_sad`, `full_pixel_diamond`,
  `full_pixel_exhaustive`, `set_mv_search_range`. **NOW real-C-locked** (2d.6,
  `full_pixel_search_diff.rs`) via `pub full_pixel_search_inter(...)` — call it for inter.
  MV cost model: `mv_cost`, `mv_err_cost`, `mvsad_err_cost`, `DvCosts`; the inter cost tables are
  `pub fill_nmv_costs(precision, joints, comp0, comp1)` (2d.5, `MV_SUBPEL_LOW`/`HIGH`) —
  `fill_dv_costs` is that at `MV_SUBPEL_NONE`.
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

## Next work (the ME kernel surface is DONE — 2d complete; next is 2f integration)

Both pre-2f ME kernels have landed (2d.5 nmv cost tables `54dd141`, 2d.6 full-pel search `7188476`);
**every motion-search primitive is now real-C-locked.** The remaining path to the byte-exact gate is
the integration (2a/2c/2e/2f/2g above), whose center of gravity is **2f `handle_inter_mode` RD** —
none of it is independently byte-testable without the RD loop (unlike the kernels). Suggested order:

1. **`av1_single_motion_search` glue** (motion_search_facade.c:120) — pure composition of the two
   locked halves: `full_pixel_search_inter` (best full-pel) → `find_best_sub_pixel_tree` (start =
   best full-pel MV) → `mv_bit_cost` (NEWMV rate). No new kernel; a real-C differential needs a
   heavier shim (full `MACROBLOCK`/`AV1_COMP` state: `mv_costs`, `mbmi_ext`, `mv_search_params`,
   `sf.mv_sf`, the inert-at-lag0 TPL gather) — or validate the composition on converging content.
2. **2e MC wire** — add `aom-inter = {path="../aom-inter"}` to `aom-encode/Cargo.toml`; call
   `aom_inter::build_inter_predictor` per plane. Kernel already byte-exact (decoder uses it); only
   the 2f caller is new.
3. **2f `handle_inter_mode` RD** (single-ref, SIMPLE motion, NEWMV/NEAREST/NEAR/GLOBALMV) — the
   integration center of gravity (see 2f above). Plug the locked ME + `find_inter_mv_refs` +
   `build_inter_predictor` + var-tx + the inter symbol writers into the leaf RD at
   `partition_pick.rs:~1375`. Add the inter mode/drl/ref cost tables the RD consumes.
4. **2a ref buffer + 2g gate** — a `RefFrame` (border-extended recon) + the 2-frame loop; the inter
   frame-header WRITE machinery is ALREADY byte-exact (`write_frame_header_obu` INTER branch,
   `aom-entropy/src/header.rs:1486`, anchor-validated) — 2a is value-derivation + ref mgmt, not the
   bit-writing. Then decode-both at the §3 mono/luma config.

Deferred ME follow-ups (speed≥1 / not needed for the speed-0 SIMPLE gate): the inter exhaustive mesh
(`mv_sf->mesh_patterns`), the full-pel `cost_list` (`calc_int_cost_list`), `second_best_mv`.

## Coordination

Work off origin/main; own `aom-encode` (ME/MC/RD) + `aom-bench` (harness) + `aom-sys-ref`
(me_shim). Concurrent agents touch `aom-decode`/`aom-inter`/`aom-entropy`(read) + `aom-encode`
(var-tx). Rebase-additive. Author `aom-rs <lilith@imazen.io>`, trailer `Co-Authored-By: Claude
Opus 4.8`. Push `HEAD:main`, verify `git merge-base --is-ancestor HEAD origin/main`. Symlink
`reference/libaom` + `conformance/data` from `/root/aom-rs/`.
