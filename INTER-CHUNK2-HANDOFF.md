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
| **2d.7** single_motion_search | `4da5829` | `aom_encode::inter_me::single_motion_search(&SingleMotionSearchParams) -> SingleMotionResult` — `av1_single_motion_search` (motion_search_facade.c:120) glue, reduced to single-ref SIMPLE_TRANSLATION speed-0. Composes the two C-locked halves: `set_mv_search_range` → `full_pixel_search_inter` (2d.6) → (unless `force_integer_mv`) `set_subpel_mv_search_range` + `find_best_sub_pixel_tree` (2d.2) → `mv_bit_cost` (2d.4, `MV_COST_WEIGHT`=108). Drops the lag-0/speed-0-inert arms (TPL gather, `skip_fullpel_search_using_startmv_refmv`, second_best_mv/cost_list). **The entire single-ref ME is now composed + callable — the RD loop (2f) calls this for NEWMV.** | `aom-encode/tests/single_motion_search_composition.rs` — (1) glue faithfulness vs a hand-composed pipeline (200+ cells × sizes/ref-MVs/step/force-int); (2) convergence to the true shift on unimodal content incl. the guaranteed zero-MV case. (Real-C `av1_single_motion_search` differential deferred: needs a full `MACROBLOCK`/`AV1_COMP` shim; both halves are already real-C-locked, so this is pure composition.) |

New oracle: `aom-sys-ref/shim/me_shim.c` (`shim_upsampled_pred`, `shim_find_best_sub_pixel_tree`,
`shim_get_mvpred_sse`, `shim_mv_bit_cost`, **`shim_build_nmv_cost_table`**, **`shim_full_pixel_search`**)
+ the `ref_*` wrappers in `aom-sys-ref/src/lib.rs`. `me_shim` registered in `aom-sys-ref/build.rs`.
`aom-encode` gained an `aom-convolve` dep (filter tables). The full-pel shim builds a
`FULLPEL_MOTION_SEARCH_PARAMS` field-by-field — the NSTEP `search_site_config` via the real
`av1_init_motion_compensation[NSTEP]` (level 0, ref stride), per-size `aom_*_c` SAD/variance fn
ptrs, mesh forced off (`force_mesh_thresh = INT_MAX`).

**So 2d — the ENTIRE single-ref motion search — is DONE and real-C-locked, INCLUDING the
composition glue (2d.7, `4da5829`).** All primitives: full-pel (`full_pixel_search_inter`, 2d.6),
subpel tree (2d.2), `upsampled_pred` (2d.1), `get_mvpred_sse` (2d.3), `mv_bit_cost` (2d.4), the
real MV cost tables (`fill_nmv_costs`, 2d.5), `aom_dist::variance`/SAD (pre-locked); the glue
`single_motion_search` (2d.7). 2b (RC) is DONE. **The single-ref ME surface is now fully composed
and callable — the RD loop (2f) calls `single_motion_search` for NEWMV.** Follow-ups deferred as
speed≥1 / later chunks: the inter exhaustive mesh (needs `mv_sf->mesh_patterns`, distinct from
intrabc's), the full-pel `cost_list` (`calc_int_cost_list`, only used by the pruned-subpel/DRL
paths — the speed-0 SUBPEL_TREE does not read it), and `second_best_mv`.

## SESSION 2026-07-19 — status + CRITICAL blocker (read before 2f/2g)

**Landed this session:** 2d.7 `single_motion_search` (`4da5829`, verified on origin/main). The ME
surface is complete. **No byte-exact P-frame is encoded yet — the SKELETON is NOT YET MET.** The
remaining work is the 2a/2c/2e/2f/2g integration, whose center of gravity (2f) is a multi-file RD
port. Two MEASURED findings bound what a byte-exact P can even be:

1. **CRITICAL — the inter var-tx COEFF arm is NOT byte-exact yet (KB-15 blocker).** `var_tx.rs`'s
   `pick_recursive_tx_size_type_yrd` is differential-locked as GLUE but **over-searches vs real C**
   because the three NN prunes are gated OFF: `prune_tx_2D` (fires on inter sets >5 types),
   `ml_predict_tx_split`, and `model_based_prune` (`var_tx.rs:205-219`, `:870-871`, `:1055-1059`).
   ⇒ **Any inter block with a NONZERO residual will pick a different tx size/type than aomenc → NOT
   byte-exact.** The var-tx arm is BYPASSED for a SKIP block (skip_txfm=1, no coeffs). **Therefore
   the ONLY byte-exact-achievable P-frame today is a SKIP-ONLY P (zero residual).** Closing coeff
   blocks needs KB-15's three prunes ported first (a shared prerequisite — it also unblocks intrabc
   real content).
2. **The achievable first-gate target is the ZERO-MV P** (`MultiFrameEncodeCell::translational(base,
   0, 0)` → `frame1 == frame0`): every block codes inter GLOBALMV/NEARESTMV `(0,0)` + skip=1, zero
   residual, no var-tx, no MV in the bitstream. The decoder handles zero-MV 4:2:0 byte-exact
   (chunk-0 finding). It still exercises the FULL 2f brain (partition search + inter mode RD picking
   inter-skip over intra + inter symbols + costs) + 2a (header + ref buffer) + 2g (decode-both) — it
   just removes coeffs/MV-coding/subpel from the bitstream. A translational (nonzero-MV) P at high cq
   where the residual quantizes to skip is a second target, but is riskier (edge blocks may carry
   residual → var-tx blocker). **Ratchet: zero-MV skip P first, then translational-skip, then (after
   KB-15 prunes) coeff blocks.**

## SESSION 2026-07-19b — 2a LANDED + zero-MV target GROUNDED (read before 2f)

**2a is DONE + on origin/main (`87630cb`).** `aom-encode::inter_frame`:
`derive_lowdelay_p_frame_header` DERIVES the §3 low-delay P `FrameHeaderObu` byte-exact vs a real
aomenc frame-1 header (gate `aom-bench/tests/inter_header_derive_diff.rs`: 16 cells cq{20,40,60,63}
× {64,128}² × {mono,420}, compared against the ACTUAL stream bytes — NOT a reader→writer round-trip,
which is LOSSY for inter). Also `RefFrame` (encode-side stored reference; decoder-RefFrame shape).

**MEASURED corrections to this handoff's §"Next work" seam map (verify, don't trust):**
- `interp_filter` is a PER-FRAME RD decision, NOT a constant: measured cq20→`MULTITAP_SHARP(2)`,
  cq60→`EIGHTTAP_REGULAR(0)` on the same content. The old "interp_filter=SHARP/SWITCHABLE" note is
  wrong. It is recon/RD-dependent (like LF levels) → a `LowDelayPHeaderParams` INPUT (bootstrapped
  from the reference until the frame interp-filter search is ported; that search is a 2f follow-up).
- `allow_high_precision_mv` IS derivable: `qindex < HIGH_PRECISION_MV_QTHRESH(128)` (mv_prec.c:411).
  The old "allow_high_precision_mv=1" note is wrong (measured false at cq60/qindex240).
- Exact real frame-1 header (cq60 64² 420, all VERIFIED-derived except the passed-in tail):
  `frame_type=1 show_frame=1 error_resilient=0 disable_cdf_update=0 order_hint=1 primary_ref=7
  refresh_frame_flags=0x02 ref_map_idx=[0;7] frame_refs_short_signaling=0 allow_hp=0 interp_filter=0
  switchable_motion_mode=0 allow_ref_frame_mvs=0 tx_mode_select=0 reference_mode_select=0
  skip_mode_allowed=0 reduced_tx_set=0 gm=identity`; recon-dependent tail `loopfilter.filter_level=
  [6,6] u=9 v=9 mode_ref_delta_enabled=1`, `cdef off`. Header = 109 bits (14 B); frame1 payload 16 B
  (⇒ tile ≈ 2 B). Non-obvious derivation gotchas the gate caught: set `prefix.superres_upscaled_
  {w,h}=max_frame_{w,h}` (else `frame_size_override_flag` mis-codes 1) and resolve the single-tile
  `tile_info` (`uniform_spacing=true,cols=rows=1,log2=0`) — the seq template carries only tile LIMITS.

**ZERO-MV target CONFIRMED ACHIEVABLE (empirical, all configs):** aomenc's zero-MV P
(`translational(base,0,0)`) decodes BYTE-IDENTICAL to frame 0's recon (`frame1==frame0`) at EVERY
cq{20,40,60,63} × {64,128}² × {mono,420} — a pure reference-copy / all-skip P. So every block is
inter GLOBALMV/NEARESTMV(0,0) + skip=1, zero residual (dodges the var-tx blocker). The tile is ~2 B.

**KEY 2f SIMPLIFICATION (verified):** the P has `primary_ref_frame = NONE`, so the frame uses
**DEFAULT inter CDFs** — the inter mode/ref/skip COST tables derive from `DEFAULT_*` inter CDFs
directly (no `FrameContext` threading needed, contra the "needs a full FrameContext" seam note).
All inter default CDFs already exist in `aom-entropy` default_cdfs.rs (intra_inter, single_ref,
newmv, zeromv, refmv, drl, skip, nmv). **2e is DEFERRABLE for the zero-MV target** (mv=(0,0) →
plain block copy of the co-located ref block; no `aom-inter` wiring needed for the skeleton).

**2g gate mechanics (confirmed):** `inter_localize::decode_both` compares decoded PIXELS (not
bitstream bytes) — `aom-bench/src/inter_localize.rs:258`. So `decode_both(port,c)==None` = the port's
all-skip P decodes identically to aomenc's (pixel-match). Byte-identity is the stronger stretch
(same partition+modes as aomenc). `MultiFrameEncodeCell` + `translational` + `c_encode_inter` +
`frame0_cell().port_encode(bootstrap)` all exist; need a new `port_encode_inter` (frame 0 KEY via
the existing byte-exact path → decode it to get the RefFrame → encode frame 1 P via 2f →
`assemble_obu_frame_single_tile(derived_p_header, 0, tile_bytes)` → concat). `obu_assemble.rs`
already has `assemble_obu_frame_single_tile`.

## SESSION 2026-07-19c — 2f PACK LANDED: the P-frame payload is BYTE-EXACT (single-block)

**Landed + verified on origin/main this session** (`41ac27f`, `3550c7d`, `9f2d1d5`, `572215b`):

| Piece | Commit | Gate |
|---|---|---|
| Inter mode/ref cost tables + `InterFrameCdfs` (P uses DEFAULT CDFs, primary_ref=NONE) | `41ac27f` | `inter_costs` unit tests incl. `default_inter_cdfs_match_libaom` (the port's tables ARE libaom's `default_{newmv,zeromv,refmv,drl}_cdf` inverted) |
| Inter RD arm `inter_rd::rd_pick_inter_mode_sb` (skip-only, search-free modes) | `3550c7d` | 3 unit tests incl. `high_residual_block_declines` |
| **`inter_pack::write_inter_leaf_mode_info` + the TILE BYTE GATE** | `9f2d1d5` | **`inter_pack_tile_diff.rs` — the port's tile == aomenc's tile, byte for byte** |
| **Full frame payload gate + multi-block pin** | `572215b` | header+tile == aomenc's whole frame-1 payload; 2-SB case pinned |

### BYTE-EXACT (asserted, not pinned)

- **`zero_mv_p_tile_bytes_byte_exact_vs_aomenc`** — the port's coded TILE for the §3
  zero-MV P equals `aomenc`'s, byte for byte. Every prediction context is DERIVED
  (partition/skip/intra_inter/single-ref) and mode+MV+mode_context come from the port's
  `find_inter_mv_refs`; nothing is copied from the reference.
- **`zero_mv_p_frame_payload_byte_exact_vs_aomenc`** — the port assembles frame 1's WHOLE
  OBU payload (derived header + tile) equal to `aomenc`'s. **Honest bootstrap:** the three
  recon-dependent header fields (LF levels/deltas, CDEF, frame `interp_filter`) come from
  the reference, exactly as `LowDelayPHeaderParams` specifies. The tile is derived from nothing.
- The derived P header is byte-exact on a **NON-SQUARE 64x128** frame too — extends 2a,
  whose gate only covered square {64,128}².
- **`tile_byte_gate_discriminates_a_wrong_mode`** — coding GLOBALMV instead of NEARESTMV does
  NOT match, so the gate can fail (anti-vacuity).

### GROUND TRUTH — measured, not assumed

Read out of `aomenc`'s own stream with the instrumented libaom decoder
(`/root/aom-inspect/examples/inspect`; the IVF comes from the new `aom-bench`
`dump_inter_stream` example):

- **64×64 zero-MV P** = ONE `PARTITION_NONE` 64×64 block, **NEARESTMV**, ref `(LAST, NONE)`,
  `SIMPLE_TRANSLATION`, skip=1, TX_64X64, qindex 240. Frame OBU 18 B, tile 2 B.
- Per-symbol accounting → the exact syntax is **partition → skip(1) → is_inter(1) →
  ref_frames(3 binary) → inter_mode(3 binary)**. Nothing else: no cdef, delta-q, DRL, MV,
  motion-mode, interp-filter, tx-size or coeff symbols.

### TWO CORRECTIONS to this document's earlier sections (verified against C — do not re-trust the old text)

1. **The pack order below ("`write_is_inter(1)` → `write_ref_frames` → `write_inter_mode` →
   `write_skip(1)`") is WRONG.** Real C (`pack_inter_mode_mvs`, bitstream.c:1092) writes
   **SKIP BEFORE is_inter**, with `write_cdef` and `write_delta_q_params` between them. The
   prologue is shape-identical to the KEY writer's; only step 7 onward differs.
2. **The mode is NEARESTMV**, not "GLOBALMV/NEARESTMV". Measured.

Also better than this doc claimed: the inter symbol layer already has the COMPOSITE writers
`write_inter_prefix` (partition.rs:2946), `write_inter_mode_drl` (:3187) and
`write_inter_mode_tail` (:3220) — the pack side was wiring, not new symbol code.

Under the §3 envelope the whole tail writes NOTHING, each by an explicit C gate: no DRL
(NEARESTMV/GLOBALMV are not in the drl-coded set), no MV (NEW* only), `write_motion_mode`
collapses at `switchable_motion_mode=0` (bitstream.c:280-287), and `write_mb_interp_filter`
returns early unless the FRAME filter is SWITCHABLE (bitstream.c:638).

### ALSO USEFUL: the §3 P codes TX_MODE_LARGEST

`derive_lowdelay_p_frame_header` sets `tx_mode_select = false`, verified byte-exact against the
real header. So `av1_txfm_search` takes the **uniform** `av1_pick_uniform_tx_size_type_yrd`
arm (tx_search.c:3831), not `av1_pick_recursive_tx_size_type_yrd`. **The KB-15 var-tx blocker
does NOT apply to this frame config** — worth re-checking before assuming coeff blocks here
need the var-tx prunes.

### PINNED OPEN — the multi-block (2-superblock) tile

`zero_mv_p_two_superblock_tile_pinned_divergent` asserts the divergence is PRESENT (fails on
match → promote). 64×128 cq60, one tile, two stacked `PARTITION_NONE` 64×64 SBs, both
NEARESTMV: **aomenc tile `[f2,24,80]` (3 B) vs port `[99,24]` (2 B)** — the port codes FEWER
bits. Neighbour-context derivation IS live (SB1: skip_ctx 0→1, single-ref p1/p3/p4 1→2,
mode_context 0→51).

**RULED OUT — do not re-chase:** the frame header (byte-identical, asserted); block 0 (proven
byte-exact at the same contexts); symbol ORDER (matches C AND the accounting for both blocks);
block 1's `mode_context` (a sweep over all 72 structurally valid encodings found NO value that
reproduces aomenc's tile); a tile-flush padding rule (`aom_stop_encode`, bitwriter.c:21, is a
plain `od_ec_enc_done` — the byte gap is real coded bits).

**NEXT CONCRETE STEP:** libaom's decoder REJECTS the port's two-SB stream outright, so the tile
desyncs rather than merely differing. Diff the instrumented decoder's per-symbol accounting
between aomenc's stream and the port's at block 1 — that shows directly whether a symbol goes
out on a different CDF row (the dominant failure mode per CLAUDE.md's desync note) or is
omitted. Suspects in order: the `skip` CDF row for block 1; the single-reference blob's
per-context row selection once counts are nonzero; the partition CDF row after
`update_ext_partition_context`.

### STILL MISSING for a port-DRIVEN P frame (state coverage honestly)

The pack and the RD arm exist and are gated, but they are **not yet wired into the partition
search / `pack_tile`**, so the port does not yet CHOOSE the inter block end-to-end. Remaining:
`PickFrameCfg::inter` + `InterLeafArgs` construction in `leaf_pick_sb_modes`; the `rd_pick.rs`
step-6 inter arm; the `encode_b_intra_dry` inter recon arm; the `pack_leaf` inter branch; and
`MultiFrameEncodeCell::port_encode_inter`. Rungs 2 (NEWMV/subpel) and 3 (real motion search)
are untouched.

## SESSION 2026-07-23 — RUNG 1 LANDED: the port's OWN SEARCH codes the zero-MV P byte-exact (single-SB); 64x128 root MEASURED

**THE SKELETON IS MET at single-SB scope.** The §7-step build plan below was executed and the
end-to-end gate is GREEN: `aom-bench/tests/inter_e2e_search.rs` —
`zero_mv_p_own_search_64x64_cq60_420_byte_exact` + the cq ladder (64² × cq{20,40,60,63} 4:2:0 +
cq{20,60} mono) assert the port's frame-1 OBU payload — derived header + a tile whose partition
tree, inter mode, contexts and skip all come from the port's OWN `pack_tile` search —
BYTE-IDENTICAL to `aomenc`'s, plus decode-both pixel identity. Wired exactly as planned:
`PickFrameCfg::inter` (`InterSearchCfg`) + per-leaf `InterLeafArgs` in `leaf_pick_sb_modes`
(ref-MV scan + intra_inter/single-ref/skip ctx from the DV grid); the `rd_pick.rs` step-6b inter
arm (inter evaluated after intra but WINS ties, matching C's inter-first `av1_default_mode_order`;
the intra candidate pays `ref_cost_intra_in_inter` on P frames) + the intra-missed-budget rescue;
the `encode_b_intra_dry` inter recon arm (co-located ref copy, skip entropy resets, padded
plane-bsize chroma extents, `SbEncodeEnv::ref_frame`); the `pack_leaf` inter branch
(`write_inter_leaf_mode_info` at grid-derived contexts, loud-fail on intra-in-P);
`InterFrameCdfs` threaded through `pack_tile_lr`/`pack_sb` with a per-SB
INTERNAL_COST_UPD_SB inter cost refresh; `inter_rd` fixes (C mode order
NEARESTMV→NEARMV→GLOBALMV; padded plane-bsize chroma sse per `set_skip_txfm`; border-replicating
ref reads); harness `MultiFrameEncodeCell::port_encode_inter_p` + `parse_inter_2frame_reference`.

**The "2-SB tile" divergence is ROOT-CAUSED (instrumented sibling libaom, byte-inert,
removed) — TWO findings, one fixed, one pinned:**

1. **The §3 GOOD-mode stream codes SB128** (`mib_size_log2=5`, libaom's GOOD default at every
   speed). A 64x128 frame is ONE column-cropped 128×128 SB whose root codes a gathered 2-way
   SPLIT/VERT symbol before the two visible 64×64 children — the hand-rolled SB64 "two-superblock"
   model could never match (it omitted the root symbol); its per-symbol suspect list
   (skip/single-ref/partition CDF rows) is OBSOLETE. 64×64 frames are walk-degenerate (SB64 ==
   SB128 symbols), which is why the single-SB gates always matched. `port_encode_inter_p` now
   drives the DECLARED SB size. The old pinned test in `inter_pack_tile_diff.rs` is superseded
   (tombstone comment in place); the live pin is
   `inter_e2e_search.rs::zero_mv_p_own_search_64x128_cropped_sb128_pinned_divergent`.
2. **The residual 64×128 divergence is a SEARCH-SPACE gap: the port models no switchable
   interp-filter rate.** Measured in C at the cropped root: SPLIT rd 248,444,575 beats VERT
   because VERT's single 64×128 sub-block FAILS `av1_txfm_search`'s final skip-guard
   (tmprd 250,728,413 > budget 247,851,864). The overprice comes from the interp-filter search:
   during encoding the frame filter is SWITCHABLE (the coded REGULAR is post-hoc), and
   `use_more_sharp_interp = 1` (speed_features.c:1139 — GOOD base, EVERY speed, non-boosted
   frames) gives SHARP a `mul=90` (10%) rd discount in `interpolation_filter_rd`; on the
   dist-heavy 64×128 model rd SHARP displaces REGULAR and costs rs 3931 (ctx 3) vs 109. The
   port's rs=0 leaf is ~3822 cheaper → picks VERT → tile `[d7,a0]` vs C `[f2,24,80]`.
   **Fix (the next rung):** port the CURVFIT model-rd (`MODELRD_TYPE_INTERP_FILTER =
   MODELRD_CURVFIT`, model_rd.h:31 — `av1_model_rd_curvfit` + tables + differential),
   `av1_get_pred_context_switchable_interp` (encode side), the switchable-interp cost table, and
   the reduced §3 interp search (zero-MV ⇒ identical predictions ⇒ pure rate/model compare with
   the SHARP mul=90 accept) — then add the winner's rs to the inter leaf's mode_rate and
   skip-guard rd. NOTE this also means C's per-block inter rate ALWAYS carries rs (the 64×64
   children pay 109 each) — the single-SB cells matched because no decision there was near a tie;
   for C-exact partition RD the rs model is required everywhere.

**Also DISCOVERED + PINNED (`good_usage_key_frame0_pinned_divergent`):** the port's KEY encode
of frame 0 at GOOD usage (`usage=0`, the §3 inter context) is NOT byte-exact (header 2 bytes
longer + mid-tile divergence at 64² cq60) — every landed KEY byte gate is ALLINTRA; the chunk-0
"frame-0 control" was decode-side only. A GOOD-vs-ALLINTRA speed-feature/header-derivation gap,
independent of the inter wiring (`PickFrameCfg::inter` is None there). Closing it is required for
full-STREAM byte identity of the 2-frame clip (the frame-1 payload gates are unaffected).

**SAME-SESSION FOLLOW-UP — the interp-rate model LANDED and the 64x128 cell is BYTE-EXACT.**
`crates/aom-encode/src/interp_rd.rs`: `av1_model_rd_curvfit` (bit-exact vs the REAL exported C
fn, `curvfit_diff.rs`; tables via `xtask/transcribe_curvfit.py`), `model_rd_with_curvfit`,
`SwitchableInterpCosts` + `pick_interp_filter_zero_mv` (the reduced zero-MV search with the
SHARP mul=90 accept); per-leaf `get_pred_context_switchable_interp` over the DV grid (which now
stamps each inter block's winner filter); rs joins the leaf mode_rate + skip-guard. GOTCHA:
`SWITCHABLE_INTERP_RATE_FACTOR` is **1** (rd.h:58). Gates: the 64x128 pin FLIPPED and is
promoted (`zero_mv_p_own_search_64x128_cropped_sb128_byte_exact` + a cq20 64x128 cell in the
map run); mono cq40 closed. NEWLY EXPOSED + PINNED: {420 cq20, 420 cq40, mono cq20} diverge
because C's **`av1_simple_motion_search_term_none`** (partition_strategy.c:809 — linear model
over sms + NONE-rd features; GOOD `!frame_is_intra_only`, live at speed 0) TERMINATES the
partition search after NONE at low cq while the port searches SPLIT (cheap ctx-0 REGULAR
children win). Those cells' earlier byte-match was a two-wrongs-cancel (rs=0 everywhere).
NEXT RUNG (named): port `simple_motion_search_prune_part_features` (the sms runs on the ported
full-pel search) + the per-bsize `term_none` linear models (+ `simple_motion_search_prune_rect`,
measured firing at interior nodes). Rebuild recipe for the throwaway instrumented sibling C
(removed): copy `reference/libaom` @ 03087864, cmake `-DCMAKE_BUILD_TYPE=Release
-DCONFIG_MULTITHREAD=0 -DCONFIG_AV1_DECODER=0 -DCONFIG_AV1_ENCODER=1`, a tiny harness
replicating `shim_encode_av1_inter_2frame` + the `inter_e2e_search.rs` cell content (validate
byte-identity vs the shim stream FIRST, then env-gate prints with `INSTR_PART=1`).

**Note on TX_MODE_LARGEST (corrects the §3 assumption below):** during the SEARCH C runs
TX_MODE_SELECT — the coded LARGEST header is a POST-encode demotion when `txb_split_count == 0`.
The all-skip zero-MV P never reaches coeffs so the port's skip-arm model still byte-matches, but
a COEFF-arm rung must run the var-tx machinery even when the final header says LARGEST.

## PRECISE 2f BUILD PLAN (zero-MV skip-only P — mirror the intrabc skip arm)

The intrabc SKIP arm is the exact template; the inter zero-MV skip arm is a SIMPLER mirror (the
predictor is a plain COPY of the co-located reference block — no DV, no interp). Verified seams:

1. **`LeafWinner` (`encode_sb.rs:167`) — add inter fields** beside the intrabc DV fields (`:222`):
   `is_inter: bool`, `ref_frame: i8` (LAST=1 for single-ref), `inter_mode: u8` (GLOBALMV/NEARESTMV),
   `mv: (i32,i32)` (always (0,0) for the target). Default them so all existing constructors compile
   (grep the ~dozen `LeafWinner {` sites; the intrabc fields show the pattern).

2. **`encode_b_intra_dry` (`encode_sb.rs:521`) — add an inter arm** modeled on the intrabc arm
   (`:555-627`), placed as a sibling `if winner.is_inter { … }`. Deltas from intrabc:
   - Predictor = a COPY of the reference frame's co-located block (zero MV): read from the passed-in
     `RefFrame` Y/U/V at `(mi_row*4, mi_col*4)` (chroma subsampled) into recon — NOT
     `intrabc_predict_*` from the current recon. `SbEncodeEnv` needs a `ref_frame: Option<&RefFrame>`
     added (thread it through; intra/intrabc pass None).
   - Everything else IDENTICAL: reset coeff entropy ctx (`above_ectx/left_ectx …fill(0)`), stamp skip
     txfm ctx (`above_tctx/left_tctx …fill(width/height*4 px)`), return empty txbs (`:598-626`).
     This is ALL the zero-MV skip block needs (no var-tx — dodges KB-15).

3. **Inter mode/ref/skip COST tables** — new fn in `real_costs.rs` (or a small `inter_costs.rs`),
   sourced from the DEFAULT inter CDFs (primary_ref=NONE ⇒ defaults; NO FrameContext). Use
   `aom_txb::cost_tokens_from_cdf` on: `DEFAULT_INTRA_INTER` (is_inter), `DEFAULT_SINGLE_REF`
   (ref = LAST: single_ref_p1..p3 path), `DEFAULT_NEWMV`/`DEFAULT_ZEROMV`/`DEFAULT_REFMV`
   (write_inter_mode cascade, `partition.rs:322` `write_inter_mode` is the exact symbol order), and
   the skip CDF. The mode_context (from 2c) indexes newmv/zeromv/refmv ctx. All in default_cdfs.rs.

4. **2c — encode-side ref-mv grid**: extend `ModeGrid`/`DvCell` (`partition_pick.rs:317`,
   `intrabc_search.rs:1118` `to_nbr` hardcodes INTRA_FRAME/NONE) to carry `ref_frame`+`mv`+`mode`;
   call `aom_entropy::dv_ref::find_inter_mv_refs` (`dv_ref.rs:989`) with a `DvGrid` closure over the
   stamped cells (decoder template `aom-decode/src/lib.rs:2419`). For the all-skip zero-MV P every
   neighbour is inter (0,0), so nearest/near/global all resolve (0,0); you only need `mode_context`.
   25 `grid.stamp` sites carry the new fields (bulk in `stamp_grid_from_tree` `partition_pick.rs:3371`).

5. **`rd_pick.rs:422` — add an inter arm** as a sibling to the intrabc step-6 arm. Compute the
   inter-skip RD: `rate = is_inter_cost + ref_cost + inter_mode_cost(mode,mode_ctx) + skip_cost[1]`,
   `dist = sse(source − ref_block)` (zero-MV predictor = ref block copy; use the visible-clipped sse
   like the intrabc skip arm's `set_skip_txfm` sse). Compete vs the assembled intra `rd` (take min),
   overwrite the winner tuple + set `is_inter`. Add inter fields to `RdPickIntraBest` (`:196-208`
   beside the intrabc fields). Gate the arm on a new `PickFrameCfg::inter` (`partition_pick.rs:506`,
   mirror the `intrabc:` field `:595`) + `InterLeafArgs` (mirror `IntrabcLeafArgs`; carry the
   `RefFrame`, cost tables, mode_context).

6. **`pack_leaf` (`pack.rs:377`) — add inter writes**: `write_is_inter(1)` → `write_ref_frames`
   (single-ref LAST) → `write_inter_mode(newmv,zeromv,refmv, mode, mode_ctx)` (`partition.rs:3079`)
   → `write_skip(1)`; SKIP the tx/coeff syntax (the intrabc skip gate at `:499` is the template).
   All these writers exist + are byte-exact in the aom-entropy partition module. NO MV coded
   (GLOBALMV/NEARESTMV don't code a diff).

7. **2g — `port_encode_inter` on `MultiFrameEncodeCell` (`aom-bench/src/lib.rs`)**:
   (a) frame 0 KEY via the existing `frame0_cell().port_encode(bootstrap)` (byte-exact);
   (b) DECODE the port frame-0 stream (`aom_decode::frame::decode_frames`) → build the `RefFrame`
       from frame 0's recon (border-extend Y/U/V);
   (c) derive the P header via `derive_lowdelay_p_frame_header` (bootstrap interp_filter + LF/cdef
       from the real frame-1 parse until their derivation lands);
   (d) search+pack the P tile via the 2f path → `assemble_obu_frame_single_tile(p_header, 0, tile)`;
   (e) concat [frame0 TU][frame1 P TU]. Gate: `decode_both(port, cell.c_encode_inter(false,false))`
       == None at `translational(base,0,0)` cq60 64² 420 cpu0 (add to a new inter-encode test).
   decode_both==None (pixel-match) is the mechanical gate; ALSO assert byte-identity of the frame-1
   TU vs aomenc's (the stronger claim — the port must pick NONE partition + inter-skip like aomenc).

Order: 3 (costs) → 1+2+4 (winner/grid plumbing) → 5 (rd arm) → 6 (pack) → 7 (gate). The whole set is
untestable until (7); build it as one arc, keep the tree COMPILING at each step (add fields with
defaults, gate the arm off until wired), commit compiling checkpoints.

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

## Next work — the 2a–2g INTEGRATION MAP (agent-verified seams, 2026-07-19)

The ME surface is complete (2d.7 landed). Everything below is the RD-loop integration — none of it
is independently byte-testable without the loop (unlike the kernels). Center of gravity: **2f**, a
multi-file port mirroring the intrabc leaf arm (KB-15, the direct template — itself still in
progress). **Target the ZERO-MV skip P first** (see the SESSION 2026-07-19 blocker above). Exact
seams, verified this session by 5 parallel source surveys:

### 2e — encoder inter MC (the MC crate `aom-inter` is READY)
`aom_inter::build_inter_predictor` (`crates/aom-inter/src/lib.rs:448`) — lowbd, supports
REGULAR/SMOOTH/SHARP per-direction filters. Signature:
`build_inter_predictor(ref_plane, ref_stride, ref_w, ref_h, dst, dst_off, dst_stride, blk_x, blk_y,
w, h, mv_row, mv_col, ss_x, ss_y, filter_x, filter_y)`. The DECODER's per-plane call pattern
(`crates/aom-decode/src/lib.rs`: luma `:2758`, chroma whole-block `:2925`, MV per-plane clamp via
`clamp_mv_to_umv_border_plane`) is the template. Add `aom-inter = { path = "../aom-inter" }` to
`aom-encode/Cargo.toml`; write an `enc_build_inter_predictor` helper looping planes with chroma
subsampling. **For the ZERO-MV target this is trivial** (mv=(0,0) → block copy, no interp), so 2e
is deferrable to the translational chunk. Highbd (bd10/12) NOT supported — a later chunk.

### 2c — inter ref-mv (the fn is byte-exact; wire the encode-side grid)
`aom_entropy::dv_ref::find_inter_mv_refs(rf0, mi_row, mi_col, bsize, own_partition, up_avail,
left_avail, tile, frame_mi_rows, frame_mi_cols, mib_size, allow_ref_frame_mvs, global_mv, gm_wmtype,
sign_bias:[i8;8], allow_high_precision_mv, is_integer_mv, grid: impl DvGrid) -> InterMvRefs`
(`crates/aom-entropy/src/dv_ref.rs:989`). Returns `InterMvRefs { mode_context, ref_mv_count,
stack:[(i32,i32);8], weight:[u32;8], nearest, near, global_mv }` — `mode_context` → inter-mode cost;
`nearest`/`near`/`global_mv`/`stack` → the NEAREST/NEAR/GLOBAL/NEW predictor MVs; `weight`+
`ref_mv_count` → DRL cost. Consumed via the `DvGrid` trait (a closure `|ro,co| -> DvNbr` works).
The encoder maintains a parallel `DvNbr` grid (like the decoder's `mi_dv`, `lib.rs:1320`), stamping
each decided block's `DvNbr { bsize, ref_frame0, ref_frame1, use_intrabc, mode, mv0_row/col,
mv1_row/col }` (`dv_ref.rs:64`) — **the `DvCell`/`DvNbr` slots already exist** (intrabc hardcodes
INTRA_FRAME/NONE; inter fills real ref_frame+mv). For zero-MV, NEAREST/NEAR/GLOBAL all resolve to
`(0,0)`; still needs the `mode_context` for the mode cost. Decoder call template:
`aom-decode/src/lib.rs:2419`.

### 2f — `handle_inter_mode` RD (THE integration center; mirror the intrabc arm)
**Integration point: `rd_pick.rs:422` (step 6 of `rd_pick_intra_mode_sb`)** — exactly where
`rd_pick_intrabc_mode_sb` already competes an intrabc winner against the assembled intra RD (take the
min). An inter mode-RD slots in as a sibling arm. The intrabc scaffold threads a single "reference"
(current frame) + DV=mv + skip through every seam inter needs — mirror it:
- **`leaf_pick_sb_modes` (`partition_pick.rs:687`)**: add an `InterLeafArgs` builder (mirror the
  intrabc block at `partition_pick.rs:1259-1336`) fed from a new `PickFrameCfg::inter`
  (`PickFrameCfg` at `partition_pick.rs:506`, `intrabc:` field at `:595` is the sibling template).
- **`LeafWinner` (`encode_sb.rs:167`)**: add inter fields beside the existing intrabc DV fields
  (`use_intrabc`/`dv_row/col`/`dv_ref_row/col` at `:222-229`): `is_inter`, `ref_frame:[i8;2]`,
  `inter_mode`, `mv:[MV;2]`, `interp_filter`, `ref_mv_idx`/`drl_idx`, and the var-tx
  `inter_tx_size[16]` plan (replaces the uniform `tx_size` for inter). rate/dist live in
  `raw_rdstats`.
- **`ModeGrid` (`partition_pick.rs:317`)**: `DvCell::to_nbr` (`intrabc_search.rs:1118`) hardcodes
  INTRA_FRAME/NONE — extend the stamped cell to carry real ref_frame+mv+mode. 25 `grid.stamp` sites
  (agent-listed; the bulk are in `stamp_grid_from_tree` `partition_pick.rs:3371`).
- **`encode_b_intra_dry` (`encode_sb.rs:521`)**: add an inter recon arm modeled on the intrabc arm
  (`:555-627`) — predict from the REFERENCE frame via `enc_build_inter_predictor` (2e), then either
  the SKIP path (reset coeff entropy ctx to skip, like intrabc `:572-573` — **this is all the
  ZERO-MV target needs**) OR the var-tx coeff arm (blocked, see SESSION blocker).
- **`pack_leaf` (`pack.rs:377`)**: intrabc already writes `use_intrabc` + DV diff and gates the
  tx/coeff syntax (`:499`); pack notes at `:487-497` already contemplate the inter tx-size write.
  Add: `write_is_inter`, `write_ref_frames`, `write_inter_mode`, `write_drl_mode`, skip, and (for
  NEWMV) the MV coder — **all byte-exact in `aom-entropy` partition module already**.
- **Inter mode/ref/skip COST tables**: `derive_real_costs` (`real_costs.rs`) already sources
  `inter_ext_tx` (2d, `44bc51c`); add the inter_mode/newmv/zeromv/refmv/drl/single_ref/intra_inter/
  comp_mode default CDFs → costs (several already in `default_cdfs.rs`). `find_inter_mv_refs.mode_
  context` feeds these.
- **var-tx** (nonzero-residual blocks only): `var_tx::pick_recursive_tx_size_type_yrd(env:&VarTxEnv,
  ref_best_rd) -> VarTxResult` (`var_tx.rs:1060`). `VarTxEnv`/`VarTxResult` fully field-mapped;
  construction template = `crates/aom-encode/tests/var_tx_recursion_diff.rs:477`. **BLOCKED on the 3
  NN prunes** (KB-15 §REMAINING items 1-3) before coeff blocks byte-match. Intra tx-search analog
  (mirror structure): `tx_search.rs:2555`.

### 2a — encode ref management + inter frame-header VALUES (writer is byte-exact)
The WRITER is done: `write_frame_header_obu` (`crates/aom-entropy/src/header.rs:1469`), inter arm
`:1486-1502`; INTER anchor test `header_diff.rs:1823` (expected `[0x30,0x3F,0xC0,0x00,0x00,0x02,0x40,
0x00,0x00]`). 2a = derive the VALUES into `FrameHeaderObu` (`header.rs:1411`) +
`FrameHeaderPrefix` (`:1101`) + `InterRefSignaling` (`:1297`). The KEY path BOOTSTRAPS values by
parsing a real header (`aom-bench/src/lib.rs:997`, `read_uncompressed_header` at `:1059`); the P
path must DERIVE them. §3 low-delay P (frame 1 → frame 0) values (agent-verified vs C):
`frame_type=1`, `order_hint=1`, `error_resilient_mode=0`, `primary_ref_frame=7 (NONE)`,
`frame_refs_short_signaling=0`, `ref_map_idx[7]` all → frame-0's slot, `refresh_frame_flags` = a
free slot (`get_refresh_frame_flags`, encode_strategy.c:655), `allow_high_precision_mv=1`,
`interp_filter=SHARP`/SWITCHABLE, `switchable_motion_mode=0`, `allow_ref_frame_mvs` gated.
C derivation: `choose_primary_ref_frame` (encode_strategy.c:168), `get_ref_frame_flags`
(encoder.h:4331). **Also needs a `RefFrame` on the encode side** (border-extended recon Y/U/V of
frame 0 + order_hint; decoder's `RefFrame` at `aom-decode/src/lib.rs:652` is the shape) + the
2-frame low-delay loop. NOTE: the recon-dependent header tail (LF levels, cdef) needs 2f's recon —
only the inter-specific fields above are derivable standalone (test them vs the parsed real frame-1
header).

### 2g — the decode-both gate
Add `port_encode_inter` to `MultiFrameEncodeCell` (`aom-bench/src/lib.rs:1945`): frame 0 KEY via
`frame0_cell().port_encode(bootstrap)` (existing, byte-exact), frame 1 P via the new 2f path,
concatenate. Then `inter_localize::decode_both(port_stream, cell.c_encode_inter(false, false))`
== 0 divergence at the ZERO-MV 4:2:0 cpu-0 config (`MultiFrameEncodeCell::translational(base,0,0)`).

### Suggested order (smallest-demoable-first, toward the ZERO-MV gate)
1. **2a ref buffer + inter frame-header VALUES** — testable standalone vs the parsed real frame-1
   header (inter fields). Net-new, isolated.
2. **2f inter RD arm, SKIP-only, GLOBALMV/NEARESTMV(0,0)** — the minimal 2f: no motion search, no
   var-tx, no MV coding; just is_inter/ref/mode/skip RD + pack, competing inter-skip vs intra at
   `rd_pick.rs:422`. This is the bulk of the work but bounded (no coeffs).
3. **2g decode-both** at zero-MV → close the SKELETON.
4. Then: 2e MC + NEWMV (translational-skip target), then KB-15's 3 var-tx prunes → coeff blocks.

Deferred ME follow-ups (speed≥1 / not needed for the speed-0 SIMPLE gate): the inter exhaustive mesh
(`mv_sf->mesh_patterns`), the full-pel `cost_list` (`calc_int_cost_list`), `second_best_mv`.

## Coordination

Work off origin/main; own `aom-encode` (ME/MC/RD) + `aom-bench` (harness) + `aom-sys-ref`
(me_shim). Concurrent agents touch `aom-decode`/`aom-inter`/`aom-entropy`(read) + `aom-encode`
(var-tx). Rebase-additive. Author `aom-rs <lilith@imazen.io>`, trailer `Co-Authored-By: Claude
Opus 4.8`. Push `HEAD:main`, verify `git merge-base --is-ancestor HEAD origin/main`. Symlink
`reference/libaom` + `conformance/data` from `/root/aom-rs/`.
