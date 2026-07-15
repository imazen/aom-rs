# aom-rs — project instructions & durable bug log

Pure-Rust, **bit-exact** reimplementation of libaom ≥ v3.14.1 as a drop-in replacement.
Validated behind differential harnesses against the REAL exported C functions (priority of
evidence: real exported C fn > synthetic-facade-over-real-fn > verbatim transcription —
transcribed oracles can carry shared bugs).

**Module-progress source of truth:** `STATUS.md` (updated per landing by the track agents).
**This file** holds project-level coordination rules + the durable **Known Bugs** log.

## Gates (definition of done)

- **Gate 1 — Decoder:** bit-identical to C across the AV1 conformance corpus (intra scope
  wired in CI: `xtask/conformance.py --fetch --scope intra`; gate = byte-identity + golden MD5).
- **Gate 2 — Encoder:** bitstream bit-identical for every `--cpu-used 0..9`.
- **Gate 3 — Performance:** ≤ 1.20× C.
- **Gate 4 — Coverage checklist** (+ a zenavif integration gate).

Primary configuration: ALLINTRA (usage=2), speed-0 KEY frame. **Single-frame (KEY-frame)
work must reach byte-exactness across BOTH tracks before inter-frame ("the rest") starts.**

## Known Bugs

Record real bugs here immediately with file:line refs (survives context loss). Do NOT close
an entry by relaxing/excluding a test — only by a landed fix verified on `origin/main`.

### KB-1 — Decoder: recon divergence at base_qindex ≥ 249 (quantizer-62/-63) — REAL CORRUPTION, CI-quarantined
- **Symptom:** decoded RECON diverges from the C oracle at `base_qindex >= 249` — the
  `quantizer-62` / `quantizer-63` conformance vectors. Reproduces at **bd8 AND bd10, luma AND
  chroma**. Divergence is an edge-local ±1 prediction cascade.
- **Root cause (CONFIRMED via isolated C-decoder instrumentation):** NOT an entropy/coeff-value
  bug. The first 311 txb records dump byte-identical (plane, tx, eob, dc_sign_ctx, txb_skip_ctx,
  levels ALL match) — the per-txb entropy decoder + context maintenance are FAITHFUL. The bug is
  the **txb ITERATION ORDER for coding blocks >64×64**: C (`decodeframe.c:929-962`,
  `decode_token_recon_block` intra path) chunks each block into BLOCK_64X64 units and within each
  chunk iterates planes→txbs → **L,U,V interleaved per 64-unit**; the port iterates each plane
  across the WHOLE block (all luma txbs, then all chroma) in `aom-decode/src/lib.rs` (~2235 luma
  loop + separate chroma loop). Identical for ≤64×64 blocks; for 128-sized blocks it desyncs the
  arithmetic decoder and everything cascades (the "edge-local ±1" symptom). Only q62/q63 pick
  partitions >64×64 (flat high-q blocks) → exact q61→q62 threshold. **Fix:** wrap luma+chroma
  reconstruction in the outer 64×64-chunk loop, plane-interleaved, matching C.
  (Earlier "entropy coefficient-decode path" localization was one layer too low.)
- **Fix #1 (VERIFIED, awaiting workspace-compile to land):** the reorder is implemented in
  `aom-decode/src/lib.rs` and proven — b10-q63 now byte-matches C and the port's 328 KEY-frame
  txb reads are byte-identical (up from the record-311 desync). The reorder is correct.
- **Bug #2 = CDEF per-unit strength stamping for >64 blocks (ROOT CAUSE CORRECTED — NOT intra-pred).**
  Exposed by fix #1; b8-q62 / b8-q63 / b10-q62 failed edge-local ±1 (b10-q63 clean). Intra-pred was
  DISPROVEN: the port's predict params for the failing 2nd 64×64 unit match C exactly (DC_PRED,
  n_top=64, n_left=32) and the DC math + left-column extension match C's `build_intra_predictors`
  line-for-line — pred+residual reconstruct the unit correctly. The scattered ±1 across a whole
  64×64 unit is CDEF's signature. C reads the CDEF strength once per 64×64 unit and stores it on the
  block's SHARED MB_MODE_INFO (`decodemv.c` read_cdef, stamped at the unit top-left mi); the frame
  walk reads it back per 64×64 unit top-left mi (`cdef.c:304`). A >64 block shares ONE mbmi across
  all its mi cells, so every covered 64×64 unit reads the same strength. The port
  (`aom-decode/src/frame.rs:1212`) stamped only the block's TOP-LEFT unit → other covered units
  stayed at −1 (CDEF skipped); for the 128-wide mi64,0 the 2nd unit (mi64,16) kept −1 so CDEF ran
  in C but not the port → the ±1. **Fix #2:** stamp `b.info.cdef_strength` on ALL 64×64 units the
  block covers (in-frame h×w extent); sub-64 blocks cover one unit, unchanged. Both bugs are
  >64-only, which is why exactly q62/q63 fail (only very high qindex picks >64 partitions).
- **Fix #1 + #2 VERIFIED GREEN (landing in one commit):** full conformance gate 269 in-scope frames,
  0 failures, WITH q60–q63 present; all four targets (b8/b10 × quantizer-62/63) byte-exact + golden
  MD5, plus 60/61 and everything else (allintra/size/intrabc/cdfupdate...), no ≤64 regression. The
  landing commit reverts the ci.yml q62/q63 rm, adds an explicit q62/q63 × bd8/bd10 regression test,
  and deletes the throwaway scratch. #21 closes only after: on origin, CI green WITH q62/63 restored,
  `merge-base --is-ancestor` confirmed.
- **Encoder cross-check (low priority):** the encoder pack must write txbs in the SAME
  64×64-chunk plane-interleaved order for >64 blocks. The encoder already byte-matches
  `diag+vbars16 256×256 cq63` (strong-LF gate 5/5), which is empirical evidence its order is
  correct — but confirm pack.rs's >64-block txb order once the decode-order fix lands.
- **CI status (TEMPORARY quarantine):** `.github/workflows/ci.yml:63-64` `rm`s the q62/q63
  vectors after fetch so Gate-1 goes green on the rest. This is a **must-fix corruption bug**
  under the zero-tolerance rule (wrong pixels are a shipping bug, never a known limitation),
  NOT an accepted limitation. The `rm` MUST be reverted in the same PR that lands the fix, and
  the specific q62/q63 vector(s) added as an explicit strong byte-identity case.
- **Tracking:** task **#21** (HIGH). Fix unblock: authorized throwaway reference-*decoder*
  instrumentation to dump the C coefficient + coeff-context/cdf state at the first diverging
  (position, plane, qindex), then revert + rebuild clean (never commit the instrument).
- **Range matters:** q62/q63 is the aggressive end of the quantizer range — exactly the
  web-compression regime this port targets.

### KB-2 — Encoder: `diag+vbars16 256x256 cq62` strong cell — FIXED ✅ (per-block intra edge filter type)
- **FIXED 2026-07-15.** Root cause: the port **never re-derived the intra edge filter type
  (`get_intra_edge_filter_type`, reconintra.c:974) per block** — it carried a frozen SB-level
  `filter_type` (always 0) down into every leaf's `TxfmYrdEnv`/`UvRdEnv`. C re-derives it per
  block from the live mode-info grid: `1` iff the above **or** left neighbour is a SMOOTH mode
  (SMOOTH_PRED=9 / SMOOTH_V_PRED=10 / SMOOTH_H_PRED=11). For the diverging cell, SB(32,32)'s
  VERT_4 strip-1 (16×64 @ mi(32,36)) has a **SMOOTH left neighbour** (strip-0, mode 9), so C
  computes `filter_type=1` while the port used `0`. That flips the intra-edge-filter strength for
  **angled** directional predictions (adj≠0; pure-vertical adj=0 skips the edge filter, which is
  why adj=0 matched exactly and only angled deltas diverged). The port's worse angled prediction
  raised V_PRED adj=−1's **model RD** to 25930 vs C's 24704; the `prune_intra_y_mode`
  `THRESH_BEST=1.5×best_model_rd` (=1.5×17236=25854) then **over-pruned adj=−1** in the port
  (25930>25854, margin 76) where C keeps it (24704<25854). C fully evaluates adj=−1, the ALLINTRA
  variance factor reorders it ahead of adj=0, and C picks adj=−1 → strip winner differs → HORZ_A
  vs VERT_4 → byte divergence. **Fix:** recompute `filter_type` per block from `above_mode`/
  `left_mode` (already read from the grid for the mode-cost context) in `partition_pick.rs`'s
  leaf search, mirroring `get_intra_edge_filter_type`; the `CPick` C-recursion reference in
  `partition_pick_diff.rs` got the identical recompute so the differential stays faithful.
- **Verified:** the cq62 cell now achieves TRUE END-TO-END BYTE MATCH vs real aomenc and is an
  **asserted** case in `encoder_gate_e2e_rich_content_strong_lf` (6/6); full `aom-encode` suite
  green; the port's angled prediction matches C pixel-for-pixel (per-tx-block SATDs identical).
- **Chroma follow-up (#26) — FIXED ✅ 2026-07-15.** The **chroma** `filter_type` (UvRdEnv) was the
  same frozen-at-0 bug on the UV plane: C's `get_intra_edge_filter_type(xd, plane=1)` is `1` iff an
  available above/left chroma neighbour's `uv_mode` is SMOOTH (UV_SMOOTH_PRED=9 / UV_SMOOTH_V=10 /
  UV_SMOOTH_H=11). Fix mirrors the KB-2 luma recompute on chroma: `ModeGrid` now carries a parallel
  `uv_modes` grid (`partition_pick.rs`, stamped alongside luma at every `stamp`/`stamp_grid_from_tree`
  site); `leaf_pick_sb_modes` recomputes the per-block chroma edge `filter_type` from the chroma
  neighbours (chroma-reference mi derivation, av1_common_int.h:1400-1416: `base=(mi_row-(mi_row&ss_y),
  mi_col-(mi_col&ss_x))`, above=`base+(-1,+ss_x)`, left=`base+(+ss_y,-1)`) and feeds it to BOTH the UV
  RD search AND — via the new `LeafWinner::uv_edge_filter_type` — the pack re-encode
  (`encode_b_intra_dry`, encode_sb.rs), which produces the coded chroma bytes. The `CPick`
  C-recursion reference in `partition_pick_diff.rs` got the identical recompute + a parallel `uv_grid`
  (randomized UV neighbours now exercise it as a differential witness). **Verified:** new
  `encoder_gate_444_bd8_chroma_edge_filter_witness` (encoder_gate_chroma_ss_e2e.rs) byte-matches real
  aomenc on all 4 cells WITH the fix and DIVERGES on the 128×128 cq12/cq32 cells with it reverted
  (proven fails-before/matches-after); `partition_pick_diff` passes with randomized smooth UV
  neighbours; full `aom-encode` suite green. Commit: partition_pick.rs + encode_sb.rs +
  partition_pick_diff.rs + encode_sb_diff.rs + the witness.
- **Historical isolation trail (how it was root-caused) below:**
- **Re-verified 2026-07-15 (still diverges), with much sharper isolation:**
  - Facts: qindex **249**, `screen_content=true` (auto-detected — the ONLY screen-content cell in
    the whole encoder suite), port tile **95 bytes vs real 100** (port codes FEWER symbols), port
    derives LF luma **[0,17]** vs real **[1,17]** (a DOWNSTREAM recon symptom, not the cause), first
    payload mismatch at byte 3 (= the header LF-level byte). First **TILE**-byte divergence is at
    **tile-byte 60 of 100** → the first ~60% of the tile is byte-identical, so the divergence is in a
    **MID-FRAME SB, NOT SB(0,0)** (unlike KB-3).
  - **RULED OUT — palette flag** (definitively): the port's RD `try_palette =
    allow_palette(allow_screen_content_tools, bsize)` (partition_pick.rs:589, no `enable_palette`
    gate) is EMPIRICALLY byte-exact — `encoder_gate_e2e_ab_attempt` is the exact
    `enable_palette=0`(standard shim) + `screen_content=1` config and byte-matches WITH it; forcing
    `try_palette=false` REGRESSED that gate. So real includes the palette-Y no-palette flag cost for
    screen-content frames regardless of `--enable-palette=0`, and the port matches. Write side
    (pack.rs:274, `allow_palette` only) matches C (bitstream.c:1042). Palette is fully correct.
  - **RULED OUT — all other screen-content RD effects** (parallel-agent survey of the sibling C,
    verified against build config): at speed-0 / full non-realtime build / ALLINTRA / KEY / qidx249 /
    <720p, there is **zero** screen-content dependence in rdmult (rd.c), quantizer (av1_quantize.c),
    coeff trellis (encodemb.c/txb_rdopt.c), tx-set context, angle-delta / filter-intra / smooth, or
    the partition search — beyond palette (handled) and the header intrabc-present bit (handled: AB
    gate proves the port's header writer emits it). The one latent tx path, `get_default_tx_type`
    forcing DCT_DCT under screen content (blockd.h:1175), is **dormant** because
    `use_intra_default_tx_only=0` in the non-realtime reference build (verified `CONFIG_REALTIME_ONLY
    0` + av1_cx_iface.c:374 default 0). RANK-3 `exhaustive_searches_thresh` differs at speed-0 but is
    inert (no motion search in all-intra). RC is bypassed (fixed AOM_Q, per-block qindex stays 249).
  - **CONCLUSION:** a plain **speed-0 coeff/partition/mode near-tie**, NOT screen-content-specific.
    Same content+generator as the cq**63** cell that byte-matches (strong_lf gate 5/5); cq62 → qidx
    249 tips a near-tie in a later SB. Class-identical to KB-1's "only very-high-qindex flips it".
  - **RD-DUMP DONE (2026-07-15) — root-caused to a single 16×64 leaf's tx/coeff evaluation.**
    Method: re-tailored sibling harness (`/root/libaom-enc-instrument/rd_harness.c`) for
    `diag+vbars16 256×256 cq62 cpu0` and VALIDATED its output == real (117-byte stream, frame OBU
    `32 69` payload = 5 hdr `44 f9 00 51 14` + 100 tile `ff 3b 14 51…`). Then per-SB partition dump
    (port PSB vs sibling C CSB): **15/16 SBs match; SB (mi=32,32) diverges — C picks PARTITION_HORZ_A
    (4), port picks PARTITION_VERT_4 (9).** Per-candidate RD at (32,32): port HORZ_A rate=33741
    dist=8751216 **rdcost=1393344729 == C's HORZ_A EXACTLY**; port VERT_4 rate=23037 dist=8757376
    **rdcost=1307466663 wins**. C's VERT_4 is INVALID: C's 4-way prune allows both HORZ4/VERT4
    (`allowed=[1,1]`, `prune_ext_partition_types_search_level=1` so the level-2 partitioning gate at
    partition_search.c:4202 does NOT fire — not a pruning diff), but C's VERT_4 sub-block search
    **bails at strip 2** (`rd_try_subblock` returns 0: strip-2's own 16×64 mode RD exceeds the
    remaining budget best−cum). **Per-strip VERT_4 at (32,32) (both mono, subsize=BLOCK_16X64=20):
    strip0 (c=32) mode=9 cum_rate=7557 cum_dist=3946048 — MATCHES C exactly; strip1 (c=36) SAME
    mode=1 (V_PRED) in both, but port Δrate=5614/Δdist=933472 vs C Δrate=9980/Δdist=1568992 — port
    UNDER-COMPUTES both.**
  - **EXACT ROOT CAUSE — angle_delta divergence on the strip-1 16×64 V_PRED leaf.** Both pick
    identical `tx_size=TX_16X64 (17)`, `skip=0`, `tt0=DCT_DCT`; the ONLY difference is the intra
    **angle_delta**: **C picks V_PRED `angle_delta=-1`, the port picks V_PRED `angle_delta=0`.** The
    port's adj=0 (rate 5614 / dist 933472) is strictly cheaper on BOTH axes than C's adj=-1 (rate
    9980 / dist 1568992) — so C's OWN adj=0 evaluation must be *worse* than the port's adj=0 (else C
    would pick 0). Both search the full delta range (`use_angle_delta` matches C exactly:
    `bsize>=BLOCK_8X8`, and 16×64=20 qualifies; port `enable_angle_delta=true` at speed 0). ⇒ the
    port's **directional-intra prediction and/or angle-delta RD for this 16×64 (1:4-aspect) leaf is
    wrong** — its adj=0 (or the delta search) is under-costed, so adj=0 wins in the port where adj=-1
    wins in C. (NOT partition pruning, NOT palette, NOT screen-content, NOT tx-size/type/skip, NOT
    #25's speed-1 bugs — this is speed-0.) strip0 (also 16×64, mode=9=D67_PRED-ish non-vertical)
    matching rules out a blanket 16×64 bug — it's specific to V_PRED angle_delta on this leaf.
  - **RESOLVED (see the FIXED block at the top of this entry).** The per-delta dump above was
    slightly mis-framed: adj=0 was **not** under-costed — it matched C exactly. The real mechanism
    is that the port never even *evaluated* adj=−1's full RD: it **model-pruned** adj=−1 at
    `prune_intra_y_mode` because its **model** RD (25930) was inflated by the wrong (0 instead of 1)
    intra edge filter type on the angled prediction, tipping it over `1.5×best_model_rd` (25854).
    The "directional-intra predictor edge/neighbour" guess was on target — it was the per-block
    `get_intra_edge_filter_type` recompute the port was missing. All temp instrumentation and the
    sibling `/root/libaom-enc-instrument` have been removed.

### KB-3 — Encoder: `vgrad 256x256 cq32` cpu-used=1 cell — FIXED (missing speed-1 `use_square_partition_only_threshold` rect-kill)
- **FIXED** (commit pending on origin): the cell now byte-matches; promoted to an asserted winner
  in `encoder_gate_speed1_textured_allintra` (14/14 cpu-used=1 content cells). Root-caused via
  **isolated sibling-libaom encoder instrumentation** (`/root/libaom-enc-instrument`, a throwaway
  copy — never the shared `reference/libaom`) dumping C's per-candidate RD at SB(0,0) 64×64 for
  the exact vgrad-256-cq32 encode. Findings: C's NONE and SPLIT RD matched the port **exactly**
  (NONE rate 36745 / dist 19456 / rdcost 7427690, rdmult 68796); C **never evaluated** the
  rectangular partitions, but the port did, and the port's HORZ (rdcost 7058801) beat NONE → port
  wrongly picked `PARTITION_HORZ`. C disables rect via the "square-partition-only" rect kill
  (`partition_search.c:5749`): `if (bsize > use_square_partition_only_threshold) {
  partition_rect_allowed[HORZ] &= !has_rows; [VERT] &= !has_cols; }`. That threshold is a
  framesize-DEPENDENT ALLINTRA speed feature: sub-480p it is `BLOCK_64X64` at speed 0 (so
  `bsize > 64X64` never holds in a ≤64 SB — why speed-0 never needed it) but drops to
  `BLOCK_32X32` at speed ≥ 1, killing rect on the 64X64 SB. **Fix:** wired the rect-kill into
  `rd_pick_partition_real` (`use_square_partition_only_threshold_allintra`, framesize+speed
  dependent), placed after `partition_rect_allowed` init and before the CNN prune (matching C's
  order). Speed-0 unaffected (threshold `BLOCK_64X64` → no-op); full `cargo test -p aom-encode`
  = 89 passed, 0 failed. NOT a learned-model prune (the CNN/prune_2d/etc. elimination below stands).
- **KB-2 is a SEPARATE root** (do NOT conflate): KB-2's cell runs at **cpu-used=0**, where this
  fix is a no-op (threshold `BLOCK_64X64`). KB-2 needs its own speed-0 root-cause pass.

<details><summary>Original isolation notes (superseded by the fix above)</summary>

Was: `vgrad 256×256 cq32` (base_qindex 128) diverged at byte 5, never re-converging.
- **Symptom:** in `encoder_gate_speed1_textured_allintra`, the `vgrad 256×256 cq32`
  (base_qindex 128) cell does not e2e byte-match aomenc. Diverges at **byte 5** (first
  tile-data byte) and **never re-converges** (`last_common_idx = 4` = last header byte) — an
  early partition/mode cascade at SB(0,0). Excluded (documented) in the winners list of that
  gate; the sibling cells (256×256 cq48, 128×128 cq32/cq48) all byte-match.
- **Isolation COMPLETE — NOT an unported learned-model prune.** The originally-suspected
  `intra_cnn_based_part_prune_level` 0→2 (intra CNN partition prune) is now **fully ported +
  wired** into `rd_pick_partition_real` (commit `a600394`) and its four flags are **bit-exact
  vs C** (`cnn_partition_decision_diff`). For this cell the CNN fires and sets
  `square_split_disabled=true` at every 64×64 SB root — **identically to C** — so it constrains
  port and C the same way and cannot cause a divergence. **Empirically confirmed:** wiring the
  CNN in left byte-5 (157 vs 8) byte-identical. Eliminated candidates (with evidence):
  `prune_2d_txfm_mode` PRUNE_2 (intra path needs `prune_tx_type_est_rd`, which is speed≥4;
  `prune_tx_2D` is `is_inter`-only); `model_based_prune_tx_search_level`,
  `av1_ml_predict_breakout`, `av1_ml_early_term_after_split`, `av1_ml_prune_rect_partition`,
  `simple_motion_search_*` (all `!frame_is_intra_only`); `ml_predict_var_partitioning` (nonrd).
- **Root cause (localized):** a **partition-search RD near-tie** (KB-2 class). The port picks
  `PARTITION_HORZ` for SB(0,0) (two 64×32 DC / TX_64X32 blocks); C picks a different partition.
  A speed-1 RD-cost delta tips the NONE/HORZ/VERT comparison for this specific content+qindex.
- **Next step:** dump the port's per-candidate RD (NONE/HORZ/VERT) at the SB(0,0) 64×64 node vs
  the C reference. Needs an **encode-side RD-dump shim** — but `shim_encode_av1_kf` currently
  lives in the decoder-owned `dec_shim.c` and drives the opaque `aom_codec` API (no `cpi->sf`
  hook), so per-feature C-side toggling / RD dumps aren't reachable from the encoder track
  without a coordinated new shim entry point. Candidate speed-1 RD deltas to bisect once that
  exists: `perform_coeff_opt=2`, `tx_domain_dist_level/thres_level=1`, `adaptive_txb_search
  _level=2`, `top_intra_model_count_allowed=3`.
- **Two LATENT speed-1 bugs found while isolating (NOT this cell's cause — both leave these 8
  cells byte-identical, so no current test exercises them; documented for a future fix + new
  validation cells):**
  1. `part4_prune.rs:234` hardcodes `LEVEL_INDEX = 0`, but C's `ml_4_partition_search_level
     _index = min(speed,3)` (set 0/1/2/3 at `if(speed>=1/2/3)`, speed_features.c:210/237/271;
     default 0 at :2305). Index expr `(LEVEL*3+res_idx)*5+bsize_idx` uses LEVEL **directly**
     (no −1) — the port's `LEVEL_INDEX` == the level. Usage: `av1_ml_prune_4_partition`,
     partition_strategy.c:1507-1510. **CRITICAL caveat (verified 2026-07-15):** at level **3**
     (speed≥3) C flips `ml_model_index = (level<3) == 0` (partition_strategy.c:1359) → a
     **different NN model, no threshold table** (`:1472-1497`, scores vs `max_score−{500,500,200}`).
     So the port's table path is correct ONLY for speeds 0/1/2 (LEVEL 0/1/2). Fix = pass
     `level=min(speed,3)` from `cfg.speed` into `predict_4partition_prune` (caller
     partition_pick.rs:2173) and use it as the table row **only when level<3**; speed≥3 needs the
     alternate (old-NN, tableless) branch = a #10 item, NOT #25. Feeding LEVEL=3 into the table
     would be wrong (that path never runs in C).
  2. `tx_search.rs:1305` `get_search_init_depth_intra_speed0` hardcodes the speed-0
     `intra_tx_size_search_init_depth_rect = 0`, but C uses 1 at speed≥1 (speed_features.c:409);
     `_sqr = 1` for ALL speeds (unconditional at :367). So at speed≥1 BOTH rect and sqr return 1.
     `get_search_init_depth` (tx_search.c:363-383) returns `_rect` when w≠h, `_sqr` when w==h.
     Fix = thread `speed` into `choose_tx_size_type_from_rd_intra` (caller of the init-depth fn,
     tx_search.rs:1356; `TxfmYrdEnv` has no `speed` field yet — add it or pass a param) and return
     `rect = (speed>=1) as i32`, `sqr = 1`.
  Both preserve speed-0 exactly (min(0,3)=0; rect=0 at speed 0). Needs new speed-1 RECT-partition
  test cells to validate — the current speed-1 gates pass WITH the bugs (they don't reach a
  divergent 4-way-prune / rect-tx decision), so exercising cells must be discovered (a speed-1
  e2e harness exists: `encoder_gate_speed1_textured_allintra`).

### KB-4 — Encoder: bd10/bd12 RD-decision divergence at high bit depth with large coefficients — REAL, tracked (task #31)
- **Symptom:** at bit depth 10/12 the port's encoded bitstream diverges from real aomenc when coefficients are large. Triggered by (a) 4:2:2/4:4:4 chroma at *moderate* content — all 4 bd10 non-420 cells in `encoder_gate_chroma_ss_e2e` diverge (444 64² byte2 len 1805 vs 1803; 444 128² byte394; 422 64² byte2; 422 128² byte3, all cq32), and (b) mono/4:2:0 with *aggressive* full-range high-frequency content at low qindex (cq12/32, e.g. mono 128² cq12: 7340 vs 7380). Tile data with DIFFERENT byte COUNTS ⇒ different mode/tx/partition RD winners, not a coding-of-fixed-decision bug.
- **Byte-exact regime (NOT affected):** bd8 all content (incl. 4:2:2/4:4:4, 18/18 in #30); bd10/12 mono+4:2:0 moderate content (`encoder_gate_bd10_diff`, CI green 20f1e70); representable-magnitude (≤255) content at bd8 AND bd10 (probe: identical ≤255 content → both match). Realistic photographic 10/12-bit content is smoother → smaller coeffs → likely byte-exact.
- **Root cause (localized BY ELIMINATION — two independent triangulations, #30 non-420-chroma + bd10-agent aggressive-HF):** NOT the block kernels — quantize (`xform_quant_diff`), coeff-coding (`encode_block_coeffs_diff`), trellis (`xform_quant_optimize_highbd_diff`, ad31529) ALL byte-exact at ±4095 bd12. NOT bd-plumbing (≤255 content matches at bd10). NOT int64 overflow (no frame-path panic; the >22-bit-coeff overflow at `optimize.rs:61` is only reachable by unrealistic random ±4095, which C also treats as UB). ⇒ the divergence is in the **RD-DECISION layer** (tx_search / intra_rd / partition_pick): large-coefficient RD-cost / distortion accumulation or compare differs from C at high bit depth (suspect a rounding / shift / bit-depth-scaling difference in an RD-cost computation).
- **Fix:** in the encoder's RD-decision files — align the high-bit-depth RD-cost/distortion scaling with C. Owned by the encoder after #25 lands (frees those files). Witness (COMMITTED 2026-07-15): `encoder_gate_bd10_non420_e2e_kb4_repro` in `encoder_gate_chroma_ss_e2e.rs` — the 4 bd10 non-420 cells (444/422 × 64²/128², cq32), a CI-green characterization asserting the divergence is still present; the bd10 aggressive-HF mono/420 corner stays in `kb4_bd10_rd_localize.rs`. Close ONLY by a landed fix that makes those cells byte-match (promote the repro to a `report_and_assert` byte-match gate) — never by weakening.

### KB-5 — Encoder: lossless (cq0 / qindex 0) KEY encode divergence — REAL, tracked (task #32)
- **Symptom:** lossless allintra KEY (cq0 → qindex 0, `coded_lossless`) diverges badly — mono 64² cq0: first-diff at byte 2 (within the 9-byte header region), port tile 5426B vs the ENTIRE real frame 4966B (port ~10% larger). A port tile larger than the whole real frame ⇒ the port is very likely NOT taking the lossless coding path.
- **Root cause (LOCALIZED + VERIFIED — chroma-ss read-only + coordinator grep — TWO bugs):** (1) HARNESS-SETUP (task #32, immediate/blocking): `run_case` does a single `read_uncompressed_header` with `cfg.coded_lossless=false`, skipping the two-pass lossless probe the parser contract requires (`aom-entropy/src/header.rs:2952` clones cfg + gates loopfilter/cdef/tx_mode reads on `cfg.coded_lossless`, a writer-mirror input by design). The real decoder does the two-pass (`aom-decode/src/frame.rs:455-487`: parse → `frame_coded_lossless(probe)` [:355 = base_qindex==0 && all 5 plane q-deltas 0] → re-parse with `cfg.coded_lossless/all_lossless=true`). Skipping it → `p.coded_lossless=false` → (a) header emits a phantom loopfilter block the real lossless header omits (byte-2 first-diff), (b) `env.lossless=false` → port runs its NON-lossless encoder at qindex0 (full DCT) → 5426B tile. The lossless SEARCH branch is correct + present (partition_pick.rs:577, tx_search.rs:1448 TX_4X4-only, pack.rs:296) — just never reached. (2) LATENT ENCODER PORT BUG (task #33, surfaces after the harness fix): NO forward Walsh-Hadamard transform in the encode path — `xform_quant` (aom-encode/src/lib.rs:172) unconditionally calls `av1_fwd_txfm2d`; libaom applies `av1_fwht4x4`/`av1_highbd_fwht4x4` for coded-lossless TX_4X4. VERIFIED: grep fwht/fwalsh across aom-encode+aom-transform = ZERO hits; the decoder HAS the inverse (`aom-transform/src/inv_txfm2d.rs:256 av1_highbd_iwht4x4_add`). So even with env.lossless=true, coeffs diverge.
- **Fix (TWO parts, BOTH required for green):** (1) HARNESS (#32, chroma-ss): mirror the decoder's two-pass in run_case — probe, compute is_lossless from probe.quant, re-parse with cfg.coded_lossless/all_lossless=true. Necessary but NOT sufficient. (2) ENCODER (#33): add forward WHT to aom-transform + route xform_quant (lib.rs:172) and the UV path (intra_uv_rd.rs:800) to it for coded-lossless TX_4X4. Witness (COMMITTED 2026-07-15): `encoder_gate_lossless_cq0_e2e_kb5_repro` in `encoder_gate_chroma_ss_e2e.rs` — mono + 4:2:0 cq0 cells, a CI-green characterization (the cq0 encode does NOT panic; it cleanly diverges — mono tile 5426B vs the real 4966B, matching the documented symptom). Close ONLY by a landed fix (promote to a byte-match gate) — never by weakening.

### KB-6 — Encoder: REAL-content RD divergence at bd8 4:2:0 (PRIMARY config) — REAL, tracked, FIRST ROOT FIXED (luma re-encode), continuing
- **FIX #1 LANDED 2026-07-15 (ca2826f) — luma re-encode intra edge filter.** The luma analogue of
  #26 (chroma). `encode_b_intra_dry` — the dry-run re-encode used by BOTH the search's inter-strip
  context propagation (`partition_pick.rs:1054/1338/1914`) AND the pack output (`pack.rs:317`) — froze
  the LUMA intra edge filter at the SB-level `env.filter_type` (always 0) instead of the per-block
  `get_intra_edge_filter_type` (reconintra.c:974). KB-2 fixed only the luma SEARCH RD (leaf y_env); the
  re-encode/stamp stayed at 0. So an angled luma leaf (angle_delta≠0) with a SMOOTH above/left neighbour
  re-encoded its prediction with edge filter 0 not 1 → wrong residual → per-txb eob flip in the coded
  bytes, AND a wrong propagated entropy context that shifted later leaves' RD. **Fix:** carry the
  per-block `luma_edge_filter_type` (already computed in the search, KB-2) on `LeafWinner` and feed it to
  `encode_b_intra_dry`'s y_env. The `CPick` differential reference had to mirror it or diverge on
  smooth-neighbour angled leaves: `CEncPlaneArgs` gained a `filter_type` field so the `COracle`
  propagation re-predicts (ref_hbd_predict_intra 9th arg) with the SAME per-block filter. Localized via
  `kb6_real_rd_localize.rs` (decode-both-streams): first divergent SB was leaf mi(12,12) bsize=BLOCK_4X16
  angled (y_mode=6, angle_delta_y=1), real eob=0 vs port eob=2, ±1 recon at (48,48). Verified: full
  aom-encode suite green; `partition_pick_diff` green with randomized SMOOTH neighbours.
- **REMAINING (continuing localization).** Map now **15/30 byte-exact** (expanded gate, commit 3d124a7):
  real size-64×64 byte-matches at cq5/12/20/48/63 (promoted to asserted gates); a SECOND near-tie
  remains at **size-64×64 cq32**. Photographic/film crops added (00-quantizer, 23-film — content
  diversity at low pixel count): quantizer-64² 4/6, film-64² 5/6 match; the multi-SB quantizer-128²
  crop mostly diverges. **Divergences cluster at cq5 (qindex 20, aggressive low-q) across ALL THREE
  crop contents — a strong content-independent low-q near-tie, the next localization target.**
- **DISTINCT SUB-GAP — partial-SB (frame dims not a multiple of 64px).** The **196×196** cells all
  diverge via a SEPARATE mechanism: the port's partition search does not model frame-EDGE (partial)
  superblocks (`rd_pick_4partition` docs: edge 4-way "simply not attempted"; edge rect "out of scope"),
  and `TileCtxState::zeroed` sizes the above-context to bare `mi_cols` not `aligned_mi_cols =
  ALIGN_POWER_OF_TWO(mi_cols, MAX_MIB_SIZE_LOG2)` (alloccommon.c:414) so a full-SB context read at the
  edge SB overflows. This is a real correctness gap for any non-64-aligned frame, tracked as part of
  KB-6 but is NOT the RD-near-tie class — fix separately (context arrays → aligned_mi_cols + edge-block
  partition search + ceil the SB loop).
- **MULTI-TILE encode is byte-exact** (commit f6e6319, `encoder_gate_multitile_e2e`): the port's own
  per-tile search+pack byte-matches real aomenc across 2×1/1×2/2×2 grids (4:4:4 128² × cq{12,32,63}).
- **DISCOVERED 2026-07-15 via the new real-image e2e gate** (`encoder_gate_real_image_e2e_kb6_repro`
  in `encoder_gate_chroma_ss_e2e.rs`): decode the first KEY frame of a small conformance vector
  (`av1-1-b8-01-size-64x64`, `av1-1-b8-01-size-196x196`; `01-size` is in CI's intra fetch scope) to
  genuine YUV via the C decode oracle, then run the port's full encode vs real aomenc byte-for-byte on
  those REAL pixels. **Every synthetic e2e gate is byte-exact, but genuine image content diverges
  across the whole quality range.** Map (bd8 4:2:0, cq5..63): the multi-SB **196×196 frame diverges at
  EVERY cq** (e.g. cq20 port tile 1457B vs real 1556B — port codes ~100 FEWER bytes); the 1-SB
  **64×64 diverges at cq5/12/32/48** and byte-matches only at the coincidental cq20/cq63. 2/12 cells
  byte-exact, 10 diverge. (Superseded by FIX #1 above: after the luma re-encode fix + the expanded
  photographic/film crop gate, the map is now 15/30 byte-exact.)
- **Signature = KB-2 class:** the port codes FEWER symbols than aomenc ⇒ it makes different (cheaper)
  partition/mode/tx RD decisions — a near-tie flip, exactly like KB-2 (`get_intra_edge_filter_type`)
  and KB-3 (speed-1 rect-kill), but now on the **PRIMARY bd8 4:2:0 speed-0 KEY** path and on REAL
  content. The hand-tuned synthetic patterns (diag/vbars/vgrad/tex_*) never exercised the diverging
  decision; real photographic/screen statistics do. **This means the "byte-exact regime: bd8 all
  content" note under KB-4 is TRUE ONLY for the synthetic gates — it is FALSE for real content.**
- **Root cause: FIRST ROOT (luma re-encode edge filter) LOCALIZED + FIXED — see FIX #1 above.** As
  predicted it was MULTIPLE KB-2-class near-ties: fixing the luma re-encode took real 64×64 from 2/6 to
  5/6. ≥1 more remains — the cq5 (low-q) near-tie and 64×64 cq32 — same class (a speed-0 RD near-tie an
  edge/neighbour/rounding difference tips), to be localized next.
- **Repro (COMMITTED, CI-green characterization):** `encoder_gate_real_image_e2e_kb6_repro` prints the
  full per-cell MATCH/MISMATCH map, asserts a byte-exact CONTROL (64×64 cq20 — harness-faithfulness +
  regression guard), and asserts the KB-6 divergence is still PRESENT (gates: when the port becomes
  byte-exact on real content the test FAILS → promote it to a full `report_and_assert` byte-match
  gate). Not a weakened test — the correct end state is full byte-identity on real content.
- **Next step (localization):** the **cq5 (qindex 20) low-q near-tie** is the highest-value target — it
  diverges across all three crop contents (quantizer-64²/128², film-64²), so it is content-independent
  and hits the aggressive-web low-q regime. Use `kb6_real_rd_localize.rs` (decode-both-streams: the port
  DECODER is bit-exact, so decode the real aomenc stream vs the port's own re-wrapped stream, diff
  per-SB partition/mode/tx to pin the first divergent block), find the flipped decision, fix it
  KB-2/KB-3-style. Then 64×64 cq32. Close ONLY by a landed fix that makes real content byte-match —
  never by weakening or by narrowing the corpus.
- **Priority note:** KB-6 hits the single most common real-world case (bd8 4:2:0 photographic content
  at web qindex), so it is arguably higher-impact than the bd10/bd12 (KB-4) and lossless (KB-5)
  corners. Sequencing is the coordinator's call.

## Encoder single-frame primary envelope (VERIFIED against reference/libaom)

Primary config = ALLINTRA (usage=2), speed-0 KEY frame. libaom's own allintra tuning
(`av1/av1_cx_iface.c:3065`) sets these **defaults** — so matching them, NOT the base defaults,
is what "single-frame exact" means:

- **CDEF: OFF** by default in allintra ("CDEF has been found to blur images, so it's disabled
  in all-intra mode"). Only `--enable-cdef` turns it on.
- **Loop-restoration: OFF** by default in allintra.
- **QM: OFF** by default in allintra. CORRECTED 2026-07-15 (the prior "QM: ON" claim was WRONG —
  it conflated the qm_min/max override with `enable_qm`). The allintra override at
  `av1_cx_iface.c:3065` sets `qm_min=4`/`qm_max=10` but does NOT assign `enable_qm`, which stays
  at its base default `0` (`:290/447`); `using_qm = enable_qm` (`:1310`). qm_min/max are INERT
  unless QM is turned on by `--enable-qm` (`:2076`) or `tune=IQ`/`SSIMULACRA2` (`:1946`).
  Empirical proof: the passing `encoder_gate_e2e_*` gates byte-match the port with `qm=None` —
  impossible if the reference allintra encodes were QM-on.
- screen_detection_mode = ANTIALIASING_AWARE.

**What the encoder track has byte-matched (`encoder_gate_e2e_*`):** own-search partition / mode /
tx / coefficients + LF-level derivation, in a **CDEF-off + restoration-off + QM-off** reference
encode (`shim encode_av1_kf`, cdef/restoration/qm passed as explicit params). This envelope
MATCHES the allintra defaults for CDEF, restoration, AND QM (all off). The frame HEADER is still
bootstrapped from the real parse (qindex, tile info, cdf-update, ...) — only LF-level is
port-derived.

**Remaining for single-frame-PRIMARY exactness (blocks "all single frame exactly"):**
- **KB-2 (#22) cq62 speed-0 — FIXED ✅ (74fb582)**: per-block `get_intra_edge_filter_type`
  recompute in `partition_pick.rs` (a SMOOTH neighbour was not raising the angled-prediction edge
  filter → model-RD over-pruned V_PRED adj=−1 → flipped SB(32,32) partition). cq62 byte-matches +
  asserted in `encoder_gate_e2e_rich_content_strong_lf`. See the KB-2 FIXED block above.
- **#25 two latent speed-1 bugs — DONE ✅** (verified 2026-07-15): both are fixed in source
  (parameterized, no longer hardcoded 0) — `part4_prune.rs` takes a `level_index` param
  (`min(speed,3)`, with the `>=3` alternate-branch guard) and `tx_search.rs` takes an
  `intra_tx_size_init_depth_rect` field — and the asserted per-feature-revert witness
  `encoder_gate_speed1_rect_and_4way_25` (in `encoder_gate_e2e_byte_match.rs`) re-diverges if either
  fix is reverted. (Earlier "need test cells to validate" note was stale.)
- **#10 cpu-used 0..9 speed-feature sweep** (Gate 2) — the large remaining item.
  (#8 qindex-from-cq and #21 decoder q62/q63 are DONE + CI-green — no longer remaining.)

**Confirmed NON-divergences (ruled out — do not re-chase):**
- **#27 `model_based_prune_tx_search_level`.** `av1_set_speed_features_qindex_dependent` sets it
  to 0 for `{<720p, base_qindex ≤ thresh}` while the port keeps 1, but the field is **inter-only**:
  the C consumer gate lives in `av1_pick_recursive_tx_size_type_yrd` behind `is_inter_block`, so it
  is inert on the all-intra KEY path and the port never reads it. `prune_tx_size_level` is inter-only
  the same way. Coordinator independently confirmed both. Empirical guard: the new asserted
  `encoder_gate_e2e_low_qindex_speed0` (cq8–30 → qindex 32–120, 12 cells) byte-matches end-to-end
  with the field left at 1 — the previously-untested aggressive-web low-q regime is now covered.

**NOT blocking single-frame-primary (non-default single-frame knobs — these ARE single-frame work
to be done before "the rest"=inter-frame, but lower priority than the primary default config):**
- **#23 QM-on encode** — reclassified here 2026-07-15 (QM is OFF by default, per the corrected
  line above). Only reached by `--enable-qm` / `tune=IQ`/`SSIMULACRA2`. Forward-quant +
  `wt_matrix` table; decoder QM decode already ported. Gate-4 knob coverage, not a primary hole.
- **#7 CDEF-strength RD search** — off by default in allintra; only for explicit `--enable-cdef`.
  Building blocks exist as shims (`cdef_find_dir`, `cdef_filter_8/16`, `shim_encode_cdef`).
- **Loop-restoration (Wiener/SGR) search** — off by default in allintra; only for explicit
  `--enable-restoration`.

## Coordination (parallel tracks)

- Max clean parallelism = **2** (one decoder agent + one encoder agent); cargo's shared
  target-dir lock serializes builds, which keeps the box safe.
- Strict crate ownership; commit with **explicit per-file staging** (`git add <paths>`, never
  `-A`/`-u`/`.`); shared `STATUS.md` via `git add -p`. Push `git push origin HEAD:main`; verify
  `git merge-base --is-ancestor HEAD origin/main`.
- Coordinator independently verifies every landing (on origin, boundary-clean, no `#[ignore]`
  / weakened asserts, gate is a real byte-identity assertion, CI green). Never trust a claim.
