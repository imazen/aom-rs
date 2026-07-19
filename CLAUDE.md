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

## Decoder desyncs: use the instrumented libaom decoder FIRST (do not bisect by hand)

An arithmetic-decoder desync almost never shows up where it is caused. The cheapest way to
localize one is to compare against the REAL libaom decoder's own instrumentation, which dumps
both C's per-block mode info and C's exact per-symbol sequence. **A build already exists at
`/root/aom-inspect/examples/inspect`** (`CONFIG_INSPECTION=1 CONFIG_ACCOUNTING=1`; rebuild the
same way if it is ever lost):

```
/root/aom-inspect/examples/inspect --limit=<N> -bs -ts -m -r -mm <vector>.ivf   # per-MI block grid
/root/aom-inspect/examples/inspect --limit=<N> -a          <vector>.ivf         # per-symbol accounting
```

The block dump is per-MI (replicated over each block's footprint), so collapse it to block
top-lefts to get C's block list; diff that against the port's walk to find the FIRST structural
divergence. Then read the accounting around that block: it gives every `aom_read_symbol` tagged
by its reading function, so **"the port read the same symbol VALUES but off a different CDF row"**
is directly visible instead of being inferred. That failure mode (a probability-only drift that
desyncs several reads later, with everything looking correct locally) is the dominant one on this
codebase — it is the KB-6 signature on the encoder side and was BOTH inter-decode roots found on
2026-07-19.

Two gotchas, both verified:
- **The JSON emitter drops the first symbol at every new block position** (`put_accounting`,
  examples/inspect.c: on a context change it emits `[x,y]` *instead of* that symbol's triple). So
  `read_skip_txfm` — the first read of most blocks — never appears. Do not conclude a symbol is
  unread; infer it from what follows (e.g. no coeff/var-tx symbols at all ⇒ that block is skip).
- Entries are `[id, bits, samples]` against `symbolsMap`, **not** `[id, value, bits]`; `samples`
  is the number of `aom_read_symbol` calls aggregated at that (tag, position). Reconciling those
  counts against the port's expected read sequence is itself a strong check — e.g. inter-intra:
  40 allowed blocks ⇒ 40 flag reads, +2 wedge flags +1 wedge index = the 43 C records.

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
- **Encoder cross-check — RESOLVED ✅ 2026-07-18 (SB128 encode landing, PARITY C8).** The encoder
  pack now writes txbs in the C 64×64-chunk **L/U/V-interleaved** order for >64 blocks
  (`av1_write_intra_coeffs_mb`, encodetxb.c:431-472) and the search/re-encode predict+reconstruct
  in the matching `av1_foreach_transformed_block_in_plane` mu-64 chunk order (encodemb.c:560-582).
  Byte-exact vs real `aomenc --sb-size=128` (`aom-bench/tests/sb128_e2e.rs::sb128_coded_128_leaf
  _e2e` — a smooth diag ramp at 256² cq55/cq63, anti-vacuity-checked to actually code a 128-level
  partition; the photographic `quantizer-00` crops in `sb128_natural_e2e` split to ≤64 even at
  cq63, so they exercise only the SEARCH's 128-NONE evaluation, not the pack interleave). NOTE the
  earlier "256×256 cq63 is evidence the
  order is correct" claim was **vacuous** — a 256² SB64 frame has no >64 coding blocks, so the
  >64-block order was never exercised until this landing (the plane-sequential pack was
  coincidentally correct only because ≤64 leaves are a single chunk).
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

### KB-4 — Encoder: bd10/bd12 coded-eob divergence (was "RD-decision divergence at high bit depth") — FIXED ✅ (BOTH roots; task #31)
- **FIXED 2026-07-16 (this landing) — OUTPUT_ENABLED tx_type_map copy semantics in `encode_b_intra_dry`.**
  The mono/4:2:0 aggressive-HF divergence (bd10 cq12, bd12 cq8, bd12 cq20 in
  `kb4_bd10_rd_localize.rs`) was NOT a high-bit-depth RD-scaling bug: the port ran C's single
  OUTPUT_ENABLED walk TWICE (the SB-root winner context/recon walk + the pack re-walk) with DRY
  (alias) tx_type_map semantics, so the first walk's `eob==0 → DCT_DCT` resets
  (encodemb.c:770-779, `update_txk_array`) leaked into the pack's re-quant input. A skip-winning
  txb (non-DCT search winner quantizing to eob 0 — exactly what aomenc codes) re-quantized as
  DCT_DCT with eob>0 in the coded bytes (e.g. the bd10 cq12 mi(14,12) BLOCK_16X8/D45 txb5:
  search=ADST_DCT/eob0, coded=DCT/eob1). C's semantics (`av1_update_state`,
  encodeframe_utils.c:217-231): DRY walks **ALIAS** `ctx->tx_type_map` — resets PERSIST into the
  stored winner map (real C behaviour; do NOT "fix" by cloning); OUTPUT_ENABLED **copies** ctx
  into the frame-level map and the resets land THERE, ctx untouched. **Fix:**
  `encode_b_intra_dry`/`encode_sb_dry` take `output_enabled`; the SB-root winner walk
  (partition_pick.rs, C partition_search.c:6010) and the pack walk (pack.rs — the same C walk,
  re-run) use a transient frame-map clone; the mid-candidate propagation (C :3613-3616) and
  non-SB winner walks (C :6023, `should_do_dry_run_encode_for_current_block` :5556 — last SPLIT
  children skipped) keep the alias. The `COracle`/`CPick` differential references mirror the
  split (they had shared the port's mis-model). bd10/12-amplified (larger RD magnitudes make
  non-DCT-eob0 near-tie txbs common) but NOT bd-specific in mechanism: the same leak closed
  KB-6's bd8 `quantizer-00 128×128 cq63` cell.
- **Prior "RD-DECISION layer bd scaling" localization REFUTED (2026-07-16):** per-tx_type
  rate+dist are byte-exact vs the REAL-C leaf chain (`kb4_txb2_probe.rs`); tx-type search order
  matches C (txk_map stays natural `{0..15}` at speed-0 — `prune_tx_2D` reorders only under
  `prune_tx_type_est_rd`, speed≥4); `ref_best_rd` threading and the `adaptive_txb_search` break
  match C, and the break never changed the winner on any divergent txb (with-break == full-eval
  on every one). The kernels were indeed byte-exact — the divergence was PASS-STRUCTURE, not
  arithmetic. (An earlier blanket per-pass-clone attempt regressed 3→5 cells because it also
  cloned C's DRY alias walks and the rd_pick CfL store-luma reencode — both must keep mutating.)
- **Gates:** mono/420 promoted to `kb4_gate_bd10_bd12_mono_hf_byte_match`
  (kb4_bd10_rd_localize.rs) — the full bd10/bd12 × cq8/12/20 × hf/ramp sweep byte-matches real
  aomenc (12/12). Non-420: the other KB-4 witness was FIXED separately by **1ecfafb** (AB HORZ_A
  nested sub-block reuse) — all 4 bd10 non-420 cells (444/422 × 64²/128² cq32) byte-match,
  asserted by `encoder_gate_bd10_non420_e2e_kb4_repro`.

### KB-5 — Encoder: lossless (cq0 / qindex 0) KEY encode — FIXED ✅ (mono + 4:2:0 both byte-exact, hard-asserted; #32 closed)
- **MONO FIXED 2026-07-16.** Mono 64² cq0 (coded-lossless allintra KEY) is now an end-to-end BYTE
  MATCH vs real aomenc, hard-asserted in `encoder_gate_lossless_cq0_e2e_kb5_repro`
  (encoder_gate_chroma_ss_e2e.rs). THREE fixes were required (the two originally localized below,
  plus a third found during landing):
  1. **Harness two-pass (#32):** `run_case` now mirrors the decoder's two-pass lossless probe —
     parse, compute coded_lossless from the probe's quant params (base_qindex==0 && all 5 plane
     q-deltas 0), re-parse with `cfg.coded_lossless/all_lossless=true`.
  2. **Forward WHT (#33):** `av1_fwht4x4` ported into aom-dsp's `transform` module
     (`crates/aom-dsp/src/transform/inv_txfm2d.rs`; bit-exact vs `av1_fwht4x4_c`,
     gated by `fwht4x4_diff`); `QuantParams` gained a `lossless` flag; `xform_quant` (lib.rs) and
     every encoder recon site (encode_intra / tx_search / intra_uv_rd) route coded-lossless TX_4X4
     through WHT/IWHT via `av1_inverse_transform_add(.., eob, lossless)`. The SATD fast model stays
     DCT (`av1_quick_txfm` forces lossless=0 in C — intra_uv_rd.rs:800 unchanged, do NOT "fix" it).
     The differential oracle (tests/common/mod.rs `c_search_tx_type_p` / `c_uniform_txfm_yrd`) uses
     `ref_fwht4x4`/`ref_highbd_iwht4x4_add` for lossless — a faithfulness correction (real C uses
     WHT for lossless, hybrid_fwd_txfm.c:83-86).
  3. **Entropy-context propagation (the actual byte-divergence root, found via decode-both
     localization `kb5_lossless_localize.rs`):** the WRITTEN `txb_skip_ctx`/`dc_sign_ctx` must
     derive from the REAL above/left neighbour entropy context ALWAYS — C's write path
     (`av1_write_coeffs_txb`, encodetxb.c:596-598) is never gated on the trellis; only C's
     trellis-local `ta/tl` fill is (encodemb.c:817-819). The port shared one ta/tl array for both
     uses (encode_intra.rs, luma + chroma arms) and seeded it from the real context only when the
     trellis was on; coded-lossless runs trellis-OFF (USE_B_QUANT_NO_TRELLIS), so a block with a
     coded left neighbour wrote ctx 1/0 instead of the real 3/1 and desynced the decoder. Fix:
     always seed ta/tl from the real neighbour context.
- **420 FIXED 2026-07-16 (mono landed as ba560eb; 420 this landing) — CfL banned at coded-lossless
  in the SEARCH.** The former "≤1-unit chroma RD near-tie" was a search-SPACE gap, not RD math:
  `partition_pick.rs`'s leaf `cfl_allowed` was `!lossless && w<=32 && h<=32`, but C's
  `is_cfl_allowed` (blockd.h) allows CfL at LOSSLESS whenever the partition size equals the
  transform size — `get_plane_block_size(bsize, ssx, ssy) == BLOCK_4X4` (at 420: every
  8×8-and-below chroma-ref leaf). Measured mechanism (instrumented-sibling-C vs port partition
  dumps, faithfulness-gated byte-identical first): at the first 16×16 node NONE matches EXACTLY
  (both 235604, rdmult 52, dist=0 everywhere at lossless) but C's 8×8 SPLIT children pick
  **UV_CFL_PRED** (~16k cheaper rate per chroma-carrying leaf; luma-only 4×4 subs byte-identical)
  → C SPLIT 235256 beats NONE by 348; the port's missing CfL candidates inflated its children and
  starved SPLIT child-3 at the 63759 remaining budget → NONE → desync. **Fix:** route the leaf
  gate through the shared (already-correct, pack.rs already used it) helper
  `aom_entropy::partition::is_cfl_allowed(bsize, env.lossless, ss_x, ss_y)` — expression-identical
  at !lossless, so non-lossless gates are untouched (verified: all chroma-ss/KB-4/KB-6 gates
  unchanged-green). The `CPick` reference in `partition_pick_diff.rs` carried the SAME transcribed
  gate (a shared bug that differential structurally could not catch) — also routed through the
  helper. **Refuted en route (do not re-chase):** the chroma UV RD math at qindex 0 is CLEAN — the
  new `txfm_uvrd_matches_c_walk_lossless_q0` differential (txfm_uvrd_diff.rs; UvRdEnv oracle
  winner-recon taught IWHT-for-lossless in common/mod.rs to match hybrid_fwd_txfm/inverse
  dispatch) proves port==C at qindex 0 across 14 chroma-ref shapes × 8 iters for rate/dist(=0,
  physics-asserted)/sse/winners/recon PLUS strict-`>` budget-boundary agreement at
  min_rd−1/min_rd/min_rd+1. **Gate:** `encoder_gate_lossless_cq0_e2e_kb5_repro` hard-asserts BOTH
  mono AND 420 byte-match (promotion from `assert_open_divergence` per its designed contract).
  The full lossless envelope (coded-lossless cq0 KEY, mono + 4:2:0) is byte-exact; #32 closed.

### KB-6 — Encoder: REAL-content RD divergence at bd8 4:2:0 (PRIMARY config) — FIXED ✅ (all roots landed; real-content map 30/30)
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
- **CLOSED 2026-07-16 — the REAL-CONTENT MAP IS 30/30 BYTE-EXACT** (was 26/30 after the KB-4
  OUTPUT_ENABLED fix + the partial-SB chunk series; 29/30 after the entropy-stamp/edge-CDF
  landing; the last cell, 196² cq48, closed by the pack write-ctx fix below). Every
  interior-crop cell now matches: size-64×64 all 6 cq (cq5/12/20/48/63 with FIX #1; cq32 with
  1ecfafb — AB HORZ_A nested sub-block reuse); quantizer-64² 6/6, film-64² 6/6, quantizer-128²
  6/6 — the former cq5 low-q cluster and the quantizer-128² cq12/20/32 near-ties cleared with
  the partial-SB chunk series' distortion-clip landings, and **quantizer-128² cq63 + 196×196
  cq63 closed 2026-07-16 by the KB-4 OUTPUT_ENABLED tx_type_map fix** (the port coded DCT-eob1
  where real codes an eob0 skip — the reset-leak signature, present in interior AND edge SB
  rows).
- **DISTINCT SUB-GAP — partial-SB (frame dims not a multiple of 64px) — FULLY FIXED (all 6 cq).** Landed: the CHUNK series (`3167800` CHUNK 0+1 true-frame harness + luma visible
  dist clip, `7c468ee` CHUNK 2 chroma visible clips via `max_block_units`, `4b8b1f1` CHUNK 3
  `set_partition_cost_for_edge_blk`), the KB-4 OUTPUT_ENABLED tx_type_map reset-leak fix
  (`a2dd28e`, closed 196² cq63), and the **frame-edge entropy-stamp tail-zero + frame-init edge
  partition CDF fix** (closed cq12/20/32; map 26/30 → **29/30**). That last root was pinned by a
  full C-vs-port symbol-level bit trace (throwaway instrumented sibling C at `/root/kb6-edge-instr`,
  byte-gate-verified vs real aomenc): the apparent "mi(48,0) 16×8-vs-8×4 over-split" was NOT a
  search decision — the port's search picks C's EXACT tree and every leaf RD matches C to the unit;
  the port's PACK also writes the same symbols. The divergence was a WRITE-side probability defect:
  (a) **`av1_set_entropy_contexts` (blockd.c:29) zeroes the beyond-visible TAIL of an edge txb's
  above/left entropy-context footprint** (`memset(a + above_contexts, 0, txs_wide - above_contexts)`)
  while the port's tile stamp (encode_sb.rs) wrote the cul across the FULL footprint — phantom
  nonzero culs at out-of-frame mi cols (50-51 luma / 25 chroma) fed later edge blocks'
  full-footprint `get_txb_ctx` reads, flipping SB(32,48)'s txb_skip_ctx (1→3 luma, 8→9 U) → same
  symbols on different-probability cdf rows → +3 bits → stream desync at tile-byte 975 → the
  decoded "over-split" artifact; (b) the CHUNK 3 edge partition-cost gather read the SB-adapted
  partition CDF, but C's `set_partition_cost_for_edge_blk` (partition_search.c:3415) reads
  **`cm->fc` — the frame-init table** (measured: C's gather rows == `default_partition_cdf`),
  a shipped-libaom mixed-source quirk (interior costs track the adapting tile state; edge gather
  does not). Note the C encode-path per-txb stamp `av1_set_txb_context` (encodemb.h) is
  full-footprint UNclipped — only the tokenize/persistent stamp clips; the port's local ta/tl
  stamps correctly mirror the former and needed no change.
  **All six 196² cells (cq5/12/20/32/48/63) are asserted byte-match gates** in
  `encoder_gate_real_image_e2e_kb6_repro` (now a FULL 30-cell byte-match gate).
  **cq48 (the LAST cell) FIXED 2026-07-16 — pack WRITE-ctx source (tokenize vs trellis):**
  decode-both + pass-context markers proved the search was ALREADY C-identical at the divergent
  leaf (mi(0,48) 32×64 SMOOTH; both OUTPUT_ENABLED walks requantize txb4 to C's coded
  (tt1, eob37)) — the decoded "(eob4, tt2)" was a desync artifact of the port's own bits. C caches
  the pack's `(txb_skip_ctx, dc_sign_ctx)` in the TOKENIZE walk
  (`av1_update_and_record_txb_context`, encodetxb.c, OUTPUT arm; `av1_write_coeffs_txb` writes the
  CACHED pair) derived from the PERSISTENT entropy arrays — whose within-leaf stamps are
  edge-CLIPPED (`av1_set_entropy_contexts`) — while the TRELLIS uses the encode walk's
  full-footprint local `av1_set_txb_context` stamps; the port used the trellis pair for the write
  too. `txb_skip_ctx` is OR-based (tail-zero inert — why the 29/30 landing sufficed there) but
  `dc_sign_ctx` is SIGN-OF-SUM: at txb blk(8,0) (16×16, vis 8×16) the above tail-zero drops +2
  (C: −4+2 = −2 → ctx 1; port: −4+4 = 0 → ctx 0) → ONE DC-sign symbol on a different cdf row →
  bits diverge at tile byte ~253 with IDENTICAL symbols everywhere. Fix: `encode_b_intra_dry`
  Step 4 (encode_sb.rs, the tokenize-equivalent stamp loop) derives the write pair from the
  persistent arrays per txb — before that txb's clipped stamp, C's exact read point — and
  overwrites the cached `TxbEncode` pair (dcs gated on `qcoeff[0] != 0`, Y+U+V planes); sole
  consumer is `pack_plane_coeffs`. Interior txbs derive identical values (structurally zero-diff
  on the green corpus).
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
- **Root cause: MULTIPLE KB-2-class near-ties, several roots landed.** FIX #1 (luma re-encode
  edge filter) took real 64×64 from 2/6 to 5/6; 1ecfafb (AB HORZ_A nested reuse) closed 64×64 cq32
  + the 4 bd10 non-420 KB-4 cells; the partial-SB chunk series (distortion visible-clips + edge
  partition cost) cleared the cq5 low-q cluster + the quantizer-128² cq12/20/32 near-ties + 196²
  cq5; the KB-4 OUTPUT_ENABLED tx_type_map fix (2026-07-16) closed quantizer-128² cq63 + 196²
  cq63; the frame-edge entropy-stamp tail-zero + edge partition CDF landing (4567e58) closed 196²
  cq12/20/32; and the pack write-ctx fix (2026-07-16) closed the final cell, 196² cq48 — the
  last three roots were all WRITE-side probability defects (identical symbols on
  different-probability cdf rows), not search decisions.
- **Repro (COMMITTED, CI-green characterization):** `encoder_gate_real_image_e2e_kb6_repro` prints the
  full per-cell MATCH/MISMATCH map, asserts a byte-exact CONTROL (64×64 cq20 — harness-faithfulness +
  regression guard), and asserts the KB-6 divergence is still PRESENT (gates: when the port becomes
  byte-exact on real content the test FAILS → promote it to a full `report_and_assert` byte-match
  gate). Not a weakened test — the correct end state is full byte-identity on real content.
- **Next step: NONE — the real-content map is complete (30/30).**
  `encoder_gate_real_image_e2e_kb6_repro` is promoted to a full byte-match gate over all 30
  cells; any real-content divergence is now a regression, not an open KB-6 axis. (KB-1, KB-5,
  KB-7, KB-8 and the Gate-2 cpu-used sweep remain separate tracks.)
- **Priority note:** KB-6 hits the single most common real-world case (bd8 4:2:0 photographic content
  at web qindex), so it is arguably higher-impact than the bd10/bd12 (KB-4) and lossless (KB-5)
  corners. Sequencing is the coordinator's call.

### KB-7 — Encoder: `--cpu-used=3/4` cq12/cq32 4:2:0 partition flips — FIXED ✅ (TWO speed-feature-port roots; speed-3 AND speed-4 gates 64/64)
- **FIXED 2026-07-16.** All 8 pinned cells (3 at speed-3 + 5 at speed-4) now BYTE-MATCH real
  aomenc; both gates assert FULL 64/64 byte-identity. The "latent chroma-RD near-tie"
  hypothesis was REFUTED by the sibling-C RD dump (throwaway instrumented C, kb7-instr inject
  pattern; validated byte-inert vs the clean build): every leaf RD — NONE/HORZ/VERT, luma AND
  chroma parts, and every SPLIT child total — matched C **to the unit**. The flips were TWO
  partition-search-SPACE / speed-feature-port gaps:
  1. **(speed>=3, closed ALL 3 speed-3 pins) `av1_ml_prune_4_partition`'s OLD-model branch was
     unported.** At `ml_4_partition_search_level_index = 3` (allintra speed>=3) C flips
     `ml_model_index = (level < 3) == 0` (partition_strategy.c:1359) → the old
     `av1_4_partition_nn_*` weight set (LABEL_SIZE=4), **UNnormalized** features,
     `int_score[i] = (int)(100*score[i])`, `thresh = max_score − {500,500,200}` (16/32/64),
     zero-then-set from the label bits (:1472-1497). On these cells it prunes HORZ_4/VERT_4 at
     every 32×32 node (measured: scores like [530,−348,0,−392], thresh=30 → only label 0 ⇒ both
     pruned). The port's `predict_4partition_prune` guarded `level_index >= 3` as a NO-OP, so it
     searched HORZ_4 and found a cheaper 4-way (two-tone 64² cq12: child-0 HORZ_4 rdcost 12.9M vs
     NONE 16.5M) → root NONE→SPLIT. **Fix:** transcribe the OLD weight tables
     (`xtask/transcribe_part4_nn.py` → `part4_nn_weights.rs` `OLD_*`) + the old-branch decision in
     `part4_prune.rs` (normalize skipped, int-score/max−thresh, OVERWRITE-from-zero semantics —
     C can resurrect a pre-ML-cleared flag; the caller re-ANDs only the interior-envelope
     frame-fit guard). Also added the missing `av1_nn_output_prec_reduce` (ml.c:19 — BOTH
     `av1_ml_prune_4_partition` call sites pass `reduce_prec=1`; C's `+ 0.5` is a DOUBLE literal)
     to part4's NN — and the same latent gap in `ab_nn_prune.rs` (the AB NN call :1296 is also
     reduce_prec=1). Witness: `part4_old_nn_diff.rs` — 4000 random-input decisions identical to a
     REAL-`av1_nn_predict_c` oracle on the same OLD tables.
  2. **(speed>=4, closed ALL 5 speed-4 pins) the chroma-HOG force-disable tail was unported.**
     The UNCONDITIONAL tail of `set_allintra_speed_features_framesize_independent`
     (speed_features.c:608-616) zeroes `chroma_intra_pruning_with_hog` whenever
     `prune_chroma_modes_using_luma_winner` is on (allintra speed>=4; this also deadens the
     speed-5/6 `=3/4` settings). Measured: the instrumented C computes ZERO chroma-HOG masks at
     cpu-used=4. The port kept the HOG live at speed 4 and HOG-pruned UV_V_PRED where C evaluates
     and picks it (two-tone 64² cq12 root NONE: C uv=V 58469617 vs port uv=SMOOTH 58779332) →
     different chroma bytes. **Fix:** the tail in `SpeedFeatures::set_allintra` + the inline
     `chroma_hog_level` gate in `partition_pick.rs` (`&& !prune_chroma_luma_winner`); the
     `UvLoopPolicy` build now threads the luma-winner prune independently of the HOG mask
     (they were coupled — dropping the HOG must not drop the luma-winner prune).
- **Verified locally (worktree, rebased over 57d5ce0):** speed-3 gate 64/64, speed-4 gate 64/64
  (both promoted from pinned-residual to full byte-identity asserts), new single-cell asserted
  witnesses `kb7_rd_localize.rs` (cpu3 + cpu4, with decode-both diff on failure),
  `part4_old_nn_diff` 4000/4000, `speed4_allintra_deltas_match_source` corrected to the
  C-source value (`chroma_intra_pruning_with_hog == 0` at speed 4), full `cargo test -p
  aom-encode` **149 passed / 0 failed**. Speed-0/1/2 byte gates unaffected (the old-model branch
  only fires at level 3; the prec-reduce is decision-neutral on those grids — now faithful).

### KB-8 — Encoder: `--cpu-used=4` speed-4 deltas — PORTED ✅ (64/64 after the KB-7 roots; luma was byte-exact at 59/64)
- **Status (2026-07-16): every documented speed-4 delta is PORTED + LIVE — 64/64 cells byte-identical**
  vs real aomenc (`encoder_gate_speed4_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}), up from 35/64 baseline → 51/64 (chunk 1 series) →
  59/64 (the winner-mode flip) → **64/64 (the KB-7 roots: the level-3 OLD-model 4-way ML prune +
  the speed>=4 chroma-HOG disable tail — see KB-7)**. ALL 32 mono cells were already byte-exact
  at 59/64 (the speed-4 LUMA path); the 5 former 4:2:0 residuals (`diag 128² cq12`, `two-tone
  64² cq12/cq32`, `vgrad 128² cq12`, `vgrad 64² cq12`) were KB-7's two roots, not a missing
  speed-4 delta (confirmed: both are speed-feature gates, one shared with speed 3, one
  speed-4-specific).
- **The full landed chunk series (each verified on origin/main):**
  1. `prune_chroma_modes_using_luma_winner` + NON_DUAL LF search (e8c662f, 51/64).
  2. SATD trellis-skip body `skip_trellis_opt_based_on_satd` (16d4d85) — unit-tested vs REAL C
     (`ref_satd` = exported `aom_satd_c`).
  3. Stage-aware `TxTypeSearchPolicy` derivation (7bd30fb) — MODE_EVAL/WINNER_MODE_EVAL coeff-opt
     + tx-domain columns per `set_mode_eval_params`, validated vs the C tables.
  4. `USE_LARGESTALL` tx-size arm (42bdffc) — `choose_largest_tx_size` demotion tables verified vs C.
  5. `use_default_intra_tx_type` in `get_tx_mask_intra` (96eeb71) + threading (9c6ed2a) —
     differential vs the C shim across use_default × screen sweeps.
  6. Winner-mode two-pass skeleton in `rd_pick_intra_sby_mode_y` (0ee9f97) — `store_winner_mode_
     stats` C-semantics unit-tested; `use_rd_based_breakout` rd_thresh (AOMMIN) in the depth loop.
  7. Est-rd tx-type prune (264bba4) — `av1_cost_coeffs_txb_laplacian` (REAL-C differential across
     15,960 cases) + `prune_txk_type` + txk_map reorder; LIVE on intra in the WINNER pass.
  8. THE FLIP (this landing): `set_allintra(4)` real values (`perform_coeff_opt=5`,
     `tx_domain_dist_thres_level=3`, `fast_intra_tx_type_search=2`, `winner_mode_tx_type_pruning=2`,
     `prune_2d_txfm_mode=PRUNE_3`, `prune_tx_type_est_rd=1`, `enable_winner_mode_for_{coeff_opt,
     use_tx_domain_dist,tx_size_srch}=1`, `multi_winner_mode_type=MULTI_WINNER_MODE_DEFAULT(=2)`);
     `use_rd_based_breakout_for_intra_tx_search=1` at speed>=3 (:460 — speed-3 gate re-verified
     61/64, empirical no-op confirmed); the two-pass wiring in `partition_pick.rs` (per-leaf
     `WinnerModeCfg` derivation); BOTH split-info prunes (`prune_ext_part_using_split_info`:
     the AB `evaluate_ab_partition_based_on_split` at level 2 = speed>=4 — inert at qindex>=128
     by its threshold formula — and the 4-way `prune_4_partition_using_split_info` at level 1 =
     speed>=3, via `split_part_rect_win` rect-win threading through the SPLIT recursion).
- **Key facts for future speeds (verified against source):** `top_intra_model_count_allowed` stays
  **3** at speeds 4 AND 5 — the `=2` drop is **speed>=6** (:533, inside the `if (speed >= 6)`
  block at :527; an earlier note here mis-attributed it to speed>=5); `MULTI_WINNER_MODE_DEFAULT=2`
  / `FAST=1` (speed_features.h:226/230), `winner_mode_count_allowed={1,2,3}`; the AB split-info
  threshold `min(3*(2*(MAXQ-q)/MAXQ),3)` is 3 for q<=127 / 0 for q>=128; C's chroma search runs
  DEFAULT_EVAL (rdopt.c:3659 resets right after the luma two-pass); the winner re-eval
  (`intra_block_yrd`) gets NO ALLINTRA variance factor yet compares vs the factored first-pass
  best_rd (C asymmetry, preserved); C's LARGESTALL arm bypasses `uniform_txfm_yrd`'s rate assembly
  — equivalent to it with `tx_mode_is_select=false` (tx_size_rate=0), which is how the port models
  it.
- **Gate asserts FULL 64/64 byte-identity** — FAILS on any regression.

### KB-9 — Encoder: `--cpu-used=5` speed-5 deltas — PORTED ✅ (64/64 byte-identical, 0 residuals)
- **Status (2026-07-16): every speed-5 delta is PORTED + LIVE — 64/64 cells byte-identical** vs
  real aomenc (`encoder_gate_speed5_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}). No pinned residuals: the two cells that had been
  KB-7-pinned at speed 4 (`two-tone 64² cq12`, `vgrad 128² cq12` — since closed there by the
  KB-7 roots) byte-match at speed 5 independently, because the AB/4-way disable (below) removes
  the near-tie's partition candidates from the search space entirely.
- **LIVE deltas (each individually witness-verified by bisect during landing):**
  1. `winner_mode_sf.multi_winner_mode_type = MULTI_WINNER_MODE_FAST` (:524): the luma two-pass
     stores/re-evaluates the top-**2** winners (speed 4: top-3) — `winner_mode_count_allowed`
     rdopt_utils.h:236, already parameterized through `WinnerModeCfg::max_winner_count`. Flips
     `two-tone 64² mono cq63` + `420 cq63` on the gate grid (the mono flip proves it luma-side).
  2. `part_sf.ext_partition_eval_thresh`: default BLOCK_8X8 through speed 4; at speed 5 the
     framesize-independent :510-511 sets `screen ? BLOCK_8X8 : BLOCK_16X16`, then the
     qindex-dependent `aggr = AOMMIN(4, speed-2)` == 3 arm (:2947-2962) sets **BLOCK_128X128
     UNCONDITIONALLY for sub-480p frames** (no boosted/intra gate) → `bsize > thresh` never holds
     → **AB + 4-way partitions are never evaluated** on sub-480p KEY frames. Consumers:
     `allow_ab_partition_search` (partition_search.c:4005) + `prune_4_way_partition_search`
     (:4136), both now read `ext_partition_eval_thresh_allintra_key` (partition_pick.rs; the
     other qindex-dep arms are dead on KEY — boosted + intra-only; speed>=6 = BLOCK_128X128 for
     ALL sizes; `ext_part_eval_based_on_cur_best` is GOOD-only, :1013). Flips the 2 cq12 cells (the former speed-4 KB-7 pins).
- **Set-then-overridden:** `chroma_intra_pruning_with_hog = 3` (:515) is zeroed by the :608-615
  final override (chroma HOG off at speed>=4 — the KB-8 entry documents the override fix).
- **Screen-only:** `intra_cnn_based_part_prune_level`: screen arm 0 → 1 (:512-513; non-screen
  stays 2). Wired through the existing CNN prune (`predict_decision` handles level 1's
  `none_disallowed` exemption); byte-inert on the (non-screen) gate grid.
- **Verified INERT on the allintra KEY envelope:** `simple_motion_search_prune_agg=LVL5` (:509,
  motion), `use_coarse_filter_level_search=0` (:517, ALREADY the default — init :2532),
  `disable_wiener/sgr_filter` (:519-520, restoration off), `prune_mesh_search=LVL_2` (:522,
  intrabc/motion), qindex-dep `winner_mode_tx_type_pruning=3` (:3059, `!(intra||screen)` —
  stays 2), qindex-dep `prune_sub_8x8_partition_level=0` (:3070, field only raised at speed>=6),
  qindex-dep `rect_partition_eval_thresh` aggr 0→1 (:2980, `!boosted`). The framesize-DEPENDENT
  setter has NO speed-5 block (:302 jumps 4→6). LF stays NON_DUAL (:496; LPF_PICK_FROM_Q is
  speed>=6), tx/winner tables all carry speed-4 values.
- **Anti-vacuous witness (asserted):** `encoder_gate_speed5_vs_speed4_sf_witness` — the port with
  SPEED-4 features vs real `aomenc --cpu-used=5` DIVERGES (4 cells incl. mono cq63); with speed-5
  features it matches. Gate asserts full 64/64 — FAILS on any regression.
- **Speed-6 prep facts (verified against source while here):** speed>=6 block :527-564 —
  `top_intra_model_count_allowed=2` (:533), `prune_filter_intra_level=2` (:529),
  `intra_pruning_with_hog=4` (:531) + `chroma_intra_pruning_with_hog=4` (:530, still overridden
  to 0), `cfl_search_range=1` (:532), `adapt_top_model_rd_count_using_neighbors=1` (:534),
  `prune_luma_odd_delta_angles_in_intra=1` (:535), `multi_winner_mode_type=OFF` (:561),
  `prune_winner_mode_eval_level=1` (:562), `dc_blk_pred_level=1` (:563), `winner_mode_tx_type_
  pruning=3` + `prune_tx_type_est_rd=0` (:551-552), `prune_intra_tx_depths_using_nn` (:553),
  `perform_coeff_opt=6` + `tx_domain_dist_level=3` (:555-556), `lpf_pick=LPF_PICK_FROM_Q` (:559 —
  **building block LANDED**: `pick_filter_level_from_q` in lf_search.rs, oracle-validated vs real
  cpu-6 header levels by `speed6_prep_lf_from_q_matches_real_aomenc`; needs only the harness flip),
  partition prunes :537-546 (`prune_rectangular_split_based_on_qidx=2`, `prune_rect_part_using_
  4x4_var_deviation/none_pred_mode`, `prune_sub_8x8_partition_level=1`, `prune_part4_search=3`,
  `default_max_partition_size=BLOCK_32X32`!), framesize-dep :304-316 (`use_square_partition_only_
  threshold=BLOCK_16X16` etc.). Substantially new machinery (LPF-from-Q, NN tx-depth prune,
  DC-block prediction, odd-delta-angle prune) — NOT a pure re-parameterization like speed 5 was.
  **All of the above LANDED — see KB-10.**

### KB-10 — Encoder: `--cpu-used=6` speed-6 deltas — PORTED ✅ (64/64 byte-identical on the canon grid; ONE pinned-open near-tie class on the noise extension)
- **Status (2026-07-16): the canon gate is 64/64 byte-identical** vs real aomenc
  (`encoder_gate_speed6_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}) + the anti-vacuous witness
  (`encoder_gate_speed6_vs_speed5_sf_witness`: port with FULL speed-5 features vs
  `aomenc --cpu-used=6` DIVERGES on `vgrad 64² cq32` mono+420; with speed-6 features it
  matches). Speed 0-5 gates all re-verified byte-unchanged. Speed 6 is NEW MACHINERY
  (speed_features.c:527-564 + framesize-dep :304-316 + qindex-dep aggr=4), not a
  re-parameterization — landed as one chunk after the KB-9 prep-facts series:
  1. **`lpf_pick = LPF_PICK_FROM_Q`** (:559): the closed-form KEY LF derivation
     (`lf_search::pick_filter_level_from_q`, chunk-1 building block 5935250,
     oracle-validated vs real cpu-6 headers) replaces the reconstruction search —
     wired in the harness LF derivation at `speed >= 6` (the `non_dual` flag's shape).
  2. **Partition prunes** (bisect: baseline chunks-2+3 took the map 0→54/64):
     `default_max_partition_size = BLOCK_32X32` (:546 — `set_max_min_partition_size`
     min(sf, CLI cap, sb) forces square-split-only at the 64² root),
     `use_square_partition_only_threshold = BLOCK_16X16` (framesize-dep :315),
     `ext_partition_eval_thresh = BLOCK_128X128` for ALL sizes (qindex-dep aggr=4
     else-arm :2963), `prune_rectangular_split_based_on_qidx = 2` (:537, the
     qindex-thirds rect kill), `prune_rect_part_using_4x4_var_deviation` (:539 — arm 2
     of the ALLINTRA var block, `do_rectangular_split = 0` when `var_max - var_min <
     3.0`; also WIDENS the stats computation to sub-16x16 nodes),
     `prune_rect_part_using_none_pred_mode` (:540 — post-NONE mode-class rect prune;
     needs the new `ModeGrid::bsizes` neighbour-bsize stamps for
     `is_neighbor_blk_larger_than_cur_blk`), `prune_sub_8x8_partition_level = 1`
     (:541 — disable splits at 8x8 when either neighbour block is larger),
     `prune_part4_search = 3` (:543 — inert: 4-way is off via the ext threshold).
  3. **Intra mode loop**: `top_intra_model_count_allowed = 2` (:533) +
     `adapt_top_model_rd_count_using_neighbors` (:534 — the neighbour-mode-adaptive
     prune slot; machinery pre-existed in intra_rd.rs, now threaded),
     `prune_luma_odd_delta_angles_in_intra` (:535 — evens-first delta order
     `{-2,2,-3,-1,1,3}` + the even-neighbour rd_thresh prune; pre-existed, now gated
     on), `intra_pruning_with_hog = 4` (:531, luma HOG threshold 0.4),
     `prune_filter_intra_level = 2` (:529 — no filter-intra search,
     intra_mode_search.c:239).
  4. **predict_dc skip-block prediction** (`dc_blk_pred_level = 1`, :563 → per-stage
     `predict_dc_levels[1] = {1,1,0}`): `predict_dc_only_block` (tx_search.c:2011) in
     the DEFAULT_EVAL + MODE_EVAL tx-type searches — `pixel_diff_stats` (DOUBLE-norm
     mse/mean/var over the visible txb) + the low-var/low-mean eob-0 skip fast path.
     KEY QUIRK ported: the skip path's `zero_blk_rate` reads `get_txb_ctx` at the
     BLOCK ORIGIN from the PERSISTENT entropy arrays (C re-derives ctxa/ctxl via
     `av1_get_entropy_contexts` and passes UN-offset pointers — every txb of the block
     shares the origin ctx; threaded as
     `TxTypeSearchInputs::predict_skip_zero_blk_rate`). Bisect: flips 4 canon cells
     (diag 128² mono+420 cq32/48 — mono proves it luma-side); 2384 luma fires on the
     canon grid, chroma fires on the flat-uv extension cells.
  5. **8x8 NN intra-tx-depth prune** (`prune_intra_tx_depths_using_nn`, :553):
     `ml_predict_intra_tx_depth_prune` (tx_search.c:2823) — transcribed weights
     (`xtask/transcribe_intra_tx_nn.py` → `intra_tx_nn_weights.rs`),
     `get_mean_dev_features` (14 features incl. log1pf(source_variance) +
     log1pf(dc_q²/256)), 16-node ReLU + prec-reduce, thresholds ±0.405465 →
     TX_PRUNE_SPLIT (skip smaller depths) / TX_PRUNE_LARGEST (abort largest eval).
     Threaded into `choose_tx_size_type_from_rd_intra`'s largest-depth walk via
     `NnDepthPruneCtx` (needs `TxfmYrdEnv::qindex`, new field). **Differential:
     `intra_tx_nn_diff` — 4000/4000 randomized decisions identical to the REAL
     `av1_nn_predict_c` (ref_nn_predict) on the same tables, all three verdicts
     exercised.** Byte-inert on the canon grid (no 8x8 leaf searches there — probes
     measured 0 calls); LIVE on the noise extension (96 Split verdicts, byte-exact at
     cq32/48).
  6. **Winner-mode restructure**: `multi_winner_mode_type = OFF` (:561) —
     `store_winner_mode_stats` returns immediately (rdopt_utils.h:688; count-1 arm in
     intra_rd.rs) and the re-eval runs ONCE on `best_mbmi` (C's else-arm,
     intra_mode_search.c:1727-1737 — including a filter-intra winner);
     `prune_winner_mode_eval_level = 1` (:562) — `bypass_winner_mode_processing`
     skips the re-eval when `source_variance < 64 - 48*qindex/256`.
  7. **Chroma narrowing**: `cfl_search_range = 1` (:532 — est-only CfL refinement +
     the range-1 invalid/overhead early-outs; machinery pre-existed in intra_uv_rd,
     now threaded via UvLoopPolicy). Bisect: flips 8 canon cells (all 4:2:0 gradient —
     vgrad/diag 64²+128² cq12-48). `prune_smooth_intra_mode_for_chroma` (:528 — prune
     UV_SMOOTH when BOTH chroma planes' per-pixel source variance < 20,
     intra_mode_search.c:850) — consumer wired (pre-existed), currently UNREACHED on
     all grids (the speed>=4 luma-winner mask only admits UV_SMOOTH when the luma
     winner is SMOOTH-family; carried transcription-faithful).
  8. **rd tables**: `perform_coeff_opt = 6` (:555, columns {432,97}/{86,16}) and
     `tx_domain_dist_level = 3` (:556 — types row {2,2,2}: the WINNER pass moves to
     tx-domain distortion); `winner_mode_tx_type_pruning = 3` + `prune_tx_type_est_rd
     = 0` (:551-552 — the est-rd prune turns OFF again; the PRUNE_5/PRUNE_2 stage rows
     are carried but inert on intra with est-rd off).
- **Verified INERT on the allintra KEY envelope:** `mv_sf.use_bsize_dependent_search_
  method = 3` (:548, motion) and `intrabc_search_level = 1` (:549, screen-only intrabc);
  `cdef_pick_method = CDEF_FAST_SEARCH_LVL4` (:558, CDEF off); qindex-dep
  `rect_partition_eval_thresh` (boosted-gated, KEY is boosted); the qindex-dep
  speed>=5 screen sub-8x8 re-zero (screen arm). `chroma_intra_pruning_with_hog = 4`
  (:530) is still zeroed by the :608-616 tail (chroma HOG stays OFF at speed>=4).
- **PINNED OPEN (near-tie class, KB-2 family):** `noise 64² cq63` (mono + 420) on the
  NEW `encoder_gate_speed6_noise_flatuv_allintra` extension diverges — localized to
  the (mi 8,0) 32×32 leaf's WINNER-pass tx-size sweep picking TX_16X16 over TX_32X32
  by a 0.19% rd margin where real keeps 32 (the search partition tree matches real
  EXACTLY; the frame codes tx_mode LARGEST post-hoc so the tx-plan difference desyncs
  the parse). Not closed by any single-feature revert (NN / predict_dc / rd tables /
  winner arm / intra-loop / partition prunes — each still diverges) ⇒ multi-feature
  interaction at qindex 255. The test asserts the divergence PRESENT (fails on match →
  promote). Next step: sibling-C RD dump of the winner sweep at (8,0). cq32/48 noise
  cells (mono+420) are hard-asserted byte-match.
- **Unit locks:** `speed6_allintra_deltas_match_source` (the full sf-block field set +
  stage policies incl. predict_dc columns + the speed-5 regression guard);
  `store_winner_mode_stats_matches_c_semantics` (the OFF count-1 no-store arm);
  `intra_tx_nn_diff` (REAL-C NN differential). The harness `max_partition_size` is now
  sf-driven (`min(default_max, CLI cap, SB)` — BLOCK_64X64 through speed 5, unchanged
  consumer outcomes; BLOCK_32X32 at 6).

### KB-11 — Encoder: `--cpu-used=7` speed-7 VAR_BASED_PARTITION — PORTED ✅ (64/64 canon; the KB-10-twin near-tie pinned open on the noise extension)
- **Status (2026-07-17): the canon gate is 64/64 byte-identical** vs real aomenc
  (`encoder_gate_speed7_textured_allintra`, {64,128}² × cq{12,32,48,63} ×
  {flat,two-tone,vgrad,diag} × {mono,420}) + the anti-vacuous witness
  (`encoder_gate_speed7_vs_speed6_sf_witness`: port with FULL speed-6 features vs
  `aomenc --cpu-used=7` DIVERGES on `vgrad 64² cq32` mono+420; speed-7 features match) +
  the deep-tree noise extension (`encoder_gate_speed7_noise_flatuv_allintra`, cq12/32/48
  hard-asserted). Speed 7 is STRUCTURALLY NEW (speed_features.c:569-575) — the partition
  tree is FIXED up front from variance thresholds, no RD partition search:
  1. **`av1_choose_var_based_partitioning` KEY arm** (`var_part.rs::
     choose_var_based_partitioning_key`): 4x4-downsampled variance tree
     (`fill_variance_4x4avg`: `aom_avg_4x4(src) − 128` per 4x4 — [`avg_4x4`]
     differentially locked 4000/4000 vs the REAL exported `aom_avg_4x4_c`,
     `avg_4x4_diff.rs`); `set_vbp_thresholds_key_frame` (`threshold_base = 120 *
     av1_ac_quant_QTX(qindex, 0, bd)`; <720p: t[2]=base/3, t[3]=base>>1; t[0]=t[1]=base;
     t[4]=base<<2); stage-2 force-split (16x16 var > t[3] and 32x32 var > t[2] propagate
     ONLY_SPLIT up; 64x64/128x128 have no key forcing rules but `set_vt_partitioning`'s
     `bsize > BLOCK_32X32 → split` KEY rule caps NONE at 32x32); the assignment descent
     with the sb64 boundary half-fit extensions (`bs_width_check = (w>>1)+1` at the frame
     edge) and edge-fit VERT/HORZ pair stamps; leaves stamped as mi-grid bsizes at block
     top-lefts (`set_block_size`), read back by `get_partition_from_stamps` (= C's
     `get_partition`, av1_common_int.h:1775, ext-partition disambiguation included).
     NOTE: interior rect stamps are reachable only on exact `variance == threshold` ties
     (stage-2 forcing fires strictly-above, NONE strictly-below) — the rect arms' real
     purpose is frame-edge blocks (unit-locked on a 48x48 frame).
  2. **`av1_rd_use_partition`** (`partition_pick.rs::rd_use_partition_real`,
     partition_search.c:1764): the fixed-tree walk running the EXISTING full-RD
     `leaf_pick_sb_modes` per leaf (`use_nonrd_pick_mode` stays 0 until speed 8) with
     C's exact context shape — HORZ/VERT strip-0-then-encode-then-strip-1
     (`encode_b_intra_dry` mid-stage propagation, the rect stage's own pattern), SPLIT
     recursion with `do_recon = i != 3` (last child skips its re-encode), per-node
     save/restore + `if (do_recon) encode_sb` (OUTPUT_ENABLED at the SB root, DRY below).
     Leaf budgets are `invalid_rdc` (INT64_MAX — no early-outs on a fixed tree).
     **Structurally DEAD at allintra speed 7 (verified, documented in the fn docs, NOT
     ported):** the PARTITION_NONE re-eval (:1827) + split-of-NONEs re-eval (:1986) —
     both need `adjust_var_based_rd_partitioning` ∈ {1,2}/{>2}, which is **0 outside
     REALTIME** (init :2288; setters :2002/:2896 are REALTIME-only) → the walk is a pure
     replay, its RD totals decision-inert; `setup_block_rdmult`'s ALLINTRA
     `intra_sb_rdmult_modifier` fold is IDENTITY (only av1_rd_pick_partition's root
     recomputes the modifier, partition_search.c:5715 — the VBP path leaves the per-SB
     reset 128, encodeframe.c:1303) → **`pack_tile` skips the SB rdmult fold at
     speed >= 7** (byte-visible: the fold is live at speeds 0-6).
  3. **sf deltas** (`speed7_allintra_deltas_match_source`): `partition_search_type =
     VAR_BASED_PARTITION` (:571; pack_tile derives the branch inline as `allintra &&
     speed >= 7` per the established pattern) + `default_min_partition_size = BLOCK_8X8`
     (:570 — assertion-only: the KEY tree never stamps below 8x8; the RD-search max/min
     clamps never run on this path). INERT (verified vs source): `cdef_pick_method =
     CDEF_PICK_FROM_Q` (:572, CDEF off in allintra), `rt_sf.mode_search_skip_flags |=
     FLAG_SKIP_INTRA_DIRMISMATCH` (:573 — sole consumer `search_intra_modes_in_interframe`,
     rdopt.c:5824, inter frames only), `rt_sf.var_part_split_threshold_shift = 7` (:574 —
     `set_vbp_thresholds_key_frame` reads it ONLY under
     `rt_sf.force_large_partition_blocks_intra`, which is 0 below speed 8/720p+
     [speed_features.c:327] and in this envelope; carried as a field for provenance).
     Everything else carries the speed-6 set unchanged (incl. LPF_PICK_FROM_Q).
- **PINNED OPEN — the KB-10 near-tie TWIN:** `noise 64² cq63` (mono + 420) on
  `encoder_gate_speed7_noise_flatuv_allintra` diverges — localized by
  `kb11_speed7_noise_localize.rs` (decode-both + the search's intended-winner dump):
  decoded partition trees IDENTICAL (SPLIT + four NONE-32s — the variance tree fixes the
  same shape real uses), every decoded mode record matches, and the port's (mi 8,0) leaf
  carries **tx_size TX_16X16 where real keeps TX_32X32** — exactly KB-10's "(mi 8,0)
  32×32 WINNER-pass uniform tx-size sweep picks TX_16X16 over TX_32X32 by 0.19%" cell
  (same leaf machinery at speeds 6 and 7; the tx-plan difference desyncs the LARGEST-tx
  parse — the decoded eob-50 / 420 "(8,8) tree diff" are desync artifacts). Both tests
  assert the divergence PRESENT (fail on match → promote). Next step: KB-10's sibling-C
  RD dump of the winner sweep at (8,0) — closes both speeds' cells at once.
- **Unit locks:** `speed7_allintra_deltas_match_source` (+ speed-6 regression guard);
  `avg_4x4_diff` (REAL-C kernel differential); var_part.rs threshold/shape/edge tests;
  `kb11_speed7_noise_cq32_control_matches` (the localization harness's own soundness).
- **Speed-8 prep facts (KB-12 seed, verified against source 2026-07-17):** speed 8 flips
  `use_nonrd_pick_mode = 1` (speed_features.c:578) — the nonrd PICKMODE, the big one:
  - `encode_nonrd_sb` (encodeframe.c:581-663): the SAME
    `av1_choose_var_based_partitioning` (KEY arm already ported, var_part.rs) fixes the
    tree, then **`av1_nonrd_use_partition`** (partition_search.c:2960) — a SINGLE-PASS
    walk: per leaf `pick_sb_modes_nonrd` + `encode_b_nonrd` IMMEDIATELY (dry_run=0 — the
    encode IS the output; NO save/restore, NO mid-strip re-encode, NO root winner walk;
    `set_mode_eval_params(DEFAULT_EVAL)` per node). HORZ/VERT strips: pick+encode strip 0
    then strip 1 (in-frame gated, `bsize > BLOCK_8X8` for strip 1). SPLIT: plain
    recursion. `try_merge` (`nonrd_check_partition_merge_mode = 1`, :580) is
    `!frame_is_intra_only`-gated → INERT on KEY; `nonrd_check_partition_split` stays 0;
    `direct_partition_merging` is `!frame_is_intra_only` too.
  - The KEY-intra leaf search is `hybrid_intra_mode_search` (partition_search.c:756):
    `hybrid_intra_pickmode = 2` (:579) → full-RD `av1_rd_pick_intra_mode_sb` (the
    EXISTING ported search) for `bsize < BLOCK_16X16 && x->source_variance >=
    var_thresh[1] = 101`; else **`av1_nonrd_pick_intra_mode`** (nonrd_pickmode.c:1582) —
    NEW machinery: `intra_mode_list` loop (RTC_INTRA_MODES = DC/V/H/SMOOTH) with
    `intra_y_mode_bsize_mask_nrd` (:583-590: INTRA_DC only >= BLOCK_32X32, INTRA_DC_H_V
    below — mask consumed where? verify: the mask gates the loop in nonrd inter path;
    the intra-frame fn loops intra_mode_list directly), per-mode
    `av1_estimate_block_intra` (foreach-txb SATD/model estimate, not full RD),
    skip_txfm-cost fold + `bmode_costs[y_mode_costs[above_ctx][left_ctx]]`, tx_size =
    min(max_txsize_lookup, biggest for tx_mode) — NO tx search, NO angle deltas, NO
    filter-intra. Palette arm gated `enable_palette && allow_screen_content_tools`
    (`prune_palette_search_nonrd = 1`, :582). CHROMA on the nonrd KEY path: locate it
    (av1_nonrd_pick_intra_mode is PLANE_Y only — uv likely inside encode_b_nonrd's
    encode_superblock or a uv estimate step; UNRESOLVED, first thing to trace).
  - `x->source_variance` IS live at speed 8: choose_var computes it per SB
    (var_based_part.c:1728, `use_nonrd_pick_mode && source_sad_nonrd > kLowSad`;
    content_state_sb.source_sad_nonrd inits kMedSad per SB, encodeframe.c:1289) — but
    verify what `pick_sb_modes_nonrd` re-derives per LEAF before trusting the SB value
    in hybrid_intra's threshold.
  - `var_part_split_threshold_shift = 8` (:581): STILL force_large-gated on KEY
    (`force_large_partition_blocks_intra` rises only at speed>=8 AND 720p+ —
    speed_features.c:326-328) → inert on sub-720p grids; LIVE at 720p+ (the
    `set_vbp_thresholds_key_frame` shift-steps arm + thresholds[2]/[3] shift_val 1 —
    port the arm when a 720p+ speed-8 cell lands).
  - `encode_b_nonrd` (partition_search.c:2100): the single-pass leaf encode
    (av1_update_state-equivalent + encode_superblock with dry_run=0 → tokens + cdf
    updates inline as it walks — the pack IS the walk; the port's search/pack split
    needs rethinking for this path, or model it as search==pack in one pass).

### KB-P29 — Encoder: palette 128² AB/4-way partition near-tie (2 cells) — PINNED (genuine; palette machinery C-faithful)
- **Status (2026-07-17 pickup):** the palette-Y+UV RD search (#29, PARITY Section B) is 5/7
  byte-exact — those 5 cells are now HARD byte-identity asserts in
  `rd_close_palette::palette_y_rd_close_gate`. The 2 remaining CLOSE cells (`ui_420_128_cq32`,
  `text_420_128_cq20`, both 128² 4:2:0) are PINNED as genuine palette-induced AB/4-way partition
  RD near-ties, NOT a palette-cost bug.
- **Decode-both localized** (`decode_diff_palette_close_cells`, the regression guard):
  `ui` diverges at (mi 0,0) BLOCK_32X32 real PARTITION_HORZ_B vs port PARTITION_HORZ_4;
  `text` at (mi 8,20) BLOCK_16X16 real PARTITION_VERT vs port PARTITION_VERT_A.
- **Both cells are BYTE-EXACT with palette OFF** (the localizer's palette-OFF control proves it),
  so partition/mode/tx are correct; the palette contribution alone tips the AB/4-way tie. The
  palette machinery is verified C-faithful: `av1_allow_palette`, `av1_get_palette_bsize_ctx`
  (`num_pels_log2[bsize] − num_pels_log2[8X8]`), `av1_get_palette_mode_ctx` (above+left palette
  count), k-means (rtcd-validated), and mid-search neighbour palette cache/ctx stamping (the
  winner's palette is threaded at every `grid.stamp`) all match C; the byte-exact 64² palette
  cells exercise the same non-square block sizes. Same class as the KB-10/KB-11 pinned near-ties.
- **Next step (deferred, the close move):** sibling-C per-candidate partition-RD dump at the two
  divergent nodes (the KB-2/3/7 method) — compare C's HORZ_B-vs-HORZ_4 / VERT-vs-VERT_A RD with
  palette ON to find whether a specific leaf's palette RD/flag (or a pruning gate) tips it. The
  localizer asserts the divergence PRESENT, so any fix self-promotes the cell into
  `BYTE_EXACT_CELLS`.
### KB-12 — Encoder: `--cpu-used=8/9` nonrd PICKMODE — PORTED ✅ (speed-9 64/64 canon + noise; speed-8 60/64 canon, 4 diag estimate-arm near-ties pinned open + noise 8/8) — GATE 2 (cpu 0-9) COMPLETE
- **Status (2026-07-17): speed 8 AND speed 9 land, Gate-2 (cpu-used 0..9) is byte-complete
  except 4 pinned speed-8 near-ties.** The nonrd PICKMODE (`use_nonrd_pick_mode = 1`,
  speed_features.c:578): the SAME `av1_choose_var_based_partitioning` KEY tree the speed-7
  gate fixes now drives **`av1_nonrd_use_partition`** (partition_pick.rs `nonrd_use_partition_
  real`) — a SINGLE-PASS walk (NO save/restore, NO mid-strip re-encode, NO root winner walk):
  per leaf `hybrid_intra_mode_search` then `encode_b_intra_dry(output_enabled=true)` (C's
  `encode_b_nonrd`, dry_run=0) immediately; bits via the unchanged `pack_sb` re-walk (same
  search==pack split proven for speeds 0-7). `try_merge`/`direct_partition_merging` are
  `!frame_is_intra_only`-gated → KEY-dead (not modelled).
  - **Leaf search — `hybrid_intra_mode_search`** (partition_search.c:756): `hybrid_intra_
    pickmode = 2` at speed 8 → full-RD `av1_rd_pick_intra_mode_sb` (the EXISTING
    `leaf_pick_sb_modes`) for `bsize < BLOCK_16X16 && source_variance >= var_thresh[1] = 101`,
    else the ESTIMATE arm `av1_nonrd_pick_intra_mode` (nonrd_pickmode.rs, NEW +880): the
    DC/V/H/SMOOTH `intra_mode_list` loop, per-mode `av1_estimate_block_intra` = one txb
    `av1_block_yrd` (LP Hadamard SATD estimate: `hadamard_lp_8x8/16x16` + `quantize_lp` +
    `satd_lp` + `block_error_lp`, all `wrapping`-i16, over the `*_lp_*_transpose` scans),
    skip-cost fold + `bmode_costs`, tx_size = max-square, NO tx search / angle delta /
    filter-intra. Speed 9: `hybrid_intra_pickmode = 0` → EVERY leaf is the estimate arm, plus
    the three estimate-loop prunes (`prune_h_pred_using_best_mode_so_far`, `enable_intra_mode_
    pruning_using_neighbors`, `prune_intra_mode_using_best_sad_so_far`) and `INTERNAL_COST_
    UPD_OFF` (<4k → every SB reads the FRAME-INIT cost tables; `sb_real` becomes an `Option`
    in `pack_tile`/`pack_tile_from_trees`, byte-visible on 128² multi-SB cells).
  - **The nonrd CHROMA path — RESOLVED (the KB-11 flagged unknown):** `av1_nonrd_pick_intra_
    mode` is PLANE_Y only and hard-sets `mi->uv_mode = UV_DC_PRED` (nonrd_pickmode.c:1735, "Keep
    DC for UV since mode test is based on Y channel only"). Estimate leaves code chroma as DC
    via the ordinary leaf encode (`LeafWinner{uv_mode:0}`); CfL never a candidate; full-RD leaves
    keep the existing uv search. Confirmed byte-exact by the mono+420 gate agreement.
  - **`output_enabled = true` (the KEY correctness item, KB-4 class):** C's nonrd walk encodes
    every leaf dry_run=0 (OUTPUT_ENABLED) → tx_type_map COPY semantics (eob-0 → DCT_DCT resets
    go to a transient frame map, the search winner's `w.tx_type_map` survives to `pack_sb`).
    `false` (alias) would re-introduce the KB-4 reset-leak on the full-RD arm. Matches the
    speed-7 SB-root walk (`output_enabled = bsize == sb_size`) + the pack walk (pack.rs:450).
  - **The 2 salvage blockers (fixed):** (1) `pack.rs` `sb_pick_cfg` dangled on the `Option<sb_
    real>` after the cost-upd-off refactor → `match &sb_real { Some => build; None => *pick_cfg }`
    (frame-init fallback = INTERNAL_COST_UPD_OFF); (2) `nonrd_use_partition_real` was DISPATCHED
    FROM NOWHERE → wired the `allintra && speed >= 8` branch into `pack_tile` (mirrors the
    speed-7 VBP dispatch; `speed >= 9` toggles the vbp 16×16 min/max-sub-var split prune, inert
    <720p). Plus the mechanical arity fixes the concurrent palette work introduced
    (`ModeGrid::stamp` + `LeafWinner` gained palette params/fields).
- **Gates (encoder_gate_e2e_byte_match.rs):** `encoder_gate_speed9_textured_allintra` **64/64**
  + `encoder_gate_speed9_noise_flatuv_allintra` **8/8** (cq12/32/48/63) + `encoder_gate_speed9_
  vs_speed8_sf_witness`; `encoder_gate_speed8_textured_allintra` **60/64** + `encoder_gate_
  speed8_noise_flatuv_allintra` **8/8** + `encoder_gate_speed8_vs_speed7_sf_witness`. Speeds
  0-7 re-verified byte-unchanged (full `cargo test -p aom-encode` green). NOTE: the KB-10/KB-11
  noise-cq63 (mi 8,0) TX_16X16-vs-TX_32X32 near-tie does NOT reproduce at speed 8/9 — the
  estimate arm codes tx_size = max-square directly (no winner-pass tx sweep to flip), so the
  speed-8/9 noise cq63 cells byte-match (unlike speeds 6/7).
- **PINNED OPEN (KB-2/KB-10/KB-11 near-tie family) — 4 speed-8 diag cells:** `diag {64² cq12,
  128² cq32}` × {mono,420}. Localized (decode-both, the `kb11_speed7_noise_localize.rs` shape)
  to ONE BLOCK_8X8 **estimate-arm** leaf (mi 2,2 on the 64² cells): partition trees IDENTICAL,
  every earlier leaf byte-matches, but `av1_nonrd_pick_intra_mode` picks **V_PRED** where real
  codes **H_PRED** — a directional near-tie at ~0.7 % rdcost (V 624968 vs H 629535; identical
  dist 563, bmode_costs V 1596 / H 1385, src_var 24 → estimate arm both). The entire traced
  estimate chain (the LP kernels + `quantize_lp` + `block_yrd` structure + the mode loop) matches
  libaom line-for-line, and speed 9's mode prunes mask it (same cells 64/64 there); only speed
  8's unpruned mixed hybrid exposes the V/H sign. **Next step:** sibling-C per-mode `this_rdc`
  dump at that leaf (the KB-10/KB-11 method) to find which of V/H's rate the port tips —
  everything readable already agrees, so the tip is sub-trace.
- **HBD (bd10/12) estimate arm + lossless TX_4X4 + palette (screen) arms NOT ported** — asserted
  dead on the 8-bit canon grid (nonrd_pickmode.rs:594/460/784); required before any high-bit-depth
  or screen-content speed-8/9 cell.

### KB-13 — Encoder: REAL-content byte-parity at speed >= 1 — MAPPED (24/60 byte-exact at cpu 1..4; report-only gate landed)
- **Status (2026-07-17):** the encoder byte-matches real conformance-decoded content at speed 0
  (KB-6, 30/30) but NOT uniformly at speed >= 1. New gate
  `encoder_gate_real_content_speed1to4_e2e` (encoder_gate_e2e_byte_match.rs) feeds the SAME real
  YUV KB-6 uses (5 cells: `01-size-64x64` full, `01-size-196x196` full, `00-quantizer-00` 64²+128²
  crops, `23-film_grain-50` 64² crop) through the PROVEN speed-N encode path
  (`attempt_case_content_uv_sep` — the exact fn every synthetic speed gate rides, generalized to
  SEPARATE U/V closures for real content) at cpu-used {1,2,3,4} × cq {12,32,63}.
- **MAP (24/60 byte-exact vs real aomenc):**
  - `01-size-64x64` (1-SB, aligned): **12/12 MATCH** — the clean single-SB real cell is byte-exact
    at every speed 1..4 × cq. The speed-0 roots (KB-6 luma edge filter, AB HORZ_A reuse,
    OUTPUT_ENABLED tx_type_map, …) carry through to speed 1..4 unbroken here.
  - `01-size-196x196` (PARTIAL-SB, multi-SB): **0/12** — every cell diverges. `our_tile_bytes` is
    systematically SMALLER than real (e.g. cq12 cpu1 2200 vs 2379) — the KB-6 "port codes FEWER
    symbols" signature ⇒ a TILE-DATA partition/mode near-tie flip, NOT a pure-LF issue (LF
    DISAGREES because the recon differs downstream; first byte-diff is the port-derived header LF
    byte 2-3). The partial-SB frame-edge path (KB-6's speed-0 CHUNK-series fixes) interacts with
    speed>=1 partition features (rect-kill / part4 / AB / ext-thresh) unreconciled at edges.
    HIGHEST-VALUE cluster (12 cells, likely few shared roots).
  - `00-quantizer-00` 64² crop (photo, 1-SB): 7/12 — DIFF at {cpu1 cq12, cpu2 cq12/cq32, cpu3 cq32,
    cpu4 cq32}. Interior partition/mode near-ties (KB-2/10/11/12 family).
  - `00-quantizer-00` 128² crop (photo, 4-SB): 3/12 — mostly DIFF (cq63 matches at cpu1/2).
  - `23-film_grain-50` 64² crop (1-SB): 4/12 — film-grain near-ties.
- **Gate = GRADUATE + PIN** (report-only on the first landing, then converted): asserts (1) the
  anti-vacuous speed-0 harness control (real 64² cq20 must byte-match — KB-6 30/30); (2) the **24
  MATCH cells** byte-match (regression guard); (3) the **36 DIFF cells** DIVERGE (self-promoting
  pin, KB-P29/KB-12 pattern — a DIFF→MATCH flip fails the pin → promote into `byte_exact`). Harness
  change: `attempt_case_content_uv` now delegates to `attempt_case_content_uv_sep` (separate U/V) —
  byte-identical for every existing synthetic caller (U==V), so speeds 0-9 synthetic gates + KB-6
  speed-0 map stay byte-unchanged (verified: speed1 + speed4 420 gates still 64/64).
- **LOCALIZED (2026-07-17, decode-both via `localize_real_speed`, kb6_real_rd_localize.rs):** EVERY
  DIFF is a PARTITION near-tie at **BLOCK_16X16 (bsize=6)** or BLOCK_8X8, NOT a frame-edge/partial-SB
  issue (the 196 first-divergences are all INTERIOR nodes inside SB(0,0), mi_col 6..12, never the
  mi_col=48 edge). The port systematically picks a MORE-COMPLEX partition than C — mostly an **AB
  partition** (HORZ_A/HORZ_B/VERT_A) or SPLIT — where C picks a simpler one:
  - 196 cpu1 cq32 (0,12): real VERT_B / ours HORZ_B; 196 cpu1 cq12 (0,12): real SPLIT / ours HORZ_B;
    196 cpu4 cq32 (12,8): real VERT / ours SPLIT.
  - quant128 cpu1 cq32 (0,8): real VERT / ours VERT_A; quant128 cpu1 cq12 (4,12): real HORZ_B / ours SPLIT.
  - quant64 cpu1 cq12 (0,12): real HORZ / ours HORZ_A; quant64 cpu2 cq32 (4,8): real SPLIT / ours HORZ_B.
  - film64 cpu1 cq12 (4,6) 8X8: real NONE / ours VERT.
  The port's chosen partition codes FEWER total bytes (map signature) yet C picks a different one ⇒
  **C PRUNES the port's cheaper candidate at speed>=1** (KB-3/KB-7 pattern: a partition C's search
  never evaluates because a speed-feature prune killed it; the port under-prunes). Consumer:
  `prune_ab_partitions` / `predict_ab_partition_prune` (ab_nn_prune.rs) + the rect/split prunes.
  Shared-root candidate (AB / rect / split pruning at speed 1..4), NOT scattered RD noise.
- **NOT the KB-15 root (checked 2026-07-19, do not conflate).** KB-15's "intra RD is slightly
  cheap" residual root-caused to a MISSING `intrabc_cost[0]` on every intra leaf
  (`intra_mode_info_cost_y`, intra_mode_search_utils.h:563-564 — see KB-15's four-roots block).
  That term is gated on `av1_allow_intrabc(cm)`, which is **0 for every KB-13 cell** (decoded
  photographic content ⇒ `allow_screen_content_tools = 0`), so it is structurally inert here.
  Empirical confirmation: all four KB-15 fixes left this gate's 36 DIFF pins asserting
  divergence and its 24 MATCH cells byte-exact (suite 340/340 unchanged). KB-13's own
  hypothesis below still stands.
- **Tracking:** task **#39**. Next: sibling-C RD/prune dump at a representative 16X16 node (the
  KB-3/KB-7 method) to find which prune C applies that the port doesn't; graduate the cells it closes.

### KB-14 — Decoder: superres single-SB-column coded frame decoded flat — FIXED ✅ (header coded-lossless false-positive; NOT the upscale)
- **FIXED 2026-07-18.** Root cause is the HEADER PARSE, **not** the normative upscale (the original
  "likely locus: `upscale_plane`/`get_upscale_convolve_x0`" guess was WRONG — `superres.rs` is
  byte-faithful to `av1_convolve_horiz_rs_c`; the coded recon fed to it was already flat).
- **Symptom (found 2026-07-18 via the ENCODER superres QTHRESH gate):** a superres KEY stream whose
  CODED frame is **a single 64-wide superblock** (`coded_w <= 64`, one SB column, recon stride 64 —
  first reached at denom 16 / exact-2:1, e.g. 128→64) decoded to a **flat-128 luma plane** (the
  whole first SB row went to the DC no-neighbour default; content cascaded wrong below it). The
  encoder was byte-exact (`port_tu == c_tu`); purely decoder-side.
- **Root cause (decode-both localized):** the two-phase header parse in `parse_frame_header`
  (`aom-decode/src/frame.rs`). The PROBE parses on the full (upscaled-width) mi grid; under superres
  its `tile_info` is OVER-SIZED, so every field the probe reads after it — the quant params included
  — comes off a shifted bit position and is garbage. `coded_lossless` was computed from that garbage
  probe (`frame_coded_lossless(&probe)`), and for these streams it **FALSE-POSITIVED** (a normal
  qindex-128 frame read as lossless). The superres re-parse then set `cfg2.coded_lossless = true`,
  which **drops the loop-filter + CDEF header sections** (`read_uncompressed_header`'s
  `if !coded_lossless` gate), so the header ended ~2 bytes early → `tile_data_off` off by 2 → the
  tile arithmetic decoder started on the wrong bytes → SB(0,0)'s BLOCK_64X64/TX_64X64 read
  `EOB=1`/skip where C reads real coeffs → flat-128. Why only `coded_w<=64`: only there does the
  probe's sb_cols (upscaled) differ from the coded sb_cols such that the misread quant lands on a
  lossless-looking pattern (denoms 9-15 and wider coded frames keep sb_cols matching, so the probe's
  quant was accidentally still right; single-SB-column plain frames re-parse identically and always
  worked). CONTENT/qindex-dependent because whether the garbage looks lossless depends on the exact
  bits.
- **Fix:** derive `coded_lossless` from quant read with the CORRECT (downscaled) mi grid. When
  superres is active, parse a probe on the downscaled `tile_info` FIRST and decide lossless from IT
  (reused as the final header unless it really is coded-lossless). Handles all four cases (±superres
  × ±lossless); non-superres path unchanged. `frame.rs` `parse_frame_header` (~L490-528).
- **Regression gate (Gate-1, asserted, NO graceful skip):**
  `superres_denom16_single_sb_column_byte_identical_kb14` in `aom-decode/tests/superres_diff.rs` — 5
  single-SB-column upscaled widths (128/126/120/116/100 → coded 64/63/60/58/50) at denom 16, each
  decoded byte-identical to the C decoder + a golden per-plane MD5 (shared `tests/common/md5.rs`) +
  an explicit "flat-128" guard. **Proven revert-catching:** stashing only the `frame.rs` fix flips
  every cell to the flat-128 mismatch. The existing `superres_diff` denoms 9/12/16 arms + the full
  Gate-1 conformance corpus + `real_bitstream` (incl. sb128/multi-tile) stay green.

### KB-15 — Encoder: IntraBC (screen content) — SEARCH + skip-arm + wiring LANDED; PINNED on the inter var-tx COEFF ARM
- **Status (2026-07-18):** `rd_pick_intrabc_mode_sb` (`aom-encode/src/intrabc_search.rs`) is WIRED
  (rd_pick.rs step 6 → real, `PickFrameCfg::intrabc` gated on `p.allow_intrabc`) and runs the full
  DV search under the screen-content gate. LANDED: the source-frame hash (chunk 3a) + the **NSTEP
  `full_pixel_diamond`** (site config + `diamond_search_sad` coarse→fine + `UPDATE_SEARCH_STEP`
  num00 collapse) + the **`full_pixel_exhaustive` mesh** (screen `exhaustive_searches_thresh=1<<20`);
  the pixel search ALWAYS runs at `intrabc_search_level 0` (rdopt.c:3570). Geometry unit-locked
  (`nstep_config_matches_c`, `mv_step_param_matches_c`, `diamond_finds_exact_repeat`). The HASH is
  square-gated (mcomp.c:1918); the diamond runs for EVERY bsize (non-square intrabc supported).
  `predict_skip_txfm` (tx_search.c:183) + `set_skip_txfm` hbd sse: the port offers intrabc ONLY in
  the skip regime (luma predict_skip fires AND chroma exact match), where `av1_txfm_search` forces
  `skip_txfm=1` and BYPASSES the coeff arm → the skip RD (`mode+mv+skip1`, `dist=sse`) is byte-exact.
  Full wiring: `LeafWinner`/`RdPickIntraBest` fields, `ModeGrid` DV grid (`dc_screen`, 25 stamp
  sites), `encode_b_intra_dry` intrabc arm (predict-from-recon + skip entropy reset + skip txfm ctx),
  `pack_leaf` (use_intrabc + DV diff, skip tx/coeff), harness (hash from source luma, LF forced 0,
  `ToggleKnobs::enable_intrabc`, `PackCfg::allow_intrabc`).
- **[SUPERSEDED 2026-07-19 — historical, as of 2026-07-18. Every "NOT ported" item below has
  since LANDED; see the three PROGRESS 2026-07-19 entries and the DV-clamp fix (`434b865d`).
  Kept for the measured census only. Do NOT read its remaining-work list as current.]**
  **PINNED — real-content byte-exactness blocked on the inter var-tx COEFF ARM (the L piece).**
  Real screen content codes the MAJORITY of intrabc blocks via the COEFF arm (nonzero quantized
  residual) and as NON-SQUARE shapes: measured on a 196² conformance crop (`intra_only-intrabc-
  extreme-dv` @ (480,180) cq48), C uses **49 intrabc blocks = 39 coeff-arm + 42 non-square, only 10
  skip**. The port codes those blocks as intra and the frame diverges. NOT ported (the remaining
  work): `av1_pick_recursive_tx_size_type_yrd` inter var-tx quadtree (`select_tx_size_and_type` /
  `select_tx_block` / `try_tx_block_no_split`/`_split`) + `prune_tx_2D` + `ml_predict_tx_split` NN
  prunes + the var-tx WRITE path in pack (`write_tx_size_vartx` exists in aom-entropy — post-reorg
  that is `crates/aom-dsp/src/entropy/partition.rs` — unused by
  pack) + `derive_real_costs` inter tx-cost fill (currently a DUMMY zero cdf; source from
  `kf.inter_ext_tx`). The 420 skip subset ALSO needs a chroma-eob-0 check (currently `chroma_sse==0`,
  exact-only). No synthetic content found that makes aomenc use ONLY skip-square intrabc (intra wins
  on synthetic repeats; real content uses the coeff arm) — so the skip-arm alone byte-matches 0 real
  cells today.
- **Gate:** `rd_close_intrabc::intrabc_dv_search_pinned` (aom-bench) — asserts anti-vacuous (C
  genuinely codes intrabc on the crop, via decode census) + PINS the divergence self-promotingly (a
  byte-match fails → promote into `BYTE_EXACT_CELLS`). Envelope UNTOUCHED: non-screen frames
  (`allow_intrabc=0`) are byte-inert — palette gate + `partition_pick_diff` + `rd_pick_intra_sb_diff`
  all green.
- **PROGRESS 2026-07-19 (inter var-tx coeff arm — the CORE is differential-locked; witness still
  PINNED):** landed on origin/main:
  - `44bc51c` — `derive_real_costs` inter ext-tx costs sourced from `kf.inter_ext_tx` (§5 #C
    one-liner; was a zero stub).
  - `db90148` — `crates/aom-encode/src/var_tx.rs`: the inter/intrabc var-tx recursion
    (`pick_recursive_tx_size_type_yrd` → `select_tx_size_and_type` → `select_tx_block` /
    `try_tx_block_no_split` / `try_tx_block_split`) + the inter per-txb leaf
    (`search_tx_type_inter` + `get_tx_mask_inter` + `trellis_rdmult_inter_y` mult-16). The inter
    LEAF is byte-locked vs the REAL C kernels (`var_tx_leaf_diff.rs`: fwd/quant/optimize/cost/
    dist hybrid + is_inter cost + adaptive break, all 19 tx sizes × amp/qidx/bd/reduced).
  - `3b9278f` — the RECURSION is byte-locked vs an independent C transcription
    (`var_tx_recursion_diff.rs`: no-split/split RD, pick-skip, txfm_partition cost, context
    threading + backtracking, adaptive_txb_search_level=1 + txb_split_cap=1, depth-2 splits).
- **PROGRESS 2026-07-19 cont. — BOTH NN prunes now differential-locked + ENABLED in the recursion:**
  - `a40d598` — `ml_predict_tx_split` NN (`ml_tx_split_thresh=8500`, bd8): weights transcribed
    (`xtask/transcribe_tx_split_nn.py` → `tx_split_nn_weights.rs`), `av1_nn_predict` eval +
    prec-reduce, wired into `select_tx_block`'s `try_split` gate (`VarTxEnv.ml_tx_split_thresh`).
    Differential-locked vs real C `av1_nn_predict` (`tx_split_nn_diff.rs`).
  - `5aa145d` + `a77a7d8` — `prune_tx_2D` NN (`TX_TYPE_PRUNE_1`): all 5 helpers
    (`get_energy_distribution_finer` / `av1_get_horver_correlation_full` / `av1_nn_fast_softmax_16`
    + `approx_exp` / `get_adaptive_thresholds` + table / `av1_sort_fi32_8/16`) + the driver ported
    (`prune_tx_2d.rs`), weights transcribed (`transcribe_prune_tx_2d_nn.py`). Differential-locked
    vs a tier-1 real-C shim (`shim_prune_tx_2D` copies the static helpers + driver verbatim,
    calls the exported scalar `av1_nn_predict_c` etc. + real nnconfig maps; `prune_tx_2d_diff.rs`,
    576 cases). WIRED into `search_tx_type_inter`'s multi-type arm (`VarTxEnv.prune_2d`, reorders
    `txk_map`). NOTE: the differential uses the SCALAR `_c` reference — the real encoder's SIMD
    `av1_nn_predict_avx2` gives ULP-different scores that flip near-tie sort ORDER (decision-inert
    for the RD pick over the same masked set) but not the decision-relevant mask.
- **PROGRESS 2026-07-19 cont. — the COEFF ARM IS WIRED END-TO-END; witness still PINNED (not
  byte-exact).** Landed on origin/main: `a33929ca` (winner carry: `inter_tx_size[16]` on
  LeafWinner/RdPickIntraBest + var-tx root tx_size and luma tx_type_map on IntrabcBest),
  `194ae39f` (the inter tx leaf parameterized for CHROMA — `get_tx_mask_inter_uv` pins the
  co-located luma tx type per tx_search.c:1841-1847, the tx-type-cost plane arg, and
  `trellis_rdmult_inter_uv` = `plane_rd_mult_chroma[inter][UV]` = 10 vs luma's 16),
  `9a5fafdc` (`txfm_uvrd_inter`: C's chroma inter arm — UNIFORM tx, EOB-based skip),
  `351143a1` (the integration: `av1_txfm_search` assembly + the `av1_encode_sb` var-tx re-encode
  + the `write_tx_size_vartx` / inter-ext-tx pack + `KfFrameContext::txfm_partition`).
  - **Item 1 (`model_based_tx_search_prune`) is CLOSED BY SOURCE PROOF — no port needed.**
    `rd_pick_intrabc_mode_sb` calls `av1_txfm_search` with `ref_best_rd = INT64_MAX`
    **hardcoded** (rdopt.c:3611 — the incoming `best_rd` is used only for the post-hoc
    `rd_stats_yuv.rdcost < best_rd` compare at :3615). The prune is gated on
    `ref_best_rd != INT64_MAX` (tx_search.c:3562-3565), so it can never fire on the intrabc path.
    The same reasoning voids every other early exit in `av1_txfm_search` (:3811, :3844) and
    `av1_txfm_uvrd` (:3737) for intrabc.
  - **Items 2 + 3 (PACK + INTRABC integration) are DONE and exercised**, including the chroma
    eob-0 skip check (the old `chroma_sse == 0` gate was a strict SUBSET of it).
  - **NOT DONE — byte-exactness.** `rd_close_intrabc::intrabc_dv_search_pinned` still PINS:
    port **1907B vs c 1891B (delta +16), first differing tile byte 646 of 1891**. The port now
    codes MORE bytes than C — the OPPOSITE of the classic "codes fewer symbols" RD-near-tie
    signature — and the first ~34% of the tile is byte-identical.
  - **A/B PROOF THE COEFF ARM IS LIVE (not inert):** re-running the witness with the coeff arm
    gated back off (`if !luma_skip { continue; }`, the pre-landing skip-only behaviour) gives
    **1974B, delta +83, first diff at byte 1113**; with the coeff arm on it is **1907B, delta
    +16, first diff at byte 646**. So the arm cuts the size delta ~5x (it codes residual instead
    of falling back to intra) but moves the first divergence EARLIER — i.e. a coeff-arm intrabc
    block BEFORE tile byte 646 is coded differently from C, while the later blocks the skip-only
    build got wrong are now right. That earlier first-divergence is the sharper localization
    target. The gate now prints that
    signature plus (when the port's stream still decodes) the first block whose mode-info differs;
    today the port's stream does NOT decode past the divergence (bitstream desync), so the
    block-level localization needs the C-side dump method (KB-2/KB-3/KB-7) or a
    truncate-at-first-divergence probe.
  - **Three real bugs found while wiring (all fixed in `351143a1`):** the pack's intrabc branch
    returned early and skipped the neighbour-grid stamp; chroma prediction buffers were sized
    `bw >> ss_x` while the chroma PLANE block is padded to a 4x4 minimum (sub-8x8 luma overran
    them); and the var-tx recursion sized its entropy/txfm context arrays by the frame-edge
    CLIPPED extent while `av1_get_entropy_contexts` fills the FULL plane block (only the txb LOOP
    is clipped) — under-running `get_txb_ctx`'s `a[..w_unit]` read on an edge block. All
    frame-edge extents now route through the validated `max_block_units`.
  - **Envelope verified byte-inert:** full `cargo test -p zenav1-aom-encode` green (97 binaries,
    0 failures) including `encoder_gate_real_image_e2e_kb6_repro` (30/30 real content); both
    var-tx differentials (`var_tx_leaf_diff`, `var_tx_recursion_diff`) still pass, plus
    `prune_tx_2d_diff` / `tx_split_nn_diff`. Non-screen frames construct no intrabc args at all.
  - **ROOT CAUSE #1 — CLOSED ✅ 2026-07-19: UNCLAMPED tile bounds fed into `av1_is_dv_valid`.**
    All THREE previously-ranked suspects are **REFUTED — do not re-chase**: it was NOT the joint
    skip decision / `set_skip_txfm` rate, NOT the chroma inter tx-type inheritance timing, and NOT
    the AVX2-vs-scalar `prune_tx_2D` sort order. The port's `is_dv_valid`
    (`aom-dsp/src/entropy/dv_ref.rs:1369`) is faithful line-for-line **and its constants are
    right** — the defect was in its INPUTS. `partition_pick.rs`'s `IntrabcLeafArgs` passed
    `env.tile_{row,col}_end` **raw**, and those are unclamped sentinels (`1 << 16`, set at
    `aom-bench/src/lib.rs:1381`). C's `av1_tile_set_row`/`av1_tile_set_col` (tile_common.c) CLAMP
    the tile end to the frame (`AOMMIN(row_start_sb[i+1] << mib_size_log2, mi_rows)`). That single
    input blows up `total_sb64_per_row = ((mi_col_end - mi_col_start - 1) >> 4) + 1` from **4** to
    **4096** on this 196² frame, which makes the already-coded-SB64 ordering gate
    `src_sb64 >= active_sb64 - INTRABC_DELAY_SB64` (mvref_common.h:328) stop rejecting anything —
    so the port accepted DVs C rejects and coded intrabc where C codes intra. **Measured at the
    first divergent block** (throwaway instrumented sibling C, byte-inertness re-verified against
    the clean build on every rebuild — the witness reported identical 1907B / first-diff-646 with
    and without the instrument): **mi(16,0) BLOCK_16X4 — C's own search finds the IDENTICAL
    `dv=(-512,0)` with `bestsme=35311`, `inrange=1`, but `dvvalid=0` → `continue`**, while the
    port had `dvvalid=1` with `tile[0,65536,0,65536]`. **Fix:** clamp to `env.mi_rows`/
    `env.mi_cols` at the `DvTileBounds` construction (`partition_pick.rs`) — the SAME clamp
    `pack.rs:1377` already applies to the decoder-facing bounds. `DvTileBounds` is intrabc-only
    (non-screen frames construct no intrabc args at all), so the envelope structurally cannot move.
    **Measured effect: size delta +16 → 0 (port 1891B == C 1891B), first differing byte 646 → 1038.**
  - **RESIDUAL (witness still PINNED) — a partition near-tie at mi(38,4). NOT a KB-15 intrabc
    defect.** At the BLOCK_16X8 node mi(38,4) C picks PARTITION_VERT_4 (four BLOCK_4X8 at mi cols
    4/5/6/7 — c=4 intra mode 10, **c=5 and c=6 themselves intrabc `skip=1 dv=(-1032,-40)`**, c=7
    intra mode 11); the port picks the AB shape **8×8 + 4×8 + 4×8**. **The intrabc COEFF ARM IS
    BYTE-EXACT AT THAT SITE** — for the 8×8 candidate the port and C agree **to the unit**
    (`dv=(-1032,24)`, rate 32747, dist 171120, sse 952608, rdcost 40519646, yrate 18114,
    ydist 170288, ysse 951776, uvrate 6, uvdist 832, uvsse 832, uvskip 1) — and **C ACCEPTS that
    same candidate** (its 40519646 beats C's incoming `best_rd` 43226527). The tip is on the
    **INTRA** side: the intra winner rd feeding the node is **43223116 in the port vs 43226527 in
    C — 3411 low (0.008%)**. So the residual belongs to the **KB-2 / KB-6 speed-0 partition/mode
    near-tie family (KB-13 is the speed≥1 analogue), a SEPARATE root** — at the divergent site the
    intrabc DV search, the DV validity, the coeff arm and the var-tx pack all match C.
  - **THE INTRA-SIDE RESIDUAL — FOUR ROOTS FOUND AND FIXED 2026-07-19.** Method: a byte-inert
    instrumented sibling C (`/root/intra-rd-instr`, throwaway, removed) dumping (a) every
    `pick_sb_modes` leaf's incoming budget + rate/dist/rdcost/mode/ibc/skip/tx, (b) every
    `av1_rd_pick_partition` node's chosen partition + best_rdc, and (c)
    `av1_rd_pick_intra_mode_sb`'s luma/chroma split (`rate_y`/`rate_uv`/`dist_y`/`dist_uv`,
    filter-intra, tx_type, CfL alpha). **Byte-inertness was verified on every rebuild** — the
    instrumented binary reproduces the clean build's stream byte-for-byte (1891-byte frame OBU).
    The KB's "3411 low" number decomposes as **exactly 6 rate units** at this cell's
    rdmult (291065). The roots (in the order they were found):
    1. **`rd_pick.rs` — the intra-budget early exit SKIPPED the intrabc search.**
       `av1_rd_pick_intra_mode_sb` (rdopt.c:3680-3690) sets `rd_cost->rate = INT_MAX` when
       `intra_yrd >= best_rd` and skips the uv search + assembly, but it does **not return**:
       `rd_pick_intrabc_mode_sb` still runs, with `best_rd` left at the INCOMING budget (the
       `best_rd = rd_cost->rdcost` tightening at :3686 is gated on `rate != INT_MAX`), and
       `RD_STATS best_rdstats = *rd_stats` (:3491) carries that INT_MAX in so a winning DV
       overwrites the whole tuple — the `assert(rd_cost->rate != INT_MAX)` at :3690 exists
       precisely for this rescue. The port returned early, dropping the intrabc candidate on
       exactly the leaves intrabc exists for (badly intra-predicted = repeated content).
       Measured at the divergent BLOCK_8X8 mi(38,4): C's VERT sub-1 at mi(38,5) is rescued by
       intrabc, the port's bailed → VERT lost → the port took NONE (the 8×8 intrabc block).
       Effect: first differing byte **1038 → 1120**.
    2. **`partition_pick.rs` — `skip_ctx` hardcoded 0.** `av1_get_skip_txfm_context` is
       `above->skip_txfm + left->skip_txfm`; `pick_sb_modes` zeroes `mbmi->skip_txfm`
       (partition_search.c:910) and the intra path never sets it, so ctx is identically 0 on a
       pure-intra KEY frame — the invariant the hardcode encoded. It BREAKS under intrabc: a
       skip-arm intrabc block has `skip_txfm = 1` (`set_skip_txfm`, tx_search.c:254), so its
       neighbours owe the dearer `skip_txfm_cost[1..2][0]`. Now read from the live DV grid
       (which already carries the per-mi `skip_txfm` projection). Byte-inert without a DV grid.
    3. **`partition_pick.rs` — `allow_intrabc: false` hardcoded in the leaf mode-cost cfg, so
       `intra_mode_info_cost_y` never added `intrabc_cost[use_intrabc]`
       (intra_mode_search_utils.h:563-564).** On an intrabc frame EVERY intra luma candidate pays
       the `use_intrabc = 0` flag (`write_intra_frame_mode_info` writes it for every block), so
       every intra leaf's luma rate was **35 units too CHEAP** on this cell while the PACK still
       wrote the flag — a systematic under-cost of the whole intra side against the intrabc arm
       and against every partition assembled from more leaves. **This is the measured root of the
       "intra RD comes out slightly cheap" signature.** Verified at the first divergent node
       mi(44,20) BLOCK_16X16: port luma rate 131946 → **131981 == C's 131981** exactly (chroma
       already matched at 2897, and after the fix `rate`/`rytok`/`mode`/`tx_size`/`tx_type`/
       filter-intra/CfL-alpha ALL match C to the unit).
    4. **`encode_sb.rs` + `aom-dsp/src/intra/cfl.rs` — `cfl_store_block` was unported.**
       `encode_superblock` (partition_search.c:580-583) runs `if (is_inter_block(mbmi) &&
       !xd->is_chroma_ref && is_cfl_allowed(xd)) cfl_store_block(xd, mbmi->bsize, mbmi->tx_size);`
       and **`is_inter_block` is TRUE for intrabc** (blockd.h:372). So an intrabc block that is not
       a chroma reference must still publish its reconstructed luma to the CfL buffer for the later
       chroma-reference sibling covering it. The port's own docs called this site "dead for intra"
       — TRUE until the intrabc arm landed, FALSE after. Without it, a CfL chroma-ref block whose
       luma footprint contains an intrabc sibling predicted from a STALE luma buffer. Found
       statistically: of 995 leaves where port and C agree on winner+rates, 985 had identical
       distortion and **all 10 divergences were `UV_CFL_PRED`**, 9 of them chroma-only and every
       delta a multiple of 16 (i.e. real SSE). The fix closed the 3 earliest — exactly the ones
       with a coded intrabc block inside their CfL luma footprint. `cfl_store_block` +
       `get_tx_size` are now ported (frame-edge clip via `max_block_wide/high` + align-up, then
       one synthetic tx). The intrabc SKIP arm's `mbmi->tx_size` was also corrected to
       `max_txsize_rect_lookup[bsize]` per `set_skip_txfm` (previously the dead intra winner's
       size; now live because `cfl_store_block` reads it for the edge alignment).
  - **SHARED-ROOT HYPOTHESIS: REFUTED (do not re-chase).** The "intra RD is slightly cheap"
    signature is shared across KB-15 / KB-13 / KB-10 / KB-11 / KB-12 / KB-P29, but the ROOT is
    not. Root 3 — the only one that is a systematic intra under-cost — is gated on
    `av1_allow_intrabc(cm)` and is therefore **structurally zero** on every other member:
    KB-13's cells are decoded photographic content (`allow_screen_content_tools = 0`), KB-10/11/12
    are synthetic noise/diag (non-screen), and the KB-P29 palette gate runs
    `enable_intrabc = 0`. Empirically confirmed: the full encode+bench suite is 340/340 both
    before and after all four fixes, and every one of those entries' pins **asserts DIVERGENCE**
    — a flip to MATCH would have failed the pin. So KB-13 keeps its own hypothesis (a speed>=1
    prune C applies that the port does not) and KB-10/11/12/P29 keep theirs.
  - **ROOT 5 — CLOSED ✅ 2026-07-19: the intrabc DV SEARCH ranked candidates against the RECON
    buffer; C searches the SOURCE frame.** `rd_pick_intrabc_mode_sb` sets `xd->plane[i].pre[0]`
    from `xd->cur_buf` (`av1_setup_pred_block`, rdopt.c:3482) and **`xd->cur_buf = cpi->source`**
    (encoder.c:4121, encodeframe.c:217) — so the hash-candidate costing (`get_mvpred_var_cost`,
    mcomp.c:1966) AND the full-pel diamond+mesh measure variance against the SOURCE frame; only
    the accepted candidate's RD stage (`av1_enc_build_inter_predictor`) predicts from the recon.
    The port passed `recon_y` as the search reference in BOTH arms (`intrabc_search.rs` hash arm
    + `FullPelSearch.refb`). The port's source-clone recon initialization MASKED the bug wherever
    the referenced region was never yet written (recon == source there) — which is why the DV
    search matched C on every previously-analyzed block. At coded regions recon ≠ source:
    measured at **mi(42,22) BLOCK_8X8** (dual byte-inert dump) — dir=1, SAME mv (-17,-39): C sme
    26062 (vs source) / port 40162 (vs recon); dir=0: the port found dv=(-1168,224) (sme 24062
    vs recon; RD 27551753 < intra 28154158 → **intrabc WON**) where C's source-based search finds
    (-116,-23) → dv=(-928,-184) (RD 34841670 → intra wins). The port coded intrabc where C codes
    intra H_PRED — the ACTUAL first divergent coded block (byte 1120 = mi(42,22)'s mode info).
    **The earlier "mi(44,20) first-order" framing is superseded**: it came from the
    CfL-distortion table, which only covered winner-AGREEING leaves; mi(44,20)'s wrong above-row
    pixels were DOWNSTREAM (they are the TR-node mi(40,20) winner re-encode's output, which
    contained the intrabc recon instead of H_PRED's). **Fix:** both search arms reference
    `a.src_y` (intrabc-only code — non-screen envelope structurally untouched). **Measured
    effect: port 1902B → 1895B (delta +11 → +4)**; dir=0 now finds C's exact mv and both sides
    reject intrabc at that node. `first_diff` unchanged at 1120 (the residual below).
  - **REMAINING RESIDUAL (witness still PINNED) — a 3-rate-unit TX-SIZE-COST TABLE gap at
    mi(42,22).** After root 5 the intra winner at mi(42,22) matches C on mode (H_PRED=2), tx
    (TX_4X4), dist_y (87440), dist_uv (8464), rate_tokenonly (27009) and rate_uv (162) — but
    **rate_y is 27763 vs C's 27766 (3 low)**. Term-by-term decomposition (dual dump):
    y_mode_cost 495==495, angle_delta_cost 45==45, intrabc_cost[0] 35==35, filter-intra n/a
    (H_PRED), tx_size_ctx 0==0, but **`tx_size_cost[cat0][ctx0][depth1]` = 179 (port) vs 182
    (C)** — the `TxSizeCosts` TABLE CONTENT differs while the pack's tx-size CODING is
    byte-exact everywhere (KB-6 30/30, all speed gates), so the port's cost fill reads a
    different cdf source than its pack writes with. Also parked: the intrabc candidate's
    same-dv RD rate differs by +45 (33136 vs 33091, both losing — decision-inert at this node,
    plausibly the same table-family root). Next: compare the port's `tx_size_cdf` slice feeding
    `fill_tx_size_costs` (mode_costs.rs:543) against C's `fc->tx_size_cdf[0][0]` at fill time;
    fix, re-run the witness, and expect the mi(42,22) chain (and the downstream mi(44,20)
    divergences) to collapse — then RAISE the floor past 1120 or promote the cell.
  Working notes: `docs/inter-vartx-coeff-arm-notes.md` (updated with the chroma inter path, the
  encode-vs-write walk-order difference, and the `set_skip_txfm` nonzero-rate detail).

## Encoder single-frame primary envelope (VERIFIED against reference/libaom)

Primary config = ALLINTRA (usage=2), speed-0 KEY frame. libaom's own allintra tuning
(`av1/av1_cx_iface.c:3065`) sets these **defaults** — so matching them, NOT the base defaults,
is what "single-frame exact" means:

- **CDEF: OFF** by default in allintra ("CDEF has been found to blur images, so it's disabled
  in all-intra mode"). Only `--enable-cdef` turns it on.
- **Loop-restoration: ON** by default in allintra (speeds 0-4). CORRECTED 2026-07-18 (the prior
  "OFF by default" claim was WRONG — verified first-hand against reference/libaom):
  `default_extra_cfg.enable_restoration = 1` (`av1_cx_iface.c:286`, the `!CONFIG_REALTIME_ONLY`
  build), NOT cleared by the `:3065` allintra override (which only touches CDEF / screen-mode /
  qm_min/max), and kept for non-realtime at `:1273-74` (`usage != REALTIME` → stays 1). So a
  plain `aomenc --allintra` runs `av1_pick_filter_restoration` and emits the seq/frame
  restoration syntax (even when every unit resolves RESTORE_NONE → different header bits from
  `--enable-restoration=0`). At **speed >= 5** C disables both Wiener+SGR
  (`speed_features.c:519-520`) → `enable_restoration &= 0` (`:2754`) → restoration OFF (the seq
  bit is 0). PARITY C2 was correct all along. The port's byte-exact LR search is now wired into
  the DEFAULT path (`aom-bench::EncodeCell::port_encode` derives the LR stage from the frame's
  `enable_restoration` = C's `is_restoration_used`); default parity is gated by
  `lr_default_parity::port_default_matches_plain_aomenc_allintra` (port default byte-matches a
  no-flags `aomenc --allintra`).
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
matches the allintra defaults for CDEF and QM (both off) but codes restoration **off** — a
NON-default config for **speeds 0-4** (the true default is restoration ON; see the corrected note
above). Those `encoder_gate_e2e_*` gates stay valid as `--enable-restoration=0` config tests
(and at speed >= 5 restoration-off IS the true default). The DEFAULT (restoration-on) path is
now separately wired + gated: `lr_restoration_gate` (the LR search, 8/8 byte-exact) +
`lr_default_parity` (port default == plain no-flags `aomenc --allintra`, restoration on). The
frame HEADER is still bootstrapped from the real parse (qindex, tile info, cdf-update, ...) —
only LF-level (and, on the default path, the restoration decision) is port-derived.

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
- **#10 cpu-used 0..9 speed-feature sweep** (Gate 2) — **DONE ✅ (all speeds 0-9)**: speeds 0-7
  (KB-8/KB-9/KB-10/KB-11; 6/7 = 64/64 canon each), speed 9 = 64/64 canon + noise, speed 8 =
  60/64 canon (4 diag estimate-arm V/H near-ties pinned open, KB-12) + noise — the nonrd
  PICKMODE (`use_nonrd_pick_mode`, `av1_nonrd_use_partition` single-pass walk,
  `av1_nonrd_pick_intra_mode` + `hybrid_intra_mode_search`). See KB-12. The speed sweep above is
  SYNTHETIC content; **REAL content at speed>=1 is a SEPARATE residual (KB-13, task #39)** — the
  synthetic gates are 64/64 but decoded-conformance content diverges at 36/60 speed-1..4 cells (all
  interior BLOCK_16X16/8X8 partition RD near-ties, port over-picks AB/SPLIT), pinned self-promoting
  in `encoder_gate_real_content_speed1to4_e2e`. Remaining Gate-2 byte-exactness is the 4 speed-8
  diag near-ties + the KB-10/KB-11 speed-6/7 noise-cq63 near-tie + the KB-13 real-content set (all a
  sibling-C RD dump away). (#8 qindex-from-cq and #21 decoder q62/q63 also DONE + CI-green.)

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
- **#23 QM-on encode — DONE ✅ (2026-07-16)**: `--enable-qm=1` allintra KEY byte-matches real
  aomenc — `encoder_gate_qm_on_e2e` (40 cells, bd8+bd10, qm ranges (5,9)+(4,10), mono+420) +
  anti-vacuous witness. QM selection runs inside the RD search (`resolve_qm` per tx in
  `xform_quant`), levels via `aom_get_qmlevel_allintra`. KEY subtlety (root-caused via
  sibling-libaom dump): C's trellis weights its DISTORTION by the forward matrix ONLY under
  `dist_metric == QM_PSNR` (tune=IQ) — with default PSNR the trellis runs `qmatrix = NULL`
  while dequant still folds `iqmatrix` (`optimize_txb_qm` now takes `Option` for the dist qm).
  tune=IQ / tune=SSIMULACRA2 (QM_PSNR dist, 444-chroma level formula, chroma deltaq,
  sharpness=7) remain out of envelope. See STATUS.md 2026-07-16.
- **#7 CDEF-strength RD search — DONE ✅ (2026-07-17), BIT-IDENTICAL**: full `av1_cdef_search`
  port (`aom-encode/src/pickcdef.rs`) + the two-pass encode→LF→search→pack architecture
  (`pack_tile_from_trees`, pack.rs) — 14/14 cells byte-match real aomenc `--enable-cdef=1`
  (real content 196²/64² cq5..63 with cdef_bits=2 per-unit literals; mono/444/420/bd10
  synthetic axes). Gate: `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close`
  (aom-bench, via the rd_close harness + full byte-identity asserts). CDEF stays off by
  default — the default envelope is untouched. FAST search levels 1..5 ported (table-level
  unit tests); only FULL (speed 0) is e2e-gated so far. See STATUS.md 2026-07-17.
- **Loop-restoration (Wiener/SGR) search — DONE ✅ (2026-07-18), BIT-IDENTICAL + DEFAULT-WIRED**:
  loop-restoration is **ON by default** in allintra (speeds 0-4; NOT a non-default knob — the
  prior "off by default" note here was WRONG, see the corrected primary-envelope note above). The
  byte-exact `av1_pick_filter_restoration` search (`crates/aom-dsp/src/restore/pick.rs`; PARITY C2) is now
  wired into the port's DEFAULT path (`aom-bench::EncodeCell::port_encode` derives the LR stage
  from the frame's `enable_restoration` = C's `is_restoration_used`). Gates: `lr_restoration_gate`
  (the search, 8/8 real content + 3/3 mono/444/bd12 format axis) + **`lr_default_parity`** (the
  port's default encode byte-matches a genuinely no-flags `aomenc --allintra` — the reference is
  the new `shim_encode_av1_kf_defaults`, tools at allintra defaults). Restoration-off remains a
  valid non-default config tested by the `encoder_gate_e2e_*` gates. Speeds 1-4 LR search arms +
  GOOD-mode arms are source-verified but PINNED (real-content base encode not yet byte-exact at
  speed >= 1, KB-13); speed >= 5 is structurally LR-off in C. See STATUS.md 2026-07-18.

## Coordination (parallel tracks)

- Max clean parallelism = **2** (one decoder agent + one encoder agent); cargo's shared
  target-dir lock serializes builds, which keeps the box safe.
- Strict crate ownership; commit with **explicit per-file staging** (`git add <paths>`, never
  `-A`/`-u`/`.`); shared `STATUS.md` via `git add -p`. Push `git push origin HEAD:main`; verify
  `git merge-base --is-ancestor HEAD origin/main`.
- Coordinator independently verifies every landing (on origin, boundary-clean, no `#[ignore]`
  / weakened asserts, gate is a real byte-identity assertion, CI green). Never trust a claim.
