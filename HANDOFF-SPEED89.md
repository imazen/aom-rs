# HANDOFF — speed-8/9 nonrd pickmode (KB-12), Gate-2

**Status: WIP, NEVER COMPILED OR TESTED.** Traced from libaom source under a
kill-order (spend-limit shutdown) before a handoff could be written. This doc is
the reconstructed state, written by the salvage agent from the committed diff +
CLAUDE.md KB-11/KB-12 prep-facts. **Expect mechanical compile fixes.** No porting,
tests, or dispatch wiring was done in the salvage pass — this is documentation only.

- **Branch:** `worktree-agent-a27658295e650d7d5`
- **Tip:** `7f18b7a` ("wip: coordinator salvage — resume-dump shutdown")
- **Base (merge-base with origin/main):** `72df1c4` (KB-11 speed-7 landed, +cpu6/7 PARITY rows)
- **True agent diff = `72df1c4..7f18b7a`:** 3 files, **+1220 / −14**
  (`git diff 72df1c4..7f18b7a` — the larger `origin/main..HEAD` stat is noise:
  origin/main diverged forward adding palette_search/intrabc files this branch predates).

The single commit `7f18b7a` is a mass "resume-dump" salvage of several sibling
worktrees at shutdown; its message is bare. The real content is these 3 files.

---

## What's committed, per file

### 1. `crates/aom-encode/src/nonrd_pickmode.rs` — NEW (+880) — the estimate-arm math library

The self-contained numeric core of `av1_nonrd_pick_intra_mode`. **Believed
content-complete for the lowbd (8-bit) canon envelope; never compiled/run.** Carries
a rich HANDOFF STATUS header (lines 1–108) with per-arm C provenance and the verified
speed-8/speed-9 sf-delta analysis — read it, it is the primary source for this port.

DONE (traced, wrapping-i16 where C wraps):
- LP kernels: `get_msb`, `hadamard_col8`, `hadamard_lp_8x8`, `hadamard_lp_8x8_dual`,
  `hadamard_lp_16x16`, `quantize_lp` (`av1_quantize_lp_c`), `satd_lp`, `block_error_lp`.
- Transposed scan tables `DEFAULT_SCAN_8X8_TRANSPOSE`, `DEFAULT_SCAN_LP_16X16_TRANSPOSE`
  (nonrd_opt.h:212/238) — the `_lp` Hadamard outputs are in C's transposed order.
- `block_yrd_lowbd` (`av1_block_yrd`, nonrd_opt.c:126 lowbd arm) — the per-txb
  Hadamard-estimate RD with C's edge clamps + final `rate <<= 2 + PROB_COST_SHIFT` fold.
- `nonrd_pick_intra_mode` (`av1_nonrd_pick_intra_mode`, nonrd_pickmode.c:1582) — the
  DC/V/H/SMOOTH estimate loop: predict → subtract → block_yrd → skip-cost fold →
  `bmode_costs[mode]` → rdcost; includes the flat-block force-DC (:1636), and the
  speed-9 prunes (h-pred :1648, neighbor :1656, best-SAD :646). Palette arm guarded dead.
- `hybrid_use_rdopt` (`hybrid_intra_mode_search` dispatch, partition_search.c:755):
  `hybrid=2` speed-8 / `0` speed-9, `var_thresh={0,101,201}`, `bsize < BLOCK_16X16`.
- `NonrdIntraLeafCtx` / `NonrdIntraPick` structs, `nonrd_leaf_tx_size`,
  `should_prune_intra_modes_using_neighbors`.
- 4 `#[cfg(test)]` unit tests (quantize_lp, two Hadamard flats, hybrid gate) — **never run.**

STUBBED / out-of-envelope (`unimplemented!` or assert-guarded — all correctly gated
so they cannot fire on the 8-bit canon grid):
- `fdct4x4_lp` + the `TX_4X4` arm of `block_yrd_lowbd` → `unimplemented!("… lossless — out
  of canon envelope")` (L460). Lossless-only; needs `default_scan_4x4` wiring if ever opened.
- **HBD (bd10/bd12) estimate arm NOT ported** — `nonrd_pick_intra_mode` asserts `env.bd == 8`
  (L594). Needs `aom_hadamard_16x16` + `av1_quantize_fp` + the `fp_16x16_transpose` scans
  (nonrd_opt.c:199–215) before any high-bit-depth speed-8 gate.
- Palette (`av1_search_palette_mode_luma`) NOT ported — `debug_assert!(!allow_screen_content_tools)`
  (L784). Dead on the canon grid (`allow_screen_content_tools=0`); required before any
  screen-content speed-8 cell.

### 2. `crates/aom-encode/src/partition_pick.rs` — MODIFIED (+298) — the nonrd walk

Header L3613: **"written under kill-order, NEVER COMPILED — see HANDOFF-SPEED89.md."**

- `nonrd_use_partition_real` (`av1_nonrd_use_partition`, partition_search.c:2960) — the
  single-pass walk over the VBP-stamped tree: PARTITION_NONE / HORZ|VERT (strip-0 then
  strip-1, gated `sub1_in_frame && bsize > BLOCK_8X8`) / SPLIT (plain recursion). The
  KEY-dead arms (`try_split_partition`, `try_merge` :3089, `direct_partition_merging`
  :3106, `reuse_inter_pred`) are documented and correctly NOT modelled.
- `nonrd_leaf_pick_and_encode` — one leaf: recompute `source_variance` via the existing
  `perpixel_variance_y`, then dispatch on `hybrid_use_rdopt`:
  - **full-RD arm** (`bsize < 16×16 && var >= 101` at speed 8) → existing `leaf_pick_sb_modes`
    with INT64_MAX budget, then `encode_b_intra_dry`, then `grid.stamp`.
  - **estimate arm** → build `NonrdIntraLeafCtx` (neighbor Y modes from the grid, KF y-mode
    ctx pair, skip ctx 0, luma edge-filter type), call `nonrd_pick_intra_mode`, construct a
    `LeafWinner` (uv=DC, angle 0, filter-intra off, tx_type_map all DCT_DCT, skip_txfm false,
    uv/luma edge-filter recomputed), `encode_b_intra_dry`, `grid.stamp`.

All referenced symbols were verified to **exist and type-match** (so these are NOT the
compile blockers): `perpixel_variance_y` (partition_pick.rs:215), `ModeGrid::at`/`at_uv`
(:368)/`stamp`, `MI_SIZE_WIDE_B`/`MI_SIZE_HIGH_B`, `LeafWinner` (all fields incl.
`skip_txfm`/`raw_rdstats`), `encode_b_intra_dry(…, output_enabled: bool)`, every read
`SbEncodeEnv` field, and `PlaneQuantRows.{round_fp,quant_fp,dequant}: &[i16;8]` (consumed
by `block_yrd_lowbd`). The frame-edge single-strip rect case is `unimplemented!` (same
interior-envelope limitation `rd_use_partition_real` carries today).

### 3. `crates/aom-encode/src/pack.rs` — MODIFIED (+56 / −14) — speed-9 cost-update gating

In BOTH `pack_tile` (~L977) and `pack_tile_from_trees` (~L1197): the per-SB
`derive_real_costs` refresh is now gated on `!(allintra && speed >= 9)`. Rationale
(verified, speed_features.c:166–177 vs :593–594): at speed-9 <4k the cost-update level
resolves to `INTERNAL_COST_UPD_OFF`, so every SB must read the FRAME-INIT cost tables, not
a per-SB refresh — byte-visible on multi-SB (128²) frames. `sb_real` became an `Option`.
`// HANDOFF(speed 9)` L986/L1220: 4k+ frames (which keep `SBROW`) are out of the canon
envelope and unmodelled.

---

## The nonrd KEY chroma path — **RESOLVED** (was the KB-11 FLAGGED-UNRESOLVED unknown)

`av1_nonrd_pick_intra_mode` is **PLANE_Y only** and hard-sets `mi->uv_mode = UV_DC_PRED`
(**nonrd_pickmode.c:1735**, C comment "Keep DC for UV since mode test is based on Y channel
only"). There is NO uv mode search and NO uv rate/dist on the estimate arm. Consequences,
as implemented in the walk:

- **Estimate leaves:** chroma is coded as **DC** by the ordinary leaf encode
  (`encode_b_intra_dry` consuming `LeafWinner{ uv_mode: 0, .. }`). `init_mbmi_nonrd`
  (nonrd_opt.h:516) zeroes palette sizes + filter_intra; CfL is never a candidate (uv fixed
  DC) so `cfl_alpha_* = 0`. The recomputed `uv_edge_filter_type` is decision-inert (DC is
  non-directional) but carried for fidelity.
- **Full-RD leaves** (`bsize < 16×16 && var >= 101`, speed-8 only): chroma is picked by the
  EXISTING ported machinery (`leaf_pick_sb_modes` → `av1_rd_pick_intra_mode_sb`), unchanged.

This is coded correctly in `nonrd_leaf_pick_and_encode`. The only residual chroma risk is
the shared `output_enabled` question below (it governs the tx_type_map copy semantics on
BOTH planes — see KB-4).

---

## What remains — exact next steps to a speed-8 gate

The walk math + wiring are drafted but **the code does not compile and the dispatch is not
connected.** Order of operations:

1. **Fix the pack.rs compile break (concrete, found in salvage).** The `Option` refactor
   updated `sb_env` (via `if let Some(sb_real) = &sb_real`) but NOT the `sb_pick_cfg` block
   that immediately follows: `pack_tile` **L1003–1009** still reads `&sb_real.mode_costs`,
   `&sb_real.tx_size_costs`, `&sb_real.skip_costs`, `&sb_real.tx_type_costs_y`,
   `&sb_real.mode_costs.intra_uv_mode_cost`, `&sb_real.cfl_costs`, `&sb_real.partition_costs`
   — field access on an `Option`, a type error. Fix: when `cost_upd_off`, `sb_pick_cfg` must
   fall back to the frame-init `pick_cfg` (the `INTERNAL_COST_UPD_OFF` semantics — SB reads
   frame-init tables), else build from `sb_real.as_ref().unwrap()`. Audit
   `pack_tile_from_trees` for the same pattern.

2. **Compile `nonrd_pickmode.rs` + the walk; fix the mechanical errors the never-compiled
   code surfaces.** Likely spots (unverified — the code was never fed to rustc): the
   `aom_entropy::partition::intra_avail(...)` call's exact arg list/arity in
   `nonrd_pick_intra_mode`; `predict_intra_high` arg order; the `LeafVisit` field set
   (`budget/rate/dist/rdcost`); `get_partition_subsize` / `get_partition_from_stamps`
   signatures; imports (`SbEncodeEnv`, `PartRdStats`, `highbd_subtract_block`,
   `predict_intra_high`). Run `cargo build -p aom-encode` and clear the diagnostics; the
   referenced symbols all exist, so these should be arity/borrow fixes, not redesigns.

3. **Resolve the `output_enabled` question (the KEY open correctness item).** Both leaf arms
   currently pass `encode_b_intra_dry(…, false)`. In C the nonrd walk encodes every leaf with
   `dry_run = 0` (**OUTPUT_ENABLED**). Per KB-4, `output_enabled` governs the tx_type_map
   **copy-vs-alias** semantics: OUTPUT_ENABLED copies ctx into the frame map (eob-0 → DCT_DCT
   resets land there, ctx untouched); DRY aliases (resets persist into the winner map). Trace
   what `encode_b_intra_dry`'s `output_enabled` gates in THIS port's split architecture (recon
   in the walk, bits in `pack_sb`) and pass the value that reproduces the speeds-0-7-proven
   root-walk leaf semantics. Getting this wrong reintroduces the KB-4 eob-0 reset-leak class.

4. **Wire the dispatch (currently absent — `nonrd_use_partition_real` is defined + self-
   recursive but CALLED FROM NOWHERE).** Add the `allintra && speed >= 8` branch to the encode
   entry (mirror the speed-7 VBP path): (a) `choose_var_based_partitioning_key` builds the
   `vbp_stamps` (already ported for KB-11 speed-7); (b) `nonrd_use_partition_real` walks it per
   SB into an `SbTree`; (c) pack via the existing `pack_sb`/`pack_tile_from_trees` (the walk
   does recon + context; the pack writes bits — the same search/pack split proven byte-exact
   for speeds 0-7, because the symbol stream is the same tree replay). Note the module docs'
   claim that "search == pack replay" holds here; validate it on the first passing cell.

5. **Build the speed-8 gate** on the canon grid, following the KB-8..KB-11 pattern:
   `encoder_gate_speed8_textured_allintra` = {64,128}² × cq{12,32,48,63} × {flat,two-tone,
   vgrad,diag} × {mono,420} vs `aomenc --cpu-used=8`, byte-for-byte; PLUS the anti-vacuous
   witness (`encoder_gate_speed8_vs_speed7_sf_witness`: port with FULL speed-7 features vs
   `aomenc --cpu-used=8` must DIVERGE; speed-8 features must match). Localize any divergent
   cell with a decode-both harness (`kb11_speed7_noise_localize.rs` is the template).

6. **Then speed 9.** Flip `hybrid = 0` (estimate arm for every leaf) + the three estimate-loop
   prunes (already wired via `cfg.speed >= 9`). Verify the framesize-dependent extras:
   - the `INTERNAL_COST_UPD_OFF` gate (pack.rs, step 1) — byte-visible on 128² (multi-SB) cells.
   - `vbp_prune_16x16_split_using_min_max_sub_blk_var` (speed-9): the 16×16 force-split becomes
     `get_part_eval_based_on_sub_blk_var` (`(max−min) > threshold16<<2` → ONLY_SPLIT else
     ONLY_NONE). **HANDOFF (nonrd_pickmode.rs:58): verify `var_part.rs` implements the ONLY_NONE
     arm (3-state PART_EVAL), not just a bool force-split** — the param exists (passed `false`
     today) but the tri-state path may be absent. Build `encoder_gate_speed9_*` + witness.

---

## `// HANDOFF:` marker index (all in the committed code)

| File:line | Item | Blocking canon grid? |
|---|---|---|
| partition_pick.rs:3613 | Whole walk NEVER COMPILED | **yes** |
| partition_pick.rs:3797, :3903 | `output_enabled` value for leaf encode (see step 3) | **yes (correctness)** |
| pack.rs:1003–1009 | `sb_pick_cfg` dangling on `Option<sb_real>` (step 1) | **yes (speed-9)** |
| pack.rs:986, :1220 | 4k+ `SBROW` cost-upd unmodelled | no (>canon) |
| nonrd_pickmode.rs:58 | speed-9 vbp_prune 16×16 ONLY_NONE tri-state | speed-9 only |
| nonrd_pickmode.rs:107, :594 | HBD estimate arm not ported | no (bd8 only) |
| nonrd_pickmode.rs:121 | dedupe `INTRA_MODE_CONTEXT` table | no (cosmetic) |
| nonrd_pickmode.rs:283, :457, :460 | lossless `fdct4x4_lp` / TX_4X4 scan | no (>envelope) |
| nonrd_pickmode.rs:469 | verify allintra pack never reads `blk_skip` | verify |
| nonrd_pickmode.rs:541 | re-verify `select_tx_mode` at speed 8/9 allintra | verify |
| nonrd_pickmode.rs:616 | caller `bmode_costs` ctx-pair parity | verify |
| nonrd_pickmode.rs:784 | palette arm (screen-content) not ported | no (scr only) |

---

## C reference file:line map (libaom, consolidated)

- **`av1_nonrd_use_partition`** partition_search.c:2960 — NONE :3017, VERT :3031, HORZ :3055,
  SPLIT :3078–3117, `try_merge` :3089, `direct_partition_merging` :3106, extended :3119–3125.
- **`pick_sb_modes_nonrd`** partition_search.c:2254 — per-leaf source_variance recompute
  :2306–2311; hybrid dispatch :2325.
- **`encode_b_nonrd`** partition_search.c:2089/2100 — `mi->skip_txfm = 0` for intra :2120.
- **`hybrid_intra_mode_search`** partition_search.c:755–772 — `var_thresh` index :762–766.
- **`av1_nonrd_pick_intra_mode`** nonrd_pickmode.c:1582 — force-DC flat :1636–1640, h-pred
  prune :1648–1650, neighbor prune :1656–1668, skip-cost fold :1676–1687, palette :1698–1731,
  **uv=DC :1734–1735** (the chroma answer).
- **`av1_block_yrd`** nonrd_opt.c:126 — edge clamps :141–144, `update_yrd_loop_vars` :43,
  `AOMMIN(tx_size, TX_16X16)` :660, sse arm :322–336. **`av1_estimate_block_intra`** SAD prune :646–668.
- **LP kernels:** aom_dsp/avg.c — `hadamard_col8` :149, `hadamard_lp_8x8` :209, `_dual` :240,
  `hadamard_lp_16x16` :291, `satd_lp` :520; fwd_txfm.c:85 `fdct4x4_lp`; av1_quantize.c:214
  `quantize_lp`; rdopt.c:907 `block_error_lp`. Scans: nonrd_opt.h:212/238.
- **Speed features** speed_features.c — `use_nonrd_pick_mode=1` :578, `hybrid_intra_pickmode=2`
  :579, `nonrd_check_partition_merge_mode=1` :580, `var_part_split_threshold_shift=8` :581,
  `prune_palette_search_nonrd=1` :582, `intra_y_mode_bsize_mask_nrd` :583–590, speed-9 block
  :592–607, framesize-dep cost-upd-off :166–177, `force_large_partition_blocks_intra` (720p+)
  :326–328.
- **`encode_nonrd_sb`** encodeframe.c:581–663 — `source_sad_nonrd` inits kMedSad :1289.
- **`choose_var_based_partitioning`** source_variance var_based_part.c:1724–1731.

---

## Validation recipe (canon grid vs `aomenc --cpu-used=8`, then `9`)

Same evidence discipline as KB-8..KB-11 (real exported C > synthetic-facade > transcription):

1. **Canon byte-identity gate.** `encoder_gate_speed8_textured_allintra`: encode the port and
   compare byte-for-byte against real `aomenc --cpu-used=8` over {64,128}² × cq{12,32,48,63} ×
   {flat,two-tone,vgrad,diag} × {mono,420} (64 cells). Assert FULL byte-identity (fails on any
   regression). Repeat at `--cpu-used=9` for `encoder_gate_speed9_textured_allintra`.
2. **Anti-vacuous witness.** Encode with the PREVIOUS speed's features vs `aomenc --cpu-used=8`
   and assert it DIVERGES on at least one cell (incl. a mono cell to prove it luma-side); with
   speed-8 features it must match. This proves the gate actually exercises the new sf deltas.
   Same for speed-9 vs speed-8.
3. **Deep-tree noise extension.** `encoder_gate_speed8_noise_flatuv_allintra` (cq12/32/48 hard-
   asserted) to exercise the sub-16×16 full-RD arm + deeper VBP trees. The `noise 64² cq63`
   near-tie (the KB-10/KB-11 twin — (mi 8,0) TX_16X16-vs-TX_32X32 winner-sweep) may resurface;
   pin it PRESENT if so, per the established pattern.
4. **Speed-9-specific coverage.** Ensure 128² (multi-SB) cells are present so `INTERNAL_COST_
   UPD_OFF` is byte-visible, and pick content that fires the `vbp_prune_16x16` ONLY_NONE arm.
5. **Localization on divergence.** Decode-both harness (template `kb11_speed7_noise_localize.rs`
   / `kb6_real_rd_localize.rs`): decode port + real streams, diff partition trees and per-leaf
   mode/tx records to find the first divergent SB; if the trees match but bytes differ, suspect
   the `output_enabled` tx_type_map semantics (step 3) or a write-ctx probability defect (KB-6
   class), not a search decision.
6. **Requires a `--cpu-used=8/9` reference-encode path.** Confirm the test harness's `aomenc`
   shim accepts cpu-used 8/9 (the KB-8..KB-11 harness already sweeps 4–7; extend it).

Do NOT relax any gate to green — a speed-8 cell that diverges is an open near-tie to root-cause,
same as KB-2/KB-6, not an accepted limitation.
