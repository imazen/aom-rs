# HANDOFF — TOGGLE SWEEP (C8/C9/C10/C11), 2026-07-17

Agent shut down mid-sweep by directive. State below is exact. Branch:
`worktree-agent-adb54d0b877f65828`, 5 commits on top of 90e69e8 (KB-10):

```
6e32167 wip: batch-E toggle roughs — trellis arms wired; cost-upd design   (NOT test-run)
bffd4d9 feat: TOGGLE SWEEP — 21 toggle byte-gates (20 EXACT + 1 pinned-open) (validated)
a6f5b04 feat: generic ctrl-pair encode oracle (shim_encode_av1_kf_ctrls)     (validated)
2c39484 feat: CLI toggle threading — IntraToolCfg / tx policy / CDF gate     (validated)
31ff88a feat: writer-side allow_update_cdf on OdEcEnc                        (validated)
```

**NOT PUSHED to main** (shutdown before the landing gate). **Validation state,
honestly:** every toggle test in `toggles_rd_close.rs` up through bffd4d9 ran
green (23 tests; log `/root/.claude/jobs/3651b35b/tmp/`); the FULL workspace
suite over the final tree was interrupted ~15/19 crates in, **0 failures at
kill** (`toggles_fullsuite_final2.log`); an earlier full suite over batch-A
passed 519/0. 6e32167's two trellis tests are compile-checked only.
**Next agent: run `cargo test --workspace` once; if green, rebase + push all
five to main** (`git push origin HEAD:main` after rebase; verify
`git merge-base --is-ancestor HEAD origin/main`).

## Per-toggle state (the full C8-C11 list)

| Toggle (ctrl) | State | Port threading | Validation |
|---|---|---|---|
| `--enable-rect-partitions=0` (73) | **DONE** | PickFrameCfg (pre-existing) | EXACT 3/3, pinned |
| `--enable-ab-partitions=0` (74) | **DONE** | PickFrameCfg | EXACT 3/3, pinned |
| `--enable-1to4-partitions=0` (75) | **DONE** | PickFrameCfg | EXACT 3/3, pinned |
| `--min-partition-size` (76) | **DONE** | px→BLOCK via C set_max_min_partition_size (harness) | EXACT 3/3 (16px), pinned |
| `--max-partition-size` (77) | **DONE** | same (min(sf_default,dim,sb)) | EXACT 3/3 (32px), pinned |
| square-only 8..32 interaction | **DONE** | all of the above | EXACT 3/3, pinned |
| `--sb-size=128` encode | **UNSTARTED (M)** | decoder+entropy are SB-generic (798ec25); encoder walk/harness SB-64-only. Own chunk; touch aom-bench SB_MI/SB consts + partition_pick sb_size plumbing + shim `ref_encode_av1_kf_sb128` (exists) | — |
| `--enable-tx64=0` (80) | **DONE** | PickFrameCfg (pre-existing) | EXACT 3/3, pinned |
| `--enable-rect-tx=0` (82) | **DONE** | PickFrameCfg | EXACT 3/3, pinned |
| `--enable-flip-idtx=0` (81) | **DONE** | TxTypeSearchPolicy→TxMaskParams (2c39484) | EXACT 3/3, pinned |
| `--use-intra-dct-only=1` (119) | **PINNED-OPEN** | threaded (same path as flip-idtx) | 64²cq32 OUT of band (+2.23%/−3.588), cq63 EXACT, 128² CLOSE — see below |
| `--use-intra-default-tx-only=1` (121) | **DONE** | pol.use_default_intra_tx_type OR-arm (MODE_EVAL, rdopt_utils.h:579) | EXACT 3/3, pinned |
| `--reduced-tx-type-set=1` (118) | **DONE** | bootstrap frame-header bit, asserted == knob | EXACT 3/3, pinned |
| `--enable-tx-size-search=0` (146) | **DONE** | pol.enable_tx_size_search → LARGESTALL single-pass + sf level 3 + tx_mode_is_select AND (2c39484). Assert is ONE-directional (C demotes SELECT→LARGEST post-hoc on zero-split frames) | EXACT 3/3, pinned |
| `--disable-trellis-quant=1/2` (62) | **WIRED, NOT RUN** (6e32167) | knob→trellis_opt_of_knob→pol.skip_trellis + env.enable_optimize_b | tests exist w/ HANDOFF notes; run them; expect EXACT |
| `--disable-trellis-quant=0` | inert-vs-default on intra | — | not celled (witness would refuse: estimate_yrd_for_sb is inter-only) |
| `--quant-b-adapt` | **UNSTARTED (S–M)** | needs the `quantize_b_adaptive` kernel family in aom-quant + policy plumb | — |
| `--enable-smooth-intra=0` (99) | **DONE** | IntraToolCfg→IntraSbyGates + UvLoopPolicy | EXACT 3/3, pinned |
| `--enable-paeth-intra=0` (100) | **DONE** | same | EXACT 3/3, pinned |
| `--enable-cfl-intra=0` (101) | **DONE** | UvLoopPolicy.enable_cfl_intra | EXACT 3/3, pinned |
| `--enable-directional-intra=0` (145) | **DONE** | IntraToolCfg + UvLoopPolicy | EXACT 3/3, pinned |
| `--enable-diagonal-intra=0` (141) | **DONE** | same | EXACT 3/3, pinned |
| `--enable-angle-delta=0` (106) | **DONE** | same | EXACT 3/3, pinned |
| `--enable-filter-intra=0` (98) | **DONE** | seq bit knob-driven, bootstrap ASSERTED equal; costs+gates+pack pre-threaded | EXACT 3/3, pinned |
| `--enable-intra-edge-filter=0` (78) | **DONE** | env.disable_edge_filter knob-driven, seq bit asserted | EXACT 3/3, pinned |
| `--cdf-update-mode=0` (44) | **DONE + REAL BUG FIX** | OdEcEnc.allow_update_cdf gate in write_symbol (31ff88a) + pack_tile sets it (2c39484) | EXACT 3/3, pinned. Pre-fix: zensim −264 (pack adapted partition/mode CDFs unconditionally) |
| `--coeff/mode-cost-upd-freq` (126/127) | **C-SIDE ONLY** (6e32167) | port gate UNWIRED — full design in ToggleKnobs doc (`HANDOFF:`): split pack.rs sb_real rebuild per table set; SB=every SB (current), SBROW=only at `c==0`, TILE/OFF=never (single-tile equal). PackCfg literal sites all break on new fields — add `cost_upd: CostUpdCfg` (Default) in one sweep | — |
| `--dv-cost-upd-freq` (142) | inert on envelope (intrabc off) | ctrl id present for completeness | not cellable (witness) |
| `--min-q/--max-q/--min-cr` | **DEFERRED** | qindex flows from bootstrap (Gate-3 caveat) — a cell would be vacuous port-side; needs the #8-family self-derived qindex first | — |
| `--full-still-picture-hdr` / annexb | **DEFERRED** | OBU framing only; port emits frame OBU payload, seq spliced from C | — |
| cost-upd default arm (SB) | proven | modeled by pack.rs per-SB derive_real_costs | pre-existing multi-SB byte gates |

## The dct-only pinned-open investigation (exact state)

Everything below is measured, in `toggles_c9_intra_dct_only_pinned_open`'s doc
+ PARITY.md section B:
- Y recon IDENTICAL port-vs-C on the divergent cell; chroma diverges from
  mi(0,0): real uv=D45/aduv2 (eob 1) vs port uv=V/aduv0 (eob 78); real
  winners frame-wide are derived-type==DCT modes = DCT-forced-search shape.
- Port's UV force IS live e2e (5093 gated mask calls counted); port's V rd
  1872917 beats DC 2157931; D45+CFL come out gated/never-evaluated in the
  port loop (visits dump).
- The REAL `get_tx_mask` facade confirms C forces DCT on the UV mask
  (uv V: 0x0002→0x0001) with the PAETH reduced-set empty-mask RESET
  (uv PAETH keeps derived ADST_ADST even under dct_only) — port matches.
- Five layer differentials now SWEEP the knob (oracle chains thread it into
  the REAL facades): txb search, luma yrd depth loop, luma mode loop, full
  leaf, UV txb walk, UV mode loop — ALL GREEN. ⇒ the divergence is a
  port+oracle SHARED mis-model of the REAL UV loop under dct_only.
- **Next step (unstarted): sibling-C instrumented dump** (KB-2/KB-7 method)
  of the mi(0,0) UV candidate list (uv_mode, this_rate, this_dist, this_rd)
  for `av1-1-b8-01-size-64x64` cq32 allintra speed-0 with
  AV1E_SET_INTRA_DCT_ONLY=1, then diff against the port's visits (dump via
  a temporary eprintln in rd_pick_intra_sbuv_mode — pattern used and
  removed this session). Suspects, in order: the CfL alpha-search gating
  under the knob (port CfL visits = None where real picks CFL at mi 14,0);
  the mode_rate early-RDCOST prune interacting with different best_rd
  trajectories; an angle-sweep gate difference.

## Validation recipe (per toggle — how everything above was proven)

1. Cell = `run_toggle_cell` in `crates/aom-bench/tests/toggles_rd_close.rs`:
   C encode via `EncodeCell::c_encode_ctrls(knobs.c_ctrls())` (real
   aom_codec_av1_cx + ctrl pairs), port via `port_encode_with(&c_tu, &knobs)`,
   compare via `rd_close::compare_cell`. Grid = 64²cq32 + 64²cq63 + 128²cq12
   real content (`GRID`).
2. **Anti-vacuity witness is mandatory**: `run_grid_and_gate` panics unless
   the knob CHANGED the C stream on ≥1 cell. Never trust an EXACT verdict
   without it.
3. EXACT cells get `expect_exact=true` (hard `bit_identical` pin). Divergent
   knobs get a pinned-open test (fails on movement EITHER way) + a PARITY
   section-B row with numbers + localization.
4. Ctrl ids: `aom_sys_ref::cx_ctrl` — ANY new id must get a
   `shim_cx_ctrl_id_by_probe` arm + PROBE_TABLE entry
   (`cx_ctrl_ids_match_reference_headers` cross-checks vs pinned headers).
5. C defaults all verified in `av1_cx_iface.c` `default_extra_cfg`
   (:280-400): every enable_* = 1, min/max part 4/128, dct/default-tx/reduced
   = 0, tx-size-search = 1, cdf-update = 1, trellis-quant = 3, cost-upd = SB.
   Allintra override block touches NONE of them (PARITY.md header note).
6. Seq/frame-header knobs (filter-intra, edge-filter, reduced-tx-set,
   tx_mode, disable_cdf_update): port side is KNOB-driven with the bootstrap
   header bit ASSERTED equal — never silently bootstrap-flowed (PARITY rule
   4). The tx_mode assert must stay ONE-directional (C's post-hoc
   SELECT→LARGEST demotion on zero-split frames).

## Gotchas discovered (do not re-lose)

- `write_coeffs_txb_full` keeps its explicit `allow_update_cdf` param —
  redundant with the new `OdEcEnc` flag but consistent (both from the same
  header bit). Unifying is optional cleanup.
- C FORBIDS `--enable-tx64=0` + `--enable-tx-size-search=0` together
  (encodeframe.c:2461 assert) — never grid that combo.
- `pkill -f "cargo test"` in a multi-agent repo can kill sibling agents'
  suites — scope kill patterns to this worktree's target path.
- aom-bench's stock `max_partition_size` was an unfaithful flat 15; it is
  now C-derived min(sf_default, dim, sb)=12 at SB64 — outcome-identical
  (consumers OR with `bsize == sb_size`), proven by the gates.
