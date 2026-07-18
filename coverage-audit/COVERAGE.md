# aom-rs single-frame coverage — master synthesis

**The living answer to "does the port cover all single-frame encoding tech?"**

**Date:** 2026-07-18 · **Basis commit:** `f29d4bd` (origin/main tip; the three source audits
were taken at `79e7a6d`) · **Reference:** libaom v3.14.1 (`reference/libaom`, git `03087864`).
**Mission scope:** single-frame / ALLINTRA (usage=2) / KEY-frame encode + intra-scope decode.
Inter/motion/TPL/GOP/rate-control-beyond-fixed-Q is **OUT-OF-SCOPE-inter** — enumerated for
completeness, never counted as a single-frame gap.

This document merges the three 2026-07-18 coverage sub-audits into one deduped, prioritized gap
matrix:

- [`control_surface_2026-07-18.md`](control_surface_2026-07-18.md) — the CLI/control-enum surface
  (≈133 single-frame knobs: 52 byte-exact-gated / 21 partial / 26 absent / 8 inert / 25 inter).
- [`coding_tools_2026-07-18.md`](coding_tools_2026-07-18.md) — the per-syntax-element bitstream
  tool surface (KEY frame, enc+dec), byte-exact both ways within ≤64×64 SB.
- [`encoder_modules_2026-07-18.md`](encoder_modules_2026-07-18.md) — the encoder algorithm/module
  surface (~199 byte-exact rows) + the loop-restoration default-parity headline.

Evidence priority (project CLAUDE.md): real exported C fn > synthetic-facade-over-real-fn >
verbatim transcription. Every "byte-exact" claim below is backed by a landed differential or
e2e byte-identity gate (cited in the source audits + `PARITY.md` Section A).

---

## COVERAGE SUMMARY

**The port covers essentially the entire single-frame / KEY-frame AV1 coding-tool surface
byte-exact — in BOTH directions (encode emits byte-identical to `aomenc`; decode is byte-identical
to the C decoder) — for the ≤64×64-superblock envelope.** Concretely, all of the following are
byte-exact both ways: every one of the 10 partition types + all ≤64 block sizes; all 19 transform
sizes + the 7 intra-reachable tx types (the other 9 are inter-only, kernel+symbol-proven); all 13
luma + 14 UV intra modes incl. CfL, angle-delta (±3), all 5 filter-intra modes, intra edge
filter/upsample; the FP/B/DC quantizer × flat/QM × 8/10/12-bit; the whole Daala od_ec + CDF stack;
deblock + CDEF + loop-restoration in-loop filters; 4:0:0/4:2:0/4:2:2/4:4:4 × 8/10/12-bit; the
sequence/frame/tile-group OBU writers; superres FIXED-denom; coded-lossless (mono+4:2:0). On the
encode side the whole speed-0..9 ALLINTRA KEY search→pack→coeff→post-filter-search→header pipeline
is byte-identical to real `aomenc` on synthetic grids (64/64 at every speed) and on real
decoded-conformance content at speed 0 (30/30), plus multi-tile, QM-on, CDEF-strength search,
loop-restoration search, tune=IQ/SSIMULACRA2, superres-FIXED, film-grain table-inject, and
deltaq-mode 3/6.

**Byte-exact fractions (each audit uses a different, valid denominator):**

| Surface | Denominator | Byte-exact | Note |
|---|---|---|---|
| **Coding-tool / bitstream** (KEY, ≤64 SB) | every syntax element | **≈100% both ways** | minus the encode-side gaps listed below (SB128, intrabc-search, palette 2/7, superres-auto, grain-estimation) |
| **CLI / control surface** (single-frame-relevant) | ≈133 knobs | **52 byte-exact-gated + 8 inert-default-off** | 21 partial / 26 absent / 25 out-of-scope-inter |
| **Encoder algorithm/module** rows | ~ full module surface | **~199 byte-exact rows** | remainder is the ABSENT/PARTIAL rollup below |
| **`xtask/coverage.py`** (auto-derived enc+dec CLI + control enums) | 349 | **90 green (25.79%)** | was a mechanical 0/349 (empty `feature_map.json`); see below |

**The single most impactful gap is a default-config mismatch, not a missing kernel:
loop-restoration is ON by default in allintra, but the (byte-exact) LR search is wired only behind
`--enable-restoration=1`, so the port's default path does not yet byte-match a plain
`aomenc --allintra` stream.** After that, the genuinely-structural single-frame holes are SB128
encode, `prune_tx_type_using_stats` (≥480p cpu≥2), the IntraBC leaf coeff-arm/DV-search, the AQ /
deltaq-2 / superres-auto / grain-estimation / tune=vmaf/butteraugli families, `--quant-b-adapt`,
and level/tier header derivation. Real-content byte-parity holds at speed 0 (30/30) but is PARTIAL
at cpu≥1 (24/60 — interior partition-prune near-ties). A handful of self-promoting pinned mode/tx
near-ties remain.

**Remaining single-frame gap count: 4 Tier-1 (default-parity) · ~16 Tier-2 (structural) · 4 Tier-3
(pinned near-ties).** Tier-0 hygiene is resolved by this landing (see below).

### Coverage-gate number (`xtask/coverage.py`)

`coverage.py` auto-derives the live libaom feature surface from `aomenc`/`aomdec --help` (182 + 20
CLI flags) plus `aomcx.h` control enums (147) = **349 features**, and marks a feature green only
when `coverage/feature_map.json` maps it to a passing test. Before this landing `feature_map.json`
was `{}`, so it reported a mechanical **0/349** — an artifact, not the real status. This landing
populates `feature_map.json` from `PARITY.md` Section A (the byte-identity ledger), mapping each
byte-exact-gated single-frame feature to its gate:

```
Coverage gate (feature surface): 90/349 green (25.79%), 259 red
  cli-dec    0/20
  cli-enc    48/182
  control    42/147
```

**Reading the 90/349 honestly:** the 90 green are the 52 byte-exact-gated single-frame knobs (each
knob maps to both its `--flag` id and its `AV1E_SET_*`/`AOME_SET_*` control id). The 259 red are
**not** all real single-frame gaps — the 349 denominator includes 25 out-of-scope-inter controls +
more inter/rate-control CLI flags, 20 `aomdec` application-I/O flags, ~20 `aomenc` application-I/O
flags (`--help`/`--codec`/`--ivf`/`--webm`/`--obu`/`--psnr`/…), getters, debug/SVC/RTC controls,
and the genuinely-absent single-frame features in the Tier-2/3 matrix below. Against the audit's
own ≈133-knob single-frame denominator, coverage is far higher than 25.79%. The gate only reads
100% at full whole-surface parity, so it stays useful as a north-star while the single-frame
mission is the near-term target. No green was invented: PARTIAL / ABSENT / inert / non-codec
features stay red (spot-checked: `--sb-size`, `--profile`, `--loopfilter-control`,
`--enable-palette`, `--enable-intrabc`, `--quant-b-adapt`, `--aq-mode` all correctly red).

---

## Envelope defaults (verified against reference/libaom)

Primary config = ALLINTRA (usage=2), speed-0 KEY. libaom's allintra override
(`av1_cx_iface.c:3065`) sets these effective defaults — matching THEM, not the base defaults, is
what "single-frame exact" means:

| Knob | base default | allintra effective | note |
|---|---|---|---|
| enable_cdef | 1 | **0 (OFF)** | override :3067 ("CDEF blurs images") |
| **enable_restoration** | **1 (ON)** | **1 (ON)** | **NOT touched by the override → LR runs by default (:286 / :1273)** |
| enable_qm | 0 | **0 (OFF)** | override sets qm_min/max=4/10 but leaves enable_qm=0 → INERT |
| screen_detection_mode | STANDARD | ANTIALIASING_AWARE(2) | override :3069 |
| deltaq_mode | OBJECTIVE(1) | OBJECTIVE(1) | TPL-gated → inert for a lone still |
| aq_mode | NO_AQ(0) | NO_AQ(0) | off |
| tuning / dist_metric | PSNR / PSNR | PSNR / PSNR | tune=IQ/SSIMULACRA2 changes 8 fields |

> **`CLAUDE.md` "Loop-restoration: OFF by default in allintra" is factually WRONG — LR is ON by
> default** (all three audits flag this independently; `PARITY.md` C2 has it right). That doc
> correction is owned by the concurrent LR-default-wiring agent (see Tier-1 #1); this synthesis
> owns the `PARITY.md`/`STATUS.md` corrections.

---

## Prioritized gap matrix

Effort: **S** ≤1 day · **M** 1–3 days · **L** multi-day (decompose). Every gap is deduped across
the three source audits; the "Evidence" column cites the authoritative source + gate/KB.

### Tier-0 — Hygiene (docs / tooling; no code) — RESOLVED this landing

| # | Gap | Status | Evidence |
|---|---|---|---|
| 0.1 | `coverage/feature_map.json` empty → coverage.py mechanical 0/349 | **FIXED** — populated from PARITY §A → **90/349** | this landing |
| 0.2 | `CLAUDE.md` "Loop-restoration: OFF by default in allintra" (WRONG — it's ON) | **owned by concurrent LR agent** (kept additive) | all 3 audits; PARITY C2 |
| 0.3 | `PARITY.md` C12 (lossless-420) reads open but is DONE (KB-5) | **CORRECTED** → Section A | control_surface Discrepancy #1 |
| 0.4 | `PARITY.md` C13 (speeds 6–9) reads open but is DONE (KB-10/11/12) | **CORRECTED** → Section A | coding_tools "known staleness" |
| 0.5 | `PARITY.md` Section A lossless row "4:2:0 still open" stale | **CORRECTED** (mono+420 both byte-exact) | control_surface Discrepancy #1 |
| 0.6 | `STATUS.md` "Next candidates" lists header/RDO/partition/intra/conformance as TODO though all landed | **CORRECTED** | encoder_modules gate posture |
| 0.7 | `STATUS.md` coverage note "0/349" | **CORRECTED** → 90/349 | this landing |

### Tier-1 — Default-parity (blocks byte-matching a plain `aomenc --allintra` stream)

| # | Gap | C ref | Status | Effort | Evidence |
|---|---|---|---|---|---|
| 1.1 | **Loop-restoration default-parity** — LR is ON by default; the byte-exact LR search is wired only behind `--enable-restoration=1`, not the default path. A default `--allintra` encode runs `av1_pick_filter_restoration` (even all-NONE changes seq/frame header bits). **Highest-value single-frame fix.** | `av1_cx_iface.c:286/1273`; `pickrst.c:2040` | search **BYTE-EXACT** (`lr_restoration_gate` 8/8 + format-axis 3/3); default path NOT wired | M | encoder_modules §E; control_surface; PARITY C2. *Being addressed by the concurrent LR agent.* |
| 1.2 | **Real-content byte-parity at cpu-used≥1** — synthetic gates 64/64 every speed, but decoded-conformance content is **24/60** byte-exact at cpu 1–4 (interior BLOCK_16X16/8X8 AB/rect/split-prune near-ties; the port under-prunes). Speed-0 real content is 30/30. | `partition_search.c` prune gates; `ab_nn_prune.rs` | PARTIAL (KB-13, task #39) | M–L | encoder_modules §B/§F; KB-13 |
| 1.3 | **Header self-derivation still bootstrapped** — base_qindex (mapping done #8, not wired into encode composition), tile-count/config choice, CICP echo, full temporal-unit / TD-OBU self-assembly. Writers all bit-exact; a wiring gap, not missing algorithms. | `bitstream.c write_uncompressed_header_obu` | PARTIAL (writers byte-exact) | M | encoder_modules §F; coding_tools §N |
| 1.4 | **level/tier computation** `av1_get_seq_level_idx` — the ONE true missing header ALGORITHM (all other header fields are bootstrap-wiring). `seq_level_idx[]`/`tier[]` echoed from the parsed real seq header. | `level.c` | **ABSENT** (bootstrapped) | M | encoder_modules §F |

### Tier-2 — Structural single-frame features (ported partially or absent)

| # | Gap | C ref | Status | Effort | Evidence |
|---|---|---|---|---|---|
| 2.1 | **SB128 encode path** — RD partition search + pack are SB-64-only (no e2e gate for a 128×128 SB root / 128→64 split / >64-block recon interleave). Decoder + entropy are already SB-generic. **Biggest structural hole.** (`--sb-size=128`) | `partition_search.c:5688` (sb_size=128) | **ABSENT** (encode); DEC byte-exact | M | all 3 audits; PARITY C8 |
| 2.2 | **`prune_tx_type_using_stats`** — luma-intra tx-type stats prune; C enables it ALLINTRA at cpu-used≥2 but only `is_480p_or_larger`. All gate frames sub-480p → unported AND unexercised. **Real hole for a ≥480p KEY frame at cpu-used≥2.** | `tx_search.c:1876` (in the `plane==0` arm, NOT is_inter-gated) | **ABSENT** | M | encoder_modules §D (unique find) |
| 2.3 | **IntraBC leaf search** — `rd_pick_intrabc_mode_sb` is a PARTIAL skeleton but **UNWIRED** (zero byte effect). ABSENT: coeff arm (SKIP-only), `min(skip,coeff)`, NSTEP diamond + mesh full-pel DV search, bd>8 sse. Hash/DV-cost/predictors/dv_ref are unit-byte-exact. (`--enable-intrabc`, screen-default-on) | `rdopt.c:3427`; `hash_motion.c`; `mcomp.c:1908` | PARTIAL (skeleton, unwired) | L | encoder_modules §C; coding_tools §F; HANDOFF-SCREEN.md |
| 2.4 | **`--quant-b-adapt`** — `aom_quantize_b_adaptive` adaptive dead-zone quantizer family (lowbd+highbd+32/64). Non-default. | `av1_quantize.c:311/455`; `quantize.c:17/174` | **ABSENT** | S–M | all 3; PARITY C9 |
| 2.5 | **AQ family** — `--aq-mode=1` VARIANCE_AQ (`av1_vaq_frame_setup`), `--aq-mode=2` COMPLEXITY_AQ, + `av1_choose_segmap_coding_method`. Single-frame-applicable segmentation-decision side; the `write_segment_id`/seg-header writers exist and are byte-exact. | `aq_variance.c`; `aq_complexity.c`; `segmentation.c` | **ABSENT** (writers present) | M | encoder_modules §E; control_surface C5 |
| 2.6 | **`--deltaq-mode=2` PERCEPTUAL (wavelet)** — `av1_compute_q_from_energy_level_deltaq_mode` + `log_block_wavelet_energy`/`haar_ac_energy`; needs `dwt.c` (Haar AC, also ABSENT). Single-frame-applicable. Modes 3/6 ARE byte-exact. | `encodeframe.c:330`; `aq_variance.c:138`; `dwt.c` | **ABSENT** | M | encoder_modules §E/§G |
| 2.7 | **deltaq-mode=3 highbd + partial-SB sub-arms** — bd8 / single-tile / dims-×8px byte-exact; bd10/12 FP-quantize arm + partial-SB source-border extension unported. | `allintra_vis.c:592` | PARTIAL | M | encoder_modules §E; PARITY C5 |
| 2.8 | **deltaq companions** — `--enable-rate-guide-deltaq` (external rate file), `--auto-intra-tools-off` (`automatic_intra_tools_off`), `--deltaq-mode=4/5`, `--delta-lf-mode` decision side (writer exists). Niche. | `allintra_vis.c:515/688/1045`; `bitstream.c` | **ABSENT** | S–M each | encoder_modules §E; control_surface C5 |
| 2.9 | **Superres AUTO/QTHRESH/RANDOM denom selection + denom-16 optimized scaler** — only FIXED denom is byte-exact. AUTO needs `analyze_hor_freq` (16×4 H_DCT, unwired) + `get_superres_denom_from_qindex_energy`; the denom-16 even-width corner trips `av1_resize_and_extend_frame`. | `superres_scale.c:184`; `resize.c` | **ABSENT** (FIXED done) | M | all 3; PARITY C6 |
| 2.10 | **Film-grain noise-model ESTIMATION** (`--denoise-noise-level`) — AR noise model + grain-param quantize + Wiener FFT denoise + `aom_denoise_and_model_run` orchestrator ABSENT (float/FFT-determinism-gated). Noise-strength solver + flat-block finder DONE (differential). Table-inject is byte-exact. | `noise_model.c` | PARTIAL (2/6 kernels) | L | all 3; PARITY C7 |
| 2.11 | **tune=vmaf / tune=butteraugli / tune=ssim** — zero port refs (vmaf needs a model; butteraugli needs its rdmult). tune=IQ/SSIMULACRA2 ARE byte-exact. | `tune_vmaf.c`; `tune_butteraugli.c` | **ABSENT** | L (vmaf/ba), M (ssim) | encoder_modules §G; control_surface C4 |
| 2.12 | **nonrd (cpu-used 8/9) HBD estimate arm + lossless TX_4X4 + screen-palette arm** — `block_yrd`/`av1_nonrd_pick_intra_mode` are 8-bit / non-lossless / non-screen only; bd10/12 + lossless + palette assert-dead. | `nonrd_pickmode.c:1582` | **ABSENT** (asserted dead) | M | encoder_modules §C/§D; KB-12 |
| 2.13 | **CDEF FAST search levels 1..5 + SB128-CDEF-on not e2e-gated** — ported + table-unit-tested; only FULL (speed-0) is e2e byte-gated (CDEF off by default). `CDEF_PICK_FROM_Q` / `CDEF_ADAPTIVE` not ported (documented dead for `--enable-cdef=1`). | `pickcdef.c` | PARTIAL | S | all 3; PARITY C1 |
| 2.14 | **cost-update-freq non-default arms** — `--coeff/mode/dv-cost-upd-freq` = default-SB arm byte-exact (multi-SB gates); non-default 1/2/3 arms not gated. | `av1_cx_iface` extra_cfg | PARTIAL | S–M | control_surface C11; coding_tools |
| 2.15 | **CICP/color-config + render-size/timing/level header echo** — `write_color_config`/`write_render_size`/`timing_info_header`/op-points are bit-exact as components, but the full self-derived echo is still bootstrapped from the parse; `--color-primaries`/`-transfer`/`-matrix`/`--chroma-sample-position`/`--color-range`/`--render-size`/`--timing-info`/`--target-seq-level-idx`/`--set-tier-mask` not gated with non-default values. | `bitstream.c write_color_config`, seq/frame header | PARTIAL (components byte-exact) | S–M | control_surface C11; coding_tools §N/§Q |
| 2.16 | **Framing / clamp arms** — `--full-still-picture-hdr`, `--annexb`, `--large-scale-tile`, `--force-video-mode`, `--error-resilient=1` encode arm, `--min-q`/`--max-q`/`--min-cr`; **highbd lossless (bd10/12)** (mono+420 bd8 done). Non-default / small. | seq/frame framing; `av1_quantize.c` | **ABSENT** (writers exist) | S each | control_surface C11/C12; coding_tools §R |

### Tier-3 — Pinned near-ties (open byte-parity residuals; each a sibling-C RD dump away)

These are **not** coverage gaps — the tool is ported and the machinery is C-faithful. Each is a
single RD near-tie the port loses; every one is regression-guarded by a self-promoting test that
asserts the divergence PRESENT (a fix flips the test → promote to a byte-match gate).

| # | Gap | Divergence | Status | Evidence |
|---|---|---|---|---|
| 3.1 | `--use-intra-dct-only=1` chroma UV-mode-loop (64²cq32) | first leaf mi(0,0) 32×32: real uv=D45 vs port uv=V; port's `txfm_rd_in_plane_uv_p` accepts a V/DCT dist=0 where C rejects it | PINNED-OPEN | PARITY §B (C9); encoder_modules §C/§D |
| 3.2 | Palette 128² AB/4-way (`ui_420_128_cq32`, `text_420_128_cq20`) | palette-induced partition near-tie (real HORZ_B/VERT vs port HORZ_4/VERT_A); byte-exact with palette OFF, machinery C-faithful | PINNED (5/7 byte-exact) | KB-P29; PARITY §B |
| 3.3 | speed-6/7 noise-cq63 (mi 8,0) 32×32 | WINNER-pass tx-size sweep picks TX_16X16 over TX_32X32 by 0.19% rd; multi-feature interaction at qindex 255 | PINNED (canon 64/64) | KB-10/KB-11 |
| 3.4 | speed-8 4 `diag` estimate-arm cells | `av1_nonrd_pick_intra_mode` picks V_PRED vs real H_PRED at ~0.7% rdcost | PINNED (60/64 canon) | KB-12 |

---

## What is byte-exact (positive ledger — pointer)

The full byte-identity ledger is `PARITY.md` Section A; the module-level detail is in the three
source audits. In brief, **BYTE-EXACT both ways** (enc emits == `aomenc`, dec == C):

- **DSP kernels:** forward+inverse transform (all 19 sizes × 16 types + WHT + 64-pt repack), quant
  (FP/B/DC × lowbd/highbd × flat/QM), od_ec range coder + CDF adaptation + default CDFs, intra
  predictors (non-dir + directional z1/z2/z3 + edge + filter-intra + CfL, lowbd+highbd), distortion
  (SAD/variance/subpel/Hadamard/SATD/block_error, lowbd+highbd), CDEF find_dir + filter, deblock
  4/6/8/14-tap, txb coeff-coding + trellis + cost. (encoder_modules §A)
- **Encode search:** partition (all 10 types, ≤64 SB) + all partition-level ML prunes; intra mode
  search (luma + chroma + angle-delta + filter-intra + CfL + HOG); tx-type + tx-size search +
  trellis; the full speed-0..9 pipeline (synthetic 64/64, real speed-0 30/30). (§B/§C/§D)
- **Post-filter search:** LF-level, CDEF-strength (`--enable-cdef=1`), loop-restoration
  (`--enable-restoration=1`), deltaq-mode 3/6. (§E)
- **Headers/OBU:** seq-header OBU (vs the REAL exported C fn), frame-header OBU, tile-info,
  tile-group OBU, leb128, quant/seg/delta-q/LF/CDEF/LR/tx_mode/film-grain header blocks,
  color_config. (§F)
- **Format + config:** 4:0:0/4:2:0/4:2:2/4:4:4 × 8/10/12-bit, multi-tile, coded-lossless
  (mono+420), QM-on, superres-FIXED, film-grain table-inject, tune=IQ/SSIMULACRA2, and the
  C8/C9/C10/C11 CLI-toggle disable arms.
- **Decoder:** Gate-1 conformance corpus (intra scope, incl. q62/q63 — KB-1 fixed) + real-bitstream
  KEY envelope (deblock/CDEF/LR/superres/SB128/lossless/QM/multi-tile/palette/intrabc/
  disable-cdf-update/4:2:2 chroma deblock).

---

## Provenance

- **Synthesis author:** aom-rs coverage-audit consolidation, 2026-07-18. Report + doc-hygiene only
  (no encoder/decoder source changed).
- **Sources:** the three sub-audits in this directory (each self-provenanced); `PARITY.md`
  Section A/B/C; `CLAUDE.md` Known-Bugs (KB-1…KB-13, KB-P29); `STATUS.md`; `coverage/checklist.json`
  (kernel-level differential ledger) + `coverage/feature_map.json` (CLI/control byte-identity map,
  populated this landing) + `xtask/coverage.py` (90/349).
- **Method:** merge the per-surface audits, dedupe each gap to a single row, tier by
  actionability, cross-check every "byte-exact" against a landed gate. Where in-repo docs were
  stale (PARITY C12/C13 + Section-A lossless row, STATUS "Next candidates" + coverage note), the
  source + gates were treated as ground truth and the docs corrected (Tier-0).
