# aom-rs control/CLI surface coverage audit — 2026-07-18

**Scope:** the libaom v3.14.1 *encoder* control/CLI surface vs the aom-rs port, for the
**single-frame / ALLINTRA (usage=2) / KEY-frame** mission. Inter-frame ("#11 THE REST") is
explicitly deferred: inter/video-only knobs are classed `OUT-OF-SCOPE-inter` and are **not**
counted as gaps.

**Method (evidence-based, not doc-trusted):** the C surface was enumerated from
`reference/libaom/av1/av1_cx_iface.c` (`encoder_ctrl_maps` :4927–5104, `default_extra_cfg`
:253–406 [the compiled `!CONFIG_REALTIME_ONLY` branch], the allintra override :3065–3078,
`handle_tuning` :1938–1978) and `reference/libaom/build/aomenc --help` (180 CLI flags) +
`aom/aomcx.h` (147 `AV1E_SET_*`/`AOME_SET_*` enums). Each was cross-checked against the port's
landed gates in `PARITY.md` Section A/B/C, the `CLAUDE.md` KB log, and the on-disk test bodies
(`crates/aom-*/tests/*`). Where docs conflicted, the **test source was treated as ground truth**
(see Discrepancies).

**Classes:** `BYTE-EXACT-GATED` (ported + a byte-identity e2e gate) · `PORTED-NOT-GATED` ·
`PARTIAL` (some arms/bitdepths/modes done) · `ABSENT` · `OUT-OF-SCOPE-inter` ·
`INERT/DEFAULT-OFF` (off/neutral in the allintra envelope, verified inert). Application-I/O,
getters, and debug-only controls are marked `N/A (non-codec)` and excluded from the
single-frame denominator.

---

## SUMMARY

Of the **single-frame-relevant** control/CLI surface (≈133 distinct knobs, excluding N/A
application-I/O + getters + debug + SVC/RC-container):

**52 BYTE-EXACT-GATED · 1 PORTED-NOT-GATED · 21 PARTIAL · 26 ABSENT · 8 INERT/DEFAULT-OFF · 25 OUT-OF-SCOPE-inter.**

The **primary allintra speed-0 KEY envelope is byte-exact** (mode / dims / bd 8·10·12 /
mono·420·422·444 / cq·qindex / tiles / LF), and every post-filter + tune + partition/tx/intra
toggle family (C1,C2,C4,C5-mode3/6,C6-fixed,C7-table,C8,C9,C10,C11) has landed a byte-identity
gate. The actionable single-frame gaps are the **PARTIAL** (superres AUTO/QTHRESH, palette
2-cell near-tie, intrabc skeleton, use-intra-dct-only chroma, cost-upd non-default arms, film-
grain estimation, CICP/render/level header echo, SB128 encode, loopfilter-control arms) and
**ABSENT** (aq-mode, deltaq 4/5, tune=ssim/vmaf/butteraugli, screen-detection, tune-content,
auto-intra-tools-off, rate-guide-deltaq, min/max-q, min-cr, full-still-picture-hdr, annexb,
resize-mode, quant-b-adapt, superres qthresh) lists below.

> **Coverage-gate note:** `xtask/coverage.py` reports **0/349 green** — this is a *mechanical
> artifact*, NOT the real status. `coverage/feature_map.json` is `{}` (empty), so no feature is
> mapped to a test regardless of what shipped. The real byte-exact ledger is `PARITY.md`
> Section A (this audit's evidence base); kernel-level differential coverage is 19 green modules
> in `coverage/checklist.json`. **Populating `feature_map.json` from PARITY.md Section A would
> move the gate off 0/349 and is the single highest-leverage coverage-hygiene fix.**

---

## Envelope defaults (verified against reference/libaom)

Base `default_extra_cfg` (:253, non-realtime) then the allintra override (:3065). "single-frame
exact" means matching the **allintra** defaults, not the base defaults:

| Knob | base default | allintra effective | note |
|---|---|---|---|
| enable_cdef | 1 | **0 (OFF)** | override :3067 ("CDEF blurs images") |
| enable_restoration | **1 (ON)** | 1 (ON) | NOT touched by override → LR runs by default |
| enable_qm | 0 | **0 (OFF)** | override sets qm_min/max=4/10 but leaves enable_qm=0 → INERT |
| screen_detection_mode | STANDARD | ANTIALIASING_AWARE(2) | override :3069 |
| deltaq_mode | OBJECTIVE(1) | OBJECTIVE(1) | TPL-gated (encodeframe.c:343) → **inert for a lone still** |
| aq_mode | NO_AQ(0) | NO_AQ(0) | off |
| tuning / dist_metric | PSNR / PSNR | PSNR / PSNR | tune=IQ/SSIMULACRA2 changes 8 fields (handle_tuning :1938) |
| disable_trellis_quant | 3 (estimate-yrd) | 3 | |
| loopfilter_control | LOOPFILTER_ALL(1) | 1 | LF runs |
| enable_{rect,ab,1to4}_partitions, tx64, rect_tx, flip_idtx, intra tools, angle_delta, palette, intrabc | 1 | 1 | screen tools gated on `allow_screen_content_tools` |

---

## Master classification table

### 1 — Primary envelope + core coding config (single-frame heart)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--allintra` / `--usage=2` | AOM_USAGE_ALL_INTRA | **BYTE-EXACT-GATED** | the whole mission envelope; `encoder_gate_e2e_byte_match` (76b1ffb) |
| `--cpu-used=0..9` | AOME_SET_CPUUSED | **BYTE-EXACT-GATED** | speed 0-9 sweep `encoder_gate_speed{1..9}_textured_allintra` (KB-8..12); Gate-2 COMPLETE. *Pins:* speed-8 4 diag near-ties (KB-12), speed-6/7 noise-cq63 (KB-10/11), real-content speed≥1 36/60 (KB-13) |
| `--cq-level` / `--end-usage=cq,q` | AOME_SET_CQ_LEVEL | **BYTE-EXACT-GATED** | fixed-Q path; qindex-from-cq #8 `qindex_from_cq_diff`; low-q `encoder_gate_e2e_low_qindex_speed0` (ec5905c) |
| `-w/--width`, `-h/--height` | (cfg) | **BYTE-EXACT-GATED** | all gates encode explicit dims incl. partial-SB 196² (KB-6) |
| `-b/--bit-depth=8/10/12` | (cfg) | **BYTE-EXACT-GATED** | `encoder_gate_bd10_diff` (20f1e70), `kb4_gate_bd10_bd12_mono_hf_byte_match` (a2dd28e) |
| `--monochrome` | (cfg) | **BYTE-EXACT-GATED** | mono asserted across every gate family |
| `--i420 / --i422 / --i444` | AV1E_SET_CHROMA_SUBSAMPLING_X/Y | **BYTE-EXACT-GATED** | `encoder_gate_chroma_ss_e2e` (2ee900d) 4:2:2/4:4:4; 4:2:0 default |
| `--tile-columns` | AV1E_SET_TILE_COLUMNS | **BYTE-EXACT-GATED** | `encoder_gate_multitile_e2e` (f6e6319) 2×1/1×2/2×2 |
| `--tile-rows` | AV1E_SET_TILE_ROWS | **BYTE-EXACT-GATED** | same |
| `--lossless` (cq0/qindex0) | AV1E_SET_LOSSLESS | **BYTE-EXACT-GATED** | `encoder_gate_lossless_cq0_e2e_kb5_repro` — **mono AND 4:2:0 bd8** both hard-asserted (ba560eb + KB-5 420 fix). *Gap:* highbd lossless |
| `--profile` | (derived) | **PORTED-NOT-GATED** | `write_profile` in `write_sequence_header_obu` bit-exact (`seq_header_matches_real_encoder`); implied by bd/subsampling gates, not varied by a dedicated gate |
| `--sb-size=64` | AV1E_SET_SUPERBLOCK_SIZE | **BYTE-EXACT-GATED** | the whole envelope is SB64 |
| `--sb-size=128` | AV1E_SET_SUPERBLOCK_SIZE | **PARTIAL/ABSENT** | decoder+entropy SB-generic (798ec25); **encoder walk SB64-only** (C8) |

### 2 — Post-reconstruction filters

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--enable-cdef=1` | AV1E_SET_ENABLE_CDEF | **BYTE-EXACT-GATED** | C1, `av1_cdef_search` `pickcdef.rs`; `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close` 14/14 (016d4dd+9850da6+c9ebf83). OFF by default in allintra. *Gap:* FAST_LVL1..5 ported but not e2e-gated; `=2`(non-ref) inert on KEY; `=3` CDEF_ADAPTIVE not ported |
| `--enable-restoration=1` | AV1E_SET_ENABLE_RESTORATION | **BYTE-EXACT-GATED** | C2, Wiener/SGR search; `lr_restoration_gate` 8/8 + format-axis 3/3 (e24cf09+96d3464+dfd757e+96534c4). **ON by default in allintra.** *Gap:* speed-1..4 LR arms pinned (base encode not byte-exact at speed≥1) |
| `--loopfilter-control` | AV1E_SET_LOOPFILTER_CONTROL | **PARTIAL** | default `=1`(ALL) byte-exact (LF-level derivation in every e2e gate); `=0`(disable) not gated; `=2/3`(non-ref/low-motion) inter → inert on lone KEY |

### 3 — Tune / quality-metric family (C4)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--tune=psnr` | AOME_SET_TUNING | **BYTE-EXACT-GATED** | default; = the primary envelope |
| `--tune=iq` | AOME_SET_TUNING | **BYTE-EXACT-GATED** | C4 bundle, `encoder_gate_tune_iq_e2e` (9 tests) + composite 54/54; OFF by default |
| `--tune=ssimulacra2` | AOME_SET_TUNING | **BYTE-EXACT-GATED** | same bundle (minus adaptive-sharpness) |
| `--sharpness=0..7` | AOME_SET_SHARPNESS | **BYTE-EXACT-GATED** | C4; `av1_build_quantizer` bias + trellis + LF; witnessed |
| `--enable-adaptive-sharpness` | AV1E_SET_ENABLE_ADAPTIVE_SHARPNESS | **BYTE-EXACT-GATED** | C4; qindex-adaptive LF sharpness cap; witnessed |
| `--enable-chroma-deltaq` | AV1E_SET_ENABLE_CHROMA_DELTAQ | **BYTE-EXACT-GATED** | C4; chroma delta-q arms |
| `--dist-metric=psnr,qm-psnr` | (extra_cfg.dist_metric) | **BYTE-EXACT-GATED** | C4 QM-PSNR trellis+tx-domain dist; `tune_shim_smoke` |
| `--tune=ssim` | AOME_SET_TUNING | **ABSENT** | SSIM-rdmult scaling not ported |
| `--tune=vmaf*` (5 variants) | AOME_SET_TUNING / AV1E_SET_VMAF_MODEL_PATH | **ABSENT** | needs VMAF model; not ported |
| `--tune=butteraugli` | AOME_SET_TUNING | **ABSENT** | not ported |

### 4 — Delta-Q / AQ family (C5)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--deltaq-mode=3` (PERCEPTUAL_AI) | AV1E_SET_DELTAQ_MODE | **BYTE-EXACT-GATED** | C5; `av1_set_mb_wiener_variance`+`av1_get_deltaq_offset`; `encoder_gate_deltaq_mode3_e2e` 7/7. *Gap:* highbd + partial-SB + rate-guide + auto-intra-tools-off |
| `--deltaq-mode=6` (VARIANCE_BOOST) | AV1E_SET_DELTAQ_MODE | **BYTE-EXACT-GATED** | landed with C4 (fed362b); `av1_get_sbq_variance_boost` |
| `--deltaq-strength` | AV1E_SET_DELTAQ_STRENGTH | **BYTE-EXACT-GATED** | for mode 4/6; covered under mode-6/tune |
| `--deltaq-mode=1` (OBJECTIVE, default) | AV1E_SET_DELTAQ_MODE | **INERT/DEFAULT-OFF** | TPL-gated (encodeframe.c:343) → inert for a lone still |
| `--deltaq-mode=2` (placeholder) | AV1E_SET_DELTAQ_MODE | **INERT** | placeholder, no effect |
| `--deltaq-mode=4` (user-rating) / `=5` (HDR) | AV1E_SET_DELTAQ_MODE | **ABSENT** | not ported |
| `--aq-mode=1/2/3` | AV1E_SET_AQ_MODE | **ABSENT** | aq_variance/aq_complexity/cyclic; needs two-pass to fire (C5, M low-pri) |
| `--delta-lf-mode` | AV1E_SET_DELTALF_MODE | **ABSENT** | delta loop filter (C5, S–M) |
| `--auto-intra-tools-off` | AV1E_SET_AUTO_INTRA_TOOLS_OFF | **ABSENT** | needs deltaq-mode=3; `automatic_intra_tools_off`+model_rd_sse (C5) |
| `--enable-rate-guide-deltaq` / `--rate-distribution-info` | AV1E_ENABLE_RATE_GUIDE_DELTAQ / AV1E_SET_RATE_DISTRIBUTION_INFO | **ABSENT** | needs deltaq-mode=3 + external rate file (C5) |

### 5 — Superres (encode side) (C6)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--superres-mode=1` (fixed) + `--superres-denominator` | AV1E_SET_ENABLE_SUPERRES (+ cfg) | **BYTE-EXACT-GATED** | C6; source downscale `av1_resize_plane`/`highbd_resize_plane` + `write_superres_scale`; `encoder_gate_superres_fixed_{real_content,mono,highbd}_rd_close` — **bd8 13/13 + bd10/12 16/16** (2505b49f+68703b1+79e7a6d). OFF by default |
| `--superres-kf-denominator` | (cfg) | **PARTIAL** | KEY-frame denom path covered by the fixed gate's KEY encodes; distinct knob not separately celled |
| `--superres-mode=2/3/4` (random/qthresh/auto) | AV1E_SET_ENABLE_SUPERRES | **ABSENT** | denom-derivation (`calculate_next_superres_scale`, `analyze_hor_freq`, recode loop) not ported (C6) |
| `--superres-qthresh` / `--superres-kf-qthresh` | (cfg) | **ABSENT** | qthresh-mode denom derivation (C6) |
| 8-bit denom-16-even-width corner | — | **ABSENT** | trips optimized `av1_resize_and_extend_frame`; gate asserts-out of it (C6 follow-up) |

### 6 — Film grain / denoise (C7)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--film-grain-table` | AV1E_SET_FILM_GRAIN_TABLE | **BYTE-EXACT-GATED** | C7; own reader `grain_table.rs`→`write_film_grain_params`; `film_grain_table_inject_{420_real,format_axes}` + no-leak witness |
| `--film-grain-test=1..16` | AV1E_SET_FILM_GRAIN_TEST_VECTOR | **BYTE-EXACT-GATED (transitive)** | the 16 vectors are the fixture source for the table-inject gate; shared param-plumbing + writer |
| `--denoise-noise-level` / `--denoise-block-size` / `--enable-dnl-denoising` | AV1E_SET_DENOISE_NOISE_LEVEL / _BLOCK_SIZE / _ENABLE_DNL_DENOISING | **PARTIAL** | C7 estimation: `noise_strength_solver` + `flat_block_finder` DONE (differential-locked); `noise_model`(AR) + `wiener_denoise_2d`(FFT) + orchestrator ABSENT (float-determinism-gated, L) |

### 7 — Partition controls (C8)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--enable-rect-partitions=0` | AV1E_SET_ENABLE_RECT_PARTITIONS | **BYTE-EXACT-GATED** | `toggles_c8_rect_partitions_off` (hard `bit_identical`) |
| `--enable-ab-partitions=0` | AV1E_SET_ENABLE_AB_PARTITIONS | **BYTE-EXACT-GATED** | `toggles_c8_ab_partitions_off` |
| `--enable-1to4-partitions=0` | AV1E_SET_ENABLE_1TO4_PARTITIONS | **BYTE-EXACT-GATED** | `toggles_c8_1to4_partitions_off` |
| `--min-partition-size` | AV1E_SET_MIN_PARTITION_SIZE | **BYTE-EXACT-GATED** | `toggles_c8_min_partition_16` + square-only band |
| `--max-partition-size` | AV1E_SET_MAX_PARTITION_SIZE | **BYTE-EXACT-GATED** | `toggles_c8_max_partition_32` |
| `--external-partition` / `--partition-info-path` / `--sb-qp-sweep` | AV1E_SET_EXTERNAL_PARTITION / _PARTITION_INFO_PATH / AV1E_ENABLE_SB_QP_SWEEP | **ABSENT** | diagnostic, lowest priority (C8, defer) |
| `--auto-tiles` | AV1E_SET_AUTO_TILES | **ABSENT** | auto tile derivation (tiles themselves gated) |
| `--num-tile-groups` (>1) / `--mtu-size` | AV1E_SET_NUM_TG / AV1E_SET_MTU | **ABSENT** | multi-tile-group OBU split (default 1 covered) |

### 8 — Transform controls (C9)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--enable-tx64=0` | AV1E_SET_ENABLE_TX64 | **BYTE-EXACT-GATED** | `toggles_c9_tx64_off` |
| `--enable-rect-tx=0` | AV1E_SET_ENABLE_RECT_TX | **BYTE-EXACT-GATED** | `toggles_c9_rect_tx_off` |
| `--enable-flip-idtx=0` | AV1E_SET_ENABLE_FLIP_IDTX | **BYTE-EXACT-GATED** | `toggles_c9_flip_idtx_off` |
| `--use-intra-default-tx-only=1` | AV1E_SET_INTRA_DEFAULT_TX_ONLY | **BYTE-EXACT-GATED** | `toggles_c9_intra_default_tx_only` |
| `--reduced-tx-type-set=1` | AV1E_SET_REDUCED_TX_TYPE_SET | **BYTE-EXACT-GATED** | `toggles_c9_reduced_tx_type_set` |
| `--enable-tx-size-search=0` | AV1E_SET_ENABLE_TX_SIZE_SEARCH | **BYTE-EXACT-GATED** | `toggles_c9_tx_size_search_off` (USE_LARGESTALL route) |
| `--disable-trellis-quant=1,2` | AV1E_SET_DISABLE_TRELLIS_QUANT | **BYTE-EXACT-GATED** | `toggles_c9_trellis_quant_off`/`_final_pass_only` (5a644c6); default `=3` proven |
| `--use-intra-dct-only=1` | AV1E_SET_INTRA_DCT_ONLY | **PARTIAL** | luma byte-faithful; **chroma UV-mode-loop divergence out of band** (Section B PINNED-OPEN, `toggles_c9_intra_dct_only_pinned_open`); sibling-C dump localized to a UV DCT tx-search RD accept/reject mis-model |
| `--quant-b-adapt` | AV1E_SET_QUANT_B_ADAPT | **ABSENT** | adaptive quantize_b family (C9, S–M) |
| `--use-inter-dct-only` | AV1E_SET_INTER_DCT_ONLY | **OUT-OF-SCOPE-inter** | inter modes |

### 9 — Intra mode toggles (C10) — all 8 BYTE-EXACT-GATED

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--enable-smooth-intra=0` | AV1E_SET_ENABLE_SMOOTH_INTRA | **BYTE-EXACT-GATED** | `toggles_c10_smooth_intra_off` |
| `--enable-paeth-intra=0` | AV1E_SET_ENABLE_PAETH_INTRA | **BYTE-EXACT-GATED** | `toggles_c10_paeth_intra_off` |
| `--enable-cfl-intra=0` | AV1E_SET_ENABLE_CFL_INTRA | **BYTE-EXACT-GATED** | `toggles_c10_cfl_intra_off` |
| `--enable-directional-intra=0` | AV1E_SET_ENABLE_DIRECTIONAL_INTRA | **BYTE-EXACT-GATED** | `toggles_c10_directional_intra_off` |
| `--enable-diagonal-intra=0` | AV1E_SET_ENABLE_DIAGONAL_INTRA | **BYTE-EXACT-GATED** | `toggles_c10_diagonal_intra_off` |
| `--enable-angle-delta=0` | AV1E_SET_ENABLE_ANGLE_DELTA | **BYTE-EXACT-GATED** | `toggles_c10_angle_delta_off` |
| `--enable-filter-intra=0` | AV1E_SET_ENABLE_FILTER_INTRA | **BYTE-EXACT-GATED** | `toggles_c10_filter_intra_off` (seq bit asserted) |
| `--enable-intra-edge-filter=0` | AV1E_SET_ENABLE_INTRA_EDGE_FILTER | **BYTE-EXACT-GATED** | `toggles_c10_intra_edge_filter_off` (seq bit asserted) |

### 10 — Quant matrices (#23)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--enable-qm=1` | AV1E_SET_ENABLE_QM | **BYTE-EXACT-GATED** | #23; `qm_encode_witness` 40 cells bd8+bd10 (5b512bf). OFF by default (enable_qm=0 in allintra) |
| `--qm-min` / `--qm-max` | AV1E_SET_QM_MIN / _MAX | **BYTE-EXACT-GATED** | level selection via `aom_get_qmlevel_allintra`; exercised in qm gate (both ranges) |
| `--qm-y` / `--qm-u` / `--qm-v` | AV1E_SET_QM_Y / _U / _V | **PARTIAL/INERT** | per-plane level; standard path derives level from qm_min/max, direct-set not exercised |

### 11 — Bitstream / global (C11)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--cdf-update-mode=0` | AV1E_SET_CDF_UPDATE_MODE | **BYTE-EXACT-GATED** | `toggles_c11_cdf_update_mode_0` (fixed a real pack bug: writer-side `allow_update_cdf` gate); default `=1` proven; `=2` selective inert on lone KEY |
| bd 8/10/12, mono, 4:2:0/4:2:2/4:4:4, tiles, seq+frame OBU writers | — | **BYTE-EXACT-GATED** | seq/frame header components all bit-exact (`seq_header_matches_real_encoder`, `frame_header_matches_real_encoder`) |
| `--coeff-cost-upd-freq` | AV1E_SET_COEFF_COST_UPD_FREQ | **PARTIAL** | default SB arm byte-exact (multi-SB gates); non-default 1/2/3 arms not gated (C11, S–M) |
| `--mode-cost-upd-freq` | AV1E_SET_MODE_COST_UPD_FREQ | **PARTIAL** | same |
| `--dv-cost-upd-freq` | AV1E_SET_DV_COST_UPD_FREQ | **PARTIAL** | default arm ok; DV path is screen/intrabc-gated |
| `--mv-cost-upd-freq` | AV1E_SET_MV_COST_UPD_FREQ | **INERT/DEFAULT-OFF** | no MV on KEY → inert |
| `--color-primaries` / `--transfer-characteristics` / `--matrix-coefficients` | AV1E_SET_COLOR_PRIMARIES / _TRANSFER_CHARACTERISTICS / _MATRIX_COEFFICIENTS | **PARTIAL** | `write_color_config` bit-exact as a component; full self-derived CICP echo still bootstrapped (C11 caveat) |
| `--chroma-sample-position` | AV1E_SET_CHROMA_SAMPLE_POSITION | **PARTIAL** | color_config component present; echo bootstrapped |
| `--color-range` (control-only) | AV1E_SET_COLOR_RANGE | **PARTIAL** | seq color_config component; not gated with non-default |
| `--render-size` (control-only) | AV1E_SET_RENDER_SIZE | **PARTIAL** | `write_render_size` component bit-exact; not gated with non-default render size |
| `--timing-info` | AV1E_SET_TIMING_INFO_TYPE | **PARTIAL** | `timing_info_header` in seq OBU bit-exact; not gated with timing on |
| `--target-seq-level-idx` / `--set-tier-mask` / `--strict-level-conformance` | AV1E_SET_TARGET_SEQ_LEVEL_IDX / _TIER_MASK / (cfg) | **PARTIAL** | seq operating-points component bit-exact; level auto-derivation/enforcement absent (C11) |
| `--min-q` / `--max-q` | (cfg) | **ABSENT** | quantizer clamps (C11, S) |
| `--min-cr` | AV1E_SET_MIN_CR | **ABSENT** | min compression ratio (C11, S) |
| `--full-still-picture-hdr` | (cfg) | **ABSENT** | **single-frame-relevant**; still-picture full header framing (C11, S) |
| `--annexb` | (cfg) | **ABSENT** | Annex-B framing (C11, S) |
| `--force-video-mode` | AV1E_SET_FORCE_VIDEO_MODE | **ABSENT** | disables still-picture header opt (single-frame relevant, low) |
| `--error-resilient` | AV1E_SET_ERROR_RESILIENT_MODE | **ABSENT (arm)** | header writer supports the bit; no e2e gate with it on |

### 12 — Screen content (C3)

| flag / control | C ref | class | evidence / gap |
|---|---|---|---|
| `--enable-palette` | AV1E_SET_ENABLE_PALETTE | **PARTIAL** | C3; Y+UV RD search (Section B) **5/7 byte-exact**, 2 128² cells PINNED (palette-induced AB/4-way partition near-tie, KB-P29); `rd_close_palette` |
| `--enable-intrabc` | AV1E_SET_ENABLE_INTRABC | **PARTIAL** | chunk 3a/3b landed (hash-table, DV costs, predictors) but **skeleton UNWIRED / envelope-inert**; coeff arm + NSTEP search + 8-step integration ABSENT (C3, L; see HANDOFF-SCREEN.md) |
| `--screen-detection-mode` | AV1E_SET_SCREEN_CONTENT_DETECTION_MODE | **ABSENT** | port takes `allow_screen_content_tools` as input; detection (`av1_set_screen_content_options`) unported (C3, S–M) |
| `--tune-content=screen,film` | AV1E_SET_TUNE_CONTENT | **ABSENT** | content-force gating (C3, S) |

### 13 — Multi-frame / inter / rate-control (OUT-OF-SCOPE-inter)

| flag / control group | C ref | class |
|---|---|---|
| `--good`, `--rt` (video deadline modes) | (cfg) | **OUT-OF-SCOPE-inter** |
| `--passes`/`--pass`/`--fpf`/`--two-pass-output`/`--second-pass-log` (multi-pass) | (cfg) | **OUT-OF-SCOPE-inter** |
| `--lag-in-frames` | (cfg) | **OUT-OF-SCOPE-inter** |
| `--end-usage=vbr,cbr`, `--target-bitrate`, `--{under,over}shoot-pct`, `--buf-*`, `--bias-pct`, `--{min,max}section-pct`, `--max-intra-rate`, `--max-inter-rate`, `--gf-cbr-boost`, `--vbr-corpus-complexity-lap` | AV1E_SET_MAX_INTER_BITRATE_PCT / AV1E_SET_GF_CBR_BOOST_PCT / AOME_SET_MAX_INTRA_BITRATE_PCT / AV1E_SET_VBR_CORPUS_COMPLEXITY_LAP | **OUT-OF-SCOPE-inter** (rate-control) |
| `--drop-frame`, `--resize-mode/-denominator/-kf-denominator` (RC resize) | (cfg) | **OUT-OF-SCOPE-inter** (resize = ABSENT if ever wanted for stills) |
| `--enable-fwd-kf`, `--kf-min-dist`, `--kf-max-dist`, `--disable-kf`, `--fwd-kf-dist`, `--kf-max-pyr-height` | (cfg) | **OUT-OF-SCOPE-inter** (keyframe placement / GOP) |
| `--sframe-dist`, `--sframe-mode` | AV1E_SET_S_FRAME_MODE | **OUT-OF-SCOPE-inter** (S-frames) |
| `--auto-alt-ref`, `--arnr-maxframes`, `--arnr-strength`, `--enable-keyframe-filtering` | AOME_SET_ENABLEAUTOALTREF / _ARNR_MAXFRAMES / _ARNR_STRENGTH / AV1E_SET_ENABLE_KEYFRAME_FILTERING | **OUT-OF-SCOPE-inter** (alt-ref / temporal filtering) |
| `--static-thresh`, `--enable-tpl-model` | AOME_SET_STATIC_THRESHOLD / AV1E_SET_ENABLE_TPL_MODEL | **OUT-OF-SCOPE-inter** (motion / TPL) |
| `--enable-dual-filter`, `--enable-order-hint`, `--enable-dist-wtd-comp`, `--enable-masked-comp`, `--enable-onesided-comp`, `--enable-interintra-comp`, `--enable-smooth-interintra`, `--enable-diff-wtd-comp`, `--enable-interinter-wedge`, `--enable-interintra-wedge`, `--enable-global-motion`, `--enable-warped-motion`, `--allow-warped-motion`, `--enable-obmc`, `--enable-overlay` | AV1E_SET_ENABLE_DUAL_FILTER / _ORDER_HINT / _DIST_WTD_COMP / _MASKED_COMP / _ONESIDED_COMP / _INTERINTRA_COMP / _SMOOTH_INTERINTRA / _DIFF_WTD_COMP / _INTERINTER_WEDGE / _INTERINTRA_WEDGE / _GLOBAL_MOTION / _WARPED_MOTION / ALLOW_WARPED_MOTION / _OBMC / _OVERLAY | **OUT-OF-SCOPE-inter** (inter tools; seq-header bits covered by bit-exact seq header) |
| `--max-reference-frames`, `--reduced-reference-set`, `--enable-ref-frame-mvs`, `--allow-ref-frame-mvs`, `--{min,max}-gf-interval`, `--gf-{min,max}-pyr-height` | AV1E_SET_MAX_REFERENCE_FRAMES / _REDUCED_REFERENCE_SET / _ENABLE_REF_FRAME_MVS / _ALLOW_REF_FRAME_MVS / _MIN_GF_INTERVAL / _MAX_GF_INTERVAL / _GF_MIN_PYRAMID_HEIGHT / _GF_MAX_PYRAMID_HEIGHT | **OUT-OF-SCOPE-inter** |
| `--global-error-resilient`, `--frame-parallel`, `--frame-boost`, `--noise-sensitivity`, `--enable-low-complexity-decode` | AV1E_SET_FRAME_PARALLEL_DECODING / _FRAME_PERIODIC_BOOST / _NOISE_SENSITIVITY / _ENABLE_LOW_COMPLEXITY_DECODE | **OUT-OF-SCOPE-inter** |
| SVC: `AOME_SET_SPATIAL_LAYER_ID`, `AOME_SET_NUMBER_SPATIAL_LAYERS`, `AV1E_SET_SVC_LAYER_ID/_PARAMS/_REF_FRAME_CONFIG/_REF_FRAME_COMP_PRED/_FRAME_DROP_MODE` | (SVC) | **OUT-OF-SCOPE-inter** |
| RTC/RC-container: `AV1E_SET_RTC_EXTERNAL_RC`, `_EXTERNAL_RATE_CONTROL`, `_QUANTIZER_ONE_PASS`, `_BITRATE_ONE_PASS_CBR`, `_MAX_CONSEC_FRAME_DROP_CBR/_MS_CBR`, `_POSTENCODE_DROP_RTC` | (RTC) | **OUT-OF-SCOPE-inter** |
| Reference/map injection: `AV1_SET_REFERENCE`, `AOME_USE_REFERENCE`, `AV1_COPY_REFERENCE`, `AOME_SET_ACTIVEMAP`, `AOME_SET_SCALEMODE`, `AOME_SET_ROI_MAP` | (inter/RT) | **OUT-OF-SCOPE-inter** (ROI could be a stills seg feature — ABSENT if ever wanted) |

### 14 — Threading / pipeline / input hints (INERT/DEFAULT-OFF — bitstream-neutral)

| flag / control | C ref | class | note |
|---|---|---|---|
| `--threads` | (cfg) | **INERT/DEFAULT-OFF** | AV1 MT is bit-exact; oracle build CONFIG_MULTITHREAD=0 |
| `--row-mt` | AV1E_SET_ROW_MT | **INERT/DEFAULT-OFF** | bitstream-neutral |
| `--fp-mt` | AV1E_SET_FP_MT | **INERT/DEFAULT-OFF** | frame-parallel MT, neutral |
| `--use-16bit-internal` | (init flag) | **INERT/DEFAULT-OFF** | internal pipeline only |
| `--validate-hbd-input` | AOME_SET_VALIDATE_HBD_INPUT | **INERT/DEFAULT-OFF** | input range check, no bitstream effect |
| `--input-chroma-subsampling-x/-y` | (cfg) | **INERT/DEFAULT-OFF** | maps to subsampling (covered by chroma_ss gates) |
| `--skip-postproc-filtering` | AV1E_SET_SKIP_POSTPROC_FILTERING | **INERT/DEFAULT-OFF** | external-recon niche |
| `AV1E_SET_ENABLE_DIST_8X8` | aomcx.h enum | **INERT/DEPRECATED** | in the 147-enum count but NOT in `encoder_ctrl_maps` → removed/no-op |

### 15 — N/A (non-codec: application-I/O, getters, debug — excluded from denominator)

- **Application/IO/container:** `--help`, `-c/--cfg`, `-D/--debug`, `-o/--output`, `--codec`, `-q/--quiet`, `-v/--verbose`, `--psnr`, `--webm`, `--ivf`, `--obu`, `--q-hist`, `--rate-hist`, `--disable-warnings`, `-y/--disable-warning-prompt`, `--test-decode`, `--limit`, `--skip`, `--input-bit-depth`, `--timebase`, `--fps`, `--stereo-mode`, `--nv12`/`--yv12` (input packing), `--forced_max_frame_width/height`, `--large-scale-tile`, `AV1E_SET_SINGLE_TILE_DECODING`, `AV1E_SET_VMAF_MODEL_PATH`.
- **Getters (query, no bitstream):** `AOME_GET_LAST_QUANTIZER[_64]`, `AOME_GET_LOOPFILTER_LEVEL`, `AV1_GET_REFERENCE`, `AV1E_GET_ACTIVEMAP`, `AV1_GET_NEW_FRAME_IMAGE`, `AV1_COPY_NEW_FRAME_IMAGE`, `AV1E_GET_SEQ_LEVEL_IDX`, `AV1E_GET_BASELINE_GF_INTERVAL`, `AV1E_GET_TARGET_SEQ_LEVEL_IDX`, `AV1E_GET_NUM_OPERATING_POINTS`, `AV1E_GET_LUMA_CDEF_STRENGTH`, `AV1E_GET_HIGH_MOTION_CONTENT_SCREEN_RTC`, `AV1E_GET_GOP_INFO`.
- **Debug/test-only:** `AV1E_ENABLE_MOTION_VECTOR_UNIT_TEST`, `AV1E_SET_FP_MT_UNIT_TEST`, `AV1E_ENABLE_EXT_TILE_DEBUG`, `AV1E_ENABLE_SB_MULTIPASS_UNIT_TEST`.

---

## Actionable gaps

### PARTIAL (single-frame-relevant — some arms done, others not)

1. **`--sb-size=128` encode** — decoder/entropy SB-generic; encoder walk SB64-only (C8, M).
2. **`--enable-cdef=3` CDEF_ADAPTIVE + FAST_LVL1..5 e2e** — FULL search gated; FAST tables unit-tested-only, ADAPTIVE (tune) not ported (C1).
3. **`--enable-restoration` speed-1..4 arms** — pinned; base real-content encode not byte-exact at speed≥1 (KB-13-coupled).
4. **`--loopfilter-control=0`** — default `=1` gated; disable-LF arm not celled.
5. **`--deltaq-mode=3` highbd + partial-SB** — bd8 mult-of-64/8 done; highbd FP-quantize + source-border extension follow-ups (C5).
6. **`--superres-mode` random/qthresh/auto + `--superres-{kf-,}qthresh`** — FIXED byte-exact; denom-derivation modes absent (C6).
7. **`--denoise-noise-level`/`-block-size`/`--enable-dnl-denoising`** — 2 of 6 estimation kernels done; noise-model(AR)+wiener-FFT+orchestrator absent (C7, L, float-gated).
8. **`--use-intra-dct-only=1`** — luma byte-faithful; chroma UV-loop divergence PINNED-OPEN (C9/Section B).
9. **`--enable-palette`** — 5/7 byte-exact; 2 128² partition near-tie cells PINNED (KB-P29).
10. **`--enable-intrabc`** — hash/DV/predictor kernels landed but skeleton UNWIRED (coeff arm + integration ABSENT) (C3, L).
11. **`--coeff-cost-upd-freq` / `--mode-cost-upd-freq` / `--dv-cost-upd-freq` non-default arms** — default SB arm byte-exact only (C11).
12. **CICP/color-config echo** (`--color-primaries`/`-transfer-characteristics`/`-matrix-coefficients`/`--chroma-sample-position`/`--color-range`) — components bit-exact; full self-derived echo still bootstrapped (C11).
13. **`--render-size`, `--timing-info`** — header components bit-exact; not gated with non-default values.
14. **`--target-seq-level-idx`/`--set-tier-mask`/`--strict-level-conformance`** — seq op-points component present; level auto-derivation/enforcement absent (C11).
15. **`--qm-y`/`-u`/`-v`** — level derived from qm_min/max; direct per-plane set not exercised.
16. **`--superres-kf-denominator`** — covered by fixed-KEY encodes; not separately celled.

### ABSENT (single-frame-relevant — no port)

1. **`--aq-mode=1/2/3`** (variance/complexity/cyclic segmentation) — needs two-pass fire (C5).
2. **`--deltaq-mode=4` (user-rating)`/=5` (HDR)** (C5).
3. **`--delta-lf-mode`** (C5).
4. **`--auto-intra-tools-off`** (deltaq-3 intra-tool disable) (C5).
5. **`--enable-rate-guide-deltaq` / `--rate-distribution-info`** (C5, external file).
6. **`--tune=ssim`** (SSIM-rdmult scaling).
7. **`--tune=vmaf` / `vmaf_*` / `butteraugli` / `vmaf_saliency_map`** (+ `--vmaf-model-path`).
8. **`--screen-detection-mode`** (detection unported; port takes the flag as input) (C3).
9. **`--tune-content=screen,film`** (C3).
10. **`--quant-b-adapt`** (adaptive quantize_b) (C9).
11. **`--min-q` / `--max-q`** (quantizer clamps) (C11).
12. **`--min-cr`** (min compression ratio) (C11).
13. **`--full-still-picture-hdr`** (still-picture full header — single-frame-specific) (C11).
14. **`--annexb`** (Annex-B framing) (C11).
15. **`--force-video-mode`** (disables still-picture header opt) (C11).
16. **`--error-resilient=1`** encode arm (header bit supported, not gated).
17. **`--resize-mode`/`-denominator`/`-kf-denominator`** (encode-side frame resize; kernel exists via superres).
18. **`--external-partition` / `--partition-info-path` / `--sb-qp-sweep`** (diagnostic) (C8).
19. **`--auto-tiles`, `--num-tile-groups`>1, `--mtu-size`** (auto/multi tile-group).
20. **8-bit superres denom-16-even-width corner** (optimized scaler) (C6).
21. **highbd lossless** (mono+420 bd8 done; bd10/12 lossless follow-up) (C12).
22. **`AOME_SET_ROI_MAP`** (ROI segmentation — if ever wanted for stills).

---

## Discrepancies found (docs vs test source — test source wins)

1. **Lossless 4:2:0 — RESOLVED byte-exact.** `PARITY.md` Section A ("4:2:0 still open — KB-5")
   and Section C12 say the 4:2:0 lossless cell is open, but `CLAUDE.md` KB-5 says fixed, and the
   actual gate `encoder_gate_lossless_cq0_e2e_kb5_repro` (encoder_gate_chroma_ss_e2e.rs:1282)
   **hard-asserts BOTH `mono.matched` AND `c420.matched`**. → Lossless is BYTE-EXACT-GATED for
   mono + 4:2:0 bd8. PARITY.md Section A row + C12 are stale; recommend correcting them.
2. **`coverage.py` 0/349 is a mechanical artifact** — `coverage/feature_map.json` is `{}`. The
   349-feature gate is disconnected from the ~52 byte-exact landings tracked in PARITY.md
   Section A. Populating `feature_map.json` from Section A is the highest-leverage coverage fix.
3. `PARITY.md` Section C2 cites `default_extra_cfg.enable_restoration = 1` at "av1_cx_iface.c:286"
   — confirmed correct (line 286 in the compiled `!CONFIG_REALTIME_ONLY` struct). (The realtime
   `#else` struct at :443 has it 0 but is not compiled here.)

---

*Report-only audit. C entry points = libaom v3.14.1 (`reference/libaom`, git 03087864). Port
evidence base = PARITY.md Section A gates + on-disk test bodies at HEAD 79e7a6d (2026-07-17).*
