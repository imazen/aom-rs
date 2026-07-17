# PARITY.md — the stills-parity ledger

Single source of truth for the **stills bulk-port pivot** (2026-07-16): port ALL absent
stills-relevant encoder features with **RD-closeness** validation first (quality + size vs
real aomenc), byte-exactness deferred per feature. Stills scope = single-frame / ALLINTRA
(usage=2) / KEY-frame encode; inter-frame/video-only features (motion, TPL, GOP, RC beyond
fixed-Q, S-frames, temporal filtering) are out of scope.

## Rules

1. **New features land OFF-by-default.** The proven byte-exact envelope (section A) must
   stay byte-exact — every landing runs the full suite and perturbs nothing. A feature is
   reached only by its explicit knob until it graduates.
2. **Every landing updates its row.** A bulk-ported feature appends a section-B row with
   its measured deltas the same commit it lands. Absent features live in section C; a
   feature moves C → B when its RD-close gate lands, and B → A when a byte-identity gate
   lands (cite the new gate + commit in the row you move).
3. **The RD-close gate is `aom_bench::rd_close`** (landed 3c5235e):
   encode the SAME input with the port (knob on) and real aomenc (same knobs), decode BOTH
   with the port decoder, score BOTH recons against the source with zensim (single-threaded,
   deterministic), record sizes. Acceptance bands (`RdBands::default()`):
   **|size_delta| <= 5% AND zensim_drop <= 0.5** (bit-identical cells fast-path as EXACT).
   Bands were sanity-anchored on first real data: byte-exact cells report 0/0 with zensim
   79.5–91.7 at web cq; a genuine cq20→cq63 divergence measures −94% size / −171 zensim —
   the bands discriminate near-ties from regressions with wide margin. Tightening per
   family is fine; widening is a test relaxation (user sign-off required).
   Usage: `cargo test -p aom-bench --test rd_close_harness -- --nocapture` (the harness's
   own gate); bulk agents call `compare_cell` / `run_stock_cell` / `splice_frame_obu` from
   their own tests with their knob wired.
4. **No bootstrap leaks.** The port's stock encode bootstraps some frame-header FIELDS from
   the C stream (qindex mapping, tile limits — the documented Gate-3 caveat). The feature
   under test must NOT flow through that bootstrap: a CDEF-search port derives its own
   strengths, an LR port its own RU params. Copying the feature's decisions from the C
   header fakes parity.
5. Cheap cells (64²/128², a few cq) so gates run often; always include at least one
   real-content cell (`EncodeCell::real_content`, the KB-6 conformance-decoded-YUV recipe)
   — synthetic-only validation has already missed real divergences once (KB-6).

## Section A — BIT-IDENTICAL (proven)

Byte-identity gates landed and green on origin/main. Any regression here is a shipping bug.

### Encoder (vs real `aomenc` path, `aom_codec_av1_cx`)

| Component / envelope | Gate (test name) | Landed |
|---|---|---|
| ALLINTRA speed-0 e2e, synthetic grids (mono+4:2:0, multi-SB 16/16, cq5..63) | `encoder_gate_e2e_byte_match` | 76b1ffb |
| speed-0 low-qindex web range (cq8–30, 12 cells) | `encoder_gate_e2e_low_qindex_speed0` | ec5905c |
| speed-0 rich-content strong-LF incl. screen-content cq62 (6/6) | `encoder_gate_e2e_rich_content_strong_lf` | 74fb582 (KB-2) |
| **REAL-content map 30/30** (bd8 4:2:0, 64²/128²/196² × cq5..63, incl. partial-SB frame edge) | `encoder_gate_real_image_e2e_kb6_repro` | ca2826f → 57d5ce0 (KB-6 series) |
| `--cpu-used=1` (14/14) | `encoder_gate_speed1_textured_allintra` | 7e2391d, ad734e4, a128655 |
| `--cpu-used=2` | `encoder_gate_speed2_textured_allintra` | a8a3992 |
| `--cpu-used=3` (64/64) | `encoder_gate_speed3_textured_allintra` | e18772c, 652423e (KB-7) |
| `--cpu-used=4` (64/64) | `encoder_gate_speed4_textured_allintra` | e8c662f → 35fdce8, 652423e (KB-8) |
| `--cpu-used=5` (64/64) | `encoder_gate_speed5_textured_allintra` (+ `encoder_gate_speed5_vs_speed4_sf_witness`) | 9aeb0ee |
| `--cpu-used=6` (64/64 canon; noise ext cq32/48 asserted, cq63 near-tie pinned open — KB-10) | `encoder_gate_speed6_textured_allintra` (+ `encoder_gate_speed6_vs_speed5_sf_witness`, `encoder_gate_speed6_noise_flatuv_allintra`) | 90e69e8 |
| `--cpu-used=7` (64/64 canon; VAR_BASED_PARTITION fixed-tree + rd_use_partition; noise ext cq12/32/48 asserted, cq63 = the KB-10 near-tie twin pinned open — KB-11) | `encoder_gate_speed7_textured_allintra` (+ `encoder_gate_speed7_vs_speed6_sf_witness`, `encoder_gate_speed7_noise_flatuv_allintra`, `kb11_speed7_noise_localize`) | a9dc5f1 |
| bd10/bd12 mono+4:2:0 aggressive-HF (12/12) | `kb4_gate_bd10_bd12_mono_hf_byte_match` | a2dd28e (KB-4) |
| bd10 non-4:2:0 (444/422 × 64²/128²) | `encoder_gate_bd10_non420_e2e_kb4_repro` | 1ecfafb |
| bd10/bd12 full-frame mono+4:2:0 | `encoder_gate_bd10_diff` | 20f1e70, 800e6fc |
| 4:2:2 / 4:4:4 bd8 full-frame | `encoder_gate_chroma_ss_e2e` | 2ee900d, 0eb42eb (#26) |
| Coded-lossless cq0 **mono** (4:2:0 still open — KB-5) | `encoder_gate_lossless_cq0_e2e_kb5_repro` | ba560eb |
| QM-on forward-quant (`--enable-qm`, 40 cells bd8+bd10) | `qm_encode_witness` | 5b512bf (parts 624e91d/a066cf8/abb68d9) |
| Multi-tile encode (2×1/1×2/2×2, 4:4:4 128²) | `encoder_gate_multitile_e2e` | f6e6319 |
| **C8 partition-control disable arms** (`--enable-rect-partitions=0`, `--enable-ab-partitions=0`, `--enable-1to4-partitions=0`, `--min-partition-size=16`, `--max-partition-size=32`, square-only 8..32 band) × real-content 64²(cq32/63)+128²(cq12), each knob anti-vacuity-witnessed (must change the C stream) | `toggles_rd_close::toggles_c8_*` (hard `bit_identical` pins) | (this landing) |
| **C10 intra-tool disable arms** (`--enable-smooth-intra=0`, `--enable-paeth-intra=0`, `--enable-cfl-intra=0`, `--enable-directional-intra=0`, `--enable-diagonal-intra=0`, `--enable-angle-delta=0`, `--enable-filter-intra=0`, `--enable-intra-edge-filter=0`) × the same witnessed grid; seq-header knobs assert the C stream's seq bits == the knob (no bootstrap flow) | `toggles_rd_close::toggles_c10_*` (hard `bit_identical` pins) | (this landing) |
| **C9 tx-control arms** (`--enable-tx64=0`, `--enable-rect-tx=0`, `--enable-flip-idtx=0`, `--use-intra-default-tx-only=1`, `--reduced-tx-type-set=1`, `--enable-tx-size-search=0` — frame-header bits/tx_mode asserted == knob) × the same witnessed grid | `toggles_rd_close::toggles_c9_*` (hard `bit_identical` pins) | (this landing) |
| **C11 `--cdf-update-mode=0` encoder e2e** × the same witnessed grid (header `disable_cdf_update` asserted == knob). Landing FIXED a real pack bug: only the coeff writer was gated — partition/mode/tx symbol writers adapted CDFs unconditionally, desyncing the stream vs the non-adapting decoder (zensim −264 vs C's +79 pre-fix). Fix = C's architecture: `allow_update_cdf` on `OdEcEnc`, gated in `write_symbol` (aom_write_symbol), set per tile in `pack_tile` (write_modes) | `toggles_rd_close::toggles_c11_cdf_update_mode_0` (hard `bit_identical` pins) | (this landing) |
| **C9 `--disable-trellis-quant` arms** (`=1` NO_TRELLIS_OPT, `=2` FINAL_PASS_TRELLIS_OPT) × the same witnessed grid. `=2` landing FIXED a real pack bug: `encode_b_intra_dry` hardcoded `dry_run_output_enabled: false`, so the OUTPUT_ENABLED pack pass did not apply FINAL_PASS trellis (search=no-trellis, pack must trellis, encodemb.h:153) → recon divergence (Δzensim 1.855 pre-fix). Fix threads the `output_enabled` arg; byte-inert for every non-FINAL_PASS gate | `toggles_rd_close::toggles_c9_trellis_quant_off` / `_final_pass_only` (hard `bit_identical` pins) | 2026-07-17 (5a644c6) |
| qindex-from-cq derivation (#8) | `qindex_from_cq_diff` | (landed pre-pivot) |
| Gate-3 perf cells byte-verified before timing | `aom-bench` `EncodeCell::assert_byte_exact` | 057bde2 |
| **CDEF-strength RD search** (`--enable-cdef=1`, #7 / family C1): 14/14 cells — real content 196²/64² cq5..63 (cdef_bits=2 four-strength joint sets, per-unit literals) + mono/4:4:4/4:2:0/bd10 axes; speed-0 FULL search; two-pass encode→LF→search→pack | `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close` (aom-bench; rd_close report + full byte-identity asserts) | 016d4dd + 9850da6 + c9ebf83 |
| **Loop-restoration RD search** (`--enable-restoration=1`, family C2): 8/8 cells BYTE-IDENTICAL + 8/8 decisions equal C's — real content 64² cq{12,32,48}, 196² cq{20,48} (partial-SB edges), 352×288 cq{32,55} (multi-unit size-descent grids), b10 352×288 cq32; decision shapes covered: all-NONE, WIENER-luma, SGRPROJ-luma, WIENER-all-3-planes, mixed SGR-luma+WIENER-chroma (b10), unit-size descent picking 128; allintra speed-0 full search (all 16 SGR eps, ±{4,2,1} Wiener tap refine, 256→128→64 size loop) | `lr_restoration_gate.rs::lr_restoration_search_rd_close_vs_real_aomenc` (aom-bench; rd_close report + full byte-identity + decision-equality asserts) | e24cf09 + 96d3464 + dfd757e + 96534c4 |

### Decoder (vs real `aom_codec_av1_dx`)

| Component / envelope | Gate | Landed |
|---|---|---|
| Gate-1 conformance corpus, intra scope, **incl. q62/q63** (KB-1 fixed) + film-grain-synthesis / monochrome / cdf-update frame-0 breadth | `conformance_corpus` (byte-identity + golden MD5, CI `xtask/conformance.py --fetch --scope intra`) | 386c24f → 463f49f → 134c43c → ae0e6a1 |
| Real-bitstream KEY envelope (deblock, CDEF, LR, superres, SB128, lossless, QM, multi-tile, palette, intrabc, disable-cdf-update, 4:2:2 chroma deblock) | `real_bitstream` gate family | b8d79b2 → 3380a91, 798ec25, a90b0e7, 8502e13, 6899bea, 1dfbcc3, 42423ab, 351a160 |

## Section B — PORTED, RD-CLOSE (not yet bit-exact)

Bulk agents append rows here as features land (rule 2). Empty at pivot start.

| Component | Knobs | Cells | size_delta | zensim_drop | Harness ref (test) | Date | Notes |
|---|---|---|---|---|---|---|---|
| Palette RD search (Y `av1_rd_pick_palette_intra_sby` + UV `_sbuv`: dim-1/2 k-means, top-colours, colour/map costs, header-rd gating + chroma early-term, palette recon + pack syntax/map tokens, neighbour cache/ctx grids) | `PickFrameCfg::palette_costs = Some` (= `--enable-palette=1`; OFF everywhere else) | 6 screen (text/UI, mono+420, 64²/128², cq12..63) + 1 real-content control | **5/7 EXACT (byte-identical)**; worst +2.55% | worst +0.190 (one cell −1.041 = port better) | `rd_close_palette::palette_y_rd_close_gate` (aom-bench) | 2026-07-17 | speed-0 sf levels (search 0 / size-search 1 / chroma early-term 1); speeds 1–5 levels wired untested-by-gate. Fixed latent UV no-palette-flag under-cost on screen frames (per-leaf `try_palette`). (CDEF search + loop-restoration search, the first two bulk families, went straight to section A — 14/14 and 8/8 EXACT.) |
| C9 `--use-intra-dct-only=1` (PINNED-OPEN: luma byte-faithful, chroma UV-mode-loop divergence) | `AV1E_SET_INTRA_DCT_ONLY=1` | 64²cq32 / 64²cq63 / 128²cq12 (real content) | +2.23% / 0 (EXACT) / −1.40% | +3.588 (OUT of band) / 0 / +0.333 | `toggles_rd_close::toggles_c9_intra_dct_only_pinned_open` | 2026-07-17 | Y recon identical; first divergent leaf mi(0,0) 32×32: real uv=D45/aduv2 (eob 1) vs port uv=V (eob 78); real winners are derived-type==DCT modes (DCT-forced-search signature). Port UV txb eval + UV mode loop both match the C-pieces oracles under the knob (txfm_uvrd_diff / intra_sbuv_mode_loop_diff sweep green; mask verified vs the REAL facade incl. the PAETH reduced-set reset) ⇒ shared port+oracle mis-model of the REAL UV loop; next: sibling-C UV-candidate rd dump (KB-2/KB-7 method). |

## Section C — ABSENT (to port), by family

Status legend: **ABSENT** = no port; **PARTIAL** = kernels/plumbing exist, search/threading/
validation missing. Size: S (≤1 day), M (1–3 days), L (multi-day → decompose). C entry
points are libaom v3.14.1 (`reference/libaom`). Defaults verified in
`av1/av1_cx_iface.c` (allintra override block :3065–3078 sets ONLY `enable_cdef=0`,
`screen_detection_mode=ANTIALIASING_AWARE`, `qm_min=4`, `qm_max=10`).

### C1 — CDEF strength search — **PORTED, BIT-IDENTICAL → section A** (2026-07-17)
- Landed 016d4dd (`aom-encode/src/pickcdef.rs`, the full `av1_cdef_search` + FAST-level
  tables) + 9850da6 (`pack_tile_from_trees` two-pass pack + `write_cdef` literal wiring) +
  c9ebf83 (the byte-identity gate, 14/14 EXACT). See the section A row + STATUS.md
  2026-07-17 for the full inventory.
- Remaining sub-scope (honest fractions): e2e-gated = speed-0 `CDEF_FULL_SEARCH` only;
  `CDEF_FAST_SEARCH_LVL1..5` are ported + table-unit-tested but not yet e2e-gated
  (cheap extension: CDEF-on cells at `--cpu-used=1..6`); `CDEF_PICK_FROM_Q`
  (speed≥7 rt) + `CDEF_ADAPTIVE` (`tune=IQ/SSIMULACRA2`, off at cq≤32) NOT ported
  (documented-dead for `--enable-cdef=1`); SB128 CDEF-on blocked on the pack's SB64
  envelope (the search's >64-fb arms are already in place).

### C2 — Loop-restoration search (Wiener/SGR) — **PORTED, BIT-IDENTICAL → section A** (2026-07-17)
- `--enable-restoration` / `AV1E_SET_ENABLE_RESTORATION`. **Allintra config default is ON
  (=1)** — verified: `default_extra_cfg.enable_restoration = 1` (av1_cx_iface.c:286),
  threaded non-realtime at :1273, NOT touched by the allintra override block. A DEFAULT
  allintra aomenc encode RUNS `av1_pick_filter_restoration` (sometimes resolving all-NONE,
  but the seq/frame header bits differ from `=0`) — this family was the highest-priority
  default-parity gap, now closed at the knob level.
- **Landed (4 chunks):** e24cf09 (write-side syntax: binary-codes writer primitives +
  `write_lr_unit`, byte-identical to the REAL C writer + exhaustive count parity);
  96d3464 (search numeric core: `compute_stats[_highbd]`, `pixel_proj_error`,
  `calc_proj_params`/`get_proj_subspace`, SGR flt producer — all diffed vs EXPORTED `_c`
  fns; Wiener solve chain transcribed, no C export exists); dfd757e (decision layer:
  per-unit RD searches, SB-coding-order walk, unit-size descent, `pick_filter_restoration`);
  96534c4 (`pack_tile_lr` RU-interleaved SB-root writes + `port_encode_lr` pipeline:
  LF apply → search → repack → derived restoration header; gate). Gate hardened to full
  byte-identity + decision-equality asserts after measuring 8/8 EXACT.
- Remaining sub-scope (honest fractions): e2e-gated = **allintra speed-0 only** (the
  full-search path: all 16 SGR eps, 256→128→64 size descent, no prunes). The allintra
  speed 1–4 `lpf_sf` arms (`lr_search_sf_allintra` — single-size qindex rule, ep-prune
  ladder, src-var/sgr-from-wiener prunes, reduced 5-tap luma window) are PORTED but not
  yet e2e-gated; GOOD-mode setters not wired (asserted against); mono/4:4:4/bd12 cells
  absent from the gate grid; speed>=5 allintra is structurally LR-off in C (sf disable +
  seq-bit clear) so no gate is possible there. `pack_tile_from_trees` unification (reuse
  the CDEF two-pass pack instead of the re-search repack) queued as an optimization.
- Decoder-side LR (apply path) was already complete + gated pre-pivot (section A decoder
  rows).

### C3 — Screen-content tools — ABSENT (L, decompose) — bulk agent live (#29)
- Palette search: `--enable-palette` (default ON, gated on `allow_screen_content_tools`).
  C: `av1/encoder/palette.c` `av1_rd_pick_palette_intra_sby/_sbuv` (k-means),
  `intra_mode_search.c` `av1_search_palette_mode_luma`;
  `intra_sf.{prune_palette_search_level, prune_luma_palette_size_search_level,
  early_term_chroma_palette_size_search}`. **MOVED to section B (2026-07-17)** — the Y+UV
  searches + palette recon + pack syntax/map tokens landed RD-close (5/7 cells byte-exact).
  Remaining inside the family: `av1_search_palette_mode[_luma]` (inter-frame callers, out of
  stills scope).
- IntraBC: `--enable-intrabc` (default ON, screen-gated). C: `av1/encoder/rdopt.c`
  `rd_pick_intrabc_mode_sb`, DV hash `av1/encoder/hash_motion.c`,
  `mv_sf.intrabc_search_level`. Port has: header intrabc-present bit + intrabc costs +
  intrabc decode + **chunk 3a landed 2026-07-17** (`aom-encode/src/intrabc_search.rs`:
  CRC-32C + the full source-frame hash-table build/query + `av1_fill_dv_costs`/
  `av1_build_nmv_cost_table` DV cost tables + the mv_cost/mv_err_cost/mvsad_err_cost
  forms + variance/SAD metrics, unit-gated). Missing: 3b `rd_pick_intrabc_mode_sb`
  (dv-ref via the ported `dv_ref.rs` `find_dv_ref_mvs`/`find_ref_dv` + per-direction
  ABOVE/LEFT mv limits + the hash candidate loop with `get_mvpred_var_cost`), 3c the
  intrabc RD/recon/pack (skip + no-skip arms, recon-copy prediction incl. the chroma
  2-tap at odd 420 DVs — decoder-side `intrabc_chroma_predict` is the reference —
  and is_inter var-tx/skip pack syntax), then the NSTEP diamond + intrabc mesh
  full-pel search (hash-only covers exact repeats first). (L, decomposed)
- Screen detection: `--screen-detection-mode` (allintra default ANTIALIASING_AWARE=2).
  C: `av1/encoder/encoder.c` `av1_set_screen_content_options`. Port takes
  `allow_screen_content_tools` as an input — the detection itself is unported. (S–M)
- `--tune-content` screen/film forcing (gates the above). (S)

### C4 — tune=IQ / tune=SSIMULACRA2 family — PARTIAL (M total) (#23 tail)
The tune bundle (`handle_tuning`, av1_cx_iface.c:1938–1978): `enable_qm=1, qm_min=2,
qm_max=10, sharpness=7, dist_metric=QM_PSNR, enable_cdef=ADAPTIVE, enable_chroma_deltaq=1,
deltaq_mode=6 (VARIANCE_BOOST)`; IQ adds `enable_adaptive_sharpness=1`.
- PRESENT: IQ/SSIMULACRA2 rdmult weight (`rd.rs:208`, bit-exact); QM forward-quant +
  `aom_get_qmlevel`/`aom_get_qmlevel_allintra` + wt_matrix (5b512bf, 40/40 byte gate);
  trellis takes sharpness 0..7 (diffed); LF-init sharpness limits (decode side).
- ABSENT: `aom_get_qmlevel_luma_ssimulacra2` + `aom_get_qmlevel_444_chroma` formulas
  (quant_common.h:111/:150) (S); QM_PSNR dist metric in tx search/trellis
  (`tx_search.c` `use_qm_dist_metric` :1150, `txb_rdopt.c` :347) (S–M); chroma deltaq
  (`--enable-chroma-deltaq`) (S–M); `--sharpness` e2e threading incl. the
  `av1_init_quantizer` rounding bias (`sharpness_adjustment`, av1_quantize.c:607) +
  allintra edge-filter reduction (S–M); adaptive sharpness (`picklpf.c:232`) (S);
  deltaq-mode=6 variance boost (see C5) (M).

### C5 — aq-mode / deltaq-mode variants — ABSENT (M–L)
- `--deltaq-mode=3` DELTA_Q_PERCEPTUAL_AI — the genuinely stills-specific arm:
  `av1/encoder/allintra_vis.c` `av1_set_mb_wiener_variance`, `av1_get_deltaq_offset`,
  + `--auto-intra-tools-off`, `--enable-rate-guide-deltaq`. (M–L)
- `--deltaq-mode=6` DELTA_Q_VARIANCE_BOOST (tune=IQ default): `allintra_vis.c`
  `av1_get_sbq_variance_boost`, `aq_variance.c` `av1_get_variance_boost_block_variance`,
  `--deltaq-strength`. (M)
- `--deltaq-mode=1` OBJECTIVE (base default) is TPL-gated (encodeframe.c:343) — **inert
  for a lone still**; document-only.
- `--aq-mode=1/2` (variance/complexity segmentation): `aq_variance.c` `av1_vaq_frame_setup`,
  `aq_complexity.c` — requires the two-pass path to fire on stills (shim note). (M, low
  priority) `--delta-lf-mode` (S–M).
- Encoder-side per-SB delta-q/delta-lf tile signaling (writer side). (S–M, shared)

### C6 — Superres (encode side) — ABSENT (M–L)
- `--superres-mode/-denominator/-kf-denominator/-qthresh/-kf-qthresh`. Default NONE.
- C: `av1/encoder/superres_scale.c` `av1_superres_post_encode` + mode/denom selection,
  `hl_sf.superres_auto_search_type=SUPERRES_AUTO_DUAL` at allintra speed-0 (dual = encode
  twice + compare); downscale `av1/common/resize.c`. Port has: normative decode-side
  upscale + `ref_encode_av1_kf_superres` oracle. Missing: source downscale, the recode
  loop, RD compare, header denominator signaling from the encoder.

### C7 — Film grain / denoise estimation — ABSENT (S table-inject, L estimation)
- `--film-grain-test` / `--film-grain-table`: inject grain params → header. C:
  `grain_test_vectors.h`, `bitstream.c` `write_film_grain_params` (:2529). Port already
  has the `write_film_grain_params` header writer bit-exact — the remaining lift is
  param plumbing + a gate. (S)
- `--denoise-noise-level/-block-size`, `--enable-dnl-denoising`: noise-model estimation +
  source denoise + grain fit. C: `aom_dsp/noise_model.c` (`aom_denoise_and_model_*`). (L)

### C8 — Partition controls — disable arms DONE (byte-exact); SB128 remains
- **DONE (this landing, → section A):** `--enable-rect-partitions=0`, `--enable-ab-partitions=0`,
  `--enable-1to4-partitions=0`, `--min-partition-size`, `--max-partition-size` + the
  square-only 8..32 interaction arm — all BYTE-IDENTICAL vs real aomenc (same ctrl) on the
  real-content grid, hard-pinned in `toggles_rd_close` (aom-bench). Infra: generic ctrl-pair
  shim `shim_encode_av1_kf_ctrls` + `ToggleKnobs`/`port_encode_with`; ctrl-id constants
  header-cross-checked (`cx_ctrl_ids_match_reference_headers`). C mapping verified:
  `set_max_min_partition_size` (partition_strategy.h:214) `min(sf_default, dim_to_size(px),
  sb)` / `min(max(BLOCK_4X4, dim), sb)`; the auto-max ML arm is inter-only.
- `--sb-size=128` ENCODE side — ABSENT: decoder + entropy layers are SB-size-generic
  (798ec25), but the encoder walk/harnesses are SB-64-only. (M)
- External partition / `--partition-info-path` / `--sb-qp-sweep`: diagnostic, lowest
  priority. (M, defer)

### C9 — Transform controls — mostly DONE (byte-exact); dct-only pinned-open
- **DONE (this landing, → section A):** `--enable-tx64=0`, `--enable-rect-tx=0`,
  `--enable-flip-idtx=0`, `--use-intra-default-tx-only=1`, `--reduced-tx-type-set=1` — all
  BYTE-IDENTICAL vs real aomenc on the witnessed grid (`toggles_rd_close::toggles_c9_*`).
  Threading landed: `TxTypeSearchPolicy.{enable_flip_idtx, use_intra_dct_only}` →
  `TxMaskParams` (tx_search.rs; C reads oxcf directly in `get_tx_mask`, stage-independent);
  partition_pick's derived winner-mode stage policies copy the CLI toggles from `cfg.pol`
  (+ the MODE_EVAL `use_default_intra_tx_type` OR, rdopt_utils.h:579). The five layer
  differentials (`uniform_txfm_yrd_diff`, `intra_sby_mode_loop_diff`, `rd_pick_intra_sb_diff`,
  `txfm_uvrd_diff`, `intra_sbuv_mode_loop_diff`) now SWEEP `use_intra_dct_only` (oracle chain
  threads it into the REAL `get_tx_mask` facades) — all green.
- **`--use-intra-dct-only=1` — PINNED-OPEN** (section B row): luma byte-faithful; chroma
  UV-mode-loop winner divergence vs real aomenc, out of band at 64²cq32. Full localization
  trail in the section-B row + the pinned test's doc comment.
- **`--enable-tx-size-search=0` DONE (this landing, → section A):** knob route landed —
  `TxTypeSearchPolicy.enable_tx_size_search` (the port's oxcf.txfm_cfg carrier): the speed-0
  single-pass method pick goes USE_FULL_RD → USE_LARGESTALL (intra_rd.rs), the winner-mode sf
  derivation forces `tx_size_search_level = 3` post-speed (speed_features.c:2726 shape,
  partition_pick.rs), and the leaf `tx_mode_is_select` init ANDs the knob (select_tx_mode →
  TX_MODE_LARGEST; the existing KB-10 LARGESTALL⇒not-select coupling handles the pass level).
  C forbids combining with `--enable-tx64=0` (encodeframe.c:2461 assert) — not celled.
- Remaining: `--disable-trellis-quant` values 1/2 as explicit knob states (stage-aware
  policies exist from KB-8; default is 3) (S); `--quant-b-adapt` (the `_adaptive` quantizer
  family) (S–M).

### C10 — Intra mode toggles — DONE (byte-exact)
- **DONE (this landing, → section A):** all 8 toggles — `--enable-smooth-intra=0`,
  `--enable-paeth-intra=0`, `--enable-cfl-intra=0`, `--enable-directional-intra=0`,
  `--enable-diagonal-intra=0`, `--enable-angle-delta=0`, `--enable-filter-intra=0`,
  `--enable-intra-edge-filter=0` — BYTE-IDENTICAL vs real aomenc on the witnessed grid
  (`toggles_rd_close::toggles_c10_*`). Threading landed: `IntraToolCfg` on `PickFrameCfg`
  (partition_pick.rs; the 5 luma flags applied onto `IntraSbyGates` after the sf
  derivation — C keeps CLI + sf gates separate and the diffed visit chain reads both);
  chroma copies ride the existing `UvLoopPolicy` fields (the speed>=3 chroma rebuild
  spreads `..cfg.uv_lp.clone()`, so they survive at all speeds). The seq-level pair
  (filter-intra / intra-edge-filter) is knob-driven on the port side with the bootstrap
  seq bits ASSERTED equal (no bootstrap flow).

### C11 — Bitstream / global — mostly PRESENT
- PRESENT: bd 8/10/12, mono, 4:2:0/4:2:2/4:4:4, tiles (multi-tile e2e), lossless-mono,
  QM signaling, header/OBU writers (seq + frame, all components bit-exact).
  `--reduced-tx-type-set=1` e2e byte gate landed with C9 (this landing).
- **`--cdf-update-mode=0` encoder e2e DONE (this landing, → section A)** — and it caught a
  REAL pack bug (see the section-A row: symbol writers adapted CDFs unconditionally; the
  writer-side `allow_update_cdf` gate now mirrors C's aom_write_symbol).
- PARTIAL: cost-upd-freq knobs
  (`--coeff/mode/dv-cost-upd-freq` non-default arms; default arm proven byte-exact via
  the multi-SB e2e gates) (S–M); self-derived seq/frame header fields (drop the Gate-3
  bootstrap caveat: qindex mapping done #8; tile limits, CICP echo, level/tier remain)
  (S–M); `--full-still-picture-hdr` / annexb framing arms (S).
- `--min-q/--max-q/--min-cr` clamps (S).

### C12 — Lossless tail — PARTIAL (S)
- KB-5 remainder: 4:2:0 cq0 chroma RD near-tie (≤1 rdcost unit, first 16×16 node). Mono is
  byte-exact + gated. Next: lossless chroma UV-RD differential at qindex 0.

### C13 — Speed levels 6–9 — PARTIAL (speed-6 in flight by another track)
- `--cpu-used=6`: chunk 1 landed (5935250, LPF_PICK_FROM_Q). 7–9: allintra speed-feature
  deltas unported (speed_features.c:527+); includes `top_intra_model_count_allowed=2` at
  speed≥6 and the speed-7+ realtime-leaning arms. (M per level)

### Priority order (proposed)
~~1. **C2 LR search**~~ DONE (section A, 2026-07-17) → ~~2. **C1 CDEF search**~~ DONE
(section A, 2026-07-17) → 3. **C3 screen content** (web stills) → 4. **C4
tune=IQ/SSIMULACRA2 tail** (image-quality tuning, small pieces) → 5. **C5 deltaq 3/6** →
6. C8/C9/C10 toggle threading (cheap wins, many S) → 7. C6 superres, C7 film grain →
8. C11/C12 tails → C13 speeds 7–9. (C2/C1 leftovers — LR speed-1..4 e2e arms, CDEF FAST
levels e2e — are follow-ups within their families, below the C3+ fronts.)
