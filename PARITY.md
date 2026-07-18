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
| `--cpu-used=8` (60/64 canon; nonrd PICKMODE — `nonrd_use_partition` single-pass walk + `av1_nonrd_pick_intra_mode` estimate/hybrid arm; noise ext cq12/32/48/63 asserted; 4 `diag` estimate-arm V/H near-ties pinned open — KB-12) | `encoder_gate_speed8_textured_allintra` (+ `encoder_gate_speed8_vs_speed7_sf_witness`, `encoder_gate_speed8_noise_flatuv_allintra`) | 9b57803 |
| `--cpu-used=9` (64/64 canon; all-estimate `hybrid_intra_pickmode=0` + the 3 speed-9 mode prunes + INTERNAL_COST_UPD_OFF; noise ext cq12/32/48/63 asserted — KB-12) — GATE 2 (cpu 0-9) COMPLETE | `encoder_gate_speed9_textured_allintra` (+ `encoder_gate_speed9_vs_speed8_sf_witness`, `encoder_gate_speed9_noise_flatuv_allintra`) | 9b57803 |
| bd10/bd12 mono+4:2:0 aggressive-HF (12/12) | `kb4_gate_bd10_bd12_mono_hf_byte_match` | a2dd28e (KB-4) |
| bd10 non-4:2:0 (444/422 × 64²/128²) | `encoder_gate_bd10_non420_e2e_kb4_repro` | 1ecfafb |
| bd10/bd12 full-frame mono+4:2:0 | `encoder_gate_bd10_diff` | 20f1e70, 800e6fc |
| 4:2:2 / 4:4:4 bd8 full-frame | `encoder_gate_chroma_ss_e2e` | 2ee900d, 0eb42eb (#26) |
| Coded-lossless cq0 **mono + 4:2:0** bd8 (both hard-asserted; KB-5 closed / #32) | `encoder_gate_lossless_cq0_e2e_kb5_repro` | ba560eb (mono) + KB-5 420 fix |
| QM-on forward-quant (`--enable-qm`, 40 cells bd8+bd10) | `qm_encode_witness` | 5b512bf (parts 624e91d/a066cf8/abb68d9) |
| Multi-tile encode (2×1/1×2/2×2, 4:4:4 128²) | `encoder_gate_multitile_e2e` | f6e6319 |
| **C8 partition-control disable arms** (`--enable-rect-partitions=0`, `--enable-ab-partitions=0`, `--enable-1to4-partitions=0`, `--min-partition-size=16`, `--max-partition-size=32`, square-only 8..32 band) × real-content 64²(cq32/63)+128²(cq12), each knob anti-vacuity-witnessed (must change the C stream) | `toggles_rd_close::toggles_c8_*` (hard `bit_identical` pins) | (this landing) |
| **C10 intra-tool disable arms** (`--enable-smooth-intra=0`, `--enable-paeth-intra=0`, `--enable-cfl-intra=0`, `--enable-directional-intra=0`, `--enable-diagonal-intra=0`, `--enable-angle-delta=0`, `--enable-filter-intra=0`, `--enable-intra-edge-filter=0`) × the same witnessed grid; seq-header knobs assert the C stream's seq bits == the knob (no bootstrap flow) | `toggles_rd_close::toggles_c10_*` (hard `bit_identical` pins) | (this landing) |
| **C9 tx-control arms** (`--enable-tx64=0`, `--enable-rect-tx=0`, `--enable-flip-idtx=0`, `--use-intra-default-tx-only=1`, `--reduced-tx-type-set=1`, `--enable-tx-size-search=0` — frame-header bits/tx_mode asserted == knob) × the same witnessed grid | `toggles_rd_close::toggles_c9_*` (hard `bit_identical` pins) | (this landing) |
| **C11 `--cdf-update-mode=0` encoder e2e** × the same witnessed grid (header `disable_cdf_update` asserted == knob). Landing FIXED a real pack bug: only the coeff writer was gated — partition/mode/tx symbol writers adapted CDFs unconditionally, desyncing the stream vs the non-adapting decoder (zensim −264 vs C's +79 pre-fix). Fix = C's architecture: `allow_update_cdf` on `OdEcEnc`, gated in `write_symbol` (aom_write_symbol), set per tile in `pack_tile` (write_modes) | `toggles_rd_close::toggles_c11_cdf_update_mode_0` (hard `bit_identical` pins) | (this landing) |
| **C9 `--disable-trellis-quant` arms** (`=1` NO_TRELLIS_OPT, `=2` FINAL_PASS_TRELLIS_OPT) × the same witnessed grid. `=2` landing FIXED a real pack bug: `encode_b_intra_dry` hardcoded `dry_run_output_enabled: false`, so the OUTPUT_ENABLED pack pass did not apply FINAL_PASS trellis (search=no-trellis, pack must trellis, encodemb.h:153) → recon divergence (Δzensim 1.855 pre-fix). Fix threads the `output_enabled` arg; byte-inert for every non-FINAL_PASS gate | `toggles_rd_close::toggles_c9_trellis_quant_off` / `_final_pass_only` (hard `bit_identical` pins) | 2026-07-17 (5a644c6) |
| **`prune_tx_type_using_stats` (luma-intra tx-type stats prune)** — the ABSENT-and-UNEXERCISED sf: C enables it ALLINTRA at cpu-used>=2 (level 1) / >=4 (level 2) but ONLY `is_480p_or_larger` (speed_features.c:262/300), so every sub-480p gate frame missed it. Ported the multi-type-arm prune in `get_tx_mask_intra` (drops tx types whose KF frame-prob < the threshold, keeping the max-prob type; `update_type = KF_UPDATE` for a lone still, `default_tx_type_probs[0]`) + the framesize+speed derivation (`SpeedFeatures::prune_tx_type_using_stats`, set in port_encode_full from `min(w,h) >= 480`; the KB-3 `use_square_partition_only_threshold` framesize analog). Byte-inert on every sub-480p gate. | `tx_mask_diff` (port `get_tx_mask_intra` == the C oracle — REAL exported `default_tx_type_probs` + the prune — across all tx_size×mode×config × prune-level 0/1/2; `default_tx_type_probs_kf_matches_c`; `stats_prune_shrinks_the_mask` 120 cases) + `tx_stats_prune_e2e` (`_knob_bites`: >=480p 512² cpu-2 noise, prune LOAD-BEARING — port-without diverges from real aomenc, port-with byte-matches; `_sub480p_unchanged` regression) | (this landing) |
| qindex-from-cq derivation (#8) | `qindex_from_cq_diff` | (landed pre-pivot) |
| Gate-3 perf cells byte-verified before timing | `aom-bench` `EncodeCell::assert_byte_exact` | 057bde2 |
| **CDEF-strength RD search** (`--enable-cdef=1`, #7 / family C1): 14/14 cells — real content 196²/64² cq5..63 (cdef_bits=2 four-strength joint sets, per-unit literals) + mono/4:4:4/4:2:0/bd10 axes; speed-0 FULL search; two-pass encode→LF→search→pack | `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close` (aom-bench; rd_close report + full byte-identity asserts) | 016d4dd + 9850da6 + c9ebf83 |
| **Loop-restoration RD search** (`--enable-restoration=1`, family C2): 8/8 cells BYTE-IDENTICAL + 8/8 decisions equal C's — real content 64² cq{12,32,48}, 196² cq{20,48} (partial-SB edges), 352×288 cq{32,55} (multi-unit size-descent grids), b10 352×288 cq32; decision shapes covered: all-NONE, WIENER-luma, SGRPROJ-luma, WIENER-all-3-planes, mixed SGR-luma+WIENER-chroma (b10), unit-size descent picking 128; allintra speed-0 full search (all 16 SGR eps, ±{4,2,1} Wiener tap refine, 256→128→64 size loop) | `lr_restoration_gate.rs::lr_restoration_search_rd_close_vs_real_aomenc` (aom-bench; rd_close report + full byte-identity + decision-equality asserts) | e24cf09 + 96d3464 + dfd757e + 96534c4 |
| **tune=IQ / tune=SSIMULACRA2 family** (`--tune=iq` / `--tune=ssimulacra2`, family C4): each bundle piece e2e byte-identical — QM-level formulas (`aom_get_qmlevel_luma_ssimulacra2` + `_444_chroma`), QM-PSNR dist metric (trellis + tx-search transform-domain distortion QM-weighted), `--sharpness` 0..7, `--enable-chroma-deltaq`, `--enable-adaptive-sharpness`, Variance-Boost `--deltaq-mode=6` — PLUS the **full composite bundle** (54/54 cells: mono/420/444 × 64/128/192 × cq12/32/50, CDEF overridden off = the separate C1 track, symbol-inert). All OFF by default (`TuneKnobs::default()` = PSNR). Anti-vacuous witnesses for sharpness/adaptive/variance-boost + `tune_shim_smoke` | `encoder_gate_tune_iq_e2e` (9 tests) + `qm_level_diff` + `tune_shim_smoke` | 2026-07-17 |
| **Superres encoder-side, FIXED denom, bd8 + bd10/12** (`--superres-mode=fixed --superres-denominator=D`, family C6): **13/13 bd8 + 16/16 bd10/12 cells BYTE-IDENTICAL** — bd8 real-content 196² 4:2:0 (denoms 9/12/14 × cq{20,32,48}) + mono (denoms 9/12 × cq{20,48}); bd10/12 textured-synthetic 128² 4:2:0 (denoms 9/12/14 × cq{20,48}) + mono (denoms 9/12 × cq32), 8 cells/bit-depth. The source is downscaled horizontally to the coded `FrameWidth` via the ported non-normative `av1_resize_plane` (bd8) / `highbd_resize_plane` (bd10/12) (`aom_encode::resize`, differentially bit-exact vs the exported C symbols — interpolate 5-band + down2_symeven/symodd + resize_multistep + `coded_superres_width`), encoded at the reduced width (existing speed-0 KEY machinery, mi grid sized to coded_w), superres denom + upscaled width signalled in the header (`write_superres_scale`); port+C decoders agree on the upscaled recon. Superres OFF by default. **Anti-vacuity**: `scale_denominator == D`, `coded_w < w`. **Follow-ups (Section C6)**: the 8-bit denom-16-even-width optimized-scaler corner (`av1_resize_and_extend_frame`), and AUTO/QTHRESH/RANDOM denom selection + the recode loop. | `encoder_gate_superres_{fixed_real_content,fixed_mono,fixed_highbd}_rd_close` (aom-bench; rd_close report + full byte-identity asserts) | 2505b49f (kernel) + 68703b1 (bd8) + (this bd10/12 landing) |
| **C7 film-grain table-inject** (`--film-grain-table` / `AV1E_SET_FILM_GRAIN_TABLE`): the port's OWN grain-table reader + lookup (`aom-encode/src/grain_table.rs`, port of `aom_dsp/grain_table.c` `aom_film_grain_table_read`/`_lookup`) → `FilmGrainParams` → the already-bit-exact `write_film_grain_params` header writer. Byte-identical vs real aomenc on 4:2:0 bd8 REAL content (64² cq20/32, 128² cq12) + mono/444/bd10 synthetic × built-in test vectors 1/2/6/15 (rich full-chroma / max-lag / chroma-points-absent / chroma-scaling-from-luma). Grain is decode-side synthesis → coded tiles UNCHANGED (the C shim replicates the plain `encode_kf_pass` control set so only the seq present bit + frame grain block are added). No-bootstrap-leak witness: injecting a different vector's params DIVERGES. | `film_grain_gate.rs::film_grain_table_inject_{420_real,format_axes}` + `film_grain_no_bootstrap_leak_witness` (aom-bench) | (this landing) |
| **`--deltaq-mode=3` DELTA_Q_PERCEPTUAL_AI** (family C5, the stills-specific perceptual-AI arm): 7/7 cells BYTE-IDENTICAL to real aomenc `--deltaq-mode=3` — real content 192²/192×128/128×192 4:2:0 cq12..63 (9/6 SBs each get a distinct wiener qindex; the delta fires + the port reproduces it). Full port of `av1_set_mb_wiener_variance` (per-8x8 intra-SATD search + FP-quantize + Weber stats + the `norm_wiener_variance` `estimate`+2-iter refinement), the map reductions (`get_{satd,sse,max_scale,window_wiener_var,var_perceptual_ai}`), `av1_get_sbq_perceptual_ai` + `av1_get_deltaq_offset` (`av1_get_deltaq_offset` differentially locked 18432/18432 vs the exported C fn), and the per-SB pack threading (`setup_delta_q_perceptual_ai` → the shared `av1_adjust_q_from_delta_q_res`, reusing the mode-6 `DeltaQFrameCtx`). OFF by default; anti-vacuous knob-bites witness. **Highbd (bd10/12) DONE** (this landing): the FP-quantize arm dispatches `av1_highbd_quantize_fp` for bd>8 in `av1_set_mb_wiener_variance` (the only bd8-specific step; predict/subtract/DCT/inverse/Weber were already bd-parameterized); 5 bd10 + 1 bd10 non-square + 3 bd12 (bd10-content ×4 promoted) cells added to the gate, all byte-identical. Scope: bd8/10/12 4:2:0, dims a multiple of 64/8px (196²-partial-SB is the remaining follow-up). | `encoder_gate_deltaq_mode3_e2e` (`deltaq_mode3_e2e.rs`: `deltaq_mode3_perceptual_ai_e2e` 16/16 hard byte-match incl. bd10/12 + `deltaq_mode3_knob_bites`) + `deltaq_perceptual_ai_diff` (`get_deltaq_offset_matches_c`) | 2026-07-17 + (this bd10/12 landing) |
| **`--delta-lf-mode=1` DELTA_LF** (family C5): 7/7 cells BYTE-IDENTICAL to real aomenc `--delta-lf-mode=1 --deltaq-mode=2` — real content 192²/192×128/128×192 4:2:0 cq12..63. Per-SB `delta_lf_from_base = ((delta_qindex/4 + res/2) & ~(res-1))` clamped (setup_delta_q, encodeframe.c:380-383, `DEFAULT_DELTA_LF_RES=2`, single/`DEFAULT_DELTA_LF_MULTI=0`), derived from each SB's `delta_qindex` (reuses the mode-2/3 delta-q) in `pack_leaf` + coded via the already-plumbed `write_delta_q_params_sb` delta-lf arm. The frame `filter_level` DEPENDS on delta-lf: the LF pick's trial deblock reads `mbmi->delta_lf_from_base` via `get_filter_level` (av1_loopfilter.c:73-88), so the port stamps the per-SB delta-lf into the LF mi grid (`stamp_lf_delta_lf`) + sets `LfSearchFrame::delta_lf_present`. OFF by default (rides on a firing delta-q mode); anti-vacuous knob-bites witness. Scope: bd8 4:2:0, dims a multiple of the 64px SB. | `delta_lf_mode_e2e` (`delta_lf_mode_e2e.rs`: 7/7 hard byte-match + `delta_lf_mode_knob_bites`) | 2026-07-18 |
| **`--deltaq-mode=2` DELTA_Q_PERCEPTUAL** (family C5, the wavelet-AC-energy arm; `DELTA_Q_PERCEPTUAL_MODULATION==1`): 7/7 cells BYTE-IDENTICAL to real aomenc `--deltaq-mode=2` — real content 192²/192×128/128×192 4:2:0 cq12..63 (per-SB wavelet energy modulates the qindex; the delta fires + the port reproduces it, decode-verified: C's per-SB `current_qindex` == the port's). Full port of the 5/3 dyadic dwt (`av1_fdwt8x8_uint8_input` + `haar_ac_sad` + `av1_haar_ac_sad_mxn_uint8_input`, dwt.c — a pure-C RTCD entry, no SIMD, so bit-exact to real aomenc; differentially locked vs the exported C fn), `haar_ac_energy`/`av1_block_wavelet_energy_level`/`av1_compute_q_from_energy_level_deltaq_mode` (aq_variance.c, single-frame `energy_midpoint=10.0`), and the `av1_rc_bits_per_mb`(KEY/AOM_Q)/`find_qindex_by_rate`/`av1_compute_qdelta_by_rate` rate model (ratectrl.c), per-SB pack threading (`setup_delta_q_perceptual` → the shared `av1_adjust_q_from_delta_q_res`, reusing the mode-3/6 `DeltaQFrameCtx`). OFF by default; anti-vacuous knob-bites witness. Scope: bd8 4:2:0, dims a multiple of the 64px SB (highbd + partial-SB are follow-ups). | `deltaq_mode2_perceptual_wavelet_e2e` (`deltaq_mode2_e2e.rs`: 7/7 hard byte-match + `deltaq_mode2_knob_bites`) + `deltaq_perceptual_wavelet_diff` (`haar_ac_sad_mxn_matches_c`) | 2026-07-18 |

### Decoder (vs real `aom_codec_av1_dx`)

| Component / envelope | Gate | Landed |
|---|---|---|
| Gate-1 conformance corpus, intra scope, **incl. q62/q63** (KB-1 fixed) + film-grain-synthesis / monochrome / cdf-update frame-0 breadth | `conformance_corpus` (byte-identity + golden MD5, CI `xtask/conformance.py --fetch --scope intra`) | 386c24f → 463f49f → 134c43c → ae0e6a1 |
| Real-bitstream KEY envelope (deblock, CDEF, LR, superres, SB128, lossless, QM, multi-tile, palette, intrabc, disable-cdf-update, 4:2:2 chroma deblock) | `real_bitstream` gate family | b8d79b2 → 3380a91, 798ec25, a90b0e7, 8502e13, 6899bea, 1dfbcc3, 42423ab, 351a160 |

## Section B — PORTED, RD-CLOSE (not yet bit-exact)

Bulk agents append rows here as features land (rule 2). Empty at pivot start.

| Component | Knobs | Cells | size_delta | zensim_drop | Harness ref (test) | Date | Notes |
|---|---|---|---|---|---|---|---|
| Palette RD search (Y `av1_rd_pick_palette_intra_sby` + UV `_sbuv`: dim-1/2 k-means, top-colours, colour/map costs, header-rd gating + chroma early-term, palette recon + pack syntax/map tokens, neighbour cache/ctx grids) | `PickFrameCfg::palette_costs = Some` (= `--enable-palette=1`; OFF everywhere else) | 6 screen (text/UI, mono+420, 64²/128², cq12..63) + 1 real-content control | **5/7 EXACT (byte-identical)**; worst +2.55% | worst +0.190 (one cell −1.041 = port better) | `rd_close_palette::palette_y_rd_close_gate` (aom-bench) | 2026-07-17 | speed-0 sf levels (search 0 / size-search 1 / chroma early-term 1); speeds 1–5 levels wired untested-by-gate. Fixed latent UV no-palette-flag under-cost on screen frames (per-leaf `try_palette`). **The 5 EXACT cells are now HARD byte-identity asserts (Section-A-grade regression guards) inside the gate** (2026-07-17 pickup). **The 2 CLOSE 128² cells (`ui_420_128_cq32`, `text_420_128_cq20`) are PINNED** — decode-both localized to genuine palette-induced AB/4-way partition near-ties (`ui`: (mi 0,0) BLOCK_32X32 real HORZ_B vs port HORZ_4; `text`: (mi 8,20) BLOCK_16X16 real VERT vs port VERT_A); both are byte-exact with palette OFF and the palette machinery (`av1_allow_palette` / `av1_get_palette_bsize_ctx`/`_mode_ctx` / k-means / neighbour cache+ctx stamping) is verified C-faithful — same class as the KB-10/KB-11 pinned near-ties (closing needs a sibling-C per-candidate partition-RD dump). Regression-guarded by `decode_diff_palette_close_cells` (asserts the divergence PRESENT → self-promotes on any fix). (CDEF search + loop-restoration search, the first two bulk families, went straight to section A — 14/14 and 8/8 EXACT.) |
| C9 `--use-intra-dct-only=1` (PINNED-OPEN: luma byte-faithful, chroma UV-mode-loop divergence) | `AV1E_SET_INTRA_DCT_ONLY=1` | 64²cq32 / 64²cq63 / 128²cq12 (real content) | +2.23% / 0 (EXACT) / −1.40% | +3.588 (OUT of band) / 0 / +0.333 | `toggles_rd_close::toggles_c9_intra_dct_only_pinned_open` | 2026-07-17 | Y recon identical; first divergent leaf mi(0,0) 32×32: real uv=D45/aduv2 (eob 1) vs port uv=V (eob 78); real winners are derived-type==DCT modes (DCT-forced-search signature). Port UV txb eval + UV mode loop both match the C-pieces oracles under the knob (txfm_uvrd_diff / intra_sbuv_mode_loop_diff sweep green; mask verified vs the REAL facade incl. the PAETH reduced-set reset) ⇒ shared port+oracle mis-model of the REAL UV loop. **Sibling-C dump DONE 2026-07-17** (throwaway ar-swapped libaom, intra_mode_search.c + tx_search.c instrumented, cq32 mi(0,0) 32×32): C evaluates only DC (this_rd 2157931) and D45 (aduv2, this_rd 1985157 — wins); C REJECTS V/H/directionals via `rd_pick_intra_angle_sbuv` anglefail (its inner `av1_txfm_rd_in_plane` returns INT_MAX) and SMOOTH/PAETH via txfmfail. The port instead ACCEPTS V (uv_mode=1, aduv0, DCT-forced tx_type=0, eob=1, **dist=0**, rate 20508 → this_rd 1872917) and V wins. Decisive: C's V prediction `block_sse`=1048576 == the port's V sse=1048576 ⇒ **the prediction MATCHES; NOT a pred bug**. Root = the port's `txfm_rd_in_plane_uv_p` computes V's DCT dist=0 / accepts where C's `av1_txfm_rd_in_plane` rejects V (same pred, same DCT) — a tx-search RD-eval / early-out mis-model shared by the port AND the txfm_uvrd_diff oracle (which is why the differential is green). NEXT: dump C's per-txb V DCT dist/coeffs inside `av1_txfm_rd_in_plane`/`search_txk_type` (the INT_MAX path fires before av1_txfm_uvrd's merge) vs the port's `search_tx_type_intra` V winner, to find why the same DCT residual yields dist=0 in the port and INT_MAX-rd in C. |

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
- Chunk-5 outcome (2026-07-17): the **format axis is now byte-exact** — `mono / 4:4:4 /
  bd12` speed-0 cells assert **3/3 BYTE-IDENTICAL** (`lr_restoration_gate.rs::
  lr_restoration_format_axis`), extending the proven LR-search coverage to 1-plane LR,
  full-res chroma LR, and the highbd-12 path (compute_stats divider 16, SGR 12-bit clamps).
  The allintra speed-1..4 arms (`lr_search_sf_allintra`) and GOOD-mode cells staged in the
  chunk-5 WIP are **LR-orthogonal near-ties, PINNED (not gated)**: a base-vs-LR split
  (throwaway `lr_localize` harness) showed the speed>=1 cells diverge in the BASE encode
  itself — the LR-OFF stream already differs (s1 real content, first byte 3, both off and
  on) — because the port's real-content speed>=1 base encode is not yet byte-exact (KB-6
  proved real content only at speed 0; the KB-8..11 speed gates are synthetic). The GOOD
  cells derive `set_allintra` base speed-features (the harness has no `set_good`), so their
  base search mismatches C's GOOD encode. `lr_search_sf_good` is now source-verified vs
  speed_features.c (:1164 `reduce_wiener_window_size` is UNCONDITIONAL — unlike allintra's
  speed>=3 gate; :1352-1358 is at speed>=3 — the WIP's `speed>=4` was an off-by-one,
  corrected) and ready for a future GOOD gate. speed>=5 allintra is structurally LR-off in
  C (sf disable + seq-bit clear). `pack_tile_from_trees` unification (reuse the CDEF
  two-pass pack instead of the re-search repack) queued as an optimization.
- Decoder-side LR (apply path) was already complete + gated pre-pivot (section A decoder
  rows).
- **DEFAULT-PATH WIRING (2026-07-18): default-config parity closed.** The byte-exact search is
  now wired into the port's DEFAULT allintra path — `aom-bench::EncodeCell::port_encode` derives
  the LR stage from the frame's `enable_restoration && !coded_lossless` (C's `is_restoration_used`,
  encoder.h:4431; the parsed seq bit already encodes C's speed>=5 clear). New gate
  `lr_default_parity.rs` asserts the port's default `port_encode` frame-OBU is **BYTE-IDENTICAL to
  a plain `aomenc --allintra`** (no tool flags) on 8/8 real-content cells, where the reference is
  the new `shim_encode_av1_kf_defaults` (every coding-tool control at its allintra default:
  cdef OFF, restoration ON, qm OFF). It also asserts `c_encode_defaults() == c_encode_lr()`
  (restoration's default IS on; palette/intrabc/deltaq inert on non-screen stills) and that the
  default stream differs from `--enable-restoration=0` even on the all-RU-NONE cell (header bits).
  The explicit-off `encoder_gate_e2e_*` gates stay valid as `--enable-restoration=0` config tests.

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
  `mv_sf.intrabc_search_level`. **SEARCH + skip-arm + full wiring LANDED 2026-07-18** —
  `rd_pick_intrabc_mode_sb` (`aom-encode/src/intrabc_search.rs`) is WIRED (rd_pick.rs step 6 →
  real) and runs under the screen-content gate (`p.allow_intrabc`):
  - **Full-pel DV search: hash + NSTEP diamond + mesh.** The source-frame hash (chunk 3a)
    + the ported `full_pixel_diamond` (NSTEP site config, `diamond_search_sad`
    coarse→fine with the `UPDATE_SEARCH_STEP` num00 collapse) + the `full_pixel_exhaustive`
    mesh (screen `exhaustive_searches_thresh = 1<<20`), SAD-metric walk / variance-cost
    result; the pixel search ALWAYS runs at `intrabc_search_level 0` (rdopt.c:3570).
    Geometry unit-locked (`nstep_config_matches_c`, `mv_step_param_matches_c`,
    `diamond_finds_exact_repeat`). The HASH is square-gated (mcomp.c:1918); the diamond
    runs for every bsize (non-square intrabc supported).
  - **`predict_skip_txfm` (tx_search.c:183)** + `set_skip_txfm` hbd sse scaling: the port
    offers an intrabc candidate ONLY in the skip regime (luma predict_skip fires AND chroma
    is an exact match), where `av1_txfm_search` forces `skip_txfm=1` and BYPASSES the coeff
    arm — there the skip RD (`rate=mode+mv+skip1`, `dist=sse`) is byte-exact.
  - **Wiring:** `LeafWinner`/`RdPickIntraBest` use_intrabc/dv/dv_ref/skip; `ModeGrid` DV grid
    (`dc_screen`, 25 stamp sites) for `find_dv_ref_mvs`; `PickFrameCfg::intrabc`
    (`IntrabcFrameCfg`: hash/dv_costs/txfm_partition_costs/error+sad_per_bit/mv_step_param);
    `encode_b_intra_dry` intrabc arm (predict-from-recon + skip entropy reset + skip txfm ctx);
    `pack_leaf` intrabc arm (use_intrabc + DV diff via `write_mb_modes_kf_fc`, skip tx/coeff);
    harness (`build_intrabc_hash_table` from source luma, `PackCfg::allow_intrabc`, LF forced
    0 for intrabc frames, `ToggleKnobs::enable_intrabc`).
  - **PINNED — real-content byte-exactness blocked on the inter var-tx COEFF ARM (L).** Real
    screen content codes the MAJORITY of intrabc blocks via the COEFF arm (nonzero quantized
    residual) and as NON-SQUARE shapes — measured on a 196² conformance crop: C uses 49
    intrabc blocks, **39 coeff-arm + 42 non-square**, only 10 skip. The
    `av1_pick_recursive_tx_size_type_yrd` inter var-tx quadtree + `prune_tx_2D` /
    `ml_predict_tx_split` NN prunes + the var-tx pack are NOT ported, so the port codes those
    blocks as intra and the frame diverges. Gate `rd_close_intrabc::intrabc_dv_search_pinned`
    (aom-bench) asserts the content is anti-vacuous (C genuinely codes intrabc) and PINS the
    divergence self-promotingly (a byte-match → promote). The 420 skip subset additionally
    needs a chroma-eob-0 check (currently `chroma_sse==0`, exact-only). Envelope untouched:
    non-screen frames (`allow_intrabc=0`) are byte-inert — palette gate + partition_pick_diff
    + rd_pick_intra_sb_diff green.
- Screen detection: `--screen-detection-mode` (allintra default ANTIALIASING_AWARE=2).
  C: `av1/encoder/encoder.c` `av1_set_screen_content_options`. Port takes
  `allow_screen_content_tools` as an input — the detection itself is unported. (S–M)
- `--tune-content` screen/film forcing (gates the above). (S)

### C4 — tune=IQ / tune=SSIMULACRA2 family — **PORTED, BIT-IDENTICAL → section A** (2026-07-17)
The tune bundle (`handle_tuning`, av1_cx_iface.c:1938–1978): `enable_qm=1, qm_min=2,
qm_max=10, sharpness=7, dist_metric=QM_PSNR, enable_cdef=ADAPTIVE, enable_chroma_deltaq=1,
deltaq_mode=6 (VARIANCE_BOOST)`; IQ adds `enable_adaptive_sharpness=1`.
- **DONE — every piece e2e byte-identical to real aomenc + the full composite bundle** (all OFF
  by default; `TuneKnobs::default()` = PSNR, the proven envelope is untouched). Gate:
  `encoder_gate_tune_iq_e2e` (9 tests) + `qm_level_diff` + `tune_shim_smoke`:
  1. QM-level formulas `aom_get_qmlevel_luma_ssimulacra2` + `aom_get_qmlevel_444_chroma`
     (quant_common) — `qm_level_diff`, byte-exact vs the real C static inlines.
  2. QM-PSNR dist metric (`dist_block_tx_domain_qm`; trellis `optimize_txb_qm` +
     tx-search transform-domain distortion weighted by the forward QM only under
     `use_qm_dist_metric`; txb_rdopt.c:346-351/:378-386) + `tune_shim_smoke` anti-vacuous.
  3. `--sharpness` 0..7 (`av1_build_quantizer` rounding bias + trellis + LF level) + witness.
  4. `--enable-chroma-deltaq` (`av1_set_quantizer` chroma delta-q arms, port-derived +
     cross-checked vs the real header).
  5. `--enable-adaptive-sharpness` (qindex-adaptive LF sharpness cap, picklpf.c) + witness.
  6. `--deltaq-mode=6` Variance Boost (`allintra_vis.rs` per-SB source-variance qindex
     modulation; `pack_tile`/`pack_tile_from_trees` per-SB delta-q threading) + witness.
  - **Composite** (`encoder_gate_tune_composite_full_e2e`): the whole `--tune=iq` /
    `--tune=ssimulacra2` bundle live at once, 54/54 cells byte-match (mono/420/444 ×
    64/128/192 × cq12/32/50), proving the knobs compose.
- **One bundle member NOT in the tune port: CDEF_ADAPTIVE.** The composite gate overrides
  `enable_cdef=0` — CDEF is the separate, already-bit-exact C1 track (`av1_cdef_search`), applied
  post-reconstruction so it is symbol-inert on the coded tile bytes. The tune-family port
  deliberately does not own it; a full tune=IQ *with* CDEF-adaptive needs the C1 CDEF search wired
  under the per-SB tune qindex (deferred, cross-track).

### C5 — aq-mode / deltaq-mode variants — PARTIALLY DONE (mode 2 + mode 3 + mode 6 bit-exact → section A)
- `--deltaq-mode=2` DELTA_Q_PERCEPTUAL (wavelet AC energy) — **PORTED, BIT-IDENTICAL → section A** (2026-07-18):
  the 5/3 dwt Haar-AC energy (dwt.c) → `av1_block_wavelet_energy_level` → the rate-ratio segment
  qindex (`av1_compute_q_from_energy_level_deltaq_mode` + `av1_compute_qdelta_by_rate` / the
  `av1_rc_bits_per_mb` KEY/AOM_Q arm / `find_qindex_by_rate`), per-SB threaded through the shared
  `DeltaQFrameCtx` + `av1_adjust_q_from_delta_q_res` (`setup_delta_q_perceptual`). 7/7 real-content
  cells byte-match real aomenc `--deltaq-mode=2` (`deltaq_mode2_perceptual_wavelet_e2e`); the dwt
  kernel is differentially locked vs the exported C (`deltaq_perceptual_wavelet_diff`).
  **Follow-ups (NOT done):** the highbd (bd10/12) dwt arm; the partial-SB source-border extension
  (dims not a multiple of 64px). Note `is_screen_content_type` (the rate enumerator 2M/1M) cancels
  in the ratio, so the non-screen envelope is faithful; a screen-detection port would be needed for
  screen content. (M)
- `--deltaq-mode=3` DELTA_Q_PERCEPTUAL_AI — **PORTED, BIT-IDENTICAL → section A** (2026-07-17):
  `av1_set_mb_wiener_variance` (per-8x8 intra-SATD search + FP-quantize + Weber stats + the
  2-iteration `norm_wiener_variance`) + `av1_get_sbq_perceptual_ai` + `av1_get_deltaq_offset`,
  per-SB pack threading reusing the mode-6 `DeltaQFrameCtx`; 7/7 real-content cells byte-match
  real aomenc `--deltaq-mode=3` (`encoder_gate_deltaq_mode3_e2e`). **Highbd (bd10/12)
  FP-quantize arm — DONE (this landing):** `av1_set_mb_wiener_variance` dispatches
  `av1_highbd_quantize_fp` for bd>8 (the sole bd8-specific step); bd10 (real b10 content) +
  bd12 (bd10 content ×4-promoted) cells added to the gate (16/16 byte-identical). **Follow-ups
  (NOT done):** the partial-SB source-border extension (frames whose
  dims aren't a multiple of 8px — the KB-6 partial-SB analogue); `--enable-rate-guide-deltaq`
  (the `get_rate_guided_quantizer` arm reading an external rate file — needs the file plumbing +
  `ext_rate_guided_quantization`); `--auto-intra-tools-off` (`automatic_intra_tools_off` +
  `model_rd_sse` accumulation, which disables smooth/paeth/cfl/diagonal intra on high-quality
  low-q frames — a separate intra-tool gate).
- `--deltaq-mode=6` DELTA_Q_VARIANCE_BOOST (tune=IQ default): **DONE, BIT-IDENTICAL** (landed
  with C4, fed362b) — `allintra_vis.c`
  `av1_get_sbq_variance_boost`, `aq_variance.c` `av1_get_variance_boost_block_variance`,
  `--deltaq-strength`. (M)
- `--deltaq-mode=1` OBJECTIVE (base default) is TPL-gated (encodeframe.c:343) — **inert
  for a lone still**; document-only.
- `--delta-lf-mode=1` DELTA_LF — **PORTED, BIT-IDENTICAL → section A** (2026-07-18): per-SB
  `delta_lf_from_base` derived from `delta_qindex` in `pack_leaf` + the LF-pick delta-lf
  application (`stamp_lf_delta_lf` + `LfSearchFrame::delta_lf_present`). Rides on a firing
  delta-q mode (gated tested with `--deltaq-mode=2`). **Follow-ups:** `delta_lf_multi=1`
  (per-plane-type deltas — `DEFAULT_DELTA_LF_MULTI=0` so untested), highbd, partial-SB.
- `--aq-mode=1/2` (variance/complexity segmentation): `aq_variance.c` `av1_vaq_frame_setup`
  (VARIANCE_AQ) / `aq_complexity.c` `av1_setup_in_frame_q_adj`+`av1_caq_select_segment`
  (COMPLEXITY_AQ). **Fires single-pass** (encoder.c:3494, NOT two-pass-gated; single-pass
  uses degenerate `avg_energy=0` → `avg_ratio=rate_ratio[0]=2.2`), but is the **first
  single-frame use of SEGMENTATION ENCODE**: it enables `cm->seg`, sets 8 per-segment
  `SEG_LVL_ALT_Q` deltas (`av1_compute_qdelta_by_rate(rate_ratio[i]/avg_ratio)` — the same
  rate model as mode 2, already ported), selects `mbmi->segment_id` per block from the block
  energy (VARIANCE_AQ: `mbmi->segment_id = av1_log_block_var(...)`-mapped, partition_search.c:
  603-608; COMPLEXITY_AQ: `av1_caq_select_segment`, :963), codes `write_segment_id` (writer
  proven), and re-selects per-segment quantizers in the RD search + pack. The seg-map decision
  + per-segment quantizer threading through the search/pack is the remaining work. (M–L,
  segmentation-encode plumbing). NOTE: `av1_log_block_var` is used but a SEPARATE fn from the
  mode-2 wavelet path.
- `--deltaq-mode=4` USER_RATING_BASED: `av1_get_sbq_user_rating_based` reads an EXTERNAL per-SB
  rating map `cpi->mb_delta_q` (from `--rate-distribution-info` / `AV1E_SET_RATE_DISTRIBUTION_INFO`)
  — needs the external-file plumbing; ABSENT.
- `--deltaq-mode=5` HDR / `enable_hdr_deltaq`: `av1_get_q_for_hdr` **asserts bd10** AND is a
  NO-OP under `DISABLE_HDR_LUMA_DELTAQ=1` (encoder.h:101 — returns `base_qindex`, so
  `deltaq_used=0` → `delta_q_present` resets to 0). **INERT** in the shipped build; document-only.
- `--enable-rate-guide-deltaq` / `--rate-distribution-info` (`get_rate_guided_quantizer`,
  allintra_vis.c:688) — needs the external rate-file plumbing (`ext_rate_guided_quantization`);
  ABSENT.
- `--auto-intra-tools-off` (`automatic_intra_tools_off` + `model_rd_sse`, allintra_vis.c:515) —
  needs `--deltaq-mode=3`; disables smooth/paeth/cfl/diagonal intra on high-Q low-q frames via
  a `model_rd_sse` frame accumulation gate. Self-contained (no seg/LF/external), a moderate
  intra-search-space arm; ABSENT.
- Encoder-side per-SB delta-q/delta-lf tile signaling (writer side): **DONE** (delta-q via mode
  2/3/6, delta-lf via `--delta-lf-mode=1`).

### C6 — Superres (encode side) — FIXED + QTHRESH/AUTO/RANDOM + denom-16 scaler DONE; only the AUTO recode loop remains (inter follow-up)
- `--superres-mode/-denominator/-kf-denominator/-qthresh/-kf-qthresh`. Default NONE.
- **DONE (Section A): FIXED mode — 8-bit 13/13 + 10/12-bit 16/16, byte-identical** — source downscale
  (`av1_resize_plane`, `aom_encode::resize`, differentially bit-exact vs exported C:
  `resize_plane_diff` 5 tests) + coded-width encode + `write_superres_scale` header
  signalling. Gate `encoder_gate_superres_{fixed_real_content,fixed_mono}_rd_close`. The
  source downscale for an ALLINTRA KEY still takes the `DISALLOW_RECODE` `encode_without_recode`
  path → `av1_resize_and_extend_frame_nonnormative` → `av1_resize_plane` (verified in
  `reference/libaom`), so there is NO recode loop for FIXED stills.
- **Remaining:**
  1. **Highbd (10/12-bit) downscale — DONE (this landing)** — `highbd_resize_plane`
     (resize.c:771+, u16 `highbd_interpolate`/`highbd_down2_*`) wired as the bd>8 arm in the
     gate's `downscale_plane`; `encoder_gate_superres_fixed_highbd_rd_close` 16/16 byte-identical
     (bd10/12 × 4:2:0 denoms 9/12/14 cq{20,48} + mono denoms 9/12 cq32). Also fixed a
     byte-neutral aom-dist `highbd_variance64` SIMD edge panic (non-mult-of-8 visible-only SSE
     → scalar twin; verified 0-failed in both dispatch modes).
  2. **8-bit denom-16-even-width corner — DONE (2026-07-18, 6b77342)** — the exact-1/2 horizontal
     ratio trips libaom's OPTIMIZED `av1_resize_and_extend_frame` (`aom_scaled_2d`, `EIGHTTAP_SMOOTH`
     / phase 8), ported as `optimized_downscale_plane_8bit` (edge-extend + 16×16-block separable
     8-tap convolve). Differentially bit-exact vs the exported `av1_resize_and_extend_frame_c`
     (`resize_opt_scaler_diff`, 5 content × luma/chroma/other dims); wired into the gate's bd8
     downscale (all-planes-or-none per C). bd8 denom-16 QTHRESH/AUTO cells now emit BYTE-IDENTICAL
     streams to real aomenc.
  3. **AUTO / QTHRESH / RANDOM denom selection — DONE (2026-07-18, 3c8a8c2)** —
     `calculate_next_superres_scale` (superres_scale.c) ported as `aom_encode::superres_select` for
     the single-frame KEY/AOM_Q envelope: `analyze_hor_freq` (16×4 H_DCT energy, bit-exact vs the
     exported `av1_fwd_txfm2d_16x4` — `superres_select_diff` facade + e2e) +
     `get_superres_denom_from_qindex_energy` + the QTHRESH gate (q vs `--superres-kf-qthresh`); AUTO =
     allintra `Dual` (qthresh 0, same energy derivation); RANDOM = the process-global static-seed lcg
     (34567 → 11,14,15,9). `q` is the AOM_Q cq-qindex (`rc::base_qindex_from_cq`, #8). The port
     re-derives the denom the real encoder chose (embedded in the stream) and matches it for EVERY
     cell (QTHRESH 21/21, AUTO 11/11 bd8/10/12; RANDOM 4 draws), then reproduces the coded bytes
     (RANDOM 4/4 byte-identical real content; QTHRESH/AUTO bd8 engaged denoms 9/10/16 byte-identical).
     Gates: `encoder_gate_superres_{qthresh,auto,random}_e2e`. Superres stays OFF by default.
  4. **AUTO recode loop — remaining (inter/GOP follow-up).** `av1_superres_in_recode_allowed` is
     AUTO && !SOLO && `frames_to_key>1`; a single-frame KEY still has `frames_to_key<=1`, so the
     recode loop NEVER fires for it (confirmed by the AUTO e2e denom match — the non-recode denom is
     exact). The `SUPERRES_AUTO_DUAL` multi-pass recode search is only reachable with a multi-frame
     GOP, so it is out of the single-frame KEY scope. **Decoder note:** the port DECODER's superres
     denom-16 (exact-2:1) upscale diverges from C (KB-14, decoder track) — the encoder is byte-exact.

### C7 — Film grain / denoise estimation — table-inject DONE (byte-exact → section A); estimation ABSENT (L)
- **`--film-grain-table` — DONE (this landing, → section A).** Ported `aom_dsp/grain_table.c`
  (`aom_film_grain_table_read` + `grain_table_entry_read` + `aom_film_grain_table_lookup`, plus
  `_write` for fixtures/round-trip) as `aom-encode/src/grain_table.rs`; wired
  `EncodeCell::port_encode_film_grain` to inject the port's own table-derived `FilmGrainParams`
  into the frame header (context fields from the cell), written by the already-bit-exact
  `write_film_grain_params`. Gate `film_grain_gate.rs` byte-matches real aomenc
  `--film-grain-table` (C shims `shim_write_grain_table_test_vector` +
  `shim_encode_av1_kf_film_grain_table`, the latter replicating the plain `encode_kf_pass`
  control set so grain adds ONLY header bytes). See the section-A row.
  - `--film-grain-test` (`AV1E_SET_FILM_GRAIN_TEST_VECTOR`, built-in `grain_test_vectors.h`)
    shares the identical param-plumbing + writer; the table-inject gate uses those 16 vectors as
    the shared fixture source (via the C `aom_film_grain_table_write`), so the test-vector param
    set is transitively covered. A direct `--film-grain-test` e2e gate (reusing the existing
    `ref_encode_av1_kf_film_grain` shim) is a trivial follow-up if a distinct knob is wanted.
- `--denoise-noise-level/-block-size`, `--enable-dnl-denoising`: noise-model ESTIMATION +
  source denoise + grain fit. C: `aom_dsp/noise_model.c` (`aom_denoise_and_model_run` →
  `aom_flat_block_finder_*` → `aom_wiener_denoise_2d` (FFT) → `aom_noise_model_*` →
  `aom_noise_model_get_grain_parameters`). (L — **all `double`/FFT float math**, so a byte-exact
  `--denoise-noise-level` stream is float-determinism-gated; the realistic deliverable is
  per-kernel DIFFERENTIAL validation against the exported `aom_noise_*`/`aom_flat_block_finder_*`
  `_c` functions. Decompose: (2) `noise_strength_solver` (linear system) — **DONE (this landing):**
  `aom-encode/src/noise_model.rs` (`linsolve` + `NoiseStrengthSolver` add/solve/get_center/get_value
  + `NoiseStrengthLut` eval/fit_piecewise), differential `noise_strength_solver_diff.rs` 300/300
  trials bit-identical to the exported `aom_noise_strength_solver_*` / `_lut_eval` /
  `_fit_piecewise` across bd 8/10/12, varying bins/obs + a near-singular case (C-quirk faithful: the
  greedy LUT reduction leaves the `residual` array un-shifted on point removal). (3) `flat_block_finder`
  (planar-model + threshold) — **DONE (this landing):** `FlatBlockFinder` (init lazy-inverse of the
  3-param planar `AᵀA`, `extract_block` planar-fit residual, `run` gradient-covariance eigen features
  + hard `is_flat` threshold + 10th-percentile sigmoid ranking), differential `flat_block_finder_diff.rs`
  48/48 trials bit-identical to the exported `aom_flat_block_finder_init`/`_run` (flat_blocks map +
  num_flat) across bs 16/32 × bd 8/10/12 × exact+partial-edge sizes, anti-vacuous (356 flat / 3060
  non-flat cells). NB the `exp` sigmoid (percentile arm) matches C's glibc `exp` bit-exactly on this
  host; `is_flat` + all features are exact `f64`/`sqrt`. (4) `noise_model` (AR estimate +
  `get_grain_parameters` quantize), (5) `wiener_denoise_2d` + FFT, (6) `denoise_and_model_run`
  orchestrator + encoder wiring — all still ABSENT.)

### C8 — Partition controls — disable arms DONE (byte-exact); SB128 encode DONE (byte-exact)
- **DONE (this landing, → section A):** `--enable-rect-partitions=0`, `--enable-ab-partitions=0`,
  `--enable-1to4-partitions=0`, `--min-partition-size`, `--max-partition-size` + the
  square-only 8..32 interaction arm — all BYTE-IDENTICAL vs real aomenc (same ctrl) on the
  real-content grid, hard-pinned in `toggles_rd_close` (aom-bench). Infra: generic ctrl-pair
  shim `shim_encode_av1_kf_ctrls` + `ToggleKnobs`/`port_encode_with`; ctrl-id constants
  header-cross-checked (`cx_ctrl_ids_match_reference_headers`). C mapping verified:
  `set_max_min_partition_size` (partition_strategy.h:214) `min(sf_default, dim_to_size(px),
  sb)` / `min(max(BLOCK_4X4, dim), sb)`; the auto-max ML arm is inter-only.
- **`--sb-size=128` ENCODE side — DONE ✅ (byte-exact, this landing):** the encoder search+pack
  now walk the frame in 128×128 superblocks and byte-match real `aomenc --sb-size=128` on
  real-image content ≥128px. Three gates in `sb128_e2e.rs` (aom-bench), each with a
  sb128-vs-sb64 anti-vacuity witness: (1) `sb128_forced_split_e2e`
  (`--sb-size=128 --max-partition-size=64`, forced SPLIT at the 128 root, all ≤64 leaves —
  isolates the 128-SB geometry + the 8-way BLOCK_128X128 partition symbol/context/cost + the
  pack walk over 128-SBs); (2) `sb128_natural_e2e` (plain `--sb-size=128` on real `quantizer-00`
  crops — the RD search EVALUATES the 128×128 NONE candidate, exercising the mu-64 SEARCH tx
  walks, then resolves to SPLIT/≤64 on this textured content); (3) `sb128_coded_128_leaf_e2e` (a
  smooth diagonal ramp at 256² cq55/cq63 — the content real aomenc actually resolves to a coded
  128-level partition [anti-vacuity-checked natural≠forced-split], so THIS gate exercises the pack
  `av1_write_intra_coeffs_mb` L/U/V 64-chunk interleave + the >64 re-encode; the photographic
  `quantizer-00` crops split to ≤64 even at cq63 and never reach it). Real-image cells:
  128² (1 SB) + 256² (2×2 SBs) × cq{12,32,63}.
  Harness threads the live SB geometry off the bootstrap seq header's `use_128x128_superblock`
  bit (aom-bench `port_encode_full`: `sb_block`/`sb_mi`/`sb_px`); the 4-way stage gets the C
  `bsize != BLOCK_128X128` gate (partition_search.c:4166). The `av1_foreach_transformed_block
  _in_plane` mu-64 chunk walk (encodemb.c:560-582) is ported into every >64 predict/reconstruct
  site — the two search tx walks (`txfm_rd_in_plane_intra`, `intra_model_rd_y`), the chroma RD
  walk, the two re-encode plane walks (`encode_intra_block_plane_y`/`_uv`) + `encode_b_intra_dry`
  Step 4 — and the pack coeff write is the `av1_write_intra_coeffs_mb` 64-chunk **L/U/V
  interleave** (encodetxb.c:431-472). This closes the **KB-1 encoder cross-check** (the >64-block
  txb order had never actually been exercised — the cited "256² cq63" evidence is an SB64 frame
  with no >64 blocks). `AV1E_SET_SUPERBLOCK_SIZE`(56)/`AOM_SUPERBLOCK_SIZE_128X128`(1) added to
  `cx_ctrl` (header-verified). **Partial-SB-at-128 (frames not a multiple of 128px) also
  byte-exact** — `sb128_partial_sb_e2e` (192² + the KB-6 196² conformance frame × cq{32,63}): the
  KB-6 partial-SB machinery (distortion visible-clips, `set_partition_cost_for_edge_blk`, the
  frame-edge entropy-stamp tail-zero) combines cleanly with the 128-SB geometry + the mu-64
  edge-clip. **Chroma formats — mono (4:0:0) + 4:4:4 coded-128-leaf also byte-exact**
  (`sb128_chroma_format_e2e`): 4:4:4 exercises the chroma mu-64 interleave at ss=0 (chroma chunks ==
  luma chunks), mono the luma-only interleave — proving the mu-64 machinery is ss-generic-correct.
  **One PINNED near-tie: `mono 256² cq63`** (port codes 1 fewer byte, the KB-2/KB-10/KB-12 "cheaper
  RD decision" signature) — NOT a mu-64 bug (4:2:0 + 4:4:4 128-leaves at cq63 AND mono cq55 all
  byte-match; only mono-cq63, with no chroma RD to break a qindex-252 tie, flips), same class as the
  pinned KB-10/KB-11/KB-12 high-qindex near-ties; the gate asserts the divergence PRESENT
  (self-promoting), closing needs a sibling-C RD dump. Deferred: a coded 128-LEAF at a frame edge
  (the partial-SB cells split to ≤64, so the 128-leaf mu-64 edge-clip itself is still untested);
  non-default knob × sb128 combos; speed≥1 × sb128.
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
  policies exist from KB-8; default is 3) (S).
- **`--quant-b-adapt` (the `aom_quantize_b_adaptive` family) — KERNEL PORTED + FUNNEL WIRED
  (this landing):** `aom_quantize_b_adaptive_helper` / `aom_highbd_quantize_b_adaptive_helper`
  (aom-quant; lowbd+highbd, qm+no-qm, the prescan-widened dead-zone `EOB_FACTOR=325` +
  `SKIP_EOB_FACTOR_ADJUST=200` tail) are **bit-identical vs the exported C
  `aom_quantize_b_adaptive_helper_c` / `_highbd_` across 192k random cells** (`quantize_b_adaptive_diff`,
  log_scale 0/1/2 × bd 8/10/12 × qm/no-qm × large+sparse regimes, both eob==0 and eob>0
  exercised). `QuantParams::with_adaptive` routes `QuantKind::B` through the adaptive helper
  in the `xform_quant` funnel. **NOT DONE:** the frame-flag threading for an e2e byte gate —
  `quant_b_adapt` only affects the B quantizer, used with trellis OFF (`USE_B_QUANT_NO_TRELLIS`,
  so the gate needs `--quant-b-adapt=1 --disable-trellis-quant=1`); threading the flag through
  the TxfmYrdEnv/partition_pick/UV-env pipeline (the same deep carriers as
  `disable_trellis_quant`) is deferred as a mechanical follow-up on the proven kernel+funnel. (S–M)

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
- **`--min-q` / `--max-q` — DONE (this landing):** the qindex clamp bounds. For a lone KEY
  still under AOM_Q, `base_qindex = clamp(quantizer_to_qindex(cq), quantizer_to_qindex(min_q),
  quantizer_to_qindex(max_q))` (rc_pick_q_and_bounds_q_mode, ratectrl.c:2158; best/worst_quality
  from rc_cfg, encoder.c:1003). `aom_encode::rc::base_qindex_from_cq_clamped` reproduces the real
  encoder's parsed `base_qindex` across a `(cq, min_q, max_q)` sweep — clamp-down, clamp-up, and
  inert cases (`min_max_q_diff`, via the new `shim_encode_av1_kf_minmaxq` / `ref_encode_av1_kf_minmaxq`
  that set `cfg.rc_min/max_quantizer`; the -1 sentinel leaves every existing caller inert). `--min-cr`
  clamp remains (S).

### C12 — Lossless tail — DONE (→ Section A) — mono + 4:2:0 both byte-exact
- **DONE (KB-5 closed, #32):** coded-lossless cq0 KEY is byte-exact for BOTH mono AND 4:2:0 bd8,
  hard-asserted in `encoder_gate_lossless_cq0_e2e_kb5_repro`. The former "4:2:0 cq0 chroma RD
  near-tie" was a search-SPACE gap (CfL was banned at coded-lossless in the SEARCH), not RD math:
  fixed by routing the leaf `cfl_allowed` through
  `aom_entropy::partition::is_cfl_allowed(bsize, lossless, ss_x, ss_y)` (C allows CfL at lossless
  when the partition size == the transform size). See CLAUDE.md KB-5.
- **Remaining (follow-up, S):** highbd lossless (bd10/12) — the mono+420 byte-exact envelope is bd8.

### C13 — Speed levels 6–9 — DONE (→ Section A) — Gate-2 (cpu 0–9) byte-complete
- **DONE (KB-10 / KB-11 / KB-12):** speeds 6, 7, 8, 9 all landed byte-identical on the synthetic
  canon grids (Section-A cpu-used rows). Speed 6 = new machinery (LPF_PICK_FROM_Q + partition prunes
  + predict_dc skip + 8×8 NN tx-depth prune + winner-mode restructure), 64/64; speed 7 =
  VAR_BASED_PARTITION fixed-tree + `av1_rd_use_partition`, 64/64; speeds 8/9 = the nonrd PICKMODE
  (`av1_nonrd_use_partition` single-pass walk + `av1_nonrd_pick_intra_mode`), speed 9 64/64 +
  speed 8 60/64. `top_intra_model_count_allowed=2` lands at speed≥6. See CLAUDE.md KB-10/11/12.
- **Remaining (pinned near-ties, self-promoting — NOT coverage gaps):** speed-6/7 noise-cq63
  (mi 8,0) TX_16X16-vs-TX_32X32 (KB-10/11); the 4 speed-8 `diag` estimate-arm V/H near-ties
  (KB-12); the nonrd bd10/12 + lossless + screen-palette arms (asserted dead on the 8-bit canon
  grid). Real content at cpu≥1 is a separate residual (KB-13, 24/60 at cpu 1–4).

### Priority order (proposed)
~~1. **C2 LR search**~~ DONE (section A, 2026-07-17) → ~~2. **C1 CDEF search**~~ DONE
(section A, 2026-07-17) → 3. **C3 screen content** (web stills) → 4. **C4
tune=IQ/SSIMULACRA2 tail** (image-quality tuning, small pieces) → 5. **C5 deltaq 3/6** →
6. C8/C9/C10 toggle threading (cheap wins, many S) → 7. C6 superres, C7 film grain →
8. C11/C12 tails → ~~C13 speeds 7–9~~ **DONE (KB-10/11/12, Gate-2 complete)**. (C2/C1
leftovers — LR speed-1..4 e2e arms, CDEF FAST levels e2e — are follow-ups within their families,
below the C3+ fronts.)
