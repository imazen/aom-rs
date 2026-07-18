# AV1 coding-tool / bitstream-syntax coverage audit — aom-rs vs libaom v3.14.1

**Date:** 2026-07-18 · **Audit basis commit:** `79e7a6d3` (origin/main tip, "C6 superres FIXED bd10/12")
**Reference:** `reference/libaom` @ v3.14.1 (`03087864`) · enums from `av1/common/enums.h` + `aom_dsp/txfm_common.h`
**Scope of "the port":** ALLINTRA (usage=2) **KEY-frame** encode + intra-scope decode — the primary gate.

## Purpose & reading guide

This is the definitive, per-syntax-element map of **every coding tool a KEY-frame AV1
bitstream can use**, classified by whether the port covers it on the **ENCODE** side (can the
port *emit* a byte-identical stream using the tool) with a **DECODE** cross-check. It is an
audit, not a plan — no code changed.

**Classification legend (per element):**

- **BYTE-EXACT (enc+dec)** — the port both *encodes* it byte-identical to `aomenc` (proven by an
  e2e byte-match gate) **and** *decodes* it byte-identical to the C decoder (conformance /
  real-bitstream gate). Full two-way coverage.
- **DEC-ONLY** — decode is bit-exact, but the port's encoder never *emits* the tool end-to-end
  (either the RDO/search that would select it is absent, or it is inter-frame-only). The encode
  **symbol writer** may still be differentially proven — noted where so, because that means the
  gap is a search/wiring gap, not a capability gap.
- **PARTIAL** — some variants covered, others not (cited).
- **ABSENT** — no port on either side.

**Evidence priority** (per project CLAUDE.md): real exported C fn > synthetic-facade-over-real-fn
> verbatim transcription. Citations are gate/test names (PARITY.md Section A, the KB log, STATUS.md)
and commit hashes.

**Two structural caveats that qualify every "enc BYTE-EXACT" row below:**

1. **Encoder is SB64-only.** The encode walk + harnesses are 64×64-superblock only
   (`--sb-size=128` encode ABSENT, PARITY C8; decoder + entropy are SB-size-generic, `798ec25`).
   So `BLOCK_128X128 / 128X64 / 64X128` and the 128-level partition node are **DEC-ONLY**. Every
   ≤64 block size / tx size / partition below is reachable and covered.
2. **The e2e encoder gate still bootstraps a few *uncompressed-header* fields** from the reference
   parse (Gate-3 "bootstrap caveat", PARITY rule 4): tile limits, CICP echo, level/tier (qindex
   mapping is self-derived, #8; LF-level is self-derived). The **coded tile-data bytes**
   (partition / mode / tx / coefficients) and all the per-tool header writers are self-derived and
   byte-exact; a handful of seq/frame-header scalars are still echoed. This is a derivation/wiring
   gap, not a missing writer — every header writer is differentially bit-exact (§N).

---

## SUMMARY

**The port covers essentially the entire KEY-frame (stills) coding-tool surface, BYTE-EXACT in
BOTH directions, for the ≤64×64-superblock envelope.** Concretely:

- **Transforms:** all 19 tx sizes (incl. TX_64X64, all rect, all 1:4) + all 16 tx-type *kernels*
  are byte-exact both ways; the **7 intra-reachable tx types** (DCT_DCT, ADST_DCT, DCT_ADST,
  ADST_ADST, IDTX, V_DCT, H_DCT) are e2e enc byte-exact. (The other 9 tx types are inter-only tx
  sets — kernel+symbol proven, not KEY-reachable.)
- **Partitions:** all 10 PARTITION_* types byte-exact both ways at ≤64 SB.
- **Intra:** all 13 Y + 14 UV modes (incl. CfL), angle_delta (±3), all 5 filter_intra modes, intra
  edge filter/upsample — byte-exact both ways.
- **Quantizer:** FP/B/DC × QM/flat × 8/10/12-bit — byte-exact both ways.
- **Entropy:** the whole od_ec + CDF stack, disable-cdf-update — byte-exact both ways.
- **In-loop filters:** deblock, CDEF, loop-restoration (Wiener/SGR/switchable) — byte-exact both
  ways (CDEF/LR search off-by-default in allintra, on via knob).
- **Format axes:** 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 and 8 / 10 / 12-bit — byte-exact both ways.
- **Headers/OBU:** seq-header OBU (vs the REAL exported C fn), frame-header OBU, tile-info,
  tile-group OBU, leb128, film-grain params, color_config — all bit-exact both ways.
- **Superres:** FIXED-denom byte-exact enc+dec (8/10/12-bit); AUTO/QTHRESH/RANDOM denom
  *selection* absent.

### Encode-side gaps (the "does NOT byte-encode this tool" list)

| Tool | Class | Note |
|---|---|---|
| **SB128 encode** (BLOCK_128X128/128X64/64X128 + 128-level partition) | DEC-ONLY | decoder byte-exact (SB128 real-bitstream gate); encoder is SB64-only (PARITY C8) |
| **IntraBC** (screen-content intra block copy) | DEC-ONLY | decoder byte-exact; encoder search skeleton UNWIRED / envelope-inert (PARITY C3, HANDOFF-SCREEN); write_intrabc_info symbol bit-exact |
| **Palette** (screen-content) | PARTIAL enc / DEC-ONLY-byte-exact | decode byte-exact; encode RD-close, **5/7 cells byte-exact**, 2 pinned AB/4-way near-ties (PARITY §B, KB-P29) |
| **Segmentation** (seg map + per-seg qindex/features) | DEC-ONLY | decoder applies it byte-exact (in envelope); encoder never emits (allintra `seg_enabled=false`; aq-mode 1/2 that drive it ABSENT) — write/read_segmentation symbols bit-exact |
| **Delta-LF** (`--delta-lf-mode`) | DEC-ONLY | write/read_delta_lflevel symbols bit-exact; encoder-emit unwired for allintra (PARITY C5); delta-Q *is* emitted |
| **Superres AUTO/QTHRESH/RANDOM** denom selection | ABSENT | FIXED-denom is byte-exact; the denom-*derivation* modes + recode loop absent (PARITY C6) |
| **8-bit denom-16 even-width superres** | ABSENT | trips the optimized `av1_resize_and_extend_frame` scaler, not yet ported (PARITY C6) |
| **Film-grain ESTIMATION** (`--denoise-noise-level`) | PARTIAL | table-inject byte-exact; noise-model estimation: solver + flat-block-finder differential-done, AR-fit / wiener-denoise FFT / orchestrator ABSENT (PARITY C7) |
| **aq-mode 1/2** (variance / complexity AQ) | ABSENT | two-pass-gated; would drive segmentation (PARITY C5) |
| **`--use-intra-dct-only=1`** | PARTIAL | luma byte-faithful; chroma UV-mode-loop divergence, pinned-open (PARITY §B, KB C9) |
| **`--quant-b-adapt`** (adaptive quant-b) | ABSENT | quantizer family variant (PARITY C9) |
| **cost-update-freq knobs** (coeff/mode/dv non-default) | PARTIAL | default arm byte-exact; non-default arms unported (PARITY C11) |
| **`--full-still-picture-hdr` / annexb framing** | ABSENT | framing variants (PARITY C11) |
| **Inter-frame tools** (all of §S) | DEC-ONLY (symbol) | every inter mode-info symbol is roundtrip bit-exact BOTH ways, but inter reconstruction (motion comp) is PARTIAL and there is no inter e2e — out of KEY-frame scope ("the rest") |

Everything not in that list is **BYTE-EXACT (enc+dec)** within the SB64 KEY-frame envelope.

---

## A. Block partitioning & sizes

`enums.h`: `PARTITION_TYPE` (10 + EXT), `BLOCK_SIZE` (BLOCK_SIZES_ALL = 22).

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Partition types | NONE, HORZ, VERT, SPLIT (the 4 base) | BYTE-EXACT (enc+dec) | `write_partition`/`read_partition` roundtrip; write_modes_sb tree walk bit-exact (`34c5d2c`); e2e gates; decoder conformance |
| Partition types (AB) | HORZ_A, HORZ_B, VERT_A, VERT_B | BYTE-EXACT (enc+dec) | STATUS "AB partitions — 10 of 10 PARTITION_* complete" (`1739`); C8 `--enable-ab-partitions=0` byte gate |
| Partition types (1:4) | HORZ_4, VERT_4 | BYTE-EXACT (enc+dec) | STATUS "4-way partitions ported" (`1331`); C8 `--enable-1to4-partitions=0` byte gate |
| Partition controls | `--min/max-partition-size`, square-only band | BYTE-EXACT (enc+dec) | PARITY A "C8 partition-control disable arms" (`toggles_c8_*` hard byte pins) |
| Block sizes ≤ 64×64 | BLOCK_4X4 … BLOCK_64X64 (13 square+rect) | BYTE-EXACT (enc+dec) | RDO picks all; KB-6 real-content map 30/30 exercises the full distribution |
| Block sizes 1:4 | BLOCK_4X16, 16X4, 8X32, 32X8, 16X64, 64X16 | BYTE-EXACT (enc+dec) | KB-2 root was a 16×64 (1:4) leaf; e2e gates + partition_pick_diff |
| Block sizes 128-level | BLOCK_128X128, 128X64, 64X128 | **DEC-ONLY** | decoder SB128 real-bitstream gate (`798ec25`); **encoder SB64-only** (PARITY C8) |
| SB size signalling | `use_128x128_superblock` (seq) | DEC-ONLY | decoder SB128; encoder emits SB64 only |

## B. Transform sizes (TX_SIZE, 19 incl. TX_64X64 + rect + 1:4)

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Square tx | TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64 | BYTE-EXACT (enc+dec) | transform crate COMPLETE (fwd+inv 1D+2D, ~15M cmp); C9 `--enable-tx64=0` byte gate (default tx64 ON) |
| Rect tx (2:1/1:2) | TX_4X8, 8X4, 8X16, 16X8, 16X32, 32X16, 32X64, 64X32 | BYTE-EXACT (enc+dec) | transform crate all sizes; C9 `--enable-rect-tx=0` byte gate |
| Rect tx (4:1/1:4) | TX_4X16, 16X4, 8X32, 32X8, TX_16X64, TX_64X16 | BYTE-EXACT (enc+dec) | transform crate all sizes; coverage checklist `transform.fwd_txfm2d` green (19 sizes) |
| 64-pt coeff repack | TX_64X64/64X32/32X64/64X16/16X64 zero-out | BYTE-EXACT (enc+dec) | checklist note "incl. 64-point coeff repacking, flips, rect Sqrt2 scaling" |
| tx_mode | ONLY_4X4, TX_MODE_LARGEST, TX_MODE_SELECT | BYTE-EXACT (enc+dec) | write/read_tx_mode; C9 `--enable-tx-size-search=0` (→ LARGEST) byte gate; SELECT is default |
| tx-size search / depth | `read/write_selected_tx_size`, var-tx recursion | BYTE-EXACT (enc+dec) | tx_size_cost_diff; recursive `write/read_tx_size_vartx` roundtrip (`ca60884`/`d5c1c5c`) |

## C. Transform types (TX_TYPE, 16) — kernels + ext-tx sets

Intra ext-tx set membership (`av1_ext_tx_used`, blockd.h:1041): a KEY frame reaches **7** tx types
(sets DTT4_IDTX + DTT4_IDTX_1DDCT); the other 9 live only in the inter sets (DTT9_IDTX_1DDCT,
ALL16).

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Intra-reachable tx types | DCT_DCT, ADST_DCT, DCT_ADST, ADST_ADST, IDTX, V_DCT, H_DCT | BYTE-EXACT (enc+dec) | transform crate; `write_tx_type`/`read_tx_type` (ext-tx symbol, `4992bc0`/`ea7c392`); search_tx_type_diff; C9 `--reduced-tx-type-set=1` byte gate |
| ext-tx set derivation | EXT_TX_SET_DCTONLY / DCT_IDTX / DTT4_IDTX / DTT4_IDTX_1DDCT | BYTE-EXACT (enc+dec) | `av1_get_ext_tx_set_type` ported; write_coeffs_txb_full gates tx_type on set-num |
| `use_intra_default_tx_only` | force DCT_DCT | BYTE-EXACT (enc+dec) | C9 `--use-intra-default-tx-only=1` byte gate |
| Inter-only tx types | FLIPADST_DCT, DCT_FLIPADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST | DEC-ONLY (kernel+symbol) | transform crate COMPLETE for all 16 + tx_type reader handles all sets; **not KEY-reachable** (inter tx sets 4/5) — no intra e2e emit |
| flip / idtx seq bit | `reduced_tx_set` / flip-idtx signalling | BYTE-EXACT (enc+dec) | C9 `--enable-flip-idtx=0` byte gate (asserts the seq/tx_mode bit; flip types themselves inter-only) |

## D. Intra prediction

`enums.h`: `PREDICTION_MODE` (INTRA_MODES=13), `UV_PREDICTION_MODE` (UV_INTRA_MODES=14 incl CfL),
`FILTER_INTRA_MODE` (5), `MAX_ANGLE_DELTA=3`.

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Y non-directional | DC_PRED, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH | BYTE-EXACT (enc+dec) | aom-intra core COMPLETE; `write/read_intra_y_mode`; C10 `--enable-smooth-intra=0` / `--enable-paeth-intra=0` byte gates |
| Y directional | V, H, D45, D135, D113, D157, D203, D67 | BYTE-EXACT (enc+dec) | aom-intra directional (z1/z2/z3); C10 `--enable-directional-intra=0` / `--enable-diagonal-intra=0` byte gates |
| angle_delta | ±1, ±2, ±3 (7 per directional mode) | BYTE-EXACT (enc+dec) | `write/read_angle_delta` (`24c04fc`); C10 `--enable-angle-delta=0` byte gate; KB-2 (angle_delta edge-filter) FIXED |
| Intra edge filter / upsample | `get_intra_edge_filter_type`, filter+upsample | BYTE-EXACT (enc+dec) | aom-intra edge DSP; C10 `--enable-intra-edge-filter=0` byte gate; KB-2/KB-6 per-block filter-type recompute |
| UV modes | UV_DC … UV_PAETH (13, mirror Y) | BYTE-EXACT (enc+dec) | `write/read_intra_uv_mode`; UV mode loop diff (intra_sbuv_mode_loop_diff, txfm_uvrd_diff) |
| CfL (chroma-from-luma) | UV_CFL_PRED, joint_sign (8) × alpha (16) | BYTE-EXACT (enc+dec) | `write/read_cfl_alphas` (`abc5df0`); cfl_alpha_search_diff; C10 `--enable-cfl-intra=0` byte gate |
| filter_intra | FILTER_DC/V/H/D157/PAETH (5 modes) | BYTE-EXACT (enc+dec) | `write/read_filter_intra_mode_info` (`65341ce`); C10 `--enable-filter-intra=0` byte gate |

## E. Palette (screen-content intra)

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Palette DECODE | Y+UV, sizes 2–8, color-index map, delta/raw color coding, V-wrap, cache-merge | **BYTE-EXACT (dec)** | decoder real-bitstream palette gate; `read_palette_mode_info` / `read_map_tokens` / `read_palette_colors_*` roundtrip (`de05e99`, `05cc122`) |
| Palette ENCODE (RD search) | Y `av1_rd_pick_palette_intra_sby` + UV `_sbuv` (k-means, colour/map costs, cache/ctx) | **PARTIAL (enc)** | PARITY §B "Palette RD search" — **5/7 cells byte-exact** (hard pins), 2 CLOSE 128² cells PINNED as AB/4-way near-ties (KB-P29); `rd_close_palette` gate |
| Palette symbol writers | flags, size, `av1_get_palette_bsize_ctx`/`_mode_ctx`, colour delta, map tokens | BYTE-EXACT (enc+dec) | `write_palette_mode_info` (`bfc057a`) + readers; the 5/7 byte-exact cells prove the writers compose |

## F. IntraBC (screen-content intra block copy)

| Element | Variants | Class | Evidence |
|---|---|---|---|
| IntraBC DECODE | full-pel DV, 2-tap chroma predictor, mono + colour | **BYTE-EXACT (dec)** | decoder real-bitstream intrabc gate + conformance `-02-allintra` intrabc "extreme DV" vector; `intrabc_{mono,colour}_streams_decode_byte_identical_to_c` |
| IntraBC ENCODE (RD search) | `rd_pick_intrabc_mode_sb`, DV hash, NSTEP+mesh search, coeff arm | **DEC-ONLY (enc absent)** | PARITY C3 — search skeleton landed but **UNWIRED / envelope-inert** (coeff arm, hbd sse, full-pel search, integration all MISSING; HANDOFF-SCREEN); no e2e emit |
| IntraBC symbols | `write/read_intrabc_info`, `av1_encode_dv` (DV via MV coder at MV_SUBPEL_NONE) | BYTE-EXACT (enc+dec) | `dfa07b7` write; roundtrip reader; DV-ref `find_dv_ref_mvs` C-validated (`dv_ref_diff`) |

## G. Quantization

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Scalar quantizer | FP (`quantize_fp`), B (`quantize_b`), DC (`quantize_dc`) | BYTE-EXACT (enc+dec) | aom-quant complete; DC quantizer (`7c575bc`); `dequant_txb` (`b1e260c`) dec side |
| Bit-depth | lowbd (8) + highbd (10/12) quant families | BYTE-EXACT (enc+dec) | `av1_highbd_quantize_fp`/`aom_highbd_quantize_b`/`_dc`; KB-4 bd10/12 e2e |
| Quant matrices (QM) | flat + `qm[rc]`/`iqm[rc]` (all qm/iqm 1..255), qm_min/max levels | BYTE-EXACT (enc+dec) | QM subsystem complete (`8c64524`/`addbca5`/`11d8cdb`); PARITY A `--enable-qm` (#23) 40-cell byte gate; decoder QM real-bitstream gate |
| Trellis (optimize_txb) | flat + QM, NO_TRELLIS / FINAL_PASS arms | BYTE-EXACT (enc+dec) | optimize_txb[_qm]; C9 `--disable-trellis-quant=1/2` byte gates (`5a644c6`, caught 2 real pack bugs) |
| Coded-lossless | WHT/IWHT 4×4 path, qindex 0 | BYTE-EXACT (enc+dec) | KB-5 mono+420 byte-exact (`encoder_gate_lossless_cq0`); decoder coded-lossless real-bitstream gate |
| `--quant-b-adapt` | adaptive quant-b | **ABSENT** | PARITY C9 |

## H. Entropy / CDF coding

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Range coder | Daala od_ec encode + decode | BYTE-EXACT (enc+dec) | aom-entropy COMPLETE (~40k seqs); STATUS "Entropy COMPLETE symbol stack" |
| CDF adaptation | `update_cdf`, `aom_write/read_symbol`, ns-symbol CDFs | BYTE-EXACT (enc+dec) | ~1M cdf-update cmp; every symbol roundtrip in partition_diff (93 tests) |
| Literals / bits | `aom_write/read_bit`, `_literal` (MSB-first) | BYTE-EXACT (enc+dec) | cdf module (`e532e07`) |
| Coefficient coding | txb_skip, eob token+extra, base/br levels, dc/ac sign, golomb | BYTE-EXACT (enc+dec) | `write_coeffs_txb`/`read_coeffs_txb` (`3b3d38d`); read_coeffs_diff ~42k |
| disable-cdf-update | `cdf-update-mode=0` (no adaptation) | BYTE-EXACT (enc+dec) | PARITY A C11 `--cdf-update-mode=0` byte gate (caught a real pack bug); decoder `disable_cdf_update_diff` |
| Default CDF tables | all frame-context init tables | BYTE-EXACT (enc+dec) | `default_cdfs.rs`; frame-init CDF used by KB-6 edge partition gather |

## I. In-loop filters

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Deblocking (loop filter) | 4/6/8/14-tap, vert+horz, luma+chroma, level derivation | BYTE-EXACT (enc+dec) | decoder real-bitstream deblock + 4:2:2 chroma deblock (STATUS `441`); enc LF-level RD search `av1_pick_filter_level` byte-exact (STATUS `1553`); loopfilter header write/read |
| LF params / mode-ref deltas | `mode_ref_delta_enabled`, delta update | BYTE-EXACT (enc+dec) | `encode_loopfilter`/`read_loopfilter` (decoder-faithful gate) |
| CDEF | `cdef_bits`, per-64×64 strength (pri/sec luma+chroma), find_dir | BYTE-EXACT (enc+dec) | decoder real-bitstream CDEF; enc `--enable-cdef=1` `av1_cdef_search` 14/14 byte gate (PARITY A C1); OFF by default in allintra |
| Loop restoration | RESTORE_NONE/WIENER/SGRPROJ/SWITCHABLE, RU sizes 64/128/256 | BYTE-EXACT (enc+dec) | decoder real-bitstream LR; enc search 8/8 byte gate (PARITY A C2, `write_lr_unit`); **ON by default in allintra** (speeds 0-4; OFF at speed>=5) — wired into the port default path + gated by `lr_default_parity` (port default == plain no-flags `aomenc --allintra`), 2026-07-18 |
| CDEF FAST search levels | LVL1..5, PICK_FROM_Q, ADAPTIVE | PARTIAL (enc) | FULL search e2e-gated; FAST levels table-tested not e2e; PICK_FROM_Q/ADAPTIVE not ported (PARITY C1) |

## J. Superres

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Superres DECODE (upscale) | denom 9..16, horizontal upscale + loop-restore boundary | BYTE-EXACT (dec) | decoder real-bitstream superres gate (STATUS `803`); `superres.rs` |
| Superres ENCODE — FIXED denom | `--superres-mode=fixed`, denom 9/12/14/16, 8/10/12-bit | BYTE-EXACT (enc+dec) | PARITY A C6 — bd8 13/13 + bd10/12 16/16 byte-identical; `av1_resize_plane`/`highbd_resize_plane` (resize_plane_diff); `write_superres_scale` |
| Superres header | superres_scale, denom signalling | BYTE-EXACT (enc+dec) | `write/read_superres_scale` |
| AUTO / QTHRESH / RANDOM denom | `calculate_next_superres_scale`, recode loop | **ABSENT** | PARITY C6 — the denom-*derivation* modes (`analyze_hor_freq`, `get_superres_denom_from_qindex_energy`) unported |
| 8-bit denom-16 even-width | optimized `av1_resize_and_extend_frame` | **ABSENT** | PARITY C6 — that ratio trips a different (unported) scaler; gate asserts out |

## K. Segmentation

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Segmentation DECODE | seg map (spatial+temporal pred), per-seg features, qindex apply | BYTE-EXACT (dec, in-envelope) | decoder frame.rs "segmentation IS in the envelope" + `bridge_segmentation`→`av1_get_qindex`; caveat: mixed-lossless-segments stays OUT |
| Segmentation header | `encode/read_segmentation` (feature data, signed max tables) | BYTE-EXACT (enc+dec, symbol) | header_diff (`689c6d7` family) + `read_segmentation` (`6d6dd2d`) |
| Per-block segment_id | `write/read_segment_id`, `av1_neg_interleave`, spatial-pred ctx | BYTE-EXACT (enc+dec, symbol) | `9281f22` + roundtrip reader |
| Segmentation ENCODE-EMIT | encoder turns seg on (aq-mode / seg update) | **DEC-ONLY** | encoder allintra `seg_enabled=false` (pack.rs:234); aq-mode 1/2 ABSENT (PARITY C5) — writer proven, never emitted |

## L. Delta-Q / Delta-LF

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Delta-Q params (header) | `delta_q_present`, `delta_q_res` | BYTE-EXACT (enc+dec) | `write/read_delta_q_params`; decoder applies (`delta_q_present`) |
| Per-SB delta-qindex | `write/read_delta_qindex` (exp-golomb) | BYTE-EXACT (enc+dec) | `e532e07` + roundtrip; per-SB pack threading |
| Delta-Q ENCODE (deltaq-mode 3) | DELTA_Q_PERCEPTUAL_AI (wiener-variance) | BYTE-EXACT (enc+dec) | PARITY A C5 `--deltaq-mode=3` 7/7 byte gate (`deltaq_mode3_e2e`) |
| Delta-Q ENCODE (deltaq-mode 6) | DELTA_Q_VARIANCE_BOOST (tune=IQ) | BYTE-EXACT (enc+dec) | PARITY A C4/C5 (`fed362b`); `allintra_vis.rs` |
| Per-SB delta-lf-level | `write/read_delta_lflevel` (exp-golomb, multi) | BYTE-EXACT (enc+dec, symbol) | `950c42a` + roundtrip |
| Delta-LF ENCODE-EMIT | `--delta-lf-mode` wired into allintra pack | **DEC-ONLY** | PARITY C5 — symbol proven, encoder-emit unwired |
| deltaq-mode 1 (OBJECTIVE) | TPL-gated | ABSENT (inert for lone still) | PARITY C5 — document-only |
| aq-mode 1/2, rate-guide-deltaq, auto-intra-tools-off | variance/complexity AQ + rate file | **ABSENT** | PARITY C5 |

## M. Film grain

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Film-grain SYNTHESIS (decode) | AR coeffs, scaling points, Gaussian, chroma-from-luma | BYTE-EXACT (dec) | decoder conformance film-grain-synthesis + `film_grain_diff`; `film_grain.rs` + `film_grain_gaussian.rs` |
| Film-grain params (header) | scalars + point/coeff arrays, all 16 test vectors | BYTE-EXACT (enc+dec) | `write/read_film_grain_params` roundtrip |
| Film-grain ENCODE (`--film-grain-table`) | own grain-table reader → params → header | BYTE-EXACT (enc) | PARITY A C7 `film_grain_gate` byte-match; `grain_table.rs` |
| Film-grain ESTIMATION | `--denoise-noise-level` (noise model, denoise, fit) | **PARTIAL / ABSENT** | PARITY C7 — `noise_strength_solver` + `flat_block_finder` differential-DONE; AR-fit / `wiener_denoise_2d` FFT / `denoise_and_model_run` orchestrator ABSENT (float-determinism-gated) |

## N. Frame / sequence / tile headers & OBU framing

| Element | Variants | Class | Evidence |
|---|---|---|---|
| OBU header | type/ext/has_size byte(s) | BYTE-EXACT (enc+dec) | `write/read_obu_header` (`ca3f9df`) |
| leb128 (OBU size) | uleb encode/decode | BYTE-EXACT (enc+dec) | `a3bbfcf` |
| Sequence-header OBU | profile, level/tier, op-points, timing, decoder-model, color-config, film-grain bit | BYTE-EXACT (enc+dec) | enc vs the **REAL exported** `av1_write_sequence_header_obu` (`cbda510`, 100k); `read_sequence_header_obu` roundtrip (`942185c`); `seq_header_matches_real_encoder` |
| Frame-header OBU (uncompressed) | full KEY + INTER frame-header assembly, all gating | BYTE-EXACT (enc+dec) | frame-header pieces 200-300k each (`dc9c167`…`ad2fdbd`); KEY+INTER spec anchors (`6ff7573`/`d3c230a`); `read_uncompressed_header` !error asserted (`aa19113`); `frame_header_matches_real_encoder` |
| Frame-header prefix state machine | show_existing / frame_type / error_resilient / order_hint / primary_ref / refresh | BYTE-EXACT (enc+dec) | `write/read_frame_header_prefix` (`6bdef82`, 300k) |
| Tile info | uniform + non-uniform spacing, tile_log2, context-update tile, size-bytes | BYTE-EXACT (enc+dec) | `write_tile_info` + `read_tile_info` (`6c415f4`); multi-tile e2e (`f6e6319`, 2×1/1×2/2×2) |
| Tile-group OBU | tile framing, size fields | BYTE-EXACT (enc+dec) | `tile_group_obu_matches_real_encoder`; decoder `tile_roundtrip` |
| Frame size / render size / superres | frame_size, render_size, frame_size_with_refs | BYTE-EXACT (enc+dec) | `write/read_frame_size*` (`1208d07`/`53c6ddc`) |
| Quantization header | base_qindex, plane deltas, qm bits | BYTE-EXACT (enc+dec) | `encode/read_quantization` (`c4c8eb9`) |
| CDEF / LR / LF / tx_mode / delta-q header blocks | per-tool frame-header content | BYTE-EXACT (enc+dec) | header module 14 components (`689c6d7` family) + readers |
| interp-filter (frame) | SWITCHABLE + 3 fixed | BYTE-EXACT (enc+dec) | `write/read_frame_interp_filter` (inert on KEY recon, header bit covered) |
| Bootstrap-caveat fields | tile limits, CICP echo, level/tier | **PARTIAL (enc self-derive)** | writers bit-exact; the e2e harness still echoes these from the ref parse (qindex + LF are self-derived) — PARITY C11 |

## O. Chroma subsampling

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Monochrome | 4:0:0 (num_planes=1) | BYTE-EXACT (enc+dec) | enc mono lossless + mono e2e gates; decoder monochrome conformance |
| 4:2:0 | ss_x=ss_y=1 | BYTE-EXACT (enc+dec) | PRIMARY config; KB-6 real-content 30/30; decoder conformance |
| 4:2:2 | ss_x=1 ss_y=0 | BYTE-EXACT (enc+dec) | `encoder_gate_chroma_ss_e2e`; decoder 4:2:2 chroma deblock (STATUS `441`) |
| 4:4:4 | ss_x=ss_y=0 | BYTE-EXACT (enc+dec) | `encoder_gate_chroma_ss_e2e`; multi-tile 4:4:4 |

## P. Bit depths

| Element | Variants | Class | Evidence |
|---|---|---|---|
| 8-bit | lowbd | BYTE-EXACT (enc+dec) | primary; all e2e gates |
| 10-bit | highbd | BYTE-EXACT (enc+dec) | KB-4 `kb4_gate_bd10_bd12`; `encoder_gate_bd10_diff`; decoder b10 conformance |
| 12-bit | highbd | BYTE-EXACT (enc+dec) | KB-4 bd12 mono+420; superres bd12; transform bd-independent verified |

## Q. Color / CICP signalling

| Element | Variants | Class | Evidence |
|---|---|---|---|
| color_config | CICP primaries/transfer/matrix, color_range, chroma_sample_position, separate_uv_delta_q, mono | BYTE-EXACT (enc+dec, symbol) | `write_color_config`/`read_color_config` (`f5de42e`) differential; profile-derived subsampling |
| CICP echo (encoder self-derive) | frame emits CICP from config | PARTIAL (enc) | writer proven; e2e harness bootstraps CICP echo (PARITY C11) |

## R. Lossless

| Element | Variants | Class | Evidence |
|---|---|---|---|
| Coded-lossless (all planes, qindex 0) | mono + 4:2:0, WHT 4×4 | BYTE-EXACT (enc+dec) | KB-5 FIXED — `encoder_gate_lossless_cq0` hard-asserts mono+420; decoder coded-lossless real-bitstream gate |
| Segment-mixed lossless | some-but-not-all segments lossless | DEC-ONLY (out of decode envelope) | decoder frame.rs rejects mixed-lossless-segments upstream; KB-5 tail note |

## S. Inter-frame tools (NOT KEY-reachable — secondary; symbol-layer bit-exact only)

A KEY frame never codes these. They are listed because the port has the **entropy symbol layer**
bit-exact in **both** directions (roundtrip-validated, memory log "ENTIRE INTER MODE CODING
bit-exact"), but there is **no inter reconstruction e2e** and the decoder conformance gate is
intra-scope, so none are enc-or-dec *frame-level* proven → **DEC-ONLY (symbol)** / effectively
out of the current audit surface ("the rest").

| Element | Variants | Class | Evidence (symbol only) |
|---|---|---|---|
| is_inter / intra_inter | per-block inter flag | DEC-ONLY (symbol) | `write/read_is_inter`, `get_intra_inter_context` (`89ba3b2`) |
| Single-ref frames | LAST/LAST2/LAST3/GOLDEN/BWDREF/ALTREF2/ALTREF | DEC-ONLY (symbol) | `write/read_ref_frames` (`235fa3e`) + 13 pred-contexts |
| Compound refs | unidir (9 pairs) + bidir (12 pairs), comp_ref_type | DEC-ONLY (symbol) | `write_ref_frames` uni/bi cascades |
| Inter modes | NEAREST/NEAR/GLOBAL/NEW MV | DEC-ONLY (symbol) | `write/read_inter_mode` (`ba82547`) |
| Compound inter modes | NEAREST_NEAREST … NEW_NEW (8) | DEC-ONLY (symbol) | `write_inter_compound_mode` (`37ae60f`) |
| MV coder | joint + component (class/int/fr/hp), NMV | DEC-ONLY (symbol) | `av1_encode_mv` / `read_mv` (`25f1a1e`/`e951588`), 300k |
| DRL | dynamic ref-list index | DEC-ONLY (symbol) | `write/read_drl_idx` |
| Interp filter (block) | EIGHTTAP/SMOOTH/SHARP, dual | DEC-ONLY (symbol) | `write_mb_interp_filter`; motion-comp convolve PARTIAL (only EIGHTTAP_REGULAR lowbd) |
| Motion mode | SIMPLE / OBMC / WARPED_CAUSAL | DEC-ONLY (symbol) | `write/read_motion_mode`; recon ABSENT |
| Compound type | AVERAGE / DISTWTD / WEDGE / DIFFWTD | DEC-ONLY (symbol) | compound-type group (`9e7d5dd`/`c44ef55`) |
| Interintra | II_DC/V/H/SMOOTH + wedge | DEC-ONLY (symbol) | `3580537` |
| Skip mode | frame skip_mode_flag + per-block | DEC-ONLY (symbol) | `write/read_skip_mode` (`c67926c`) |
| Global motion | IDENTITY/TRANSLATION/ROTZOOM/AFFINE + subexp params | DEC-ONLY (symbol) | `write/read_global_motion` (`4dacfc3`), all 7 refs, 300k |
| Var-tx (inter) | recursive txfm_partition tree | DEC-ONLY (symbol) | `write/read_tx_size_vartx` (`826939f`) — used only by inter |

**Inter reconstruction status:** motion-comp convolve is PARTIAL (`av1_convolve_{x,y,2d}_sr`
EIGHTTAP_REGULAR lowbd only, 120k); OBMC / warped / compound-mask blend / inter-pred highbd
kernels ABSENT. This is the multi-month "the rest" per the project gate ("single-frame byte-exact
across both tracks before inter starts").

---

## Provenance

- **Author:** aom-rs coverage-audit agent, 2026-07-18. Report-only (no source changed).
- **Sources:** `reference/libaom/av1/common/enums.h` + `aom_dsp/txfm_common.h` +
  `av1/common/blockd.h` (tool enums / ext-tx sets); `PARITY.md` Section A/B/C;
  `CLAUDE.md` Known-Bugs (KB-1…KB-13, KB-P29); `STATUS.md` section headers; the project
  auto-memory symbol-layer log; crate source under `crates/aom-{transform,quant,entropy,txb,
  intra,encode,decode,cdef,loopfilter,restore,convolve,dist}`.
- **Method:** enumerate every element from the reference enums, then classify each against the
  landed differential/e2e/conformance gates (evidence priority: real exported C fn >
  facade-over-real > transcription). Encode-side is the primary lens; decode is the cross-check.
- **Known staleness corrected:** PARITY.md Section C rows C12 (lossless 4:2:0) and C13 (speeds
  6–9) are stale — superseded by KB-5 (lossless FIXED) and KB-10/11/12 (speeds 6–9 done). This
  audit reflects the current KB/Section-A/STATUS state, not the stale Section-C text.
