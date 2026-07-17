# aom-rs status

Reference target: **libaom v3.14.1** (`03087864`). Oracle built from source
(single-thread deterministic config, `reference/BUILD_CONFIG.md`).

## Done (bit-exact vs C oracle, differential-fuzz verified)

- **C7 film-grain table-inject** (`--film-grain-table`, encoder side) — the port's own
  grain-table reader/lookup (`aom-encode/src/grain_table.rs`, port of `aom_dsp/grain_table.c`)
  feeds the already-bit-exact `write_film_grain_params` header writer. E2E byte-identical vs real
  aomenc `--film-grain-table` across 4:2:0/mono/4:4:4/bd10 × built-in test vectors 1/2/6/15.
  Gate: `crates/aom-bench/tests/film_grain_gate.rs` (+ no-bootstrap-leak witness). Root fix during
  landing: the grain C shim must replicate the plain `encode_kf_pass` control set (CDEF/restoration/
  deltaq/aq off) — grain is decode-side synthesis, so the coded tiles MUST equal the plain encode
  (the port reproduces plain tiles + injects the header grain block). Grain ESTIMATION
  (`--denoise-noise-level`, `aom_dsp/noise_model.c`) remains ABSENT — L, float/FFT-gated (see PARITY C7).
- **Forward 1-D transforms** (`av1_fwd_txfm1d.c`), all 12 kernels:
  `fdct{4,8,16,32,64}`, `fadst{4,8,16}`, `fidentity{4,8,16,32}`.
  Harness: `crates/aom-transform/tests/txfm1d_diff.rs`. Coverage: 4.8M
  differential comparisons (100k random inputs × 4 cos_bit × 12 kernels) +
  edge cases, all byte-identical to C.

- **Forward 2-D transform** (`av1_fwd_txfm2d.c`), all 19 tx sizes: config tables
  (flip/vtx/htx/cos_bit/txfm_type/shift), `av1_round_shift_array`, the row/col
  composition with flips + rectangular Sqrt2 scaling + transpose, and the
  64-point coefficient repacking. Harness: `tests/txfm2d_diff.rs`. Coverage: all
  193 valid (tx_type × tx_size) combos, ~386k full-transform differential
  comparisons + edge cases, byte-identical to C.
  Oracle needs `av1_rtcd()` init (some `_c` entry points call SIMD-dispatched
  helpers); handled once in `aom-sys-ref::ref_init`.

- **Inverse 1-D transforms** (`av1_inv_txfm1d.c`), all 12 kernels:
  `idct{4,8,16,32,64}`, `iadst{4,8,16}`, `iidentity{4,8,16,32}`. Adds a live
  per-stage `clamp_value(stage_range)` (transpiler extended to track stage index
  + emit clamps). Harness: `tests/inv_txfm1d_diff.rs`, 4.8M differential
  comparisons + edge cases, byte-identical to C. (Decoder track.)

- **Inverse 2-D transform + reconstruction** (`av1_inv_txfm2d.c`), all 19 sizes:
  inverse config tables, `av1_gen_inv_stage_range` (reduces to a bd-constant),
  32-cap input remapping, `NewInvSqrt2` rectangular scaling, per-stage clamp,
  `clamp_buf` stages, and `highbd_clip_pixel_add` reconstruction. Harness:
  `tests/inv_txfm2d_diff.rs`, all 193 combos × bd{8,10,12}, ~405k full-frame
  reconstruction comparisons + edge cases, byte-identical to C. (Decoder track.)

**→ The transform subsystem is now complete: forward 1-D/2-D + inverse 1-D/2-D,
both tracks, fully bit-exact.**

- **Quantization** (`av1_quantize.c`, `aom_dsp/quantize.c`), encoder track:
  `av1_quantize_fp` family (no-qmatrix fast path, log_scale 0/1/2) and
  `aom_quantize_b` (dead-zone + quant/quant_shift). Harnesses:
  `aom-quant/tests/quantize_{fp,b}_diff.rs`, ~480k differential comparisons
  (qcoeff + dqcoeff + eob) + edge cases, byte-identical to C.
  (Quant-matrix path + adaptive variants: TODO.)

- **Highbd (10/12-bit) quantizers** (`av1_quantize.c`, `aom_dsp/quantize.c`):
  av1_highbd_quantize_fp + aom_highbd_quantize_b (no qmatrix), 64-bit paths,
  byte-identical to C over log_scale{0,1,2} x 12-bit magnitudes.
  Harness: `aom-quant/tests/highbd_quant_diff.rs`.

- **Quant-matrix (QM) quantizers — full family (aom-quant)**: the per-position
  weighted paths of all four scalar quantizers — aom_quantize_b_qm /
  aom_highbd_quantize_b_qm (from `aom_*quantize_b_helper_c`) and av1_quantize_fp_qm
  / av1_highbd_quantize_fp_qm (the QM branch of `*quantize_fp_helper_c`). The two
  b-quantizers diff against the exported C helpers directly; the two (static) fp
  helpers are reached through the *real* facades (`av1_*quantize_fp_facade`) via
  `shim/quant_fp_shim.c` — no transcription. Harness:
  `aom-quant/tests/quantize_qm_diff.rs`, n={16,64,256,1024} x ls{0,1,2} x 2000,
  qm/iqm over the full qm_val_t range 1..=255, byte-identical qcoeff/dqcoeff/eob.
  With the no-qm variants this closes the entire scalar quantizer surface
  (b + fp, lowbd + highbd, qm + flat). Adaptive `_adaptive` variants: TODO.

- **DC-only quantizer (aom-quant)**: av1_quantize_dc / av1_highbd_quantize_dc — the
  AV1_XFORM_QUANT_DC path (quantize coefficient 0 only, zeroing the rest). Reached
  through the real av1_*quantize_dc_facade (no transcription). With FP + B this
  completes the quant_func_list (FP/B/DC) x (lowbd/highbd) x (flat/QM). Harness:
  quantize_dc_diff.rs. Wired into aom-encode as QuantKind::Dc across all 3 harnesses.

- **Entropy coder** (`aom_dsp/entenc.c`, `entdec.c`), both tracks: the Daala
  `od_ec` range coder. Encoder (`od_ec_enc`) produces byte-identical output to C
  (uint64 low + backward carry propagation + flush); decoder (`od_ec_dec`,
  uint32 dif window + refill) recovers identical symbols. Harness:
  `aom-entropy/tests/entropy_diff.rs` — ~40k random op sequences, encode
  byte-exact + decode symbol-exact + pure-Rust round trip. Oracle via a C shim
  (`aom-sys-ref/shim/entropy_shim.c`) exposing opaque handles. TODO:
  `od_ec_enc_bits` (raw literals) + the `aom_writer`/`aom_reader` CDF-adaptation
  layer on top.

- **CDF adaptation + symbol coding** (`update_cdf`, `aom_write_symbol`,
  `aom_read_symbol`): completes the symbol-coding stack on top of `od_ec`.
  Harness: `aom-entropy/tests/cdf_diff.rs` — `update_cdf` bit-exact over 1M
  updates; adaptive encode byte-identical + adaptive decode symbol-identical to
  C over 10k sequences.

- **Intra prediction (non-directional)** (`aom_dsp/intrapred.c`), both tracks:
  DC / DC_top / DC_left / DC_128 / V / H / Paeth / Smooth / Smooth_V / Smooth_H,
  generic over all 19 block sizes. Harness: `aom-intra/tests/intra_diff.rs` —
  10 modes x 19 sizes x 2000 = 380k comparisons, byte-identical to C.
  (Directional z1/z2/z3 predictors + highbd: TODO.)

- **Intra prediction (directional)** (`av1/common/reconintra.c`), both tracks:
  `av1_dr_prediction_z1/z2/z3` over valid angle-derived (dx,dy) + `dr_intra_derivative`
  table. Harness: `aom-intra/tests/dir_diff.rs` — ~4k angle x size x upsample
  combos, byte-identical to C. Core intra prediction family now complete.
  (Highbd: TODO.)

- **Intra edge filter / upsample DSP** (`av1/common/reconintra.c`), both tracks:
  intra_edge_filter_strength + av1_use_intra_edge_upsample (verified EXHAUSTIVELY)
  + av1_filter_intra_edge_c (5-tap, sz 2..65) + av1_upsample_intra_edge_c
  (sz 1..16), byte-identical to C. Highbd (10/12-bit) filter+upsample too. Directional intra pre-conditioning complete (both bit depths).

- **Deblocking loop filter (lowbd)** (`aom_dsp/loopfilter.c`), both tracks:
  horizontal + vertical, widths 4/6/8/14 (filter4/6/8/14, hev/flat/flat2 masks,
  signed-char-clamp domain). Harness: `aom-loopfilter/tests/lpf_diff.rs` —
  240k comparisons over branch-exercising pixel/threshold strategies,
  byte-identical to C. (Highbd variants: TODO.)

- **Distortion metrics (encoder-critical, speed-0 path)** (`aom_dsp/sad.c`,
  `variance.c`): `aom_sad`, `aom_variance`, `aom_sub_pixel_variance` (bilinear
  2-tap) over all 22 block sizes. Harness: `aom-dist/tests/dist_diff.rs` —
  ~198k comparisons (SAD + variance + subpel var/sse), byte-identical to C.
  The full SAD family (sad/avg/masked/obmc, lowbd+highbd) is complete.

- **Masked SAD (wedge / diff-weighted compound)** (`aom_dsp/sad_av1.c`),
  both tracks: `aom_masked_sad*_c` + `aom_highbd_masked_sad*_c`, all 22 sizes,
  A64-mask blend + invert_mask, byte-identical to C.

- **Avg-SAD (compound prediction)** (`aom_dsp/sad.c`), both tracks:
  `aom_sad*_avg_c` + `aom_highbd_sad*_avg_c` (SAD vs round(ref+second_pred)/2)
  over the 17 non-4-side sizes libaom compiles in this config. Byte-identical to
  C. (ref_init needed — avg-SAD calls the RTCD-dispatched aom_comp_avg_pred.)

- **Highbd (10/12-bit) SAD + variance** (`aom_dsp/sad.c`, `variance.c`),
  encoder critical path: `aom_highbd_sad*` + `aom_highbd_{8,10,12}_variance*` over
  all 22 sizes × 3 bit depths. Harness: `aom-dist/tests/hbd_dist_diff.rs` —
  ~297k comparisons (SAD + variance + sub-pixel variance), byte-identical
  to C. The highbd distortion trio (speed-0 highbd motion-search / RDO) is complete.

- **Coefficient-coding kernels (aom-txb)** (`av1/encoder/encodetxb.c`,
  `av1/common/txb_common.{h,c}`, `av1/common/scan.c`), encoder critical path —
  the first step toward `av1_write_coeffs_txb` / `av1_cost_coeffs_txb`:
  `av1_txb_init_levels` (transposed-layout padded level map, exact write
  footprint) + `av1_get_nz_map_contexts` (per-coefficient entropy contexts, all
  3 TX classes; both RTCD-dispatched hot kernels), `av1_get_eob_pos_token`, the
  `av1_nz_map_ctx_offset` tables (5,232 entries, exact 19-way alias mapping),
  and the full `av1_scan_orders` scan+iscan tables (20,064 entries, 19x16).
  Harness: `aom-txb/tests/txb_diff.rs` — table data entry-for-entry vs C, all
  1024 eob tokens, kernels byte-identical over 19 sizes x 7 tx_types x 300
  iters (~160k context-array comparisons).

- **av1_write_coeffs_txb (aom-txb)** — the first module that emits real encoder
  **bitstream bytes**, byte-identical to C. Full symbol chain on aom-entropy's
  bit-exact od_ec (txb_skip, EOB token 16..1024 + extra bits, coeff_base_eob /
  coeff_base, coeff_br range loop, DC-sign + raw signs, golomb) with in-lockstep
  update_cdf. Harness: `write_coeffs_diff.rs` — bytes AND adapted CDF arena
  identical over 19 sizes x 7 tx_types x plane{0,1} x update{on,off} x 40 blocks.
  (av1_write_tx_type / plane-0 tx_type signaling: next.)

- **av1_cost_coeffs_txb (aom-txb)** — the RD-cost twin of the writer and the
  single hottest speed-0 function (per-candidate-txb in mode / tx-type search):
  same symbol chain, sums LV_MAP_COEFF_COST / LV_MAP_EOB_COST entries instead of
  emitting bits (get_eob_cost, get_br_cost/golomb, get_br_ctx_eob, 3-phase scan
  walk). Integer-identical to C. Harness: `cost_coeffs_diff.rs` — 19 sizes x 7
  tx_types x 120 random (cost-table, block) pairs.

- **av1_cost_tokens_from_cdf (aom-txb)** — the CDF -> per-symbol cost-table
  derivation feeding cost_coeffs_txb: av1_prob_cost[128] + av1_cost_symbol
  (verified across ALL 32767 Q15 probs) + AOM_ICDF differencing / EC_MIN_PROB
  floor / inv_map. `prob_cost_diff.rs`, ~36k derivations. Coefficient-coding RD
  loop (CDF -> costs -> cost_coeffs_txb) is now end-to-end bit-exact.

- **av1_fill_coeff_costs (aom-txb)** — assemble the per-(txs_ctx, plane)
  LV_MAP_COEFF_COST tables from a frame's coeff CDFs (the production source of
  cost_coeffs_txb's tables). Newly verified: base_cost[4..7] trellis-diff +
  lps_cost cumulation/diff fixups. `fill_diff.rs`, 4000 random CDF sets. The
  coefficient-coding cost pipeline (frame CDFs -> fill -> LvMapCoeffCost ->
  cost_coeffs_txb) is end-to-end bit-exact.

- **ext-tx derivation + av1_write_tx_type (aom-txb)** — the plane-0 tx_type
  selection + signaling the coeff functions left out; closes the full-txb write
  path. TxSetType selection / eset / symbol-index / arity / intra-dir tables,
  emitted via bit-exact aom_write_symbol. `ext_tx_diff.rs` — derivation
  EXHAUSTIVE over the full (tx_size x is_inter x reduced x tx_type x
  filter-intra) space; composed write byte-identical.

- **trellis per-coefficient cost helpers (aom-txb)** — the per-candidate-level
  building blocks of av1_optimize_txb: get_br_cost_with_diff (golomb tables +
  level-1-vs-level `diff`), get_two_coeff_cost_simple (cost + cost_low),
  get_coeff_cost_eob/general. Integer-identical to C (`trellis_cost_diff.rs`).

- **av1_optimize_txb — the coefficient trellis (aom-txb), BOTH paths** — RD-optimal
  coefficient rounding, the largest/hottest speed-0 function. Full trellis
  (update_coeff_general/_eob/_simple + update_skip, RDCOST/get_coeff_dist/
  get_qc_dqc_low) byte-identical to C: optimized qcoeff/dqcoeff + reduced eob +
  rate. `optimize_diff.rs` — 19 sizes x 7 tx_types x 60 self-consistent blocks x
  sharpness 0..7. **Quant-matrix path (`optimize_txb_qm`)** shares one core: the
  two QM-dependent helpers — get_dqv (folds iqmatrix: `(iqm*dqv+16)>>5`) and
  get_coeff_dist (folds qmatrix: `((diff*qm)^2+512)>>10`) — take `Option<&[u8]>`,
  non-QM delegates with None/None (behavior unchanged). Helpers diffed directly vs
  the real C static inlines (400k cases, in-module); full QM trellis diffed via
  `optimize_qm_diff.rs` (transcribed shim threaded with the real inlines). The
  rate path (cost_coeffs_txb) is QM-independent, so no other trellis change.

- **Framing / OBU-wrapper primitives (aom-entropy)** — the byte-level writers that
  wrap the (already bit-exact) coefficient coding into an AV1 bitstream: `WriteBitBuffer`
  (aom_write_bit_buffer — the byte-aligned MSB-first bit writer for the uncompressed
  headers, distinct from od_ec), the `leb128` varint codec (aom_uleb_encode/decode/
  size_in_bytes — OBU sizes, UINT32_MAX-capped), and `write_obu_header`
  (av1_write_obu_header byte output). All diffed vs C (leb128/wb via exported C or a
  driver shim; the OBU header via a verbatim transcription + an independent spec anchor).
  The `WriteBitBuffer` also carries the **subexpfin primitive family** (recenter_nonneg/
  finite / quniform / subexpfin / refsubexpfin / write_signed_primitive_refsubexpfin),
  validated on its own vs the real aom_wb (`wb_diff.rs`, 900k cases over the GM ranges).

- **Sequence-header OBU — COMPLETE and validated against the REAL exported C.**
  `write_sequence_header_obu` (= `av1_write_sequence_header_obu`) assembles the whole
  OBU payload byte-for-byte: profile, still/reduced flags, timing + decoder-model info,
  the operating-points loop (idc / level / tier / per-op decoder+display model params),
  then the sequence-header body, color config, film-grain flag, and trailing bits. The
  oracle is the **direct exported function** — the shim populates a real `SequenceHeader`
  (by field name) and calls the actual `av1_write_sequence_header_obu`, so a misreading
  of the C shows as a byte mismatch (not a shared transcription bug). `header_diff.rs`
  100k spec-valid random headers, byte-identical, no assert trips. This is the first
  complete OBU in the encoder track.

- **Uncompressed-header content components (aom-entropy `header` module)** — 19 of the
  sequence/frame-header content pieces, each byte-identical to C libaom and diffed at
  200k+ random cases in `header_diff.rs`: `encode_quantization`, `encode_loopfilter`,
  `encode_cdef`, `encode_segmentation` (real exported av1_seg_feature_data_max/signed),
  `write_frame_interp_filter`, `write_superres_scale`/`write_render_size`/
  `write_frame_size`, `write_tile_info` (+ `wb_write_uniform`), `encode_restoration_mode`,
  `write_delta_q_params` + `write_tx_mode`, `write_film_grain_params`,
  `write_global_motion` (subexp-coded model params, all 7 refs), `write_sequence_header`
  (+ write_sb_size), `write_ext_tile_info` (+ byte_align_zeros), `write_color_config`
  (+ write_bitdepth), and `write_timing_info_header` / `write_decoder_model_info` /
  `write_dec_model_op_parameters`. The `WriteBitBuffer` also carries `write_uvlc` and the
  subexpfin family, each independently validated vs the real aom_wb. Component oracles are
  transcribed control flow over the **real aom_wb primitives** (debug-only asserts + xd
  side effects omitted — no byte effect); quant has an independent spec-layout anchor.

- **Frame-header OBU — every piece bit-exact; assembled but the top-level stitch is not
  yet end-to-end differential-tested.** All of `write_uncompressed_header_obu`'s parts are
  ported and diffed at 200k–300k cases each in `header_diff.rs`:
  `write_frame_header_prefix` (show-existing / frame-type / show / error-resilient /
  screen-content / integer-mv / frame-size-override / order-hint / primary-ref /
  buffer-removal loop / refresh flags / ref order hints), `write_frame_size_with_refs`,
  `write_inter_ref_signaling` (short-signaling + per-ref map idx + modular delta-frame-id
  + rtc config), `write_refresh_frame_context`, and `write_frame_header_trailing_flags`;
  the tail is the already-bit-exact component set. `write_frame_header_obu` composes all
  of these in the C's exact order (threading the prefix-computed frame_size_override).
  **HONEST GAP:** the assembly compiles and calls only validated pieces, but the 40-line
  ordering/gating dispatch is NOT yet differentially tested end-to-end — the function is
  `static` and needs the sub-writers composed over one shared aom_wb (a shim `_wb`-inner
  refactor + monolithic oracle). Until that lands, treat the stitch order as unverified.
  Remaining framing after that: the per-superblock tile-data delta-q/delta-lf + mode-info
  signaling (aom_writer arithmetic-coder path, not aom_wb).

- **RD-search primitives (aom-encode `rd` module + aom-dist)** — the estimator set the
  mode search composes, all bit-exact: `model_rd_from_var_lapndz` (the fixed-point
  Laplacian rate/dist model from variance+qstep, 3 q10 tables extracted from rd.c;
  exported oracle, 720k cases), `pixel_distortion` (reconstruction-domain SSE:
  `pred + inv_txfm(dqcoeff)` clamped vs source — composes the validated inverse
  transform + SSE), and `sum_squares_i16` / `sum_squares_2d_i16` (the residual energy
  feeding the model). With the trellis rate + block_error, the RD-cost inputs are all
  in place; the remaining RDO piece is libaom's search *control flow* (candidate order,
  early-termination, hashing).

- **RD + front-end primitives (aom-dist)** — `block_error` / `highbd_block_error`
  (av1_[highbd_]block_error_c): transform-domain distortion, `error=sum((coeff-dqcoeff)^2)`
  + `ssz=sum(coeff^2)` (lowbd 32-bit products; highbd 64-bit + rounded-shift 2*(bd-8)) —
  the fast distortion the RD search pairs with the trellis rate. `block_error_qm`
  (av1_block_error_qm — the quant-matrix RD distortion; a static inline, so the oracle
  is a transcription cross-validated against the real av1_highbd_block_error_c in the
  flat-matrix case + the transcription for the weighted case). `subtract_block` /
  `highbd_subtract_block` (aom_[highbd_]subtract_block_c): the residual generator
  `diff=src-pred`, natively strided — completing the front end (pred -> subtract ->
  xform_quant). Both diffed vs exported C (highbd subtract via a CONVERT_TO_BYTEPTR shim).
  `sum_squares_i16` / `sum_squares_2d_i16` (residual energy), `vector_var`
  (motion/RD vector variance; `mean_abs^2` in unsigned 32-bit), and the **SATD family
  completed both bit depths**: `hadamard_32x32` (lowbd) + `highbd_hadamard_8x8/16x16/32x32`
  (distinct i16-first / i32-second passes, no column swap) — all diffed vs exported C.

- **Transform-block loop — one coding-block plane (aom-encode)** — encode_coding_block_plane
  iterates a plane's txbs in raster order (av1_foreach_transformed_block_in_plane),
  threading the above/left ENTROPY_CONTEXT arrays: get_txb_ctx reads the neighbour context,
  the txb encodes (encode_block_coeffs_full), then av1_set_entropy_contexts (interior memset)
  fills the txb footprint with its entropy byte for the next txb. Differential re-runs the
  loop with the C context/quant/optimize refs + the C-validated write: byte-identical bytes,
  final contexts, and both adapted CDFs across 7 tilings (square+rect, 2x2..4x4) x plane 0/1
  x FP/B/Dc x bd 8/12 x QM/flat. Uniform tx only; frame-edge clipping + block->tx partition +
  chroma subsampling are the remaining lifts to a full coding block.

- **Complete plane-0 txb bitstream: av1_write_tx_type wired in (aom-txb)** — the full
  av1_write_coeffs_txb order (txb_skip -> luma tx_type -> coefficients). The eob+coeffs
  payload is factored into a shared write_txb_body, so write_coeffs_txb (coeff-only) is
  byte-unchanged and write_coeffs_txb_full reuses it; tx_type is written (via the ported
  write_tx_type) iff luma + ext-tx set >1 + the qindex/skip/seg gate. Oracle: the shim
  gains the tx_type params + ext-tx CDF and writes the same symbol (real
  av1_get_ext_tx_set_type / av1_num_ext_tx_set / av1_ext_tx_ind); signal_gate=0
  reproduces the coeff-only oracle. write_txb_full_diff.rs (19 tx x 16 tt x intra/inter x
  reduced x filter-intra x plane 0/1 x gate/update) + aom-encode's encode_block_coeffs_full
  (residual -> complete txb bytes, encode_block_full_diff.rs) are byte-identical to C with
  both CDF buffers adapting in lockstep.

- **Full speed-0 block coefficient pipeline, residual -> bitstream bytes (aom-encode)**
  — the new composition crate wiring the per-block encoder from the already bit-exact
  sub-modules. `xform_quant` = av1_xform (av1_fwd_txfm2d) + av1_quant (quantizer
  dispatch FP/B x flat/QM + entropy-ctx), with av1_get_tx_scale log_scale,
  av1_get_max_eob n_coeffs, use_optimize_b deferral. `xform_quant_optimize` adds
  get_txb_ctx + av1_optimize_b (trellis, or av1_cost_skip_txb at eob 0) + the final
  entropy-ctx write. `encode_block_coeffs` adds av1_write_coeffs_txb on the od_ec range
  coder, so a residual becomes **byte-identical entropy-coded coefficient bytes** with
  identical CDF adaptation. Three harnesses (coeff / optimized-coeff / bytes+CDF) each
  chain the C reference steps as oracle and cover all 19 tx sizes x 7 tx types x FP/B x
  QM/flat (x update on/off for the writer), with coverage guards (nonzero-eob fraction,
  trellis-reduced-eob, byte-producing fraction). av1_write_tx_type (plane-0 tx_type) out
  of scope on both sides. **Both bit depths**: the AV1 forward transform is bd-independent
  in this build (verified bd 8 == bd 12 output — stage_range only feeds the disabled
  forward range-check), and the trellis / entropy-ctx / writer all operate on quantized
  coefficients, so highbd differs *only* in the quantizer family. QuantParams.bd > 8
  dispatches the 64-bit highbd quantizers; the residual->bytes capstone is byte-identical
  to C at bd 8 AND bd 12.

- **per-block entropy-context propagation (aom-txb)** — the neighbour-context
  loop gating every txb (both tracks): get_txb_ctx (above/left -> txb_skip_ctx /
  dc_sign_ctx; general algorithm verified vs C's size-specialised variants) +
  av1_get_txb_entropy_context (block -> packed neighbour ctx). `entropy_ctx_diff.rs`.

## Coverage gate (auto-derived, honest)

`xtask/coverage.py` enumerates the live libaom feature surface (aomenc/aomdec
`--help` + `aomcx.h` control enums) = **349 features**; a feature is green only
if `coverage/feature_map.json` maps it to a passing test. Current: **0/349**
(no kernel maps to a *complete* CLI feature yet). This is the truthful coverage
state. Kernel-level differential coverage is tracked separately in
`checklist.json` (transform/quant/entropy/intra/loopfilter/dist all green).

- **CDEF direction search** (`av1/common/cdef_block.c`), both tracks:
  `cdef_find_dir` (8x8 partial-sum direction cost search) over bd 8/10/12.
  Harness: `aom-cdef/tests/cdef_diff.rs` — 600k comparisons (dir+var),
  byte-identical to C. (cdef_filter_block + full CDEF path: TODO.)

- **Inter-pred convolution (SR, EIGHTTAP_REGULAR)** (`av1/common/convolve.c`),
  both tracks / encoder critical path: `av1_convolve_x_sr` + `av1_convolve_y_sr`
  (separable 8-tap FIR + subpel kernel table). Harness:
  `aom-convolve/tests/convolve_diff.rs` — 80k comparisons validating ported
  filter table + FIR math, byte-identical to C. (2D_sr, dist_wtd, smooth/sharp/
  4-tap filters, highbd: TODO.)

- **CDEF filter block** (`av1/common/cdef_block.c`), both tracks: all 4
  `cdef_filter_8_{0,1,2,3}` variants (primary/secondary enable combos),
  `constrain` + directional primary/secondary taps + CDEF_VERY_LARGE clipping,
  plus the u16-store `cdef_filter_16_*` twins (highbd + the u16-plane bd-8
  store). Harness: `aom-cdef/tests/cdef_filter_diff.rs` — 320k + 240k
  comparisons, byte-identical to C. CDEF kernels complete lowbd + highbd;
  the frame application walk is green too (see the CDEF milestone below).

## Safety: #![forbid(unsafe_code)]

All 8 shipping crates (aom-transform/quant/entropy/intra/loopfilter/cdef/
convolve/dist) enforce `#![forbid(unsafe_code)]` — zero `unsafe`. SIMD uses
**archmage** `#[autoversion]` (path dep `/root/work/archmage`), not raw
`core::arch` intrinsics. The only `unsafe` in the repo is the test-only C-FFI
differential oracle `aom-sys-ref` (a dev-dependency that links reference
libaom; FFI is inherently unsafe and is isolated there).

## Infrastructure standing

- Rust workspace + `aom-sys-ref` FFI oracle crate linking the reference `libaom.a`.
- Transpiler `xtask/transpile_txfm1d.py` for the regular ping-pong butterflies.
- Coverage ledger `coverage/checklist.json`.

## Next candidates

1. **Top-level uncompressed-header assembly** — `write_uncompressed_header_obu` /
   `write_sequence_header_obu` now that all 14 content components are bit-exact: the
   ordering + the frame-type / show-frame / ref-frame-signaling state machine
   (order_hint, primary_ref_frame, ref_frame_idx, delta_frame_id, refresh flags),
   plus the seq-header framing (profile/timing/decoder-model/color-config) and the
   inline trailing flags (reference_mode/skip_mode/warped/reduced_tx_set). Needs an
   AV1_COMMON-shaped state struct; produces a full header byte-for-byte.
2. **Per-superblock tile-data signaling** — the aom_writer (arithmetic-coder) side of
   delta-q/delta-lf + the mode-info symbols, distinct from the aom_wb header path.
3. **Mode & partition search (RDO)** — the hardest bit-identity target; drives
   encode_coding_block_plane per candidate (RD-cost inputs are all bit-exact now).
4. **Intra prediction** (`av1/common/reconintra`, `aom_dsp` intra predictors) —
   per-mode bit-exact, differential per predictor.
5. AVX2/NEON SIMD specializations (perf gate), each diffed lane-level vs scalar.
6. **Decoder conformance corpus run** (gate 1) — wire the AV1 conformance vectors +
   libaom decode tests through the ported decoder path.

## Real-bitstream decode milestone (2026-07-14, decoder track)

**First real-bitstream decode landed** (b8d79b2): `aom_decode::frame::decode_frame_obus`
parses full temporal units (OBU walk + all headers via aom-entropy readers) and drives
the KEY-frame tile decoder from `KfFrameContext::default_for_qindex` (b11e00c — libaom's
default CDF tables generated from source, byte-identical to the compiled
`av1_setup_past_independence` across all 4 coeff-CDF qindex bands). Gate-1-shaped test
(`aom-decode/tests/real_bitstream.rs`): **56 bitstreams from the REAL libaom v3.14.1
encoder** (`aom_codec_av1_cx` public API = the aomenc path, `--cpu-used=0 --end-usage=q`)
decode **byte-identical on all planes vs the REAL C decoder** (`aom_codec_av1_dx`).
Envelope (hard-errors outside it, never mis-decodes): one shown KEY frame, sb64,
single tile, no CDEF/restoration/film-grain/superres/screen-content/qm/segmentation/
lossless, `disable_cdf_update` off, loop-filter levels 0 (verified per stream from our
own parse; cq grid probed 2026-07-14). Covered: 64x64 / 96x80 / 100x76 (8px-aligned mi
crop), 4:2:0 + 4:4:4 + monochrome, bd 8 + 10, q 8..144 (all four TOKEN_CDF bands),
TX_MODE_SELECT live. Deferred-shim backlog closed the same session: CfL kernels +
tx-size/chroma facades now diffed DIRECTLY vs C (dec_shim.c), shared-misread risk
retired.

## Deblocking applied — lf==0 envelope constraint DROPPED (2026-07-14, decoder track)

**The biggest envelope constraint is gone** (10a5fae application layer, e7c96ab wiring):
`aom_loopfilter::frame` ports the `av1_loop_filter_frame` application walk — the
`lpf_opt_level == 0` single-threaded decoder path: `check_planes_to_loop_filter` →
`av1_loop_filter_frame_init` (sharpness limits + hev + the seg/ref/mode-delta level
table) → per-32-mi-strip → plane → dir(vert, horz) → SB-col
`av1_filter_block_plane_vert/horz` with per-4px-line `set_lpf_parameters` (tu-edge
gate, skip rules `(curr||pv) && (!pv_skip || !curr_skip || pu_edge)`, min-tx filter
length luma {4,8,14}/chroma {4,6}, level fallback to the non-skipped side), dispatching
into the already-bit-exact kernels. u16 planes at every depth (hbd kernels at bd 8 ==
C's lowbd path — proven by the diff, whose C side runs the REAL lowbd path).
Oracle is REAL exported functions, zero transcription: `shim_lf_filter_frame`
(dec_shim.c §4) builds a synthetic `AV1_COMMON` + per-cell `MB_MODE_INFO` grid and
drives `av1_filter_block_plane_vert/horz` + `av1_loop_filter_frame_init` in
`loop_filter_rows` order. `lf_apply_diff.rs`: 400 init-table cases + 224 whole-frame
cases (6 shapes incl. non-8-multiple crops + multi-strip, 420/444/mono full + 422
luma-only, bd 8/10/12, intra/inter/intrabc/skip cells, per-cell vartx, delta-lf,
seg LF features, lossless segments, sharpness 0-7, strided buffers) + zero-level
no-op + zero-DERIVED-level walk-runs-but-touches-nothing — all byte-identical;
harness mutation-verified. `real_bitstream.rs` now: **87 real cpu-used=0 KEY streams
byte-identical** — 76 level-0 + **11 genuinely deblocked (6 also chroma)**, incl.
the previously-excluded bd10 cq>6 arm and (100x76,444) cq36. The one wiring bug the
real gate caught (synthetic diff could not): aligned-vs-crop dims passed as the
`dst.width` guard — fixed to header frame dims. **4:2:2 chroma deblocking is now
byte-exact and IN the envelope** (2026-07-15 — see the "4:2:2 chroma deblock"
milestone below; the earlier "unportable BLOCK_INVALID OOB" claim was VERIFIED
FALSE — the branch is dead for conformant 4:2:2). Next envelope tools: CDEF
(kernels partial), segmentation, palette, SB128, multi-tile, superres, intrabc
(loop restoration landed 2026-07-14 — see below).

## 4:2:2 chroma deblock — byte-exact; the "unportable OOB" claim was FALSE (2026-07-15, decoder track)

The decode envelope had rejected 4:2:2 (ss=(1,0)) streams with nonzero chroma
filter levels, citing an unportable OOB read of `max_txsize_rect_lookup[BLOCK_INVALID]`
in libaom's chroma loop filter "for tall blocks." That was a past-session
assumption, not a measured fact — **VERIFIED FALSE**. Rejection removed (frame.rs);
4:2:2 chroma deblock is byte-exact vs the C decoder.

What the OOB actually is, and why it is DEAD code:
- `av1_ss_size_lookup[bsize][1][0]` (common_data.c:17) marks the eight PORTRAIT
  (height>width) luma sizes `BLOCK_INVALID` (4x8/8x16/16x32/32x64/64x128/4x16/
  8x32/16x64). `av1_get_max_uv_txsize`→`get_plane_block_size` indexes
  `max_txsize_rect_lookup` with that, so a portrait `mbmi->bsize` at ss=(1,0)
  would read OOB (asserts compiled out in the `-DNDEBUG` Release oracle). The
  read is a static-array OOB, not a frame-buffer border read.
- But portrait luma blocks are **structurally impossible** in a conformant 4:2:2
  stream. `decode_partition` (decodeframe.c:1359-1371) rejects any partition
  whose `subsize` maps to `BLOCK_INVALID` under the chroma subsampling
  (`AOM_CODEC_CORRUPT_FRAME`) BEFORE the block is created; the encoder gates
  every VERT / VERT_4 / VERT_A / VERT_B split on the same check
  (partition_search.c). So the chroma loop filter (and reconstruction, which
  uses the identical raw-`bsize` lookup with no guard) never sees a portrait
  block. The `BLOCK_INVALID` branch is unreachable for valid 4:2:2.
- MEASURED: instrumented the real libaom decoder's `set_lpf_parameters` (the
  decoder's `lpf_opt_level=0` path) and swept the real encoder+decoder over 4:2:2
  across sizes / cq / cpu-used and content engineered to force vertical splits —
  **ZERO OOB hits**; the chroma path only ever sees wide/square bsizes. The
  encoder never even produces portrait blocks in 4:2:2. Instrumentation reverted.

Ported the same conformance check into the Rust `decode_partition` (lib.rs,
scoped to ss=(1,0)) so a malformed portrait-partition 4:2:2 stream fails fast
like C instead of an opaque OOB panic in reconstruction; inert for conformant
streams (exercised, non-firing, by the 14-stream gate).

Gate: `deblocked_422_chroma_byte_identical_to_c` (real_bitstream.rs) — 14 real
4:2:2 KEY streams, 8- and 10-bit, sizes incl. non-8-multiple widths (130/100/66 →
ODD chroma half-widths 65/33), full-frame Y/U/V byte-identical to the C decoder.
ANTI-VACUOUS: asserts ss=(1,0); ≥8 streams with NONZERO U/V filter levels (up to
31); ≥8 whose chroma pixels the deblock ACTUALLY changed (recomposed without the
deblock stage); both bit depths; odd chroma half-width present; real chroma AC
content (non-flat plane). Full aom-decode + aom-entropy regression stays green —
all 4:2:0 / 4:4:4 / mono / palette / qm / lossless / intrabc / sb128 / multi-tile /
film-grain / superres gates byte-identical (this change touches only the 4:2:2
path). Commits: 42423ab (deblock + gate), <partition-check> (conformance guard).

## CDEF applied — enable_cdef envelope constraint DROPPED (2026-07-14, decoder track)

**The second post-filter is in** (000e413 walk, edb4a55 wiring): `aom_cdef::frame`
ports the `av1_cdef_frame` application walk — the single-threaded decoder path:
per-64x64 fb, strength index from the mi grid's unit-top-left `cdef_strength`
(the literal read at the fb's first non-skip block; `-1` skips), Y/UV
`level = s/4` / `sec = s%4` (+1 when 3), `is_8x8_block_skip` all-skip
aggregation into the dlist (any of the 2x2 mi non-skip ⇒ filtered),
`cdef_prepare_fb` border priming (16-bit src at CDEF_BSTRIDE: colbuf saves the
fb's UNFILTERED right columns for the next fb's left border, ping-pong top +
bottom linebufs hold pre-filter rows across fb-row boundaries, `cstart=-8`
reads the frame directly when the left fb was unfiltered, frame edges fill
CDEF_VERY_LARGE), and `av1_cdef_filter_fb` per-8x8 dispatch (luma primary
variance-adjusted via `cdef_find_dir`, chroma damping-1, `dir=0` when the
plane primary is 0, conv422/conv440 remap applied once on plane 1). u16
planes at every depth (bd-8 u16 store proven value-identical to C's u8 path).
Oracle is the REAL EXPORTED `av1_cdef_frame` + `av1_cdef_init_fb_row`
(`shim_cdef_frame`, dec_shim.c §5 — synthetic AV1_COMMON + per-cell mi grid,
work buffers per the `av1_alloc_cdef_buffers` single-worker formulas).
`cdef_frame_diff.rs`: 420 whole-frame cases (8 shapes incl. non-mult-64 +
8-aligned crops + 24x16, 4:2:0/4:2:2/4:4:4/mono, bd 8/10/12, damping 3..6,
strength grids incl. zero-Y/zero-UV/all-zero slots, the -1 arm, skip kinds
none/mixed/heavy/all) — byte-identical, plus a padding-invariance proof (the
two sides get DIFFERENT out-of-frame padding; identical output pins that the
aligned-row linebuf reads never influence pixels). `real_bitstream.rs` now:
**126 real cpu-used=0 KEY streams byte-identical** — the 87-stream envelope
stays green and 39 `--enable-cdef=1` arms join: 29 carry CDEF syntax (ten
cq-6 arms search to all-zero strengths and code none) and **all 29 genuinely
change pixels** (verified by recomposing the pipeline without the CDEF stage),
cq-52 bd8 arms chain deblock+CDEF. Envelope after this chunk: loop
restoration, film grain, segmentation, palette/screen-content, SB128,
multi-tile, superres, qm, lossless, intrabc, 4:2:2-chroma-deblock still
rejected; CDEF has no residual constraint.

## Loop restoration applied — enable_restoration constraint DROPPED (2026-07-14, decoder track)

**The third (and last) post-filter is in** (ec0cfbd syntax, df2f893 tile parse,
6296d65 kernels, 7ee08bc walk, 52615ce wiring+gate). Four layers, each C-diffed:

- **RU-params syntax** (`aom_entropy::lr`): `read_primitive_{quniform,subexpfin,
  refsubexpfin}` + `inv_recenter` on the od_ec decoder; `read_wiener_filter` /
  `read_sgrproj_filter` / `read_lr_unit` with per-tile reference chaining
  (`av1_reset_loop_restoration` defaults); 3 new LR CDFs in `KfFrameContext`
  (dump-diff extended to 6431 u16, green all bands). Oracle: REAL exported
  `aom_write/read_primitive_refsubexpfin` + `aom_write/read_symbol` roundtrip
  (`shim_lr_units_roundtrip`) — values + adapted CDFs identical, 400 cases.
- **Tile parse** (aom-decode): `decode_partition` runs the
  `loop_restoration_read_sb_coeffs` reads at the 64x64 SB root (parse arm,
  decodeframe.c:1325-1343) over the `av1_loop_restoration_corners_in_sb`
  rectangle — THE RU PARAMS ARE INTERLEAVED IN THE TILE DATA before each SB's
  first partition symbol. Geometry (`lr_count_units`/`lr_corners_in_sb`) diffed
  vs REAL `av1_alloc_restoration_struct` + corners fn incl. an exactly-once
  RU coverage proof. 624-tile synthetic roundtrip unchanged (no-LR streams).
- **Kernels** (NEW crate `aom-restore`): `wiener_convolve_add_src` =
  `av1_[highbd_]wiener_convolve_add_src_c` (fixed 8-tap separable, u16
  intermediate, `get_conv_params_wiener` rounding, h+7 intermediate rows —
  the highbd h+8th is dead work, verified against both C variants); `sgr` =
  `av1_[apply_]selfguided_restoration_c` (boxsums, A/B intermediate with u32
  wrapping muls, r=2 fast + r=1 full passes, `av1_decode_xq`, i16-narrowed
  projection). Diffed vs the REAL `_c` exports — bd 8 through the production
  LOWBD u8 kernels — over 12/11 dim sets x 3 bd x taps/eps sweeps.
- **Frame walk** (`aom_restore::frame`): `av1_loop_restoration_filter_frame` +
  `av1_loop_restoration_save_boundary_lines` in the decoder's ordering:
  after_cdef=0 saves DEBLOCKED (pre-CDEF) rows as internal stripe-boundary
  context, after_cdef=1 saves CDEF rows at frame edges; per-stripe context-row
  swap with save/restore; the `optimized_lr` no-CDEF arm (no saves, ±3rd row
  duplicated from ±2nd); 150%-last-unit tiling, 8px stripe voffset, Wiener
  chunk widths rounded to 16 (dead over-writes, like C). Oracle: REAL exported
  save+filter over real bordered YV12 buffers (`shim_lr_filter_frame`), both
  arms, adversarially-unrelated deblocked-vs-current content — byte-identical.

`decode_frame_obus` mirrors decodeframe.c:5404-5482: deblock → snapshot →
cdef → restoration; rejection dropped. **GATE: 234 real cpu-used=0 KEY streams
byte-identical vs the C decoder** — 126 prior stay green + 108 restoration
arms (optimized-GOOD, boundary-swap-GOOD, ALLINTRA x both, bd8 cq52
deblock+CDEF+LR chains). Probed: 71 arms carry LR syntax, ALL 71 genuinely
change pixels, RU populations 44 wiener + 85 sgrproj. **54 arms are
`usage=2` (AOM_USAGE_ALL_INTRA)** — the zenavif/avifenc still-image mode, per
the ALLINTRA-primary directive (encoder header deltas verified from
av1_cx_iface.c: kf_max_dist=0 required, enable_cdef defaulted off for
allintra but explicit control overrides, enable_qm stays 0 without TUNE_IQ).
Envelope after this chunk: film grain, segmentation, palette/screen-content,
SB128, multi-tile, superres, qm, lossless, intrabc, 4:2:2-chroma-deblock
still rejected; restoration has NO residual constraint.

## 128x128 superblocks applied — SB128 rejection DROPPED (2026-07-14, decoder track)

**`use_128x128_superblock` is in the envelope** (d87168f driver, 122d68a
oracle, 798ec25 gate). The gap was ISOLATED to the decode driver: the entropy
layer (`aom-entropy` partition / `has_top_right` / `has_bottom_left` /
`read_cdef` / `lr_corners_in_sb`) was ALREADY fully generic over `sb_size`
(the `BLOCK_128X128` branches, `MAX_MIB_MASK=31`, `cdef_transmitted[4]`, the
`col+2*row` 4-way index — all pre-existing and byte-identical to
`reference/libaom`, cross-checked against the C source this chunk). Only
`aom-decode` hardcoded the 64x64 constants (`SB_MI=16`, `BLOCK_64X64` root,
`mib_size_log2=4`, the `16,16` corners-in-sb args, the OBU-level SB128
rejection).

- **Driver** (`aom-decode`): new `KfTileConfig::sb_size_128`. `TileKf::new`
  derives the per-tile `mib_size` (16/32) + root bsize
  (`BLOCK_64X64`/`BLOCK_128X128`) and threads them into
  `KfBlockState.{mib_size,sb_size,bsize}`, the aligned recon dims,
  `decode_tile_kf`'s SB row/col step + partition-tree root, the SB-root
  LR-corners gate (`bsize == self.st.sb_size`), `lr_corners_in_sb`'s
  `mi_size_wide/high` args, and both `intra_avail` sb_size args. `frame.rs`
  reads `mib_size_log2 = 5` from the sequence header when set (was hardcoded
  4) and populates `sb_size_128`; the OBU-level "128x128 superblocks"
  rejection is dropped. The `left_e/left_p/left_t [T;32]` context arrays
  needed no resize — already sized for the 128 case; the whole-array reset
  per SB row makes the existing `& 31` indexing correct at either mib_size.
- **Oracle** (`aom-sys-ref`, APPEND-ONLY on the shared file):
  `shim_encode_av1_kf_sb128` + `ref_encode_av1_kf_sb128` — the real
  `aom_codec_av1_cx` at `AV1E_SET_SUPERBLOCK_SIZE=AOM_SUPERBLOCK_SIZE_128X128`.
  `shim_encode_av1_kf` refactored to a thin wrapper over a shared impl,
  byte-identical behavior; the decode oracle is unchanged (SB size is a
  stream fact the real decoder reads itself).

**GATE: 160 real cpu-used=0 `--sb-size=128` KEY streams byte-identical vs the
C decoder** (`sb128_streams_decode_byte_identical_to_c`), a SEPARATE gate
alongside the 336-stream 64x64 main gate (which stays green — 64x64 streams
parse unchanged, the tile_roundtrip synthetic 624-tile roundtrip too). 4 sizes
(128x128 exact-SB, 192x160 + 256x224 multi-SB grids — 80 arms span >1 SB on an
axis — 100x76 sub-SB clipped root) x 5 (bd,ss,mono) combos x 8 arms. **140 are
`usage=2` (AOM_USAGE_ALL_INTRA)** — the zenavif/avifenc still-image primary
path — + 20 GOOD. cq {2,6,20,36} lands all four qindex bands [20,20,20,100];
141 code TX_MODE_SELECT. The SB128-specific geometry genuinely fires:
CDEF 40/40 gated+applied (the `cdef_transmitted[4]` 4-way per-64-unit index)
and LR 37/37 gated+applied (corners-in-sb at the 32-mi SB extent), every
gated stream verified to change pixels vs the no-filter recompose. Envelope
after this chunk: film grain, palette/screen-content, multi-tile, superres,
qm, lossless, intrabc, 4:2:2-chroma-deblock still rejected; SB size (64/128)
has NO residual constraint.

## Coded-lossless applied — lossless rejection DROPPED (2026-07-14, decoder track)

**Coded-lossless (`--lossless=1`) is in the envelope** (ca3de30 WHT, a90b0e7
decoder+oracle+gate). `AV1E_SET_LOSSLESS=1` forces `base_qindex 0` +
`coded_lossless`, which flips every block's transform to forced `TX_4X4` + the
4x4 Walsh–Hadamard (WHT), narrows `is_cfl_allowed` to `BLOCK_4X4`, and gates the
header's loop-filter / CDEF / restoration / tx-mode reads off.

- **WHT** (`aom-transform`): `av1_highbd_iwht4x4_add`
  (`av1/common/av1_inv_txfm2d.c` + the `idct.c` `xd->lossless` routing), the 4x4
  reversible Walsh–Hadamard, dispatched on `eob` (>1 full 16-point / else the
  DC-only special case, which idct.c flags as significant for lossless, not an
  optimization). One bd-parameterized highbd kernel (this build has only the
  highbd WHT; u16 planes). Diffed vs the exported `_c` kernels
  (`inv_txfm2d_diff.rs::highbd_iwht4x4_add_matches_c`, ~72k cases: bd 8/10/12 x
  both eob arms x tight+strided dest, full dequant clamp range).
- **Header parse** (`aom-decode` `frame.rs`): a TWO-PHASE parse.
  `read_uncompressed_header` gates LF/CDEF/restoration/tx-mode on
  `cfg.coded_lossless` — a writer-mirror INPUT the minimal-header anchor tests
  rely on — but the decoder derives lossless status from the parsed quant +
  segmentation like `decodeframe.c` does mid-header. Those parse BEFORE the
  gated sections, so a probe pass yields exact quant/seg; recompute
  `coded_lossless` (`frame_coded_lossless`, matching C's `is_coded_lossless`) and
  re-parse with correct gating when lossless. The two blanket lossless rejects
  become ONE narrowed reject for a genuinely MIXED frame (some-but-not-all
  segments lossless) — never emitted by the real encoder, not differentially
  testable, and it would need per-segment `cfl_allowed`.
- **Driver** (`aom-decode` `lib.rs`): `TileKf::new` computes `coded_lossless` ->
  `st.coded_lossless` (also suppresses the header CDEF-strength read, already
  threaded). Per block: `tx_size` preempts to `TX_4X4` + the `TX_4X4`
  `set_txfm_ctxs` stamp; chroma `uv_tx` forced `TX_4X4`; `cfl_allowed` threads
  lossless; `reconstruct_txb_wht` (`dequant_txb` + WHT) replaces
  `reconstruct_txb` for luma + chroma. The tx-type gate already emits `DCT_DCT`
  at qindex 0 and `self.dequants` is already the qindex-0 row, so neither
  changed.
- **Oracle** (`aom-sys-ref`, APPEND-ONLY): `shim_encode_av1_kf_lossless` /
  `ref_encode_av1_kf_lossless` (real `aom_codec_av1_cx` at `AV1E_SET_LOSSLESS=1`,
  threaded through the existing shared encode impl; existing wrappers pass
  `lossless=0`, a no-op) + `shim_highbd_iwht4x4_add` (the WHT unit oracle).

**GATE: `lossless_streams_decode_byte_identical_to_c`, 8 arms** (8/10-bit x
4:2:0 / 4:4:4 / monochrome x SB-multiple 64x64 + non-SB 96x80 + non-8-multiple
100x76 crops x GOOD + ALL_INTRA). Each arm asserts BOTH byte-identity vs the
REAL C decoder AND — since the stream is truly lossless — equality to the
ORIGINAL source pixels. Anti-vacuous: every block is `TX_4X4` and the run
exercises **3547 luma + 2918 chroma WHT txbs**. Envelope after this chunk:
film grain, quant matrices, superres, `disable_cdf_update`, 4:2:2-chroma-deblock
(nonzero chroma levels), mixed-lossless-segments, and multi-tile-GROUP (>1
`OBU_TILE_GROUP`) still rejected; whole-frame lossless has NO residual
constraint.

## Quantization matrices applied — `using_qmatrix` rejection DROPPED (2026-07-14, decoder track)

**QM streams (`--enable-qm=1`) are in the envelope** (d09b3f5 decode path + table,
8502e13 oracle + gate). With `using_qmatrix`, each 2-D-transform coefficient is
dequantized with a per-position inverse-QM weight
(`dqcoeff = (qcoeff*dequant*iqmatrix[pos] + 16) >> 5`) instead of the flat step —
the decode side of `av1/common/quant_common.c`'s `av1_get_iqmatrix` /
`av1_qm_init`. `dequant_txb` (`aom-txb`) already folds the weight (`get_dqv`), so
the only gap was selecting the right matrix per txb; the reject in `frame.rs` is
dropped.

- **Table** (`aom-decode` `qm_tables.rs`, generated by `xtask/extract_qm_tables.py`):
  the `iwt_matrix_ref[NUM_QM_LEVELS-1][2][QM_TOTAL_SIZE]` = `[15][2][3344]` inverse-QM
  bases, transcribed VERBATIM from libaom `quant_common.c` (100,320 `u8`,
  count-verified). Only the INVERSE (decode) matrix is ported; the forward
  `wt_matrix_ref` is encoder-only. The tables are NOT in any shared Rust crate
  (`aom-quant`'s QM quantizers take the `iqm` bytes as a parameter — they don't
  hold the bases), so this is a decoder-side generation, not a reuse.
- **Selection** (`aom-decode` `qm.rs`): `qm::iqmatrix(qm_level, plane, tx_size,
  tx_type)` reproduces `av1_get_iqmatrix` — `None` (flat dequant, libaom's NULL
  `giqmatrix[15]` rows) for the flat top level (QM off / lossless segment) and for
  1-D / identity transforms (`tx_type >= IDTX`, `!is_2d_transform`), else the
  `iwt` slice for `(level, plane>=1 ? chroma : luma, adjusted tx)` in raster order.
  `QM_OFFSET` reproduces `av1_qm_init`'s packing (accumulates to exactly 3344;
  64-point sizes alias their 32-capped matrix, `av1_get_adjusted_tx_size`).
- **Driver** (`aom-decode` `lib.rs`): `KfTileConfig` gains `using_qmatrix` +
  `qm_y/u/v`; `TileKf` tracks a per-block `block_qm_level` recomputed by
  `frame_qm_levels` (`setup_segmentation_dequant`'s `qmlevel_{y,u,v}` =
  `av1_use_qmatrix ? qmatrix_level_* : 15`, where `use_qmatrix = using_qmatrix &&
  !lossless[seg]` on the FRAME per-segment qindex). The three `reconstruct_txb`
  sites (luma uniform, intrabc var-tx, chroma) thread the selected matrix. Non-QM
  frames keep `block_qm_level == [15;3]` -> `iqmatrix` `None` -> the flat dequant,
  byte-identical to the pre-QM path.
- **Oracle** (`aom-sys-ref`, APPEND-ONLY): `shim_encode_av1_kf_qm` /
  `ref_encode_av1_kf_qm` (real `aom_codec_av1_cx` at `AV1E_SET_ENABLE_QM=1` +
  `AV1E_SET_QM_MIN==QM_MAX==L`, forcing `qmatrix_level_{y,u,v}==L`). Threaded
  through the shared encode impl gated on `enable_qm` (existing wrappers pass
  `enable_qm=0`, a no-op — bytes unchanged).

**GATE: `qm_streams_decode_byte_identical_to_c`, 15 arms** (8/10-bit x 4:2:0 /
4:4:4 / monochrome x 64x64 + 96x80 + non-8-multiple 100x76 crop x GOOD +
ALL_INTRA x forced levels 0/5/8/12). Each asserts plane-by-plane byte-identity vs
the REAL C decoder. Anti-vacuous: `using_qmatrix` set, decoded qm level non-flat
(`0..=14`, verified 0..=12 with 8 steep `L<=5` arms), non-lossless, luma
non-constant (AC coefficients present); byte-identity on a non-flat level would
FAIL if the decoder ignored QM. A `qm::tests` unit suite additionally anchors the
table to the C bytes and proves the dequant differs from flat at every level.
**Honest gaps**: QM composed with CDEF/LR/intrabc/palette/segmentation is not a
gated combination (the gate uses `--enable-cdef=0 --enable-restoration=0`, no
screen-content); QM is orthogonal to those post-reconstruction stages, but the
combination isn't differentially pinned. Envelope after this chunk: film grain,
superres, `disable_cdf_update`, 4:2:2-chroma-deblock (nonzero chroma levels),
mixed-lossless-segments, and multi-tile-GROUP still rejected.

## Film grain synthesis applied — `film_grain_params_present` rejection DROPPED (2026-07-14, decoder track)

**Film-grain KEY streams are in the envelope** (9db05fc Y-only synthesis + oracle,
f3f914c chroma + chroma-from-luma gate, 2bbac15 end-to-end decode wiring). Film
grain is applied at the DECODER as the final post-reconstruction OUTPUT stage
(`av1_add_film_grain`, `av1/decoder/grain_synthesis.c`): a seeded 16-bit LFSR
builds grain templates from the fixed gaussian sequence, a piecewise-linear
scaling LUT maps luma -> grain scale, and the scaled grain is blended into the
decoded planes with 2-tap subblock overlap and optional chroma-from-luma.

- **Synthesis** (`aom-decode` `film_grain.rs` + `film_grain_gaussian.rs`): a
  faithful pure-Rust port of `grain_synthesis.c` (`av1_add_film_grain` /
  `add_film_grain_run`): `get_random_number` LFSR + `init_random_generator`,
  `generate_luma_grain_block` / `generate_chroma_grain_blocks` (AR grain with the
  cfl luma-avg position), `init_scaling_function` / `scale_LUT` (10/12-bit
  interpolation), `ver_/hor_boundary_overlap`, `extend_even`, and a UNIFIED
  `add_noise_to_block` — the C lowbd (uint8_t) and hbd (uint16_t) noise-add paths
  are numerically identical at `bit_depth==8` (offset/clip/scale-LUT formulas all
  reduce to the lowbd constants when `bit_depth-8==0`), so one bit_depth-
  parameterized kernel matches C at 8/10/12-bit. Planes are `u16`; arithmetic is
  `i32` (matching C `int`). `#![forbid(unsafe_code)]` intact.
- **Parse**: `read_film_grain_params` (already in `aom-entropy`) is now reachable
  — `parse_frame_header` populates the reader's monochrome/subsampling context so
  the chroma-absent gate parses mono/4:2:0/4:2:2/4:4:4 correctly.
- **Wiring** (`frame.rs`): the seq-level reject is gone; after `finish_frame`,
  when `film_grain.apply_grain`, `add_film_grain` runs on the cropped display
  planes (mc_identity from `matrix_coefficients`, now threaded through
  `KfTileConfig`), exactly as `aom_codec_get_frame` does.
- **Oracle** (`aom-sys-ref`, APPEND-ONLY): (1) `shim_add_film_grain` /
  `ref_add_film_grain` — the REAL exported `av1_add_film_grain` over two
  constructed `aom_image_t`s (function-level synthesis oracle, NOT a
  transcription); (2) `shim_encode_av1_kf_film_grain` /
  `ref_encode_av1_kf_film_grain` — real `aom_codec_av1_cx` with
  `AV1E_SET_FILM_GRAIN_TEST_VECTOR` (libaom's built-in `grain_test_vectors[]`),
  producing streams that carry `film_grain_params_present=1` + per-frame params.

**GATES (`film_grain_diff.rs`):**
- Function-level synthesis, **1440 trials byte-identical** to `av1_add_film_grain`:
  `film_grain_y_only_matches_c` (576), `film_grain_chroma_matches_c` (432,
  independent cb/cr points + mult/offset), `film_grain_cfl_matches_c` (432,
  chroma_scaling_from_luma) — 8/10/12-bit x mono/4:2:0/4:4:4/4:2:2 x 4 sizes
  (incl. odd dims for `extend_even`) x overlap/clip on/off, 100% altering pixels.
- End-to-end, **30 REAL streams byte-identical** to the C decoder (grain applied):
  `film_grain_streams_decode_byte_identical_to_c` — grain vectors {1,2,15} x
  8/10-bit x 4:2:0/4:4:4/mono x 64x64 & 96x80. Anti-vacuous: an ungrained
  reconstruction of each stream differs from the grained output (proves the C
  decoder applied grain, not the skip flag); chroma_grain=24, cfl=8.

**Honest gaps**: single KEY frame only (inter grain-param `update`/ref-load path
is unreachable here — the reader supports it but no inter frame enters the
envelope); grain composed with CDEF/LR/segmentation/palette is orthogonal
(post-reconstruction) but not a separately gated combination. Envelope after this
chunk: superres, `disable_cdf_update`, 4:2:2-chroma-deblock (nonzero chroma
levels), mixed-lossless-segments, and multi-tile-GROUP still rejected.

## Superres upscaling applied — `superres scaled` rejection DROPPED (2026-07-15, decoder track)

**Normative superres is in** (3380a91): superres-scaled KEY streams
(`SuperresDenom` in [9,16]) now decode in the envelope. The frame is coded at a
reduced (downscaled) `FrameWidth = (UpscaledWidth*8 + Denom/2)/Denom` and
upscaled back to the full `UpscaledWidth` **horizontally only**, as a normative
post-CDEF stage (`av1_upscale_normative_rows`), spliced between CDEF and loop
restoration.

- `aom-decode::superres` ports the 8-tap 64-phase polyphase horizontal upscale
  (`av1_convolve_horiz_rs` + `upscale_normative_rect` + `av1_get_upscale_convolve_step`
  / `get_upscale_convolve_x0` + the `av1_resize_filter_normative` table). One
  `u16` implementation covers 8/10/12-bit (`clip_pixel` / `clip_pixel_highbd` are
  the same clamp on the shared store; the sum/round math is bit-depth
  independent). Frame edges use an index clamp to the mi-aligned downscaled width
  — byte-identical to libaom's save/`memset`/restore border extension for the
  single tile column (both edges are frame edges).
- Integration (`frame.rs`): the tile decode / deblock / CDEF run in the CODED
  (downscaled) domain (mi grid sized to `FrameWidth`), so the tile-info syntax is
  re-parsed over the downscaled MiCols (a two-phase parse, like coded-lossless);
  the deblock crop is the coded width (was the upscaled width — that filtered the
  mi-overhang columns the upscale then samples, corrupting the right edge);
  `optimized_lr = !do_cdef && !do_superres` (superres always takes the
  non-optimized LR path); the pre-CDEF deblock snapshot that feeds LR's internal
  stripe boundaries is upscaled the same way (matching
  `save_deblock_boundary_lines`, which runs the normative upscale on those rows),
  so loop restoration runs entirely in the upscaled domain.
- `aom-entropy::lr::lr_corners_in_sb` gained the superres arm of
  `av1_loop_restoration_corners_in_sb`: the downscaled mi position maps into the
  upscaled RU grid via `mi_to_num_x = mi_size_x*Denom`, `denom_x = size*8`.
- Oracle (append-only): `dec_shim.c` `shim_encode_av1_kf_superres`
  (`rc_superres_mode = AOM_SUPERRES_FIXED` + `rc_superres_kf_denominator = D`) +
  `ref_encode_av1_kf_superres` binding — the REAL `aom_codec_av1_cx` encoder.

**GATES (`superres_diff.rs`, byte-identical vs the C decoder `aom_codec_av1_dx`):**
- `superres_luma_mono_byte_identical_to_c` — **24 streams**: 4 sizes (incl.
  non-8-multiple 100x76) x 8/10-bit x denoms {9,12,16}, monochrome.
- `superres_color_byte_identical_to_c` — **45 streams**: 3 sizes x
  {4:2:0/4:4:4 8-bit, 4:2:0/4:4:4 10-bit, 4:2:0 12-bit} x denoms {9,12,16},
  rotating GOOD/ALL_INTRA + CDEF on/off (chroma upscaled at its subsampled
  width).
- `superres_lr_composed_byte_identical_to_c` — **24 streams**: 2 sizes x
  {4:2:0/4:4:4/mono, 8/10-bit} x denoms {9,12,16}, `--enable-restoration=1` with
  restoration genuinely active (asserted floor) — the superres+LR composition.
- **93 real cpu-used=0 KEY streams byte-identical.** Anti-vacuous per stream:
  `SuperresDenom > 8`, coded `FrameWidth < UpscaledWidth` (steepest ~0.5x at
  D=16), non-flat AC content. No regression: the full decoder suite is 0 failed.

**Honest gaps**: single-tile-column only — multi-tile superres is rejected (needs
`av1_upscale_normative_rows`' per-tile-column convolve loop; the AVIF-still / KEY
superres path is single-tile). Superres is horizontal-only, so the vertical axis
is untouched by construction. Envelope after this chunk:
4:2:2-chroma-deblock (nonzero chroma levels), mixed-lossless-segments,
multi-tile-GROUP, and multi-tile superres still rejected
(`disable_cdf_update` landed next — see below).

## `disable_cdf_update` decode — rejection DROPPED (2026-07-15, decoder track)

**Non-adapting symbol decode is in** (1dfbcc3): KEY streams whose uncompressed
header carries `disable_cdf_update = 1` now decode in the envelope. When the flag
is set the tile symbol reader does NOT adapt CDFs — every `read_symbol` leaves its
CDF at the loaded/initial value for the whole tile, instead of applying the
post-decode `update_cdf` step.

- **Exact C mechanism reproduced**: `aom_dsp/bitreader.h`'s `aom_read_symbol_`
  adapts only `if (r->allow_update_cdf) update_cdf(cdf, ret, nsymbs)`. The decoder
  derives `allow_update_cdf = (!tiles.large_scale) && !features.disable_cdf_update`
  (`av1/decoder/decodeframe.c:2893`) and stores it on the reader
  (`r->allow_update_cdf = allow_update_cdf`, :1470). A single-tile KEY frame is
  never `large_scale`, so the reader adapts iff `!disable_cdf_update`.
- **Port** (exact mirror): `OdEcDec` gained `allow_update_cdf` (default `true`);
  `aom_entropy::read_symbol` gates its `update_cdf` on it. `read_symbol` is the
  SOLE `update_cdf` site — both the mode-info reads (partition/intra/mv/... in
  `aom_entropy::partition`) AND the `aom_txb` coefficient reads (`rsym`'s
  `upd=true` branch delegates to `read_symbol`) flow through it, so one field +
  one branch covers every adapting read. `KfTileConfig.disable_cdf_update` threads
  the parsed header flag; `decode_frame_tiles_kf` sets
  `dec.allow_update_cdf = !disable_cdf_update` once per tile reader. The flag-off
  path is byte-identical plus one predictable, always-taken branch.
- `disable_frame_end_update_cdf` (which `error_resilient_mode` also forces on) is
  decode-irrelevant here: it only governs whether this frame's END adapted CDFs
  are SAVED as a frame context for LATER frames that reference it
  (`REFRESH_FRAME_CONTEXT_BACKWARD`); this driver decodes exactly one shown KEY
  frame with no forward reference chain, so the saved context is never read back.
- Oracle (append-only): `dec_shim.c` `shim_encode_av1_kf_disable_cdf` +
  `ref_encode_av1_kf_disable_cdf` — the REAL `aom_codec_av1_cx` encoder with
  `AV1E_SET_CDF_UPDATE_MODE = 0` (`--cdf-update-mode=0`), which forces
  `cm->features.disable_cdf_update = 1` for every frame (`encoder.c:4375`, the
  `case 0:` arm). NOTE: `error_resilient_mode` does NOT set the flag in libaom's
  encoder — traced to the `cdf_update_mode` switch; the control is the real path.

**GATE (`disable_cdf_update_diff.rs`, byte-identical vs the C decoder):**
- `disable_cdf_update_streams_decode_byte_identical_to_c` — **54 real cpu-used=0
  KEY streams**: bd {8,10,12} x {4:2:0, 4:4:4, monochrome} x 3 sizes (incl.
  non-8-multiple 96x80) x cq {20,48}, every plane byte-identical. Anti-vacuous per
  stream: the port's OWN parse asserts `disable_cdf_update = 1` (both
  `FrameDecode` and the parsed header/config), non-flat AC content present, and
  the same image re-encoded WITHOUT the flag yields DIFFERENT bytes.
- `disable_cdf_update_gate_is_load_bearing` — proves at the symbol layer that
  flipping `allow_update_cdf` changes both the adapted CDF STATE and the decoded
  symbol VALUES on an adapting-coded stream (flag-on reproduces + tracks the
  encoder's adapted CDF; flag-off freezes the CDF and mis-decodes). The gate is
  not a no-op.
- **No regression**: the full decoder suite (`aom-decode` + `aom-entropy`, all
  existing gates — real_bitstream / sb128 / multi-tile / film_grain / superres /
  qm / palette / intrabc / lossless / tile_roundtrip) stays **0 failed**.

**Honest gaps**: KEY frames only (the whole envelope is KEY-only); the flag is
threaded through the single-tile reader construction, so a hypothetical
large-scale-tile stream (which would force `allow_update_cdf=0` independently) is
moot — `large_scale` tiles are out of envelope regardless. Envelope after this
chunk: 4:2:2-chroma-deblock (nonzero chroma levels), mixed-lossless-segments,
multi-tile-GROUP, and multi-tile superres still rejected.

## Perf gate honest number

Like-for-like vs C's production AVX2 (`aom_sad64x64_avx2`): Rust AVX2 SAD is
**~2.2x** (direct kernel) / ~2.5x (with runtime dispatch) — gate is <=1.20x, so
NOT met. The gap is the kernel (libaom hand-tuned asm ~2x faster), not dispatch
(~0.15x). The earlier 1.42x figure was vs C *scalar* and was replaced.

## (0,0)-corner tx_search overflow — FIXED (2026-07-14, encoder track)

**The panic blocking a true-(0,0)-corner single-SB frame is fixed.**
`crates/aom-encode/src/tx_search.rs`'s `txfm_rd_in_plane_intra` (and the
identical chroma walk in `crates/aom-encode/src/intra_uv_rd.rs`'s
`txfm_rd_in_plane_uv`) computed `ref_best_rd - current_rd` as a checked `i64`
subtraction feeding `search_tx_type_intra`'s early-termination budget. At the
frame's TRUE (0,0) SB (no up/left neighbour, so the top-level intra mode
search's first candidate has no reference RD yet: `ref_best_rd ==
i64::MAX`), `current_rd` can go deeply negative from a genuine, independently-
verified C behaviour: `aom_dist::block_error`'s lowbd path wraps its
per-coefficient product at 32 bits (bit-exact to C's `int` arithmetic), and
a no-neighbour corner prediction can produce a residual whose forward-DCT
coefficient pushes that wrap into `dist`/`current_rd`. **Traced the real C**
(`block_rd_txfm`, `av1/encoder/tx_search.c:3104`): it computes
`args->best_rd - args->current_rd` as a RAW, UNGUARDED `int64_t` subtraction
— no `ref_best_rd == INT64_MAX` special case at this call site (contrast
`av1_txfm_search`'s `rd_thresh` derivation two hundred lines away, which
DOES special-case it: `ref_best_rd == INT64_MAX ? INT64_MAX : ref_best_rd -
mode_rd`). Real compiled aomenc (no UBSan/`-ftrapv` in the production build)
performs this as a plain two's-complement wraparound. **Fix**: both sites now
use `ref_best_rd.wrapping_sub(current_rd)` — NOT `saturating_sub` (which
would compute a value C never produces). The wrapped value is not dead: at
speed-0 allintra `adaptive_txb_search_level == 1`
(`TxTypeSearchPolicy::speed0_allintra`), so `search_tx_type_intra`'s
`(best_rd - (best_rd >> level)) > ref_best_rd` early-break reads it.

**Empirically confirmed, not just theoretical.** A ~150-combination sweep
(`allintra x chroma-ss(0,0)/(1,1) x qindex 0..256 step 7`, true (0,0) corner,
`crates/aom-encode/tests/pack_tile_roundtrip.rs`'s `run_pack_roundtrip_case`)
found the exact pre-fix panic at `allintra=true, ss=(1,1), qindex=98`:
`thread ... panicked at crates/aom-encode/src/tx_search.rs:1118:49: attempt
to subtract with overflow`. Reverting the fix reproduces it; restoring it
passes. That combination is now a permanent 4th case in
`pack_tile_roundtrips_true_corner`. A companion test,
`pack_tile_roundtrips_through_the_read_side`, still covers the padded/interior
scope unchanged (same 3 cases, `pad = SB_MI`, refactored into the shared
`run_pack_roundtrip_case(ss_x, ss_y, allintra, qindex, pad)` helper so both
share setup). `partition_pick_diff.rs` and `uniform_txfm_yrd_diff.rs` still
green (unaffected — neither's `ref_best_rd` ever wraps in the cases they
sweep). Commit: (this session).

### qindex=0 partition-population mismatch + "skip always 0" violation — FIXED (2026-07-14, encoder track)

**Both previously-tracked `qindex=0` bugs were ONE root cause, not two, and
both are now fixed.** Traced via the real C source (`is_coded_lossless`,
`av1_common_int.h:1862-1876`; `xd->lossless[i]` assignment,
`av1/encoder/encodeframe.c:2263-2266`): at `base_qindex == 0` with no
delta-q (this envelope's constant assumption — `PackCfg::base_qindex`'s
doc), real AV1 derives `xd->lossless[segment_id] = true` — **qindex=0 is
not "a small quantizer step inside the normal DCT pipeline", it's a
completely different coding mode**: `tx_mode` is forced `ONLY_4X4`
(`av1/decoder/decodeframe.c:141`'s `read_tx_mode`) and every tx-type
search is forced to `DCT_DCT` (`av1_get_tx_type`, `blockd.h:1288-1290`).

**Root cause (confirmed, not guessed): a write/read tx-type scan-order
desync, not two independent bugs.** `crates/aom-encode/tests/
pack_tile_roundtrip.rs`'s `run_pack_roundtrip_case` hardcoded
`SbEncodeEnv::lossless: false` unconditionally, so at qindex=0 the search's
tx-type search stayed free to pick a non-DCT_DCT winner — this port's own
`get_tx_mask_intra`/`get_tx_mask_uv_intra` (`tx_search.rs`) already force
DCT_DCT when `lossless`, they just never SAW `lossless = true`. Separately,
`pack.rs`'s `signal_gate: qindex > 0` correctly suppresses WRITING the
tx_type symbol at qindex=0 (matches `av1_write_tx_type`'s `base_qindex > 0`
gate, `bitstream.c:815-819`) — but `aom_txb::write_coeffs_txb_full` still
SCANS/serializes the coefficients using the search's real (possibly
non-DCT_DCT) `tx_type` regardless of `signal_gate` (matches C:
`av1_write_tx_type` only gates the SYMBOL — the encoder's own
`mbmi->tx_type` still has to be right independently; real aomenc never
gets this wrong because ITS tx-type search is itself lossless-gated).
Meanwhile `aom_txb::read_tx_type`, since nothing was signaled, defaults to
DCT_DCT unconditionally (matches `av1_read_tx_type`'s `*tx_type = DCT_DCT;
... if (qindex == 0) return;`). Whenever the search's real winner wasn't
DCT_DCT, write and read disagreed on scan order for that txb — garbage
coefficients — the shared entropy-coder state desynced for the rest of the
frame. That ONE desync is what surfaced as BOTH the partition-population
mismatch (read_partition returning garbage after the desync point) AND the
spurious `skip == 1` read (same garbage, different symbol) — confirmed by
fixing the one root cause and seeing both symptoms disappear together in
the same run.

**Fix** (both sites are self-consistency bugs, not spec deviations —
`aom-encode`'s lossless-forcing logic was already correct everywhere else
it's threaded): (1) `pack_tile_roundtrip.rs`'s `run_pack_roundtrip_case`
now sets `lossless: qindex == 0` instead of hardcoding `false`, matching
`encodeframe.c`'s derivation exactly (no delta-q in this envelope, so the
five-way AND reduces to the qindex check alone). (2)
`crates/aom-encode/src/partition_pick.rs`'s `TxfmYrdEnv` construction
hardcoded `tx_mode_is_select: true` regardless of `env.lossless` — harmless
while every caller also hardcoded `lossless: false`, but once qindex=0
correctly threads `lossless: true` this violated
`pick_uniform_tx_size_type_yrd_intra`'s own "lossless implies ONLY_4X4"
`debug_assert` (real AV1's `select_tx_mode`/`read_tx_mode`: `coded_lossless`
forces `ONLY_4X4`, never `TX_MODE_SELECT`). Now `tx_mode_is_select:
!env.lossless`. This source fix is a no-op for every other caller (all
still hardcode `lossless: false`, so `!false == true`, unchanged) —
confirmed by the full suite staying green with only the one new test added.

**Verified.** New permanent regression test
`pack_tile_roundtrips_qindex_zero` (`pack_tile_roundtrip.rs`) covers all 4
qindex=0 combinations, including both original bug repros verbatim
(`run_pack_roundtrip_case(0, 0, true, 0, 0)` and `(1, 1, false, 0, 0)`) —
confirmed to reproduce the EXACT documented pre-fix failure (`left: (34,
22, 8, 5, 1) right: (42, 2, 6, 10, 10)`) before the fix, and to pass
(partition population match AND `skip == 0` holds for every leaf) after.
Full `cargo test -p aom-encode --all-targets`: 71/71 passing before this
chunk (0 failed) → 72/72 after (0 failed, +1 new permanent test); the 7/7
`encoder_gate_e2e_textured_attempt` gate (qindex=128, unaffected) stays
green. `cargo clippy -p aom-encode --all-targets --no-deps`: clean.
Commit: (this session).

## Real (CDF-derived) cost tables wired into the search — PARTIAL (2026-07-14, encoder track)

**Fraction: 6 of 7 real-AV1 cost-table families wired; the 7th (coefficient
costs) is wired at reduced fidelity (1 representative `txs_ctx` instead of
the real 5).** Task: "feed the search the SAME cost tables aomenc uses"
(`av1_fill_mode_rates` + `av1_fill_coeff_costs`, rd.c) so `rd_pick_partition_
real`/the leaf search's decisions are cost-driven the way real aomenc's are,
not the synthetic-but-valid random tables `pack_tile_roundtrip.rs` uses to
verify pack glue only.

**Present** (`crates/aom-encode/src/real_costs.rs`, new module,
`derive_real_costs(kf: &KfFrameContext, enable_filter_intra: bool) ->
RealCosts`): mode costs (Y/UV/filter-intra flag+mode/angle-delta/intrabc/
palette-Y-flag, via the ALREADY-PORTED `fill_intra_mode_costs` +
`fill_palette_uv_mode_costs`), CfL costs (`fill_cfl_costs`), tx-size costs
(`fill_tx_size_costs`), intra tx-type costs (`aom_txb::fill_tx_type_costs`,
fed by a new `repack_intra_ext_tx_cdf` that widens `KfFrameContext::
ext_tx_1ddct`/`ext_tx_dtt4`'s narrow per-eset storage into the uniform
17-wide stride `fill_tx_type_costs` expects), and two NEW small fills this
chunk added to close real gaps in `av1_fill_mode_rates`'s coverage:
`fill_partition_costs` + `fill_skip_costs` (`crates/aom-encode/src/mode_costs.rs`).
All derive from `KfFrameContext::default_for_qindex`'s CDFs -- the REAL
libaom default probability tables (already C-verified; `pack_tile_
roundtrip.rs`'s own doc comment already relied on this for the entropy
coder), matching what `av1_fill_mode_rates(cm, &x->mode_costs, cm->fc)`
computes from a frame's freshly-inited context (real aomenc's ACTUAL
starting cost snapshot -- RD costs are derived once per frame, not
re-derived per adapted symbol, so this is not a simplification vs. the C).

**Reduced-fidelity** (documented in `real_costs.rs`'s module doc + the new
test's doc comment): coefficient-coding costs. Real AV1 has 5 (`txs_ctx`,
tx-size category) x 2 (plane) = 10 distinct `LV_MAP_COEFF_COST` tables
(`av1_fill_coeff_costs`, ALSO already ported: `aom_txb::fill_lv_map_coeff_
cost`, plus a NEW `fill_lv_map_coeff_cost_from_arena` this chunk added that
slices them directly from a live CDF arena using `write.rs`'s own `A_*`
region offsets) -- but `SbEncodeEnv::coeff_costs_y`/`coeff_costs_uv` (and
every downstream `TxTypeSearchInputs`/`TxfmYrdEnv`/`UvRdEnv::coeff_costs`
field, ~6 struct fields across `tx_search.rs`/`intra_uv_rd.rs`) is a SINGLE
`&CoeffCostTables` reference used for EVERY tx size -- an architectural
mismatch with real AV1's per-size costs that predates this chunk (not
introduced by it). This chunk's test (`pack_tile_roundtrips_with_real_costs`)
uses ONE representative `txs_ctx` (2, mid-size) for all luma tx sizes and
one for chroma -- REAL CDF-derived data, genuinely closer to aomenc than
random, but not size-correct. ALSO not yet derived: `eob_multi*_cost`
(`CoeffCostTables::eob`, the EOB-position cost `av1_fill_coeff_costs` ALSO
fills) -- zeroed in the new test, a second, smaller gap on the same table.

**Verified working end-to-end**: `pack_tile_roundtrips_with_real_costs`
(`crates/aom-encode/tests/pack_tile_roundtrip.rs`) drives the FULL search+
pack+read-back pipeline (not just unit-testing the fill functions) with
`derive_real_costs`'s tables at 3 (ss, allintra, qindex) combinations,
INCLUDING at the true (0,0) corner (a bonus re-verification of the
`wrapping_sub` fix above under a real, not hand-picked-extreme, cost
landscape) -- partition-type population + all 4 CDF arenas (coeff/
partition/kf_y/tx_size) agree between write and read, proving the wiring
doesn't desync the bitstream. Deliberately a NEW, separate test (not a
parameter on `run_pack_roundtrip_case`) so this exploratory wiring can never
perturb the already-verified synthetic-cost tests.

**NEXT chunk (the real remaining work)**: thread `txs_ctx` through the
coeff-cost path so `rd_pick_partition_real` sees the TRUE per-tx-size
`LV_MAP_COEFF_COST`, not one representative table. Concretely: change
`SbEncodeEnv::coeff_costs_y`/`coeff_costs_uv` from `&CoeffCostTables` to
`&[CoeffCostTables; 5]` (or a small lookup wrapper), update `encode_sb.rs`
(2 call sites), `partition_pick.rs` (3), `tx_search.rs`'s `TxfmYrdEnv`/
`TxTypeSearchInputs` (2 struct fields + their construction sites) and
`intra_uv_rd.rs`'s `UvRdEnv`/`TxTypeSearchInputs` use (1 field), selecting
by `aom_txb::txsize_entropy_ctx(tx_size)` at each txb. Also derive
`eob_multi*_cost` (needs tracing `av1_fill_coeff_costs`'s EOB-cost slice,
not yet examined this chunk). Then re-run
`pack_tile_roundtrips_with_real_costs` (rename once real) plus every
existing differential test in `aom-encode`/`aom-txb` to confirm the
signature change is behavior-preserving where inputs are unchanged.

## Frame-header OBU byte-matches real aomenc — ASSEMBLY-verified (2026-07-14, encoder track)

**The frame-header OBU now byte-matches real aomenc, for the minimal
single-SB single-tile no-post-filter envelope.** New:
`crates/aom-encode/tests/frame_header_matches_real_encoder.rs`
(`frame_header_matches_real_aomenc_output`), the same shape as the
already-landed `seq_header_matches_real_encoder.rs`: encode a real KEY
frame via `ref_encode_av1_kf` (`enable_cdef=false, enable_restoration=
false` -- the task's "no-post-filter envelope"), walk the OBU stream for
the sequence-header OBU + the frame OBU (`OBU_FRAME`=6, real aomenc emits
the combined header+tile-data form, not a standalone `OBU_FRAME_HEADER`),
build the `FrameHeaderObu` `cfg` template from the parsed seq header
(mirrors `aom-decode/src/frame.rs::parse_frame_header`'s own `cfg`
construction -- that function plus its `tile_limits`/`mi_dim` helpers are
private to the decoder-owned `aom-decode` crate, so this is a TRANSCRIPTION
of ~60 lines, not a call), parse the real frame-header bits with the
ALREADY-VALIDATED `read_uncompressed_header` (aom-entropy, decoder-owned;
gated on 336+ byte-identical real streams), then re-serialize with
`write_frame_header_obu` and assert BIT-exact match (via
`ReadBitBuffer::bit_position()` — the frame header does not end on a byte
boundary; the trailing partial byte's low bits belong to the tile-group
header that follows in the same `OBU_FRAME` payload, so the comparison
masks to only the bits the frame header actually owns).

**Passed on the first run**, 4 cases (`w×h`, mono, ALLINTRA usage=2 / GOOD
usage=0, single- and 2-SB): `frame_obu_type=6`, `bit_len` 35-38 bits (~5-6
bytes, non-trivial), `qindex` 128/160 (non-degenerate), `lf_level=[0,0]`
(the flat-gray-128 test content genuinely needs no deblocking — not forced,
observed), `cdef_on=false`, `restoration_on=false`, `tile_cols=tile_rows=1`
— confirmed via diagnostic printout, not a vacuous pass. A seq-header
sanity re-check (independent of `seq_header_matches_real_encoder.rs`) is
inlined too.

**Honest fraction — this is option (a) from the task brief, ASSEMBLY-
verified, NOT derivation-verified.** The parsed `FrameHeaderObu` holds real
aomenc's OWN chosen field values (quant, loop-filter level/deltas, tile
info, ...); the test proves `write_frame_header_obu`'s ordering/gating
serializes those values back byte-for-byte given the real values, NOT that
this port can DERIVE them from scratch via RDO. Specifically NOT proven:
loop-filter-LEVEL search (only ever observed at the trivial 0 value here —
a nonzero-LF case is untested), CDEF-strength search (sidestepped by
`enable_cdef=false`), and the coefficient-cost-driven mode/partition search
achieving the SAME decisions real aomenc's RDO makes (Task 2's coeff-cost
gap above is exactly this). Two OBUs now byte-match real aomenc
(sequence-header: fully; frame-header: assembly-verified) out of the three
this envelope needs (sequence-header + frame-header + tile-group).

**NEXT chunk — tile-group OBU assembly, the last piece for a genuine
minimal end-to-end byte match**: `write_tile_group_header` (aom-entropy,
decoder-owned, already used by `aom-decode`'s reader side per
`read_tile_group_header`) + `pack_tile`'s per-SB payload (already produces
real coefficient bytes, `crates/aom-encode/src/pack.rs`) + the
leb128/OBU-header wrapping (`aom-entropy::leb128`/`obu`, both already
bit-exact) assembled into one `OBU_FRAME` payload (frame-header bits +
`byte_align()` + tile-group header + tile data, matching
`aom-decode/src/frame.rs:477-483`'s read-side shape) and diffed against
`frame_payload[full_bytes..]` from THIS chunk's test (the frame-header's
end is already the tile-group's start, so extending this exact test is the
natural next step). For a genuinely single-tile frame the tile-group header
is close to a no-op (`num_tiles==1`), so this is smaller than the
frame-header piece was. Once assembled, the TRUE end-to-end byte match
still needs Task 2's coeff-cost-per-txs_ctx gap closed (else the coded
coefficient bits, though validly-formed, won't match real aomenc's RDO
choices) — so "bytes assemble correctly" and "decisions match aomenc" are
two separate, both-still-open claims even after the tile-group piece lands.

## Coefficient-cost decision parity — COMPLETE (2026-07-14, encoder track)

**The reduced-fidelity gap the previous chunk left open is closed.** The search
now reads REAL per-`(txs_ctx, eob_multi_size)` AV1 coefficient-coding cost tables
at every txb — matching `av1_fill_coeff_costs` exactly (5 tx-size categories x
2 plane types for `LV_MAP_COEFF_COST`, a SEPARATE 7-way `eob_multi_size` x 2
plane axis for `LV_MAP_EOB_COST` — the two axes do NOT collapse to one: e.g.
`TX_8X8`/`TX_4X8`/`TX_16X4` share `txs_ctx==1` but split across `eob_multi_size`
1/2) — instead of one representative `txs_ctx` table shared across every tx size
with zeroed eob-position costs.

**aom-txb** (new, additive): `CoeffCostSet` (`by_txs_ctx: [LvMapCoeffCost; 5]` +
`eob_by_multi_size: [[i32; 22]; 7]`) + `CoeffCostSet::tables(tx_size)` (the same
two independent lookups `av1_cost_coeffs_txb`/`av1_write_coeffs_txb` perform) +
`fill_eob_cost_from_arena` (derives `LV_MAP_EOB_COST` from the live
coefficient-CDF arena's `eob_flag_cdf{16..1024}` regions — previously zeroed) +
`fill_coeff_cost_set_from_arena` (the constructor). `LvMapCoeffCost` now derives
`Clone`.

**aom-encode** (architecture fix — the search now selects the cost table
PER-CANDIDATE-tx_size, not once per frame): `real_costs::RealCosts` gains
`coeff_costs_y`/`coeff_costs_uv: CoeffCostSet` (the actual `av1_fill_coeff_costs`
equivalent). `SbEncodeEnv::coeff_costs_y`/`coeff_costs_uv`: `&CoeffCostTables` ->
`&CoeffCostSet` (frame-level — spans every tx_size the recursive partition search
visits; `encode_b_intra_dry` selects `.tables(winner.tx_size)`/`.tables(uv_tx)`
once per leaf, a fixed size at that point). `tx_search::TxfmYrdEnv::coeff_costs`:
same type change — the luma tx-size DEPTH SEARCH
(`choose_tx_size_type_from_rd_intra`) tries multiple tx_size candidates against
the SAME env, so the `.tables(tx_size)` lookup moved inside
`txfm_rd_in_plane_intra`, keyed by the per-candidate parameter — this is the
actual bias the previous chunk's single-table simplification introduced.
`intra_uv_rd::UvRdEnv::coeff_costs` deliberately STAYS `&CoeffCostTables`:
chroma has no tx-size depth search (`av1_get_tx_size_uv` is a pure function of
bsize/lossless/subsampling), so the caller (`partition_pick::leaf_pick_sb_modes`)
pre-selects the ONE table this env's whole lifetime uses. `rd_pick::
rd_pick_intra_mode_sb`'s `coeff_costs_y` param: `&CoeffCostTables` ->
`&CoeffCostSet` (the luma-winner CfL re-encode needs the table for `y.tx_size`,
only known AFTER the sby search picks a winner inside this function).

Verified: full aom-encode test suite green (67 tests, 0 failed — including
`pack_tile_roundtrips_with_real_costs`, now FULLY real with the
`REPRESENTATIVE_TXS_CTX` simplification and zeroed eob costs both dropped, and
`pack_tile_roundtrips_true_corner`, the (0,0)-corner `wrapping_sub` regression,
re-confirmed under the now-fully-real cost landscape) + aom-txb's own 24-test
suite green.

## Tile-group OBU assembly — COMPLETE and REAL-encoder-verified (2026-07-14, encoder track)

**The last framing piece for a genuine minimal end-to-end byte match.** New
`aom_encode::obu_assemble` module composing already-bit-exact pieces (all
aom-entropy, decoder-owned but CALLED not modified: `write_frame_header_obu`,
`write_tile_group_header`, `write_obu_header`, `leb128::uleb_encode`) into one
real `OBU_FRAME` byte sequence, per the AV1 spec's `frame_obu(sz)`:
`frame_header_obu(); byte_alignment(); tile_group_obu(sz)` — and
`tile_group_obu`'s own header bits + its OWN `byte_alignment()` before the tile
payload. `assemble_frame_obu_payload_single_tile`/`assemble_obu_frame_single_tile`:
`num_tiles==1` only (`tiles_log2==0`, asserted) — `write_tile_group_header`'s
own C twin hard-returns zero bits at `tiles_log2==0` (`av1/encoder/bitstream.c`),
so this collapses to frame-header bits + one byte-align + the sole/last tile's
raw bytes verbatim (no `tile_size_bytes` length prefix, matching the decoder's
`split_tiles`). Multi-tile (length-prefixed non-last tiles) is NOT implemented —
the next lift once the envelope needs more than one tile.

**Verified against REAL aomenc output, not just constructed**:
`tile_group_obu_matches_real_encoder.rs` extends `frame_header_matches_real_
encoder.rs`'s exact setup one step further — extracts the REAL raw tile bytes
following the real parsed frame header, explicitly verifies the frame header's
trailing partial byte is zero-padded (not assumed), re-assembles via
`assemble_frame_obu_payload_single_tile`, and asserts the result equals the
COMPLETE real `OBU_FRAME` payload byte-for-byte. Passes on all 4 cases (64x64
420/mono ALLINTRA, 64x64 420 GOOD, 128x64 2-SB-wide ALLINTRA — all genuinely
`tiles_log2==0`, `frame_payload` 6-7 bytes). Assembly-verified (matches
`frame_header_matches_real_encoder.rs`'s own honesty framing): proves the
WRAPPING reproduces real aomenc's byte layout given the SAME header values and
tile bytes, not that this port's own search DERIVES those tile bytes.

## Full encoder-gate e2e byte match — the smallest frame TRUE-DERIVED end to end (2026-07-14, encoder track)

**The headline encoder-gate deliverable: this port's OWN search + pack pipeline
(not real aomenc's bytes copied back) produces byte-identical output to real
aomenc, for the smallest single-SB all-intra frame.**
`crates/aom-encode/tests/encoder_gate_e2e_byte_match.rs` drives
`rd_pick_partition_real` + `pack_tile` (Task 1's now-full per-txs_ctx coeff
costs) over the SAME source pixels real aomenc encoded, wraps the result via
`obu_assemble`, and compares byte-for-byte against the complete real `OBU_FRAME`
payload. Only the frame header is bootstrapped verbatim from the real parse
(qindex, tile info, tx-mode-select, cdf-update flag, loop-filter level — no
LF-level/CDEF-strength search is ported, matching `frame_header_matches_real_
encoder.rs`'s already-documented boundary); every partition/mode/tx/coefficient
decision and the resulting tile-group bytes are this port's own derivation.

**`encoder_gate_e2e_attempt` (asserted, hard regression gate): 3/3 flat-content
64x64 cases** (mono/4:2:0 ALLINTRA, 4:2:0 GOOD) **achieve a TRUE end-to-end byte
match.** Honestly labelled as the SMALLEST possible case — a 1-byte tile payload
(EOB=0/txb_skip=1 everywhere) that does not exercise the coefficient-cost tables
at all.

**`encoder_gate_e2e_textured_attempt` (exploratory, NOT asserted): 6 of 7
genuinely textured 64x64 mono ALLINTRA cases ALSO byte-match end-to-end**
(horizontal/vertical/diagonal gradients, two-tone left-right and top-bottom
splits, a 16px checkerboard — up to 45 bytes of real, non-trivial
coefficient-coded tile data, genuinely exercising the coeff-cost fix above —
this is the substantive result, not the trivial flat case). The 7th,
pseudo-random noise (the hardest case — forces many independent per-block RD
decisions across the SB), DIVERGES: first mismatched byte at offset 1139 of
~1520-1536 total tile-group bytes (i.e. roughly the first 1130+ bytes DO
match), this port's encode 15 bytes SMALLER (1516 vs 1531 tile bytes) —
consistent with a genuine RDO decision divergence somewhere deep in the SB's
recursive search (missing AB/4-way partition types, or a search-order/pruning
subtlety), not a wiring bug or crash (a bug would far more plausibly produce
garbage/a crash than a smaller-but-still-valid, 1130-bytes-agreeing bitstream).
**Root cause NOT isolated this session** — would need decode-both-and-diff-trees
investigation (parse both bitstreams past byte 1139, compare the partition
tree / mode / tx choices each side made) to pin down exactly which decision
diverges and why.

### MISSING (honest, as of this chunk)

- **AB partition types** — `HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B` are unported
  (6 of 10 -> now 8 of 10 with 4-way landed, see the "4-way partitions
  ported" milestone below). AB additionally needs the `reuse_prev_rd_results_
  for_part_ab` context-copy mechanism (`pick_sb_modes`'s `rd_mode_is_ready`
  early-return — LIVE at speed 0 both usages, unlike `reuse_best_prediction_
  for_part_ab`'s mode-cache path which is OFF at speed 0 both usages, so that
  half is dead-and-skippable) plus its own NN prune (`av1_prune_ab_
  partitions`, partition_strategy.c ~line 1300, same shape as the 4-way NN
  this chunk ported — not yet traced). The AB-probe test
  (`encoder_gate_e2e_ab_attempt`, unasserted) is currently CONFOUNDED by the
  nonzero-LF gap below (mismatches at byte 0, inside the header, before any
  AB-relevant data) — needs either LF-level search landing first, or
  lower-contrast AB-triggering content that still avoids the LF search
  picking a nonzero level, to get a clean read.
- **Loop-filter-level search** (`av1_pick_filter_level`) — every e2e case
  previously observed `lf_level=[0,0]` (bootstrapped from the real parse, not
  derived); this chunk's AB-probe content is the FIRST case to reach
  `lf_level != [0,0]` end-to-end (`[7,8]` .. `[8,16]` observed) and it
  mismatches at byte 0 (inside the bootstrapped header) — CONFIRMED broken,
  not just untested. Root cause not investigated (out of this chunk's
  partition-type scope); next LF-track session should start here instead of
  re-discovering it.
- **CDEF-strength search** — sidestepped via `enable_cdef=false` throughout;
  the CDEF-strength RD search itself is not part of this chunk.
- **The qindex-from-cq-level / rate-control mapping** — this chunk always
  reads `p.quant.base_qindex` from the REAL parsed header rather than deriving
  it from `--cq-level`; the encoder-side rate-control/qindex-selection logic is
  out of scope here.
- **Multi-SB / multi-tile frames** — `obu_assemble` hard-asserts
  `tiles_log2==0`; `pack_tile` itself supports multi-SB tiles (see the
  pack-stage milestone) but the e2e byte-match harness here has only been run
  at n_sb=1 (single 64x64 SB). The 128x64 (2-SB) case is verified for Task 2's
  ASSEMBLY test (real tile bytes) but not attempted for Task 3's full
  derivation.
- ~~The exact root cause of the pseudo-random-noise divergence~~ — **FOUND**:
  see the "4-way partitions ported" milestone below (decode-diff isolated it
  to a `PARTITION_VERT_4` choice at a specific 16x16 node; now fixed).

## 4-way partitions (HORZ_4/VERT_4) ported — noise-case divergence RESOLVED (2026-07-14, encoder track)

**Decode-diff methodology (Task 1):** new
`crates/aom-encode/tests/decode_diff_noise_case.rs` decodes BOTH this port's
own bitstream and real aomenc's bitstream for the one previously-diverging
`encoder_gate_e2e_textured_attempt` case (pseudo-random noise) with the
already-bit-exact-vs-C decoder (`aom_decode::frame::
decode_frame_obus_prefilter`), then diffs `KfTileDecode::tree` (the
pre-order partition-symbol sequence) index-by-index instead of the raw
bytes — byte offset alone doesn't localize the true first divergent symbol
because range-coder carry propagation can shift the visible effect later
than the actual diverging decision. **Finding: first divergence at
(mi_row=8, mi_col=8, bsize=BLOCK_16X16) — real aomenc chose
`PARTITION_VERT_4`, this port's search chose `PARTITION_VERT`** (VERT_4
wasn't in its candidate set). A second 4-way node (`HORZ_4`) appears later
in the real tree too. Confirmed, not guessed — direct decode-side evidence.

**Port (Task 2):** `rd_pick_4partition` (partition_search.c:3919) — the
HORZ_4/VERT_4 4-equal-strip RD search, reusing the already-verified
`rd_pick_rect_partition` leaf primitive (its budget-subtraction +
accumulate-or-invalidate shape is exactly `rd_try_subblock`, already
partition-type/position-generic) — plus `prune_4_way_partition_search`'s
gating (partition_search.c:4120), including the REAL neural-net prune
(`av1_ml_prune_4_partition`, partition_strategy.c:1326): confirmed LIVE at
speed 0 both usages (`part_sf.ml_prune_partition = 1` unconditionally at
the top of both `set_allintra_speed_features_framesize_independent` and
`set_good_speed_features_framesize_independent` — NOT gated by any `speed
>= N`), unlike the rect-stage's NN prunes which are all intra-dead. Only
the `ml_model_index == 1` ("hd_" weight set, 3-way softmax) branch is
reachable (`ml_4_partition_search_level_index` stays 0 at speed 0 both
usages, `0 < 3` picks that branch) — the other weight variant is
intentionally not transcribed. Weight tables (mean/std/2-layer-MLP
weights+biases for bsize 16/32/64, plus the search/not-search threshold
tables) mechanically transcribed from `partition_model_weights.h` by a new
`xtask/transcribe_part4_nn.py` (mirrors the established
`transcribe_nz_ctx.py` pattern) into `crates/aom-encode/src/
part4_nn_weights.rs`; the NN forward pass + feature engineering +
threshold decision live in the new `part4_prune.rs`. `SbTree` gained
`Horz4`/`Vert4` variants (`Box<[LeafWinner; 4]>`), threaded through
`encode_sb.rs`'s dry-run walk and `pack.rs`'s real OUTPUT_ENABLED walk —
both mirror the exact C recursion (`encode_sb`'s own
`PARTITION_HORZ_4`/`VERT_4` arms, partition_search.c:1690-1705, cross-
checked against the decoder's already-verified `decode_partition`'s
matching arms for the same quarter-strip stepping + frame-bound trim).
Interior-envelope simplification (matching the existing HORZ/VERT scope):
this port only attempts a 4-way type when all 4 quarter-strips are
guaranteed in-frame, not just the C's own half-block `has_rows`/`has_cols`
check — an edge 4-way candidate with fewer than 4 codeable strips is out of
scope (next lift). New `PickFrameCfg::enable_1to4_partitions` flag gates
the whole stage (`false` at every pre-existing call site, preserving their
established NONE/SPLIT/HORZ/VERT-only behavior/assertions exactly; `true`
only at the two e2e-relevant test files).

**Result: `encoder_gate_e2e_textured_attempt` now 7/7 (was 6/7) — PROMOTED
to an asserted hard regression gate** (was exploratory/unasserted while the
noise case was open). `decode_diff_noise_case.rs` independently confirms
not just byte-identity but that the decoded partition trees AND every
shared leaf's mode/tx/uv_mode fields are identical — now also a hard gate
(was a diagnostic).

**New AB-partition probe** (`encoder_gate_e2e_ab_attempt`, unasserted,
exploratory): content engineered to make an AB split RD-attractive (one
half flat, the other split into two differently-textured quadrants).
Result: 0/4, but CONFOUNDED — all 4 mismatch at byte 0 (inside the
bootstrapped frame header, before any partition data), because this
content pushes real aomenc's independent LF search to a nonzero level for
the first time in this test family, hitting the separate pre-existing gap
now documented above. Does not (yet) give a clean read on AB need
specifically.

**AB (`HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B`) is NOT ported this chunk.**
Commits: `756dfa1` (decode-diff investigation), `ae19fe6` (pre-existing fmt
drift, separated), `9cbc11a` (the 4-way port).

### AB partition NEXT-CHUNK plan (traced this session, not yet implemented)

Full C read completed (partition_search.c:3175-4023 + partition_strategy.c
:1223-2029), so this is a precise implementation plan, not a re-derivation
task. AB is genuinely a comparable-or-larger unit of work than 4-way was —
it has its OWN NN (different shape/weights than 4-way's) plus a correctness-
critical mutable-state subtlety in the C reference that must be threaded
through several ALREADY-SHIPPED, verified function signatures. Decomposed
here so a future session can move directly to implementation.

**1. Structural leaf search** (`crates/aom-encode/src/partition_pick.rs`):
- `rd_test_partition3` (partition_search.c:3177): loop over 3 sub-blocks
  (`SUB_PARTITIONS_AB`), each a `rd_try_subblock` call — reuse
  [`rd_pick_rect_partition`] as the per-leaf primitive exactly like the
  4-way port did (same budget-subtract + accumulate-or-invalidate shape;
  early-bail after EVERY sub-block, not just the last).
- `ab_subsize`/`ab_mi_pos` tables per AB type (partition_search.c
  :3805-3831): HORZ_A = `{split_bsize2@origin, split_bsize2@(mi_row,
  mi_col_edge), horz_subsize@(mi_row_edge, mi_col)}`; HORZ_B =
  `{horz_subsize@origin, split_bsize2@(mi_row_edge, mi_col),
  split_bsize2@(mi_row_edge, mi_col_edge)}`; VERT_A/VERT_B mirrored on the
  column axis. `split_bsize2 = get_partition_subsize(bsize, SPLIT)`.
- Dry-run propagation after sub-blocks 0 and 1 (not 2, the last) — same
  `encode_b_intra_dry` + `grid.stamp` pattern as 4-way's loop.
- New `SbTree::HorzA/HorzB/VertA/VertB(Box<[LeafWinner; 3]>)` variants +
  matching arms in `encode_sb.rs::encode_sb_dry`, `pack.rs::pack_sb`,
  `partition_pick.rs::stamp_grid_from_tree` — mirror `encode_sb`'s own
  `PARTITION_HORZ_A/HORZ_B/VERT_A/VERT_B` arms (partition_search.c
  :1652-1673) exactly (already cross-referenced against the decoder's
  matching `decode_partition` arms, `aom-decode/src/lib.rs:1916-1935`).

**2. `allow_ab_partition_search`** (partition_search.c:3992): same shape as
4-way's `partition4_allowed_base` gate (`do_rectangular_split && bsize >
BLOCK_8X8 && has_rows && has_cols`; `ext_part_eval_based_on_cur_best`
branch verified dead at speed 0 both usages, same as 4-way — omit).

**3. `av1_prune_ab_partitions`** (partition_strategy.c:1901) — MORE gating
than 4-way had, because `prune_ext_partition_types_search_level == 1`
(nonzero) is LIVE at speed 0 both usages (unlike 4-way's own consumption of
the SAME sf at `== 2`, which is dead) — so ALL of this runs, not just the
NN:
  - `horzab_partition_allowed`/`vertab_partition_allowed` base gate:
    `ext_partition_allowed && enable_ab_partitions (default true, same
    pattern as enable_1to4_partitions/enable_rect_partitions —
    `disable_ab_partition_type == 0`) && partition_rect_allowed[HORZ/VERT]`.
  - `level == 1` branch (:1924-1934): kills horzab/vertab unless
    `pc_tree->partitioning` (this port's `pc_tree_partitioning`, ALREADY
    tracked as of the 4-way chunk) is HORZ/SPLIT (or NONE with
    `pb_source_variance < 32`) — vertab mirrors on VERT.
  - RD-ratio pruning (:1956-1970, `case 1` branch since level==1):
    `ab_partitions_allowed[HORZ_A] &= (horz_rd[1]+split_rd[0]+split_rd[1])
    /16*14 < best_rdcost` (HORZ_B/VERT_A/VERT_B mirrored per
    :1957-1990's exact index pairing) — `rect_part_rd`/`split_rd` are
    ALREADY threaded (4-way chunk), just need the `< INT64_MAX ? : 0`
    clamp at :1941-1948 applied first.
  - `ml_prune_ab_partition` (below) gated by `ml_prune_partition` (already
    established LIVE at speed 0 both usages) `&& partition_rect_allowed
    [HORZ] && [VERT]`.
  - `evaluate_ab_partition_based_on_split` (:2009-2028): gated by
    `prune_ext_part_using_split_info >= 2`, which is 0 at speed 0 both
    usages (established in the 4-way chunk) — DEAD, omit.

**4. `ml_prune_ab_partition`** (partition_strategy.c:1223) — a SEPARATE,
SIMPLER NN than 4-way's (no mean/std normalization, no softmax — same
`int_score = 100*score`, `thresh = max_score - bsize_offset`, 16-way
bitmask decode pattern as 4-way's DEAD `ml_model_index==0` branch, except
here it's the ONLY variant and it's LIVE). 10 features (`part_ctx`,
`get_unsigned_bits(x->source_variance)` — NOT `pb_source_variance`, see
gotcha #1 below —, 8 RD-ratio features from `rect_part_rd`/`split_rd`
identical in shape to 4-way's first 10). Weight tables: `av1_ab_partition_
nnconfig_{16,32,64,128}` in `partition_model_weights.h:332/654/976/1298`
(4 bsizes, note 128 IS reachable for AB unlike 4-way's 16/32/64-only) — a
NEW `xtask/transcribe_ab_nn.py` following the EXACT pattern of
`transcribe_part4_nn.py` (regex-parse, re-emit Rust, verify via the same
f64-round-trip spot-check) is the fastest, safest way to get these in.
`ext_ml_model_decision_after_rect` (the external-model hook) is dead for
the same reason 4-way's `ext_ml_model_decision_after_part_ab` was
(`!frame_is_intra_only` required, always false here).

**5. `reuse_prev_rd_results_for_part_ab`** (LIVE at speed 0 both usages,
unconditional at the top of both allintra/good speed-feature setters —
verified this session) — the FULL PICK_MODE_CONTEXT copy-not-research
mechanism (`pick_sb_modes`'s `rd_mode_is_ready` early-return,
partition_search.c:854-855, currently dead in this port — module docs on
`leaf_pick_sb_modes` already flagged this as "dead until the AB chunk").
Needs, PER AB SUB-BLOCK POSITION THAT GEOMETRICALLY COINCIDES with an
earlier stage's OWN leaf:
  - `is_split_ctx_is_ready[0]`/`[1]` (SPLIT children 0/1 only,
    partition_search.c:4599-4608): true iff that child's OWN top-level
    winning partition was NONE (or `bsize <= BLOCK_8X8`) AND its winning
    leaf is palette-free (always true, palette unsearched) AND
    `uv_mode != UV_CFL_PRED`. Requires the SPLIT stage to EXPOSE each
    child's `(is_none_leaf: bool, leaf: LeafWinner)` up to the caller —
    currently `rd_pick_partition_real`'s SPLIT-stage recursion only
    returns the whole subtree (`SbTree`), so this needs either a
    `matches!(child_tree, SbTree::Leaf(_))` check on the ALREADY-RETURNED
    `SbTree::Split` children (the tree shape alone tells you if a child
    was a NONE leaf — no new state needed, just pattern-match) — simpler
    than it first looks.
  - `is_rect_ctx_is_ready[HORZ]`/`[VERT]` (ALREADY tracked since the
    original HORZ/VERT port) feeds HORZ_B/VERT_B's first sub-block.
  - When ready, HORZ_A/VERT_A's first 1-2 sub-blocks and HORZ_B/VERT_B's
    first sub-block reuse the EARLIER stage's `LeafWinner` + `PartRdStats`
    VERBATIM (no re-search) instead of calling `leaf_pick_sb_modes` fresh
    — this is NOT equivalent to re-searching (the budget available differs
    between when the original stage ran and now), so it must be an actual
    copy-and-skip, not a "should give the same answer anyway" shortcut.
  - `reuse_best_prediction_for_part_ab` (the OTHER reuse mechanism,
    `x->use_mb_mode_cache`) is confirmed DEAD at speed 0 both usages (only
    set at `speed >= 1` allintra / `speed >= 2` good) — do NOT implement
    the mode-cache shortcut in `pick_sb_modes`/`av1_rd_pick_intra_mode_sb`,
    it is unreachable in this envelope.

**Gotcha #1 (correctness-critical, easy to get wrong): `x->source_variance`
staleness.** `ml_prune_ab_partition` reads `x->source_variance` — a
MACROBLOCK-level field every `pick_sb_modes` call overwrites unconditionally
(win or lose) — NOT `pb_source_variance` (the snapshotted once-per-node
value the 4-way NN correctly uses). The C's own comment flags this as
imprecise ("TODO: may not be the current block's variance... need to
retrain to fix it") but it is what real aomenc ACTUALLY runs, so bit-exact
parity requires replicating the staleness, not the "intended" behavior. By
the time `av1_prune_ab_partitions` runs (after SPLIT + HORZ + VERT), this
holds whatever the LAST leaf search set it to (VERT's sub-block 1 if VERT
ran, else HORZ's sub-block 1, else whatever SPLIT's last recursive leaf
touched). Implementation: thread a `last_source_variance: u32` through
`rd_pick_partition_real`'s scope, updated unconditionally after EVERY
`leaf_pick_sb_modes` call (win or lose) — requires widening
`leaf_pick_sb_modes` and `rd_pick_rect_partition`'s return tuples to also
carry `source_variance` (currently computed locally inside
`leaf_pick_sb_modes` and discarded). Touches 3 already-shipped, verified
functions (`leaf_pick_sb_modes`, `rd_pick_rect_partition`, and by
extension `rd_pick_4partition` which calls `rd_pick_rect_partition`) —
budget real regression-test time for this refactor, don't rush it.

**Gotcha #2:** `pc_tree->partitioning` (this port's `pc_tree_partitioning`)
must reflect the 4-way stage's own win too by the time a PARENT node's own
AB stage reads it — already correctly threaded as of the 4-way chunk (the
4-way win site sets it), so this is just a note that the plumbing already
exists, not new work.

**Suggested verification path once implemented:** re-run
`encoder_gate_e2e_ab_attempt` (currently 0/4, confounded by the unrelated
nonzero-LF gap — fix or route around that first, e.g. by tuning content
contrast down further, to get a clean AB-specific read) and extend
`decode_diff_noise_case.rs`'s pattern (tree-diff against the real decode)
to any case that still doesn't match, to keep the same "confirm or refute
with the dump" discipline this session used for VERT_4.

## Loop-filter-level RD search ported — `av1_pick_filter_level` (2026-07-14, encoder track)

**New crate module `crates/aom-encode/src/lf_search.rs`**: `pick_filter_level`
(= `av1_pick_filter_level`) + `search_filter_level`, for the envelope this
port targets (single shown KEY frame, ALLINTRA usage=2 or GOOD usage=0,
one-pass). Every simplification is INDIVIDUALLY VERIFIED against the C
source (not assumed), documented in the module's doc comment:
`method == LPF_PICK_FROM_FULL_IMAGE` is the speed-0 default for BOTH usages
(grepped every `lpf_pick =` assignment site in speed_features.c — only
overridden at speed>=4/5/6/7, never at 0); `last_frame_filter_level` is
always `[0,0,0,0]` (`frame_is_intra_only` — KEY frame), so every search
starts at `filt_mid=0`, `filter_step=4`; `max_filter_level` is always
`MAX_LOOP_FILTER`=63 (one-pass, never `is_stat_consumption_stage_twopass`);
`use_coarse_filter_level_search`/`skip_loop_filter_using_filt_error`/
`adaptive_luma_loop_filter_skip` are all 0 (dead) at speed 0 both usages.
**Correctness-critical catch, verified against the real parsed header, not
assumed:** `cm->lf.mode_ref_delta_enabled` is actually **`true`** for a KEY
frame (`av1_setup_past_independence` -> `set_default_lf_deltas`), NOT false
as a first guess would assume — since every block in this envelope is intra
(`ref0==INTRA_FRAME==0`, `mode_lf==0`), this adds a uniform `+1` (or `+2`
once `base_level>=32`) to every block's effective filter level via
`ref_deltas[0]==1`. Trial deblocking reuses the ALREADY bit-exact
`aom_loopfilter::frame::loop_filter_frame` verbatim (new `aom-loopfilter`
dependency added to aom-encode) — never reimplemented; a new `sse_plane` fn
(bit-exact vs `aom_get_{y,u,v}_sse`/`highbd_get_sse` via the same
integer-addition-associativity argument `aom-dist`'s existing SSE-family
code relies on) computes each trial's cost. A new `build_lf_mi_grid` walks
this port's OWN picked+packed `Vec<SbTree>` (mirrors the existing
`stamp_grid_from_tree`/`ModeGrid` recursion pattern in `partition_pick.rs`,
duplicated for a different per-cell payload — 4th copy of this exact
recursion in the crate, consistent with the established convention).

**Wired into `encoder_gate_e2e_byte_match.rs`**: `attempt_case_content` now
calls `pick_filter_level` on THIS PORT'S OWN reconstruction (from
`pack_tile`) + the original source, overwriting the bootstrapped
`p.loopfilter.filter_level`/`_u`/`_v` before assembly (sharpness/deltas stay
bootstrapped — correct and constant in this envelope, not this mission's
scope). Every case now prints `DERIVED lf_level=... -- REAL(bootstrapped)
lf_level=... -- LF-LEVEL AGREES/DISAGREES`.

**Verified, not assumed: all 10 previously-passing cases (3/3 flat +
7/7 textured) re-derive `[0,0]` — a correct search DERIVES the quiet case,
it doesn't just default to it — and all 10 STILL byte-match end-to-end
(`encoder_gate_e2e_attempt`/`encoder_gate_e2e_textured_attempt` both stay
green, unchanged assertions).** This is the regression-safety proof the
mission required before touching any nonzero case.

**Nonzero-LF agreement (unasserted `encoder_gate_e2e_ab_attempt` probe, 4
cases): 2/4 EXACT match** (`[8,8]` vs `[8,8]` twice), **2/4 off by one
filter step** (`[8,16]` derived vs `[7,16]` real; `[7,8]` derived vs `[8,3]`
real) — plausibly attributable to the AB-probe content's OWN confound (AB
partitions unported, so this port's reconstruction is NOT byte-identical to
real aomenc's on this content — `real_tile_bytes.len()` vs
`our_tile_bytes.len()` differ on 3 of 4 cases — so the two searches are
optimizing SSE against slightly different pixels, not a proven bug in the
search itself). Root cause of the 2 near-misses NOT further isolated this
chunk (would need a genuinely AB-free nonzero-LF case to test the search in
isolation — see next steps).

### CONFIRMED BUG, NOT MINE TO FIX (`aom-entropy`, decoder-owned) — the AB-probe's real byte-0 mismatch has NEVER been about LF-level

**The prior session's hypothesis ("until lf_level is really derived,
byte-match is impossible... mismatches at byte 0") is REFUTED by direct
measurement.** With LF-level now genuinely derived and even EXACTLY
agreeing with the real value on 2 of the 4 AB-probe cases, ALL 4 cases
STILL mismatch at byte 0 with the IDENTICAL byte pair
(`our_payload[0]=76, real_payload[0]=70`) as before the LF-level fix —
proving LF-level was never the byte-0 cause (`encode_loopfilter`'s bits
are written many fields after `write_frame_header_prefix`/`write_frame_size`
in `write_frame_header_obu`'s order, so they structurally cannot affect
byte 0).

**Root-caused via a clean single-threaded diagnostic** (`cargo test
--test-threads=1` — the default parallel test runner interleaves `eprintln!`
output across the 3 test fns in this file, which produced a misleading
"identical struct" false-positive on a first, confounded attempt at this
comparison; flag for future sessions in this file): all 4 AB-probe cases
parse `p.prefix.allow_screen_content_tools == true` (real aomenc
auto-detected screen content in the checkerboard patterns — the
period-4/6 checkerboards look like exact-repeat "screen content" to the
heuristic) — but `p.allow_screen_content_tools` (the OUTER
`FrameHeaderObu` field, a DIFFERENT field from `p.prefix`'s own copy)
stays `false` in every case, including these.

**Exact location**: `crates/aom-entropy/src/header.rs`. `write_frame_header_obu`
(line ~1483) reads the OUTER field: `if p.allow_screen_content_tools &&
!p.superres_scaled { wb.write_bit(p.allow_intrabc as u32); }`. But
`read_uncompressed_header` (line 2940-2963) only ever sets `p.prefix.allow_screen_content_tools`
(via `read_frame_header_prefix`, line 2758) and separately computes
`p.allow_intrabc` (line 2962-2963, correctly reading the real bit using
`p.prefix.allow_screen_content_tools` — the READ side is fine) — but
**never assigns `p.allow_screen_content_tools = p.prefix.allow_screen_content_tools`
anywhere**. Since `read_uncompressed_header` starts from `let mut p =
cfg.clone();`, the outer field is silently left at whatever the caller's
`cfg` template had (`false` in every test in this file, since the field
isn't set there) — REGARDLESS of the actual per-frame parsed value.

**Effect**: for ANY KEY frame where `allow_screen_content_tools` is true
(auto-detected screen content, or an explicit `--enable-tools` flag), this
port's own `write_frame_header_obu` call SKIPS the `allow_intrabc` bit that
the real bitstream actually contains — a genuine one-bit-short
re-serialization, corrupting every bit from that point on. This explains
100% of the observed symptom (identical `76` vs `70` byte pair on all 4
AB-probe cases, a structural missing-bit shift, not a content-dependent
wrong value) and is UNRELATED to AB partitions or LF-level.

**Not fixed here**: `header.rs` is `aom-entropy`, decoder-owned — my crate
boundary forbids editing it (mission brief: "you must NEVER edit
aom-decode/aom-entropy/aom-intra/aom-restore/dec_shim.c... If you need a
symbol outside your crates, STOP and report"). The fix (for whoever owns
that file) is a one-line addition inside `read_uncompressed_header`, e.g.
right after `p.prefix = prefix;`: `p.allow_screen_content_tools =
p.prefix.allow_screen_content_tools;`. Flagging here + will report to the
coordinator directly; NOT attempted.

### Systematic search for a clean nonzero-LF case — 44 more combinations tried, NONE clean (honest, as of this chunk)

Two new exploratory (unasserted) tests, `encoder_gate_e2e_nonzero_lf_sweep`
and `encoder_gate_e2e_nonzero_lf_chroma_sweep`, swept a further **44
content/quality-level combinations** looking for a case that drives a
genuinely nonzero LF level (luma or chroma) WITHOUT hitting the
screen-content-tools bug above or needing AB partitions / unverified
multi-SB derivation:

- **19 single-SB (64x64) luma candidates** (steep row/diagonal gradients,
  high-contrast two-tone splits, a bright bar, a radial blob, pseudo-random
  noise, a non-exact-repeat amplitude-drifting ripple), each at cq
  {32,48,50,60,63} where applicable (qindex up to 255, i.e. the most
  aggressive quantization this port's cq-level mapping reaches): **every
  single one derives `lf_level=[0,0]`, agreeing exactly with the real
  value.** Strong NEGATIVE evidence, not a gap: a single 64x64 superblock
  apparently almost never has enough accumulated blocking-artifact SSE to
  outweigh `search_filter_level`'s bias-toward-lower-levels term, REGARDLESS
  of content sharpness or quantizer aggressiveness — nonzero LF at this
  scale seems to specifically need fine/periodic texture (like the AB-probe
  checkerboards), which is exactly the content shape that ALSO trips
  screen-content-tools auto-detection.
- **6 multi-SB candidates** (128x128 / 256x256, vertical stripes + noise):
  3 of the 4 stripe cases **DO** derive a nonzero level where the real
  value is `[0,0]` (e.g. derived `[0,5]` vs real `[0,0]`) — but ALL 6
  multi-SB cases ALSO show `real_tile_bytes.len() != our_tile_bytes.len()`
  (or a tile-data mismatch even when lengths coincidentally match), i.e.
  this port's OWN reconstruction is NOT byte-identical to real aomenc's at
  multi-SB scale for ANY of these cases — a SEPARATE, PRE-EXISTING,
  out-of-scope gap (STATUS.md's own e2e milestone already flagged multi-SB
  full derivation as untried: "the e2e byte-match harness here has only
  been run at n_sb=1"). **This disagreement is NOT attributable to a proven
  bug in `pick_filter_level`/`search_filter_level`** — it's at least as
  consistent with the search correctly reacting to a genuinely different
  (non-byte-identical) reconstruction as with a search bug, and multi-SB
  e2e derivation was never a solid foundation to test LF-level against in
  the first place. Flagged for whoever picks up multi-SB e2e derivation
  next — re-test LF-level agreement once THAT foundation is solid.
- **8 chroma-only candidates** (flat luma 128, textured chroma: checkerboards
  p2/p4, stripes, noise) at cq {32,48}: **every one derives
  `filter_level_u=filter_level_v=0`**, agreeing with the real value; 7/8
  achieve a FULL end-to-end byte match (including a genuinely-textured
  "chroma noise" case) — good additional regression-safety evidence, but
  none exercises the chroma search against a genuinely nonzero real value
  either.

**Honest conclusion: across 10 (original) + 44 (this chunk) = 54 total
content/config combinations, zero single-SB cases were found where BOTH (a)
LF level is genuinely nonzero AND (b) the case is free of the
screen-content-tools bug or an AB/multi-SB confound.** This looks like a
structural correlation at this frame scale, not bad luck in content
selection — worth treating as a real finding for whoever continues this:
either (1) the `aom-entropy` screen-content-tools bug needs fixing first
(it's a one-line fix for whoever owns that file — see above — after which
the AB-probe's 2/4 EXACT lf_level matches would very plausibly become full
end-to-end byte matches on their own, giving a clean nonzero-LF gate for
free), or (2) multi-SB e2e derivation needs to be solidified first (a
larger, separate undertaking), or (3) accept that this envelope's smallest
demoable nonzero-LF proof requires one of the above rather than purely
single-SB content.

**What IS solidly proven, independent of the above:** the LF-level search
control-flow (`search_filter_level`'s binary-search-like walk, the bias
term, the dir=2→0→1 sequencing, the `mode_ref_delta_enabled=true` per-block
level correction) reproduces real aomenc's decision EXACTLY across every
zero-LF case tried (54/54 on the zero side) and on 2 of the AB-probe's 4
nonzero cases despite those specific 2 cases NOT being reconstruction-clean
— i.e. the search algorithm itself is not just "always guesses 0", it
correctly reproduces a genuinely-searched nonzero AGREEMENT when handed
inputs close enough to real aomenc's own.

## AB partitions (HORZ_A/HORZ_B/VERT_A/VERT_B) ported — 10 of 10 `PARTITION_*` types complete (2026-07-14, encoder track)

**Mission re-measurement first: `encoder_gate_e2e_ab_attempt` with the header
fix (`0d144b6`) in place is still 0/4, but the fix is CONFIRMED effective —
the byte-0 confound documented in the "Loop-filter-level RD search ported"
milestone is completely gone.** All 4 AB-probe cases now mismatch at byte 2
or byte 6 (never byte 0). The 2 cases with EXACT LF-level agreement ("top
split/bottom flat", "left flat/right split") now isolate cleanly to byte 6
— one byte past the 5-byte header, genuinely inside tile-group data; the
other 2 remain confounded by the separately-documented LF-level near-miss
(mismatch at byte 2, inside the header).

**New `decode_diff_ab_probe.rs` decode-diffs both clean cases against real
aomenc** (same method as `decode_diff_noise_case.rs`'s VERT_4 finding).
Result, not a guess: both cases' FIRST divergence is an IDENTICAL
NONE-vs-SPLIT decision at (mi_row=0, mi_col=8, bsize=BLOCK_32X32) — real
aomenc SPLITs, this port's search picks NONE. For "top split/bottom flat",
real's full tree uses ZERO AB partition types anywhere — proving this
specific divergence is a SEPARATE, non-AB RD gap (root cause not
investigated, out of this chunk's scope). For "left flat/right split",
real's tree DOES use one `HORZ_B` node (deep inside the (mi_row=8,
mi_col=8) quadrant) — confirming AB is genuinely needed there, but this
port's search never reaches that subtree because it already diverged
earlier at the SAME (0,8,32x32) node. **Net: the AB port alone does not
make either clean probe case byte-match end-to-end — a separate
NONE-vs-SPLIT gap blocks both, unchanged even with AB fully implemented and
enabled** (re-verified after the port landed: identical FIRST DIVERGENCE at
the identical position, byte-for-byte identical mismatch content). Flagged
for whoever investigates that gap next — it is NOT an AB/4-way/partition-
type-coverage issue (case 1 proves this directly: real's tree has no AB
anywhere and still diverges the same way).

**[SUPERSEDED 2026-07-14 — see "(0,8,32x32) NONE-vs-SPLIT was a palette-flag
write desync, NOT an RD gap — FIXED" below. The budget-propagation lead in this
paragraph was a RED HERRING: the search always picked SPLIT correctly at
(0,8,32x32); NONE failing there is FINE (real uses SPLIT too). The apparent
"(0,8) reads NONE" came from the decode-diff decoding a DESYNCED bitstream —
`pack.rs` omitted a palette-usage flag — not from any RD/budget decision.
Kept below as a record of the (incorrect) lead so nobody re-derives it.]**

A bounded (temporary, reverted, not committed) instrumentation pass at
`rd_pick_partition_real`'s NONE/SPLIT win checks (`partition_pick.rs`,
around the `if this_rdc.rdcost < best_rdc.rdcost` / `if reached_last_index
&& sum_rdc.rdcost < best_rdc.rdcost` sites) narrowed this further for a
future session: for "top split/bottom flat" specifically, the NONE stage at
(0,8,32x32) does not merely lose to SPLIT on cost — it produces NO valid
result at all (`this_rdc.rate == i32::MAX`, no leaf found), with the
`best_rdc` budget entering this node still `i64::MAX` (unbounded) at that
point. That points upstream, to how budget propagates down from the SB
root's own recursion (the outer (0,0,64x64) node's own NONE/rect stages)
rather than to anything local to (0,8,32x32) itself — worth starting there,
not re-deriving this narrowing from scratch. For "left flat/right split",
by contrast, NONE DOES succeed at this node (`rdcost=76062038`) and SPLIT's
own accumulated cost (`75392074`) is genuinely lower by a real ~0.9%
margin — a normal, close RD comparison, not a failure — so the two probe
cases may not even share the SAME root cause despite sharing the same
divergence position; treat them as two leads, not one.

**The AB port itself is complete and independently verified working**,
despite not closing the specific probe cases above (which were always known
to be confounded, per the 4-way chunk's own honest labelling). Ported (every
piece cross-referenced against libaom v3.14.1 `partition_search.c`/
`partition_strategy.c`, not guessed):

- `SbTree::HorzA/HorzB/VertA/VertB` + matching arms in `encode_sb.rs`'s
  `encode_sb_dry`, `pack.rs`'s `pack_sb`, `partition_pick.rs`'s
  `stamp_grid_from_tree`, and `lf_search.rs`'s `stamp_lf_tree` (a 4th
  independent tree-walk this port already had — caught by the compiler's
  own non-exhaustive-match error, not missed by inspection).
- `allow_ab_partition_search`, `av1_prune_ab_partitions` (base gate +
  `prune_ext_partition_types_search_level==1` structural RD-ratio pruning,
  LIVE at speed 0 — MORE gating than 4-way had), `ml_prune_ab_partition`
  (the AB-specific NN: `FEATURE_SIZE=10`, `LABEL_SIZE=16`, hidden=64 uniform
  across all 4 reachable bsizes, no mean/std normalization — weights
  transcribed by `xtask/transcribe_ab_nn.py` into `ab_nn_weights.rs`,
  consumed by `ab_nn_prune.rs`).
- `rd_pick_ab_part` (fuses the C's `rd_pick_ab_part` + `rd_test_partition3`):
  the 3-subblock search reusing `rd_pick_rect_partition` as the per-subblock
  primitive, early-bail after every sub-block, dry-run propagation after
  sub-blocks 0/1 (not 2, the last).
- `reuse_prev_rd_results_for_part_ab` (confirmed LIVE at speed 0, confirmed
  NON-OPTIONAL for bit-exactness — not just a perf optimization: a fresh
  re-search under AB's own budget can legitimately find a WORSE result than
  what an earlier stage found under a larger budget, since `pick_sb_modes`'s
  `rd_mode_is_ready` early-return copies `ctx->rd_stats` with NO budget
  re-check). The SPLIT stage now captures `is_split_ctx_is_ready[0]/[1]` +
  the children's own `LeafWinner` (survives past the SPLIT block regardless
  of SPLIT's own eventual win, matching the C's unconditional per-child
  bookkeeping); the rect stage captures `is_rect_ctx_is_ready` + its own
  sub-0 winner (previously tracked-but-discarded).
- The `x->source_variance` staleness gotcha (STATUS.md's own prior "AB
  NEXT-CHUNK plan" gotcha #1): `LeafWinner` gained `raw_rdstats: PartRdStats`
  (the Rust equivalent of `PICK_MODE_CONTEXT.rd_stats`) and
  `leaf_pick_sb_modes`/`rd_pick_rect_partition`/`rd_pick_4partition`/
  `rd_pick_partition_real` were widened to thread a `last_source_variance:
  &mut u32` in/out param (mirrors the existing `none_rd_out` pattern),
  updated unconditionally on every leaf search (win or lose), NOT touched by
  a reused sub-block (matching the C's early return skipping that
  assignment).

**Verification (honest, evidence not narrative):**
- Landed in 2 commits behind a new `PickFrameCfg::enable_ab_partitions`
  flag, `false` at every pre-existing call site (mirrors the established
  4-way rollout precedent exactly): the foundation commit (source_variance/
  raw_rdstats threading + NN weights) and the structural-search commit were
  BOTH independently verified behavior-neutral — full `aom-encode` suite,
  46/46 test-result lines pass, unchanged, with the flag off.
- Flipping `enable_ab_partitions: true` at the two AB-relevant test files
  (`encoder_gate_e2e_byte_match.rs`, `decode_diff_ab_probe.rs`): full
  `aom-encode` suite STILL 46/46, including both hard e2e gates unchanged
  (`encoder_gate_e2e_attempt` 3/3, `encoder_gate_e2e_textured_attempt` 7/7)
  — AB being available never regresses an already-verified case.
- **Direct, measured, positive evidence AB actually fires and is correct,
  not just "doesn't break anything":** `encoder_gate_e2e_nonzero_lf_chroma_sweep`
  (exploratory, unasserted) improved from 7/8 to 8/8 with AB enabled — the
  "chroma noise cq48" case (flat luma / pseudo-random-noise chroma, 4:2:0)
  flips from a byte mismatch to a TRUE END-TO-END BYTE MATCH. A temporary
  debug print (added, verified, then reverted — not committed) at the AB-win
  site directly confirmed AB types 1 (`HORZ_B`), 2 (`VERT_A`), and 3
  (`VERT_B`) winning at multiple nodes across this exact case, with the
  final assembled bitstream STILL byte-identical to real aomenc's own —
  i.e. this port's own AB decisions are bit-exact on a real, exercised case,
  not merely "compiles and doesn't crash."
- **HORZ_A observation gap now CLOSED (2026-07-14).** The one AB type the
  landing chunk had not caught winning — `HORZ_A` (type 0) — is now confirmed
  byte-exact by the same method. An env-gated AB-win print (added, verified,
  reverted — NOT committed) run over `encoder_gate_e2e_nonzero_lf_sweep`
  single-threaded (so per-case attribution is deterministic) observed all 4 AB
  types winning (HORZ_A 14x, HORZ_B 8x, VERT_A 9x, VERT_B 6x across the sweep),
  and in **2 independent cases HORZ_A won AND the frame is a TRUE END-TO-END
  BYTE MATCH vs real aomenc**: HORZ_A at (mi_row=0,mi_col=8,BLOCK_16X16) on the
  64x64 mono cq48 screen-content case, and at (8,12,BLOCK_16X16) on the 64x64
  mono cq60 case. So the `HORZ_A` pack path is bit-exact where exercised, not
  just structurally present — all 4 AB types now have direct positive
  byte-match evidence. (The other 3 HORZ_A-winning sweep cases mismatch, but on
  the separately-tracked header/LF or deep-RD gaps, not on the HORZ_A leaf.)

**10 of 10 `PARTITION_*` types are now structurally ported** (NONE/SPLIT/
HORZ/VERT/HORZ_4/VERT_4/HORZ_A/HORZ_B/VERT_A/VERT_B), each independently
byte-verified on at least one real, non-trivial case — but "ported" here
means the search+pack machinery is complete and producing bit-exact output
where exercised, NOT that the AB-probe's own specific content now passes
(it doesn't, see above — a separate, still-open NONE-vs-SPLIT gap). Commits:
foundation (source_variance/raw_rdstats threading, NN weight transcription,
decode-diff evidence) + structural search/wiring (this milestone's headline
port).

## (0,8,32x32) NONE-vs-SPLIT was a palette-flag write desync, NOT an RD gap — FIXED (2026-07-14, encoder track)

**Root cause (corrects the AB milestone's budget-propagation lead above).** The
AB-probe's identical "(mi_row=0, mi_col=8, BLOCK_32X32) real=SPLIT / ours=NONE"
divergence was never an RDO-parity bug. This port's partition SEARCH always
picked SPLIT at that node — matching real. The mismatch was a WRITE→DECODE
desync inside our own pack path, and the decode-diff (which decodes the port's
own bytes to recover its tree) was misreading a corrupted downstream node.

Mechanism: `pack.rs` hardcoded `KfBlockState::allow_palette = false`. But
`av1_allow_palette(cm->features.allow_screen_content_tools, bsize)` (blockd.h)
is TRUE for every DC-predicted block in `[BLOCK_8X8, 64x64]` when screen-content
tools are on — and the checkerboard AB-probe content turns
`allow_screen_content_tools` ON in the real frame header (`usage=2`,
allintra + the SCT content heuristic). The decoder (`aom-decode`'s
`decode_block`: `st.allow_palette = av1_allow_palette(...)`) reads a
palette-usage flag UNCONDITIONALLY for those blocks. With `allow_palette=false`
the pack never emitted that symbol, so the arithmetic coder desynced starting at
the very first (0,0) 32x32 leaf. The decoded tree collapsed (search picked 25
nodes; the desynced decode recovered only 5), and the decode-diff then
misattributed (0,8) as NONE. NONE legitimately failing at (0,8,32x32) in the
search (the reverted instrumentation's `rate==INT_MAX` observation) was a
correct, harmless outcome — SPLIT is what wins there — not the bug.

**Fix (entirely within `aom-encode`, no decoder/entropy changes — commit
`5e3e5c7`):**
- Thread `allow_screen_content_tools` into `PackCfg` (must equal the real frame
  header's value); real-stream harnesses pass the parsed header field.
- `pack_leaf` sets `kfs.allow_palette = allow_palette(cfg.allow_screen_content_
  tools, bsize)` — the SAME per-block gate the decoder applies. This envelope
  never uses palette, so `write_mb_modes_kf_fc` still emits the no-palette
  symbol, but the symbol is now WRITTEN where the decoder expects to read it.

**Permanent regression** (`decode_diff_ab_probe.rs`, both LF-clean cases): the
test now ASSERTS that the port's packed bytes decode back to exactly the
partition tree its OWN search chose (`sbtree_seq(search) == replay_tree(decoded)`;
the two walks are structural twins by construction). Pre-fix this FAILED
(search len 25 vs decoded len 5); post-fix both cases match exactly (25==25,
33==33), and the (0,8,32x32) node reads SPLIT in port-search, decoded, AND real.

**Verification (evidence, not narrative):** full `cargo test -p aom-encode
--all-targets` green (exit 0, zero failures/panics); both hard e2e gates
unchanged (`encoder_gate_e2e_attempt` 3/3, `encoder_gate_e2e_textured_attempt`
7/7; `encoder_gate_e2e_ab_attempt` + both `nonzero_lf` sweeps all ok). Commits
verified on `origin/main`: `2237b9b` (style: cargo fmt), `5e3e5c7` (the fix).

**[SUPERSEDED 2026-07-14 — RESOLVED below in "the (2,12,BLOCK_8X8) small-block
screen-content divergence was a `recon_intra` last-txb variance-factor bug". Both
cases now match real aomenc's tree exactly; the AB-probe is a full port-vs-real
tree gate. Kept for the localization record.]**

**Honest remaining gap — the AB-probe is NOT a full port-vs-real tree match
yet.** With the desync gone, the trees agree with real down to ~node 10, then
diverge at DEEP small-block nodes:
- "top split/bottom flat": first divergence (2,12,BLOCK_8X8) — real=NONE,
  ours=HORZ.
- "left flat/right split": first divergence (8,12,BLOCK_16X16) — real=SPLIT,
  ours=HORZ.

Both are genuine deep RD-parity differences (the port over-selects HORZ at
small blocks), NOT desyncs — the bytes round-trip perfectly (self-consistency
asserted), the port's search simply makes a different (valid) partition choice
than real's RDO. Tracked diagnostic-only in the same test (the `FIRST
DIVERGENCE` print, unasserted).

**Measured port-side costs at the two diverging nodes** (env-gated PART_DBG
instrumentation in `rd_pick_partition_real`, run then reverted — NOT committed;
`rdmult` folded, PSNR, cq=32):
- "top split/bottom flat" (2,12,BLOCK_8X8): NONE rdcost=4,208,820 (dist=6021),
  SPLIT=4,183,283 (dist=3030), HORZ=**4,005,471** (dist=3392) — HORZ wins by
  ~4.9% over NONE / ~4.3% over SPLIT. NONE's whole-8x8 distortion (6021) is ~2x
  the sub-partitioned versions; not a near-tie. Real nonetheless picks NONE.
- "left flat/right split" (8,12,BLOCK_16X16): SPLIT rdcost=17,329,743
  (dist=17779), HORZ=**17,248,986** (dist=15366) — HORZ wins by only **0.47%**
  (HORZ trades +rate for −dist). A genuine near-tie; a sub-1% rate/dist error
  flips it. Real picks SPLIT.

**Ruled out** (so the next session doesn't re-check): this is NOT a cost-table,
speed, or config difference. The AB-probe and the byte-exact e2e gate
(`encoder_gate_e2e_byte_match`, 7/7 textured) build costs with the IDENTICAL
`derive_real_costs(&kf_write, …)` (now FULL per-`txs_ctx` coeff costs) and use
byte-for-byte identical `PickFrameCfg` (speed 0, `less_rectangular_check_level`
= 1, `enable_ab/1to4/rect_partitions` = true, min/max partition 0/15). The ONLY
variable is content — the checkerboard exposes a per-node RD divergence on the
SAME machinery that is byte-exact on the 7 textured cases.

**Concrete next attempt** (not yet done — the real remaining chunk): extract the
C oracle's OWN per-node NONE/SPLIT/HORZ `rdcost` at these two exact nodes (a new
append-only trace hook in `rd_shim.c`, which the encoder track owns) and diff
against the measured port values above. That is the only way to localize whether
the port's NONE-dist (case 1) or its HORZ rate/dist (case 2 near-tie) is the one
that disagrees with C — port and C share cost tables + speed + config, so the
divergence is in a per-node cost/context term that only a C-side dump can pin
down. The `x->source_variance` / entropy-context state feeding these leaves (all
correct-by-construction since the tree matches real to node 10) is the first
suspect for a context-carried difference.

## RESOLVED: the (2,12,BLOCK_8X8) small-block screen-content divergence was a `recon_intra` last-txb variance-factor bug — FIXED (2026-07-14, encoder track)

**Both AB-probe cases now produce partition trees byte-for-structure IDENTICAL to
real aomenc** (`decode_diff_ab_probe` ratcheted from diagnostic-only to a hard
`ours_seq == real_seq` gate — commit `335c1c9`). The "Concrete next attempt"
above was carried out: a per-node C rdcost trace (gdb over a `-O0 -g` debug
libaom build; integer RD decisions are `-O`-independent, so decisions match the
Release oracle) localized the divergence term-by-term.

**Root cause (not a cost-table / budget / partition-signaling issue — a
recon-plane side-effect the ALLINTRA variance factor reads).** At (mi_row=2,
mi_col=12, BLOCK_8X8), C evaluates NONE/SPLIT/HORZ/VERT and finds NONE the ONLY
viable candidate — SPLIT/HORZ/VERT each fail because one sub-block's mode search
returns INT_MAX (its best mode's rd exceeds the RD budget). The port instead
found HORZ viable and it won. The trace showed the HORZ bottom 8x4 (3,12): C's DC
leaf gets `intra_rd_variance_factor` = **2.999999** (pre_rd 3,309,769 → post_rd
9,929,305 ≫ budget 3,515,356 → rejected), but the port got factor **1.0**
(post_rd ≈ 3.31M < budget → accepted).

The factor divergence traced to the RECON PLANE the factor reads
(`xd->plane[0].dst.buf`). C's `dist_block_px_domain` (tx_search.c:1042-1064)
reconstructs into a **temp** buffer, and `recon_intra` (tx_search.c:930-932)
writes the reconstruction back into the dst plane **only for non-last txbs**
(guard `blk_row + txh_unit < mi_high || blk_col + txw_unit < mi_wide`) — the
LAST/bottom-right txb keeps the raw PREDICTION (nothing chains from it). So for a
flat DC block the dst reads as per-4x4-constant (variance ~0 → factor up to 3.0).
The port's `txfm_rd_in_plane_intra` reconstructed **every** txb, so the last txb
held the high-variance reconstruction → factor 1.0. Direct recon-plane dump: C =
`[81,81,81,81,104,104,104,104]` (flat), port (pre-fix) =
`[81,81,81,81,82,79,176,171]` (reconstruction); post-fix the port matches C
byte-for-byte and factor = 2.999999.

**Fix (`f5ffa70`, `crates/aom-encode/src/tx_search.rs`):** guard the
`recon_intra` reconstruction on `win.best_eob > 0 && (blk_row + txh_unit <
max_blocks_high || blk_col + txw_unit < max_blocks_wide)`. Only the recon-plane
CONTENT of the last txb changes (its rate/dist/rdcost were already computed by
`search_tx_type_intra`; nothing else in the RD search reads the last-txb recon —
the committed recon comes from the separate winner re-encode, unchanged). Case 2
(8,12,BLOCK_16X16, real=SPLIT) resolved with the SAME fix — it was the same root
term, as predicted.

**Two differential C-reference oracles had the identical missing guard** (which
is why they matched the pre-fix port): `common::c_uniform_txfm_yrd` and the
`uniform_txfm_yrd_diff` inlined walk both reconstructed the last txb. Both got
the same guard so they stay faithful to the real encoder (confirmed against the
real encoder by gdb: BLOCK_4X4 DC leaves flat prediction 128/152/177/105 in dst).
This is oracle correction, NOT test weakening — the fixed port matches real
aomenc (AB-probe + all e2e gates), and the oracles now match real aomenc too.

**Verification (evidence):** full `cargo test -p aom-encode --all-targets
--no-fail-fast` green — **73 tests, 0 failed**. Hard e2e gates unchanged
(`encoder_gate_e2e_attempt` 3/3, `encoder_gate_e2e_textured_attempt` 7/7,
`encoder_gate_e2e_ab_attempt`, both `nonzero_lf` sweeps ok). Commits verified on
`origin/main`: `f5ffa70` (fix + oracle corrections), `335c1c9` (ratcheted gate).

## Loop-filter-LEVEL derivation PROVEN bit-exact — the AB-probe's last mismatch is NOT an LF bug (2026-07-14, encoder track)

**Prior misdiagnosis corrected.** The "Loop-filter-level RD search ported"
milestone (and the mission that followed it) framed the AB-probe's nonzero-LF
near-miss (`[8,6]` derived vs `[8,3]` real on "top flat / bottom split") as a
suspected bug in `search_filter_level` (a bias term / filter-step / dir
sequencing / mode-ref-delta correction). **Direct measurement REFUTES that: the
LF-level search is bit-exact.** The near-miss is a *reconstruction* divergence,
not a search divergence.

**Airtight isolation experiment (now an asserted gate,
`encoder_gate_lf_level_bit_exact_vs_real`, commit `a358c92`).** For each AB-probe
case, decode real aomenc's OWN bytes to recover its EXACT pre-loop-filter
reconstruction + mi grid (`aom_decode::frame::{decode_frame_obus_prefilter,
build_lf_inputs}` — both already bit-exact vs C), and run THIS PORT's
`pick_filter_level` on THOSE pixels. Result: it reproduces real's coded level
**EXACTLY on all four cases**, including `[8,3]` for "top flat / bottom split"
(the case that "disagreed" end-to-end). Real codes GENUINELY NONZERO levels here
(`[8,8]`/`[8,3]`/`[7,16]`/`[8,8]` at cq32), so this exercises the nonzero search
path — not the trivial `[0,0]` every flat/textured e2e case reaches. The gate
asserts `saw_nonzero` so it can never pass vacuously.

**Why the e2e case still mismatches (localized, not guessed).** With a temp
diagnostic (added, measured, reverted before commit) the failing case's coded
tile is `TILE-BYTES-IDENTICAL=false` vs real (first differing byte 171), while
the 3 passing AB cases are byte-identical AND agree on LF. Decoding both trees
localizes the divergence to a SINGLE partition node: **`(mi_row=8, mi_col=12,
BLOCK_16X16)` — real picks `PARTITION_HORZ`, this port's search picks
`PARTITION_SPLIT`** (real tree `[3,0,0,3,8,8,8,8,3,7,1,…]` vs ours
`[3,0,0,3,8,8,8,8,3,7,3,…]`, first divergence at pre-order index 10). So the LF
search legitimately optimizes SSE against different pixels and picks `[8,6]`
instead of `[8,3]`. This is the SAME class of deep partition-RD gap the AB
milestone flagged ("NOT a full port-vs-real tree match yet"); the `recon_intra`
fix closed the (2,12)/(8,12)-for-two-cases instances, this is a residual one on
"top flat / bottom split" (real=HORZ vs ours=SPLIT — opposite direction from the
`recon_intra`-resolved (8,12) case).

**Net for the LF-level mission: DONE — the derivation is bit-exact and asserted.**
54/54 zero-LF + 3/3 byte-identical nonzero-LF e2e agreements + 4/4 on-real-recon
nonzero agreements. `encoder_gate_e2e_ab_attempt` is now 3/4 (was framed as
LF-blocked; actually blocked on the partition-RD node above). Stale
`encoder_gate_e2e_ab_attempt` docs ("AB partitions unported", "nonzero-LF header
path broken") corrected in the same commit — AB is ported and the header path is
exercised + correct on the 3 byte-matching nonzero-LF cases.

**Next attempt for whoever picks up the residual partition-RD gap** (a separate,
non-LF investigation): C-oracle the per-candidate rdcost at (8,12,BLOCK_16X16)
for the "top flat / bottom split" content (NONE/HORZ/VERT/SPLIT) via a debug
libaom trace or an append-only `rd_shim.c` hook, and diff against this port's
`rd_pick_partition_real` costs at that node — the port's SPLIT must be beating
HORZ where C's does not, so the divergent term is a per-node cost/context or
sub-block-RD difference (the recon plane / `source_variance` feeding those leaves
is the first suspect, as with `recon_intra`). Full `cargo test -p aom-encode
--all-targets`: **74 tests, 0 failed**.

## RESOLVED: the (8,12,BLOCK_16X16) real=HORZ/ours=SPLIT gap was a MISSING screen-content palette flag cost — FIXED (2026-07-15, encoder track)

**`encoder_gate_e2e_ab_attempt` is now 4/4 byte-identical and ASSERTED.** The last
AB-probe mismatch ("top flat / bottom split", 64x64 mono cq32 all-intra) is closed.
Root cause traced term-by-term against a from-source C oracle (throwaway `fprintf`
in a debug-rebuilt `libaom.a`, driven through the existing `shim_encode_av1_kf`
path, reverted after — NOT committed).

**The chain, measured on both sides at the exact node:**
- At `(mi_row=8, mi_col=12, BLOCK_16X16)` real picks `PARTITION_HORZ` (cost
  17,175,728), port picked `PARTITION_SPLIT` (16,908,071). The port's SPLIT was
  ~428k CHEAPER, so it beat HORZ; and that tighter `best_rdc` starved HORZ's
  second 16x8 sub-block of budget → its leaf returned `INT_MAX` → HORZ rejected.
- The SPLIT children diverged. Child 0 `(8,12,BLOCK_8X8)` matched C on its four
  4x4 SPLIT leaves but NOT on its `PARTITION_NONE` 8x8 leaf; children 1/2/3 (which
  read child 0's committed reconstruction as neighbours) then all diverged.
- Drilling the 8x8 `PARTITION_NONE` leaf at `(8,12)`: real picks `V_PRED`
  (mode 1, rdcost 5,930,130), port picked `DC_PRED` (mode 0, 5,929,731). The
  per-mode dump showed **identical** tokenonly rate (61623), dist (3792 for DC,
  1933 for V), and variance factor (1.0) — and V_PRED matched real EXACTLY. The
  ONLY divergence was DC's mode-info cost: **C = 1626, port = 1600, off by exactly
  26 bits**, which flipped a near-tie (DC 5,927,493 < V 5,927,892 in the port vs
  DC 5,929,731 > V 5,927,892 in C).

**Root cause (not a partition-signaling / variance-factor / recon issue — a leaf
RATE term):** `intra_mode_info_cost_y` adds the palette-Y "no-palette" flag cost
to every `DC_PRED` block when `av1_allow_palette(allow_screen_content_tools, bsize)`
holds — i.e. on `bsize >= BLOCK_8X8` in a screen-content frame (these checkerboard
AB-probes trigger real's screen-content auto-detection). The port had hardcoded
`try_palette: false` in `leaf_pick_sb_modes` (partition_pick.rs), omitting that
26-bit flag from EVERY DC candidate. The palette SEARCH stays out of scope; only
the FLAG cost was missing.

**Fix (partition_pick.rs):** added `PickFrameCfg::allow_screen_content_tools`
threaded into the leaf; the leaf now sets `try_palette =
allow_palette(cfg.allow_screen_content_tools, bsize)`, `palette_bsize_ctx =
palette_bsize_ctx(bsize)`, `palette_mode_ctx = palette_mode_ctx(up, 0, left, 0)`
(always 0 — this port never picks palette, so every neighbour's palette_size is
0; the real helper is called with the known-zero sizes so the invariant is
explicit). `intra_mode_info_cost_y` + the filled `palette_y_mode_cost` table were
already present. With DC correctly costed, V_PRED wins the leaf, the SPLIT children
match C, the node picks HORZ, and the tile is byte-identical. Verified: the port's
8x8 leaf now reports mode=1/rate=66018/dist=1933/rdcost=5,930,130 — byte-for-byte
C's values.

**Guardrail check:** no partition-RD term was changed on a guess — the 26-bit
palette flag was the single measured C-vs-port divergence at the node. The
`recon_intra` last-txb fix (f5ffa70) that closed the sibling real=NONE/SPLIT vs
ours=HORZ cases is untouched and independent (that was a variance-factor recon
term; this is a mode-info rate term). The other e2e gates stay green.

## Gate posture (honest)

Real, verified, ratcheting progress across BOTH tracks — but still a fraction of
the whole. Green so far: full transform subsystem, the *entire* scalar quantizer
surface (fp+b, lowbd+highbd, QM+flat), the full coefficient trellis (QM+flat), and
the entire symbol-coding stack (range coder + CDF adaptation). **The encoder-gate
MVP milestone is reached**: this port's own search+pack pipeline produces a
byte-identical AV1 bitstream to real aomenc for the smallest single-SB all-intra
KEY frame (flat content, asserted) and for 7 of 7 genuinely-textured variants
of it (also now asserted — see the "4-way partitions ported" milestone: the
4-way partition port fixed the one remaining pseudo-random-noise divergence)
— **the loop-filter LEVEL is now this port's own derivation too, PROVEN
bit-exact** (`av1_pick_filter_level` ported + wired into the e2e gate;
54/54 zero-LF + 3/3 byte-identical nonzero-LF e2e agreements + a new
asserted on-real-recon gate reproducing real's coded nonzero levels
`[8,8]`/`[8,3]`/`[7,16]`/`[8,8]` EXACTLY — see the "Loop-filter-LEVEL
derivation PROVEN bit-exact" milestone; the AB-probe's one remaining e2e
mismatch is a separate partition-RD node, not an LF bug).
CDEF-strength search and the qindex-from-cq-level mapping remain
bootstrapped from the real parse; sharpness/ref-deltas/mode-deltas stay
bootstrapped too but are provably frame-constant/correct in this envelope.
Every coded byte of the tile-group payload is this port's own derivation.
**10 of 10 `PARTITION_*` types are now structurally ported**
(NONE/SPLIT/HORZ/VERT/HORZ_4/VERT_4/HORZ_A/HORZ_B/VERT_A/VERT_B — see the
"AB partitions ported" milestone). The `aom-entropy` `allow_screen_content_
tools` outer-field-sync bug that previously blocked a clean AB-probe read is
FIXED (`0d144b6`); the AB-probe's own 2 clean cases still don't byte-match
end-to-end, but this is now root-caused to a SEPARATE, non-AB,
non-partition-type-coverage NONE-vs-SPLIT RD gap at `BLOCK_32X32` (proven by
a case whose real tree uses no AB partitions anywhere and still diverges
identically) — not a missing capability, an open investigation for a future
session. AB itself is independently verified bit-exact on real, exercised
content (a genuine byte-match flip in an exploratory sweep, with 3 of 4 AB
types directly observed winning and producing byte-identical output).
None of the four project gates (full-corpus correctness, ≤1.20× perf, full
coverage, zenavif parity) is satisfied yet; the machinery that makes each
mechanically checkable is in place and every landed module is byte-exact vs
C within it.

## Multi-SB scale e2e byte match — 256x256 + 512x512 ASSERTED (2026-07-15, encoder track)

**Every prior end-to-end gate was a single 64x64 superblock.** This lands the
FIRST multi-superblock byte-match proof: `encoder_gate_e2e_multi_sb_scale`
(`crates/aom-encode/tests/encoder_gate_e2e_byte_match.rs`) drives the port's
own `rd_pick_partition_real` + `pack_tile` pipeline over **256x256 (16 SB64)**
and **512x512 (64 SB64)** frames and asserts the assembled `OBU_FRAME` payload
is byte-identical to real aomenc. This exercises, for the first time e2e, the
multi-SB path: cross-SB above/left neighbour-context threading, one adapting
CDF shared across every SB of the tile, and the deblock/LF-level search over
interior SB edges. Commit `ac24458` (gate + localizer), `3961cae` (sharpened
localizer).

**Coverage: 12 of 16 swept (size, content, cq) cells byte-match, ASSERTED.**
The swept matrix is {256x256, 512x512} x {flat, two-tone L/R split, vertical
gradient, diagonal ramp} x {cq32, cq48}. Byte-matching (asserted): flat 256/512
cq32+cq48 (pure structural proof), two-tone-split 256/512 cq32+cq48 (hard
edges), vertical-gradient 256/512 (both cq at 512, cq48 at 256), diagonal-ramp
256 cq48. Single-tile, mono, ALLINTRA, speed 0; frame header bootstrapped from
the real parse (same boundary as the 64x64 gates). STRIDE in
`attempt_case_content_uv` widened `320 -> 320.max(w+4)` so 512-wide frames fit;
all <=316px cases keep stride 320 and are byte-for-byte unchanged (stride is
buffer padding only). Existing gates stay green: `encoder_gate_e2e_attempt`
3/3, `_textured_attempt` 7/7, `_ab_attempt` 4/4, `lf_level_bit_exact_vs_real`.

**Honest gap — the 4 excluded cells (steepest content at higher quality) and
their EXACT localized term.** The smooth diagonal ramp (energy on both axes) at
cq32 (256 + 512) and cq48 (512), plus the steep 256px vertical gradient at
cq32, diverge in the coded tile data. `decode_diff_multisb.rs` (a committed
diagnostic) decodes BOTH bitstreams with the bit-exact decoder and compares
partition trees + per-leaf mode/tx + per-txb `(eob, tx_type)` + reconstruction:

- It is NOT a structural multi-SB bug — FLAT 256x256/512x512, which exercises
  the identical cross-SB structure, byte-matches; only real-content coefficient
  competition diverges.
- Every diverging block's partition, bsize, intra Y-mode, angle-delta,
  filter-intra flag, tx_size and uv_mode MATCH real aomenc exactly. The
  divergence is confined to the **coefficient-optimization (`av1_optimize_b`
  trellis) layer**, two flavors:
  - **code-vs-skip** (diagonal ramp, block mi=(32,0) BLOCK_64X64 DCT_DCT): real
    codes `eob=1` (a DC coeff), ours codes `eob=0` (txb-skip), same tx_type,
    reconstruction IDENTICAL -> a rate-only trellis tie-break on a
    zero-recon-effect DC.
  - **coefficient-level** (vertical gradient): partition+mode+tx_size+eob+
    tx_type all identical, reconstruction diverges at luma (row=0,col=66)
    real=81 ours=80 -> a trellis level choice, off-by-one.
- Why cq48 matches but cq32 diverges: at heavier quant (qindex 192) there are no
  marginal coefficients for the trellis to tie-break on; at qindex 128 there
  are. The `optimize_txb` KERNEL is already bit-exact vs C given identical
  inputs, so the smallest next chunk is to C-oracle-trace the trellis INPUTS
  (coeff cost tables / rdmult / pre-trellis quantized levels) at block (32,0) of
  the diagonal-ramp-cq32 256x256 case and find which input diverges.

Multi-tile is the next e2e lift (now landed -- see below).

## Multi-tile e2e byte match — 2x1 / 2x2 / 4x1 ASSERTED (2026-07-15, encoder track)

**Multi-tile keyframes now byte-match real aomenc end to end.**
`encoder_gate_multitile_byte_match` (`crates/aom-encode/tests/encoder_gate_multitile.rs`,
ASSERTED 11/11): this port's OWN per-tile `pack_tile` +
`obu_assemble::assemble_multitile_frame_obu_payload` produce a byte-identical
`OBU_FRAME` payload vs real aomenc encoded with `AV1E_SET_TILE_COLUMNS`/`_ROWS`
(the committed append-only oracle `ref_encode_av1_kf_tiles`). Commit `6899bea`.

Coverage: tile grids **2x1, 2x2, 4x1** at 128/256/512, cq48 + cq32, flat AND
boundary-crossing gradients (`hgrad`/`vgrad`, whose value varies ACROSS the
tile-column/row seam). Each AV1 tile is entropy-INDEPENDENT: the gate gives each
tile a fresh `KfFrameContext` + `OdEcEnc` and an `SbEncodeEnv` whose
`tile_row/col_start/end` are the tile's OWN MI bounds (from the parsed header's
`col/row_start_sb`), so intra prediction / tx-size context / the RD search treat
the tile edges as unavailable and never read the adjacent tile's reconstruction.
The gradient cases specifically prove that isolation -- a cross-tile prediction
leak would change their coded bytes. New `assemble_multitile_frame_obu_payload`
appends the `num_tg == 1` tile-group (one `0x00` present-flag+align byte) then
the per-tile payloads, every tile except the last prefixed by a
`tile_size_bytes`-byte LE `tile_size_minus_1` (inverse of the decoder's
`split_tiles`; verified vs `av1/encoder/bitstream.c`: `num_tg==1 -> OBU_FRAME`,
present flag `= num_tg>1 = 0`).

**HONEST GAP #1 (decoder-owned, blocks multi-tile HEADER serialization):**
`aom-entropy::write_tile_info` (`crates/aom-entropy/src/header.rs:426-427`)
HARDCODES the multi-tile tail as `context_update_tile_id = 0` and
`tile_size_bytes_minus_1 = 3` (`tile_size_bytes = 4`) instead of writing the
real `p.context_update_tile_id` / `p.tile_size_bytes - 1`. Real aomenc picks the
minimum tile-size width (1-2 bytes for still images), so re-serializing a
multi-tile `FrameHeaderObu` does NOT round-trip -- it diverges in the tile_info
byte(s). `write_tile_info` takes only `&TileInfoHeader`, which doesn't even carry
those two fields (they live on `FrameHeaderObu`), so it can't be fixed from the
caller. The multi-tile gate therefore bootstraps the frame-header bytes VERBATIM
from the real parse (the header is bootstrapped anyway) and asserts only the TILE
machinery + tile-group assembly. Fix needed in aom-entropy: thread
`context_update_tile_id` + `tile_size_bytes` into `write_tile_info` and its call
site (`header.rs:1511`), mirroring libaom's `write_tile_info`
(`cm->context_update_tile_id`, `cm->tile_size_bytes - 1`).

**HONEST GAP #2 — RESOLVED (2026-07-15, encoder track): strong-LF was NEVER an
LF-search bug; it was reconstruction, now fixed.** The prior finding — a
combined-content probe (gradient + hard edges + texture in one frame) drove real
aomenc to STRONG loop-filter levels (`[15,6]`/`[16,5]`) where the port appeared
to disagree — was tested on the END-TO-END path, where the port's OWN
reconstruction (then lacking coeff-trellis + partition-RDO + `INTERNAL_COST_UPD_SB`)
differed from real's, so the port's LF search saw different pixels.

Isolated properly (feed real aomenc's OWN decoded pre-filter reconstruction + mi
grid to the port's `pick_filter_level`, via `lf_derived_vs_real_on_real_recon`),
the strong-LF SEARCH is **bit-exact**. A discovery sweep of 8 combined
high-texture generators × {128,256} × {40..63} found **16 strong (level >= 12)
cells — `[0,25]`, `[26,0]`, `[6,20]` (both axes), `[15,0]`, `[0,15]`, ... — and
the port derived real's coded level EXACTLY on real's own recon in ALL of them
(0 mismatches)**. So the earlier honest-stop was reconstruction, not the LF
search; the trellis/partition/`INTERNAL_COST_UPD_SB` fixes have since made recon
accurate enough that **15/16 of those strong cells ALSO byte-match END-TO-END**
(port's own recon + own LF derivation → identical bytes), including with
`screen_content=true`.

Landed as two asserted gates in `encoder_gate_e2e_byte_match.rs`:
- `encoder_gate_lf_level_bit_exact_vs_real` — EXTENDED to the strong regime:
  5 strong cases (`[15,0]`/`[6,20]`/`[0,15]`/`[0,25]`/`[26,0]`, on real's recon)
  alongside the 4 weak AB cases, with a `saw_strong` (level >= 12) anti-vacuous
  guard next to the existing `saw_nonzero`.
- `encoder_gate_e2e_rich_content_strong_lf` — NEW: the SAME 5 rich strong-LF
  generators byte-match real aomenc end-to-end (5/5), the promotion of the
  honest-stopped rich-content variant. Shared `fn` generators keep the two gates
  in lockstep (a flat regression fails the LF gate; a recon drift fails this one).

The only strong cell that did NOT e2e-match (`diag+vbars16 256x256 cq62`, real
`[1,17]`) is a residual non-LF coeff/partition near-tie, unrelated to the LF
search. Commit `4940315`. Full `aom-encode` suite green (49 test binaries, 0
failed); every guardrail gate unaffected.

## Multi-SB e2e byte match at 16/16 — `INTERNAL_COST_UPD_SB` per-SB cost update ported (2026-07-15, encoder track)

`encoder_gate_e2e_multi_sb_scale` now sweeps the FULL 16-cell grid (flat /
two-tone / vertical-gradient / diagonal-ramp × 256x256 + 512x512 × cq32 + cq48)
and asserts **all 16 byte-identical end-to-end** vs real aomenc. The 4
steep-content cells the gate previously excluded (256x256 vgrad+diag cq32; 512x512
diag cq32+cq48) now match. Commits `53431ae` (fix) + `76b1ffb` (gate). Full
`aom-encode` + `aom-txb` suite green (103 tests), no regressions.

**Root cause (one bug, two symptoms).** At speed 0 libaom's default cost-update
level is `INTERNAL_COST_UPD_SB` (`speed_features.c`: `coeff_cost_upd_level` =
`mode_cost_upd_level` = `INTERNAL_COST_UPD_SB`). At the start of EVERY superblock
it re-derives BOTH the coefficient cost tables (`av1_fill_coeff_costs`) AND every
mode-rate table (`av1_fill_mode_rates`: y_mode / tx_size / angle_delta / partition
/ skip / cfl / intra tx-type) from the CURRENT adapting tile entropy context
(`encodeframe_utils.c:1643/1658`); the search + encode of that SB use those costs.
The port used frame-init tables for the whole frame, so on the LATER superblocks
of steep continuous-tone content — whose CDFs have adapted enough to move the
tables — the stale rate flipped near-tie RD decisions:
- **Coefficient (code-vs-skip):** diagonal-ramp mi=(32,0) BLOCK_64X64 real eob=1
  vs port eob=0 — the stale TX_64X64 coeff cost, compounded by an unadapted br
  slot (see below).
- **Intra-mode (DC vs directional/PAETH):** 512x512 diagonal mi=(32,64)
  BLOCK_64X64 real D45 / port DC. C-oracle per-mode trace: distortion matched
  EXACTLY (DC 15856, D45 29424) but the port over-charged the mode-signaling rate
  (stale y_mode/tx_size/angle_delta), giving DC mode_info 1564 vs C 1355 and D45
  2918 vs C 2183 — flipping a near-tie the real encoder wins for the directional
  mode (correct for diagonal content).

**Fix.** `pack_tile` runs a full `derive_real_costs(kf, ..)` per superblock (`kf`
is the pack-adapting context, == C's `xd->tile_ctx`) and threads the result into
BOTH the search (`sb_pick_cfg`) and the encode (`sb_env`), reproducing both
updates in one shot. SB 0 / single-SB frames read the frame-init defaults
unchanged, so every smaller gate is byte-for-byte unaffected.

**Also fixed (`fill.rs`):** `av1_fill_coeff_costs` reads
`coeff_br_cdf[AOMMIN(tx_size, TX_32X32)]` (rd.c) — TX_64X64 shares TX_32X32's br
CDF and the arena's txs_ctx==4 br slot is never adapted; capping the br index at
TX_32X32 stops TX_64X64 mis-costing every level-range (Golomb) term off an
unadapted uniform CDF. New `coeff_costs_fill_diff.rs` pins the port's CDF->cost
derivation integer-for-integer against real `av1_fill_coeff_costs` across qindex /
txs_ctx / plane / eob_multi_size (new append-only `shim_fill_coeff_costs`).
`decode_diff_multisb.rs` (the structural localizer that pinned the last two cells
to the intra-mode divergence) is kept as the regression localizer and now reports
identical reconstruction on the cases it once flagged.

## Gate 2 — cpu-used sweep opened: speed-1 scaffold + all-intra sf-delta work-plan (2026-07-15, encoder track)

**Gate 2 requires the encoder bitstream bit-identical for EVERY `aomenc
--cpu-used` 0..=9.** Speed-0 is the frozen baseline (every section above).
This opens speed-1 for the all-intra KEY path.

### Scaffold — `SpeedFeatures` (`crates/aom-encode/src/speed_features.rs`, 11fb764)

Before this, the pipeline hardcoded speed-0 sf values at every call site
(`TxTypeSearchPolicy::speed0_allintra()`, `PickFrameCfg { speed: 0,
intra_pruning_with_hog: true, less_rectangular_check_level: 1, .. }`).
`SpeedFeatures::set_allintra(speed, allow_screen_content_tools, use_hbd)`
centralizes the speed→value mapping, transcribed from
`set_allintra_speed_features_framesize_independent` (+ `_dependent`) and the
`init_*_sf` defaults it overrides. `set_allintra(0,..)` reproduces today's
hardcoded values EXACTLY — locked field-by-field by
`speed0_allintra_matches_hardcoded`. Speed-0 is a byte-exact no-op (full
aom-encode suite green, 82 passed). Producer
`SpeedFeatures::tx_type_search_policy(skip_trellis, sharpness)` resolves the
level→threshold tables through the `DEFAULT_EVAL` column (winner-mode two-pass
`enable_winner_mode_for_*` first fires at speed>=4, speed_features.c:502-505,
so speed 0..=3 read DEFAULT_EVAL).

### Speed-0 → speed-1 sf deltas for a KEY / all-intra frame (the work plan)

`set_allintra_speed_features_framesize_independent` `if (speed >= 1)` block =
speed_features.c:386-422; `set_allintra_speed_feature_framesize_dependent`
speed-1 = :209-234. RELEVANT to an intra-still KEY frame (harness encodes
CDEF + restoration OFF):

| # | sf field | s0 → s1 | line | port consumes at s0? |
|---|----------|---------|------|----------------------|
| 1 | part_sf.intra_cnn_based_part_prune_level | 0 → 2 (screen 0) | 387 | NO — CNN intra partition prune not ported |
| 2 | part_sf.reuse_best_prediction_for_part_ab | 0 → 1 | 397 | partial — AB mode cache seeding |
| 3 | part_sf.ml_4_partition_search_level_index (dep) | 0 → 1 | 210 | ? 4-way ML prune thresh index |
| 4 | intra_sf.top_intra_model_count_allowed | 4 → 3 | 404 | NO — top-N model→full-RD gate |
| 5 | intra_sf.prune_palette_search_level | 0 → 1 | 402 | N/A (palette search out of scope) |
| 6 | intra_sf.prune_luma_palette_size_search_level | 1 → 2 | 403 | N/A |
| 7 | tx_sf.adaptive_txb_search_level | 1 → 2 | 406 | YES (tx_search.rs early-term) |
| 8 | tx_sf.intra_tx_size_search_init_depth_rect | 0 → 1 | 409 | ? intra tx-size search start depth |
| 9 | tx_sf.model_based_prune_tx_search_level | 1 → 0 | 410 | ? (1→0 turns a prune OFF) |
| 10 | tx_type_search.ml_tx_split_thresh | 8500 → 4000 | 411 | ? tx-split ML thresh |
| 11 | tx_type_search.prune_2d_txfm_mode | PRUNE_1 → PRUNE_2 | 412 | NO — 2D tx-type ML prune off at s0 (tx_search.rs:150) |
| 12 | tx_type_search.skip_tx_search | 0 → 1 | 413 | YES (tx_search.rs:795 all-zero break) |
| 13 | rd_sf.perform_coeff_opt | 1 → 2 (thresh 3200→1728) | 415 | YES (via policy) |
| 14 | rd_sf.tx_domain_dist_level | 0 → 1 (dist-type 0→1) | 416 | partial — tx-domain distortion |
| 15 | rd_sf.tx_domain_dist_thres_level | 0 → 1 (MAX→22026) | 417 | partial |

lpf (CDEF/restoration search): `cdef_pick_method` FULL→FAST_LVL1,
`dual_sgr_penalty_level` 0→1, `enable_sgr_ep_pruning` 0→1 — modeled but do NOT
affect bytes while the harness encodes CDEF + restoration OFF.

IRRELEVANT to intra-still KEY (inter/motion/intraBC-screen; each verified
consumer-gated `!frame_is_intra_only` in partition_search.c /
partition_strategy.c): `simple_motion_search_*`, `ml_predict_breakout_level`,
`ml_partition_search_breakout_*`, `ml_early_term_after_part_split_level`,
`inter_tx_size_search_init_depth_*`, `mv_sf.*`.

CAVEAT — qindex-dependent setter (`av1_set_speed_features_qindex_dependent`,
speed_features.c:2904-2937): its `if (speed == 0)` block turns ON many tx/RD
speed-1 features at LOW qindex, and does NOT run at speed 1. Match the existing
gates' (higher) qindex regime so the observed speed0↔speed1 delta is clean.

Winner-mode two-pass (`enable_winner_mode_for_coeff_opt` etc.) first activates
at speed>=4 → NOT needed for speed 1. `intra_pruning_with_hog` (stays 1 at s1,
→2 at s2), `less_rectangular_check_level` (stays 1 at s1, →2 at s3), and every
`UvLoopPolicy` field are UNCHANGED at speed 1.

### Coverage (honest fraction) — updated after slice 1

- sf-delta fields modeled in `SpeedFeatures`: 15 of 15 intra-still-relevant (values transcribed + unit-asserted vs source; `speed1_allintra_deltas_match_source`).
- **e2e harness wired to source from `SpeedFeatures::set_allintra(speed)`** — `encoder_gate_e2e_byte_match.rs` `attempt_case_content_uv` now takes `(cpu_used, speed)`; speed-0 callers pass `(0,0)` → byte-identical (all speed-0 gates green). The all-intra path sources `pol` / `speed` / `intra_pruning_with_hog` / `less_rectangular_check_level` from the scaffold.
- sf-deltas WIRED + BYTE-MATCHED at cpu-used=1: **5 of 15** — the tx-policy group #7 `adaptive_txb_search_level` 1→2, #12 `skip_tx_search` 0→1, #13 `perform_coeff_opt` 1→2, #14 `tx_domain_dist_level` 0→1, #15 `tx_domain_dist_thres_level` 0→1. #14 additionally required implementing `calc_pixel_domain_distortion_final` (tx_search.rs:2378-2381 recompute of the winner's pixel-domain distortion) — was a speed-0 `debug_assert` stub, now ported; guarded so speed-0 is byte-identical.
- content cases byte-identical at cpu-used=1: **14 of 14** (was 13/14; the vgrad-256-cq32 holdout is now FIXED — see KB-3 resolution below) — all 6 flat (64/128/256², cq32/cq48) + all 8 gentle-slope textured (two-tone/vgrad/diag @128²+256² cq48, vgrad@128² cq32, **vgrad@256² cq32**). Asserted in `encoder_gate_speed1_flat_allintra` + `encoder_gate_speed1_textured_allintra`.

NEXT LOCALIZATION TARGET (1 diverging cell): **vgrad 256×256 cq32** at speed 1
(byte-matches at speed 0 — an asserted `encoder_gate_e2e_multi_sb_scale` winner —
so a genuine speed-1 delta bites). Localized: the 5-byte frame header matches; the
first differing byte is **byte 5 = the first tile-data byte** (`our[5]=157`,
`real[5]=8`), i.e. an early **SB(0,0)** symbol (partition / first-leaf mode / first
tx). RULED OUT by experiment (divergence byte + values are identical either way):

- the tx-policy deltas #7/#12/#13/#14/#15 — all 7 textured winners (incl. vgrad
  256² cq48 and vgrad 128² cq32, which exercise the same tx path) match;
- **#4 `top_intra_model_count_allowed` 4→3** — wired from the scaffold, tested,
  did NOT change the divergence (reverted, unverifiable in isolation);
- **#8 `intra_tx_size_search_init_depth_rect` 0→1** — probed (forced rect init
  depth to 1), divergence unchanged (byte 5, 157 vs 8), reverted;
- #9 `model_based_prune_tx_search_level` 1→0 — no-op (the port's tx mask never
  applies model-based pruning: tx_search.rs:150, so 1 and 0 behave identically).

So the culprit is a **learned-model** speed-1 delta hitting SB(0,0)'s early
symbols: **#1 `intra_cnn_based_part_prune_level` 0→2** (the CNN split-vs-nonsplit
intra partition prune, `av1/encoder/partition_strategy.c` `intra_mode_cnn_partition`
— frame-intra-gated so it runs on KEY) or **#11 `prune_2d_txfm_mode`
PRUNE_1→PRUNE_2** (the 2D tx-type NN prune, structurally off in the port).

### ISOLATION PROVEN — the culprit is #1, the intra CNN partition prune (2026-07-15)

Confirmed against the **REAL libaom CNN + DNN inference**, not by inference alone.
Added an oracle `shim_intra_cnn_partition_decision` (`rd_shim.c`, exposed as
`ref_intra_cnn_partition_decision`) that reproduces `intra_mode_cnn_partition`
verbatim over the real exported `av1_cnn_predict_img_multi_out` +
`av1_nn_predict_c` + the real static-const weights/thresholds. Test
`isolate_vgrad256_cq32_cnn_partition_prune` (encoder_gate_e2e_byte_match.rs) runs
it on SB(0,0)'s 64×64 window of the exact vgrad-256 source (65×65, replicated
top/left border per `av1_copy_and_extend_frame`). Result at the **real qindex=128**
(cq32; confirmed content-independent from the flat-256 cq32 case):

| block | logits[0] | threshold | decision |
|---|---|---|---|
| 64×64 root | −3.408 | neutral band [−4.10, 1.89] | **no-op** (RD decides) |
| 32×32 (×4) | −6.34 .. −7.22 | no_split −4.564 | **square-split DISABLED** |
| 16×16 (×16) | ≈ −16.3 | no_split −5.695 | **square-split DISABLED** |
| 8×8 (×64) | ≈ −59 | no_split −1.484 | **square-split DISABLED** |

So the CNN forbids `PARTITION_SPLIT` on every sub-block of SB(0,0) while leaving
the 64×64 root free. The port (no CNN prune) keeps its speed-0 search, which can
still SPLIT those sub-blocks → a different partition tree → the byte-5 (first
tile-data byte) divergence. This is the smallest-margin isolation: the 32×32 sits
1.77 below its threshold, well outside the DNN prec-reduce bucket (1/512). **#11
(2D tx-type prune) is ruled out** — it affects tx_type, coded after partition +
mode, and would not move the FIRST tile byte.

**Feasibility for the port:** the CNN's ONLY bitstream effect is the 4 per-block
decision FLAGS (none_disallowed / do_square_split / rect_disabled /
square_split_disabled) — the logit values never enter the stream. libaom's CNN C
and AVX2 agree only to `MSE_FLOAT_TOL=1E-6` (test/cnn_test.cc), but the DNN
`av1_nn_predict(reduce_prec=1)` quantises logits to 1/512, and these margins are
large, so a faithful C-scalar Rust port will match the AVX2 oracle's FLAGS. Byte-
exactness therefore reduces to flag-parity (verifiable against the oracle), not
bit-identical floats. **Port scope:** `av1_nn_predict`+prec_reduce (small) →
`av1_cnn_predict_img_multi_out` 5-layer conv engine (large) → weight tables (~2k
floats, extract from `partition_cnn_weights.h`) → feature assembly + thresholds +
decision → integrate into `rd_pick_partition_real` (per-64×64 CNN cache + the
gating `frame_is_intra && level && sb_size>=64 && bsize<=64 && whole_blk_in_frame`).

### CNN PORTED + WIRED — but the vgrad-256-cq32 divergence is NOT the CNN (CORRECTION, 2026-07-15)

The intra CNN partition prune is now **fully ported and bit-exact**, landed on origin/main:
- `b625539` isolation oracle, `e3315d4` DNN (`av1_nn_predict`+prec_reduce), `5fae56a` CNN
  5-layer conv engine + weights, `9071bf8` full decision (log_q + feature assembly + thresholds
  + 4 flags), `a600394` wire-in to `rd_pick_partition_real` (quad_tree_idx threading +
  `extract_intra_cnn_window` + the exact gating/order). Flags are bit-exact vs the real libaom
  AVX2 inference (`cnn_partition_decision_diff`, `cnn_partition_cnn_diff`, `cnn_partition_nn_diff`).

**The "ISOLATION PROVEN — culprit is #1 (CNN)" verdict above is SUPERSEDED.** That verdict was
based on the CNN *would-prune* on a synthetic window — not on a differential test of whether the
CNN causes a port-vs-C *divergence*. With the CNN now actually wired in, the definitive result:

- The CNN fires on vgrad-256-cq32 and (on the real encode's source window) sets
  `square_split_disabled` at every 64×64 SB root — but these flags **match C bit-exactly**, so
  port and C get the SAME CNN constraints. Wiring the CNN in left byte-5 (157 vs 8) **unchanged**.
  The CNN therefore CANNOT be the source of this divergence. **#1 is eliminated.**
- **#11 `prune_2d_txfm_mode` PRUNE_2 is also eliminated:** the intra path that consumes it
  (`prune_txk_type`) is gated on `prune_tx_type_est_rd`, which is speed≥4; `prune_tx_2D` is
  `is_inter`-only. So PRUNE_1 vs PRUNE_2 does not affect an intra KEY frame at speed 1.
- Also eliminated (all `!frame_is_intra_only`, or nonrd/speed≥4): `model_based_prune_tx`,
  `av1_ml_predict_breakout`, `av1_ml_early_term_after_split`, `av1_ml_prune_rect_partition`,
  `simple_motion_search_*`, `ml_predict_var_partitioning`.

**Actual root (localized, KB-3):** a partition-search **RD near-tie** (KB-2 class). Dumping the
port's SB(0,0) tree: the port picks **PARTITION_HORZ** (two 64×32 DC / TX_64X32 blocks); C picks
a different partition. Diverges at byte 5 and **never re-converges** (`last_common_idx=4` = last
header byte) — an early partition cascade, not a missing prune. A speed-1 RD-cost delta tips the
NONE/HORZ/VERT comparison for this content+qindex.

**Two LATENT speed-1 bugs found while isolating** (neither is this cell's cause — forcing each to
its speed-1 value leaves the 8 cells byte-identical, so no current test distinguishes them; both
recorded in KB-3 for a future threaded fix + new RECT-partition speed-1 validation cells):
`part4_prune.rs` hardcodes the 4-way DNN `LEVEL_INDEX=0` (C: `min(speed,3)`); `tx_search.rs`
`get_search_init_depth_intra_speed0` hardcodes rect init-depth 0 (C: `intra_tx_size_search
_init_depth_rect=1` at speed≥1).

### KB-3 RESOLVED — missing speed-1 `use_square_partition_only_threshold` rect-kill (2026-07-15)

Root-caused with an **isolated sibling-libaom encoder instrument** (a throwaway copy at
`/root/libaom-enc-instrument`, mirroring the decoder's `/root/libaom-dbg` playbook — NEVER the
shared `reference/libaom`). Instrumented C's `av1_rd_pick_partition` to `fprintf` per-candidate RD
at SB(0,0) 64×64, drove it with a standalone `rd_harness.c` replicating `shim_encode_av1_kf`'s
exact config, and **validated the harness bitstream byte-for-byte against the e2e real_payload**
before trusting the dump. Findings at SB(0,0):

| candidate | C rate | C dist | C rdcost | port |
|---|---|---|---|---|
| NONE | 36745 | 19456 | 7427690 | **identical** (rdmult 68796) |
| SPLIT | — | — | (matched) | **identical** |
| HORZ / VERT | **never evaluated by C** | | | port evaluated HORZ → rdcost 7058801 < NONE → picked PARTITION_HORZ |

C's NONE and SPLIT RD matched the port **exactly** — the only divergence was that **C never
evaluated the rectangular partitions**. C disables them via the "square-partition-only" rect kill
(`partition_search.c:5749`): `if (bsize > use_square_partition_only_threshold) {
partition_rect_allowed[HORZ] &= !has_rows; [VERT] &= !has_cols; }`. That threshold is a
framesize-DEPENDENT ALLINTRA speed feature: **sub-480p it is `BLOCK_64X64` at speed 0** (so
`bsize > 64X64` is false on a ≤64 SB — which is why speed-0 never needed it) **but drops to
`BLOCK_32X32` at speed ≥ 1**, so the 64×64 SB gets rect disabled. The port had no such kill →
evaluated HORZ → wrong pick.

**Fix** (`partition_pick.rs`): added `use_square_partition_only_threshold_allintra(speed, w, h)`
(framesize+speed dependent, transcribed from `set_allintra_speed_feature_framesize_dependent`)
and wired the rect kill into `rd_pick_partition_real` immediately after `partition_rect_allowed`
init and before the CNN prune (matching C's order). Speed-0 unaffected (threshold `BLOCK_64X64` →
no-op). vgrad-256-cq32 now byte-matches (45/45); promoted to an asserted winner. `less_rectangular
_check_level` stays 1 at speed 1 (C: `speed>=3`) — verified in the sibling, not a latent bug.

**KB-2 is a SEPARATE root** (do NOT conflate): its cell runs at cpu-used=**0**, where this fix is
a no-op. KB-2 needs its own speed-0 root-cause pass.

**Still-latent (unrelated to this cell)**: the two speed-1 bugs recorded above (`part4_prune.rs`
`LEVEL_INDEX=0`; `tx_search.rs` rect init-depth 0) remain — forcing each leaves all 14 cells
byte-identical, so no current cell distinguishes them; queued as #25.

**cpu-used=1 all-intra coverage: 14 of 14 content cells byte-identical** (all 6 flat + all 8
gentle-slope textured). Full `cargo test -p aom-encode`: 89 passed, 0 failed.

---

## Gate 2 — `--cpu-used=3` (speed-3) all-intra KEY byte-match

Ported the speed≥3 `set_allintra_speed_features_*` deltas that are LIVE on the bd8 4:2:0 all-intra
KEY path, on top of the speed-2 baseline (a8a3992). New `if speed >= 3` block in
`speed_features.rs` (asserted by `speed3_allintra_deltas_match_source`):

- **`chroma_intra_pruning_with_hog = 2`** (speed_features.c:454) — **the dominant LIVE delta.** Turns
  ON the CHROMA directional-mode HOG prune (OFF at speed 0/1/2). New `prune_intra_mode_with_hog_uv`
  (hog.rs): U-plane HOG on the LUMA-bsize `>> ss` dims, `× (1+ss_x)(1+ss_y)` scale, intra-frame
  threshold `thresh[1][1] = -1.2`; proven **byte-identical to C** across 900 cases
  (`prune_intra_mode_with_hog_uv_matches_c`). Wired per-block in `leaf_pick_sb_modes`
  (`partition_pick.rs`) via a local `UvLoopPolicy` whose `chroma_hog_skip_mask` the existing consumer
  (intra_uv_rd.rs:1457) already reads. Fixes **13 of the 16** 4:2:0 cq12/cq32 cells that diverged at
  speed 3 without it (anti-vacuous witness — all 16 byte-match aomenc `--cpu-used=2`).
- **`intra_pruning_with_hog = 3`** (speed_features.c:455) — luma HOG threshold `-1.2 → -0.6`
  (level-3 row). `partition_pick.rs` now derives the threshold from the level (was hardcoded -1.2).
  Byte no-op on this grid (every mono cell byte-matches).
- **`less_rectangular_check_level = 2`** (speed_features.c:444) — already-wired consumer
  (partition_pick.rs:1844, `level == 2 || idx <= 2`); just threaded the sf value. Byte no-op here.
- **`ml_4_partition_search_level_index = 3`** (speed_features.c:271) — the port's `part4_prune`
  already returns unchanged at level ≥ 3 (C's different-model branch, #10). `prune_palette_search
  _level = 2` carried for faithfulness. All other speed-3 deltas verified INERT on this path
  (inter/motion, loop-restoration OFF, `max_intra_bsize`/breakout INTER-only, `use_skip_flag
  _prediction` vestigial, palette needs screen-content) — documented in `set_allintra`.

**cpu-used=3 all-intra coverage: 61 of 64 cells byte-identical** ({64,128}² × cq{12,32,48,63} ×
{flat,two-tone,vgrad,diag} × {mono,420}). The 3 residuals (two-tone 64²/128², vgrad 128², all
**cq12 4:2:0**) are a **KB-6-class latent chroma-RD partition near-tie** exposed (not caused) by the
correct chroma HOG — localized to a SB-root partition flip (real BLOCK_64X64 NONE vs port split),
pinned exactly in the gate (fails if a KB-6 fix makes one match → promote, or a regression breaks a
new cell). See KB-7. New gate `encoder_gate_speed3_textured_allintra`; full `cargo test -p
aom-encode` green.

## Gate 2 — `--cpu-used=4` (speed-4) all-intra KEY byte-match (62/64 after the speed-5 landing's override fix; LUMA BYTE-EXACT)

**Every documented speed-4 `set_allintra` delta is now ported + live** (the KB-8 chunk series —
see CLAUDE.md KB-8 for the commit-by-commit list): the SATD trellis-skip, the stage-aware
MODE_EVAL/WINNER_MODE_EVAL policies, the LUMA winner-mode two-pass (MODE_EVAL loop at
USE_LARGESTALL + default-tx-type → top-3 `store_winner_mode_stats` → WINNER_MODE_EVAL re-eval at
USE_FULL_RD), the est-rd tx-type prune (`av1_cost_coeffs_txb_laplacian`, REAL-C differential) +
txk_map reorder, `use_rd_based_breakout_for_intra_tx_search` (also speed≥3; gate re-verified
61/64), and both `prune_ext_part_using_split_info` prunes (AB evaluate at level 2, 4-way rect-win
at level 1, via `split_part_rect_win` threading). **59 of 64 cells byte-identical** vs real aomenc
`--cpu-used=4` ({64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,420}) — **all 32
mono cells match: the speed-4 luma path is byte-exact.** The 5 pinned residuals are 4:2:0
KB-7-class chroma low-q near-ties (same content cells as KB-7's speed-3 pins) — tracked under
KB-7, not a speed-4 gap. Gate `encoder_gate_speed4_textured_allintra` pins them exactly.
*(UPDATE, speed-5 landing: 3 of those 5 pins were actually the missing `speed_features.c:608-615`
chroma-HOG override, not KB-7 ties — fixed, gate now **62/64** with a 2-cell KB-7 pin; see the
cpu-used=5 section below + CLAUDE.md KB-8/KB-9.)*

Earlier partial landing (51/64) below for history:

- **`prune_chroma_modes_using_luma_winner = 1`** (speed_features.c:480) — prunes any chroma `uv_mode`
  not in `av1_derived_chroma_intra_mode_used_flag[luma_winner_mode]` (intra_mode_search.c:939-941).
  The consumer already existed (intra_uv_rd.rs:1497); wired per-block via `cfg.speed >= 4` in
  `partition_pick.rs` + a new `prune_chroma_modes_using_luma_winner` sf field
  (`speed4_allintra_deltas_match_source`). Fixes **10 cells** (incl. resolving KB-7's `two-tone
  128x128 420 cq12`).
- **`lpf_pick = LPF_PICK_FROM_FULL_IMAGE_NON_DUAL`** (speed_features.c:496) — the Y loop-filter-level
  search drops the two single-direction refine passes, leaving both luma levels at the combined
  `dir=2` winner (picklpf.c:376). Wired via a `non_dual` flag on `lf_search::pick_filter_level`.
  Fixes **6 cells** whose divergence was a header-only LF byte (tile data already byte-identical:
  `vgrad 128x128 cq32/48`, `diag 128x128 cq63`, each mono + 420).

**Anti-vacuous witness:** both deltas are load-bearing — reverting either re-diverges its cells, and
the gate asserts them byte-identical. **13 residuals pinned (KB-8):** 10 need the LUMA winner-mode
two-pass (`multi_winner_mode_type`/`enable_winner_mode_for_*` + `perform_coeff_opt=5`, which needs the
unported SATD trellis-skip at tx_search.rs:664) — every cell whose mono sibling also diverges; 3 are
pure-chroma cq12 (unported chroma DEFAULT_EVAL coeff-opt-5). `diag 64x64 cq63` byte-matches at speed 4
(NOT pinned). Full `cargo test -p aom-encode` green.

## KB-6 — partial-SB (frame-edge) encode: 196×196 chunk series (CHUNK 0-3) + first full partial-SB byte match

Landed `3167800` (CHUNK 0+1), `7c468ee` (CHUNK 2), `4b8b1f1` (CHUNK 3) — the KB-6 frame-edge
series for non-64-aligned frames (196 = 3·64+4: partial right-column + bottom-row superblocks).

- **CHUNK 0 — true-frame harness.** `run_case`/`localize_real` now encode the REAL 196×196: ceil
  `n_sb`, size + replicate-border-extend the source planes to the SB-aligned extent (matching
  `aom_extend_frame_borders` — C reads full-extent blocks from the border-extended source and
  measures distortion only over the visible area). Kills the silent floor-to-192 wrong-size skip
  and the edge-SB `perpixel_variance_y` overrun panic.
- **CHUNK 1 — luma visible-distortion clip.** `dist_block_px_domain` (tx_search.rs) +
  `search_tx_type_intra` measure SSE over `get_txb_visible_dimensions` (rdopt_utils.h:307) instead
  of full tx dims at edge txbs.
- **CHUNK 2 — chroma visible clips (with subsampling).** New `max_block_units` =
  `max_block_wide/high` (av1_common_int.h:1567/1581) helper; clipped loops + per-txb visible dims
  in `txfm_rd_in_plane_uv`/`intra_model_rd_uv` (intra_uv_rd.rs), `encode_intra_block_plane_uv`
  (encode_intra.rs), and the UV entropy stamp (encode_sb.rs). Took the top 3 SB rows INCLUDING
  right-edge partial SBs byte-exact at cq32 (was a chroma angle_delta flip at mi(32,48)).
- **CHUNK 3 — edge partition-cost override.** `set_partition_cost_for_edge_blk`
  (partition_search.c:3411) ported: gathered 2-way HORZ/SPLIT (bottom) / VERT/SPLIT (right) /
  forced-SPLIT (corner) cost via `partition_gather_vert/horz_alike` + `cost_tokens_from_cdf`,
  all-others `av1_cost_symbol(0)`; new `PickFrameCfg::partition_cdfs` raw-CDF plumbing (SB-adapted
  in pack.rs). Witness `edge_cost_matches_gathered_composition` locks the composition; primitives
  already byte-exact vs C.

**Result: `av1-1-b8-01-size-196x196 420 cq5` is a TRUE END-TO-END BYTE MATCH — the first full
partial-SB real-content frame match — promoted to an asserted gate in
`encoder_gate_real_image_e2e_kb6_repro` (KB-6 map now 24/30).** Residual: 196 cq12/20/32/48/63
diverge on a bottom-edge-SB-row RD near-tie (real 16×8 vs port 8×4 at mi(48,0); cq12 matches
through byte 2224/2230) — pinned assert-diverge in `kb6_characterize_196_partial_sb`; next step is
the sibling-C per-candidate RD dump at that node. Full `cargo test -p aom-encode` green (61 result
lines, 0 failed).

## KB-6 — frame-edge entropy-stamp tail-zero + frame-init edge partition CDF: 196² cq12/20/32 byte-match (map 29/30)

Pinned via a full C-vs-port symbol-level bit trace (throwaway instrumented sibling C encoder at
`/root/kb6-edge-instr`: `partition_search.c` + `bitstream.c` + `encodetxb.c` override objects,
oracle-flag compile, byte-gate `FRAME_PAYLOAD == 1006` verified vs real aomenc before every dump).
The prior "mi(48,0) 16×8-vs-8×4 over-split near-tie" hypothesis was WRONG in an instructive way:
the port's SEARCH picks C's exact partition tree at every bottom-edge node and every leaf RD
matches C to the unit (SPLIT 4155584 vs HORZ 2954657 at the 16×16 — not even close); the PACK also
writes the identical symbols (p/ctx/cdf-row/eob traces equal). The "over-split" was the port's own
conformance-proven decoder reading the port's DESYNCED bytes.

Two measured write-side defects (port X vs C Y):

- **Entropy-stamp footprint at frame-edge txbs.** C `av1_set_entropy_contexts` (blockd.c:29,
  cul passed via encodetxb.c:730/867): visible part of the txb's above/left footprint = cul,
  **beyond-visible tail = memset 0** (both the mb_to_right/above and mb_to_bottom/left arms).
  The port's persistent tile stamp (encode_sb.rs luma + UV) wrote the cul across the FULL
  footprint. Measured phantom: `above_ectx[mi cols 50,51] = 15`, `above_ectx[1][chroma col 25]
  = 19` (all out-of-frame; C keeps them 0). SB(32,48)'s 32×64 leaf reads its skip ctx over the
  full 8-unit width → port txb_skip_ctx 3 vs C 1 (luma), 9 vs 8 (U) → identical all-skip symbols
  coded on different-probability rows → port leaf 10 bits vs C 7 (tells 7792→7802 vs 7792→7799)
  → first divergent tile byte 975 → bottom-row decode desync. Fix: tail-zero in both encode_sb.rs
  stamps (`vis_w/vis_h` clamp; interior txbs unchanged — vis == full width). The C encode-path
  per-txb stamp `av1_set_txb_context` (encodemb.h) is full-footprint and the port's local ta/tl
  stamps already mirror it correctly (no change).
- **Edge partition-cost CDF source.** C `set_partition_cost_for_edge_blk`
  (partition_search.c:3415) gathers from `cm->fc->partition_cdf` — the FRAME-INIT table
  (measured: C's gather rows == `default_partition_cdf` at every bottom-edge node) — while the
  port's CHUNK 3 wiring fed the per-SB-adapted snapshot (pack.rs `sb_pick_cfg`). Corrects my own
  CHUNK 3 (`4b8b1f1`): the search-side edge costs were 380/669 vs C's 248/927 at the 16×16 —
  didn't flip this cell's tree but is a genuine divergence that can flip true near-ties. Fix:
  drop the override; the `..*pick_cfg` spread supplies the frame-init table.

Measured result: 196×196 cq12, cq20, cq32 all TRUE END-TO-END BYTE MATCH; **KB-6 map 26/30 →
29/30**. Promotions: 196² cq5/12/20/32/63 asserted in `encoder_gate_real_image_e2e_kb6_repro`;
`kb6_characterize_196_partial_sb` re-pinned at the one remaining cell — 196² cq48, which diverges
at tile-byte 253/616 (MID-stream, SB rows 1-2; a distinct near-tie, not the bottom-row class).
Full `cargo test -p aom-encode` green before landing.

## KB-6 — pack write-ctx source (tokenize vs trellis): 196² cq48 byte-match — REAL-CONTENT MAP 30/30 ✅

The LAST real-content cell. The inherited "which eval lands in the winner tx_type_map"
localization was REFUTED by pass-context markers on both sides (port KB6P_PASS/RD/WIN/PICKED/WALK
dumps + C KB4-CAND/WIN and KB6E_WSKIP/WTT in the byte-gate-verified `/root/kb6-edge-instr`
sibling, FRAME_PAYLOAD==616): at the divergent leaf mi(0,48) 32×64 VERT-sub0 the port's search was
ALREADY C-identical end-to-end — same 8 (mode,delta) passes reaching the TX_16X16 depth (pass 3 =
SMOOTH wins), same per-pass tx-type winners, same improvement stores, same picked map origins
[0,2,2,1], and BOTH OUTPUT_ENABLED walks requantize txb4 to (tt1, eob37) = C's coded value. The
decoded-port "txb4 = (eob4, tt2)" was the port decoder faithfully reading the port's OWN
wrong-probability bits — a desync artifact (same class as the 29/30 landing's "over-split").

**Measured write-side defect (the one and only input mismatch at the leaf):** the port wrote txb
blk(8,0)'s DC-sign symbol on `dc_sign_cdf[0][0]`; C writes it on `dc_sign_cdf[0][1]`
(KB6E_WSKIP: C cached ectx_raw=20 → skip 4, dcs 1; port walk dump: tsc 4, dsc 0). All other txbs'
(tsc, dsc) pairs matched. Bits diverge at tile byte ~253 with IDENTICAL symbols everywhere.

**Root cause — C derives the pack's write ctx in the TOKENIZE walk, not the encode walk:**
`av1_update_and_record_txb_context` (encodetxb.c, OUTPUT_ENABLED arm) computes
`(txb_skip_ctx, dc_sign_ctx)` from the PERSISTENT `pd->above/left_entropy_context` arrays — whose
within-leaf state carries the earlier txbs' edge-CLIPPED `av1_set_entropy_contexts` stamps — and
caches the pair in `cb_coef_buff->entropy_ctx` (`skip | dcs<<4`, dcs only when `tcoeff[0] != 0`);
`av1_write_coeffs_txb` writes with the CACHED pair and never re-derives. The TRELLIS meanwhile
uses the encode walk's LOCAL ta/tl with FULL-footprint `av1_set_txb_context` stamps
(encode_block_intra, encodemb.c). The port derived one pair (the trellis one) and fed it to both
consumers. Interior txbs: identical (no clip). Edge txbs: `txb_skip_ctx` is OR-based (tail-zero
inert — why the 29/30 landing sufficed there) but `dc_sign_ctx` is SIGN-OF-SUM — the tail-zero
REMOVES terms: above = txb blk(4,0)'s stamp cell 19 (cul3 | pos<<3): C reads [19,19,0,0] → +2,
port read [19,19,19,19] → +4; left rows 8..12 sum −4 ⇒ C −2 → ctx 1, port 0 → ctx 0.

**Fix (encode_sb.rs, `encode_b_intra_dry` Step 4 — the tokenize-equivalent stamp loop):** derive
`(tok_tsc, tok_dsc) = get_txb_ctx(...)` from the persistent `state.above_ectx/left_ectx` at the
TOP of each per-txb iteration (before that txb's clipped stamp — exactly C's tokenize read point)
and overwrite the cached `TxbEncode.txb_skip_ctx/dc_sign_ctx` (dcs gated on `qcoeff[0] != 0`, all
three planes). Sole consumer of the pair is `pack_plane_coeffs` (grep-verified); the trellis pair
inside `xform_quant_optimize` is untouched. Interior txbs derive identical values — structurally
zero-diff on the previously-green corpus.

Measured result: **196×196 cq48 TRUE END-TO-END BYTE MATCH — the real-content map is 30/30**.
Promotions: cq48 added to the asserted 196² cells in `encoder_gate_real_image_e2e_kb6_repro` and
the repro's assert-diverge tail replaced with `report_and_assert` over ALL 30 cells (full
byte-match gate); `kb6_characterize_196_partial_sb` flipped to assert the cell KEEPS matching.
Full `cargo test -p aom-encode --no-fail-fast` green before landing.

## KB-7 CLOSED — speed-3 AND speed-4 gates at FULL 64/64 byte-identity (2026-07-16, encoder track)

The 8 pinned cq12/cq32 4:2:0 cells (3 at `--cpu-used=3`, 5 at `--cpu-used=4`) were NOT a latent
chroma-RD arithmetic near-tie: the sibling-C RD dump (throwaway instrumented
partition_search/rdopt/intra_mode_search/partition_strategy, byte-inert vs the clean build)
showed EVERY leaf RD — the 64×64 root NONE (rate 2612035 / dist 150041 / rdcost 58839818),
each 32×32 child NONE, every HORZ/VERT rect leaf, luma AND chroma parts — matching the port
**to the unit**. The flips were two unported speed-feature gates:

1. **Level-3 OLD-model 4-way ML prune (speed>=3; closed the 3 speed-3 pins).**
   `ml_model_index = (ml_4_partition_search_level_index < 3)` (partition_strategy.c:1359) — at
   level 3 C runs the old `av1_4_partition_nn_*` net (LABEL_SIZE=4) on UNNORMALIZED features and
   decides via `int_score = (int)(100·score)`, `thresh = max − {500,500,200}`, zero-then-set
   from label bits (:1472-1497). Measured on `two-tone 64² cq12 cpu3`: scores [530,−348,0,−392]
   thresh 30 → HORZ_4/VERT_4 BOTH pruned at every 32×32 node. The port's `level_index >= 3`
   no-op guard let it search HORZ_4 (child-0: 12.9M < NONE 16.5M) → SB root flipped NONE→SPLIT.
   Fix: `OLD_*` tables transcribed (`xtask/transcribe_part4_nn.py`) + the old branch in
   `part4_prune.rs` (normalize skipped, C's overwrite-from-zero semantics; caller re-ANDs only
   the interior frame-fit envelope guard). Plus the missing `av1_nn_output_prec_reduce`
   (reduce_prec=1 at BOTH 4-way call sites; C's `+0.5` is double) — also added to the same-shape
   gap in `ab_nn_prune.rs` (:1296 is reduce_prec=1 too). Witness: `part4_old_nn_diff.rs`, 4000
   random-input decisions == a REAL `av1_nn_predict_c` oracle on the same tables.

2. **Chroma-HOG force-disable tail (speed>=4; closed the 5 speed-4 pins).** The unconditional
   tail of `set_allintra_speed_features_framesize_independent` (speed_features.c:608-616) zeroes
   `chroma_intra_pruning_with_hog` whenever `prune_chroma_modes_using_luma_winner` is on
   (allintra speed>=4). Measured: the instrumented C computes ZERO chroma-HOG masks at cpu4.
   The port kept the HOG live → HOG-pruned UV_V_PRED where C picks it (root NONE: C uv=V
   58469617 vs port uv=SMOOTH 58779332) → different chroma bytes. Fix: the tail in
   `SpeedFeatures::set_allintra` + the `chroma_hog_level` gate in `partition_pick.rs`, with the
   `UvLoopPolicy` build threading the luma-winner prune independently of the HOG mask.

Promotions: `encoder_gate_speed3_textured_allintra` and `encoder_gate_speed4_textured_allintra`
now assert FULL 64/64 byte-identity (pinned-residual lists removed); new asserted single-cell
witnesses `kb7_rd_localize.rs` (cpu3 + cpu4 two-tone 64² cq12, decode-both diff on failure);
`speed4_allintra_deltas_match_source` corrected to the C value (`chroma_intra_pruning_with_hog
== 0` at speed 4). Full `cargo test -p aom-encode --no-fail-fast` = 149 passed / 0 failed,
rebased over 57d5ce0 (KB-6 30/30).

## Gate 2 — `--cpu-used=5` (speed-5) all-intra KEY byte-match (64/64 — FULL GRID)

**Every speed-5 `set_allintra` delta is ported + live — 64/64 cells byte-identical** vs real
aomenc `--cpu-used=5` ({64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,420}),
asserted by `encoder_gate_speed5_textured_allintra`. Zero pinned residuals — the first speed
level since 0 with a fully-clean textured grid. Speed 5 is a re-parameterization of the KB-8
speed-4 machinery (no new search subsystems); full delta table + inert list in CLAUDE.md KB-9.

- **`multi_winner_mode_type = MULTI_WINNER_MODE_FAST`** (speed_features.c:524): the luma
  winner-mode two-pass keeps the top-**2** winners (speed 4: top-3) — flows through the existing
  `WinnerModeCfg::max_winner_count` (`winner_mode_count_allowed` rdopt_utils.h:236). Bisect
  witness: flips `two-tone 64² mono cq63` + `420 cq63` (mono ⇒ luma-side).
- **`ext_partition_eval_thresh`** (:510-511 + qindex-dep :2947-2962 `aggr = AOMMIN(4, speed-2)`
  == 3): BLOCK_128X128 unconditionally on sub-480p frames → `bsize > thresh` never holds →
  **AB + 4-way partitions never evaluated** at speed 5 on this grid. New
  `ext_partition_eval_thresh_allintra_key` (partition_pick.rs) feeds BOTH C consumer sites
  (`allow_ab_partition_search` partition_search.c:4005, `prune_4_way_partition_search` :4136;
  raw-enum bsize compare now matches C literally). ≥480p KEY: `screen ? BLOCK_8X8 :
  BLOCK_16X16`; speed>=6: BLOCK_128X128 everywhere; the other qindex arms are dead on KEY
  (boosted / intra-only) — unit-locked by `thresh_matches_source_arms`. Bisect witness: flips
  `two-tone 64² 420 cq12` + `vgrad 128² 420 cq12` (the former speed-4 KB-7 pins — closed there
  by KB-7's NN root; at speed 5 the disable removes those candidates before any NN runs).
- **`chroma_intra_pruning_with_hog = 3`** (:515) — transcribed, then zeroed by the **:608-616
  final override** (`prune_chroma_modes_using_luma_winner` on since speed 4 ⇒ chroma HOG OFF).
  The missing override was found INDEPENDENTLY by this landing and by the KB-7 landing (its
  speed-4 second root — see the KB-7 CLOSED section above); both model it identically (the
  `set_allintra` tail + the `chroma_hog_level` gate), and the merged tree carries one copy.
  Wiring gotcha (re-verified here): `prune_chroma_modes_using_luma_winner` must thread into
  `UvLoopPolicy` independently of the HOG mask — it originally piggybacked on the mask's
  branch, so zeroing HOG alone silently disabled the luma-winner prune (24-cell chroma
  breakage caught during this landing) until the threading was split.
- **`intra_cnn_based_part_prune_level`**: screen arm 0 → 1 (:512-513; non-screen stays 2) —
  wired through the existing CNN prune (level-1 `none_disallowed` exemption already modeled);
  byte-inert on the non-screen grid.
- **Verified inert** (CLAUDE.md KB-9 list): simple-motion LVL5, `use_coarse_filter_level_search=0`
  (already default), wiener/sgr disables, mesh-prune, qindex-dep winner_mode_tx_type_pruning=3
  (`!intra`), prune_sub_8x8 (=0 no-op at 5), rect_partition_eval_thresh (`!boosted`); no
  framesize-dependent speed-5 block; LF stays NON_DUAL.

**Anti-vacuous witness (asserted):** `encoder_gate_speed5_vs_speed4_sf_witness` — port with
SPEED-4 features vs aomenc `--cpu-used=5` DIVERGES (two-tone 64² mono+420 cq63); with speed-5
features the same cells byte-match. Full-grid cross-check measured 60/64 at speed-4-features vs
64/64 at speed-5 (4 load-bearing cells). Full `cargo test --workspace` green before landing.

**cpu-used sweep state: speeds 0-5 done** (0: baseline everywhere; 1: 14/14; 2: 16/16;
3: 64/64 + 4: 64/64 — both closed by the KB-7 landing; 5: 64/64). Next: speed 6 — NOT a pure re-parameterization
(new machinery: LPF_PICK_FROM_Q, `prune_intra_tx_depths_using_nn`, `dc_blk_pred_level`,
`prune_luma_odd_delta_angles_in_intra`, `adapt_top_model_rd_count_using_neighbors`,
`multi_winner_mode_type=OFF`, partition prunes :537-546 incl. `default_max_partition_size=
BLOCK_32X32`, `use_square_partition_only_threshold=BLOCK_16X16`) — prep facts in CLAUDE.md KB-9.
Speed-6 chunk 1 building block LANDED with this landing: `pick_filter_level_from_q`
(lf_search.rs) — the LPF_PICK_FROM_Q closed-form KEY-frame LF derivation (picklpf.c:266-330),
true-oracle-validated against REAL `aomenc --cpu-used=6` header LF levels across cq{5,12,32,48,63}
× {mono,420} incl. both clamp extremes (`speed6_prep_lf_from_q_matches_real_aomenc`); no caller
below speed 6 (byte-inert for speeds 0-5), the speed-6 flip wires it like `non_dual` was for 4/5.

## Gate 2 — `--cpu-used=6` (speed-6) all-intra KEY byte-match (64/64 — FULL CANON GRID) (2026-07-16, encoder track)

**Every speed-6 delta is ported + live — 64/64 cells byte-identical** vs real aomenc
`--cpu-used=6` ({64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,420},
`encoder_gate_speed6_textured_allintra`) + the asserted anti-vacuous witness
(`encoder_gate_speed6_vs_speed5_sf_witness`: port with FULL speed-5 features vs cpu-6 diverges
on `vgrad 64² cq32` mono+420; speed-6 features match). Landed machinery (full delta table +
per-feature bisect evidence in CLAUDE.md KB-10): the LPF_PICK_FROM_Q harness flip; the partition
prune set (`default_max_partition_size=BLOCK_32X32` square-split-forced root,
`use_square_partition_only_threshold=BLOCK_16X16`, ext-partition thresh BLOCK_128X128 for all
sizes, qidx rect prune, 4x4-var-deviation rect prune arm 2, none-pred-mode rect prune + the new
`ModeGrid::bsizes` neighbour-bsize stamps, sub-8x8 neighbour prune) — chunks 2+3 took the
baseline map to 54/64; the intra-loop set (`top_intra_model_count_allowed=2` +
neighbour-adaptive slot, odd-delta-angle reorder+prune, HOG level 4, `prune_filter_intra_level=
2`); `predict_dc_only_block` skip prediction (DEFAULT+MODE stages; the block-origin
zero_blk_rate ctx quirk; flips 4 cells incl. mono => luma-side); the 8x8 NN tx-depth prune
(transcribed weights + `intra_tx_nn_diff` 4000/4000 vs REAL `av1_nn_predict_c`; byte-inert on
the canon grid — 0 fires measured — LIVE with 96 Split verdicts on the noise extension);
the winner-mode OFF restructure (no stats stored, single best_mbmi re-eval, source-variance
bypass level 1); chroma narrowing (`cfl_search_range=1` — flips 8 4:2:0 cells;
`prune_smooth_intra_mode_for_chroma` carried, consumer-wired, unreached on current grids);
`perform_coeff_opt=6` + `tx_domain_dist_level=3` (winner pass moves to tx-domain distortion) +
`winner_mode_tx_type_pruning=3`/`prune_tx_type_est_rd=0`.

**Coverage extension** `encoder_gate_speed6_noise_flatuv_allintra` (noise luma + flat uv, 64² x
cq{32,48,63} x {mono,420}): cq32/48 hard-asserted byte-match (the NN's live cells); **cq63
mono+420 PINNED OPEN** — a KB-2-class multi-feature near-tie at qindex 255 (winner-pass tx-size
sweep picks TX_16X16 over TX_32X32 by 0.19% where real keeps 32; search partition tree matches
real exactly; no single-feature revert closes it; asserted-diverge, fails on match -> promote).
Next step: sibling-C RD dump of the (mi 8,0) winner sweep. Speeds 0-5 gates re-verified
byte-unchanged; full `cargo test -p aom-encode` green before landing.

**cpu-used sweep state at this landing: speeds 0-6 done** (6: 64/64 canon + the pinned-open
noise-cq63 pair). Speed 7 landed next (see the Gate-2 cpu-used=7 section below) — `default_min_partition_size = BLOCK_8X8` + `partition_search_type =
VAR_BASED_PARTITION` + `cdef_pick_method = CDEF_PICK_FROM_Q` (CDEF off in allintra => inert) +
`rt_sf.mode_search_skip_flags |= FLAG_SKIP_INTRA_DIRMISMATCH` + `rt_sf.var_part_split_threshold_
shift = 7` (speed_features.c:566-575). VERIFIED walk structure (encodeframe.c:848-895
`encode_rd_sb`): at allintra speed 7 (use_nonrd_pick_mode stays 0 until speed 8) the RD sb
encoder takes the VAR_BASED_PARTITION arm — `av1_choose_var_based_partitioning` FIXES the
partition tree per SB, then **`av1_rd_use_partition`** walks the fixed tree with the normal RD
intra mode search per block (a new fixed-tree walk fn; the leaf machinery reuses the existing
pick path). The nonrd PICKMODE (`encode_nonrd_sb`/`av1_nonrd_use_partition`) only starts at
speed 8 (`use_nonrd_pick_mode = 1`). So speed 7 = port av1_choose_var_based_partitioning (+
var_part_split_threshold_shift consumer) + av1_rd_use_partition; NOT a prune
re-parameterization.

## Gate 2 — `--cpu-used=7` (speed-7) all-intra KEY byte-match (64/64 — FULL CANON GRID) (2026-07-17, encoder track)

**Every speed-7 delta is ported + live — 64/64 cells byte-identical** vs real aomenc
`--cpu-used=7` ({64,128}² × cq{12,32,48,63} × {flat,two-tone,vgrad,diag} × {mono,420},
`encoder_gate_speed7_textured_allintra`) + the asserted anti-vacuous witness
(`encoder_gate_speed7_vs_speed6_sf_witness`: port with FULL speed-6 features vs cpu-7 diverges
on `vgrad 64² cq32` mono+420; speed-7 features match). Speed 7 is the STRUCTURAL flip the KB-10
prep note predicted (KB-11 in CLAUDE.md): `partition_search_type = VAR_BASED_PARTITION`
replaces the RD partition search with a variance-threshold FIXED tree + an RD mode-search walk
over it — fast-allintra = the avifenc-style fast-stills path (stills-parity scope). Landed
machinery:

- **`av1_choose_var_based_partitioning` KEY arm** (`var_part.rs`): the 4x4-downsampled source
  variance tree (`aom_avg_4x4` − 128 per 4x4; kernel differentially locked 4000/4000 vs the
  REAL exported `aom_avg_4x4_c`, `avg_4x4_diff.rs`), `set_vbp_thresholds_key_frame`
  (`base = 120 * av1_ac_quant_QTX(qindex)`; <720p: t2=base/3, t3=base>>1;
  `force_large_partition_blocks_intra == 0` on this envelope so the speed-7
  `var_part_split_threshold_shift = 7` is byte-INERT), the stage-2 force-split walk (16x16 >
  t3 / 32x32 > t2 propagate up; 64x64+ always split on KEY), and the `set_vt_partitioning`
  assignment (NONE under threshold; edge-fit VERT/HORZ pairs incl. the sb64 boundary +1 tricks;
  8x8 leaf floor) stamped as mi-grid bsizes. `get_partition_from_stamps` = C's `get_partition`
  read-back (full ext-partition disambiguation transcribed). Unit locks: threshold arithmetic,
  flat→four-NONE-32 shape, interior-edge force-split, 48x48 frame-edge single-strip rect stamps.
- **`av1_rd_use_partition`** (`partition_pick.rs::rd_use_partition_real`): the fixed-tree walk
  driving the EXISTING `leaf_pick_sb_modes` machinery (`use_nonrd_pick_mode` stays 0 until
  speed 8 — the leaf mode search is the full RD one) with C's exact context shape: HORZ/VERT
  mid-strip `encode_b_intra_dry` propagation, SPLIT recursion with `do_recon = i != 3`,
  per-node save/restore + the root OUTPUT_ENABLED winner walk. Structurally DEAD arms verified
  + documented, not ported: the NONE/split-of-NONEs re-evaluations (both gated on
  `adjust_var_based_rd_partitioning`, which is 0 outside REALTIME — the walk is a pure replay
  on this envelope) and the ALLINTRA SB rdmult modifier (only `av1_rd_pick_partition`'s root
  recomputes it; the VBP path leaves the per-SB reset 128 ⇒ identity — `pack_tile` skips the
  fold at speed >= 7).
- **sf deltas** (`speed7_allintra_deltas_match_source`): `partition_search_type =
  VAR_BASED_PARTITION` (:571) + `default_min_partition_size = BLOCK_8X8` (:570,
  assertion-only — the KEY tree never stamps below 8x8). INERT (verified vs source):
  `cdef_pick_method = CDEF_PICK_FROM_Q` (:572, CDEF off in allintra),
  `mode_search_skip_flags |= FLAG_SKIP_INTRA_DIRMISMATCH` (:573 — sole consumer is the
  INTER-frame intra search, rdopt.c:5824), `var_part_split_threshold_shift = 7` (:574, dead
  while force_large_partition_blocks_intra == 0).

**Coverage extension** `encoder_gate_speed7_noise_flatuv_allintra` (noise luma + flat uv, 64² ×
cq{12,32,48,63} × {mono,420}): cq12/32/48 hard-asserted byte-match — noise forces the variance
tree DEEP (8x8 leaves, ~1560-byte tiles) so the walk's full recursion + do_recon semantics are
byte-proven; **cq63 mono+420 PINNED OPEN** — the SAME two cells and (localized:
`kb11_speed7_noise_localize.rs`, decode-both + intended-winner dump) the SAME root as KB-10's
pinned-open speed-6 near-tie: decoded partition trees IDENTICAL (SPLIT + four NONE-32s), every
mode record matches, and the port's (mi 8,0) leaf carries tx_size TX_16X16 where real keeps
TX_32X32 — the KB-10 winner-pass 0.19% near-tie verbatim (same leaf machinery at both speeds;
the tx-plan difference desyncs the LARGEST-tx parse). Asserted-diverge, fails on match →
promote; KB-10's next step (sibling-C RD dump at (8,0)) closes both speeds at once. Speeds 0-6
gates re-verified byte-unchanged; full `cargo test -p aom-encode` green before landing.

**cpu-used sweep state: speeds 0-7 done** (7: 64/64 canon + the KB-10-twin pinned-open
noise-cq63 pair). Next: speed 8 — the nonrd PICKMODE flip (`use_nonrd_pick_mode = 1`,
speed_features.c:577-591) — KB-12 prep facts in CLAUDE.md: `encode_nonrd_sb`
(encodeframe.c:581-663) runs the SAME `av1_choose_var_based_partitioning` then
`av1_nonrd_use_partition` (partition_search.c:2960) — a SINGLE-PASS walk (pick_sb_modes_nonrd +
encode_b_nonrd per leaf immediately, dry_run=0, NO save/restore or re-encode) whose KEY-intra
leaf search is `hybrid_intra_mode_search` (partition_search.c:756): full-RD
`av1_rd_pick_intra_mode_sb` for bsize < 16x16 with source_variance >= 101
(`hybrid_intra_pickmode = 2`), else the NEW `av1_nonrd_pick_intra_mode`
(nonrd_pickmode.c:1582) with `intra_y_mode_bsize_mask_nrd` (DC only >= 32x32, DC/H/V below).
`nonrd_check_partition_merge_mode = 1` is `!frame_is_intra_only`-gated (inert on KEY);
`var_part_split_threshold_shift = 8` stays force_large-gated (inert sub-720p);
`x->source_variance` at speed 8 IS computed per SB in choose_var (`use_nonrd_pick_mode &&
source_sad_nonrd(kMedSad init) > kLowSad`, var_based_part.c:1728) — verify what
pick_sb_modes_nonrd re-derives per leaf before trusting the SB value.

## #23 QM-on forward-quant encode — byte-match gate (2026-07-16, encoder track)

**`--enable-qm=1` ALLINTRA KEY encodes byte-match real aomenc end-to-end** — the
one remaining piece of #23 (`docs/qm_rd_threading_staged.md`'s RD-search
threading) is landed and gated. QM selection now runs INSIDE every search-side
`xform_quant`/`xform_quant_optimize` (per `(tx_size, tx_type)`, mirroring
`av1_setup_qmatrix`), so the matrices shape the mode/tx/partition RD exactly as
C — not just the final coefficients.

- **Inverse table in aom-quant** (`qm_inv_tables.rs`, generated by
  `xtask/extract_qm_tables.py iwt` — byte-identical to the decoder's copy) +
  `aom_quant::iqmatrix` selector. Needed in-crate: `aom-encode` cannot depend
  on `aom-decode`'s validated copy (library-graph cycle — aom-decode depends on
  aom-encode). Diff: `qm_inv_select_diff` (912 cells vs the REAL
  `av1_qm_init` `giqmatrix` via `ref_iqm_giqmatrix`, + fwd/inv Some-ness
  agreement over the full 16x3x19x16 grid).
- **Funnel** (`aom-encode/src/lib.rs`): `QmCtx { qm_level, plane }` +
  `QuantParams::with_qm` + `resolve_qm` at the top of `xform_quant` /
  `xform_quant_optimize` — flat/explicit-slice paths byte-identical to before
  (QM-off invariant; full suite green).
- **Threading**: `qm_levels: Option<[usize; 3]>` on `PickFrameCfg` /
  `SbEncodeEnv` / `TxfmYrdEnv` / `UvRdEnv`, `qm_level` on
  `TxTypeSearchInputs` / `EncodeIntraYEnv` — every search + re-encode + pack
  quantize site carries the frame QM context (`av1_set_quantizer`'s
  `qmatrix_level_{y,u,v}`, derived via `aom_get_qmlevel_allintra` for BOTH
  planes in the allintra arm; the `_444_chroma` variant is tune=IQ-only).
- **Trellis dist-qm gating — the root cause of the one near-tie divergence**
  (localized via sibling-libaom instrumented dump, KB-2/KB-3 method, at
  qm(8,8) mono 64x64 cq32: mi(8,4) BLOCK_4X16 D203/adly3 3rd TX_4X4, real
  tt=ADST_ADST vs port DCT_DCT): C's `av1_optimize_txb` (txb_rdopt.c) selects
  the FORWARD matrix for its distortion weighting ONLY under
  `dist_metric == AOM_DIST_METRIC_QM_PSNR` (tune=IQ / SSIMULACRA2); with the
  default PSNR metric the trellis runs `qmatrix = NULL` (unweighted dist)
  while the dequant still folds `iqmatrix`. The port passed BOTH matrices →
  QM-weighted trellis distortion → a keep/drop near-tie flipped (the port
  dropped the eob coefficient qc[9] where real kept it).
  `aom_txb::optimize_txb_qm`'s `qmatrix` is now `Option<&[u8]>` (mirroring
  C's nullable qmatrix); the production QM path passes `None`, the
  explicit-slice kernel-diff path keeps `Some` (that arm models QM_PSNR and
  stays validated by `optimize_qm_diff` / the `ref_optimize_txb_qm` oracle).
- **GATE: `encoder_gate_qm_on_e2e`** (encoder_gate_chroma_ss_e2e.rs, via the
  new `run_case_ext(..., qm: Option<(qm_min, qm_max)>)` +
  `ref_encode_av1_kf_qm`): 40 cells — bd8 x {mono, 4:2:0} x {64,128}^2 x
  cq{12,32,48,63} x qm{(5,9) base default, (4,10) allintra default} + a bd10
  x (4,10) arm — every cell a TRUE end-to-end byte match. `run_case_ext`
  additionally asserts the derived `qmatrix_level_{y,u,v}` == the levels the
  real encoder signalled in the frame header (derivation witness).
  Anti-vacuous: `encoder_gate_qm_on_anti_vacuous_witness` proves C(QM-on) !=
  C(QM-off) on the gate's exact content for every (range, cq) — combined with
  the QM-off gates (port == C(off)) and this gate (port == C(on)), the port's
  QM path provably changes the port's own bytes.
- **Envelope after this chunk**: tune=IQ / tune=SSIMULACRA2 remain out
  (QM_PSNR trellis dist weighting, `aom_get_qmlevel_444_chroma`,
  `aom_get_qmlevel_luma_ssimulacra2`, chroma deltaq, sharpness=7,
  CDEF_ADAPTIVE) — the nullable-qmatrix plumbing is the entry point when
  that knob is scheduled. Multi-tile + QM composition not gated (single-tile
  only; the QM arm asserts tiles == (0,0)).

## Gate 3 — performance track OPENED: harness + first SIMD kernel (2026-07-16, perf track)

**Infrastructure (landed 057bde2 + this chunk series):**
- **`crates/aom-bench`** — the Gate-3 measurement harness: zenbench paired
  benches (`benches/gate3.rs`, c_oracle as per-group baseline so the port
  row's paired CI IS the Gate-3 ratio) + `gate3_profile` callgrind driver.
  Encode cells replicate the byte-exact e2e gates verbatim (KB-6 real-content
  recipe at speed-0 {64²,128²,196²}×cq{12,32,63} + one speed-4 synthetic
  point); decode cells = first-KEY TU of 5 conformance vectors (64², 196²,
  352×288×q{00,32,63}). EVERY cell byte-verifies port-vs-C in setup before
  timing (`assert_byte_exact`). `just bench-gate3` (quiet box only) /
  `just bench-smoke` / `just profile`.
- **`crates/aom-dispatch`** — the SIMD dispatch policy crate:
  `AOM_FORCE_SCALAR=1` (`just test-scalar`) disables every archmage token
  process-wide at first dispatch, so every `incant!` falls through to
  `_scalar` = the transcribed port, verbatim. ONE archmage universe:
  aom-dist/aom-intra migrated from the `/root/work/archmage` path dep to
  registry `archmage = "0.9.27"` (CI clone workaround removed per its TODO).
- **Profile ranking committed**: `benchmarks/gate3_profile_ranking_2026-07-16.md`
  (+ .meta) — callgrind Ir, load-tolerant. Encode (128² cq32 s0): transforms
  ~33%, txb kernels ~18%, quantize_fp ~5%, highbd_variance ~3%, alloc ~3.5%.
  Decode (352×288 q32): CDEF ~27%, wiener ~13%, inv-txfm ~12%, deblock ~9.5%,
  od_ec entropy ~6.5% (serial — not SIMD-able), txb-read ctx ~5%.
  PROVISIONAL smoke wall ratios (noisy box, directional only): encode port
  ~3.4-4.2× C at s0 cq12/32, ~7× cq63, ~9× s4; decode ~1.5× (q00) to ~8×
  (q63). Honest quiet-box zenbench baseline: PENDING (box must drain).

**SIMD kernel #1 — `av1_quantize_fp_no_qmatrix` (the speed-0 search
quantizer, ~5% of encode Ir):** `aom_quant::simd::
av1_quantize_fp_no_qmatrix_dispatch` — ONE magetypes generic i32x8 kernel
(`#[magetypes(v3, neon, wasm128, -scalar)]`), hand-written `_scalar` variant
= the transcribed port verbatim, `incant!` dispatch, `aom_dispatch::
scalar_forced()` pin at entry. Bit-exactness is argued in-module (the two
reformulated steps — the gate threshold and the i64 rounding add — are exact
integer identities on the FULL i32×i16 domain) and pinned by
`tests/quantize_fp_simd_diff.rs`: SIMD == scalar port on qcoeff/dqcoeff/eob
across 9 permutation tiers × {production + fully-adversarial tables/coeffs +
edge spikes} × 9 sizes × 3 log_scales (anti-vacuous: asserts AVX2 present on
x86 CI). Wired into the `xform_quant` Fp/flat/lowbd arm (aom-encode).

**Parity discipline (per landing, from this chunk on):** full workspace
suite 0-failed in BOTH dispatch modes (`just test` + `just test-scalar`) +
the bench harness byte-verify. Float decision helpers (av1_nn_predict, HOG,
softmax, variance-factor) stay SCALAR permanently — decision-side, cheap,
and float SIMD reassociation would shift RD decisions.

**SIMD kernel #1 measured (callgrind Ir, enc_s0_128_cq32):** whole-encode
155.21B -> 148.86B Ir (-4.1%); the quantize cluster 7.65B -> 2.25B (~3.4x),
AVX2 tier confirmed live in the profile.

**SIMD kernel #2 — `cdef_filter_block_16` (decode's #1, ~27% of the CDEF-on
vector's Ir):** `aom_cdef::simd` — width-8 row kernel (i16x8, one row/vector)
+ width-4 two-row kernel (4+4 lanes), same magetypes pattern; the dispatch
lives INSIDE `cdef_filter_block_16` so every caller (the frame walk) gets it
transparently, and `cdef_filter_block_16_scalar` is the never-routed
reference twin. Exactness: structural-domain argument in-module (pixels <=
4095 | CDEF_VERY_LARGE, header strength/damping ranges -> every i16 lane op
matches the scalar core's own narrowings, incl. the deliberate i16 sum wrap
and the tap*constrain `as i16` truncation == mullo). Pins: the PRE-EXISTING
240k-case `cdef_filter16_byte_identical` C-differential now drives the
DISPATCHING function (bw=4 and bw=8 route SIMD) against the REAL C kernels,
the 420-case `cdef_frame_matches_real_c_walk` oracle covers the frame-walk
integration, and the new `cdef_filter_simd_diff.rs` pins SIMD == scalar core
at every token permutation over the structural domain + boundary fills.
**Measured (callgrind Ir, dec_352x288_q32):** whole-decode 3.657B -> 2.807B
(-23.2%); the ~985M cdef_filter cluster collapsed to ~40M SIMD + residual
(cdef_find_dir 61M, next on the CDEF list).

**SIMD kernels #3-#5 (landed 2026-07-17, each with per-tier permutation
differentials + the pre-existing C-differentials now driving the dispatch,
each gated 0-failed in BOTH dispatch modes):**
- **#3 `wiener_convolve_add_src`** (16eb3b3): i32x8 8-column lanes, ZERO
  reformulation (the scalar port's exact i32 expressions lane-wise);
  overlap-back width tails; w<8 scalar. Decode Ir 2.807B -> 2.553B
  (**-30.2% cumulative** vs the 3.657B all-scalar baseline).
- **#4 `txb_init_levels`** (0730a68): all geometries (h>=8 column chunks;
  h=4 two-columns-per-vector); |x|.min(127) exact on the FULL i32 domain
  incl. the i32::MIN lane. Encode Ir -> 145.30B.
- **#5 `highbd_variance64`** (b164600): pixel-domain-exact row kernel (lane
  mullo wrap == scalar i32 wrap; i32 lane adds == u32 adds bitwise). Encode
  Ir -> **142.61B (cumulative -8.1%** vs 155.21B; w=4 blocks still scalar —
  a two-rows-per-vector arm is a cheap follow-up).

**NEXT (the elephant): the transform stack — full design worked out, not yet
built (record of the 2026-07-17 analysis; the next perf session should start
here):**
- **Shape:** vectorize ACROSS independent 1-D transforms. The 2-D drivers'
  COLUMN pass reads `buf[r*col_n + c..c+8]` contiguously per row when lanes =
  8 adjacent columns — NO transposes needed (fwd: the col pass is first; inv:
  the col pass is second). The ROW pass needs 8x8 in-register transposes (or
  gather-arrays first cut) — do the col pass per kernel FIRST (half the
  transform cost, zero shuffle risk), rows later.
- **Per-kernel incremental landing:** dispatch per `func_col` — blocks whose
  1-D col kernel is in the ported set take the 8-column vector path, others
  keep the scalar per-column loop. Port order by profile: idct16/iadst16
  (the workhorses), idct8/iadst8, idct32, idct4/iadst4, fdct/fadst same
  order (fwd shares the pattern), identity kernels (trivial, no butterflies)
  anytime.
- **EXACTNESS TRAP (the crux, verified against the scalar port 2026-07-17):**
  `half_btf` wraps each PRODUCT in i32 (`w.wrapping_mul(in)` — matches C's
  int multiply) but sums the two wrapped products + rounding in **i64**.
  With driver clamp bounds (row pass clamps input to (bd+8) bits = +/-2^19
  at bd12; cospi <= 2^13) a product reaches 2^32 and the SUM needs 33 bits:
  an i32-sum SIMD (libaom's own SSE4/AVX2 shape) diverges from our scalar
  port on CRAFTED-BUT-DECODABLE streams that push dequantized coefficients
  to the clamp bounds. Zero-tolerance => the SIMD must reproduce the i64
  sum exactly. (libaom's own SIMD has this divergence vs their _c — their
  corpus never hits it; OUR standard is scalar==SIMD on every decodable
  stream.)
- **Exact-i64 recipe on AVX2 (per 8 lanes):** p0 = mullo(w0, in0), p1 =
  mullo(w1, in1) (wrapped products == scalar); widen each 128-half via
  vpmovsxdq to 2x i64x4; s = p0_64 + p1_64 + rnd (vpaddq); the >> bit +
  truncate-to-i32 pair is EXACT via LOGICAL vpsrlq (bit in {12,13}) + take
  the LOW dword of each 64-lane (vpermd gather) — because
  `((v >>arith b) as i32) == low32(v >>logical b)` for any v (sign bits
  land above the truncation). ~15 vector ops per 8-lane half_btf vs ~40
  scalar ops. AVX2 lacks vpsraq (64-bit arithmetic shift) — the
  logical+truncate trick dodges it. magetypes has NO integer widening ops
  (verified) => this needs the raw-intrinsic escape (`i32x8::raw()` /
  `from_m256i`) inside a hand-written `_v3` variant (pattern C; value
  intrinsics are safe under #[arcane], #![forbid(unsafe_code)] holds).
  NEON deferred (falls to scalar; the perf gate box is x86).
- **Kernel bodies:** mechanical lane-generic rewrites of the generated
  butterfly kernels (inv_txfm1d_gen.rs / txfm1d_gen.rs) — extend
  `xtask/transpile_txfm1d.py` to ALSO emit lane-generic variants over a
  small local trait {add/sub (wrapping), half_btf, clamp_bits, neg}, whose
  scalar impl delegates to the existing helpers (so the scalar path is the
  EXISTING code) and whose v3 impl is the recipe above. clamp_value on
  lanes = clamp(splat(min), splat(max)) — trivial. round_shift_array /
  clamp_buf / the final highbd_clip_pixel_add row are separate lane-wise
  steps in the driver (easy, exact).
- **Differential plan:** per 1-D kernel, SIMD == scalar at every token
  permutation over inputs SWEEPING THE DRIVER CLAMP BOUNDS (+/-2^(bd+7),
  dense random + the exact boundary values + sign patterns engineered to
  maximize |p0+p1|) x cos_bit {12,13} x all stage_range values the drivers
  pass; plus the existing inv_txfm2d/txfm2d C-differentials (193 combos,
  ~400k comparisons) drive the dispatching 2-D entries end-to-end.

**Other remaining kernels (profile order, after transforms):** deblock
filters (~6% of decode; horizontal edges = natural 4-column lanes like the
CDEF w4 pattern; vertical edges need 4x14 transposes — do horizontal
first), get_nz_mag/get_lower_levels_ctx (encode ~9% — requires switching
the cost/optimize scan loops to the ALREADY-PORTED batch
av1_get_nz_map_contexts shape first, a call-site restructure with its own
parity gate), cdef_find_dir (decode 2.4%, shuffle-heavy partial sums),
sub_pixel_variance's bilinear filter, the w=4 arms of variance/init_levels,
and the alloc-reuse pass (malloc/memset ~4% of encode — scratch-buffer
reuse in xform_quant/tx_search, algorithmic not SIMD).

**Wall-clock status:** the committed zenbench baseline (quiet box) is STILL
PENDING — every run so far was on a loaded box (resource-gate flagged).
Instruction counts are the only committed numbers; per the no-fabrication
rule no wall ratios are claimed beyond the provisional noisy smoke in
`benchmarks/gate3_profile_ranking_2026-07-16.meta`. When the box drains:
`just bench-gate3`, commit results to `benchmarks/gate3_baseline_<date>.csv`
+ `.meta`, and compute the Gate-3 ratios from the paired CIs.

## Gate 3 — transform SIMD LANDED (2026-07-17, perf track): the full 1-D lane + 2-D pass stack, AVX2, bit-identical

The transform stack designed above is **built, verified, and measured** — the
elephant of the Gate-3 profile (transforms ~33% of encode Ir, inv-txfm ~12% of
decode). AVX2 (`X64V3`) only; the scalar path is byte-untouched and
`AOM_FORCE_SCALAR=1` routes every dispatch to it.

**What landed** (`crates/aom-transform/src/simd/`, wired into
`av1_inv_txfm2d_add` / `fwd_txfm2d_core`):
- All 24 1-D lane kernels — the idct4/8/16/32/64, iadst8/16, fdct8/16/32/64,
  fadst8/16 butterfly kernels via the transpiler's new `--lanes` mode
  (`inv1d_v3_gen.rs` / `txfm1d_v3_gen.rs`, byte-reproducible — `just gen-txfm1d`
  regenerates all four gen files identically); the hand-written iadst4 (i64-pair),
  fdct4, fadst4, i/fidentity4/8/16/32 (`hand_v3.rs`).
- The exact-i64 `half_btf` recipe (`hb`): mullo wrapped products (== scalar
  `wrapping_mul`), `vpmovsxdq` widen, i64 sum + rounding, LOGICAL `vpsrlq` +
  low-dword truncate — reproduces the scalar i32-product / i64-sum semantics
  bit-for-bit at the bd12 row-clamp bound ±2^19 where an i32-sum SIMD (libaom's
  own SSE4/AVX2 shape) would diverge. AVX2 has no `vpsraq`; the logical-shift +
  truncate identity dodges it. `#![forbid(unsafe_code)]` holds (value intrinsics
  inside archmage `#[rite]`/`#[arcane]` target_feature regions).
- Four vector passes: inv col + row, fwd col + row (8 columns/rows per batch),
  gated `col_n % 8 == 0` / `row_n % 8 == 0` (W=4/H=4 tails stay scalar). `hb` +
  `rshiftv`/`mul_rshiftv`/`shl_clamp64v`/`clampv`/`revv`/`transpose8`/`widen16`
  helpers carry the per-stage clamp/shift/rect/flip/clip-add lane-wise.

**Parity (the zero-tolerance integer rule — SIMD MUST equal scalar):**
- Per-kernel differential (`simd::tests::inv1d_v3_bit_identical_to_scalar_at_every_tier`):
  all 24 kernels, SIMD == scalar at every token permutation over the driver
  clamp domain (±2^15/17/19) + exact-bound sign patterns + full i32 + extremes,
  × cos_bit 10..=13 × stage_range {16,17,18,20}. Green in BOTH dispatch modes.
- **NEW 2-D permutation-equality integration test**
  (`tests/txfm2d_simd_perm_diff.rs`): `av1_inv_txfm2d_add` / `av1_fwd_txfm2d`
  over every valid (tx_type × tx_size × bd × input), inverse coeffs pushed to
  **±2^20** (past the ±2^16 the vs-C harness caps at, because C's `half_btf`
  overflows i32 (UB) at the true bd12 bound) + full-range i16 forward residuals,
  tight + strided dest. Every token permutation must byte-match the first;
  6 v3 + 3 scalar permutations compared, green in BOTH modes. This pins the
  PASS PLUMBING (flips/clamps/shifts/rect/transpose/clip-add) SIMD==scalar over
  the domain the vs-C harnesses can't reach.
- The pre-existing C-differentials (`inv_txfm2d_diff` 3/3, `txfm2d_diff` 2/2,
  `inv_txfm1d_diff`, `txfm1d_diff`, `fdct_diff`) now drive the SIMD passes live
  end-to-end (~400k comparisons, 193 combos × bd) — green.
- Added `fwd_kernel_n` debug_asserts (symmetry with the inv passes).

**Measured (callgrind Ir, port side; `benchmarks/gate3_transform_simd_2026-07-17.{md,meta}`):**
before = `fa1c55c` (SIMD #1–#5, no transforms), after = this stack:
- **encode** `enc_s0_128_cq32` 6it: **142.72B → 99.89B (−30.0%)** — RD search
  runs both fwd (analysis) and inv (recon) transforms per candidate.
- **decode** `dec_352x288_q32` 30it: **2.553B → 2.140B (−16.2%)** — inv only.
- Cumulative vs the all-scalar `057bde2` origin: encode −35.6%, decode −41.5%.
- AVX2 tier confirmed live in the annotate (inv/fwd col+row passes,
  run_inv1d/run_fwd1d → avx2). Baselines match the #5/#3 STATUS numbers to
  <0.1% (stable workload), so the delta is the transform SIMD.

**Follow-ups (documented, by profile):** the W=4/H=4 half-vector arms are the
largest residual transform cost in encode (`av1_fdct4` 0.80%, `av1_fadst4`
0.74%, `av1_iadst4` 0.60%, `av1_idct4` 0.53% — all scalar because the passes
gate on %8); then AVX-512 (`X64V4`, 16-lane, native `vpsraq`), then NEON (falls
to scalar — the perf gate box is x86). Design record: `HANDOFF-TXSIMD.md`.

## #7 CDEF-strength RD search — BIT-IDENTICAL, first bulk-port family lands directly in PARITY section A (2026-07-17, CDEF bulk track)

- **Scope**: the full `av1_cdef_search` (av1/encoder/pickcdef.c) for the
  `--enable-cdef=1` (CDEF_ALL) one-pass allintra KEY envelope — the first
  non-default stills knob of the bulk-port pivot. CDEF stays OFF by default;
  nothing in the proven byte-exact envelope changes (full aom-encode suite
  re-verified 20 binaries / 0 failed with the port in tree).
- **What is ported** (`aom-encode/src/pickcdef.rs`, ~800 lines vs C source,
  never assumed): the per-64x64-fb MSE grid (`av1_cdef_mse_calc_block` /
  `cdef_mse_calc_frame`) with the search-specific `av1_cdef_filter_fb` shape
  (dirinit-cached luma directions shared across all strength indices AND
  planes of one fb; the bd8 direct-`aom_sse` paths incl. the full-fb vs
  per-dlist-block split; the high-bitdepth PACKED 16-bit layout +
  `compute_cdef_dist_highbd` `>> 2*coeff_shift`); `get_cdef_filter_strengths`
  for FULL + all five FAST levels (priconv/secconv tables);
  `search_one[_dual]` + `joint_strength_search[_dual]` greedy+refinement with
  C's exact gates (mono refinement `!fast`-gated, dual unconditional); the
  0..=3 signaling-bits `RDCOST(rdmult, av1_cost_literal(sb_count*i +
  nb*6*(1|2)), tot_mse*16)` loop; per-unit best-index stamping; fast-method
  strength re-mapping. Reuses the bit-exact aom-cdef kernels (`cdef_find_dir`
  600k-diffed, `cdef_filter_block_16`) — never reimplemented. Documented
  NOT-ported (dead for this config, verified): CDEF_ADAPTIVE arms,
  PICK_FROM_Q (speed>=7 rt), rtc paths, the MT walk, the dual/quad SSE merge
  regroupings (bit-identical totals without them).
- **Two-pass architecture** (`pack_tile_from_trees`, pack.rs): C searches
  CDEF AFTER the frame encode but signals the strengths INSIDE the tile data,
  so the fused search+pack walk cannot write them. The port now mirrors C's
  real structure: phase 1 = `pack_tile` into a throwaway entropy coder (C's
  OUTPUT_ENABLED encode pass: recon + one adapted tile ctx), then LF
  derive+apply (`loop_filter_frame`, gated on nonzero luma levels —
  encoder.c:2887), then `av1_cdef_search` on (source, deblocked recon) using
  the frame-level rdmult (`loopfilter_frame` re-loads `td.mb.rdmult =
  rd.RDMULT` before the filter stage — verified, no per-SB fold leaks), then
  phase 2 = `pack_tile_from_trees` re-walks the phase-1 trees from a FRESH
  frame context (C's `av1_pack_bitstream`), interleaving `write_cdef`
  literals (CDF-free, so both passes adapt through identical symbol streams;
  per-SB `derive_real_costs(kf)` states match SB-for-SB). The per-unit
  strength grid rides `MiNbrGrid::cdef` (`CdefPackState`) into `pack_leaf`'s
  `write_cdef` — C's read of the unit-top-left mbmi (bitstream.c:909-915).
- **RESULT: 14/14 cells BIT-IDENTICAL to real aomenc on the FIRST complete
  run** — sizes equal, zensim deltas 0.000, streams byte-equal: real content
  (`av1-1-b8-01-size-196x196` cq{5,12,20,32,48,63} — cdef_bits=2,
  four-strength joint sets, per-unit literals genuinely coded;
  `av1-1-b8-01-size-64x64` cq{12,32,63}) + synthetic mono 128^2 cq{12,48}
  (single-plane search), 4:4:4 128^2 cq32, 4:2:0 128^2 cq32, bd10 4:2:0
  128^2 cq32 (16-bit MSE path). Every cell also asserts: the spliced port
  stream decodes IDENTICALLY through the real C decoder and the port decoder
  (the signalled CDEF syntax means to C what it means to us), and the
  derived damping/bits/strength sets equal the real header's.
- **GATE**: `encoder_gate_cdef_{real_content,synthetic_axes}_rd_close`
  (aom-bench/tests/encoder_gate_cdef_e2e.rs) through the shared
  `aom_bench::rd_close` harness — reports the RD table, then asserts full
  per-cell byte-identity (`assert_all_exact`; an EXACT→CLOSE slip is a
  regression and fails). PARITY.md: C1 → section A directly (rule 2's B→A
  citation in the same landing that would have created the B row).
- **Encoder-side `av1_cdef_frame` apply deliberately NOT invoked** in the
  encode pipeline: in a single-frame KEY encode nothing consumes the
  post-CDEF reconstruction (no reference frames, restoration off) and the
  coded bytes are proven complete without it (the 14/14 byte-identity).
  The bit-exact apply walk already exists (`aom_cdef::frame::cdef_frame`,
  the decoder-track landing) — wire it into the encode recon path when the
  loop-restoration search (which reads the post-CDEF recon as its input)
  lands, and for CDEF-on encoder-recon MD5 checks.
- **Follow-ups (documented, not blocking)**: FAST search levels 1..5 are
  ported + unit-tested but only FULL (speed 0) is e2e-gated — the speed>=1
  CDEF-on cells need the speed-parameterized pick_cfg derivation threaded
  into `port_encode_cdef` (the speed gates' harness shape); SB128 CDEF-on
  (>64 mbmi arms are in place, pack envelope is SB64); CDEF_ADAPTIVE /
  tune=IQ out of envelope (documented in pickcdef.rs docs).

## #29 Palette RD search (Y + UV) — RD-CLOSE landing, 5/7 cells byte-exact (2026-07-17, screen-content track)

**The first stills-parity bulk-port landing (PARITY.md section B row 1): the full
palette mode decision is now REAL in the port** — search + recon + pack — behind
`PickFrameCfg::palette_costs: Option<&PaletteCosts>` (`Some` = `--enable-palette=1`;
`None` everywhere else, keeping every pre-existing byte gate's envelope untouched
by construction).

**Ported (all against reference/libaom v3.14.1 source):**
- `crates/aom-encode/src/palette_search.rs` (new): the dim-1/dim-2 k-means kernels
  (`k_means_template.h` — `calc_indices`/`calc_centroids`/ping-pong `av1_k_means`,
  `lcg_rand16` reseeding), `av1_count_colors[_highbd]` (8-bit-domain threshold +
  full-depth counts), `find_top_colors` (priority-queue top-N), `fill_data_and_get_bounds`,
  `remove_duplicates`, `optimize_palette_colors` (cache snap, stride 1/2),
  `extend_palette_color_map`, `delta_encode_cost` + `av1_palette_color_cost_y/_uv`
  (cache-signal + delta bits, V wrap-delta vs raw), `av1_cost_color_map` (wavefront
  colour-index token rate over `get_palette_color_index_context`), `write_uniform_cost`,
  `palette_rd_y` (header-rd gating levels 0/1/2 + the `header_rd >> shift` breakout),
  `perform_top_color_palette_search` + `perform_k_means_palette_search` (ascending +
  descending sweeps, level-1 coarse two-stage via `set_stage2_params`, level-2
  no-improvement return), `av1_rd_pick_palette_intra_sby`,
  `av1_rd_pick_palette_intra_sbuv` (dim-2 (U,V) pairs, U-ascending pair sort, chroma
  `early_term_chroma_palette_size_search` break), and `chroma_block_dims`
  (`av1_get_block_dimensions` plane-1 incl. the chroma sub-8x8 +2 correction —
  palette IS reachable on 4x16/16x4 luma whose 4:2:0 chroma is sub-4).
- Cost tables: `PaletteCosts` (`palette_{y,uv}_size_cost` + `palette_{y,uv}_color_cost`)
  filled from the live CDFs in `derive_real_costs` (rd.c:136-152 slices);
  `intra_mode_info_cost_y/_uv` gained the `use_palette` branch (+ the
  `av1_filter_intra_allowed` palette_size[0]==0 gate the flag-only port hardcoded).
- Search integration: the `av1_rd_pick_palette_intra_sby` hook after the sby mode loop
  (before filter-intra, C order; palette candidates skip the ALLINTRA variance factor —
  C asymmetry preserved), winner-mode multi-winner entries carry the palette + colour
  map (speed>=4 second-pass restore), the UV hook after the sbuv loop; prediction =
  colour-map fill (`av1_predict_intra_block`'s use_palette arm) threaded through
  `pick_uniform_tx_size_type_yrd_intra`/`txfm_rd_in_plane_intra` (Y) and
  `txfm_uvrd_p`/`txfm_rd_in_plane_uv_p` (UV) as `Option<&PaletteYrd/PaletteUvPred>`.
- Neighbour state (cache + ctx): `ModeGrid` carries per-mi palette sizes+colours
  (search side, stamped at all 22 stamp sites; snapshot-free — same commit discipline
  as the uv_modes grid), `MiNbrGrid` the pack-side twin; `av1_get_palette_cache` +
  `av1_get_palette_mode_ctx` now read REAL neighbour palette state on both walks.
- Recon/pack: `encode_b_intra_dry` + `encode_intra_block_plane_y/_uv` palette map-fill
  prediction; `pack_leaf` writes the palette syntax (sizes/colours via the existing
  bit-exact `write_mb_modes_kf_fc` — its above/left palette-nbr params now live) + the
  colour-map tokens (`encode_color_map_tokens`, write_modes_b order: mbmi -> maps ->
  tx_size -> coeffs) + `kfs.mb_to_top_edge` (the cache's SB-row gate).
- Speed features: sf levels threaded per leaf (allintra speed 0: prune_palette_search=0,
  prune_luma_palette_size_search=1 (header-rd gating shift 1), chroma early-term=1;
  speed>=1: 1/2; speed>=3: 2/2 — from the already-landed SpeedFeatures fields).
- **Latent bug fixed while wiring:** the UV no-palette FLAG cost was hardcoded off
  (`UvLoopPolicy::speed0_allintra().try_palette = false`) — C costs it per leaf via
  `av1_allow_palette(sct, bsize)` regardless of `--enable-palette` (the Y-side analogue
  was KB-2-adjacent, already fixed). Now recomputed per leaf. Screen-content frames
  only; the full suite stays green (near-ties didn't flip on the existing gates).

**GATE: `rd_close_palette::palette_y_rd_close_gate`** (aom-bench, the PARITY.md rule-3
harness): 6 synthetic screen cells (few-colour text/UI, mono+4:2:0, 64²/128²,
cq12/32/63/20) + 1 real-content control, both sides `--enable-palette=1
--enable-intrabc=0` via `ref_encode_av1_kf_screen_content` /
`EncodeCell::port_encode_with(enable_palette)`. **5/7 cells BYTE-IDENTICAL to real
aomenc** (all 64² cells incl. mono + both EXACT-fastpath, and the control); the two
128² multi-SB cells are CLOSE: `ui_420_128_cq32` +2.55% size / −1.04 zensim (port
BETTER), `text_420_128_cq20` same size, +0.19 zensim. Anti-vacuous witnesses: screen
detection asserted fired per cell; C(palette-on) != C(palette-off); PORT(palette-on)
!= PORT(palette-off) — the port's search provably picks palette.

**Honest gaps:** speeds 1-5 palette sf levels are wired but only speed 0 is gate-covered;
`av1_search_palette_mode[_luma]` (inter-frame callers) out of stills scope; UV palette
+ 4:2:2 not gate-covered (machinery is subsampling-generic); palette winner-mode
interaction at speed>=4 wired per C but not gate-covered.

## TOGGLE SWEEP — C8/C9/C10 CLI-toggle byte gates (2026-07-17, encoder track)

**23 of the C8-C11 toggle-family knob arms are BYTE-IDENTICAL vs real aomenc
(same ctrl) and hard-pinned; 1 pinned-open** (`--use-intra-dct-only`;
`aom-bench/tests/toggles_rd_close.rs`, 25 tests). Every knob set runs the
witnessed 3-cell real-content grid (64² cq32
+ cq63, 128²-crop cq12) with an ANTI-VACUITY witness: the toggle must change
the C encoder's output on at least one cell, or the test panics (an EXACT
verdict on a cell the knob never reaches proves nothing).

- **Infra:** generic ctrl-pair oracle `shim_encode_av1_kf_ctrls` (dec_shim.c —
  base config identical to `shim_encode_av1_kf`, plus raw
  `(aome_enc_control_id, value)` pairs applied in caller order; every
  pre-existing shim wrapper is byte-inert NULL/0) + `ref_encode_av1_kf_ctrls` +
  the `cx_ctrl` constant module, cross-checked against the pinned v3.14.1
  headers at shim compile time (`shim_cx_ctrl_id_by_probe` +
  `cx_ctrl_ids_match_reference_headers`). Harness: `ToggleKnobs` +
  `EncodeCell::port_encode_with` / `c_encode_ctrls` (aom-bench) — one struct
  drives both sides of every cell; `Default` == the stock proven envelope
  (control test pins that byte-exactly).
- **C8 partition controls (6/6 arms EXACT):** rect=0 / ab=0 / 1to4=0 / min=16px
  / max=32px / square-only-8..32. `PickFrameCfg` already carried the fields;
  the harness now maps px→BLOCK via C's `set_max_min_partition_size`
  (`min(sf_default_max, dim_to_size, sb)` — partition_strategy.h:214; auto-max
  ML arm verified inter-only) instead of the flat 15.
- **C10 intra toggles (8/8 arms EXACT):** smooth/paeth/cfl/directional/
  diagonal/angle-delta + the seq-level filter-intra/intra-edge-filter pair.
  New `IntraToolCfg` on `PickFrameCfg` → applied onto `IntraSbyGates` (the
  C-diffed visit chain reads CLI + sf gates independently); chroma rides the
  existing `UvLoopPolicy` enable fields. Seq-level knobs drive the port
  directly and ASSERT the bootstrap seq bits agree (no bootstrap flow).
- **C9 tx controls (5 arms EXACT + 1 pinned-open):** tx64=0 / rect-tx=0 /
  flip-idtx=0 / default-tx-only=1 / reduced-tx-set=1 EXACT.
  `TxTypeSearchPolicy` gained `enable_flip_idtx` + `use_intra_dct_only`
  (threaded into `TxMaskParams` — C reads oxcf directly in `get_tx_mask`,
  stage-independent; partition_pick's winner-mode stage policies copy the CLI
  toggles from `cfg.pol` and OR the default-tx-only knob into MODE_EVAL per
  rdopt_utils.h:579). The reduced-tx-set frame-header bit is asserted == knob.
- **`--use-intra-dct-only=1` PINNED-OPEN** (KB-5/KB-10 pattern;
  `toggles_c9_intra_dct_only_pinned_open` fails on movement either way):
  64²cq32 out of band (+2.23% size, zensim −3.588), 64²cq63 EXACT, 128²cq12
  CLOSE (−1.40%, +0.333). Localized: Y recon IDENTICAL; first divergent leaf
  mi(0,0) 32×32 — real picks uv D45/aduv2 (eob 1), port picks uv V (eob 78);
  real's winners frame-wide are derived-type==DCT modes (the DCT-forced-search
  signature). The port's UV txb eval AND UV mode loop both match the C-pieces
  oracle chain under the knob — the five layer differentials
  (`uniform_txfm_yrd_diff` / `intra_sby_mode_loop_diff` /
  `rd_pick_intra_sb_diff` / `txfm_uvrd_diff` / `intra_sbuv_mode_loop_diff`)
  now SWEEP `use_intra_dct_only` with the oracle chain threading it into the
  REAL `get_tx_mask` facades (all green; the mask itself verified vs the REAL
  facade incl. the PAETH reduced-set empty-mask reset) ⇒ a shared port+oracle
  mis-model of the REAL UV loop under the knob. **Sibling-C dump DONE
  2026-07-17** (throwaway ar-swapped libaom, intra_mode_search.c + tx_search.c
  fprintf-instrumented, cq32 mi(0,0) 32×32; reverted after): C evaluates ONLY
  DC (this_rd 2157931) + D45/aduv2 (this_rd 1985157, WINS); C rejects V and the
  other directionals at `rd_pick_intra_angle_sbuv` anglefail (its
  `av1_txfm_rd_in_plane` returns INT_MAX). The port ACCEPTS V (uv_mode=1, aduv0,
  DCT tx_type=0, eob=1, **dist=0**, this_rd 1872917) so V WINS. Decisive: C's V
  prediction `block_sse`=1048576 == the port's V sse ⇒ the **prediction MATCHES
  — NOT a pred bug**. Root: the port's `txfm_rd_in_plane_uv_p` accepts V with
  DCT-dist 0 where C's `av1_txfm_rd_in_plane` rejects the identical V (same
  pred, same DCT) — a tx-search RD-eval/early-out mis-model shared by the port
  AND the `txfm_uvrd_diff` oracle (hence the green differential). Next: dump C's
  per-txb V DCT dist/coeffs inside `av1_txfm_rd_in_plane`/`search_txk_type` (the
  INT_MAX path fires before av1_txfm_uvrd's merge) vs the port's
  `search_tx_type_intra` V winner — find why the same DCT residual yields dist=0
  in the port and INT_MAX-rd in C. Cell stays pinned-open.
- **Remaining in the toggle families:** `--quant-b-adapt` (S–M),
  cost-upd-freq non-default arms (S–M), min/max-q clamps (S), SB128 encode
  (M, own chunk) — PARITY.md C8/C9/C11. (`--enable-tx-size-search=0`,
  `--disable-trellis-quant` 1/2, `--cdf-update-mode=0` all now EXACT.)

### Toggle-sweep addendum (same landing): tx-size-search + cdf-update-mode

- **C9 `--enable-tx-size-search=0` — EXACT (3/3 cells):**
  `TxTypeSearchPolicy.enable_tx_size_search` threaded three ways (all
  C-anchored): speed-0 single-pass method USE_FULL_RD → USE_LARGESTALL
  (intra_rd.rs pass-method pick), winner-mode sf derivation forces
  `tx_size_search_level = 3` after the speed derivation (speed_features.c:2726),
  leaf `tx_mode_is_select` init ANDs the knob (select_tx_mode ⇒
  TX_MODE_LARGEST; harness asserts the bootstrap header's tx_mode == knob).
  C forbids the `--enable-tx64=0` combo (encodeframe.c:2461 assert).
- **C11 `--cdf-update-mode=0` encoder e2e — EXACT (3/3 cells) + REAL BUG
  FIX:** the pack adapted partition/mode/tx-size symbol CDFs unconditionally
  (only `write_coeffs_txb_full` was gated), so a disable-cdf-update stream
  desynced against the non-adapting decoder (measured zensim −264 vs C +79
  pre-fix). Fix mirrors C's architecture: `OdEcEnc.allow_update_cdf`
  (default true — byte-inert for every existing caller), gated inside
  `aom_entropy::cdf::write_symbol` exactly like `aom_write_symbol`, set per
  tile in `pack_tile` from `PackCfg::allow_update_cdf` (C's write_modes).
  The decoder twin (`OdEcDec.allow_update_cdf`, landed 1dfbcc3) now has its
  encoder mirror.

### Toggle-sweep addendum (2026-07-17 pickup): trellis arms + REAL BUG FIX

- **C9 `--disable-trellis-quant=1` (NO_TRELLIS_OPT) — EXACT (3/3 cells):**
  the search `skip_trellis` and pack `enable_optimize_b` both resolve
  no-trellis; matched with no fix.
- **C9 `--disable-trellis-quant=2` (FINAL_PASS_TRELLIS_OPT) — EXACT (3/3
  cells) after a REAL BUG FIX (5a644c6):** `encode_b_intra_dry` hardcoded
  `dry_run_output_enabled: false` for its luma + chroma final-quant envs, so
  the OUTPUT_ENABLED pack pass did NOT apply FINAL_PASS trellis. `is_trellis_used`
  is flag-independent for NO/FULL/NO_ESTIMATE_YRD/lossless (why every prior gate
  matched), but FINAL_PASS trellises ONLY when `dry_run == OUTPUT_ENABLED`
  (encodemb.h:153): the search must not, the pack must. Pre-fix the arm was OUT
  of the RD-close band (Δzensim up to 1.855, recon divergence). Fix threads
  `encode_b_intra_dry`'s `output_enabled` arg into both envs — byte-inert for
  every existing gate (the flag is dead outside FINAL_PASS). Both trellis arms
  drive both sides through the real `pack_tile`.

## C2 loop-restoration ENCODER search — BIT-IDENTICAL, second bulk-port family straight to PARITY section A (2026-07-17, LR bulk track)

**`--enable-restoration=1` allintra speed-0 is BYTE-IDENTICAL to real aomenc — 8/8 gate
cells + 8/8 decision equality** (e24cf09 write syntax, 96d3464 numeric core, dfd757e
decision layer, 96534c4 wiring+gate). The full `av1/encoder/pickrst.c` search is ported;
this was the highest-priority default-parity family (PARITY C2: C's allintra default is
`enable_restoration=1` — av1_cx_iface.c:286, NOT touched by the allintra override).

Four layers, each C-diffed at its own boundary:

- **Write-side RU syntax** (`aom_entropy::lr`): `recenter_finite_nonneg` (forward) +
  `write/count_primitive_{quniform,subexpfin,refsubexpfin}` (aom_dsp/binary_codes_writer.c)
  + `write_wiener_filter` / `write_sgrproj_filter` / `write_lr_unit`
  (loop_restoration_write_sb_coeffs) + the pickrst bit counters `count_wiener_bits` /
  `count_sgrproj_bits`. Oracle: 400 randomized unit sequences BYTE-IDENTICAL to the REAL
  C arithmetic writer (`shim_lr_units_roundtrip` — EXPORTED `aom_write_primitive_
  refsubexpfin` + `aom_write_symbol` over REAL default LR CDFs), writer-side adapted CDFs
  equal C's, pure-Rust roundtrip; `count_primitive_refsubexpfin` EXHAUSTIVE vs the
  EXPORTED counter over all 4 coded (n,k) pairs (21,760 cells). `lr_write_diff.rs`.
- **Search numeric core** (`aom_restore::pick`): `find_average(_highbd)`,
  `compute_stats` (lowbd incl. `use_downsampled_wiener_stats` row accumulators; a
  `dgd_origin` base gives the C pointer semantics for the ±3 window reads at plane
  edges) + `compute_stats_highbd` (bit-depth divider), the separable-symmetric Wiener
  solve (`linsolve_wiener` with the b/278065963 overflow scaling, `update_{a,b}_sep_sym`,
  `wiener_decompose_sep_sym` 4-iter, `compute_score` identity gate, `finalize_sym_filter`),
  SGR projection (`pixel_proj_error` lowbd+highbd, `calc_proj_params` +
  `get_proj_subspace` with the C overflow guards, `encode_xq`). Oracles: EXPORTED
  `av1_compute_stats[_highbd]_c`, `av1_{lowbd,highbd}_pixel_proj_error_c`,
  `av1_calc_proj_params[_high_bd]_c`, `av1_selfguided_restoration_c` (the search's
  flt0/flt1 producer diffed DIRECTLY) via the new `pickrst_shim.c` — M/H byte-identical
  over 40×6×2×2 lowbd + 24×4×2×3 highbd cells, all 3 radius classes × bd 8/10/12.
  HONEST GAP: the Wiener solve chain is static in C (no export) — transcribed, covered
  by range/symmetry invariants + the e2e gate. `pick_diff.rs`.
- **Decision layer** (`aom_restore::pick`): `PlaneCtx` staging (padded dgd +
  `av1_extend_frame` + the encoder's TWO `save_boundary_lines` passes + trial dst —
  REUSES the C-proven frame-walk internals, opened pub(crate); never re-ported),
  `try_restoration_unit` (= `filter_unit` with stripe swaps, optimized_lr=0),
  `search_norestore/wiener/sgrproj/switchable` with C's exact bits accounting
  (AV1_PROB_COST_SHIFT) + `RDCOST_DBL_WITH_NATIVE_BD_DIST` f64 costing + the
  dual-SGR/switchable penalties, `restoration_search` (tile → SB raster →
  `corners_in_sb` units in CODING order for ref chaining; 150% last-unit + stripe
  voffset limits), `pick_filter_restoration` (unit-size descent max→min with the
  sb-width clamp, per-plane frame-type RD, all-none/worse-than-larger early breaks).
  Self-consistency gates (`pick_search.rs`): perfect recon → all-NONE; noisy recon →
  restoration found and applying it through the C-proven walk strictly reduces SSE
  (bd8+bd10, 320×130 multi-unit); deterministic; sf disables constrain exactly.
- **Encoder wiring + gate** (aom-encode pack + aom-bench): `pack_tile_lr` writes each
  SB's corner RUs at the SB root BEFORE the partition symbol into the SAME adapting
  tile context (per-tile `LrRefState`), `pack_tile` delegates with lr=None
  (byte-identical); `EncodeCell::c_encode_lr` (AV1E_SET_ENABLE_RESTORATION=1) +
  `port_encode_lr`: stock encode → derived-LF APPLY (`loop_filter_frame`, C's
  loopfilter_frame ordering) → the ported search on (source, deblocked recon) with
  `av1_fill_lr_rates` costs from frame-init LR CDFs + frame RDMULT +
  `lr_search_sf_allintra` (framesize-independent allintra setters + the
  qindex-dependent unit-size rule) → derived restoration header fields → repack with
  interleaved RU params when any plane restores. Decisions DERIVED, never bootstrapped
  (`parse_restoration_decision` witnesses real-vs-port at the bitstream level).

**GATE (`lr_restoration_gate.rs`, aom-bench): 8 real-content cells, ALL BYTE-IDENTICAL,
ALL decisions equal C's** — 64² cq{12,32,48}, 196² cq{20,48} (partial-SB edges), 352×288
quantizer-00 cq{32,55} (multi-unit descent grids), b10 352×288 cq32. Decision shapes:
all-NONE (cq12), WIENER-luma, SGRPROJ-luma, WIENER-all-3-planes (chroma 5-tap window +
subsampled procunits), mixed SGR-luma+WIENER-chroma (b10), and the size descent picking
128 at 196² cq20. Anti-vacuous floors (real-LR-active + port-LR-active) + full
byte-identity + decision-equality asserts (hardened after measuring 8/8 EXACT).

**Honest fractions:** e2e-gated = allintra speed-0 (full search) only. Speed 1–4 lpf_sf
arms ported (`lr_search_sf_allintra`) but NOT e2e-gated yet; GOOD-mode setters not wired
(asserted against); mono/4:4:4/bd12 gate cells absent; speed>=5 allintra is structurally
LR-off in C (sf disable + seq-bit clear). Follow-ups: speed-arm cells, format cells,
`pack_tile_from_trees` repack unification (drop the second search), SIMD for
`compute_stats` (the search hot spot) on the Gate-3 track.
## #29 pickup — palette exactness pin + intrabc 3b VERIFY-hazards resolved (2026-07-17)

**Palette (localized + graduated + pinned):** the 2 former "128² CLOSE" cells were
decode-both localized (`rd_close_palette::decode_diff_palette_close_cells`) to genuine
palette-induced AB/4-way partition RD near-ties — NOT multi-SB neighbour-cache bugs as the
prior gap note guessed:
- `ui_420_128_cq32`   — first divergence (mi 0,0) BLOCK_32X32 real HORZ_B vs port HORZ_4.
- `text_420_128_cq20` — first divergence (mi 8,20) BLOCK_16X16 real VERT vs port VERT_A.
Both are BYTE-EXACT with palette OFF (localizer's palette-OFF control), and the palette
machinery is verified C-faithful (`av1_allow_palette`; `av1_get_palette_bsize_ctx` =
`num_pels_log2[bsize]−num_pels_log2[8X8]`; `av1_get_palette_mode_ctx` above+left count;
k-means rtcd-validated; mid-search neighbour palette threaded at every `grid.stamp`). Same
class as the KB-10/KB-11 pinned near-ties → PINNED, closing needs a sibling-C per-candidate
partition-RD dump. The 5 EXACT cells + control are now HARD byte-identity asserts in the gate;
the localizer asserts the divergence PRESENT (self-promotes on any fix). See KB-P29.

**IntraBC chunk 3b skeleton — VERIFY hazards resolved (still UNWIRED, envelope-inert):**
- `set_mv_search_range`: MAX_FULL_PEL_VAL = **1023** verified vs mcomp_structs.h
  (MAX_MVSEARCH_STEPS=11); dropped the predecessor's misleading dead 2047 const.
- `intrabc_predict_chroma`: differential-tested byte-identical to the conformance-bit-exact
  decoder `intrabc_chroma_predict` over full-pel DVs × {420,422,444} × bd{8,10,12}
  (`intrabc_search::tests::intrabc_chroma_predict_matches_decoder`). Derivation proven equal.
- dv_ref 4-tuple order (nearest,near): already C-validated by `aom-entropy/tests/dv_ref_diff.rs`.
- `DEFAULT_TXFM_PARTITION_CDF`: byte-identical to the decoder's `default_txfm_partition_cdf`.
Still PINNED (the L piece, `// HANDOFF:` in-file + HANDOFF-SCREEN.md): the coeff arm
(skip-arm-only today, biased), hbd sse scaling, NSTEP diamond/mesh full-pel search, and the
whole 8-step integration map (ModeGrid dvs/skips → LeafWinner → rd_pick hook →
`encode_b_intra_dry` intrabc arm → pack → frame hash-table plumbing → harness knob → gate).
Do NOT run an intrabc RD-closeness gate until the coeff arm lands.

## C4 tune=IQ / tune=SSIMULACRA2 family — ALL 6 PIECES BYTE-EXACT + full composite (2026-07-17, tune track)

- **Scope**: the `handle_tuning` bundle (`av1_cx_iface.c:1938-1978`) that `--tune=iq` /
  `--tune=ssimulacra2` install — `enable_qm=1 qm(2,10)`, `sharpness=7`, `dist_metric=QM_PSNR`,
  `enable_chroma_deltaq=1`, `deltaq_mode=6` (Variance Boost), `enable_cdef=CDEF_ADAPTIVE`; IQ adds
  `enable_adaptive_sharpness=1`. **Everything OFF by default** (`TuneKnobs::default()` = PSNR); the
  proven byte-exact envelope is untouched (`dist_block_tx_domain_qm` returns the plain
  `dist_block_tx_domain` when `use_qm_dist_metric` is false — verified at the code level + by the
  full envelope suite staying green with the port in tree).
- **Each piece e2e byte-identical to real aomenc** (`encoder_gate_tune_iq_e2e`, 9 tests):
  1. **QM-level formulas** `aom_get_qmlevel_luma_ssimulacra2` + `aom_get_qmlevel_444_chroma`
     (`quant_common.rs`, `QM_FIRST/LAST_IQ_SSIMULACRA2` = 2/10) — `qm_level_diff` byte-exact vs the
     real C static inlines (4 formulas × 6 ranges × qindex 0..=255).
  2. **QM-PSNR dist metric** — the trellis distortion (`optimize_txb_qm`) AND the tx-search
     transform-domain distortion (`dist_block_tx_domain_qm` = `aom_dist::block_error_qm`) weighted
     by the forward QM only under `use_qm_dist_metric` (`txb_rdopt.c:346-351` dist fold,
     `:378-386` the trellis rshift-7 vs 5 gain); `TxTypeSearchPolicy::with_tune_knobs` forces
     transform-domain distortion on (`rdopt_utils.h:516-522`). Anti-vacuous: `tune_shim_smoke`
     (aom-sys-ref) — the C shim's QM-PSNR stream genuinely differs from PSNR.
  3. **`--sharpness` 0..7** — `av1_build_quantizer` rounding bias (`sharpness_adjustment`) + the
     trellis scale + the LF `sharpness_level`. Witness `sharpness_witness_knobs_bite`:
     `--sharpness=7` must change the C stream vs `--sharpness=0`.
  4. **`--enable-chroma-deltaq`** — `av1_set_quantizer`'s chroma delta-q arms; the port DERIVES the
     u/v dc/ac deltas from the knob and CROSS-CHECKS them against the real header (leak-free).
  5. **`--enable-adaptive-sharpness`** — the qindex-adaptive LF `sharpness_level` cap
     (`picklpf.c`; `frame_lf_sharpness`), cross-checked vs the real header. Witness: adaptive must
     change a sharpness-7 stream at qindex 200.
  6. **`--deltaq-mode=6` Variance Boost** — `allintra_vis.rs` full port
     (`variance_boost_block_variance`, `av1_get_sbq_variance_boost`, `variance_boost_delta_q_res`,
     `av1_adjust_q_from_delta_q_res`, `setup_delta_q_variance_boost`); `pack_tile` +
     `pack_tile_from_trees` per-SB delta-q threading (`SbEncodeEnv::deltaq`, per-SB qindex +
     rdmult + row re-select). Delta-q OFF reduces byte-identically to the prior path. Witness:
     `--deltaq-mode=6` must change the C stream; `--deltaq-strength=200` must differ from 100.
- **Composite** (`encoder_gate_tune_composite_full_e2e`, ADDED this landing): the whole bundle
  live at once for both IQ and SSIMULACRA2 — QM + QM-PSNR + sharpness=7 + chroma-deltaq +
  deltaq=6 (+ adaptive-sharpness for IQ). **54/54 cells byte-match** real `aomenc --tune=iq` /
  `--tune=ssimulacra2` (mono/420/444 × 64/128/192 × cq12/32/50). Proves the knobs COMPOSE, not
  merely that each works alone.
- **The one bundle member NOT ported here: `CDEF_ADAPTIVE`.** The composite overrides
  `enable_cdef=0` in the C reference: CDEF is the separate, already-bit-exact C1 track
  (`av1_cdef_search`), applied post-reconstruction so it is symbol-inert on the coded tile bytes.
  A full tune=IQ *with* CDEF-adaptive needs the C1 CDEF search wired under the per-SB tune qindex
  (cross-track, deferred).
- **Adoption / rebase**: adopted from the C4 predecessor WIP (worktree branches
  `a07347614b295cb09` / `a6595a97d6d5ead8b`) — compiled (fixing the never-compiled pack.rs
  delta-q threading), validated, composite arm added; rebased over the toggle-sweep (C8-C11) +
  loop-restoration (C2) landings (additive conflict resolutions in `tx_search.rs` /
  `partition_pick.rs` / `aom-sys-ref/src/lib.rs`, all keep-both).
