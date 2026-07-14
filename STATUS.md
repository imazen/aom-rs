# aom-rs status

Reference target: **libaom v3.14.1** (`03087864`). Oracle built from source
(single-thread deterministic config, `reference/BUILD_CONFIG.md`).

## Done (bit-exact vs C oracle, differential-fuzz verified)

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
`dst.width` guard — fixed to header frame dims. Remaining deblock envelope hole:
**4:2:2 chroma deblocking rejected** (libaom reads
`max_txsize_rect_lookup[BLOCK_INVALID=255]` OOB for tall blocks at ss=(1,0) —
NDEBUG production build; not portable). 4:2:2 luma-only IS in envelope. Next
envelope tools: CDEF (kernels partial), segmentation, palette, SB128, multi-tile,
superres, intrabc (loop restoration landed 2026-07-14 — see below).

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

### Known Bugs (tracked, NOT fixed this session — out of scope for the overflow fix)

- **qindex=0 partition-population mismatch** (`crates/aom-encode/tests/
  pack_tile_roundtrip.rs:852`, found by the same true-corner sweep above,
  `pad=0`): at `qindex=0` (both `allintra=true,ss=(0,0)` and
  `allintra=false,ss=(0,0)`), the DECODED partition-type population
  disagrees with the SEARCH's own winning trees (`assert_eq!` failure,
  e.g. `left: (34,22,8,5,1) right: (42,2,6,10,10)` for
  `(leaves,none,split,horz,vert)`). Not an overflow — a genuine read/write
  divergence specific to the finest quantizer. Root cause NOT investigated
  (out of scope here); repro: `run_pack_roundtrip_case(0, 0, true, 0, 0)`.
- **qindex=0 "skip always 0" assumption violated** (`pack_tile_roundtrip.rs:
  287`, same sweep, `allintra=false, ss=(1,1), qindex=0`): the read-side
  harness asserts `info.skip == 0` (documented KEY-intra-envelope
  assumption — matches `block_rd_txfm`'s C comment "Signal non-skip_txfm for
  Intra blocks", i.e. the currently-ported scope always signals the TXB
  loop as non-skip) but decodes `info.skip == 1` at this qindex. Unconfirmed
  whether this is a genuine block-level skip/no-skip decision
  (`av1_txfm_search`'s `choose_skip_txfm`, tx_search.c:3862-3869) reachable
  at qindex=0 that the current port doesn't yet wire, or a decode-side
  symptom of the same root cause as the partition-population bug above.
  Repro: `run_pack_roundtrip_case(1, 1, false, 0, 0)`.

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

**AB (`HORZ_A`/`HORZ_B`/`VERT_A`/`VERT_B`) is NOT ported this chunk** — see
the MISSING section above for exactly what's needed (the
`reuse_prev_rd_results_for_part_ab` context-copy mechanism, LIVE at speed 0
unlike its mode-cache-only cousin which is dead, plus AB's own NN prune).
Commits: `756dfa1` (decode-diff investigation), `ae19fe6` (pre-existing fmt
drift, separated), then the 4-way port itself (this session — see `git log`
for the exact SHA landed after this doc update).

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
— the frame header is bootstrapped from the real parse (LF-level/CDEF-strength
search unported — and the nonzero-LF header path is now CONFIRMED broken, not
just untested, per the AB-probe finding above), but every coded byte of the
tile-group payload is this port's own derivation. 8 of 10 `PARTITION_*` types
are ported (NONE/SPLIT/HORZ/VERT/HORZ_4/VERT_4); AB (HORZ_A/HORZ_B/VERT_A/
VERT_B) remains. None of the four project gates (full-corpus correctness,
≤1.20× perf, full coverage, zenavif parity) is satisfied yet; the machinery
that makes each mechanically checkable is in place and every landed module is
byte-exact vs C within it.
