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

**HONEST GAP #2 (encoder track, LF-level search on strong filters):** a
combined-content probe (gradient + hard edges + texture in one frame, task-3's
"richer content") drives real aomenc to STRONG nonzero loop-filter levels
(e.g. `[15,6]`/`[16,5]`) where this port's `pick_filter_level` disagrees, even at
cq60/cq63. The existing `encoder_gate_lf_level_bit_exact_vs_real` only covers the
AB-probe checkerboard levels (`[8,8]`/`[8,3]`/`[7,16]`); the strong-filter regime
is unverified. This blocks the combined-in-one-frame content case (its high
texture also feeds the coeff-trellis gap -> our recon differs -> our LF search
sees different pixels). Smallest next chunk: extend the LF-level gate to
strong-filter content (fed real's own reconstruction, as that gate already does)
and C-trace the first divergent `pick_filter_level` term at a high level.
