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

- **RD + front-end primitives (aom-dist)** — `block_error` / `highbd_block_error`
  (av1_[highbd_]block_error_c): transform-domain distortion, `error=sum((coeff-dqcoeff)^2)`
  + `ssz=sum(coeff^2)` (lowbd 32-bit products; highbd 64-bit + rounded-shift 2*(bd-8)) —
  the fast distortion the RD search pairs with the trellis rate. `subtract_block` /
  `highbd_subtract_block` (aom_[highbd_]subtract_block_c): the residual generator
  `diff=src-pred`, natively strided — completing the front end (pred -> subtract ->
  xform_quant). Both diffed vs exported C (highbd subtract via a CONVERT_TO_BYTEPTR shim).

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
  `constrain` + directional primary/secondary taps + CDEF_VERY_LARGE clipping.
  Harness: `aom-cdef/tests/cdef_filter_diff.rs` — 320k comparisons,
  byte-identical to C. CDEF (direction search + filter) now complete lowbd.

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

1. **Reconstruction / distortion for RD**: inverse-transform dqcoeff (aom-transform inv
   is validated) + pixel SSE (aom-dist) to get the block distortion; with the trellis
   rate this is the RD cost the mode search compares. First clean non-bitstream RD piece.
2. **Coding-block lifts**: frame-edge context clipping (av1_set_entropy_contexts edge
   case), chroma subsampling (plane_bsize derivation), non-uniform block->tx partition.
3. **Mode & partition search (RDO)** — the hardest bit-identity target; the layer that
   drives encode_coding_block_plane per candidate.
2. **Intra prediction** (`av1/common/reconintra`, `aom_dsp` intra predictors) —
   per-mode bit-exact, differential per predictor.
3. **Loop filters**: deblock, CDEF, loop-restoration (decoder + encoder search).
4. AVX2/NEON SIMD specializations (perf gate), each diffed lane-level vs scalar.
5. Encoder RDO + rate control (hardest bit-identity target).

## Perf gate honest number

Like-for-like vs C's production AVX2 (`aom_sad64x64_avx2`): Rust AVX2 SAD is
**~2.2x** (direct kernel) / ~2.5x (with runtime dispatch) — gate is <=1.20x, so
NOT met. The gap is the kernel (libaom hand-tuned asm ~2x faster), not dispatch
(~0.15x). The earlier 1.42x figure was vs C *scalar* and was replaced.

## Gate posture (honest)

Real, verified, ratcheting progress across BOTH tracks — but still a fraction of
the whole. Green so far: full transform subsystem, the *entire* scalar quantizer
surface (fp+b, lowbd+highbd, QM+flat), the full coefficient trellis (QM+flat), and
the entire symbol-coding stack (range coder + CDF adaptation). None of the four project
gates (full-corpus correctness, ≤1.20× perf, full coverage, zenavif parity) is
satisfied yet; the machinery that makes each mechanically checkable is in place
and every landed module is byte-exact vs C within it.
