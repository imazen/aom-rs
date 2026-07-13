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
  (Quant-matrix path + adaptive/highbd variants: TODO.)

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
  (Highbd + edge filter/upsample: TODO.)

- **Deblocking loop filter (lowbd)** (`aom_dsp/loopfilter.c`), both tracks:
  horizontal + vertical, widths 4/6/8/14 (filter4/6/8/14, hev/flat/flat2 masks,
  signed-char-clamp domain). Harness: `aom-loopfilter/tests/lpf_diff.rs` —
  240k comparisons over branch-exercising pixel/threshold strategies,
  byte-identical to C. (Highbd variants: TODO.)

- **Distortion metrics (encoder-critical, speed-0 path)** (`aom_dsp/sad.c`,
  `variance.c`): `aom_sad`, `aom_variance`, `aom_sub_pixel_variance` (bilinear
  2-tap) over all 22 block sizes. Harness: `aom-dist/tests/dist_diff.rs` —
  ~198k comparisons (SAD + variance + subpel var/sse), byte-identical to C.
  (avg/masked/obmc SAD + highbd: TODO.)

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

## Infrastructure standing

- Rust workspace + `aom-sys-ref` FFI oracle crate linking the reference `libaom.a`.
- Transpiler `xtask/transpile_txfm1d.py` for the regular ping-pong butterflies.
- Coverage ledger `coverage/checklist.json`.

## Next candidates

1. **Coefficient coding**: scan orders, txb context, `av1_write_coeffs_txb` /
   `read_coeffs_txb` — ties transform+quant to the entropy layer (both tracks).
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
the whole. Green so far: full transform subsystem, two quantizers, and the entire
symbol-coding stack (range coder + CDF adaptation). None of the four project
gates (full-corpus correctness, ≤1.20× perf, full coverage, zenavif parity) is
satisfied yet; the machinery that makes each mechanically checkable is in place
and every landed module is byte-exact vs C within it.
