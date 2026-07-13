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

## Gate posture (honest)

Real, verified, ratcheting progress across BOTH tracks — but still a fraction of
the whole. Green so far: full transform subsystem, two quantizers, and the entire
symbol-coding stack (range coder + CDF adaptation). None of the four project
gates (full-corpus correctness, ≤1.20× perf, full coverage, zenavif parity) is
satisfied yet; the machinery that makes each mechanically checkable is in place
and every landed module is byte-exact vs C within it.
