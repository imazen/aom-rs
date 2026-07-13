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

## Infrastructure standing

- Rust workspace + `aom-sys-ref` FFI oracle crate linking the reference `libaom.a`.
- Transpiler `xtask/transpile_txfm1d.py` for the regular ping-pong butterflies.
- Coverage ledger `coverage/checklist.json`.

## Next (critical path — speed-0 encoder)

1. **Forward 2-D transform** (`av1_fwd_txfm2d.c`): row/col config, transpose,
   rounding/shift stages, `av1_transform_config`. Composes the 1-D kernels above.
2. **Quantization / dequant** (`av1_quantize.c`) — next bit-exact encoder stage.
3. AVX2/NEON SIMD specializations of the 2-D transform, each diffed against the
   scalar port (lane-level), matching libaom's own C-vs-SIMD contract.
4. Begin the decoder track's inverse transforms in parallel (`av1_inv_txfm1d.c`)
   — same methodology, mirror harness.

## Gate posture (honest)

Two gates are *mechanized but far from satisfied*: correctness (only the txfm1d
module is green so far) and coverage (12 items green of the full surface). Perf,
full encoder/decoder bitstream, and zenavif gates are not yet exercised. The
value delivered so far is the verified methodology + first module, not a
finished codec.
