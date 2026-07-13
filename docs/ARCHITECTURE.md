# aom-rs — architecture & gate enforcement

Target reference: **libaom v3.14.1** (pinned). Everything is defined as bit-for-bit
equivalence against a from-source build of that exact tag.

## Guiding principle

We never claim a module "done" on inspection. A module is done when a **differential
harness** feeds identical input to the C reference and the Rust port and asserts
byte-identical output, over (a) a fixed corpus and (b) randomized fuzzing. The harness
is the product; the port is downstream of it.

## Crate decomposition (mirrors libaom's internal module graph)

Bottom-up so each layer is testable against C before anything depends on it.

| crate | mirrors (libaom) | bit-exact oracle |
|-------|------------------|------------------|
| `aom-dsp-prim`  | `aom_dsp/` primitives: bit reader/writer, cdf, rounding | direct C fn export |
| `aom-transform` | `av1/common/av1_txfm`, `av1_inv_txfm`, `av1_fwd_txfm` | per-tx-type C fn |
| `aom-quant`     | `av1/encoder/av1_quantize`, dequant | per-config C fn |
| `aom-entropy`   | `aom_dsp/entdec`, `entenc`, `av1/common/entropy*` | symbol stream |
| `aom-predict`   | intra/inter prediction, `av1/common/reconintra` | per-mode C fn |
| `aom-loopfilter`| deblock, CDEF, loop-restoration | per-plane C fn |
| `aom-decoder`   | `av1/decoder/*` | full-frame decode |
| `aom-rc`        | rate control (`av1/encoder/ratectrl`) | state trace |
| `aom-rdo`       | mode search / RDO (`av1/encoder/*`) | decision trace |
| `aom-encoder`   | `av1/encoder/encoder.c` top level | full bitstream |
| `aom-sys-ref`   | FFI bindings to the pinned C libaom (oracle only, test-cfg) | — |

Each crate has a `simd/` submodule: scalar reference first (bit-exact, portable),
then AVX2 / NEON specializations that must produce **identical** output to scalar
(a lane-level differential test), matching libaom's own C-vs-SIMD contract.

## The four gates, made mechanical

1. **Decoder correctness** — `harness/decode_diff`: for each file in the AV1
   conformance corpus + libaom `test/*.ivf`, decode with C and Rust, assert every
   output frame's planes are identical (MD5 per frame, as libaom's own tests do).

2. **Encoder correctness** — `harness/encode_diff`: for the matrix
   {cpu-used 0..9} × {good, realtime, allintra} × {test clips}, encode with both,
   assert the **emitted bitstream bytes** are identical. Divergences get a row in
   `docs/DIVERGENCES.md` with root cause + justification; an undocumented divergence
   fails CI.

3. **Performance** — `harness/bench`: criterion wall-time per preset vs C, ratchet
   file `perf/baseline.json`. `ratio > 1.20` on the HW matrix fails CI.

4. **Coverage** — `coverage/checklist.json` is **auto-derived** from libaom's CLI
   (`aomenc --help`, `aomdec --help`) + the enum surface in `aom/aomcx.h` /
   `aom/aom_encoder.h` (every `AOME_SET_*`, `AV1E_SET_*` control). Each item maps to a
   test id; `xtask coverage` prints red/green. "Done" = all green.

5. **zenavif** — `aom-rs` implements the same trait zenavif's C-aom backend does;
   `harness/avif_parity` asserts identical AVIF output for both backends.

## Divergence policy

Bit-identity is the default and the CI-enforced contract. Known classes where C
libaom is not even self-consistent (multithread tie-breaks, `--enable-thread` vs
single, some `float` reduction orders) are pinned to the **single-thread, C-scalar,
deterministic** build config of the reference so the target is well-defined. That
build config is recorded in `reference/BUILD_CONFIG.md`; we match *that*.
