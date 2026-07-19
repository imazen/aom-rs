# Build-time decomposition: decode-only vs full — 2026-07-19

Question: after the 17→4 crate consolidation (#2/#3), how well does the `decode`
feature actually decompose the build?

Answer: **decode-only saves 9.5% of wall-clock and 30% of CPU work.** The payoff is
bounded by `zenav1-aom-dsp`, which is compiled in full either way and accounts for
97% of the decode-only critical path.

## Measured (clean release builds, separate target dirs, sequential)

No compiler cache: `RUSTC_WRAPPER`/`RUSTC_WORKSPACE_WRAPPER` unset, sccache not
installed. 16 cores. Library only — no dev-deps, no test binaries, so **zero C is
compiled** (`zenav1-aom-sys-ref` is a dev-dependency of every crate and is the sole
`build.rs` in the workspace; a consumer build never invokes cmake).

| config | wall | CPU-s |
|---|---|---|
| deps only (`-p archmage`) | 3.46 s | 4.32 s |
| `+ zenav1-aom-dsp` | 9.43 s | 19.22 s |
| **decode-only** (`--no-default-features --features decode`) | **9.70 s** | **22.44 s** |
| encode-only (`--no-default-features --features encode`) | 10.81 s | 29.51 s |
| **full** (default features) | **10.72 s** | **32.28 s** |

Derived per-unit cost:

| unit | CPU-s | share of full | LOC |
|---|---|---|---|
| external deps | 4.32 | 13% | — |
| `zenav1-aom-dsp` | 15.00 | 46% | 51,228 |
| `zenav1-aom-decode` | ~3.2 | 10% | 12,588 |
| `zenav1-aom-encode` | ~10.3 | 32% | 42,782 |

`aom-dsp` solo recompile (deps warm) measured directly at 5.04 s wall / 15.00 s CPU.

Feature gating verified structurally: 0 `aom_encode` artifacts in the decode-only
target dir, 3 in the full one.

## Why wall-clock moves so much less than CPU work

Critical path is `deps → aom-dsp → {decode ∥ encode}`. Decode and encode are
independent and compile concurrently, so excluding the encoder removes 10.3
CPU-seconds of work but only shortens the path by the difference between the two
arms. The serial prefix (deps + dsp) is 9.43 s of the 9.70 s decode-only wall — 97%.

## Why whole-module gating inside aom-dsp wouldn't help much

Cross-referencing `aom_dsp::<module>` references per consumer crate:

| module | LOC | decode refs | encode refs |
|---|---|---|---|
| entropy | 15,602 | 41 | 78 |
| quant | 10,653 | 9 | 54 |
| transform | 8,155 | 2 | 15 |
| txb | 4,727 | 14 | 45 |
| restore | 3,333 | 3 | 0 (via pick) |
| intra | 2,672 | 5 | 13 |
| inter | 1,747 | 24 | 2 |
| cdef | 1,556 | 3 | 1 |
| loopfilter | 1,543 | 7 | 4 |
| **dist** | **847** | **0** | **31** |
| convolve | 160 | 0 | 5 |
| recon | 51 | 2 | 3 |

`dist` (847 LOC, 1.6% of the crate) is the only cleanly encode-only module. The real
encode-only weight lives *inside* shared modules — forward transforms within
`transform`, quantizers and trellis within `quant`, write/cost paths within `txb`,
`pick` within `restore` — so extracting it means `#[cfg(feature)]` surgery through
the four largest and most bit-exactness-critical modules in the crate.

## Recommendation: do not split aom-dsp

The absolute numbers don't justify it. A perfect split saves single-digit seconds off
a 10.7 s build, and it would place cfg boundaries through code where byte-exactness
lives — each gate is somewhere a decode-only regression can hide, and proving it
didn't requires showing decode-only output stays byte-identical.

Revisit if `aom-dsp` grows substantially. The inter-frame work (#11) will expand
`inter` (currently 1,747 LOC) considerably, which shifts the ratio.

## Not measured

**Binary size.** Whether dead-code elimination strips the encode-only DSP from a
final artifact needs a real LTO link of a decode-only binary vs a full one. No claim
either way here.
