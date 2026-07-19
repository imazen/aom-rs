# zenav1-aom — architecture & gate enforcement

Target reference: **libaom v3.14.1** (pinned). Everything is defined as bit-for-bit
equivalence against a from-source build of that exact tag.

> Rewritten 2026-07-19 to match the tree. The previous version described an original
> design that was never built (a 10-crate split with `aom-predict` / `aom-rc` /
> `aom-rdo`, a `harness/` directory, `perf/baseline.json`, criterion benches). None of
> those exist. What follows is verified against the source.

## Guiding principle

We never claim a module "done" on inspection. A module is done when a **differential
harness** feeds identical input to the C reference and the Rust port and asserts
byte-identical output, over (a) a fixed corpus and (b) randomized fuzzing. The harness
is the product; the port is downstream of it.

**Evidence hierarchy** — not all oracles are equal, and the difference has cost real
debugging time:

1. **Real exported C function** — best. The port is compared against libaom's own symbol.
2. **Facade over a real C function** — acceptable. A thin shim that calls the real thing.
3. **Verbatim transcription** — weakest. A hand-copied C algorithm can carry a *shared*
   bug that the differential structurally cannot catch. This has happened here (see the
   KB-5 `is_cfl_allowed` entry, where port and reference shared a transcribed gate).

## Crate decomposition

Six packages. The 2026-07 consolidation collapsed an earlier 17-crate split — fine-grained
crates aided parallel porting, but a public release wants a small surface.

| crate | mirrors (libaom) | bit-exact oracle |
|-------|------------------|------------------|
| `zenav1-aom` | facade: re-exports under `decode` / `encode` features | — |
| `zenav1-aom-dsp` | `aom_dsp/` + `av1/common/` kernels (see module table) | per-fn C export |
| `zenav1-aom-decode` | `av1/decoder/*` | full-frame decode |
| `zenav1-aom-encode` | `av1/encoder/*` | full bitstream |
| `zenav1-aom-sys-ref` | FFI to the pinned C libaom (oracle only, **dev-dep**) | — |
| `zenav1-aom-bench` | Gate-3 performance harness (bench-only, never published) | — |

`zenav1-aom-dsp` is the shared kernel crate, organised by libaom module:

| module | LOC | module | LOC |
|---|---|---|---|
| `entropy` | 15,602 | `cdef` | 1,556 |
| `quant` | 10,653 | `loopfilter` | 1,543 |
| `transform` | 8,155 | `dist` | 847 |
| `txb` | 4,727 | `convolve` | 160 |
| `restore` | 3,333 | `dispatch` | 153 |
| `intra` | 2,672 | `recon` | 51 |
| `inter` | 1,747 | | |

### Two invariants worth protecting

**A consumer build compiles zero C.** `zenav1-aom-sys-ref` is a *dev-dependency* of every
crate and is the sole `build.rs` in the workspace, so nothing downstream invokes cmake or
touches the oracle. Only `cargo test` builds C.

**The facade's features gate cleanly.** `decode` and `encode` are optional dependencies;
`--no-default-features --features decode` excludes the encoder entirely. Note the payoff
is bounded: `aom-dsp` is not optional, so decode-only still compiles all of it. Measured
at `benchmarks/build_time_decompose_2026-07-19.md` — 9.5% wall-clock saving, 30% CPU.

### SIMD

Dispatch goes through **archmage 0.9.27** + **magetypes** (capability tokens, `#[arcane]` /
`#[rite]`), with an `avx512` cargo feature. Scalar paths are the bit-exact reference;
vector paths must produce identical output, matching libaom's own C-vs-SIMD contract.
There is no per-crate `simd/` submodule — that was the old design.

## The C oracle

`upstream/` is a git submodule pinned to the reference commit. `crates/aom-sys-ref/build.rs`
checks the toolchain, auto-initialises the submodule, cmake-builds libaom in the
deterministic single-thread config, caches the result stamped by submodule SHA, then
compiles and links the shim translation units. No manual setup step.

The exact oracle build config lives in `reference/BUILD_CONFIG.md`.

## The four gates, made mechanical

1. **Decoder correctness** — `crates/aom-decode/tests/` (11 files as of 2026-07-19,
   `conformance_corpus.rs` and `real_bitstream.rs` being the broad ones). The corpus is the
   official AV1 decode-conformance set; `xtask/conformance.py` parses libaom's own
   `upstream/test/test-data.sha1` manifest, fetches the vectors, and categorises them by
   bit-depth and feature. Each vector ships a companion `.md5` holding one MD5 per decoded
   frame — that per-frame list is the golden answer libaom's own tests assert against, and
   it is ours. CI scope: `xtask/conformance.py --fetch --scope intra`.

2. **Encoder correctness** — `crates/aom-encode/tests/` (95 files) plus the e2e gates in
   `crates/aom-bench/tests/` (17). The contract is byte-identity of the emitted bitstream
   vs real `aomenc` across `--cpu-used 0..9`. Gates are named `encoder_gate_*` and assert
   full byte-identity, not a tolerance.

   Several gates are **self-promoting**: they pin a *known* divergence by asserting it is
   still present, so the test fails the moment the port becomes byte-exact — at which point
   you promote the cell into the byte-exact list. This keeps a known gap honest instead of
   letting it rot as a silent skip.

3. **Performance** — `crates/aom-bench`, using **zenbench 0.1.9** paired/interleaved
   benchmarks against the real C oracle in-process via `aom-sys-ref`, plus a callgrind
   profile driver. Target is ≤ 1.20× C. Results are committed under `benchmarks/` with a
   companion `.meta` recording commit, host, and the exact command.

4. **Coverage** — `xtask/coverage.py` auto-derives the feature checklist from libaom's live
   CLI (`aomenc --help` / `aomdec --help`) and the control-enum surface, then cross-references
   `coverage/feature_map.json`. A feature is green **only** if it maps to a passing test id;
   the tool does not invent green. Standing audits live in `coverage-audit/`.

5. **zenavif integration** — `crates/aom-encode/tests/avif_parity.rs` muxes the port's
   byte-exact AV1 payload into an AVIF still via `zenavif-serialize`, then closes the loop
   twice: the container round-trip must return the coded bytes verbatim, and the extracted
   stream must decode (through the port's own decoder) to pixels matching a decode of real
   aomenc's stream.

## Divergence policy

Bit-identity is the default and the enforced contract.

Known divergences are tracked as numbered **KB entries in `CLAUDE.md`** ("Known Bugs"), each
with `file:line` references, the root cause once found, and the gate that closes it. There
is no `docs/DIVERGENCES.md`. Module-level progress is tracked in `STATUS.md`.

A divergence is closed only by a landed fix verified on `origin/main` — never by relaxing a
test. Adding `#[ignore]`, loosening a threshold, changing a golden, or letting a test skip
itself at runtime all count as relaxing it.

Classes where C libaom is not self-consistent (multithread tie-breaks, `--enable-thread` vs
single, some `float` reduction orders) are pinned to the **single-thread, C-scalar,
deterministic** build config of the reference so the target is well-defined. We match *that*.
