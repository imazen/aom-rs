# zenav1-aom [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenav1-aom/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenav1-aom/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenav1-aom?style=flat-square)](https://crates.io/crates/zenav1-aom) [![lib.rs](https://img.shields.io/crates/v/zenav1-aom?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenav1-aom) [![docs.rs](https://img.shields.io/docsrs/zenav1-aom?style=flat-square)](https://docs.rs/zenav1-aom) [![license](https://img.shields.io/crates/l/zenav1-aom?style=flat-square)](#license)

Pure-Rust, bit-exact reimplementation of [libaom](https://aomedia.googlesource.com/aom)
(the Alliance for Open Media AV1 reference codec), built module-by-module behind
differential harnesses. Every ported kernel is validated against a pinned C libaom
**v3.14.1** oracle (`03087864`), and landed decode/encode paths are held to
byte-exact bitstream gates — the port is measured against the real exported C
functions, not a transcription of them.

`#![forbid(unsafe_code)]` · runtime SIMD dispatch via [archmage](https://github.com/imazen/archmage) · `libaom v3.14.1`

## Crates

Four published crates. The facade is the one to depend on; the others are its
building blocks, publishable on their own for consumers who want a narrower slice.

| Crate | What it is |
|---|---|
| **`zenav1-aom`** | Thin facade. Re-exports the DSP kernels plus the feature-gated decoder and encoder as one dependency. Start here. |
| **`zenav1-aom-dsp`** | The consolidated DSP + entropy kernels: transform, quant, txb, cdef, restore, intra, loopfilter, dist, inter, convolve, recon, dispatch, and the MSAC range coder — each a module. |
| **`zenav1-aom-decode`** | The AV1 decoder: partition walk, per-leaf mode-info/coeff decode, intra predict, inverse transform, and the post-filter (deblock/CDEF/restoration) frame walk. |
| **`zenav1-aom-encode`** | The AV1 encoder: RD partition/mode/tx search, forward transform + quantize + entropy coding, and bitstream pack. |

Two more crates are dev-only (`publish = false`) and never ship: `zenav1-aom-sys-ref`
(the C-libaom FFI oracle the differential harnesses diff against) and
`zenav1-aom-bench` (the Gate-3 performance harness).

### Install

```toml
[dependencies]
# decoder + encoder (default)
zenav1-aom = "0.0.1"
```

The encoder is feature-gated, so a size-sensitive or wasm consumer can build a
decode-only stack — the encoder crate is then never compiled:

```toml
[dependencies]
zenav1-aom = { version = "0.0.1", default-features = false, features = ["decode"] }
```

Depending on any published crate pulls in **no C toolchain and no `build.rs`** —
the C libaom oracle is a dev-dependency of the harnesses only, never a normal
dependency of the shipping crates.

## Status: early development

This is a work in progress, not yet a drop-in libaom replacement. What holds today,
measured against the C oracle:

- **Decoder — intra is conformance-clean.** Bit-identical to C across the AV1
  intra conformance scope (the CI-wired `xtask/conformance.py --scope intra`:
  byte-identity + golden per-plane MD5), including the aggressive q62/q63 quantizer
  range, superres, 128×128 superblocks, multi-tile, and film-grain synthesis.
  Inter-frame decode is progressing byte-exact through a single-reference feature
  ladder and several real frames.
- **Encoder — ALLINTRA byte-matches aomenc.** The all-intra (usage=2), KEY-frame
  path byte-matches real `aomenc` across `--cpu-used 0..9` on synthetic grids and
  on real conformance-decoded content at speed 0. Non-default stills knobs are
  byte-exact too (QM, CDEF search, loop-restoration search, SB128, multi-tile,
  film grain, lossless, 10/12-bit). A handful of high-qindex partition/mode
  near-ties, real-content parity above speed 0, and the IntraBC + inter var-tx
  coefficient arm are still open — each pinned by a self-promoting gate. Inter-frame
  encode is an early skeleton.

[`STATUS.md`](STATUS.md) tracks what has landed module-by-module; [`PARITY.md`](PARITY.md)
is the stills-parity ledger; [`PORTING.md`](PORTING.md) maps each Rust module to the
`upstream/` libaom file(s) it ports and to the differential test that gates it.

## Building and testing (fresh box)

The C libaom oracle lives in-repo as a pinned git submodule at `upstream/`, and the
test build drives it through cargo — there is no manual oracle-build step:

```sh
git clone --recurse-submodules https://github.com/imazen/zenav1-aom.git
cd zenav1-aom
cargo test          # builds the libaom oracle once, then runs the differential suite
```

The first `cargo test` (or `cargo build -p zenav1-aom-sys-ref`) auto-initializes the
`upstream/` submodule if it is empty and builds libaom once via cmake, in the
deterministic single-thread oracle config ([`reference/BUILD_CONFIG.md`](reference/BUILD_CONFIG.md)),
cached forever after on the submodule SHA. It needs **cmake, nasm, and a C compiler**
on `PATH` — if any is missing the build fails loud with the one-line install
(`sudo apt-get install cmake nasm build-essential`), never a cryptic linker error.

[`just`](justfile) wraps the common flows: `just test` (full differential suite),
`just test-scalar` (the `AOM_FORCE_SCALAR` pin that forces every SIMD kernel through
its scalar twin), `just test-fast` (same coverage, optimized), and `just bench-gate3`
(the Gate-3 port-vs-C paired benchmark).

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

Upstream C code from [libaom](https://aomedia.googlesource.com/aom) is
BSD-2-Clause with the Alliance for Open Media Patent License 1.0 — see
[LICENSE](LICENSE) and [PATENTS](PATENTS); those terms continue to cover the
upstream work this port derives from. libaom is battle-tested, carefully
engineered code — this port stands entirely on that foundation.

### Path to MIT

If someone covers Imazen's 2026 AI + server costs, we'll release this port
under MIT — or under the original upstream license (BSD-2-Clause + AOM
Patent License 1.0). Contact support@imazen.io.
