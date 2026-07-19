# Changelog

## Workspace

### [Unreleased]

### Changed

- **Consolidated the 13 DSP/entropy kernel crates into one `zenav1-aom-dsp`**
  (transform, quant, txb, cdef, restore, intra, loopfilter, dist, inter,
  convolve, recon, dispatch, entropy) — each is now a module, e.g.
  `aom_dsp::transform`, `aom_dsp::entropy`. Shrinks the release surface from 12
  publishable sub-crates to one. Byte-exactness unchanged (pure namespacing —
  only module paths moved); the differential gates stay green. (GitHub #2;
  20324ad, cf0541e, a9a995e, be7586b, c63c3f9, c51fdce, e57c31e)
- **Renamed every crate to the `zenav1-aom-*` prefix** (`zenav1-aom-dsp`,
  `zenav1-aom-decode`, `zenav1-aom-encode`, `zenav1-aom-sys-ref`,
  `zenav1-aom-bench`). Short `[lib] name`s (`aom_dsp`, `aom_decode`, …) are
  retained so interior `use aom_dsp::…` does not churn; only package names, dep
  keys, and CI/justfile `-p` args changed. (GitHub #3 Phase 2; 52be170)
- Publish flags corrected: `zenav1-aom-sys-ref` is now `publish = false` (was
  wrongly publish=default); `zenav1-aom-decode` / `zenav1-aom-encode` are now
  publishable (the facade re-exports them). End state: 4 publishable
  (`zenav1-aom`, `-dsp`, `-decode`, `-encode`) + 2 dev-only (`-sys-ref`,
  `-bench`). (52be170)
- Relicensed to `AGPL-3.0-only OR LicenseRef-Imazen-Commercial` — the standard
  Imazen dual license (LICENSE-AGPL3 + LICENSE-COMMERCIAL added). Upstream
  libaom LICENSE (BSD-2-Clause) and PATENTS (AOM Patent License 1.0) restored
  at the repo root; they continue to cover the upstream work this port derives
  from. We will release this port under MIT or the original upstream license
  if Imazen's 2026 AI + server costs are covered. (527852efc15a)
- CI: added the org-bar platform matrix — `windows-11-arm`, `macos-15-intel`,
  and `i686-unknown-linux-gnu` (via cross) — as pure-Rust portability jobs
  (invariant A: no C toolchain, no cmake/nasm), while the full C-oracle
  differential suite stays on the linux jobs. Also renamed the CI comment's
  stale `crates/aom-dispatch` ref to `aom_dsp::dispatch`. (GitHub #3 Phase 4;
  fb7e8da)

### Added

- **`zenav1-aom` facade crate** re-exporting `dsp` plus feature-gated `decode` /
  `encode` (both default). `default-features = false, features = ["decode"]`
  builds a decode-only stack (the encoder crate is never compiled) for
  size-sensitive / wasm consumers. (GitHub #2; 52be170)
- Rust-consumer docs for the 4-crate `zenav1-aom-*` structure (GitHub #3
  Phase 3): a rewritten Rust-facing README.md (crate map, install snippet,
  honest early-dev status, fresh-box `--recurse-submodules && cargo test` flow,
  `imazen/zenav1-aom` badges; 5bfa09a); `PORTING.md`, the C→Rust auditability
  map pairing each module with its `upstream/` libaom source + differential gate
  (9d8ddce); and minimal per-crate READMEs for the 4 published crates (e8ec2c1).
  (initial README + this changelog: 527852efc15a)
