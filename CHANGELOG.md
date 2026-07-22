# Changelog

## Workspace

### [Unreleased]

### Fixed

- **Encoder intrabc (screen content): DV search + var-tx cost now match libaom to
  the unit at the KB-15 witness leaf mi(40,28)** — three independent roots, each
  localized by a byte-inert instrumented sibling-C dump (0cd64bf):
  1. the DV-search `error_per_bit` used the frame rdmult instead of the per-block
     `x->rdmult` (per-SB `intra_sb_rdmult_modifier` fold) — now
     `av1_set_error_per_bit(env.rdmult)`;
  2. the intrabc pixel search modelled NSTEP (12-point tangent stages) where
     libaom uses NSTEP_8PT (16 stages, 8-point, `tan=radius`) — the diamond is now
     parameterized by an `eight_pt` flag, intrabc passing NSTEP_8PT;
  3. the intrabc var-tx `txfm_partition_cost` was a frame constant instead of the
     per-SB (INTERNAL_COST_UPD_SB) value from the adapting `txfm_partition` CDF —
     `txfm_partition_costs` added to `RealCosts`/`SbEncodeEnv`.
  The port now finds C's exact `dv=(-816,-888)` and flips mi(40,28) to
  PARTITION_VERT matching C. Intrabc-only / per-SB-additive: intra envelope
  byte-inert (aom-encode+aom-bench 340/340). The witness stays PINNED (first-diff
  floor 1120) — the remaining byte-1120 divergence is a separate PACK-side residual.

### QUEUED BREAKING CHANGES

- **`zenav1-aom-decode`: `KfTileDecode.recon/recon_u/recon_v` are now
  `ReconPlane { LowBd(Vec<u8>), HighBd(Vec<u16>) }` instead of `Vec<u16>`**
  (bd8 frames store `u8` planes; `ReconPlane::to_u16()`/`px()` widen
  bit-exactly). `FrameDecode` and `RefFrame` stay `u16` — only consumers
  reading the pre-filter tile planes directly must migrate. (5336e65)

- **`zenav1-aom-decode` public entry points now return `Result<_, DecodeError>`
  instead of `Result<_, String>`.** `decode_frame_obus` / `decode_frames` (and
  the parse helpers) carry a structured, category-bearing `DecodeError` enum
  (implements `core::error::Error`; `pub use` of `DecodeError` + `LimitKind`).
  Consumers matching on the old `String` error must migrate to the enum. (c43440b)

### Added

- **bd8 decode Phase C: i16-lane inverse-transform column pass** — the u8
  column pass runs idct4/8/16/32/64 on `i16x16` lanes (16 columns per AVX2
  vector; two-domain design keeps the unclamped butterfly transients in exact
  i32 pairs so it is byte-identical to the scalar port, NOT the libaom lowbd
  saturate-early shape). iadst/identity columns stay i32 (audited not
  i16-safe: `xtask/audit_i16_safety.py`). Measured: DCT columns −57% Ir,
  whole column pass −31.5%, 4K decode −1.3%/−2.6% Ir; see
  `benchmarks/bd8_i16_transform_2026-07-22.md`. (1d29acaf)

- **`zenav1-aom-decode` production-hardening surface** (deliberate API additions
  for the untrusted-input / zenavif decode path):
  - `DecodeConfig` / `DecodeLimits` threaded through `decode_frame_obus_with` /
    `decode_frames_with` / `_prefilter_with` — bounded resource limits for
    untrusted bitstreams. (e25c556)
  - Cooperative cancellation via `enough::Stop`, polled per SB-row / tile /
    frame → `DecodeError::Cancelled`. (e6c7795)
  - Optional `whereat` feature (default OFF) adding `*_at` source-located error
    entries. (edaf579)
  - `AllocMode` fallible-alloc pre-flight (`try_reserve` probe → `AllocFailed`)
    + `max_memory_bytes` enforcement — a byte-preserving allocation ceiling
    against attacker-controlled dimensions. (70b50c6)
  - Malformed-input hardening: frame-dimension DoS ceiling (reject >2^28 px
    before recon alloc) + panic→`Err` conversions found by a structured-random
    fuzz sweep + a stable-toolchain fuzz regression harness. (1b65d61, 88b4de3,
    606813d, 5922c47, bbd7bc4)
  Decode output is byte-identical on valid input (the error type is a rename;
  limits / stop / whereat / alloc all default to unchanged behavior).

### Changed

- **Decoder bd8 lowbd Phase B: the u8 kernels are LIVE** — bd8 frames now
  decode through `predict_intra_u8`, `reconstruct_txb_u8_into`,
  `av1_iwht4x4_add_u8`, u8 intrabc/palette stores (43b7d60), and the salvaged
  `loop_filter_frame_u8` deblock walk (3ca1495, 1ae33ee). CDEF stays on the
  byte-identical widen/narrow delegation by measurement (direct-u8 is +6.61%
  Ir worse); LR/superres/inter-MC/CfL keep delegation (no u8 kernels).
  Output bit-identical at every bit depth (full decode suite, default +
  `AOM_FORCE_SCALAR=1`).

- **Decoder bd8 recon planes are stored as `u8` (`ReconPlane::LowBd`), Phase A
  of the lowbd pipeline** — every kernel still runs the unchanged highbd path
  via byte-identical widen/narrow delegation (no u8 kernel wired yet), so
  decoded output is bit-identical at every bit depth (full decode suite green
  in default + `AOM_FORCE_SCALAR=1`); bd10/12 keep `u16` planes untouched.
  Phase B swaps the delegation arms for the landed `*_u8` kernels. (5336e65)

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
