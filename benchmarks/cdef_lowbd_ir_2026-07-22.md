# bd8 CDEF lowbd — callgrind Ir (2026-07-22)

Measures the bd8 lowbd CDEF lever: `cdef_frame_u8` (the u8-plane frame walk)
against the alternative a bd8 tile would use WITHOUT it — DELEGATION: widen the
whole `u8` plane to `u16`, run the highbd `cdef_frame`, narrow back.

- **Machine:** dev-32gb; `rustc 1.97.0`.
- **Base:** `origin/main` b08be98 (the byte-identical `cdef_frame_u8` landing).
- **Build:** `cargo build --release -p zenav1-aom-bench --bin cdef_lowbd_profile`
  (release, **no** `-C target-cpu=native`, per the perf rules).
- **Tool:** `valgrind --tool=callgrind` (exact deterministic Ir), inclusive
  `summary:` total.
- **Command:** `valgrind --tool=callgrind ./target/release/cdef_lowbd_profile <side> 40`
- **Workload:** 3 frames (256², 512², 384×320), 4:2:0, CDEF ON (mixed nonzero
  Y/UV strengths, damping 4), filter-heavy skip (~1/16 mi skipped — the q32
  "filter-dominated" regime), every 64×64 unit strength-enabled. 40 iterations
  → 120 frame-passes. The profiler cross-checks u8 == delegate byte-identity
  before profiling (a corrupt build is never measured).

## Totals (inclusive Ir, 40 iters × 3 frames)

| side | total Ir | vs delegate | per frame-pass |
|---|---:|---:|---:|
| `delegate` (widen→`cdef_frame`→narrow) | 2,514,875,954 | — | baseline |
| **`cdef_frame_u8` — direct-u8 SIMD store** | 2,681,010,928 | **+6.61%** | +1,384,458 |
| `cdef_frame_u8` — per-block u16 scratch (first cut) | 3,109,071,420 | +23.6% | +4,864,610 |

Direct-u8 store is **−13.8%** vs the scratch first cut (that cut is what was
byte-identically landed in b08be98; this file's measurement drove replacing it).

## Per-function attribution (`callgrind_annotate`, direct-u8 vs delegate)

Shared and identical on both sides (the u16-domain filter — CDEF cannot narrow
it): `cdef_find_dir` 451.5M, `cdef_filter_block_16/_u8` dispatch ~48.5M.

The whole 6.6% residual is TWO plane-touch costs, both irreducible in the
magetypes-generic kernel:

1. **u8 narrow store (+152M).** `cdef_filter_8_w8_impl` 1,138.5M vs the u16
   `cdef_filter_16_w8_impl` 1,001.7M (+136M / +13.6%); w4 +16M. The ONLY
   difference is the store: the u8 kernel does `y.to_array()` + 8 scalar byte
   stores (the vector spills to stack), where the u16 kernel does one vector
   store. magetypes 0.9.27 exposes `to_u8` only on FLOAT types — there is no
   integer `u16x8 -> u8x8` truncating pack (it is in magetypes FEATURE_REQUESTS),
   so a vectorised u8 store would need per-arch hand-intrinsics (out of the
   family's scope; a magetypes feature-request is the right fix).
2. **per-fb widen vs memcpy (+37M).** `cdef_frame_u8` 152.1M vs `cdef_frame`
   115.5M: priming the u16 `src`/linebuf from a u8 plane (element-wise widen)
   vs from a u16 plane (`copy_from_slice` memcpy). Delegation pays a single
   amortised whole-plane widen (44M, in `main`) instead.

## Conclusion

**CDEF is the exception to "direct-u8 wins".** Its filter is intrinsically
`u16` and dominates (~72%); the u8 plane only affects the small conversion tail,
where delegation's amortised whole-plane widen + fast memcpy prime + vector
store beat the direct-u8 per-fb widen + scalar narrow store. **Recommend the
tile-plane flip DELEGATE CDEF** (widen→`cdef_frame`→narrow); keep `cdef_frame_u8`
as the byte-identical reference / for callers avoiding the transient u16 plane.

Note: the kernel is DORMANT today (tile recon planes are still `Vec<u16>`; the
`ReconPlanes` u8 flip has not landed for any family), so this is a flip-time
kernel-microbench, not a decoder-cell delta — the decode path is byte- and
Ir-unchanged by this work (conformance corpus 240 vectors + golden MD5 green).
