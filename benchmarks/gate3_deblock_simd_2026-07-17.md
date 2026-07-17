# Gate-3 deblock-SIMD — callgrind Ir deltas (2026-07-17)

Whole-decode instruction-reference (Ir) counts measured with valgrind callgrind
on the headline 2K photographic stills-decode cells, **port side**, before and
after the deblock loop-filter SIMD (the four `aom_highbd_lpf_*` filter widths
lane-batched into one `i32x4` magetypes kernel, AVX2 / `X64V3` tier). Ir is
deterministic, so these are load-independent (the box was busy running the
both-dispatch-mode test suite; wall-clock is a separate quiet-box concern).

The "before" is `2d3831f` — it already carries the transform-SIMD landing plus
the earlier decode SIMD (cdef, txb, intra-edge), so the deblock loop filter was
the **top remaining scalar decode hotspot** (~17-25% of decode Ir on these
`aomenc --allintra` stills, where CDEF/LR are off so deblock is the only live
post-filter). The delta below is the deblock SIMD **alone** — the deblock
frame-walk parameter derivation (`set_lpf_parameters`, `get_filter_level`,
`get_transform_size`) is byte-identical Ir in both binaries (it was NOT
vectorized; see follow-ups), so the entire whole-decode delta is the four
filter kernels.

## Measured Ir (`dec` cell, N=3 → 1 C-oracle + 4 port decodes; the C decode is
identical in both binaries and cancels in the delta)

| cell | before (`2d3831f`, scalar deblock) | after (deblock SIMD) | Δ whole-decode |
|---|---|---|---|
| `dec_mosaic_2k_cq20` (qindex 80, coeff-heavy)  | 3,374,647,537 | **3,024,137,482** | **−10.4 %** |
| `dec_mosaic_2k_cq40` (qindex 160, deblock-heavy) | 2,372,417,902 | **2,069,029,723** | **−12.8 %** |

The **deblock filter-kernel cluster** itself (the sum of the scalar
`highbd::{filter4,filter6,filter8,filter14,lpf_4,lpf_6,lpf_8,lpf_14}` before,
vs the one SIMD `simd::__arcane_lpf_impl_v3` after, each including the std-lib
lines callgrind attributes via inlining):

| cell | before kernels | after kernels (SIMD) | Δ kernels |
|---|---|---|---|
| `dec_mosaic_2k_cq20` | 645,428,784 | 290,949,096 | **−54.9 %** |
| `dec_mosaic_2k_cq40` | 579,726,584 | 272,895,088 | **−52.9 %** |

The kernel delta (−354.5 M / −306.8 M) equals the whole-decode delta
(−350.5 M / −303.4 M) to ~1 % — confirming the win is entirely the deblock
filter kernels and nothing else moved (the ~1 % residual is code-layout noise
between two independently-compiled binaries). cq40 (higher qindex → fewer
coefficients, more of the frame deblocked) shows the larger share, as expected.

## SIMD confirmed live (callgrind_annotate, after, `dec_mosaic_2k_cq40`)

The AVX2 tier fires — the ONE dispatched deblock kernel and its intrinsics are
attributed in the profile, and NO scalar `highbd::filter*` / `highbd::lpf_*`
remain:

- `simd::__arcane_lpf_impl_v3` **10.96 %** (226.7 M self) + its ssse3/avx2 lane
  intrinsics (`core_arch/src/x86/ssse3.rs:__arcane_lpf_impl_v3` 0.95 %, …).
- Frame-walk parameter derivation UNCHANGED (identical Ir before/after):
  `set_lpf_parameters` 83,204,600 (both), `get_filter_level` 22,643,040 (both),
  `get_transform_size` 21,481,744 (both) — proof only the kernels changed.
- `assert_byte_exact` (port-decode == C-decode) passed in the after binary on
  both cells → the SIMD decode is byte-identical to the C oracle in the real
  decode, not just the unit differential.

## What is SIMD vs still scalar (honest fraction)

- **SIMD (this landing):** all four highbd deblock filter widths — `lpf_4`
  (filter4 base), `lpf_6`, `lpf_8` (2-tap wide + flat blend), `lpf_14` (3-way
  wide/wide/base blend). One `i32x4` kernel; the 4 edge positions of each 4-px
  segment are the 4 lanes. Covers the whole `~580-645 M` filter-kernel cluster.
- **Still scalar (documented follow-ups):**
  1. **`set_lpf_parameters` + `get_filter_level` + `get_transform_size`**
     (~127 M, ~4-6 % of decode) — the per-edge parameter derivation (mode-info
     grid lookups, tx-boundary tests, level fallback). Branchy per-edge, not a
     lane-parallel pixel kernel; libaom/rav1d compute whole-region masks here.
     A real but separate vectorization (whole-strip mask derivation).
  2. **lowbd 8-bit lane path** — the decoder runs the HIGHBD (u16, i32-lane)
     filters at every bit depth. For bd8 (pixels 0..255) the wide-14 sums fit
     i16, so an i16-lane path would get ~2× the lanes/register (dav1d/rav1d keep
     separate bitdepth paths). This kernel is i32-lane for all bd (bd10/12 need
     it); the bd8 i16 specialization is a follow-up. It must stay bit-identical
     for bd8 and keep the i32 path for bd10/12.
  3. **AVX-512 (`X64V4`, 16-lane) + NEON** — this kernel is `#[magetypes(v3,
     neon, wasm128)]`, so NEON/WASM tiers exist but the perf box is x86; AVX-512
     is a wider follow-up.

## Reproduce

```
# before binary: build gate3_profile at 2d3831f (scalar deblock)
# after  binary: build gate3_profile at 1ce19a5 (deblock SIMD)
valgrind --tool=callgrind --callgrind-out-file=<out> \
  ./target/profiling/gate3_profile dec port dec_mosaic_2k_cq20 3
callgrind_annotate --auto=no --inclusive=no <out>
```

Raw callgrind outputs: NOT committed (>300 KB each); regenerate as above. The
`mosaic-2k-*.ivf` cells are gitignored (regenerable per
`benchmarks/decode_hotspots_2026-07-17.md`).
