# Gate-3 DECODE re-profile — current hotspot ranking + port-vs-C module gap (2026-07-19)

The prior Gate-3 decode ranking (`gate3_intra_simd_2026-07-18.md`,
`decode_hotspots_2026-07-17.md`) predates the inter-decode landings, the KB-1
q62/q63 txb reorder + CDEF stamping fix, and the KB-14 superres header fix. This
re-profile replaces it. **The old plan's premise — "the remaining small kernels
are ~6.6 % of profile" — is superseded**: the top of the profile has moved.

The perf track was stood down mid-flight (user deferred perf work). This file is
the measurement salvage: it is a complete, reusable ranking, and three landed
optimizations are measured against it. Nothing here is projected or estimated.

## Method (what makes this different from the earlier port-only rankings)

`gate3_profile dec port <cell> <N>` runs **one C-oracle decode** (the harness's
byte-exactness setup) followed by **N+1 port decodes**, all inside one callgrind
run. libaom is statically linked into the same binary, so object-file splitting
does not work; instead the two sides are separated by walking the call graph:

* **Authoritative ratio** = inclusive Ir of `aom_decode::frame::decode_frame_obus`
  (÷ port decode count) over inclusive Ir of `shim_decode_av1_kf` (÷ 1). Both are
  exact, and Ir is deterministic and load-independent, so this is reproducible on
  a noisy box where wall-clock is not.
* **Per-module rollup** classifies each symbol by name (Rust `::` ⇒ port, plain C
  identifier ⇒ libaom) and attributes shared libc leaves (memset/memcpy/calloc)
  through their *caller* edges, which callgrind records with exact inclusive cost.

Caveat, stated because it matters: the module rollup's **C** column sums to more
than the C entry-inclusive total (1,703.8 M vs 1,507.3 M on `dec_mosaic_4k_cq20`)
because some libc-leaf edges are attributed to C-named symbols outside the decode
subtree. Use the module rollup for **relative shares and per-module ratios**; use
the entry-inclusive number for the ratio itself.

The C oracle side also includes codec init/destroy (`ref_decode_av1_kf` =
init + decode + plane copy + destroy), which inflates C slightly and therefore
*understates* the port/C ratio — most visibly on the tiny cells. The committed
wall-clock gate numbers have always measured the same thing, so the two are
consistent.

## Ratio, per cell (baseline = `93228daf`, before this session's changes)

| cell | port Ir/decode | C Ir/decode | **Ir ratio** | wall-clock ratio |
|---|---:|---:|---:|---:|
| `dec_mosaic_4k_cq20` | 3,183,499,464 | 1,507,323,206 | **2.112x** | 1.45x |
| `dec_mosaic_4k_cq40` | 2,060,645,369 |   845,650,902 | **2.437x** | 1.39x |
| `dec_mosaic_2k_cq20` |   642,948,632 |   281,964,424 | **2.280x** | 1.91x |
| `dec_mosaic_2k_cq40` |   447,417,855 |   173,829,515 | **2.574x** | 2.04x |
| `dec_352x288_q00`    |   125,323,250 |    87,571,566 | **1.431x** | 1.33x |
| `dec_352x288_q32`    |    59,815,656 |    24,221,718 | **2.470x** | 1.78x |
| `dec_352x288_q63`    |    22,823,595 |     7,930,363 | **2.878x** | 3.10x (wide CI) |
| `dec_196x196`        |    14,154,805 |     6,779,319 | **2.088x** | 2.45x |

Wall-clock is the paired zenbench decode group run on this box the same day
(raw in `gate3_decode_profile_2026-07-19.zenbench.txt`). Every decode cell
gate-truncated to 4 rounds, exactly as the 2026-07-17 baseline did — that is a
structural property of this shared box, not a transient. **The 4K cells are the
trustworthy wall-clock numbers**; `dec_352x288_q63` swung +136.6 %..+483.6 % and
must be quoted as a band.

**Gate 3 target is ≤ 1.20x. The headline 4K cells are 1.45x / 1.39x wall-clock —
the same band the 2026-07-17 measurement reported (1.50x / 1.41x). The gate is
NOT met and was not met by this session's work.**

The port executes **~2.1x the instructions** of C at 4K while running only 1.45x
the wall-clock — i.e. the port's IPC is materially better, and instruction count
is the thing to attack.

## Where the gap actually is (`dec_mosaic_4k_cq20`, self-Ir rollup)

| module | port Ir/dec | % of port | C Ir/dec | **port/C** | absolute gap |
|---|---:|---:|---:|---:|---:|
| coeff/txb        | 917,242,418 | 28.8 % | 541,853,496 | 1.69x | **+375.4 M** |
| transform        | 454,471,276 | 14.3 % |  92,751,604 | **4.90x** | **+361.7 M** |
| loopfilter       | 507,448,364 | 15.9 % | 239,628,064 | 2.12x | **+267.8 M** |
| intra-pred       | 301,556,908 |  9.5 % |  94,470,183 | **3.19x** | **+207.1 M** |
| entropy (od_ec)  | 480,692,670 | 15.1 % | 343,466,100 | 1.40x | +137.2 M |
| decode driver    | 216,169,432 |  6.8 % | 123,056,816 | 1.76x |  +93.1 M |
| alloc (libc)     | 273,620,762 |  8.6 % | 220,257,302 | 1.24x |  +53.4 M |
| recon            |  12,637,360 |  0.4 % |  16,857,539 | 0.75x |   −4.2 M |

The ranking is stable across all four mosaic cells (see the CSV); at cq40
(higher qindex, fewer coefficients) transform/loopfilter/intra-pred rise and
coeff/txb falls, as expected.

### Finding 1 — the biggest *ratio* gaps are all the bd8 lowbd-vs-highbd lane width

C decodes 8-bit content with **lowbd kernels**: `lowbd_inv_txfm2d_add_no_identity_avx2`
(i16 lanes, u8 output), `aom_lpf_*_sse2` (u8), `build_non_directional_intra_predictors`
(u8). The port runs the **highbd (u16 buffer, i32 lane)** pipeline at every bit
depth. That single structural difference is visible directly in the three worst
ratios — transform 4.90x, intra-pred 3.19x, loopfilter 2.12x — which together are
**+836.6 M of the +1,676 M total gap, i.e. ~50 % of the entire Gate-3 shortfall.**
This confirms the previously-guessed "bd8 i16-lane path" candidate and, for the
first time, quantifies it.

### Finding 2 — within `transform`, small transforms are fully scalar

`av1_inv_txfm2d_add` is entered ~126 k times per 4K decode and the profile shows
`inv_txfm1d_gen::av1_idct4` at ~274 k calls/decode plus `special::av1_iadst4` at
~138 k — i.e. the **scalar** 1-D kernels. `try_inv_row_pass` returns `false` when
`row_n % 8 != 0` and the column pass likewise, so **every 4-tall / 4-wide
transform bypasses the landed transform SIMD entirely**. On photographic 4K at
qindex 80 those are a large share of all transforms. A 2-rows-per-`i32x8` path
(the shape the CDEF w4 kernel already uses) is the obvious next kernel.

### Finding 3 — allocation was real, but it is NOT the "arena" story it looked like

Per 4K decode the port made ~126 k transform-block allocations. Measured
allocator traffic before this session: `__rust_alloc_zeroed` 115.9 M + `__rust_dealloc`
37.6 M + `__rust_alloc` 9.2 M = **162.7 M Ir/decode**, driven by
`av1_inv_txfm2d_add` (52.3 M — its row-pass `buf` *and* `remap_input`'s copy),
`TileKf::decode_intra_block_body` (36.6 M, the per-block `tcoeff` vectors) and
`reconstruct_txb` (23.1 M, `dqcoeff`).

But **C's own allocation share is proportionally larger, not smaller** (12.9 % of
the C decode vs 8.6 % of the port's), so allocation was never going to close the
gate on its own — worth fixing, and now largely fixed (below), but it is a ~2 %
lever, not a ~20 % one. Recording this explicitly because the pre-session plan
named "arena allocation" as one of two headline candidates.

### Finding 4 — the per-coefficient context helpers were not being inlined

`get_nz_mag` and `get_lower_levels_ctx` are `AOM_FORCE_INLINE`/static-inline in C
(`txb_common.h`) but carried only a plain `#[inline]` hint here, which LLVM
declined — `nm` showed real out-of-line symbols. Measured **102.7 Ir per
`get_lower_levels_ctx` call, inclusive, over 2.85 M calls/decode = 9.2 % of the
whole decode**, of which ~9 Ir was function prologue alone.

## What landed against this baseline (three chunks, all byte-exact)

1. **Force-inline the per-coefficient txb context chain** — `get_nz_mag`,
   `get_lower_levels_ctx`, `get_br_ctx`, `get_nz_map_ctx_from_stats`,
   `get_padded_idx`, `min3`, `tables::nz_map_ctx_offset`.
2. **Reuse the per-transform-block recon scratch** — new `ReconScratch` /
   `reconstruct_txb_into` and `InvTxfmScratch` / `av1_inv_txfm2d_add_into`
   (additive API; the existing entry points keep their signatures and behaviour,
   so `aom-encode` is untouched). `TileKf` owns one scratch across its five
   reconstruction call sites.
3. **Stop `remap_input` allocating + copying** — only the five 64-point-family
   tx sizes expand a 32-capped coded region; every other size now borrows the
   packed input instead of `vec![...]` + `copy_from_slice` per transform block.

Reuse is byte-identical by construction, verified by reading both code paths:
`dequant_txb` zero-fills its whole output before writing, and the transform row
pass writes every element of `buf` before the column pass reads it (the SIMD path
stores all `row_n * col_n` positions and declines outright when `row_n % 8 != 0`,
falling back to the scalar loop which also writes every row).

| cell | Ir/dec before | Ir/dec after | Δ | Ir ratio before → after |
|---|---:|---:|---:|---|
| `dec_mosaic_4k_cq20` | 3,183,499,464 | 2,977,417,517 | **−6.47 %** | 2.112x → **1.975x** |
| `dec_mosaic_2k_cq20` |   642,948,632 |   607,794,643 | −5.47 % | 2.280x → 2.156x |
| `dec_352x288_q00`    |   125,323,250 |   118,813,802 | −5.19 % | 1.431x → 1.357x |
| `dec_mosaic_4k_cq40` | 2,060,645,369 | 1,962,410,316 | −4.77 % | 2.437x → 2.321x |
| `dec_mosaic_2k_cq40` |   447,417,855 |   429,142,838 | −4.08 % | 2.574x → 2.469x |
| `dec_352x288_q32`    |    59,815,656 |    57,489,692 | −3.89 % | 2.470x → 2.373x |
| `dec_196x196`        |    14,154,805 |    13,823,005 | −2.34 % | 2.088x → 2.039x |
| `dec_352x288_q63`    |    22,823,595 |    22,423,946 | −1.75 % | 2.878x → 2.828x |

**No post-change wall-clock re-measure was taken** (the track was stood down
before it could run). Do not quote a wall-clock delta for these chunks — only the
instruction-count deltas above are measured.

## If/when the perf track resumes — ranked by measured evidence

1. **bd8 lowbd lane path** (~50 % of the gap; transform + intra-pred + loopfilter).
   Structurally large: the decoder's frame buffers are u16 everywhere, so this
   means a real second pipeline, exactly as C has.
2. **4-wide / 4-tall inverse transform SIMD** (Finding 2) — a self-contained
   kernel, no pipeline change, and it is the largest single ratio gap (4.90x).
3. **Remaining allocation** — `TileKf::decode_intra_block_body`'s per-block
   `tcoeff` / `tcoeff_uv` vectors (44.7 M Ir/dec after the landed chunks) and the
   `DecodedBlockKf::txbs` `Vec` push growth (~15.5 M Ir/dec, wants
   `with_capacity`). Both are aom-decode-local.
4. **`read_txb_body`'s `levels_buf`** zeroes the full `TX_PAD_2D` (1312 B) on
   every coded txb; C zeroes only `(height + TX_PAD_HOR) * (width + TX_PAD_VER)
   + TX_PAD_END` and only when `eob > 1`, using `get_br_ctx_eob` (which reads no
   levels) for the eob coefficient. Measured small (11.6 M Ir/dec, 0.36 %) — do
   not prioritise it, but the faithfulness gap is real.

Reproduce any row:

```
cargo build --profile profiling -p zenav1-aom-bench --bin gate3_profile
valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out \
    ./target/profiling/gate3_profile dec port dec_mosaic_4k_cq20 3
callgrind_annotate --auto=yes --threshold=75 /tmp/cg.out
```

Raw callgrind out-files are NOT committed (>300 KB each); regenerate as above.
The `mosaic-{2k,4k}-*.ivf` cells are gitignored (regeneration documented in
`benchmarks/decode_hotspots_2026-07-17.md`).
