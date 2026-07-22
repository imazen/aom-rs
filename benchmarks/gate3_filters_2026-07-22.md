# Gate-3 filter cells — fresh post-Phase-B/C filter-share ranking (2026-07-22)

The FILTER-CELLS lever's required first deliverable: a fresh callgrind re-profile
of the filter-dominated cells on current `origin/main` `1616a1ae` (which includes
the bd8 lowbd pipeline Phases A+B AND the Phase-C i16 transform). The prior
filter shares (task brief: "q32 wiener ~12%, loopfilter ~8%, cdef ~7%") predate
those landings; this ranking supersedes them. Nothing here is projected; every
number is a fresh callgrind Ir measurement on this box, method identical to
`gate3_decode_profile_2026-07-19.meta` (port = inclusive Ir of
`decode_frame_obus` ÷ port-decode count; C = inclusive Ir of
`shim_decode_av1_kf` ÷ 1; no `-C target-cpu=native`; the harness byte-verifies
every cell in setup before profiling).

## Entry-inclusive ratios (current main)

| cell | port Ir/decode | C Ir/decode | **Ir ratio** | prior (07-22 wired / 07-19) |
|---|---:|---:|---:|---|
| `dec_mosaic_4k_cq20` | 2,684,460,257 | 1,507,323,689 | **1.781x** | 1.975x (07-19 post-chunks) |
| `dec_mosaic_4k_cq40` | 1,820,303,428 |   845,651,396 | **2.153x** | 2.321x |
| `dec_mosaic_2k_cq20` |   550,209,547 |   281,964,907 | **1.951x** | 2.156x |
| `dec_mosaic_2k_cq40` |   399,886,721 |   173,830,146 | **2.301x** | 2.469x |
| `dec_352x288_q32`    |    52,713,297 |    24,223,496 | **2.176x** | 2.184x |

(The 4K ratios reproduce the Phase-C landing's 1.781x/2.153x to the third
decimal — a live method-stability check; the C side matches the 07-19/07-22
baselines to the digit.)

## THE RANKING — what actually dominates the filter share NOW

### Finding 1 (the surprise): the 2K/4K mosaic cells contain NO CDEF, NO LR, NO superres

On all four mosaic cells, inclusive `run_post_filters` == inclusive
`apply_deblock` to within 180 Ir: the aomenc-encoded mosaic streams carry no
CDEF syntax, RESTORE_NONE on every plane, and no superres. **Deblock is the
ONLY post-recon filter running in the Gate-3 headline cells.** Wiener/SGR and
CDEF levers cannot move the 4K wall gate at all on this bench set; they are
q32-regime (and conformance-corpus) levers.

### Finding 2 — deblock residual: a stable 1.73–1.77x across every cell (the top gate-cell item)

| cell | port deblock Ir/dec | C `av1_loop_filter_frame_mt` | ratio | gap | gap as % of cell's total port−C gap |
|---|---:|---:|---:|---:|---:|
| 4K cq20 | 414,217,354 | 239,897,275 | 1.73x | +174.3 M | 14.8 % |
| 4K cq40 | 386,571,437 | 217,913,025 | 1.77x | +168.7 M | 17.3 % |
| 2K cq20 |  97,356,462 |  56,308,854 | 1.73x | +41.0 M | 15.3 % |
| 2K cq40 |  90,114,697 |  51,490,968 | 1.75x | +38.6 M | 17.1 % |
| q32     |   5,760,970 |   3,421,954 | 1.68x |  +2.3 M |  8.2 % |

Split at 4K cq40 (per decode): kernel `__arcane_lpf_impl_u8_v3` = 275.2 M over
533,426 calls = **516 Ir/call**; walk + per-edge param derivation = 111.4 M.
The port's kernel ALONE exceeds C's entire deblock stage (kernels + walk =
217.9 M). Inside the kernel (callgrind line annotation, 4K cq40 totals):

* scalar 4-byte tap gather loads (`buf[(c + k*ts)..]` x4 lanes): 236.7 M = 21.5 %
* scalar 4-byte scatter stores: 176.5 M = 16.0 %
* `#[magetypes]` prologue/frame line: 67.2 M = 6.1 %
* remainder ≈ the i32x4 filter math + masks.

The shape gap vs C: C's lowbd SSE2 kernels load tap rows as vectors
(transpose for vertical edges), pack 2 taps per 128-bit register in i16
lanes, and store vectors; the port does one i32x4 vector PER TAP with 4
scalar u8 loads + 4 scalar stores each and a fresh `incant!` dispatch per
4-px edge segment.

### Finding 3 — q32 (the worst-ratio filter cell): LR/wiener is 9.25x, CDEF 4.11x

q32 per-decode filter stages (port ÷13 decodes vs C):

| stage | port Ir/dec | % of port decode | C Ir/dec | ratio | gap |
|---|---:|---:|---:|---:|---:|
| LR (`apply_restoration`) | 9,876,662 | 18.7 % | 1,067,465 (`av1_loop_restoration_filter_frame`) | **9.25x** | +8.8 M |
| CDEF (`apply_cdef`) | 8,050,525 | 15.3 % | 1,958,180 (`av1_cdef_frame`) | **4.11x** | +6.1 M |
| deblock (`apply_deblock`) | 5,760,970 | 10.9 % | 3,421,954 | 1.68x | +2.3 M |
| all post-filters | 23,814,339 | 45.2 % | ~6.4 M | 3.7x | +17.4 M |

The q32 total gap is 28.5 M/decode; the three filter stages are +17.4 M = 61 %
of it.

LR detail: the stream is all-Wiener (`filter_unit` 111.98 M total ==
`wiener_impl_v3` 111.41 M — no SGR units). Kernel-vs-kernel:
`__arcane_wiener_impl_v3` 8,570,208/dec vs C `av1_wiener_convolve_add_src_avx2`
758,348 = **11.3x**. Attribution inside the port wiener: per-call
`vec![0u16; (h+7)*128]` temp allocation 1,094,970/dec (12.8 % of the kernel);
scalar `from_fn` u16→i32 widen loads (9 per 8-wide output vector per pass) and
scalar `to_array` stores make up the bulk — C's AVX2 lowbd kernel runs 16-wide
i16 `madd` lanes with vector loads/stores. The take_wide/put_wide LR delegation
copies are ONLY 0.25 M/dec at 352x288 (apply_restoration − filter_frame) —
the delegation-copy hypothesis is REFUTED at this size; the kernel shape is
the whole story.

CDEF detail: `cdef_filter_block_16` 4,490,626/dec + `cdef_find_dir`
2,486,880/dec (the find_dir is scalar — C uses `cdef_find_dir_avx2`); the
apply_cdef − cdef_frame delegation widen/narrow = 0.36 M/dec (small). The
u16-domain filter kernel itself is the gap, NOT the (measured, kept)
delegation choice — attacking it does not touch the `cdef_frame_u8`-vs-delegate
decision.

Superres: absent from every profiled cell (no superres bench stream exists);
its cost is NOT measured here — no claim made.

## Attack order implied by the measurement

1. **Deblock kernel shape** (the only gate-cell filter item; +174 M/+169 M per
   4K decode): vectorize the tap load/store paths (contiguous `[u8;4]` loads
   for horizontal edges, per-row fixed-window loads + in-register transpose
   for vertical), hoist the per-call dispatch, keep the i32x4 arithmetic
   byte-identical.
2. **Wiener kernel** (q32; 11.3x kernel ratio): reusable scratch (kill the
   per-call temp alloc), vector u16 loads/stores, then the C-lowbd i16-madd
   lane shape if still needed.
3. **CDEF `cdef_find_dir` SIMD** (q32; 2.5 M/dec scalar): port the C
   `cdef_find_dir_avx2` shape. Does NOT re-open the delegate decision.

## Provenance

Box: dedicated aom-rs workstation. Tree: jj workspace on `origin/main`
`1616a1ae` (no code changes at capture). Oracle: `upstream/` libaom v3.14.1
@ `03087864`, in-process `shim_decode_av1_kf`. Corpus:
`AOM_CONFORMANCE_DIR=/root/aom-rs/conformance/data` (244 `.ivf` incl. the 4
regenerated mosaic streams). Commands: exactly
`gate3_decode_profile_2026-07-19.meta`'s (`gate3_profile dec port <cell> <N>`,
N=12 for q32, N=3 for the mosaics). Raw callgrind outs: `/tmp/g3cg/`
(ephemeral, >300 KB each, regenerate per the meta).
