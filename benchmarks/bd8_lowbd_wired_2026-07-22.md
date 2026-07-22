# bd8 lowbd pipeline WIRED (Phases A+B) — Gate-3 measurement, 2026-07-22

The follow-up to `bd8_pipeline_2026-07-22.md` ("0/4 families wired"): the
`ReconPlane` u8/u16 carrier (Phase A, `5336e657`) and the live u8 kernels
(Phase B, `43b7d603` + the salvaged deblock `3ca14956`/`1ae33eed`) are now ON
`origin/main` and this file measures them. Nothing here is projected; every
number is a fresh measurement on this box, method identical to the earlier
metas (callgrind Ir + zenbench paired wall, no `-C target-cpu=native`).

## What is LIVE-lowbd now (bd8 frames)

| family | status |
|---|---|
| intra prediction | **u8 kernels live** (`predict_intra_u8`, luma + non-CfL chroma, direct u8 plane reads) |
| recon (dequant+itx+add) | **u8 kernels live** (`reconstruct_txb_u8_into`, all 5 sites) |
| lossless WHT | **u8 live** (`reconstruct_txb_wht_u8` → `av1_iwht4x4_add_u8`) |
| intrabc copies / chroma bilinear / palette | **u8 live** (direct u8 arms) |
| deblock | **u8 frame walk live** (`loop_filter_frame_u8`, salvaged orphan) |
| CDEF | DELEGATED **by measurement** (`cdef_lowbd_ir_2026-07-22.md`: direct-u8 +6.61% Ir worse) |
| LR / superres | delegated (no u8 walk exists) |
| inter MC / OBMC / inter-intra | delegated (no u8 kernels exist) |
| CfL store + CfL chroma predict | delegated (the CfL AC add is intrinsically u16) |

bd10/12 run the untouched u16 pipeline. Byte-identity: full decode suite 62/62
in BOTH default and `AOM_FORCE_SCALAR=1` (240-vector corpus + golden MD5,
real_bitstream, sb128, multi-tile, superres, film-grain, inter, fuzz) + the
per-cell `assert_byte_exact` on every bench stream below (including the 2K/4K
photographic mosaics through the full lowbd pipeline).

## Callgrind Ir (vs the reconciliation's `d00b07ad` numbers — the only decode
## changes in between are Phases A+B, so this delta IS the lowbd effect)

| cell | port Ir/decode now | @d00b07ad | Δ port | C Ir | ratio now | was |
|---|---:|---:|---:|---:|---:|---:|
| `dec_352x288_q00` | 98,500,470 | 103,896,282 | **−5.19%** | 87,572,319 | **1.125x** | 1.186x |
| `dec_352x288_q32` | 52,911,892 | 53,136,560 | −0.42% | 24,223,274 | **2.184x** | 2.194x |

The C-side Ir matches the baseline to the digit (87,572,319 / 24,223,274) — a
live method-stability check. Reading: the coeff/intra-heavy cell gains a solid
−5.2% (live u8 intra + recon); the filter-heavy cell barely moves in Ir
(CDEF delegation still pays widen/narrow, and Ir does not see bandwidth).

## Wall clock (zenbench paired, `cargo bench --bench gate3 -- --group=dec`)

Baseline = `gate3_decode_profile_2026-07-19.zenbench.txt` (@`93228daf`).
Content identity: the gitignored mosaic vectors were REGENERATED this pass and
verified faithful — `mk_mosaic_y4m` (on main, `benchmarks/mk_mosaic_y4m.rs`)
over `/root/work/codec-corpus/gb82` reproduced the surviving 2K y4m
BYTE-IDENTICALLY, and the pinned `upstream/build/aomenc` re-encodes match the
recorded payload sizes exactly (2K 148127/54108 B, 4K 956186/322294 B).

| cell | port 07-19 | port NOW | Δ port | ratio 07-19 | **ratio NOW** (95% CI) |
|---|---:|---:|---:|---:|---:|
| `dec_mosaic_4k_cq20` | 340.4 ms | **278.1 ms** | **−18.3%** | 1.45x | **1.286x** (+27.1–30.2%) |
| `dec_mosaic_4k_cq40` | 190.6 ms | **176.2 ms** | −7.6% | 1.39x | **1.250x** (+21.9–28.7%) |
| `dec_mosaic_2k_cq20` | 59.6 ms | 55.6 ms | −6.7% | 1.90x | 1.675x (+63.2–71.5%) |
| `dec_mosaic_2k_cq40` | 38.5 ms | 36.4 ms | −5.5% | 2.03x | 1.916x (+84.6–103.5%) |
| `dec_352x288_q00` | 12.2 ms | 11.2 ms | −8.2% | 1.33x | 1.232x (+18.9–27.6%) |
| `dec_352x288_q32` | 5.7 ms | 4.6 ms | −19.3% | 1.77x | 1.895x (+88.3–90.7%) |

Caveats, stated plainly:
- Cross-run C-side wall drift exists (4K cq20 C 234.7→216.2 ms; q32 C 3.2→2.5
  ms with the baseline's C flagged noisy) — the box is not identical between
  sessions. The PAIRED within-run ratio is the gate metric; both columns are
  paired ratios. Every cell keeps zenbench's 4-round noise-gate truncation
  (same as both baselines).
- Attribution: the wall delta vs 2026-07-19 includes the entropy/BCE Ir levers
  (`a00aa51`/`879e24b`/`046b897`) AND lowbd. No wall measurement exists at
  `d00b07ad` to split them; the reconciliation carried 1.45x/1.39x forward as
  "unchanged" on the byte-identical-code argument, but that is an argument,
  not a measurement. The Ir table above IS cleanly attributed (both endpoints
  measured, only lowbd in between).

## Gate-3 posture

**4K headline: 1.286x / 1.250x** (was 1.45x / 1.39x). Target ≤ 1.20x is NOT
yet met but the gap more than halved on the headline cells. The named
remaining levers, in measured-priority order:
1. **i16 SIMD-lane transform narrowing** (`lowbd_txfm_foundation_2026-07-22.md`
   "second phase"): both 1-D passes on i16 lanes (i32 multiply-accumulate),
   ~2x lane throughput; byte-identity-safe at bd8 (`av1_gen_inv_stage_range`
   opt_range == 16 clamps every inter-stage value to i16). A kernel program:
   per-family i16 butterflies + differentials vs the C `_c` reference.
2. The 2K-regime gap (1.68x/1.92x) — entropy/coeff-dominated at 2K density;
   profile before choosing (the 4-tall/4-wide scalar-fallback transform gate
   noted in `rav1d_borrow_leads_2026-07-19.md` is a candidate).
3. CDEF-heavy small-stream regime (q32 2.18x Ir) — bounded by the measured
   "CDEF is intrinsically u16" finding; wins here are NOT lowbd-shaped.

## Provenance

Box: dedicated aom-rs workstation. Tree: `1ae33eed` (= origin/main at
capture). Oracle: `upstream/` libaom v3.14.1 @ `03087864`, in-process
`shim_decode_av1_kf` / `aomenc` from the same build. Corpus:
`AOM_CONFORMANCE_DIR=/root/aom-rs/conformance/data` (240 vectors + the 4
regenerated mosaic streams; regenerables backed up at `/root/mosaic-sources/`
+ `/root/conformance-backup/`). Commands: exactly those in
`bd8_pipeline_2026-07-22.meta` (Ir) and `gate3_decode_profile_2026-07-19.meta`
(wall). Raw logs: committed alongside as `bd8_lowbd_wired_2026-07-22.zenbench.txt`;
callgrind outs in `/tmp/g3cg/` (ephemeral).
