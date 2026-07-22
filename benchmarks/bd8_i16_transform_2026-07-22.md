# bd8 Phase C — i16-lane inverse-transform column pass, measured (2026-07-22)

The "second phase" lever named in `lowbd_txfm_foundation_2026-07-22.md` and
`bd8_lowbd_wired_2026-07-22.md`: the bd8 u8 COLUMN pass now runs the five DCT
column kernels (idct4/8/16/32/64) on `i16x16` lanes — 16 columns per AVX2
vector, 2x the lane throughput of the i32x8 pass — landed as `1d29acaf`.
Nothing here is projected; every number is a same-environment paired
measurement (see the `.meta` for the method-stability digits).

## What is narrowed (coverage stated as fractions)

**5 of 12 column 1-D kernels are i16-narrowed:** idct4, idct8, idct16,
idct32, idct64 — every column whose vertical transform is DCT, any width
(4/8/16/32/64 all dispatch to the i16 pass).

**7 of 12 are NOT narrowed and stay on the byte-identical i32x8 pass, each
with a mechanical reason** (from `xtask/audit_i16_safety.py` over the
generated scalar kernels — the audit that also PROVED the five DCT kernels
safe):

| kernel | why not i16 |
|---|---|
| iadst8, iadst16 | terminal stage emits UNCLAMPED `wrapping_neg()` of i16 values (+32768 possible) AND raw 17-bit butterfly transients straight into the round-shift — an i16 lane cannot hold either |
| iadst4 | no internal clamps at all (sinpi sum form, outputs to ~3·2^15) |
| iidentity4/8/16/32 | no clamps; the NewSqrt2 / x2 / x4 multiplies produce 17-18-bit unclamped terminals |

**The row pass is untouched (0 narrowed)** — it is shared i32 with highbd and
was explicitly out of Phase C scope; it remains a candidate follow-up lever
(same audit machinery applies, but the row input clamp is bd+8 and identity
row outputs overflow i16 the same way).

## Why it is byte-identical (the design, in one paragraph)

This is NOT libaom's lowbd SIMD shape (which packs-saturates after every
butterfly — a deviation from the scalar `_c` semantics on hostile inputs).
Two value domains, statically tracked by the `--lanes16` transpiler: i16
values (kernel inputs + every `clamp_value` output) live in i16 lanes where
saturating add/sub IS the normative `clamp_value(_,16)`; the UNCLAMPED
half_btf outputs (bounded 17-bit transients) live as exact i32 pairs in
unpack order, added exactly and only then saturate-packed at the scalar
kernel's clamp sites. `madd`-based butterflies are exact for |w|<=4095,
|in|<=2^15 (products <=2^27 — no wrap on either side); `round_shift(_,4)` is
`mulhrs(v, 2^11)`, proven exact for all i16. Gates: the new per-kernel
differential (full i16 domain + saturation boundary patterns, every token
permutation), `inv_txfm2d_lowbd_diff` (u8 == real C == highbd port), full
aom-dsp 353/353 and decode conformance 62/62 — all in BOTH default and
`AOM_FORCE_SCALAR=1` dispatch.

## Callgrind Ir — microbench (`lowbd_txfm_profile u8 200`)

Method stability: the u16 control and the re-built 7b972e5 u8 baseline BOTH
reproduce the foundation doc to the digit (1,683,395,100 / 855,800,916).

| metric (inclusive Ir) | baseline 7b972e5 | Phase C 1d29acaf | delta |
|---|---:|---:|---:|
| `av1_inv_txfm2d_add_u8_into` | 1,695,097,484 | 1,433,681,703 | **-15.4%** |
| u8 column pass (total) | 855,800,916 | 585,909,372 | **-31.5%** |
| — of which DCT columns | 473,562,432 | 203,670,888 | **-57.0%** |
| — of which iadst/identity (i32, unchanged) | 382,238,484 | 382,238,484 | 0 |
| u8 vs u16-highbd entry | +0.7% (worse) | **-14.8% (better)** | — |

The foundation doc's prediction held: the u8-storage step was Ir-neutral, the
i16 lane narrowing is the instruction-count win, and it is concentrated
exactly where predicted (the DCT butterflies halve and better).

## Callgrind Ir — decode cells (same-environment paired, port Ir/decode)

C-side Ir is IDENTICAL TO THE DIGIT between baseline and Phase C runs on
every cell — the strongest available control.

| cell | base | Phase C | Δ port | port/C ratio |
|---|---:|---:|---:|---:|
| `dec_352x288_q00` | 98,769,685 | 98,769,692 | **0.00%** (+7 Ir) | 1.128x (unchanged) |
| `dec_352x288_q32` | 52,842,935 | 52,713,297 | −0.25% | 2.182x → 2.176x |
| `dec_mosaic_4k_cq20` | 2,720,315,015 | 2,684,460,257 | **−1.32%** | 1.805x → **1.781x** |
| `dec_mosaic_4k_cq40` | 1,868,062,048 | 1,820,303,428 | **−2.56%** | 2.209x → **2.153x** |

Reading, honestly: **`q00` is the LOSSLESS vector — its recon is 100% 4x4
WHT and the DCT column pass never runs**, so Phase C is structurally inert
there (+7 Ir = the no-regression control, and the profile shows zero
`av1_inv_txfm2d_add_u8_into` frames). The prior docs' framing of q00 as the
transform-heavy Ir cell does not hold for THIS lever; the 4K photographic
mosaics are where the DCT transform lives, and they move −1.3%/−2.6% whole-
decode. The transform column pass is simply not a large whole-decode Ir
fraction (the entropy decoder dominates); the microbench isolates the kernel
win precisely.

## Wall clock — NOT cleanly measured this session (deliberate)

A paired zenbench run (`cargo bench --bench gate3 -- --group=dec`) was
started but the box carried concurrent heavy work (the coordinator's
independent conformance verification of this landing, plus a ~1-core foreign
trace job); zenbench kept rejecting noisy rounds and the 4K cells never
completed cleanly. The run was killed on coordinator direction rather than
report contention-fouled numbers. **No Phase-C wall figure is claimed.** The
wall headline therefore stands at the Phase-B measurement (1.286x cq20 /
1.250x cq40, `bd8_lowbd_wired_2026-07-22.zenbench.txt` @7b972e5) until a
quiet-box run; the deterministic, load-independent Ir deltas above are the
Phase-C perf evidence. For a transform-only change, Ir is the reliable
number: the i16 pass executes strictly fewer instructions per DCT column
(−57%), and wall can only benefit or stay flat — but that expectation is NOT
a measurement and is not reported as one.

## Gate-3 posture

**Ir movement: 4K cq20 port/C 1.805x → 1.781x, 4K cq40 2.209x → 2.153x**
(whole-decode instruction ratios; the wall-clock Gate-3 headline is
unchanged-as-measured at 1.286x/1.250x pending a quiet-box wall run).
Target <= 1.20x wall not yet re-assessed this session. Remaining named
levers, measured-priority order:
1. The 2K-regime gap (1.73x/1.90x wall; q32 2.18x Ir) — entropy/coeff-
   dominated; profile before choosing (`rav1d_borrow_leads_2026-07-19.md`
   candidates).
2. Row-pass i16 narrowing (the audit machinery + transpiler mode now exist;
   the row clamp is bd+8 so the same per-kernel audit is required).
3. CDEF-heavy small-stream regime — bounded by the measured "CDEF is
   intrinsically u16" finding; wins there are NOT lowbd-shaped.

## Provenance

Box: dedicated aom-rs workstation. Tree measured: `1d29acaf` (= origin/main
at capture) vs `7b972e5`, both built in the `zenav1-aom--bd8-planeC` jj
workspace. Oracle: `upstream/` libaom v3.14.1 @ `03087864` (prebuilt
`libaom.a` shared read-only). Corpus: `AOM_CONFORMANCE_DIR=
/root/aom-rs/conformance/data` (244 vectors incl. the 4 mosaic streams).
Commands + honest limits: `bd8_i16_transform_2026-07-22.meta`. Callgrind
outs in `/tmp/g3cg/` (ephemeral); no zenbench log is committed because no
clean wall run exists (see the wall section).
