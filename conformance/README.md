# AV1 decode-conformance corpus (Gate 1)

This directory defines **Gate 1's authoritative test set**: the official AV1
decode-conformance vectors that libaom's own decode tests assert against. Our
decoder passes Gate 1 when it reproduces every vector's per-frame output
byte-for-byte (verified via the shipped golden MD5s) across the full corpus.

Until now the decoder track has been validated against *synthetic* streams
generated through the C shim (`crates/aom-decode/tests/real_bitstream.rs`).
Those remain valuable, but they are streams *we* asked the C encoder to make.
This corpus is the independent, standardized set — the real bar.

## What's here

| File | Tracked? | What it is |
|------|----------|------------|
| `../xtask/conformance.py` | yes | Tool: categorize, fetch (+sha1 verify), probe |
| `README.md` (this file) | yes | Corpus definition + the full vector list below |
| `vectors.json` | no (gitignored, ~94 KB) | Generated manifest: per-vector sha1, golden `.md5` ref, family, scope, probed frame count |
| `data/` | no (gitignored) | Downloaded `.ivf` vectors + `.md5` goldens |

`vectors.json` is regenerated from libaom's `test/test-data.sha1` by the tool
(`python3 xtask/conformance.py`); it is a build artifact, not committed. The
vector *bytes* live in `data/` and never enter git (per repo policy). This
human-readable file is the portable, git-tracked definition.

## The corpus

240 distinct `av1-1-*.ivf` vectors, from libaom v3.14.1's `test-data.sha1`,
hosted at `https://storage.googleapis.com/aom-test-data/<name>`. Each vector
ships a companion `<name>.md5` with one MD5 per decoded frame over the raw i420
output — that per-frame MD5 list is the golden answer.

**Decode-scope split** (heuristic by family; `--probe` measures real frame counts):

| scope | count | meaning |
|-------|-------|---------|
| intra | 230 | decodes with intra/KEY tooling only — **in scope for the current decoder** |
| inter | 5 | needs motion compensation / multi-ref (`05-mv`, `06-mfmv`, `22-svc`) — not yet ported |
| special | 5 | needs an extra tool: superres (`03-sizeup/down`), film grain (`23-film`), monochrome (`24-monochrome`), cross-frame CDF carry (`04-cdfupdate`) |

The **ALLINTRA-primary** targets are `02-allintra` (39-frame all-intra),
`16-intra` (intra-only, incl. the intrabc extreme-DV vector), the 128
`00-quantizer` per-qindex vectors (b8 + b10), and the 100 `01-size`
frame-dimension vectors. Note the `b10-*` vectors are 10-bit — the current
decoder is 8-bit, so those are in-scope by family but gated on high-bit-depth
support.

## Usage

```bash
# (Re)build the manifest and print the scope/family summary:
python3 xtask/conformance.py

# Fetch a family (or a scope-filtered sample) into conformance/data/, sha1-verified:
python3 xtask/conformance.py --fetch --family 02-allintra
python3 xtask/conformance.py --fetch --scope intra --limit 8

# Record decoded-frame counts for fetched vectors via the C aomdec:
python3 xtask/conformance.py --probe
```

## How the decode gate consumes this (next step, decoder track)

A test in `crates/aom-decode/tests/` (owned by the decoder track) will, for each
fetched in-scope vector: decode with our decoder → emit each frame as raw i420 →
MD5 each frame → assert the sequence equals the golden `data/<name>.md5`. This
mirrors the existing `real_bitstream.rs` byte-identity pattern, but against the
standardized corpus instead of synthetic streams. Vectors whose scope the
decoder doesn't yet cover (inter, 10-bit, special tools) are skipped **by an
explicit caller-visible filter keyed on `scope_hint`/`bitdepth` in the
manifest** — never by a silent in-test early-return.

## Full vector list

### 00-quantizer  — 128 vector(s), scope: **intra**
- `av1-1-b10-00-quantizer-00.ivf`  (b10)
- `av1-1-b10-00-quantizer-01.ivf`  (b10)
- `av1-1-b10-00-quantizer-02.ivf`  (b10)
- `av1-1-b10-00-quantizer-03.ivf`  (b10)
- `av1-1-b10-00-quantizer-04.ivf`  (b10)
- `av1-1-b10-00-quantizer-05.ivf`  (b10)
- `av1-1-b10-00-quantizer-06.ivf`  (b10)
- `av1-1-b10-00-quantizer-07.ivf`  (b10)
- `av1-1-b10-00-quantizer-08.ivf`  (b10)
- `av1-1-b10-00-quantizer-09.ivf`  (b10)
- `av1-1-b10-00-quantizer-10.ivf`  (b10)
- `av1-1-b10-00-quantizer-11.ivf`  (b10)
- `av1-1-b10-00-quantizer-12.ivf`  (b10)
- `av1-1-b10-00-quantizer-13.ivf`  (b10)
- `av1-1-b10-00-quantizer-14.ivf`  (b10)
- `av1-1-b10-00-quantizer-15.ivf`  (b10)
- `av1-1-b10-00-quantizer-16.ivf`  (b10)
- `av1-1-b10-00-quantizer-17.ivf`  (b10)
- `av1-1-b10-00-quantizer-18.ivf`  (b10)
- `av1-1-b10-00-quantizer-19.ivf`  (b10)
- `av1-1-b10-00-quantizer-20.ivf`  (b10)
- `av1-1-b10-00-quantizer-21.ivf`  (b10)
- `av1-1-b10-00-quantizer-22.ivf`  (b10)
- `av1-1-b10-00-quantizer-23.ivf`  (b10)
- `av1-1-b10-00-quantizer-24.ivf`  (b10)
- `av1-1-b10-00-quantizer-25.ivf`  (b10)
- `av1-1-b10-00-quantizer-26.ivf`  (b10)
- `av1-1-b10-00-quantizer-27.ivf`  (b10)
- `av1-1-b10-00-quantizer-28.ivf`  (b10)
- `av1-1-b10-00-quantizer-29.ivf`  (b10)
- `av1-1-b10-00-quantizer-30.ivf`  (b10)
- `av1-1-b10-00-quantizer-31.ivf`  (b10)
- `av1-1-b10-00-quantizer-32.ivf`  (b10)
- `av1-1-b10-00-quantizer-33.ivf`  (b10)
- `av1-1-b10-00-quantizer-34.ivf`  (b10)
- `av1-1-b10-00-quantizer-35.ivf`  (b10)
- `av1-1-b10-00-quantizer-36.ivf`  (b10)
- `av1-1-b10-00-quantizer-37.ivf`  (b10)
- `av1-1-b10-00-quantizer-38.ivf`  (b10)
- `av1-1-b10-00-quantizer-39.ivf`  (b10)
- `av1-1-b10-00-quantizer-40.ivf`  (b10)
- `av1-1-b10-00-quantizer-41.ivf`  (b10)
- `av1-1-b10-00-quantizer-42.ivf`  (b10)
- `av1-1-b10-00-quantizer-43.ivf`  (b10)
- `av1-1-b10-00-quantizer-44.ivf`  (b10)
- `av1-1-b10-00-quantizer-45.ivf`  (b10)
- `av1-1-b10-00-quantizer-46.ivf`  (b10)
- `av1-1-b10-00-quantizer-47.ivf`  (b10)
- `av1-1-b10-00-quantizer-48.ivf`  (b10)
- `av1-1-b10-00-quantizer-49.ivf`  (b10)
- `av1-1-b10-00-quantizer-50.ivf`  (b10)
- `av1-1-b10-00-quantizer-51.ivf`  (b10)
- `av1-1-b10-00-quantizer-52.ivf`  (b10)
- `av1-1-b10-00-quantizer-53.ivf`  (b10)
- `av1-1-b10-00-quantizer-54.ivf`  (b10)
- `av1-1-b10-00-quantizer-55.ivf`  (b10)
- `av1-1-b10-00-quantizer-56.ivf`  (b10)
- `av1-1-b10-00-quantizer-57.ivf`  (b10)
- `av1-1-b10-00-quantizer-58.ivf`  (b10)
- `av1-1-b10-00-quantizer-59.ivf`  (b10)
- `av1-1-b10-00-quantizer-60.ivf`  (b10)
- `av1-1-b10-00-quantizer-61.ivf`  (b10)
- `av1-1-b10-00-quantizer-62.ivf`  (b10)
- `av1-1-b10-00-quantizer-63.ivf`  (b10)
- `av1-1-b8-00-quantizer-00.ivf`  (b8)
- `av1-1-b8-00-quantizer-01.ivf`  (b8)
- `av1-1-b8-00-quantizer-02.ivf`  (b8)
- `av1-1-b8-00-quantizer-03.ivf`  (b8)
- `av1-1-b8-00-quantizer-04.ivf`  (b8)
- `av1-1-b8-00-quantizer-05.ivf`  (b8)
- `av1-1-b8-00-quantizer-06.ivf`  (b8)
- `av1-1-b8-00-quantizer-07.ivf`  (b8)
- `av1-1-b8-00-quantizer-08.ivf`  (b8)
- `av1-1-b8-00-quantizer-09.ivf`  (b8)
- `av1-1-b8-00-quantizer-10.ivf`  (b8)
- `av1-1-b8-00-quantizer-11.ivf`  (b8)
- `av1-1-b8-00-quantizer-12.ivf`  (b8)
- `av1-1-b8-00-quantizer-13.ivf`  (b8)
- `av1-1-b8-00-quantizer-14.ivf`  (b8)
- `av1-1-b8-00-quantizer-15.ivf`  (b8)
- `av1-1-b8-00-quantizer-16.ivf`  (b8)
- `av1-1-b8-00-quantizer-17.ivf`  (b8)
- `av1-1-b8-00-quantizer-18.ivf`  (b8)
- `av1-1-b8-00-quantizer-19.ivf`  (b8)
- `av1-1-b8-00-quantizer-20.ivf`  (b8)
- `av1-1-b8-00-quantizer-21.ivf`  (b8)
- `av1-1-b8-00-quantizer-22.ivf`  (b8)
- `av1-1-b8-00-quantizer-23.ivf`  (b8)
- `av1-1-b8-00-quantizer-24.ivf`  (b8)
- `av1-1-b8-00-quantizer-25.ivf`  (b8)
- `av1-1-b8-00-quantizer-26.ivf`  (b8)
- `av1-1-b8-00-quantizer-27.ivf`  (b8)
- `av1-1-b8-00-quantizer-28.ivf`  (b8)
- `av1-1-b8-00-quantizer-29.ivf`  (b8)
- `av1-1-b8-00-quantizer-30.ivf`  (b8)
- `av1-1-b8-00-quantizer-31.ivf`  (b8)
- `av1-1-b8-00-quantizer-32.ivf`  (b8)
- `av1-1-b8-00-quantizer-33.ivf`  (b8)
- `av1-1-b8-00-quantizer-34.ivf`  (b8)
- `av1-1-b8-00-quantizer-35.ivf`  (b8)
- `av1-1-b8-00-quantizer-36.ivf`  (b8)
- `av1-1-b8-00-quantizer-37.ivf`  (b8)
- `av1-1-b8-00-quantizer-38.ivf`  (b8)
- `av1-1-b8-00-quantizer-39.ivf`  (b8)
- `av1-1-b8-00-quantizer-40.ivf`  (b8)
- `av1-1-b8-00-quantizer-41.ivf`  (b8)
- `av1-1-b8-00-quantizer-42.ivf`  (b8)
- `av1-1-b8-00-quantizer-43.ivf`  (b8)
- `av1-1-b8-00-quantizer-44.ivf`  (b8)
- `av1-1-b8-00-quantizer-45.ivf`  (b8)
- `av1-1-b8-00-quantizer-46.ivf`  (b8)
- `av1-1-b8-00-quantizer-47.ivf`  (b8)
- `av1-1-b8-00-quantizer-48.ivf`  (b8)
- `av1-1-b8-00-quantizer-49.ivf`  (b8)
- `av1-1-b8-00-quantizer-50.ivf`  (b8)
- `av1-1-b8-00-quantizer-51.ivf`  (b8)
- `av1-1-b8-00-quantizer-52.ivf`  (b8)
- `av1-1-b8-00-quantizer-53.ivf`  (b8)
- `av1-1-b8-00-quantizer-54.ivf`  (b8)
- `av1-1-b8-00-quantizer-55.ivf`  (b8)
- `av1-1-b8-00-quantizer-56.ivf`  (b8)
- `av1-1-b8-00-quantizer-57.ivf`  (b8)
- `av1-1-b8-00-quantizer-58.ivf`  (b8)
- `av1-1-b8-00-quantizer-59.ivf`  (b8)
- `av1-1-b8-00-quantizer-60.ivf`  (b8)
- `av1-1-b8-00-quantizer-61.ivf`  (b8)
- `av1-1-b8-00-quantizer-62.ivf`  (b8)
- `av1-1-b8-00-quantizer-63.ivf`  (b8)

### 01-size  — 100 vector(s), scope: **intra**
- `av1-1-b8-01-size-16x16.ivf`  (b8)
- `av1-1-b8-01-size-16x18.ivf`  (b8)
- `av1-1-b8-01-size-16x32.ivf`  (b8)
- `av1-1-b8-01-size-16x34.ivf`  (b8)
- `av1-1-b8-01-size-16x64.ivf`  (b8)
- `av1-1-b8-01-size-16x66.ivf`  (b8)
- `av1-1-b8-01-size-18x16.ivf`  (b8)
- `av1-1-b8-01-size-18x18.ivf`  (b8)
- `av1-1-b8-01-size-18x32.ivf`  (b8)
- `av1-1-b8-01-size-18x34.ivf`  (b8)
- `av1-1-b8-01-size-18x64.ivf`  (b8)
- `av1-1-b8-01-size-18x66.ivf`  (b8)
- `av1-1-b8-01-size-196x196.ivf`  (b8)
- `av1-1-b8-01-size-196x198.ivf`  (b8)
- `av1-1-b8-01-size-196x200.ivf`  (b8)
- `av1-1-b8-01-size-196x202.ivf`  (b8)
- `av1-1-b8-01-size-196x208.ivf`  (b8)
- `av1-1-b8-01-size-196x210.ivf`  (b8)
- `av1-1-b8-01-size-196x224.ivf`  (b8)
- `av1-1-b8-01-size-196x226.ivf`  (b8)
- `av1-1-b8-01-size-198x196.ivf`  (b8)
- `av1-1-b8-01-size-198x198.ivf`  (b8)
- `av1-1-b8-01-size-198x200.ivf`  (b8)
- `av1-1-b8-01-size-198x202.ivf`  (b8)
- `av1-1-b8-01-size-198x208.ivf`  (b8)
- `av1-1-b8-01-size-198x210.ivf`  (b8)
- `av1-1-b8-01-size-198x224.ivf`  (b8)
- `av1-1-b8-01-size-198x226.ivf`  (b8)
- `av1-1-b8-01-size-200x196.ivf`  (b8)
- `av1-1-b8-01-size-200x198.ivf`  (b8)
- `av1-1-b8-01-size-200x200.ivf`  (b8)
- `av1-1-b8-01-size-200x202.ivf`  (b8)
- `av1-1-b8-01-size-200x208.ivf`  (b8)
- `av1-1-b8-01-size-200x210.ivf`  (b8)
- `av1-1-b8-01-size-200x224.ivf`  (b8)
- `av1-1-b8-01-size-200x226.ivf`  (b8)
- `av1-1-b8-01-size-202x196.ivf`  (b8)
- `av1-1-b8-01-size-202x198.ivf`  (b8)
- `av1-1-b8-01-size-202x200.ivf`  (b8)
- `av1-1-b8-01-size-202x202.ivf`  (b8)
- `av1-1-b8-01-size-202x208.ivf`  (b8)
- `av1-1-b8-01-size-202x210.ivf`  (b8)
- `av1-1-b8-01-size-202x224.ivf`  (b8)
- `av1-1-b8-01-size-202x226.ivf`  (b8)
- `av1-1-b8-01-size-208x196.ivf`  (b8)
- `av1-1-b8-01-size-208x198.ivf`  (b8)
- `av1-1-b8-01-size-208x200.ivf`  (b8)
- `av1-1-b8-01-size-208x202.ivf`  (b8)
- `av1-1-b8-01-size-208x208.ivf`  (b8)
- `av1-1-b8-01-size-208x210.ivf`  (b8)
- `av1-1-b8-01-size-208x224.ivf`  (b8)
- `av1-1-b8-01-size-208x226.ivf`  (b8)
- `av1-1-b8-01-size-210x196.ivf`  (b8)
- `av1-1-b8-01-size-210x198.ivf`  (b8)
- `av1-1-b8-01-size-210x200.ivf`  (b8)
- `av1-1-b8-01-size-210x202.ivf`  (b8)
- `av1-1-b8-01-size-210x208.ivf`  (b8)
- `av1-1-b8-01-size-210x210.ivf`  (b8)
- `av1-1-b8-01-size-210x224.ivf`  (b8)
- `av1-1-b8-01-size-210x226.ivf`  (b8)
- `av1-1-b8-01-size-224x196.ivf`  (b8)
- `av1-1-b8-01-size-224x198.ivf`  (b8)
- `av1-1-b8-01-size-224x200.ivf`  (b8)
- `av1-1-b8-01-size-224x202.ivf`  (b8)
- `av1-1-b8-01-size-224x208.ivf`  (b8)
- `av1-1-b8-01-size-224x210.ivf`  (b8)
- `av1-1-b8-01-size-224x224.ivf`  (b8)
- `av1-1-b8-01-size-224x226.ivf`  (b8)
- `av1-1-b8-01-size-226x196.ivf`  (b8)
- `av1-1-b8-01-size-226x198.ivf`  (b8)
- `av1-1-b8-01-size-226x200.ivf`  (b8)
- `av1-1-b8-01-size-226x202.ivf`  (b8)
- `av1-1-b8-01-size-226x208.ivf`  (b8)
- `av1-1-b8-01-size-226x210.ivf`  (b8)
- `av1-1-b8-01-size-226x224.ivf`  (b8)
- `av1-1-b8-01-size-226x226.ivf`  (b8)
- `av1-1-b8-01-size-32x16.ivf`  (b8)
- `av1-1-b8-01-size-32x18.ivf`  (b8)
- `av1-1-b8-01-size-32x32.ivf`  (b8)
- `av1-1-b8-01-size-32x34.ivf`  (b8)
- `av1-1-b8-01-size-32x64.ivf`  (b8)
- `av1-1-b8-01-size-32x66.ivf`  (b8)
- `av1-1-b8-01-size-34x16.ivf`  (b8)
- `av1-1-b8-01-size-34x18.ivf`  (b8)
- `av1-1-b8-01-size-34x32.ivf`  (b8)
- `av1-1-b8-01-size-34x34.ivf`  (b8)
- `av1-1-b8-01-size-34x64.ivf`  (b8)
- `av1-1-b8-01-size-34x66.ivf`  (b8)
- `av1-1-b8-01-size-64x16.ivf`  (b8)
- `av1-1-b8-01-size-64x18.ivf`  (b8)
- `av1-1-b8-01-size-64x32.ivf`  (b8)
- `av1-1-b8-01-size-64x34.ivf`  (b8)
- `av1-1-b8-01-size-64x64.ivf`  (b8)
- `av1-1-b8-01-size-64x66.ivf`  (b8)
- `av1-1-b8-01-size-66x16.ivf`  (b8)
- `av1-1-b8-01-size-66x18.ivf`  (b8)
- `av1-1-b8-01-size-66x32.ivf`  (b8)
- `av1-1-b8-01-size-66x34.ivf`  (b8)
- `av1-1-b8-01-size-66x64.ivf`  (b8)
- `av1-1-b8-01-size-66x66.ivf`  (b8)

### 02-allintra  — 1 vector(s), scope: **intra**
- `av1-1-b8-02-allintra.ivf`  (b8)

### 16-intra  — 1 vector(s), scope: **intra**
- `av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf`  (b8)

### 04-cdfupdate  — 1 vector(s), scope: **special**
- `av1-1-b8-04-cdfupdate.ivf`  (b8)

### 05-mv  — 1 vector(s), scope: **inter**
- `av1-1-b8-05-mv.ivf`  (b8)

### 06-mfmv  — 1 vector(s), scope: **inter**
- `av1-1-b8-06-mfmv.ivf`  (b8)

### 22-svc  — 3 vector(s), scope: **inter**
- `av1-1-b8-22-svc-L1T2.ivf`  (b8)
- `av1-1-b8-22-svc-L2T1.ivf`  (b8)
- `av1-1-b8-22-svc-L2T2.ivf`  (b8)

### 23-film  — 2 vector(s), scope: **special**
- `av1-1-b10-23-film_grain-50.ivf`  (b10)
- `av1-1-b8-23-film_grain-50.ivf`  (b8)

### 24-monochrome  — 2 vector(s), scope: **special**
- `av1-1-b10-24-monochrome.ivf`  (b10)
- `av1-1-b8-24-monochrome.ivf`  (b8)