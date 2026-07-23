# Intra near-tie deltas under the 5% criterion — 2026-07-23

**Question answered:** the remaining intra encoder divergences vs C are pinned "RD
near-ties". Under the new acceptance criterion — a tie-break only matters if it
changes **encoded size by >5%**, OR **ssim2 by >5%**, OR **hurts RD by >5%**
(Pareto-worse / below C's local RD curve) — how many of the pinned cells
actually matter?

**Headline:** of the **30** currently-divergent cells, **13 emit valid streams**
and of those only **1** exceeds any 5% axis (and only marginally, at an extreme
operating point). The other **17 are not tie-breaks at all**: the port emits
**invalid AV1 streams** on them (the real C decoder rejects 16 outright; 1 does
not decode in the port's own decoder either). Under the 5% criterion, the
genuine near-tie work is effectively DONE; what remains on these cells is a
**stream-validity bug class**, not RD polishing.

## Provenance

- Date: 2026-07-23, host `dev-32gb` (dedicated box).
- Source measured: `zenav1-aom` @ `046b897f5e09` (jj working copy base at
  measurement time; **no source or test modifications** — the harness was an
  uncommitted throwaway example reusing the gate machinery verbatim).
- C reference: from-source libaom **v3.14.1** (`upstream/` @ 03087864) through
  the `aom-sys-ref` shims (`ref_encode_av1_kf*` = the aomenc path,
  `ref_decode_av1_kf` = the gold C pixel oracle).
- Port encode for KB-13/KB-10/KB-11/KB-12 cells: the canonical gate harness
  `attempt_case_content_uv_sep` (transplanted verbatim from
  `crates/aom-encode/tests/encoder_gate_e2e_byte_match.rs`, modified only to
  return the payloads). Port encode for KB-P29/KB-15: `EncodeCell::
  port_encode_with` (`crates/aom-bench`), the machinery those pins are defined
  with.
- Decode: BOTH streams decoded with the port decoder
  (`aom_decode::frame::decode_frame_obus`, Gate-1 conformance-proven), plus a
  **cross-decoder check**: every stream also decoded with the REAL C decoder;
  when both accept, planes were asserted byte-identical (they always were).
- ssim2 = **SSIMULACRA2** via `fast-ssim2-cli 0.6.0`
  (`~/work/fast-ssim2`, prebuilt), scoring each decoded frame against the
  **source pixels** both encoders consumed, after the identical BT.601
  limited-range YUV→RGB transform (`aom_bench::rd_close::yuv_to_rgb8`) on both
  sides so the colorimetry approximation cancels. zensim 0.2.7 computed as a
  cross-check column (CSV only).
- Gate freshness: all five gates were re-run first on this source
  (`encoder_gate_real_content_speed1to4_e2e` 41/60 + 19 DIFF exactly as pinned;
  speed6/7 noise, speed8 textured, `rd_close_palette`, `rd_close_intrabc` all
  green with their pins intact). Every measured cell ASSERTED its expected
  divergence state; 9 byte-exact control cells asserted byte-identity through
  the same harness (harness fidelity).

## Conventions

- **size**: full temporal-unit bytes. The port stream is the C stream with the
  frame OBU payload spliced (`rd_close::splice_frame_obu`), so the seq-header
  bytes are identical on both sides. `size_delta_% = 100·(port−c)/c`.
- **ssim2 delta**: reported in **points** (`port − c`). The >5% test uses the
  **relative drop** `100·(c−port)/|c|` — CAVEAT: at cq63 several cells have
  *negative* ssim2 bases (extreme quantization, e.g. 88-byte 128² frames), where
  a small point-delta produces a large relative %. Point deltas are given so
  readers can apply their own convention.
- **RD-hurt**: C's local RD curve from bracket encodes at cq±{3,6} (clamped to
  [1,63]); ssim2 linearly interpolated in ln(bytes) at the port's byte count;
  `rd_hurt_% = 100·(interp − port_ssim2)/|interp|`, positive = port below C's
  curve. Points outside the bracket are extrapolated (marked). This is a
  single-axis local proxy, not BD-rate.
- **stream**: `ok` = both decoders accept and agree; `c_rejects` = the REAL C
  decoder errors on the port stream (the port decoder parsed it leniently — the
  quality numbers shown are that lenient parse's); `no_decode` = the port
  stream does not decode at all. Any non-`ok` stream is counted EXCEEDS (a
  stream real decoders reject is more than a 5% quality change by definition).

## Results — every currently-divergent intra near-tie cell

| KB | cell | port_B | c_B | Δsize% | port_ssim2 | c_ssim2 | Δssim2 pts | rd_hurt% | stream | EXCEEDS_5PCT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| KB-13 | 196x196 cpu1 cq12 | 2220 | 2400 | −7.50 | −168.61 | 89.28 | −257.89 | +290.4 | c_rejects | **YES** |
| KB-13 | 196x196 cpu1 cq32 | 979 | 1053 | −7.03 | −225.45 | 80.19 | −305.64 | +386.9 | c_rejects | **YES** |
| KB-13 | 196x196 cpu1 cq63 | 153 | 168 | −8.93 | −297.73 | −38.15 | −259.58 | +578.4 (extrap) | c_rejects | **YES** |
| KB-13 | 196x196 cpu2 cq12 | 2254 | 2411 | −6.51 | −384.04 | 89.13 | −473.17 | +533.8 | c_rejects | **YES** |
| KB-13 | 196x196 cpu2 cq32 | 1029 | 1096 | −6.11 | −113.44 | 78.91 | −192.35 | +245.0 | c_rejects | **YES** |
| KB-13 | 196x196 cpu2 cq63 | 163 | 177 | −7.91 | −341.56 | −33.77 | −307.79 | +762.7 (extrap) | c_rejects | **YES** |
| KB-13 | 196x196 cpu3 cq12 | 2268 | 2404 | −5.66 | −192.26 | 89.62 | −281.87 | +316.6 | c_rejects | **YES** |
| KB-13 | 196x196 cpu3 cq32 | 1030 | 1110 | −7.21 | −126.42 | 78.22 | −204.64 | +262.6 | c_rejects | **YES** |
| KB-13 | 196x196 cpu3 cq63 | 167 | 183 | −8.74 | −286.80 | −37.45 | −249.35 | +548.8 (extrap) | c_rejects | **YES** |
| KB-13 | 196x196 cpu4 cq12 | 2335 | 2507 | −6.86 | −265.95 | 89.03 | −354.98 | +400.9 | c_rejects | **YES** |
| KB-13 | 196x196 cpu4 cq32 | 1109 | 1170 | −5.21 | −151.69 | 78.35 | −230.04 | +297.3 | c_rejects | **YES** |
| KB-13 | 196x196 cpu4 cq63 | 163 | 177 | −7.91 | −212.76 | −37.18 | −175.58 | +405.2 (extrap) | c_rejects | **YES** |
| KB-13 | q00-64 cpu4 cq32 | 558 | 560 | −0.36 | 68.78 | 68.73 | +0.05 | +0.02 | ok | no |
| KB-13 | q00-128 cpu3 cq63 | 90 | 88 | +2.27 | −43.37 | −40.68 | −2.69 | +8.25 | ok | **YES**¹ |
| KB-13 | q00-128 cpu4 cq12 | 4623 | 4632 | −0.19 | 87.13 | 87.11 | +0.01 | −0.04 | ok | no |
| KB-13 | q00-128 cpu4 cq32 | 1912 | 1905 | +0.37 | 67.46 | 67.47 | −0.01 | +0.18 | ok | no |
| KB-13 | q00-128 cpu4 cq63 | 83 | 82 | +1.22 | −45.73 | −45.24 | −0.49 | +2.17 | ok | no |
| KB-13 | fg-64 cpu3 cq63 | 72 | 72 | 0.00 | 37.02 | 37.92 | −0.89 | +2.36 | ok | no |
| KB-13 | fg-64 cpu4 cq32 | 521 | 520 | +0.19 | 77.19 | 77.24 | −0.06 | +0.08 | ok | no |
| KB-10 | noise64 s6 mono cq63 | 21 | 21 | 0.00 | 25.84 | 26.17 | −0.33 | +1.26 | c_rejects | **YES** |
| KB-10 | noise64 s6 420 cq63 | 24 | 24 | 0.00 | −45.14 | 26.17 | −71.30 | +272.5 | c_rejects | **YES** |
| KB-11 | noise64 s7 mono cq63 | 21 | 21 | 0.00 | 25.84 | 26.17 | −0.33 | +1.26 | c_rejects | **YES** |
| KB-11 | noise64 s7 420 cq63 | 24 | 24 | 0.00 | −45.14 | 26.17 | −71.30 | +272.5 | c_rejects | **YES** |
| KB-12 | diag64 s8 mono cq12 | 136 | 137 | −0.73 | 97.35 | 97.45 | −0.10 | +0.11 | ok | no |
| KB-12 | diag64 s8 420 cq12 | 613 | 613 | 0.00 | 84.55 | 84.59 | −0.03 | +0.04 | ok | no |
| KB-12 | diag128 s8 mono cq32 | 127 | 122 | +4.10 | 87.68 | 89.38 | −1.70 | −10.66² | ok | no |
| KB-12 | diag128 s8 420 cq32 | 1473 | 1467 | +0.41 | 60.22 | 59.71 | +0.51 | −0.58 | ok | no |
| KB-P29 | ui_420_128_cq32 | 362 | 353 | +2.55 | 90.15 | 88.97 | +1.19 | +2.14 | ok | no |
| KB-P29 | text_420_128_cq20 | 1245 | 1245 | 0.00 | 95.56 | 95.74 | −0.17 | +0.18 | ok | no |
| KB-15 | scc_480x180_196_cq48 | 1909 | 1905 | +0.21 | n/a | 61.98 | n/a | n/a | no_decode | **YES** |

¹ `q00-128 cpu3 cq63` exceeds only under the *relative* convention on a
**negative ssim2 base**: the port is −2.69 ssim2 **points** worse and +2 bytes
bigger at an 88-byte / 128² operating point (0.0045 bpp) where BOTH encodes
score ssim2 ≈ −41 (i.e. far past useful quality). In absolute points this is
the largest valid-stream quality delta but still under 3 points; whether it
"matters" at that operating point is a judgment call — flagged honestly per the
stated convention. It is also the only valid-stream cell that is
Pareto-dominated with a >5% axis.

² Negative rd_hurt = the port sits ABOVE C's local curve there (fine); the
sparse 122–137-byte curve segment makes the magnitude noisy, but the sign is
what matters.

## Summary

| | count |
|---|---:|
| Divergent cells measured (all currently-pinned/DIFF intra cells) | **30 / 30** |
| — emit **valid** streams (both decoders accept, planes agree) | 13 |
| — emit **invalid** streams (**C decoder rejects**) | 16 |
| — stream does **not decode at all** (port decoder either) | 1 (KB-15) |
| Exceed 5% on **size** | **12** (all = the 196² cluster, −5.2%…−8.9%) |
| Exceed 5% on **ssim2** (rel. convention, incl. broken streams) | **18** |
| Exceed 5% on **RD-hurt** | **15** |
| **EXCEED ANY (the criterion)** | **18** = 17 broken-stream + 1 valid¹ |
| Valid-stream cells under the bar on every axis | **12 of 13** |

**Reading:** under the 5% criterion the *near-tie* problem is solved — every
valid-stream divergence except one marginal extreme-quantization cell¹ is well
under 5% on size (≤2.6%), ssim2 (≤0.9 pts / ≤2.4% rel) and RD-hurt (≤2.4%).
The 18 exceeding cells are NOT fixable by tie-break polishing: 17 of them are a
different defect class entirely (below).

## KEY FINDING — 17 of the 30 "near-ties" are stream-validity bugs, not ties

1. **KB-13 196² cluster (12 cells): the port's stream is INVALID AV1.**
   `aom_codec_decode` (real libaom) errors on every one of the port's 196²
   streams; the port's own decoder parses them leniently into garbage recon
   (ssim2 −113…−384 vs source). The famous "codes 5–9% fewer bytes" signature
   on this cluster is therefore NOT an RD win — the missing bytes come with a
   broken bitstream. This matches KB-13's note that the 196² partial-SB cluster
   is a SEPARATE root (AB-vs-AB flip at partial-SB edges): the search/pack
   disagreement corrupts the stream, it does not merely pick a different valid
   tree. **These cells leave the "near-tie" bucket entirely: they are encoder
   correctness bugs and must be fixed regardless of any % criterion.**
2. **KB-10/KB-11 noise-cq63 (4 cells): same class.** The known "(mi 8,0)
   TX_16X16-vs-TX_32X32 winner-sweep tie desyncs the LARGEST-tx parse" (KB-10)
   produces streams the C decoder rejects at both speed 6 and speed 7 (mono and
   420). The mono cells' lenient parse happens to land near-identical pixels
   (Δssim2 −0.33 pts) but the stream is still invalid.
3. **KB-15 intrabc (1 cell): the port stream does not decode** (bitstream
   desync mid-tile — already documented in the pin). Size delta is +0.21%
   (1909B vs 1905B full-TU; +4B frame-payload, consistent with the pin's
   1895/1891). Quality unmeasurable until the stream decodes.

### Side-findings (measure-only; not fixed here)

- **Port-decoder leniency:** the port decoder decodes all 16 C-rejected streams
  without error (garbage output, no failure signal). The C decoder's bitstream
  consistency checks catch what the port's parse does not — a decode-side
  strictness gap worth its own tracking issue.
- **Harness inconsistency:** `aom_bench::EncodeCell::port_encode` and the
  canonical gate harness (`attempt_case_content_uv_sep`) produce DIFFERENT
  streams on at least `196x196 cpu1 cq63` — aom-bench's port encode
  byte-matches real aomenc there while the gate harness diverges (verified both
  ways in this session; the gate re-run confirms its 19-DIFF map). Unverified
  hypothesis: the speed-3 qindex arm of `less_rectangular_check_level` landed
  in aom-bench's cfg-fill (KB-13 note) but not in the gate harness's. Worth a
  follow-up: if porting that arm into the gate harness flips 196² cells to
  byte-exact, part of the cluster closes.
- The 4 KB-12 speed-8 diag cells, both KB-P29 palette cells, and the 6 interior
  KB-13 cells with valid streams are all comfortably inside the band — under
  the 5% criterion those pins document divergences that do not matter.

## What remains for intra, under the 5% criterion

- **Nothing** on: KB-12 (4 cells), KB-P29 (2 cells), and 6 of the 7 interior
  KB-13 cells — all valid streams, all far under every 5% axis.
- **Judgment call** on: `q00-128 cpu3 cq63` (valid stream, −2.7 ssim2 pts /
  +2B at an 88-byte operating point; exceeds only via the negative-base
  relative convention).
- **Must fix (correctness, criterion-independent):** the 196² partial-SB
  search/pack desync (12 cells), the KB-10/11 tx-plan desync (4 cells — one
  root, two speeds), the KB-15 intrabc desync (1 cell). Plus the two
  side-findings above (decoder strictness; harness inconsistency).

## Data files

- `benchmarks/intra_tiebreak_deltas_2026-07-23.csv` — full per-cell table
  (incl. zensim cross-check columns, pareto flags, interp/extrap markers).
- RD-curve raw points (C bracket encodes: cq, bytes, ssim2, zensim per cell):
  appendix below.

<details>
<summary>Appendix: C local RD-curve points (bracket encodes)</summary>

```csv
kb,cell,curve_cq,c_bytes,c_ssim2,c_zensim
KB-13,196x196 cpu1 cq12,6,4014,92.406622,92.842574
KB-13,196x196 cpu1 cq12,9,2911,90.445124,90.807494
KB-13,196x196 cpu1 cq12,15,2005,87.624610,88.337262
KB-13,196x196 cpu1 cq12,18,1806,86.691114,87.162265
KB-13,196x196 cpu1 cq12,12,2400,89.278413,89.877717
KB-13,196x196 cpu1 cq32,26,1333,83.264380,84.163549
KB-13,196x196 cpu1 cq32,29,1187,81.505998,82.720063
KB-13,196x196 cpu1 cq32,35,956,78.060089,79.859878
KB-13,196x196 cpu1 cq32,38,858,75.870863,77.195554
KB-13,196x196 cpu1 cq32,32,1053,80.186970,80.789877
KB-13,196x196 cpu1 cq63,57,471,26.977241,38.038133
KB-13,196x196 cpu1 cq63,60,316,0.610694,10.388634
KB-13,196x196 cpu1 cq63,63,168,-38.147467,-35.190098
KB-13,196x196 cpu2 cq12,6,4027,92.564409,92.895272
KB-13,196x196 cpu2 cq12,9,2946,90.424932,90.670385
KB-13,196x196 cpu2 cq12,15,2050,87.652729,88.481995
KB-13,196x196 cpu2 cq12,18,1823,86.311144,87.360747
KB-13,196x196 cpu2 cq12,12,2411,89.133782,89.636047
KB-13,196x196 cpu2 cq32,26,1361,82.691226,83.739397
KB-13,196x196 cpu2 cq32,29,1230,80.905963,82.218759
KB-13,196x196 cpu2 cq32,35,967,77.621085,78.962871
KB-13,196x196 cpu2 cq32,38,876,74.070929,76.412991
KB-13,196x196 cpu2 cq32,32,1096,78.906615,80.422275
KB-13,196x196 cpu2 cq63,57,469,25.242157,36.561340
KB-13,196x196 cpu2 cq63,60,309,5.590557,13.666031
KB-13,196x196 cpu2 cq63,63,177,-33.769184,-32.417060
KB-13,196x196 cpu3 cq12,6,4023,92.125578,92.731645
KB-13,196x196 cpu3 cq12,9,2998,90.404017,90.794831
KB-13,196x196 cpu3 cq12,15,2105,87.648712,88.159141
KB-13,196x196 cpu3 cq12,18,1832,86.035537,86.836372
KB-13,196x196 cpu3 cq12,12,2404,89.616504,89.846574
KB-13,196x196 cpu3 cq32,26,1411,83.022229,83.930675
KB-13,196x196 cpu3 cq32,29,1273,81.601688,82.911691
KB-13,196x196 cpu3 cq32,35,1003,77.570409,78.763209
KB-13,196x196 cpu3 cq32,38,906,74.878613,77.016809
KB-13,196x196 cpu3 cq32,32,1110,78.218231,79.943201
KB-13,196x196 cpu3 cq63,57,451,23.358690,34.899555
KB-13,196x196 cpu3 cq63,60,318,3.329697,12.606331
KB-13,196x196 cpu3 cq63,63,183,-37.452174,-37.854508
KB-13,196x196 cpu4 cq12,6,4123,92.253526,92.512302
KB-13,196x196 cpu4 cq12,9,3089,90.528876,90.843415
KB-13,196x196 cpu4 cq12,15,2181,87.776362,88.042150
KB-13,196x196 cpu4 cq12,18,1937,85.699397,86.852954
KB-13,196x196 cpu4 cq12,12,2507,89.030239,89.544604
KB-13,196x196 cpu4 cq32,26,1465,82.587260,83.283815
KB-13,196x196 cpu4 cq32,29,1311,81.095146,81.626921
KB-13,196x196 cpu4 cq32,35,1059,75.641551,77.463851
KB-13,196x196 cpu4 cq32,38,952,73.859530,75.500631
KB-13,196x196 cpu4 cq32,32,1170,78.350604,79.990556
KB-13,196x196 cpu4 cq63,57,456,25.866245,35.423593
KB-13,196x196 cpu4 cq63,60,312,-3.251999,4.087636
KB-13,196x196 cpu4 cq63,63,177,-37.179611,-33.507866
KB-13,q00-64 cpu4 cq32,26,731,78.764606,78.801430
KB-13,q00-64 cpu4 cq32,29,633,72.530945,75.831055
KB-13,q00-64 cpu4 cq32,35,468,71.936814,68.134004
KB-13,q00-64 cpu4 cq32,38,390,65.661586,60.862736
KB-13,q00-64 cpu4 cq32,32,560,68.729327,70.164894
KB-13,q00-128 cpu3 cq63,57,267,-4.335830,-5.865454
KB-13,q00-128 cpu3 cq63,60,187,-20.004071,-28.520999
KB-13,q00-128 cpu3 cq63,63,88,-40.679026,-58.481329
KB-13,q00-128 cpu4 cq12,6,6632,91.704074,91.897974
KB-13,q00-128 cpu4 cq12,9,5394,88.960284,89.341348
KB-13,q00-128 cpu4 cq12,15,4024,85.329527,85.197817
KB-13,q00-128 cpu4 cq12,18,3636,83.258528,83.062810
KB-13,q00-128 cpu4 cq12,12,4632,87.114319,86.996889
KB-13,q00-128 cpu4 cq32,26,2651,76.287449,75.587745
KB-13,q00-128 cpu4 cq32,29,2319,73.606433,72.941238
KB-13,q00-128 cpu4 cq32,35,1616,63.018869,61.668125
KB-13,q00-128 cpu4 cq32,38,1307,58.685755,56.973047
KB-13,q00-128 cpu4 cq32,32,1905,67.465897,67.401577
KB-13,q00-128 cpu4 cq63,57,254,-12.394940,-13.295839
KB-13,q00-128 cpu4 cq63,60,147,-22.146543,-28.977555
KB-13,q00-128 cpu4 cq63,63,82,-45.237084,-65.144334
KB-13,fg-64 cpu3 cq63,57,176,55.663094,34.515091
KB-13,fg-64 cpu3 cq63,60,131,48.668239,-0.058358
KB-13,fg-64 cpu3 cq63,63,72,37.918891,-26.851557
KB-13,fg-64 cpu4 cq32,26,641,78.433243,81.738785
KB-13,fg-64 cpu4 cq32,29,570,77.564555,78.767316
KB-13,fg-64 cpu4 cq32,35,442,74.860580,73.219370
KB-13,fg-64 cpu4 cq32,38,406,71.975582,66.139016
KB-13,fg-64 cpu4 cq32,32,520,77.241914,76.033819
KB-10,noise64 s6 mono cq63,57,137,37.309568,-120.600541
KB-10,noise64 s6 mono cq63,60,21,25.895958,-182.394239
KB-10,noise64 s6 mono cq63,63,21,26.165384,-182.662954
KB-10,noise64 s6 420 cq63,57,141,37.309568,-120.600541
KB-10,noise64 s6 420 cq63,60,24,25.895958,-182.394239
KB-10,noise64 s6 420 cq63,63,24,26.165384,-182.662954
KB-11,noise64 s7 mono cq63,57,137,37.309568,-120.600541
KB-11,noise64 s7 mono cq63,60,21,25.895958,-182.394239
KB-11,noise64 s7 mono cq63,63,21,26.165384,-182.662954
KB-11,noise64 s7 420 cq63,57,141,37.309568,-120.600541
KB-11,noise64 s7 420 cq63,60,24,25.895958,-182.394239
KB-11,noise64 s7 420 cq63,63,24,26.165384,-182.662954
KB-12,diag64 s8 mono cq12,6,171,97.989724,94.616300
KB-12,diag64 s8 mono cq12,9,131,97.490087,93.224444
KB-12,diag64 s8 mono cq12,15,124,97.077979,92.700725
KB-12,diag64 s8 mono cq12,18,126,96.426759,89.980855
KB-12,diag64 s8 mono cq12,12,137,97.451531,92.047114
KB-12,diag64 s8 420 cq12,6,775,90.451428,90.641352
KB-12,diag64 s8 420 cq12,9,659,88.667568,88.727229
KB-12,diag64 s8 420 cq12,15,562,80.571273,84.181464
KB-12,diag64 s8 420 cq12,18,554,76.619439,84.092828
KB-12,diag64 s8 420 cq12,12,613,84.586154,86.722937
KB-12,diag128 s8 mono cq32,26,145,87.415479,85.054769
KB-12,diag128 s8 mono cq32,29,115,89.698176,87.617827
KB-12,diag128 s8 mono cq32,35,105,89.806737,87.773369
KB-12,diag128 s8 mono cq32,38,126,78.750755,78.518256
KB-12,diag128 s8 mono cq32,32,122,89.381196,84.194900
KB-12,diag128 s8 420 cq32,26,1967,71.420793,70.272171
KB-12,diag128 s8 420 cq32,29,1708,65.873246,66.431689
KB-12,diag128 s8 420 cq32,35,1143,53.662105,54.191492
KB-12,diag128 s8 420 cq32,38,922,41.618454,42.213936
KB-12,diag128 s8 420 cq32,32,1467,59.707637,61.042552
KB-P29,ui_420_128_cq32,26,357,93.703619,95.061397
KB-P29,ui_420_128_cq32,29,359,93.005930,93.416833
KB-P29,ui_420_128_cq32,35,371,89.512543,90.846970
KB-P29,ui_420_128_cq32,38,359,86.252078,85.507454
KB-P29,ui_420_128_cq32,32,353,88.965104,90.809882
KB-P29,text_420_128_cq20,14,1250,100.000000,100.000000
KB-P29,text_420_128_cq20,17,1247,98.191364,97.991170
KB-P29,text_420_128_cq20,23,1246,95.126024,96.342690
KB-P29,text_420_128_cq20,26,1243,95.814602,95.929082
KB-P29,text_420_128_cq20,20,1245,95.735091,96.397076
```

</details>
