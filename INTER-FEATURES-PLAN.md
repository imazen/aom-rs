# Inter-decode feature ladder — chunk-4 plan (STEP-0 census)

**✅ CHUNK 4 (OBMC) DONE** — `av1-1-b8-01-size-16x18` frame 1 byte-exact (md5
`08db98983320105666c9496dc1dba209`), gated by
`inter_ratchet.rs::inter_ratchet_16x18_obmc_frame1_byte_identical`. See
`INTER-CHUNK1-HANDOFF.md` "Chunk 4 — OBMC". NOTE: the census below UNDER-STATED
16x18 as "OBMC only" — it also codes **inter var-tx** (`TX_MODE_SELECT`; the OBMC
block splits TX_16X8 → 2× TX_8X8) AND an **inter-intra flag** (coded 0 for the
BLOCK_16X8 block, but the read is required for entropy sync). Both landed with OBMC.
Next: chunk 5 = WARP (target 16x34).

**Status as of chunk 3 (partial-edge single-ref):** byte-exact for
`av1-1-b8-01-size-{64x64, 16x16, 64x66}` frame 1. The single-ref translational
path — multi-block, residual-carrying, partial-SB (128-SB `BLOCK_64X128` clipped
to a 64x66 frame), with the reference-crop-dim MC border — is DONE and gated
(`aom-decode/tests/inter_{walking_skeleton,ratchet}.rs`).

This doc is the STEP-0 census that plans chunk 4. It answers: **what is the
single smallest next byte-exact feature, and which vector isolates it?**

## Census method

Throwaway C instrument `/tmp/inspect_frame.c` (public `aom_codec_decode` +
`AV1D_GET_MI_INFO` per mi cell) — dumps per-block `bsize / mode / uv_mode /
ref_frame[0,1] / mv0 / skip_txfm / motion_mode / tx_size / interp(fy,fx) / seg`
for every frame. Motion-mode enum: `0 = SIMPLE_TRANSLATION`, `1 = OBMC_CAUSAL`,
`2 = WARPED_CAUSAL`. Ref-pair decoding: `ref=[0,-1]` = intra block in an inter
frame; `ref=[k,-1]` (k>0) = single-ref inter; `ref=[k,0]` (k>0) = **interintra**
(`ref_frame[1] == INTRA_FRAME`); `ref=[k,m]` (k,m>0) = **compound**.

## Census results

### Partial-edge `01-size-*` frame 1 (single-ref family, LOCAL) — the graded ladder

| vector | frame-1 structure | tools beyond chunk-3 scope |
|---|---|---|
| **64x66** | 1× `BLOCK_64X128` (clipped), NEWMV, SIMPLE, skip=1 | **none — DONE (chunk 3)** |
| **16x18** | 4× `BLOCK_4X16` (NEWMV+3 NEARESTMV, SIMPLE) + **1× `BLOCK_16X8` OBMC** @ mi(4,0) | **OBMC only** (1 block) |
| **16x34** | mix of SIMPLE + OBMC (mi 0,2 / 2,2 / 8,0) + **1× WARPED_CAUSAL** @ mi(4,0) | OBMC + **WARP** |
| **16x66** | SIMPLE + **WARP** @ mi(4,0) + OBMC @ mi(12,0)/(16,0) + **switchable interp** (mi 0,0 SHARP vs others EIGHTTAP) | switchable-interp-nbr + OBMC + WARP |

Every block in all four is **single `LAST` ref** (`ref=[1,-1]`), no compound, no
interintra. So the `01-size` family is the perfectly-graded single-ref ladder:
each step adds exactly one motion-mode/interp feature on top of the working base.

### Real inter conformance frame 1 (full-toolset, for contrast) — NOT minimal

`av1-1-b8-05-mv` F1 (352x288, 647 inter blocks): motion_mode {540 SIMPLE, **82
OBMC, 25 WARP**}; refs {199 single, **327 compound `[1,7]` (LAST+ALTREF)**, 13
interintra `[1,0]`, 97 `[7,-1]` (ALTREF single → needs temporal/altref DPB)};
switchable interp (8 fy/fx combos); compound + NEAREST_NEARESTMV/NEAR_NEARMV
(m18-m24) inter-compound modes. **The entire toolset at once.**

`av1-1-b10-00-quantizer-{00,15,45}` F1 (640x360, bd10): 693–2773 inter blocks,
always {SIMPLE + OBMC (216/543/213) + WARP (169/331/103)}, interintra (`[1,0]`,
146/71/23), mixed with intra blocks. Matches the chunk-1 handoff census of the
b8 quantizer frames ("OBMC + WARPED_CAUSAL + interintra, full tool set"). Also
bd10 → the decoder inter/MC path is currently lowbd-only (`aom-inter` bd8).

**Conclusion:** neither `05-mv` nor any `-00-quantizer` frame 1 is a minimal
next step — they require OBMC **and** warp **and** compound **and** interintra
**and** (for `05-mv`/altref) a multi-ref DPB + temporal-MV projection
simultaneously. The `01-size` partial-edge ladder is the correct sequencing.

## RECOMMENDED CHUNK 4 — **OBMC (overlapped block motion compensation)**

**Target vector: `av1-1-b8-01-size-16x18` frame 1.** It isolates OBMC perfectly:
single `LAST` ref, all `SIMPLE` except **one** `OBMC_CAUSAL` block
(`BLOCK_16X8` @ mi(4,0)), non-switchable EIGHTTAP, no warp/compound/interintra.
The OBMC block sits at mi_col 0 (frame left edge → **no left neighbour**) with a
full row of `BLOCK_4X16` above it, so only the **above**-neighbour OBMC blend is
exercised — the smallest possible OBMC surface. Golden `.md5` line 2:
`53cd765e2dacdc5acef9e40b707e448a` (F0) / `08db98983320105666c9496dc1dba209` (F1).

Byte-exact gate: extend `inter_ratchet.rs` with the 16x18 cell (helper already
parameterised) + a per-kernel differential for the OBMC blend vs real C.

### What chunk 4 must port (C file:line — from `reference/libaom`)

1. **motion_mode symbol read — ALREADY PORTED.** `read_motion_mode`
   (`aom-entropy/src/partition.rs:4408`; C `read_motion_mode`, decodemv.c) reads
   the 2-symbol `obmc_cdf[bsize]` flag when the ceiling is `OBMC_CAUSAL`. Chunk 4
   only needs to *call* it: replace the guard assertion in
   `TileKf::decode_block_inter` (`aom-decode/src/lib.rs:1996`) with the real read,
   gated by `motion_mode_allowed`.
2. **`motion_mode_allowed`** (`blockd.h:1477`): the ceiling — `SIMPLE` unless the
   frame allows switchable motion modes AND `is_motion_variation_allowed_bsize`
   (`min(bw,bh) >= 8`) AND `has_overlappable_candidates`; then `OBMC_CAUSAL` if
   `!allow_warped_motion` or the block can't warp, else `WARPED_CAUSAL` ceiling.
3. **`av1_count_overlappable_neighbors`** (`reconinter.c:801`) +
   `foreach_overlappable_nb_above/left` — the above/left neighbour scan (mi-grid
   walk, `is_neighbor_overlappable` = neighbour is inter). Feeds gate (2) and the
   blend's neighbour list.
4. **OBMC prediction build (decoder driver)** —
   `dec_build_obmc_inter_predictors_sb` (`decodeframe.c:818`):
   - `av1_setup_obmc_dst_bufs` — two scratch dst buffers.
   - `dec_build_prediction_by_above_preds` (`decodeframe.c:736`) +
     `av1_setup_build_prediction_by_above_pred` (`reconinter.c`): for each
     overlappable above neighbour, MC-predict a `min(bw, ...)`×`overlap` strip
     using the **neighbour's** MV + ref + interp filter into scratch buf1.
   - `dec_build_prediction_by_left_preds` (`decodeframe.c:791`) — the left twin
     (INERT for the 16x18 target: the OBMC block is at the left frame edge).
5. **OBMC blend** — `av1_build_obmc_inter_prediction` (`reconinter.c:935`) →
   `build_obmc_inter_pred_above` (`:852`) + `build_obmc_inter_pred_left` (`:891`):
   blend the block's own predictor with the neighbour strips using the raised-
   cosine `av1_get_obmc_mask(length)` (`reconinter.c:774`; tables `obmc_mask_{1,
   2,4,8,16,32,64}` at `:751-782`), `AOM_BLEND_A64` per pixel.
6. **`av1_skip_u4x4_pred_in_obmc`** (obmc.h) — the sub-4x4 skip predicate used by
   the above/left strip walks.

### New MC kernel to add to `aom-inter`

`build_for_obmc` prediction is a **plane strip** MC (the neighbour's motion into
a narrow overlap band) followed by a masked blend. Add to `aom-inter`:
- an OBMC-strip predictor: reuse `build_inter_predictor` for the strip MC. The
  neighbour predictor covers the neighbour's mi width/height (`op_mi_size *
  MI_SIZE`), predicted from the neighbour's own MV/ref/interp; the blend then
  only touches the overlap band — `overlap_above = min(block_high[bsize], 64) >>
  1`, `overlap_left = min(block_wide[bsize], 64) >> 1` (`reconinter.c:860/899`),
  subsampled per plane.
- the blend `build_obmc_inter_pred_{above,left}` (`reconinter.c:852/891`):
  `aom_blend_a64_vmask` (above) / `aom_blend_a64_hmask` (left) with
  `av1_get_obmc_mask(overlap >> ss)` — a straight `AOM_BLEND_A64` of the block's
  own predictor against the neighbour strip.

Each lands with a **differential vs the REAL exported C** (`av1_build_obmc_inter_
prediction` / `av1_get_obmc_mask` / `aom_blend_a64_{v,h}mask` via a new
`aom-sys-ref` shim) — no graceful skip, verified-or-nothing, matching the
chunk-1d `inter_pred_diff.rs` pattern.

### Dependencies / ordering

- Depends on the working single-ref translational MC (chunk 1-3) — DONE.
- The neighbour scan reuses the mi-grid `DvNbr`/`stamp_dv` infra (mv0/ref stamped
  per block) already present in `decode_block_inter`.
- Chunk 5 = **WARPED_CAUSAL** (target `16x34`): local warp model estimation
  (`av1_find_projection` / `av1_warp_plane`) — larger; needs its own census pass.
- Chunk 6 = **switchable-interp with neighbours** (target `16x66`): the
  `av1_get_pred_context_switchable_interp` neighbour-filter grid — the entropy
  side landed (origin/main `835b0c0`). **WIRING LANDED (this chunk):**
  `read_mb_interp_filter` is wired into `decode_block_inter` — the per-direction
  switchable read on the neighbour-context CDF (`get_pred_context_switchable_
  interp`), a per-mi interp-filter grid (`TileKf::mi_interp`, stamped/read like
  `mi_dv`), and the `av1_is_interp_needed` gate (dead `!interp_needed` branch is
  the WARPED_CAUSAL seam for chunk 5 — a WARPED_CAUSAL block reads NO interp
  symbol; `motion_mode` must be threaded in ABOVE the interp read per the C
  order, `read_inter_block_mode_info` :1422 motion_mode / :1481 interp). The
  no-neighbour switchable read is byte-gated by the 64x64 skeleton (SHARP); the
  with-neighbour + dual-filter context is C-locked at the primitive layer
  (`partition_diff.rs`). **The 16x66 STEP-0 census corrects the plan: frame 1 is
  `TX_MODE_SELECT` (var-tx) — mi(8,x)/(16,0) carry non-largest tx — so 16x66
  needs var-tx AND OBMC (chunk 4, mi(12,0)/(16,0)) AND WARP (chunk 5, mi(4,0)),
  not just 4+5.** The port's inter envelope hard-asserts LARGEST tx / SIMPLE
  motion, so it cannot yet parse frame 1 to completion in isolation. Gate:
  `inter_ratchet.rs::inter_ratchet_16x66_switchable_interp_frame1` — SELF-
  PROMOTING (frame-0 KEY byte-exact anchor + frame-1 pinned on the current
  inter-envelope guard, auto-flips to the golden `9d7759bc…` byte-match when
  var-tx + OBMC + WARP all land).
- Compound / interintra / temporal-MV + multi-ref DPB (the `05-mv` / `-00-
  quantizer` frames) come after the single-ref ladder is complete; `05-mv` also
  needs an ALTREF slot in the reference store and order-hint MV projection.
