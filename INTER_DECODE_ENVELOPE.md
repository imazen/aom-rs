# Animated-AVIF inter-decode envelope — status ledger

Mission: decode zenavif's animated test vectors (libavif `colors-animated-*.avif`)
byte-identically to rav1d-safe / aomdec, per shown frame, on the pure-Rust decoder.

Corpus source: `/root/zenavif/tests/vectors/libavif/` (read-only), per-frame AV1
temporal units extracted with `tools/avif-extract` (standalone crate, zenavif-parse).
Local scratch: `local-corpus/animated/<stem>/{frame_<i>.obu, alpha_<i>.obu,
color.obu, alpha.obu, manifest.txt}` (gitignored). Committed test fixtures:
`crates/aom-decode/tests/data/animated/` (the concatenated per-track streams,
~100-250 B each + golden per-frame md5s from aomdec 3.14.1).

## 1. Corpus characterization (MEASURED, 2026-07-23)

Tools: `cargo run -p zenav1-aom-decode --example inspect_headers -- <stream>`
(header inventory via the port's own `read_uncompressed_header`) + the
instrumented libaom (`/root/aom-inspect/examples/inspect -m -r -mm -ct -mv -f`)
for the per-block census, on IVF-wrapped tracks.

### Stream structure

| track | seq | frames (decode order) |
|---|---|---|
| 8bpc color (== 8bpc-audio color) | 150x150 8-bit 420, order_hint bits 7, ref_frame_mvs=1, dual_filter=0, jnt_comp=0, sb128=0, restoration=0 | KEY(show,0xff) ; TU{INTER hidden oh=4 r=0x02 pri=NONE ; INTER hidden oh=2 r=0x04 pri=NONE ; INTER shown oh=1 r=0x08 pri=NONE} ; SHOW_EXISTING idx=2 ; INTER shown oh=3 r=0x10 pri=1 ; INTER shown oh=4 r=0x00 pri=0 |
| 8bpc-alpha / 8bpc-depth color | 150x150 8-bit 420, dual_filter=1, jnt_comp=1, sb128=1, restoration=1 | KEY(0xff) ; 4x INTER shown, refresh {0x02,0x04,0x08,0x10}, primary_ref {6,3,3,3}, interp=SWITCHABLE(4) on... (color track interp=0 f1? measured: color interp=0/4 mixed; alpha track interp=4) |
| 8bpc-alpha / 8bpc-depth alpha | 150x150 8-bit **monochrome**, sb128=1 | same shape as color track; ALL blocks DC_PRED (intra-in-inter), primary_ref {6,3,3,3} |
| 12bpc color | 64x64 **12-bit** 420 profile 2, sb128=1 | KEY ; INTER shown pri=6 (all DC_PRED) ; SEQ+KEY ; SEQ+KEY ; INTER shown pri=6 (NEARESTMV, gm wmtype[2]=3 parsed — unused by blocks) |
| 12bpc alpha | 64x64 12-bit monochrome, sb128=1 | KEY ; INTER pri=6 (NEARESTMV/LAST zero-MV) ; SEQ+KEY ; SEQ+KEY ; SEQ+KEY |

All: superres inert (denom 8), 1x1 tiles, seg=0, no film grain, err_res=0,
disable_cdf_update=0, skip_mode_allowed=0, allow_warped_motion=0 at frame level,
frame dims == seq max dims, no scaled references.

`poc_b_506387278.avif` (fuzz PoC, 2 samples, KEY-only + padding OBUs + a frame
whose 2 tile groups span a sample boundary + a TILE_GRP-only sample) — secondary;
exercises multi-TILE_GROUP-per-frame and padding, not inter.

### Per-block census (instrumented libaom, every frame of every track)

- **Every inter-coded block is single-ref, zero-MV NEARESTMV, SIMPLE_TRANSLATION,
  COMPOUND_AVERAGE(=no masked/wedge), interintra absent.** Non-inter blocks inside
  inter frames are DC_PRED (intra-in-inter, already ported).
- Referenced ref-frame types: LAST(1), GOLDEN(4), ALTREF(7) — mapped to varying
  DPB slots per frame.
- No OBMC, no warped motion, no compound modes, no NEWMV (no MV residual coding),
  no palette/intrabc inside inter frames, zero nonzero motion vectors anywhere.

### Required tools (the REAL minimal envelope), in dependency order

1. **8-slot DPB** (`ref_frame_map`): refresh via `refresh_frame_flags`,
   per-frame ref binding via `ref_frame_idx[7]` (full 3-bit signaling — not
   short signaling), stored per slot: filtered recon, order_hint,
   `ref_order_hints[7]`, end-of-frame CDFs, lf ref/mode deltas, seg data,
   global motion params, per-8x8 MV grid (`MV_REF`), showable flag.
2. **Hidden frames** (`show_frame=0, showable=1`): decode + install, do not
   output. **show_existing_frame**: output stored slot copy (no KEY
   show_existing state reset needed in-corpus — target is an INTER showable).
3. **Forward CDF inheritance** (`primary_ref_frame != NONE`, values {0,1,3,6}
   in-corpus): load ref's saved CDFs + lf deltas + seg + gm as parse context.
   Backward save: end-of-frame adapted CDFs of `context_update_tile_id` tile
   (single-tile here) unless `refresh_frame_context_disabled` (always 0 here)
   — plus `disable_frame_end_update_cdf` handling.
4. **Temporal MV field** (`allow_ref_frame_mvs=1` on every inter frame):
   `av1_setup_motion_field` projection + the temporal-candidate arm of
   `setup_ref_mv_list` (currently explicitly dropped in dv_ref.rs). In-corpus
   all stored MVs are zero/intra so candidates are trivial — but stack counts
   / mode_context / zeromv_ctx depend on the scan being exact.
5. **Multi-ref single-ref symbol decode**: `comp_inter` (reference_select=1 on
   most frames) + `single_ref_p1..p6` reads with real per-block contexts.
6. **Switchable interp filter reads** (interp=SWITCHABLE + enable_dual_filter
   on the alpha-vector seqs): per-block filter symbol(s) incl. dual x/y.
7. **Zero-MV NEARESTMV MC from an arbitrary DPB slot** (integer-pel copy path
   of the existing single-ref MC), 8-bit and **12-bit** (u16 pipeline), SB64 +
   **SB128** inter frames, monochrome inter frames.
8. Driver: repeated mid-stream SEQ headers (already handled), multiple frames
   per temporal unit (already handled), shown-frame-only output.

NOT needed by this corpus: NEWMV/MV decode (readers exist anyway), compound
modes, dist-wtd/masked compound, OBMC, warped/global-motion MC (gm params parse
only), skip_mode, segmentation, scaled references, superres-on-inter,
INTRA_ONLY frames, S-frames, film grain on animation, error resilience,
frame-id numbers, temporal/spatial layers, multi-tile-group frames (poc_b only).

## 2. Status ledger

| Chunk | What | Status |
|---|---|---|
| 0 | Corpus extraction + characterization + this doc | DONE (2026-07-23) |
| 1 | DPB + hidden frames + show_existing + multi-slot driver | not started |
| 2 | primary_ref CDF inheritance + per-slot contexts | not started |
| 3 | temporal MV field + temporal arm of ref-MV list | not started |
| 4 | multi-ref single-ref reads + switchable/dual interp reads | not started |
| 5 | Full-corpus per-frame byte gate vs golden md5 | not started |

## 3. Verification

- Golden refs: aomdec 3.14.1 (`--rawvideo`) per-frame md5 over i420 planes at
  coded depth; cross-checked vs rav1d-safe on the zenavif side (rav1d-safe is
  the production decoder the animation tests compare against).
- Gate: `crates/aom-decode/tests/animated_avif.rs` — per-track, per-shown-frame
  byte compare with per-frame pass/fail reporting; KEY-frame conformance corpus
  stays green (`cargo test -p zenav1-aom-decode`).
