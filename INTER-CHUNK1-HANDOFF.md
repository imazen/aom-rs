# Inter-decode Chunk 1 (walking skeleton) — HANDOFF

**STATUS: ✅ COMPLETE — the frame-1 byte-exact gate is MET** (`av1-1-b8-01-size-64x64` frame 1
decodes to golden md5 `0c189b10dfe6b033c548901ab82dedef`; frame 0 KEY unchanged). See
"Honest fraction" at the bottom for exactly what landed + ratchet limitations. The sections
below are the original de-risk/architecture notes (still accurate).

Read this + `INTER-ROADMAP.md` before continuing Chunk 1.

## TL;DR — the target changed (de-risk finding)

**The roadmap's named target `av1-1-b8-00-quantizer-63` frame 1 is OUT of single-ref
translational scope** and cannot be the walking skeleton. STEP-0 de-risk (instrumented C
decoder, `AV1D_GET_MI_INFO` per-block census) proved frame 1 of q63 uses the FULL inter
tool set: `allow_ref_frame_mvs=1`, **OBMC on 19 blocks, WARPED_CAUSAL on 11 blocks,
interintra on 2 blocks**, 76 inter blocks. "Smallest payload" ≠ "smallest tool surface" —
the roadmap §3 caveat about exactly this fired. Every `-00-quantizer-*` vector's frame 1 is
the same (OBMC+WARP+temporal). Reaching q63 needs Chunks 8 (OBMC) + 11 (local warp) + 13
(interintra), not Chunk 1.

**Corrected walking-skeleton target: `av1-1-b8-01-size-64x64` frame 1.**
- Single SB64, **ONE inter block**, mode **NEWMV**, motion_mode **SIMPLE**, `interp_filter`
  frame-level = SWITCHABLE (the block uses filter type 2 = SHARP), no
  OBMC/warp/compound/interintra/intrabc/skip_mode.
- `primary_ref_frame = 7` (PRIMARY_REF_NONE) → **default CDFs** (no ref-CDF load).
- `tx_mode = 1` (TX_MODE_LARGEST) → **no var-tx read** (single largest tx).
- Block is at mi (0,0) → **empty ref-mv candidate list** (no spatial neighbours;
  temporal + sign-bias + global-motion all inert). NEWMV predictor = global MV = (0,0),
  so mv = coded_diff.
- `allow_ref_frame_mvs=1` but **temporal MV is inert**: verified `tpl_mvs` has 0 valid
  entries after `av1_setup_motion_field` (the only ref is the KEY frame 0, which has no
  motion field; `is_lst_overlay` skips the LAST projection, `get_relative_dist<0` skips
  BWD/ALT/ALT2, and the LAST2 projection reads frame 0's intra mvs → projects nothing).
  So Chunk 9's projection math is NOT needed — just init `tpl_mvs` to INVALID and
  `add_tpl_ref_mv` reads it empty.
- **Frame 0 already decodes byte-identical** (port==C==golden) on this base — foundation solid.

Golden MD5s (`conformance/data/av1-1-b8-01-size-64x64.ivf.md5`):
- frame 0 (KEY): `8e852a5a3f68353612e7024904e8b855`
- **frame 1 (INTER, the target): `0c189b10dfe6b033c548901ab82dedef`**

Ratchet after 64x64: `av1-1-b8-01-size-16x16` frame 1 (4 blocks: 1 NEWMV + 3 NEARESTMV,
EIGHTTAP non-switchable — exercises the spatial ref-mv scan), then `01-size-64x66`, then
the OBMC/warp/interintra frames (Chunks 8/11/13). `01-size-*` sizes 16..66 are all
SIMPLE-heavy; sizes ≥196 and every `00-quantizer`/`04-cdfupdate` pull in OBMC/warp.

### De-risk instrument (throwaway, NOT committed per methodology)
Rebuild the C census tool from source to inspect any vector's frame-1 tools:
```
clang -O2 inspect_frame.c -I reference/libaom -I reference/libaom/build \
  -L reference/libaom/build -laom -lm -lpthread -lstdc++ -o /tmp/inspect_frame
```
`inspect_frame.c` uses the INTERNAL decoder API (`av1_decoder_create(BufferPool*)` +
`av1_receive_compressed_data`), then reads `pbi->common`: dumps `frame_type`,
`allow_ref_frame_mvs`, `switchable_motion_mode`, `reference_mode`, `interp_filter`,
`skip_mode_*`, seq enable flags, `global_motion[].wmtype`, per-mi `MB_MODE_INFO` census
(mode / motion_mode / ref_frame[2] / interintra / compound / use_intrabc via the mi grid),
and `tpl_mvs` valid count. Setup mirrors `av1_dx_iface.c` `init_decoder` (BufferPool +
`av1_alloc_internal_frame_buffers` + `av1_get/release_frame_buffer` cbs; CONFIG_MULTITHREAD=0
so no mutex). This is the STEP-0 tool for every future ratchet target.

## What is LANDED on origin/main
- **Chunk 1c CDF tables** (`b1fe0a3`): the missing inter default CDF tables in
  `crates/aom-entropy/src/default_cdfs.rs` via `xtask/gen_default_cdfs.py`:
  `DEFAULT_INTRA_INTER`, `DEFAULT_SINGLE_REF`, `DEFAULT_NEWMV/ZEROMV/REFMV`, `DEFAULT_DRL`,
  `DEFAULT_SWITCHABLE_INTERP`, `DEFAULT_MOTION_MODE`, `DEFAULT_OBMC`, `DEFAULT_SKIP_MODE`.
  (nmvc reuses the existing `DEFAULT_NMV_JOINTS`/`DEFAULT_NMV_COMPS` = `default_nmv_context`.)
  Byte-exact from the C initializer text (mode CDFs are `av1_copy`'d verbatim by
  `av1_setup_past_independence`, so parse == compiled). Additive, no consumer yet.
- **Chunk 0** (`aom-recon` crate, `reconstruct_txb`) is already wired into aom-decode
  (`crates/aom-decode/src/lib.rs:160` `use aom_recon::reconstruct_txb;`) — use it for 1e.

## Chunk 1d — LANDED (`9b6a061`, verified)
- **`aom-inter` crate** (`crates/aom-inter/src/lib.rs`): single-ref translational MC.
  ```rust
  pub fn build_inter_predictor(
      ref_plane:&[u16], ref_stride:usize, ref_w:usize, ref_h:usize,
      dst:&mut[u16], dst_off:usize, dst_stride:usize,
      blk_x:usize, blk_y:usize, w:usize, h:usize,
      mv_row:i32, mv_col:i32, ss_x:usize, ss_y:usize, filter_x:usize, filter_y:usize)
  ```
  plus `pub fn inter_predictor(..)` (facade) + `pub fn build_mc_border(..)`. Ported from
  `dec_calc_subpel_params` (unscaled), `build_mc_border`/`extend_mc_border`,
  `inter_predictor`→`av1_convolve_2d_facade`→`convolve_2d_facade_single`. u16 bd8 planes
  gathered into a u8 scratch for the aom-convolve kernels.
- **Differentials (byte-identical vs real C), all passing** (`tests/inter_pred_diff.rs`):
  `facade_matches_c` 40k iters vs C `inter_predictor` (sizes {8,16,32,64}², subpel 0..15
  each axis, filters 0..2, incl. dual-filter 2-D); `build_mc_border_matches_c` 40k iters vs
  C `build_mc_border` (all 4 OOB edges); smoke test of the full public path.
- **Limitations (later chunks):** `w<=4`/`h<=4` 4-tap NOT handled (`debug_assert!(w>4&&h>4)`;
  aom-convolve is 8-tap only — the 64x64 target's 64/32-wide blocks are fine, but the 16x16
  RATCHET will hit sub-8-wide blocks → port `av1_get_interp_filter_params_with_block_size`
  4-tap first). No highbd / compound / OBMC / warp / scaled. The frame-edge MV clamp
  `clamp_mv_to_umv_border_sb` (reconinter.h:343, needs `mb_to_*_edge`) is the CALLER's job
  (apply it in the 1c driver before calling `build_inter_predictor`).

## Architecture map (verified file:line — from source, not the roadmap's drifted refs)

### Decode driver (`crates/aom-decode/`)
- `decode_frame_obus(data:&[u8]) -> Result<FrameDecode,String>` — `frame.rs:643`. Single
  temporal-unit, single-KEY-frame. **Hard-errors on a 2nd frame OBU** (`frame.rs:921-923`)
  and on `frame_type != 0` (`frame.rs:537-539`). This is what the multi-frame loop replaces.
- OBU walk: `decode_frame_obus_prefilter` `frame.rs:866`, loop `frame.rs:874`.
- Two-phase header probe: `frame.rs:455-531` (probe `468`, superres re-probe `490-501`,
  final select `504-531`). KEY-only gates `frame.rs:534-542`.
- `parse_frame_header` `frame.rs:374` builds a `FrameHeaderObu` cfg then calls
  `read_uncompressed_header`.
- `FrameDecode` struct `frame.rs:111-162` (cropped tight-strided `Vec<u16>` planes at any bd).
- Working recon `KfTileDecode` `lib.rs:613-630` (SB-aligned, `stride=aligned_mi_cols*4`, NO
  border). Tile driver `decode_frame_tiles_kf` `lib.rs:3123` (fresh
  `KfFrameContext::default_for_qindex` per tile `lib.rs:3140` — for frame 1 primary_ref=NONE
  → this same default path is correct, no ref-CDF load needed).
- Partition read `decode_partition` `lib.rs:2935` → `read_partition` `lib.rs:2970`.
- **Intra** mode-info read `read_mb_modes_kf_fc` call `lib.rs:1689`. THE INTER ANALOG MUST BE
  ADDED here (an `is_inter` arm).
- 64×64-chunk recon loop (`decode_token_recon_block`) `lib.rs:2250-2827`; luma
  predict+`reconstruct_txb` `lib.rs:2418-2468`, chroma `lib.rs:2737-2814`.
- **Intrabc is the closest existing inter analog** (`is_inter_block` on KEY): DV read
  `read_intrabc_info` (`partition.rs:4539`) resolved in driver `lib.rs:1707-1755`
  (`find_dv_ref_mvs` `lib.rs:1722`, `assign_and_validate_dv` `lib.rs:1735`); luma full-pel
  copy `lib.rs:2392-2400`; chroma `intrabc_chroma_predict` `lib.rs:1045`; var-tx read
  `read_tx_size_vartx` `lib.rs:880`. **No RefFrame pool / multi-frame state exists anywhere.**

### Entropy (`crates/aom-entropy/`)
- **All inter symbol readers EXIST + are C-tested (`tests/partition_diff.rs`) but DORMANT**
  (no decode caller). In `partition.rs`: `read_is_inter` `:4340`, `read_ref_frames` `:3837`
  (returns `(is_compound,comp_ref_type,ref0,ref1)`), `read_inter_mode` `:3697`,
  `read_drl_idx` `:3793`, `read_mv` `:3767` (LIVE via intrabc) + `read_mv_component` `:3724`,
  `read_mb_interp_filter` `:4370`, `read_motion_mode` `:4354`, `read_skip_mode` `:4522`,
  `read_inter_compound_mode` `:3684`, `read_interintra_info` `:4599`. Helpers:
  `mode_context_analyzer` `:2927`, `av1_drl_ctx` `:360`, `have_nearmv_in_inter_mode` `:375`.
- **Pred-context helpers EXIST** (`partition.rs`, C-tested): `get_intra_inter_context` `:848`,
  `get_reference_mode_context` `:887`, `collect_neighbors_ref_counts` `:3132`,
  `single_ref_p1_context` `:1007` + P2..P6 family `:1033-1049`.
- **MV-ref scan** `dv_ref.rs` — `find_dv_ref_mvs` `:584` is the intrabc reduction. DROPPED
  for inter (must ADD): `mode_context`/`newmv_count` outputs (feed `read_inter_mode`/`drl`),
  compound branch, temporal `add_tpl_ref_mv`, global-motion MV, sign-bias negation. The
  spatial scan itself is byte-exact + reusable. `DvNbr` `:64`, `DvGrid` trait `:300`.
- **Inter frame HEADER is DONE**: `read_uncompressed_header` `header.rs:2952` parses the full
  inter branch `2982-3032` (ref signaling, frame_size_with_refs, allow_high_precision_mv,
  interp_filter, switchable_motion_mode, allow_ref_frame_mvs) + trailing flags `3081-3090`
  (reference_mode_select, skip_mode_flag, allow_warped_motion, reduced_tx_set) + global
  motion `3091-3093`. `FrameHeaderObu` struct `header.rs:1411`. It needs ref-derived cfg
  inputs (order hints, ref crop sizes, skip_mode_allowed, might_allow_* gates,
  ref_global_motion) — for the 64x64 target these are trivial (single KEY ref, order_hint
  0→1, identity GM, skip_mode_allowed=0).
- `KfFrameContext` `partition.rs:5777` holds intra+intrabc CDFs and OMITS inter CDFs by
  design (doc `:5758-5776`). For the walking skeleton the inter CDFs can be built inline
  from the new `DEFAULT_*` consts per block (single block → no intra-frame adaptation
  matters); the ratchet (16x16, 4 blocks) needs them threaded with `update_cdf`.

## Remaining Chunk-1 sub-steps (ordered, with the minimal-slice shortcuts)

**1a — Ref-frame buffer + 2-frame loop** (aom-decode, new module e.g. `inter.rs`):
- `RefFrame { y/u/v: Vec<u16>, stride*, w/h*, order_hint }` — for 64x64 just frame 0's
  FILTERED recon (post deblock/CDEF/LR/superres, pre-film-grain) at coded size. The MC fn
  clamps reads to `[0,w)×[0,h)` (extend_mc_border), so NO pre-bordering needed.
  primary_ref=NONE → no saved CDFs; temporal inert → no stored ref mvs; GM identity.
- A stateful `MultiFrameDecoder` decoding a sequence of IVF TUs: KEY → existing path +
  store as ref (`ref_frame_map` per `refresh_frame_flags=0xff`); INTER → new inter path
  using the ref. `refresh_frame_flags` for frame 1 = `0xc`. Reuse the tile machinery.

**1b — Inter header state**: feed `read_uncompressed_header` the ref-derived cfg; run
`av1_setup_frame_sign_bias`/`av1_calculate_ref_frame_side`/`av1_setup_skip_mode_allowed`
(all inert/trivial here); init `tpl_mvs` = INVALID (no projection needed).

**1c — Inter mode-info driver** (new in aom-decode, wiring the dormant readers): for the
single block — `read_is_inter`(→1) → `read_ref_frames`(single→LAST) → `find_inter_mv_refs`
(generalize `dv_ref` to emit `mode_context` for the EMPTY-neighbour case + the (0,0)
global predictor) → `read_inter_mode`(→NEWMV) → drl (0 cands → no read) →
`assign_mv`/`read_mv`(pred (0,0)+diff) → `read_mb_interp_filter`(SWITCHABLE, per-dir) →
`read_motion_mode`(→SIMPLE). Then tx: tx_mode=LARGEST → single largest tx (no vartx). Then
`av1_copy_frame_mvs` (store the block's mv per-8x8 — inert for the ratchet's temporal but
needed for later). Verify PARSE against the C instrument's mode/mv/ref census before pixels.

**1d — MC** (from `aom-inter`): luma 64×64 predictor from frame 0's Y at the MV; chroma
32×32 from U/V at the subsampled MV. facade dispatch + border. **First convolve differential
already exists.**

**1e — Reconstruct**: predict (1d) into recon, then `aom_recon::reconstruct_txb` for the
residual (inter ext-tx CDF `DEFAULT_INTER_EXT_TX` already present; coeff read reuses the
intrabc-wired `read_coeffs_txb_full`).

**1f — Gate**: extend `scope_for` (`conformance_corpus.rs:273`) to accept frame 1 of a
2-frame family as an inter target, and assert the port's frame-1 i420 md5 ==
`0c189b10dfe6b033c548901ab82dedef` AND frame 0 still matches. Plus the per-kernel
differentials (MC facade = 1d's; inter MV list; inter mode parse). A stateful C decode
oracle (`aom-sys-ref` `ref_decode_av1_stream(tus, n)`) helps pixel-diff debugging (decode
frame 0 then N via the public codec API, return frame N's planes) — append to aom-sys-ref
AFTER the 1d agent lands to avoid a shim collision.

## Honest fraction — ✅ WALKING SKELETON COMPLETE (frame-1 byte-exact gate MET)

**The frame-1 byte-exact gate is MET.** `av1-1-b8-01-size-64x64` frame 1 decodes to the golden
md5 `0c189b10dfe6b033c548901ab82dedef` AND frame 0 (KEY) stays byte-identical, via the new
multi-frame `aom_decode::frame::decode_frames`. Gate: `aom-decode/tests/inter_walking_skeleton.rs`
(`inter_walking_skeleton_frame1_byte_identical`, a true `md5_helper.h::Add` byte-identity assert).

STEP-0 census (`/tmp/inspect_frame.c`, throwaway) proved frame 1 is the simplest possible inter
frame: ONE `BLOCK_64X64` `PARTITION_NONE` `NEWMV` block, `skip=1` (**pure MC, NO residual — no
coeff read / no `reconstruct_txb`**), single `LAST` ref, `SIMPLE_TRANSLATION` (no overlappable
neighbours → `motion_mode_allowed` early-outs, NO motion_mode symbol), `interintra=0` (64×64 not
allowed → no read), `interp=[SHARP,SHARP]`, `mv=(-1,-7)`, `tx=TX_64X64`, `primary_ref=NONE`,
`allow_high_precision_mv=1`, `tx_mode=LARGEST`, `allow_ref_frame_mvs=1` but tpl empty (0 valid).

### What landed (this continuation)
1. **Generalized MV scan** (`38890de`/`cdba774`, pushed): `find_inter_mv_refs` in `dv_ref.rs` +
   `shim_find_inter_mv_refs`/`ref_find_inter_mv_refs` + `find_inter_mv_refs_matches_c` (3000×200
   grids, value-identical vs real C `av1_find_mv_refs`). mode_context/newmv_count/gm-fill/sign-
   bias/integer-mv restored. Intrabc `find_dv_ref_mvs` regression unchanged. Fixed latent
   `GLOBAL_GLOBALMV` 21→23.
2. **1a/1b/1c/1e/1f** (this commit): `RefFrame` + `decode_frames` multi-frame OBU walk +
   `run_post_filters`/`finish_and_grain`/`build_tile_cfg` extractions (KEY path behaviour-
   preserved); `InterFrameCfg` + inline `InterCdfs` (from the chunk-1c DEFAULT_* tables) +
   `TileKf::decode_block_inter` (the faithful inter mode-info reads: skip → is_inter →
   ref_frames → find_inter_mv_refs → inter_mode → drl → mv → interp_filter) + `build_inter_
   predictor` luma+chroma MC + `clamp_mv_to_umv_border` + `decode_frame_tiles_inter`. Header fix:
   `read_uncompressed_header` now gates `allow_high_precision_mv` on the STREAM-read
   `cur_frame_force_integer_mv` (the `force_integer_mv==SELECT` bit), not the caller cfg input —
   the desync that first blocked the gate (seq `force_integer_mv==2`).

### Known limitations (documented, NOT blocking the skeleton; ratchet work)
- Inter CDFs are built **inline per block** (correct for ONE block — no CDF is re-read within it);
  the multi-inter-block ratchet (`01-size-16x16`, 4 blocks incl. NEARESTMV) needs them threaded +
  `update_cdf` like `KfFrameContext`, plus the interp-filter neighbour context (a filter grid) and
  the deblock inter mode-delta.
- `clamp_mv_to_umv_border` is a faithful LUMA-domain clamp — exact when it doesn't fire (every
  `01-size-*` target); an MV that clamps *differently* per plane needs the per-plane chroma clamp.
- `decode_block_inter` asserts the envelope (seg/skip_mode/delta-q off, single LAST, skip=1); the
  ratchet must generalize (non-skip residual via `reconstruct_txb`, NEAREST/NEAR/GLOBAL modes).

## Chunk 2 — the 16x16 ratchet — ✅ COMPLETE (frame-1 byte-exact gate MET)

**`av1-1-b8-01-size-16x16` frame 1 decodes to golden md5 `f4b0078dfbc8b581fa959d4512b9940a`**
(`.md5` line 2), frame 0 (KEY) unchanged, and the 64x64 skeleton + full KEY conformance stay green.
Gate: `aom-decode/tests/inter_ratchet.rs::inter_ratchet_16x16_frame1_byte_identical`.

STEP-0 census (public-API `AV1D_GET_MI_INFO` + an internal-decode `tx_type_map` peek, both
throwaway): the 16x16 SB is `PARTITION_HORZ_4` -> 4× `BLOCK_16X4`: block 0 NEWMV, blocks 1-3
NEARESTMV (each MV from the spatial scan of the block above), single LAST, EIGHTTAP non-switchable,
SIMPLE_TRANSLATION, tx=TX_16X4, **every block skip=0 (carries residual)**; frame flags interp=0
(non-sw) reference_mode=SINGLE tx_mode=LARGEST disable_cdf_update=0(adapts) cdef_bits=0(NO
read_cdef) reduced_tx=0(FULL ext-tx set) allow_warped=1 error_resilient=0.

### What landed
- **4-tap interp** (`cfd39e0`, pushed): `aom-inter` selects the 4-tap kernel per direction (x by w,
  y by h; side<=4). `facade_4tap_matches_c` locks it vs real C for the 16x4 luma / 8x2 chroma shapes;
  the 8-tap path is byte-unchanged.
- **Inter CDF threading (item 2):** `InterCdfs` is now a persistent `TileKf` field, reset to
  defaults per tile in `start_tile`, snapshotted+adapted+persisted per block (the `single_ref`
  sub-tree's adaptations copied back through `ref_frame_cdfs`). Blocks 2-4 desync without it.
- **Spatial NEARESTMV scan (item 1):** `find_inter_mv_refs` produces the nearest MV from the
  neighbour scan; the 3 NEARESTMV blocks read their MV from it (matches census).
- **Non-skip residual (item 5):** per block, MC (predict) then read coeffs + `reconstruct_txb` ADD
  onto the prediction — luma (TX_16X4, inter ext-tx symbol via `inter_ext_tx_cdf`), chroma at the
  chroma-reference block (sub-8x8 8x4 U/V; tx-type = co-located luma tx-type). Entropy-context
  threading (`get_txb_ctx` / `set_entropy_ctx` over `above_e`/`left_e`) mirrors the intrabc path.
- **Sub-8x8 chroma MC (item 5):** chroma predicted only at the chroma-reference block (odd mi_row),
  as per-4x4-subblock 8x2 strips using each covered luma block's own MV
  (`build_inter_predictors_sub8x8`). All 4 MVs are equal here, but the strip walk is faithful.
- **THE ROOT-CAUSE BUG:** `might_allow_warped_motion` was never set (defaulted false), so
  `read_uncompressed_header` SKIPPED the `allow_warped_motion` bit, shifting `reduced_tx_set` (which
  then read the warped bit =1). The 1-bit slip is hidden by the header's trailing byte-alignment, so
  partition/mode/mv all parsed correctly, but the tx-type set was the REDUCED (DCT_IDTX) set instead
  of the full set -> tt=IDTX where C uses V_FLIPADST -> wrong residual (luma fully wrong, chroma
  mostly). Fix: `parse_frame_header_ext` sets `cfg.might_allow_warped_motion = enable_warped_motion`;
  `read_uncompressed_header` combines it with the parsed `!intra && !error_resilient` gate.

### Inert for THIS vector (deferred, NOT false-completed)
- **Interp-filter neighbour context (item 3):** NOT exercised — the frame is non-switchable EIGHTTAP,
  so `read_mb_interp_filter` early-returns with NO symbol/context read (the task's "the context read
  still happens" premise is FALSE here — verified). The switchable no-neighbour context (used by the
  64x64 skeleton) is kept; a switchable block WITH neighbours asserts (guarded). Porting
  `av1_get_pred_context_switchable_interp` + the filter grid is the next SWITCHABLE target's work.
- **Per-plane chroma MV clamp (item 6):** the luma-domain `clamp_mv_to_umv_border` is exact when it
  doesn't fire (it doesn't for these small MVs). A per-plane clamp is later-chunk.

### Next ratchet

## Chunk 3 — partial-edge single-ref — ✅ COMPLETE (`01-size-64x66` frame-1 byte-exact)

**`av1-1-b8-01-size-64x66` frame 1 decodes to golden md5 `86f20606b0408bd3ba6771a6a37df429`**
(`.md5` line 2), frame 0 (KEY) unchanged + the 64x64 skeleton + 16x16 ratchet stay green.
Gate: `aom-decode/tests/inter_ratchet.rs::inter_ratchet_64x66_partial_edge_frame1_byte_identical`.

STEP-0 census (`/tmp/inspect_frame`) of the four partial-edge targets found only **64x66** stays
in single-ref translational scope: its frame 1 is ONE `BLOCK_64X128` clipped to 64x66 (a
`use_128x128_superblock` frame → `split_or_vert` forced partition at the right edge), NEWMV
mv=(-1,-7), single LAST, SIMPLE, skip=1 (pure MC, no residual). The other three pull in OBMC /
WARP / switchable-interp and are **out of Part-A scope → chunk-4+ feature targets** (see
`INTER-FEATURES-PLAN.md`).

**Root cause of the partial-edge divergence (NOT the MV clamp).** `clamp_mv_to_umv_border`
(reconinter.h:343) was already implemented and does NOT fire for 64x66 (the MV is far inside the
border). The real bug: the MC reference dims. C's `av1_setup_pre_planes` loads
`src->crop_widths/crop_heights` (the VISIBLE 64x66 / UV 32x33) into `pre_buf->width/height`, and
`build_mc_border` edge-replicates against those. The port's `RefFrame` stored the SB/mi-aligned
recon extent (64x72 / UV 32x36), so a bottom-edge block's interp taps read the invisible
mi-aligned recon rows past the crop boundary instead of replicating the last visible row → the
bottom output rows diverged. **Fix:** `RefFrame::from_filtered` now stores the frame's crop
dims (`aom-decode/src/lib.rs` + the caller in `frame.rs`); the MC kernel (`build_mc_border`,
already C-differentially-locked in `aom-inter/tests/inter_pred_diff.rs` across all 4 OOB edges)
is unchanged — only the dims fed to it. 64x64/16x16 were byte-exact already because for them the
visible dims equal the mi-aligned dims (multiples of 8px); 64x66 is the first where they differ.

## Chunk 4 — OBMC — ✅ COMPLETE (`01-size-16x18` frame-1 byte-exact)

**`av1-1-b8-01-size-16x18` frame 1 decodes to golden md5 `08db98983320105666c9496dc1dba209`**
(`.md5` line 2), frame 0 (KEY) unchanged + the 64x64 skeleton + 16x16 + 64x66 ratchets stay green.
Gate: `aom-decode/tests/inter_ratchet.rs::inter_ratchet_16x18_obmc_frame1_byte_identical`.

STEP-0 census: frame 1 = four `BLOCK_4X16` strips (NEWMV + 3 NEARESTMV, SIMPLE) above ONE
`BLOCK_16X8` `OBMC_CAUSAL` block at mi(4,0), single LAST, non-switchable EIGHTTAP. The OBMC block
is at mi_col 0 (frame LEFT edge → no left neighbour) with a full `BLOCK_4X16` row above, so ONLY
the ABOVE-neighbour blend fires; chroma above-OBMC is skipped (`av1_skip_u4x4_pred_in_obmc` →
BLOCK_8X4 dir 0). num_proj_ref=4, allow_warped_motion=1 → motion_mode ceiling = **WARPED_CAUSAL**
(the 3-symbol `motion_mode_cdf`, resolving to OBMC).

**What landed (three sub-features — the census under-stated the scope):**
1. **motion_mode read** (`decode_block_inter`, reordered BEFORE interp per decodemv.c): the
   `motion_mode_allowed` ceiling (`has_overlappable_neighbors` = `foreach_overlappable_nb_above/left`
   + `num_proj_ref` = count-only `av1_findSamples`) → `read_motion_mode`. WARPED_CAUSAL result guarded
   (chunk 5). `DEFAULT_OBMC` / `DEFAULT_MOTION_MODE` CDFs threaded on `InterCdfs`.
2. **OBMC blend** (aom-inter kernels + `obmc_above_blend`/`obmc_left_blend`): `av1_get_obmc_mask` +
   `aom_blend_a64_vmask`/`hmask`, **differentially locked vs real C** in
   `aom-inter/tests/inter_pred_diff.rs` (`obmc_mask_matches_c`, `blend_a64_masks_match_c` + the new
   `obmc_shim.c` oracle). The above-neighbour luma strips reuse the existing `build_inter_predictor`
   with the neighbour's mv/ref/filter into a scratch, then vmask-feather into the block's own
   predictor over the top `overlap = min(bh,64)>>1` rows. `clamp_mv_umv_border_sb` faithful (inert
   here). Left blend written+guarded (inert: left frame edge). Non-switchable → neighbour filter =
   frame filter (switchable-neighbour filter = chunk 6).
3. **inter var-tx** (`TX_MODE_SELECT`, MISSED by the plan's "OBMC only" census): the OBMC block
   splits TX_16X8 → uniform 2× TX_8X8. Wired `read_tx_size_vartx` (reusing the intrabc var-tx
   machinery) into `decode_block_inter`; the uniform-leaf residual loop uses the read tx size.
   Non-uniform partitions guarded (`collect_vartx_leaves` path is a later target).

**THE ROOT-CAUSE BUG (found via tell/rng bit-trace vs an instrumented throwaway C decoder):** the
port desynced only the OBMC block's LUMA (chroma + strips byte-exact, because the OBMC block is the
LAST block and its chroma has no residual). Same context + CDF + bit-position as C at the var-tx
split, but different `rng` → an earlier read narrowed the range differently. The culprit: the
**inter-intra flag** (`decodemv.c:1387`, `interintra_cdf[size_group_lookup[bsize]]`) is coded for an
interintra-allowed block (BLOCK_8X8..=BLOCK_32X32 — the BLOCK_16X8 OBMC block qualifies; the
BLOCK_4X16 strips do NOT) when `enable_interintra_compound`. It is 0 (no inter-intra), but the flag
read still narrows the arithmetic range — the port skipped it and desynced. **Fix:** read the flag
(new `DEFAULT_INTERINTRA` CDF + `enable_interintra_compound` on `InterFrameCfg`), assert 0
(inter-intra prediction is a later chunk). Placed AFTER the mv read, BEFORE motion_mode (C order).

## Chunk 5+ — feature ladder → see `INTER-FEATURES-PLAN.md`

Next chunk 5 = **WARP** (WARPED_CAUSAL), target `01-size-16x34` frame 1 (mix of SIMPLE + OBMC +
**one WARPED_CAUSAL** @ mi(4,0)); needs `av1_findSamples` full sample record + `av1_find_projection`
/ `av1_warp_plane`. Then switchable-interp-nbr (`16x66`, also needs OBMC+WARP), then compound /
interintra prediction / temporal-MV + multi-ref DPB (the `00-quantizer` / `05-mv` frames —
full-toolset, censused as NOT minimal). Full C file:line map + differentials in that doc.

**New envelope guard (chunk 4):** `decode_block_inter` now asserts the block's reference has
IDENTITY global motion (`gm_wmtype[ref0] == 0`) — `find_inter_mv_refs` still hardcodes the (0,0)
global-MV base, correct for every target through 16x18 (all identity-GM). A non-identity-GM frame
pins cleanly here instead of desyncing on a wrong NEWMV base.

**DISCOVERED chunk-6 blocker (16x66) — mi(0,0) mode-info desync, NOT global motion.** Landing OBMC
removed the `TX_MODE_LARGEST` assert that used to pin `16x66` frame 1 at its very first block, so
its mode-info now runs for the first time. `mi(0,0)` (BLOCK_16X16, NEWMV, switchable) DESYNCS: the
port decodes **mv=(0,-15)** where C codes **(-1,-7)**, so the inter-intra flag then reads a garbage
1 and trips the inter-intra guard. Ruled out: global motion (LAST is IDENTITY), and chunk-4 OBMC
(16x18 is byte-exact through the SAME merged mode-info path). It is a pre-existing switchable-frame
mode/drl/mv-read bug for a 16x16 block — the switchable-interp + WARP track must root-cause it (the
`inter_ratchet_16x66` gate is pinned on the SYMPTOM and documents this).
