# Inter-decode Chunk 1 (walking skeleton) — HANDOFF

Status as of this landing. Read this + `INTER-ROADMAP.md` before continuing Chunk 1.

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

## Honest fraction
DONE (verified, on main): STEP-0 de-risk (+ corrected target with full evidence); Chunk 1c
CDF tables (`b1fe0a3`); **Chunk 1d MC crate `aom-inter` (`9b6a061`, 80k-iter differentials
byte-identical vs C)**. NOT STARTED: 1a (multi-frame loop + RefFrame — aom-decode has zero
multi-frame state), 1b (inter header state wiring), the 1c inter mode-info DRIVER +
generalized `dv_ref` scan (mode_context/newmv_count), 1e (reconstruct), 1f (gate). **The
frame-1 byte-exact gate is NOT met — the skeleton is NOT done.**

### Recommended continuation order (MC + CDFs are now ready)
1. **1c generalized scan first** (self-contained, cleanly differential-verifiable now that
   aom-sys-ref is free): extend `dv_ref.rs` `find_dv_ref_mvs` → an inter `find_inter_mv_refs`
   emitting `mode_context`/`newmv_count` (restore the dropped `av1_mode_context_analyzer`
   arm + `process_single_ref_mv_candidate` sign-bias + gm_mv). Differential: extend
   `shim_find_dv_ref_mvs` (dec_shim.c) to a real LAST_FRAME ref + extract `mode_context`.
   For the 64x64 target this returns EMPTY + the isolated-block mode_context.
2. **1a multi-frame loop + RefFrame** (aom-decode): stateful decoder; frame 0 via the
   existing path (verify frame-0 md5 unchanged as a regression), store filtered recon as a
   RefFrame; parse frame 1's header.
3. **1c driver** wiring the dormant readers (+ the scan) for the single inter block, then
   **1e** (`build_inter_predictor` → `reconstruct_txb`), then **1f** (`scope_for` + md5 gate).
Verify the PARSE against the `/tmp/inspect_frame` census (mode=NEWMV, ref=LAST, the mv)
before chasing pixels.
