# HANDOFF — screen-content stills track (#29): palette + intrabc

Session end state (2026-07-17). Author context: the screen-content bulk-port agent.
Everything below `LANDED` is on origin/main and verified; everything below `SKELETON`
is committed on this worktree branch only (`wip:` commit), compiles, but is
integration-incomplete — grep `// HANDOFF:` in the code for the precise gaps.

## LANDED on origin/main (verified `merge-base --is-ancestor`, full suite green)

Landing 1 = commits `0c60c65..4e70236` (wip + style + palette feat + intrabc 3a + KB-11 fixups):

1. **Palette Y+UV RD search — COMPLETE, RD-close-gated, 5/7 cells byte-exact.**
   - `crates/aom-encode/src/palette_search.rs` (1631 lines): the entire
     `av1_rd_pick_palette_intra_sby`/`_sbuv` machinery (dim-1/2 k-means, top colours,
     colour/map costs, header-rd gating levels 0/1/2, chroma early-term, chroma
     sub-8x8 dims). All speed levels' sf values threaded (gate covers speed 0).
   - Integration: map-fill prediction through the Y/UV tx walks + CfL re-encode +
     `encode_b_intra_dry` + pack (syntax + colour-map tokens via the pre-existing
     bit-exact entropy writers); `ModeGrid`/`MiNbrGrid` palette neighbour state
     (cache + flag ctx) on both walks; `PaletteCosts` in `derive_real_costs`.
   - Knob: `PickFrameCfg::palette_costs: Option<&PaletteCosts>` (None = every
     pre-existing envelope, byte-stable by construction).
   - GATE: `crates/aom-bench/tests/rd_close_palette.rs` — text/UI screen cells +
     real-content control, both sides `--enable-palette=1 --enable-intrabc=0` via
     `ref_encode_av1_kf_screen_content` / `EncodeCell::port_encode_with(bootstrap, true)`.
     Result: 5/7 EXACT (byte-identical incl. mono + all 64² + control), 2 CLOSE
     (128²: +2.55%/−1.04 zensim port-better; 0.00%/+0.19). PARITY.md section B row 1.
   - Fixed latent bug: UV no-palette FLAG cost was off; C costs it per leaf via
     `av1_allow_palette(sct, bsize)` regardless of `--enable-palette`
     (partition_pick.rs per-leaf `uv_lp.try_palette` recompute).
   - **Byte-exactness follow-up:** the two 128² CLOSE cells are multi-SB near-ties —
     first localization candidates: per-SB decision dump (KB-2 methodology); suspects
     = neighbour palette cache/ctx state across SB boundaries during candidate churn,
     k-means determinism at partial data. See STATUS.md "#29 Palette RD search" entry.

2. **Intrabc decoder recon — pre-existing, verified present** (aom-decode: DV grid,
   `intrabc_chroma_predict`, var-tx read, conformance-gated). Nothing to do.

3. **Intrabc chunk 3a — hash + DV costs (unit-gated building block).**
   `crates/aom-encode/src/intrabc_search.rs`: CRC-32C (KAT-tested), the full
   source-frame hash-table build/query (`hash_motion.c` — hierarchical
   no-two-adjacent insertion, 256/bucket, LE u32[4] CRC serialization matching the
   x86 oracle), `av1_fill_dv_costs`/`av1_build_nmv_cost_table` at MV_SUBPEL_NONE
   over the port's 69-u16 nmv packing, `mv_cost`/`av1_mv_bit_cost(120)`/
   `mv_err_cost(shift 14)`/`mvsad_err_cost(shift 9)`, generic variance/SAD.

## SKELETON on this worktree branch only (compiles; NOT wired, NOT validated)

`crates/aom-encode/src/intrabc_search.rs`, second half (this session's final commit):

- `DEFAULT_TXFM_PARTITION_CDF` (21 rows, copied from the decoder's TileKf which
  carries its own "relocate into KfFrameContext" FORK NOTE — do that relocation) +
  `fill_txfm_partition_costs` (rd.c:108 slice).
- `intrabc_predict_luma` (full-pel recon copy) + `intrabc_predict_chroma`
  (the decoder's 2-tap closed forms, mirrored — diff-test against
  aom-decode's `intrabc_chroma_predict` on random DVs before trusting).
- `DvCell` (per-mi DV/skip projection -> `DvNbr`), `FullMvLimits`,
  `set_mv_search_range` (mcomp.c:233; MAX_FULL_PEL_VAL=1023 — VERIFY against
  mcomp.h, I wrote both 2047 and 1023 candidates, kept 1023).
- `rd_pick_intrabc_mode_sb` (rdopt.c:3427): dv-ref (via the ported
  `dv_ref::find_dv_ref_mvs` — **VERIFY the 4-tuple return order** nearest-vs-near
  against the decoder call site), ABOVE/LEFT direction mv limits (C:3510-3542),
  the hash candidate loop (`av1_intrabc_hash_search` with variance +
  `mv_err_cost`), `is_dv_valid`, and the SKIP-ARM-ONLY RD.

### `// HANDOFF:` gaps inside the skeleton (in file, exact locations marked)
1. **Coeff arm** — currently skip-arm only, which is BIASED (under-codes inexact
   matches). Implement per the in-file note: per-txb `xform_quant_optimize`
   (DCT_DCT, max uniform tx; the `encode_intra.rs:438-465` call pattern),
   `txfm_partition` no-split costs, inter tx-type cost
   (`TxTypeCosts.inter[eset][sqr][DCT]` — **`derive_real_costs` fills inter with a
   DUMMY zero cdf today; fill from `kf.inter_ext_tx`**), skip0 cost, then
   `min(skip, coeff)` per `av1_txfm_search` (tx_search.c:3856-3908).
   Do NOT run the RD-closeness gate with skip-only in place.
2. **hbd sse scaling** in the skip arm (`ROUND_POWER_OF_TWO(sse, 2*(bd-8))`
   before `<<4`) — reuse tx_search.rs's pixel-dist helpers instead of the inline.
3. **No full-pel diamond (NSTEP) / mesh search** (mcomp.c:1481/1615/1768) — the
   next chunk after the coeff arm; hash-only covers exact repeats. Screen
   `exhaustive_searches_thresh = 1<<20`.
4. C's tx set for the coeff arm is the full INTER set + var-tx quadtree
   (av1_pick_recursive_tx_size_type_yrd) — chunk after that.

### Integration map (NOT started — the wiring TODO list, in dependency order)
1. `ModeGrid` (partition_pick.rs): add `dvs: Vec<DvCell>` + `skips` (allocated when
   intrabc on, like `pal_sizes`); stamp at all 23 stamp sites from the winner
   (regex on `.palette_uv.as_ref(),` finds them); `dv_grid` closure for the search =
   bounds-checked read relative to (mi_row, mi_col).
2. `LeafWinner` (encode_sb.rs): `use_intrabc: bool, dv_row/col: i32,
   dv_ref_row/col: i32` (+ skip_txfm already exists). `IntraSbyBest` untouched —
   the intrabc hook lives at `rd_pick.rs` step 6 (the documented no-op site),
   comparing `IntrabcBest.rdcost` against the assembled intra `rd`.
3. `rd_pick.rs`: build `IntrabcLeafArgs` (skip_ctx from the DvCell grid via
   `skip_txfm_context`; `error_per_bit = max(rdmult >> 6, 1)`), call after step 5;
   on win: overwrite the winner tuple (mode fields dead, matching C's
   `*mbmi = best_mbmi` semantics where intrabc sets DC_PRED/UV_DC_PRED).
4. `encode_b_intra_dry` (encode_sb.rs): intrabc arm — predict via
   `intrabc_predict_luma/chroma` into recon; skip arm: recon = pred +
   `av1_reset_entropy_context` stamps (zeros) + `set_txfm_ctxs(bw, bh)` skip-inter
   convention; coeff arm: subtract + `xform_quant_optimize` + inverse-add per txb
   (produce `TxbEncode` with the persistent-array write-ctx pair — the KB-6
   pack-write-ctx lesson applies verbatim).
5. `pack_leaf` (pack.rs): `info.use_intrabc=1, dv_row/col = dv - dv_ref` — CHECK
   `write_intrabc_info`'s diff convention (it takes diff_row/diff_col directly);
   `kfs.allow_intrabc` from a new `PackCfg::allow_intrabc` (= parsed header bit);
   skip: NO tx_size/coeff writes (write_mb_modes_kf_fc's skip flag already coded);
   non-skip: `write_tx_size_vartx` (exists, aom-entropy:1400) with
   `inter_tx_size[all] = max_tx` + a `[[u16;3];21]` txfm cdf carried per tile +
   the inter tx-type cdf slice `kf.inter_ext_tx[eset][sqr]` into
   `write_coeffs_txb_full(is_inter=true)`; nbr stamp: skip + DV twin.
   ALSO: intrabc blocks write NOTHING intra after the DV (write_kf_tail early
   return ✓ already implemented in aom-entropy).
6. Frame plumbing: `PickFrameCfg::enable_intrabc` (+ hash table built ONCE per
   frame in `pack_tile` from the SOURCE luma via `build_intrabc_hash_table` when
   enabled — C builds in encodeframe.c:2199 from cpi->source); `DvCosts` from
   `kf.ndvc_*` via `fill_dv_costs` at frame start; `fill_txfm_partition_costs`.
7. Harness (aom-bench lib.rs `port_encode_with`): add `enable_intrabc` knob;
   when the parsed header has `allow_intrabc`: **LF derivation must be skipped —
   levels forced 0** (C: `!coded_lossless && !allow_intrabc` gates
   av1_pick_filter_level; the decoder forces 0 the same way). CDEF/LR off anyway.
8. Gate: extend `rd_close_palette.rs` (or a sibling `rd_close_intrabc.rs`) with
   `c_encode_screen(enable_palette, enable_intrabc=true)` cells on EXACT-REPEAT
   content (tiled text — the hash search's home turf) + anti-vacuous witnesses
   (PORT(on) != PORT(off), C(on) != C(off)) + the real-content control.

### Validation recipe (per PARITY.md rules)
- Unit: chroma-predict diff vs decoder; `find_dv_ref_mvs` tuple-order witness;
  hash build/query agreement (landed test covers this).
- E2E: `cargo test -p aom-bench --test rd_close_palette -- --nocapture` — bands
  |Δsize| ≤ 5%, Δzensim ≤ 0.5; expect EXACT only after the coeff arm + diamond land.
- Suite ONCE before landing: `cargo test -p aom-encode -p aom-bench -p aom-entropy`
  (the other crates are untouched by this track — verify with
  `git diff origin/main -- crates/aom-{entropy,decode,...}` = empty).
- Landing: pathspec-scoped commits, author `aom-rs <lilith@imazen.io>`, trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, push `HEAD:main`,
  verify `git merge-base --is-ancestor HEAD origin/main`, PARITY.md row same commit.

### Known coordination facts
- The standard shim (`ref_encode_av1_kf`) hardcodes `--enable-palette=0
  --enable-intrabc=0` — every pre-existing byte gate depends on the port keeping
  both knobs OFF there. `ref_encode_av1_kf_screen_content` exposes both.
- `x->color_palette_thresh = 64` (encodeframe.c:1297). `min_alloc_size = 4`
  (mi_alloc_bsize BLOCK_4X4 for the ≤512² test frames).
- Speed-0 palette sf: prune_search 0 / size-search 1 (header-rd shift 1) /
  chroma early-term 1. Intrabc sf: `intrabc_search_level=0` (both directions,
  hash + pixel), `use_intrabc=1`, `hash_max_8x8_intrabc_blocks=0`,
  speed>=1 adds `prune_intrabc_candidate_block_hash_search=1` (count cap 64).
