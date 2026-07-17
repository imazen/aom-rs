# HANDOFF — transform-SIMD campaign (Gate 3), 2026-07-17

Agent died at spend limit mid-campaign; coordinator salvage-committed WIP
(`32882c5`, on top of my `3735b3c`). **Everything below is on this branch and
`cargo check -p aom-transform` PASSED (0 warnings shown) immediately before
death.** NOT yet run: any test. NOT pushed to main.

## What is built (all committed)

The ENTIRE transform-SIMD stack per the STATUS.md (a56744f) design — all 25
1-D lane kernels + all 4 vector passes + driver wiring:

| piece | file | state |
|---|---|---|
| transpiler `--lanes` mode | `xtask/transpile_txfm1d.py` | done; scalar emission verified byte-identical to committed gen files (full `diff` — run again after any edit) |
| inverse lane kernels idct4/8/16/32/64 + iadst8/16 | `crates/aom-transform/src/simd/inv1d_v3_gen.rs` (GENERATED) | done |
| forward lane kernels fdct8/16/32/64 + fadst8/16 | `crates/aom-transform/src/simd/txfm1d_v3_gen.rs` (GENERATED) | done |
| hand lane kernels fdct4, iadst4 (i64-pair), fadst4, i/fidentity4/8/16/32 | `crates/aom-transform/src/simd/hand_v3.rs` | done |
| helpers hb/clampv/negv/rshiftv/mul_rshiftv/shl_clamp64v/revv/widen16/transpose8/low32_of_i64 | `crates/aom-transform/src/simd/mod.rs` | done |
| inv col pass (8-col batches) + inv row pass (8-row, transpose stores) | `simd/mod.rs` `try_inv_{col,row}_pass` | done, wired into `inv_txfm2d.rs` |
| fwd col pass (8-col, i16 loads, shl-clamp input stage) + fwd row pass (8-row, transpose loads) | `simd/mod.rs` `try_fwd_{col,row}_pass` | done, wired into `txfm2d.rs` |
| per-kernel SIMD==scalar unit differential (all 25 kernels, token permutations) | `simd/mod.rs` `mod tests` | WRITTEN, **NEVER RUN** |
| `just gen-txfm1d` regeneration recipe | `justfile` | done |

Scalar paths are byte-untouched: both drivers keep their original loops
verbatim inside `if !simd_done { ... }`; non-x86_64 (incl. i686) compiles the
scalar loop only (cfg at call sites + `lib.rs` `#[cfg(target_arch="x86_64")]
pub(crate) mod simd`).

## IMMEDIATE next steps (in order — nothing else before these)

1. `just test` — expect the new `simd::tests::inv1d_v3_bit_identical_to_scalar_at_every_tier`
   plus ALL existing transform differentials (`inv_txfm2d_diff`, `txfm2d_diff`,
   `fdct_diff`, `txfm1d_diff`, `inv_txfm1d_diff`) to pass. The existing 2-D
   C-differentials now drive the SIMD passes live automatically (dispatch is
   inside `av1_inv_txfm2d_add` / `fwd_txfm2d_core`) — they are the
   SIMD-vs-C end-to-end pins, ~400k comparisons, all 193 combos × bd.
2. `just test-scalar` — the AOM_FORCE_SCALAR pin must stay 0-failed (scalar
   loops untouched; the unit differential stays non-vacuous because
   `for_each_token_permutation` RESETS token disabled-state itself —
   verified in archmage 0.9.27 source, testing.rs:351).
3. Missing test to ADD (designed, not written): 2-D permutation-equality
   integration test — for each token permutation run `av1_fwd_txfm2d` +
   `av1_inv_txfm2d_add` over all valid (tx_type, tx_size) × bd × random
   inputs (coeffs ±2^20 for inv; full-range i16 for fwd) + zeros/spikes,
   collect outputs, assert byte-equality ACROSS permutations (the all-off
   permutation IS the scalar reference). This pins the PASS plumbing
   (flips, clamps, shifts, transposes) SIMD-vs-scalar beyond what the vs-C
   harnesses cover (they use ±2^16 coeffs; bd12 clamp bound ±2^19 is only
   reachable SIMD-vs-scalar because C's half_btf overflows UB there).
4. If unit-test runtime is excessive (25 kernels × 4 sr × 4 cos_bit × ~176
   batches × permutations), cut `rep` counts (24→8) BEFORE cutting pattern
   coverage; never drop the bound-sign patterns or full-i32 arm.
5. Then: both-modes full suite, pathspec-scoped landing commit(s), push,
   `git merge-base --is-ancestor` verify, re-profile
   (`just profile enc port enc_s0_128_cq32 6` + dec cell), commit Ir deltas
   to `benchmarks/` + STATUS.md. Box was DEDICATED — no nice needed per
   memory (but keep logs in /root/.claude/jobs/3651b35b/tmp/).

## Verified API facts (cost me real probing — trust these)

- **Safe raw intrinsics under `#[forbid(unsafe_code)]`**: VALUE intrinsics
  (`_mm256_mullo_epi32`, `cvtepi32_epi64`, `srl/sll_epi64`, `permutevar8x32`,
  `blend_epi32`, `blendv_epi8`, `cmpgt_epi64`, `mul_epi32/epu32`, unpack*,
  `permute2x128`) are callable WITHOUT `unsafe` inside `#[arcane]`/`#[rite]`
  fns (their `#[target_feature]` region). Verified by compile+run on rustc
  1.97. MEMORY intrinsics (loadu/storeu) are NOT — use
  `i32x8::from_slice/from_array/store` (store takes `&mut [i32; 8]`:
  `(&mut buf[a..a+8]).try_into().unwrap()`).
- **`#[arcane]` takes NO argument** — it reads the token type from the fn
  signature (`#[arcane(X64V3Token)]` is a compile error).
- **`#[rite]` modes**: token-param mode (`t: X64V3Token` in signature, no
  attr arg) or tier mode `#[rite(v3)]` (tokenless — used for helpers that
  only shuffle `__m256i`, e.g. `low32_of_i64`, `widen64`, `mulc64`). Both
  emit `#[target_feature]+#[inline]` and cfg-out on wrong arch.
- **`#[target_feature]` fns cannot coerce to `fn(...)` pointers** — hence the
  `Inv1d`/`Fwd1d` enums + `run_inv1d`/`run_fwd1d` match dispatchers (rite;
  rite→rite calls inline within the feature region). Test-side entry is a
  tiny `#[arcane] fn run_v3`.
- **magetypes 0.9.27**: `magetypes::simd::i32x8` is the CONCRETE
  `generic::i32x8<X64V3Token>` alias on x86_64. Escape hatches:
  `.raw() -> __m256i`, `i32x8::from_m256i(token, v)`. Integer `+`/`-`/`*`
  operators are wrapping on every backend; `shl_const::<N>()` wrapping;
  `.clamp(lo, hi)` lane min/max. NO integer widening ops (why hb needs raw).
- **`archmage::testing::for_each_token_permutation`** resets stale disabled
  token state at entry and owns it per permutation — so unit differentials
  run the v3 arm even under `AOM_FORCE_SCALAR` (fire
  `aom_dispatch::scalar_forced()` BEFORE the harness, as the test does, to
  make ordering deterministic).
- The transpiled scalar regeneration IS byte-identical (verified full-file
  diff) with file order: inv = idct4,8,16,32,64,iadst8,16; fwd =
  fdct8,16,32,64,fadst8,16 (see `just gen-txfm1d`).

## The exact-i64 recipe — where each scalar op maps

All lane ops are **full-i32-domain exact** vs the scalar port (no domain
reasoning needed anywhere except iadst4's i64 bounds):

| scalar op | lane impl | exactness key |
|---|---|---|
| `half_btf` | `hb`: 2×`vpmulld` (wrapped products == `wrapping_mul`), widen halves `vpmovsxdq`, `vpaddq` (+rnd; \|sum\| ≤ 2^32+2^31 < 2^63), LOGICAL `vpsrlq`, low-dword gather | `((v >>_arith b) as i32) == low32(v >>_logical b)` for 1 ≤ b ≤ 32 — sign-fill bits land ≥ bit 32, truncated. AVX2 has no vpsraq; this dodges it |
| `round_shift(v as i64, bit)` (positive `round_shift_array` arm; inv shift[1]=-4→bit4, inv shift[0]∈{0,1,2}, fwd shift[1]/[2]) | `rshiftv`: same widen/srl/truncate, single operand | same identity; bit==0 handled by SKIPPING the call (scalar arm is a no-op) |
| negative `round_shift_array` arm (`clamp_i64(v << k)`) — ONLY fwd col input stage, shift[0]=2 | `shl_clamp64v`: widen, `vpsllq`, i64 min/max via `vpcmpgtq`+`vpblendvb`, low-dword gather | k ≤ 4 → \|v<<k\| < 2^36 exact in i64; after clamp the value is i32 so truncation exact |
| `round_shift(v as i64 * mul, bit)` (NewSqrt2/NewInvSqrt2 rect scalings, iidentity4/16, fidentity4/16) | `mul_rshiftv`: `vpmuldq` even + (`vpsrlq 32` then `vpmuldq`) odd → exact signed 32×32→64 products; +rnd, srl, reassemble `blend 0xAA` with `vpsllq 32` of odd | full products == `v as i64 * mul`; \|mul\| ≤ 11586 < 2^14 → no rnd overflow |
| `clamp_value(v, bit)` | `clampv`: identity for bit ≤ 0 or ≥ 32 (scalar i64 bounds cover i32 at 32), else splat+`.clamp` | bounds i32-representable for 1..=31 |
| wrapping add/sub/neg | `+`/`-`, `negv` = 0−v | lane ops wrap; **`-a + b` ≡ `b − a`** (two's complement) — the transpiler emits the latter; asserted no `-a − b` form exists in libaom |
| `highbd_clip_pixel_add` | lane add (wraps like scalar `wrapping_add`), clamp(0, (1<<bd)−1), `as u16` per lane | narrowing exact after clamp |
| iadst4 (ALL-i64 scalar math) | `hand_v3.rs`: `V64` = 2×i64x4 halves; products `mulc64` (c·v mod 2^64 = `vpmuludq(v_lo,c)` + `vpmuludq(v_hi_u,c)<<32`, needs c ≥ 0 — sinpi all positive, debug_assert), `vpaddq/vpsubq`, `rshift64` | scalar i64 ops never wrap for i32 inputs (\|products\| ≤ 2^46, sums ≤ 2^47) and vpaddq wraps mod 2^64 anyway — identical either way |
| iadst4/fadst4 zero-input early-out | NOT branched on lanes — computed through | on zero input every product/sum is 0 and `round_shift(0,bit) == 0`, so compute-through is bit-identical; differential mixes zero + extreme columns to pin it |

## Pass-level load-bearing subtleties (each cost thought — don't re-derive)

- **inv col pass, lr_flip**: output cols c..c+8 read buf cols col_n−1−(c..c+8)
  = the ascending 8-col load at `col_n−c−8`, LANES REVERSED (`revv` =
  vpermd [7..0]). ud_flip is a ROW-index swap on the store loop (tout
  pre-shifted in place, then `tout[row_n−1−r]` per output row r — bijective,
  each entry shifted exactly once).
- **fwd col pass, lr_flip**: store `revv(v)` at `r*col_n + (col_n−cg−8)`.
  ud_flip is on the LOAD row index (src_r).
- **Scalar op ORDER preserved**: inv row pass = rect-mul THEN clamp_buf; fwd
  row pass = round_shift_array THEN rect-mul. Do not "optimize" the order.
- **fwd scalar col pass uses `output` as scratch** (temp_in/temp_out) — the
  vector path must not touch `output` in the col pass (it doesn't; it writes
  `buf` only). Any mixed vector-col/scalar-row combination is safe.
- **Layouts**: inverse input `mod_input` is COLUMN-major (`[c*row_n + r]`),
  `buf` row-major, output pixels row-major-strided. Forward: input i16
  row-major-strided, `buf` row-major, `output` COLUMN-major (`[c*row_n+r]`).
  Hence: inv col pass = all-contiguous; inv row pass = contiguous loads +
  transpose8 stores (per-lane scatter for the W=4 tail `col_n & !7 ..`);
  fwd col pass = all-contiguous (i16 loads via `widen16`); fwd row pass =
  transpose8 loads (per-lane gather for W=4 tail) + contiguous stores.
- **Gating**: col passes need `col_n % 8 == 0` (W=4 sizes → scalar); row
  passes need `row_n % 8 == 0` (H=4 sizes → scalar: 4x4, 8x4, 16x4 whole-
  scalar for rows; 4x4 fully scalar). Kernel point-count == the OTHER dim
  (inv col kernel = H-point, inv row kernel = W-point;
  `debug_assert_eq!(inv_kernel_n(k), row_n / col_n)` in the two inv
  entries — fwd entries have NO such assert yet, add `fwd_kernel_n` for
  symmetry). `remap_input`, fwd 64-pt zero/repack post-processing, and the
  WHT/lossless path remain scalar (memory moves / separate kernel).
- **cos_bit is runtime** (inv fixed 12; fwd 10..13 from COS_BIT tables) —
  shifts use `_mm256_srl_epi64` with an `_mm_cvtsi32_si128(bit)` count, not
  const shifts.
- Buffers `[i32x8; 64]` ×2 per pass (4 KB stack) — fine; idct64 kernel holds
  128 live vectors → LLVM spills, same shape as libaom's own SIMD.

## Known gaps / follow-ups (each marked in code where local)

1. 2-D permutation-equality integration test — see IMMEDIATE #3 (main gap).
2. `fwd_kernel_n` debug_asserts (symmetry with inv) — trivial.
3. H=4 row batches (4-lane half-vectors or 2-rows-per-vector) and W=4 col
   batches — small blocks, do only if post-landing profile says so.
4. AVX-512 (`_v4x`/v4 i32x16, native `vpsraq` kills the logical-shift dance;
   16-col batches) — only after the AVX2 numbers are committed.
5. NEON — deferred by design (falls to scalar; perf gate box is x86).
6. The `.claude/worktrees` branch needs rebase onto origin/main before
   landing (stills-pivot agents land concurrently) — commits are
   pathspec-scoped to transform files + justfile + xtask, so rebase should
   be clean.
7. `conformance/data` shows as untracked (worktree setup symlink) — do NOT
   commit it.

## Differential plan recap (what the written test asserts)

Per kernel × cos_bit {10..13} × stage_range {16,17,18,20}: (a) dense random
at ±2^15/±2^17/±2^19 (driver clamp domains bd8/10/12), (b) exact-bound sign
patterns (all +B, all −B, two alternations, pseudo-random ±B — the
|p0+p1| maximizers for the i64-sum trap), (c) full-range i32 random,
(d) extreme lanes (i32::MIN/MAX, ±2^19) mixed with all-zero columns
(pins the iadst compute-through and lane independence). v3 lanes vs 8
scalar-kernel columns, exact equality, at every token permutation, with a
non-vacuity counter (`v3_ran >= 1`).
