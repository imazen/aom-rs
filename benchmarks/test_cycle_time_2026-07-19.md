# Test suite / cycle time — where it actually goes (2026-07-19)

Question: can we speed up the test suites and cycles via binary consolidation or other means?

**Answer: not by binary consolidation, and not by changing the linker — both measured and
rejected. The bottleneck is that `cargo test` runs test *binaries* sequentially while the
cost is concentrated in a handful of e2e tests.**

## Scale

| | count |
|---|---|
| test binaries (autodiscovered, 1 per `tests/*.rs`) | **219** |
| ├ `zenav1-aom-dsp` | 94 |
| ├ `zenav1-aom-encode` | 95 |
| ├ `zenav1-aom-bench` | 17 (+1 bench) |
| ├ `zenav1-aom-decode` | 11 |
| └ `zenav1-aom-sys-ref` | 2 |

Per-binary: ~16 MiB, of which **7.2 MiB is `.text`** — the crate + libaom duplicated into
every one. `target/` totals 19.6 GB.

## REJECTED: binary consolidation (link time is not the bottleneck)

Full `cargo test --no-run -p zenav1-aom-dsp` (94 tests + lib), **fresh target dir**:

| linker | wall | CPU |
|---|---|---|
| default bfd | **17.65 s** | 68.85 s |
| rust-lld (bundled with the toolchain) | **19.27 s** | 67.94 s |

lld is **9% slower** in wall time; CPU is a wash (1%). The whole cold test build of the
94-test crate is ~18 s, so collapsing 219 link steps into ~5 would save seconds at best.
Consolidation is not worth the migration or the loss of `cargo test --test <name>`.

(Feasibility was checked anyway and it *is* safe — 0 tests mutate env vars, call
`process::exit`, hold global mutable state, or `set_current_dir`, and duplicate `fn` names
across files would not collide under `mod`. It's simply not worth doing.)

## THE ACTUAL BOTTLENECK: sequential binaries + concentrated cost

`cargo test` runs each test binary in turn, threading only *within* a binary.

**`zenav1-aom-decode`** — per-binary wall times, summing to 458.5 s (matching total wall,
which confirms serial execution):

```
364.39 s   real_bitstream (15 tests)   <-- 80% of the suite
 39.17 s   ·   37.41 s   ·   12.73 s   ·   3.27 s   ·   1.54 s
  ~0.00 s  x 8 more binaries
```

**`zenav1-aom-dsp`** — same shape: 57.4 s serial across 95 binaries, top 10 = **61%**,
most at ~0.00 s. Slowest: `dv_ref_diff` 10.23 s, `hbd_dist_diff` 6.32 s, `dist_diff` 4.65 s.

### Measured gain from a global work pool

Running the same 95 built `aom-dsp` binaries via a 16-way pool instead of serially:

| | wall |
|---|---|
| sequential (today's `cargo test`) | **57.02 s** |
| global pool, `xargs -P 16` | **12.57 s** — 4.5x |
| floor (slowest single binary) | 10.23 s |

The pool lands near the theoretical floor. Measured with one encoder agent competing for
CPU, so the real gain is at least this.

**This is what `cargo-nextest` does** (one global queue across all binaries, per-test
timing, `--partition` for CI sharding). Not currently installed.

### Caveat: the pool does NOT help `aom-decode`

There 80% of the time is inside ONE binary, so the floor is 364 s and a pool buys ~1.2x.
Worse, that binary parallelises poorly internally: 15 tests, 364 s at default threading vs
**>600 s** at `--test-threads=1` — only ~1.7x on 16 cores. The lever there is *inside*
`real_bitstream`: identify the dominant test and split or shard it.

## Other findings

- **Profile matters more than anything above.** `target/debug/deps` is 7.2 GB at 18 MiB
  median because the default `test` profile inherits full debuginfo. `profile.test-fast`
  (opt-level 3, `debug = "line-tables-only"`) already exists and the e2e byte gates are
  10-20x faster under it — the win is making sure it is the path actually used.
- **The C oracle cache is per-worktree.** `aom-sys-ref` caches libaom keyed on the submodule
  SHA, but the stamp lives in `upstream/build/` — so every new worktree pays a full cmake
  libaom build. Relevant to the multi-agent workflow.
- **192 GB across 65 stale agent worktrees** in `.claude/worktrees/`, 60 with their own
  `target/`. Disk is 50% full so this is not yet causing pressure, but it is pure
  accumulation. Not cleaned up here — some may predate this session and one belonged to a
  live agent.

## Recommendation

1. Adopt `cargo-nextest` for the global pool — 4.5x measured where work spreads across
   binaries. Biggest win for effort.
2. Split or shard `real_bitstream`; it is 80% of the decode suite and threads at only ~1.7x.
3. Keep `--profile test-fast` as the default developer path.
4. Do NOT consolidate test binaries. Do NOT switch to lld. Both measured, both rejected.
