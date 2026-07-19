# aom-rs task runner. `just --list` for a summary.

# Full differential + gate suite (tokens enabled — SIMD dispatch live).
test:
    cargo test --workspace --no-fail-fast

# SCALAR-PIN mode: AOM_FORCE_SCALAR disables every archmage SIMD token
# process-wide (crates/aom-dispatch), so every kernel dispatch falls through
# to the scalar port. The full suite passing here proves SIMD work left the
# scalar path untouched. Run BOTH `just test` AND `just test-scalar` before
# landing any kernel change.
test-scalar:
    AOM_FORCE_SCALAR=1 cargo test --workspace --no-fail-fast

# FAST full-suite run — identical coverage to `just test`, built at opt-level 3
# (the `test-fast` profile) so the e2e byte gates (real full encodes/decodes,
# minutes each unoptimized) run far quicker. debug-assertions + overflow-checks
# stay on, integer results are identical (byte gates stay byte-exact). The
# first run pays a one-time optimized compile; reruns are fast. Use this for
# routine "did I break anything" checks; `just test` remains the debug default.
test-fast:
    cargo test --profile test-fast --workspace --no-fail-fast

# FAST scalar-pin run (AOM_FORCE_SCALAR, opt-level 3). Pair with `test-fast`
# for the both-dispatch-modes parity gate at a fraction of the debug wall time.
test-fast-scalar:
    AOM_FORCE_SCALAR=1 cargo test --profile test-fast --workspace --no-fail-fast

# QUICK SIMD-parity subset (opt-level 3) — the Gate-3 kernel crates' per-kernel
# SIMD==scalar differentials + the transform 2-D permutation-equality gate,
# WITHOUT the minutes-long encoder e2e gates. For tight iteration on SIMD /
# transform work; run `just test-fast` + `just test-fast-scalar` before landing.
# Measured 2026-07-17: ~45s cold (optimized build-dominated), test-RUN a few
# seconds — the transform per-kernel differential is 1.5s here vs ~10s in debug.
test-simd:
    cargo test --profile test-fast -p zenav1-aom-dsp --test txfm2d_simd_perm_diff --test quantize_fp_simd_diff --test cdef_filter_simd_diff --test sad_simd --test hbd_variance_simd_diff --test txb_init_levels_simd_diff --test intra_simd_diff --test lpf_simd_diff --test wiener_simd_diff --test convolve_diff --no-fail-fast

# Gate-3 paired benchmark, port vs C oracle (zenbench interleaved rounds).
# QUIET BOX ONLY — the resource gate flags noisy rounds; a loaded box makes
# the numbers worthless. Results: commit to benchmarks/ per CLAUDE.md.
bench-gate3:
    cargo bench -p zenav1-aom-bench --bench gate3

# Harness smoke: proves the bench runs end-to-end (byte-verify + tiny rounds,
# resource gate off). NUMBERS ARE MEANINGLESS — never quote them.
bench-smoke:
    AOM_BENCH_SMOKE=1 cargo bench -p zenav1-aom-bench --bench gate3

# Callgrind instruction-count profile of one Gate-3 cell (load-tolerant).
# kind=enc|dec side=port|c cell=<label> iters=N; see gate3_profile --help
# for cell labels. Output: /tmp/cg_<cell>_<side>.out (annotate with
# `callgrind_annotate --threshold=95 <file>`).
profile kind side cell iters:
    cargo build --profile profiling -p zenav1-aom-bench --bin gate3_profile
    valgrind --tool=callgrind --callgrind-out-file=/tmp/cg_{{cell}}_{{side}}.out \
        ./target/profiling/gate3_profile {{kind}} {{side}} {{cell}} {{iters}}
    callgrind_annotate --threshold=95 /tmp/cg_{{cell}}_{{side}}.out | head -60

# Regenerate the transform 1-D kernels (scalar + AVX2 lane twins) from the
# extracted C. Scalar output must be byte-identical to the committed files
# (verified: `diff` after regenerating). The lane files are the SIMD twins.
gen-txfm1d:
    python3 xtask/transpile_txfm1d.py --inv reference/extracted/idct4.c reference/extracted/idct8.c reference/extracted/idct16.c reference/extracted/idct32.c reference/extracted/idct64.c reference/extracted/iadst8.c reference/extracted/iadst16.c > crates/aom-dsp/src/transform/inv_txfm1d_gen.rs
    python3 xtask/transpile_txfm1d.py reference/extracted/fdct8.c reference/extracted/fdct16.c reference/extracted/fdct32.c reference/extracted/fdct64.c reference/extracted/fadst8.c reference/extracted/fadst16.c > crates/aom-dsp/src/transform/txfm1d_gen.rs
    python3 xtask/transpile_txfm1d.py --inv --lanes reference/extracted/idct4.c reference/extracted/idct8.c reference/extracted/idct16.c reference/extracted/idct32.c reference/extracted/idct64.c reference/extracted/iadst8.c reference/extracted/iadst16.c > crates/aom-dsp/src/transform/simd/inv1d_v3_gen.rs
    python3 xtask/transpile_txfm1d.py --lanes reference/extracted/fdct8.c reference/extracted/fdct16.c reference/extracted/fdct32.c reference/extracted/fdct64.c reference/extracted/fadst8.c reference/extracted/fadst16.c > crates/aom-dsp/src/transform/simd/txfm1d_v3_gen.rs
