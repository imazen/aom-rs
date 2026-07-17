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

# Gate-3 paired benchmark, port vs C oracle (zenbench interleaved rounds).
# QUIET BOX ONLY — the resource gate flags noisy rounds; a loaded box makes
# the numbers worthless. Results: commit to benchmarks/ per CLAUDE.md.
bench-gate3:
    cargo bench -p aom-bench --bench gate3

# Harness smoke: proves the bench runs end-to-end (byte-verify + tiny rounds,
# resource gate off). NUMBERS ARE MEANINGLESS — never quote them.
bench-smoke:
    AOM_BENCH_SMOKE=1 cargo bench -p aom-bench --bench gate3

# Callgrind instruction-count profile of one Gate-3 cell (load-tolerant).
# kind=enc|dec side=port|c cell=<label> iters=N; see gate3_profile --help
# for cell labels. Output: /tmp/cg_<cell>_<side>.out (annotate with
# `callgrind_annotate --threshold=95 <file>`).
profile kind side cell iters:
    cargo build --profile profiling -p aom-bench --bin gate3_profile
    valgrind --tool=callgrind --callgrind-out-file=/tmp/cg_{{cell}}_{{side}}.out \
        ./target/profiling/gate3_profile {{kind}} {{side}} {{cell}} {{iters}}
    callgrind_annotate --threshold=95 /tmp/cg_{{cell}}_{{side}}.out | head -60

# Regenerate the transform 1-D kernels (scalar + AVX2 lane twins) from the
# extracted C. Scalar output must be byte-identical to the committed files
# (verified: `diff` after regenerating). The lane files are the SIMD twins.
gen-txfm1d:
    python3 xtask/transpile_txfm1d.py --inv reference/extracted/idct4.c reference/extracted/idct8.c reference/extracted/idct16.c reference/extracted/idct32.c reference/extracted/idct64.c reference/extracted/iadst8.c reference/extracted/iadst16.c > crates/aom-transform/src/inv_txfm1d_gen.rs
    python3 xtask/transpile_txfm1d.py reference/extracted/fdct8.c reference/extracted/fdct16.c reference/extracted/fdct32.c reference/extracted/fdct64.c reference/extracted/fadst8.c reference/extracted/fadst16.c > crates/aom-transform/src/txfm1d_gen.rs
    python3 xtask/transpile_txfm1d.py --inv --lanes reference/extracted/idct4.c reference/extracted/idct8.c reference/extracted/idct16.c reference/extracted/idct32.c reference/extracted/idct64.c reference/extracted/iadst8.c reference/extracted/iadst16.c > crates/aom-transform/src/simd/inv1d_v3_gen.rs
    python3 xtask/transpile_txfm1d.py --lanes reference/extracted/fdct8.c reference/extracted/fdct16.c reference/extracted/fdct32.c reference/extracted/fdct64.c reference/extracted/fadst8.c reference/extracted/fadst16.c > crates/aom-transform/src/simd/txfm1d_v3_gen.rs
