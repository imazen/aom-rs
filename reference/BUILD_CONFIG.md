# Reference oracle build config

- Source: libaom, tag **v3.14.1**, git `03087864cf4bea6abb0d28f95cf7843511413d8f`
  — the pinned **`upstream/`** git submodule (canonical). The gitignored
  `reference/libaom` clone remains as a fallback.
- Toolchain: gcc 15.2.0 / clang 21.1.8 / nasm 3.01, cmake 4.2.3
- CMake:
  ```
  -DCMAKE_BUILD_TYPE=Release
  -DCONFIG_MULTITHREAD=0     # single-thread → deterministic encoder output target
  -DENABLE_TESTS=1 -DENABLE_EXAMPLES=1 -DENABLE_TOOLS=1
  -DCONFIG_AV1_DECODER=1 -DCONFIG_AV1_ENCODER=1
  ```
- Artifacts: `upstream/build/{libaom.a, aomenc, aomdec}`, built automatically by
  `crates/aom-sys-ref/build.rs` (cached by the submodule SHA).
- `CONFIG_COEFFICIENT_RANGE_CHECKING = 0`, `DO_RANGE_CHECK_CLAMP` off (default),
  so transform range-check functions are no-ops. This is the definition against
  which aom-rs bit-exactness is measured.

Build: cargo-driven — `cargo test` (or `cargo build -p aom-sys-ref`) builds the
oracle from `upstream/` into `upstream/build/` automatically, once, cached by the
submodule SHA. If `upstream/` is empty, build.rs auto-runs
`git submodule update --init upstream`. `bash reference/build.sh` remains a
fallback that clones + builds `reference/libaom`.
