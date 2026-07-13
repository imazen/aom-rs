use std::path::PathBuf;

fn main() {
    // Path to the from-source reference build (see reference/BUILD_CONFIG.md).
    let build_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/libaom/build")
        .canonicalize()
        .expect("reference libaom build not found — build it via reference/build.sh");

    let lib = build_dir.join("libaom.a");
    assert!(lib.exists(), "missing {}", lib.display());

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=aom");
    // libaom is C, but the archive is linked by CXX; pull in libstdc++ + libm
    // in case any TU needs them.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rerun-if-changed={}", lib.display());
}
