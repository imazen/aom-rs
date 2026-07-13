use std::path::PathBuf;

fn main() {
    // Path to the from-source reference build (see reference/BUILD_CONFIG.md).
    let build_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/libaom/build")
        .canonicalize()
        .expect("reference libaom build not found — build it via reference/build.sh");

    let lib = build_dir.join("libaom.a");
    assert!(lib.exists(), "missing {}", lib.display());

    // Compile the entropy-coder shim against the libaom source + generated config.
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/libaom")
        .canonicalize()
        .unwrap();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shim_c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shim/entropy_shim.c");
    let obj = out_dir.join("entropy_shim.o");
    let lib = out_dir.join("libentropy_shim.a");
    let status = std::process::Command::new("clang")
        .args(["-O2", "-c"])
        .arg(&shim_c)
        .arg("-o")
        .arg(&obj)
        .arg(format!("-I{}", src_dir.display()))
        .arg(format!("-I{}", src_dir.join("build").display()))
        .status()
        .expect("clang failed to run");
    assert!(status.success(), "shim compile failed");
    let ar = std::process::Command::new("ar")
        .arg("crus")
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("ar failed");
    assert!(ar.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=entropy_shim");
    println!("cargo:rerun-if-changed={}", shim_c.display());

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=aom");
    // libaom is C, but the archive is linked by CXX; pull in libstdc++ + libm
    // in case any TU needs them.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rerun-if-changed={}", lib.display());
}
