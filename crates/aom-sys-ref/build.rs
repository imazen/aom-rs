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
    let shim_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shim");
    let lib = out_dir.join("libaom_shim.a");
    let mut objs = Vec::new();
    for name in ["entropy_shim", "intra_shim", "sadvar_shim", "convolve_shim", "cdef_shim", "highbd_intra_shim", "hbd_lpf_shim", "hbd_sadvar_shim", "txb_shim", "intra_edge_shim"] {
        let shim_c = shim_dir.join(format!("{name}.c"));
        let obj = out_dir.join(format!("{name}.o"));
        let status = std::process::Command::new("clang")
            .args(["-O2", "-c"])
            .arg(&shim_c)
            .arg("-o")
            .arg(&obj)
            .arg(format!("-I{}", src_dir.display()))
            .arg(format!("-I{}", src_dir.join("build").display()))
            .status()
            .expect("clang failed to run");
        assert!(status.success(), "{name} compile failed");
        println!("cargo:rerun-if-changed={}", shim_c.display());
        objs.push(obj);
    }
    let mut ar = std::process::Command::new("ar");
    ar.arg("crus").arg(&lib);
    for o in &objs {
        ar.arg(o);
    }
    assert!(ar.status().expect("ar failed").success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=aom_shim");

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=aom");
    // libaom is C, but the archive is linked by CXX; pull in libstdc++ + libm
    // in case any TU needs them.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rerun-if-changed={}", lib.display());
}
