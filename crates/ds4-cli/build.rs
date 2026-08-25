fn main() {
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../..");
    let linenoise = root.join("linenoise.c");
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out_dir.join("linenoise.o");
    let lib = out_dir.join("liblinenoise.a");
    let status = std::process::Command::new("cc")
        .args(["-c", "-fPIC", "-fno-stack-protector", "-o"])
        .arg(&obj)
        .arg(&linenoise)
        .arg("-I")
        .arg(&root)
        .status()
        .expect("compile linenoise.c");
    if !status.success() {
        panic!("cc failed to compile linenoise.c");
    }
    let status = std::process::Command::new("ar")
        .args(["rcs"])
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("archive liblinenoise.a");
    if !status.success() {
        panic!("ar failed to archive liblinenoise.a");
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=linenoise");
    println!("cargo:rerun-if-changed={}", linenoise.display());
}
