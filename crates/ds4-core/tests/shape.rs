//! C↔Rust catalog / DeepSeek select / architecture dump.

use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_SHAPE_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/shape_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/shape_c_oracle (missing {})",
        p.display()
    );
    p
}

#[test]
fn catalog_select_arch_match_c_oracle() {
    let out = Command::new(require_oracle())
        .output()
        .expect("run shape_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let c = String::from_utf8(out.stdout).expect("oracle utf8");
    let rust = ds4_core::dump_oracle();
    assert_eq!(rust, c);
}
