//! C↔Rust DSML decode-state + sampling-override tapes (no model).

use ds4_server::dsml_dump_script;
use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_DSML_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/dsml_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/dsml_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_str(name: &str) -> String {
    let out = Command::new(require_oracle())
        .arg(name)
        .output()
        .expect("run dsml_c_oracle");
    assert!(
        out.status.success(),
        "oracle {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn assert_script(name: &str) {
    let rust = dsml_dump_script(name);
    let c = c_str(name);
    if rust != c {
        panic!("{name} mismatch\n--- rust ---\n{rust}\n--- c ---\n{c}");
    }
}

#[test]
fn dsml_states_match_c() {
    for name in [
        "state-prefix",
        "state-path-param",
        "state-path-closing",
        "state-json-struct",
        "state-json-string",
        "state-done",
    ] {
        assert_script(name);
    }
}

#[test]
fn sampling_override_matches_c() {
    for name in [
        "override-required",
        "override-think-cap",
        "override-tool-result",
        "cap-low-128",
        "cap-high-128",
        "cap-max-600",
    ] {
        assert_script(name);
    }
    let req = dsml_dump_script("override-required");
    assert_eq!(req.trim(), "103 204");
    let think = dsml_dump_script("override-think-cap");
    assert_eq!(think.trim(), "0 305");
    let tr = dsml_dump_script("override-tool-result");
    assert_eq!(tr.trim(), "0 305");
}
