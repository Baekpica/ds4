use super::compact::{overflow_error, should_compact, tail_start};
use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(path) = std::env::var("DS4_AGENT_C_ORACLE") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/agent_c_oracle")
}

fn c_out(args: &[&str]) -> String {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build C oracle: make tests/parity/agent_c_oracle"
    );
    let output = Command::new(oracle)
        .args(args)
        .output()
        .expect("run C agent oracle");
    assert!(
        output.status.success(),
        "C agent oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn decode_oracle_hex(text: &str) -> Vec<u8> {
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn should_compact_matches_c_thresholds() {
    let cases = [
        (0, 0),
        (10000, 0),
        (10000, 8500),
        (10000, 8499),
        (10000, 7500),
        (10000, 7499),
        (1000, 850),
        (1000, 1),
        (100, 75),
        (100, 76),
        (4, 1),
    ];
    for (ctx, used) in cases {
        let rust = i32::from(should_compact(ctx, used));
        let c = c_out(&["should-compact", &ctx.to_string(), &used.to_string()])
            .parse::<i32>()
            .expect("oracle int");
        assert_eq!(rust, c, "should_compact({ctx}, {used})");
    }
}

#[test]
fn tail_start_matches_c_budget_and_user_boundary() {
    let tokens = [1, 2, 9, 3, 4, 9, 5];
    let token_args: Vec<String> = tokens.iter().map(i32::to_string).collect();
    let mut args = vec![
        "tail-start".into(),
        "100".into(),
        "7".into(),
        "1".into(),
        "9".into(),
    ];
    args.extend(token_args);
    let rust = tail_start(100, 7, 1, 9, &tokens);
    let c = c_out(&args.iter().map(String::as_str).collect::<Vec<_>>())
        .parse::<i32>()
        .expect("oracle int");
    assert_eq!(rust, c);
    assert_eq!(tail_start(100, 7, 1, -1, &tokens), 1);
}

#[test]
fn overflow_error_matches_c_bytes() {
    assert_eq!(
        overflow_error(9000, 8192),
        decode_oracle_hex(&c_out(&["overflow-error", "9000", "8192"]))
    );
}
