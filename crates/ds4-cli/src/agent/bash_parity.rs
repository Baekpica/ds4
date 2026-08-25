use super::bash::{bash_result, parse_timeout, BashArgs, BashTable, BASH, BASH_STATUS};
use super::web_tools::{handle_round_with_tools, Browser, ReadCursor};
use std::path::PathBuf;

struct NoWeb;

impl Browser for NoWeb {
    fn google_search(&mut self, _query: &str) -> Result<String, String> {
        Err("unused".into())
    }

    fn visit_page(&mut self, _url: &str) -> Result<String, String> {
        Err("unused".into())
    }
}

fn oracle() -> PathBuf {
    if let Ok(path) = std::env::var("DS4_AGENT_C_ORACLE") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/agent_c_oracle")
}

fn decode_oracle_hex(stdout: &[u8]) -> Vec<u8> {
    std::str::from_utf8(stdout)
        .expect("oracle hex UTF-8")
        .trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn c_bash(args: &[&str]) -> Vec<u8> {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build C oracle: make tests/parity/agent_c_oracle"
    );
    let output = std::process::Command::new(oracle)
        .args(args)
        .output()
        .expect("run C agent oracle");
    assert!(
        output.status.success(),
        "C agent oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    decode_oracle_hex(&output.stdout)
}

fn bash_call(name: &str, args: &[(&str, &str)]) -> Vec<u8> {
    let mut body = format!("<｜DSML｜invoke name=\"{name}\">\n");
    for (key, value) in args {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"{key}\" string=\"true\">{value}</｜DSML｜parameter>\n"
        ));
    }
    body.push_str("</｜DSML｜invoke>\n");
    format!("<｜DSML｜tool_calls>\n{body}</｜DSML｜tool_calls>").into_bytes()
}

fn mask_live(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("bash job=") {
            let _ = rest;
            out.push_str("bash job=ID pid=PID status=");
            if line.contains("status=running") {
                out.push_str("running elapsed_sec=ELAPSED timeout_sec=");
                if let Some(timeout) = line.split("timeout_sec=").nth(1) {
                    out.push_str(timeout);
                }
            } else {
                out.push_str("done elapsed_sec=ELAPSED timed_out=");
                if let Some(tail) = line.split("timed_out=").nth(1) {
                    out.push_str(tail);
                }
            }
        } else if let Some(rest) = line.strip_prefix("output_path=") {
            if let Some((_, stats)) = rest.split_once(" (") {
                out.push_str("output_path=PATH (");
                out.push_str(stats);
            } else {
                out.push_str(line);
            }
        } else if line.starts_with("<head -") || line.starts_with("<tail -") {
            out.push_str(if line.starts_with("<head -") {
                "<head -N PATH>"
            } else {
                "<tail -N PATH>"
            });
        } else if line.contains("Use bash_status job=") {
            out.push_str(
                "Use bash_status job=ID to get info before refresh time; use bash_stop job=ID to stop execution",
            );
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

#[test]
fn bash_missing_command_and_missing_job_match_c() {
    let mut table = BashTable::default();
    assert_eq!(
        bash_result(
            &mut table,
            BASH,
            &BashArgs {
                command: None,
                timeout_sec: None,
                refresh_sec: None,
                job: None,
                pid: None,
            }
        ),
        c_bash(&["bash"])
    );
    assert_eq!(
        bash_result(
            &mut table,
            BASH_STATUS,
            &BashArgs {
                command: None,
                timeout_sec: None,
                refresh_sec: None,
                job: None,
                pid: None,
            }
        ),
        c_bash(&["bash-status"])
    );
}

#[test]
fn bash_timeout_parsing_matches_c() {
    assert_eq!(parse_timeout(None), 3600);
    assert_eq!(parse_timeout(Some("")), 3600);
    assert_eq!(parse_timeout(Some("30x")), 30);
    assert_eq!(parse_timeout(Some("0")), 3600);
    assert_eq!(parse_timeout(Some("-1")), 3600);
    assert_eq!(parse_timeout(Some("bogus")), 3600);
    assert_eq!(parse_timeout(Some("0.5")), 1);
    assert_eq!(parse_timeout(Some("999999")), 86400);
}

#[test]
fn bash_echo_done_observation_matches_c_shape() {
    let mut web = NoWeb;
    let mut cursor = ReadCursor::default();
    let mut jobs = BashTable::default();
    let rust = handle_round_with_tools(
        &bash_call(BASH, &[("command", "printf 'hi\\n'")]),
        &mut web,
        &mut cursor,
        &mut jobs,
    )
    .expect("bash echo");
    let rust_body = rust
        .observation
        .strip_prefix(b"Tool result 1 (bash):\n")
        .expect("tool result header");
    let c = c_bash(&["bash", "printf 'hi\\n'"]);
    assert_eq!(mask_live(rust_body), mask_live(&c));
    let text = std::str::from_utf8(&rust.observation).unwrap();
    assert!(text.contains("exit_status=0"));
    assert!(text.contains("<output>\nhi\n</output>"));
}
