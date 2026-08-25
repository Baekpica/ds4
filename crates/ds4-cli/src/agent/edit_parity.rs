use super::edit::{edit_result, EDIT};
use super::web_tools::{handle_round_with_cursor, Browser, ReadCursor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

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

fn c_edit(path: Option<&str>, old: Option<&str>, new: Option<&str>) -> Vec<u8> {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build C oracle: make tests/parity/agent_c_oracle"
    );
    let mut command = std::process::Command::new(oracle);
    command.arg("edit");
    if let Some(path) = path {
        command.arg(path);
        if let Some(old) = old {
            command.arg(old);
            if let Some(new) = new {
                command.arg(new);
            }
        }
    }
    let output = command.output().expect("run C agent oracle");
    assert!(
        output.status.success(),
        "C agent oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    decode_oracle_hex(&output.stdout)
}

fn edit_call(path: Option<&str>, old: Option<&str>, new: Option<&str>) -> Vec<u8> {
    let mut body = String::from("<｜DSML｜invoke name=\"edit\">\n");
    if let Some(path) = path {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n"
        ));
    }
    if let Some(old) = old {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"old\" string=\"true\">{old}</｜DSML｜parameter>\n"
        ));
    }
    if let Some(new) = new {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"new\" string=\"true\">{new}</｜DSML｜parameter>\n"
        ));
    }
    body.push_str("</｜DSML｜invoke>\n");
    format!("<｜DSML｜tool_calls>\n{body}</｜DSML｜tool_calls>").into_bytes()
}

fn observation(result: &[u8]) -> Vec<u8> {
    let mut out = format!("Tool result 1 ({EDIT}):\n").into_bytes();
    out.extend_from_slice(result);
    if !result.is_empty() && !result.ends_with(b"\n") {
        out.push(b'\n');
    }
    out
}

fn temp_path(name: &str) -> PathBuf {
    PathBuf::from(format!(
        "/tmp/ds4_agent_edit_{}_{}_{name}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn edit_missing_args_match_c() {
    assert_eq!(
        edit_result(None, Some("a"), Some("b")),
        c_edit(None, None, None)
    );
    assert_eq!(
        edit_result(Some(""), Some("a"), Some("b")),
        c_edit(Some(""), None, None)
    );
    assert_eq!(
        edit_result(Some("/tmp/x"), None, Some("b")),
        c_edit(Some("/tmp/x"), None, None)
    );
    assert_eq!(
        edit_result(Some("/tmp/x"), Some(""), Some("b")),
        c_edit(Some("/tmp/x"), Some(""), None)
    );
    assert_eq!(
        edit_result(Some("/tmp/x"), Some("a"), None),
        c_edit(Some("/tmp/x"), Some("a"), None)
    );
}

#[test]
fn edit_unique_replace_and_empty_new_match_c() {
    let path = temp_path("file.txt");
    let path_text = path.to_str().expect("UTF-8 edit path");
    std::fs::write(&path, b"alpha\nkeep\n").unwrap();
    let mut web = NoWeb;
    let mut cursor = ReadCursor::default();

    let expected = {
        std::fs::write(&path, b"alpha\nkeep\n").unwrap();
        observation(&c_edit(Some(path_text), Some("alpha"), Some("beta")))
    };
    std::fs::write(&path, b"alpha\nkeep\n").unwrap();
    let replaced = handle_round_with_cursor(
        &edit_call(Some(path_text), Some("alpha"), Some("beta")),
        &mut web,
        &mut cursor,
    )
    .expect("unique edit");
    assert_eq!(replaced.observation, expected);
    assert_eq!(std::fs::read(&path).unwrap(), b"beta\nkeep\n");

    std::fs::write(&path, b"alpha\nkeep\n").unwrap();
    let deleted = handle_round_with_cursor(
        &edit_call(Some(path_text), Some("alpha\n"), Some("")),
        &mut web,
        &mut cursor,
    )
    .expect("empty new");
    std::fs::write(&path, b"alpha\nkeep\n").unwrap();
    assert_eq!(
        deleted.observation,
        observation(&c_edit(Some(path_text), Some("alpha\n"), Some("")))
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn edit_not_found_and_not_unique_match_c() {
    let path = temp_path("dup.txt");
    let path_text = path.to_str().expect("UTF-8 edit path");
    std::fs::write(&path, b"aaaa").unwrap();
    assert_eq!(
        edit_result(Some(path_text), Some("zzz"), Some("y")),
        c_edit(Some(path_text), Some("zzz"), Some("y"))
    );
    assert_eq!(
        edit_result(Some(path_text), Some("aaa"), Some("y")),
        c_edit(Some(path_text), Some("aaa"), Some("y"))
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn edit_upto_paths_match_c() {
    let path = temp_path("upto.txt");
    let path_text = path.to_str().expect("UTF-8 edit path");
    std::fs::write(&path, b"head\nmiddle\ntail\nhead\n").unwrap();
    assert_eq!(
        edit_result(Some(path_text), Some("head\n[upto]tail\n"), Some("X\n")),
        {
            std::fs::write(&path, b"head\nmiddle\ntail\nhead\n").unwrap();
            c_edit(Some(path_text), Some("head\n[upto]tail\n"), Some("X\n"))
        }
    );
    std::fs::write(&path, b"head\nmiddle\ntail\nhead\n").unwrap();
    assert_eq!(
        edit_result(Some(path_text), Some("a[upto]b[upto]c"), Some("X")),
        c_edit(Some(path_text), Some("a[upto]b[upto]c"), Some("X"))
    );
    assert_eq!(
        edit_result(Some(path_text), Some("head[upto]   "), Some("X")),
        c_edit(Some(path_text), Some("head[upto]   "), Some("X"))
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn edit_open_and_too_large_match_c() {
    let missing = temp_path("missing.txt");
    let missing_text = missing.to_str().expect("UTF-8 missing path");
    assert_eq!(
        edit_result(Some(missing_text), Some("a"), Some("b")),
        c_edit(Some(missing_text), Some("a"), Some("b"))
    );

    let huge = temp_path("huge.bin");
    let huge_text = huge.to_str().expect("UTF-8 huge path");
    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&huge)
        .unwrap()
        .set_len((super::web_tools::FILE_MAX_BYTES + 1) as u64)
        .unwrap();
    assert_eq!(
        edit_result(Some(huge_text), Some("a"), Some("b")),
        c_edit(Some(huge_text), Some("a"), Some("b"))
    );
    std::fs::remove_file(huge).unwrap();
}
