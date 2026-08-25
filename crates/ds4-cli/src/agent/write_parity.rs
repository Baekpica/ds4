use super::web_tools::{handle_round_with_cursor, Browser, ReadCursor};
use super::write::{write_result, WRITE};
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

fn c_write(path: Option<&str>, content: Option<&str>) -> Vec<u8> {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build C oracle: make tests/parity/agent_c_oracle"
    );
    let mut command = std::process::Command::new(oracle);
    command.arg("write");
    if let Some(path) = path {
        command.arg(path);
        if let Some(content) = content {
            command.arg(content);
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

fn write_call(path: Option<&str>, content: Option<&str>) -> Vec<u8> {
    let mut body = String::from("<｜DSML｜invoke name=\"write\">\n");
    if let Some(path) = path {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n"
        ));
    }
    if let Some(content) = content {
        body.push_str(&format!(
            "<｜DSML｜parameter name=\"content\" string=\"true\">{content}</｜DSML｜parameter>\n"
        ));
    }
    body.push_str("</｜DSML｜invoke>\n");
    format!("<｜DSML｜tool_calls>\n{body}</｜DSML｜tool_calls>").into_bytes()
}

fn observation(result: &[u8]) -> Vec<u8> {
    let mut out = format!("Tool result 1 ({WRITE}):\n").into_bytes();
    out.extend_from_slice(result);
    if !result.is_empty() && !result.ends_with(b"\n") {
        out.push(b'\n');
    }
    out
}

fn temp_path(name: &str) -> PathBuf {
    PathBuf::from(format!(
        "/tmp/ds4_agent_write_{}_{}_{name}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

#[test]
fn write_missing_path_and_content_match_c() {
    assert_eq!(write_result(None, Some("x")), c_write(None, None));
    assert_eq!(write_result(Some(""), Some("x")), c_write(Some(""), None));
    assert_eq!(
        write_result(Some("/tmp/x"), None),
        c_write(Some("/tmp/x"), None)
    );
}

#[test]
fn write_creates_overwrites_and_allows_empty_content() {
    let path = temp_path("file.txt");
    let path_text = path.to_str().expect("UTF-8 write path");
    let mut web = NoWeb;
    let mut cursor = ReadCursor::default();

    let created = handle_round_with_cursor(
        &write_call(Some(path_text), Some("hello")),
        &mut web,
        &mut cursor,
    )
    .expect("write create");
    assert_eq!(
        created.observation,
        observation(&c_write(Some(path_text), Some("hello")))
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"hello");

    let overwritten = handle_round_with_cursor(
        &write_call(Some(path_text), Some("z")),
        &mut web,
        &mut cursor,
    )
    .expect("write overwrite");
    assert_eq!(
        overwritten.observation,
        observation(&c_write(Some(path_text), Some("z")))
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"z");

    let empty = handle_round_with_cursor(
        &write_call(Some(path_text), Some("")),
        &mut web,
        &mut cursor,
    )
    .expect("write empty");
    assert_eq!(
        empty.observation,
        observation(&c_write(Some(path_text), Some("")))
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn write_open_error_matches_c_strerror() {
    let missing_parent = temp_path("no-such-dir").join("file.txt");
    let path_text = missing_parent.to_str().expect("UTF-8 write path");
    let mut web = NoWeb;
    let mut cursor = ReadCursor::default();
    let failed = handle_round_with_cursor(
        &write_call(Some(path_text), Some("hello")),
        &mut web,
        &mut cursor,
    )
    .expect("write open error");
    assert_eq!(
        failed.observation,
        observation(&c_write(Some(path_text), Some("hello")))
    );
    assert!(failed
        .observation
        .starts_with(b"Tool result 1 (write):\nTool error: open for write failed: "));
}
