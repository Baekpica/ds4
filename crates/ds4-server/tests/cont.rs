//! C↔Rust Inc 5a/5b/5c continuation registry (tape oracle, no model).

use ds4_server::{
    dump_script, handle_client_inner, unix_now, Api, ScriptedDecode, ServerConfig, ServerInner,
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::thread;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_CONT_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/cont_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/cont_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_str(script: &str) -> String {
    let out = Command::new(require_oracle())
        .arg(script)
        .output()
        .expect("run cont_c_oracle");
    assert!(
        out.status.success(),
        "oracle {script} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn assert_script(name: &str) {
    let rust = dump_script(name);
    let c = c_str(name);
    if rust != c {
        panic!("{name} mismatch\n--- rust ---\n{rust}\n--- c ---\n{c}");
    }
}

#[test]
fn publish_resolve_demote_matches_c() {
    assert_script("publish-resolve-demote");
}

#[test]
fn supersede_cap_matches_c() {
    assert_script("supersede-cap");
}

#[test]
fn grace_hold_matches_c() {
    assert_script("grace-hold");
}

#[test]
fn ttl_matches_c() {
    assert_script("ttl");
}

#[test]
fn bank_claim_matches_c() {
    assert_script("bank-claim");
}

#[test]
fn bank_protection_retry_matches_c() {
    assert_script("bank-protection");
}

fn http_post(addr: std::net::SocketAddr, path: &str, body: &str) -> String {
    let mut c = TcpStream::connect(addr).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    c.write_all(req.as_bytes()).unwrap();
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    String::from_utf8_lossy(&out).into_owned()
}

#[test]
fn serial_hold_sheds_unrelated_with_retry_after() {
    let mut cfg = ServerConfig::test_cfg();
    cfg.default_tokens = 16;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let now = unix_now() as f64;
    let h = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        {
            let mut g = inner.lock().unwrap();
            g.creg
                .publish_serial(Api::Anthropic, &["toolu_hold".into()], 4, 70, now);
        }
        let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), None);
    });
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
    let s = http_post(addr, "/v1/chat/completions", body);
    h.join().unwrap();
    assert!(s.starts_with("HTTP/1.1 503 "), "{s}");
    assert!(s.to_ascii_lowercase().contains("retry-after:"), "{s}");
    assert!(s.contains("live tool continuation"), "{s}");
}

#[test]
fn live_only_tool_result_conflicts_when_reference_moved() {
    let mut cfg = ServerConfig::test_cfg();
    cfg.default_tokens = 16;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let now = unix_now() as f64;
    let h = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        {
            let mut g = inner.lock().unwrap();
            g.creg
                .publish_serial(Api::Anthropic, &["toolu_x".into()], 7, 100, now);
        }
        let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), None);
    });
    let body = r#"{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_x","content":"ok"}]}],"max_tokens":8}"#;
    let s = http_post(addr, "/v1/messages", body);
    h.join().unwrap();
    assert!(s.starts_with("HTTP/1.1 409 "), "{s}");
    assert!(
        s.contains("Anthropic continuation state is not available"),
        "{s}"
    );
}

#[test]
fn anthropic_tool_turn_publishes_and_holds_next_seat() {
    let mut cfg = ServerConfig::test_cfg();
    cfg.default_tokens = 16;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let h = thread::spawn(move || {
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let mut engine = ScriptedDecode::from_pieces(&[concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"bash\">\n",
            "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
            "</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>"
        )
        .as_bytes()]);
        for _ in 0..3 {
            let (mut s, _) = listener.accept().unwrap();
            handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), None);
        }
        let g = inner.lock().unwrap();
        assert_eq!(g.creg.n_live(), 0);
    });
    let tools = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"thinking":{"type":"disabled"},"tools":[{"name":"bash","input_schema":{"type":"object"}}]}"#;
    let first = http_post(addr, "/v1/messages", tools);
    assert!(first.starts_with("HTTP/1.1 200 "), "{first}");
    assert!(first.contains("\"tool_use\""), "{first}");
    let tool_id = first
        .split("\"type\":\"tool_use\",\"id\":\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    assert_eq!(tool_id.len(), 38);
    assert!(tool_id.starts_with("toolu_"));
    let second = http_post(
        addr,
        "/v1/chat/completions",
        r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#,
    );
    assert!(second.starts_with("HTTP/1.1 503 "), "{second}");
    assert!(second.contains("live tool continuation"), "{second}");
    let resumed = http_post(
        addr,
        "/v1/messages",
        &format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"{tool_id}","content":"ok"}}]}}],"max_tokens":8}}"#
        ),
    );
    h.join().unwrap();
    assert!(resumed.starts_with("HTTP/1.1 200 "), "{resumed}");
}
