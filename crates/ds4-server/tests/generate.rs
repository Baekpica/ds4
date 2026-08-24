//! Scripted decode + HTTP generate path. No GGUF.

use ds4_server::{
    generate_and_write, handle_client_inner, generation_blocked, stop_list_find_from, ParseEnv,
    ParsedRequest, ScriptedDecode, ScriptedStep, ServerConfig, ServerInner, ThinkMode,
    CREATED_TEST, TAPE_PLAIN,
};
use ds4_server::parse::parse_request;
use ds4_server::route::WireSurface;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::thread;

fn env() -> ParseEnv {
    ParseEnv {
        default_model: "ds4".into(),
        default_tokens: 16,
        default_effort: ThinkMode::None,
        default_temp: 0.0,
        live_ids: Vec::new(),
    }
}

fn user_req() -> ParsedRequest {
    let mut r = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"thinking":{"type":"disabled"}}"#,
    )
    .unwrap();
    r.think_mode = ThinkMode::None;
    r.temperature = 0.0;
    r
}

#[test]
fn stop_list_find_matches_c_order() {
    let stops = vec!["STOP".into(), "END".into()];
    assert_eq!(
        stop_list_find_from(&stops, b"hello STOP tail END", 0),
        Some((6, 4))
    );
}

#[test]
fn family_generate_allows_tools() {
    let parsed = user_req();
    assert_eq!(generation_blocked(&parsed, 3), None);
    assert_eq!(generation_blocked(&parsed, 2), None);
    let mut tools = parsed.clone();
    tools.has_tools = true;
    assert_eq!(generation_blocked(&tools, 0), None);
}

#[test]
fn scripted_buffered_openai_has_text_and_stop() {
    let parsed = user_req();
    let mut engine = ScriptedDecode::from_pieces(
        &TAPE_PLAIN
            .iter()
            .map(|s| s.as_bytes())
            .collect::<Vec<_>>(),
    );
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-1",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("Hello world."), "{s}");
    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
    assert!(s.contains("\"object\":\"chat.completion\""), "{s}");
    assert!(
        s.contains("\"cache_write_tokens\":1"),
        "cold serial prompt should count as a KV write: {s}"
    );
    assert!(s.contains("\"timings\":{\"ttft_ms\":"), "serial path should emit timings: {s}");
    assert!(s.contains("\"prefill_tokens\":1"), "{s}");
}

#[test]
fn scripted_http_door_generates() {
    let cfg = ServerConfig {
        model_id: "ds4".into(),
        model_name: "ds4".into(),
        have_engine: true,
        default_tokens: 16,
        ..ServerConfig::default()
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let h = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine));
    });
    let mut c = TcpStream::connect(addr).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    c.write_all(req.as_bytes()).unwrap();
    let _ = c.shutdown(std::net::Shutdown::Write);
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    h.join().unwrap();
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("ok"), "{s}");
}

#[test]
fn scripted_motif_generates_over_http() {
    let cfg = ServerConfig {
        have_engine: true,
        ..ServerConfig::default()
    };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let h = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let mut engine = ScriptedDecode {
            model_id: 3,
            ..ScriptedDecode::from_pieces(&[b"ok"])
        };
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine));
    });
    let mut c = TcpStream::connect(addr).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    c.write_all(req.as_bytes()).unwrap();
    let _ = c.shutdown(std::net::Shutdown::Write);
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    h.join().unwrap();
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("ok"), "{s}");
}

#[test]
fn bridge_null_oracle_ok() {
    let p = if let Ok(v) = std::env::var("DS4_BRIDGE_NULL_ORACLE") {
        PathBuf::from(v)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/bridge_null_oracle")
    };
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/bridge_null_oracle (missing {})",
        p.display()
    );
    let out = Command::new(&p).output().expect("run bridge_null_oracle");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"ok\n");
}

#[test]
fn scripted_dsml_tools_emit_tool_calls() {
    let mut parsed = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"},"max_tokens":16,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]}"#,
    )
    .unwrap();
    parsed.think_mode = ThinkMode::None;
    parsed.temperature = 0.0;
    assert!(parsed.has_tools);

    let block = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"bash\">\n",
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let mut engine = ScriptedDecode::from_pieces(&[block.as_bytes()]);
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-tools",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("\"finish_reason\":\"tool_calls\""), "{s}");
    assert!(s.contains("\"tool_calls\":["), "{s}");
    assert!(s.contains("\"name\":\"bash\""), "{s}");
    assert!(s.contains("\"id\":\"chatcmpl-tools_tool_0\""), "{s}");
    assert!(s.contains("ls"), "{s}");
}

fn tools_req() -> ParsedRequest {
    let mut parsed = parse_request(
        WireSurface::OpenaiChat,
        &env(),
        r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"},"max_tokens":16,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]}"#,
    )
    .unwrap();
    parsed.think_mode = ThinkMode::None;
    parsed.temperature = 0.0;
    parsed
}

#[test]
fn scripted_invalid_dsml_retries_and_emits_tool_calls() {
    let parsed = tools_req();
    let invalid = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let valid = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke name=\"bash\">\n",
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
        "</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let mut engine = ScriptedDecode {
        steps: vec![
            ScriptedStep {
                token: 1,
                piece: invalid.as_bytes().to_vec(),
                stop: false,
            },
            ScriptedStep {
                token: 3,
                piece: valid.as_bytes().to_vec(),
                stop: false,
            },
            ScriptedStep {
                token: 4,
                piece: Vec::new(),
                stop: true,
            },
        ],
        suffix_tokens: vec![10, 11],
        ..ScriptedDecode::from_pieces(&[b"x"])
    };
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-retry",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("\"finish_reason\":\"tool_calls\""), "{s}");
    assert!(s.contains("\"name\":\"bash\""), "{s}");
    assert!(s.contains("ls"), "{s}");
    assert!(engine.idx >= 2, "second decode pass should consume the valid call");
}

#[test]
fn scripted_motif_does_not_retry_invalid_tools() {
    let parsed = tools_req();
    let invalid = concat!(
        "<｜DSML｜tool_calls>\n",
        "<｜DSML｜invoke>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜tool_calls>"
    );
    let mut engine = ScriptedDecode {
        model_id: 3,
        steps: vec![
            ScriptedStep {
                token: 1,
                piece: invalid.as_bytes().to_vec(),
                stop: false,
            },
            ScriptedStep {
                token: 2,
                piece: Vec::new(),
                stop: true,
            },
            ScriptedStep {
                token: 3,
                piece: b"should-not-run".to_vec(),
                stop: false,
            },
        ],
        ..ScriptedDecode::from_pieces(&[b"x"])
    };
    let mut out = Vec::new();
    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-motif",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(!s.contains("should-not-run"), "{s}");
    assert!(s.contains("DSML"), "{s}");
    assert_eq!(engine.idx, 1, "Motif must not consume a second decode pass");
}
