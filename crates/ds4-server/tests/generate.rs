//! Scripted decode + HTTP generate path. No GGUF.

use ds4_server::{
    generate_and_write, handle_client_inner, generation_blocked, render_prompt,
    stop_list_find_from, ContStepper, DecodeIo, GenerateError, ParseEnv, ParsedRequest,
    ReqTimings, ScriptedDecode, ScriptedStep, ServerConfig, ServerInner, ThinkMode, CREATED_TEST,
    TAPE_PLAIN,
};
use ds4_server::parse::parse_request;
use ds4_server::route::WireSurface;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

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

struct PromptSyncDecode {
    inner: ScriptedDecode,
    cached_tokens: i32,
    effective_prompt_pos: i32,
    prompt_sync_calls: usize,
    sync_calls: usize,
    disk_eligible: Vec<bool>,
    thinking_visible_eligible: Vec<bool>,
    prompt_sync_elapsed: Option<Duration>,
    remembered: Vec<(Vec<u8>, i32)>,
    invalidations: usize,
}

impl PromptSyncDecode {
    fn new(inner: ScriptedDecode, cached_tokens: i32, effective_prompt_pos: i32) -> Self {
        Self {
            inner,
            cached_tokens,
            effective_prompt_pos,
            prompt_sync_calls: 0,
            sync_calls: 0,
            disk_eligible: Vec::new(),
            thinking_visible_eligible: Vec::new(),
            prompt_sync_elapsed: None,
            remembered: Vec::new(),
            invalidations: 0,
        }
    }
}

impl DecodeIo for PromptSyncDecode {
    fn model_id(&self) -> i32 {
        self.inner.model_id()
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        self.inner.tokenize_text(text)
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        self.inner.tokenize_rendered_chat(text)
    }

    fn tokenizes_control_literals(&self) -> bool {
        self.inner.tokenizes_control_literals()
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        self.inner.token_text(token)
    }

    fn token_is_stop(&self, token: i32) -> bool {
        self.inner.token_is_stop(token)
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        self.sync_calls += 1;
        self.inner.sync(tokens)
    }

    fn sync_prompt(
        &mut self,
        _prompt: &[u8],
        tokens: &[i32],
        disk_eligible: bool,
        thinking_visible_eligible: bool,
    ) -> Result<i32, GenerateError> {
        self.prompt_sync_calls += 1;
        self.disk_eligible.push(disk_eligible);
        self.thinking_visible_eligible
            .push(thinking_visible_eligible);
        self.inner.live = tokens.to_vec();
        self.inner.pos = self.effective_prompt_pos;
        Ok(self.cached_tokens)
    }

    fn prompt_sync_elapsed(&self) -> Option<Duration> {
        self.prompt_sync_elapsed
    }

    fn eval(&mut self, token: i32) -> Result<(), GenerateError> {
        self.inner.eval(token)
    }

    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        self.inner
            .sample(temperature, top_k, top_p, min_p, rng)
    }

    fn pos(&self) -> i32 {
        self.inner.pos()
    }

    fn ctx(&self) -> i32 {
        self.inner.ctx()
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    fn session_tokens(&self) -> Vec<i32> {
        self.inner.session_tokens()
    }

    fn remember_thinking_visible_checkpoint(&mut self, text: Vec<u8>) {
        self.remembered.push((text, self.pos()));
    }

    fn invalidate(&mut self) {
        self.invalidations += 1;
        self.inner.live.clear();
        self.inner.pos = 0;
    }
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
fn prompt_sync_reports_buffered_cache_usage_from_effective_pos() {
    let parsed = user_req();
    let inner = ScriptedDecode::from_pieces(
        &TAPE_PLAIN
            .iter()
            .map(|s| s.as_bytes())
            .collect::<Vec<_>>(),
    );
    let mut engine = PromptSyncDecode::new(inner, 4, 6);
    engine.prompt_sync_elapsed = Some(Duration::from_secs(2));
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cache-buffered",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("\"prompt_tokens\":6"), "{s}");
    assert!(
        s.contains("\"cached_tokens\":4,\"cache_write_tokens\":2"),
        "cache writes must use effective engine pos minus cache reads: {s}"
    );
    assert!(s.contains("\"prefill_tokens\":2"), "{s}");
    assert!(s.contains("\"prefill_cached_tokens\":4"), "{s}");
    assert!(s.contains("\"prefill_tok_s\":1.0"), "{s}");
    assert_eq!(engine.prompt_sync_calls, 1);
    assert_eq!(engine.sync_calls, 0);
    assert_eq!(engine.disk_eligible, [true]);
    assert_eq!(engine.thinking_visible_eligible, [true]);
}

#[test]
fn prompt_sync_reports_streaming_cache_usage_from_effective_pos() {
    let mut parsed = user_req();
    parsed.stream = true;
    parsed.stream_include_usage = true;
    let inner = ScriptedDecode::from_pieces(
        &TAPE_PLAIN
            .iter()
            .map(|s| s.as_bytes())
            .collect::<Vec<_>>(),
    );
    let mut engine = PromptSyncDecode::new(inner, 4, 6);
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cache-stream",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("\"prompt_tokens\":6"), "{s}");
    assert!(
        s.contains("\"cached_tokens\":4,\"cache_write_tokens\":2"),
        "stream usage must use effective engine pos minus cache reads: {s}"
    );
    assert_eq!(engine.prompt_sync_calls, 1);
    assert_eq!(engine.sync_calls, 0);
    assert_eq!(engine.disk_eligible, [true]);
    assert_eq!(engine.thinking_visible_eligible, [true]);
}

#[test]
fn prompt_sync_receives_thinking_visible_surface_gate() {
    let cases = [
        (
            WireSurface::OpenaiChat,
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            true,
        ),
        (
            WireSurface::OpenaiCompletion,
            r#"{"prompt":"hi"}"#,
            false,
        ),
        (
            WireSurface::Anthropic,
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#,
            true,
        ),
        (WireSurface::Responses, r#"{"input":"hi"}"#, false),
    ];

    for (surface, body, expected) in cases {
        let parsed = parse_request(surface, &env(), body).unwrap();
        let inner = ScriptedDecode::from_pieces(&[b"ok"]);
        let mut engine = PromptSyncDecode::new(inner, 0, 1);
        let mut out = Vec::new();

        generate_and_write(
            &mut engine,
            &parsed,
            "visible-surface",
            CREATED_TEST,
            false,
            16,
            &mut out,
        )
        .unwrap();

        assert_eq!(engine.thinking_visible_eligible, [expected], "{surface:?}");
    }
}

#[test]
fn motif3_no_think_remembers_canonical_visible_checkpoint() {
    let parsed = user_req();
    let inner = ScriptedDecode {
        model_id: 3,
        ..ScriptedDecode::from_pieces(&[b"  Clear skies.  "])
    };
    let mut engine = PromptSyncDecode::new(inner, 0, 1);
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-visible",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    let mut expected = render_prompt(&parsed, 3).unwrap();
    assert!(expected.ends_with(b"<think></think>"));
    expected.truncate(expected.len() - b"<think></think>".len());
    expected.extend_from_slice(b"Clear skies.");
    assert_eq!(engine.remembered, [(expected, engine.pos())]);
    assert!(!engine.remembered[0].0.ends_with(b"<|endofturn|>"));

    let mut length = parsed;
    length.max_tokens = 1;
    length.max_tokens_set = true;
    let inner = ScriptedDecode {
        model_id: 3,
        ..ScriptedDecode::from_pieces(&[b"partial"])
    };
    let mut engine = PromptSyncDecode::new(inner, 0, 1);
    let mut out = Vec::new();
    let outcome = generate_and_write(
        &mut engine,
        &length,
        "chatcmpl-visible-length",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();
    assert_eq!(outcome.finish, "length");
    assert!(engine.remembered.is_empty());
}

#[test]
fn motif3_no_think_invalidates_user_stop_and_tool_syntax_cut() {
    let cases = [
        (vec!["STOP".into()], b"Clear STOP tail".as_slice()),
        (Vec::new(), b"Clear <tool_call>".as_slice()),
    ];

    for (stops, piece) in cases {
        let mut parsed = user_req();
        parsed.stops = stops;
        let inner = ScriptedDecode {
            model_id: 3,
            ..ScriptedDecode::from_pieces(&[piece])
        };
        let mut engine = PromptSyncDecode::new(inner, 0, 1);
        let mut out = Vec::new();

        let outcome = generate_and_write(
            &mut engine,
            &parsed,
            "chatcmpl-visible-cut",
            CREATED_TEST,
            false,
            16,
            &mut out,
        )
        .unwrap();

        assert_eq!(outcome.finish, "stop");
        assert_eq!(engine.invalidations, 1);
        assert_eq!(engine.pos(), 0);
        assert!(engine.remembered.iter().all(|(_, frontier)| *frontier == 0));
    }
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
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), None);
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
        handle_client_inner(&cfg, &inner, &mut s, Some(&mut engine), None);
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
fn cont_stepper_buffered_matches_serial_shape() {
    let mut parsed = user_req();
    parsed.stream = false;
    let (mut st, head) = ContStepper::new(
        &parsed,
        0,
        "chatcmpl-1",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );
    assert!(head.is_empty(), "buffered request must not stream a head");
    for p in TAPE_PLAIN {
        let step = st.feed(p.as_bytes());
        assert!(step.bytes.is_empty());
        assert!(!step.done);
    }
    let (bytes, outcome) = st.finalize(true, 0, 1, ReqTimings::default(), false);
    let s = String::from_utf8_lossy(&bytes);
    assert!(s.starts_with("HTTP/1.1 200 OK"), "{s}");
    assert!(s.contains("Hello world."), "{s}");
    assert!(s.contains("\"finish_reason\":\"stop\""), "{s}");
    assert!(
        s.contains("\"cached_tokens\":0,\"cache_write_tokens\":1"),
        "engine split maps into the client frame: {s}"
    );
    assert_eq!(outcome.finish, "stop");
}

#[test]
fn cont_stepper_streams_and_stops_on_budget() {
    let mut parsed = user_req();
    parsed.stream = true;
    parsed.max_tokens = 2;
    parsed.max_tokens_set = true;
    let (mut st, head) = ContStepper::new(
        &parsed,
        0,
        "chatcmpl-2",
        CREATED_TEST,
        false,
        16,
        b"prompt".to_vec(),
        1,
        8192,
    );
    let h = String::from_utf8_lossy(&head);
    assert!(h.contains("text/event-stream"), "{h}");
    assert!(h.contains("chat.completion.chunk"), "{h}");
    let first = st.feed(TAPE_PLAIN[0].as_bytes());
    assert!(!first.done);
    let second = st.feed(TAPE_PLAIN[1].as_bytes());
    assert!(second.done, "host budget must stop the sequence");
    let (bytes, outcome) = st.finalize(false, 0, 1, ReqTimings::default(), false);
    let s = String::from_utf8_lossy(&bytes);
    assert!(s.contains("\"finish_reason\":\"length\""), "{s}");
    assert!(s.contains("data: [DONE]"), "{s}");
    assert_eq!(outcome.finish, "length");
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

fn retrying_tool_decode() -> ScriptedDecode {
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
    ScriptedDecode {
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
    }
}

#[test]
fn scripted_invalid_dsml_retries_and_emits_tool_calls() {
    let parsed = tools_req();
    let mut engine = retrying_tool_decode();
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
fn recovery_suffix_uses_sync_not_prompt_sync() {
    let parsed = tools_req();
    let mut engine = PromptSyncDecode::new(retrying_tool_decode(), 0, 1);
    let mut out = Vec::new();

    generate_and_write(
        &mut engine,
        &parsed,
        "chatcmpl-cache-retry",
        CREATED_TEST,
        false,
        16,
        &mut out,
    )
    .unwrap();

    assert_eq!(engine.prompt_sync_calls, 1, "only the initial prompt uses the hook");
    assert_eq!(engine.sync_calls, 1, "the recovery suffix uses ordinary sync");
    assert_eq!(engine.disk_eligible, [false]);
    assert!(engine.inner.idx >= 2, "the retry must run a second decode pass");
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
