//! Blocking accept loop: HTTP door + parsers + admission + projectors +
//! serial decode (including finalize tool_calls) when a `DecodeIo` is supplied.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::admit::{
    enqueue, enqueue_release, enqueue_shed_error, next_job_id, preparse_shed, AdmitState, EnqVerdict,
    SHED_CONT_HOLD,
};
use crate::cont::ContRegistry;
use crate::error::{http_response_bytes, wire_http_error_bytes};
use crate::generate::{generate_and_write, DecodeIo, GenerateError};
use crate::http::{chunked_enabled, parse_surface_for_path, read_http_request, shed_surface_for_path};
use crate::metrics::{
    gov_modes_from_env, render_metrics, render_stats_json_ex, RouteMetrics, RuntimeMetrics,
};
#[cfg(feature = "native")]
use crate::metrics::MemCell;
use crate::models::{model_id_known, model_one_json, models_list_json};
use crate::parse::{parse_request, ParseEnv};
use crate::route::{route_decide, Api, RouteEnv, ThinkMode, WireSurface, LANE_CONTINUOUS};
use crate::serve_cont::{cont_prompt_tokens, ContExec};
use crate::stream::unix_now;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub listen_host: String,
    pub listen_port: u16,
    pub model_id: String,
    pub model_name: String,
    pub ctx: i32,
    pub default_tokens: i32,
    pub cors: bool,
    pub codex_models_json: Option<String>,
    pub max_queue: i32,
    pub max_queue_bytes: u64,
    pub max_clients: i32,
    pub have_engine: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen_host: "127.0.0.1".into(),
            listen_port: 8000,
            model_id: "ds4".into(),
            model_name: "ds4".into(),
            ctx: 8192,
            default_tokens: 393216,
            cors: false,
            codex_models_json: None,
            max_queue: 0,
            max_queue_bytes: 0,
            max_clients: 0,
            have_engine: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct ServerInner {
    pub admit: AdmitState,
    pub metrics: RouteMetrics,
    pub runtime: RuntimeMetrics,
    pub creg: ContRegistry,
    pub boot_stamp: u64,
    pub have_engine: bool,
}

impl ServerInner {
    pub fn from_cfg(cfg: &ServerConfig) -> Self {
        let mut s = Self::default();
        s.admit.max_queue = cfg.max_queue;
        s.admit.max_queue_bytes = cfg.max_queue_bytes;
        s.admit.max_clients = cfg.max_clients;
        s.runtime.memgov.gov_modes = gov_modes_from_env();
        s.boot_stamp = unix_now() as u64;
        s.have_engine = cfg.have_engine;
        s
    }

    pub fn render_runtime(&self, now: u64) -> RuntimeMetrics {
        let mut rt = self.runtime.clone();
        rt.uptime_seconds = if self.boot_stamp != 0 && now >= self.boot_stamp {
            now - self.boot_stamp
        } else {
            0
        };
        rt.creg_records_live = self.creg.n_live() as u64;
        #[cfg(feature = "native")]
        if self.have_engine {
            overlay_live_census(&mut rt.memgov);
        }
        rt
    }
}

#[cfg(feature = "native")]
fn overlay_live_census(g: &mut crate::metrics::MemgovSnap) {
    let snap = ds4_core::snapshot_mem();
    g.census_supported = snap.census.supported;
    g.census_faults = snap.census.faults;
    g.census_epoch = snap.census.epoch;
    g.torn_fallbacks = snap.census.torn_fallbacks;
    for c in 0..17 {
        for d in 0..2 {
            let src = snap.census.cells[c][d];
            g.cells[c][d] = MemCell {
                requested: src.requested,
                committed: src.committed,
                freed_requested: src.freed_requested,
                freed_committed: src.freed_committed,
            };
        }
    }
    g.obs_status = snap.observe.status.clamp(0, 2) as u8;
    g.obs_source = snap.observe.source.clamp(0, 2) as u8;
    g.obs_free = snap.observe.free_bytes;
    g.obs_total = snap.observe.total_bytes;
    g.obs_cuda_free = snap.observe.cuda_free_bytes;
    g.obs_meminfo = snap.observe.meminfo_avail_bytes;
    g.substrate_outstanding = snap.substrate_outstanding;
    g.emit_substrate = true;
}

fn write_all(stream: &mut TcpStream, bytes: &[u8]) {
    let _ = stream.write_all(bytes);
}

fn api_for_surface(surf: WireSurface) -> Api {
    match surf {
        WireSurface::Anthropic => Api::Anthropic,
        WireSurface::Responses => Api::Responses,
        _ => Api::Openai,
    }
}

fn continuation_conflict_msg(api: Api) -> &'static str {
    match api {
        Api::Responses => {
            "Responses continuation state is not available; retry by replaying the full input history"
        }
        _ => {
            "Anthropic continuation state is not available; retry by replaying the full messages history"
        }
    }
}

pub fn handle_client(cfg: &ServerConfig, stream: &mut TcpStream) {
    let inner = Mutex::new(ServerInner::from_cfg(cfg));
    handle_client_inner(cfg, &inner, stream, None, None);
}

fn lock_inner(inner: &Mutex<ServerInner>) -> std::sync::MutexGuard<'_, ServerInner> {
    inner.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn handle_client_inner(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    stream: &mut TcpStream,
    engine: Option<&mut dyn DecodeIo>,
    mut cont: Option<&mut dyn ContExec>,
) {
    let req = match read_http_request(stream, chunked_enabled()) {
        Some(r) => r,
        None => {
            write_all(
                stream,
                &wire_http_error_bytes(
                    WireSurface::OpenaiChat,
                    400,
                    "bad HTTP request",
                    cfg.cors,
                    None,
                ),
            );
            return;
        }
    };
    if req.method == "OPTIONS" {
        write_all(stream, &http_response_bytes(204, None, None, cfg.cors, ""));
        return;
    }
    if req.method == "GET" && req.path == "/v1/models" {
        let body = models_list_json(
            &cfg.model_id,
            &cfg.model_name,
            cfg.ctx,
            cfg.default_tokens,
            cfg.codex_models_json.as_deref(),
        );
        write_all(
            stream,
            &http_response_bytes(200, Some("application/json"), None, cfg.cors, &body),
        );
        return;
    }
    if req.method == "GET" && req.path.starts_with("/v1/models/") {
        let id = &req.path["/v1/models/".len()..];
        if model_id_known(&cfg.model_id, id) {
            let body = model_one_json(id, &cfg.model_name, cfg.ctx, cfg.default_tokens);
            write_all(
                stream,
                &http_response_bytes(200, Some("application/json"), None, cfg.cors, &body),
            );
            return;
        }
    }
    if req.method == "GET" && req.path == "/metrics" {
        let body = {
            let g = lock_inner(inner);
            let rt = g.render_runtime(unix_now() as u64);
            render_metrics(&g.metrics, &g.admit, &rt)
        };
        write_all(
            stream,
            &http_response_bytes(200, Some("text/plain; version=0.0.4"), None, cfg.cors, &body),
        );
        return;
    }
    if req.method == "GET" && req.path == "/v1/stats" {
        let body = {
            let g = lock_inner(inner);
            let rt = g.render_runtime(unix_now() as u64);
            render_stats_json_ex(&g.metrics, &g.admit, &rt)
        };
        write_all(
            stream,
            &http_response_bytes(200, Some("application/json"), None, cfg.cors, &body),
        );
        return;
    }
    if req.method == "POST" {
        let inference = shed_surface_for_path(&req.path).is_some();
        let generation = inference || req.path == "/v1/batch";
        if let Some((reason, code, retry, msg)) = {
            let g = lock_inner(inner);
            preparse_shed(&g.admit, inference, generation, req.body.len() as u64)
        } {
            lock_inner(inner).metrics.record_shed(reason);
            let surf = shed_surface_for_path(&req.path).unwrap_or(WireSurface::OpenaiChat);
            write_all(
                stream,
                &wire_http_error_bytes(surf, code, msg, cfg.cors, Some(retry)),
            );
            return;
        }
        if let Some(surf) = parse_surface_for_path(&req.path) {
            let now = unix_now() as f64;
            let live_ids = {
                let mut g = lock_inner(inner);
                g.creg.expire(now);
                g.creg.live_ids(api_for_surface(surf))
            };
            let env = ParseEnv {
                default_model: cfg.model_id.clone(),
                default_tokens: cfg.default_tokens,
                default_effort: ThinkMode::Low,
                default_temp: crate::parse::default_temperature(),
                live_ids,
            };
            let body = std::str::from_utf8(&req.body).unwrap_or("");
            match parse_request(surf, &env, body) {
                Err(e) => {
                    let msg = if e.is_empty() { "invalid JSON request" } else { &e };
                    write_all(
                        stream,
                        &wire_http_error_bytes(surf, 400, msg, cfg.cors, None),
                    );
                    return;
                }
                Ok(parsed) => {
                    let v = {
                        let mut g = lock_inner(inner);
                        enqueue(&mut g.admit, req.body.len() as u64)
                    };
                    if let Some((reason, code, retry, msg)) = enqueue_shed_error(v) {
                        lock_inner(inner).metrics.record_shed(reason);
                        write_all(
                            stream,
                            &wire_http_error_bytes(surf, code, msg, cfg.cors, Some(retry)),
                        );
                        return;
                    }
                    /* The continuous gate needs the tokenized prompt length
                     * (the C server tokenizes during job prep, before
                     * route_decide). Without a cont lane the env keeps the
                     * serial-only shape the oracle tests pin. */
                    let cont_gate = match (cont.as_deref(), engine.is_some()) {
                        (Some(exec), true) => cont_prompt_tokens(exec, &parsed)
                            .ok()
                            .map(|(_, toks)| (toks.len() as i32, exec.seq_cap())),
                        _ => None,
                    };
                    let route_env = RouteEnv {
                        coalesce: cont_gate.is_some(),
                        have_cont: cont_gate.is_some(),
                        cont_anthropic: false,
                        cont_responses: false,
                        cont_tools_anthropic: false,
                        cont_tools_responses: false,
                        seq_cap: cont_gate.map_or(cfg.ctx, |(_, cap)| cap),
                        prompt_len: cont_gate.map_or(0, |(len, _)| len),
                    };
                    let dec = route_decide(parsed.needs, surf, &route_env);
                    let id = {
                        let mut g = lock_inner(inner);
                        next_job_id(&mut g.admit, parsed.kind)
                    };
                    let body_len = req.body.len() as u64;
                    if let Some(engine) = engine {
                        let pin_id = parsed.live_call_ids.first().cloned();
                        if let Some(ref id) = pin_id {
                            let mut g = lock_inner(inner);
                            let _ = g.creg.pin_live(parsed.api, id, now);
                        }
                        if let Some(retry) = {
                            let mut g = lock_inner(inner);
                            g.creg.serial_hold(parsed.api, &parsed.live_call_ids, now)
                        } {
                            {
                                let mut g = lock_inner(inner);
                                if let Some(ref pid) = pin_id {
                                    g.creg.unpin_id(parsed.api, pid);
                                }
                            }
                            lock_inner(inner).metrics.record_shed(SHED_CONT_HOLD);
                            lock_inner(inner).metrics.record_route(
                                surf,
                                dec.lane,
                                dec.reason,
                                parsed.think_mode,
                            );
                            enqueue_release(&mut lock_inner(inner).admit, body_len);
                            write_all(
                                stream,
                                &wire_http_error_bytes(
                                    surf,
                                    503,
                                    "serial capacity is reserved for a live tool continuation; retry shortly",
                                    cfg.cors,
                                    Some(retry),
                                ),
                            );
                            return;
                        }
                        let resolved = {
                            let mut g = lock_inner(inner);
                            g.creg.resolve_serial(
                                parsed.api,
                                &parsed.live_call_ids,
                                engine.generation(),
                                engine.pos(),
                                now,
                            )
                        };
                        let requires_live = parsed.anthropic_requires_live_tool_state
                            || parsed.responses_requires_live_tool_state;
                        if requires_live && !resolved {
                            {
                                let mut g = lock_inner(inner);
                                if let Some(ref pid) = pin_id {
                                    g.creg.unpin_id(parsed.api, pid);
                                }
                            }
                            lock_inner(inner).metrics.record_route(
                                surf,
                                dec.lane,
                                dec.reason,
                                parsed.think_mode,
                            );
                            enqueue_release(&mut lock_inner(inner).admit, body_len);
                            write_all(
                                stream,
                                &wire_http_error_bytes(
                                    surf,
                                    409,
                                    continuation_conflict_msg(parsed.api),
                                    cfg.cors,
                                    None,
                                ),
                            );
                            return;
                        }
                        let created = unix_now();
                        let mut actual_lane = if dec.lane == LANE_CONTINUOUS {
                            LANE_CONTINUOUS
                        } else {
                            crate::route::LANE_SERIAL
                        };
                        let result = if dec.lane == LANE_CONTINUOUS {
                            match cont.as_mut() {
                                Some(exec) => match exec.generate(
                                    &parsed,
                                    &id,
                                    created,
                                    cfg.cors,
                                    cfg.default_tokens,
                                    stream,
                                ) {
                                    /* Rejected before any bytes: the C
                                     * fallback runs the job on the single
                                     * path once the cont loop is free. */
                                    Err(GenerateError::Unsupported(_)) => {
                                        actual_lane = crate::route::LANE_SERIAL;
                                        generate_and_write(
                                            engine,
                                            &parsed,
                                            &id,
                                            created,
                                            cfg.cors,
                                            cfg.default_tokens,
                                            stream,
                                        )
                                    }
                                    r => r,
                                },
                                None => generate_and_write(
                                    engine,
                                    &parsed,
                                    &id,
                                    created,
                                    cfg.cors,
                                    cfg.default_tokens,
                                    stream,
                                ),
                            }
                        } else {
                            generate_and_write(
                                engine,
                                &parsed,
                                &id,
                                created,
                                cfg.cors,
                                cfg.default_tokens,
                                stream,
                            )
                        };
                        lock_inner(inner).metrics.record_route(
                            surf,
                            actual_lane,
                            dec.reason,
                            parsed.think_mode,
                        );
                        {
                            let mut g = lock_inner(inner);
                            if let Some(ref pid) = pin_id {
                                g.creg.unpin_id(parsed.api, pid);
                            }
                            match &result {
                                Ok(out) => {
                                    if matches!(parsed.api, Api::Anthropic | Api::Responses)
                                        && !out.tool_ids.is_empty()
                                        && out.finish != "error"
                                        && out.finish != "length"
                                    {
                                        g.creg.publish_serial(
                                            parsed.api,
                                            &out.tool_ids,
                                            out.generation,
                                            out.frontier,
                                            now,
                                        );
                                    } else {
                                        g.creg.demote_serial();
                                    }
                                }
                                Err(_) => {}
                            }
                        }
                        enqueue_release(&mut lock_inner(inner).admit, body_len);
                        if let Err(e) = result {
                            let (code, msg) = match e {
                                GenerateError::Unsupported(m) => (503, m.to_string()),
                                GenerateError::Engine(m) => (500, m),
                                GenerateError::Io => return,
                            };
                            write_all(
                                stream,
                                &wire_http_error_bytes(surf, code, &msg, cfg.cors, None),
                            );
                        }
                        return;
                    }
                    lock_inner(inner).metrics.record_route(
                        surf,
                        dec.lane,
                        dec.reason,
                        parsed.think_mode,
                    );
                    enqueue_release(&mut lock_inner(inner).admit, body_len);
                    let msg = if cfg.have_engine {
                        "generation remains on C ds4-server"
                    } else {
                        "model not loaded"
                    };
                    write_all(
                        stream,
                        &wire_http_error_bytes(surf, 503, msg, cfg.cors, None),
                    );
                    return;
                }
            }
        }
    }
    write_all(
        stream,
        &wire_http_error_bytes(WireSurface::OpenaiChat, 404, "unknown endpoint", cfg.cors, None),
    );
}

pub fn listen(cfg: &ServerConfig) -> std::io::Result<TcpListener> {
    TcpListener::bind((cfg.listen_host.as_str(), cfg.listen_port))
}

pub fn accept_loop(listener: TcpListener, cfg: ServerConfig) {
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let cfg = cfg.clone();
        let inner = Arc::clone(&inner);
        thread::spawn(move || {
            let _ = stream.set_nodelay(true);
            handle_client_inner(&cfg, &inner, &mut stream, None, None);
        });
    }
}

/// Serial accept when a live engine is attached. The C engine is not
/// `Send`; one request at a time matches the serial lane.
pub fn accept_loop_with_engine(
    listener: TcpListener,
    cfg: ServerConfig,
    engine: &mut dyn DecodeIo,
) {
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let _ = stream.set_nodelay(true);
        handle_client_inner(&cfg, &inner, &mut stream, Some(engine), None);
    }
}

/// Serial accept with a continuous lane attached: requests whose needs
/// word fits the continuous mask run on the engine's rolling scheduler
/// (width-1 today), everything else stays on the serial path.
pub fn accept_loop_with_engine_cont(
    listener: TcpListener,
    cfg: ServerConfig,
    engine: &mut dyn DecodeIo,
    cont: &mut dyn ContExec,
) {
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let _ = stream.set_nodelay(true);
        handle_client_inner(&cfg, &inner, &mut stream, Some(engine), Some(cont));
    }
}

/// Test helper: unused EnqVerdict keep-alive for the admit oracle dump.
pub fn enq_verdict_name(v: EnqVerdict) -> &'static str {
    match v {
        EnqVerdict::Ok => "ok",
        EnqVerdict::Stopping => "stopping",
        EnqVerdict::ShedQueueDepth => "shed_queue_depth",
        EnqVerdict::ShedQueueBytes => "shed_queue_bytes",
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;
    use crate::generate::{GenerateOutcome, ScriptedDecode};
    use crate::parse::ParsedRequest;
    use std::io::{Read, Write};

    struct RejectCont;

    impl ContExec for RejectCont {
        fn model_id(&self) -> i32 {
            0
        }

        fn seq_cap(&self) -> i32 {
            8192
        }

        fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
            vec![1]
        }

        fn encode_text(&self, _text: &str) -> Vec<i32> {
            vec![1]
        }

        fn generate(
            &mut self,
            _parsed: &ParsedRequest,
            _job_id: &str,
            _created: i64,
            _cors: bool,
            _default_tokens: i32,
            _out: &mut dyn Write,
        ) -> Result<GenerateOutcome, GenerateError> {
            Err(GenerateError::Unsupported("serial fallback"))
        }
    }

    #[test]
    fn fallback_records_actual_serial_lane() {
        let cfg = ServerConfig {
            have_engine: true,
            default_tokens: 8,
            ..ServerConfig::default()
        };
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
        write!(
            client,
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
        let mut cont = RejectCont;

        handle_client_inner(
            &cfg,
            &inner,
            &mut server,
            Some(&mut engine),
            Some(&mut cont),
        );
        drop(server);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();

        assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200 OK"));
        let g = inner.lock().unwrap();
        assert_eq!(g.metrics.route_requests[0][0], 1);
        assert_eq!(g.metrics.route_requests[0][1], 0);
    }
}
