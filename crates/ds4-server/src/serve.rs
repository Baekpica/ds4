//! Blocking accept loop: HTTP door + /v1/models. Generation stays C.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::thread;

use crate::error::{http_response_bytes, wire_http_error_bytes};
use crate::http::{chunked_enabled, parse_surface_for_path, read_http_request};
use crate::models::{model_id_known, model_one_json, models_list_json};
use crate::parse::{parse_request, ParseEnv};
use crate::route::{ThinkMode, WireSurface};

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
        }
    }
}

fn write_all(stream: &mut TcpStream, bytes: &[u8]) {
    let _ = stream.write_all(bytes);
}

pub fn handle_client(cfg: &ServerConfig, stream: &mut TcpStream) {
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
    if req.method == "POST" {
        if let Some(surf) = parse_surface_for_path(&req.path) {
            let env = ParseEnv {
                default_model: cfg.model_id.clone(),
                default_tokens: cfg.default_tokens,
                default_effort: ThinkMode::Low,
                default_temp: crate::parse::default_temperature(),
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
                Ok(_) => {}
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
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let cfg = cfg.clone();
        thread::spawn(move || {
            let _ = stream.set_nodelay(true);
            handle_client(&cfg, &mut stream);
        });
    }
}
