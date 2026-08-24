//! Shadow HTTP door. GET /v1/models is live; generation stays on C ds4-server.

use ds4_server::{accept_loop, listen, model_id_from_gguf_path, ServerConfig};

fn main() {
    let mut cfg = ServerConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen" => {
                cfg.listen_host = args.next().unwrap_or_else(|| usage());
                cfg.listen_port = args
                    .next()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--model-id" => cfg.model_id = args.next().unwrap_or_else(|| usage()),
            "--model" | "-m" => {
                let path = args.next().unwrap_or_else(|| usage());
                if let Some(id) = model_id_from_gguf_path(&path) {
                    if cfg.model_id == "ds4" {
                        cfg.model_id = id;
                    }
                }
            }
            "--tokens" => {
                cfg.default_tokens = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "-c" | "--ctx" => {
                cfg.ctx = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--cors" => cfg.cors = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("ds4-server-rs: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if cfg.model_name == "ds4" {
        cfg.model_name = cfg.model_id.clone();
    }
    let listener = listen(&cfg).unwrap_or_else(|e| {
        eprintln!("ds4-server-rs: listen {}:{}: {e}", cfg.listen_host, cfg.listen_port);
        std::process::exit(1);
    });
    eprintln!(
        "ds4-server-rs: listening on {}:{} model_id={} (shadow; generation remains C)",
        cfg.listen_host, cfg.listen_port, cfg.model_id
    );
    accept_loop(listener, cfg);
}

fn usage() -> ! {
    eprintln!(
        "usage: ds4-server-rs --listen HOST PORT [--model-id ID] [-m GGUF] [--tokens N] [-c N] [--cors]"
    );
    std::process::exit(2);
}
