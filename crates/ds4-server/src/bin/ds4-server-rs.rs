//! Shadow HTTP host. GET surfaces are live; family decode uses the
//! native FFI when `-m` opens a model. Continuation registry is host-owned.
//! Incremental live DSML tool projection is host-owned.

use ds4_core::{Backend, Model};
use ds4_server::{
    accept_loop, accept_loop_with_engine, accept_loop_with_engine_cont, listen,
    model_id_from_gguf_path, ContLane, NativeDecode, ServerConfig,
};

fn main() {
    let mut cfg = ServerConfig::default();
    let mut model_path: Option<String> = None;
    let mut backend = Backend::Cuda;
    let mut n_threads = 0i32;
    let mut cont_width = 2i32;
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
                model_path = Some(path);
            }
            "--backend" => {
                backend = match args.next().unwrap_or_else(|| usage()).as_str() {
                    "cuda" => Backend::Cuda,
                    "cpu" => Backend::Cpu,
                    "metal" => Backend::Metal,
                    other => {
                        eprintln!("ds4-server-rs: unknown backend {other}");
                        std::process::exit(2);
                    }
                };
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
            "-t" | "--threads" => {
                n_threads = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--cont-width" => {
                cont_width = args
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

    let model = match model_path.as_deref() {
        Some(path) => match Model::open(path, backend, n_threads, true) {
            Ok(m) => {
                cfg.have_engine = true;
                Some(m)
            }
            Err(e) => {
                eprintln!("ds4-server-rs: open {path}: {e}");
                std::process::exit(1);
            }
        },
        None => None,
    };

    let listener = listen(&cfg).unwrap_or_else(|e| {
        eprintln!(
            "ds4-server-rs: listen {}:{}: {e}",
            cfg.listen_host, cfg.listen_port
        );
        std::process::exit(1);
    });
    eprintln!(
        "ds4-server-rs: listening on {}:{} model_id={} engine={} host_vocab={} (host continuation registry + incremental live DSML tool stream + corrective retry)",
        cfg.listen_host,
        cfg.listen_port,
        cfg.model_id,
        if cfg.have_engine { "open" } else { "none" },
        if model.is_some() { "yes" } else { "no" }
    );

    if let Some(ref model) = model {
        let lane = if cont_width > 0 && backend == Backend::Cuda {
            match model.batch_ctx_fit(cfg.ctx, cont_width, cfg.ctx.saturating_mul(cont_width)) {
                Ok(batch) => {
                    eprintln!(
                        "ds4-server-rs: continuous lane ready (width={} seq_cap={})",
                        batch.max_seq(),
                        batch.seq_cap()
                    );
                    Some(ContLane {
                        batch,
                        vocab: model.vocab(),
                        model_id: model.model_id(),
                        eos: model.token_eos(),
                    })
                }
                Err(e) => {
                    eprintln!("ds4-server-rs: continuous lane unavailable ({e}); serial only");
                    None
                }
            }
        } else {
            None
        };
        let mut engine = NativeDecode::new(model, cfg.ctx).with_vocab(model.vocab());
        match lane {
            Some(mut lane) => {
                accept_loop_with_engine_cont(listener, cfg, &mut engine, &mut lane)
            }
            None => accept_loop_with_engine(listener, cfg, &mut engine),
        }
    } else {
        accept_loop(listener, cfg);
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: ds4-server-rs --listen HOST PORT [--model-id ID] [-m GGUF] [--backend cuda|cpu|metal] [--tokens N] [-c N] [-t N] [--cont-width N] [--cors]"
    );
    std::process::exit(2);
}
