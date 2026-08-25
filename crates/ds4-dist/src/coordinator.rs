//! Coordinator: accept HELLO, register workers, build a route, dispatch WORK.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use crate::activation::decode_activation;
use crate::codec::{
    decode_hello_payload, Hello, Telemetry, Work, MSG_HELLO, MSG_RESULT, RESULT_HIDDEN_STATE,
    RESULT_LOGITS, ROUTE_F_OUTPUT_LOGITS, WORK_F_INPUT_HC, WORK_F_OUTPUT_LOGITS,
    WORK_F_RESET_SESSION,
};
use crate::hash::token_hash_update_span;
use crate::plan::{build_route_plan, register_worker, CoordinatorView, RoutePlan, WorkerInfo};
use crate::transport::read_frame;
use crate::work::{
    decode_logits_payload, decode_result_body, encode_work_frame, result_hash, result_request_id,
    work_with_ids, WorkBody,
};

#[derive(Debug, Clone)]
pub struct RegisteredWorker {
    pub info: WorkerInfo,
    pub model_name: String,
}

pub struct Coordinator {
    pub view: CoordinatorView,
    pub model_id: u32,
    pub activation_bits: u32,
    pub debug: bool,
    workers: Vec<WorkerInfo>,
    generation: u64,
}

impl Coordinator {
    pub fn new(view: CoordinatorView, model_id: u32, activation_bits: u32) -> Self {
        Self {
            view,
            model_id,
            activation_bits: if activation_bits == 0 {
                32
            } else {
                activation_bits
            },
            debug: false,
            workers: Vec::new(),
            generation: 0,
        }
    }

    pub fn register_hello(&mut self, hello: &Hello, model_name: &str, peer_host: &str) {
        register_worker(
            &mut self.workers,
            WorkerInfo {
                peer_host: peer_host.to_string(),
                listen_port: hello.listen_port,
                model_id: hello.model_id,
                quant_bits: hello.quant_bits,
                layer_start: hello.layer_start,
                layer_end: hello.layer_end,
                has_output: hello.has_output != 0,
                has_hidden: hello.has_hidden != 0,
            },
        );
        let _ = model_name;
        self.generation += 1;
    }

    pub fn workers(&self) -> &[WorkerInfo] {
        &self.workers
    }

    pub fn plan(&self) -> Result<RoutePlan, String> {
        build_route_plan(&self.view, &self.workers)
    }

    pub fn accept_hello<R: Read>(
        &mut self,
        r: &mut R,
        peer_host: &str,
    ) -> Result<(Hello, String), String> {
        let (typ, body) = read_frame(r).map_err(|e| e.to_string())?;
        if typ != MSG_HELLO {
            return Err(format!("expected HELLO frame, got type {typ}"));
        }
        let (hello, name) = decode_hello_payload(&body).map_err(|e| e.to_string())?;
        self.register_hello(&hello, &name, peer_host);
        Ok((hello, name))
    }
}

pub fn listen(host: &str, port: u16) -> std::io::Result<TcpListener> {
    TcpListener::bind((host, port))
}

pub const ERR_DATA_LISTEN_PORT: &str = "could not determine data listener port";

pub fn open_data_listener(host: Option<&str>, port: u16) -> std::io::Result<(TcpListener, u16)> {
    let host = host.unwrap_or("0.0.0.0");
    let listener = listen(host, port)?;
    let bound = listener.local_addr()?.port();
    if bound == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AddrNotAvailable,
            ERR_DATA_LISTEN_PORT,
        ));
    }
    Ok((listener, bound))
}

/// Dispatch one remote eval on an already-connected first-hop stream.
///
/// `prefix_hash` is the committed token-hash *before* this span (C session
/// timeline). `pos0 == 0` + reset uses `TOKEN_HASH_INIT`.
pub fn dispatch_eval<S: Read + Write>(
    coord: &Coordinator,
    stream: &mut S,
    tokens: &[i32],
    pos0: u32,
    session_id: u64,
    request_id: u64,
    reset: bool,
    prefix_hash: u64,
    input_hc: &[f32],
) -> Result<EvalOutcome, String> {
    let plan = coord.plan()?;
    if plan.entries.is_empty() {
        return Err("distributed route has no remote worker".into());
    }
    let first = &plan.entries[0];
    let expect_hash = token_hash_update_span(prefix_hash, tokens);
    let mut flags = 0u32;
    if first.layer_start != 0 {
        flags |= WORK_F_INPUT_HC;
    }
    if reset {
        flags |= WORK_F_RESET_SESSION;
    }
    if (first.flags & ROUTE_F_OUTPUT_LOGITS) != 0 {
        flags |= WORK_F_OUTPUT_LOGITS;
    }
    let work = work_with_ids(
        Work {
            model_id: coord.model_id,
            pos0,
            n_tokens: tokens.len() as u32,
            layer_start: first.layer_start,
            layer_end: first.layer_end,
            flags,
            token_bytes: 0,
            input_hc_bytes: 0,
            input_hc_bits: coord.activation_bits,
            route_count: plan.entries.len() as u32,
            route_index: 0,
            route_bytes: 0,
            ..Work::default()
        },
        session_id,
        request_id,
        prefix_hash,
        expect_hash,
    );
    let body = WorkBody {
        work,
        tokens: tokens.to_vec(),
        input_hc: input_hc.to_vec(),
        route_blob: plan.blob,
    };
    let frame = encode_work_frame(&body).map_err(|e| e.to_string())?;
    stream.write_all(&frame).map_err(|e| e.to_string())?;
    let (typ, reply) = read_frame(stream).map_err(|e| e.to_string())?;
    if typ != MSG_RESULT {
        return Err("distributed worker returned invalid frame".into());
    }
    let result = decode_result_body(&reply).map_err(|e| e.to_string())?;
    if coord.debug {
        for (hop, tel) in result.telemetry.iter().enumerate() {
            eprint!("{}", format_telemetry_line(request_id, hop as u32, tel));
        }
    }
    if result_request_id(&result.hdr) != request_id {
        return Err("distributed result metadata mismatch".into());
    }
    if result.hdr.status != 0 {
        let msg = if result.payload.is_empty() {
            "distributed worker returned an error".into()
        } else {
            String::from_utf8_lossy(&result.payload).into_owned()
        };
        return Err(msg);
    }
    if result_hash(&result.hdr) != expect_hash {
        return Err("distributed result prefix hash mismatch".into());
    }
    match result.hdr.result_kind {
        RESULT_LOGITS => Ok(EvalOutcome::Logits(
            decode_logits_payload(&result.payload).map_err(|e| e.to_string())?,
        )),
        RESULT_HIDDEN_STATE => {
            let hidden =
                decode_activation(&result.payload, result.hdr.payload_bits).ok_or_else(|| {
                    "distributed route returned invalid hidden-state size".to_string()
                })?;
            Ok(EvalOutcome::Hidden(hidden))
        }
        _ => Err("distributed route did not return logits or hidden-state".into()),
    }
}

pub fn format_telemetry_line(request_id: u64, hop: u32, tel: &Telemetry) -> String {
    format!(
        "ds4: distributed telemetry: request={request_id} hop={hop} layers={}:{} route={} pos={} tokens={} eval={:.3}ms downstream_wait={:.3}ms forward_send={:.3}ms input={:.2}MiB output={:.2}MiB\n",
        tel.layer_start,
        tel.layer_end,
        tel.route_index,
        tel.pos0,
        tel.n_tokens,
        f64::from(tel.eval_usec) / 1000.0,
        f64::from(tel.downstream_wait_usec) / 1000.0,
        f64::from(tel.forward_send_usec) / 1000.0,
        f64::from(tel.input_bytes) / (1024.0 * 1024.0),
        f64::from(tel.output_bytes) / (1024.0 * 1024.0),
    )
}

#[derive(Debug, Clone)]
pub enum EvalOutcome {
    Logits(Vec<f32>),
    Hidden(Vec<f32>),
}

/// Shared registry used by an accept thread.
pub type SharedCoordinator = Arc<Mutex<Coordinator>>;

pub fn accept_loop(listener: TcpListener, coord: SharedCoordinator) {
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let peer = stream
            .peer_addr()
            .map(|a| a.ip().to_string())
            .unwrap_or_else(|_| "unknown".into());
        let mut guard = match coord.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let _ = guard.accept_hello(&mut stream, &peer);
    }
}

pub fn recv_hello_only<R: Read>(r: &mut R) -> Result<(Hello, String), String> {
    let (typ, body) = read_frame(r).map_err(|e| e.to_string())?;
    if typ != MSG_HELLO {
        return Err(format!("expected HELLO frame, got type {typ}"));
    }
    decode_hello_payload(&body).map_err(|e| e.to_string())
}

pub fn token_span_hashes(committed: &[i32], span: &[i32]) -> (u64, u64) {
    let prefix = crate::hash::token_hash_prefix(committed);
    let result = token_hash_update_span(prefix, span);
    (prefix, result)
}
