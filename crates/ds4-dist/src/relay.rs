//! Persistent worker-to-worker forwarder and RESULT relay thread.

use std::io::{self, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::codec::{Telemetry, MSG_RESULT};
use crate::forward::{
    opened_forwarder_message, PendingQueue, PendingRequest, ERR_FORWARD, ERR_INVALID_RESULT,
    ERR_NEXT_CLOSED, ERR_OOM_TRACK, ERR_RESULT_METADATA, ERR_RESULT_TOO_LARGE,
    ERR_TELEMETRY_TOO_LARGE,
};
use crate::reconnect::connect_endpoint;
use crate::route::RouteEntry;
use crate::transport::read_frame;
use crate::work::{
    decode_result_body, encode_result_frame, encode_work_frame, result_request_id, WorkBody,
};

pub fn now_sec() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

pub fn local_work_telemetry(
    work: &crate::codec::Work,
    eval_usec: u32,
    output_bytes: u32,
) -> Telemetry {
    Telemetry {
        layer_start: work.layer_start,
        layer_end: work.layer_end,
        route_index: work.route_index,
        pos0: work.pos0,
        n_tokens: work.n_tokens,
        eval_usec,
        downstream_wait_usec: 0,
        forward_send_usec: 0,
        input_bytes: work.token_bytes.saturating_add(work.input_hc_bytes),
        output_bytes,
    }
}

pub fn usec_since(t0: f64, t1: f64) -> u32 {
    if t1 <= t0 {
        return 0;
    }
    let usec = (t1 - t0) * 1_000_000.0;
    if usec >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        (usec + 0.5) as u32
    }
}

pub fn prepend_telemetry(result_payload: &[u8], local: Telemetry) -> Result<Vec<u8>, &'static str> {
    let mut body = decode_result_body(result_payload).map_err(|_| ERR_RESULT_METADATA)?;
    if body.telemetry.len() >= u32::MAX as usize {
        return Err(ERR_TELEMETRY_TOO_LARGE);
    }
    body.telemetry.push(local);
    encode_result_frame(&body).map_err(|_| ERR_RESULT_TOO_LARGE)
}

pub struct Forwarder {
    pub host: String,
    pub port: u32,
    writer: Mutex<TcpStream>,
    queue: Arc<PendingQueue>,
    stop: Arc<AtomicBool>,
    relay: Option<JoinHandle<()>>,
}

impl Forwarder {
    pub fn connect(
        host: &str,
        port: u32,
        window: u32,
        on_result: impl Fn(Vec<u8>) + Send + 'static,
        on_error: impl Fn(u64, &'static str) + Send + 'static,
    ) -> Result<Self, String> {
        let stream = connect_endpoint(host, port as u16).map_err(|e| e.to_string())?;
        let mut reader = stream.try_clone().map_err(|e| e.to_string())?;
        let queue = Arc::new(PendingQueue::new(window));
        let stop = Arc::new(AtomicBool::new(false));
        let relay_q = Arc::clone(&queue);
        let relay_stop = Arc::clone(&stop);
        let relay = thread::Builder::new()
            .name(format!("ds4-dist-relay-{host}:{port}"))
            .spawn(move || relay_main(&mut reader, &relay_q, &relay_stop, on_result, on_error))
            .map_err(|e| e.to_string())?;
        eprintln!("{}", opened_forwarder_message(host, port, window));
        Ok(Self {
            host: host.to_string(),
            port,
            writer: Mutex::new(stream),
            queue,
            stop,
            relay: Some(relay),
        })
    }

    pub fn send_work(&self, request: PendingRequest, frame: &[u8]) -> Result<(), &'static str> {
        let request_id = request.request_id;
        if !self.queue.enqueue(request) {
            return Err(ERR_OOM_TRACK);
        }
        let send_t0 = now_sec();
        let write_rc = self.writer.lock().expect("forward send").write_all(frame);
        let send_t1 = now_sec();
        self.queue
            .note_send_done(request_id, usec_since(send_t0, send_t1), send_t1);
        if write_rc.is_err() {
            self.queue.remove(request_id);
            let _ = self
                .writer
                .lock()
                .expect("forward send")
                .shutdown(std::net::Shutdown::Both);
            return Err(ERR_FORWARD);
        }
        Ok(())
    }

    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.queue.close();
        if let Ok(w) = self.writer.lock() {
            let _ = w.shutdown(std::net::Shutdown::Both);
        }
        if let Some(relay) = self.relay.take() {
            let _ = relay.join();
        }
        self.queue.clear();
    }
}

impl Drop for Forwarder {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn relay_main(
    reader: &mut TcpStream,
    queue: &PendingQueue,
    stop: &AtomicBool,
    on_result: impl Fn(Vec<u8>),
    on_error: impl Fn(u64, &'static str),
) {
    while !stop.load(Ordering::Relaxed) {
        let (typ, body) = match read_frame(reader) {
            Ok(v) => v,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof || stop.load(Ordering::Relaxed) => {
                if let Some(pending) = queue.pop() {
                    on_error(pending.request_id, ERR_NEXT_CLOSED);
                }
                break;
            }
            Err(_) => {
                if let Some(pending) = queue.pop() {
                    on_error(pending.request_id, ERR_NEXT_CLOSED);
                }
                break;
            }
        };
        if typ != MSG_RESULT {
            if let Some(pending) = queue.pop() {
                on_error(pending.request_id, ERR_INVALID_RESULT);
            }
            break;
        }
        let decoded = match decode_result_body(&body) {
            Ok(v) => v,
            Err(_) => {
                if let Some(pending) = queue.pop() {
                    on_error(pending.request_id, ERR_RESULT_METADATA);
                }
                break;
            }
        };
        let Some(mut pending) = queue.pop() else {
            break;
        };
        if result_request_id(&decoded.hdr) != pending.request_id {
            on_error(pending.request_id, ERR_RESULT_METADATA);
            break;
        }
        pending.telemetry.downstream_wait_usec = usec_since(pending.downstream_t0, now_sec());
        match prepend_telemetry(&body, pending.telemetry) {
            Ok(frame) => on_result(frame),
            Err(msg) => on_error(pending.request_id, msg),
        }
    }
    queue.close();
}

pub fn forward_work_blocking<W: Write>(
    next: &RouteEntry,
    body: &WorkBody,
    upstream: &mut W,
    mut local: Telemetry,
) -> Result<(), String> {
    let mut hop = connect_endpoint(&next.host, next.port as u16).map_err(|_| ERR_FORWARD)?;
    let frame = encode_work_frame(body).map_err(|e| e.to_string())?;
    let send_t0 = now_sec();
    hop.write_all(&frame).map_err(|_| ERR_FORWARD)?;
    let send_t1 = now_sec();
    local.forward_send_usec = usec_since(send_t0, send_t1);
    let (typ, reply) = read_frame(&mut hop).map_err(|_| ERR_FORWARD)?;
    if typ != MSG_RESULT {
        return Err(ERR_INVALID_RESULT.into());
    }
    local.downstream_wait_usec = usec_since(send_t1, now_sec());
    let prepended = prepend_telemetry(&reply, local).map_err(|e| e.to_string())?;
    upstream.write_all(&prepended).map_err(|e| e.to_string())
}
