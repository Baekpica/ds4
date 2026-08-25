//! Worker-to-worker forward window and pending-request queue.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

use crate::codec::Telemetry;

pub const FORWARD_WINDOW_DEFAULT: u32 = 4;
pub const FORWARD_WINDOW_MIN: u32 = 1;
pub const FORWARD_WINDOW_MAX: u32 = 64;

pub const ERR_NEXT_CLOSED: &str = "next worker closed connection";
pub const ERR_INVALID_RESULT: &str = "next worker did not return valid RESULT";
pub const ERR_CLOSED_WHILE_RESULT: &str = "next worker closed while returning RESULT";
pub const ERR_RESULT_METADATA: &str = "next worker RESULT metadata mismatch";
pub const ERR_TELEMETRY_TOO_LARGE: &str = "distributed telemetry chain is too large";
pub const ERR_RESULT_TOO_LARGE: &str = "distributed RESULT frame is too large";
pub const ERR_OOM_FORWARDER: &str = "out of memory creating worker-to-worker forwarder";
pub const ERR_RELAY_THREAD: &str = "failed to start worker-to-worker relay thread";
pub const ERR_OOM_TRACK: &str = "out of memory tracking forwarded request";
pub const ERR_FORWARD: &str = "failed to forward distributed work";
pub const ERR_FORWARD_HIDDEN: &str = "invalid forwarded hidden-state size";

pub fn forward_window_from(env: Option<&str>) -> u32 {
    let Some(raw) = env else {
        return FORWARD_WINDOW_DEFAULT;
    };
    if raw.is_empty() {
        return FORWARD_WINDOW_DEFAULT;
    }
    match parse_strtol(raw) {
        Some(v) if v >= i64::from(FORWARD_WINDOW_MIN) && v <= i64::from(FORWARD_WINDOW_MAX) => {
            v as u32
        }
        _ => FORWARD_WINDOW_DEFAULT,
    }
}

pub fn forward_window() -> u32 {
    forward_window_from(
        std::env::var("DS4_DIST_WORKER_FORWARD_WINDOW")
            .ok()
            .as_deref(),
    )
}

pub fn opened_forwarder_message(host: &str, port: u32, window: u32) -> String {
    format!(
        "ds4: distributed worker: opened pipelined worker-to-worker connection to {host}:{port} (window {window})"
    )
}

fn parse_strtol(s: &str) -> Option<i64> {
    let s = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if s.is_empty() {
        return None;
    }
    let (neg, digits) = match s.as_bytes()[0] {
        b'+' => (false, &s[1..]),
        b'-' => (true, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() || !digits.as_bytes()[0].is_ascii_digit() {
        return None;
    }
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let v = digits.parse::<i64>().ok()?;
    Some(if neg { -v } else { v })
}

#[derive(Debug, Clone)]
pub struct PendingRequest {
    pub request_id: u64,
    pub telemetry: Telemetry,
    pub downstream_t0: f64,
}

struct QueueInner {
    depth: u32,
    queued: u32,
    jobs: VecDeque<PendingRequest>,
    closing: bool,
}

pub struct PendingQueue {
    inner: Mutex<QueueInner>,
    not_full: Condvar,
}

impl PendingQueue {
    pub fn new(depth: u32) -> Self {
        Self {
            inner: Mutex::new(QueueInner {
                depth: depth.max(1),
                queued: 0,
                jobs: VecDeque::new(),
                closing: false,
            }),
            not_full: Condvar::new(),
        }
    }

    pub fn depth(&self) -> u32 {
        self.inner.lock().expect("forward queue").depth
    }

    pub fn enqueue(&self, job: PendingRequest) -> bool {
        let mut g = self.inner.lock().expect("forward queue");
        while !g.closing && g.queued >= g.depth {
            g = self.not_full.wait(g).expect("forward queue");
        }
        if g.closing {
            return false;
        }
        g.jobs.push_back(job);
        g.queued += 1;
        true
    }

    pub fn pop(&self) -> Option<PendingRequest> {
        let mut g = self.inner.lock().expect("forward queue");
        let job = g.jobs.pop_front()?;
        g.queued = g.queued.saturating_sub(1);
        self.not_full.notify_one();
        Some(job)
    }

    pub fn remove(&self, request_id: u64) -> bool {
        let mut g = self.inner.lock().expect("forward queue");
        if let Some(at) = g.jobs.iter().position(|job| job.request_id == request_id) {
            g.jobs.remove(at);
            g.queued = g.queued.saturating_sub(1);
            self.not_full.notify_one();
            return true;
        }
        false
    }

    pub fn note_send_done(&self, request_id: u64, forward_send_usec: u32, downstream_t0: f64) {
        let mut g = self.inner.lock().expect("forward queue");
        if let Some(job) = g.jobs.iter_mut().find(|job| job.request_id == request_id) {
            job.telemetry.forward_send_usec = forward_send_usec;
            job.downstream_t0 = downstream_t0;
        }
    }

    pub fn close(&self) {
        let mut g = self.inner.lock().expect("forward queue");
        g.closing = true;
        self.not_full.notify_all();
    }

    pub fn clear(&self) {
        let mut g = self.inner.lock().expect("forward queue");
        g.jobs.clear();
        g.queued = 0;
        self.not_full.notify_all();
    }
}
