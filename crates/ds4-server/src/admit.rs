//! Admission bounds and job-id mint from `ds4_server.c` at v0.6.3-dfm.

use crate::route::ReqKind;

pub const SHED_CLIENTS: u8 = 0;
pub const SHED_QUEUE_DEPTH: u8 = 1;
pub const SHED_QUEUE_BYTES: u8 = 2;
pub const SHED_QUEUE_AGE: u8 = 3;
pub const SHED_SLOW_READER: u8 = 4;
pub const SHED_CONT_HOLD: u8 = 5;
pub const SHED_REASONS: usize = 6;

pub const SHED_NAMES: [&str; SHED_REASONS] = [
    "clients",
    "queue_depth",
    "queue_bytes",
    "queue_age",
    "slow_reader",
    "continuation_hold",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqVerdict {
    Ok = 0,
    Stopping = 1,
    ShedQueueDepth = 2,
    ShedQueueBytes = 3,
}

#[derive(Debug, Clone)]
pub struct AdmitState {
    pub stopping: bool,
    pub queued: i32,
    pub inflight_body_bytes: u64,
    pub max_queue: i32,
    pub max_queue_bytes: u64,
    pub max_clients: i32,
    pub clients: i32,
    pub seq: u64,
}

impl Default for AdmitState {
    fn default() -> Self {
        Self {
            stopping: false,
            queued: 0,
            inflight_body_bytes: 0,
            max_queue: 0,
            max_queue_bytes: 0,
            max_clients: 0,
            clients: 0,
            seq: 0,
        }
    }
}

/// Authoritative enqueue under the queue lock. Matches C `enqueue`.
pub fn enqueue(s: &mut AdmitState, body_bytes: u64) -> EnqVerdict {
    if s.stopping {
        return EnqVerdict::Stopping;
    }
    if s.max_queue > 0 && s.queued >= s.max_queue {
        return EnqVerdict::ShedQueueDepth;
    }
    if s.max_queue_bytes > 0 && s.inflight_body_bytes + body_bytes > s.max_queue_bytes {
        return EnqVerdict::ShedQueueBytes;
    }
    s.queued += 1;
    s.inflight_body_bytes += body_bytes;
    EnqVerdict::Ok
}

pub fn queue_unlink_head(s: &mut AdmitState) {
    if s.queued > 0 {
        s.queued -= 1;
    }
}

pub fn enqueue_release(s: &mut AdmitState, body_bytes: u64) {
    queue_unlink_head(s);
    if s.inflight_body_bytes >= body_bytes {
        s.inflight_body_bytes -= body_bytes;
    } else {
        s.inflight_body_bytes = 0;
    }
}

/// Pre-parse cheap bounds from `client_main`. `generation` covers `/v1/batch`.
pub fn preparse_shed(
    s: &AdmitState,
    inference: bool,
    generation: bool,
    incoming_body: u64,
) -> Option<(u8, i32, i32, &'static str)> {
    if generation && s.max_clients > 0 && s.clients > s.max_clients {
        return Some((
            SHED_CLIENTS,
            503,
            10,
            "server connection capacity reached; retry later",
        ));
    }
    if inference && s.max_queue > 0 && s.queued >= s.max_queue {
        return Some((
            SHED_QUEUE_DEPTH,
            429,
            5,
            "request queue is full; retry later",
        ));
    }
    if inference
        && s.max_queue_bytes > 0
        && s.inflight_body_bytes + incoming_body > s.max_queue_bytes
    {
        return Some((
            SHED_QUEUE_BYTES,
            429,
            5,
            "server request-body budget exhausted; retry later",
        ));
    }
    None
}

pub fn mint_job_id(kind: ReqKind, seq: u64) -> String {
    if kind == ReqKind::Chat {
        format!("chatcmpl-{seq}")
    } else {
        format!("cmpl-{seq}")
    }
}

pub fn next_job_id(s: &mut AdmitState, kind: ReqKind) -> String {
    s.seq += 1;
    mint_job_id(kind, s.seq)
}

/// C `client_main` enqueue Stopping body (`ds4_server.c`).
pub const SERVER_SHUTTING_DOWN: &str = "server shutting down";

pub fn enqueue_shed_error(v: EnqVerdict) -> Option<(u8, i32, i32, &'static str)> {
    match v {
        EnqVerdict::Ok => None,
        EnqVerdict::Stopping => Some((SHED_CLIENTS, 503, 10, SERVER_SHUTTING_DOWN)),
        EnqVerdict::ShedQueueDepth => Some((
            SHED_QUEUE_DEPTH,
            429,
            5,
            "request queue is full; retry later",
        )),
        EnqVerdict::ShedQueueBytes => Some((
            SHED_QUEUE_BYTES,
            429,
            5,
            "server request-body budget exhausted; retry later",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopping_503_body_matches_c_client_main() {
        // Given: C enqueue Stopping at ds4_server.c client_main
        // When: the shed mapper runs
        let shed = enqueue_shed_error(EnqVerdict::Stopping).expect("stopping sheds");

        // Then: exact C production bytes, not the invented "is"
        assert_eq!(shed.0, SHED_CLIENTS);
        assert_eq!(shed.1, 503);
        assert_eq!(shed.3, "server shutting down");
        assert_ne!(shed.3, "server is shutting down");
    }
}
