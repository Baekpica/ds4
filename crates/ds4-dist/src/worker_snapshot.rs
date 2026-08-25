//! Worker-side DS4D snapshot save/load frame handling.

use std::io::{Read, Write};

use crate::codec::{
    decode_snapshot_chunk_body, decode_snapshot_load_begin_body, encode_snapshot_begin_body,
    encode_snapshot_chunk_body, encode_snapshot_done_body, u64_from_halves, u64_to_halves,
    SnapshotBegin, SnapshotReq, MSG_SNAPSHOT_BEGIN, MSG_SNAPSHOT_CHUNK, MSG_SNAPSHOT_DONE,
    SNAPSHOT_BEGIN_FIXED_BYTES, SNAPSHOT_CHUNK_BYTES, SNAPSHOT_REQ_FIXED_BYTES,
};
use crate::hash::token_hash_prefix;
use crate::transport::{read_frame, write_frame};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerSnapshotIdentity {
    pub model_id: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub ctx_size: u32,
}

fn write_begin<S: Write>(
    stream: &mut S,
    begin: &SnapshotBegin,
    tokens: &[i32],
    message: &[u8],
) -> Result<(), String> {
    let body = encode_snapshot_begin_body(begin, tokens, message).map_err(|e| e.to_string())?;
    write_frame(stream, MSG_SNAPSHOT_BEGIN, &body).map_err(|e| e.to_string())
}

fn write_error_begin<S: Write>(
    stream: &mut S,
    identity: WorkerSnapshotIdentity,
    request_id: u64,
    session_id: u64,
    message: &str,
) -> Result<(), String> {
    let (session_hi, session_lo) = u64_to_halves(session_id);
    let (request_hi, request_lo) = u64_to_halves(request_id);
    let begin = SnapshotBegin {
        model_id: identity.model_id,
        session_hi,
        session_lo,
        request_hi,
        request_lo,
        token_hash_hi: 0,
        token_hash_lo: 0,
        token_count: 0,
        layer_start: identity.layer_start,
        layer_end: identity.layer_end,
        payload_hi: 0,
        payload_lo: 0,
        status: 1,
        token_bytes: 0,
        message_bytes: message.len() as u32,
    };
    write_begin(stream, &begin, &[], message.as_bytes())
}

fn write_done<S: Write>(
    stream: &mut S,
    request_id: u64,
    status: u32,
    message: &str,
) -> Result<(), String> {
    let body = encode_snapshot_done_body(request_id, status, message.as_bytes())
        .map_err(|e| e.to_string())?;
    write_frame(stream, MSG_SNAPSHOT_DONE, &body).map_err(|e| e.to_string())
}

fn write_chunks<S: Write, R: Read>(
    stream: &mut S,
    request_id: u64,
    payload: &mut R,
    mut remaining: u64,
) -> Result<(), String> {
    let mut buf = Vec::new();
    if remaining != 0 {
        buf.try_reserve_exact(SNAPSHOT_CHUNK_BYTES)
            .map_err(|_| "out of memory sending worker KV shard".to_string())?;
        buf.resize(SNAPSHOT_CHUNK_BYTES, 0);
    }
    while remaining != 0 {
        let n = remaining.min(SNAPSHOT_CHUNK_BYTES as u64) as usize;
        payload
            .read_exact(&mut buf[..n])
            .map_err(|_| "failed to read worker KV shard".to_string())?;
        let body = encode_snapshot_chunk_body(request_id, &buf[..n]).map_err(|e| e.to_string())?;
        write_frame(stream, MSG_SNAPSHOT_CHUNK, &body).map_err(|e| e.to_string())?;
        remaining -= n as u64;
    }
    Ok(())
}

impl WorkerSnapshotIdentity {
    pub fn accepts(self, model_id: u32, start: u32, end: u32, tokens: u32) -> bool {
        model_id == self.model_id
            && start == self.layer_start
            && end == self.layer_end
            && tokens <= self.ctx_size
    }
}

/// Answer one SAVE_REQ body with BEGIN+CHUNK+DONE, or a C error BEGIN.
pub fn worker_handle_snapshot_save<S, R>(
    stream: &mut S,
    identity: WorkerSnapshotIdentity,
    req_body: &[u8],
    payload: Result<(&mut R, u64), String>,
) -> Result<(), String>
where
    S: Write,
    R: Read,
{
    if req_body.len() != SNAPSHOT_REQ_FIXED_BYTES {
        return write_error_begin(
            stream,
            identity,
            0,
            0,
            "invalid distributed snapshot save request",
        );
    }
    let req = SnapshotReq::decode(req_body).map_err(|e| e.to_string())?;
    let request_id = u64_from_halves(req.request_hi, req.request_lo);
    let session_id = u64_from_halves(req.session_hi, req.session_lo);
    if !identity.accepts(
        req.model_id,
        req.layer_start,
        req.layer_end,
        req.token_count,
    ) {
        return write_error_begin(
            stream,
            identity,
            request_id,
            session_id,
            "snapshot save request does not match worker state",
        );
    }
    let (payload, payload_bytes) = match payload {
        Ok(ok) => ok,
        Err(message) => {
            return write_error_begin(stream, identity, request_id, session_id, &message);
        }
    };
    let (payload_hi, payload_lo) = u64_to_halves(payload_bytes);
    let begin = SnapshotBegin {
        model_id: identity.model_id,
        session_hi: req.session_hi,
        session_lo: req.session_lo,
        request_hi: req.request_hi,
        request_lo: req.request_lo,
        token_hash_hi: req.token_hash_hi,
        token_hash_lo: req.token_hash_lo,
        token_count: 0,
        layer_start: identity.layer_start,
        layer_end: identity.layer_end,
        payload_hi,
        payload_lo,
        status: 0,
        token_bytes: 0,
        message_bytes: 0,
    };
    write_begin(stream, &begin, &[], b"")?;
    write_chunks(stream, request_id, payload, payload_bytes)?;
    write_done(stream, request_id, 0, "")
}

/// Consume one LOAD_BEGIN body plus following CHUNKs, then send DONE.
pub fn worker_handle_snapshot_load<S, W>(
    stream: &mut S,
    identity: WorkerSnapshotIdentity,
    begin_body: &[u8],
    vocab_size: u32,
    payload: &mut W,
) -> Result<WorkerLoadOffer, String>
where
    S: Read + Write,
    W: Write,
{
    worker_handle_snapshot_load_restore(stream, identity, begin_body, vocab_size, payload, |_| {
        Ok(())
    })
}

/// Same as [`worker_handle_snapshot_load`], but runs `restore` before DONE.
pub fn worker_handle_snapshot_load_restore<S, W, F>(
    stream: &mut S,
    identity: WorkerSnapshotIdentity,
    begin_body: &[u8],
    vocab_size: u32,
    payload: &mut W,
    restore: F,
) -> Result<WorkerLoadOffer, String>
where
    S: Read + Write,
    W: Write,
    F: FnOnce(&WorkerLoadOffer) -> Result<(), String>,
{
    if begin_body.len() < SNAPSHOT_BEGIN_FIXED_BYTES {
        return Err("invalid distributed snapshot load header".into());
    }

    let Ok((begin, tokens, _)) = decode_snapshot_load_begin_body(begin_body) else {
        write_done(stream, 0, 1, "invalid distributed snapshot load header")?;
        return Err("invalid distributed snapshot load header".into());
    };
    let request_id = u64_from_halves(begin.request_hi, begin.request_lo);
    let session_id = u64_from_halves(begin.session_hi, begin.session_lo);
    let token_hash = u64_from_halves(begin.token_hash_hi, begin.token_hash_lo);
    let payload_bytes = u64_from_halves(begin.payload_hi, begin.payload_lo);
    let mut err = if tokens
        .iter()
        .any(|token| *token < 0 || (*token as u32) >= vocab_size)
    {
        "snapshot token id is outside the model vocabulary"
    } else if token_hash_prefix(&tokens) != token_hash {
        "snapshot load token hash mismatch"
    } else if !identity.accepts(
        begin.model_id,
        begin.layer_start,
        begin.layer_end,
        begin.token_count,
    ) {
        "snapshot load request does not match worker state"
    } else {
        ""
    }
    .to_string();

    let mut received = 0u64;
    while err.is_empty() && received < payload_bytes {
        let (typ, body) = match read_frame(stream) {
            Ok(frame) => frame,
            Err(_) => {
                let _ = write_done(stream, request_id, 1, "expected distributed snapshot chunk");
                return Err("expected distributed snapshot chunk".into());
            }
        };
        if typ != MSG_SNAPSHOT_CHUNK {
            err = "expected distributed snapshot chunk".into();
            break;
        }
        match decode_snapshot_chunk_body(&body, request_id, payload_bytes - received) {
            Ok((_, chunk)) => {
                if payload.write_all(chunk).is_err() {
                    err = "failed to write worker KV shard temp file".into();
                    break;
                }
                received += chunk.len() as u64;
            }
            Err(_) => {
                err = "invalid distributed snapshot chunk".into();
                break;
            }
        }
    }

    if err.is_empty() && payload.flush().is_err() {
        err = "failed to flush worker KV shard restore file".into();
    }
    let offer = WorkerLoadOffer {
        session_id,
        request_id,
        token_hash,
        tokens,
        payload_bytes,
    };
    if err.is_empty() {
        if let Err(message) = restore(&offer) {
            err = message;
        }
    }
    let status = u32::from(!err.is_empty());
    write_done(stream, request_id, status, &err)?;
    if !err.is_empty() {
        return Err(err);
    }
    Ok(offer)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerLoadOffer {
    pub session_id: u64,
    pub request_id: u64,
    pub token_hash: u64,
    pub tokens: Vec<i32>,
    pub payload_bytes: u64,
}
