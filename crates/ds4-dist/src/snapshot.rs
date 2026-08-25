//! Coordinator-side DS4D snapshot save/load transport.

use std::io::{self, Read, Write};

use crate::codec::{
    decode_frame_header, encode_frame_header, u64_from_halves, u64_to_halves, FrameHeader,
    SnapshotBegin, SnapshotChunk, SnapshotDone, SnapshotReq, FRAME_HEADER_BYTES,
    MSG_SNAPSHOT_BEGIN, MSG_SNAPSHOT_CHUNK, MSG_SNAPSHOT_DONE, MSG_SNAPSHOT_LOAD_BEGIN,
    MSG_SNAPSHOT_SAVE_REQ, SNAPSHOT_BEGIN_FIXED_BYTES, SNAPSHOT_CHUNK_BYTES,
    SNAPSHOT_CHUNK_FIXED_BYTES, SNAPSHOT_DONE_FIXED_BYTES,
};
use crate::transport::write_frame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotMeta {
    pub model_id: u32,
    pub session_id: u64,
    pub request_id: u64,
    pub token_hash: u64,
    pub layer_start: u32,
    pub layer_end: u32,
}

fn token_count(tokens: &[i32]) -> Result<u32, String> {
    u32::try_from(tokens.len()).map_err(|_| "invalid distributed snapshot token length".to_string())
}

fn c_message(message: &[u8]) -> String {
    let end = message
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(message.len());
    String::from_utf8_lossy(&message[..end]).into_owned()
}

fn read_header<R: Read>(r: &mut R, closed: &'static str) -> Result<FrameHeader, String> {
    let mut buf = [0; FRAME_HEADER_BYTES];
    r.read_exact(&mut buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            closed.to_string()
        } else {
            format!("failed to read frame header: {e}")
        }
    })?;
    decode_frame_header(&buf).map_err(|e| e.to_string())
}

fn read_or<R: Read>(r: &mut R, buf: &mut [u8], message: &'static str) -> Result<(), String> {
    r.read_exact(buf).map_err(|_| message.to_string())
}

fn discard<R: Read>(r: &mut R, mut bytes: u32) -> io::Result<()> {
    let mut buf = [0; 4096];
    while bytes != 0 {
        let n = bytes.min(buf.len() as u32) as usize;
        r.read_exact(&mut buf[..n])?;
        bytes -= n as u32;
    }
    Ok(())
}

fn read_message<R: Read>(
    r: &mut R,
    bytes: u32,
    read_error: &'static str,
    discard_error: &'static str,
) -> Result<Vec<u8>, String> {
    let copied = bytes.min(255) as usize;
    let mut message = vec![0; copied];
    read_or(r, &mut message, read_error)?;
    discard(r, bytes - copied as u32).map_err(|_| discard_error.to_string())?;
    Ok(message)
}

fn read_begin<R: Read>(r: &mut R) -> Result<(SnapshotBegin, Vec<u8>), String> {
    let header = read_header(r, "distributed worker closed snapshot connection")?;
    if header.typ != MSG_SNAPSHOT_BEGIN || header.bytes < SNAPSHOT_BEGIN_FIXED_BYTES as u32 {
        let _ = discard(r, header.bytes);
        return Err("distributed worker returned invalid snapshot frame".into());
    }

    let mut fixed = [0; SNAPSHOT_BEGIN_FIXED_BYTES];
    read_or(r, &mut fixed, "failed to read distributed snapshot header")?;
    let begin = SnapshotBegin::decode(&fixed)
        .map_err(|_| "invalid distributed snapshot response header".to_string())?;
    let mut body = header.bytes - SNAPSHOT_BEGIN_FIXED_BYTES as u32;
    let expected_token_bytes = u64::from(begin.token_count) * 4;
    if expected_token_bytes > u64::from(u32::MAX)
        || begin.token_bytes != expected_token_bytes as u32
        || begin.token_bytes > body
        || begin.message_bytes > body - begin.token_bytes
    {
        let _ = discard(r, body);
        return Err("invalid distributed snapshot response header".into());
    }

    discard(r, begin.token_bytes)
        .map_err(|_| "failed to discard distributed snapshot response tokens".to_string())?;
    body -= begin.token_bytes;
    let message = read_message(
        r,
        begin.message_bytes,
        "failed to read distributed snapshot response message",
        "failed to discard distributed snapshot response message",
    )?;
    body -= begin.message_bytes;
    discard(r, body).map_err(|_| {
        "failed to discard trailing distributed snapshot response bytes".to_string()
    })?;
    Ok((begin, message))
}

fn read_done<R: Read>(r: &mut R, request_id: u64) -> Result<(), String> {
    let header = read_header(r, "distributed worker closed before snapshot completion")?;
    if header.typ != MSG_SNAPSHOT_DONE || header.bytes < SNAPSHOT_DONE_FIXED_BYTES as u32 {
        let _ = discard(r, header.bytes);
        return Err("distributed worker returned invalid snapshot completion frame".into());
    }

    let mut fixed = [0; SNAPSHOT_DONE_FIXED_BYTES];
    read_or(
        r,
        &mut fixed,
        "failed to read distributed snapshot completion",
    )?;
    let done = SnapshotDone::decode(&fixed)
        .map_err(|_| "invalid distributed snapshot completion message".to_string())?;
    let mut body = header.bytes - SNAPSHOT_DONE_FIXED_BYTES as u32;
    if done.message_bytes > body {
        let _ = discard(r, body);
        return Err("invalid distributed snapshot completion message".into());
    }
    let message = read_message(
        r,
        done.message_bytes,
        "failed to read distributed snapshot completion message",
        "failed to discard distributed snapshot completion message",
    )?;
    body -= done.message_bytes;
    discard(r, body).map_err(|_| {
        "failed to discard trailing distributed snapshot completion bytes".to_string()
    })?;

    if u64_from_halves(done.request_hi, done.request_lo) != request_id {
        return Err("distributed snapshot completion request mismatch".into());
    }
    if done.status != 0 {
        let message = c_message(&message);
        return Err(if message.is_empty() {
            "distributed worker failed snapshot request".into()
        } else {
            message
        });
    }
    Ok(())
}

/// Ask a connected worker for one KV shard and stream it into `payload`.
pub fn coordinator_save_snapshot<S: Read + Write, W: Write>(
    stream: &mut S,
    meta: SnapshotMeta,
    tokens: &[i32],
    payload: &mut W,
) -> Result<u64, String> {
    let token_count = token_count(tokens)?;
    let (session_hi, session_lo) = u64_to_halves(meta.session_id);
    let (request_hi, request_lo) = u64_to_halves(meta.request_id);
    let (token_hash_hi, token_hash_lo) = u64_to_halves(meta.token_hash);
    let request = SnapshotReq {
        model_id: meta.model_id,
        session_hi,
        session_lo,
        request_hi,
        request_lo,
        token_hash_hi,
        token_hash_lo,
        token_count,
        layer_start: meta.layer_start,
        layer_end: meta.layer_end,
    };
    write_frame(stream, MSG_SNAPSHOT_SAVE_REQ, &request.encode())
        .map_err(|_| "failed to request distributed KV shard".to_string())?;

    let (begin, message) = read_begin(stream)?;
    if begin.status != 0 {
        let message = c_message(&message);
        return Err(if message.is_empty() {
            "distributed worker refused KV snapshot".into()
        } else {
            message
        });
    }

    let payload_bytes = u64_from_halves(begin.payload_hi, begin.payload_lo);
    if begin.model_id != meta.model_id
        || u64_from_halves(begin.session_hi, begin.session_lo) != meta.session_id
        || u64_from_halves(begin.request_hi, begin.request_lo) != meta.request_id
        || u64_from_halves(begin.token_hash_hi, begin.token_hash_lo) != meta.token_hash
        || (begin.token_count != 0 && begin.token_count != token_count)
        || begin.layer_start != meta.layer_start
        || begin.layer_end != meta.layer_end
    {
        return Err("distributed KV shard metadata mismatch".into());
    }

    let mut buf = Vec::new();
    if payload_bytes != 0 {
        buf.try_reserve_exact(SNAPSHOT_CHUNK_BYTES)
            .map_err(|_| "out of memory receiving distributed KV shard".to_string())?;
        buf.resize(SNAPSHOT_CHUNK_BYTES, 0);
    }
    let mut received = 0u64;
    while received < payload_bytes {
        let header = read_header(stream, "distributed worker closed while sending KV shard")?;
        if header.typ != MSG_SNAPSHOT_CHUNK || header.bytes < SNAPSHOT_CHUNK_FIXED_BYTES as u32 {
            let _ = discard(stream, header.bytes);
            return Err("expected distributed KV shard chunk".into());
        }

        let mut fixed = [0; SNAPSHOT_CHUNK_FIXED_BYTES];
        read_or(
            stream,
            &mut fixed,
            "failed to read distributed KV shard chunk header",
        )?;
        let chunk = SnapshotChunk::decode(&fixed)
            .map_err(|_| "invalid distributed KV shard chunk".to_string())?;
        let chunk_bytes = header.bytes - SNAPSHOT_CHUNK_FIXED_BYTES as u32;
        if u64_from_halves(chunk.request_hi, chunk.request_lo) != meta.request_id
            || chunk.chunk_bytes != chunk_bytes
            || chunk_bytes as usize > SNAPSHOT_CHUNK_BYTES
            || u64::from(chunk_bytes) > payload_bytes - received
        {
            let _ = discard(stream, chunk_bytes);
            return Err("invalid distributed KV shard chunk".into());
        }
        let chunk_bytes = chunk_bytes as usize;
        read_or(
            stream,
            &mut buf[..chunk_bytes],
            "failed to read distributed KV shard chunk",
        )?;
        payload
            .write_all(&buf[..chunk_bytes])
            .map_err(|_| "failed to write distributed KV shard".to_string())?;
        received += chunk_bytes as u64;
    }
    read_done(stream, meta.request_id)?;
    Ok(payload_bytes)
}

/// Stream one caller-owned KV shard payload to a connected worker.
pub fn coordinator_load_snapshot<S: Read + Write, R: Read>(
    stream: &mut S,
    meta: SnapshotMeta,
    tokens: &[i32],
    payload: &mut R,
    payload_bytes: u64,
) -> Result<(), String> {
    let token_count = token_count(tokens)?;
    let token_bytes = token_count
        .checked_mul(4)
        .ok_or_else(|| "invalid distributed snapshot token length".to_string())?;
    let (session_hi, session_lo) = u64_to_halves(meta.session_id);
    let (request_hi, request_lo) = u64_to_halves(meta.request_id);
    let (token_hash_hi, token_hash_lo) = u64_to_halves(meta.token_hash);
    let (payload_hi, payload_lo) = u64_to_halves(payload_bytes);
    let begin = SnapshotBegin {
        model_id: meta.model_id,
        session_hi,
        session_lo,
        request_hi,
        request_lo,
        token_hash_hi,
        token_hash_lo,
        token_count,
        layer_start: meta.layer_start,
        layer_end: meta.layer_end,
        payload_hi,
        payload_lo,
        status: 0,
        token_bytes,
        message_bytes: 0,
    };
    let frame_bytes = SNAPSHOT_BEGIN_FIXED_BYTES as u64 + u64::from(token_bytes);
    if frame_bytes > u64::from(u32::MAX) {
        return Err("distributed snapshot begin frame is too large".into());
    }
    let send_begin = || "failed to send distributed KV shard restore request".to_string();
    stream
        .write_all(&encode_frame_header(
            MSG_SNAPSHOT_LOAD_BEGIN,
            frame_bytes as u32,
        ))
        .map_err(|_| send_begin())?;
    stream
        .write_all(&begin.encode())
        .map_err(|_| send_begin())?;
    for &token in tokens {
        stream
            .write_all(&(token as u32).to_be_bytes())
            .map_err(|_| send_begin())?;
    }

    let mut buf = Vec::new();
    if payload_bytes != 0 {
        buf.try_reserve_exact(SNAPSHOT_CHUNK_BYTES)
            .map_err(|_| "failed to send distributed KV shard restore payload".to_string())?;
        buf.resize(SNAPSHOT_CHUNK_BYTES, 0);
    }
    let mut remaining = payload_bytes;
    while remaining != 0 {
        let n = remaining.min(SNAPSHOT_CHUNK_BYTES as u64) as usize;
        read_or(
            payload,
            &mut buf[..n],
            "failed to send distributed KV shard restore payload",
        )?;
        let (request_hi, request_lo) = u64_to_halves(meta.request_id);
        let chunk = SnapshotChunk {
            request_hi,
            request_lo,
            chunk_bytes: n as u32,
        };
        let frame_bytes = SNAPSHOT_CHUNK_FIXED_BYTES as u32 + n as u32;
        let send_chunk = || "failed to send distributed KV shard restore payload".to_string();
        stream
            .write_all(&encode_frame_header(MSG_SNAPSHOT_CHUNK, frame_bytes))
            .map_err(|_| send_chunk())?;
        stream
            .write_all(&chunk.encode())
            .map_err(|_| send_chunk())?;
        stream.write_all(&buf[..n]).map_err(|_| send_chunk())?;
        remaining -= n as u64;
    }
    read_done(stream, meta.request_id)
}
