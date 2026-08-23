//! Explicit integer codecs for the DS4D wire. Do not `#[repr(C)]` these
//! records onto the socket — C uses `htonl` per `uint32_t` field.

use std::fmt;

pub const MAGIC: u32 = 0x4453_3444; /* DS4D */
pub const MSG_HELLO: u32 = 1;
pub const MSG_ERROR: u32 = 2;
pub const MSG_WORK: u32 = 3;
pub const MSG_RESULT: u32 = 4;
pub const MSG_SNAPSHOT_SAVE_REQ: u32 = 5;
pub const MSG_SNAPSHOT_BEGIN: u32 = 6;
pub const MSG_SNAPSHOT_CHUNK: u32 = 7;
pub const MSG_SNAPSHOT_DONE: u32 = 8;
pub const MSG_SNAPSHOT_LOAD_BEGIN: u32 = 9;
pub const MAX_MODEL_NAME: u32 = 127;
pub const WORK_F_INPUT_HC: u32 = 0x0000_0001;
pub const WORK_F_OUTPUT_LOGITS: u32 = 0x0000_0002;
pub const WORK_F_RESET_SESSION: u32 = 0x0000_0004;
pub const WORK_F_ACK_ONLY: u32 = 0x0000_0008;
pub const WORK_F_VALID_MASK: u32 =
    WORK_F_INPUT_HC | WORK_F_OUTPUT_LOGITS | WORK_F_RESET_SESSION | WORK_F_ACK_ONLY;
pub const RESULT_ACK: u32 = 0;
pub const RESULT_HIDDEN_STATE: u32 = 1;
pub const RESULT_LOGITS: u32 = 2;
pub const ROUTE_F_OUTPUT_LOGITS: u32 = 0x0000_0001;
pub const ROUTE_RETURN_UPSTREAM: u32 = 1;
pub const FRAME_HEADER_BYTES: usize = 12;
pub const HELLO_FIXED_BYTES: usize = 40;
pub const WORK_FIXED_BYTES: usize = 80;
pub const ROUTE_FIXED_BYTES: usize = 20;
pub const ROUTE_RETURN_FIXED_BYTES: usize = 12;
pub const RESULT_FIXED_BYTES: usize = 40;
pub const TELEMETRY_FIXED_BYTES: usize = 40;
pub const SNAPSHOT_REQ_FIXED_BYTES: usize = 40;
pub const SNAPSHOT_BEGIN_FIXED_BYTES: usize = 60;
pub const SNAPSHOT_CHUNK_FIXED_BYTES: usize = 12;
pub const SNAPSHOT_DONE_FIXED_BYTES: usize = 16;
pub const NI_MAXHOST: usize = 1025;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    Truncated,
    BadMagic(u32),
    Invalid(&'static str),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::Truncated => write!(f, "truncated distributed frame"),
            CodecError::BadMagic(m) => write!(f, "bad frame magic 0x{m:08x}"),
            CodecError::Invalid(s) => write!(f, "{s}"),
        }
    }
}

pub fn put_u32_be(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

pub fn get_u32_be(buf: &[u8], off: &mut usize) -> Result<u32, CodecError> {
    let end = off.checked_add(4).ok_or(CodecError::Truncated)?;
    let s = buf.get(*off..end).ok_or(CodecError::Truncated)?;
    *off = end;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

pub fn u64_to_halves(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, v as u32)
}

pub fn u64_from_halves(hi: u32, lo: u32) -> u64 {
    ((hi as u64) << 32) | u64::from(lo)
}

pub fn bytes_have_nul(p: &[u8]) -> bool {
    p.contains(&0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub typ: u32,
    pub bytes: u32,
}

pub fn encode_frame_header(typ: u32, bytes: u32) -> [u8; FRAME_HEADER_BYTES] {
    let mut out = [0u8; FRAME_HEADER_BYTES];
    out[0..4].copy_from_slice(&MAGIC.to_be_bytes());
    out[4..8].copy_from_slice(&typ.to_be_bytes());
    out[8..12].copy_from_slice(&bytes.to_be_bytes());
    out
}

pub fn decode_frame_header(buf: &[u8]) -> Result<FrameHeader, CodecError> {
    if buf.len() < FRAME_HEADER_BYTES {
        return Err(CodecError::Truncated);
    }
    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != MAGIC {
        return Err(CodecError::BadMagic(magic));
    }
    Ok(FrameHeader {
        typ: u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]),
        bytes: u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]),
    })
}

macro_rules! rec {
    ($name:ident, $n:expr, [ $($field:ident),+ $(,)? ]) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct $name {
            $(pub $field: u32,)+
        }
        impl $name {
            pub fn encode(&self) -> [u8; $n] {
                let mut out = Vec::with_capacity($n);
                $(put_u32_be(&mut out, self.$field);)+
                let mut arr = [0u8; $n];
                arr.copy_from_slice(&out);
                arr
            }
            pub fn decode(buf: &[u8]) -> Result<Self, CodecError> {
                if buf.len() < $n {
                    return Err(CodecError::Truncated);
                }
                let mut off = 0;
                Ok(Self {
                    $($field: get_u32_be(buf, &mut off)?,)+
                })
            }
        }
    };
}

rec!(Hello, HELLO_FIXED_BYTES, [
    model_id, quant_bits, layer_start, layer_end, has_output, has_hidden,
    ctx_size, n_layers, listen_port, model_name_len
]);
rec!(Work, WORK_FIXED_BYTES, [
    model_id, session_hi, session_lo, request_hi, request_lo,
    prefix_hash_hi, prefix_hash_lo, result_hash_hi, result_hash_lo,
    pos0, n_tokens, layer_start, layer_end, flags, token_bytes,
    input_hc_bytes, input_hc_bits, route_count, route_index, route_bytes
]);
rec!(Route, ROUTE_FIXED_BYTES, [host_len, port, layer_start, layer_end, flags]);
rec!(RouteReturn, ROUTE_RETURN_FIXED_BYTES, [kind, host_len, port]);
rec!(ResultHdr, RESULT_FIXED_BYTES, [
    request_hi, request_lo, result_hash_hi, result_hash_lo, status,
    result_kind, telemetry_count, telemetry_bytes, payload_bytes, payload_bits
]);
rec!(Telemetry, TELEMETRY_FIXED_BYTES, [
    layer_start, layer_end, route_index, pos0, n_tokens, eval_usec,
    downstream_wait_usec, forward_send_usec, input_bytes, output_bytes
]);
rec!(SnapshotReq, SNAPSHOT_REQ_FIXED_BYTES, [
    model_id, session_hi, session_lo, request_hi, request_lo,
    token_hash_hi, token_hash_lo, token_count, layer_start, layer_end
]);
rec!(SnapshotBegin, SNAPSHOT_BEGIN_FIXED_BYTES, [
    model_id, session_hi, session_lo, request_hi, request_lo,
    token_hash_hi, token_hash_lo, token_count, layer_start, layer_end,
    payload_hi, payload_lo, status, token_bytes, message_bytes
]);
rec!(SnapshotChunk, SNAPSHOT_CHUNK_FIXED_BYTES, [request_hi, request_lo, chunk_bytes]);
rec!(SnapshotDone, SNAPSHOT_DONE_FIXED_BYTES, [request_hi, request_lo, status, message_bytes]);

pub fn encode_hello_payload(h: &Hello, model_name: &str) -> Result<Vec<u8>, CodecError> {
    let name = model_name.as_bytes();
    let n = name.len().min(MAX_MODEL_NAME as usize);
    if bytes_have_nul(&name[..n]) {
        return Err(CodecError::Invalid("HELLO model family contains NUL bytes"));
    }
    let mut rec = *h;
    rec.model_name_len = n as u32;
    let mut out = rec.encode().to_vec();
    out.extend_from_slice(&name[..n]);
    Ok(out)
}

pub fn decode_hello_payload(buf: &[u8]) -> Result<(Hello, String), CodecError> {
    let h = Hello::decode(buf)?;
    let name = buf.get(HELLO_FIXED_BYTES..).ok_or(CodecError::Truncated)?;
    if h.model_name_len as usize != name.len() || h.model_name_len > MAX_MODEL_NAME {
        return Err(CodecError::Invalid("invalid HELLO model name length"));
    }
    if bytes_have_nul(name) {
        return Err(CodecError::Invalid("HELLO model family contains NUL bytes"));
    }
    Ok((h, String::from_utf8_lossy(name).into_owned()))
}

/// Tokens on the work frame are `htonl((uint32_t)token)`.
pub fn encode_tokens_be(tokens: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        put_u32_be(&mut out, t as u32);
    }
    out
}

pub fn decode_tokens_be(buf: &[u8]) -> Result<Vec<i32>, CodecError> {
    if buf.len() % 4 != 0 {
        return Err(CodecError::Truncated);
    }
    let mut off = 0;
    let mut out = Vec::with_capacity(buf.len() / 4);
    while off < buf.len() {
        out.push(get_u32_be(buf, &mut off)? as i32);
    }
    Ok(out)
}

pub fn encode_error_frame(msg: &str) -> Vec<u8> {
    let bytes = msg.as_bytes();
    let mut out = encode_frame_header(MSG_ERROR, bytes.len() as u32).to_vec();
    out.extend_from_slice(bytes);
    out
}
