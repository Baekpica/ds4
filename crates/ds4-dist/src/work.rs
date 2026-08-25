//! WORK / RESULT frame bodies. Tokens are BE u32; activations use the packers.

use crate::activation::{bits_or_default, decode_activation, encode_activation};
use crate::codec::{
    decode_tokens_be, encode_frame_header, encode_tokens_be, u64_from_halves, u64_to_halves,
    CodecError, ResultHdr, Telemetry, Work, MSG_RESULT, MSG_WORK, RESULT_FIXED_BYTES,
    TELEMETRY_FIXED_BYTES, WORK_FIXED_BYTES,
};

#[derive(Debug, Clone)]
pub struct WorkBody {
    pub work: Work,
    pub tokens: Vec<i32>,
    pub input_hc: Vec<f32>,
    pub route_blob: Vec<u8>,
}

pub fn encode_work_body(body: &WorkBody) -> Result<Vec<u8>, CodecError> {
    if body.tokens.is_empty() {
        return Err(CodecError::Invalid("WORK frame has no tokens"));
    }
    let mut work = body.work;
    work.n_tokens = body.tokens.len() as u32;
    work.token_bytes = work.n_tokens * 4;
    let hc_bits = bits_or_default(work.input_hc_bits);
    let hc_wire = if body.input_hc.is_empty() {
        Vec::new()
    } else {
        encode_activation(&body.input_hc, hc_bits)
            .ok_or(CodecError::Invalid("invalid distributed activation width"))?
    };
    work.input_hc_bytes = hc_wire.len() as u32;
    work.input_hc_bits = hc_bits;
    work.route_bytes = body.route_blob.len() as u32;
    let mut out = work.encode().to_vec();
    out.extend_from_slice(&encode_tokens_be(&body.tokens));
    out.extend_from_slice(&hc_wire);
    out.extend_from_slice(&body.route_blob);
    Ok(out)
}

pub fn encode_work_frame(body: &WorkBody) -> Result<Vec<u8>, CodecError> {
    let payload = encode_work_body(body)?;
    let mut frame = encode_frame_header(MSG_WORK, payload.len() as u32).to_vec();
    frame.extend_from_slice(&payload);
    Ok(frame)
}

pub fn decode_work_body(buf: &[u8]) -> Result<WorkBody, CodecError> {
    if buf.len() < WORK_FIXED_BYTES {
        return Err(CodecError::Invalid("truncated distributed WORK frame"));
    }
    let work = Work::decode(buf)?;
    let rest = &buf[WORK_FIXED_BYTES..];
    let expected =
        work.token_bytes as usize + work.input_hc_bytes as usize + work.route_bytes as usize;
    if work.token_bytes as usize != work.n_tokens as usize * 4 || rest.len() != expected {
        return Err(CodecError::Invalid(
            "invalid distributed WORK payload sizes",
        ));
    }
    let (tok, rest) = rest.split_at(work.token_bytes as usize);
    let tokens = decode_tokens_be(tok)?;
    let (hc, rest) = rest.split_at(work.input_hc_bytes as usize);
    let input_hc = if hc.is_empty() {
        Vec::new()
    } else {
        decode_activation(hc, work.input_hc_bits)
            .ok_or(CodecError::Invalid("invalid distributed activation width"))?
    };
    Ok(WorkBody {
        work,
        tokens,
        input_hc,
        route_blob: rest.to_vec(),
    })
}

#[derive(Debug, Clone)]
pub struct ResultBody {
    pub hdr: ResultHdr,
    pub telemetry: Vec<Telemetry>,
    pub payload: Vec<u8>,
}

pub fn encode_result_body(body: &ResultBody) -> Result<Vec<u8>, CodecError> {
    let mut hdr = body.hdr;
    hdr.telemetry_count = body.telemetry.len() as u32;
    hdr.telemetry_bytes = hdr.telemetry_count * TELEMETRY_FIXED_BYTES as u32;
    hdr.payload_bytes = body.payload.len() as u32;
    let mut out = hdr.encode().to_vec();
    for t in &body.telemetry {
        out.extend_from_slice(&t.encode());
    }
    out.extend_from_slice(&body.payload);
    Ok(out)
}

pub fn encode_result_frame(body: &ResultBody) -> Result<Vec<u8>, CodecError> {
    let payload = encode_result_body(body)?;
    let mut frame = encode_frame_header(MSG_RESULT, payload.len() as u32).to_vec();
    frame.extend_from_slice(&payload);
    Ok(frame)
}

pub fn decode_result_body(buf: &[u8]) -> Result<ResultBody, CodecError> {
    if buf.len() < RESULT_FIXED_BYTES {
        return Err(CodecError::Invalid(
            "distributed worker returned invalid frame",
        ));
    }
    let hdr = ResultHdr::decode(buf)?;
    let rest = &buf[RESULT_FIXED_BYTES..];
    if hdr.telemetry_bytes as usize % TELEMETRY_FIXED_BYTES != 0
        || hdr.telemetry_count as usize != hdr.telemetry_bytes as usize / TELEMETRY_FIXED_BYTES
        || hdr.telemetry_bytes as usize > rest.len()
        || hdr.payload_bytes as usize != rest.len() - hdr.telemetry_bytes as usize
    {
        return Err(CodecError::Invalid(
            "distributed result telemetry metadata mismatch",
        ));
    }
    let (tel, rest) = rest.split_at(hdr.telemetry_bytes as usize);
    let mut telemetry = Vec::new();
    for chunk in tel.chunks_exact(TELEMETRY_FIXED_BYTES) {
        telemetry.push(Telemetry::decode(chunk)?);
    }
    if rest.len() != hdr.payload_bytes as usize {
        return Err(CodecError::Truncated);
    }
    Ok(ResultBody {
        hdr,
        telemetry,
        payload: rest.to_vec(),
    })
}

pub fn result_request_id(hdr: &ResultHdr) -> u64 {
    u64_from_halves(hdr.request_hi, hdr.request_lo)
}

pub fn result_hash(hdr: &ResultHdr) -> u64 {
    u64_from_halves(hdr.result_hash_hi, hdr.result_hash_lo)
}

pub fn encode_logits_payload(logits: &[f32]) -> Vec<u8> {
    encode_activation(logits, 32).unwrap_or_default()
}

pub fn decode_logits_payload(buf: &[u8]) -> Result<Vec<f32>, CodecError> {
    decode_activation(buf, 32).ok_or(CodecError::Invalid("invalid logits payload"))
}

pub fn work_with_ids(mut work: Work, session: u64, request: u64, prefix: u64, result: u64) -> Work {
    let (shi, slo) = u64_to_halves(session);
    let (rhi, rlo) = u64_to_halves(request);
    let (phi, plo) = u64_to_halves(prefix);
    let (ehi, elo) = u64_to_halves(result);
    work.session_hi = shi;
    work.session_lo = slo;
    work.request_hi = rhi;
    work.request_lo = rlo;
    work.prefix_hash_hi = phi;
    work.prefix_hash_lo = plo;
    work.result_hash_hi = ehi;
    work.result_hash_lo = elo;
    work
}

pub fn ok_result_hdr(request: u64, hash: u64, kind: u32, bits: u32) -> ResultHdr {
    let (rhi, rlo) = u64_to_halves(request);
    let (hhi, hlo) = u64_to_halves(hash);
    ResultHdr {
        request_hi: rhi,
        request_lo: rlo,
        result_hash_hi: hhi,
        result_hash_lo: hlo,
        status: 0,
        result_kind: kind,
        telemetry_count: 0,
        telemetry_bytes: 0,
        payload_bytes: 0,
        payload_bits: bits,
    }
}

pub fn error_result_frame(request: u64, msg: &str) -> Vec<u8> {
    let (rhi, rlo) = u64_to_halves(request);
    let payload = msg.as_bytes().to_vec();
    let body = ResultBody {
        hdr: ResultHdr {
            request_hi: rhi,
            request_lo: rlo,
            result_hash_hi: 0,
            result_hash_lo: 0,
            status: 1,
            result_kind: 0,
            telemetry_count: 0,
            telemetry_bytes: 0,
            payload_bytes: payload.len() as u32,
            payload_bits: 0,
        },
        telemetry: Vec::new(),
        payload,
    };
    encode_result_frame(&body).unwrap_or_else(|_| crate::codec::encode_error_frame(msg))
}
