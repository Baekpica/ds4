//! Route blob: `[Route + host]*` then `RouteReturn + host`. No struct memcpy.

use crate::codec::{
    bytes_have_nul, CodecError, Route, RouteReturn, NI_MAXHOST, ROUTE_FIXED_BYTES,
    ROUTE_F_OUTPUT_LOGITS, ROUTE_RETURN_FIXED_BYTES, ROUTE_RETURN_UPSTREAM,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteEntry {
    pub host: String,
    pub port: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnTarget {
    pub kind: u32,
    pub host: String,
    pub port: u32,
}

pub fn encode_route_blob(
    entries: &[RouteEntry],
    ret: &ReturnTarget,
) -> Result<Vec<u8>, CodecError> {
    let mut out = Vec::new();
    for e in entries {
        let host = e.host.as_bytes();
        if host.is_empty() || host.len() >= NI_MAXHOST || bytes_have_nul(host) {
            return Err(CodecError::Invalid("invalid route host length"));
        }
        let rec = Route {
            host_len: host.len() as u32,
            port: e.port,
            layer_start: e.layer_start,
            layer_end: e.layer_end,
            flags: e.flags,
        };
        out.extend_from_slice(&rec.encode());
        out.extend_from_slice(host);
    }
    let rhost = ret.host.as_bytes();
    if rhost.len() >= NI_MAXHOST || bytes_have_nul(rhost) {
        return Err(CodecError::Invalid(
            "invalid route final destination host length",
        ));
    }
    let rec = RouteReturn {
        kind: ret.kind,
        host_len: rhost.len() as u32,
        port: ret.port,
    };
    out.extend_from_slice(&rec.encode());
    out.extend_from_slice(rhost);
    Ok(out)
}

pub fn decode_route_blob(
    blob: &[u8],
    route_count: u32,
) -> Result<(Vec<RouteEntry>, ReturnTarget), CodecError> {
    let mut remaining = blob;
    let mut entries = Vec::new();
    for _ in 0..route_count {
        if remaining.len() < ROUTE_FIXED_BYTES {
            return Err(CodecError::Invalid("truncated route entry"));
        }
        let fixed = Route::decode(remaining)?;
        remaining = &remaining[ROUTE_FIXED_BYTES..];
        if fixed.host_len == 0
            || fixed.host_len as usize >= NI_MAXHOST
            || fixed.host_len as usize > remaining.len()
        {
            return Err(CodecError::Invalid("invalid route host length"));
        }
        let host = &remaining[..fixed.host_len as usize];
        if bytes_have_nul(host) {
            return Err(CodecError::Invalid("route host contains NUL bytes"));
        }
        entries.push(RouteEntry {
            host: String::from_utf8_lossy(host).into_owned(),
            port: fixed.port,
            layer_start: fixed.layer_start,
            layer_end: fixed.layer_end,
            flags: fixed.flags,
        });
        remaining = &remaining[fixed.host_len as usize..];
    }
    if remaining.len() < ROUTE_RETURN_FIXED_BYTES {
        return Err(CodecError::Invalid(
            "route payload missing final destination",
        ));
    }
    let ret = RouteReturn::decode(remaining)?;
    remaining = &remaining[ROUTE_RETURN_FIXED_BYTES..];
    if ret.host_len as usize >= NI_MAXHOST || ret.host_len as usize > remaining.len() {
        return Err(CodecError::Invalid(
            "invalid route final destination host length",
        ));
    }
    let host = &remaining[..ret.host_len as usize];
    if bytes_have_nul(host) {
        return Err(CodecError::Invalid(
            "route final destination host contains NUL bytes",
        ));
    }
    remaining = &remaining[ret.host_len as usize..];
    if !remaining.is_empty() {
        return Err(CodecError::Invalid("route payload has trailing bytes"));
    }
    Ok((
        entries,
        ReturnTarget {
            kind: ret.kind,
            host: String::from_utf8_lossy(host).into_owned(),
            port: ret.port,
        },
    ))
}

pub fn validate_route_blob(blob: &[u8], route_count: u32, n_layers: u32) -> Result<(), CodecError> {
    if route_count == 0 {
        return if blob.is_empty() {
            Ok(())
        } else {
            Err(CodecError::Invalid(
                "route payload has entries without a route count",
            ))
        };
    }
    let (entries, ret) = decode_route_blob(blob, route_count)?;
    let mut prev_end = u32::MAX;
    for (i, e) in entries.iter().enumerate() {
        if e.port == 0 || e.port > 65535 {
            return Err(CodecError::Invalid("invalid route port"));
        }
        if e.layer_start >= n_layers || e.layer_end >= n_layers || e.layer_end < e.layer_start {
            return Err(CodecError::Invalid("invalid route layer range"));
        }
        if (e.flags & !ROUTE_F_OUTPUT_LOGITS) != 0 {
            return Err(CodecError::Invalid("invalid route flags"));
        }
        if (e.flags & ROUTE_F_OUTPUT_LOGITS) != 0 && e.layer_end + 1 != n_layers {
            return Err(CodecError::Invalid("route logits require final layer"));
        }
        if i != 0 && e.layer_start != prev_end + 1 {
            return Err(CodecError::Invalid("route layer ranges are not contiguous"));
        }
        prev_end = e.layer_end;
    }
    if ret.kind != ROUTE_RETURN_UPSTREAM {
        return Err(CodecError::Invalid("unsupported route final destination"));
    }
    if !ret.host.is_empty() || ret.port != 0 {
        return Err(CodecError::Invalid(
            "invalid upstream route final destination",
        ));
    }
    Ok(())
}
