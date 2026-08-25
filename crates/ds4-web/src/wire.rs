//! Request builders that match `ds4_web.c` snprintf / concat order.

use crate::encode::{json_quote, url_encode};

pub fn http_request(method: &str, port: i32, path: &str) -> String {
    format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
}

pub fn ws_handshake(path: &str, host: &str, port: i32, key: &str) -> String {
    format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: {key}\r\n\
         Sec-WebSocket-Version: 13\r\n\r\n"
    )
}

pub fn cdp_request(id: i32, method: &str, params: Option<&str>) -> String {
    let mut out = format!("{{\"id\":{id},\"method\":");
    out.push_str(&json_quote(method));
    if let Some(params) = params {
        if !params.is_empty() {
            out.push_str(",\"params\":");
            out.push_str(params);
        }
    }
    out.push('}');
    out
}

pub fn eval_params(expr: &str) -> String {
    let mut out = String::from("{\"expression\":");
    out.push_str(&json_quote(expr));
    out.push_str(",\"returnByValue\":true,\"awaitPromise\":true,\"includeCommandLineAPI\":true}");
    out
}

pub fn navigate_params(url: &str) -> String {
    let mut out = String::from("{\"url\":");
    out.push_str(&json_quote(url));
    out.push('}');
    out
}

pub fn create_target_params(url: &str) -> String {
    let mut out = String::from("{\"url\":");
    out.push_str(&json_quote(url));
    out.push_str(",\"background\":true,\"newWindow\":false}");
    out
}

pub fn google_search_url(query: &str) -> String {
    let mut out = String::from("https://www.google.com/search?q=");
    out.push_str(&url_encode(query));
    out
}

pub fn close_path(target_id: &str) -> String {
    let mut out = String::from("/json/close/");
    out.push_str(&url_encode(target_id));
    out
}

pub fn page_ws_url(port: i32, target_id: &str) -> String {
    format!("ws://127.0.0.1:{port}/devtools/page/{target_id}")
}

pub fn ws_text_frame(text: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let len = text.len();
    let mut hdr = Vec::new();
    hdr.push(0x81);
    if len < 126 {
        hdr.push(0x80 | (len as u8));
    } else if len <= 0xffff {
        hdr.push(0x80 | 126);
        hdr.push((len >> 8) as u8);
        hdr.push(len as u8);
    } else {
        hdr.push(0x80 | 127);
        for i in (0..8).rev() {
            hdr.push((len >> (i * 8)) as u8);
        }
    }
    hdr.extend_from_slice(&mask);
    let mut out = hdr;
    for (i, &b) in text.iter().enumerate() {
        out.push(b ^ mask[i & 3]);
    }
    out
}

pub fn ws_pong_frame(payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let len = payload.len().min(125);
    let mut hdr = vec![0x8a, 0x80 | (len as u8)];
    hdr.extend_from_slice(&mask);
    for i in 0..len {
        hdr.push(payload[i] ^ mask[i & 3]);
    }
    hdr
}
