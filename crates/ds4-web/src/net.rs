//! Blocking TCP / HTTP / WebSocket matching `ds4_web.c` poll + read loops.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::encode::{base64, random_bytes};
use crate::wire::{http_request, ws_handshake, ws_pong_frame, ws_text_frame};
use crate::{CDP_TIMEOUT_MS, CONNECT_TIMEOUT_MS, MAX_RESULT_BYTES};

pub fn tcp_connect(host: &str, port: i32, timeout_ms: u64) -> Result<TcpStream, String> {
    let timeout = Duration::from_millis(timeout_ms);
    let addrs = match (host, port as u16).to_socket_addrs() {
        Ok(a) => a,
        Err(e) => return Err(format!("getaddrinfo {host}: {e}")),
    };
    let mut last = format!("connect {host}:{port} failed: connection refused");
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => last = format!("connect {host}:{port} failed: {e}"),
        }
    }
    Err(last)
}

fn write_all(stream: &mut TcpStream, buf: &[u8]) -> Result<(), String> {
    stream.write_all(buf).map_err(|e| e.to_string())
}

fn read_some(stream: &mut TcpStream, buf: &mut [u8], timeout_ms: u64) -> Result<usize, String> {
    stream
        .set_read_timeout(Some(Duration::from_millis(timeout_ms)))
        .map_err(|e| e.to_string())?;
    match stream.read(buf) {
        Ok(n) => Ok(n),
        Err(e) if e.kind() == std::io::ErrorKind::TimedOut || e.kind() == std::io::ErrorKind::WouldBlock => {
            Ok(0)
        }
        Err(e) => Err(e.to_string()),
    }
}

/// GET/POST-style request to 127.0.0.1. Status code is ignored; body is
/// everything after the first `\r\n\r\n`, matching C `web_http_request`.
pub fn http_local(method: &str, port: i32, path: &str) -> Result<String, String> {
    let mut fd = tcp_connect("127.0.0.1", port, CONNECT_TIMEOUT_MS)?;
    let req = http_request(method, port, path);
    if write_all(&mut fd, req.as_bytes()).is_err() {
        return Err(format!("write HTTP request failed: {}", std::io::Error::last_os_error()));
    }
    let mut resp = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = match read_some(&mut fd, &mut tmp, CONNECT_TIMEOUT_MS) {
            Ok(n) => n,
            Err(_) => return Err(format!("read HTTP response failed: {}", std::io::Error::last_os_error())),
        };
        if n == 0 {
            break;
        }
        resp.extend_from_slice(&tmp[..n]);
    }
    if resp.is_empty() {
        return Err("empty HTTP response".into());
    }
    let Some(pos) = find_header_end(&resp) else {
        return Err("malformed HTTP response".into());
    };
    Ok(String::from_utf8_lossy(&resp[pos + 4..]).into_owned())
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

pub struct Ws {
    stream: TcpStream,
    pub next_id: i32,
}

impl Ws {
    pub fn connect(ws_url: &str) -> Result<Self, String> {
        if !ws_url.starts_with("ws://") {
            return Err(format!("unsupported websocket URL: {ws_url}"));
        }
        let rest = &ws_url[5..];
        let Some(slash) = rest.find('/') else {
            return Err("malformed websocket URL".into());
        };
        let mut hostport = rest[..slash].to_string();
        if hostport.len() > 255 {
            hostport.truncate(255);
        }
        let (host, port) = if let Some(colon) = hostport.rfind(':') {
            let port = hostport[colon + 1..].parse::<i32>().unwrap_or(0);
            (hostport[..colon].to_string(), port)
        } else {
            (hostport, 80)
        };
        let path = &rest[slash..];
        let mut fd = tcp_connect(&host, port, CONNECT_TIMEOUT_MS)?;
        let rnd = random_bytes(16);
        let key = base64(&rnd);
        let req = ws_handshake(path, &host, port, &key);
        if write_all(&mut fd, req.as_bytes()).is_err() {
            return Err("websocket handshake write failed".into());
        }
        let mut resp = Vec::new();
        let mut tmp = [0u8; 1024];
        while find_header_end(&resp).is_none() {
            let n = match read_some(&mut fd, &mut tmp, CONNECT_TIMEOUT_MS) {
                Ok(n) if n > 0 => n,
                _ => return Err("websocket handshake read failed".into()),
            };
            resp.extend_from_slice(&tmp[..n]);
            if resp.len() > 8192 {
                break;
            }
        }
        let text = String::from_utf8_lossy(&resp);
        if !text.contains(" 101 ") {
            return Err("websocket handshake rejected".into());
        }
        Ok(Self {
            stream: fd,
            next_id: 1,
        })
    }

    pub fn send_text(&mut self, text: &str) -> Result<(), String> {
        let mut mask = [0u8; 4];
        mask.copy_from_slice(&random_bytes(4));
        let frame = ws_text_frame(text.as_bytes(), mask);
        write_all(&mut self.stream, &frame)
            .map_err(|_| format!("websocket write failed: {}", std::io::Error::last_os_error()))
    }

    fn send_pong(&mut self, payload: &[u8]) -> Result<(), String> {
        let mut mask = [0u8; 4];
        mask.copy_from_slice(&random_bytes(4));
        let frame = ws_pong_frame(payload, mask);
        write_all(&mut self.stream, &frame)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), String> {
        let mut off = 0;
        while off < buf.len() {
            let n = match read_some(&mut self.stream, &mut buf[off..], CDP_TIMEOUT_MS) {
                Ok(n) if n > 0 => n,
                Ok(_) => return Err("websocket read timeout".into()),
                Err(_) => return Err("websocket frame read failed".into()),
            };
            off += n;
        }
        Ok(())
    }

    pub fn read_message(&mut self) -> Result<String, String> {
        let mut msg = Vec::new();
        loop {
            let mut h = [0u8; 2];
            if self.read_exact(&mut h).is_err() {
                return Err("websocket read timeout".into());
            }
            let fin = (h[0] & 0x80) != 0;
            let opcode = h[0] & 0x0f;
            let masked = (h[1] & 0x80) != 0;
            let mut len = u64::from(h[1] & 0x7f);
            if len == 126 {
                let mut x = [0u8; 2];
                self.read_exact(&mut x).map_err(|_| "websocket frame read failed")?;
                len = (u64::from(x[0]) << 8) | u64::from(x[1]);
            } else if len == 127 {
                let mut x = [0u8; 8];
                self.read_exact(&mut x).map_err(|_| "websocket frame read failed")?;
                len = 0;
                for b in x {
                    len = (len << 8) | u64::from(b);
                }
            }
            let mut mask = [0u8; 4];
            if masked {
                self.read_exact(&mut mask).map_err(|_| "websocket frame read failed")?;
            }
            if len > MAX_RESULT_BYTES as u64 * 4 {
                return Err("websocket message too large".into());
            }
            let mut payload = vec![0u8; len as usize];
            if len > 0 {
                self.read_exact(&mut payload).map_err(|_| "websocket frame read failed")?;
            }
            if masked {
                for i in 0..payload.len() {
                    payload[i] ^= mask[i & 3];
                }
            }
            match opcode {
                0x8 => return Err("websocket closed".into()),
                0x9 => {
                    let _ = self.send_pong(&payload);
                }
                0x1 | 0x0 => {
                    msg.extend_from_slice(&payload);
                    if fin {
                        return Ok(String::from_utf8_lossy(&msg).into_owned());
                    }
                }
                _ => {}
            }
        }
    }
}
