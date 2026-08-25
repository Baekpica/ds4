//! End-to-end google_search / visit_page against a blocking mock CDP.
//! Chrome is never spawned: `/json/version` is already alive.

use ds4_web::{
    json_get_string, json_quote, Config, Web, CLICK_GOOGLE_CONSENT, EXTRACT_PAGE, EXTRACT_SEARCH,
    PAGE_PROBE, READY_STATE, SCROLL_DYNAMIC,
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

struct MockCdp {
    port: i32,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockCdp {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port() as i32;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = thread::spawn(move || loop {
            if flag.load(Ordering::SeqCst) {
                break;
            }
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).ok();
                    thread::spawn(move || handle_conn(stream, port));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        });
        Self {
            port,
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for MockCdp {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(("127.0.0.1", self.port as u16));
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn handle_conn(mut stream: TcpStream, port: i32) {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        match stream.read(&mut tmp) {
            Ok(0) => return,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => return,
        }
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..pos]).into_owned();
            if head.to_ascii_lowercase().contains("upgrade: websocket") {
                let _ = stream.write_all(b"HTTP/1.1 101 Switching Protocols\r\n\r\n");
                serve_cdp(&mut stream);
                return;
            }
            if head.starts_with("GET /json/version") {
                let body = format!(
                    "{{\"webSocketDebuggerUrl\":\"ws://127.0.0.1:{port}/devtools/browser/B1\"}}"
                );
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
                return;
            }
            if head.starts_with("GET /json/close/") {
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\n\r\nOK");
                return;
            }
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\n\r\n");
            return;
        }
        if buf.len() > 8192 {
            return;
        }
    }
}

fn serve_cdp(stream: &mut TcpStream) {
    loop {
        let msg = match read_ws_text(stream) {
            Ok(m) => m,
            Err(_) => return,
        };
        let id = extract_id(&msg);
        let method = json_get_string(&msg, "method").unwrap_or_default();
        let body = if method == "Runtime.evaluate" {
            let expr = json_get_string(&msg, "expression").unwrap_or_default();
            let value = eval_reply(&expr);
            format!(
                "{{\"id\":{id},\"result\":{{\"result\":{{\"type\":\"string\",\"value\":{}}}}}}}",
                json_quote(&value)
            )
        } else if method == "Target.createTarget" {
            format!("{{\"id\":{id},\"result\":{{\"targetId\":\"PAGE1\"}}}}")
        } else {
            format!("{{\"id\":{id},\"result\":{{}}}}")
        };
        if write_ws_text(stream, &body).is_err() {
            return;
        }
    }
}

fn extract_id(msg: &str) -> i32 {
    let Some(p) = msg.find("\"id\"") else {
        return 1;
    };
    let rest = &msg[p + 4..];
    let Some(colon) = rest.find(':') else {
        return 1;
    };
    let s = rest[colon + 1..].trim_start_matches([' ', '\t']);
    let end = s
        .find(|c: char| c != '-' && !c.is_ascii_digit())
        .unwrap_or(s.len());
    s[..end].parse().unwrap_or(1)
}

fn eval_reply(expr: &str) -> String {
    if expr == READY_STATE {
        "complete".into()
    } else if expr == PAGE_PROBE {
        "https://www.google.com/search?q=rust\ncomplete\n42".into()
    } else if expr == CLICK_GOOGLE_CONSENT {
        String::new()
    } else if expr == EXTRACT_SEARCH {
        "# Google search results\n\nOK-SEARCH".into()
    } else if expr == EXTRACT_PAGE {
        "# Example\n\nOK-PAGE".into()
    } else if expr == SCROLL_DYNAMIC {
        "scroll skipped hooks=0 text=42".into()
    } else {
        "ok".into()
    }
}

fn read_ws_text(stream: &mut TcpStream) -> Result<String, ()> {
    let mut h = [0u8; 2];
    stream.read_exact(&mut h).map_err(|_| ())?;
    let mut len = (h[1] & 0x7f) as usize;
    let masked = (h[1] & 0x80) != 0;
    if len == 126 {
        let mut x = [0u8; 2];
        stream.read_exact(&mut x).map_err(|_| ())?;
        len = ((x[0] as usize) << 8) | x[1] as usize;
    } else if len == 127 {
        let mut x = [0u8; 8];
        stream.read_exact(&mut x).map_err(|_| ())?;
        len = 0;
        for b in x {
            len = (len << 8) | b as usize;
        }
    }
    let mut mask = [0u8; 4];
    if masked {
        stream.read_exact(&mut mask).map_err(|_| ())?;
    }
    let mut payload = vec![0u8; len];
    if len > 0 {
        stream.read_exact(&mut payload).map_err(|_| ())?;
    }
    if masked {
        for i in 0..payload.len() {
            payload[i] ^= mask[i & 3];
        }
    }
    Ok(String::from_utf8_lossy(&payload).into_owned())
}

fn write_ws_text(stream: &mut TcpStream, text: &str) -> Result<(), ()> {
    let bytes = text.as_bytes();
    let mut frame = Vec::new();
    frame.push(0x81);
    if bytes.len() < 126 {
        frame.push(bytes.len() as u8);
    } else if bytes.len() <= 0xffff {
        frame.push(126);
        frame.push((bytes.len() >> 8) as u8);
        frame.push(bytes.len() as u8);
    } else {
        return Err(());
    }
    frame.extend_from_slice(bytes);
    stream.write_all(&frame).map_err(|_| ())
}

fn web_on(port: i32) -> Web {
    Web::new(Config {
        home_dir: Some(std::env::temp_dir().join("ds4-web-test-home")),
        port,
        confirm: Some(Box::new(|_| Ok(true))),
        log: None,
    })
}

#[test]
fn google_search_empty_query() {
    let mut web = web_on(1);
    assert_eq!(
        web.google_search("").unwrap_err(),
        "google_search requires query"
    );
}

#[test]
fn visit_page_empty_url() {
    let mut web = web_on(1);
    assert_eq!(web.visit_page("").unwrap_err(), "visit_page requires url");
}

#[test]
fn confirm_required_when_cdp_down() {
    let mut web = Web::new(Config {
        home_dir: Some(std::env::temp_dir().join("ds4-web-test-home")),
        port: 1,
        confirm: None,
        log: None,
    });
    assert_eq!(
        web.google_search("q").unwrap_err(),
        "starting a visible Chrome browser requires interactive approval"
    );
}

#[test]
fn confirm_denied() {
    let mut web = Web::new(Config {
        home_dir: Some(std::env::temp_dir().join("ds4-web-test-home")),
        port: 1,
        confirm: Some(Box::new(|msg| {
            assert_eq!(
                msg,
                "The web tool wants to start a visible Chrome browser. Allow? (y/n) "
            );
            Ok(false)
        })),
        log: None,
    });
    assert_eq!(
        web.google_search("q").unwrap_err(),
        "user denied Chrome browser start"
    );
}

#[test]
fn google_search_via_mock_cdp() {
    let mock = MockCdp::start();
    let mut web = web_on(mock.port);
    let out = web.google_search("rust host").expect("search");
    assert!(out.contains("OK-SEARCH"), "{out}");
}

#[test]
fn visit_page_via_mock_cdp() {
    let mock = MockCdp::start();
    let mut web = web_on(mock.port);
    let out = web.visit_page("https://example.com/").expect("visit");
    assert!(out.contains("OK-PAGE"), "{out}");
}
