//! Local HTTP against a std thread. Status codes are ignored, matching C.

use ds4_web::{http_local, http_request, ws_handshake};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

#[test]
fn http_local_returns_body_after_headers() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port() as i32;
    let server = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut buf = [0u8; 2048];
        let n = s.read(&mut buf).unwrap_or(0);
        let got = String::from_utf8_lossy(&buf[..n]);
        assert!(got.contains("GET /json/version HTTP/1.1"));
        assert!(got.contains(&format!("Host: 127.0.0.1:{port}")));
        s.write_all(b"HTTP/1.1 500 UNUSED\r\nX-A: 1\r\n\r\nBODY-OK")
            .unwrap();
    });
    let body = http_local("GET", port, "/json/version").expect("http_local");
    assert_eq!(body, "BODY-OK");
    assert_eq!(
        http_request("GET", port, "/json/version").matches("Connection: close").count(),
        1
    );
    server.join().unwrap();
}

#[test]
fn http_local_empty_and_malformed() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port() as i32;
    let server = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut buf = [0u8; 512];
        let _ = s.read(&mut buf);
        s.write_all(b"not-http").unwrap();
    });
    let err = http_local("GET", port, "/x").unwrap_err();
    assert_eq!(err, "malformed HTTP response");
    server.join().unwrap();
}

#[test]
fn ws_handshake_bytes() {
    let req = ws_handshake("/devtools/page/1", "127.0.0.1", 9, "abcd");
    assert!(req.starts_with("GET /devtools/page/1 HTTP/1.1\r\n"));
    assert!(req.contains("Sec-WebSocket-Version: 13\r\n"));
    assert!(req.contains("Upgrade: websocket\r\n"));
}
