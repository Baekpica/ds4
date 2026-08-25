use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

use crate::admit::enqueue_shed_error;
use crate::error::wire_http_error_bytes;
use crate::generate::ScriptedDecode;
use crate::route::WireSurface;
use crate::serve::{handle_client_inner, ServerConfig, ServerInner};
use crate::serve_serial_reclaim::serial_capacity_refuse_msg;

const C_SHUTTING_DOWN: &str = "server shutting down";

fn drive_chat(cfg: &ServerConfig, inner: &Mutex<ServerInner>, engine: bool) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    write!(
        client,
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    if engine {
        let mut decode = ScriptedDecode::from_pieces(&[]);
        handle_client_inner(cfg, inner, &mut server, Some(&mut decode), None);
    } else {
        handle_client_inner(cfg, inner, &mut server, None, None);
    }
    drop(server);
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    String::from_utf8(response).unwrap()
}

fn assert_c_503_message(response: &str, msg: &str) {
    assert!(
        response.starts_with("HTTP/1.1 503 Service Unavailable"),
        "{response}"
    );
    let expected = String::from_utf8(wire_http_error_bytes(
        WireSurface::OpenaiChat,
        503,
        msg,
        false,
        None,
    ))
    .unwrap();
    let expected_retry = String::from_utf8(wire_http_error_bytes(
        WireSurface::OpenaiChat,
        503,
        msg,
        false,
        Some(10),
    ))
    .unwrap();
    assert!(
        response == expected || response == expected_retry,
        "{response}"
    );
    assert!(
        !response.contains("generation remains on C ds4-server"),
        "{response}"
    );
    assert!(!response.contains("model not loaded"), "{response}");
    assert!(
        !response.contains("server could not track the client connection"),
        "{response}"
    );
    assert!(!response.contains("server is shutting down"), "{response}");
}

#[test]
fn missing_engine_503_reuses_c_shutting_down() {
    // Given: no DecodeIo (C never serves HTTP without a loaded engine)
    let cfg = ServerConfig::default();
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));

    // When: a chat request is admitted
    let response = drive_chat(&cfg, &inner, false);

    // Then: public 503 is the C Stopping body, not an invented twin
    assert_c_503_message(&response, C_SHUTTING_DOWN);
}

#[test]
fn have_engine_without_decode_io_503_reuses_c_shutting_down() {
    // Given: have_engine advertised but no DecodeIo was supplied
    let cfg = ServerConfig::test_cfg();
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));

    // When: a chat request is admitted
    let response = drive_chat(&cfg, &inner, false);

    // Then: no "generation remains on C ds4-server"
    assert_c_503_message(&response, C_SHUTTING_DOWN);
}

#[test]
fn stopping_enqueue_503_matches_c_client_main() {
    // Given: admit.stopping like C enqueue Stopping
    let cfg = ServerConfig::test_cfg();
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));
    inner.lock().unwrap().admit.stopping = true;

    // When: a chat request is admitted
    let response = drive_chat(&cfg, &inner, true);

    // Then: C "server shutting down"
    let (_, _, retry, msg) = enqueue_shed_error(crate::admit::EnqVerdict::Stopping).unwrap();
    assert_eq!(msg, C_SHUTTING_DOWN);
    let expected = String::from_utf8(wire_http_error_bytes(
        WireSurface::OpenaiChat,
        503,
        C_SHUTTING_DOWN,
        false,
        Some(retry),
    ))
    .unwrap();
    assert_eq!(response, expected);
}

#[test]
fn serial_capacity_refuse_stays_c_bytes() {
    // Given: C serial_session_ensure_fit
    // When: format the refuse
    let msg = serial_capacity_refuse_msg(8);

    // Then: existing C production string
    assert_eq!(
        msg,
        "Server is temporarily at capacity for a 8-token serial request \
         (no session graph fits beside the batch banks); retry shortly"
    );
}
