//! Persistent forwarder relay and telemetry prepend.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use ds4_dist::{
    decode_result_body, encode_result_frame, forward_work_blocking, local_work_telemetry,
    ok_result_hdr, prepend_telemetry, read_frame, result_request_id, usec_since, Forwarder,
    ForwarderPool, PendingRequest, ResultBody, RouteEntry, Telemetry, Work, WorkBody,
    ERR_NEXT_CLOSED, FRAME_HEADER_BYTES, MSG_RESULT, RESULT_ACK,
};

fn pending(id: u64) -> PendingRequest {
    PendingRequest {
        request_id: id,
        telemetry: Telemetry {
            layer_start: 2,
            layer_end: 3,
            route_index: 1,
            pos0: 0,
            n_tokens: 1,
            eval_usec: 11,
            downstream_wait_usec: 0,
            forward_send_usec: 0,
            input_bytes: 8,
            output_bytes: 4,
        },
        downstream_t0: 0.0,
    }
}

fn result_frame(request_id: u64, payload: &[u8]) -> Vec<u8> {
    let mut hdr = ok_result_hdr(request_id, 0, RESULT_ACK, 0);
    hdr.payload_bytes = payload.len() as u32;
    encode_result_frame(&ResultBody {
        hdr,
        telemetry: Vec::new(),
        payload: payload.to_vec(),
    })
    .unwrap()
}

fn tune(s: &std::net::TcpStream) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(2)));
}

#[test]
fn local_work_telemetry_matches_c_fields() {
    let work = Work {
        layer_start: 2,
        layer_end: 3,
        route_index: 1,
        pos0: 4,
        n_tokens: 2,
        token_bytes: 8,
        input_hc_bytes: 16,
        ..Work::default()
    };
    let tel = local_work_telemetry(&work, 12, 32);
    assert_eq!(tel.layer_start, 2);
    assert_eq!(tel.layer_end, 3);
    assert_eq!(tel.route_index, 1);
    assert_eq!(tel.pos0, 4);
    assert_eq!(tel.n_tokens, 2);
    assert_eq!(tel.eval_usec, 12);
    assert_eq!(tel.downstream_wait_usec, 0);
    assert_eq!(tel.forward_send_usec, 0);
    assert_eq!(tel.input_bytes, 24);
    assert_eq!(tel.output_bytes, 32);
}

#[test]
fn usec_since_matches_c_rounding() {
    assert_eq!(usec_since(1.0, 1.0), 0);
    assert_eq!(usec_since(1.0, 0.5), 0);
    assert_eq!(usec_since(0.0, 0.000_001), 1);
    assert_eq!(usec_since(0.0, 1.5), 1_500_000);
}

#[test]
fn prepend_telemetry_appends_local_record() {
    let frame = result_frame(9, b"ok");
    let body = &frame[FRAME_HEADER_BYTES..];
    let local = pending(9).telemetry;
    let out = prepend_telemetry(body, local).unwrap();
    let (typ, payload) = {
        let mut cur = std::io::Cursor::new(out);
        read_frame(&mut cur).unwrap()
    };
    assert_eq!(typ, MSG_RESULT);
    let decoded = decode_result_body(&payload).unwrap();
    assert_eq!(result_request_id(&decoded.hdr), 9);
    assert_eq!(decoded.payload, b"ok");
    assert_eq!(decoded.telemetry.len(), 1);
    assert_eq!(decoded.telemetry[0], local);
}

#[test]
fn forwarder_relays_result_and_prepends_telemetry() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    let err_tx = tx.clone();
    let mut forwarder = Forwarder::connect(
        "127.0.0.1",
        addr.port() as u32,
        4,
        move |frame| {
            let _ = tx.send(Ok(frame));
        },
        move |id, msg| {
            let _ = err_tx.send(Err((id, msg)));
        },
    )
    .unwrap();

    let next = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        tune(&stream);
        let (_typ, body) = read_frame(&mut stream).unwrap();
        assert_eq!(body, b"WORK");
        stream.write_all(&result_frame(7, b"logits")).unwrap();
    });

    forwarder
        .send_work(pending(7), &{
            let mut frame = ds4_dist::encode_frame_header(ds4_dist::MSG_WORK, 4).to_vec();
            frame.extend_from_slice(b"WORK");
            frame
        })
        .unwrap();
    let frame = rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .expect("relayed RESULT");
    let mut cur = std::io::Cursor::new(frame);
    let (typ, payload) = read_frame(&mut cur).unwrap();
    assert_eq!(typ, MSG_RESULT);
    let decoded = decode_result_body(&payload).unwrap();
    assert_eq!(decoded.payload, b"logits");
    assert_eq!(decoded.telemetry.len(), 1);
    assert_eq!(decoded.telemetry[0].layer_start, 2);
    assert_eq!(decoded.telemetry[0].eval_usec, 11);
    forwarder.shutdown();
    next.join().unwrap();
}

fn work_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame =
        ds4_dist::encode_frame_header(ds4_dist::MSG_WORK, payload.len() as u32).to_vec();
    frame.extend_from_slice(payload);
    frame
}

#[test]
fn forwarder_reports_closed_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    let mut forwarder = Forwarder::connect(
        "127.0.0.1",
        addr.port() as u32,
        2,
        move |_| {},
        move |id, msg| {
            let _ = tx.send((id, msg));
        },
    )
    .unwrap();
    let (stream, _) = listener.accept().unwrap();
    tune(&stream);
    forwarder.send_work(pending(3), &work_frame(&[1])).unwrap();
    drop(stream);
    let (id, msg) = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(id, 3);
    assert_eq!(msg, ERR_NEXT_CLOSED);
    forwarder.shutdown();
}

#[test]
fn blocking_forward_prepends_local_telemetry() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let next = RouteEntry {
        host: "127.0.0.1".into(),
        port: addr.port() as u32,
        layer_start: 2,
        layer_end: 3,
        flags: 0,
    };
    let work = Work {
        layer_start: 1,
        layer_end: 1,
        route_index: 0,
        pos0: 4,
        n_tokens: 1,
        token_bytes: 4,
        request_lo: 7,
        ..Work::default()
    };
    let body = WorkBody {
        work,
        tokens: vec![1],
        input_hc: vec![1.0, 2.0],
        route_blob: Vec::new(),
    };
    let local = local_work_telemetry(&work, 11, 8);
    let next_thread = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        tune(&stream);
        let (_typ, _body) = read_frame(&mut stream).unwrap();
        stream.write_all(&result_frame(7, b"logits")).unwrap();
    });
    let mut upstream = Vec::new();
    forward_work_blocking(&next, &body, &mut upstream, local).unwrap();
    let mut cur = std::io::Cursor::new(upstream);
    let (typ, payload) = read_frame(&mut cur).unwrap();
    assert_eq!(typ, MSG_RESULT);
    let decoded = decode_result_body(&payload).unwrap();
    assert_eq!(decoded.payload, b"logits");
    assert_eq!(decoded.telemetry.len(), 1);
    assert_eq!(decoded.telemetry[0].eval_usec, 11);
    assert_eq!(decoded.telemetry[0].layer_start, 1);
    assert_eq!(decoded.telemetry[0].input_bytes, 4);
    assert_eq!(decoded.telemetry[0].output_bytes, 8);
    next_thread.join().unwrap();
}

#[test]
fn pool_reuses_forwarder_and_prepends_telemetry() {
    let next_l = TcpListener::bind("127.0.0.1:0").unwrap();
    let up_l = TcpListener::bind("127.0.0.1:0").unwrap();
    let next_addr = next_l.local_addr().unwrap();
    let up_addr = up_l.local_addr().unwrap();
    let upstream = TcpStream::connect(up_addr).unwrap();
    tune(&upstream);
    let (mut up_peer, _) = up_l.accept().unwrap();
    tune(&up_peer);
    let mut pool = ForwarderPool::new();
    pool.bind(Arc::new(Mutex::new(upstream)));
    let next = RouteEntry {
        host: "127.0.0.1".into(),
        port: next_addr.port() as u32,
        layer_start: 2,
        layer_end: 3,
        flags: 0,
    };
    let work = Work {
        layer_start: 1,
        layer_end: 1,
        n_tokens: 1,
        token_bytes: 4,
        request_lo: 7,
        ..Work::default()
    };
    let body = WorkBody {
        work,
        tokens: vec![1],
        input_hc: vec![1.0, 2.0],
        route_blob: Vec::new(),
    };
    let next_thread = thread::spawn(move || {
        let (mut stream, _) = next_l.accept().unwrap();
        tune(&stream);
        let (_typ, _body) = read_frame(&mut stream).unwrap();
        stream.write_all(&result_frame(7, b"logits")).unwrap();
        let (_typ, _body) = read_frame(&mut stream).unwrap();
        stream.write_all(&result_frame(8, b"again")).unwrap();
    });
    pool.forward(&next, &body, local_work_telemetry(&work, 11, 8))
        .unwrap();
    assert_eq!(pool.len(), 1);
    let mut body2 = body;
    body2.work.request_lo = 8;
    pool.forward(&next, &body2, local_work_telemetry(&work, 12, 8))
        .unwrap();
    assert_eq!(pool.len(), 1);
    let (typ, payload) = read_frame(&mut up_peer).unwrap();
    assert_eq!(typ, MSG_RESULT);
    let decoded = decode_result_body(&payload).unwrap();
    assert_eq!(decoded.payload, b"logits");
    assert_eq!(decoded.telemetry.len(), 1);
    assert_eq!(decoded.telemetry[0].eval_usec, 11);
    let (typ, payload) = read_frame(&mut up_peer).unwrap();
    assert_eq!(typ, MSG_RESULT);
    let decoded = decode_result_body(&payload).unwrap();
    assert_eq!(decoded.payload, b"again");
    assert_eq!(decoded.telemetry[0].eval_usec, 12);
    pool.shutdown();
    next_thread.join().unwrap();
}
