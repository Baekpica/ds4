//! C↔Rust DS4D wire bytes. No `#[repr(C)]` on the Rust side.

use ds4_dist::{
    decode_frame_header, decode_hello_payload, decode_snapshot_begin_body,
    decode_snapshot_chunk_body, decode_snapshot_done_body, decode_snapshot_load_begin_body,
    encode_activation, encode_frame_header, encode_hello_payload, encode_route_blob,
    encode_snapshot_begin_body, encode_snapshot_chunk_body, encode_snapshot_done_body,
    encode_tokens_be, f32_to_f16, f32_to_f8_e4m3, read_frame, token_hash_prefix,
    validate_route_blob, write_frame, Hello, ResultHdr, ReturnTarget, RouteEntry, SnapshotBegin,
    SnapshotChunk, SnapshotDone, SnapshotReq, Work, FRAME_HEADER_BYTES, HELLO_FIXED_BYTES, MAGIC,
    MSG_HELLO, MSG_WORK, RESULT_FIXED_BYTES, ROUTE_RETURN_UPSTREAM, SNAPSHOT_CHUNK_BYTES,
    WORK_FIXED_BYTES,
};
use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_DIST_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/dist_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/dist_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run dist_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn struct_sizes_match_c() {
    let line = c_out(&["sizes"]);
    assert!(line.contains("frame 12"), "{line}");
    assert!(line.contains("hello 40"), "{line}");
    assert!(line.contains("work 80"), "{line}");
    assert!(line.contains(&format!("work {WORK_FIXED_BYTES}")), "{line}");
    assert_eq!(FRAME_HEADER_BYTES, 12);
    assert_eq!(HELLO_FIXED_BYTES, 40);
    assert_eq!(RESULT_FIXED_BYTES, 40);
}

#[test]
fn frame_header_matches_c() {
    let rust = encode_frame_header(MSG_WORK, 80);
    assert_eq!(hex(&rust), c_out(&["frame", "3", "80"]));
    let h = decode_frame_header(&rust).unwrap();
    assert_eq!(h.typ, MSG_WORK);
    assert_eq!(h.bytes, 80);
    assert_eq!(u32::from_be_bytes(rust[0..4].try_into().unwrap()), MAGIC);
}

#[test]
fn hello_matches_c() {
    let rec = Hello {
        model_id: 3,
        quant_bits: 2,
        layer_start: 2,
        layer_end: 51,
        has_output: 1,
        has_hidden: 1,
        ctx_size: 8192,
        n_layers: 53,
        listen_port: 7000,
        model_name_len: 0,
    };
    let rust = encode_hello_payload(&rec, "motif-3").unwrap();
    assert_eq!(
        hex(&rust),
        c_out(&["hello", "3", "2", "2", "51", "1", "1", "8192", "53", "7000", "motif-3"])
    );
    let (back, name) = decode_hello_payload(&rust).unwrap();
    assert_eq!(name, "motif-3");
    assert_eq!(back.model_id, 3);
    assert_eq!(back.listen_port, 7000);
}

#[test]
fn work_and_tokens_match_c() {
    let w = Work {
        model_id: 3,
        session_hi: 0,
        session_lo: 9,
        request_hi: 0,
        request_lo: 11,
        prefix_hash_hi: 1,
        prefix_hash_lo: 2,
        result_hash_hi: 3,
        result_hash_lo: 4,
        pos0: 16,
        n_tokens: 3,
        layer_start: 2,
        layer_end: 10,
        flags: 1,
        token_bytes: 12,
        input_hc_bytes: 0,
        input_hc_bits: 32,
        route_count: 1,
        route_index: 0,
        route_bytes: 0,
    };
    assert_eq!(
        hex(&w.encode()),
        c_out(&[
            "work", "3", "0", "9", "0", "11", "1", "2", "3", "4", "16", "3", "2", "10", "1", "12",
            "0", "32", "1", "0", "0"
        ])
    );
    let tokens = [7i32, -1, 256];
    assert_eq!(
        hex(&encode_tokens_be(&tokens)),
        c_out(&["tokens", "7", "-1", "256"])
    );
}

#[test]
fn token_hash_matches_c() {
    assert_eq!(
        format!("{:016x}", token_hash_prefix(&[1, 2, 3, 99])),
        c_out(&["token-hash", "1", "2", "3", "99"])
    );
}

#[test]
fn route_blob_round_trip() {
    let entries = [RouteEntry {
        host: "10.0.0.2".into(),
        port: 7100,
        layer_start: 2,
        layer_end: 20,
        flags: 0,
    }];
    let ret = ReturnTarget {
        kind: ROUTE_RETURN_UPSTREAM,
        host: String::new(),
        port: 0,
    };
    let blob = encode_route_blob(&entries, &ret).unwrap();
    assert_eq!(
        hex(&blob[..20 + 8]),
        c_out(&["route", "10.0.0.2", "7100", "2", "20", "0"])
    );
    validate_route_blob(&blob, 1, 53).unwrap();
}

#[test]
fn result_matches_c() {
    let r = ResultHdr {
        request_hi: 0,
        request_lo: 11,
        result_hash_hi: 3,
        result_hash_lo: 4,
        status: 0,
        result_kind: 2,
        telemetry_count: 1,
        telemetry_bytes: 40,
        payload_bytes: 16,
        payload_bits: 32,
    };
    assert_eq!(
        hex(&r.encode()),
        c_out(&["result", "0", "11", "3", "4", "0", "2", "1", "40", "16", "32"])
    );
}

#[test]
fn f16_f8_match_c() {
    let samples = [
        0x00000000u32,
        0x3f800000,
        0xbf800000,
        0x3f000000,
        0x7f800000,
        0x7fc00000,
        0x00000001,
        0x3fc00000,
    ];
    for bits in samples {
        let hex_bits = format!("{bits:08x}");
        let f = f32::from_bits(bits);
        assert_eq!(
            hex(&f32_to_f16(f).to_le_bytes()),
            c_out(&["f16", &hex_bits]),
            "f16 {hex_bits}"
        );
        assert_eq!(
            hex(&[f32_to_f8_e4m3(f)]),
            c_out(&["f8", &hex_bits]),
            "f8 {hex_bits}"
        );
    }
}

#[test]
fn blocking_hello_round_trip() {
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let rec = Hello {
        model_id: 1,
        quant_bits: 4,
        layer_start: 0,
        layer_end: 11,
        has_output: 0,
        has_hidden: 1,
        ctx_size: 2048,
        n_layers: 48,
        listen_port: 9,
        model_name_len: 0,
    };
    let payload = encode_hello_payload(&rec, "solar").unwrap();
    let reply = payload.clone();
    let server = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let (typ, body) = read_frame(&mut s).unwrap();
        assert_eq!(typ, MSG_HELLO);
        let (h, name) = decode_hello_payload(&body).unwrap();
        assert_eq!(name, "solar");
        assert_eq!(h.n_layers, 48);
        write_frame(&mut s, MSG_HELLO, &reply).unwrap();
    });
    let mut client = TcpStream::connect(addr).unwrap();
    write_frame(&mut client, MSG_HELLO, &payload).unwrap();
    let (typ, body) = read_frame(&mut client).unwrap();
    assert_eq!(typ, MSG_HELLO);
    let (_, name) = decode_hello_payload(&body).unwrap();
    assert_eq!(name, "solar");
    server.join().unwrap();
}

#[test]
fn activation_f32_is_le_host_layout() {
    let src = [1.0f32, -2.5];
    let wire = encode_activation(&src, 32).unwrap();
    let mut expect = Vec::new();
    expect.extend_from_slice(&1.0f32.to_bits().to_le_bytes());
    expect.extend_from_slice(&(-2.5f32).to_bits().to_le_bytes());
    assert_eq!(wire, expect);
}

#[test]
fn snapshot_records_match_c() {
    let req = SnapshotReq {
        model_id: 3,
        session_hi: 1,
        session_lo: 2,
        request_hi: 3,
        request_lo: 4,
        token_hash_hi: 5,
        token_hash_lo: 6,
        token_count: 7,
        layer_start: 8,
        layer_end: 9,
    };
    assert_eq!(
        hex(&req.encode()),
        c_out(&[
            "snapshot-req",
            "3",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9"
        ])
    );

    let begin = SnapshotBegin {
        model_id: 3,
        session_hi: 1,
        session_lo: 2,
        request_hi: 3,
        request_lo: 4,
        token_hash_hi: 5,
        token_hash_lo: 6,
        token_count: 2,
        layer_start: 8,
        layer_end: 9,
        payload_hi: 10,
        payload_lo: 11,
        status: 0,
        token_bytes: 8,
        message_bytes: 0,
    };
    assert_eq!(
        hex(&begin.encode()),
        c_out(&[
            "snapshot-begin",
            "3",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "2",
            "8",
            "9",
            "10",
            "11",
            "0",
            "8",
            "0"
        ])
    );

    let chunk = SnapshotChunk {
        request_hi: 3,
        request_lo: 4,
        chunk_bytes: 5,
    };
    assert_eq!(
        hex(&chunk.encode()),
        c_out(&["snapshot-chunk", "3", "4", "5"])
    );

    let done = SnapshotDone {
        request_hi: 3,
        request_lo: 4,
        status: 1,
        message_bytes: 5,
    };
    assert_eq!(
        hex(&done.encode()),
        c_out(&["snapshot-done", "3", "4", "1", "5"])
    );
}

#[test]
fn snapshot_bodies_enforce_c_lengths_and_ids() {
    let begin = SnapshotBegin {
        model_id: 3,
        session_hi: 0,
        session_lo: 2,
        request_hi: 0,
        request_lo: 4,
        token_hash_hi: 0,
        token_hash_lo: 6,
        token_count: 2,
        layer_start: 8,
        layer_end: 9,
        payload_hi: 0,
        payload_lo: 5,
        status: 0,
        token_bytes: 8,
        message_bytes: 0,
    };
    let body = encode_snapshot_begin_body(&begin, &[7, 8], b"").unwrap();
    let (back, tokens, message) = decode_snapshot_load_begin_body(&body).unwrap();
    assert_eq!(back, begin);
    assert_eq!(tokens, [7, 8]);
    assert!(message.is_empty());

    let mut bad = begin;
    bad.token_bytes = 4;
    assert!(encode_snapshot_begin_body(&bad, &[7, 8], b"").is_err());
    assert!(decode_snapshot_begin_body(&body[..body.len() - 1]).is_err());

    let payload = b"kv-shard";
    let chunk = encode_snapshot_chunk_body(4, payload).unwrap();
    let (_, decoded) = decode_snapshot_chunk_body(&chunk, 4, payload.len() as u64).unwrap();
    assert_eq!(decoded, payload);
    assert!(decode_snapshot_chunk_body(&chunk, 5, payload.len() as u64).is_err());
    assert!(decode_snapshot_chunk_body(&chunk, 4, (payload.len() - 1) as u64).is_err());
    assert!(encode_snapshot_chunk_body(4, &vec![0; SNAPSHOT_CHUNK_BYTES + 1]).is_err());

    let done = encode_snapshot_done_body(4, 1, b"failed").unwrap();
    let (header, message) = decode_snapshot_done_body(&done, 4).unwrap();
    assert_eq!(header.status, 1);
    assert_eq!(message, b"failed");
    assert!(decode_snapshot_done_body(&done, 5).is_err());
}
