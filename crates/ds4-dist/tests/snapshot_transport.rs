use std::io::{self, Cursor, Read, Write};

use ds4_dist::{
    coordinator_load_snapshot, coordinator_save_snapshot, decode_snapshot_chunk_body,
    decode_snapshot_load_begin_body, encode_frame_header, encode_snapshot_begin_body,
    encode_snapshot_chunk_body, encode_snapshot_done_body, read_frame, write_frame, SnapshotBegin,
    SnapshotMeta, SnapshotReq, MSG_SNAPSHOT_BEGIN, MSG_SNAPSHOT_CHUNK, MSG_SNAPSHOT_DONE,
    MSG_SNAPSHOT_LOAD_BEGIN, MSG_SNAPSHOT_SAVE_REQ, SNAPSHOT_CHUNK_BYTES,
};

#[derive(Default)]
struct ScriptedStream {
    read: Cursor<Vec<u8>>,
    written: Vec<u8>,
}

impl ScriptedStream {
    fn new(read: Vec<u8>) -> Self {
        Self {
            read: Cursor::new(read),
            written: Vec::new(),
        }
    }
}

impl Read for ScriptedStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read.read(buf)
    }
}

impl Write for ScriptedStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.written.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct RejectLargeReads {
    read: Cursor<Vec<u8>>,
    written: Vec<u8>,
    largest_read: usize,
}

impl RejectLargeReads {
    fn new(read: Vec<u8>) -> Self {
        Self {
            read: Cursor::new(read),
            written: Vec::new(),
            largest_read: 0,
        }
    }
}

impl Read for RejectLargeReads {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.largest_read = self.largest_read.max(buf.len());
        assert!(buf.len() <= SNAPSHOT_CHUNK_BYTES);
        self.read.read(buf)
    }
}

impl Write for RejectLargeReads {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.written.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn meta() -> SnapshotMeta {
    SnapshotMeta {
        model_id: 3,
        session_id: 0x11_2233,
        request_id: 0x44_5566,
        token_hash: 0x77_8899,
        layer_start: 8,
        layer_end: 12,
    }
}

fn begin(meta: SnapshotMeta, token_count: u32, payload_bytes: u64) -> SnapshotBegin {
    let (session_hi, session_lo) = ds4_dist::u64_to_halves(meta.session_id);
    let (request_hi, request_lo) = ds4_dist::u64_to_halves(meta.request_id);
    let (token_hash_hi, token_hash_lo) = ds4_dist::u64_to_halves(meta.token_hash);
    let (payload_hi, payload_lo) = ds4_dist::u64_to_halves(payload_bytes);
    SnapshotBegin {
        model_id: meta.model_id,
        session_hi,
        session_lo,
        request_hi,
        request_lo,
        token_hash_hi,
        token_hash_lo,
        token_count,
        layer_start: meta.layer_start,
        layer_end: meta.layer_end,
        payload_hi,
        payload_lo,
        status: 0,
        token_bytes: token_count * 4,
        message_bytes: 0,
    }
}

fn push_frame(out: &mut Vec<u8>, typ: u32, body: &[u8]) {
    write_frame(out, typ, body).unwrap();
}

#[test]
fn coordinator_save_follows_c_frame_order_and_lengths() {
    let meta = meta();
    let tokens = [7, 8];
    let mut payload = vec![0x5a; SNAPSHOT_CHUNK_BYTES + 3];
    payload[SNAPSHOT_CHUNK_BYTES] = 1;
    payload[SNAPSHOT_CHUNK_BYTES + 1] = 2;
    payload[SNAPSHOT_CHUNK_BYTES + 2] = 3;
    let mut scripted = Vec::new();

    let mut begin_body = encode_snapshot_begin_body(
        &begin(meta, tokens.len() as u32, payload.len() as u64),
        &tokens,
        b"",
    )
    .unwrap();
    begin_body.extend_from_slice(b"ignored trailing begin bytes");
    push_frame(&mut scripted, MSG_SNAPSHOT_BEGIN, &begin_body);
    push_frame(
        &mut scripted,
        MSG_SNAPSHOT_CHUNK,
        &encode_snapshot_chunk_body(meta.request_id, &payload[..SNAPSHOT_CHUNK_BYTES]).unwrap(),
    );
    push_frame(
        &mut scripted,
        MSG_SNAPSHOT_CHUNK,
        &encode_snapshot_chunk_body(meta.request_id, &payload[SNAPSHOT_CHUNK_BYTES..]).unwrap(),
    );
    let mut done = encode_snapshot_done_body(meta.request_id, 0, b"").unwrap();
    done.extend_from_slice(b"ignored trailing done bytes");
    push_frame(&mut scripted, MSG_SNAPSHOT_DONE, &done);

    let mut stream = ScriptedStream::new(scripted);
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut stream, meta, &tokens, &mut saved).unwrap(),
        payload.len() as u64
    );
    assert_eq!(saved, payload);

    let mut written = Cursor::new(stream.written);
    let (typ, body) = read_frame(&mut written).unwrap();
    assert_eq!(typ, MSG_SNAPSHOT_SAVE_REQ);
    let request = SnapshotReq::decode(&body).unwrap();
    assert_eq!(request.model_id, meta.model_id);
    assert_eq!(
        ds4_dist::u64_from_halves(request.session_hi, request.session_lo),
        meta.session_id
    );
    assert_eq!(
        ds4_dist::u64_from_halves(request.request_hi, request.request_lo),
        meta.request_id
    );
    assert_eq!(request.token_count, tokens.len() as u32);
    assert!(read_frame(&mut written).is_err());
}

#[test]
fn coordinator_save_rejects_status_metadata_and_frame_sequence_errors() {
    let meta = meta();
    let tokens = [7, 8];

    let mut refused = begin(meta, 0, 0);
    refused.status = 1;
    refused.message_bytes = 12;
    let body = encode_snapshot_begin_body(&refused, &[], b"refused\0tail").unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "refused"
    );

    let message = vec![b'x'; 300];
    refused.message_bytes = message.len() as u32;
    let body = encode_snapshot_begin_body(&refused, &[], &message).unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "x".repeat(255)
    );

    let mut wrong = begin(meta, tokens.len() as u32, 0);
    wrong.layer_end += 1;
    let body = encode_snapshot_begin_body(&wrong, &tokens, b"").unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "distributed KV shard metadata mismatch"
    );

    let body = encode_snapshot_begin_body(&begin(meta, 0, 4), &[], b"").unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    push_frame(
        &mut frames,
        MSG_SNAPSHOT_CHUNK,
        &encode_snapshot_chunk_body(meta.request_id, b"ab").unwrap(),
    );
    push_frame(
        &mut frames,
        MSG_SNAPSHOT_DONE,
        &encode_snapshot_done_body(meta.request_id, 0, b"").unwrap(),
    );
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "expected distributed KV shard chunk"
    );

    let body = encode_snapshot_begin_body(&begin(meta, 0, 2), &[], b"").unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    push_frame(
        &mut frames,
        MSG_SNAPSHOT_CHUNK,
        &encode_snapshot_chunk_body(meta.request_id, b"abc").unwrap(),
    );
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "invalid distributed KV shard chunk"
    );

    let body = encode_snapshot_begin_body(&begin(meta, 0, 1), &[], b"").unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    push_frame(
        &mut frames,
        MSG_SNAPSHOT_CHUNK,
        &encode_snapshot_chunk_body(meta.request_id + 1, b"a").unwrap(),
    );
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "invalid distributed KV shard chunk"
    );

    let body = encode_snapshot_begin_body(&begin(meta, 0, 1), &[], b"").unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    for _ in 0..2 {
        push_frame(
            &mut frames,
            MSG_SNAPSHOT_CHUNK,
            &encode_snapshot_chunk_body(meta.request_id, b"a").unwrap(),
        );
    }
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "distributed worker returned invalid snapshot completion frame"
    );

    let body = encode_snapshot_begin_body(&begin(meta, 0, 0), &[], b"").unwrap();
    let mut frames = Vec::new();
    push_frame(&mut frames, MSG_SNAPSHOT_BEGIN, &body);
    push_frame(
        &mut frames,
        MSG_SNAPSHOT_DONE,
        &encode_snapshot_done_body(meta.request_id + 1, 0, b"").unwrap(),
    );
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(frames), meta, &tokens, &mut saved,)
            .unwrap_err(),
        "distributed snapshot completion request mismatch"
    );
}

#[test]
fn coordinator_save_bounds_peer_lengths_and_reports_partial_stages() {
    let meta = meta();

    let mut stream =
        RejectLargeReads::new(encode_frame_header(MSG_SNAPSHOT_BEGIN, u32::MAX).to_vec());
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut stream, meta, &[], &mut saved).unwrap_err(),
        "failed to read distributed snapshot header"
    );
    assert!(stream.largest_read <= 60);

    let mut partial_done = Vec::new();
    push_frame(
        &mut partial_done,
        MSG_SNAPSHOT_BEGIN,
        &encode_snapshot_begin_body(&begin(meta, 0, 0), &[], b"").unwrap(),
    );
    partial_done.extend_from_slice(&encode_frame_header(MSG_SNAPSHOT_DONE, 16));
    partial_done.extend_from_slice(&[0; 8]);
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(
            &mut ScriptedStream::new(partial_done),
            meta,
            &[],
            &mut saved,
        )
        .unwrap_err(),
        "failed to read distributed snapshot completion"
    );
}

#[test]
fn coordinator_load_chunks_payload_and_checks_completion() {
    let meta = meta();
    let tokens = vec![7, 8];
    let mut payload = vec![0x5a; SNAPSHOT_CHUNK_BYTES + 3];
    payload[SNAPSHOT_CHUNK_BYTES] = 1;
    payload[SNAPSHOT_CHUNK_BYTES + 1] = 2;
    payload[SNAPSHOT_CHUNK_BYTES + 2] = 3;

    let mut reply = Vec::new();
    let mut done = encode_snapshot_done_body(meta.request_id, 0, b"").unwrap();
    done.extend_from_slice(b"trailing");
    push_frame(&mut reply, MSG_SNAPSHOT_DONE, &done);
    let mut stream = ScriptedStream::new(reply);
    let payload_len = payload.len() as u64;
    coordinator_load_snapshot(
        &mut stream,
        meta,
        &tokens,
        &mut Cursor::new(&payload),
        payload_len,
    )
    .unwrap();

    let mut written = Cursor::new(stream.written);
    let (typ, body) = read_frame(&mut written).unwrap();
    assert_eq!(typ, MSG_SNAPSHOT_LOAD_BEGIN);
    let (header, got_tokens, message) = decode_snapshot_load_begin_body(&body).unwrap();
    assert_eq!(got_tokens, tokens);
    assert!(message.is_empty());
    assert_eq!(
        ds4_dist::u64_from_halves(header.payload_hi, header.payload_lo),
        payload.len() as u64
    );

    let (typ, body) = read_frame(&mut written).unwrap();
    assert_eq!(typ, MSG_SNAPSHOT_CHUNK);
    let (_, chunk) =
        decode_snapshot_chunk_body(&body, meta.request_id, payload.len() as u64).unwrap();
    assert_eq!(chunk.len(), SNAPSHOT_CHUNK_BYTES);
    let (typ, body) = read_frame(&mut written).unwrap();
    assert_eq!(typ, MSG_SNAPSHOT_CHUNK);
    let (_, chunk) = decode_snapshot_chunk_body(&body, meta.request_id, 3).unwrap();
    assert_eq!(chunk, [1, 2, 3]);
    assert!(read_frame(&mut written).is_err());

    let mut failed = Vec::new();
    push_frame(
        &mut failed,
        MSG_SNAPSHOT_DONE,
        &encode_snapshot_done_body(meta.request_id, 1, b"restore failed").unwrap(),
    );
    assert_eq!(
        coordinator_load_snapshot(
            &mut ScriptedStream::new(failed),
            meta,
            &tokens,
            &mut Cursor::new(b"payload"),
            7,
        )
        .unwrap_err(),
        "restore failed"
    );
}

#[test]
fn coordinator_zero_payload_does_not_require_or_send_chunks() {
    let meta = meta();

    let mut save_reply = Vec::new();
    push_frame(
        &mut save_reply,
        MSG_SNAPSHOT_BEGIN,
        &encode_snapshot_begin_body(&begin(meta, 0, 0), &[], b"").unwrap(),
    );
    push_frame(
        &mut save_reply,
        MSG_SNAPSHOT_DONE,
        &encode_snapshot_done_body(meta.request_id, 0, b"").unwrap(),
    );
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut ScriptedStream::new(save_reply), meta, &[], &mut saved,)
            .unwrap(),
        0
    );
    assert!(saved.is_empty());

    let mut reply = Vec::new();
    push_frame(
        &mut reply,
        MSG_SNAPSHOT_DONE,
        &encode_snapshot_done_body(meta.request_id, 0, b"").unwrap(),
    );
    let mut stream = ScriptedStream::new(reply);
    let mut source = Cursor::new(b"not part of the declared payload");
    coordinator_load_snapshot(&mut stream, meta, &[], &mut source, 0).unwrap();
    assert_eq!(source.position(), 0);

    let mut written = Cursor::new(stream.written);
    let (typ, _) = read_frame(&mut written).unwrap();
    assert_eq!(typ, MSG_SNAPSHOT_LOAD_BEGIN);
    assert!(read_frame(&mut written).is_err());
}
