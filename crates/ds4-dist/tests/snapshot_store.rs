use std::io::Cursor;
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use ds4_dist::{
    coordinator_load_snapshot, coordinator_save_snapshot, token_hash_prefix, MemorySnapshotStore,
    SliceExec, SnapshotLoad, SnapshotMeta, SnapshotSave, SnapshotStore, WorkOutput, WorkRequest,
    Worker,
};

#[derive(Clone)]
struct MockExec {
    model_id: u32,
    n_layers: u32,
    vocab: u32,
    ctx_size: u32,
    layer_start: u32,
    layer_end: u32,
}

impl SliceExec for MockExec {
    fn model_id(&self) -> u32 {
        self.model_id
    }
    fn n_layers(&self) -> u32 {
        self.n_layers
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn ctx_size(&self) -> u32 {
        self.ctx_size
    }
    fn hidden_values(&self) -> u64 {
        2
    }
    fn has_output(&self) -> bool {
        true
    }
    fn layer_start(&self) -> u32 {
        self.layer_start
    }
    fn layer_end(&self) -> u32 {
        self.layer_end
    }
    fn eval(&mut self, _req: &WorkRequest) -> Result<WorkOutput, String> {
        Ok(WorkOutput {
            hidden: None,
            logits: Some(vec![0.0; 4]),
        })
    }
}

fn exec() -> MockExec {
    MockExec {
        model_id: 3,
        n_layers: 16,
        vocab: 32,
        ctx_size: 128,
        layer_start: 8,
        layer_end: 12,
    }
}

fn meta(tokens: &[i32]) -> SnapshotMeta {
    SnapshotMeta {
        model_id: 3,
        session_id: 0x11_2233,
        request_id: 0x44_5566,
        token_hash: token_hash_prefix(tokens),
        layer_start: 8,
        layer_end: 12,
    }
}

fn tune(s: &TcpStream) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
}

#[test]
fn store_save_reports_missing_session_with_c_text() {
    let mut store = MemorySnapshotStore::new();
    let err = store
        .save(SnapshotSave {
            session_id: 7,
            token_count: 1,
            token_hash: 1,
        })
        .err()
        .unwrap();
    assert_eq!(err, "worker has no distributed session to snapshot");
}

#[test]
fn store_save_reports_token_count_mismatch_with_c_text() {
    let mut store = MemorySnapshotStore::new();
    store.insert(7, vec![1, 2], b"shard").unwrap();
    let err = store
        .save(SnapshotSave {
            session_id: 7,
            token_count: 1,
            token_hash: token_hash_prefix(&[1, 2]),
        })
        .err()
        .unwrap();
    assert_eq!(err, "worker snapshot token count mismatch");
}

#[test]
fn store_save_reports_token_hash_mismatch_with_c_text() {
    let mut store = MemorySnapshotStore::new();
    store.insert(7, vec![1, 2], b"shard").unwrap();
    let err = store
        .save(SnapshotSave {
            session_id: 7,
            token_count: 2,
            token_hash: token_hash_prefix(&[1, 2]) ^ 1,
        })
        .err()
        .unwrap();
    assert_eq!(err, "worker snapshot token hash mismatch");
}

#[test]
fn store_load_sets_hash_valid_and_clears_it_on_restore_failure() {
    let mut store = MemorySnapshotStore::new();
    store.insert(7, vec![1], b"old").unwrap();
    assert!(store.token_hash_valid(7));

    let tokens = [3, 4];
    let hash = token_hash_prefix(&tokens);
    store
        .load(
            SnapshotLoad {
                session_id: 7,
                tokens: &tokens,
                token_hash: hash,
                payload_bytes: 4,
            },
            &mut Cursor::new(b"next"),
        )
        .unwrap();
    assert!(store.token_hash_valid(7));
    assert_eq!(store.token_hash(7), Some(hash));
    assert_eq!(store.tokens(7), Some(tokens.as_slice()));
    assert_eq!(store.read_shard(7).unwrap(), b"next");

    store.fail_restore(true);
    let err = store
        .load(
            SnapshotLoad {
                session_id: 7,
                tokens: &tokens,
                token_hash: hash,
                payload_bytes: 3,
            },
            &mut Cursor::new(b"bad"),
        )
        .unwrap_err();
    assert_eq!(err, "failed to restore worker KV shard");
    assert!(!store.token_hash_valid(7));
}

#[test]
fn worker_serve_save_load_round_trips_memory_shard() {
    let tokens = [7i32, 8];
    let meta = meta(&tokens);
    let first = b"kv-shard-one".to_vec();
    let second = b"kv-shard-two".to_vec();
    let mut store = MemorySnapshotStore::new();
    store
        .insert(meta.session_id, tokens.to_vec(), &first)
        .unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let worker_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        tune(&stream);
        let mut worker = Worker::with_store(exec(), store);
        worker.serve(&mut stream).unwrap();
    });

    let (mut stream, _) = listener.accept().unwrap();
    tune(&stream);
    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut stream, meta, &tokens, &mut saved).unwrap(),
        first.len() as u64
    );
    assert_eq!(saved, first);

    coordinator_load_snapshot(
        &mut stream,
        meta,
        &tokens,
        &mut Cursor::new(&second),
        second.len() as u64,
    )
    .unwrap();

    let mut saved = Vec::new();
    assert_eq!(
        coordinator_save_snapshot(&mut stream, meta, &tokens, &mut saved).unwrap(),
        second.len() as u64
    );
    assert_eq!(saved, second);

    drop(stream);
    worker_thread.join().unwrap();
}
