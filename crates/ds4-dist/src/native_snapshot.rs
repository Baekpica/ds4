//! Bind snapshot frames to a SnapshotStore with C restore-before-DONE order.

use std::fs::File;
use std::io::{self, Read, Write};

use crate::codec::{
    u64_from_halves, SnapshotReq, MSG_SNAPSHOT_LOAD_BEGIN, MSG_SNAPSHOT_SAVE_REQ,
    SNAPSHOT_REQ_FIXED_BYTES,
};
use crate::worker_snapshot::{
    worker_handle_snapshot_load_restore, worker_handle_snapshot_save, WorkerLoadOffer,
    WorkerSnapshotIdentity,
};

pub use crate::memory_snapshot::{MemorySnapshotStore, SnapshotLoad, SnapshotSave, SnapshotStore};
pub use crate::snapshot_temp::{copy_chunked, create_temp, TempShard, LOAD_PREFIX, SAVE_PREFIX};

pub fn prepare_snapshot_save<Store: SnapshotStore>(
    store: &mut Store,
    identity: WorkerSnapshotIdentity,
    req_body: &[u8],
) -> Result<(Store::SaveReader, u64), String> {
    if req_body.len() != SNAPSHOT_REQ_FIXED_BYTES {
        return Err(String::new());
    }
    let req = SnapshotReq::decode(req_body).map_err(|e| e.to_string())?;
    if !identity.accepts(
        req.model_id,
        req.layer_start,
        req.layer_end,
        req.token_count,
    ) {
        return Err(String::new());
    }
    store.save(SnapshotSave {
        session_id: u64_from_halves(req.session_hi, req.session_lo),
        token_count: req.token_count,
        token_hash: u64_from_halves(req.token_hash_hi, req.token_hash_lo),
    })
}

pub fn apply_snapshot_load<S, Store>(
    stream: &mut S,
    identity: WorkerSnapshotIdentity,
    begin_body: &[u8],
    vocab_size: u32,
    store: &mut Store,
) -> Result<WorkerLoadOffer, String>
where
    S: Read + Write,
    Store: SnapshotStore,
{
    match create_temp(LOAD_PREFIX) {
        Ok(mut tmp) => {
            let path = tmp.path.clone();
            worker_handle_snapshot_load_restore(
                stream,
                identity,
                begin_body,
                vocab_size,
                &mut tmp,
                |offer| {
                    let mut replay = File::open(&path)
                        .map_err(|_| "failed to rewind worker KV shard restore file".to_string())?;
                    store.load(
                        SnapshotLoad {
                            session_id: offer.session_id,
                            tokens: &offer.tokens,
                            token_hash: offer.token_hash,
                            payload_bytes: offer.payload_bytes,
                        },
                        &mut replay,
                    )
                },
            )
        }
        Err(_) => {
            let mut sink = io::sink();
            worker_handle_snapshot_load_restore(
                stream,
                identity,
                begin_body,
                vocab_size,
                &mut sink,
                |_| Err("failed to create worker snapshot restore temp file".into()),
            )
        }
    }
}

pub fn dispatch_worker_snapshot<Stream, Store>(
    stream: &mut Stream,
    typ: u32,
    body: &[u8],
    identity: WorkerSnapshotIdentity,
    vocab: u32,
    store: &mut Store,
) -> Result<Option<(u64, u64)>, String>
where
    Stream: Read + Write,
    Store: SnapshotStore,
{
    match typ {
        MSG_SNAPSHOT_SAVE_REQ => {
            match prepare_snapshot_save(store, identity, body) {
                Ok((mut reader, n)) => {
                    worker_handle_snapshot_save(stream, identity, body, Ok((&mut reader, n)))?;
                }
                Err(e) => {
                    worker_handle_snapshot_save::<_, io::Cursor<Vec<u8>>>(
                        stream,
                        identity,
                        body,
                        Err(e),
                    )?;
                }
            }
            Ok(None)
        }
        MSG_SNAPSHOT_LOAD_BEGIN => {
            let offer = apply_snapshot_load(stream, identity, body, vocab, store)?;
            Ok(Some((offer.session_id, offer.token_hash)))
        }
        _ => Err(format!("rejected unsupported frame type {typ}")),
    }
}
