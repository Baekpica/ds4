//! Bind worker SNAPSHOT frames to a live `Session` layer payload.

use std::cell::RefCell;
use std::io::{Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use ds4_core::{LayerPayloadLoad, Session};
use ds4_dist::{
    copy_chunked, create_temp, token_hash_prefix, SnapshotLoad, SnapshotSave, SnapshotStore,
    TempShard, LOAD_PREFIX, SAVE_PREFIX,
};

pub fn check_save_tokens(live: &[i32], req: &SnapshotSave) -> Result<(), String> {
    if live.len() as u32 != req.token_count {
        return Err("worker snapshot token count mismatch".into());
    }
    if token_hash_prefix(live) != req.token_hash {
        return Err("worker snapshot token hash mismatch".into());
    }
    Ok(())
}

pub struct SessionSnapshotStore<'m> {
    session: Rc<RefCell<Session<'m>>>,
    layer_start: u32,
    layer_end: u32,
}

impl<'m> SessionSnapshotStore<'m> {
    pub fn new(session: Session<'m>, layer_start: u32, layer_end: u32) -> Self {
        Self::from_shared(Rc::new(RefCell::new(session)), layer_start, layer_end)
    }

    pub fn from_shared(
        session: Rc<RefCell<Session<'m>>>,
        layer_start: u32,
        layer_end: u32,
    ) -> Self {
        Self {
            session,
            layer_start,
            layer_end,
        }
    }

    pub fn session(&self) -> &Rc<RefCell<Session<'m>>> {
        &self.session
    }
}

impl SnapshotStore for SessionSnapshotStore<'_> {
    type SaveReader = TempShard;

    fn save(&mut self, req: SnapshotSave) -> Result<(TempShard, u64), String> {
        check_save_tokens(self.session.borrow().host().tokens(), &req)?;
        let mut tmp = create_temp(SAVE_PREFIX)
            .map_err(|_| "failed to create worker snapshot temp file".to_string())?;
        self.session
            .borrow()
            .save_layer_payload(tmp.path(), self.layer_start, self.layer_end)
            .map_err(|_| "failed to save worker KV shard".to_string())?;
        tmp.rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        let payload_bytes = tmp
            .seek(SeekFrom::End(0))
            .map_err(|_| "failed to measure worker KV shard".to_string())?;
        tmp.rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        Ok((tmp, payload_bytes))
    }

    fn load(&mut self, req: SnapshotLoad<'_>, payload: &mut dyn Read) -> Result<(), String> {
        let mut tmp = create_temp(LOAD_PREFIX)
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        copy_chunked(payload, &mut tmp, req.payload_bytes)
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        tmp.flush()
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        self.session
            .borrow_mut()
            .load_layer_payload(LayerPayloadLoad {
                path: tmp.path(),
                payload_bytes: req.payload_bytes,
                tokens: req.tokens,
                layer_start: self.layer_start,
                layer_end: self.layer_end,
            })
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_token_checks_match_c() {
        let live = [1, 2, 3];
        let hash = token_hash_prefix(&live);
        assert!(check_save_tokens(
            &live,
            &SnapshotSave {
                session_id: 1,
                token_count: 3,
                token_hash: hash,
            }
        )
        .is_ok());
        assert_eq!(
            check_save_tokens(
                &live,
                &SnapshotSave {
                    session_id: 1,
                    token_count: 2,
                    token_hash: hash,
                }
            )
            .unwrap_err(),
            "worker snapshot token count mismatch"
        );
        assert_eq!(
            check_save_tokens(
                &live,
                &SnapshotSave {
                    session_id: 1,
                    token_count: 3,
                    token_hash: hash.wrapping_add(1),
                }
            )
            .unwrap_err(),
            "worker snapshot token hash mismatch"
        );
    }
}
