//! In-memory SnapshotStore with C worker status strings.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom, Write};

use crate::hash::token_hash_prefix;
use crate::snapshot_temp::{copy_chunked, create_temp, TempShard, LOAD_PREFIX, SAVE_PREFIX};

pub struct SnapshotSave {
    pub session_id: u64,
    pub token_count: u32,
    pub token_hash: u64,
}

pub struct SnapshotLoad<'a> {
    pub session_id: u64,
    pub tokens: &'a [i32],
    pub token_hash: u64,
    pub payload_bytes: u64,
}

pub trait SnapshotStore {
    type SaveReader: Read;
    fn save(&mut self, req: SnapshotSave) -> Result<(Self::SaveReader, u64), String>;
    fn load(&mut self, req: SnapshotLoad<'_>, payload: &mut dyn Read) -> Result<(), String>;
}

struct StoredSession {
    tokens: Vec<i32>,
    shard: TempShard,
    token_hash: u64,
    token_hash_valid: bool,
}

pub struct MemorySnapshotStore {
    sessions: HashMap<u64, StoredSession>,
    fail_restore: bool,
}

impl MemorySnapshotStore {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            fail_restore: false,
        }
    }

    pub fn insert(
        &mut self,
        session_id: u64,
        tokens: Vec<i32>,
        shard: &[u8],
    ) -> Result<(), String> {
        let mut tmp = create_temp(SAVE_PREFIX)
            .map_err(|_| "failed to create worker snapshot temp file".to_string())?;
        copy_chunked(&mut io::Cursor::new(shard), &mut tmp, shard.len() as u64)
            .map_err(|_| "failed to save worker KV shard".to_string())?;
        tmp.flush()
            .map_err(|_| "failed to flush worker KV shard".to_string())?;
        tmp.rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        let token_hash = token_hash_prefix(&tokens);
        self.sessions.insert(
            session_id,
            StoredSession {
                tokens,
                shard: tmp,
                token_hash,
                token_hash_valid: true,
            },
        );
        Ok(())
    }

    pub fn fail_restore(&mut self, fail: bool) {
        self.fail_restore = fail;
    }

    pub fn token_hash_valid(&self, session_id: u64) -> bool {
        self.sessions
            .get(&session_id)
            .is_some_and(|s| s.token_hash_valid)
    }

    pub fn token_hash(&self, session_id: u64) -> Option<u64> {
        self.sessions.get(&session_id).map(|s| s.token_hash)
    }

    pub fn tokens(&self, session_id: u64) -> Option<&[i32]> {
        self.sessions.get(&session_id).map(|s| s.tokens.as_slice())
    }

    pub fn read_shard(&mut self, session_id: u64) -> Result<Vec<u8>, String> {
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or_else(|| "worker has no distributed session to snapshot".to_string())?;
        session
            .shard
            .rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        let mut out = Vec::new();
        session
            .shard
            .read_to_end(&mut out)
            .map_err(|_| "failed to read worker KV shard".to_string())?;
        session
            .shard
            .rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        Ok(out)
    }
}

impl Default for MemorySnapshotStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotStore for MemorySnapshotStore {
    type SaveReader = TempShard;

    fn save(&mut self, req: SnapshotSave) -> Result<(TempShard, u64), String> {
        let session = self
            .sessions
            .get_mut(&req.session_id)
            .ok_or_else(|| "worker has no distributed session to snapshot".to_string())?;
        if session.tokens.len() as u32 != req.token_count {
            return Err("worker snapshot token count mismatch".into());
        }
        if token_hash_prefix(&session.tokens) != req.token_hash {
            return Err("worker snapshot token hash mismatch".into());
        }
        let mut tmp = create_temp(SAVE_PREFIX)
            .map_err(|_| "failed to create worker snapshot temp file".to_string())?;
        session
            .shard
            .rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        let len = session
            .shard
            .seek(SeekFrom::End(0))
            .map_err(|_| "failed to measure worker KV shard".to_string())?;
        session
            .shard
            .rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        copy_chunked(&mut session.shard, &mut tmp, len)
            .map_err(|_| "failed to save worker KV shard".to_string())?;
        tmp.flush()
            .map_err(|_| "failed to flush worker KV shard".to_string())?;
        let payload_bytes = tmp
            .stream_position()
            .map_err(|_| "failed to measure worker KV shard".to_string())?;
        tmp.rewind()
            .map_err(|_| "failed to rewind worker KV shard".to_string())?;
        Ok((tmp, payload_bytes))
    }

    fn load(&mut self, req: SnapshotLoad<'_>, payload: &mut dyn Read) -> Result<(), String> {
        if self.fail_restore {
            if let Some(session) = self.sessions.get_mut(&req.session_id) {
                session.token_hash_valid = false;
            }
            return Err("failed to restore worker KV shard".into());
        }
        let mut shard = create_temp(LOAD_PREFIX)
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        copy_chunked(payload, &mut shard, req.payload_bytes)
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        shard
            .rewind()
            .map_err(|_| "failed to restore worker KV shard".to_string())?;
        self.sessions.insert(
            req.session_id,
            StoredSession {
                tokens: req.tokens.to_vec(),
                shard,
                token_hash: req.token_hash,
                token_hash_valid: true,
            },
        );
        Ok(())
    }
}
