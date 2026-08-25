use super::catalog::{session_title, KvError};
use super::identity::{
    decode_title_trailer, encode_title_trailer, file_identity_sha, identity_sha,
};
use ds4_kv::{
    decode_file, encode_file, path_for_sha, write_path, Header, Reason, Record, EXT_SESSION_TITLE,
};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SwitchPlan {
    pub(crate) sha: String,
    pub(crate) path: PathBuf,
    pub(crate) needs_prefill: bool,
    pub(crate) title: String,
    pub(crate) created_at: u64,
    pub(crate) text: Vec<u8>,
    pub(crate) payload_offset: u64,
    pub(crate) payload_bytes: u64,
    pub(crate) tokens: u32,
}

pub(crate) struct SaveSpec<'a> {
    pub(crate) title: &'a str,
    pub(crate) created_at: u64,
    pub(crate) last_used: u64,
    pub(crate) text: &'a [u8],
    pub(crate) payload: &'a [u8],
    pub(crate) tokens: u32,
    pub(crate) model_id: u8,
    pub(crate) quant_bits: u8,
    pub(crate) ctx_size: u32,
}

pub(crate) struct SessionStore {
    dir: PathBuf,
    model_id: u8,
}

impl SessionStore {
    pub(crate) fn open(dir: impl Into<PathBuf>, model_id: u8) -> Self {
        Self {
            dir: dir.into(),
            model_id,
        }
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    pub(super) fn model_id(&self) -> u8 {
        self.model_id
    }

    pub(crate) fn save(&self, spec: SaveSpec<'_>) -> Result<String, KvError> {
        if spec.quant_bits != 2 && spec.quant_bits != 4 {
            return Err(KvError::Io(
                "unsupported routed quantization for KV save".into(),
            ));
        }
        let sha = identity_sha(spec.title, spec.created_at);
        fs::create_dir_all(&self.dir).map_err(|error| KvError::Io(error.to_string()))?;
        let record = Record {
            header: Header {
                quant_bits: spec.quant_bits,
                reason: Reason::AgentSession,
                ext_flags: EXT_SESSION_TITLE,
                model_id: spec.model_id,
                tokens: spec.tokens,
                hits: 0,
                ctx_size: spec.ctx_size,
                created_at: spec.created_at,
                last_used: spec.last_used,
                payload_bytes: spec.payload.len() as u64,
                text_bytes: spec.text.len() as u32,
            },
            text: spec.text.to_vec(),
            payload: spec.payload.to_vec(),
            trailer: encode_title_trailer(spec.title),
        };
        let path = path_for_sha(&self.dir, &sha);
        write_path(&path, &record).map_err(|error| KvError::Io(error.to_string()))?;
        debug_assert_eq!(&encode_file(&record)[..3], b"KVC");
        Ok(sha)
    }

    pub(crate) fn delete(&self, prefix: &str) -> Result<String, KvError> {
        let (sha, path) = self.find(prefix)?;
        fs::remove_file(path).map_err(|error| KvError::Io(error.to_string()))?;
        Ok(sha)
    }

    pub(crate) fn strip(
        &self,
        prefix: &str,
        token_count: u32,
        now: u64,
    ) -> Result<String, KvError> {
        let (sha, path) = self.find(prefix)?;
        let bytes = fs::read(&path).map_err(|error| KvError::Io(error.to_string()))?;
        let mut record = decode_file(&bytes).map_err(|error| KvError::Io(error.to_string()))?;
        let title = decode_title_trailer(&record.trailer);
        let actual = file_identity_sha(
            record.header.ext_flags,
            title.as_deref(),
            record.header.created_at,
            &record.text,
        );
        if actual != sha {
            return Err(KvError::IdentityMismatch);
        }
        record.payload.clear();
        record.header.payload_bytes = 0;
        record.header.tokens = token_count;
        record.header.last_used = now;
        write_path(&path, &record).map_err(|error| KvError::Io(error.to_string()))?;
        Ok(sha)
    }

    pub(crate) fn switch_plan(&self, prefix: &str) -> Result<SwitchPlan, KvError> {
        let (sha, path) = self.find(prefix)?;
        let bytes = fs::read(&path).map_err(|error| KvError::Io(error.to_string()))?;
        let record = decode_file(&bytes).map_err(|error| KvError::Io(error.to_string()))?;
        let title = session_title(&record, 0);
        let trailer_title = decode_title_trailer(&record.trailer);
        let actual = file_identity_sha(
            record.header.ext_flags,
            trailer_title.as_deref(),
            record.header.created_at,
            &record.text,
        );
        if actual != sha {
            return Err(KvError::IdentityMismatch);
        }
        let payload_offset = (ds4_kv::FIXED_HEADER + 4 + record.text.len()) as u64;
        Ok(SwitchPlan {
            sha,
            path,
            needs_prefill: record.header.payload_bytes == 0,
            title,
            created_at: record.header.created_at,
            text: record.text,
            payload_offset,
            payload_bytes: record.header.payload_bytes,
            tokens: record.header.tokens,
        })
    }
}
