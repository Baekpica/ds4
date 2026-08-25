use super::identity::{clip_title, decode_title_trailer, title_from_text};
use super::store::SessionStore;
use ds4_kv::{decode_file, sha_hex_name, Record, EXT_SESSION_TITLE};
use std::fs;
use std::path::PathBuf;

#[derive(Debug)]
pub(crate) enum KvError {
    InvalidPrefix,
    Missing { prefix: String },
    Ambiguous { prefix: String },
    NothingToSave,
    IdentityMismatch,
    Io(String),
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPrefix => f.write_str("invalid session SHA prefix"),
            Self::Missing { prefix } => write!(f, "no saved session matches {prefix:.40}"),
            Self::Ambiguous { prefix } => {
                write!(f, "session prefix {prefix:.40} is ambiguous")
            }
            Self::NothingToSave => f.write_str("nothing to save"),
            Self::IdentityMismatch => {
                f.write_str("cached session identity does not match file name")
            }
            Self::Io(msg) => f.write_str(msg),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ListedSession {
    pub(crate) sha: String,
    pub(crate) title: String,
    pub(crate) last_used: u64,
    pub(crate) created_at: u64,
    pub(crate) tokens: u32,
    pub(crate) file_size: u64,
    pub(crate) stripped: bool,
}

impl SessionStore {
    pub(crate) fn list(&self) -> Result<Vec<ListedSession>, KvError> {
        let entries = fs::read_dir(self.dir()).map_err(|error| KvError::Io(error.to_string()))?;
        let mut sessions = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| KvError::Io(error.to_string()))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(sha) = sha_hex_name(name) else {
                continue;
            };
            let path = entry.path();
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let Ok(record) = decode_file(&bytes) else {
                continue;
            };
            if record.header.model_id != self.model_id() {
                continue;
            }
            sessions.push(ListedSession {
                sha,
                title: session_title(&record, 0),
                last_used: record.header.last_used,
                created_at: record.header.created_at,
                tokens: record.header.tokens,
                file_size: bytes.len() as u64,
                stripped: record.header.payload_bytes == 0,
            });
        }
        sessions.sort_by(|left, right| {
            recency(right)
                .cmp(&recency(left))
                .then_with(|| left.sha.cmp(&right.sha))
        });
        Ok(sessions)
    }

    pub(crate) fn find(&self, prefix: &str) -> Result<(String, PathBuf), KvError> {
        if !valid_prefix(prefix) {
            return Err(KvError::InvalidPrefix);
        }
        let entries = fs::read_dir(self.dir()).map_err(|error| KvError::Io(error.to_string()))?;
        let mut match_sha = None;
        let mut match_path = None;
        let mut matches = 0u32;
        for entry in entries {
            let entry = entry.map_err(|error| KvError::Io(error.to_string()))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(sha) = sha_hex_name(name) else {
                continue;
            };
            if !sha.starts_with(&prefix.to_ascii_lowercase()) {
                continue;
            }
            let path = entry.path();
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let Ok(record) = decode_file(&bytes) else {
                continue;
            };
            if record.header.model_id != self.model_id() {
                continue;
            }
            matches += 1;
            if matches == 1 {
                match_sha = Some(sha);
                match_path = Some(path);
            }
        }
        match (matches, match_sha, match_path) {
            (1, Some(sha), Some(path)) => Ok((sha, path)),
            (0, _, _) => Err(KvError::Missing {
                prefix: prefix.to_string(),
            }),
            _ => Err(KvError::Ambiguous {
                prefix: prefix.to_string(),
            }),
        }
    }
}

fn recency(session: &ListedSession) -> u64 {
    if session.last_used != 0 {
        session.last_used
    } else {
        session.created_at
    }
}

fn valid_prefix(prefix: &str) -> bool {
    let len = prefix.len();
    (1..=40).contains(&len) && prefix.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(super) fn session_title(record: &Record, max_bytes: usize) -> String {
    if record.header.ext_flags & EXT_SESSION_TITLE != 0 {
        if let Some(title) = decode_title_trailer(&record.trailer) {
            return clip_title(&title, max_bytes);
        }
    }
    title_from_text(&record.text, max_bytes)
}
