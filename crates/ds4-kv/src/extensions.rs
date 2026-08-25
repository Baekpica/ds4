//! Bank thinking bit + KTM extension records.
//!
//! Thinking is `EXT_THINKING_VISIBLE` on the KVC header (no extra magic).
//! The only appended trailer magic is C's `KTM\x01`. A missing or invalid
//! section is ignored the way `kv_tool_map_load_from_pos` returns 0.

use crate::format::{Record, EXT_BANK_REPLAY_V1, EXT_THINKING_VISIBLE, EXT_TOOL_MAP};

pub const KTM_MAGIC: [u8; 3] = [b'K', b'T', b'M'];
pub const KTM_VERSION: u8 = 1;
const KTM_HEADER: [u8; 4] = [b'K', b'T', b'M', KTM_VERSION];
const DEFAULT_MAX_IDS: usize = 100_000;
const MAX_ID_BYTES: usize = 256;
const MAX_DSML_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtensionRecord {
    pub id: Vec<u8>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BankThinkingExtensions {
    pub thinking_visible: bool,
    pub records: Vec<ExtensionRecord>,
}

impl BankThinkingExtensions {
    pub fn ext_flags(&self) -> u8 {
        self.wire().0
    }

    pub fn encode_trailer(&self) -> Vec<u8> {
        self.wire().1
    }

    fn wire(&self) -> (u8, Vec<u8>) {
        let trailer = encode_ktm(&self.records);
        let mut flags = EXT_BANK_REPLAY_V1;
        if self.thinking_visible {
            flags |= EXT_THINKING_VISIBLE;
        }
        if !trailer.is_empty() {
            flags |= EXT_TOOL_MAP;
        }
        (flags, trailer)
    }

    pub fn restore(ext_flags: u8, trailer: &[u8]) -> Self {
        Self {
            thinking_visible: ext_flags & EXT_THINKING_VISIBLE != 0,
            records: decode_ktm(trailer),
        }
    }
}

impl Record {
    pub fn persist_bank_extensions(&mut self, ext: &BankThinkingExtensions) {
        let (flags, trailer) = ext.wire();
        self.header.ext_flags = flags;
        self.trailer = trailer;
        self.header.text_bytes = self.text.len() as u32;
        self.header.payload_bytes = self.payload.len() as u64;
    }
}

fn wire_ok(id: &[u8], body: &[u8]) -> bool {
    !id.is_empty()
        && !body.is_empty()
        && id.len() <= MAX_ID_BYTES
        && body.len() <= MAX_DSML_BYTES
        && !id.contains(&0)
        && !body.contains(&0)
}

pub fn encode_ktm(records: &[ExtensionRecord]) -> Vec<u8> {
    let kept: Vec<&ExtensionRecord> = records
        .iter()
        .filter(|record| wire_ok(&record.id, &record.body))
        .collect();
    if kept.is_empty() {
        return Vec::new();
    }
    let Ok(count) = u32::try_from(kept.len()) else {
        return Vec::new();
    };
    let mut out = Vec::from(KTM_HEADER);
    out.extend_from_slice(&count.to_le_bytes());
    for record in kept {
        let id_len = record.id.len() as u32;
        let body_len = record.body.len() as u32;
        out.extend_from_slice(&id_len.to_le_bytes());
        out.extend_from_slice(&body_len.to_le_bytes());
        out.extend_from_slice(&record.id);
        out.extend_from_slice(&record.body);
    }
    out
}

pub fn decode_ktm(bytes: &[u8]) -> Vec<ExtensionRecord> {
    decode_ktm_bounded(bytes, DEFAULT_MAX_IDS)
}

fn decode_ktm_bounded(bytes: &[u8], max_ids: usize) -> Vec<ExtensionRecord> {
    if bytes.len() < 8 || bytes[..4] != KTM_HEADER {
        return Vec::new();
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let max_ids = if max_ids == 0 {
        DEFAULT_MAX_IDS
    } else {
        max_ids
    };
    let max_count = (max_ids as u64).saturating_mul(4);
    if u64::from(count) > max_count {
        return Vec::new();
    }
    let mut records = Vec::new();
    let mut pos = 8usize;
    for _ in 0..count {
        let Some(lens_end) = pos.checked_add(8).filter(|&end| end <= bytes.len()) else {
            return records;
        };
        let id_len =
            u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        let body_len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        if id_len == 0 || id_len > MAX_ID_BYTES || body_len == 0 || body_len > MAX_DSML_BYTES {
            return records;
        }
        pos = lens_end;
        let Some(id_end) = pos.checked_add(id_len).filter(|&end| end <= bytes.len()) else {
            return records;
        };
        let Some(body_end) = id_end
            .checked_add(body_len)
            .filter(|&end| end <= bytes.len())
        else {
            return records;
        };
        let id = until_nul(&bytes[pos..id_end]);
        let body = until_nul(&bytes[id_end..body_end]);
        if !id.is_empty() && !body.is_empty() {
            records.push(ExtensionRecord {
                id: id.to_vec(),
                body: body.to_vec(),
            });
        }
        pos = body_end;
    }
    records
}

fn until_nul(bytes: &[u8]) -> &[u8] {
    match bytes.iter().position(|&b| b == 0) {
        Some(i) => &bytes[..i],
        None => bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_records_write_no_ktm_header() {
        let ext = BankThinkingExtensions {
            thinking_visible: true,
            records: Vec::new(),
        };
        assert_eq!(ext.ext_flags(), EXT_BANK_REPLAY_V1 | EXT_THINKING_VISIBLE);
        assert!(ext.encode_trailer().is_empty());
    }

    #[test]
    fn invalid_records_do_not_set_tool_map() {
        let ext = BankThinkingExtensions {
            thinking_visible: false,
            records: vec![ExtensionRecord {
                id: Vec::new(),
                body: b"x".to_vec(),
            }],
        };
        assert_eq!(ext.ext_flags(), EXT_BANK_REPLAY_V1);
        assert!(ext.encode_trailer().is_empty());
    }
}
