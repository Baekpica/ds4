//! Directory-backed KVC catalog: refresh, prefix lookup, LCP, evict.

use crate::format::{
    is_automatic_exact_replay, is_bank_replay_v1, path_for_sha, read_path, sha_hex_name,
    text_sha_hex, write_path, Header, Record,
};
use crate::policy::{
    eviction_score, file_size_fits, EvictionContext, Options, ScoreEntry, DEFAULT_MB,
};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Entry {
    pub sha: String,
    pub path: PathBuf,
    pub header: Header,
    pub file_size: u64,
}

#[derive(Debug)]
pub struct Store {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    pub reject_different_quant: bool,
    pub opt: Options,
    pub continued_last_store_tokens: i32,
    entries: Vec<Entry>,
}

impl Store {
    pub fn open(
        dir: impl AsRef<Path>,
        budget_mb: u64,
        reject_different_quant: bool,
        opt: Options,
    ) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let budget_mb = if budget_mb == 0 { DEFAULT_MB } else { budget_mb };
        let mut store = Self {
            dir,
            budget_bytes: budget_mb * 1024 * 1024,
            reject_different_quant,
            opt,
            continued_last_store_tokens: 0,
            entries: Vec::new(),
        };
        store.evict(0, None);
        Ok(store)
    }

    pub fn refresh(&mut self) {
        self.entries.clear();
        let Ok(rd) = fs::read_dir(&self.dir) else {
            return;
        };
        for ent in rd.flatten() {
            let name = ent.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(sha) = sha_hex_name(name) else { continue };
            let path = ent.path();
            let Ok(meta) = fs::metadata(&path) else { continue };
            let Ok(rec) = read_path(&path) else { continue };
            if meta.len() < 48 + 4 + u64::from(rec.header.text_bytes) + rec.header.payload_bytes {
                continue;
            }
            self.entries.push(Entry {
                sha,
                path,
                header: rec.header,
                file_size: meta.len(),
            });
        }
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn write(&mut self, mut record: Record) -> io::Result<PathBuf> {
        if !file_size_fits(
            self.budget_bytes,
            record.text.len() as u64,
            record.payload.len() as u64,
            record.trailer.len() as u64,
        ) {
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "KVC file exceeds budget"));
        }
        record.header.text_bytes = record.text.len() as u32;
        record.header.payload_bytes = record.payload.len() as u64;
        let sha = text_sha_hex(&record.text);
        let path = path_for_sha(&self.dir, &sha);
        let incoming = EvictionContext {
            text: &record.text,
            model_id: record.header.model_id,
            quant_bits: record.header.quant_bits,
            ctx_size: record.header.ctx_size,
            reject_different_quant: self.reject_different_quant,
        };
        let extra = 48 + 4 + record.text.len() as u64 + record.payload.len() as u64 + record.trailer.len() as u64;
        self.evict(extra, Some(&incoming));
        write_path(&path, &record).map_err(|e| io::Error::other(e))?;
        self.refresh();
        Ok(path)
    }

    pub fn read(&self, path: &Path) -> io::Result<Record> {
        read_path(path).map_err(|e| io::Error::other(e))
    }

    pub fn find_text_prefix(&mut self, prompt: &[u8], model_id: u8, quant_bits: u8, ctx_size: u32) -> Option<usize> {
        self.find_prefix(prompt, model_id, quant_bits, ctx_size, false)
    }

    pub fn find_bank_text_prefix(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
    ) -> Option<usize> {
        self.find_prefix(prompt, model_id, quant_bits, ctx_size, true)
    }

    fn find_prefix(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        require_suffix: bool,
    ) -> Option<usize> {
        self.refresh();
        let prompt_bytes = prompt.len();
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !is_automatic_exact_replay(e.header.reason, e.header.ext_flags) {
                continue;
            }
            if e.header.text_bytes as usize > prompt_bytes {
                continue;
            }
            if (require_suffix || is_bank_replay_v1(e.header.reason, e.header.ext_flags))
                && e.header.text_bytes as usize == prompt_bytes
            {
                continue;
            }
            if (e.header.tokens as i32) < self.opt.min_tokens {
                continue;
            }
            if e.header.model_id != model_id {
                continue;
            }
            if ctx_size < e.header.ctx_size {
                continue;
            }
            if self.reject_different_quant && e.header.quant_bits != quant_bits {
                continue;
            }
            if let Some(b) = best {
                let be = &self.entries[b];
                if (e.header.text_bytes as usize) < be.header.text_bytes as usize {
                    continue;
                }
                if e.header.text_bytes == be.header.text_bytes && e.header.tokens <= be.header.tokens {
                    continue;
                }
            }
            let sha = text_sha_hex(&prompt[..e.header.text_bytes as usize]);
            if sha == e.sha {
                best = Some(i);
            }
        }
        best
    }

    pub fn find_text_lcp(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        min_lcp: usize,
    ) -> Option<(usize, usize)> {
        if min_lcp == 0 || prompt.len() < min_lcp {
            return None;
        }
        self.refresh();
        let mut best: Option<(usize, usize, u64)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !is_automatic_exact_replay(e.header.reason, e.header.ext_flags) {
                continue;
            }
            if (e.header.tokens as i32) < self.opt.min_tokens {
                continue;
            }
            if e.header.model_id != model_id || ctx_size < e.header.ctx_size {
                continue;
            }
            if self.reject_different_quant && e.header.quant_bits != quant_bits {
                continue;
            }
            if (e.header.text_bytes as usize) < min_lcp {
                continue;
            }
            if u64::from(e.header.text_bytes) > 8 * prompt.len() as u64 {
                continue;
            }
            let Ok(rec) = read_path(&e.path) else { continue };
            let want = (e.header.text_bytes as usize).min(prompt.len());
            let mut lcp = 0;
            while lcp < want && rec.text[lcp] == prompt[lcp] {
                lcp += 1;
            }
            if lcp < min_lcp {
                continue;
            }
            if 8 * (lcp as u64) < u64::from(e.header.text_bytes) {
                continue;
            }
            if let Some((_, best_lcp, best_text)) = best {
                if lcp < best_lcp {
                    continue;
                }
                if lcp == best_lcp && u64::from(e.header.text_bytes) >= best_text {
                    continue;
                }
            }
            best = Some((i, lcp, u64::from(e.header.text_bytes)));
        }
        best.map(|(i, lcp, _)| (i, lcp))
    }

    pub fn evict(&mut self, extra_bytes: u64, incoming: Option<&EvictionContext<'_>>) {
        if self.budget_bytes == 0 || extra_bytes > self.budget_bytes {
            return;
        }
        self.refresh();
        let now = now_secs();
        let target = self.budget_bytes - extra_bytes;
        loop {
            let total: u64 = self.entries.iter().map(|e| e.file_size).sum();
            if total <= target || self.entries.is_empty() {
                break;
            }
            let mut victim = 0;
            let mut victim_score = self.score_at(0, now, incoming);
            for i in 1..self.entries.len() {
                let score = self.score_at(i, now, incoming);
                if score < victim_score
                    || (score == victim_score && self.entries[i].header.last_used < self.entries[victim].header.last_used)
                {
                    victim = i;
                    victim_score = score;
                }
            }
            let path = self.entries[victim].path.clone();
            let _ = fs::remove_file(path);
            self.entries.remove(victim);
        }
    }

    fn score_at(&self, i: usize, now: u64, incoming: Option<&EvictionContext<'_>>) -> f64 {
        let e = &self.entries[i];
        eviction_score(
            &ScoreEntry {
                sha: &e.sha,
                quant_bits: e.header.quant_bits,
                model_id: e.header.model_id,
                reason: e.header.reason,
                tokens: e.header.tokens,
                hits: e.header.hits,
                ctx_size: e.header.ctx_size,
                created_at: e.header.created_at,
                last_used: e.header.last_used,
                text_bytes: u64::from(e.header.text_bytes),
                file_size: e.file_size,
            },
            now,
            incoming,
        )
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{Header, Reason};

    fn rec(text: &[u8], tokens: u32) -> Record {
        Record {
            header: Header {
                quant_bits: 2,
                reason: Reason::Cold,
                ext_flags: 0,
                model_id: 0,
                tokens,
                hits: 0,
                ctx_size: 2048,
                created_at: 1,
                last_used: 1,
                payload_bytes: 1,
                text_bytes: text.len() as u32,
            },
            text: text.to_vec(),
            payload: vec![0xAA],
            trailer: Vec::new(),
        }
    }

    #[test]
    fn prefix_prefers_longest_matching_text() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-prefix-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        store.write(rec(b"hello", 512)).unwrap();
        store.write(rec(b"hello world", 512)).unwrap();
        let idx = store
            .find_text_prefix(b"hello world!", 0, 2, 2048)
            .unwrap();
        assert_eq!(store.entries()[idx].header.text_bytes, 11);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bank_lookup_requires_suffix() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-bank-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut r = rec(b"hello", 512);
        r.header.reason = Reason::BankEvict;
        r.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1;
        store.write(r).unwrap();
        assert!(store.find_bank_text_prefix(b"hello", 0, 2, 2048).is_none());
        assert!(store.find_bank_text_prefix(b"hello!", 0, 2, 2048).is_some());
        let _ = fs::remove_dir_all(&dir);
    }
}
