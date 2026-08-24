//! mmap-backed GGUF v3 metadata reader.
//!
//! Copied from `ds4.c` (`cursor_*`, `skip_value`, `parse_metadata`,
//! `model_get_*`). Strings and values stay in the mapping.

use std::path::Path;

use crate::mapped::MappedFile;

pub const GGUF_MAGIC: u32 = 0x4655_4747; /* "GGUF", little endian */
pub const GGUF_VERSION: u32 = 3;

pub const GGUF_VALUE_UINT8: u32 = 0;
pub const GGUF_VALUE_INT8: u32 = 1;
pub const GGUF_VALUE_UINT16: u32 = 2;
pub const GGUF_VALUE_INT16: u32 = 3;
pub const GGUF_VALUE_UINT32: u32 = 4;
pub const GGUF_VALUE_INT32: u32 = 5;
pub const GGUF_VALUE_FLOAT32: u32 = 6;
pub const GGUF_VALUE_BOOL: u32 = 7;
pub const GGUF_VALUE_STRING: u32 = 8;
pub const GGUF_VALUE_ARRAY: u32 = 9;
pub const GGUF_VALUE_UINT64: u32 = 10;
pub const GGUF_VALUE_INT64: u32 = 11;
pub const GGUF_VALUE_FLOAT64: u32 = 12;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    TooSmall,
    NotGguf,
    UnsupportedVersion(u32),
    Truncated,
    Nest,
    Type,
    ArrayTooLarge,
}

impl GgufError {
    pub fn token(&self) -> String {
        match self {
            GgufError::Io(_) => "io".into(),
            GgufError::TooSmall => "too-small".into(),
            GgufError::NotGguf => "not-gguf".into(),
            GgufError::UnsupportedVersion(v) => format!("unsupported-version {v}"),
            GgufError::Truncated => "truncated".into(),
            GgufError::Nest => "nest".into(),
            GgufError::Type => "type".into(),
            GgufError::ArrayTooLarge => "array-too-large".into(),
        }
    }
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::InvalidData
            && e.to_string().contains("too small")
        {
            GgufError::TooSmall
        } else {
            GgufError::Io(e)
        }
    }
}

pub fn value_type_name(t: u32) -> &'static str {
    match t {
        GGUF_VALUE_UINT8 => "UINT8",
        GGUF_VALUE_INT8 => "INT8",
        GGUF_VALUE_UINT16 => "UINT16",
        GGUF_VALUE_INT16 => "INT16",
        GGUF_VALUE_UINT32 => "UINT32",
        GGUF_VALUE_INT32 => "INT32",
        GGUF_VALUE_FLOAT32 => "FLOAT32",
        GGUF_VALUE_BOOL => "BOOL",
        GGUF_VALUE_STRING => "STRING",
        GGUF_VALUE_ARRAY => "ARRAY",
        GGUF_VALUE_UINT64 => "UINT64",
        GGUF_VALUE_INT64 => "INT64",
        GGUF_VALUE_FLOAT64 => "FLOAT64",
        _ => "UNKNOWN",
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KvEntry {
    pub key_off: usize,
    pub key_len: usize,
    pub typ: u32,
    pub value_pos: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ArrayRef {
    pub typ: u32,
    pub len: u64,
    pub data_pos: usize,
}

pub struct GgufFile {
    map: MappedFile,
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    pub alignment: u64,
    pub tensor_dir_pos: usize,
    kv: Vec<KvEntry>,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn has(&self, n: u64) -> Result<(), GgufError> {
        let n = usize::try_from(n).map_err(|_| GgufError::Truncated)?;
        if n > self.data.len() || self.pos > self.data.len() - n {
            return Err(GgufError::Truncated);
        }
        Ok(())
    }

    fn skip(&mut self, n: u64) -> Result<(), GgufError> {
        self.has(n)?;
        self.pos += n as usize;
        Ok(())
    }

    fn read_exact<'b>(&'b mut self, n: usize) -> Result<&'b [u8], GgufError> {
        self.has(n as u64)?;
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<(usize, usize), GgufError> {
        let len = self.u64()?;
        let n = usize::try_from(len).map_err(|_| GgufError::Truncated)?;
        self.has(len)?;
        let off = self.pos;
        self.pos += n;
        Ok((off, n))
    }
}

fn scalar_value_size(typ: u32) -> u64 {
    match typ {
        GGUF_VALUE_UINT8 | GGUF_VALUE_INT8 | GGUF_VALUE_BOOL => 1,
        GGUF_VALUE_UINT16 | GGUF_VALUE_INT16 => 2,
        GGUF_VALUE_UINT32 | GGUF_VALUE_INT32 | GGUF_VALUE_FLOAT32 => 4,
        GGUF_VALUE_UINT64 | GGUF_VALUE_INT64 | GGUF_VALUE_FLOAT64 => 8,
        _ => 0,
    }
}

fn skip_value(c: &mut Cursor<'_>, typ: u32, depth: i32) -> Result<(), GgufError> {
    if depth > 8 {
        return Err(GgufError::Nest);
    }
    let scalar = scalar_value_size(typ);
    if scalar != 0 {
        return c.skip(scalar);
    }
    if typ == GGUF_VALUE_STRING {
        c.string()?;
        return Ok(());
    }
    if typ == GGUF_VALUE_ARRAY {
        let item_type = c.u32()?;
        let len = c.u64()?;
        let item_size = scalar_value_size(item_type);
        if item_size != 0 {
            if item_size != 0 && len > u64::MAX / item_size {
                return Err(GgufError::ArrayTooLarge);
            }
            return c.skip(len * item_size);
        }
        for _ in 0..len {
            skip_value(c, item_type, depth + 1)?;
        }
        return Ok(());
    }
    Err(GgufError::Type)
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let map = match MappedFile::open_ro(path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                return Err(GgufError::TooSmall);
            }
            Err(e) => return Err(GgufError::Io(e)),
        };
        Self::parse(map)
    }

    fn parse(map: MappedFile) -> Result<Self, GgufError> {
        let data = map.as_slice();
        let mut c = Cursor::new(data);
        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::NotGguf);
        }
        let version = c.u32()?;
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        if version != GGUF_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let mut kv = Vec::with_capacity(n_kv.min(1024) as usize);
        let mut alignment = 32u64;
        for _ in 0..n_kv {
            let (key_off, key_len) = c.string()?;
            let typ = c.u32()?;
            let value_pos = c.pos;
            let key = &data[key_off..key_off + key_len];
            if key == b"general.alignment" && typ == GGUF_VALUE_UINT32 {
                if let Some(bytes) = data.get(value_pos..value_pos + 4) {
                    let a = u32::from_le_bytes(bytes.try_into().unwrap());
                    if a != 0 {
                        alignment = u64::from(a);
                    }
                }
            }
            skip_value(&mut c, typ, 0)?;
            kv.push(KvEntry {
                key_off,
                key_len,
                typ,
                value_pos,
            });
        }
        let tensor_dir_pos = c.pos;

        Ok(Self {
            map,
            version,
            n_tensors,
            n_kv,
            alignment,
            tensor_dir_pos,
            kv,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.map.as_slice()
    }

    pub fn kv_entries(&self) -> &[KvEntry] {
        &self.kv
    }

    pub fn key_bytes(&self, e: &KvEntry) -> &[u8] {
        &self.map.as_slice()[e.key_off..e.key_off + e.key_len]
    }

    fn find(&self, key: &str) -> Option<&KvEntry> {
        let want = key.as_bytes();
        self.kv.iter().find(|e| self.key_bytes(e) == want)
    }

    fn cursor_at(&self, pos: usize) -> Result<Cursor<'_>, GgufError> {
        let data = self.map.as_slice();
        if pos > data.len() {
            return Err(GgufError::Truncated);
        }
        Ok(Cursor { data, pos })
    }

    /// C `model_get_string`: STRING only.
    pub fn get_string(&self, key: &str) -> Option<&[u8]> {
        let e = self.find(key)?;
        if e.typ != GGUF_VALUE_STRING {
            return None;
        }
        let mut c = self.cursor_at(e.value_pos).ok()?;
        let (off, len) = c.string().ok()?;
        Some(&self.map.as_slice()[off..off + len])
    }

    /// C `model_get_u32`: UINT32 only (not INT32).
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        let e = self.find(key)?;
        if e.typ != GGUF_VALUE_UINT32 {
            return None;
        }
        let mut c = self.cursor_at(e.value_pos).ok()?;
        c.u32().ok()
    }

    /// C `model_get_u16`: UINT16 only (llama.cpp splitter `split.count`).
    pub fn get_u16(&self, key: &str) -> Option<u16> {
        let e = self.find(key)?;
        if e.typ != GGUF_VALUE_UINT16 {
            return None;
        }
        let mut c = self.cursor_at(e.value_pos).ok()?;
        let b = c.read_exact(2).ok()?;
        Some(u16::from_le_bytes(b.try_into().ok()?))
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        let e = self.find(key)?;
        if e.typ != GGUF_VALUE_UINT64 {
            return None;
        }
        let mut c = self.cursor_at(e.value_pos).ok()?;
        c.u64().ok()
    }

    /// C `model_get_u64_compat`: UINT64 or UINT32.
    pub fn get_u64_compat(&self, key: &str) -> Option<u64> {
        if let Some(v) = self.get_u64(key) {
            return Some(v);
        }
        self.get_u32(key).map(u64::from)
    }

    /// C `model_get_f32_compat`: F32, F64, UINT32, INT32.
    pub fn get_f32_compat(&self, key: &str) -> Option<f32> {
        let e = self.find(key)?;
        let mut c = self.cursor_at(e.value_pos).ok()?;
        match e.typ {
            GGUF_VALUE_FLOAT32 => {
                let b = c.read_exact(4).ok()?;
                Some(f32::from_le_bytes(b.try_into().ok()?))
            }
            GGUF_VALUE_FLOAT64 => {
                let b = c.read_exact(8).ok()?;
                Some(f64::from_le_bytes(b.try_into().ok()?) as f32)
            }
            GGUF_VALUE_UINT32 => Some(c.u32().ok()? as f32),
            GGUF_VALUE_INT32 => {
                let b = c.read_exact(4).ok()?;
                Some(i32::from_le_bytes(b.try_into().ok()?) as f32)
            }
            _ => None,
        }
    }

    /// C `model_get_bool`: BOOL only.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        let e = self.find(key)?;
        if e.typ != GGUF_VALUE_BOOL {
            return None;
        }
        let mut c = self.cursor_at(e.value_pos).ok()?;
        let b = c.read_exact(1).ok()?;
        Some(b[0] != 0)
    }

    /// C `model_get_array`.
    pub fn get_array(&self, key: &str) -> Option<ArrayRef> {
        let e = self.find(key)?;
        if e.typ != GGUF_VALUE_ARRAY {
            return None;
        }
        let mut c = self.cursor_at(e.value_pos).ok()?;
        let typ = c.u32().ok()?;
        let len = c.u64().ok()?;
        Some(ArrayRef {
            typ,
            len,
            data_pos: c.pos,
        })
    }

    pub fn array_strings(&self, arr: &ArrayRef) -> Result<Vec<&[u8]>, GgufError> {
        if arr.typ != GGUF_VALUE_STRING {
            return Err(GgufError::Type);
        }
        let mut c = self.cursor_at(arr.data_pos)?;
        let mut out = Vec::with_capacity(arr.len.min(1_048_576) as usize);
        for _ in 0..arr.len {
            let (off, n) = c.string()?;
            out.push(&self.as_bytes()[off..off + n]);
        }
        Ok(out)
    }

    pub fn array_u32s(&self, arr: &ArrayRef) -> Result<Vec<u32>, GgufError> {
        if arr.typ != GGUF_VALUE_UINT32 {
            return Err(GgufError::Type);
        }
        self.array_le_u32s(arr)
    }

    /// C swiglu walk: FLOAT32 or FLOAT64 (narrowed).
    pub fn array_f32s(&self, arr: &ArrayRef) -> Result<Vec<f32>, GgufError> {
        let mut c = self.cursor_at(arr.data_pos)?;
        let mut out = Vec::with_capacity(arr.len.min(1_048_576) as usize);
        match arr.typ {
            GGUF_VALUE_FLOAT32 => {
                for _ in 0..arr.len {
                    let b = c.read_exact(4)?;
                    out.push(f32::from_le_bytes(b.try_into().map_err(|_| GgufError::Truncated)?));
                }
            }
            GGUF_VALUE_FLOAT64 => {
                for _ in 0..arr.len {
                    let b = c.read_exact(8)?;
                    out.push(f64::from_le_bytes(b.try_into().map_err(|_| GgufError::Truncated)?) as f32);
                }
            }
            _ => return Err(GgufError::Type),
        }
        Ok(out)
    }

    /// C EXAONE SWA pattern: BOOL bytes.
    pub fn array_bools(&self, arr: &ArrayRef) -> Result<Vec<bool>, GgufError> {
        if arr.typ != GGUF_VALUE_BOOL {
            return Err(GgufError::Type);
        }
        let mut c = self.cursor_at(arr.data_pos)?;
        let mut out = Vec::with_capacity(arr.len.min(1_048_576) as usize);
        for _ in 0..arr.len {
            let b = c.read_exact(1)?;
            out.push(b[0] != 0);
        }
        Ok(out)
    }

    /// C `vocab_load` token_type walk: four little-endian bytes, UINT32 or INT32.
    pub fn array_le_u32s(&self, arr: &ArrayRef) -> Result<Vec<u32>, GgufError> {
        if arr.typ != GGUF_VALUE_UINT32 && arr.typ != GGUF_VALUE_INT32 {
            return Err(GgufError::Type);
        }
        let mut c = self.cursor_at(arr.data_pos)?;
        let mut out = Vec::with_capacity(arr.len.min(1_048_576) as usize);
        for _ in 0..arr.len {
            out.push(c.u32()?);
        }
        Ok(out)
    }

    /// C `model_get_token_id`: lossless non-negative integer variants.
    pub fn get_token_id(&self, key: &str) -> Option<i32> {
        let e = self.find(key)?;
        let mut c = self.cursor_at(e.value_pos).ok()?;
        match e.typ {
            GGUF_VALUE_UINT32 => {
                let v = c.u32().ok()?;
                if v > i32::MAX as u32 {
                    None
                } else {
                    Some(v as i32)
                }
            }
            GGUF_VALUE_INT32 => {
                let b = c.read_exact(4).ok()?;
                let v = i32::from_le_bytes(b.try_into().ok()?);
                if v < 0 {
                    None
                } else {
                    Some(v)
                }
            }
            GGUF_VALUE_UINT64 => {
                let v = c.u64().ok()?;
                if v > i32::MAX as u64 {
                    None
                } else {
                    Some(v as i32)
                }
            }
            GGUF_VALUE_INT64 => {
                let b = c.read_exact(8).ok()?;
                let v = i64::from_le_bytes(b.try_into().ok()?);
                if v < 0 || v > i32::MAX as i64 {
                    None
                } else {
                    Some(v as i32)
                }
            }
            _ => None,
        }
    }

    /// First-shard `split.count`. Sibling remap lives in `tensors`.
    pub fn split_count(&self) -> u32 {
        if let Some(v) = self.get_u16("split.count") {
            return u32::from(v);
        }
        self.get_u32("split.count").unwrap_or(0)
    }

    pub fn dump_header_kv(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "HEADER version={} n_tensors={} n_kv={} alignment={}\n",
            self.version, self.n_tensors, self.n_kv, self.alignment
        ));
        for e in &self.kv {
            let key = String::from_utf8_lossy(self.key_bytes(e));
            out.push_str(&format!("KV {key} {}\n", value_type_name(e.typ)));
        }
        out.push_str(&format!("SPLIT {}\n", self.split_count()));
        match self.get_string("general.architecture") {
            None => out.push_str("ARCH missing\n"),
            Some(a) => {
                out.push_str("ARCH ");
                out.push_str(&String::from_utf8_lossy(a));
                out.push('\n');
            }
        }
        out
    }
}
