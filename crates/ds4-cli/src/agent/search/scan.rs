use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;

const FILE_MAX_BYTES: usize = 16 * 1024 * 1024;
const RESULT_MAX_BYTES: usize = 128 * 1024;

pub(super) struct BodyBuf {
    pub bytes: Vec<u8>,
    truncated: bool,
}

impl BodyBuf {
    pub(super) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            truncated: false,
        }
    }

    pub(super) fn append(&mut self, bytes: &[u8]) {
        if bytes.is_empty() || self.truncated {
            return;
        }
        let room = RESULT_MAX_BYTES.saturating_sub(self.bytes.len());
        let n = room.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..n]);
        self.truncated = n < bytes.len();
    }
}

pub(super) fn read_file(path: &[u8]) -> Option<Vec<u8>> {
    let file = File::open(OsStr::from_bytes(path)).ok()?;
    let mut data = Vec::new();
    let cap = u64::try_from(FILE_MAX_BYTES.saturating_add(1)).unwrap_or(u64::MAX);
    file.take(cap).read_to_end(&mut data).ok()?;
    if data.len() > FILE_MAX_BYTES {
        return None;
    }
    Some(data)
}

pub(super) fn split_lines(data: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let start = pos;
        while pos < data.len() && !matches!(data[pos], b'\n' | b'\r') {
            pos += 1;
        }
        let content_end = pos;
        if pos < data.len() {
            if data[pos] == b'\r' && data.get(pos + 1).copied() == Some(b'\n') {
                pos += 2;
            } else {
                pos += 1;
            }
        }
        spans.push((start, content_end));
    }
    spans
}

pub(super) fn literal_match(haystack: &[u8], query: &[u8], case_sensitive: bool) -> bool {
    if query.is_empty() {
        return true;
    }
    if query.len() > haystack.len() {
        return false;
    }
    haystack.windows(query.len()).any(|window| {
        window.iter().zip(query).all(|(left, right)| {
            if case_sensitive {
                left == right
            } else {
                left.to_ascii_lowercase() == right.to_ascii_lowercase()
            }
        })
    })
}

pub(super) fn glob_match(pattern: &[u8], name: &[u8]) -> bool {
    glob_at(pattern, 0, name, 0)
}

fn glob_at(pattern: &[u8], pi: usize, name: &[u8], ni: usize) -> bool {
    if pi == pattern.len() {
        return ni == name.len();
    }
    match pattern[pi] {
        b'\\' if pi + 1 < pattern.len() => {
            ni < name.len() && pattern[pi + 1] == name[ni] && glob_at(pattern, pi + 2, name, ni + 1)
        }
        b'?' => ni < name.len() && glob_at(pattern, pi + 1, name, ni + 1),
        b'*' => {
            let mut next = pi;
            while next < pattern.len() && pattern[next] == b'*' {
                next += 1;
            }
            glob_at(pattern, next, name, ni)
                || (ni..name.len()).any(|at| glob_at(pattern, next, name, at + 1))
        }
        byte => ni < name.len() && name[ni] == byte && glob_at(pattern, pi + 1, name, ni + 1),
    }
}

pub(crate) fn parse_int_default(value: Option<&str>, default: i32, min: i32, max: i32) -> i32 {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return default;
    };
    let trimmed = value.trim_start_matches(|char: char| {
        matches!(char, ' ' | '\t' | '\n' | '\r' | '\u{000b}' | '\u{000c}')
    });
    let (sign, digits) = match trimmed.as_bytes().first().copied() {
        Some(b'+') => (1i128, &trimmed[1..]),
        Some(b'-') => (-1i128, &trimmed[1..]),
        _ => (1i128, trimmed),
    };
    let Some(first) = digits.as_bytes().first() else {
        return default;
    };
    if !first.is_ascii_digit() {
        return default;
    }
    let consumed = digits
        .as_bytes()
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(digits.len());
    let rest = digits[consumed..].trim_start_matches([' ', '\t', '\r', '\n']);
    if !rest.is_empty() {
        return default;
    }
    let parsed = match digits[..consumed].parse::<i128>() {
        Ok(parsed) => parsed.saturating_mul(sign),
        Err(_) => {
            return if sign < 0 { min } else { max };
        }
    };
    match i32::try_from(parsed.clamp(i128::from(min), i128::from(max))) {
        Ok(value) => value,
        Err(_) => default,
    }
}

pub(super) fn clamp_usize(value: i32) -> usize {
    usize::try_from(value).unwrap_or(0)
}
