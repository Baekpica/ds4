//! C `agent_trace*` one-shot log format from `ds4_agent.c`.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use ds4_core::{Backend, ChatThinkMode, Model};

pub(super) struct Trace {
    file: Option<File>,
}

impl Trace {
    pub(super) fn open(path: Option<&str>, name: &str) -> Result<Self, String> {
        let Some(path) = path.filter(|path| !path.is_empty()) else {
            return Ok(Self { file: None });
        };
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => Ok(Self { file: Some(file) }),
            Err(error) => Err(format!("{name}: failed to open trace {path}: {error}")),
        }
    }

    pub(super) fn event(&mut self, message: &str) -> Result<(), String> {
        write_line(self.file.as_mut(), message)
    }

    pub(super) fn text(&mut self, label: &str, bytes: &[u8]) -> Result<(), String> {
        let mut line = format!(" {label}=\"").into_bytes();
        escape_into(&mut line, bytes);
        line.push(b'"');
        write_line_bytes(self.file.as_mut(), &line)
    }

    pub(super) fn token(&mut self, token: i32, text: &[u8], index: i32) -> Result<(), String> {
        write_line_bytes(self.file.as_mut(), &format_token_line(token, text, index))
    }

    pub(super) fn tokens(
        &mut self,
        label: &str,
        model: &Model,
        tokens: &[i32],
        start: usize,
    ) -> Result<(), String> {
        let start = start.min(tokens.len());
        self.event(&format!(
            "tokens label={label} start={start} len={}",
            tokens.len()
        ))?;
        for (index, token) in tokens.iter().copied().enumerate().skip(start) {
            let text = model.token_text(token).map_err(|error| error.to_string())?;
            self.token(token, &text, index as i32)?;
        }
        Ok(())
    }
}

pub(super) fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
        Backend::Cpu => "cpu",
    }
}

pub(super) fn think_mode_name(mode: ChatThinkMode) -> &'static str {
    match mode {
        ChatThinkMode::None => "none",
        ChatThinkMode::Low => "low",
        ChatThinkMode::High => "high",
        ChatThinkMode::Max => "max",
    }
}

pub(super) fn escape(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    escape_into(&mut out, bytes);
    out
}

fn escape_into(out: &mut Vec<u8>, bytes: &[u8]) {
    for &byte in bytes {
        match byte {
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            b'"' => out.extend_from_slice(b"\\\""),
            0..=31 | 127 => out.extend_from_slice(format!("\\x{byte:02x}").as_bytes()),
            _ => out.push(byte),
        }
    }
}

fn format_token_line(token: i32, text: &[u8], index: i32) -> Vec<u8> {
    let mut line = format!(
        " token index={index} id={token} bytes={} text=\"",
        text.len()
    )
    .into_bytes();
    escape_into(&mut line, text);
    line.extend_from_slice(b"\" hex=");
    for byte in text {
        line.extend_from_slice(format!("{byte:02x}").as_bytes());
    }
    line
}

fn write_line(file: Option<&mut File>, message: &str) -> Result<(), String> {
    write_line_bytes(file, message.as_bytes())
}

fn write_line_bytes(file: Option<&mut File>, message: &[u8]) -> Result<(), String> {
    let Some(file) = file else {
        return Ok(());
    };
    write!(file, "{} ", format_trace_time()).map_err(|error| error.to_string())?;
    file.write_all(message)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.flush())
        .map_err(|error| error.to_string())
}

fn format_trace_time() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let mut ts = libc::timespec {
        tv_sec: now.as_secs() as libc::time_t,
        tv_nsec: i64::from(now.subsec_nanos()),
    };
    // SAFETY: `clock_gettime` writes a valid `timespec` on success; on failure
    // we keep the SystemTime fallback already stored in `ts`.
    unsafe {
        libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts);
    }
    let mut tm = libc::tm {
        tm_sec: 0,
        tm_min: 0,
        tm_hour: 0,
        tm_mday: 0,
        tm_mon: 0,
        tm_year: 0,
        tm_wday: 0,
        tm_yday: 0,
        tm_isdst: 0,
        tm_gmtoff: 0,
        tm_zone: std::ptr::null(),
    };
    // SAFETY: `localtime_r` fills the caller-owned `tm` from `ts.tv_sec`.
    unsafe {
        libc::localtime_r(&ts.tv_sec, &mut tm);
    }
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        ts.tv_nsec / 1_000_000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_matches_c() {
        assert_eq!(escape(br#"a\b"#), br#"a\\b"#);
        assert_eq!(escape(b"a\nb\rc\td\"e"), br#"a\nb\rc\td\"e"#);
        assert_eq!(escape(&[0x01, 0x7f]), br#"\x01\x7f"#);
        assert_eq!(escape("한글".as_bytes()), "한글".as_bytes());
    }

    #[test]
    fn token_line_matches_c_shape() {
        assert_eq!(
            format_token_line(7, b"ab\n", 3),
            br#" token index=3 id=7 bytes=3 text="ab\n" hex=61620a"#
        );
    }

    #[test]
    fn backend_and_think_names_match_c() {
        assert_eq!(backend_name(Backend::Cuda), "cuda");
        assert_eq!(backend_name(Backend::Metal), "metal");
        assert_eq!(backend_name(Backend::Cpu), "cpu");
        assert_eq!(think_mode_name(ChatThinkMode::None), "none");
        assert_eq!(think_mode_name(ChatThinkMode::Low), "low");
        assert_eq!(think_mode_name(ChatThinkMode::High), "high");
        assert_eq!(think_mode_name(ChatThinkMode::Max), "max");
    }
}
