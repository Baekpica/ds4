//! POSIX `regcomp`/`regexec` matching C `agent_tool_search` regex mode.

use std::ffi::{CStr, CString};
use std::ptr;

pub struct PosixRegex {
    raw: libc::regex_t,
}

impl PosixRegex {
    pub fn compile(pattern: &str, case_sensitive: bool) -> Result<Self, Vec<u8>> {
        let end = pattern
            .as_bytes()
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(pattern.len());
        let cpat =
            CString::new(&pattern.as_bytes()[..end]).expect("pattern slice has no interior NUL");
        let mut raw = unsafe { std::mem::zeroed::<libc::regex_t>() };
        let mut flags = libc::REG_EXTENDED | libc::REG_NOSUB;
        if !case_sensitive {
            flags |= libc::REG_ICASE;
        }
        let rc = unsafe { libc::regcomp(&mut raw, cpat.as_ptr(), flags) };
        if rc != 0 {
            let mut buf = [0u8; 256];
            unsafe {
                libc::regerror(rc, &raw, buf.as_mut_ptr().cast::<libc::c_char>(), buf.len());
            }
            let msg =
                unsafe { CStr::from_ptr(buf.as_ptr().cast::<libc::c_char>()) }.to_string_lossy();
            return Err(format!("Tool error: invalid regex: {msg}\n").into_bytes());
        }
        Ok(Self { raw })
    }

    pub fn is_match(&self, line: &[u8]) -> bool {
        let mut buf = Vec::with_capacity(line.len() + 1);
        buf.extend_from_slice(line);
        buf.push(0);
        unsafe {
            libc::regexec(
                &self.raw,
                buf.as_ptr().cast::<libc::c_char>(),
                0,
                ptr::null_mut(),
                0,
            ) == 0
        }
    }
}

impl Drop for PosixRegex {
    fn drop(&mut self) {
        unsafe { libc::regfree(&mut self.raw) };
    }
}
