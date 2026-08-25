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
        // SAFETY: [Category 4 — Uninitialized Memory] `regex_t` is a C POD;
        // all-zero is a valid bit pattern. `regcomp` writes it before any field
        // is read. POSIX leaves contents undefined on failure; we do not `regfree`.
        let mut raw = unsafe { std::mem::zeroed::<libc::regex_t>() };
        let mut flags = libc::REG_EXTENDED | libc::REG_NOSUB;
        if !case_sensitive {
            flags |= libc::REG_ICASE;
        }
        // SAFETY: [Category 8 — FFI] `raw` is the zeroed `regex_t` above; `cpat`
        // is a live CString borrowed for the call. libc does not retain the pattern.
        let rc = unsafe { libc::regcomp(&mut raw, cpat.as_ptr(), flags) };
        if rc != 0 {
            let mut buf = [0u8; 256];
            // SAFETY: [Category 8 — FFI] `regerror` writes a NUL-terminated message
            // into the caller-owned `buf`. `raw` is the failed compile object;
            // POSIX allows `regerror` on it. Pointers are not retained.
            unsafe {
                libc::regerror(rc, &raw, buf.as_mut_ptr().cast::<libc::c_char>(), buf.len());
            }
            // SAFETY: [Category 8 — FFI] `buf` is a 256-byte stack array that
            // `regerror` NUL-terminated; we never read past the terminator.
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
        // SAFETY: [Category 8 — FFI] `self.raw` was compiled by `regcomp` and is
        // not yet dropped. `buf` is a NUL-terminated copy of `line` borrowed for
        // the call (`nmatch=0`, NULL pmatch). libc does not retain the pointer.
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
        // SAFETY: [Category 12 — Double free] `raw` was successfully `regcomp`'d
        // into `Self` and this Drop is the only `regfree`. Failed compiles never
        // construct `Self`.
        unsafe { libc::regfree(&mut self.raw) };
    }
}
