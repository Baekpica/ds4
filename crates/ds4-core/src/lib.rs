//! Safe Model / Session wrappers around `ds4-sys`.
//!
//! `unsafe` is confined to the FFI calls in this crate. Application
//! crates (`ds4-cli`, `ds4-server`, …) must not call `ds4-sys` directly.

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::ptr::{self, NonNull};

use ds4_sys::{
    ds4_bridge_eval, ds4_bridge_model, ds4_bridge_model_free, ds4_bridge_model_open,
    ds4_bridge_model_open_options, ds4_bridge_session, ds4_bridge_session_argmax,
    ds4_bridge_session_create, ds4_bridge_session_free, ds4_bridge_session_pos,
    ds4_bridge_session_sync, DS4_BRIDGE_BACKEND_CPU, DS4_BRIDGE_BACKEND_CUDA,
    DS4_BRIDGE_BACKEND_METAL,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
    Cpu,
}

impl Backend {
    fn to_c(self) -> i32 {
        match self {
            Backend::Cuda => DS4_BRIDGE_BACKEND_CUDA,
            Backend::Metal => DS4_BRIDGE_BACKEND_METAL,
            Backend::Cpu => DS4_BRIDGE_BACKEND_CPU,
        }
    }
}

#[derive(Debug)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.message.is_empty() {
            write!(f, "ds4 bridge error {}", self.code)
        } else {
            write!(f, "ds4 bridge error {}: {}", self.code, self.message)
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub struct TokenBuffer {
    tokens: Vec<i32>,
}

impl TokenBuffer {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn from_tokens(tokens: Vec<i32>) -> Self {
        Self { tokens }
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.tokens
    }

    pub fn push(&mut self, token: i32) {
        self.tokens.push(token);
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalResult {
    pub pos: i32,
}

pub struct Model {
    raw: NonNull<ds4_bridge_model>,
    _not_send: PhantomData<*const ()>,
}

pub struct Session<'m> {
    raw: NonNull<ds4_bridge_session>,
    _model: PhantomData<&'m Model>,
    _not_send: PhantomData<*const ()>,
}

fn c_err(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

fn fail(code: i32, buf: &[u8]) -> Error {
    Error {
        code,
        message: c_err(buf),
    }
}

fn cstring_path(path: &str) -> Result<CString> {
    CString::new(path).map_err(|_| Error {
        code: 1,
        message: "model path contains NUL".into(),
    })
}

impl Model {
    pub fn open(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
    ) -> Result<Self> {
        let c_path = cstring_path(path)?;
        let opt = ds4_bridge_model_open_options {
            model_path: c_path.as_ptr(),
            backend: backend.to_c(),
            n_threads,
            defer_boot_prewarm: i32::from(defer_boot_prewarm),
        };
        let mut raw = ptr::null_mut();
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_model_open(
                &mut raw,
                &opt,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_model_open returned NULL".into(),
        })?;
        Ok(Self {
            raw,
            _not_send: PhantomData,
        })
    }

    pub fn session(&self, ctx_size: i32) -> Result<Session<'_>> {
        let mut raw = ptr::null_mut();
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_create(
                &mut raw,
                self.raw.as_ptr(),
                ctx_size,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_session_create returned NULL".into(),
        })?;
        Ok(Session {
            raw,
            _model: PhantomData,
            _not_send: PhantomData,
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ds4_bridge_model_free(self.raw.as_ptr()) }
    }
}

impl Session<'_> {
    pub fn sync(&mut self, tokens: &TokenBuffer) -> Result<()> {
        if tokens.len() > i32::MAX as usize {
            return Err(Error {
                code: 1,
                message: "token buffer exceeds i32 length".into(),
            });
        }
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_sync(
                self.raw.as_ptr(),
                tokens.as_slice().as_ptr(),
                tokens.len() as i32,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        Ok(())
    }

    pub fn eval(&mut self, token: i32) -> Result<EvalResult> {
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_eval(
                self.raw.as_ptr(),
                token,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        Ok(EvalResult { pos: self.pos() })
    }

    pub fn argmax(&self) -> i32 {
        unsafe { ds4_bridge_session_argmax(self.raw.as_ptr()) }
    }

    pub fn pos(&self) -> i32 {
        unsafe { ds4_bridge_session_pos(self.raw.as_ptr()) }
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        unsafe { ds4_bridge_session_free(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_codes_match_bridge_header() {
        assert_eq!(Backend::Cuda.to_c(), 0);
        assert_eq!(Backend::Metal.to_c(), 1);
        assert_eq!(Backend::Cpu.to_c(), 2);
    }

    #[test]
    fn token_buffer_round_trip() {
        let mut buf = TokenBuffer::new();
        buf.push(1);
        buf.push(2);
        assert_eq!(buf.as_slice(), &[1, 2]);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn path_rejects_embedded_nul() {
        let err = cstring_path("a\0b").unwrap_err();
        assert_eq!(err.code, 1);
        assert!(err.message.contains("NUL"));
    }
}
