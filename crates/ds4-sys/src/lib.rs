//! The only unsafe Rust boundary onto the existing ds4 native runtime.
//!
//! Bindings cover `native/bridge/ds4_bridge.h` only. Do not bindgen `ds4.h`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct ds4_bridge_model {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct ds4_bridge_session {
    _opaque: [u8; 0],
}

pub type ds4_bridge_backend = c_int;

pub const DS4_BRIDGE_BACKEND_CUDA: ds4_bridge_backend = 0;
pub const DS4_BRIDGE_BACKEND_METAL: ds4_bridge_backend = 1;
pub const DS4_BRIDGE_BACKEND_CPU: ds4_bridge_backend = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_model_open_options {
    pub model_path: *const c_char,
    pub backend: c_int,
    pub n_threads: c_int,
    pub defer_boot_prewarm: c_int,
}

extern "C" {
    pub fn ds4_bridge_model_open(
        out: *mut *mut ds4_bridge_model,
        opt: *const ds4_bridge_model_open_options,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_model_free(m: *mut ds4_bridge_model);

    pub fn ds4_bridge_session_create(
        out: *mut *mut ds4_bridge_session,
        m: *mut ds4_bridge_model,
        ctx_size: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_session_free(s: *mut ds4_bridge_session);

    pub fn ds4_bridge_session_sync(
        s: *mut ds4_bridge_session,
        tokens: *const i32,
        n_tokens: c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_eval(
        s: *mut ds4_bridge_session,
        token: i32,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_session_argmax(s: *mut ds4_bridge_session) -> c_int;

    pub fn ds4_bridge_session_pos(s: *mut ds4_bridge_session) -> c_int;
}
