//! The only unsafe Rust boundary onto the existing ds4 native runtime.
//!
//! Bindings cover `native/bridge/ds4_bridge.h` plus the platform libc numeric
//! parsers needed to preserve frozen C environment-variable semantics. Do not
//! bindgen `ds4.h`.

#![allow(non_camel_case_types)]

use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int, c_void};
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, Ordering};

unsafe extern "C" {
    fn atoi(value: *const c_char) -> c_int;
    fn strtoull(value: *const c_char, end: *mut *mut c_char, base: c_int) -> u64;
    fn atof(value: *const c_char) -> f64;
    #[cfg(unix)]
    fn signal(sig: c_int, handler: usize) -> usize;
    #[cfg(unix)]
    fn _exit(status: c_int) -> !;
}

#[cfg(unix)]
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn stop_signal_handler(_signal: c_int) {
    if STOP_REQUESTED.swap(true, Ordering::Relaxed) {
        unsafe { _exit(130) }
    }
}

#[cfg(unix)]
pub fn install_stop_handlers() -> bool {
    STOP_REQUESTED.store(false, Ordering::Relaxed);
    unsafe {
        let handler = stop_signal_handler as *const () as usize;
        signal(2, handler) != usize::MAX && signal(15, handler) != usize::MAX
    }
}

#[cfg(not(unix))]
pub fn install_stop_handlers() -> bool {
    true
}

#[cfg(unix)]
pub fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::Relaxed)
}

#[cfg(not(unix))]
pub fn stop_requested() -> bool {
    false
}

pub fn libc_atoi(value: &[u8]) -> i32 {
    CString::new(value).map_or(0, |value| unsafe { atoi(value.as_ptr()) })
}

pub fn libc_strtoull10(value: &[u8]) -> u64 {
    CString::new(value).map_or(0, |value| unsafe {
        strtoull(value.as_ptr(), std::ptr::null_mut(), 10)
    })
}

pub fn libc_atof(value: &[u8]) -> f64 {
    CString::new(value).map_or(0.0, |value| unsafe { atof(value.as_ptr()) })
}

#[repr(C)]
pub struct ds4_bridge_model {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct ds4_bridge_session {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct ds4_bridge_snapshot {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct ds4_bridge_batch_ctx {
    _opaque: [u8; 0],
}

pub type ds4_bridge_prefill_fn = Option<unsafe extern "C" fn(*mut c_void, i32, i32)>;

pub type ds4_bridge_backend = c_int;

pub const DS4_BRIDGE_BACKEND_CUDA: ds4_bridge_backend = 0;
pub const DS4_BRIDGE_BACKEND_METAL: ds4_bridge_backend = 1;
pub const DS4_BRIDGE_BACKEND_CPU: ds4_bridge_backend = 2;

pub const DS4_BRIDGE_MAX_DIMS: usize = 8;
pub const DS4_BRIDGE_MEMC_COUNT: usize = 17;
pub const DS4_BRIDGE_MEMD_COUNT: usize = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_token_score {
    pub id: i32,
    pub logit: c_float,
    pub logprob: c_float,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_mem_cell {
    pub requested: u64,
    pub committed: u64,
    pub freed_requested: u64,
    pub freed_committed: u64,
    pub alloc_calls: u64,
    pub free_calls: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_mem_census {
    pub supported: i32,
    pub faults: u64,
    pub epoch: u64,
    pub torn_fallbacks: u64,
    pub cells: [[ds4_bridge_mem_cell; DS4_BRIDGE_MEMD_COUNT]; DS4_BRIDGE_MEMC_COUNT],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_mem_observe {
    pub status: i32,
    pub source: i32,
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub cuda_free_bytes: u64,
    pub meminfo_avail_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_bind_slot {
    pub name: *const c_char,
    pub required: u32,
    pub ndim: u32,
    pub dim: [u64; DS4_BRIDGE_MAX_DIMS],
    pub r#type: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub bytes: u64,
    pub shard: u32,
    pub found: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_shard {
    pub path: *const c_char,
    pub size: u64,
    pub base: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_bind_plan {
    pub n_slots: u32,
    pub slots: *const ds4_bridge_bind_slot,
    pub n_shards: u32,
    pub shards: *const ds4_bridge_shard,
    pub data_pos: u64,
    pub alignment: u64,
    pub page: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_model_open_options {
    pub model_path: *const c_char,
    pub backend: c_int,
    pub n_threads: c_int,
    pub defer_boot_prewarm: c_int,
    pub plan: *const ds4_bridge_bind_plan,
    pub tensors: *const ds4_host_tensor_dir,
    pub shape: *const ds4_host_shape,
    pub vocab: *const ds4_host_vocab,
    pub bind: *const ds4_host_bind_map,
    pub mtp_path: *const c_char,
    pub dspark_path: *const c_char,
    pub mtp_bind: *const ds4_host_bind_map,
    pub dspark_bind: *const ds4_host_bind_map,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_host_shape {
    pub variant: u32,
    pub n_compress: u32,
    pub compress: *const u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_host_str {
    pub ptr: *const c_char,
    pub len: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_host_vocab {
    pub n_vocab: u32,
    pub tokens: *const ds4_host_str,
    pub n_merges: u32,
    pub merges: *const ds4_host_str,
    pub n_user_defined: u32,
    pub user_defined: *const i32,
    pub user_defined_max_len: u32,
    pub user_defined_first: [u8; 256],
    pub motif3_added_first: [u8; 256],
    pub bos_id: i32,
    pub eos_id: i32,
    pub system_id: i32,
    pub eot_id: i32,
    pub im_start_id: i32,
    pub im_content_id: i32,
    pub im_end_id: i32,
    pub user_id: i32,
    pub assistant_id: i32,
    pub start_of_turn_id: i32,
    pub end_of_turn_id: i32,
    pub tool_id: i32,
    pub reference_id: i32,
    pub plan_start_id: i32,
    pub plan_end_id: i32,
    pub observation_id: i32,
    pub sop_id: i32,
    pub think_start_id: i32,
    pub think_end_id: i32,
    pub tool_call_start_id: i32,
    pub tool_call_end_id: i32,
    pub tool_response_start_id: i32,
    pub tool_response_end_id: i32,
    pub arg_key_start_id: i32,
    pub arg_key_end_id: i32,
    pub arg_value_start_id: i32,
    pub latent_start_id: i32,
    pub latent_pad_id: i32,
    pub latent_end_id: i32,
    pub tool_schema_start_id: i32,
    pub tool_schema_end_id: i32,
    pub dsml_id: i32,
    pub dots3_endofsystem_id: i32,
    pub dots3_endofuser_id: i32,
    pub dots3_endoftext_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_host_bind_look {
    pub name: *const c_char,
    pub required: u32,
    pub found: u32,
    pub index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_host_bind_map {
    pub n: u32,
    pub v: *const ds4_host_bind_look,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_host_tensor {
    pub name: *const c_char,
    pub ndim: u32,
    pub dim: [u64; DS4_BRIDGE_MAX_DIMS],
    pub r#type: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_host_tensor_dir {
    pub n: u32,
    pub v: *const ds4_host_tensor,
    pub data_pos: u64,
    pub alignment: u64,
}

extern "C" {
    pub fn ds4_bridge_bind_plan_check(
        plan: *const ds4_bridge_bind_plan,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_bind_plan_match(
        host: *const ds4_bridge_bind_plan,
        native: *const ds4_bridge_bind_plan,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_model_open(
        out: *mut *mut ds4_bridge_model,
        opt: *const ds4_bridge_model_open_options,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_model_boot_prewarm(m: *mut ds4_bridge_model);

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

    pub fn ds4_bridge_session_sync_cb(
        s: *mut ds4_bridge_session,
        tokens: *const i32,
        n_tokens: c_int,
        progress: ds4_bridge_prefill_fn,
        ud: *mut c_void,
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

    pub fn ds4_bridge_session_argmax_excluding(
        s: *mut ds4_bridge_session,
        excluded_id: i32,
    ) -> c_int;

    pub fn ds4_bridge_session_pos(s: *mut ds4_bridge_session) -> c_int;

    pub fn ds4_bridge_session_ctx(s: *mut ds4_bridge_session) -> c_int;

    pub fn ds4_bridge_session_rewind(s: *mut ds4_bridge_session, pos: c_int);

    pub fn ds4_bridge_session_invalidate(s: *mut ds4_bridge_session);

    pub fn ds4_bridge_session_generation(s: *mut ds4_bridge_session) -> u64;

    pub fn ds4_bridge_session_prefill_cap(s: *mut ds4_bridge_session) -> c_int;

    pub fn ds4_bridge_session_exaone_rewind_span(s: *mut ds4_bridge_session) -> c_int;

    pub fn ds4_bridge_session_sample(
        s: *mut ds4_bridge_session,
        temperature: c_float,
        top_k: c_int,
        top_p: c_float,
        min_p: c_float,
        rng: *mut u64,
    ) -> c_int;

    pub fn ds4_bridge_session_save_payload(
        s: *mut ds4_bridge_session,
        path: *const c_char,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_session_load_payload(
        s: *mut ds4_bridge_session,
        path: *const c_char,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_session_load_payload_range(
        s: *mut ds4_bridge_session,
        path: *const c_char,
        offset: u64,
        length: u64,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_snapshot_create(
        out: *mut *mut ds4_bridge_snapshot,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_snapshot_free(snap: *mut ds4_bridge_snapshot);

    pub fn ds4_bridge_snapshot_len(snap: *const ds4_bridge_snapshot) -> u64;

    pub fn ds4_bridge_session_save_snapshot(
        s: *mut ds4_bridge_session,
        snap: *mut ds4_bridge_snapshot,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_session_load_snapshot(
        s: *mut ds4_bridge_session,
        snap: *const ds4_bridge_snapshot,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_tokenize_text(
        m: *mut ds4_bridge_model,
        text: *const c_char,
        out: *mut i32,
        cap: c_int,
        n_out: *mut c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_tokenize_rendered_chat(
        m: *mut ds4_bridge_model,
        text: *const c_char,
        out: *mut i32,
        cap: c_int,
        n_out: *mut c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_token_text(
        m: *mut ds4_bridge_model,
        token: i32,
        out: *mut c_char,
        cap: usize,
        n_out: *mut usize,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_token_eos(m: *mut ds4_bridge_model) -> c_int;

    pub fn ds4_bridge_token_is_stop(m: *mut ds4_bridge_model, token: i32) -> c_int;

    pub fn ds4_bridge_model_id(m: *mut ds4_bridge_model) -> c_int;

    pub fn ds4_bridge_model_routed_quant_bits(m: *mut ds4_bridge_model) -> c_int;

    pub fn ds4_bridge_encode_chat_prompt(
        m: *mut ds4_bridge_model,
        system: *const c_char,
        prompt: *const c_char,
        think_mode: c_int,
        out: *mut i32,
        cap: c_int,
        n_out: *mut c_int,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_session_top_logprobs(
        s: *mut ds4_bridge_session,
        out: *mut ds4_bridge_token_score,
        k: c_int,
    ) -> c_int;

    pub fn ds4_bridge_mem_census_snap(out: *mut ds4_bridge_mem_census) -> c_int;

    pub fn ds4_bridge_mem_observe_snap(out: *mut ds4_bridge_mem_observe) -> c_int;

    pub fn ds4_bridge_mem_substrate_outstanding() -> u64;

    pub fn ds4_bridge_batch_ctx_create_fit(
        m: *mut ds4_bridge_model,
        ctx_size: c_int,
        max_seq: c_int,
        max_total_tokens: c_int,
        out: *mut *mut ds4_bridge_batch_ctx,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_batch_ctx_destroy(c: *mut ds4_bridge_batch_ctx);

    pub fn ds4_bridge_batch_ctx_max_seq(c: *mut ds4_bridge_batch_ctx) -> c_int;

    pub fn ds4_bridge_batch_ctx_seq_cap(c: *mut ds4_bridge_batch_ctx) -> c_int;

    pub fn ds4_bridge_batch_ctx_bank_snapshot(
        c: *mut ds4_bridge_batch_ctx,
        bank: i32,
        tokens: *mut i32,
        cap: i32,
        n_tokens: *mut i32,
        generation: *mut u64,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_batch_ctx_bank_save_payload(
        c: *mut ds4_bridge_batch_ctx,
        bank: i32,
        path: *const c_char,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_batch_ctx_bank_load_payload_range(
        c: *mut ds4_bridge_batch_ctx,
        bank: i32,
        path: *const c_char,
        offset: u64,
        length: u64,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;

    pub fn ds4_bridge_continuous_generate(
        c: *mut ds4_bridge_batch_ctx,
        admit: Option<unsafe extern "C" fn(*mut c_void, *mut ds4_bridge_cont_request) -> c_int>,
        on_token: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> c_int>,
        on_done: Option<
            unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const i32,
                i32,
                i32,
                *const ds4_bridge_cont_stats,
            ),
        >,
        ud: *mut c_void,
        err: *mut c_char,
        errlen: usize,
    ) -> c_int;
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ds4_bridge_cont_stats {
    pub decode_ms: f64,
    pub decode_tokens: u32,
    pub decode_steps: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ds4_bridge_cont_request {
    pub tokens: *const i32,
    pub n: i32,
    pub max_new: i32,
    pub eos: i32,
    pub user: *mut c_void,
    pub temperature: c_float,
    pub top_k: i32,
    pub top_p: c_float,
    pub min_p: c_float,
    pub seed: u64,
    pub sample_override:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    pub alive: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    pub on_admitted: Option<
        unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, c_int, c_int) -> c_int,
    >,
    pub place_bank: i32,
    pub n_cached: i32,
    pub bank_used: *mut i32,
    pub fork_bank: i32,
}
