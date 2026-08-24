//! Continuous batching over the native persistent batch context.
//!
//! The engine's rolling scheduler (mid-flight admit/evict, ragged prefill,
//! per-seq sampling) stays native; the host drives it through a `ContDriver`.
//! All callbacks run on the thread that called `continuous_generate`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::{self, NonNull};

use ds4_sys::{
    ds4_bridge_batch_ctx, ds4_bridge_batch_ctx_create_fit, ds4_bridge_batch_ctx_destroy,
    ds4_bridge_batch_ctx_max_seq, ds4_bridge_batch_ctx_seq_cap, ds4_bridge_cont_request,
    ds4_bridge_continuous_generate,
};

use crate::{Error, Model, Result};

/// `ds4_cont_request.sample_override` result encoding (`DS4_SAMPLE_OVERRIDE_*`).
pub const CONT_SAMPLE_NONE: i32 = 0;
pub const CONT_SAMPLE_GREEDY: i32 = 1;

pub fn cont_sample_token(token_id: i32) -> i32 {
    token_id + 2
}

/// One admission. `tokens` moves into the batch context and stays alive
/// until that request's `on_done` fires (the engine borrows the buffer).
#[derive(Debug, Clone)]
pub struct ContAdmit {
    pub user: usize,
    pub tokens: Vec<i32>,
    pub max_new: i32,
    /// `< 0` selects the engine/family default EOS.
    pub eos: i32,
    /// `<= 0` is greedy argmax (ignores the rest of the sampling block).
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    /// Bank id + 1 placement directive; 0 = engine's choice.
    pub place_bank: i32,
    /// Committed prefix length for a warm admit; 0 = cold.
    pub n_cached: i32,
    /// Source bank id + 1 for fork-by-copy; 0 = no fork.
    pub fork_bank: i32,
}

impl ContAdmit {
    pub fn cold(user: usize, tokens: Vec<i32>, max_new: i32) -> Self {
        Self {
            user,
            tokens,
            max_new,
            eos: -1,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
            seed: 0,
            place_bank: 0,
            n_cached: 0,
            fork_bank: 0,
        }
    }
}

/// Host half of `ds4_engine_continuous_generate`. Same contracts as the C
/// callbacks: `admit` returning `None` plus an empty active set ends the
/// loop; `on_token(false)` aborts that sequence; `on_admitted(false)`
/// cancels before prefill; `alive(false)` abandons a pending admission.
pub trait ContDriver {
    fn admit(&mut self) -> Option<ContAdmit>;
    fn on_token(&mut self, user: usize, token: i32) -> bool;
    fn on_done(&mut self, user: usize, tokens: &[i32], finish: i32);
    fn sample_override(&mut self, _user: usize) -> i32 {
        CONT_SAMPLE_NONE
    }
    fn alive(&mut self, _user: usize) -> bool {
        true
    }
    fn on_admitted(&mut self, _user: usize, _n_cached: i32, _n_computed: i32, _bank: i32) -> bool {
        true
    }
}

struct TrampCtx<'a> {
    driver: &'a mut dyn ContDriver,
    /// Prompt buffers the engine may still read; freed on that user's done.
    tokens_live: HashMap<usize, Vec<i32>>,
}

unsafe extern "C" fn tramp_admit(ud: *mut c_void, req: *mut ds4_bridge_cont_request) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    let admitted = catch_unwind(AssertUnwindSafe(|| t.driver.admit())).unwrap_or(None);
    let Some(a) = admitted else { return 0 };
    let user = a.user;
    let entry = t.tokens_live.entry(user).or_default();
    *entry = a.tokens;
    let r = &mut *req;
    r.tokens = entry.as_ptr();
    r.n = entry.len() as i32;
    r.max_new = a.max_new;
    r.eos = a.eos;
    r.user = user as *mut c_void;
    r.temperature = a.temperature;
    r.top_k = a.top_k;
    r.top_p = a.top_p;
    r.min_p = a.min_p;
    r.seed = a.seed;
    r.sample_override = Some(tramp_sample_override);
    r.alive = Some(tramp_alive);
    r.on_admitted = Some(tramp_on_admitted);
    r.place_bank = a.place_bank;
    r.n_cached = a.n_cached;
    r.bank_used = ptr::null_mut();
    r.fork_bank = a.fork_bank;
    1
}

unsafe extern "C" fn tramp_on_token(ud: *mut c_void, user: *mut c_void, token: i32) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    let cont = catch_unwind(AssertUnwindSafe(|| t.driver.on_token(user as usize, token)))
        .unwrap_or(false);
    i32::from(cont)
}

unsafe extern "C" fn tramp_on_done(
    ud: *mut c_void,
    user: *mut c_void,
    tokens: *const i32,
    n: i32,
    finish: i32,
) {
    let t = &mut *(ud as *mut TrampCtx);
    let user = user as usize;
    let toks: &[i32] = if tokens.is_null() || n <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(tokens, n as usize)
    };
    let _ = catch_unwind(AssertUnwindSafe(|| t.driver.on_done(user, toks, finish)));
    t.tokens_live.remove(&user);
}

unsafe extern "C" fn tramp_sample_override(ud: *mut c_void, user: *mut c_void) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    catch_unwind(AssertUnwindSafe(|| t.driver.sample_override(user as usize)))
        .unwrap_or(CONT_SAMPLE_NONE)
}

unsafe extern "C" fn tramp_alive(ud: *mut c_void, user: *mut c_void) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    i32::from(catch_unwind(AssertUnwindSafe(|| t.driver.alive(user as usize))).unwrap_or(true))
}

unsafe extern "C" fn tramp_on_admitted(
    ud: *mut c_void,
    user: *mut c_void,
    n_cached: c_int,
    n_computed: c_int,
    bank: c_int,
) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    i32::from(
        catch_unwind(AssertUnwindSafe(|| {
            t.driver
                .on_admitted(user as usize, n_cached, n_computed, bank)
        }))
        .unwrap_or(true),
    )
}

pub struct BatchCtx<'m> {
    raw: NonNull<ds4_bridge_batch_ctx>,
    _model: PhantomData<&'m Model>,
    _not_send: PhantomData<*const ()>,
}

impl Model {
    /// `ds4_batch_ctx_create_fit`: `max_seq` is a cap; the engine sizes the
    /// bank count down to the memory budget. Read the width back with
    /// [`BatchCtx::max_seq`].
    pub fn batch_ctx_fit(
        &self,
        ctx_size: i32,
        max_seq: i32,
        max_total_tokens: i32,
    ) -> Result<BatchCtx<'_>> {
        let mut raw = ptr::null_mut();
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_batch_ctx_create_fit(
                self.raw_ptr(),
                ctx_size,
                max_seq,
                max_total_tokens,
                &mut raw,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_batch_ctx_create_fit returned NULL".into(),
        })?;
        Ok(BatchCtx {
            raw,
            _model: PhantomData,
            _not_send: PhantomData,
        })
    }
}

impl BatchCtx<'_> {
    pub fn max_seq(&self) -> i32 {
        unsafe { ds4_bridge_batch_ctx_max_seq(self.raw.as_ptr()) }
    }

    pub fn seq_cap(&self) -> i32 {
        unsafe { ds4_bridge_batch_ctx_seq_cap(self.raw.as_ptr()) }
    }

    /// Runs the engine's rolling loop until the active set is empty and
    /// `driver.admit()` returns `None`.
    pub fn continuous_generate(&mut self, driver: &mut dyn ContDriver) -> Result<()> {
        let mut t = TrampCtx {
            driver,
            tokens_live: HashMap::new(),
        };
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_continuous_generate(
                self.raw.as_ptr(),
                Some(tramp_admit),
                Some(tramp_on_token),
                Some(tramp_on_done),
                &mut t as *mut TrampCtx as *mut c_void,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        let _ = &t;
        Ok(())
    }
}

impl Drop for BatchCtx<'_> {
    fn drop(&mut self) {
        unsafe { ds4_bridge_batch_ctx_destroy(self.raw.as_ptr()) }
    }
}
