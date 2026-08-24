//! Safe Model / Session wrappers around `ds4-sys`, plus the host-owned
//! GGUF shape catalog, tensor inventory, bind plan, validate, vocab,
//! bind lookup, layout, tokenizer, session ledger, DSV4 payload prefix,
//! and live memgov census snapshot (Phase 8).
//!
//! `unsafe` is confined to the FFI calls and the mmap adapter in this
//! crate. Application crates (`ds4-cli`, `ds4-server`, …) must not call
//! `ds4-sys` directly.

mod batch;
mod bind;
mod gguf;
mod identify;
mod layout;
mod mapped;
mod mem;
mod payload;
mod session;
mod shape;
mod tensors;
mod tok;
mod validate;

pub use bind::{
    bind_dspark_names, bind_mtp_names, bind_names, catalog_from_bind_name,
    dots3_layer_is_full_attention, dump_bind_check_oracle, dump_bind_dspark_shape,
    dump_bind_lookup_tapes, dump_bind_match_oracle, dump_bind_mtp_shape, dump_bind_names,
    dump_bind_names_shape, dump_bind_names_variant, dump_bind_support, expected_compress_ratio,
    host_bind_lookup, match_plans, solar_layer_is_gqa, variant_from_bind_name, BindError,
    BindName, BindNeed,
    BindPlan, BindSlot, HostBindLook, SupportCatalog, DSPARK_MARKOV_RANK, DSPARK_N_LAYER,
    HOST_BIND_MISS,
};
pub use batch::{
    cont_sample_token, BatchCtx, ContAdmit, ContDriver, CONT_SAMPLE_GREEDY, CONT_SAMPLE_NONE,
};
pub use gguf::{GgufError, GgufFile};
pub use identify::{dump_parse, identify_file, identify_gguf, Identified, IdentifyError};
pub use mem::{snapshot_mem, MemCell as HostMemCell, MemCensus, MemObserve, MemSnap, MEMC_COUNT, MEMD_COUNT};
pub use layout::{
    dump_expected_dspark_shape, dump_expected_layouts, dump_expected_layouts_shape,
    dump_expected_layouts_variant, dump_expected_mtp_shape, dump_expected_support,
    dump_layout_check_tapes, expected_dspark_layouts, expected_layouts, expected_mtp_layouts,
    validate_dspark_layouts, validate_layouts, validate_mtp_layouts, validate_support_layouts,
    LayoutError, LayoutSpec, TypeClass,
};
pub use payload::{
    dump_cmd as payload_dump_cmd, dump_script as payload_dump_script, encode_fields, parse_prefix,
    put_u32, tail as payload_tail, HostPrefix, PayloadError, PayloadLayout, HEADER_BYTES,
    LAYOUT_DOTS3, LAYOUT_EXAONE, LAYOUT_MOTIF3, LAYOUT_SOLAR, MAGIC as PAYLOAD_MAGIC,
    U32_FIELDS as PAYLOAD_U32_FIELDS, VERSION as PAYLOAD_VERSION,
};
pub use shape::{
    dump_oracle, route_architecture, select_shape_from_metadata, shape_for_variant, ArchRoute,
    DeepSeekDims, ModelFamily, Shape, Variant, SHAPE_DOTS3_NOTE_PREV, SHAPE_FLASH,
    SHAPE_KEXAONE_236B, SHAPE_MOTIF3, SHAPE_PRO, SHAPE_SOLAR_OPEN2_250B,
};
pub use session::{
    dump_cmd as session_dump_cmd, RewriteKind, SessionBackend, SessionLedger, SyncPlan,
};
pub use tok::{dump_cmd, dump_vocab_apply_tapes, ChatThinkMode, TokError, Vocab};
pub use validate::{
    dump_validate, host_compress_ratios, validate_file, validate_gguf, ValidateError,
};
pub use tensors::{
    apply_host_dir, consume_host_dir, dump_apply_tapes, dump_consume_tapes, dump_nbytes_table,
    dump_sibling_script,
    model_split_sibling_path, tensor_nbytes, tensor_type_name, TensorError, TensorInfo,
    TensorInventory,
};

use std::ffi::CString;
use std::marker::PhantomData;
use std::os::raw::c_char;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt as _;
use std::path::Path;
use std::ptr::{self, NonNull};

use ds4_sys::{
    ds4_bridge_bind_plan, ds4_bridge_bind_plan_check, ds4_bridge_bind_slot,
    ds4_bridge_encode_chat_prompt, ds4_bridge_eval, ds4_bridge_model, ds4_bridge_model_free,
    ds4_bridge_model_id, ds4_bridge_model_routed_quant_bits,
    ds4_bridge_model_open, ds4_bridge_model_open_options, ds4_bridge_session,
    ds4_bridge_session_argmax, ds4_bridge_session_argmax_excluding,
    ds4_bridge_session_create,
    ds4_bridge_session_exaone_rewind_span, ds4_bridge_session_free,
    ds4_bridge_session_generation, ds4_bridge_session_invalidate,
    ds4_bridge_session_load_payload, ds4_bridge_session_load_payload_range,
    ds4_bridge_session_prefill_cap,
    ds4_bridge_session_rewind, ds4_bridge_session_sample,
    ds4_bridge_session_load_snapshot, ds4_bridge_session_save_payload,
    ds4_bridge_session_save_snapshot, ds4_bridge_session_sync,
    ds4_bridge_session_top_logprobs, ds4_bridge_shard, ds4_bridge_snapshot,
    ds4_bridge_snapshot_create, ds4_bridge_snapshot_free, ds4_bridge_snapshot_len,
    ds4_bridge_token_score,
    ds4_host_bind_look, ds4_host_bind_map, ds4_host_shape, ds4_host_str, ds4_host_tensor,
    ds4_host_tensor_dir, ds4_host_vocab,
    DS4_BRIDGE_BACKEND_CPU,
    DS4_BRIDGE_BACKEND_CUDA,
    DS4_BRIDGE_BACKEND_METAL, DS4_BRIDGE_MAX_DIMS,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenScore {
    pub id: i32,
    pub logit: f32,
    pub logprob: f32,
}

pub struct Model {
    raw: NonNull<ds4_bridge_model>,
    family: ModelFamily,
    backend: Backend,
    inventory: TensorInventory,
    bind_plan: BindPlan,
    vocab: Vocab,
    _not_send: PhantomData<*const ()>,
}

struct FfiBindMap {
    _names: Vec<CString>,
    looks: Vec<ds4_host_bind_look>,
    raw: ds4_host_bind_map,
}

fn pack_host_bind_map(plan: &BindPlan) -> Result<FfiBindMap> {
    let mut names = Vec::with_capacity(plan.slots.len());
    for s in &plan.slots {
        names.push(CString::new(s.name.as_str()).map_err(|_| Error {
            code: 1,
            message: "bind slot name contains NUL".into(),
        })?);
    }
    let looks: Vec<ds4_host_bind_look> = plan
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| ds4_host_bind_look {
            name: names[i].as_ptr(),
            required: u32::from(s.need.required()),
            found: u32::from(s.tensor.is_some()),
            index: s.index.unwrap_or(u32::MAX),
        })
        .collect();
    let raw = ds4_host_bind_map {
        n: looks.len() as u32,
        v: looks.as_ptr(),
    };
    Ok(FfiBindMap {
        _names: names,
        looks,
        raw,
    })
}

impl FfiBindMap {
    fn as_c(&mut self) -> *const ds4_host_bind_map {
        self.raw.v = self.looks.as_ptr();
        &self.raw
    }
}

struct FfiBindPlan {
    _names: Vec<CString>,
    _paths: Vec<CString>,
    slots: Vec<ds4_bridge_bind_slot>,
    shards: Vec<ds4_bridge_shard>,
    plan: ds4_bridge_bind_plan,
}

fn pack_bind_plan(plan: &BindPlan, inventory: &TensorInventory) -> Result<FfiBindPlan> {
    let mut names = Vec::with_capacity(plan.slots.len());
    for s in &plan.slots {
        names.push(CString::new(s.name.as_str()).map_err(|_| Error {
            code: 1,
            message: "bind slot name contains NUL".into(),
        })?);
    }
    let mut paths = Vec::with_capacity(inventory.shards.len());
    for sh in &inventory.shards {
        paths.push(
            CString::new(sh.path.to_string_lossy().into_owned()).map_err(|_| Error {
                code: 1,
                message: "shard path contains NUL".into(),
            })?,
        );
    }
    let slots: Vec<ds4_bridge_bind_slot> = plan
        .slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let t = s.tensor.as_ref();
            let mut dim = [0u64; DS4_BRIDGE_MAX_DIMS];
            if let Some(t) = t {
                dim.copy_from_slice(&t.dim);
            }
            ds4_bridge_bind_slot {
                name: names[i].as_ptr(),
                required: u32::from(s.need.required()),
                ndim: t.map(|t| t.ndim).unwrap_or(0),
                dim,
                r#type: t.map(|t| t.typ).unwrap_or(0),
                rel_offset: t.map(|t| t.rel_offset).unwrap_or(0),
                abs_offset: t.map(|t| t.abs_offset).unwrap_or(0),
                bytes: t.map(|t| t.bytes).unwrap_or(0),
                shard: t.map(|t| t.shard).unwrap_or(0),
                found: u32::from(t.is_some()),
            }
        })
        .collect();
    let shards: Vec<ds4_bridge_shard> = inventory
        .shards
        .iter()
        .enumerate()
        .map(|(i, sh)| ds4_bridge_shard {
            path: paths[i].as_ptr(),
            size: sh.size,
            base: sh.base,
        })
        .collect();
    let c_plan = ds4_bridge_bind_plan {
        n_slots: slots.len() as u32,
        slots: slots.as_ptr(),
        n_shards: shards.len() as u32,
        shards: shards.as_ptr(),
        data_pos: inventory.data_pos,
        alignment: inventory.alignment,
        page: inventory.page,
    };
    Ok(FfiBindPlan {
        _names: names,
        _paths: paths,
        slots,
        shards,
        plan: c_plan,
    })
}

struct FfiTensorDir {
    _names: Vec<CString>,
    rows: Vec<ds4_host_tensor>,
    dir: ds4_host_tensor_dir,
}

fn pack_tensor_dir(inventory: &TensorInventory) -> Result<FfiTensorDir> {
    let mut names = Vec::with_capacity(inventory.tensors.len());
    for t in &inventory.tensors {
        names.push(CString::new(t.name.as_str()).map_err(|_| Error {
            code: 1,
            message: "tensor name contains NUL".into(),
        })?);
    }
    let rows: Vec<ds4_host_tensor> = inventory
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut dim = [0u64; DS4_BRIDGE_MAX_DIMS];
            dim.copy_from_slice(&t.dim);
            ds4_host_tensor {
                name: names[i].as_ptr(),
                ndim: t.ndim,
                dim,
                r#type: t.typ,
                rel_offset: t.rel_offset,
                abs_offset: t.abs_offset,
                bytes: t.bytes,
            }
        })
        .collect();
    let dir = ds4_host_tensor_dir {
        n: rows.len() as u32,
        v: rows.as_ptr(),
        data_pos: inventory.data_pos,
        alignment: inventory.alignment,
    };
    Ok(FfiTensorDir {
        _names: names,
        rows,
        dir,
    })
}

impl FfiTensorDir {
    fn as_c(&mut self) -> *const ds4_host_tensor_dir {
        self.dir.v = self.rows.as_ptr();
        &self.dir
    }
}

struct FfiVocab {
    tokens: Vec<ds4_host_str>,
    merges: Vec<ds4_host_str>,
    user_defined: Vec<i32>,
    raw: ds4_host_vocab,
}

fn bools_to_u8(src: &[bool; 256]) -> [u8; 256] {
    let mut out = [0u8; 256];
    for (i, b) in src.iter().enumerate() {
        out[i] = u8::from(*b);
    }
    out
}

fn pack_host_vocab(v: &Vocab) -> FfiVocab {
    let tokens: Vec<ds4_host_str> = v
        .tokens()
        .iter()
        .map(|t| ds4_host_str {
            ptr: t.as_ptr() as *const c_char,
            len: t.len() as u64,
        })
        .collect();
    let merges: Vec<ds4_host_str> = v
        .merges()
        .iter()
        .map(|t| ds4_host_str {
            ptr: t.as_ptr() as *const c_char,
            len: t.len() as u64,
        })
        .collect();
    let user_defined = v.user_defined_ids();
    let raw = ds4_host_vocab {
        n_vocab: tokens.len() as u32,
        tokens: tokens.as_ptr(),
        n_merges: merges.len() as u32,
        merges: merges.as_ptr(),
        n_user_defined: user_defined.len() as u32,
        user_defined: user_defined.as_ptr(),
        user_defined_max_len: v.user_defined_max_len(),
        user_defined_first: bools_to_u8(v.user_defined_first()),
        motif3_added_first: bools_to_u8(v.motif3_added_first()),
        bos_id: v.bos_id,
        eos_id: v.eos_id,
        system_id: v.system_id,
        eot_id: v.eot_id,
        im_start_id: v.im_start_id,
        im_content_id: v.im_content_id,
        im_end_id: v.im_end_id,
        user_id: v.user_id,
        assistant_id: v.assistant_id,
        start_of_turn_id: v.start_of_turn_id,
        end_of_turn_id: v.end_of_turn_id,
        tool_id: v.tool_id,
        reference_id: v.reference_id,
        plan_start_id: v.plan_start_id,
        plan_end_id: v.plan_end_id,
        observation_id: v.observation_id,
        sop_id: v.sop_id,
        think_start_id: v.think_start_id,
        think_end_id: v.think_end_id,
        tool_call_start_id: v.tool_call_start_id,
        tool_call_end_id: v.tool_call_end_id,
        tool_response_start_id: v.tool_response_start_id,
        tool_response_end_id: v.tool_response_end_id,
        arg_key_start_id: v.arg_key_start_id,
        arg_key_end_id: v.arg_key_end_id,
        arg_value_start_id: v.arg_value_start_id,
        latent_start_id: v.latent_start_id,
        latent_pad_id: v.latent_pad_id,
        latent_end_id: v.latent_end_id,
        tool_schema_start_id: v.tool_schema_start_id,
        tool_schema_end_id: v.tool_schema_end_id,
        dsml_id: v.dsml_id,
        dots3_endofsystem_id: v.dots3_endofsystem_id,
        dots3_endofuser_id: v.dots3_endofuser_id,
        dots3_endoftext_id: v.dots3_endoftext_id,
    };
    FfiVocab {
        tokens,
        merges,
        user_defined,
        raw,
    }
}

impl FfiVocab {
    fn as_c(&mut self) -> *const ds4_host_vocab {
        self.raw.tokens = self.tokens.as_ptr();
        self.raw.merges = self.merges.as_ptr();
        self.raw.user_defined = self.user_defined.as_ptr();
        &self.raw
    }
}

impl FfiBindPlan {
    fn as_c(&mut self) -> *const ds4_bridge_bind_plan {
        self.plan.slots = self.slots.as_ptr();
        self.plan.shards = self.shards.as_ptr();
        &self.plan
    }
}

pub struct Session<'m> {
    raw: NonNull<ds4_bridge_session>,
    host: SessionLedger,
    _model: PhantomData<&'m Model>,
    _not_send: PhantomData<*const ()>,
}

pub struct SessionSnapshot {
    raw: NonNull<ds4_bridge_snapshot>,
    host: Option<SessionLedger>,
    _not_send: PhantomData<*const ()>,
}

impl SessionSnapshot {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        let mut err = [0u8; 256];
        let rc = unsafe {
            ds4_bridge_snapshot_create(
                &mut raw,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_snapshot_create returned NULL".into(),
        })?;
        Ok(Self {
            raw,
            host: None,
            _not_send: PhantomData,
        })
    }

    pub fn len(&self) -> u64 {
        if self.host.is_none() {
            0
        } else {
            unsafe { ds4_bridge_snapshot_len(self.raw.as_ptr()) }
        }
    }
}

impl Drop for SessionSnapshot {
    fn drop(&mut self) {
        unsafe { ds4_bridge_snapshot_free(self.raw.as_ptr()) }
    }
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

fn cstring_payload_path(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    let bytes = path.as_os_str().as_bytes();
    #[cfg(not(unix))]
    let bytes = path.to_str().ok_or_else(|| Error {
        code: 1,
        message: "payload path is not UTF-8".into(),
    })?.as_bytes();
    CString::new(bytes).map_err(|_| Error {
        code: 1,
        message: "payload path contains NUL".into(),
    })
}

/// Host-resolved DeepSeek sibling (MTP / DSpark drafter): the path CString
/// and the packed bind map must outlive the bridge open call.
struct FfiSupport {
    path: CString,
    map: FfiBindMap,
}

fn pack_support(
    kind: &str,
    path: &str,
    shape: Shape,
    resolve: impl Fn(Shape, &TensorInventory) -> BindPlan,
    validate: impl Fn(&BindPlan) -> std::result::Result<(), LayoutError>,
) -> Result<FfiSupport> {
    let inv = TensorInventory::open(std::path::Path::new(path)).map_err(|e| Error {
        code: 1,
        message: format!("{kind} tensor inventory failed: {}", e.token()),
    })?;
    let plan = resolve(shape, &inv);
    if let Some(name) = plan.missing_required().first() {
        return Err(Error {
            code: 1,
            message: format!("{kind} required tensor is missing: {name}"),
        });
    }
    validate(&plan).map_err(|e| Error {
        code: 1,
        message: format!("{kind} layout failed: {}", e.token()),
    })?;
    Ok(FfiSupport {
        path: cstring_path(path)?,
        map: pack_host_bind_map(&plan)?,
    })
}

impl Model {
    pub fn open(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
    ) -> Result<Self> {
        Self::open_with_support(path, backend, n_threads, defer_boot_prewarm, None, None)
    }

    /// `mtp_path` / `dspark_path` attach the DeepSeek-only sibling support
    /// models. The host resolves each sibling's bind catalog and expected
    /// layouts, then native skips that sibling's name walk and layout check.
    pub fn open_with_support(
        path: &str,
        backend: Backend,
        n_threads: i32,
        defer_boot_prewarm: bool,
        mtp_path: Option<&str>,
        dspark_path: Option<&str>,
    ) -> Result<Self> {
        let identified = identify_gguf(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("identify failed: {}", e.token()),
        })?;
        if (mtp_path.is_some() || dspark_path.is_some())
            && identified.shape.family != ModelFamily::DeepSeek4
        {
            return Err(Error {
                code: 1,
                message: "MTP and DSpark support models are DeepSeek-only".into(),
            });
        }
        let g = GgufFile::open(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("validate failed: {}", e.token()),
        })?;
        validate_file(&g, &identified.shape).map_err(|e| Error {
            code: 1,
            message: format!("validate failed: {}", e.token()),
        })?;
        let vocab = Vocab::load(&g, identified.shape.family).map_err(|e| Error {
            code: 1,
            message: format!("vocab failed: {e}"),
        })?;
        let mut ffi_vocab = pack_host_vocab(&vocab);
        let compress = host_compress_ratios(&identified.shape);
        let ffi_shape = ds4_host_shape {
            variant: identified.shape.variant as u32,
            n_compress: compress.len() as u32,
            compress: if compress.is_empty() {
                ptr::null()
            } else {
                compress.as_ptr()
            },
        };
        let inventory = TensorInventory::open(std::path::Path::new(path)).map_err(|e| Error {
            code: 1,
            message: format!("tensor inventory failed: {}", e.token()),
        })?;
        let bind_plan = BindPlan::resolve(identified.shape, &inventory);
        if let Some(name) = bind_plan.missing_required().first() {
            return Err(Error {
                code: 1,
                message: format!("required tensor is missing: {name}"),
            });
        }
        validate_layouts(&bind_plan).map_err(|e| Error {
            code: 1,
            message: format!("layout failed: {}", e.token()),
        })?;
        let mut ffi_plan = pack_bind_plan(&bind_plan, &inventory)?;
        let mut ffi_bind = pack_host_bind_map(&bind_plan)?;
        let mut ffi_dir = pack_tensor_dir(&inventory)?;
        let mut mtp_support = match mtp_path {
            None => None,
            Some(p) => Some(pack_support(
                "mtp",
                p,
                identified.shape,
                BindPlan::resolve_mtp,
                validate_mtp_layouts,
            )?),
        };
        let mut dspark_support = match dspark_path {
            None => None,
            Some(p) => Some(pack_support(
                "dspark",
                p,
                identified.shape,
                BindPlan::resolve_dspark,
                |plan| validate_dspark_layouts(plan, DSPARK_MARKOV_RANK),
            )?),
        };
        let mut err = [0u8; 512];
        let check = unsafe {
            ds4_bridge_bind_plan_check(
                ffi_plan.as_c(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if check != 0 {
            return Err(fail(check, &err));
        }
        let c_path = cstring_path(path)?;
        let (mtp_path_ptr, mtp_bind_ptr) = match mtp_support.as_mut() {
            Some(s) => (s.path.as_ptr(), s.map.as_c()),
            None => (ptr::null(), ptr::null()),
        };
        let (dspark_path_ptr, dspark_bind_ptr) = match dspark_support.as_mut() {
            Some(s) => (s.path.as_ptr(), s.map.as_c()),
            None => (ptr::null(), ptr::null()),
        };
        let opt = ds4_bridge_model_open_options {
            model_path: c_path.as_ptr(),
            backend: backend.to_c(),
            n_threads,
            defer_boot_prewarm: i32::from(defer_boot_prewarm),
            plan: ffi_plan.as_c(),
            tensors: ffi_dir.as_c(),
            shape: &ffi_shape,
            vocab: ffi_vocab.as_c(),
            bind: ffi_bind.as_c(),
            mtp_path: mtp_path_ptr,
            dspark_path: dspark_path_ptr,
            mtp_bind: mtp_bind_ptr,
            dspark_bind: dspark_bind_ptr,
        };
        let mut raw = ptr::null_mut();
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
            family: identified.shape.family,
            backend,
            inventory,
            bind_plan,
            vocab,
            _not_send: PhantomData,
        })
    }

    pub fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn inventory(&self) -> &TensorInventory {
        &self.inventory
    }

    pub fn bind_plan(&self) -> &BindPlan {
        &self.bind_plan
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub(crate) fn raw_ptr(&self) -> *mut ds4_bridge_model {
        self.raw.as_ptr()
    }

    pub fn backend(&self) -> Backend {
        self.backend
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
        let prefill = unsafe { ds4_bridge_session_prefill_cap(raw.as_ptr()) };
        let host_backend = match self.backend {
            Backend::Cpu => SessionBackend::Cpu,
            Backend::Cuda | Backend::Metal => SessionBackend::Cuda,
        };
        Ok(Session {
            raw,
            host: SessionLedger::new(
                self.family,
                host_backend,
                ctx_size,
                prefill.max(0) as u32,
            ),
            _model: PhantomData,
            _not_send: PhantomData,
        })
    }

    pub fn model_id(&self) -> i32 {
        unsafe { ds4_bridge_model_id(self.raw.as_ptr()) }
    }

    pub fn routed_quant_bits(&self) -> i32 {
        unsafe { ds4_bridge_model_routed_quant_bits(self.raw.as_ptr()) }
    }

    pub fn token_eos(&self) -> i32 {
        self.vocab.engine_eos()
    }

    pub fn token_is_stop(&self, token: i32) -> bool {
        self.vocab.is_stop(token)
    }

    /// CLI chat-template encode through the engine (`ds4_encode_chat_prompt`):
    /// exact C `-p` prompt-token parity for the proof harness.
    pub fn encode_chat_prompt(
        &self,
        system: Option<&str>,
        prompt: &str,
        think_mode: i32,
    ) -> Result<TokenBuffer> {
        let c_system = match system {
            Some(s) => Some(CString::new(s).map_err(|_| Error {
                code: 1,
                message: "system contains NUL".into(),
            })?),
            None => None,
        };
        let c_prompt = CString::new(prompt).map_err(|_| Error {
            code: 1,
            message: "prompt contains NUL".into(),
        })?;
        // BPE merges only shrink and specials add a bounded prefix.
        let cap = prompt.len() + system.map_or(0, str::len) + 256;
        let mut out = vec![0i32; cap];
        let mut n_out = 0i32;
        let mut err = [0u8; 256];
        let rc = unsafe {
            ds4_bridge_encode_chat_prompt(
                self.raw.as_ptr(),
                c_system.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                c_prompt.as_ptr(),
                think_mode,
                out.as_mut_ptr(),
                cap as i32,
                &mut n_out,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        out.truncate(n_out.max(0) as usize);
        Ok(TokenBuffer::from_tokens(out))
    }

    pub fn tokenize_text(&self, text: &str) -> Result<TokenBuffer> {
        Ok(TokenBuffer::from_tokens(self.vocab.encode_text(text)))
    }

    pub fn tokenize_rendered_chat(&self, text: &str) -> Result<TokenBuffer> {
        Ok(TokenBuffer::from_tokens(self.vocab.encode_rendered_chat(text)))
    }

    pub fn token_text(&self, token: i32) -> Result<Vec<u8>> {
        Ok(self.vocab.token_text(token))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { ds4_bridge_model_free(self.raw.as_ptr()) }
    }
}

impl Session<'_> {
    pub fn host(&self) -> &SessionLedger {
        &self.host
    }

    pub fn generation(&self) -> u64 {
        self.host.generation
    }

    pub fn last_plan(&self, tokens: &[i32]) -> SyncPlan {
        let span = unsafe { ds4_bridge_session_exaone_rewind_span(self.raw.as_ptr()) };
        self.host.plan_sync(tokens, span)
    }

    pub fn sync(&mut self, tokens: &TokenBuffer) -> Result<()> {
        if tokens.len() > i32::MAX as usize {
            return Err(Error {
                code: 1,
                message: "token buffer exceeds i32 length".into(),
            });
        }
        let plan = self.last_plan(tokens.as_slice());
        if plan.bounds {
            return Err(Error {
                code: 1,
                message: "prompt exceeds context".into(),
            });
        }
        if plan.fence {
            return Err(Error {
                code: 1,
                message: "whole-prompt prefill fenced".into(),
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
        self.host.commit_sync(tokens.as_slice(), &plan);
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
        self.host.commit_eval(token);
        Ok(EvalResult { pos: self.host.pos() })
    }

    pub fn rewind(&mut self, pos: i32) {
        unsafe { ds4_bridge_session_rewind(self.raw.as_ptr(), pos) };
        self.host.rewind(pos);
    }

    pub fn invalidate(&mut self) {
        unsafe { ds4_bridge_session_invalidate(self.raw.as_ptr()) };
        self.host.invalidate();
    }

    pub fn rewrite_from_common(&self, prompt: &[i32], common: i32) -> RewriteKind {
        self.host.rewrite_from_common(prompt, common)
    }

    pub fn native_generation(&self) -> u64 {
        unsafe { ds4_bridge_session_generation(self.raw.as_ptr()) }
    }

    pub fn argmax(&self) -> i32 {
        unsafe { ds4_bridge_session_argmax(self.raw.as_ptr()) }
    }

    pub fn argmax_excluding(&self, excluded_id: i32) -> i32 {
        unsafe { ds4_bridge_session_argmax_excluding(self.raw.as_ptr(), excluded_id) }
    }

    /// Post-prefill distribution head, up to `k` entries (`k` clamps to the
    /// C CLI's 128). Empty when the backend keeps no logits.
    pub fn top_logprobs(&self, k: usize) -> Vec<TokenScore> {
        const SCORE_CAP: usize = 128;
        let k = k.clamp(1, SCORE_CAP);
        let mut raw = vec![
            ds4_bridge_token_score {
                id: -1,
                logit: 0.0,
                logprob: 0.0,
            };
            k
        ];
        let n = unsafe {
            ds4_bridge_session_top_logprobs(self.raw.as_ptr(), raw.as_mut_ptr(), k as i32)
        };
        if n <= 0 {
            return Vec::new();
        }
        raw.truncate(n as usize);
        raw.into_iter()
            .map(|s| TokenScore {
                id: s.id,
                logit: s.logit,
                logprob: s.logprob,
            })
            .collect()
    }

    pub fn pos(&self) -> i32 {
        self.host.pos()
    }

    pub fn ctx(&self) -> i32 {
        self.host.ctx
    }

    pub fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        unsafe {
            ds4_bridge_session_sample(
                self.raw.as_ptr(),
                temperature,
                top_k,
                top_p,
                min_p,
                rng,
            )
        }
    }

    pub fn save_snapshot(&self, snapshot: &mut SessionSnapshot) -> Result<()> {
        snapshot.host = None;
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_save_snapshot(
                self.raw.as_ptr(),
                snapshot.raw.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        snapshot.host = Some(self.host.clone());
        Ok(())
    }

    pub fn load_snapshot(&mut self, snapshot: &SessionSnapshot) -> Result<()> {
        let saved = snapshot.host.as_ref().ok_or_else(|| Error {
            code: 1,
            message: "session snapshot is empty".into(),
        })?;
        if saved.family != self.host.family
            || saved.backend != self.host.backend
            || saved.ctx != self.host.ctx
            || saved.prefill_cap != self.host.prefill_cap
        {
            return Err(Error {
                code: 1,
                message: "session snapshot belongs to an incompatible session".into(),
            });
        }

        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_load_snapshot(
                self.raw.as_ptr(),
                snapshot.raw.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            self.host.clear_checkpoint_keep_generation();
            self.host.generation = self.native_generation();
            return Err(fail(rc, &err));
        }
        self.host = saved.clone();
        self.host.generation = self.native_generation();
        Ok(())
    }

    /// Native writes the full DSV4 file (header + tokens + GPU tail).
    /// Host re-reads the prefix and requires token identity to match the ledger.
    pub fn save_payload(&self, path: impl AsRef<Path>) -> Result<()> {
        if !self.host.valid {
            return Err(Error {
                code: 1,
                message: "session has no valid checkpoint to save".into(),
            });
        }
        let path = path.as_ref();
        let c_path = cstring_payload_path(path)?;
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_save_payload(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        let mut file = std::fs::File::open(path).map_err(|e| Error {
            code: 1,
            message: format!("failed to reopen session payload: {e}"),
        })?;
        let payload_bytes = file
            .metadata()
            .map_err(|e| Error {
                code: 1,
                message: format!("failed to measure session payload: {e}"),
            })?
            .len();
        let prefix = crate::payload::read_prefix_range(
            &mut file,
            0,
            payload_bytes,
            self.host.family,
            self.host.ctx,
        )
        .map_err(|e| Error {
            code: 1,
            message: e.to_string(),
        })?;
        let host_tok: Vec<u32> = self.host.tokens().iter().map(|&t| t as u32).collect();
        if prefix.tokens != host_tok {
            return Err(Error {
                code: 1,
                message: "host/native token mismatch".into(),
            });
        }
        Ok(())
    }

    /// Host validates the DSV4 prefix independently, then native restores the
    /// GPU/logits tail. Generation follows native (`ds4_session_load_payload`
    /// bumps it). Tokens come from the host-parsed prefix.
    pub fn load_payload(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| Error {
            code: 1,
            message: format!("failed to open session payload: {e}"),
        })?;
        let payload_bytes = file
            .metadata()
            .map_err(|e| Error {
                code: 1,
                message: format!("failed to measure session payload: {e}"),
            })?
            .len();
        let prefix = crate::payload::read_prefix_range(
            &mut file,
            0,
            payload_bytes,
            self.host.family,
            self.host.ctx,
        )
        .map_err(|e| Error {
            code: 1,
            message: e.to_string(),
        })?;
        let c_path = cstring_payload_path(path)?;
        let mut err = [0u8; 512];
        let generation_before = self.native_generation();
        let rc = unsafe {
            ds4_bridge_session_load_payload(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            let generation_after = self.native_generation();
            if generation_after != generation_before {
                self.host.clear_checkpoint_keep_generation();
            }
            self.host.generation = generation_after;
            return Err(fail(rc, &err));
        }
        self.host.apply_payload(&prefix).map_err(|e| Error {
            code: 1,
            message: e.message.to_string(),
        })?;
        self.host.generation = self.native_generation();
        Ok(())
    }

    /// Restore one DSV4 payload embedded in a larger file. Only the host
    /// prefix is read in Rust; the native loader consumes the bounded range.
    pub fn load_payload_range(
        &mut self,
        path: impl AsRef<Path>,
        offset: u64,
        length: u64,
    ) -> Result<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|e| Error {
            code: 1,
            message: format!("failed to open session payload range: {e}"),
        })?;
        let prefix = crate::payload::read_prefix_range(
            &mut file,
            offset,
            length,
            self.host.family,
            self.host.ctx,
        )
        .map_err(|e| Error {
            code: 1,
            message: e.to_string(),
        })?;
        let c_path = cstring_payload_path(path)?;
        let mut err = [0u8; 512];
        let generation_before = self.native_generation();
        let rc = unsafe {
            ds4_bridge_session_load_payload_range(
                self.raw.as_ptr(),
                c_path.as_ptr(),
                offset,
                length,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            let generation_after = self.native_generation();
            if generation_after != generation_before {
                self.host.clear_checkpoint_keep_generation();
            }
            self.host.generation = generation_after;
            return Err(fail(rc, &err));
        }
        self.host.apply_payload(&prefix).map_err(|e| Error {
            code: 1,
            message: e.message.to_string(),
        })?;
        self.host.generation = self.native_generation();
        Ok(())
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

    #[cfg(unix)]
    #[test]
    fn payload_path_preserves_non_utf8_bytes() {
        use std::os::unix::ffi::OsStringExt as _;

        let raw = b"/tmp/ds4-payload-\xff".to_vec();
        let path = std::path::PathBuf::from(std::ffi::OsString::from_vec(raw.clone()));

        let c_path = cstring_payload_path(&path).unwrap();
        assert_eq!(c_path.as_bytes(), raw);
    }

    #[cfg(unix)]
    #[test]
    fn payload_path_rejects_embedded_nul_with_specific_error() {
        use std::os::unix::ffi::OsStringExt as _;

        let path = std::path::PathBuf::from(std::ffi::OsString::from_vec(
            b"/tmp/ds4-payload-\0suffix".to_vec(),
        ));

        let err = cstring_payload_path(&path).unwrap_err();
        assert_eq!(err.code, 1);
        assert_eq!(err.message, "payload path contains NUL");
    }
}
