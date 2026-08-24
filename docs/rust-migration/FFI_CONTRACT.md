# FFI contract

Rust talks to the existing native runtime through **one** header:
`native/bridge/ds4_bridge.h`.

```text
Rust application
    ↓
ds4-core (safe)
    ↓
ds4-sys (unsafe)
    ↓
ds4_bridge.h          ← the only stable ABI
    ↓
ds4_bridge.c
    ↓
existing ds4.h / ds4.c / ds4_cuda.cu internals
```

## Hard rules

1. Do **not** bindgen `ds4.h`, `ds4_gpu.h`, or `cuda/mmq/*.h`.
2. Do **not** expose C struct layouts to Rust. Handles are opaque.
3. Do **not** put `CUstream`, device pointers, MMQ descriptors, or
   graph execs in Rust application code.
4. `unsafe` belongs in `ds4-sys` or a tiny native adapter. Application
   crates (`ds4-server`, `ds4-cli`, `ds4-kv`, `ds4-dist`) showing
   `unsafe {` is an architecture defect.
5. Errors cross the boundary as `int` + caller-provided `char *err`
   buffer, matching the existing C session helpers. No C++ exceptions.
   No Rust panic across FFI.
6. Strings and token buffers passed into the bridge are borrowed for
   the duration of the call. The bridge must copy if it retains them.
7. Every successful `*_create` / `*_open` has exactly one `*_free`.
   Rust `Drop` is the only application-side destructor.

## Opaque handles (Phase 1 skeleton)

```c
typedef struct ds4_bridge_model ds4_bridge_model;
typedef struct ds4_bridge_session ds4_bridge_session;
```

Rust side (Phase 2):

```rust
pub struct Model {
    raw: NonNull<ds4_bridge_model>,
}

pub struct Session {
    raw: NonNull<ds4_bridge_session>,
}
```

`Session` must not outlive `Model`. Encode that in the safe wrapper
(`Session` holds a borrow or an `Arc<Model>`), not by leaking raw
pointers into callers.

Do not typedef these to `ds4_engine` / `ds4_session`. The bridge
structs may *contain* those pointers. Rust must not know.

## Initial ABI surface

Phase 1 declares this set. Later phases add functions; they do not
widen the existing ones to dump internals.

| Function | Meaning |
|---|---|
| `ds4_bridge_bind_plan_check` | host inventory + required-name table; 0 if native can consume it |
| `ds4_bridge_bind_plan_match` | host vs native slot identity (name/need/found/type/dims/offsets/bytes/shard + remap) |
| `ds4_bridge_model_open` | mmap-backed `ds4_engine_open`; optional `opt->plan` is checked first; optional `opt->tensors` replaces `parse_tensors` (owned name copies); optional `opt->shape` applies the pinned C literal + DeepSeek compress table and skips `config_validate_model`; optional `opt->vocab` applies host token/merge/specials and skips `vocab_load`; optional `opt->bind` is the host name→tensor-dir index so `model_find_tensor` skips the C name walk and the main-model `weights_validate_layout` |
| `ds4_bridge_model_free` | `ds4_engine_close` |
| `ds4_bridge_session_create` | `ds4_session_create` |
| `ds4_bridge_session_free` | `ds4_session_free` |
| `ds4_bridge_session_sync` | prefix tokens → `ds4_session_sync` |
| `ds4_bridge_eval` | `ds4_session_eval` one token |
| `ds4_bridge_session_argmax` | greedy next id |
| `ds4_bridge_session_pos` | native committed timeline (host `SessionLedger` is authoritative) |
| `ds4_bridge_session_ctx` | session context length |
| `ds4_bridge_session_rewind` | `ds4_session_rewind` |
| `ds4_bridge_session_invalidate` | `ds4_session_invalidate` |
| `ds4_bridge_session_generation` | Inc 5a content generation |
| `ds4_bridge_session_prefill_cap` | chunk cap used by the native graph |
| `ds4_bridge_session_exaone_rewind_span` | EXAONE sliding-window reuse span |
| `ds4_bridge_session_sample` | `ds4_session_sample` (caller-owned rng) |
| `ds4_bridge_session_save_payload` | path wrapper over `ds4_session_save_payload` (native writes header+tokens+GPU tail) |
| `ds4_bridge_session_load_payload` | path wrapper; `payload_bytes` = file size |
| `ds4_bridge_tokenize_text` | caller-owned `int32_t *` + cap + `n_out` |
| `ds4_bridge_tokenize_rendered_chat` | same buffer contract, special-token path |
| `ds4_bridge_token_text` | caller-owned byte buffer; C frees the malloc |
| `ds4_bridge_token_eos` | engine EOS / family EOT |
| `ds4_bridge_token_is_stop` | `ds4_token_is_stop` (1/0) |
| `ds4_bridge_model_id` | `ds4_engine_model_id` (syntax dispatch) |
| `ds4_bridge_mem_census_snap` | process-global CUDA census image (seqlock + last-stable torn cache); `supported=0` when the backend keeps no census |
| `ds4_bridge_mem_observe_snap` | typed observation (`status`/`source` + free/total/cuda_free/meminfo) |
| `ds4_bridge_mem_substrate_outstanding` | `ds4_gpu_substrate_outstanding` (0 on Metal/CPU stubs) |

Family/shape identify (`ds4-core::identify_gguf`) is host-owned: it
mmaps GGUF metadata and does **not** call `ds4_bridge_model_open`.
Tensor inventory + `-0000N-of-` shard remap (`ds4-core::TensorInventory`)
is also host-owned; `Model::open` builds that plan before the bridge
call. The family `weights_bind` name catalog (`ds4-core::BindPlan`) is
host-owned and is passed as `ds4_bridge_bind_plan` so native
`ds4_bridge_bind_plan_check` consumes it before `ds4_engine_open`.
The full host inventory (`ds4_host_tensor_dir` / `opt->tensors`) is
installed for that open. When present, native skips `parse_tensors`
and applies the host table (names are `strdup`'d; the Rust
`CString`s die when `Model::open` returns). Optional `opt->shape`
(`ds4_host_shape`) is the host `config_validate` result: native
applies the pinned `g_ds4_shape` literal plus the verified DeepSeek
compress table and skips C `config_validate_model`. Optional `opt->bind` (`ds4_host_bind_map`) is the host-resolved
name→tensor-dir index: when installed, `model_find_tensor` uses it
instead of scanning `m->tensors`. Names not in the map (MTP/DSpark
siblings) fall back to the C walk. When that map is installed, native
also skips the main-model `weights_validate_layout` because
`Model::open` already ran the host table. Host owns the DeepSeek
MTP/DSpark sibling name and expected-layout catalogs (`mtp-flash` /
`dspark-pro`) and can resolve/validate a sibling `BindPlan` against a
host inventory (`--bind-names mtp-flash --bind-plan`). Optional
`opt->mtp_path` / `opt->dspark_path` attach the DeepSeek-only sibling
support models through the same open; the optional `opt->mtp_bind` /
`opt->dspark_bind` maps are host-resolved name→index tables for THAT
sibling's tensor dir (`Model::open_with_support` runs sibling
resolve + expected-layout validation first). Native keeps them in
separate slots, swaps each into the active map only around its own
sibling open+bind window, and skips that sibling's C layout check —
sibling pointer assignment stays C. After base
`weights_bind`, native clears host tensor-dir / bind-map / vocab /
shape so a later sibling `model_open` cannot apply the base GGUF
tables. The C CLI/server
leave tensors/shape/vocab/bind and the sibling paths/maps NULL, so
the GGUF cursor walk, C validate,
C `vocab_load`, C `model_find_tensor` name walk, and C
`weights_validate_layout` (base and sibling) stay the production
default. Weight upload / VMM bind stay native. Tokenizer encode / decode / special / stop
(`ds4-core::Vocab`) is host-owned; `Model` keeps the `Vocab` so
native token-string pointers stay valid for the engine lifetime.
`--tokenize` / `--validate` still must not open the engine. Session timeline / sync plan /
rewrite / rewind / generation (`ds4-core::SessionLedger`) is host-owned;
`Session::pos` / `generation` read the ledger. The DSV4 payload prefix
(13×u32 LE header + token ids; magic `DSV4`, version 3) is host-owned
(`ds4-core::payload`); GPU / logits / family tensor tails stay native.
The Inc 5 continuation registry (`ds4-server::ContRegistry`) is
host-owned: publish / resolve / hold / pin / TTL / bank claim do not
cross FFI. Weight bind and native prefill/eval still go through the
bridge. `Model` tokenize/stop/eos/`token_text` use the host `Vocab`.
The bridge tokenize helpers remain for the C engine path.

Open options stay a small C struct of scalars, `const char *`
paths, and optional borrowed plan/tensor-dir/shape/vocab/bind pointers
(model path, backend enum, thread count, defer-prewarm, then the
optional `mtp_path` / `dspark_path` / `mtp_bind` / `dspark_bind`
sibling fields appended at the end). Do
not pass `ds4_engine_options` by value into Rust — that struct will
keep growing on the C side and is not the ABI.

Token arrays are `const int32_t *` + length. Do not export
`ds4_tokens`.

## Ownership

| Object | Allocated by | Freed by | Rust view |
|---|---|---|---|
| `ds4_bridge_model` | bridge | `ds4_bridge_model_free` | `NonNull`, `Drop` |
| `ds4_bridge_session` | bridge | `ds4_bridge_session_free` | `NonNull`, `Drop` |
| error buffer | Rust caller | Rust caller | `&mut [u8]` / `CString` scratch |
| token scratch | Rust caller | Rust caller | `&[i32]` |
| GPU tensors / graphs | native session | native session free | invisible |

The bridge must not return interior pointers into engine arenas.

## What the bridge must not grow into

These remain C-internal or later *narrow* additions with their own
review, not dump-the-header expansions:

- `ds4_gpu_*` (thousands of lines of tensor ops)
- `ds4_batch_ctx` / reclaim / governor ledgers (the `/metrics` `/v1/stats` text is host-owned; live census + observation copy through `ds4_bridge_mem_*_snap`, not a bindgen of `ds4_mem_census.h`)
- family-specific test hooks (`ds4_engine_motif3_*`, dots3 logits)
- Metal / CUDA types
- distributed pthread/socket internals
- KV store `kv_buf` / malloc arena

KV (Phase 4) and distributed (Phase 6) get their own explicit codecs
or file-format modules in Rust. They do not ride on a giant bindgen.

## Linkage

Phase 1–3 link the existing `make cuda-spark` objects plus
`native/bridge/ds4_bridge.c`. The CUDA Driver API, CUDA Runtime, and
cuBLAS stay on the native link line (`-lcuda -lcudart -lcublas`).

Rust must not add a second CUDA stack (cudarc-driven kernels, a
second context, a second VMM arena).

## Versioning

The ABI is source-stable on `rust-host`. Changing a function
signature is a dedicated commit that updates this file, the header,
`ds4-sys`, and the parity tests together.

There is no promise of binary compatibility with out-of-tree
callers. The only in-tree consumer is `ds4-sys`.

## Final CUDA-facing ABI (Phase 8/9 target)

Once the host is Rust, the remaining native exports shrink toward:

```text
backend_create
load_weights
session_create
prefill
decode
kv_save / kv_load primitives
backend_destroy
```

Until then, `ds4_bridge_*` wrapping the current `ds4_engine` /
`ds4_session` API is the correct strangler seam: same CUDA path,
no kernel rewrite.
