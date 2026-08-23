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
| `ds4_bridge_model_open` | mmap-backed `ds4_engine_open` |
| `ds4_bridge_model_free` | `ds4_engine_close` |
| `ds4_bridge_session_create` | `ds4_session_create` |
| `ds4_bridge_session_free` | `ds4_session_free` |
| `ds4_bridge_session_sync` | prefix tokens → `ds4_session_sync` |
| `ds4_bridge_eval` | `ds4_session_eval` one token |
| `ds4_bridge_session_argmax` | greedy next id |
| `ds4_bridge_session_pos` | committed timeline length |

Open options stay a small C struct of scalars and `const char *`
paths (model path, backend enum, thread count, defer-prewarm). Do
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
- `ds4_batch_ctx` / reclaim / memgov ledgers
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
