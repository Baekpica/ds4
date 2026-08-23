# C → Rust host migration

This directory is the working contract for moving the **production host
runtime** of `ds4-dfm` from C to Rust. It does not redesign inference,
CUDA, or the wire API.

The C implementation at tag `v0.6.3-dfm` is the golden oracle. Rust
reproduces that observable behavior. Architectural freedom belongs to
the later independent `dfm-rs` repository, not to this branch.

## Status

| Item | Value |
|---|---|
| Branch | `rust-host` |
| C golden baseline | `v0.6.3-dfm` (`516456fe35510e4fb8350396c9d88807ac1f760b`) |
| Current phase | Phase 4 KVC format/policy ported (4-way green). Session payload restore still C. |
| Default production path | C (`ds4`, `ds4-server`, `ds4-bench`, `ds4-agent`) |
| Target production path | Rust host + unchanged native CUDA/MMQ backend |
| Repo split | **not started**. Requires `SPLIT_READINESS.md` green. |

Live subsystem progress lives in [STATUS.md](STATUS.md).

## Documents

| File | Role |
|---|---|
| [BASELINE.md](BASELINE.md) | Frozen C tag, host, published Spark numbers |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Strangler shape, crate map, CUDA stays native |
| [FFI_CONTRACT.md](FFI_CONTRACT.md) | Narrow opaque ABI; what Rust must never see |
| [PARITY_MATRIX.md](PARITY_MATRIX.md) | Numerical / token / KV / wire / performance gates |
| [STATUS.md](STATUS.md) | Subsystem matrix (always current) |
| `SPLIT_READINESS.md` | Last artifact before `Baekpica/dfm-rs`. Do not create until green. |

Related frozen oracles outside this directory:

- `AGENT.md` — correctness before speed, mmap-backed loading, narrow public API, long-context captured/eager parity
- `docs/ds4-dfm-model-families.md` — family contract and published Spark evidence
- `docs/ds4-api-surface-matrix.md` — four wire surfaces and serving-lane routing

## What this migration is

Strangler replacement of the **host**: model lifecycle, session
ownership, KV store, HTTP server, distributed coordination, and
memory policy. Each step leaves a working executable.

```text
request → Rust API / scheduler / session / KV / model
                │
          narrow native ABI
                │
                ▼
          CUDA backend / VMM / CUDA Graph / MMQ / FA / MoE
                │
                ▼
               GB10
```

## What this migration is not

Do not do any of the following on `rust-host`:

- Translate `ds4.c` into one `ds4.rs`
- Rewrite CUDA/MMQ kernels or change quantization
- Change speculative/MTP semantics, tokenizer behavior, or checkpoint format
- Introduce Tokio / Axum / async schedulers before `dfm-rs` exists
- Clean up HTTP IDs, `finish_reason`, stream event order, or error envelopes
- `git merge dfm` (that makes the oracle a moving target)
- Mix a port commit with an optimization commit
- Create `Baekpica/dfm-rs` before `SPLIT_READINESS.md` is green

If `dfm` later lands a CUDA correctness or performance fix, identify
that commit, cherry-pick it onto `rust-host` on purpose, and rerun the
parity matrix. Do not absorb the whole branch.

## Phase order (fixed)

0. Freeze `v0.6.3-dfm` and write these documents (this commit).
1. Rust workspace + `native/bridge` FFI skeleton. No C port.
2. Safe `ds4-core` wrappers. `unsafe` stays in `ds4-sys`.
3. Shadow binaries `ds4-rs` / `ds4-bench-rs` calling the same C core.
4. KV store (`ds4_kvstore.c` → `crates/ds4-kv`), 4-way checkpoint matrix.
5. Web/utility (`ds4_web.c`) with std blocking I/O. No HTTP redesign.
6. Distributed runtime (`ds4_distributed.c` → `crates/ds4-dist`). Explicit codecs.
7. Server shadow by feature (wire / routing / runtime / continuation).
8. Decompose `ds4.c` (metadata → mmap GGUF catalog → tokenizer → session → dispatch).
9. Promote Rust binaries to the default names; keep C oracles until split.

CPU reference and Metal are not cut-over blockers. The primary release
gate is the DFM CUDA / Linux production path.

## Commit rule

One meaning per commit. Every migration commit records:

```text
Migration area:
C source replaced:

Correctness:
- tests ...
- parity ...

Performance:
- C ...
- Rust ...

Known remaining C dependency:
...
```

## Definition of done (pre-split)

`v0.6.3-dfm` observable behavior remains the oracle; CUDA/MMQ stays
native; Rust owns model / session / KV / server / distributed
orchestration; numerical, token, KV, wire, and GB10 performance
parity are proven. Then write `SPLIT_READINESS.md`. Only that green
commit is the `dfm-rs` genesis point.
