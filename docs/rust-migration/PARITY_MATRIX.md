# Parity matrix

“Rust works” is not `cargo build`. A host change is accepted only
when all five axes that it can affect are green. CUDA optimization
commits already require a correctness proof **and** a speed proof
(`AGENT.md`). The same bar applies to migration commits.

## Five axes

| Contract | Requirement | Oracle |
|---|---|---|
| Numerical | logits / fixtures stay inside existing tolerances | family CUDA fixtures, `test-mmq-parity`, `test-model-family-kernels`, CPU-vs-GPU refs where they exist |
| Token | deterministic path emits the same token ids | greedy / temp-0 benches, tokenizer goldens, motif/solar sentinels |
| KV | save / restore / reuse / prefix / eviction identical | KVC 4-way matrix; `ds4_session_save_payload` / load; bank restore |
| Wire/API | schema, event automaton, route engagement, semantic equivalence | `docs/ds4-api-surface-matrix.md` + `ds4_test --server` |
| Performance | no unexplained regression vs C on GB10 | [BASELINE.md](BASELINE.md) thresholds; AB/BA or ABBA |

Live sampled HTTP bodies are **not** a byte oracle. Continuous temp-0
can jitter. Live gates check schema, automata, routing, and semantic
equivalence — the same rule the API matrix already records.

Do not refresh goldens to hide a Rust miss.

## Phase gates

### Phase 1 — workspace + FFI skeleton

- `cargo check -p ds4-sys` (and empty member crates) succeeds
- Header exists; no bindgen of `ds4.h`
- No production binary replaced

### Phase 3 — FFI shadow (`ds4-rs` / `ds4-bench-rs`)

Same model, same arguments, C vs Rust wrapper around the **same**
C core:

| Check | Pass |
|---|---|
| Token output | identical on deterministic path |
| KV behavior | save/load round-trip matches C |
| Prefill / decode tok/s | inside BASELINE.md allowances |
| Memory | host RSS / GPU resident inside allowances |

If FFI alone regresses performance, **stop**. Do not port subsystems
on a dirty seam.

### Phase 4 — KV store

Preserve magic `KVC`, file version `1`, payload ABI `2`
(`ds4_kvstore.c`), 48-byte fixed header, endianness, key = rendered
byte prefix SHA, eviction and prefix policy, trailer hooks.

Hard gate (all four):

```text
C save    → Rust load
Rust save → C load
Rust save → Rust load
C save    → C load
```

No new checkpoint format.

### Phase 5 — web utility

Resource ownership (`OwnedFd`, `TcpStream`, `Child`, `Vec<u8>`)
without changing the subprocess/search contract. No Tokio.

### Phase 6 — distributed

Explicit integer codecs. Wire bytes match C. Do not serialize
`#[repr(C)]` Rust structs.

Runtime (blocking, one WORK at a time): CLI/option error strings,
route-plan search order, HELLO register + stale prepend, WORK
validate strings, RESULT logits/hidden. Not yet: pipelined prefetch
(`DS4_DIST_WORKER_PREFETCH_DEPTH`), worker-to-worker relay threads,
SNAPSHOT_* / DSV4 gather, `ds4_dist_session_*` wired through `ds4.c`.

### Phase 7 — server shadow

All four surfaces:

```text
POST /v1/chat/completions
POST /v1/completions
POST /v1/messages
POST /v1/responses
```

plus `/v1/models`, `/v1/stats`, `/metrics`.

Preserve ID formats (including the documented Anthropic
`chatcmpl-N` quirk), `finish_reason` mapping, native error
envelopes, stream event order, continuation / trust-domain
semantics, chunked bodies, typed schema-format refusal (v0.6.3).

Replay the `ds4_test --server` fixture inventory listed in the API
matrix. Live gates follow that document, not byte dumps.

### Phase 8 — `ds4.c` leaves

| Slice | Gate |
|---|---|
| Model metadata | enum/shape values match `g_ds4_shape` / catalog |
| GGUF catalog | mmap; no full-file `Vec<u8>`; split-GGUF identity |
| Tokenizer | encode/decode/special/template/stop exact token ids |
| Session | ownership + sync/eval/rewind/payload |
| Backend dispatch | CUDA path unchanged; no extra dynamic dispatch on the hot path |

### Proof harness (any phase that can touch CUDA execution)

```text
make proof-cuda-smoke
make proof-cuda-long
make proof-cuda-opp-c
```

must be runnable on the Rust execution path before that path is
default. Long-context captured-vs-eager stays a release gate.

### Phase 9 / split

See work instruction §26. `SPLIT_READINESS.md` is written only when
Runtime, Correctness, Performance, and Architecture (`unsafe` confined
to `ds4-sys` / native wrappers) are all green.

```bash
rg -n 'unsafe \{' crates/
```

Almost every hit must sit in `crates/ds4-sys` or a native adapter.

## Family regression set (must stay green)

Makefile variables, not flags:

| Family | Targets |
|---|---|
| Motif-3 | `test-motif3-{loader,reference,tokenizer,cuda,resident,batch}` |
| Solar | `test-solar-{loader,tokenizer,forward,session,kda,...}` |
| K-EXAONE | `test-exaone-{ref,kernels,batch}` |
| dots3-note | `test-dots3-{loader,tokenizer,reference,resident,batch}` |
| Shared | `test-model-family-kernels`, `test-mmq-parity` |

Resident tests load real weights. Use tmux + `scripts/guarded-run.sh`.
One resident model at a time.

## Performance protocol

Same host, same artifact, clocks recorded.

Minimum order:

```text
C → Rust → Rust → C
```

or ABBA. Publish SHA + GGUF identity + command/env with the numbers.
Unexplained 2–3% is a fail even if it sits inside the provisional
percent table.

## Status of this matrix

Track per-subsystem color in [STATUS.md](STATUS.md). This file is
the definition of the colors; STATUS is the current paint.
