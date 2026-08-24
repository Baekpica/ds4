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
| Token output | Motif live: C CLI emitted `hi`; Rust CLI `--tokens 1,8320 --predict 8` emitted 9 ids. Same-prompt server pair both returned `Hi.` / stop / 13+2 |
| KV behavior | save/load round-trip matches C (synthetic). Live Rust benchmark restored the 64-token frontier before incrementally syncing to 128; checkpoint sizes were 18,632,800 and 24,556,624 bytes |
| Prefill / decode tok/s | C CLI 60.15 / 7.88; C server 86.3 / 10.6. Rust benchmark smoke printed both timings, but matched C/Rust ABBA is still pending |
| Memory | sequential; C server peak `nvidia-smi` 102833 MiB, Rust server 99077 MiB; teardown + `clear_cache` returned ~115 GiB available |

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

`route_decide` + `request_compute_needs` are the routing oracle.
The HTTP door (reader, native error envelopes, schema-format refusal,
`/v1/models` porcelain), the four JSON parsers (chat / completion /
Anthropic / Responses, tokenize/render cut off), the four tape
stream projectors plus buffered finals and incremental live DSML tool stream (OpenAI + Anthropic; Responses has no incremental tool stream in C),
enqueue/shed + job-id mint, the host-owned `/metrics` `/v1/stats`
porcelain including the memgov census format (live CUDA census +
observation via `ds4_bridge_mem_*_snap` when an engine is open;
absence-not-zero when unsupported), family
render including tool-schema / invoke reconstruct
(DSML / Motif / EXAONE / dots3 / Solar C oracle),
NULL-handle tokenize/sample FFI, the serial decode driver
(scripted tape, no GGUF), generated-message tool parse
(DSML / Hermes / dots3 / Solar C goldens), SemAccum stop /
no-tools cut / DSML `tool_calls` verdict, and finalize
`tool_calls` / `tool_use` wire, the Inc 5a/5b/5c
continuation registry (publish / resolve / hold / pin / TTL /
bank claim + serial 503/409), and corrective recovery retry
(`decode_again` / model-visible tool error + tag-completion
repair) are `make test-server-parity`.
Live Motif generate: content/finish/usage-count/`cache_write_tokens`
match. Live CUDA census: `census_supported=1` epoch=1636
weight_artifact 86.07 GiB (`scratch/rust-host-live/`).
Serial buffered responses emit a C-shaped `timings` object.
Remaining: Motif continuous-lane on the Rust server; proof harness
on the Rust execution path.
Do not improve the table.

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
| Model metadata | enum/shape values match `g_ds4_shape` / catalog — **green** (`make test-catalog-parity`) |
| GGUF catalog | mmap; no full-file `Vec<u8>`; first-shard `split.count` identity — **green** on synthetic v3; host `weights_bind` name catalog + bind-plan check/match + host tensor-dir apply (skip `parse_tensors` when installed) + host `config_validate` / apply shape+compress (skip C validate when installed) + host bind map (skip C name walk when installed) + host `weights_validate_layout` (skip C main-model layout when bind map installed) + host MTP/DSpark sibling name/layout catalogs + sibling BindPlan resolve/validate + sibling bind map FFI (native swaps the sibling map around its own open+bind window and skips that sibling's C layout check; live DSpark drafter 81/81 + Rust `--dspark` decode green) + host-table clear before sibling `model_open` green; sibling pointer assignment / CUDA upload still native |
| Tokenizer | encode/decode/special/template/stop exact token ids — **green** on synthetic GGUF (`make test-tokenizer-parity`); host vocab apply (skip C `vocab_load` when installed) + `Model` host tokenize green; live Motif specials + encode `hi`→8320 |
| Session | ownership + sync/eval/rewind/payload — **ledger + DSV4 prefix green** (`make test-session-parity`: rewrite/prefix/plan/rewind/generation + header/token tapes); GPU/logits tail / CUDA eval still native |
| Backend dispatch | CUDA path unchanged; no extra dynamic dispatch on the hot path |

### Proof harness (any phase that can touch CUDA execution)

```text
make proof-cuda-smoke
make proof-cuda-long
make proof-cuda-opp-c
make proof-rust-cuda-opp-c
```

`proof-cuda-opp-c` remains the native-drift gate against a tracked
golden. A persisted `CUDA_ARCH=sm_121` build selects the GB10 snapshot
validated against `v0.6.3-dfm`; other builds keep their original generic
snapshot and require separate architecture approval. `proof-rust-cuda-opp-c`
is the separate host-parity gate: it writes a temporary snapshot with the
current C oracle, then checks the Rust binary through the same stable runner
path. Its artifacts remain under `/tmp/ds4_proof/proof-rust-cuda-opp-c.*`
for audit; it does not rewrite or replace either tracked native golden.

Both OPP-C gates must be green before the Rust execution path becomes
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
| Solar | `test-solar-{loader,tokenizer,forward,session,kda,kda-prefill,kda-chunk,gates,kv}` |
| K-EXAONE | `test-exaone-{ref,kernels,batch}` |
| dots3-note | `test-dots3-{loader,tokenizer,resident}` |
| Shared | `test-model-family-kernels`, `test-mmq-parity` |

Resident tests load real weights. Use tmux + workspace-local
`../scripts/guarded-run.sh` (outside this repository).
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

### Recorded ABBA (2026-08-24, cross-lane)

Host SHA `9463b3c` (+ scratch runner `scratch/rust-host-live/abba.sh`),
GGUF `Motif-3-MQ87-88-FIT.gguf`, ctx 8192, cold 6782-token prompt,
114 decode tokens, temp 0, thinking disabled, sequential loads with
teardown + `clear_cache` between cells:

| Cell | Lane | prefill tok/s | decode tok/s | ttft ms | peak GPU MiB |
|---|---|---|---|---|---|
| C #1 | continuous | 633.4 | 14.8 | 10755 | 112145 |
| Rust #1 | serial | 579.8 | 15.4 | 11834 | 102357 |
| Rust #2 | serial | 577.5 | 15.5 | 11877 | 102357 |
| C #2 | continuous | 633.0 | 14.8 | 10780 | 114529 |

All four completions are byte-identical greedy text (SHA1
`4d34f24352e7`) — the Token axis passes across hosts and lanes.
Pair spreads are ≤0.4%. The −8.6% prefill / +4.4% decode delta was
attributed to the lane difference and re-measured after the
continuous port (below).

### Recorded ABBA (2026-08-24, same lane: both continuous)

Same protocol, `ds4-server-rs --cont-width 2` routing
`openai_chat_continuous`, all four completions byte-identical
(same SHA1 `4d34f24352e7`):

| Cell | Lane | prefill tok/s | decode tok/s | ttft ms |
|---|---|---|---|---|
| C #1 | continuous | 579.2 | 14.9 | 11767 |
| Rust #1 | continuous | 590.8 | 14.8 | 11521 |
| Rust #2 | continuous | 592.9 | 14.7 | 11480 |
| C #2 | continuous | 633.9 | 14.9 | 10751 |

Rust pair spread 0.4%; the C pair spread is 9.0% (579/634 — the
cross-lane run's C pair sat at 633.4/633.0), so the −2.4% Rust mean
prefill delta is inside the C cell noise; decode is within 1%. No
unexplained regression. Rust continuous is width-1 today (serial
accept): the rolling scheduler, bank admit, per-seq eos and engine
usage split are the same native path; multi-client width and the
Anthropic/Responses cont promotion are follow-ups.

### Recorded ABBA (2026-08-24, local benchmark shadow)

Host commit `a9ba7b38cbf683f03c21e2604e055356ec2a2ad2`
(`v0.6.3-dfm`), NVIDIA GB10, driver 610.43.02, CUDA 13.3. C binary
SHA-256 `3c0af3d1d860cf3d2800261d51e729c3ca96aae3d048551ed721ed717aed0f8c`;
Rust binary SHA-256
`a66d97c48755af0e54178c57a5d70765019eade45cb9a038ccfd5615c940dae3`.
Model artifact:
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
(86,720,111,488 bytes; model label `DeepSeek V4 Flash`, revision label
`0731-chat-v2`, quant label `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8`).

Order was C → Rust → Rust → C with teardown and `clear_cache` between
cells. Both binaries used the same raw prompt, frontiers 1024→2048,
`ctx_alloc=2081`, 32 decode tokens, width 1, and default packed FP8 KV
+ FP4 index; no `DS4_*` override was set. Logs and CSVs are under
`scratch/rust-host-live/bench-abba-20260824/`.

```bash
$BIN --cuda -m "$MODEL" --prompt-file tests/long_context_essay_prompt.txt \
  --ctx-start 1024 --ctx-max 2048 --ctx-alloc 2081 --step-incr 1024 \
  --gen-tokens 32 --csv "$CELL.csv"
```

| Cell | Frontier | Prefill tok/s | Decode tok/s | Steady decode tok/s | First decode token s | Host max RSS KiB | KV bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| C #1 | 1024 | 838.58 | 22.78 | 22.94 | 0.0531 | 8,972,888 | 28,482,336 |
| C #1 | 2048 | 945.37 | 23.10 | 23.10 | 0.0433 | 8,972,888 | 32,968,864 |
| Rust #1 | 1024 | 850.48 | 22.83 | 22.96 | 0.0517 | 9,024,668 | 28,482,336 |
| Rust #1 | 2048 | 953.26 | 23.10 | 23.10 | 0.0433 | 9,024,668 | 32,968,864 |
| Rust #2 | 1024 | 845.62 | 22.88 | 23.01 | 0.0518 | 9,025,144 | 28,482,336 |
| Rust #2 | 2048 | 954.29 | 22.93 | 22.92 | 0.0433 | 9,025,144 | 32,968,864 |
| C #2 | 1024 | 843.66 | 22.91 | 23.06 | 0.0523 | 8,972,584 | 28,482,336 |
| C #2 | 2048 | 947.30 | 23.10 | 23.11 | 0.0439 | 8,972,584 | 32,968,864 |

Rust/C mean ratios were 100.82% / 100.79% prefill and 100.04% /
99.63% decode at the two frontiers. Rust host max RSS was +0.58%; KV
bytes matched exactly. This passes the Phase 3 FFI-overhead thresholds.
GPU resident peak was not sampled, so this is not the full pre-split
performance gate.

## Status of this matrix

Track per-subsystem color in [STATUS.md](STATUS.md). This file is
the definition of the colors; STATUS is the current paint.
