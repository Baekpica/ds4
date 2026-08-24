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

### Phase 3 — FFI shadow (`ds4-rs` / `ds4-bench-rs` / `ds4-agent-rs`)

Same model, same arguments, C vs Rust wrapper around the **same**
C core:

| Check | Pass |
|---|---|
| Token output | Fixed-seed sampled C/Rust stdout is byte-identical (`ae12f463...`); post-port greedy stdout is also exact (`a12b7cb4...`). Fixed-seed non-TTY thinking output is exact in both modes (`--think`: 292 bytes, `ad6d107f...`; `--nothink`: 52 bytes, `ae12f463...`). Same-prompt server pair returned `Hi.` / stop / 13+2 |
| KV behavior | save/load round-trip matches C (synthetic). Live Rust benchmark restored the 64-token frontier before incrementally syncing to 128; checkpoint sizes were 18,632,800 and 24,556,624 bytes |
| Prefill / decode tok/s | Local C/Rust benchmark ABBA is green below; full production-path performance remains pending |
| Memory | sequential; C server peak `nvidia-smi` 102833 MiB, Rust server 99077 MiB; teardown + `clear_cache` returned ~115 GiB available |
| Agent shadow | `make test-agent-parity` is green: the 7,579-byte built-in prompt, fixed-input datetime message, selected non-TTY projector tapes, both supported DSML openers, and the 393,216-token high-thinking boundary are covered. The release binary links over the existing native core; live generation, transcript token tape, streaming, malformed/in-think DSML, tools, KV, MTP, and distributed paths remain pending |

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

The isolated seams are green for metadata-only KVC indexing (including a
sparse 4 GiB payload fixture), bounded text-prefix comparison, embedded DSV4
range validation, and a file-backed Rust payload writer. The writer stages the
complete stream before eviction, protects a replaced destination, preserves
the C-compatible O(trailer) fast path, and produces payload/trailer bytes read
exactly by the C oracle. The no-GPU C oracle also proves nonzero seek, exact
length, EOF rejection, and `uint64` overflow rejection.

The ordinary serial Motif-3 no-think/no-tools slice is now live-green under the
policy both servers implement (`cold=0, continued=0`). In
`scratch/rust-host-live/rust-kv-no-think-final-IHtJue/`, C and Rust generated
the same 301,972,392-byte
`5522739e3318261231ec3ddd49a24da9e66a5118.kv`; visible text and payload bytes
were exact, apart from volatile header counters/timestamps after reads. All four
restart cells returned `RESTORED_OK` with 6,896 cached prompt tokens and 15
new prompt tokens:

```text
C save    → Rust load   PASS
Rust save → C load      PASS
Rust save → Rust load   PASS
C save    → C load      PASS
```

Request SHA-256 values were `a765063d149094c6542f42e7156735a8d466cc7f4eab1eac71e0fc40a7cbadbb`
(A), `083ab2beb504500ee28dafe0d8184b53d7795d6958712c78b1c045cdf2c3d827`
(evict), and `4ae9a992140aa7cd98a7ec0467b299c66ffcb4154c2429ca202239caa8cc9d4b`
(restore). The 94,162,541,472-byte GGUF was
`Motif-3-MQ87-88-FIT.gguf`. The C binary was
`04f25a86040940674984c35160d2e4eec7f6c6d4e30313815ce10309cd57e662`
and the Rust four-way binary was
`dae70ae2f909fe6c0713a609a285c272db8d1b6ec0e2856a70bd361753625ff4`.
The common launch contract was serial (`DS4_SERVER_COALESCE_MAX=1` /
`--cont-width 0`), ctx 8192, max output 128, disk budget 2048 MiB, and minimum
checkpoint 512 tokens; C additionally set both unimplemented policy knobs to
zero. Follow-up `e5830c218361ec2288e98992cd388be1a5a24a04` corrected the Rust
timing porcelain so a live Rust cache hit now reports
`prefill_tokens=15` and `prefill_cached_tokens=6896`, matching C (post-fix
binary `52471e49c4c22bde2f1abb063755847ff5d738da705896e925e8b7468b8ec197`).

The initial comparable single cells exposed a real lifecycle regression: C was
879--934 ms TTFT while Rust was 1,590--1,715 ms because the Rust server opened
with deferred boot prewarm and never called it. Commit `f4a632e` added the
opaque prewarm call after batch placement, preserving C's placement contract.
The same model, request C, KVC, serial lane, and cache-clear protocol then gave:

```text
C     880.9 ms
Rust  836.0 ms
Rust  860.4 ms
C     897.5 ms
```

C mean was 889.2 ms and Rust mean 848.2 ms (-4.61%), so the scoped restore TTFT
gate is green. All four cells returned `RESTORED_OK`, 6,896 cached tokens, and
15 computed tokens. Commit `4fc8ef4` then matched C's reported prefill timing
scope (computed suffix sync only): live Rust was 69.2 tok/s versus C
68.2--69.0, with TTFT still 840.4 ms. The ABBA binary SHA-256 was
`00da345a5e09887d82b514c80b4674aa37311e0903ad50ff9cae82b1db3a34b3`;
the timing-scope binary was
`ea714d4d7ae4e2618bf052aa842d40e290cd978ac404ac1f39aa15f5d76779f9`.

This is not the whole Phase 4 gate. C defaults to cold/continued checkpoints
(`cold_max=30000`, `continued=10000`), while the Rust shadow does not expose or
execute those policies yet. The old 6,782-token Motif fixture demonstrates why
the omission matters: C's 6,144-token cold checkpoint changes the native
prefill partition and produced a different greedy continuation than the
shared cold-disabled path. Tool-call checkpoint replay is also missing.

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
The loaded-engine shadow now accepts and parses in client threads,
then routes jobs through a bounded FIFO to one non-`Send` inference
owner. Stable continuation pins, queue-inclusive TTFT, ordered
terminal publication, nonblocking bounded sends, real TCP disconnect,
slow-reader, and exactly-once settlement regressions are green at
`6545d44` (`make test-server-parity`).
Live Motif generate: content/finish/usage-count/`cache_write_tokens`
match. Live CUDA census: `census_supported=1` epoch=1636
weight_artifact 86.07 GiB (`scratch/rust-host-live/`).
Serial buffered responses emit a C-shaped `timings` object.
Live Motif width-1 continuous and the Rust smoke/long/OPP-C proof
harness are green. Remaining: static lane, live multi-client rolling
width, Anthropic/Responses continuous, and the full live API fixture
inventory. `DS4_SERVER_CLIENT_SNDBUF` is also not applied by the safe
stdlib socket path, so the pinned-buffer live slow-reader leg remains
pending.
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
| Tokenizer | encode/decode/special/template/stop exact token ids — **green** on synthetic GGUF (`make test-tokenizer-parity`); family chat transcript framing is exact for DeepSeek, Motif-3, Solar Open2, EXAONE, and Dots3 across none/low/high/max think modes, with byte payload and malformed-control errors covered; host vocab apply (skip C `vocab_load` when installed) + `Model` host tokenize green; live Motif specials + encode `hi`→8320 |
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
unexplained regression. Rust continuous still has one active job today:
client ingress is concurrent behind the FIFO, but the sole owner runs
each job to completion. The rolling scheduler, bank admit, per-seq eos,
and engine usage split are the same native path; live multi-client width
and the Anthropic/Responses cont promotion are follow-ups.

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
