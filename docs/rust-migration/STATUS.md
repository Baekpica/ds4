# Migration status

Update this table in the same commit that changes a subsystem’s
state. Colors: `green` = gate passed, `yellow` = partial / in
progress, `—` = not started or not applicable, `n/a` = axis does
not apply.

C golden baseline: `v0.6.3-dfm`
(`516456fe35510e4fb8350396c9d88807ac1f760b`).

## Subsystem matrix

| Subsystem | C Oracle | Rust | Default | Parity | Perf |
|---|---|---|---|---|---|
| model metadata | yes | yes (catalog + DeepSeek select + arch dispatch + bind names + MTP/DSpark sibling catalogs + host `config_validate` + host `weights_validate_layout`) | C (`ds4_bridge_model_open`) | catalog + bind-name + MTP/DSpark support + validate + layout token dump green (`make test-catalog-parity`) | n/a |
| GGUF loading (mmap) | yes | metadata mmap + identify + tensor inventory + split-shard remap + host `weights_bind` catalog / bind-plan check + host tensor-dir apply (native skips `parse_tensors` when installed) + host `config_validate` (native applies pinned shape/compress and skips C validate when installed) + host vocab apply (native skips `vocab_load` when installed) + host bind map (native skips the `model_find_tensor` name walk when installed) + host `weights_validate_layout` (native skips the main-model layout check when the bind map is installed) + host MTP/DSpark sibling name/layout catalogs + sibling BindPlan resolve/validate + **sibling bind map FFI** (`Model::open_with_support` / `opt->mtp_path|dspark_path|mtp_bind|dspark_bind`: native swaps each sibling map into the active slot around its own open+bind window and skips that sibling's C layout check; sibling pointer assignment stays C); host tables are cleared after base `weights_bind` so sibling `model_open` does not reuse the base GGUF tables; CUDA weight upload still C | C (`ds4_bridge_model_open` upload) | synthetic v3 identify + tensor/nbytes/sibling + bind-name/check/match + MTP/DSpark support dump + consume/apply tapes + validate token tapes + vocab apply tapes + bind lookup tapes + layout SPEC/check/support tapes green (`make test-catalog-parity`); live Motif-3-MQ87-88-FIT identify/validate/open green; live DeepSeek-V4-Flash + DSpark-drafter-Q2K-Q8 (81/81 slots FOUND, `dspark-flash` catalog): C env-fallback run kept the strict C validate (`Hi`, 29.27/21.27 t/s), Rust `--dspark` open bound the drafter through the host map + layout skip and decoded 9 ids (`scratch/rust-host-live/{c,rust}-dspark.log`) | — |
| tokenizer | yes | yes (host-owned encode/decode/special/stop and family chat-transcript token framing + host vocab apply so native skips `vocab_load` when installed; `Model` tokenize/stop/eos/`token_text` use host `Vocab`; `ds4-server-rs` uses `Model::vocab`; FFI tokenize remains for the C engine path) | C | synthetic C↔Rust encode/decode/stop + apply tapes green; chat transcripts match C across five families and all four think modes (`make test-tokenizer-parity`); live Motif specials + encode `hi`→8320 on the production GGUF | — |
| session lifecycle | yes | partial (host ledger + DSV4 prefix, including bounded embedded-range restore; engine session, GPU/logits tail, and payload execution remain native) | C | synthetic C↔Rust green (`make test-session-parity`); range prefix/EOF + C seek/length/overflow oracle green; live same-model CUDA payload pending | — |
| KV store | yes | envelope/policy crate with payload-free metadata index, bounded LCP text reads, and a file-backed staged payload writer; ordinary serial Motif-3 no-think/no-tools visible cold + evict/replay plus final-sync/decode continued checkpoints are production-integrated in `ds4-server-rs` | C | file-format 4-way + Rust streamed payload/trailer read by C + sparse 4 GiB indexing + embedded payload-range seam green; live same-model C/Rust 4-way save/load green at `cold=0, continued=0`; C-default 6,144-token cold partition exact; final-sync call/order CPU-green and scoped decode-frontier continued 6,800-token text/payload bodies plus four-way cross-read exact | same-policy restore TTFT ABBA green; scoped cold and continued correctness/cross-read green; intermediate-prefill/tool-map/bank/default-policy perf pending |
| web utility | yes | isolated blocking-I/O crate; not production-integrated | C (`ds4-agent`) | encode/wire + mock CDP green | n/a |
| server (four surfaces) | yes | partial (`route_decide` + HTTP door + parsers + projectors + admission + metrics + render/tool/continuation machinery + scripted/FFI decode + bounded FIFO single inference owner + per-client parse/drain + stable continuation pins + ordered terminal publication + ordinary serial Motif visible disk replay) | C | owner-FIFO, queue/TTFT, disconnect/slow-reader, terminal, and C-oracle CPU contracts green (`make test-server-parity`); live Motif serial/width-1 continuous and scoped KV restart/cross-read green; static lane, live multi-client continuous batching, and Anthropic/Responses continuous are not implemented | n/a |
| distributed runtime | yes | isolated codecs + blocking orchestration crate; not production-integrated | C | codecs + CLI/route/mock hop green | n/a |
| CLI / bench / agent host | yes | partial: greedy/seeded-sampling `ds4-rs` with non-TTY thinking formatting, local non-MTP/non-distributed `ds4-bench-rs`, one-turn no-tool `ds4-agent-rs`, and `ds4-server-rs` | C | greedy, fixed-seed sampled, and fixed-seed thinking/non-thinking one-shot CLI stdout match C byte-for-byte; local benchmark ABBA green. Agent built-in prompt bytes, fixed datetime message, selected non-TTY projector tapes, and DSML refusal are green (`make test-agent-parity`), but live agent generation parity is not established | local benchmark green; full gate pending |
| CPU reference backend | yes | no (not a cut-over blocker) | C | — | n/a |
| Metal backend | native | unchanged | native | — | n/a |
| CUDA / MMQ / VMM | native | unchanged | native | green (C baseline) | green (published band) |
| FFI bridge (`ds4_bridge`) | n/a | broad C-host bridge; not the final narrow CUDA ABI | n/a | FFI error path + bounded payload seek/length/EOF/overflow oracle green | — |
| proof harness on Rust path | C binaries | partial | C | Rust smoke 2/2 and long 6/6 green; same-host C→Rust OPP-C 5/5 green. The GB10 `sm_121a` native snapshot is frozen separately: detached `v0.6.3-dfm` and current C both pass 5/5 with the same stable plan/token oracle | — |

Rust entries above describe isolated ownership/parity unless `Default` is
Rust. They must not be read as production-path integration.

## Phase checklist

| Phase | Name | State |
|---|---|---|
| 0 | Freeze baseline + this document set | **done** (docs-only commit) |
| 1 | Cargo workspace + FFI skeleton | **done** (`cargo check --workspace`, `make rust-bridge`) |
| 2 | `ds4-core` safe wrappers | **wrapper layer green**; production model/session ownership remains native |
| 3 | Shadow `ds4-rs` / `ds4-bench-rs` / `ds4-agent-rs` | **linked; CLI live CUDA and benchmark ABBA green** (`make ds4-rs ds4-bench-rs ds4-agent-rs`). Greedy, fixed-seed sampled, and fixed-seed thinking/non-thinking CLI stdout match C exactly. The benchmark completed snapshot/restore smoke and a C→Rust→Rust→C local sweep inside prefill/decode/RSS thresholds. The agent shadow owns one non-interactive no-tool turn and has no-GPU prompt/projector parity; live agent generation, tools, KV, TTY, MTP, and distributed paths remain pending |
| 4 | KV store port + 4-way matrix | **format/policy + payload-free index + embedded range seam + replacement-safe staged writer green** (`make test-kv-parity`, focused core/bridge oracles). Ordinary serial Motif-3 no-think/no-tools visible save/evict/restart restore is wired; live C→Rust, Rust→C, Rust→Rust, and C→C are green at the shared `cold=0, continued=0` policy. Restore TTFT ABBA and timing-count/rate scope are green after deferred boot prewarm. C-default cold checkpoints are live cross-read green. The scoped final-sync/decode continued slice is also four-way live-green at 6,800 tokens. Intermediate-prefill continued staging, tool-map checkpoint replay, continuous-bank checkpoints, and full default-policy ABBA remain pending |
| 5 | Web utility port | **isolated parity green** (`make test-web-parity`); production `ds4-agent` still uses C `ds4_web.c` |
| 6 | Distributed runtime port | **isolated parity green** (`make test-dist-parity`); production still uses C pipelined prefetch, snapshot, and `ds4_dist_session_*` |
| 7 | Server shadow by feature | **partial** — bounded owner-FIFO/terminal/disconnect CPU contract is green at `6545d44`; fixtures, live Motif serial/width-1 continuous, and scoped ordinary serial Motif disk replay are green, but static, live multi-client continuous batching, Anthropic/Responses continuous, and the remaining KV policies/surfaces are missing |
| 8 | `ds4.c` decomposition | **partial** — metadata, mmap catalogs, tokenizer (including family chat-transcript framing), and ledger slices are green; engine/model/session/scheduler execution and CUDA upload remain native |
| 9 | Promote Rust binaries to default names | **not started** — blocked on Rust model/session/scheduler ownership, leaf-crate production integration, a narrow CUDA ABI, full CLI/bench/agent/server parity, native-drift proof, pre/post family regression, and pre-split performance/soak evidence |
| split | `SPLIT_READINESS.md` + `dfm-rs` genesis | blocked on the full pre-split readiness gate; do not create the document or repository yet |

## Current default binaries

| Name | Implementation |
|---|---|
| `ds4` | C |
| `ds4-server` | C |
| `ds4-bench` | C |
| `ds4-eval` | C |
| `ds4-agent` | C |
| `ds4-rs` | Partial Rust CLI host, same C CUDA core (`make ds4-rs`); diagnostics plus greedy, seeded-sampling, and non-TTY thinking-output one-shot paths are host-owned. TTY color, MTP, REPL, batch, and distributed CLI remain C-only |
| `ds4-bench-rs` | Local raw-prompt benchmark shadow with the C incremental sync, snapshot/decode/restore timing, and 8-column CSV contract. Live CUDA two-frontier smoke is green; MTP, distributed, chat prompt, logits dump, output-head, warm, quality, and power modes remain C-only |
| `ds4-agent-rs` | One-turn `--non-interactive -p` no-tool agent shadow over the native model/session/CUDA bridge. Rust owns the built-in tool prompt, transcript assembly, datetime message, sampling loop, and non-TTY projection for this narrow lane; interactive/stdin-repeat, tool execution, KV/resume, MTP, and distributed paths remain C-only |
| `ds4-server-rs` | Partial Rust HTTP host over the native model/session/scheduler bridge. Client threads parse and drain bounded output while one Rust owner executes native inference through FIFO. Serial, ordinary Motif visible cold + evict/replay, final-sync/decode continued checkpoints, and width-1 OpenAI continuous paths exist; static, live multi-client continuous batching, Anthropic/Responses continuous, intermediate-prefill continued staging, continuous-bank checkpoints, and tool-map replay do not |
| `ds4_weight_server` | native CUDA (unchanged) |

Phase 3/7 live Motif evidence is in `scratch/rust-host-live/` (tmux + workspace-local `../scripts/guarded-run.sh`, outside this repository; sequential C then Rust, `clear_cache` between loads). Production defaults stay C. Live CUDA census on `ds4-server-rs` is green (`census_supported=1`, epoch 1636, weight_artifact 86.07 GiB, observation ok). Live DSpark sibling FFI is green (DeepSeek-V4-Flash-IQ2XXS + DSpark-drafter-Q2K-Q8: C env fallback `Hi` 29.27/21.27 t/s with strict C validate; Rust `--dspark` host-map bind + layout skip, 9 decode ids; sequential, teardown + `clear_cache`). The Rust server has a continuous lane (`--cont-width`, default 2; one active job behind the FIFO owner): `ds4_bridge_batch_ctx_create_fit` + `ds4_bridge_continuous_generate` drive the native rolling scheduler, the host owns per-token stop/tool/think semantics (`ContStepper`), and live Motif routes `openai_chat_continuous` with `Hi.` / stop / 13+2 — byte-equal to C. Client ingress is now concurrent, but the owner still runs each job to completion, so multi-client rolling width is not yet live-proven. Same-lane ABBA is recorded (PARITY_MATRIX: Rust 590.8/592.9 t/s prefill vs C 579.2/633.9; decode within 1%; all four completions byte-identical).

Phase 4 live ordinary replay evidence is under `scratch/rust-host-live/rust-kv-no-think-final-IHtJue/`. With Motif-3 MQ87-88, ctx 8192, temp 0, no thinking, no tools, and the shared `cold=0, continued=0` policy, C and Rust produced the same 114-token answer and a byte-identical visible-text/payload KVC body (301,972,392 bytes). C→Rust, Rust→C, Rust→Rust, and C→C restart loads all returned `RESTORED_OK` with 6,896 cached + 15 newly evaluated prompt tokens. Rust binary `dae70ae2...` ran the four-way matrix; post-fix `52471e49...` also matches C timing counts (`prefill_tokens=15`, `prefill_cached_tokens=6896`). The initial Rust 1,590--1,715 ms TTFT regression was the deferred native boot prewarm never being called. Commit `f4a632e` restored C's `batch fit → prewarm → listen` ordering; `scratch/rust-host-live/prewarm-restore-LH1LIp/` records C→Rust→Rust→C TTFT 880.9/836.0/860.4/897.5 ms (Rust mean 4.61% below C). Commit `4fc8ef4` also narrowed reported prefill time to the computed suffix sync; the live Rust result is 69.2 tok/s versus C 68.2--69.0.

The C-default ordinary cold slice is live-green under `scratch/rust-host-live/rust-cold-default-0IeCd9/`. The 6,782-token Motif request produced the same 6,144-token cold partition in C and Rust: rendered text SHA-256 `ebd540ec...` and payload SHA-256 `0f0af1d1...` are exact. Rust→Rust, Rust→C, and C→Rust restart cells restored 6,144 cached tokens and produced the same 106-token output (`49b5f8a8...`) with TTFT 2,038.4/2,078.9/2,018.0 ms. This proves cold correctness and cross-read, not a full default-policy ABBA performance gate.

The ordinary serial continued final-sync call/order is CPU-green, while the decode frontier is live-green under `scratch/rust-host-live/continued-fourway-oJTYrf/`. With `cold=0`, `continued=6800`, trim 0, and align 1, C and Rust wrote the same 300,423,338-byte reason-continued record at 6,800 tokens (text SHA-256 `02523123...`, payload SHA-256 `be4d59a9...`) and emitted the same 114-token answer. A Motif history fixture that explicitly preserves the live `<think></think>` bytes then passed C→Rust, Rust→C, Rust→Rust, and C→C: every cell returned `RESTORED_OK`, prompt/cached/computed/output counts 6,911/6,800/111/4, and semantic response SHA-256 `f61bf763...`. The plain assistant-history fixture is an expected C/Rust miss because Motif drops the empty think pair when closing history. This proves the scoped final/decode slice, not intermediate prefill, tool-map replay, continuous-bank checkpoints, or full default-policy performance. See PARITY_MATRIX for hashes and binary provenance.

The Rust benchmark smoke is in `scratch/rust-host-live/bench-rs-two-frontier.log` (GB10, DeepSeek-V4-Flash IQ2XXS, binary SHA-256 `a66d97c4...`). It completed 64 and 128 token frontiers with four decode tokens each, nonzero checkpoint sizes, exit 0, and post-run cache cleanup. These two rows prove execution and restore flow only; they are not a performance comparison.

CLI sampling evidence is under `scratch/rust-host-live/cli-sampling-parity/` (Rust binary SHA-256 `1287e404...`): C and Rust fixed-seed stdout both produced 52 bytes with SHA-256 `ae12f463...`. The post-change greedy rerun under `cli-greedy-post-sampling/` produced byte-identical `OK\n` (`a12b7cb4...`). Both pairs exited 0 with teardown and cache cleanup between model loads.

CLI thinking evidence is under `scratch/rust-host-live/cli-thinking-parity-20260825/`: with the same fixed-seed DeepSeek V4 Flash workload, C and Rust `--think` stdout are both 292 bytes (`ad6d107f...`) and C and Rust `--nothink` stdout are both 52 bytes (`ae12f463...`). All four runs exited 0; each sequential teardown and cache clear left no GPU application.

Rust proof evidence after the rendered-prompt fix is in `scratch/rust-host-live/proof-{smoke,long}-fixed.log`, `proof-oppc-{c-oracle,rust-fixed}.log`, and `proof-oppc-rust-current.log`: smoke 2/2, long 6/6, and same-host OPP-C C/Rust 5/5 all pass. The final Rust proof used one fixed binary SHA (`056dbdf6...`). The GB10 native gate is also green: detached `v0.6.3-dfm@516456f` matched the current C oracle 5/5, and `CUDA_ARCH=sm_121` now selects the separately tracked `sm_121a` golden; the older generic golden is unchanged. Remaining before Phase 9 includes full user-binary parity, the family regression set, multi-client/server surface follow-ups, and pre-split performance/soak evidence. After Phase 9 + `SPLIT_READINESS.md`, follow [DFM_RS_SPLIT_PLAN.md](DFM_RS_SPLIT_PLAN.md).

## Known remaining C dependency

All default entry points are still C. The web/distributed leaf crates are not
wired into production binaries; the KV crate is wired only for the scoped
ordinary serial cold/evict/continued slices above. The Rust server owns part of the HTTP host surface,
but model/session/scheduler execution remains behind a broad native bridge,
and the full CLI, advanced benchmark modes, interactive/tool/KV agent, and
serving-lane contracts have not been replaced. The decided native-forever pieces remain CUDA/MMQ/VMM (and Metal as
a non-blocking compile).

## Notes

- `dfm` is not merged into `rust-host`. Cherry-pick CUDA
  correctness or performance fixes only, then rerun this matrix.
- `SPLIT_READINESS.md` must not be added until the §26 required
  list is evidence-green.
- Here, pre-split readiness means the migration instruction's §26 gate.
  `DFM_RS_SPLIT_PLAN.md` §26 is the later post-split release gate.
- The default Rust toolchain currently lacks `rustfmt` and `clippy`.
  The installed 1.95 fallback shows pre-existing workspace formatting
  drift and clippy warnings. This blocks the post-split release gate in
  `DFM_RS_SPLIT_PLAN.md` §26; it is not by itself a pre-split readiness gate.
- Unsafe-audit command: `rg -n 'unsafe \{' crates/`
  Current count is 40: 37 in `crates/ds4-core` native/mmap adapters
  (25 in `lib.rs`, 5 in `batch.rs`, 4 in `mapped.rs`, 3 in `mem.rs`) and
  3 libc parser calls in the allowed `crates/ds4-sys` boundary.
  `ds4-kv`, `ds4-web`, `ds4-dist`, and `ds4-server` have none.
- `DS4_SERVER_CLIENT_SNDBUF` is not applied by the Rust shadow because
  safe `std::net::TcpStream` has no send-buffer setter. The bounded-send
  contract is CPU-tested, but the pinned-buffer live slow-reader leg remains
  a Phase 7 gap.
