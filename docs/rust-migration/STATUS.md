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
| tokenizer | yes | yes (host-owned encode/decode/special/stop + host vocab apply so native skips `vocab_load` when installed; `Model` tokenize/stop/eos/`token_text` use host `Vocab`; `ds4-server-rs` uses `Model::vocab`; FFI tokenize remains for the C engine path) | C | synthetic C↔Rust encode/decode/stop + apply tapes green (`make test-tokenizer-parity`); live Motif specials + encode `hi`→8320 on the production GGUF | — |
| session lifecycle | yes | yes (host ledger + DSV4 prefix: magic/version/token_count/tokens; native still writes GPU/logits tail) | C | synthetic C↔Rust green (`make test-session-parity`); live same-model CUDA payload pending | — |
| KV store | yes | yes (envelope + policy) | C | 4-way green | n/a |
| web utility | yes | yes (blocking I/O) | C (`ds4-agent`) | encode/wire + mock CDP green | n/a |
| server (four surfaces) | yes | yes (`route_decide` + HTTP door + parsers + tape projectors + admit + host `/metrics` including memgov census porcelain + live census/observe FFI overlay + family render including tool-schema/invoke + generated-message parse + SemAccum + finalize `tool_calls` + incremental live DSML tool stream + required-prefix / structural greedy sampling + corrective retry (`decode_again`) + Inc 5a/5b/5c continuation registry + scripted/FFI decode driver) | C | reason table + HTTP porcelain + parse + tape + enqueue + DSML/Motif/EXAONE/dots3/Solar render + tool-schema/invoke + generated-tool parse + scripted tool wire + incremental live DSML tool stream + required-prefix / DSML greedy override + corrective retry suffix/repair/`decode_again` + continuation tape/hold/409/publish + full `/metrics` / `/v1/stats` memgov census porcelain + census FFI (NULL/unsupported + live Motif) green; live Motif generate `Hi.` / stop / 13+2 / `cache_write_tokens=13`; live `/v1/stats` `census_supported=true` epoch=1636 weight_artifact 86.07 GiB device_live 95.96 GiB observation ok/cuda_free; Rust lane is serial vs C continuous | n/a |
| distributed runtime | yes | yes (codecs + blocking orchestration) | C | codecs + CLI/route/mock hop green | n/a |
| CLI / bench / agent host | yes | shadow `ds4-rs` / `ds4-bench-rs` / `ds4-server-rs` | C | HTTP door + parsers + projectors + admit + family render + tool-schema/invoke + generated-tool parse + incremental live DSML tool stream + required-prefix / structural greedy + corrective retry + scripted decode + continuation registry + host `/metrics` memgov porcelain + live census FFI overlay green; live Motif `ds4-rs` open+decode and `ds4-server-rs` generate green | — |
| CPU reference backend | yes | no (not a cut-over blocker) | C | — | n/a |
| Metal backend | native | unchanged | native | — | n/a |
| CUDA / MMQ / VMM | native | unchanged | native | green (C baseline) | green (published band) |
| FFI bridge (`ds4_bridge`) | n/a | linked | n/a | FFI error path green | — |
| proof harness on Rust path | C binaries | no | C | — | — |

## Phase checklist

| Phase | Name | State |
|---|---|---|
| 0 | Freeze baseline + this document set | **done** (docs-only commit) |
| 1 | Cargo workspace + FFI skeleton | **done** (`cargo check --workspace`, `make rust-bridge`) |
| 2 | `ds4-core` safe wrappers | **done** (unit tests; no live model) |
| 3 | Shadow `ds4-rs` / `ds4-bench-rs` | **linked + live Motif open/decode** (`make ds4-rs`; sequential C-vs-Rust on Motif-3-MQ87-88-FIT, ctx=2048; same catalog 2287 / artifacts 86.07 GiB / device 93.08 GiB / KV 0.119 GiB; C CLI 60.15/7.88 t/s; Rust CLI emitted 9 decode ids). Full same-prompt tok/s table still only on the server pair |
| 4 | KV store port + 4-way matrix | **format/policy green** (`make test-kv-parity`); live session payload still C |
| 5 | Web utility port | **done** (`make test-web-parity`); `ds4-agent` still links C `ds4_web.c` |
| 6 | Distributed runtime port | **blocking runtime green** (`make test-dist-parity`); C still owns pipelined prefetch, snapshot, `ds4_dist_session_*` |
| 7 | Server shadow by feature | **route_decide + HTTP door + parsers + tape projectors + enqueue/shed + host `/metrics` `/v1/stats` including memgov census porcelain + live census/observe FFI + family render + tool-schema/invoke reconstruct + generated-message parse + SemAccum + finalize `tool_calls` + incremental live DSML tool stream + required-prefix / structural greedy sampling + corrective retry (`decode_again` / model-visible tool error) + Inc 5a/5b/5c continuation registry + FFI tokenize/sample + scripted decode + live Motif generate + live CUDA census + continuous lane green** (`make test-server-parity`; `scratch/rust-host-live/`: serial and continuous both answer `Hi.` / stop / 13+2 / `cache_write_tokens=13`; route `openai_chat_continuous=1`; same-lane ABBA within C cell noise). Buffered `timings` on serial and continuous paths. Width-1 accept; multi-client width + Anthropic/Responses cont promotion are follow-ups |
| 8 | `ds4.c` decomposition | **metadata + mmap identify + tensor inventory / split remap + `weights_bind` name catalog + bind-plan FFI check + host tensor-dir apply (skip `parse_tensors`) + host `config_validate` (skip C validate when shape installed) + host vocab apply (skip `vocab_load` when installed) + host bind map (skip C name walk when installed) + host `weights_validate_layout` (skip C main-model layout when bind map installed) + host MTP/DSpark sibling name/layout catalogs + sibling BindPlan resolve/validate + sibling bind map FFI (skip sibling C layout when installed; live DSpark drafter green) + host-table clear before sibling `model_open` + `Model` host tokenize + tokenizer + session ledger + DSV4 prefix green** (`make test-catalog-parity`, `make test-tokenizer-parity`, `make test-session-parity`); sibling pointer assignment / CUDA weight upload / `ds4_engine_open` still native |
| 9 | Promote Rust binaries to default names | not started |
| split | `SPLIT_READINESS.md` + `dfm-rs` genesis | blocked on 9 |

## Current default binaries

| Name | Implementation |
|---|---|
| `ds4` | C |
| `ds4-server` | C |
| `ds4-bench` | C |
| `ds4-eval` | C |
| `ds4-agent` | C |
| `ds4-rs` / `ds4-bench-rs` | Rust host, same C CUDA core (`make ds4-rs`); `--identify` / `--inventory` / `--bind-names` / `--layout` / `--bind-plan` / `--validate` / `--tokenize` / `--session-plan` / `--session-payload` are host-owned (no engine open); `--mtp` / `--dspark` attach DeepSeek sibling support models through `Model::open_with_support` (host sibling resolve/validate + native layout skip) |
| `ds4-server-rs` | Rust HTTP door + parsers + projectors + admit + host `/metrics` `/v1/stats` (memgov census porcelain + live CUDA census/observe FFI when `-m` opens an engine) + family render + tool-schema/invoke + generated-tool parse + finalize `tool_calls` + incremental live DSML tool stream + required-prefix / structural greedy sampling + corrective retry + Inc 5 continuation registry + host `Model` Vocab tokenize + host SessionLedger + FFI decode + continuous lane (`--cont-width`, width-1 serial accept; host `ContStepper` per-token semantics over the native rolling scheduler) (`make ds4-server-rs`) |
| `ds4_weight_server` | native CUDA (unchanged) |

Phase 3/7 live Motif evidence is in `scratch/rust-host-live/` (tmux + `scripts/guarded-run.sh`, sequential C then Rust, `clear_cache` between loads). Production defaults stay C. Live CUDA census on `ds4-server-rs` is green (`census_supported=1`, epoch 1636, weight_artifact 86.07 GiB, observation ok). Live DSpark sibling FFI is green (DeepSeek-V4-Flash-IQ2XXS + DSpark-drafter-Q2K-Q8: C env fallback `Hi` 29.27/21.27 t/s with strict C validate; Rust `--dspark` host-map bind + layout skip, 9 decode ids; sequential, teardown + `clear_cache`). The Rust server has a continuous lane (`--cont-width`, default 2; width-1 serial accept): `ds4_bridge_batch_ctx_create_fit` + `ds4_bridge_continuous_generate` drive the native rolling scheduler, the host owns per-token stop/tool/think semantics (`ContStepper`), and live Motif routes `openai_chat_continuous` with `Hi.` / stop / 13+2 — byte-equal to C. Same-lane ABBA is recorded (PARITY_MATRIX: Rust 590.8/592.9 t/s prefill vs C 579.2/633.9; decode within 1%; all four completions byte-identical). Remaining before Phase 9: proof harness on the Rust path; multi-client cont width and Anthropic/Responses cont promotion are documented follow-ups. After Phase 9 + `SPLIT_READINESS.md`, follow [DFM_RS_SPLIT_PLAN.md](DFM_RS_SPLIT_PLAN.md).

## Known remaining C dependency

**Everything in the production host.** The only decided native
forever-piece is the CUDA/MMQ/VMM backend (and Metal as a
non-blocking compile).

## Notes

- `dfm` is not merged into `rust-host`. Cherry-pick CUDA
  correctness or performance fixes only, then rerun this matrix.
- `SPLIT_READINESS.md` must not be added until the §26 required
  list is evidence-green.
- Unsafe-audit command: `rg -n 'unsafe \{' crates/`
  Current hits are `crates/ds4-core` FFI adapters plus the GGUF mmap
  adapter (`mapped.rs`). `ds4-sys` is `extern "C"` declarations.
  `ds4-kv`, `ds4-web`, `ds4-dist`, and `ds4-server` have none.
