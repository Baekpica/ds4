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
| session lifecycle | yes | partial (host ledger + DSV4 prefix only; engine session, GPU/logits tail, and payload execution remain native) | C | synthetic C↔Rust green (`make test-session-parity`); live same-model CUDA payload pending | — |
| KV store | yes | isolated envelope/policy crate; not production-integrated | C | file-format 4-way green | n/a |
| web utility | yes | isolated blocking-I/O crate; not production-integrated | C (`ds4-agent`) | encode/wire + mock CDP green | n/a |
| server (four surfaces) | yes | partial (`route_decide` + HTTP door + parsers + projectors + admission + metrics + render/tool/continuation machinery + scripted/FFI decode) | C | fixtures and live Motif serial/width-1 continuous green; static lane, multi-client continuous, and Anthropic/Responses continuous are not implemented | n/a |
| distributed runtime | yes | isolated codecs + blocking orchestration crate; not production-integrated | C | codecs + CLI/route/mock hop green | n/a |
| CLI / bench / agent host | yes | partial: greedy/seeded-sampling `ds4-rs`, local non-MTP/non-distributed `ds4-bench-rs`, and `ds4-server-rs`; no `ds4-agent-rs` | C | one-shot CLI and local benchmark ABBA green; thinking/MTP/REPL/batch/distributed CLI, advanced benchmark modes, and agent parity not established | local benchmark green; full gate pending |
| CPU reference backend | yes | no (not a cut-over blocker) | C | — | n/a |
| Metal backend | native | unchanged | native | — | n/a |
| CUDA / MMQ / VMM | native | unchanged | native | green (C baseline) | green (published band) |
| FFI bridge (`ds4_bridge`) | n/a | broad C-host bridge; not the final narrow CUDA ABI | n/a | FFI error path green | — |
| proof harness on Rust path | C binaries | partial | C | Rust smoke 2/2 and long 6/6 green; same-host C→Rust OPP-C 5/5 green after rendered-prompt parity fix. The committed OPP-C snapshot is stale (three profiles, old token MD5/plan), so the architecture-specific drift gate is not yet green | — |

Rust entries above describe isolated ownership/parity unless `Default` is
Rust. They must not be read as production-path integration.

## Phase checklist

| Phase | Name | State |
|---|---|---|
| 0 | Freeze baseline + this document set | **done** (docs-only commit) |
| 1 | Cargo workspace + FFI skeleton | **done** (`cargo check --workspace`, `make rust-bridge`) |
| 2 | `ds4-core` safe wrappers | **wrapper layer green**; production model/session ownership remains native |
| 3 | Shadow `ds4-rs` / `ds4-bench-rs` | **linked + live CUDA/ABBA green** (`make ds4-rs ds4-bench-rs`). Greedy and fixed-seed sampled one-shot stdout match C exactly. The benchmark completed snapshot/restore smoke and a C→Rust→Rust→C local sweep inside prefill/decode/RSS thresholds; advanced modes and full production performance remain pending |
| 4 | KV store port + 4-way matrix | **format/policy green** (`make test-kv-parity`); live session payload still C |
| 5 | Web utility port | **isolated parity green** (`make test-web-parity`); production `ds4-agent` still uses C `ds4_web.c` |
| 6 | Distributed runtime port | **isolated parity green** (`make test-dist-parity`); production still uses C pipelined prefetch, snapshot, and `ds4_dist_session_*` |
| 7 | Server shadow by feature | **partial** — fixtures and live Motif serial/width-1 continuous are green, but static, multi-client continuous, and Anthropic/Responses continuous remain missing |
| 8 | `ds4.c` decomposition | **partial** — metadata, mmap catalogs, tokenizer, and ledger slices are green; engine/model/session/scheduler execution and CUDA upload remain native |
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
| `ds4-rs` | Partial Rust CLI host, same C CUDA core (`make ds4-rs`); diagnostics plus greedy and seeded-sampling one-shot paths are host-owned. Thinking formatting, MTP, REPL, batch, and distributed CLI remain C-only |
| `ds4-bench-rs` | Local raw-prompt benchmark shadow with the C incremental sync, snapshot/decode/restore timing, and 8-column CSV contract. Live CUDA two-frontier smoke is green; MTP, distributed, chat prompt, logits dump, output-head, warm, quality, and power modes remain C-only |
| `ds4-server-rs` | Partial Rust HTTP host over the native model/session/scheduler bridge. Serial and width-1 OpenAI continuous paths exist; static, multi-client continuous, and Anthropic/Responses continuous do not |
| `ds4_weight_server` | native CUDA (unchanged) |

Phase 3/7 live Motif evidence is in `scratch/rust-host-live/` (tmux + workspace-local `../scripts/guarded-run.sh`, outside this repository; sequential C then Rust, `clear_cache` between loads). Production defaults stay C. Live CUDA census on `ds4-server-rs` is green (`census_supported=1`, epoch 1636, weight_artifact 86.07 GiB, observation ok). Live DSpark sibling FFI is green (DeepSeek-V4-Flash-IQ2XXS + DSpark-drafter-Q2K-Q8: C env fallback `Hi` 29.27/21.27 t/s with strict C validate; Rust `--dspark` host-map bind + layout skip, 9 decode ids; sequential, teardown + `clear_cache`). The Rust server has a continuous lane (`--cont-width`, default 2; width-1 serial accept): `ds4_bridge_batch_ctx_create_fit` + `ds4_bridge_continuous_generate` drive the native rolling scheduler, the host owns per-token stop/tool/think semantics (`ContStepper`), and live Motif routes `openai_chat_continuous` with `Hi.` / stop / 13+2 — byte-equal to C. Same-lane ABBA is recorded (PARITY_MATRIX: Rust 590.8/592.9 t/s prefill vs C 579.2/633.9; decode within 1%; all four completions byte-identical).

The Rust benchmark smoke is in `scratch/rust-host-live/bench-rs-two-frontier.log` (GB10, DeepSeek-V4-Flash IQ2XXS, binary SHA-256 `a66d97c4...`). It completed 64 and 128 token frontiers with four decode tokens each, nonzero checkpoint sizes, exit 0, and post-run cache cleanup. These two rows prove execution and restore flow only; they are not a performance comparison.

CLI sampling evidence is under `scratch/rust-host-live/cli-sampling-parity/` (Rust binary SHA-256 `1287e404...`): C and Rust fixed-seed stdout both produced 52 bytes with SHA-256 `ae12f463...`. The post-change greedy rerun under `cli-greedy-post-sampling/` produced byte-identical `OK\n` (`a12b7cb4...`). Both pairs exited 0 with teardown and cache cleanup between model loads.

Rust proof evidence after the rendered-prompt fix is in `scratch/rust-host-live/proof-{smoke,long}-fixed.log`, `proof-oppc-{c-oracle,rust-fixed}.log`, and `proof-oppc-rust-current.log`: smoke 2/2, long 6/6, and same-host OPP-C C/Rust 5/5 all pass. The final Rust proof used one fixed binary SHA (`056dbdf6...`). The GB10 native gate is also green: detached `v0.6.3-dfm@516456f` matched the current C oracle 5/5, and `CUDA_ARCH=sm_121` now selects the separately tracked `sm_121a` golden; the older generic golden is unchanged. Remaining before Phase 9 includes full user-binary parity, the family regression set, multi-client/server surface follow-ups, and pre-split performance/soak evidence. After Phase 9 + `SPLIT_READINESS.md`, follow [DFM_RS_SPLIT_PLAN.md](DFM_RS_SPLIT_PLAN.md).

## Known remaining C dependency

All default entry points are still C. The Rust leaf crates are not wired into
the production binaries. The Rust server owns part of the HTTP host surface,
but model/session/scheduler execution remains behind a broad native bridge,
and the full CLI, advanced benchmark modes, agent, and serving-lane contracts
have not been replaced. The decided native-forever pieces remain CUDA/MMQ/VMM (and Metal as
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
  Current hits are `crates/ds4-core` FFI adapters plus the GGUF mmap
  adapter (`mapped.rs`). `ds4-sys` is `extern "C"` declarations.
  `ds4-kv`, `ds4-web`, `ds4-dist`, and `ds4-server` have none.
