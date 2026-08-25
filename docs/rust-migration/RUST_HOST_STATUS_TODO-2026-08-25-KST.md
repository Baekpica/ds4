# Rust host migration pause snapshot and TODO

- Recorded: 2026-08-25 11:44 KST
- Branch: `rust-host`
- Snapshot commit: `5abdd53f52a849589a1267d3783ae021682faf11`
- C golden baseline: `v0.6.3-dfm`
  (`516456fe35510e4fb8350396c9d88807ac1f760b`)
- State: **implementation and live-gate work intentionally paused**

This document is the resume index for work completed after the last formal
status-document commit, the remaining code work, and the later consolidated
acceptance gate. It does not declare a phase green merely because code exists.

## 1. How to read this snapshot

Three states must remain separate:

| State | Meaning |
|---|---|
| Implemented | The Rust code path exists at this snapshot. |
| Cheap gate green | Formatting, compile, unit, focused integration, or C-oracle tests passed for that slice. |
| Integrated gate green | The required real-model, cross-host, API, long-context, memory, or performance comparison passed. |

The user approved a code-first ordering on 2026-08-25:

1. finish the remaining Phase 2–8 implementation in bounded functional slices;
2. run the cheapest relevant checks before each functional commit;
3. commit and push each independently meaningful slice;
4. defer expensive model loads, API matrices, long-context proof, ABBA/RSS,
   and soak into one consolidated comparison before Phase 9;
5. do not promote Rust binaries or create a split-readiness document until
   that consolidated gate is green.

This changes gate scheduling, not the acceptance contract. Numerical, token,
KV, wire/API, and performance parity are still all required.

## 2. Exact repository state at the pause

| Item | State |
|---|---|
| Local branch | `rust-host` |
| Local HEAD | `5abdd53f52a849589a1267d3783ae021682faf11` |
| `origin/rust-host` | Same commit as local HEAD |
| Distance from baseline | 106 commits after `v0.6.3-dfm` |
| Production default binaries | C (`ds4`, `ds4-server`, `ds4-bench`, `ds4-agent`, `ds4-eval`) |
| Rust shadows | `ds4-rs`, `ds4-server-rs`, `ds4-bench-rs`, `ds4-agent-rs` |
| Rust formatting | Workspace `rustfmt.toml`; use Cargo `rustfmt` |
| CUDA/MMQ/VMM | Native implementation unchanged; not a migration target |
| Phase 9 | Not started |
| `SPLIT_READINESS.md` | Intentionally absent |
| Remote `dfm-rs` repository | Must not be created yet |

The tracked tree was clean at the pause. The following local untracked files
and generated binaries were deliberately not staged by the migration commits:

```text
docs/rust-migration/HANDOFF-2026-08-24-KST.md
docs/rust-migration/ds4_dfm_c_to_rust_migration_plan.md
docs/rust-migration/dfm_rs_repository_split_followup_plan.md
ds4-agent-rs
tests/cuda_long_context_smoke
tests/parity/agent_c_oracle
tests/test_dots3_*
tests/test_exaone_batch
tests/test_mmq_parity
tests/test_model_family_kernels
tests/test_motif3_*
tests/test_repack_premapped
tests/test_solar_forward
```

Do not use a broad `git add .` when resuming. Generated binaries and the three
local reference documents above need an explicit keep/track decision.

## 3. Fixed decisions and guardrails

- Preserve `v0.6.3-dfm` as the immutable C correctness oracle.
- Continue the strangler migration; do not translate `ds4.c` into one Rust file.
- Do not rewrite or optimize CUDA/MMQ kernels as part of host migration.
- Keep CUDA/VMM/session details behind opaque handles and narrow the final ABI.
- Preserve mmap-backed GGUF loading; never replace it with a full-file `Vec<u8>`.
- Preserve the four HTTP surfaces and serial/continuous/static lane semantics.
- Preserve checkpoint bytes, wire codecs, token IDs, stop behavior, errors,
  identifiers, stream ordering, and continuation semantics.
- Do not introduce Tokio, Axum, a new scheduler, or a new checkpoint/wire format.
- Keep `unsafe` concentrated in `ds4-sys` and reviewed native adapters.
- Keep CPU and Metal backends buildable, but do not make them CUDA cut-over blockers.
- Format Rust with `cargo fmt --all`; check with
  `cargo fmt --all -- --check` before every Rust commit.
- Keep migration and optimization commits separate.
- Use one functional meaning per commit and push to `origin/rust-host` after its
  focused gate is green.
- Run only one large resident model at a time. Stop the target before invoking
  `/usr/local/bin/clear_cache`.
- Do not start Phase 9, rename production binaries, delete C oracles, create
  `SPLIT_READINESS.md`, or create/push `Baekpica/dfm-rs` during the current stage.

## 4. Document authority and precedence

When documents differ, use this order:

1. the user's full C-to-Rust migration instruction and explicit later decisions;
2. `AGENT.md` correctness, safety, mmap, API, and long-context rules;
3. frozen architecture, FFI, API-surface, model-family, and baseline contracts;
4. current code, C oracle, committed tests, and reproducible run artifacts;
5. this pause snapshot for work after `042562b` and the resume order;
6. `STATUS.md`, `PARITY_MATRIX.md`, and the older handoff for historical evidence;
7. the repository-split plans, which remain inactive until all pre-split gates pass.

`STATUS.md` and `PARITY_MATRIX.md` remain formal ledgers, but their current text
was last synchronized at `042562b`. They do not include commits `16f5911` through
`5abdd53`. In particular, any statement that workspace `rustfmt` is missing is
superseded by `a778529`. Update the two formal ledgers together in a later
documentation-only synchronization commit; do not rewrite their earlier live
evidence.

## 5. Complete reference document registry

The tables below list every document that should be considered when resuming.
“Mandatory” means read before changing that area. “Conditional” means read only
when the named subsystem, family, hardware path, or release step is in scope.

### 5.1 Always-read migration and correctness contracts

| Document | Use |
|---|---|
| [`AGENT.md`](../../AGENT.md) | Repository-wide correctness-before-speed, mmap, API, CUDA captured/eager, MTP, test, style, and safety rules. |
| [`VERSION`](../../VERSION) | Confirms the frozen DFM release identity; not Markdown, but part of baseline verification. |
| [`README.md`](README.md) | Migration mission, fixed phase order, commit discipline, and pre-split definition of done. |
| [`ds4_dfm_c_to_rust_migration_plan.md`](ds4_dfm_c_to_rust_migration_plan.md) | Full user-authored Phase 0–9 instruction, acceptance thresholds, code-first supplement, promotion blockers, and split gate. Currently local/untracked. |
| [`BASELINE.md`](BASELINE.md) | Baseline SHA, host/CUDA/GPU facts, published performance cells, thresholds, and proof commands. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Strangler boundary, crate responsibilities, ownership model, mmap contract, lanes, and promotion shape. |
| [`FFI_CONTRACT.md`](FFI_CONTRACT.md) | Opaque handles, lifetime/ownership rules, error convention, linkage, forbidden bridge growth, and final native ABI target. |
| [`STATUS.md`](STATUS.md) | Formal subsystem/phase/default-binary ledger through `042562b`; stale for later implementation and due for synchronization. |
| [`PARITY_MATRIX.md`](PARITY_MATRIX.md) | Formal phase gates, family target list, proof gates, live evidence, ABBA protocol, and threshold rules through `042562b`. |
| [`HANDOFF-2026-08-24-KST.md`](HANDOFF-2026-08-24-KST.md) | Earlier live KV/bank evidence and operational restart detail. Its HEAD and immediate-next-task fields are historical. Currently local/untracked. |
| `RUST_HOST_STATUS_TODO-2026-08-25-KST.md` | This post-`042562b` status and resume index. It supplements rather than weakens the formal contracts. |

### 5.2 API, continuation, agent, and serving references

| Document | Use |
|---|---|
| [`../ds4-api-surface-matrix.md`](../ds4-api-surface-matrix.md) | Frozen four-surface schema, lane routing, event automata, error/ID quirks, trust domain, metrics, and `ds4_test --server` fixture inventory. Mandatory for Phase 7. |
| [`../../misc/ANTHROPIC_LIVE_CONTINUATION.md`](../../misc/ANTHROPIC_LIVE_CONTINUATION.md) | Anthropic tool-continuation contract, fallbacks, QA checklist, and known implementation behavior. |
| [`../../misc/RESPONSE_API.md`](../../misc/RESPONSE_API.md) | Responses protocol model, continuation plan, implementation progress, and QA cases. |
| [`../../misc/COMPACT.md`](../../misc/COMPACT.md) | Agent context-compaction triggers, summary contract, UI behavior, and validation log. Mandatory before porting compaction. |

### 5.3 Model, CUDA, proof, and performance references

| Document | Use |
|---|---|
| [`../ds4-dfm-model-families.md`](../ds4-dfm-model-families.md) | Integrated family contracts, common API/KV behavior, weight owner, memory hygiene, family evidence, limits, and profiling order. |
| [`../../misc/proof-harness/README.md`](../../misc/proof-harness/README.md) | Stable proof runner, budgets, weight server, smoke/long plans, output, and current limitations. |
| [`../../misc/cuda-env-vars.md`](../../misc/cuda-env-vars.md) | CUDA dispatcher, diagnostic, VMM, and backend environment-variable behavior. Required before reproducing CUDA runs. |
| [`../../misc/cuda-mtp/README.md`](../../misc/cuda-mtp/README.md) | GB10 MTP build/options, benchmark method, result interpretation, and profiling detail. Mandatory for remaining CLI/bench MTP work. |
| [`../../cuda/mmq/VENDOR.md`](../../cuda/mmq/VENDOR.md) | MMQ upstream pin, local shim/symbol provenance, unsupported features, resync method, and test matrix. Read before any intentional upstream CUDA port. |
| [`../../speed-bench/README.md`](../../speed-bench/README.md) | Benchmark runner/output conventions when producing consolidated performance evidence. |
| [`../../speed-bench/mtp-compare-2026-05-14/README.md`](../../speed-bench/mtp-compare-2026-05-14/README.md) | Historical MTP comparison method and context; conditional on MTP performance work. |
| [`../../tests/test-vectors/README.md`](../../tests/test-vectors/README.md) | Test-vector provenance and expected use; read before changing golden vector handling. |

### 5.4 Family-specific live and operational references

| Document | Use |
|---|---|
| [`../dots3-note-spark-porting-handoff.md`](../dots3-note-spark-porting-handoff.md) | Dots3 math, loader/tokenizer/resident behavior, Spark operational state, limits, and profiling order. |
| [`../motif-3-spark-optimization-handoff.md`](../motif-3-spark-optimization-handoff.md) | Motif-3 integration, owner/worker lifecycle, correctness evidence, 256K validation, and performance history. |
| [`../motif3-partial-reuse-2026-08-22.md`](../motif3-partial-reuse-2026-08-22.md) | Motif-3 partial-prefix behavior, correctness, A/B, memory evidence, and limits. |
| [`../solar-partial-reuse-2026-08-21.md`](../solar-partial-reuse-2026-08-21.md) | Solar partial-prefix behavior, correctness, A/B, memory evidence, and limits. |
| [`../v062-dfm-motif3-opt-resume-2026-08-21.md`](../v062-dfm-motif3-opt-resume-2026-08-21.md) | Older Motif profiling/recovery commands and provenance. Use as historical operational context, not as the current branch state. |

### 5.5 Root project and platform references

| Document | Use |
|---|---|
| [`../../README.md`](../../README.md) | Current user-facing build, CLI, server, model, and platform contract. Check before changing public behavior or Phase 9 names. |
| [`../../CHANGELOG.md`](../../CHANGELOG.md) | Release history and behavior provenance. Needed for release/promotion documentation. |
| [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) | Contribution/build/test conventions that remain applicable. |
| [`../../METAL_DSPARK.md`](../../METAL_DSPARK.md) | Metal and D-Spark platform notes. Conditional compile/smoke reference; Metal is not the primary cut-over gate. |
| [`../../MODEL_CARD.md`](../../MODEL_CARD.md) | Model-facing claims and supported behavior. Conditional on user-facing support/documentation changes. |
| [`../../dir-steering/README.md`](../../dir-steering/README.md) | Directory-steering behavior. Conditional if CLI/agent path semantics touch this feature. |

### 5.6 GGUF tooling references

These documents are conditional. They are not current Rust-host gate ledgers,
but must be read if model conversion, imatrix generation, mixed quantization,
or quality scoring becomes part of a migration slice:

| Document | Scope |
|---|---|
| [`../../gguf-tools/README.md`](../../gguf-tools/README.md) | GGUF tool suite overview. |
| [`../../gguf-tools/imatrix/README.md`](../../gguf-tools/imatrix/README.md) | Imatrix workflow. |
| [`../../gguf-tools/imatrix/dataset/README.md`](../../gguf-tools/imatrix/dataset/README.md) | Imatrix dataset provenance/use. |
| [`../../gguf-tools/mixed/README.md`](../../gguf-tools/mixed/README.md) | Mixed-quantization workflow. |
| [`../../gguf-tools/quality-testing/README.md`](../../gguf-tools/quality-testing/README.md) | Quant/model quality test workflow. |

### 5.7 Explicitly inactive or out-of-scope plans

| Document | State |
|---|---|
| [`DFM_RS_SPLIT_PLAN.md`](DFM_RS_SPLIT_PLAN.md) | Tracked condensed post-split plan. **Inactive** until Phase 9 and split readiness are green. |
| [`dfm_rs_repository_split_followup_plan.md`](dfm_rs_repository_split_followup_plan.md) | Full local post-split plan. **Inactive** and currently untracked. Reconcile it with the condensed plan only at genesis preparation. |
| `SPLIT_READINESS.md` | Must remain absent until every runtime, correctness, performance, and architecture requirement passes. |
| [`../ds4-w2a16-fused-dequant-gemm-design.md`](../ds4-w2a16-fused-dequant-gemm-design.md) | Trainer/kernel design sketch, not Rust host migration work. Read only if a separately approved CUDA change is intentionally ported. |

### 5.8 Operational sources of truth that are not prose documents

| Source | Use |
|---|---|
| `Cargo.toml`, `Cargo.lock` | Workspace members, feature graph, and locked Rust dependencies. |
| `rustfmt.toml` | Formatting baseline introduced by `a778529`. |
| `Makefile` | Canonical build, parity, family, and proof targets. |
| `tests/parity/` | C byte/error/state-machine oracles. |
| `crates/*/tests/` | Focused Rust regression and cross-language tests. |
| `git log v0.6.3-dfm..rust-host` | Semantic migration history and per-commit evidence. |
| `scratch/rust-host-live/` | Local real-model evidence. It is not a substitute for committed ledgers and may not exist on another host. |
| workspace `../scripts/guarded-run.sh` | Guarded resident-model execution used by the formal live protocol. |

## 6. Work completed after the last formal status update

The formal ledgers stop at `042562b`. The following committed and pushed slices
are additional current implementation evidence:

| Commit | Phase | Implemented | Cheap evidence | Consolidated gate |
|---|---:|---|---|---|
| `16f5911` | 7 | Stateless Anthropic continuous routing/projection | Server parity and native all-target check | Pending live four-surface matrix |
| `e2f0718` | 3/5 | Agent `google_search` execution and continuation | Web/agent C-oracle and native agent check | Pending full agent/live parity |
| `58ccd96` | 3/6 | Rust CLI distributed options, model lifecycle, worker launch, coordinator route-ready | Bridge, dist, CLI/core, native build | Pending distributed live/perf |
| `d8d766d` | 4/7 | Width-greater-than-one continuous warm-bank set ownership | Bank/server unit, C-oracle, native check | Pending true multi-client/multi-bank live gate |
| `fc241f6` | 7 | Stateless Responses buffered/streaming continuous projection | Focused Responses and full server parity | Pending live tool/continuation matrix |
| `fda6205` | 3/6 | Distributed benchmark coordinator sweeps and CSV lifecycle | CLI and dist parity, native bench build | Pending distributed benchmark comparison |
| `36ed697` | 3/5 | Agent `visit_page` execution and bounded staging | Web/agent parity and native agent check | Pending full agent/live parity |
| `02ecd86` | 4/7 | Continuation-bank protection from warm victim selection | Registry/bank/server parity | Pending fork/pin/claim and live saturation |
| `a778529` | all Rust | Workspace formatting baseline | Format, workspace, C-oracle, native checks | n/a |
| `8c8e25f` | 6 | Exact snapshot request/begin/chunk/done codecs and validation | Dist runtime and C-wire oracle | Pending transport/session integration |
| `45c954f` | 4/7 | Clean 503 when every bank is continuation-protected | Server unit/C-oracle/native check | Pending live directed-claim/reclaim behavior |
| `8f06632` | 2/7 | Safe owned wrapper for native static batch generation | Core, NULL/ragged/short-buffer, ASan, bridge | Static server owner and live fallback pending |
| `f012b77` | 3/5 | Byte-safe agent `read` with C bounds/NUL behavior | Agent/tokenizer parity and native build | Pending full tool/agent gate |
| `0448b3b` | 3 | Benchmark chat-prompt and no-thinking input modes | Bench/core/native and byte-edge tests | Pending benchmark integration comparison |
| `ff72bbc` | 6 | Coordinator-side streaming snapshot save/load transport | 14 runtime, 5 snapshot, 12 wire tests | Worker/session/live snapshot pending |
| `2bc4e57` | 3/5 C oracle | Fixed C `more` alias, stale cursor, and overflow bugs | Extended C oracle | Used by the Rust port below |
| `9fcd3c8` | 3/5 | Stateful agent `more` cursor across tool rounds | 22 agent and 3 tokenizer tests | Pending full tool/agent gate |
| `5abdd53` | 2/3 | Safe model-open quality, warm-weight, and power options; benchmark flags | Bench/core/bridge/native checks and rustfmt | Pending benchmark integration comparison |

No CUDA/MMQ kernel was changed by these slices.

## 7. Phase 0–9 status at the pause

| Phase | Current state | Code still missing | Expensive evidence still deferred |
|---:|---|---|---|
| 0 | **Complete.** Baseline tag/SHA and migration documents exist. | Later ledger synchronization only. | None. |
| 1 | **Complete.** Cargo workspace and opaque bridge skeleton exist. | Final ABI narrowing belongs to Phase 8. | None. |
| 2 | **Partial.** Safe `Model`/`Session` wrappers, payload seams, batch context, safe static batch result ownership, and model-open tuning exist. | Rust must own production model/session/backend lifecycle without the broad C host bridge. | Final ownership/unsafe audit and integrated lifecycle proof. |
| 3 | **Partial.** CLI generation/sampling/thinking, benchmark local/distributed/chat/tuning paths, and an increasingly capable agent shadow exist. | CLI REPL/TTY/batch/MTP; benchmark MTP/output-head/frontier-logit modes; remaining agent tools, compaction, approval, interactive/KV/MTP/dist modes. | Full binary argument/output/KV/perf parity. |
| 4 | **Partial/active.** KVC format/index/staged writer, serial cold/continued/tool-map, width-1 bank replay, multi-bank set, continuation protection, and protected saturation handling exist. | Default periodic 10,000→10,240 checkpoint path, live bank-evict completion, partial-prefix fork, pin/claim/directed tool turn, thinking/extensions, and serial emergency reclaim need code completion where absent. | Default-policy 4-way/live multi-bank/ABBA/RSS/soak and remaining family/surface cells. |
| 5 | **Partial.** Blocking web crate plus agent Google search, page visit, byte-safe read, and stateful more are Rust-integrated. | Agent list/search/write/edit/bash/status/stop, approval, compaction, terminal visualization, and full lifecycle. | Full C/Rust tool transcript and live agent generation comparison. |
| 6 | **Partial.** Explicit codecs, blocking runtime, CLI lifecycle, distributed bench, snapshot bodies, and coordinator streaming transport exist. | Worker snapshot receive/send, native session payload integration, reconnect/pipelined prefetch, relay/forwarding threads, telemetry, and Rust server integration. | Real coordinator/worker snapshot, reconnect, distributed inference, memory, and performance gates. |
| 7 | **Partial.** Four parsers/projectors, serial and continuous lanes, FIFO owner, stateless Anthropic/Responses continuous, KV integrations, bank set/protection, and clean saturation exist. A safe static core call exists. | Static server owner/coalescing/fallback; true multi-client rolling width; partial fork/pin/claim/tool continuation; bank thinking/extensions; serial emergency reclaim; remaining streaming continuation edges. | Full four-surface × three-lane fixtures, disconnect/backpressure, live GPU, TTFT/RSS/soak. |
| 8 | **Partial.** Metadata, mmap GGUF catalogs/bind plans, tokenizer/templates/stops, validation, host ledger, and selected sibling binding are Rust-owned. | Production model/session/scheduler execution, sibling pointer assignment, CUDA weight upload orchestration, backend dispatch, and final narrow native ABI remain C-owned. | Family regression, ownership shutdown order, memory residency, unsafe/ABI audit. |
| 9 | **Not started.** C binaries remain default. | Atomic default/oracle rename and build/proof rewiring only after Phase 2–8 and integrated gates close. `ds4-eval` remains C unless separately decided. | Pre/post family manifest, full proof/perf/soak, exact binary mapping. |

The repository split remains blocked after Phase 9. This snapshot is not a
genesis candidate.

## 8. Ordered implementation TODO before the consolidated gate

Each checkbox is a code slice, not a phase-green declaration. Keep the C oracle
and focused tests beside each port.

### 8.1 Resume hygiene

- [ ] Confirm `rust-host`, local HEAD, and `origin/rust-host` still match the
  snapshot or review every later commit before using this list.
- [ ] Run `git status --short --branch`; preserve unrelated/untracked user files.
- [ ] Re-read the always-read references in §5.1 and the subsystem references
  for the selected slice.
- [ ] Update this snapshot only if the work order or fixed decisions change.

### 8.2 Finish the agent and web utility surface (Phase 3/5)

Port one tool family per commit, reusing the current C oracle:

- [ ] `list` exact path, sort/order, hidden-entry, error, and observation bounds.
- [ ] local `search` modes, glob/case/context, binary/NUL, limits, and errors.
- [ ] `write` with approval and exact failure/overwrite behavior.
- [ ] `edit` preflight, single/multiple match behavior, byte preservation, and
  approval semantics.
- [ ] `bash`, `status`, and `stop` process lifecycle, output bounds, exit/signal,
  cancellation, and cleanup.
- [ ] context compaction according to `misc/COMPACT.md`.
- [ ] interactive/stdin-repeat, terminal visualization, KV/resume, MTP, and
  distributed agent modes.

Immediate low-risk resume slice: port `list`, add the smallest C-oracle matrix,
run agent parity plus native compile, format, commit, and push. Port local
`search` separately afterward.

### 8.3 Finish CLI and benchmark shadows (Phase 3)

- [ ] CLI MTP option/lifecycle parity.
- [ ] CLI REPL, TTY color/stream formatting, batch input, and distributed mode
  parity without changing public flags.
- [ ] Benchmark MTP mode.
- [ ] Benchmark output-head and frontier-logit dump modes.
- [ ] Confirm chat/raw prompt, quality, warm-weight, power, local/distributed,
  snapshot/restore, CSV columns, help, and error text cover every C mode.

### 8.4 Complete distributed production behavior (Phase 6)

- [ ] Implement worker-side SNAPSHOT save/load frame handling against the exact
  codecs and coordinator transport already in Rust.
- [ ] Bind snapshot streams to native session payload save/load with bounded
  memory and exact status/error strings.
- [ ] Port worker reconnect and
  `DS4_DIST_WORKER_PREFETCH_DEPTH` pipelined prefetch behavior.
- [ ] Port worker-to-worker relay/forwarding threads and shutdown ownership.
- [ ] Port distributed telemetry/state reporting.
- [ ] Integrate the Rust distributed runtime into `ds4-server-rs` without wire
  or scheduler redesign.

### 8.5 Complete server and continuous/static ownership (Phase 4/7)

- [ ] Build the static-lane Rust owner/coalescer around the safe
  `StaticBatchContext` boundary.
- [ ] Preserve C `n >= 2`, ragged outputs, per-call fallback, queue timing,
  terminal settlement, and error behavior.
- [ ] Run multiple client jobs in one rolling continuous scheduler rather than
  one job-to-completion behind the owner.
- [ ] Complete multi-bank partial-prefix fork, continuation pin/claim, directed
  tool-turn placement, and protected saturation/retry behavior.
- [ ] Integrate bank tool maps, thinking state, extension records, periodic
  checkpoint, eviction, shutdown, and restore across all required surfaces.
- [ ] Implement the default configured 10,000-token/effective aligned
  10,240-token periodic checkpoint path.
- [ ] Complete live bank-evict and serial emergency-reclaim ownership.
- [ ] Finish stateful Anthropic/Responses tool continuation and remaining stream,
  disconnect, backpressure, and retry edges.

### 8.6 Close the Phase 8 ownership boundary

- [ ] Move production `Model` ownership and shutdown ordering into Rust while
  retaining mmap and native CUDA allocation behavior.
- [ ] Move production `Session`, KV, scratch, continuation, and batch-bank
  lifecycle ownership into safe Rust types.
- [ ] Replace C sibling pointer assignment with a Rust-owned, oracle-compatible
  lifecycle.
- [ ] Move graph/backend dispatch orchestration needed by the production CUDA
  path without rewriting kernels.
- [ ] Reduce the broad bridge to the planned native backend operations:
  create/load/session/prefill/decode/KV primitives/destroy.
- [ ] Audit `rg -n 'unsafe \{' crates/`; justify or relocate every application
  layer hit.
- [ ] Preserve Metal compile/basic ABI smoke and C CPU reference access.

## 9. Cheap gate required before each functional commit

Select the smallest applicable set, then format. Do not load an 80+ GiB model
for every leaf commit.

```bash
cargo fmt --all -- --check
cargo test --workspace --no-default-features
cargo check --workspace --all-targets

make test-kv-parity
make test-web-parity
make test-dist-parity
make test-server-parity
make test-catalog-parity
make test-tokenizer-parity
make test-session-parity
make test-agent-parity

make rust-bridge
make ds4-rs
make ds4-bench-rs
make ds4-agent-rs
make ds4-server-rs
```

Do not run every target blindly for a trivial slice. Run the focused crate/unit
test, its C oracle, the affected native link target, and workspace rustfmt.
Run the broader no-default workspace sweep when crate boundaries or shared types
change.

## 10. Consolidated integrated gate after code completion

This gate is deferred, not waived. Record exact commit, binary hashes, model
revision, quant revision, GPU, driver, CUDA, context, width/batch, KV policy,
environment, and artifacts.

### 10.1 Static and cross-language correctness

- [ ] `cargo fmt --all -- --check`.
- [ ] Full workspace tests and all parity targets in §9.
- [ ] C-save→Rust-load, Rust-save→C-load, Rust-save→Rust-load, and C-save→C-load
  for ordinary, continued, tool-map, periodic, eviction, shutdown, and partial
  bank records.
- [ ] Exact tokenizer encode/decode/special/template/stop vectors.
- [ ] Unsafe/FFI lifetime, NULL, overflow, partial I/O, cleanup, and stale-build
  ABI checks.

### 10.2 Model-family regression

Use Makefile variables, not invented flags:

```text
Motif-3:  test-motif3-{loader,reference,tokenizer,cuda,resident,batch}
Solar:    test-solar-{loader,tokenizer,forward,session,kda,kda-prefill,
          kda-chunk,gates,kv}
K-EXAONE: test-exaone-{ref,kernels,batch}
Dots3:    test-dots3-{loader,tokenizer,resident}
Shared:   test-model-family-kernels, test-mmq-parity
```

- [ ] Run one resident model at a time through tmux and the guarded runner.
- [ ] Preserve the pre/post binary, model, environment, token, and memory
  manifest for Phase 9 comparison.

### 10.3 API and serving lanes

- [ ] Run all four surfaces:
  `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`.
- [ ] Cover serial, continuous, and static routing according to the API matrix.
- [ ] Verify schema, event order, IDs, finish/incomplete semantics, errors,
  tool continuation, trust domain, route metrics, disconnect, timeout,
  backpressure, admission, and terminal settlement.
- [ ] Run true concurrent clients with barrier-synchronized starts and prove
  effective rolling width; a listening port is not readiness.

### 10.4 CUDA proof and long context

```bash
make proof-cuda-smoke
make proof-cuda-long
make proof-cuda-opp-c
make proof-rust-cuda-opp-c
```

- [ ] Keep captured-vs-eager long-context parity as a release gate.
- [ ] Do not update a golden merely to absorb a regression.
- [ ] Keep the C baseline/native-drift and C→Rust host-parity OPP-C gates distinct.

### 10.5 Performance, memory, and soak

- [ ] Use C→Rust→Rust→C or equivalent ABBA ordering on the same workload.
- [ ] Prefill must be at least 97% of C.
- [ ] Decode must be at least 98% of C.
- [ ] TTFT must be no more than 5% above C.
- [ ] Host RSS must be no more than 5% above C.
- [ ] GPU residency must show no meaningful increase.
- [ ] Explain even a 2–3% regression; inspect extra copies, allocations, lock
  scope, FFI granularity, and scheduler changes.
- [ ] Include default KV policy, true multi-client continuous, static batching,
  distributed execution, long context, and server soak.

## 11. Phase 9 and split work that must remain blocked

Only after every Phase 2–8 code item and §10 gate is green:

1. freeze the exact promotion SHA and evidence manifest;
2. preserve explicit C oracle names and proof paths;
3. atomically map `ds4`, `ds4-server`, `ds4-bench`, and `ds4-agent` to Rust;
4. retain `*-c` reference/oracle binaries through the split gate;
5. keep `ds4-eval` C unless a separate, recorded decision changes it;
6. rerun the same family/API/KV/proof/performance/soak manifest after promotion;
7. update `STATUS.md` and `PARITY_MATRIX.md` to green only from evidence;
8. create `SPLIT_READINESS.md` with the recommended immutable genesis SHA;
9. only then begin the inactive repository-split plan.

Do not delete legacy C host files merely because a Rust implementation exists.
Deletion requires Rust implementation, unit parity, live parity, performance,
and soak to be green.

## 12. Current blockers and risk register

There is no known external decision or user action blocking the next cheap code
slice. The remaining time is mostly implementation breadth plus expensive final
evidence, not a missing credential or repository permission.

| Risk | Control |
|---|---|
| Formal ledgers are stale after `042562b` | Use this snapshot for current code; later synchronize `STATUS.md` and `PARITY_MATRIX.md` together. |
| Large model loads are slow and consume roughly 80–100 GiB | Defer to consolidated gate; one resident at a time; guarded tmux run; clear cache only after exit. |
| Static lane has a safe core call but no Rust server owner | Port owner/coalescing/fallback as a separate Phase 7 slice. |
| Distributed snapshot is coordinator-only | Implement worker and native session payload counterpart before claiming production integration. |
| Broad FFI still exposes C host behavior | Narrow only after Rust lifecycle ownership exists; do not churn symbols prematurely. |
| Multi-bank unit behavior can hide scheduler serialization | Require barrier-synchronized live clients and effective-width evidence. |
| Untracked plans/handoff are referenced by this committed index | Preserve them locally and explicitly decide whether to commit them in a later documentation-only change. |
| Public API additions can widen the compatibility surface | Add only narrow options needed to reproduce an existing C contract, with C error/default parity. |

## 13. Resume command checklist

```bash
git switch rust-host
git status --short --branch
git rev-parse HEAD
git rev-parse origin/rust-host
git rev-parse 'v0.6.3-dfm^{commit}'

cargo fmt --all -- --check
```

Before editing, inspect the selected C source, every caller, its Rust shadow,
existing helpers, and the smallest relevant C oracle. After the focused gate:

```bash
git diff --check
git diff --stat
git status --short
```

Stage explicit paths only. The commit body must record:

```text
Migration area:
C source replaced:

Correctness:
- ...

Performance:
- ...

Known remaining C dependency:
- ...
```

Then push the functional commit to `origin/rust-host`. Do not bundle status
rewrites, unrelated cleanup, optimization, generated binaries, or user files.

## 14. Intentional stop point

The repository is paused immediately after `5abdd53` with all tracked work
pushed. No implementation process or gate should be resumed merely by reading
this document. The first recommended code slice on explicit resume is the
agent `list` tool; the next structural Phase 6 slice is worker-side snapshot
handling. Both must remain separate commits.
