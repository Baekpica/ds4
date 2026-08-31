# Migration status

> **Freshness (2026-08-31 KST):** the `v0.6.5-dfm`/Qwen campaign is green.
> The current 60-cell host matrix is PASS 57 + PASS* 3 (C-shared E-2, E-3,
> E-6), with no Rust-only failure or blocked cell. Qwen Q5+Sidecar passed the
> configured/exact 262K gates, KV four-way, MTP/static checks, ABBA, and the
> only required two-hour soak. The live family matrix is frozen at `eb4ba77`;
> the final 13/13 cheap/parity audit, clippy, `DS4_SERVER_CLIENT_SNDBUF`
> compatibility fix, and all default/C oracle builds pass at `d126e56`.
> Split readiness is recorded in
> [SPLIT_READINESS.md](SPLIT_READINESS.md). DeepSeek retained its ordinary
> functional/parity/proof/performance gates and did not repeat a long soak.

Update this table in the same commit that changes a subsystem’s
state. Colors: `green` = gate passed, `yellow` = partial / in
progress, `—` = not started or not applicable, `n/a` = axis does
not apply.

C golden baseline: `v0.6.5-dfm` (`d02e2a4`), with Qwen observable behavior
frozen at C cut `4d40d97`.

## Subsystem matrix

| Subsystem | C Oracle | Rust | Default | Parity | Perf |
|---|---|---|---|---|---|
| model metadata | yes | yes (catalog + DeepSeek select + arch dispatch + bind names + MTP/DSpark sibling catalogs + host `config_validate` + host `weights_validate_layout`) | Rust host over native load | catalog + bind-name + MTP/DSpark support + validate + layout token dump green (`make test-catalog-parity`) | n/a |
| GGUF loading (mmap) | yes | metadata mmap + identify + tensor inventory + split-shard remap + host `weights_bind` catalog / bind-plan check + host tensor-dir apply (native skips `parse_tensors` when installed) + host `config_validate` (native applies pinned shape/compress and skips C validate when installed) + host vocab apply (native skips `vocab_load` when installed) + host bind map (native skips the `model_find_tensor` name walk when installed) + host `weights_validate_layout` (native skips the main-model layout check when the bind map is installed) + host MTP/DSpark sibling name/layout catalogs + sibling BindPlan resolve/validate + **sibling bind map FFI** (`Model::open_with_support` / `opt->mtp_path|dspark_path|mtp_bind|dspark_bind`: native swaps each sibling map into the active slot around its own open+bind window and skips that sibling's C layout check; sibling pointer assignment stays C); host tables are cleared after base `weights_bind` so sibling `model_open` does not reuse the base GGUF tables; CUDA weight upload still C | Rust host (`ds4_bridge_model_open` performs native upload) | synthetic v3 identify + tensor/nbytes/sibling + bind-name/check/match + MTP/DSpark support dump + consume/apply tapes + validate token tapes + vocab apply tapes + bind lookup tapes + layout SPEC/check/support tapes green (`make test-catalog-parity`); live Motif-3-MQ87-88-FIT identify/validate/open green; live DeepSeek-V4-Flash + DSpark-drafter-Q2K-Q8 (81/81 slots FOUND, `dspark-flash` catalog): C env-fallback run kept the strict C validate (`Hi`, 29.27/21.27 t/s), Rust `--dspark` open bound the drafter through the host map + layout skip and decoded 9 ids (`scratch/rust-host-live/{c,rust}-dspark.log`) | — |
| tokenizer | yes | host-owned encode/decode/special/stop and family chat framing; native token helpers remain for C oracles | Rust host | synthetic C↔Rust and all six production-family tokenizer gates green | — |
| session lifecycle | yes | host ledger, ownership, continuation, and bounded DSV4 prefix/range policy; CUDA session execution and payload tail remain native | Rust host | synthetic ledger/range/error oracles plus live C↔Rust KV, continuation, bank, and exact-context gates green | — |
| KV store | yes | Rust envelope/policy/index/staged writer/KTM handling integrated in serial and bank lanes; opaque GPU payload stays native | Rust host | ordinary/continued/tool-map/periodic/evict/partial/shutdown and Qwen recurrent/image four-way gates green | Motif restore ABBA and Qwen text/image ABBA green |
| web utility | yes | `ds4-web` blocking I/O is used by the Rust default agent | Rust host | encode/wire, mock CDP, and agent projector parity green | n/a |
| server (four surfaces) | yes | Rust owns HTTP parsing/rendering, admission, metrics, FIFO owner, serial/continuous/static routing, continuation, and disk-KV policy over native execution | Rust host | DeepSeek four surfaces × three lanes, Qwen text/image surfaces, barriers, static/serial, invalid inputs, and 262K live gates green | Motif and Qwen ABBA green; Qwen two-hour soak green |
| distributed runtime | yes | explicit codecs and blocking coordinator/worker orchestration integrated in Rust CLI/server; layer eval and GPU snapshots stay native | Rust host | codec, route, assemble/reconnect, prefetch, snapshot gather/scatter, and live matrix cells green | n/a |
| CLI / bench / agent host | yes | Rust defaults for CLI, benchmark, agent, and server; C implementations retained as `*-c` oracles | Rust host | sampling/thinking/agent/projector parity, default-name build, and local/live gates green | local benchmark and server ABBA green |
| CPU reference backend | yes | no (not a cut-over blocker) | C | — | n/a |
| Metal backend | native | unchanged | native | — | n/a |
| CUDA / MMQ / VMM | native | unchanged | native | six-family restamp, Qwen Q5+Sidecar kernels/batch, shared MMQ, and resident ownership gates green or C-shared annotated | Qwen C/Rust text+image ABBA and exact 262K green |
| FFI bridge (`ds4_bridge`) | n/a | opaque strangler ABI; current 68-function surface frozen, with CUDA pointers/types hidden from safe Rust | native boundary | error/range/callback/panic/snapshot/bank/continuous/Qwen image oracles green; no unplanned bridge growth | — |
| proof harness on Rust path | C binaries | yes | Rust default + `ds4-c` oracle | smoke 2/2, long 6/6, tracked v0.6.5 OPP-C 5/5, and C→Rust OPP-C 5/5 green; identical-hash/inode false-green guard active | — |

Rust entries above describe isolated ownership/parity unless `Default` is
Rust. They must not be read as production-path integration.

## Phase checklist

| Phase | Name | State |
|---|---|---|
| 0 | Freeze baseline + this document set | **done** (docs-only commit) |
| 1 | Cargo workspace + FFI skeleton | **done** (`cargo check --workspace`, `make rust-bridge`) |
| 2 | `ds4-core` safe wrappers | **done for host scope** — Rust wrappers own model/session handles and lifecycle; opaque CUDA execution remains native |
| 3 | Shadow `ds4-rs` / `ds4-bench-rs` / `ds4-agent-rs` | **done and promoted** — CLI sampling/thinking, benchmark ABBA, agent prompt/projector, live tools/KV/TTY, MTP, and distributed routing are covered by the current gates; legacy `*-rs` names remain aliases |
| 4 | KV store port + 4-way matrix | **done** — format/policy, ordinary/continued/tool-map/bank bodies, periodic/evict/partial/shutdown behavior, and C↔Rust four-way reads are green |
| 5 | Web utility port | **done** — blocking `ds4-web` is production-integrated in the Rust agent; parity green |
| 6 | Distributed runtime port | **done for host scope** — Rust owns codecs and blocking orchestration; native layer execution/snapshots remain backend primitives |
| 7 | Server shadow by feature | **done** — four surfaces, three lanes, KV/continuation, Qwen image/MTP, barriers, and failure paths re-stamped |
| 8 | `ds4.c` decomposition | **done for split scope** — host catalog/tokenizer/lifecycle/policy ownership is Rust; native CUDA/session execution intentionally remains behind the bridge |
| 9 | Promote Rust binaries to default names | **done and v0.6.5 revalidated** — defaults are Rust, `*-c` remain distinct oracles, 60 logical cells are green, and only Qwen runs the long soak |
| split | `SPLIT_READINESS.md` + `ds4-dfm-rs` code genesis | **green for seeding** — readiness document exists; scaffold replacement remains the next repository operation |

## Current default binaries

| Name | Implementation |
|---|---|
| `ds4` | Rust host, same C CUDA core (Cargo bin still `ds4-rs`) |
| `ds4-server` | Rust HTTP host over native CUDA (`--listen`) |
| `ds4-bench` | Rust bench host |
| `ds4-agent` | Rust agent host |
| `ds4-eval` | C (unchanged) |
| `ds4-c` | C CLI oracle |
| `ds4-server-c` | C HTTP oracle (`--host`) |
| `ds4-bench-c` | C bench oracle |
| `ds4-agent-c` | C agent oracle |
| `ds4-rs` / `ds4-server-rs` / … | Deprecated aliases (copy of the Rust defaults) |
| `ds4_weight_server` | native CUDA (unchanged) |

The resumed Qwen evidence is in
[QWEN_V065_RESTAMP_2026-08-31.md](QWEN_V065_RESTAMP_2026-08-31.md) and
`scratch/rust-host-live/qwen-v065-restamp-20260831/`. It uses only the primary
Q5 main GGUF plus shared SSD-PLE sidecars. C→Rust→Rust→C returned exact text
and image outputs; Rust/C means were 99.97% prefill, 100.00% decode, +0.95%
TTFT, +4.40% worker VmHWM, and identical worker GPU residency. All target
processes exited before each cache clear; final `MemAvailable` was 119 GiB.
The promotion hard gate then passed 3,610/3,610 requests over 7,202.3 seconds
with 158 barrier batches and 79 image requests, exact four-way KV restore, and
zero swap. Exact 262K C/Rust direct logits had 248,320 finite values, argmax
198, and zero f32 mismatches; MTP improved token-per-step from 1.01 to 2.00
with identical output, and forced static routing recorded static=2 only.

The following incremental Phase 3–7 paragraphs are retained as historical
`v0.6.3-dfm` evidence. Phase 3/7 live Motif evidence is in
`scratch/rust-host-live/` (tmux + workspace-local `../scripts/guarded-run.sh`,
outside this repository; sequential C then Rust, `clear_cache` between loads).
Live CUDA census on `ds4-server-rs` is green (`census_supported=1`, epoch 1636,
weight_artifact 86.07 GiB, observation ok). Live DSpark sibling FFI is green
(DeepSeek-V4-Flash-IQ2XXS + DSpark-drafter-Q2K-Q8: C env fallback `Hi`
29.27/21.27 t/s with strict C validate; Rust `--dspark` host-map bind + layout
skip, 9 decode ids; sequential, teardown + `clear_cache`). The Rust server
continuous lane produced byte-equal output in the recorded same-lane ABBA.
Any “pending” scope statement in those historical paragraphs describes that
older checkpoint and is superseded by the current 60-cell result above.

Phase 4 live ordinary replay evidence is under `scratch/rust-host-live/rust-kv-no-think-final-IHtJue/`. With Motif-3 MQ87-88, ctx 8192, temp 0, no thinking, no tools, and the shared `cold=0, continued=0` policy, C and Rust produced the same 114-token answer and a byte-identical visible-text/payload KVC body (301,972,392 bytes). C→Rust, Rust→C, Rust→Rust, and C→C restart loads all returned `RESTORED_OK` with 6,896 cached + 15 newly evaluated prompt tokens. Rust binary `dae70ae2...` ran the four-way matrix; post-fix `52471e49...` also matches C timing counts (`prefill_tokens=15`, `prefill_cached_tokens=6896`). The initial Rust 1,590--1,715 ms TTFT regression was the deferred native boot prewarm never being called. Commit `f4a632e` restored C's `batch fit → prewarm → listen` ordering; `scratch/rust-host-live/prewarm-restore-LH1LIp/` records C→Rust→Rust→C TTFT 880.9/836.0/860.4/897.5 ms (Rust mean 4.61% below C). Commit `4fc8ef4` also narrowed reported prefill time to the computed suffix sync; the live Rust result is 69.2 tok/s versus C 68.2--69.0.

The C-default ordinary cold slice is live-green under `scratch/rust-host-live/rust-cold-default-0IeCd9/`. The 6,782-token Motif request produced the same 6,144-token cold partition in C and Rust: rendered text SHA-256 `ebd540ec...` and payload SHA-256 `0f0af1d1...` are exact. Rust→Rust, Rust→C, and C→Rust restart cells restored 6,144 cached tokens and produced the same 106-token output (`49b5f8a8...`) with TTFT 2,038.4/2,078.9/2,018.0 ms. This proves cold correctness and cross-read, not a full default-policy ABBA performance gate.

The ordinary serial continued final-sync call/order is CPU-green, while the decode frontier is live-green under `scratch/rust-host-live/continued-fourway-oJTYrf/`. With `cold=0`, `continued=6800`, trim 0, and align 1, C and Rust wrote the same 300,423,338-byte reason-continued record at 6,800 tokens (text SHA-256 `02523123...`, payload SHA-256 `be4d59a9...`) and emitted the same 114-token answer. A Motif history fixture that explicitly preserves the live `<think></think>` bytes then passed C→Rust, Rust→C, Rust→Rust, and C→C: every cell returned `RESTORED_OK`, prompt/cached/computed/output counts 6,911/6,800/111/4, and semantic response SHA-256 `f61bf763...`. The plain assistant-history fixture is an expected C/Rust miss because Motif drops the empty think pair when closing history. This proves the scoped final/decode slice, not intermediate prefill, continuous-bank checkpoints, or full default-policy performance. See PARITY_MATRIX for hashes and binary provenance.

The scoped DeepSeek ordinary-serial intermediate-prefill gate is live-green
under `scratch/rust-host-live/intermediate-prefill-fourway-40gDCF/` at
`8361116`. A deterministic 6,782-token no-think/no-tools OpenAI Chat prompt made the
4,096-token continued frontier durable during native prefill; the final prompt
and five-token replay output remained below the next 8,192 frontier. C and
Rust wrote the same 41,956,589-byte reason-continued record: model 0, ext 0,
ctx 8,192, 14,617 text bytes and 41,941,920 payload bytes. Text SHA-256 is
`2dfc39c82411c86cadeabdfc517d0450054a0cba7c32bf4afb2b5f197dea9384`
and payload SHA-256 is
`00d8719acee5341f3e92feb45347a43fa41d11f0102148c1775feffcdd318f1c`
for both producers. Fresh C→Rust, Rust→C, Rust→Rust, and C→C loaders all
returned `RESTORED_OK` with prompt/cached/computed/output counts
6,782/4,096/2,686/5 and the serial route. This proves the scoped DeepSeek CUDA
ordinary-serial prefill callback/save/cross-read slice only; it does not itself
cover the separately green width-1 bank lane. Other families/surfaces and
default-policy ABBA performance remain pending.

The scoped ordinary-serial tool-map gate is live-green under
`scratch/rust-host-live/tool-fourway-20260825/`. C and Rust DeepSeek producers
both emitted `pair_values({"a":1,"b":2})`, then an unrelated request evicted a
424-token reason-evict record. Both records use filename
`29cf875c11df369237f528cb8155b8d1979f5b32.kv`, `ext_flags=1`, identical
rendered-text SHA-256 `20e492de...`, identical payload SHA-256 `f427f926...`,
and a valid one-entry 297-byte `KTM\x01` trailer containing the exact sampled
244-byte DSML. The trailers differ only in the process-unique wire ID; header
timestamps are run-specific. Fresh loaders were
given the same full history with arguments deliberately reordered to
`{"b":2,"a":1}`. C→Rust, Rust→C, Rust→Rust, and C→C all returned
`RESTORED_OK` with prompt/cached/computed/output counts 452/424/28/5 and a
serial route. This proves the scoped OpenAI Chat, DeepSeek, no-think,
ordinary-serial slice only; Anthropic/Responses, bank tool-map integration,
combined extension records, and default-policy performance remain pending.

The scoped width-1 continuous-bank gate is live-green under
`scratch/rust-host-live/bank-fourway-20260825-093322/` at `15b016c`, with
native timing porcelain corrected at `98d81b9`. The Motif-3 OpenAI Chat,
no-think/no-tools producers ran at ctx 8,192 with `cold=0`, `continued=0`, and
periodic bank checkpoints disabled. C and Rust wrote the same
301,972,407-byte `3faf064c1bf3e92ef70f356f5c1c7baeb0dd62bc.kv`: reason/ext
8/16, model 3, 6,896 tokens, 24,591 text bytes, 301,947,764 payload bytes,
and no trailer. Rendered-text SHA-256 is `bb9a55e9...` and payload SHA-256 is
`df417b44...` for both producers. Fresh C→Rust, Rust→C, Rust→Rust, and C→C
loaders all returned `RESTORED_OK` on the continuous route with
prompt/cached/computed/output counts 6,911/6,896/15/4. The 114-token
C→Rust→Rust→C restore ABBA produced exact completion text (`faf7c021...`):
C/Rust mean prefill was 76.05/76.25 tok/s, decode 14.90/14.75 tok/s, and TTFT
386.85/387.90 ms. The ABBA Rust binary was `287c821a...`. Post-fix Rust binary
`6c25d952...` then repeated the short C→Rust loader cell at 12.3 tok/s and
1.25 tok/step, matching the C cells' 12.0--12.1 tok/s and 1.25 tok/step.
This proves the scoped width-1 bank-shutdown body/cross-read and
throughput/TTFT gate only. Default periodic reason-bank-checkpoint,
reason-bank-evict, multi-bank fork/partial and pin/claim behavior, other
families/surfaces, peak host RSS, and full default-policy performance remain
pending.

The Rust benchmark smoke is in `scratch/rust-host-live/bench-rs-two-frontier.log` (GB10, DeepSeek-V4-Flash IQ2XXS, binary SHA-256 `a66d97c4...`). It completed 64 and 128 token frontiers with four decode tokens each, nonzero checkpoint sizes, exit 0, and post-run cache cleanup. These two rows prove execution and restore flow only; they are not a performance comparison.

CLI sampling evidence is under `scratch/rust-host-live/cli-sampling-parity/` (Rust binary SHA-256 `1287e404...`): C and Rust fixed-seed stdout both produced 52 bytes with SHA-256 `ae12f463...`. The post-change greedy rerun under `cli-greedy-post-sampling/` produced byte-identical `OK\n` (`a12b7cb4...`). Both pairs exited 0 with teardown and cache cleanup between model loads.

CLI thinking evidence is under `scratch/rust-host-live/cli-thinking-parity-20260825/`: with the same fixed-seed DeepSeek V4 Flash workload, C and Rust `--think` stdout are both 292 bytes (`ad6d107f...`) and C and Rust `--nothink` stdout are both 52 bytes (`ae12f463...`). All four runs exited 0; each sequential teardown and cache clear left no GPU application.

Rust proof evidence after the rendered-prompt fix is in `scratch/rust-host-live/proof-{smoke,long}-fixed.log`, `proof-oppc-{c-oracle,rust-fixed}.log`, and `proof-oppc-rust-current.log`: smoke 2/2, long 6/6, and same-host OPP-C C/Rust 5/5 all pass. The current v0.6.5 proof re-stamp is indexed by [PARITY_MATRIX.md](PARITY_MATRIX.md): smoke 2/2, long 6/6, tracked OPP-C 5/5, and C→Rust OPP-C 5/5. The older generic and v0.6.3 snapshots remain unchanged.

## Intentionally retained native dependency

The default CLI, server, bench, and agent hosts are Rust. Native code retains
CUDA/MMQ/VMM/Metal kernels, graph and memory primitives, weight upload, opaque
GPU session execution, `ds4_weight_server`, `ds4-eval`, and the explicit
`*-c` parity oracles. Rust owns the model/session handles and lifecycle policy,
HTTP/API, scheduling/admission, KV metadata and persistence policy, memory
policy, and distributed orchestration across that boundary. This is the
planned Strangler boundary, not unfinished host ownership.

## Notes

- `dfm` is not merged into `rust-host`. Cherry-pick CUDA
  correctness or performance fixes only, then rerun this matrix.
- `SPLIT_READINESS.md` was not added until the §26 required list became
  evidence-green.
- Here, pre-split readiness means the migration instruction's §26 gate.
  `DFM_RS_SPLIT_PLAN.md` §26 is the later post-split release gate.
- Rust 1.98.0 has `rustfmt` and `clippy`; fmt, workspace tests/checks, and
  clippy exit 0 at `d126e56`. Existing non-denied warnings remain visible in
  the audit log and do not hide a failed command.
- Unsafe-audit command: `rg -n 'unsafe \{' crates/`
  Current count is 97: 61 in reviewed `ds4-core` FFI/mmap/native adapters,
  27 in localized CLI POSIX/linenoise adapters, 7 in `ds4-sys` (including its
  socket regression test), and 2 in distributed test-only raw-FD fixtures.
  Production `ds4-server`, `ds4-kv`, `ds4-dist`, and `ds4-web` contain none.
- `DS4_SERVER_CLIENT_SNDBUF` is applied through the safe `ds4-sys` socket
  adapter at `d126e56`; its real-socket regression and full server C-parity
  suite pass.
