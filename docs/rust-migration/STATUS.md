# Migration status

> **Freshness (2026-08-31 KST):** the post-promote campaign was reopened for
> the `v0.6.5-dfm`/Qwen delta. Q5+Sidecar behavior, family fixtures, and
> C→Rust→Rust→C ABBA are green. The Qwen-only two-hour soak, configured-262K
> live smoke, exact 262K direct logits, KV four-way, MTP target, and static
> lane are green through `8006198`. Default names remain Rust and C oracles
> remain `*-c`. The complete cross-family/proof re-stamp and
> `SPLIT_READINESS.md` are still pending. DeepSeek keeps its normal
> functional/parity/performance gates but does not repeat a long soak.

Update this table in the same commit that changes a subsystem’s
state. Colors: `green` = gate passed, `yellow` = partial / in
progress, `—` = not started or not applicable, `n/a` = axis does
not apply.

C golden baseline: `v0.6.5-dfm` (`d02e2a4`), with Qwen observable behavior
frozen at C cut `4d40d97`.

## Subsystem matrix

| Subsystem | C Oracle | Rust | Default | Parity | Perf |
|---|---|---|---|---|---|
| model metadata | yes | yes (catalog + DeepSeek select + arch dispatch + bind names + MTP/DSpark sibling catalogs + host `config_validate` + host `weights_validate_layout`) | C (`ds4_bridge_model_open`) | catalog + bind-name + MTP/DSpark support + validate + layout token dump green (`make test-catalog-parity`) | n/a |
| GGUF loading (mmap) | yes | metadata mmap + identify + tensor inventory + split-shard remap + host `weights_bind` catalog / bind-plan check + host tensor-dir apply (native skips `parse_tensors` when installed) + host `config_validate` (native applies pinned shape/compress and skips C validate when installed) + host vocab apply (native skips `vocab_load` when installed) + host bind map (native skips the `model_find_tensor` name walk when installed) + host `weights_validate_layout` (native skips the main-model layout check when the bind map is installed) + host MTP/DSpark sibling name/layout catalogs + sibling BindPlan resolve/validate + **sibling bind map FFI** (`Model::open_with_support` / `opt->mtp_path|dspark_path|mtp_bind|dspark_bind`: native swaps each sibling map into the active slot around its own open+bind window and skips that sibling's C layout check; sibling pointer assignment stays C); host tables are cleared after base `weights_bind` so sibling `model_open` does not reuse the base GGUF tables; CUDA weight upload still C | C (`ds4_bridge_model_open` upload) | synthetic v3 identify + tensor/nbytes/sibling + bind-name/check/match + MTP/DSpark support dump + consume/apply tapes + validate token tapes + vocab apply tapes + bind lookup tapes + layout SPEC/check/support tapes green (`make test-catalog-parity`); live Motif-3-MQ87-88-FIT identify/validate/open green; live DeepSeek-V4-Flash + DSpark-drafter-Q2K-Q8 (81/81 slots FOUND, `dspark-flash` catalog): C env-fallback run kept the strict C validate (`Hi`, 29.27/21.27 t/s), Rust `--dspark` open bound the drafter through the host map + layout skip and decoded 9 ids (`scratch/rust-host-live/{c,rust}-dspark.log`) | — |
| tokenizer | yes | yes (host-owned encode/decode/special/stop and family chat-transcript token framing + host vocab apply so native skips `vocab_load` when installed; `Model` tokenize/stop/eos/`token_text` use host `Vocab`; `ds4-server-rs` uses `Model::vocab`; FFI tokenize remains for the C engine path) | C | synthetic C↔Rust encode/decode/stop + apply tapes green; family chat transcripts match C; Qwen official text/chat goldens and stop set pass on the production Q5 GGUF; live Motif specials + encode `hi`→8320 | — |
| session lifecycle | yes | partial (host ledger + DSV4 prefix, including bounded embedded-range restore; engine session, GPU/logits tail, and payload execution remain native) | C | synthetic C↔Rust green (`make test-session-parity`); range prefix/EOF + C seek/length/overflow oracle green; live same-model CUDA payload pending | — |
| KV store | yes | envelope/policy crate with payload-free metadata index, bounded LCP text reads, a file-backed staged payload writer, and bounded KTM trailer memory; ordinary serial visible cold + evict/replay, final-sync/decode and DeepSeek intermediate-prefill continued checkpoints, scoped DeepSeek no-think tool-map replay, and scoped width-1 OpenAI Chat no-think/no-tools continuous-bank shutdown persistence/replay are production-integrated in `ds4-server-rs` | C | file-format 4-way + Rust streamed payload/trailer read by C + sparse 4 GiB indexing + embedded payload-range seam green; live same-model C/Rust ordinary 4-way save/load green; C-default 6,144-token cold partition exact; scoped continued 6,800-token bodies/four-way exact; DeepSeek 4,096-token intermediate-prefill bodies/four-way exact; DeepSeek tool-map producers have exact C/Rust text+payload bodies and all four reordered-history loader cells restore 424 cached tokens; width-1 bank-shutdown producers have exact 6,896-token text/payload bodies and all four cross-host loader cells restore 6,896 cached tokens | same-policy restore TTFT ABBA green; scoped cold, continued (final/decode and DeepSeek intermediate-prefill), tool-map, and width-1 bank correctness/cross-read green; scoped width-1 bank restore ABBA green; configured default-policy/full memory performance pending |
| web utility | yes | isolated blocking-I/O crate; not production-integrated | C (`ds4-agent`) | encode/wire + mock CDP green | n/a |
| server (four surfaces) | yes | Rust owns HTTP parsing/rendering, admission, metrics, FIFO owner, serial/continuous/static routing, continuation, and disk-KV policy over native execution | Rust host | CPU oracle suite green; Qwen Chat/Responses/Anthropic image requests, barrier width 2 (`served=2 fallback=0`), static width 2, serial fallback, image-aware live/disk KV, 8,192-boundary image prefill, KV four-way, and configured-262K live smoke are green; OpenAI Completions remains text-only by contract | Qwen ABBA + two-hour soak + 262K direct + MTP/static focused gates green; full cross-family serving re-stamp pending |
| distributed runtime | yes | isolated codecs + blocking orchestration crate; not production-integrated | C | codecs + CLI/route/mock hop green | n/a |
| CLI / bench / agent host | yes | partial: greedy/seeded-sampling `ds4-rs` with non-TTY thinking formatting, local non-MTP/non-distributed `ds4-bench-rs`, one-turn no-tool `ds4-agent-rs`, and `ds4-server-rs` | C | greedy, fixed-seed sampled, and fixed-seed thinking/non-thinking one-shot CLI stdout match C byte-for-byte; local benchmark ABBA green. Agent built-in prompt bytes, fixed datetime message, selected non-TTY projector tapes, and DSML refusal are green (`make test-agent-parity`), but live agent generation parity is not established | local benchmark green; full gate pending |
| CPU reference backend | yes | no (not a cut-over blocker) | C | — | n/a |
| Metal backend | native | unchanged | native | — | n/a |
| CUDA / MMQ / VMM | native | unchanged | native | Qwen Q5 loader/PLE/QSA/MoE/GDN/batch + shared MMQ green | Qwen C/Rust text+image ABBA green; complete family/proof re-stamp pending |
| FFI bridge (`ds4_bridge`) | n/a | broad C-host bridge; not the final narrow CUDA ABI | n/a | FFI error path + bounded payload seek/length/EOF/overflow oracle green; legacy sync ABI plus additive same-thread prefill progress callback, cleanup/error/panic/exact-prefix oracles green; opaque bank snapshot/load/save and native continuous decode duration/token/step callback oracles green | — |
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
| 4 | KV store port + 4-way matrix | **format/policy + payload-free index + embedded range seam + replacement-safe staged writer green** (`make test-kv-parity`, focused core/bridge oracles). Ordinary serial Motif-3 no-think/no-tools visible save/evict/restart restore is wired; live C→Rust, Rust→C, Rust→Rust, and C→C are green at the shared `cold=0, continued=0` policy. Restore TTFT ABBA and timing-count/rate scope are green after deferred boot prewarm. C-default cold checkpoints are live cross-read green. The scoped final-sync/decode continued slice is four-way live-green at 6,800 tokens. The scoped DeepSeek ordinary-serial intermediate-prefill slice is four-way live-green from a 4,096-token durable frontier inside a 6,782-token prompt. The scoped DeepSeek OpenAI Chat no-think tool-map slice is four-way live-green with reordered JSON arguments and 424 cached tokens. The scoped Motif-3 width-1 OpenAI Chat no-think/no-tools bank-shutdown slice is four-way live-green with exact C/Rust bodies and a green 114-token restore ABBA; post-fix short-response timing also matches C. Default configured 10,000/effective aligned 10,240 periodic checkpoints, live bank-evict evidence, multi-bank fork/partial ownership, and full default-policy ABBA remain pending |
| 5 | Web utility port | **isolated parity green** (`make test-web-parity`); production `ds4-agent` still uses C `ds4_web.c` |
| 6 | Distributed runtime port | **isolated parity green** (`make test-dist-parity`); production still uses C pipelined prefetch, snapshot, and `ds4_dist_session_*` |
| 7 | Server shadow by feature | **Qwen re-stamp green; broader campaign partial** — CPU contracts plus Qwen serial/continuous/static, true width-2, Chat/Responses/Anthropic, image input, image-aware live/disk KV, MTP, and two-bank fork/partial behavior are green. Remaining cross-family/API and failure-path re-stamp stays open. |
| 8 | `ds4.c` decomposition | **partial** — metadata, mmap catalogs, tokenizer (including family chat-transcript framing), and ledger slices are green; engine/model/session/scheduler execution and CUDA upload remain native |
| 9 | Promote Rust binaries to default names | **names already promoted; v0.6.5 revalidation active**. Qwen is green through `8006198`, including its two-hour soak and 262K/focused gates; the full pre/post cross-family/proof/performance manifest must be re-stamped before this phase is closed again. DeepSeek does not repeat a long soak. |
| split | `SPLIT_READINESS.md` + `ds4-dfm-rs` code genesis | destination is metadata-only; blocked on remaining host work and the complete v0.6.5 cross-family/proof re-stamp |

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

The following Phase 3/7 Motif paragraph is retained as historical
`v0.6.3-dfm` evidence. Phase 3/7 live Motif evidence is in
`scratch/rust-host-live/` (tmux + workspace-local `../scripts/guarded-run.sh`,
outside this repository; sequential C then Rust, `clear_cache` between loads).
Live CUDA census on `ds4-server-rs` is green (`census_supported=1`, epoch 1636,
weight_artifact 86.07 GiB, observation ok). Live DSpark sibling FFI is green
(DeepSeek-V4-Flash-IQ2XXS + DSpark-drafter-Q2K-Q8: C env fallback `Hi`
29.27/21.27 t/s with strict C validate; Rust `--dspark` host-map bind + layout
skip, 9 decode ids; sequential, teardown + `clear_cache`). The Rust server
continuous lane produced byte-equal output in the recorded same-lane ABBA.

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

Rust proof evidence after the rendered-prompt fix is in `scratch/rust-host-live/proof-{smoke,long}-fixed.log`, `proof-oppc-{c-oracle,rust-fixed}.log`, and `proof-oppc-rust-current.log`: smoke 2/2, long 6/6, and same-host OPP-C C/Rust 5/5 all pass. The final Rust proof used one fixed binary SHA (`056dbdf6...`). The GB10 native gate is also green: detached `v0.6.3-dfm@516456f` matched the current C oracle 5/5, and `CUDA_ARCH=sm_121` now selects the separately tracked `sm_121a` golden; the older generic golden is unchanged. Remaining before Phase 9 includes full user-binary parity, the family regression set, multi-client/server surface follow-ups, and pre-split performance/soak evidence. After Phase 9 + `SPLIT_READINESS.md`, follow [DFM_RS_SPLIT_PLAN.md](DFM_RS_SPLIT_PLAN.md).

## Known remaining C dependency

All default entry points are still C. The web/distributed leaf crates are not
wired into production binaries; the KV crate is wired only for the scoped
ordinary serial cold/evict/continued (including DeepSeek intermediate-prefill)/tool-map
and width-1 OpenAI Chat no-think/no-tools bank-shutdown/replay slices above. The Rust server owns part of the HTTP host surface,
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
