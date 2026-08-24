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
| model metadata | yes | no | C | — | n/a |
| GGUF loading (mmap) | yes | no | C | — | — |
| tokenizer | yes | no | C | — | — |
| session lifecycle | yes | wrapper | C | — | — |
| KV store | yes | yes (envelope + policy) | C | 4-way green | n/a |
| web utility | yes | yes (blocking I/O) | C (`ds4-agent`) | encode/wire + mock CDP green | n/a |
| server (four surfaces) | yes | yes (`route_decide`) | C | reason table + needs green | n/a |
| distributed runtime | yes | yes (codecs + blocking orchestration) | C | codecs + CLI/route/mock hop green | n/a |
| CLI / bench / agent host | yes | shadow `ds4-rs` / `ds4-bench-rs` | C | — | — |
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
| 3 | Shadow `ds4-rs` / `ds4-bench-rs` | **linked** (`make ds4-rs`); same-model token/perf gate pending |
| 4 | KV store port + 4-way matrix | **format/policy green** (`make test-kv-parity`); live session payload still C |
| 5 | Web utility port | **done** (`make test-web-parity`); `ds4-agent` still links C `ds4_web.c` |
| 6 | Distributed runtime port | **blocking runtime green** (`make test-dist-parity`); C still owns pipelined prefetch, snapshot, `ds4_dist_session_*` |
| 7 | Server shadow by feature | **route_decide / compute_needs green** (`make test-route-parity`); HTTP/wire still C |
| 8 | `ds4.c` decomposition | not started |
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
| `ds4-rs` / `ds4-bench-rs` | Rust host, same C CUDA core (`make ds4-rs`) |
| `ds4_weight_server` | native CUDA (unchanged) |

Phase 3 live parity (same GGUF, token/KV/prefill/decode/memory) has not been run. Do that in tmux + `scripts/guarded-run.sh`; do not load a production GGUF from an interactive agent session.

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
  Current hits are only `crates/ds4-core` FFI adapters. `ds4-sys` is
  `extern "C"` declarations. `ds4-kv`, `ds4-web`, `ds4-dist`, and
  `ds4-server` have none.
