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
| session lifecycle | yes | no | C | — | — |
| KV store | yes | no | C | — | — |
| web utility | yes | no | C | — | — |
| server (four surfaces) | yes | no | C | — | — |
| distributed runtime | yes | no | C | — | — |
| CLI / bench / agent host | yes | no | C | — | — |
| CPU reference backend | yes | no (not a cut-over blocker) | C | — | n/a |
| Metal backend | native | unchanged | native | — | n/a |
| CUDA / MMQ / VMM | native | unchanged | native | green (C baseline) | green (published band) |
| FFI bridge (`ds4_bridge`) | n/a | no | n/a | — | — |
| proof harness on Rust path | C binaries | no | C | — | — |

## Phase checklist

| Phase | Name | State |
|---|---|---|
| 0 | Freeze baseline + this document set | **done** (docs-only commit) |
| 1 | Cargo workspace + FFI skeleton | not started |
| 2 | `ds4-core` safe wrappers | not started |
| 3 | Shadow `ds4-rs` / `ds4-bench-rs` | not started |
| 4 | KV store port + 4-way matrix | not started |
| 5 | Web utility port | not started |
| 6 | Distributed runtime port | not started |
| 7 | Server shadow by feature | not started |
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
| `ds4_weight_server` | native CUDA (unchanged) |

No `*-rs` shadows exist yet.

## Known remaining C dependency

**Everything in the production host.** The only decided native
forever-piece is the CUDA/MMQ/VMM backend (and Metal as a
non-blocking compile).

## Notes

- `dfm` is not merged into `rust-host`. Cherry-pick CUDA
  correctness or performance fixes only, then rerun this matrix.
- `SPLIT_READINESS.md` must not be added until the §26 required
  list is evidence-green.
- Unsafe-audit command once crates exist:
  `rg -n 'unsafe \{' crates/`
