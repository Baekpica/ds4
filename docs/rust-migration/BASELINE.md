# C golden baseline

`v0.6.3-dfm` is not a convenience tag. It is the immutable correctness
oracle for every Rust host change on `rust-host`.

Do not treat the C tree as “legacy reference only” while porting.
If Rust and C disagree, Rust is wrong until a documented C bug is
fixed in C first, identified as its own commit, and then ported.

## Frozen identity

```text
C baseline tag:     v0.6.3-dfm
C baseline commit:  516456fe35510e4fb8350396c9d88807ac1f760b
VERSION file:       v0.6.3-dfm
git describe:       v0.6.3-dfm
Annotated tag:      c2a6802facb89f2a3366aa534fd6637a912118a8
                    (peels to 516456f)
Parent merge:       e633543 Merge Entrpi v0.6.3 into ds4-dfm as v0.6.3-dfm
Upstream absorbed:  Entrpi v0.6.3 (d92d93a)
```

`rust-host` was created with:

```bash
git checkout v0.6.3-dfm
git checkout -b rust-host
```

The `dfm` branch must not be merged into `rust-host`. Semantics are
frozen at this SHA.

## Host recorded at branch creation

Recorded 2026-08-24 on the reference DGX Spark. This is the
environment identity, not a new throughput run.

```text
CUDA:          13.3 (nvcc V13.3.73, cuda_13.3.r13.3/compiler.38244171_0)
Driver:        610.43.02
Target GPU:    NVIDIA GB10 / sm_121a
Compute cap:   12.1
Host memory:   121 GiB unified
Kernel:        Linux 6.17.0-1031-nvidia
Build:         make cuda-spark
               (CUDA_ARCH=sm_121, gencode compute_121a/sm_121a)
Rust toolchain (host, not yet in tree): rustc 1.98.0 / cargo 1.98.0
```

`make cuda-spark` remains the only production build. Cubins must be
`sm_121a`. `-arch=sm_121a` alone is the documented silent-wrong-arch
failure.

## What `v0.6.3-dfm` itself proved

From `docs/ds4-dfm-model-families.md` § Integration evidence for
`v0.6.3-dfm` (commit `516456f`):

- Absorbs typed schema-format refusal, chunked request bodies,
  think-dial counters / completion line, full-1M decode dispatch
  (DeepSeek MLA), whole-prompt depth fence, and best-fit trim victims.
- Family-side fix: Solar / EXAONE / Motif-3 continuous banks accept
  an exactly-full payload (`18dbd1e`).
- Gates on this binary: extractor self-test, `ds4_test --server`
  (including v0.6.3 refusal/fence/think units), split-GGUF,
  `test-model-family-kernels`, `test-mmq-parity`, `cuda-regression`,
  Motif loader/tokenizer/reference/CUDA, EXAONE kernels/reference,
  Solar loader/tokenizer/KDA/prefill/chunk/gates/KV + forward,
  dots3 loader/tokenizer.
- Live Motif MQ87-88 VMM owner + worker (`-c 2048`, 32 banks): four
  API surfaces, continuous route, 4/0. Typed `response_format`
  refusal is HTTP 400 in the native envelope.

`v0.6.3-dfm` did **not** remeasure Motif or Solar throughput. Earlier
tags are not moved. The published tables below remain the Spark
evidence band. Before the Phase 3 FFI performance gate, remeasure the
representative cells on this SHA (or a documented cherry-pick) with
AB/BA or ABBA. Do not treat a later `dfm` tip as the C number.

## Representative published Spark cells

Authoritative write-up: `docs/ds4-dfm-model-families.md`.
All cells are GB10, driver 610.43.02, CUDA 13.3, `sm_121a`.

### Motif-3 (production MQ87-88, aligned-Q8 VMM owner)

Published `v0.5.6.3-dfm` table (`593d251` HTTP, `cc2f277` 8K bench):

| model | context | prefill tok/s | decode tok/s | TTFT | GPU memory | host RSS |
|---|---:|---:|---:|---|---|---|
| Motif-3 MQ87-88 | 8,192 (`ds4-bench`) | 519.55 | 12.28 (64 tok) | not claimed (non-stream bench) | aligned-Q8 owner path | source-GGUF map RSS 29,632 KiB after 256K cell (that cell) |
| Motif-3 MQ87-88 | 32,768 Chat | 396.47 | 8.96 (43 tok) | no independent TTFM | — | — |
| Motif-3 MQ87-88 | 262,080 Chat | 175.61 | 2.52 (43 tok) | no independent TTFM | worker 9.703 GiB; latent KV 4.119 GiB | owner+worker `VmSwap: 0` |

Later remesure on the `v0.6.2-dfm` line (engine tip `2c81427`, kernels
through `a09ff4f`; **not** a moved tag):

| model | context | prefill tok/s | decode tok/s | TTFT | GPU memory | host RSS |
|---|---:|---:|---:|---|---|---|
| Motif-3 MQ87-88 | 8,192 (`ds4-bench`) | 627.19 | 15.06 (64 tok) | not claimed | aligned-Q8 owner | — |
| Motif-3 MQ87-88 | 32,768 Chat | 546.7 | 12.8 | no independent TTFM | — | — |
| Motif-3 MQ87-88 | 262,080 Chat | 238.59 | 5.97 (43 tok) | no independent TTFM | worker 10,429 MiB; latent KV 4.119 GiB | `VmSwap: 0`; ~11–12 GiB available |

196K multi-bank Motif cell (`03b7002` / `cf605e0`): 8,214-token cold
prefill 266.3 tok/s; 192-token decode 490.4 ms TTFT, 12.9 tok/s;
owner 90,119 MiB, worker 22,283 MiB after the 8K requests.

`--no-repack-q8-aligned` is not the published Motif point.

### Solar Open2 (MXQ-v1, `b2e52b9`)

| model | context | prefill tok/s | decode tok/s | TTFT | GPU memory | host RSS |
|---|---:|---:|---:|---|---|---|
| Solar-Open2-250B MXQ-v1 | 8,222 Chat | 1,050.7 | 19.05 p50 / 18.9 API (128 tok) | not separately published | 3 banks at `-c 196608` | — |
| Solar-Open2-250B MXQ-v1 | 66,761 Chat | 804.5 | 13.07 p50 / 14.1 API | not separately published | same | — |

### Unvalidated on this host

- dots3-note 524,288-token prefill + decode: **unvalidated**
- Motif concurrent 256K banks: **not claimed**
- Solar 1,048,576-token serving: **not claimed**
- DeepSeek full-model GPU tests: no DeepSeek GGUF on this host

## Performance gate numbers (provisional)

Rust may ship a host change only when a same-host AB/BA or ABBA
remeasure against the C binary from this baseline (or an identified
cherry-pick) stays inside:

| Metric | Rust allowance |
|---|---:|
| Prefill | ≥ 97% of C |
| Decode | ≥ 98% of C |
| TTFT | ≤ +5% vs C |
| Host RSS | ≤ +5% vs C |
| GPU resident | no meaningful increase |
| Token correctness | existing contract |

A 2–3% regression with no explained cause (extra memcpy, allocation,
lock scope, FFI granularity, scheduler change) is a fail.

## Proof harness that must remain reachable

These Makefile targets are release gates, not smokes. The Rust
execution path must become able to run the same contracts:

```text
make proof-cuda-smoke
make proof-cuda-long
make proof-cuda-opp-c
```

`proof-cuda-long` is captured-vs-eager parity (essay prompt, n=1024,
FP32, every enabled overlay). Migration does not relax it.

## Artifacts this baseline serves

Paths are host-local; identities are from the family doc.

| Family | Serving input |
|---|---|
| Motif-3 | `Motif-3-MQ87-88-FIT.gguf` (94,162,541,472 bytes canonical) |
| Solar Open2 | `Solar-Open2-250B-MXQ-v1` 11 shards (95,533,532,160 bytes) |
| K-EXAONE | 3-shard MXQ (85.56 GiB) |
| dots3-note | `dots3-note-prev-MQ87` 10 shards (80.156 GiB) |

## Re-measure checklist (before Phase 3 performance sign-off)

Record SHA, GGUF revision, exact command/env, token counts, cold/warm,
cached tokens, TTFM/TTFT, prefill/decode wall and tok/s, memory peak,
swap, clocks, and the correctness fixture result. Prefer:

```text
C → Rust → Rust → C
```

on the same Motif 8K `ds4-bench` cell and the same Solar 8K Chat cell
used above. Do not update this file’s published historical rows to
hide a regression.
