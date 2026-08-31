# Qwen v0.6.5 Rust-host re-stamp (2026-08-31)

This is the Qwen pre-migration gate for the resumed `rust-host` campaign.
It replaces the Qwen portion of the old `v0.6.3-dfm` evidence; it does not
replace historical evidence for the other model families.

## Frozen identities

| Item | Identity |
|---|---|
| Release baseline | `v0.6.5-dfm` → `d02e2a4777a34a9f52fd987453b3ea1801fac52e` |
| Qwen C golden cut | `4d40d97f1e575400237a6e5cef21d7f74404a38d` |
| Merge into `rust-host` | `d0384d91a201a409118d811772a129985b44c6bf` |
| Rust candidate | `6ca85c8534d3decd5b3a16b927a3257a074e8cc8` |
| C server SHA-256 | `b13a093b9232f61185a4e47fdbb5b3ca3babed0cb0a2d8f8eeae074da2b1aa2a` |
| Rust server SHA-256 | `2bdbbc044216705bb1077feba279b48efa9028cba42207cd03ef4e7d1416635f` |

The annotated release tag predates the final image and two-bank Qwen
increments. Therefore the observable Qwen oracle is the product-code cut
`4d40d97`, which descends from `v0.6.5-dfm`. The C checkout used for the
measurements was at `cad9ef5`; `git diff 4d40d97..cad9ef5` contains only
`.github/FUNDING.yml`, so its built product sources are identical to the
frozen Qwen cut.

## Artifact scope

Only the primary mixed-quant deployment artifact was admitted:

- `MQ-Q5-SSD-PLE-BF16`, three GGUF shards totaling 83,274,984,384 bytes;
- its `ple` symlink to the shared four-file BF16 SSD-PLE sidecar, each file
  25,600,196,608 bytes;
- embedded MTP with `--mtp-draft 2`;
- text and still-image serving at configured context 196,608.

Safetensors reference weights, a resident BF16 GGUF, the Q6 main GGUF, and
`test-qwen4exp-ple-reference` / `test-qwen4exp-ple-forward` were intentionally
not loaded. This is the requested Q5+Sidecar gate, not a claim about those
variants.

Image decoding reuses the pinned `vendor/stb_image.h` through the narrow
native image ABI. Rust owns schema normalization, payload lifetime, token
placement, and decoded-pixel cache identity; the vision/CUDA implementation
remains native. No new Rust image framework was added.

## Automated and family gates

- `cargo test --workspace -- --test-threads=1`: pass. A parallel run exposed
  one pre-existing distributed test interaction; that test passed alone and
  the full serialized workspace passed.
- `cargo fmt --all -- --check`, `git diff --check`: pass.
- `tests/parity/bridge_null_oracle`: pass.
- Q5/Sidecar family set: loader, tokenizer, PLE store, PLE CUDA direct I/O,
  primitives, HC forward, PLE compute, QSA, real-Q5 QSA forward, MoE,
  real-Q5 MoE forward, GDN, real-Q5 GDN forward, shared family kernels, and
  MMQ parity all pass. Log SHA-256:
  `6927fc95b5f3d7eb6ca48b9e73b08a9c36643e05eae89e2163ce0ef1fd1df28d`.
- `test-qwen4exp-batch`: two-bank parity, embedded-MTP target verification,
  disk-KV, partial fork, and graph retire/rebuild pass. Log SHA-256:
  `0554e3c0a4df0d58f1b4773b332c3abcf6007ad4df6988d521a31c82afe21e35`.
- Sidecar CUDA test peak RSS was 187,576 KiB while the sidecar is 95.368 GiB;
  the full sidecar did not become resident.

## Rust live behavior gate

The Rust server used one Q5 owner/worker pair under the 109/115 GiB
reclaim/hard guard and then shut down before the ABBA run.

- `/v1/models` and textual `/v1/stats` reported the 196,608 context and zero
  request/census/governor faults.
- Text Chat returned the requested exact marker with embedded MTP active.
- The same JPEG returned `MEN WALK ON MOON` through Chat Completions,
  Responses, and Anthropic Messages.
- A same-image continuation reused 332 tokens; a same-geometry different-pixel
  PNG reused zero.
- Cross-worker disk restore reused 2,131 of 2,154 prompt tokens.
- An 8,377-token image request crossed the 8,192 prefill boundary and returned
  the same headline.
- Barrier-synchronized Responses and Anthropic image requests logged
  `served=2 fallback=0`.
- `DS4_SERVER_CONTINUOUS=0` produced a two-request static batch and a separate
  serial request with zero sheds or faults.

## C → Rust → Rust → C ABBA

Every cell used a fresh owner and worker, the same model, context 196,608,
two Qwen banks, 8,192-token prefill chunks, MTP draft 2, temperature 0, and
the same request bytes. Each worker and owner exited before
`/usr/local/bin/clear_cache`; no C and Rust model were resident together.

Text request SHA-256 is
`a1165c1758609750dbe8da8aa8ce9542560742a9d88fecee0d090946df8b9ffc`.
All cells returned the same 64-token text, SHA-256
`3ab21f93950479e839c5aa43e0d4799f0d8e03bc504a798e5a8c4042ba9152b4`.

| Cell | Prefill tok/s | Decode tok/s | TTFT ms | Worker VmHWM KiB | Worker GPU MiB |
|---|---:|---:|---:|---:|---:|
| C1 | 519.7 | 14.4 | 7,954.7 | 1,852,404 | 23,471 |
| Rust1 | 518.6 | 14.5 | 8,044.9 | 1,933,444 | 23,471 |
| Rust2 | 518.8 | 14.4 | 8,041.3 | 1,929,004 | 23,471 |
| C2 | 518.0 | 14.5 | 7,979.9 | 1,847,240 | 23,471 |

Rust/C mean ratios are 99.97% prefill, 100.00% decode, 100.95% TTFT, and
104.40% worker VmHWM. They pass the 97% / 98% / +5% / +5% thresholds. GPU
residency is identical.

Image request SHA-256 is
`56a3614fd11a428339f83e202ffc792d9a160e2765dd7a4a319ae125a0aa9f86`.
All cells returned `MEN WALK ON MOON`, SHA-256
`991530ac66fb7421db0beb25874cff2c849d4077c4656ca12c7251339e20184d`.

| Cell | Image prefill tok/s | Image TTFT ms |
|---|---:|---:|
| C1 | 399.7 | 2,576.7 |
| Rust1 | 402.1 | 2,584.1 |
| Rust2 | 398.6 | 2,595.4 |
| C2 | 399.4 | 2,586.3 |

Rust/C means are 100.20% image prefill and +0.32% image TTFT. The six-token
image decode rate is retained in the raw responses but is too short to use as
a performance gate.

Evidence is under
`scratch/rust-host-live/qwen-v065-restamp-20260831/`. Final teardown left no
GPU compute process and restored 119 GiB `MemAvailable` after `clear_cache`.

## Decision

The Qwen Q5+Sidecar pre-migration gate is green at `6ca85c8`. This allows the
host migration to continue. It does not make `SPLIT_READINESS.md` green: the
remaining v0.6.5 family/proof/soak manifest and unresolved host ownership work
still gate the `ds4-dfm-rs` split.
