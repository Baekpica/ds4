# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model GPU graph inference
  (Metal on macOS, CUDA on Linux).
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short GPU smoke tests for build verification
  (Metal on macOS, CUDA on Linux).

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `ds4_cuda.cu`: CUDA backend. Single TU; mirrors `ds4_metal.m`'s role on
  NVIDIA. CUDA env vars and dispatcher behavior are documented in
  `misc/cuda-env-vars.md`; CUDA MTP specifics in `misc/cuda-mtp/README.md`.
- `cuda/mmq/`: vendored llama.cpp ggml-cuda matmul kernels + ds4-side adapter.
  See `cuda/mmq/VENDOR.md` for the upstream pin and re-sync procedure.
- `tools/ds4_weight_server.cu`: optional CUDA weight-server sidecar for
  multi-process testing. See `misc/proof-harness/README.md`.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material. A few
  reference docs are force-added (`cuda-env-vars.md`, `cuda-mtp/`,
  `proof-harness/`, `ANTHROPIC_LIVE_CONTINUATION.md`, `RESPONSE_API.md`).

## CUDA captured-decode rules

- Captured decode kernels that consume `pos0`, `n_comp`, `n_index_comp`,
  `raw_start` / `raw_row`, `n_raw`, selected-row counts, or scratch pointers
  MUST read live substrate state (`g_decode_dev` / `g_layer_dev[il]`) or be
  keyed by regime into the graph cache. By-value kernel arguments are baked at
  graph queue time and replay stale. Reference: 7c4b84d, a1cff19, 8fb3c54.
- Long-context captured-vs-eager parity (essay prompt, n=1024, FP32, every
  enabled overlay) is a release gate, not a smoke test. See `make proof-cuda-long`.
- Optimization commits land with a correctness proof AND a speed proof. The
  proof harness records both: `tests/ds4_proof.py --scenario ...`. Skipping the
  correctness proof on the grounds that "we already had it before" is how the
  pos0 regression slipped past three previous commits.

## MTP / compressed-KV decode rules

- Lossless = verified tokens AND committed compressed-KV == the accepted-prefix transition. Not bit-identity (cross-width MoE order differs), not tokens alone (carried state sets future logits).
- Emit compressor state in the verify forward, roll back rejects; never re-emit in a second pass (a commit-reforward emit was ~100% wrong).
- Gate on ‖Δcompressed-cache‖ and long context, not short-prompt tokens — short context hides cache corruption behind raw/windowed KV.
- FP vs structural is element-wise magnitude (rel ≈1e-6 vs ≈1); a late token flip is not FP.
- Size GPU-readback buffers to the tensor and fail loud — undersized returns a constant → false "identical".
- At D>=1 the batched verify (width 1+D) is NOT bit-identical to a width-1 decode (cross-width MoE-order FP, same as N=1 vs N=8). At **N=1** the committed cache IS invariant to rejected-draft VALUES (width-matched, no cross-row terms) — validate the rollback there expecting bit-identical cache; comparing D>=1 vs width-1 mode-0 conflates inherent FP with real bugs.
- At **N>1** the committed cache is NOT value-invariant to rejected drafts, and that is NOT a bug: the in-forward verify is itself draft-VALUE-dependent via cross-row MoE-reduction FP — holding cur's input identical, varying a causally-masked rejected draft shifts the verify logits by FP while the argmax (committed token) is unchanged, and that FP rides an expert flip into the committed KV (deep-layer ‖Δ‖ ≈ O(10) while the final logit moves ≈0.03 and the token does not flip). So N>1 losslessness is TOKEN-LEVEL: gate the committed-row FRONTIER (counts) + token stream invariance to rejected-draft values; treat committed-cache VALUES as inherent-FP-noisy (informational), and add a byte-exact restore self-check (read back each restored lane vs its checkpoint source) to guard the capture/restore plumbing independently. Don't chase the N>1 cache-value diff as a rollback bug — bisect: the divergence onset is a cur-row (M=1) logit FP in the forward, not the restore (which is byte-exact).

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and a GPU backend are available. Use live server tests only when
intentionally testing the API surface.

Multi-process testing (proof harness, multi-profile sweeps, MTP correctness
work that loads base + MTP gguf into the same device) goes through
`ds4_weight_server`. See `misc/proof-harness/README.md`. Single-process
runs hit the same prefill ceiling without a sidecar via the in-process
VMM arena, which is on by default.
