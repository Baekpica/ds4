# CUDA backend env-var reference

The CUDA backend (`ds4_cuda.cu`) mirrors `ds4_metal.m`'s role on NVIDIA. It
dispatches Q8_0 dense matmuls through one of three kernel families, has its
own n_tok=1 mmvq decode path, optionally captures the decode-block kernel
sequence into a `cudaGraphExec_t`, and can either allocate weight memory
in-process (default) or import it from the `ds4_weight_server` sidecar.

Every CUDA-specific env var is below, with the intent behind each default.

## Q8_0 dispatcher

cuBLAS is initialised unconditionally at backend startup regardless of the
selected strategy: on sm_121 we observed that this triggers CUDA driver state
making mmq ~4&times; faster than a binary that skips `cublasCreate`, so the
cublas path stays resident even when not selected.

| Strategy | When picked                                       | What it runs                                       |
|----------|---------------------------------------------------|----------------------------------------------------|
| `mmq`    | default; every CUDA arch we've validated          | vendored llama.cpp `mul_mat_q` (`cuda/mmq/`)       |
| `cublas` | explicit override or fallback if mmq init fails   | `cuda_q8_f16_ptr` Q8&rarr;FP16 cache + `cublasGemmEx` |
| `warp8`  | explicit override or last-resort fallback         | native `matmul_q8_0_preq_*_kernel` family          |

Logged once on first dispatch, e.g.:

    ds4: CUDA Q8_0 dispatch: mmq (sm_120, 1792 GB/s memory bandwidth) [default]

The bandwidth figure is informational; we don't tier on it.

## Env-var inventory

- `DS4_MODEL_ANON_HUGE=N` (Linux, default off; lives in ds4.c model_open, not
  the CUDA backend). Copy GPU-backend model files out of the file-backed mmap
  into anonymous `MADV_HUGEPAGE` memory at load. `N<=1` copies every GPU
  model; `N>1` copies only files of at least `N` GiB (e.g. `32` = base model
  only). WHY: the mmap'd GGUF leaves weights on 4 KB page-cache folios whose
  physical contiguity depends on how they entered the cache; on GB10 the
  GPU-side cost is decided at `cudaHostRegister` pin time — fragmented folios
  cost ~5x on routed-MoE spans, and even sequentially-rewarmed cache pays
  2-3.6x vs anon-THP (measured 2026-07-02: base decode 86-96 &rarr; 68.5
  ms/tok, w1 MoE span 0.276 ms = isolated-kernel floor). CAUTION: anon memory
  is unevictable — only enable when total copied bytes leave ~10 GB of true
  free RAM after banks/runtime, else the kernel splits the huge pages under
  reclaim pressure and performance lands BELOW plain mmap (measured: the
  95 GB base+MTP+drafter set on a 121 GB GB10). The copy streams with
  `posix_fadvise(DONTNEED)` so it does not compete with itself for RAM; boot
  cost is one sequential read of the file (~25 s for 81 GB on GB10 NVMe).
  A pressure guard skips the copy (keeping the file mapping, with a log line)
  when it would leave less than `DS4_MODEL_ANON_HUGE_MARGIN_GB` (default 20)
  of MemAvailable — the margin must also cover KV banks and runtime, and
  ctx-scaled bank growth at 32k+ can still exceed it.

- `DS4_MODEL_ANON_HUGE_MARGIN_GB=N` (default 20). Minimum GiB of MemAvailable
  that must remain after an anon huge-page model copy; below it the copy is
  skipped. See `DS4_MODEL_ANON_HUGE`.

- Budgeting note (no env var; behavior change 2026-07-02): on integrated GPUs
  the KV-bank budget (`ds4_gpu_mem_info`) now subtracts GPU-pinned
  *file-backed* weight registrations from MemAvailable. The kernel keeps
  counting pinned page-cache pages as reclaimable, so the old budget spent
  the model's own residency on banks — the measured NVMe-thrash mechanism at
  32 banks. Default bank counts on GB10 drop accordingly (the honest number);
  `DS4_SERVER_COALESCE_MAX` still caps explicitly.

- `DS4_CUDA_PREFILL_PATH=mmq|cublas|warp8|auto` (default `auto` &rarr; mmq).
  Explicit override. `auto` and unset both resolve to mmq.

- `DS4_CUDA_USE_MMQ=0` (legacy alias). Equivalent to
  `DS4_CUDA_PREFILL_PATH=cublas`. The newer variable takes precedence.

- `DS4_CUDA_MMQ_MOE_MIN_TOKENS=N` (default 2). Minimum `n_tokens` at which
  the routed-MoE mmq path activates. At n=1 mmq's matrix-matrix-shaped path
  has higher per-launch cost than the vector path; that case is handled by
  the mmvq decode branch.

- `DS4_CUDA_MOE_SMALL16=N` (default 16; set 0 to disable). Direct routed-MoE
  vector tier for MTP/DSpark verifier widths 9..16 with top-k=6. It bypasses
  the MMQ tile path for these small batches and reuses the CUDA MMVQ/Q8_1
  vector machinery with internal column chunking above width 8.
  `N` may be 9..16 to cap coverage; `DS4_CUDA_MOE_NO_SMALL16=1` is an alias
  for disable. The tier applies to verifier forwards and the small Q4_K MTP
  support model by default, but leaves the larger Q4_K DSpark drafter on its
  existing MMQ numerics. Set `DS4_CUDA_MOE_NO_SMALL16_MTP_DRAFT=1` to restrict
  it to verifier forwards only, or `DS4_CUDA_MOE_SMALL16_ALL=1` for diagnostic
  all-model A/B runs. `DS4_CUDA_MOE_SMALL16_DIRECT=1` selects the experimental
  Q8_K direct fallback instead of preserving the normal MMQ backup path.
  `DS4_CUDA_MOE_Q81_FUSED=1` selects experimental canonical-Q8_1 fused
  gate+up+mid and down+sum helpers; early GB10 probes preserved acceptance but
  regressed throughput at wider routed batches, so this is diagnostic-only.
  `DS4_CUDA_MOE_Q8K_GATE_IN_VEC=1` and `DS4_CUDA_MOE_Q8K_DOWN_IN_VEC=1`
  are stage-isolation diagnostics: they keep the MMVQ branch active but swap
  only gate/up or only down to the Q8_K direct kernels.
  `DS4_CUDA_MOE_PAIR_RAW_VEC=1` selects an exact gate/up MMVQ diagnostic that
  quantizes X to Q8_1 once, runs the trusted MMVQ gate and up matvecs from that
  shared buffer, and leaves the existing clamp-aware SwiGLU kernel in place.
  `DS4_CUDA_MOE_PROFILE=1` prints per-call MoE stage timings for the fallback
  direct path and, in eager mode, for this MMVQ small16 path. For MMVQ timing
  runs, set `DS4_CUDA_LAYER_GRAPHS=0 DS4_CUDA_MOE_GRAPHS=0` so event recording
  is not inside a captured CUDA graph.

- `DS4_CUDA_MMQ_X_MAX=N`. Clip `get_mmq_x_max_host` to N (rounded down to a
  multiple of 8) when sweeping tile widths. Diagnostic only; the vanilla
  128 wins on sm_120.

- `DS4_CUDA_NO_MMVQ_DECODE`. Opt-out of the vendored `mul_mat_vec_q` decode
  path. mmvq is structurally optimal for n_tok=1 routed-MoE and dense
  attention projection (one block per output row, no column-tile waste).
  Wires into `routed_moe_launch` and `cuda_matmul_q8_0_tensor_labeled`.

- `DS4_CUDA_MMVQ_DECODE_MAX_TOKENS=N` (default 8). Cap on n_tokens routed
  through the mmvq decode branch in `routed_moe_launch`. Range 0&ndash;8;
  0 disables. Values 2&ndash;8 extend mmvq coverage to short-prefill
  batches, subject to the `DS4_CUDA_MOE_VEC_MAX_ASSIGN` assignment envelope.

- `DS4_CUDA_MOE_GRAPHS=0` (default on). Opt-out of CUDA Graph
  capture+replay around the mmvq routed-MoE decode block and the n_tok=1
  dense Q8_0 vec path. Each captured launch is bracketed by
  `cudaEventRecord` / `cudaStreamWaitEvent` so g_moe_stream and stream=0
  stay correctly ordered across the boundary.

- `DS4_CUDA_LAYER_GRAPHS=0` (default on). Opt-out of per-layer
  decode-body CUDA Graph capture+replay. On by default since the Step 7
  determinism + perf gates passed: each transformer layer's decode body
  is captured into its own `cudaGraphExec_t`, keyed on layer index /
  token-flags / double-buffer parity, and replayed on subsequent
  matching tokens. Per-token state rides device-resident scalar
  substrates so it never enters the graph key. Verified bit-identical
  to eager decode through n=256 on sm_120 (PRO 6000) and sm_121 (GB10);
  decode-only, prefill is untouched. Set to 0 (also `off`/`no`/`false`)
  to fall back to the eager per-layer decode path. Also forced off when
  `DS4_CUDA_NO_MMVQ_DECODE` is set (the legacy non-MMVQ decode path is not
  capture-safe).

- `DS4_CUDA_LAYER_GRAPHS_HASH_DUMP=1` (default off). Arms the
  captured-decode per-kernel hash-dump diagnostic. When set, the
  `ds4_cuda_dump_hash_*` entry points FNV-1a a probed device buffer into a
  slot table and print one `DS4_HASH pos=N slot=I hexhash label` line per
  used slot at each token flush; when unset every entry point is a no-op,
  so a normal build is unaffected. Used to localize a
  captured-graph-vs-eager output divergence: probe the same prompt with
  and without `DS4_CUDA_LAYER_GRAPHS=0` and diff the `DS4_HASH` lines — the
  first `(pos,slot)` that differs is the divergent kernel. The probe call
  sites are added temporarily by the investigator (see the comment block
  above the implementation in `ds4_cuda.cu`); only the substrate is
  permanent. See also `tests/cuda_layer_graph_determinism_probe.sh`.

- `DS4_CUDA_MTP_VERIFIER_USE_MMQ` (default 0). Bisection switch. Normally
  `ds4.c` brackets every MTP verifier call with
  `ds4_gpu_set_mtp_verifier(1/0)` and the CUDA backend routes Q8_0
  matmuls onto `warp8` for the duration. mmq's stream-k + MMA FP32
  reduction order drifts ~1 ULP/layer from warp8; the drafter is trained
  against legacy decoding so an mmq verifier flips tight-margin tokens
  (0/314 acceptance on GB10 with mmq verifier active). Set to 1 to
  reproduce the broken behavior for bisection.

## DSpark / DFlash diagnostics

- `DS4_DSPARK_PROFILE=1` (default off). Print aggregate DSpark continuous
  decode timings split into verifier forward, accept loop, inject pack,
  inject projection, inject store, rollback, deferred commit, block draft,
  and Markov refine. Use with `DS4_CONT_PROFILE=1` when comparing against
  whole-engine forward/sample buckets.

- `DS4_DSPARK_COMPACT_INJECT=1` (default off). Diagnostic path that injects
  only accepted verifier rows into the DSpark KV rings instead of every
  verifier row. It packs captured target hidden rows with a CUDA gather
  kernel before the existing DSpark projection/inject stages. Intended to
  quantify whether rejected-row injection is material.

- `DS4_DSPARK_VERIFY_DEPTH=N` (range `1..4`). Diagnostic speed/acceptance
  dial for DSpark block verify width. Unset uses the production policy:
  verify all four drafts at `n_live<=2`, verify three drafts at `n_live==3`,
  and let the existing `DS4_DSPARK_MAX_NLIVE` gate disable DSpark above that.
  When set, the value is exact and disables the `n_live==3` auto-depth rule.
  The block drafter still generates four candidate drafts; this dial controls
  how many are consumed by the verifier on the next step.

- `DS4_DSPARK_ADAPT_DEPTH=1` (default off). Diagnostic per-bank verify-depth
  controller. Each live bank shrinks its next verifier width after a miss and
  grows it after accepting the full currently verified prefix. Correctness is
  unchanged because accepted tokens still come only from the target verifier;
  this only trades verifier rows against draft yield.

## In-process VMM weight arena

The arena allocates each weight tensor in its own CUDA Driver VMM
region (`cuMemCreate` &rarr; `cuMemAddressReserve` &rarr; `cuMemMap`
&rarr; `cuMemSetAccess`), giving every tensor its own
2&nbsp;MiB-aligned virtual address.  This matches what the
out-of-process `ds4_weight_server` provides imported workers.  On
discrete GPUs this is worth roughly 2&times; prefill; on integrated
GPUs it's neutral-to-positive.

### Why per-tensor chunks specifically

The chunk-size bisect we ran during development clarified the
mechanism.  VMM with one large chunk (e.g.
`DS4_CUDA_VMM_ARENA_CHUNK_MB=1792`) performs identically to the
cudaMalloc-backed arena (~1080 t/s prefill on PRO 6000), even though
the underlying memory is still 2&nbsp;MiB-paged.  The actual
differentiator is **per-tensor 2&nbsp;MiB-aligned base addresses**:
when each weight tensor sits at its own fresh
`cuMemAddressReserve`-handed VA, matmul kernels' tile-load coalescing
and L2 spatial-locality patterns improve enough to roughly double
prefill.  Pack the same VMM-paged memory into one big chunk and the
bases land at sub-granularity offsets &mdash; the perf advantage
disappears.

This also unifies cleanly with the drift below: same root cause, two
effects you cannot separate.

### Known trade-off: FP32 reduction-order drift vs official vectors

Per-tensor VMM-allocated weight ranges produce a small but real
**reduction-order drift** in the matmul kernels relative to the
cudaMalloc-backed arena.  The same cache/tile-arrival-order behavior
that gives the 2&times; perf win also changes the order in which tile
partial sums reach the FP32 accumulator; FP32 is non-associative, so
the order matters.  This is structural to the kernels' parallel
reduction strategy, not a misuse of the API.

Investigation established:

1. The uploaded weight bytes are byte-identical between the two
   allocators (verified by post-upload checksum of all 138 weight
   ranges).
2. Kernels do not read past tensor bounds (verified by poisoning the
   chunk tail with 0xAB instead of zero &mdash; output unchanged).
3. The drift is shared by both the vendored mmq family and the legacy
   `warp8` native kernels and is therefore upstream of the Q8_0
   dispatcher.  Same drift on PRO 6000 sm_120 and GB10 sm_121.
4. Logit-level magnitude is small (~0.08 logprob units at step 0)
   &mdash; bounded, deterministic, of the same shape as the documented
   mmq-vs-warp8 ULP-per-layer drift behind `DS4_CUDA_MTP_VERIFIER_USE_MMQ`
   (Option D).  Most tokens are unaffected; only tight-margin choices
   flip.

**Observable cost:** in `./ds4_test --logprob-vectors`, one of four
test vectors (`short_code_completion`, step 1: the `c` language tag
after triple-backticks) flips to a textually-equivalent but
byte-different alternative under the VMM-arena default.  The other
seven failures in that test family are pre-existing on the CUDA
backend and reproduce identically with `DS4_CUDA_VMM_ARENA=0`.

**Workaround for users who need official-vector byte equivalence:**
set `DS4_CUDA_VMM_ARENA=0` to use the cudaMalloc-backed arena.  Prefill
ceiling drops by ~50% on discrete GPUs in exchange for the parity.

### Env vars

- `DS4_CUDA_VMM_ARENA=0`. Disable; fall back to the cudaMalloc-backed
  arena.  Also the workaround for the reduction-order drift above.

- `DS4_CUDA_VMM_ARENA_CHUNK_MB=N`. Minimum chunk size per `cuMemCreate`.
  Default 0 (chunk = request size, rounded up to the driver-reported
  granularity; matches the weight server's per-range allocation).
  Values 1024+ collapse the per-tensor placement and forfeit the perf
  benefit; useful only for bisection.

- `DS4_CUDA_WEIGHT_IPC_MANIFEST=/path/to/manifest.json`. Worker-side
  import path for weights owned by `ds4_weight_server`. When set, the
  in-process VMM arena is hard-gated off because the sidecar already
  provides identical VMM ranges and running both would double-allocate
  the model. See `misc/proof-harness/README.md` for the sidecar
  lifecycle.
