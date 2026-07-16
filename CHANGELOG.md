# Changelog

All notable fork-side changes to this project are documented here.
Fork: [Entrpi/ds4](https://github.com/Entrpi/ds4) of
[antirez/ds4](https://github.com/antirez/ds4); upstream fork point `e16ead1`
(2026-05-29). Upstream's own changes are not repeated here.

## Unreleased

- **Launch defaults: `ds4-server -c N` boots the full stack on a standard
  install.** With `-m` omitted the server resolves the base model from
  `$DS4_GGUF_DIR` (default `~/gguf`); when the MTP head and/or DSpark
  drafter GGUFs sit beside the base model and `--mtp`/`--dspark` are not
  given, they are attached and MTP-2 + DSpark speculative decode is
  enabled. Every auto choice is reported on one boot line — never silent.
  Explicit flags and env (`DS4_CONT_MTP_MODE`, `DS4_CONT_DSPARK`,
  `DS4_DSPARK_MODEL`) always win; file names follow the `ds4-serve`
  wrapper's env (`GGUF_FILE`/`MTP_FILE`/`DSPARK_FILE`). New flags:
  `--dspark FILE` (CLI form of `DS4_DSPARK_MODEL`), `--preset spark`
  (require the full stack, fail loudly if any piece is missing),
  `--no-mtp` / `--no-dspark` / `--no-spec` (per-component opt-outs).
  `DS4_CONT_DSPARK=0` (or empty) now reads as OFF — it was
  presence-tested before, so `=0` counter-intuitively armed the drafter.
  Gate scripts pass `--no-mtp` on their `MTP=""` legs so MTP-droppable
  legs stay genuinely MTP-less; the serial-path repro boots `--no-spec`.
  Release note: `ds4-on-spark`'s `ds4-serve` wrapper should forward
  `--no-spec`/`--no-dspark` to the server when its pin reaches this
  version (its wrapper-level downgrade flags currently rely on the
  server not auto-detecting).

## v0.2.2 — 2026-07-16

Closes a silent performance-tier cliff between weight-server and standalone
boots — found chasing a latency-table discrepancy, disclosed on the
announcement thread the same day — plus a request-compatibility alias.

- **Every boot now builds the aligned fast-path artifacts** (`73c9727`).
  The fast decode and prefill dispatches (aligned-SoA D2R tiers) read
  derived repack artifacts that only `ds4_weight_server` built, so a
  standalone (self-load) boot silently fell to the raw-layout tier:
  decode 13.9 vs 17.8 tok/s p50, prefill 488 vs 853 tok/s, TTFT roughly
  doubled at the pp≈2048/tg=256 bench shape — with zero log tells. The
  one-command installer boots exactly that way. The repack builders now
  live in one shared library (`cuda/mmq/ds4_repack.{h,cu}`) compiled into
  both the engine and the weight server; manifest-less boots build the
  same artifacts in-process (78.7 GiB in ~22–26 s on GB10) and register
  them through the same lookup the import path uses. Precedence: manifest
  import > in-process build > raw fallback. `DS4_CUDA_BUILD_ARTIFACTS=0`
  opts out of the boot-time build; `DS4_CUDA_NO_DERIVED_WEIGHTS` still
  disables derived artifacts entirely. Gated: 474/474 FNV-1a artifact
  bit-identity vs the weight-server build, byte-identical accept traces
  per tier pair, and decode/prefill/TTFT parity with the weight-server
  control (17.8 tok/s / 868 tok/s / 5.3 s) on a standalone boot. The
  deep-context gate re-stamped on the release binary also improved:
  deep decode 163 ms/tok at 517K (was ~177 on the raw tier) and the
  cold half-million-token admit ~33 min (was ~41).
- **The active perf tier is never silent again.** One canonical boot line
  (`built in-process` / `imported from weight server` / `none` plus the
  reason), `ds4_derived_artifacts{source=…}` and
  `ds4_derived_artifact_bytes` gauges on `/metrics`, and
  `artifact_source` in `/v1/stats`.
- **`enable_thinking` is accepted as a `think` alias** (`f090ed2`) — the
  Qwen/vLLM convention, honored at all three request-parse sites alongside
  the existing `think` and `thinking.type` forms.

## v0.2.1 — 2026-07-16

Serving observability plus a models-list fix, both requested by users on the
v0.2 announcement thread within a day of posting. No kernel or placement
changes; quality surfaces are untouched.

- **Observability: one metrics core, three porcelains** (`118592e`). A single
  `ds4_metrics` registry (relaxed-atomic counters plus a 60-second rolling
  window) feeds every user-facing surface; readers never take the generation
  lock, so metrics stay pollable even during a minutes-long deep-context
  prefill. The surfaces:
  - a **`timings` block next to `usage`** in every response (and on the final
    streaming event with `stream_options.include_usage`): TTFT, prefill
    tokens with the cached/computed split, prefill and decode tok/s, and
    speculative acceptance + tokens-per-step when DSpark is active.
    Inapplicable fields are omitted, not null.
  - **`GET /metrics`**: Prometheus text exposition (hand-rolled, no
    dependencies) — request outcomes, token totals, rolling decode tok/s,
    live banks, KV pages, admission classes, speculation and quench counters.
  - **`GET /v1/stats`**: human-readable sectioned status text (JSON with
    `Accept: application/json`); `watch curl` works as a status board.
  The release gates now assert health from `/metrics` counter deltas as
  primary, with the stderr greps retained as fallback for one release.
  Overhead gated at zero: fresh-boot A/B produced byte-identical accept
  traces at 51.3 vs 51.4 ms/tok.
- **`/v1/models` lists only the loaded model** (`74928f0`). The endpoint
  hardcoded both `deepseek-v4-flash` and `deepseek-v4-pro`, so a Flash-only
  box advertised two ids as if selectable when the `model` field never
  switches weights (it is a label plus the `deepseek-chat` /
  `deepseek-reasoner` thinking toggle). `GET /v1/models/{id}` stays
  permissive for both known ids.

## v0.2 — 2026-07-15

The robust-serving release. v0.1.1 was held back when our own tool-calling
gate exposed two ship-path CUDA crashes; v0.2 ships those fixes plus the two
serving capabilities the agent workloads actually needed — speculation for
thinking/tool traffic and deep-context capacity — behind standing release
gates that run on every ship candidate.

- **Crash class fixed: cont admission-chunk OOB mirror reads** (`f16820c`).
  On ≥128-row admission chunks the token-tile comp-mirror kernel eagerly
  decoded `[0, max-over-banks n_comp)` of a *shallow* bank's demand-mapped
  mirror — an unmapped READ (dead ctx / zeroed-value corruption) that was
  placement-deterministic, not a race. Fix: bank-true `n_comp` for the
  single-run path. Warm + partial-prefix admission defaults re-enabled.
  Companion hardening: in-flight work now drains before sticky-scratch
  growth frees (`86a25e0`).
- **Speculation for agents: DSML sampler override on the continuous path**
  (`353c749`). Tools + thinking (or any temperature) no longer fall off to
  the serial path: the per-token structural/payload sampler swap that
  tool-call grammar needs now runs inside continuous batching, so agent
  traffic rides cont+DSpark. Tool-eval-bench thinking leg: same score band,
  all-batched, −27 % wall clock; Hermes agent end-to-end: every generate leg
  speculative at 80–97 % accept, 3.4–4.5 tok/step, zero serial fallbacks.
  Lossless at any temperature (delta-proposal speculative sampling — the
  verify-row logits + the request's own sampler/RNG are the only token
  source).
- **`DS4_SERVER_DEFAULT_TEMP`** (`d32e9a6`): default temperature for requests
  that omit one (agent frameworks usually do); explicit temperatures are
  untouched.
- **Deep-context capacity: 766K tokens served concurrently on one GB10**
  (`010cc08`, `32e1dab`). Admission/placement fixes for the multi-agent deep
  shape — a 518K-token orchestrator + 248K subagent concurrently at ctx
  524288, needles exact to 518K, warm in-place turn-2 TTFT 1.2 s (vs ~41 min
  cold prefill), deep decode 146–177 ms/tok at 248–519K:
  - batch geometry now rejects configs exceeding the uint32 absolute-row ABI
    instead of wrapping (`010cc08`);
  - comp-cache page budget refreshes live at admission time when free memory
    has grown since boot (pinned deterministic via `DS4_BATCH_VMM_BUDGET_MB`);
  - `DS4_SERVER_PIN_MIN_TOKENS` (default 65536): warm records above the
    threshold form a pinned LRU tier — a deep orchestrator trunk is never
    evicted by short-lived tenants while shallow victims exist;
  - deep-trunk fork guard: warm *fork* placement is refused above the pin
    threshold (fork-by-copy re-maps the whole committed extent, ~10 MiB per
    1K tokens at F32 — a 518K fork projected 6.8 GiB); the in-place warm path
    rides existing pages instead. The same rule now covers *partial-prefix*
    forks, keyed on the cut extent: sequential unique deep prompts sharing a
    long prefix previously stacked a fresh multi-GiB bank per request until
    a scratch allocation aborted the server (observed at six ~130K banks
    under a pinned page budget); past the threshold the trunk is truncated
    in place instead of copied.
- **Standing release gates** (`749dada`, `83f3ae5`, plus this release's
  additions in `speed-bench/`): `teb_gates.sh` (69-scenario tool-calling
  crash/fast/think legs, band 81–86, health+engagement gated; opt-in
  hardmode 73/100, error-injection @20 % 82/100 — both twice — and pass^k
  trials: mean 84.7 ± 2.3, pass@k 81.2, pass^k 71.0), `deep_ctx_gate.sh` (the
  766K/518K capacity gate above, one boot, three stages; re-passed on the
  release binary: ~750K concurrent, needles exact, warm turn-2 TTFT 1.7 s),
  `bank_churn_soak.sh` (pinned deep trunk + 12 cycling shallow tenants:
  PASS, 41 rounds/61 min, needle miss 0, deep evictions 0, memory drift
  1.0 GiB), `needle_sweep.sh` (formal retrieval matrix: **20/20 exact** —
  10 depths × {248K, 519K actual tokens}, cold TTFT flat across depth at
  843 s / 2434 s, memory flat across the 9-hour run). Quality restamp vs
  the June baseline on the release binary: GSM8K 484/500, MMLU 364/570,
  HumanEval 149/164, MBPP 178/200, IFEval strict 442/541 (a same-day
  control build scored 443 — the engine moves 1 item of 541; June 451),
  needle 45/45 inline + 64k 15/15 + 128k 10/10. Tool-eval-bench on the
  release binary: fast 82 / think 83 (band 81–86), crash legs clean, and
  `--mtp`-less fast 82 — exact parity, MTP is genuinely droppable.

## v0.1.1 — 2026-07-13

Decode is now net-positive by default across content and depth: the terminal
yield quench floors low-acceptance requests at ~0.96× plain while the kv-depth
gate handles >64k, so served decode ≈ max(speculative, plain) everywhere.
Frontier chart: `speed-bench/v011_decode_overlay.svg` (W&P prose floor line +
C-source favorable line, both at the ship config).

- **DSpark terminal yield quench** (`DS4_DSPARK_QUENCH`, default ON; `=0` to
  disable): per-request cumulative-regret controller — every verify step,
  `debt += guard − tokens_committed` (guard 2.22 ≈ the measured 2.17
  plain-step cost of one spec step); once debt exceeds a 4-plain-step budget
  with the yield EWMA below guard, speculation turns off for the REST OF THAT
  REQUEST (terminal, reset at admit), riding the kv-gate's lossless per-bank
  nd=0 path. Calibrated offline on 60 traced requests; the naive zero-clamped
  debt variant was measured to false-quench long bursty winners and rejected.
  Gates: forced-quench identity 1.000× vs plain; gsm8k 117/120, mbpp 37/40
  through the full serving path; W&P frontier floor 0.72× → 0.96× vs plain with
  shallow wins (1.2–1.7× structured) intact; suite holds 0.99 of always-spec.
  Tunables `DS4_DSPARK_SHADOW_{GUARD,ALPHA,MINEV,BUDGET,CREDIT_CAP}`;
  `DS4_DSPARK_QUENCH_FORCE_STEP` for identity testing. Supersedes
  `DS4_DSPARK_ADAPT_GATE` when both are set.
- **DSpark per-step trace + offline policy replayer**
  (`DS4_DSPARK_TRACE=1` + `tools/dspark_trace_replay.py`): per-request
  per-step yield/comparisons/drafts/latency telemetry, validated to reproduce
  `CONT_MTP_ACCEPT` aggregates exactly; the replayer calibrates quench
  parameters against recorded traces (`validate` / `replay --grid` /
  `inspect` / `selftest`).
- **Q2K drafter is the ship default** (was Q4K): equal throughput and
  acceptance in A/B (accept ±3pp, mean within noise), 6.49 vs 10.71 GiB in the
  weight server — the freed 4.2 GiB removes the deep-context boot knife-edge —
  and required for ~1M-token KV.
- Rollback checkpoint capture now derives from actually-packed draft rows
  (skipped when a step packs no drafts, e.g. every request's first MTP step);
  the restore pass provably no-ops there, so no-draft steps match plain cost.
- **DSpark kv-depth auto-gate** (`DS4_DSPARK_MAX_KV`, default 65536, 0 = off):
  speculative decoding is auto-disabled per sequence once its kv frontier crosses
  the threshold — acceptance decays with depth while the multi-row verify forward's
  cost grows with kv, netting a loss at 64k+ on prose (0.75–0.90×). Gated banks
  decode plain (verify = 1 row, no draft/injection); lossless by construction.
  Default set by the 2026-07-11 probes: spec still wins at 49k on both prose
  (1.10–1.49×) and code (1.19×); raise further for code-heavy serving.
- **DSpark adaptive kv gate** (`DS4_DSPARK_ADAPT_GATE=1`, opt-in, experimental):
  replaces the static cutoff with a runtime measure-and-switch controller past
  `DS4_DSPARK_ADAPT_START` — times the settled mode, probes the alternative,
  keeps the faster with hysteresis, re-probing periodically. Correct decisions
  8/8 in probes; costs ~5–12% vs oracle-best in probe overhead, hence opt-in.
  Solo-stream only; ring injection stays on during spec-off windows so spec can
  re-enter safely. See `misc/cuda-env-vars.md`.
- Fix serial-path lazy graph alloc OOM under bank starvation (`1da9467`): cont
  token-id echo, session-graph fit gate (`DS4_SESSION_GRAPH_FIT`,
  `DS4_SESSION_GRAPH_HEADROOM_MB`), allocation early-bail.

## v0.1.0 — 2026-07-10

384 fork commits on `batched-serving`, released as branch `release/v0.1.0`.
Headline numbers measured on GB10 (DGX Spark class, sm_121),
DeepSeek-V4-Flash IQ2XXS-mixed GGUF; methodology in the release notes.

### Serving & API

- **Continuous batching** (`DS4_SERVER_CONTINUOUS=1`, default): mid-flight admit/evict,
  per-bank KV state, FCFS pending-prefill interleave (`DS4_CONT_PREFILL_CHUNK_LIVE`),
  chunked cold admission (~1.9× admit, `DS4_CONT_PREFILL_CHUNK=4096`).
- **Request coalescing** default-on for non-streaming groups (`b5c6a83`), budgeted by
  prompt+output token footprint (`9339ab2`).
- **Warm start & fork**: per-bank prefix warm start (TTFT ~7×), D2D bank-clone fork
  fan-out (N=4 TTFT ~49×), partial-prefix fork (`8a929a1`); victim order
  invalid > superseded > LRU.
- Stops + tool calls ride the batched path (tools batch greedy-only); OpenAI tool-argument
  streaming with incomplete-call rejection.
- Budget-computed bank fit from MemAvailable (not cudaMemGetInfo); KV ≈ 9.46 KiB/token +
  ~94 MiB bank floor; lazy single-session graph allocation (boot-time GPU footprint ~0).
- Batched forward scales 6.2× from batch 1→128; batchable stops/tools/MTP all supported.

### Prefill engine (CUDA) — 12k cold prefill 305 → ~800 tok/s

Each landing independently gated (bit-parity or value-parity → same-boot ABBA → nsys →
slice evals) and reversible by env switch:

- sm_121 native arch build (`98c55ad`, `make cuda-spark`) — 306→339
- mm_ids case-1 fast path + heads8 occupancy pin + SoA-direct mmq tile loaders
  (`86c5d4d`/`2e519d0`/`42cf8ca`) — 339→420
- Sanitize/rms_norm/hc_expand folds into GEMM activation converts (`6e415a7`/`a574241`/
  `57e0821`) — 420→446
- mm_ids W8192 smem-cliff kill, two-pass no-smem (`c9da2fa`; default chunk stays 4096)
- **D2R (decode-to-registers) MoE GEMM family** — direct-to-register dequant kernels
  reading the weight server's aligned SoA artifacts in place:
  - Q2_K down-GEMM (`49329aa`) — 427→493
  - IQ2_XXS gate/up pair GEMM (`e5668be`) — 493→557
  - Expert-major CTA schedule, L2-reuse grid order (`2e68f52`) — 674→768
  - Dense-Q8 16-warp m128n128 kernel on the kind-5 aligned artifact for shared-expert +
    q_b projections (`154174e`) — 779→800
- **Token-tile HMMA attention** (indexed prefill `47438d7` + decode-mixed `9de3044`),
  replacing heads8_online — 557→640
- Memset audit: blanket GEMM output zeroing dropped, bit-exact gated (`9adc3df`) — 640→678

### Decode engine (CUDA) — plain cont+capture w1 48.9 ms/tok @HEAD (C1-era 54.0); DSpark ship 3.02 tok/step

- M1/M2 fused decode kernels: aligned-SoA IQ2_XXS + Q8_0 decode paths, fused
  gate+up+SwiGLU, fused HC stage, fused router (top-6), fused compressor pair+store,
  fused QKV-post (head_rms+rope / kv-rope+fp8+store), q8 activation folds.
- Q2K moe-down aligned row-pair-SoA repack, bit-exact decode twin (`e221241`, default ON).
- **C1: per-layer CUDA-graph capture** of the batched decode step (`d8cf4f9`, default ON;
  45.1→37.8 ms/tok ship).
- C2: bank-agnostic cont graphs (state lanes, on-device bank resolve), multi-live
  PLAIN + VERIFY capture (`749a1e4`/`23dcb1f`/`46ab301`).
- C3: batched fused router/compressor/HC at small widths (`cd8ad1a`/`0a48aac`/`ee3da19`),
  indexer-producer gating (`e82cbc7`).
- Decode width tiers: NATIVE_F16 / MMVQ_DECODE_MAX_TOKENS=8 dispatch, expert-vec split
  for all decode widths, multi-column output_a widths 2–8.

### Speculative decoding

- **MTP verifier paths** (top-2 verify, fused two-token MoE down); mode-2 batched
  spec-decode opt-in (`DS4_CONT_MTP_MODE=2`).
- **DSpark block-draft spec decode, lossless** (`82b2622` + Phase D chain): on-device
  Markov refine, multi-seq inject with per-bank KV slabs, prefill-region injection,
  auto verify-depth, concurrency auto-gate (nlive≤1); ~2.0 tok/step at 85.7% accept
  on eval workloads.
- **WS-served drafter** (`d8cc99d`): 58.8→44.3 ms/tok in DSpark ship config at 72.3%
  accept; ships Q4K, with the Q2K drafter variant available (−4.2 GiB, 87% accept;
  required for ~1M-token KV).

### Weight server

- CUDA weight-server lifecycle with VMM backend: allocate/upload model ranges, broker
  fds to clients, direct-I/O uploads, scoped imports, manifest staleness rejection,
  ownership locks, telemetry (`f2f424e`…`e7f7ce1` chain).
- **Aligned-SoA repack artifacts** (iq2 + q8 default-on `6de508d`, q2k default-on
  `6f44a5e`): the artifacts both decode-vec and D2R GEMM kernels read in place.
- Parallel aligned-repack builders (WS boot tax 63→21 s, `802e4f3`).
- Drafter serving with never-split range plan.

### Long context & KV

- Compressed-KV storage tiers: FP8 codes primary (`74a617a`), FP4 e2m1 indexer
  primary (`e7c4826`) — bit-lossless vs F32 storage (caches hold model-quantized values).
- VMM demand-mapped comp/index slabs; 128k contexts served correctly (~4.9 GB/bank at
  139k ctx, max_seq fit-reduces); DSA prefill context-flat.
- Needle 8k–128k: 70/70 at the June baseline; HEAD re-stamp 45/45 + 15/15 + 10/10
  (128k tier at `63d9d5e`).

### Fixed

- **Token-tile attention launch failure at 94k+ context** (`63d9d5e`, found by the v0.1.0
  needle gate): union-builder smem opt-in now budgets static+dynamic; prompts in the
  ~94k-98k band no longer fail chunked prefill.
- **Silent truncation of coalesced batched generation** (`9339ab2`, found by the v0.1.0
  eval gate): batch ring now sized for the generation horizon; budget clamps log loudly;
  coalesce groups bounded by footprint.
- Cross-stream use-after-free of pool scratch (cont BOS-spam root, `66d6dc3`);
  mm_ids_helper warp race + DS4_CUDA_DETERMINISTIC (`491f584`); quantizer unwritten-tail
  nondeterminism; mmvq −1 router ids; ring-lane addressing by live-slot ordinal
  (`cd00600`); MTP validator F32 ffn_gate_inp; free-on-grow graph-cache invalidation;
  macOS Metal-stub link fixes (`26b8816`, `259075d`).

### Method & infrastructure

- Engine proof runner + weight-server proof flow; stdlib-only resumable eval harness
  (GSM8K/MMLU/HumanEval/MBPP/IFEval/needle, inline scoring, watchdog-supervised).
- Measurement discipline: same-boot ABBA with SM-clock medians for <5% deltas; proven
  path engagement + finish-reason shape asserts in eval gates; ncu counter law
  TF = 906·IPC/inst-per-MMA; negative results recorded in ledgers.
