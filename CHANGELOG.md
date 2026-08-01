# Changelog

All notable fork-side changes to this project are documented here.
Fork: [Entrpi/ds4](https://github.com/Entrpi/ds4) of
[antirez/ds4](https://github.com/antirez/ds4); upstream fork point `e16ead1`
(2026-05-29). Upstream's own changes are not repeated here.

## v0.5.0 — 2026-08-01

The deep-prefill/decode substrate release, shipped together with the
DeepSeek-V4-Flash-**0731** weights refresh. Two stories in one cut: the
engine closes its flat-pool arc (every remaining per-layer activation
requantize retired, bit-exactly) and unlocks CUDA-graph capture at every
depth; the model refresh re-bases the quality baselines dramatically
upward on the identical ship-recipe quant.

**Engine — flat-pool arc (all bit-exact, each gated separately):**

- **Fused own `out_a`** (`9339bf0`): attention out_a epilogue becomes one
  fused kernel, retiring a pack + cuBLAS pair. 515K prefill record
  679.3 s = 759.6 t/s (+2.3%) at its ship — since re-taken at the
  v0.5 tip: **517,963 tokens / 667.1 s = 776.4 t/s** (+2.2% again, the
  stream-topk tier working inside deep prefill selection); ABBA 2k
  +4.4%, 64k +3.5%.
- **q8_1 dual-emit** (`26662e0`): out_a epilogue dual-emits the q8_1 of
  low, retiring out_b's input quantize (verify 0/16384 diffs).
- **MoE gate/up Y-indirect staging** (`9c1c4cb`): quantize once per
  token + ids_src map folded into one prologue register per thread
  (trip-invariance; NT64 SASS byte-identical; verify 172/172 clean).
- **Norm triple-emit** (`76f5266` + `507921d`): the RMS-norm producers
  emit f32 + f16 + the q8 D4 activation in one pass behind a
  pointer-keyed producer registry — all FIVE per-layer K=4096 input
  quantizes retired, zero dispatch-site changes. Verify 344/344 +
  172/172 clean; ABBA 64k +1.33% / 2k +2.77%, above the GPU-time bill
  because 215 retired launches per window also relieve the CPU-bound
  shallow window. Bench records: 2k 958–960, 64k ~933 t/s.
- Eager-launch jitter is BANKED AS A NAMED FLOOR (13.8 ms/win at ~4,070
  launches/win, launch-rate-insensitive; the lever is prefill capture,
  future work).
- Credit: parts of the prefill kernel line trace inspiration to Marco's
  MIT-licensed GB10 fork
  ([DS4-GB10-GX10-DSpark-CUDA](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA));
  ported pieces carry commit credit and Portions-Copyright headers
  in-tree.

**Engine — deep substrate:**

- **Exact mxf4 scorer select at all depths** (`3af807b` + follow-ups):
  the indexer scorer's mxf4 selection chain is exact at every context
  depth on the serving path.
- **Streaming top-512 at every depth** (`9922a98`): the topk dispatcher
  takes a capture-stable streaming tier for all widths and both regimes
  above 8192 compressed rows; the >8192-row CUDA-graph capture
  exclusion is LIFTED. Deep decode at 240K: **−4.7% per verify step**
  (132.1 vs 138.5 ms), −10.9% ms/tok on the gated leg. The legacy
  chunked tree survives only as a forensic escape
  (`DS4_CUDA_NO_TOPK_STREAM`); an in-tree dual-run instrument
  (`DS4_CUDA_TOPK_STREAM_VERIFY`) byte-compared stream vs tree across
  ~5,600 live launches: zero diffs.
- **Boot prewarm** (`d872609`) + **quality/test hardening** (`07dea00`):
  goldens re-based cross-box (abs=0), per-step vector exclusion and a
  golden record mode in the tests, build-config guard
  (`.ds4-cuda-config.mk`).
- **Quench recal for the 0731 identity**: `dspark_shadow_guard`
  2.10 → **2.16** (same-day 12k pair, 166-step sample: plain 50.5
  ms/step, spec 109.3 → C 2.165; spec worth 1.42× at 12k). The deep
  break-even ramp (measured ~2.37 at 240K) remains open, tracked for
  v0.6. Gates: teb fast 86 (top of band; score and spec-hit totals
  identical to the 2.10 control) and think 83 in band, counters clean.

**Model — DeepSeek-V4-Flash-0731 refresh:**

- Ship quant: `...chat-v2-imatrix-0731.gguf` (86.7 GiB, published
  upstream — same recipe the fork's baselines are stamped on). Perf
  parity with the old quant confirmed by ABBA (+0.07%/+0.15%).
- Quality re-base on the identical harness/corpus (2,248 items, zero
  request errors; frozen caps, with the June cap-correction addendum
  repeated — both bases quoted per the reporting convention):
  **MMLU 79.5% vs 63.5** (the 2-bit-expert knowledge-recall weakness
  largely closed) · GSM8K 96.4 frozen / 96.8 corrected (band) ·
  HumanEval 88.4 frozen / **89.0 corrected** (above the old 88.4) ·
  MBPP 90.0 (exact) · **needle 70/70** (baseline-exact through 130K) ·
  think-GSM8K 39/40 · IFEval strict 74.3 frozen / **82.6 corrected**
  vs the old 83.4/86.9. The IFEval frozen drop is mostly a cap
  artifact: 0731 writes ~30% longer on open instructions and hit the
  frozen 768-token cap 2× as often (177/541 vs 85); 143 of 180 capped
  items pass strict when allowed to finish. The residual ~−4 pts
  corrected-vs-corrected is a real 0731 style delta (longer, looser
  instruction adherence), model-side, not engine.
- **MTP retired for 0731**: the upstream checkpoint replaced the
  single-block MTP module with the DSpark stages; there is no 0731 MTP
  head. Speculation is DSpark-only; the ds4-on-spark launcher and
  installer refuse the legacy-MTP × 0731 pairing and manage the weight
  upgrade (optional old-weight removal, prompted).
- **0731 DSpark drafter** re-extracted from the new checkpoint
  (Q2K-Q8, 6.97 GB): accept 73.7% / 3.08 tok/step at 12k — parity
  with the old pairing's band. Published at
  `bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF` as
  `DSpark-drafter-Q2K-Q8-0731.gguf` (the installer's default fetch).
- Model-behavior notes: 0731 answers deep-context prompts tersely
  (deep gates' long-generation samples shrank 512 → 8 tokens), and the
  tool-eval safety surface moved (new advisory TC-58; the old model's
  known TC-60 no longer fires).

**Known issue (fix scheduled v0.5.1):** the demand-mapped comp/index
cache pool is grow-only; long-running agentic serving at deep ctx can
walk banks to their maximum extent and squeeze the weight page cache
(field-reported throughput cliff). Workaround: pin the pool with
`DS4_BATCH_VMM_BUDGET_MB` (admissions beyond it reject cleanly). The
v0.5.1 trim-on-evict releases evicted banks' pages under pressure.

Also in this release: **adaptive side-model residency** (`6f59a16`) —
the server mlocks the drafter at load and a dedicated watchdog thread
releases all locks one-way under memory pressure. Post-ship A/B
measurement banked ZERO speed win (two ABBA nulls with bitwise-identical
speculation counters; the hold zone and the refault-tax zone are
disjoint on 121 GiB unified memory), so the mechanism is scheduled for
removal in v0.5.1 alongside trim-on-evict — the wedge-forensics and
capacity laws it produced are the keeper.

## v0.4.2 — 2026-07-24

Community fix: thinking-mode conversations now reuse KV on the
continuous path. Contributed by [@fabiopili](https://github.com/fabiopili)
in [#4](https://github.com/Entrpi/ds4/pull/4) — an excellent two-symptom
diagnosis and a fix that mirrors the session path's own design.

- **The bug**: `cont_warm_retire()` bailed on every thinking-mode row,
  so thinking conversations never built warm records. Two symptoms:
  every turn cold-re-prefilled the full history (measured 40 s vs 0.8 s
  on turn 2 of a 29K-token thinking agent preamble), and thinking banks
  never persisted to the disk KV tier (`kv_cache_store_bank` requires a
  warm-valid bank) — the root cause of
  [ds4-on-spark#4](https://github.com/Entrpi/ds4-on-spark/issues/4).
- **The fix**: port the session path's visible-transcript checkpoint
  key to the continuous path. The warm record keys on the rendered
  visible transcript (`</think>` + content + end marker for toolless
  turns; verbatim reasoning replay under tool context), which is
  byte-identical to what the prompt renderer emits on the next turn.
  Safety is by construction: a full record match reuses the bank's
  exact committed tokens under the engine's prefix validation, and
  partial cuts are token-LCP against committed history, so a cut can
  never land inside hidden reasoning. Bonus (by design): a
  `deepseek-chat` continuation of a `deepseek-reasoner` conversation
  byte-matches the thinking bank's visible record and reuses it.
- **Gates**: deep 240K all-stages pass, turn 2 **54.2 ms/tok** at 2.69
  tokens/verify-step with a warm-in-place reuse of the full 240K prefix
  (`warm admit cached=240156`; v0.4.1 stamp 57.3 at 2.76); deep 12K
  36.9 ms/tok at 2.95 (v0.4.1: 36.6 at 2.95), reuse + needles exact,
  counters clean. tool-eval-bench fast: score 88 ×5 runs, zero
  failures/serial/rejects/illegal ×5, admit streams byte-identical
  across runs and emit/draft totals conserved (19453/16206); spec-step
  tiling wobbles ±1 counter with arrival timing — a same-day control
  on the v0.4.1 binary bounded the class, and the non-thinking retire
  path is byte-identical code to v0.4.1. Thinking gate on the merged
  tree (DGX Spark, 4 legs): toolless turn-2 reuse (`fork admit
  cached=4140`), preserved-reasoning reuse (cached=4473), non-thinking
  control, and clean-SIGTERM persist (12 bank lines, 6 disk files,
  previously always empty for thinking) with reboot restore (`bank
  restore hit tokens=4232 load=159.5 ms`) and post-restore reuse —
  plus a needle planted in the reused prefix answered exactly on every
  leg. make test tensor equivalence: all-zero deltas.

## v0.4.1 — 2026-07-22

The quench recalibration patch. The terminal yield-quench controller's
break-even guard was still the v0.1.1 calibration (2.22, from a measured
spec-step cost C=2.17); v0.4's substrate work cut the verify cost to a
measured C of 2.03–2.08 on the 2k–64k band (2.40/2.38 at 240K/515K), so
the shipped guard sat 8% above break-even and terminally quenched
winners. One constant changes: `dspark_shadow_guard` 2.22 → **2.10**.

- **Method** (tools tracked in-repo): `DS4_DSPARK_TRACE=1` always-spec
  collections on four shapes (W&P and code low bands; 240K and 515K
  turn-2 via `deep_ctx_gate.sh`, which gains a `SPEC=0` plain-reference
  knob), replayed with `tools/dspark_trace_replay.py` (validate: 74/74
  engine SHADOW lines bit-match the replayer). Break-even C is
  identity-calibrated per band: C = geo(yield) / geo(measured same-run
  speedup). Candidate families swept under per-request-cost economics:
  flat guards, depth-ramped guard(pos0), non-terminal re-arm. Re-arm
  LOSES even at an optimistic zero-cost bound (floor 0.926); the depth
  ramp ties flat 2.10 at Δ2e-4 and is deferred (reopening trigger:
  observed deep content yielding between the guard and C_deep ≈ 2.4).
- **Measured** (GB10, fresh same-day plain reference both sides):
  code-corpus band geomean vs plain 1.084 → **1.103**; adversarial W&P
  band 1.02 → **1.044**; deep stamps 12K **36.6 ms/tok** at 2.95
  tokens/verify-step (identical to v0.4.0), 240K **57.3** at 2.76,
  515K **59.9** at 2.79 (v0.4.0: 62.1 at 2.75) — the deep gate corpora
  yield 2.6–2.9, above either guard, so no deep behavior change;
  forced-quench identity 1.0014 (12K ABBA) and 1.004 (41K); make test
  tensor equivalence all-zero deltas.
- **Floor honesty** (measured; replaces the "~0.96 floor" line): the
  quench worst case is pre-quench learning debt — minev (4) speculative
  steps times the window's yield deficit — which lands short (tg 128)
  adversarial generations at 0.93–0.97x plain depending on the draw,
  under ANY guard at or above break-even (the old guard 2.22 measured
  0.932 on 2026-07-22 draws at the very point where v0.4.0's release
  leg drew 0.97). Post-quench serving is identical to plain (the
  identity above). The README now states the mechanism and the typical
  range instead of a single draw.
- tool-eval-bench fast ×2: score 88 both runs (previous band 81–86;
  the pair is bit-identical — spec counters 14173/200 twice). Counters
  legitimately move off v0.4.0's 13659/201: the recalibrated guard
  retains more speculation.

## v0.4.0 — 2026-07-21

The deep-decode substrate release. Serving decode over deep
conversations drops 27–35% release-over-release (240K: 76.3 → **55.8
ms/tok**, 515K: 95.2 → **62.1 ms/tok**, turn-2 512-token generations
at 2.57/2.75 tokens per verify step; 12K stamps 36.6 ms/tok), the
static 64K depth gate on speculative decode is gone, and a
community-contributed server fix reaps queued requests whose client
disconnected before any prefill is spent. Quality held at the release
battery: evals band-exact vs v0.3.0 (HumanEval 149/164 identical,
IFEval strict 453 vs 443, needles 70/70), tool-eval-bench 86 twice
with bit-identical engine counters, tensor equivalence all-zero.

- **Head-group flash-decode for dense mixed attention** (`17a7d76`).
  The deep-decode depth tax was the ratio-128 mixed-attention family:
  eight heads now share one f32-staged KV tile (fp8 decoded once per
  head-group instead of once per head), with q held in registers,
  warp-per-head dots, online-softmax partials over row chunks, and a
  fixed-order combine. Isolated width-5 verify launches went 1876 →
  391 µs. Escape: `DS4_CUDA_NO_ATTN_HG`.
- **Indexed-path gather flash-decode at serving widths** (`e0ed742`).
  The same head-group idea for indexed attention: 12k 42.6 → 37.7,
  240K 66.2 → 64.4, 515K 73.5 → 71.5 ms/tok at landing time.
- **Speculative decode armed at every depth** (`b5a9ab1`). The static
  64K KV-depth gate defaults off; the terminal yield-quench
  controller (v0.2) is now the only governor of when drafting pays.
  240K 64.4 → 59.4, 515K 71.5 → 65.5 ms/tok at landing time. Setting
  `DS4_DSPARK_MAX_KV>0` restores a hard cap.
- **Dense verify tier: aligned Q8_0 at widths 1–8** (`7ffe821`). The
  aligned SoA dense artifacts previously served only single-token
  decode; verify steps fell back to raw mmvq (~90–200 GB/s). A
  width-templated kernel reads the aligned stream once per row with
  per-token accumulator columns: 240K 59.4 → 55.2 ms/tok. Escape:
  `DS4_CUDA_NO_Q8_ALIGNED_NC`.
- **Indexer scorer token-loop** (`18b40d7`). The WMMA scorer launched
  one CTA per (row-tile, token), re-staging the same K tile per
  verify token; one CTA now stages K once per sequence run and loops
  the tokens in-CTA, bitwise-identical to the per-token kernel. 240K
  55.2 → 50.6, 515K 66.1 → 61.7 ms/tok at landing time. Escape:
  `DS4_CUDA_NO_IDX_V5E`.
- **MoE gate_up first-owner expert dedup at verify widths**
  (`6e27e1e`). Live routing shows ~40% expert overlap across a
  width-5 verify's 30 slots and the 124 MB working set defeats L2;
  the first CTA owning an expert now accumulates every matching slot
  as extra q8_1 columns, so duplicate expert weights are read from
  DRAM once. Bitwise-identical outputs; win is routing-dependent
  (deeper on prose/code). The sibling dedup for the q2k down
  projection was prototyped and measured net-negative (its 2.68
  MB/expert distinct set is already L2-absorbed) — not integrated,
  verdict recorded in the proto. Escape: `DS4_CUDA_NO_MOE_DEDUP`.
- **Server: reap queued requests whose client disconnected before
  admission** (`881bcca`, contributed — PR #3 by @guptaavi, from a
  production GB10 report). A client that disconnects while its
  request is still queued used to hold an admission slot invisibly
  and then burn a full prefill into a dead socket; under continuous
  batching the zombies compound while every metric reads healthy. A
  zero-byte `MSG_PEEK` probe (peer-FIN only — a slow but connected
  client can never be false-positived) now gates all three FIFO
  exits, and dead jobs finish through the existing failure machinery
  before any engine work: visible as `outcome=failed` in `/metrics`.
- **ds4-bench: fixed illegal access sweeping past 32K at large
  `--ctx-alloc`** (`9caaa50`, reported on the NVIDIA developer
  forum). The release frontier sweep runs 2048–65536 clean at
  `--ctx-alloc 131072`.
- README GB10 leader assets refreshed from this release's runs:
  frontier sweep CSV/chart (prefill peaks 817 t/s, 535 at 65K; gen
  21.6 → 16.7 t/s across the band) and CLI speed-table rows.



Deep-context serving gets a tensor-core scorer and a durable KV tier.
Batched deep decode is 13–21% faster (240K→515K context), deep
conversations now survive bank eviction and server restarts (an
80K-token resume takes ~3 s instead of ~2 minutes of re-prefill), and
disk checkpoints store packed rows natively at ~2.3× smaller — with
cross-config restores proven bitwise exact.

- **Durable pinned banks: continuous-batching KV survives eviction and
  restarts.** Until now only the serial session's KV could persist to
  `--kv-disk-dir`; a continuous-batching bank (the per-conversation KV a
  warm record points at) evaporated whenever its bank was recycled or
  the server restarted, and a deep conversation paid its full prefill
  again. Banks now serialize through the same walkers and the same wire
  format as serial checkpoints (a bank record IS a valid serial
  checkpoint — the payload-integrity gate leg restores one in
  serial-only mode and byte-compares two fresh-boot continuations).
  Policy, with no new knobs: deep records at or above the existing pin
  threshold persist when a foreign admit destroys their bank
  (`bank-evict`); every valid record persists at graceful shutdown
  (`bank-shutdown`). At admission, a request no live record serves is
  checked against the disk tier and restored directly into a free bank
  — preferring no-value banks, with a deep-over-shallow displacement
  rule mirroring the eviction tiers so churn can't lock deep trunks out
  — after which the normal warm matching, engine-side frontier
  validation, and suffix prefill proceed exactly as for a live record.
  Restores honor the packed-native v3 format (packed primaries upload
  mirrors verbatim; write-dead F32 pages stay unmapped). The
  persist/restore runs synchronously in the admission path (seconds at
  deep contexts, replacing minutes of re-prefill); asynchronous staging
  is future work, as is carrying tool-call maps through bank restores
  (a post-restart tool-turn re-render currently degrades to a cold
  prefill, never to wrong output). New standing gate:
  `speed-bench/bank_persist_gate.sh` — which also flushed out a latent
  serial-mode bug on its first deep serial restore: the fp8 predecode
  scratch resized in single-row steps as the compressed count grew,
  eventually attempting an allocation inside an active capture window
  and killing generation a few tokens in. It is now sized once by the
  per-layer compressed cap (fixed in the preceding commit).

- **Batched-decode indexer scoring rewritten on tensor cores — deep
  decode ~13% faster.** The multiseq scorer (every continuous-batching
  decode step, all widths including width 1) was a scalar
  per-(row, token) kernel that was latency-bound, not bandwidth-bound:
  33 block-wide barriers and a 32 KB q re-read per compressed row put
  its cost at ~16 ms/step at 240K context, and the fp4 mirror's 6.4×
  byte reduction moved nothing. The new kernel stages 32-row tiles of K
  as unscaled e2m1 levels (always fp16-exact; the scaled values
  underflow fp16 at deep-context block scales, which is what rules out
  a naive half conversion) and MMAs them against per-warp q tiles,
  folding the F32 block scales back per 32-wide chunk — chunk width
  equals scale-block width, so the fold is exact. Q levels divide by
  the exact commit scale via a new scale-only emit from the indexer-Q
  QAT (1 KiB/token); the F32-primary config re-derives K block scales
  in-kernel with the emit QAT's own derivation, so fp4-primary and
  F32-primary configs stay bit-identical on the same binary. Measured:
  scorer 744→188 µs/launch at the 240K shape (3.97×, .15 proto ladder,
  top-k flips 0/2048 on every leg, 4-token batch ≡ single-token
  launches bitwise), 240K turn-2 deep decode 87.6→76.3 ms/tok
  (−12.9%), 12k turn-2 unchanged in its 44–48 band, needles exact at
  12k/240K/250K/500K. Wins at every tested shape (43→21 µs even at 12k
  context), so dispatch is unconditional. Serial (non-batched) decode
  keeps the scalar kernel: serial refuses deep contexts by design, and
  at shallow depths the scorer is under 1 ms/step.

- **Disk-KV payloads store packed rows natively (format v3).** Packed-
  primary sessions serialize the fp8/fp4 mirror codes+scales verbatim
  instead of dequant-expanding to F32: checkpoints shrink ~2.3× at 16K
  context (263→115 MB, growing toward ~3× at depth as compressed rows
  dominate), saves drop the expansion pass, and a packed-primary restore
  uploads the codes verbatim — no re-encode, and the write-dead F32
  pages stay unmapped. Cross-config restores remain exact in every
  direction (an F32-primary reader dequant-expands packed payloads
  through the emit scratch; a packed reader of an F32-row payload
  re-encodes as before), v2 payloads remain readable forever, and the
  kv_crossmode gate proves the full 6-leg matrix byte-identical. Found
  and fixed along the way: the e4m3 decode table was only initialized
  when `DS4_CUDA_FP8_KV` was enabled, so an F32-primary server
  dequanting packed codes read an all-zeros table (every non-RoPE lane
  restored as 0.0) — it is now initialized unconditionally.

- **Disk-KV restores into packed-primary servers are exact and enabled
  again.** v0.2.4 refused them because the restore-time mirror re-encode
  measurably drifted; the mechanism is now root-caused and fixed. The
  quantizer scale derivation `exp2f(ceilf(log2f(amax/T)))` is wrong at
  exact power-of-two ratios under `--use_fast_math`: `lg2.approx.f32`
  errs one-sided above at `2^-n`, `ceilf` rounds the exact negative
  integer up, and the derived scale doubles. A live encode essentially
  never presents an exactly-pow2 ratio, but a re-encode of committed
  values does routinely (a committed block max is codec level × pow2
  scale) — with the doubled scale, bottom-of-grid values fell off
  (measured: 10.5% of fp4 indexer lanes, including level-1 values
  collapsing to zero; a thin deep-subnormal fp8 tail). All QAT scale
  sites (six CUDA kernels, the CPU reference, three Metal kernels) now
  share an exact frexpf/ldexpf derivation, and codec sign tests use
  `signbit` so `-0.0` survives the round trip bitwise. Re-encode is now
  bitwise idempotent for any on-grid committed input — including
  checkpoints written by older binaries. Repro + fix proof:
  `cuda/mmq/test/proto_kv_reencode_idem.cu` (0 value mismatches / 84M
  lanes on the fixed path); end-to-end proof: `kv_crossmode_gate.sh`
  legs 3/5 now assert packed restores hit and byte-match F32 restores.
  The `DS4_KV_DISK_PACKED_RESTORE` investigation override is gone.
  Note: absolute speculative-counter values shift vs older binaries
  (committed values change in the boundary band where the old formula
  was wrong); same-binary packed-vs-F32 identity is unchanged.

## v0.2.4 — 2026-07-18

Packed FP8/FP4 compressed KV becomes the primary storage. Deep-context
decode gets 19–26% faster, every context size fits in ~3× less KV memory,
and the change is bit-lossless. The MTP head is no longer loaded when a
DSpark drafter is armed.

- **MTP-droppable is now the launch default.** With a DSpark drafter
  armed, the MTP head is fully shadowed: teb fast MTP-less is
  byte-identical in every speculative counter (13577/197) with a
  slightly better wall, and back-to-back 240K-deep stamps read 86.9
  (MTP-less) vs 87.2 ms/tok (full stack) — equal or better everywhere,
  for ~3.55 GiB of weights plus spec scratch back. The launch defaults
  therefore skip the MTP auto-attach beside an armed drafter and say so
  on the boot line (`mtp=dropped`). An explicit `--mtp` always wins,
  `--preset spark` still demands the full stack, and disarming the
  drafter (`--no-dspark`, `DS4_CONT_DSPARK=0`) restores the MTP
  auto-attach — MTP-2 remains the fallback speculation when no drafter
  is present. The `ds4-on-spark` wrapper no longer passes `--mtp` in its
  full-stack mode.

- **FP8 comp-KV + FP4 indexer primaries default ON.** The attention
  compressed-KV row stores the model's own e4m3-quantized values as 448
  1-byte codes + F32 rotary tail (704+28 B vs 2048 B F32); the indexer row
  packs e2m1 nibbles with per-32-lane scales (64+16 B vs 512 B). Because
  the F32 caches only ever held model-quantized values at F32 width, the
  packed forms are bit-lossless: teb temp-0 seed-7 speculative counters
  are byte-identical to F32 on every leg (crash/fast/think). The F32 rows
  go write-dead inside the VMM demand-mapped slabs, so their pages never
  become resident — that is where the memory comes back.
  `DS4_CUDA_FP8_KV=0` / `DS4_CUDA_FP4_INDEX=0` restore F32-primary.
- **Deep-context decode wins** (GB10 sm_121, 512-token turn-2 stamps,
  fresh boots): 120.9 ms/tok at 516K depth vs 162.7 F32 (−26%); 87.6 vs
  107.4 at 240K (−19%); +5.6 GiB MemAvailable after the 766K-token
  charter gate (needles exact to 518K both configs). The same charter
  shape at F32 now sits on a capacity cliff — on a degraded box it takes
  a clean 503 budget reject where the packed primaries pass with >6 GiB
  spare, so the flip is robustness as much as reach. Kernel levers that
  paid for it: the e4m3 decode table moved off `__constant__` (divergent
  per-code indexing serialized up to 32-way replays), address hoisting on
  all FP8 read sites, and a pair-lane layout (uchar2 codes + shared scale
  reads, quad dot) in both the dense and indexed decode-mixed kernels.
- **Shallow contexts carry a ≤2% decode tax** (12k: ~+2% ABBA-clean; teb
  mid-ctx wall +2.2%), the named floor being the in-kernel scalar FP8 dot
  kept for spec-verify counter identity plus attribution noise; prefill
  is flat at both ends. Ledgers and the elimination lever live in
  `local/docs/briefs/brief-kv-efficiency-arc.md`.
- **VMM-availability guard**: the primaries refuse to engage — loud boot
  line — when the batch comp slabs cannot be VMM demand-mapped (device
  without VMM support, `DS4_BATCH_VMM_COMP=0`, or slab-poison
  diagnostics). Eager F32+packed would be strictly worse than F32-only,
  so the server stays F32-primary there. A/B recipe note: an empty env is
  no longer the F32 control — pass explicit `=0`s.
- **Disk-KV checkpoints are now safe across storage modes** (found by the
  new `speed-bench/kv_crossmode_gate.sh`). The local session serializer
  was missing the packed-row expansion its distributed sibling got in the
  P2 era, so a packed-primary server checkpointed the write-dead F32 comp
  rows — an F32 boot restoring such a file produced visibly deranged
  output. Saves now expand the packed rows to their exact F32 values
  (file format unchanged); gate-proven by F32 boots restoring
  packed-written checkpoints byte-identically to F32-written ones.
  Restores *into* a packed-primary server are refused with a loud
  warning for now — the restore-time mirror re-encode is measurably not
  bit-exact yet, and a refused hit costs one prefill where a drifting
  hit silently changes output. Exact packed restore (persist
  codes+scales, or prove re-encode idempotence) is queued for v0.3;
  `DS4_KV_DISK_PACKED_RESTORE=1` overrides for investigation. Disk KV is
  a serial-path feature — continuous-batch serving is unaffected.
- **Counter-identity made structural.** The fp8-vs-F32 byte-equality of
  teb temp-0 speculative counters — the arc's losslessness tripwire —
  turned out to be a compiled-form coincidence: after the decode-kernel
  rewrites, think-leg `spec_hits` drifted by single digits in **both**
  configs (F32 moved −1 with its source untouched; fast-math is free to
  re-schedule the dot chains on any recompile) while every transcript
  stayed byte-identical, i.e. the drift lives entirely at draft-accept
  margins the verifier corrects. The config-branched comp dot chains in
  the decode-mixed and indexed-mixed kernels (fp8 *and* F32 branches) are
  now pinned to one in-order chain with `__fmaf_rn`/`__fmul_rn`, so
  same-binary cross-config counter identity no longer depends on
  compiler mood. Verified: fp8 and F32 think legs land identical
  `spec_hits` on the pinned binary.

## v0.2.3 — 2026-07-17

One-command serving: the launch defaults move into the engine binary.

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
  The `ds4-on-spark` v0.2.3 pin updates its `ds4-serve` wrapper to
  forward `--no-spec`/`--no-dspark` to the server (its wrapper-level
  downgrade flags previously relied on the server not auto-detecting).
  Gated: `launch_defaults_gate.sh` (zero-config / `--no-spec` /
  `--preset spark` legs) ALL PASS on GB10 sm_121 with live speculative
  engagement, plus teb on the release binary — crash 83+100, fast 86,
  think 83 (band 81–86), 0 serial starts, 0 admission rejects.

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
