# Solar Open2 DGX Spark optimization handoff

Status: **resumed and advanced on 2026-08-10 KST; no model process running at
capture.** The chunked-KDA prefill landed as the default production path,
removed KDA from the top of the kernel profile, and moved the first-order
bottleneck to IQ2_XXS `mul_mat_q` as the previous capture predicted.

## Goal

> Implement and optimize Solar Open 2 directly as a CUDA-KDA family in
> `Baekpica/ds4`, hand only expensive non-model development state to DGX Spark
> through an HF Bucket, then keep the GGUF revision fixed while specializing
> and optimizing `sm_121` until one GB10 serves the full 1,048,576-token
> context from resident memory.

The goal is **not complete**. Native loading, short-context correctness,
resident model copy, staged KDA lifecycle work, compressed-KV kernels, and a
2K full-model KV comparison are complete. The 32K/64K lifecycle gates,
128K/256K/512K/exact-1M admission and semantics, and the 1M API server remain.

## Immutable inputs

- Model repository: `Baekpica/Solar-Open2-250B-Mixed-Quant-GGUF`
- Fixed weight revision: `cd504d0ce46c0850dd34c319659e020a17ea7194`
- Files: `Solar-Open2-250B-MXQ-v1-00001-of-00011.gguf` through
  `Solar-Open2-250B-MXQ-v1-00011-of-00011.gguf`
- Exact split total: `95,533,532,160` bytes (`88.972535133362 GiB`)
- Hash manifest: `MXQ-v1-SHA256SUMS`
- Local first shard:
  `/home/sunghoon/workspace/ds4-exaone/models/Solar-Open2-250B-Mixed-Quant-GGUF/Solar-Open2-250B-MXQ-v1-00001-of-00011.gguf`
- Expensive non-model state:
  `hf://buckets/Baekpica/solar-open2-spark-handoff`
- Latest received handoff:
  `/home/sunghoon/workspace/ds4-exaone/solar-open2-spark-handoff`

Do not rebuild, merge, requantize, or silently replace these weights during
Spark runtime work. A README-only Hub update does not change the fixed weight
revision or any shard bytes.

## Publication completed at pause

- The ds4 optimization branch was pushed to
  `origin/feature/solar-open2-model-loader`.
- The model card was updated on Hugging Face in README-only commit
  `b2669f2e6464f3cc1b38953d290b12100d975c55`.
- The published `README.md` SHA-256 is
  `aaaecd4711f38e06569ef42951d5d2c75609f73b29be089f504e0c56d54f5021`.
- Re-downloading that exact Hub revision matched the upload byte-for-byte.
- A shallow Git audit of the Hub commit reported only `M README.md`; no GGUF,
  manifest, hash, recipe, license, or other repository file changed.

## Checkout and host

| Item | Value |
|---|---|
| Checkout | `/home/sunghoon/workspace/ds4-exaone/ds4-solar-open2` |
| Repository | `https://github.com/Baekpica/ds4.git` |
| Branch | `feature/solar-open2-model-loader` |
| Received baseline | `ab05f5ee428fd3185871f5b39dd2f7b742ed0094` |
| Last measured code commit | `c36e3f2` |
| GPU | NVIDIA GB10, compute capability 12.1 |
| Native target | `sm_121a` |
| Driver | `595.71.05` |
| CUDA | `13.3` |
| Kernel | `Linux 6.17.0-1029-nvidia aarch64` |

Build with the Spark target or explicitly pass `CUDA_ARCH=sm_121`. Do not
reuse an H100 object:

```bash
cd /home/sunghoon/workspace/ds4-exaone/ds4-solar-open2
make cuda-spark
```

## Commits after the received handoff

Each performance unit was committed separately after its local correctness or
profiling gate:

| Commit | Unit |
|---|---|
| `7552c93` | make UMA resident-copy admission account for context memory |
| `813fd9a` | add staged CUDA lifecycle profiling hooks |
| `018e9f8` | cache KDA decay and normalization scalars |
| `636ea36` | add opt-in two-way KDA state parallelism |
| `c93f772` | tile compressed-KV prefill on tensor cores |
| `10a3135` | add repeatable compressed-KV prefill profiling |
| `ffa89eb` | use native FP8 decode in tiled prefill |
| `9673342` | add repeatable compressed-KV decode profiling |
| `1d21a09` | use native FP8 conversion in compressed attention |
| `778a52e` | use native FP4 decode in tiled prefill |
| `a3b2f19` | use native FP4 conversion in compressed attention |
| `f30ecb9` | overlap exact-order KDA state updates |
| `c36e3f2` | add the one-engine full-model KV format harness |
| `0b58219` | chunk KDA prefill through the delta-rule UT transform |
| `def87d4` | cover chunked KDA prefill at production width |
| `7eb5bec` | compare KV formats on the release KDA path |

## Established correctness and capacity

- Native `sm_121a` CUDA build completed on GB10.
- The complete 11-shard model mapped as one 88.97 GiB range with 1,083
  tensors and the expected 48-layer `[GQA, KDA, KDA, KDA] x 12` schedule.
- Shard hashes, tokenizer parity, S1 resident correctness, reset, snapshot,
  replay, fork, cancellation, and resynchronization gates passed at the tested
  frontiers.
- The deterministic long fixture tokenizes to 130,800 tokens. Tokenizer
  admission parity has been checked at 128K, 256K, 512K, and 1M; this is not a
  substitute for model execution at those depths.
- An 8K production-shape KDA lifecycle run passed. The remaining required
  lifecycle frontiers are 32K and 64K.
- Compute Sanitizer on the final exact-order KDA fixture reported:
  `memcheck: 0 errors`, `racecheck: 0 hazards`, and `synccheck: 0 errors`.

The current context estimator, including KDA state and prefill buffers,
projects the following exact-1M allocations. These are estimates, not completed
admission runs:

| Runtime KV | Context allocation | Resident model + context |
|---|---:|---:|
| BF16 | 49.422 GiB | 138.382 GiB — impossible on one GB10 |
| FP8 | 25.797 GiB | 114.757 GiB |
| K-FP8 / V-FP4 hybrid | 19.797 GiB | 108.757 GiB |
| FP4 | 13.797 GiB | 102.757 GiB |

FP8 remains a speed/correctness reference, not a selected release layout.
Hybrid and FP4 are the capacity candidates.

## KDA optimization result

`DS4_SOLAR_KDA_STATE_PARTS=2` divides each 128x128 recurrent state column
between two workers and is the fastest microbenchmark path, but its partial
dot-product reduction changes floating-point accumulation order. It remains
opt-in.

`DS4_SOLAR_KDA_STATE_PARTS=2-exact` lets the second worker perform disjoint
state decay/update writes while worker zero performs both dot products in the
same key order as the generic kernel. On the production shape (8,192 tokens,
64 heads, head dimension 128):

| KDA specialization | nsys kernel time | Versus generic | Numerical result |
|---|---:|---:|---|
| generic `<1,false>` | 751.884864 ms | 1.000x | reference |
| two-part `<2,false>` | 440.336064 ms | 1.707x | small expected FP drift |
| exact two-part `<2,true>` | 504.375872 ms | **1.490x** | output and state bitwise identical |

The exact path used 55 registers per thread, 8,208 bytes shared memory, no
local spill, and was stable around 0.505 seconds across repeated runs. Its
official FLA fixture now requires zero error against the generic output and
recurrent state.

The exact path is still opt-in and has not yet been used in a full-model run,
including the required 32K/64K lifecycle frontiers. Its standalone KDA output
and state are bitwise identical in the production-shape A/B.

Profile:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/nsys-solar-kda-exact2-8k-sm121-final2.nsys-rep`

SHA-256:
`e84769265877e92d3c92aad5e5947177b3757e4b0dc1589be6e88bc462d2096b`

## Chunked KDA prefill (landed 2026-08-10)

Multi-token production-shape prefill now runs the recurrence 64 tokens per
chunk through the delta-rule UT transform (`0b58219`); decode stays on the
recurrent kernel and `DS4_SOLAR_KDA_STATE_PARTS` still pins the sequence
variants for isolation. Every pairwise decay exponent is factored around its
16-row sub-block start so the -5 gate clamp at chunk depth (B down to -320)
stays inside f32 range. Five launches per prefill call: prep (bit-identical
conv/SiLU/L2/gate math into head-major planes), conv tail, chunk-parallel
factorization (Mq, the triangular inverse, U_loc = T beta V, W = T beta K~),
a persistent per-head scan whose 128x64 state slice lives in registers, and
the chunk-parallel intra output.

On the 8,192-token production shape:

| KDA path | wall time | Versus generic | Numerical result |
|---|---:|---:|---|
| generic `<1,false>` | 744.6 ms | 1.000x | reference |
| exact two-part `<2,true>` | 504.7 ms | 1.475x | bitwise identical |
| chunked (default) | 111.2 ms | **6.69x** | out max_abs 5.5e-10 vs generic |

Against the official FLA fixture the chunked path scores abs 5.290650e-04 —
the generic path's own 5.290651e-04, i.e. the fixture reference's precision,
not the reorder, dominates. Conv states stay bit-exact. Memcheck and
racecheck: zero errors, zero warnings. Coverage lives in
`tests/test_solar_kda_chunk` (scalar mirror at head_dim 128, ragged and
split-launch continuation) plus a chunked leg in the FLA fixture test.

Tuning ground truth this work established for GB10 (`sm_121`, 48 SMs, 24 MiB
L2, 100 KiB shared/SM):

- Sustained SM clock under compute load is ~942 MHz (idle 208, max 3003);
  budget cycles at ~1 GHz.
- Nsight Compute is blocked (`RmProfilingAdminOnly: 1`, no passwordless
  sudo); localize cost with template-mode ablations of the real kernel
  (`scratch/solar-open2-spark/bench-gb10-kda-micro.cu`) plus nsys.
- A per-iteration walk over rows 512 B apart in UMA memory exposes a full
  memory latency per step and can dominate a kernel 3x; bulk-stage such rows
  through shared and keep per-(chunk, head) data in contiguous 32 KiB spans.
- The two-blocks-per-SM boundary sits at ~50,176 B dynamic shared per
  256-thread block; crossing it can cost more than a staged tile saves.
- cuBLAS on the scan shapes: 128 sequential steps of 3 batched
  [64x128]@[128x128] GEMMs take 17.25 ms (launch-latency-bound); the same
  math as one parallel batched GEMM takes 0.086 ms (~20 TFLOPS f32).

Remaining KDA headroom (~3x of the 111 ms, not yet worth blocking on): the
factor kernel (48 ms) and scan (37 ms) still issue one scalar LDS per
element; the documented next step is a register-tiled GEMM rewrite (state in
shared, 4x4 micro-tiles, float4 segments) and, in the factor kernel, the
same treatment for the sub-block dots.

## Compressed GQA KV optimization result

The opt-in `DS4_SOLAR_KV_PREFILL_HMMA=1` path dequantizes one 16-key tile into
shared memory and serves 64 queries with the existing tensor-core attention
kernel. FP8 and FP4 conversion now use CUDA 13.3 native `__nv_fp8_e4m3` and
`__nv_fp4_e2m1` types in both tiled prefill and scalar compressed attention.

Repeated 257-token, 64-launch microbenchmarks converged to roughly 83-85 us for
all HMMA formats:

- FP8 tiled prefill: 274.42 us formula path to about 84.3 us native.
- FP4 tiled prefill: 261.04 us formula path to about 82.7 us native.
- Hybrid tiled prefill: 172.75 us after native FP8 to about 84.9 us after
  native FP4.
- Scalar FP8 prefill/decode: 415.68/135.84 us to 258.27/102.13 us.
- Scalar FP4 prefill/decode: 651.10/235.87 us to 352.81/158.28 us.
- Scalar hybrid prefill/decode: 529.55/192.34 us to 300.27/152.91 us.

All unit accuracy metrics were unchanged. Memcheck and racecheck reported zero
errors.

## Final full-model run completed before pause

One resident engine compared BF16, FP8, hybrid, and FP4 sessions on the same
2,048-token prefix. KDA was intentionally held on the generic path to isolate
KV-format effects; HMMA compressed prefill was enabled. Only FP8 was captured
by nsys through `cudaProfilerApi`.

- Model copy: 88.96 GiB in 137.687 seconds.
- Planned resident footprint: 88.96 GiB model + 0.26 GiB KV/state + 1.24 GiB
  buffers = 90.46 GiB.
- NVIDIA compute-process allocation observed during the run: 93,171 MiB.
- No swap growth, OOM, cgroup max event, NaN, or non-finite logit.
- All four formats selected prefill token 11047 and decode token 27294.

| KV | Context | Prefill | Prefill rate | Decode | BF16 prefill rel-RMS / 1-cos | BF16 decode rel-RMS / 1-cos |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1.500 GiB | 13.600382 s | 150.584 tok/s | 144.182 ms | reference | reference |
| FP8 | 1.454 GiB | 12.873574 s | **159.086 tok/s** | 111.528 ms | 1.71895 / 0.72171 | 0.63269 / 0.20585 |
| hybrid | 1.442 GiB | 13.679214 s | 149.716 tok/s | **110.832 ms** | 0.23706 / 0.02162 | 0.43587 / 0.08708 |
| FP4 | 1.430 GiB | 17.522045 s | 116.881 tok/s | 110.943 ms | 0.31509 / 0.03606 | 0.47039 / 0.11055 |

Interpretation:

- FP8 is fastest in this one 2K run, but it has by far the largest full-logit
  drift. Matching top-1 once is not enough to promote it to the 1M release
  layout.
- Hybrid is much closer to BF16 than FP8, has the smallest projected 1M
  allocation that still keeps K at FP8, and has the fastest observed decode.
- FP4 is a viable capacity layout but its current prefill store/quantization
  cost made this 2K run 14.1% slower than BF16.
- Run the semantic suite before selecting any compressed layout.

Log:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/solar-kv-full2k.log`

SHA-256:
`8f8b3d945ab9ebb5809633fd47a791be78270bb915b2a26488a58238d0594d2b`

FP8 nsys profile:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/nsys-solar-kv-full2k-fp8-hmma-sm121.nsys-rep`

SHA-256:
`73ac060dd77bd1ae7d9f493f9e1fc80894dfe0a9879af0911456944c8314e4ae`

## Full-model run with chunked KDA (2026-08-10)

The same one-engine four-format comparison, rerun with the chunked KDA
default (`7eb5bec` removed the harness pin that had held KDA on the generic
path — an intermediate rerun with that pin still in place reproduced the
pre-pause numbers, so the surrounding code carries no regression; that log
is preserved as `solar-kv-full2k-generic-rerun.log`). All four formats still
select prefill token 11047 and decode token 27294, and the run passed with
no non-finite logit.

| KV | Prefill | Prefill rate | Prefill vs generic rerun | Decode |
|---|---:|---:|---:|---:|
| BF16 | 10.989448 s | 186.361 tok/s | 13.786961 s | 142.015 ms |
| FP8 | 7.594036 s | **269.685 tok/s** | 12.925570 s | 112.200 ms |
| hybrid | 7.998880 s | 256.036 tok/s | 12.615811 s | 112.189 ms |
| FP4 | 9.003993 s | 227.455 tok/s | 13.737557 s | 111.312 ms |

FP8 prefill drops 12.93 s to 7.59 s (1.70x, +70% throughput); the ~5.3 s
saved matches the microbenchmark's prediction for the KDA share. Decode is
unchanged, as designed — the recurrent decode kernel was not touched. BF16
improves least because the first leg still carries the one-time warmup.

Log:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/solar-kv-full2k-kdachunk.log`

SHA-256:
`abecc853ee91aa1c840be16af4031d357c537b550f8700397a208117ac0beb6c`

FP8 nsys profile:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/nsys-solar-kv-full2k-fp8-hmma-kdachunk-sm121.nsys-rep`

SHA-256:
`88828e7a1c729449655ef07614521d23b67e809bdac1f4b87be76fa1fdfdf37f`

## Current bottleneck from the final nsys capture

The chunked prefill removed KDA from the top of the profile. In the
2026-08-10 FP8 full-model capture (7.697 s of GPU time, chunked KDA
default), the KDA family costs 1.287 s (16.7%) against the generic path's
7.245 s (55.8%) in the previous capture — a 5.63x full-model reduction:

| Rank | Kernel family | GPU time | Share |
|---:|---|---:|---:|
| 1 | IQ2_XXS `mul_mat_q` | 3.261140 s | **42.4%** |
| 2 | Q8_0 `mul_mat_q` | 1.078688 s | **14.0%** |
| 3 | Q3_K routed worklist MMQ | 0.692650 s | 9.0% |
| 4 | KDA chunk factor | 0.576321 s | 7.5% |
| 5 | KDA chunk scan | 0.442324 s | 5.7% |
| 6 | Q4_K routed worklist MMQ | 0.312230 s | 4.1% |
| 7 | `quantize_mmq_q8_1` | 0.164003 s | 2.1% |
| 8 | KDA chunk intra | 0.161572 s | 2.1% |
| - | KDA chunk prep | 0.106621 s | 1.4% |
| - | compressed GQA HMMA prefill | 0.090710 s | 1.2% |

The next optimization order is therefore IQ2_XXS `mul_mat_q` first (the
routed-expert weight format, 42.4%), Q8_0 second, then either the Q3_K
worklist or the documented KDA register-tiling rewrite. Do not retune
compressed attention without a new profile showing it has regressed.

## Experiments intentionally rejected

These were measured, reverted, and never committed:

- Direct half-bit/half-multiply dequantization worsened the three-format HMMA
  unit from about 0.773 ms to 1.126 ms; FP8 alone regressed from about 166.7 us
  to 414 us.
- Staging compressed scales in shared memory worsened the same total from
  about 0.773 ms to 0.989 ms.
- A warp-normalization KDA rewrite preserved output but improved the 8K
  generic path by only 0.15% and the two-part path by 0.69%, below the bar for
  added complexity.
- Token-major chunked-KDA workspace planes ([token][head*dim]) ran the whole
  chunked path 1.4x slower than head-major: every per-chunk row sat 32 KiB
  from the next and fell out of L1.
- Staging the factor kernel's V rows through shared cost more than it saved:
  the extra 1 KiB pushed the block across the two-blocks-per-SM boundary
  (48 ms to 60 ms). Reverted to streaming V from global.

Do not recreate these experiments unless a later kernel changes the premise.

## Upstage vLLM fork review

The relevant Upstage fork point inspected was
`upstage/v0.22.0-solar-open2` at `00907fc9`.

Transferable ideas:

- separate recurrent decode from chunked prefill;
- use a fixed KDA prefill chunk (64 in that implementation);
- fuse gate work around the tensor-core KDA prefill path.

Non-transferable as-is:

- Triton kernels and vLLM scheduling interfaces do not map mechanically onto
  ds4's CUDA-KDA family;
- the checked MoE tuning tables target H100/B200 rather than GB10 `sm_121`;
- the fork is based on an older vLLM line and should be mined selectively, not
  merged wholesale.

## UMA and process-safety notes

GB10 uses one physical memory pool. CPU and GPU accounting must not be added as
if they were separate capacities.

- Use one large model process at a time.
- Long loads and servers belong in detached tmux under
  `/home/sunghoon/workspace/ds4-exaone/scripts/guarded-run.sh`.
- Monitor `/proc/meminfo`, actual model RSS/PSS, cgroup events, and NVIDIA
  process allocation together. CUDA UMA was not fully charged to
  `memory.current` in the final run, so cgroup counters alone are insufficient.
- The final process exited cleanly, but the driver retained memory. At handoff
  capture, `MemAvailable` was about 27.9 GiB with no model process alive.
- The operator owns a `clear_cache` alias in `~/.bashrc` for this known Spark
  bug. **Do not invoke it automatically.** Ask the operator to run it only
  before the next large cold load when reclaim is required.
- Do not reset the driver, reboot, drop caches, or add elaborate cache policy
  workarounds as part of model optimization.

The exact final-run launcher is preserved at:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/run-solar-kv-full2k.sh`

## Resume order

Keep correctness before capacity and capacity before performance.

1. Confirm no `ds4`, `ds4-server`, or full-model test process is alive. Ask the
   operator for `clear_cache` if a new cold load needs recovered UMA.
2. Rebuild `sm_121a` from the pinned branch and rerun the KDA unit/FLA suite,
   now including `tests/test_solar_kda_chunk`.
3. Run production KDA lifecycle gates at 32K and 64K on the release path
   (chunked prefill + recurrent decode). Compare snapshots, replay, fork,
   cancellation, resync, continuation, and final state against the sequence
   path pinned with `DS4_SOLAR_KDA_STATE_PARTS=1`.
4. Run multiple full-model prompts through BF16/FP8/hybrid/FP4. Require finite
   logits, greedy identity where margins are not ties, retrieval correctness,
   and post-prefill continuation. The single 2K top-1 match is only a smoke
   result.
5. Select a candidate runtime KV layout. Current evidence favors investigating
   hybrid first, with FP4 as the maximum-headroom fallback; this is not yet a
   release decision.
6. Execute 128K, 256K, 512K, then exactly 1,048,576 tokens. Record requested,
   admitted, and actual token counts; all memory pools; `MemAvailable`; NVIDIA
   allocation; cold prefill; TTFT; and decode.
7. At exactly 1M, run beginning/middle/end needles, repeated-key and
   multi-needle retrieval, Korean long-context QA, summarization, 64-256 output
   tokens, snapshot/restore, and a second-turn prefix-reuse check.
8. Only after those pass, start `ds4-server --cuda -c 1048576` on
   `0.0.0.0:8001` and validate `/v1/models`, streaming and non-streaming chat,
   long prompts, reuse, cancellation, and malformed requests.
9. Capture nsys at every frontier where the top kernel changes materially.
   The current reference order is IQ2_XXS, Q8_0, Q3_K, then either the Q4_K
   worklist or the KDA register-tiling rewrite.

## Useful commands

Build and short CUDA regressions:

```bash
cd /home/sunghoon/workspace/ds4-exaone/ds4-solar-open2
make -j2 CUDA_ARCH=sm_121 \
  tests/test_solar_kda \
  tests/test_solar_kda_prefill \
  tests/test_solar_kda_chunk \
  tests/test_solar_kda_fla \
  tests/test_solar_kv_formats

./tests/test_solar_kda
./tests/test_solar_kda_prefill
./tests/test_solar_kda_chunk
./tests/test_solar_kda_fla \
  /home/sunghoon/workspace/ds4-exaone/solar-open2-spark-handoff/fixtures/kda/synthetic-fla.bin
```

The KDA A/B microbenchmark (generic vs parts vs chunked, 8K production
shape) lives at
`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/bench-solar-kda-ab.cu`;
the ablation probe used to localize kernel cost without Nsight Compute is
`bench-gb10-kda-micro.cu` beside it.

Full-model KV comparison without nsys:

```bash
DS4_SOLAR_KV_PREFILL_HMMA=1 \
./tests/test_solar_kv_formats \
  /home/sunghoon/workspace/ds4-exaone/models/Solar-Open2-250B-Mixed-Quant-GGUF/Solar-Open2-250B-MXQ-v1-00001-of-00011.gguf \
  /home/sunghoon/workspace/ds4-exaone/solar-open2-spark-handoff/fixtures/long-context/context-131072.txt \
  2048
```

Low-overhead nsys preset:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cudabacktrace=none \
  --resolve-symbols=false \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --kill=none \
  --wait=primary \
  -o PROFILE_PREFIX \
  COMMAND
```

Always call `cudaDeviceSynchronize()` before `cudaProfilerStop()` in a
capture-range harness.

## Acceptance boundary

Do not describe this as a single-DGX-Spark 1M release yet. The defensible
current statement is:

> The fixed 88.97 GiB Solar Open2 mixed-quant GGUF loads as a native
> `sm_121a` CUDA-KDA family with resident weights on one GB10. Short-context
> CUDA correctness, 8K KDA lifecycle behavior, and a 2K four-format KV
> comparison pass. The chunked-KDA prefill is the default production path,
> validated against the official FLA fixture at the generic path's error and
> raising 2K FP8 prefill from 159 to 270 tok/s. Exact-1M resident admission,
> long-context semantics, the 32K/64K lifecycle gates on the chunked path,
> and API serving remain unverified.
