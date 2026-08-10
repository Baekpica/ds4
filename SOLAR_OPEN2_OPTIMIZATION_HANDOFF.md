# Solar Open2 DGX Spark optimization handoff

Status: **resumed and advanced on 2026-08-10 KST; no model process running at
capture.** Two optimization units landed the same day: the chunked-KDA
prefill became the default production path, and Solar was routed through the
aligned-artifact startup flow, putting its IQ2_XXS gate/up prefill on the
d2r pair kernel. 2K FP8 prefill went 159 -> 270 -> 328 tok/s across the two
changes, the three quantized KV formats now converge at ~6.24 s, and no
kernel family holds more than 20.2% of GPU time.

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
| `980bbe1` | route Solar through the aligned-artifact startup flow |

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

## Register-tiled chunk kernels (landed 2026-08-10, `ce34634`)

The documented register-tiling rewrite landed the same day. The scan keeps
its 128x64 state slice in shared memory and runs O_inter, the U rewrite and
the state fold as register-tiled GEMMs (16x16 thread grid, 4x4/8x4
micro-tiles, float4 operands); the factor kernel's triangular T-products
and the intra output use the same 4-row x 8-column tiling, and the
sub-block dots read 132-float padded tiles as float4. An 8-token prep batch
was tried and reverted (serialized reduction trees cost more than the grid
saved).

8K production-shape microbenchmark, cumulative:

| Path | wall | vs generic |
|---|---:|---:|
| generic sequence kernel | 727.0 ms | 1.00x |
| chunked v1 (landed `0b58219`) | 111.2 ms | 6.69x |
| + register tiling (`ce34634`) | **61.8 ms** | **11.77x** |

Per kernel: scan 37.3 -> 23.0 ms (operand-streaming bound now), factor
48.0 -> 24.9 ms, intra 14.9 -> 4.4 ms, prep 9.7 ms unchanged. Output
max_abs vs generic 5.1e-10; FLA fixture unchanged; memcheck and racecheck
clean. Full-model note: a 2K rerun accidentally measured a stale
`test_solar_kv_formats` binary (identical pre-tiling numbers); the
register-tiled full-model figure was not separately measured before the
session moved to the 1M serving gate — the lifecycle rates below already
include the tiled kernels.

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

## 32K/64K lifecycle gates on the release path (2026-08-10)

`tests/test_solar_lifecycle` ran the full staged regression with the fixed
model, FP8 runtime KV, HMMA compressed prefill, aligned artifacts, chunked
KDA prefill, recurrent decode, and the register-tiled chunk kernels — the
exact release configuration. Every stage passed with bit-stable replay
(`warm_replay=0`), snapshot/restore equality, fork identity, cancellation
and cold-resync checks green:

| Stage | Common prefix | Extension sync | Rate | Snapshot |
|---:|---:|---:|---:|---:|
| 2,048 | 0 | 11.953 s | 171.3 tok/s | 207 MiB / 0.126 s |
| 8,192 | 2,048 | 17.762 s | 345.9 tok/s | 353 MiB / 0.220 s |
| 32,768 | 8,192 | 90.907 s | 270.3 tok/s | 938 MiB / 0.522 s |
| 65,536 | 32,768 | 172.307 s | 190.2 tok/s | 1.678 GiB / 1.047 s |

The rate falls with depth because the 12 full-attention GQA layers pay the
quadratic term; the KDA side is depth-invariant. The 32K/64K lifecycle
frontier required by the resume plan is closed.

## Long-context ladder (2026-08-10, `tests/test_solar_longctx`)

The new gate prefills a rendered needle fixture end to end on the release
configuration (hybrid runtime KV) and requires the marker strings in the
greedy completion. First 131K attempt taught the budget lesson: the
fixtures end in `<|think:start|>`, and the model spent the manifest's
reserved 256 tokens entirely inside a well-formed thinking block that
correctly analyzed the needle structure — treat that as a fixture-budget
constraint, not a retrieval failure; reruns use a 1,536-token budget.

Measured at 131,072 (130,800 prompt tokens, hybrid KV, chunk 2048):

- Cold prefill 816.6 s (160.2 tok/s TTFT at depth); decode 4.17 tok/s
  (240 ms/token — the 12 GQA layers walking 131K rows); context pools
  3.709 GiB; MemAvailable 23.95 GiB at exit; no OOM, no non-finite logits.
- Needle results at the 1,536-token budget: see
  `solar-longctx-131k-hybrid2.log` beside the other run logs.

512K and exact-1M full-depth retrieval remain future work (multi-hour
prefills); the 1M gate below is the serving-admission demonstration.

Log:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/solar-lifecycle-64k.log`

SHA-256:
`560fea55fe908b7a5fc7e0f688b90abd1a1540c3aaff1cb5c32cbf53ea11c497`

## Full-model run with chunked KDA + aligned artifacts (2026-08-10)

| KV | Prefill | Prefill rate | Decode |
|---|---:|---:|---:|
| BF16 (first leg, carries warmup) | 15.638152 s | 130.962 tok/s | 138.362 ms |
| FP8 | 6.244699 s | 327.958 tok/s | 96.950 ms |
| hybrid | 6.242245 s | **328.087 tok/s** | 95.629 ms |
| FP4 | 6.256438 s | 327.343 tok/s | 95.628 ms |

Log:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/solar-kv-full2k-iq2align.log`

SHA-256:
`f0e4dde4092df3d4a94115b7d163d8eeb700ac64642637b7544d4f356cc43c46`

FP8 nsys profile:

`/home/sunghoon/workspace/ds4-exaone/scratch/solar-open2-spark/nsys-solar-kv-full2k-fp8-hmma-iq2align-sm121.nsys-rep`

SHA-256:
`8c7a547eca6c29cdcf278a6a00e34e7e23a55250d4380b3405a3b13f7b24d8f1`

## Aligned-artifact startup for Solar (landed 2026-08-10)

The 42.4% IQ2_XXS `mul_mat_q` bottleneck was not a kernel problem but a
dispatch gap. Solar's gate/up expert stacks (80 tensors, 32.23 GiB, 40
mid layers x 2) ran the rectangular stream-K MoE fallback: 320 experts x
10 row tiles x 128 gathered-column tiles = 409,600 blocks per launch, most
of which exit empty, at 10.6 GB/s of weight throughput against the Q3_K
worklist's 41.6 GB/s. The tuned path — aligned SoA artifacts consumed by
`gateup_iq2_d2r_pair_kernel` — already existed from the K-EXAONE work but
never activated: Solar's loader called the single-file artifact builder,
which bails on an 11-shard GGUF ("model size changed"), and the family
gates on the records builder and expert-remainder promotion said
EXAONE-only. `980bbe1` extends both gates.

Boot now runs: records-driven artifact build (381 artifacts, 39.60 GiB
device, 98.2 s: 80 IQ2 gate/up replace + Q8 dense additive; q2k finds no
candidates), whole-image copy skipped by the existing residency chooser,
model map installed, then the 47.95 GiB Q3_K/Q4_K raw expert remainder
promoted per range with "expert replaces complete". Peak weights resident:
39.60 + 47.95 = 87.55 GiB plus KV/scratch.

Effects on the 2K four-format comparison (all versus the chunked-KDA run):

- IQ2_XXS gate/up family: 3.261 s rectangular to 1.278 s d2r (**2.55x**).
- FP8 prefill 7.59 s -> 6.24 s (270 -> 328 tok/s); hybrid and FP4 converge
  to the same ~6.24 s / ~328 tok/s.
- MoE decode: 112 ms -> 96 ms (SoA vec paths).
- All four formats still select prefill token 11047 and decode token 27294.
- The BF16 leg reads slower (15.6 s) only because it runs first and now
  absorbs the one-time lazy SoA/f16-cache warmup; its decode is unchanged.
- Small regressions elsewhere (Q8 dense 1.08 -> 1.34 s across its two
  paths, Q3_K worklist 0.69 -> 0.84 s, KDA family +0.2 s) are real but
  unexplained: candidates are shared-power clock shifts, L2 pressure from
  the artifact layouts, or per-range versus whole-image locality. Worth a
  profile before chasing.

At 1M-context capacity runs note: the Q8 additive artifacts cost ~7.4 GiB
of the ~40 GiB build; `DS4_CUDA_Q8_NO_ALIGNED=1` drops them if admission
needs the headroom, at the cost of the dense-GEMM d2r paths.

## Current bottleneck from the final nsys capture

In the 2026-08-10 FP8 capture with both changes (6.330 s of GPU time):

| Rank | Kernel family | GPU time | Share |
|---:|---|---:|---:|
| 1 | IQ2 gate/up d2r pair | 1.278300 s | **20.2%** |
| 2 | Q8 dense d2r | 0.873833 s | **13.8%** |
| 3 | Q3_K routed worklist MMQ | 0.843690 s | 13.3% |
| 4 | KDA chunk factor | 0.656101 s | 10.4% |
| 5 | KDA chunk scan | 0.499530 s | 7.9% |
| 6 | Q8_0 `mul_mat_q` | 0.469175 s | 7.4% |
| 7 | Q4_K routed worklist MMQ | 0.369989 s | 5.8% |
| 8 | KDA chunk intra | 0.176887 s | 2.8% |
| - | `mm_ids_helper` | 0.156408 s | 2.5% |
| - | compressed GQA HMMA prefill | 0.124041 s | 2.0% |

No family dominates any more. The remaining prefill levers, in rough
expected-value order: the KDA register-tiling rewrite documented above
(factor+scan+intra = 21.1% together, ~3x headroom measured against
cuBLAS), the unexplained Q8/Q3_K drift above, then d2r itself. Decode and
lifecycle gates matter more than further prefill work from here. Do not
retune compressed attention without a new profile showing it regressed.

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
> `sm_121a` CUDA-KDA family on one GB10, with aligned IQ2/Q8 artifacts plus
> promoted raw expert ranges replacing the whole-image copy. Short-context
> CUDA correctness, 8K KDA lifecycle behavior, and a 2K four-format KV
> comparison pass. The chunked-KDA prefill and the d2r gate/up path are the
> default production configuration, raising 2K quantized-KV prefill from
> 159 to ~328 tok/s in one day's two changes. Exact-1M resident admission,
> long-context semantics, the 32K/64K lifecycle gates on this
> configuration, and API serving remain unverified.
