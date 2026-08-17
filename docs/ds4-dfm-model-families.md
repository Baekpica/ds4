# ds4-dfm model families on DGX Spark

`ds4-dfm` is the Baekpica release line for serving Korean
**DFM (독자 파운데이션 모델, 독파모)** model families with DwarfStar. It follows
the CUDA serving base in [Entrpi/ds4](https://github.com/Entrpi/ds4), which in
turn follows [antirez/ds4](https://github.com/antirez/ds4). The versioning rule
is deliberately small: an Entrpi release such as `v0.5.6.3` becomes
`v0.5.6.3-dfm` after the additional model families pass this repository's
integration gates.

The reference target is one NVIDIA DGX Spark with a GB10 GPU and 128 GB of
unified memory. Other operating systems and accelerators are not release
targets for the DFM additions yet.

## Design contract

ds4 is not a general GGUF runtime. A model is accepted only when its GGUF
metadata and tensor layouts match one of the explicit shapes in `ds4.c`.
Adding a family means adding its validator, weight binder, tokenizer and chat
protocol, state lifecycle, and the C/CUDA kernels its topology requires.

The implementation stays close to upstream's style:

- model selection is a small enum and direct switch;
- shared arithmetic reuses the existing CUDA primitives and aligned weight
  artifacts;
- genuinely different attention, recurrent state, or expert math gets a
  direct family path;
- no plugin registry, graph framework, or broad abstraction layer is added;
- DeepSeek MTP and DSpark support models remain DeepSeek-only.

This keeps the changes reviewable for a possible future upstream contribution.

## Integrated families

| Family | Shape selected from | Native state/runtime | Current server lane |
|---|---|---|---|
| DeepSeek V4 Flash | `general.architecture=deepseek4` | Entrpi compressed KV and continuous graph | continuous or serial |
| Solar Open2 250B | `general.architecture=solar-open2` | recurrent KDA state plus compressed GQA KV | persistent multi-bank |
| K-EXAONE 236B A23B | `general.architecture=exaone-moe` | LLLG full/sliding GQA KV | persistent multi-bank |
| Motif-3 | `general.architecture=motif3` | normalized latent KV, rotated `k_pe`, and SWA rings | persistent multi-bank |

The scheduler implementation may differ because the model states differ, but
the operator and client contract is the same. Changing `-m` to a GGUF from a
different supported family selects the corresponding runtime in the same
binary.

## Common serving surface

Every family is served by `ds4-server` and exposes:

| Protocol | Endpoint |
|---|---|
| OpenAI Chat Completions | `/v1/chat/completions` |
| OpenAI Completions | `/v1/completions` |
| OpenAI Responses | `/v1/responses` |
| Anthropic Messages | `/v1/messages` |
| Model discovery | `/v1/models` |
| Runtime state | `/v1/stats` and `/metrics` |

The model-family dispatch covers prompt rendering, generated-message parsing,
tool-call syntax, streaming tails, thinking controls, and generation stop
tokens. `--model-id` sets the `/v1/models` id for every family. When it is
omitted, the server parses the GGUF path: a parent directory ending in
`GGUF` or containing `Mixed-Quant` (the usual artifact bucket) wins,
otherwise the file stem with any `-00001-of-00011` shard suffix removed.
A listening port is not an acceptance result; `/v1/models`, a real
generation request, and settled `/v1/stats` counters must all pass.

## Common disk-KV contract

`--kv-disk-dir` and `--kv-disk-space-mb` use the same server policy for every
integrated family. DeepSeek/GLM keeps its compressed-KV payload, Solar keeps
recurrent KDA plus GQA state, EXAONE keeps its full/sliding LLLG rings, and
Motif-3 keeps normalized latent KV plus rotated `k_pe` rings. Serial sessions
and continuous banks share the family payload format, validate their tagged
layout before any restore, and reject truncated or cross-family data.

```sh
./ds4-server -m "$MODEL" --cuda -c 131072 \
  --kv-disk-dir /path/to/ssd/ds4-kv --kv-disk-space-mb 32768
```

Successful loads remain on disk until the configured space-budget eviction
removes them, so more than one restart can reuse a prefix. The cache is ordinary
SSD persistence, not active-bank offload: context length and concurrency must
still fit unified memory before the worker starts. Its quant identity comes from
the first populated routed-expert layer, including dense-first model families.

## Weight owner and inference worker

On a 128 GB unified-memory machine, keep one weight owner alive and restart
only inference workers while developing or profiling. The owner maps split
GGUFs as one logical model, uploads VMM ranges, builds byte-neutral aligned
IQ2/Q2K expert artifacts, and brokers POSIX file descriptors to workers.

Start with a dry run:

```sh
MODEL=/path/to/model.gguf
RUN=/path/to/run-directory

./ds4_weight_server \
  --base "$MODEL" \
  --manifest "$RUN/weights.manifest" \
  --backend vmm \
  --scope base \
  --reserve-gb 24 \
  --no-repack-q8-aligned \
  --dry-run
```

If the memory preflight passes, run the same command without `--dry-run` in a
durable tmux session. Do not start a worker until the owner reports both
`broker listening` and `ready manifest=...`.

The worker command is common to all four families:

```sh
DS4_CUDA_WEIGHT_IPC_MANIFEST="$RUN/weights.manifest" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
./ds4-server -m "$MODEL" --cuda -c 2048 \
  --host 127.0.0.1 --port 8001 --no-update-check
```

For a split model, `MODEL` is its first shard. DeepSeek can place a DSpark
drafter beside the base model; the standard launch resolver attaches it
automatically when its expected file name is present. The other families do
not accept MTP or DSpark attachments.

## DGX Spark memory hygiene

Before changing large models:

1. Check the compute-process view in `nvtop` and the process/RSS view in
   `btop` or `htop`.
2. Stop the inference worker and confirm its PID and listening port are gone.
3. Stop the weight owner and confirm its PID is gone and `nvtop` lists no
   remaining compute process.
4. Run `/usr/local/bin/clear_cache` only after those processes have exited.
5. Recheck `nvtop`, `btop` or `htop`, `free -h`, and swap before starting the
   next owner.

`clear_cache` does not reclaim allocations from a live CUDA process. Never run
a second full-model owner beside the first one on the reference machine.

## Integration evidence for `v0.5.6.3-dfm`

The following production GGUF integration gates were run on the same GB10
host and release line with a 2,048-token development context. The Motif row
also includes the later strict long-context gate documented below:

| Family | Weight-owner evidence | Server evidence |
|---|---|---|
| DeepSeek V4 Flash | 80.76 GiB base plus 6.49 GiB DSpark; 72.56 GiB aligned artifacts | detected DSpark automatically; one Chat request completed with zero failures |
| Solar Open2 250B | 11 shards, 88.97 GiB; 32.23 GiB aligned IQ2 artifacts | two persistent banks; two concurrent Chat requests completed on the continuous route |
| K-EXAONE 236B A23B | 3 shards, 85.56 GiB; 30.16 GiB aligned IQ2 artifacts | two persistent banks; two concurrent Chat requests completed on the continuous route |
| Motif-3 | 94,162,541,472-byte canonical GGUF; current owner exports 7.00 GiB raw plus 80.68 GiB in 153 aligned expert artifacts | all four API surfaces, strict 262,080-token prompt plus decode, and three concurrent 196K-context banks passed |

The Motif artifact is 94.16 GB, or 87.6957 GiB; 87.70 is its binary GiB size,
not its decimal GB size. The current owner exported 207 VMM ranges: 54 raw
ranges plus 153 aligned Q2_K and IQ2_XXS expert artifacts. The worker imported
those ranges without a duplicate model copy.

## Motif-3 DGX Spark performance evidence

The 32K/256K HTTP gates used
[`593d251`](https://github.com/Baekpica/ds4/commit/593d2511a10694f5a33fbafbd997ca24e819a853).
The 8K `ds4-bench` throughput below used
[`cc2f277`](https://github.com/Baekpica/ds4/commit/cc2f27712482318aef4d83c30f59974739166990)
(FATTN occupancy: drop the Q shared tile so three CTAs fit on GB10), built with CUDA 13.3
as `sm_121a` on one DGX Spark GB10 running driver 610.43.02 and Linux
6.17.0-1029-nvidia. The server used the production MQ87-88 artifact, the VMM
owner above, a 4,096-token prefill chunk, greedy sampling, no thinking, no
speculation, and one request at a time.

| Gate | Interface | Prompt | Prefill | Decode | Correctness |
|---|---|---:|---:|---:|---|
| 8K | `ds4-bench` | 8,192 | 519.55 tok/s | 64 tokens at 12.28 tok/s | throughput fixture; prefill-only 519.55, decode-run 516.17 / 12.28 |
| 32K | OpenAI Chat | 32,768 | 82.649 s; 396.47 tok/s | 43 in 4.799 s; 8.96 tok/s | beginning, middle, and end sentinels exact |
| 256K | OpenAI Chat, `-c 262144` | 262,080 | 1,492.375 s; 175.61 tok/s | 43 in 17.072 s; 2.52 tok/s | all sentinels exact; `finish_reason=stop`; 262,123 total tokens |

The two HTTP gates were non-streaming, so they do not provide an independent
network-visible time-to-first-message measurement. The table reports the
server's prompt-complete and decode timings and makes no separate TTFM claim.

The 256K session reported 4,422,546,432 bytes (4.119 GiB) of latent KV and
rotated-key payload. Including the default 4,096-token execution graph, its
physical worker allocation was 9.703 GiB. Source-GGUF mapping RSS remained
29,632 KiB after inference, and engine shutdown left 637,251,584 bytes of CUDA
module/driver state, below the 896 MiB lifecycle gate. During the full request,
the worker and owner both remained at `VmSwap: 0`; system memory retained about
12 GiB available. Loaded clock samples remained between 2,398 and 2,411 MHz,
so the earlier 611 MHz pin did not recur.

Nsight Systems on the final 32K prefill ranked aggregate CUDA kernel time as
expanded FATTN 15.5%, paired Q8 projection 11.0%, latent attention 9.7%, BF16
rounding 8.5%, W_UV value projection 7.5%, routed gate/up 7.2%, and QK absorb
4.1%. Focused 4,096-row Nsight Compute runs measured:

| Kernel | Before | Final | Reduction |
|---|---:|---:|---:|
| expanded FATTN | 55.79 ms | 28.83 ms | 48.3% |
| Motif group-5 QK absorb | 38.91 ms | 10.97 ms | 71.8% |

The final strict gate JSON and server log were retained with SHA-256
`b8551d5c96a0bdc1b6244275b79a5ac9ac9f8932862a93f6256ff51df00d7a9f` and
`90b064268bcc31498e653d27fcf5087064910bf9a6963f4fe239dc295b0fbeda`,
respectively.

### Motif-3 196K multi-bank serving evidence

The persistent-bank extension at `03b7002` and `cf605e0` was built as
`sm_121a` and run with `-c 196608`, three banks, an 8,192-token prefill chunk,
and `--no-spec`. The explicit 6 GiB batch-fit headroom left the measured
configuration at three banks instead of the conservative default reducing it
to two.

| Gate | Result |
|---|---|
| API surface | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and `/v1/messages` each returned HTTP 200 with the native response shape |
| 8K cold prefill | 8,214 prompt tokens at 266.3 tok/s; `LONG_OK` returned exactly |
| Single decode | 192 output tokens; 490.4 ms TTFT, 12.9 tok/s decode, 15.350 s HTTP wall time |
| Three simultaneous Chat requests | 192 output tokens each in 24.885--25.030 s; 23.01 aggregate output tok/s; server log `served=3 fallback=0` |

After the gates, `/v1/stats` reported 11 completed requests, zero failures,
zero serial requests, zero continuous-batch failures, three total and zero
live banks, and zero speculative drafts. The VMM owner used 90,119 MiB, the
worker used 22,283 MiB after the 8K requests, and the system retained about
6.5 GiB available without an OOM event. Loaded SM clock remained 2,411 MHz;
the 611 MHz pin did not recur.

## Current limits

- Motif-3 has three verified persistent banks at `-c 196608` on the reference
  Spark. Concurrent 256K banks are not claimed.
- The Motif-3 256K result validates one strict serial request on this exact
  artifact and GB10 host. It does not validate concurrent 256K banks or other
  accelerators.
- Motif-3 serving uses plain decoding; MTP and DSpark support models remain
  DeepSeek-only.
- Solar, EXAONE, and Motif-3 serial snapshots now reject corrupted family tags,
  and their continuous banks restore into a different idle bank before a
  one-token warm suffix. The CUDA lifecycle gates passed on the production
  mixed-quant GGUFs; DeepSeek/GLM retains the existing compressed-KV format.
- The 2026-08-15 Motif restart gate persisted a 738-token bank as 43.82 MiB,
  then restored all 738 cached tokens in 33.1 ms and computed only the
  24-token suffix. The cache file remained after the successful load.
- Disk KV reduces repeated prefill across eviction or restart. It does not lower
  the resident KV allocation of a live bank; the Motif Spark operating point
  remains two banks at `-c 196608` with a 4,096-token prefill chunk.
- Model cards contain only verified behavior and performance. Profiling
  results, failed experiments, and proposed kernels belong in the technical
  handoff until a release gate validates them.

## Profiling order

Do not use a 256K run as the first performance experiment. Establish a short
correctness baseline, profile an 8K or 16K prefill and a separate decode
window with Nsight Systems, then use Nsight Compute only on kernels that rank
as material bottlenecks. Keep one change per measurement and require both
the focused fixture and a full-model A/B before changing the default path.
