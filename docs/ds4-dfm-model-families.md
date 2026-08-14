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
| Motif-3 | `general.architecture=motif3` | normalized latent KV, rotated `k_pe`, and SWA rings | serial native session |

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
tokens. A listening port is not an acceptance result; `/v1/models`, a real
generation request, and settled `/v1/stats` counters must all pass.

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

The following production GGUF gates were run on the same GB10 host with the
same `ds4-server` binary and a 2,048-token development context:

| Family | Weight-owner evidence | Server evidence |
|---|---|---|
| DeepSeek V4 Flash | 80.76 GiB base plus 6.49 GiB DSpark; 72.56 GiB aligned artifacts | detected DSpark automatically; one Chat request completed with zero failures |
| Solar Open2 250B | 11 shards, 88.97 GiB; 32.23 GiB aligned IQ2 artifacts | two persistent banks; two concurrent Chat requests completed on the continuous route |
| K-EXAONE 236B A23B | 3 shards, 85.56 GiB; 30.16 GiB aligned IQ2 artifacts | two persistent banks; two concurrent Chat requests completed on the continuous route |
| Motif-3 | 87.70 GiB; 80.68 GiB aligned IQ2/Q2K artifacts | native session loaded; Chat, Completions, Responses, and Messages each completed with zero request failures |

The Motif aligned-artifact gate also completed a real prefill and decode after
confirming the 384-expert IQ2/Q2K path was active. These are integration and
kernel-lifecycle gates, not published quality or long-context performance
claims.

## Current limits

- Motif-3 does not yet have a native persistent multi-bank runtime, so it uses
  the safe serial session lane behind the same APIs.
- Motif-3 has not yet completed a strict 262,144-token prompt followed by an
  actual decode token on DGX Spark. It must not be described as a Spark 256K
  pass until that run completes.
- EXAONE durable bank payload serialization is intentionally unavailable; its
  fixed banks support exact-frontier in-process reuse and fork copies.
- Model cards contain only verified behavior and performance. Profiling
  results, failed experiments, and proposed kernels belong in the technical
  handoff until a release gate validates them.

## Profiling order

Do not use a 256K run as the first performance experiment. Establish a short
correctness baseline, profile an 8K or 16K prefill and a separate decode
window with Nsight Systems, then use Nsight Compute only on kernels that rank
as material bottlenecks. Keep one change per measurement and require both
the focused fixture and a full-model A/B before changing the default path.
