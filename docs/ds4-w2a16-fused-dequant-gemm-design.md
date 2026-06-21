# Design sketch — differentiable W2A16 fused-dequant grouped GEMM (ds4 GSQ trainer)

**Status:** design scope, not yet implemented. **Author context:** the profiled rung-4 training step is GPU-compute-bound (99% util); ~**50%** of it is the 2-bit→bf16 expert dequant (HBM write of the materialized weights), **doubled** because activation checkpointing re-runs the forward in the backward. The MoE matmul is only ~12%. So the lever that matters is *not* the matmul — it's eliminating the bf16 weight materialization. This sketches a fused kernel that streams the frozen 2-bit codes, dequants in-register, does the grouped GEMM, and **never writes the bf16 weight to HBM** — while staying differentiable to the trainable scales `s`.

## 1. Why this is tractable (the key insight)

The student weight for one block (256 weights) is (`ds4_gpu_dequant.materialize_blocks`):

```
W = (d0 · exp(s)) · unit + bias
```

where, per ggml format:
- **IQ2_XXS (tt=16, w1/w3 gate&up):** `unit = gm_expanded · sv`, `sv = grid[gidx]·ksign[sidx]` (int8 codewords from the 256×8 lattice grid + 128×8 sign table), `gm = (0.5+ls)·0.25` (per-group fp16 scale, 8 groups/block), `bias = 0`.
- **Q2_K (tt=10, w2/down):** `unit = sc_expanded · q`, `q ∈ {0..3}` (2-bit), `sc` (4-bit sub-scale, 16 sub-blocks), `bias = −(dmin·m)` (16 sub-mins).

Everything except `s` is **frozen** (derived from the immutable codes). `s` (per-block log-scale, shape `[nb]`) enters **only** as the scalar multiplier `d = d0·exp(s)`. Therefore:

```
∂W_ij/∂s_b  =  d0_b·exp(s_b)·unit_ij  =  W_ij − bias_ij          (for element ij in block b)
```

So the scale-gradient is a **simple per-block scaled reduction** of the weight-gradient — *not* a general QAT straight-through estimator. This is what makes a fused, differentiable kernel feasible:

```
dL/ds_b = Σ_{ij ∈ block b}  (dL/dW)_ij · (W_ij − bias_ij)
```

## 2. What the kernel computes

Per expert `e` (grouped/contiguous layout, sorted-by-expert with per-expert offsets `offs`, reusing the validated `GroupedMoE` routing): `Y_e = Xq_e @ W_e` where `Xq_e` is the fp8-fake-quant activation (`_aq`, preserved — QAT faithfulness) and `W_e` is the dequantized expert weight `[in, out]`. Three GEMMs/layer (gate W1, up W3, down W2), SwiGLU between, exactly as today — but `W_e` is produced **in-register from the codes**, never in HBM.

**Forward (fused dequant-GEMM):** stream code tiles → unpack `unit` + read `d0`, `bias`, apply `d=d0·exp(s)` → `W_tile = d·unit + bias` in registers/shared → `acc += Xq_tile @ W_tile`. Grouped over experts via `offs`. Output `Y` only.

**Backward dX (fused):** `dX_e = dY_e @ W_eᵀ` — re-stream codes, rebuild `W_tile` in-register (same prologue), accumulate. (No saved bf16 weight → no recompute penalty; re-streaming 2-bit codes is ~8× less HBM than re-reading bf16.)

**Backward ds (the novel part):** we need `dL/dW_e = Xq_eᵀ @ dY_e` (the standard weight-grad GEMM, shape `[in,out]`), then the per-block reduction `dL/ds_b = Σ_block (dL/dW)·(W−bias)`. **Fuse the reduction into the dW GEMM epilogue:** as each `dW_tile` is produced, rebuild `(W−bias)_tile = d·unit` in-register (the same unpack, no bias add) and accumulate `Σ dW_tile ⊙ (W−bias)_tile` into the block's `ds` scalar. `dW` itself is never written to HBM — only the reduced `ds` ([nb] per part) leaves. (We do NOT need dW for the codes — they're frozen — so dW exists only transiently to form ds.)

Net HBM per step for the experts: read 2-bit codes (×~3 passes: fwd, dX, dW) + small scales/offsets; write only `Y`, `dX`, and `ds`. The ~13 GB/layer bf16 materialization (×2 for recompute) is **gone**.

## 3. Differentiability wrapper

A `torch.autograd.Function` per GEMM (or one wrapping the expert MLP), reusing `GroupedMoE`'s routing/`offs`:
- `forward(ctx, Xq, codes, d0, unit_meta, s, offs)` → calls the fused fwd kernel; saves `Xq`, `codes/d0/bias/offs`, `s` (small) for backward — **not** `W` (the point).
- `backward(ctx, dY)` → fused dX kernel (→ `dXq`) + fused dW-reduce kernel (→ `ds`); returns grads for `Xq` (→ flows to prior layers / the `_aq` STE) and `s` (→ the trained scales). Codes/d0/bias get `None` (frozen).

Parity target (same bar the grouped-mm prototype hit): grad cosine vs the `materialize_blocks` + `torch._grouped_mm` reference **> 0.9999**, forward reldiff < 1e-2.

## 4. Implementation plan (incremental, validate each phase)

0. **Q2_K forward-only** (simplest unpack: direct 2-bit + 16 sub-scales). Triton grouped GEMM with a Q2_K dequant prologue. Parity vs `materialize_blocks`+`_grouped_mm`; measure HBM + ms vs the materialize path. *(De-risks the hardest unknown — perf of dequant-in-prologue on sm100/sm103 — on the easy format first.)*
1. **IQ2_XXS forward** (the trickier prologue: 256×8 grid + 128×8 sign lookups in shared/constant mem, 8-group scale). Parity + perf.
2. **Backward dX** for both formats. Parity vs autograd `dX`.
3. **Backward ds** — the fused dW-reduce. Parity vs autograd `ds` (the novel, must-verify path). This is the highest-risk numerics step.
4. **Drop-in `FusedGroupedMoE`** (same `__init__`/`forward` as `BatchedMoE`/`GroupedMoE` → same monkeypatch deploy), behind a flag. Full-trainer validation vs round-1's known trajectory (baseline top1 79.4%, step-25 ~82.4%) + step-time measurement.

## 5. Priors to reuse (don't start from scratch)

- **`ds4_gpu_dequant.unpack_iq2_xxs_gpu` / `unpack_q2_k_gpu`** — the exact bit-extraction to port into the Triton prologue (and the bit-exact CPU reference in its `_validate`).
- **axolotl `feat/dsv4-dequant-grouped-experts-v2`** (`marlin_w4a16/`, `cutlass_fp4/grouped.py`, `dequant_grouped.py`): the **structure** of dequant-in-GEMM + the grouped `m_indices`/`offs` contiguous layout + the Triton grouped-GEMM tiling. It's W4A16/NVFP4 (different format, and inference-only — no scale-grad), so not a code port, but the GEMM skeleton + the "16-byte stride / tile-pad" rules transfer. (Marlin/CUTLASS are sm120; we target sm100/sm103, so plain Triton, not their CUDA exts.)
- **`GroupedMoE`** (`ds4_batched_moe_grouped.py`) — the validated routing/sort/`offs` scaffold + the `_aq` placement + the deterministic scatter-back. The fused kernel slots in where its `torch._grouped_mm` calls are.
- **DeepGEMM grouped-contiguous** as a reference for the variable-M grouped layout (sm90/sm100).

## 6. Effort, risk, payoff

- **Effort:** real kernel project — ~1–2 weeks focused. Triton W2A16 grouped GEMM × 2 unpack prologues + the fused ds-reduce backward + autograd wrapper + 5-phase validation.
- **Risks:** (1) dequant-in-prologue perf on sm100/sm103 (Triton, no vendored Marlin ext) — *Phase 0 de-risks first*; (2) the IQ2_XXS grid lookup in-kernel (256×8 codebook in shared/constant mem — bank conflicts); (3) the fused ds-reduce numerics (Phase 3 parity gate); (4) preserving the exact po2 `_aq` fp8 act-quant semantics.
- **Payoff:** targets the ~50% dequant **and** removes the recompute doubling (backward re-streams cheap codes instead of re-materializing bf16). If it lands well, plausibly ~1.7–2× end-to-end step speedup — vs grouped-mm's ~15%. This is the only identified path to a step-change, and it pays back across every remaining GSQ round (and any future 2-bit GSQ model on this rail).

## 7. Decision gate before building

Phase 0 (Q2_K forward-only parity + perf) is the cheap go/no-go: if a Triton dequant-in-prologue GEMM on sm100/sm103 can't beat the `materialize`+`_grouped_mm` path on HBM/wall-time for one part, the whole approach is questioned and we stop at the grouped-mm (~15%) win. Build Phase 0, measure, then decide.
