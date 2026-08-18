# dots3-note DGX Spark 포팅 handoff

기준 시각: **2026-08-18 11:43 KST**

상태 요약: **로더·토크나이저·서버 프로토콜·CUDA latent 그래프·CPU
레퍼런스와 실가중치 레지던트 계산 게이트가 통과했다. aligned-aware MoE,
CPU↔GPU forward (`cos=0.997536`, argmax 3925), 1600-token 청크/링 및 prefix
재사용(`cache_cos=1.0`), DSA 2600 결정성(argmax 151721), 256K resident
그래프 생성까지 확인했다.** 524,288-token 실제 prefill/decode와 continuous
다중 bank, MTP 실행은 여전히 미검증/미구현이다.

## 0. 다음 작업 시작점

1. **owner부터 확인한다.** tmux `sunghoon-0:dots3-owner`, PID 693024,
   nvidia-smi 82399 MiB가 이 세션 종료 시점의 정상 상태다.
   ```sh
   pgrep -af ds4_weight_server
   grep 'ready manifest' scratch/dots3/run-20260818/logs/owner.log
   ```
   owner가 살아있는 동안 `clear_cache`를 실행하지 않는다.
2. 정합을 다시 건드리기 전 `resident7.log`의 전체 게이트를 기준선으로
   사용한다. 청크만 재현할 때는
   `tests/test_dots3_resident <shard1> --chunk-only`를 쓴다.
3. 다음 큰 조각은 524,288-token 실제 prefill/decode 또는 dots3 continuous
   bank다. 둘 다 현재 릴리스 주장의 일부가 아니다. 최적화는 §7의
   nsys→ncu 순서를 따른다.

## 1. 현재 git/파일 상태

- 워크트리: `ds4-model-families-v0563/` @ `feature/model-families-v0563`,
  포팅 베이스 `13f72c4` (= 작업 시작 당시 origin/dfm). 배포 대상은
  `origin/dfm`; 로컬 `dfm` 브랜치는 stale이라 사용하지 않는다.
- 커밋:
  - `0659b0a` feat(dots3): loader, dual-geometry validation, qwen2 tokenizer
  - `0c4013d` feat(dots3): latent MLA graph, DSA indexer, serial sessions,
    server protocol
  - `2c84f99` wip(dots3): aligned expert consumption + Spark porting handoff
    (`dots3_graph_ffn` routed 경로를
    `ds4_gpu_routed_moe_batch_tensor`(clamp=0.0f)로 교체 + `routed_out`).
- 이 문서를 포함하는 최종 수정: CPU 레퍼런스 힙 오버플로 수정,
  prefix tail replay, 레지던트 테스트의 청크 전용 모드와 lazy 256K sync,
  CUDA graph-pool cleanup trim, family 문서 동기화.
- **CPU 레퍼런스 수정 (`ds4.c`)**:
  `nw`를 `n_embd`(5120), dequant scratch를
  `max(n_ff_dense, n_head*n_value_mla)`=16384, `kv_proj`를
  `n_head*(nope+v)`=32768로 확대. `dots3_ref_dequant_row`에
  `out_n` 한도 가드(`ds4_die`).
- 워크스페이스 루트 `CLAUDE.md`는 dfm/dots3 현황으로 갱신 완료 (git 밖).
- 최종 수정 파일: `ds4.c`, `ds4_cuda.cu`,
  `tests/test_dots3_resident.c`, 이 문서,
  `docs/ds4-dfm-model-families.md`.

## 2. 통과한 검증 / 실기에서 새로 확인된 것

| 게이트 | 결과 |
|---|---|
| `make test-dots3-loader` | 통과 (10-shard 80.16 GiB, 956 텐서 + 이중기하) |
| `make test-dots3-tokenizer` | 통과 — HF 골든 15 + stop {151643,151668} |
| `./ds4_test --server` | 통과 (dots3 렌더/파서 포함) |
| `make cuda-regression test-model-family-kernels` | 통과 |
| Solar/Motif loader+tokenizer 회귀 | 통과 |
| cubin 아치 | `sm_121a` |
| owner dry-run/기동 | 통과 — 184 ranges, `--no-repack-q8-aligned`, reserve 24 GiB |
| VMM import / no-duplicate | 통과 — worker CUDA delta 0.51–0.72 GiB (owner 80.5 GiB 대비 미복제) |
| aligned IQ2/Q2K MoE 경로 | 통과 — boot prewarm 512 tok / 약 20s, fused gate+up, M2 Q2K down. 1차 illegal access는 이 경로로 해소 |
| CPU↔GPU forward gate | **통과** — 21토큰, `cos(gpu,ref)=0.997536` `cos(one,split)=0.999356` argmax 3925/3925/3925 |
| serial 512 세션 | **통과** — `dots3 no-think decode: OK` |
| 1600-token 청크/링 패리티 | **통과** — argmax 63594/63594, `batch_cos=0.99920081`, `cache_cos=1.0`, `cache_nrmse=0` |
| 2600-token DSA 결정성 | **통과** — argmax 151721/151721 |
| 256K 캐시 / 그래프 | **통과** — lazy sync에서 resident KV 5.714 GiB 계획을 실제 생성 |
| CUDA lifecycle | **통과** — `resident7.log` cleanup remainder 0 bytes, 전체 `EXIT=0` |
| 4K server e2e | **통과** — `/v1/models`, Chat HTTP 200 `OK`, stats completed 1 / failed 0 / inflight 0 |

### 1차 실패 (illegal access) — 해소됨

owner는 IQ2_XXS/Q2_K expert를 aligned 아티팩트로 REPLACE하고 raw 범위를
업로드에서 제외한다. EXAONE 체인(`ds4_gpu_routed_iq2_q3_*`)은 raw mmap을
GPU가 읽어 폴트. DeepSeek `ds4_gpu_routed_moe_batch_tensor`로 교체
(`2c84f99`). 로그: `resident.log`, `dbg1.log`(`ds4_mmq_q2_K_moe`).

### 2차 실패 (munmap_chunk) — 해소됨

`resident2.log` / `resident2-munmap-20260818.log`: aligned 경로 진입 후
CPU 레퍼런스 ~8분 → `munmap_chunk(): invalid pointer` EXIT=134.
원인: 레퍼런스 스크래치가 full-attention 폭보다 작음.

| 버퍼 | 구 크기 | 실제 필요 | 덮어쓰인 텐서 |
|---|---|---|---|
| `nw` | 1024 (`n_swa_kv_lora`) | 5120 | `attn_norm` / `ffn_norm` / `output_norm` |
| `scratch` | 13824 (`n_ff_dense`) | 16384 | full `attn_output` in_dim |
| `kv_proj` | 24576 (`n_head*n_key_mla`) | 32768 | full `attn_kv_b` out |

`resident3`는 이 수정 바이너리로 레퍼런스를 끝까지 돌렸고
`dots3 CPU reference done` 후 forward gate 숫자를 남겼다.

### 3차 실패 (청크/링) — 해소됨

공식 vLLM의 SWA 범위는 `min(seq_len, query_len + 512)`이고 로컬의 513
링 수학도 일치했다. 불일치는 attention이 아니라 prefix 700에서 재개할
때 마지막 60-token 조각이 cold 320-token 경로와 다른 quantized-MoE batch
tier를 타는 실행 스케줄 문제였다. prefix가 prefill chunk 경계가 아니면
마지막 partial chunk까지만 되감아 다시 계산한다. SWA 링은
`window + prefill_cap`을 보존하므로 이 replay 범위를 수용한다.

### 4차 실패 (256K/cleanup 테스트) — 해소됨

lazy graph인 세션을 `create`만 하고 메모리를 재던 테스트를 실제 21-token
`sync`로 바꿨다. `cudaMemGetInfo`의 물리 delta는 그래프 풀 재사용에 따라
0–3.45 GiB로 달라질 수 있으므로 성공 판정은 graph pending 해제와 sync
성공을 쓴다. cleanup은 captured graph를 파기한 뒤 모든 sticky scratch와
stream을 먼저 해제하고 마지막에 `cudaDeviceGraphMemTrim`한다.

## 3. dots3-note 수학 스펙 (재유도 불필요 — 이 섹션이 결론)

근거: transformers PR #47844 공식 modeling
(`miraclezqc/transformers@9daa8668`, 로컬 사본
`scratch/dots3/official-modeling/`), handoff의 BF16 imatrix forward,
config 기본값(`n_group=1, topk_group=1` → 그룹 라우팅 무효,
`use_dsa=True`, `norm_topk_prob=True`, `routed_scaling_factor=1.0`,
`moe_gating_fp32=False`, `k_rope_only_layernorm=True`).

- 레이어: 47 블록 = 46 텍스트 + blk.46 MTP. full-attention =
  `il==0 || il%4==1` (13개: 0,1,5,...,45). 나머지 33 + MTP = SWA.
- 기하: full 128헤드/q_lora 1024/kv_lora 512/nope 128/rope 64/v 128/θ 8e7,
  scale 192^-0.5. SWA 64헤드/kv_lora 1024/nope 192/rope 64/v 128/θ 5e4,
  scale 256^-0.5, window 513(자기+512).
- **LoRA rescale**: q_a_norm 뒤 ×√(5120/1024), kv_a_norm 뒤 ×√(5120/kv_lora).
  구현은 dequant한 norm 벡터(GGUF에 Q8_0로 저장됨!)에 접어서 디바이스
  버퍼로 업로드(`dots3_graph_upload_norm`). 인덱서 q 입력(q_lora)도 동일
  rescale 적용값을 공유하므로 접기가 유효함(공식 코드에서 확인).
- k_pe: kv_a_mqa 꼬리 64차원 → `attn_k_rope_norm` RMS → **GPT-J interleaved
  RoPE**(짝 2f,2f+1; 다른 패밀리의 NeoX와 다름!). q도 각 헤드 꼬리 64에
  interleaved.
- 출력 게이트: `sigmoid(attn_gate(normed_hidden))` per-head, o_proj 전 곱.
- MoE: sigmoid(router F32) → top-8 by (p+bias) → w = p/(Σp **+1e-20**)
  (전용 라우터 커널; EXAONE 커널의 6.1e-5 플로어와 다름), shared expert
  무가중 합산, SwiGLU, scaling 1.0.
- DSA 인덱서 (full 레이어만): q = idx_q_b(q_lora) 64헤드×128,
  k = LayerNorm_w,b(idx_k(normed_hidden)) 128 — **rope는 앞 64차원**
  (main과 반대), θ는 full의 8e7. k는 BF16 라운드 후, q는 fp32에서 바로
  **FP8-E4M3 라운드트립**(블록 128, scale=clamp(amax,1e-4)/448, SATFINITE).
  score = Σ_h w_h·relu(q_h·k), w = idx_w(h) × 128^-0.5 × 64^-0.5.
  top-2048은 공용 `ds4_gpu_indexer_topk_tensor` (legacy 모드 `(0,UINT32_MAX)`).
  **end ≤ 2048이면 선택=전체 인과구간과 수학적으로 동일**하므로 스킵 —
  이 등가 구간이 CPU 레퍼런스 정합 게이트의 근거. 1600토큰 청크 테스트는
  이 등가 구간에 있다(인덱서 스킵이 맞음).
- 캐시: latent+k_pe BF16 (full: ctx행, swa: 513+prefill_chunk 링),
  idx key F32 13레이어 (512K에서 layer당 256 MB). MTP는 캐시 슬롯 없음,
  실행 안 함(바인딩+검증만; `token_embd_mtp`/eh_proj/enorm/hnorm/
  shared_head_norm은 nextn_* 필드).
- 실행 형태: prefill/decode 모두 absorbed-latent 단일 경로
  (`dots3_graph_forward_chunk`). absorb/value-project는 motif의 제네릭
  Q8_0 엔트리 재사용(group=1); latent attention은 전용 커널
  (latent_dim/128 그룹, window 또는 selected 리스트 walk).
- 독립 정합 앵커: `ds4_engine_dots3_reference_logits` — 공식 수식 그대로의
  FP32 CPU forward (dequant 가중치, ≤256토큰). 게이트
  `ds4_engine_dots3_forward_test`: cos(gpu,ref)>0.99 + argmax 일치 +
  원샷/분할 cos>0.999. **21토큰 실가중치에서 통과(위 표).**

## 4. 실행 인프라 상태

- **weight owner (실행 중, 유지)**: tmux `sunghoon-0:dots3-owner`
  PID 693024, 82399 MiB.
  ```sh
  ./ds4_weight_server --base ../models/dots3-note-prev-Mixed-Quant-GGUF/dots3-note-prev-MQ87-00001-of-00010.gguf \
    --manifest $RUN/weights.manifest --backend vmm --scope base \
    --reserve-gb 24 --no-repack-q8-aligned
  # RUN=/home/sunghoon/workspace/ds4-exaone/scratch/dots3/run-20260818
  ```
  로그 `$RUN/logs/owner.log`. aligned-Q8은 의도적으로 OFF(메모리 우선;
  prefill 최적화 단계에서 재검토 — motif의 발표 수치는 aligned-Q8 필요였음).
- 레지던트 worker는 각 gate 뒤 종료하고 owner만 유지한다. owner가
  살아있는 동안 `clear_cache` 하지 말 것.
- 로그:
  - `resident.log` — 1차 illegal access (raw Q2K / latent attn)
  - `dbg1.log` — `CUDA_LAUNCH_BLOCKING=1`로 `ds4_mmq_q2_K_moe` 특정
  - `resident2.log`, `resident2-prev.log`,
    `resident2-munmap-20260818.log` — munmap (CPU-ref 오버플로)
  - `resident3.log` — 버퍼 수정본. forward gate 통과 + 청크/링 실패 +
    DSA 시작 직후 중단
  - `chunk-rewind.log` — partial-chunk replay 수정의 표적 통과
  - `resident4.log` — 계산 게이트 통과; lazy 256K/cleanup 테스트 결함 발견
  - `resident5.log` — 실제 256K sync와 cleanup trim 통과, 물리 delta 판정 결함
  - `resident6.log` — 계산/256K 통과; cleanup trim 시점의 비결정성 재현
  - `resident7.log` — scratch/stream 해제 뒤 trim하는 최종 전체 게이트
- HF 정리: 잘못 만들어진 **모델 repo** `Baekpica/dots3-note-prev-spark-handoff`
  삭제 완료. **버킷**(동명)은 무손상.
- tmux: `dots3-owner` 유지. 완료된 테스트/서버 worker는 종료한다.
  `motif-owner-8002` / `motif-server-8002`는 빈 bash(프로세스 없음).
- scratch 도구: `scratch/dots3/`에 gguf_inspect.py(+kv/tensors 덤프),
  gen_goldens.py(+venv: tokenizers/jinja2), template-renders.json,
  official-modeling/, src-aux/.

## 5. 서빙 e2e — 4K 통과

```sh
RUN=.../run-20260818
DS4_CUDA_WEIGHT_IPC_MANIFEST=$RUN/weights.manifest DS4_CUDA_WEIGHT_IPC_SCOPE=base \
../scripts/guarded-run.sh -m 112 -l $RUN/logs/server-8003.log -- \
./ds4-server -m ../models/dots3-note-prev-Mixed-Quant-GGUF/dots3-note-prev-MQ87-00001-of-00010.gguf \
  --cuda -c 4096 --host 0.0.0.0 --port 8003 --model-id dots3-note-prev \
  --no-spec --no-update-check --mem-floor-gb 8
```

- `server-8003.log`: `/v1/models`가 `dots3-note-prev`, context 4096을 반환.
- OpenAI Chat: 21-token prompt, HTTP 200, content `OK`, `finish_reason=stop`,
  completion 1 token.
- `/v1/stats`: started/completed `1/1`, failed `0`, inflight `0`, serial `1`,
  continuous `0`, imported artifacts 135.
- 확인 뒤 server worker와 port 8003은 종료했고 owner PID 693024만 유지했다.

다음 long-context 사다리는 131072 → 262144 → **524288**다. 각 단계에서
`/v1/models` + 실제 생성 + `/v1/stats` 안정까지가 준비 기준이며 리슨
포트나 캐시 할당만으로 prefill 통과를 주장하지 않는다.

- 524288은 **정확히 524,288-token prefill + 실제 decode 토큰**이 나와야
  통과 주장 가능. 메모리 주의: 512K에서 idx_scores 스크래치
  (128행×ctx×4B=256 MB) + 캐시 ~11.4 GiB + cap 4096 스크래치 ~4.7 GiB;
  부족하면 `DS4_DOTS3_PREFILL_CHUNK=2048`로 낮추고 owner `--reserve-gb`
  조정. dots3는 **serial 전용**(`supports_batching=false`) — continuous
  bank는 후속(§8).

## 6. 문서/배포 범위

1. Git은 이 handoff, family 문서, `ds4.c`, `ds4_cuda.cu`, resident 테스트만
   명시적으로 stage하고 `git push origin HEAD:dfm`한다. 로컬 stale
   `dfm` 브랜치는 사용하지 않는다.
2. HF는 `Baekpica/dots3-note-prev-Mixed-Quant-GGUF`의 `README.md`만 별도
   allowlist 디렉터리에서 올린다. remote 재다운로드 `cmp`와 tree OID
   비교로 README 외 경로가 불변인지 확인한다.
3. 모델카드는 text-only serial ds4와 위 4K/256K 범위만 말한다. 한국 DFM
   설명이나 524K 실제 실행, continuous, MTP 실행 주장은 넣지 않는다.

## 7. 최적화 루프 (서빙 후 상시 작업 — 사용자 요청)

- 프로파일링 순서(dfm 문서 규약): 8–16K prefill + 별도 decode 창을
  **nsys** → 상위 커널만 **ncu**(현재 `RmProfilingAdminOnly: 0`으로 사용
  가능, /usr/local/cuda-13.3/bin). 측정당 변경 1개, fixture+풀모델 A/B.
- 예상 1순위 후보: ① `dots3_latent_attention_kernel`(경합/점유율 미조정;
  full 레이어 decode에 motif처럼 split-K 변형 필요할 것) ② `dots3_idx_score`
  (블록당 32KB smem, 나이브 dot; 512K decode에서 layer당 ~4.3 GFLOP)
  ③ value_project가 raw Q8 폴백(motif式 transposed 아티팩트를 dots3 형상
  [512,32768]/[1024,20480]에 추가하면 개선) ④ aligned-Q8 owner 재검토.

## 8. 열린 경계 / 알려진 한계 (정직하게 유지할 것)

- **DSA >2048 정합**: 공식 활성값 fixture가 없어 기계 검증은
  ≤2048(dense 등가) 구간의 CPU 레퍼런스 게이트까지. >2048은 공식 수식
  대조 구현 + 동일 입력 2회 결정성 스모크(레지던트 테스트 내 2600-token)
  까지이며 argmax 151721/151721로 통과했다.
- NFC 정규화 미적용(다른 byte-BPE 패밀리와 동일 정책; 입력은 NFC 가정).
- MTP(blk.46) 미실행(바인딩/검증만). 외부 MTP/DSpark attachment는
  DeepSeek 전용이고 dots3의 in-file MTP 실행은 후속이다.
- continuous 다중 뱅크 미구현(§5) — family_banked shim에 dots3 훅 추가가
  다음 큰 조각.
- toolless thinking visible-prefix 숏컷은 dots3 미구현(정확 토큰 replay는
  동작).
- `routed_moe_batch` decode 티어(q81-fused)는 top-6 전용 분기가 있어
  top-8은 다른 vec 티어로 떨어진다. 정합에는 문제 없지만 decode 최적화
  후보로 남는다.
- 라우터를 F32 가중치×f32 activ로 계산(공식 기본은 BF16 게이팅) — 실질
  무영향으로 판단, 기록만.

## 9. 참조 위치

- 스펙 원장: `dots3-note-prev-spark-handoff/manifests/{architecture.json,
  quant-recipe.yaml}` + `reproduction/converter/convert_dots3_gguf.py`
  (GGUF 키/텐서명 전수) + `reproduction/calibration/run_bf16_imatrix.py`
  (공식 수식). GGUF 덤프: `scratch/dots3/gguf-kv.txt`, `gguf-tensors.tsv`.
- 아티팩트: 10샤드 80.156 GiB, `MQ87-SHA256SUMS`/`MQ87-VERIFY.json`
  (956텐서, min cosine 0.9407). 병합 단일본은 로컬에 없음 — v0.6.0-dfm은
  split 직접 서빙 지원하므로 필요 없음(샤드1이 -m 인자).
- 소스 핀: `dots-studio/dots3-note-prev` @
  `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`, Apache-2.0.
- 공식 서빙 구현 대조: vLLM native dots3-note merge `9035151d6c9f`,
  재확인한 main `cdb8545a91be`; `vllm/models/dots3_note/nvidia/attention.py`
  의 SWA metadata와 window 범위를 기준으로 비교했다.
