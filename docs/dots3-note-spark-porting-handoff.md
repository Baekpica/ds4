# dots3-note DGX Spark 포팅 handoff

기준 시각: **2026-08-18 09:35 KST경** (세션 사용량 소진으로 중단)

상태 요약: **dots3-note 패밀리를 dfm 라인
(`ds4-model-families-v0563/`, branch `feature/model-families-v0563`,
publish 대상 `origin/dfm`)에 추가하는 작업이 로더·토크나이저·서버 프로토콜·
CUDA latent 그래프·CPU 레퍼런스까지 구현 완료, 유닛/구조 테스트 전부 통과.
실가중치 레지던트 게이트는 1차 시도에서 illegal access(원인 규명 완료:
raw Q2_K가 미고정 호스트 mmap에서 GPU로 읽힘) → aligned-aware MoE 경로로
수정 적용, 재빌드 rc=0까지 완료. 재실행 직전에 중단됨.**
**Spark에서 dots3 정합/서빙/524288 통과 주장은 아직 금지다.**

## 0. 다음 세션 재개 순서 (이 순서 그대로)

1. **owner 생존 확인.** dots3 VMM weight owner가 tmux `sunghoon-0` 세션의
   `dots3-owner` 윈도우에서 **아직 실행 중**이어야 정상 (~85 GiB 사용).
   ```sh
   pgrep -af ds4_weight_server    # --base ...dots3-note-prev-MQ87-00001... 이어야 함
   grep 'ready manifest' scratch/dots3/run-20260818/logs/owner.log
   ```
   죽어 있으면 §4의 명령으로 재기동(dry-run 먼저). owner가 살아있는 동안
   `clear_cache` 금지. 메모리 작업 전 nvtop+htop 확인 (사용자 지시 절차).
2. **레지던트 게이트 재실행** (직전에 하려던 바로 그 명령):
   ```sh
   cd ds4-model-families-v0563
   RUN=/home/sunghoon/workspace/ds4-exaone/scratch/dots3/run-20260818
   tmux new-window -t sunghoon-0 -n dots3-t2 -d \
     "DS4_CUDA_WEIGHT_IPC_MANIFEST=$RUN/weights.manifest \
      DS4_CUDA_WEIGHT_IPC_SCOPE=base CUDA_VISIBLE_DEVICES=0 \
      ../scripts/guarded-run.sh -m 112 -l $RUN/logs/resident2.log -- \
      ./tests/test_dots3_resident \
      ../models/dots3-note-prev-Mixed-Quant-GGUF/dots3-note-prev-MQ87-00001-of-00010.gguf; \
      echo EXIT=\$? >> $RUN/logs/resident2.log"
   ```
   실패 시 `CUDA_LAUNCH_BLOCKING=1`을 앞에 붙여 재실행하면 정확한 커널이
   특정된다(1차 디버깅 때 그렇게 `ds4_mmq_q2_K_moe`를 잡았음).
   테스트 내부 단계: VMM import/no-duplicate → CPU 레퍼런스 vs CUDA forward
   게이트(레퍼런스는 싱글스레드라 수 분 소요; `dots3 forward gate:` 라인의
   cos/argmax 확인) → serial 세션 decode → 1600-token 청크/링 패리티 →
   2600-token DSA 경계 결정성 → 256K 캐시 할당 → 종료 잔류 검사.
3. 게이트 통과 시 **커밋** (미커밋 fix가 이미 워킹트리에 있음, §2 참고):
   `git add ds4.c && git commit` — "fix(dots3): consume aligned IQ2/Q2K
   expert artifacts via the DeepSeek routed executor" 취지로.
4. **서빙 e2e** (§5), 이후 모델카드/HF/push (§6), nsys/ncu (§7).

## 1. 현재 git/파일 상태

- 워크트리: `ds4-model-families-v0563/` @ `feature/model-families-v0563`,
  베이스 `13f72c4` (= origin/dfm). **push는 아직 안 함** (게이트 전 push 금지).
- 신규 커밋 3개:
  - `0659b0a` feat(dots3): loader, dual-geometry validation, qwen2 tokenizer
  - `0c4013d` feat(dots3): latent MLA graph, DSA indexer, serial sessions,
    server protocol
  - (이 문서와 함께 커밋) fix(dots3): `dots3_graph_ffn`의 routed 경로를
    `ds4_gpu_routed_iq2_q3_handoff/gate_up/bounded` 체인에서
    **`ds4_gpu_routed_moe_batch_tensor`** (DeepSeek의 aligned-aware 실행기,
    clamp=0.0f) 호출로 교체 + `routed_out` 버퍼 추가. 재빌드 rc=0 확인,
    **하드웨어 검증은 §0-2가 해야 함 — 이 커밋은 게이트 미통과 상태의
    체크포인트이므로 push 금지.**
- 워크스페이스 루트 `CLAUDE.md`는 dfm/dots3 현황으로 갱신 완료 (git 밖 파일).
- 터치한 파일: `ds4.c`(shape/validator/binder/tokenizer/chat/reference/
  graph/session/payload), `ds4.h`(dots3 test decls), `ds4_gpu.h`(dots3 엔트리
  선언), `ds4_cuda.cu`(dots3 커널 섹션, motif differential 엔트리 뒤),
  `ds4_server.c`(SYNTAX_DOTS3 전체), `Makefile`(DS4_DOTS3_MODEL,
  test-dots3-{loader,tokenizer,resident} 타깃), `tests/test_dots3_loader.c`,
  `tests/test_dots3_tokenizer.c`, `tests/dots3_tokenizer_goldens.inc`,
  `tests/test_dots3_resident.c`.

## 2. 통과한 검증 (이 바이너리에서 재현 가능)

| 게이트 | 결과 |
|---|---|
| `DS4_DOTS3_MODEL=<shard1> make test-dots3-loader` | 통과 (10-shard 80.16 GiB 실아티팩트, 956 텐서 바인딩+레이아웃+이중기하 검증) |
| `make test-dots3-tokenizer` | 통과 — HF `tokenizers` 골든 15케이스 + 빌더 패리티 + stop set {151643,151668} + 라운드트립 |
| `./ds4_test --server` | 통과 (dots3 렌더/파서 유닛테스트 포함; jinja2 ground truth 대비) |
| Solar/Motif loader+tokenizer 회귀 | 통과 (무영향 확인) |
| cubin 아치 | `sm_121a` 확인 (`cuobjdump ds4_cuda.o`) |
| owner dry-run/기동 | 통과 — 184 ranges (raw 49 + aligned IQ2 90 43.51 GiB + Q2K 45 27.69 GiB), `--no-repack-q8-aligned`, reserve 24 GiB |
| 레지던트 게이트 | **미통과** — 1차 illegal access 원인 규명·수정 적용, 재실행 대기 |

1차 실패의 교훈(중요): **owner는 IQ2_XXS/Q2_K expert를 aligned 아티팩트로
REPLACE**하고 raw 범위를 업로드에서 제외한다. raw를 읽는 routed 경로
(EXAONE 체인)는 미고정 호스트 mmap을 GPU가 참조하다 폴트. EXAONE이 무사한
이유는 down이 Q3_K(리팩 후보 아님)라 raw가 디바이스에 있기 때문.
IQ2 gate/up + Q2_K down 조합은 DeepSeek 실행기
(`ds4_gpu_routed_moe_batch_tensor`)의 공인 조합이며 aligned 프로브를 갖췄다.

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
  이 등가 구간이 CPU 레퍼런스 정합 게이트의 근거.
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
  원샷/분할 cos>0.999.

## 4. 실행 인프라 상태

- **weight owner (실행 중)**: tmux `sunghoon-0:dots3-owner`
  ```sh
  ./ds4_weight_server --base ../models/dots3-note-prev-Mixed-Quant-GGUF/dots3-note-prev-MQ87-00001-of-00010.gguf \
    --manifest $RUN/weights.manifest --backend vmm --scope base \
    --reserve-gb 24 --no-repack-q8-aligned
  # RUN=/home/sunghoon/workspace/ds4-exaone/scratch/dots3/run-20260818
  ```
  로그 `$RUN/logs/owner.log`. aligned-Q8은 의도적으로 OFF(메모리 우선;
  prefill 최적화 단계에서 재검토 — motif의 발표 수치는 aligned-Q8 필요였음).
- 이전 Motif owner/worker는 세션 초반에 절차대로 종료 + clear_cache 완료.
- HF 정리: 잘못 만들어진 **모델 repo** `Baekpica/dots3-note-prev-spark-handoff`
  삭제 완료(빈 껍데기였음). **버킷**(동명, 54파일 4.2GB)은 무손상 —
  로컬 미러 SHA 52/52 검증됨.
- tmux 잔여 윈도우: `dots3-owner`(유지!), `dots3-test`/`dots3-dbg`(종료된
  1차 시도 — 닫아도 됨).
- scratch 도구: `scratch/dots3/`에 gguf_inspect.py(+kv/tensors 덤프),
  gen_goldens.py(+venv: tokenizers/jinja2), template-renders.json(서버
  렌더러의 바이트 기준), official-modeling/, src-aux/(SHA 3종이 GGUF 기록과
  일치 검증됨: config 99b7de68…, tokenizer 7f4e21a1…, template 2475e840…).

## 5. 서빙 e2e 계획 (레지던트 게이트 통과 후)

```sh
RUN=.../run-20260818
DS4_CUDA_WEIGHT_IPC_MANIFEST=$RUN/weights.manifest DS4_CUDA_WEIGHT_IPC_SCOPE=base \
../scripts/guarded-run.sh -m 112 -l $RUN/logs/server-8003.log -- \
./ds4-server -m ../models/dots3-note-prev-Mixed-Quant-GGUF/dots3-note-prev-MQ87-00001-of-00010.gguf \
  --cuda -c 4096 --host 0.0.0.0 --port 8003 --model-id dots3-note-prev \
  --no-spec --no-update-check --mem-floor-gb 8
```
- ctx 사다리: 4096 → 131072 → 262144 → **524288**. 각 단계에서
  `/v1/models` + 실제 생성 + `/v1/stats` 안정까지가 준비 기준(리슨 포트는
  기준 아님). 벤치마크는 `"thinking":{"type":"disabled"}` +
  `stream_options.include_usage` + 셀별 disjoint corpus.
- 524288은 **정확히 524,288-token prefill + 실제 decode 토큰**이 나와야
  통과 주장 가능. 메모리 주의: 512K에서 idx_scores 스크래치
  (128행×ctx×4B=256 MB) + 캐시 ~11.4 GiB + cap 4096 스크래치 ~4.7 GiB;
  부족하면 `DS4_DOTS3_PREFILL_CHUNK=2048`로 낮추고 owner `--reserve-gb`
  조정. dots3는 **serial 전용**(`supports_batching=false`) — continuous
  bank는 후속(§8).

## 6. 문서/배포 (게이트 후, doc-sync 순서 준수)

1. 이 handoff와 `docs/ds4-dfm-model-families.md`에 검증 사실 반영
   (families 표에 dots3 행: `general.architecture=dots3-note`, serial lane).
2. `models/dots3-note-prev-Mixed-Quant-GGUF/README.md` 갱신: "Native
   Spark / ds4 execution **not yet implemented**" 행을 검증된 사실로 교체
   + ds4 서빙 방법 섹션. 스타일은 Motif-3 카드를 따르되 **독파모(dfm) 설명
   섹션은 넣지 않는다** (사용자 지시: 한국 DFM 모델이 아니고 dfm 브랜치에
   패밀리로 추가될 뿐). 실패/미검증은 카드에 넣지 말고 여기(handoff)에.
3. `git push origin HEAD:dfm` (로컬 `dfm` 브랜치는 96커밋 뒤처진 stale —
   쓰지 말 것).
4. HF: `hf upload Baekpica/dots3-note-prev-Mixed-Quant-GGUF README.md`
   (인증 Baekpica 확인됨) → 재다운로드 byte-compare.
5. 필요 시 CHANGELOG에 dots3 추가 항목.

## 7. 최적화 루프 (서빙 후 상시 작업 — 사용자 요청)

- 프로파일링 순서(dfm 문서 규약): 8–16K prefill + 별도 decode 창을
  **nsys** → 상위 커널만 **ncu**(현재 `RmProfilingAdminOnly: 0`으로 사용
  가능, /usr/local/cuda-13.3/bin). 측정당 변경 1개, fixture+풀모델 A/B.
- 예상 1순위 후보: ① 내 `dots3_latent_attention_kernel`(경합/占유율 미조정;
  full 레이어 decode에 motif처럼 split-K 변형 필요할 것) ② `dots3_idx_score`
  (블록당 32KB smem, 나이브 dot; 512K decode에서 layer당 ~4.3 GFLOP)
  ③ value_project가 raw Q8 폴백(motif式 transposed 아티팩트를 dots3 형상
  [512,32768]/[1024,20480]에 추가하면 개선) ④ aligned-Q8 owner 재검토.

## 8. 열린 경계 / 알려진 한계 (정직하게 유지할 것)

- **DSA >2048 정합**: 공식 활성값 fixture가 없어 기계 검증은
  ≤2048(dense 등가) 구간의 CPU 레퍼런스 게이트까지. >2048은 공식 수식
  대조 구현 + 동일 입력 2회 결정성 스모크(레지던트 테스트 내 2600-token)
  까지만. 모델카드에 성능/정합 주장 시 이 경계를 그대로 서술.
- NFC 정규화 미적용(다른 byte-BPE 패밀리와 동일 정책; 입력은 NFC 가정).
- MTP(blk.46) 미실행(바인딩/검증만) — DeepSeek 외 패밀리 공통 정책.
- continuous 다중 뱅크 미구현(§5) — family_banked shim에 dots3 훅 추가가
  다음 큰 조각.
- toolless thinking visible-prefix 숏컷은 dots3 미구현(정확 토큰 replay는
  동작).
- `routed_moe_batch` decode 티어(q81-fused)는 top-6 전용 분기가 있어
  top-8은 다른 vec 티어로 떨어짐 — 레지던트 게이트의 decode 구간이 이를
  실증해야 함(1차 시도는 그 전에 실패).
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
