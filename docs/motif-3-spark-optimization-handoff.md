# Motif-3 DGX Spark 최적화 작업 handoff

기준 시각: **2026-08-14 12:25 KST** (재개 세션 진행 내역은 §0에 누적;
후속 재개 결과는 §0.15까지 반영)

상태: **§9.1–9.9 완료. 합성 BF16 projection 실패는 current full-copy보다
같은 host 주소의 과거 range가 먼저 선택된 것이 원인이었고, 최소 resolver
수정과 결정적 회귀 테스트로 해결했다(§0.12). CPU/server/Motif fixture,
cuda-spark, synthetic long-context CUDA regression까지 통과했다. 코드 리뷰에서
발견한 resident/long test의 v0.5.6.3 API 및 구식 full-copy 수명주기 충돌도
VMM owner/import 계약으로 포트했다(§0.13). 전체 diff/warning review와 관련
회귀 재실행도 완료했다(§0.14). 원본 네 commit의 v0.5.6.3 이식은
`64c67a0`까지 완료했다(§0.15). 다음 단계는 메모리 preflight 뒤 VMM owner
dry-run과 2K 실모델 gate다. 모델 프로세스·weight owner·서버는 아직
실행하지 않았다.**

## 0. 재개 세션 진행 내역 (2026-08-14, §9 순서 수행)

§9의 재개 순서를 그대로 수행했다. 완료 항목과 검증 증거:

1. **§9.1 상태 재검증** — `CHERRY_PICK_HEAD=0a360db`, staged 14 files,
   `git diff --check --cached` 통과, 워킹트리 클린 확인.
2. **§9.2 GLM 잔재 제거** — `parse_glm_generated_message_ex`와 전용 헬퍼
   `trim_const_span` 삭제(호출부 0 확인). GLM은 추후 별도 family로 재도입한다.
3. **§9.3 syntax 소스 연결** — 4개 파서(chat/anthropic/responses/completion)의
   `request_init` 직후 `r->model_syntax = server_model_syntax_for_engine(e)`.
4. **§9.4 render/parse dispatch 연결** —
   - chat 렌더 3곳 → `render_chat_prompt_text_for_syntax`;
   - `parse_completion_request`에 Motif 분기(기본 system+user 턴을 공식
     템플릿 렌더러로 통과, chat 경로와 byte-identical);
   - live tool tail 2곳(responses/anthropic) → `_for_syntax`;
   - `parse_generated_message_for_response`에 syntax 인자 추가, 실제 4곳 +
     테스트 2곳 호출부 갱신;
   - DSML 절단 repair 2곳(직렬 generate_job, cont_resolve_chat)을
     `SERVER_MODEL_SYNTAX_DEEPSEEK` 전용으로 게이트(Motif JSON 절단은
     기존 parse-failure recovery 경로 사용);
   - 스트림 마커: `find_any_tool_start/end` 후보에 `<tool_call>`/`</tool_call>`
     싱귤러 추가, `tool_marker_stream_safe_len` hold-back에 `<tool_call>` 추가.
     upstream이 이미 bare `<tool_calls>`를 관대한 스펠링으로 취급하는 철학과
     동일하며, 수신 브랜치와 같은 방식이다.
   - 스트리밍 tool 흐름 구조 검증: OpenAI/Responses/Anthropic 세 머신 모두
     Motif `<tool_call>`에서 DSML tool-stream init이 명시적으로 실패 →
     SUPPRESS(와이어 차단) → finalize에서 파싱된 완전한 calls 일괄 방출.
     부분 마커 hold-back은 후행 `<` 80바이트 보수적 hold로 문법 무관 커버.
5. **§9.5 stop/think/tool 종료 조건** —
   - 엔진: `vocab_token_is_generation_stop`이 Motif 공식 generation_config의
     eos [0,3,6](endoftext/user/endofturn)을 커버(기수신 코드).
   - 직렬 경로: `server_token_ends_generation` 헬퍼 신설 —
     `ds4_token_is_stop`(DeepSeek은 정확히 EOS와 동일해 동작 보존) +
     Motif 한정 no-think think-control stop. sample/speculative 두 지점 교체.
   - 연속 경로: 엔진 cont core의 `hit_eos` 4곳(seed, MTP accept 2곳, plain
     decode)을 `beos[b] >= 0` 게이트 하에 vocab stop으로 확장 —
     `req.eos = -1` 결정론 full-budget 계약 보존, DeepSeek 동작 불변.
     서버 `cont_on_token`에 Motif no-think think-control abort(누산기 도달
     전 차단, verdict "stop" 선설정), `cont_needs_text`에 Motif no-think
     행 포함(엔진 raw 버퍼 대신 누산기에서 finalize).
   - 배치/직렬 응답 writer: `detok_result_until_eos`가 family stop에서 절단.
   - CLI: eos 비교 7곳 → `ds4_token_is_stop`(DeepSeek 불변).
6. **§9.6 양자화기** — `repack_fp4_weight_mxfp4`를 수신 브랜치에서 그대로
   이식(`byte_buf` typedef 뒤로 배치). `gguf-tools/deepseek4-quantize` 빌드
   통과(-Wall -Wextra 무경고).
7. **§9.7 CPU guard** — **클린 v0.5.6.3 base에서 동일 오류 재현 확인**
   (scratch/ds4-v0563-baseline). upstream 회귀로 판정하고 upstream 스텁
   스타일로 최소 패치 3건:
   payload-region walker 2개를 `#ifndef DS4_NO_GPU`로 격리(호출 4곳 모두
   GPU 분기 내 확인), 기존 41051 스텁 섹션에 cont/batch API 3개 추가,
   기존 31886 스텁 섹션에 dspark validator 4개 추가. upstream PR 후보.
8. **§9.8 빌드·테스트** —
   - `make cpu`: 5개 바이너리 링크 성공. 후속 강제 재검증에서 확인한
     `DS4_NO_GPU` 전용 Motif/reference helper unused 경고는 §0.14에서
     명시적 `DS4_MAYBE_UNUSED`로 정리했다. 최종적으로 upstream baseline
     경고만 남고 Motif 신규 경고는 0건이다.
   - 서버 유닛(-DDS4_SERVER_TEST): **ok** (신규 Motif 렌더/파서/체크포인트
     3테스트 포함 전체 통과).
   - `make test-motif3-reference`: router/PolyNorm/mHC/expanded GDLA 통과.
   - `make test-motif3-loader`(STRUCTURAL GGUF): 토폴로지/양자 정책/MTP
     바인더 통과. 테스트를 v0.5.6.3의 2-인자 `weights_bind`에 맞게 수정.
   - `make test-motif3-tokenizer`: 16 raw + 5 rendered fixture 통과.
   - `make test`: eval extractor 자체 테스트 통과 후 모델 의존 구간이
     `ds4flash.gguf` 부재로 중단 — DeepSeek GGUF가 없는 이 호스트의 환경
     한계이며 baseline도 동일(회귀 아님).
9. **참조 라인 심볼 이식**(사용자 지시: Solar/EXAONE/Motif 계열은 GLM과
   달리 제거가 아니라 참조 디렉터리에서 정의를 이식) —
   - `ds4_gpu_embed_tokens_q8_0_tensor` + 커널을 ds4_cuda.cu/ds4_gpu.h에
     이식, EXAONE 라인의 교훈대로 invalid-ID 가드 추가(무효 id는 0 임베드,
     유효 id는 동일 수치 경로);
   - `motif3_q8_0_dot_row_dev`(수신 `glm_q8_0_dot_row_dev`와 동일 수식,
     GLM 접두어 제거) 및 `cuda_use_mmq()` → v0.5.6.3의
     `ds4_cuda_use_mmq()` 개명 반영;
   - `matvec_expert_down`을 실제 아티팩트 타입(IQ2_XXS/Q2_K)으로 좁혀
     이식(그 외 타입은 명시적 die);
   - `ds4_gpu.h`에 `extern "C"` C++ 가드 추가(수신 헤더와 동일),
     `test_motif3_cuda.cu`의 다중GPU 헤더(`ds4_gpu_mgpu.h`) include 제거
     (다중GPU 계층은 이식 제외 원칙, 테스트는 해당 심볼 미사용 확인).
10. **워크스페이스 정리** — §6의 비교용 scratch worktree 2개
    (`ds4-motif-port-solarbase`, `ds4-motif-merge-v0563`)를 abort 후
    worktree/branch 제거. baseline 비교용 `scratch/ds4-v0563-baseline`은
    §9.9 디버깅에 계속 쓰이므로 유지한다. 워크스페이스 루트 `CLAUDE.md`는
    3개 모델 라인·worktree 구조·현행 호스트 사실(driver 610.43.02,
    CUDA 13.3, clear_cache 절차)로 재작성했다(리포 외부 파일).
    참고: 모든 엔진 디렉터리는 `ds4/.git` 하나의 linked worktree다
    (`git worktree list`로 관리; branch/stash/remote 공유).

11. **§9.9 최초 cuda-spark/CUDA 시도에서 발견한 실패
    (후속 §0.12에서 해결)** —
    - `make cuda-spark` 성공(EXIT=0, 경고만). 빌드 로그:
      `scratch/motif3-v0563-cuda-spark-build-20260814.log`.
    - `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent`,
      `ds4_weight_server` 6개 바이너리 생성. `cuobjdump` 기준
      `ds4-server`·`ds4_weight_server`의 cubin arch는 **`sm_121a` 단일**
      (다른 arch 0건).
    - `make test-motif3-cuda` (GB10 실행): **6개 그룹 중 5개 통과** —
      router, PolyNorm, mHC(fixture), expanded GDLA, latent GDLA.
      latent GDLA는 자체 합성 model-map 설치(1번째 설치) 경로까지 통과.
    - **실패 1건**: 마지막 `test_bf16_projection` —
      `CUDA BF16 mHC projection mismatch at 0: got 1.2538113e+38
      want -0.00664997101` (index 0부터 garbage 크기의 값 → 수치 오차가
      아니라 잘못된 메모리 읽기).
    - 수집한 사실:
      (a) 이 테스트는 **2번째** `ds4_gpu_set_model_map` 설치
      (`DS4_CUDA_COPY_MODEL=1` 강제 device copy, copy 로그 2회 확인) 후
      `ds4_gpu_matmul_bf16_tensor`(offset 0)를 호출한다. 1번째 설치를
      쓰는 latent GDLA가 통과하므로 **map 재설치/치환 경로** 또는 BF16
      GEMM 경로가 용의선상이다.
      (b) `ds4_gpu_matmul_bf16_tensor`는 f32→bf16 변환 커널을
      `ds4_current_stream()`(캡처 밖에서는 stream 0)에 올린 뒤
      `cublasGemmEx(g_cublas, ...)`를 부른다. v0.5.6.3에는
      `cublasSetStream` 호출이 전무하고, upstream 자체 f16 GEMM 경로는
      GemmEx 직전 반드시 `cuda_cublas_ws_prep(ds4_current_stream())`를
      부르는데 **Motif BF16 경로에는 이 호출이 없다** — upstream cuBLAS
      사용 규약과의 첫 번째 구조적 차이.
      (c) 전역 cuBLAS kill switch는 없다(attention 한정
      `DS4_CUDA_NO_CUBLAS_ATTENTION`류만 존재).
    - 당시 정한 격리 순서:
      1) `ds4_gpu_matmul_bf16_tensor`의 cublas 분기를 일시 우회(코드에서
         `g_cublas_ready` 조건 임시 false)해 native
         `matmul_bf16_kernel` 경로로 같은 테스트를 재실행 — 통과하면
         cuBLAS 호출 규약(ws_prep/stream) 문제, 실패하면 weight-pointer
         해석(2번째 map 설치) 문제로 양분된다.
      2) 전자면 `cuda_cublas_ws_prep` 도입(upstream f16 경로와 동형)
         후 재검증. 후자면 `cuda_model_range_ptr`가 2번째 설치에서
         돌려주는 포인터와 device copy base를 printf로 대조하고
         `cuda_model_preserve_current_direct_mapping()`/파생 레지스트리의
         잔존 상태를 본다.
      3) 수정 후 `make test-motif3-cuda` 전체 재실행, 이어서 §9.10
         (cherry-pick --continue)으로 진행한다.

12. **§9.9 후속 재개: BF16 root cause 수정과 전체 CUDA gate 통과** —
    - 원래 명령 `make test-motif3-cuda`로 index 0의 동일 garbage 값을 다시
      재현했다. cuBLAS 분기를 임시로 끄고 native `matmul_bf16_kernel`로
      실행해도 값이 동일하게 깨져 workspace/stream 가설을 기각하고
      weight resolver로 범위를 좁혔다. 임시 우회 코드는 즉시 제거했다.
    - 진단 출력에서 첫 map과 둘째 map의 host 주소가 모두
      `0xc43453d0a4f0`였고 크기만 `2,228,224` → `524,288` bytes로 달랐다.
      allocator가 주소를 재사용한 뒤 `g_model_range_by_offset[0]`의 과거
      full device copy가 host 주소 일치만으로 현재 map보다 먼저 선택됐다.
    - 수정은 `cuda_model_range_ptr()`의 시작에서 **현재
      `g_model_device_owned` whole-model copy를 과거 range보다 우선**하도록
      한 경계 검사다. registered mapping, VMM import, explicit derived range의
      기존 우선순위는 바꾸지 않았다. 새 추상화나 영구 diagnostic flag는
      추가하지 않았다.
    - `test_bf16_projection`은 같은 host buffer에 서로 다른 크기와 내용을
      연속 설치하도록 바꿔 주소 재사용을 결정적으로 재현한다. 수정 전
      `mismatch at 12`, 수정 후 Motif CUDA 6개 그룹 전체가 3회 연속 통과했고,
      full `cuda-spark` 재빌드 뒤에도 다시 통과했다.
    - 추가로 `make cuda-regression`의 stale API 인자 오류를 clean
      v0.5.6.3 baseline에서도 동일 재현했다. legacy-safe 인자
      (`n_comp_max=0`, no substrate/FP8 mirror)를 test call site에만 보충했고,
      top-k 32,768×32 및 compressed-attention overflow synthetic regression이
      full rebuild 전후 모두 통과했다. 이 test drift도 upstream PR 후보다.
    - 최종 재검증:
      - `make cpu`: 5 binaries 링크 성공;
      - `./ds4_test --server`: 전체 server unit **OK**;
      - Motif reference, STRUCTURAL loader, tokenizer(16 raw + 5 rendered),
        quantizer build: 모두 통과;
      - `make test`: extractor self-test 통과, 이후 기존과 동일하게
        `ds4flash.gguf` 부재에서만 중단;
      - `make cuda-spark`: 6 binaries 성공;
      - `cuobjdump --list-elf`: `ds4-server`와 `ds4_weight_server`의 모든
        cubin이 **`sm_121a` 단일**.
    - 대형 모델/weight owner/server는 실행하지 않았다. 재개 시작 시
      121 GiB 중 118 GiB available, swap 0이었고 `clear_cache`가 필요한
      teardown도 없었다.

13. **§9.10 리뷰 중 발견한 resident/long test 포트 충돌 해결** —
    - 아직 컴파일하지 않았던 `test_motif3_resident.c`와
      `test_motif3_long.c`가 v0.5.6.3에서 제거된 `ds4_engine_options` 필드
      (`context_size`, `placement_ctx_hint`, `prefill_chunk`)와 존재하지 않는
      introspection 함수 두 개를 사용하고 있었다. link target에도 존재하지
      않는 `ds4_gpu_args.o`가 남아 있었다.
    - 더 중요한 안전 문제로 두 테스트가 재부팅 전 OOM 경로인
      `DS4_CUDA_COPY_MODEL_CHUNKED=1`을 강제로 다시 켰다. 두 테스트 모두
      **live `DS4_CUDA_WEIGHT_IPC_MANIFEST`가 없으면 CUDA/model open 전에
      exit 2**한다. worker에서는 `COPY_MODEL`/`COPY_MODEL_CHUNKED`와 legacy
      direct/cache/preload 변수를 명시적으로 unset하고 IPC scope를 `base`로
      고정해 stale shell 환경이 self-load나 MTP-only 경로를 택하지 못한다.
    - resident gate의 의미를 “worker 자체가 모델 크기만큼 CUDA memory를
      중복 소유”에서 “기존 VMM owner range를 import하고 worker 추가 할당이
      모델 크기보다 작음”으로 고쳤다. 고정 GGUF 크기는 `stat`, embedding
      width는 현행 `hidden_f32_values / n_hc` 공개 API로 검증한다.
    - obsolete `ds4_gpu_args.o` dependency를 두 Make target에서 제거했다.
      두 test binary 모두 compile/link 통과했고, 빈 manifest 환경에서 각각
      의도한 안전 거부(exit 2)를 확인했다. **실모델 gate는 아직 실행하지
      않았으며 VMM owner가 ready인 뒤에만 실행한다.**

14. **§9.10 최종 diff/warning review와 회귀 재실행** —
    - newer parent에서 우연히 딸려온 비-Motif Makefile 변수/phony/clean 항목,
      호출점이 0인 GLM 전용 pretokenizer, 호출점이 0인 구형 범용
      `weights_layer_has_required` wrapper와 불필요 helper를 제거했다. 모델
      family 구현은 기존 enum/switch와 직접 호출 구조를 유지하며 새
      registry/plugin 추상화는 추가하지 않았다.
    - CPU 전용 빌드에서만 호출점이 사라지는 Motif reference/diagnostic 함수
      5개에는 기존 코드베이스의 `DS4_MAYBE_UNUSED` 표기만 적용했다. router
      임시 배열도 0 초기화해 실제 신규 `maybe-uninitialized` 경고를 없앴다.
      강제 `ds4_cpu.o`/`ds4.o` 재컴파일 결과 Motif 신규 경고는 0건이고,
      남은 경고는 clean v0.5.6.3 baseline과 동일하다.
    - 최종 재실행 결과: `./ds4_test --server` 전체 OK, 실제 94,162,541,472-byte
      GGUF의 53-layer topology loader 통과, official tokenizer/chat fixture
      16 raw + 5 rendered 통과, reference fixture 통과, Motif CUDA 6그룹 통과,
      synthetic long-context CUDA regression 통과, quantizer build 통과,
      `git diff --check` 통과.
    - resident와 long binary를 현행 CUDA 객체로 다시 링크했다. long에는 실제
      32K token fixture까지 전달한 상태에서 live manifest 부재가 model open
      전에 정확히 exit 2임을 확인했다. 따라서 이번 검증 중 full GGUF device
      load나 모델 프로세스 생성은 없었다.

15. **원본 commit sequence 완료와 문서/API 충돌 해소** —
    - `01e1be0` Add native Motif-3 loader and latent CUDA runtime
    - `b085b1c` Document resident Motif-3 serving
    - `ae0d3a2` Advertise Motif model ID from OpenAI server
    - `64c67a0` Preserve complete question in Motif 256K gate
    - README 충돌에서는 v0.5.6.3에서 제거된 `ds4_help.c`를 복원하지 않았고,
      존재하지 않는 `--model/--ctx/--prefill-chunk/--batched-session` 예시를
      현행 `-m/-c`와 VMM owner/import 명령으로 교체했다. 실모델 API·성능·
      256K는 fixture 결과와 분리해 미검증 release gate로 명시했다.
    - `/v1/models`는 여러 alias를 선택 가능한 모델처럼 나열하지 않고
      `server_model_id_from_engine()`으로 실제 로드된 family 하나만 광고하는
      v0.5.6.3 동작을 보존했다. DeepSeek 두 ID와 Motif 두 ID의 alias 단위
      테스트를 추가했고 전체 server unit이 다시 통과했다.
    - 마지막 commit의 v2 fixture 질문 tail 25-token 보존을 적용한 뒤 long
      binary를 재링크했다. 262,144-token source fixture를 전달한 상태에서도
      live manifest가 없으면 model open 전에 정확히 exit 2임을 재확인했다.
      branch는 `upstream/batched-serving` 대비 ahead 4이고 소스 tracked
      worktree는 clean이다(생성 바이너리만 untracked).

남은 경계(사실로 기록):
- 엔진 **static batch core**(`ds4_batch_generate_core`)는 여전히 per-seq
  eos 단일 비교 — 순수 코어에 vocab 접근이 없어 보류. 서버측 detok 절단이
  응답 청결성은 보장하나, Motif 배치 행은 endofturn 이후 budget까지 초과
  decode할 수 있다(실험적 surface, /v1/batch).
- tool 종료 의미론은 수신 브랜치·DSML 계약과 동일하게 **턴당 1 tool
  round**(첫 블록 close에서 finish=tool_calls). Motif가 한 턴에 여러
  `<tool_call>`을 직렬 방출하는 케이스는 연속 턴으로 처리된다.
- `make test`의 모델 의존 구간은 DeepSeek GGUF 확보 전까지 실행 불가.
- 현재 `test_motif3_long`은 262,144-token **source fixture**에서 filler 64개를
  제거해 262,080 prompt + 최대 64 decode를 native 262,144 window에 넣는다.
  이는 full-window semantic/decode gate지만 §3의 더 엄격한 “262,144 prompt
  prefill 뒤 decode”와는 다르다. 현행 session은 `prompt.len >= ctx_size`를
  거부하고 Motif native cap도 262,144이므로, 최종 주장 전에 이 경계를
  의도적으로 해결하거나 strict gate 미충족으로 남겨야 한다.

## 1. 최종 목표

Motif-3를 별도 제품이나 별도 서버로 만들지 않고 `Baekpica/ds4`의 명시적인 모델 패밀리로 편입한다. 최종 사용 경험은 모델 경로 외에는 동일해야 한다.

```bash
./ds4-server -m /path/to/model.gguf -c 262144 --host 0.0.0.0 --port 8001
```

Solar, Motif, EXAONE, GLM, DeepSeek 사이에서 바뀌는 것은 우선 GGUF 경로이며, 다음 항목은 공통이어야 한다.

- GGUF 메타데이터 기반 모델 패밀리 자동 감지
- 동일한 `ds4-server` CLI와 실패 방식
- 동일한 별도 weight-owner + 재시작 가능한 inference worker 수명주기
- 동일한 포트, 로그, `/v1/models`, `/v1/stats`, `/metrics` 관측 방식
- OpenAI `/v1/chat/completions`
- OpenAI `/v1/completions`
- OpenAI `/v1/responses`
- Anthropic `/v1/messages`
- 각 API의 streaming/non-streaming, usage, error, reasoning, tool-call round trip

모델별로 필요한 것은 enum/switch, loader, graph, kernel, prompt/stop protocol처럼 명시적인 작은 코드다. 플러그인 프레임워크나 범용 추상화 계층을 새로 만들지 않는다. 이 방향은 추후 `antirez/ds4` 업스트림 PR 가능성을 고려한 것이다. 현 단계에서는 다른 OS나 GPU를 미리 일반화하지 않고 이 DGX Spark/GB10을 우선한다.

## 2. 절대 고정 입력

| 항목 | 값 |
|---|---|
| Mixed-Quant 모델 | `Baekpica/Motif-3-Mixed-Quant-GGUF` |
| Mixed-Quant weight revision | `efd6044e25e7f8e3b459a737d021091e2e69b6c6` |
| 양자화 | `MQ87-88-FIT`, 87.6957 GiB, 11 shards |
| 병합 GGUF | `/home/sunghoon/workspace/ds4-exaone/models/Motif-3-Mixed-Quant-GGUF/Motif-3-MQ87-88-FIT.gguf` |
| 병합 GGUF 크기 | `94,162,541,472` bytes |
| 병합 GGUF SHA-256 | `15755a735753bc1396e5ffa539e65a779a4fd769e8833360a4d743c4c60c2f25` |
| Q8_0 reference revision | `5c266c95bf8c8d822d50e5e1cce9d108eaadb2af` |
| 재현 저장소 | `f2b86cf286bd99fdb245e07ceba5710e297dca2f` |
| 수신 ds4 Motif 브랜치 | `d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9` |
| 공식 모델 소스 | `Motif-Technologies/Motif-3@ccceb1a5fd7b5eb32e47841216b3caf5666c07bc` |
| 공식 vLLM 참고 구현 | `MotifTechnologies/vllm@4cd9eb4129883565e69d508038d783d59ee01867` |
| private handoff | `hf://buckets/Baekpica/motif-3-spark-handoff` |

로컬 private handoff는 145 files, 2,465,794,154 bytes다. 원격 전체 재다운로드 후 `manifests/SHA256SUMS`의 144개 항목이 모두 통과했고 private 상태 및 중복 GGUF 0개를 확인했다.

고정 revision의 weight를 Spark 작업 중 재양자화하거나 조용히 교체하지 않는다. 11개 원본 shard도 보존한다.

## 3. 검증 주장 경계

H200에서는 2K, 32K, 64K, 128K correctness가 통과했다. 256K 두 번의 시도는 각각 245,760 및 106,496 prompt token에서 멈췄고 decode가 수행되지 않았다.

따라서 다음 문장은 금지한다.

- H200 256K 통과
- Spark 256K 통과
- 256K serving 확인

Spark에서 정확히 262,144-token prefill을 완료하고 실제 decode token을 얻은 뒤에만 256K 통과로 기록한다. 포트 listen이나 prefill-only는 통과가 아니다.

## 4. 현재 호스트와 메모리 상태

| 항목 | 2026-08-14 10:00 KST 현재 값 |
|---|---|
| Host | `thinkstationpgx-8abc` |
| Kernel | `6.17.0-1029-nvidia` |
| GPU | NVIDIA GB10, compute capability 12.1 |
| Native build target | `sm_121a` |
| NVIDIA driver | `610.43.02` |
| CUDA compiler | 13.3 |
| Unified memory | 121 GiB total, 119 GiB available |
| Swap | 15 GiB total, 0 used |
| Active model/weight server | 없음 |
| tmux monitoring | `sunghoon` group에 `btop`과 `nvtop` pane 유지 |

재부팅 전 journal에는 NVIDIA `Out of memory [NV_ERR_NO_MEMORY]`가 남았다. 이전 Motif 경로는 파생 Q2 약 31.38 GiB, IQ2 약 49.31 GiB, Q8 약 5.33 GiB를 만든 뒤 `DS4_CUDA_COPY_MODEL_CHUNKED=1`로 원본 94.16 GB까지 다시 복사하려 했다. 약 86.02 GiB 파생물과 전체 raw copy가 겹친 것이 OOM/hang의 핵심 위험이다.

그 경로와 환경변수를 다시 실행하지 않는다. 첫 실모델 실행은 반드시 **단일 VMM weight owner**가 raw/repacked range를 소유하고 worker가 import하는 방식이어야 한다.

v0.5.6.3 weight-server dry-run에서 확인한 값:

- VMM supported
- POSIX FD transfer supported
- VMM granularity 2 MiB
- logical raw model 87.69 GiB
- allocation plan 87.83 GiB
- 당시 free 116.14 GiB / total 121.63 GiB
- `--reserve-gb 24` preflight 통과

## 5. 메모리 관리 규칙

전체 모델을 띄우기 전과 종료한 뒤 다음 순서를 지킨다.

1. `nvtop`의 Compute view에서 모델/worker/weight-owner PID를 확인한다.
2. 동시에 `htop` 또는 현재 tmux의 `btop`에서 RSS, process tree, swap을 확인한다.
3. 기존 server와 worker를 정상 종료한다.
4. 대상 PID가 실제로 사라진 것을 `pgrep`/`ps`와 `nvtop` 양쪽에서 확인한다.
5. 그 후에만 `/usr/local/bin/clear_cache`를 실행한다.
6. `nvtop`과 `htop`/`btop`, `free -h`를 다시 확인한다.
7. 두 번째 대형 모델이나 두 번째 raw-copy owner가 없을 때만 다음 실행으로 넘어간다.

권장 점검 명령:

```bash
tmux attach -t sunghoon
pgrep -af 'ds4-server|ds4_weight_server|llama-server|vllm|python.*serve'
ps -eo pid,ppid,rss,stat,comm,args --sort=-rss | head -n 30
free -h
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

`clear_cache`는 살아 있는 모델 프로세스의 메모리를 회수하는 도구가 아니다. PID가 사라지기 전에 실행하지 않는다.

## 6. 작업 디렉터리와 Git 상태

### 보존용 이전 구현

```text
/home/sunghoon/workspace/ds4-exaone/ds4-motif-3
feature/motif-3-model-loader@d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9
```

이 worktree는 수신 구현과 기존 fixture 증거를 보존하기 위한 것이다. 새 통합 작업으로 덮어쓰지 않는다.

### 실제 통합 worktree

```text
/home/sunghoon/workspace/ds4-exaone/ds4-motif-3-v0563
branch: feature/motif-3-v0563-port
base: b9c97adfb4a921096a4df24e672599067269a7e2 (v0.5.6.3)
```

현재 remote:

```text
origin   https://github.com/Baekpica/ds4.git
upstream https://github.com/Entrpi/ds4.git
```

향후 실제 최상위 upstream 검토 시 사용자가 지정한 `https://github.com/antirez/ds4`와의 관계를 명시적으로 정리한다. 현재 remote를 작업 도중 조용히 바꾸지 않는다.

### 현재 cherry-pick 경계

원본 네 commit은 v0.5.6.3 위에 다음 로컬 SHA로 모두 이식됐다.

```text
01e1be0 Add native Motif-3 loader and latent CUDA runtime
b085b1c Document resident Motif-3 serving
ae0d3a2 Advertise Motif model ID from OpenAI server
64c67a0 Preserve complete question in Motif 256K gate
```

`CHERRY_PICK_HEAD`는 없고 branch는 base 대비 ahead 4다. conflict marker,
staged/unstaged tracked 변경은 없다. 생성된 테스트/weight-server 바이너리는
검증 산출물이므로 Git에 추가하지 않는다.

비교용 scratch worktree 두 개는 실제 산출물이 아니며 검토 후 제거했다.

```text
/home/sunghoon/workspace/ds4-exaone/scratch/ds4-motif-port-solarbase
/home/sunghoon/workspace/ds4-exaone/scratch/ds4-motif-merge-v0563
```

clean v0.5.6.3 비교용 `scratch/ds4-v0563-baseline`만 upstream 회귀 판정
증거로 유지한다.

## 7. 현재까지 반영한 코드

### 모델/loader/graph

- DeepSeek와 Motif를 명시적인 model-family enum/switch로 분리했다.
- Motif metadata/config validation, weight binding, layout validation을 이식했다.
- 53 layers, 14 full attention + 39 SWA, 384 routed experts/top-8 구조를 다루는 Motif graph를 이식했다.
- Motif tokenizer, chat control token, stop/thinking semantics를 이식했다.
- Motif session create/free/prefill/decode/sync dispatch를 v0.5.6.3 수명주기에 맞춰 조정했다.
- 최대 context는 262,144로 제한하고 CUDA-only, single-device GB10을 우선한다.
- 이전 OOM을 만든 강제 full-copy env/path는 이식하지 않았다.
- v0.5.6.3의 VMM manifest import/self-build 구조를 보존했다.

### CUDA

- BF16 matmul 및 Motif-specific primitive 선언을 추가했다.
- Motif router, PolyNorm, mHC, GDLA/differential attention, MoE 관련 CUDA kernel을 직접 이식했다.
- v0.5.6.3 single-device API에 맞추는 작은 GB10 compatibility helper만 추가했다.
- GLM multi-GPU/plugin 계층은 끌어오지 않았다.
- v0.5.6.3의 model-map/VMM range 접근을 유지하고 전체 raw model 강제 복사를 제거했다.

### 양자화기

- Motif fused BF16 expert slice와 DeepSeek FP4+scale expert 입력을 명시적으로 분리했다.
- Motif BF16 expert를 FP4 dequant 함수에 잘못 넣지 않도록 `tensor_to_f32` 경로로 분기했다.
- Motif BF16 source에 MXFP4 repack을 요청하면 명시적으로 실패하게 했다.

### 공통 API server 기반

- `SERVER_MODEL_SYNTAX_DEEPSEEK` / `SERVER_MODEL_SYNTAX_MOTIF3`의 작은 enum을 추가했다.
- request에 model syntax를 담고 Motif model ID/alias를 인식할 기반을 추가했다.
- Motif 공식 chat template, tool schema, tool call JSON parser, live tool tail, no-think tool checkpoint 코드를 이식했다.
- OpenAI, Responses, Anthropic이 같은 request/response machinery를 쓰는 v0.5.6.3 서버를 보존한다.

이 부분의 v0.5.6.3 call-site 배선은 §0.2–0.5에서 완료했다. GLM-only parser
fragment는 제거했고 DeepSeek/Motif의 작은 enum/switch 구조만 남겼다. GLM은
추후 같은 dispatch에 독립 모델 패밀리로 추가한다.

### 테스트 소스

다음 테스트가 추가된 상태다.

```text
tests/test_motif3_loader.c
tests/test_motif3_reference.c
tests/test_motif3_tokenizer.c
tests/test_motif3_cuda.cu
tests/test_motif3_resident.c
tests/test_motif3_long.c
```

새 v0.5.6.3 통합 worktree에서 CPU reference, CUDA primitive 6그룹,
구조/loader, tokenizer/chat fixture가 모두 통과했다. resident/long binary도
현행 API로 compile/link 및 manifest safety gate까지 확인했지만, full GGUF
실행은 아직 하지 않았다.

## 8. 일시 정지 시 빌드 상태 (2026-08-14 10:00 기준 — §0이 최신)

**이 절은 재개 세션 이전의 스냅숏이다. 아래 1–6번 항목은 §0에서 모두
해소되었고(§9.9의 BF16 테스트 1건 제외), 현재 상태는 §0을 본다.**

확인 완료:

```text
code conflict marker: 0
git diff --check: pass
git diff --check --cached: pass
ds4_server_cpu.o: compile pass (warning 있음)
```

아직 해결할 빌드 항목:

1. `make cpu`는 `ds4.c`의 `ds4_gpu_graph`/`payload_region` guard 오류에서 멈춘다. 비교 결과 이 구조는 v0.5.6.3 base에도 존재하므로 Motif 코드만의 오류로 단정하지 말고 baseline과 port를 분리해서 처리한다.
2. `gguf-tools/deepseek4-quantize`는 newer parent에 있던 `repack_fp4_weight_mxfp4` helper가 v0.5.6.3에 없어 컴파일이 멈춘다. helper를 작게 이식하거나, 실제 지원 범위에서 MXFP4 branch를 명시적으로 제외한다.
3. server의 model-syntax helper가 실제 parser/render/parse call site에 아직 연결되지 않았다.
4. conflict 해소 중 남은 사용하지 않는 GLM parser fragment를 제거해야 한다.
5. `make cuda-spark`와 새 worktree의 Motif unit test는 아직 실행하지 않았다.
6. weight owner, resident import, full-model correctness, benchmark, profiling, API server는 아직 실행하지 않았다.

따라서 현재 코드는 merge/remote publish 가능한 상태가 아니다.

## 9. 재개 시 코드 작업 순서

1. 현재 staged diff와 cherry-pick state부터 다시 읽는다.

   ```bash
   cd /home/sunghoon/workspace/ds4-exaone/ds4-motif-3-v0563
   git status
   git diff --cached --stat
   git diff --check --cached
   git rev-parse CHERRY_PICK_HEAD
   ```

2. GLM-only 잔여 코드를 제거하고 DeepSeek/Motif 두 syntax만 남긴다.
3. `server_model_syntax_for_engine()` 값을 모든 API parser의 request에 설정한다.
4. chat/completions, Responses, Anthropic의 render/live-tail/generated-message parser를 syntax dispatch에 연결한다.
5. Motif stop token과 think/tool 종료 조건이 serial 및 continuous path 모두에서 동일하게 작동하도록 연결한다.
6. 양자화기 MXFP4 helper 의존성을 해결한다.
7. CPU guard 문제는 clean v0.5.6.3 base 재현 여부를 확인한 뒤 최소 패치한다.
8. CPU/object build, server unit, Motif loader/reference/tokenizer test를 통과시킨다.
9. **완료:** `make cuda-spark` 후 cubin `sm_121a`, CUDA primitive 및
   synthetic long-context regression을 통과시킨다(증거 §0.12).
10. **완료:** `git cherry-pick --continue`와 남은 세 commit을 순서대로
    이식했다(최종 `64c67a0`, 증거 §0.15).
11. **완료:** 각 충돌을 현행 v0.5.6.3 lifecycle/API 기준으로 review하고
    `git diff --check`, server unit, long safety gate를 재실행했다.

## 10. weight owner와 inference worker 운용 기준

Solar와 동일하게 weight server는 오래 유지하고 inference server만 재빌드/재시작한다. 최초 실모델 단계에서 Q2_K/IQ2 aligned replacement는 사용하되, 약 6 GiB를 추가하는 optional Q8 aligned artifact는 profiling으로 필요성이 입증되기 전에는 끈다.

예상 첫 preflight 형태:

```bash
MODEL=/home/sunghoon/workspace/ds4-exaone/models/Motif-3-Mixed-Quant-GGUF/Motif-3-MQ87-88-FIT.gguf
RUN=/home/sunghoon/workspace/ds4-exaone/scratch/motif-3-v0563-runtime

./ds4_weight_server \
  --base "$MODEL" \
  --manifest "$RUN/weights.manifest" \
  --backend vmm \
  --scope base \
  --reserve-gb 24 \
  --no-repack-q8-aligned \
  --dry-run
```

dry-run 로그의 logical/allocated/free/reserve와 VMM/POSIX FD를 보존한다. live owner는 같은 인자에서 `--dry-run`만 제거해 독립 tmux pane에서 실행한다. `ready manifest=...`와 broker listening을 확인하기 전 worker를 시작하지 않는다.

worker는 manifest import를 명시하고 같은 GGUF 경로를 사용한다.

```bash
DS4_CUDA_WEIGHT_IPC_MANIFEST="$RUN/weights.manifest" \
./ds4-server -m "$MODEL" -c 2048 --host 0.0.0.0 --port 8001
```

실제 env/manifest 명칭은 build 완료 후 `--help`와 import 로그로 다시 확인한다. listening socket만으로 ready라 하지 않는다.

## 11. correctness와 성능 기준점

실행 순서는 다음 gate를 건너뛰지 않는다.

1. VMM owner preflight
2. VMM owner live ready + worker import 확인
3. 2K short greedy correctness 및 reference token 비교
4. reset/restart/reimport 후 동일 결과
5. baseline TTFM, prefill tok/s, decode tok/s 기록
6. 8K 또는 16K medium prefill/decode profile
7. 최적화 후 같은 command/seed/token으로 A/B
8. 32K → 64K → 128K → 정확한 256K

모든 성능 표에는 다음을 함께 기록한다.

- Git SHA와 GGUF revision/SHA
- exact command/env
- context/prompt/decode token 수
- cold/warm 여부 및 cached token 수
- TTFM/TTFT
- prefill wall time와 tok/s
- decode wall time와 tok/s
- peak/available unified memory와 swap
- clocks 및 active process
- correctness fixture/greedy token 결과

잠재적인 속도 목표는 Solar Mixed-Quant와 납득 가능한 같은 order의 TTFM/prefill/decode다. 아직 Motif 실측치가 없으므로 구체적인 달성 숫자나 성능 향상을 미리 주장하지 않는다.

## 12. nsys 사용 방식

곧바로 256K로 가지 않는다. correctness와 baseline 이후 8K 또는 16K처럼 짧지만 steady-state kernel 비중이 드러나는 prefill과 별도 decode window를 profile한다.

예시 형태:

```bash
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --output="$RUN/nsys/motif3-medium" \
  <exact benchmark or server command>

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  "$RUN/nsys/motif3-medium.nsys-rep"
```

nsys에서 우선 확인할 것:

- GPU time 상위 kernel과 launch count
- prefill/decode별 kernel composition
- CPU-GPU synchronization 및 API wait
- tiny kernel launch 폭증
- H2D/D2H 또는 unified-memory page migration
- cuBLAS와 custom kernel 사이의 gap
- MoE expert matvec/matmul, router, PolyNorm, mHC, GDLA/differential attention 비중

ranked table을 만든 뒤 상위 병목만 ncu로 넘긴다.

## 13. ncu 사용 방식

ncu를 전체 실행에 무차별 적용하지 않는다. nsys가 비용을 입증한 특정 kernel과 재현 가능한 짧은 launch window만 잡는다.

확인 항목:

- achieved occupancy
- registers/thread와 local spill
- shared memory/block 및 blocks/SM
- DRAM/L2 throughput과 cache hit
- arithmetic intensity와 tensor/core utilization
- warp stall reason
- branch/replay 및 memory coalescing

예시 형태:

```bash
ncu \
  --kernel-name 'regex:<target-kernel>' \
  --launch-skip <N> \
  --launch-count <N> \
  --section SpeedOfLight \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --export "$RUN/ncu/<target>" \
  <short reproducible command>
```

먼저 profiling permission을 확인한다. `RmProfilingAdminOnly` 등으로 차단되면 그 사실을 기록하고, 권한/보안 설정을 조용히 바꾸지 않는다.

최적화는 한 모듈씩 수행한다. fixture → full-model A/B → keep/revert 순서를 지키며, 단순 acceptance rate나 microbenchmark만으로 release path를 바꾸지 않는다.

## 14. 256K 검증 방식

각 context 단계에서 최소한 다음을 통과한다.

- admission 및 prefill 완료
- 실제 decode 1개 이상, 최종 gate에서는 여러 token
- NaN/Inf 없음
- reset/reuse/restart 동작
- tokenizer와 prompt 길이 증거
- `/v1/stats`의 settled gauge
- 메모리 peak와 종료 후 회수 확인
- 가능하면 long-context semantic/needle fixture

262,144 strict gate는 prompt가 정확히 요구 길이를 포함해야 하며, prefill
도중 중단되면 실패/미완료다. 245,760 같은 근접 수치를 256K로 반올림하지
않는다. 현재 long harness의 262,080 prompt + 64 decode는 별도의
“native-window-full” 증거로 기록하고 strict 262,144-prompt pass로 표기하지
않는다.

## 15. API acceptance matrix

256K 이전 short/medium context에서 API 기능을 먼저 고정하고, 마지막에 long-context surface를 재검증한다.

| Surface | 필수 확인 |
|---|---|
| `/v1/models` | 실제 감지된 Motif model ID |
| `/v1/chat/completions` | stream/non-stream, reasoning on/off, tool call/continuation |
| `/v1/completions` | stream/non-stream, raw prompt semantics, stop/usage |
| `/v1/responses` | stream/non-stream, function call/output continuation, visible/live KV replay |
| `/v1/messages` | Anthropic stream/non-stream, thinking, tool_use/tool_result continuation |
| `/v1/stats` | model family, context, queue/cache/continuous 상태 |
| `/metrics` | request, prefill, decode, cache, error counter |

port open은 readiness가 아니다. `/v1/models`, `/v1/stats`, 실제 generation 요청을 모두 확인한다. continuous serving을 켠 경우 종료 후 active/queued gauge가 0으로 정착하는 것도 확인한다.

## 16. 문서와 원격 동기화 규칙

장기 작업 중에는 유의미한 KST 경계에서 약 4시간 간격으로 checkpoint한다. 우선 기준은 12:00, 16:00, 20:00, 22:00 이후의 다음 유의미한 경계이며, 테스트 중간의 깨진 상태를 시간만 맞추려고 push하지 않는다.

항상 순서는 다음과 같다.

1. 이 handoff 또는 해당 기술 문서를 먼저 갱신한다.
2. 검증된 사실과 미완료 경계를 분리한다.
3. `git diff --check`, 관련 test, status를 확인한다.
4. Git commit/push를 수행한다.
5. HF handoff bucket에는 재현에 필요한 문서, manifest, 로그, profile만 최소 업로드한다.
6. 공개 model repo는 검증된 성능/사용법/한계만 반영한다.
7. HF 업로드 후 반환 revision을 다시 다운로드해 대상 파일을 byte-for-byte/hash 비교한다.

Git과 HF 어느 쪽도 문서보다 먼저 갱신하지 않는다. model card에는 실패한 실험, 추측, future work journal을 넣지 않는다. 그런 내용은 이 기술 handoff에 남긴다. 요청받지 않은 GGUF, shard, manifest 또는 다른 파일을 model repo에 업로드하지 않는다.

## 17. 현재까지도 하지 않은 일

- Git remote push (로컬 통합 commit은 완료)
- Hugging Face upload
- weight owner live allocation
- inference worker 실행
- `clear_cache` 재실행 (모델 프로세스를 하나도 띄우지 않았으므로 불필요)
- nsys/ncu profile
- full-model correctness/benchmark (구조/fixture 테스트만 수행)
- 32K 이상 Spark 실행
- 256K 통과 주장

다음 단계: §5의 `nvtop`+`btop`/PID/메모리 점검을 수행하고, §10의
`ds4_weight_server --dry-run` preflight를 기록한다. 통과한 뒤에만 live VMM
owner와 2K worker correctness/API gate로 넘어간다.
