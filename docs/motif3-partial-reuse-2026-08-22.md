# Motif-3 partial prefix reuse

검증일: 2026-08-22, DGX Spark GB10, CUDA 13.3 `sm_121a`
구현 커밋: `cf817c7` (Solar 선행 구현 `042bcea`의 공용화 + Motif-3 이식)
모델: `Motif-3-MQ87-88-FIT.gguf` (병합 단일 파일, 94,162,541,472 B)

## 구현 범위

Motif-3는 recurrent state가 없다. 뱅크의 되감기 불가 상태는 SWA 39개
레이어의 128-token 윈도우 ring뿐이고, full-attention 14개 레이어의
latent KV rows는 소스 뱅크에 전체 히스토리가 남는다. 따라서 체크포인트는
KDA state 전체가 아니라 SWA 윈도우 슬라이스만 저장한다.

- Solar의 slot/ref 부기를 `ds4_partial_checkpoint`로 공용화 (32-slot,
  bank-ref bitmask, LRU). Solar는 같은 구조 위에서 기존 동작 유지.
- Motif-3 슬롯: SWA 39 레이어 × 128 rows × (kv_lora 512 + rope 64) ×
  BF16 = **5.48 MiB/slot** (Solar 157.50 MiB 대비 ~1/28). 32 슬롯 가상
  175.4 MiB, VMM demand mapping.
- 캡처 3지점: request 경계(semantic, logits 포함), long-prefill 청크
  stride 경계(logits 없음), 디코드 stride 경계(logits 포함). stride는
  `max(4096, ceil(ctx/24))`를 4096 단위로 올림 (`-c 65536`에서 4096).
- partial fork 복원: LCP 이하 가장 가까운 체크포인트의 SWA 윈도우 행을
  dst ring의 `pos % cap` 슬롯에 기록(최대 2 세그먼트), full-attention
  rows `[0, pos)`는 소스 뱅크에서 D2D 복사, 나머지 suffix만 replay.
  MTP 블록 캐시는 뱅크 레인이 실행하지 않으므로 0으로 리셋.
- exact fork는 스냅샷 복사 없이 reference 상속; cold reset, payload
  restore, bank trim, 히스토리 무효화에서 stale reference 제거.
- `DS4_SERVER_FORK_PARTIAL=0`이면 체크포인트 VA도 예약하지 않는 완전한
  control. 캡처는 memory floor + serial reserve 미달이면 건너뛰는
  best-effort. EXAONE 뱅크 레인은 exact-frontier 재사용만 유지.

## Correctness gate

`tests/test_motif3_batch <model> --partial-only`. VMM weight owner 없이
단독 로드로 실행했고, 두 stage 모두 cold oracle과 greedy 토큰이 같았다.

Stage 1 (chunk 16, ctx 128 — 선형 윈도우 영역, Solar 게이트 미러):

| Source | Proposed cut | Restored checkpoint | Replay | Greedy token |
|---:|---:|---:|---:|---:|
| 32 | 19 | 16 | 4 | 123 |
| 32 | 27 | 24 | 4 | 689 |

Stage 2 (chunk 4096, ctx 8192 — SWA cap 4225, ring wrap 영역):

| Source | Proposed cut | Restored checkpoint | 종류 | Replay | Greedy token |
|---:|---:|---:|---|---:|---:|
| 4500 | 4200 | 4096 | periodic (stride) | 105 | 123 |
| 4500 | 4400 | 4300 | boundary + **2-세그먼트 ring wrap** | 101 | 173 |

4300 체크포인트의 윈도우 행 `[4172, 4300)`은 4225-슬롯 ring을 감아
두 세그먼트로 복원된다. Gate는 `Motif-3 partial reuse checkpoint gate
passed`로 종료했다.

## 동일 workload A/B

두 leg 모두 같은 binary(`cf817c7`), 같은 VMM owner(87.83 GiB plan,
`--reserve-gb 16`), 4 banks, `-c 65536`, prefill chunk 4096, greedy
streaming Chat(`thinking disabled`, `include_usage`). Control만
`DS4_SERVER_FORK_PARTIAL=0`. 16,837-token source를 넣고 같은 rendered
prompt 내부 약 7.1K/14.1K 위치에서 분기했다.

| Request | Control cached/computed | Treatment cached/computed | Control TTFT | Treatment TTFT | Speedup |
|---|---:|---:|---:|---:|---:|
| source 16,837 | 0 / 16,837 | 0 / 16,837 | 28,646.2 ms | 28,711.9 ms | 0.998x |
| branch 7,140 | 0 / 7,140 | 4,096 / 3,044 | 11,319.4 ms | 5,196.3 ms | **2.18x** |
| branch 14,095 | 0 / 14,095 | 12,288 / 1,807 | 23,563.1 ms | 3,627.9 ms | **6.50x** |

세 요청 모두 control/treatment 응답 텍스트가 byte-identical했다.
Source 캡처 비용은 이 단일 표본에서 65.7 ms, 0.23%였다. 서버 로그의
proposed cuts는 7,127/14,083 tokens였고, engine이 인정한 cached bases는
usage가 보고한 4,096/12,288이었다. Treatment `/v1/stats`:

```text
requests_completed:3
requests_failed:0
admits_cold:1
admits_partial_fork:2
tokens_prefilled_cached:16384
tokens_prefilled_computed:21688
```

## Memory evidence

- owner 부팅 plan: need 87.83 GiB, budget 117.67 GiB, `--reserve-gb 16`
- treatment 부팅 로그: `partial=32 x 5.48 MiB shared checkpoints
  stride=4096`; control 부팅 로그: `partial=0 ... stride=0`
- owner + worker 서빙 중: used 110 GiB / available 11 GiB (floor 8 GiB)
- worker→owner 순서 종료 후 `clear_cache`: available 117 GiB,
  `nvidia-smi` compute processes 0

## 운영 한계

- checkpoint pool은 worker-local이며 bank payload/disk KV로 직렬화하지
  않는다. 소스 뱅크의 token history와 full-attention rows가 사라지면 그
  lineage의 partial lookup도 사라진다.
- 32 slots 고정 LRU, Solar와 공유 정책. 실제 cached 토큰 수는 proposed
  LCP가 아니라 `usage.prompt_tokens_details.cached_tokens`로 판단한다.
- EXAONE 뱅크 레인은 여전히 exact-frontier warm/fork만 지원한다.

## 이 작업과 무관한 기존 회귀 (기록)

full `test_motif3_batch`(3-bank vs serial oracle)의 row 1
("Two plus two equals")이 토큰 2/3에서 oracle과 어긋난다. 체크포인트
코드가 없는 `042bcea` 바이너리(v062-dfm-sync 트리)로 **동일하게
재현**되므로 이 변경과 무관한 v0.6.2-dfm 라인의 기존 drift다 (Motif
perf 사이클 이후 3-bank 게이트 미재실행 구간에서 유입된 것으로 추정).
별도 추적 필요; partial-only 게이트와 라이브 A/B는 전부 통과했다.
