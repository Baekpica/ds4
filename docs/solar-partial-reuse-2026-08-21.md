# Solar Open2 partial prefix reuse

검증일: 2026-08-21, DGX Spark GB10, CUDA 13.3 `sm_121a`

## 구현 범위

Solar의 KDA state는 임의 token 위치로 되감을 수 없다. continuous bank마다
snapshot 하나를 추가하는 대신, runtime 전체가 다음 bounded cache를 공유한다.

- KDA checkpoint 32 slots, slot당 157.50 MiB virtual VMM range
- exact fork는 snapshot을 복사하지 않고 reference만 상속
- request prompt 경계와 long-prefill 주기 경계를 checkpoint로 저장
- 주기 간격은 `max(4096, ceil(ctx/24))`를 4096 token 단위로 올림
- partial fork는 제안된 token LCP 이하에서 가장 가까운 checkpoint를 복원
- source bank의 GQA rows를 checkpoint 위치까지만 복사하고 나머지 prompt를 replay
- cold reset, payload restore, static batch replacement, bank trim 시 stale reference 제거
- LRU 32-slot 상한과 live memory floor를 넘지 않는 best-effort demand mapping
- `DS4_SERVER_FORK_PARTIAL=0`이면 checkpoint VA도 예약하지 않는 완전한 control

이 구조는 bank 수보다 많은 과거 분기점을 보존하지만 radix tree는 아니다. KDA
snapshot은 공유되며, token history와 GQA prefix source는 retained bank에 남는다.
따라서 bank가 제거되면 그 bank만 참조하던 checkpoint도 회수 가능해진다.

## Correctness gate

VMM weight owner를 먼저 올린 뒤 partial-only gate를 실행했다.

```sh
DS4_CUDA_WEIGHT_IPC_MANIFEST="$RUN/weights.manifest" \
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
DS4_MEM_FLOOR_GB=10 \
./tests/test_solar_session "$MODEL" --partial-only
```

한 source bank에 16, 24, 32-token semantic checkpoint를 순서대로 만든 뒤 두
분기를 cold oracle과 비교했다.

| Source | Proposed cut | Restored checkpoint | Replay | Greedy token |
|---:|---:|---:|---:|---:|
| 32 | 19 | 16 | 4 | 9024 |
| 32 | 27 | 24 | 4 | 9024 |

두 partial 결과는 각각의 cold oracle과 같은 token을 냈고 source frontier 32를
보존했다. Gate는 `Solar partial reuse checkpoint gate passed`로 종료했다.

기존 전체 `test_solar_session`은 새 gate보다 앞선 snapshot cold/warm 허용치에서
현재 imported artifact 기준 `max_abs=0.331638336`으로 기존 한계 0.25를 넘어
중단된다. 이 unrelated baseline 허용치는 변경하지 않았다. CPU/server unit gate는
`./ds4_test --server`에서 `server: OK`, `ds4 tests: ok`를 통과했다.

## 동일 workload A/B

두 leg 모두 같은 binary, VMM owner, 4 banks, `-c 65536`, prefill chunk 4096,
greedy Chat 요청을 사용했다. Control만 `DS4_SERVER_FORK_PARTIAL=0`이었다.
12,123-token source를 먼저 넣고 같은 rendered prompt 내부의 약 6K/10K
위치에서 각각 분기했다.

| Request | Control cached/computed | Treatment cached/computed | Control TTFT | Treatment TTFT | Speedup |
|---|---:|---:|---:|---:|---:|
| source | 0 / 12,123 | 0 / 12,123 | 10,654.0 ms | 10,710.7 ms | 0.995x |
| branch 6K | 0 / 6,079 | 4,096 / 1,983 | 5,323.7 ms | 1,868.8 ms | 2.85x |
| branch 10K | 0 / 10,148 | 8,192 / 1,956 | 8,920.2 ms | 1,929.8 ms | 4.62x |

두 branch의 control/treatment 응답은 모두 `The text repeats the`로
byte-identical했다. Source capture 비용은 이 단일 표본에서 56.7 ms, 0.53%였다.
Treatment `/v1/stats`는 다음을 보고했다.

```text
requests_completed:3
requests_failed:0
cont_batch_failures:0
admits_cold:1
admits_partial_fork:2
tokens_prefilled_cached:12288
tokens_prefilled_computed:16062
cont_admit_rejects:0
```

서버 로그의 proposed cuts는 6,057/10,127 tokens였고, engine이 실제로 인정한
cached bases는 API usage가 보고한 4,096/8,192 tokens였다.

## Memory evidence

- Treatment boot census: device live 100.27 GiB
- 세 요청 뒤 census: device live 102.04 GiB
- 같은 시점 system available: 14.59 GiB, configured floor: 8 GiB
- checkpoint capture 또는 admission refusal: 0
- worker와 owner를 순서대로 종료한 뒤 `clear_cache`: available 119 GiB,
  `nvtop` compute processes 0

32 slots의 5.04 GiB는 virtual maximum이다. 실제 pages는 checkpoint capture 때만
mapping되고, memory floor와 serial reserve를 넘길 수 없으면 그 checkpoint만
건너뛴다. 기존 exact warm/fork와 cold generation은 계속 동작한다.

## 운영 한계

- checkpoint pool은 worker-local이며 bank payload나 disk KV 파일에 직렬화하지 않는다.
- source bank의 token history/GQA rows가 사라지면 해당 lineage의 partial lookup도
  사라진다. 완전한 cross-bank radix cache가 필요해질 때 별도 page ownership이 필요하다.
- 32 slots는 고정 LRU다. 현재 workload에서 범위가 부족하다는 측정이 나오기 전에는
  tree/index/config abstraction을 추가하지 않는다.
- 실제 cached token 수는 proposed LCP가 아니라 `usage.prompt_tokens_details.cached_tokens`
  또는 `on_admitted` 값으로 판단한다.
