# Motif-3 DGX Spark 최적화 작업 handoff

기준 시각: **2026-08-14 17:02 KST** (재개 세션 진행 내역은 §0에 누적;
후속 재개 결과는 §0.26까지 반영; **v0.6.2-dfm 최적화 세션은 §A에 누적**.
재개 에이전트용 단권 핸드오프:
`docs/v062-dfm-motif3-opt-resume-2026-08-21.md`)

## §A. v0.6.2-dfm 최적화 세션 (2026-08-21 KST)

전제: Entrpi `v0.6.2` 흡수 후 `v0.6.2-dfm` 태그 publish 완료. 이 세션의
목표는 (1) 깊은 컨텍스트에서 prefill/decode가 급락하는 현상의 완화(에이전트
활용 시나리오), (2) 절대 속도 향상. 모든 측정은 GB10 (driver 610.43.02,
CUDA 13.3, sm_121a), MQ87-88 병합 GGUF, aligned-Q8 VMM owner
(`--reserve-gb 24`), `context-32768.txt` 코퍼스, greedy·no-think 조건.

### §A.0 v0.6.2-dfm 베이스라인 (merge 직후, 70d5823)

| cell | prefill tok/s | decode tok/s |
|---|---:|---:|
| 8K prefill ×2 | 519.90 / 518.02 | — |
| 8K decode(64) | 515.84 | 12.62 |
| 32K decode(64) | 445.03 | 9.68 |

v0.6.0-dfm 리메저 밴드(520.39/516.83, 12.24)와 동일 — merge 성능 회귀 없음.

### §A.1 nsys 32K decode-window 분해 (gen 128, GPU busy 96.6%)

sqlite에서 마지막 13.0 s(decode 창)만 집계:

| % | kernel | 성격 |
|---:|---|---|
| 32.0 | `motif3_latent_attention_bf16_decode_hg_partial` | **깊이 비례** (14 full layers × 10 head-group KV 재독) |
| 18.9 | `q8_0_aligned_dense_vec` | dense Q8 matvec, roofline 근접 |
| 22.5 | `mul_mat_q` ×2 (MoE gate/up + down) | roofline 대비 ~2배 여지 |
| 4.2 | rms_norm_weight | 157회/token |
| 3.3/3.2/2.0 | qk_absorb / q8 pair / value_project | — |

decode 103.5 ms/token 중 attention ≈ 33 ms가 유일한 depth-linear 항 —
256K에서 ~8× 성장해 지배(공개 2.52 tok/s의 실체).

### §A.2 사이클 1 — latent flash-decode head-group 확장 (d03bd89, 반영)

원인: 80 heads / 8 heads-per-CTA = 10그룹이 latent 링을 각각 재독 +
CTA당 8-row 타일 직렬 barrier 체인. 마이크로벤치 스캔
(`scratch/motif3-opt-v062/bench-hg-scan.cu`, 합성 80-head fixture)에서
(warp당 2 heads × 8 warps = 16 heads/CTA, 16-row 타일, split =
visible/4096 clamp [32,64])이 8K/32K/256K에서 2.03x/1.84x/2.15x,
출력 rel_rms ~1e-6 (split-order 재결합).

full-model A/B (owner 동일 세션):

| cell | base | HG16 | Δ |
|---|---:|---:|---:|
| 8K decode | 12.62 | 13.14 | +4.1% |
| 32K decode | 9.68 | **11.28** | **+16.5%** |
| 8K/32K prefill | 519.9/445.0 | 519.6/445.0 | 불변 |

정합성: 8K frontier logits **비트 동일**(prefill 무변경 확인),
Motif CUDA fixture 6그룹 통과, 32K OpenAI Chat 센티널 게이트
**3/3 exact** (HTTP decode 11.2 tok/s; 공개 시절 8.96 대비 +25%).
greedy 텍스트는 base 대비 토큰 수준 분기 — softmax 분할 순서 변경의
예상 드리프트(과거 FATTN occupancy/absorb 최적화와 동일 클래스)로,
게이트 기준은 센티널 정확성이다.

### §A.3 사이클 2 — SWA prefill의 expanded/HMMA 라우팅 (b0db5a1, 반영)

nsys prefill 분해(32K, 74.6 s): fattn_hmma 15.7%, q8 pair 12.3%,
value_project 8.7%(대부분 SWA latent 경로), gateup_iq2 8.1%, SWA latent
generic ~6.5%, f32_to_bf16 5.4%(43,884회), qk_absorb 4.5%, round_bf16
3.2%(66,650회). SWA prefill이 latent 경로(rows×80 heads의 W_UV 사영)로
지불하는 ~15%를, window 지원이 이미 있는 HMMA FATTN
(`ds4_mmq_motif3_prefill_attn_hmma`)으로 회수한다. 구현:
`expanded_attention_range`에 window 관통, SWA 프리픽스는 ring 랩 기준
최대 2개 선형 슬롯 세그먼트로 분할해 기존 load/kv_b/prepare/merge
시퀀스 재사용, FATTN epilogue가 빈 쿼리 행에 0/-inf 중립 partial을
기록하도록 수정(스테일 merge 방지).

결과 (b0db5a1, 반영):

| cell | 사이클1 후 | 사이클2 후 | Δ |
|---|---:|---:|---:|
| 8K prefill | 519.6 | **620.9 / 620.0** | +19.5% |
| 32K prefill | 445.0 | **526.6** | +18.3% |
| 8K/32K decode | 13.14 / 11.28 | 13.14 / 11.27 | 불변 |

32K 센티널 게이트 3/3 exact, TTFT 73.4→62.8 s. Motif CUDA fixture
(공식 expanded GDLA 대조 포함)·model-family kernels 통과. 8K frontier
logits는 rel_rms 1.4e-1로 이동(argmax·top-4 불변) — SWA prefill이
absorbed-latent 반올림 경계 대신 공식 expanded 경계를 갖게 된 결과로,
full 레이어가 이미 채택한 것과 같은 트레이드다.

### §A.4 사이클 3 — MoE 디코드의 D2R 스케줄 편입 (91823ca, 반영)

nsys 디코드 창에서 MoE `mul_mat_q`(gate/up+down)가 22.5%로, roofline
추정(~10.8 ms/token) 대비 ~2배. 원인: SoA D2R MoE 커널은 존재하지만
`d2r_min_cols()` 기본 1024 게이트에 걸려 디코드(8 assignments)가 classic
MMQ 타일로 낙하. env probe(`DS4_MMQ_D2R_MIN_COLS=1`):

| cell | 기본 | D2R 강제 | Δ |
|---|---:|---:|---:|
| 8K decode | 13.14 | **15.06** | +14.6% |
| 32K decode | 11.27 | **12.66** | +12.3% |
| prefill | 620/527 | 619/521 | ~불변 |

반영: 공유 기본값(1024, DeepSeek prefill 믹스 기준)을 옮기지 않고, SoA
MoE 엔트리 2종에 명시적 `d2r_ncols_floor` 파라미터(0=기존 정책)를 추가해
Motif 라우티드 디스패치만 8을 전달. 타 패밀리·DeepSeek 호출부는 0으로
무변경. 반영 A/B: 8K decode 15.10 / 32K decode 12.73 tok/s (probe 재현),
prefill 불변, 32K 센티널 3/3 exact (HTTP decode 12.5), mmq-parity·CUDA
fixture·family kernels 통과. 커밋 완료.

### §A.5 3사이클 누적 (v0.6.2-dfm 베이스라인 대비)

| cell | 베이스라인 | 3사이클 후 | 누적 Δ |
|---|---:|---:|---:|
| 8K prefill | 519.9 | ~620 | **+19%** |
| 32K prefill | 445.0 | 524.9 | **+18%** |
| 8K decode | 12.62 | 15.10 | **+19.6%** |
| 32K decode | 9.68 | 12.73 | **+31.5%** |

깊이 축: 32K decode 개선폭(+31.5%)이 8K(+19.6%)보다 커 깊이-감쇠가
완화됨(HG16의 depth-linear 항 절반). 256K 검증은 최종 게이트에서.

### §A.6 2026-08-21 11:47 재부팅 (검증)

사이클 3 푸시 후 owner를 내리고 ncu 창을 열었다. HG16 마이크로벤치
ncu(`scratch/motif3-opt-v062/ncu/hg16-256k.txt`)는 완료. 이어서
`run-ncu-real.sh`가 owner 없이 standalone `ds4-bench`로 풀 모델
artifact 86.07 GiB + device 93.08 GiB를 올린 채 fattn ncu를 시작해
11:35 systemd-oomd가 user.slice 압력 72%로 tmux-spawn을 죽였다.
사용자는 11:47에 재기동. 커널 OOM killer 기록은 없다. 재발 방지:
ncu는 마이크로벤치만, 풀모델 `ds4-bench`/`ds4-server`에 ncu 금지.
상세는 `docs/v062-dfm-motif3-opt-resume-2026-08-21.md` §4.

### §A.7 HG16 ncu (유효) — 사이클 4 입력

`hg_partial_t<2,16>` @ 256K: SM/Mem 둘 다 63.35%, theoretical
occupancy 33.3% (regs **그리고** smem이 CTA를 2개/SM로 제한),
L1TEX scoreboard stall 33.8%, L2 11%. 각 스레드가 16헤드
`out0[16]/out1[16]`를 모두 accumulate하는 것이 레지스터 폭주.
워프-로컬 2-head value accum이 다음 가설. 재부팅 직후 호스트는
118 GiB available, GPU 프로세스 0.

### §A.8 사이클 4 — HG16 워프-로컬 accum (원복)

ncu: occupancy 33%가 regs+smem 공동 제한, L1TEX stall 33.8%.
가설: 각 워프가 자기 2헤드만 accumulate/기록하면 레지스터와 value
FMA가 줄어든다.

첫 마이크로벤치(`bench-hg-occ.cu` 초기본)는 1.08–1.15x + rms 0으로
보였으나, `partials`를 런치마다 제로화하지 않아 워프가 쓰지 않은
448/512 차원이 기준 출력에 남았다. 엔진 이식 후
`test-motif3-cuda`가 `CUDA latent GDLA split decode mismatch at 10849:
got 0`으로 실패 — 블록 전역 `(d0,d1)` 매핑에서 워프-로컬 쓰기는
헤드당 64차원만 덮는다. 커널 원복, 픽스처 재통과.

정직 재측정(partials memset, 동일 지오메트리):

| variant | 8K | 32K | 256K | rms |
|---|---:|---:|---:|---|
| reload-smem (value 배열 제거) | 1.00x | 1.00x | 1.00x | 0 |
| bf16 latent smem | 0.97x | 0.97x | 0.97x | 0 |
| bf16+reload | 0.97x | 0.97x | 0.98x | 0 |

HG16 점유율/레지스터 변형은 해소 가능 병목이 아니다. 다음 사이클은
사이클 3 바이너리로 **재-nsys** 한 뒤 새 1위를 고른다.

### §A.9 재-nsys 32K gen128 (사이클 3 바이너리, 2026-08-21 12:37)

Owner IPC, `ds4-bench` HEAD `a530fd6` (커널은 91823ca와 동일).
CSV: prefill 521.43 tok/s, decode 12.71 tok/s (128 tok) — 사이클 3
밴드 재현. 산출: `scratch/motif3-opt-v062/nsys/v062-c3-32k-gen128.*`.

전체 76 s(prefill 지배): fattn_hmma 18.4%, q8 pair batch 12.4%,
gateup_iq2_d2r 9.6%, f32_to_bf16 5.5% (43,884회), down_q2k_d2r 5.4%.

**Decode 창** (첫 HG launch → 끝, GPU 커널 9.558 s, 128 tok ≈ 74.7
ms/token; 사이클 전 103.5 ms에서 하락):

| % | kernel | 비고 |
|---:|---|---|
| 25.2 | `q8_0_aligned_dense_vec` | 34,300회, ~268/token. 절대속도 1위. 깊이 무관. aligned N=1 경로, 과거 roofline 근접 |
| 22.5 | HG16 `hg_partial` | 1,792회 = 14 full × 128. **남은 깊이 비례항**. 사이클 4 점유율 변형은 닫힘 |
| 10.8+5.7 | D2R MoE gateup+down | 사이클 3 반영 후. 구 22.5% classic MMQ에서 하락 |
| 5.5 | rms_norm_weight | 20,350회 |
| 4.4 | qk_absorb group5 | |
| 4.3 | q8 pair aligned | |
| 4.2 | `motif3_latent_attention_bf16` (비-HG) | 4,992회 = 39 SWA × 128. window=129라 HG 게이트(≥1024) 밖 |

사이클 5 후보 우선순위: (1) dense vec — 마이크로벤치 ncu만, owner
down, (2) HG L1TEX coalescing (점유율 말고 접근 패턴), (3) prefill
fattn/q8-pair. 풀모델 ncu 금지.

### §A.10 사이클 5 — HG16 cp.async BF16 더블버퍼 (반영)

Q8 dense vec occupancy 변형은 Motif 형상 마이크로벤치에서 닫힘
(`scratch/motif3-opt-v062/logs/q8-dense-motif.txt`): 큰 GEMV는 이미
253–270 GB/s, 2-row/2-warp는 손해. shexp down `K=1280`은 aligned
계약(`K % 1024 == 0`) 밖.

ncu L1TEX scoreboard 33.8%를 숨기기 위해 HG16 KV 타일을 BF16
공유메모리에 두고 다음 타일을 compute 중 `cp.async` (기존
`tt_cp_async_*` 재사용). 점유율은 ~38 KiB로 2 CTA/SM 유지.
합성 80-head (`bench-hg-l1tex.cu`):

| ctx | shipped | db_bf16 | Δ | rms |
|---:|---:|---:|---:|---:|
| 8K | 0.281 ms | 0.268 ms | +4.7% | 0 |
| 32K | 1.119 ms | 1.038 ms | +7.8% | 0 |
| 256K | 8.513 ms | 7.135 ms | **+19.3%** | 0 |

full-model A/B (owner 동일, 사이클 3 대비):

| cell | c3 | c5 | Δ |
|---|---:|---:|---:|
| 8K decode | 15.10 | **15.21** | +0.7% |
| 32K decode | 12.73 | **13.02** | +2.3% |
| 8K prefill | 621.7 | 616–622 | 불변 |

e2e가 커널 Δ보다 작은 이유: HG는 decode의 22.5%. 32K에서
0.225 × 7.8% ≈ 1.8%와 측정 +2.3%가 맞는다. 256K에선 깊이 비례항이
커져 커널 +19%가 더 많이 드러난다.

정합성: Motif CUDA fixture 통과, 32K 센티널 **ALL-EXACT** (HTTP
prefill 519.4 / decode 12.7). CSV:
`scratch/motif3-opt-v062/logs/l1tex-*.csv`.

시작점(v0.6.2-dfm merge) 대비 누적: 8K prefill +19%, 32K prefill
+18%, 8K decode **+20.5%** (12.62→15.21), 32K decode **+34.5%**
(9.68→13.02).

### §A.11 사이클 6 — Motif FATTN HMMA TK=32 (반영)

재-nsys 전체 76 s의 1위는 `motif3_fattn_hmma` 18.4%. 출하 TK=16은
K 타일마다 syncthreads. 합성 벤치
(`scratch/motif3-opt-v062/bench-fattn-tk.cu`, owner 유지):

| shape | TK16 | TK32 | Δ | TK16 vs TK32 rms |
|---|---:|---:|---:|---|
| chunk-4k (qpos0=0) | 10.15 ms | 8.99 ms | +12.9% | 0 |
| late-16k | 79.86 ms | 62.41 ms | +28.0% | 0 |
| late-32k | 185.57 ms | 141.54 ms | **+31.1%** | 0 |
| SWA late (win=129) | 3.22 ms | 3.09 ms | +4.2% | — |

TK=64는 3-CTA 예산을 깨고 전부 손해. GQA 80/16=5라 Solar식
2-head pair는 홀수 그룹 + 256-thread CTA가 이미 TK16보다 느림
(닫힘). 엔진은 `M3_FA_TK=32`, consume를 16-key 스텝 2회로 유지,
`__launch_bounds__(128, 3)` 유지 (~21 KiB smem).

full-model A/B (owner 동일 세션, c5 바이너리 재측정):

| cell | c5 | TK32 | Δ |
|---|---:|---:|---:|
| 8K prefill | 616.27 | **627.19** | +1.8% |
| 32K prefill | 519.57 | **545.62** | **+5.0%** |
| 8K decode | 15.21 | 15.06 | 불변(노이즈) |
| 32K decode | 13.00 | 12.95 | 불변 |

정합성: `test-motif3-cuda` 통과, 32K 센티널 **ALL-EXACT** (HTTP
prefill 546.7 / decode 12.8). CSV:
`scratch/motif3-opt-v062/logs/fattn-tk32-*.csv`,
`logs/c5-recheck-32k.csv`, `logs/sent-fattn-tk32.txt`.

시작점 대비 누적: 8K prefill **+21%** (519.9→627.2), 32K prefill
**+23%** (445.0→545.6). decode는 사이클 5와 동일 밴드. 이 숫자를
새 published 8K로 올리지 말 것 — 256K 미검증.

다음 후보: q8 pair prefill (전체 12.4%), shexp-down K=1280 aligned
(decode ~1.8%, owner artifact 재빌드), 가능하면 256K 센티널.

### §A.12 사이클 7 — Q8 pair tok8 타일 (원복)

가설: coalesced pair가 토큰마다 weight row를 재독한다. 합성
(`scratch/motif3-opt-v062/bench-q8-pair.cu`, shexp M=1280 K=4096 N=4096)
에서 TOK=8이 **+38.5%**, rms 0. TOK=16은 스필로 손해. dense
M=12288는 기존 `max_out<=2048` 캡이 맞고 coalesced가 stride보다
느리다.

엔진 이식 후 e2e는 반대: 8K prefill 627.19→587.74, 32K
545.62→515.07. 대형 `ds4_cuda.cu` TU에서 레지스터/점유율이
합성 벤치와 달랐다. 커널은 사이클 6 상태로 원복, 8K 재측정
625.45(밴드 복귀), `test-motif3-cuda` 통과. Q8 pair 토큰 타일은
닫힌 길.

다음 후보: shexp-down `K=1280` aligned (owner artifact 재빌드),
또는 256K 센티널. 풀모델 ncu 금지.

### §A.13 사이클 8 — shexp-down K=1280 aligned (닫힘)

가설: Motif `ffn_down_shexp`는 M=4096, K=1280이라 aligned decode
커널의 `K % 1024 == 0` 계약 밖으로 raw mmvq에 떨어진다. 테일 가드
(nb=40 = 32+8)를 넣으면 같은 SoA 경로를 탈 수 있다. owner는 이
형상을 아직 만들지 않는다(`ds4_repack_q8_candidate`가
`dims[0] % 1024 == 0`).

합성 벤치 (`scratch/motif3-opt-v062/bench-shexp-down-k1280.cu`,
owner 유지, ncu 없음):

| path | us | GB/s | Δ |
|---|---:|---:|---:|
| raw packed Q8 GEMV | 6.423 | 867 | — |
| aligned tail | 6.167 | 903 | +4.2% |

51층 × 0.02 ms = 토큰당 0.02 ms. 15 tok/s decode(~67 ms/tok)의
~0.03%. 아티팩트 재빌드는 51텐서 ≈ 284 MiB + owner 재시작이 필요하고
이득이 그 비용을 못 이긴다. 커널/후보 게이트/owner 모두 불변.
로그: `scratch/motif3-opt-v062/logs/c8-shexp-down.txt`.

닫힌 길 추가: shexp-down K=1280 aligned. 다음 게이트는 256K 센티널.

### §A.14 256K 직렬 센티널 (통과, 2026-08-21 17:30–17:49 KST)

엔진 팁 `2c81427` (커널은 `a09ff4f` FATTN TK=32까지). owner는 기존
tmux `motif3-v062-owner` (`--reserve-gb 24`). 워커는 직렬
`DS4_SERVER_COALESCE_MAX=1`, `-c 262144`, 공식
`context-262144-server.txt` + system, thinking off, greedy,
`max_tokens=64`. 스크립트 `scratch/motif3-opt-v062/sentinel-256k.sh`.

| 항목 | 값 |
|---|---|
| prompt | **262,080** (cached=0) |
| prefill | **1,098.433 s; 238.59 tok/s** |
| decode | **43 in 7.205 s; 5.97 tok/s** |
| finish | `stop`; total 262,123 |
| 센티널 | **ALL-EXACT** 3/3 JSON |
| route | `openai_chat.serial=1` |
| KV | 4.119 GiB (4,422,546,432 B) |
| worker | 10,429 MiB; available 11–12 GiB |
| clocks | 2,411–2,496 MHz (611 pin 없음) |
| VmSwap | owner 0, worker 0 |

공개 `v0.5.6.3-dfm` 행(175.61 / 2.52) 대비 prefill **+35.9%**,
decode **+137%**. HG16+L1TEX가 256K 깊이에서 드러난 결과와 맞다.
245,760에서 멈추지 않았고 decode가 실제 43토큰을 냈다. 동시 256K
뱅크는 주장하지 않는다.

증거 SHA-256:
`sent-256k-response.json`
`f4aafb4c969c46889daceb64feb01177c4682e75efff555a6539202f78cd42aa`,
`server-256k.log`
`4f6e2bdb9c1cf607bf4d02da2575c73e0b9af0fbf103923d1907eb16cd711096`.
요약: `scratch/motif3-opt-v062/logs/sent-256k-summary.txt`.

상태: **Entrpi `v0.5.6.3` 위의 통합 worktree에서 DeepSeek, Solar Open2,
K-EXAONE, Motif-3를 같은 `ds4-server -m <GGUF>` 형태로 실모델 로드했다.
Solar와 K-EXAONE은 2-bank continuous 요청, DeepSeek은 DSpark 자동 부착,
Motif는 안전한 native serial session과 4개 API surface를 통과했다. Motif
weight-owner의 기본 IQ2/Q2K aligned artifact를 384-expert 전용 경로에서 직접
소비하도록 수정해 illegal access를 해소했고 실제 prefill/decode를 완료했다.
현재 대형 프로세스는 모두 종료했으며 `clear_cache` 후 119 GiB available을
확인했다. 최종 source/test/doc review와 CPU/CUDA 회귀를 완료했고
`v0.5.6.3-dfm` Git publish와 HF collection card 갱신도 완료했다.
다음 단계는 private handoff checkpoint, 이후 nsys→ncu 기반
Motif 최적화와 strict 256K다.
Spark 256K 통과 주장은 아직 금지다.**

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

16. **통합 worktree 확정과 EXAONE 현행 serving 이식** —
    - 실제 통합 경로를
      `/home/sunghoon/workspace/ds4-exaone/ds4-model-families-v0563`, branch
      `feature/model-families-v0563`로 확정했다.
    - Motif v0.5.6.3 port 위에 Solar Open2 v0.5.6.3 라인을 merge하고,
      K-EXAONE loader/tokenizer/chat/GPU graph를 model family로 편입했다.
    - K-EXAONE은 Solar의 공통 continuous lifecycle을 따르되, LLLG KV는
      bank별로 유지하고 prefill scratch와 weight-bound decode stage를 공유한다.
      exact-frontier warm/fork, 2-row batched decode, 사전 메모리 fit을 연결했다.
    - EXAONE durable bank payload는 wire layout이 없으므로 명시적으로
      unavailable이며 DeepSeek walker로 잘못 직렬화하지 않는다.

17. **Motif 실모델 VMM owner와 illegal-access root cause** —
    - dry-run: logical 87.69 GiB, allocation 87.83 GiB, reserve 24 GiB,
      free 약 119.31 GiB로 통과했다.
    - 기본 owner는 raw 7.00 GiB + IQ2/Q2K aligned artifact 80.68 GiB,
      207 POSIX FD로 ready가 됐다.
    - 첫 worker는 layer 11에서 `ds4_mmq_iq2_xxs_moe_pair` illegal access.
      `CUDA_LAUNCH_BLOCKING=1`로 raw expert pointer를 읽는 Motif 전용 MoE와,
      raw tensor를 대체한 aligned manifest의 계약 충돌로 격리했다.
    - IQ2/Q2K repack을 모두 끈 raw owner에서는 같은 모델이 prefill/decode를
      통과해 모델/latent lifecycle 자체가 아니라 artifact dispatch 문제임을
      확인했다.

18. **Motif 384-expert aligned MoE 수정과 회귀** —
    - `ds4_gpu_motif3_routed_moe_batch_tensor`가 runtime 384 experts로
      `CUDA_DERIVED_IQ2_XXS_ALIGNED_MOE`와
      `CUDA_DERIVED_Q2_K_ALIGNED_MOE`를 resolve한다.
    - gate/up은 `ds4_mmq_iq2_xxs_moe_pair_soa`, down은
      `ds4_mmq_q2_K_moe_soa`를 직접 사용한다. aligned tile opt-out 시 기존
      persistent derepack scratch, artifact가 없을 때 raw VMM range로 fallback한다.
    - 기본 aligned owner에서 blocking/non-blocking 실제 prefill/decode를 모두
      통과했고 로그로 `IQ2=1 Q2K=1 experts=384` engagement를 확인했다.
    - Motif CUDA 6그룹, model-family kernel, EXAONE kernel, Solar KDA/chunk/KV/
      output gate synthetic 회귀도 모두 통과했다.

19. **남은 cross-family 잘못된 dispatch 차단** —
    - Motif generate/session decode가 DeepSeek raw-SWA graph로 떨어지던 경로를
      native public session과 latent decode로 연결했다.
    - Motif는 전용 persistent bank가 생기기 전까지
      `ds4_engine_supports_batching=false`로 두어 DeepSeek bank graph를 절대
      할당하지 않고 server serial lane을 사용한다.
    - DeepSeek 전용 graph/head/prompt/imatrix 진단은 family gate로 닫았다.
    - MTP/DSpark가 Solar, K-EXAONE, Motif에 붙으면 engine open에서 즉시
      명시적 오류를 내도록 했으며 실제 Motif+`DS4_DSPARK_MODEL` 거부를
      exit 1로 확인했다.

20. **네 모델 실모델 공통-command/API 회귀** —
    - 공통 형식은
      `DS4_CUDA_WEIGHT_IPC_MANIFEST=... ./ds4-server -m <GGUF> --cuda -c 2048`.
      모델마다 바뀐 것은 GGUF와 해당 owner manifest뿐이다.
    - DeepSeek V4 Flash: base 80.76 GiB + DSpark 6.49 GiB, aligned 72.56 GiB.
      sibling DSpark 자동 부착, Chat 1건 완료/실패 0.
    - Solar Open2: 11 shards 88.97 GiB, aligned IQ2 32.23 GiB. persistent
      2 banks에서 동시 Chat 2건 모두 continuous, 실패 0.
    - K-EXAONE: 3 shards 85.56 GiB, aligned IQ2 30.16 GiB. persistent
      2 banks에서 동시 Chat 2건 모두 continuous, 실패 0.
    - Motif-3: `/v1/models`, Chat, Completions, Responses, Anthropic Messages
      각 1건 완료, 총 요청 실패 0, 종료 후 inflight 0. 현재 serial lane임을
      `/v1/stats`로 확인했다.
    - 위 값은 2K integration/lifecycle 증거이며 공개 성능 또는 long-context
      결과로 사용하지 않는다.

21. **메모리 정리와 Entrpi sync 확인** —
    - 각 family 전환마다 worker→owner 순서로 Ctrl-C, PID/port 소멸,
      `nvtop` compute process 0을 확인한 뒤 `/usr/local/bin/clear_cache`를
      실행했다. 최종 `free -h`는 119 GiB available, swap 사용 약 737 MiB다.
    - `git fetch upstream --tags --prune` 결과 Entrpi 최신 serving tag/branch는
      계속 `v0.5.6.3`/`b9c97ad`; 통합 HEAD는
      `upstream/batched-serving` 대비 behind 0이다.

22. **publish 전 최종 source/test/doc review** —
    - `git diff --check` 통과, conflict marker 0건. 생성 test binary는
      untracked로 분리했고 source/doc만 stage 대상으로 확정했다.
    - `make cpu`, `make cuda-spark`, server unit, model-family kernel,
      Motif reference/loader/tokenizer/CUDA 6그룹, EXAONE kernel, Solar
      KDA/prefill/chunk/gates/KV, `make cuda-regression`, quantizer build가
      모두 통과했다. 남은 compiler warning은 clean Entrpi
      `v0.5.6.3` baseline과 같다.
    - `cuobjdump --list-elf` 기준 `ds4-server`/`ds4_weight_server`의
      모든 cubin은 `sm_121a` 단일이다.
    - 네 family의 다른 state/runtime은 직접 분기로 남기고
      공통 wire contract만 공유했다. registry/plugin/graph-framework
      추상화는 추가하지 않아 Entrpi/antirez 스타일과 후속
      upstream PR 가능성을 보존했다.
    - root README와 `docs/ds4-dfm-model-families.md`는 영어로 작성했고,
      한국어는 사용자가 지정한
      `DFM (독자 파운데이션 모델, 독파모)` 풀이에만 사용했다.

23. **Git publish ref 계약** —
    - Entrpi 추적 branch와 DFM 산출물을 구분하기 위해
      `origin/batched-serving`은 이번 publish에서 덮어쓰지 않는다.
    - 검증된 통합 HEAD는 재현용
      `feature/model-families-v0563`, 이후 Entrpi 증분을 계속 받을
      moving branch `dfm`, 고정 release tag `v0.5.6.3-dfm`으로 같이
      publish한다. `origin/main`의 서로 다른 antirez line은 force-push하지
      않는다.
    - 공개 model card의 engine 링크는 모호한 branch tip 대신
      `https://github.com/Baekpica/ds4/tree/v0.5.6.3-dfm`을 사용해
      이 검증 release를 고정한다.

24. **Git integration publish 완료** —
    - 문서 갱신과 최종 회귀 완료 후 release commit
      `a388f92ae8803ebefd0af4d92dc6093e1f8db8f7`을
      `origin/feature/model-families-v0563`, `origin/dfm`, annotated tag
      `v0.5.6.3-dfm`으로 publish했다.
    - `git ls-remote`에서 두 branch와 tag peeled ref가 모두
      `a388f92ae8803ebefd0af4d92dc6093e1f8db8f7`을 가리키는 것을
      확인했다. GitHub API의 annotated tag object는
      `d7db4840560618fa4e48d39ad3cc27ac487ebbf2`다.
    - tag 생성 후 `ds4_server.o`를 재빌드해
      `./ds4-server --version` = `ds4-server v0.5.6.3-dfm`을 확인했다.

25. **HF collection card publish 전 문서 gate** —
    - `hf collections info` 기준 collection
      `Baekpica/ds4-mixed-quant-for-spark-6a79321cc8a55c35231d3ed3`의 model은
      Motif-3, Solar Open2 250B, K-EXAONE 236B A23B 세 개다.
    - 각 README의 소개/요약 직후에 동일한 영어 `## ds4-dfm`
      section을 추가했다. 한국어는
      `DFM (독자 파운데이션 모델, 독파모)` 풀이에만 사용했다.
    - section은 one Spark/128 GB unified memory, 명시적 C/CUDA family
      kernel, 공통 `ds4-server`/4 API surface, 고정 engine tag link를
      설명한다. 성능·context·quality는 각 card의 기존 검증 범위를
      넘지 않는다고 명시했다.
    - upload 대상은 README 3개뿐이며 새 local SHA-256은
      K-EXAONE `f41ccf78d866b35924964666bee0898234abb620a97d3e252f388ce6f96a15a1`,
      Motif-3 `0885d9c62ea6daa28629bfc31a6834fe6588f8b86fbee9b71a9c5e371d5b0b5a`,
      Solar `19565fe9d7b03d063a40f304bfc340089c213a2ee66bc344328cf12f0e4776ff`다.
      GGUF/asset/manifest는 갱신하지 않는다.

26. **HF collection card publish/재검증 완료** —
    - README 단일 파일만 각 model repo에 upload했다: Motif-3
      `843b57ab6da24ac2583118e10e633929617d535e`, Solar
      `75ecb81efd9f6e6f60f8aff39926b5788d318122`, K-EXAONE
      `af82aa9c9cd5af593e07435abfa13b0cf2cc71f6`.
    - 각 반환 commit을 `--revision` 으로 지정해 README를 새 디렉터리에
      재다운로드했고, `cmp`와 SHA-256이 §0.25의 local 작성본과
      세 파일 모두 byte-identical이었다.
    - remote section 위치는 Motif/Solar line 52, K-EXAONE line 45의
      `## ds4-dfm`이다. Motif의 `94.16 GB (87.70 GiB)` 소개와
      one-Spark 256K 미통과 표기는 그대로 보존했다.

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
| 양자화 | `MQ87-88-FIT`, 94.16 GB / 87.70 GiB, 11 shards |
| 병합 GGUF | `/home/sunghoon/workspace/ds4-exaone/models/Motif-3-Mixed-Quant-GGUF/Motif-3-MQ87-88-FIT.gguf` |
| 병합 GGUF 크기 | `94,162,541,472` bytes |
| 병합 GGUF SHA-256 | `15755a735753bc1396e5ffa539e65a779a4fd769e8833360a4d743c4c60c2f25` |
| Q8_0 reference revision | `5c266c95bf8c8d822d50e5e1cce9d108eaadb2af` |
| 재현 저장소 | `f2b86cf286bd99fdb245e07ceba5710e297dca2f` |
| 수신 ds4 Motif 브랜치 | `d878ea1a1d67bc0f0bd60e20e75b4a011aa2d8d9` |
| 공식 모델 소스 | `Motif-Technologies/Motif-3@ccceb1a5fd7b5eb32e47841216b3caf5666c07bc` |
| 공식 vLLM 참고 구현 | `MotifTechnologies/vllm@4cd9eb4129883565e69d508038d783d59ee01867` |
| private handoff | `hf://buckets/Baekpica/motif-3-spark-handoff` |

로컬 private handoff는 145 files, 2,465,794,154 bytes다. 원격 전체 재다운로드 후 `manifests/SHA256SUMS`의 144개 항목이 모두 통과했고 private 상태 및 중복 GGUF 0개를 확인했다. 공개 카드의 decimal 크기는 `94.16 GB`로 표기하고 `87 GB` 또는 `88 GB`로 쓰지 않는다. binary 단위가 필요하면 `87.70 GiB`를 함께 쓴다.

고정 revision의 weight를 Spark 작업 중 재양자화하거나 조용히 교체하지 않는다. 11개 원본 shard도 보존한다.

## 3. 검증 주장 경계

H200에서는 2K, 32K, 64K, 128K correctness가 통과했다. 256K 두 번의 시도는 각각 245,760 및 106,496 prompt token에서 멈췄고 decode가 수행되지 않았다.

따라서 다음 문장은 금지한다.

- H200 256K 통과
- Spark 256K 통과
- 256K serving 확인

Spark에서 정확히 262,144-token prefill을 완료하고 실제 decode token을 얻은 뒤에만 256K 통과로 기록한다. 포트 listen이나 prefill-only는 통과가 아니다.

## 4. 현재 호스트와 메모리 상태

| 항목 | 2026-08-14 17:02 KST 현재 값 |
|---|---|
| Host | `thinkstationpgx-8abc` |
| Kernel | `6.17.0-1029-nvidia` |
| GPU | NVIDIA GB10, compute capability 12.1 |
| Native build target | `sm_121a` |
| NVIDIA driver | `610.43.02` |
| CUDA compiler | 13.3 |
| Unified memory | 121 GiB total, 119 GiB available |
| Swap | 15 GiB total, 약 737 MiB used |
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
/home/sunghoon/workspace/ds4-exaone/ds4-model-families-v0563
branch: feature/model-families-v0563
base: b9c97adfb4a921096a4df24e672599067269a7e2 (v0.5.6.3)
```

현재 remote:

```text
origin   https://github.com/Baekpica/ds4.git
upstream https://github.com/Entrpi/ds4.git
```

향후 실제 최상위 upstream 검토 시 사용자가 지정한 `https://github.com/antirez/ds4`와의 관계를 명시적으로 정리한다. 현재 remote를 작업 도중 조용히 바꾸지 않는다.

### 현재 통합 경계

원본 네 commit은 v0.5.6.3 위에 다음 로컬 SHA로 모두 이식됐다.

```text
01e1be0 Add native Motif-3 loader and latent CUDA runtime
b085b1c Document resident Motif-3 serving
ae0d3a2 Advertise Motif model ID from OpenAI server
64c67a0 Preserve complete question in Motif 256K gate
```

이후 `6699c8a`에서 Solar Open2 v0.5.6.3 라인을 merge했고, K-EXAONE model
family와 현행 common serving lifecycle commit이 그 위에 있다. `CHERRY_PICK_HEAD`는
없다. 현재 최종 EXAONE persistent banks, Motif aligned-MoE/dispatch 수정,
문서/version 변경은 publish 전 review를 위해 working tree에 있다. 생성된
테스트/weight-server 바이너리는 검증 산출물이므로 Git에 추가하지 않는다.

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

Solar와 K-EXAONE도 같은 `ds4-server`와 OpenAI Chat/Completions, Responses,
Anthropic surface를 사용한다. Solar/K-EXAONE은 family-native persistent bank,
DeepSeek은 upstream continuous graph, Motif는 native serial session으로 내부
state lifecycle만 다르다. 외부 실행/HTTP 계약은 모델 경로만 바꾸는 형태다.

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
현행 API로 compile/link 및 manifest safety gate를 통과했다. 이후 §0.17–20의
VMM owner/import로 full GGUF short prefill/decode와 공통 API까지 실행했다.
32K 이상 resident/long gate는 아직 실행하지 않았다.

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
DS4_CUDA_WEIGHT_IPC_SCOPE=base \
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

잠재적인 속도 목표는 Solar Mixed-Quant와 납득 가능한 같은 order의 TTFM/prefill/decode다. 20-token 개발 gate에서 raw owner는 prefill 14.17 tok/s,
decode 11.64 tok/s, aligned owner는 실행별 prefill 14.02–14.40 tok/s를 보였다.
decode는 2–8 token 표본이라 9.12–15.50 tok/s로 분산이 커 기준점이나 공개
성능으로 쓰지 않는다. nsys medium-prefill과 충분한 decode 표본 전에는
구체적인 달성 숫자나 성능 향상을 주장하지 않는다.

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

- HF private handoff의 이번 재개 로그/문서 checkpoint
- nsys/ncu profile
- 32K 이상 Spark 실행
- 256K 통과 주장

완료된 항목은 VMM owner dry-run/live, raw/aligned Motif worker, 네 family
공통-command 실제 API gate, 매 전환의 `clear_cache`, publish 전
source/test/doc review, Git branch/tag publish, 영어 model-card 갱신이다.
다음은 private HF handoff에 이 재개 문서 checkpoint를 보존한 뒤 Motif
owner를 다시 올려 §12의 nsys medium-prefill부터 재개한다.
