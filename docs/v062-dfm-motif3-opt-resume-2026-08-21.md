# Motif-3 / v0.6.2-dfm 최적화 재개 핸드오프

작성: 2026-08-21 12:20 KST. 다른 에이전트가 이 파일만 읽고 이어서
작업할 수 있게 현재 상태·증거·금지사항·다음 명령을 고정한다.
권위 문서:
`docs/ds4-dfm-model-families.md`,
`docs/motif-3-spark-optimization-handoff.md` §A.
이 파일과 §A가 숫자에서 어긋나면 **측정 CSV / ncu txt / git SHA**가 이긴다.

## 0. 한 줄 상태

Entrpi `v0.6.2` 흡수와 `v0.6.2-dfm` 태그는 끝났고, Motif nsys→A/B 사이클
1–3도 반영·푸시됐다. 사이클 4용 ncu 배치가 풀모델 standalone
`ds4-bench`를 올려 호스트 메모리 압력이 터졌고, 사용자가 11:47에
재부팅했다. 지금 호스트는 클린(118 GiB available, GPU 프로세스 0).
HG16 마이크로벤치 ncu는 유효하고, fattn 풀모델 ncu는 실패했다.

목표(원본 `/goal`)의 측정 게이트는 채워졌다. 사이클 1–3·5·6은 반영,
사이클 4·7·8은 원복/닫힘. 256K 직렬 센티널은 2026-08-21 17:49 KST에
통과(262,080 prefill 238.59 tok/s, 43 decode 5.97 tok/s, ALL-EXACT).
문서 커밋/`HEAD:dfm`/카드 반영은 아직이다. 태그 `v0.6.2-dfm`은 고정.

## 1. 원본 목표 (축소 금지)

- 모델: `models/Motif-3-Mixed-Quant-GGUF/`
  (HF `Baekpica/Motif-3-Mixed-Quant-GGUF`, 핀은 병합
  `Motif-3-MQ87-88-FIT.gguf`)
- 엔진: `Baekpica/ds4` `dfm` (`git push origin HEAD:dfm`)
- `/init`으로 워크스페이스 `CLAUDE.md` 갱신
- Entrpi `v0.6.2` 흡수, 태그 `v0.6.0-dfm` → `v0.6.2-dfm`
- 충돌 시 서빙 기능은 upstream, 패밀리 커널은 dfm
- Prefill/Decode 절대 속도 + 깊은 컨텍스트에서 급락 완화
- 절차: nsys → 큰 병목만 ncu → A/B → 반영/원복. 최소 3회 + 여지 소진
- 상주 작업: nvtop+htop, 종료 후 `clear_cache`. **ncu는 owner down**
- Solar 참조: 같은 엔진 8K decode 19.05 tok/s (`b2e52b9`)

## 2. 워크트리 / 태그 / 원격

| 항목 | 값 |
|---|---|
| 활성 워크트리 | `ds4-model-families-v0563/` (`feature/model-families-v0563`) |
| 작업 카피 | `scratch/ds4-v062-dfm-sync/` (`feature/v062-dfm-sync`, 같은 HEAD) |
| HEAD / `origin/dfm` | `6eaef3425a7cfdd3235909989ca3cab81f83e29c` |
| 태그 | `v0.6.2-dfm` (존재, 머지 `c72eb50` 계열) |
| 로컬 `dfm` 브랜치 | 오래됨. **쓰지 말 것.** 푸시는 `git push origin HEAD:dfm` |
| 게시 순서 | 핸드오프 먼저 → `git diff --check` → 테스트 → 커밋 → `HEAD:dfm` |

핵심 커밋:

| SHA | 내용 |
|---|---|
| `c72eb50` | Entrpi v0.6.2 머지 → v0.6.2-dfm |
| `5f9bc86` | Solar standalone loader / forward 테스트 링크 수리 |
| `de36c64` | family bank에 `bank_last_use` 할당 (첫  gener 세그폴트) |
| `70d5823` | v0.6.2-dfm 통합 증거 |
| `d03bd89` | 사이클 1 HG16 latent flash-decode |
| `b0db5a1` | 사이클 2 SWA prefill → windowed HMMA |
| `91823ca` | 사이클 3 Motif MoE D2R floor=8 |
| `6eaef34` | §A 사이클 1–3 기록 (현재 HEAD) |

## 3. 완료된 것 (증거와 함께)

### 3.1 CLAUDE.md /init

워크스페이스 `CLAUDE.md`는 4라인 complete + dots3 통합 + ncu/owner-down
주의 + 호스트 2026-08-21로 갱신됨. 워크트리 표의 “last tag
`v0.6.0-dfm`” 한 줄은 이 재개 시점에 아직 stale — `v0.6.2-dfm`으로
고쳐야 한다.

### 3.2 upstream 흡수

충돌 파일: `CHANGELOG.md`, `README.md`, `VERSION`, `ds4.c`,
`ds4_server.c`. `ds4_cuda.cu`는 자동 병합. 규칙: 서빙=upstream,
패밀리 커널=dfm.

머지 후 추가 수리 2건:

1. split-GGUF content fingerprint — owner가 논리 매핑을 지문
   (`tools/ds4_weight_server.cu`). Motif 단일 파일 import는
   `content identity verified`.
2. `bank_last_use` — Solar/EXAONE/Motif persistent-bank create/free.
   없으면 첫 cold admission에서 워커 세그폴트 (gdb 재현).

게이트(머지 직후): server unit, extractor, split-GGUF,
`test-model-family-kernels`, `test-mmq-parity`, Motif 6그룹,
EXAONE kernels/ref, Solar loader/tokenizer/KDA/…/forward, dots3
loader/tokenizer, `make cuda-regression`. 라이브 owner+worker: API 4면
200, served=4/failed=0, continuous, `-c 2048` 32 banks.

### 3.3 베이스라인과 3사이클 숫자

계기: `scratch/ds4-v062-dfm-sync/ds4-bench`, aligned-Q8 VMM owner
(`--reserve-gb 24`), 코퍼스
`motif-3-spark-handoff/fixtures/long-context/context-32768.txt`,
greedy·no-think. 스크립트: `scratch/motif3-opt-v062/run-*.sh`.

| cell | v0.6.2-dfm base | 사이클1 HG16 | 사이클2 SWA | 사이클3 D2R | 누적 Δ |
|---|---:|---:|---:|---:|---:|
| 8K prefill | 519.90 / 518.02 | 519.6 | **620.9 / 620.0** | 621.66 | **+19%** |
| 32K prefill | 445.03 | 445.0 | **526.6** | 524.94 | **+18%** |
| 8K decode | 12.62 | 13.14 | 13.14 | **15.10** | **+19.6%** |
| 32K decode | 9.68 | 11.28 | 11.27 | **12.73** | **+31.5%** |

CSV: `scratch/motif3-opt-v062/logs/{base,hg16,swaexp,d2rdef}-*.csv`.
32K 센티널 3/3 exact는 사이클 1·2·3 모두. HTTP decode는 사이클3에서
12.5 tok/s. 공개 시절 32K decode 8.96 대비 약 +42%.

사이클 1 greedy 텍스트는 base 대비 토큰 분기(softmax 분할 순서).
게이트는 센티널이지 비트 동일 텍스트가 아니다. 사이클 1 frontier
logits는 prefill 비트 동일. 사이클 2 8K frontier logits는 rel_rms
1.4e-1 (argmax·top-4 불변) — SWA가 absorbed-latent 대신 official
expanded 경계를 쓴 대가. full 레이어와 같은 트레이드.

### 3.4 nsys (사이클 전, 32K gen128)

프로파일: `scratch/motif3-opt-v062/nsys/` 와
`logs/nsys-32k.log`. sqlite에서 **마지막 ~13.0 s decode 창만** 집계.

Decode 창: GPU busy 96.6%, ~103.5 ms/token.

| % | kernel | 비고 |
|---:|---|---|
| 32.0 | `motif3_latent_attention_bf16_decode_hg_partial` | 유일한 깊이 비례항. 14 full layer × 당시 10 head-group KV 재독 |
| 22.5 | `mul_mat_q` ×2 MoE | D2R 게이트에 걸려 classic MMQ로 낙하 → 사이클 3 |
| 18.9 | `q8_0_aligned_dense_vec` | roofline 근접 |
| 4.2 | `rms_norm_weight` | 157회/token |
| 3.3 / 3.2 / 2.0 | qk_absorb / q8 pair / value_project | |

Prefill 32K (74.6 s): fattn_hmma 15.7%, q8 pair 12.3%, value_project
8.7% (대부분 당시 SWA latent), gateup_iq2 8.1%, SWA latent generic
~6.5%, f32_to_bf16 5.4% (43,884회), qk_absorb 4.5%, round_bf16 3.2%
(66,650회). 사이클 2가 SWA latent/value_project를 HMMA로 회수.

**이 nsys는 사이클 1–3 이전이다.** 다음 병목 주장은 재-nsys 없이는
간접 증거다.

## 4. 리부팅 원인 (검증됨)

시간선 (저널 + 파일):

1. 사이클 3 푸시 `6eaef34`. owner를 내리고 `clear_cache` (11:32
   `drop-caches`). 당시 118 GiB available, GPU 프로세스 0.
2. HG16 **마이크로벤치** ncu 성공:
   `scratch/motif3-opt-v062/ncu/hg16-256k.txt` (16 passes × 2, 완전).
3. `scratch/motif3-opt-v062/run-ncu-real.sh`가 **owner 없이**
   standalone `ds4-bench`로 풀 Motif를 올림. 로그
   `ncu/fattn-16k.txt`:
   - in-process aligned artifacts **86.07 GiB / 24.7 s**
   - `device 93.08 GiB` + 87.70 GiB mmap
   - 그 위에서 ncu가 `motif3_fattn_hmma_kernel` 프로파일 시작
   - 파일은 `Profiling "motif3_fattn_hmma_kernel": 0%`에서 끊김
4. 11:35 `systemd-oomd`가 user.slice 메모리 압력 72.41% > 50% for >20s
   로 `tmux-spawn-ebdd69f5-...` 킬. 커널 OOM killer 기록은 없음
   (cgroup pressure kill). 스냅샷 cgroup RSS는 작음 — GB10 unified
   memory의 CUDA 할당이 cgroup RSS에 잘 안 잡힌 전형적인 패턴.
5. tmux 서버가 11:48에 다시 뜨고, 사용자 재부팅은 **11:47**.

직접 원인: “ncu는 owner down”을 지켰지만, standalone `ds4-bench`가
owner와 같은 86 GiB artifact를 **프로세스 안에** 다시 만든 뒤 ncu
카운터를 겹쳤다. owner+ncu OOM과 같은 클래스, 경로만 다름.

**금지 (재발 방지)**

- ncu를 풀모델 `ds4-bench` / `ds4-server`에 걸지 말 것 (IPC든
  standalone든). 이 호스트에서 모델 상주 + ncu는 문서화된 OOM.
- ncu는 마이크로벤치 또는 커널-only fixture만. owner는 반드시 down.
- ncu 직전에 `drop-caches`/`clear_cache`를 때리지 말 것 (페이지
  캐시 재충전 = oomd pressure).
- Chromium 등 데스크톱과 ncu를 겹치지 말 것.
- `DS4_CUDA_COPY_MODEL_CHUNKED=1` 금지 (문서화된 OOM).

## 5. 재부팅 직후 호스트 (2026-08-21 12:12 KST)

```
uptime          25 min (boot 11:47)
Mem available   118 GiB / 121 GiB, swap 0
GPU apps        none
ds4/ncu/nsys    none
tmux            sunghoon-0 (모니터링 그룹만)
driver/CUDA     610.43.02 / 13.3, kernel 6.17.0-1031-nvidia
```

다음 상주 작업 전: `nvtop` Compute + `free -h` 재확인. owner를 올릴
때는 dry-run 먼저, `broker listening` + `ready manifest=` 후에만
워커/벤치.

## 6. HG16 ncu 결과 (유효, 사이클 4 입력)

대상: `hg_partial_t<2,16>` = 현행 기본 경로
(`M3_ATTN_HG_HEADS=16`, `WARP_HEADS=2`, `ROWS=16`, split=64).
파일: `scratch/motif3-opt-v062/ncu/hg16-256k.txt`.

256K 마이크로벤치 스캔(같은 파일 상단): shipped ref 22.445 ms,
승자 `wh2 rows16 split64` 10.004 ms (**2.24x**), rms 1.1e-6.

Speed of Light (승자 CTA):

| metric | value |
|---|---|
| Duration | 10.34 ms |
| SM / Memory throughput | **둘 다 63.35%** (balanced) |
| L1/TEX | 66.62% |
| L2 | 11.17% (여유) |
| Theoretical occupancy | 33.33% (16 warps/SM = 2 CTA) |
| Achieved occupancy | 30.90% |
| Block limit regs | **2** |
| Block limit smem | **2** |
| Warp cycles / issued | 9.96 |
| L1TEX scoreboard stall | **33.8%** of issue gap (local speedup 추정 33.8%) |
| Occupancy local speedup 추정 | 66.67% (100% occ 가정, 비현실) |
| Active threads / warp | 27.43 / 32 |

커널 (`ds4_cuda.cu` ≈34363): 각 스레드가 `out0[16]`, `out1[16]`으로
**16헤드 전부**를 accumulate한다. 워프는 스코어를 자기 2헤드만
계산하지만 value 단계는 전 헤드를 돈다. 레지스터 폭주 + value FMA
8배의 원인. smem은 `latent_sm[16][516]` float ≈ 33 KiB 등 합
~38 KiB → CTA 2개/SM.

사이클 4 (원복, 2026-08-21 12:30 KST): 워프-로컬 쓰기는 헤드당 64/512
차원만 덮는다. 첫 마이크로벤치는 `partials`를 안 지워 rms 0 허위
양성이 났고, `test-motif3-cuda`가 `got 0`으로 잡았다. 정직 재측정
(reload-smem / bf16-smem)은 1.00x 또는 0.97x. 커널은 사이클 3 상태로
원복, 픽스처 재통과. HG16 점유율 변형은 닫힌 길.

## 7. 다음에 할 일 (순서 고정)

사이클 6 반영 (FATTN HMMA TK=32). Owner는 tmux `motif3-v062-owner`에
살아 있음. ncu를 하려면 먼저 이 owner를 내리고 `clear_cache`할 것.

1. Q8 dense vec occupancy / HG L1TEX / FATTN TK=64 / Motif GQA-pair /
   Q8 pair tok8 / shexp-down K=1280 aligned 는 닫힘. ncu는
   마이크로벤치만. shexp-down 합성 +4.2%는 토큰당 0.02 ms
   (~0.03% e2e)라 owner 재빌드 가치가 없다.
2. 256K 직렬 센티널은 통과. 증거
   `scratch/motif3-opt-v062/logs/sent-256k-summary.txt`.
3. 문서 커밋 → `HEAD:dfm` (요청 시). 태그 `v0.6.2-dfm`은 통합
   컷에 고정. 모델 카드 256K 행은 아직 `v0.5.6.3-dfm` 숫자.

## 8. 재개 명령

활성 엔진 트리는 `scratch/ds4-v062-dfm-sync` (빌드 산출물 있음) 또는
`ds4-model-families-v0563` (같은 SHA). `make cuda-spark` (gencode
`compute_121a,sm_121a`). cubin이 `sm_121a`인지 확인.

```sh
# 위생
pgrep -a -f 'ds4|ncu|nsys' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
free -h
# 프로세스 0일 때만
/usr/local/bin/clear_cache
free -h
```

Owner (ncu가 아닐 때, tmux 디태치, 가드 래퍼):

```sh
# dry-run 먼저
cd /home/sunghoon/workspace/ds4-exaone/scratch/ds4-v062-dfm-sync
MODEL=/home/sunghoon/workspace/ds4-exaone/models/Motif-3-Mixed-Quant-GGUF/Motif-3-MQ87-88-FIT.gguf
RUN=/home/sunghoon/workspace/ds4-exaone/scratch/motif-3-v062-runtime
./ds4_weight_server --base "$MODEL" --manifest "$RUN/weights.manifest" \
  --backend vmm --scope base --reserve-gb 24 --dry-run
# 통과 후 같은 커맨드에서 --dry-run 제거, tmux + scripts/guarded-run.sh
# ready: broker listening + ready manifest=... + fingerprint 37f52a40c21047c4
```

벤치 (owner ready 후):

```sh
export DS4_CUDA_WEIGHT_IPC_MANIFEST=$RUN/weights.manifest
export DS4_CUDA_WEIGHT_IPC_SCOPE=base
P32=/home/sunghoon/workspace/ds4-exaone/motif-3-spark-handoff/fixtures/long-context/context-32768.txt
# 8K decode
./ds4-bench -m "$MODEL" --cuda --prompt-file "$P32" \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8321 --gen-tokens 64 --csv ...
# 32K decode
./ds4-bench -m "$MODEL" --cuda --prompt-file "$P32" \
  --ctx-start 32743 --ctx-max 32743 --ctx-alloc 32912 --gen-tokens 64 --csv ...
# 8K prefill-only
./ds4-bench -m "$MODEL" --cuda --prompt-file "$P32" \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8257 --gen-tokens 0 --csv ...
```

픽스처: `DS4_MOTIF3_MODEL`을 병합 GGUF로. scratch 워크트리는
`../motif-3-mixed-ds4` 상대경로가 깨지므로 변수를 반드시 준다.
챗 벤치는 `"thinking": {"type": "disabled"}`, 스트림은
`stream_options.include_usage=true`.

ncu 예 (마이크로벤치만):

```sh
# owner down 확인 후
/usr/local/cuda-13.3/bin/ncu -k regex:hg_partial \
  --section SpeedOfLight --section Occupancy --section WarpStateStats \
  ./bench-hg-occ 32768
```

## 9. 경로 지도

| 무엇 | 어디 |
|---|---|
| 엔진 소스 | `scratch/ds4-v062-dfm-sync/{ds4.c,ds4_cuda.cu,ds4_server.c,tools/ds4_weight_server.cu}` |
| 권위 통합 문서 | `ds4-model-families-v0563/docs/ds4-dfm-model-families.md` |
| Motif 최적화 역사 | `ds4-model-families-v0563/docs/motif-3-spark-optimization-handoff.md` §A |
| 이번 세션 로그/CSV/ncu | `scratch/motif3-opt-v062/` |
| Owner 런타임 | `scratch/motif-3-v062-runtime/` (manifest, owner.log) |
| 모델 | `models/Motif-3-Mixed-Quant-GGUF/Motif-3-MQ87-88-FIT.gguf` |
| 가드 래퍼 | `scripts/guarded-run.sh` |
| 엔진 규율 | 각 워크트리 `AGENT.md` (C only, 정확성 우선, 플래그로 의미 분기 금지) |

## 10. 하지 말 것

- 로컬 `dfm` 브랜치 체크아웃/리셋
- `ds4-motif-3/` 덮어쓰기 (preservation)
- owner 떠 있는 채 ncu
- 풀모델 ncu
- 3사이클을 “목표 완료”로 선언. 재-nsys + 여지 소진 + (가능하면)
  256K가 남았다.
- 공개 테이블에 사이클 3 숫자를 새 published Motif 8K로 올리는 것 —
  아직 최종 게이트/문서 동기 전. §A와 이 파일이 작업 기록이다.
