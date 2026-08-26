# rust-host 캠페인 현황 (2026-08-26 KST)

이 문서는 `rust-host`에서 Phase 2–8 잔여 슬라이스를 닫고 §10 비교까지 돌린
뒤의 **작업 기록 + 남은 일**이다. 이름 승격(`cb11c0b`)과 post-promote
§10 GREEN 이후 `STATUS.md` / `PARITY_MATRIX.md`를 이 스탬프에서 다시
그렸다. Soak와 `SPLIT_READINESS.md`는 아직이다.

| 항목 | 값 |
|---|---|
| Branch | `rust-host` (`origin/rust-host`와 동기화) |
| HEAD | `cb11c0b` `Make: Promote Rust host to the default binary names` |
| C golden | `v0.6.3-dfm` (`516456fe35510e4fb8350396c9d88807ac1f760b`) |
| 기본 바이너리 | **Rust** (`ds4` / `ds4-server` / `ds4-bench` / `ds4-agent`). C 오라클 `*-c`. `ds4-eval`은 C. |
| §10 pre-promote | GREEN (`11c59e4`) |
| §10 post-promote | **GREEN** (`cb11c0b`, `task-54-rerun-cb11c0b.txt`) |
| Phase 9 names | **promoted** (`cb11c0b`). Soak / `SPLIT_READINESS.md` / dfm-rs 시딩은 아직. |
| 플랜 | `.omo/plans/rust-host-remaining-to-phase-9.md` (todos 1–61 `[x]`, F1–F4는 최종 감사) |
| 작업트리 재검증 | 2026-08-26 22:40 KST. post-promote §10 GREEN (§3.16). |

증거는 `.omo/evidence/task-*-rust-host-remaining-to-phase-9.txt`에 있다
(untracked, `git add` 금지 목록).

---

## 1. 한 줄 결론

이름 승격 완료. pre-promote와 post-promote §10 모두 **GREEN**
(`11c59e4`, `cb11c0b`). 기본 이름은 Rust, C는 `*-c`. PASS* = E-2..E-6.
다음: T1.8 soak 2h. `SPLIT_READINESS.md` / dfm-rs 시딩은 soak 전 금지.

`ds4-c` SHA는 T1.6 `proof-cuda-opp-c` relink로 e9efb70e→95fa6828
(여전히 C CLI). 다른 오라클·Rust 기본 이름은 T1.5 매핑 유지.

대략:

```text
문서/프로세스/금지 항목 준수     ~95%
Phase 0–3 골격 + shadow         ~80%
Phase 4–8 호스트 코드           ~80%  (quote FFI + trim wrap + HostKvView; TickOp KEEP)
§10 integrated gate             ~70%  (Motif evict/periodic/surfaces 추가, 전체 NOT_GREEN)
Phase 9 + dfm-rs DoD            ~10%
```

---

## 2. 하지 않은 것 (올바름)

- CUDA/MMQ 커널 rewrite, tokenizer/MTP/checkpoint format 변경 없음
- Tokio / Axum / async scheduler 없음
- `git merge dfm` 없음
- production 바이너리 rename 없음 (`ds4` 등은 여전히 C ELF)
- `docs/rust-migration/SPLIT_READINESS.md` 없음
- `Baekpica/dfm-rs` / `gh repo create` 없음
- `STATUS.md`를 “코드가 있으니 green”으로 다시 쓰지 않음
- `.omo/`, 로컬 rust-migration 초안, `ds4-agent-rs`, `tests/test_*` 바이너리를
  커밋하지 않음

---

## 3. 이번에 닫힌 구현 (crate / cheap-gate)

플랜 todos **1–52, 59–61**은 `[x]`. 의미만 묶으면:

### 3.1 Distributed worker (8.4)

- 데이터 리슨 accept, hop `serve_once`, `reconnect_local` (`!Send`)
- `ds4-rs` / `ds4-server-rs` `--role worker`가 `assemble_worker`를 탐
  (`run_distributed_worker` 제거). `ds4-server`는 `ds4-cli`에 의존하지 않음
- `serve_prefetch` + `Session` 금지. `!Send` receive prefetch
  (`DS4_DIST_DISABLE_WORKER_PREFETCH` unset=on)
- hop relay / telemetry 배선
- C `ds4_bridge_model_run_distributed_worker`는 oracle용으로 유지

### 3.2 Static lane (8.5)

- `LANE_STATIC` → `BatchCtx::generate_static` (serial 우회 아님)
- C `n>=2`, ragged, overflow `"out of memory"`
- owner FIFO coalesce (`try_recv`, no Tokio)
- 터미널 `finish_reason` / queue timing

**라이브 주의:** 짧은 프롬프트 + `have_cont`이면 C도 continuous를 고른다.
static 셀을 “짧은 greedy POST”로 치면 C/Rust 둘 다 static이 아니다.
이건 라우터 버그가 아니라 픽스처 설계 문제다. static을 다시 치려면
`prompt_len > seq_cap` 이거나 `have_cont=false`인 요청이 필요하다.

### 3.3 Rolling continuous (8.5)

- `OneJob` 제거, `RollDriver` / `ContRoll` / `generate_pair`
- fork / pin / `BankEvict` hold
- memgov charge (`AdmitAlways` + C reason map; live quote는 아직 C)
- prefill interleave: `tick_roll_prefill`가 `serve_pair`에서 호출됨.
  실제 청크 루프는 여전히 C `continuous_generate`
- OpenAI chat no-think no-tools bank 테스트 green
- **라이브:** barrier 2클라이언트 200/stop, 503 없음 (C·Rust)

### 3.4 Tools / KV / reclaim / stream (8.5)

- `cont_tools_anthropic` / `_responses` C default ON, env=`0`만 off
- streaming Anthropic tool-turn → bank-owned continuation, mismatch 409
- KVC thinking bit + `KTM\x01` persist/restore (crate + C oracle)
- rolling periodic 10000 → 10240 (`bank_checkpoint_due`)
- live bank-evict persist (`KvReason::BankEvict`, pin skip)
- serial emergency reclaim 게이트가 `run_serial` 앞에 붙음
  (3.9: quote FFI + C `trim_free` wrap. 호스트 두 번째 trimmer 없음)
- Anthropic/Responses stream disconnect / backpressure (panic 없음)

### 3.5 Agent / CLI (8.2 / 8.3)

- TUI, `/save` `/list` `/switch` `/del` `/strip`
- interactive write/edit `Ask`
- `--non-interactive` 없이도 parse Ok
- bash/compact, worker reject, dir-steering 회귀 green
- TTY thinking grey (`\x1b[90m`), pipe는 ESC 없음
- `--dump-logits` / `--dump-tokens`
- C help/CSV 커버 (C에 없는 새 public 플래그 추가 안 함)

### 3.6 Ownership / ABI (8.6)

- `ds4_bridge_model_*` keep vs move inventory
- Drop: session → model → C `ds4_engine_close` (MTP, DSpark, base)
- `SiblingAttach`가 MTP/DSpark bind-map lifecycle 소유
  (mmap open은 여전히 C)
- host ledger가 `Session::ctx()` public truth
- CLI `unsafe {` 27곳 SAFETY 분류. server 0
- bridge freeze: 새 `ds4_bridge_*`는 create/load/session/prefill/decode/KV/destroy만
- Metal: Linux라 SKIP

### 3.7 게이트 직후 핫픽스 (라이브 FAIL / F2)

| 커밋 | 내용 |
|---|---|
| `401cc98` | `ServerConfig::test_cfg()`, integration `--tests` E0451 해소. `serial_fit`은 `pub(crate)` |
| `9fa5d98` | `--cont-width`를 `--help`에서 숨김. `DS4_SERVER_COALESCE_MAX`가 C knob. 스크립트용 숨은 alias는 유지 |
| `1d1fe03` | Anthropic/Responses streaming이 `Unsupported` serial fallback 하지 않음 |
| `4417906` | public 503을 C `"server shutting down"`에 맞춤. invent된 세 문자열 삭제 |

**Motif 라이브 재확인 (HEAD `4417906`):**

- `POST /v1/messages` stream → HTTP 200, finish=`end_turn`,
  `/metrics` anthropic **continuous 0→1**, serial 0
- `POST /v1/responses` stream → HTTP 200, finish=`completed`,
  `/metrics` responses **continuous 0→1**, serial 0
- 한 모델만, `guarded-run -m 112`, 내린 뒤에만 `clear_cache`

증거: `.omo/evidence/task-live-anthropic-cont-recheck-rust-host-remaining-to-phase-9.txt`

### 3.8 2026-08-26 작업트리 follow-up (커밋됨, `5ae1f82`…`661f5af`)

- `bridge_null_oracle`에 현재 bridge ABI의 최소 session stub을 추가했다.
- C와 Rust 모두 continuous bank가 마지막 미커밋 토큰 때문에 tool close marker
  중간에서 잘린 경우에도 bounded tool-memory에서 exact sampled block을 찾는다.
- Motif/Solar의 family-native multi-call form과 `raw_tool_text`를 exact replay
  소스로 처리한다. dots3 형식을 DSML scanner로 오인하지 않는다.
- Rust continuous lane이 tool turn을 bank 대상으로 받아들이고, producer replay를
  `WarmRecord.trailer`에 `KTM\x01`로 저장하며 restore-before-render를 수행한다.
- `Makefile`의 Rust release copy는 실제 `CARGO_TARGET_DIR`를 따른다.
- 4-way harness는 producer/loader의 tool schema JSON을 같은 compact form으로
  만든다. 공백이 다르면 KTM이 정상이어도 schema에서 먼저 갈라져 256-token
  partial hit만 보이는 false failure가 된다.
- workspace 재검증 중 발견한 `ds4-cli` list oracle 병렬 race는 child process의
  cwd를 전용 fixture로 고정해 제거했다.
- Rust `ContLane`에 C와 같은 live/disk partial-prefix reuse를 넣었다.
  (`warm_fork_partial`, `warm_disk_partial`, `bank_text_lcp_candidate`,
  `BatchCtx::supports_partial_reuse`)
- Rust가 C의 `DS4_SERVER_CONTINUOUS=0` kill switch를 읽도록
  `ServerConfig.continuous`를 연결했다. owner FIFO는 batch ctx를 유지한 채
  short prompt를 `LANE_STATIC`으로 보낸다.
  회귀: `continuous_zero_keeps_the_batch_ctx_for_two_short_static_jobs`
- Rust static gather가 C와 같이 `DS4_SERVER_COALESCE_WAIT_MS`를 읽는다.
  회귀: `late_sibling_joins_when_coalesce_wait_is_set`
- Motif none-think continuous bank retire는 official history와 같이
  생성 prefix의 빈 `<think></think>`를 키에서 뺀다.
  (`motif3_no_think_retire_prompt_len` / `motif3_history_retire_prompt`)
  히스토리 렌더 자체는 바꾸지 않는다.
- Motif/reasoning family 감사는 producer가 툴 앞에 낸 visible text를
  loader가 그대로 복사하고, `usage.cached_tokens`는 빈 think 2토큰만큼
  짧아도 된다. 엔진 복원(`timings.prefill_cached_tokens`)과
  `RESTORED_OK`는 그대로 필수다. `cached_tokens=0`은 FAIL.

증거: `.omo/evidence/task-p0-followup-rust-host-remaining-to-phase-9.txt`

### 3.9 2026-08-26 오후 Wave A–C (HEAD `4abaec3`)

Wave A (호스트만):

- `ce21e79` Stopping 503에서 Retry-After 제거 (C `wire_http_error`)
- `904efd6` `retire()`가 warm `ext_flags` 유지
- `f74850d` evict가 live `persist_bank` / `save_bank_record`를 탐
- `4693db0` continued target / bank-due가 `HostKvView`만 읽음

Wave B (quote + trim wrap):

- `c04ddf3` `ds4_bridge_session_graph_fit_quote` + null stub
- `663f155` `fail_open`이면 unquoted fallback. graph-fit은 margin이지 floor가 아님
- `4abaec3` idle-bank trim은 C `ds4_batch_ctx_trim_free` 한 번. 호스트 두 번째 trimmer 없음

Wave C (TickOp) — **BLOCKED, 커밋 없음:**

- `ContDriver::admit` / `on_token`은 R4 청크/디코드 순서를 강제할 수 없다
- C에 prefill-chunk / decode-step 공개 API가 없다
- step FFI를 새로 빼면 KEEP인 CUDA continuous 루프가 된다
- `generate_pair`는 `ds4_bridge_continuous_generate` 원샷을 유지
- `owner_tick_pair`는 호스트 스케줄 모델일 뿐 `_ops`는 계속 버려진다

### 3.10 2026-08-26 14:55 KST — serial n=1 + family serve

커밋 `40a321c`: HTTP static gather n==1 → serial (`ds4_server.c`
`worker_main`). `generate_static` n>=2 / n>=2 실패는 serial로 안 접힘.
`cargo test -p ds4-server --lib` 206 PASS.

라이브 (한 모델 → guarded-run -m 112 → SIGTERM → clear_cache):

| Family | 경로 | 결과 |
|---|---|---|
| Solar-Open2 | VMM owner + `ds4-server-rs` `-c 2048` | **PASS** `/v1/models` + chat `ok` stop, `continuous=1` |
| K-EXAONE | standalone `CONTINUOUS=0` n=1 | serial **entry** PASS (`serial=1`, 400 없음). 생성은 lazy graph FAIL |
| K-EXAONE | VMM owner + rust worker `-c 2048` | **PASS** chat `ok` stop, `continuous=1` |
| DeepSeek Flash | standalone `-c 2048` 81G GGUF | **PASS** chat `ok` stop. decision `continuous=1`, entry `serial=1` |

증거 (untracked):
`scratch/rust-host-live/family-serve-20260826-retry/`

standalone Solar는 여전히 boot_prewarm `cudaFreeAsync` IMA (CUDA KEEP).
긴 owner sock 경로는 `AF_UNIX` 한도에 걸림 — `/tmp/ds4-*-rs/` 사용.
EXAONE standalone serial 생성은 C `serial_session_ensure_fit` rightsize가
아직 호스트에 없다. 뱅크 옆 full `-c` lazy graph가 실패한다.
(→ §3.11에서 rightsize는 호스트로 이식됐다. 단 이 EXAONE standalone
FAIL의 근본 원인 재판독: `server.log`상 boot prewarm 중 CUDA IMA
(`CUDA tensor read failed: an illegal memory access`, exaone prefill
logits readback)가 먼저 발생했고, 이후 모든 CUDA alloc이 IMA로 실패해
`lazy session graph alloc failed`가 하류 증상으로 나타났다. 당시
avail ~35 GiB, ctx=512에서도 동일 — 용량 문제가 아니다.)

### 3.11 2026-08-26 오후 — serial rightsize 호스트 이식 (HEAD `fba963e`)

- `fba963e` `Rust(server): Host C serial_session_ensure_fit rightsize`
  - C `serial_session_fit_plan`/`serial_session_reuse_ok`를 C 테스트
    벡터와 함께 이식 (`crates/ds4-server/src/serve_serial_fit.rs`)
  - `run_serial`이 `generate_terminal_at`을 prepare(render+tokenize) /
    generate 2단계로 갈라 sync에 실제로 들어갈 토큰 수로 fit을 판정
  - Resize: free-before-probe → C settle window(20×100ms) →
    필요시 C trim 1회(`trim_idle_banks`, deficit+headroom 또는
    numberless면 whole-commons) → [need_min, need_full+32768] 이진
    탐색(1024 granularity) → `Model::session(target)` 재생성
  - RefusePreserve: ResolvedLive 프레임은 frontier 보존 + C 문구 503.
    Capacity refuse는 C 문구 503 + `requests_refused_deep_serial` +
    `rejected[serial][live_headroom]`
  - `DS4_SERVER_SERIAL_RIGHTSIZE=0`(정확히 "0"만 off)과 batch-ctx 없는
    부트는 v0.5.1 full--c 계약 유지
  - 새 bridge seam: `ds4_bridge_session_graph_pending` (freeze 카테고리
    session; null stub 추가로 `make test-server-parity` 링크 유지)
- cheap-gate 재스탬프 (HEAD `fba963e`): `cargo fmt --check`,
  `git diff --check`, `cargo test --workspace`(50 suite),
  `--no-default-features`, `cargo check --workspace --all-targets`,
  `make test-{server,kv,web,dist,catalog,tokenizer,session,agent}-parity`
  모두 PASS. `ds4-server` lib 단위 215.
- **Motif rust serial 4-surface 재스탬프 PASS** (P0-2 닫힘):
  `scratch/rust-host-live/task53/motif-serial-rust-20260826-154104/`
  — `CONTINUOUS=0` width=1, chat/completion/anthropic/responses 각
  `*_serial=1`, finish stop/length/end_turn/완료. Rust SHA
  `8d794e727fc5af3c61e8747e1170111a9a7e7a1203b6fe349c215c0e7e96c177`,
  C 셀은 `motif-surfaces-20260826-120759`의 기존 PASS 유지
  (C 바이너리 SHA `9dd45c7a…` 불변).
- 주의: 라이브 스크립트의 `rg`는 이 호스트 비대화형 셸에 없다 —
  `grep -Eq`로 치환했다 (`motif-serial-rust.sh`, `motif-surfaces.sh`,
  `exaone-serial-rightsize.sh`).

### 3.12 2026-08-26 오후 — EXAONE standalone IMA 분류 (C 대조 실험)

증거: `scratch/rust-host-live/task53/exaone-serial-rightsize-20260826-154403/`
(HEAD `fba963e`, C `9dd45c7a…`, Rust `8d794e72…`, owner `7eb5c9d2…`).
한 모델씩, `guarded-run -m 112`, phase 사이 SIGTERM → compute PID 없음 →
`clear_cache`.

| Phase | 셀 | 결과 |
|---|---|---|
| A | standalone `ds4-server-rs` `CONTINUOUS=0` `-c 2048` n=1 | 진입 serial ✓ (`openai_chat_serial=1`, `static_no_cont=1`). boot prewarm **IMA** (`CUDA tensor read failed: an illegal memory access`, exaone prefill logits readback) → 이후 CUDA alloc 전부 IMA → HTTP 500 `lazy session graph alloc failed`. rightsize의 fit quote는 순수 산술이라 오염된 컨텍스트를 볼 수 없고 REUSE로 통과 — C도 같은 지점에서 같은 결말 |
| B | standalone **C** `ds4-server` 동일 플래그 (대조군) | **같은 boot prewarm IMA, 같은 500** (`lazy session graph alloc failed (ctx=2048 prefill_cap=512)`). |
| C | VMM owner (`ds4_weight_server` ranges=529, `/tmp/ds4-exaone-rs/`) + `ds4-server-rs` worker `CONTINUOUS=0` `COALESCE_MAX=1` `-c 2048` n=1 | **PASS**: HTTP 200, content `"ok"`, finish=`stop`, prompt 20 → completion 1, TTFT 480.2 ms, prefill 57.0 tok/s. `openai_chat_serial=1`, `static_no_cont=1`. worker boot prewarm 2.0s 정상 — IMA는 in-process artifacts standalone 경로 한정임을 재확인. ensure-fit는 REUSE 레그로 통과 (resize 불필요) |

**판정: EXAONE standalone small-ctx 부트의 boot prewarm IMA는 C/Rust
공통의 엔진 측 갭이다 (Solar standalone `cudaFreeAsync` IMA와 같은
클래스, CUDA KEEP).** 호스트 rightsize 미이식이 원인이라던 이전 기록은
정정한다. rightsize 이식(`fba963e`)은 별개의 유효한 호스트 슬라이스로
남는다 (bank-holding 부트의 full-`-c` graph 500 클래스). **EXAONE rust
serial 생성(200 + 실제 decode + `serial=1`)은 정본 owner+worker 경로에서
PASS다.**

### 3.13 2026-08-26 저녁 — 2-트랙 플랜 확정 + Track 0

사용자가 `Baekpica/dfm-rs`를 선생성하고(스캐폴드 `a293973`) 2-트랙 실행
플랜을 승인했다. 이 문서와 `DFM_RS_SPLIT_PLAN.md`("실행 전제 변경 기록"
절)가 그 기록이다. **사용자 확정 결정 4건:**

1. **GREEN 스코프 = 호스트-패리티.** 게이트 셀의 실패 모드가 **같은
   명령의 C 대조 실행**으로 동일 재현되고 그 대조가 증거로 기록되면
   `PASS*(engine-gap E-n)`으로 표기하고 todo-54 판정에서 PASS로
   계수한다. C는 통과하는데 Rust만 실패하면 FAIL 유지. 셀 삭제·완화
   금지. 레지스트리는 [ENGINE_GAPS.md](ENGINE_GAPS.md)
   (E-1 Motif partial cached=0 / E-2 EXAONE IMA / E-3 Solar IMA /
   E-4 Motif batch / E-5 Motif resident RSS / **E-6 dots3 chunk/ring
   logit parity** — 전부 C-control reproduced).
2. **dfm-rs 시딩은 문서대로**: SPLIT_READINESS green 후 genesis SHA를
   `a293973` 위에 ancestry force-push. 그 전에는 코드 push 금지.
3. **로컬 클론은 workspace 루트** `../dfm-rs` (이동 완료; main
   protection 없음·PUBLIC·non-fork 확인).
4. **push 상시 승인**: rust-host는 origin과 동기화 유지, 검증된 커밋
   단위마다 push.

Track 0 완료 내역: rust-host push(`e9b1de0`→origin, 이후 계속),
web_tools.rs cwd flake proper fix(`0f8b1cb` — dirty 해소, tracked
클린), split 문서 통합(`de372e9` — 두 초안 SUPERSEDED로 tracked화),
ENGINE_GAPS.md(`054109c`), workspace CLAUDE.md에 dfm-rs 행 추가.

**남은 로드맵:** T1.3 공식 60셀 + T1.4 GREEN 완료 (§3.15). 다음
T1.5 §18.2 원자 승격(Rust가 기본 이름, C→`*-c`, proof oracle
`./ds4-c` + 동일-inode/해시 가드) → T1.6 승격 후 §10 전량 재실행 +
§18.3 pre/post SHA → T1.7 STATUS/PARITY 재도색 → T1.8 soak →
T1.9 SPLIT_READINESS + genesis → T1.10 push → T2.2 dfm-rs 시딩.

운영 메모: 이 스크립트의 worker teardown이 owner가 아직 살아 있는 상태에서
`clear_cache`를 한 번 호출했다 (`TEARDOWN_COMPUTE_STILL_ALIVE` 후).
페이지 캐시 드랍이라 결과에는 영향이 없지만 규율 위반이므로 다음 하니스는
owner 생존 중 clear_cache를 건너뛰도록 고칠 것. 최종 owner teardown 후의
clear_cache와 host 확인(compute 앱 0, avail 117Gi)은 정상.

### 3.14 2026-08-26 저녁 — §10 셰이크아웃 drain (CAND `f368131`, 공식 표 아님)

증거: `.omo/evidence/task-53-rerun-f368131.txt`,
`scratch/rust-host-live/gate-20260826/`. **todo-54 판정 금지** — 이
런은 하니스/호스트 갭 노출용. 공식 60셀은 다음 CAND에서 다시 돈다.

셀 집계 (중복 (5.5) 포함 61행): PASS 43 / FAIL 18.

닫힌 것:

- (11) Motif barrier width=2: C·Rust 둘 다 200/stop, 503 없음
- (13) Motif ABBA+RSS: PASS. prefill 99.9%, decode 100.3%, TTFT
  +1.1%, host HWM +1.0%
  (`g2-logs/abba-rss.log`, ratios pass=true)

호스트 갭 → 이미 수리:

- (10.*) 12셀 FAIL. stranded cont fallback이 continuous lane entry를
  안 찍음 (C는 serial+continuous 둘 다 tick; `ds4_server.c` ~10794).
  **`05a9357`로 수리·push.** 같은 런의 픽스처는 `-c 2048`이라 cont
  레인 펀딩 불가 (budget 70.4 MiB < projection 126.5 MiB). 공식 셀은
  `CTX=8192`.

엔진 갭 (golden v0.6.3-dfm worktree 4/4 동일 실패, `a3325ff` 무관):

| 셀 | 분류 |
|---|---|
| (7.3) / E-2 | candidate `test-exaone-batch`: logits readback IMA. golden: `cudaStreamBeginCapture (dense)` → `cudaMallocAsync :139` |
| (6.4) / E-3 | C standalone Solar `-c 2048` `cudaFreeAsync :167` 재현. golden `test_solar_session`도 BeginCapture/`cudaMallocAsync :139` |
| (5.6) / E-4 | Motif batch row 1 `got=2753,173,122,689 want=2753,173,203024,439` (candidate=golden 바이트 동일) |
| (5.5) / E-5 | VMM owner 하 RSS 370,412 KiB > 262,144 (golden 동일 수치) |

(8.3) dots3-resident는 셰이크아웃에서 owner 없이 실패 (하니스 전제).
공식 G5는 owner 변형. golden A/B는 아직 없음 — owner 런이 RSS/IMA로
죽으면 그때 E-n.

ENGINE_GAPS.md는 E-2/E-3 확장 + E-4/E-5 신설. 엔진 수리 커밋 없음.

### 3.15 2026-08-26 밤 — 공식 §10 GREEN (CAND `11c59e4`)

증거: `.omo/evidence/task-53-rerun-11c59e4.txt`,
`.omo/evidence/task-54-rerun-11c59e4.txt`,
`scratch/rust-host-live/gate-11c59e4/`.

60셀: PASS 55 + PASS* 5 (E-2..E-6) + FAIL 0 + BLOCKED 0 → **GREEN**.

| 그룹 | 결과 |
|---|---|
| G0 | 14 PASS |
| G1a DeepSeek KV+vectors | 7 PASS (`CTX=8192`) |
| G1b surfaces+proofs | 16 PASS — (10.*) 12셀 전부 PASS (`05a9357`) |
| G2 Motif | (5.1–5.4) PASS; (5.5) PASS*(E-5); (5.6) PASS*(E-4); (3.shutdown) PASS; (11) PASS; (13) ABBA PASS (prefill 100.09%, decode 101.0%, TTFT +0.9%, HWM +1.0%) |
| G3 Solar | (6.1–6.3, 6.5–6.9) PASS; (6.4) PASS*(E-3); E-3 C-control reproduced |
| G4 EXAONE | (7.1–7.2) PASS; (7.3) PASS*(E-2) |
| G5 dots3 | (8.1–8.2) PASS; (8.3) PASS*(E-6) |

G2 첫 패스는 `barrier_pair` `set -u`로 (11)에서 죽음. (11)/(13)은
`run-g2-rem2.sh`로 재스탬프. (8.3) golden `516456f`가 candidate와
같은 chunk/ring 수치로 실패 → E-6.

호스트 FAIL은 없다. 엔진 갭은 ENGINE_GAPS에만 남는다.

### 3.16 2026-08-26 밤 — T1.5 승격 + T1.6 post-promote GREEN

이름: `cb11c0b`. Rust 기본, C=`*-c`, `ds4-eval` C.

post-promote 표: `.omo/evidence/task-53-rerun-cb11c0b.txt` —
60셀 PASS 55 + PASS* 5 + FAIL 0 + BLOCKED 0. (11)/(13) 포함
한 패스에 PASS. FAIL 시그니처는 pre-promote와 동일 (E-2..E-6).

§18.3: Rust 기본 4종 + C `ds4-server-c`/`ds4-bench-c`/`ds4-agent-c`/
`ds4-eval`은 T1.5 해시 유지. **`ds4-c`만** proof-cuda-opp-c relink로
e9efb70e→95fa6828 (C CLI 정체 유지).

STATUS/PARITY는 이 표로 재스탬프. 다음 soak.

---

## 4. §10 비교 결과 (pre `11c59e4` GREEN; post `cb11c0b` GREEN)

계약: **모든 열거 셀이 PASS**여야 GREEN. FAIL 또는 BLOCKED가 하나라도 있으면
NOT_GREEN. crate 컴파일은 GREEN이 아니다.

### 4.1 첫 비교 (모델 env 없이, `4a02e24`)

증거: `.omo/evidence/task-53-rust-host-remaining-to-phase-9.txt`

**PASS=18 FAIL=3 BLOCKED=39** (60셀)

당시 FAIL:

1. `cargo test --workspace --no-default-features` — `serial_fit` E0451
2. `cargo check --workspace --all-targets` — 동일
3. `make test-server-parity` — `bridge_null_oracle` 링크 실패
   (`ds4_session_eval_layer_slice` 등 stub 없음)

2026-08-26 12:15 KST cheap 재스탬프 (HEAD `4abaec3`):

- `cargo fmt --all -- --check` / `git diff --check` — **PASS**
- `cargo test --workspace --no-default-features` — **PASS**
- `cargo test --workspace` — **PASS**
- `cargo check --workspace --all-targets` — **PASS** (기존 unused 경고만)
- `make test-{server,kv,web,dist,catalog,tokenizer,session,agent}-parity`
  — **PASS** (`ds4-server` 202 unit 포함)

BLOCKED는 대부분 `DS4_PROOF_BASE` / `DS4_*_MODEL` 미설정.

### 4.2 라이브 재실행 (Motif 88G → DeepSeek 81G, 한 번에 하나)

증거: `.omo/evidence/task-53-live-rust-host-remaining-to-phase-9.txt`

**PASS=13 FAIL=7 BLOCKED=5** (당시 남은 25 라이브 셀)

그 뒤 Anthropic/Responses 2셀은 핫픽스 후 **PASS로 뒤집힘**.

| 셀 | 당시 | 지금 |
|---|---|---|
| (3.shutdown) Motif bank 4-way | PASS | PASS (`RESTORED_OK` cached=6896, 4방향) |
| (3.tool-map) Motif/DeepSeek | FAIL | **PASS (DeepSeek + Motif)** — 아래 2026-08-26 라이브 |
| (3.ordinary/continued) | BLOCKED | **PASS (DeepSeek)** — 기존 serial cold/continued 4-way |
| (3.periodic/evict/partial) | BLOCKED | **PASS (DeepSeek + Motif evict/periodic)**; Motif partial HTTP **BLOCKED** |
| (10) serial 4 surface | PASS | **PASS (DeepSeek; Motif C+Rust)**. Rust n=1 400 닫힘 (`40a321c`); Motif rust serial 4-surface `fba963e` 바이너리로 재스탬프 PASS (§3.11) |
| (10) continuous chat/completion | PASS | **PASS (DeepSeek + Motif)** |
| (10) continuous anthropic/responses | FAIL (Rust serial) | **PASS (DeepSeek + Motif)** |
| (10) static 4 surface | FAIL | **PASS (DeepSeek + Motif, continuous off)** |
| (11) barrier width 2 | PASS | PASS |
| (12.1–12.4) proof smoke/long/opp-c/rust-opp-c | PASS | PASS (DeepSeek) |
| (13) Motif ABBA | PASS (throughput only) | **PASS + host RSS** — 아래 2026-08-26 재측정 |

DeepSeek tool-map 재검증:

- 결과:
  `scratch/rust-host-live/task53/tool-fourway-deepseek-v4-fixed-20260826-013629/`
- 모델: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
  (86,720,111,488 bytes)
- C/Rust producer KVC가 모두 `reason=8`, `ext=17`
  (`EXT_BANK_REPLAY_V1 | EXT_TOOL_MAP`), `tokens=423`, trailer=297,
  magic=`4b544d01` (`KTM\x01`)
- C→C, C→Rust, Rust→C, Rust→Rust 모두
  `RESTORED_OK`, finish=`stop`, prompt=452, cached=423, computed=29
- 바이너리 SHA256:
  C `469ea278e1a30e4fd36b01fa4a28aea19cf5462b3eaee233f65e7c19c27812b0`,
  Rust `4eedbfebeea9aef988260d2e73608c4f5dbf872b3b7f495cb5c0757b54a1951c`
- Motif full-model tool-map은 아래 2026-08-26 10:13 런에서 PASS.

ABBA 숫자 (Motif ctx 8192, prompt 6782, completion 114):

| | prefill tok/s | decode tok/s | TTFT ms | GPU MiB |
|---|---:|---:|---:|---:|
| C1 | 633.5 | 14.9 | 10781 | 114535 |
| R1 | 632.0 | 14.8 | 10817 | 102597 |
| R2 | 631.3 | 14.8 | 10829 | 102597 |
| C2 | 585.8 | 14.8 | 11649 | 114523 |

C2는 32뱅크 + memgov가 boot prewarm을 거절했다. 이 표의 RSS는 미측정.
2026-08-26 RSS 재측정은 `DS4_SERVER_COALESCE_MAX=2`로 아래를 쓴다.

DeepSeek KV/static 재검증 (2026-08-26, 한 모델씩, `guarded-run -m 112`):

- partial:
  `scratch/rust-host-live/task53/bank-partial-fourway-20260826-085417/`
  (`ctx=8192`, reason=`BankPartial=8`, ext=16, tokens=6786).
  4방향 loader 모두 `PARTIAL_RESTORED`, cached=6656, computed=128.
  C/Rust producer payload SHA는 달랐고, 호환은 cross-load로 판정했다.
- evict:
  `scratch/rust-host-live/task53/bank-evict-fourway-20260826-090152/`
  (`ctx=8192`, reason=`BankEvict=7`, tokens=6786).
  4방향 loader 모두 `EVICT_RESTORED`, cached=6786, computed=12.
- periodic:
  `scratch/rust-host-live/task53/bank-periodic-fourway-20260826-091902/`
  (`ctx=10400`, reason=`BankCheckpoint=9`, tokens=10298).
  producer는 기본 6 GiB derived headroom에서 continuous lane을 열었다.
  loader 4셀은 boot-fit 재현을 위해 `DS4_BATCH_FIT_HEADROOM_MB=5120`을 pin했다
  (`c-to-c-5g` … `rust-to-rust-5g`). 모두 `RESTORED_AGAIN`, cached=10298,
  computed=12. 12K 기본 headroom에서는 Rust memgov가 serial-only로 강등했다.
- static:
  `scratch/rust-host-live/task53/static-live-deepseek-20260826-093703/`
  (`ctx=1024`, `DS4_SERVER_CONTINUOUS=0`, width=4).
  C/Rust 모두 `openai_chat.serial=1`, `continuous=0`, `static=2`.
  blocker prompt=506 (stop-scan → serial), static pair prompt=773, n=2
  (`batch_eval n=1546 pos0=773`). Rust SHA
  `ba1b68f17b6911ebb45ee617329ba0b96dbce066758e0ecd2738501978405d80`.

Motif ABBA RSS 재측정 (2026-08-26):

- 결과: `scratch/rust-host-live/task53/motif-abba-rss-20260826-094218/`
- 모델: `Motif-3-MQ87-88-FIT.gguf` (`MQ87-88-FIT-SHA256SUMS`)
- 설정: `-c 8192`, `--tokens 128`, `DS4_SERVER_COALESCE_MAX=2`
- 4셀 모두 prompt=6782, completion=114, finish=`stop`, cached=0,
  content SHA `faf7c0214ae53aaf72ba9e8fd9404b62c6ac8a03ba97d18193c17835075332d3`
- swap 0, GPU resident 102597 MiB (C/R 동일)

| | prefill tok/s | decode tok/s | TTFT ms | VmHWM KiB | VmRSS KiB |
|---|---:|---:|---:|---:|---:|
| C1 | 631.3 | 14.8 | 10828.2 | 7739436 | 893280 |
| R1 | 629.3 | 14.7 | 10944.0 | 7819076 | 949708 |
| R2 | 633.8 | 15.0 | 10866.6 | 7819468 | 949680 |
| C2 | 632.9 | 14.8 | 10776.2 | 7740088 | 893308 |

비율 (Rust 평균 / C 평균): prefill 99.9%, decode 100.3%, TTFT +1.0%,
host HWM +1.0%. 임계값 prefill ≥97% / decode ≥98% / TTFT ≤+5% /
host RSS ≤+5%를 통과한다.

Motif tool-map 4-way (2026-08-26, 새 persist 키, resume 금지 후 재생산):

- 결과:
  `scratch/rust-host-live/task53/tool-fourway-Motif-3-MQ87-88-FIT-20260826-101314/`
- 모델: `Motif-3-MQ87-88-FIT.gguf` (94,162,541,472 bytes)
- C/Rust producer 모두 reason=8, ext=17, tokens=190, text=868,
  trailer=131, magic=`4b544d01`. 툴 앞 visible text:
  `I need to call the pair_values function with a=1 and b=2 before answering. Let me do that now.`
- 4방향 loader 모두 `RESTORED_OK`, finish=`stop`,
  `timings.prefill_cached_tokens=190`.
  `usage.cached_tokens=188` (249−61 또는 250−62). 빈 think 2토큰
  slack이며 엔진 복원 190과 모순되지 않는다.
- 바이너리 SHA256:
  C `9dd45c7a683fe14dd38ab0a54a43c5ff07b182424ac9ff4be2d1190fea378a46`,
  Rust `59ce472dd89945a17979a58e11508573d9a59a6a59766cc8b04f93c043751965`

DeepSeek 나머지 static surface (2026-08-26, `DS4_SERVER_CONTINUOUS=0`,
`DS4_SERVER_COALESCE_WAIT_MS=250`, thinking disabled, max_tokens=64):

- 결과: `scratch/rust-host-live/task53/static-surfaces-deepseek-20260826-103043/`
- C/Rust 모두 completion/anthropic/responses `static=2`, serial=0, continuous=0
- 짧은 reasoning 기본값(think=low)과 wait=0 single-gather collapse는
  픽스처 문제였다. 엔진 라우터를 “짧게 치면 static”으로 바꾸지 않았다.
- Rust SHA
  `01e943eeb7b81faae2a0c585a922afbcaca492d2cf52b95682807e96419e535c`

Motif periodic/evict/partial (2026-08-26 오후, HEAD `4abaec3`,
`COALESCE_MAX=2`, port 8765, thinking disabled):

- 바이너리 SHA256: C `9dd45c7a683fe14dd38ab0a54a43c5ff07b182424ac9ff4be2d1190fea378a46`,
  Rust `79eaa7738e982d6771551c54f2643408fa3ce0a8b4ba4c95cd397a00217a67b0`
- evict **PASS**:
  `scratch/rust-host-live/task53/bank-evict-fourway-motif-20260826-113523/`
  (`ctx=8192`, reason=`BankEvict=7`, ext=16, model=3, tokens=6418).
  4방향 모두 `timings.prefill_cached_tokens=6418`,
  `usage.cached_tokens=6416` (think 2토큰 slack).
- periodic **PASS**:
  `scratch/rust-host-live/task53/bank-periodic-fourway-motif-20260826-115513/`
  (`ctx=11264`, `HEADROOM_MB=5120`, reason=`BankCheckpoint=9`, tokens=10540).
  4방향 모두 cached=10540, usage=10538.
- partial **BLOCKED**:
  `scratch/rust-host-live/task53/bank-partial-fourway-motif-20260826-114317/`
  C 엔진은 `partial fork cut=6393 suffix=11` / `lcp=22904`를 찍었으나
  HTTP `timings.prefill_cached_tokens=0`. 계획 감사는 `cached=0`을 FAIL로
  본다. Motif family `last_done` stats가 C에서도 0이다. 호스트 하니스 실수가 아님.

Motif remaining surfaces (2026-08-26 오후, `-c 2048`, thinking disabled,
`COALESCE_WAIT_MS=250`, live body는 byte oracle 아님):

- 결과: `scratch/rust-host-live/task53/motif-surfaces-20260826-120759/`
- C/Rust continuous 4 surface: 각 `*.continuous=1` **PASS**
- C/Rust static 4 surface (`CONTINUOUS=0`, n=2): 각 `*.static=2` **PASS**
- C serial 4 surface (`CONTINUOUS=0`, width=1): 각 `*.serial=1` **PASS**
- Rust serial n=1 width-400 **닫힘** (`40a321c`): C `worker_main`처럼
  gather n==1은 `run_job_single`. `generate_static`는 n>=2 유지.
  EXAONE `CONTINUOUS=0` n=1: `static_no_cont=1`, `openai_chat.serial=1`,
  400 없음. Motif 4-surface rust serial은 이 HEAD에서 재스탬프하지 않음.

Family Makefile (서빙 없음, 2026-08-26 12:00 KST):

- `test-motif3-{reference,cuda}` **PASS**
- `test-solar-kda{,-prefill,-chunk}` **PASS**
- `test-exaone-kernels` **PASS** (모델 경로 없이)
- `test-mmq-parity` / `test-model-family-kernels` **PASS**
- Solar 250B / EXAONE 236B / DeepSeek 서빙: 아래 3.10 (2026-08-26 14:55 KST)

### 4.3 메모리 운영 (다음에 라이브 돌릴 때 필수)

- 모델은 **한 개만**. Motif 88G와 DeepSeek 81G를 동시에 올리지 말 것
- 기동: `../scripts/guarded-run.sh -m 112`
- 내리기: SIGTERM → pid/CUDA app 없음 확인 → **그다음** `/usr/local/bin/clear_cache`
- 서버가 살아있는 동안 `clear_cache` 금지
- Motif `-c 8192` 기본 C 32뱅크는 avail이 3–8 GiB까지 떨어짐.
  surface/4-way는 `DS4_SERVER_COALESCE_MAX=2`
- 포트 8765. 기존 스크립트: `scratch/rust-host-live/{abba,bank-fourway,run}.sh`

모델 경로:

```text
Motif-3:
  /home/sunghoon/workspace/ds4-exaone/models/Motif-3-Mixed-Quant-GGUF/Motif-3-MQ87-88-FIT.gguf
DeepSeek Flash IQ2XXS imatrix:
  /home/sunghoon/workspace/ds4-exaone/DeepSeek-GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

Solar 250B / EXAONE 236B standalone rust는 이 호스트에서 IMA/lazy-graph로
실패한 적이 있다. 2026-08-26 재시도는 **VMM owner + rust worker**로
둘 다 생성까지 PASS (3.10). DeepSeek 81G standalone도 PASS.

---

## 5. 지금 해야 할 일 (우선순위)

### P0 — GREEN을 가로막는 제품/게이트

1. **`make test-server-parity` 링크 — DONE (작업트리)**
   - 현재 bridge ABI의 최소 null stub을 추가했다.
   - server parity, no-default workspace test, all-target check를 재기록했다.

2. **Motif (또는 DeepSeek) tool-map 4-way — DONE (DeepSeek + Motif)**
   - C/Rust producer 모두 `EXT_TOOL_MAP` + `KTM\x01` bank-shutdown KVC를 썼다.
   - DeepSeek 4방향은 423 cached token과 `RESTORED_OK`.
   - Motif 4방향은 persist 키에서 빈 think를 뺀 뒤
     `timings.prefill_cached_tokens=190`과 `RESTORED_OK`.
     `usage.cached_tokens=188`은 감사 slack (think 2토큰).

3. **4-way 나머지 픽스처 — DONE (DeepSeek; Motif evict/periodic)**
   - ordinary/continued는 기존 serial 4-way.
   - DeepSeek periodic/evict/partial는
     `scratch/rust-host-live/intermediate-prefill-fourway-40gDCF/producer.request.json`
     을 compact JSON으로 재사용했다.
   - Motif evict/periodic은 같은 픽스처 + Motif history 텍스트 키 (3.9).
   - Motif partial은 엔진이 cut=6393을 찍었으나 HTTP cached=0 → **BLOCKED**.
   - 16K/width-2 standalone DeepSeek는 112 GiB 가드에서 OOM이 났고,
     이후 셀은 8K–11.2K와 `COALESCE_MAX=2`로만 올렸다.

4. **static 라이브 셀 — DONE (DeepSeek 4 surface, continuous off)**
   - 라우터를 “짧게 치면 static”으로 바꾸지 않았다.
   - chat는 기존 런. completion/anthropic/responses는 thinking off +
     coalesce wait 250ms로 n=2 static을 찍었다.
   - `prompt_len > seq_cap` 16K×3-copy 픽스처는 쓰지 않았다.

### P1 — 호스트가 아직 C에 맡긴 것

5. **native graph-fit quote / idle-bank trim — DONE (`c04ddf3`/`663f155`/`4abaec3`)**
   - quote FFI: `ds4_bridge_session_graph_fit_quote`. 산술은 C `ds4.c`.
   - `fail_open`이면 unquoted fallback (`need=0`, `avail=MAX`).
   - C는 floor가 아니라 **margin**이다. 그 의미를 유지했다.
   - idle-bank trim: 호스트는 C `ds4_batch_ctx_trim_free`를 한 번 부른다.
     두 번째 trimmer를 만들지 않았다.

6. **prefill interleave 실행권 — BLOCKED (커밋 없음)**
   - `owner_tick_pair` 결과는 `_ops`로 버려짐
   - `ContDriver` admit/on_token은 R4 청크/디코드 순서를 강제할 수 없다
   - C에 prefill-chunk / decode-step 공개 API가 없다
   - step FFI를 새로 빼면 KEEP인 CUDA continuous 루프가 된다
   - `generate_pair`는 `ds4_bridge_continuous_generate` 원샷을 유지한다

7. **`HostKvView`를 serve 경로에 연결 — DONE (`4693db0`)**
   - continued target / bank-due는 `{live_tokens, stored_tokens}`만 읽는다
   - 10240 aligned 의미는 그대로다

8. **`persist_bank_evict`와 live `persist_bank` 단일화 — DONE (`f74850d`/`904efd6`)**
   - evict는 `persist_bank(..., BankEvict)` + pin skip
   - `retire()`는 이전 warm `ext_flags`를 유지한다

9. **sibling mmap — KEEP**
   - Rust는 bind-map만 소유
   - `ds4_engine_open`이 여전히 `e->mtp_model` / `e->dspark_model`을 mmap
   - 이 캠페인에서 옮기지 않는다

10. **Stopping envelope — DONE (`ce21e79`)**
    - Stopping 503은 Retry-After 없음 (C `wire_http_error`)
    - queue-full 429는 5, preparse max-clients 503은 10 유지

### P2 — 라이브 게이트를 다시 채우기

11. **workspace cheap 재스탬프 — DONE (HEAD `4abaec3`, 12:15 KST)**
    - `cargo test --workspace --no-default-features`
    - `cargo test --workspace`
    - `cargo check --workspace --all-targets`
    - `cargo fmt --all -- --check`, `git diff --check`
    - `make test-{server,kv,web,dist,catalog,tokenizer,session,agent}-parity`

12. **Motif/DeepSeek 외 family — Makefile 셀 DONE, 서빙 안 함**
    - `test-motif3-{reference,cuda}`, `test-solar-kda{,-prefill,-chunk}`,
      `test-exaone-kernels`, `test-mmq-parity`, `test-model-family-kernels`
    - Solar 250B / EXAONE 236B 서빙 프로세스 없음
    - resident/batch는 한 모델 규칙을 위해 이 턴에서 안 돌림

13. **tokenizer vectors — DONE (DeepSeek)**
    - 제공된 DeepSeek GGUF를 `DS4_TEST_MODEL`로 사용했다.
    - `short_italian_fact`, `short_code_completion`,
      `short_reasoning_plain`, `long_memory_archive`, `long_code_audit`
      모두 PASS (`logprob-vectors: OK`, exit 0).
    - 증거:
      `scratch/rust-host-live/task53/deepseek-logprob-vectors-20260826.out`

14. **ABBA RSS — DONE (Motif, width=2)**
    - prefill 99.9%, decode 100.3%, TTFT +1.0%, host HWM +1.0%
    - 4셀 모두 `VmSwap=0`, GPU 102597 MiB
    - 기본 32뱅크 C가 아니라 `DS4_SERVER_COALESCE_MAX=2`다. 기록에 명시

15. **4 surface × 3 lane을 C와 나란히 재스탬프 — Motif 부분 DONE; serial n=1 닫힘**
    - DeepSeek continuous + static 4 surface: 기존 PASS
    - Motif continuous 4 surface: C/Rust **PASS**
    - Motif static 4 surface (`CONTINUOUS=0`, n=2): C/Rust **PASS**
    - Motif serial: C **PASS**. Rust n=1 width-400는 `40a321c`로 닫힘
      (crate + EXAONE metrics). Motif rust serial 4-surface는 미재스탬프
    - Solar/EXAONE owner+worker + DeepSeek standalone 생성: **PASS** (3.10)
    - 스키마/이벤트 순서/ID/finish는
      `docs/ds4-api-surface-matrix.md` (live body는 byte oracle 아님)

### P3 — Phase 9 (지금은 금지)

16. todo 53 표를 **FAIL=0, BLOCKED=0**으로 다시 채운다
17. todo 54가 GREEN일 때만:
    - production 이름을 Rust로, C는 oracle
    - 같은 §10을 **승격 후** 한 번 더
    - 그다음 `STATUS.md` / `PARITY_MATRIX.md`에 HEAD를 적는다
    - 그다음 `SPLIT_READINESS.md`
18. `Baekpica/dfm-rs`는 이 플랜이 GREEN이어도 **만들지 않는다**

---

## 6. 알려진 잔여 C 의존 (호스트)

| 영역 | 아직 C |
|---|---|
| Model open / CUDA·VMM alloc / weight upload | `ds4_bridge_model_open*` |
| Session eval / prefill / decode hot path | native session + CUDA |
| Continuous 청크 interleave 실행 | `ds4_bridge_continuous_generate` (TickOp KEEP) |
| Serial graph-fit quote 산술 | C `ds4.c` (호스트는 FFI wrap) |
| Idle-bank trim 본체 | C `ds4_batch_ctx_trim_free` (호스트는 한 번 호출) |
| Sibling mmap pointer | `ds4_engine_open` (KEEP) |
| Prefetch eval 외 파이프라인 일부 | dist는 receive queue만 호스트 |
| `ds4-eval` | 계속 C |
| CPU reference / Metal | cut-over blocker 아님 |

`ds4-sys` / `ds4-core`의 `unsafe`는 지정 FFI. CLI 27곳은 분류됨.

---

## 7. 최종 감사 (F1–F4)

| | 결과 | 메모 |
|---|---|---|
| F1 plan compliance | APPROVE | 1–61 체크 + 증거. 55–57 SKIP/CANCEL |
| F2 code quality | 당시 REJECT | `serial_fit` / `--cont-width` / invent 503 → **이후 커밋으로 해소**. F2 재감사는 안 함 |
| F3 runnable QA | APPROVE | 작업트리에서 fmt, workspace 2종, all-target check, server parity, DeepSeek vector/4-way PASS |
| F4 scope fidelity | APPROVE | rename/SPLIT/dfm-rs/CUDA rewrite/Tokio/merge 없음 |

F2를 HEAD `4abaec3`에서 다시 돌리면 세 required 항목은 코드상 닫혀 있다.
TickOp / Motif partial HTTP / EXAONE standalone serial rightsize는
이 감사 표 밖의 BLOCKED다. serial n=1 HTTP 400은 `40a321c`로 닫혔다.

---

## 8. 다음에 손대는 순서 (추천)

완료: Wave A–B 호스트 슬라이스, cheap+parity 재스탬프,
DeepSeek KV/static, Motif ABBA/tool-map/evict/periodic,
Motif continuous+static 4 surface, family Makefile 셀,
serial rightsize 호스트 이식(`fba963e`) + Motif rust serial 4-surface
재스탬프 (§3.11).

§10 호스트-패리티는 GREEN. 엔진 갭은 백로그 (승격과 혼합 금지):

1. TickOp 소비 — C continuous 루프 KEEP
2. E-1 Motif partial HTTP `cached=0` (이 60셀의 셀-오브-레코드 아님)
3. E-2 EXAONE IMA — 정본 owner+worker
4. sibling mmap — KEEP
5. E-3 Solar IMA — 정본 owner+worker
6. E-4 Motif batch / E-5 Motif RSS / E-6 dots3 chunk/ring

다음 손: **T1.8 soak** (DeepSeek, `./ds4-server`, 2h). 그다음
SPLIT_READINESS + genesis. 엔진 갭 커밋과 호스트 커밋은 섞지 않는다.

라이브는 항상: **한 모델 → guarded-run → 측정 → 프로세스 종료 → clear_cache → 다음.**
