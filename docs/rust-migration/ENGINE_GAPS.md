# Engine gaps (C-shared) — living backlog

C `v0.6.3-dfm`과 Rust 호스트가 **동일하게** 보이는 엔진 측 결함의
레지스트리다. 호스트 마이그레이션 게이트는 호스트-패리티 계약이므로,
여기 등재된 갭은 게이트 셀에서 `PASS*(engine-gap E-n)` annotation의
근거가 된다 — 단 **같은 명령의 C 대조 실행이 증거로 기록된 항목만**
(`C-control: reproduced`). C-control이 pending인 항목은 annotation
근거로 쓸 수 없다.

규율: 갭 수리는 엔진(C/CUDA) 작업이며 `AGENT.md`를 따른다.
**엔진 수리 커밋과 호스트 마이그레이션 커밋을 혼합하지 않는다.**
이 파일은 genesis 트리에 남아 `dfm-rs`로 이관된다. 항목이 수리되면
"Fixed" 상태와 수리 커밋을 남기고 삭제하지 않는다.

---

## E-1 — Motif-3 partial-reuse HTTP `cached=0`

- **증상:** partial prefix reuse에서 엔진은 `partial fork cut=6393
  suffix=11` / `lcp=22904`를 찍지만 HTTP `timings.prefill_cached_tokens`
  가 0으로 보고된다. Motif family의 `last_done` stats 배선 갭.
- **C-control:** **reproduced** — C 서버도 HTTP `cached=0`
  (하니스 실수 아님; 2026-08-26 판독).
- **Canonical 경로:** DeepSeek partial 4-way는 정상
  (`PARTIAL_RESTORED`, cached=6656) — 갭은 Motif family stats에 한정.
- **증거:**
  `scratch/rust-host-live/task53/bank-partial-fourway-motif-20260826-114317/`
- **스코프:** 엔진 (C `ds4.c`/`ds4_server.c` Motif stats 경로).
- **상태:** Open.

## E-2 — K-EXAONE standalone boot-prewarm CUDA IMA

- **증상:** in-process aligned-artifacts standalone 부트에서 boot
  prewarm 중 `CUDA tensor read failed: an illegal memory access`
  (exaone prefill logits readback). 이후 CUDA 컨텍스트가 오염되어 모든
  alloc이 IMA로 실패 → 요청은 `lazy session graph alloc failed` 500.
  메모리 용량 문제 아님 (avail ~35 GiB, ctx=512에서도 재현).
- **C-control:** **reproduced** — C `ds4-server` 동일 플래그에서 같은
  IMA·같은 500 (2026-08-26 phase B).
- **Canonical 경로:** VMM owner + worker는 정상 — worker prewarm 2.0s
  PASS, serial 생성 200 (`openai_chat_serial=1`, TTFT 480 ms).
- **증거:**
  `scratch/rust-host-live/task53/exaone-serial-rightsize-20260826-154403/`
  (phase A rust / phase B C-control / phase C owner+worker).
- **스코프:** 엔진 (EXAONE prefill + in-process artifacts 경로, CUDA).
- **상태:** Open.

## E-3 — Solar-Open2 standalone boot-prewarm `cudaFreeAsync` IMA

- **증상:** standalone `-c 2048` 부트의 boot_prewarm에서 `cudaFreeAsync`
  IMA (`cuda/mmq/ds4_ggml_stubs.cu:167`).
- **C-control:** **pending** — rust standalone에서만 관측됨. §10 rerun의
  Solar 그룹(G3)에서 **같은 플래그의 C standalone 부트 1회로 대조**해야
  한다. C가 재현하면 E-2와 같은 클래스로 확정; C가 통과하면 이 항목은
  엔진 갭이 아니라 **호스트 측 조사 항목으로 격상**되며 annotation
  근거로 쓸 수 없다 (GREEN 블로커).
- **Canonical 경로:** VMM owner + rust worker `-c 2048` PASS
  (chat `ok` stop, `continuous=1`, TTFT 254 ms).
- **증거:** `scratch/rust-host-live/family-serve-20260826-retry/solar/`
  (workspace-level scratch).
- **스코프:** 미정 (C-control 결과에 따라 엔진 또는 호스트).
- **상태:** Open — C-control 필요.
