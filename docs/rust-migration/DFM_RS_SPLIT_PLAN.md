# ds4-dfm-rs 독립 Repository 분리 및 후속 작업 지시서

### 범위: `ds4:rust-host` Split Readiness 통과 → `Baekpica/ds4-dfm-rs` 독립 운영 안정화

> **목표:** `ds4-dfm`의 C → Rust host migration이 완료되고 `SPLIT_READINESS.md`가 green 상태가 된
> commit을 기준으로, 기존 GitHub fork 관계에서 벗어난 독립 `ds4-dfm-rs` repository를 생성하고
> Rust-first inference runtime으로 운영 체계를 전환한다.
>
> **핵심 원칙:** 새 repository는 clean-room rewrite가 아니다.
> `antirez/ds4 → Entrpi/ds4 → Baekpica/ds4 → ds4-dfm-rs`의 기술적·저작권적 lineage를 명확히
> 보존하면서, 프로젝트의 architecture와 release lifecycle만 독립시킨다.

이 문서는 `SPLIT_READINESS.md`가 작성되기 **전**에는 실행하지 않는 후속 지시서다.
`SPLIT_READINESS.md` 자체는 §26(마이그레이션 게이트)이 증거 기반 green일 때만 작성한다.

## 실행 전제 변경 기록 (2026-08-26 통합)

이 절은 두 초안(`ds4_dfm_c_to_rust_migration_plan.md`,
`dfm_rs_repository_split_followup_plan.md` — 이제 SUPERSEDED 배너를 달고
tracked)과 이 문서를 **의도적으로 통합**한 결과이며, 이 문서가 단일
권위다. 초안의 "두 문서 통합 전 remote add/push 금지" 조건은 이 통합으로
해소됐다.

1. **레포 선생성 및 rename 사실:** initial scaffold `a293973`은
   `b01d1fa`에서 `ds4-dfm-rs`로 rename됐다. 2026-08-31 현재 원격은
   `Baekpica/ds4-dfm-rs`, PUBLIC, non-fork, default `main`이고 remote
   `main`은 `b01d1fa4172a5c957fe1232774629a192493efe4`다.
   **시딩 시점은 불변**: genesis 확정 전에는 코드를 push하지 않는다.
2. **시딩 방식 보정:** 기존 scaffold를 annotated archival tag
   `pre-genesis-scaffold-b01d1fa`로 먼저 보존한다. remote `main`이 여전히
   exact `b01d1fa`일 때만
   `--force-with-lease=refs/heads/main:b01d1fa4172a5c957fe1232774629a192493efe4`
   로 genesis를 시딩한다. baseline `v0.6.5-dfm` provenance tag도 함께
   push한다.
3. **원격 URL:** canonical HTTPS URL은
   `https://github.com/Baekpica/ds4-dfm-rs.git`이다. rename 전 URL은
   redirect로만 취급한다.
4. **로컬 클론 위치:** `/home/sunghoon/workspace/ds4-exaone/ds4-dfm-rs`
   (workspace 루트; 2026-08-26에 ds4-dfm 내부 중첩 위치에서 이동).
5. **GREEN 판정 스코프 (사용자 결정 2026-08-26):** §26 게이트는
   호스트-패리티 계약이다. 어떤 셀의 실패 모드가 **같은 명령의 C 대조
   실행**으로 동일 재현되고 그 대조가 증거로 기록되면
   `PASS*(engine-gap E-n)`으로 표기하고 GREEN 판정에서 PASS로 계수한다.
   C는 통과하는데 Rust만 실패하면 FAIL 유지. 셀 삭제·완화 금지.
   엔진 갭 레지스트리는 `docs/rust-migration/ENGINE_GAPS.md`.

---

# 0. 시작 조건

본 작업은 이전 C → Rust migration 작업이 완료되고 다음 조건이 충족된 상태에서만 시작한다.

```text
Baekpica/ds4
└── rust-host
    └── <DFM_RS_GENESIS_SHA>
```

그리고 다음 문서가 존재해야 한다.

```text
docs/rust-migration/SPLIT_READINESS.md
```

최소 상태:

```text
Baseline:              v0.6.5-dfm
Rust host parity:      PASS
CUDA backend:          unchanged / native
Model families:        PASS
API surfaces:          PASS
KV:                    PASS
Performance:           PASS
Known Rust regressions: NONE
C-shared engine gaps:  E-2, E-3, E-6
Recommended split commit: <DFM_RS_GENESIS_SHA>
```

이 commit을 이후 문서에서 **DFM-RS Genesis Commit**으로 부른다.

---

# 1. 새 Repository의 정체성

새 repository는 GitHub의 `Fork` 버튼으로 생성하지 않는다.

대상: `Baekpica/ds4-dfm-rs` (확정. `ds4-rs` 안은 기각)

이유:

- 기존 `ds4`와 architecture lifecycle을 분리할 수 있음
- Rust-first runtime이라는 정체성이 분명함
- 기존 DFM edition의 연속성을 유지함
- 향후 ds4 upstream 변화와 독립적으로 release할 수 있음

---

# 2. Git history 전략

## 2.1 History는 유지한다

새 repository라고 해서 source history를 squash하거나 지우지 않는다.

권장 방식:

```bash
git checkout rust-host
git remote add ds4-dfm-rs https://github.com/Baekpica/ds4-dfm-rs.git
git push ds4-dfm-rs pre-genesis-scaffold-b01d1fa
git push ds4-dfm-rs \
  --force-with-lease=refs/heads/main:b01d1fa4172a5c957fe1232774629a192493efe4 \
  <DFM_RS_GENESIS_SHA>:refs/heads/main
git push ds4-dfm-rs v0.6.5-dfm
```

또는 local branch를 먼저 정리한 뒤:

```bash
git checkout -b main <DFM_RS_GENESIS_SHA>
git push -u ds4-dfm-rs main
```

이렇게 하면 GitHub상 repository는 non-fork이지만 Git object ancestry는 보존된다.

## 2.2 History rewrite 금지

초기 분리 시 다음은 하지 않는다.

- `git filter-repo`로 C history 제거
- squash to one commit
- CUDA vendor history 제거
- author 정보 제거
- copyright header 일괄 교체

독립화와 history cleanup을 동시에 하지 않는다.

---

# 3. Lineage 명시

README 최상단 또는 별도 문서 `docs/LINEAGE.md`를 만든다.

```text
ds4-dfm-rs originates from the ds4 codebase and the DFM/Blackwell
work developed in Baekpica/ds4.

Project lineage:

antirez/ds4
    ↓
Entrpi/ds4
    ↓
Baekpica/ds4 (DFM edition)
    ↓
Baekpica/ds4-dfm-rs
```

README에서는 최소한 다음을 명시한다.

```text
This project is an independent Rust-first continuation of the DFM branch
developed from ds4.

It preserves the optimized native CUDA backend and reworks the host runtime
around Rust ownership, lifecycle, serving, and scheduling primitives.
```

새 프로젝트가 마치 처음부터 독립 구현이었던 것처럼 표현하지 않는다.

---

# 4. LICENSE / Copyright

기존 MIT license notice를 유지한다. substantial portions가 남아 있는 경우 기존
copyright notice(The ds4.c authors / Entrpi / ggml authors 등)를 삭제하지 않는다.

Rust migration 이후 사용자 기여분에 대해 필요하면
`Copyright (c) 2026 Baekpica / ds4-dfm-rs contributors` 형태로 병기한다.

**라이선스 변경 금지:** 초기 split 과정에서 MIT → Apache-2.0, MIT → dual license
같은 변경은 하지 않는다. 라이선스 정책 변경은 독립 repo 안정화 이후 별도 decision.

---

# 5. Repository 초기 구조

Genesis 시점 tree는 최소 다음 형태를 권장한다.

```text
ds4-dfm-rs/
├── Cargo.toml / Cargo.lock / rust-toolchain.toml
├── LICENSE / README.md / CHANGELOG.md / VERSION
├── crates/
│   ├── dfm-core/  dfm-kv/  dfm-server/  dfm-dist/  dfm-cli/  dfm-sys/
├── native/
│   ├── bridge/  cuda/  metal/
├── cuda/mmq/
├── metal/
├── tests/
│   ├── parity/  regression/  live/  proof/
├── benches/
├── docs/
│   ├── LINEAGE.md  ARCHITECTURE.md  CUDA_BACKEND.md  PERFORMANCE.md  COMPATIBILITY.md
└── legacy/          # only-if-still-required
```

---

# 6. Naming cleanup

한 번에 전부 바꾸지 않는다.

- **Phase A** (user-facing binary 먼저): `ds4-server-rs`→`dfm-server`, `ds4-bench-rs`→`dfm-bench`, `ds4-rs`→`dfm`, `ds4-agent-rs`→`dfm-agent`
- **Phase B** (crate): `ds4-core`→`dfm-core`, `ds4-server`→`dfm-server`, `ds4-kv`→`dfm-kv`
- **Phase C** (native symbol은 마지막): `ds4_cuda_*` / `ds4_bridge_*`는 무리하게 즉시 rename하지
  않는다. 이유: native/backend regression 위험, debug symbol continuity, git blame 가독성,
  upstream vendor diff 추적. 초기에는 old symbol naming을 compatibility detail로 허용한다.

**`ds4-eval` carve-out:** `ds4-eval`은 naming cleanup 대상이 아니다.
`make test`의 extractor oracle이며 Rust candidate가 없으므로 genesis와
`v0.1` 동안 이름과 C 구현을 그대로 유지하고, Rust로 승격했다고 주장하지
않는다.

---

# 7. Versioning 전략

새 version namespace `v0.1.0`에서 시작. README / CHANGELOG에
`ds4-dfm-rs v0.1.0 — derived from ds4-dfm v0.6.5-dfm baseline`을 명시한다.

피해야 할 방식: `v0.6.6-dfm-rs`, `v0.6.5.1`처럼 기존 ds4 version sequence에 종속되는 형태.

---

# 8. 첫 Release의 의미

`ds4-dfm-rs v0.1.0`은 feature release가 아니라 **Repository split / Rust-host parity release**다.

```text
same observable serving contract
same supported model families
same CUDA backend behavior
same performance class
different host-language architecture
```

첫 release에서 새 feature를 추가하지 않는다.

---

# 9. Release Candidate 단계

바로 `v0.1.0`을 찍지 않고 `v0.1.0-rc.1`, `v0.1.0-rc.2`, … 를 거친다.

각 RC는 최소: build / unit / parity / CUDA proof / server surface tests / long-context /
GB10 benchmark / soak 를 통과해야 한다. 이 campaign의 long soak 대상은 Qwen
Q5+Sidecar 한 모델이며 DeepSeek는 ordinary functional/parity/performance gate만
반복한다. 최종 `v0.1.0`은 RC와 source-identical 또는 documentation-only delta.

**추가 게이트 (2026-08-26 통합으로 흡수; 승격·genesis·RC에 공통):**

- native tracked-golden OPP-C: 고정 `v0.6.5-dfm` golden 대비 OPP-C가
  현 candidate에서 green일 것.
- C→Rust host OPP-C를 **명시적 binary 경로**로 실행 (oracle과 candidate
  경로를 커맨드에 적고 로그에 남긴다; 기본 이름 추론 금지).
- default/oracle binary **SHA-256 매핑 표**: 승격 전후로
  `ds4`/`ds4-server`/`ds4-bench`/`ds4-agent` ↔ `*-c` 의 해시를 기록하고
  pre-Rust SHA == post-default SHA, pre-C SHA == post-`*-c` SHA를
  검증한다. proof 하니스는 candidate와 oracle이 같은 inode 또는 같은
  binary hash면 실패해야 한다.
- `ds4-eval --self-test-extractors` PASS.
- Phase 9 pre/post family manifest replay: §18.3 매니페스트(고정 순서
  model-family smoke → API/KV fixtures → CUDA smoke → CUDA long →
  OPP-C → ABBA/perf → soak)를 pre/post 동일 입력으로 재현.

---

# 10. CI 구조

권장 job: lint-rust, fmt, clippy, build-linux, build-macos, unit, ffi-contract,
api-fixtures, kv-cross-compat, cuda-compile, metal-compile.

실 GPU CI가 불가능하면 cuda-compile / proof metadata validation / offline fixture tests까지
GitHub-hosted CI에서 처리하고, 실 GPU gate는 별도 release process로 유지한다.

---

# 11. Rust Toolchain 고정

`rust-toolchain.toml` 생성:

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
profile = "minimal"
```

compiler regression 때문에 pin이 필요하면 `1.xx.y` 형태로 명시한다.

---

# 12. Dependency 정책

dependency 추가 시 binary size / compile time / allocator·runtime behavior /
transitive dependencies를 검토한다. core runtime에서는 large async framework,
general-purpose reflection, heavy serialization framework, runtime plugin system을
무조건 도입하지 않는다.

---

# 13. `dfm-sys`의 장기 역할

목표: Rust application → safe Rust core → dfm-sys → CUDA/Metal native backend.

허용: CUDA driver/backend bridge, Metal bridge, MMQ bindings, native profiler hooks.
비권장(장기): C tokenizer, C session, C scheduler, C server를 감싸는 compatibility layer.

---

# 14. Legacy C 제거 단계

독립 repository 생성 후에만 진행. 순서:

1. unused C host source identify
2. Rust parity confirmation
3. dependency graph 확인
4. build target 제거
5. source 제거
6. full proof suite

한 commit에서 massive deletion하지 않는다. `cleanup: remove legacy C KV store` 식으로 분할.

---

# 15. `legacy/` 제거 기준

Rust production runtime stable + 2개 이상 stable release + C oracle이 필요한 active
regression 없음 + baseline tag 외부 보존 + proof fixtures가 old executable에서 독립 —
이면 `legacy/` 자체를 삭제할 수 있다. C baseline은 git tag/history에 남는다.

---

# 16. Native CUDA directory 재배치

CUDA kernel은 별도 repo로 분리하지 않는다. Rust host와 CUDA backend는 **같은 commit에서
versioned**되어야 runtime/kernel/ABI regression을 한 git bisect로 추적할 수 있다.

---

# 17. CUDA ABI version 추가

native boundary에 `#define DFM_NATIVE_ABI_VERSION 1`을 명시하고 Rust startup에서
expected/actual ABI를 비교한다. stale build artifact를 빠르게 감지한다.

---

# 18. C ABI 안정성 범위

`v0.x` 동안 native ABI는 **repository-internal unstable ABI**로 문서화한다.
`dfm-sys ↔ native backend`는 같은 source tree에서 build하는 것을 전제. 외부 third-party
backend plugin ABI는 제공하지 않는다.

---

# 19. Model-family contract 분리

hardcoded conditional을 계속 늘리지 않는다. trait 또는 enum dispatch:

```rust
enum Family { DeepSeek(...), SolarOpen2(...), KExaone(...), Motif3(...) }
```

inference hot path에 불필요한 dynamic dispatch를 강제하지 않는다.

---

# 20. Architecture freedom 시작점

`v0.1.0` parity release 이후부터 기존 migration 단계에서 금지했던 architectural change
(Tokio/async, HTTP stack 교체, scheduler redesign, admission redesign, typed wire session,
crate boundary 재설계 등)를 검토할 수 있다. **단 하나씩 한다.**

---

# 21. Async runtime 도입 여부

판단 기준: 현재 blocking architecture 병목 존재? HTTP concurrency가 GPU scheduling보다
병목인가? thread count 증가가 실제 문제인가? stream fan-out 비용이 의미 있는가?
"Rust니까 Tokio" 식으로 도입하지 않는다.

---

# 22. Server architecture redesign 시 Gate

GPU inference를 건드리지 않고 old Rust server ↔ new async server를 같은 native runtime
위에서 A/B한다. 검증: schema, stream ordering, finish semantics, continuation,
backpressure, disconnect, timeout, admission, TTFT.

---

# 23. Scheduler redesign

별도 milestone(`Scheduler v2`)로 둔다. Rust cleanup + continuous batching redesign +
new admission policy를 한 작업으로 묶지 않는다(회귀 attribution 불가).

---

# 24. Performance Baseline 재정의

`ds4-dfm-rs v0.1.0`을 새 baseline으로 `docs/PERFORMANCE.md`에 기록한다
(Reference release / Hardware DGX Spark GB10 / CUDA / Models / Prefill / Decode / TTFT /
Context / Memory). 이후 모든 major optimization은 이 baseline 또는 직전 stable과 비교.

---

# 25. Benchmark naming

benchmark result에는 반드시 commit SHA, release, model revision, quant revision, GPU,
driver, CUDA, context, batch/width, KV mode를 기록한다. `tok/s` 숫자만 README에 남기지 않는다.

---

# 26. Release Gate 재정의

stable release 최소 게이트: `cargo fmt --check`, `cargo clippy`, `cargo test`,
native compile, API fixture suite, KV suite, model-family tests, CUDA smoke, CUDA long,
performance proof, server soak. CUDA production path에서 long-context proof는 계속 release gate.

---

# 27. Compatibility 정책

- **Public compatibility(가능하면 유지):** HTTP API, CLI core flags, model loading,
  checkpoint format, observable sampling semantics
- **Internal compatibility(보장 안 함):** crate API, Rust types, native ABI,
  internal file layout, scheduler internals

README에 명시한다.

---

# 28. CLI compatibility

기존 주요 flag를 deprecated alias로 유지할 수 있으나 영구 유지하지 않는다.
`v0.x` 동안 deprecation window를 두고 정리한다.

---

# 29. Config / Environment Variable 정리

`DS4_*`는 compatibility alias로 유지, 새 namespace `DFM_*` 추가.
전환: `DFM_*` preferred / `DS4_*` compatibility. stable major에서 old namespace 제거 검토.

---

# 30. Binary 이름

권장: `dfm`, `dfm-server`, `dfm-bench`, `dfm-agent`. transitional symlink(`ds4`,
`ds4-server`) 허용. README canonical command는 `dfm-*`.

---

# 31. Observability

error taxonomy와 metrics internals를 type-safe하게 개선하되, 외부 HTTP error envelope의
compatibility contract는 깨지 않는다. 내부 error type 개선과 wire behavior 변경을 구분한다.

---

# 32. Error handling 정책

production runtime에서 `unwrap()` / `expect()` / `panic!()` 최소화. 특히 request parsing,
session lifecycle, CUDA resource handling, streaming, KV persistence, distributed runtime은
typed `Result`. panic은 programmer invariant 위반에 한정.

---

# 33. Unsafe 정책

`docs/UNSAFE_POLICY.md` 생성. unsafe 허용 위치: dfm-sys, CUDA/Metal FFI wrappers,
mmap boundary, carefully reviewed zero-copy primitives. 그 외 crate는 rationale 요구.
CI에서 `rg "unsafe" crates/` 결과를 audit artifact로 남긴다.

---

# 34. Memory ownership 문서화

`docs/MEMORY_MODEL.md`: Model ownership, weight mapping lifetime, CUDA allocation lifetime,
session lifetime, KV lifetime, batch bank ownership, continuation ownership,
scratch allocation, shutdown order.

---

# 35. Thread / Concurrency model 문서화

`docs/CONCURRENCY.md`: accept thread/runtime, request worker, scheduler, GPU submission,
stream writer, distributed workers, shutdown, lock-order / shared-state ownership.

---

# 36. Upstream ds4 추적 정책

자동 merge 관계는 종료한다. antirez/ds4, Entrpi/ds4, Baekpica/ds4에서 correctness fixes /
CUDA fixes / model semantics fixes / vendor updates만 수동 분석해서 port한다.

---

# 37. Upstream Port Log

`docs/UPSTREAM_PORTS.md`에 Source SHA / Topic / ds4-dfm-rs commit / Port type / Notes를 기록.
cherry-pick 가능 여부보다 **semantic provenance**가 중요하다.

---

# 38. Vendor 관리

`cuda/mmq/VENDOR.md` 유지: upstream repo, upstream commit, local patches, sync procedure,
validation. rename 과정에서 vendor provenance를 잃지 않는다.

---

# 39. Issue / Milestone 체계

권장: v0.1 parity release → v0.2 host cleanup → v0.3 architecture cleanup →
v0.4 scheduler/runtime improvements → v1.0 compatibility contract.
migration / cleanup / optimization을 분리한다.

---

# 40. `v1.0` 의미

Rust host architecture stable, public CLI policy stable, HTTP compatibility policy stable,
checkpoint compatibility policy stable, model-family extension mechanism stable,
native backend boundary stable enough, release gates automated/reproducible —
이 조건이 갖춰졌을 때. 그 전까지 `v0.x`에서 internal API를 자유롭게 개선.

---

# 41. README 개편

1. What is ds4-dfm-rs? 2. Hardware / backend support 3. Supported model families
4. Quick start 5. Performance 6. Architecture 7. API compatibility 8. DFM lineage
9. Build 10. Testing / proof methodology 11. License / acknowledgements

---

# 42. Project positioning

좋음: "Rust-first local inference runtime derived from ds4-dfm, preserving optimized
CUDA/MMQ kernels while moving host runtime ownership, serving, session, and memory
orchestration into Rust."

피할 표현: "ds4 rewritten in Rust" (CUDA/native lineage를 지나치게 단순화).

---

# 43. 성능 claim 정책

"4x faster" / "1000 tok/s" 같은 claim은 반드시 benchmark conditions와 연결.
hardware-specific이면 `DGX Spark / GB10 / sm_121a`를 명시. 일반 CUDA runtime 전체 성능으로
오해하게 만들지 않는다.

---

# 44. Model support tier

Tier 1: DeepSeek, Solar Open2, K-EXAONE, Motif-3 / Tier 2: experimental families /
Reference only. 각 tier는 test requirement가 달라야 한다.

---

# 45. Hardware support tier

Tier 1: NVIDIA GB10 / sm_121a. Tier 2: other Blackwell. Build-only: Metal.
Experimental: other CUDA architectures. 검증하지 않은 hardware를 generic support로 표기하지 않는다.

---

# 46. Docker / Packaging

순서: source build 안정화 → release binary → container image → package manager.
Docker / prebuilt binary는 core release 안정 후.

---

# 47. Prebuilt Binary

release artifact에 commit, rustc version, CUDA requirements, minimum driver,
target architecture, sha256 명시. 예: `dfm-server-v0.1.0-linux-aarch64-gb10`.

---

# 48. Reproducibility

`cargo build --release --locked` 고정, native CUDA build도 deterministic command 문서화.
`Cargo.lock`은 commit한다.

---

# 49. Security / Supply Chain

cargo audit, cargo deny, dependency license checks, Dependabot 검토.
단 이것들이 inference release gate를 대신하지 않는다.

---

# 50. Documentation split

`docs/rust-migration/*`은 완료 후 `docs/history/rust-migration/` 등으로 이동 가능.
**삭제하지 않는다** — architecture decision provenance.

---

# 51. ADR 도입

`docs/adr/0001-rust-host.md`, `0002-native-cuda-boundary.md`, `0003-server-runtime.md`,
`0004-model-family-dispatch.md` — decision / rationale / consequences에 집중.

---

# 52. 성능 회귀 관리

runtime-sensitive change(scheduler, FFI, memory, KV, CUDA launch, graph, batching,
tokenizer hot path)는 `perf-sensitive` label + benchmark 결과 요구.

---

# 53. Profiling workflow

nsys / ncu / host flamegraph / heap profile workflow를 문서화하고 command를 version control:
`tools/profile/{nsys_prefill.sh, nsys_decode.sh, ncu_mmq.sh, host_flamegraph.sh}`.

---

# 54. Optimization policy

hypothesis → profile → patch → correctness proof → speed proof.
"Rust로 바꿨으니 빨라질 것" 같은 언어-level assumption을 성능 근거로 사용하지 않는다.

---

# 55. Repo 분리 완료 Definition of Done

```text
Baekpica/ds4-dfm-rs exists as non-fork
main points to DFM-RS Genesis Commit
git ancestry preserved
LICENSE preserved
LINEAGE documented
README rewritten
crate/binary naming stable enough
CI green
native build green
v0.1.0-rc.1 released
CUDA live proof green
GB10 benchmark recorded
old Baekpica/ds4 points users to ds4-dfm-rs
```

---

# 56. 기존 `Baekpica/ds4` 처리

새 repo 생성 직후 기존 repo를 삭제하지 않는다. README 상단에
"Active development moved to: Baekpica/ds4-dfm-rs" migration notice.

---

# 57. 기존 Repo Archive 시점

ds4-dfm-rs stable release ≥ 1, critical regression 없음, documentation migration 완료,
issue/PR migration 불필요, old repo 신규 개발 없음 — 이후 archive 검토. 바로 archive하지 않는다.

---

# 58. 최종 Architecture 목표

```text
                    ds4-dfm-rs
┌────────────────────────────────────────┐
│                Rust                    │
│ API / protocol / scheduler / admission │
│ model lifecycle / session / KV         │
│ memory policy / distributed runtime    │
│ observability                          │
└──────────────────┬─────────────────────┘
                   │  narrow native ABI
┌──────────────────▼─────────────────────┐
│             Native backend             │
│ CUDA runtime / driver / VMM            │
│ CUDA Graph / MMQ / fused attention     │
│ MoE kernels / Metal backend            │
└────────────────────────────────────────┘
```

---

# 59. 독립화 이후의 원칙

Migration 이전: **Behavior preservation first** →
독립화 직후: **Stabilize the new ownership boundary** →
그 이후: **Architecture and performance evolution**.
새 repo를 만들자마자 모든 architecture를 다시 뒤집지 않는다.

---

# 60. 최종 요약

```text
ds4-dfm fork
     ↓
C → Rust migration
     ↓
behavior / performance parity
     ↓
genesis commit freeze
     ↓
independent non-fork repository
     ↓
ds4-dfm-rs v0.1 parity release
     ↓
legacy host C cleanup
     ↓
independent architecture evolution
```

프로젝트의 provenance는 보존하면서 lifecycle만 독립시킨다. `ds4-dfm-rs` 생성 직후에는 여전히
`ds4-dfm v0.6.5-dfm`의 semantics와 성능이 기준점이다. 새 repository의 진짜 독립성은
source history를 끊는 데서 오는 것이 아니라:

- Rust가 host runtime의 ownership을 완전히 담당하고
- CUDA backend boundary가 명확하며
- release/version policy가 독립되고
- upstream 변화가 자동 merge가 아닌 선택적 semantic port가 되고
- 자체 performance/correctness gate를 갖는 것

에서 나온다.
