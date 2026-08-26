> **SUPERSEDED (2026-08-26):** 이 초안은 tracked `DFM_RS_SPLIT_PLAN.md`로
> 통합됐다("실행 전제 변경 기록" 절). 이 파일은 provenance 보존용이며,
> 이 파일과 `DFM_RS_SPLIT_PLAN.md`가 다르면 후자가 이긴다.

# dfm-rs 독립 Repository 분리 및 후속 작업 지시서
### 범위: `ds4:rust-host` Split Readiness 통과 → `Baekpica/dfm-rs` 독립 운영 안정화

> **목표:** `ds4-dfm`의 C → Rust host migration이 완료되고 `SPLIT_READINESS.md`가 green 상태가 된 commit을 기준으로, 기존 GitHub fork 관계에서 벗어난 독립 `dfm-rs` repository를 생성하고 Rust-first inference runtime으로 운영 체계를 전환한다.
>
> **핵심 원칙:** 새 repository는 clean-room rewrite가 아니다. `antirez/ds4 → Entrpi/ds4 → Baekpica/ds4 → dfm-rs`의 기술적·저작권적 lineage를 명확히 보존하면서, 프로젝트의 architecture와 release lifecycle만 독립시킨다.

---

## 현재 활성화 상태 — 2026-08-25 KST

> **INACTIVE / BLOCKED — 이 지시서는 아직 실행하지 않는다.**

현재 `rust-host`의 production 기본 바이너리는 모두 C이고, Phase 2–8은
partial, Phase 9는 시작하지 않았다. Phase 4의 scoped final-sync/decode
continued checkpoint, bounded tool-map replay, 그리고 ordinary-serial
DeepSeek intermediate-prefill progress (`49a2b65`, `8361116`)는 live
four-way까지 green이다. Scoped width-1 Motif-3 OpenAI Chat no-think/no-tools
bank-shutdown/replay도 `e9dfd77`, `0e4a178`, `15b016c`와 live four-way/
114-token restore ABBA까지 green이다. `98d81b9`의 native continuous timing
fix 뒤 short Rust restore는 12.3 tok/s, 1.25 tok/step으로 C와 일치했다.
하지만 default configured 10,000/effective 10,240 periodic checkpoint,
live bank-evict, full default-policy ABBA, multi-bank fork/partial과 pin/claim,
bank extensions, other wire surfaces, ownership/integration/full
parity/perf/soak는 아직 pending이다. Source tip과 `origin/rust-host`는
`042562b`에서 일치한다.
`SPLIT_READINESS.md`는 존재하지 않으며, 이 상태가 올바르다.

이 문서를 실행 가능한 지시서로 전환하기 전에 반드시 다음을 모두
확인한다.

```text
STATUS.md Phase 9                           green
SPLIT_READINESS.md                          exists and green
Recommended split commit                    clean, immutable SHA
working tree at genesis selection            clean
family/API/KV/proof/performance/soak gates   green
Phase 9 pre/post evidence manifest           verified
explicit C/Rust OPP-C paths                  verified
default/oracle binary SHA mapping            verified
ds4-eval retained-C decision                 recorded in SPLIT_READINESS.md
repository name                              resolved
```

현재 repository에는 이 full follow-up 문서와 별도로 tracked condensed
`DFM_RS_SPLIT_PLAN.md`가 있다. 두 문서는 byte-identical하지 않으며,
condensed 문서는 `Baekpica/dfm-rs`를 확정한 반면 이 문서는
`dfm-rs`/`ds4-rs` 선택지를 남겨 둔다. Genesis 작업 전에 두 문서를
의도적으로 통합하고 이름을 하나로 확정한다. 그 전에는 remote 추가,
repository 생성, push, history 정리 중 어느 것도 실행하지 않는다.

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
Baseline:
v0.6.3-dfm

Rust host parity:
PASS

CUDA backend:
unchanged / native

Model families:
PASS

API surfaces:
PASS

KV:
PASS

Performance:
PASS

Known regressions:
NONE

Recommended split commit:
<DFM_RS_GENESIS_SHA>
```

이 commit을 이후 문서에서:

> **DFM-RS Genesis Commit**

으로 부른다.

---

# 1. 새 Repository의 정체성

새 repository는 GitHub의 `Fork` 버튼으로 생성하지 않는다.

대상:

```text
Baekpica/dfm-rs
```

또는 실제 naming 결정에 따라:

```text
Baekpica/ds4-rs
```

중 하나를 사용한다.

권장 이름은:

```text
dfm-rs
```

이다.

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

git remote add dfm-rs git@github.com:Baekpica/dfm-rs.git

git push dfm-rs <DFM_RS_GENESIS_SHA>:refs/heads/main
```

또는 local branch를 먼저 정리한 뒤:

```bash
git checkout -b main <DFM_RS_GENESIS_SHA>
git push -u dfm-rs main
```

한다.

이렇게 하면 GitHub상 repository는 non-fork이지만 Git object ancestry는 보존된다.

---

## 2.2 History rewrite 금지

초기 분리 시 다음은 하지 않는다.

```text
git filter-repo로 C history 제거
squash to one commit
CUDA vendor history 제거
author 정보 제거
copyright header 일괄 교체
```

독립화와 history cleanup을 동시에 하지 않는다.

---

# 3. Lineage 명시

README 최상단 또는 별도 문서:

```text
docs/LINEAGE.md
```

를 만든다.

예시 구조:

```text
dfm-rs originates from the ds4 codebase and the DFM/Blackwell
work developed in Baekpica/ds4.

Project lineage:

antirez/ds4
    ↓
Entrpi/ds4
    ↓
Baekpica/ds4 (DFM edition)
    ↓
Baekpica/dfm-rs
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

기존 MIT license notice를 유지한다.

특히 substantial portions가 남아 있는 경우 기존 copyright notice:

```text
The ds4.c authors
Entrpi
ggml authors
```

등을 삭제하지 않는다.

Rust migration 이후 사용자 기여분에 대해 필요하면 추가:

```text
Copyright (c) 2026 Baekpica / dfm-rs contributors
```

형태로 병기한다.

## 라이선스 변경 금지

초기 repository split 과정에서:

```text
MIT → Apache-2.0
MIT → dual license
```

같은 license 변경은 하지 않는다.

라이선스 정책 변경은 독립 repo 안정화 이후 별도 decision으로 처리한다.

---

# 5. Repository 초기 구조

Genesis 시점 tree는 최소 다음 형태를 권장한다.

```text
dfm-rs/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── LICENSE
├── README.md
├── CHANGELOG.md
├── VERSION
│
├── crates/
│   ├── dfm-core/
│   ├── dfm-kv/
│   ├── dfm-server/
│   ├── dfm-dist/
│   ├── dfm-cli/
│   └── dfm-sys/
│
├── native/
│   ├── bridge/
│   ├── cuda/
│   └── metal/
│
├── cuda/
│   └── mmq/
│
├── metal/
│
├── tests/
│   ├── parity/
│   ├── regression/
│   ├── live/
│   └── proof/
│
├── benches/
│
├── docs/
│   ├── LINEAGE.md
│   ├── ARCHITECTURE.md
│   ├── CUDA_BACKEND.md
│   ├── PERFORMANCE.md
│   └── COMPATIBILITY.md
│
└── legacy/
    └── only-if-still-required
```

---

# 6. Naming cleanup

Repository 분리 직후 naming을 정리하되 한 번에 전부 바꾸지 않는다.

## Phase A

먼저 user-facing naming만 변경한다.

```text
ds4-server-rs
→
dfm-server

ds4-bench-rs
→
dfm-bench

ds4-rs
→
dfm

ds4-agent-rs
→
dfm-agent
```

`ds4-eval`은 genesis/v0.1에서 이름과 C 구현을 유지한다. Rust replacement
후보가 없으며 `make test`의 extractor oracle 역할도 하므로 user-facing
naming cleanup 대상에 포함하지 않는다.

## Phase B

crate names:

```text
ds4-core
→
dfm-core

ds4-server
→
dfm-server

ds4-kv
→
dfm-kv
```

## Phase C

native symbol rename은 가장 마지막.

다음과 같은 internal ABI symbol:

```text
ds4_cuda_*
ds4_bridge_*
```

를 무리하게 즉시 rename하지 않는다.

이유:

- native/backend regression 위험
- debug symbol continuity
- git blame 가독성
- upstream vendor diff 추적

초기에는 old symbol naming을 compatibility detail로 허용한다.

---

# 7. Versioning 전략

기존 DFM release와 독립된 version namespace를 시작한다.

권장:

```text
v0.1.0
```

단 README / CHANGELOG에 명확히:

```text
dfm-rs v0.1.0
derived from ds4-dfm v0.6.3-dfm baseline
```

을 남긴다.

## 피해야 할 방식

```text
v0.6.4-dfm-rs
v0.6.3.1
```

처럼 기존 ds4 version sequence에 계속 종속되는 형태는 피한다.

새 repo를 독립시키는 목적과 모순된다.

---

# 8. 첫 Release의 의미

`dfm-rs v0.1.0`은 feature release가 아니다.

정의:

> **Repository split / Rust-host parity release**

즉 다음을 의미한다.

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

바로 `v0.1.0`을 찍지 않고:

```text
v0.1.0-rc.1
v0.1.0-rc.2
...
```

형태를 권장한다.

각 RC는 최소:

```text
build
unit
parity
CUDA proof
server surface tests
long-context
GB10 benchmark
soak
```

를 통과해야 한다.

최종:

```text
v0.1.0
```

은 RC와 source-identical 또는 최소한 documentation-only delta가 되도록 한다.

---

# 10. CI 구조

독립 repo 생성 직후 GitHub Actions를 새 namespace 기준으로 정리한다.

권장 job:

```text
lint-rust
fmt
clippy
build-linux
build-macos
unit
ffi-contract
api-fixtures
kv-cross-compat
cuda-compile
metal-compile
```

실 GPU CI가 불가능하면:

```text
cuda-compile
proof metadata validation
offline fixture tests
```

까지 GitHub-hosted CI에서 처리하고,

실 GPU gate는 별도 release process로 유지한다.

---

# 11. Rust Toolchain 고정

`rust-toolchain.toml`을 생성한다.

예:

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
profile = "minimal"
```

특정 compiler regression 때문에 pin이 필요하다면:

```text
1.xx.y
```

형태로 명시한다.

Release artifact가 compiler drift에 영향을 받지 않도록 한다.

---

# 12. Dependency 정책

독립 이후에도 inference runtime 특성상 dependency 증가를 보수적으로 관리한다.

기본 원칙:

```text
dependency 추가
→ binary size
→ compile time
→ allocator/runtime behavior
→ transitive dependencies
```

를 검토한다.

특히 core runtime에서는:

```text
large async framework
general-purpose reflection
heavy serialization framework
runtime plugin system
```

을 무조건 도입하지 않는다.

---

# 13. `dfm-sys`의 장기 역할

Split 직후 `dfm-sys`가 남아 있어도 문제 없다.

목표는:

```text
Rust application
      ↓
safe Rust core
      ↓
dfm-sys
      ↓
CUDA / Metal native backend
```

이다.

단 장기적으로 `dfm-sys`가 C host compatibility layer 전체를 감싸는 구조로 남아서는 안 된다.

허용:

```text
CUDA driver/backend bridge
Metal bridge
MMQ bindings
native profiler hooks
```

비권장:

```text
C tokenizer
C session
C scheduler
C server
```

---

# 14. Legacy C 제거 단계

독립 repository 생성 후에만 legacy C cleanup을 진행한다.

순서:

```text
1. unused C host source identify
2. Rust parity confirmation
3. dependency graph 확인
4. build target 제거
5. source 제거
6. full proof suite
```

한 commit에서 massive deletion하지 않는다.

예:

```text
cleanup: remove legacy C KV store

cleanup: remove legacy C HTTP path

cleanup: remove legacy C session wrapper
```

형태로 분할한다.

`ds4-eval` source와 target은 Rust replacement 및 extractor parity가 생기기
전에는 이 cleanup 대상으로 분류하지 않는다.

---

# 15. `legacy/` 제거 기준

다음이 충족되면 `legacy/` 자체를 삭제할 수 있다.

```text
Rust production runtime stable
2개 이상 stable release
no active regression requiring C oracle
baseline tag externally preserved
proof fixtures independent from old executable
```

C baseline은 git tag/history에 남아 있으므로 repository 안에 영구적으로 duplicate source를 유지할 필요는 없다.

---

# 16. Native CUDA directory 재배치

독립화 이후에도 CUDA kernel은 별도 repo로 분리하지 않는다.

권장:

```text
native/cuda/
├── backend/
├── graph/
├── vmm/
└── mmq/
```

또는 기존 layout을 유지한다.

핵심:

> Rust host와 CUDA backend는 같은 commit에서 versioned되어야 한다.

그래야:

```text
runtime regression
kernel regression
ABI mismatch
```

를 한 git bisect에서 추적할 수 있다.

---

# 17. CUDA ABI version 추가

독립 repository에서는 native boundary에 ABI version을 명시하는 것을 권장한다.

예:

```c
#define DFM_NATIVE_ABI_VERSION 1
```

Rust startup에서:

```text
expected ABI
actual ABI
```

를 비교하도록 한다.

잘못된 shared object / stale build artifact를 빠르게 감지한다.

---

# 18. C ABI 안정성 범위

외부 public ABI를 약속할 필요는 없다.

초기 `v0.x` 동안 native ABI는:

> **repository-internal unstable ABI**

로 문서화한다.

즉:

```text
dfm-sys ↔ native backend
```

는 같은 source tree에서 build하는 것을 전제로 한다.

외부 third-party backend plugin ABI는 아직 제공하지 않는다.

---

# 19. Model-family contract 분리

독립 이후 모델 family 지원을 core에 hardcoded conditional로 계속 늘리지 않는다.

권장 개념:

```rust
trait ModelFamily {
    fn architecture(&self) -> ...
    fn tokenizer(&self) -> ...
    fn tensor_map(&self) -> ...
    fn prompt_rules(&self) -> ...
    fn stop_rules(&self) -> ...
}
```

단 inference hot path에 불필요한 dynamic dispatch를 강제하지 않는다.

필요하면 enum dispatch:

```rust
enum Family {
    DeepSeek(...),
    SolarOpen2(...),
    KExaone(...),
    Motif3(...),
}
```

을 사용한다.

---

# 20. Architecture freedom 시작점

독립 repository가 만들어지고 `v0.1.0` parity release가 나온 이후부터는 기존 migration 단계에서 금지했던 architectural change를 검토할 수 있다.

예:

```text
Tokio / async runtime
HTTP stack 교체
scheduler redesign
request admission redesign
typed wire session
better internal API
model-family abstraction
crate boundary 재설계
```

단 하나씩 한다.

---

# 21. Async runtime 도입 여부

독립 후 Tokio/Axum 등을 도입할 수는 있지만 필수는 아니다.

판단 기준:

```text
현재 blocking architecture 병목 존재?
HTTP concurrency가 GPU scheduling보다 병목인가?
thread count 증가가 실제 문제인가?
stream fan-out 비용이 의미 있는가?
```

가 먼저다.

단순히:

> Rust니까 Tokio

식으로 도입하지 않는다.

---

# 22. Server architecture redesign 시 Gate

서버 runtime을 변경한다면 GPU inference를 건드리지 않고 먼저 wire parity를 확보한다.

```text
old Rust server
      │
      ▼
native runtime

new async server
      │
      ▼
same native runtime
```

형태로 A/B한다.

검증:

```text
schema
stream ordering
finish semantics
continuation
backpressure
disconnect
timeout
admission
TTFT
```

---

# 23. Scheduler redesign

scheduler 변경은 별도 milestone로 둔다.

예:

```text
Milestone: Scheduler v2
```

절대:

```text
Rust cleanup
+
continuous batching redesign
+
new admission policy
```

를 한 작업으로 묶지 않는다.

성능 regression의 attribution이 불가능해진다.

---

# 24. Performance Baseline 재정의

`dfm-rs v0.1.0`을 새 baseline으로 만든다.

문서:

```text
docs/PERFORMANCE.md
```

예:

```text
Reference release:
dfm-rs v0.1.0

Hardware:
DGX Spark / GB10

CUDA:
...

Models:
...

Prefill:
...

Decode:
...

TTFT:
...

Context:
...

Memory:
...
```

이후 모든 major optimization은 이 baseline 또는 직전 stable release와 비교한다.

---

# 25. Benchmark naming

benchmark result에는 반드시:

```text
commit SHA
release
model revision
quant revision
GPU
driver
CUDA
context
batch/width
KV mode
```

를 기록한다.

`tok/s` 숫자만 README에 남기지 않는다.

---

# 26. Release Gate 재정의

독립 repo stable release에는 최소:

```text
cargo fmt --check
cargo clippy
cargo test
native compile
API fixture suite
KV suite
model-family tests
CUDA smoke
CUDA long
native tracked-golden OPP-C
C→Rust host OPP-C with explicit binary paths
default/oracle binary SHA mapping
ds4-eval --self-test-extractors
Phase 9 pre/post family manifest replay
performance proof
server soak
```

를 요구한다.

CUDA production path에서 long-context proof는 계속 release gate다.

---

# 27. Compatibility 정책

`v0.x` 동안 compatibility를 두 층으로 나눈다.

## Public compatibility

가능하면 유지:

```text
HTTP API
CLI core flags
model loading
checkpoint format
observable sampling semantics
```

## Internal compatibility

보장하지 않음:

```text
crate API
Rust types
native ABI
internal file layout
scheduler internals
```

README에 명시한다.

---

# 28. CLI compatibility

기존 ds4-dfm CLI 사용자를 고려해 초기에는 기존 주요 flag를 alias로 유지할 수 있다.

예:

```text
old flag
→ deprecated alias
→ new canonical flag
```

하지만 compatibility alias를 영구 유지하지 않는다.

`v0.x` 동안 deprecation window를 두고 정리한다.

---

# 29. Config / Environment Variable 정리

기존:

```text
DS4_*
```

환경변수는 초기에는 compatibility alias로 유지한다.

새 namespace:

```text
DFM_*
```

을 추가할 수 있다.

전환 단계:

```text
DFM_* preferred
DS4_* compatibility
```

향후 stable major version에서 old namespace 제거를 검토한다.

---

# 30. Binary 이름

권장:

```text
dfm
dfm-server
dfm-bench
dfm-agent
ds4-eval  (retained C reference/extractor)
```

필요시 transitional symlink / wrapper:

```text
ds4
ds4-server
```

를 유지할 수 있다.

단 README의 canonical command는 `dfm-*`로 바꾼다.

---

# 31. Observability

Rust migration의 장점을 활용해 error taxonomy와 metrics internals를 type-safe하게 만들 수 있다.

예:

```rust
enum RuntimeError {
    Admission(...),
    Backend(...),
    Kv(...),
    Protocol(...),
    Model(...),
}
```

단 외부 HTTP error envelope는 compatibility contract를 깨지 않는다.

내부 error type 개선과 wire behavior 변경을 구분한다.

---

# 32. Error handling 정책

독립 이후에도 production runtime에서 다음 사용을 최소화한다.

```rust
unwrap()
expect()
panic!()
```

특히:

```text
request parsing
session lifecycle
CUDA resource handling
streaming
KV persistence
distributed runtime
```

에서는 recoverable error를 typed `Result`로 처리한다.

panic은 programmer invariant 위반에 한정한다.

---

# 33. Unsafe 정책

Repository root에:

```text
docs/UNSAFE_POLICY.md
```

생성을 권장한다.

원칙:

```text
unsafe 허용 위치:
- dfm-sys
- CUDA/Metal FFI wrappers
- mmap boundary
- carefully reviewed zero-copy primitives
```

그 외 crate에서 unsafe를 사용할 경우 rationale을 요구한다.

CI에서 필요하면:

```bash
rg "unsafe" crates/
```

결과를 audit artifact로 남긴다.

---

# 34. Memory ownership 문서화

Rust migration의 주요 목적 중 하나이므로:

```text
docs/MEMORY_MODEL.md
```

을 별도로 작성한다.

최소:

```text
Model ownership
Weight mapping lifetime
CUDA allocation lifetime
Session lifetime
KV lifetime
Batch bank ownership
Continuation ownership
Scratch allocation
Shutdown order
```

를 명시한다.

---

# 35. Thread / Concurrency model 문서화

독립 repository에서:

```text
docs/CONCURRENCY.md
```

를 작성한다.

내용:

```text
accept thread/runtime
request worker
scheduler
GPU submission
stream writer
distributed workers
shutdown
```

그리고 lock-order / shared-state ownership을 문서화한다.

---

# 36. Upstream ds4 추적 정책

독립화했다고 upstream을 완전히 무시하지 않는다.

단 자동 merge 관계는 종료한다.

정책:

```text
antirez/ds4
Entrpi/ds4
Baekpica/ds4
```

에서 유용한:

```text
correctness fixes
CUDA fixes
model semantics fixes
vendor updates
```

만 수동으로 분석해서 port한다.

---

# 37. Upstream Port Log

다음 문서를 생성한다.

```text
docs/UPSTREAM_PORTS.md
```

예:

```text
Source:
Entrpi/ds4 @ <SHA>

Topic:
CUDA graph stale pos fix

dfm-rs commit:
<SHA>

Port type:
manual semantic port

Notes:
...
```

새 repo에서는 cherry-pick 가능 여부보다 **semantic provenance**가 중요하다.

---

# 38. Vendor 관리

`cuda/mmq/`처럼 외부 vendor source가 존재하면 pin 문서를 유지한다.

예:

```text
cuda/mmq/VENDOR.md
```

반드시 포함:

```text
upstream repo
upstream commit
local patches
sync procedure
validation
```

독립 repo rename 과정에서 vendor provenance를 잃지 않는다.

---

# 39. Issue / Milestone 체계

초기 milestone 추천:

```text
v0.1 parity release
v0.2 host cleanup
v0.3 architecture cleanup
v0.4 scheduler/runtime improvements
v1.0 compatibility contract
```

실제 version은 필요에 따라 조정한다.

중요한 것은 migration / cleanup / optimization을 분리하는 것이다.

---

# 40. `v1.0` 의미

`v1.0`은 단순히 성능이 좋아진 시점이 아니다.

권장 조건:

```text
Rust host architecture stable
public CLI policy stable
HTTP compatibility policy stable
checkpoint compatibility policy stable
model-family extension mechanism stable
native backend boundary stable enough
release gates automated/reproducible
```

그 전까지는 `v0.x`에서 internal API를 자유롭게 개선한다.

---

# 41. README 개편

독립 repo README는 기존 fork README를 그대로 유지하지 않는다.

추천 구성:

```text
1. What is dfm-rs?
2. Hardware / backend support
3. Supported model families
4. Quick start
5. Performance
6. Architecture
7. API compatibility
8. DFM lineage
9. Build
10. Testing / proof methodology
11. License / acknowledgements
```

---

# 42. Project positioning

프로젝트 설명은 다음 방향을 권장한다.

좋음:

> Rust-first local inference runtime derived from ds4-dfm, preserving optimized CUDA/MMQ kernels while moving host runtime ownership, serving, session, and memory orchestration into Rust.

피할 표현:

> ds4 rewritten in Rust

이 표현은 CUDA/native code와 technical lineage를 지나치게 단순화한다.

---

# 43. 성능 claim 정책

README 또는 release note에:

```text
4x faster
1000 tok/s
```

같은 claim을 넣을 경우 반드시 benchmark conditions를 연결한다.

특히 hardware-specific optimization이면:

```text
DGX Spark / GB10 / sm_121a
```

를 명시한다.

일반 CUDA runtime 전체의 성능으로 오해하게 만들지 않는다.

---

# 44. Model support tier

독립화 후 model family support를 tier로 분류하는 것을 권장한다.

예:

```text
Tier 1
- DeepSeek
- Solar Open2
- K-EXAONE
- Motif-3

Tier 2
- experimental families

Reference only
- ...
```

각 tier는 test requirement가 달라야 한다.

---

# 45. Hardware support tier

예:

```text
Tier 1:
- NVIDIA GB10 / sm_121a

Tier 2:
- other Blackwell

Build-only:
- Metal

Experimental:
- other CUDA architectures
```

실제로 검증하지 않은 hardware를 generic support로 표기하지 않는다.

---

# 46. Docker / Packaging

초기 split과 동시에 packaging 생태계를 과도하게 확장하지 않는다.

순서:

```text
source build 안정화
→ release binary
→ container image
→ package manager
```

Docker / prebuilt binary는 core release가 안정된 뒤 진행한다.

---

# 47. Prebuilt Binary

제공한다면 release artifact에:

```text
commit
rustc version
CUDA requirements
minimum driver
target architecture
sha256
```

를 명시한다.

CUDA architecture-specific binary는 이름에도 반영한다.

예:

```text
dfm-server-v0.1.0-linux-aarch64-gb10
```

---

# 48. Reproducibility

가능하면 release build command를 고정한다.

예:

```bash
cargo build --release --locked
```

native CUDA build 역시 deterministic command를 문서화한다.

`Cargo.lock`은 application repository이므로 commit한다.

---

# 49. Security / Supply Chain

독립 repo 이후 다음을 추가 검토한다.

```text
cargo audit
cargo deny
dependency license checks
GitHub Dependabot
```

단 이것들이 inference release gate를 대신하지 않는다.

---

# 50. Documentation split

기존 migration 문서:

```text
docs/rust-migration/*
```

은 완료 후:

```text
docs/history/rust-migration/
```

등으로 이동할 수 있다.

삭제하지 않는다.

이 문서는 architecture decision provenance로 가치가 있다.

---

# 51. ADR 도입

독립 이후 큰 architecture decision에는 간단한 ADR 사용을 권장한다.

예:

```text
docs/adr/
0001-rust-host.md
0002-native-cuda-boundary.md
0003-server-runtime.md
0004-model-family-dispatch.md
```

긴 문서보다 decision / rationale / consequences에 집중한다.

---

# 52. 성능 회귀 관리

PR에서 runtime-sensitive change는:

```text
perf-sensitive
```

label을 붙이고 benchmark 결과를 요구한다.

예:

```text
scheduler
FFI
memory
KV
CUDA launch
graph
batching
tokenizer hot path
```

---

# 53. Profiling workflow

독립 repo에서는 공식 profiling workflow를 문서화한다.

예:

```text
nsys
ncu
host flamegraph
heap/profile
```

그리고 benchmark와 profiler run command를 version control한다.

예:

```text
tools/profile/
├── nsys_prefill.sh
├── nsys_decode.sh
├── ncu_mmq.sh
└── host_flamegraph.sh
```

---

# 54. Optimization policy

Optimization은 항상:

```text
hypothesis
→ profile
→ patch
→ correctness proof
→ speed proof
```

순서로 한다.

"Rust로 바꿨으니 빨라질 것" 같은 언어-level assumption을 성능 근거로 사용하지 않는다.

---

# 55. Repo 분리 완료 Definition of Done

다음이 모두 충족되면 repository split 작업은 완료로 본다.

```text
Baekpica/dfm-rs exists as non-fork
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
old Baekpica/ds4 points users to dfm-rs
```

---

# 56. 기존 `Baekpica/ds4` 처리

새 repository 생성 직후 기존 repo를 삭제하지 않는다.

권장 상태:

```text
Baekpica/ds4
Status:
DFM C/Rust migration reference repository

Active development moved to:
Baekpica/dfm-rs
```

README 상단에 migration notice를 둔다.

---

# 57. 기존 Repo Archive 시점

다음 조건 이후 archive를 검토한다.

```text
dfm-rs stable release ≥ 1
critical regression 없음
documentation migration 완료
issue/PR migration 필요 없음
old repo 신규 개발 없음
```

바로 archive할 필요는 없다.

---

# 58. 최종 Architecture 목표

독립화 이후 최종적인 방향은 다음이다.

```text
                    dfm-rs
┌────────────────────────────────────────┐
│                Rust                    │
│                                        │
│ API / protocol                         │
│ scheduler                              │
│ admission                              │
│ model lifecycle                        │
│ session                                │
│ KV                                     │
│ memory policy                          │
│ distributed runtime                    │
│ observability                          │
└──────────────────┬─────────────────────┘
                   │
             narrow native ABI
                   │
┌──────────────────▼─────────────────────┐
│             Native backend             │
│                                        │
│ CUDA runtime / driver                  │
│ VMM                                    │
│ CUDA Graph                             │
│ MMQ                                    │
│ fused attention                        │
│ MoE kernels                            │
│ Metal backend                          │
└────────────────────────────────────────┘
```

---

# 59. 독립화 이후의 원칙

Migration 이전:

> **Behavior preservation first**

독립화 직후:

> **Stabilize the new ownership boundary**

그 이후:

> **Architecture and performance evolution**

순서를 지킨다.

즉 새 repo를 만들자마자 모든 architecture를 다시 뒤집지 않는다.

---

# 60. 최종 요약

이 후속 작업의 목적은 단순한 repository rename이 아니다.

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
dfm-rs v0.1 parity release
     ↓
legacy host C cleanup
     ↓
independent architecture evolution
```

의 순서로 **프로젝트의 provenance는 보존하면서 lifecycle만 독립시키는 것**이다.

`dfm-rs`가 생성된 직후에는 여전히 `ds4-dfm v0.6.3-dfm`의 semantics와 성능이 기준점이다.

새 repository의 진짜 독립성은 source history를 끊는 데서 오는 것이 아니라:

- Rust가 host runtime의 ownership을 완전히 담당하고
- CUDA backend boundary가 명확하며
- release/version policy가 독립되고
- upstream 변화가 자동 merge가 아닌 선택적 semantic port가 되고
- 자체 performance/correctness gate를 갖는 것

에서 나온다.
