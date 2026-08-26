> **SUPERSEDED (2026-08-26):** 이 초안은 tracked `DFM_RS_SPLIT_PLAN.md`로
> 통합됐다("실행 전제 변경 기록" 절). 이 파일은 provenance 보존용이며,
> 이 파일과 `DFM_RS_SPLIT_PLAN.md`가 다르면 후자가 이긴다.

# ds4-dfm C → Rust 마이그레이션 작업 지시서  
### 범위: `v0.6.3-dfm` 기준선 확정 → `dfm-rs` 독립 repository 생성 직전

> **목표:** `ds4-dfm`의 CUDA/MMQ 최적화 자산과 동작 계약을 보존하면서, production host runtime을 C에서 Rust로 단계적으로 이전한다.  
> **완료 시점:** Rust host가 CUDA production path의 기본 runtime이 되고, 기존 C 구현은 reference/oracle 또는 제한된 compatibility code로만 남아 독립 `dfm-rs` repository로 분리할 수 있는 상태.

현재 repository의 `AGENT.md`는 correctness-before-speed, mmap-backed loading, narrow public API, long-context captured/eager parity를 핵심 원칙으로 두고 있다. 이 원칙은 Rust migration에서도 그대로 유지한다.   
또한 API surface matrix는 네 가지 wire surface와 serving lane semantics를 이미 frozen oracle로 정의하고 있으므로, Rust server는 새로운 동작을 설계하지 않고 이 계약을 재현해야 한다. 

---

## 실행 현황 보충 — 2026-08-25 KST

이 절은 아래의 원래 Phase/gate를 완화하지 않는다. 현재 상태의
source of truth는 다음 우선순위를 따른다.

1. 커밋된 revision의 상태: `docs/rust-migration/STATUS.md`
2. 같은 revision의 gate 정의와 실측 증거:
   `docs/rust-migration/PARITY_MATRIX.md`
3. 현재 dirty 실험과 재개 순서:
   `docs/rust-migration/HANDOFF-2026-08-24-KST.md`

`STATUS.md`나 `PARITY_MATRIX.md` 자체가 working-tree diff에 포함돼 있으면
그 서술은 commit 전 제안 상태이며 HEAD의 완료 증거로 읽지 않는다.

현재 source tip과 `origin/rust-host`는 `042562b`에서 일치한다. Bank live
evidence는 `STATUS.md`와 `PARITY_MATRIX.md`에 commit/push됐고, tracked
working tree는 clean이다.

현재 정직한 phase paint는 다음과 같다.

| Phase | 상태 |
|---|---|
| 0–1 | 완료 |
| 2–3 | wrapper/shadow gate는 green, production coverage는 partial |
| 4 | **진행 중** — ordinary/default cold, final/decode continued, scoped DeepSeek intermediate-prefill/tool-map, scoped width-1 bank-shutdown four-way/ABBA는 green; default-policy, bank-evict, multi-bank는 pending |
| 5–6 | isolated parity green, production integration pending |
| 7–8 | partial |
| 9 | 시작하지 않음 |
| split | `SPLIT_READINESS.md`가 없으므로 금지 |

Phase 4의 남은 작업은 하나의 큰 commit으로 묶지 않고 다음 수직
slice로 분리한다.

1. ordinary serial final-sync/decode continued checkpoint — `0e9eed1`, scoped gate green
2. tool-map checkpoint replay — payload-skipping trailer seam `4958905`,
   C↔Rust KTM wire contract `0104689`, process-unique IDs `13ae91a`, bounded
   production memory/filter/restore/store `cc1bf0f`; DeepSeek OpenAI Chat
   no-think live four-way green
3. intermediate-prefill continued staging — additive progress FFI `49a2b65`,
   scoped ordinary-serial production integration `8361116`; DeepSeek CUDA
   no-think/no-tools 6,782-token prompt 내부의 4,096 frontier가 live
   four-way green (`b170c79`에 evidence 기록)
4. scoped width-1 continuous-bank checkpoint — opaque bank seam `e9dfd77`,
   strict-suffix candidate `0e4a178`, production shutdown/replay `15b016c`,
   native timing fix `98d81b9`; Motif-3 OpenAI Chat no-think/no-tools
   bank-shutdown four-way와 scoped restore ABBA green. Default periodic/evict,
   full default-policy ABBA, multi-bank와 bank extension은 별도 pending

각 slice는 CPU policy test만으로 완료 처리하지 않는다. 특히 live
response text가 `RESTORED_OK`여도 `cached_tokens=0`이면 restore 실패다.
continued slice의 최소 live gate는 다음과 같다.

```text
C save    → Rust load   cached_tokens > 0
Rust save → C load      cached_tokens > 0
Rust save → Rust load   cached_tokens > 0
C save    → C load      cached_tokens > 0

AND

checkpoint reason/frontier/text/payload parity
semantic output parity
clean teardown + post-run memory recovery
```

2026-08-25 scoped continued gate는
`scratch/rust-host-live/continued-fourway-oJTYrf/`에서 통과했다. C와
Rust가 같은 6,800-token reason-continued text/payload bodies를 만들었다.
live `<think></think>` bytes를 보존하는 replay fixture로
C→Rust, Rust→C, Rust→Rust, C→C 모두 `cached_tokens=6800`과 동일 semantic
output을 확인했다. 일반 assistant-history fixture의 `cached_tokens=0`은
양쪽 renderer가 빈 think pair를 닫힌 history에서 제거하기 때문에 생기는
oracle-consistent miss이며 Rust 단독 결함이 아니다. 이 6,800-token
artifact 자체는 final-sync call/order CPU gate와 decode-frontier live slice만
green으로 올리며 intermediate-prefill, tool-map, 아래의 별도 width-1 bank
gate, default-policy ABBA를 대신하지 않는다.

2026-08-25 ordinary serial tool-map runtime은 `cc1bf0f`에서 CPU/oracle/link
gate를 통과했고, `scratch/rust-host-live/tool-fourway-20260825/`에서 scoped
live gate도 통과했다. 범위는 OpenAI Chat, DeepSeek, no-think, live protocol
ID 없음이다. C와 Rust producer는 같은 424-token reason-evict text/payload
body와 nonempty KTM을 만들었고, loader history의 arguments를
`{"a":1,"b":2}`에서 `{"b":2,"a":1}`로 뒤집은 fresh-process
C→Rust/Rust→C/Rust→Rust/C→C 네 셀이 모두 `cached_tokens=424`와 동일
`RESTORED_OK` semantics를 확인했다. pending과 settled tool history 모두
render 전에 exact sampled DSML을 복원하며, KTM은 exact checkpoint text에
포함된 block만 기록한다. 공유 block 직렬화 확장은 512 MiB에서 선제
차단하고, 차단 시 실제 저장과 cold/continued marker 전진을 모두
건너뛴다. Width-1 no-tool bank shutdown/replay는 아래에서 별도 green이지만,
bank tool-map integration, other wire surfaces, combined extension은 이후
별도 slice로 유지한다.

2026-08-25 scoped intermediate-prefill continued gate도
`scratch/rust-host-live/intermediate-prefill-fourway-40gDCF/`에서 통과했다.
Legacy sync ABI는 그대로 두고 `49a2b65`의 additive same-thread progress
callback과 `8361116`의 DeepSeek CUDA ordinary-serial integration만 사용했다.
C와 Rust producer는 6,782-token prompt의 중간 4,096 frontier에서 같은
reason-continued record를 썼다. Rendered text SHA-256 `2dfc39c8...`와 payload
SHA-256 `00d8719a...`가 정확히 같고, fresh-process
C→Rust/Rust→C/Rust→Rust/C→C 모두 cached/computed/output
4,096/2,686/5와 `RESTORED_OK`를 확인했다. 이 결과는 DeepSeek OpenAI Chat
no-think/no-tools ordinary-serial correctness/cross-read만 green으로 올린다.
이 artifact 자체는 아래의 별도 width-1 bank gate를 증명하지 않는다. 다른
family/surface, configured default-policy ABBA와 Phase 9는 여전히 pending이다.

2026-08-25 scoped width-1 continuous-bank gate는 `e9dfd77`, `0e4a178`,
`15b016c`에서 CPU/oracle/link gate를 통과했고,
`scratch/rust-host-live/bank-fourway-20260825-093322/`에서 live gate를
통과했다. 범위는 Motif-3, OpenAI Chat, no-think/no-tools, ctx 8,192,
`cold=0`, `continued=0`, periodic bank checkpoint off이다. C와 Rust producer는
같은 6,896-token reason-bank-shutdown text/payload body를 썼고,
C→Rust/Rust→C/Rust→Rust/C→C 모두 cached/computed/output 6,896/15/4와
`RESTORED_OK`를 확인했다. 같은 C record를 사용한 114-token
C→Rust→Rust→C restore ABBA도 exact output과 prefill/decode/TTFT 기준 안에서
green이다. `98d81b9`는 native continuous decode duration/token/step을 timing
porcelain에 전달하며 execution은 바꾸지 않는다. Post-fix Rust short loader는
12.3 tok/s, 1.25 tok/step으로 C의 12.0--12.1 tok/s, 1.25 tok/step과
일치했다. 이 결과는 width-1 bank-shutdown body/cross-read와 scoped
throughput/TTFT만 green으로 올린다. Default configured 10,000/effective
10,240 reason-bank-checkpoint, live bank-evict, full default-policy ABBA,
multi-bank fork/partial과 pin/claim, bank tool/thinking/extension, 다른
family/surface, peak host RSS/soak는 pending이다.

---

## 0. 전제

작업 시작 전에 다음 상태가 완료되어 있다고 가정한다.

```text
Baekpica/ds4

dfm
└── tag: v0.6.3-dfm
```

이 전제는 충족됐다. 현재 golden은 `v0.6.3-dfm`
(`516456fe35510e4fb8350396c9d88807ac1f760b`)이며 `rust-host`는 이
commit을 움직이지 않는 C oracle로 사용한다. 이후 `dfm` 전체 merge는
하지 않고 필요한 correctness/CUDA fix만 의도적으로 port한다.

### `v0.6.3-dfm`의 의미

이 태그를 단순 release가 아니라:

> **C implementation golden baseline**

으로 취급한다.

Migration 도중 C 코드가 "낡았으니 참고만 하는 코드"가 되어서는 안 된다. `v0.6.3-dfm`의 동작이 Rust 구현의 correctness oracle이다.

---

# 1. 최상위 원칙

## 1.1 Rewrite 금지

다음 방식으로 작업하지 않는다.

```text
ds4.c
   ↓ 번역
ds4.rs
```

또는:

```text
기존 코드를 보고
Rust로 처음부터 새 inference engine 작성
```

금지.

반드시 **Strangler Migration**으로 진행한다.

```text
C implementation
████████████████████████████

Rust
░░░░░░░░░░░░░░░░░░░░░░░░░░


→


C
██████████████░░░░░░░░░░░░░

Rust
░░░░░░░░░░██████████████████


→


C
██░░░░░░░░░░░░░░░░░░░░░░░░

Rust
░░██████████████████████████
```

각 단계에서 executable이 동작해야 한다.

---

## 1.2 CUDA/MMQ는 migration 대상이 아니다

다음은 그대로 유지한다.

```text
ds4_cuda.cu

cuda/mmq/
├── ds4_mmq.cu
├── ds4_mmq_d2r.cu
├── ds4_fattn.cu
├── ds4_repack.cu
├── mmid.cu
├── mmvq.cu
└── ...
```

현재 build는 CUDA Driver API, CUDA Runtime, cuBLAS와 직접 링크하며 GB10의 `sm_121a` 및 MXF4 특화 경로까지 가지고 있다. 이 자산을 Rust migration 과정에서 재작성하지 않는다. 

목표 architecture:

```text
┌─────────────────────────────────┐
│            Rust host            │
│                                 │
│ Model / Session / KV            │
│ Scheduler / Server              │
│ API / Distributed               │
│ Memory Policy                   │
└────────────────┬────────────────┘
                 │
           narrow native ABI
                 │
                 ▼
┌─────────────────────────────────┐
│       Native GPU backend        │
│                                 │
│ ds4_cuda.cu                     │
│ cuda/mmq/*.cu                   │
│ cuBLAS                          │
│ CUDA Driver API / VMM           │
└─────────────────────────────────┘
```

Rust migration을 이유로 CUDA kernel behavior를 동시에 변경하지 않는다.

---

# 2. Git / branch 전략

`v0.6.3-dfm` 태그에서 바로 migration branch를 생성한다.

```bash
git checkout v0.6.3-dfm
git checkout -b rust-host
```

구조:

```text
Baekpica/ds4

dfm
│
├── v0.6.3-dfm          ← immutable C golden baseline
│
└──── rust-host         ← C → Rust migration
```

### 중요한 규칙

`rust-host`를 개발하는 동안 `dfm`의 모든 변경을 계속 merge하지 않는다.

그렇게 하면 migration target이 계속 움직인다.

원칙:

> **v0.6.3-dfm semantics freeze**

를 적용한다.

향후 `dfm`에 중요한 CUDA 성능 개선이나 correctness fix가 들어오면:

1. C implementation에 먼저 fix 존재 여부 확인
2. 해당 change를 별도 commit으로 식별
3. migration branch에 의도적으로 port/cherry-pick
4. parity test 재실행

한다.

단순히:

```bash
git merge dfm
```

하는 방식은 금지한다.

---

# 3. Migration 문서 먼저 생성

`rust-host` 첫 commit에는 코드를 넣지 않는다.

다음 문서를 먼저 만든다.

```text
docs/rust-migration/
├── README.md
├── BASELINE.md
├── ARCHITECTURE.md
├── FFI_CONTRACT.md
├── PARITY_MATRIX.md
└── STATUS.md
```

### `BASELINE.md`

반드시 기록:

```text
C baseline tag:
v0.6.3-dfm

C baseline commit:
<SHA>

CUDA:
<version>

Driver:
<version>

Target GPU:
GB10 / sm_121a

Build:
make cuda-spark
```

그리고 대표 baseline:

```text
model
context
prefill tok/s
decode tok/s
TTFT
GPU memory
host RSS
```

을 남긴다.

---

# 4. Migration acceptance contract

Rust 구현의 "동작한다" 기준은 compile 성공이 아니다.

다음 5개 axis를 모두 통과해야 한다.

| Contract | 요구사항 |
|---|---|
| Numerical | 기존 허용 범위 내 동일 |
| Token | deterministic path 동일 |
| KV | save/restore/reuse contract 동일 |
| Wire/API | schema/state machine 동일 |
| Performance | C baseline 대비 유의미한 regression 없음 |

현재 프로젝트에서도 CUDA optimization commit은 correctness proof와 speed proof를 모두 요구하도록 명시돼 있다. 이 원칙을 Rust migration commit에도 그대로 적용한다. 

---

# 5. Phase 1 — Rust workspace 및 FFI skeleton

아직 C 코드를 port하지 않는다.

다음 structure부터 추가한다.

```text
Cargo.toml

crates/
├── ds4-sys/
├── ds4-core/
├── ds4-cli/
├── ds4-kv/
├── ds4-server/
└── ds4-dist/

native/
└── bridge/
    ├── ds4_bridge.c
    └── ds4_bridge.h
```

초기에는 필요 없는 crate를 빈 scaffold로 두어도 된다.

---

## `ds4-sys`

역할:

> Rust ↔ existing ds4 native runtime의 유일한 unsafe boundary

예:

```rust
extern "C" {
    fn ds4_bridge_model_open(...);
    fn ds4_bridge_session_create(...);
    fn ds4_bridge_eval(...);
    fn ds4_bridge_session_free(...);
}
```

### 절대 하지 말 것

`ds4.h` 전체를 그대로 Rust에 bindgen하여 내부 struct 수백 개를 expose하지 않는다.

대신:

```text
Rust
 ↓
small stable API
 ↓
ds4_bridge.h
 ↓
existing internal C APIs
```

형태로 한다.

### Handle 규칙

Rust에 C struct layout을 노출하지 않는다.

```c
typedef struct ds4_bridge_model ds4_bridge_model;
typedef struct ds4_bridge_session ds4_bridge_session;
```

처럼 opaque handle을 사용한다.

Rust:

```rust
pub struct Model {
    raw: NonNull<ds4_bridge_model>,
}
```

그리고 `Drop`에서 native destructor 호출.

---

# 6. Phase 2 — Rust safe core wrapper

`ds4-core`에서 `ds4-sys`를 감싼다.

목표:

```text
unsafe code
        ↓
┌─────────────────┐
│     ds4-sys     │
└─────────────────┘

safe Rust
        ↓
Model
Session
TokenBuffer
EvalResult
Backend
```

### 규칙

`unsafe`는 기본적으로 `ds4-sys` 또는 극히 작은 native adapter에만 허용한다.

다음 코드가 application 영역에 보이기 시작하면 architecture 문제로 판단한다.

```rust
unsafe {
    ...
}
```

---

# 7. Phase 3 — Rust shadow executable

기존 executable을 대체하지 않는다.

먼저:

```text
ds4
ds4-server
ds4-bench

+

ds4-rs
ds4-bench-rs
```

를 동시에 만든다.

Rust version은 **같은 C inference core**를 호출한다.

즉:

```text
ds4-bench
   │
   ▼
C host → CUDA

ds4-bench-rs
   │
   ▼
Rust wrapper → C core → CUDA
```

부터 비교한다.

이 단계의 목적은 Rust FFI 자체의 overhead / lifetime / linkage 문제를 제거하는 것이다.

### Gate

동일 model + 동일 arguments에서:

```text
token output
KV behavior
prefill
decode
memory
```

비교.

FFI 도입만으로 성능 차이가 생기면 이후 port를 진행하지 않는다.

---

# 8. Phase 4 — leaf host subsystem migration

이제 C host component를 외곽부터 Rust로 옮긴다.

순서는 고정한다.

## 4-A. KV store

첫 migration target:

```text
ds4_kvstore.c
→
crates/ds4-kv
```

현재 KV store는 dynamic buffer의 malloc/realloc/capacity/serialization을 직접 관리하고 있으므로 Rust migration ROI가 높다. 

반드시 유지:

- file format
- magic/version
- payload ABI
- endian representation
- key semantics
- eviction behavior
- prefix behavior
- checkpoint restore behavior

**새로운 format을 설계하지 않는다.**

C-generated checkpoint를 Rust가 읽고,

Rust-generated checkpoint를 C가 읽어야 한다.

### Hard gate

```text
C save → Rust load
Rust save → C load
Rust save → Rust load
C save → C load
```

4-way matrix.

---

# 9. Phase 5 — Web / utility subsystem

다음:

```text
ds4_web.c
→
Rust
```

현재 implementation은 socket, `poll`, `fcntl`, subprocess lifecycle, manually managed buffer를 사용한다. 

Rust에서는:

```text
OwnedFd
TcpStream
Child
Vec<u8>
PathBuf
```

등을 이용해 resource ownership을 명확히 한다.

### 이 단계에서 금지

Tokio/Axum 기반으로 전체 서버 architecture를 동시에 변경하지 않는다.

**언어 migration과 concurrency architecture redesign을 한 번에 하지 않는다.**

---

# 10. Phase 6 — Distributed runtime

다음:

```text
ds4_distributed.c
→
crates/ds4-dist
```

기존 구현은 socket + poll + pthread를 사용하고 coordinator registry를 accept thread와 inference thread가 공유한다. 

Rust에서는:

```rust
Arc<>
Mutex<>
RwLock<>
OwnedFd
JoinHandle
```

등을 사용할 수 있다.

그러나 wire protocol은 변경하지 않는다.

### Wire struct는 Rust struct를 그대로 serialize하지 않는다

금지:

```rust
#[repr(C)]
struct Work { ... }

write(&work_as_bytes)
```

대신:

```rust
encode_work()
decode_work()
```

에서 explicit integer encoding을 사용한다.

목적:

> Rust memory representation ≠ network protocol

을 유지.

---

# 11. Phase 7 — Rust server shadow implementation

이 시점부터 `ds4_server.c`를 port한다.

하지만 파일 단위 translation을 하지 않는다.

기능 단위로 분해한다.

```text
ds4-server

wire/
├── openai_chat
├── openai_completion
├── anthropic
└── responses

routing/
├── needs
└── route

runtime/
├── serial
├── continuous
└── static

server/
├── request
├── stream
├── worker
└── admission

continuation/
metrics/
```

---

## API contract

다음 네 wire surface를 모두 보존한다.

```text
POST /v1/chat/completions
POST /v1/completions
POST /v1/messages
POST /v1/responses
```

현재 API matrix상 serial / continuous / static 세 serving lane이 존재하며 surface별 route semantics가 다르므로 이를 변경하지 않는다. 

### 중요

Rust server implementation을 계기로 API를 "더 깔끔하게" 만들지 않는다.

예:

```text
ID format
finish_reason
error envelope
stream event order
continuation semantics
```

전부 legacy oracle 유지.

---

# 12. Async runtime 금지 — migration 기간

`dfm-rs` repo가 분리되기 전에는 특별한 이유가 없는 한:

```text
Tokio migration
HTTP stack redesign
async scheduler redesign
```

을 하지 않는다.

현재 concurrency semantics에 가까운:

```text
std::thread
channels
Mutex / Condvar
blocking sockets
```

부터 구현한다.

이유는 Rust migration regression과 scheduler redesign regression을 구분할 수 있어야 하기 때문이다.

`dfm-rs` 독립 후 필요하면 async architecture를 별도 프로젝트로 검토한다.

---

# 13. API parity gate

`docs/ds4-api-surface-matrix.md`를 source of truth로 사용한다.

특히 live sampled output은 현재 문서 자체가 byte oracle로 쓰지 말고:

- schema
- event automaton
- route engagement
- semantic equivalence

를 검사하라고 정의한다. 

따라서 Rust parity test도 동일하게 한다.

---

# 14. Phase 8 — `ds4.c` 분해

여기서부터 가장 위험한 단계.

현재 monolithic `ds4.c`를 그대로 `.rs` 하나로 옮기지 않는다.

우선 책임을 구분한다.

```text
ds4.c
├── GGUF loading
├── model metadata
├── tokenizer
├── CPU reference
├── session
├── KV
├── serialization
├── graph orchestration
├── backend dispatch
└── platform glue
```

그리고 Rust module로 하나씩 옮긴다.

---

## 권장 migration 순서

### 8-A. Model metadata

먼저:

```rust
ModelConfig
ModelFamily
TensorMetadata
QuantType
LayerInfo
```

등의 pure-data 영역.

FFI pointer ownership이 적기 때문에 가장 안전하다.

---

### 8-B. GGUF metadata parser / model catalog

조건:

> 기존 mmap-backed loading 유지.

현재 project goal에 full GGUF eager copy를 하지 말라고 명시돼 있다. 

따라서 Rust port도:

```text
mmap
 ↓
metadata/index
 ↓
lazy tensor access
```

를 유지한다.

전체 모델을 `Vec<u8>`로 읽는 구현 금지.

---

### 8-C. Tokenizer

tokenizer port는 golden vectors가 확보된 뒤 한다.

Gate:

```text
encode exact parity
decode exact parity
special token parity
model-family template parity
stop token parity
```

token ID sequence가 다르면 port 실패다.

---

### 8-D. Session lifecycle

Rust ownership 모델로:

```rust
Model
  └── Session
       ├── KV
       ├── backend state
       ├── scratch
       └── continuation
```

를 표현한다.

목표는 다음 질문을 compile-time model에 반영하는 것이다.

```text
누가 session을 소유하는가?
누가 KV를 소유하는가?
언제 GPU allocation을 release하는가?
어떤 object가 stream보다 오래 살아야 하는가?
```

---

### 8-E. Backend dispatch

최종적으로 Rust production path는:

```rust
enum Backend {
    Cuda(CudaBackend),
    Metal(MetalBackend),
}
```

또는 equivalent abstraction을 사용한다.

하지만 backend hot path에 필요 없는 dynamic dispatch를 넣지 않는다.

---

# 15. CUDA FFI final form

Migration 말기의 CUDA boundary는 대략 다음 정도여야 한다.

```text
Rust
 │
 ├── backend_create()
 ├── load_weights()
 ├── session_create()
 ├── prefill()
 ├── decode()
 ├── kv_save/load primitives
 └── backend_destroy()
       │
       ▼
native CUDA backend
```

CUDA 내부 세부 struct:

```text
CUstream
CUgraphExec
CUmemGenericAllocationHandle
device pointer
MMQ descriptor
kernel scratch
```

등은 Rust application code에 직접 노출하지 않는다.

---

# 16. CPU backend

CPU reference backend는 **production Rust cut-over의 blocking dependency로 삼지 않는다.**

현 C CPU backend가 correctness/reference 용도라면:

```text
production CUDA host → Rust

CPU reference → C 유지 허용
```

한다.

단 새 independent repo 생성 시 C reference code를 계속 유지할지는 별도 결정한다.

이 migration의 목적은:

> **production host runtime C dependency 제거**

이지 모든 `.c` 파일 제거가 아니다.

---

# 17. Metal

Metal도 동일하게 migration blocker로 만들지 않는다.

기존:

```text
ds4_metal.m
metal/*.metal
```

은 native backend로 유지할 수 있다.

단 build를 깨뜨리지는 않는다.

최소 기준:

```text
macOS compile
existing Metal API ABI
basic smoke
```

를 유지.

본 migration의 primary release gate는 **DFM CUDA / Linux production path**로 둔다. 

---

# 18. Phase 9 — Rust binary를 primary path로 승격

## 18.1 승격 전 hard blockers

Phase 9는 proof나 model-family test 일부만 green이라고 시작하지 않는다.
다음 항목이 모두 `STATUS.md`와 `PARITY_MATRIX.md`에서 green이어야 한다.

```text
model/session/KV/server/distributed ownership
web/distributed leaf production integration
narrow native CUDA ABI
ds4 / ds4-server / ds4-bench / ds4-agent required modes
four API surfaces and three serving lanes
KV cross-read/write and continuation/tool-map/bank behavior
current-candidate CUDA smoke/long/OPP-C
family regressions
ABBA performance, memory, soak
```

`ds4-eval`에는 Rust shadow가 없다. 초기 Phase 9에서는 이를 C
reference-only binary로 명시적으로 유지하며 Rust로 승격했다고 주장하지
않는다. production requirement가 생기면 별도 candidate와 parity gate를
먼저 추가한다.

## 18.2 C oracle 이름과 proof target 재배선

이름 전환은 다음 대응표를 하나의 atomic promotion으로 적용한다.

```text
ds4-rs        → ds4          기존 C → ds4-c
ds4-server-rs → ds4-server   기존 C → ds4-server-c
ds4-bench-rs  → ds4-bench    기존 C → ds4-bench-c
ds4-agent-rs  → ds4-agent    기존 C → ds4-agent-c
ds4-eval      → ds4-eval     C reference/extractor로 유지
```

그 다음 Rust candidate를 기본 이름으로 옮긴다. 현재
`proof-rust-cuda-opp-c`가 `./ds4`를 C oracle로 사용하므로, 승격 commit은
반드시 이를 `./ds4-c`로 재배선한다. `proof-cuda-opp-c`, 관련 build/test
dependency, rollback alias도 같은 commit에서 명시적으로 갱신한다.
proof harness는 candidate와 oracle이 같은 inode 또는 같은 binary hash면
실패해야 한다. 그렇지 않으면 Rust-vs-Rust false green이 가능하다.

## 18.3 재현 가능한 pre/post family protocol

승격 전후 regression은 같은 순서와 입력으로 실행하고 다음 manifest를
각 run directory에 남긴다.

```text
phase: pre-promotion | post-promotion
commit SHA
.ds4-cuda-config.mk SHA-256
C oracle binary path + SHA-256
Rust candidate binary path + SHA-256
model absolute path/revision/size + SHA-256 (또는 immutable manifest hash)
fixture/request path + SHA-256
GPU/driver/CUDA
relevant environment and full command
test order
per-test stdout/stderr log path
teardown and memory-recovery result
```

순서는 model-family smoke → API/KV fixtures → CUDA smoke → CUDA long →
OPP-C → ABBA/perf → soak로 고정하고, pre/post 중 하나라도 입력이나 환경이
달라지면 같은 gate로 비교하지 않는다. GPU 작업은 `make -j1`, tmux와
`../scripts/guarded-run.sh`, resident model 1개 원칙을 사용하고 각 cell의
exit code와 종료 후 compute/listener 부재를 남긴다. 전환 후에는 pre Rust
SHA가 post default SHA와, pre C SHA가 post `*-c` SHA와 각각 같아야 한다.
family target 대부분은 `ds4.o`를 직접 링크하므로 default-name 실행 smoke도
별도로 수행한다.

이 단계 전까지:

```text
ds4-server      → C oracle
ds4-server-rs   → Rust candidate
```

였다면,

parity 확보 후:

```text
ds4-server      → Rust
ds4-server-c    → legacy/reference
```

로 바꾼다.

동일하게:

```text
ds4
ds4-bench
ds4-agent
```

도 전환한다.

단 C oracle binary는 repo split 전까지 삭제하지 않는다.

---

# 19. 성능 gate

언어 migration이라는 이유로 성능 regression을 받아들이지 않는다.

GB10 기준 동일 환경에서 **AB/BA 또는 ABBA** 방식으로 측정한다.

최소:

```text
C → Rust → Rust → C
```

또는:

```text
A B B A
```

를 사용.

### Hard provisional thresholds

| Metric | Rust 허용 범위 |
|---|---:|
| Prefill | C baseline의 ≥97% |
| Decode | C baseline의 ≥98% |
| TTFT | C 대비 ≤+5% |
| Host RSS | ≤+5% |
| GPU resident | 의미 있는 증가 없음 |
| Token correctness | 기존 contract 그대로 |

단 2~3% regression도 원인이 설명되지 않으면 그냥 승인하지 않는다.

특히 inference loop가 동일 CUDA backend를 사용하는데 성능이 떨어졌다면 대개:

```text
extra memcpy
extra allocation
lock scope
FFI granularity
scheduler change
```

중 하나다.

---

# 20. CUDA correctness hard gates

기존 project rule을 그대로 유지한다.

특히:

```text
make proof-cuda-smoke
make proof-cuda-long
make proof-cuda-opp-c
make proof-rust-cuda-opp-c
```

계열의 proof contract를 Rust execution path에서도 실행 가능하게 만든다.

현재 `AGENT.md`상 long-context captured-vs-eager parity는 smoke가 아니라 release gate다. 

Rust migration 때문에 이를 완화하지 않는다.

---

# 21. Commit 규칙

각 commit은 하나의 의미만 가져야 한다.

좋음:

```text
rust: add opaque session FFI

rust: port KV checkpoint reader

rust: port KV checkpoint writer

rust: add OpenAI Chat projector parity

rust: move distributed frame decoder
```

나쁨:

```text
rust: migrate server and optimize batching
```

또는:

```text
rust rewrite
```

---

## 모든 migration commit에 요구되는 정보

commit message 또는 PR body에:

```text
Migration area:
C source replaced:

Correctness:
- tests ...
- parity ...

Performance:
- C ...
- Rust ...

Known remaining C dependency:
...
```

를 남긴다.

---

# 22. C 코드 삭제 정책

Rust 코드가 생겼다고 즉시 C code를 삭제하지 않는다.

삭제 조건:

```text
Rust implementation exists
AND
unit parity green
AND
live parity green
AND
performance green
AND
minimum soak green
```

를 만족한 subsystem에 한해서 삭제.

그리고 최초 일정 기간은 해당 C implementation을 `legacy/` 또는 baseline tag에서 쉽게 비교할 수 있어야 한다.

---

# 23. Migration status matrix

`docs/rust-migration/STATUS.md`에 항상 다음 표를 유지한다.

| Subsystem | C Oracle | Rust | Default | Parity | Perf |
|---|---|---|---|---|---|
| model metadata | yes | yes | Rust | green | n/a |
| GGUF loading | yes | yes | Rust | green | green |
| tokenizer | yes | yes | Rust | green | green |
| session | yes | yes | Rust | green | green |
| KV store | yes | yes | Rust | green | green |
| server | yes | partial | C | yellow | — |
| distributed | yes | yes | Rust | green | green |
| CUDA backend | native | unchanged | native | green | green |

한눈에 migration progress를 알 수 있어야 한다.

---

# 24. 하지 말아야 할 작업

Migration 기간에는 다음을 명시적으로 금지한다.

- CUDA/MMQ kernel rewrite
- quantization algorithm 변경
- speculative/MTP semantics 변경
- new scheduler architecture
- HTTP API cleanup
- tokenizer behavior 개선
- checkpoint format redesign
- async runtime 도입
- major dependency framework화
- API rename
- baseline golden update로 regression 숨기기
- Rust스럽게 만들기 위해 동작 계약 변경
- upstream 전체 merge로 moving target 만들기

특히:

> **Migration commit과 optimization commit을 섞지 않는다.**

---

# 25. 허용되는 optimization

Rust migration 자체로 자연스럽게 제거되는:

```text
unnecessary allocation
copy
temporary buffer
lock lifetime
resource leak
```

은 허용할 수 있다.

단 반드시:

```text
port commit
     ↓
parity established
     ↓
optimization commit
```

으로 분리한다.

---

# 26. Repo split readiness gate

아래가 전부 충족되기 전에는 `dfm-rs` repository를 만들지 않는다.

## Required

### Runtime

- Rust binary가 기본 production path
- Model lifecycle Rust
- Session lifecycle Rust
- KV Rust
- Server Rust
- Distributed runtime Rust
- Rust memory/resource ownership 정립
- CUDA backend와의 ABI narrow

### Correctness

- model-family regression green
- tokenizer golden green
- KV cross-read/write green
- API surface matrix green
- captured/eager long-context green
- proof harness green
- continuation/session green

### Performance

- GB10 prefill parity
- GB10 decode parity
- TTFT parity
- memory residency parity
- continuous batching parity
- soak green

### Architecture

Application layer에서:

```bash
grep -R "unsafe {" crates/
```

했을 때 대부분의 `unsafe`가:

```text
ds4-sys
native backend wrapper
```

에 국한되어 있어야 한다.

---

# 27. 독립 repo 직전 예상 tree

최종 `Baekpica/ds4:rust-host`는 대략:

```text
ds4/
├── Cargo.toml
├── Cargo.lock
│
├── crates/
│   ├── ds4-core/
│   ├── ds4-kv/
│   ├── ds4-server/
│   ├── ds4-dist/
│   ├── ds4-cli/
│   └── ds4-sys/
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
│   ├── proof/
│   └── ...
│
├── docs/
│   └── rust-migration/
│
└── legacy/
    └── C oracle components as required
```

정도가 적절하다.

---

# 28. `dfm-rs` 생성 조건

이 migration 프로젝트의 **마지막 작업**은 repo 생성이 아니다.

다음 문서를 작성하는 것이다.

```text
docs/rust-migration/SPLIT_READINESS.md
```

내용:

```text
Baseline:
v0.6.3-dfm

Rust host parity:
PASS

CUDA backend:
unchanged / native

Model families:
PASS ...

API surfaces:
PASS ...

KV:
PASS

Performance:
PASS

Remaining C:
ds4-eval: retained C reference/extractor tool
...

Remaining C++/CUDA:
...

Known regressions:
NONE / ...

Recommended split commit:
<SHA>
```

이 문서가 green 상태가 된 commit을:

> **dfm-rs genesis commit**

으로 지정한다.

그 다음 단계에서만 독립 non-fork repository:

```text
Baekpica/dfm-rs
```

를 생성한다.

---

# 29. 최종 Definition of Done

`dfm-rs` 분리 **직전** 상태를 한 문장으로 정의하면:

> **`v0.6.3-dfm`의 observable behavior를 golden oracle로 유지하면서, CUDA/MMQ native backend는 보존하고, model/session/KV/server/distributed 등 production host runtime의 ownership과 orchestration을 Rust가 담당하며, C implementation과 correctness/performance parity가 입증된 상태.**

그리고 코드 흐름은 최종적으로:

```text
                  Rust
          ┌──────────────────┐
request → │ API / scheduler  │
          │ session / KV     │
          │ model / memory   │
          └────────┬─────────┘
                   │
               narrow FFI
                   │
                   ▼
          ┌──────────────────┐
          │ CUDA backend     │
          │ VMM              │
          │ CUDA Graph       │
          │ MMQ / FA / MoE   │
          └────────┬─────────┘
                   │
                   ▼
                  GB10
```

가 되어야 합니다.

특히 이번 migration에서는 **“Rust답게 새로 설계한다”보다 “C가 이미 증명해 놓은 semantics와 성능을 Rust ownership model 아래로 안전하게 옮긴다”**를 계속 최우선 기준으로 두는 게 좋습니다. `dfm-rs`라는 이름으로 architectural freedom을 행사하는 시점은 **repo를 독립시킨 이후**가 적절합니다.
