# DS4 API surface matrix

Status: v0.5.6 Inc 0a baseline. This document records what each HTTP
generation surface supports TODAY, which serving lane executes it, and the
known gaps. It is the frozen oracle for the API-promotion arc: later
increments change routing and must update this file in the same commit.

DS4 serves four wire surfaces. "Surface" means a distinct wire contract
(object types, stream events, ID scheme, finish mapping, error envelope),
not an API vendor. OpenAI Chat and legacy Completions are different
surfaces: they share an endpoint family but stream different objects.

| Surface | Endpoint | Buffered object | Stream objects |
|---|---|---|---|
| OpenAI Chat | `POST /v1/chat/completions` | `chat.completion` | `chat.completion.chunk` deltas, `[DONE]` |
| OpenAI Completion | `POST /v1/completions` | `text_completion` | `text_completion` chunks with `choices[].text`, `[DONE]` |
| Anthropic Messages | `POST /v1/messages` | `type:"message"` | `message_start` .. `content_block_*` .. `message_delta`, `message_stop` |
| OpenAI Responses | `POST /v1/responses` | `object:"response"` | `response.created` .. item/delta events with `sequence_number`, `response.completed` |

## Serving lanes and current routing

Three lanes serve generation requests:

- **serial** (`generate_job`): one request at a time on the session graph.
  Full feature set, including live tool continuation and corrective tool
  recovery.
- **continuous** (`generate_continuous_jobs`): the batched engine with
  per-row sampling, streaming, stops, and tools.
- **static** (`generate_batch_jobs`): coalesced buffered greedy batches.

Routing today (`job_is_batchable`, `job_is_static_batchable`,
`worker_main`):

| Surface | serial | continuous | static |
|---|---|---|---|
| OpenAI Chat | fallback | yes (prompt fits one bank) | buffered + greedy + non-thinking + no stops/tools |
| OpenAI Completion | fallback | yes | same conditions as Chat |
| Anthropic Messages | always | never | never |
| OpenAI Responses | always | never | never |

The Anthropic/Responses exclusion is by API identity alone: the first line
of `job_is_batchable` returns false for `r->api != API_OPENAI`. It is a
projection scoping choice, not an engine capability limit — the engine is
protocol-blind. Promoting these surfaces onto the batched lanes is the
goal of this arc.

Within OpenAI, `job_is_batchable` also keeps on serial: non-streaming
`return_token_ids` chat, completion-kind requests with
`return_token_ids`, and completion-kind requests carrying tools.

## Output-budget (`max_tokens`) semantics today

All four parsers accept any integer without range validation. An omitted
budget becomes the server default (`--tokens`, default 393216). The lanes
then disagree on `<= 0`:

- **serial** honors it: negative clamps to 0, and 0 runs prefill with zero
  decode tokens (finish `length`; Anthropic projects `stop_reason:
  "max_tokens"` with empty content).
- **continuous and static** replace any `<= 0` with the server default,
  turning a zero-token request into a full-budget decode.

Because Anthropic always routes serial today, an Anthropic
`max_tokens: 0` request currently behaves like upstream's documented
cache-prewarm (prefill, empty content, `stop_reason: "max_tokens"`).
That behavior must survive route promotion: budget normalization becomes
route-invariant in Inc 0b before any routing changes.

The Anthropic parser requires `messages` but does not require
`max_tokens` (upstream requires it); an omitted value gets the server
default like every other surface.

## Explicitly unsupported

- **Responses durable references**: non-null `previous_response_id` or
  `conversation` values are rejected at parse time with
  "not supported; replay full input instead". DS4 serves a stateless
  Responses subset; clients replay full history. Literal `null` is
  accepted and ignored.
- **`Idempotency-Key`**: the HTTP reader parses only `Content-Length` and
  `Accept`; the header is accepted and discarded, so a retry is a new
  generation with new IDs.
- **`/v1/batch`** is a bulk scheduling consumer, not a projection surface.
- **`return_token_ids`** is an OpenAI Chat extension only.

## Known defects recorded as fixtures (fixed in later increments)

1. **FIXED (Inc 0b): continuous legacy-Completion streaming emitted chat
   deltas.** `cont_on_token` projected every streaming row through the
   chat delta machine, so a `text_completion` client received
   `chat.completion.chunk` objects. Completion rows now stream the
   serial oracle's plain `text_completion` chunks
   (`cont_stream_emit_plain`); the Inc 0a negative fixture is inverted
   (`test_cont_completion_stream_matches_serial_oracle`) and
   `speed-bench/completion_stream_gate.sh` holds the live schema +
   cont-engagement line.
2. **Engine-failure stranding finalizes live streams as `length`**
   (continuous lane), conflating a failure with a budget stop.
3. **`http_error` sends the OpenAI envelope to every surface** for parse,
   409, 503, and shutdown failures. Only context-length errors and stream
   errors partially branch per protocol
   (`http_error_context_length_exceeded`, `sse_error_event`).
4. **Admission accounting drops decode-growth commitments** once a bank's
   prefill lands (the `outstanding` charge covers pending prompt targets
   only); lifetime admission credits land before eligibility broadens.
5. **The v0.5.5 budget-cut honesty fix (#13) is serial-only.** On the
   continuous lane — where chat+tools actually routes — a `max_tokens`
   cut landing inside a tool call either silently repairs to a full
   `finish="tool_calls"` success or, when unrepairable, reports
   `finish="error"`; the serial lane reports the honest `length` with
   the partial call as content and logs
   "tool call cut by token budget". The two lanes therefore disagree on
   the same cut, and `finish_reason_gate.sh`'s engagement oracle (that
   serial log line) cannot fire on a cont-routed run — the gate as
   written only proves the serial fix. Found by the Inc 0a baseline
   battery (deterministic fail, reproduced on the tip-parent binary);
   the cont-side honesty fix and a lane-aware gate oracle belong to the
   typed-outcome work.

## Recorded quirks (current behavior, not upstream-shaped)

- **Anthropic response IDs**: the serial lane mints one `chatcmpl-N` /
  `cmpl-N` job ID for every surface, and `anthropic_final_response` /
  the Anthropic stream use it directly — so Anthropic clients see
  `"id":"chatcmpl-N"` instead of an upstream-shaped `msg_*` ID.
  Responses is unaffected (`responses_final_response` and
  `responses_stream_init` mint their own `resp_*`/`rs_*`/`msg_*` IDs).
  Identity minting moves into the typed wire session in a later
  increment; until then this is frozen, documented behavior.

## Route observation metrics

`GET /metrics` exposes `ds4_route_requests_total{surface=...,lane=...}`
(fixed cardinality: 4 surfaces x 3 lanes, all cells always emitted);
`GET /v1/stats` mirrors it as the `routes` section. One increment per job
at the moment a lane takes it — a failed batched attempt that falls back
to serial increments both lanes. These counters are observation only;
they exist so route promotion can prove engagement (an eligible request
actually moved lanes) instead of inferring it.

## Fixture inventory (`ds4_test --server`)

Deterministic token/text tapes replayed through the CURRENT projectors,
validated by protocol event validators (event order, one open/close per
block/item, contiguous Responses `sequence_number`, UTF-8 hold-back):

- `test_tape_openai_chat_stream_projection` (thinking + UTF-8-split tapes)
- `test_tape_openai_completion_stream_projection` (the oracle for defect 1)
- `test_tape_anthropic_stream_projection`
- `test_tape_responses_stream_projection`
- `test_tape_buffered_final_responses` (all four buffered objects +
  finish mapping, including Anthropic `length -> max_tokens`)
- `test_cont_completion_stream_matches_serial_oracle` (the inverted
  Inc 0a negative fixture for defect 1 — cont and serial now share the
  legacy-Completion stream shape)
- `test_route_decisions_record_current_dispatch`
- `test_idempotency_key_header_is_ignored`
- `test_responses_durable_references_rejected_at_parse`
- `test_error_envelopes_record_current_shapes` (defect 3)

Existing per-feature streaming tests (`test_openai_tool_stream_*`,
`test_anthropic_tool_stream_*`, `test_responses_*`) cover tool-call
projection per surface and remain part of the oracle.

Live-sampled output is never a byte oracle: continuous temp-0 emissions
jitter run-to-run, so live end-to-end gates assert schema, event automata,
route engagement, and semantic equivalence — not byte identity.
