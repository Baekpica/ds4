//! Continuous-lane execution: a push-based per-token stepper (pure, tape
//! testable) plus the native `ContLane` that drives the engine's rolling
//! scheduler through `ds4-core::BatchCtx`. The accept loop admits one request
//! at a time, while the persistent batch context owns every configured bank;
//! the engine-side bank/admit machinery is the same one C's continuous lane uses.
//! Corrective retry (`decode_again`) and continuation publish route serial
//! by the needs word, so this path never re-decodes.

use std::io::Write;
#[cfg(any(feature = "native", test))]
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "native")]
use ds4_kv::bank_checkpoint_due;
use ds4_kv::Store as KvStore;
#[cfg(any(feature = "native", test))]
use ds4_kv::{Reason as KvReason, EXT_BANK_REPLAY_V1};

use crate::dsml::{SampleOverride, SamplePolicy};
#[cfg(any(feature = "native", test))]
use crate::generate::ordinary_disk_cache_eligible;
use crate::generate::{
    render_prompt, responses_ids, stream_req_from_parsed, GenerateError, GenerateOutcome,
};
use crate::parse::{ParsedRequest, ToolChoice};
use crate::parse::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
use crate::render::syntax_for_model_id;
use crate::retry::{terminal_finish, truncation_outcome, TruncationOutcome};
use crate::route::{decode_budget, think_mode_enabled, Api, ReqKind};
use crate::stream::{
    anthropic_final_response, anthropic_sse_finish_live, anthropic_sse_start_live,
    anthropic_sse_stream_update, final_response, openai_sse_finish_live, openai_sse_stream_update,
    openai_stream_start, responses_final_response, responses_sse_created,
    responses_sse_finish_live, responses_sse_stream_update, responses_stream_init, sse_chunk,
    sse_done, sse_headers, AnthropicStream, OpenaiStream, ReqTimings, ResponsesStream, StreamReq,
    Writer,
};
use crate::tools::{assign_tool_ids, parse_generated_for_response, SemAccum};

/// Pure continuous-request stepper. The caller feeds decoded pieces and
/// receives wire bytes; the engine (or a tape) owns token production.
/// The four frozen wire surfaces share this stepper; route eligibility still
/// decides which request shapes may enter the continuous lane.
pub struct ContStepper {
    pub req: StreamReq,
    pub job_id: String,
    pub model_id: i32,
    pub prompt: Vec<u8>,
    pub prompt_n: i32,
    pub max_tokens: i32,
    acc: SemAccum,
    w: Writer,
    oa: Option<OpenaiStream>,
    anth: Option<AnthropicStream>,
    resp: Option<ResponsesStream>,
    finish: &'static str,
    stops: Vec<String>,
    think_mode: crate::route::ThinkMode,
    tool_choice: ToolChoice,
    has_tool_results: bool,
    parsed_max_tokens: i32,
    required_tool_prefix: Vec<i32>,
    required_think_end_prefix: Vec<i32>,
    started: bool,
}

pub struct ContStep {
    pub bytes: Vec<u8>,
    /// Set when the host wants the sequence aborted (stop hit / tool block
    /// closed). The engine's own EOS/budget finish arrives via `finalize`.
    pub done: bool,
}

impl ContStepper {
    pub fn new(
        parsed: &ParsedRequest,
        model_id: i32,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        prompt: Vec<u8>,
        prompt_n: i32,
        seq_room: i32,
    ) -> (Self, Vec<u8>) {
        let req = stream_req_from_parsed(parsed, model_id);
        let mut max_tokens =
            decode_budget(parsed.max_tokens_set, parsed.max_tokens, default_tokens);
        let room = seq_room - prompt_n;
        if room >= 0 && max_tokens > room {
            max_tokens = room;
        }
        let acc = SemAccum::init(
            parsed.kind == ReqKind::Chat,
            parsed.has_tools,
            think_mode_enabled(parsed.think_mode),
            req.chat_format,
            &prompt,
        );
        let mut w = Writer::new(created);
        let mut oa = None;
        let mut anth = None;
        let mut resp = None;
        if req.stream {
            w.out.extend_from_slice(&sse_headers(cors));
            match req.api {
                Api::Openai if req.kind == ReqKind::Chat => {
                    let mut st = openai_stream_start(&req);
                    st.tool.use_random_ids();
                    oa = Some(st);
                    sse_chunk(&mut w, &req, job_id, None, None);
                }
                Api::Anthropic => {
                    let mut st = anthropic_sse_start_live(&mut w, &req, job_id, prompt_n);
                    st.tool.use_random_ids();
                    anth = Some(st);
                }
                Api::Responses => {
                    let (response_id, reasoning_id, message_id) = responses_ids(job_id);
                    let mut st =
                        responses_stream_init(&req, &response_id, &reasoning_id, &message_id);
                    responses_sse_created(&mut w, &req, &mut st, created);
                    resp = Some(st);
                }
                Api::Openai => {}
            }
        }
        let head = std::mem::take(&mut w.out);
        (
            Self {
                req,
                job_id: job_id.to_string(),
                model_id,
                prompt,
                prompt_n,
                max_tokens,
                acc,
                w,
                oa,
                anth,
                resp,
                finish: "length",
                stops: parsed.stops.clone(),
                think_mode: parsed.think_mode,
                tool_choice: parsed.tool_choice,
                has_tool_results: parsed.has_tool_results,
                parsed_max_tokens: parsed.max_tokens,
                required_tool_prefix: parsed.required_tool_prefix.clone(),
                required_think_end_prefix: parsed.required_think_end_prefix.clone(),
                started: true,
            },
            head,
        )
    }

    /// Per-token sampling override, same policy the serial `decode_pass`
    /// consults (required prefixes + DSML structural greedy).
    pub fn sample_override(&mut self) -> SampleOverride {
        let policy = SamplePolicy {
            tool_choice: self.tool_choice,
            has_tool_results: self.has_tool_results,
            think_mode: self.think_mode,
            max_tokens: self.parsed_max_tokens,
            required_tool_prefix: &self.required_tool_prefix,
            required_think_end_prefix: &self.required_think_end_prefix,
        };
        self.acc.sampling_override(&policy)
    }

    /// Effective sampling block for the engine's per-seq sampler. Thinking
    /// requests pin the serial defaults, mirroring `decode_pass`.
    pub fn sampling(&self, parsed: &ParsedRequest) -> (f32, i32, f32, f32) {
        if think_mode_enabled(parsed.think_mode) {
            (DEFAULT_TEMPERATURE, 0, DEFAULT_TOP_P, DEFAULT_MIN_P)
        } else {
            (parsed.temperature, parsed.top_k, parsed.top_p, parsed.min_p)
        }
    }

    pub fn feed(&mut self, piece: &[u8]) -> ContStep {
        let feed = self.acc.feed(piece, &self.stops);
        if self.req.stream {
            let view = &self.acc.text[..feed.emit_limit.min(self.acc.text.len())];
            match (self.req.api, self.req.kind) {
                (Api::Openai, ReqKind::Completion) => {
                    if let Some(delta) = last_delta(&self.acc.text, feed.emit_limit, piece.len()) {
                        sse_chunk(&mut self.w, &self.req, &self.job_id, Some(delta), None);
                    }
                }
                (Api::Openai, ReqKind::Chat) => {
                    if let Some(st) = self.oa.as_mut() {
                        openai_sse_stream_update(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            view,
                            false,
                        );
                    }
                }
                (Api::Anthropic, _) => {
                    if let Some(st) = self.anth.as_mut() {
                        anthropic_sse_stream_update(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            view,
                            false,
                        );
                    }
                }
                (Api::Responses, _) => {
                    if let Some(st) = self.resp.as_mut() {
                        responses_sse_stream_update(&mut self.w, &self.req, st, view, false);
                    }
                }
            }
        }
        let mut done = false;
        if feed.hit_stop {
            self.finish = "stop";
            done = true;
        } else if self.acc.track_tools
            && self.acc.saw_tool_end
            && self.req.chat_format == crate::stream::ChatFormat::DeepSeek
        {
            self.finish = "tool_calls";
            done = true;
        } else if self.acc.completion >= self.max_tokens {
            self.finish = "length";
            done = true;
        }
        ContStep {
            bytes: std::mem::take(&mut self.w.out),
            done,
        }
    }

    /// Engine finished (EOS = 1, budget/abort = 0). Runs the serial tail:
    /// tag-completion repair (no re-decode on this lane), generated-message
    /// parse, tool ids, usage/timings, stream finish or buffered final.
    pub fn finalize(
        &mut self,
        engine_eos: bool,
        n_cached: i32,
        n_computed: i32,
        timings: ReqTimings,
        cors: bool,
    ) -> (Vec<u8>, GenerateOutcome) {
        assert!(self.started);
        if self.finish == "length" && engine_eos {
            self.finish = "stop";
        }
        let syntax = syntax_for_model_id(self.model_id);
        self.finish = terminal_finish(self.acc.thinking_inside(), self.finish);
        if let TruncationOutcome::Repair(text) = truncation_outcome(
            syntax,
            self.req.chat_format,
            self.req.kind == ReqKind::Chat,
            self.acc.track_tools,
            self.acc.saw_tool_start,
            self.acc.saw_tool_end,
            self.finish,
            self.req.stream,
            true, /* recovery_attempted: decode_again routes serial */
            &self.acc.text,
            &self.req.tool_orders,
        ) {
            self.acc.text = text;
            self.acc.saw_tool_end = true;
        }
        let mut parsed_gen = if self.req.kind == ReqKind::Chat {
            let (pg, recovered_finish) = parse_generated_for_response(
                syntax,
                &self.acc.text,
                self.acc.track_tools,
                self.acc.saw_tool_start,
                crate::route::think_mode_enabled(self.think_mode),
                self.req.chat_format,
                &self.req.tool_orders,
                self.finish,
            );
            self.finish = recovered_finish;
            pg
        } else {
            crate::tools::ParsedGenerated {
                content: self.acc.text.clone(),
                ok: true,
                ..Default::default()
            }
        };
        let completion = self.acc.completion;
        if !parsed_gen.calls.is_empty() {
            if let Some(st) = self.oa.as_ref() {
                st.tool.apply_ids(&mut parsed_gen.calls);
            }
            if let Some(st) = self.anth.as_ref() {
                st.tool.apply_ids(&mut parsed_gen.calls);
            }
            assign_tool_ids(
                &mut parsed_gen.calls,
                if self.req.api == Api::Anthropic {
                    "toolu_"
                } else {
                    "call_"
                },
            );
            self.finish = "tool_calls";
        }
        /* cont_usage_apply_engine_split, ordinary-request frame: whatever
         * the engine computed fresh is uncached client work (capped by the
         * prompt), the rest of the prompt was served from cache. */
        let p = self.prompt_n;
        let write = n_computed.clamp(0, p.max(0));
        self.req.cache_write_tokens = write;
        self.req.cache_read_tokens = (p - write).max(0);
        let _ = n_cached;
        self.req.timings = timings;
        if self.req.stream {
            match (self.req.api, self.req.kind) {
                (Api::Openai, ReqKind::Completion) => {
                    sse_chunk(
                        &mut self.w,
                        &self.req,
                        &self.job_id,
                        None,
                        Some(self.finish),
                    );
                    sse_done(
                        &mut self.w,
                        &self.req,
                        &self.job_id,
                        self.prompt_n,
                        completion,
                    );
                }
                (Api::Openai, ReqKind::Chat) => {
                    if let Some(st) = self.oa.as_mut() {
                        openai_sse_finish_live(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            &self.acc.text,
                            self.finish,
                            self.prompt_n,
                            completion,
                            &parsed_gen.calls,
                        );
                    }
                }
                (Api::Anthropic, _) => {
                    if let Some(st) = self.anth.as_mut() {
                        anthropic_sse_finish_live(
                            &mut self.w,
                            &self.req,
                            &self.job_id,
                            st,
                            &self.acc.text,
                            self.finish,
                            self.acc.matched_stop.as_deref(),
                            completion,
                            &parsed_gen.calls,
                        );
                    }
                }
                (Api::Responses, _) => {
                    if let Some(st) = self.resp.as_mut() {
                        let created = self.w.created;
                        responses_sse_finish_live(
                            &mut self.w,
                            &self.req,
                            st,
                            &self.acc.text,
                            self.finish,
                            self.prompt_n,
                            completion,
                            0,
                            created,
                            &parsed_gen.calls,
                        );
                    }
                }
            }
        } else {
            let bytes = match self.req.api {
                Api::Anthropic => anthropic_final_response(
                    &self.req,
                    &self.job_id,
                    &parsed_gen.content,
                    Some(&parsed_gen.reasoning),
                    self.finish,
                    self.acc.matched_stop.as_deref(),
                    self.prompt_n,
                    completion,
                    cors,
                    &parsed_gen.calls,
                ),
                Api::Responses => {
                    let (response_id, reasoning_id, message_id) = responses_ids(&self.job_id);
                    responses_final_response(
                        &self.req,
                        &parsed_gen.content,
                        Some(&parsed_gen.reasoning),
                        self.finish,
                        self.prompt_n,
                        completion,
                        0,
                        self.w.created,
                        cors,
                        &response_id,
                        &reasoning_id,
                        &message_id,
                        &parsed_gen.calls,
                    )
                }
                Api::Openai => final_response(
                    &self.req,
                    &self.job_id,
                    &parsed_gen.content,
                    Some(&parsed_gen.reasoning),
                    self.finish,
                    self.prompt_n,
                    completion,
                    self.w.created,
                    cors,
                    &parsed_gen.calls,
                ),
            };
            self.w.out.extend_from_slice(&bytes);
        }
        let outcome = GenerateOutcome {
            tool_ids: parsed_gen
                .calls
                .iter()
                .map(|c| c.id.clone())
                .filter(|id| !id.is_empty())
                .collect(),
            generation: 0,
            frontier: self.prompt_n + completion,
            finish: self.finish.to_string(),
        };
        (std::mem::take(&mut self.w.out), outcome)
    }
}

fn last_delta(raw: &[u8], emit_limit: usize, piece_len: usize) -> Option<&[u8]> {
    if emit_limit == 0 {
        return None;
    }
    let start = raw.len().saturating_sub(piece_len).min(emit_limit);
    if start >= emit_limit {
        return None;
    }
    Some(&raw[start..emit_limit])
}

#[cfg(any(feature = "native", test))]
fn bank_scope(parsed: &ParsedRequest) -> bool {
    parsed.kind == ReqKind::Chat && ordinary_disk_cache_eligible(parsed)
}

#[cfg(any(feature = "native", test))]
#[derive(Debug, Default)]
struct WarmBank {
    record: Option<WarmRecord>,
    /// Exact native committed frontier recorded at retire/restore. Banks are
    /// idle between owner calls, so this is also the C victim-picker depth.
    committed_tokens: i32,
    stored_tokens: i32,
    last_use: u64,
}

#[cfg(any(feature = "native", test))]
#[derive(Debug)]
struct WarmRecord {
    text: Vec<u8>,
    generation: u64,
}

#[cfg(any(feature = "native", test))]
fn warm_match_pick(banks: &[WarmBank], prompt: &[u8]) -> Option<usize> {
    let mut best = None;
    let mut best_len = 0;
    for (bank, state) in banks.iter().enumerate() {
        let Some(record) = state.record.as_ref() else {
            continue;
        };
        if !record.text.is_empty()
            && record.text.len() < prompt.len()
            && record.text.len() > best_len
            && prompt.starts_with(&record.text)
        {
            best = Some(bank);
            best_len = record.text.len();
        }
    }
    best
}

#[cfg(any(feature = "native", test))]
fn warm_victim_pick(
    banks: &[WarmBank],
    protected: &[bool],
    exclude: Option<usize>,
    pin_min: i32,
    evict_lru: bool,
) -> Option<usize> {
    debug_assert!(protected.is_empty() || protected.len() == banks.len());
    let mut superseded = None;
    let mut superseded_use = u64::MAX;
    let mut shallow = None;
    let mut shallow_use = u64::MAX;
    let mut deep = None;
    let mut deep_use = u64::MAX;

    for (bank, state) in banks.iter().enumerate() {
        if Some(bank) == exclude || protected.get(bank).copied().unwrap_or(false) {
            continue;
        }
        if state.record.is_none() {
            return Some(bank);
        }
        let is_deep = pin_min > 0 && state.committed_tokens >= pin_min;
        if is_deep {
            if state.last_use < deep_use {
                deep = Some(bank);
                deep_use = state.last_use;
            }
        } else if state.last_use < shallow_use {
            shallow = Some(bank);
            shallow_use = state.last_use;
        }
        if state.last_use < superseded_use && warm_record_superseded(banks, bank) {
            superseded = Some(bank);
            superseded_use = state.last_use;
        }
    }
    superseded.or_else(|| evict_lru.then_some(shallow.or(deep)).flatten())
}

#[cfg(any(feature = "native", test))]
fn warm_record_superseded(banks: &[WarmBank], bank: usize) -> bool {
    let Some(record) = banks.get(bank).and_then(|state| state.record.as_ref()) else {
        return false;
    };
    if record.text.is_empty() {
        return false;
    }
    banks.iter().enumerate().any(|(other_bank, state)| {
        other_bank != bank
            && state
                .record
                .as_ref()
                .is_some_and(|other| !other.text.is_empty() && other.text.starts_with(&record.text))
    })
}

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WarmPlacement {
    source: usize,
    target: usize,
    fork: bool,
}

#[cfg(any(feature = "native", test))]
fn warm_placement(
    banks: &[WarmBank],
    protected: &[bool],
    source: usize,
    cached: i32,
    pin_min: i32,
    fork_enabled: bool,
) -> Option<WarmPlacement> {
    let target = (fork_enabled && (pin_min <= 0 || cached < pin_min))
        .then(|| warm_victim_pick(banks, protected, Some(source), pin_min, false))
        .flatten();
    if let Some(target) = target {
        return Some(WarmPlacement {
            source,
            target,
            fork: true,
        });
    }
    if protected.get(source).copied().unwrap_or(false) {
        return None;
    }
    Some(WarmPlacement {
        source,
        target: source,
        fork: false,
    })
}

#[cfg(any(feature = "native", test))]
fn committed_key(
    prompt: &[u8],
    tokens: &[i32],
    mut token_text: impl FnMut(i32) -> Vec<u8>,
) -> Vec<u8> {
    let mut key = prompt.to_vec();
    for &token in tokens.iter().take(tokens.len().saturating_sub(1)) {
        key.extend(token_text(token));
    }
    key
}

#[cfg(any(feature = "native", test))]
fn warm_admit_tokens(
    warm: &WarmBank,
    prompt: &[u8],
    snapshot_tokens: &[i32],
    generation: u64,
    seq_cap: i32,
    tokenize_suffix: impl FnOnce(&[u8]) -> Vec<i32>,
) -> Option<(Vec<i32>, i32)> {
    let record = warm.record.as_ref()?;
    if record.generation != generation {
        return None;
    }
    let key = &record.text;
    if key.is_empty() || key.len() >= prompt.len() || !prompt.starts_with(key) {
        return None;
    }
    let cached = i32::try_from(snapshot_tokens.len())
        .ok()
        .filter(|n| *n > 0)?;
    let mut tokens = snapshot_tokens.to_vec();
    tokens.extend(tokenize_suffix(&prompt[key.len()..]));
    if tokens.len() <= snapshot_tokens.len()
        || i32::try_from(tokens.len())
            .ok()
            .filter(|n| *n <= seq_cap)
            .is_none()
    {
        return None;
    }
    Some((tokens, cached))
}

#[cfg(any(feature = "native", test))]
fn disk_restore_allowed(
    warm: &WarmBank,
    live: Option<(u64, usize)>,
    disk_tokens: u32,
    pin_min: i32,
) -> bool {
    let Some(record) = warm.record.as_ref() else {
        return true;
    };
    let Some((generation, live_tokens)) = live else {
        return true;
    };
    if record.generation != generation {
        return true;
    }
    let Ok(pin_min) = usize::try_from(pin_min) else {
        return false;
    };
    pin_min > 0 && disk_tokens as usize >= pin_min && live_tokens < pin_min
}

#[cfg(any(feature = "native", test))]
fn disk_restore_target_allowed(
    banks: &[WarmBank],
    target: usize,
    disk_tokens: u32,
    pin_min: i32,
) -> bool {
    let Some(state) = banks.get(target) else {
        return false;
    };
    if warm_record_superseded(banks, target) {
        return true;
    }
    let live = state
        .record
        .as_ref()
        .map(|record| (record.generation, state.committed_tokens.max(0) as usize));
    disk_restore_allowed(state, live, disk_tokens, pin_min)
}

#[cfg(any(feature = "native", test))]
fn bank_retire_allowed(bank_enabled: bool, admitted: bool, done_called: bool) -> bool {
    bank_enabled && admitted && done_called
}

#[cfg(any(feature = "native", test))]
fn retired_bank(
    bank_enabled: bool,
    admitted: bool,
    done_called: bool,
    actual_bank: Option<i32>,
    max_seq: i32,
) -> Option<i32> {
    if !bank_retire_allowed(bank_enabled, admitted, done_called) {
        return None;
    }
    actual_bank.filter(|bank| *bank >= 0 && *bank < max_seq)
}

#[cfg(any(feature = "native", test))]
fn save_bank_record(
    store: &mut KvStore,
    warm: &mut WarmBank,
    identity: (u8, u8, u32),
    committed: i32,
    generation: u64,
    reason: KvReason,
    save_payload: impl FnOnce(&Path) -> Result<(), GenerateError>,
) -> Result<bool, GenerateError> {
    if committed <= 0 {
        return Ok(false);
    }
    let Some(text) = warm
        .record
        .as_ref()
        .filter(|record| record.generation == generation)
        .map(|record| &record.text)
        .filter(|text| !text.is_empty())
        .cloned()
    else {
        return Ok(false);
    };
    let tokens = u32::try_from(committed)
        .map_err(|_| GenerateError::Engine("bank token count exceeds u32".into()))?;
    let (model_id, quant_bits, ctx) = identity;
    let mut header = crate::generate::kv_header(model_id, quant_bits, ctx, tokens);
    header.reason = reason;
    header.ext_flags = EXT_BANK_REPLAY_V1;
    let payload = store
        .payload_temp()
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    save_payload(payload.path())?;
    store
        .write_payload_file(header, &text, payload.path(), &[])
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    warm.stored_tokens = committed;
    Ok(true)
}

/// Trait seam so `handle_client_inner` can drive a continuous lane without
/// the native feature (tests supply a scripted implementation).
pub trait ContExec {
    fn model_id(&self) -> i32;
    fn seq_cap(&self) -> i32;
    fn encode_chat(&self, rendered: &[u8]) -> Vec<i32>;
    fn encode_text(&self, text: &str) -> Vec<i32>;
    /// `bank_hold_retry` keeps registry locking outside the native owner; a
    /// missing live reference asks the registry to preserve C's fail-closed rule.
    fn generate(
        &mut self,
        parsed: &ParsedRequest,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        t_arrive: Instant,
        bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        store: Option<&mut KvStore>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError>;

    fn shutdown(&mut self, _store: Option<&mut KvStore>) {}

    /// Static owner when this lane also holds a `BatchCtx`.
    fn as_static(&mut self) -> Option<&mut dyn crate::serve_static::StaticExec> {
        None
    }
}

/// Render + tokenize a request for routing (`prompt_len` feeds
/// `route_decide` before any lane is entered), mirroring the C server's
/// job-prep order.
pub fn cont_prompt_tokens(
    exec: &dyn ContExec,
    parsed: &ParsedRequest,
) -> Result<(Vec<u8>, Vec<i32>), GenerateError> {
    let prompt = render_prompt(parsed, exec.model_id())?;
    let tokens = match parsed.kind {
        ReqKind::Completion => exec.encode_text(std::str::from_utf8(&prompt).unwrap_or("")),
        ReqKind::Chat => exec.encode_chat(&prompt),
    };
    Ok((prompt, tokens))
}

#[cfg(feature = "native")]
pub use native::ContLane;

#[cfg(feature = "native")]
mod native {
    use super::*;

    use ds4_core::{BatchCtx, ContAdmit, ContDriver, Vocab, CONT_SAMPLE_GREEDY, CONT_SAMPLE_NONE};

    use crate::serve_static::{BatchStatic, CoalesceLimits, StaticExec, StaticJob, StaticRow};

    /// Native continuous lane: one Rust owner call at a time over every bank
    /// exposed by the persistent native batch context.
    pub struct ContLane<'m> {
        batch: BatchCtx<'m>,
        vocab: &'m Vocab,
        model_id: i32,
        quant_bits: i32,
        ctx: i32,
        /// Family EOT for the per-seq stop, like the C server's job prep;
        /// the engine's `-1` default is the base EOS, not the family EOT.
        eos: i32,
        warm: Vec<WarmBank>,
        warm_clock: u64,
        warm_fork: bool,
        warm_pin_min: i32,
        warm_checkpoint: bool,
    }

    struct WarmAdmitPlan {
        source: usize,
        tokens: Vec<i32>,
        cached: i32,
    }

    struct OneJob<'a, W: Write> {
        admit: Option<ContAdmit>,
        /* Client-visible transport is committed at on_admitted, the C
         * cont_stream_start point: a rejected request never sees bytes and
         * can fall back to the serial lane transport-clean. */
        head: Option<Vec<u8>>,
        stepper: &'a mut ContStepper,
        vocab: &'a Vocab,
        out: &'a mut W,
        admitted: bool,
        bank: Option<i32>,
        io_failed: bool,
        host_abort: bool,
        engine_eos: bool,
        capture_done: bool,
        done_tokens: Vec<i32>,
        n_cached: i32,
        n_computed: i32,
        t_arrive: Instant,
        t_admit: Option<Instant>,
        t_first: Option<Instant>,
        t_done: Option<Instant>,
        decode_ms: f64,
        decode_tokens: i32,
        decode_steps: i32,
    }

    impl<W: Write> OneJob<'_, W> {
        fn transport_alive(&mut self) -> bool {
            if !self.io_failed && self.out.flush().is_err() {
                self.io_failed = true;
            }
            !self.io_failed
        }

        fn push(&mut self, bytes: &[u8]) {
            if bytes.is_empty() || self.io_failed {
                return;
            }
            if self.out.write_all(bytes).is_err() || self.out.flush().is_err() {
                self.io_failed = true;
            }
        }
    }

    impl<W: Write> ContDriver for OneJob<'_, W> {
        fn admit(&mut self) -> Option<ContAdmit> {
            self.admit.take()
        }

        fn on_token(&mut self, _user: usize, token: i32) -> bool {
            if !self.transport_alive() {
                self.host_abort = true;
                return false;
            }
            if self.t_first.is_none() {
                self.t_first = Some(Instant::now());
            }
            /* Serial parity: the host stop set (family EOT / eos / role
             * starts) is checked before the token text is ever fed. */
            if self.vocab.is_stop(token) {
                self.stepper.mark_stop();
                self.host_abort = true;
                return false;
            }
            let piece = self.vocab.token_text(token);
            let step = self.stepper.feed(&piece);
            self.push(&step.bytes);
            if self.io_failed {
                self.host_abort = true;
                return false;
            }
            if step.done {
                self.host_abort = true;
                return false;
            }
            true
        }

        fn on_done(
            &mut self,
            _user: usize,
            tokens: &[i32],
            finish: i32,
            decode_ms: f64,
            decode_tokens: i32,
            decode_steps: i32,
        ) {
            self.engine_eos = finish == 1;
            if self.capture_done {
                self.done_tokens.extend_from_slice(tokens);
            }
            self.decode_ms = decode_ms;
            self.decode_tokens = decode_tokens;
            self.decode_steps = decode_steps;
            self.t_done = Some(Instant::now());
        }

        fn sample_override(&mut self, _user: usize) -> i32 {
            match self.stepper.sample_override() {
                SampleOverride::None => CONT_SAMPLE_NONE,
                SampleOverride::Greedy => CONT_SAMPLE_GREEDY,
                SampleOverride::Token(t) => ds4_core::cont_sample_token(t),
            }
        }

        fn alive(&mut self, _user: usize) -> bool {
            self.transport_alive()
        }

        fn on_admitted(&mut self, _user: usize, n_cached: i32, n_computed: i32, bank: i32) -> bool {
            self.n_cached = n_cached;
            self.n_computed = n_computed;
            self.bank = Some(bank);
            self.t_admit = Some(Instant::now());
            self.admitted = true;
            if let Some(head) = self.head.take() {
                self.push(&head);
            }
            !self.io_failed
        }
    }

    impl<'m> ContLane<'m> {
        pub fn new(
            batch: BatchCtx<'m>,
            vocab: &'m Vocab,
            model_id: i32,
            quant_bits: i32,
            ctx: i32,
            eos: i32,
        ) -> Self {
            let max_seq = usize::try_from(batch.max_seq().max(0)).unwrap_or(0);
            let warm_pin_min = crate::serve::env_i32_bound("DS4_SERVER_PIN_MIN_TOKENS", 65536);
            let warm_fork = std::env::var_os("DS4_SERVER_FORK").is_none_or(|value| value != "0");
            let warm_checkpoint =
                std::env::var_os("DS4_SERVER_BANK_CHECKPOINT").is_none_or(|value| value != "0");
            Self {
                batch,
                vocab,
                model_id,
                quant_bits,
                ctx,
                eos,
                warm: (0..max_seq).map(|_| WarmBank::default()).collect(),
                warm_clock: 0,
                warm_fork,
                warm_pin_min,
                warm_checkpoint,
            }
        }

        fn identity(&self) -> Option<(u8, u8, u32)> {
            crate::generate::kv_identity(self.model_id, self.quant_bits, self.ctx)
        }

        fn note_use(&mut self, bank: usize) {
            self.warm_clock = self.warm_clock.wrapping_add(1);
            if let Some(state) = self.warm.get_mut(bank) {
                state.last_use = self.warm_clock;
            }
        }

        fn warm_plan(&mut self, prompt: &[u8]) -> Option<WarmAdmitPlan> {
            let source = warm_match_pick(&self.warm, prompt)?;
            let snapshot = self.batch.bank_snapshot(i32::try_from(source).ok()?).ok()?;
            let (tokens, cached) = warm_admit_tokens(
                self.warm.get(source)?,
                prompt,
                &snapshot.tokens,
                snapshot.generation,
                self.batch.seq_cap(),
                |suffix| self.vocab.encode_rendered_bytes(suffix),
            )?;
            self.warm[source].committed_tokens = cached;
            Some(WarmAdmitPlan {
                source,
                tokens,
                cached,
            })
        }

        fn protected_banks(
            &self,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        ) -> (Vec<bool>, Option<i32>) {
            let mut protected = Vec::with_capacity(self.warm.len());
            let mut retry_min = None;
            for (bank, state) in self.warm.iter().enumerate() {
                let live = state
                    .record
                    .as_ref()
                    .map(|record| (record.generation, state.committed_tokens));
                let retry = bank_hold_retry(i32::try_from(bank).unwrap_or(-1), live);
                protected.push(retry.is_some());
                if let Some(retry) = retry {
                    retry_min = Some(retry_min.map_or(retry, |current: i32| current.min(retry)));
                }
            }
            (protected, retry_min)
        }

        fn disk_victim(&self, protected: &[bool], disk_tokens: u32) -> Option<usize> {
            if let Some(bank) =
                warm_victim_pick(&self.warm, protected, None, self.warm_pin_min, false)
            {
                return Some(bank);
            }
            if self.warm_pin_min <= 0 || disk_tokens < self.warm_pin_min as u32 {
                return None;
            }
            let bank = warm_victim_pick(&self.warm, protected, None, self.warm_pin_min, true)?;
            (self.warm[bank].committed_tokens < self.warm_pin_min).then_some(bank)
        }

        fn disk_plan(
            &mut self,
            store: &mut KvStore,
            prompt: &[u8],
            protected: &[bool],
        ) -> Option<WarmAdmitPlan> {
            let identity = self.identity()?;
            let (path, envelope) = store
                .bank_text_prefix_candidate(prompt, identity.0, identity.1, identity.2)
                .ok()??;
            let target = self.disk_victim(protected, envelope.header.tokens)?;
            if !disk_restore_target_allowed(
                &self.warm,
                target,
                envelope.header.tokens,
                self.warm_pin_min,
            ) {
                return None;
            }
            let snapshot = match self.batch.load_bank_payload_range(
                i32::try_from(target).ok()?,
                &path,
                envelope.payload_offset,
                envelope.header.payload_bytes,
            ) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    eprintln!("ds4-server-rs: bank restore skipped: {error}");
                    return None;
                }
            };
            if usize::try_from(envelope.header.tokens).ok() != Some(snapshot.tokens.len()) {
                let _ = store.discard_bank(&path);
                return None;
            }
            let committed = i32::try_from(snapshot.tokens.len()).ok()?;
            self.warm[target].record = Some(WarmRecord {
                text: envelope.text,
                generation: snapshot.generation,
            });
            self.warm[target].committed_tokens = committed;
            self.warm[target].stored_tokens = committed;
            self.note_use(target);
            let _ = store.touch_hit(&path);
            let (tokens, cached) = warm_admit_tokens(
                &self.warm[target],
                prompt,
                &snapshot.tokens,
                snapshot.generation,
                self.batch.seq_cap(),
                |suffix| self.vocab.encode_rendered_bytes(suffix),
            )?;
            Some(WarmAdmitPlan {
                source: target,
                tokens,
                cached,
            })
        }

        fn retire(&mut self, bank: i32, prompt: &[u8], done_tokens: &[i32]) {
            let Ok(bank) = usize::try_from(bank) else {
                return;
            };
            if bank >= self.warm.len() {
                return;
            }
            let Ok(snapshot) = self.batch.bank_snapshot(bank as i32) else {
                self.warm[bank].record = None;
                return;
            };
            self.warm[bank].committed_tokens = snapshot.tokens.len() as i32;
            self.note_use(bank);
            if snapshot.tokens.is_empty() {
                self.warm[bank].record = None;
                return;
            }
            let key = if done_tokens.is_empty() {
                let retained =
                    committed_key(&[], &snapshot.tokens, |token| self.vocab.token_text(token));
                if retained.is_empty() || !prompt.starts_with(&retained) {
                    self.warm[bank].record = None;
                    return;
                }
                self.warm[bank].stored_tokens = self.warm[bank]
                    .stored_tokens
                    .min(snapshot.tokens.len() as i32);
                retained
            } else {
                committed_key(prompt, done_tokens, |token| self.vocab.token_text(token))
            };
            self.warm[bank].record = (!key.is_empty()).then_some(WarmRecord {
                text: key,
                generation: snapshot.generation,
            });
        }

        fn persist_bank(
            &mut self,
            store: &mut KvStore,
            bank: usize,
            reason: KvReason,
            min_committed: i32,
            due_only: bool,
        ) {
            if min_committed <= 0 {
                return;
            }
            if bank >= self.warm.len() {
                return;
            }
            let Some(identity) = self.identity() else {
                return;
            };
            let Ok(bank_i32) = i32::try_from(bank) else {
                return;
            };
            let Ok(snapshot) = self.batch.bank_snapshot(bank_i32) else {
                return;
            };
            let Ok(committed) = i32::try_from(snapshot.tokens.len()) else {
                return;
            };
            if self.warm[bank]
                .record
                .as_ref()
                .is_none_or(|record| record.generation != snapshot.generation)
                || committed < min_committed
                || (due_only
                    && !bank_checkpoint_due(&store.opt, committed, self.warm[bank].stored_tokens))
            {
                return;
            }
            let (batch, states) = (&mut self.batch, &mut self.warm);
            let state = &mut states[bank];
            if let Err(error) = save_bank_record(
                store,
                state,
                identity,
                committed,
                snapshot.generation,
                reason,
                |path| {
                    batch
                        .save_bank_payload(bank_i32, path)
                        .map_err(|error| GenerateError::Engine(error.to_string()))
                },
            ) {
                eprintln!(
                    "ds4-server-rs: bank checkpoint failed bank={bank} reason={reason:?}: {error}"
                );
            }
        }

        fn place_warm(
            &mut self,
            plan: &WarmAdmitPlan,
            protected: &[bool],
            store: Option<&mut KvStore>,
        ) -> Option<WarmPlacement> {
            let placement = warm_placement(
                &self.warm,
                protected,
                plan.source,
                plan.cached,
                self.warm_pin_min,
                self.warm_fork,
            )?;
            self.note_use(placement.source);
            if placement.fork {
                if let Some(store) = store {
                    self.persist_bank(
                        store,
                        placement.target,
                        KvReason::BankEvict,
                        self.warm_pin_min,
                        false,
                    );
                }
                let stored = self.warm[placement.source].stored_tokens;
                self.warm[placement.target].record = None;
                self.warm[placement.target].stored_tokens = stored;
            } else {
                self.warm[placement.source].record = None;
            }
            Some(placement)
        }

        fn place_cold(&mut self, protected: &[bool], store: Option<&mut KvStore>) -> Option<usize> {
            let target = warm_victim_pick(&self.warm, protected, None, self.warm_pin_min, true)?;
            if let Some(store) = store {
                self.persist_bank(store, target, KvReason::BankEvict, self.warm_pin_min, false);
            }
            self.warm[target].record = None;
            self.warm[target].committed_tokens = 0;
            self.warm[target].stored_tokens = 0;
            Some(target)
        }

        fn shutdown_banks(&mut self, store: &mut KvStore) {
            for bank in 0..self.warm.len() {
                self.persist_bank(store, bank, KvReason::BankShutdown, 1, false);
            }
        }
    }

    impl StaticExec for ContLane<'_> {
        fn generate_static(
            &mut self,
            jobs: &[StaticJob<'_>],
        ) -> Result<Vec<StaticRow>, GenerateError> {
            BatchStatic::new(&mut self.batch).generate_static(jobs)
        }

        fn ctx_max_seq(&self) -> i32 {
            self.batch.max_seq()
        }

        fn coalesce_limits(&self) -> CoalesceLimits {
            CoalesceLimits {
                cap: self.batch.max_seq().max(1) as usize,
                max_tok_total: 0,
            }
        }
    }

    impl ContExec for ContLane<'_> {
        fn model_id(&self) -> i32 {
            self.model_id
        }

        fn as_static(&mut self) -> Option<&mut dyn StaticExec> {
            Some(self)
        }

        fn seq_cap(&self) -> i32 {
            self.batch.seq_cap()
        }

        fn encode_chat(&self, rendered: &[u8]) -> Vec<i32> {
            self.vocab.encode_rendered_bytes(rendered)
        }

        fn encode_text(&self, text: &str) -> Vec<i32> {
            self.vocab.encode_text(text)
        }

        fn generate(
            &mut self,
            parsed: &ParsedRequest,
            job_id: &str,
            created: i64,
            cors: bool,
            default_tokens: i32,
            t_arrive: Instant,
            bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            mut store: Option<&mut KvStore>,
            out: &mut dyn Write,
        ) -> Result<GenerateOutcome, GenerateError> {
            let prompt = render_prompt(parsed, self.model_id)?;
            let tokens = match parsed.kind {
                ReqKind::Completion => self
                    .vocab
                    .encode_text(std::str::from_utf8(&prompt).unwrap_or("")),
                ReqKind::Chat => self.vocab.encode_rendered_bytes(&prompt),
            };
            let prompt_n = tokens.len() as i32;
            let (mut stepper, head) = ContStepper::new(
                parsed,
                self.model_id,
                job_id,
                created,
                cors,
                default_tokens,
                prompt,
                prompt_n,
                self.batch.seq_cap(),
            );
            let (temperature, top_k, top_p, min_p) = stepper.sampling(parsed);
            let bank_enabled = bank_scope(parsed);
            let canonical_tokens = tokens;
            let (protected, hold_retry) = self.protected_banks(bank_hold_retry);
            let warm = if bank_enabled {
                self.warm_plan(&stepper.prompt).or_else(|| {
                    store
                        .as_deref_mut()
                        .and_then(|store| self.disk_plan(store, &stepper.prompt, &protected))
                })
            } else {
                None
            };
            let placement = warm
                .as_ref()
                .and_then(|plan| self.place_warm(plan, &protected, store.as_deref_mut()));
            let mut admit = if let (Some(plan), Some(placement)) = (warm, placement) {
                let mut admit = ContAdmit::cold(1, plan.tokens, stepper.max_tokens.max(1));
                admit.place_bank = i32::try_from(placement.target + 1).unwrap_or(0);
                admit.n_cached = plan.cached;
                if placement.fork {
                    admit.fork_bank = i32::try_from(placement.source + 1).unwrap_or(0);
                }
                admit
            } else {
                let target = self.place_cold(&protected, store.as_deref_mut()).ok_or(
                    GenerateError::ContinuationHold {
                        retry_after: hold_retry.unwrap_or(1),
                    },
                )?;
                let mut admit = ContAdmit::cold(1, canonical_tokens, stepper.max_tokens.max(1));
                admit.place_bank = i32::try_from(target + 1).unwrap_or(0);
                admit
            };
            admit.eos = self.eos;
            admit.temperature = temperature;
            admit.top_k = top_k;
            admit.top_p = top_p;
            admit.min_p = min_p;
            admit.seed = parsed.seed;
            let mut adapter = WriteAdapter(&mut *out);
            let mut job = OneJob {
                admit: Some(admit),
                head: if head.is_empty() { None } else { Some(head) },
                stepper: &mut stepper,
                vocab: self.vocab,
                out: &mut adapter,
                admitted: false,
                bank: None,
                io_failed: false,
                host_abort: false,
                engine_eos: false,
                capture_done: bank_enabled,
                done_tokens: Vec::new(),
                n_cached: 0,
                n_computed: 0,
                t_arrive,
                t_admit: None,
                t_first: None,
                t_done: None,
                decode_ms: 0.0,
                decode_tokens: 0,
                decode_steps: 0,
            };
            let native_result = self.batch.continuous_generate(&mut job);
            let timings = {
                let completion = job.stepper.completion();
                match job.t_first {
                    Some(first) if completion > 0 => ReqTimings {
                        valid: true,
                        ttft_ms: first.duration_since(job.t_arrive).as_secs_f64() * 1e3,
                        prefill_ms: first
                            .duration_since(job.t_admit.unwrap_or(job.t_arrive))
                            .as_secs_f64()
                            * 1e3,
                        decode_ms: job.decode_ms,
                        prefill_tokens: job.n_computed,
                        prefill_cached: job.n_cached,
                        decode_tokens: job.decode_tokens,
                        decode_steps: job.decode_steps,
                    },
                    _ => ReqTimings::default(),
                }
            };
            let engine_eos = job.engine_eos && !job.host_abort;
            let n_cached = job.n_cached;
            let n_computed = job.n_computed;
            let io_ok = !job.io_failed;
            let done_called = job.t_done.is_some();
            let done_tokens = std::mem::take(&mut job.done_tokens);
            let admitted = job.admitted;
            let actual_bank = job.bank;
            drop(job);
            drop(adapter);
            if let Some(bank) = retired_bank(
                bank_enabled,
                admitted,
                done_called,
                actual_bank,
                self.batch.max_seq(),
            ) {
                self.retire(bank, &stepper.prompt, &done_tokens);
                if self.warm_checkpoint {
                    if let Some(store) = store.as_deref_mut() {
                        self.persist_bank(
                            store,
                            bank as usize,
                            KvReason::BankCheckpoint,
                            self.warm_pin_min,
                            true,
                        );
                    }
                }
            }
            if let Err(error) = native_result {
                return Err(GenerateError::Engine(error.to_string()));
            }
            if !admitted {
                /* Engine declined (oversized/budget): no bytes were sent,
                 * the caller falls back to the serial lane. */
                return Err(GenerateError::Unsupported(
                    "continuous admission rejected; serial fallback",
                ));
            }
            if !io_ok {
                return Err(GenerateError::Io);
            }
            let (tail, outcome) = stepper.finalize(engine_eos, n_cached, n_computed, timings, cors);
            if io_ok && !tail.is_empty() {
                out.write_all(&tail).map_err(|_| GenerateError::Io)?;
                let _ = out.flush();
            }
            Ok(outcome)
        }

        fn shutdown(&mut self, store: Option<&mut KvStore>) {
            if let Some(store) = store {
                self.shutdown_banks(store);
            }
        }
    }

    struct WriteAdapter<'a>(&'a mut dyn Write);

    impl Write for WriteAdapter<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.0.flush()
        }
    }
}

impl ContStepper {
    pub fn completion(&self) -> i32 {
        self.acc.completion
    }

    /// Host stop-token verdict (family EOT / eos / role starts), the same
    /// pre-eval check the serial `decode_pass` runs; the stop token's text
    /// is never fed.
    pub fn mark_stop(&mut self) {
        self.finish = "stop";
    }
}

#[cfg(test)]
mod bank_tests {
    use super::*;
    use crate::parse::{parse_chat_request, ParseEnv};
    use crate::route::{Api, ReqKind, ThinkMode};
    use ds4_kv::{Options, Reason, Store, EXT_BANK_REPLAY_V1};
    use std::fs;

    fn temp_store(tag: &str) -> (std::path::PathBuf, Store) {
        let dir =
            std::env::temp_dir().join(format!("ds4-server-bank-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::open(&dir, 16, false, Options::default()).unwrap();
        (dir, store)
    }

    fn warm(text: &str, last_use: u64) -> WarmBank {
        WarmBank {
            record: Some(WarmRecord {
                text: text.as_bytes().to_vec(),
                generation: 1,
            }),
            committed_tokens: 10,
            stored_tokens: 0,
            last_use,
        }
    }

    #[test]
    fn bank_scope_is_openai_chat_without_thinking_or_tools() {
        let mut env = ParseEnv::default();
        env.default_effort = ThinkMode::None;
        let mut parsed = parse_chat_request(
            &env,
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#,
        )
        .unwrap();
        assert!(bank_scope(&parsed));

        parsed.api = Api::Anthropic;
        assert!(!bank_scope(&parsed));
        parsed.api = Api::Openai;
        parsed.kind = ReqKind::Completion;
        assert!(!bank_scope(&parsed));
        parsed.kind = ReqKind::Chat;
        parsed.think_mode = ThinkMode::Low;
        assert!(!bank_scope(&parsed));
        parsed.think_mode = ThinkMode::None;
        parsed.has_tools = true;
        assert!(!bank_scope(&parsed));
        parsed.has_tools = false;
        parsed.has_tool_results = true;
        assert!(!bank_scope(&parsed));
        parsed.has_tool_results = false;
        parsed.live_call_ids.push("call_1".into());
        assert!(!bank_scope(&parsed));
    }

    #[test]
    fn warm_key_drops_the_uncommitted_last_sample() {
        assert_eq!(
            committed_key(b"prompt:", &[1, 2, 3], |token| vec![b'0' + token as u8]),
            b"prompt:12"
        );
        assert_eq!(
            committed_key(b"prompt:", &[], |_| unreachable!()),
            b"prompt:"
        );
    }

    #[test]
    fn warm_match_requires_a_strict_suffix_and_exact_snapshot_tokens() {
        let warm = WarmBank {
            record: Some(WarmRecord {
                text: b"shared".to_vec(),
                generation: 7,
            }),
            committed_tokens: 2,
            stored_tokens: 0,
            last_use: 0,
        };
        assert!(warm_admit_tokens(&warm, b"shared", &[10, 11], 7, 8, |_| vec![12]).is_none());
        assert!(warm_admit_tokens(&warm, b"other", &[10, 11], 7, 8, |_| vec![12]).is_none());
        assert!(
            warm_admit_tokens(&warm, b"shared suffix", &[10, 11], 8, 8, |_| vec![12]).is_none()
        );
        let (tokens, cached) =
            warm_admit_tokens(&warm, b"shared suffix", &[10, 11], 7, 8, |suffix| {
                assert_eq!(suffix, b" suffix");
                vec![20, 21]
            })
            .unwrap();
        assert_eq!(tokens, vec![10, 11, 20, 21]);
        assert_eq!(cached, 2);
        assert!(
            warm_admit_tokens(&warm, b"shared suffix", &[10, 11], 7, 3, |_| vec![20, 21]).is_none()
        );
    }

    #[test]
    fn multi_bank_match_uses_the_longest_strict_prefix() {
        let banks = vec![warm("shared", 3), warm("shared turn", 2), warm("other", 1)];
        assert_eq!(warm_match_pick(&banks, b"shared turn suffix"), Some(1));
        assert_eq!(warm_match_pick(&banks, b"shared turn"), Some(0));
        assert_eq!(warm_match_pick(&banks, b"unrelated"), None);
    }

    #[test]
    fn cold_victim_matches_c_no_value_superseded_and_depth_tiers() {
        let mut banks = vec![warm("trunk", 30), WarmBank::default(), warm("tenant", 10)];
        banks[0].committed_tokens = 70_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(1));

        banks[1] = warm("trunk child", 20);
        banks[1].committed_tokens = 70_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(0));

        banks[0] = warm("alpha", 5);
        banks[1] = warm("beta", 4);
        banks[2] = warm("gamma", 10);
        banks[0].committed_tokens = 70_000;
        banks[1].committed_tokens = 70_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(2));
        banks[2].committed_tokens = 90_000;
        assert_eq!(warm_victim_pick(&banks, &[], None, 65_536, true), Some(1));
    }

    #[test]
    fn fork_target_never_spends_a_plain_tenant_record() {
        let banks = vec![warm("source", 3), warm("tenant-a", 1), warm("tenant-b", 2)];
        assert_eq!(warm_victim_pick(&banks, &[], Some(0), 65_536, false), None);

        let mut spare = banks;
        spare[2] = WarmBank::default();
        assert_eq!(
            warm_victim_pick(&spare, &[], Some(0), 65_536, false),
            Some(2)
        );
    }

    #[test]
    fn warm_fork_preserves_the_trunk_only_when_a_safe_target_exists() {
        let mut with_spare = vec![warm("trunk", 3), warm("tenant", 1), WarmBank::default()];
        assert_eq!(
            warm_placement(&with_spare, &[], 0, 10, 65_536, true),
            Some(WarmPlacement {
                source: 0,
                target: 2,
                fork: true,
            })
        );

        let no_spare = vec![warm("trunk", 3), warm("tenant-a", 1), warm("tenant-b", 2)];
        assert_eq!(
            warm_placement(&no_spare, &[], 0, 10, 65_536, true),
            Some(WarmPlacement {
                source: 0,
                target: 0,
                fork: false,
            })
        );
        with_spare[0].committed_tokens = 70_000;
        assert!(
            !warm_placement(&with_spare, &[], 0, 70_000, 65_536, true)
                .unwrap()
                .fork
        );
    }

    #[test]
    fn protected_trunk_forks_safely_or_refuses_in_place_extension() {
        let with_spare = vec![warm("trunk", 3), WarmBank::default()];
        assert!(
            warm_placement(&with_spare, &[true, false], 0, 10, 65_536, true)
                .is_some_and(|placement| placement.fork && placement.target == 1)
        );

        let no_spare = vec![warm("trunk", 3), warm("tenant", 2)];
        assert_eq!(
            warm_placement(&no_spare, &[true, false], 0, 10, 65_536, true),
            None
        );
    }

    #[test]
    fn victim_selection_skips_a_protected_bank_at_every_tier() {
        let mut banks = vec![WarmBank::default(), warm("shallow", 1), warm("deep", 2)];
        banks[2].committed_tokens = 70_000;
        assert_eq!(
            warm_victim_pick(&banks, &[true, false, false], None, 65_536, true),
            Some(1)
        );

        banks[0] = warm("trunk", 1);
        banks[1] = warm("trunk child", 2);
        assert_eq!(
            warm_victim_pick(&banks, &[true, false, false], None, 65_536, false),
            None
        );

        banks[0] = warm("shallow", 1);
        banks[1].committed_tokens = 70_000;
        assert_eq!(
            warm_victim_pick(&banks, &[true, false, false], None, 65_536, true),
            Some(1)
        );

        let all_protected = [true, true, true];
        assert_eq!(
            warm_victim_pick(&banks, &all_protected, None, 65_536, true),
            None
        );
    }

    #[test]
    fn disk_restore_replaces_only_empty_stale_or_shallow_live_banks() {
        let empty = WarmBank::default();
        assert!(disk_restore_allowed(&empty, Some((7, 70_000)), 1, 65_536));

        let live = WarmBank {
            record: Some(WarmRecord {
                text: b"live".to_vec(),
                generation: 7,
            }),
            committed_tokens: 70_000,
            stored_tokens: 0,
            last_use: 0,
        };
        assert!(disk_restore_allowed(&live, Some((8, 70_000)), 1, 65_536));
        assert!(disk_restore_allowed(
            &live,
            Some((7, 32_000)),
            70_000,
            65_536,
        ));
        assert!(!disk_restore_allowed(
            &live,
            Some((7, 70_000)),
            70_000,
            65_536,
        ));
        assert!(!disk_restore_allowed(
            &live,
            Some((7, 32_000)),
            32_000,
            65_536,
        ));
    }

    #[test]
    fn disk_restore_may_replace_a_superseded_deep_record() {
        let mut banks = vec![warm("trunk", 1), warm("trunk child", 2)];
        banks[0].committed_tokens = 70_000;
        banks[1].committed_tokens = 70_000;
        assert!(disk_restore_target_allowed(&banks, 0, 10, 65_536));
        assert!(!disk_restore_target_allowed(&banks, 1, 10, 65_536));
    }

    #[test]
    fn bank_retires_only_after_native_done() {
        assert!(bank_retire_allowed(true, true, true));
        assert!(!bank_retire_allowed(false, true, true));
        assert!(!bank_retire_allowed(true, false, true));
        assert!(!bank_retire_allowed(true, true, false));
    }

    #[test]
    fn bank_retirement_uses_the_native_reported_bank() {
        assert_eq!(retired_bank(true, true, true, Some(2), 3), Some(2));
        assert_eq!(retired_bank(true, true, true, None, 3), None);
        assert_eq!(retired_bank(true, true, true, Some(3), 3), None);
    }

    #[test]
    fn bank_save_stages_before_reuse_and_advances_marker_only_on_success() {
        let (dir, mut store) = temp_store("save-order");
        let mut warm = WarmBank {
            record: Some(WarmRecord {
                text: b"shared prefix".to_vec(),
                generation: 7,
            }),
            committed_tokens: 3,
            stored_tokens: 1,
            last_use: 0,
        };
        assert!(!save_bank_record(
            &mut store,
            &mut warm,
            (0, 2, 8192),
            3,
            8,
            Reason::BankShutdown,
            |_| panic!("stale records must not stage payloads"),
        )
        .unwrap());
        assert!(store.entries().is_empty());
        assert_eq!(warm.stored_tokens, 1);
        assert!(save_bank_record(
            &mut store,
            &mut warm,
            (0, 2, 8192),
            3,
            7,
            Reason::BankShutdown,
            |path| fs::write(path, b"opaque").map_err(|e| GenerateError::Engine(e.to_string())),
        )
        .unwrap());
        assert_eq!(warm.stored_tokens, 3);
        let path = store.entries()[0].path.clone();
        let record = store.read(&path).unwrap();
        assert_eq!(record.header.reason, Reason::BankShutdown);
        assert_eq!(record.header.ext_flags, EXT_BANK_REPLAY_V1);
        assert_eq!(record.header.tokens, 3);
        assert_eq!(record.header.ctx_size, 8192);
        assert_eq!(record.text, b"shared prefix");
        assert_eq!(record.payload, b"opaque");

        warm.stored_tokens = 1;
        let error = save_bank_record(
            &mut store,
            &mut warm,
            (0, 2, 8192),
            3,
            7,
            Reason::BankShutdown,
            |_| Err(GenerateError::Engine("stage failed".into())),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "stage failed");
        assert_eq!(warm.stored_tokens, 1);
        assert_eq!(store.read(&path).unwrap().payload, b"opaque");
        let _ = fs::remove_dir_all(dir);
    }
}
