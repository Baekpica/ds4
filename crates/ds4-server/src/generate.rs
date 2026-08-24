//! Serial decode driver: render (including tool-schema / invoke reconstruct)
//! → host Vocab tokenize (FFI fallback) → host SessionLedger pos/generation
//! → native prefill/eval → SemAccum + generated-message parse → stream
//! projectors.
//! Incremental live DSML tool projection, required-prefix / structural
//! greedy sampling, and corrective retry (`decode_again` / model-visible
//! tool error) are host-owned. Continuation publish/hold/resolve is
//! host-owned (`cont`).

use std::io::Write;
#[cfg(any(feature = "native", test))]
use std::path::Path;
use std::time::Instant;
#[cfg(any(feature = "native", test))]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(any(feature = "native", test))]
use ds4_kv::{Header as KvHeader, Reason as KvReason, Store as KvStore};

use crate::dsml::{SampleOverride, SamplePolicy};
use crate::parse::{ParsedRequest, ToolChoice};
use crate::retry::{
    build_invalid_tool_error_suffix, parse_failure_should_retry, terminal_finish,
    truncation_outcome, TruncationOutcome,
};
use crate::render::{
    render_chat_choice, syntax_for_model_id, RenderError, ModelSyntax,
};
use crate::parse::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
use crate::route::{decode_budget, think_mode_enabled, Api, ReqKind};
use crate::stream::{
    anthropic_final_response, anthropic_sse_finish_live, anthropic_sse_start_live,
    anthropic_sse_stream_update, final_response, openai_sse_finish_live, openai_sse_stream_update,
    openai_stream_start, responses_final_response, responses_sse_created, responses_sse_finish_live,
    responses_sse_stream_update, responses_stream_init, sse_chunk, sse_done, sse_headers,
    think_end, think_start, AnthropicStream, ChatFormat, OpenaiStream, ReqTimings, ResponsesStream,
    StreamReq, Writer,
};
use crate::tools::{
    assign_tool_ids, parse_generated_for_response, SemAccum,
};

#[derive(Debug)]
pub enum GenerateError {
    Unsupported(&'static str),
    Engine(String),
    Io,
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Unsupported(s) => f.write_str(s),
            GenerateError::Engine(s) => write!(f, "{s}"),
            GenerateError::Io => f.write_str("client stream write failed"),
        }
    }
}

impl From<RenderError> for GenerateError {
    fn from(e: RenderError) -> Self {
        GenerateError::Unsupported(e.0)
    }
}

pub trait DecodeIo {
    fn model_id(&self) -> i32;
    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError>;
    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError>;
    fn tokenizes_control_literals(&self) -> bool {
        true
    }
    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError>;
    fn token_is_stop(&self, token: i32) -> bool;
    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError>;
    fn sync_prompt(
        &mut self,
        _prompt: &[u8],
        tokens: &[i32],
        _disk_eligible: bool,
    ) -> Result<i32, GenerateError> {
        self.sync(tokens)?;
        Ok(0)
    }
    fn eval(&mut self, token: i32) -> Result<(), GenerateError>;
    fn sample(&mut self, temperature: f32, top_k: i32, top_p: f32, min_p: f32, rng: &mut u64)
        -> i32;
    fn pos(&self) -> i32;
    fn ctx(&self) -> i32;
    fn generation(&self) -> u64;
    fn session_tokens(&self) -> Vec<i32> {
        Vec::new()
    }
}

#[cfg(any(feature = "native", test))]
#[derive(Clone, Copy)]
struct DiskSyncPolicy {
    save_current: bool,
    load: bool,
}

#[cfg(any(feature = "native", test))]
trait SerialKvIo {
    fn ctx(&self) -> i32;
    fn live_tokens(&self) -> Vec<i32>;
    fn render_tokens(&self, tokens: &[i32]) -> Result<Vec<u8>, GenerateError>;
    fn tokenize_suffix(&mut self, suffix: &[u8]) -> Result<Vec<i32>, GenerateError>;
    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError>;
    fn save_payload(&mut self, path: &Path) -> Result<(), GenerateError>;
    fn load_payload_range(
        &mut self,
        path: &Path,
        offset: u64,
        length: u64,
    ) -> Result<(), GenerateError>;
    fn invalidate(&mut self);
}

#[cfg(any(feature = "native", test))]
fn kv_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(any(feature = "native", test))]
fn kv_identity(model_id: i32, quant_bits: i32, ctx: i32) -> Option<(u8, u8, u32)> {
    let model_id = u8::try_from(model_id).ok()?;
    let quant_bits = match quant_bits {
        2 | 4 => quant_bits as u8,
        _ => return None,
    };
    let ctx = u32::try_from(ctx).ok().filter(|ctx| *ctx > 0)?;
    Some((model_id, quant_bits, ctx))
}

#[cfg(any(feature = "native", test))]
fn kv_header(model_id: u8, quant_bits: u8, ctx: u32, tokens: u32) -> KvHeader {
    let now = kv_now();
    KvHeader {
        quant_bits,
        reason: KvReason::Evict,
        ext_flags: 0,
        model_id,
        tokens,
        hits: 0,
        ctx_size: ctx,
        created_at: now,
        last_used: now,
        payload_bytes: 0,
        text_bytes: 0,
    }
}

#[cfg(any(feature = "native", test))]
fn try_store_current(
    io: &mut impl SerialKvIo,
    store: &mut KvStore,
    model_id: u8,
    quant_bits: u8,
    ctx: u32,
) -> Result<(), GenerateError> {
    let live = io.live_tokens();
    if live.len() < store.opt.min_tokens.max(0) as usize {
        return Ok(());
    }
    let tokens = u32::try_from(live.len())
        .map_err(|_| GenerateError::Engine("KVC token count exceeds u32".into()))?;
    let text = io.render_tokens(&live)?;
    let header = kv_header(model_id, quant_bits, ctx, tokens);
    if store
        .reuse_compatible(header.clone(), &text, b"")
        .map_err(|error| GenerateError::Engine(error.to_string()))?
        .is_some()
    {
        return Ok(());
    }
    let payload = store
        .payload_temp()
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    io.save_payload(payload.path())?;
    store
        .write_payload_file(header, &text, payload.path(), b"")
        .map_err(|error| GenerateError::Engine(error.to_string()))?;
    Ok(())
}

#[cfg(any(feature = "native", test))]
fn cold_sync(io: &mut impl SerialKvIo, tokens: &[i32]) -> Result<i32, GenerateError> {
    io.sync(tokens)?;
    Ok(0)
}

#[cfg(any(feature = "native", test))]
fn discard_loaded(store: &mut KvStore, io: &mut impl SerialKvIo, path: &Path) {
    store.continued_last_store_tokens = 0;
    let _ = store.discard(path);
    io.invalidate();
}

#[cfg(any(feature = "native", test))]
fn disk_sync_prompt(
    io: &mut impl SerialKvIo,
    store: &mut KvStore,
    model_id: i32,
    quant_bits: i32,
    prompt: &[u8],
    canonical_tokens: &[i32],
    policy: DiskSyncPolicy,
) -> Result<i32, GenerateError> {
    let live = io.live_tokens();
    if !live.is_empty() && canonical_tokens.starts_with(&live) {
        io.sync(canonical_tokens)?;
        return Ok(live.len() as i32);
    }
    if !live.is_empty() {
        let rendered = io.render_tokens(&live)?;
        if prompt.starts_with(&rendered) {
            let cached = live.len() as i32;
            let mut effective = live;
            effective.extend(io.tokenize_suffix(&prompt[rendered.len()..])?);
            io.sync(&effective)?;
            return Ok(cached);
        }
    }

    store.continued_last_store_tokens = 0;
    let Some((model_id, quant_bits, ctx)) = kv_identity(model_id, quant_bits, io.ctx()) else {
        return cold_sync(io, canonical_tokens);
    };
    if policy.save_current {
        let _ = try_store_current(io, store, model_id, quant_bits, ctx);
    }
    if !policy.load {
        return cold_sync(io, canonical_tokens);
    }
    let candidate = match store.text_prefix_candidate(prompt, model_id, quant_bits, ctx) {
        Ok(candidate) => candidate,
        Err(_) => return cold_sync(io, canonical_tokens),
    };
    let Some((path, envelope)) = candidate else {
        return cold_sync(io, canonical_tokens);
    };
    if envelope.header.ext_flags != 0 || envelope.trailer_bytes != 0 {
        return cold_sync(io, canonical_tokens);
    }
    if io
        .load_payload_range(
            &path,
            envelope.payload_offset,
            envelope.header.payload_bytes,
        )
        .is_err()
    {
        io.invalidate();
        return cold_sync(io, canonical_tokens);
    }
    let loaded = io.live_tokens();
    if loaded.len() != envelope.header.tokens as usize {
        io.invalidate();
        let _ = store.discard(&path);
        return cold_sync(io, canonical_tokens);
    }
    let cached = loaded.len() as i32;
    store.continued_last_store_tokens = cached;
    let _ = store.touch_hit(&path);
    let mut effective = loaded;
    let suffix = &prompt[envelope.text.len()..];
    let suffix_tokens = match io.tokenize_suffix(suffix) {
        Ok(tokens) => tokens,
        Err(error) => {
            discard_loaded(store, io, &path);
            return Err(error);
        }
    };
    effective.extend(suffix_tokens);
    if let Err(error) = io.sync(&effective) {
        discard_loaded(store, io, &path);
        return Err(error);
    }
    Ok(cached)
}

#[derive(Debug, Clone, Default)]
pub struct GenerateOutcome {
    pub tool_ids: Vec<String>,
    pub generation: u64,
    pub frontier: i32,
    pub finish: String,
}

pub fn generation_blocked(_parsed: &ParsedRequest, _model_id: i32) -> Option<&'static str> {
    None
}

pub fn chat_format_for_syntax(syntax: ModelSyntax) -> ChatFormat {
    match syntax {
        ModelSyntax::SolarOpen2 => ChatFormat::SolarOpen2,
        ModelSyntax::Exaone => ChatFormat::Exaone,
        ModelSyntax::DeepSeek | ModelSyntax::Motif3 | ModelSyntax::Dots3 => ChatFormat::DeepSeek,
    }
}

pub fn stream_req_from_parsed(parsed: &ParsedRequest, model_id: i32) -> StreamReq {
    StreamReq {
        kind: parsed.kind,
        api: parsed.api,
        model: parsed.model.clone(),
        think_mode: parsed.think_mode,
        has_tools: parsed.has_tools,
        stream: parsed.stream,
        stream_include_usage: parsed.stream_include_usage,
        reasoning_summary_emit: parsed.reasoning_summary_emit,
        chat_format: chat_format_for_syntax(syntax_for_model_id(model_id)),
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        timings: ReqTimings::default(),
        tool_orders: parsed.tool_orders.clone(),
    }
}

pub fn render_prompt(parsed: &ParsedRequest, model_id: i32) -> Result<Vec<u8>, GenerateError> {
    match parsed.kind {
        ReqKind::Completion => Ok(parsed.prompt_text.clone().unwrap_or_default().into_bytes()),
        ReqKind::Chat => Ok(render_chat_choice(
            syntax_for_model_id(model_id),
            &parsed.messages,
            &parsed.tool_schemas,
            &parsed.tool_orders,
            parsed.think_mode,
            parsed.tool_choice,
        )?),
    }
}

fn ordinary_disk_cache_eligible(parsed: &ParsedRequest) -> bool {
    parsed.api == Api::Openai
        && !think_mode_enabled(parsed.think_mode)
        && !parsed.has_tools
        && !parsed.has_tool_results
        && parsed.live_call_ids.is_empty()
}

fn prepare_required_prefixes(
    engine: &dyn DecodeIo,
    parsed: &mut ParsedRequest,
    format: ChatFormat,
) -> Result<(), GenerateError> {
    if !engine.tokenizes_control_literals() {
        return Ok(());
    }
    if parsed.tool_choice != ToolChoice::Required && !parsed.has_tool_results {
        return Ok(());
    }
    if parsed.required_think_end_prefix.is_empty() {
        let toks = engine.tokenize_rendered_chat(think_end(format).as_bytes())?;
        if toks.is_empty() {
            return Err(GenerateError::Engine(
                "failed to tokenize thinking control prefix".into(),
            ));
        }
        parsed.required_think_end_prefix = toks;
    }
    if parsed.tool_choice == ToolChoice::Required && parsed.required_tool_prefix.is_empty() {
        let marker = match format {
            ChatFormat::SolarOpen2 => crate::render::SOLAR_TOOL_CALLS,
            ChatFormat::Exaone => "<tool_call>",
            ChatFormat::DeepSeek => crate::tools::DSML_TOOL_CALLS_START,
        };
        let toks = engine.tokenize_rendered_chat(marker.as_bytes())?;
        if toks.is_empty() {
            return Err(GenerateError::Engine(
                "failed to tokenize required tool control prefix".into(),
            ));
        }
        parsed.required_tool_prefix = toks;
    }
    Ok(())
}

fn find_substr(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

pub fn stop_list_find_from(stops: &[String], text: &[u8], from: usize) -> Option<(usize, usize)> {
    if stops.is_empty() || from > text.len() {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        let needle = s.as_bytes();
        if from + needle.len() > text.len() {
            continue;
        }
        if let Some(rel) = find_substr(&text[from..], needle) {
            let pos = from + rel;
            if best.map(|(p, _)| pos < p).unwrap_or(true) {
                best = Some((pos, needle.len()));
            }
        }
    }
    best
}

pub fn stop_list_max_len(stops: &[String]) -> usize {
    stops.iter().map(|s| s.len()).max().unwrap_or(0)
}

pub fn stop_list_stream_safe_len(stops: &[String], text_len: usize) -> usize {
    let max = stop_list_max_len(stops);
    if max <= 1 || text_len <= max - 1 {
        return if max <= 1 { text_len } else { 0 };
    }
    text_len - (max - 1)
}

#[allow(dead_code)]
fn split_think(raw: &[u8], think: bool, fmt: ChatFormat) -> (Vec<u8>, Vec<u8>) {
    if !think {
        return (raw.to_vec(), Vec::new());
    }
    let start = think_start(fmt).as_bytes();
    let end = think_end(fmt).as_bytes();
    let body = if raw.starts_with(start) {
        &raw[start.len()..]
    } else {
        raw
    };
    if let Some(i) = find_substr(body, end) {
        (body[i + end.len()..].to_vec(), body[..i].to_vec())
    } else {
        (Vec::new(), body.to_vec())
    }
}

fn responses_ids(job_id: &str) -> (String, String, String) {
    let mut h = 2_166_136_261u32;
    for b in job_id.as_bytes() {
        h ^= u32::from(*b);
        h = h.wrapping_mul(16_777_619);
    }
    let hex = format!("{h:08x}{h:08x}{h:08x}");
    (
        format!("resp_{hex}"),
        format!("rs_{hex}"),
        format!("msg_{hex}"),
    )
}

fn flush(w: &mut Writer, out: &mut impl Write) -> Result<(), GenerateError> {
    if w.out.is_empty() {
        return Ok(());
    }
    out.write_all(&w.out).map_err(|_| GenerateError::Io)?;
    w.out.clear();
    Ok(())
}

fn append_recovery_suffix(
    engine: &mut dyn DecodeIo,
    suffix: &[u8],
) -> Result<i32, GenerateError> {
    if suffix.is_empty() {
        return Ok(0);
    }
    let before = engine.pos();
    let mut target = engine.session_tokens();
    let extra = engine.tokenize_rendered_chat(suffix)?;
    target.extend(extra);
    engine.sync(&target)?;
    let delta = engine.pos() - before;
    Ok(if delta > 0 { delta } else { 0 })
}

fn decode_pass(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    req: &StreamReq,
    job_id: &str,
    acc: &mut SemAccum,
    finish: &mut &'static str,
    max_tokens: i32,
    rng: &mut u64,
    w: &mut Writer,
    out: &mut impl Write,
    mut oa: Option<&mut OpenaiStream>,
    mut anth: Option<&mut AnthropicStream>,
    mut resp: Option<&mut ResponsesStream>,
    first_tok: &mut Option<Instant>,
    decode_steps: &mut i32,
) -> Result<(), GenerateError> {
    while acc.completion < max_tokens && engine.pos() < engine.ctx() {
        out.flush().map_err(|_| GenerateError::Io)?;
        let mut temperature = parsed.temperature;
        let mut top_k = parsed.top_k;
        let mut top_p = parsed.top_p;
        let mut min_p = parsed.min_p;
        if think_mode_enabled(parsed.think_mode) {
            temperature = DEFAULT_TEMPERATURE;
            top_k = 0;
            top_p = DEFAULT_TOP_P;
            min_p = DEFAULT_MIN_P;
        }
        let policy = SamplePolicy {
            tool_choice: parsed.tool_choice,
            has_tool_results: parsed.has_tool_results,
            think_mode: parsed.think_mode,
            max_tokens: parsed.max_tokens,
            required_tool_prefix: &parsed.required_tool_prefix,
            required_think_end_prefix: &parsed.required_think_end_prefix,
        };
        let ov = acc.sampling_override(&policy);
        if matches!(ov, SampleOverride::Greedy) {
            temperature = 0.0;
        }
        let token = if let SampleOverride::Token(t) = ov {
            t
        } else {
            engine.sample(temperature, top_k, top_p, min_p, rng)
        };
        if token < 0 || engine.token_is_stop(token) {
            *finish = "stop";
            break;
        }
        engine.eval(token)?;
        *decode_steps += 1;
        if first_tok.is_none() {
            *first_tok = Some(Instant::now());
        }
        let piece = engine.token_text(token)?;
        let feed = acc.feed(&piece, &parsed.stops);

        if req.stream {
            let view = &acc.text[..feed.emit_limit.min(acc.text.len())];
            match req.api {
                Api::Openai if req.kind == ReqKind::Completion => {
                    if let Some(delta) = last_delta(&acc.text, feed.emit_limit, piece.len()) {
                        sse_chunk(w, req, job_id, Some(delta), None);
                    }
                }
                Api::Openai => {
                    if let Some(st) = oa.as_mut() {
                        openai_sse_stream_update(w, req, job_id, st, view, false);
                    }
                }
                Api::Anthropic => {
                    if let Some(st) = anth.as_mut() {
                        anthropic_sse_stream_update(w, req, job_id, st, view, false);
                    }
                }
                Api::Responses => {
                    if let Some(st) = resp.as_mut() {
                        responses_sse_stream_update(w, req, st, view, false);
                    }
                }
            }
            flush(w, out)?;
        }

        if feed.hit_stop {
            *finish = "stop";
            break;
        }
        if acc.track_tools && acc.saw_tool_end && req.chat_format == ChatFormat::DeepSeek {
            *finish = "tool_calls";
            break;
        }
    }
    Ok(())
}

pub fn generate_and_write(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    job_id: &str,
    created: i64,
    cors: bool,
    default_tokens: i32,
    out: &mut impl Write,
) -> Result<GenerateOutcome, GenerateError> {
    generate_and_write_at(
        engine,
        parsed,
        job_id,
        created,
        cors,
        default_tokens,
        Instant::now(),
        out,
    )
}

/// Queue-aware serial entry point. `t_arrive` is captured by the HTTP owner;
/// callers outside the queued server should use [`generate_and_write`].
pub fn generate_and_write_at(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    job_id: &str,
    created: i64,
    cors: bool,
    default_tokens: i32,
    t_arrive: Instant,
    out: &mut impl Write,
) -> Result<GenerateOutcome, GenerateError> {
    let (outcome, terminal) = generate_terminal_at(
        engine,
        parsed,
        job_id,
        created,
        cors,
        default_tokens,
        t_arrive,
        out,
    )?;
    out.write_all(&terminal).map_err(|_| GenerateError::Io)?;
    Ok(outcome)
}

/// Runs serial generation while withholding only the final wire terminal.
/// Streaming headers/deltas still flow through `out` as they are produced.
pub(crate) fn generate_terminal_at(
    engine: &mut dyn DecodeIo,
    parsed: &ParsedRequest,
    job_id: &str,
    created: i64,
    cors: bool,
    default_tokens: i32,
    t_arrive: Instant,
    out: &mut impl Write,
) -> Result<(GenerateOutcome, Vec<u8>), GenerateError> {
    if let Some(msg) = generation_blocked(parsed, engine.model_id()) {
        return Err(GenerateError::Unsupported(msg));
    }

    let mut parsed = parsed.clone();
    prepare_required_prefixes(engine, &mut parsed, chat_format_for_syntax(syntax_for_model_id(engine.model_id())))?;

    let prompt = render_prompt(&parsed, engine.model_id())?;
    let tokens = match parsed.kind {
        ReqKind::Completion => {
            let text = std::str::from_utf8(&prompt).unwrap_or("");
            engine.tokenize_text(text)?
        }
        ReqKind::Chat => engine.tokenize_rendered_chat(&prompt)?,
    };
    let t_prefill = Instant::now();
    let cached = engine.sync_prompt(
        &prompt,
        &tokens,
        ordinary_disk_cache_eligible(&parsed),
    )?;
    let decode_t0 = Instant::now();
    let mut first_tok = None;
    let mut decode_steps = 0i32;

    let prompt_n = engine.pos();
    let mut rng = parsed.seed;
    let mut req = stream_req_from_parsed(&parsed, engine.model_id());
    req.cache_read_tokens = cached.clamp(0, prompt_n);
    req.cache_write_tokens = prompt_n - req.cache_read_tokens;
    let syntax = syntax_for_model_id(engine.model_id());
    let mut acc;
    let mut finish;
    let mut recovery_attempted = false;

    let mut w = Writer::new(created);
    let mut oa = if req.stream && req.api == Api::Openai && req.kind == ReqKind::Chat {
        Some(openai_stream_start(&req))
    } else {
        None
    };
    let mut anth = if req.stream && req.api == Api::Anthropic {
        Some(anthropic_sse_start_live(&mut w, &req, job_id, prompt_n))
    } else {
        None
    };
    let mut resp = if req.stream && req.api == Api::Responses {
        let (rid, rsid, mid) = responses_ids(job_id);
        let mut st = responses_stream_init(&req, &rid, &rsid, &mid);
        responses_sse_created(&mut w, &req, &mut st, created);
        Some(st)
    } else {
        None
    };

    if req.stream {
        if req.api == Api::Openai {
            w.out.extend_from_slice(&sse_headers(cors));
            if req.kind == ReqKind::Chat {
                sse_chunk(&mut w, &req, job_id, None, None);
            }
        } else {
            let mut hdr = sse_headers(cors);
            hdr.extend_from_slice(&w.out);
            w.out = hdr;
        }
        flush(&mut w, out)?;
    }

    let mut parsed_gen;
    loop {
        let mut max_tokens = decode_budget(parsed.max_tokens_set, parsed.max_tokens, default_tokens);
        let room = engine.ctx() - engine.pos();
        if room >= 0 && max_tokens > room {
            max_tokens = room;
        }
        acc = SemAccum::init(
            parsed.kind == ReqKind::Chat,
            parsed.has_tools,
            think_mode_enabled(parsed.think_mode),
            req.chat_format,
            &prompt,
        );
        finish = "length";

        decode_pass(
            engine,
            &parsed,
            &req,
            job_id,
            &mut acc,
            &mut finish,
            max_tokens,
            &mut rng,
            &mut w,
            out,
            oa.as_mut(),
            anth.as_mut(),
            resp.as_mut(),
            &mut first_tok,
            &mut decode_steps,
        )?;

        finish = terminal_finish(acc.thinking_inside(), finish);
        match truncation_outcome(
            syntax,
            req.chat_format,
            parsed.kind == ReqKind::Chat,
            parsed.has_tools,
            acc.saw_tool_start,
            acc.saw_tool_end,
            finish,
            parsed.stream,
            recovery_attempted,
            &acc.text,
            &parsed.tool_orders,
        ) {
            TruncationOutcome::Repair(text) => {
                acc.text = text;
                acc.saw_tool_end = true;
            }
            TruncationOutcome::RetryUnterminated => {
                if append_recovery_suffix(
                    engine,
                    &build_invalid_tool_error_suffix(
                        req.chat_format,
                        parsed.think_mode,
                        acc.thinking_inside(),
                        &prompt,
                        "unterminated tool call",
                    ),
                )
                .is_ok()
                {
                    recovery_attempted = true;
                    continue;
                }
                finish = "error";
            }
            TruncationOutcome::ErrorUnterminated => {
                finish = "error";
            }
            TruncationOutcome::None => {}
        }

        parsed_gen = if parsed.kind == ReqKind::Chat {
            let (pg, recovered_finish) = parse_generated_for_response(
                syntax,
                &acc.text,
                parsed.has_tools,
                acc.saw_tool_start,
                think_mode_enabled(parsed.think_mode),
                req.chat_format,
                &parsed.tool_orders,
                finish,
            );
            finish = recovered_finish;
            pg
        } else {
            crate::tools::ParsedGenerated {
                content: acc.text.clone(),
                ok: true,
                ..Default::default()
            }
        };
        if !parsed_gen.ok
            && parse_failure_should_retry(
                syntax,
                parsed.stream,
                recovery_attempted,
                finish,
                parsed_gen.recovered,
                parsed.has_tools,
                acc.saw_tool_start,
            )
        {
            if append_recovery_suffix(
                engine,
                &build_invalid_tool_error_suffix(
                    req.chat_format,
                    parsed.think_mode,
                    acc.thinking_inside(),
                    &prompt,
                    "invalid tool call",
                ),
            )
            .is_ok()
            {
                recovery_attempted = true;
                continue;
            }
            finish = "error";
        }
        break;
    }

    let completion = acc.completion;
    if completion > 0 {
        if let Some(t_first) = first_tok {
            req.timings = ReqTimings {
                valid: true,
                ttft_ms: t_first.duration_since(t_arrive).as_secs_f64() * 1e3,
                prefill_ms: decode_t0.duration_since(t_prefill).as_secs_f64() * 1e3,
                decode_ms: Instant::now().duration_since(t_first).as_secs_f64() * 1e3,
                prefill_tokens: prompt_n,
                prefill_cached: req.cache_read_tokens,
                decode_tokens: completion,
                decode_steps,
            };
        }
    }
    if !parsed_gen.calls.is_empty() {
        if let Some(st) = oa.as_ref() {
            st.tool.apply_ids(&mut parsed_gen.calls);
        }
        if let Some(st) = anth.as_ref() {
            st.tool.apply_ids(&mut parsed_gen.calls);
        }
        assign_tool_ids(&mut parsed_gen.calls, job_id);
        finish = "tool_calls";
    }
    let matched_stop = acc.matched_stop.clone();
    let terminal = if req.stream {
        match req.api {
            Api::Openai if req.kind == ReqKind::Completion => {
                sse_chunk(&mut w, &req, job_id, None, Some(finish));
                sse_done(&mut w, &req, job_id, prompt_n, completion);
            }
            Api::Openai => {
                if let Some(st) = oa.as_mut() {
                    openai_sse_finish_live(
                        &mut w,
                        &req,
                        job_id,
                        st,
                        &acc.text,
                        finish,
                        prompt_n,
                        completion,
                        &parsed_gen.calls,
                    );
                }
            }
            Api::Anthropic => {
                if let Some(st) = anth.as_mut() {
                    anthropic_sse_finish_live(
                        &mut w,
                        &req,
                        job_id,
                        st,
                        &acc.text,
                        finish,
                        matched_stop.as_deref(),
                        completion,
                        &parsed_gen.calls,
                    );
                }
            }
            Api::Responses => {
                if let Some(st) = resp.as_mut() {
                    responses_sse_finish_live(
                        &mut w,
                        &req,
                        st,
                        &acc.text,
                        finish,
                        prompt_n,
                        completion,
                        0,
                        created,
                        &parsed_gen.calls,
                    );
                }
            }
        }
        std::mem::take(&mut w.out)
    } else {
        let bytes = match req.api {
            Api::Anthropic => anthropic_final_response(
                &req,
                job_id,
                &parsed_gen.content,
                Some(&parsed_gen.reasoning),
                finish,
                matched_stop.as_deref(),
                prompt_n,
                completion,
                cors,
                &parsed_gen.calls,
            ),
            Api::Responses => {
                let (rid, rsid, mid) = responses_ids(job_id);
                responses_final_response(
                    &req,
                    &parsed_gen.content,
                    Some(&parsed_gen.reasoning),
                    finish,
                    prompt_n,
                    completion,
                    0,
                    created,
                    cors,
                    &rid,
                    &rsid,
                    &mid,
                    &parsed_gen.calls,
                )
            }
            Api::Openai => final_response(
                &req,
                job_id,
                &parsed_gen.content,
                Some(&parsed_gen.reasoning),
                finish,
                prompt_n,
                completion,
                created,
                cors,
                &parsed_gen.calls,
            ),
        };
        bytes
    };
    let outcome = GenerateOutcome {
        tool_ids: parsed_gen
            .calls
            .iter()
            .map(|c| c.id.clone())
            .filter(|id| !id.is_empty())
            .collect(),
        generation: engine.generation(),
        frontier: engine.pos(),
        finish: finish.to_string(),
    };
    Ok((outcome, terminal))
}

fn last_delta(raw: &[u8], emit_limit: usize, piece_len: usize) -> Option<&[u8]> {
    if emit_limit == 0 {
        return None;
    }
    let start = raw.len().saturating_sub(piece_len);
    if start >= emit_limit {
        return None;
    }
    Some(&raw[start..emit_limit])
}

/// Tape engine for tests. Does not open a GGUF.
pub struct ScriptedDecode {
    pub model_id: i32,
    pub prompt_tokens: Vec<i32>,
    pub steps: Vec<ScriptedStep>,
    pub idx: usize,
    pub pos: i32,
    pub ctx: i32,
    pub generation: u64,
    pub live: Vec<i32>,
    pub suffix_tokens: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct ScriptedStep {
    pub token: i32,
    pub piece: Vec<u8>,
    pub stop: bool,
}

impl ScriptedDecode {
    pub fn from_pieces(pieces: &[&[u8]]) -> Self {
        let steps = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| ScriptedStep {
                token: (i as i32) + 1,
                piece: p.to_vec(),
                stop: false,
            })
            .chain(std::iter::once(ScriptedStep {
                token: 99,
                piece: Vec::new(),
                stop: true,
            }))
            .collect();
        Self {
            model_id: 0,
            prompt_tokens: vec![1],
            steps,
            idx: 0,
            pos: 0,
            ctx: 8192,
            generation: 1,
            live: Vec::new(),
            suffix_tokens: Vec::new(),
        }
    }
}

impl DecodeIo for ScriptedDecode {
    fn model_id(&self) -> i32 {
        self.model_id
    }

    fn tokenize_text(&self, _text: &str) -> Result<Vec<i32>, GenerateError> {
        Ok(self.prompt_tokens.clone())
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        if find_substr(text, b"Tool error:").is_some() {
            if self.suffix_tokens.is_empty() {
                Ok(vec![7])
            } else {
                Ok(self.suffix_tokens.clone())
            }
        } else {
            Ok(self.prompt_tokens.clone())
        }
    }

    fn tokenizes_control_literals(&self) -> bool {
        false
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        Ok(self
            .steps
            .iter()
            .find(|s| s.token == token)
            .map(|s| s.piece.clone())
            .unwrap_or_default())
    }

    fn token_is_stop(&self, token: i32) -> bool {
        self.steps.iter().any(|s| s.token == token && s.stop)
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        self.live = tokens.to_vec();
        self.pos = tokens.len() as i32;
        Ok(())
    }

    fn eval(&mut self, _token: i32) -> Result<(), GenerateError> {
        self.pos += 1;
        Ok(())
    }

    fn sample(
        &mut self,
        _temperature: f32,
        _top_k: i32,
        _top_p: f32,
        _min_p: f32,
        _rng: &mut u64,
    ) -> i32 {
        if self.idx >= self.steps.len() {
            return -1;
        }
        let t = self.steps[self.idx].token;
        self.idx += 1;
        t
    }

    fn pos(&self) -> i32 {
        self.pos
    }

    fn ctx(&self) -> i32 {
        self.ctx
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn session_tokens(&self) -> Vec<i32> {
        if self.live.is_empty() {
            self.prompt_tokens.clone()
        } else {
            self.live.clone()
        }
    }
}

#[cfg(feature = "native")]
struct NativeSerialKvIo<'s, 'm, 'v> {
    session: &'s mut ds4_core::Session<'m>,
    vocab: &'v ds4_core::Vocab,
}

#[cfg(feature = "native")]
impl SerialKvIo for NativeSerialKvIo<'_, '_, '_> {
    fn ctx(&self) -> i32 {
        self.session.ctx()
    }

    fn live_tokens(&self) -> Vec<i32> {
        if self.session.host().valid {
            self.session.host().tokens().to_vec()
        } else {
            Vec::new()
        }
    }

    fn render_tokens(&self, tokens: &[i32]) -> Result<Vec<u8>, GenerateError> {
        let mut text = Vec::new();
        for &token in tokens {
            text.extend(self.vocab.token_text(token));
        }
        Ok(text)
    }

    fn tokenize_suffix(&mut self, suffix: &[u8]) -> Result<Vec<i32>, GenerateError> {
        Ok(self.vocab.encode_rendered_bytes(suffix))
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        let tokens = ds4_core::TokenBuffer::from_tokens(tokens.to_vec());
        self.session
            .sync(&tokens)
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn save_payload(&mut self, path: &Path) -> Result<(), GenerateError> {
        self.session
            .save_payload(path)
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn load_payload_range(
        &mut self,
        path: &Path,
        offset: u64,
        length: u64,
    ) -> Result<(), GenerateError> {
        self.session
            .load_payload_range(path, offset, length)
            .map_err(|error| GenerateError::Engine(error.to_string()))
    }

    fn invalidate(&mut self) {
        self.session.invalidate();
    }
}

#[cfg(feature = "native")]
pub struct NativeDecode<'a> {
    model: &'a ds4_core::Model,
    vocab: Option<&'a ds4_core::Vocab>,
    session: Option<ds4_core::Session<'a>>,
    store: Option<KvStore>,
    session_disk_storable: bool,
    ctx: i32,
}

#[cfg(feature = "native")]
impl<'a> NativeDecode<'a> {
    pub fn new(model: &'a ds4_core::Model, ctx: i32) -> Self {
        Self {
            model,
            vocab: None,
            session: None,
            store: None,
            session_disk_storable: false,
            ctx,
        }
    }

    pub fn with_vocab(mut self, vocab: &'a ds4_core::Vocab) -> Self {
        self.vocab = Some(vocab);
        self
    }

    pub fn with_store(mut self, store: KvStore) -> Self {
        self.store = Some(store);
        self
    }

    fn session(&mut self) -> Result<&mut ds4_core::Session<'a>, GenerateError> {
        if self.session.is_none() {
            let s = self
                .model
                .session(self.ctx)
                .map_err(|e| GenerateError::Engine(e.to_string()))?;
            self.session = Some(s);
        }
        Ok(self.session.as_mut().unwrap())
    }
}

#[cfg(feature = "native")]
impl DecodeIo for NativeDecode<'_> {
    fn model_id(&self) -> i32 {
        self.model.model_id()
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        if let Some(v) = self.vocab {
            return Ok(v.encode_text(text));
        }
        self.model
            .tokenize_text(text)
            .map(|b| b.as_slice().to_vec())
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        if let Some(v) = self.vocab {
            return Ok(v.encode_rendered_bytes(text));
        }
        let s = std::str::from_utf8(text).map_err(|_| GenerateError::Engine("prompt not utf8".into()))?;
        self.model
            .tokenize_rendered_chat(s)
            .map(|b| b.as_slice().to_vec())
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn token_text(&self, token: i32) -> Result<Vec<u8>, GenerateError> {
        if let Some(v) = self.vocab {
            return Ok(v.token_text(token));
        }
        self.model
            .token_text(token)
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn token_is_stop(&self, token: i32) -> bool {
        if let Some(v) = self.vocab {
            return v.is_stop(token);
        }
        self.model.token_is_stop(token)
    }

    fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
        let buf = ds4_core::TokenBuffer::from_tokens(tokens.to_vec());
        self.session()?
            .sync(&buf)
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn sync_prompt(
        &mut self,
        prompt: &[u8],
        tokens: &[i32],
        disk_eligible: bool,
    ) -> Result<i32, GenerateError> {
        if self.store.is_none() {
            self.session_disk_storable = false;
            self.sync(tokens)?;
            return Ok(0);
        }

        let model_id = self.model.model_id();
        let quant_bits = self.model.routed_quant_bits();
        let vocab = self.vocab.unwrap_or_else(|| self.model.vocab());
        let save_current = self.session_disk_storable;
        self.session()?;
        let session = self.session.as_mut().ok_or_else(|| {
            GenerateError::Engine("native session was not created".into())
        })?;
        let store = self.store.as_mut().ok_or_else(|| {
            GenerateError::Engine("disk KV store was not configured".into())
        })?;
        let mut io = NativeSerialKvIo { session, vocab };
        let result = disk_sync_prompt(
            &mut io,
            store,
            model_id,
            quant_bits,
            prompt,
            tokens,
            DiskSyncPolicy {
                save_current,
                load: disk_eligible,
            },
        );
        self.session_disk_storable = result.is_ok() && disk_eligible;
        result
    }

    fn eval(&mut self, token: i32) -> Result<(), GenerateError> {
        self.session()?
            .eval(token)
            .map(|_| ())
            .map_err(|e| GenerateError::Engine(e.to_string()))
    }

    fn sample(
        &mut self,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        rng: &mut u64,
    ) -> i32 {
        match self.session() {
            Ok(s) => s.sample(temperature, top_k, top_p, min_p, rng),
            Err(_) => -1,
        }
    }

    fn pos(&self) -> i32 {
        self.session.as_ref().map(|s| s.pos()).unwrap_or(0)
    }

    fn ctx(&self) -> i32 {
        self.session.as_ref().map(|s| s.ctx()).unwrap_or(self.ctx)
    }

    fn generation(&self) -> u64 {
        self.session.as_ref().map(|s| s.generation()).unwrap_or(0)
    }

    fn session_tokens(&self) -> Vec<i32> {
        self.session
            .as_ref()
            .map(|s| s.host().tokens().to_vec())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod disk_sync_tests {
    use super::{
        discard_loaded, disk_sync_prompt, ordinary_disk_cache_eligible, DiskSyncPolicy,
        GenerateError, SerialKvIo,
    };
    use crate::parse::{parse_request, ParseEnv};
    use crate::route::{Api, ThinkMode, WireSurface};
    use ds4_kv::{read_envelope, Header, Options, Reason, Record, Store};
    use std::fs;
    use std::path::{Path, PathBuf};

    #[derive(Debug)]
    struct FakeSerial {
        ctx: i32,
        live: Vec<i32>,
        rendered_live: Vec<u8>,
        suffix_tokens: Vec<i32>,
        loaded_tokens: Vec<i32>,
        fail_load: bool,
        fail_sync: bool,
        invalidations: usize,
        syncs: Vec<Vec<i32>>,
        suffixes: Vec<Vec<u8>>,
        loads: Vec<(PathBuf, u64, u64)>,
        events: Vec<&'static str>,
    }

    impl FakeSerial {
        fn new(live: &[i32], rendered_live: &[u8]) -> Self {
            Self {
                ctx: 4096,
                live: live.to_vec(),
                rendered_live: rendered_live.to_vec(),
                suffix_tokens: Vec::new(),
                loaded_tokens: Vec::new(),
                fail_load: false,
                fail_sync: false,
                invalidations: 0,
                syncs: Vec::new(),
                suffixes: Vec::new(),
                loads: Vec::new(),
                events: Vec::new(),
            }
        }
    }

    impl SerialKvIo for FakeSerial {
        fn ctx(&self) -> i32 {
            self.ctx
        }

        fn live_tokens(&self) -> Vec<i32> {
            self.live.clone()
        }

        fn render_tokens(&self, _tokens: &[i32]) -> Result<Vec<u8>, GenerateError> {
            Ok(self.rendered_live.clone())
        }

        fn tokenize_suffix(&mut self, suffix: &[u8]) -> Result<Vec<i32>, GenerateError> {
            self.suffixes.push(suffix.to_vec());
            Ok(self.suffix_tokens.clone())
        }

        fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
            self.events.push("sync");
            self.syncs.push(tokens.to_vec());
            if self.fail_sync {
                return Err(GenerateError::Engine("injected suffix sync failure".into()));
            }
            self.live = tokens.to_vec();
            Ok(())
        }

        fn save_payload(&mut self, path: &Path) -> Result<(), GenerateError> {
            self.events.push("save");
            fs::write(path, b"current-payload").map_err(|e| GenerateError::Engine(e.to_string()))
        }

        fn load_payload_range(
            &mut self,
            path: &Path,
            offset: u64,
            length: u64,
        ) -> Result<(), GenerateError> {
            self.events.push("load");
            self.loads.push((path.to_path_buf(), offset, length));
            if self.fail_load {
                return Err(GenerateError::Engine("injected payload load failure".into()));
            }
            self.live = self.loaded_tokens.clone();
            Ok(())
        }

        fn invalidate(&mut self) {
            self.events.push("invalidate");
            self.invalidations += 1;
            self.live.clear();
        }
    }

    fn store(tag: &str) -> (PathBuf, Store) {
        let dir =
            std::env::temp_dir().join(format!("ds4-server-disk-sync-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let options = Options {
            min_tokens: 1,
            cold_max_tokens: 32,
            continued_interval_tokens: 8,
            boundary_trim_tokens: 0,
            boundary_align_tokens: 0,
        };
        let store = Store::open(&dir, 16, true, options).unwrap();
        (dir, store)
    }

    fn candidate(store: &mut Store, text: &[u8], tokens: u32) -> PathBuf {
        store
            .write(Record {
                header: Header {
                    quant_bits: 2,
                    reason: Reason::Evict,
                    ext_flags: 0,
                    model_id: 0,
                    tokens,
                    hits: 0,
                    ctx_size: 4096,
                    created_at: 1,
                    last_used: 1,
                    payload_bytes: 0,
                    text_bytes: 0,
                },
                text: text.to_vec(),
                payload: b"candidate-payload".to_vec(),
                trailer: Vec::new(),
            })
            .unwrap()
    }

    #[test]
    fn disk_sync_prompt_cold_miss_syncs_canonical_prompt() {
        let (dir, mut store) = store("cold");
        let mut io = FakeSerial::new(&[], b"");

        let cached = disk_sync_prompt(
            &mut io,
            &mut store,
            0,
            2,
            b"cold prompt",
            &[1, 2],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 0);
        assert_eq!(io.syncs, [vec![1, 2]]);
        assert!(io.loads.is_empty());
        assert_eq!(io.events, ["sync"]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_reuses_current_exact_and_byte_prefixes() {
        let (dir, mut store) = store("current");
        let mut exact = FakeSerial::new(&[1, 2], b"unused");
        let cached = disk_sync_prompt(
            &mut exact,
            &mut store,
            0,
            2,
            b"hello",
            &[1, 2, 3],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 2);
        assert_eq!(exact.syncs, [vec![1, 2, 3]]);
        assert!(exact.loads.is_empty());

        let mut byte_prefix = FakeSerial::new(&[9], b"hello ");
        byte_prefix.suffix_tokens = vec![10, 11];
        let cached = disk_sync_prompt(
            &mut byte_prefix,
            &mut store,
            0,
            2,
            b"hello world",
            &[90, 91],
            DiskSyncPolicy {
                save_current: false,
                load: false,
            },
        )
        .unwrap();
        assert_eq!(cached, 1);
        assert_eq!(byte_prefix.suffixes, [b"world".to_vec()]);
        assert_eq!(byte_prefix.syncs, [vec![9, 10, 11]]);
        assert!(byte_prefix.loads.is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_hit_loads_exact_tokens_retokenizes_suffix_and_touches_hit() {
        let (dir, mut store) = store("hit");
        let path = candidate(&mut store, b"hello ", 2);
        let envelope = read_envelope(&path).unwrap();
        let mut io = FakeSerial::new(&[7], b"old conversation");
        io.loaded_tokens = vec![41, 42];
        io.suffix_tokens = vec![43];

        let cached =
            disk_sync_prompt(
                &mut io,
                &mut store,
                0,
                2,
                b"hello world",
                &[90, 91],
                DiskSyncPolicy {
                    save_current: true,
                    load: true,
                },
            )
            .unwrap();

        assert_eq!(cached, 2);
        assert_eq!(io.suffixes, [b"world".to_vec()]);
        assert_eq!(io.syncs, [vec![41, 42, 43]]);
        assert_eq!(
            io.loads,
            [(
                path.clone(),
                envelope.payload_offset,
                envelope.header.payload_bytes
            )]
        );
        let save = io
            .events
            .iter()
            .position(|event| *event == "save")
            .expect("live checkpoint save event");
        let load = io
            .events
            .iter()
            .position(|event| *event == "load")
            .expect("candidate load event");
        assert!(
            save < load,
            "live checkpoint must be staged before a disk restore: {:?}",
            io.events
        );
        assert!(store
            .text_prefix_candidate(b"old conversation", 0, 2, 4096)
            .unwrap()
            .is_some());
        assert_eq!(store.read(&path).unwrap().header.hits, 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_token_count_mismatch_discards_then_cold_syncs() {
        let (dir, mut store) = store("count-mismatch");
        let path = candidate(&mut store, b"hello ", 2);
        let mut io = FakeSerial::new(&[], b"");
        io.loaded_tokens = vec![41];
        io.suffix_tokens = vec![43];

        let cached =
            disk_sync_prompt(
                &mut io,
                &mut store,
                0,
                2,
                b"hello world",
                &[90, 91],
                DiskSyncPolicy {
                    save_current: false,
                    load: true,
                },
            )
            .unwrap();

        assert_eq!(cached, 0);
        assert_eq!(io.invalidations, 1);
        assert_eq!(io.syncs, [vec![90, 91]]);
        assert!(!path.exists());
        assert!(store.entries().is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_load_failure_keeps_candidate_and_cold_syncs() {
        let (dir, mut store) = store("load-failure");
        let path = candidate(&mut store, b"hello ", 2);
        let mut io = FakeSerial::new(&[], b"");
        io.fail_load = true;

        let cached = disk_sync_prompt(
            &mut io,
            &mut store,
            0,
            2,
            b"hello world",
            &[90, 91],
            DiskSyncPolicy {
                save_current: false,
                load: true,
            },
        )
        .unwrap();

        assert_eq!(cached, 0);
        assert_eq!(io.invalidations, 1);
        assert_eq!(io.syncs, [vec![90, 91]]);
        assert!(path.exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_sync_prompt_suffix_sync_failure_discards_and_returns_error() {
        let (dir, mut store) = store("sync-failure");
        let path = candidate(&mut store, b"hello ", 2);
        let mut io = FakeSerial::new(&[], b"");
        io.loaded_tokens = vec![41, 42];
        io.suffix_tokens = vec![43];
        io.fail_sync = true;

        let error =
            disk_sync_prompt(
                &mut io,
                &mut store,
                0,
                2,
                b"hello world",
                &[90, 91],
                DiskSyncPolicy {
                    save_current: false,
                    load: true,
                },
            )
            .unwrap_err();

        assert!(error.to_string().contains("injected suffix sync failure"));
        assert_eq!(io.invalidations, 1);
        assert_eq!(io.syncs, [vec![41, 42, 43]]);
        assert!(!path.exists());
        assert!(store.entries().is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discard_failure_still_resets_marker_and_invalidates_session() {
        let (dir, mut store) = store("discard-race");
        let mut io = FakeSerial::new(&[41, 42], b"hello");
        store.continued_last_store_tokens = 2;

        discard_loaded(&mut store, &mut io, &dir.join("missing.kvc"));

        assert_eq!(store.continued_last_store_tokens, 0);
        assert_eq!(io.invalidations, 1);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_cache_eligibility_excludes_protocol_state() {
        let env = ParseEnv {
            default_model: "ds4".into(),
            default_tokens: 16,
            default_effort: ThinkMode::None,
            default_temp: 0.0,
            live_ids: Vec::new(),
        };
        let ordinary = parse_request(
            WireSurface::OpenaiChat,
            &env,
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#,
        )
        .unwrap();
        assert!(ordinary_disk_cache_eligible(&ordinary));

        let mut cases = Vec::new();
        let mut request = ordinary.clone();
        request.api = Api::Anthropic;
        cases.push(request);
        let mut request = ordinary.clone();
        request.api = Api::Responses;
        cases.push(request);
        let mut request = ordinary.clone();
        request.think_mode = ThinkMode::Low;
        cases.push(request);
        let mut request = ordinary.clone();
        request.has_tools = true;
        cases.push(request);
        let mut request = ordinary.clone();
        request.has_tool_results = true;
        cases.push(request);
        let mut request = ordinary;
        request.live_call_ids.push("call_1".into());
        cases.push(request);

        assert!(cases.iter().all(|request| !ordinary_disk_cache_eligible(request)));
    }
}
