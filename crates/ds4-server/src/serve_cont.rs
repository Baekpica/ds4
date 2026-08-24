//! Continuous-lane execution: a push-based per-token stepper (pure, tape
//! testable) plus the native `ContLane` that drives the engine's rolling
//! scheduler through `ds4-core::BatchCtx`. Width-1 today: the accept loop
//! is serial, so one request occupies the lane at a time; the engine-side
//! bank/admit machinery is the same one C's continuous lane uses.
//! Corrective retry (`decode_again`) and continuation publish route serial
//! by the needs word, so this path never re-decodes.

use std::io::Write;
use std::time::Instant;

use crate::dsml::{SamplePolicy, SampleOverride};
use crate::generate::{
    render_prompt, stream_req_from_parsed, GenerateError, GenerateOutcome,
};
use crate::parse::{ParsedRequest, ToolChoice};
use crate::parse::{DEFAULT_MIN_P, DEFAULT_TEMPERATURE, DEFAULT_TOP_P};
use crate::render::syntax_for_model_id;
use crate::retry::{terminal_finish, truncation_outcome, TruncationOutcome};
use crate::route::{decode_budget, think_mode_enabled, Api, ReqKind};
use crate::stream::{
    final_response, openai_sse_finish_live, openai_sse_stream_update, openai_stream_start,
    sse_chunk, sse_done, sse_headers, OpenaiStream, ReqTimings, StreamReq, Writer,
};
use crate::tools::{assign_tool_ids, parse_generated_for_response, SemAccum};

/// Pure continuous-request stepper. The caller feeds decoded pieces and
/// receives wire bytes; the engine (or a tape) owns token production.
/// OpenAI chat/completions only — every other surface routes serial.
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
        if req.stream {
            w.out.extend_from_slice(&sse_headers(cors));
            if req.api == Api::Openai && req.kind == ReqKind::Chat {
                let mut st = openai_stream_start(&req);
                st.tool.use_random_ids();
                oa = Some(st);
                sse_chunk(&mut w, &req, job_id, None, None);
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
            (
                parsed.temperature,
                parsed.top_k,
                parsed.top_p,
                parsed.min_p,
            )
        }
    }

    pub fn feed(&mut self, piece: &[u8]) -> ContStep {
        let feed = self.acc.feed(piece, &self.stops);
        if self.req.stream {
            let view = &self.acc.text[..feed.emit_limit.min(self.acc.text.len())];
            match self.req.kind {
                ReqKind::Completion => {
                    if let Some(delta) = last_delta(&self.acc.text, feed.emit_limit, piece.len()) {
                        sse_chunk(&mut self.w, &self.req, &self.job_id, Some(delta), None);
                    }
                }
                ReqKind::Chat => {
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
            match self.req.kind {
                ReqKind::Completion => {
                    sse_chunk(&mut self.w, &self.req, &self.job_id, None, Some(self.finish));
                    sse_done(&mut self.w, &self.req, &self.job_id, self.prompt_n, completion);
                }
                ReqKind::Chat => {
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
            }
        } else {
            let bytes = final_response(
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
            );
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

/// Trait seam so `handle_client_inner` can drive a continuous lane without
/// the native feature (tests supply a scripted implementation).
pub trait ContExec {
    fn model_id(&self) -> i32;
    fn seq_cap(&self) -> i32;
    fn encode_chat(&self, rendered: &[u8]) -> Vec<i32>;
    fn encode_text(&self, text: &str) -> Vec<i32>;
    fn generate(
        &mut self,
        parsed: &ParsedRequest,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        t_arrive: Instant,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError>;
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

    /// Native continuous lane: engine batch context + host vocab. Width-1
    /// (the accept loop admits one request at a time).
    pub struct ContLane<'m> {
        pub batch: BatchCtx<'m>,
        pub vocab: &'m Vocab,
        pub model_id: i32,
        /// Family EOT for the per-seq stop, like the C server's job prep;
        /// the engine's `-1` default is the base EOS, not the family EOT.
        pub eos: i32,
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
        io_failed: bool,
        host_abort: bool,
        engine_eos: bool,
        n_cached: i32,
        n_computed: i32,
        t_arrive: Instant,
        t_admit: Option<Instant>,
        t_first: Option<Instant>,
        t_done: Option<Instant>,
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

        fn on_done(&mut self, _user: usize, _tokens: &[i32], finish: i32) {
            self.engine_eos = finish == 1;
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

        fn on_admitted(&mut self, _user: usize, n_cached: i32, n_computed: i32, _bank: i32) -> bool {
            self.n_cached = n_cached;
            self.n_computed = n_computed;
            self.t_admit = Some(Instant::now());
            self.admitted = true;
            if let Some(head) = self.head.take() {
                self.push(&head);
            }
            !self.io_failed
        }
    }

    impl ContExec for ContLane<'_> {
        fn model_id(&self) -> i32 {
            self.model_id
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
            out: &mut dyn Write,
        ) -> Result<GenerateOutcome, GenerateError> {
            let prompt = render_prompt(parsed, self.model_id)?;
            let tokens = match parsed.kind {
                ReqKind::Completion => {
                    self.vocab
                        .encode_text(std::str::from_utf8(&prompt).unwrap_or(""))
                }
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
            let mut admit = ContAdmit::cold(1, tokens, stepper.max_tokens.max(1));
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
                io_failed: false,
                host_abort: false,
                engine_eos: false,
                n_cached: 0,
                n_computed: 0,
                t_arrive,
                t_admit: None,
                t_first: None,
                t_done: None,
            };
            self.batch
                .continuous_generate(&mut job)
                .map_err(|e| GenerateError::Engine(e.to_string()))?;
            if !job.admitted {
                /* Engine declined (oversized/budget): no bytes were sent,
                 * the caller falls back to the serial lane. */
                return Err(GenerateError::Unsupported(
                    "continuous admission rejected; serial fallback",
                ));
            }
            if job.io_failed {
                return Err(GenerateError::Io);
            }
            let timings = {
                let done = job.t_done.unwrap_or_else(Instant::now);
                let completion = job.stepper.completion();
                match job.t_first {
                    Some(first) if completion > 0 => ReqTimings {
                        valid: true,
                        ttft_ms: first.duration_since(job.t_arrive).as_secs_f64() * 1e3,
                        prefill_ms: first
                            .duration_since(job.t_admit.unwrap_or(job.t_arrive))
                            .as_secs_f64()
                            * 1e3,
                        decode_ms: done.duration_since(first).as_secs_f64() * 1e3,
                        prefill_tokens: job.n_computed,
                        prefill_cached: job.n_cached,
                        decode_tokens: completion,
                        decode_steps: completion,
                    },
                    _ => ReqTimings::default(),
                }
            };
            let engine_eos = job.engine_eos && !job.host_abort;
            let n_cached = job.n_cached;
            let n_computed = job.n_computed;
            let io_ok = !job.io_failed;
            drop(job);
            drop(adapter);
            let (tail, outcome) = stepper.finalize(engine_eos, n_cached, n_computed, timings, cors);
            if io_ok && !tail.is_empty() {
                out.write_all(&tail).map_err(|_| GenerateError::Io)?;
                let _ = out.flush();
            }
            Ok(outcome)
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
