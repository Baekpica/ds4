//! Blocking accept loop: HTTP door + parsers + admission + projectors +
//! serial decode (including finalize tool_calls) when a `DecodeIo` is supplied.

use std::ffi::OsStr;
use std::io::{Error, ErrorKind, Write};
use std::net::{TcpListener, TcpStream};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ds4_sys::{libc_atof, libc_atoi, libc_strtoull10};

use crate::admit::{
    enqueue, enqueue_release, enqueue_shed_error, next_job_id, preparse_shed, queue_unlink_head,
    AdmitState, EnqVerdict, SERVER_SHUTTING_DOWN, SHED_CONT_HOLD, SHED_QUEUE_AGE, SHED_SLOW_READER,
};
use crate::cont::{monotonic_now, place_bank_continuation, ContOwner, ContPin, ContRegistry};
use crate::error::{http_response_bytes, wire_http_error_bytes};
use crate::generate::{generate_terminal_at, DecodeIo, GenerateError, GenerateOutcome};
use crate::http::{
    chunked_enabled, parse_surface_for_path, read_http_request, shed_surface_for_path,
};
#[cfg(feature = "native")]
use crate::metrics::MemCell;
use crate::metrics::{
    gov_modes_from_env, render_metrics, render_stats_json_ex, RouteMetrics, RuntimeMetrics,
};
use crate::models::{model_id_known, model_one_json, models_list_json};
use crate::parse::{parse_request, ParseEnv};
use crate::route::{
    route_decide, Api, RouteEnv, ThinkMode, WireSurface, LANE_CONTINUOUS, LANE_STATIC,
    NEED_BANK_FRONTIER,
};
use crate::serve_cont::{cont_prompt_tokens, ContExec};
use crate::serve_serial_reclaim::{
    resolve_serial_fit, serial_capacity_refuse_msg, serial_fit_from_native, serial_reclaim_gate,
    MemFloor, SerialFitQuote, SerialReclaimOutcome,
};
use crate::serve_static::{
    run_static_routed, write_static_completion, DetachedStatic, StaticFinish, StaticJob, StaticRow,
    StaticSettle, STATIC_WIDTH_ERR,
};

#[path = "serve_owner_cont.rs"]
mod owner_cont;
#[path = "serve_owner_static.rs"]
mod owner_static;
use crate::stream::unix_now;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub listen_host: String,
    pub listen_port: u16,
    pub model_id: String,
    pub model_name: String,
    pub ctx: i32,
    pub default_tokens: i32,
    pub cors: bool,
    pub codex_models_json: Option<String>,
    pub max_queue: i32,
    pub max_queue_bytes: u64,
    pub max_clients: i32,
    pub max_queue_age_s: f64,
    pub out_agg_cap_bytes: u64,
    pub out_agg_evict_min_bytes: u64,
    pub disconnect_abort: bool,
    pub continuous: bool,
    pub have_engine: bool,
    pub stop_requested: Option<fn() -> bool>,
    pub mem_floor_gb: u64,
    pub(crate) serial_fit: Option<SerialFitQuote>,
}

pub(crate) fn env_i32_bound(name: &str, default: i32) -> i32 {
    let value = std::env::var_os(name);
    parse_i32_bound(value.as_deref(), default)
}

fn env_u64_bound(name: &str, default: u64) -> u64 {
    let value = std::env::var_os(name);
    parse_u64_bound(value.as_deref(), default)
}

fn env_f64_bound(name: &str, default: f64) -> f64 {
    let value = std::env::var_os(name);
    parse_f64_bound(value.as_deref(), default)
}

fn os_str_bytes(value: &OsStr) -> &[u8] {
    #[cfg(unix)]
    {
        value.as_bytes()
    }
    #[cfg(not(unix))]
    {
        value.as_encoded_bytes()
    }
}

fn parse_i32_bound(value: Option<&OsStr>, default: i32) -> i32 {
    match value {
        Some(value) => libc_atoi(os_str_bytes(value)).max(0),
        None => default,
    }
}

fn parse_u64_bound(value: Option<&OsStr>, default: u64) -> u64 {
    match value {
        Some(value) => libc_strtoull10(os_str_bytes(value)),
        None => default,
    }
}

fn parse_f64_bound(value: Option<&OsStr>, default: f64) -> f64 {
    match value {
        Some(value) => {
            let value = libc_atof(os_str_bytes(value));
            if value < 0.0 {
                0.0
            } else {
                value
            }
        }
        None => default,
    }
}

fn parse_default_on(value: Option<&OsStr>) -> bool {
    value.map(os_str_bytes) != Some(b"0")
}

fn process_cont_tools() -> (bool, bool) {
    crate::route::cont_tools_from_env(
        std::env::var_os("DS4_SERVER_CONT_TOOLS_ANTHROPIC")
            .as_deref()
            .map(os_str_bytes),
        std::env::var_os("DS4_SERVER_CONT_TOOLS_RESPONSES")
            .as_deref()
            .map(os_str_bytes),
    )
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen_host: "127.0.0.1".into(),
            listen_port: 8000,
            model_id: "ds4".into(),
            model_name: "ds4".into(),
            ctx: 8192,
            default_tokens: 393216,
            cors: false,
            codex_models_json: None,
            max_queue: env_i32_bound("DS4_SERVER_MAX_QUEUE", 256),
            max_queue_bytes: env_u64_bound("DS4_SERVER_MAX_QUEUE_BYTES", 256 * 1024 * 1024),
            max_clients: env_i32_bound("DS4_SERVER_MAX_CLIENTS", 256),
            max_queue_age_s: env_f64_bound("DS4_SERVER_MAX_QUEUE_AGE_S", 600.0),
            out_agg_cap_bytes: env_u64_bound("DS4_SERVER_OUT_AGG_CAP", JOB_SINK_AGG_CAP_BYTES),
            out_agg_evict_min_bytes: env_u64_bound(
                "DS4_SERVER_OUT_AGG_EVICT_MIN",
                JOB_SINK_AGG_EVICT_MIN_BYTES,
            ),
            disconnect_abort: parse_default_on(
                std::env::var_os("DS4_SERVER_DISCONNECT_ABORT").as_deref(),
            ),
            continuous: parse_default_on(std::env::var_os("DS4_SERVER_CONTINUOUS").as_deref()),
            have_engine: false,
            stop_requested: None,
            mem_floor_gb: mem_floor_gb_from_env(),
            serial_fit: None,
        }
    }
}

fn mem_floor_gb_from_env() -> u64 {
    let env = std::env::var_os("DS4_MEM_FLOOR_GB");
    MemFloor::from_env_gb(env.as_deref().map(os_str_bytes)).gb()
}

impl ServerConfig {
    /// Integration-test fixture. Built inside the crate so `serial_fit` stays
    /// `pub(crate)` and external tests never need functional-update syntax.
    pub fn test_cfg() -> Self {
        let mut cfg = Self::default();
        cfg.have_engine = true;
        cfg
    }

    pub fn apply_mem_floor_gb(&mut self, raw: &str) {
        self.mem_floor_gb = MemFloor::from_cli_or_env(Some(raw.as_bytes()), None).gb();
    }
}

#[derive(Debug, Default)]
pub struct ServerInner {
    pub admit: AdmitState,
    pub metrics: RouteMetrics,
    pub runtime: RuntimeMetrics,
    pub creg: ContRegistry,
    pub boot_stamp: u64,
    pub have_engine: bool,
    disconnect_abort: bool,
    out_agg_cap_bytes: u64,
    out_agg_evict_min_bytes: u64,
}

impl ServerInner {
    pub fn from_cfg(cfg: &ServerConfig) -> Self {
        let mut s = Self::default();
        s.admit.max_queue = cfg.max_queue;
        s.admit.max_queue_bytes = cfg.max_queue_bytes;
        s.admit.max_clients = cfg.max_clients;
        s.runtime.memgov.gov_modes = gov_modes_from_env();
        s.boot_stamp = unix_now() as u64;
        s.have_engine = cfg.have_engine;
        s.disconnect_abort = cfg.disconnect_abort;
        s.out_agg_cap_bytes = cfg.out_agg_cap_bytes;
        s.out_agg_evict_min_bytes = cfg.out_agg_evict_min_bytes;
        s
    }

    pub fn render_runtime(&self, now: u64) -> RuntimeMetrics {
        let mut rt = self.runtime.clone();
        rt.uptime_seconds = if self.boot_stamp != 0 && now >= self.boot_stamp {
            now - self.boot_stamp
        } else {
            0
        };
        rt.creg_records_live = self.creg.n_live() as u64;
        #[cfg(feature = "native")]
        if self.have_engine {
            overlay_live_census(&mut rt.memgov);
        }
        rt
    }
}

#[cfg(feature = "native")]
fn overlay_live_census(g: &mut crate::metrics::MemgovSnap) {
    let snap = ds4_core::snapshot_mem();
    g.census_supported = snap.census.supported;
    g.census_faults = snap.census.faults;
    g.census_epoch = snap.census.epoch;
    g.torn_fallbacks = snap.census.torn_fallbacks;
    for c in 0..17 {
        for d in 0..2 {
            let src = snap.census.cells[c][d];
            g.cells[c][d] = MemCell {
                requested: src.requested,
                committed: src.committed,
                freed_requested: src.freed_requested,
                freed_committed: src.freed_committed,
            };
        }
    }
    g.obs_status = snap.observe.status.clamp(0, 2) as u8;
    g.obs_source = snap.observe.source.clamp(0, 2) as u8;
    g.obs_free = snap.observe.free_bytes;
    g.obs_total = snap.observe.total_bytes;
    g.obs_cuda_free = snap.observe.cuda_free_bytes;
    g.obs_meminfo = snap.observe.meminfo_avail_bytes;
    g.substrate_outstanding = snap.substrate_outstanding;
    g.emit_substrate = true;
}

fn write_all(stream: &mut TcpStream, bytes: &[u8]) {
    let _ = stream.write_all(bytes);
}

const JOB_SINK_CAP_BYTES: u64 = 16 * 1024 * 1024;
const JOB_SINK_AGG_CAP_BYTES: u64 = 64 * 1024 * 1024;
const JOB_SINK_AGG_EVICT_MIN_BYTES: u64 = 256 * 1024;
const CLIENT_POLL: Duration = Duration::from_millis(50);
const CLIENT_IO_TIMEOUT: Duration = Duration::from_secs(10);
const CLIENT_SEND_STALL_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Default)]
struct SinkBuffer {
    bytes: Vec<u8>,
    gone: bool,
    slow: bool,
    closed: bool,
}

struct SinkState {
    inner: Arc<Mutex<ServerInner>>,
    probe: Option<TcpStream>,
    buffer: Mutex<SinkBuffer>,
    ready: Condvar,
}

impl SinkState {
    fn lock(&self) -> std::sync::MutexGuard<'_, SinkBuffer> {
        self.buffer.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn gone(&self) -> bool {
        self.lock().gone
    }

    fn slow(&self) -> bool {
        self.lock().slow
    }

    fn socket_disconnected(&self) -> bool {
        self.probe.as_ref().is_some_and(client_disconnected)
    }

    fn observe_disconnect(&self) -> bool {
        if !self.socket_disconnected() {
            return false;
        }
        self.cancel(false);
        true
    }

    #[cfg(test)]
    fn backlog_bytes(&self) -> u64 {
        self.lock().bytes.len() as u64
    }

    fn take(&self) -> Vec<u8> {
        let mut buffer = self.lock();
        let bytes = std::mem::take(&mut buffer.bytes);
        let mut g = lock_inner(&self.inner);
        g.runtime.out_backlog_bytes = g
            .runtime
            .out_backlog_bytes
            .saturating_sub(bytes.len() as u64);
        bytes
    }

    fn cancel(&self, slow: bool) {
        let mut buffer = self.lock();
        buffer.gone = true;
        buffer.slow |= slow;
        let bytes = std::mem::take(&mut buffer.bytes);
        let mut g = lock_inner(&self.inner);
        g.runtime.out_backlog_bytes = g
            .runtime
            .out_backlog_bytes
            .saturating_sub(bytes.len() as u64);
        drop(g);
        drop(buffer);
        drop(bytes);
        self.ready.notify_all();
    }
}

struct JobSink {
    state: Arc<SinkState>,
}

impl Write for JobSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut buffer = self.state.lock();
        if buffer.gone || buffer.closed {
            return Err(Error::new(ErrorKind::BrokenPipe, "client disconnected"));
        }
        if bytes.is_empty() {
            return Ok(0);
        }
        let mut g = lock_inner(&self.state.inner);
        let job_next = buffer.bytes.len().saturating_add(bytes.len()) as u64;
        let aggregate_next = g
            .runtime
            .out_backlog_bytes
            .saturating_add(bytes.len() as u64);
        if job_next > JOB_SINK_CAP_BYTES
            || (g.out_agg_cap_bytes > 0
                && aggregate_next > g.out_agg_cap_bytes
                && job_next >= g.out_agg_evict_min_bytes)
        {
            let pending = std::mem::take(&mut buffer.bytes);
            g.runtime.out_backlog_bytes = g
                .runtime
                .out_backlog_bytes
                .saturating_sub(pending.len() as u64);
            buffer.slow = true;
            buffer.gone = true;
            drop(g);
            drop(buffer);
            drop(pending);
            self.state.ready.notify_all();
            return Err(Error::new(ErrorKind::WouldBlock, "client output sink full"));
        }
        buffer.bytes.extend_from_slice(bytes);
        g.runtime.out_backlog_bytes = aggregate_next;
        drop(g);
        drop(buffer);
        self.state.ready.notify_one();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let abort = lock_inner(&self.state.inner).disconnect_abort;
        if self.state.gone() || (abort && self.state.observe_disconnect()) {
            Err(Error::new(ErrorKind::BrokenPipe, "client disconnected"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalCommit {
    Committed,
    GoneBeforeStart,
    FailedAfterStart,
    Full,
}

trait TerminalSink: Write {
    fn commit_tool_terminal(
        &mut self,
        inner: &Mutex<ServerInner>,
        api: Api,
        generated: &GenerateOutcome,
        terminal: Vec<u8>,
    ) -> TerminalCommit;
}

fn publish_tool_turn(inner: &Mutex<ServerInner>, api: Api, generated: &GenerateOutcome) {
    lock_inner(inner).creg.publish_serial(
        api,
        &generated.tool_ids,
        generated.generation,
        generated.frontier,
        monotonic_now(),
    );
}

fn reserve_terminal(inner: &Mutex<ServerInner>, bytes: u64) -> bool {
    let mut g = lock_inner(inner);
    if bytes > JOB_SINK_CAP_BYTES
        || (g.out_agg_cap_bytes > 0
            && g.runtime.out_backlog_bytes.saturating_add(bytes) > g.out_agg_cap_bytes)
    {
        return false;
    }
    g.runtime.out_backlog_bytes = g.runtime.out_backlog_bytes.saturating_add(bytes);
    true
}

fn release_terminal(inner: &Mutex<ServerInner>, bytes: u64) {
    let mut g = lock_inner(inner);
    g.runtime.out_backlog_bytes = g.runtime.out_backlog_bytes.saturating_sub(bytes);
}

enum DirectTerminalIo<'a> {
    Disconnected,
    Send(&'a [u8]),
}

struct DirectSink<'a>(&'a mut TcpStream);

impl Write for DirectSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        send_all_nonblocking(self.0, bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

fn direct_terminal_commit(
    inner: &Mutex<ServerInner>,
    api: Api,
    generated: &GenerateOutcome,
    terminal: &[u8],
    mut io: impl FnMut(DirectTerminalIo<'_>) -> std::io::Result<bool>,
) -> TerminalCommit {
    let bytes = terminal.len() as u64;
    if !reserve_terminal(inner, bytes) {
        return TerminalCommit::Full;
    }
    publish_tool_turn(inner, api, generated);
    if !matches!(io(DirectTerminalIo::Disconnected), Ok(false)) {
        lock_inner(inner).creg.demote_serial();
        release_terminal(inner, bytes);
        return TerminalCommit::GoneBeforeStart;
    }
    let sent = io(DirectTerminalIo::Send(terminal)).is_ok();
    release_terminal(inner, bytes);
    if sent {
        TerminalCommit::Committed
    } else {
        TerminalCommit::FailedAfterStart
    }
}

impl TerminalSink for DirectSink<'_> {
    fn commit_tool_terminal(
        &mut self,
        inner: &Mutex<ServerInner>,
        api: Api,
        generated: &GenerateOutcome,
        terminal: Vec<u8>,
    ) -> TerminalCommit {
        direct_terminal_commit(inner, api, generated, &terminal, |action| match action {
            DirectTerminalIo::Disconnected => Ok(client_disconnected(self.0)),
            DirectTerminalIo::Send(bytes) => send_all_nonblocking(self.0, bytes).map(|_| false),
        })
    }
}

impl TerminalSink for JobSink {
    fn commit_tool_terminal(
        &mut self,
        inner: &Mutex<ServerInner>,
        api: Api,
        generated: &GenerateOutcome,
        terminal: Vec<u8>,
    ) -> TerminalCommit {
        let n = terminal.len() as u64;
        let mut buffer = self.state.lock();
        let mut g = lock_inner(inner);
        let job_next = buffer.bytes.len().saturating_add(terminal.len()) as u64;
        let aggregate_next = g.runtime.out_backlog_bytes.saturating_add(n);
        if job_next > JOB_SINK_CAP_BYTES
            || (g.out_agg_cap_bytes > 0 && aggregate_next > g.out_agg_cap_bytes)
            || buffer.bytes.try_reserve(terminal.len()).is_err()
        {
            let pending = std::mem::take(&mut buffer.bytes);
            g.runtime.out_backlog_bytes = g
                .runtime
                .out_backlog_bytes
                .saturating_sub(pending.len() as u64);
            buffer.slow = true;
            buffer.gone = true;
            drop(g);
            drop(buffer);
            drop(pending);
            self.state.ready.notify_all();
            return TerminalCommit::Full;
        }

        g.runtime.out_backlog_bytes = aggregate_next;
        g.creg.publish_serial(
            api,
            &generated.tool_ids,
            generated.generation,
            generated.frontier,
            monotonic_now(),
        );
        if buffer.gone || self.state.socket_disconnected() {
            g.creg.demote_serial();
            let pending = std::mem::take(&mut buffer.bytes);
            g.runtime.out_backlog_bytes = g
                .runtime
                .out_backlog_bytes
                .saturating_sub(n.saturating_add(pending.len() as u64));
            buffer.gone = true;
            drop(g);
            drop(buffer);
            drop(pending);
            self.state.ready.notify_all();
            return TerminalCommit::GoneBeforeStart;
        }
        buffer.bytes.extend_from_slice(&terminal);
        drop(g);
        drop(buffer);
        self.state.ready.notify_one();
        TerminalCommit::Committed
    }
}

impl Drop for JobSink {
    fn drop(&mut self) {
        self.state.lock().closed = true;
        self.state.ready.notify_all();
    }
}

#[cfg(test)]
fn job_sink(inner: Arc<Mutex<ServerInner>>) -> (JobSink, Arc<SinkState>) {
    job_sink_with_probe(inner, None)
}

fn job_sink_with_probe(
    inner: Arc<Mutex<ServerInner>>,
    probe: Option<TcpStream>,
) -> (JobSink, Arc<SinkState>) {
    let state = Arc::new(SinkState {
        inner,
        probe,
        buffer: Mutex::new(SinkBuffer::default()),
        ready: Condvar::new(),
    });
    (
        JobSink {
            state: Arc::clone(&state),
        },
        state,
    )
}

fn api_for_surface(surf: WireSurface) -> Api {
    match surf {
        WireSurface::Anthropic => Api::Anthropic,
        WireSurface::Responses => Api::Responses,
        _ => Api::Openai,
    }
}

fn continuation_conflict_msg(api: Api) -> &'static str {
    match api {
        Api::Responses => {
            "Responses continuation state is not available; retry by replaying the full input history"
        }
        _ => {
            "Anthropic continuation state is not available; retry by replaying the full messages history"
        }
    }
}

fn refuse_bank_continuation<W: Write>(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    job: &mut PreparedJob,
    exec: &dyn ContExec,
    out: &mut W,
) -> Option<Settlement> {
    if job.parsed.needs & NEED_BANK_FRONTIER == 0 {
        return None;
    }
    let now = monotonic_now();
    let claim = lock_inner(inner)
        .creg
        .bank_claim(job.parsed.api, &job.parsed.live_call_ids, now);
    let live = claim.and_then(|(bank, _, _)| exec.bank_live(bank));
    match place_bank_continuation(claim, live) {
        Ok(bank) => {
            job.parsed.directed_bank = Some(bank);
            None
        }
        Err(_) => Some(write_continuation_conflict(cfg, job, out)),
    }
}

fn write_continuation_conflict<W: Write>(
    cfg: &ServerConfig,
    job: &PreparedJob,
    out: &mut W,
) -> Settlement {
    let ok = out
        .write_all(&wire_http_error_bytes(
            job.surface,
            409,
            continuation_conflict_msg(job.parsed.api),
            cfg.cors,
            None,
        ))
        .is_ok();
    if ok {
        Settlement::COMPLETED
    } else {
        Settlement::CANCELED
    }
}

fn publish_continuous_tool_turn(
    inner: &Mutex<ServerInner>,
    api: Api,
    bank: Option<i32>,
    generated: &GenerateOutcome,
) {
    if !matches!(api, Api::Anthropic | Api::Responses) {
        return;
    }
    if generated.tool_ids.is_empty() || generated.finish == "error" || generated.finish == "length"
    {
        return;
    }
    let Some(bank) = bank.filter(|bank| *bank >= 0) else {
        return;
    };
    lock_inner(inner).creg.publish_bank(
        api,
        &generated.tool_ids,
        bank,
        generated.generation,
        generated.frontier,
        monotonic_now(),
    );
}

fn settle_bank_continuation<W: Write>(
    cfg: &ServerConfig,
    job: &PreparedJob,
    result: Result<GenerateOutcome, GenerateError>,
    out: &mut W,
) -> Settlement {
    match result {
        Err(GenerateError::Unsupported(_)) => write_continuation_conflict(cfg, job, out),
        other => settle_generation_result(cfg, job, other, out),
    }
}

struct PreparedJob {
    parsed: crate::parse::ParsedRequest,
    surface: WireSurface,
    body_bytes: u64,
    arrived_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Settle {
    Completed,
    Failed,
    Canceled,
    Shed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Settlement {
    outcome: Settle,
    shed: Option<u8>,
}

impl Settlement {
    const COMPLETED: Self = Self {
        outcome: Settle::Completed,
        shed: None,
    };
    const FAILED: Self = Self {
        outcome: Settle::Failed,
        shed: None,
    };
    const CANCELED: Self = Self {
        outcome: Settle::Canceled,
        shed: None,
    };

    fn shed(reason: u8) -> Self {
        Self {
            outcome: Settle::Shed,
            shed: Some(reason),
        }
    }

    fn transport_gone(mut self) -> Self {
        if self.outcome == Settle::Completed {
            self.outcome = Settle::Canceled;
        }
        self
    }

    fn slow_reader(mut self) -> Self {
        if self.shed.is_none() {
            self.shed = Some(SHED_SLOW_READER);
        }
        if !matches!(self.outcome, Settle::Failed | Settle::Shed) {
            self.outcome = Settle::Canceled;
        }
        self
    }
}

fn acquire_continuation_pin(
    inner: &mut ServerInner,
    parsed: &mut crate::parse::ParsedRequest,
    now: f64,
) -> Option<ContPin> {
    let pin = inner
        .creg
        .pin_live(parsed.api, parsed.live_call_ids.first()?, now)?;
    if inner.creg.pin_owner(pin) == Some(ContOwner::BatchBank) {
        parsed.live_state_bank_owned = true;
        parsed.finish_needs();
    }
    Some(pin)
}

struct BorrowedPin<'a> {
    inner: &'a Mutex<ServerInner>,
    pin: Option<ContPin>,
}

impl Drop for BorrowedPin<'_> {
    fn drop(&mut self) {
        if let Some(pin) = self.pin.take() {
            lock_inner(self.inner).creg.unpin(pin);
        }
    }
}

struct JobLease {
    inner: Arc<Mutex<ServerInner>>,
    body_bytes: u64,
    queued: bool,
    settlement: Settlement,
    pin: Option<ContPin>,
}

impl JobLease {
    fn new(inner: Arc<Mutex<ServerInner>>, body_bytes: u64, pin: Option<ContPin>) -> Self {
        Self {
            inner,
            body_bytes,
            queued: true,
            settlement: Settlement::CANCELED,
            pin,
        }
    }

    fn start(&mut self) {
        if !self.queued {
            return;
        }
        queue_unlink_head(&mut lock_inner(&self.inner).admit);
        self.queued = false;
    }
}

impl Drop for JobLease {
    fn drop(&mut self) {
        let mut g = lock_inner(&self.inner);
        if self.queued {
            queue_unlink_head(&mut g.admit);
        }
        if let Some(pin) = self.pin.take() {
            g.creg.unpin(pin);
        }
        g.admit.inflight_body_bytes = g.admit.inflight_body_bytes.saturating_sub(self.body_bytes);
        g.runtime.requests_inflight = g.runtime.requests_inflight.saturating_sub(1);
        if let Some(reason) = self.settlement.shed {
            g.metrics.record_shed(reason);
        }
        match self.settlement.outcome {
            Settle::Completed => g.runtime.requests_completed += 1,
            Settle::Failed => g.runtime.requests_failed += 1,
            Settle::Canceled => g.runtime.requests_canceled += 1,
            Settle::Shed => {}
        }
    }
}

struct ClientLease {
    inner: Arc<Mutex<ServerInner>>,
}

impl ClientLease {
    fn new(inner: Arc<Mutex<ServerInner>>) -> Self {
        lock_inner(&inner).admit.clients += 1;
        Self { inner }
    }
}

impl Drop for ClientLease {
    fn drop(&mut self) {
        let mut g = lock_inner(&self.inner);
        g.admit.clients = g.admit.clients.saturating_sub(1);
    }
}

struct OwnerJob {
    prepared: PreparedJob,
    sink: JobSink,
    done: Sender<JobLease>,
    lease: JobLease,
}

struct JobDrain {
    done: Receiver<JobLease>,
    state: Arc<SinkState>,
}

impl Drop for JobDrain {
    fn drop(&mut self) {
        self.state.cancel(false);
    }
}

#[cfg(test)]
fn owner_job(prepared: PreparedJob, lease: JobLease) -> (OwnerJob, JobDrain) {
    owner_job_with_probe(prepared, lease, None)
}

fn owner_job_with_probe(
    prepared: PreparedJob,
    lease: JobLease,
    probe: Option<TcpStream>,
) -> (OwnerJob, JobDrain) {
    let (sink, state) = job_sink_with_probe(Arc::clone(&lease.inner), probe);
    let (done_tx, done_rx) = mpsc::channel();
    (
        OwnerJob {
            prepared,
            sink,
            done: done_tx,
            lease,
        },
        JobDrain {
            done: done_rx,
            state,
        },
    )
}

pub fn handle_client(cfg: &ServerConfig, stream: &mut TcpStream) {
    let inner = Mutex::new(ServerInner::from_cfg(cfg));
    handle_client_inner(cfg, &inner, stream, None, None);
}

fn lock_inner(inner: &Mutex<ServerInner>) -> std::sync::MutexGuard<'_, ServerInner> {
    inner.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn handle_client_inner(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    stream: &mut TcpStream,
    engine: Option<&mut dyn DecodeIo>,
    cont: Option<&mut dyn ContExec>,
) {
    let Some(mut job) = prepare_client(cfg, inner, stream) else {
        return;
    };
    let _ = stream.set_nonblocking(true);
    let mut out = DirectSink(stream);
    let body_bytes = job.body_bytes;
    let have_engine = engine.is_some();
    let (verdict, pin) = {
        let mut g = lock_inner(inner);
        let verdict = enqueue(&mut g.admit, body_bytes);
        let pin = if verdict == EnqVerdict::Ok && have_engine {
            acquire_continuation_pin(&mut g, &mut job.parsed, monotonic_now())
        } else {
            None
        };
        (verdict, pin)
    };
    if let Some((reason, code, retry, msg)) = enqueue_shed_error(verdict) {
        lock_inner(inner).metrics.record_shed(reason);
        let _ = out.write_all(&wire_http_error_bytes(
            job.surface,
            code,
            msg,
            cfg.cors,
            retry,
        ));
        return;
    }
    let pin = BorrowedPin { inner, pin };
    let settlement = run_prepared(cfg, inner, &mut job, engine, cont, &mut out, None);
    if let Some(reason) = settlement.shed {
        lock_inner(inner).metrics.record_shed(reason);
    }
    drop(pin);
    enqueue_release(&mut lock_inner(inner).admit, body_bytes);
}

fn prepare_client(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    stream: &mut TcpStream,
) -> Option<PreparedJob> {
    let req = match read_http_request(stream, chunked_enabled()) {
        Some(r) => r,
        None => {
            write_all(
                stream,
                &wire_http_error_bytes(
                    WireSurface::OpenaiChat,
                    400,
                    "bad HTTP request",
                    cfg.cors,
                    None,
                ),
            );
            return None;
        }
    };
    let arrived_at = Instant::now();
    if req.method == "OPTIONS" {
        write_all(stream, &http_response_bytes(204, None, None, cfg.cors, ""));
        return None;
    }
    if req.method == "GET" && req.path == "/v1/models" {
        let body = models_list_json(
            &cfg.model_id,
            &cfg.model_name,
            cfg.ctx,
            cfg.default_tokens,
            cfg.codex_models_json.as_deref(),
        );
        write_all(
            stream,
            &http_response_bytes(200, Some("application/json"), None, cfg.cors, &body),
        );
        return None;
    }
    if req.method == "GET" && req.path.starts_with("/v1/models/") {
        let id = &req.path["/v1/models/".len()..];
        if model_id_known(&cfg.model_id, id) {
            let body = model_one_json(id, &cfg.model_name, cfg.ctx, cfg.default_tokens);
            write_all(
                stream,
                &http_response_bytes(200, Some("application/json"), None, cfg.cors, &body),
            );
            return None;
        }
    }
    if req.method == "GET" && req.path == "/metrics" {
        let body = {
            let g = lock_inner(inner);
            let rt = g.render_runtime(unix_now() as u64);
            render_metrics(&g.metrics, &g.admit, &rt)
        };
        write_all(
            stream,
            &http_response_bytes(
                200,
                Some("text/plain; version=0.0.4"),
                None,
                cfg.cors,
                &body,
            ),
        );
        return None;
    }
    if req.method == "GET" && req.path == "/v1/stats" {
        let body = {
            let g = lock_inner(inner);
            let rt = g.render_runtime(unix_now() as u64);
            render_stats_json_ex(&g.metrics, &g.admit, &rt)
        };
        write_all(
            stream,
            &http_response_bytes(200, Some("application/json"), None, cfg.cors, &body),
        );
        return None;
    }
    if req.method == "POST" {
        let inference = shed_surface_for_path(&req.path).is_some();
        let generation = inference || req.path == "/v1/batch";
        if let Some((reason, code, retry, msg)) = {
            let g = lock_inner(inner);
            preparse_shed(&g.admit, inference, generation, req.body.len() as u64)
        } {
            lock_inner(inner).metrics.record_shed(reason);
            let surf = shed_surface_for_path(&req.path).unwrap_or(WireSurface::OpenaiChat);
            write_all(
                stream,
                &wire_http_error_bytes(surf, code, msg, cfg.cors, Some(retry)),
            );
            return None;
        }
        if let Some(surf) = parse_surface_for_path(&req.path) {
            let now = monotonic_now();
            let live_ids = {
                let mut g = lock_inner(inner);
                g.creg.expire(now);
                g.creg.live_ids(api_for_surface(surf))
            };
            let env = ParseEnv {
                default_model: cfg.model_id.clone(),
                default_tokens: cfg.default_tokens,
                default_effort: ThinkMode::Low,
                default_temp: crate::parse::default_temperature(),
                live_ids,
            };
            let body = std::str::from_utf8(&req.body).unwrap_or("");
            match parse_request(surf, &env, body) {
                Err(e) => {
                    let msg = if e.is_empty() {
                        "invalid JSON request"
                    } else {
                        &e
                    };
                    write_all(
                        stream,
                        &wire_http_error_bytes(surf, 400, msg, cfg.cors, None),
                    );
                    return None;
                }
                Ok(parsed) => {
                    let body_bytes = req.body.len() as u64;
                    return Some(PreparedJob {
                        parsed,
                        surface: surf,
                        body_bytes,
                        arrived_at,
                    });
                }
            }
        }
    }
    write_all(
        stream,
        &wire_http_error_bytes(
            WireSurface::OpenaiChat,
            404,
            "unknown endpoint",
            cfg.cors,
            None,
        ),
    );
    None
}

fn run_prepared<W: TerminalSink>(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    job: &mut PreparedJob,
    engine: Option<&mut dyn DecodeIo>,
    cont: Option<&mut dyn ContExec>,
    out: &mut W,
    arrived_at: Option<Instant>,
) -> Settlement {
    let cont_gate = match (cont.as_deref(), engine.is_some()) {
        (Some(exec), true) => cont_prompt_tokens(exec, &job.parsed)
            .ok()
            .map(|(_, toks)| (toks.len() as i32, exec.seq_cap())),
        _ => None,
    };
    let (cont_tools_anthropic, cont_tools_responses) = process_cont_tools();
    let route_env = RouteEnv {
        coalesce: cont_gate.is_some(),
        have_cont: cont_gate.is_some() && cfg.continuous,
        cont_anthropic: parse_default_on(std::env::var_os("DS4_SERVER_CONT_ANTHROPIC").as_deref()),
        cont_responses: parse_default_on(std::env::var_os("DS4_SERVER_CONT_RESPONSES").as_deref()),
        cont_tools_anthropic,
        cont_tools_responses,
        seq_cap: cont_gate.map_or(cfg.ctx, |(_, cap)| cap),
        prompt_len: cont_gate.map_or(0, |(len, _)| len),
    };
    let dec = route_decide(job.parsed.needs, job.surface, &route_env);
    let id = next_job_id(&mut lock_inner(inner).admit, job.parsed.kind);
    let arrived_at = arrived_at.unwrap_or_else(Instant::now);
    let (actual_lane, settlement) = match engine {
        Some(engine) => run_engine(cfg, inner, job, &id, dec, engine, cont, out, arrived_at),
        None => {
            let ok = out
                .write_all(&wire_http_error_bytes(
                    job.surface,
                    503,
                    SERVER_SHUTTING_DOWN,
                    cfg.cors,
                    None,
                ))
                .is_ok();
            (
                dec.lane,
                if ok {
                    Settlement::COMPLETED
                } else {
                    Settlement::CANCELED
                },
            )
        }
    };
    lock_inner(inner).metrics.record_route(
        job.surface,
        actual_lane,
        dec.reason,
        job.parsed.think_mode,
    );
    settlement
}

fn run_engine<W: TerminalSink>(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    job: &mut PreparedJob,
    id: &str,
    dec: crate::route::RouteDecision,
    engine: &mut dyn DecodeIo,
    mut cont: Option<&mut dyn ContExec>,
    out: &mut W,
    arrived_at: Instant,
) -> (u8, Settlement) {
    if dec.lane == LANE_CONTINUOUS {
        if let Some(exec) = cont.as_mut() {
            if let Some(settlement) = refuse_bank_continuation(cfg, inner, job, *exec, out) {
                return (LANE_CONTINUOUS, settlement);
            }
            let store = engine.kv_store_mut();
            let mut bank_hold_retry = |bank, live| {
                lock_inner(inner)
                    .creg
                    .bank_hold_retry(bank, live, monotonic_now())
            };
            let result = exec.generate(
                &job.parsed,
                id,
                unix_now(),
                cfg.cors,
                cfg.default_tokens,
                arrived_at,
                &mut bank_hold_retry,
                store,
                out,
            );
            if let Ok(outcome) = &result {
                publish_continuous_tool_turn(inner, job.parsed.api, exec.placed_bank(), outcome);
            }
            let bank_follow = job.parsed.needs & NEED_BANK_FRONTIER != 0;
            if bank_follow || !matches!(result, Err(GenerateError::Unsupported(_))) {
                let settlement = if bank_follow {
                    settle_bank_continuation(cfg, job, result, out)
                } else {
                    settle_generation_result(cfg, job, result, out)
                };
                return (LANE_CONTINUOUS, settlement);
            }
        }
    }
    if dec.lane == LANE_STATIC {
        let tokens = match cont.as_deref() {
            Some(exec) => cont_prompt_tokens(exec, &job.parsed)
                .map(|(_, toks)| toks)
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let current = StaticJob {
            tokens: &tokens,
            max_new_tokens: job.parsed.max_tokens,
            eos: -1,
        };
        let mut detached = DetachedStatic;
        let result = match cont.as_mut() {
            Some(exec) => match exec.as_static() {
                Some(owner) => run_static_routed(owner, current),
                None => run_static_routed(&mut detached, current),
            },
            None => run_static_routed(&mut detached, current),
        };
        return (
            LANE_STATIC,
            settle_static_lane(
                cfg,
                job,
                id,
                engine,
                i32::try_from(tokens.len()).unwrap_or(i32::MAX),
                result,
                out,
            ),
        );
    }
    (
        crate::route::LANE_SERIAL,
        run_serial(cfg, inner, job, id, engine, out, arrived_at),
    )
}

fn settle_static_lane<W: Write>(
    cfg: &ServerConfig,
    job: &PreparedJob,
    id: &str,
    engine: &dyn DecodeIo,
    prompt_n: i32,
    result: Result<Vec<StaticRow>, GenerateError>,
    out: &mut W,
) -> Settlement {
    match result {
        Ok(rows) => {
            let empty = StaticRow {
                tokens: Vec::new(),
                finish: StaticFinish::Length,
            };
            let row = rows.last().unwrap_or(&empty);
            let bytes = write_static_completion(
                StaticSettle {
                    parsed: &job.parsed,
                    job_id: id,
                    created: unix_now(),
                    cors: cfg.cors,
                    default_tokens: cfg.default_tokens,
                    model_id: engine.model_id(),
                    prompt_n,
                    row,
                },
                |token| engine.token_is_stop(token),
                |token| engine.token_text(token).unwrap_or_default(),
            );
            let ok = out.write_all(&bytes).is_ok();
            if ok {
                Settlement::COMPLETED
            } else {
                Settlement::CANCELED
            }
        }
        Err(GenerateError::Engine(msg)) if msg == STATIC_WIDTH_ERR => {
            refuse_static_width(cfg, job, &msg, out)
        }
        Err(error) => settle_generation_result(cfg, job, Err(error), out),
    }
}

fn refuse_static_width<W: Write>(
    cfg: &ServerConfig,
    job: &PreparedJob,
    msg: &str,
    out: &mut W,
) -> Settlement {
    let ok = out
        .write_all(&wire_http_error_bytes(
            job.surface,
            400,
            msg,
            cfg.cors,
            None,
        ))
        .is_ok();
    if ok {
        Settlement::COMPLETED
    } else {
        Settlement::CANCELED
    }
}

fn run_serial<W: TerminalSink>(
    cfg: &ServerConfig,
    inner: &Mutex<ServerInner>,
    job: &PreparedJob,
    id: &str,
    engine: &mut dyn DecodeIo,
    out: &mut W,
    arrived_at: Instant,
) -> Settlement {
    let parsed = &job.parsed;
    let now = monotonic_now();
    let retry = {
        let mut g = lock_inner(inner);
        g.creg.serial_hold(parsed.api, &parsed.live_call_ids, now)
    };
    if let Some(retry) = retry {
        let _ = out.write_all(&wire_http_error_bytes(
            job.surface,
            503,
            "serial capacity is reserved for a live tool continuation; retry shortly",
            cfg.cors,
            Some(retry),
        ));
        return Settlement::shed(SHED_CONT_HOLD);
    }

    let generation = engine.generation();
    let pos = engine.pos();
    let resolved = {
        let mut g = lock_inner(inner);
        g.creg
            .resolve_serial(parsed.api, &parsed.live_call_ids, generation, pos, now)
    };
    let requires_live =
        parsed.anthropic_requires_live_tool_state || parsed.responses_requires_live_tool_state;
    if requires_live && !resolved {
        let ok = out
            .write_all(&wire_http_error_bytes(
                job.surface,
                409,
                continuation_conflict_msg(parsed.api),
                cfg.cors,
                None,
            ))
            .is_ok();
        return if ok {
            Settlement::COMPLETED
        } else {
            Settlement::CANCELED
        };
    }

    let live = engine.native_graph_fit(cfg.ctx).and_then(|quote| {
        serial_fit_from_native(
            quote.need_bytes,
            quote.avail_bytes,
            quote.headroom_bytes,
            0,
            quote.fail_open,
        )
    });
    let quote = resolve_serial_fit(cfg.serial_fit, live);
    match serial_reclaim_gate(quote.ask(MemFloor::from_gb(cfg.mem_floor_gb))) {
        SerialReclaimOutcome::Admit { .. } => {}
        SerialReclaimOutcome::Refuse { .. } => {
            let prompt_n = match job.parsed.prompt_text.as_deref() {
                Some(text) => engine.tokenize_text(text),
                None => engine.tokenize_rendered_chat(&[]),
            }
            .map(|toks| i32::try_from(toks.len()).unwrap_or(i32::MAX))
            .unwrap_or(0);
            let ok = out
                .write_all(&wire_http_error_bytes(
                    job.surface,
                    503,
                    &serial_capacity_refuse_msg(prompt_n),
                    cfg.cors,
                    None,
                ))
                .is_ok();
            return if ok {
                Settlement::COMPLETED
            } else {
                Settlement::CANCELED
            };
        }
    }

    lock_inner(inner).runtime.requests_serial += 1;
    let result = generate_terminal_at(
        engine,
        parsed,
        id,
        unix_now(),
        cfg.cors,
        cfg.default_tokens,
        arrived_at,
        out,
    );
    let (generated, terminal) = match result {
        Ok(result) => result,
        Err(error) => return settle_generation_result(cfg, job, Err(error), out),
    };
    let publish = matches!(parsed.api, Api::Anthropic | Api::Responses)
        && !generated.tool_ids.is_empty()
        && generated.finish != "error"
        && generated.finish != "length";
    if publish {
        return match out.commit_tool_terminal(inner, parsed.api, &generated, terminal) {
            TerminalCommit::Committed => Settlement::COMPLETED,
            TerminalCommit::GoneBeforeStart | TerminalCommit::FailedAfterStart => {
                Settlement::CANCELED
            }
            TerminalCommit::Full => Settlement::shed(SHED_SLOW_READER),
        };
    }
    lock_inner(inner).creg.demote_serial();
    if out.write_all(&terminal).is_ok() {
        Settlement::COMPLETED
    } else {
        Settlement::CANCELED
    }
}

fn settle_generation_result<W: Write>(
    cfg: &ServerConfig,
    job: &PreparedJob,
    result: Result<GenerateOutcome, GenerateError>,
    out: &mut W,
) -> Settlement {
    match result {
        Ok(_) => Settlement::COMPLETED,
        Err(GenerateError::Io) => Settlement::CANCELED,
        Err(GenerateError::Unsupported(msg)) => {
            let _ = out.write_all(&wire_http_error_bytes(
                job.surface,
                503,
                msg,
                cfg.cors,
                None,
            ));
            Settlement::FAILED
        }
        Err(GenerateError::ContinuationHold { retry_after }) => {
            let _ = out.write_all(&wire_http_error_bytes(
                job.surface,
                503,
                "batch capacity is reserved for live tool continuations; retry shortly",
                cfg.cors,
                Some(retry_after.max(1)),
            ));
            Settlement::shed(SHED_CONT_HOLD)
        }
        Err(GenerateError::Engine(msg)) => {
            let _ = out.write_all(&wire_http_error_bytes(
                job.surface,
                500,
                &msg,
                cfg.cors,
                None,
            ));
            Settlement::FAILED
        }
    }
}

pub fn listen(cfg: &ServerConfig) -> std::io::Result<TcpListener> {
    TcpListener::bind((cfg.listen_host.as_str(), cfg.listen_port))
}

fn accept_error_retryable(_kind: ErrorKind) -> bool {
    true
}

fn retry_accept(kind: ErrorKind, polling_stop: bool) {
    if kind != ErrorKind::Interrupted {
        thread::sleep(Duration::from_millis(if polling_stop { 10 } else { 1 }));
    }
}

fn stop_requested(cfg: &ServerConfig) -> bool {
    cfg.stop_requested.is_some_and(|stop| stop())
}

pub fn accept_loop(listener: TcpListener, cfg: ServerConfig) {
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let polling_stop = cfg.stop_requested.is_some();
    if polling_stop {
        if let Err(error) = listener.set_nonblocking(true) {
            eprintln!("ds4-server-rs: stop polling unavailable: {error}");
            lock_inner(&inner).admit.stopping = true;
            return;
        }
    }
    loop {
        if stop_requested(&cfg) {
            break;
        }
        let mut stream = match listener.accept() {
            Ok((stream, _)) => {
                if stop_requested(&cfg) {
                    break;
                }
                stream
            }
            Err(e) if accept_error_retryable(e.kind()) => {
                retry_accept(e.kind(), polling_stop);
                continue;
            }
            Err(_) => break,
        };
        let cfg = cfg.clone();
        let inner = Arc::clone(&inner);
        let _ = thread::Builder::new().spawn(move || {
            let _ = stream.set_nodelay(true);
            configure_client_socket(&stream);
            handle_client_inner(&cfg, &inner, &mut stream, None, None);
        });
    }
    lock_inner(&inner).admit.stopping = true;
}

fn configure_client_socket(stream: &TcpStream) {
    // Known C-parity gap: std::net::TcpStream has no safe send-buffer setter,
    // so DS4_SERVER_CLIENT_SNDBUF remains unsupported without unsafe or a dependency.
    let _ = stream.set_read_timeout(Some(CLIENT_IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(CLIENT_IO_TIMEOUT));
}

fn send_all_nonblocking(stream: &mut TcpStream, mut bytes: &[u8]) -> std::io::Result<()> {
    let mut deadline = Instant::now() + CLIENT_SEND_STALL_TIMEOUT;
    while !bytes.is_empty() {
        match stream.write(bytes) {
            Ok(0) => {
                return Err(Error::new(
                    ErrorKind::WriteZero,
                    "socket write returned zero",
                ))
            }
            Ok(n) => {
                bytes = &bytes[n..];
                deadline = Instant::now() + CLIENT_SEND_STALL_TIMEOUT;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(Error::new(ErrorKind::TimedOut, "socket send stalled"));
                }
                thread::sleep(CLIENT_POLL.min(remaining));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn client_disconnected(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let mut byte = [0u8; 1];
    let result = stream.peek(&mut byte);
    match result {
        Ok(0) => true,
        Err(e) => matches!(
            e.kind(),
            ErrorKind::BrokenPipe | ErrorKind::ConnectionReset | ErrorKind::NotConnected
        ),
        Ok(_) => false,
    }
}

fn drain_job(mut stream: TcpStream, drain: JobDrain) {
    configure_client_socket(&stream);
    loop {
        let mut buffer = drain.state.lock();
        if buffer.bytes.is_empty() && !buffer.closed && !buffer.gone {
            buffer = drain
                .state
                .ready
                .wait(buffer)
                .unwrap_or_else(|e| e.into_inner());
        }
        let have_bytes = !buffer.bytes.is_empty();
        let finished = buffer.closed || buffer.gone;
        drop(buffer);
        if have_bytes {
            if send_all_nonblocking(&mut stream, &drain.state.take()).is_err() {
                drain.state.cancel(false);
                break;
            }
        } else if finished {
            break;
        }
    }
    let Ok(mut lease) = drain.done.recv() else {
        return;
    };
    if drain.state.gone() {
        lease.settlement = lease.settlement.transport_gone();
    }
}

fn queue_client(
    cfg: ServerConfig,
    inner: Arc<Mutex<ServerInner>>,
    jobs: mpsc::Sender<OwnerJob>,
    mut stream: TcpStream,
    _client: ClientLease,
) {
    let _ = stream.set_nodelay(true);
    configure_client_socket(&stream);
    let Some(mut prepared) = prepare_client(&cfg, &inner, &mut stream) else {
        return;
    };
    let surface = prepared.surface;
    let _ = stream.set_nonblocking(true);
    let probe = match stream.try_clone() {
        Ok(probe) => Some(probe),
        Err(_) => {
            let _ = send_all_nonblocking(
                &mut stream,
                &wire_http_error_bytes(surface, 503, SERVER_SHUTTING_DOWN, cfg.cors, None),
            );
            return;
        }
    };
    let body_bytes = prepared.body_bytes;
    let mut g = lock_inner(&inner);
    let verdict = enqueue(&mut g.admit, body_bytes);
    if let Some((reason, code, retry, msg)) = enqueue_shed_error(verdict) {
        g.metrics.record_shed(reason);
        drop(g);
        let _ = send_all_nonblocking(
            &mut stream,
            &wire_http_error_bytes(surface, code, msg, cfg.cors, retry),
        );
        return;
    }
    let pin = acquire_continuation_pin(&mut g, &mut prepared.parsed, monotonic_now());
    g.runtime.requests_started += 1;
    g.runtime.requests_inflight += 1;
    let lease = JobLease::new(Arc::clone(&inner), body_bytes, pin);
    let (job, drain) = owner_job_with_probe(prepared, lease, probe);
    let sent = jobs.send(job);
    drop(g);
    if let Err(err) = sent {
        let mut job = err.0;
        let wrote = send_all_nonblocking(
            &mut stream,
            &wire_http_error_bytes(surface, 503, SERVER_SHUTTING_DOWN, cfg.cors, None),
        )
        .is_ok();
        job.lease.settlement = if wrote {
            Settlement::FAILED
        } else {
            Settlement::CANCELED
        };
        return;
    }
    drain_job(stream, drain);
}

fn run_owner_job(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &mut dyn DecodeIo,
    cont: Option<&mut dyn ContExec>,
    job: OwnerJob,
) {
    let OwnerJob {
        mut prepared,
        mut sink,
        done,
        mut lease,
    } = job;
    lease.start();
    let queue_age = prepared.arrived_at.elapsed().as_secs_f64();
    let mut settlement = if sink.state.gone() || sink.state.observe_disconnect() {
        Settlement::CANCELED
    } else if cfg.max_queue_age_s > 0.0 && queue_age > cfg.max_queue_age_s {
        let msg = format!(
            "request waited {queue_age:.1}s in queue, over the {:.0}s limit; server overloaded, retry later",
            cfg.max_queue_age_s
        );
        let _ = sink.write_all(&wire_http_error_bytes(
            prepared.surface,
            503,
            &msg,
            cfg.cors,
            Some(30),
        ));
        Settlement::shed(SHED_QUEUE_AGE)
    } else {
        let arrived_at = prepared.arrived_at;
        run_prepared(
            cfg,
            inner,
            &mut prepared,
            Some(engine),
            cont,
            &mut sink,
            Some(arrived_at),
        )
    };
    if sink.state.slow() {
        settlement = settlement.slow_reader();
    }
    drop(sink);
    lease.settlement = settlement;
    if let Err(err) = done.send(lease) {
        let mut lease = err.0;
        lease.settlement = lease.settlement.transport_gone();
    }
}

fn owner_loop(
    listener: TcpListener,
    cfg: ServerConfig,
    engine: &mut dyn DecodeIo,
    mut cont: Option<&mut dyn ContExec>,
) {
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let polling_stop = cfg.stop_requested.is_some();
    if polling_stop {
        if let Err(error) = listener.set_nonblocking(true) {
            eprintln!("ds4-server-rs: stop polling unavailable: {error}");
            shutdown_owner(engine, cont);
            return;
        }
    }
    let (jobs_tx, jobs_rx) = mpsc::channel();
    let accept_cfg = cfg.clone();
    let accept_inner = Arc::clone(&inner);
    let accept = thread::Builder::new().spawn(move || {
        loop {
            if stop_requested(&accept_cfg) {
                break;
            }
            let stream = match listener.accept() {
                Ok((stream, _)) => {
                    if stop_requested(&accept_cfg) {
                        break;
                    }
                    stream
                }
                Err(e) if accept_error_retryable(e.kind()) => {
                    retry_accept(e.kind(), polling_stop);
                    continue;
                }
                Err(_) => break,
            };
            let cfg = accept_cfg.clone();
            let inner = Arc::clone(&accept_inner);
            let jobs = jobs_tx.clone();
            let client = ClientLease::new(Arc::clone(&inner));
            let _ = thread::Builder::new()
                .spawn(move || queue_client(cfg, inner, jobs, stream, client));
        }
        lock_inner(&accept_inner).admit.stopping = true;
    });
    if accept.is_err() {
        lock_inner(&inner).admit.stopping = true;
        shutdown_owner(engine, cont);
        return;
    }

    let mut lookahead = None;
    loop {
        let job = match lookahead.take() {
            Some(job) => job,
            None => match jobs_rx.recv() {
                Ok(job) => job,
                Err(_) => break,
            },
        };
        lookahead = match cont.as_mut() {
            Some(exec) => owner_static::run_owner_maybe_coalesce(
                &cfg,
                &inner,
                engine,
                &mut **exec,
                job,
                &jobs_rx,
            ),
            None => {
                run_owner_job(&cfg, &inner, engine, None, job);
                None
            }
        };
    }
    shutdown_owner(engine, cont);
}

fn shutdown_owner(engine: &mut dyn DecodeIo, mut cont: Option<&mut dyn ContExec>) {
    if let Err(error) = engine.shutdown() {
        eprintln!("ds4-server-rs: serial shutdown checkpoint failed: {error}");
    }
    if let Some(exec) = cont.as_deref_mut() {
        exec.shutdown(engine.kv_store_mut());
    }
}

/// Client threads read and parse; this caller remains the sole non-`Send`
/// inference owner. Continuous jobs roll: a second queued request can admit
/// while the first is generating. Static jobs may coalesce. Serial stays FIFO.
pub fn accept_loop_with_engine(
    listener: TcpListener,
    cfg: ServerConfig,
    engine: &mut dyn DecodeIo,
) {
    owner_loop(listener, cfg, engine, None);
}

/// The same owner selects continuous or serial execution after FIFO dequeue.
pub fn accept_loop_with_engine_cont(
    listener: TcpListener,
    cfg: ServerConfig,
    engine: &mut dyn DecodeIo,
    cont: &mut dyn ContExec,
) {
    owner_loop(listener, cfg, engine, Some(cont));
}

/// Test helper: unused EnqVerdict keep-alive for the admit oracle dump.
pub fn enq_verdict_name(v: EnqVerdict) -> &'static str {
    match v {
        EnqVerdict::Ok => "ok",
        EnqVerdict::Stopping => "stopping",
        EnqVerdict::ShedQueueDepth => "shed_queue_depth",
        EnqVerdict::ShedQueueBytes => "shed_queue_bytes",
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;
    use crate::generate::{generate_and_write_at, GenerateOutcome, ScriptedDecode};
    use crate::parse::{ParseEnv, ParsedRequest};
    use crate::route::WireSurface;
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static TEST_STOP: AtomicBool = AtomicBool::new(false);
    static TEST_STOP_POLLS: AtomicUsize = AtomicUsize::new(0);

    fn test_stop_requested() -> bool {
        TEST_STOP_POLLS.fetch_add(1, Ordering::Relaxed);
        TEST_STOP.load(Ordering::Relaxed)
    }

    enum TestCont {
        Accept,
        Reject,
        Fail,
        Hold(i32),
        Shutdown(Arc<Mutex<Vec<&'static str>>>),
    }

    struct SlowPrepDecode {
        sampled: bool,
        pos: i32,
        shutdown_order: Option<Arc<Mutex<Vec<&'static str>>>>,
    }

    struct DisconnectDecode {
        started: Option<Sender<()>>,
        resume: Receiver<()>,
        samples: usize,
        evals: usize,
        pos: i32,
    }

    impl DecodeIo for DisconnectDecode {
        fn model_id(&self) -> i32 {
            0
        }
        fn tokenize_text(&self, _text: &str) -> Result<Vec<i32>, GenerateError> {
            Ok(vec![1])
        }
        fn tokenize_rendered_chat(&self, _text: &[u8]) -> Result<Vec<i32>, GenerateError> {
            Ok(vec![1])
        }
        fn tokenizes_control_literals(&self) -> bool {
            false
        }
        fn token_text(&self, _token: i32) -> Result<Vec<u8>, GenerateError> {
            Ok(b"x".to_vec())
        }
        fn token_is_stop(&self, _token: i32) -> bool {
            false
        }
        fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
            self.pos = tokens.len() as i32;
            Ok(())
        }
        fn eval(&mut self, _token: i32) -> Result<(), GenerateError> {
            self.evals += 1;
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
            if let Some(started) = self.started.take() {
                started.send(()).unwrap();
                self.resume.recv_timeout(Duration::from_secs(1)).unwrap();
            }
            if self.samples >= 5 {
                return -1;
            }
            self.samples += 1;
            1
        }
        fn pos(&self) -> i32 {
            self.pos
        }
        fn ctx(&self) -> i32 {
            8192
        }
        fn generation(&self) -> u64 {
            1
        }
        fn invalidate(&mut self) {
            self.pos = 0;
        }
    }

    impl DecodeIo for SlowPrepDecode {
        fn model_id(&self) -> i32 {
            0
        }
        fn tokenize_text(&self, _text: &str) -> Result<Vec<i32>, GenerateError> {
            thread::sleep(Duration::from_millis(100));
            Ok(vec![1])
        }
        fn tokenize_rendered_chat(&self, _text: &[u8]) -> Result<Vec<i32>, GenerateError> {
            self.tokenize_text("")
        }
        fn tokenizes_control_literals(&self) -> bool {
            false
        }
        fn token_text(&self, _token: i32) -> Result<Vec<u8>, GenerateError> {
            Ok(b"x".to_vec())
        }
        fn token_is_stop(&self, _token: i32) -> bool {
            false
        }
        fn sync(&mut self, tokens: &[i32]) -> Result<(), GenerateError> {
            thread::sleep(Duration::from_millis(40));
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
            if self.sampled {
                -1
            } else {
                self.sampled = true;
                1
            }
        }
        fn pos(&self) -> i32 {
            self.pos
        }
        fn ctx(&self) -> i32 {
            8192
        }
        fn generation(&self) -> u64 {
            1
        }
        fn invalidate(&mut self) {
            self.pos = 0;
        }
        fn shutdown(&mut self) -> Result<(), GenerateError> {
            if let Some(order) = &self.shutdown_order {
                order.lock().unwrap().push("serial");
            }
            Ok(())
        }
    }

    impl ContExec for TestCont {
        fn model_id(&self) -> i32 {
            0
        }

        fn seq_cap(&self) -> i32 {
            8192
        }

        fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
            vec![1]
        }

        fn encode_text(&self, _text: &str) -> Vec<i32> {
            vec![1]
        }

        fn generate(
            &mut self,
            _parsed: &ParsedRequest,
            _job_id: &str,
            _created: i64,
            cors: bool,
            _default_tokens: i32,
            _t_arrive: Instant,
            _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
            _store: Option<&mut ds4_kv::Store>,
            out: &mut dyn Write,
        ) -> Result<GenerateOutcome, GenerateError> {
            match self {
                Self::Accept => {
                    out.write_all(&http_response_bytes(
                        200,
                        Some("application/json"),
                        None,
                        cors,
                        "{}",
                    ))
                    .map_err(|_| GenerateError::Io)?;
                    Ok(GenerateOutcome {
                        generation: 1,
                        frontier: 1,
                        finish: "stop".into(),
                        ..GenerateOutcome::default()
                    })
                }
                Self::Reject => Err(GenerateError::Unsupported("serial fallback")),
                Self::Fail => Err(GenerateError::Engine("native failure".into())),
                Self::Hold(retry_after) => Err(GenerateError::ContinuationHold {
                    retry_after: *retry_after,
                }),
                Self::Shutdown(_) => unreachable!("shutdown-only continuous lane generated"),
            }
        }

        fn shutdown(&mut self, _store: Option<&mut ds4_kv::Store>) {
            if let Self::Shutdown(order) = self {
                order.lock().unwrap().push("bank");
            }
        }
    }

    fn test_cfg() -> ServerConfig {
        let mut cfg = ServerConfig::test_cfg();
        cfg.default_tokens = 8;
        cfg
    }

    #[test]
    fn test_cfg_enables_engine_and_keeps_serial_fit_unset() {
        let cfg = ServerConfig::test_cfg();
        assert!(cfg.have_engine);
        assert!(cfg.serial_fit.is_none());
    }

    #[test]
    fn c_admission_defaults_are_preserved() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.max_clients, 256);
        assert_eq!(cfg.max_queue, 256);
        assert_eq!(cfg.max_queue_bytes, 256 * 1024 * 1024);
        assert_eq!(cfg.max_queue_age_s, 600.0);
        assert_eq!(cfg.out_agg_cap_bytes, 64 * 1024 * 1024);
        assert_eq!(cfg.out_agg_evict_min_bytes, 256 * 1024);
        assert_eq!(
            cfg.mem_floor_gb,
            crate::serve_serial_reclaim::DEFAULT_MEM_FLOOR_GB
        );
        assert!(cfg.serial_fit.is_none());
    }

    #[test]
    fn c_admission_environment_overrides_parse_without_unsafe_env_mutation() {
        let os = OsStr::new;
        assert_eq!(parse_i32_bound(Some(os("7")), 256), 7);
        assert_eq!(parse_i32_bound(Some(os(" 7jobs")), 256), 7);
        assert_eq!(parse_i32_bound(Some(os("\u{2003}7")), 256), 0);
        assert_eq!(parse_i32_bound(Some(os("-1")), 256), 0);
        assert_eq!(parse_i32_bound(Some(os("bad")), 256), 0);
        assert_eq!(parse_u64_bound(Some(os("0")), 64), 0);
        assert_eq!(parse_u64_bound(Some(os("1048576")), 64), 1_048_576);
        assert_eq!(parse_u64_bound(Some(os("1048576bytes")), 64), 1_048_576);
        assert_eq!(parse_u64_bound(Some(os("-1")), 64), u64::MAX);
        assert_eq!(
            parse_u64_bound(Some(os("18446744073709551616")), 64),
            u64::MAX
        );
        assert_eq!(parse_u64_bound(Some(os("bad")), 64), 0);
        assert_eq!(parse_f64_bound(Some(os("2.5")), 600.0), 2.5);
        assert_eq!(parse_f64_bound(Some(os(" 2.5seconds")), 600.0), 2.5);
        assert_eq!(parse_f64_bound(Some(os("0x1p2seconds")), 600.0), 4.0);
        assert!(parse_f64_bound(Some(os("nan")), 600.0).is_nan());
        assert_eq!(parse_f64_bound(Some(os("-1")), 600.0), 0.0);
        assert_eq!(parse_f64_bound(Some(os("bad")), 600.0), 0.0);
    }

    #[cfg(unix)]
    #[test]
    fn c_admission_environment_overrides_preserve_non_utf8_bytes() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        assert_eq!(parse_i32_bound(Some(OsStr::from_bytes(b"7\xff")), 256), 7);
        assert_eq!(
            parse_u64_bound(Some(OsStr::from_bytes(b"1048576\xff")), 64),
            1_048_576
        );
        assert_eq!(
            parse_f64_bound(Some(OsStr::from_bytes(b"2.5\xff")), 600.0),
            2.5
        );
        assert!(!parse_default_on(Some(OsStr::from_bytes(b"0"))));
        assert!(parse_default_on(Some(OsStr::from_bytes(b"0\xff"))));
    }

    #[test]
    fn c_disconnect_abort_environment_contract_is_preserved() {
        assert!(parse_default_on(None));
        assert!(!parse_default_on(Some(OsStr::new("0"))));
        assert!(parse_default_on(Some(OsStr::new("00"))));
        assert!(parse_default_on(Some(OsStr::new("1"))));
        assert!(parse_default_on(Some(OsStr::new("bad"))));
    }

    #[test]
    fn default_queue_bound_limits_owner_channel_metadata() {
        let mut cfg = test_cfg();
        cfg.max_clients = 0;
        assert_eq!(cfg.max_queue, 256);
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (jobs_tx, jobs_rx) = mpsc::channel();
        let body = r#"{"prompt":"queued","max_tokens":0}"#;
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut clients = Vec::with_capacity(256);
        let mut threads = Vec::with_capacity(256);

        for expected in 1..=256 {
            let mut client = TcpStream::connect(addr).unwrap();
            let (server, _) = listener.accept().unwrap();
            client.write_all(request.as_bytes()).unwrap();
            let client_cfg = cfg.clone();
            let client_inner = Arc::clone(&inner);
            let client_jobs = jobs_tx.clone();
            let lease = ClientLease::new(Arc::clone(&inner));
            threads.push(thread::spawn(move || {
                queue_client(client_cfg, client_inner, client_jobs, server, lease)
            }));
            for _ in 0..1000 {
                if lock_inner(&inner).admit.queued == expected {
                    break;
                }
                thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(lock_inner(&inner).admit.queued, expected);
            clients.push(client);
        }

        let mut refused = TcpStream::connect(addr).unwrap();
        refused
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let (server, _) = listener.accept().unwrap();
        refused.write_all(request.as_bytes()).unwrap();
        let refuse_cfg = cfg.clone();
        let refuse_inner = Arc::clone(&inner);
        let refuse_jobs = jobs_tx.clone();
        let refuse_lease = ClientLease::new(Arc::clone(&inner));
        let refuse_thread = thread::spawn(move || {
            queue_client(refuse_cfg, refuse_inner, refuse_jobs, server, refuse_lease)
        });
        let mut response = Vec::new();
        refused.read_to_end(&mut response).unwrap();
        refuse_thread.join().unwrap();
        assert!(
            String::from_utf8_lossy(&response).starts_with("HTTP/1.1 429 Too Many Requests"),
            "{}",
            String::from_utf8_lossy(&response)
        );
        assert_eq!(lock_inner(&inner).admit.queued, 256);

        let jobs: Vec<_> = jobs_rx.try_iter().collect();
        assert_eq!(
            jobs.len(),
            256,
            "unbounded channel must be admission-bounded"
        );
        drop(jobs);
        drop(clients);
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(lock_inner(&inner).admit.queued, 0);
    }

    #[test]
    fn fallback_records_actual_serial_lane() {
        let cfg = test_cfg();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let body =
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
        write!(
            client,
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[b"ok"]);
        let mut cont = TestCont::Reject;

        handle_client_inner(
            &cfg,
            &inner,
            &mut server,
            Some(&mut engine),
            Some(&mut cont),
        );
        drop(server);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();

        assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200 OK"));
        let g = inner.lock().unwrap();
        assert_eq!(g.metrics.route_requests[0][0], 1);
        assert_eq!(g.metrics.route_requests[0][1], 0);
    }

    #[test]
    fn continuous_lane_does_not_take_serial_hold() {
        let cfg = test_cfg();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        lock_inner(&inner).creg.publish_serial(
            Api::Responses,
            &["live-call".into()],
            1,
            1,
            monotonic_now(),
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let body =
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
        write!(
            client,
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[]);
        let mut cont = TestCont::Accept;

        handle_client_inner(
            &cfg,
            &inner,
            &mut server,
            Some(&mut engine),
            Some(&mut cont),
        );
        drop(server);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();

        assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200 OK"));
        let g = lock_inner(&inner);
        assert_eq!(g.metrics.route_requests[0][1], 1);
        assert_eq!(g.metrics.shed[SHED_CONT_HOLD as usize], 0);
        assert_eq!(g.runtime.requests_serial, 0);
    }

    #[test]
    fn protected_bank_saturation_sheds_with_retry_without_serial_fallback() {
        let cfg = test_cfg();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let body =
            r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#;
        write!(
            client,
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[b"serial fallback"]);
        let mut cont = TestCont::Hold(7);

        handle_client_inner(
            &cfg,
            &inner,
            &mut server,
            Some(&mut engine),
            Some(&mut cont),
        );
        drop(server);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        let response = String::from_utf8(response).unwrap();

        assert!(
            response.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{response}"
        );
        assert!(response.contains("Retry-After: 7\r\n"), "{response}");
        let (_, body) = response.split_once("\r\n\r\n").unwrap();
        assert_eq!(
            body,
            "{\"error\":{\"message\":\"batch capacity is reserved for live tool continuations; retry shortly\",\"type\":\"server_error\"}}\n"
        );
        let g = lock_inner(&inner);
        assert_eq!(g.metrics.shed[SHED_CONT_HOLD as usize], 1);
        assert_eq!(g.runtime.requests_serial, 0);
    }

    #[test]
    fn stateless_anthropic_uses_continuous_lane() {
        let cfg = test_cfg();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let body = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}"#;
        write!(
            client,
            "POST /v1/messages HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[]);
        let mut cont = TestCont::Accept;

        handle_client_inner(
            &cfg,
            &inner,
            &mut server,
            Some(&mut engine),
            Some(&mut cont),
        );
        drop(server);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();

        assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200 OK"));
        let g = lock_inner(&inner);
        assert_eq!(
            g.metrics.route_requests[WireSurface::Anthropic as usize][1],
            1
        );
        assert_eq!(g.runtime.requests_serial, 0);
    }

    #[test]
    fn stateless_responses_uses_continuous_lane() {
        let cfg = test_cfg();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let body = r#"{"input":"hi","max_output_tokens":8}"#;
        write!(
            client,
            "POST /v1/responses HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[]);
        let mut cont = TestCont::Accept;

        handle_client_inner(
            &cfg,
            &inner,
            &mut server,
            Some(&mut engine),
            Some(&mut cont),
        );
        drop(server);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();

        assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200 OK"));
        let g = lock_inner(&inner);
        assert_eq!(
            g.metrics.route_requests[WireSurface::Responses as usize][1],
            1
        );
        assert_eq!(g.runtime.requests_serial, 0);
    }

    #[test]
    fn job_sink_is_bounded_and_disconnect_aware() {
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&ServerConfig::default())));
        let (mut slow, slow_state) = job_sink(Arc::clone(&inner));
        let err = slow
            .write_all(&vec![0; JOB_SINK_CAP_BYTES as usize + 1])
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock);
        assert!(slow_state.slow());
        assert_eq!(lock_inner(&inner).runtime.out_backlog_bytes, 0);

        lock_inner(&inner).runtime.out_backlog_bytes = JOB_SINK_AGG_CAP_BYTES;
        let (mut aggregate, aggregate_state) = job_sink(Arc::clone(&inner));
        aggregate.write_all(b"x").unwrap();
        assert_eq!(aggregate_state.backlog_bytes(), 1);
        assert_eq!(
            aggregate
                .write_all(&vec![0; 256 * 1024 - 1])
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::WouldBlock
        );
        assert!(aggregate_state.slow());
        lock_inner(&inner).runtime.out_backlog_bytes = 0;

        let (mut gone, gone_state) = job_sink(Arc::clone(&inner));
        gone_state.cancel(false);
        let err = gone.write_all(b"x").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::BrokenPipe);
        assert!(gone_state.gone());
        assert!(!gone_state.slow());

        let (mut sink, state) = job_sink(Arc::clone(&inner));
        let expected: Vec<u8> = (0..100_000).map(|i| (i % 251) as u8).collect();
        for (i, byte) in expected.iter().enumerate() {
            sink.write_all(std::slice::from_ref(byte)).unwrap();
            if i == 256 {
                assert_eq!(state.backlog_bytes(), 257);
            }
        }
        assert_eq!(state.backlog_bytes(), expected.len() as u64);
        assert!(!state.slow());
        assert_eq!(
            lock_inner(&inner).runtime.out_backlog_bytes,
            expected.len() as u64
        );
        assert_eq!(state.take(), expected);
        assert_eq!(state.backlog_bytes(), 0);
        assert_eq!(lock_inner(&inner).runtime.out_backlog_bytes, 0);
    }

    fn tool_outcome(id: &str, generation: u64, frontier: i32) -> GenerateOutcome {
        GenerateOutcome {
            tool_ids: vec![id.into()],
            generation,
            frontier,
            finish: "tool_calls".into(),
        }
    }

    #[test]
    fn terminal_commit_ladder_matches_c_ordering() {
        let cfg = test_cfg();

        let clean_inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let (mut clean, clean_state) = job_sink(Arc::clone(&clean_inner));
        let clean_watch = Arc::clone(&clean_state);
        let clean_registry = Arc::clone(&clean_inner);
        let wake = thread::spawn(move || {
            let mut buffer = clean_watch.lock();
            while buffer.bytes.is_empty() {
                buffer = clean_watch
                    .ready
                    .wait(buffer)
                    .unwrap_or_else(|e| e.into_inner());
            }
            drop(buffer);
            assert!(
                lock_inner(&clean_registry).creg.live_has_id(
                    Api::Responses,
                    "call-clean",
                    monotonic_now()
                ),
                "terminal became drainable before publication"
            );
        });
        assert_eq!(
            clean.commit_tool_terminal(
                &clean_inner,
                Api::Responses,
                &tool_outcome("call-clean", 2, 20),
                b"terminal".to_vec(),
            ),
            TerminalCommit::Committed
        );
        wake.join().unwrap();
        assert_eq!(clean_state.take(), b"terminal");
        assert_eq!(lock_inner(&clean_inner).runtime.out_backlog_bytes, 0);

        let gone_inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let (mut gone, gone_state) = job_sink(Arc::clone(&gone_inner));
        gone_state.cancel(false);
        assert_eq!(
            gone.commit_tool_terminal(
                &gone_inner,
                Api::Responses,
                &tool_outcome("call-gone", 3, 30),
                b"terminal".to_vec(),
            ),
            TerminalCommit::GoneBeforeStart
        );
        {
            let mut g = lock_inner(&gone_inner);
            assert!(g.creg.id_known("call-gone"));
            assert!(!g
                .creg
                .live_has_id(Api::Responses, "call-gone", monotonic_now()));
            assert_eq!(g.runtime.out_backlog_bytes, 0);
        }

        let full_inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        {
            let mut g = lock_inner(&full_inner);
            g.creg
                .publish_serial(Api::Responses, &["call-old".into()], 1, 10, monotonic_now());
            g.runtime.out_backlog_bytes = g.out_agg_cap_bytes;
        }
        let (mut full, full_state) = job_sink(Arc::clone(&full_inner));
        assert_eq!(
            full.commit_tool_terminal(
                &full_inner,
                Api::Responses,
                &tool_outcome("call-full", 4, 40),
                b"terminal".to_vec(),
            ),
            TerminalCommit::Full
        );
        let mut g = lock_inner(&full_inner);
        assert!(!g.creg.id_known("call-full"));
        assert!(g
            .creg
            .live_has_id(Api::Responses, "call-old", monotonic_now()));
        assert!(full_state.slow());
    }

    #[test]
    fn production_terminal_probe_demotes_a_closed_tcp_client() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();
        client.shutdown(std::net::Shutdown::Both).unwrap();
        drop(client);
        let (mut sink, state) = job_sink_with_probe(Arc::clone(&inner), Some(server));

        assert_eq!(
            sink.commit_tool_terminal(
                &inner,
                Api::Responses,
                &tool_outcome("call-tcp-gone", 5, 50),
                b"terminal".to_vec(),
            ),
            TerminalCommit::GoneBeforeStart
        );
        assert!(state.gone());
        let mut g = lock_inner(&inner);
        assert!(g.creg.id_known("call-tcp-gone"));
        assert!(!g
            .creg
            .live_has_id(Api::Responses, "call-tcp-gone", monotonic_now()));
        assert_eq!(g.runtime.out_backlog_bytes, 0);
    }

    #[test]
    fn disconnect_abort_zero_disables_only_mid_generation_probe() {
        for (disconnect_abort, expect_error) in [(false, false), (true, true)] {
            let mut cfg = test_cfg();
            cfg.disconnect_abort = disconnect_abort;
            let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let (server, _) = listener.accept().unwrap();
            server.set_nonblocking(true).unwrap();
            client.shutdown(std::net::Shutdown::Both).unwrap();
            drop(client);
            let (mut sink, state) = job_sink_with_probe(Arc::clone(&inner), Some(server));

            assert_eq!(sink.flush().is_err(), expect_error);
            assert_eq!(state.gone(), expect_error);
        }
    }

    #[test]
    fn terminal_capture_preserves_anthropic_and_responses_stream_prefix() {
        let env = ParseEnv {
            default_model: "ds4".into(),
            default_tokens: 16,
            default_effort: ThinkMode::None,
            default_temp: 0.0,
            live_ids: Vec::new(),
        };
        let mut parsed = parse_request(
            WireSurface::Anthropic,
            &env,
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"thinking":{"type":"disabled"},"tools":[{"name":"bash","input_schema":{"type":"object"}}]}"#,
        )
        .unwrap();
        parsed.think_mode = ThinkMode::None;
        parsed.temperature = 0.0;
        let block = concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"bash\">\n",
            "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
            "</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>"
        );

        for (api, terminal_marker) in [
            (Api::Anthropic, "event: message_stop"),
            (Api::Responses, "\"type\":\"response.completed\""),
        ] {
            parsed.api = api;
            let arrive = Instant::now();
            let mut captured_engine = ScriptedDecode::from_pieces(&[block.as_bytes()]);
            let mut prefix = Vec::new();
            let (outcome, terminal) = generate_terminal_at(
                &mut captured_engine,
                &parsed,
                "tool-stream",
                1,
                false,
                16,
                arrive,
                &mut prefix,
            )
            .unwrap();
            assert!(!outcome.tool_ids.is_empty());
            let prefix_text = String::from_utf8_lossy(&prefix);
            let terminal_text = String::from_utf8_lossy(&terminal);
            assert!(!prefix_text.contains(terminal_marker), "{prefix_text}");
            assert!(terminal_text.contains(terminal_marker), "{terminal_text}");

            let mut full_engine = ScriptedDecode::from_pieces(&[block.as_bytes()]);
            let mut full = Vec::new();
            generate_and_write_at(
                &mut full_engine,
                &parsed,
                "tool-stream",
                1,
                false,
                16,
                arrive,
                &mut full,
            )
            .unwrap();
            let stable = [b"call_".as_slice(), b"toolu_".as_slice()]
                .iter()
                .filter_map(|needle| {
                    prefix
                        .windows(needle.len())
                        .position(|window| window == *needle)
                })
                .min()
                .unwrap_or(prefix.len());
            assert!(full.starts_with(&prefix[..stable]));
            assert!(String::from_utf8_lossy(&full).contains(terminal_marker));
        }
    }

    #[test]
    fn direct_send_failure_after_start_keeps_tool_turn_live() {
        let cfg = test_cfg();
        let inner = Mutex::new(ServerInner::from_cfg(&cfg));
        let outcome = tool_outcome("call-partial", 5, 50);
        let result =
            direct_terminal_commit(&inner, Api::Responses, &outcome, b"terminal", |action| {
                match action {
                    DirectTerminalIo::Disconnected => Ok(false),
                    DirectTerminalIo::Send(_) => {
                        Err(Error::new(ErrorKind::BrokenPipe, "late failure"))
                    }
                }
            });

        assert_eq!(result, TerminalCommit::FailedAfterStart);
        let mut g = lock_inner(&inner);
        assert!(g
            .creg
            .live_has_id(Api::Responses, "call-partial", monotonic_now()));
        assert_eq!(g.runtime.out_backlog_bytes, 0);
    }

    #[test]
    fn terminal_full_sheds_once_without_publishing_new_turn() {
        let mut cfg = test_cfg();
        cfg.out_agg_cap_bytes = 1;
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        lock_inner(&inner).runtime.out_backlog_bytes = 1;
        let env = ParseEnv {
            default_model: "ds4".into(),
            default_tokens: 16,
            default_effort: ThinkMode::None,
            default_temp: 0.0,
            live_ids: Vec::new(),
        };
        let parsed = parse_request(
            WireSurface::Anthropic,
            &env,
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"thinking":{"type":"disabled"},"tools":[{"name":"bash","input_schema":{"type":"object"}}]}"#,
        )
        .unwrap();
        let mut prepared = PreparedJob {
            parsed,
            surface: WireSurface::Anthropic,
            body_bytes: 11,
            arrived_at: Instant::now(),
        };
        let lease = admit_test_job(&inner, &mut prepared);
        let (job, drain) = owner_job(prepared, lease);
        let block = concat!(
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"bash\">\n",
            "<｜DSML｜parameter name=\"command\" string=\"true\">ls",
            "</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>"
        );
        let mut engine = ScriptedDecode::from_pieces(&[block.as_bytes()]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        assert!(settle_drain(drain).is_empty());

        let g = lock_inner(&inner);
        assert_eq!(g.metrics.shed[SHED_SLOW_READER as usize], 1);
        assert_eq!(g.runtime.requests_completed, 0);
        assert_eq!(g.runtime.requests_canceled, 0);
        assert_eq!(g.runtime.requests_failed, 0);
        assert_eq!(g.creg.n_live(), 0);
    }

    fn admit_test_job(inner: &Arc<Mutex<ServerInner>>, prepared: &mut PreparedJob) -> JobLease {
        let mut g = lock_inner(inner);
        assert_eq!(enqueue(&mut g.admit, prepared.body_bytes), EnqVerdict::Ok);
        let pin = acquire_continuation_pin(&mut g, &mut prepared.parsed, monotonic_now());
        g.runtime.requests_started += 1;
        g.runtime.requests_inflight += 1;
        drop(g);
        JobLease::new(Arc::clone(inner), prepared.body_bytes, pin)
    }

    fn queued_completion(prompt: &str, body_bytes: u64) -> PreparedJob {
        let env = ParseEnv {
            default_model: "ds4".into(),
            default_tokens: 8,
            default_effort: ThinkMode::None,
            default_temp: 0.0,
            live_ids: Vec::new(),
        };
        let body = format!(r#"{{"prompt":"{prompt}","max_tokens":0}}"#);
        let parsed = parse_request(WireSurface::OpenaiCompletion, &env, &body).unwrap();
        PreparedJob {
            parsed,
            surface: WireSurface::OpenaiCompletion,
            body_bytes,
            arrived_at: Instant::now(),
        }
    }

    fn response_number(response: &str, key: &str) -> f64 {
        let start = response.find(key).unwrap() + key.len();
        let end = response[start..]
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .map(|n| start + n)
            .unwrap();
        response[start..end].parse().unwrap()
    }

    fn settle_drain(drain: JobDrain) -> Vec<u8> {
        let bytes = drain.state.take();
        let mut lease = drain.done.recv().unwrap();
        if drain.state.gone() {
            lease.settlement = lease.settlement.transport_gone();
        }
        drop(lease);
        bytes
    }

    #[test]
    fn owner_fifo_settles_body_once() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut first = queued_completion("first", 10);
        let mut second = queued_completion("second", 20);
        let first_lease = admit_test_job(&inner, &mut first);
        let second_lease = admit_test_job(&inner, &mut second);
        let (first, first_drain) = owner_job(first, first_lease);
        let (second, second_drain) = owner_job(second, second_lease);
        let (tx, rx) = mpsc::channel();
        tx.send(first).unwrap();
        tx.send(second).unwrap();
        drop(tx);
        let mut engine = ScriptedDecode::from_pieces(&[]);

        let first = rx.recv().unwrap();
        assert_eq!(first.prepared.parsed.prompt_text.as_deref(), Some("first"));
        run_owner_job(&cfg, &inner, &mut engine, None, first);
        assert!(first_drain.state.backlog_bytes() > 0);
        {
            let g = lock_inner(&inner);
            assert_eq!(g.admit.queued, 1);
            assert_eq!(g.admit.inflight_body_bytes, 30);
            assert_eq!(g.runtime.requests_completed, 0);
            assert_eq!(g.runtime.requests_inflight, 2);
        }

        let second = rx.recv().unwrap();
        assert_eq!(
            second.prepared.parsed.prompt_text.as_deref(),
            Some("second")
        );
        run_owner_job(&cfg, &inner, &mut engine, None, second);
        let second_bytes = settle_drain(second_drain);
        assert!(String::from_utf8_lossy(&second_bytes).contains("\"id\":\"cmpl-2\""));
        {
            let g = lock_inner(&inner);
            assert_eq!(g.admit.queued, 0);
            assert_eq!(g.admit.inflight_body_bytes, 10);
            assert_eq!(g.runtime.requests_completed, 1);
            assert_eq!(g.runtime.requests_inflight, 1);
        }
        let first_bytes = settle_drain(first_drain);
        assert!(String::from_utf8_lossy(&first_bytes).contains("\"id\":\"cmpl-1\""));
        let g = lock_inner(&inner);
        assert_eq!(g.admit.queued, 0);
        assert_eq!(g.admit.inflight_body_bytes, 0);
        assert_eq!(g.runtime.requests_started, 2);
        assert_eq!(g.runtime.requests_completed, 2);
        assert_eq!(g.runtime.requests_inflight, 0);
        assert_eq!(g.runtime.requests_serial, 2);
    }

    #[test]
    fn queued_pin_releases_and_ttl_uses_execution_time() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let now = monotonic_now();
        {
            let mut g = lock_inner(&inner);
            g.creg.ttl_s = 0.01;
            g.creg
                .publish_serial(Api::Responses, &["call-1".into()], 1, 1, now);
        }
        let mut prepared = queued_completion("tool output", 19);
        prepared.surface = WireSurface::Responses;
        prepared.parsed.api = Api::Responses;
        prepared.parsed.live_call_ids = vec!["call-1".into()];
        prepared.parsed.responses_requires_live_tool_state = true;
        let lease = admit_test_job(&inner, &mut prepared);
        assert_eq!(lock_inner(&inner).creg.serial_live_hard_refs(), 1);
        thread::sleep(Duration::from_millis(20));
        let (job, drain) = owner_job(prepared, lease);
        let mut engine = ScriptedDecode::from_pieces(&[]);
        engine.pos = 1;

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let response = String::from_utf8(settle_drain(drain)).unwrap();

        assert!(response.starts_with("HTTP/1.1 409 Conflict"), "{response}");
        let g = lock_inner(&inner);
        assert_eq!(g.creg.serial_live_hard_refs(), 0);
        assert_eq!(g.admit.inflight_body_bytes, 0);
        assert_eq!(g.runtime.requests_completed, 1);
        assert_eq!(g.runtime.requests_inflight, 0);
    }

    #[test]
    fn queued_bank_pin_releases_after_failed_job() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let now = monotonic_now();
        {
            let mut g = lock_inner(&inner);
            g.creg.grace_s = 0.0;
            g.creg
                .publish_bank(Api::Responses, &["call-bank".into()], 2, 7, 100, now);
        }
        let mut prepared = queued_completion("tool output", 19);
        prepared.surface = WireSurface::Responses;
        prepared.parsed.api = Api::Responses;
        prepared.parsed.live_call_ids = vec!["call-bank".into()];
        prepared.parsed.responses_requires_live_tool_state = true;
        let lease = admit_test_job(&inner, &mut prepared);
        assert!(lock_inner(&inner)
            .creg
            .bank_protected(2, Some((7, 100)), monotonic_now()));
        let (job, drain) = owner_job(prepared, lease);
        let mut engine = ScriptedDecode::from_pieces(&[]);
        engine.pos = 100;

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let response = String::from_utf8(settle_drain(drain)).unwrap();

        assert!(response.starts_with("HTTP/1.1 409 Conflict"), "{response}");
        assert!(!lock_inner(&inner)
            .creg
            .bank_protected(2, Some((7, 100)), monotonic_now()));
    }

    #[test]
    fn continuation_hold_is_shed_only_once() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        lock_inner(&inner).creg.publish_serial(
            Api::Responses,
            &["live-call".into()],
            1,
            1,
            monotonic_now(),
        );
        let mut prepared = queued_completion("unrelated", 17);
        let lease = admit_test_job(&inner, &mut prepared);
        let (job, drain) = owner_job(prepared, lease);
        let mut engine = ScriptedDecode::from_pieces(&[]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let response = String::from_utf8(settle_drain(drain)).unwrap();

        assert!(
            response.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{response}"
        );
        let g = lock_inner(&inner);
        assert_eq!(g.metrics.shed[SHED_CONT_HOLD as usize], 1);
        assert_eq!(g.runtime.requests_completed, 0);
        assert_eq!(g.runtime.requests_failed, 0);
        assert_eq!(g.runtime.requests_canceled, 0);
        assert_eq!(g.runtime.requests_inflight, 0);
        assert_eq!(g.runtime.requests_serial, 0);
    }

    #[test]
    fn stale_queue_head_is_reaped_before_inference() {
        let mut cfg = test_cfg();
        cfg.max_queue_age_s = 0.01;
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut prepared = queued_completion("stale", 17);
        prepared.arrived_at = Instant::now() - Duration::from_millis(20);
        let lease = admit_test_job(&inner, &mut prepared);
        let (job, drain) = owner_job(prepared, lease);
        let mut engine = ScriptedDecode::from_pieces(&[]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let response = String::from_utf8(settle_drain(drain)).unwrap();

        assert!(
            response.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{response}"
        );
        assert!(response.contains("Retry-After: 30"), "{response}");
        assert_eq!(engine.pos, 0);
        let g = lock_inner(&inner);
        assert_eq!(g.metrics.shed[SHED_QUEUE_AGE as usize], 1);
        assert_eq!(g.runtime.requests_completed, 0);
        assert_eq!(g.runtime.requests_canceled, 0);
    }

    #[test]
    fn engine_failure_precedes_transport_cancel() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut prepared = queued_completion("fail", 13);
        prepared.surface = WireSurface::OpenaiChat;
        prepared.parsed.kind = crate::route::ReqKind::Chat;
        prepared.parsed.prompt_text = None;
        prepared.parsed.needs = 0;
        let lease = admit_test_job(&inner, &mut prepared);
        let (job, drain) = owner_job(prepared, lease);
        let mut engine = ScriptedDecode::from_pieces(&[]);
        let mut cont = TestCont::Fail;

        run_owner_job(&cfg, &inner, &mut engine, Some(&mut cont), job);
        drain.state.cancel(false);
        let mut lease = drain.done.recv().unwrap();
        lease.settlement = lease.settlement.transport_gone();
        assert_eq!(lease.settlement.outcome, Settle::Failed);
        drop(lease);

        let g = lock_inner(&inner);
        assert_eq!(g.runtime.requests_failed, 1);
        assert_eq!(g.runtime.requests_canceled, 0);
        assert_eq!(g.runtime.requests_inflight, 0);
        assert_eq!(g.runtime.out_backlog_bytes, 0);
    }

    #[test]
    fn queued_wait_is_included_in_serial_ttft() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut prepared = queued_completion("timed", 15);
        prepared.parsed.max_tokens = 1;
        prepared.parsed.max_tokens_set = true;
        prepared.arrived_at = Instant::now() - Duration::from_millis(100);
        let lease = admit_test_job(&inner, &mut prepared);
        let (job, drain) = owner_job(prepared, lease);
        let mut engine = ScriptedDecode::from_pieces(&[b"x"]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let response = String::from_utf8(settle_drain(drain)).unwrap();
        let ttft_ms = response_number(&response, "\"ttft_ms\":");
        assert!(ttft_ms >= 80.0, "ttft={ttft_ms} response={response}");
    }

    #[test]
    fn prefill_timer_starts_immediately_before_sync() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut prepared = queued_completion("timed", 15);
        prepared.parsed.max_tokens = 1;
        prepared.parsed.max_tokens_set = true;
        let lease = admit_test_job(&inner, &mut prepared);
        let (job, drain) = owner_job(prepared, lease);
        let mut engine = SlowPrepDecode {
            sampled: false,
            pos: 0,
            shutdown_order: None,
        };

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let response = String::from_utf8(settle_drain(drain)).unwrap();
        let prefill_tok_s = response_number(&response, "\"prefill_tok_s\":");
        assert!(
            prefill_tok_s > 15.0,
            "tokenization leaked into prefill timing={prefill_tok_s}: {response}"
        );
    }

    #[test]
    fn request_arrival_is_stamped_after_body_read() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let (jobs_tx, jobs_rx) = mpsc::channel();
        let client_lease = ClientLease::new(Arc::clone(&inner));
        let client_cfg = cfg.clone();
        let client_inner = Arc::clone(&inner);
        let h = thread::spawn(move || {
            queue_client(client_cfg, client_inner, jobs_tx, server, client_lease)
        });
        let body = r#"{"prompt":"delayed","max_tokens":1}"#;
        write!(
            client,
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .unwrap();
        thread::sleep(Duration::from_millis(120));
        client.write_all(body.as_bytes()).unwrap();
        let job = jobs_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[b"x"]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        h.join().unwrap();

        let response = String::from_utf8(response).unwrap();
        let ttft_ms = response_number(&response, "\"ttft_ms\":");
        assert!(
            ttft_ms < 80.0,
            "body read leaked into ttft={ttft_ms}: {response}"
        );
    }

    #[test]
    fn c_accept_loop_retries_errors_until_shutdown() {
        assert!(accept_error_retryable(ErrorKind::Interrupted));
        assert!(accept_error_retryable(ErrorKind::WouldBlock));
        assert!(accept_error_retryable(ErrorKind::ConnectionAborted));
        assert!(accept_error_retryable(ErrorKind::TimedOut));
        assert!(accept_error_retryable(ErrorKind::InvalidInput));
        assert!(accept_error_retryable(ErrorKind::PermissionDenied));
    }

    #[test]
    fn stop_drains_then_saves_serial_before_continuous_banks() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut cfg = test_cfg();
        cfg.stop_requested = Some(test_stop_requested);
        TEST_STOP.store(false, Ordering::Relaxed);
        TEST_STOP_POLLS.store(0, Ordering::Relaxed);
        let order = Arc::new(Mutex::new(Vec::new()));
        let thread_order = Arc::clone(&order);
        let (done_tx, done_rx) = mpsc::channel();
        let owner = thread::spawn(move || {
            let mut engine = SlowPrepDecode {
                sampled: false,
                pos: 0,
                shutdown_order: Some(Arc::clone(&thread_order)),
            };
            let mut cont = TestCont::Shutdown(thread_order);
            owner_loop(listener, cfg, &mut engine, Some(&mut cont));
            done_tx.send(()).unwrap();
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        while TEST_STOP_POLLS.load(Ordering::Relaxed) == 0 && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(TEST_STOP_POLLS.load(Ordering::Relaxed) > 0);
        TEST_STOP.store(true, Ordering::Relaxed);
        if done_rx.recv_timeout(Duration::from_secs(1)).is_err() {
            let _ = TcpStream::connect(addr);
            panic!("owner loop did not stop after the stop predicate changed");
        }
        owner.join().unwrap();

        assert_eq!(*order.lock().unwrap(), ["serial", "bank"]);
    }

    #[test]
    fn disconnect_probe_keeps_post_parse_socket_nonblocking() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();

        assert!(!client_disconnected(&server));
        let writer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            client.write_all(b"x").unwrap();
        });
        let mut byte = [0u8; 1];
        let err = server.read(&mut byte).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::WouldBlock);
        writer.join().unwrap();
    }

    #[test]
    fn direct_nonblocking_sink_retries_partial_writes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();
        let payload = vec![b'x'; 16 * 1024 * 1024];
        let expected = payload.len();
        let reader = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            let mut received = Vec::new();
            client.read_to_end(&mut received).unwrap();
            received.len()
        });

        DirectSink(&mut server).write_all(&payload).unwrap();
        server.shutdown(std::net::Shutdown::Write).unwrap();

        assert_eq!(reader.join().unwrap(), expected);
    }

    #[test]
    fn tcp_client_runs_through_owner_sink() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let (jobs_tx, jobs_rx) = mpsc::channel();
        let client_lease = ClientLease::new(Arc::clone(&inner));
        let client_cfg = cfg.clone();
        let client_inner = Arc::clone(&inner);
        let h = thread::spawn(move || {
            queue_client(client_cfg, client_inner, jobs_tx, server, client_lease)
        });
        let body = r#"{"prompt":"hello","max_tokens":0}"#;
        write!(
            client,
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let job = jobs_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let mut engine = ScriptedDecode::from_pieces(&[]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        h.join().unwrap();

        let response = String::from_utf8_lossy(&response);
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains("\"id\":\"cmpl-1\""), "{response}");
        let g = lock_inner(&inner);
        assert_eq!(g.admit.clients, 0);
        assert_eq!(g.admit.queued, 0);
        assert_eq!(g.admit.inflight_body_bytes, 0);
        assert_eq!(g.runtime.requests_completed, 1);
    }

    #[test]
    fn slow_sink_cancels_once() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut prepared = queued_completion("slow", 31);
        prepared.parsed.stream = true;
        let lease = admit_test_job(&inner, &mut prepared);
        let (mut sink, state) = job_sink(Arc::clone(&inner));
        sink.write_all(&vec![0; JOB_SINK_CAP_BYTES as usize])
            .unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let job = OwnerJob {
            lease,
            prepared,
            sink,
            done: done_tx,
        };
        let mut engine = ScriptedDecode::from_pieces(&[]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        assert!(state.slow());
        let lease = done_rx.recv().unwrap();
        assert_eq!(lease.settlement.outcome, Settle::Canceled);
        drop(lease);

        let g = lock_inner(&inner);
        assert_eq!(g.admit.inflight_body_bytes, 0);
        assert_eq!(g.runtime.requests_canceled, 1);
        assert_eq!(g.runtime.requests_inflight, 0);
        assert_eq!(g.metrics.shed[SHED_SLOW_READER as usize], 1);
    }

    #[test]
    fn disconnected_drain_cancels_and_releases_body() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut prepared = queued_completion("gone", 23);
        let lease = admit_test_job(&inner, &mut prepared);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();
        client.shutdown(std::net::Shutdown::Both).unwrap();
        drop(client);
        let (job, drain) = owner_job_with_probe(prepared, lease, Some(server));
        let mut engine = ScriptedDecode::from_pieces(&[]);

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        assert!(settle_drain(drain).is_empty());
        assert_eq!(engine.pos, 0, "disconnected queue head must not prefill");

        let g = lock_inner(&inner);
        assert_eq!(g.admit.queued, 0);
        assert_eq!(g.admit.inflight_body_bytes, 0);
        assert_eq!(g.runtime.requests_canceled, 1);
        assert_eq!(g.runtime.requests_inflight, 0);
        assert_eq!(g.metrics.shed[SHED_SLOW_READER as usize], 0);
    }

    #[test]
    fn serial_owner_aborts_after_started_client_disconnect() {
        let cfg = test_cfg();
        let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
        let mut prepared = queued_completion("gone-after-start", 23);
        prepared.parsed.max_tokens = 5;
        prepared.parsed.max_tokens_set = true;
        let lease = admit_test_job(&inner, &mut prepared);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        server.set_nonblocking(true).unwrap();
        let probe = server.try_clone().unwrap();
        let (job, drain) = owner_job_with_probe(prepared, lease, Some(probe));
        let state = Arc::clone(&drain.state);
        let drain_thread = thread::spawn(move || drain_job(server, drain));
        let (started_tx, started_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let disconnect = thread::spawn(move || {
            started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            client.shutdown(std::net::Shutdown::Both).unwrap();
            drop(client);
            resume_tx.send(()).unwrap();
        });
        let mut engine = DisconnectDecode {
            started: Some(started_tx),
            resume: resume_rx,
            samples: 0,
            evals: 0,
            pos: 0,
        };

        run_owner_job(&cfg, &inner, &mut engine, None, job);
        disconnect.join().unwrap();
        drain_thread.join().unwrap();

        assert!(state.gone(), "decode probe must observe the TCP disconnect");
        assert!(
            engine.evals <= 1,
            "decode continued for {} tokens",
            engine.evals
        );
        let g = lock_inner(&inner);
        assert_eq!(g.runtime.requests_canceled, 1);
        assert_eq!(g.runtime.requests_inflight, 0);
    }

    mod stream_disconnect;
}
