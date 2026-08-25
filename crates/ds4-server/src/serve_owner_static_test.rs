use std::io::Write;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::super::*;
use super::run_owner_maybe_coalesce;
use crate::generate::{GenerateError, GenerateOutcome, ScriptedDecode};
use crate::parse::ParseEnv;
use crate::route::{ThinkMode, WireSurface, LANE_STATIC};
use crate::serve_cont::ContExec;
use crate::serve_static::{
    static_fallback_error, CoalesceLimits, StaticExec, StaticFinish, StaticJob, StaticRow,
};

#[derive(Default)]
struct OwnerSpy {
    calls: usize,
    ns: Vec<usize>,
}

impl StaticExec for OwnerSpy {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError> {
        self.calls += 1;
        self.ns.push(jobs.len());
        Ok(jobs
            .iter()
            .map(|job| StaticRow {
                tokens: job.tokens.to_vec(),
                finish: StaticFinish::Stop,
            })
            .collect())
    }

    fn fallback_static(&mut self, err: GenerateError) -> Result<Vec<StaticRow>, GenerateError> {
        Err(static_fallback_error(err))
    }
}

impl ContExec for OwnerSpy {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        0
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        vec![77]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        vec![77]
    }
    fn generate(
        &mut self,
        _parsed: &crate::parse::ParsedRequest,
        _job_id: &str,
        _created: i64,
        _cors: bool,
        _default_tokens: i32,
        _t_arrive: Instant,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut ds4_kv::Store>,
        _out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        panic!("continuous generate must not run on a static-routed request");
    }
    fn as_static(&mut self) -> Option<&mut dyn StaticExec> {
        Some(self)
    }
}

fn static_chat_job(inner: &Arc<Mutex<ServerInner>>, tag: &str) -> (OwnerJob, JobDrain) {
    let env = ParseEnv {
        default_model: "ds4".into(),
        default_tokens: 8,
        default_effort: ThinkMode::None,
        default_temp: 0.0,
        live_ids: Vec::new(),
    };
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"{tag}"}}],"thinking":{{"type":"disabled"}},"temperature":0}}"#
    );
    let parsed = crate::parse::parse_request(WireSurface::OpenaiChat, &env, &body).unwrap();
    let prepared = PreparedJob {
        parsed,
        surface: WireSurface::OpenaiChat,
        body_bytes: body.len() as u64,
        arrived_at: Instant::now(),
    };
    let mut g = lock_inner(inner);
    assert_eq!(enqueue(&mut g.admit, prepared.body_bytes), EnqVerdict::Ok);
    g.runtime.requests_started += 1;
    g.runtime.requests_inflight += 1;
    drop(g);
    let lease = JobLease::new(Arc::clone(inner), prepared.body_bytes, None);
    owner_job(prepared, lease)
}

#[test]
fn two_queued_static_jobs_coalesce_on_owner_fifo() {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = static_chat_job(&inner, "a");
    let (job_b, drain_b) = static_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = OwnerSpy::default();
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.ns, vec![2]);
    let text_a = String::from_utf8(drain_a.state.take()).unwrap();
    let text_b = String::from_utf8(drain_b.state.take()).unwrap();
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
    assert!(text_a.starts_with("HTTP/1.1 200 OK"), "{text_a}");
    assert!(text_b.starts_with("HTTP/1.1 200 OK"), "{text_b}");
    assert!(!text_a.contains("serial-fallback"), "{text_a}");
    assert!(!text_b.contains("serial-fallback"), "{text_b}");
    let g = lock_inner(&inner);
    assert_eq!(g.runtime.requests_serial, 0);
    assert_eq!(
        g.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_STATIC as usize],
        2
    );
}

#[test]
fn owner_ctx_overflow_uses_c_fallback_not_serial() {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = static_chat_job(&inner, "a");
    let (job_b, drain_b) = static_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = OverflowSpy {
        inner: OwnerSpy::default(),
    };
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.inner.calls, 0);
    let text_a = String::from_utf8(drain_a.state.take()).unwrap();
    let text_b = String::from_utf8(drain_b.state.take()).unwrap();
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
    assert!(text_a.contains("out of memory"), "{text_a}");
    assert!(text_b.contains("out of memory"), "{text_b}");
    assert!(!text_a.contains("serial-fallback"), "{text_a}");
    assert_eq!(lock_inner(&inner).runtime.requests_serial, 0);
}

struct OverflowSpy {
    inner: OwnerSpy,
}

impl StaticExec for OverflowSpy {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError> {
        self.inner.generate_static(jobs)
    }
    fn fallback_static(&mut self, err: GenerateError) -> Result<Vec<StaticRow>, GenerateError> {
        self.inner.fallback_static(err)
    }
    fn ctx_max_seq(&self) -> i32 {
        1
    }
    fn coalesce_limits(&self) -> CoalesceLimits {
        CoalesceLimits::UNBOUNDED
    }
}

impl ContExec for OverflowSpy {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        0
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        vec![77]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        vec![77]
    }
    fn generate(
        &mut self,
        parsed: &crate::parse::ParsedRequest,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        t_arrive: Instant,
        bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        store: Option<&mut ds4_kv::Store>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        self.inner.generate(
            parsed,
            job_id,
            created,
            cors,
            default_tokens,
            t_arrive,
            bank_hold_retry,
            store,
            out,
        )
    }
    fn as_static(&mut self) -> Option<&mut dyn StaticExec> {
        Some(self)
    }
}
