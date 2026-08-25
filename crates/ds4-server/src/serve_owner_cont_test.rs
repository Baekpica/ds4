use std::io::Write;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::super::*;
use super::run_owner_maybe_roll;
use crate::generate::{GenerateError, GenerateOutcome, ScriptedDecode};
use crate::parse::ParseEnv;
use crate::route::{ThinkMode, WireSurface};
use crate::serve_cont::ContExec;
use crate::serve_cont_prefill::{owner_tick_call_count, reset_owner_tick_call_count};

#[derive(Default)]
struct RollSpy {
    generate_calls: usize,
}

impl ContExec for RollSpy {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        8192
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        vec![1, 2, 3]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        vec![1, 2, 3]
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
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        self.generate_calls += 1;
        out.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}")
            .map_err(|_| GenerateError::Io)?;
        Ok(GenerateOutcome {
            generation: 1,
            frontier: 1,
            finish: "stop".into(),
            ..GenerateOutcome::default()
        })
    }
}

fn cont_chat_job(inner: &Arc<Mutex<ServerInner>>, tag: &str) -> (OwnerJob, JobDrain) {
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
fn owner_roll_pair_invokes_tick_roll_prefill() {
    reset_owner_tick_call_count();
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = cont_chat_job(&inner, "a");
    let (job_b, drain_b) = cont_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = RollSpy::default();
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_roll(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.generate_calls, 2);
    assert!(
        owner_tick_call_count() >= 1,
        "production rolling owner must call tick_roll_prefill"
    );
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
}
