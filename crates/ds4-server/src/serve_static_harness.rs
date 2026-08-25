use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

use crate::generate::{GenerateError, GenerateOutcome, ScriptedDecode};
use crate::parse::ParsedRequest;
use crate::serve::{handle_client_inner, ServerConfig, ServerInner};
use crate::serve_cont::ContExec;
use crate::serve_static::{
    static_fallback_error, CoalesceLimits, OwnedStaticJob, StaticExec, StaticFinish, StaticJob,
    StaticRow,
};

#[derive(Clone, Copy)]
pub(super) enum FailKind {
    Engine(&'static str),
    Unsupported(&'static str),
}

pub(super) struct SpyStatic {
    pub calls: usize,
    pub fallbacks: usize,
    pub ns: Vec<usize>,
    pub siblings: Vec<OwnedStaticJob>,
    pub fail: Option<FailKind>,
    pub ctx_max_seq: i32,
    pub ctx_max_tokens: i32,
    pub limits: CoalesceLimits,
}

impl Default for SpyStatic {
    fn default() -> Self {
        Self {
            calls: 0,
            fallbacks: 0,
            ns: Vec::new(),
            siblings: Vec::new(),
            fail: None,
            ctx_max_seq: i32::MAX,
            ctx_max_tokens: i32::MAX,
            limits: CoalesceLimits::UNBOUNDED,
        }
    }
}

impl StaticExec for SpyStatic {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError> {
        self.calls += 1;
        self.ns.push(jobs.len());
        match self.fail {
            Some(FailKind::Engine(msg)) => Err(GenerateError::Engine(msg.to_string())),
            Some(FailKind::Unsupported(msg)) => Err(GenerateError::Unsupported(msg)),
            None => Ok(jobs
                .iter()
                .map(|job| StaticRow {
                    tokens: job.tokens.to_vec(),
                    finish: StaticFinish::Stop,
                })
                .collect()),
        }
    }

    fn fallback_static(&mut self, err: GenerateError) -> Result<Vec<StaticRow>, GenerateError> {
        self.fallbacks += 1;
        Err(static_fallback_error(err))
    }

    fn pending_siblings(&self) -> &[OwnedStaticJob] {
        &self.siblings
    }

    fn coalesce_limits(&self) -> CoalesceLimits {
        self.limits
    }

    fn ctx_max_seq(&self) -> i32 {
        self.ctx_max_seq
    }

    fn ctx_max_tokens(&self) -> i32 {
        self.ctx_max_tokens
    }
}

pub(super) fn pair_jobs<'a>(first: &'a [i32], second: &'a [i32]) -> [StaticJob<'a>; 2] {
    [
        StaticJob {
            tokens: first,
            max_new_tokens: 8,
            eos: 99,
        },
        StaticJob {
            tokens: second,
            max_new_tokens: 1,
            eos: -1,
        },
    ]
}

pub(super) struct ForceStaticCont;

impl ContExec for ForceStaticCont {
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
        _parsed: &ParsedRequest,
        _job_id: &str,
        _created: i64,
        _cors: bool,
        _default_tokens: i32,
        _t_arrive: std::time::Instant,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut ds4_kv::Store>,
        _out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        panic!("continuous generate must not run on a static-routed request");
    }
}

impl ContExec for SpyStatic {
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
        _parsed: &ParsedRequest,
        _job_id: &str,
        _created: i64,
        _cors: bool,
        _default_tokens: i32,
        _t_arrive: std::time::Instant,
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

pub(super) fn drive_static_chat() -> (Vec<u8>, ServerInner) {
    let mut cont = ForceStaticCont;
    drive_static_chat_with(&mut cont)
}

pub(super) fn drive_static_chat_with(cont: &mut dyn ContExec) -> (Vec<u8>, ServerInner) {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"},"temperature":0}"#;
    write!(
        client,
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);
    handle_client_inner(&cfg, &inner, &mut server, Some(&mut engine), Some(cont));
    drop(server);
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    (response, inner.into_inner().unwrap())
}
