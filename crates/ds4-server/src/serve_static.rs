//! Static-lane owner around `BatchCtx::generate_static` (C `n>=2`).

use crate::generate::GenerateError;

/// C `ds4_bridge_batch_ctx_generate_static` when `n` is not a legal width.
pub const STATIC_WIDTH_ERR: &str = "static batch request count is out of range";

/// C `generate_batch_jobs` admission: coalesced group only.
pub const STATIC_N_MIN: usize = 2;

/// One greedy static row. Tokens are borrowed only for the call.
#[derive(Clone, Copy)]
pub struct StaticJob<'a> {
    pub tokens: &'a [i32],
    pub max_new_tokens: i32,
    pub eos: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticRow {
    pub tokens: Vec<i32>,
}

/// Trait seam so tests can spy on `generate_static` without a GGUF.
pub trait StaticExec {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError>;
}

/// `None` means `n` is admitted at the owner boundary.
pub const fn static_width_error(n: usize) -> Option<&'static str> {
    if n < STATIC_N_MIN {
        Some(STATIC_WIDTH_ERR)
    } else {
        None
    }
}

/// Owner entry: refuse `n<2` with the C width string; otherwise call
/// [`StaticExec::generate_static`].
pub fn run_static(
    exec: &mut dyn StaticExec,
    jobs: &[StaticJob<'_>],
) -> Result<Vec<StaticRow>, GenerateError> {
    if let Some(msg) = static_width_error(jobs.len()) {
        return Err(GenerateError::Engine(msg.to_string()));
    }
    exec.generate_static(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Mutex;

    use crate::generate::{GenerateError, GenerateOutcome, ScriptedDecode};
    use crate::parse::ParsedRequest;
    use crate::route::LANE_STATIC;
    use crate::serve::{handle_client_inner, ServerConfig, ServerInner};
    use crate::serve_cont::ContExec;
    use crate::WireSurface;

    struct SpyStatic {
        calls: usize,
    }

    impl StaticExec for SpyStatic {
        fn generate_static(
            &mut self,
            jobs: &[StaticJob<'_>],
        ) -> Result<Vec<StaticRow>, GenerateError> {
            self.calls += 1;
            Ok(jobs
                .iter()
                .map(|job| StaticRow {
                    tokens: job.tokens.to_vec(),
                })
                .collect())
        }
    }

    /// `seq_cap == 0` makes `route_decide` pick `LANE_STATIC`
    /// (`REASON_STATIC_PROMPT_BOUNDS`) when coalesce is on.
    struct ForceStaticCont;

    impl ContExec for ForceStaticCont {
        fn model_id(&self) -> i32 {
            0
        }
        fn seq_cap(&self) -> i32 {
            0
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

    fn drive_static_chat() -> (Vec<u8>, ServerInner) {
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
        let mut cont = ForceStaticCont;
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
        (response, inner.into_inner().unwrap())
    }

    #[test]
    fn static_width_error_matches_c_bridge_string_when_n_lt_2() {
        // Given: C generate_batch_jobs is n>=2
        // When: owner sees n=1
        // Then: C bridge width string
        assert_eq!(static_width_error(0), Some(STATIC_WIDTH_ERR));
        assert_eq!(static_width_error(1), Some(STATIC_WIDTH_ERR));
        assert_eq!(
            STATIC_WIDTH_ERR,
            "static batch request count is out of range"
        );
    }

    #[test]
    fn static_width_admits_n_eq_2() {
        // Given: a coalesced pair
        // When: owner checks width
        // Then: admitted
        assert_eq!(static_width_error(2), None);
        assert_eq!(static_width_error(3), None);
    }

    #[test]
    fn run_static_returns_c_error_string_when_n_eq_1() {
        // Given: one fixture row
        let tokens = [10, 20];
        let jobs = [StaticJob {
            tokens: &tokens,
            max_new_tokens: 8,
            eos: 99,
        }];
        let mut spy = SpyStatic { calls: 0 };

        // When
        let err = run_static(&mut spy, &jobs).unwrap_err();

        // Then: C string, generate_static not called
        match err {
            GenerateError::Engine(msg) => assert_eq!(msg, STATIC_WIDTH_ERR),
            other => panic!("expected Engine, got {other:?}"),
        }
        assert_eq!(spy.calls, 0);
    }

    #[test]
    fn run_static_invokes_generate_static_when_n_eq_2() {
        // Given: two fixture rows (no GGUF)
        let first = [10, 20];
        let second = [30];
        let jobs = [
            StaticJob {
                tokens: &first,
                max_new_tokens: 8,
                eos: 99,
            },
            StaticJob {
                tokens: &second,
                max_new_tokens: 1,
                eos: -1,
            },
        ];
        let mut spy = SpyStatic { calls: 0 };

        // When
        let rows = run_static(&mut spy, &jobs).unwrap();

        // Then
        assert_eq!(spy.calls, 1);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].tokens, first);
        assert_eq!(rows[1].tokens, second);
    }

    #[test]
    fn static_routed_request_returns_c_width_error_not_serial() {
        // Given: route_decide picked LANE_STATIC (seq_cap=0, needs=0)
        // When: a single static-routed request is served (n=1, no coalesce yet)
        let (response, inner) = drive_static_chat();
        let text = String::from_utf8_lossy(&response);

        // Then: C width string, never the serial session path
        assert!(text.starts_with("HTTP/1.1 400 Bad Request"), "{text}");
        assert!(
            text.contains(STATIC_WIDTH_ERR),
            "missing C width string: {text}"
        );
        assert!(
            !text.contains("serial-fallback"),
            "serial decode must not run: {text}"
        );
        assert_eq!(inner.runtime.requests_serial, 0);
        assert_eq!(
            inner.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_STATIC as usize],
            1
        );
        assert_eq!(
            inner.metrics.route_requests[WireSurface::OpenaiChat as usize][0],
            0
        );
    }
}
