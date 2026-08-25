//! C↔Rust enqueue / job-id / host-metrics fragment.

use ds4_server::{
    enqueue, mint_job_id, preparse_shed, render_metrics_fragment, AdmitState, EnqVerdict, ReqKind,
    RouteMetrics, SHED_CLIENTS, SHED_QUEUE_BYTES, SHED_QUEUE_DEPTH,
};

use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_ADMIT_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/admit_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/admit_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_str(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run admit_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn rust_enqueue(
    stopping: i32,
    queued: i32,
    max_queue: i32,
    inflight: u64,
    max_bytes: u64,
    body: u64,
) -> (EnqVerdict, i32, u64) {
    let mut s = AdmitState {
        stopping: stopping != 0,
        queued,
        max_queue,
        inflight_body_bytes: inflight,
        max_queue_bytes: max_bytes,
        ..AdmitState::default()
    };
    let v = enqueue(&mut s, body);
    (v, s.queued, s.inflight_body_bytes)
}

fn verdict_name(v: EnqVerdict) -> &'static str {
    match v {
        EnqVerdict::Ok => "ok",
        EnqVerdict::Stopping => "stopping",
        EnqVerdict::ShedQueueDepth => "shed_queue_depth",
        EnqVerdict::ShedQueueBytes => "shed_queue_bytes",
    }
}

#[test]
fn enqueue_verdicts_match_c() {
    let rows = [
        (0, 0, 0, 0u64, 0u64, 10u64),
        (1, 0, 8, 0, 0, 10),
        (0, 8, 8, 0, 0, 10),
        (0, 2, 8, 90, 100, 20),
        (0, 2, 8, 90, 100, 5),
        (0, 0, 1, 0, 0, 1),
    ];
    for (st, q, mq, inf, mb, body) in rows {
        let (v, nq, ninf) = rust_enqueue(st, q, mq, inf, mb, body);
        let c = c_str(&[
            "enqueue",
            &st.to_string(),
            &q.to_string(),
            &mq.to_string(),
            &inf.to_string(),
            &mb.to_string(),
            &body.to_string(),
        ]);
        let rust = format!("{} queued={} inflight={}\n", verdict_name(v), nq, ninf);
        assert_eq!(
            rust, c,
            "enqueue st={st} q={q} mq={mq} inf={inf} mb={mb} body={body}"
        );
    }
}

#[test]
fn mint_job_id_matches_c() {
    assert_eq!(
        mint_job_id(ReqKind::Chat, 1),
        c_str(&["mint", "0", "1"]).trim()
    );
    assert_eq!(
        mint_job_id(ReqKind::Completion, 42),
        c_str(&["mint", "1", "42"]).trim()
    );
    assert_eq!(mint_job_id(ReqKind::Chat, 1), "chatcmpl-1");
    assert_eq!(mint_job_id(ReqKind::Completion, 7), "cmpl-7");
}

#[test]
fn preparse_shed_matches_c() {
    let mut s = AdmitState {
        max_clients: 2,
        clients: 3,
        ..AdmitState::default()
    };
    let rust = preparse_shed(&s, true, true, 10).unwrap();
    let c = c_str(&["preparse", "2", "3", "0", "0", "0", "0", "10", "1", "1"]);
    assert_eq!(
        format!("shed {} {} {} {}\n", "clients", rust.1, rust.2, rust.3),
        c
    );
    assert_eq!(rust.0, SHED_CLIENTS);

    s = AdmitState {
        max_queue: 1,
        queued: 1,
        ..AdmitState::default()
    };
    let rust = preparse_shed(&s, true, true, 10).unwrap();
    let c = c_str(&["preparse", "0", "0", "1", "1", "0", "0", "10", "1", "1"]);
    assert_eq!(
        format!("shed {} {} {} {}\n", "queue_depth", rust.1, rust.2, rust.3),
        c
    );
    assert_eq!(rust.0, SHED_QUEUE_DEPTH);

    s = AdmitState {
        max_queue_bytes: 10,
        inflight_body_bytes: 8,
        ..AdmitState::default()
    };
    let rust = preparse_shed(&s, true, true, 8).unwrap();
    let c = c_str(&["preparse", "0", "0", "0", "0", "10", "8", "8", "1", "1"]);
    assert_eq!(
        format!("shed {} {} {} {}\n", "queue_bytes", rust.1, rust.2, rust.3),
        c
    );
    assert_eq!(rust.0, SHED_QUEUE_BYTES);

    s = AdmitState::default();
    assert!(preparse_shed(&s, true, true, 100).is_none());
    assert_eq!(
        c_str(&["preparse", "0", "0", "0", "0", "0", "0", "100", "1", "1"]),
        "ok\n"
    );
}

#[test]
fn metrics_zero_fragment_matches_c() {
    let rust = render_metrics_fragment(&RouteMetrics::default(), &AdmitState::default());
    let c = c_str(&["metrics-zero"]);
    assert_eq!(rust, c);
}
