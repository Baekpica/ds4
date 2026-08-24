//! Coordinator/worker runtime: CLI strings, route search, HELLO/WORK/RESULT.

use ds4_dist::{
    build_route_plan, dispatch_eval, parse_cli, parse_layers, parse_role, prepare_engine_options,
    register_worker, resolved_layer_end, send_hello, token_span_hashes, validate_layers_for_model,
    validate_options, Coordinator, CoordinatorView, EvalOutcome, SliceExec, Worker, WorkerInfo,
    WorkOutput, WorkRequest, TOKEN_HASH_INIT,
};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn oracle() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("DS4_DIST_C_ORACLE") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/dist_c_oracle")
}

fn require_oracle() -> std::path::PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/dist_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> String {
    let out = std::process::Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run dist_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn rust_cli(args: &[&str]) -> Result<(), String> {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    let (opt, unmatched) = parse_cli(&owned)?;
    if let Some(u) = unmatched.first() {
        return Err(format!("unmatched {u}"));
    }
    validate_options(&opt)
}

fn rust_cli_layers(n_layers: u32, args: &[&str]) -> Result<(), String> {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    let (opt, unmatched) = parse_cli(&owned)?;
    if let Some(u) = unmatched.first() {
        return Err(format!("unmatched {u}"));
    }
    validate_options(&opt)?;
    validate_layers_for_model(&opt, n_layers)
}

fn assert_cli(args: &[&str]) {
    let mut cargs = vec!["cli"];
    cargs.extend_from_slice(args);
    let c = c_out(&cargs);
    let r = rust_cli(args);
    if c == "OK" {
        r.expect("rust rejected a CLI C accepts");
    } else {
        let msg = c.strip_prefix("ERROR:").unwrap_or(&c);
        let err = r.expect_err("rust accepted a CLI C rejects");
        assert_eq!(err, msg, "args={args:?}");
    }
}

#[derive(Clone)]
struct MockExec {
    model_id: u32,
    n_layers: u32,
    vocab: u32,
    ctx_size: u32,
    hidden_values: u64,
    has_output: bool,
    layer_start: u32,
    layer_end: u32,
    hidden: Vec<f32>,
    logits: Vec<f32>,
}

impl SliceExec for MockExec {
    fn model_id(&self) -> u32 {
        self.model_id
    }
    fn n_layers(&self) -> u32 {
        self.n_layers
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn ctx_size(&self) -> u32 {
        self.ctx_size
    }
    fn hidden_values(&self) -> u64 {
        self.hidden_values
    }
    fn has_output(&self) -> bool {
        self.has_output
    }
    fn layer_start(&self) -> u32 {
        self.layer_start
    }
    fn layer_end(&self) -> u32 {
        self.layer_end
    }
    fn eval(&mut self, req: &WorkRequest) -> Result<WorkOutput, String> {
        Ok(WorkOutput {
            hidden: if req.produce_hidden {
                Some(self.hidden.clone())
            } else {
                None
            },
            logits: if req.produce_logits {
                Some(self.logits.clone())
            } else {
                None
            },
        })
    }
}

fn tune(s: &TcpStream) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
}

#[test]
fn layers_and_role_match_c() {
    assert_eq!(c_out(&["layers", "0:1"]), "start=0 end=1 has_output=0");
    assert_eq!(
        c_out(&["layers", "21:output"]),
        "start=21 end=4294967295 has_output=1"
    );
    let l = parse_layers("21:output").unwrap();
    assert_eq!(l.start, 21);
    assert!(l.has_output);
    assert_eq!(resolved_layer_end(&l, 53), 52);
    assert_eq!(c_out(&["layers", "foo"]), "ERROR:expected A:B or A:output");
    assert_eq!(parse_layers("foo").unwrap_err(), "expected A:B or A:output");
    assert_eq!(c_out(&["layers", "x:1"]), "ERROR:invalid start layer in x:1");
    assert_eq!(
        parse_layers("x:1").unwrap_err(),
        "invalid start layer in x:1"
    );
    assert_eq!(
        c_out(&["layers", "3:1"]),
        "ERROR:layer range end precedes start in 3:1"
    );
    assert_eq!(c_out(&["role", "coordinator"]), "coordinator");
    assert_eq!(parse_role("coordinator"), Some(ds4_dist::Role::Coordinator));
    assert!(c_out(&["role", "boss"]).starts_with("ERROR:invalid distributed role: boss"));
    assert!(parse_role("boss").is_none());
}

#[test]
fn cli_validate_matches_c() {
    assert_cli(&[]);
    assert_cli(&["--listen", "127.0.0.1", "7000"]);
    assert_cli(&["--role", "coordinator"]);
    assert_cli(&[
        "--role",
        "coordinator",
        "--layers",
        "0:1",
        "--listen",
        "127.0.0.1",
        "7000",
    ]);
    assert_cli(&[
        "--role",
        "coordinator",
        "--layers",
        "0:1",
        "--listen",
        "127.0.0.1",
        "7000",
        "--coordinator",
        "10.0.0.1",
        "9",
    ]);
    assert_cli(&["--role", "worker", "--layers", "2:output"]);
    assert_cli(&[
        "--role",
        "worker",
        "--layers",
        "2:output",
        "--coordinator",
        "127.0.0.1",
        "7000",
    ]);
    assert_cli(&[
        "--role",
        "worker",
        "--layers",
        "2:output",
        "--coordinator",
        "127.0.0.1",
        "7000",
        "--dist-prefill-chunk",
        "128",
    ]);
    assert_cli(&["--role", "nope"]);
    assert_cli(&["--layers", "0:1:2"]);
    assert_cli(&["--dist-prefill-window", "99"]);
    assert_cli(&[
        "--role",
        "coordinator",
        "--layers",
        "0:1",
        "--listen",
        "127.0.0.1",
        "0",
    ]);
}

#[test]
fn layers_for_model_match_c() {
    let args = [
        "--role",
        "coordinator",
        "--layers",
        "2:3",
        "--listen",
        "127.0.0.1",
        "7000",
    ];
    let c = c_out(
        &[
            "layers-for-model",
            "4",
            "--role",
            "coordinator",
            "--layers",
            "2:3",
            "--listen",
            "127.0.0.1",
            "7000",
        ],
    );
    let r = rust_cli_layers(4, &args);
    assert_eq!(c, "ERROR:coordinator layer range must start at layer 0");
    assert_eq!(r.unwrap_err(), "coordinator layer range must start at layer 0");

    let past = rust_cli_layers(
        4,
        &[
            "--role",
            "worker",
            "--layers",
            "9:9",
            "--coordinator",
            "127.0.0.1",
            "1",
        ],
    );
    assert_eq!(past.unwrap_err(), "layer range starts past final model layer 3");
}

#[test]
fn replay_check_requires_coordinator() {
    let owned = vec![
        "--role".into(),
        "worker".into(),
        "--layers".into(),
        "2:output".into(),
        "--coordinator".into(),
        "127.0.0.1".into(),
        "7000".into(),
        "--dist-replay-check".into(),
    ];
    let (opt, _) = parse_cli(&owned).unwrap();
    validate_options(&opt).unwrap();
    assert_eq!(
        prepare_engine_options(&opt).unwrap_err(),
        "--dist-replay-check requires --role coordinator"
    );
}

fn worker(start: u32, end: u32, output: bool) -> WorkerInfo {
    WorkerInfo {
        peer_host: format!("10.0.0.{end}"),
        listen_port: 7000 + end,
        model_id: 3,
        quant_bits: 2,
        layer_start: start,
        layer_end: end,
        has_output: output,
        has_hidden: true,
    }
}

#[test]
fn route_plan_local_complete_is_empty() {
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 3,
        local_has_output: true,
        local_can_output_head: false,
    };
    let plan = build_route_plan(&view, &[]).unwrap();
    assert!(plan.entries.is_empty());
    assert!(plan.blob.is_empty());
}

#[test]
fn route_plan_requires_layer_zero() {
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 1,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    assert_eq!(
        build_route_plan(&view, &[]).unwrap_err(),
        "coordinator route does not start at layer 0"
    );
}

#[test]
fn route_plan_missing_layer_message() {
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    assert_eq!(
        build_route_plan(&view, &[]).unwrap_err(),
        "distributed route incomplete: missing layer 2"
    );
}

#[test]
fn route_plan_prefers_output_then_longer_end() {
    let view = CoordinatorView {
        n_layers: 6,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    let workers = vec![
        worker(2, 3, false),
        worker(2, 5, true),
        worker(2, 4, false),
    ];
    let plan = build_route_plan(&view, &workers).unwrap();
    assert_eq!(plan.entries.len(), 1);
    assert_eq!(plan.entries[0].layer_end, 5);
    assert_ne!(plan.entries[0].flags, 0);
}

#[test]
fn register_worker_replaces_stale_and_prepends() {
    let mut list = vec![worker(2, 5, true)];
    list[0].peer_host = "10.0.0.9".into();
    let neu = WorkerInfo {
        peer_host: "10.0.0.9".into(),
        listen_port: 7101,
        model_id: 3,
        quant_bits: 4,
        layer_start: 2,
        layer_end: 5,
        has_output: true,
        has_hidden: true,
    };
    register_worker(&mut list, neu.clone());
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].listen_port, 7101);
    let other = worker(2, 3, false);
    register_worker(&mut list, other);
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].layer_end, 3);
}

#[test]
fn single_hop_hello_work_logits() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let logits = vec![0.1f32, 0.2, 0.7, 0.0];
    let hidden = vec![1.0f32, 2.0, 3.0, 4.0];
    let exec = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: true,
        layer_start: 2,
        layer_end: 3,
        hidden: hidden.clone(),
        logits: logits.clone(),
    };
    let worker_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        tune(&stream);
        let mut worker = Worker::new(exec);
        let (hello, name) = worker.hello(7000, 128, "mock");
        send_hello(&mut stream, &hello, &name).unwrap();
        worker.serve(&mut stream).unwrap();
    });

    let (mut stream, _) = listener.accept().unwrap();
    tune(&stream);
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    let mut coord = Coordinator::new(view, 3, 32);
    coord.accept_hello(&mut stream, "127.0.0.1").unwrap();
    let tokens = [1i32, 2];
    let out = dispatch_eval(
        &coord,
        &mut stream,
        &tokens,
        0,
        9,
        11,
        true,
        TOKEN_HASH_INIT,
        &hidden,
    )
    .unwrap();
    match out {
        EvalOutcome::Logits(v) => assert_eq!(v, logits),
        other => panic!("expected logits, got {other:?}"),
    }
    drop(stream);
    worker_thread.join().unwrap();
}

#[test]
fn two_hop_blocking_forward() {
    let data = TcpListener::bind("127.0.0.1:0").unwrap();
    let data_port = data.local_addr().unwrap().port() as u32;
    let coord_l = TcpListener::bind("127.0.0.1:0").unwrap();
    let coord_addr = coord_l.local_addr().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel::<()>();

    let tail = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: true,
        layer_start: 2,
        layer_end: 3,
        hidden: vec![9.0, 8.0],
        logits: vec![0.0, 1.0, 0.0, 0.0],
    };
    let mid = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: false,
        layer_start: 1,
        layer_end: 1,
        hidden: vec![0.5, 1.5],
        logits: vec![],
    };

    let tail_thread = thread::spawn(move || {
        let mut hello_s = TcpStream::connect(coord_addr).unwrap();
        tune(&hello_s);
        let mut worker = Worker::new(tail);
        let (hello, name) = worker.hello(data_port, 128, "tail");
        send_hello(&mut hello_s, &hello, &name).unwrap();
        ready_tx.send(()).unwrap();
        let (mut work_s, _) = data.accept().unwrap();
        tune(&work_s);
        worker.serve(&mut work_s).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();

    let mid_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(coord_addr).unwrap();
        tune(&stream);
        let mut worker = Worker::new(mid);
        let (hello, name) = worker.hello(7100, 128, "mid");
        send_hello(&mut stream, &hello, &name).unwrap();
        worker.serve(&mut stream).unwrap();
    });

    let mut first = None;
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 0,
        local_has_output: false,
        local_can_output_head: false,
    };
    let mut coord = Coordinator::new(view, 3, 32);
    for _ in 0..2 {
        let (mut s, _) = coord_l.accept().unwrap();
        tune(&s);
        let (hello, _) = coord.accept_hello(&mut s, "127.0.0.1").unwrap();
        if hello.layer_start == 1 {
            first = Some(s);
        }
    }
    let mut stream = first.expect("mid worker stream");
    let tokens = [3i32];
    let hc = vec![0.25f32, 0.75];
    let out = dispatch_eval(
        &coord,
        &mut stream,
        &tokens,
        0,
        1,
        2,
        true,
        TOKEN_HASH_INIT,
        &hc,
    )
    .unwrap();
    match out {
        EvalOutcome::Logits(v) => assert_eq!(v, vec![0.0, 1.0, 0.0, 0.0]),
        other => panic!("expected logits, got {other:?}"),
    }
    drop(stream);
    mid_thread.join().unwrap();
    tail_thread.join().unwrap();
}

#[test]
fn worker_rejects_model_mismatch_with_c_string() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let exec = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: true,
        layer_start: 2,
        layer_end: 3,
        hidden: vec![0.0; 4],
        logits: vec![1.0; 4],
    };
    let worker_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        tune(&stream);
        let mut worker = Worker::new(exec);
        let (hello, name) = worker.hello(7000, 128, "mock");
        send_hello(&mut stream, &hello, &name).unwrap();
        worker.serve(&mut stream).unwrap();
    });
    let (mut stream, _) = listener.accept().unwrap();
    tune(&stream);
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    let mut coord = Coordinator::new(view, 99, 32);
    coord.accept_hello(&mut stream, "127.0.0.1").unwrap();
    let err = dispatch_eval(
        &coord,
        &mut stream,
        &[1],
        0,
        1,
        1,
        true,
        TOKEN_HASH_INIT,
        &[0.0, 0.0],
    )
    .unwrap_err();
    assert_eq!(err, "model id mismatch: work=99 worker=3");
    drop(stream);
    worker_thread.join().unwrap();
}

#[test]
fn prefix_hash_continues_committed_timeline() {
    let committed = [7i32, 8];
    let span = [9i32];
    let (prefix, result) = token_span_hashes(&committed, &span);
    assert_eq!(prefix, ds4_dist::token_hash_prefix(&committed));
    assert_eq!(result, ds4_dist::token_hash_update_span(prefix, &span));
    assert_ne!(prefix, TOKEN_HASH_INIT);
}

#[test]
fn usage_text_mentions_roles() {
    assert!(ds4_dist::USAGE.contains("--role ROLE"));
    assert!(ds4_dist::USAGE.contains("21:output"));
}
