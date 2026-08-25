//! Prefetch depth, job queue, reconnect strings, and pipelined serve.

use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use ds4_dist::{
    cleared_sessions_message, connect_retryable, connected_message, disconnected_message,
    hello_failed_message, prefetch_depth_from, prefetch_disabled_from, prefetch_enabled_message,
    reconnect_with, retrying_message, send_hello, JobQueue, SliceExec, WorkOutput, WorkRequest,
    Worker, ERR_OOM_QUEUE, ERR_OOM_READ, PREFETCH_DEPTH_DEFAULT, PREFETCH_DEPTH_MAX,
    PREFETCH_DEPTH_MIN,
};

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

fn mock() -> MockExec {
    MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: true,
        layer_start: 2,
        layer_end: 3,
        hidden: vec![1.0, 2.0],
        logits: vec![0.1, 0.2, 0.7, 0.0],
    }
}

fn tune(s: &TcpStream) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
}

#[test]
fn prefetch_depth_matches_c() {
    assert_eq!(prefetch_depth_from(None), PREFETCH_DEPTH_DEFAULT);
    assert_eq!(prefetch_depth_from(Some("")), PREFETCH_DEPTH_DEFAULT);
    assert_eq!(prefetch_depth_from(Some("2")), 2);
    assert_eq!(prefetch_depth_from(Some("1")), PREFETCH_DEPTH_MIN);
    assert_eq!(prefetch_depth_from(Some("8")), PREFETCH_DEPTH_MAX);
    assert_eq!(prefetch_depth_from(Some("0")), PREFETCH_DEPTH_DEFAULT);
    assert_eq!(prefetch_depth_from(Some("9")), PREFETCH_DEPTH_DEFAULT);
    assert_eq!(prefetch_depth_from(Some("-1")), PREFETCH_DEPTH_DEFAULT);
    assert_eq!(prefetch_depth_from(Some("3 ")), PREFETCH_DEPTH_DEFAULT);
    assert_eq!(prefetch_depth_from(Some(" 4")), 4);
    assert_eq!(prefetch_depth_from(Some("+5")), 5);
    assert_eq!(prefetch_depth_from(Some("2foo")), PREFETCH_DEPTH_DEFAULT);
}

#[test]
fn prefetch_disable_is_presence() {
    assert!(!prefetch_disabled_from(None));
    assert!(prefetch_disabled_from(Some(std::ffi::OsStr::new(""))));
    assert!(prefetch_disabled_from(Some(std::ffi::OsStr::new("1"))));
}

#[test]
fn prefetch_and_oom_strings_match_c() {
    assert_eq!(
        prefetch_enabled_message(2),
        "ds4: distributed worker: receive prefetch depth 2 enabled"
    );
    assert_eq!(ERR_OOM_QUEUE, "out of memory queueing distributed WORK");
    assert_eq!(ERR_OOM_READ, "out of memory reading distributed WORK frame");
}

#[test]
fn job_queue_blocks_at_depth_and_finish_drains() {
    let q = JobQueue::new(2);
    assert_eq!(q.depth(), 2);
    assert!(q.enqueue(vec![1]));
    assert!(q.enqueue(vec![2]));
    let (tx, rx) = mpsc::channel();
    thread::scope(|s| {
        s.spawn(|| {
            assert!(q.enqueue(vec![3]));
            tx.send(()).unwrap();
        });
        thread::sleep(Duration::from_millis(30));
        assert!(rx.try_recv().is_err());
        assert_eq!(q.pop(), Some(vec![1]));
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    });
    assert_eq!(q.pop(), Some(vec![2]));
    q.finish();
    assert_eq!(q.pop(), Some(vec![3]));
    assert_eq!(q.pop(), None);
}

#[test]
fn job_queue_cancel_drops_queued() {
    let q = JobQueue::new(2);
    assert!(q.enqueue(vec![9]));
    q.cancel();
    assert!(!q.enqueue(vec![8]));
    assert_eq!(q.pop(), None);
}

#[test]
fn reconnect_messages_match_c() {
    assert_eq!(
        retrying_message("unable to connect to 127.0.0.1:9: Connection refused"),
        "ds4: distributed worker: unable to connect to 127.0.0.1:9: Connection refused; retrying"
    );
    assert_eq!(
        connected_message("127.0.0.1", "7000"),
        "ds4: distributed worker: connected to coordinator 127.0.0.1:7000"
    );
    let err = io::Error::from_raw_os_error(32);
    assert!(
        hello_failed_message(&err).starts_with("ds4: distributed worker: failed to send HELLO:")
    );
    assert_eq!(
        cleared_sessions_message(3),
        "ds4: distributed worker: cleared 3 sessions after coordinator disconnect"
    );
    assert_eq!(
        disconnected_message(false),
        "ds4: distributed worker: coordinator disconnected; reconnecting"
    );
    assert_eq!(
        disconnected_message(true),
        "ds4: distributed worker: coordinator disconnected after error; reconnecting"
    );
}

#[test]
fn connect_retryable_matches_c_errno_set() {
    assert!(connect_retryable(&io::Error::from_raw_os_error(111))); // ECONNREFUSED
    assert!(connect_retryable(&io::Error::from_raw_os_error(113))); // EHOSTUNREACH
    assert!(connect_retryable(&io::Error::from_raw_os_error(101))); // ENETUNREACH
    assert!(connect_retryable(&io::Error::from_raw_os_error(110))); // ETIMEDOUT
    assert!(connect_retryable(&io::Error::from_raw_os_error(99))); // EADDRNOTAVAIL
    assert!(!connect_retryable(&io::Error::from_raw_os_error(13))); // EACCES
}

#[test]
fn clear_sessions_returns_count() {
    let mut worker = Worker::new(mock());
    assert_eq!(worker.session_count(), 0);
    assert_eq!(worker.clear_sessions(), 0);
}

#[test]
fn reconnect_retries_then_serves_and_clears() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let attempts = AtomicU32::new(0);
    let (ready_tx, ready_rx) = mpsc::channel();
    let worker_thread = thread::spawn(move || {
        let mut worker = Worker::new(mock());
        let (hello, name) = worker.hello(7000, 128, "mock");
        let mut sleeps = 0u32;
        reconnect_with(
            &mut worker,
            || {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(io::Error::new(io::ErrorKind::ConnectionRefused, "refused"))
                } else {
                    let s = TcpStream::connect(addr)?;
                    tune(&s);
                    Ok(s)
                }
            },
            &hello,
            &name,
            || {
                sleeps += 1;
            },
            || attempts.load(Ordering::SeqCst) > 1 && ready_rx.try_recv().is_ok(),
            false,
        )
        .unwrap();
        assert_eq!(worker.session_count(), 0);
        assert!(sleeps >= 1);
    });

    let (mut stream, _) = listener.accept().unwrap();
    tune(&stream);
    let _ = ds4_dist::recv_hello(&mut stream).unwrap();
    ready_tx.send(()).unwrap();
    drop(stream);
    worker_thread.join().unwrap();
}

#[test]
fn serve_prefetch_hello_then_eof() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let worker_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        tune(&stream);
        let mut worker = Worker::new(mock());
        let (hello, name) = worker.hello(7000, 128, "mock");
        send_hello(&mut stream, &hello, &name).unwrap();
        worker.serve_prefetch(&mut stream).unwrap();
        assert_eq!(worker.clear_sessions(), 0);
    });
    let (mut stream, _) = listener.accept().unwrap();
    tune(&stream);
    let (hello, name) = ds4_dist::recv_hello(&mut stream).unwrap();
    assert_eq!(hello.model_id, 3);
    assert_eq!(name, "mock");
    drop(stream);
    worker_thread.join().unwrap();
}
