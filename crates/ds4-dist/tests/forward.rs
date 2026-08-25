//! Forward-window env parse and pending-request queue.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use ds4_dist::{
    forward_window_from, opened_forwarder_message, PendingQueue, PendingRequest, Telemetry,
    ERR_CLOSED_WHILE_RESULT, ERR_FORWARD, ERR_FORWARD_HIDDEN, ERR_INVALID_RESULT, ERR_NEXT_CLOSED,
    ERR_OOM_FORWARDER, ERR_OOM_TRACK, ERR_RELAY_THREAD, ERR_RESULT_METADATA, ERR_RESULT_TOO_LARGE,
    ERR_TELEMETRY_TOO_LARGE, FORWARD_WINDOW_DEFAULT, FORWARD_WINDOW_MAX, FORWARD_WINDOW_MIN,
};

fn pending(id: u64) -> PendingRequest {
    PendingRequest {
        request_id: id,
        telemetry: Telemetry {
            layer_start: 0,
            layer_end: 1,
            route_index: 0,
            pos0: 0,
            n_tokens: 1,
            eval_usec: 0,
            downstream_wait_usec: 0,
            forward_send_usec: 0,
            input_bytes: 0,
            output_bytes: 0,
        },
        downstream_t0: 0.0,
    }
}

#[test]
fn forward_window_matches_c() {
    assert_eq!(forward_window_from(None), FORWARD_WINDOW_DEFAULT);
    assert_eq!(forward_window_from(Some("")), FORWARD_WINDOW_DEFAULT);
    assert_eq!(forward_window_from(Some("4")), 4);
    assert_eq!(forward_window_from(Some("1")), FORWARD_WINDOW_MIN);
    assert_eq!(forward_window_from(Some("64")), FORWARD_WINDOW_MAX);
    assert_eq!(forward_window_from(Some("0")), FORWARD_WINDOW_DEFAULT);
    assert_eq!(forward_window_from(Some("65")), FORWARD_WINDOW_DEFAULT);
    assert_eq!(forward_window_from(Some("-1")), FORWARD_WINDOW_DEFAULT);
    assert_eq!(forward_window_from(Some("8 ")), FORWARD_WINDOW_DEFAULT);
    assert_eq!(forward_window_from(Some(" 8")), 8);
    assert_eq!(forward_window_from(Some("+16")), 16);
}

#[test]
fn forwarder_strings_match_c() {
    assert_eq!(
        opened_forwarder_message("10.0.0.2", 7100, 4),
        "ds4: distributed worker: opened pipelined worker-to-worker connection to 10.0.0.2:7100 (window 4)"
    );
    assert_eq!(ERR_NEXT_CLOSED, "next worker closed connection");
    assert_eq!(
        ERR_INVALID_RESULT,
        "next worker did not return valid RESULT"
    );
    assert_eq!(
        ERR_CLOSED_WHILE_RESULT,
        "next worker closed while returning RESULT"
    );
    assert_eq!(ERR_RESULT_METADATA, "next worker RESULT metadata mismatch");
    assert_eq!(
        ERR_TELEMETRY_TOO_LARGE,
        "distributed telemetry chain is too large"
    );
    assert_eq!(
        ERR_RESULT_TOO_LARGE,
        "distributed RESULT frame is too large"
    );
    assert_eq!(
        ERR_OOM_FORWARDER,
        "out of memory creating worker-to-worker forwarder"
    );
    assert_eq!(
        ERR_RELAY_THREAD,
        "failed to start worker-to-worker relay thread"
    );
    assert_eq!(ERR_OOM_TRACK, "out of memory tracking forwarded request");
    assert_eq!(ERR_FORWARD, "failed to forward distributed work");
    assert_eq!(ERR_FORWARD_HIDDEN, "invalid forwarded hidden-state size");
}

#[test]
fn pending_queue_blocks_at_window_and_pop_unblocks() {
    let q = PendingQueue::new(2);
    assert_eq!(q.depth(), 2);
    assert!(q.enqueue(pending(1)));
    assert!(q.enqueue(pending(2)));
    let (tx, rx) = mpsc::channel();
    thread::scope(|s| {
        s.spawn(|| {
            assert!(q.enqueue(pending(3)));
            tx.send(()).unwrap();
        });
        thread::sleep(Duration::from_millis(30));
        assert!(rx.try_recv().is_err());
        assert_eq!(q.pop().unwrap().request_id, 1);
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
    });
    assert_eq!(q.pop().unwrap().request_id, 2);
    assert_eq!(q.pop().unwrap().request_id, 3);
    assert!(q.pop().is_none());
}

#[test]
fn pending_queue_remove_and_note_send_done() {
    let q = PendingQueue::new(4);
    assert!(q.enqueue(pending(9)));
    assert!(q.enqueue(pending(10)));
    q.note_send_done(10, 42, 1.5);
    assert!(q.remove(9));
    let got = q.pop().unwrap();
    assert_eq!(got.request_id, 10);
    assert_eq!(got.telemetry.forward_send_usec, 42);
    assert_eq!(got.downstream_t0, 1.5);
    assert!(!q.remove(9));
}

#[test]
fn pending_queue_close_rejects_enqueue() {
    let q = PendingQueue::new(2);
    q.close();
    assert!(!q.enqueue(pending(1)));
    q.clear();
    assert!(q.pop().is_none());
}
