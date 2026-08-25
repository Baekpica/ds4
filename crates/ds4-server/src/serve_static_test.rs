use super::harness::{drive_static_chat, drive_static_chat_with, pair_jobs, SpyStatic};
use super::*;
use crate::generate::GenerateError;
use crate::route::LANE_STATIC;
use crate::WireSurface;

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
    let mut spy = SpyStatic::default();

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
    let jobs = pair_jobs(&first, &second);
    let mut spy = SpyStatic::default();

    // When
    let rows = run_static(&mut spy, &jobs).unwrap();

    // Then: per-row lengths stay ragged; fallback stays cold
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.fallbacks, 0);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].tokens, first);
    assert_eq!(rows[1].tokens, second);
    assert_eq!([rows[0].tokens.len(), rows[1].tokens.len()], [2, 1]);
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

#[test]
fn static_routed_request_invokes_generate_static_when_n_eq_2() {
    // Given: route_decide picked LANE_STATIC and one sibling is waiting
    let mut spy = SpyStatic {
        siblings: vec![OwnedStaticJob {
            tokens: vec![30],
            max_new_tokens: 1,
            eos: -1,
        }],
        ..Default::default()
    };

    // When: the HTTP request is served through handle_client_inner
    let (response, inner) = drive_static_chat_with(&mut spy);
    let text = String::from_utf8_lossy(&response);

    // Then: generate_static ran; serial session path stayed cold
    assert_eq!(spy.calls, 1);
    assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
    assert!(
        !text.contains(STATIC_WIDTH_ERR),
        "width refuse must not fire at n=2: {text}"
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
}
