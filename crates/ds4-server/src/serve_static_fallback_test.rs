use super::harness::{drive_static_chat_with, pair_jobs, FailKind, SpyStatic};
use super::*;
use crate::generate::GenerateError;

#[test]
fn run_static_takes_fallback_when_generate_static_returns_err() {
    // Given: n=2 and generate_static returns a C bridge error
    let first = [10, 20];
    let second = [30];
    let jobs = pair_jobs(&first, &second);
    let mut spy = SpyStatic {
        fail: Some(FailKind::Engine("static batch native result is invalid")),
        ..Default::default()
    };

    // When
    let err = run_static(&mut spy, &jobs).unwrap_err();

    // Then: fallback path, same C text, no new envelope
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.fallbacks, 1);
    match err {
        GenerateError::Engine(msg) => {
            assert_eq!(msg, "static batch native result is invalid");
        }
        other => panic!("expected Engine, got {other:?}"),
    }
}

#[test]
fn run_static_fallback_uses_c_oom_text_when_generate_static_has_no_engine_msg() {
    // Given: n=2 and generate_static returns a non-Engine miss
    let first = [10, 20];
    let second = [30];
    let jobs = pair_jobs(&first, &second);
    let mut spy = SpyStatic {
        fail: Some(FailKind::Unsupported("static owner is not attached")),
        ..Default::default()
    };

    // When
    let err = run_static(&mut spy, &jobs).unwrap_err();

    // Then: C generate_batch_jobs empty-err default
    assert_eq!(spy.fallbacks, 1);
    match err {
        GenerateError::Engine(msg) => assert_eq!(msg, STATIC_FALLBACK_ERR),
        other => panic!("expected Engine, got {other:?}"),
    }
    assert_eq!(STATIC_FALLBACK_ERR, "out of memory");
}

#[test]
fn static_routed_request_uses_c_error_not_serial_when_generate_static_fails() {
    // Given: LANE_STATIC n=2 and generate_static fails
    let mut spy = SpyStatic {
        siblings: vec![OwnedStaticJob {
            tokens: vec![30],
            max_new_tokens: 1,
            eos: -1,
        }],
        fail: Some(FailKind::Engine("static batch native result is invalid")),
        ..Default::default()
    };

    // When
    let (response, inner) = drive_static_chat_with(&mut spy);
    let text = String::from_utf8_lossy(&response);

    // Then: fallback taken, C text, serial stays cold
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.fallbacks, 1);
    assert!(text.starts_with("HTTP/1.1 500"), "{text}");
    assert!(
        text.contains("static batch native result is invalid"),
        "missing C error text: {text}"
    );
    assert!(
        !text.contains("serial-fallback"),
        "serial decode must not run: {text}"
    );
    assert_eq!(inner.runtime.requests_serial, 0);
}
