//! C↔Rust DSV4 host-prefix codec: header + token ids.

use std::path::PathBuf;
use std::process::Command;

use ds4_core::{
    parse_prefix, payload_dump_script, payload_tail, HostPrefix, ModelFamily, SessionBackend,
    SessionLedger, HEADER_BYTES, PAYLOAD_MAGIC, PAYLOAD_VERSION,
};

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_PAYLOAD_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/payload_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/payload_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_script(name: &str) -> String {
    let out = Command::new(require_oracle())
        .arg(name)
        .output()
        .expect("run payload_c_oracle");
    assert!(
        out.status.success(),
        "oracle {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn assert_script(name: &str) {
    let c = c_script(name);
    let rust = payload_dump_script(name);
    assert_eq!(rust, c, "mismatch {name}\nrust:\n{rust}c:\n{c}");
}

#[test]
fn encode_tapes_match_c() {
    for name in [
        "encode-deepseek",
        "encode-cpu",
        "encode-solar",
        "encode-exaone",
        "encode-motif3",
        "encode-dots3",
    ] {
        assert_script(name);
    }
}

#[test]
fn inspect_and_reject_tapes_match_c() {
    for name in [
        "inspect-deepseek",
        "inspect-solar",
        "inspect-exaone",
        "inspect-motif3",
        "inspect-dots3",
        "tail-offset",
        "reject-trunc",
        "reject-magic",
        "reject-version-0",
        "reject-version-4",
    ] {
        assert_script(name);
    }
}

#[test]
fn apply_tapes_match_c() {
    assert_script("apply-deepseek");
    assert_script("apply-family-miss");
}

#[test]
fn prefix_round_trip_and_tail() {
    let hex = payload_dump_script("encode-deepseek");
    let raw: String = hex.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let mut bytes = Vec::with_capacity(raw.len() / 2);
    let b = raw.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let hi = if b[i] <= b'9' { b[i] - b'0' } else { b[i] - b'a' + 10 };
        let lo = if b[i + 1] <= b'9' {
            b[i + 1] - b'0'
        } else {
            b[i + 1] - b'a' + 10
        };
        bytes.push((hi << 4) | lo);
        i += 2;
    }
    let prefix = parse_prefix(&bytes).expect("parse encoded fixture");
    assert_eq!(prefix.fields[0], PAYLOAD_MAGIC);
    assert_eq!(prefix.version(), PAYLOAD_VERSION);
    assert_eq!(prefix.token_count(), 3);
    assert_eq!(prefix.tokens, vec![10, 20, 30]);
    assert_eq!(prefix.prefix_len(), HEADER_BYTES + 12);
    assert_eq!(prefix.layout().family(), ModelFamily::DeepSeek4);

    let mut with_tail = prefix.encode();
    with_tail.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    let again = parse_prefix(&with_tail).expect("tail is native leftover");
    assert_eq!(again, prefix);
    assert_eq!(payload_tail(&with_tail).unwrap(), &[0xde, 0xad, 0xbe, 0xef]);

    let mut host = SessionLedger::new(
        ModelFamily::DeepSeek4,
        SessionBackend::Cuda,
        8192,
        64,
    );
    host.apply_payload(&prefix).unwrap();
    assert!(host.valid);
    assert_eq!(host.pos(), 3);
    assert_eq!(host.tokens(), &[10, 20, 30]);
    assert!(!host.mtp_draft_valid);
    assert_eq!(host.generation, 1);
}

#[test]
fn encode_matches_host_prefix() {
    let p = HostPrefix {
        fields: [
            PAYLOAD_MAGIC,
            PAYLOAD_VERSION,
            8,
            1,
            8,
            8,
            2,
            0,
            1,
            1,
            1,
            8,
            0,
        ],
        tokens: Vec::new(),
    };
    let bytes = p.encode();
    assert_eq!(bytes.len(), HEADER_BYTES);
    assert_eq!(parse_prefix(&bytes).unwrap(), p);
}
