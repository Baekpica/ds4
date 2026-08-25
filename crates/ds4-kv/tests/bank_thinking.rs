//! Bank thinking bit + one KTM extension record.
//!
//! Given / When / Then against the C KVC envelope: thinking is
//! `EXT_THINKING_VISIBLE`, the only post-DSV4 magic trailer is `KTM\x01`.

use ds4_kv::{
    decode_file, encode_file, read_path, write_path, BankThinkingExtensions, ExtensionRecord,
    Header, Reason, Record, EXT_BANK_REPLAY_V1, EXT_THINKING_VISIBLE, EXT_TOOL_MAP,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn sample_bank() -> Record {
    Record {
        header: Header {
            quant_bits: 2,
            reason: Reason::BankCheckpoint,
            ext_flags: EXT_BANK_REPLAY_V1,
            model_id: 3,
            tokens: 64,
            hits: 1,
            ctx_size: 4096,
            created_at: 1_700_000_000,
            last_used: 1_700_000_050,
            payload_bytes: 6,
            text_bytes: 11,
        },
        text: b"hello world".to_vec(),
        payload: b"PAYLOD".to_vec(),
        trailer: Vec::new(),
    }
}

fn tmp(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("ds4-kv-bank-think-{}-{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir.join("bank.kv")
}

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_KV_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/kv_c_oracle")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn c_write(path: &Path, rec: &Record) {
    let status = Command::new(oracle())
        .args([
            "write",
            "--path",
            path.to_str().unwrap(),
            "--text-hex",
            &hex(&rec.text),
            "--payload-hex",
            &hex(&rec.payload),
            "--model-id",
            &rec.header.model_id.to_string(),
            "--quant",
            &rec.header.quant_bits.to_string(),
            "--reason",
            &(rec.header.reason as u8).to_string(),
            "--ext",
            &rec.header.ext_flags.to_string(),
            "--tokens",
            &rec.header.tokens.to_string(),
            "--hits",
            &rec.header.hits.to_string(),
            "--ctx",
            &rec.header.ctx_size.to_string(),
            "--created",
            &rec.header.created_at.to_string(),
            "--used",
            &rec.header.last_used.to_string(),
            "--trailer-hex",
            &hex(&rec.trailer),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "C write failed");
}

fn c_read(path: &Path) -> Record {
    let out = Command::new(oracle())
        .args(["read", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "C read failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let mut rec = sample_bank();
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k {
            "model_id" => rec.header.model_id = v.parse().unwrap(),
            "quant_bits" => rec.header.quant_bits = v.parse().unwrap(),
            "reason" => rec.header.reason = Reason::from_u8(v.parse().unwrap()),
            "ext_flags" => rec.header.ext_flags = v.parse().unwrap(),
            "tokens" => rec.header.tokens = v.parse().unwrap(),
            "hits" => rec.header.hits = v.parse().unwrap(),
            "ctx_size" => rec.header.ctx_size = v.parse().unwrap(),
            "created_at" => rec.header.created_at = v.parse().unwrap(),
            "last_used" => rec.header.last_used = v.parse().unwrap(),
            "payload_bytes" => rec.header.payload_bytes = v.parse().unwrap(),
            "text_bytes" => rec.header.text_bytes = v.parse().unwrap(),
            "text_hex" => rec.text = unhex(v),
            "payload_hex" => rec.payload = unhex(v),
            "trailer_hex" => rec.trailer = unhex(v),
            _ => {}
        }
    }
    rec
}

fn one_extension() -> BankThinkingExtensions {
    BankThinkingExtensions {
        thinking_visible: true,
        records: vec![ExtensionRecord {
            id: b"call_abc".to_vec(),
            body: b"<dsml>function_calls>x".to_vec(),
        }],
    }
}

#[test]
fn bank_save_load_thinking_bit_and_one_extension_record() {
    // Given: a bank checkpoint with thinking-visible text and one KTM record
    let ext = one_extension();
    let mut rec = sample_bank();
    rec.persist_bank_extensions(&ext);

    // When: the envelope is written and read back
    let path = tmp("roundtrip");
    write_path(&path, &rec).unwrap();
    let got = read_path(&path).unwrap();
    let restored = BankThinkingExtensions::restore(got.header.ext_flags, &got.trailer);

    // Then: thinking bit + BANK_REPLAY_V1 + TOOL_MAP and the one record survive
    assert_eq!(
        got.header.ext_flags,
        EXT_BANK_REPLAY_V1 | EXT_THINKING_VISIBLE | EXT_TOOL_MAP
    );
    assert!(restored.thinking_visible);
    assert_eq!(restored.records, ext.records);
    assert_eq!(&got.trailer[..4], b"KTM\x01");
    assert_eq!(encode_file(&rec)[..3], *b"KVC");

    let _ = fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn missing_extension_section_is_ignored_like_c() {
    // Given: a bank file with replay-v1 only (no thinking bit, no trailer)
    let rec = sample_bank();
    let bytes = encode_file(&rec);
    let got = decode_file(&bytes).unwrap();

    // When: restore runs on a missing section
    let restored = BankThinkingExtensions::restore(got.header.ext_flags, &got.trailer);

    // Then: C-like ignore — thinking off, zero records, file still valid
    assert_eq!(got.header.ext_flags, EXT_BANK_REPLAY_V1);
    assert!(!restored.thinking_visible);
    assert!(restored.records.is_empty());
}

#[test]
fn invalid_ktm_magic_is_ignored_like_c() {
    // Given: a trailer that is not KTM\x01
    let restored = BankThinkingExtensions::restore(EXT_TOOL_MAP, b"XXX\x01\x01\0\0\0");

    // When/Then: C kv_tool_map_load_from_pos returns 0
    assert!(!restored.thinking_visible);
    assert!(restored.records.is_empty());
}

#[test]
fn rust_bank_thinking_extension_is_c_readable() {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build the C oracle first: make tests/parity/kv_c_oracle (missing {})",
        oracle.display()
    );

    let ext = one_extension();
    let mut rec = sample_bank();
    rec.persist_bank_extensions(&ext);
    rec.header.text_bytes = rec.text.len() as u32;
    rec.header.payload_bytes = rec.payload.len() as u64;

    let path = tmp("rust-c");
    write_path(&path, &rec).unwrap();
    let got = c_read(&path);
    let restored = BankThinkingExtensions::restore(got.header.ext_flags, &got.trailer);

    assert_eq!(got.header.ext_flags, rec.header.ext_flags);
    assert_eq!(got.trailer, rec.trailer);
    assert!(restored.thinking_visible);
    assert_eq!(restored.records, ext.records);

    let _ = fs::remove_dir_all(path.parent().unwrap());
}

#[test]
fn c_bank_thinking_extension_loads_in_rust() {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build the C oracle first: make tests/parity/kv_c_oracle (missing {})",
        oracle.display()
    );

    let ext = one_extension();
    let mut rec = sample_bank();
    rec.persist_bank_extensions(&ext);
    rec.header.text_bytes = rec.text.len() as u32;
    rec.header.payload_bytes = rec.payload.len() as u64;

    let path = tmp("c-rust");
    c_write(&path, &rec);
    let got = decode_file(&fs::read(&path).unwrap()).unwrap();
    let restored = BankThinkingExtensions::restore(got.header.ext_flags, &got.trailer);

    assert_eq!(got.header.ext_flags, rec.header.ext_flags);
    assert_eq!(got.trailer, rec.trailer);
    assert!(restored.thinking_visible);
    assert_eq!(restored.records, ext.records);

    let _ = fs::remove_dir_all(path.parent().unwrap());
}
