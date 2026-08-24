//! Four-way KVC matrix: C save/load × Rust save/load.

use ds4_kv::{
    decode_file, encode_file, eviction_score, read_path, store_len, write_path, Header, Options,
    Reason, Record, ScoreEntry, Store,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_KV_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/kv_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/kv_c_oracle (missing {})",
        p.display()
    );
    p
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

fn sample() -> Record {
    Record {
        header: Header {
            quant_bits: 2,
            reason: Reason::Cold,
            ext_flags: 0,
            model_id: 3,
            tokens: 512,
            hits: 7,
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

fn c_write(path: &Path, rec: &Record) {
    let status = Command::new(require_oracle())
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
        ])
        .status()
        .unwrap();
    assert!(status.success(), "C write failed");
}

fn c_read(path: &Path) -> Record {
    let out = Command::new(require_oracle())
        .args(["read", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success(), "C read failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let mut rec = sample();
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
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

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ds4-kv-4way-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn same_envelope(a: &Record, b: &Record) {
    assert_eq!(a.header.model_id, b.header.model_id);
    assert_eq!(a.header.quant_bits, b.header.quant_bits);
    assert_eq!(a.header.reason, b.header.reason);
    assert_eq!(a.header.ext_flags, b.header.ext_flags);
    assert_eq!(a.header.tokens, b.header.tokens);
    assert_eq!(a.header.hits, b.header.hits);
    assert_eq!(a.header.ctx_size, b.header.ctx_size);
    assert_eq!(a.header.created_at, b.header.created_at);
    assert_eq!(a.header.last_used, b.header.last_used);
    assert_eq!(a.header.payload_bytes, b.header.payload_bytes);
    assert_eq!(a.header.text_bytes, b.header.text_bytes);
    assert_eq!(a.text, b.text);
    assert_eq!(a.payload, b.payload);
    assert_eq!(a.trailer, b.trailer);
}

#[test]
fn rust_save_rust_load() {
    let rec = sample();
    let path = tmp("rr.kv");
    write_path(&path, &rec).unwrap();
    let got = read_path(&path).unwrap();
    same_envelope(&rec, &got);
    assert_eq!(encode_file(&rec), fs::read(&path).unwrap());
}

#[test]
fn c_save_c_load() {
    let rec = sample();
    let path = tmp("cc.kv");
    c_write(&path, &rec);
    let got = c_read(&path);
    same_envelope(&rec, &got);
}

#[test]
fn rust_save_c_load() {
    let rec = sample();
    let path = tmp("rc.kv");
    write_path(&path, &rec).unwrap();
    let got = c_read(&path);
    same_envelope(&rec, &got);
}

#[test]
fn c_save_rust_load() {
    let rec = sample();
    let path = tmp("cr.kv");
    c_write(&path, &rec);
    let got = decode_file(&fs::read(&path).unwrap()).unwrap();
    same_envelope(&rec, &got);
}

#[test]
fn rust_stream_payload_with_trailer_loads_in_c() {
    let base = tmp("stream-rust-c");
    let store_dir = base.join("store");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let mut rec = sample();
    rec.payload = b"DSV4 opaque native payload\0\xff".to_vec();
    rec.trailer = b"KVT1 tool-map trailer".to_vec();
    rec.header.ext_flags = ds4_kv::EXT_TOOL_MAP;
    rec.header.text_bytes = rec.text.len() as u32;
    rec.header.payload_bytes = rec.payload.len() as u64;
    let payload_path = ds4_kv::path_for_sha(&store_dir, &ds4_kv::text_sha_hex(&rec.text))
        .with_extension(format!("kv.tmp.{}", std::process::id()));
    fs::create_dir_all(&store_dir).unwrap();
    fs::write(&payload_path, &rec.payload).unwrap();

    let mut store = Store::open(&store_dir, 16, false, Options::default()).unwrap();
    let path = store
        .write_payload_file(rec.header.clone(), &rec.text, &payload_path, &rec.trailer)
        .unwrap();
    assert_eq!(fs::read(&payload_path).unwrap(), rec.payload);
    let got = c_read(&path);
    same_envelope(&rec, &got);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn sha1_matches_c() {
    let text = b"hello world";
    let out = Command::new(require_oracle())
        .args(["sha1", "--text-hex", &hex(text)])
        .output()
        .unwrap();
    assert!(out.status.success());
    let c = String::from_utf8(out.stdout).unwrap();
    assert_eq!(c.trim(), ds4_kv::text_sha_hex(text));
}

#[test]
fn store_len_matches_c() {
    for tokens in [100, 512, 2048, 4096 + 40, 20000] {
        let out = Command::new(require_oracle())
            .args(["store-len", "--tokens", &tokens.to_string()])
            .output()
            .unwrap();
        assert!(out.status.success());
        let c: i32 = String::from_utf8(out.stdout).unwrap().trim().parse().unwrap();
        assert_eq!(store_len(&Options::default(), tokens), c, "tokens={tokens}");
    }
}

#[test]
fn eviction_score_matches_c() {
    let out = Command::new(require_oracle())
        .args([
            "score",
            "--hits",
            "4",
            "--tokens",
            "512",
            "--file-size",
            "4096",
            "--reason",
            "1",
            "--created",
            "1000",
            "--used",
            "1000",
            "--now",
            "1000",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let c: f64 = String::from_utf8(out.stdout).unwrap().trim().parse().unwrap();
    let rust = eviction_score(
        &ScoreEntry {
            sha: "x",
            quant_bits: 2,
            model_id: 0,
            reason: Reason::Cold,
            tokens: 512,
            hits: 4,
            ctx_size: 2048,
            created_at: 1000,
            last_used: 1000,
            text_bytes: 11,
            file_size: 4096,
        },
        1000,
        None,
    );
    assert!((rust - c).abs() <= 1e-12, "rust={rust} c={c}");
}
