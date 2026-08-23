//! Byte compare Rust helpers against tests/parity/web_c_oracle.

use ds4_web::{
    base64, cdp_request, create_target_params, eval_params, google_search_url, http_request,
    json_get_string, json_id_matches, json_quote, linux_chrome_args, navigate_params, url_encode,
    ws_handshake, ws_text_frame,
};
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_WEB_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/web_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/web_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run web_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}\n{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn url_encode_matches_c() {
    for s in ["", "abc-_.~", "a b", "q=1&x", "한글", "A/Z"] {
        assert_eq!(url_encode(s), c_out(&["url-encode", s]), "url_encode {s:?}");
    }
}

#[test]
fn base64_matches_c() {
    for data in [b"".as_slice(), b"a", b"ab", b"abc", b"0123456789abcdef"] {
        assert_eq!(base64(data), c_out(&["base64", &hex(data)]), "base64 {data:?}");
    }
}

#[test]
fn json_quote_matches_c() {
    for s in ["", "ab", "a\"b", "a\\b", "a\nb", "a\tb", "\u{01}"] {
        assert_eq!(json_quote(s), c_out(&["json-quote", s]), "json_quote {s:?}");
    }
}

#[test]
fn json_get_and_id_match_c() {
    let json = r#"{"id": 7, "value":"hello\n", "t":"\uD83D\uDE00"}"#;
    assert_eq!(
        json_get_string(json, "value").unwrap(),
        c_out(&["json-get", json, "value"])
    );
    assert_eq!(
        json_get_string(json, "t").unwrap(),
        c_out(&["json-get", json, "t"])
    );
    assert_eq!(
        if json_id_matches(json, 7) { "yes" } else { "no" },
        c_out(&["json-id", json, "7"])
    );
    assert_eq!(
        if json_id_matches(json, 8) { "yes" } else { "no" },
        c_out(&["json-id", json, "8"])
    );
}

#[test]
fn request_builders_match_c() {
    assert_eq!(
        http_request("GET", 9333, "/json/version"),
        c_out(&["http-req", "GET", "9333", "/json/version"])
    );
    assert_eq!(
        ws_handshake("/devtools/browser/x", "127.0.0.1", 9333, "dGVzdA=="),
        c_out(&[
            "ws-handshake",
            "/devtools/browser/x",
            "127.0.0.1",
            "9333",
            "dGVzdA=="
        ])
    );
    assert_eq!(
        cdp_request(3, "Runtime.evaluate", Some("{}")),
        c_out(&["cdp-req", "3", "Runtime.evaluate", "{}"])
    );
    assert_eq!(
        eval_params("document.readyState"),
        c_out(&["eval-params", "document.readyState"])
    );
    assert_eq!(
        navigate_params("https://example.com/a?b=1"),
        c_out(&["navigate-params", "https://example.com/a?b=1"])
    );
    assert_eq!(
        create_target_params("about:blank"),
        c_out(&["create-target-params", "about:blank"])
    );
    assert_eq!(
        google_search_url("rust host"),
        c_out(&["search-url", "rust host"])
    );
}

#[test]
fn ws_frame_matches_c() {
    let text = b"{\"id\":1}";
    let mask = [0x11, 0x22, 0x33, 0x44];
    assert_eq!(
        hex(&ws_text_frame(text, mask)),
        c_out(&["ws-frame", &hex(text), &hex(&mask)])
    );
}

#[test]
fn chrome_linux_args_match_c() {
    let profile = Path::new("/tmp/ds4-web-profile");
    let rust = linux_chrome_args(9333, profile, false).join("\n") + "\n";
    assert_eq!(
        rust,
        c_out(&["chrome-linux-args", "9333", "/tmp/ds4-web-profile", "user"])
    );
    let rust_root = linux_chrome_args(9333, profile, true).join("\n") + "\n";
    assert_eq!(
        rust_root,
        c_out(&["chrome-linux-args", "9333", "/tmp/ds4-web-profile", "root"])
    );
}
