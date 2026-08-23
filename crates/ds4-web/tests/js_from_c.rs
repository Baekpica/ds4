//! JS extractors must stay byte-identical to `ds4_web.c`.

use ds4_web::{CLICK_GOOGLE_CONSENT, EXTRACT_PAGE, EXTRACT_SEARCH, PAGE_PROBE, SCROLL_DYNAMIC};
use std::fs;
use std::path::PathBuf;

fn unescape_c(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' {
            out.push(b[i] as char);
            i += 1;
            continue;
        }
        i += 1;
        if i >= b.len() {
            break;
        }
        match b[i] {
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'\\' => out.push('\\'),
            b'"' => out.push('"'),
            b'\'' => out.push('\''),
            other => out.push(other as char),
        }
        i += 1;
    }
    out
}

fn extract_c_strings(src: &str, after: &str) -> String {
    let start = src
        .find(after)
        .unwrap_or_else(|| panic!("missing {after}"));
    let mut i = start + after.len();
    let bytes = src.as_bytes();
    let mut out = String::new();
    while i < bytes.len() && bytes[i] != b';' {
        if bytes[i] == b'"' {
            i += 1;
            let lit_start = i;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'"' {
                    break;
                }
                i += 1;
            }
            out.push_str(&unescape_c(&src[lit_start..i]));
            i += 1;
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn js_matches_ds4_web_c() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ds4_web.c");
    let src = fs::read_to_string(&path).expect("read ds4_web.c");
    assert_eq!(
        CLICK_GOOGLE_CONSENT,
        extract_c_strings(&src, "static const char *web_click_google_consent_js =")
    );
    assert_eq!(
        EXTRACT_SEARCH,
        extract_c_strings(&src, "static const char *web_extract_search_js =")
    );
    assert_eq!(
        EXTRACT_PAGE,
        extract_c_strings(&src, "static const char *web_extract_page_js =")
    );
    let scroll_src = &src[src.find("static void web_scroll_dynamic_page").unwrap()..];
    assert_eq!(
        SCROLL_DYNAMIC,
        extract_c_strings(scroll_src, "const char *expr =")
    );
    let probe_src = &src[src.find("static bool web_page_probe").unwrap()..];
    assert_eq!(PAGE_PROBE, extract_c_strings(probe_src, "const char *expr ="));
}
