//! Byte-identical ports of the encode/JSON helpers in `ds4_web.c`.

pub fn url_encode(s: &str) -> String {
    url_encode_bytes(s.as_bytes())
}

pub fn url_encode_bytes(s: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::new();
    for &c in s {
        if c.is_ascii_alphanumeric() || c == b'-' || c == b'_' || c == b'.' || c == b'~' {
            out.push(c as char);
        } else {
            out.push('%');
            out.push(HEX[(c >> 4) as usize] as char);
            out.push(HEX[(c & 15) as usize] as char);
        }
    }
    out
}

pub fn base64(data: &[u8]) -> String {
    const TAB: &[u8] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i < data.len() {
        let mut v = u32::from(data[i]) << 16;
        if i + 1 < data.len() {
            v |= u32::from(data[i + 1]) << 8;
        }
        if i + 2 < data.len() {
            v |= u32::from(data[i + 2]);
        }
        out.push(TAB[((v >> 18) & 63) as usize] as char);
        out.push(TAB[((v >> 12) & 63) as usize] as char);
        out.push(if i + 1 < data.len() {
            TAB[((v >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if i + 2 < data.len() {
            TAB[(v & 63) as usize] as char
        } else {
            '='
        });
        i += 3;
    }
    out
}

pub fn json_quote(s: &str) -> String {
    json_quote_bytes(s.as_bytes())
}

pub fn json_quote_bytes(s: &[u8]) -> String {
    let mut out = String::from("\"");
    for &c in s {
        match c {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            c if c < 0x20 => out.push_str(&format!("\\u{c:04x}")),
            c => out.push(c as char),
        }
    }
    out.push('"');
    out
}

fn hex4(p: &[u8]) -> Option<u32> {
    if p.len() < 4 {
        return None;
    }
    let mut v = 0u32;
    for i in 0..4 {
        let c = p[i];
        let x = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        };
        v = (v << 4) | u32::from(x);
    }
    Some(v)
}

fn utf8_append_bytes(out: &mut Vec<u8>, code: u32) {
    if code <= 0x7f {
        out.push(code as u8);
    } else if code <= 0x7ff {
        out.push((0xc0 | (code >> 6)) as u8);
        out.push((0x80 | (code & 0x3f)) as u8);
    } else if code <= 0xffff {
        out.push((0xe0 | (code >> 12)) as u8);
        out.push((0x80 | ((code >> 6) & 0x3f)) as u8);
        out.push((0x80 | (code & 0x3f)) as u8);
    } else {
        out.push((0xf0 | (code >> 18)) as u8);
        out.push((0x80 | ((code >> 12) & 0x3f)) as u8);
        out.push((0x80 | ((code >> 6) & 0x3f)) as u8);
        out.push((0x80 | (code & 0x3f)) as u8);
    }
}

pub fn json_parse_string_at(input: &[u8]) -> Option<(Vec<u8>, usize)> {
    if input.first().copied() != Some(b'"') {
        return None;
    }
    let mut i = 1;
    let mut out = Vec::new();
    while i < input.len() && input[i] != b'"' {
        if input[i] != b'\\' {
            out.push(input[i]);
            i += 1;
            continue;
        }
        i += 1;
        if i >= input.len() {
            break;
        }
        match input[i] {
            b'"' => {
                out.push(b'"');
                i += 1;
            }
            b'\\' => {
                out.push(b'\\');
                i += 1;
            }
            b'/' => {
                out.push(b'/');
                i += 1;
            }
            b'b' => {
                out.push(b'\x08');
                i += 1;
            }
            b'f' => {
                out.push(b'\x0c');
                i += 1;
            }
            b'n' => {
                out.push(b'\n');
                i += 1;
            }
            b'r' => {
                out.push(b'\r');
                i += 1;
            }
            b't' => {
                out.push(b'\t');
                i += 1;
            }
            b'u' => {
                let v = hex4(input.get(i + 1..i + 5)?)?;
                i += 5;
                if (0xd800..=0xdbff).contains(&v)
                    && i + 1 < input.len()
                    && input[i] == b'\\'
                    && input[i + 1] == b'u'
                {
                    if let Some(lo) = hex4(input.get(i + 2..i + 6)?) {
                        if (0xdc00..=0xdfff).contains(&lo) {
                            let code = 0x10000 + (((v - 0xd800) << 10) + (lo - 0xdc00));
                            utf8_append_bytes(&mut out, code);
                            i += 6;
                            continue;
                        }
                    }
                }
                utf8_append_bytes(&mut out, v);
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    if i >= input.len() || input[i] != b'"' {
        return None;
    }
    Some((out, i + 1))
}

pub fn json_get_string_bytes(json: &str, key: &str) -> Option<Vec<u8>> {
    let pat = format!("\"{key}\"");
    let mut search = json;
    while let Some(idx) = search.find(&pat) {
        let mut p = &search[idx + pat.len()..];
        p = p.trim_start_matches([' ', '\t', '\r', '\n']);
        if !p.starts_with(':') {
            search = &search[idx + pat.len()..];
            continue;
        }
        p = p[1..].trim_start_matches([' ', '\t', '\r', '\n']);
        if p.starts_with('"') {
            let (bytes, _) = json_parse_string_at(p.as_bytes())?;
            return Some(bytes);
        }
        search = &search[idx + pat.len()..];
    }
    None
}

pub fn json_get_string(json: &str, key: &str) -> Option<String> {
    json_get_string_bytes(json, key).map(|b| String::from_utf8_lossy(&b).into_owned())
}

/// C `web_json_id_matches`: skip only space/tab after `:`, then `atoi`.
pub fn json_id_matches(json: &str, id: i32) -> bool {
    let Some(p) = json.find("\"id\"") else {
        return false;
    };
    let rest = &json[p + 4..];
    let Some(colon) = rest.find(':') else {
        return false;
    };
    let mut s = rest[colon + 1..].trim_start_matches([' ', '\t']);
    let mut sign = 1i32;
    if let Some(rest) = s.strip_prefix('-') {
        sign = -1;
        s = rest;
    } else if let Some(rest) = s.strip_prefix('+') {
        s = rest;
    }
    let mut n = 0i32;
    let mut any = false;
    for c in s.chars() {
        let Some(d) = c.to_digit(10) else { break };
        n = n.wrapping_mul(10).wrapping_add(d as i32);
        any = true;
    }
    let v = if any { n.wrapping_mul(sign) } else { 0 };
    v == id
}

pub fn random_bytes(len: usize) -> Vec<u8> {
    let mut buf = vec![0u8; len];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        if f.read_exact(&mut buf).is_ok() {
            return buf;
        }
    }
    let mut x = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        ^ (u64::from(std::process::id()) << 32);
    for b in &mut buf {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (x >> 32) as u8;
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_encode_unreserved_and_space() {
        assert_eq!(url_encode("abc-_.~"), "abc-_.~");
        assert_eq!(url_encode("a b"), "a%20b");
        assert_eq!(url_encode("q=1&x"), "q%3D1%26x");
    }

    #[test]
    fn base64_padding() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"a"), "YQ==");
        assert_eq!(base64(b"ab"), "YWI=");
        assert_eq!(base64(b"abc"), "YWJj");
    }

    #[test]
    fn json_quote_escapes() {
        assert_eq!(json_quote("ab"), "\"ab\"");
        assert_eq!(json_quote("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_quote("a\nb"), "\"a\\nb\"");
        assert_eq!(json_quote("\u{01}"), "\"\\u0001\"");
    }

    #[test]
    fn json_get_string_first_value() {
        let json = r#"{"id":1,"value":"hello\n","value":"x"}"#;
        assert_eq!(json_get_string(json, "value").as_deref(), Some("hello\n"));
        assert!(json_id_matches(json, 1));
        assert!(!json_id_matches(json, 2));
    }

    #[test]
    fn json_surrogate_pair() {
        let json = r#"{"t":"\uD83D\uDE00"}"#;
        assert_eq!(json_get_string(json, "t").as_deref(), Some("😀"));
    }
}
