use super::TOOL_UNSUPPORTED_ERROR;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const DSML_OPEN: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const DSML_OPEN_MISSING_BAR: &[u8] = "<DSML｜tool_calls>".as_bytes();
const DSML_CLOSE_PREFIX: &str = "</｜DSML｜";
const GOOGLE_SEARCH: &str = "google_search";
const QUERY_ARG: &str = "query";
const VISIT_PAGE: &str = "visit_page";
const URL_ARG: &str = "url";
const READ_FILE: &str = "read";
const PATH_ARG: &str = "path";
const START_LINE_ARG: &str = "start_line";
const MAX_LINES_ARG: &str = "max_lines";
const WHOLE_ARG: &str = "whole";
const RAW_ARG: &str = "raw";
const FILE_MAX_BYTES: usize = 16 * 1024 * 1024;
const RESULT_MAX_BYTES: usize = 128 * 1024;
const READ_ERROR_MAX_BYTES: usize = 255;
const READ_DEFAULT_LINES: usize = 500;
const WEB_HEAD_BYTES: usize = 8 * 1024;
const WEB_HEAD_LINES: usize = 100;
const TEMP_ATTEMPTS: u64 = 1024;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub(super) struct ToolRound {
    pub(super) visible: Vec<u8>,
    pub(super) observation: Vec<u8>,
}

pub(super) trait Browser {
    fn google_search(&mut self, query: &str) -> Result<String, String>;
    fn visit_page(&mut self, url: &str) -> Result<String, String>;
}

impl Browser for ds4_web::Web {
    fn google_search(&mut self, query: &str) -> Result<String, String> {
        ds4_web::Web::google_search(self, query)
    }

    fn visit_page(&mut self, url: &str) -> Result<String, String> {
        ds4_web::Web::visit_page(self, url)
    }
}

pub(super) fn non_interactive_web() -> ds4_web::Web {
    ds4_web::Web::new(ds4_web::Config {
        confirm: Some(Box::new(|_| {
            Err("visible Chrome browser startup requires interactive approval".into())
        })),
        ..ds4_web::Config::default()
    })
}

struct ToolCall {
    name: String,
    args: Vec<(String, String)>,
}

impl ToolCall {
    fn arg(&self, name: &str) -> Option<&str> {
        self.args
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

fn bytes_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn start(raw: &[u8]) -> Option<(usize, usize)> {
    let canonical = bytes_find(raw, DSML_OPEN).map(|at| (at, DSML_OPEN.len()));
    let missing =
        bytes_find(raw, DSML_OPEN_MISSING_BAR).map(|at| (at, DSML_OPEN_MISSING_BAR.len()));
    match (canonical, missing) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (Some(found), None) | (None, Some(found)) => Some(found),
        (None, None) => None,
    }
}

fn skip_space(text: &str, mut at: usize) -> usize {
    while text.as_bytes().get(at).is_some_and(u8::is_ascii_whitespace) {
        at += 1;
    }
    at
}

fn close_at(text: &str, at: usize, name: &str) -> Option<usize> {
    let prefix = format!("{DSML_CLOSE_PREFIX}{name}");
    if !text.get(at..)?.starts_with(&prefix) {
        return None;
    }
    let mut end = skip_space(text, at + prefix.len());
    if text.get(end..)?.starts_with('｜') {
        end += '｜'.len_utf8();
    }
    end = skip_space(text, end);
    (text.as_bytes().get(end) == Some(&b'>')).then_some(end + 1)
}

fn find_close(text: &str, mut at: usize, name: &str) -> Option<(usize, usize)> {
    while let Some(relative) = text.get(at..)?.find(DSML_CLOSE_PREFIX) {
        let start = at + relative;
        if let Some(end) = close_at(text, start, name) {
            return Some((start, end));
        }
        at = start + 1;
    }
    None
}

fn attr(tag: &str, name: &str) -> Option<String> {
    let pattern = format!("{name}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag.get(start..)?.find('"')? + start;
    Some(tag[start..end].to_string())
}

fn parse_calls(raw: &[u8], open_at: usize, open_len: usize) -> Result<Vec<ToolCall>, String> {
    let text = std::str::from_utf8(raw).map_err(|_| "invalid UTF-8 in DSML tool call")?;
    let mut at = open_at + open_len;
    let mut current: Option<ToolCall> = None;
    let mut calls = Vec::new();

    loop {
        at = skip_space(text, at);
        if close_at(text, at, "tool_calls").is_some() {
            if let Some(call) = current.take() {
                calls.push(call);
            }
            return Ok(calls);
        }
        if let Some(end) = close_at(text, at, "invoke") {
            if let Some(call) = current.take() {
                calls.push(call);
            }
            at = end;
            continue;
        }

        let end = text
            .get(at..)
            .and_then(|tail| tail.find('>'))
            .map(|relative| at + relative + 1)
            .ok_or_else(|| "incomplete DSML tool call".to_string())?;
        let tag = &text[at..end];
        if tag.starts_with("<｜DSML｜invoke") {
            let name = attr(tag, "name").ok_or_else(|| "tool invoke without name".to_string())?;
            current = Some(ToolCall {
                name,
                args: Vec::new(),
            });
            at = end;
            continue;
        }
        if tag.starts_with("<｜DSML｜parameter") {
            let name =
                attr(tag, "name").ok_or_else(|| "tool parameter without name".to_string())?;
            let (value_end, close_end) = find_close(text, end, "parameter")
                .ok_or_else(|| "incomplete DSML tool parameter".to_string())?;
            if let Some(call) = current.as_mut() {
                call.args.push((name, text[end..value_end].to_string()));
            }
            at = close_end;
            continue;
        }
        let display: String = tag.chars().take(80).collect();
        return Err(format!("unexpected DSML tag: {display}"));
    }
}

pub(super) fn block_complete(raw: &[u8]) -> bool {
    let Some((open_at, open_len)) = start(raw) else {
        return false;
    };
    let Ok(text) = std::str::from_utf8(raw) else {
        return false;
    };
    find_close(text, open_at + open_len, "tool_calls").is_some()
}

pub(super) fn has_block(raw: &[u8]) -> bool {
    start(raw).is_some()
}

#[derive(Clone, Copy)]
struct LineSpan {
    start: usize,
    content_end: usize,
    end: usize,
}

#[derive(Default)]
struct ResultBuf {
    bytes: Vec<u8>,
    truncated: bool,
}

impl ResultBuf {
    fn append(&mut self, bytes: &[u8]) {
        if bytes.is_empty() || self.truncated {
            return;
        }
        let room = RESULT_MAX_BYTES.saturating_sub(self.bytes.len());
        let n = room.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..n]);
        self.truncated = n < bytes.len();
    }

    fn push(&mut self, byte: u8) {
        self.append(&[byte]);
    }

    fn into_vec(self) -> Vec<u8> {
        self.bytes
    }
}

// Match C's CR/LF splitting without allocating one span per input line.
struct Lines<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Lines<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl Iterator for Lines<'_> {
    type Item = LineSpan;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let start = self.pos;
        while self.pos < self.data.len() && !matches!(self.data[self.pos], b'\n' | b'\r') {
            self.pos += 1;
        }
        let content_end = self.pos;
        if self.pos < self.data.len() {
            if self.data[self.pos] == b'\r' && self.data.get(self.pos + 1).copied() == Some(b'\n') {
                self.pos += 2;
            } else {
                self.pos += 1;
            }
        }
        Some(LineSpan {
            start,
            content_end,
            end: self.pos,
        })
    }
}

fn parse_read_int(value: Option<&str>, default: usize) -> usize {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return default;
    };
    let value = value
        .trim_start_matches(|char: char| char.is_ascii_whitespace())
        .trim_end_matches([' ', '\t', '\r', '\n']);
    let digits = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return default;
    }
    let parsed = match value.parse::<i128>() {
        Ok(parsed) => parsed,
        Err(error) => match error.kind() {
            std::num::IntErrorKind::PosOverflow => i128::MAX,
            std::num::IntErrorKind::NegOverflow => i128::MIN,
            _ => return default,
        },
    };
    parsed.clamp(1, i128::from(i32::MAX)) as usize
}

fn parse_bool(value: Option<&str>, default: bool) -> bool {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return default;
    };
    if value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes") || value == "1" {
        return true;
    }
    if value.eq_ignore_ascii_case("false") || value.eq_ignore_ascii_case("no") || value == "0" {
        return false;
    }
    default
}

fn io_detail(error: &std::io::Error) -> String {
    let detail = error.to_string();
    if error.raw_os_error().is_some() {
        if let Some(at) = detail.rfind(" (os error ") {
            return detail[..at].to_string();
        }
    }
    detail
}

fn read_error(message: String) -> Vec<u8> {
    let mut error = message.into_bytes();
    error.truncate(READ_ERROR_MAX_BYTES);
    error
}

fn read_bytes(path: &str) -> Result<Vec<u8>, Vec<u8>> {
    let file = File::open(path)
        .map_err(|error| read_error(format!("open {path}: {}", io_detail(&error))))?;
    let mut data = Vec::new();
    file.take((FILE_MAX_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|error| read_error(format!("read {path}: {}", io_detail(&error))))?;
    if data.len() > FILE_MAX_BYTES {
        return Err(read_error(format!(
            "file too large: {path} exceeds {FILE_MAX_BYTES} bytes"
        )));
    }
    Ok(data)
}

fn read_result(call: &ToolCall) -> Vec<u8> {
    let Some(path) = call.arg(PATH_ARG).filter(|path| !path.is_empty()) else {
        return format!("Tool error: {READ_FILE} requires {PATH_ARG}\n").into_bytes();
    };
    let data = match read_bytes(path) {
        Ok(data) => data,
        Err(error) => {
            let mut out = b"Tool error: ".to_vec();
            out.extend_from_slice(&error);
            out.push(b'\n');
            return out;
        }
    };

    let total_lines = Lines::new(&data).count();
    let start_line = parse_read_int(call.arg(START_LINE_ARG), 1);
    let start_idx = start_line.saturating_sub(1).min(total_lines);
    let whole = parse_bool(call.arg(WHOLE_ARG), false);
    let max_lines = if whole {
        total_lines - start_idx
    } else {
        parse_read_int(call.arg(MAX_LINES_ARG), READ_DEFAULT_LINES)
    };
    let end_idx = start_idx.saturating_add(max_lines).min(total_lines);

    if parse_bool(call.arg(RAW_ARG), false) {
        let start = Lines::new(&data)
            .nth(start_idx)
            .map(|span| span.start)
            .unwrap_or(data.len());
        let end = if end_idx > start_idx {
            Lines::new(&data)
                .nth(end_idx - 1)
                .map(|span| span.end)
                .unwrap_or(start)
        } else {
            start
        };
        let mut out = ResultBuf::default();
        out.append(&data[start..end]);
        if end > start && !out.bytes.ends_with(b"\n") {
            out.push(b'\n');
        }
        if end_idx < total_lines {
            out.append(
                format!(
                    "[Read truncated at line {end_idx} of {total_lines}. continue_offset={}. \
                     Call more with count={max_lines} to read the next chunk.]\n",
                    end_idx + 1
                )
                .as_bytes(),
            );
        }
        return out.into_vec();
    }

    let first_line = if total_lines == 0 { 0 } else { start_idx + 1 };
    let header = if end_idx < total_lines {
        format!(
            "{path}: lines {first_line}-{end_idx} of {total_lines}; continue_offset={}; \
             call more with count={max_lines} to read the next chunk\n",
            end_idx + 1
        )
        .into_bytes()
    } else {
        format!("{path}: lines {first_line}-{end_idx} of {total_lines}\n").into_bytes()
    };
    let mut out = ResultBuf::default();
    out.append(&header);
    for (index, span) in Lines::new(&data)
        .enumerate()
        .skip(start_idx)
        .take(end_idx - start_idx)
    {
        if out.truncated {
            break;
        }
        out.append(format!("{} ", index + 1).as_bytes());
        out.append(&data[span.start..span.content_end]);
        out.push(b'\n');
    }
    out.into_vec()
}

fn count_lines(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.bytes().filter(|byte| *byte == b'\n').count() + usize::from(!text.ends_with('\n'))
}

fn string_head(text: &str) -> (&str, usize, bool) {
    let bytes = text.as_bytes();
    let mut used = 0;
    let mut lines = 0;
    while used < bytes.len() && used < WEB_HEAD_BYTES && lines < WEB_HEAD_LINES {
        if bytes[used] == b'\n' {
            lines += 1;
        }
        used += 1;
    }
    let byte_limited = used < bytes.len() && used >= WEB_HEAD_BYTES;
    while !text.is_char_boundary(used) {
        used -= 1;
    }
    if used > 0 && bytes[used - 1] != b'\n' && lines < WEB_HEAD_LINES {
        lines += 1;
    }
    (&text[..used], lines, byte_limited)
}

fn write_temp(text: &str) -> Result<PathBuf, String> {
    let clock = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let seed =
        clock ^ (u64::from(std::process::id()) << 24) ^ TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let mut collision = None;

    for attempt in 0..TEMP_ATTEMPTS {
        let suffix = format!("{:06x}", seed.wrapping_add(attempt) & 0x00ff_ffff);
        let path = PathBuf::from(format!("/tmp/ds4_agent_web_{suffix}"));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path);
        let mut file = match file {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                collision = Some(error);
                continue;
            }
            Err(error) => return Err(format!("failed to create temporary file: {error}")),
        };
        if let Err(error) = file.write_all(text.as_bytes()) {
            drop(file);
            let _ = std::fs::remove_file(&path);
            return Err(format!("failed to write temporary file: {error}"));
        }
        drop(file);
        return Ok(path);
    }

    Err(format!(
        "failed to create temporary file: {}",
        collision
            .map(|error| error.to_string())
            .unwrap_or_else(|| "name collision".into())
    ))
}

fn visit_result<B: Browser>(web: &mut B, url: Option<&str>) -> String {
    let Some(url) = url.filter(|url| !url.is_empty()) else {
        return format!("Tool error: {VISIT_PAGE} requires {URL_ARG}\n");
    };
    let markdown = match web.visit_page(url) {
        Ok(markdown) => markdown,
        Err(error) => {
            let detail = if error.is_empty() {
                "unknown error"
            } else {
                &error
            };
            return format!("Tool error: {VISIT_PAGE} failed: {detail}\n");
        }
    };
    let path = match write_temp(&markdown) {
        Ok(path) => path,
        Err(error) => return format!("Tool error: {VISIT_PAGE} failed: {error}\n"),
    };

    let total_lines = count_lines(&markdown);
    let (head, shown_lines, byte_limited) = string_head(&markdown);
    let truncated = byte_limited || shown_lines < total_lines;
    let mut out = format!(
        "{VISIT_PAGE} url={url}\noutput_path={} ({} bytes, {total_lines} lines)\n",
        path.display(),
        markdown.len()
    );
    if truncated {
        out.push_str(&format!("<head -{WEB_HEAD_LINES} {}>\n", path.display()));
        out.push_str(head);
        if !head.is_empty() && !head.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("</head>\n");
        out.push_str(
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n",
        );
    } else {
        out.push_str("<markdown>\n");
        out.push_str(head);
        if !head.is_empty() && !head.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("</markdown>\n");
    }
    out
}

pub(super) fn handle_round<B: Browser>(raw: &[u8], web: &mut B) -> Result<ToolRound, String> {
    let (open_at, open_len) = start(raw).ok_or_else(|| "missing DSML tool call".to_string())?;
    let calls = parse_calls(raw, open_at, open_len)?;
    let mut observation = ResultBuf::default();
    if calls.is_empty() {
        observation.append(b"Tool error: empty tool call block\n");
    }
    for (index, call) in calls.iter().enumerate() {
        let result = match call.name.as_str() {
            GOOGLE_SEARCH => match call.arg(QUERY_ARG) {
                None | Some("") => {
                    format!("Tool error: {GOOGLE_SEARCH} requires {QUERY_ARG}\n").into_bytes()
                }
                Some(query) => match web.google_search(query) {
                    Ok(markdown) => markdown.into_bytes(),
                    Err(error) => {
                        format!("Tool error: {GOOGLE_SEARCH} failed: {error}\n").into_bytes()
                    }
                },
            },
            VISIT_PAGE => visit_result(web, call.arg(URL_ARG)).into_bytes(),
            READ_FILE => read_result(call),
            _ => return Err(TOOL_UNSUPPORTED_ERROR.into()),
        };
        observation.append(format!("Tool result {} ({}):\n", index + 1, call.name).as_bytes());
        // C combines tool results through strlen(), so embedded NUL ends the result.
        let result_end = result
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(result.len());
        let result = &result[..result_end];
        observation.append(result);
        if !result.is_empty() && !result.ends_with(b"\n") {
            observation.push(b'\n');
        }
    }
    Ok(ToolRound {
        visible: raw[..open_at].to_vec(),
        observation: observation.into_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeWeb {
        result: Result<String, String>,
    }

    impl Browser for FakeWeb {
        fn google_search(&mut self, _query: &str) -> Result<String, String> {
            self.result.clone()
        }

        fn visit_page(&mut self, _url: &str) -> Result<String, String> {
            self.result.clone()
        }
    }

    fn call(query: &str) -> Vec<u8> {
        format!(
            "answer prefix<｜DSML｜tool_calls>\n\
             <｜DSML｜invoke name=\"google_search\">\n\
             <｜DSML｜parameter name=\"query\" string=\"true\">{query}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n\
             </｜DSML｜tool_calls>"
        )
        .into_bytes()
    }

    fn visit_call(url: &str) -> Vec<u8> {
        format!(
            "<｜DSML｜tool_calls>\n\
             <｜DSML｜invoke name=\"visit_page\">\n\
             <｜DSML｜parameter name=\"url\" string=\"true\">{url}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n\
             </｜DSML｜tool_calls>"
        )
        .into_bytes()
    }

    fn output_path(observation: &[u8]) -> &str {
        std::str::from_utf8(observation)
            .expect("UTF-8 web observation")
            .split_once("output_path=")
            .and_then(|(_, tail)| tail.split_once(" (").map(|(path, _)| path))
            .expect("output path")
    }

    #[test]
    fn google_search_round_reaches_web_and_formats_c_observation() {
        let mut web = FakeWeb {
            result: Ok("# Results\n- [DS4](https://example.com)".into()),
        };

        let raw = call("rust host");
        assert!(block_complete(&raw));
        let round = handle_round(&raw, &mut web).expect("valid DSML");

        assert_eq!(round.visible, b"answer prefix");
        assert_eq!(
            round.observation.as_slice(),
            b"Tool result 1 (google_search):\n# Results\n- [DS4](https://example.com)\n"
        );
    }

    #[test]
    fn google_search_error_and_missing_query_match_c_tool_text() {
        let mut failed = FakeWeb {
            result: Err("Chrome unavailable".into()),
        };
        assert_eq!(
            handle_round(&call("rust host"), &mut failed)
                .expect("valid DSML")
                .observation
                .as_slice(),
            b"Tool result 1 (google_search):\nTool error: google_search failed: Chrome unavailable\n"
        );

        let mut web = FakeWeb {
            result: Ok(String::new()),
        };
        assert_eq!(
            handle_round(&call(""), &mut web)
                .expect("valid DSML")
                .observation
                .as_slice(),
            b"Tool result 1 (google_search):\nTool error: google_search requires query\n"
        );
    }

    #[test]
    fn incomplete_or_non_web_dsml_does_not_execute() {
        let mut web = FakeWeb {
            result: Ok("must not be used".into()),
        };
        let incomplete = b"<\xef\xbd\x9cDSML\xef\xbd\x9ctool_calls>";
        assert!(!block_complete(incomplete));
        assert!(handle_round(incomplete, &mut web).is_err());

        let unsupported = call("rust host")
            .windows("google_search".len())
            .position(|window| window == b"google_search")
            .map(|at| {
                let mut raw = call("rust host");
                raw.splice(at..at + "google_search".len(), b"bash".iter().copied());
                raw
            })
            .expect("tool name");
        assert!(handle_round(&unsupported, &mut web).is_err());
    }

    #[test]
    fn missing_bar_opener_uses_the_same_web_path() {
        let mut raw = call("rust host");
        let at = raw
            .windows(DSML_OPEN.len())
            .position(|window| window == DSML_OPEN)
            .expect("canonical opener");
        raw.splice(
            at..at + DSML_OPEN.len(),
            DSML_OPEN_MISSING_BAR.iter().copied(),
        );
        let mut web = FakeWeb {
            result: Ok("result".into()),
        };

        assert_eq!(
            handle_round(&raw, &mut web)
                .expect("accepted C detector spelling")
                .observation
                .as_slice(),
            b"Tool result 1 (google_search):\nresult\n"
        );
    }

    #[test]
    fn visit_page_writes_full_markdown_and_formats_c_observation() {
        let markdown = "# Page\nbody\n";
        let mut web = FakeWeb {
            result: Ok(markdown.into()),
        };

        let round =
            handle_round(&visit_call("https://example.com/"), &mut web).expect("visit_page tool");
        let path = output_path(&round.observation);

        assert_eq!(std::fs::read_to_string(path).unwrap(), markdown);
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            round.observation,
            format!(
                "Tool result 1 (visit_page):\n\
                 visit_page url=https://example.com/\n\
                 output_path={path} (12 bytes, 2 lines)\n\
                 <markdown>\n# Page\nbody\n</markdown>\n"
            )
            .into_bytes()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn visit_page_errors_match_c_tool_text() {
        let mut failed = FakeWeb {
            result: Err("Chrome unavailable".into()),
        };
        assert_eq!(
            handle_round(&visit_call("https://example.com/"), &mut failed)
                .expect("visit_page error observation")
                .observation
                .as_slice(),
            b"Tool result 1 (visit_page):\nTool error: visit_page failed: Chrome unavailable\n"
        );

        let mut web = FakeWeb {
            result: Ok(String::new()),
        };
        assert_eq!(
            handle_round(&visit_call(""), &mut web)
                .expect("missing URL observation")
                .observation
                .as_slice(),
            b"Tool result 1 (visit_page):\nTool error: visit_page requires url\n"
        );
    }

    #[test]
    fn visit_page_keeps_full_file_but_caps_observation_at_100_lines() {
        let markdown: String = (1..=101).map(|line| format!("line-{line}\n")).collect();
        let mut web = FakeWeb {
            result: Ok(markdown.clone()),
        };

        let round = handle_round(&visit_call("https://example.com/long"), &mut web)
            .expect("long visit_page tool");
        let path = output_path(&round.observation);
        let observation = std::str::from_utf8(&round.observation).unwrap();

        assert_eq!(std::fs::read_to_string(path).unwrap(), markdown);
        assert!(observation.contains(&format!("<head -100 {path}>\n")));
        assert!(observation.contains("line-100\n</head>"));
        assert!(!observation.contains("line-101\n"));
        assert!(observation.contains("Use read path=<output_path>"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn visit_page_caps_head_at_8192_bytes() {
        let markdown = "x".repeat(WEB_HEAD_BYTES + 1);
        let mut web = FakeWeb {
            result: Ok(markdown.clone()),
        };

        let round = handle_round(&visit_call("https://example.com/wide"), &mut web)
            .expect("wide visit_page tool");
        let path = output_path(&round.observation);
        let marker = format!("<head -100 {path}>\n").into_bytes();
        let head_start = bytes_find(&round.observation, &marker).unwrap() + marker.len();
        let head_end = bytes_find(&round.observation[head_start..], b"\n</head>").unwrap();

        assert_eq!(head_end, WEB_HEAD_BYTES);
        assert_eq!(std::fs::read_to_string(path).unwrap(), markdown);
        std::fs::remove_file(path).unwrap();
    }

    fn read_call(path: Option<&str>, args: &[(&str, &str)]) -> Vec<u8> {
        let mut raw = String::from("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n");
        if let Some(path) = path {
            raw.push_str(&format!(
                "<｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n"
            ));
        }
        for (name, value) in args {
            raw.push_str(&format!(
                "<｜DSML｜parameter name=\"{name}\" string=\"false\">{value}</｜DSML｜parameter>\n"
            ));
        }
        raw.push_str("</｜DSML｜invoke>\n</｜DSML｜tool_calls>");
        raw.into_bytes()
    }

    fn read_twice_call(path: &str) -> Vec<u8> {
        format!(
            "<｜DSML｜tool_calls>\n\
             <｜DSML｜invoke name=\"read\">\n\
             <｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n\
             <｜DSML｜invoke name=\"read\">\n\
             <｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n\
             </｜DSML｜invoke>\n\
             </｜DSML｜tool_calls>"
        )
        .into_bytes()
    }

    fn oracle() -> PathBuf {
        if let Ok(path) = std::env::var("DS4_AGENT_C_ORACLE") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/agent_c_oracle")
    }

    fn c_read_bytes(path: Option<&str>, args: &[Option<&str>; 4]) -> Vec<u8> {
        let path = path.unwrap_or("-");
        let oracle = oracle();
        assert!(oracle.exists(), "build C oracle: make test-agent-parity");
        let output = std::process::Command::new(oracle)
            .args(["read", path])
            .args(args.iter().map(|arg| arg.unwrap_or("-")))
            .output()
            .expect("run C read oracle");
        assert!(
            output.status.success(),
            "C read oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let hex = std::str::from_utf8(&output.stdout)
            .expect("oracle hex UTF-8")
            .trim();
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).expect("hex pair")
            })
            .collect()
    }

    fn c_read_observation(path: Option<&str>, args: &[Option<&str>; 4]) -> Vec<u8> {
        let result = c_read_bytes(path, args);
        let mut observation = ResultBuf::default();
        observation.append(b"Tool result 1 (read):\n");
        observation.append(&result);
        if !result.is_empty() && !result.ends_with(b"\n") {
            observation.push(b'\n');
        }
        observation.into_vec()
    }

    fn c_read_twice_observation(path: &str) -> Vec<u8> {
        let output = std::process::Command::new(oracle())
            .args(["read2", path])
            .output()
            .expect("run C combined read oracle");
        assert!(output.status.success());
        std::str::from_utf8(&output.stdout)
            .expect("oracle hex UTF-8")
            .trim()
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn read_ranges_and_errors_match_c_oracle() {
        let path = write_temp("one\r\ntwo\nthree").expect("fixture");
        let path = path.to_str().expect("UTF-8 fixture path");
        let mut web = FakeWeb {
            result: Ok(String::new()),
        };
        let cases: &[(&[(&str, &str)], [Option<&str>; 4])] = &[
            (&[], [None, None, None, None]),
            (
                &[("start_line", "2"), ("max_lines", "1")],
                [Some("2"), Some("1"), None, None],
            ),
            (
                &[("start_line", "2"), ("max_lines", "1"), ("whole", "true")],
                [Some("2"), Some("1"), Some("true"), None],
            ),
            (
                &[("start_line", "1"), ("max_lines", "1"), ("raw", "yes")],
                [Some("1"), Some("1"), None, Some("yes")],
            ),
            (
                &[("start_line", "0"), ("max_lines", "0")],
                [Some("0"), Some("0"), None, None],
            ),
            (
                &[("start_line", "999999999999999999999999999999999999999x")],
                [
                    Some("999999999999999999999999999999999999999x"),
                    None,
                    None,
                    None,
                ],
            ),
        ];

        for (dsml_args, oracle_args) in cases {
            let round = handle_round(&read_call(Some(path), dsml_args), &mut web)
                .expect("read tool executes");
            assert_eq!(
                round.observation,
                c_read_observation(Some(path), oracle_args)
            );
        }

        let relative = PathBuf::from(format!(
            "ds4_agent_read_link_{}_{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::os::unix::fs::symlink(path, &relative).unwrap();
        let relative_text = relative.to_str().expect("UTF-8 relative path");
        let round = handle_round(&read_call(Some(relative_text), &[]), &mut web)
            .expect("relative symlink read");
        assert_eq!(
            round.observation,
            c_read_observation(Some(relative_text), &[None; 4])
        );
        std::fs::remove_file(relative).unwrap();

        let round = handle_round(&read_call(None, &[]), &mut web).expect("missing path result");
        assert_eq!(round.observation, c_read_observation(None, &[None; 4]));
        std::fs::remove_file(path).unwrap();

        let round = handle_round(&read_call(Some(path), &[]), &mut web).expect("open error result");
        assert_eq!(
            round.observation,
            c_read_observation(Some(path), &[None; 4])
        );

        let long_path = "x".repeat(300);
        let round = handle_round(&read_call(Some(&long_path), &[]), &mut web)
            .expect("long open error result");
        assert_eq!(
            round.observation,
            c_read_observation(Some(&long_path), &[None; 4])
        );
    }

    #[test]
    fn read_preserves_c_size_and_binary_boundaries() {
        let path = write_temp("").expect("fixture");
        let path_text = path.to_str().expect("UTF-8 fixture path");
        let mut web = FakeWeb {
            result: Ok(String::new()),
        };

        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((FILE_MAX_BYTES + 1) as u64)
            .unwrap();
        let round = handle_round(&read_call(Some(path_text), &[]), &mut web)
            .expect("oversized read result");
        assert_eq!(
            round.observation,
            c_read_observation(Some(path_text), &[None; 4])
        );

        let mut exact = File::create(&path).unwrap();
        let block = [b'x'; 8192];
        for _ in 0..FILE_MAX_BYTES / block.len() {
            exact.write_all(&block).unwrap();
        }
        drop(exact);
        let round = handle_round(&read_call(Some(path_text), &[]), &mut web)
            .expect("exact-limit read result");
        assert_eq!(round.observation.len(), RESULT_MAX_BYTES);
        assert_eq!(
            round.observation,
            c_read_observation(Some(path_text), &[None; 4])
        );
        let twice =
            handle_round(&read_twice_call(path_text), &mut web).expect("combined read result");
        assert_eq!(twice.observation.len(), RESULT_MAX_BYTES);
        assert_eq!(twice.observation, c_read_twice_observation(path_text));

        std::fs::write(&path, [0xff, b'\n']).unwrap();
        let round =
            handle_round(&read_call(Some(path_text), &[]), &mut web).expect("invalid UTF-8 result");
        assert_eq!(
            round.observation,
            c_read_observation(Some(path_text), &[None; 4])
        );

        std::fs::write(&path, [b'a', 0, b'b', b'\n']).unwrap();
        let round =
            handle_round(&read_call(Some(path_text), &[]), &mut web).expect("NUL read result");
        assert_eq!(
            round.observation,
            c_read_observation(Some(path_text), &[None; 4])
        );
        std::fs::remove_file(path).unwrap();
    }
}
