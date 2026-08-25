use super::TOOL_UNSUPPORTED_ERROR;

const DSML_OPEN: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const DSML_OPEN_MISSING_BAR: &[u8] = "<DSML｜tool_calls>".as_bytes();
const DSML_CLOSE_PREFIX: &str = "</｜DSML｜";
const GOOGLE_SEARCH: &str = "google_search";
const QUERY_ARG: &str = "query";

pub(super) struct ToolRound {
    pub(super) visible: Vec<u8>,
    pub(super) observation: String,
}

pub(super) trait Browser {
    fn google_search(&mut self, query: &str) -> Result<String, String>;
}

impl Browser for ds4_web::Web {
    fn google_search(&mut self, query: &str) -> Result<String, String> {
        ds4_web::Web::google_search(self, query)
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

pub(super) fn handle_round<B: Browser>(raw: &[u8], web: &mut B) -> Result<ToolRound, String> {
    let (open_at, open_len) = start(raw).ok_or_else(|| "missing DSML tool call".to_string())?;
    let calls = parse_calls(raw, open_at, open_len)?;
    let mut observation = String::new();
    if calls.is_empty() {
        observation.push_str("Tool error: empty tool call block\n");
    }
    for (index, call) in calls.iter().enumerate() {
        if call.name != GOOGLE_SEARCH {
            return Err(TOOL_UNSUPPORTED_ERROR.into());
        }
        let result = match call.arg(QUERY_ARG) {
            None | Some("") => format!("Tool error: {GOOGLE_SEARCH} requires {QUERY_ARG}\n"),
            Some(query) => match web.google_search(query) {
                Ok(markdown) => markdown,
                Err(error) => format!("Tool error: {GOOGLE_SEARCH} failed: {error}\n"),
            },
        };
        observation.push_str(&format!("Tool result {} ({GOOGLE_SEARCH}):\n", index + 1));
        observation.push_str(&result);
        if !result.is_empty() && !result.ends_with('\n') {
            observation.push('\n');
        }
    }
    Ok(ToolRound {
        visible: raw[..open_at].to_vec(),
        observation,
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
            round.observation,
            "Tool result 1 (google_search):\n# Results\n- [DS4](https://example.com)\n"
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
                .observation,
            "Tool result 1 (google_search):\nTool error: google_search failed: Chrome unavailable\n"
        );

        let mut web = FakeWeb {
            result: Ok(String::new()),
        };
        assert_eq!(
            handle_round(&call(""), &mut web)
                .expect("valid DSML")
                .observation,
            "Tool result 1 (google_search):\nTool error: google_search requires query\n"
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
                raw.splice(at..at + "google_search".len(), b"read".iter().copied());
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
                .observation,
            "Tool result 1 (google_search):\nresult\n"
        );
    }
}
