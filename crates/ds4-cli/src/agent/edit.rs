use super::web_tools::{io_detail, read_bytes};
use std::fs::File;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;

pub(crate) const EDIT: &str = "edit";
const UPTO: &[u8] = b"[upto]";
const CONTEXT_BEFORE: i32 = 5;
const CONTEXT_AFTER: i32 = 8;
const EDITED_HEAD: i32 = 18;
const EDITED_TAIL: i32 = 18;

struct LineSpan {
    start: usize,
    content_end: usize,
    end: usize,
}

pub(crate) fn edit_result(path: Option<&str>, old: Option<&str>, new: Option<&str>) -> Vec<u8> {
    let Some(path) = path.filter(|path| !path.is_empty()) else {
        return b"Tool error: edit requires path\n".to_vec();
    };
    let Some(old) = old.filter(|old| !old.is_empty()) else {
        return b"Tool error: edit requires non-empty old text\n".to_vec();
    };
    let Some(new) = new else {
        return b"Tool error: edit requires new text\n".to_vec();
    };
    let data = match read_bytes(path) {
        Ok(data) => data,
        Err(error) => return wrap_error(&error),
    };
    let (offset, remove_len, anchored) = match find_old_span(&data, old.as_bytes()) {
        Ok(found) => found,
        Err(error) => return wrap_error(error.as_bytes()),
    };
    apply_splice(path, &data, offset, remove_len, new.as_bytes(), anchored)
}

fn wrap_error(detail: &[u8]) -> Vec<u8> {
    let mut out = b"Tool error: ".to_vec();
    out.extend_from_slice(detail);
    out.push(b'\n');
    out
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len())
        .position(|window| window == needle)
}

fn find_unique(data: &[u8], needle: &[u8], label: &str) -> Result<usize, String> {
    if needle.is_empty() {
        return Err(format!("{label} anchor is empty"));
    }
    let Some(first) = find_bytes(data, needle) else {
        return Err(format!("{label} anchor not found"));
    };
    let rest = first.saturating_add(1);
    if rest <= data.len() && find_bytes(&data[rest..], needle).is_some() {
        return Err(format!("{label} anchor is not unique"));
    }
    Ok(first)
}

fn find_unique_after(
    data: &[u8],
    start: usize,
    needle: &[u8],
    label: &str,
) -> Result<usize, String> {
    if needle.is_empty() {
        return Err(format!("{label} anchor is empty"));
    }
    if start > data.len() {
        return Err(format!("{label} search starts outside file"));
    }
    let Some(rel) = find_bytes(&data[start..], needle) else {
        return Err(format!("{label} anchor not found after old head"));
    };
    let first = start + rel;
    let rest = first.saturating_add(1);
    if rest <= data.len() && find_bytes(&data[rest..], needle).is_some() {
        return Err(format!("{label} anchor is not unique after old head"));
    }
    Ok(first)
}

fn find_old_span(data: &[u8], old: &[u8]) -> Result<(usize, usize, bool), String> {
    match find_bytes(old, UPTO) {
        None => {
            let at = find_unique(data, old, "old text")?;
            Ok((at, old.len(), false))
        }
        Some(upto) => {
            if find_bytes(&old[upto + UPTO.len()..], UPTO).is_some() {
                return Err("old text contains more than one [upto] marker".into());
            }
            let head = &old[..upto];
            let tail = &old[upto + UPTO.len()..];
            if !tail.iter().any(|byte| !byte.is_ascii_whitespace()) {
                return Err("old text after [upto] must include a unique tail anchor".into());
            }
            let head_at = find_unique(data, head, "old head")?;
            let tail_at = find_unique_after(data, head_at + head.len(), tail, "old tail")?;
            Ok((head_at, tail_at - head_at + tail.len(), true))
        }
    }
}

fn split_lines(data: &[u8]) -> Vec<LineSpan> {
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let start = pos;
        while pos < data.len() && !matches!(data[pos], b'\n' | b'\r') {
            pos += 1;
        }
        let content_end = pos;
        if pos < data.len() {
            if data[pos] == b'\r' && data.get(pos + 1) == Some(&b'\n') {
                pos += 2;
            } else {
                pos += 1;
            }
        }
        spans.push(LineSpan {
            start,
            content_end,
            end: pos,
        });
    }
    spans
}

fn line_for_offset(spans: &[LineSpan], offset: usize) -> i32 {
    if spans.is_empty() {
        return 1;
    }
    for (index, span) in spans.iter().enumerate() {
        if offset < span.end {
            return (index + 1) as i32;
        }
    }
    spans.len() as i32
}

fn write_bytes(path: &str, data: &[u8]) -> Result<(), String> {
    let os_path = std::ffi::OsStr::from_bytes(path.as_bytes());
    let mut file =
        File::create(os_path).map_err(|error| format!("open {path}: {}", io_detail(&error)))?;
    file.write_all(data)
        .and_then(|_| file.flush())
        .map_err(|error| format!("write {path}: {}", io_detail(&error)))
}

fn apply_splice(
    path: &str,
    data: &[u8],
    offset: usize,
    remove_len: usize,
    insert: &[u8],
    anchored: bool,
) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(offset + insert.len() + data.len().saturating_sub(offset + remove_len));
    out.extend_from_slice(&data[..offset]);
    out.extend_from_slice(insert);
    out.extend_from_slice(&data[offset + remove_len..]);
    if let Err(error) = write_bytes(path, &out) {
        return wrap_error(error.as_bytes());
    }
    let old_spans = split_lines(data);
    let new_spans = split_lines(&out);
    let kind = if anchored {
        "anchored old/new replacement"
    } else {
        "old/new replacement"
    };
    if old_spans.is_empty() {
        return format!("Edited {path} using {kind}\n").into_bytes();
    }
    let old_last = if remove_len > 0 {
        offset + remove_len - 1
    } else {
        offset
    }
    .min(data.len().saturating_sub(1));
    let start_line = line_for_offset(&old_spans, offset);
    let end_line = line_for_offset(&old_spans, old_last);
    let delta = new_spans.len() as i32 - old_spans.len() as i32;
    format_edit_result(path, start_line, end_line, delta, &out, kind)
}

fn format_edit_result(
    path: &str,
    start_line: i32,
    end_line: i32,
    delta: i32,
    new_data: &[u8],
    kind: &str,
) -> Vec<u8> {
    let mut out = format!("Edited {path} using {kind}\n").into_bytes();
    if start_line > 0 && end_line >= start_line {
        out.extend_from_slice(
            format!(
                "Touched old lines {start_line}-{end_line}; current post-edit context follows.\n"
            )
            .as_bytes(),
        );
        if delta != 0 {
            out.extend_from_slice(
                format!(
                    "Line shift: old lines after {end_line} moved by {delta:+} (old line {} is now line {}). Re-read before relying on old line numbers there.\n",
                    end_line + 1,
                    end_line + 1 + delta
                )
                .as_bytes(),
            );
        }
        let mut anchor_end = end_line + delta;
        if anchor_end < start_line {
            anchor_end = start_line;
        }
        append_context(&mut out, path, new_data, start_line, anchor_end);
    }
    out
}

fn append_context(out: &mut Vec<u8>, path: &str, data: &[u8], mut start: i32, mut end: i32) {
    let spans = split_lines(data);
    if spans.is_empty() {
        return;
    }
    let n = spans.len() as i32;
    start = start.clamp(1, n);
    end = end.clamp(start, n);
    let ctx_start = (start - CONTEXT_BEFORE).max(1);
    let ctx_end = (end + CONTEXT_AFTER).min(n);
    out.extend_from_slice(
        format!("Current file around edit: {path} lines {ctx_start}-{ctx_end} of {n}\n").as_bytes(),
    );
    let edited = end - start + 1;
    if edited <= EDITED_HEAD + EDITED_TAIL {
        append_lines(out, data, &spans, ctx_start, ctx_end);
        return;
    }
    let head_end = start + EDITED_HEAD - 1;
    let tail_start = end - EDITED_TAIL + 1;
    append_lines(out, data, &spans, ctx_start, head_end);
    out.extend_from_slice(
        format!(
            "... {} edited lines omitted ...\n",
            tail_start - head_end - 1
        )
        .as_bytes(),
    );
    append_lines(out, data, &spans, tail_start, ctx_end);
}

fn append_lines(out: &mut Vec<u8>, data: &[u8], spans: &[LineSpan], from: i32, to: i32) {
    for line in from..=to {
        let span = &spans[(line - 1) as usize];
        out.extend_from_slice(format!("{line} ").as_bytes());
        out.extend_from_slice(&data[span.start..span.content_end]);
        out.push(b'\n');
    }
}
