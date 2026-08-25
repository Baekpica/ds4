use super::{split_lines, LineSpan};

const CONTEXT_BEFORE: i32 = 5;
const CONTEXT_AFTER: i32 = 8;
const EDITED_HEAD: i32 = 18;
const EDITED_TAIL: i32 = 18;

pub(super) fn format_edit_result(
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
