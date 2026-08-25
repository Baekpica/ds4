mod scan;

use super::web_tools::parse_bool;
use scan::{
    clamp_usize, glob_match, literal_match, parse_int_default, read_file, split_lines, BodyBuf,
};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;

const SEARCH_MAX_DEPTH: i32 = 24;
const CONTEXT_MAX: i32 = 5;
const MAX_RESULTS_DEFAULT: i32 = 50;
const MAX_RESULTS_CAP: i32 = 500;

pub(crate) const SEARCH: &str = "search";

pub(crate) struct SearchArgs<'a> {
    pub query: Option<&'a str>,
    pub path: Option<&'a str>,
    pub mode: Option<&'a str>,
    pub glob: Option<&'a str>,
    pub case_sensitive: Option<&'a str>,
    pub context: Option<&'a str>,
    pub max_results: Option<&'a str>,
}

pub(crate) enum SearchOutcome {
    Output(Vec<u8>),
    Unsupported,
}

struct SearchCtx<'a> {
    query: &'a [u8],
    glob: Option<&'a [u8]>,
    case_sensitive: bool,
    context: usize,
    max_results: usize,
    results: usize,
    out: BodyBuf,
}

pub(crate) fn default_search_path(path: Option<&str>) -> &str {
    path.filter(|path| !path.is_empty()).unwrap_or(".")
}

pub(crate) fn search_result(args: SearchArgs<'_>) -> SearchOutcome {
    let Some(query) = args.query.filter(|query| !query.is_empty()) else {
        return SearchOutcome::Output(b"Tool error: search requires query\n".to_vec());
    };
    if args.mode == Some("regex") {
        return SearchOutcome::Unsupported;
    }
    let path = default_search_path(args.path);
    let mut ctx = SearchCtx {
        query: query.as_bytes(),
        glob: args.glob.filter(|glob| !glob.is_empty()).map(str::as_bytes),
        case_sensitive: parse_bool(args.case_sensitive, true),
        context: clamp_usize(parse_int_default(args.context, 0, 0, CONTEXT_MAX)),
        max_results: clamp_usize(parse_int_default(
            args.max_results,
            MAX_RESULTS_DEFAULT,
            1,
            MAX_RESULTS_CAP,
        )),
        results: 0,
        out: BodyBuf::new(),
    };
    search_path(&mut ctx, path.as_bytes(), 0);
    SearchOutcome::Output(finish_search(ctx))
}

fn finish_search(ctx: SearchCtx<'_>) -> Vec<u8> {
    if ctx.out.bytes.is_empty() {
        return b"No matches\n".to_vec();
    }
    let plural = if ctx.results == 1 { "" } else { "es" };
    let mut out = format!("{} match{plural} shown\n\n", ctx.results).into_bytes();
    out.extend_from_slice(&ctx.out.bytes);
    out
}

fn search_path(ctx: &mut SearchCtx<'_>, path: &[u8], depth: i32) {
    if ctx.results >= ctx.max_results || depth > SEARCH_MAX_DEPTH {
        return;
    }
    let os_path = OsStr::from_bytes(path);
    let meta = match std::fs::symlink_metadata(os_path) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    if meta.file_type().is_file() {
        search_file(ctx, path);
        return;
    }
    if !meta.file_type().is_dir() {
        return;
    }
    let dir = match std::fs::read_dir(os_path) {
        Ok(dir) => dir,
        Err(_) => return,
    };
    for entry in dir {
        if ctx.results >= ctx.max_results {
            break;
        }
        let Ok(entry) = entry else {
            break;
        };
        let name = entry.file_name();
        let name = name.as_bytes();
        if name == b"." || name == b".." || name == b".git" {
            continue;
        }
        let mut child = Vec::with_capacity(path.len() + 1 + name.len());
        child.extend_from_slice(path);
        child.push(b'/');
        child.extend_from_slice(name);
        search_path(ctx, &child, depth + 1);
    }
}

fn search_file(ctx: &mut SearchCtx<'_>, path: &[u8]) {
    if ctx.results >= ctx.max_results {
        return;
    }
    if let Some(glob) = ctx.glob {
        let base = match path.iter().rposition(|byte| *byte == b'/') {
            Some(at) => &path[at + 1..],
            None => path,
        };
        if !glob_match(glob, base) && !glob_match(glob, path) {
            return;
        }
    }
    let Some(data) = read_file(path) else {
        return;
    };
    if data.contains(&0) {
        return;
    }
    let spans = split_lines(&data);
    let mut printed_file = false;
    let mut last_context_line: i32 = -1;
    let mut index = 0i32;
    let span_len = i32::try_from(spans.len()).unwrap_or(i32::MAX);
    let context = i32::try_from(ctx.context).unwrap_or(CONTEXT_MAX);
    while index < span_len && ctx.results < ctx.max_results {
        let Some((start, content_end)) = spans.get(clamp_usize(index)).copied() else {
            break;
        };
        if !literal_match(&data[start..content_end], ctx.query, ctx.case_sensitive) {
            index += 1;
            continue;
        }
        if !printed_file {
            ctx.out.append(path);
            ctx.out.append(b"\n");
            printed_file = true;
        }
        let mut from = index - context;
        let mut to = index + context;
        if from < 0 {
            from = 0;
        }
        if to >= span_len {
            to = span_len - 1;
        }
        if from <= last_context_line {
            from = last_context_line + 1;
        }
        let mut line = from;
        while line <= to {
            emit_line(ctx, &data, spans[clamp_usize(line)], clamp_usize(line) + 1);
            last_context_line = line;
            line += 1;
        }
        ctx.results += 1;
        index += 1;
    }
    if printed_file {
        ctx.out.append(b"\n");
    }
}

fn emit_line(ctx: &mut SearchCtx<'_>, data: &[u8], span: (usize, usize), line_no: usize) {
    ctx.out.append(format!("  {line_no} ").as_bytes());
    ctx.out.append(&data[span.0..span.1]);
    ctx.out.append(b"\n");
}
