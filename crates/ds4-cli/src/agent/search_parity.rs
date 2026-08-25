use super::search::{default_search_path, search_result, SearchArgs, SearchOutcome, SEARCH};
use super::web_tools::{handle_round_with_cursor, Browser, ReadCursor};
use super::TOOL_UNSUPPORTED_ERROR;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct NoWeb;

impl Browser for NoWeb {
    fn google_search(&mut self, _query: &str) -> Result<String, String> {
        Err("unused".into())
    }

    fn visit_page(&mut self, _url: &str) -> Result<String, String> {
        Err("unused".into())
    }
}

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn create() -> Self {
        let root = PathBuf::from(format!(
            "/tmp/ds4_agent_search_{}_{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("hit.txt"), b"alpha\nneedle here\nomega\n").unwrap();
        std::fs::write(root.join("miss.txt"), b"nothing here\n").unwrap();
        std::fs::write(root.join(".hidden"), b"needle hidden\n").unwrap();
        std::fs::write(root.join(".git").join("config"), b"needle in git\n").unwrap();
        std::fs::write(root.join("bin.dat"), b"needle\0binary\n").unwrap();
        std::fs::write(root.join("sub").join("code.rs"), b"needle rust\n").unwrap();
        std::fs::write(
            root.join("context.txt"),
            b"keep\nneedle\nmid\nneedle\ntail\n",
        )
        .unwrap();
        let mut many = Vec::new();
        for index in 0..10 {
            many.extend_from_slice(format!("needle {index}\n").as_bytes());
        }
        std::fs::write(root.join("many.txt"), many).unwrap();
        std::fs::write(root.join("Case.txt"), b"Needle\n").unwrap();
        Self { root }
    }

    fn path(&self) -> &str {
        self.root.to_str().expect("UTF-8 search fixture")
    }

    fn child(&self, name: &str) -> String {
        self.root
            .join(name)
            .to_str()
            .expect("UTF-8 child")
            .to_string()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn oracle() -> PathBuf {
    if let Ok(path) = std::env::var("DS4_AGENT_C_ORACLE") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/agent_c_oracle")
}

fn decode_oracle_hex(stdout: &[u8]) -> Vec<u8> {
    std::str::from_utf8(stdout)
        .expect("oracle hex UTF-8")
        .trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn c_search(args: &[&str]) -> Vec<u8> {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build C oracle: make tests/parity/agent_c_oracle"
    );
    let mut command = std::process::Command::new(&oracle);
    command.arg("search");
    command.args(args);
    let output = command.output().expect("run C agent oracle");
    assert!(
        output.status.success(),
        "C agent oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    decode_oracle_hex(&output.stdout)
}

fn c_search_in(cwd: &Path, args: &[&str]) -> Vec<u8> {
    let oracle = oracle();
    assert!(
        oracle.exists(),
        "build C oracle: make tests/parity/agent_c_oracle"
    );
    let mut command = std::process::Command::new(&oracle);
    command.current_dir(cwd);
    command.arg("search");
    command.args(args);
    let output = command.output().expect("run C agent oracle");
    assert!(
        output.status.success(),
        "C agent oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    decode_oracle_hex(&output.stdout)
}

fn rust_search(args: SearchArgs<'_>) -> Vec<u8> {
    match search_result(args) {
        SearchOutcome::Output(bytes) => bytes,
        SearchOutcome::Unsupported => panic!("literal search must not be unsupported"),
    }
}

fn args<'a>(
    query: Option<&'a str>,
    path: Option<&'a str>,
    mode: Option<&'a str>,
    glob: Option<&'a str>,
    case_sensitive: Option<&'a str>,
    context: Option<&'a str>,
    max_results: Option<&'a str>,
) -> SearchArgs<'a> {
    SearchArgs {
        query,
        path,
        mode,
        glob,
        case_sensitive,
        context,
        max_results,
    }
}

fn search_dsml(query: Option<&str>, path: Option<&str>, mode: Option<&str>) -> Vec<u8> {
    let mut params = String::new();
    if let Some(query) = query {
        params.push_str(&format!(
            "<｜DSML｜parameter name=\"query\" string=\"true\">{query}</｜DSML｜parameter>\n"
        ));
    }
    if let Some(path) = path {
        params.push_str(&format!(
            "<｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n"
        ));
    }
    if let Some(mode) = mode {
        params.push_str(&format!(
            "<｜DSML｜parameter name=\"mode\" string=\"true\">{mode}</｜DSML｜parameter>\n"
        ));
    }
    format!(
        "<｜DSML｜tool_calls>\n\
         <｜DSML｜invoke name=\"{SEARCH}\">\n\
         {params}\
         </｜DSML｜invoke>\n\
         </｜DSML｜tool_calls>"
    )
    .into_bytes()
}

fn observation(result: &[u8]) -> Vec<u8> {
    let mut out = b"Tool result 1 (search):\n".to_vec();
    out.extend_from_slice(result);
    if !result.is_empty() && !result.ends_with(b"\n") {
        out.push(b'\n');
    }
    out
}

#[test]
fn search_requires_query_when_missing_or_empty() {
    let expected = b"Tool error: search requires query\n".as_slice();
    assert_eq!(
        rust_search(args(None, None, None, None, None, None, None)),
        expected
    );
    assert_eq!(
        rust_search(args(Some(""), Some("/tmp"), None, None, None, None, None)),
        expected
    );
    assert_eq!(c_search(&[]), expected);
    assert_eq!(c_search(&["-"]), expected);
    assert_eq!(c_search(&[""]), expected);

    let listed = handle_round_with_cursor(
        &search_dsml(None, Some("/tmp"), None),
        &mut NoWeb,
        &mut ReadCursor::default(),
    )
    .expect("missing query is a tool error, not unsupported");
    assert_eq!(listed.observation, observation(expected));
}

#[test]
fn search_literal_hit_and_miss_match_c_oracle() {
    let fixture = Fixture::create();
    let hit = fixture.child("hit.txt");
    let miss = fixture.child("miss.txt");

    let rust_hit = rust_search(args(
        Some("needle"),
        Some(&hit),
        None,
        None,
        None,
        None,
        None,
    ));
    assert_eq!(rust_hit, c_search(&["needle", &hit]));
    assert!(rust_hit.starts_with(b"1 match shown\n\n"));
    assert!(std::str::from_utf8(&rust_hit)
        .unwrap()
        .contains("needle here"));

    let rust_miss = rust_search(args(
        Some("needle"),
        Some(&miss),
        None,
        None,
        None,
        None,
        None,
    ));
    assert_eq!(rust_miss, b"No matches\n");
    assert_eq!(rust_miss, c_search(&["needle", &miss]));
}

#[test]
fn search_skips_git_dir_and_searches_other_hidden_files() {
    let fixture = Fixture::create();
    let root = fixture.path();
    let rust = rust_search(args(
        Some("needle"),
        Some(root),
        None,
        None,
        None,
        None,
        None,
    ));
    assert_eq!(rust, c_search(&["needle", root]));
    let text = String::from_utf8_lossy(&rust);
    assert!(text.contains(".hidden"));
    assert!(text.contains("needle hidden"));
    assert!(!text.contains(".git"));
    assert!(!text.contains("needle in git"));
}

#[test]
fn search_skips_nul_binary_files() {
    let fixture = Fixture::create();
    let binary = fixture.child("bin.dat");
    let rust = rust_search(args(
        Some("needle"),
        Some(&binary),
        None,
        None,
        None,
        None,
        None,
    ));
    assert_eq!(rust, b"No matches\n");
    assert_eq!(rust, c_search(&["needle", &binary]));
}

#[test]
fn search_glob_filters_by_basename() {
    let fixture = Fixture::create();
    let root = fixture.path();
    let rust = rust_search(args(
        Some("needle"),
        Some(root),
        None,
        Some("*.txt"),
        None,
        None,
        None,
    ));
    assert_eq!(rust, c_search(&["needle", root, "-", "*.txt"]));
    let text = String::from_utf8_lossy(&rust);
    assert!(text.contains("hit.txt"));
    assert!(text.contains("context.txt"));
    assert!(text.contains("many.txt"));
    assert!(!text.contains("code.rs"));
    assert!(!text.contains(".hidden"));
}

#[test]
fn search_case_insensitive_literal_matches_c_oracle() {
    let fixture = Fixture::create();
    let case_file = fixture.child("Case.txt");
    let rust = rust_search(args(
        Some("NEEDLE"),
        Some(&case_file),
        None,
        None,
        Some("false"),
        None,
        None,
    ));
    assert_eq!(rust, c_search(&["NEEDLE", &case_file, "-", "-", "false"]));
    assert!(std::str::from_utf8(&rust).unwrap().contains("Needle"));
}

#[test]
fn search_context_one_coalesces_adjacent_matches() {
    let fixture = Fixture::create();
    let context = fixture.child("context.txt");
    let rust = rust_search(args(
        Some("needle"),
        Some(&context),
        None,
        None,
        None,
        Some("1"),
        None,
    ));
    assert_eq!(rust, c_search(&["needle", &context, "-", "-", "-", "1"]));
    let mut expected = format!("2 matches shown\n\n{context}\n").into_bytes();
    expected.extend_from_slice(b"  1 keep\n  2 needle\n  3 mid\n  4 needle\n  5 tail\n\n");
    assert_eq!(rust, expected);
}

#[test]
fn search_max_results_caps_matching_lines() {
    let fixture = Fixture::create();
    let many = fixture.child("many.txt");
    let rust = rust_search(args(
        Some("needle"),
        Some(&many),
        None,
        None,
        None,
        None,
        Some("3"),
    ));
    assert_eq!(rust, c_search(&["needle", &many, "-", "-", "-", "-", "3"]));
    assert!(rust.starts_with(b"3 matches shown\n\n"));
    let text = String::from_utf8_lossy(&rust);
    assert!(text.contains("needle 0"));
    assert!(text.contains("needle 2"));
    assert!(!text.contains("needle 3"));
}

#[test]
fn search_empty_and_missing_path_both_mean_dot() {
    assert_eq!(default_search_path(None), ".");
    assert_eq!(default_search_path(Some("")), ".");
    assert_eq!(default_search_path(Some("/tmp/x")), "/tmp/x");

    let fixture = Fixture::create();
    let omitted = c_search_in(&fixture.root, &["needle"]);
    let dash = c_search_in(&fixture.root, &["needle", "-"]);
    let empty = c_search_in(&fixture.root, &["needle", ""]);
    assert_eq!(omitted, dash);
    assert_eq!(dash, empty);

    let rust = rust_search(args(
        Some("needle"),
        Some(fixture.path()),
        None,
        None,
        None,
        None,
        None,
    ));
    assert_eq!(rust, c_search(&["needle", fixture.path()]));
}

#[test]
fn search_regex_mode_is_still_unsupported() {
    let fixture = Fixture::create();
    let path = fixture.path();
    match search_result(args(
        Some("needle"),
        Some(path),
        Some("regex"),
        None,
        None,
        None,
        None,
    )) {
        SearchOutcome::Unsupported => {}
        SearchOutcome::Output(bytes) => panic!("regex should stay unsupported, got {bytes:?}"),
    }
    let error = match handle_round_with_cursor(
        &search_dsml(Some("needle"), Some(path), Some("regex")),
        &mut NoWeb,
        &mut ReadCursor::default(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("regex mode stays on the unsupported tool path"),
    };
    assert_eq!(error, TOOL_UNSUPPORTED_ERROR);
}
