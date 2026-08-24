//! C↔Rust incremental DSML tool-stream tapes (no model).

use ds4_server::tool_stream_dump_script;
use std::path::PathBuf;
use std::process::Command;

const DSML_TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";
const DSML_PARAM_START: &str = "<｜DSML｜parameter";

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_TOOL_STREAM_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/tool_stream_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/tool_stream_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_bytes(name: &str) -> Vec<u8> {
    let out = Command::new(require_oracle())
        .arg(name)
        .output()
        .expect("run tool_stream_c_oracle");
    assert!(
        out.status.success(),
        "oracle {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn assert_bytes_eq(label: &str, rust: &[u8], c: &[u8]) {
    if rust != c {
        let r = String::from_utf8_lossy(rust);
        let k = String::from_utf8_lossy(c);
        panic!("{label} mismatch\n--- rust ---\n{r}\n--- c ---\n{k}");
    }
}

fn assert_script(name: &str) -> String {
    let rust = tool_stream_dump_script(name);
    let c = c_bytes(name);
    assert_bytes_eq(name, &rust, &c);
    String::from_utf8_lossy(&rust).into_owned()
}

fn count_substr(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

#[test]
fn openai_partial_arguments_match_c() {
    let out = assert_script("openai-partial");
    let text = out.find("\"content\":\"Before.\"").expect("text");
    let tool = out.find("\"tool_calls\"").expect("tool");
    let key = out.find("\\\"command\\\":\\\"").expect("key");
    let partial = out.find("\"arguments\":\"echo partial\"").expect("partial");
    let rest = out.find("\"arguments\":\" done\"").expect("rest");
    assert!(text < tool && tool < partial && partial < rest);
    assert!(key < partial);
    assert_eq!(count_substr(&out, "\"id\":\"call_"), 1);
    assert!(out.contains("\"id\":\"call_0\""));
    assert!(!out.contains(DSML_TOOL_CALLS_START));
    assert!(!out.contains(DSML_PARAM_START));
    assert!(out.contains("data: [DONE]"));
}

#[test]
fn openai_raw_arguments_match_c() {
    let out = assert_script("openai-raw");
    assert!(out.contains("\"name\":\"edit\""));
    assert!(out.contains("\\\"edits\\\":"));
    assert!(out.contains("\"arguments\":\"[1,2,3\""));
    assert!(!out.contains(DSML_TOOL_CALLS_START));
}

#[test]
fn openai_waits_for_incomplete_tags_match_c() {
    let out = assert_script("openai-wait-tag");
    assert!(out.contains("\"name\":\"bash\""));
    assert!(!out.contains(DSML_PARAM_START));
}

#[test]
fn openai_holds_partial_dsml_entities_match_c() {
    let out = assert_script("openai-entity");
    assert!(out.contains("\"arguments\":\"echo \""));
    assert!(out.contains("\"arguments\":\"& done\""));
    assert!(!out.contains("&amp"));
}

#[test]
fn openai_think_then_tool_match_c() {
    let out = assert_script("openai-think-tool");
    let role = out.find("\"role\":\"assistant\"").expect("role");
    let thinking = out
        .find("\"reasoning_content\":\"need a tool\"")
        .expect("thinking");
    let text = out.find("\"content\":\"Hello.\"").expect("text");
    let tool = out.find("\"tool_calls\"").expect("tool");
    let done = out.find("data: [DONE]").expect("done");
    assert!(role < thinking && thinking < text && text < tool && tool < done);
    assert!(!out.contains(DSML_TOOL_CALLS_START));
    assert!(!out.contains("<think>"));
}

#[test]
fn openai_utf8_arguments_match_c() {
    let rust = tool_stream_dump_script("openai-utf8");
    let c = c_bytes("openai-utf8");
    assert_bytes_eq("openai-utf8", &rust, &c);
    let out = String::from_utf8_lossy(&rust);
    assert!(out.contains("\"arguments\":\"flag \""));
    assert!(rust.windows(4).any(|w| w == [0xf0, 0x9f, 0x9a, 0xa9]));
    assert!(!rust.windows(3).any(|w| w == [0xef, 0xbf, 0xbd]));
}

#[test]
fn openai_multiple_calls_match_c() {
    let out = assert_script("openai-multi");
    assert_eq!(count_substr(&out, "\"id\":\"call_"), 2);
    assert!(out.contains("\"name\":\"read\""));
    assert!(out.contains("\"name\":\"bash\""));
    assert!(out.contains("\\\"path\\\":"));
    assert!(out.contains("\\\"command\\\":"));
}

#[test]
fn anthropic_partial_tool_use_match_c() {
    let out = assert_script("anthropic-partial");
    let text = out.find("\"text\":\"Before.\"").expect("text");
    let tool = out.find("\"type\":\"tool_use\"").expect("tool");
    let key = out.find("\\\"command\\\":\\\"").expect("key");
    let partial = out
        .find("\"partial_json\":\"echo partial\"")
        .expect("partial");
    let rest = out.find("\"partial_json\":\" done\"").expect("rest");
    let stop = out.find("event: message_stop").expect("stop");
    assert!(text < tool && tool < key && key < partial && partial < rest && rest < stop);
    assert_eq!(count_substr(&out, "\"type\":\"tool_use\""), 1);
    assert!(out.contains("\"id\":\"toolu_0\""));
    assert!(!out.contains(DSML_TOOL_CALLS_START));
    assert!(!out.contains(DSML_PARAM_START));
}

#[test]
fn anthropic_think_then_tool_match_c() {
    let out = assert_script("anthropic-think-tool");
    let start = out.find("event: message_start").expect("start");
    let thinking = out.find("\"thinking\":\"need a tool\"").expect("thinking");
    let signature = out.find("\"type\":\"signature_delta\"").expect("sig");
    let text = out.find("\"text\":\"Hello.\"").expect("text");
    let tool = out.find("\"type\":\"tool_use\"").expect("tool");
    let stop = out.find("event: message_stop").expect("stop");
    assert!(start < thinking && thinking < signature && signature < text && text < tool && tool < stop);
    assert!(!out.contains(DSML_TOOL_CALLS_START));
}
