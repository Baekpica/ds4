//! Generated-message parse + SemAccum goldens from `ds4_server.c` unit tests.

use ds4_server::{
    append_tool_calls_json, assign_tool_ids, parse_generated_for_response, parse_generated_message,
    ChatFormat, ModelSyntax, SemAccum, ToolCall, ToolSchemaOrder,
};

const DSML_START: &str = "<｜DSML｜tool_calls>";
const DSML_END: &str = "</｜DSML｜tool_calls>";
const DSML_INVOKE: &str = "<｜DSML｜invoke";
const DSML_INVOKE_END: &str = "</｜DSML｜invoke>";
const DSML_PARAM: &str = "<｜DSML｜parameter";
const DSML_PARAM_END: &str = "</｜DSML｜parameter>";

#[test]
fn parse_dots3_tool_call_message() {
    let text = b"<think>\nplan\n</think>\n\nChecking now.\n\
        <dots_function_call>\n\
        <invoke name=\"get_weather\">\n\
        <parameter name=\"city\">\nSeoul\n</parameter>\n\
        <parameter name=\"days\">\n3\n</parameter>\n\
        </invoke>\n\
        </dots_function_call>";
    let p = parse_generated_message(ModelSyntax::Dots3, text, true, ChatFormat::DeepSeek, &[]);
    assert!(p.ok);
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, "{\"city\": \"Seoul\", \"days\": 3}");
    assert_eq!(p.content, b"\n\nChecking now.");
    assert_eq!(p.reasoning, b"\nplan\n");

    let multi = b"</think>\n\n<dots_function_call>\n\
        <invoke name=\"a\">\n<parameter name=\"x\">\n1\n</parameter>\n</invoke>\n\
        <invoke name=\"b\">\n<parameter name=\"y\">\ntwo\n</parameter>\n</invoke>\n\
        </dots_function_call>";
    let p = parse_generated_message(ModelSyntax::Dots3, multi, true, ChatFormat::DeepSeek, &[]);
    assert!(p.ok);
    assert_eq!(p.calls.len(), 2);
    assert_eq!(p.calls[0].arguments, "{\"x\": 1}");
    assert_eq!(p.calls[1].arguments, "{\"y\": \"two\"}");
}

#[test]
fn parse_motif3_tool_call_message() {
    let text = b"<think>need weather</think>\n\
        <tool_call>{\"name\": \"get_weather\", \"arguments\": \
        {\"city\": \"Seoul\"}}</tool_call>";
    let p = parse_generated_message(ModelSyntax::Motif3, text, true, ChatFormat::DeepSeek, &[]);
    assert!(p.ok);
    assert_eq!(p.reasoning, b"need weather");
    assert!(p.content.is_empty());
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "get_weather");
    assert!(p.calls[0].arguments.contains("\"city\""));
    assert!(p.raw_tool_text.starts_with("\n<tool_call>"));
}

#[test]
fn parse_exaone_two_hermes_calls() {
    let text = b"<think>need weather</think>\n\n\
        <tool_call>{\"name\":\"get_weather\",\"arguments\":\
        {\"city\":\"Seoul\"}}</tool_call>\n\
        <tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>";
    let p = parse_generated_message(ModelSyntax::Exaone, text, true, ChatFormat::Exaone, &[]);
    assert!(p.ok);
    assert_eq!(p.reasoning, b"need weather");
    assert!(p.content.is_empty());
    assert_eq!(p.calls.len(), 2);
    assert_eq!(p.calls[0].name, "get_weather");
    assert_eq!(p.calls[0].arguments, "{\"city\":\"Seoul\"}");
    assert_eq!(p.calls[1].name, "get_time");
}

#[test]
fn parse_dsml_nested_parameters() {
    let generated = format!(
        "review done\n\n{DSML_START}\n{DSML_INVOKE} name=\"edit\">\n\
         {DSML_PARAM} name=\"path\">/private/tmp/tetris.c{DSML_PARAM_END}\n\
         {DSML_PARAM} name=\"edits\">\n\
         {DSML_PARAM} name=\"oldText\" string=\"true\">old &lt;text&gt;{DSML_PARAM_END}\n\
         {DSML_PARAM} name=\"newText\" string=\"true\">new text{DSML_PARAM_END}\n\
         {DSML_INVOKE_END}\n{DSML_END}"
    );
    let p = parse_generated_message(
        ModelSyntax::DeepSeek,
        generated.as_bytes(),
        false,
        ChatFormat::DeepSeek,
        &[],
    );
    assert!(p.ok);
    assert_eq!(p.content, b"review done");
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "edit");
    assert!(p.calls[0].arguments.contains("\"path\": \"/private/tmp/tetris.c\""));
    assert!(p.calls[0].arguments.contains("\"edits\": {"));
    assert!(
        p.calls[0].arguments.contains("\"oldText\":\"old <text>\""),
        "{}",
        p.calls[0].arguments
    );
    assert!(p.calls[0].arguments.contains("\"newText\":\"new text\""));
}

#[test]
fn parse_solar_native_tool_call() {
    let generated = format!(
        "I should inspect the directory.<|think:end|><|tool_call:start|>list_files\n\
         <|tool_arg:start|>path<|tool_arg:value|>/tmp<|tool_arg:end|>\n\
         <|tool_arg:start|>recursive<|tool_arg:value|>false<|tool_arg:end|>\n\
         <|tool_arg:start|>literal<|tool_arg:value|>false<|tool_arg:end|>\n\
         <|tool_call:end|>"
    );
    let orders = [ToolSchemaOrder {
        name: "list_files".into(),
        prop: vec!["path".into(), "recursive".into(), "literal".into()],
        prop_type: vec!["string".into(), "boolean".into(), "string".into()],
        ..Default::default()
    }];
    let p = parse_generated_message(
        ModelSyntax::SolarOpen2,
        generated.as_bytes(),
        true,
        ChatFormat::SolarOpen2,
        &orders,
    );
    assert!(p.ok);
    assert_eq!(p.calls.len(), 1);
    assert_eq!(p.calls[0].name, "list_files");
    assert!(p.calls[0].arguments.contains("\"path\": \"/tmp\""));
    assert!(p.calls[0].arguments.contains("\"recursive\": false"));
    assert!(p.calls[0].arguments.contains("\"literal\": \"false\""));
    assert!(p.content.is_empty());
    assert_eq!(p.reasoning, b"I should inspect the directory.");
    assert!(p.raw_dsml.starts_with("<|tool_call:start|>"));
}

#[test]
fn no_tools_does_not_extract_calls() {
    let text = format!(
        "hi\n\n{DSML_START}\n{DSML_INVOKE} name=\"bash\">\n\
         {DSML_PARAM} name=\"command\" string=\"true\">ls{DSML_PARAM_END}\n\
         {DSML_INVOKE_END}\n{DSML_END}"
    );
    let (p, finish) = parse_generated_for_response(
        ModelSyntax::DeepSeek,
        text.as_bytes(),
        false,
        true,
        false,
        ChatFormat::DeepSeek,
        &[],
        "stop",
    );
    assert!(p.ok);
    assert!(p.calls.is_empty());
    assert_eq!(finish, "stop");
}

#[test]
fn append_tool_calls_json_uses_job_fallback_id() {
    let calls = [ToolCall {
        name: "edit".into(),
        arguments: "{\"path\":\"/tmp\"}".into(),
        ..Default::default()
    }];
    let mut out = Vec::new();
    append_tool_calls_json(&mut out, &calls, "chatcmpl-1");
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("\"id\":\"chatcmpl-1_tool_0\""));
    assert!(s.contains("\"type\":\"function\""));
    assert!(s.contains("\"name\":\"edit\""));
    assert!(s.contains("\"arguments\":\"{\\\"path\\\":\\\"/tmp\\\"}\""));
}

#[test]
fn sem_accum_dsml_closes_and_no_tools_cuts() {
    let block = format!(
        "{DSML_START}\n{DSML_INVOKE} name=\"bash\">\n\
         {DSML_PARAM} name=\"command\" string=\"true\">ls{DSML_PARAM_END}\n\
         {DSML_INVOKE_END}\n{DSML_END}"
    );
    let mut acc = SemAccum::init(true, true, false, ChatFormat::DeepSeek, b"");
    let f = acc.feed(block.as_bytes(), &[]);
    assert!(acc.saw_tool_start);
    assert!(acc.saw_tool_end);
    assert_eq!(acc.verdict, Some("tool_calls"));
    assert!(f.tool_block_closed);

    let mut cut = SemAccum::init(true, false, false, ChatFormat::DeepSeek, b"");
    let f = cut.feed(b"hello <|", &[]);
    assert!(!f.hit_stop);
    let f = cut.feed(format!("{DSML_START} tail").as_bytes(), &[]);
    assert!(f.hit_stop);
    assert!(f.tool_syntax_cut);
    assert_eq!(cut.verdict, Some("stop"));
    assert!(!cut.text.windows(DSML_START.len()).any(|w| w == DSML_START.as_bytes()));
}

#[test]
fn assign_tool_ids_fills_empty() {
    let mut calls = vec![ToolCall {
        name: "a".into(),
        arguments: "{}".into(),
        ..Default::default()
    }];
    assign_tool_ids(&mut calls, "id");
    assert_eq!(calls[0].id, "id_tool_0");
}
