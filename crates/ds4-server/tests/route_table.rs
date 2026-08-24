//! C reason table + C↔Rust `route_decide` / `compute_needs` bytes.

use ds4_server::{
    compute_needs, route_decide, Api, NeedInput, ReqKind, RouteEnv, WireSurface, LANE_CONTINUOUS,
    LANE_NONE, LANE_SERIAL, LANE_STATIC, NEED_BANK_FRONTIER, NEED_CONTINUATION_PUBLISH,
    NEED_CORRECTIVE_RECOVERY, NEED_DURABLE_RESPONSE, NEED_LIVE_FRONTIER, NEED_PER_ROW_SAMPLING,
    NEED_PREFILL_ONLY, NEED_STOP_SCAN, NEED_STREAMING, NEED_THINKING, NEED_TOKEN_IDS,
    NEED_TOOL_SCAN, REASON_COALESCE_OFF, REASON_CONT, REASON_CONT_BANK, REASON_CONT_UNAVAILABLE,
    REASON_NEED_CONTINUATION_PUBLISH, REASON_NEED_DURABLE, REASON_NEED_LIVE_FRONTIER,
    REASON_NEED_PREFILL_ONLY, REASON_STATIC_NO_CONT, REASON_STATIC_PROMPT_BOUNDS, REASON_SURFACE,
    REASON_TOKEN_IDS_PROJECTION, REASON_TOOLS_COMPLETION,
};
use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_ROUTE_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/route_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/route_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_decide(needs: u32, surf: WireSurface, env: &RouteEnv) -> (u8, u8) {
    let out = Command::new(require_oracle())
        .args([
            "decide",
            &needs.to_string(),
            &(surf as i32).to_string(),
            &(env.coalesce as i32).to_string(),
            &(env.have_cont as i32).to_string(),
            &(env.cont_anthropic as i32).to_string(),
            &(env.cont_responses as i32).to_string(),
            &(env.cont_tools_anthropic as i32).to_string(),
            &(env.cont_tools_responses as i32).to_string(),
            &env.seq_cap.to_string(),
            &env.prompt_len.to_string(),
        ])
        .output()
        .expect("run route_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let mut lane = 0u8;
    let mut reason = 0u8;
    for part in s.split_whitespace() {
        if let Some(v) = part.strip_prefix("lane=") {
            lane = v.parse().unwrap();
        }
        if let Some(v) = part.strip_prefix("reason=") {
            reason = v.parse().unwrap();
        }
    }
    (lane, reason)
}

fn assert_row(needs: u32, surf: WireSurface, env: &RouteEnv, lane: u8, reason: u8) {
    let d = route_decide(needs, surf, env);
    assert_eq!(
        (d.lane, d.reason),
        (lane, reason),
        "rust needs={needs:#x} surf={surf:?} env={env:?}"
    );
    let c = c_decide(needs, surf, env);
    assert_eq!(
        c,
        (lane, reason),
        "c oracle needs={needs:#x} surf={surf:?} env={env:?}"
    );
}

fn on() -> RouteEnv {
    RouteEnv {
        coalesce: true,
        have_cont: true,
        cont_anthropic: true,
        cont_responses: true,
        cont_tools_anthropic: true,
        cont_tools_responses: true,
        seq_cap: 4096,
        prompt_len: 16,
    }
}

fn no_ctx() -> RouteEnv {
    let mut e = on();
    e.have_cont = false;
    e.seq_cap = 0;
    e
}

fn off() -> RouteEnv {
    let mut e = on();
    e.coalesce = false;
    e
}

fn big() -> RouteEnv {
    let mut e = on();
    e.prompt_len = 4097;
    e
}

fn empty() -> RouteEnv {
    let mut e = on();
    e.prompt_len = 0;
    e
}

#[test]
fn reason_table_matches_c() {
    let on = on();
    let no_ctx = no_ctx();
    let off = off();
    let big = big();
    let empty = empty();
    let mut anth_off = on;
    anth_off.cont_anthropic = false;
    let mut resp_off = on;
    resp_off.cont_responses = false;

    assert_row(0, WireSurface::OpenaiChat, &on, LANE_CONTINUOUS, REASON_CONT);
    assert_row(
        NEED_STREAMING,
        WireSurface::OpenaiChat,
        &no_ctx,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(0, WireSurface::OpenaiChat, &no_ctx, LANE_STATIC, REASON_STATIC_NO_CONT);
    assert_row(0, WireSurface::OpenaiChat, &big, LANE_STATIC, REASON_STATIC_PROMPT_BOUNDS);
    assert_row(0, WireSurface::OpenaiChat, &empty, LANE_STATIC, REASON_STATIC_PROMPT_BOUNDS);
    assert_row(0, WireSurface::OpenaiChat, &off, LANE_SERIAL, REASON_COALESCE_OFF);

    assert_row(
        NEED_TOKEN_IDS | NEED_STREAMING,
        WireSurface::OpenaiChat,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT,
    );
    assert_row(
        NEED_TOKEN_IDS,
        WireSurface::OpenaiChat,
        &on,
        LANE_SERIAL,
        REASON_TOKEN_IDS_PROJECTION,
    );
    assert_row(
        NEED_TOKEN_IDS | NEED_STREAMING,
        WireSurface::OpenaiCompletion,
        &on,
        LANE_SERIAL,
        REASON_TOKEN_IDS_PROJECTION,
    );
    assert_row(NEED_TOOL_SCAN, WireSurface::OpenaiChat, &on, LANE_CONTINUOUS, REASON_CONT);
    assert_row(
        NEED_TOOL_SCAN,
        WireSurface::OpenaiCompletion,
        &on,
        LANE_SERIAL,
        REASON_TOOLS_COMPLETION,
    );

    assert_row(0, WireSurface::Anthropic, &on, LANE_CONTINUOUS, REASON_CONT);
    assert_row(
        NEED_THINKING | NEED_STOP_SCAN | NEED_PER_ROW_SAMPLING,
        WireSurface::Anthropic,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT,
    );
    assert_row(0, WireSurface::Anthropic, &anth_off, LANE_SERIAL, REASON_SURFACE);
    assert_row(0, WireSurface::Responses, &on, LANE_CONTINUOUS, REASON_CONT);
    assert_row(
        NEED_THINKING | NEED_PER_ROW_SAMPLING,
        WireSurface::Responses,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT,
    );
    assert_row(0, WireSurface::Responses, &resp_off, LANE_SERIAL, REASON_SURFACE);
    assert_row(0, WireSurface::Anthropic, &resp_off, LANE_CONTINUOUS, REASON_CONT);
    assert_row(0, WireSurface::Responses, &anth_off, LANE_CONTINUOUS, REASON_CONT);

    assert_row(NEED_STREAMING, WireSurface::Anthropic, &on, LANE_CONTINUOUS, REASON_CONT);
    assert_row(
        NEED_STREAMING | NEED_THINKING | NEED_STOP_SCAN,
        WireSurface::Anthropic,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT,
    );
    assert_row(NEED_STREAMING, WireSurface::Anthropic, &anth_off, LANE_SERIAL, REASON_SURFACE);
    assert_row(
        NEED_STREAMING,
        WireSurface::Anthropic,
        &no_ctx,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(
        NEED_STREAMING,
        WireSurface::Anthropic,
        &big,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(NEED_STREAMING, WireSurface::Responses, &on, LANE_CONTINUOUS, REASON_CONT);
    assert_row(
        NEED_STREAMING | NEED_THINKING,
        WireSurface::Responses,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT,
    );
    assert_row(NEED_STREAMING, WireSurface::Responses, &resp_off, LANE_SERIAL, REASON_SURFACE);
    assert_row(
        NEED_STREAMING,
        WireSurface::Responses,
        &no_ctx,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(
        NEED_STREAMING,
        WireSurface::Responses,
        &big,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );

    assert_row(0, WireSurface::Anthropic, &big, LANE_STATIC, REASON_STATIC_PROMPT_BOUNDS);
    assert_row(0, WireSurface::Anthropic, &no_ctx, LANE_STATIC, REASON_STATIC_NO_CONT);
    assert_row(0, WireSurface::Responses, &big, LANE_STATIC, REASON_STATIC_PROMPT_BOUNDS);
    assert_row(
        NEED_STOP_SCAN,
        WireSurface::Anthropic,
        &no_ctx,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(
        NEED_THINKING,
        WireSurface::Responses,
        &big,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    let mut anth_off_noctx = no_ctx;
    anth_off_noctx.cont_anthropic = false;
    assert_row(0, WireSurface::Anthropic, &anth_off_noctx, LANE_SERIAL, REASON_SURFACE);
    let mut resp_off_noctx = no_ctx;
    resp_off_noctx.cont_responses = false;
    assert_row(0, WireSurface::Responses, &resp_off_noctx, LANE_SERIAL, REASON_SURFACE);

    let stream_tools = NEED_STREAMING | NEED_TOOL_SCAN | NEED_CONTINUATION_PUBLISH;
    assert_row(stream_tools, WireSurface::Anthropic, &on, LANE_CONTINUOUS, REASON_CONT);
    assert_row(
        stream_tools | NEED_THINKING | NEED_PER_ROW_SAMPLING | NEED_STOP_SCAN,
        WireSurface::Anthropic,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT,
    );
    assert_row(stream_tools, WireSurface::Responses, &on, LANE_CONTINUOUS, REASON_CONT);
    let mut tools_off_a = on;
    tools_off_a.cont_tools_anthropic = false;
    assert_row(
        stream_tools,
        WireSurface::Anthropic,
        &tools_off_a,
        LANE_SERIAL,
        REASON_NEED_CONTINUATION_PUBLISH,
    );
    assert_row(stream_tools, WireSurface::Responses, &tools_off_a, LANE_CONTINUOUS, REASON_CONT);
    let mut tools_off_r = on;
    tools_off_r.cont_tools_responses = false;
    assert_row(
        stream_tools,
        WireSurface::Responses,
        &tools_off_r,
        LANE_SERIAL,
        REASON_NEED_CONTINUATION_PUBLISH,
    );
    assert_row(
        stream_tools,
        WireSurface::Anthropic,
        &anth_off,
        LANE_SERIAL,
        REASON_NEED_CONTINUATION_PUBLISH,
    );
    assert_row(
        stream_tools,
        WireSurface::Anthropic,
        &no_ctx,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(
        stream_tools,
        WireSurface::Anthropic,
        &big,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(stream_tools, WireSurface::Anthropic, &off, LANE_SERIAL, REASON_COALESCE_OFF);

    assert_row(
        NEED_BANK_FRONTIER,
        WireSurface::Anthropic,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT_BANK,
    );
    assert_row(
        NEED_BANK_FRONTIER | stream_tools,
        WireSurface::Responses,
        &on,
        LANE_CONTINUOUS,
        REASON_CONT_BANK,
    );
    assert_row(
        NEED_BANK_FRONTIER | NEED_TOOL_SCAN | NEED_CONTINUATION_PUBLISH | NEED_CORRECTIVE_RECOVERY,
        WireSurface::Anthropic,
        &on,
        LANE_SERIAL,
        REASON_NEED_CONTINUATION_PUBLISH,
    );
    assert_row(
        NEED_BANK_FRONTIER,
        WireSurface::Anthropic,
        &tools_off_a,
        LANE_SERIAL,
        REASON_NEED_LIVE_FRONTIER,
    );
    assert_row(
        NEED_BANK_FRONTIER,
        WireSurface::Anthropic,
        &anth_off,
        LANE_SERIAL,
        REASON_NEED_LIVE_FRONTIER,
    );
    assert_row(
        NEED_BANK_FRONTIER,
        WireSurface::Anthropic,
        &no_ctx,
        LANE_SERIAL,
        REASON_CONT_UNAVAILABLE,
    );
    assert_row(
        NEED_BANK_FRONTIER,
        WireSurface::OpenaiChat,
        &on,
        LANE_SERIAL,
        REASON_NEED_LIVE_FRONTIER,
    );

    assert_row(
        NEED_TOOL_SCAN | NEED_CONTINUATION_PUBLISH | NEED_CORRECTIVE_RECOVERY,
        WireSurface::Anthropic,
        &on,
        LANE_SERIAL,
        REASON_NEED_CONTINUATION_PUBLISH,
    );
    assert_row(
        NEED_LIVE_FRONTIER | NEED_TOOL_SCAN,
        WireSurface::Responses,
        &on,
        LANE_SERIAL,
        REASON_NEED_LIVE_FRONTIER,
    );
    assert_row(
        NEED_PREFILL_ONLY,
        WireSurface::Anthropic,
        &on,
        LANE_SERIAL,
        REASON_NEED_PREFILL_ONLY,
    );
    assert_row(
        NEED_DURABLE_RESPONSE,
        WireSurface::Responses,
        &on,
        LANE_NONE,
        REASON_NEED_DURABLE,
    );
}

fn c_needs(r: &NeedInput) -> u32 {
    let out = Command::new(require_oracle())
        .args([
            "needs",
            &(r.api as i32).to_string(),
            &(r.stream as i32).to_string(),
            &r.temperature.to_string(),
            &(r.think as i32).to_string(),
            &r.stop_count.to_string(),
            &(r.has_tools as i32).to_string(),
            &(r.return_token_ids as i32).to_string(),
            &(r.responses_requires_live_tool_state as i32).to_string(),
            &(r.responses_requires_live_reasoning as i32).to_string(),
            &(r.anthropic_requires_live_tool_state as i32).to_string(),
            &(r.live_state_bank_owned as i32).to_string(),
            &(r.max_tokens_set as i32).to_string(),
            &r.max_tokens.to_string(),
        ])
        .output()
        .expect("run route_c_oracle needs");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
}

fn blank() -> NeedInput {
    NeedInput {
        api: Api::Openai,
        kind: ReqKind::Chat,
        stream: false,
        temperature: 0.0,
        think: false,
        stop_count: 0,
        has_tools: false,
        return_token_ids: false,
        responses_requires_live_tool_state: false,
        responses_requires_live_reasoning: false,
        anthropic_requires_live_tool_state: false,
        live_state_bank_owned: false,
        max_tokens_set: false,
        max_tokens: 128,
    }
}

#[test]
fn compute_needs_matches_c() {
    let cases = [
        blank(),
        NeedInput {
            stream: true,
            temperature: 0.7,
            think: true,
            stop_count: 1,
            has_tools: true,
            ..blank()
        },
        NeedInput {
            api: Api::Anthropic,
            has_tools: true,
            stream: false,
            ..blank()
        },
        NeedInput {
            api: Api::Anthropic,
            has_tools: true,
            stream: true,
            ..blank()
        },
        NeedInput {
            api: Api::Anthropic,
            max_tokens_set: true,
            max_tokens: 0,
            ..blank()
        },
        NeedInput {
            api: Api::Responses,
            max_tokens_set: true,
            max_tokens: 0,
            ..blank()
        },
        NeedInput {
            api: Api::Anthropic,
            anthropic_requires_live_tool_state: true,
            live_state_bank_owned: true,
            ..blank()
        },
        NeedInput {
            api: Api::Responses,
            responses_requires_live_tool_state: true,
            live_state_bank_owned: true,
            ..blank()
        },
        NeedInput {
            api: Api::Responses,
            responses_requires_live_reasoning: true,
            live_state_bank_owned: true,
            ..blank()
        },
    ];
    for r in cases {
        let rust = compute_needs(&r);
        let c = c_needs(&r);
        assert_eq!(rust, c, "needs mismatch rust={rust:#x} c={c:#x}");
    }
    let anth_tools = NeedInput {
        api: Api::Anthropic,
        has_tools: true,
        stream: false,
        ..blank()
    };
    assert_eq!(
        compute_needs(&anth_tools),
        NEED_TOOL_SCAN | NEED_CONTINUATION_PUBLISH | NEED_CORRECTIVE_RECOVERY
    );
    let anth_zero = NeedInput {
        api: Api::Anthropic,
        max_tokens_set: true,
        max_tokens: 0,
        ..blank()
    };
    assert_eq!(compute_needs(&anth_zero), NEED_PREFILL_ONLY);
    let resp_zero = NeedInput {
        api: Api::Responses,
        max_tokens_set: true,
        max_tokens: 0,
        ..blank()
    };
    assert_eq!(compute_needs(&resp_zero), 0);
    let bank = NeedInput {
        api: Api::Anthropic,
        anthropic_requires_live_tool_state: true,
        live_state_bank_owned: true,
        ..blank()
    };
    assert_eq!(compute_needs(&bank), NEED_BANK_FRONTIER);
    let reason = NeedInput {
        api: Api::Responses,
        responses_requires_live_reasoning: true,
        live_state_bank_owned: true,
        ..blank()
    };
    assert_eq!(compute_needs(&reason), NEED_LIVE_FRONTIER);
}

#[test]
fn decode_budget_three_states_match_c() {
    fn c(set: bool, tokens: i32, def: i32) -> i32 {
        let out = Command::new(require_oracle())
            .args([
                "budget",
                &(set as i32).to_string(),
                &tokens.to_string(),
                &def.to_string(),
            ])
            .output()
            .unwrap();
        assert!(out.status.success());
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
    }
    let def = 393216;
    assert_eq!(ds4_server::decode_budget(false, def, def), def);
    assert_eq!(c(false, def, def), def);
    assert_eq!(ds4_server::decode_budget(true, 4096, def), 4096);
    assert_eq!(c(true, 4096, def), 4096);
    assert_eq!(ds4_server::decode_budget(true, 0, def), 0);
    assert_eq!(c(true, 0, def), 0);
    assert_eq!(ds4_server::decode_budget(true, -5, def), 0);
    assert_eq!(c(true, -5, def), 0);
}

#[test]
fn reasoning_effort_names_match_c() {
    fn c(name: &str) -> String {
        let out = Command::new(require_oracle())
            .args(["effort", name])
            .output()
            .unwrap();
        assert!(out.status.success());
        String::from_utf8_lossy(&out.stdout).into_owned()
    }
    let rows = [
        ("max", Some(ds4_server::ThinkMode::Max)),
        ("high", Some(ds4_server::ThinkMode::High)),
        ("xhigh", Some(ds4_server::ThinkMode::High)),
        ("low", Some(ds4_server::ThinkMode::Low)),
        ("medium", Some(ds4_server::ThinkMode::Low)),
        ("minimal", Some(ds4_server::ThinkMode::Low)),
        ("none", Some(ds4_server::ThinkMode::None)),
        ("off", Some(ds4_server::ThinkMode::None)),
        ("banana", None),
    ];
    for (name, want) in rows {
        assert_eq!(ds4_server::parse_reasoning_effort_name(name), want);
        match want {
            Some(m) => assert_eq!(c(name), (m as i32).to_string()),
            None => assert_eq!(c(name), "ERROR"),
        }
    }
    assert_eq!(
        ds4_server::think_mode_from_enabled(false, ds4_server::ThinkMode::High),
        ds4_server::ThinkMode::None
    );
    assert_eq!(
        ds4_server::think_mode_from_enabled(true, ds4_server::ThinkMode::High),
        ds4_server::ThinkMode::High
    );
}
