const BUILT_IN_TOOLS_PROMPT: &str = include_str!("agent_tools_prompt.txt");
const DSML_OPEN: &[u8] = "<｜DSML｜tool_calls>".as_bytes();
const DSML_OPEN_MISSING_BAR: &[u8] = "<DSML｜tool_calls>".as_bytes();
const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";
const TOOL_UNSUPPORTED_ERROR: &str =
    "tool execution is not implemented in ds4-agent-rs; use ./ds4-agent";

mod web_tools;

#[derive(Debug)]
pub struct AgentArgs {
    model: String,
    backend: ds4_core::Backend,
    ctx: i32,
    threads: i32,
    tokens: i32,
    prompt: Option<String>,
    system: String,
    temp: f32,
    top_p: f32,
    min_p: f32,
    seed: u64,
    think: ds4_core::ChatThinkMode,
    chdir: Option<String>,
    non_interactive: bool,
    help: bool,
}

impl Default for AgentArgs {
    fn default() -> Self {
        Self {
            model: "ds4flash.gguf".into(),
            backend: default_backend(),
            ctx: 100000,
            threads: 0,
            tokens: 50000,
            prompt: None,
            system: "You are a helpful coding assistant running inside ds4-agent.".into(),
            temp: 1.0,
            top_p: 1.0,
            min_p: 0.05,
            seed: 0,
            think: ds4_core::ChatThinkMode::Low,
            chdir: None,
            non_interactive: false,
            help: false,
        }
    }
}

fn default_backend() -> ds4_core::Backend {
    if cfg!(target_os = "macos") {
        ds4_core::Backend::Metal
    } else {
        ds4_core::Backend::Cuda
    }
}

fn need_value(option: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("missing value for {option}"))
}

fn parse_positive_i32(option: &str, value: String) -> Result<i32, String> {
    value
        .parse::<i32>()
        .ok()
        .filter(|parsed| *parsed > 0)
        .ok_or_else(|| format!("invalid value for {option}: {value}"))
}

fn parse_f32_range(option: &str, value: String, min: f32, max: f32) -> Result<f32, String> {
    value
        .parse::<f32>()
        .ok()
        .filter(|parsed| parsed.is_finite() && *parsed >= min && *parsed <= max)
        .ok_or_else(|| format!("invalid value for {option}: {value}"))
}

fn parse_seed(option: &str, value: String) -> Result<u64, String> {
    value
        .parse::<u64>()
        .ok()
        .filter(|parsed| *parsed != 0)
        .ok_or_else(|| format!("invalid value for {option}: {value}"))
}

fn unsupported(option: &str) -> Result<AgentArgs, String> {
    Err(format!(
        "{option} is not implemented in the ds4-agent-rs one-turn shadow; use ./ds4-agent"
    ))
}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<AgentArgs, String> {
    let mut parsed = AgentArgs::default();
    let mut args = args.into_iter();
    let _argv0 = args.next();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                parsed.help = true;
                return Ok(parsed);
            }
            "--non-interactive" => parsed.non_interactive = true,
            "-p" | "--prompt" => parsed.prompt = Some(need_value(&arg, args.next())?),
            "-m" | "--model" => parsed.model = need_value(&arg, args.next())?,
            "-c" | "--ctx" => {
                let value = need_value(&arg, args.next())?;
                parsed.ctx = parse_positive_i32(&arg, value)?;
            }
            "-n" | "--tokens" => {
                let value = need_value(&arg, args.next())?;
                parsed.tokens = parse_positive_i32(&arg, value)?;
            }
            "-sys" | "--system" => parsed.system = need_value(&arg, args.next())?,
            "--temp" => {
                let value = need_value(&arg, args.next())?;
                parsed.temp = parse_f32_range(&arg, value, 0.0, 100.0)?;
            }
            "--top-p" => {
                let value = need_value(&arg, args.next())?;
                parsed.top_p = parse_f32_range(&arg, value, 0.0, 1.0)?;
            }
            "--min-p" => {
                let value = need_value(&arg, args.next())?;
                parsed.min_p = parse_f32_range(&arg, value, 0.0, 1.0)?;
            }
            "--seed" => {
                let value = need_value(&arg, args.next())?;
                parsed.seed = parse_seed(&arg, value)?;
            }
            "--think" => parsed.think = ds4_core::ChatThinkMode::Low,
            "--think-max" => parsed.think = ds4_core::ChatThinkMode::High,
            "--nothink" => parsed.think = ds4_core::ChatThinkMode::None,
            "--backend" => {
                parsed.backend = match need_value(&arg, args.next())?.as_str() {
                    "cuda" => ds4_core::Backend::Cuda,
                    "metal" => ds4_core::Backend::Metal,
                    "cpu" => ds4_core::Backend::Cpu,
                    value => return Err(format!("invalid backend: {value}")),
                };
            }
            "--cuda" => parsed.backend = ds4_core::Backend::Cuda,
            "--metal" => parsed.backend = ds4_core::Backend::Metal,
            "--cpu" => parsed.backend = ds4_core::Backend::Cpu,
            "-t" | "--threads" => {
                let value = need_value(&arg, args.next())?;
                parsed.threads = parse_positive_i32(&arg, value)?;
            }
            "--chdir" => parsed.chdir = Some(need_value(&arg, args.next())?),
            "--trace"
            | "--mtp"
            | "--mtp-draft"
            | "--mtp-margin"
            | "--quality"
            | "--power"
            | "--warm-weights"
            | "--dir-steering-file"
            | "--dir-steering-ffn"
            | "--dir-steering-attn"
            | "--role"
            | "--layers"
            | "--listen"
            | "--coordinator"
            | "--dist-prefill-chunk"
            | "--dist-prefill-window"
            | "--dist-activation-bits"
            | "--dist-replay-check"
            | "--debug" => return unsupported(&arg),
            _ => return Err(format!("unknown option: {arg}")),
        }
    }
    if !parsed.non_interactive {
        return Err("this shadow requires --non-interactive".into());
    }
    if parsed.prompt.is_none() {
        return Err("--non-interactive requires -p/--prompt in this one-turn shadow".into());
    }
    Ok(parsed)
}

const THINK_EFFORT_MIN_CONTEXT: i32 = 393216;

fn effective_think(args: &AgentArgs) -> ds4_core::ChatThinkMode {
    if matches!(
        args.think,
        ds4_core::ChatThinkMode::High | ds4_core::ChatThinkMode::Max
    ) && args.ctx < THINK_EFFORT_MIN_CONTEXT
    {
        ds4_core::ChatThinkMode::Low
    } else {
        args.think
    }
}

fn epoch_seconds() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".into())
}

fn current_local_datetime() -> String {
    // The parity oracle supplies a fixed `when`; this production-only clock leaf
    // is deliberately outside the byte-for-byte parity claim.
    let fallback = epoch_seconds();
    match std::process::Command::new("/bin/date")
        .arg("+%Y-%m-%d %H:%M:%S %Z")
        .output()
    {
        Ok(output) => validate_clock_output(output.status.success(), &output.stdout, &fallback),
        Err(_) => fallback,
    }
}

fn build_system_transcript(
    vocab: &ds4_core::Vocab,
    args: &AgentArgs,
    think: ds4_core::ChatThinkMode,
) -> Result<ds4_core::TokenBuffer, String> {
    let mut transcript = ds4_core::TokenBuffer::new();
    vocab
        .chat_begin(&mut transcript)
        .map_err(|error| error.to_string())?;
    vocab.chat_append_effort_prefix(&mut transcript, think);
    for token in vocab.encode_rendered_chat(built_in_tools_prompt()) {
        transcript.push(token);
    }
    if !args.system.is_empty() {
        let extra = format!("\n\n{}", args.system);
        vocab
            .chat_append_message(&mut transcript, "system", extra.as_bytes())
            .map_err(|error| error.to_string())?;
    }
    Ok(transcript)
}

fn append_user_turn(
    vocab: &ds4_core::Vocab,
    transcript: &mut ds4_core::TokenBuffer,
    prompt: &str,
    when: &str,
    think: ds4_core::ChatThinkMode,
) -> Result<(), String> {
    let datetime = format_datetime_context(when);
    vocab
        .chat_append_message(transcript, "system", datetime.as_bytes())
        .map_err(|error| error.to_string())?;
    vocab
        .chat_append_message(transcript, "user", prompt.as_bytes())
        .map_err(|error| error.to_string())?;
    vocab
        .chat_append_assistant_prefix(transcript, think)
        .map_err(|error| error.to_string())
}

fn random_seed() -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    now.as_secs() ^ (u64::from(std::process::id()) << 32) ^ u64::from(now.subsec_nanos())
}

#[cfg(test)]
fn tool_marker_in_new_suffix(raw: &[u8], appended: usize) -> bool {
    let max_marker = DSML_OPEN.len().max(DSML_OPEN_MISSING_BAR.len());
    let start = raw
        .len()
        .saturating_sub(appended.saturating_add(max_marker.saturating_sub(1)));
    contains_bytes(&raw[start..], DSML_OPEN) || contains_bytes(&raw[start..], DSML_OPEN_MISSING_BAR)
}

pub fn run(name: &str, args: AgentArgs) -> Result<i32, String> {
    use std::io::Write;

    if args.help {
        print!("{}", help_text(name));
        return Ok(0);
    }
    if let Some(path) = args.chdir.as_deref() {
        std::env::set_current_dir(path)
            .map_err(|error| format!("failed to chdir to {path}: {error}"))?;
    }

    let think = effective_think(&args);
    if args.think == ds4_core::ChatThinkMode::High && think != ds4_core::ChatThinkMode::High {
        eprintln!(
            "{name}: warning: --think-max needs --ctx >= {THINK_EFFORT_MIN_CONTEXT}; \
             ctx={} uses normal thinking instead",
            args.ctx
        );
    }

    let model = ds4_core::Model::open(&args.model, args.backend, args.threads, false)
        .map_err(|error| error.to_string())?;
    let mut session = model.session(args.ctx).map_err(|error| error.to_string())?;
    let mut transcript = build_system_transcript(model.vocab(), &args, think)?;
    session
        .sync(&transcript)
        .map_err(|error| error.to_string())?;

    let prompt = args
        .prompt
        .as_deref()
        .ok_or_else(|| "--non-interactive requires -p/--prompt".to_string())?;
    append_user_turn(
        model.vocab(),
        &mut transcript,
        prompt,
        &current_local_datetime(),
        think,
    )?;
    session
        .sync(&transcript)
        .map_err(|error| error.to_string())?;

    let mut rng = if args.seed == 0 {
        random_seed()
    } else {
        args.seed
    };
    let mut web = web_tools::non_interactive_web();
    let mut read_cursor = web_tools::ReadCursor::default();
    let mut output = Vec::new();
    loop {
        let room = session
            .ctx()
            .saturating_sub(session.pos())
            .saturating_sub(1);
        let max_tokens = args.tokens.min(room).max(0);
        let mut raw = Vec::new();
        for _ in 0..max_tokens {
            let token = session.sample(args.temp, 0, args.top_p, args.min_p, &mut rng);
            if token < 0 {
                return Err("failed to sample the next token".into());
            }
            if token == model.token_eos() {
                break;
            }
            session.eval(token).map_err(|error| error.to_string())?;
            transcript.push(token);
            let piece = model.token_text(token).map_err(|error| error.to_string())?;
            raw.extend_from_slice(&piece);
            if web_tools::block_complete(&raw) {
                break;
            }
        }
        transcript.push(model.token_eos());

        if web_tools::has_block(&raw) {
            let round = web_tools::handle_round_with_cursor(&raw, &mut web, &mut read_cursor)
                .map_err(|_| TOOL_UNSUPPORTED_ERROR.to_string())?;
            output.extend_from_slice(&project_output(&[&round.visible])?);
            model
                .vocab()
                .chat_append_message(&mut transcript, "tool", &round.observation)
                .map_err(|error| error.to_string())?;
            model
                .vocab()
                .chat_append_assistant_prefix(&mut transcript, think)
                .map_err(|error| error.to_string())?;
            session
                .sync(&transcript)
                .map_err(|error| error.to_string())?;
            continue;
        }
        output.extend_from_slice(&project_output(&[&raw])?);
        break;
    }

    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    stdout
        .write_all(&output)
        .map_err(|error| error.to_string())?;
    stdout.flush().map_err(|error| error.to_string())?;
    Ok(0)
}

fn help_text(name: &str) -> String {
    format!(
        "Usage: {name} --non-interactive -p TEXT [options]\n\
         \n\
         One-turn ds4-agent shadow. Supports google_search, visit_page, read, more, and list. \
         Use ./ds4-agent for other tools, interactive, KV, MTP, or distributed execution.\n\
         \n\
         Options:\n\
           -m, --model FILE        GGUF model path. Default: ds4flash.gguf\n\
           -c, --ctx N            Context size. Default: 100000\n\
           -n, --tokens N         Max generated tokens. Default: 50000\n\
           -p, --prompt TEXT      Required one-turn prompt.\n\
           --non-interactive      Required shadow mode.\n\
           -sys, --system TEXT    Extra system prompt.\n\
           --temp F               Sampling temperature. Default: 1\n\
           --top-p F              Nucleus probability. Default: 1\n\
           --min-p F              Min-p threshold. Default: 0.05\n\
           --seed N               Nonzero sampling seed.\n\
           --think                Normal thinking mode. Default.\n\
           --think-max            High effort when ctx >= 393216.\n\
           --nothink              Disable thinking.\n\
           --backend NAME         metal, cuda, or cpu.\n\
           --metal, --cuda, --cpu Select backend explicitly.\n\
           -t, --threads N        CPU helper threads.\n\
           --chdir DIR            Change directory before model load.\n\
           -h, --help             Show this help.\n"
    )
}

fn built_in_tools_prompt() -> &'static str {
    BUILT_IN_TOOLS_PROMPT
}

fn format_datetime_context(when: &str) -> String {
    format!(
        "Current local date and time at session start: {when}. \
         Use this only when date or time matters."
    )
}

fn validate_clock_output(success: bool, stdout: &[u8], fallback: &str) -> String {
    if success {
        if let Ok(text) = std::str::from_utf8(stdout) {
            let text = text.trim_end_matches(['\r', '\n']);
            if !text.is_empty() {
                return text.to_string();
            }
        }
    }
    fallback.to_string()
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn project_output(chunks: &[&[u8]]) -> Result<Vec<u8>, &'static str> {
    let raw: Vec<u8> = chunks
        .iter()
        .flat_map(|chunk| chunk.iter().copied())
        .collect();
    if contains_bytes(&raw, DSML_OPEN) || contains_bytes(&raw, DSML_OPEN_MISSING_BAR) {
        return Err(TOOL_UNSUPPORTED_ERROR);
    }

    let mut out = Vec::with_capacity(raw.len() + 2);
    let mut post_think_gap = false;
    let mut wrote_visible = false;
    let mut last_output_newline = true;
    let mut i = 0;
    while i < raw.len() {
        let rem = &raw[i..];
        if rem.starts_with(THINK_OPEN) {
            post_think_gap = false;
            i += THINK_OPEN.len();
            continue;
        }
        if rem.starts_with(THINK_CLOSE) {
            if !last_output_newline {
                out.push(b'\n');
            }
            out.push(b'\n');
            last_output_newline = true;
            post_think_gap = true;
            i += THINK_CLOSE.len();
            continue;
        }

        let byte = raw[i];
        if post_think_gap && matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
            i += 1;
            continue;
        }
        post_think_gap = false;
        out.push(byte);
        wrote_visible = true;
        last_output_newline = byte == b'\n';
        i += 1;
    }

    if wrote_visible {
        if !last_output_newline {
            out.push(b'\n');
        }
        out.push(b'\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    fn oracle() -> PathBuf {
        if let Ok(path) = std::env::var("DS4_AGENT_C_ORACLE") {
            return PathBuf::from(path);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/agent_c_oracle")
    }

    fn c_out(args: &[&str]) -> Vec<u8> {
        let path = oracle();
        assert!(
            path.exists(),
            "build the C oracle first: make tests/parity/agent_c_oracle (missing {})",
            path.display()
        );
        let out = Command::new(path)
            .args(args)
            .output()
            .expect("run agent_c_oracle");
        assert!(
            out.status.success(),
            "agent_c_oracle failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    }

    fn unhex_line(bytes: &[u8]) -> Vec<u8> {
        let text = std::str::from_utf8(bytes).expect("oracle hex utf8").trim();
        assert_eq!(text.len() % 2, 0, "even oracle hex length");
        text.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).expect("hex pair utf8");
                u8::from_str_radix(pair, 16).expect("hex pair")
            })
            .collect()
    }

    fn argv(items: &[&str]) -> Vec<String> {
        std::iter::once("ds4-agent-rs".to_string())
            .chain(items.iter().map(|item| (*item).to_string()))
            .collect()
    }

    #[test]
    fn built_in_prompt_matches_current_c_bytes() {
        assert_eq!(
            built_in_tools_prompt().as_bytes(),
            unhex_line(&c_out(&["prompt"]))
        );
    }

    #[test]
    fn fixed_datetime_message_matches_current_c_bytes() {
        let when = "2026-08-25 12:34:56 KST";
        assert_eq!(
            format_datetime_context(when).as_bytes(),
            unhex_line(&c_out(&["datetime", when]))
        );
    }

    #[test]
    fn non_tty_projection_matches_c_tapes() {
        let cases: &[(bool, &[&str])] = &[
            (false, &["hello"]),
            (true, &["hello"]),
            (true, &["reasoning", "</thi", "nk>", "  \nanswer"]),
            (false, &["<thi", "nk>trace</think>\n\nanswer"]),
            (true, &["line with newline\n", "</think>", "\t \nanswer\n"]),
        ];
        for (thinking, chunks) in cases {
            let mut args = vec!["project", if *thinking { "1" } else { "0" }];
            args.extend_from_slice(chunks);
            let rust = project_output(&chunks.iter().map(|s| s.as_bytes()).collect::<Vec<_>>())
                .expect("no tool tape");
            assert_eq!(rust, unhex_line(&c_out(&args)), "case {args:?}");
        }
    }

    #[test]
    fn both_c_dsml_openers_are_rejected() {
        for marker in ["<｜DSML｜tool_calls>", "<DSML｜tool_calls>"] {
            assert_eq!(c_out(&["dsml", marker]), b"match=1 complete=1\n");
            assert!(project_output(&[b"prefix ", marker.as_bytes()]).is_err());

            let split = marker.len() / 2;
            let mut raw = b"prefix ".to_vec();
            raw.extend_from_slice(&marker.as_bytes()[..split]);
            assert!(!tool_marker_in_new_suffix(&raw, split));
            raw.extend_from_slice(&marker.as_bytes()[split..]);
            assert!(tool_marker_in_new_suffix(&raw, marker.len() - split));
        }
    }

    #[test]
    fn clock_output_validation_falls_back_on_failure_empty_or_invalid_utf8() {
        let fallback = "1770000000";
        assert_eq!(
            validate_clock_output(true, b"2026-08-25 12:34:56 KST\n", fallback),
            "2026-08-25 12:34:56 KST"
        );
        assert_eq!(validate_clock_output(false, b"ignored", fallback), fallback);
        assert_eq!(validate_clock_output(true, b"\r\n", fallback), fallback);
        assert_eq!(validate_clock_output(true, b"\xff\n", fallback), fallback);
    }

    #[test]
    fn narrow_cli_defaults_match_c_agent() {
        let parsed =
            parse_args(argv(&["--non-interactive", "-p", "hello"])).expect("minimal invocation");
        assert_eq!(parsed.model, "ds4flash.gguf");
        assert_eq!(parsed.ctx, 100000);
        assert_eq!(parsed.tokens, 50000);
        assert_eq!(
            parsed.system,
            "You are a helpful coding assistant running inside ds4-agent."
        );
        assert_eq!(parsed.temp, 1.0);
        assert_eq!(parsed.top_p, 1.0);
        assert_eq!(parsed.min_p, 0.05);
        assert_eq!(parsed.think, ds4_core::ChatThinkMode::Low);
        assert_eq!(parsed.prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn high_thinking_uses_the_c_context_boundary() {
        let mut args = AgentArgs::default();
        args.think = ds4_core::ChatThinkMode::High;
        args.ctx = THINK_EFFORT_MIN_CONTEXT - 1;
        assert_eq!(effective_think(&args), ds4_core::ChatThinkMode::Low);
        args.ctx = THINK_EFFORT_MIN_CONTEXT;
        assert_eq!(effective_think(&args), ds4_core::ChatThinkMode::High);
    }

    #[test]
    fn narrow_cli_requires_one_shot_shape_and_rejects_deferred_flags() {
        assert!(parse_args(argv(&["-p", "hello"]))
            .unwrap_err()
            .contains("--non-interactive"));
        assert!(parse_args(argv(&["--non-interactive"]))
            .unwrap_err()
            .contains("-p/--prompt"));
        for flag in [
            "--trace",
            "--mtp",
            "--mtp-draft",
            "--mtp-margin",
            "--quality",
            "--power",
            "--warm-weights",
            "--dir-steering-file",
            "--role",
        ] {
            let error = parse_args(argv(&["--non-interactive", "-p", "hello", flag, "ignored"]))
                .unwrap_err();
            assert!(error.contains("not implemented"), "{flag}: {error}");
        }
    }

    #[test]
    fn help_names_the_supported_tool_subset() {
        let help = help_text("ds4-agent-rs");
        assert!(help.contains("google_search, visit_page, read, more, and list"));
        assert!(!help.contains("Tool calls are rejected"));
    }
}
