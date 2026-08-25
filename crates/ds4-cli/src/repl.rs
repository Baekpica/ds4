//! Interactive CLI commands matching `ds4_cli.c` `print_repl_help` / `run_repl`.

pub const REPL_HELP: &str = "\
Commands:
  /help          Show this help.
  /think         Use normal thinking mode.
  /think-max     Use Think Max only when context is at least 393216 tokens.
  /nothink       Disable thinking mode.
  /ctx N         Set context size for following prompts.
  /power N       Set GPU duty cycle percentage, 1..100.
  /read FILE     Read a prompt from FILE and run it.
  /quit, /exit   Leave the prompt.
  Ctrl+C         Stop generation and return to the prompt.
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplLine {
    Help,
    Think,
    ThinkMax,
    NoThink,
    Ctx(i32),
    Power(Option<i32>),
    Read(String),
    Quit,
    Prompt(String),
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplError {
    UnknownCommand(String),
    CtxNeedsPositive,
    PowerRange,
    ReadNeedsPath,
}

impl ReplError {
    pub fn message(&self) -> String {
        match self {
            ReplError::UnknownCommand(cmd) => {
                format!("ds4: unknown command: {cmd}\nds4: type /help for commands\n")
            }
            ReplError::CtxNeedsPositive => "ds4: /ctx needs a positive integer\n".into(),
            ReplError::PowerRange => "ds4: /power must be between 1 and 100\n".into(),
            ReplError::ReadNeedsPath => "ds4: /read needs a file path\n".into(),
        }
    }
}

pub fn trim_inplace(s: &str) -> &str {
    s.trim()
}

pub fn parse_repl_line(line: &str) -> Result<ReplLine, ReplError> {
    let cmd = trim_inplace(line);
    if cmd.is_empty() {
        return Ok(ReplLine::Empty);
    }
    if cmd == "/help" {
        return Ok(ReplLine::Help);
    }
    if cmd == "/think" {
        return Ok(ReplLine::Think);
    }
    if cmd == "/think-max" {
        return Ok(ReplLine::ThinkMax);
    }
    if cmd == "/nothink" {
        return Ok(ReplLine::NoThink);
    }
    if cmd == "/quit" || cmd == "/exit" {
        return Ok(ReplLine::Quit);
    }
    if let Some(rest) = strip_command(cmd, "/power") {
        if rest.is_empty() {
            return Ok(ReplLine::Power(None));
        }
        return parse_power_percent(rest).map(|v| ReplLine::Power(Some(v)));
    }
    if let Some(rest) = strip_command(cmd, "/ctx") {
        if rest.is_empty() {
            return Err(ReplError::CtxNeedsPositive);
        }
        return parse_positive_int(rest)
            .ok_or(ReplError::CtxNeedsPositive)
            .map(ReplLine::Ctx);
    }
    if let Some(rest) = strip_command(cmd, "/read") {
        if rest.is_empty() {
            return Err(ReplError::ReadNeedsPath);
        }
        return Ok(ReplLine::Read(rest.to_string()));
    }
    if cmd.starts_with('/') {
        return Err(ReplError::UnknownCommand(cmd.to_string()));
    }
    Ok(ReplLine::Prompt(cmd.to_string()))
}

fn strip_command<'a>(cmd: &'a str, name: &str) -> Option<&'a str> {
    let rest = cmd.strip_prefix(name)?;
    if rest.is_empty() {
        return Some("");
    }
    if rest.starts_with(|c: char| c.is_ascii_whitespace()) {
        return Some(rest.trim());
    }
    None
}

fn parse_positive_int(s: &str) -> Option<i32> {
    let v = s.parse::<i32>().ok()?;
    (v > 0).then_some(v)
}

fn parse_power_percent(s: &str) -> Result<i32, ReplError> {
    let v = s.parse::<i32>().map_err(|_| ReplError::PowerRange)?;
    if (1..=100).contains(&v) {
        Ok(v)
    } else {
        Err(ReplError::PowerRange)
    }
}

pub fn think_mode_message(mode: ThinkRepl) -> &'static str {
    match mode {
        ThinkRepl::High => "Thinking mode: high.",
        ThinkRepl::Max => "Thinking mode: max.",
        ThinkRepl::HighBelowCtx => "Thinking mode: high (ctx below 393216).",
        ThinkRepl::None => "Thinking mode: none.",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkRepl {
    High,
    Max,
    HighBelowCtx,
    None,
}

pub const THINK_MAX_MIN_CTX: i32 = 393216;

pub fn think_max_active(ctx: i32) -> bool {
    ctx >= THINK_MAX_MIN_CTX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_text_matches_c() {
        assert!(REPL_HELP.contains("/help          Show this help."));
        assert!(REPL_HELP.contains("/quit, /exit   Leave the prompt."));
        assert!(REPL_HELP.contains("Ctrl+C         Stop generation and return to the prompt."));
    }

    #[test]
    fn parses_commands() {
        assert_eq!(parse_repl_line("  /help  ").unwrap(), ReplLine::Help);
        assert_eq!(parse_repl_line("/think").unwrap(), ReplLine::Think);
        assert_eq!(parse_repl_line("/think-max").unwrap(), ReplLine::ThinkMax);
        assert_eq!(parse_repl_line("/nothink").unwrap(), ReplLine::NoThink);
        assert_eq!(parse_repl_line("/quit").unwrap(), ReplLine::Quit);
        assert_eq!(parse_repl_line("/exit").unwrap(), ReplLine::Quit);
        assert_eq!(parse_repl_line("/ctx 8192").unwrap(), ReplLine::Ctx(8192));
        assert_eq!(parse_repl_line("/power").unwrap(), ReplLine::Power(None));
        assert_eq!(
            parse_repl_line("/power 50").unwrap(),
            ReplLine::Power(Some(50))
        );
        assert_eq!(
            parse_repl_line("/read notes.txt").unwrap(),
            ReplLine::Read("notes.txt".into())
        );
        assert_eq!(
            parse_repl_line("hello there").unwrap(),
            ReplLine::Prompt("hello there".into())
        );
        assert_eq!(parse_repl_line("   ").unwrap(), ReplLine::Empty);
    }

    #[test]
    fn unknown_and_invalid_match_c() {
        assert_eq!(
            parse_repl_line("/nope").unwrap_err().message(),
            "ds4: unknown command: /nope\nds4: type /help for commands\n"
        );
        assert_eq!(
            parse_repl_line("/ctx").unwrap_err().message(),
            "ds4: /ctx needs a positive integer\n"
        );
        assert_eq!(
            parse_repl_line("/power 0").unwrap_err().message(),
            "ds4: /power must be between 1 and 100\n"
        );
        assert_eq!(
            parse_repl_line("/read").unwrap_err().message(),
            "ds4: /read needs a file path\n"
        );
    }

    #[test]
    fn think_messages_match_c() {
        assert_eq!(think_mode_message(ThinkRepl::High), "Thinking mode: high.");
        assert_eq!(think_mode_message(ThinkRepl::Max), "Thinking mode: max.");
        assert_eq!(
            think_mode_message(ThinkRepl::HighBelowCtx),
            "Thinking mode: high (ctx below 393216)."
        );
        assert_eq!(think_mode_message(ThinkRepl::None), "Thinking mode: none.");
        assert!(!think_max_active(32768));
        assert!(think_max_active(393216));
    }
}
