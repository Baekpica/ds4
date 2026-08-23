//! Shadow CLI host. Calls the same C inference core through `ds4-core`.

use ds4_core::Backend;

#[derive(Debug)]
pub struct ShadowArgs {
    pub model: Option<String>,
    pub backend: Backend,
    pub ctx: i32,
    pub threads: i32,
    pub tokens: Vec<i32>,
    pub predict: i32,
    pub lifecycle_only: bool,
    pub help: bool,
}

impl Default for ShadowArgs {
    fn default() -> Self {
        Self {
            model: None,
            backend: Backend::Cuda,
            ctx: 2048,
            threads: 0,
            tokens: Vec::new(),
            predict: 0,
            lifecycle_only: false,
            help: false,
        }
    }
}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<ShadowArgs, String> {
    let mut parsed = ShadowArgs::default();
    let mut iter = args.into_iter().peekable();
    let _argv0 = iter.next();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => parsed.help = true,
            "-m" | "--model" => {
                parsed.model = Some(require_value(&arg, iter.next())?);
            }
            "--backend" => {
                parsed.backend = parse_backend(&require_value(&arg, iter.next())?)?;
            }
            "-c" | "--ctx" => {
                parsed.ctx = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "-t" | "--threads" => {
                parsed.threads = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--tokens" => {
                parsed.tokens = parse_tokens(&require_value(&arg, iter.next())?)?;
            }
            "--predict" => {
                parsed.predict = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--lifecycle" => parsed.lifecycle_only = true,
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(parsed)
}

pub fn help_text(name: &str) -> String {
    format!(
        "\
{name} — Rust shadow of the C host (same C inference core)

Usage:
  {name} --help
  {name} -m MODEL [--backend cuda|cpu|metal] [-c CTX] [--lifecycle]
  {name} -m MODEL --tokens 1,2,3 [--predict N]

Tokenizer text prompts are not in the Phase 3 ABI yet. Pass raw token
ids, or --lifecycle to open a model and create a session only.
"
    )
}

pub fn run(name: &str, args: ShadowArgs) -> Result<i32, String> {
    if args.help {
        print!("{}", help_text(name));
        return Ok(0);
    }
    let model_path = args
        .model
        .as_deref()
        .ok_or_else(|| "missing -m/--model (or pass --help)".to_string())?;

    let model = ds4_core::Model::open(model_path, args.backend, args.threads, true)
        .map_err(|e| e.to_string())?;
    let mut session = model.session(args.ctx).map_err(|e| e.to_string())?;

    if args.lifecycle_only {
        println!(
            "lifecycle ok backend={:?} ctx={} pos={}",
            args.backend,
            args.ctx,
            session.pos()
        );
        return Ok(0);
    }

    if args.tokens.is_empty() && args.predict <= 0 {
        println!(
            "open ok backend={:?} ctx={} pos={} (pass --tokens or --lifecycle)",
            args.backend,
            args.ctx,
            session.pos()
        );
        return Ok(0);
    }

    if !args.tokens.is_empty() {
        let buf = ds4_core::TokenBuffer::from_tokens(args.tokens);
        session.sync(&buf).map_err(|e| e.to_string())?;
        print!("{}", session.argmax());
        for _ in 0..args.predict {
            let token = session.argmax();
            session.eval(token).map_err(|e| e.to_string())?;
            print!(" {}", session.argmax());
        }
        println!();
    }
    Ok(0)
}

fn require_value(flag: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_i32(flag: &str, value: &str) -> Result<i32, String> {
    value
        .parse()
        .map_err(|_| format!("{flag}: invalid integer {value}"))
}

fn parse_backend(value: &str) -> Result<Backend, String> {
    match value {
        "cuda" => Ok(Backend::Cuda),
        "cpu" => Ok(Backend::Cpu),
        "metal" => Ok(Backend::Metal),
        other => Err(format!("unknown backend: {other}")),
    }
}

fn parse_tokens(value: &str) -> Result<Vec<i32>, String> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse()
                .map_err(|_| format!("invalid token id: {part}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        std::iter::once("ds4-rs".to_string())
            .chain(parts.iter().map(|s| (*s).to_string()))
            .collect()
    }

    #[test]
    fn parses_lifecycle() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--lifecycle", "-c", "4096"])).unwrap();
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
        assert_eq!(parsed.ctx, 4096);
        assert!(parsed.lifecycle_only);
        assert_eq!(parsed.backend, Backend::Cuda);
    }

    #[test]
    fn parses_token_ids() {
        let parsed = parse_args(args(&["--tokens", "1, 2,3", "--predict", "4"])).unwrap();
        assert_eq!(parsed.tokens, vec![1, 2, 3]);
        assert_eq!(parsed.predict, 4);
    }

    #[test]
    fn rejects_unknown() {
        assert!(parse_args(args(&["--nope"])).is_err());
    }
}
