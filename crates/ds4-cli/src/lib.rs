//! Shadow CLI host. Calls the same C inference core through `ds4-core`.

use ds4_core::Backend;

#[derive(Debug)]
pub struct ShadowArgs {
    pub model: Option<String>,
    pub mtp: Option<String>,
    pub dspark: Option<String>,
    pub backend: Backend,
    pub ctx: i32,
    pub threads: i32,
    pub tokens: Vec<i32>,
    pub predict: i32,
    pub lifecycle_only: bool,
    pub identify: bool,
    pub inventory: bool,
    pub tokenize: bool,
    pub tok_family: Option<String>,
    pub tok_cmd: Option<String>,
    pub tok_arg: Option<String>,
    pub session_plan: bool,
    pub session_cmd: Option<String>,
    pub session_args: Vec<String>,
    pub session_payload: bool,
    pub payload_cmd: Option<String>,
    pub payload_args: Vec<String>,
    pub bind_names: bool,
    pub bind_names_variant: Option<String>,
    pub bind_plan: bool,
    pub validate: bool,
    pub layout: bool,
    pub layout_variant: Option<String>,
    pub help: bool,
}

impl Default for ShadowArgs {
    fn default() -> Self {
        Self {
            model: None,
            mtp: None,
            dspark: None,
            backend: Backend::Cuda,
            ctx: 2048,
            threads: 0,
            tokens: Vec::new(),
            predict: 0,
            lifecycle_only: false,
            identify: false,
            inventory: false,
            tokenize: false,
            tok_family: None,
            tok_cmd: None,
            tok_arg: None,
            session_plan: false,
            session_cmd: None,
            session_args: Vec::new(),
            session_payload: false,
            payload_cmd: None,
            payload_args: Vec::new(),
            bind_names: false,
            bind_names_variant: None,
            bind_plan: false,
            validate: false,
            layout: false,
            layout_variant: None,
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
            "--mtp" => {
                parsed.mtp = Some(require_value(&arg, iter.next())?);
            }
            "--dspark" => {
                parsed.dspark = Some(require_value(&arg, iter.next())?);
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
            "--identify" => parsed.identify = true,
            "--inventory" => parsed.inventory = true,
            "--tokenize" => {
                parsed.tokenize = true;
                parsed.tok_family = Some(require_value(&arg, iter.next())?);
                parsed.tok_cmd = Some(require_value(&arg, iter.next())?);
                if let Some(peek) = iter.peek() {
                    if !peek.starts_with('-') {
                        parsed.tok_arg = iter.next();
                    }
                }
            }
            "--session-plan" => {
                parsed.session_plan = true;
                parsed.session_cmd = Some(require_value(&arg, iter.next())?);
                while let Some(peek) = iter.peek() {
                    if peek.starts_with('-') {
                        break;
                    }
                    parsed.session_args.push(iter.next().unwrap());
                }
            }
            "--session-payload" => {
                parsed.session_payload = true;
                parsed.payload_cmd = Some(require_value(&arg, iter.next())?);
                while let Some(peek) = iter.peek() {
                    if peek.starts_with('-') {
                        break;
                    }
                    parsed.payload_args.push(iter.next().unwrap());
                }
            }
            "--bind-names" => {
                parsed.bind_names = true;
                parsed.bind_names_variant = Some(require_value(&arg, iter.next())?);
            }
            "--bind-plan" => parsed.bind_plan = true,
            "--validate" => parsed.validate = true,
            "--layout" => {
                parsed.layout = true;
                parsed.layout_variant = Some(require_value(&arg, iter.next())?);
            }
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
  {name} -m MODEL --identify
  {name} -m MODEL --inventory
  {name} --bind-names VARIANT
  {name} --layout VARIANT
  {name} -m MODEL --bind-plan
  {name} -m MODEL --validate
  {name} -m MODEL --tokenize FAMILY CMD [ARG]
  {name} --session-plan CMD [ARGS...]
  {name} --session-payload CMD [ARGS...]
  {name} -m MODEL [--backend cuda|cpu|metal] [-c CTX] [--lifecycle]
  {name} -m MODEL --tokens 1,2,3 [--predict N]
  {name} -m MODEL [--mtp GGUF] [--dspark GGUF] ...

--mtp/--dspark attach the DeepSeek-only sibling support models; the host
resolves each sibling bind catalog + expected layouts, native skips that
sibling's name walk and layout check.

--identify mmaps GGUF metadata only (no CUDA, no ds4_bridge_model_open).
--inventory mmaps the tensor directory + split remap (no CUDA, no engine open).
--bind-names dumps the host weights_bind catalog (no GGUF, no engine open).
--layout dumps the host weights_validate_layout table (no GGUF, no engine open).
--bind-plan resolves a catalog against the host inventory (no CUDA).
With --bind-names VARIANT, --bind-plan uses that catalog (including
mtp-flash / dspark-pro) instead of identifying the GGUF family.
--validate runs host-owned config_validate (no CUDA, no engine open).
VARIANT is flash|pro|solar-open2|motif3|exaone-moe|dots3-note
or DeepSeek sibling mtp-flash|mtp-pro|dspark-flash|dspark-pro.
--tokenize loads the host-owned GPT-2/BPE vocab (no engine open).
--session-plan dumps the host session ledger (no engine open).
--session-payload dumps the host DSV4 prefix codec (no engine open).
FAMILY is deepseek4|motif3|solar-open2|exaone-moe|dots3-note.
CMD is specials | encode HEX | render HEX | decode ID | stop ID.
"
    )
}

pub fn run(name: &str, args: ShadowArgs) -> Result<i32, String> {
    if args.help {
        print!("{}", help_text(name));
        return Ok(0);
    }
    if args.session_plan {
        let cmd = args
            .session_cmd
            .as_deref()
            .ok_or_else(|| "--session-plan requires CMD".to_string())?;
        let argv: Vec<&str> = args.session_args.iter().map(String::as_str).collect();
        print!("{}", ds4_core::session_dump_cmd(cmd, &argv));
        return Ok(0);
    }
    if args.session_payload {
        let cmd = args
            .payload_cmd
            .as_deref()
            .ok_or_else(|| "--session-payload requires CMD".to_string())?;
        let argv: Vec<&str> = args.payload_args.iter().map(String::as_str).collect();
        print!("{}", ds4_core::payload_dump_cmd(cmd, &argv));
        return Ok(0);
    }
    if args.layout {
        let name = args
            .layout_variant
            .as_deref()
            .ok_or_else(|| "--layout requires VARIANT".to_string())?;
        let dump = ds4_core::dump_expected_layouts_variant(name)
            .ok_or_else(|| format!("unknown layout variant: {name}"))?;
        print!("{dump}");
        return Ok(0);
    }
    if args.bind_names && !args.bind_plan {
        let name = args
            .bind_names_variant
            .as_deref()
            .ok_or_else(|| "--bind-names requires VARIANT".to_string())?;
        let dump = ds4_core::dump_bind_names_variant(name)
            .ok_or_else(|| format!("unknown bind-names variant: {name}"))?;
        print!("{dump}");
        return Ok(0);
    }

    let model_path = args
        .model
        .as_deref()
        .ok_or_else(|| "missing -m/--model (or pass --help)".to_string())?;

    if args.identify {
        let id = ds4_core::identify_gguf(std::path::Path::new(model_path))
            .map_err(|e| e.to_string())?;
        println!("{}", id.report_line(model_path));
        return Ok(0);
    }

    if args.inventory {
        let inv = ds4_core::TensorInventory::open(std::path::Path::new(model_path))
            .map_err(|e| e.to_string())?;
        print!("{}", inv.dump());
        return Ok(0);
    }
    if args.bind_plan {
        let inv = ds4_core::TensorInventory::open(std::path::Path::new(model_path))
            .map_err(|e| e.to_string())?;
        let (support, shape) = if let Some(name) = args.bind_names_variant.as_deref() {
            let (support, v) = ds4_core::catalog_from_bind_name(name)
                .ok_or_else(|| format!("unknown bind-names variant: {name}"))?;
            (support, ds4_core::shape_for_variant(v))
        } else {
            let id = ds4_core::identify_gguf(std::path::Path::new(model_path))
                .map_err(|e| e.to_string())?;
            (None, id.shape)
        };
        print!(
            "{}",
            ds4_core::BindPlan::resolve_catalog(support, shape, &inv).dump()
        );
        return Ok(0);
    }
    if args.validate {
        print!("{}", ds4_core::dump_validate(std::path::Path::new(model_path)));
        return Ok(0);
    }

    if args.tokenize {
        let fam_name = args
            .tok_family
            .as_deref()
            .ok_or_else(|| " --tokenize requires FAMILY".to_string())?;
        let family = ds4_core::ModelFamily::from_oracle_name(fam_name)
            .ok_or_else(|| format!("unknown tokenizer family: {fam_name}"))?;
        let cmd = args
            .tok_cmd
            .as_deref()
            .ok_or_else(|| "--tokenize requires CMD".to_string())?;
        let vocab = ds4_core::Vocab::load_path(std::path::Path::new(model_path), family)
            .map_err(|e| e.to_string())?;
        print!(
            "{}",
            ds4_core::dump_cmd(&vocab, cmd, args.tok_arg.as_deref().unwrap_or(""))
        );
        return Ok(0);
    }

    let model = ds4_core::Model::open_with_support(
        model_path,
        args.backend,
        args.threads,
        true,
        args.mtp.as_deref(),
        args.dspark.as_deref(),
    )
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

    #[test]
    fn parses_identify() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--identify"])).unwrap();
        assert!(parsed.identify);
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
    }

    #[test]
    fn parses_inventory() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--inventory"])).unwrap();
        assert!(parsed.inventory);
    }

    #[test]
    fn parses_tokenize() {
        let parsed = parse_args(args(&[
            "-m",
            "m.gguf",
            "--tokenize",
            "motif3",
            "encode",
            "6869",
        ]))
        .unwrap();
        assert!(parsed.tokenize);
        assert_eq!(parsed.tok_family.as_deref(), Some("motif3"));
        assert_eq!(parsed.tok_cmd.as_deref(), Some("encode"));
        assert_eq!(parsed.tok_arg.as_deref(), Some("6869"));
    }

    #[test]
    fn parses_session_plan() {
        let parsed = parse_args(args(&[
            "--session-plan",
            "rewrite",
            "1024",
            "1100",
            "1024",
        ]))
        .unwrap();
        assert!(parsed.session_plan);
        assert_eq!(parsed.session_cmd.as_deref(), Some("rewrite"));
        assert_eq!(parsed.session_args, vec!["1024", "1100", "1024"]);
    }

    #[test]
    fn parses_session_payload() {
        let parsed = parse_args(args(&["--session-payload", "encode-deepseek"])).unwrap();
        assert!(parsed.session_payload);
        assert_eq!(parsed.payload_cmd.as_deref(), Some("encode-deepseek"));
        assert!(parsed.payload_args.is_empty());
    }

    #[test]
    fn parses_bind_names() {
        let parsed = parse_args(args(&["--bind-names", "motif3"])).unwrap();
        assert!(parsed.bind_names);
        assert_eq!(parsed.bind_names_variant.as_deref(), Some("motif3"));
    }

    #[test]
    fn parses_bind_names_support() {
        let parsed = parse_args(args(&["--bind-names", "mtp-flash"])).unwrap();
        assert!(parsed.bind_names);
        assert_eq!(parsed.bind_names_variant.as_deref(), Some("mtp-flash"));
    }

    #[test]
    fn parses_layout_support() {
        let parsed = parse_args(args(&["--layout", "dspark-pro"])).unwrap();
        assert!(parsed.layout);
        assert_eq!(parsed.layout_variant.as_deref(), Some("dspark-pro"));
    }

    #[test]
    fn parses_bind_plan() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--bind-plan"])).unwrap();
        assert!(parsed.bind_plan);
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
    }

    #[test]
    fn parses_bind_plan_with_support_catalog() {
        let parsed = parse_args(args(&[
            "-m",
            "mtp.gguf",
            "--bind-names",
            "mtp-flash",
            "--bind-plan",
        ]))
        .unwrap();
        assert!(parsed.bind_plan);
        assert!(parsed.bind_names);
        assert_eq!(parsed.bind_names_variant.as_deref(), Some("mtp-flash"));
        assert_eq!(parsed.model.as_deref(), Some("mtp.gguf"));
    }

    #[test]
    fn parses_validate() {
        let parsed = parse_args(args(&["-m", "m.gguf", "--validate"])).unwrap();
        assert!(parsed.validate);
        assert_eq!(parsed.model.as_deref(), Some("m.gguf"));
    }

    #[test]
    fn parses_layout() {
        let parsed = parse_args(args(&["--layout", "flash"])).unwrap();
        assert!(parsed.layout);
        assert_eq!(parsed.layout_variant.as_deref(), Some("flash"));
    }
}
