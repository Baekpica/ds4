//! Shadow CLI host. Calls the same C inference core through `ds4-core`.

pub mod agent;
pub mod bench;

use ds4_core::Backend;

#[derive(Debug)]
pub struct ShadowArgs {
    pub model: Option<String>,
    pub mtp: Option<String>,
    pub dspark: Option<String>,
    pub backend: Backend,
    pub ctx: i32,
    pub ctx_set: bool,
    pub threads: i32,
    pub tokens: Vec<i32>,
    pub predict: i32,
    pub n_predict: i32,
    pub system: String,
    pub nothink: bool,
    pub prompt: Option<String>,
    pub prompt_file: Option<String>,
    pub dump_logprobs: Option<String>,
    pub logprobs_top_k: i32,
    pub temp: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
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
            model: Some("ds4flash.gguf".into()),
            mtp: None,
            dspark: None,
            backend: default_backend(),
            ctx: 32768,
            ctx_set: false,
            threads: 0,
            tokens: Vec::new(),
            predict: 0,
            n_predict: 50000,
            system: "You are a helpful assistant".into(),
            nothink: false,
            prompt: None,
            prompt_file: None,
            dump_logprobs: None,
            logprobs_top_k: 20,
            temp: 1.0,
            top_p: 1.0,
            min_p: 0.05,
            seed: 0,
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
            /* C CLI backend spellings, so the proof harness command line
             * (`--cuda -m ...`) drives this shadow unchanged. */
            "--cuda" => parsed.backend = Backend::Cuda,
            "--cpu" => parsed.backend = Backend::Cpu,
            "--metal" => parsed.backend = Backend::Metal,
            "-c" | "--ctx" => {
                parsed.ctx = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
                parsed.ctx_set = true;
            }
            "-n" | "--tokens" | "--n-predict" => {
                let value = require_value(&arg, iter.next())?;
                if arg == "--tokens" && value.contains(',') {
                    return Err(
                        "--tokens is the output budget; use --token-ids for raw token IDs"
                            .into(),
                    );
                }
                parsed.n_predict = parse_positive_i32(&arg, &value)?;
            }
            "--temp" => {
                let v = require_value(&arg, iter.next())?;
                parsed.temp = parse_f32_range(&arg, &v, 0.0, 100.0)?;
            }
            "--top-p" => {
                let v = require_value(&arg, iter.next())?;
                parsed.top_p = parse_f32_range(&arg, &v, 0.0, 1.0)?;
            }
            "--min-p" => {
                let v = require_value(&arg, iter.next())?;
                parsed.min_p = parse_f32_range(&arg, &v, 0.0, 1.0)?;
            }
            "--seed" => {
                let v = require_value(&arg, iter.next())?;
                parsed.seed = parse_positive_u64(&arg, &v)?;
            }
            "--think" => parsed.nothink = false,
            "--nothink" => parsed.nothink = true,
            "-sys" | "--system" => {
                parsed.system = require_value(&arg, iter.next())?;
            }
            "-p" | "--prompt" => {
                if parsed.prompt.is_some() || parsed.prompt_file.is_some() {
                    return Err("specify only one prompt source".into());
                }
                parsed.prompt = Some(require_value(&arg, iter.next())?);
            }
            "--prompt-file" => {
                if parsed.prompt.is_some() || parsed.prompt_file.is_some() {
                    return Err("specify only one prompt source".into());
                }
                parsed.prompt_file = Some(require_value(&arg, iter.next())?);
            }
            "--dump-logprobs" => {
                parsed.dump_logprobs = Some(require_value(&arg, iter.next())?);
            }
            "--logprobs-top-k" => {
                parsed.logprobs_top_k = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "-t" | "--threads" => {
                parsed.threads = parse_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--token-ids" => {
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

const DEEPSEEK_BOS: &str = "<｜begin▁of▁sentence｜>";

fn is_rendered_chat_prompt(prompt: &str) -> bool {
    prompt.starts_with(DEEPSEEK_BOS)
}

fn prompt_text(args: &ShadowArgs) -> Result<String, String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => {
            std::fs::read_to_string(path).map_err(|e| format!("prompt-file: {e}"))
        }
        (Some(_), Some(_)) => Err("specify only one prompt source".into()),
        (None, None) => Err(
            "one-shot generation requires -p or --prompt-file; REPL is not implemented".into(),
        ),
    }
}

fn validate_one_shot_args(args: &ShadowArgs) -> Result<(), String> {
    if args.prompt.is_none() && args.prompt_file.is_none() {
        return Err(
            "one-shot generation requires -p or --prompt-file; REPL is not implemented".into(),
        );
    }
    if args.mtp.is_some() || args.dspark.is_some() {
        return Err("one-shot generation with --mtp/--dspark is not implemented in ds4-rs".into());
    }
    Ok(())
}

fn generation_limit(ctx: i32, pos: i32, requested: i32) -> i32 {
    let room = ctx.saturating_sub(pos);
    requested.min(room).max(0)
}

fn sampled_generation_limit(ctx: i32, pos: i32, requested: i32) -> i32 {
    let room = ctx.saturating_sub(pos).saturating_sub(1);
    requested.min(room).max(0)
}

const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";

// Non-TTY byte parity only. The Rust shadow intentionally emits no ANSI color.
struct TokenPrinter {
    format_thinking: bool,
    pending: Vec<u8>,
    last_output_newline: bool,
}

impl TokenPrinter {
    fn new(format_thinking: bool) -> Self {
        Self {
            format_thinking,
            pending: Vec::new(),
            last_output_newline: true,
        }
    }

    fn write_text<W: std::io::Write>(&mut self, out: &mut W, text: &[u8]) -> std::io::Result<()> {
        if !self.format_thinking {
            out.write_all(text)?;
            if let Some(last) = text.last() {
                self.last_output_newline = *last == b'\n';
            }
            return Ok(());
        }

        let mut bytes = std::mem::take(&mut self.pending);
        bytes.extend_from_slice(text);
        let mut i = 0;
        while i < bytes.len() {
            let rem = &bytes[i..];
            if rem.starts_with(THINK_OPEN) {
                i += THINK_OPEN.len();
                continue;
            }
            if rem.starts_with(THINK_CLOSE) {
                if !self.last_output_newline {
                    out.write_all(b"\n")?;
                    self.last_output_newline = true;
                }
                i += THINK_CLOSE.len();
                continue;
            }
            if rem[0] == b'<'
                && ((rem.len() < THINK_OPEN.len() && THINK_OPEN.starts_with(rem))
                    || (rem.len() < THINK_CLOSE.len() && THINK_CLOSE.starts_with(rem)))
            {
                self.pending.extend_from_slice(rem);
                break;
            }

            out.write_all(&rem[..1])?;
            self.last_output_newline = rem[0] == b'\n';
            i += 1;
        }
        Ok(())
    }

    fn finish<W: std::io::Write>(&mut self, out: &mut W) -> std::io::Result<()> {
        if self.format_thinking && !self.pending.is_empty() {
            out.write_all(&self.pending)?;
            self.last_output_newline = self.pending.last() == Some(&b'\n');
            self.pending.clear();
        }
        if !self.last_output_newline {
            out.write_all(b"\n")?;
            self.last_output_newline = true;
        }
        out.flush()
    }
}

fn run_one_shot(model: &ds4_core::Model, args: &ShadowArgs, text: &str) -> Result<i32, String> {
    use std::io::Write;

    const THINK_NONE: i32 = 0;
    const THINK_LOW: i32 = 1;

    let prompt = if is_rendered_chat_prompt(&text) {
        model.tokenize_rendered_chat(&text)
    } else {
        model.encode_chat_prompt(
            Some(args.system.as_str()),
            &text,
            if args.nothink { THINK_NONE } else { THINK_LOW },
        )
    }
    .map_err(|e| e.to_string())?;

    let mut session = model.session(args.ctx).map_err(|e| e.to_string())?;
    session.sync(&prompt).map_err(|e| e.to_string())?;

    let sampled = args.temp > 0.0;
    let max_tokens = if sampled {
        sampled_generation_limit(session.ctx(), session.pos(), args.n_predict)
    } else {
        generation_limit(session.ctx(), session.pos(), args.n_predict)
    };
    let mut rng = args.seed;
    if sampled && rng == 0 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        rng = (now ^ (u64::from(std::process::id()) << 32)) | 1;
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut printer = TokenPrinter::new(!args.nothink);
    let mut decode_error = None;

    for generated in 0..max_tokens {
        let token = if sampled {
            session.sample(args.temp, 0, args.top_p, args.min_p, &mut rng)
        } else {
            session.argmax()
        };
        if token < 0 {
            decode_error = Some(if sampled {
                "failed to sample the next token".into()
            } else {
                "failed to select the next token".into()
            });
            break;
        }
        if model.token_is_stop(token) {
            break;
        }
        if sampled {
            if let Err(e) = session.eval(token) {
                decode_error = Some(e.to_string());
                break;
            }
        }

        let piece = model.token_text(token).map_err(|e| e.to_string())?;
        printer
            .write_text(&mut out, &piece)
            .map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;

        if !sampled {
            if generated + 1 == max_tokens {
                break;
            }
            if let Err(e) = session.eval(token) {
                decode_error = Some(e.to_string());
                break;
            }
        }
    }
    printer.finish(&mut out).map_err(|e| e.to_string())?;
    match decode_error {
        Some(error) => Err(error),
        None => Ok(0),
    }
}

/// Proof-harness dump: mirror of the C CLI `run_logprob_dump` loop
/// (top_logprobs -> argmax -> write step -> stop check -> eval).  Prompt
/// rendering follows the C CLI; token text and the stop set use the host vocab.
fn run_logprob_dump(
    model: &ds4_core::Model,
    args: &ShadowArgs,
    prompt_text: &str,
) -> Result<i32, String> {
    use std::io::Write;

    const THINK_NONE: i32 = 0;
    const THINK_LOW: i32 = 1;
    const TOP_K_CAP: usize = 128;

    let think = if args.nothink { THINK_NONE } else { THINK_LOW };
    let prompt = if is_rendered_chat_prompt(&prompt_text) {
        model.tokenize_rendered_chat(&prompt_text)
    } else {
        model.encode_chat_prompt(Some(args.system.as_str()), &prompt_text, think)
    }
    .map_err(|e| e.to_string())?;
    let n_prompt = prompt.len() as i32;

    /* C defaults ctx to 262144 on CUDA; the ids do not depend on ctx size,
     * so grow to fit unless -c was explicit. */
    let ctx = if args.ctx_set {
        args.ctx
    } else {
        args.ctx.max(n_prompt + args.n_predict + 8)
    };
    let mut session = model.session(ctx).map_err(|e| e.to_string())?;
    session.sync(&prompt).map_err(|e| e.to_string())?;

    let path = args.dump_logprobs.as_deref().unwrap();
    let mut fp = std::io::BufWriter::new(
        std::fs::File::create(path).map_err(|e| format!("dump-logprobs {path}: {e}"))?,
    );
    let k = (args.logprobs_top_k.max(1) as usize).min(TOP_K_CAP);
    let mut w = |s: String| fp.write_all(s.as_bytes()).map_err(|e| e.to_string());

    w(format!(
        "{{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":{n_prompt},\n  \"ctx\":{ctx},\n  \"top_k\":{k},\n  \"steps\":[\n"
    ))?;

    let mut max_tokens = args.n_predict;
    let room = session.ctx() - session.pos();
    if room <= 1 {
        max_tokens = 0;
    } else if max_tokens > room - 1 {
        max_tokens = room - 1;
    }

    for generated in 0..max_tokens {
        let scores = session.top_logprobs(k);
        let token = session.argmax();
        if generated > 0 {
            w(",\n".into())?;
        }
        w(format!("    {{\"step\":{generated},\"selected\":"))?;
        w(json_token(model, token))?;
        w(",\"top_logprobs\":[".into())?;
        for (i, s) in scores.iter().take_while(|s| s.id >= 0).enumerate() {
            if i > 0 {
                w(",".into())?;
            }
            /* Rust shortest-roundtrip floats, not C's %.9g: the md5
             * contract reads only the selected ids. */
            w(format!(
                "{{\"token\":{},\"logit\":{},\"logprob\":{}}}",
                json_token(model, s.id),
                s.logit,
                s.logprob
            ))?;
        }
        w("]}".into())?;

        if model.token_is_stop(token) {
            break;
        }
        session.eval(token).map_err(|e| e.to_string())?;
    }

    w("\n  ]\n}\n".into())?;
    fp.flush().map_err(|e| e.to_string())?;
    Ok(0)
}

fn json_token(model: &ds4_core::Model, token: i32) -> String {
    let bytes = model.token_text(token).unwrap_or_default();
    let mut s = format!("{{\"id\":{token},\"text\":\"");
    for &b in &bytes {
        match b {
            b'"' => s.push_str("\\\""),
            b'\\' => s.push_str("\\\\"),
            0x20..=0x7e => s.push(b as char),
            _ => s.push_str(&format!("\\u{:04x}", b)),
        }
    }
    s.push_str("\",\"bytes\":[");
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&b.to_string());
    }
    s.push_str("]}");
    s
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
  {name} -m MODEL (-p PROMPT | --prompt-file FILE) [-n N]
      [--think|--nothink] [--temp F --top-p F --min-p F --seed N]
  {name} -m MODEL --token-ids 1,2,3 [--predict N]
  {name} -m MODEL [--mtp GGUF] [--dspark GGUF] ...
  {name} --cuda -m MODEL --temp 0 -n N [--nothink] --dump-logprobs F \\
      --logprobs-top-k K (-p PROMPT | --prompt-file FILE)

--mtp/--dspark attach the DeepSeek-only sibling support models; the host
resolves each sibling bind catalog + expected layouts, native skips that
sibling's name walk and layout check.
--dump-logprobs mirrors the C CLI proof loop (chat-template encode via
the engine, argmax decode, host stop set); ctx grows to fit prompt+n
unless -c is explicit.
One-shot sampling uses the native C sampler; an explicit --seed is
reproducible across the C and Rust hosts.
Thinking tags follow the C non-TTY byte contract; ANSI coloring is omitted.
--tokens is the C-compatible output budget; --token-ids is the shadow-only
raw token-ID diagnostic.

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

    let raw_token_diagnostic = !args.tokens.is_empty() || args.predict > 0;
    let generation_text = if args.prompt.is_some() || args.prompt_file.is_some() {
        Some(prompt_text(&args)?)
    } else {
        None
    };
    if args.dump_logprobs.is_none() && !args.lifecycle_only && !raw_token_diagnostic {
        validate_one_shot_args(&args)?;
    }
    if args.dump_logprobs.is_some() && generation_text.is_none() {
        return Err("--dump-logprobs requires -p or --prompt-file".into());
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

    if args.dump_logprobs.is_some() {
        return run_logprob_dump(&model, &args, generation_text.as_deref().unwrap());
    }

    if args.lifecycle_only {
        let session = model.session(args.ctx).map_err(|e| e.to_string())?;
        println!(
            "lifecycle ok backend={:?} ctx={} pos={}",
            args.backend,
            args.ctx,
            session.pos()
        );
        return Ok(0);
    }

    if !args.tokens.is_empty() {
        let mut session = model.session(args.ctx).map_err(|e| e.to_string())?;
        let buf = ds4_core::TokenBuffer::from_tokens(args.tokens);
        session.sync(&buf).map_err(|e| e.to_string())?;
        print!("{}", session.argmax());
        for _ in 0..args.predict {
            let token = session.argmax();
            session.eval(token).map_err(|e| e.to_string())?;
            print!(" {}", session.argmax());
        }
        println!();
        return Ok(0);
    }
    if args.predict > 0 {
        /* Preserve the old diagnostic's no-op behavior when --predict was
         * supplied without raw token IDs. */
        let _session = model.session(args.ctx).map_err(|e| e.to_string())?;
        return Ok(0);
    }
    run_one_shot(&model, &args, generation_text.as_deref().unwrap())
}

fn require_value(flag: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_i32(flag: &str, value: &str) -> Result<i32, String> {
    value
        .parse()
        .map_err(|_| format!("{flag}: invalid integer {value}"))
}

fn parse_positive_i32(flag: &str, value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("invalid value for {flag}: {value}"))
}

fn parse_f32_range(flag: &str, value: &str, min: f32, max: f32) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite() && *v >= min && *v <= max)
        .ok_or_else(|| format!("invalid value for {flag}: {value}"))?;
    Ok(parsed)
}

fn parse_positive_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("invalid value for {flag}: {value}"))
}

#[cfg(target_os = "macos")]
fn default_backend() -> Backend {
    Backend::Metal
}

#[cfg(not(target_os = "macos"))]
fn default_backend() -> Backend {
    Backend::Cuda
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

    fn format_fragments(format_thinking: bool, parts: &[&[u8]]) -> Vec<u8> {
        let mut printer = TokenPrinter::new(format_thinking);
        let mut out = Vec::new();
        for part in parts {
            printer.write_text(&mut out, part).unwrap();
        }
        printer.finish(&mut out).unwrap();
        out
    }

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
        assert_eq!(parsed.backend, default_backend());
    }

    #[test]
    fn uses_c_cli_core_defaults() {
        let parsed = parse_args(args(&[])).unwrap();

        assert_eq!(parsed.model.as_deref(), Some("ds4flash.gguf"));
        assert_eq!(parsed.backend, default_backend());
        assert_eq!(parsed.ctx, 32768);
        assert_eq!(parsed.n_predict, 50000);
        assert_eq!(parsed.system, "You are a helpful assistant");
        assert_eq!(parsed.temp, 1.0);
        assert_eq!(parsed.top_p, 1.0);
        assert_eq!(parsed.min_p, 0.05);
        assert_eq!(parsed.seed, 0);
        assert!(!parsed.nothink);
    }

    #[test]
    fn parses_c_cli_sampling_options() {
        let parsed = parse_args(args(&[
            "--temp", "0.8", "--top-p", "0.9", "--min-p", "0.02", "--seed", "424242",
        ]))
        .unwrap();

        assert_eq!(parsed.temp, 0.8);
        assert_eq!(parsed.top_p, 0.9);
        assert_eq!(parsed.min_p, 0.02);
        assert_eq!(parsed.seed, 424242);
    }

    #[test]
    fn rejects_sampling_outside_c_cli_ranges() {
        for bad in [
            ["--temp", "NaN"],
            ["--temp", "101"],
            ["--top-p", "-0.1"],
            ["--top-p", "1.1"],
            ["--min-p", "-0.1"],
            ["--min-p", "1.1"],
            ["--seed", "0"],
        ] {
            assert!(parse_args(args(&bad)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn parses_output_budget_and_explicit_raw_token_ids() {
        let parsed = parse_args(args(&[
            "--tokens",
            "17",
            "--token-ids",
            "1, 2,3",
            "--predict",
            "4",
        ]))
        .unwrap();

        assert_eq!(parsed.n_predict, 17);
        assert_eq!(parsed.tokens, vec![1, 2, 3]);
        assert_eq!(parsed.predict, 4);
    }

    #[test]
    fn parses_token_ids() {
        let parsed = parse_args(args(&["--token-ids", "1, 2,3", "--predict", "4"])).unwrap();
        assert_eq!(parsed.tokens, vec![1, 2, 3]);
        assert_eq!(parsed.predict, 4);
    }

    #[test]
    fn raw_token_ids_have_an_actionable_tokens_collision_error() {
        let err = parse_args(args(&["--tokens", "1,2,3"])).unwrap_err();

        assert!(err.contains("use --token-ids for raw token IDs"), "{err}");
    }

    #[test]
    fn rejects_non_positive_c_cli_integer_ranges() {
        assert_eq!(
            parse_args(args(&["--tokens", "0"])).unwrap_err(),
            "invalid value for --tokens: 0"
        );
        assert_eq!(
            parse_args(args(&["--ctx", "-1"])).unwrap_err(),
            "invalid value for --ctx: -1"
        );
    }

    #[test]
    fn rejects_more_than_one_prompt_source() {
        assert_eq!(
            parse_args(args(&["-p", "one", "--prompt-file", "two.txt"])).unwrap_err(),
            "specify only one prompt source"
        );
        assert_eq!(
            parse_args(args(&["--prompt-file", "one.txt", "-p", "two"])).unwrap_err(),
            "specify only one prompt source"
        );
    }

    #[test]
    fn parses_system_and_last_think_mode_wins() {
        let thinking = parse_args(args(&[
            "-sys",
            "system",
            "--nothink",
            "--think",
            "-p",
            "hello",
        ]))
        .unwrap();
        assert_eq!(thinking.system, "system");
        assert!(!thinking.nothink);

        let direct = parse_args(args(&["--think", "--nothink", "-p", "hello"])).unwrap();
        assert!(direct.nothink);
    }

    #[test]
    fn validates_only_supported_one_shot_routes() {
        let greedy = parse_args(args(&["-p", "hello", "--temp", "0", "--nothink"])).unwrap();
        assert!(validate_one_shot_args(&greedy).is_ok());

        let sampled = parse_args(args(&["-p", "hello", "--temp", "0.5", "--nothink"])).unwrap();
        assert!(validate_one_shot_args(&sampled).is_ok());

        let thinking = parse_args(args(&["-p", "hello", "--temp", "0", "--think"])).unwrap();
        assert!(validate_one_shot_args(&thinking).is_ok());

        let mtp = parse_args(args(&[
            "-p",
            "hello",
            "--temp",
            "0",
            "--nothink",
            "--mtp",
            "mtp.gguf",
        ]))
        .unwrap();
        assert!(validate_one_shot_args(&mtp)
            .unwrap_err()
            .contains("--mtp/--dspark"));
    }

    #[test]
    fn rejects_the_unimplemented_repl_route() {
        let parsed = parse_args(args(&[])).unwrap();

        assert!(validate_one_shot_args(&parsed)
            .unwrap_err()
            .contains("REPL is not implemented"));
    }

    #[test]
    fn clamps_generation_to_the_remaining_context() {
        assert_eq!(generation_limit(32768, 100, 50000), 32668);
        assert_eq!(generation_limit(128, 127, 10), 1);
        assert_eq!(generation_limit(128, 128, 10), 0);
        assert_eq!(generation_limit(128, 120, 4), 4);

        assert_eq!(sampled_generation_limit(128, 127, 10), 0);
        assert_eq!(sampled_generation_limit(128, 126, 10), 1);
        assert_eq!(sampled_generation_limit(128, 120, 4), 4);
    }

    #[test]
    fn detects_rendered_chat_prompt() {
        assert!(is_rendered_chat_prompt(
            "<｜begin▁of▁sentence｜>already rendered"
        ));
        assert!(!is_rendered_chat_prompt("plain user prompt"));
    }

    #[test]
    fn formats_thinking_tags_split_across_pieces() {
        assert_eq!(
            format_fragments(
                true,
                &[b"<thi", b"nk>plan", b"</thi", b"nk>answer"],
            ),
            b"plan\nanswer\n"
        );
        assert_eq!(
            format_fragments(true, &[b"<thi", b"x"]),
            b"<thix\n"
        );
    }

    #[test]
    fn preserves_c_thinking_newline_and_finish_rules() {
        assert_eq!(
            format_fragments(true, &[b"</think>", b"answer\n"]),
            b"answer\n"
        );
        assert_eq!(
            format_fragments(
                true,
                &[b"plan\n</think>", b"\nanswer"],
            ),
            b"plan\n\nanswer\n"
        );
        assert_eq!(
            format_fragments(true, &[b"plan</thi"]),
            b"plan</thi\n"
        );
        assert_eq!(format_fragments(true, &[]), b"");
    }

    #[test]
    fn leaves_nothink_output_unformatted() {
        assert_eq!(
            format_fragments(false, &[b"<thi", b"nk>x</think>"]),
            b"<think>x</think>\n"
        );
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
