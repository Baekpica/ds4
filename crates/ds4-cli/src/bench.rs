use ds4_core::{Backend, Model, Session, SessionSnapshot, TokenBuffer};
use std::io::{BufWriter, Write};
use std::time::Instant;

const CSV_HEADER: &str = "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_tps_ss,first_token_sec,kvcache_bytes";

#[derive(Debug)]
pub struct BenchArgs {
    model: String,
    prompt_file: Option<String>,
    backend: Backend,
    threads: i32,
    ctx_start: i32,
    ctx_max: i32,
    ctx_alloc: i32,
    step_incr: i32,
    gen_tokens: i32,
    step_mul: f64,
    csv: Option<String>,
    dist: ds4_dist::Options,
    help: bool,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            model: "ds4flash.gguf".into(),
            prompt_file: None,
            backend: default_backend(),
            threads: 0,
            ctx_start: 2048,
            ctx_max: 32768,
            ctx_alloc: 0,
            step_incr: 2048,
            gen_tokens: 128,
            step_mul: 1.0,
            csv: None,
            dist: ds4_dist::Options::default(),
            help: false,
        }
    }
}

struct BenchRow {
    ctx_tokens: i32,
    prefill_tokens: i32,
    prefill_tps: f64,
    gen_tokens: i32,
    gen_tps: f64,
    gen_tps_ss: f64,
    first_token_sec: f64,
    kvcache_bytes: u64,
}

impl BenchRow {
    fn csv_line(&self) -> String {
        format!(
            "{},{},{:.2},{},{:.2},{:.2},{:.4},{}",
            self.ctx_tokens,
            self.prefill_tokens,
            self.prefill_tps,
            self.gen_tokens,
            self.gen_tps,
            self.gen_tps_ss,
            self.first_token_sec,
            self.kvcache_bytes,
        )
    }
}

pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<BenchArgs, String> {
    let mut parsed = BenchArgs::default();
    let mut iter = args.into_iter();
    let _argv0 = iter.next();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                parsed.help = true;
                return Ok(parsed);
            }
            "-m" | "--model" => parsed.model = require_value(&arg, iter.next())?,
            "--prompt-file" => parsed.prompt_file = Some(require_value(&arg, iter.next())?),
            "--backend" => parsed.backend = parse_backend(&require_value(&arg, iter.next())?)?,
            "--cuda" => parsed.backend = Backend::Cuda,
            "--metal" => parsed.backend = Backend::Metal,
            "--cpu" => parsed.backend = Backend::Cpu,
            "-t" | "--threads" => {
                parsed.threads = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--ctx-start" => {
                parsed.ctx_start = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--ctx-max" => {
                parsed.ctx_max = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--ctx-alloc" => {
                parsed.ctx_alloc = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--step-incr" => {
                parsed.step_incr = parse_positive_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--step-mul" => {
                parsed.step_mul = parse_f64(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--gen-tokens" | "--tokens" | "-n" => {
                parsed.gen_tokens =
                    parse_nonnegative_i32(&arg, &require_value(&arg, iter.next())?)?;
            }
            "--csv" => parsed.csv = Some(require_value(&arg, iter.next())?),
            "--chat-prompt-file"
            | "-sys"
            | "--system"
            | "--mtp"
            | "--mtp-draft"
            | "--mtp-margin"
            | "--quality"
            | "--warm-weights"
            | "--power"
            | "--output-head-bench"
            | "--dump-frontier-logits-dir" => {
                return Err(format!("{arg} is not implemented in ds4-bench-rs"));
            }
            _ => match ds4_dist::parse_cli_arg(&arg, &mut iter, &mut parsed.dist)? {
                ds4_dist::CliResult::Matched => {}
                ds4_dist::CliResult::NotMatched => {
                    return Err(format!("unsupported ds4-bench-rs option: {arg}"));
                }
                ds4_dist::CliResult::Error => unreachable!(),
            },
        }
    }

    if parsed.prompt_file.is_none() {
        return Err("--prompt-file is required".into());
    }
    if parsed.ctx_start > parsed.ctx_max {
        return Err("--ctx-start must be <= --ctx-max".into());
    }
    if parsed.step_mul < 1.0 {
        return Err("--step-mul must be >= 1".into());
    }
    let live_tokens = parsed
        .ctx_max
        .checked_add(parsed.gen_tokens)
        .ok_or_else(|| "requested context is too large".to_string())?;
    if parsed.ctx_alloc == 0 {
        parsed.ctx_alloc = live_tokens
            .checked_add(1)
            .ok_or_else(|| "requested context is too large".to_string())?;
    }
    if parsed.ctx_alloc <= live_tokens {
        return Err("--ctx-alloc must be greater than measured context + gen-tokens".into());
    }
    ds4_dist::prepare_engine_options(&parsed.dist)?;
    if parsed.dist.role == ds4_dist::Role::Worker {
        return Err("--role worker is a serving mode; start workers with ./ds4".into());
    }
    Ok(parsed)
}

fn uses_distributed_replay(args: &BenchArgs) -> bool {
    args.dist.role == ds4_dist::Role::Coordinator
}

fn wait_distributed_route(session: &Session<'_>) -> Result<(), String> {
    let mut ticks = 0u32;
    loop {
        if session
            .distributed_route_ready()
            .map_err(|e| format!("distributed route readiness failed: {e}"))?
        {
            if ticks != 0 {
                eprintln!("ds4-bench-rs: distributed route ready");
            }
            return Ok(());
        }
        if ticks % 20 == 0 {
            eprintln!("ds4-bench-rs: waiting for distributed route: route incomplete");
        }
        ticks = ticks.wrapping_add(1);
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn maybe_warn_distributed_step_shape(args: &BenchArgs, session: &Session<'_>) {
    let chunk = if args.dist.prefill_chunk != 0 {
        args.dist.prefill_chunk
    } else {
        session.host().prefill_cap
    };
    if chunk != 0
        && args.step_mul == 1.0
        && args.step_incr > 0
        && (args.step_incr as u32) < chunk
        && args.ctx_start < args.ctx_max
    {
        eprintln!(
            "ds4-bench-rs: note: --step-incr={} is smaller than distributed prefill chunk {}; suffix rows will not show multi-chunk pipeline overlap",
            args.step_incr, chunk
        );
    }
}

pub fn run(args: BenchArgs) -> Result<i32, String> {
    if args.help {
        print!("{}\nDistributed:\n{}", help_text(), ds4_dist::USAGE);
        return Ok(0);
    }

    let prompt_path = args
        .prompt_file
        .as_deref()
        .ok_or_else(|| "--prompt-file is required".to_string())?;
    let text = std::fs::read_to_string(prompt_path)
        .map_err(|e| format!("failed to read {prompt_path}: {e}"))?;
    let native_dist = crate::distributed_config(&args.dist);
    let model = match native_dist.as_ref() {
        Some(config) => Model::open_distributed(
            &args.model,
            args.backend,
            args.threads,
            false,
            None,
            None,
            config,
        ),
        None => Model::open(&args.model, args.backend, args.threads, false),
    }
    .map_err(|e| e.to_string())?;
    let prompt = model.tokenize_text(&text).map_err(|e| e.to_string())?;
    if prompt.len() < args.ctx_max as usize {
        return Err(format!(
            "prompt has {} tokens, need at least {}",
            prompt.len(),
            args.ctx_max
        ));
    }

    let mut session = model.session(args.ctx_alloc).map_err(|e| e.to_string())?;
    if uses_distributed_replay(&args) {
        wait_distributed_route(&session)?;
        maybe_warn_distributed_step_shape(&args, &session);
    }
    let mut snapshot = if uses_distributed_replay(&args) {
        None
    } else {
        Some(SessionSnapshot::new().map_err(|e| e.to_string())?)
    };

    if let Some(path) = args.csv.as_deref() {
        let file =
            std::fs::File::create(path).map_err(|e| format!("failed to open {path}: {e}"))?;
        let mut out = BufWriter::new(file);
        run_sweep(
            &args,
            &model,
            &prompt,
            &mut session,
            &mut snapshot,
            &mut out,
        )?;
    } else {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        run_sweep(
            &args,
            &model,
            &prompt,
            &mut session,
            &mut snapshot,
            &mut out,
        )?;
    }
    Ok(0)
}

fn run_sweep<W: Write>(
    args: &BenchArgs,
    model: &Model,
    prompt: &TokenBuffer,
    session: &mut Session<'_>,
    snapshot: &mut Option<SessionSnapshot>,
    out: &mut W,
) -> Result<(), String> {
    writeln!(out, "{CSV_HEADER}").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;

    let eos = model.token_eos();
    let distributed = uses_distributed_replay(args);
    let mut previous = 0;
    let mut frontier = args.ctx_start;

    loop {
        let prefix = TokenBuffer::from_tokens(prompt.as_slice()[..frontier as usize].to_vec());
        let prefill_t0 = Instant::now();
        session.sync(&prefix).map_err(|e| e.to_string())?;
        let prefill_sec = prefill_t0.elapsed().as_secs_f64();
        let prefill_tokens = frontier - previous;

        if args.gen_tokens > 0 && !distributed {
            let snapshot = snapshot
                .as_mut()
                .ok_or_else(|| "local bench snapshot is missing".to_string())?;
            session
                .save_snapshot(snapshot)
                .map_err(|e| format!("snapshot at {frontier} failed: {e}"))?;
        }

        let gen_t0 = Instant::now();
        let mut generated = 0;
        let mut after_first = None;
        let mut first_call_tokens = 0;
        while generated < args.gen_tokens {
            if session.pos().saturating_add(1) >= session.ctx() {
                return Err(format!(
                    "generation would exceed allocated context at frontier {frontier}"
                ));
            }
            let token = session.argmax_excluding(eos);
            if token < 0 {
                return Err(format!(
                    "failed to choose non-EOS token at frontier {frontier}"
                ));
            }
            session
                .eval(token)
                .map_err(|e| format!("decode at frontier {frontier} failed: {e}"))?;
            generated += 1;
            if after_first.is_none() {
                after_first = Some(Instant::now());
                first_call_tokens = generated;
            }
        }
        let gen_t1 = Instant::now();

        if args.gen_tokens > 0 && frontier < args.ctx_max {
            if distributed {
                session
                    .sync(&prefix)
                    .map_err(|e| format!("distributed replay restore at {frontier} failed: {e}"))?;
            } else {
                let snapshot = snapshot
                    .as_ref()
                    .ok_or_else(|| "local bench snapshot is missing".to_string())?;
                session
                    .load_snapshot(snapshot)
                    .map_err(|e| format!("restore at {frontier} failed: {e}"))?;
            }
        }

        let gen_sec = gen_t1.duration_since(gen_t0).as_secs_f64();
        let first_token_sec = after_first
            .map(|t| t.duration_since(gen_t0).as_secs_f64())
            .unwrap_or(0.0);
        let ss_sec = after_first
            .map(|t| gen_t1.duration_since(t).as_secs_f64())
            .unwrap_or(0.0);
        let ss_tokens = args.gen_tokens - first_call_tokens;
        let row = BenchRow {
            ctx_tokens: frontier,
            prefill_tokens,
            prefill_tps: rate(prefill_tokens, prefill_sec),
            gen_tokens: args.gen_tokens,
            gen_tps: rate(args.gen_tokens, gen_sec),
            gen_tps_ss: rate(ss_tokens, ss_sec),
            first_token_sec,
            kvcache_bytes: if distributed {
                0
            } else {
                snapshot.as_ref().map_or(0, SessionSnapshot::len)
            },
        };
        writeln!(out, "{}", row.csv_line()).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;

        previous = frontier;
        if frontier >= args.ctx_max {
            break;
        }
        frontier = next_frontier(args, frontier);
    }
    Ok(())
}

fn rate(tokens: i32, seconds: f64) -> f64 {
    if tokens > 0 && seconds > 0.0 {
        f64::from(tokens) / seconds
    } else {
        0.0
    }
}

fn next_frontier(args: &BenchArgs, cur: i32) -> i32 {
    if cur >= args.ctx_max {
        return args.ctx_max;
    }
    let next = if args.step_mul == 1.0 {
        cur.checked_add(args.step_incr).unwrap_or(args.ctx_max)
    } else {
        let value = (f64::from(cur) * args.step_mul).ceil();
        if value > f64::from(i32::MAX) {
            args.ctx_max
        } else {
            let next = value as i32;
            if next <= cur {
                cur + 1
            } else {
                next
            }
        }
    };
    next.min(args.ctx_max)
}

fn require_value(flag: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive_i32(flag: &str, value: &str) -> Result<i32, String> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))?;
    if parsed <= 0 {
        return Err(format!("invalid value for {flag}: {value}"));
    }
    Ok(parsed)
}

fn parse_nonnegative_i32(flag: &str, value: &str) -> Result<i32, String> {
    let parsed = value
        .parse::<i32>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))?;
    if parsed < 0 {
        return Err(format!("invalid value for {flag}: {value}"));
    }
    Ok(parsed)
}

fn parse_f64(flag: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid value for {flag}: {value}"))?;
    if !parsed.is_finite() {
        return Err(format!("invalid value for {flag}: {value}"));
    }
    Ok(parsed)
}

fn parse_backend(value: &str) -> Result<Backend, String> {
    match value {
        "cuda" => Ok(Backend::Cuda),
        "metal" => Ok(Backend::Metal),
        "cpu" => Ok(Backend::Cpu),
        _ => Err(format!("invalid backend: {value}")),
    }
}

#[cfg(target_os = "macos")]
fn default_backend() -> Backend {
    Backend::Metal
}

#[cfg(not(target_os = "macos"))]
fn default_backend() -> Backend {
    Backend::Cuda
}

fn help_text() -> &'static str {
    "Usage: ds4-bench-rs --prompt-file FILE [options]\n\
     \n\
     Non-MTP throughput sweep over one fixed raw prompt.\n\
     \n\
     -m, --model FILE       GGUF model path (default: ds4flash.gguf)\n\
     --cuda|--metal|--cpu   Select backend\n\
     -t, --threads N        CPU helper threads\n\
     --prompt-file FILE     Raw UTF-8 benchmark prompt\n\
     --ctx-start N          First frontier (default: 2048)\n\
     --ctx-max N            Last frontier (default: 32768)\n\
     --ctx-alloc N          Allocated context\n\
     --step-mul F           Multiplicative step (default: 1)\n\
     --step-incr N          Linear step (default: 2048)\n\
     --gen-tokens N         Greedy decode tokens (default: 128)\n\
     --csv FILE             Write CSV instead of stdout\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once("ds4-bench-rs")
            .chain(args.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn parses_mmq_harness_arguments() {
        let args = parse_args(argv(&[
            "--cuda",
            "--model",
            "model.gguf",
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "1024",
            "--ctx-max",
            "16384",
            "--step-mul",
            "2",
            "--step-incr",
            "2048",
            "--gen-tokens",
            "128",
            "--csv",
            "out.csv",
        ]))
        .unwrap();

        assert_eq!(args.backend, ds4_core::Backend::Cuda);
        assert_eq!(args.model, "model.gguf");
        assert_eq!(args.prompt_file.as_deref(), Some("prompt.txt"));
        assert_eq!(args.ctx_alloc, 16513);
        assert_eq!(args.csv.as_deref(), Some("out.csv"));
    }

    #[test]
    fn parses_distributed_coordinator_for_native_bench_runtime() {
        let args = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--role",
            "coordinator",
            "--layers",
            "0:20",
            "--listen",
            "127.0.0.1",
            "7000",
            "--dist-prefill-chunk",
            "4096",
            "--dist-prefill-window",
            "4",
            "--dist-activation-bits",
            "16",
        ]))
        .unwrap();

        assert_eq!(args.dist.role, ds4_dist::Role::Coordinator);
        assert_eq!(args.dist.layers.start, 0);
        assert_eq!(args.dist.layers.end, 20);
        assert_eq!(args.dist.listen_host.as_deref(), Some("127.0.0.1"));
        assert_eq!(args.dist.listen_port, 7000);
        assert_eq!(args.dist.prefill_chunk, 4096);
        assert_eq!(args.dist.prefill_window, 4);
        assert_eq!(args.dist.activation_bits, 16);
        assert!(uses_distributed_replay(&args));
    }

    #[test]
    fn rejects_distributed_worker_as_a_serving_mode() {
        let err = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--role",
            "worker",
            "--layers",
            "21:output",
            "--coordinator",
            "127.0.0.1",
            "7000",
        ]))
        .unwrap_err();

        assert_eq!(
            err,
            "--role worker is a serving mode; start workers with ./ds4"
        );
    }

    #[test]
    fn walks_linear_and_multiplicative_frontiers() {
        let args = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "1024",
            "--ctx-max",
            "16384",
            "--step-mul",
            "2",
        ]))
        .unwrap();
        let mut got = Vec::new();
        let mut cur = args.ctx_start;
        loop {
            got.push(cur);
            if cur >= args.ctx_max {
                break;
            }
            cur = next_frontier(&args, cur);
        }
        assert_eq!(got, [1024, 2048, 4096, 8192, 16384]);

        let linear = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "2048",
            "--ctx-max",
            "6144",
            "--step-incr",
            "2048",
        ]))
        .unwrap();
        assert_eq!(next_frontier(&linear, 2048), 4096);
        assert_eq!(next_frontier(&linear, 4096), 6144);
    }

    #[test]
    fn validates_context_and_formats_c_csv() {
        let err = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "2048",
            "--ctx-max",
            "2048",
            "--ctx-alloc",
            "2176",
            "--gen-tokens",
            "128",
        ]))
        .unwrap_err();
        assert!(err.contains("ctx-alloc"));

        let row = BenchRow {
            ctx_tokens: 2048,
            prefill_tokens: 2048,
            prefill_tps: 123.456,
            gen_tokens: 128,
            gen_tps: 10.004,
            gen_tps_ss: 9.5,
            first_token_sec: 0.12345,
            kvcache_bytes: 4096,
        };
        assert_eq!(
            CSV_HEADER,
            "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_tps_ss,first_token_sec,kvcache_bytes"
        );
        assert_eq!(
            row.csv_line(),
            "2048,2048,123.46,128,10.00,9.50,0.1235,4096"
        );
    }

    #[test]
    fn rejects_unported_mtp_benchmark_mode() {
        let mtp = parse_args(argv(&[
            "--prompt-file",
            "prompt.txt",
            "--mtp",
            "draft.gguf",
        ]))
        .unwrap_err();
        assert!(mtp.contains("not implemented"));
    }
}
