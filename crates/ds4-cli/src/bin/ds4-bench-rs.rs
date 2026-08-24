fn main() {
    match ds4_cli::bench::parse_args(std::env::args()) {
        Ok(args) => match ds4_cli::bench::run(args) {
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("ds4-bench-rs: {err}");
                std::process::exit(1);
            }
        },
        Err(err) => {
            eprintln!("ds4-bench-rs: {err}");
            std::process::exit(2);
        }
    }
}
