fn main() {
    match ds4_cli::agent::parse_args(std::env::args()) {
        Ok(args) => match ds4_cli::agent::run("ds4-agent-rs", args) {
            Ok(code) => std::process::exit(code),
            Err(error) => {
                eprintln!("ds4-agent-rs: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("ds4-agent-rs: {error}");
            std::process::exit(2);
        }
    }
}
