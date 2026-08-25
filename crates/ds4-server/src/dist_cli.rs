//! Distributed CLI flags for `ds4-server-rs`, matching C `ds4_dist_parse_cli_arg`.

use ds4_dist::{parse_cli_arg, prepare_engine_options, CliResult, Options, Role};

#[derive(Debug, Default)]
pub struct DistArgs {
    pub opt: Options,
}

impl DistArgs {
    pub fn parse_arg(
        &mut self,
        option: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> Result<bool, String> {
        match parse_cli_arg(option, args, &mut self.opt)? {
            CliResult::Matched => Ok(true),
            CliResult::NotMatched => Ok(false),
            CliResult::Error => unreachable!(),
        }
    }

    /// C uses `--listen` for the coordinator data port and `--host`/`--port`
    /// for HTTP. The Rust shadow used `--listen` for HTTP; if no `--role` is
    /// set, keep that meaning.
    pub fn finish(&mut self, http_host: &mut String, http_port: &mut u16) -> Result<(), String> {
        if self.opt.role == Role::None {
            if let Some(host) = self.opt.listen_host.take() {
                if self.opt.listen_port != 0 {
                    *http_host = host;
                    *http_port = u16::try_from(self.opt.listen_port)
                        .map_err(|_| "ds4-server-rs: --listen port is out of range".to_string())?;
                    self.opt.listen_port = 0;
                }
            }
        }
        prepare_engine_options(&self.opt)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DistArgs;
    use ds4_dist::Role;

    fn parse(args: &[&str]) -> Result<(DistArgs, String, u16), String> {
        let mut dist = DistArgs::default();
        let mut host = "127.0.0.1".to_string();
        let mut port = 8000u16;
        let mut rest = args.iter().map(|s| (*s).to_string());
        while let Some(arg) = rest.next() {
            if !dist.parse_arg(&arg, &mut rest)? {
                return Err(format!("unmatched {arg}"));
            }
        }
        dist.finish(&mut host, &mut port)?;
        Ok((dist, host, port))
    }

    #[test]
    fn listen_without_role_stays_http() {
        let (dist, host, port) = parse(&["--listen", "0.0.0.0", "9001"]).unwrap();
        assert_eq!(dist.opt.role, Role::None);
        assert!(dist.opt.listen_host.is_none());
        assert_eq!(dist.opt.listen_port, 0);
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 9001);
    }

    #[test]
    fn coordinator_keeps_listen_for_workers() {
        let (dist, host, port) = parse(&[
            "--role",
            "coordinator",
            "--layers",
            "0:19",
            "--listen",
            "169.254.1.1",
            "1234",
        ])
        .unwrap();
        assert_eq!(dist.opt.role, Role::Coordinator);
        assert_eq!(dist.opt.listen_host.as_deref(), Some("169.254.1.1"));
        assert_eq!(dist.opt.listen_port, 1234);
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 8000);
    }

    #[test]
    fn worker_requires_coordinator() {
        let err = parse(&["--role", "worker", "--layers", "20:output"]).unwrap_err();
        assert!(err.contains("--coordinator"));
    }

    #[test]
    fn worker_parses_coordinator_address() {
        let (dist, _, _) = parse(&[
            "--role",
            "worker",
            "--layers",
            "20:output",
            "--coordinator",
            "127.0.0.1",
            "1234",
        ])
        .unwrap();
        assert_eq!(dist.opt.role, Role::Worker);
        assert_eq!(dist.opt.coordinator_host.as_deref(), Some("127.0.0.1"));
        assert_eq!(dist.opt.coordinator_port, 1234);
        assert!(dist.opt.layers.has_output);
    }
}
