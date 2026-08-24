use ds4_kv::{Options, Store};
use std::path::PathBuf;

#[derive(Debug)]
pub struct DiskKvArgs {
    dir: Option<PathBuf>,
    space_mb: u64,
    options: Options,
    reject_different_quant: bool,
}

impl Default for DiskKvArgs {
    fn default() -> Self {
        Self {
            dir: None,
            space_mb: 0,
            options: Options::default(),
            reject_different_quant: false,
        }
    }
}

fn require_value(option: &str, value: Option<String>) -> Result<String, String> {
    value.ok_or_else(|| format!("ds4-server-rs: missing value for {option}"))
}

fn positive_i32(option: &str, value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("ds4-server-rs: invalid value for {option}: {value}"))
}

impl DiskKvArgs {
    pub fn parse_arg(
        &mut self,
        option: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> Result<bool, String> {
        let mut value = || require_value(option, args.next());
        match option {
            "--kv-disk-dir" => self.dir = Some(value()?.into()),
            "--kv-disk-space-mb" => self.set_space_mb(&value()?)?,
            "--kv-cache-min-tokens" => self.set_min_tokens(&value()?)?,
            "--kv-cache-reject-different-quant" => self.reject_different_quant = true,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn set_space_mb(&mut self, value: &str) -> Result<(), String> {
        self.space_mb = positive_i32("--kv-disk-space-mb", value)? as u64;
        Ok(())
    }

    fn set_min_tokens(&mut self, value: &str) -> Result<(), String> {
        self.options.min_tokens = positive_i32("--kv-cache-min-tokens", value)?;
        Ok(())
    }

    pub fn open(&self) -> Option<Store> {
        let dir = self.dir.as_ref()?;
        match Store::open(
            dir,
            self.space_mb,
            self.reject_different_quant,
            self.options,
        ) {
            Ok(store) => {
                eprintln!(
                    "ds4-server-rs: KV disk cache {} (budget={} MiB, cross-quant={}, min={}; ordinary serial evict/load)",
                    dir.display(),
                    store.budget_bytes / (1024 * 1024),
                    if self.reject_different_quant { "reject" } else { "accept" },
                    self.options.min_tokens,
                );
                Some(store)
            }
            Err(error) => {
                eprintln!(
                    "ds4-server-rs: failed to create KV cache directory {}: {error}",
                    dir.display()
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DiskKvArgs;
    use ds4_kv::Options;
    use std::fs;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ds4-server-kv-cli-{tag}-{}", std::process::id()))
    }

    #[test]
    fn defaults_disable_disk_cache_and_keep_tuning_inert() {
        let mut kv = DiskKvArgs::default();
        assert!(kv.dir.is_none());
        assert_eq!(kv.space_mb, 0);
        assert_eq!(kv.options.min_tokens, Options::default().min_tokens);
        assert!(!kv.reject_different_quant);

        kv.set_space_mb("64").unwrap();
        kv.set_min_tokens("1024").unwrap();
        kv.reject_different_quant = true;
        assert!(kv.open().is_none());
    }

    #[test]
    fn ordinary_serial_kv_flags_parse_without_claiming_inactive_policy() {
        let mut kv = DiskKvArgs::default();
        let cases = [
            ("--kv-disk-dir", "cache"),
            ("--kv-disk-space-mb", "64"),
            ("--kv-cache-min-tokens", "1024"),
        ];
        for (option, value) in cases {
            let mut values = [value.to_string()].into_iter();
            assert!(kv.parse_arg(option, &mut values).unwrap());
        }
        let mut empty = std::iter::empty();
        assert!(kv
            .parse_arg("--kv-cache-reject-different-quant", &mut empty)
            .unwrap());
        for inactive in [
            "--kv-cache-cold-max-tokens",
            "--kv-cache-continued-interval-tokens",
            "--kv-cache-boundary-trim-tokens",
            "--kv-cache-boundary-align-tokens",
        ] {
            assert!(!kv.parse_arg(inactive, &mut empty).unwrap());
        }

        assert_eq!(kv.dir.as_deref(), Some(std::path::Path::new("cache")));
        assert_eq!(kv.space_mb, 64);
        assert_eq!(kv.options.min_tokens, 1024);
        assert!(kv.reject_different_quant);
    }

    #[test]
    fn positive_values_and_int_max_are_accepted() {
        let mut kv = DiskKvArgs::default();
        kv.set_space_mb("2147483647").unwrap();
        kv.set_min_tokens("2147483647").unwrap();
        assert_eq!(kv.space_mb, i32::MAX as u64);
        assert_eq!(kv.options.min_tokens, i32::MAX);
    }

    #[test]
    fn invalid_numeric_values_match_the_c_contract() {
        for value in ["", "0", "-1", "junk", "2147483648"] {
            let mut kv = DiskKvArgs::default();
            assert_eq!(
                kv.set_space_mb(value).unwrap_err(),
                format!("ds4-server-rs: invalid value for --kv-disk-space-mb: {value}")
            );
            assert_eq!(
                kv.set_min_tokens(value).unwrap_err(),
                format!("ds4-server-rs: invalid value for --kv-cache-min-tokens: {value}")
            );
        }
    }

    #[test]
    fn missing_value_is_reported() {
        let mut missing = DiskKvArgs::default();
        let mut empty = std::iter::empty();
        assert_eq!(
            missing.parse_arg("--kv-disk-dir", &mut empty).unwrap_err(),
            "ds4-server-rs: missing value for --kv-disk-dir"
        );
    }

    #[test]
    fn omitted_budget_opens_with_the_c_default() {
        let dir = temp_path("default-budget");
        let _ = fs::remove_dir_all(&dir);
        let mut kv = DiskKvArgs::default();
        kv.dir = Some(dir.clone());

        let store = kv.open().unwrap();

        assert_eq!(store.budget_bytes, 4096 * 1024 * 1024);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn directory_open_failure_is_returned_for_nonfatal_disable() {
        let parent = temp_path("open-failure");
        let _ = fs::remove_file(&parent);
        let _ = fs::remove_dir_all(&parent);
        fs::write(&parent, b"not a directory").unwrap();
        let mut kv = DiskKvArgs::default();
        kv.dir = Some(parent.join("child"));

        assert!(kv.open().is_none());

        let _ = fs::remove_file(parent);
    }
}
