//! Host-ledger token counts for KV policy. Callers pass these fields;
//! policy never reads a native session or CUDA scratch.

use crate::policy::{bank_checkpoint_due, continued_store_target, Options};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostKvView {
    pub live_tokens: i32,
    pub stored_tokens: i32,
}

pub fn continued_store_target_from_host(opt: &Options, host: HostKvView) -> i32 {
    continued_store_target(opt, host.stored_tokens, host.live_tokens)
}

pub fn bank_checkpoint_due_from_host(opt: &Options, host: HostKvView) -> bool {
    bank_checkpoint_due(opt, host.live_tokens, host.stored_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continued_target_uses_host_live_tokens_not_native_pos() {
        let opt = Options::default();
        let due = HostKvView {
            live_tokens: 10240,
            stored_tokens: 0,
        };
        let raw_interval = HostKvView {
            live_tokens: 10000,
            stored_tokens: 0,
        };
        assert_eq!(continued_store_target_from_host(&opt, due), 10240);
        assert_eq!(continued_store_target_from_host(&opt, raw_interval), 0);
    }

    #[test]
    fn bank_checkpoint_due_uses_host_committed_frontier() {
        let opt = Options::default();
        assert!(bank_checkpoint_due_from_host(
            &opt,
            HostKvView {
                live_tokens: 10240,
                stored_tokens: 0,
            }
        ));
        assert!(!bank_checkpoint_due_from_host(
            &opt,
            HostKvView {
                live_tokens: 10000,
                stored_tokens: 0,
            }
        ));
        assert!(!bank_checkpoint_due_from_host(
            &opt,
            HostKvView {
                live_tokens: 9999,
                stored_tokens: 0,
            }
        ));
    }
}
