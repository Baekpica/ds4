//! Coordinator KV gather order from `ds4_dist_session_save_payload`.
//!
//! Host owns shard walk, empty-shard refusal, and token hashing. Layer
//! payload bytes and DSV4 header/logits merge stay native.

use crate::hash::token_hash_prefix;

pub const ERR_NO_TIMELINE: &str = "distributed session has no valid token timeline";
pub const ERR_EMPTY_SHARD: &str = "distributed KV shard is empty";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherShard {
    pub layer_start: u32,
    pub layer_end: u32,
    pub local: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatheredShards {
    pub token_count: u32,
    pub token_hash: u64,
    pub shards: Vec<(GatherShard, Vec<u8>)>,
}

/// C save walk: shard 0 is the owner layer payload; later shards are
/// remote. Empty shard bytes fail before any DSV4 merge.
pub fn gather_kv_shards<F>(
    tokens: &[i32],
    plan: &[GatherShard],
    mut fetch: F,
) -> Result<GatheredShards, String>
where
    F: FnMut(GatherShard) -> Result<Vec<u8>, String>,
{
    if tokens.len() > u32::MAX as usize {
        return Err(ERR_NO_TIMELINE.into());
    }
    if plan.is_empty() {
        return Err(ERR_EMPTY_SHARD.into());
    }
    let token_count = tokens.len() as u32;
    let token_hash = token_hash_prefix(tokens);
    let mut shards = Vec::with_capacity(plan.len());
    for (i, spec) in plan.iter().copied().enumerate() {
        if i == 0 && !spec.local {
            return Err("distributed KV shard 0 must be the owner layer".into());
        }
        if i != 0 && spec.local {
            return Err("distributed KV remote shard marked local".into());
        }
        let bytes = fetch(spec)?;
        if bytes.is_empty() {
            return Err(ERR_EMPTY_SHARD.into());
        }
        shards.push((spec, bytes));
    }
    Ok(GatheredShards {
        token_count,
        token_hash,
        shards,
    })
}

/// C load walk: shard 0 is the owner layer payload; later shards are
/// remote. Host owns the order. Layer apply and DSV4 parse stay native.
pub fn scatter_kv_shards<F>(
    tokens: &[i32],
    plan: &[GatherShard],
    shards: &[Vec<u8>],
    mut apply: F,
) -> Result<u64, String>
where
    F: FnMut(GatherShard, &[u8]) -> Result<(), String>,
{
    if tokens.len() > u32::MAX as usize {
        return Err(ERR_NO_TIMELINE.into());
    }
    if plan.is_empty() || plan.len() != shards.len() {
        return Err(ERR_EMPTY_SHARD.into());
    }
    for (i, (spec, bytes)) in plan.iter().copied().zip(shards.iter()).enumerate() {
        if i == 0 && !spec.local {
            return Err("distributed KV shard 0 must be the owner layer".into());
        }
        if i != 0 && spec.local {
            return Err("distributed KV remote shard marked local".into());
        }
        if bytes.is_empty() {
            return Err(ERR_EMPTY_SHARD.into());
        }
        apply(spec, bytes)?;
    }
    Ok(token_hash_prefix(tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_shard_plan() -> [GatherShard; 2] {
        [
            GatherShard {
                layer_start: 0,
                layer_end: 10,
                local: true,
            },
            GatherShard {
                layer_start: 10,
                layer_end: 20,
                local: false,
            },
        ]
    }

    #[test]
    fn gather_walks_owner_then_remote() {
        let tokens = [1i32, 2, 3];
        let got = gather_kv_shards(&tokens, &two_shard_plan(), |spec| {
            Ok(vec![spec.layer_start as u8, spec.layer_end as u8])
        })
        .unwrap();
        assert_eq!(got.token_count, 3);
        assert_eq!(got.token_hash, token_hash_prefix(&tokens));
        assert_eq!(got.shards.len(), 2);
        assert!(got.shards[0].0.local);
        assert!(!got.shards[1].0.local);
        assert_eq!(got.shards[0].1, vec![0, 10]);
        assert_eq!(got.shards[1].1, vec![10, 20]);
    }

    #[test]
    fn gather_refuses_empty_shard() {
        let err = gather_kv_shards(&[1], &two_shard_plan(), |spec| {
            if spec.local {
                Ok(vec![1])
            } else {
                Ok(vec![])
            }
        })
        .unwrap_err();
        assert_eq!(err, ERR_EMPTY_SHARD);
    }

    #[test]
    fn gather_requires_owner_first() {
        let plan = [GatherShard {
            layer_start: 0,
            layer_end: 1,
            local: false,
        }];
        let err = gather_kv_shards(&[1], &plan, |_| Ok(vec![1])).unwrap_err();
        assert!(err.contains("owner layer"));
    }

    #[test]
    fn gather_accepts_empty_token_timeline() {
        let got = gather_kv_shards(&[], &two_shard_plan(), |_| Ok(vec![1])).unwrap();
        assert_eq!(got.token_count, 0);
        assert_eq!(got.token_hash, token_hash_prefix(&[]));
    }

    #[test]
    fn scatter_walks_owner_then_remote() {
        let tokens = [7i32];
        let mut seen = Vec::new();
        let hash = scatter_kv_shards(
            &tokens,
            &two_shard_plan(),
            &[vec![1], vec![2]],
            |spec, bytes| {
                seen.push((spec.local, bytes.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(hash, token_hash_prefix(&tokens));
        assert_eq!(seen, vec![(true, vec![1]), (false, vec![2])]);
    }

    #[test]
    fn scatter_refuses_empty_shard() {
        let err = scatter_kv_shards(&[1], &two_shard_plan(), &[vec![1], vec![]], |_, _| Ok(()))
            .unwrap_err();
        assert_eq!(err, ERR_EMPTY_SHARD);
    }
}
