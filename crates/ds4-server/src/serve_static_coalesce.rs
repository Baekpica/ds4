//! C `coalesce_gather` + StaticBatchContext overflow, without a new scheduler.

use crate::route::WireSurface;

/// C `DS4_COALESCE_HARD_MAX`.
pub const COALESCE_HARD_MAX: usize = 64;

/// C worker gather bounds: `cmax` and `cmaxtok` (`0` = unbounded tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoalesceLimits {
    pub cap: usize,
    pub max_tok_total: i64,
}

impl CoalesceLimits {
    pub const UNBOUNDED: Self = Self {
        cap: COALESCE_HARD_MAX,
        max_tok_total: 0,
    };

    pub const fn clamp(self) -> Self {
        let cap = if self.cap < 1 {
            1
        } else if self.cap > COALESCE_HARD_MAX {
            COALESCE_HARD_MAX
        } else {
            self.cap
        };
        Self {
            cap,
            max_tok_total: if self.max_tok_total < 0 {
                0
            } else {
                self.max_tok_total
            },
        }
    }
}

impl Default for CoalesceLimits {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

/// C `job_tok_footprint`: prompt tokens + requested decode budget.
pub const fn job_tok_footprint(prompt_len: usize, max_new_tokens: i32) -> i64 {
    let budget = if max_new_tokens > 0 {
        max_new_tokens as i64
    } else {
        0
    };
    prompt_len as i64 + budget
}

/// One FIFO candidate in front of `coalesce_gather`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoalescePeer {
    pub footprint: i64,
    pub peer_ok: bool,
}

/// C `job_static_peer_ok`: needs-free on a static-servable surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticPeerSpec {
    pub needs: u32,
    pub surface: WireSurface,
    pub cont_anthropic: bool,
    pub cont_responses: bool,
}

pub const fn static_peer_ok(spec: StaticPeerSpec) -> bool {
    if spec.needs != 0 {
        return false;
    }
    match spec.surface {
        WireSurface::OpenaiChat | WireSurface::OpenaiCompletion => true,
        WireSurface::Anthropic => spec.cont_anthropic,
        WireSurface::Responses => spec.cont_responses,
    }
}

/// Extra jobs taken from the front of `queued`. Head is already counted as 1.
pub fn coalesce_take(
    head_footprint: i64,
    queued: &[CoalescePeer],
    limits: CoalesceLimits,
) -> usize {
    let limits = limits.clamp();
    let mut n = 1usize;
    let mut tok_total = head_footprint;
    for peer in queued {
        if n >= limits.cap {
            break;
        }
        if !peer.peer_ok {
            break;
        }
        if limits.max_tok_total > 0 && tok_total + peer.footprint > limits.max_tok_total {
            break;
        }
        tok_total += peer.footprint;
        n += 1;
    }
    n - 1
}

/// C `use_ctx` negation: group does not fit the persistent StaticBatchContext.
pub const fn static_ctx_overflow(
    n: usize,
    packed_prompt: i64,
    max_seq: i32,
    max_tokens: i32,
) -> bool {
    max_seq <= 0 || n > max_seq as usize || packed_prompt > max_tokens as i64
}
