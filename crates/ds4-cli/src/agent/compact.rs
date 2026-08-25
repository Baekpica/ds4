const SOFT_PERCENT: i32 = 85;
const MIN_FREE_TOKENS: i32 = 8192;
const TAIL_DIVISOR: i32 = 10;
const TAIL_CAP_TOKENS: i32 = 50_000;
pub(crate) const TOOL_RESULT_RESERVE_TOKENS: i32 = 1024;

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn should_compact(ctx: i32, used: i32) -> bool {
    if ctx <= 0 || used <= 0 {
        return false;
    }
    if used >= (ctx * SOFT_PERCENT) / 100 {
        return true;
    }
    let proportional = ctx / 4;
    let free_threshold = MIN_FREE_TOKENS.min(proportional);
    ctx - used <= free_threshold
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn tail_start(ctx: i32, bottom: i32, sys_len: i32, user_id: i32, tokens: &[i32]) -> i32 {
    let mut tail_budget = ctx / TAIL_DIVISOR;
    if tail_budget > TAIL_CAP_TOKENS {
        tail_budget = TAIL_CAP_TOKENS;
    }
    if tail_budget < 1 {
        tail_budget = 1;
    }
    let mut target = bottom - tail_budget;
    if target < sys_len {
        target = sys_len;
    }
    if user_id < 0 {
        return target;
    }
    let start = usize::try_from(target.max(0)).unwrap_or(0);
    let end = usize::try_from(bottom.max(0))
        .unwrap_or(0)
        .min(tokens.len());
    for (index, token) in tokens
        .iter()
        .enumerate()
        .skip(start)
        .take(end.saturating_sub(start))
    {
        if *token == user_id {
            return index as i32;
        }
    }
    target
}

pub(crate) fn overflow_error(projected: i32, ctx: i32) -> Vec<u8> {
    format!(
        "Tool error: tool result still does not fit after context compaction \
         (projected_prompt={projected} tokens, ctx={ctx}, reserve={TOOL_RESULT_RESERVE_TOKENS}). \
         Retry with a smaller read/search/bash output.\n"
    )
    .into_bytes()
}
