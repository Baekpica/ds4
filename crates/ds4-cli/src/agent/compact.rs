const SOFT_PERCENT: i32 = 85;
const MIN_FREE_TOKENS: i32 = 8192;
const TAIL_DIVISOR: i32 = 10;
const TAIL_CAP_TOKENS: i32 = 50_000;
pub(crate) const TOOL_RESULT_RESERVE_TOKENS: i32 = 1024;
pub(crate) const SUMMARY_MAX_TOKENS: i32 = 4096;

const SUMMARY_PROMPT: &str = "\
Internal ds4-agent context compaction request. This is not a user request.\n\
Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n\
- user goals, constraints, and preferences\n\
- files inspected or edited\n\
- commands run and important results\n\
- decisions, rejected approaches, known bugs, and pending next steps\n\
- reloadable bulky data with exact paths/ranges/commands when available\n\n\
Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n\
After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n\
Output only the compact summary.\n";

const SUMMARY_HEAD: &str =
    "\n\n[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n";
const SUMMARY_TAIL: &str =
    "[End compacted summary. Recent conversation continues verbatim below.]\n\n";

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

pub(crate) fn summary_prompt(reason: &str) -> String {
    if reason.is_empty() {
        SUMMARY_PROMPT.to_string()
    } else {
        format!("{SUMMARY_PROMPT}\nCompaction reason: {reason}\n")
    }
}

pub(crate) fn wrap_summary(summary: &str) -> String {
    let mut out = String::from(SUMMARY_HEAD);
    out.push_str(summary);
    if !summary.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(SUMMARY_TAIL);
    out
}

pub(crate) fn special_token_id(vocab: &ds4_core::Vocab, rendered: &str) -> i32 {
    let tokens = vocab.encode_rendered_chat(rendered);
    if tokens.len() == 1 {
        tokens[0]
    } else {
        -1
    }
}

pub(crate) fn compact_if_needed(
    model: &ds4_core::Model,
    session: &mut ds4_core::Session<'_>,
    vocab: &ds4_core::Vocab,
    ctx: i32,
    sys: &ds4_core::TokenBuffer,
    transcript: &mut ds4_core::TokenBuffer,
    reason: &str,
) -> Result<(), String> {
    let used = i32::try_from(transcript.len()).unwrap_or(i32::MAX);
    if !should_compact(ctx, used) {
        return Ok(());
    }
    compact(model, session, vocab, ctx, sys, transcript, reason)
}

pub(crate) fn compact(
    model: &ds4_core::Model,
    session: &mut ds4_core::Session<'_>,
    vocab: &ds4_core::Vocab,
    ctx: i32,
    sys: &ds4_core::TokenBuffer,
    transcript: &mut ds4_core::TokenBuffer,
    reason: &str,
) -> Result<(), String> {
    let bottom = i32::try_from(transcript.len()).unwrap_or(i32::MAX);
    if bottom <= 0 {
        return Ok(());
    }
    let sys_len = i32::try_from(sys.len()).unwrap_or(i32::MAX);
    if bottom <= sys_len {
        return Ok(());
    }

    eprintln!(
        "COMPACTING {}: summarizing durable task state",
        if reason.is_empty() { "context" } else { reason }
    );

    let mut prompt = ds4_core::TokenBuffer::from_tokens(transcript.as_slice().to_vec());
    vocab
        .chat_append_message(&mut prompt, "user", summary_prompt(reason).as_bytes())
        .map_err(|error| error.to_string())?;
    vocab
        .chat_append_assistant_prefix(&mut prompt, ds4_core::ChatThinkMode::None)
        .map_err(|error| error.to_string())?;

    let prompt_len = i32::try_from(prompt.len()).unwrap_or(i32::MAX);
    let summary_room = ctx.saturating_sub(prompt_len).saturating_sub(1);
    if summary_room < 256 {
        session.invalidate();
        return Err("not enough context left to request compaction summary".into());
    }
    let summary_max = summary_room.min(SUMMARY_MAX_TOKENS);
    if let Err(error) = session.sync(&prompt) {
        session.invalidate();
        return Err(error.to_string());
    }

    let think_end_id = special_token_id(vocab, "</think>");
    let dsml_id = special_token_id(vocab, "｜DSML｜");
    let eos = model.token_eos();
    let mut summary = Vec::new();
    for _ in 0..summary_max {
        let token = session.argmax();
        if token == eos {
            break;
        }
        if token == think_end_id || token == dsml_id {
            if token == dsml_id && summary.last() == Some(&b'<') {
                summary.pop();
            }
            break;
        }
        if let Err(error) = session.eval(token) {
            session.invalidate();
            return Err(error.to_string());
        }
        let piece = model.token_text(token).map_err(|error| error.to_string())?;
        summary.extend_from_slice(&piece);
        eprint!("{}", String::from_utf8_lossy(&piece));
    }
    eprintln!();
    if summary.is_empty() {
        session.invalidate();
        return Err("compaction summary was empty".into());
    }

    let user_id = special_token_id(vocab, "<｜User｜>");
    let tail = tail_start(ctx, bottom, sys_len, user_id, transcript.as_slice());
    let tail_start_idx = usize::try_from(tail.max(0)).unwrap_or(0);
    let bottom_idx = usize::try_from(bottom.max(0)).unwrap_or(0);
    let mut compacted = ds4_core::TokenBuffer::from_tokens(sys.as_slice().to_vec());
    let wrapped = wrap_summary(&String::from_utf8_lossy(&summary));
    vocab
        .chat_append_message(&mut compacted, "system", wrapped.as_bytes())
        .map_err(|error| error.to_string())?;
    for token in transcript
        .as_slice()
        .get(tail_start_idx..bottom_idx)
        .unwrap_or(&[])
    {
        compacted.push(*token);
    }

    eprintln!(
        "COMPACTING rebuilding context: old={bottom} summary+tail={} tail={}",
        compacted.len(),
        bottom_idx.saturating_sub(tail_start_idx)
    );

    let previous = ds4_core::TokenBuffer::from_tokens(transcript.as_slice().to_vec());
    *transcript = compacted;
    if let Err(error) = session.sync(transcript) {
        session.invalidate();
        *transcript = previous;
        return Err(error.to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_prompt_matches_c() {
        assert!(summary_prompt("").starts_with(
            "Internal ds4-agent context compaction request. This is not a user request.\n"
        ));
        assert!(summary_prompt("").ends_with("Output only the compact summary.\n"));
        assert_eq!(
            summary_prompt("soft limit before user turn"),
            format!(
                "{}\nCompaction reason: soft limit before user turn\n",
                summary_prompt("")
            )
        );
    }

    #[test]
    fn wrap_summary_matches_c() {
        assert_eq!(
            wrap_summary("kept files"),
            "\n\n[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n\
             kept files\n\
             [End compacted summary. Recent conversation continues verbatim below.]\n\n"
        );
        assert_eq!(
            wrap_summary("kept files\n"),
            "\n\n[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n\
             kept files\n\
             [End compacted summary. Recent conversation continues verbatim below.]\n\n"
        );
    }
}
