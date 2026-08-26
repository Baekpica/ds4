//! C `serial_session_ensure_fit` decision logic (`ds4_server.c` v0.5.2 inc1).
//!
//! The serial session is created at server `-c` and its S6 lazy graph is
//! sized by that ctx, not the request. On a bank-holding boot the full
//! graph can never fit and every serial request dies at the alloc. A serial
//! request needs only prompt+budget positions, so when the full graph
//! cannot fit the session is re-created at the largest ctx the fit gate
//! passes, bounded by what the request could use. When even a minimal
//! output window cannot fit, refuse 503 (retryable capacity), never the
//! doomed alloc. Pure arithmetic lives here; the session swap driver is in
//! `serve.rs` (`ensure_serial_session_fit`).

/// C `serial_frame_kind`. `RequiredMiss` never reaches the host driver —
/// `run_serial` answers the protocol-native 409 before generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SerialFrameKind {
    Canonical,
    ResolvedLive,
    RequiredMiss,
}

/// C `serial_fit_plan`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SerialFitPlan {
    Reuse,
    Resize,
    RefusePreserve,
    PassNative,
}

/// C `serial_session_reuse_ok`: a graphless boot-shape session has no state
/// to preserve; do not let a tiny first serial request commit a deep
/// full-`-c` graph merely because it fits before the batch banks grow.
pub(crate) fn serial_session_reuse_ok(
    cur_ctx: i64,
    graph_pending: bool,
    need_min: i64,
    request_cap: i64,
    current_graph_fits: bool,
) -> bool {
    if cur_ctx < need_min {
        return false;
    }
    if !graph_pending {
        return true;
    }
    cur_ctx <= request_cap && current_graph_fits
}

/// C `serial_session_fit_plan`.
pub(crate) fn serial_session_fit_plan(
    cur_ctx: i64,
    graph_pending: bool,
    need_min: i64,
    request_cap: i64,
    current_graph_fits: bool,
    frame_kind: SerialFrameKind,
) -> SerialFitPlan {
    if frame_kind == SerialFrameKind::RequiredMiss {
        return SerialFitPlan::PassNative;
    }
    if serial_session_reuse_ok(
        cur_ctx,
        graph_pending,
        need_min,
        request_cap,
        current_graph_fits,
    ) {
        return SerialFitPlan::Reuse;
    }
    if frame_kind == SerialFrameKind::ResolvedLive {
        return SerialFitPlan::RefusePreserve;
    }
    SerialFitPlan::Resize
}

/// C `serial_session_ensure_fit` request bounds. `live_need_min` is uncapped
/// (the live-preserve check compares it against the retained session ctx);
/// the rightsize bounds are capped at boot `-c`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SerialFitBounds {
    pub live_need_min: i64,
    pub need_min: i64,
    pub request_cap: i64,
}

/// C arithmetic: budget floors at 1; the minimal output window is
/// `min(budget, 1024)`; the search cap grants 32768 continuation headroom so
/// a serial conversation grows for many turns inside one session.
pub(crate) fn serial_fit_bounds(prompt_len: i64, budget: i64, boot_ctx: i64) -> SerialFitBounds {
    let budget = budget.max(1);
    let live_need_min = prompt_len + budget.min(1024);
    let need_full = (prompt_len + budget).min(boot_ctx);
    let need_min = live_need_min.min(boot_ctx);
    let request_cap = (need_full + 32768).min(boot_ctx);
    SerialFitBounds {
        live_need_min,
        need_min,
        request_cap,
    }
}

/// C `serial_live_capacity_refuse` client message.
pub(crate) fn serial_live_preserve_msg(effective_prompt_len: i64, cur_ctx: i64) -> String {
    format!(
        "Server cannot extend the retained {effective_prompt_len}-token continuation inside \
         its {cur_ctx}-token serial context; retry with a smaller output budget \
         or replay the full history"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // C test_serial_pending_session_respects_request_cap vectors.
    #[test]
    fn reuse_ok_matches_c_vectors() {
        assert!(!serial_session_reuse_ok(262144, true, 1050, 33818, true));
        assert!(serial_session_reuse_ok(16384, true, 1050, 16384, true));
        assert!(serial_session_reuse_ok(32768, true, 1050, 33818, true));
        assert!(!serial_session_reuse_ok(32768, true, 1050, 33818, false));
        assert!(serial_session_reuse_ok(262144, false, 1050, 33818, false));
        assert!(!serial_session_reuse_ok(1024, false, 1050, 33818, true));
    }

    #[test]
    fn fit_plan_matches_c_vectors() {
        assert_eq!(
            serial_session_fit_plan(
                32066,
                false,
                251024,
                262144,
                false,
                SerialFrameKind::ResolvedLive
            ),
            SerialFitPlan::RefusePreserve
        );
        assert_eq!(
            serial_session_fit_plan(
                262144,
                false,
                251024,
                262144,
                false,
                SerialFrameKind::ResolvedLive
            ),
            SerialFitPlan::Reuse
        );
        assert_eq!(
            serial_session_fit_plan(
                32066,
                false,
                64,
                32800,
                false,
                SerialFrameKind::RequiredMiss
            ),
            SerialFitPlan::PassNative
        );
        assert_eq!(
            serial_session_fit_plan(32066, false, 1024, 33792, false, SerialFrameKind::Canonical),
            SerialFitPlan::Reuse
        );
    }

    #[test]
    fn undersized_canonical_session_resizes() {
        assert_eq!(
            serial_session_fit_plan(2048, true, 5000, 40000, true, SerialFrameKind::Canonical),
            SerialFitPlan::Resize
        );
        // Pending full--c session whose graph cannot alloc beside the banks.
        assert_eq!(
            serial_session_fit_plan(250000, true, 1050, 33818, false, SerialFrameKind::Canonical),
            SerialFitPlan::Resize
        );
    }

    #[test]
    fn bounds_follow_c_arithmetic() {
        // Default output budget degenerates to -c for budget-omitting clients.
        let b = serial_fit_bounds(26, 393216, 250000);
        assert_eq!(b.live_need_min, 26 + 1024);
        assert_eq!(b.need_min, 1050);
        assert_eq!(b.request_cap, 250000);

        let b = serial_fit_bounds(26, 384, 250000);
        assert_eq!(b.live_need_min, 26 + 384);
        assert_eq!(b.need_min, 410);
        assert_eq!(b.request_cap, 26 + 384 + 32768);

        // Zero/explicit-zero budgets floor at 1 like C;
        // request_cap = min(need_full + 32768, boot).
        let b = serial_fit_bounds(100, 0, 4096);
        assert_eq!(b.live_need_min, 101);
        assert_eq!(b.need_min, 101);
        assert_eq!(b.request_cap, 4096);

        // Prompt past boot ctx clamps everywhere.
        let b = serial_fit_bounds(300000, 1024, 250000);
        assert_eq!(b.live_need_min, 301024);
        assert_eq!(b.need_min, 250000);
        assert_eq!(b.request_cap, 250000);
    }

    #[test]
    fn live_preserve_msg_matches_c_shape() {
        let msg = serial_live_preserve_msg(251000, 32066);
        assert!(msg.starts_with("Server cannot extend the retained 251000-token continuation"));
        assert!(msg.contains("its 32066-token serial context"));
        assert!(msg.ends_with("or replay the full history"));
    }
}
