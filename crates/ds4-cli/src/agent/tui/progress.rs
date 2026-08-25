//! C `agent_progress_bar` / `build_status_text` byte shapes.

use super::{Status, Surface, WorkerPhase};

const BAR_WIDTH: i32 = 32;
pub const STATUS_STYLE_START: &[u8] = b"\x1b[48;5;238;38;5;252m";
pub const STATUS_BAR_FILL: &[u8] = b"\x1b[48;5;238;38;5;201;1m";
const PREFILL_LABELS: [&str; 6] = [
    "reading",
    "absorbing",
    "studying",
    "gathering",
    "crunching",
    "scrutinizing",
];

pub fn format_ctx_size(ctx_size: i32) -> String {
    if ctx_size >= 1000 {
        if ctx_size % 1000 == 0 {
            format!("{}k", ctx_size / 1000)
        } else {
            format!("{:.1}k", f64::from(ctx_size) / 1000.0)
        }
    } else {
        ctx_size.to_string()
    }
}

pub fn progress_bar(done: i32, total: i32, color: bool) -> Vec<u8> {
    let total = if total <= 0 { 1 } else { total };
    let done = done.clamp(0, total);
    let mut filled = i32::try_from(i64::from(done) * i64::from(BAR_WIDTH) / i64::from(total))
        .unwrap_or(BAR_WIDTH);
    filled = filled.clamp(0, BAR_WIDTH);
    if color && filled == 0 && done < total {
        filled = 1;
    }
    let mut out = Vec::from(*b"[");
    if color {
        out.extend_from_slice(STATUS_BAR_FILL);
    }
    for i in 0..BAR_WIDTH {
        if color && i == filled {
            out.extend_from_slice(STATUS_STYLE_START);
        }
        out.extend_from_slice(if i < filled {
            "▶".as_bytes()
        } else {
            "·".as_bytes()
        });
    }
    if color {
        out.extend_from_slice(STATUS_STYLE_START);
    }
    out.push(b']');
    out
}

fn power_suffix(power_percent: i32) -> String {
    if power_percent > 0 && power_percent < 100 {
        format!(" | ⚡ {power_percent}%")
    } else {
        String::new()
    }
}

fn prefill_label(index: u32) -> &'static str {
    PREFILL_LABELS[usize::try_from(index).unwrap_or(0) % PREFILL_LABELS.len()]
}

fn status_text(status: &Status<'_>, color: bool) -> Vec<u8> {
    let used = format_ctx_size(status.ctx_used);
    let total_ctx = format_ctx_size(status.ctx_size);
    let power = power_suffix(status.power_percent);
    match status.phase {
        WorkerPhase::Prefill => {
            let total = if status.prefill_total > 0 {
                status.prefill_total
            } else {
                1
            };
            let done = status.prefill_done.min(total);
            let pct = 100.0 * f64::from(done) / f64::from(total);
            let bar = progress_bar(done, total, color);
            let mut out = format!(
                "ctx {used}/{total_ctx} | {} ",
                prefill_label(status.prefill_label)
            )
            .into_bytes();
            out.extend_from_slice(&bar);
            out.extend_from_slice(format!(" {done}/{total} {pct:.1}%{power}").as_bytes());
            out
        }
        WorkerPhase::Generating => format!(
            "ctx {used}/{total_ctx} | generation {} tokens {:.1} t/s{power}",
            status.generated, status.gen_tps
        )
        .into_bytes(),
        WorkerPhase::Compacting => format!(
            "ctx {used}/{total_ctx} | COMPACTING summary {} tokens {:.1} t/s{power}",
            status.generated, status.gen_tps
        )
        .into_bytes(),
        WorkerPhase::Draining => {
            format!("ctx {used}/{total_ctx} | stopping after distributed cluster drains{power}")
                .into_bytes()
        }
        WorkerPhase::Saving => {
            format!("ctx {used}/{total_ctx} | saving session{power}").into_bytes()
        }
        WorkerPhase::Error => {
            let err = if status.error.is_empty() {
                "unknown error"
            } else {
                status.error
            };
            format!("ctx {used}/{total_ctx} | error: {err}{power}").into_bytes()
        }
        WorkerPhase::Stopped => format!("ctx {used}/{total_ctx} | interrupted{power}").into_bytes(),
        WorkerPhase::Idle => format!("ctx {used}/{total_ctx} | idle{power}").into_bytes(),
    }
}

pub fn progress_bytes(status: &Status<'_>, surface: Surface) -> Vec<u8> {
    if !surface.is_tui() {
        return Vec::new();
    }
    status_text(status, true)
}
