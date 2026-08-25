//! C `ds4_agent.c` TUI: prefill progress, status footer, generated text.
//! Non-TTY skips TUI chrome the way C skips color / linenoise hide-show.

mod progress;
mod sink;

#[cfg(test)]
mod tests;

use std::io::{self, IsTerminal};

pub(super) use sink::{clear_progress_frame, GeneratedSink};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Surface {
    Tui,
    Plain,
}

impl Surface {
    pub(super) fn from_tty(is_tty: bool) -> Self {
        if is_tty {
            Self::Tui
        } else {
            Self::Plain
        }
    }

    pub(super) fn from_stdout() -> Self {
        Self::from_tty(io::stdout().is_terminal())
    }

    pub(super) const fn is_tui(self) -> bool {
        matches!(self, Self::Tui)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WorkerPhase {
    Idle,
    Prefill,
    Generating,
    Compacting,
    Draining,
    Saving,
    Error,
    Stopped,
}

pub(super) struct Status<'a> {
    pub(super) phase: WorkerPhase,
    pub(super) ctx_used: i32,
    pub(super) ctx_size: i32,
    pub(super) prefill_done: i32,
    pub(super) prefill_total: i32,
    pub(super) prefill_label: u32,
    pub(super) generated: i32,
    pub(super) gen_tps: f64,
    pub(super) power_percent: i32,
    pub(super) error: &'a str,
}

pub(super) struct PrefillView {
    pub(super) ctx_size: i32,
    pub(super) done: i32,
    pub(super) total: i32,
    pub(super) label: u32,
    pub(super) power_percent: i32,
}

pub(super) struct GenerationView {
    pub(super) ctx_used: i32,
    pub(super) ctx_size: i32,
    pub(super) generated: i32,
    pub(super) gen_tps: f64,
    pub(super) power_percent: i32,
}

pub(super) fn emit_prefill(surface: Surface, view: PrefillView) {
    let status = Status {
        phase: WorkerPhase::Prefill,
        ctx_used: view.done,
        ctx_size: view.ctx_size,
        prefill_done: view.done,
        prefill_total: view.total,
        prefill_label: view.label,
        generated: 0,
        gen_tps: 0.0,
        power_percent: view.power_percent,
        error: "",
    };
    let _ = sink::write_progress_frame(
        &mut io::stderr(),
        &progress::progress_bytes(&status, surface),
    );
}

pub(super) fn emit_generation(surface: Surface, view: GenerationView) {
    let status = Status {
        phase: WorkerPhase::Generating,
        ctx_used: view.ctx_used,
        ctx_size: view.ctx_size,
        prefill_done: 0,
        prefill_total: 0,
        prefill_label: 0,
        generated: view.generated,
        gen_tps: view.gen_tps,
        power_percent: view.power_percent,
        error: "",
    };
    let _ = sink::write_progress_frame(
        &mut io::stderr(),
        &progress::progress_bytes(&status, surface),
    );
}
