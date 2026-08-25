use super::progress::{
    format_ctx_size, progress_bar, progress_bytes, STATUS_BAR_FILL, STATUS_STYLE_START,
};
use super::{GeneratedSink, Status, Surface, WorkerPhase};

#[test]
fn progress_bar_bytes_match_c_when_half_filled_without_color() {
    // Given: 16/32 tokens, no color (C `stdout_is_tty() == 0`)
    // When: render the 32-cell bar
    // Then: `[` + 16 ▶ + 16 · + `]`
    let mut expected = Vec::from(*b"[");
    expected.extend(
        std::iter::repeat("▶".as_bytes())
            .take(16)
            .flatten()
            .copied(),
    );
    expected.extend(
        std::iter::repeat("·".as_bytes())
            .take(16)
            .flatten()
            .copied(),
    );
    expected.push(b']');
    assert_eq!(progress_bar(16, 32, false), expected);
}

#[test]
fn progress_bar_bytes_match_c_when_empty_without_color() {
    let mut expected = Vec::from(*b"[");
    expected.extend(
        std::iter::repeat("·".as_bytes())
            .take(32)
            .flatten()
            .copied(),
    );
    expected.push(b']');
    assert_eq!(progress_bar(0, 32, false), expected);
}

#[test]
fn progress_bar_bytes_match_c_when_full_without_color() {
    let mut expected = Vec::from(*b"[");
    expected.extend(
        std::iter::repeat("▶".as_bytes())
            .take(32)
            .flatten()
            .copied(),
    );
    expected.push(b']');
    assert_eq!(progress_bar(32, 32, false), expected);
}

#[test]
fn progress_bar_bytes_match_c_when_color_bumps_empty_fill() {
    // Given: color TTY, done=0, total=32
    // When: C forces filled=1 so the magenta cell is visible
    // Then: FILL + one ▶ + STYLE + 31 · + STYLE + ]
    let mut expected = Vec::from(*b"[");
    expected.extend_from_slice(STATUS_BAR_FILL);
    expected.extend_from_slice("▶".as_bytes());
    expected.extend_from_slice(STATUS_STYLE_START);
    expected.extend(
        std::iter::repeat("·".as_bytes())
            .take(31)
            .flatten()
            .copied(),
    );
    expected.extend_from_slice(STATUS_STYLE_START);
    expected.push(b']');
    assert_eq!(progress_bar(0, 32, true), expected);
}

#[test]
fn progress_bar_bytes_match_c_when_half_filled_with_color() {
    let mut expected = Vec::from(*b"[");
    expected.extend_from_slice(STATUS_BAR_FILL);
    expected.extend(
        std::iter::repeat("▶".as_bytes())
            .take(16)
            .flatten()
            .copied(),
    );
    expected.extend_from_slice(STATUS_STYLE_START);
    expected.extend(
        std::iter::repeat("·".as_bytes())
            .take(16)
            .flatten()
            .copied(),
    );
    expected.extend_from_slice(STATUS_STYLE_START);
    expected.push(b']');
    assert_eq!(progress_bar(16, 32, true), expected);
}

#[test]
fn format_ctx_size_matches_c() {
    assert_eq!(format_ctx_size(999), "999");
    assert_eq!(format_ctx_size(1000), "1k");
    assert_eq!(format_ctx_size(1500), "1.5k");
    assert_eq!(format_ctx_size(100_000), "100k");
}

#[test]
fn prefill_status_line_matches_c_shape_when_tty() {
    // Given: 16/32 prefill, 50k/100k ctx, label 0, full power
    // When: render the TTY status footer
    // Then: C `build_status_text` PREFILL shape
    let status = Status {
        phase: WorkerPhase::Prefill,
        ctx_used: 50_000,
        ctx_size: 100_000,
        prefill_done: 16,
        prefill_total: 32,
        prefill_label: 0,
        generated: 0,
        gen_tps: 0.0,
        power_percent: 100,
        error: "",
    };
    let bar = progress_bar(16, 32, true);
    let mut expected = format!("ctx 50k/100k | reading ").into_bytes();
    expected.extend_from_slice(&bar);
    expected.extend_from_slice(b" 16/32 50.0%");
    assert_eq!(progress_bytes(&status, Surface::Tui), expected);
}

#[test]
fn generation_status_line_matches_c_shape_when_tty() {
    let status = Status {
        phase: WorkerPhase::Generating,
        ctx_used: 50_000,
        ctx_size: 100_000,
        prefill_done: 0,
        prefill_total: 0,
        prefill_label: 0,
        generated: 10,
        gen_tps: 5.0,
        power_percent: 70,
        error: "",
    };
    assert_eq!(
        progress_bytes(&status, Surface::Tui),
        "ctx 50k/100k | generation 10 tokens 5.0 t/s | ⚡ 70%".as_bytes()
    );
}

#[test]
fn progress_bytes_are_empty_when_not_tty() {
    // Given: stdout is not a TTY (C skips TUI chrome)
    // When: render a prefill status
    // Then: no progress/status bytes
    let status = Status {
        phase: WorkerPhase::Prefill,
        ctx_used: 1,
        ctx_size: 1000,
        prefill_done: 1,
        prefill_total: 2,
        prefill_label: 0,
        generated: 0,
        gen_tps: 0.0,
        power_percent: 100,
        error: "",
    };
    assert!(progress_bytes(&status, Surface::Plain).is_empty());
}

#[test]
fn generated_sink_skips_live_write_when_not_tty() {
    // Given: non-TTY surface
    // When: push generated text
    // Then: no live TUI write (caller batches like C non-interactive)
    let mut buf = Vec::new();
    let mut sink = GeneratedSink::new(&mut buf, Surface::Plain);
    sink.push(b"hello").expect("plain push");
    assert!(buf.is_empty());
}

#[test]
fn generated_sink_writes_live_when_tty() {
    let mut buf = Vec::new();
    let mut sink = GeneratedSink::new(&mut buf, Surface::Tui);
    sink.push(b"hello").expect("tty push");
    assert_eq!(buf, b"hello");
}

#[test]
fn surface_from_tty_flag_skips_tui_when_false() {
    assert!(!Surface::from_tty(false).is_tui());
    assert!(Surface::from_tty(true).is_tui());
}

fn base_status(phase: WorkerPhase) -> Status<'static> {
    Status {
        phase,
        ctx_used: 50_000,
        ctx_size: 100_000,
        prefill_done: 0,
        prefill_total: 0,
        prefill_label: 0,
        generated: 3,
        gen_tps: 1.5,
        power_percent: 100,
        error: "",
    }
}

#[test]
fn remaining_status_lines_match_c_shape_when_tty() {
    assert_eq!(
        progress_bytes(&base_status(WorkerPhase::Idle), Surface::Tui),
        b"ctx 50k/100k | idle"
    );
    assert_eq!(
        progress_bytes(&base_status(WorkerPhase::Compacting), Surface::Tui),
        "ctx 50k/100k | COMPACTING summary 3 tokens 1.5 t/s".as_bytes()
    );
    assert_eq!(
        progress_bytes(&base_status(WorkerPhase::Draining), Surface::Tui),
        b"ctx 50k/100k | stopping after distributed cluster drains"
    );
    assert_eq!(
        progress_bytes(&base_status(WorkerPhase::Saving), Surface::Tui),
        b"ctx 50k/100k | saving session"
    );
    assert_eq!(
        progress_bytes(&base_status(WorkerPhase::Stopped), Surface::Tui),
        b"ctx 50k/100k | interrupted"
    );
    let mut err = base_status(WorkerPhase::Error);
    err.error = "";
    assert_eq!(
        progress_bytes(&err, Surface::Tui),
        b"ctx 50k/100k | error: unknown error"
    );
}
