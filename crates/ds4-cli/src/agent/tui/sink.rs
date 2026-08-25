//! Live generated-text sink and progress-frame I/O.

use std::io::{self, Write};

use super::Surface;

pub fn write_progress_frame(out: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    out.write_all(b"\r")?;
    out.write_all(bytes)?;
    out.write_all(b"\x1b[K")?;
    out.flush()
}

pub fn clear_progress_frame(out: &mut impl Write, surface: Surface) -> io::Result<()> {
    if !surface.is_tui() {
        return Ok(());
    }
    out.write_all(b"\r\x1b[K")?;
    out.flush()
}

pub struct GeneratedSink<W> {
    out: W,
    surface: Surface,
}

impl<W: Write> GeneratedSink<W> {
    pub fn new(out: W, surface: Surface) -> Self {
        Self { out, surface }
    }

    pub fn push(&mut self, bytes: &[u8]) -> io::Result<()> {
        if !self.surface.is_tui() || bytes.is_empty() {
            return Ok(());
        }
        self.out.write_all(bytes)?;
        self.out.flush()
    }
}
