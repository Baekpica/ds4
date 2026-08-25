//! C `token_printer_*` from `ds4_cli.c`: hide `<think>` / `</think>`, grey the body on TTY.

use std::io::{self, IsTerminal, Write};

const THINK_OPEN: &[u8] = b"<think>";
const THINK_CLOSE: &[u8] = b"</think>";

/// C `token_printer_set_grey`: SGR 90 bright-black.
const GREY: &[u8] = b"\x1b[90m";
/// C `token_printer_reset_color`.
const RESET: &[u8] = b"\x1b[0m";

pub(crate) struct TokenPrinter {
    format_thinking: bool,
    in_think: bool,
    color_open: bool,
    use_color: bool,
    pending: Vec<u8>,
    last_output_newline: bool,
}

impl TokenPrinter {
    pub(crate) fn new(format_thinking: bool) -> Self {
        Self::with_color(format_thinking, io::stdout().is_terminal())
    }

    pub(crate) fn with_color(format_thinking: bool, use_color: bool) -> Self {
        Self {
            format_thinking,
            in_think: format_thinking,
            color_open: false,
            use_color,
            pending: Vec::new(),
            last_output_newline: true,
        }
    }

    pub(crate) fn write_text<W: Write>(&mut self, out: &mut W, text: &[u8]) -> io::Result<()> {
        if !self.format_thinking {
            out.write_all(text)?;
            if let Some(last) = text.last() {
                self.last_output_newline = *last == b'\n';
            }
            return Ok(());
        }
        self.process(out, text, false)
    }

    pub(crate) fn finish<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.format_thinking {
            self.process(out, &[], true)?;
            self.reset_color(out)?;
        }
        if !self.last_output_newline {
            out.write_all(b"\n")?;
            self.last_output_newline = true;
        }
        out.flush()
    }

    fn process<W: Write>(&mut self, out: &mut W, text: &[u8], finish: bool) -> io::Result<()> {
        let mut bytes = std::mem::take(&mut self.pending);
        bytes.extend_from_slice(text);
        let mut i = 0;
        while i < bytes.len() {
            let rem = &bytes[i..];
            if rem.starts_with(THINK_OPEN) {
                self.in_think = true;
                i += THINK_OPEN.len();
                continue;
            }
            if rem.starts_with(THINK_CLOSE) {
                self.in_think = false;
                self.reset_color(out)?;
                if !self.last_output_newline {
                    out.write_all(b"\n")?;
                    self.last_output_newline = true;
                }
                i += THINK_CLOSE.len();
                continue;
            }
            if !finish
                && rem[0] == b'<'
                && ((rem.len() < THINK_OPEN.len() && THINK_OPEN.starts_with(rem))
                    || (rem.len() < THINK_CLOSE.len() && THINK_CLOSE.starts_with(rem)))
            {
                self.pending.extend_from_slice(rem);
                break;
            }
            self.write_char(out, rem[0])?;
            i += 1;
        }
        Ok(())
    }

    fn write_char<W: Write>(&mut self, out: &mut W, c: u8) -> io::Result<()> {
        if self.in_think {
            self.set_grey(out)?;
        }
        out.write_all(&[c])?;
        self.last_output_newline = c == b'\n';
        Ok(())
    }

    fn set_grey<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.use_color && !self.color_open {
            out.write_all(GREY)?;
            self.color_open = true;
        }
        Ok(())
    }

    fn reset_color<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.use_color && self.color_open {
            out.write_all(RESET)?;
            self.color_open = false;
        }
        Ok(())
    }
}
