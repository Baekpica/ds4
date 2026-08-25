//! Agent KV resume: `/save` `/list` `/switch` `/del` `/strip`.
//! Files live under `~/.ds4/kvcache` with C `KVC` magic.

mod catalog;
mod handle;
mod identity;
mod store;

#[cfg(test)]
mod tests;

pub(crate) use catalog::ListedSession;
pub(crate) use handle::{handle_slash, unix_now, Live};
pub(crate) use identity::{identity_sha, SessionIdentity};
pub(crate) use store::{SaveSpec, SessionStore};

use std::ffi::OsString;
use std::path::PathBuf;

pub(crate) fn default_cache_dir() -> PathBuf {
    cache_dir_from_home(std::env::var_os("HOME"))
}

pub(crate) fn cache_dir_from_home(home: Option<OsString>) -> PathBuf {
    let home = home
        .filter(|home| !home.is_empty())
        .unwrap_or_else(|| OsString::from("."));
    PathBuf::from(home).join(".ds4/kvcache")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlashCmd<'a> {
    Save,
    List,
    Switch(&'a str),
    Del(&'a str),
    Strip(&'a str),
}

pub(crate) fn parse_slash(line: &str) -> Option<Result<SlashCmd<'_>, &'static str>> {
    let cmd = line.trim();
    if cmd == "/save" {
        return Some(Ok(SlashCmd::Save));
    }
    if cmd == "/list" {
        return Some(Ok(SlashCmd::List));
    }
    if let Some(rest) = slash_arg(cmd, "/switch") {
        return Some(if rest.is_empty() {
            Err("usage: /switch <sha-prefix>")
        } else {
            Ok(SlashCmd::Switch(rest))
        });
    }
    if let Some(rest) = slash_arg(cmd, "/del") {
        return Some(if rest.is_empty() {
            Err("usage: /del <sha-prefix>")
        } else {
            Ok(SlashCmd::Del(rest))
        });
    }
    if let Some(rest) = slash_arg(cmd, "/strip") {
        return Some(if rest.is_empty() {
            Err("usage: /strip <sha-prefix>")
        } else {
            Ok(SlashCmd::Strip(rest))
        });
    }
    None
}

fn slash_arg<'a>(cmd: &'a str, name: &str) -> Option<&'a str> {
    if !cmd.starts_with(name) {
        return None;
    }
    let rest = &cmd[name.len()..];
    if rest.is_empty() {
        return Some("");
    }
    let first = rest.as_bytes()[0];
    if first != b' ' && first != b'\t' {
        return None;
    }
    Some(rest.trim())
}
