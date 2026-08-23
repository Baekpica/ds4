//! Chrome discovery and spawn. Matches `ds4_web.c`; does not kill the
//! process when the `Web` handle is dropped.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use crate::net::http_local;

pub const DEFAULT_PORT: i32 = 9333;
pub const CONFIRM_PROMPT: &str =
    "The web tool wants to start a visible Chrome browser. Allow? (y/n) ";

const LINUX_PATHS: &[&str] = &[
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
    "/opt/google/chrome/chrome",
];

const PATH_NAMES: &[&str] = &[
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
];

pub fn profile_dir(home: &Path) -> PathBuf {
    home.join(".ds4").join("browser")
}

pub fn effective_uid_is_root() -> bool {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Uid:") {
                let mut fields = rest.split_whitespace();
                let _real = fields.next();
                if let Some(effective) = fields.next() {
                    return effective == "0";
                }
            }
        }
    }
    false
}

pub fn chrome_executable() -> PathBuf {
    if let Ok(env) = std::env::var("DS4_CHROME") {
        if !env.is_empty() {
            return PathBuf::from(env);
        }
    }
    #[cfg(target_os = "macos")]
    {
        let mac = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ];
        for p in mac {
            if is_executable(p) {
                return PathBuf::from(p);
            }
        }
    }
    for p in LINUX_PATHS {
        if is_executable(p) {
            return PathBuf::from(p);
        }
    }
    if let Ok(pathenv) = std::env::var("PATH") {
        for dir in pathenv.split(':') {
            let dir = if dir.is_empty() { "." } else { dir };
            for name in PATH_NAMES {
                let candidate = Path::new(dir).join(name);
                if is_executable(&candidate) {
                    return candidate;
                }
            }
        }
    }
    PathBuf::from("google-chrome")
}

fn is_executable(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path)
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub fn linux_chrome_args(port: i32, profile: &Path, root: bool) -> Vec<String> {
    let mut args = vec![
        format!("--remote-debugging-port={port}"),
        "--remote-allow-origins=*".into(),
        format!("--user-data-dir={}", profile.display()),
        "--no-first-run".into(),
        "--no-default-browser-check".into(),
        "--disable-sync".into(),
        "--password-store=basic".into(),
    ];
    if root {
        args.push("--no-sandbox".into());
    }
    args.push("--mute-audio".into());
    args.push("about:blank".into());
    args
}

pub fn mkdir_p(path: &Path) -> Result<(), String> {
    let mut cur = PathBuf::new();
    for part in path.components() {
        cur.push(part);
        if cur.as_os_str().is_empty() || cur == Path::new("/") {
            continue;
        }
        match fs::create_dir(&cur) {
            Ok(()) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = fs::set_permissions(&cur, fs::Permissions::from_mode(0o700));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                return Err(format!(
                    "failed to create Chrome profile dir {}: {e}",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

pub fn cdp_alive(port: i32) -> bool {
    match http_local("GET", port, "/json/version") {
        Ok(body) => body.contains("webSocketDebuggerUrl"),
        Err(_) => false,
    }
}

pub fn spawn_chrome(port: i32, profile: &Path) -> Result<Child, String> {
    mkdir_p(profile)?;
    let exe = chrome_executable();
    let args = linux_chrome_args(port, profile, effective_uid_is_root());
    #[cfg(target_os = "macos")]
    {
        let _ = &args;
        // macOS `open -na` path is compiled but DFM gate is Linux.
    }
    Command::new(&exe)
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to fork Chrome: {e}"))
}

pub fn wait_cdp_ready(port: i32, child: &mut Child) -> Result<(), String> {
    for _ in 0..80 {
        if cdp_alive(port) {
            return Ok(());
        }
        match child.try_wait() {
            Ok(Some(_)) => {
                return Err("Chrome exited before CDP became ready".into());
            }
            _ => std::thread::sleep(std::time::Duration::from_millis(250)),
        }
    }
    Err(format!("Chrome did not expose CDP on port {port}"))
}

/// Forget a live child so `Drop` does not wait for Chrome (C never reaps
/// on `ds4_web_free`).
pub fn detach(child: Child) {
    std::mem::forget(child);
}
