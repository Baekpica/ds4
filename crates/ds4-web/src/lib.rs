//! Agent web utility. Blocking I/O port of `ds4_web.c`. No Tokio.

use std::path::PathBuf;
use std::process::Child;
use std::thread;
use std::time::Duration;

mod cdp;
mod chrome;
mod encode;
mod js;
mod net;
mod wire;

pub use chrome::{chrome_executable, linux_chrome_args, profile_dir, CONFIRM_PROMPT, DEFAULT_PORT};
pub use encode::{
    base64, json_get_string, json_get_string_bytes, json_id_matches, json_parse_string_at,
    json_quote, json_quote_bytes, url_encode, url_encode_bytes,
};
pub use js::{
    CLICK_GOOGLE_CONSENT, EXTRACT_PAGE, EXTRACT_SEARCH, PAGE_PROBE, READY_STATE, SCROLL_DYNAMIC,
};
pub use net::http_local;
pub use wire::{
    cdp_request, close_path, create_target_params, eval_params, google_search_url, http_request,
    navigate_params, page_ws_url, ws_handshake, ws_pong_frame, ws_text_frame,
};

pub const CONNECT_TIMEOUT_MS: u64 = 3000;
pub const CDP_TIMEOUT_MS: u64 = 20000;
pub const MAX_RESULT_BYTES: usize = 1024 * 1024;

pub struct Config {
    pub home_dir: Option<PathBuf>,
    pub port: i32,
    pub confirm: Option<Box<dyn Fn(&str) -> Result<bool, String>>>,
    pub log: Option<Box<dyn Fn(&str)>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            home_dir: None,
            port: 0,
            confirm: None,
            log: None,
        }
    }
}

pub struct Web {
    profile_dir: PathBuf,
    port: i32,
    chrome: Option<Child>,
    browser_allowed: bool,
    confirm: Option<Box<dyn Fn(&str) -> Result<bool, String>>>,
    log: Option<Box<dyn Fn(&str)>>,
}

impl Web {
    pub fn new(cfg: Config) -> Self {
        let home = cfg.home_dir.unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| PathBuf::from("."))
        });
        Self {
            profile_dir: chrome::profile_dir(&home),
            port: if cfg.port > 0 { cfg.port } else { DEFAULT_PORT },
            chrome: None,
            browser_allowed: false,
            confirm: cfg.confirm,
            log: cfg.log,
        }
    }

    pub fn port(&self) -> i32 {
        self.port
    }

    pub fn profile_dir(&self) -> &std::path::Path {
        &self.profile_dir
    }

    fn log(&self, msg: &str) {
        if let Some(log) = &self.log {
            log(msg);
        }
    }

    fn ensure_browser(&mut self) -> Result<(), String> {
        if chrome::cdp_alive(self.port) {
            return Ok(());
        }
        if let Some(mut child) = self.chrome.take() {
            let _ = child.try_wait();
            if child.try_wait().ok().flatten().is_none() {
                chrome::detach(child);
            }
        }
        if !self.browser_allowed {
            let Some(confirm) = &self.confirm else {
                return Err(
                    "starting a visible Chrome browser requires interactive approval".into(),
                );
            };
            match confirm(CONFIRM_PROMPT) {
                Ok(true) => self.browser_allowed = true,
                Ok(false) => return Err("user denied Chrome browser start".into()),
                Err(e) => {
                    if e.is_empty() {
                        return Err("user denied Chrome browser start".into());
                    }
                    return Err(e);
                }
            }
        }
        let mut child = chrome::spawn_chrome(self.port, &self.profile_dir)?;
        chrome::wait_cdp_ready(self.port, &mut child)?;
        self.log("Chrome browser session is ready");
        self.chrome = Some(child);
        Ok(())
    }

    fn close_tab(&self, target_id: &str) {
        let path = close_path(target_id);
        match net::http_local("GET", self.port, &path) {
            Ok(_) => {}
            Err(e) => self.log(&e),
        }
    }

    fn open_tab(&self, url: &str) -> Result<(String, String), String> {
        let browser_url = cdp::browser_ws_url(self.port)?;
        let mut browser = net::Ws::connect(&browser_url)?;
        let id = cdp::create_target(&mut browser, url)?;
        let ws_url = page_ws_url(self.port, &id);
        Ok((id, ws_url))
    }

    fn run_page_js(&mut self, url: &str, js: &str, dynamic_scroll: bool) -> Result<String, String> {
        self.ensure_browser()?;
        let (tab_id, tab_ws) = self.open_tab("about:blank")?;
        let mut ws = match net::Ws::connect(&tab_ws) {
            Ok(ws) => ws,
            Err(e) => {
                self.close_tab(&tab_id);
                return Err(e);
            }
        };
        let fail = |web: &mut Web, tab_id: &str, e: String| {
            web.close_tab(tab_id);
            Err(e)
        };
        if let Err(e) = cdp::cdp_prepare_page(&mut ws) {
            return fail(self, &tab_id, e);
        }
        if let Err(e) = cdp::cdp_navigate(&mut ws, url) {
            return fail(self, &tab_id, e);
        }
        if let Err(e) = cdp::wait_navigated_ready(&mut ws) {
            return fail(self, &tab_id, e);
        }
        match cdp::cdp_eval_string(&mut ws, crate::js::CLICK_GOOGLE_CONSENT) {
            Ok(clicked) if !clicked.is_empty() => {
                self.log(&clicked);
                thread::sleep(Duration::from_millis(1500));
                let _ = cdp::wait_navigated_ready(&mut ws);
            }
            _ => {}
        }
        if dynamic_scroll {
            cdp::scroll_dynamic_page(&mut ws);
        }
        let out = match cdp::cdp_eval_string(&mut ws, js) {
            Ok(v) => v,
            Err(e) => return fail(self, &tab_id, e),
        };
        self.close_tab(&tab_id);
        Ok(out)
    }

    pub fn google_search(&mut self, query: &str) -> Result<String, String> {
        if query.is_empty() {
            return Err("google_search requires query".into());
        }
        let url = google_search_url(query);
        self.run_page_js(&url, crate::js::EXTRACT_SEARCH, false)
    }

    pub fn visit_page(&mut self, url: &str) -> Result<String, String> {
        if url.is_empty() {
            return Err("visit_page requires url".into());
        }
        self.run_page_js(url, crate::js::EXTRACT_PAGE, true)
    }
}

impl Drop for Web {
    fn drop(&mut self) {
        // Do not kill Chrome. The browser profile is user-visible state.
        if let Some(child) = self.chrome.take() {
            chrome::detach(child);
        }
    }
}
