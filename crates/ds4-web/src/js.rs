//! Page extractors copied from `ds4_web.c`. The files under `js/` are the
//! exact C string contents (not the source-escaped form).

pub const CLICK_GOOGLE_CONSENT: &str = include_str!("js/consent.js");
pub const EXTRACT_SEARCH: &str = include_str!("js/search.js");
pub const EXTRACT_PAGE: &str = include_str!("js/page.js");
pub const SCROLL_DYNAMIC: &str = include_str!("js/scroll.js");
pub const PAGE_PROBE: &str = include_str!("js/probe.js");
pub const READY_STATE: &str = "document.readyState";
