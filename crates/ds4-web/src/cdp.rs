//! Chrome DevTools Protocol over the C WebSocket client.

use std::thread;
use std::time::Duration;

use crate::encode::{json_get_string, json_id_matches};
use crate::js;
use crate::net::Ws;
use crate::wire::{cdp_request, create_target_params, eval_params, navigate_params};

pub fn cdp_call(ws: &mut Ws, method: &str, params: Option<&str>) -> Result<String, String> {
    let id = ws.next_id;
    ws.next_id += 1;
    let wire = cdp_request(id, method, params);
    ws.send_text(&wire)?;
    loop {
        let msg = ws.read_message()?;
        if json_id_matches(&msg, id) {
            return Ok(msg);
        }
    }
}

pub fn cdp_call_optional(ws: &mut Ws, method: &str, params: &str) {
    let _ = cdp_call(ws, method, Some(params));
}

pub fn cdp_eval_string(ws: &mut Ws, expr: &str) -> Result<String, String> {
    let params = eval_params(expr);
    let resp = cdp_call(ws, "Runtime.evaluate", Some(&params))?;
    if resp.contains("\"exceptionDetails\"") {
        return Err("JavaScript evaluation failed".into());
    }
    json_get_string(&resp, "value").ok_or_else(|| "Runtime.evaluate did not return a string".into())
}

pub fn wait_ready(ws: &mut Ws) -> Result<(), String> {
    for _ in 0..80 {
        match cdp_eval_string(ws, js::READY_STATE) {
            Ok(state) if state == "complete" || state == "interactive" => {
                thread::sleep(Duration::from_millis(800));
                return Ok(());
            }
            _ => thread::sleep(Duration::from_millis(250)),
        }
    }
    Ok(())
}

pub fn cdp_navigate(ws: &mut Ws, url: &str) -> Result<(), String> {
    let params = navigate_params(url);
    cdp_call(ws, "Page.navigate", Some(&params))?;
    Ok(())
}

fn page_probe(ws: &mut Ws) -> Result<(String, String, i64), String> {
    let probe = cdp_eval_string(ws, js::PAGE_PROBE)?;
    let mut lines = probe.splitn(3, '\n');
    let href = lines
        .next()
        .ok_or_else(|| "page readiness probe returned malformed data".to_string())?;
    let ready = lines
        .next()
        .ok_or_else(|| "page readiness probe returned malformed data".to_string())?;
    let rest = lines
        .next()
        .ok_or_else(|| "page readiness probe returned malformed data".to_string())?;
    let text_len = rest.parse::<i64>().unwrap_or_else(|_| {
        rest.split(|c: char| !c.is_ascii_digit() && c != '-')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    });
    Ok((href.to_string(), ready.to_string(), text_len))
}

pub fn wait_navigated_ready(ws: &mut Ws) -> Result<(), String> {
    let mut last_len: i64 = -1;
    let mut stable = 0;
    let mut saw_real_url = false;
    for i in 0..100 {
        let (href, ready, text_len) = match page_probe(ws) {
            Ok(v) => v,
            Err(_) => {
                thread::sleep(Duration::from_millis(250));
                continue;
            }
        };
        let real_url = !href.is_empty() && href != "about:blank" && !href.starts_with("chrome://");
        let ready_state = ready == "complete" || ready == "interactive";
        if real_url {
            saw_real_url = true;
        }
        if text_len > 0 && text_len == last_len {
            stable += 1;
        } else {
            stable = 0;
        }
        last_len = text_len;
        if saw_real_url && ready_state && text_len > 0 && stable >= 2 {
            thread::sleep(Duration::from_millis(500));
            return Ok(());
        }
        if saw_real_url && ready_state && i >= 24 {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
    Ok(())
}

pub fn cdp_prepare_page(ws: &mut Ws) -> Result<(), String> {
    cdp_call(ws, "Page.enable", Some("{}"))?;
    cdp_call(ws, "Runtime.enable", Some("{}"))?;
    cdp_call_optional(
        ws,
        "Emulation.setFocusEmulationEnabled",
        "{\"enabled\":true}",
    );
    cdp_call_optional(
        ws,
        "Emulation.setDeviceMetricsOverride",
        "{\"width\":1365,\"height\":900,\"deviceScaleFactor\":1,\"mobile\":false}",
    );
    wait_ready(ws)
}

pub fn scroll_dynamic_page(ws: &mut Ws) {
    let _ = cdp_eval_string(ws, js::SCROLL_DYNAMIC);
}

pub fn create_target(ws: &mut Ws, url: &str) -> Result<String, String> {
    let params = create_target_params(url);
    let resp = cdp_call(ws, "Target.createTarget", Some(&params))?;
    json_get_string(&resp, "targetId")
        .ok_or_else(|| "Chrome did not return a page target id".into())
}

pub fn browser_ws_url(port: i32) -> Result<String, String> {
    let body = crate::net::http_local("GET", port, "/json/version")?;
    json_get_string(&body, "webSocketDebuggerUrl")
        .ok_or_else(|| "Chrome did not return a browser WebSocket URL".into())
}
