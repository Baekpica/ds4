//! Cached worker-to-worker Forwarder, matching C `dist_worker_get_forwarder`.

use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use crate::codec::{u64_from_halves, Telemetry};
use crate::forward::{forward_window, PendingRequest, ERR_FORWARD};
use crate::relay::Forwarder;
use crate::route::RouteEntry;
use crate::work::{encode_work_frame, error_result_frame, WorkBody};

pub struct ForwarderPool {
    window: u32,
    items: HashMap<(String, u32), Forwarder>,
    upstream: Option<Arc<Mutex<TcpStream>>>,
}

impl ForwarderPool {
    pub fn new() -> Self {
        Self {
            window: forward_window(),
            items: HashMap::new(),
            upstream: None,
        }
    }

    pub fn bind(&mut self, upstream: Arc<Mutex<TcpStream>>) {
        self.upstream = Some(upstream);
    }

    pub fn is_bound(&self) -> bool {
        self.upstream.is_some()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn forward(
        &mut self,
        next: &RouteEntry,
        body: &WorkBody,
        local: Telemetry,
    ) -> Result<(), String> {
        let key = (next.host.clone(), next.port);
        if !self.items.contains_key(&key) {
            let fw = self.connect_new(&next.host, next.port)?;
            self.items.insert(key.clone(), fw);
        }
        let request_id = u64_from_halves(body.work.request_hi, body.work.request_lo);
        let frame = encode_work_frame(body).map_err(|e| e.to_string())?;
        let pending = PendingRequest {
            request_id,
            telemetry: local,
            downstream_t0: 0.0,
        };
        self.items
            .get(&key)
            .ok_or(ERR_FORWARD)?
            .send_work(pending, &frame)
            .map_err(str::to_string)
    }

    pub fn shutdown(&mut self) {
        self.items.clear();
        self.upstream = None;
    }

    fn connect_new(&self, host: &str, port: u32) -> Result<Forwarder, String> {
        let up = self.upstream.clone().ok_or(ERR_FORWARD)?;
        let up_err = Arc::clone(&up);
        Forwarder::connect(
            host,
            port,
            self.window,
            move |frame| {
                let _ = up.lock().expect("upstream write").write_all(&frame);
            },
            move |id, msg| {
                let frame = error_result_frame(id, msg);
                let _ = up_err.lock().expect("upstream write").write_all(&frame);
            },
        )
    }
}

impl Default for ForwarderPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ForwarderPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}
