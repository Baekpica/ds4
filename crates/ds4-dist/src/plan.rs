//! Worker registry and route search. Matches `dist_coordinator_build_route_plan`.

use crate::codec::{ROUTE_F_OUTPUT_LOGITS, ROUTE_RETURN_UPSTREAM};
use crate::route::{encode_route_blob, ReturnTarget, RouteEntry};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerInfo {
    pub peer_host: String,
    pub listen_port: u32,
    pub model_id: u32,
    pub quant_bits: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub has_output: bool,
    pub has_hidden: bool,
}

#[derive(Debug, Clone)]
pub struct CoordinatorView {
    pub n_layers: u32,
    pub local_start: u32,
    pub local_end: u32,
    pub local_has_output: bool,
    pub local_can_output_head: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePlan {
    pub entries: Vec<RouteEntry>,
    pub blob: Vec<u8>,
}

fn worker_cmp(a: &WorkerInfo, b: &WorkerInfo) -> std::cmp::Ordering {
    a.layer_start
        .cmp(&b.layer_start)
        .then_with(|| b.has_output.cmp(&a.has_output))
        .then_with(|| b.layer_end.cmp(&a.layer_end))
}

fn candidate_ok(view: &CoordinatorView, w: &WorkerInfo, last: u32) -> bool {
    let needs_hidden = w.layer_end < last || !w.has_output;
    if needs_hidden && !w.has_hidden {
        return false;
    }
    if w.layer_end >= last && !w.has_output && !view.local_can_output_head {
        return false;
    }
    true
}

fn search(
    view: &CoordinatorView,
    workers: &[WorkerInfo],
    next: u32,
    last: u32,
    path: &mut Vec<usize>,
    missing: &mut u32,
) -> bool {
    let mut saw_start = false;
    for (i, w) in workers.iter().enumerate() {
        if w.layer_start < next {
            continue;
        }
        if w.layer_start > next {
            break;
        }
        saw_start = true;
        if !candidate_ok(view, w, last) {
            continue;
        }
        path.push(i);
        if w.layer_end >= last {
            return true;
        }
        let mut child_missing = w.layer_end + 1;
        if search(view, workers, child_missing, last, path, &mut child_missing) {
            return true;
        }
        if child_missing > *missing {
            *missing = child_missing;
        }
        path.pop();
    }
    if !saw_start && next > *missing {
        *missing = next;
    }
    false
}

pub fn build_route_plan(
    view: &CoordinatorView,
    workers: &[WorkerInfo],
) -> Result<RoutePlan, String> {
    if view.local_start != 0 {
        return Err("coordinator route does not start at layer 0".into());
    }
    let last = view.n_layers.saturating_sub(1);
    if view.local_end == last && (view.local_has_output || view.local_can_output_head) {
        return Ok(RoutePlan {
            entries: Vec::new(),
            blob: Vec::new(),
        });
    }

    let mut sorted = workers.to_vec();
    sorted.sort_by(worker_cmp);
    let next = view.local_end + 1;
    let mut path = Vec::new();
    let mut missing = next;
    if !search(view, &sorted, next, last, &mut path, &mut missing) {
        return Err(format!(
            "distributed route incomplete: missing layer {missing}"
        ));
    }

    let mut entries = Vec::new();
    for idx in path {
        let w = &sorted[idx];
        entries.push(RouteEntry {
            host: w.peer_host.clone(),
            port: w.listen_port,
            layer_start: w.layer_start,
            layer_end: w.layer_end,
            flags: if w.has_output {
                ROUTE_F_OUTPUT_LOGITS
            } else {
                0
            },
        });
    }
    let blob = if entries.is_empty() {
        Vec::new()
    } else {
        encode_route_blob(
            &entries,
            &ReturnTarget {
                kind: ROUTE_RETURN_UPSTREAM,
                host: String::new(),
                port: 0,
            },
        )
        .map_err(|e| e.to_string())?
    };
    Ok(RoutePlan { entries, blob })
}

/// Replace a stale worker with the same host/model/layers/output (C prepends).
pub fn register_worker(list: &mut Vec<WorkerInfo>, neu: WorkerInfo) {
    list.retain(|old| {
        !(old.peer_host == neu.peer_host
            && old.model_id == neu.model_id
            && old.layer_start == neu.layer_start
            && old.layer_end == neu.layer_end
            && old.has_output == neu.has_output)
    });
    list.insert(0, neu);
}
