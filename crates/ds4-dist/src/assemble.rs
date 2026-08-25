//! Assemble a worker HELLO and `Worker` without a CLI crate.

use std::borrow::Cow;

use crate::codec::{Hello, MAX_MODEL_NAME};
use crate::exec::SliceExec;
use crate::native_snapshot::SnapshotStore;
use crate::options::{resolved_layer_end, Layers};
use crate::worker::Worker;

pub const UNKNOWN_MODEL_NAME: &str = "unknown";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceMeta {
    pub model_id: u32,
    pub n_layers: u32,
    pub vocab: u32,
    pub ctx_size: u32,
    pub hidden_values: u64,
    pub has_output: bool,
    pub layer_start: u32,
    pub layer_end: u32,
}

pub fn slice_meta(
    model_id: u32,
    n_layers: u32,
    vocab: u32,
    ctx_size: u32,
    hidden_values: u64,
    layers: &Layers,
) -> SliceMeta {
    SliceMeta {
        model_id,
        n_layers,
        vocab,
        ctx_size,
        hidden_values,
        has_output: layers.has_output,
        layer_start: layers.start,
        layer_end: resolved_layer_end(layers, n_layers),
    }
}

pub fn worker_model_name(name: &str) -> &str {
    if name.is_empty() {
        UNKNOWN_MODEL_NAME
    } else {
        name
    }
}

pub fn worker_hello(
    meta: &SliceMeta,
    quant_bits: u32,
    listen_port: u32,
    model_name: &str,
) -> Hello {
    let name = worker_model_name(model_name);
    let name_len = name.len().min(MAX_MODEL_NAME as usize) as u32;
    Hello {
        model_id: meta.model_id,
        quant_bits,
        layer_start: meta.layer_start,
        layer_end: meta.layer_end,
        has_output: u32::from(meta.has_output),
        has_hidden: 1,
        ctx_size: meta.ctx_size,
        n_layers: meta.n_layers,
        listen_port,
        model_name_len: name_len,
    }
}

pub struct WorkerPlan {
    pub hello: Hello,
    pub model_name: String,
}

pub fn worker_plan(
    meta: &SliceMeta,
    quant_bits: u32,
    listen_port: u32,
    model_name: &str,
) -> WorkerPlan {
    let model_name = worker_model_name(model_name).to_string();
    let hello = worker_hello(meta, quant_bits, listen_port, &model_name);
    WorkerPlan { hello, model_name }
}

pub struct AssembledWorker<E: SliceExec, S: SnapshotStore> {
    pub hello: Hello,
    pub model_name: String,
    pub worker: Worker<E, S>,
}

pub fn assemble_worker<E: SliceExec, S: SnapshotStore>(
    exec: E,
    store: S,
    hello: Hello,
    model_name: String,
) -> AssembledWorker<E, S> {
    AssembledWorker {
        hello,
        model_name,
        worker: Worker::with_store(exec, store),
    }
}

pub fn worker_listen_port(listen_port: i32) -> u16 {
    u16::try_from(listen_port).unwrap_or(0)
}

pub struct WorkerListenBanner<'a> {
    pub layer_start: u32,
    pub has_output: bool,
    pub layer_end: u32,
    pub model_id: i32,
    pub listen_host: Option<&'a str>,
    pub listen_port: u32,
    pub coordinator_host: &'a str,
    pub coordinator_port: i32,
}

pub fn worker_listen_banner(spec: &WorkerListenBanner<'_>) -> String {
    let layer_end: Cow<'_, str> = if spec.has_output {
        Cow::Borrowed("output")
    } else {
        Cow::Owned(spec.layer_end.to_string())
    };
    let listen_host = spec.listen_host.unwrap_or("*");
    format!(
        "ds4: distributed worker: layers {}:{} model_id={} data_listen={}:{} connecting to coordinator {}:{}",
        spec.layer_start,
        layer_end,
        spec.model_id,
        listen_host,
        spec.listen_port,
        spec.coordinator_host,
        spec.coordinator_port,
    )
}

pub fn print_worker_listen_banner(spec: &WorkerListenBanner<'_>) {
    eprintln!("{}", worker_listen_banner(spec));
}
