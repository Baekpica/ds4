//! Assemble a same-thread worker from shared Session exec/store.

use std::cell::RefCell;
use std::rc::Rc;

use ds4_core::Session;
use ds4_dist::{Hello, Worker};

use crate::session_exec::{SessionSliceExec, SliceMeta};
use crate::session_snapshot::SessionSnapshotStore;
use crate::worker_hello::{worker_hello, worker_model_name};

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

pub struct AssembledWorker<'m> {
    pub hello: Hello,
    pub model_name: String,
    pub worker: Worker<SessionSliceExec<'m>, SessionSnapshotStore<'m>>,
}

pub fn assemble_worker<'m>(
    session: Rc<RefCell<Session<'m>>>,
    meta: SliceMeta,
    quant_bits: u32,
    listen_port: u32,
    model_name: &str,
) -> AssembledWorker<'m> {
    let plan = worker_plan(&meta, quant_bits, listen_port, model_name);
    let exec = SessionSliceExec::from_meta(Rc::clone(&session), meta);
    let store = SessionSnapshotStore::from_shared(session, meta.layer_start, meta.layer_end);
    AssembledWorker {
        hello: plan.hello,
        model_name: plan.model_name,
        worker: Worker::with_store(exec, store),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_exec::slice_meta;
    use ds4_core::SHAPE_FLASH;
    use ds4_dist::Layers;

    #[test]
    fn worker_plan_uses_bound_data_listen_port() {
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        let (_listener, port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
        let plan = worker_plan(&meta, 2, u32::from(port), SHAPE_FLASH.name);
        assert_eq!(plan.hello.listen_port, u32::from(port));
        assert_ne!(plan.hello.listen_port, 0);
    }

    #[test]
    fn worker_plan_matches_hello_and_layer_range() {
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        let plan = worker_plan(&meta, 2, 7100, SHAPE_FLASH.name);
        assert_eq!(plan.hello.model_id, 7);
        assert_eq!(plan.hello.quant_bits, 2);
        assert_eq!(plan.hello.layer_start, meta.layer_start);
        assert_eq!(plan.hello.layer_end, meta.layer_end);
        assert_eq!(plan.hello.has_output, 1);
        assert_eq!(plan.hello.has_hidden, 1);
        assert_eq!(plan.hello.listen_port, 7100);
        assert_eq!(plan.model_name, SHAPE_FLASH.name);
        assert_eq!(plan.hello.model_name_len, SHAPE_FLASH.name.len() as u32);
    }
}
