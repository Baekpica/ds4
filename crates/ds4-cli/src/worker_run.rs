//! Assemble a same-thread worker from shared Session exec/store.

use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use ds4_core::{Model, Session};
use ds4_dist::{Hello, Options, Worker};

use crate::session_exec::{slice_meta, SessionSliceExec, SliceMeta};
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

pub fn worker_listen_port(listen_port: i32) -> u16 {
    u16::try_from(listen_port).unwrap_or(0)
}

pub fn run_assembled_worker(model: &Model, ctx: i32, dist: &Options) -> Result<i32, String> {
    let session = Rc::new(RefCell::new(model.session(ctx).map_err(|e| e.to_string())?));
    let (listener, bound) = ds4_dist::open_data_listener(
        dist.listen_host.as_deref(),
        worker_listen_port(dist.listen_port),
    )
    .map_err(|e| e.to_string())?;
    let shape = &model.bind_plan().shape;
    let mut meta = slice_meta(
        u32::try_from(model.model_id()).unwrap_or(0),
        shape,
        u32::try_from(ctx).unwrap_or(0),
        &dist.layers,
    );
    meta.vocab = u32::try_from(model.vocab().n_vocab().max(0)).unwrap_or(0);
    let mut assembled = assemble_worker(
        session,
        meta,
        u32::try_from(model.routed_quant_bits().max(0)).unwrap_or(0),
        u32::from(bound),
        shape.name,
    );
    let host = dist
        .coordinator_host
        .as_deref()
        .ok_or("--role worker requires --coordinator HOST PORT")?;
    print_worker_listen_banner(&WorkerListenBanner {
        layer_start: assembled.hello.layer_start,
        has_output: assembled.hello.has_output != 0,
        layer_end: assembled.hello.layer_end,
        model_id: model.model_id(),
        listen_host: dist.listen_host.as_deref(),
        listen_port: assembled.hello.listen_port,
        coordinator_host: host,
        coordinator_port: dist.coordinator_port,
    });
    let port = u16::try_from(dist.coordinator_port)
        .map_err(|_| "--role worker requires --coordinator HOST PORT")?;
    ds4_dist::reconnect_local(
        &mut assembled.worker,
        ds4_dist::LocalReconnect {
            connect: || ds4_dist::connect_endpoint(host, port),
            hello: &assembled.hello,
            model_name: &assembled.model_name,
            sleep: ds4_dist::sleep_reconnect,
            should_stop: || false,
            listener: Some(&listener),
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(0)
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
    fn assemble_worker_compiles_against_reconnect_local() {
        type SessionWorker<'m> = Worker<SessionSliceExec<'m>, SessionSnapshotStore<'m>>;
        fn bind<'m, C, Sl, St>(
            worker: &mut SessionWorker<'m>,
            spec: ds4_dist::LocalReconnect<'_, C, Sl, St>,
        ) -> std::io::Result<()>
        where
            C: FnMut() -> std::io::Result<std::net::TcpStream>,
            Sl: FnMut(),
            St: FnMut() -> bool,
        {
            ds4_dist::reconnect_local(worker, spec)
        }
        let _ = bind::<fn() -> std::io::Result<std::net::TcpStream>, fn(), fn() -> bool>;
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

    #[test]
    fn worker_listen_banner_contains_data_listen_and_coordinator() {
        // Given: known worker meta (C layers 20:output, model_id 7)
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        let plan = worker_plan(&meta, 2, 7100, SHAPE_FLASH.name);

        // When: format the C boot stderr line
        let line = worker_listen_banner(&WorkerListenBanner {
            layer_start: plan.hello.layer_start,
            has_output: plan.hello.has_output != 0,
            layer_end: plan.hello.layer_end,
            model_id: i32::try_from(plan.hello.model_id).expect("model_id fits i32"),
            listen_host: Some("127.0.0.1"),
            listen_port: plan.hello.listen_port,
            coordinator_host: "10.0.0.1",
            coordinator_port: 1234,
        });

        // Then: C tokens and full fprintf shape
        assert!(
            line.contains("data_listen="),
            "banner must contain data_listen=: {line}"
        );
        assert!(
            line.contains("connecting to coordinator"),
            "banner must contain connecting to coordinator: {line}"
        );
        assert_eq!(
            line,
            "ds4: distributed worker: layers 20:output model_id=7 data_listen=127.0.0.1:7100 connecting to coordinator 10.0.0.1:1234"
        );
        assert_eq!(
            ds4_dist::connected_message("10.0.0.1", "1234"),
            "ds4: distributed worker: connected to coordinator 10.0.0.1:1234"
        );
        assert_eq!(
            ds4_dist::disconnected_message(false),
            "ds4: distributed worker: coordinator disconnected; reconnecting"
        );
    }

    #[test]
    fn worker_listen_banner_prints_star_when_host_missing() {
        // Given: no listen host (C listen_host == NULL)
        let spec = WorkerListenBanner {
            layer_start: 20,
            has_output: false,
            layer_end: 30,
            model_id: 7,
            listen_host: None,
            listen_port: 7100,
            coordinator_host: "10.0.0.1",
            coordinator_port: 1234,
        };

        // When: format the C boot stderr line
        let line = worker_listen_banner(&spec);

        // Then: missing host is `*` like C
        assert!(
            line.contains("data_listen=*:7100"),
            "missing host must print *: {line}"
        );
        assert!(
            line.contains("connecting to coordinator"),
            "banner must contain connecting to coordinator: {line}"
        );
        assert_eq!(
            line,
            "ds4: distributed worker: layers 20:30 model_id=7 data_listen=*:7100 connecting to coordinator 10.0.0.1:1234"
        );
    }
}
