//! Worker reconnect: connect retries, HELLO, session clear, 1s backoff.

use std::io::{self, ErrorKind};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

use crate::codec::Hello;
use crate::exec::SliceExec;
use crate::native_snapshot::SnapshotStore;
use crate::prefetch::prefetch_disabled;
use crate::serve_local::{serve_coordinator_and_hops, spawn_hop_accept, HopAccept};
use crate::worker::{send_hello, Worker};

pub const RECONNECT_SLEEP: Duration = Duration::from_secs(1);
pub const CONNECT_RETRY_DELAY: Duration = Duration::from_millis(25);
pub const CONNECT_RETRY_ATTEMPTS: u32 = 200;

pub fn sleep_reconnect() {
    thread::sleep(RECONNECT_SLEEP);
}

pub fn connect_retryable(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        ErrorKind::ConnectionRefused
            | ErrorKind::HostUnreachable
            | ErrorKind::NetworkUnreachable
            | ErrorKind::TimedOut
            | ErrorKind::AddrNotAvailable
    )
}

pub fn connect_error(host: &str, port: u16, err: &io::Error) -> String {
    format!("unable to connect to {host}:{port}: {err}")
}

pub fn retrying_message(err: &str) -> String {
    format!("ds4: distributed worker: {err}; retrying")
}

pub fn connected_message(host: &str, port: &str) -> String {
    format!("ds4: distributed worker: connected to coordinator {host}:{port}")
}

pub fn hello_failed_message(err: &io::Error) -> String {
    format!("ds4: distributed worker: failed to send HELLO: {err}")
}

pub fn cleared_sessions_message(n: u32) -> String {
    format!("ds4: distributed worker: cleared {n} sessions after coordinator disconnect")
}

pub fn disconnected_message(after_error: bool) -> String {
    if after_error {
        "ds4: distributed worker: coordinator disconnected after error; reconnecting".into()
    } else {
        "ds4: distributed worker: coordinator disconnected; reconnecting".into()
    }
}

pub fn connect_endpoint_once(host: &str, port: u16) -> io::Result<TcpStream> {
    let stream = TcpStream::connect((host, port))?;
    let _ = stream.set_nodelay(true);
    Ok(stream)
}

pub fn connect_endpoint(host: &str, port: u16) -> io::Result<TcpStream> {
    let mut last = None;
    for _ in 0..CONNECT_RETRY_ATTEMPTS {
        match connect_endpoint_once(host, port) {
            Ok(s) => return Ok(s),
            Err(e) => {
                let retry = connect_retryable(&e);
                last = Some(e);
                if !retry {
                    break;
                }
                thread::sleep(CONNECT_RETRY_DELAY);
            }
        }
    }
    Err(last.unwrap_or_else(|| io::Error::new(ErrorKind::Other, "unable to connect")))
}

pub fn peer_name(stream: &TcpStream) -> (String, String) {
    match stream.peer_addr() {
        Ok(SocketAddr::V4(a)) => (a.ip().to_string(), a.port().to_string()),
        Ok(SocketAddr::V6(a)) => (a.ip().to_string(), a.port().to_string()),
        Err(_) => ("?".into(), "?".into()),
    }
}

pub struct LocalReconnect<'a, Connect, Sleep, Stop> {
    pub connect: Connect,
    pub hello: &'a Hello,
    pub model_name: &'a str,
    pub sleep: Sleep,
    pub should_stop: Stop,
    pub listener: Option<&'a TcpListener>,
}

fn reconnect_loop<E, S, Connect, Sleep, Stop, Serve>(
    worker: &mut Worker<E, S>,
    connect: &mut Connect,
    hello: &Hello,
    model_name: &str,
    sleep: &mut Sleep,
    should_stop: &mut Stop,
    mut serve: Serve,
) -> io::Result<()>
where
    E: SliceExec,
    S: SnapshotStore,
    Connect: FnMut() -> io::Result<TcpStream>,
    Sleep: FnMut(),
    Stop: FnMut() -> bool,
    Serve: FnMut(&mut Worker<E, S>, &mut TcpStream, &mut Stop) -> io::Result<()>,
{
    while !should_stop() {
        let mut stream = match connect() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{}", retrying_message(&e.to_string()));
                sleep();
                continue;
            }
        };
        let (peer_host, peer_port) = peer_name(&stream);
        eprintln!("{}", connected_message(&peer_host, &peer_port));
        if let Err(e) = send_hello(&mut stream, hello, model_name) {
            eprintln!("{}", hello_failed_message(&e));
            sleep();
            continue;
        }
        let rc = serve(worker, &mut stream, should_stop);
        let after_error = rc.is_err();
        let dropped = worker.clear_sessions();
        if dropped != 0 {
            eprintln!("{}", cleared_sessions_message(dropped));
        }
        eprintln!("{}", disconnected_message(after_error));
        if should_stop() {
            return rc;
        }
        sleep();
        if after_error {
            let _ = rc;
        }
    }
    Ok(())
}

pub fn reconnect_with<E, S, Connect, Sleep, Stop>(
    worker: &mut Worker<E, S>,
    mut connect: Connect,
    hello: &Hello,
    model_name: &str,
    mut sleep: Sleep,
    mut should_stop: Stop,
    use_prefetch: bool,
) -> io::Result<()>
where
    E: SliceExec + Send,
    S: SnapshotStore + Send,
    Connect: FnMut() -> io::Result<TcpStream>,
    Sleep: FnMut(),
    Stop: FnMut() -> bool,
{
    reconnect_loop(
        worker,
        &mut connect,
        hello,
        model_name,
        &mut sleep,
        &mut should_stop,
        |w, stream, _stop| {
            if use_prefetch && !prefetch_disabled() {
                w.serve_prefetch(stream)
            } else {
                w.serve(stream)
            }
        },
    )
}

/// Reconnect a `!Send` worker, polling the coordinator and accepted hop streams.
pub fn reconnect_local<E, S, Connect, Sleep, Stop>(
    worker: &mut Worker<E, S>,
    mut spec: LocalReconnect<'_, Connect, Sleep, Stop>,
) -> io::Result<()>
where
    E: SliceExec,
    S: SnapshotStore,
    Connect: FnMut() -> io::Result<TcpStream>,
    Sleep: FnMut(),
    Stop: FnMut() -> bool,
{
    let hops = spec.listener.map(spawn_hop_accept).transpose()?;
    let rx = hops.as_ref().and_then(HopAccept::receiver);
    reconnect_loop(
        worker,
        &mut spec.connect,
        spec.hello,
        spec.model_name,
        &mut spec.sleep,
        &mut spec.should_stop,
        |w, stream, stop| serve_coordinator_and_hops(w, stream, rx, stop),
    )
}
