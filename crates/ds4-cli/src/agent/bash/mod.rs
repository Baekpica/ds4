mod obs;
mod spawn;

use super::search::scan::parse_int_default as parse_int;
use obs::observation;
use spawn::spawn_job;
use std::ffi::CString;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::Child;
use std::time::Instant;

pub(crate) const BASH: &str = "bash";
pub(crate) const BASH_STATUS: &str = "bash_status";
pub(crate) const BASH_STOP: &str = "bash_stop";

const DEFAULT_TIMEOUT: i32 = 3600;
const MAX_TIMEOUT: f64 = 24.0 * 3600.0;
const DEFAULT_REFRESH: i32 = 60;
const MAX_REFRESH: i32 = 3600;

unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
    fn strtod(nptr: *const std::ffi::c_char, endptr: *mut *mut std::ffi::c_char) -> f64;
}

const SIGTERM: i32 = 15;
const SIGKILL: i32 = 9;

pub(crate) struct BashTable {
    pub(super) next_id: i32,
    pub(super) jobs: Vec<BashJob>,
}

impl Default for BashTable {
    fn default() -> Self {
        Self {
            next_id: 1,
            jobs: Vec::new(),
        }
    }
}

pub(crate) struct BashJob {
    pub(super) id: i32,
    pub(super) pid: i32,
    pub(super) path: PathBuf,
    pub(super) child: Option<Child>,
    pub(super) start: Instant,
    pub(super) timeout_sec: f64,
    pub(super) observed_once: bool,
    pub(super) exit_status: i32,
    pub(super) running: bool,
    pub(super) timed_out: bool,
}

impl Drop for BashTable {
    fn drop(&mut self) {
        for job in &mut self.jobs {
            job.kill_group(SIGKILL);
            let _ = job.reap(true);
        }
    }
}

impl BashJob {
    fn kill_group(&self, sig: i32) {
        if self.pid > 0 {
            unsafe {
                // SAFETY: pid is a process group we created with process_group(0).
                kill(-self.pid, sig);
                kill(self.pid, sig);
            }
        }
    }

    fn reap(&mut self, block: bool) -> bool {
        let Some(child) = self.child.as_mut() else {
            return !self.running;
        };
        let status = if block {
            child.wait().ok()
        } else {
            child.try_wait().ok().flatten()
        };
        if let Some(status) = status {
            self.exit_status = status
                .code()
                .unwrap_or_else(|| status.signal().map(|sig| 128 + sig).unwrap_or(-1));
            self.running = false;
            self.child = None;
            true
        } else {
            false
        }
    }

    fn poll(&mut self) {
        if !self.running {
            return;
        }
        if self.reap(false) {
            return;
        }
        if self.start.elapsed().as_secs_f64() >= self.timeout_sec {
            self.timed_out = true;
            self.kill_group(SIGKILL);
            self.reap(true);
        }
    }
}

pub(crate) fn parse_timeout(value: Option<&str>) -> i32 {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return DEFAULT_TIMEOUT;
    };
    let Ok(c_value) = CString::new(value) else {
        return DEFAULT_TIMEOUT;
    };
    let mut end = std::ptr::null_mut();
    let parsed = unsafe { strtod(c_value.as_ptr(), &mut end) };
    if end == c_value.as_ptr().cast_mut() || !parsed.is_finite() || parsed <= 0.0 {
        return DEFAULT_TIMEOUT;
    }
    parsed.clamp(1.0, MAX_TIMEOUT) as i32
}

pub(crate) fn bash_result(table: &mut BashTable, name: &str, call: &BashArgs<'_>) -> Vec<u8> {
    match name {
        BASH => start_bash(table, call),
        BASH_STATUS | BASH_STOP => follow_bash(table, name, call),
        _ => b"Tool error: unknown tool\n".to_vec(),
    }
}

pub(crate) struct BashArgs<'a> {
    pub command: Option<&'a str>,
    pub timeout_sec: Option<&'a str>,
    pub refresh_sec: Option<&'a str>,
    pub job: Option<&'a str>,
    pub pid: Option<&'a str>,
}

fn start_bash(table: &mut BashTable, call: &BashArgs<'_>) -> Vec<u8> {
    let Some(command) = call.command.filter(|command| !command.is_empty()) else {
        return b"Tool error: bash requires command\n".to_vec();
    };
    let timeout = parse_timeout(call.timeout_sec);
    let refresh = parse_int(call.refresh_sec, DEFAULT_REFRESH, 1, MAX_REFRESH);
    match spawn_job(table, command, timeout) {
        Ok(index) => finish_job(table, index, true, refresh, false),
        Err(error) => format!("Tool error: bash failed to start: {error}\n").into_bytes(),
    }
}

fn follow_bash(table: &mut BashTable, name: &str, call: &BashArgs<'_>) -> Vec<u8> {
    let job_id = parse_int(call.job, 0, 0, i32::MAX);
    let pid = parse_int(call.pid, 0, 0, i32::MAX);
    let refresh = parse_int(call.refresh_sec, DEFAULT_REFRESH, 1, MAX_REFRESH);
    let Some(index) = find_job(table, job_id, pid) else {
        return format!("Tool error: bash job not found: job={job_id} pid={pid}\n").into_bytes();
    };
    finish_job(table, index, name == BASH_STOP, refresh, name == BASH_STOP)
}

fn find_job(table: &BashTable, id: i32, pid: i32) -> Option<usize> {
    table
        .jobs
        .iter()
        .position(|job| (id > 0 && job.id == id) || (id <= 0 && pid > 0 && job.pid == pid))
}

fn finish_job(
    table: &mut BashTable,
    index: usize,
    wait: bool,
    refresh: i32,
    stop: bool,
) -> Vec<u8> {
    if stop {
        stop_job(&mut table.jobs[index]);
    }
    if wait || stop {
        refresh_for(&mut table.jobs[index], refresh);
    } else {
        table.jobs[index].poll();
    }
    let out = observation(&mut table.jobs[index]);
    if !table.jobs[index].running {
        table.jobs.remove(index);
    }
    out
}

fn stop_job(job: &mut BashJob) {
    if !job.running {
        return;
    }
    job.kill_group(SIGTERM);
    let start = Instant::now();
    while job.running && start.elapsed().as_secs_f64() < 1.0 {
        job.poll();
        if !job.running {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if job.running {
        job.kill_group(SIGKILL);
        job.reap(true);
    }
}

fn refresh_for(job: &mut BashJob, refresh: i32) {
    let start = Instant::now();
    let limit = f64::from(refresh);
    while job.running && start.elapsed().as_secs_f64() < limit {
        job.poll();
        if !job.running {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    job.poll();
}
