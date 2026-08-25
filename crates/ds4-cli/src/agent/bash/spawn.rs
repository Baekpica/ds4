use super::{BashJob, BashTable};
use crate::agent::web_tools::io_detail;
use std::ffi::CString;
use std::fs::File;
use std::os::fd::FromRawFd;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

unsafe extern "C" {
    fn mkstemp(template: *mut std::ffi::c_char) -> i32;
}

pub(super) fn spawn_job(
    table: &mut BashTable,
    command: &str,
    timeout: i32,
) -> Result<usize, String> {
    let (path, file) = create_output_file()?;
    let stdout = file.try_clone().map_err(|error| {
        format!(
            "failed to create temporary output file: {}",
            io_detail(&error)
        )
    })?;
    let stderr = file.try_clone().map_err(|error| {
        format!(
            "failed to create temporary output file: {}",
            io_detail(&error)
        )
    })?;
    drop(file);
    let child = Command::new("/bin/sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::inherit())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0)
        .spawn()
        .map_err(|error| format!("failed to fork: {}", io_detail(&error)))?;
    let pid = child.id() as i32;
    if table.next_id <= 0 {
        table.next_id = 1;
    }
    let id = table.next_id;
    table.next_id += 1;
    table.jobs.insert(
        0,
        BashJob {
            id,
            pid,
            path,
            child: Some(child),
            start: Instant::now(),
            timeout_sec: f64::from(timeout),
            observed_once: false,
            exit_status: -1,
            running: true,
            timed_out: false,
        },
    );
    Ok(0)
}

fn create_output_file() -> Result<(PathBuf, File), String> {
    let mut tmpl = CString::new("/tmp/ds4_agent_output_XXXXXX")
        .map_err(|_| "failed to create temporary output file: invalid template".to_string())?
        .into_bytes_with_nul();
    let fd = unsafe { mkstemp(tmpl.as_mut_ptr().cast()) };
    if fd < 0 {
        return Err(format!(
            "failed to create temporary output file: {}",
            io_detail(&std::io::Error::last_os_error())
        ));
    }
    let path = CString::from_vec_with_nul(tmpl)
        .map_err(|_| "failed to create temporary output file: invalid path".to_string())?
        .into_string()
        .map_err(|_| "failed to create temporary output file: invalid path".to_string())?;
    Ok((PathBuf::from(path), unsafe { File::from_raw_fd(fd) }))
}
