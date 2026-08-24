//! Read-only file mmap for GGUF metadata.
//!
//! The mapping is the file. Callers store offsets; they do not slurp the
//! GGUF into `Vec<u8>`. `unsafe` stays in this adapter.

use std::fs::File;
use std::io;
use std::path::Path;
use std::ptr;

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

#[cfg(unix)]
mod sys {
    use std::os::raw::{c_int, c_void};

    pub const PROT_READ: c_int = 1;
    pub const MAP_PRIVATE: c_int = 2;

    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        pub fn sysconf(name: c_int) -> i64;
    }

    pub const _SC_PAGESIZE: c_int = 30;

    pub fn map_failed(p: *mut c_void) -> bool {
        p == !0 as *mut c_void
    }
}

pub struct MappedFile {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    pub fn open_ro(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len < 32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "model file is too small to be GGUF",
            ));
        }
        if len > isize::MAX as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "model file is too large to map",
            ));
        }
        let len = len as usize;
        #[cfg(unix)]
        {
            let ptr = unsafe {
                sys::mmap(
                    ptr::null_mut(),
                    len,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if sys::map_failed(ptr) {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                ptr: ptr as *mut u8,
                len,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = file;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "GGUF mmap is unix-only",
            ))
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// C `sysconf(_SC_PAGESIZE)` with the same 4096 fallback as `model_open_split`.
pub fn page_size() -> u64 {
    #[cfg(unix)]
    {
        let n = unsafe { sys::sysconf(sys::_SC_PAGESIZE) };
        if n > 0 {
            return n as u64;
        }
    }
    4096
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let _ = sys::munmap(self.ptr as *mut _, self.len);
        }
    }
}
