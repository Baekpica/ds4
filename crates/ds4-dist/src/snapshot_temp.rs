//! Bounded temp files for worker KV shard save/load.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::codec::SNAPSHOT_CHUNK_BYTES;

pub const SAVE_PREFIX: &str = "ds4-dist-save";
pub const LOAD_PREFIX: &str = "ds4-dist-load";

pub struct TempShard {
    pub(crate) file: File,
    pub(crate) path: PathBuf,
}

impl Drop for TempShard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Read for TempShard {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(buf)
    }
}

impl Write for TempShard {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl Seek for TempShard {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

fn temp_root() -> PathBuf {
    #[cfg(unix)]
    {
        PathBuf::from("/tmp")
    }
    #[cfg(not(unix))]
    {
        std::env::temp_dir()
    }
}

pub fn create_temp(prefix: &str) -> io::Result<TempShard> {
    let root = temp_root();
    for n in 0u32..1024 {
        let path = root.join(format!("{prefix}.{}.{:04x}", std::process::id(), n));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok(TempShard { file, path }),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "temp name space exhausted",
    ))
}

pub fn copy_chunked(
    src: &mut dyn Read,
    dest: &mut dyn Write,
    mut remaining: u64,
) -> io::Result<()> {
    if remaining == 0 {
        return Ok(());
    }
    let mut buf = vec![0u8; SNAPSHOT_CHUNK_BYTES];
    while remaining != 0 {
        let n = remaining.min(SNAPSHOT_CHUNK_BYTES as u64) as usize;
        src.read_exact(&mut buf[..n])?;
        dest.write_all(&buf[..n])?;
        remaining -= n as u64;
    }
    Ok(())
}
