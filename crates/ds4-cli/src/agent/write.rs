use super::web_tools::io_detail;
use std::fs::File;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;

pub(crate) const WRITE: &str = "write";

pub(crate) fn write_result(path: Option<&str>, content: Option<&str>) -> Vec<u8> {
    let Some(path) = path.filter(|path| !path.is_empty()) else {
        return b"Tool error: write requires path\n".to_vec();
    };
    let Some(content) = content else {
        return b"Tool error: write requires content\n".to_vec();
    };
    let os_path = std::ffi::OsStr::from_bytes(path.as_bytes());
    let mut file = match File::create(os_path) {
        Ok(file) => file,
        Err(error) => {
            return format!("Tool error: open for write failed: {}\n", io_detail(&error))
                .into_bytes();
        }
    };
    if let Err(error) = file
        .write_all(content.as_bytes())
        .and_then(|_| file.flush())
    {
        return format!("Tool error: write failed: {}\n", io_detail(&error)).into_bytes();
    }
    format!("Wrote {} bytes to {path}\n", content.len()).into_bytes()
}
