use super::approval::{Approval, Ask, DENIED_EDIT, DENIED_WRITE};
use super::bash::BashTable;
use super::web_tools::{handle_round_with_tools, Browser, ReadCursor, ToolRound};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct NoWeb;

impl Browser for NoWeb {
    fn google_search(&mut self, _query: &str) -> Result<String, String> {
        Err("unused".into())
    }

    fn visit_page(&mut self, _url: &str) -> Result<String, String> {
        Err("unused".into())
    }
}

struct DenyAsk;

impl Ask for DenyAsk {
    fn yes_no(&mut self, _prompt: &str) -> bool {
        false
    }
}

fn temp_path(name: &str) -> PathBuf {
    PathBuf::from(format!(
        "/tmp/ds4_agent_approval_round_{}_{}_{name}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn write_call(path: &str, content: &str) -> Vec<u8> {
    format!(
        "<｜DSML｜tool_calls>\n\
         <｜DSML｜invoke name=\"write\">\n\
         <｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n\
         <｜DSML｜parameter name=\"content\" string=\"true\">{content}</｜DSML｜parameter>\n\
         </｜DSML｜invoke>\n\
         </｜DSML｜tool_calls>"
    )
    .into_bytes()
}

fn edit_call(path: &str, old: &str, new: &str) -> Vec<u8> {
    format!(
        "<｜DSML｜tool_calls>\n\
         <｜DSML｜invoke name=\"edit\">\n\
         <｜DSML｜parameter name=\"path\" string=\"true\">{path}</｜DSML｜parameter>\n\
         <｜DSML｜parameter name=\"old\" string=\"true\">{old}</｜DSML｜parameter>\n\
         <｜DSML｜parameter name=\"new\" string=\"true\">{new}</｜DSML｜parameter>\n\
         </｜DSML｜invoke>\n\
         </｜DSML｜tool_calls>"
    )
    .into_bytes()
}

fn round(raw: &[u8], approval: &mut Approval<'_>) -> ToolRound {
    handle_round_with_tools(
        raw,
        &mut NoWeb,
        &mut ReadCursor::default(),
        &mut BashTable::default(),
        approval,
    )
    .expect("valid DSML")
}

#[test]
fn write_round_deny_does_not_mutate_file() {
    let path = temp_path("write-deny.txt");
    std::fs::write(&path, b"old").unwrap();
    let path_text = path.to_str().expect("utf8");
    let mut ask = DenyAsk;
    let mut approval = Approval::Interactive(&mut ask);
    let out = round(&write_call(path_text, "new"), &mut approval);
    let mut expected = b"Tool result 1 (write):\n".to_vec();
    expected.extend_from_slice(DENIED_WRITE);
    assert_eq!(out.observation, expected);
    assert_eq!(std::fs::read(&path).unwrap(), b"old");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn edit_round_deny_does_not_mutate_file() {
    let path = temp_path("edit-deny.txt");
    std::fs::write(&path, b"alpha\nkeep\n").unwrap();
    let path_text = path.to_str().expect("utf8");
    let mut ask = DenyAsk;
    let mut approval = Approval::Interactive(&mut ask);
    let out = round(&edit_call(path_text, "alpha", "beta"), &mut approval);
    let mut expected = b"Tool result 1 (edit):\n".to_vec();
    expected.extend_from_slice(DENIED_EDIT);
    assert_eq!(out.observation, expected);
    assert_eq!(std::fs::read(&path).unwrap(), b"alpha\nkeep\n");
    std::fs::remove_file(path).unwrap();
}
