use super::{
    cache_dir_from_home, identity_sha, parse_slash, ListedSession, SaveSpec, SessionStore, SlashCmd,
};
use ds4_kv::decode_file;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn create() -> Self {
        let root = std::env::temp_dir().join(format!(
            "ds4_agent_kv_{}_{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    fn store(&self, model_id: u8) -> SessionStore {
        SessionStore::open(&self.root, model_id)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn save_named(
    store: &SessionStore,
    title: &str,
    created_at: u64,
    last_used: u64,
    payload: &[u8],
) -> String {
    store
        .save(SaveSpec {
            title,
            created_at,
            last_used,
            text: format!("<｜User｜>{title}<｜Assistant｜>ok").as_bytes(),
            payload,
            tokens: 8,
            model_id: 1,
            quant_bits: 2,
            ctx_size: 1024,
        })
        .unwrap()
}

#[test]
fn default_cache_dir_is_home_ds4_kvcache() {
    // Given: HOME like C getenv, or empty fallback "."
    // When: resolve the agent cache dir
    // Then: `$HOME/.ds4/kvcache` — not a new directory
    assert_eq!(
        cache_dir_from_home(Some("/home/alice".into())),
        PathBuf::from("/home/alice/.ds4/kvcache")
    );
    assert_eq!(
        cache_dir_from_home(Some("".into())),
        PathBuf::from("./.ds4/kvcache")
    );
    assert_eq!(cache_dir_from_home(None), PathBuf::from("./.ds4/kvcache"));
}

#[test]
fn list_orders_by_recent_then_sha() {
    // Given: three same-model sessions; older last_used first on disk
    let fx = Fixture::create();
    let store = fx.store(1);
    let older = save_named(&store, "older", 100, 100, b"PAY1");
    let newer = save_named(&store, "newer", 200, 300, b"PAY2");
    let tie_b = save_named(&store, "tie-b", 150, 200, b"PAY3");
    let tie_a = save_named(&store, "tie-a", 160, 200, b"PAY4");
    let _other_model = store
        .save(SaveSpec {
            title: "other-model",
            created_at: 400,
            last_used: 400,
            text: b"x",
            payload: b"P",
            tokens: 1,
            model_id: 9,
            quant_bits: 2,
            ctx_size: 1024,
        })
        .unwrap();
    std::fs::write(fx.root.join("sysprompt.kv"), b"not-a-session").unwrap();

    // When: list
    let listed = store.list().unwrap();

    // Then: last_used desc, sha asc on ties; sysprompt and other model_id dropped
    let mut tied = [tie_a.as_str(), tie_b.as_str()];
    tied.sort_unstable();
    let shas: Vec<&str> = listed.iter().map(|item| item.sha.as_str()).collect();
    assert_eq!(shas, vec![newer.as_str(), tied[0], tied[1], older.as_str()]);
    assert!(listed.iter().all(|item| !item.stripped));
    assert_eq!(listed[0].sha, newer);
    let _ = ListedSession {
        sha: newer,
        title: String::new(),
        last_used: 0,
        created_at: 0,
        tokens: 0,
        file_size: 0,
        stripped: false,
    };
}

#[test]
fn strip_then_switch_sets_prefill_flag() {
    // Given: a saved session with a KV payload
    let fx = Fixture::create();
    let store = fx.store(1);
    let sha = save_named(&store, "rebuild-me", 10, 10, b"HEAVY-KV");
    assert!(!store.switch_plan(&sha[..8]).unwrap().needs_prefill);

    // When: strip, then switch
    store.strip(&sha[..8], 12, 99).unwrap();
    let plan = store.switch_plan(&sha[..8]).unwrap();

    // Then: payload gone and switch rebuilds by prefill
    assert!(plan.needs_prefill);
    assert_eq!(plan.payload_bytes, 0);
    assert_eq!(plan.tokens, 12);
    assert_eq!(plan.sha, sha);
    let listed = store.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert!(listed[0].stripped);
}

#[test]
fn del_removes_file() {
    // Given: one saved session file
    let fx = Fixture::create();
    let store = fx.store(1);
    let sha = save_named(&store, "delete-me", 1, 1, b"KV");
    let path = fx.root.join(format!("{sha}.kv"));
    assert!(path.exists());

    // When: /del by prefix
    let deleted = store.delete(&sha[..8]).unwrap();

    // Then: the file is gone
    assert_eq!(deleted, sha);
    assert!(!path.exists());
    assert!(store.list().unwrap().is_empty());
}

#[test]
fn missing_sha_errors_like_c() {
    // Given: an empty cache dir
    let fx = Fixture::create();
    let store = fx.store(1);

    // When: find/switch/del/strip a missing prefix
    // Then: C `no saved session matches %.40s`
    assert_eq!(
        store.find("deadbeef").unwrap_err().to_string(),
        "no saved session matches deadbeef"
    );
    assert_eq!(
        store.switch_plan("abc").unwrap_err().to_string(),
        "no saved session matches abc"
    );
    assert_eq!(
        store.delete("fff").unwrap_err().to_string(),
        "no saved session matches fff"
    );
    assert_eq!(
        store.strip("0", 1, 0).unwrap_err().to_string(),
        "no saved session matches 0"
    );
    assert_eq!(
        store.find("").unwrap_err().to_string(),
        "invalid session SHA prefix"
    );
    assert_eq!(
        store.find("not-hex").unwrap_err().to_string(),
        "invalid session SHA prefix"
    );
}

#[test]
fn save_keeps_kvc_magic_and_stable_identity() {
    // Given: a title + created_at identity
    let fx = Fixture::create();
    let store = fx.store(1);
    let title = "stable-id";
    let created_at = 1_700_000_042u64;
    let expected = identity_sha(title, created_at);

    // When: save twice with a growing payload
    let first = save_named(&store, title, created_at, 10, b"v1");
    let second = save_named(&store, title, created_at, 20, b"v2-longer");

    // Then: same SHA filename; file still starts with C magic KVC
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    let bytes = std::fs::read(fx.root.join(format!("{expected}.kv"))).unwrap();
    assert_eq!(&bytes[..3], b"KVC");
    let record = decode_file(&bytes).unwrap();
    assert_eq!(record.payload, b"v2-longer");
}

#[test]
fn parse_slash_requires_sha_args_like_c() {
    assert_eq!(parse_slash("/save"), Some(Ok(SlashCmd::Save)));
    assert_eq!(parse_slash("/list"), Some(Ok(SlashCmd::List)));
    assert_eq!(
        parse_slash("/switch"),
        Some(Err("usage: /switch <sha-prefix>"))
    );
    assert_eq!(parse_slash("/del"), Some(Err("usage: /del <sha-prefix>")));
    assert_eq!(
        parse_slash("/strip"),
        Some(Err("usage: /strip <sha-prefix>"))
    );
    assert_eq!(
        parse_slash("/switch abcd"),
        Some(Ok(SlashCmd::Switch("abcd")))
    );
    assert_eq!(parse_slash("/help"), None);
}

#[test]
fn ambiguous_prefix_errors_like_c() {
    let fx = Fixture::create();
    let store = fx.store(1);
    let a = save_named(&store, "alpha", 1, 1, b"A");
    let b = save_named(&store, "bravo", 2, 2, b"B");
    let shared: String = a
        .chars()
        .zip(b.chars())
        .take_while(|(left, right)| left == right)
        .map(|(left, _)| left)
        .collect();
    if shared.is_empty() {
        return;
    }
    let err = store.find(&shared).unwrap_err().to_string();
    assert_eq!(err, format!("session prefix {shared:.40} is ambiguous"));
}
