//! KVC disk store. File format, SHA key, eviction, and prefix lookup
//! match `ds4_kvstore.c` at `v0.6.3-dfm`. Payload bytes stay opaque.

mod format;
mod policy;
mod sha1;
mod store;

pub use format::{
    decode_file, encode_file, fill_header, key_kind, parse_header, path_for_sha, read_envelope,
    read_path, read_trailer, sha_hex_name, text_sha_hex, write_path, Envelope, FormatError, Header,
    Reason, Record, EXT_BANK_REPLAY_V1, EXT_RESPONSES_VISIBLE, EXT_SESSION_TITLE,
    EXT_THINKING_VISIBLE, EXT_TOOL_MAP, FIXED_HEADER, PAYLOAD_ABI, VERSION,
};
pub use policy::{
    bank_checkpoint_due, chat_anchor_pos, continued_store_target, eviction_score, file_size_fits,
    store_len, EvictionContext, Options, ScoreEntry,
};
pub use sha1::sha1_hex;
pub use store::{Entry, PayloadTemp, Store};
