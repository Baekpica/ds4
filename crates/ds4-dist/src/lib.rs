//! Distributed coordinator/worker wire. Explicit integer codecs only.

mod activation;
mod codec;
mod hash;
mod route;
mod transport;

pub use activation::{
    bits_or_default, bits_valid, decode_activation, encode_activation, f16_to_f32, f32_to_f16,
    f32_to_f8_e4m3, f8_e4m3_to_f32, values_from_wire_bytes, wire_bytes, wire_bytes_from_f32_bytes,
    BITS_DEFAULT,
};
pub use codec::{
    bytes_have_nul, decode_frame_header, decode_hello_payload, decode_tokens_be,
    encode_error_frame, encode_frame_header, encode_hello_payload, encode_tokens_be, get_u32_be,
    put_u32_be, u64_from_halves, u64_to_halves, CodecError, FrameHeader, Hello, ResultHdr, Route,
    RouteReturn, SnapshotBegin, SnapshotChunk, SnapshotDone, SnapshotReq, Telemetry, Work,
    FRAME_HEADER_BYTES, HELLO_FIXED_BYTES, MAGIC, MAX_MODEL_NAME, MSG_ERROR, MSG_HELLO, MSG_RESULT,
    MSG_SNAPSHOT_BEGIN, MSG_SNAPSHOT_CHUNK, MSG_SNAPSHOT_DONE, MSG_SNAPSHOT_LOAD_BEGIN,
    MSG_SNAPSHOT_SAVE_REQ, MSG_WORK, RESULT_ACK, RESULT_FIXED_BYTES, RESULT_HIDDEN_STATE,
    RESULT_LOGITS, ROUTE_FIXED_BYTES, ROUTE_F_OUTPUT_LOGITS, ROUTE_RETURN_FIXED_BYTES,
    ROUTE_RETURN_UPSTREAM, SNAPSHOT_BEGIN_FIXED_BYTES, SNAPSHOT_CHUNK_FIXED_BYTES,
    SNAPSHOT_DONE_FIXED_BYTES, SNAPSHOT_REQ_FIXED_BYTES, TELEMETRY_FIXED_BYTES, WORK_FIXED_BYTES,
    WORK_F_ACK_ONLY, WORK_F_INPUT_HC, WORK_F_OUTPUT_LOGITS, WORK_F_RESET_SESSION, WORK_F_VALID_MASK,
};
pub use hash::{token_hash_prefix, token_hash_update, token_hash_update_span, TOKEN_HASH_INIT};
pub use route::{
    decode_route_blob, encode_route_blob, validate_route_blob, ReturnTarget, RouteEntry,
};
pub use transport::{read_frame, write_frame};
