//! Distributed coordinator/worker wire and blocking runtime. Explicit integer
//! codecs only — do not `#[repr(C)]` records onto the socket.

mod activation;
mod codec;
mod coordinator;
mod exec;
mod forward;
mod hash;
mod hops;
mod memory_snapshot;
mod native_snapshot;
mod options;
mod plan;
mod prefetch;
mod reconnect;
mod relay;
mod route;
mod snapshot;
mod snapshot_temp;
mod transport;
mod work;
mod worker;
mod worker_snapshot;

pub use activation::{
    bits_or_default, bits_valid, decode_activation, encode_activation, f16_to_f32, f32_to_f16,
    f32_to_f8_e4m3, f8_e4m3_to_f32, values_from_wire_bytes, wire_bytes, wire_bytes_from_f32_bytes,
    BITS_DEFAULT,
};
pub use codec::{
    bytes_have_nul, decode_frame_header, decode_hello_payload, decode_snapshot_begin_body,
    decode_snapshot_chunk_body, decode_snapshot_done_body, decode_snapshot_load_begin_body,
    decode_tokens_be, encode_error_frame, encode_frame_header, encode_hello_payload,
    encode_snapshot_begin_body, encode_snapshot_chunk_body, encode_snapshot_done_body,
    encode_tokens_be, get_u32_be, put_u32_be, u64_from_halves, u64_to_halves, CodecError,
    FrameHeader, Hello, ResultHdr, Route, RouteReturn, SnapshotBegin, SnapshotChunk, SnapshotDone,
    SnapshotReq, Telemetry, Work, FRAME_HEADER_BYTES, HELLO_FIXED_BYTES, MAGIC, MAX_MODEL_NAME,
    MSG_ERROR, MSG_HELLO, MSG_RESULT, MSG_SNAPSHOT_BEGIN, MSG_SNAPSHOT_CHUNK, MSG_SNAPSHOT_DONE,
    MSG_SNAPSHOT_LOAD_BEGIN, MSG_SNAPSHOT_SAVE_REQ, MSG_WORK, RESULT_ACK, RESULT_FIXED_BYTES,
    RESULT_HIDDEN_STATE, RESULT_LOGITS, ROUTE_FIXED_BYTES, ROUTE_F_OUTPUT_LOGITS,
    ROUTE_RETURN_FIXED_BYTES, ROUTE_RETURN_UPSTREAM, SNAPSHOT_BEGIN_FIXED_BYTES,
    SNAPSHOT_CHUNK_BYTES, SNAPSHOT_CHUNK_FIXED_BYTES, SNAPSHOT_DONE_FIXED_BYTES,
    SNAPSHOT_REQ_FIXED_BYTES, TELEMETRY_FIXED_BYTES, WORK_FIXED_BYTES, WORK_F_ACK_ONLY,
    WORK_F_INPUT_HC, WORK_F_OUTPUT_LOGITS, WORK_F_RESET_SESSION, WORK_F_VALID_MASK,
};
pub use coordinator::{
    accept_loop, dispatch_eval, format_telemetry_line, listen, recv_hello_only, token_span_hashes,
    Coordinator, EvalOutcome, RegisteredWorker, SharedCoordinator,
};
pub use exec::{SliceExec, WorkOutput, WorkRequest};
pub use forward::{
    forward_window, forward_window_from, opened_forwarder_message, PendingQueue, PendingRequest,
    ERR_CLOSED_WHILE_RESULT, ERR_FORWARD, ERR_FORWARD_HIDDEN, ERR_INVALID_RESULT, ERR_NEXT_CLOSED,
    ERR_OOM_FORWARDER, ERR_OOM_TRACK, ERR_RELAY_THREAD, ERR_RESULT_METADATA, ERR_RESULT_TOO_LARGE,
    ERR_TELEMETRY_TOO_LARGE, FORWARD_WINDOW_DEFAULT, FORWARD_WINDOW_MAX, FORWARD_WINDOW_MIN,
};
pub use hash::{token_hash_prefix, token_hash_update, token_hash_update_span, TOKEN_HASH_INIT};
pub use hops::ForwarderPool;
pub use native_snapshot::{
    apply_snapshot_load, dispatch_worker_snapshot, prepare_snapshot_save, MemorySnapshotStore,
    SnapshotLoad, SnapshotSave, SnapshotStore, TempShard,
};
pub use options::{
    parse_cli, parse_cli_arg, parse_layers, parse_role, prepare_engine_options, resolved_layer_end,
    validate_layers_for_model, validate_options, CliResult, Layers, Options, Role, USAGE,
};
pub use plan::{build_route_plan, register_worker, CoordinatorView, RoutePlan, WorkerInfo};
pub use prefetch::{
    prefetch_depth, prefetch_depth_from, prefetch_disabled, prefetch_disabled_from,
    prefetch_enabled_message, JobQueue, ERR_OOM_QUEUE, ERR_OOM_READ, PREFETCH_DEPTH_DEFAULT,
    PREFETCH_DEPTH_MAX, PREFETCH_DEPTH_MIN,
};
pub use reconnect::{
    cleared_sessions_message, connect_endpoint, connect_endpoint_once, connect_error,
    connect_retryable, connected_message, disconnected_message, hello_failed_message, peer_name,
    reconnect_with, retrying_message, sleep_reconnect, CONNECT_RETRY_ATTEMPTS, CONNECT_RETRY_DELAY,
    RECONNECT_SLEEP,
};
pub use relay::{
    forward_work_blocking, local_work_telemetry, now_sec, prepend_telemetry, usec_since, Forwarder,
};
pub use route::{
    decode_route_blob, encode_route_blob, validate_route_blob, ReturnTarget, RouteEntry,
};
pub use snapshot::{coordinator_load_snapshot, coordinator_save_snapshot, SnapshotMeta};
pub use transport::{read_frame, write_frame};
pub use work::{
    decode_logits_payload, decode_result_body, decode_work_body, encode_logits_payload,
    encode_result_body, encode_result_frame, encode_work_body, encode_work_frame,
    error_result_frame, ok_result_hdr, result_hash, result_request_id, work_with_ids, ResultBody,
    WorkBody,
};
pub use worker::{recv_hello, send_hello, Worker};
pub use worker_snapshot::{
    worker_handle_snapshot_load, worker_handle_snapshot_load_restore, worker_handle_snapshot_save,
    WorkerLoadOffer, WorkerSnapshotIdentity,
};
