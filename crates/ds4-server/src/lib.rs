//! Server host. Phase 7 ports surfaces by feature; no API redesign.

pub mod error;
pub mod format;
pub mod http;
pub mod json;
pub mod models;
pub mod route;
pub mod serve;

pub use error::{
    anthropic_error_body, anthropic_error_type, http_head, http_reason, http_response_bytes,
    openai_error_body, openai_error_type, retry_after_header, wire_error_body, wire_http_error_bytes,
};
pub use format::{
    output_format_type_supported, parse_output_config_effort, parse_output_config_format,
    parse_output_format_value, parse_reasoning_effort_value, parse_responses_text_value,
};
pub use http::{
    chunked_enabled, content_length, header_accepts_json, header_chunked, header_end,
    parse_surface_for_path, read_http_request, shed_surface_for_path, HttpRequest,
};
pub use json::{json_escape, json_skip_value, json_string, Json};
pub use models::{
    append_model_json_values, json_models_array_dup, model_alias_known, model_id_from_gguf_path,
    model_id_known, model_one_json, models_list_json,
};
pub use route::{
    compute_needs, decode_budget, parse_reasoning_effort_name, route_decide, think_mode_enabled,
    think_mode_from_enabled, wire_surface_for, Api, NeedInput, ReqKind, RouteDecision, RouteEnv,
    ThinkMode, WireSurface, LANE_CONTINUOUS, LANE_NONE, LANE_SERIAL, LANE_STATIC,
    NEED_BANK_FRONTIER, NEED_CONTINUATION_PUBLISH, NEED_CORRECTIVE_RECOVERY, NEED_DURABLE_RESPONSE,
    NEED_LIVE_FRONTIER, NEED_PER_ROW_SAMPLING, NEED_PREFILL_ONLY, NEED_STOP_SCAN, NEED_STREAMING,
    NEED_THINKING, NEED_TOKEN_IDS, NEED_TOOL_SCAN, REASON_COALESCE_OFF, REASON_CONT,
    REASON_CONT_BANK, REASON_CONT_UNAVAILABLE, REASON_NAMES, REASON_NEED_CONTINUATION_PUBLISH,
    REASON_NEED_CORRECTIVE_RECOVERY, REASON_NEED_DURABLE, REASON_NEED_LIVE_FRONTIER,
    REASON_NEED_PREFILL_ONLY, REASON_STATIC_NO_CONT, REASON_STATIC_PROMPT_BOUNDS, REASON_SURFACE,
    REASON_TOKEN_IDS_PROJECTION, REASON_TOOLS_COMPLETION,
};
pub use serve::{accept_loop, handle_client, listen, ServerConfig};
