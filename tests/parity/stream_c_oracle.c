/* C stream-projector oracle. Copied from ds4_server.c at v0.6.3-dfm so
 * Rust can compare the four tape projectors and buffered finals without
 * linking the server. Live DSML tool-call projection is not included. */

#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CREATED_TEST 1767225600L
#define TEST_RESP_ID "resp_aaaaaaaaaaaaaaaaaaaaaaaa"
#define TEST_RS_ID   "rs_aaaaaaaaaaaaaaaaaaaaaaaa"
#define TEST_MSG_ID  "msg_aaaaaaaaaaaaaaaaaaaaaaaa"

typedef struct { char *ptr; size_t len, cap; } buf;

static void die(const char *m) {
    fprintf(stderr, "stream_c_oracle: %s\n", m);
    exit(2);
}

static void buf_grow(buf *b, size_t n) {
    if (b->len + n + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + n + 1) cap *= 2;
    char *p = realloc(b->ptr, cap);
    if (!p) die("oom");
    b->ptr = p;
    b->cap = cap;
}

static void buf_putc(buf *b, char c) {
    buf_grow(b, 1);
    b->ptr[b->len++] = c;
    b->ptr[b->len] = 0;
}

static void buf_puts(buf *b, const char *s) {
    size_t n = strlen(s);
    buf_grow(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
}

static void buf_append(buf *b, const char *s, size_t n) {
    buf_grow(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
}

static void buf_printf(buf *b, const char *fmt, ...) {
    char tmp[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) die("printf");
    if ((size_t)n < sizeof(tmp)) { buf_puts(b, tmp); return; }
    char *big = malloc((size_t)n + 1);
    if (!big) die("oom");
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    buf_puts(b, big);
    free(big);
}

static void buf_free(buf *b) { free(b->ptr); memset(b, 0, sizeof(*b)); }

static char *xstrndup(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (!p) die("oom");
    memcpy(p, s ? s : "", n);
    p[n] = 0;
    return p;
}

static void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { buf_putc(b, '\\'); buf_putc(b, (char)c); }
        else if (c == '\n') buf_puts(b, "\\n");
        else if (c == '\r') buf_puts(b, "\\r");
        else if (c == '\t') buf_puts(b, "\\t");
        else if (c < 0x20) buf_printf(b, "\\u%04x", (unsigned)c);
        else buf_putc(b, (char)c);
    }
    buf_putc(b, '"');
}

static void json_escape_n(buf *b, const char *s, size_t n) {
    char *tmp = xstrndup(s ? s : "", n);
    json_escape(b, tmp);
    free(tmp);
}

static void json_escape_fragment_n(buf *b, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { buf_putc(b, '\\'); buf_putc(b, (char)c); }
        else if (c == '\n') buf_puts(b, "\\n");
        else if (c == '\r') buf_puts(b, "\\r");
        else if (c == '\t') buf_puts(b, "\\t");
        else if (c < 0x20) buf_printf(b, "\\u%04x", (unsigned)c);
        else buf_putc(b, (char)c);
    }
}

static int utf8_expected_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

static size_t utf8_stream_safe_len(const char *s, size_t start, size_t limit, bool final) {
    (void)final;
    if (!s || limit <= start) return limit;
    size_t p = limit;
    int cont = 0;
    while (p > start && cont < 4 && (((unsigned char)s[p - 1] & 0xc0) == 0x80)) {
        p--;
        cont++;
    }
    if (p == limit)
        return utf8_expected_len((unsigned char)s[limit - 1]) > 1 ? limit - 1 : limit;
    if (p == start && (((unsigned char)s[p] & 0xc0) == 0x80)) return start;
    size_t lead = p - 1;
    int need = utf8_expected_len((unsigned char)s[lead]);
    return (limit - lead) < (size_t)need ? lead : limit;
}

static char *utf8_trim_tail_dup(const char *s) {
    return s ? xstrndup(s, utf8_stream_safe_len(s, 0, strlen(s), true)) : NULL;
}

static size_t text_stream_safe_limit(const char *raw, size_t start, size_t raw_len,
                                    bool has_tools, bool final) {
    if (raw_len <= start) return raw_len;
    size_t limit = raw_len;
    if (has_tools && !final) {
        while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
        const size_t max_marker = 80;
        size_t scan = raw_len - start > max_marker ? raw_len - max_marker : start;
        for (size_t i = raw_len; i > scan; i--) {
            if (raw[i - 1] == '<') {
                if (i - 1 < limit) limit = i - 1;
                break;
            }
        }
    }
    return utf8_stream_safe_len(raw, start, limit, final);
}

enum { REQ_CHAT = 0, REQ_COMPLETION = 1 };
enum { API_OPENAI = 0, API_ANTHROPIC = 1, API_RESPONSES = 2 };
enum { DS4_THINK_NONE = 0, DS4_THINK_LOW = 1 };

typedef struct {
    int kind;
    int api;
    char model[64];
    int think_mode;
    bool has_tools;
    bool stream;
    bool stream_include_usage;
    bool reasoning_summary_emit;
    int cache_read_tokens;
    int cache_write_tokens;
} request;

static long g_created = CREATED_TEST;
static buf g_out;

static bool think_on(int m) { return m != DS4_THINK_NONE; }
static const char *think_start(void) { return "<think>"; }
static const char *think_end(void) { return "</think>"; }

static int clamp_usage(int value, int max) {
    if (value < 0) return 0;
    if (max >= 0 && value > max) return max;
    return value;
}

static void append_openai_usage(buf *b, const request *r, int prompt, int completion) {
    int cached = clamp_usage(r->cache_read_tokens, prompt);
    int write = clamp_usage(r->cache_write_tokens, prompt - cached);
    buf_printf(b,
        "{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
        "\"prompt_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d}}",
        prompt, completion, prompt + completion, cached, write);
}

static void append_anthropic_usage(buf *b, const request *r, int prompt, int completion) {
    int cr = clamp_usage(r->cache_read_tokens, prompt);
    int cw = clamp_usage(r->cache_write_tokens, prompt - cr);
    int input = prompt - cr - cw;
    if (input < 0) input = 0;
    buf_printf(b,
        "{\"input_tokens\":%d,\"output_tokens\":%d,"
        "\"cache_read_input_tokens\":%d,\"cache_creation_input_tokens\":%d}",
        input, completion, cr, cw);
}

static void append_responses_usage(buf *b, const request *r, int input, int output, int reasoning) {
    int cached = clamp_usage(r->cache_read_tokens, input);
    int write = clamp_usage(r->cache_write_tokens, input - cached);
    reasoning = clamp_usage(reasoning, output);
    buf_printf(b,
        "{\"input_tokens\":%d,\"input_tokens_details\":{\"cached_tokens\":%d,\"cache_write_tokens\":%d},"
        "\"output_tokens\":%d,\"output_tokens_details\":{\"reasoning_tokens\":%d},"
        "\"total_tokens\":%d}",
        input, cached, write, output, reasoning, input + output);
}

static void http_response(bool cors, int code, const char *type, const char *body) {
    const char *reason = code == 200 ? "OK" : "Error";
    size_t body_len = body ? strlen(body) : 0;
    buf_printf(&g_out, "HTTP/1.1 %d %s\r\nContent-Length: %zu\r\n", code, reason, body_len);
    if (type && type[0]) {
        buf_puts(&g_out, "Content-Type: ");
        buf_puts(&g_out, type);
        buf_puts(&g_out, "\r\n");
    }
    if (cors)
        buf_puts(&g_out, "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n");
    buf_puts(&g_out, "Connection: close\r\n\r\n");
    if (body_len) buf_append(&g_out, body, body_len);
}

static void sse_headers(bool cors) {
    buf_puts(&g_out, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n");
    if (cors)
        buf_puts(&g_out, "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n");
    buf_puts(&g_out, "Connection: close\r\n\r\n");
}

static void sse_event(const char *event, const char *data) {
    buf_puts(&g_out, "event: ");
    buf_puts(&g_out, event);
    buf_puts(&g_out, "\ndata: ");
    buf_puts(&g_out, data);
    buf_puts(&g_out, "\n\n");
}

static void sse_chunk(const request *r, const char *id, const char *text, const char *finish) {
    buf b = {0};
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, g_created);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":");
        if (text) { buf_puts(&b, "{\"content\":"); json_escape(&b, text); buf_putc(&b, '}'); }
        else buf_puts(&b, finish ? "{}" : "{\"role\":\"assistant\"}");
        buf_puts(&b, ",\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    } else {
        buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, g_created);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text ? text : "");
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        if (finish) json_escape(&b, finish); else buf_puts(&b, "null");
        buf_puts(&b, "}]}\n\n");
    }
    buf_append(&g_out, b.ptr, b.len);
    buf_free(&b);
}

static void sse_done(const request *r, const char *id, int prompt, int completion) {
    if (r->stream_include_usage) {
        buf b = {0};
        if (r->kind == REQ_CHAT)
            buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, g_created);
        else
            buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, g_created);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[],\"usage\":");
        append_openai_usage(&b, r, prompt, completion);
        buf_puts(&b, "}\n\n");
        buf_append(&g_out, b.ptr, b.len);
        buf_free(&b);
    }
    buf_puts(&g_out, "data: [DONE]\n\n");
}

static void sse_chat_delta_n(const request *r, const char *id, const char *field,
                             const char *text, size_t len) {
    if (len == 0) return;
    buf b = {0};
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, g_created);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{");
    json_escape(&b, field);
    buf_putc(&b, ':');
    json_escape_n(&b, text, len);
    buf_puts(&b, "},\"finish_reason\":null}]}\n\n");
    buf_append(&g_out, b.ptr, b.len);
    buf_free(&b);
}

enum { OA_THINKING, OA_TEXT, OA_SUPPRESS };
typedef struct {
    int mode;
    size_t emit_pos;
    bool active;
    bool checked_think_prefix;
} openai_stream;

static void openai_stream_start(const request *r, openai_stream *st) {
    memset(st, 0, sizeof(*st));
    st->active = true;
    st->mode = think_on(r->think_mode) ? OA_THINKING : OA_TEXT;
}

static bool openai_sse_stream_update(const request *r, const char *id, openai_stream *st,
                                     const char *raw, size_t raw_len, bool final) {
    if (!st->active || !raw) return true;
    if (st->mode == OA_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = think_start();
            size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) return true;
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) st->emit_pos = open_len;
            st->checked_think_prefix = true;
        }
        const char *close_s = think_end();
        const char *close = strstr(raw + st->emit_pos, close_s);
        size_t limit;
        if (close) limit = (size_t)(close - raw);
        else if (final) limit = utf8_stream_safe_len(raw, st->emit_pos, raw_len, true);
        else {
            size_t hold = strlen(close_s) - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }
        if (limit > st->emit_pos) {
            sse_chat_delta_n(r, id, "reasoning_content", raw + st->emit_pos, limit - st->emit_pos);
            st->emit_pos = limit;
        }
        if (close) { st->emit_pos = (size_t)(close - raw) + strlen(close_s); st->mode = OA_TEXT; }
        else if (final) { st->mode = OA_SUPPRESS; return true; }
        else return true;
    }
    if (st->mode == OA_TEXT) {
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len, r->has_tools, final);
        if (limit > st->emit_pos) {
            sse_chat_delta_n(r, id, "content", raw + st->emit_pos, limit - st->emit_pos);
            st->emit_pos = limit;
        }
        if (final) st->mode = OA_SUPPRESS;
    }
    return true;
}

static bool openai_sse_finish_live(const request *r, const char *id, openai_stream *st,
                                   const char *raw, size_t raw_len, const char *finish,
                                   int prompt, int completion) {
    if (!openai_sse_stream_update(r, id, st, raw, raw_len, true)) return false;
    buf b = {0};
    buf_printf(&b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", id, g_created);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&b, finish);
    buf_puts(&b, "}]}\n\n");
    buf_append(&g_out, b.ptr, b.len);
    buf_free(&b);
    sse_done(r, id, prompt, completion);
    return true;
}

enum { ANTH_THINKING, ANTH_TEXT, ANTH_SUPPRESS };
enum { ANTH_BLOCK_NONE, ANTH_BLOCK_THINKING, ANTH_BLOCK_TEXT };
typedef struct {
    int mode, open_block, next_index;
    size_t emit_pos;
    bool active, checked_think_prefix, sent_thinking, sent_text;
} anthropic_stream;

static bool anth_open(anthropic_stream *st, int type) {
    if (st->open_block == type) return true;
    if (st->open_block != ANTH_BLOCK_NONE) return false;
    buf b = {0};
    if (type == ANTH_BLOCK_THINKING)
        buf_printf(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}", st->next_index);
    else
        buf_printf(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}", st->next_index);
    sse_event("content_block_start", b.ptr);
    buf_free(&b);
    st->open_block = type;
    return true;
}

static void anth_delta(const anthropic_stream *st, int type, const char *text, size_t len) {
    if (len == 0) return;
    buf b = {0};
    if (type == ANTH_BLOCK_THINKING)
        buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":", st->next_index);
    else
        buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"text_delta\",\"text\":", st->next_index);
    json_escape_n(&b, text, len);
    buf_puts(&b, "}}");
    sse_event("content_block_delta", b.ptr);
    buf_free(&b);
}

static bool anth_close(const char *id, anthropic_stream *st) {
    if (st->open_block == ANTH_BLOCK_NONE) return true;
    buf b = {0};
    if (st->open_block == ANTH_BLOCK_THINKING) {
        buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"signature_delta\",\"signature\":", st->next_index);
        json_escape(&b, id);
        buf_puts(&b, "}}");
        sse_event("content_block_delta", b.ptr);
        buf_free(&b);
    }
    buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}", st->next_index);
    sse_event("content_block_stop", b.ptr);
    buf_free(&b);
    st->open_block = ANTH_BLOCK_NONE;
    st->next_index++;
    return true;
}

static void anth_start(const request *r, const char *id, int prompt, anthropic_stream *st) {
    buf b = {0};
    json_escape(&b, r->model);
    char *model = xstrndup(b.ptr, b.len);
    buf_free(&b);
    buf_printf(&b,
        "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\","
        "\"role\":\"assistant\",\"model\":%s,\"content\":[],\"stop_reason\":null,"
        "\"stop_sequence\":null,\"usage\":", id, model);
    append_anthropic_usage(&b, r, prompt, 0);
    buf_puts(&b, "}}");
    sse_event("message_start", b.ptr);
    buf_free(&b);
    free(model);
    memset(st, 0, sizeof(*st));
    st->active = true;
    st->mode = think_on(r->think_mode) ? ANTH_THINKING : ANTH_TEXT;
}

static bool anth_update(const request *r, const char *id, anthropic_stream *st,
                        const char *raw, size_t raw_len, bool final) {
    if (!st->active || !raw) return true;
    if (st->mode == ANTH_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = think_start();
            size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) return true;
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) st->emit_pos = open_len;
            st->checked_think_prefix = true;
        }
        const char *close_s = think_end();
        const char *close = strstr(raw + st->emit_pos, close_s);
        size_t limit;
        if (close) limit = (size_t)(close - raw);
        else if (final) limit = utf8_stream_safe_len(raw, st->emit_pos, raw_len, true);
        else {
            size_t hold = strlen(close_s) - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }
        if (limit > st->emit_pos) {
            if (!anth_open(st, ANTH_BLOCK_THINKING)) return false;
            anth_delta(st, ANTH_BLOCK_THINKING, raw + st->emit_pos, limit - st->emit_pos);
            st->sent_thinking = true;
            st->emit_pos = limit;
        }
        if (close || final) {
            if (!anth_close(id, st)) return false;
            if (close) { st->emit_pos = (size_t)(close - raw) + strlen(close_s); st->mode = ANTH_TEXT; }
            else { st->mode = ANTH_SUPPRESS; return true; }
        } else return true;
    }
    if (st->mode == ANTH_TEXT) {
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len, r->has_tools, final);
        if (limit > st->emit_pos) {
            if (!anth_open(st, ANTH_BLOCK_TEXT)) return false;
            anth_delta(st, ANTH_BLOCK_TEXT, raw + st->emit_pos, limit - st->emit_pos);
            st->sent_text = true;
            st->emit_pos = limit;
        }
        if (final) {
            if (!anth_close(id, st)) return false;
            st->mode = ANTH_SUPPRESS;
        }
    }
    return true;
}

static const char *anth_stop_reason(const char *finish, const char *matched) {
    if (finish && !strcmp(finish, "tool_calls")) return "tool_use";
    if (finish && !strcmp(finish, "length")) return "max_tokens";
    if (matched && matched[0] && finish && !strcmp(finish, "stop")) return "stop_sequence";
    return "end_turn";
}

static void append_anth_stop(buf *b, const char *finish, const char *matched) {
    const char *reason = anth_stop_reason(finish, matched);
    buf_puts(b, "\"stop_reason\":");
    json_escape(b, reason);
    buf_puts(b, ",\"stop_sequence\":");
    if (!strcmp(reason, "stop_sequence")) json_escape(b, matched);
    else buf_puts(b, "null");
}

static bool anth_finish(const request *r, const char *id, anthropic_stream *st,
                        const char *raw, size_t raw_len, const char *finish,
                        const char *matched, int completion) {
    if (!anth_update(r, id, st, raw, raw_len, true)) return false;
    if (st->sent_thinking && !st->sent_text) {
        if (!anth_open(st, ANTH_BLOCK_TEXT)) return false;
        if (!anth_close(id, st)) return false;
    }
    buf b = {0};
    buf_puts(&b, "{\"type\":\"message_delta\",\"delta\":{");
    append_anth_stop(&b, finish, matched);
    buf_printf(&b, "},\"usage\":{\"output_tokens\":%d}}", completion);
    sse_event("message_delta", b.ptr);
    buf_free(&b);
    sse_event("message_stop", "{\"type\":\"message_stop\"}");
    return true;
}

enum { RESP_THINKING, RESP_TEXT, RESP_SUPPRESS };
typedef struct {
    int mode;
    size_t emit_pos;
    bool active, checked_think_prefix;
    bool reasoning_item_opened, reasoning_item_closed, reasoning_summary_started;
    bool reasoning_closed_naturally, message_item_opened, message_text_part_open;
    bool message_item_closed, reasoning_emitted_any, message_emitted_any;
    size_t reasoning_start, reasoning_end, message_start, message_end;
    size_t message_tail_start, message_tail_end;
    char response_id[40], reasoning_id[40], message_id[40];
    int reasoning_index, message_index, next_output_index, sequence;
    long created_at;
} responses_stream;

static void resp_init(const request *r, responses_stream *st) {
    memset(st, 0, sizeof(*st));
    st->mode = think_on(r->think_mode) ? RESP_THINKING : RESP_TEXT;
    snprintf(st->response_id, sizeof(st->response_id), "%s", TEST_RESP_ID);
    snprintf(st->reasoning_id, sizeof(st->reasoning_id), "%s", TEST_RS_ID);
    snprintf(st->message_id, sizeof(st->message_id), "%s", TEST_MSG_ID);
    st->reasoning_index = -1;
    st->message_index = -1;
}

static void resp_emit(responses_stream *st, const char *body) {
    buf b = {0};
    buf_puts(&b, "data: ");
    const char *type_close = NULL;
    if (body[0] == '{') {
        const char *p = body + 1;
        if (!strncmp(p, "\"type\":\"", 8)) {
            const char *q = p + 8;
            while (*q && *q != '"') {
                if (*q == '\\' && q[1]) q += 2;
                else q++;
            }
            if (*q == '"') type_close = q + 1;
        }
    }
    if (type_close) {
        buf_append(&b, body, (size_t)(type_close - body));
        buf_printf(&b, ",\"sequence_number\":%d", st->sequence++);
        buf_puts(&b, type_close);
    } else buf_puts(&b, body);
    buf_puts(&b, "\n\n");
    buf_append(&g_out, b.ptr, b.len);
    buf_free(&b);
}

static void resp_created(const request *r, responses_stream *st, long created_at) {
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.created\",\"response\":{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,\"status\":\"in_progress\",\"model\":",
               st->response_id, created_at);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"output\":[]}}");
    resp_emit(st, b.ptr);
    buf_free(&b);
    st->created_at = created_at;
}

static void resp_reason_added(responses_stream *st) {
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.output_item.added\",\"output_index\":%d,\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"in_progress\",\"summary\":[]}}",
               st->reasoning_index, st->reasoning_id);
    resp_emit(st, b.ptr);
    buf_free(&b);
}

static void resp_reason_part(responses_stream *st) {
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.reasoning_summary_part.added\",\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"\"}}",
               st->reasoning_id, st->reasoning_index);
    resp_emit(st, b.ptr);
    buf_free(&b);
}

static void resp_reason_delta(responses_stream *st, const char *text, size_t len) {
    if (!len) return;
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"delta\":",
               st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, text, len);
    buf_putc(&b, '}');
    resp_emit(st, b.ptr);
    buf_free(&b);
}

static const char *resp_item_status(const char *finish) {
    return (finish && (!strcmp(finish, "length") || !strcmp(finish, "error"))) ? "incomplete" : "completed";
}

static const char *resp_status(const char *finish) {
    if (finish && !strcmp(finish, "length")) return "incomplete";
    if (finish && !strcmp(finish, "error")) return "failed";
    return "completed";
}

static void resp_msg_text_esc(buf *b, const responses_stream *st, const char *raw) {
    buf_putc(b, '"');
    if (raw) {
        if (st->message_end > st->message_start)
            json_escape_fragment_n(b, raw + st->message_start, st->message_end - st->message_start);
        if (st->message_tail_end > st->message_tail_start)
            json_escape_fragment_n(b, raw + st->message_tail_start, st->message_tail_end - st->message_tail_start);
    }
    buf_putc(b, '"');
}

static bool resp_reason_done(responses_stream *st, const char *raw) {
    const char *item_status = st->reasoning_closed_naturally ? "completed" : "incomplete";
    const char *rtext = raw ? raw + st->reasoning_start : "";
    size_t rlen = st->reasoning_end > st->reasoning_start ? st->reasoning_end - st->reasoning_start : 0;
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.reasoning_summary_text.done\",\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"text\":",
               st->reasoning_id, st->reasoning_index);
    json_escape_n(&b, rtext, rlen);
    buf_putc(&b, '}');
    resp_emit(st, b.ptr);
    if (st->reasoning_summary_started) {
        buf_free(&b);
        buf_printf(&b, "{\"type\":\"response.reasoning_summary_part.done\",\"item_id\":\"%s\",\"output_index\":%d,\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":",
                   st->reasoning_id, st->reasoning_index);
        json_escape_n(&b, rtext, rlen);
        buf_puts(&b, "}}");
        resp_emit(st, b.ptr);
    }
    buf_free(&b);
    buf_printf(&b, "{\"type\":\"response.output_item.done\",\"output_index\":%d,\"item\":{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[",
               st->reasoning_index, st->reasoning_id, item_status);
    if (rlen) {
        buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
        json_escape_n(&b, rtext, rlen);
        buf_putc(&b, '}');
    }
    buf_puts(&b, "]}}");
    resp_emit(st, b.ptr);
    buf_free(&b);
    return true;
}

static void resp_msg_added(responses_stream *st) {
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.output_item.added\",\"output_index\":%d,\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}}",
               st->message_index, st->message_id);
    resp_emit(st, b.ptr);
    buf_free(&b);
}

static void resp_msg_part(responses_stream *st) {
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.content_part.added\",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}",
               st->message_id, st->message_index);
    resp_emit(st, b.ptr);
    buf_free(&b);
}

static void resp_text_delta(responses_stream *st, const char *text, size_t len) {
    if (!len) return;
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.output_text.delta\",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"delta\":",
               st->message_id, st->message_index);
    json_escape_n(&b, text, len);
    buf_putc(&b, '}');
    resp_emit(st, b.ptr);
    buf_free(&b);
}

static bool resp_msg_done(responses_stream *st, const char *raw, const char *finish) {
    const char *item_status = resp_item_status(finish);
    buf b = {0};
    buf_printf(&b, "{\"type\":\"response.output_text.done\",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"text\":",
               st->message_id, st->message_index);
    resp_msg_text_esc(&b, st, raw);
    buf_putc(&b, '}');
    resp_emit(st, b.ptr);
    buf_free(&b);
    buf_printf(&b, "{\"type\":\"response.content_part.done\",\"item_id\":\"%s\",\"output_index\":%d,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":",
               st->message_id, st->message_index);
    resp_msg_text_esc(&b, st, raw);
    buf_puts(&b, ",\"annotations\":[]}}");
    resp_emit(st, b.ptr);
    buf_free(&b);
    buf_printf(&b, "{\"type\":\"response.output_item.done\",\"output_index\":%d,\"item\":{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
               st->message_index, st->message_id, item_status);
    resp_msg_text_esc(&b, st, raw);
    buf_puts(&b, ",\"annotations\":[]}]}}");
    resp_emit(st, b.ptr);
    buf_free(&b);
    return true;
}

static void resp_completed(const request *r, responses_stream *st, const char *raw,
                           const char *finish, int prompt, int completion, int reasoning, long created_at) {
    const char *event_type = "response.completed";
    if (finish && !strcmp(finish, "error")) event_type = "response.failed";
    else if (finish && !strcmp(finish, "length")) event_type = "response.incomplete";
    const char *status = resp_status(finish);
    const char *item_status = resp_item_status(finish);
    buf b = {0};
    buf_printf(&b, "{\"type\":\"%s\",\"response\":{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\",\"model\":",
               event_type, st->response_id, created_at, status);
    json_escape(&b, r->model);
    if (!strcmp(event_type, "response.failed"))
        buf_puts(&b, ",\"error\":{\"code\":\"server_error\",\"message\":\"generation failed\"}");
    else if (!strcmp(event_type, "response.incomplete"))
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    buf_puts(&b, ",\"output\":[");
    bool wrote = false;
    if (st->reasoning_emitted_any) {
        const char *rs = st->reasoning_closed_naturally ? "completed" : "incomplete";
        size_t rlen = st->reasoning_end > st->reasoning_start ? st->reasoning_end - st->reasoning_start : 0;
        buf_printf(&b, "{\"id\":\"%s\",\"type\":\"reasoning\",\"status\":\"%s\",\"summary\":[", st->reasoning_id, rs);
        if (rlen && raw) {
            buf_puts(&b, "{\"type\":\"summary_text\",\"text\":");
            json_escape_n(&b, raw + st->reasoning_start, rlen);
            buf_putc(&b, '}');
        }
        buf_puts(&b, "]}");
        wrote = true;
    }
    if (st->message_emitted_any) {
        if (wrote) buf_putc(&b, ',');
        buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
                   st->message_id, item_status);
        resp_msg_text_esc(&b, st, raw);
        buf_puts(&b, ",\"annotations\":[]}]}");
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage(&b, r, prompt, completion, reasoning);
    buf_puts(&b, "}}");
    resp_emit(st, b.ptr);
    buf_free(&b);
}

static bool resp_update(const request *r, responses_stream *st, const char *raw, size_t raw_len, bool final) {
    if (!st->active || !raw) return true;
    bool emit_reasoning = r->reasoning_summary_emit;
    if (st->mode == RESP_THINKING) {
        if (!st->checked_think_prefix) {
            const char *open = think_start();
            size_t open_len = strlen(open);
            if (raw_len < open_len && !strncmp(raw, open, raw_len) && !final) return true;
            if (raw_len >= open_len && !strncmp(raw, open, open_len)) st->emit_pos = open_len;
            st->checked_think_prefix = true;
        }
        const char *close_s = think_end();
        const char *close = strstr(raw + st->emit_pos, close_s);
        size_t limit;
        if (close) limit = (size_t)(close - raw);
        else if (final) limit = utf8_stream_safe_len(raw, st->emit_pos, raw_len, true);
        else {
            size_t hold = strlen(close_s) - 1;
            limit = raw_len > hold ? raw_len - hold : st->emit_pos;
            limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false);
        }
        if (limit > st->emit_pos) {
            if (emit_reasoning) {
                if (!st->reasoning_item_opened) {
                    st->reasoning_index = st->next_output_index++;
                    resp_reason_added(st);
                    st->reasoning_item_opened = true;
                }
                if (!st->reasoning_summary_started) {
                    resp_reason_part(st);
                    st->reasoning_summary_started = true;
                }
                resp_reason_delta(st, raw + st->emit_pos, limit - st->emit_pos);
                if (!st->reasoning_emitted_any) st->reasoning_start = st->emit_pos;
                st->reasoning_end = limit;
                st->reasoning_emitted_any = true;
            }
            st->emit_pos = limit;
        }
        if (close) {
            st->emit_pos = (size_t)(close - raw) + strlen(close_s);
            st->mode = RESP_TEXT;
            st->reasoning_closed_naturally = true;
        } else if (final) { st->mode = RESP_SUPPRESS; return true; }
        else return true;
    }
    if (st->mode == RESP_TEXT) {
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, raw_len, r->has_tools, final);
        if (limit > st->emit_pos) {
            if (!st->message_item_opened) {
                st->message_index = st->next_output_index++;
                resp_msg_added(st);
                st->message_item_opened = true;
            }
            if (!st->message_text_part_open) {
                resp_msg_part(st);
                st->message_text_part_open = true;
            }
            resp_text_delta(st, raw + st->emit_pos, limit - st->emit_pos);
            if (!st->message_emitted_any) st->message_start = st->emit_pos;
            st->message_end = limit;
            st->message_emitted_any = true;
            st->emit_pos = limit;
        }
        if (final) st->mode = RESP_SUPPRESS;
    }
    return true;
}

static bool resp_finish(const request *r, responses_stream *st, const char *raw, size_t raw_len,
                        const char *finish, int prompt, int completion, int reasoning, long created_at) {
    if (!resp_update(r, st, raw, raw_len, true)) return false;
    if (st->reasoning_end > raw_len) st->reasoning_end = raw_len;
    if (st->reasoning_start > st->reasoning_end) st->reasoning_start = st->reasoning_end;
    if (st->message_end > raw_len) st->message_end = raw_len;
    if (st->message_start > st->message_end) st->message_start = st->message_end;
    if (st->reasoning_item_opened && !st->reasoning_item_closed) {
        if (!resp_reason_done(st, raw)) return false;
        st->reasoning_item_closed = true;
    }
    if (st->message_item_opened && !st->message_item_closed) {
        if (!resp_msg_done(st, raw, finish)) return false;
        st->message_item_closed = true;
    }
    resp_completed(r, st, raw, finish, prompt, completion, reasoning, created_at);
    return true;
}

static void append_anth_content(buf *b, const char *text, const char *reasoning) {
    buf_putc(b, '[');
    bool wrote = false, after = false;
    if (reasoning && reasoning[0]) {
        buf_puts(b, "{\"type\":\"thinking\",\"thinking\":");
        json_escape(b, reasoning);
        buf_puts(b, ",\"signature\":\"\"}");
        wrote = true;
    }
    if (text && text[0]) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":");
        json_escape(b, text);
        buf_putc(b, '}');
        wrote = true;
        after = true;
    }
    if (!wrote || ((reasoning && reasoning[0]) && !after)) {
        if (wrote) buf_putc(b, ',');
        buf_puts(b, "{\"type\":\"text\",\"text\":\"\"}");
    }
    buf_putc(b, ']');
}

static void final_openai(const request *r, const char *id, const char *text, const char *finish,
                         int prompt, int completion) {
    char *text_t = utf8_trim_tail_dup(text);
    text = text_t;
    buf b = {0};
    if (r->kind == REQ_CHAT) {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", id, g_created);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        json_escape(&b, text ? text : "");
        buf_puts(&b, "},\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    } else {
        buf_printf(&b, "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":", id, g_created);
        json_escape(&b, r->model);
        buf_puts(&b, ",\"choices\":[{\"text\":");
        json_escape(&b, text);
        buf_puts(&b, ",\"index\":0,\"finish_reason\":");
        json_escape(&b, finish);
        buf_puts(&b, "}],\"usage\":");
    }
    append_openai_usage(&b, r, prompt, completion);
    buf_puts(&b, "}\n");
    http_response(false, 200, "application/json", b.ptr);
    buf_free(&b);
    free(text_t);
}

static void final_anthropic(const request *r, const char *id, const char *text, const char *finish,
                            int prompt, int completion) {
    char *text_t = utf8_trim_tail_dup(text);
    text = text_t;
    buf b = {0};
    buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    json_escape(&b, r->model);
    buf_puts(&b, ",\"content\":");
    append_anth_content(&b, text, NULL);
    buf_putc(&b, ',');
    append_anth_stop(&b, finish, NULL);
    buf_puts(&b, ",\"usage\":");
    append_anthropic_usage(&b, r, prompt, completion);
    buf_puts(&b, "}\n");
    http_response(false, 200, "application/json", b.ptr);
    buf_free(&b);
    free(text_t);
}

static void final_responses(const request *r, const char *text, const char *finish,
                            int prompt, int completion, int reasoning) {
    char *text_t = utf8_trim_tail_dup(text);
    text = text_t;
    const char *status = resp_status(finish);
    const char *item_status = resp_item_status(finish);
    buf b = {0};
    buf_printf(&b, "{\"id\":\"%s\",\"object\":\"response\",\"created_at\":%ld,\"status\":\"%s\",\"model\":",
               TEST_RESP_ID, g_created, status);
    json_escape(&b, r->model);
    if (finish && !strcmp(finish, "length"))
        buf_puts(&b, ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    buf_puts(&b, ",\"output\":[");
    if (text && text[0]) {
        buf_printf(&b, "{\"id\":\"%s\",\"type\":\"message\",\"status\":\"%s\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":",
                   TEST_MSG_ID, item_status);
        json_escape(&b, text);
        buf_puts(&b, ",\"annotations\":[]}]}");
    }
    buf_putc(&b, ']');
    buf_puts(&b, ",\"usage\":");
    append_responses_usage(&b, r, prompt, completion, reasoning);
    buf_putc(&b, '}');
    http_response(false, 200, "application/json", b.ptr);
    buf_free(&b);
    free(text_t);
}

static const char *tape_plain[] = { "Hel", "lo", " wor", "ld." };
static const char *tape_thinking[] = { "plan", "</think>", "Answer", " done." };
static const char tape_utf8_0[] = "caf";
static const char tape_utf8_1[] = { (char)0xC3, 0 };
static const char tape_utf8_2[] = { (char)0xA9, 0 };
static const char tape_utf8_3[] = " ok";
static const char *tape_utf8[] = { tape_utf8_0, tape_utf8_1, tape_utf8_2, tape_utf8_3 };

static void req_init(request *r, int kind) {
    memset(r, 0, sizeof(*r));
    r->kind = kind;
    r->api = API_OPENAI;
    snprintf(r->model, sizeof(r->model), "deepseek-v4-flash");
    r->think_mode = DS4_THINK_LOW;
    r->stream = true;
}

static void run_openai_chat(void) {
    request r; req_init(&r, REQ_CHAT);
    openai_stream st;
    openai_stream_start(&r, &st);
    sse_headers(false);
    sse_chunk(&r, "chatcmpl_tape", NULL, NULL);
    buf raw = {0};
    for (size_t i = 0; i < 4; i++) {
        buf_puts(&raw, tape_thinking[i]);
        openai_sse_stream_update(&r, "chatcmpl_tape", &st, raw.ptr, raw.len, false);
    }
    openai_sse_finish_live(&r, "chatcmpl_tape", &st, raw.ptr, raw.len, "stop", 7, 4);
    buf_free(&raw);
}

static void run_openai_utf8(void) {
    request r; req_init(&r, REQ_CHAT);
    r.think_mode = DS4_THINK_NONE;
    openai_stream st;
    openai_stream_start(&r, &st);
    sse_headers(false);
    buf raw = {0};
    for (size_t i = 0; i < 4; i++) {
        buf_puts(&raw, tape_utf8[i]);
        openai_sse_stream_update(&r, "chatcmpl_tape8", &st, raw.ptr, raw.len, false);
    }
    openai_sse_finish_live(&r, "chatcmpl_tape8", &st, raw.ptr, raw.len, "stop", 4, 4);
    buf_free(&raw);
}

static void run_openai_completion(void) {
    request r; req_init(&r, REQ_COMPLETION);
    sse_headers(false);
    for (size_t i = 0; i < 4; i++) sse_chunk(&r, "cmpl_tape", tape_plain[i], NULL);
    sse_chunk(&r, "cmpl_tape", NULL, "stop");
    sse_done(&r, "cmpl_tape", 4, 4);
}

static void run_anthropic(void) {
    request r; req_init(&r, REQ_CHAT);
    r.api = API_ANTHROPIC;
    anthropic_stream st;
    anth_start(&r, "msg_tape", 7, &st);
    buf raw = {0};
    for (size_t i = 0; i < 4; i++) {
        buf_puts(&raw, tape_thinking[i]);
        anth_update(&r, "msg_tape", &st, raw.ptr, raw.len, false);
    }
    anth_finish(&r, "msg_tape", &st, raw.ptr, raw.len, "stop", NULL, 4);
    buf_free(&raw);
}

static void run_responses(void) {
    request r; req_init(&r, REQ_CHAT);
    r.api = API_RESPONSES;
    r.reasoning_summary_emit = true;
    responses_stream st;
    resp_init(&r, &st);
    st.active = true;
    resp_created(&r, &st, g_created);
    buf raw = {0};
    for (size_t i = 0; i < 4; i++) {
        buf_puts(&raw, tape_thinking[i]);
        resp_update(&r, &st, raw.ptr, raw.len, false);
    }
    resp_finish(&r, &st, raw.ptr, raw.len, "stop", 7, 4, 1, g_created);
    buf_free(&raw);
}

static void run_final(const char *surf, const char *text, const char *finish) {
    request r; req_init(&r, REQ_CHAT);
    r.stream = false;
    if (!strcmp(surf, "openai-chat")) final_openai(&r, "chatcmpl_buf", text, finish, 4, 4);
    else if (!strcmp(surf, "openai-completion")) {
        r.kind = REQ_COMPLETION;
        final_openai(&r, "cmpl_buf", text, finish, 4, 4);
    } else if (!strcmp(surf, "anthropic")) {
        r.api = API_ANTHROPIC;
        final_anthropic(&r, "msg_buf", text, finish, 4, 4);
    } else if (!strcmp(surf, "responses")) {
        r.api = API_RESPONSES;
        final_responses(&r, text, finish, 4, 4, 0);
    } else die("unknown final surface");
}

static void run_utf8_safe(size_t start, size_t limit, int final, const char *hex) {
    size_t n = strlen(hex) / 2;
    char *s = malloc(n + 1);
    if (!s) die("oom");
    for (size_t i = 0; i < n; i++) {
        unsigned v = 0;
        sscanf(hex + 2 * i, "%2x", &v);
        s[i] = (char)v;
    }
    s[n] = 0;
    printf("%zu\n", utf8_stream_safe_len(s, start, limit, final != 0));
    free(s);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: stream_c_oracle <cmd> [...]\n");
        return 2;
    }
    const char *env = getenv("DS4_TEST_CREATED");
    if (env && env[0]) g_created = strtol(env, NULL, 10);
    if (!strcmp(argv[1], "utf8-safe")) {
        if (argc < 6) die("utf8-safe START LIMIT FINAL HEX");
        run_utf8_safe((size_t)atoi(argv[2]), (size_t)atoi(argv[3]), atoi(argv[4]), argv[5]);
        return 0;
    }
    if (!strcmp(argv[1], "openai-chat-tape")) run_openai_chat();
    else if (!strcmp(argv[1], "openai-chat-utf8")) run_openai_utf8();
    else if (!strcmp(argv[1], "openai-completion-tape")) run_openai_completion();
    else if (!strcmp(argv[1], "anthropic-tape")) run_anthropic();
    else if (!strcmp(argv[1], "responses-tape")) run_responses();
    else if (!strcmp(argv[1], "final")) {
        if (argc < 5) die("final SURFACE TEXT FINISH");
        run_final(argv[2], argv[3], argv[4]);
    } else die("unknown cmd");
    fwrite(g_out.ptr ? g_out.ptr : "", 1, g_out.len, stdout);
    buf_free(&g_out);
    return 0;
}
