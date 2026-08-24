/* Incremental DSML tool-stream oracle. Copied from ds4_server.c at v0.6.3-dfm.
 * IDs are deterministic: call_N / toolu_N. Created timestamp is CREATED_TEST. */

#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CREATED_TEST 1767225600L
#define DS4_DSML "｜DSML｜"
#define DS4_DSML_SHORT "DSML｜"
#define DS4_TOOL_CALLS_START "<" DS4_DSML "tool_calls>"
#define DS4_TOOL_CALLS_END "</" DS4_DSML "tool_calls>"
#define DS4_INVOKE_START "<" DS4_DSML "invoke"
#define DS4_INVOKE_END "</" DS4_DSML "invoke>"
#define DS4_PARAM_START "<" DS4_DSML "parameter"
#define DS4_PARAM_END "</" DS4_DSML "parameter>"
#define DS4_TOOL_CALLS_START_SHORT "<" DS4_DSML_SHORT "tool_calls>"
#define DS4_TOOL_CALLS_END_SHORT "</" DS4_DSML_SHORT "tool_calls>"
#define DS4_INVOKE_START_SHORT "<" DS4_DSML_SHORT "invoke"
#define DS4_INVOKE_END_SHORT "</" DS4_DSML_SHORT "invoke>"
#define DS4_PARAM_START_SHORT "<" DS4_DSML_SHORT "parameter"
#define DS4_PARAM_END_SHORT "</" DS4_DSML_SHORT "parameter>"

typedef struct { char *ptr; size_t len, cap; } buf;
static buf g_out;
static const char *g_model = "deepseek-v4-flash";

static void die(const char *m) { fprintf(stderr, "tool_stream_c_oracle: %s\n", m); exit(2); }
static void buf_grow(buf *b, size_t n) {
    if (b->len + n + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + n + 1) cap *= 2;
    char *p = realloc(b->ptr, cap);
    if (!p) die("oom");
    b->ptr = p; b->cap = cap;
}
static void buf_putc(buf *b, char c) { buf_grow(b, 1); b->ptr[b->len++] = c; b->ptr[b->len] = 0; }
static void buf_puts(buf *b, const char *s) {
    size_t n = strlen(s); buf_grow(b, n); memcpy(b->ptr + b->len, s, n); b->len += n; b->ptr[b->len] = 0;
}
static void buf_printf(buf *b, const char *fmt, ...) {
    char tmp[2048]; va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap); va_end(ap);
    if (n < 0) die("printf");
    if ((size_t)n < sizeof(tmp)) { buf_puts(b, tmp); return; }
    char *big = malloc((size_t)n + 1); if (!big) die("oom");
    va_start(ap, fmt); vsnprintf(big, (size_t)n + 1, fmt, ap); va_end(ap);
    buf_puts(b, big); free(big);
}
static char *xstrndup(const char *s, size_t n) {
    char *p = malloc(n + 1); if (!p) die("oom"); memcpy(p, s ? s : "", n); p[n] = 0; return p;
}
static void json_escape_n(buf *b, const char *s, size_t n) {
    buf_putc(b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { buf_putc(b, '\\'); buf_putc(b, (char)c); }
        else if (c == '\n') buf_puts(b, "\\n");
        else if (c == '\r') buf_puts(b, "\\r");
        else if (c == '\t') buf_puts(b, "\\t");
        else if (c < 0x20) buf_printf(b, "\\u%04x", (unsigned)c);
        else buf_putc(b, (char)c);
    }
    buf_putc(b, '"');
}
static void json_escape(buf *b, const char *s) { json_escape_n(b, s, strlen(s)); }
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
    size_t p = limit; int cont = 0;
    while (p > start && cont < 4 && (((unsigned char)s[p - 1] & 0xc0) == 0x80)) { p--; cont++; }
    if (p == limit) return utf8_expected_len((unsigned char)s[limit - 1]) > 1 ? limit - 1 : limit;
    if (p == start && (((unsigned char)s[p] & 0xc0) == 0x80)) return start;
    size_t lead = p - 1;
    int need = utf8_expected_len((unsigned char)s[lead]);
    return (limit - lead) < (size_t)need ? lead : limit;
}
static const char *find_tool_start(const char *s);
static size_t trim_tool_separator_ws(const char *raw, size_t start, size_t limit) {
    while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
    return limit;
}
static size_t text_stream_safe_limit(const char *raw, size_t start, size_t raw_len, bool has_tools, bool final) {
    if (raw_len <= start) return raw_len;
    size_t limit = raw_len;
    if (has_tools) {
        const char *tool = find_tool_start(raw + start);
        if (tool) {
            limit = trim_tool_separator_ws(raw, start, (size_t)(tool - raw));
            return utf8_stream_safe_len(raw, start, limit, true);
        }
        if (!final) {
            while (limit > start && isspace((unsigned char)raw[limit - 1])) limit--;
            size_t max_marker = 80;
            size_t scan = raw_len - start > max_marker ? raw_len - max_marker : start;
            for (size_t i = raw_len; i > scan; i--) {
                if (raw[i - 1] == '<') { if (i - 1 < limit) limit = i - 1; break; }
            }
            limit = trim_tool_separator_ws(raw, start, limit);
        }
    }
    return utf8_stream_safe_len(raw, start, limit, final);
}
static const char *find_lit(const char *s, const char *lit) { return s ? strstr(s, lit) : NULL; }
static const char *find_tool_start(const char *s) {
    const char *cands[] = { find_lit(s, DS4_TOOL_CALLS_START), find_lit(s, DS4_TOOL_CALLS_START_SHORT), find_lit(s, "<tool_calls>"), find_lit(s, "<tool_call>"), find_lit(s, "<dots_function_call>") };
    const char *best = NULL;
    for (size_t i = 0; i < 5; i++) if (cands[i] && (!best || cands[i] < best)) best = cands[i];
    return best;
}
static bool raw_full_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    return pos <= raw_len && raw_len - pos >= n && !memcmp(raw + pos, lit, n);
}
static bool raw_partial_lit(const char *raw, size_t raw_len, size_t pos, const char *lit) {
    size_t n = strlen(lit);
    if (pos > raw_len || raw_len - pos >= n) return false;
    return !memcmp(raw + pos, lit, raw_len - pos);
}
static bool raw_partial_any(const char *raw, size_t raw_len, size_t pos, const char *a, const char *b) {
    return raw_partial_lit(raw, raw_len, pos, a) || raw_partial_lit(raw, raw_len, pos, b);
}
static const char *find_lit_bounded(const char *s, size_t n, const char *lit) {
    size_t m = strlen(lit);
    if (m == 0) return s;
    if (n < m) return NULL;
    for (size_t i = 0; i <= n - m; i++) if (!memcmp(s + i, lit, m)) return s + i;
    return NULL;
}
static char *dsml_unescape(const char *s) {
    size_t n = strlen(s); char *out = malloc(n + 1); size_t o = 0;
    for (size_t i = 0; i < n; ) {
        if (s[i] != '&') { out[o++] = s[i++]; continue; }
        if (!strncmp(s + i, "&amp;", 5)) { out[o++] = '&'; i += 5; }
        else if (!strncmp(s + i, "&lt;", 4)) { out[o++] = '<'; i += 4; }
        else if (!strncmp(s + i, "&gt;", 4)) { out[o++] = '>'; i += 4; }
        else if (!strncmp(s + i, "&quot;", 6)) { out[o++] = '"'; i += 6; }
        else if (!strncmp(s + i, "&apos;", 6)) { out[o++] = '\''; i += 6; }
        else out[o++] = s[i++];
    }
    out[o] = 0; return out;
}
static char *dsml_attr(const char *tag, const char *name) {
    char pat[64]; snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p) return NULL;
    p += strlen(pat);
    const char *e = strchr(p, '"');
    if (!e) return NULL;
    char *raw = xstrndup(p, (size_t)(e - p));
    char *u = dsml_unescape(raw);
    free(raw);
    return u;
}
static size_t dsml_entity_stream_safe_len(const char *raw, size_t start, size_t limit) {
    static const char *ents[] = {"&amp;", "&lt;", "&gt;", "&quot;", "&apos;"};
    size_t scan = limit > start + 6 ? limit - 6 : start;
    for (size_t i = limit; i > scan; i--) {
        if (raw[i - 1] != '&') continue;
        size_t amp = i - 1, tail = limit - amp;
        for (size_t ei = 0; ei < 5; ei++) {
            size_t elen = strlen(ents[ei]);
            if (tail < elen && !memcmp(raw + amp, ents[ei], tail)) return amp;
        }
        break;
    }
    return limit;
}
static size_t tool_param_value_stream_safe_len(const char *raw, size_t start, size_t raw_len, const char *param_end, bool is_string) {
    size_t limit = raw_len, end_len = strlen(param_end);
    size_t scan = raw_len > start + end_len ? raw_len - end_len : start;
    for (size_t i = raw_len; i > scan; i--) {
        if (raw[i - 1] != '<') continue;
        size_t marker = i - 1, tail = raw_len - marker;
        if (tail < end_len && !memcmp(raw + marker, param_end, tail)) limit = marker;
        break;
    }
    if (is_string) limit = dsml_entity_stream_safe_len(raw, start, limit);
    return utf8_stream_safe_len(raw, start, limit, false);
}

typedef enum { ST_INV, ST_PAR, ST_VAL, ST_DONE, ST_ERR } tstate;
typedef struct {
    tstate state; const char *tc_end, *inv_s, *inv_e, *par_s, *par_e;
    size_t parse_pos; int index; bool active, emitted_any, args_open, first_param, param_is_string;
    const char *id_prefix;
} tool_st;

static bool tool_init(tool_st *ts, const char *raw, size_t n, size_t pos, const char *prefix) {
    memset(ts, 0, sizeof(*ts));
    ts->active = true; ts->state = ST_INV; ts->id_prefix = prefix;
    if (raw_full_lit(raw, n, pos, DS4_TOOL_CALLS_START)) {
        ts->parse_pos = pos + strlen(DS4_TOOL_CALLS_START);
        ts->tc_end = DS4_TOOL_CALLS_END; ts->inv_s = DS4_INVOKE_START; ts->inv_e = DS4_INVOKE_END;
        ts->par_s = DS4_PARAM_START; ts->par_e = DS4_PARAM_END; return true;
    }
    if (raw_full_lit(raw, n, pos, DS4_TOOL_CALLS_START_SHORT)) {
        ts->parse_pos = pos + strlen(DS4_TOOL_CALLS_START_SHORT);
        ts->tc_end = DS4_TOOL_CALLS_END_SHORT; ts->inv_s = DS4_INVOKE_START_SHORT; ts->inv_e = DS4_INVOKE_END_SHORT;
        ts->par_s = DS4_PARAM_START_SHORT; ts->par_e = DS4_PARAM_END_SHORT; return true;
    }
    if (raw_full_lit(raw, n, pos, "<tool_calls>")) {
        ts->parse_pos = pos + strlen("<tool_calls>");
        ts->tc_end = "</tool_calls>"; ts->inv_s = "<invoke"; ts->inv_e = "</invoke>";
        ts->par_s = "<parameter"; ts->par_e = "</parameter>"; return true;
    }
    ts->active = false; ts->state = ST_ERR; return false;
}

static void sse_event(const char *ev, const char *data) {
    buf_printf(&g_out, "event: %s\ndata: %s\n\n", ev, data);
}

/* kind 0=openai 1=anthropic */
static bool emit_start(int kind, const char *job, int index, const char *id, const char *name, int *next, int *open) {
    if (kind == 0) {
        buf_printf(&g_out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", job, CREATED_TEST);
        json_escape(&g_out, g_model);
        buf_printf(&g_out, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":%d,\"id\":", index);
        json_escape(&g_out, id);
        buf_puts(&g_out, ",\"type\":\"function\",\"function\":{\"name\":");
        json_escape(&g_out, name);
        buf_puts(&g_out, ",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
        return true;
    }
    if (*open == 3) return true;
    if (*open != 0) return false;
    buf b = {0};
    buf_printf(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"tool_use\",\"id\":", *next);
    json_escape(&b, id); buf_puts(&b, ",\"name\":"); json_escape(&b, name); buf_puts(&b, ",\"input\":{}}}");
    sse_event("content_block_start", b.ptr); free(b.ptr);
    *open = 3; return true;
}
static bool emit_args(int kind, const char *job, int index, const char *text, size_t len, int next) {
    if (len == 0) return true;
    if (kind == 0) {
        buf_printf(&g_out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", job, CREATED_TEST);
        json_escape(&g_out, g_model);
        buf_printf(&g_out, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":%d,\"function\":{\"arguments\":", index);
        json_escape_n(&g_out, text, len);
        buf_puts(&g_out, "}}]},\"finish_reason\":null}]}\n\n");
        return true;
    }
    buf b = {0};
    buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":", next);
    json_escape_n(&b, text, len); buf_puts(&b, "}}");
    sse_event("content_block_delta", b.ptr); free(b.ptr);
    return true;
}
static bool emit_close(int kind, const char *job, int *next, int *open) {
    if (kind == 0) return true;
    if (*open == 0) return true;
    if (*open == 1) {
        buf b = {0};
        buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"signature_delta\",\"signature\":", *next);
        json_escape(&b, job); buf_puts(&b, "}}");
        sse_event("content_block_delta", b.ptr); free(b.ptr);
    }
    buf b = {0};
    buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}", *next);
    sse_event("content_block_stop", b.ptr); free(b.ptr);
    *open = 0; (*next)++; return true;
}

static bool emit_string(int kind, const char *job, int index, const char *text, size_t len, int next) {
    if (len == 0) return true;
    char *raw = xstrndup(text, len);
    char *u = dsml_unescape(raw);
    buf frag = {0}; json_escape_fragment_n(&frag, u, strlen(u));
    bool ok = emit_args(kind, job, index, frag.ptr ? frag.ptr : "", frag.len, next);
    free(frag.ptr); free(u); free(raw); return ok;
}
static bool emit_prefix(int kind, const char *job, int index, const char *name, bool is_str, bool first, int next) {
    buf frag = {0};
    if (!first) buf_putc(&frag, ',');
    json_escape(&frag, name); buf_putc(&frag, ':');
    if (is_str) buf_putc(&frag, '"');
    bool ok = emit_args(kind, job, index, frag.ptr ? frag.ptr : "", frag.len, next);
    free(frag.ptr); return ok;
}

static bool tool_update(tool_st *ts, int kind, const char *job, const char *raw, size_t n, int *next, int *open) {
    while (ts->active && ts->parse_pos < n) {
        if (ts->state == ST_INV) {
            while (ts->parse_pos < n && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= n) return true;
            if (raw_full_lit(raw, n, ts->parse_pos, ts->tc_end)) { ts->parse_pos += strlen(ts->tc_end); ts->active = false; ts->state = ST_DONE; return true; }
            if (raw_partial_any(raw, n, ts->parse_pos, ts->tc_end, ts->inv_s)) return true;
            if (raw_full_lit(raw, n, ts->parse_pos, ts->inv_s)) {
                const char *tag_end = memchr(raw + ts->parse_pos, '>', n - ts->parse_pos);
                if (!tag_end) return true;
                char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
                char *name = dsml_attr(tag, "name"); free(tag);
                if (!name) { ts->active = false; ts->state = ST_ERR; return true; }
                char id[32]; snprintf(id, sizeof(id), "%s%d", ts->id_prefix, ts->index);
                if (!emit_start(kind, job, ts->index, id, name, next, open) ||
                    !emit_args(kind, job, ts->index, "{", 1, *next)) { free(name); return false; }
                free(name);
                ts->emitted_any = true; ts->args_open = true; ts->first_param = true;
                ts->parse_pos = (size_t)(tag_end - raw) + 1; ts->state = ST_PAR; continue;
            }
            ts->active = false; ts->state = ST_ERR; return true;
        }
        if (ts->state == ST_PAR) {
            while (ts->parse_pos < n && isspace((unsigned char)raw[ts->parse_pos])) ts->parse_pos++;
            if (ts->parse_pos >= n) return true;
            if (raw_full_lit(raw, n, ts->parse_pos, ts->inv_e)) {
                if (ts->args_open && !emit_args(kind, job, ts->index, "}", 1, *next)) return false;
                ts->args_open = false;
                if (!emit_close(kind, job, next, open)) return false;
                ts->parse_pos += strlen(ts->inv_e); ts->index++; ts->state = ST_INV; continue;
            }
            if (raw_partial_any(raw, n, ts->parse_pos, ts->inv_e, ts->par_s)) return true;
            if (raw_full_lit(raw, n, ts->parse_pos, ts->par_s)) {
                const char *tag_end = memchr(raw + ts->parse_pos, '>', n - ts->parse_pos);
                if (!tag_end) return true;
                char *tag = xstrndup(raw + ts->parse_pos, (size_t)(tag_end - (raw + ts->parse_pos) + 1));
                char *name = dsml_attr(tag, "name"); char *is = dsml_attr(tag, "string"); free(tag);
                if (!name || !is) { free(name); free(is); ts->active = false; ts->state = ST_ERR; return true; }
                bool sv = !strcmp(is, "true");
                if (!emit_prefix(kind, job, ts->index, name, sv, ts->first_param, *next)) { free(name); free(is); return false; }
                ts->first_param = false; ts->param_is_string = sv;
                ts->parse_pos = (size_t)(tag_end - raw) + 1; ts->state = ST_VAL;
                free(name); free(is); continue;
            }
            ts->active = false; ts->state = ST_ERR; return true;
        }
        if (ts->state == ST_VAL) {
            const char *end = find_lit_bounded(raw + ts->parse_pos, n - ts->parse_pos, ts->par_e);
            if (end) {
                size_t ve = (size_t)(end - raw);
                if (ve > ts->parse_pos) {
                    bool ok = ts->param_is_string ?
                        emit_string(kind, job, ts->index, raw + ts->parse_pos, ve - ts->parse_pos, *next) :
                        emit_args(kind, job, ts->index, raw + ts->parse_pos, ve - ts->parse_pos, *next);
                    if (!ok) return false;
                }
                if (ts->param_is_string && !emit_args(kind, job, ts->index, "\"", 1, *next)) return false;
                ts->parse_pos = ve + strlen(ts->par_e); ts->state = ST_PAR; continue;
            }
            size_t limit = tool_param_value_stream_safe_len(raw, ts->parse_pos, n, ts->par_e, ts->param_is_string);
            if (limit > ts->parse_pos) {
                bool ok = ts->param_is_string ?
                    emit_string(kind, job, ts->index, raw + ts->parse_pos, limit - ts->parse_pos, *next) :
                    emit_args(kind, job, ts->index, raw + ts->parse_pos, limit - ts->parse_pos, *next);
                if (!ok) return false;
                ts->parse_pos = limit;
            }
            return true;
        }
        return true;
    }
    return true;
}

typedef enum { OA_TH, OA_TX, OA_TL, OA_SU } oa_mode;
typedef struct { oa_mode mode; size_t emit_pos; bool checked; tool_st tool; } oa_st;
typedef enum { AN_TH, AN_TX, AN_TL, AN_SU } an_mode;
typedef struct { an_mode mode; int open, next; size_t emit_pos; bool checked, sent_th, sent_tx; tool_st tool; } an_st;

static void sse_chat_delta(const char *job, const char *field, const char *text, size_t len) {
    if (len == 0) return;
    buf_printf(&g_out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", job, CREATED_TEST);
    json_escape(&g_out, g_model);
    buf_puts(&g_out, ",\"choices\":[{\"index\":0,\"delta\":{");
    json_escape(&g_out, field); buf_putc(&g_out, ':'); json_escape_n(&g_out, text, len);
    buf_puts(&g_out, "},\"finish_reason\":null}]}\n\n");
}
static void sse_chunk_role(const char *job) {
    buf_printf(&g_out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", job, CREATED_TEST);
    json_escape(&g_out, g_model);
    buf_puts(&g_out, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n");
}

static void oa_start(oa_st *st, bool think) {
    memset(st, 0, sizeof(*st));
    st->mode = think ? OA_TH : OA_TX;
}
static bool oa_update(oa_st *st, const char *job, const char *raw, size_t n, bool think, bool has_tools, bool final) {
    if (st->mode == OA_TH) {
        if (!st->checked) {
            const char *open = "<think>"; size_t ol = 7;
            if (n < ol && !strncmp(raw, open, n) && !final) return true;
            if (n >= ol && !strncmp(raw, open, ol)) st->emit_pos = ol;
            st->checked = true;
        }
        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) limit = (size_t)(close - raw);
        else if (final) limit = utf8_stream_safe_len(raw, st->emit_pos, n, true);
        else { size_t hold = 7 - 1; limit = n > hold ? n - hold : st->emit_pos; limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false); }
        if (limit > st->emit_pos) { sse_chat_delta(job, "reasoning_content", raw + st->emit_pos, limit - st->emit_pos); st->emit_pos = limit; }
        if (close) { st->emit_pos = (size_t)(close - raw) + 8; st->mode = OA_TX; }
        else if (final) { st->mode = OA_SU; return true; }
        else return true;
    }
    if (st->mode == OA_TX) {
        const char *tool = has_tools ? find_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, n, has_tools, final);
        if (limit > st->emit_pos) { sse_chat_delta(job, "content", raw + st->emit_pos, limit - st->emit_pos); st->emit_pos = limit; }
        if (tool) {
            st->emit_pos = (size_t)(tool - raw);
            if (tool_init(&st->tool, raw, n, st->emit_pos, "call_")) st->mode = OA_TL;
            else st->mode = OA_SU;
        } else if (final) st->mode = OA_SU;
        (void)think;
    }
    if (st->mode == OA_TL) {
        int dummy_next = 0, dummy_open = 0;
        if (!tool_update(&st->tool, 0, job, raw, n, &dummy_next, &dummy_open)) return false;
        if (!st->tool.active) st->mode = OA_SU;
    }
    return true;
}
static void oa_dump_swapped_bash(const char *job) {
    buf_printf(&g_out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", job, CREATED_TEST);
    json_escape(&g_out, g_model);
    buf_puts(&g_out, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":");
    char fallback[128]; snprintf(fallback, sizeof(fallback), "%s_tool_0", job);
    json_escape(&g_out, fallback);
    buf_puts(&g_out, ",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":");
    json_escape(&g_out, "{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    buf_puts(&g_out, "}}]},\"finish_reason\":null}]}\n\n");
}

static void oa_finish(oa_st *st, const char *job, const char *raw, size_t n, bool think, const char *finish, bool dump_calls) {
    oa_update(st, job, raw, n, think, true, true);
    if (dump_calls && !st->tool.emitted_any)
        oa_dump_swapped_bash(job);
    buf_printf(&g_out, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", job, CREATED_TEST);
    json_escape(&g_out, g_model);
    buf_puts(&g_out, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
    json_escape(&g_out, finish);
    buf_puts(&g_out, "}]}\n\n");
    buf_puts(&g_out, "data: [DONE]\n\n");
}

static void an_open_block(an_st *st, int ty) {
    if (st->open == ty) return;
    buf b = {0};
    if (ty == 1)
        buf_printf(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}", st->next);
    else
        buf_printf(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}", st->next);
    sse_event("content_block_start", b.ptr); free(b.ptr); st->open = ty;
}
static void an_delta(an_st *st, int ty, const char *text, size_t len) {
    if (len == 0) return;
    buf b = {0};
    if (ty == 1)
        buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":", st->next);
    else
        buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"text_delta\",\"text\":", st->next);
    json_escape_n(&b, text, len); buf_puts(&b, "}}");
    sse_event("content_block_delta", b.ptr); free(b.ptr);
}
static void an_close(an_st *st, const char *id) {
    if (st->open == 0) return;
    if (st->open == 1) {
        buf b = {0};
        buf_printf(&b, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"signature_delta\",\"signature\":", st->next);
        json_escape(&b, id); buf_puts(&b, "}}");
        sse_event("content_block_delta", b.ptr); free(b.ptr);
    }
    buf b = {0}; buf_printf(&b, "{\"type\":\"content_block_stop\",\"index\":%d}", st->next);
    sse_event("content_block_stop", b.ptr); free(b.ptr);
    st->open = 0; st->next++;
}
static void an_start(an_st *st, const char *id, bool think, int prompt) {
    memset(st, 0, sizeof(*st));
    st->mode = think ? AN_TH : AN_TX;
    buf b = {0};
    buf_printf(&b, "{\"type\":\"message_start\",\"message\":{\"id\":\"%s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":", id);
    json_escape(&b, g_model);
    buf_printf(&b, ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":%d,\"output_tokens\":0,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0}}}", prompt);
    sse_event("message_start", b.ptr); free(b.ptr);
}
static bool an_update(an_st *st, const char *id, const char *raw, size_t n, bool has_tools, bool final) {
    if (st->mode == AN_TH) {
        if (!st->checked) {
            const char *open = "<think>"; size_t ol = 7;
            if (n < ol && !strncmp(raw, open, n) && !final) return true;
            if (n >= ol && !strncmp(raw, open, ol)) st->emit_pos = ol;
            st->checked = true;
        }
        const char *close = strstr(raw + st->emit_pos, "</think>");
        size_t limit;
        if (close) limit = (size_t)(close - raw);
        else if (final) limit = utf8_stream_safe_len(raw, st->emit_pos, n, true);
        else { size_t hold = 7 - 1; limit = n > hold ? n - hold : st->emit_pos; limit = utf8_stream_safe_len(raw, st->emit_pos, limit, false); }
        if (limit > st->emit_pos) { an_open_block(st, 1); an_delta(st, 1, raw + st->emit_pos, limit - st->emit_pos); st->sent_th = true; st->emit_pos = limit; }
        if (close || final) {
            an_close(st, id);
            if (close) { st->emit_pos = (size_t)(close - raw) + 8; st->mode = AN_TX; }
            else { st->mode = AN_SU; return true; }
        } else return true;
    }
    if (st->mode == AN_TX) {
        const char *tool = has_tools ? find_tool_start(raw + st->emit_pos) : NULL;
        size_t limit = text_stream_safe_limit(raw, st->emit_pos, n, has_tools, final);
        if (limit > st->emit_pos) { an_open_block(st, 2); an_delta(st, 2, raw + st->emit_pos, limit - st->emit_pos); st->sent_tx = true; st->emit_pos = limit; }
        if (tool) {
            an_close(st, id);
            st->emit_pos = (size_t)(tool - raw);
            if (!final && tool_init(&st->tool, raw, n, st->emit_pos, "toolu_")) st->mode = AN_TL;
            else st->mode = AN_SU;
        } else if (final) { an_close(st, id); st->mode = AN_SU; }
    }
    if (st->mode == AN_TL) {
        if (!tool_update(&st->tool, 1, id, raw, n, &st->next, &st->open)) return false;
        if (!st->tool.active) st->mode = AN_SU;
    }
    return true;
}
static void an_dump_swapped_bash(an_st *st, const char *id) {
    if (st->tool.emitted_any) return;
    char fallback[128]; snprintf(fallback, sizeof(fallback), "toolu_%s_0", id);
    buf b = {0};
    buf_printf(&b, "{\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"tool_use\",\"id\":", st->next);
    json_escape(&b, fallback); buf_puts(&b, ",\"name\":\"bash\",\"input\":{}}}");
    sse_event("content_block_start", b.ptr); free(b.ptr);
    buf b2 = {0};
    buf_printf(&b2, "{\"type\":\"content_block_delta\",\"index\":%d,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":", st->next);
    json_escape(&b2, "{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    buf_puts(&b2, "}}");
    sse_event("content_block_delta", b2.ptr); free(b2.ptr);
    buf b3 = {0}; buf_printf(&b3, "{\"type\":\"content_block_stop\",\"index\":%d}", st->next);
    sse_event("content_block_stop", b3.ptr); free(b3.ptr);
    st->next++;
}

static void an_finish(an_st *st, const char *id, const char *raw, size_t n, const char *finish, int completion, bool dump_calls) {
    an_update(st, id, raw, n, true, true);
    if (dump_calls) an_dump_swapped_bash(st, id);
    const char *reason = !strcmp(finish, "tool_calls") ? "tool_use" : "end_turn";
    buf b = {0};
    buf_printf(&b, "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
    json_escape(&b, reason);
    buf_puts(&b, ",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
    buf_printf(&b, "%d}}", completion);
    sse_event("message_delta", b.ptr); free(b.ptr);
    sse_event("message_stop", "{\"type\":\"message_stop\"}");
}

static void dump_and_print(void) {
    fwrite(g_out.ptr ? g_out.ptr : "", 1, g_out.len, stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: tool_stream_c_oracle SCRIPT\n"); return 2; }
    const char *name = argv[1];
    if (!strcmp(name, "openai-partial")) {
        oa_st st; oa_start(&st, false);
        sse_chunk_role("chatcmpl_partial_tool");
        char raw1[512]; snprintf(raw1, sizeof(raw1),
            "Before.\n\n%s\n%s name=\"bash\">\n%s name=\"command\" string=\"true\">echo partial",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        oa_update(&st, "chatcmpl_partial_tool", raw1, strlen(raw1), false, true, false);
        char raw2[640]; snprintf(raw2, sizeof(raw2),
            "Before.\n\n%s\n%s name=\"bash\">\n%s name=\"command\" string=\"true\">echo partial done%s\n%s\n%s",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START, DS4_PARAM_END, DS4_INVOKE_END, DS4_TOOL_CALLS_END);
        oa_update(&st, "chatcmpl_partial_tool", raw2, strlen(raw2), false, true, false);
        oa_finish(&st, "chatcmpl_partial_tool", raw2, strlen(raw2), false, "tool_calls", false);
    } else if (!strcmp(name, "openai-raw")) {
        oa_st st; oa_start(&st, false);
        char raw[512]; snprintf(raw, sizeof(raw),
            "%s\n%s name=\"edit\">\n%s name=\"edits\" string=\"false\">[1,2,3",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        oa_update(&st, "chatcmpl_raw_tool", raw, strlen(raw), false, true, false);
    } else if (!strcmp(name, "openai-wait-tag")) {
        oa_st st; oa_start(&st, false);
        char raw1[256]; snprintf(raw1, sizeof(raw1), "%s\n%s", DS4_TOOL_CALLS_START, DS4_INVOKE_START);
        oa_update(&st, "chatcmpl_incomplete_tool", raw1, strlen(raw1), false, true, false);
        char raw2[320]; snprintf(raw2, sizeof(raw2), "%s\n%s name=\"bash\">\n%s",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        oa_update(&st, "chatcmpl_incomplete_tool", raw2, strlen(raw2), false, true, false);
    } else if (!strcmp(name, "openai-entity")) {
        oa_st st; oa_start(&st, false);
        char raw1[512]; snprintf(raw1, sizeof(raw1),
            "%s\n%s name=\"bash\">\n%s name=\"command\" string=\"true\">echo &amp",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        oa_update(&st, "chatcmpl_entity_tool", raw1, strlen(raw1), false, true, false);
        char raw2[640]; snprintf(raw2, sizeof(raw2),
            "%s\n%s name=\"bash\">\n%s name=\"command\" string=\"true\">echo &amp; done%s\n%s\n%s",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START, DS4_PARAM_END, DS4_INVOKE_END, DS4_TOOL_CALLS_END);
        oa_update(&st, "chatcmpl_entity_tool", raw2, strlen(raw2), false, true, false);
    } else if (!strcmp(name, "openai-think-tool")) {
        oa_st st; oa_start(&st, true);
        sse_chunk_role("chatcmpl_test");
        const char *raw1 = "<think>need a tool</think>Hello.\n\n";
        oa_update(&st, "chatcmpl_test", raw1, strlen(raw1), true, true, false);
        char raw2[320]; snprintf(raw2, sizeof(raw2), "<think>need a tool</think>Hello.\n\n%s\n", DS4_TOOL_CALLS_START);
        oa_update(&st, "chatcmpl_test", raw2, strlen(raw2), true, true, false);
        oa_finish(&st, "chatcmpl_test", raw2, strlen(raw2), true, "tool_calls", true);
    } else if (!strcmp(name, "anthropic-partial")) {
        an_st st; an_start(&st, "msg_tool", false, 7);
        char raw1[512]; snprintf(raw1, sizeof(raw1),
            "Before.\n\n%s\n%s name=\"bash\">\n%s name=\"command\" string=\"true\">echo partial",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        an_update(&st, "msg_tool", raw1, strlen(raw1), true, false);
        char raw2[640]; snprintf(raw2, sizeof(raw2),
            "Before.\n\n%s\n%s name=\"bash\">\n%s name=\"command\" string=\"true\">echo partial done%s\n%s\n%s",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START, DS4_PARAM_END, DS4_INVOKE_END, DS4_TOOL_CALLS_END);
        an_update(&st, "msg_tool", raw2, strlen(raw2), true, false);
        an_finish(&st, "msg_tool", raw2, strlen(raw2), "tool_calls", 5, false);
    } else if (!strcmp(name, "anthropic-think-tool")) {
        an_st st; an_start(&st, "msg_test", true, 10);
        const char *raw1 = "need a tool</think>Hello.\n\n";
        an_update(&st, "msg_test", raw1, strlen(raw1), true, false);
        char raw2[320]; snprintf(raw2, sizeof(raw2), "need a tool</think>Hello.\n\n%s\n", DS4_TOOL_CALLS_START);
        an_update(&st, "msg_test", raw2, strlen(raw2), true, false);
        an_finish(&st, "msg_test", raw2, strlen(raw2), "tool_calls", 8, true);
    } else if (!strcmp(name, "openai-utf8")) {
        oa_st st; oa_start(&st, false);
        char prefix[512]; snprintf(prefix, sizeof(prefix),
            "%s\n%s name=\"write\">\n%s name=\"content\" string=\"true\">flag ",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START);
        size_t plen = strlen(prefix);
        char *raw1 = malloc(plen + 3); if (!raw1) die("oom");
        memcpy(raw1, prefix, plen); raw1[plen] = (char)0xf0; raw1[plen + 1] = (char)0x9f; raw1[plen + 2] = 0;
        oa_update(&st, "chatcmpl_utf8_tool", raw1, plen + 2, false, true, false);
        const char suffix[] = " done" DS4_PARAM_END "\n" DS4_INVOKE_END "\n" DS4_TOOL_CALLS_END;
        char flag[4] = {(char)0xf0, (char)0x9f, (char)0x9a, (char)0xa9};
        size_t slen = strlen(suffix);
        char *raw2 = malloc(plen + 4 + slen + 1); if (!raw2) die("oom");
        memcpy(raw2, prefix, plen); memcpy(raw2 + plen, flag, 4); memcpy(raw2 + plen + 4, suffix, slen + 1);
        oa_update(&st, "chatcmpl_utf8_tool", raw2, plen + 4 + slen, false, true, false);
        free(raw1); free(raw2);
    } else if (!strcmp(name, "openai-multi")) {
        oa_st st; oa_start(&st, false);
        char raw[768]; snprintf(raw, sizeof(raw),
            "%s\n%s name=\"read\">\n%s name=\"path\" string=\"true\">a.c%s\n%s\n%s name=\"bash\">\n%s name=\"command\" string=\"true\">wc -l a.c%s\n%s\n%s",
            DS4_TOOL_CALLS_START, DS4_INVOKE_START, DS4_PARAM_START, DS4_PARAM_END, DS4_INVOKE_END,
            DS4_INVOKE_START, DS4_PARAM_START, DS4_PARAM_END, DS4_INVOKE_END, DS4_TOOL_CALLS_END);
        oa_update(&st, "chatcmpl_multi_tool", raw, strlen(raw), false, true, false);
    } else { fprintf(stderr, "unknown script\n"); return 2; }
    dump_and_print();
    return 0;
}
