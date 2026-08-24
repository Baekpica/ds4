/* C four-surface parser oracle. Copied from ds4_server.c at v0.6.3-dfm.
 * Tokenize / render / KV restore / continuation pin are cut off so a NULL
 * engine can dump parse fields. prepare_tool_choice sets has_tools and
 * rejects required-without-schemas; it does not tokenize prefixes. */

#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSON_MAX_NESTING 256
#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

enum { REQ_CHAT = 0, REQ_COMPLETION };
enum { API_OPENAI = 0, API_ANTHROPIC, API_RESPONSES };
enum { TOOL_CHOICE_AUTO = 0, TOOL_CHOICE_NONE, TOOL_CHOICE_REQUIRED };
enum { DS4_THINK_NONE, DS4_THINK_LOW, DS4_THINK_HIGH, DS4_THINK_MAX };

enum {
    DS4_NEED_STREAMING            = 1u << 0,
    DS4_NEED_PER_ROW_SAMPLING     = 1u << 1,
    DS4_NEED_THINKING             = 1u << 2,
    DS4_NEED_STOP_SCAN            = 1u << 3,
    DS4_NEED_TOOL_SCAN            = 1u << 4,
    DS4_NEED_TOKEN_IDS            = 1u << 5,
    DS4_NEED_LIVE_FRONTIER        = 1u << 6,
    DS4_NEED_CONTINUATION_PUBLISH = 1u << 7,
    DS4_NEED_CORRECTIVE_RECOVERY  = 1u << 8,
    DS4_NEED_DURABLE_RESPONSE     = 1u << 9,
    DS4_NEED_PREFILL_ONLY         = 1u << 10,
    DS4_NEED_BANK_FRONTIER        = 1u << 11,
};

typedef struct { char *ptr; size_t len, cap; } buf;
typedef struct { char **v; int len, cap; } stop_list;
typedef struct { char *id, *name, *arguments; } tool_call;
typedef struct { tool_call *v; int len, cap; } tool_calls;
typedef struct {
    char *role, *content, *reasoning, *tool_call_id;
    char **tool_call_ids;
    int tool_call_ids_len, tool_call_ids_cap;
    tool_calls calls;
} chat_msg;
typedef struct { chat_msg *v; int len, cap; } chat_msgs;
typedef struct {
    int kind, api;
    char *model;
    bool model_from_request;
    int max_tokens;
    bool max_tokens_set;
    int top_k;
    float temperature, top_p, min_p;
    uint64_t seed;
    bool stream, stream_include_usage, return_token_ids;
    int think_mode;
    bool has_tools, has_tool_results;
    int tool_choice;
    stop_list stops;
    bool reasoning_summary_emit;
    bool responses_requires_live_tool_state;
    bool responses_requires_live_reasoning;
    bool anthropic_requires_live_tool_state;
    chat_msgs messages;
    char *tool_schemas;
    uint32_t needs;
} request;

static void die(const char *m) {
    fprintf(stderr, "parse_c_oracle: %s\n", m);
    exit(2);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("oom");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("oom");
    return q;
}

static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s) + 1;
    char *p = xmalloc(n);
    memcpy(p, s, n);
    return p;
}

static void buf_grow(buf *b, size_t n) {
    if (b->len + n + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < b->len + n + 1) cap *= 2;
    b->ptr = xrealloc(b->ptr, cap);
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

static char *buf_take(buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_put(buf *b, uint32_t cp) {
    if (cp <= 0x7f) buf_putc(b, (char)cp);
    else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static bool json_string(const char **p, char **out) {
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') { buf_putc(&b, (char)c); continue; }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) { buf_free(&b); return false; }
            if (cp >= 0xd800 && cp <= 0xdbff && json_u16(p, &lo) &&
                lo >= 0xdc00 && lo <= 0xdfff)
                cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
            utf8_put(&b, cp);
            break;
        }
        default:
            buf_free(&b);
            return false;
        }
    }
    if (**p != '"') { buf_free(&b); return false; }
    (*p)++;
    *out = b.ptr ? b.ptr : xstrdup("");
    return true;
}

static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') { (*p)++; return true; }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') { (*p)++; return true; }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null")) return true;
    double v = 0.0;
    return json_number(p, &v);
}

static bool json_skip_value(const char **p) { return json_skip_value_depth(p, 0); }

static bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    if (v < 0) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

static bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) { *out = true; return true; }
    if (json_lit(p, "false")) { *out = false; return true; }
    return false;
}

static bool json_raw_value(const char **p, char **out) {
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = xmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

static bool json_content(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) { *out = xstrdup(""); return true; }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) { buf_free(&b); return false; }
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) { buf_free(&b); return false; }
                json_ws(p);
                if (**p != ':') { free(key); buf_free(&b); return false; }
                (*p)++;
                if (!strcmp(key, "text")) {
                    char *s = NULL;
                    if (!json_string(p, &s)) { free(key); buf_free(&b); return false; }
                    buf_puts(&b, s);
                    free(s);
                } else if (!json_skip_value(p)) {
                    free(key);
                    buf_free(&b);
                    return false;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') { buf_free(&b); return false; }
            (*p)++;
        } else if (!json_skip_value(p)) {
            buf_free(&b);
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') { buf_free(&b); return false; }
    (*p)++;
    *out = buf_take(&b);
    return true;
}

static bool output_format_type_supported(const char *field, const char *type,
                                         char *err, size_t errlen) {
    if (!strcmp(type, "text")) return true;
    if (!strcmp(type, "json_object") || !strcmp(type, "json_schema")) {
        snprintf(err, errlen,
                 "%s type '%s' is not implemented: structured output is "
                 "unsupported; omit %s or use type \"text\"",
                 field, type, field);
        return false;
    }
    snprintf(err, errlen, "%s type '%s' is not supported", field, type);
    return false;
}

static bool parse_output_format_value(const char **p, const char *field,
                                      char *err, size_t errlen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p == '"') {
        char *type = NULL;
        if (!json_string(p, &type)) return false;
        bool ok = output_format_type_supported(field, type, err, errlen);
        free(type);
        return ok;
    }
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    char *type = NULL;
    bool ok = true;
    while (ok && **p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) { ok = false; break; }
        json_ws(p);
        if (**p != ':') { free(key); ok = false; break; }
        (*p)++;
        if (!strcmp(key, "type")) {
            json_ws(p);
            free(type);
            type = NULL;
            if (!json_string(p, &type)) ok = false;
        } else if (!json_skip_value(p)) {
            ok = false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (ok && **p == '}') (*p)++; else ok = false;
    if (ok && type) ok = output_format_type_supported(field, type, err, errlen);
    free(type);
    return ok;
}

static bool parse_responses_text_value(const char **p, char *err, size_t errlen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        if (!strcmp(key, "format")) {
            if (!parse_output_format_value(p, "text.format", err, errlen)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_reasoning_effort_name(const char *s, int *out) {
    if (!s) return false;
    if (!strcmp(s, "max")) { *out = DS4_THINK_MAX; return true; }
    if (!strcmp(s, "high") || !strcmp(s, "xhigh")) { *out = DS4_THINK_HIGH; return true; }
    if (!strcmp(s, "low") || !strcmp(s, "medium") || !strcmp(s, "minimal")) {
        *out = DS4_THINK_LOW; return true;
    }
    if (!strcmp(s, "none") || !strcmp(s, "off")) { *out = DS4_THINK_NONE; return true; }
    return false;
}

static bool parse_reasoning_effort_value(const char **p, int *out) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    char *effort = NULL;
    if (!json_string(p, &effort)) return false;
    bool ok = parse_reasoning_effort_name(effort, out);
    free(effort);
    return ok;
}

static bool parse_output_config_effort(const char **p, int *effort,
                                       char *err, size_t errlen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        if (!strcmp(key, "effort")) {
            if (!parse_reasoning_effort_value(p, effort)) {
                free(key);
                return false;
            }
        } else if (!strcmp(key, "format")) {
            if (!parse_output_format_value(p, "output_config.format", err, errlen)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_thinking_control_value(const char **p, bool *thinking_enabled) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p == 't' || **p == 'f') return json_bool(p, thinking_enabled);
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        if (!strcmp(key, "type")) {
            char *type = NULL;
            if (!json_string(p, &type)) { free(key); return false; }
            if (!strcmp(type, "enabled")) *thinking_enabled = true;
            else if (!strcmp(type, "disabled")) *thinking_enabled = false;
            free(type);
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static float server_default_temperature(void) {
    const char *e = getenv("DS4_SERVER_DEFAULT_TEMP");
    return (e && *e) ? (float)atof(e) : DS4_DEFAULT_TEMPERATURE;
}

static bool model_alias_disables_thinking(const char *model) {
    return model && (!strcmp(model, "deepseek-chat") ||
                     !strcmp(model, "k-exaone-236b-a23b-chat"));
}

static bool model_alias_enables_thinking(const char *model) {
    return model && !strcmp(model, "deepseek-reasoner");
}

static int think_mode_from_enabled(bool enabled, int effort) {
    if (!enabled || effort == DS4_THINK_NONE) return DS4_THINK_NONE;
    return effort;
}

static bool think_mode_enabled(int mode) { return mode != DS4_THINK_NONE; }

static void stop_list_clear(stop_list *s) {
    for (int i = 0; i < s->len; i++) free(s->v[i]);
    free(s->v);
    memset(s, 0, sizeof(*s));
}

static void stop_list_push(stop_list *s, char *v) {
    if (!v || !v[0]) { free(v); return; }
    if (s->len == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 4;
        s->v = xrealloc(s->v, (size_t)s->cap * sizeof(s->v[0]));
    }
    s->v[s->len++] = v;
}

static bool parse_stop(const char **p, stop_list *out) {
    json_ws(p);
    stop_list_clear(out);
    if (**p == '"') {
        char *s = NULL;
        if (!json_string(p, &s)) return false;
        stop_list_push(out, s);
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            stop_list_push(out, s);
        } else if (!json_skip_value(p)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool parse_stream_options(const char **p, bool *include_usage) {
    json_ws(p);
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        if (!strcmp(key, "include_usage")) {
            if (!json_bool(p, include_usage)) { free(key); return false; }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static void tool_calls_free(tool_calls *c) {
    for (int i = 0; i < c->len; i++) {
        free(c->v[i].id);
        free(c->v[i].name);
        free(c->v[i].arguments);
    }
    free(c->v);
    memset(c, 0, sizeof(*c));
}

static void tool_calls_push(tool_calls *c, tool_call tc) {
    if (c->len == c->cap) {
        c->cap = c->cap ? c->cap * 2 : 4;
        c->v = xrealloc(c->v, (size_t)c->cap * sizeof(c->v[0]));
    }
    c->v[c->len++] = tc;
}

static void chat_msg_add_tool_call_id(chat_msg *m, const char *id) {
    if (!id || !id[0]) return;
    if (!m->tool_call_id || !m->tool_call_id[0]) {
        free(m->tool_call_id);
        m->tool_call_id = xstrdup(id);
    }
    for (int i = 0; i < m->tool_call_ids_len; i++)
        if (!strcmp(m->tool_call_ids[i], id)) return;
    if (m->tool_call_ids_len == m->tool_call_ids_cap) {
        m->tool_call_ids_cap = m->tool_call_ids_cap ? m->tool_call_ids_cap * 2 : 4;
        m->tool_call_ids = xrealloc(m->tool_call_ids,
                                    (size_t)m->tool_call_ids_cap * sizeof(char *));
    }
    m->tool_call_ids[m->tool_call_ids_len++] = xstrdup(id);
}

static void chat_msg_free(chat_msg *m) {
    free(m->role);
    free(m->content);
    free(m->reasoning);
    free(m->tool_call_id);
    for (int i = 0; i < m->tool_call_ids_len; i++) free(m->tool_call_ids[i]);
    free(m->tool_call_ids);
    tool_calls_free(&m->calls);
    memset(m, 0, sizeof(*m));
}

static void chat_msgs_free(chat_msgs *msgs) {
    for (int i = 0; i < msgs->len; i++) chat_msg_free(&msgs->v[i]);
    free(msgs->v);
    memset(msgs, 0, sizeof(*msgs));
}

static void chat_msgs_push(chat_msgs *msgs, chat_msg msg) {
    if (msgs->len == msgs->cap) {
        msgs->cap = msgs->cap ? msgs->cap * 2 : 4;
        msgs->v = xrealloc(msgs->v, (size_t)msgs->cap * sizeof(msgs->v[0]));
    }
    msgs->v[msgs->len++] = msg;
}

static bool parse_function_call(const char **p, tool_call *tc) {
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        if (!strcmp(key, "name")) {
            free(tc->name);
            if (!json_string(p, &tc->name)) { free(key); return false; }
        } else if (!strcmp(key, "arguments")) {
            free(tc->arguments);
            json_ws(p);
            if (**p == '"') {
                if (!json_string(p, &tc->arguments)) { free(key); return false; }
            } else if (!json_raw_value(p, &tc->arguments)) {
                free(key);
                return false;
            }
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_tool_calls_value(const char **p, tool_calls *out) {
    json_ws(p);
    tool_calls_free(out);
    if (json_lit(p, "null")) return true;
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        tool_call tc = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) { free(tc.id); free(tc.name); free(tc.arguments); return false; }
            json_ws(p);
            if (**p != ':') { free(key); free(tc.id); free(tc.name); free(tc.arguments); return false; }
            (*p)++;
            if (!strcmp(key, "id")) {
                free(tc.id);
                if (!json_string(p, &tc.id)) { free(key); free(tc.name); free(tc.arguments); return false; }
            } else if (!strcmp(key, "function")) {
                if (!parse_function_call(p, &tc)) { free(key); free(tc.id); free(tc.name); free(tc.arguments); return false; }
            } else if (!json_skip_value(p)) {
                free(key); free(tc.id); free(tc.name); free(tc.arguments); return false;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') { free(tc.id); free(tc.name); free(tc.arguments); return false; }
        (*p)++;
        if (tc.name && tc.name[0] && tc.arguments && tc.arguments[0])
            tool_calls_push(out, tc);
        else {
            free(tc.id); free(tc.name); free(tc.arguments);
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool parse_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) { chat_msg_free(&msg); return false; }
            json_ws(p);
            if (**p != ':') { free(key); chat_msg_free(&msg); return false; }
            (*p)++;
            if (!strcmp(key, "role")) {
                free(msg.role);
                if (!json_string(p, &msg.role)) { free(key); chat_msg_free(&msg); return false; }
            } else if (!strcmp(key, "content")) {
                free(msg.content);
                if (!json_content(p, &msg.content)) { free(key); chat_msg_free(&msg); return false; }
            } else if (!strcmp(key, "reasoning_content")) {
                free(msg.reasoning);
                if (!json_content(p, &msg.reasoning)) { free(key); chat_msg_free(&msg); return false; }
            } else if (!strcmp(key, "tool_call_id")) {
                char *id = NULL;
                if (!json_string(p, &id)) { free(key); chat_msg_free(&msg); return false; }
                chat_msg_add_tool_call_id(&msg, id);
                free(id);
            } else if (!strcmp(key, "tool_calls")) {
                if (!parse_tool_calls_value(p, &msg.calls)) { free(key); chat_msg_free(&msg); return false; }
            } else if (!json_skip_value(p)) {
                free(key); chat_msg_free(&msg); return false;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') { chat_msg_free(&msg); return false; }
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        chat_msgs_push(msgs, msg);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static void append_tool_result_text(buf *b, const char *s) {
    const char *end = "</tool_result>";
    while (s && *s) {
        if (!strncmp(s, end, strlen(end))) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static bool parse_anthropic_content_block(const char **p, chat_msg *msg) {
    if (**p != '{') return false;
    (*p)++;
    char *type = NULL, *text = NULL, *thinking = NULL, *id = NULL;
    char *name = NULL, *input = NULL, *tool_result = NULL;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') { free(key); goto bad; }
        (*p)++;
        if (!strcmp(key, "type")) {
            free(type);
            if (!json_string(p, &type)) { free(key); goto bad; }
        } else if (!strcmp(key, "text")) {
            free(text);
            if (!json_content(p, &text)) { free(key); goto bad; }
        } else if (!strcmp(key, "thinking")) {
            free(thinking);
            if (!json_content(p, &thinking)) { free(key); goto bad; }
        } else if (!strcmp(key, "id") || !strcmp(key, "tool_use_id")) {
            free(id);
            if (!json_string(p, &id)) { free(key); goto bad; }
        } else if (!strcmp(key, "name")) {
            free(name);
            if (!json_string(p, &name)) { free(key); goto bad; }
        } else if (!strcmp(key, "input")) {
            free(input);
            if (!json_raw_value(p, &input)) { free(key); goto bad; }
        } else if (!strcmp(key, "content")) {
            free(tool_result);
            if (!json_content(p, &tool_result)) { free(key); goto bad; }
        } else if (!json_skip_value(p)) {
            free(key); goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;
    if (type && !strcmp(type, "tool_use")) {
        tool_call tc = {0};
        tc.id = xstrdup(id ? id : "");
        tc.name = xstrdup(name ? name : "");
        tc.arguments = xstrdup(input ? input : "{}");
        tool_calls_push(&msg->calls, tc);
    } else if (type && !strcmp(type, "tool_result")) {
        if (id) chat_msg_add_tool_call_id(msg, id);
        buf b = {0};
        if (msg->content) buf_puts(&b, msg->content);
        buf_puts(&b, "<tool_result>");
        append_tool_result_text(&b, tool_result ? tool_result : "");
        buf_puts(&b, "</tool_result>");
        free(msg->content);
        msg->content = buf_take(&b);
    } else {
        if (text) {
            buf b = {0};
            if (msg->content) buf_puts(&b, msg->content);
            buf_puts(&b, text);
            free(msg->content);
            msg->content = buf_take(&b);
        }
        if (thinking) {
            buf b = {0};
            if (msg->reasoning) buf_puts(&b, msg->reasoning);
            buf_puts(&b, thinking);
            free(msg->reasoning);
            msg->reasoning = buf_take(&b);
        }
    }
    free(type); free(text); free(thinking); free(id);
    free(name); free(input); free(tool_result);
    return true;
bad:
    free(type); free(text); free(thinking); free(id);
    free(name); free(input); free(tool_result);
    return false;
}

static bool parse_anthropic_content(const char **p, chat_msg *msg) {
    json_ws(p);
    if (**p == '"') {
        free(msg->content);
        return json_string(p, &msg->content);
    }
    if (json_lit(p, "null")) {
        free(msg->content);
        msg->content = xstrdup("");
        return true;
    }
    if (**p != '[') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) return false;
            buf b = {0};
            if (msg->content) buf_puts(&b, msg->content);
            buf_puts(&b, s);
            free(s);
            free(msg->content);
            msg->content = buf_take(&b);
        } else if (**p == '{') {
            if (!parse_anthropic_content_block(p, msg)) return false;
        } else if (!json_skip_value(p)) {
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool parse_anthropic_messages(const char **p, chat_msgs *msgs) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') return false;
        (*p)++;
        chat_msg msg = {0};
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) { chat_msg_free(&msg); return false; }
            json_ws(p);
            if (**p != ':') { free(key); chat_msg_free(&msg); return false; }
            (*p)++;
            if (!strcmp(key, "role")) {
                free(msg.role);
                if (!json_string(p, &msg.role)) { free(key); chat_msg_free(&msg); return false; }
            } else if (!strcmp(key, "content")) {
                free(msg.content);
                msg.content = NULL;
                if (!parse_anthropic_content(p, &msg)) { free(key); chat_msg_free(&msg); return false; }
            } else if (!json_skip_value(p)) {
                free(key); chat_msg_free(&msg); return false;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
        }
        if (**p != '}') { chat_msg_free(&msg); return false; }
        (*p)++;
        if (!msg.role) msg.role = xstrdup("user");
        if (!msg.content) msg.content = xstrdup("");
        chat_msgs_push(msgs, msg);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static bool anthropic_system_part_is_private(const char *s) {
    return s && !strncmp(s, "x-anthropic-", 12);
}

static void append_anthropic_system_part(buf *b, const char *s) {
    if (!s || !s[0] || anthropic_system_part_is_private(s)) return;
    if (b->len && b->ptr[b->len - 1] != '\n') buf_putc(b, '\n');
    buf_puts(b, s);
}

static bool parse_anthropic_system_object(const char **p, buf *out) {
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        if (!strcmp(key, "text")) {
            char *text = NULL;
            if (!json_string(p, &text)) { free(key); return false; }
            append_anthropic_system_part(out, text);
            free(text);
        } else if (!json_skip_value(p)) {
            free(key);
            return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_anthropic_system(const char **p, char **out) {
    json_ws(p);
    buf b = {0};
    if (**p == '"') {
        char *text = NULL;
        if (!json_string(p, &text)) return false;
        append_anthropic_system_part(&b, text);
        free(text);
        *out = buf_take(&b);
        return true;
    }
    if (json_lit(p, "null")) { *out = xstrdup(""); return true; }
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *text = NULL;
            if (!json_string(p, &text)) { buf_free(&b); return false; }
            append_anthropic_system_part(&b, text);
            free(text);
        } else if (**p == '{') {
            if (!parse_anthropic_system_object(p, &b)) { buf_free(&b); return false; }
        } else if (!json_skip_value(p)) {
            buf_free(&b);
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') { buf_free(&b); return false; }
    (*p)++;
    *out = buf_take(&b);
    return true;
}

static bool parse_prompt(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (**p != '[') {
        if (!json_skip_value(p)) return false;
        *out = xstrdup("");
        return true;
    }
    (*p)++;
    json_ws(p);
    if (**p == '"') {
        if (!json_string(p, out)) return false;
    } else {
        *out = xstrdup("");
        if (**p && **p != ']' && !json_skip_value(p)) return false;
    }
    while (**p && **p != ']') {
        json_ws(p);
        if (**p == ',') {
            (*p)++;
            if (!json_skip_value(p)) return false;
        } else break;
    }
    if (**p != ']') return false;
    (*p)++;
    return true;
}

static char *openai_function_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) return NULL;
        json_ws(&p);
        if (*p != ':') { free(key); return NULL; }
        p++;
        if (!strcmp(key, "function")) {
            free(key);
            char *v = NULL;
            if (!json_raw_value(&p, &v)) return NULL;
            return v;
        }
        free(key);
        if (!json_skip_value(&p)) return NULL;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    return NULL;
}

static char *responses_special_schema_from_tool(const char *raw) {
    const char *p = raw;
    json_ws(&p);
    if (*p != '{') return NULL;
    p++;
    char *typ = NULL, *description = NULL, *parameters = NULL;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') { free(key); goto bad; }
        p++;
        if (!strcmp(key, "type")) {
            free(typ);
            if (!json_string(&p, &typ)) { free(key); goto bad; }
        } else if (!strcmp(key, "description")) {
            free(description);
            if (!json_string(&p, &description)) { free(key); goto bad; }
        } else if (!strcmp(key, "parameters")) {
            free(parameters);
            if (!json_raw_value(&p, &parameters)) { free(key); goto bad; }
        } else if (!json_skip_value(&p)) {
            free(key); goto bad;
        }
        free(key);
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (typ && !strcmp(typ, "tool_search")) {
        const char *desc = description ? description : "Search available tools.";
        const char *params = parameters ? parameters : "{\"type\":\"object\",\"properties\":{}}";
        buf b = {0};
        buf_puts(&b, "{\"name\":\"tool_search\",\"description\":\"");
        for (const char *s = desc; *s; s++) {
            if (*s == '"' || *s == '\\') { buf_putc(&b, '\\'); buf_putc(&b, *s); }
            else buf_putc(&b, *s);
        }
        buf_puts(&b, "\",\"parameters\":");
        buf_puts(&b, params);
        buf_putc(&b, '}');
        free(typ); free(description); free(parameters);
        return buf_take(&b);
    }
bad:
    free(typ); free(description); free(parameters);
    return NULL;
}

static void append_raw_json_line(buf *b, const char *s) {
    if (!s || !s[0]) return;
    if (b->len) buf_putc(b, '\n');
    buf_puts(b, s);
}

static bool parse_tools_value(const char **p, char **out) {
    json_ws(p);
    if (json_lit(p, "null")) { *out = xstrdup(""); return true; }
    if (**p != '[') return false;
    (*p)++;
    buf schemas = {0};
    json_ws(p);
    while (**p && **p != ']') {
        char *raw = NULL;
        if (!json_raw_value(p, &raw)) { buf_free(&schemas); return false; }
        char *function = openai_function_schema_from_tool(raw);
        if (function) append_raw_json_line(&schemas, function);
        else {
            char *special = responses_special_schema_from_tool(raw);
            if (special) {
                append_raw_json_line(&schemas, special);
                free(special);
            } else {
                append_raw_json_line(&schemas, raw);
            }
        }
        free(function);
        free(raw);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') { buf_free(&schemas); return false; }
    (*p)++;
    *out = buf_take(&schemas);
    return true;
}

static bool parse_openai_tool_choice_value(const char **p, int *mode,
                                           char *err, size_t errlen) {
    json_ws(p);
    if (**p != '"') {
        snprintf(err, errlen, **p == '{' ? "forced tool_choice not supported" :
                                           "invalid tool_choice");
        return false;
    }
    char *choice = NULL;
    if (!json_string(p, &choice)) return false;
    bool ok = true;
    if (!strcmp(choice, "auto")) *mode = TOOL_CHOICE_AUTO;
    else if (!strcmp(choice, "none")) *mode = TOOL_CHOICE_NONE;
    else if (!strcmp(choice, "required")) *mode = TOOL_CHOICE_REQUIRED;
    else {
        snprintf(err, errlen, "tool_choice=%s not supported", choice);
        ok = false;
    }
    free(choice);
    return ok;
}

static bool parse_anthropic_tool_choice_value(const char **p, int *mode,
                                              char *err, size_t errlen) {
    json_ws(p);
    if (**p != '{') {
        snprintf(err, errlen, "invalid tool_choice");
        return false;
    }
    (*p)++;
    char *type = NULL;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) goto bad;
        json_ws(p);
        if (**p != ':') { free(key); goto bad; }
        (*p)++;
        if (!strcmp(key, "type")) {
            free(type);
            type = NULL;
            if (!json_string(p, &type)) { free(key); goto bad; }
        } else if (!json_skip_value(p)) {
            free(key); goto bad;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') goto bad;
    (*p)++;
    if (!type) goto bad;
    if (!strcmp(type, "auto")) *mode = TOOL_CHOICE_AUTO;
    else if (!strcmp(type, "none")) *mode = TOOL_CHOICE_NONE;
    else if (!strcmp(type, "any")) *mode = TOOL_CHOICE_REQUIRED;
    else if (!strcmp(type, "tool")) {
        snprintf(err, errlen, "forced tool_choice not supported");
        free(type);
        return false;
    } else {
        snprintf(err, errlen, "tool_choice type=%s not supported", type);
        free(type);
        return false;
    }
    free(type);
    return true;
bad:
    free(type);
    if (!err[0]) snprintf(err, errlen, "invalid tool_choice");
    return false;
}

static bool parse_budget(const char **p, const char *key, request *r,
                         char *err, size_t errlen) {
    double budget = 0.0;
    if (!json_number(p, &budget)) return false;
    if (budget < 0) {
        snprintf(err, errlen, "%s must be >= 0", key);
        return false;
    }
    r->max_tokens = budget > (double)INT_MAX ? INT_MAX : (int)budget;
    r->max_tokens_set = true;
    return true;
}

static bool role_is_system(const char *role) {
    return role && (!strcmp(role, "system") || !strcmp(role, "developer"));
}

static bool chat_msg_is_model_tool_result(const chat_msg *m) {
    if (!m || !m->role) return false;
    if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) return true;
    return !strcmp(m->role, "user") &&
           ((m->tool_call_id && m->tool_call_id[0]) || m->tool_call_ids_len > 0);
}

static bool chat_history_has_pending_tool_results(const chat_msgs *msgs) {
    bool saw = false;
    for (int i = msgs ? msgs->len - 1 : -1; i >= 0; i--) {
        const chat_msg *m = &msgs->v[i];
        if (!m->role || role_is_system(m->role)) continue;
        if (chat_msg_is_model_tool_result(m)) { saw = true; continue; }
        if (!strcmp(m->role, "assistant")) return saw;
    }
    return saw;
}

static bool msg_has_call_id(const chat_msg *m, const char *id) {
    if (!m || !m->role || strcmp(m->role, "assistant") || !id || !id[0]) return false;
    for (int i = 0; i < m->calls.len; i++)
        if (m->calls.v[i].id && !strcmp(m->calls.v[i].id, id)) return true;
    return false;
}

static const chat_msg *find_prior_call(const chat_msgs *msgs, int before, const char *id) {
    for (int i = before - 1; i >= 0; i--)
        if (msg_has_call_id(&msgs->v[i], id)) return &msgs->v[i];
    return NULL;
}

static void collect_ids(const chat_msg *m, stop_list *ids) {
    if (m->tool_call_id && m->tool_call_id[0]) stop_list_push(ids, xstrdup(m->tool_call_id));
    for (int i = 0; i < m->tool_call_ids_len; i++)
        stop_list_push(ids, xstrdup(m->tool_call_ids[i]));
}

static bool prepare_tool_choice(request *r, bool have_schemas, char *err, size_t errlen) {
    r->has_tools = have_schemas && r->tool_choice != TOOL_CHOICE_NONE;
    if (r->tool_choice == TOOL_CHOICE_REQUIRED && !have_schemas) {
        snprintf(err, errlen, "tool_choice=required requires at least one tool");
        return false;
    }
    return true;
}

static bool anthropic_validate(const chat_msgs *msgs, bool *live, char *err, size_t errlen) {
    *live = false;
    if (!msgs) return true;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!m->role || strcmp(m->role, "user")) continue;
        if ((!m->tool_call_id || !m->tool_call_id[0]) && m->tool_call_ids_len == 0) continue;
        stop_list ids = {0};
        collect_ids(m, &ids);
        for (int j = 0; j < ids.len; j++) {
            if (!find_prior_call(msgs, i, ids.v[j])) {
                snprintf(err, errlen,
                         "Anthropic continuation state is not available for tool_use_id %s; retry by replaying the full messages history",
                         ids.v[j]);
                stop_list_clear(&ids);
                return false;
            }
        }
        stop_list_clear(&ids);
    }
    return true;
}

static bool responses_validate(const chat_msgs *msgs, int think, bool *lt, bool *lr,
                               char *err, size_t errlen) {
    *lt = false;
    *lr = false;
    if (!msgs) return true;
    bool needs_r = think_mode_enabled(think);
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!m->role || (strcmp(m->role, "tool") && strcmp(m->role, "function"))) continue;
        stop_list ids = {0};
        collect_ids(m, &ids);
        for (int j = 0; j < ids.len; j++) {
            const chat_msg *prior = find_prior_call(msgs, i, ids.v[j]);
            if (!prior) {
                snprintf(err, errlen,
                         "Responses continuation state is not available for call_id %s; retry by replaying the full input history",
                         ids.v[j]);
                stop_list_clear(&ids);
                return false;
            }
            if (needs_r && (!prior->reasoning || !prior->reasoning[0])) *lr = true;
        }
        stop_list_clear(&ids);
    }
    return true;
}

static uint32_t compute_needs(const request *r) {
    uint32_t n = 0;
    if (r->stream) n |= DS4_NEED_STREAMING;
    if (r->temperature > 0.0f) n |= DS4_NEED_PER_ROW_SAMPLING;
    if (think_mode_enabled(r->think_mode)) n |= DS4_NEED_THINKING;
    if (r->stops.len > 0) n |= DS4_NEED_STOP_SCAN;
    if (r->has_tools) n |= DS4_NEED_TOOL_SCAN;
    if (r->return_token_ids) n |= DS4_NEED_TOKEN_IDS;
    if (r->responses_requires_live_tool_state || r->responses_requires_live_reasoning ||
        r->anthropic_requires_live_tool_state) {
        n |= DS4_NEED_LIVE_FRONTIER;
    }
    if (r->has_tools && r->api != API_OPENAI) {
        n |= DS4_NEED_CONTINUATION_PUBLISH;
        if (!r->stream) n |= DS4_NEED_CORRECTIVE_RECOVERY;
    }
    if (r->api == API_ANTHROPIC && r->max_tokens_set && r->max_tokens <= 0)
        n |= DS4_NEED_PREFILL_ONLY;
    return n;
}

static void request_init(request *r, int kind, int max_tokens) {
    memset(r, 0, sizeof(*r));
    r->kind = kind;
    r->api = API_OPENAI;
    r->model = xstrdup("deepseek-v4-flash");
    r->max_tokens = max_tokens;
    r->temperature = server_default_temperature();
    r->top_p = DS4_DEFAULT_TOP_P;
    r->min_p = DS4_DEFAULT_MIN_P;
    r->think_mode = DS4_THINK_LOW;
    r->tool_choice = TOOL_CHOICE_AUTO;
}

static void request_free(request *r) {
    free(r->model);
    stop_list_clear(&r->stops);
    chat_msgs_free(&r->messages);
    free(r->tool_schemas);
    memset(r, 0, sizeof(*r));
}

static void apply_think(request *r, bool got, bool enabled, int effort) {
    if (!got && model_alias_disables_thinking(r->model)) enabled = false;
    if (!got && model_alias_enables_thinking(r->model)) enabled = true;
    r->think_mode = think_mode_from_enabled(enabled, effort);
}

static bool parse_responses_content_array(const char **p, char **out) {
    json_ws(p);
    if (**p == '"') return json_string(p, out);
    if (json_lit(p, "null")) { *out = xstrdup(""); return true; }
    if (**p != '[') return false;
    (*p)++;
    buf b = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p == '"') {
            char *s = NULL;
            if (!json_string(p, &s)) { buf_free(&b); return false; }
            buf_puts(&b, s);
            free(s);
        } else if (**p == '{') {
            (*p)++;
            char *type = NULL, *text = NULL;
            json_ws(p);
            while (**p && **p != '}') {
                char *key = NULL;
                if (!json_string(p, &key)) { free(type); free(text); buf_free(&b); return false; }
                json_ws(p);
                if (**p != ':') { free(key); free(type); free(text); buf_free(&b); return false; }
                (*p)++;
                if (!strcmp(key, "type")) {
                    free(type);
                    if (!json_string(p, &type)) { free(key); free(text); buf_free(&b); return false; }
                } else if (!strcmp(key, "text")) {
                    free(text);
                    json_ws(p);
                    if (json_lit(p, "null")) text = xstrdup("");
                    else if (!json_string(p, &text)) { free(key); free(type); buf_free(&b); return false; }
                } else if (!json_skip_value(p)) {
                    free(key); free(type); free(text); buf_free(&b); return false;
                }
                free(key);
                json_ws(p);
                if (**p == ',') (*p)++;
                json_ws(p);
            }
            if (**p != '}') { free(type); free(text); buf_free(&b); return false; }
            (*p)++;
            bool is_text = type && (!strcmp(type, "input_text") || !strcmp(type, "output_text") ||
                                    !strcmp(type, "text") || !strcmp(type, "summary_text") ||
                                    !strcmp(type, "reasoning_text"));
            if (!is_text || !text) { free(type); free(text); buf_free(&b); return false; }
            buf_puts(&b, text);
            free(type);
            free(text);
        } else {
            buf_free(&b);
            return false;
        }
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') { buf_free(&b); return false; }
    (*p)++;
    *out = buf_take(&b);
    return true;
}

static bool parse_responses_reasoning(const char **p, int *effort, bool *summary_opted,
                                      bool *effort_seen) {
    json_ws(p);
    if (json_lit(p, "null")) return true;
    if (**p != '{') return json_skip_value(p);
    (*p)++;
    json_ws(p);
    while (**p && **p != '}') {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        json_ws(p);
        if (**p != ':') { free(key); return false; }
        (*p)++;
        if (!strcmp(key, "effort")) {
            json_ws(p);
            if (!json_lit(p, "null")) {
                if (!parse_reasoning_effort_value(p, effort)) { free(key); return false; }
                if (effort_seen) *effort_seen = true;
            }
        } else if (!strcmp(key, "summary")) {
            json_ws(p);
            if (json_lit(p, "null")) {
            } else if (**p == '"') {
                char *mode = NULL;
                if (!json_string(p, &mode)) { free(key); return false; }
                if (summary_opted &&
                    (!strcmp(mode, "auto") || !strcmp(mode, "concise") || !strcmp(mode, "detailed")))
                    *summary_opted = true;
                free(mode);
            } else if (!json_skip_value(p)) {
                free(key); return false;
            }
        } else if (!json_skip_value(p)) {
            free(key); return false;
        }
        free(key);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != '}') return false;
    (*p)++;
    return true;
}

static bool parse_responses_input(const char **p, chat_msgs *msgs, buf *loaded) {
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    buf pending = {0};
    json_ws(p);
    while (**p && **p != ']') {
        if (**p != '{') { buf_free(&pending); return false; }
        (*p)++;
        char *type = NULL, *role = NULL, *content = NULL, *name = NULL;
        char *namespace = NULL, *call_id = NULL, *item_id = NULL, *arguments = NULL;
        char *output = NULL, *input_str = NULL, *summary = NULL, *action = NULL;
        char *result = NULL, *tools_json = NULL, *status_str = NULL;
        json_ws(p);
        while (**p && **p != '}') {
            char *key = NULL;
            if (!json_string(p, &key)) goto item_fail;
            json_ws(p);
            if (**p != ':') { free(key); goto item_fail; }
            (*p)++;
            if (!strcmp(key, "type")) {
                free(type);
                if (!json_string(p, &type)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "role")) {
                free(role);
                if (!json_string(p, &role)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "content")) {
                free(content);
                if (!parse_responses_content_array(p, &content)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "name")) {
                free(name);
                if (!json_string(p, &name)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "namespace")) {
                free(namespace);
                if (!json_string(p, &namespace)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "call_id")) {
                free(call_id);
                if (!json_string(p, &call_id)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "id")) {
                free(item_id);
                if (!json_string(p, &item_id)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "arguments")) {
                free(arguments);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &arguments)) { free(key); goto item_fail; }
                } else if (!json_raw_value(p, &arguments)) {
                    free(key); goto item_fail;
                }
            } else if (!strcmp(key, "output")) {
                free(output);
                json_ws(p);
                if (**p == '[') {
                    if (!parse_responses_content_array(p, &output)) { free(key); goto item_fail; }
                } else if (**p == '"') {
                    if (!json_string(p, &output)) { free(key); goto item_fail; }
                } else if (!json_raw_value(p, &output)) {
                    free(key); goto item_fail;
                }
            } else if (!strcmp(key, "input")) {
                free(input_str);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &input_str)) { free(key); goto item_fail; }
                } else if (!json_raw_value(p, &input_str)) {
                    free(key); goto item_fail;
                }
            } else if (!strcmp(key, "summary")) {
                free(summary);
                if (!parse_responses_content_array(p, &summary)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "action")) {
                free(action);
                if (!json_raw_value(p, &action)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "result")) {
                free(result);
                json_ws(p);
                if (**p == '"') {
                    if (!json_string(p, &result)) { free(key); goto item_fail; }
                } else if (!json_raw_value(p, &result)) {
                    free(key); goto item_fail;
                }
            } else if (!strcmp(key, "status")) {
                free(status_str);
                if (!json_string(p, &status_str)) { free(key); goto item_fail; }
            } else if (!strcmp(key, "tools")) {
                free(tools_json);
                if (!json_raw_value(p, &tools_json)) { free(key); goto item_fail; }
            } else if (!json_skip_value(p)) {
                free(key); goto item_fail;
            }
            free(key);
            json_ws(p);
            if (**p == ',') (*p)++;
            json_ws(p);
            continue;
        item_fail:
            free(type); free(role); free(content); free(name); free(namespace);
            free(call_id); free(item_id); free(arguments); free(output);
            free(input_str); free(summary); free(action); free(result);
            free(tools_json); free(status_str);
            buf_free(&pending);
            return false;
        }
        if (**p != '}') {
            free(type); free(role); free(content); free(name); free(namespace);
            free(call_id); free(item_id); free(arguments); free(output);
            free(input_str); free(summary); free(action); free(result);
            free(tools_json); free(status_str);
            buf_free(&pending);
            return false;
        }
        (*p)++;
        if (status_str && status_str[0] && strcmp(status_str, "completed") != 0) {
            free(type); free(role); free(content); free(name); free(namespace);
            free(call_id); free(item_id); free(arguments); free(output);
            free(input_str); free(summary); free(action); free(result);
            free(tools_json); free(status_str);
            buf_free(&pending);
            return false;
        }
        const char *t = type ? type : "message";
        bool consumes = (!strcmp(t, "message") && role && !strcmp(role, "assistant")) ||
                        !strcmp(t, "function_call") || !strcmp(t, "custom_tool_call") ||
                        !strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
                        !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call");
        bool bookkeeping = !strcmp(t, "compaction") || !strcmp(t, "context_compaction");
        if (!consumes && !bookkeeping && pending.len) {
            chat_msg flush = {0};
            flush.role = xstrdup("assistant");
            flush.content = xstrdup("");
            flush.reasoning = buf_take(&pending);
            chat_msgs_push(msgs, flush);
        }
        if (!strcmp(t, "message")) {
            chat_msg msg = {0};
            msg.role = xstrdup(role ? role : "user");
            msg.content = content ? content : xstrdup("");
            content = NULL;
            if (!strcmp(msg.role, "assistant") && pending.len)
                msg.reasoning = buf_take(&pending);
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "function_call") || !strcmp(t, "custom_tool_call")) {
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            const char *args = arguments ? arguments : input_str ? input_str : "{}";
            tc.arguments = xstrdup(args);
            if (strcmp(t, "custom_tool_call") && namespace && namespace[0] && name && name[0]) {
                buf q = {0};
                buf_puts(&q, namespace);
                buf_puts(&q, name);
                tc.name = buf_take(&q);
            } else {
                tc.name = xstrdup(name ? name : "");
            }
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending.len) msg.reasoning = buf_take(&pending);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "function_call_output") || !strcmp(t, "custom_tool_call_output")) {
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            msg.content = output ? output : xstrdup("");
            output = NULL;
            if (call_id || item_id) chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            chat_msgs_push(msgs, msg);
        } else if (!strcmp(t, "reasoning")) {
            if (summary && summary[0]) {
                if (pending.len) buf_putc(&pending, '\n');
                buf_puts(&pending, summary);
            }
            if (content && content[0]) {
                if (pending.len) buf_putc(&pending, '\n');
                buf_puts(&pending, content);
            }
        } else if (!strcmp(t, "local_shell_call") || !strcmp(t, "web_search_call") ||
                   !strcmp(t, "tool_search_call") || !strcmp(t, "image_generation_call")) {
            tool_call tc = {0};
            tc.id = xstrdup(call_id ? call_id : item_id ? item_id : "");
            if (!strcmp(t, "tool_search_call")) tc.name = xstrdup("tool_search");
            else if (!strcmp(t, "local_shell_call")) tc.name = xstrdup("local_shell");
            else tc.name = xstrdup(t);
            const char *args = action ? action : arguments ? arguments : input_str ? input_str : "{}";
            tc.arguments = xstrdup(args);
            chat_msg *last = msgs->len ? &msgs->v[msgs->len - 1] : NULL;
            if (last && !strcmp(last->role, "assistant")) {
                if (pending.len && (!last->reasoning || !last->reasoning[0])) {
                    free(last->reasoning);
                    last->reasoning = buf_take(&pending);
                }
                tool_calls_push(&last->calls, tc);
            } else {
                chat_msg msg = {0};
                msg.role = xstrdup("assistant");
                msg.content = xstrdup("");
                if (pending.len) msg.reasoning = buf_take(&pending);
                tool_calls_push(&msg.calls, tc);
                chat_msgs_push(msgs, msg);
            }
        } else if (!strcmp(t, "local_shell_call_output") || !strcmp(t, "web_search_call_output") ||
                   !strcmp(t, "tool_search_output") || !strcmp(t, "tool_search_call_output") ||
                   !strcmp(t, "image_generation_call_output")) {
            if (!strcmp(t, "tool_search_output") && tools_json) {
                const char *tp = tools_json;
                char *schemas = NULL;
                if (!parse_tools_value(&tp, &schemas)) {
                    free(schemas);
                    free(type); free(role); free(content); free(name); free(namespace);
                    free(call_id); free(item_id); free(arguments); free(output);
                    free(input_str); free(summary); free(action); free(result);
                    free(tools_json); free(status_str);
                    buf_free(&pending);
                    return false;
                }
                if (schemas && schemas[0] && loaded) {
                    if (loaded->len) buf_putc(loaded, '\n');
                    buf_puts(loaded, schemas);
                }
                free(schemas);
            }
            chat_msg msg = {0};
            msg.role = xstrdup("tool");
            const char *body = output ? output : result ? result : tools_json ? tools_json : "";
            msg.content = xstrdup(body);
            if (call_id || item_id) chat_msg_add_tool_call_id(&msg, call_id ? call_id : item_id);
            chat_msgs_push(msgs, msg);
        } else if (!bookkeeping) {
            free(type); free(role); free(content); free(name); free(namespace);
            free(call_id); free(item_id); free(arguments); free(output);
            free(input_str); free(summary); free(action); free(result);
            free(tools_json); free(status_str);
            buf_free(&pending);
            return false;
        }
        free(type); free(role); free(content); free(name); free(namespace);
        free(call_id); free(item_id); free(arguments); free(output);
        free(input_str); free(summary); free(action); free(result);
        free(tools_json); free(status_str);
        json_ws(p);
        if (**p == ',') (*p)++;
        json_ws(p);
    }
    if (**p != ']') { buf_free(&pending); return false; }
    (*p)++;
    if (pending.len) {
        chat_msg msg = {0};
        msg.role = xstrdup("assistant");
        msg.content = xstrdup("");
        msg.reasoning = buf_take(&pending);
        chat_msgs_push(msgs, msg);
    }
    buf_free(&pending);
    return true;
}

static bool parse_chat_request(const char *body, int def_tokens, request *r,
                               char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    bool got_messages = false, got_thinking = false, thinking_enabled = true;
    int reasoning_effort = DS4_THINK_LOW;
    chat_msgs msgs = {0};
    char *tool_schemas = NULL;
    const char *p = body;
    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') { free(key); goto bad; }
        p++;
        bool ok = true;
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&msgs);
            ok = parse_messages(&p, &msgs);
            if (ok) got_messages = true;
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            ok = parse_tools_value(&p, &tool_schemas);
        } else if (!strcmp(key, "tool_choice")) {
            ok = parse_openai_tool_choice_value(&p, &r->tool_choice, err, errlen);
        } else if (!strcmp(key, "model")) {
            free(r->model);
            ok = json_string(&p, &r->model);
            if (ok) r->model_from_request = true;
        } else if (!strcmp(key, "max_tokens") || !strcmp(key, "max_completion_tokens")) {
            ok = parse_budget(&p, key, r, err, errlen);
        } else if (!strcmp(key, "temperature")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->top_p = (float)v;
        } else if (!strcmp(key, "min_p")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->min_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            ok = json_int(&p, &r->top_k);
        } else if (!strcmp(key, "seed")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->seed = v > 0.0 ? (uint64_t)v : 0;
        } else if (!strcmp(key, "stream")) {
            ok = json_bool(&p, &r->stream);
        } else if (!strcmp(key, "stream_options")) {
            ok = parse_stream_options(&p, &r->stream_include_usage);
        } else if (!strcmp(key, "return_token_ids")) {
            ok = json_bool(&p, &r->return_token_ids);
        } else if (!strcmp(key, "thinking")) {
            ok = parse_thinking_control_value(&p, &thinking_enabled);
            if (ok) got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            ok = parse_reasoning_effort_value(&p, &reasoning_effort);
        } else if (!strcmp(key, "think") || !strcmp(key, "enable_thinking")) {
            ok = json_bool(&p, &thinking_enabled);
            if (ok) got_thinking = true;
        } else if (!strcmp(key, "stop")) {
            ok = parse_stop(&p, &r->stops);
        } else if (!strcmp(key, "response_format")) {
            ok = parse_output_format_value(&p, "response_format", err, errlen);
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) goto bad;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        chat_msgs_free(&msgs);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    r->has_tool_results = chat_history_has_pending_tool_results(&msgs);
    if (!prepare_tool_choice(r, tool_schemas && tool_schemas[0], err, errlen)) {
        chat_msgs_free(&msgs);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    apply_think(r, got_thinking, thinking_enabled, reasoning_effort);
    r->messages = msgs;
    r->tool_schemas = tool_schemas;
    r->needs = compute_needs(r);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(tool_schemas);
    if (!err[0]) snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static bool parse_completion_request(const char *body, int def_tokens, request *r,
                                     char *err, size_t errlen) {
    request_init(r, REQ_COMPLETION, def_tokens);
    char *prompt = NULL;
    bool got_thinking = false, thinking_enabled = true;
    int reasoning_effort = DS4_THINK_LOW;
    const char *p = body;
    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') { free(key); goto bad; }
        p++;
        bool ok = true;
        if (!strcmp(key, "prompt")) {
            free(prompt);
            ok = parse_prompt(&p, &prompt);
        } else if (!strcmp(key, "model")) {
            free(r->model);
            ok = json_string(&p, &r->model);
            if (ok) r->model_from_request = true;
        } else if (!strcmp(key, "max_tokens")) {
            ok = parse_budget(&p, key, r, err, errlen);
        } else if (!strcmp(key, "temperature")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->top_p = (float)v;
        } else if (!strcmp(key, "min_p")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->min_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            ok = json_int(&p, &r->top_k);
        } else if (!strcmp(key, "seed")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->seed = v > 0.0 ? (uint64_t)v : 0;
        } else if (!strcmp(key, "stream")) {
            ok = json_bool(&p, &r->stream);
        } else if (!strcmp(key, "stream_options")) {
            ok = parse_stream_options(&p, &r->stream_include_usage);
        } else if (!strcmp(key, "return_token_ids")) {
            ok = json_bool(&p, &r->return_token_ids);
        } else if (!strcmp(key, "thinking")) {
            ok = parse_thinking_control_value(&p, &thinking_enabled);
            if (ok) got_thinking = true;
        } else if (!strcmp(key, "reasoning_effort")) {
            ok = parse_reasoning_effort_value(&p, &reasoning_effort);
        } else if (!strcmp(key, "think") || !strcmp(key, "enable_thinking")) {
            ok = json_bool(&p, &thinking_enabled);
            if (ok) got_thinking = true;
        } else if (!strcmp(key, "stop")) {
            ok = parse_stop(&p, &r->stops);
        } else if (!strcmp(key, "response_format")) {
            ok = parse_output_format_value(&p, "response_format", err, errlen);
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) goto bad;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!prompt) {
        snprintf(err, errlen, "missing prompt");
        request_free(r);
        return false;
    }
    apply_think(r, got_thinking, thinking_enabled, reasoning_effort);
    free(prompt);
    r->needs = compute_needs(r);
    return true;
bad:
    free(prompt);
    if (!err[0]) snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static bool parse_anthropic_request(const char *body, int def_tokens, request *r,
                                    char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    r->api = API_ANTHROPIC;
    bool got_messages = false, got_thinking = false, thinking_enabled = true;
    int reasoning_effort = DS4_THINK_LOW;
    chat_msgs msgs = {0};
    char *system = NULL, *tool_schemas = NULL;
    const char *p = body;
    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') { free(key); goto bad; }
        p++;
        bool ok = true;
        if (!strcmp(key, "messages")) {
            chat_msgs_free(&msgs);
            ok = parse_anthropic_messages(&p, &msgs);
            if (ok) got_messages = true;
        } else if (!strcmp(key, "system")) {
            free(system);
            ok = parse_anthropic_system(&p, &system);
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            ok = parse_tools_value(&p, &tool_schemas);
        } else if (!strcmp(key, "tool_choice")) {
            ok = parse_anthropic_tool_choice_value(&p, &r->tool_choice, err, errlen);
        } else if (!strcmp(key, "model")) {
            free(r->model);
            ok = json_string(&p, &r->model);
            if (ok) r->model_from_request = true;
        } else if (!strcmp(key, "max_tokens")) {
            ok = parse_budget(&p, key, r, err, errlen);
        } else if (!strcmp(key, "temperature")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->top_p = (float)v;
        } else if (!strcmp(key, "top_k")) {
            ok = json_int(&p, &r->top_k);
        } else if (!strcmp(key, "stream")) {
            ok = json_bool(&p, &r->stream);
        } else if (!strcmp(key, "stop_sequences")) {
            ok = parse_stop(&p, &r->stops);
        } else if (!strcmp(key, "thinking")) {
            ok = parse_thinking_control_value(&p, &thinking_enabled);
            if (ok) got_thinking = true;
        } else if (!strcmp(key, "output_config")) {
            ok = parse_output_config_effort(&p, &reasoning_effort, err, errlen);
        } else if (!strcmp(key, "output_format")) {
            ok = parse_output_format_value(&p, "output_format", err, errlen);
        } else if (!strcmp(key, "reasoning_effort")) {
            ok = parse_reasoning_effort_value(&p, &reasoning_effort);
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) goto bad;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_messages) {
        snprintf(err, errlen, "missing messages");
        chat_msgs_free(&msgs);
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    if (system && system[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = system;
        system = NULL;
        chat_msgs_push(&msgs, msg);
    }
    r->has_tool_results = chat_history_has_pending_tool_results(&msgs);
    if (!prepare_tool_choice(r, tool_schemas && tool_schemas[0], err, errlen)) {
        chat_msgs_free(&msgs);
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    apply_think(r, got_thinking, thinking_enabled, reasoning_effort);
    if (!anthropic_validate(&msgs, &r->anthropic_requires_live_tool_state, err, errlen)) {
        chat_msgs_free(&msgs);
        free(system);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    r->messages = msgs;
    r->tool_schemas = tool_schemas;
    free(system);
    r->needs = compute_needs(r);
    return true;
bad:
    chat_msgs_free(&msgs);
    free(system);
    free(tool_schemas);
    if (!err[0]) snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static bool parse_responses_request(const char *body, int def_tokens, request *r,
                                    char *err, size_t errlen) {
    request_init(r, REQ_CHAT, def_tokens);
    r->api = API_RESPONSES;
    bool got_input = false, got_thinking = false, thinking_enabled = true;
    int reasoning_effort = DS4_THINK_LOW;
    chat_msgs msgs = {0};
    buf loaded = {0};
    char *instructions = NULL, *tool_schemas = NULL;
    const char *p = body;
    json_ws(&p);
    if (*p != '{') goto bad;
    p++;
    json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!json_string(&p, &key)) goto bad;
        json_ws(&p);
        if (*p != ':') { free(key); goto bad; }
        p++;
        bool ok = true;
        if (!strcmp(key, "input")) {
            chat_msgs_free(&msgs);
            json_ws(&p);
            if (*p == '"') {
                char *plain = NULL;
                ok = json_string(&p, &plain);
                if (ok) {
                    chat_msg msg = {0};
                    msg.role = xstrdup("user");
                    msg.content = plain;
                    chat_msgs_push(&msgs, msg);
                    got_input = true;
                }
            } else {
                ok = parse_responses_input(&p, &msgs, &loaded);
                if (ok) got_input = true;
            }
        } else if (!strcmp(key, "instructions")) {
            free(instructions);
            instructions = NULL;
            json_ws(&p);
            if (json_lit(&p, "null")) {
                instructions = xstrdup("");
            } else {
                ok = json_string(&p, &instructions);
            }
        } else if (!strcmp(key, "tools")) {
            free(tool_schemas);
            tool_schemas = NULL;
            ok = parse_tools_value(&p, &tool_schemas);
        } else if (!strcmp(key, "tool_choice")) {
            ok = parse_openai_tool_choice_value(&p, &r->tool_choice, err, errlen);
        } else if (!strcmp(key, "model")) {
            free(r->model);
            ok = json_string(&p, &r->model);
            if (ok) r->model_from_request = true;
        } else if (!strcmp(key, "max_output_tokens") || !strcmp(key, "max_tokens")) {
            ok = parse_budget(&p, key, r, err, errlen);
        } else if (!strcmp(key, "temperature")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->temperature = (float)v;
        } else if (!strcmp(key, "top_p")) {
            double v = 0;
            ok = json_number(&p, &v);
            if (ok) r->top_p = (float)v;
        } else if (!strcmp(key, "stream")) {
            ok = json_bool(&p, &r->stream);
        } else if (!strcmp(key, "reasoning")) {
            bool effort_seen = false;
            ok = parse_responses_reasoning(&p, &reasoning_effort,
                                           &r->reasoning_summary_emit, &effort_seen);
            if (ok && effort_seen) {
                got_thinking = true;
                if (reasoning_effort == DS4_THINK_NONE) thinking_enabled = false;
            }
        } else if (!strcmp(key, "text")) {
            ok = parse_responses_text_value(&p, err, errlen);
        } else if (!strcmp(key, "previous_response_id") || !strcmp(key, "conversation")) {
            json_ws(&p);
            if (!json_lit(&p, "null")) {
                snprintf(err, errlen, "%s is not supported; replay full input instead", key);
                free(key);
                chat_msgs_free(&msgs);
                buf_free(&loaded);
                free(instructions);
                free(tool_schemas);
                request_free(r);
                return false;
            }
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) goto bad;
        json_ws(&p);
        if (*p == ',') p++;
        json_ws(&p);
    }
    if (*p != '}') goto bad;
    if (!got_input) {
        snprintf(err, errlen, "missing input");
        chat_msgs_free(&msgs);
        buf_free(&loaded);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    if (instructions && instructions[0]) {
        chat_msg msg = {0};
        msg.role = xstrdup("system");
        msg.content = instructions;
        instructions = NULL;
        chat_msgs_push(&msgs, msg);
        if (msgs.len > 1) {
            chat_msg tmp = msgs.v[msgs.len - 1];
            for (int i = msgs.len - 1; i > 0; i--) msgs.v[i] = msgs.v[i - 1];
            msgs.v[0] = tmp;
        }
    }
    buf combined = {0};
    if (tool_schemas && tool_schemas[0]) buf_puts(&combined, tool_schemas);
    if (loaded.len) {
        if (combined.len) buf_putc(&combined, '\n');
        buf_puts(&combined, loaded.ptr);
    }
    r->has_tool_results = chat_history_has_pending_tool_results(&msgs);
    if (!prepare_tool_choice(r, combined.len != 0, err, errlen)) {
        chat_msgs_free(&msgs);
        buf_free(&combined);
        buf_free(&loaded);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    apply_think(r, got_thinking, thinking_enabled, reasoning_effort);
    if (!responses_validate(&msgs, r->think_mode, &r->responses_requires_live_tool_state,
                            &r->responses_requires_live_reasoning, err, errlen)) {
        chat_msgs_free(&msgs);
        buf_free(&combined);
        buf_free(&loaded);
        free(instructions);
        free(tool_schemas);
        request_free(r);
        return false;
    }
    r->messages = msgs;
    r->tool_schemas = buf_take(&combined);
    buf_free(&loaded);
    free(instructions);
    free(tool_schemas);
    r->needs = compute_needs(r);
    return true;
bad:
    chat_msgs_free(&msgs);
    buf_free(&loaded);
    free(instructions);
    free(tool_schemas);
    if (!err[0]) snprintf(err, errlen, "invalid JSON request");
    request_free(r);
    return false;
}

static void dump_request(const request *r) {
    printf("OK\n");
    printf("kind=%d api=%d model=%s from_req=%d\n",
           r->kind, r->api, r->model ? r->model : "", r->model_from_request ? 1 : 0);
    printf("max_tokens=%d max_set=%d top_k=%d seed=%llu\n",
           r->max_tokens, r->max_tokens_set ? 1 : 0, r->top_k,
           (unsigned long long)r->seed);
    printf("temp=%.6f top_p=%.6f min_p=%.6f\n",
           (double)r->temperature, (double)r->top_p, (double)r->min_p);
    printf("stream=%d usage=%d echo=%d think=%d tools=%d tool_results=%d choice=%d\n",
           r->stream ? 1 : 0, r->stream_include_usage ? 1 : 0,
           r->return_token_ids ? 1 : 0, r->think_mode, r->has_tools ? 1 : 0,
           r->has_tool_results ? 1 : 0, r->tool_choice);
    printf("stops=%d summary=%d needs=%u nmsg=%d\n",
           r->stops.len, r->reasoning_summary_emit ? 1 : 0, r->needs, r->messages.len);
    for (int i = 0; i < r->messages.len && i < 8; i++) {
        const chat_msg *m = &r->messages.v[i];
        printf("msg%d_role=%s msg%d_ncalls=%d\n",
               i, m->role ? m->role : "", i, m->calls.len);
    }
}

static const char *join_from(int argc, char **argv, int start) {
    static char *acc = NULL;
    free(acc);
    acc = NULL;
    size_t n = 0;
    for (int i = start; i < argc; i++) n += strlen(argv[i]) + 1;
    acc = xmalloc(n + 1);
    acc[0] = 0;
    for (int i = start; i < argc; i++) {
        if (i > start) strcat(acc, " ");
        strcat(acc, argv[i]);
    }
    return acc;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: parse_c_oracle <cmd> ...");
    const char *cmd = argv[1];
    int def = 393216;
    const char *body;
    if (!strcmp(cmd, "chat") || !strcmp(cmd, "completion") ||
        !strcmp(cmd, "anthropic") || !strcmp(cmd, "responses")) {
        if (argc < 3) die("surface BODY");
        int argi = 2;
        if (argc >= 4 && argv[2][0] >= '0' && argv[2][0] <= '9' &&
            strspn(argv[2], "0123456789") == strlen(argv[2])) {
            def = atoi(argv[2]);
            argi = 3;
        }
        body = join_from(argc, argv, argi);
        request r;
        char err[256] = {0};
        bool ok = false;
        if (!strcmp(cmd, "chat")) ok = parse_chat_request(body, def, &r, err, sizeof(err));
        else if (!strcmp(cmd, "completion"))
            ok = parse_completion_request(body, def, &r, err, sizeof(err));
        else if (!strcmp(cmd, "anthropic"))
            ok = parse_anthropic_request(body, def, &r, err, sizeof(err));
        else
            ok = parse_responses_request(body, def, &r, err, sizeof(err));
        if (!ok) {
            printf("ERROR\n%s\n", err);
            return 0;
        }
        dump_request(&r);
        request_free(&r);
        return 0;
    }
    if (!strcmp(cmd, "tool-choice-openai") && argc >= 3) {
        const char *p = join_from(argc, argv, 2);
        int mode = 0;
        char err[160] = {0};
        if (parse_openai_tool_choice_value(&p, &mode, err, sizeof(err)))
            printf("OK\n%d\n", mode);
        else
            printf("ERROR\n%s\n", err);
        return 0;
    }
    if (!strcmp(cmd, "tool-choice-anthropic") && argc >= 3) {
        const char *p = join_from(argc, argv, 2);
        int mode = 0;
        char err[160] = {0};
        if (parse_anthropic_tool_choice_value(&p, &mode, err, sizeof(err)))
            printf("OK\n%d\n", mode);
        else
            printf("ERROR\n%s\n", err);
        return 0;
    }
    die("unknown cmd");
    return 2;
}