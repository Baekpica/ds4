/* C HTTP-door oracle. Copied from ds4_server.c at v0.6.3-dfm so Rust can
 * compare escape / envelopes / format refusal / model-id / models JSON
 * without linking the server. */

#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSON_MAX_NESTING 256

typedef struct { char *ptr; size_t len, cap; } buf;

static void die(const char *m) {
    fprintf(stderr, "server_c_oracle: %s\n", m);
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

static void buf_printf(buf *b, const char *fmt, ...) {
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) die("printf");
    if ((size_t)n < sizeof(tmp)) {
        buf_puts(b, tmp);
        return;
    }
    char *big = malloc((size_t)n + 1);
    if (!big) die("oom");
    va_start(ap, fmt);
    vsnprintf(big, (size_t)n + 1, fmt, ap);
    va_end(ap);
    buf_puts(b, big);
    free(big);
}

static void buf_free(buf *b) { free(b->ptr); memset(b, 0, sizeof(*b)); }

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
    *out = b.ptr ? b.ptr : calloc(1, 1);
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

static void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') buf_puts(b, "\\n");
        else if (c == '\r') buf_puts(b, "\\r");
        else if (c == '\t') buf_puts(b, "\\t");
        else if (c < 0x20) buf_printf(b, "\\u%04x", (unsigned)c);
        else buf_putc(b, (char)c);
    }
    buf_putc(b, '"');
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
        const bool ok = output_format_type_supported(field, type, err, errlen);
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

enum { DS4_THINK_NONE, DS4_THINK_LOW, DS4_THINK_HIGH, DS4_THINK_MAX };

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
            if (!parse_reasoning_effort_value(p, effort)) { free(key); return false; }
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

static const char *http_reason(int code) {
    return code == 200 ? "OK" :
           code == 204 ? "No Content" :
           code == 400 ? "Bad Request" :
           code == 404 ? "Not Found" :
           code == 409 ? "Conflict" :
           code == 429 ? "Too Many Requests" :
           code == 500 ? "Internal Server Error" :
           code == 503 ? "Service Unavailable" : "Error";
}

static const char *openai_error_type(int code) {
    if (code == 429) return "rate_limit_error";
    if (code >= 500) return "server_error";
    return "invalid_request_error";
}

static const char *anthropic_error_type(int code) {
    if (code == 429) return "rate_limit_error";
    if (code == 404) return "not_found_error";
    if (code == 503) return "overloaded_error";
    if (code >= 500) return "api_error";
    return "invalid_request_error";
}

static void openai_error_body(buf *b, int code, const char *msg) {
    buf_puts(b, "{\"error\":{\"message\":");
    json_escape(b, msg);
    buf_puts(b, code == 429 ? ",\"type\":\"rate_limit_error\"}}\n"
               : code >= 500 ? ",\"type\":\"server_error\"}}\n"
                             : ",\"type\":\"invalid_request_error\"}}\n");
}

static void anthropic_error_body(buf *b, int code, const char *msg) {
    buf_puts(b, "{\"type\":\"error\",\"error\":{\"type\":\"");
    buf_puts(b, anthropic_error_type(code));
    buf_puts(b, "\",\"message\":");
    json_escape(b, msg);
    buf_puts(b, "}}\n");
}

static void append_cors_headers(buf *h) {
    buf_puts(h,
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n");
}

static void http_head(buf *h, int code, const char *type,
                      const char *extra, int cors, size_t body_len) {
    buf_printf(h, "HTTP/1.1 %d %s\r\nContent-Length: %zu\r\n",
               code, http_reason(code), body_len);
    if (type && type[0]) {
        buf_puts(h, "Content-Type: ");
        buf_puts(h, type);
        buf_puts(h, "\r\n");
    }
    if (extra && extra[0]) buf_puts(h, extra);
    if (cors) append_cors_headers(h);
    buf_puts(h, "Connection: close\r\n\r\n");
}

static bool server_parent_is_generic_dir(const char *parent) {
    return !parent || !parent[0] ||
           !strcmp(parent, ".") || !strcmp(parent, "..") ||
           !strcmp(parent, "gguf") || !strcmp(parent, "GGUF") ||
           !strcmp(parent, "models") || !strcmp(parent, "model") ||
           !strcmp(parent, "weights") || !strcmp(parent, "artifacts") ||
           !strcmp(parent, "tmp") || !strcmp(parent, "temp") ||
           !strcmp(parent, "scratch") || !strcmp(parent, "data");
}

static bool server_parent_is_gguf_artifact_dir(const char *parent) {
    size_t n;
    if (server_parent_is_generic_dir(parent)) return false;
    if (strstr(parent, "Mixed-Quant")) return true;
    n = strlen(parent);
    return n > 4 &&
           (parent[n - 4] == 'G' || parent[n - 4] == 'g') &&
           (parent[n - 3] == 'G' || parent[n - 3] == 'g') &&
           (parent[n - 2] == 'U' || parent[n - 2] == 'u') &&
           (parent[n - 1] == 'F' || parent[n - 1] == 'f');
}

static void server_strip_gguf_ext(char *name) {
    size_t n = strlen(name);
    if (n >= 5 && name[n - 5] == '.' &&
        (name[n - 4] == 'g' || name[n - 4] == 'G') &&
        (name[n - 3] == 'g' || name[n - 3] == 'G') &&
        (name[n - 2] == 'u' || name[n - 2] == 'U') &&
        (name[n - 1] == 'f' || name[n - 1] == 'F'))
        name[n - 5] = '\0';
}

static void server_strip_gguf_shard_suffix(char *name) {
    size_t n = strlen(name);
    size_t i = n;
    while (i > 0 && name[i - 1] >= '0' && name[i - 1] <= '9') i--;
    if (i == n || i < 4 || strncmp(name + i - 4, "-of-", 4) != 0) return;
    size_t of = i - 4;
    size_t d = of;
    while (d > 0 && name[d - 1] >= '0' && name[d - 1] <= '9') d--;
    if (d == of || d == 0 || name[d - 1] != '-') return;
    name[d - 1] = '\0';
}

static const char *server_model_id_from_gguf_path(const char *path,
                                                 char *dst, size_t dst_n) {
    const char *end, *base, *parent_end, *parent;
    size_t n, parent_len, base_len;
    char stem[256];

    if (!path || !path[0] || !dst || dst_n == 0) return NULL;
    n = strlen(path);
    while (n > 1 && path[n - 1] == '/') n--;
    end = path + n;
    base = end;
    while (base > path && base[-1] != '/') base--;
    base_len = (size_t)(end - base);
    if (base_len == 0 || base_len >= sizeof(stem)) return NULL;
    memcpy(stem, base, base_len);
    stem[base_len] = '\0';
    server_strip_gguf_ext(stem);
    server_strip_gguf_shard_suffix(stem);
    if (!stem[0]) return NULL;

    parent = NULL;
    parent_len = 0;
    if (base > path) {
        parent_end = base - 1;
        parent = parent_end;
        while (parent > path && parent[-1] != '/') parent--;
        parent_len = (size_t)(parent_end - parent);
    }
    if (parent_len > 0 && parent_len < dst_n) {
        char parent_name[256];
        if (parent_len < sizeof(parent_name)) {
            memcpy(parent_name, parent, parent_len);
            parent_name[parent_len] = '\0';
            if (server_parent_is_gguf_artifact_dir(parent_name)) {
                memcpy(dst, parent_name, parent_len + 1);
                return dst;
            }
        }
    }
    if (strlen(stem) >= dst_n) return NULL;
    memcpy(dst, stem, strlen(stem) + 1);
    return dst;
}

static void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    buf_puts(b, "{\"id\":");
    json_escape(b, id);
    buf_puts(b,
        ",\"object\":\"model\","
        "\"created\":1767225600,"
        "\"owned_by\":\"ds4.c\","
        "\"name\":");
    json_escape(b, name);
    buf_printf(b,
        ","
        "\"context_length\":%d,"
        "\"top_provider\":{"
            "\"context_length\":%d,"
            "\"max_completion_tokens\":%d,"
            "\"is_moderated\":false},"
        "\"supported_parameters\":["
            "\"tools\","
            "\"tool_choice\","
            "\"max_tokens\","
            "\"temperature\","
            "\"top_p\","
            "\"top_k\","
            "\"min_p\","
            "\"stop\","
            "\"seed\","
            "\"stream\","
            "\"reasoning_effort\"]}",
        ctx, ctx, max_completion);
}

static char *json_models_array_dup(const char *text) {
    const char *p = text;
    while ((p = strstr(p, "\"models\"")) != NULL) {
        const char *q = p + strlen("\"models\"");
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q != ':') { p = q; continue; }
        q++;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q != '[') { p = q; continue; }
        const char *start = q;
        int depth = 0;
        bool in_str = false, esc = false;
        for (; *q; q++) {
            const char c = *q;
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') in_str = true;
            else if (c == '[') depth++;
            else if (c == ']' && --depth == 0)
                return strndup(start, (size_t)(q + 1 - start));
        }
        return NULL;
    }
    return NULL;
}

static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n')
            return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

static bool header_accepts_json(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 7 && strncasecmp(line, "Accept:", 7) == 0) {
            for (size_t i = 7; i + 16 <= len; i++)
                if (strncasecmp(line + i, "application/json", 16) == 0) return true;
        }
        if (p < end) p++;
    }
    return false;
}

static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            return strtol(v, NULL, 10);
        }
        if (p < end) p++;
    }
    return 0;
}

static bool header_chunked(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 18 && strncasecmp(line, "Transfer-Encoding:", 18) == 0) {
            for (size_t i = 18; i + 7 <= len; i++)
                if (strncasecmp(line + i, "chunked", 7) == 0) return true;
        }
        if (p < end) p++;
    }
    return false;
}

static bool decode_hex(const char *hex, buf *out) {
    size_t n = strlen(hex);
    if (n % 2) return false;
    for (size_t i = 0; i < n; i += 2) {
        int hi = json_hex(hex[i]), lo = json_hex(hex[i + 1]);
        if (hi < 0 || lo < 0) return false;
        buf_putc(out, (char)((hi << 4) | lo));
    }
    return true;
}

static const char *join_from(int argc, char **argv, int i) {
    static char acc[8192];
    acc[0] = 0;
    size_t n = 0;
    for (; i < argc; i++) {
        size_t L = strlen(argv[i]);
        if (n + L + 2 >= sizeof(acc)) die("arg too long");
        if (n) acc[n++] = ' ';
        memcpy(acc + n, argv[i], L);
        n += L;
        acc[n] = 0;
    }
    return acc;
}

static void print_ok_err(bool ok, const char *err) {
    if (ok) printf("OK\n");
    else printf("ERROR\n%s\n", err ? err : "");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: server_c_oracle <cmd> ...\n");
        return 2;
    }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "escape")) {
        buf b = {0};
        json_escape(&b, argc > 2 ? join_from(argc, argv, 2) : "");
        fwrite(b.ptr, 1, b.len, stdout);
        buf_free(&b);
        return 0;
    }
    if (!strcmp(cmd, "openai-error") && argc >= 4) {
        buf b = {0};
        openai_error_body(&b, atoi(argv[2]), join_from(argc, argv, 3));
        fwrite(b.ptr, 1, b.len, stdout);
        buf_free(&b);
        return 0;
    }
    if (!strcmp(cmd, "anth-error") && argc >= 4) {
        buf b = {0};
        anthropic_error_body(&b, atoi(argv[2]), join_from(argc, argv, 3));
        fwrite(b.ptr, 1, b.len, stdout);
        buf_free(&b);
        return 0;
    }
    if (!strcmp(cmd, "http-head") && argc >= 7) {
        int code = atoi(argv[2]);
        const char *type = strcmp(argv[3], "-") ? argv[3] : "";
        int cors = atoi(argv[4]);
        const char *extra = strcmp(argv[5], "-") ? argv[5] : "";
        size_t blen = (size_t)strtoul(argv[6], NULL, 10);
        buf h = {0};
        http_head(&h, code, type, extra, cors, blen);
        fwrite(h.ptr, 1, h.len, stdout);
        buf_free(&h);
        return 0;
    }
    if (!strcmp(cmd, "wire-http") && argc >= 6) {
        int surf = atoi(argv[2]);
        int code = atoi(argv[3]);
        int cors = atoi(argv[4]);
        int retry = atoi(argv[5]);
        const char *msg = join_from(argc, argv, 6);
        buf body = {0};
        if (surf == 2) anthropic_error_body(&body, code, msg);
        else openai_error_body(&body, code, msg);
        char extra[48] = {0};
        if (retry >= 0) snprintf(extra, sizeof(extra), "Retry-After: %d\r\n", retry);
        buf h = {0};
        http_head(&h, code, "application/json", extra[0] ? extra : "", cors, body.len);
        fwrite(h.ptr, 1, h.len, stdout);
        fwrite(body.ptr, 1, body.len, stdout);
        buf_free(&h);
        buf_free(&body);
        return 0;
    }
    if (!strcmp(cmd, "format-type") && argc >= 4) {
        char err[256] = {0};
        bool ok = output_format_type_supported(argv[2], argv[3], err, sizeof(err));
        print_ok_err(ok, err);
        return 0;
    }
    if (!strcmp(cmd, "format-value") && argc >= 4) {
        char err[256] = {0};
        const char *p = join_from(argc, argv, 3);
        bool ok = parse_output_format_value(&p, argv[2], err, sizeof(err));
        print_ok_err(ok, err);
        return 0;
    }
    if (!strcmp(cmd, "text-value") && argc >= 3) {
        char err[256] = {0};
        const char *p = join_from(argc, argv, 2);
        bool ok = parse_responses_text_value(&p, err, sizeof(err));
        print_ok_err(ok, err);
        return 0;
    }
    if (!strcmp(cmd, "output-config") && argc >= 3) {
        char err[256] = {0};
        int effort = 0;
        const char *p = join_from(argc, argv, 2);
        bool ok = parse_output_config_effort(&p, &effort, err, sizeof(err));
        print_ok_err(ok, err);
        return 0;
    }
    if (!strcmp(cmd, "model-id") && argc >= 3) {
        char dst[256];
        const char *id = server_model_id_from_gguf_path(argv[2], dst, sizeof(dst));
        if (!id) { printf("NULL\n"); return 0; }
        printf("%s\n", id);
        return 0;
    }
    if (!strcmp(cmd, "models-list") && argc >= 6) {
        buf b = {0};
        buf_puts(&b, "{\"object\":\"list\",\"data\":[");
        append_model_json_values(&b, argv[2], argv[3], atoi(argv[4]), atoi(argv[5]));
        buf_puts(&b, "]");
        if (argc >= 7) {
            buf_puts(&b, ",\"models\":");
            buf_puts(&b, argv[6]);
        }
        buf_puts(&b, "}\n");
        fwrite(b.ptr, 1, b.len, stdout);
        buf_free(&b);
        return 0;
    }
    if (!strcmp(cmd, "model-one") && argc >= 6) {
        buf b = {0};
        append_model_json_values(&b, argv[2], argv[3], atoi(argv[4]), atoi(argv[5]));
        buf_putc(&b, '\n');
        fwrite(b.ptr, 1, b.len, stdout);
        buf_free(&b);
        return 0;
    }
    if (!strcmp(cmd, "models-array") && argc >= 3) {
        char *arr = json_models_array_dup(join_from(argc, argv, 2));
        if (!arr) { printf("NULL\n"); return 0; }
        fwrite(arr, 1, strlen(arr), stdout);
        free(arr);
        return 0;
    }
    if (!strcmp(cmd, "header-end") && argc >= 3) {
        buf h = {0};
        if (!decode_hex(argv[2], &h)) die("hex");
        printf("%zd\n", header_end(h.ptr, h.len));
        buf_free(&h);
        return 0;
    }
    if (!strcmp(cmd, "content-length") && argc >= 3) {
        buf h = {0};
        if (!decode_hex(argv[2], &h)) die("hex");
        printf("%ld\n", content_length(h.ptr, h.len));
        buf_free(&h);
        return 0;
    }
    if (!strcmp(cmd, "header-chunked") && argc >= 3) {
        buf h = {0};
        if (!decode_hex(argv[2], &h)) die("hex");
        printf("%d\n", header_chunked(h.ptr, h.len) ? 1 : 0);
        buf_free(&h);
        return 0;
    }
    if (!strcmp(cmd, "accept-json") && argc >= 3) {
        buf h = {0};
        if (!decode_hex(argv[2], &h)) die("hex");
        printf("%d\n", header_accepts_json(h.ptr, h.len) ? 1 : 0);
        buf_free(&h);
        return 0;
    }
    if (!strcmp(cmd, "openai-type") && argc >= 3) {
        printf("%s\n", openai_error_type(atoi(argv[2])));
        return 0;
    }
    if (!strcmp(cmd, "anth-type") && argc >= 3) {
        printf("%s\n", anthropic_error_type(atoi(argv[2])));
        return 0;
    }
    fprintf(stderr, "server_c_oracle: unknown cmd %s\n", cmd);
    return 2;
}
