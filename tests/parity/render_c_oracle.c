/* C family render oracle. Copied from ds4_server.c / ds4.c at
 * v0.6.3-dfm so Rust can compare prompt bytes without linking the
 * server. Tool-schema / invoke reconstruct lives in render_tools_inc.c. */

#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    DS4_THINK_NONE = 0,
    DS4_THINK_LOW = 1,
    DS4_THINK_HIGH = 2,
    DS4_THINK_MAX = 3
} ds4_think_mode;

static const char DS4_REASONING_EFFORT_HIGH_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

static const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

static const char *ds4_think_effort_prefix(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_HIGH: return DS4_REASONING_EFFORT_HIGH_PREFIX;
    case DS4_THINK_MAX:  return DS4_REASONING_EFFORT_MAX_PREFIX;
    case DS4_THINK_NONE:
    case DS4_THINK_LOW:
        break;
    }
    return "";
}

static bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode != DS4_THINK_NONE;
}

typedef struct { char *ptr; size_t len, cap; } buf;
typedef struct {
    const char *id;
    const char *name;
    const char *arguments;
} tool_call;
typedef struct {
    const char *role;
    const char *content;
    const char *reasoning;
    const char *tool_call_id;
    const tool_call *calls;
    int n_calls;
    const char *raw_dsml;
    const char *raw_tool_text;
} chat_msg;
typedef struct { chat_msg *v; int len; } chat_msgs;

#define MSG(role, content, reasoning) \
    ((chat_msg){ (role), (content), (reasoning), NULL, NULL, 0, NULL, NULL })

#define DS4_SOLAR_IM_START            "<|im:start|>"
#define DS4_SOLAR_IM_CONTENT          "<|im:content|>"
#define DS4_SOLAR_IM_END              "<|im:end|>"
#define DS4_SOLAR_THINK_START         "<|think:start|>"
#define DS4_SOLAR_THINK_END           "<|think:end|>"
#define DS4_SOLAR_TOOL_RESPONSE_START "<|tool_response:start|>"
#define DS4_SOLAR_TOOL_RESPONSE_END   "<|tool_response:end|>"

static void die(const char *m) {
    fprintf(stderr, "render_c_oracle: %s\n", m);
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

static void buf_puts(buf *b, const char *s) {
    size_t n = strlen(s);
    buf_grow(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
}

static void buf_putc(buf *b, char c) {
    buf_grow(b, 1);
    b->ptr[b->len++] = c;
    b->ptr[b->len] = 0;
}

static void buf_append(buf *b, const char *s, size_t n) {
    buf_grow(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
}

static void buf_printf(buf *b, const char *fmt, ...) {
    char tmp[512];
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

static void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; s && *s; s++) {
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

static void motif3_buf_put_trimmed(buf *out, const char *text) {
    const char *begin = text ? text : "";
    while (*begin && isspace((unsigned char)*begin)) begin++;
    const char *end = begin + strlen(begin);
    while (end > begin && isspace((unsigned char)end[-1])) end--;
    buf_append(out, begin, (size_t)(end - begin));
}

static bool exaone_buf_put_trimmed(buf *out, const char *text) {
    const char *begin = text ? text : "";
    while (*begin && isspace((unsigned char)*begin)) begin++;
    const char *end = begin + strlen(begin);
    while (end > begin && isspace((unsigned char)end[-1])) end--;
    buf_append(out, begin, (size_t)(end - begin));
    return end > begin;
}

static void buf_free(buf *b) { free(b->ptr); memset(b, 0, sizeof(*b)); }

static char *buf_take(buf *b) {
    char *p = b->ptr ? b->ptr : strdup("");
    memset(b, 0, sizeof(*b));
    return p;
}

static bool role_is_system(const char *role) {
    return !strcmp(role, "system") || !strcmp(role, "developer");
}

static bool role_is_user_like(const char *role) {
    return !strcmp(role, "user") || !strcmp(role, "tool") || !strcmp(role, "function");
}

static bool chat_history_uses_tool_context(const chat_msgs *msgs,
                                           const char *tool_schemas) {
    if (tool_schemas && tool_schemas[0]) return true;
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if ((!strcmp(m->role, "assistant") && m->n_calls > 0) ||
            !strcmp(m->role, "tool") || !strcmp(m->role, "function"))
            return true;
    }
    return false;
}

static void append_tool_result_text(buf *b, const char *s) {
    const char *end = "</tool_result>";
    const size_t endlen = strlen(end);
    for (s = s ? s : ""; *s;) {
        if (!strncmp(s, end, endlen)) {
            buf_puts(b, "&lt;");
            s++;
        } else {
            buf_putc(b, *s++);
        }
    }
}

static char *render_dsml_chat_prompt_text_choice(
        const chat_msgs *msgs, const char *tool_schemas,
        ds4_think_mode think_mode) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool tool_context = chat_history_uses_tool_context(msgs, tool_schemas);
    int last_user_idx = -1;
    buf system = {0};
    if (tool_schemas && tool_schemas[0]) die("tools not in no-tools oracle");
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role)) continue;
        if (system.len) buf_puts(&system, "\n\n");
        buf_puts(&system, m->content ? m->content : "");
    }
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_user_like(m->role)) last_user_idx = i;
    }

    buf out = {0};
    buf_puts(&out, "<｜begin▁of▁sentence｜>");
    buf_puts(&out, ds4_think_effort_prefix(think_mode));
    buf_puts(&out, system.ptr ? system.ptr : "");

    bool pending_assistant = false;
    bool pending_tool_result = false;
    for (int i = 0; i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            continue;
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<｜User｜>");
            buf_puts(&out, m->content ? m->content : "");
            pending_assistant = true;
            pending_tool_result = false;
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            if (!pending_tool_result) buf_puts(&out, "<｜User｜>");
            buf_puts(&out, "<tool_result>");
            append_tool_result_text(&out, m->content);
            buf_puts(&out, "</tool_result>");
            pending_assistant = true;
            pending_tool_result = true;
        } else if (!strcmp(m->role, "assistant")) {
            if (pending_assistant) {
                buf_puts(&out, "<｜Assistant｜>");
                if (think) {
                    if (tool_context || i > last_user_idx) {
                        buf_puts(&out, "<think>");
                        buf_puts(&out, m->reasoning ? m->reasoning : "");
                        buf_puts(&out, "</think>");
                    } else {
                        buf_puts(&out, "</think>");
                    }
                } else {
                    buf_puts(&out, "</think>");
                }
            }
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "<｜end▁of▁sentence｜>");
            pending_assistant = false;
            pending_tool_result = false;
        }
    }

    if (pending_assistant) {
        buf_puts(&out, "<｜Assistant｜>");
        buf_puts(&out, think ? "<think>" : "</think>");
    }

    buf_free(&system);
    return buf_take(&out);
}

static char *render_motif3_chat_prompt_text(
        const chat_msgs *msgs, ds4_think_mode think_mode) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool first_system = msgs && msgs->len > 0 &&
        role_is_system(msgs->v[0].role);
    int last_assistant = -1;
    for (int i = 0; msgs && i < msgs->len; i++)
        if (!strcmp(msgs->v[i].role, "assistant")) last_assistant = i;

    buf out = {0};
    buf_puts(&out, "<|beginoftext|>");
    if (first_system) {
        buf_puts(&out, "<|startofturn|><|system|>");
        buf_puts(&out, msgs->v[0].content ? msgs->v[0].content : "");
        buf_puts(&out, "<|endofturn|>");
    }
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (i == 0 && first_system) {
            continue;
        } else if (role_is_system(m->role)) {
            buf_puts(&out, "<|startofturn|><|system|>");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "<|endofturn|>");
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<|startofturn|><|user|>");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "<|endofturn|>");
        } else if (!strcmp(m->role, "assistant")) {
            buf_puts(&out, "<|startofturn|><|assistant|>");
            if (m->reasoning && m->reasoning[0] && i == last_assistant) {
                buf_puts(&out, "<think>");
                motif3_buf_put_trimmed(&out, m->reasoning);
                buf_puts(&out, "</think>");
            }
            motif3_buf_put_trimmed(&out, m->content);
            buf_puts(&out, "<|endofturn|>");
        } else if (!strcmp(m->role, "tool") || !strcmp(m->role, "function")) {
            const bool group_start = i == 0 ||
                (strcmp(msgs->v[i - 1].role, "tool") &&
                 strcmp(msgs->v[i - 1].role, "function"));
            const bool group_end = i + 1 == msgs->len ||
                (strcmp(msgs->v[i + 1].role, "tool") &&
                 strcmp(msgs->v[i + 1].role, "function"));
            if (group_start) buf_puts(&out, "<|startofturn|><|tool|>");
            buf_puts(&out, "<tool_response>{\"tool_call_id\": ");
            json_escape(&out, m->tool_call_id ? m->tool_call_id : "");
            buf_puts(&out, ", \"content\": ");
            json_escape(&out, m->content ? m->content : "");
            buf_puts(&out, "}</tool_response>");
            if (group_end) buf_puts(&out, "<|endofturn|>");
        }
    }
    buf_puts(&out, "<|startofturn|><|assistant|><think>");
    if (!think) buf_puts(&out, "</think>");
    return buf_take(&out);
}

static char *render_dots3_chat_prompt_text(
        const chat_msgs *msgs, ds4_think_mode think_mode) {
    const bool think = ds4_think_mode_enabled(think_mode);
    const bool first_system = msgs && msgs->len > 0 &&
        role_is_system(msgs->v[0].role);
    buf out = {0};
    buf_puts(&out, "<|system|>");
    if (first_system) {
        buf_puts(&out, msgs->v[0].content ? msgs->v[0].content : "");
    } else {
        buf_puts(&out, "You are a helpful assistant.");
    }
    buf_puts(&out, "<|endofsystem|>");
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        const bool is_user = m->role && !strcmp(m->role, "user");
        if (i == 0 && first_system) continue;
        if (is_user || role_is_system(m->role)) {
            const char *content = m->content ? m->content : "";
            buf_puts(&out, "<|user|>");
            buf_puts(&out, content);
            if (is_user && !think) {
                const size_t clen = strlen(content);
                if (clen < 10 || strcmp(content + clen - 10, "<no_think>") != 0)
                    buf_puts(&out, "<no_think>");
            }
            buf_puts(&out, "<|endofuser|>");
        } else if (m->role && !strcmp(m->role, "assistant")) {
            const char *content = m->content ? m->content : "";
            char *reason_heap = NULL;
            char *content_heap = NULL;
            const char *reasoning = m->reasoning ? m->reasoning : "";
            if (!reasoning[0]) {
                const char *close = strstr(content, "</think>");
                if (close) {
                    const char *rb = content;
                    const char *open = NULL;
                    for (const char *scan = content; scan < close;
                         scan = strstr(scan, "<think>")) {
                        if (!scan || scan >= close) break;
                        open = scan;
                        scan += 7;
                    }
                    if (open) rb = open + 7;
                    const char *re = close;
                    while (re > rb && re[-1] == '\n') re--;
                    while (rb < re && rb[0] == '\n') rb++;
                    reason_heap = malloc((size_t)(re - rb) + 1);
                    if (!reason_heap) die("oom");
                    memcpy(reason_heap, rb, (size_t)(re - rb));
                    reason_heap[re - rb] = 0;
                    const char *cb = close + 8;
                    while (cb[0] == '\n') cb++;
                    content_heap = strdup(cb);
                    if (!content_heap) die("oom");
                    reasoning = reason_heap;
                    content = content_heap;
                }
            }
            const char *tb = reasoning;
            while (*tb && isspace((unsigned char)*tb)) tb++;
            const char *te = tb + strlen(tb);
            while (te > tb && isspace((unsigned char)te[-1])) te--;
            buf_puts(&out, "<|assistant|>");
            if (!think) {
                buf_puts(&out, "<think>\n\n</think>\n\n");
                buf_puts(&out, content);
            } else if (te > tb) {
                buf_puts(&out, "<think>\n");
                buf_append(&out, tb, (size_t)(te - tb));
                buf_puts(&out, "\n</think>\n\n");
                buf_puts(&out, content);
            } else {
                buf_puts(&out, content);
            }
            buf_puts(&out, "<|endofassistant|>");
            free(content_heap);
            free(reason_heap);
        } else if (m->role && (!strcmp(m->role, "tool") ||
                               !strcmp(m->role, "function"))) {
            const bool group_start = i == 0 ||
                (strcmp(msgs->v[i - 1].role, "tool") &&
                 strcmp(msgs->v[i - 1].role, "function"));
            const bool group_end = i + 1 == msgs->len ||
                (strcmp(msgs->v[i + 1].role, "tool") &&
                 strcmp(msgs->v[i + 1].role, "function"));
            if (group_start) buf_puts(&out, "<|user|>");
            buf_puts(&out, "\n<dots_function_response>\n");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "\n</dots_function_response>");
            if (group_end) buf_puts(&out, "<|endofuser|>");
        }
    }
    buf_puts(&out, "<|assistant|>");
    if (!think) buf_puts(&out, "<think>\n\n</think>\n\n");
    return buf_take(&out);
}

static bool chat_msg_is_model_tool_result(const chat_msg *m) {
    if (!m || !m->role) return false;
    return !strcmp(m->role, "tool") || !strcmp(m->role, "function");
}

static char *render_exaone_chat_prompt_text(
        const chat_msgs *msgs, ds4_think_mode think_mode) {
    int last_user_idx = -1;
    for (int i = 0; msgs && i < msgs->len; i++) {
        if (!strcmp(msgs->v[i].role, "user") &&
            !chat_msg_is_model_tool_result(&msgs->v[i]))
            last_user_idx = i;
    }
    buf out = {0};
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (chat_msg_is_model_tool_result(m)) {
            int end = i;
            while (end < msgs->len && chat_msg_is_model_tool_result(&msgs->v[end]))
                end++;
            buf_puts(&out, "<|tool|>\n");
            for (int k = i; k < end; k++) {
                if (k > i) buf_putc(&out, '\n');
                buf_puts(&out, "<tool_result>");
                buf_puts(&out, msgs->v[k].content ? msgs->v[k].content : "");
                buf_puts(&out, "</tool_result>");
            }
            buf_puts(&out, "<|endofturn|>\n");
            i = end - 1;
        } else if (role_is_system(m->role)) {
            buf_puts(&out, "<|system|>\n");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "<|endofturn|>\n");
        } else if (!strcmp(m->role, "user")) {
            buf_puts(&out, "<|user|>\n");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, "<|endofturn|>\n");
        } else if (!strcmp(m->role, "assistant")) {
            buf_puts(&out, "<|assistant|>\n<think>\n");
            if (m->reasoning && m->reasoning[0] && i > last_user_idx)
                exaone_buf_put_trimmed(&out, m->reasoning);
            buf_puts(&out, "\n</think>\n\n");
            exaone_buf_put_trimmed(&out, m->content);
            buf_puts(&out, "<|endofturn|>\n");
        }
    }
    buf_puts(&out, "<|assistant|>\n<think>\n");
    if (!ds4_think_mode_enabled(think_mode))
        buf_puts(&out, "\n</think>\n\n");
    return buf_take(&out);
}

static void append_solar_role_open(buf *b, const char *role) {
    buf_puts(b, DS4_SOLAR_IM_START);
    buf_puts(b, role);
    buf_puts(b, DS4_SOLAR_IM_CONTENT);
}

static void append_solar_tool_response_text(buf *b, const char *s) {
    const char *end = DS4_SOLAR_TOOL_RESPONSE_END;
    const size_t endlen = strlen(end);
    const char *p = s ? s : "";
    const char *limit = p + strlen(p);
    while (p < limit) {
        if ((size_t)(limit - p) >= endlen && !strncmp(p, end, endlen)) {
            buf_puts(b, "&lt;");
            p++;
        } else {
            buf_putc(b, *p++);
        }
    }
}

static char *render_solar_chat_prompt_text(
        const chat_msgs *msgs, ds4_think_mode think_mode) {
    int last_user_idx = -1;
    for (int i = 0; msgs && i < msgs->len; i++) {
        if (!strcmp(msgs->v[i].role, "user") &&
            !chat_msg_is_model_tool_result(&msgs->v[i]))
            last_user_idx = i;
    }
    buf system = {0};
    bool have_system = false;
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (!role_is_system(m->role)) continue;
        if (!have_system) {
            buf_puts(&system, "## System Prompt");
            have_system = true;
        }
        buf_puts(&system, "\n\n");
        buf_puts(&system, m->content ? m->content : "");
    }
    buf out = {0};
    if (system.len) {
        append_solar_role_open(&out, "system");
        buf_append(&out, system.ptr, system.len);
        buf_puts(&out, DS4_SOLAR_IM_END "\n");
    }
    for (int i = 0; msgs && i < msgs->len; i++) {
        const chat_msg *m = &msgs->v[i];
        if (role_is_system(m->role)) {
            continue;
        } else if (!strcmp(m->role, "user") &&
                   !chat_msg_is_model_tool_result(m)) {
            append_solar_role_open(&out, "user");
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, DS4_SOLAR_IM_END "\n");
        } else if (chat_msg_is_model_tool_result(m)) {
            int end = i;
            while (end < msgs->len && chat_msg_is_model_tool_result(&msgs->v[end]))
                end++;
            append_solar_role_open(&out, "tool");
            bool first = true;
            for (int k = i; k < end; k++) {
                if (!first) buf_putc(&out, '\n');
                first = false;
                buf_puts(&out, DS4_SOLAR_TOOL_RESPONSE_START);
                append_solar_tool_response_text(&out, msgs->v[k].content);
                buf_puts(&out, DS4_SOLAR_TOOL_RESPONSE_END);
            }
            buf_puts(&out, "\n" DS4_SOLAR_IM_END "\n");
            i = end - 1;
        } else if (!strcmp(m->role, "assistant")) {
            append_solar_role_open(&out, "assistant");
            buf_puts(&out, DS4_SOLAR_THINK_START);
            if (m->reasoning && m->reasoning[0] && i > last_user_idx)
                buf_puts(&out, m->reasoning);
            buf_puts(&out, DS4_SOLAR_THINK_END);
            buf_puts(&out, m->content ? m->content : "");
            buf_puts(&out, DS4_SOLAR_IM_END "\n");
        }
    }
    append_solar_role_open(&out, "assistant");
    buf_puts(&out, DS4_SOLAR_THINK_START);
    if (!ds4_think_mode_enabled(think_mode))
        buf_puts(&out, DS4_SOLAR_THINK_END);
    buf_free(&system);
    return buf_take(&out);
}

#include "render_tools_inc.c"

static ds4_think_mode parse_think(const char *s) {
    if (!strcmp(s, "none")) return DS4_THINK_NONE;
    if (!strcmp(s, "low")) return DS4_THINK_LOW;
    if (!strcmp(s, "high")) return DS4_THINK_HIGH;
    if (!strcmp(s, "max")) return DS4_THINK_MAX;
    die("think mode");
    return DS4_THINK_NONE;
}

static void emit(const chat_msgs *msgs, ds4_think_mode mode) {
    char *p = render_dsml_chat_prompt_text_choice(msgs, "", mode);
    fputs(p, stdout);
    free(p);
}

static void emit_fam(const char *fam, const chat_msgs *msgs, ds4_think_mode mode) {
    char *p = NULL;
    if (!strcmp(fam, "motif")) p = render_motif3_chat_prompt_text(msgs, mode);
    else if (!strcmp(fam, "exaone")) p = render_exaone_chat_prompt_text(msgs, mode);
    else if (!strcmp(fam, "dots3")) p = render_dots3_chat_prompt_text(msgs, mode);
    else if (!strcmp(fam, "solar")) p = render_solar_chat_prompt_text(msgs, mode);
    else die("family");
    fputs(p, stdout);
    free(p);
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: render_c_oracle <case> ...");
    const char *cmd = argv[1];
    if (!strcmp(cmd, "user")) {
        if (argc < 4) die("user think content");
        chat_msg m = MSG("user", argv[3], "");
        chat_msgs msgs = {&m, 1};
        emit(&msgs, parse_think(argv[2]));
    } else if (!strcmp(cmd, "system-user")) {
        if (argc < 5) die("system-user think sys user");
        chat_msg m[2] = {
            MSG("system", argv[3], ""),
            MSG("user", argv[4], ""),
        };
        chat_msgs msgs = {m, 2};
        emit(&msgs, parse_think(argv[2]));
    } else if (!strcmp(cmd, "history")) {
        if (argc < 6) die("history think user asst user");
        chat_msg m[3] = {
            MSG("user", argv[3], ""),
            MSG("assistant", argv[4], ""),
            MSG("user", argv[5], ""),
        };
        chat_msgs msgs = {m, 3};
        emit(&msgs, parse_think(argv[2]));
    } else if (!strcmp(cmd, "think-hist")) {
        if (argc < 7) die("think-hist think user asst-reason asst user");
        chat_msg m[3] = {
            MSG("user", argv[3], ""),
            MSG("assistant", argv[5], argv[4]),
            MSG("user", argv[6], ""),
        };
        chat_msgs msgs = {m, 3};
        emit(&msgs, parse_think(argv[2]));
    } else if (!strcmp(cmd, "tool-result")) {
        if (argc < 5) die("tool-result think user tool");
        chat_msg m[2] = {
            MSG("user", argv[3], ""),
            MSG("tool", argv[4], ""),
        };
        chat_msgs msgs = {m, 2};
        emit(&msgs, parse_think(argv[2]));
    } else if (!strcmp(cmd, "tool-escape")) {
        chat_msg m[2] = {
            MSG("user", "q", ""),
            MSG("tool", "x</tool_result>y", ""),
        };
        chat_msgs msgs = {m, 2};
        emit(&msgs, DS4_THINK_NONE);
    } else if (!strcmp(cmd, "developer")) {
        chat_msg m[2] = {
            MSG("developer", "dev", ""),
            MSG("user", "hi", ""),
        };
        chat_msgs msgs = {m, 2};
        emit(&msgs, DS4_THINK_NONE);
    } else if (!strcmp(cmd, "fam-user")) {
        if (argc < 5) die("fam-user fam think content");
        chat_msg m = MSG("user", argv[4], "");
        chat_msgs msgs = {&m, 1};
        emit_fam(argv[2], &msgs, parse_think(argv[3]));
    } else if (!strcmp(cmd, "fam-system-user")) {
        if (argc < 6) die("fam-system-user fam think sys user");
        chat_msg m[2] = {
            MSG("system", argv[4], ""),
            MSG("user", argv[5], ""),
        };
        chat_msgs msgs = {m, 2};
        emit_fam(argv[2], &msgs, parse_think(argv[3]));
    } else if (!strcmp(cmd, "fam-history")) {
        if (argc < 7) die("fam-history fam think user asst user");
        chat_msg m[3] = {
            MSG("user", argv[4], ""),
            MSG("assistant", argv[5], ""),
            MSG("user", argv[6], ""),
        };
        chat_msgs msgs = {m, 3};
        emit_fam(argv[2], &msgs, parse_think(argv[3]));
    } else if (!strcmp(cmd, "fam-think-hist")) {
        if (argc < 8) die("fam-think-hist fam think user reason asst user");
        chat_msg m[3] = {
            MSG("user", argv[4], ""),
            MSG("assistant", argv[6], argv[5]),
            MSG("user", argv[7], ""),
        };
        chat_msgs msgs = {m, 3};
        emit_fam(argv[2], &msgs, parse_think(argv[3]));
    } else if (!strcmp(cmd, "fam-tool")) {
        if (argc < 6) die("fam-tool fam think user tool");
        chat_msg m[2] = {
            MSG("user", argv[4], ""),
            {"tool", argv[5], "", "call1", NULL, 0, NULL, NULL},
        };
        chat_msgs msgs = {m, 2};
        emit_fam(argv[2], &msgs, parse_think(argv[3]));
    } else if (!strcmp(cmd, "dsml-tools")) {
        if (argc < 5) die("dsml-tools think schemas user");
        chat_msg m = MSG("user", argv[4], "");
        chat_msgs msgs = {&m, 1};
        char *p = render_dsml_tools(&msgs, argv[3], parse_think(argv[2]), TOOL_CHOICE_AUTO);
        fputs(p, stdout);
        free(p);
    } else if (!strcmp(cmd, "dsml-tools-req")) {
        if (argc < 5) die("dsml-tools-req think schemas user");
        chat_msg m = MSG("user", argv[4], "");
        chat_msgs msgs = {&m, 1};
        char *p = render_dsml_tools(&msgs, argv[3], parse_think(argv[2]), TOOL_CHOICE_REQUIRED);
        fputs(p, stdout);
        free(p);
    } else if (!strcmp(cmd, "dsml-invoke")) {
        if (argc < 5) die("dsml-invoke think name args");
        tool_call tc = { "call1", argv[3], argv[4] };
        chat_msg m[2] = {
            MSG("user", "q", ""),
            { "assistant", "", "", NULL, &tc, 1, NULL, NULL },
        };
        chat_msgs msgs = {m, 2};
        char *p = render_dsml_tools(&msgs, "", parse_think(argv[2]), TOOL_CHOICE_AUTO);
        fputs(p, stdout);
        free(p);
    } else if (!strcmp(cmd, "fam-tools")) {
        if (argc < 6) die("fam-tools fam think schemas user");
        chat_msg m = MSG("user", argv[5], "");
        chat_msgs msgs = {&m, 1};
        const char *fam = argv[2];
        ds4_think_mode mode = parse_think(argv[3]);
        const char *schemas = argv[4];
        char *p = NULL;
        if (!strcmp(fam, "motif")) p = render_motif3_tools(&msgs, schemas, NULL, 0, mode);
        else if (!strcmp(fam, "exaone")) p = render_exaone_tools(&msgs, schemas, mode);
        else if (!strcmp(fam, "dots3")) p = render_dots3_tools(&msgs, schemas, mode);
        else if (!strcmp(fam, "solar")) p = render_solar_tools(&msgs, schemas, NULL, 0, mode);
        else die("family");
        fputs(p, stdout);
        free(p);
    } else if (!strcmp(cmd, "fam-invoke")) {
        if (argc < 6) die("fam-invoke fam think name args");
        tool_call tc = { "call1", argv[4], argv[5] };
        chat_msg m[2] = {
            MSG("user", "q", ""),
            { "assistant", "", "", NULL, &tc, 1, NULL, NULL },
        };
        chat_msgs msgs = {m, 2};
        const char *fam = argv[2];
        ds4_think_mode mode = parse_think(argv[3]);
        char *p = NULL;
        if (!strcmp(fam, "motif")) p = render_motif3_tools(&msgs, "", NULL, 0, mode);
        else if (!strcmp(fam, "exaone")) p = render_exaone_tools(&msgs, "", mode);
        else if (!strcmp(fam, "dots3")) p = render_dots3_tools(&msgs, "", mode);
        else if (!strcmp(fam, "solar")) p = render_solar_tools(&msgs, "", NULL, 0, mode);
        else die("family");
        fputs(p, stdout);
        free(p);
    } else if (!strcmp(cmd, "motif-tools-order")) {
        if (argc < 4) die("motif-tools-order schemas user");
        const char *props[] = { "city", "unit" };
        oracle_order order = { "get_weather", props, 2 };
        chat_msg m = MSG("user", argv[3], "");
        chat_msgs msgs = {&m, 1};
        char *p = render_motif3_tools(&msgs, argv[2], &order, 1, DS4_THINK_NONE);
        fputs(p, stdout);
        free(p);
    } else if (!strcmp(cmd, "dots3-embed")) {
        chat_msg m[2] = {
            MSG("user", "q", ""),
            MSG("assistant", "<think>\nplan\n</think>\n\nAnswer", ""),
        };
        chat_msgs msgs = {m, 2};
        emit_fam("dots3", &msgs, DS4_THINK_LOW);
    } else {
        die("unknown case");
    }
    return 0;
}
